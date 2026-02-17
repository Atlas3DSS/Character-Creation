#!/usr/bin/env python3
"""
Phase 4: Reasoning-Safe Transport Map.

Computes per-layer transport directions from contrastive activation deltas,
projected to be orthogonal to the reasoning subspace. Also trains optional
lightweight flow models for conditional transport (per-sample, not just mean).

The key insight: the flow model failed because we only have deltas (skippy - base),
not paired source/target activations. The correct approach:
1. Compute mean delta per layer (the average personality shift)
2. Project out the reasoning component (safety constraint)
3. Optionally: train a PCA-based conditional transport for variance-aware shifts

Usage:
    python train_flow_transport.py                           # Compute transport map
    python train_flow_transport.py --layers 22 23 24 25 26   # High-impact only
    python train_flow_transport.py --with-pca                # Include PCA transport

Output:
    ./contrastive_data/flow_models/
"""
import argparse
import json
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from tqdm import tqdm

# ─── Config ──────────────────────────────────────────────────────────────

ACTIVATIONS_DIR = Path("./contrastive_data/activations")
PROBES_DIR = Path("./contrastive_data/personality_probes")
OUTPUT_DIR = Path("./contrastive_data/flow_models")
EXTRACT_LAYERS = list(range(9, 27))  # layers 9-26

HIDDEN_DIM = 4096


# ─── Data Loading ────────────────────────────────────────────────────────

def load_deltas(layer_idx: int) -> torch.Tensor | None:
    """Load activation deltas (skippy - base) for a layer."""
    delta_file = ACTIVATIONS_DIR / f"deltas_layer_{layer_idx:02d}.pt"
    if not delta_file.exists():
        return None
    return torch.load(delta_file, weights_only=True)  # (N, 4096)


def load_reasoning_subspace(layer_idx: int) -> torch.Tensor | None:
    """Load reasoning subspace for geometry-aware constraint."""
    subspace_file = PROBES_DIR / f"reasoning_subspace_layer{layer_idx:02d}.pt"
    if subspace_file.exists():
        return torch.load(subspace_file, weights_only=True)  # (K, 4096)
    return None


def load_personality_directions(layer_idx: int) -> torch.Tensor | None:
    """Load orthogonalized personality probe directions."""
    dirs_file = PROBES_DIR / "orthogonalized_personality_dirs.pt"
    if dirs_file.exists():
        all_dirs = torch.load(dirs_file, weights_only=True)
        # all_dirs structure: {trait_name: {layer_idx: direction_tensor}}
        layer_dirs = []
        for trait_name, layer_dict in all_dirs.items():
            if layer_idx in layer_dict:
                layer_dirs.append(layer_dict[layer_idx])
        if layer_dirs:
            return torch.stack(layer_dirs)  # (N_traits, 4096)
    return None


# ─── Transport Map Computation ────────────────────────────────────────────

def compute_transport_for_layer(
    layer_idx: int,
    deltas: torch.Tensor,
    reasoning_subspace: torch.Tensor | None = None,
    personality_dirs: torch.Tensor | None = None,
) -> dict:
    """Compute the reasoning-safe transport direction for a layer.

    Three levels of transport:
    1. Raw mean delta (baseline)
    2. Reasoning-projected mean delta (remove reasoning component)
    3. Personality-projected delta (project onto known personality directions)

    Returns dict with all three + analysis metrics.
    """
    N, D = deltas.shape

    # 1. Raw mean delta
    mean_delta = deltas.mean(dim=0)  # (4096,)
    raw_norm = mean_delta.norm().item()

    # 2. Remove reasoning component
    if reasoning_subspace is not None:
        # Project mean delta onto reasoning subspace
        reasoning_proj = reasoning_subspace @ mean_delta  # (K,)
        reasoning_component = reasoning_subspace.T @ reasoning_proj  # (4096,)
        safe_delta = mean_delta - reasoning_component
        reasoning_removed = reasoning_component.norm().item()
    else:
        safe_delta = mean_delta.clone()
        reasoning_removed = 0.0

    safe_norm = safe_delta.norm().item()

    # 3. Project onto personality directions (if available)
    personality_delta = None
    personality_alignment = 0.0
    if personality_dirs is not None and personality_dirs.shape[0] > 0:
        # Project safe delta onto personality directions
        projections = personality_dirs @ safe_delta  # (N_traits,)
        personality_delta = (personality_dirs.T @ projections)  # (4096,)
        personality_alignment = F.cosine_similarity(
            safe_delta.unsqueeze(0), personality_delta.unsqueeze(0)
        ).item()

    # Compute delta distribution statistics
    delta_norms = deltas.norm(dim=1)  # (N,)
    cos_sims = F.cosine_similarity(
        deltas, mean_delta.unsqueeze(0).expand_as(deltas), dim=1
    )

    # Variance analysis: how much do individual deltas vary from the mean?
    centered = deltas - mean_delta.unsqueeze(0)
    variance = centered.pow(2).mean().item()

    metrics = {
        "raw_norm": round(raw_norm, 4),
        "safe_norm": round(safe_norm, 4),
        "reasoning_removed_norm": round(reasoning_removed, 4),
        "reasoning_fraction": round(reasoning_removed / max(raw_norm, 1e-8), 4),
        "personality_alignment": round(personality_alignment, 4),
        "mean_delta_norm": round(delta_norms.mean().item(), 4),
        "std_delta_norm": round(delta_norms.std().item(), 4),
        "mean_cos_sim_to_mean": round(cos_sims.mean().item(), 4),
        "delta_variance": round(variance, 4),
        "n_pairs": N,
    }

    result = {
        "raw_mean_delta": mean_delta,
        "safe_mean_delta": safe_delta,
        "metrics": metrics,
    }
    if personality_delta is not None:
        result["personality_delta"] = personality_delta

    return result


def compute_pca_transport(
    deltas: torch.Tensor,
    reasoning_subspace: torch.Tensor | None = None,
    n_components: int = 32,
) -> dict:
    """Compute PCA-based conditional transport.

    Instead of just the mean delta, find the top principal components
    of the delta distribution. This allows for more nuanced transport
    that captures the variance in personality shifts.
    """
    N, D = deltas.shape

    # Center deltas
    mean_delta = deltas.mean(dim=0)
    centered = deltas - mean_delta.unsqueeze(0)

    # SVD on centered deltas
    # For efficiency, compute on a subsample if N is large
    if N > 10000:
        idx = torch.randperm(N)[:10000]
        sample = centered[idx]
    else:
        sample = centered

    U, S, Vt = torch.linalg.svd(sample, full_matrices=False)
    components = Vt[:n_components]  # (K, 4096)
    explained_var = (S[:n_components] ** 2).sum() / (S ** 2).sum()

    # Project out reasoning components
    if reasoning_subspace is not None:
        safe_components = []
        for comp in components:
            reasoning_proj = reasoning_subspace @ comp
            reasoning_component = reasoning_subspace.T @ reasoning_proj
            safe_comp = comp - reasoning_component
            safe_comp = safe_comp / (safe_comp.norm() + 1e-8)
            safe_components.append(safe_comp)
        components = torch.stack(safe_components)

    return {
        "components": components,
        "singular_values": S[:n_components],
        "explained_variance": explained_var.item(),
        "mean_delta": mean_delta,
    }


# ─── Main ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Phase 4: Reasoning-Safe Transport Map")
    parser.add_argument("--layers", type=int, nargs="+", default=None,
                        help="Specific layers (default: all 9-26)")
    parser.add_argument("--with-pca", action="store_true",
                        help="Include PCA-based conditional transport")
    parser.add_argument("--pca-components", type=int, default=32,
                        help="Number of PCA components for conditional transport")
    args = parser.parse_args()

    layers = args.layers or EXTRACT_LAYERS

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 60)
    print("PHASE 4: REASONING-SAFE TRANSPORT MAP")
    print("=" * 60)
    print(f"  Layers: {layers}")
    print(f"  PCA transport: {args.with_pca}")

    transport_map = {}
    all_metrics = {}

    for layer_idx in tqdm(layers, desc="Computing transport"):
        print(f"\n{'─' * 40}")
        print(f"Layer {layer_idx}")
        print(f"{'─' * 40}")

        deltas = load_deltas(layer_idx)
        if deltas is None:
            print(f"  No activation data, skipping")
            continue

        print(f"  {deltas.shape[0]} pairs, shape {deltas.shape}")

        reasoning_sub = load_reasoning_subspace(layer_idx)
        personality_dirs = load_personality_directions(layer_idx)

        if reasoning_sub is not None:
            print(f"  Reasoning subspace: {reasoning_sub.shape[0]} components")
        if personality_dirs is not None:
            print(f"  Personality directions: {personality_dirs.shape[0]} traits")

        # Compute transport
        result = compute_transport_for_layer(
            layer_idx, deltas, reasoning_sub, personality_dirs
        )
        metrics = result["metrics"]

        print(f"  Raw mean norm:      {metrics['raw_norm']:.4f}")
        print(f"  Safe mean norm:     {metrics['safe_norm']:.4f}")
        print(f"  Reasoning removed:  {metrics['reasoning_removed_norm']:.4f} "
              f"({metrics['reasoning_fraction']*100:.1f}%)")
        print(f"  Personality align:  {metrics['personality_alignment']:.4f}")
        print(f"  Cos sim to mean:    {metrics['mean_cos_sim_to_mean']:.4f}")

        transport_map[layer_idx] = result["safe_mean_delta"]
        all_metrics[layer_idx] = metrics

        # Optional: PCA transport
        if args.with_pca:
            pca_result = compute_pca_transport(
                deltas, reasoning_sub, args.pca_components
            )
            torch.save(pca_result, OUTPUT_DIR / f"pca_transport_layer_{layer_idx:02d}.pt")
            print(f"  PCA: {args.pca_components} components, "
                  f"{pca_result['explained_variance']*100:.1f}% variance")

    # Save transport map
    torch.save(transport_map, OUTPUT_DIR / "flow_transport_map.pt")
    print(f"\n  Saved transport map: {OUTPUT_DIR}/flow_transport_map.pt")

    # Save per-layer raw and safe deltas for ablation
    raw_map = {}
    for layer_idx in layers:
        deltas = load_deltas(layer_idx)
        if deltas is not None:
            raw_map[layer_idx] = deltas.mean(dim=0)
    torch.save(raw_map, OUTPUT_DIR / "raw_mean_deltas.pt")

    # Save training log
    summary = {
        "layers": list(transport_map.keys()),
        "metrics": {str(k): v for k, v in all_metrics.items()},
        "with_pca": args.with_pca,
    }
    with open(OUTPUT_DIR / "flow_training_log.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Summary table
    print(f"\n{'=' * 60}")
    print("PHASE 4 COMPLETE — Transport Map Summary")
    print(f"{'=' * 60}")
    print(f"  {'Layer':>5s}  {'Raw':>8s}  {'Safe':>8s}  {'Removed%':>8s}  {'Align':>6s}  {'CosSim':>6s}")
    for layer_idx in sorted(all_metrics.keys()):
        m = all_metrics[layer_idx]
        print(f"  L{layer_idx:>3d}  {m['raw_norm']:>8.4f}  {m['safe_norm']:>8.4f}  "
              f"{m['reasoning_fraction']*100:>7.1f}%  {m['personality_alignment']:>6.3f}  "
              f"{m['mean_cos_sim_to_mean']:>6.3f}")
    print(f"\n  Output: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
