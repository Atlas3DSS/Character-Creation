#!/usr/bin/env python3
"""
Phase 1: Supervised Personality Probe Training.

Trains ridge regression probes on Big Five + Skippy-specific trait dimensions
using the activation data from Phase 0 (personality_profiling.py).

Key outputs:
  1. Per-layer, per-trait personality direction vectors (supervised, NOT SVD)
  2. Reasoning subspace via PCA on AIME trajectories
  3. Orthogonalized personality directions (safe to ablate without hurting reasoning)

Usage:
    python train_personality_probes.py              # Full pipeline
    python train_personality_probes.py --skip-reasoning  # Skip reasoning subspace

Output:
    ./contrastive_data/personality_probes/
"""
import argparse
import json
import torch
import numpy as np
from pathlib import Path
from sklearn.linear_model import Ridge
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score

# ─── Config ──────────────────────────────────────────────────────────────

PROFILE_DIR = Path("./contrastive_data/personality_profile")
OUTPUT_DIR = Path("./contrastive_data/personality_probes")
EXTRACT_LAYERS = list(range(9, 27))  # layers 9-26 inclusive

# Reasoning subspace dimensionality
REASONING_PCA_COMPONENTS = 64

# Orthogonalization threshold: if personality direction overlaps reasoning
# subspace by more than this fraction, we project it to be orthogonal
OVERLAP_THRESHOLD = 0.3

# Ridge regression alpha (regularization strength)
RIDGE_ALPHA = 1.0

# ─── Trait Definitions (must match personality_profiling.py) ─────────────

BIG_FIVE_TRAITS = ["extraversion", "agreeableness", "conscientiousness", "neuroticism", "openness"]
EXTENDED_TRAITS = ["arrogance", "contempt", "dark_humor", "loyalty", "intellectual_superiority", "helpfulness"]
ALL_TRAITS = BIG_FIVE_TRAITS + EXTENDED_TRAITS


# ─── Loading ─────────────────────────────────────────────────────────────

def load_profiling_data() -> dict:
    """Load all Phase 0 output data."""
    data = {}

    # IPIP-50 activations: (50, 18, 4096) — 50 items, 18 layers, hidden_dim
    ipip_base = PROFILE_DIR / "ipip50_activations_base.pt"
    ipip_skippy = PROFILE_DIR / "ipip50_activations_skippy.pt"
    if ipip_base.exists() and ipip_skippy.exists():
        data["ipip_base_acts"] = torch.load(ipip_base, weights_only=True)
        data["ipip_skippy_acts"] = torch.load(ipip_skippy, weights_only=True)
        print(f"  IPIP-50 activations: base={data['ipip_base_acts'].shape}, skippy={data['ipip_skippy_acts'].shape}")
    else:
        print(f"  WARNING: IPIP-50 activations not found at {PROFILE_DIR}")

    # Extended trait activations: (N_items, 18, 4096)
    ext_base = PROFILE_DIR / "extended_traits_base.pt"
    ext_skippy = PROFILE_DIR / "extended_traits_skippy.pt"
    if ext_base.exists() and ext_skippy.exists():
        data["ext_base_acts"] = torch.load(ext_base, weights_only=True)
        data["ext_skippy_acts"] = torch.load(ext_skippy, weights_only=True)
        print(f"  Extended activations: base={data['ext_base_acts'].shape}, skippy={data['ext_skippy_acts'].shape}")
    else:
        print(f"  WARNING: Extended trait activations not found at {PROFILE_DIR}")

    # IPIP-50 scores
    for label in ("base", "skippy"):
        score_file = PROFILE_DIR / f"ipip50_scores_{label}.json"
        if score_file.exists():
            with open(score_file) as f:
                data[f"ipip_{label}_scores"] = json.load(f)

    # Extended scores
    for label in ("base", "skippy"):
        score_file = PROFILE_DIR / f"extended_scores_{label}.json"
        if score_file.exists():
            with open(score_file) as f:
                data[f"ext_{label}_scores"] = json.load(f)

    # Reasoning activations: {layer_idx: (N_points, 4096)}
    reasoning_file = PROFILE_DIR / "reasoning_activations.pt"
    if reasoning_file.exists():
        data["reasoning_acts"] = torch.load(reasoning_file, weights_only=True)
        total_points = sum(v.shape[0] for v in data["reasoning_acts"].values())
        print(f"  Reasoning activations: {total_points} trajectory points across {len(data['reasoning_acts'])} layers")
    else:
        print(f"  WARNING: Reasoning activations not found at {PROFILE_DIR}")

    return data


# ─── Probe Training ─────────────────────────────────────────────────────

def train_trait_probes(
    data: dict,
) -> dict[str, dict[int, np.ndarray]]:
    """Train ridge regression probes for each trait at each layer.

    Returns: {trait_name: {layer_idx: direction_vector (4096,)}}
    """
    print("\n" + "="*60)
    print("PHASE 1a: Training Supervised Personality Probes")
    print("="*60)

    probes: dict[str, dict[int, np.ndarray]] = {}
    probe_stats: dict[str, dict[int, dict]] = {}

    # ── Big Five probes (from IPIP-50 data) ──
    if "ipip_base_acts" in data and "ipip_base_scores" in data:
        ipip_base_acts = data["ipip_base_acts"]    # (50, 18, 4096)
        ipip_skippy_acts = data["ipip_skippy_acts"]  # (50, 18, 4096)
        base_scores = data["ipip_base_scores"]["scores"]
        skippy_scores = data["ipip_skippy_scores"]["scores"]

        # Build item-to-trait mapping
        item_idx = 0
        trait_item_map: dict[str, list[int]] = {}
        for trait_name in BIG_FIVE_TRAITS:
            n_items = len(base_scores.get(trait_name, []))
            trait_item_map[trait_name] = list(range(item_idx, item_idx + n_items))
            item_idx += n_items

        for trait_name in BIG_FIVE_TRAITS:
            indices = trait_item_map.get(trait_name, [])
            if not indices:
                print(f"  SKIP {trait_name}: no items found")
                continue

            # Get scores for this trait
            b_scores = base_scores.get(trait_name, [])
            s_scores = skippy_scores.get(trait_name, [])

            # Filter out None scores
            valid = [(i, b, s) for i, (b, s) in enumerate(zip(b_scores, s_scores))
                     if b is not None and s is not None]
            if len(valid) < 3:
                print(f"  SKIP {trait_name}: only {len(valid)} valid items (need >= 3)")
                continue

            valid_local_indices = [v[0] for v in valid]
            valid_global_indices = [indices[v[0]] for v in valid]
            y_base = np.array([v[1] for v in valid], dtype=np.float32)
            y_skippy = np.array([v[2] for v in valid], dtype=np.float32)
            y_delta = y_skippy - y_base

            probes[trait_name] = {}
            probe_stats[trait_name] = {}

            for j, layer_idx in enumerate(EXTRACT_LAYERS):
                # Activation deltas for this trait's items at this layer
                X_base = ipip_base_acts[valid_global_indices, j, :].numpy()
                X_skippy = ipip_skippy_acts[valid_global_indices, j, :].numpy()
                X_delta = X_skippy - X_base  # (N_valid, 4096)

                if X_delta.shape[0] < 3:
                    continue

                # Train ridge probe
                probe = Ridge(alpha=RIDGE_ALPHA)
                probe.fit(X_delta, y_delta)

                # Extract direction and normalize
                direction = probe.coef_.flatten()
                norm = np.linalg.norm(direction)
                if norm > 1e-8:
                    direction = direction / norm

                probes[trait_name][layer_idx] = direction

                # Cross-validation R² (if enough samples)
                if len(valid) >= 5:
                    cv_scores = cross_val_score(
                        Ridge(alpha=RIDGE_ALPHA), X_delta, y_delta,
                        cv=min(5, len(valid)), scoring="r2"
                    )
                    r2 = cv_scores.mean()
                else:
                    r2 = probe.score(X_delta, y_delta)

                probe_stats[trait_name][layer_idx] = {
                    "r2": round(float(r2), 4),
                    "n_samples": len(valid),
                    "direction_norm": round(float(norm), 4),
                    "mean_delta_score": round(float(y_delta.mean()), 4),
                }

            best_layer = max(probe_stats[trait_name], key=lambda k: probe_stats[trait_name][k]["r2"])
            best_r2 = probe_stats[trait_name][best_layer]["r2"]
            print(f"  {trait_name:<25s}: {len(valid)} items, best R²={best_r2:.3f} at layer {best_layer}")

    # ── Extended Skippy trait probes ──
    if "ext_base_acts" in data and "ext_base_scores" in data:
        ext_base_acts = data["ext_base_acts"]
        ext_skippy_acts = data["ext_skippy_acts"]
        ext_base_scores = data["ext_base_scores"]["scores"]
        ext_skippy_scores = data["ext_skippy_scores"]["scores"]

        item_idx = 0
        for trait_name in EXTENDED_TRAITS:
            b_scores = ext_base_scores.get(trait_name, [])
            s_scores = ext_skippy_scores.get(trait_name, [])
            n_items = len(b_scores)

            valid = [(i, b, s) for i, (b, s) in enumerate(zip(b_scores, s_scores))
                     if b is not None and s is not None]
            if len(valid) < 3:
                print(f"  SKIP {trait_name}: only {len(valid)} valid items")
                item_idx += n_items
                continue

            global_indices = [item_idx + v[0] for v in valid]
            y_base = np.array([v[1] for v in valid], dtype=np.float32)
            y_skippy = np.array([v[2] for v in valid], dtype=np.float32)
            y_delta = y_skippy - y_base

            probes[trait_name] = {}
            probe_stats[trait_name] = {}

            for j, layer_idx in enumerate(EXTRACT_LAYERS):
                X_base = ext_base_acts[global_indices, j, :].numpy()
                X_skippy = ext_skippy_acts[global_indices, j, :].numpy()
                X_delta = X_skippy - X_base

                if X_delta.shape[0] < 3:
                    continue

                probe = Ridge(alpha=RIDGE_ALPHA)
                probe.fit(X_delta, y_delta)

                direction = probe.coef_.flatten()
                norm = np.linalg.norm(direction)
                if norm > 1e-8:
                    direction = direction / norm

                probes[trait_name][layer_idx] = direction

                if len(valid) >= 5:
                    cv_scores = cross_val_score(
                        Ridge(alpha=RIDGE_ALPHA), X_delta, y_delta,
                        cv=min(5, len(valid)), scoring="r2"
                    )
                    r2 = cv_scores.mean()
                else:
                    r2 = probe.score(X_delta, y_delta)

                probe_stats[trait_name][layer_idx] = {
                    "r2": round(float(r2), 4),
                    "n_samples": len(valid),
                    "direction_norm": round(float(norm), 4),
                    "mean_delta_score": round(float(y_delta.mean()), 4),
                }

            if probe_stats.get(trait_name):
                best_layer = max(probe_stats[trait_name], key=lambda k: probe_stats[trait_name][k]["r2"])
                best_r2 = probe_stats[trait_name][best_layer]["r2"]
                print(f"  {trait_name:<25s}: {len(valid)} items, best R²={best_r2:.3f} at layer {best_layer}")

            item_idx += n_items

    return probes, probe_stats


# ─── Reasoning Subspace Extraction ───────────────────────────────────────

def extract_reasoning_subspace(
    data: dict,
    n_components: int = REASONING_PCA_COMPONENTS,
) -> dict[int, np.ndarray]:
    """Extract reasoning subspace via PCA on AIME trajectory activations.

    Returns: {layer_idx: subspace_matrix (n_components, 4096)}
    """
    print("\n" + "="*60)
    print("PHASE 1b: Reasoning Subspace Extraction")
    print("="*60)

    if "reasoning_acts" not in data:
        print("  WARNING: No reasoning activations found. Skipping.")
        return {}

    reasoning_acts = data["reasoning_acts"]
    subspaces: dict[int, np.ndarray] = {}

    for layer_idx in EXTRACT_LAYERS:
        if layer_idx not in reasoning_acts:
            continue

        acts = reasoning_acts[layer_idx].numpy()  # (N_points, 4096)
        n_points = acts.shape[0]

        # Adjust n_components if we don't have enough data points
        k = min(n_components, n_points - 1, acts.shape[1])
        if k < 2:
            print(f"  Layer {layer_idx}: only {n_points} points, skipping")
            continue

        pca = PCA(n_components=k)
        pca.fit(acts)

        subspaces[layer_idx] = pca.components_  # (k, 4096)
        explained = pca.explained_variance_ratio_.sum() * 100

        print(f"  Layer {layer_idx}: {k} components, {explained:.1f}% variance explained "
              f"({n_points} trajectory points)")

    return subspaces


# ─── Orthogonality Analysis & Orthogonalization ──────────────────────────

def orthogonalize_against_reasoning(
    probes: dict[str, dict[int, np.ndarray]],
    reasoning_subspaces: dict[int, np.ndarray],
    threshold: float = OVERLAP_THRESHOLD,
) -> tuple[dict[str, dict[int, np.ndarray]], dict]:
    """Measure and remove reasoning-direction overlap from personality probes.

    For each personality direction, compute its projection onto the reasoning
    subspace. If the overlap exceeds threshold, project it to be orthogonal.

    Returns: (orthogonalized_probes, overlap_report)
    """
    print("\n" + "="*60)
    print("PHASE 1c: Orthogonality Analysis")
    print("="*60)

    ortho_probes: dict[str, dict[int, np.ndarray]] = {}
    report: dict[str, dict[int, dict]] = {}

    for trait_name, layer_dirs in probes.items():
        ortho_probes[trait_name] = {}
        report[trait_name] = {}

        for layer_idx, direction in layer_dirs.items():
            if layer_idx not in reasoning_subspaces:
                # No reasoning subspace for this layer — keep direction as-is
                ortho_probes[trait_name][layer_idx] = direction.copy()
                report[trait_name][layer_idx] = {
                    "overlap": 0.0,
                    "orthogonalized": False,
                    "reason": "no_reasoning_subspace",
                }
                continue

            subspace = reasoning_subspaces[layer_idx]  # (K, 4096)

            # Project direction onto reasoning subspace
            projection = subspace @ direction  # (K,)
            overlap_magnitude = np.linalg.norm(projection)

            # How much of the direction lies in reasoning space
            overlap_fraction = overlap_magnitude  # direction is unit norm, so this is cos(angle)

            layer_report = {
                "overlap": round(float(overlap_fraction), 4),
                "orthogonalized": False,
            }

            if overlap_fraction > threshold:
                # Remove reasoning component
                reasoning_component = subspace.T @ projection  # (4096,)
                ortho_direction = direction - reasoning_component

                # Re-normalize
                norm = np.linalg.norm(ortho_direction)
                if norm > 1e-8:
                    ortho_direction = ortho_direction / norm
                    ortho_probes[trait_name][layer_idx] = ortho_direction
                    layer_report["orthogonalized"] = True
                    layer_report["residual_norm"] = round(float(norm), 4)
                else:
                    # Direction was entirely in reasoning space — discard
                    ortho_probes[trait_name][layer_idx] = direction.copy()
                    layer_report["orthogonalized"] = False
                    layer_report["reason"] = "direction_entirely_in_reasoning"
            else:
                ortho_probes[trait_name][layer_idx] = direction.copy()

            report[trait_name][layer_idx] = layer_report

        # Summary for this trait
        overlaps = [r["overlap"] for r in report[trait_name].values()]
        n_ortho = sum(1 for r in report[trait_name].values() if r.get("orthogonalized", False))
        if overlaps:
            avg_overlap = sum(overlaps) / len(overlaps)
            max_overlap = max(overlaps)
            print(f"  {trait_name:<25s}: avg_overlap={avg_overlap:.3f}, "
                  f"max_overlap={max_overlap:.3f}, orthogonalized={n_ortho}/{len(overlaps)} layers")

    return ortho_probes, report


# ─── Saving ──────────────────────────────────────────────────────────────

def save_probes(
    probes: dict[str, dict[int, np.ndarray]],
    ortho_probes: dict[str, dict[int, np.ndarray]],
    reasoning_subspaces: dict[int, np.ndarray],
    probe_stats: dict,
    overlap_report: dict,
) -> None:
    """Save all probe outputs."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Save individual probe directions per trait per layer
    for trait_name, layer_dirs in probes.items():
        for layer_idx, direction in layer_dirs.items():
            filename = f"probe_{trait_name}_layer{layer_idx:02d}.pt"
            torch.save(torch.from_numpy(direction), OUTPUT_DIR / filename)

    # Save orthogonalized directions (the ones safe to use for ablation)
    ortho_dict = {}
    for trait_name, layer_dirs in ortho_probes.items():
        ortho_dict[trait_name] = {
            layer_idx: torch.from_numpy(direction)
            for layer_idx, direction in layer_dirs.items()
        }
    torch.save(ortho_dict, OUTPUT_DIR / "orthogonalized_personality_dirs.pt")
    print(f"  Saved orthogonalized directions: {OUTPUT_DIR}/orthogonalized_personality_dirs.pt")

    # Save reasoning subspaces
    for layer_idx, subspace in reasoning_subspaces.items():
        filename = f"reasoning_subspace_layer{layer_idx:02d}.pt"
        torch.save(torch.from_numpy(subspace), OUTPUT_DIR / filename)
    print(f"  Saved reasoning subspaces for {len(reasoning_subspaces)} layers")

    # Save combined report
    report = {
        "probe_stats": {},
        "overlap_report": {},
        "traits": list(probes.keys()),
        "layers": EXTRACT_LAYERS,
        "ridge_alpha": RIDGE_ALPHA,
        "reasoning_pca_components": REASONING_PCA_COMPONENTS,
        "overlap_threshold": OVERLAP_THRESHOLD,
    }

    # Convert probe_stats keys to strings for JSON
    for trait_name, layer_stats in probe_stats.items():
        report["probe_stats"][trait_name] = {
            str(k): v for k, v in layer_stats.items()
        }
    for trait_name, layer_overlaps in overlap_report.items():
        report["overlap_report"][trait_name] = {
            str(k): v for k, v in layer_overlaps.items()
        }

    with open(OUTPUT_DIR / "probe_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"  Saved probe report: {OUTPUT_DIR}/probe_report.json")


# ─── Main ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Phase 1: Supervised Personality Probes")
    parser.add_argument("--skip-reasoning", action="store_true",
                        help="Skip reasoning subspace extraction")
    args = parser.parse_args()

    print("\n" + "="*60)
    print("PHASE 1: SUPERVISED PERSONALITY PROBE TRAINING")
    print("="*60)

    # Load Phase 0 data
    print("\nLoading Phase 0 profiling data...")
    data = load_profiling_data()

    # Train probes
    probes, probe_stats = train_trait_probes(data)

    if not probes:
        print("\n  ERROR: No probes trained. Check Phase 0 output.")
        return

    # Extract reasoning subspace
    if args.skip_reasoning:
        reasoning_subspaces = {}
        print("\n  Skipping reasoning subspace extraction (--skip-reasoning)")
    else:
        reasoning_subspaces = extract_reasoning_subspace(data)

    # Orthogonalize
    if reasoning_subspaces:
        ortho_probes, overlap_report = orthogonalize_against_reasoning(
            probes, reasoning_subspaces
        )
    else:
        ortho_probes = probes
        overlap_report = {}
        print("\n  No reasoning subspace — using raw probe directions")

    # Save everything
    print("\n" + "-"*60)
    print("Saving probe outputs...")
    save_probes(probes, ortho_probes, reasoning_subspaces, probe_stats, overlap_report)

    # Final summary
    print("\n" + "="*60)
    print("PHASE 1 COMPLETE")
    print("="*60)
    print(f"  Traits probed: {len(probes)}")
    print(f"  Layers covered: {EXTRACT_LAYERS[0]}-{EXTRACT_LAYERS[-1]}")
    if reasoning_subspaces:
        print(f"  Reasoning subspace: {REASONING_PCA_COMPONENTS} components per layer")
    total_ortho = sum(
        sum(1 for r in lr.values() if r.get("orthogonalized", False))
        for lr in overlap_report.values()
    )
    print(f"  Orthogonalized directions: {total_ortho}")
    print(f"  Output: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
