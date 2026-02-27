#!/usr/bin/env python3
"""
Offline analysis of GPT-OSS-20B probe results.

Loads saved raw activations from skippy_gptoss/deep_probe/raw/ and runs
the full analysis pipeline without needing the model loaded.

Usage:
    python analyze_probe_results.py --raw-dir skippy_gptoss/deep_probe/raw --output skippy_gptoss/deep_probe
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def load_raw_activations(raw_dir: str) -> tuple[dict, dict, dict, dict]:
    """Load saved raw activations."""
    personality_acts: dict[int, torch.Tensor] = {}
    base_acts: dict[int, torch.Tensor] = {}
    personality_router: dict[int, torch.Tensor] = {}
    base_router: dict[int, torch.Tensor] = {}

    for f in sorted(Path(raw_dir).glob("personality_layer_*.pt")):
        idx = int(f.stem.split("_")[-1])
        personality_acts[idx] = torch.load(f, map_location="cpu", weights_only=True)
        print(f"  Loaded personality layer {idx}: {personality_acts[idx].shape}")

    for f in sorted(Path(raw_dir).glob("base_layer_*.pt")):
        idx = int(f.stem.split("_")[-1])
        base_acts[idx] = torch.load(f, map_location="cpu", weights_only=True)

    for f in sorted(Path(raw_dir).glob("personality_router_*.pt")):
        idx = int(f.stem.split("_")[-1])
        personality_router[idx] = torch.load(f, map_location="cpu", weights_only=True)

    for f in sorted(Path(raw_dir).glob("base_router_*.pt")):
        idx = int(f.stem.split("_")[-1])
        base_router[idx] = torch.load(f, map_location="cpu", weights_only=True)

    print(f"  Loaded {len(personality_acts)} personality layers, {len(base_acts)} base layers")
    print(f"  Router data: {len(personality_router)} personality, {len(base_router)} base")
    return personality_acts, base_acts, personality_router, base_router


def analyze(personality_acts: dict, base_acts: dict,
            personality_router: dict, base_router: dict,
            output_dir: str) -> dict:
    """Run full analysis on loaded activations."""

    os.makedirs(output_dir, exist_ok=True)
    actual_layers = sorted(personality_acts.keys())
    n_prompts = personality_acts[actual_layers[0]].shape[0]
    hidden_dim = personality_acts[actual_layers[0]].shape[1]

    print(f"\n{'='*60}")
    print(f"ANALYSIS: {n_prompts} prompts, {len(actual_layers)} layers, hidden_dim={hidden_dim}")
    print(f"{'='*60}")

    # 1. Per-neuron z-scores
    print("\n1. Per-neuron z-scores (personality vs base mode)")
    neuron_scores: dict[int, torch.Tensor] = {}
    layer_importance: dict[int, float] = {}

    for idx in actual_layers:
        p_acts = personality_acts[idx]
        b_acts = base_acts[idx]
        delta = p_acts - b_acts
        delta_mean = delta.mean(dim=0)
        delta_std = delta.std(dim=0) + 1e-8
        z_scores = delta_mean / delta_std
        neuron_scores[idx] = z_scores
        layer_importance[idx] = float(z_scores.abs().mean())

        n_sig = int((z_scores.abs() > 2).sum())
        n_pos = int((z_scores > 2).sum())
        n_neg = int((z_scores < -2).sum())
        top_z = float(z_scores.abs().max())
        top_dim = int(z_scores.abs().argmax())

        print(f"  L{idx:2d}: |z|_mean={z_scores.abs().mean():.3f}, "
              f"significant={n_sig:4d} (push={n_pos:3d}, pull={n_neg:3d}), "
              f"max|z|={top_z:.2f} (dim {top_dim})")

    # 2. Layer importance ranking
    print("\n2. Layer importance ranking (mean |z-score|)")
    sorted_layers = sorted(layer_importance.items(), key=lambda x: x[1], reverse=True)
    for rank, (idx, imp) in enumerate(sorted_layers):
        bar = "█" * int(imp * 20)
        print(f"  #{rank+1:2d} L{idx:2d}: {imp:.4f} {bar}")

    # 3. Expert routing analysis
    print("\n3. Expert routing analysis (personality vs base)")
    routing_analysis: dict[int, dict] = {}

    for idx in actual_layers:
        p_router = personality_router.get(idx)
        b_router = base_router.get(idx)

        if p_router is None or b_router is None:
            continue

        # Both should be (n_prompts, n_experts)
        p_probs = F.softmax(p_router, dim=-1)
        b_probs = F.softmax(b_router, dim=-1)

        p_mean = p_probs.mean(dim=0)
        b_mean = b_probs.mean(dim=0)
        route_delta = p_mean - b_mean
        n_experts = route_delta.shape[0]

        kl_div = float(F.kl_div(b_mean.log().clamp(min=-10), p_mean, reduction='sum'))

        top_up = torch.topk(route_delta, k=min(5, n_experts))
        top_down = torch.topk(-route_delta, k=min(5, n_experts))

        routing_analysis[idx] = {
            "kl_div": kl_div,
            "top_personality_experts": [(int(i), float(v)) for i, v in zip(top_up.indices, top_up.values)],
            "top_assistant_experts": [(int(i), float(v)) for i, v in zip(top_down.indices, top_down.values)],
            "personality_mean_probs": p_mean.tolist(),
            "base_mean_probs": b_mean.tolist(),
        }

        print(f"  L{idx:2d}: KL_div={kl_div:.6f}")
        for exp_idx, exp_val in zip(top_up.indices[:3], top_up.values[:3]):
            print(f"    Personality expert #{exp_idx}: +{exp_val:.6f}")

    # 4. SVD analysis
    print("\n4. SVD analysis — personality subspace dimensionality per layer")
    svd_analysis: dict[int, dict] = {}

    for idx in actual_layers:
        delta = personality_acts[idx] - base_acts[idx]
        delta_centered = delta - delta.mean(dim=0, keepdim=True)

        try:
            U, S, Vh = torch.linalg.svd(delta_centered, full_matrices=False)
            S = S.numpy()
            total_var = float(np.sum(S ** 2))
            cum_var = np.cumsum(S ** 2) / total_var
            k50 = int(np.searchsorted(cum_var, 0.50)) + 1
            k80 = int(np.searchsorted(cum_var, 0.80)) + 1
            k95 = int(np.searchsorted(cum_var, 0.95)) + 1
            sv_ratio = float(S[0] / S.sum()) if S.sum() > 0 else 0

            svd_analysis[idx] = {
                "k50": k50, "k80": k80, "k95": k95,
                "top_sv_ratio": sv_ratio, "total_variance": total_var,
                "top_10_svs": S[:10].tolist(),
            }
            print(f"  L{idx:2d}: K(50%)={k50:3d}, K(80%)={k80:3d}, K(95%)={k95:3d}, "
                  f"top_sv_ratio={sv_ratio:.4f}")
        except Exception as e:
            print(f"  L{idx:2d}: SVD failed: {e}")

    # 5. Cross-layer neuron consistency
    print("\n5. Cross-layer neuron analysis — top personality neurons")
    top_neurons_per_layer: dict[int, dict] = {}
    neuron_freq: dict[int, list] = {}

    for idx in actual_layers:
        z = neuron_scores[idx]
        top_push = torch.topk(z, k=20)
        top_pull = torch.topk(-z, k=20)
        top_neurons_per_layer[idx] = {
            "push": [(int(i), float(v)) for i, v in zip(top_push.indices, top_push.values)],
            "pull": [(int(i), float(v)) for i, v in zip(top_pull.indices, top_pull.values)],
        }

        # Track cross-layer frequency
        for dim_idx, z_val in zip(top_push.indices[:10], top_push.values[:10]):
            dim_idx = int(dim_idx)
            if dim_idx not in neuron_freq:
                neuron_freq[dim_idx] = []
            neuron_freq[dim_idx].append((idx, float(z_val), "push"))
        for dim_idx, z_val in zip(top_pull.indices[:10], top_pull.values[:10]):
            dim_idx = int(dim_idx)
            if dim_idx not in neuron_freq:
                neuron_freq[dim_idx] = []
            neuron_freq[dim_idx].append((idx, float(-z_val), "pull"))

        top_push_str = ", ".join(f"d{i}({v:+.2f})" for i, v in zip(top_push.indices[:5], top_push.values[:5]))
        top_pull_str = ", ".join(f"d{i}({v:+.2f})" for i, v in zip(top_pull.indices[:5], top_pull.values[:5]))
        print(f"  L{idx:2d} push: {top_push_str}")
        print(f"  L{idx:2d} pull: {top_pull_str}")

    # Cross-layer identity neurons
    cross_layer = [(dim, entries) for dim, entries in neuron_freq.items() if len(entries) >= 3]
    cross_layer.sort(key=lambda x: len(x[1]), reverse=True)

    print(f"\n6. Cross-layer personality neurons (appear in ≥3 layers)")
    if cross_layer:
        for dim, entries in cross_layer[:15]:
            layers_str = ", ".join(f"L{l}" for l, _, _ in sorted(entries))
            avg_z = np.mean([abs(z) for _, z, _ in entries])
            direction = entries[0][2]
            print(f"  Dim {dim:4d}: {len(entries):2d} layers ({layers_str}), "
                  f"avg |z|={avg_z:.2f}, direction={direction}")
    else:
        print("  None found — personality signal is layer-local in MoE")

    # ── Save results ──────────────────────────────────────────────────

    print(f"\n{'='*60}")
    print(f"Saving results to {output_dir}")
    print(f"{'='*60}")

    # Neuron z-scores
    torch.save(
        {idx: neuron_scores[idx] for idx in actual_layers},
        os.path.join(output_dir, "neuron_zscores.pt")
    )

    # Activation deltas
    for idx in actual_layers:
        delta = personality_acts[idx] - base_acts[idx]
        torch.save(delta, os.path.join(output_dir, f"deltas_layer_{idx:02d}.pt"))

    # Summary JSON
    summary = {
        "model": "GPT-OSS-20B",
        "architecture": "MoE (32 experts, Top-4)",
        "n_prompts": n_prompts,
        "n_layers": len(actual_layers),
        "total_layers": 24,
        "hidden_dim": hidden_dim,
        "layer_importance": {str(k): v for k, v in layer_importance.items()},
        "layer_importance_ranked": [(idx, imp) for idx, imp in sorted_layers],
        "routing_analysis": {str(k): v for k, v in routing_analysis.items()},
        "svd_analysis": {str(k): v for k, v in svd_analysis.items()},
        "top_neurons_per_layer": {str(k): v for k, v in top_neurons_per_layer.items()},
        "significant_neuron_counts": {},
        "cross_layer_neurons": [
            {"dim": dim, "n_layers": len(entries),
             "layers": [(l, z, d) for l, z, d in sorted(entries)],
             "avg_abs_z": float(np.mean([abs(z) for _, z, _ in entries]))}
            for dim, entries in cross_layer[:20]
        ],
    }

    for idx in actual_layers:
        z = neuron_scores[idx]
        summary["significant_neuron_counts"][str(idx)] = {
            "push_gt2": int((z > 2).sum()),
            "pull_lt_neg2": int((z < -2).sum()),
            "push_gt3": int((z > 3).sum()),
            "pull_lt_neg3": int((z < -3).sum()),
        }

    with open(os.path.join(output_dir, "probe_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n  Saved:")
    print(f"    neuron_zscores.pt — per-neuron z-scores across modes")
    print(f"    deltas_layer_XX.pt — activation deltas per layer")
    print(f"    probe_summary.json — full analysis results")

    return summary


def main():
    parser = argparse.ArgumentParser(description="Offline analysis of GPT-OSS probe results")
    parser.add_argument("--raw-dir", type=str, default="skippy_gptoss/deep_probe/raw")
    parser.add_argument("--output", type=str, default="skippy_gptoss/deep_probe")
    args = parser.parse_args()

    print(f"{'='*60}")
    print(f"GPT-OSS-20B Probe Analysis (offline)")
    print(f"{'='*60}")

    print(f"\nLoading raw activations from {args.raw_dir}...")
    p_acts, b_acts, p_router, b_router = load_raw_activations(args.raw_dir)

    if not p_acts:
        print("ERROR: No activation data found!")
        return

    summary = analyze(p_acts, b_acts, p_router, b_router, args.output)

    print(f"\n{'='*60}")
    print(f"ANALYSIS COMPLETE")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
