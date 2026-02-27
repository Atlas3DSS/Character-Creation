#!/usr/bin/env python3
"""
Cross-Model Neuron Map Comparison: Qwen3-VL-8B vs GPT-OSS-20B

Compares personality neuron landscapes between a dense model (Qwen) and
an MoE model (GPT-OSS) to understand architectural differences in how
personality steering signals manifest.

Data sources:
- Qwen: contrastive_data/persona_probe/ (57K-pair analysis, layers 9-26, dim=4096)
- GPT-OSS: skippy_gptoss/deep_probe/ (1K-prompt probe, layers 0-23, dim=2880)

Comparison dimensions:
1. Layer importance curves (where does personality live?)
2. SVD spectra (how many dimensions encode personality?)
3. Neuron concentration (sparse vs distributed?)
4. Expert routing shift (GPT-OSS only — MoE-specific)
5. Functional signatures (what statistical patterns differentiate the models?)

Usage:
    python compare_neuron_maps.py \
        --qwen-probe contrastive_data/persona_probe \
        --qwen-svd contrastive_data/svd_results \
        --gptoss-probe skippy_gptoss/deep_probe \
        --output skippy_gptoss/cross_model_comparison
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch


def load_qwen_data(probe_dir: str, svd_dir: str) -> dict:
    """Load Qwen probe data from the 57K-pair contrastive analysis."""
    data = {}

    # Load persona dims summary
    dims_path = os.path.join(probe_dir, "qwen_persona_dims.json")
    if os.path.exists(dims_path):
        with open(dims_path) as f:
            data["persona_dims"] = json.load(f)
        print(f"  Qwen persona dims: {data['persona_dims']['n_pairs']} pairs, "
              f"layers {data['persona_dims']['layers_analyzed']}")
    else:
        print(f"  Warning: {dims_path} not found")
        return data

    # Load persona vectors (z-scores per layer)
    vectors_path = os.path.join(probe_dir, "persona_vectors.pt")
    if os.path.exists(vectors_path):
        data["persona_vectors"] = torch.load(vectors_path, map_location="cpu", weights_only=True)
        print(f"  Qwen persona vectors: {len(data['persona_vectors'])} layers")

    # Load SVD subspaces
    data["svd"] = {}
    svd_files = sorted(Path(svd_dir).glob("layer_*_subspace.pt"))
    for f in svd_files:
        layer_idx = int(f.stem.split("_")[1])
        svd_data = torch.load(f, map_location="cpu", weights_only=True)
        data["svd"][layer_idx] = svd_data
    print(f"  Qwen SVD subspaces: {len(data['svd'])} layers")

    # Load pattern analysis
    pattern_path = os.path.join(probe_dir, "pattern_analysis.json")
    if os.path.exists(pattern_path):
        with open(pattern_path) as f:
            data["pattern_analysis"] = json.load(f)

    return data


def load_gptoss_data(probe_dir: str) -> dict:
    """Load GPT-OSS probe data."""
    data = {}

    # Load probe summary
    summary_path = os.path.join(probe_dir, "probe_summary.json")
    if os.path.exists(summary_path):
        with open(summary_path) as f:
            data["summary"] = json.load(f)
        print(f"  GPT-OSS probe: {data['summary'].get('n_prompts', '?')} prompts, "
              f"{data['summary'].get('n_layers', '?')} layers")
    else:
        print(f"  Warning: {summary_path} not found — probe may still be running")
        return data

    # Load neuron z-scores
    zscores_path = os.path.join(probe_dir, "neuron_zscores.pt")
    if os.path.exists(zscores_path):
        data["neuron_zscores"] = torch.load(zscores_path, map_location="cpu", weights_only=True)
        print(f"  GPT-OSS z-scores: {len(data['neuron_zscores'])} layers")

    # Load activation deltas per layer
    data["deltas"] = {}
    delta_files = sorted(Path(probe_dir).glob("deltas_layer_*.pt"))
    for f in delta_files:
        layer_idx = int(f.stem.split("_")[-1])
        data["deltas"][layer_idx] = torch.load(f, map_location="cpu", weights_only=True)
    print(f"  GPT-OSS deltas: {len(data['deltas'])} layers")

    # Load sample responses
    samples_path = os.path.join(probe_dir, "sample_responses.json")
    if os.path.exists(samples_path):
        with open(samples_path) as f:
            data["samples"] = json.load(f)

    return data


def compare_layer_importance(qwen: dict, gptoss: dict) -> dict:
    """Compare where personality signal concentrates across layers."""
    print(f"\n{'='*70}")
    print("1. LAYER IMPORTANCE COMPARISON")
    print(f"{'='*70}")

    results = {"qwen": {}, "gptoss": {}}

    # Qwen: compute layer importance from persona_dims
    qwen_layers = []
    if "persona_dims" in qwen:
        per_layer = qwen["persona_dims"]["per_layer"]
        for layer_str, layer_data in per_layer.items():
            layer_idx = int(layer_str)
            # Compute mean |z| from top neurons
            z_scores = [abs(n["z_score"]) for n in layer_data.get("qwen_top20", [])]
            mean_z = np.mean(z_scores) if z_scores else 0
            max_z = max(z_scores) if z_scores else 0
            qwen_layers.append((layer_idx, mean_z, max_z))
        qwen_layers.sort(key=lambda x: x[0])

        # Also try loading full z-score vectors for more accurate stats
        if "persona_vectors" in qwen:
            vectors = qwen["persona_vectors"]
            for layer_idx in sorted(vectors.keys()):
                z = vectors[layer_idx]
                if isinstance(z, torch.Tensor):
                    mean_abs_z = float(z.abs().mean())
                    max_abs_z = float(z.abs().max())
                    results["qwen"][layer_idx] = {
                        "mean_abs_z": mean_abs_z,
                        "max_abs_z": max_abs_z,
                        "n_significant_gt2": int((z.abs() > 2).sum()),
                        "n_significant_gt3": int((z.abs() > 3).sum()),
                    }

    # GPT-OSS: from probe summary
    if "summary" in gptoss:
        layer_imp = gptoss["summary"].get("layer_importance", {})
        sig_counts = gptoss["summary"].get("significant_neuron_counts", {})
        for layer_str, imp in layer_imp.items():
            layer_idx = int(layer_str)
            sig = sig_counts.get(layer_str, {})
            results["gptoss"][layer_idx] = {
                "mean_abs_z": imp,
                "n_significant_gt2": sig.get("push_gt2", 0) + sig.get("pull_lt_neg2", 0),
                "n_significant_gt3": sig.get("push_gt3", 0) + sig.get("pull_lt_neg3", 0),
            }
            # Get max z from top neurons
            top_neurons = gptoss["summary"].get("top_neurons_per_layer", {}).get(layer_str, {})
            push_top = top_neurons.get("push", [])
            pull_top = top_neurons.get("pull", [])
            max_z = 0
            if push_top:
                max_z = max(max_z, abs(push_top[0][1]))
            if pull_top:
                max_z = max(max_z, abs(pull_top[0][1]))
            results["gptoss"][layer_idx]["max_abs_z"] = max_z

    # Print comparison
    print(f"\n  {'Model':<12} {'Layers':<12} {'Hidden':<8} {'Architecture':<12}")
    print(f"  {'─'*12} {'─'*12} {'─'*8} {'─'*12}")
    print(f"  {'Qwen-VL':<12} {'9-26 (18)':<12} {'4096':<8} {'Dense':<12}")
    print(f"  {'GPT-OSS':<12} {'0-23 (24)':<12} {'2880':<8} {'MoE (32×4)':<12}")

    # Normalize layer positions to [0, 1] for comparison
    print(f"\n  Layer importance by relative position (normalized):")
    print(f"  {'Rel Pos':<8} {'Qwen L':<8} {'Qwen |z|':<10} {'GPT-OSS L':<10} {'GPT-OSS |z|':<12}")
    print(f"  {'─'*8} {'─'*8} {'─'*10} {'─'*10} {'─'*12}")

    qwen_sorted = sorted(results.get("qwen", {}).items())
    gptoss_sorted = sorted(results.get("gptoss", {}).items())

    # Create normalized position mapping
    if qwen_sorted:
        qwen_total = 36  # Qwen has 36 layers
        for idx, data in qwen_sorted:
            rel_pos = idx / qwen_total
            # Find closest GPT-OSS layer
            gptoss_match = None
            gptoss_total = 24
            for g_idx, g_data in gptoss_sorted:
                g_rel = g_idx / gptoss_total
                if gptoss_match is None or abs(g_rel - rel_pos) < abs(gptoss_match[0] / gptoss_total - rel_pos):
                    gptoss_match = (g_idx, g_data)

            g_str = f"L{gptoss_match[0]}" if gptoss_match else "—"
            g_z = f"{gptoss_match[1]['mean_abs_z']:.3f}" if gptoss_match else "—"
            print(f"  {rel_pos:.2f}     L{idx:<5d} {data['mean_abs_z']:.3f}     {g_str:<9s} {g_z:<12s}")

    # Summary statistics
    print(f"\n  Summary:")
    if results["qwen"]:
        q_importances = [d["mean_abs_z"] for d in results["qwen"].values()]
        q_max_z = max(d.get("max_abs_z", 0) for d in results["qwen"].values())
        q_total_sig = sum(d["n_significant_gt2"] for d in results["qwen"].values())
        print(f"    Qwen:    avg |z|={np.mean(q_importances):.3f}, "
              f"max |z|={q_max_z:.2f}, "
              f"total sig neurons (|z|>2)={q_total_sig}")
    if results["gptoss"]:
        g_importances = [d["mean_abs_z"] for d in results["gptoss"].values()]
        g_max_z = max(d.get("max_abs_z", 0) for d in results["gptoss"].values())
        g_total_sig = sum(d["n_significant_gt2"] for d in results["gptoss"].values())
        print(f"    GPT-OSS: avg |z|={np.mean(g_importances):.3f}, "
              f"max |z|={g_max_z:.2f}, "
              f"total sig neurons (|z|>2)={g_total_sig}")

    return results


def compare_svd_spectra(qwen: dict, gptoss: dict) -> dict:
    """Compare personality subspace dimensionality."""
    print(f"\n{'='*70}")
    print("2. SVD SPECTRA — PERSONALITY SUBSPACE DIMENSIONALITY")
    print(f"{'='*70}")

    results = {"qwen": {}, "gptoss": {}}

    # Qwen SVD from subspace files
    if "svd" in qwen:
        for layer_idx, svd_data in sorted(qwen["svd"].items()):
            if isinstance(svd_data, dict):
                S = svd_data.get("singular_values", svd_data.get("S", None))
                if S is not None:
                    if isinstance(S, torch.Tensor):
                        S = S.numpy()
                    total_var = float(np.sum(S ** 2))
                    if total_var > 0:
                        cum_var = np.cumsum(S ** 2) / total_var
                        k50 = int(np.searchsorted(cum_var, 0.50)) + 1
                        k80 = int(np.searchsorted(cum_var, 0.80)) + 1
                        k95 = int(np.searchsorted(cum_var, 0.95)) + 1
                        results["qwen"][layer_idx] = {
                            "k50": k50, "k80": k80, "k95": k95,
                            "top_sv_ratio": float(S[0] / S.sum()),
                        }
            elif isinstance(svd_data, torch.Tensor):
                # It might be the Vh matrix or the full subspace
                # Try to infer
                pass

    # GPT-OSS SVD from probe summary
    if "summary" in gptoss:
        svd_analysis = gptoss["summary"].get("svd_analysis", {})
        for layer_str, svd_data in svd_analysis.items():
            layer_idx = int(layer_str)
            results["gptoss"][layer_idx] = {
                "k50": svd_data["k50"],
                "k80": svd_data["k80"],
                "k95": svd_data["k95"],
                "top_sv_ratio": svd_data["top_sv_ratio"],
            }

    # Print comparison
    print(f"\n  Dimensions needed for X% variance explained:")
    print(f"  {'Model':<10} {'Avg K50':<10} {'Avg K80':<10} {'Avg K95':<10} {'Avg SV1%':<10}")
    print(f"  {'─'*10} {'─'*10} {'─'*10} {'─'*10} {'─'*10}")

    for name, data in [("Qwen", results["qwen"]), ("GPT-OSS", results["gptoss"])]:
        if data:
            avg_k50 = np.mean([d["k50"] for d in data.values()])
            avg_k80 = np.mean([d["k80"] for d in data.values()])
            avg_k95 = np.mean([d["k95"] for d in data.values()])
            avg_sv1 = np.mean([d["top_sv_ratio"] for d in data.values()])
            print(f"  {name:<10} {avg_k50:<10.1f} {avg_k80:<10.1f} {avg_k95:<10.1f} {avg_sv1:<10.4f}")

    # Per-layer detail for GPT-OSS (we know Qwen's K95 is 267-367)
    if results["gptoss"]:
        print(f"\n  GPT-OSS per-layer SVD:")
        for idx in sorted(results["gptoss"]):
            d = results["gptoss"][idx]
            print(f"    L{idx:2d}: K50={d['k50']:3d}, K80={d['k80']:3d}, K95={d['k95']:3d}, SV1={d['top_sv_ratio']:.4f}")

    # Key insight
    print(f"\n  Known Qwen K95 range: 267-367 per layer (from 57K pairs)")
    if results["gptoss"]:
        g_k95s = [d["k95"] for d in results["gptoss"].values()]
        print(f"  GPT-OSS K95 range: {min(g_k95s)}-{max(g_k95s)} per layer (from 1K prompts)")
        if np.mean(g_k95s) < 200:
            print(f"  → GPT-OSS personality signal is MORE concentrated (fewer dims)")
            print(f"    This could mean MoE routing pre-separates personality from capability")
        else:
            print(f"  → Similar dimensionality — personality is equally distributed")

    return results


def compare_neuron_concentration(qwen: dict, gptoss: dict) -> dict:
    """Analyze how concentrated vs distributed personality neurons are."""
    print(f"\n{'='*70}")
    print("3. NEURON CONCENTRATION — SPARSE vs DISTRIBUTED")
    print(f"{'='*70}")

    results = {}

    # Qwen: from persona_dims top neurons
    if "persona_dims" in qwen:
        per_layer = qwen["persona_dims"]["per_layer"]
        qwen_stats = []
        for layer_str, layer_data in per_layer.items():
            top20 = layer_data.get("qwen_top20", [])
            if top20:
                z_scores = [abs(n["z_score"]) for n in top20]
                qwen_stats.append({
                    "layer": int(layer_str),
                    "top1_z": z_scores[0],
                    "top5_z": np.mean(z_scores[:5]),
                    "top20_z": np.mean(z_scores),
                    "ratio_top1_top20": z_scores[0] / np.mean(z_scores) if np.mean(z_scores) > 0 else 0,
                })
        results["qwen"] = qwen_stats

    # GPT-OSS: from probe summary top neurons
    if "summary" in gptoss:
        top_neurons = gptoss["summary"].get("top_neurons_per_layer", {})
        gptoss_stats = []
        for layer_str, neurons in top_neurons.items():
            push = neurons.get("push", [])
            pull = neurons.get("pull", [])
            all_z = [abs(z) for _, z in push[:20]] + [abs(z) for _, z in pull[:20]]
            all_z.sort(reverse=True)
            if all_z:
                gptoss_stats.append({
                    "layer": int(layer_str),
                    "top1_z": all_z[0],
                    "top5_z": np.mean(all_z[:5]) if len(all_z) >= 5 else np.mean(all_z),
                    "top20_z": np.mean(all_z[:20]) if len(all_z) >= 20 else np.mean(all_z),
                    "ratio_top1_top20": all_z[0] / np.mean(all_z[:20]) if len(all_z) >= 20 and np.mean(all_z[:20]) > 0 else 0,
                })
        results["gptoss"] = gptoss_stats

    # Print comparison
    print(f"\n  Neuron concentration (higher ratio = more concentrated on few neurons):")
    print(f"  {'Model':<10} {'Avg Top1 |z|':<14} {'Avg Top5 |z|':<14} {'Top1/Top20 ratio':<18}")
    print(f"  {'─'*10} {'─'*14} {'─'*14} {'─'*18}")

    for name, stats in [("Qwen", results.get("qwen", [])), ("GPT-OSS", results.get("gptoss", []))]:
        if stats:
            avg_top1 = np.mean([s["top1_z"] for s in stats])
            avg_top5 = np.mean([s["top5_z"] for s in stats])
            avg_ratio = np.mean([s["ratio_top1_top20"] for s in stats])
            print(f"  {name:<10} {avg_top1:<14.3f} {avg_top5:<14.3f} {avg_ratio:<18.3f}")

    # Identify "identity neuron" equivalents in GPT-OSS
    print(f"\n  Known Qwen identity neurons:")
    print(f"    Dim 994: z=-13.96 at L9, present across ALL 18 layers (avg |z|=8.76)")
    print(f"    Dim 270: z=-8.41 at L9, present across ALL 18 layers (avg |z|=7.68)")

    if "summary" in gptoss:
        print(f"\n  GPT-OSS top neurons per layer:")
        top_neurons = gptoss["summary"].get("top_neurons_per_layer", {})

        # Find neurons that appear across multiple layers
        neuron_freq: dict[int, list] = {}
        for layer_str, neurons in top_neurons.items():
            layer_idx = int(layer_str)
            for dim_idx, z_val in neurons.get("push", [])[:10]:
                if dim_idx not in neuron_freq:
                    neuron_freq[dim_idx] = []
                neuron_freq[dim_idx].append((layer_idx, z_val, "push"))
            for dim_idx, z_val in neurons.get("pull", [])[:10]:
                if dim_idx not in neuron_freq:
                    neuron_freq[dim_idx] = []
                neuron_freq[dim_idx].append((layer_idx, -z_val, "pull"))

        # Sort by frequency (neurons appearing in most layers = cross-layer identity neurons)
        cross_layer = [(dim, entries) for dim, entries in neuron_freq.items() if len(entries) >= 3]
        cross_layer.sort(key=lambda x: len(x[1]), reverse=True)

        if cross_layer:
            print(f"\n  Cross-layer personality neurons (appear in ≥3 layers):")
            for dim, entries in cross_layer[:10]:
                layers_str = ", ".join(f"L{l}" for l, _, _ in sorted(entries))
                avg_z = np.mean([abs(z) for _, z, _ in entries])
                direction = entries[0][2]
                print(f"    Dim {dim:4d}: {len(entries)} layers ({layers_str}), "
                      f"avg |z|={avg_z:.2f}, direction={direction}")
        else:
            print(f"    No cross-layer neurons found (≥3 layers) — personality may be layer-local in MoE")

    return results


def analyze_expert_routing(gptoss: dict) -> dict:
    """Analyze MoE expert routing differences between personality/base modes."""
    print(f"\n{'='*70}")
    print("4. MoE EXPERT ROUTING ANALYSIS (GPT-OSS only)")
    print(f"{'='*70}")

    if "summary" not in gptoss:
        print("  No GPT-OSS data available")
        return {}

    routing = gptoss["summary"].get("routing_analysis", {})
    if not routing:
        print("  No routing data captured — router hooks may not have fired")
        print("  This could mean GPT-OSS routes at a different module path")
        return {}

    results = {}
    print(f"\n  Expert routing KL-divergence (personality vs base):")
    print(f"  {'Layer':<8} {'KL Div':<10} {'Top Personality Expert':<25} {'Top Assistant Expert':<25}")
    print(f"  {'─'*8} {'─'*10} {'─'*25} {'─'*25}")

    total_kl = 0
    for layer_str in sorted(routing.keys(), key=lambda x: int(x)):
        r = routing[layer_str]
        kl = r["kl_div"]
        total_kl += kl

        p_expert = r["top_personality_experts"][0] if r["top_personality_experts"] else (0, 0)
        a_expert = r["top_assistant_experts"][0] if r["top_assistant_experts"] else (0, 0)

        print(f"  L{int(layer_str):2d}     {kl:<10.4f} "
              f"Expert #{p_expert[0]} (+{p_expert[1]:.4f})     "
              f"Expert #{a_expert[0]} (+{a_expert[1]:.4f})")

        results[int(layer_str)] = {
            "kl_div": kl,
            "personality_experts": r["top_personality_experts"][:3],
            "assistant_experts": r["top_assistant_experts"][:3],
        }

    avg_kl = total_kl / len(routing) if routing else 0
    print(f"\n  Average KL divergence: {avg_kl:.4f}")

    if avg_kl < 0.01:
        print(f"  → VERY LOW routing shift — personality barely affects expert selection")
        print(f"    Implication: Personality signal is WITHIN expert computations, not routing")
        print(f"    Strategy: Target expert MLPs directly (need custom LoRA for 3D tensors)")
    elif avg_kl < 0.1:
        print(f"  → MODERATE routing shift — some experts specialize for personality")
        print(f"    Strategy: Target both routing-preferred experts AND attention layers")
    else:
        print(f"  → STRONG routing shift — experts clearly separate personality from capability")
        print(f"    Strategy: Can selectively fine-tune personality-preferred experts")

    return results


def compute_architectural_insights(qwen: dict, gptoss: dict,
                                    layer_results: dict,
                                    svd_results: dict,
                                    neuron_results: dict,
                                    routing_results: dict) -> dict:
    """Synthesize all comparisons into actionable insights."""
    print(f"\n{'='*70}")
    print("5. CROSS-ARCHITECTURE INSIGHTS & TRAINING STRATEGY")
    print(f"{'='*70}")

    insights = []

    # Insight 1: Layer importance comparison
    if layer_results.get("qwen") and layer_results.get("gptoss"):
        q_vals = [d["mean_abs_z"] for d in layer_results["qwen"].values()]
        g_vals = [d["mean_abs_z"] for d in layer_results["gptoss"].values()]

        q_mean = np.mean(q_vals)
        g_mean = np.mean(g_vals)
        ratio = g_mean / q_mean if q_mean > 0 else 0

        if ratio < 0.5:
            insight = (f"GPT-OSS personality signal is {ratio:.1f}x WEAKER than Qwen. "
                      f"MoE routing may distribute personality across experts, diluting per-neuron signal.")
            insights.append(("SIGNAL STRENGTH", insight))
        elif ratio > 1.5:
            insight = (f"GPT-OSS personality signal is {ratio:.1f}x STRONGER than Qwen. "
                      f"MoE may concentrate personality more effectively.")
            insights.append(("SIGNAL STRENGTH", insight))
        else:
            insight = (f"Similar signal strength (ratio={ratio:.2f}). "
                      f"Personality manifests comparably despite architectural differences.")
            insights.append(("SIGNAL STRENGTH", insight))

    # Insight 2: Where personality lives
    if layer_results.get("gptoss"):
        g_sorted = sorted(layer_results["gptoss"].items(), key=lambda x: x[1]["mean_abs_z"], reverse=True)
        top_layers = [idx for idx, _ in g_sorted[:5]]
        total_layers = 24

        if all(l > total_layers * 0.6 for l in top_layers):
            insights.append(("LAYER LOCATION",
                           f"Personality concentrates in LATE layers ({top_layers}) — "
                           f"same pattern as Qwen (L18-26). Architecture doesn't change this."))
        elif all(l < total_layers * 0.4 for l in top_layers):
            insights.append(("LAYER LOCATION",
                           f"Personality concentrates in EARLY layers ({top_layers}) — "
                           f"DIFFERENT from Qwen. MoE may process personality earlier."))
        else:
            insights.append(("LAYER LOCATION",
                           f"Personality spread across layers ({top_layers}). "
                           f"Less concentrated than Qwen's late-layer pattern."))

    # Insight 3: MoE routing implications
    if routing_results:
        kl_vals = [r["kl_div"] for r in routing_results.values()]
        avg_kl = np.mean(kl_vals)
        if avg_kl < 0.01:
            insights.append(("MoE ROUTING",
                           "Expert routing is personality-AGNOSTIC. Personality signal lives "
                           "WITHIN shared expert computations, NOT in routing decisions. "
                           "Standard attention-only LoRA should work — MLP expert targeting is LESS critical."))
        elif avg_kl < 0.1:
            insights.append(("MoE ROUTING",
                           "Moderate routing shift detected. Some experts process personality "
                           "preferentially. Consider expert-aware LoRA targeting."))
        else:
            insights.append(("MoE ROUTING",
                           "Strong routing specialization! Personality has dedicated expert pathways. "
                           "Expert-specific LoRA could be highly effective."))

    # Insight 4: Cross-layer neurons
    if neuron_results.get("gptoss"):
        gptoss_stats = neuron_results["gptoss"]
        if gptoss_stats:
            avg_top1 = np.mean([s["top1_z"] for s in gptoss_stats])
            avg_ratio = np.mean([s["ratio_top1_top20"] for s in gptoss_stats])

            if avg_top1 < 5:
                insights.append(("NEURON SPARSITY",
                               f"GPT-OSS has NO dominant identity neurons (max z≈{avg_top1:.1f}). "
                               f"Unlike Qwen's dim 994 (z=14.0), personality is distributed. "
                               f"Push/pull regularization needs to target many neurons at low strength."))
            elif avg_top1 > 10:
                insights.append(("NEURON SPARSITY",
                               f"GPT-OSS HAS strong identity neurons (max z≈{avg_top1:.1f}). "
                               f"Similar to Qwen — push/pull on targeted neurons should be effective."))

    # Print insights
    for title, text in insights:
        print(f"\n  [{title}]")
        print(f"  {text}")

    # Training strategy recommendation
    print(f"\n  {'─'*60}")
    print(f"  RECOMMENDED TRAINING STRATEGY:")
    print(f"  {'─'*60}")

    if routing_results and np.mean([r["kl_div"] for r in routing_results.values()]) < 0.01:
        print(f"""
  Since routing is personality-agnostic:
  1. Attention-only LoRA is valid (no need for expert MLP targeting)
  2. Push/pull regularization works on hidden states (post-MoE)
  3. BUT: Need lower reg weight than Qwen (diluted signal)
  4. Consider higher LoRA rank (r=64 vs r=32) to compensate
  5. Profile neuron shifts DURING training to verify signal moves""")
    elif routing_results:
        print(f"""
  Since routing shows personality specialization:
  1. Target personality-preferred experts with custom LoRA
  2. Consider router fine-tuning to strengthen personality routing
  3. Push/pull regularization on routing logits, not just hidden states""")
    else:
        print(f"""
  No routing data — cannot determine MoE behavior.
  Recommendations depend on probe results.""")

    return {"insights": insights}


def main():
    parser = argparse.ArgumentParser(description="Cross-Model Neuron Map Comparison")
    parser.add_argument("--qwen-probe", type=str,
                        default="contrastive_data/persona_probe")
    parser.add_argument("--qwen-svd", type=str,
                        default="contrastive_data/svd_results")
    parser.add_argument("--gptoss-probe", type=str,
                        default="skippy_gptoss/deep_probe")
    parser.add_argument("--output", type=str,
                        default="skippy_gptoss/cross_model_comparison")
    args = parser.parse_args()

    print(f"{'='*70}")
    print(f"CROSS-MODEL NEURON MAP: Qwen3-VL-8B (Dense) vs GPT-OSS-20B (MoE)")
    print(f"{'='*70}")

    # Load data
    print(f"\nLoading Qwen data...")
    qwen = load_qwen_data(args.qwen_probe, args.qwen_svd)

    print(f"\nLoading GPT-OSS data...")
    gptoss = load_gptoss_data(args.gptoss_probe)

    if not gptoss.get("summary"):
        print(f"\n  ERROR: GPT-OSS probe data not found at {args.gptoss_probe}")
        print(f"  Run probe_gptoss_neurons.py first, then re-run this script.")
        sys.exit(1)

    # Run comparisons
    layer_results = compare_layer_importance(qwen, gptoss)
    svd_results = compare_svd_spectra(qwen, gptoss)
    neuron_results = compare_neuron_concentration(qwen, gptoss)
    routing_results = analyze_expert_routing(gptoss)
    strategy = compute_architectural_insights(
        qwen, gptoss, layer_results, svd_results, neuron_results, routing_results
    )

    # Save results
    os.makedirs(args.output, exist_ok=True)
    report = {
        "models": {
            "qwen": {
                "name": "Qwen3-VL-8B-Instruct",
                "architecture": "dense",
                "layers": 36,
                "hidden_dim": 4096,
                "probe_pairs": qwen.get("persona_dims", {}).get("n_pairs", 0),
            },
            "gptoss": {
                "name": "GPT-OSS-20B",
                "architecture": "MoE (32 experts, Top-4)",
                "layers": 24,
                "hidden_dim": 2880,
                "probe_prompts": gptoss.get("summary", {}).get("n_prompts", 0),
            },
        },
        "layer_importance": layer_results,
        "svd_spectra": svd_results,
        "neuron_concentration": neuron_results,
        "routing_analysis": routing_results,
        "insights": strategy.get("insights", []),
    }

    with open(os.path.join(args.output, "comparison_report.json"), "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\n  Saved full report to {args.output}/comparison_report.json")

    print(f"\n{'='*70}")
    print(f"COMPARISON COMPLETE")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
