#!/usr/bin/env python3
"""
SAE–Connectome Alignment Analysis

Computes cosine similarity between the 20 connectome category z-score vectors
and all 81,920 SAE decoder columns at L50 and L44.

Key questions:
  1. Does dim 2028's polysemantic encoding at L50 decompose into separate
     Code, Math, and Sadness features with distinct decoder columns?
  2. Which SAE features align (cosine > 0.5) with each connectome category?
  3. Do aligned features show monosemantic or polysemantic category loading?

This is purely computational — no GPU needed. Runs on CPU in < 1 minute.

Usage:
    python scripts/sae/sae_connectome_alignment.py
"""
from __future__ import annotations

import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

# ── Paths ────────────────────────────────────────────────────
PROJECT_ROOT = Path("/home/orwel/dev_genius/experiments/Character Creation")
SAE_MODELS_DIR = PROJECT_ROOT / "sae_models" / "base"
CONNECTOME_PATH = PROJECT_ROOT / "qwen35_map" / "27b" / "connectome_zscores.pt"
OUTPUT_DIR = PROJECT_ROOT / "sae_analysis"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CATEGORIES = {
    "Domain: Code": 0,
    "Domain: History": 1,
    "Domain: Math": 2,
    "Domain: Science": 3,
    "Emotion: Anger": 4,
    "Emotion: Fear": 5,
    "Emotion: Joy": 6,
    "Emotion: Sadness": 7,
    "Identity": 8,
    "Language: EN vs CN": 9,
    "Reasoning: Analytical": 10,
    "Reasoning: Certainty": 11,
    "Role: Authority": 12,
    "Role: Teacher": 13,
    "Safety: Refusal": 14,
    "Sentiment: Positive": 15,
    "Tone: Formal": 16,
    "Tone: Polite": 17,
    "Tone: Sarcastic": 18,
    "Verbosity: Brief": 19,
}

# Hub neuron dim 2028 — the super-hub that codes for Code, Math, Sadness, Science, Analytical
HUB_DIM = 2028
HUB_CATEGORIES = ["Domain: Code", "Domain: Math", "Emotion: Sadness",
                   "Domain: Science", "Reasoning: Analytical"]

D_MODEL = 5120
D_SAE = 81920
K = 64

# Thresholds
HIGH_ALIGN_THRESHOLD = 0.5
MODERATE_ALIGN_THRESHOLD = 0.3


# ── SAE Model (minimal, for loading weights) ────────────────
class TopKSAE(nn.Module):
    def __init__(self, d_model: int, d_sae: int, k: int = 64):
        super().__init__()
        self.W_enc = nn.Parameter(torch.empty(d_sae, d_model))
        self.b_enc = nn.Parameter(torch.zeros(d_sae))
        self.W_dec = nn.Parameter(torch.empty(d_model, d_sae))
        self.b_dec = nn.Parameter(torch.zeros(d_model))


def load_sae_decoder(layer_idx: int) -> torch.Tensor:
    """Load SAE decoder weights W_dec [d_model, d_sae] for a given layer."""
    path = SAE_MODELS_DIR / f"L{layer_idx}" / "sae_final.pt"
    if not path.exists():
        raise FileNotFoundError(f"SAE model not found: {path}")
    state = torch.load(path, map_location="cpu", weights_only=True)
    print(f"[INFO] Loaded SAE L{layer_idx}: {path.name} ({path.stat().st_size / 1e9:.1f} GB)")
    return state["W_dec"]  # [d_model, d_sae]


def load_connectome_zscores() -> torch.Tensor:
    """Load connectome z-scores [20, 64, 5120]."""
    z = torch.load(CONNECTOME_PATH, map_location="cpu", weights_only=True)
    print(f"[INFO] Loaded connectome z-scores: {z.shape}")
    return z


# ── Core Analysis ────────────────────────────────────────────

def compute_alignment(
    z_scores_layer: torch.Tensor,  # [20, d_model]
    W_dec: torch.Tensor,           # [d_model, d_sae]
) -> torch.Tensor:
    """Compute cosine similarity between each z-score vector and each decoder column.

    Returns: [20, d_sae] cosine similarity matrix.
    """
    # Normalize z-score vectors to unit length
    z_norm = F.normalize(z_scores_layer, dim=1)  # [20, d_model]

    # Normalize decoder columns to unit length
    dec_norm = F.normalize(W_dec, dim=0)  # [d_model, d_sae]

    # Cosine similarity: [20, d_sae]
    cosines = z_norm @ dec_norm
    return cosines


def analyze_hub_decomposition(
    cosines: torch.Tensor,  # [20, d_sae]
    layer_idx: int,
) -> dict:
    """Analyze whether the super-hub (dim 2028) decomposes into separate features.

    For each hub category (Code, Math, Sadness, Science, Analytical), find features
    with high alignment. Then check if those features are DISTINCT (monosemantic
    decomposition) or SHARED (polysemantic blends).
    """
    hub_cat_indices = [CATEGORIES[c] for c in HUB_CATEGORIES]

    # For each hub category, find features with cosine > threshold
    category_features: dict[str, list[tuple[int, float]]] = {}
    for cat_name in HUB_CATEGORIES:
        cat_idx = CATEGORIES[cat_name]
        cos_row = cosines[cat_idx]  # [d_sae]
        high_mask = cos_row > HIGH_ALIGN_THRESHOLD
        mod_mask = cos_row > MODERATE_ALIGN_THRESHOLD

        high_features = [(int(i), float(cos_row[i])) for i in high_mask.nonzero(as_tuple=True)[0]]
        high_features.sort(key=lambda x: -x[1])

        mod_features = [(int(i), float(cos_row[i])) for i in mod_mask.nonzero(as_tuple=True)[0]]
        mod_features.sort(key=lambda x: -x[1])

        category_features[cat_name] = high_features

        print(f"\n  {cat_name}:")
        print(f"    Features with cosine > {HIGH_ALIGN_THRESHOLD}: {len(high_features)}")
        print(f"    Features with cosine > {MODERATE_ALIGN_THRESHOLD}: {len(mod_features)}")
        if high_features:
            top5 = high_features[:5]
            for feat_idx, cos_val in top5:
                # Check if this feature also aligns with other hub categories
                other_alignments = []
                for other_cat in HUB_CATEGORIES:
                    if other_cat != cat_name:
                        other_cos = float(cosines[CATEGORIES[other_cat], feat_idx])
                        if other_cos > MODERATE_ALIGN_THRESHOLD:
                            other_alignments.append(f"{other_cat}={other_cos:.3f}")
                cross_str = f" (also: {', '.join(other_alignments)})" if other_alignments else " (MONOSEMANTIC)"
                print(f"      Feature {feat_idx}: cos={cos_val:.4f}{cross_str}")

    # Compute category separation: for each pair of hub categories, what fraction
    # of their aligned features are shared vs distinct?
    all_feature_sets = {
        cat: set(f[0] for f in feats)
        for cat, feats in category_features.items()
    }

    separation_matrix: dict[str, dict[str, float]] = {}
    print(f"\n  Hub Category Feature Overlap (cosine > {HIGH_ALIGN_THRESHOLD}):")
    for cat_a in HUB_CATEGORIES:
        separation_matrix[cat_a] = {}
        for cat_b in HUB_CATEGORIES:
            if cat_a == cat_b:
                continue
            set_a = all_feature_sets[cat_a]
            set_b = all_feature_sets[cat_b]
            if len(set_a) == 0 and len(set_b) == 0:
                jaccard = 0.0
            elif len(set_a | set_b) == 0:
                jaccard = 0.0
            else:
                jaccard = len(set_a & set_b) / len(set_a | set_b)
            separation_matrix[cat_a][cat_b] = jaccard

    for cat_a in HUB_CATEGORIES:
        overlaps = []
        for cat_b in HUB_CATEGORIES:
            if cat_a != cat_b:
                j = separation_matrix[cat_a].get(cat_b, 0)
                if j > 0:
                    overlaps.append(f"{cat_b.split(': ')[-1]}={j:.2f}")
        n_feats = len(all_feature_sets[cat_a])
        overlap_str = ", ".join(overlaps) if overlaps else "none"
        print(f"    {cat_a}: {n_feats} features, Jaccard overlap: {overlap_str}")

    return {
        "category_features": {
            cat: [(f, round(c, 4)) for f, c in feats[:20]]
            for cat, feats in category_features.items()
        },
        "separation_matrix": separation_matrix,
        "n_features_per_category": {
            cat: len(feats) for cat, feats in category_features.items()
        },
    }


def analyze_all_categories(
    cosines: torch.Tensor,  # [20, d_sae]
    layer_idx: int,
) -> dict:
    """Full analysis of all 20 categories against all SAE features."""
    results: dict[str, dict] = {}

    print(f"\n{'='*70}")
    print(f"  ALL CATEGORY ALIGNMENT — L{layer_idx}")
    print(f"{'='*70}")

    for cat_name, cat_idx in CATEGORIES.items():
        cos_row = cosines[cat_idx]
        n_high = int((cos_row > HIGH_ALIGN_THRESHOLD).sum())
        n_mod = int((cos_row > MODERATE_ALIGN_THRESHOLD).sum())

        top_val, top_idx = cos_row.topk(5)
        top_features = [(int(top_idx[i]), float(top_val[i])) for i in range(5)]

        # For each top feature, find which OTHER categories also align
        top_with_cross: list[dict] = []
        for feat_idx, cos_val in top_features:
            cross = {}
            for other_cat, other_idx in CATEGORIES.items():
                if other_cat != cat_name:
                    other_cos = float(cosines[other_idx, feat_idx])
                    if other_cos > MODERATE_ALIGN_THRESHOLD:
                        cross[other_cat] = round(other_cos, 4)
            top_with_cross.append({
                "feature_idx": feat_idx,
                "cosine": round(cos_val, 4),
                "cross_category_alignments": cross,
                "is_monosemantic": len(cross) == 0,
            })

        results[cat_name] = {
            "n_high_align": n_high,
            "n_moderate_align": n_mod,
            "top5_features": top_with_cross,
            "max_cosine": round(float(cos_row.max()), 4),
            "mean_cosine": round(float(cos_row.mean()), 4),
        }

        # Compact print
        mono_count = sum(1 for f in top_with_cross if f["is_monosemantic"])
        top1 = top_with_cross[0]
        cross_str = ""
        if top1["cross_category_alignments"]:
            cross_names = [c.split(": ")[-1] for c in list(top1["cross_category_alignments"].keys())[:3]]
            cross_str = f" [+{', '.join(cross_names)}]"

        print(f"  {cat_name:25s} | high={n_high:4d} mod={n_mod:5d} | "
              f"top={top1['cosine']:.3f} (F{top1['feature_idx']}){cross_str} | "
              f"mono={mono_count}/5")

    return results


def dim_2028_deep_dive(
    W_dec: torch.Tensor,   # [d_model, d_sae]
    z_scores_layer: torch.Tensor,  # [20, d_model]
) -> dict:
    """Deep analysis of dim 2028: which SAE features load most on this neuron?"""
    print(f"\n{'='*70}")
    print(f"  DIM 2028 DEEP DIVE — Which SAE features load on the super-hub?")
    print(f"{'='*70}")

    # Decoder column for each feature tells us its direction in model space.
    # The component along dim 2028 tells us how much each feature "uses" dim 2028.
    dim_2028_loadings = W_dec[HUB_DIM, :]  # [d_sae]

    # Top features by absolute loading on dim 2028
    abs_loadings = dim_2028_loadings.abs()
    top_vals, top_indices = abs_loadings.topk(20)

    results: list[dict] = []
    print(f"\n  Top 20 features by |loading| on dim 2028:")
    print(f"  {'Feature':>8} | {'Loading':>8} | {'Primary Category (highest cosine)':>40} | Other Cats")
    print(f"  {'-'*100}")

    for rank in range(20):
        feat_idx = int(top_indices[rank])
        loading = float(dim_2028_loadings[feat_idx])

        # Find this feature's alignment with all categories
        feat_column = W_dec[:, feat_idx]  # [d_model]
        feat_norm = F.normalize(feat_column.unsqueeze(0), dim=1)
        z_norm = F.normalize(z_scores_layer, dim=1)
        cosines_for_feat = (z_norm @ feat_norm.t()).squeeze()  # [20]

        top_cos, top_cat_idx = cosines_for_feat.topk(3)
        primary_cat = list(CATEGORIES.keys())[int(top_cat_idx[0])]
        primary_cos = float(top_cos[0])

        other_cats = []
        for j in range(1, 3):
            cat_name = list(CATEGORIES.keys())[int(top_cat_idx[j])]
            cos_val = float(top_cos[j])
            if cos_val > MODERATE_ALIGN_THRESHOLD:
                other_cats.append(f"{cat_name.split(': ')[-1]}={cos_val:.3f}")

        others_str = ", ".join(other_cats) if other_cats else "—"
        print(f"  {feat_idx:>8} | {loading:>+8.4f} | {primary_cat:>30} cos={primary_cos:.3f} | {others_str}")

        results.append({
            "feature_idx": feat_idx,
            "dim_2028_loading": round(loading, 4),
            "primary_category": primary_cat,
            "primary_cosine": round(primary_cos, 4),
            "all_category_cosines": {
                cat: round(float(cosines_for_feat[idx]), 4)
                for cat, idx in CATEGORIES.items()
                if float(cosines_for_feat[idx]) > MODERATE_ALIGN_THRESHOLD
            },
        })

    # Key question: do the top features separate Code vs Math vs Sadness?
    print(f"\n  DECOMPOSITION TEST: Do top dim-2028 features separate hub categories?")
    code_best = None
    math_best = None
    sadness_best = None

    for feat_info in results:
        cos_dict = feat_info["all_category_cosines"]
        code_cos = cos_dict.get("Domain: Code", 0)
        math_cos = cos_dict.get("Domain: Math", 0)
        sad_cos = cos_dict.get("Emotion: Sadness", 0)

        if code_cos > 0.3 and (code_best is None or code_cos > code_best[1]):
            code_best = (feat_info["feature_idx"], code_cos, math_cos, sad_cos)
        if math_cos > 0.3 and (math_best is None or math_cos > math_best[1]):
            math_best = (feat_info["feature_idx"], code_cos, math_cos, sad_cos)
        if sad_cos > 0.3 and (sadness_best is None or sad_cos > sadness_best[1]):
            sadness_best = (feat_info["feature_idx"], code_cos, math_cos, sad_cos)

    decomposition_verdict = "UNKNOWN"
    if code_best and math_best and sadness_best:
        # Check if they are distinct features
        feat_set = {code_best[0], math_best[0], sadness_best[0]}
        if len(feat_set) == 3:
            decomposition_verdict = "DECOMPOSED — 3 distinct features for Code, Math, Sadness"
        elif len(feat_set) == 2:
            decomposition_verdict = "PARTIAL — 2 of 3 hub categories map to distinct features"
        else:
            decomposition_verdict = "ENTANGLED — all hub categories map to the same feature"
    elif code_best or math_best or sadness_best:
        decomposition_verdict = "PARTIAL — only some hub categories have aligned features"
    else:
        decomposition_verdict = "NO ALIGNMENT — no features with >0.3 cosine to hub categories"

    print(f"\n  VERDICT: {decomposition_verdict}")
    if code_best:
        print(f"    Code best:    F{code_best[0]} (Code={code_best[1]:.3f}, Math={code_best[2]:.3f}, Sad={code_best[3]:.3f})")
    if math_best:
        print(f"    Math best:    F{math_best[0]} (Code={math_best[1]:.3f}, Math={math_best[2]:.3f}, Sad={math_best[3]:.3f})")
    if sadness_best:
        print(f"    Sadness best: F{sadness_best[0]} (Code={sadness_best[1]:.3f}, Math={sadness_best[2]:.3f}, Sad={sadness_best[3]:.3f})")

    return {
        "top_features": results,
        "decomposition_verdict": decomposition_verdict,
        "code_best": code_best,
        "math_best": math_best,
        "sadness_best": sadness_best,
    }


# ── Main ─────────────────────────────────────────────────────

def main() -> None:
    print("=" * 70)
    print("  SAE–CONNECTOME ALIGNMENT ANALYSIS")
    print("  Qwen3.5-27B | L50 (super-hub) + L44 (sarcasm region)")
    print("=" * 70)

    # Load data
    z_scores = load_connectome_zscores()  # [20, 64, 5120]

    full_results: dict[str, dict] = {}

    for layer_idx in [50, 44]:
        print(f"\n{'#'*70}")
        print(f"  LAYER {layer_idx}")
        print(f"{'#'*70}")

        W_dec = load_sae_decoder(layer_idx)  # [d_model, d_sae]
        z_layer = z_scores[:, layer_idx, :]  # [20, d_model]

        # Compute cosine similarity matrix
        cosines = compute_alignment(z_layer, W_dec)  # [20, d_sae]
        print(f"[INFO] Cosine matrix: {cosines.shape} | "
              f"max={cosines.max():.4f} min={cosines.min():.4f}")

        # Full category analysis
        all_cat_results = analyze_all_categories(cosines, layer_idx)

        # Hub decomposition (L50 focus but run for both)
        if layer_idx == 50:
            hub_results = analyze_hub_decomposition(cosines, layer_idx)
            dim_results = dim_2028_deep_dive(W_dec, z_layer)
        else:
            hub_results = {}
            dim_results = {}

        layer_results = {
            "cosine_stats": {
                "max": round(float(cosines.max()), 4),
                "min": round(float(cosines.min()), 4),
                "mean": round(float(cosines.mean()), 4),
                "std": round(float(cosines.std()), 4),
            },
            "all_categories": all_cat_results,
            "hub_decomposition": hub_results,
            "dim_2028_analysis": dim_results,
        }

        full_results[f"L{layer_idx}"] = layer_results

        # Save per-layer cosine matrix for further analysis
        torch.save(cosines, OUTPUT_DIR / f"cosine_matrix_L{layer_idx}.pt")
        print(f"[INFO] Saved cosine matrix to {OUTPUT_DIR / f'cosine_matrix_L{layer_idx}.pt'}")

    # Save full results JSON
    output_path = OUTPUT_DIR / "connectome_alignment_results.json"
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(full_results, f, indent=2, ensure_ascii=False)
    print(f"\n[INFO] Full results saved to {output_path}")

    # Summary
    print(f"\n{'='*70}")
    print(f"  SUMMARY")
    print(f"{'='*70}")

    for layer_key, lr in full_results.items():
        stats = lr["cosine_stats"]
        n_high_total = sum(
            cat["n_high_align"] for cat in lr["all_categories"].values()
        )
        n_mono = sum(
            sum(1 for f in cat["top5_features"] if f["is_monosemantic"])
            for cat in lr["all_categories"].values()
        )
        print(f"  {layer_key}: max_cos={stats['max']:.3f}, "
              f"features with cosine>{HIGH_ALIGN_THRESHOLD}: {n_high_total}, "
              f"monosemantic top-5 features: {n_mono}/100")

    if "dim_2028_analysis" in full_results.get("L50", {}) and full_results["L50"]["dim_2028_analysis"]:
        verdict = full_results["L50"]["dim_2028_analysis"]["decomposition_verdict"]
        print(f"\n  DIM 2028 VERDICT: {verdict}")


if __name__ == "__main__":
    main()
