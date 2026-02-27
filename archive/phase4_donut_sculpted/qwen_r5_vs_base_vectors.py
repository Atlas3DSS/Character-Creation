#!/usr/bin/env python3
"""
Connectome divergence analysis: R5-specific vs base Qwen steering vectors.

Computes per-layer cosine similarity between R5 and base Qwen connectomes for:
  1. The raw sarcasm direction (category index 6: "Tone: Sarcastic")
  2. The compound steering vector (same recipe as qwen_sculpted_donut.py)

Identifies which layers have the most divergence between the two connectomes.
No model loading needed — pure tensor math on saved connectome files.

Output: qwen_r5_vs_base_vectors_results.json

Category index reference (from qwen_connectome_probe.py CATEGORIES list):
  0  = Identity
  1  = Emotion: Joy
  2  = Emotion: Sadness
  3  = Emotion: Anger         <- push (weight 0.5)
  4  = Emotion: Fear
  5  = Tone: Formal           <- pull (weight -0.3)
  6  = Tone: Sarcastic        <- push (weight 1.0) / primary sarcasm direction
  7  = Tone: Polite           <- pull (weight -0.5)
  8  = Domain: Math           <- protect (Gram-Schmidt)
  9  = Domain: Science        <- protect (Gram-Schmidt)
  10 = Domain: Code           <- protect (Gram-Schmidt)
  11 = Domain: History
  12 = Reasoning: Analytical  <- protect (Gram-Schmidt)
  13 = Reasoning: Certainty
  14 = Safety: Refusal
  15 = Role: Teacher
  16 = Role: Authority        <- push (weight 0.3)
  17 = Verbosity: Brief       <- push (push in compound, per sculpted donut recipe)
  18 = Language: EN vs CN
  19 = Sentiment: Positive    <- pull (weight -0.3)

Compound recipe (from qwen_sculpted_donut.py build_compound):
  PUSH:    cat 6 (sarcasm) * 1.0, cat 3 (anger) * 0.5, cat 16 (authority) * 0.3
  PULL:    cat 7 (polite)  *-0.5, cat 5 (formal)*-0.3, cat 19 (positive)  *-0.3
  PROTECT: cats [8, 10, 9, 12] (math, code, science, analytical) via Gram-Schmidt
"""

import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F


# ─── Paths ────────────────────────────────────────────────────────────────────

BASE_CONNECTOME_PATH = (
    "/home/orwel/dev_genius/experiments/Character Creation/"
    "qwen_connectome/analysis/connectome_zscores.pt"
)
R5_CONNECTOME_PATH = (
    "/home/orwel/dev_genius/experiments/Character Creation/"
    "qwen_r5_connectome/analysis/connectome_zscores.pt"
)
OUTPUT_PATH = (
    "/home/orwel/dev_genius/experiments/Character Creation/"
    "qwen_r5_vs_base_vectors_results.json"
)

# ─── Compound vector recipe (mirrors qwen_sculpted_donut.py) ──────────────────

PUSH_CATS = {6: 1.0, 3: 0.5, 16: 0.3}   # sarcasm, anger, authority
PULL_CATS = {7: -0.5, 5: -0.3, 19: -0.3} # polite, formal, positive
PROTECT_CATS = [8, 10, 9, 12]            # math, code, science, analytical

# Human-readable category names by index
CATEGORY_NAMES = {
    0:  "Identity",
    1:  "Emotion: Joy",
    2:  "Emotion: Sadness",
    3:  "Emotion: Anger",
    4:  "Emotion: Fear",
    5:  "Tone: Formal",
    6:  "Tone: Sarcastic",
    7:  "Tone: Polite",
    8:  "Domain: Math",
    9:  "Domain: Science",
    10: "Domain: Code",
    11: "Domain: History",
    12: "Reasoning: Analytical",
    13: "Reasoning: Certainty",
    14: "Safety: Refusal",
    15: "Role: Teacher",
    16: "Role: Authority",
    17: "Verbosity: Brief",
    18: "Language: EN vs CN",
    19: "Sentiment: Positive",
}


# ─── Helpers ──────────────────────────────────────────────────────────────────

def gram_schmidt_project_out(vec: torch.Tensor, protect_vecs: list[torch.Tensor]) -> torch.Tensor:
    """Remove components in the direction of each protect vector (Gram-Schmidt)."""
    result = vec.clone()
    for pv in protect_vecs:
        pn = torch.dot(pv, pv)
        if pn > 1e-8:
            result = result - (torch.dot(result, pv) / pn) * pv
    return result


def build_compound_vectors(connectome: torch.Tensor) -> dict[int, torch.Tensor]:
    """
    Build normalized compound steering vectors for every layer.

    connectome: shape (n_cats, n_layers, hidden_dim)
    Returns: dict layer_idx -> unit vector of shape (hidden_dim,)
    """
    n_cats, n_layers, hidden_dim = connectome.shape
    compound: dict[int, torch.Tensor] = {}

    for layer in range(n_layers):
        vec = torch.zeros(hidden_dim)

        # Push + pull weighted sum
        for cat, weight in {**PUSH_CATS, **PULL_CATS}.items():
            if cat < n_cats:
                vec = vec + weight * connectome[cat, layer, :]

        # Gram-Schmidt: project out protect directions
        protect_vecs = [connectome[p, layer, :] for p in PROTECT_CATS if p < n_cats]
        vec = gram_schmidt_project_out(vec, protect_vecs)

        # Normalize to unit vector
        norm = vec.norm()
        if norm > 1e-8:
            vec = vec / norm
        else:
            # Zero vector — connectome is degenerate at this layer
            vec = torch.zeros(hidden_dim)

        compound[layer] = vec

    return compound


def cosine_similarity_1d(a: torch.Tensor, b: torch.Tensor) -> float:
    """Cosine similarity between two 1-D vectors (handles zero norm gracefully)."""
    na = a.norm()
    nb = b.norm()
    if na < 1e-8 or nb < 1e-8:
        return float("nan")
    return float(torch.dot(a, b) / (na * nb))


def analyze_divergence(
    similarities: dict[int, float],
    label: str,
) -> dict:
    """Compute summary statistics over per-layer cosine similarities."""
    vals = [v for v in similarities.values() if not (v != v)]  # drop NaN
    if not vals:
        return {"label": label, "error": "all NaN"}

    mean_sim = sum(vals) / len(vals)
    min_sim = min(vals)
    max_sim = max(vals)
    n_layers = len(vals)

    # Divergence = 1 - cosine_similarity (higher = more different)
    divergences = {layer: 1.0 - sim for layer, sim in similarities.items() if sim == sim}
    sorted_by_divergence = sorted(divergences.items(), key=lambda x: x[1], reverse=True)

    return {
        "label": label,
        "n_layers": n_layers,
        "mean_cosine_similarity": round(mean_sim, 6),
        "min_cosine_similarity": round(min_sim, 6),
        "max_cosine_similarity": round(max_sim, 6),
        "mean_divergence": round(1.0 - mean_sim, 6),
        "top5_most_divergent_layers": [
            {"layer": int(l), "divergence": round(d, 6), "cosine_sim": round(1.0 - d, 6)}
            for l, d in sorted_by_divergence[:5]
        ],
        "top5_most_similar_layers": [
            {"layer": int(l), "divergence": round(d, 6), "cosine_sim": round(1.0 - d, 6)}
            for l, d in sorted_by_divergence[-5:][::-1]
        ],
        "per_layer_cosine_similarity": {
            str(layer): round(sim, 6)
            for layer, sim in sorted(similarities.items())
        },
        "per_layer_divergence": {
            str(layer): round(div, 6)
            for layer, div in sorted(divergences.items())
        },
    }


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 60)
    print("R5 vs Base Qwen Connectome Divergence Analysis")
    print("=" * 60)

    # ── Load connectomes ──────────────────────────────────────────────────────
    base_path = Path(BASE_CONNECTOME_PATH)
    r5_path   = Path(R5_CONNECTOME_PATH)

    for p, label in [(base_path, "Base"), (r5_path, "R5")]:
        if not p.exists():
            print(f"ERROR: {label} connectome not found at: {p}", file=sys.stderr)
            sys.exit(1)

    print(f"\nLoading base connectome from:\n  {base_path}")
    base_ct = torch.load(str(base_path), map_location="cpu", weights_only=True)

    print(f"Loading R5 connectome from:\n  {r5_path}")
    r5_ct = torch.load(str(r5_path), map_location="cpu", weights_only=True)

    print(f"\nBase connectome shape: {tuple(base_ct.shape)}")
    print(f"R5 connectome shape:   {tuple(r5_ct.shape)}")

    if base_ct.shape != r5_ct.shape:
        print(
            f"WARNING: shape mismatch! base={base_ct.shape}, r5={r5_ct.shape}. "
            "Analysis will proceed on the common dimensions.",
            file=sys.stderr,
        )

    n_cats   = min(base_ct.shape[0], r5_ct.shape[0])
    n_layers = min(base_ct.shape[1], r5_ct.shape[1])
    hidden   = min(base_ct.shape[2], r5_ct.shape[2])

    # Trim to common size if shapes differ
    base_ct = base_ct[:n_cats, :n_layers, :hidden]
    r5_ct   = r5_ct[:n_cats, :n_layers, :hidden]

    print(f"Working dimensions: n_cats={n_cats}, n_layers={n_layers}, hidden={hidden}")

    # ── 1. Per-category raw vector cosine similarities ────────────────────────
    print("\n--- Per-category raw direction similarities (all layers) ---")
    per_cat_mean_sim: dict[int, float] = {}
    per_cat_layer_sim: dict[int, dict[int, float]] = {}

    for cat in range(n_cats):
        sims = {}
        for layer in range(n_layers):
            bv = base_ct[cat, layer, :]
            rv = r5_ct[cat, layer, :]
            sims[layer] = cosine_similarity_1d(bv, rv)
        mean_s = sum(v for v in sims.values() if v == v) / n_layers
        per_cat_mean_sim[cat] = mean_s
        per_cat_layer_sim[cat] = sims
        cat_name = CATEGORY_NAMES.get(cat, f"cat_{cat}")
        print(f"  cat {cat:2d} ({cat_name:30s}): mean cosine sim = {mean_s:.4f}")

    # ── 2. Sarcasm direction per-layer (category 6) ──────────────────────────
    print("\n--- Sarcasm direction (cat 6: Tone: Sarcastic) per-layer ---")
    sarcasm_sims: dict[int, float] = {}
    for layer in range(n_layers):
        bv = base_ct[6, layer, :]
        rv = r5_ct[6, layer, :]
        sim = cosine_similarity_1d(bv, rv)
        sarcasm_sims[layer] = sim
        print(f"  L{layer:02d}: cosine_sim={sim:.4f}  divergence={1.0 - sim:.4f}")

    sarcasm_analysis = analyze_divergence(sarcasm_sims, "Sarcasm direction (cat 6)")

    # ── 3. Compound steering vector per-layer ────────────────────────────────
    print("\n--- Building compound steering vectors ---")
    base_compound = build_compound_vectors(base_ct)
    r5_compound   = build_compound_vectors(r5_ct)
    print(f"  Base compound: {len(base_compound)} layers")
    print(f"  R5 compound:   {len(r5_compound)} layers")

    print("\n--- Compound vector per-layer similarities ---")
    compound_sims: dict[int, float] = {}
    for layer in range(n_layers):
        bv = base_compound[layer]
        rv = r5_compound[layer]
        sim = cosine_similarity_1d(bv, rv)
        compound_sims[layer] = sim
        print(f"  L{layer:02d}: cosine_sim={sim:.4f}  divergence={1.0 - sim:.4f}")

    compound_analysis = analyze_divergence(compound_sims, "Compound steering vector")

    # ── 4. Cross-category structural similarity at each layer ─────────────────
    # Full category-pair cosine sim matrix at every layer for base vs R5
    # Summarised as: mean |cos_sim(base_cat_i, r5_cat_i)| per layer
    print("\n--- Per-layer mean across all categories ---")
    layer_mean_sims: dict[int, float] = {}
    for layer in range(n_layers):
        layer_sims = []
        for cat in range(n_cats):
            bv = base_ct[cat, layer, :]
            rv = r5_ct[cat, layer, :]
            s = cosine_similarity_1d(bv, rv)
            if s == s:  # not NaN
                layer_sims.append(s)
        mean_s = sum(layer_sims) / len(layer_sims) if layer_sims else float("nan")
        layer_mean_sims[layer] = mean_s
        print(f"  L{layer:02d}: mean cat cosine sim = {mean_s:.4f}  divergence = {1.0 - mean_s:.4f}")

    layer_mean_analysis = analyze_divergence(layer_mean_sims, "Mean across all categories per layer")

    # ── 5. Highlight layers most relevant to the donut range ─────────────────
    DONUT_QUALITY_LAYERS = list(range(16, 28))  # L16-27 (quality-preserving donut)
    DONUT_FULL_LAYERS    = list(range(8, 28))   # L8-27 (full donut)

    def layer_range_stats(sims: dict[int, float], layers: list[int], label: str) -> dict:
        vals = [sims[l] for l in layers if l in sims and sims[l] == sims[l]]
        if not vals:
            return {}
        mean_s = sum(vals) / len(vals)
        return {
            "label": label,
            "layers": layers,
            "mean_cosine_similarity": round(mean_s, 6),
            "mean_divergence": round(1.0 - mean_s, 6),
        }

    donut_sarcasm_quality = layer_range_stats(sarcasm_sims,  DONUT_QUALITY_LAYERS, "sarcasm @ L16-27")
    donut_sarcasm_full    = layer_range_stats(sarcasm_sims,  DONUT_FULL_LAYERS,    "sarcasm @ L8-27")
    donut_compound_quality= layer_range_stats(compound_sims, DONUT_QUALITY_LAYERS, "compound @ L16-27")
    donut_compound_full   = layer_range_stats(compound_sims, DONUT_FULL_LAYERS,    "compound @ L8-27")

    # ── Assemble results ──────────────────────────────────────────────────────
    results = {
        "metadata": {
            "base_connectome_path": str(base_path),
            "r5_connectome_path":   str(r5_path),
            "shape": {"n_cats": n_cats, "n_layers": n_layers, "hidden_dim": hidden},
            "compound_recipe": {
                "push_cats":    {str(k): v for k, v in PUSH_CATS.items()},
                "pull_cats":    {str(k): v for k, v in PULL_CATS.items()},
                "protect_cats": PROTECT_CATS,
                "push_cat_names":    {str(k): CATEGORY_NAMES.get(k, "?") for k in PUSH_CATS},
                "pull_cat_names":    {str(k): CATEGORY_NAMES.get(k, "?") for k in PULL_CATS},
                "protect_cat_names": {str(k): CATEGORY_NAMES.get(k, "?") for k in PROTECT_CATS},
            },
        },
        "sarcasm_direction_analysis": sarcasm_analysis,
        "compound_vector_analysis": compound_analysis,
        "layer_mean_all_cats_analysis": layer_mean_analysis,
        "donut_range_summaries": {
            "sarcasm_quality_donut_L16_27":   donut_sarcasm_quality,
            "sarcasm_full_donut_L8_27":       donut_sarcasm_full,
            "compound_quality_donut_L16_27":  donut_compound_quality,
            "compound_full_donut_L8_27":      donut_compound_full,
        },
        "per_category_mean_similarity": {
            str(cat): {
                "name": CATEGORY_NAMES.get(cat, f"cat_{cat}"),
                "mean_cosine_sim": round(sim, 6),
                "mean_divergence": round(1.0 - sim, 6),
                "per_layer": {str(l): round(s, 6) for l, s in per_cat_layer_sim[cat].items()},
            }
            for cat, sim in sorted(per_cat_mean_sim.items())
        },
    }

    # ── Print summary ─────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    print(f"\nSarcasm direction (cat 6):")
    print(f"  Mean cosine sim:  {sarcasm_analysis['mean_cosine_similarity']:.4f}")
    print(f"  Mean divergence:  {sarcasm_analysis['mean_divergence']:.4f}")
    print(f"  Top divergent layers: {[x['layer'] for x in sarcasm_analysis['top5_most_divergent_layers']]}")
    print(f"  Most similar layers:  {[x['layer'] for x in sarcasm_analysis['top5_most_similar_layers']]}")

    print(f"\nCompound steering vector:")
    print(f"  Mean cosine sim:  {compound_analysis['mean_cosine_similarity']:.4f}")
    print(f"  Mean divergence:  {compound_analysis['mean_divergence']:.4f}")
    print(f"  Top divergent layers: {[x['layer'] for x in compound_analysis['top5_most_divergent_layers']]}")
    print(f"  Most similar layers:  {[x['layer'] for x in compound_analysis['top5_most_similar_layers']]}")

    print(f"\nDonut range summaries:")
    for key, val in results["donut_range_summaries"].items():
        if val:
            print(f"  {val['label']:35s}: sim={val['mean_cosine_similarity']:.4f}, div={val['mean_divergence']:.4f}")

    print(f"\nPer-category divergence ranking (highest first):")
    ranked_cats = sorted(per_cat_mean_sim.items(), key=lambda x: x[1])
    for cat, sim in ranked_cats:
        div = 1.0 - sim
        name = CATEGORY_NAMES.get(cat, f"cat_{cat}")
        marker = " <-- PUSH/PULL/PROTECT" if cat in {*PUSH_CATS, *PULL_CATS, *PROTECT_CATS} else ""
        print(f"  cat {cat:2d} ({name:30s}): sim={sim:.4f}, div={div:.4f}{marker}")

    # ── Save results ──────────────────────────────────────────────────────────
    out_path = Path(OUTPUT_PATH)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to:\n  {out_path}")
    print("Done.")


if __name__ == "__main__":
    main()
