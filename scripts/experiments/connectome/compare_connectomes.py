#!/usr/bin/env python3
"""
Compare connectomes between base Qwen3.5-27B and abliterated variant.

Analyzes:
1. Per-category z-score differences (what did abliteration change?)
2. Refusal/compliance direction identification
3. Sarcasm/personality preservation check
4. Math/reasoning impact assessment
5. Layer-by-layer cosine similarity of category directions

Usage:
    python compare_connectomes.py \
        --base ./qwen35_map/27b \
        --abliterated ./qwen35_map/27b-abliterated \
        --output ./results/abliteration_comparison
"""

import argparse
import json
import torch
import numpy as np
from pathlib import Path
from datetime import datetime


def load_connectome(path: Path) -> tuple[torch.Tensor, list[str], dict]:
    """Load connectome zscores and metadata."""
    zscores = torch.load(path / "connectome_zscores.pt", map_location="cpu", weights_only=True)
    with open(path / "connectome_stats.json") as f:
        stats = json.load(f)
    cat_names = stats["categories"]
    return zscores, cat_names, stats


def cosine_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    """Cosine similarity between two vectors."""
    na, nb = a.norm(), b.norm()
    if na < 1e-8 or nb < 1e-8:
        return 0.0
    return (torch.dot(a, b) / (na * nb)).item()


def compare(base_path: Path, ablit_path: Path, output_dir: Path) -> None:
    """Run full comparison between base and abliterated connectomes."""
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading connectomes...")
    base_z, base_cats, base_stats = load_connectome(base_path)
    ablit_z, ablit_cats, ablit_stats = load_connectome(ablit_path)

    n_cats, n_layers, hidden_dim = base_z.shape
    print(f"  Base:        [{n_cats}, {n_layers}, {hidden_dim}]")
    print(f"  Abliterated: {list(ablit_z.shape)}")

    # Verify categories match
    if base_cats != ablit_cats:
        print("WARNING: Category lists don't match!")
        print(f"  Base: {base_cats}")
        print(f"  Abliterated: {ablit_cats}")
        # Use intersection
        common = [c for c in base_cats if c in ablit_cats]
        print(f"  Using {len(common)} common categories")
    else:
        common = base_cats
        print(f"  {len(common)} categories matched")

    cat_to_idx = {c: i for i, c in enumerate(base_cats)}
    ablit_cat_to_idx = {c: i for i, c in enumerate(ablit_cats)}

    report = []
    report.append(f"# Abliteration Comparison Report")
    report.append(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"**Base model**: {base_stats.get('model', 'Qwen/Qwen3.5-27B-FP8')}")
    report.append(f"**Abliterated**: huihui-ai/Huihui-Qwen3.5-27B-abliterated")
    report.append(f"**Method**: remove-refusals-with-transformers (single direction at 60% depth, projected out at all layers)")
    report.append(f"**Dimensions**: {n_cats} categories × {n_layers} layers × {hidden_dim} hidden\n")

    # ── 1. Per-Category Z-Score Magnitude Comparison ──────────────
    report.append("## 1. Z-Score Magnitude Changes (What Did Abliteration Do?)")
    report.append("")
    report.append("| Category | Base max|z| | Ablit max|z| | Delta | Base Top Layer | Ablit Top Layer | Direction Changed? |")
    report.append("|---|---|---|---|---|---|---|")

    cat_changes = {}
    for cat in common:
        bi = cat_to_idx[cat]
        ai = ablit_cat_to_idx[cat]

        base_max_z = base_z[bi].abs().max().item()
        ablit_max_z = ablit_z[ai].abs().max().item()
        delta = ablit_max_z - base_max_z

        base_top_layer = base_z[bi].abs().max(dim=1).values.argmax().item()
        ablit_top_layer = ablit_z[ai].abs().max(dim=1).values.argmax().item()

        # Per-layer cosine between base and abliterated directions
        cos_per_layer = []
        for l in range(n_layers):
            cos_per_layer.append(cosine_sim(base_z[bi, l], ablit_z[ai, l]))
        mean_cos = np.mean(cos_per_layer)

        changed = "YES" if mean_cos < 0.8 else ("partial" if mean_cos < 0.95 else "no")

        cat_changes[cat] = {
            "base_max_z": round(base_max_z, 2),
            "ablit_max_z": round(ablit_max_z, 2),
            "delta": round(delta, 2),
            "base_top_layer": base_top_layer,
            "ablit_top_layer": ablit_top_layer,
            "mean_cos": round(mean_cos, 4),
            "changed": changed,
            "cos_per_layer": [round(c, 4) for c in cos_per_layer],
        }

        report.append(
            f"| {cat} | {base_max_z:.2f} | {ablit_max_z:.2f} | "
            f"{delta:+.2f} | L{base_top_layer} | L{ablit_top_layer} | {changed} |"
        )

    report.append("")

    # ── 2. Most Changed Categories ────────────────────────────────
    report.append("## 2. Most Changed Categories (Sorted by |Delta|)")
    report.append("")
    sorted_by_change = sorted(cat_changes.items(), key=lambda x: abs(x[1]["delta"]), reverse=True)
    for cat, info in sorted_by_change[:10]:
        report.append(f"- **{cat}**: {info['delta']:+.2f} z-score change, "
                      f"direction cosine={info['mean_cos']:.3f}, changed={info['changed']}")
    report.append("")

    # ── 3. Layer-by-Layer Direction Cosine ────────────────────────
    report.append("## 3. Layer-by-Layer Direction Preservation")
    report.append("")
    report.append("Mean cosine between base and abliterated directions per layer, "
                  "averaged across all categories:")
    report.append("")

    layer_mean_cos = []
    for l in range(n_layers):
        cos_vals = []
        for cat in common:
            bi = cat_to_idx[cat]
            ai = ablit_cat_to_idx[cat]
            cos_vals.append(cosine_sim(base_z[bi, l], ablit_z[ai, l]))
        layer_mean_cos.append(np.mean(cos_vals))

    report.append("| Layer | Mean Cosine | Interpretation |")
    report.append("|---|---|---|")
    for l in range(n_layers):
        interp = "PRESERVED" if layer_mean_cos[l] > 0.95 else (
            "modified" if layer_mean_cos[l] > 0.80 else "**DISRUPTED**"
        )
        if l % 4 == 0 or layer_mean_cos[l] < 0.90:  # Show every 4th layer + any disrupted
            report.append(f"| L{l:02d} | {layer_mean_cos[l]:.4f} | {interp} |")

    report.append("")

    # Find the most disrupted layers
    disrupted = [(l, layer_mean_cos[l]) for l in range(n_layers) if layer_mean_cos[l] < 0.90]
    if disrupted:
        report.append(f"**Most disrupted layers**: {', '.join(f'L{l}({c:.3f})' for l, c in sorted(disrupted, key=lambda x: x[1]))}")
    else:
        report.append("**No severely disrupted layers** (all > 0.90 cosine)")
    report.append("")

    # ── 4. The Refusal Direction ──────────────────────────────────
    report.append("## 4. Identifying the Abliteration Direction")
    report.append("")
    report.append("The abliteration tool extracts at 60% depth (L38 for 64-layer model) "
                  "using mean(harmful) - mean(harmless). We can reconstruct what changed "
                  "by computing the difference between base and abliterated connectome directions.")
    report.append("")

    # Compute per-category, per-layer difference vectors
    # The abliteration should show up as a consistent direction removed
    # across categories that involve compliance/refusal

    # Key categories to check for refusal-related changes
    refusal_adjacent = ["Identity", "Tone: Polite", "Tone: Formal", "Sentiment: Positive"]
    personality_cats = ["Tone: Sarcastic", "Emotion: Anger", "Role: Authority", "Verbosity: Brief"]
    reasoning_cats = ["Domain: Math", "Domain: Code", "Domain: Science", "Reasoning: Analytical"]

    report.append("### Refusal-Adjacent Categories")
    for cat in refusal_adjacent:
        if cat in cat_changes:
            info = cat_changes[cat]
            report.append(f"- **{cat}**: delta={info['delta']:+.2f}, cos={info['mean_cos']:.3f}, changed={info['changed']}")

    report.append("\n### Personality Categories (Should Be Preserved)")
    for cat in personality_cats:
        if cat in cat_changes:
            info = cat_changes[cat]
            report.append(f"- **{cat}**: delta={info['delta']:+.2f}, cos={info['mean_cos']:.3f}, changed={info['changed']}")

    report.append("\n### Reasoning Categories (Should Be Preserved)")
    for cat in reasoning_cats:
        if cat in cat_changes:
            info = cat_changes[cat]
            report.append(f"- **{cat}**: delta={info['delta']:+.2f}, cos={info['mean_cos']:.3f}, changed={info['changed']}")
    report.append("")

    # ── 5. Abliteration at L38 specifically ───────────────────────
    l38 = 38  # 60% of 64 layers
    report.append(f"## 5. Layer {l38} Analysis (Abliteration Extraction Point)")
    report.append("")
    report.append("The abliteration tool extracts the refusal direction from L38 specifically.")
    report.append("")

    report.append(f"| Category | Base z@L{l38} | Ablit z@L{l38} | Cosine@L{l38} |")
    report.append("|---|---|---|---|")
    for cat in common:
        bi = cat_to_idx[cat]
        ai = ablit_cat_to_idx[cat]
        bz = base_z[bi, l38].abs().max().item()
        az = ablit_z[ai, l38].abs().max().item()
        cos = cosine_sim(base_z[bi, l38], ablit_z[ai, l38])
        report.append(f"| {cat} | {bz:.2f} | {az:.2f} | {cos:.4f} |")
    report.append("")

    # ── 6. Global Difference Vector ───────────────────────────────
    report.append("## 6. Global Abliteration Fingerprint")
    report.append("")
    report.append("Compute the mean absolute difference across all categories to find "
                  "which dimensions were most affected by abliteration:")
    report.append("")

    # Mean abs difference per layer
    diff_per_layer = torch.zeros(n_layers)
    for l in range(n_layers):
        diffs = []
        for cat in common:
            bi = cat_to_idx[cat]
            ai = ablit_cat_to_idx[cat]
            diffs.append((base_z[bi, l] - ablit_z[ai, l]).abs().mean().item())
        diff_per_layer[l] = np.mean(diffs)

    top_layers = diff_per_layer.argsort(descending=True)[:10]
    report.append(f"**Most affected layers**: {', '.join(f'L{l.item()}({diff_per_layer[l]:.3f})' for l in top_layers)}")
    report.append("")

    # Top affected dimensions at L38
    all_dim_diff = torch.zeros(hidden_dim)
    for cat in common:
        bi = cat_to_idx[cat]
        ai = ablit_cat_to_idx[cat]
        all_dim_diff += (base_z[bi, l38] - ablit_z[ai, l38]).abs()
    all_dim_diff /= len(common)

    top_dims = all_dim_diff.argsort(descending=True)[:20]
    report.append(f"**Top 20 affected dimensions at L{l38}**:")
    for d in top_dims:
        report.append(f"  - dim {d.item()}: mean |delta|={all_dim_diff[d]:.3f}")
    report.append("")

    # ── 7. Implications for Our Steering ──────────────────────────
    report.append("## 7. Implications for Steering Research")
    report.append("")
    report.append("Key questions this comparison answers:")
    report.append("1. Does the abliterated model have a cleaner personality signal "
                  "(less compliance interference)?")
    report.append("2. Are our steering vectors (L48-L62) in a subspace that overlaps "
                  "with the abliteration direction?")
    report.append("3. Could we combine abliteration + steering for better results?")
    report.append("4. Does the abliterated model's connectome suggest better layer "
                  "targets than we found on the base model?")
    report.append("")

    # Check overlap with our steering band
    steering_layers = list(range(48, 63))
    steering_cos = [layer_mean_cos[l] for l in steering_layers if l < n_layers]
    report.append(f"**Steering band (L48-L62) preservation**: mean cosine = {np.mean(steering_cos):.4f}")
    if np.mean(steering_cos) > 0.95:
        report.append("→ Our steering vectors should work on the abliterated model with minimal change")
    elif np.mean(steering_cos) > 0.80:
        report.append("→ Our steering vectors may need re-calibration on the abliterated model")
    else:
        report.append("→ The abliteration significantly changed the steering band — vectors need re-extraction")
    report.append("")

    # ── Save ──────────────────────────────────────────────────────
    report_text = "\n".join(report)
    report_path = output_dir / "abliteration_comparison.md"
    with open(report_path, "w") as f:
        f.write(report_text)
    print(f"\nReport saved to {report_path}")
    print(report_text[:3000])

    # Save structured data
    data = {
        "categories": common,
        "cat_changes": cat_changes,
        "layer_mean_cos": [round(c, 4) for c in layer_mean_cos],
        "diff_per_layer": diff_per_layer.tolist(),
        "top_affected_dims_L38": [d.item() for d in top_dims],
        "steering_band_cos": round(float(np.mean(steering_cos)), 4),
    }
    with open(output_dir / "comparison_data.json", "w") as f:
        json.dump(data, f, indent=2)
    print(f"Data saved to {output_dir / 'comparison_data.json'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare base vs abliterated connectomes")
    parser.add_argument("--base", default="./qwen35_map/27b",
                        help="Path to base model connectome directory")
    parser.add_argument("--abliterated", default="./qwen35_map/27b-abliterated",
                        help="Path to abliterated model connectome directory")
    parser.add_argument("--output", default="./results/abliteration_comparison",
                        help="Output directory for comparison results")
    args = parser.parse_args()

    compare(Path(args.base), Path(args.abliterated), Path(args.output))


if __name__ == "__main__":
    main()
