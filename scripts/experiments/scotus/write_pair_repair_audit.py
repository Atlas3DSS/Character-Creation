#!/usr/bin/env python3
"""Write a concise audit report for repaired SCOTUS pair manifests."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PAIRS = PROJECT_ROOT / "data" / "scotus" / "scotus_matched_pairs_v2.jsonl"
DEFAULT_QUALITY = PROJECT_ROOT / "data" / "scotus" / "manifests" / "scotus_pair_quality_v2.json"
DEFAULT_BASELINES = PROJECT_ROOT / "data" / "scotus" / "manifests" / "scotus_baseline_results_v2.json"
DEFAULT_EXCLUDED = PROJECT_ROOT / "data" / "scotus" / "processed" / "scotus_excluded_chunk_inventory_v2.jsonl"
DEFAULT_REPORT = PROJECT_ROOT / "reports" / "scotus_pair_repair_audit.md"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return "\n".join(lines)


def best_masked_scores(baselines: dict[str, Any]) -> list[list[Any]]:
    rows: list[list[Any]] = []
    for key, payload in sorted(baselines.items()):
        if not key.endswith("/masked"):
            continue
        test_models = payload.get("models", {}).get("test", {})
        text_scores = {
            model_name: metrics.get("balanced_accuracy", 0.0)
            for model_name, metrics in test_models.items()
            if "tfidf" in model_name
        }
        best_model, best_score = max(text_scores.items(), key=lambda item: item[1]) if text_scores else ("none", 0.0)
        metadata_score = test_models.get("metadata_logreg", {}).get("balanced_accuracy", 0.0)
        length_score = test_models.get("length_citation_logreg", {}).get("balanced_accuracy", 0.0)
        pair = key.split("/", 1)[0]
        decision = "activation-ready" if best_score >= 0.75 else "exploratory" if best_score >= 0.70 else "no-go"
        rows.append([pair, decision, best_model, f"{best_score:.3f}", f"{metadata_score:.3f}", f"{length_score:.3f}"])
    return rows


def masked_gate_scores(baselines: dict[str, Any]) -> dict[str, float]:
    scores: dict[str, float] = {}
    for key, payload in sorted(baselines.items()):
        if not key.endswith("/masked"):
            continue
        test_models = payload.get("models", {}).get("test", {})
        text_scores = [
            metrics.get("balanced_accuracy", 0.0)
            for model_name, metrics in test_models.items()
            if "tfidf" in model_name
        ]
        scores[key.split("/", 1)[0]] = max(text_scores) if text_scores else 0.0
    return scores


def write_report(
    report: Path,
    pairs: list[dict[str, Any]],
    quality: dict[str, Any],
    baselines: dict[str, Any],
    excluded: list[dict[str, Any]],
) -> None:
    lines = [
        "# SCOTUS Pair Repair Audit",
        "",
        "Phase 3.5D-E output audit for sectioned, reasoning-filtered SCOTUS chunks.",
        "",
        "## Pair Counts",
        "",
    ]
    pair_count_rows = []
    pair_counts = quality.get("pair_counts", {})
    for key, splits in sorted(pair_counts.items()):
        pair_count_rows.append(
            [
                key,
                splits.get("train", 0),
                splits.get("dev", 0),
                splits.get("test", 0),
                sum(int(v) for v in splits.values()),
                quality.get("same_case_pair_counts", {}).get(key, 0),
            ]
        )
    lines.append(markdown_table(["Pair/Variant", "Train", "Dev", "Test", "Total", "Same-case"], pair_count_rows))

    lines.extend(["", "## Eligible Chunk Counts", ""])
    excluded_counts = Counter(
        row.get("justice", "unknown")
        for row in excluded
        if row.get("text_variant") in {None, "raw_clean"}
    )
    chunk_rows = []
    for key, counts in sorted(quality.get("chunk_counts", {}).items()):
        if "/posture" in key:
            continue
        chunk_rows.append([key, counts.get("eligible", 0), excluded_counts.get(key, 0)])
    lines.append(markdown_table(["Justice", "Eligible raw chunks", "Excluded block/chunk records"], chunk_rows))

    lines.extend(["", "## Posture Mix", ""])
    posture_rows = []
    for key, counts in sorted(quality.get("chunk_counts", {}).items()):
        if "/posture" not in key:
            continue
        justice = key.split("/", 1)[0]
        for posture, count in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
            posture_rows.append([justice, posture, count])
    lines.append(markdown_table(["Justice", "Section Posture", "Eligible raw chunks"], posture_rows))

    lines.extend(["", "## Baseline Gate", ""])
    lines.append(
        markdown_table(
            [
                "Pair",
                "Decision",
                "Best masked test model",
                "Best masked balanced accuracy",
                "Metadata-only",
                "Length/citation-only",
            ],
            best_masked_scores(baselines),
        )
    )

    lines.extend(["", "## Matching Diagnostics", ""])
    key_counts: Counter[tuple[str, str]] = Counter()
    same_case_examples = []
    for pair in pairs:
        if pair["text_variant"] != "masked":
            continue
        key_counts[(pair["pair"], "|".join(str(x) for x in pair.get("matching_key", [])))] += 1
        if pair.get("case_id_a") == pair.get("case_id_b") and len(same_case_examples) < 12:
            same_case_examples.append(
                [
                    pair["pair"],
                    pair["split"],
                    pair.get("case_id_a"),
                    pair.get("issue_area_label"),
                    pair.get("section_posture_a"),
                    pair.get("chunk_position_bucket_a"),
                ]
            )
    top_key_rows = [[pair, key, count] for (pair, key), count in key_counts.most_common(20)]
    lines.append(markdown_table(["Pair", "Matching Key", "Masked pairs"], top_key_rows))

    lines.extend(["", "## Same-Case Examples", ""])
    lines.append(markdown_table(["Pair", "Split", "Case ID", "Issue Area", "Posture", "Position"], same_case_examples))

    gate_scores = masked_gate_scores(baselines)
    scalia_ginsburg = gate_scores.get("Scalia_vs_Ginsburg", 0.0)
    thomas_souter = gate_scores.get("Thomas_vs_Souter", 0.0)
    sg_decision = (
        "- Scalia vs. Ginsburg clears the repaired-corpus activation gate."
        if scalia_ginsburg >= 0.75
        else "- Scalia vs. Ginsburg does not clear the repaired-corpus activation gate."
    )
    ts_decision = (
        "- Thomas vs. Souter also clears the numeric masked held-out gate in this run, but remains secondary until dev/test behavior stabilizes."
        if thomas_souter >= 0.75
        else "- Thomas vs. Souter remains secondary and does not drive the Phase 4 decision."
    )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            sg_decision,
            ts_decision,
            "- Metadata-only remains at chance when its score is near 0.500, so the matching metadata is not by itself separating the labels.",
        ]
    )
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write SCOTUS v2 pair repair audit report.")
    parser.add_argument("--pairs", type=Path, default=DEFAULT_PAIRS)
    parser.add_argument("--quality", type=Path, default=DEFAULT_QUALITY)
    parser.add_argument("--baselines", type=Path, default=DEFAULT_BASELINES)
    parser.add_argument("--excluded", type=Path, default=DEFAULT_EXCLUDED)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pairs = read_jsonl(args.pairs)
    quality = json.loads(args.quality.read_text(encoding="utf-8"))
    baselines = json.loads(args.baselines.read_text(encoding="utf-8"))
    excluded = read_jsonl(args.excluded) if args.excluded.exists() else []
    write_report(args.report, pairs, quality, baselines, excluded)
    print(f"Wrote {args.report}")


if __name__ == "__main__":
    main()
