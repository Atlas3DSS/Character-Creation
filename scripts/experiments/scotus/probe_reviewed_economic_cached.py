#!/usr/bin/env python3
"""Probe reviewed Economic Activity dominance labels using cached features."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SCOTUS_DIR = PROJECT_ROOT / "data" / "scotus"
DEFAULT_REVIEWED = SCOTUS_DIR / "scotus_economic_pocket_dominance_adjudicated_20260501.jsonl"
DEFAULT_SOURCE_RUN = PROJECT_ROOT / "sweep_v4" / "scotus_source_frame_probe_20260501_014711"
DEFAULT_OUTPUT = PROJECT_ROOT / "sweep_v4" / "scotus_economic_reviewed_broad_limits_cached_20260501"

if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from probe_scotus_style import (  # noqa: E402
    evaluate_text_baseline,
    load_feature_artifacts,
    markdown_table,
    now_iso,
    select_probe,
    subset_feature_artifacts,
    write_json,
    write_jsonl,
)
from probe_scotus_source_frames import save_raw_direction  # noqa: E402


BROAD = "dominant_broad_commerce"
LIMITS = "dominant_commerce_limits"
LABEL_BY_REVIEW = {LIMITS: 0, BROAD: 1}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON") from exc
    return rows


def parse_c_grid(raw: str) -> list[float]:
    values = [float(part) for part in raw.split(",") if part.strip()]
    if not values:
        raise ValueError("C grid cannot be empty")
    return values


def reviewed_label(row: dict[str, Any]) -> int | None:
    if row.get("review_confidence") not in {"medium", "high"}:
        return None
    return LABEL_BY_REVIEW.get(str(row.get("dominant_frame_label")))


def selected_indices(
    examples: list[dict[str, Any]],
    meta_rows: list[dict[str, Any]],
    reviewed: dict[str, dict[str, Any]],
) -> tuple[list[int], list[dict[str, Any]], np.ndarray]:
    if len(examples) != len(meta_rows):
        raise RuntimeError(f"example/meta length mismatch: {len(examples)} versus {len(meta_rows)}")
    indices: list[int] = []
    selected_examples: list[dict[str, Any]] = []
    labels: list[int] = []
    for idx, (example, meta) in enumerate(zip(examples, meta_rows, strict=True)):
        if example["example_id"] != meta["example_id"]:
            raise RuntimeError(f"example/meta order mismatch at {idx}: {example['example_id']} != {meta['example_id']}")
        if example.get("frame_task") != "economic_broad_vs_limits":
            continue
        review_row = reviewed.get(str(example.get("chunk_id")))
        if review_row is None:
            continue
        label = reviewed_label(review_row)
        if label is None:
            continue
        updated = dict(example)
        updated["label"] = label
        updated["justice"] = "reviewed_broad_commerce" if label == 1 else "reviewed_commerce_limits"
        updated["frame_label"] = review_row["dominant_frame_label"]
        updated["review_confidence"] = review_row["review_confidence"]
        updated["review_notes"] = review_row["review_notes"]
        selected_examples.append(updated)
        indices.append(idx)
        labels.append(label)
    return indices, selected_examples, np.array(labels, dtype=np.int64)


def apply_labels_to_meta(meta_rows: list[dict[str, Any]], reviewed_examples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    updated_rows: list[dict[str, Any]] = []
    for meta, example in zip(meta_rows, reviewed_examples, strict=True):
        row = dict(meta)
        row["label"] = int(example["label"])
        row["justice"] = example["justice"]
        row["frame_label"] = example["frame_label"]
        row["review_confidence"] = example["review_confidence"]
        row["review_notes"] = example["review_notes"]
        updated_rows.append(row)
    return updated_rows


def result_table(probe: dict[str, Any], text_baseline: dict[str, Any]) -> list[list[Any]]:
    best = probe["best"]
    split_metrics = probe["split_metrics"]
    text_test = text_baseline.get("test", {}).get("balanced_accuracy")
    return [
        ["Best readout", f"{best['region']} @ L{best['layer']}"],
        ["Best C", best["C"]],
        ["Dev balanced accuracy", f"{best['dev_metrics']['balanced_accuracy']:.3f}"],
        ["Test balanced accuracy", f"{split_metrics['test']['balanced_accuracy']:.3f}"],
        ["Text test balanced accuracy", f"{text_test:.3f}" if text_test is not None else text_baseline.get("error", "")],
    ]


def write_report(
    path: Path,
    *,
    args: argparse.Namespace,
    examples: list[dict[str, Any]],
    probe: dict[str, Any],
    text_baseline: dict[str, Any],
) -> None:
    counts = Counter((row["split"], row["label"]) for row in examples)
    case_counts = Counter((row["split"], row["label"], row["source_case_id"], row["case_name"]) for row in examples)
    top_rows = [
        [
            row["region"],
            row["layer"],
            row["C"],
            f"{row['dev_metrics']['balanced_accuracy']:.3f}",
            f"{row['test_metrics_diagnostic']['balanced_accuracy']:.3f}",
        ]
        for row in probe["searches"][:12]
    ]
    test_ba = probe["split_metrics"]["test"]["balanced_accuracy"]
    text_test = text_baseline.get("test", {}).get("balanced_accuracy", 0.0)
    lines = [
        "# SCOTUS Economic Reviewed Broad-vs-Limits Cached Probe",
        "",
        "## Purpose",
        "",
        "This reruns the Economic Activity broad-Commerce versus Commerce-limits probe on the cached Qwen3.5 BF16 feature matrix, replacing the original regex source labels with the internal dominance-review labels.",
        "",
        "## Inputs",
        "",
        markdown_table(
            ["Field", "Value"],
            [
                ["Reviewed labels", args.reviewed],
                ["Source feature run", args.source_run],
                ["Rows", len(examples)],
                ["C grid", args.c_grid],
                ["Positive label", "dominant_broad_commerce"],
                ["Negative label", "dominant_commerce_limits"],
            ],
        ),
        "",
        "## Label Counts",
        "",
        markdown_table(["Split", "Label", "N"], [[split, label, count] for (split, label), count in sorted(counts.items())]),
        "",
        "## Case Coverage",
        "",
        markdown_table(
            ["Split", "Label", "Case id", "Case", "N"],
            [[split, label, case_id, case_name, count] for (split, label, case_id, case_name), count in sorted(case_counts.items())],
        ),
        "",
        "## Result",
        "",
        markdown_table(["Metric", "Value"], result_table(probe, text_baseline)),
        "",
        "## Top Cached Configs",
        "",
        markdown_table(["Region", "Layer", "C", "Dev BA", "Diagnostic test BA"], top_rows),
        "",
        "## Decision",
        "",
    ]
    if test_ba >= text_test + 0.05:
        lines.append("The reviewed-label activation result beats the cue-masked text baseline by at least 0.05 test BA. This can advance to a narrow causal pocket pilot with prompt-matched random controls.")
    else:
        lines.append("The reviewed-label activation result does not beat the cue-masked text baseline by at least 0.05 test BA. Do not run a causal pocket pilot from this reviewed source direction.")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reviewed", type=Path, default=DEFAULT_REVIEWED)
    parser.add_argument("--source-run", type=Path, default=DEFAULT_SOURCE_RUN)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--c-grid", default="0.001,0.003,0.01,0.03,0.1,0.3,1.0")
    parser.add_argument("--classifier-max-iter", type=int, default=2000)
    parser.add_argument("--classifier-tol", type=float, default=1e-3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    reviewed_rows = {str(row["record_id"]): row for row in read_jsonl(args.reviewed)}
    source_examples = read_jsonl(args.source_run / "probe_examples.jsonl")
    extracted = load_feature_artifacts(args.source_run)
    indices, examples, labels = selected_indices(source_examples, extracted["meta_rows"], reviewed_rows)
    if not len(examples):
        raise RuntimeError("No reviewed examples selected")
    subset = subset_feature_artifacts(extracted, indices)
    subset["labels"] = labels
    subset["meta_rows"] = apply_labels_to_meta(subset["meta_rows"], examples)
    c_grid = parse_c_grid(args.c_grid)

    text_baseline = evaluate_text_baseline(examples, template_variant="plain", c_value=1.0)
    probe = select_probe(
        subset["regions"],
        subset["meta_rows"],
        subset["labels"],
        c_grid,
        classifier_solver="lbfgs",
        classifier_max_iter=args.classifier_max_iter,
        classifier_tol=args.classifier_tol,
        test_diagnostic_refit=False,
    )

    write_jsonl(args.output_dir / "probe_examples.jsonl", examples)
    write_jsonl(args.output_dir / "feature_meta.jsonl", subset["meta_rows"])
    for split, rows in probe["predictions"].items():
        write_jsonl(args.output_dir / f"{split}_predictions.jsonl", rows)
    write_jsonl(args.output_dir / "searches.jsonl", probe["searches"])
    write_json(args.output_dir / "text_baseline.json", text_baseline)
    manifest = {
        "started_at": now_iso(),
        "finished_at": now_iso(),
        "reviewed": str(args.reviewed),
        "source_run": str(args.source_run),
        "output_dir": str(args.output_dir),
        "rows": len(examples),
        "c_grid": c_grid,
        "layers": subset["layers"],
        "positive_label": BROAD,
        "negative_label": LIMITS,
    }
    write_json(args.output_dir / "manifest.json", manifest)
    save_raw_direction(
        args.output_dir / "direction.npz",
        clf=probe["final_clf"],
        best=probe["best"],
        task_name="economic_reviewed_broad_vs_limits",
        positive_frame=BROAD,
    )
    write_report(args.output_dir / "report.md", args=args, examples=examples, probe=probe, text_baseline=text_baseline)
    print(f"Wrote {args.output_dir / 'report.md'}")


if __name__ == "__main__":
    main()
