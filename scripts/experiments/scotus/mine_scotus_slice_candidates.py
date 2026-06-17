#!/usr/bin/env python3
"""Mine cached SCOTUS activation features for less text-dominated justice slices."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RUN = PROJECT_ROOT / "sweep_v4" / "scotus_phase41_normal_20260425_102519"
DEFAULT_OUT = PROJECT_ROOT / "reports" / "scotus_slice_candidate_mining_20260501.md"
DEFAULT_JSON = PROJECT_ROOT / "reports" / "scotus_slice_candidate_mining_20260501.json"

if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from probe_scotus_style import (  # noqa: E402
    evaluate_text_baseline,
    markdown_table,
    select_probe,
)


@dataclass(frozen=True)
class SliceSpec:
    name: str
    fields: tuple[tuple[str, str], ...]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON") from exc
    return rows


def split_layer_key(key: str) -> tuple[str, int] | None:
    if "__L" not in key:
        return None
    region, layer_raw = key.rsplit("__L", 1)
    try:
        return region, int(layer_raw)
    except ValueError:
        return None


def load_regions(features_path: Path, *, layers: set[int], regions: set[str]) -> dict[str, dict[int, np.ndarray]]:
    arrays = np.load(features_path)
    loaded: dict[str, dict[int, np.ndarray]] = {}
    for key in arrays.files:
        parsed = split_layer_key(key)
        if parsed is None:
            continue
        region, layer = parsed
        if region not in regions or layer not in layers:
            continue
        loaded.setdefault(region, {})[layer] = arrays[key]
    if not loaded:
        raise RuntimeError(f"No requested feature arrays found in {features_path}")
    return loaded


def merge_rows(run_dir: Path) -> list[dict[str, Any]]:
    examples = read_jsonl(run_dir / "probe_examples.jsonl")
    feature_meta = read_jsonl(run_dir / "feature_meta.jsonl")
    by_id = {str(row["example_id"]): row for row in examples}
    merged: list[dict[str, Any]] = []
    for meta in feature_meta:
        example = by_id.get(str(meta["example_id"]))
        if example is None:
            raise RuntimeError(f"Missing example row for {meta['example_id']}")
        row = dict(meta)
        row["text"] = example["text"]
        merged.append(row)
    return merged


def label_split_counts(rows: list[dict[str, Any]]) -> Counter[tuple[str, int]]:
    return Counter((str(row["split"]), int(row["label"])) for row in rows)


def slice_ok(rows: list[dict[str, Any]], *, min_train_per_label: int, min_dev_per_label: int, min_test_per_label: int) -> bool:
    counts = label_split_counts(rows)
    return (
        counts[("train", 0)] >= min_train_per_label
        and counts[("train", 1)] >= min_train_per_label
        and counts[("dev", 0)] >= min_dev_per_label
        and counts[("dev", 1)] >= min_dev_per_label
        and counts[("test", 0)] >= min_test_per_label
        and counts[("test", 1)] >= min_test_per_label
    )


def candidate_slices(
    rows: list[dict[str, Any]],
    *,
    min_train_per_label: int,
    min_dev_per_label: int,
    min_test_per_label: int,
) -> list[SliceSpec]:
    specs: list[SliceSpec] = [SliceSpec("all", ())]
    single_fields = [
        "issue_area_label",
        "section_posture",
        "decade",
        "decision_direction",
        "chunk_position_bucket",
    ]
    for field in single_fields:
        values = sorted({str(row.get(field) or "unknown") for row in rows})
        for value in values:
            if value == "unknown":
                continue
            subset = [row for row in rows if str(row.get(field) or "unknown") == value]
            if slice_ok(
                subset,
                min_train_per_label=min_train_per_label,
                min_dev_per_label=min_dev_per_label,
                min_test_per_label=min_test_per_label,
            ):
                specs.append(SliceSpec(f"{field}={value}", ((field, value),)))

    combo_fields = [
        ("issue_area_label", "section_posture"),
        ("issue_area_label", "decade"),
        ("section_posture", "decade"),
    ]
    for left, right in combo_fields:
        values = sorted({(str(row.get(left) or "unknown"), str(row.get(right) or "unknown")) for row in rows})
        for left_value, right_value in values:
            if "unknown" in (left_value, right_value):
                continue
            subset = [
                row
                for row in rows
                if str(row.get(left) or "unknown") == left_value and str(row.get(right) or "unknown") == right_value
            ]
            if slice_ok(
                subset,
                min_train_per_label=min_train_per_label,
                min_dev_per_label=min_dev_per_label,
                min_test_per_label=min_test_per_label,
            ):
                specs.append(SliceSpec(f"{left}={left_value}__{right}={right_value}", ((left, left_value), (right, right_value))))

    return specs


def apply_slice(rows: list[dict[str, Any]], spec: SliceSpec) -> tuple[list[int], list[dict[str, Any]]]:
    indices: list[int] = []
    subset: list[dict[str, Any]] = []
    for idx, row in enumerate(rows):
        if all(str(row.get(field) or "unknown") == value for field, value in spec.fields):
            indices.append(idx)
            subset.append(row)
    return indices, subset


def subset_regions(regions: dict[str, dict[int, np.ndarray]], indices: list[int]) -> dict[str, dict[int, np.ndarray]]:
    idx_arr = np.array(indices, dtype=np.int64)
    return {region: {layer: arr[idx_arr] for layer, arr in layer_map.items()} for region, layer_map in regions.items()}


def metric_value(payload: dict[str, Any], split: str, metric: str) -> float | None:
    try:
        value = payload[split][metric]
    except KeyError:
        return None
    return float(value)


def append_markdown_table(lines: list[str], headers: list[str], rows: list[list[Any]]) -> None:
    lines.append(markdown_table(headers, rows))


def run_mining(args: argparse.Namespace) -> list[dict[str, Any]]:
    rows = merge_rows(args.run_dir)
    layers = {int(part) for part in args.layers.split(",") if part.strip()}
    region_names = {part.strip() for part in args.regions.split(",") if part.strip()}
    regions = load_regions(args.run_dir / "features.npz", layers=layers, regions=region_names)
    specs = candidate_slices(
        rows,
        min_train_per_label=args.min_train_per_label,
        min_dev_per_label=args.min_dev_per_label,
        min_test_per_label=args.min_test_per_label,
    )

    results: list[dict[str, Any]] = []
    for spec in tqdm(specs, desc="mine slices"):
        indices, subset = apply_slice(rows, spec)
        counts = label_split_counts(subset)
        text_baseline = evaluate_text_baseline(subset, template_variant=args.prompt_template)
        if "error" in text_baseline:
            continue
        labels = np.array([int(row["label"]) for row in subset], dtype=np.int64)
        probe = select_probe(
            subset_regions(regions, indices),
            subset,
            labels,
            [float(part) for part in args.c_grid.split(",") if part.strip()],
            classifier_solver=args.classifier_solver,
            classifier_max_iter=args.classifier_max_iter,
            classifier_tol=args.classifier_tol,
            test_diagnostic_refit=False,
        )
        test_ba = float(probe["split_metrics"]["test"]["balanced_accuracy"])
        dev_ba = float(probe["split_metrics"]["dev"]["balanced_accuracy"])
        text_test_ba = metric_value(text_baseline, "test", "balanced_accuracy")
        text_dev_ba = metric_value(text_baseline, "dev", "balanced_accuracy")
        gap = None if text_test_ba is None else test_ba - text_test_ba
        best = probe["best"]
        results.append(
            {
                "slice": spec.name,
                "fields": list(spec.fields),
                "n": len(subset),
                "counts": {f"{split}/{label}": count for (split, label), count in sorted(counts.items())},
                "activation_dev_ba": dev_ba,
                "activation_test_ba": test_ba,
                "text_dev_ba": text_dev_ba,
                "text_test_ba": text_test_ba,
                "activation_minus_text_test_ba": gap,
                "best_region": best["region"],
                "best_layer": int(best["layer"]),
                "best_C": float(best["C"]),
                "ci95": probe["test_balanced_accuracy_ci_95"],
            }
        )
    results.sort(
        key=lambda row: (
            row["activation_minus_text_test_ba"] if row["activation_minus_text_test_ba"] is not None else -999.0,
            row["activation_test_ba"],
            row["activation_dev_ba"],
        ),
        reverse=True,
    )
    return results


def write_report(path: Path, *, results: list[dict[str, Any]], args: argparse.Namespace) -> None:
    top_rows = []
    for row in results[:30]:
        top_rows.append(
            [
                row["slice"],
                row["n"],
                f"{row['activation_dev_ba']:.3f}",
                f"{row['activation_test_ba']:.3f}",
                f"{row['text_test_ba']:.3f}" if row["text_test_ba"] is not None else "n/a",
                f"{row['activation_minus_text_test_ba']:.3f}" if row["activation_minus_text_test_ba"] is not None else "n/a",
                f"{row['best_region']} @ L{row['best_layer']}",
                row["best_C"],
            ]
        )
    candidates = [
        row
        for row in results
        if row["activation_test_ba"] >= args.min_promote_test_ba
        and row["activation_dev_ba"] >= args.min_promote_dev_ba
        and (row["activation_minus_text_test_ba"] or -999.0) >= args.min_promote_gap
    ]
    candidate_rows = [
        [
            row["slice"],
            row["n"],
            f"{row['activation_dev_ba']:.3f}",
            f"{row['activation_test_ba']:.3f}",
            f"{row['text_test_ba']:.3f}" if row["text_test_ba"] is not None else "n/a",
            f"{row['activation_minus_text_test_ba']:.3f}" if row["activation_minus_text_test_ba"] is not None else "n/a",
            f"{row['best_region']} @ L{row['best_layer']}",
        ]
        for row in candidates
    ]

    lines = [
        "# SCOTUS Slice Candidate Mining",
        "",
        "## Purpose",
        "",
        "This reuses cached Phase 4.1 Qwen3.6-27B FP8 activation features to search for justice-style slices where hidden-state probes beat a matched cue-masked text baseline. It does not load the model and does not establish steering by itself.",
        "",
        "## Inputs",
        "",
        f"- Run directory: `{args.run_dir}`",
        f"- Layers: `{args.layers}`",
        f"- Regions: `{args.regions}`",
        f"- C grid: `{args.c_grid}`",
        "",
        "## Promotion Rule",
        "",
        f"A mined slice is only a candidate if activation dev BA >= `{args.min_promote_dev_ba}`, activation test BA >= `{args.min_promote_test_ba}`, and activation-minus-text test BA >= `{args.min_promote_gap}`.",
        "",
        "## Candidate Rows",
        "",
    ]
    if candidate_rows:
        append_markdown_table(lines, ["Slice", "N", "Dev BA", "Test BA", "Text test BA", "Gap", "Best readout"], candidate_rows)
    else:
        lines.append("No slice cleared the promotion rule.")
    lines.extend(["", "## Top Rows By Activation Minus Text", ""])
    append_markdown_table(lines, ["Slice", "N", "Dev BA", "Test BA", "Text test BA", "Gap", "Best readout", "C"], top_rows)
    lines.extend(
        [
            "",
            "## Use Rules",
            "",
            "1. Treat this as a cheap triage pass only; the cached run used Qwen3.6 FP8, not the Qwen3.5 BF16 source of record.",
            "2. Do not run steering from a mined slice unless it survives BF16 recapture or an existing BF16 feature equivalent.",
            "3. Slices with high text baselines are leakage diagnostics, not candidates.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--json-output", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--layers", default="4,9,16,19,40")
    parser.add_argument("--regions", default="prompt_last,prompt_mean,excerpt_mean")
    parser.add_argument("--c-grid", default="0.003,0.01,0.03,0.1")
    parser.add_argument("--prompt-template", default="normal")
    parser.add_argument("--classifier-solver", default="lbfgs", choices=["lbfgs", "liblinear", "saga", "sgd"])
    parser.add_argument("--classifier-max-iter", type=int, default=1000)
    parser.add_argument("--classifier-tol", type=float, default=1e-3)
    parser.add_argument("--min-train-per-label", type=int, default=20)
    parser.add_argument("--min-dev-per-label", type=int, default=5)
    parser.add_argument("--min-test-per-label", type=int, default=5)
    parser.add_argument("--min-promote-dev-ba", type=float, default=0.75)
    parser.add_argument("--min-promote-test-ba", type=float, default=0.75)
    parser.add_argument("--min-promote-gap", type=float, default=0.08)
    parser.add_argument(
        "--report-from-json",
        action="store_true",
        help="Regenerate only the Markdown report from --json-output without rerunning probe sweeps.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.report_from_json:
        results = json.loads(args.json_output.read_text(encoding="utf-8"))
        write_report(args.output, results=results, args=args)
        print(f"Wrote {args.output}")
        return
    results = run_mining(args)
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(results, indent=2, sort_keys=True), encoding="utf-8")
    write_report(args.output, results=results, args=args)
    print(f"Wrote {args.output}")
    print(f"Wrote {args.json_output}")


if __name__ == "__main__":
    main()
