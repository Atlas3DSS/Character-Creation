#!/usr/bin/env python3
"""Probe source-grounded SCOTUS frame labels.

This is a diagnostic bridge between synthetic frame contrasts and causal
generation tests. Inputs are real opinion chunks with strict silver labels from
`build_source_frame_labels.py`; outputs include activation probe directions and
text baselines for each requested frame contrast.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from probe_scotus_style import (  # noqa: E402
    capture_features,
    evaluate_text_baseline,
    load_feature_artifacts,
    markdown_table,
    now_iso,
    select_probe,
    write_json,
    write_jsonl,
)


DEFAULT_MODEL = Path("/home/orwel/dev_genius/models/Qwen3.5-27B")
DEFAULT_LABELS = REPO_ROOT / "data" / "scotus" / "scotus_source_frame_labels_v1.jsonl"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "sweep_v4"
DEFAULT_LAYERS = "8,12,16"
DEFAULT_C_GRID = "0.003,0.01,0.03,0.1,0.3,1.0"
DEFAULT_TASKS = (
    "article3_article1_vs_case=article3_article1_tribunal:article3_case_or_controversy,"
    "article3_finality_vs_case=article3_final_judgment_separation:article3_case_or_controversy,"
    "fourth_technology_vs_incident=fourth_technology_privacy:fourth_search_incident_chimel,"
    "fourth_plain_view_vs_incident=fourth_plain_view_independent_source:fourth_search_incident_chimel,"
    "fourth_home_vs_incident=fourth_home_exigency:fourth_search_incident_chimel"
)


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


def parse_tasks(raw: str) -> dict[str, tuple[str, str]]:
    tasks: dict[str, tuple[str, str]] = {}
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        name, frames = part.split("=", 1)
        positive, negative = frames.split(":", 1)
        tasks[name.strip()] = (positive.strip(), negative.strip())
    if not tasks:
        raise ValueError("No tasks were specified")
    return tasks


def task_label_counts(rows: list[dict[str, Any]]) -> list[list[Any]]:
    counts = Counter((row["frame_task"], row["split"], row["frame_label"]) for row in rows)
    return [[task, split, label, count] for (task, split, label), count in sorted(counts.items())]


def stable_int(value: str) -> int:
    return int(hashlib.sha256(value.encode("utf-8")).hexdigest()[:8], 16)


def should_skip_conflict(source: dict[str, Any], positive_frame: str, negative_frame: str, exclude_conflicts: bool) -> bool:
    if not exclude_conflicts:
        return False
    matched_frames = source.get("matched_frames")
    if not isinstance(matched_frames, list):
        return False
    frame_set = {str(frame) for frame in matched_frames}
    return positive_frame in frame_set and negative_frame in frame_set


def reassign_task_splits(examples: list[dict[str, Any]]) -> None:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for example in examples:
        grouped.setdefault(str(example["frame_task"]), []).append(example)

    for task_name, rows in grouped.items():
        split_by_cluster = stratified_group_split(rows, task_name)
        for row in rows:
            cluster = str(row.get("source_case_id") or row.get("pair_id") or row.get("chunk_id"))
            row["source_split"] = row["split"]
            row["split"] = split_by_cluster[cluster]


def stratified_group_split(rows: list[dict[str, Any]], task_name: str) -> dict[str, str]:
    """Assign one held-out split per source cluster while roughly balancing labels."""
    cluster_label_counts: dict[str, Counter[int]] = defaultdict(Counter)
    for row in rows:
        cluster = str(row.get("source_case_id") or row.get("pair_id") or row.get("chunk_id"))
        cluster_label_counts[cluster][int(row["label"])] += 1

    labels = sorted({int(row["label"]) for row in rows})
    if len(labels) < 2:
        clusters = sorted(cluster_label_counts, key=lambda value: stable_int(f"{task_name}:{value}"))
        return {cluster: "train" for cluster in clusters}

    label_totals: Counter[int] = Counter()
    for counts in cluster_label_counts.values():
        label_totals.update(counts)
    total_rows = sum(label_totals.values())
    split_fracs = {"train": 0.70, "dev": 0.15, "test": 0.15}
    label_targets = {
        split: {label: max(1.0, label_totals[label] * frac) for label in labels}
        for split, frac in split_fracs.items()
    }
    total_targets = {split: max(1.0, total_rows * frac) for split, frac in split_fracs.items()}

    split_by_cluster: dict[str, str] = {}
    split_label_counts: dict[str, Counter[int]] = {split: Counter() for split in split_fracs}
    split_totals: Counter[str] = Counter()

    def cluster_size(cluster: str) -> int:
        return sum(cluster_label_counts[cluster].values())

    def assign(cluster: str, split: str) -> None:
        split_by_cluster[cluster] = split
        split_label_counts[split].update(cluster_label_counts[cluster])
        split_totals[split] += cluster_size(cluster)

    # Seed dev/test with label-bearing clusters before greedily filling targets.
    for split in ("dev", "test"):
        for label in labels:
            candidates = [
                cluster
                for cluster, counts in cluster_label_counts.items()
                if cluster not in split_by_cluster and counts[label] > 0
            ]
            if not candidates:
                continue
            candidates.sort(
                key=lambda cluster: (
                    cluster_label_counts[cluster][label]
                    - 0.35 * (cluster_size(cluster) - cluster_label_counts[cluster][label]),
                    -abs(cluster_size(cluster) - label_targets[split][label]),
                    -stable_int(f"{task_name}:{split}:{label}:{cluster}"),
                ),
                reverse=True,
            )
            assign(candidates[0], split)

    remaining = [cluster for cluster in cluster_label_counts if cluster not in split_by_cluster]
    remaining.sort(key=lambda cluster: (cluster_size(cluster), stable_int(f"{task_name}:{cluster}")), reverse=True)
    for cluster in remaining:
        best_split = "train"
        best_score: float | None = None
        for split in ("train", "dev", "test"):
            score = 0.0
            for label in labels:
                after = split_label_counts[split][label] + cluster_label_counts[cluster][label]
                target = label_targets[split][label]
                score += ((after - target) / target) ** 2
            after_total = split_totals[split] + cluster_size(cluster)
            score += 0.4 * ((after_total - total_targets[split]) / total_targets[split]) ** 2
            if split != "train" and split_totals[split] >= total_targets[split] * 1.35:
                score += 10.0
            if best_score is None or score < best_score:
                best_score = score
                best_split = split
        assign(cluster, best_split)

    return split_by_cluster


def build_examples(
    label_rows: list[dict[str, Any]],
    tasks: dict[str, tuple[str, str]],
    text_field: str,
    *,
    exclude_conflicts: bool,
    reassign_splits: bool,
) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    rows_by_frame: dict[str, list[dict[str, Any]]] = {}
    for row in label_rows:
        rows_by_frame.setdefault(str(row["frame"]), []).append(row)

    for task_name, (positive_frame, negative_frame) in tasks.items():
        for frame, label in ((negative_frame, 0), (positive_frame, 1)):
            frame_rows = rows_by_frame.get(frame, [])
            if not frame_rows:
                raise ValueError(f"Task {task_name} references frame with no rows: {frame}")
            for source in frame_rows:
                if should_skip_conflict(source, positive_frame, negative_frame, exclude_conflicts):
                    continue
                text = str(source.get(text_field) or source.get("text") or "")
                if not text.strip():
                    continue
                example_id = f"{task_name}|{source['record_id']}"
                examples.append(
                    {
                        "example_id": example_id,
                        "chunk_id": source["record_id"],
                        "pair_id": f"{task_name}|{source.get('cluster_id')}",
                        "split": source["split"],
                        "label": int(label),
                        "justice": positive_frame if label else negative_frame,
                        "positive_justice": positive_frame,
                        "frame_task": task_name,
                        "frame_label": frame,
                        "positive_frame": positive_frame,
                        "negative_frame": negative_frame,
                        "issue_area_label": source.get("issue_area_label", ""),
                        "opinion_type": "source_frame",
                        "section_posture": source.get("section_posture", ""),
                        "case_name": source.get("case_name", ""),
                        "justice_author": source.get("justice", ""),
                        "source_url": source.get("source_url", ""),
                        "source_chunk_id": source.get("chunk_id", ""),
                        "source_case_id": source.get("source_case_id", ""),
                        "has_public_private_conflict": bool(source.get("has_public_private_conflict", False)),
                        "text": text,
                    }
                )
    if reassign_splits:
        reassign_task_splits(examples)
    examples.sort(key=lambda row: (row["frame_task"], row["split"], row["label"], row["chunk_id"]))
    return examples


def subset_extracted(extracted: dict[str, Any], task_name: str) -> dict[str, Any]:
    meta_rows = extracted["meta_rows"]
    indices = [idx for idx, row in enumerate(meta_rows) if row["frame_task"] == task_name]
    if not indices:
        raise RuntimeError(f"No rows for task {task_name}")
    idx_arr = np.array(indices, dtype=np.int64)
    return {
        "regions": {
            region: {layer: arr[idx_arr] for layer, arr in layer_map.items()}
            for region, layer_map in extracted["regions"].items()
        },
        "meta_rows": [meta_rows[idx] for idx in indices],
        "labels": extracted["labels"][idx_arr],
        "layers": extracted["layers"],
    }


def save_raw_direction(path: Path, *, clf: Any, best: dict[str, Any], task_name: str, positive_frame: str) -> None:
    scaler = clf.named_steps["scaler"]
    logreg = clf.named_steps["clf"]
    coef = logreg.coef_.astype(np.float32)
    scale = scaler.scale_.astype(np.float32)
    raw_direction = (coef[0] / np.maximum(scale, 1e-12)).astype(np.float32)
    raw_norm = float(np.linalg.norm(raw_direction))
    raw_unit = raw_direction / max(raw_norm, 1e-12)
    np.savez_compressed(
        path,
        raw_direction_unit=raw_unit.astype(np.float32),
        raw_direction_norm=np.array([raw_norm], dtype=np.float32),
        coef=coef,
        intercept=logreg.intercept_.astype(np.float32),
        scaler_mean=scaler.mean_.astype(np.float32),
        scaler_scale=scale,
        region=np.array([best["region"]]),
        layer=np.array([int(best["layer"])]),
        C=np.array([float(best["C"])], dtype=np.float32),
        task_name=np.array([task_name]),
        positive_frame=np.array([positive_frame]),
    )


def result_rows(task_results: dict[str, dict[str, Any]]) -> list[list[Any]]:
    rows: list[list[Any]] = []
    for task_name, result in sorted(task_results.items()):
        best = result["probe"]["best"]
        split_metrics = result["probe"]["split_metrics"]
        text_baseline = result["text_baseline"]
        rows.append(
            [
                task_name,
                result["positive_frame"],
                result["negative_frame"],
                result["n_train"],
                result["n_dev"],
                result["n_test"],
                best["region"],
                best["layer"],
                f"{best['C']:.4g}",
                f"{best['dev_metrics']['balanced_accuracy']:.3f}",
                f"{split_metrics['test']['balanced_accuracy']:.3f}",
                f"{text_baseline['test']['balanced_accuracy']:.3f}",
                result["direction_path"],
            ]
        )
    return rows


def distribution_rows(task_results: dict[str, dict[str, Any]]) -> list[list[Any]]:
    rows: list[list[Any]] = []
    for task_name, result in sorted(task_results.items()):
        dist = result["probe"]["search_distribution"]
        rows.append(
            [
                task_name,
                dist["n_configs"],
                f"{dist['dev_balanced_accuracy']['median']:.3f}",
                dist["dev_balanced_accuracy"]["configs_above_0_75"],
                f"{dist['test_balanced_accuracy_diagnostic']['median']:.3f}",
                dist["test_balanced_accuracy_diagnostic"]["configs_above_0_75"],
            ]
        )
    return rows


def write_report(
    path: Path,
    *,
    manifest: dict[str, Any],
    examples: list[dict[str, Any]],
    task_results: dict[str, dict[str, Any]],
) -> None:
    top_rows: list[list[Any]] = []
    for task_name, result in sorted(task_results.items()):
        for row in result["probe"]["searches"][:6]:
            top_rows.append(
                [
                    task_name,
                    row["region"],
                    row["layer"],
                    f"{row['C']:.4g}",
                    f"{row['dev_metrics']['balanced_accuracy']:.3f}",
                    f"{row['test_metrics_diagnostic']['balanced_accuracy']:.3f}",
                ]
            )

    lines = [
        "# SCOTUS Source Frame Probe",
        "",
        "## Method Note",
        "",
        f"This is a source-grounded frame probe over real SCOTUS opinion chunks. Labels are loaded from `{Path(manifest['labels']).name}`; this is still diagnostic and should not be cited as a final steering result without manual label review and causal controls.",
        "",
        "## Configuration",
        "",
        markdown_table(
            ["Field", "Value"],
            [
                ["Started", manifest["started_at"]],
                ["Finished", manifest.get("finished_at", "")],
                ["Model", manifest["model_path"]],
                ["Layers", ", ".join(str(layer) for layer in manifest["layers"])],
                ["Rows", len(examples)],
                ["Text field", manifest["text_field"]],
                ["Prompt template", manifest["prompt_template"]],
                ["Use chat template", manifest["use_chat_template"]],
                ["Exclude conflict rows", manifest.get("exclude_conflict_rows", False)],
                ["Reassign task splits", manifest.get("reassign_task_splits", False)],
                ["C grid", ", ".join(str(c) for c in manifest["c_grid"])],
            ],
        ),
        "",
        "## Label Counts",
        "",
        markdown_table(["Task", "Split", "Frame", "N"], task_label_counts(examples)),
        "",
        "## Best Results",
        "",
        markdown_table(
            [
                "Task",
                "Positive",
                "Negative",
                "Train",
                "Dev",
                "Test",
                "Region",
                "Layer",
                "C",
                "Dev BA",
                "Test BA",
                "Text test BA",
                "Direction",
            ],
            result_rows(task_results),
        ),
        "",
        "## Sweep Distribution",
        "",
        markdown_table(
            ["Task", "Configs", "Median dev BA", "Dev >=0.75", "Median diagnostic test BA", "Test >=0.75"],
            distribution_rows(task_results),
        ),
        "",
        "## Top Configs",
        "",
        markdown_table(["Task", "Region", "Layer", "C", "Dev BA", "Diagnostic test BA"], top_rows),
        "",
        "## Interpretation Rules",
        "",
        "1. Treat high text-baseline scores as evidence that frame labels remain lexically visible.",
        "2. Do not promote a source-frame direction unless it has enough dev/test examples, survives manual label review, and later beats prompt-matched random controls in generation.",
        "3. Small frames with one or two dev/test examples are candidate discovery only.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--features-dir", type=Path, default=None)
    parser.add_argument("--tasks", default=DEFAULT_TASKS)
    parser.add_argument("--text-field", default="text")
    parser.add_argument(
        "--exclude-conflict-rows",
        action="store_true",
        help="Skip rows whose matched_frames contain both sides of the requested contrast.",
    )
    parser.add_argument(
        "--reassign-task-splits",
        action="store_true",
        help="Reassign strict source-cluster-held-out splits separately for each task after filtering.",
    )
    parser.add_argument("--layers", default=DEFAULT_LAYERS)
    parser.add_argument("--device-map", default="single")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=768)
    parser.add_argument("--prompt-template", default="plain")
    parser.add_argument("--use-chat-template", action="store_true")
    parser.add_argument("--c-grid", default=DEFAULT_C_GRID)
    parser.add_argument("--classifier-solver", default="lbfgs", choices=["lbfgs", "liblinear", "saga", "sgd"])
    parser.add_argument("--classifier-max-iter", type=int, default=1000)
    parser.add_argument("--classifier-tol", type=float, default=1e-3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tasks = parse_tasks(args.tasks)
    c_grid = [float(part) for part in args.c_grid.split(",") if part.strip()]
    stamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    out_dir = args.features_dir or (args.output_root / f"scotus_source_frame_probe_{stamp}")
    out_dir.mkdir(parents=True, exist_ok=True)

    label_rows = read_jsonl(args.labels)
    examples = build_examples(
        label_rows,
        tasks,
        text_field=args.text_field,
        exclude_conflicts=args.exclude_conflict_rows,
        reassign_splits=args.reassign_task_splits,
    )
    write_jsonl(out_dir / "probe_examples.jsonl", examples)

    manifest = {
        "started_at": now_iso(),
        "model_path": str(args.model_path),
        "output_dir": str(out_dir),
        "labels": str(args.labels),
        "tasks": tasks,
        "text_field": args.text_field,
        "layers_spec": args.layers,
        "device_map": args.device_map,
        "batch_size": args.batch_size,
        "max_length": args.max_length,
        "prompt_template": args.prompt_template,
        "use_chat_template": bool(args.use_chat_template),
        "exclude_conflict_rows": bool(args.exclude_conflict_rows),
        "reassign_task_splits": bool(args.reassign_task_splits),
        "c_grid": c_grid,
        "classifier": {
            "solver": args.classifier_solver,
            "max_iter": args.classifier_max_iter,
            "tol": args.classifier_tol,
        },
    }
    write_json(out_dir / "manifest.json", manifest)

    if args.features_dir is not None:
        extracted = load_feature_artifacts(out_dir)
    else:
        extracted = capture_features(
            examples,
            model_path=args.model_path,
            device_map=args.device_map,
            layers_spec=args.layers,
            batch_size=args.batch_size,
            max_length=args.max_length,
            template_variant=args.prompt_template,
            use_chat_template=args.use_chat_template,
            out_dir=out_dir,
        )
    manifest["layers"] = extracted["layers"]

    task_results: dict[str, dict[str, Any]] = {}
    for task_name, (positive_frame, negative_frame) in tasks.items():
        task_out = out_dir / task_name
        task_out.mkdir(parents=True, exist_ok=True)
        task_extracted = subset_extracted(extracted, task_name)
        task_examples = [row for row in examples if row["frame_task"] == task_name]
        text_baseline = evaluate_text_baseline(task_examples, template_variant=args.prompt_template)
        probe = select_probe(
            task_extracted["regions"],
            task_extracted["meta_rows"],
            task_extracted["labels"],
            c_grid,
            classifier_solver=args.classifier_solver,
            classifier_max_iter=args.classifier_max_iter,
            classifier_tol=args.classifier_tol,
            test_diagnostic_refit=False,
        )
        for split, rows in probe["predictions"].items():
            write_jsonl(task_out / f"{split}_predictions.jsonl", rows)
        write_jsonl(task_out / "searches.jsonl", probe["searches"])
        write_json(task_out / "text_baseline.json", text_baseline)
        direction_path = task_out / "direction.npz"
        save_raw_direction(
            direction_path,
            clf=probe["final_clf"],
            best=probe["best"],
            task_name=task_name,
            positive_frame=positive_frame,
        )
        split_counts = Counter((row["split"], row["label"]) for row in task_examples)
        task_results[task_name] = {
            "probe": probe,
            "text_baseline": text_baseline,
            "direction_path": str(direction_path),
            "positive_frame": positive_frame,
            "negative_frame": negative_frame,
            "n_train": split_counts[("train", 0)] + split_counts[("train", 1)],
            "n_dev": split_counts[("dev", 0)] + split_counts[("dev", 1)],
            "n_test": split_counts[("test", 0)] + split_counts[("test", 1)],
        }

    manifest["finished_at"] = now_iso()
    write_json(out_dir / "manifest.json", manifest)
    write_report(out_dir / "report.md", manifest=manifest, examples=examples, task_results=task_results)
    gc.collect()
    print(f"Wrote {out_dir / 'report.md'}", flush=True)


if __name__ == "__main__":
    main()
