#!/usr/bin/env python3
"""Inspect top-activating SCOTUS chunks for selected SAE features."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

import scipy.sparse as sp


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SAE_RUN_DIR = PROJECT_ROOT / "sweep_v4" / "scotus_sae_probe_20260430_135117"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return "\n".join(lines)


def clean_snippet(text: str, max_chars: int) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= max_chars:
        return text
    cut = text[: max_chars + 1]
    if " " in cut:
        cut = cut[: cut.rfind(" ")]
    return cut.rstrip(" ,.;:") + "..."


def counter_summary(values: list[str], limit: int = 5) -> str:
    counts = Counter(values)
    return ", ".join(f"{key}: {value}" for key, value in counts.most_common(limit))


def rooted_path(raw_path: str | Path) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def feature_cache_path(run_manifest: dict[str, Any], best: dict[str, Any]) -> Path:
    cache_dir = rooted_path(run_manifest["feature_cache_dir"])
    return cache_dir / f"{best['sae_name']}__{best['region']}__L{int(best['layer']):02d}.npz"


def load_rows_for_matrix(manifest: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    probe_dir = rooted_path(manifest["probe_dir"])
    meta_rows = read_jsonl(probe_dir / "feature_meta.jsonl")
    example_path = probe_dir / "probe_examples.jsonl"
    if not example_path.exists():
        example_path = probe_dir / "examples.jsonl"
    examples = read_jsonl(example_path)
    text_by_id = {row["example_id"]: row for row in examples}
    if len(meta_rows) != len(examples):
        raise RuntimeError(f"Row count mismatch: {len(meta_rows)} meta rows vs {len(examples)} examples")
    missing = [row["example_id"] for row in meta_rows if row["example_id"] not in text_by_id]
    if missing:
        raise RuntimeError(f"Missing text for {len(missing)} meta rows, first={missing[0]}")
    return meta_rows, text_by_id


def infer_label_names(meta_rows: list[dict[str, Any]]) -> dict[int, str]:
    grouped: dict[int, list[str]] = {0: [], 1: []}
    for row in meta_rows:
        label = int(row["label"])
        if label in grouped:
            grouped[label].append(str(row.get("justice", f"label_{label}")))
    names: dict[int, str] = {}
    for label, values in grouped.items():
        names[label] = Counter(values).most_common(1)[0][0] if values else f"label_{label}"
    return names


def direction_label(feature: dict[str, Any], label_names: dict[int, str]) -> str:
    target_label = 1 if float(feature.get("weight", 0.0)) >= 0 else 0
    return f"toward_{label_names.get(target_label, f'label_{target_label}')}"


def inspect_feature(
    *,
    feature: dict[str, Any],
    matrix: sp.csr_matrix,
    meta_rows: list[dict[str, Any]],
    text_by_id: dict[str, dict[str, Any]],
    top_examples: int,
    snippet_chars: int,
    label_names: dict[int, str],
) -> dict[str, Any]:
    feature_id = int(feature["feature"])
    col = matrix.getcol(feature_id).tocoo()
    activations = sorted(zip(col.row.tolist(), col.data.tolist(), strict=True), key=lambda pair: pair[1], reverse=True)
    active_rows = [row_idx for row_idx, _value in activations]
    active_meta = [meta_rows[row_idx] for row_idx in active_rows]
    top_rows: list[dict[str, Any]] = []
    for row_idx, value in activations[:top_examples]:
        meta = meta_rows[row_idx]
        source = text_by_id[meta["example_id"]]
        top_rows.append(
            {
                "feature": feature_id,
                "activation": float(value),
                "row_index": int(row_idx),
                "split": meta["split"],
                "justice": meta["justice"],
                "label": int(meta["label"]),
                "issue_area_label": meta.get("issue_area_label"),
                "section_posture": meta.get("section_posture"),
                "case_id": meta.get("case_id"),
                "chunk_id": meta.get("chunk_id"),
                "source_url": meta.get("source_url"),
                "token_count": meta.get("token_count"),
                "snippet": clean_snippet(source.get("text", ""), snippet_chars),
            }
        )
    return {
        "feature": feature_id,
        "direction": feature.get("direction"),
        "direction_label": direction_label(feature, label_names),
        "weight": feature.get("weight"),
        "train_df": feature.get("train_df"),
        "df_positive": feature.get("df_positive"),
        "df_negative": feature.get("df_negative"),
        "n_active_all_splits": len(active_rows),
        "justice_counts": dict(Counter(str(row.get("justice")) for row in active_meta).most_common()),
        "split_counts": dict(Counter(str(row.get("split")) for row in active_meta).most_common()),
        "issue_counts": dict(Counter(str(row.get("issue_area_label")) for row in active_meta).most_common()),
        "posture_counts": dict(Counter(str(row.get("section_posture")) for row in active_meta).most_common()),
        "top_examples": top_rows,
    }


def write_report(
    path: Path,
    *,
    manifest: dict[str, Any],
    best: dict[str, Any],
    inspections: list[dict[str, Any]],
) -> None:
    overview_rows = [
        [
            item["feature"],
            item["direction_label"],
            f"{float(item['weight']):.4g}",
            item["train_df"],
            item["n_active_all_splits"],
            ", ".join(f"{k}: {v}" for k, v in item["justice_counts"].items()),
            ", ".join(f"{k}: {v}" for k, v in item["issue_counts"].items()),
        ]
        for item in inspections
    ]
    lines = [
        "# SCOTUS SAE Feature Example Inspection",
        "",
        "## Configuration",
        "",
        markdown_table(
            ["Field", "Value"],
            [
                ["SAE run", manifest["output_dir"]],
                ["SAE", best["sae_name"]],
                ["Region", best["region"]],
                ["Layer", best["layer"]],
                ["Min train DF", manifest.get("min_train_df")],
                ["Source probe dir", manifest["probe_dir"]],
            ],
        ),
        "",
        "## Feature Overview",
        "",
        markdown_table(
            ["Feature", "Direction", "Weight", "Train DF", "Active rows", "Justice counts", "Issue counts"],
            overview_rows,
        ),
        "",
        "## Top Activating Examples",
        "",
    ]
    for item in inspections:
        lines.extend(
            [
                f"### Feature {item['feature']} ({item['direction_label']})",
                "",
                markdown_table(
                    ["Field", "Value"],
                    [
                        ["Raw direction label", item["direction"]],
                        ["Weight", f"{float(item['weight']):.6g}"],
                        ["Train DF", item["train_df"]],
                        ["All active rows", item["n_active_all_splits"]],
                        ["Justice counts", ", ".join(f"{k}: {v}" for k, v in item["justice_counts"].items())],
                        ["Split counts", ", ".join(f"{k}: {v}" for k, v in item["split_counts"].items())],
                        ["Issue counts", ", ".join(f"{k}: {v}" for k, v in item["issue_counts"].items())],
                        ["Posture counts", ", ".join(f"{k}: {v}" for k, v in item["posture_counts"].items())],
                    ],
                ),
                "",
            ]
        )
        example_rows = [
            [
                f"{row['activation']:.6g}",
                row["split"],
                row["justice"],
                row["issue_area_label"],
                row["section_posture"],
                row["case_id"],
                row["snippet"].replace("|", "\\|"),
            ]
            for row in item["top_examples"]
        ]
        lines.extend(
            [
                markdown_table(
                    ["Activation", "Split", "Justice", "Issue", "Posture", "Case", "Snippet"],
                    example_rows,
                ),
                "",
            ]
        )
    lines.extend(
        [
            "## Reading Notes",
            "",
            "- These are prompt-mean SAE activations from the already-rendered Phase 4 prompts, not fresh token-level SAE traces.",
            "- A feature whose top examples cluster by issue area, procedural posture, named statutes, or boilerplate should be treated as an artifact/confound candidate.",
            "- A feature whose top examples recur across issues and postures while tracking legal reasoning style is a better candidate for deeper inspection.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect top activating examples for SAE features.")
    parser.add_argument("--sae-run-dir", type=Path, default=DEFAULT_SAE_RUN_DIR)
    parser.add_argument("--top-features", type=int, default=12)
    parser.add_argument("--top-examples", type=int, default=8)
    parser.add_argument("--snippet-chars", type=int, default=420)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = args.sae_run_dir
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    best = manifest["best"]
    cache_path = feature_cache_path(manifest, best)
    if not cache_path.exists():
        raise FileNotFoundError(cache_path)
    matrix = sp.load_npz(cache_path).tocsr()
    meta_rows, text_by_id = load_rows_for_matrix(manifest)
    label_names = infer_label_names(meta_rows)
    top_features = read_jsonl(run_dir / "top_sae_features.jsonl")[: args.top_features]
    inspections = [
        inspect_feature(
            feature=feature,
            matrix=matrix,
            meta_rows=meta_rows,
            text_by_id=text_by_id,
            top_examples=args.top_examples,
            snippet_chars=args.snippet_chars,
            label_names=label_names,
        )
        for feature in top_features
    ]
    write_jsonl(run_dir / "top_sae_feature_examples.jsonl", inspections)
    write_report(run_dir / "feature_example_inspection.md", manifest=manifest, best=best, inspections=inspections)
    print(f"Wrote {run_dir / 'feature_example_inspection.md'}")
    print(f"Wrote {run_dir / 'top_sae_feature_examples.jsonl'}")


if __name__ == "__main__":
    main()
