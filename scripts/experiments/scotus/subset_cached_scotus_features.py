#!/usr/bin/env python3
"""Create a filtered copy of a cached SCOTUS feature run."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "sweep_v4"


def now_stamp() -> str:
    return datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


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


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def parse_where(raw_values: list[str]) -> list[tuple[str, str]]:
    filters: list[tuple[str, str]] = []
    for raw in raw_values:
        if "=" not in raw:
            raise ValueError(f"Expected field=value filter, got {raw!r}")
        field, value = raw.split("=", 1)
        field = field.strip()
        value = value.strip()
        if not field or not value:
            raise ValueError(f"Expected non-empty field=value filter, got {raw!r}")
        filters.append((field, value))
    return filters


def parse_where_in(raw_values: list[str]) -> list[tuple[str, set[str]]]:
    filters: list[tuple[str, set[str]]] = []
    for raw in raw_values:
        if "=" not in raw:
            raise ValueError(f"Expected field=value1|value2 filter, got {raw!r}")
        field, values_raw = raw.split("=", 1)
        field = field.strip()
        values = {value.strip() for value in values_raw.split("|") if value.strip()}
        if not field or not values:
            raise ValueError(f"Expected non-empty field=value1|value2 filter, got {raw!r}")
        filters.append((field, values))
    return filters


def row_matches(
    row: dict[str, Any],
    *,
    where: list[tuple[str, str]],
    where_in: list[tuple[str, set[str]]],
) -> bool:
    for field, value in where:
        if str(row.get(field) or "unknown") != value:
            return False
    for field, values in where_in:
        if str(row.get(field) or "unknown") not in values:
            return False
    return True


def count_table(rows: list[dict[str, Any]], fields: tuple[str, ...]) -> list[dict[str, Any]]:
    counts = Counter(
        tuple("unknown" if row.get(field) is None else str(row.get(field)) for field in fields)
        for row in rows
    )
    payload: list[dict[str, Any]] = []
    for key, count in sorted(counts.items()):
        payload.append({field: value for field, value in zip(fields, key, strict=True)} | {"n": int(count)})
    return payload


def markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return "\n".join(lines)


def write_report(path: Path, *, manifest: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    issue_rows = [
        [row["issue_area_label"], row["split"], row["justice"], row["n"]]
        for row in count_table(rows, ("issue_area_label", "split", "justice"))
    ]
    label_rows = [[row["split"], row["label"], row["n"]] for row in count_table(rows, ("split", "label"))]
    lines = [
        "# Cached SCOTUS Feature Subset",
        "",
        "## Inputs",
        "",
        f"- Source run: `{manifest['subset_source_dir']}`",
        f"- Output dir: `{manifest['output_dir']}`",
        f"- Rows kept: `{manifest['n_rows']}` of `{manifest['source_n_rows']}`",
        f"- Filters: `{manifest['subset_filter_label']}`",
        "",
        "## Split/Label Counts",
        "",
        markdown_table(["Split", "Label", "Rows"], label_rows),
        "",
        "## Issue/Split/Justice Counts",
        "",
        markdown_table(["Issue", "Split", "Justice", "Rows"], issue_rows),
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--tag", default="scotus_cached_subset")
    parser.add_argument("--where", action="append", default=[], help="Keep rows matching field=value. Can repeat.")
    parser.add_argument(
        "--where-in",
        action="append",
        default=[],
        help="Keep rows where field is one of value1|value2. Can repeat.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_dir = args.features_dir
    where = parse_where(args.where)
    where_in = parse_where_in(args.where_in)
    source_examples = read_jsonl(source_dir / "probe_examples.jsonl")
    source_meta = read_jsonl(source_dir / "feature_meta.jsonl")
    if len(source_examples) != len(source_meta):
        raise RuntimeError("probe_examples.jsonl and feature_meta.jsonl row counts do not match")
    for idx, (example, meta) in enumerate(zip(source_examples, source_meta, strict=True)):
        if str(example.get("example_id")) != str(meta.get("example_id")):
            raise RuntimeError(f"Row {idx} example_id mismatch between probe_examples and feature_meta")

    keep_indices = [
        idx
        for idx, row in enumerate(source_meta)
        if row_matches(row, where=where, where_in=where_in)
    ]
    if not keep_indices:
        raise RuntimeError("No rows matched the requested filters")

    output_dir = args.output_root / f"{args.tag}_{now_stamp()}"
    output_dir.mkdir(parents=True, exist_ok=True)
    kept_examples = [dict(source_examples[idx]) for idx in keep_indices]
    kept_meta = [dict(source_meta[idx]) for idx in keep_indices]
    write_jsonl(output_dir / "probe_examples.jsonl", kept_examples)
    write_jsonl(output_dir / "feature_meta.jsonl", kept_meta)

    idx_arr = np.array(keep_indices, dtype=np.int64)
    with np.load(source_dir / "features.npz") as data:
        subset_arrays = {key: data[key][idx_arr] for key in data.files}
    np.savez_compressed(output_dir / "features.npz", **subset_arrays)

    source_manifest = read_json(source_dir / "manifest.json")
    filter_bits = [f"{field}={value}" for field, value in where]
    filter_bits.extend(f"{field} in {'|'.join(sorted(values))}" for field, values in where_in)
    manifest = dict(source_manifest)
    manifest.update(
        {
            "started_at": now_iso(),
            "finished_at": now_iso(),
            "subset_source_dir": str(source_dir),
            "output_dir": str(output_dir),
            "source_n_rows": len(source_meta),
            "n_rows": len(kept_meta),
            "subset_filters": {
                "where": [{"field": field, "value": value} for field, value in where],
                "where_in": [
                    {"field": field, "values": sorted(values)}
                    for field, values in where_in
                ],
            },
            "subset_filter_label": "; ".join(filter_bits) if filter_bits else "none",
        }
    )
    write_json(output_dir / "manifest.json", manifest)
    write_report(output_dir / "report.md", manifest=manifest, rows=kept_meta)
    print(f"Wrote {output_dir}")


if __name__ == "__main__":
    main()
