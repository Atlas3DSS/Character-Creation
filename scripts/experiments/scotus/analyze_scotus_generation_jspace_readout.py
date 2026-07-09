#!/usr/bin/env python3
"""Summarize generated-boundary SCOTUS J-space readout records."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any, Iterable


def now_iso() -> str:
    return datetime.now().astimezone().isoformat()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def clean_key(value: Any) -> str:
    if value is None:
        return "unknown"
    return str(value)


def hit_markers(record: dict[str, Any], field: str) -> list[str]:
    return [str(hit.get("marker")) for hit in record.get(field, []) if hit.get("marker")]


def row_has_hits(record: dict[str, Any], field: str) -> bool:
    return bool(record.get(field))


def summarize_group(records: list[dict[str, Any]], keys: tuple[str, ...]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        key = tuple(clean_key(record.get(part)) for part in keys)
        groups[key].append(record)

    summaries: list[dict[str, Any]] = []
    for key, rows in groups.items():
        pos_hit_rows = sum(1 for row in rows if row_has_hits(row, "positive_legal_hits"))
        neg_hit_rows = sum(1 for row in rows if row_has_hits(row, "negative_legal_hits"))
        pos_markers: Counter[str] = Counter()
        neg_markers: Counter[str] = Counter()
        for row in rows:
            pos_markers.update(hit_markers(row, "positive_legal_hits"))
            neg_markers.update(hit_markers(row, "negative_legal_hits"))
        summary = {name: value for name, value in zip(keys, key)}
        summary.update(
            {
                "records": len(rows),
                "mean_transported_norm": round(mean(float(row["transported_norm"]) for row in rows), 4),
                "positive_legal_hit_rate": round(pos_hit_rows / len(rows), 4),
                "negative_legal_hit_rate": round(neg_hit_rows / len(rows), 4),
                "positive_marker_counts": dict(pos_markers.most_common(12)),
                "negative_marker_counts": dict(neg_markers.most_common(12)),
            }
        )
        summaries.append(summary)
    return summaries


def side_delta_summary(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    matched: dict[tuple[str, ...], dict[str, dict[str, Any]]] = defaultdict(dict)
    for record in records:
        side = clean_key(record.get("side"))
        if side not in {"a", "b"}:
            continue
        if not clean_key(record.get("boundary_kind")).startswith("generated_"):
            continue
        key = (
            clean_key(record.get("generated_pair_id")),
            clean_key(record.get("comparison_axis")),
            clean_key(record.get("boundary_kind")),
            clean_key(record.get("boundary_index")),
            clean_key(record.get("layer")),
            clean_key(record.get("requested_position")),
        )
        matched[key][side] = record

    deltas: dict[tuple[str, ...], list[dict[str, float]]] = defaultdict(list)
    for key, sides in matched.items():
        if "a" not in sides or "b" not in sides:
            continue
        _, axis, boundary_kind, boundary_index, layer, position = key
        a = sides["a"]
        b = sides["b"]
        delta_key = (axis, boundary_kind, boundary_index, layer, position)
        deltas[delta_key].append(
            {
                "norm_delta_a_minus_b": float(a["transported_norm"]) - float(b["transported_norm"]),
                "pos_hit_delta_a_minus_b": float(row_has_hits(a, "positive_legal_hits"))
                - float(row_has_hits(b, "positive_legal_hits")),
                "neg_hit_delta_a_minus_b": float(row_has_hits(a, "negative_legal_hits"))
                - float(row_has_hits(b, "negative_legal_hits")),
            }
        )

    summaries: list[dict[str, Any]] = []
    for key, rows in deltas.items():
        axis, boundary_kind, boundary_index, layer, position = key
        summaries.append(
            {
                "comparison_axis": axis,
                "boundary_kind": boundary_kind,
                "boundary_index": boundary_index,
                "layer": layer,
                "requested_position": position,
                "matched_pairs": len(rows),
                "mean_norm_delta_a_minus_b": round(mean(row["norm_delta_a_minus_b"] for row in rows), 4),
                "mean_pos_hit_delta_a_minus_b": round(mean(row["pos_hit_delta_a_minus_b"] for row in rows), 4),
                "mean_neg_hit_delta_a_minus_b": round(mean(row["neg_hit_delta_a_minus_b"] for row in rows), 4),
            }
        )
    return summaries


def marker_totals(records: list[dict[str, Any]]) -> dict[str, Any]:
    positive: Counter[str] = Counter()
    negative: Counter[str] = Counter()
    for record in records:
        positive.update(hit_markers(record, "positive_legal_hits"))
        negative.update(hit_markers(record, "negative_legal_hits"))
    return {
        "positive": dict(positive.most_common(30)),
        "negative": dict(negative.most_common(30)),
    }


def table(headers: list[str], rows: Iterable[dict[str, Any]], limit: int) -> str:
    rows = list(rows)[:limit]
    if not rows:
        return "_No rows._\n"
    out = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for row in rows:
        out.append("| " + " | ".join(str(row.get(header, "")) for header in headers) + " |")
    return "\n".join(out) + "\n"


def make_report(records: list[dict[str, Any]], summary: dict[str, Any]) -> str:
    by_axis_boundary = sorted(
        summary["by_axis_boundary"],
        key=lambda row: (row["comparison_axis"], row["boundary_kind"], row["side"], int(row["layer"])),
    )
    legal_sorted = sorted(
        by_axis_boundary,
        key=lambda row: (row["positive_legal_hit_rate"], row["mean_transported_norm"]),
        reverse=True,
    )
    deltas = sorted(
        summary["side_deltas"],
        key=lambda row: abs(float(row["mean_norm_delta_a_minus_b"])),
        reverse=True,
    )
    counts = Counter(clean_key(row.get("boundary_kind")) for row in records)
    axes = Counter(clean_key(row.get("comparison_axis")) for row in records)
    lines = [
        "# SCOTUS Generated-Boundary J-space Readout",
        "",
        f"Generated: {now_iso()}",
        f"Records: {len(records)}",
        f"Axes: {dict(axes)}",
        f"Boundary kinds: {dict(counts)}",
        "",
        "## Highest Legal-Marker Hit Rates",
        "",
        table(
            [
                "comparison_axis",
                "boundary_kind",
                "side",
                "layer",
                "records",
                "mean_transported_norm",
                "positive_legal_hit_rate",
                "negative_legal_hit_rate",
            ],
            legal_sorted,
            24,
        ),
        "",
        "## Largest Generated A/B Norm Deltas",
        "",
        table(
            [
                "comparison_axis",
                "boundary_kind",
                "boundary_index",
                "layer",
                "requested_position",
                "matched_pairs",
                "mean_norm_delta_a_minus_b",
                "mean_pos_hit_delta_a_minus_b",
            ],
            deltas,
            24,
        ),
        "",
        "## Marker Totals",
        "",
        "Positive markers:",
        json.dumps(summary["marker_totals"]["positive"], indent=2, sort_keys=True),
        "",
        "Negative markers:",
        json.dumps(summary["marker_totals"]["negative"], indent=2, sort_keys=True),
    ]
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = read_jsonl(args.records)
    if not records:
        raise RuntimeError(f"No records found in {args.records}")
    output_dir = args.output_dir or args.records.parent / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "created_at": now_iso(),
        "records": len(records),
        "source_records": str(args.records),
        "by_axis_boundary": summarize_group(records, ("comparison_axis", "boundary_kind", "side", "layer")),
        "by_variant_layer": summarize_group(records, ("variant", "layer")),
        "side_deltas": side_delta_summary(records),
        "marker_totals": marker_totals(records),
    }
    write_json(output_dir / "summary.json", summary)
    write_text(output_dir / "report.md", make_report(records, summary))
    print(output_dir)


if __name__ == "__main__":
    main()
