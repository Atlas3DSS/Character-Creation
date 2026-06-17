#!/usr/bin/env python3
"""Materialize a condition-specific subset from a completed meta-format eval run."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Materialize a condition-specific eval subset")
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--condition-id", required=True)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = json.loads((input_dir / "manifest.json").read_text())
    selected_conditions = [c for c in manifest.get("conditions", []) if c.get("condition_id") == args.condition_id]
    if not selected_conditions:
        raise ValueError(f"Condition {args.condition_id!r} not found in {input_dir / 'manifest.json'}")

    records: list[dict[str, Any]] = []
    shard_index = 0
    for src in sorted(input_dir.glob("records_shard_*.jsonl")):
        shard_rows = [row for row in load_jsonl(src) if row.get("condition_id") == args.condition_id]
        if not shard_rows:
            continue
        write_jsonl(output_dir / f"records_shard_{shard_index:02d}.jsonl", shard_rows)
        shard_index += 1
        records.extend(shard_rows)

    if not records:
        raise RuntimeError(f"No rows found for condition {args.condition_id!r}")

    personas_src = input_dir / "personas.jsonl"
    prompts_src = input_dir / "prompts.jsonl"
    if personas_src.exists():
        (output_dir / "personas.jsonl").write_text(personas_src.read_text(), encoding="utf-8")
    if prompts_src.exists():
        (output_dir / "prompts.jsonl").write_text(prompts_src.read_text(), encoding="utf-8")

    new_manifest = dict(manifest)
    new_manifest["dataset"] = f"{manifest.get('dataset', 'personality_meta_eval')}_{args.condition_id}"
    new_manifest["source_dataset"] = manifest.get("dataset")
    new_manifest["source_input_dir"] = str(input_dir.resolve())
    new_manifest["condition_ids"] = [args.condition_id]
    new_manifest["conditions"] = selected_conditions
    new_manifest["n_conditions"] = 1
    new_manifest["n_tasks_total"] = len(records)
    new_manifest["materialized_timestamp"] = __import__("datetime").datetime.now().isoformat()
    (output_dir / "manifest.json").write_text(json.dumps(new_manifest, indent=2), encoding="utf-8")

    summary = {
        "timestamp": __import__("datetime").datetime.now().isoformat(),
        "dataset": new_manifest["dataset"],
        "source_dataset": manifest.get("dataset"),
        "condition_id": args.condition_id,
        "rows": len(records),
        "reasoning_rows": sum(1 for r in records if r.get("track") == "reasoning"),
        "format_adherent": sum(1 for r in records if r.get("format_adherent")),
        "visible_thinking": sum(1 for r in records if r.get("contains_thinking_process")),
        "truncated": sum(1 for r in records if str(r.get("finish_reason") or "") == "length"),
    }
    summary["format_adherence_rate"] = summary["format_adherent"] / summary["rows"]
    summary["visible_thinking_rate"] = summary["visible_thinking"] / summary["rows"]
    summary["truncation_rate"] = summary["truncated"] / summary["rows"]
    reasoning_scored = [r for r in records if r.get("track") == "reasoning" and r.get("is_correct") is not None]
    summary["reasoning_scored"] = len(reasoning_scored)
    summary["reasoning_correct"] = sum(1 for r in reasoning_scored if r.get("is_correct") is True)
    summary["reasoning_accuracy"] = (
        summary["reasoning_correct"] / summary["reasoning_scored"] if summary["reasoning_scored"] else None
    )
    (output_dir / "subset_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines = [
        f"# {new_manifest['dataset']} subset",
        "",
        f"- source dataset: {manifest.get('dataset')}",
        f"- condition: {args.condition_id}",
        f"- rows: {summary['rows']}",
        f"- format adherence: {summary['format_adherence_rate']}",
        f"- visible thinking: {summary['visible_thinking_rate']}",
        f"- truncation: {summary['truncation_rate']}",
        f"- reasoning accuracy: {summary['reasoning_accuracy']}",
    ]
    (output_dir / "subset_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
