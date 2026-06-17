#!/usr/bin/env python3
"""Summarize held-out A/B/C meta-format eval shards."""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
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


def avg(values: list[float]) -> float | None:
    return mean(values) if values else None


def rate(num: int, den: int) -> float | None:
    return (num / den) if den else None


def summarize_group(rows: list[dict[str, Any]]) -> dict[str, Any]:
    reasoning = [r for r in rows if r.get("track") == "reasoning"]
    scored = [r for r in reasoning if r.get("is_correct") is not None]
    correct = [r for r in scored if r.get("is_correct") is True]
    finish = Counter(str(r.get("finish_reason") or "unknown") for r in rows)
    return {
        "responses": len(rows),
        "avg_gen_tokens": avg([float(r.get("n_gen_tokens") or 0.0) for r in rows]),
        "avg_latency_s": avg([float(r.get("latency_s") or 0.0) for r in rows]),
        "format_adherent": sum(1 for r in rows if r.get("format_adherent")),
        "format_adherence_rate": rate(sum(1 for r in rows if r.get("format_adherent")), len(rows)),
        "truncated": sum(1 for r in rows if str(r.get("finish_reason") or "") == "length"),
        "truncation_rate": rate(sum(1 for r in rows if str(r.get("finish_reason") or "") == "length"), len(rows)),
        "visible_thinking": sum(1 for r in rows if r.get("contains_thinking_process")),
        "visible_thinking_rate": rate(sum(1 for r in rows if r.get("contains_thinking_process")), len(rows)),
        "trait_label_leak": sum(1 for r in rows if r.get("trait_label_leak")),
        "trait_label_leak_rate": rate(sum(1 for r in rows if r.get("trait_label_leak")), len(rows)),
        "reasoning_responses": len(reasoning),
        "reasoning_scored": len(scored),
        "reasoning_correct": len(correct),
        "reasoning_accuracy": rate(len(correct), len(scored)),
        "reasoning_coverage": rate(len(scored), len(reasoning)),
        "finish_reasons": dict(finish),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize meta-format eval shards")
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    manifest_path = input_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    records: list[dict[str, Any]] = []
    for fp in sorted(input_dir.glob("records_shard_*.jsonl")):
        records.extend(load_jsonl(fp))

    expected_total = int(manifest.get("n_tasks_total") or 0)
    by_condition_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_server_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_track_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        by_condition_rows[str(row.get("condition_id") or "unknown")].append(row)
        by_server_rows[str(row.get("server_label") or "unknown")].append(row)
        by_track_rows[str(row.get("track") or "unknown")].append(row)

    summary = {
        "timestamp": __import__("datetime").datetime.now().isoformat(),
        "dataset": manifest.get("dataset"),
        "goal": manifest.get("goal"),
        "expected_total": expected_total,
        "completed_total": len(records),
        "completion_rate": rate(len(records), expected_total),
        "pending_total": max(expected_total - len(records), 0),
        "overall": summarize_group(records),
        "by_condition": {k: summarize_group(v) for k, v in sorted(by_condition_rows.items())},
        "by_server": {k: summarize_group(v) for k, v in sorted(by_server_rows.items())},
        "by_track": {k: summarize_group(v) for k, v in sorted(by_track_rows.items())},
    }

    if records:
        mean_tokens = float(summary["overall"]["avg_gen_tokens"] or 0.0)
        summary["estimated_total_gen_tokens"] = int(round(mean_tokens * expected_total)) if expected_total else None
    else:
        summary["estimated_total_gen_tokens"] = None

    Path(args.output_json).write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines: list[str] = []
    lines.append(f"# {manifest.get('dataset', 'personality_meta_eval')} summary")
    lines.append("")
    lines.append(f"- completed: {summary['completed_total']} / {summary['expected_total']}")
    lines.append(f"- completion rate: {summary['completion_rate']}")
    lines.append(f"- estimated total gen tokens: {summary['estimated_total_gen_tokens']}")
    lines.append(f"- overall format adherence: {summary['overall']['format_adherence_rate']}")
    lines.append(f"- overall visible thinking: {summary['overall']['visible_thinking_rate']}")
    lines.append(f"- overall truncation: {summary['overall']['truncation_rate']}")
    lines.append("")
    lines.append("## By Condition")
    lines.append("")
    for cond, payload in summary["by_condition"].items():
        lines.append(f"### {cond}")
        lines.append(f"- responses: {payload['responses']}")
        lines.append(f"- avg gen tokens: {payload['avg_gen_tokens']}")
        lines.append(f"- avg latency: {payload['avg_latency_s']}")
        lines.append(f"- format adherence: {payload['format_adherence_rate']}")
        lines.append(f"- visible thinking: {payload['visible_thinking_rate']}")
        lines.append(f"- truncation: {payload['truncation_rate']}")
        lines.append(f"- trait label leak: {payload['trait_label_leak_rate']}")
        if payload["reasoning_responses"]:
            lines.append(f"- reasoning accuracy: {payload['reasoning_accuracy']}")
            lines.append(f"- reasoning coverage: {payload['reasoning_coverage']}")
        lines.append("")
    lines.append("## By Server")
    lines.append("")
    for server, payload in summary["by_server"].items():
        lines.append(f"### {server}")
        lines.append(f"- responses: {payload['responses']}")
        lines.append(f"- avg gen tokens: {payload['avg_gen_tokens']}")
        lines.append(f"- avg latency: {payload['avg_latency_s']}")
        lines.append(f"- format adherence: {payload['format_adherence_rate']}")
        lines.append("")
    Path(args.output_md).write_text("\n".join(lines).strip() + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
