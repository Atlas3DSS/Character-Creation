#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any


BEHAVIORS = (
    "conflict_detection",
    "constraint_preservation",
    "repair_after_challenge",
    "selective_introspection",
    "state_carryover",
)


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def canonical_item_signature(item: dict[str, Any]) -> str:
    title = re.sub(r"\W+", " ", item["title"].lower()).strip()
    setup = re.sub(r"\W+", " ", item["setup"].lower()).strip()
    turns = " || ".join(re.sub(r"\W+", " ", t["content"].lower()).strip() for t in item["turns"])
    return f"{item['behavior']}::{title}::{setup}::{turns}"


def slim_row(row: dict[str, Any], source_dir: str) -> dict[str, Any]:
    item = row["item"]
    return {
        "source_dir": source_dir,
        "candidate_id": item["candidate_id"],
        "behavior": item["behavior"],
        "title": item["title"],
        "setup": item["setup"],
        "turns": item["turns"],
        "contrast": item["contrast"],
        "expected_pass": item["expected_pass"],
        "expected_fail": item["expected_fail"],
        "metrics": item["metrics"],
        "notes": item.get("notes", ""),
        "combined_score": row["combined_score"],
        "hard_score": row["hard_score"],
        "judge_score": row["judge_score"],
        "judge_note": row.get("judge_rating", {}).get("note", ""),
    }


def stratified_split(rows: list[dict[str, Any]], train_per_behavior: int, val_per_behavior: int, test_per_behavior: int) -> dict[str, list[dict[str, Any]]]:
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[row["behavior"]].append(row)
    split = {"train": [], "val": [], "test": []}
    for behavior in BEHAVIORS:
        bucket = buckets[behavior]
        split["train"].extend(bucket[:train_per_behavior])
        split["val"].extend(bucket[train_per_behavior : train_per_behavior + val_per_behavior])
        split["test"].extend(bucket[train_per_behavior + val_per_behavior : train_per_behavior + val_per_behavior + test_per_behavior])
    return split


def build_report(output_dir: Path, selected: list[dict[str, Any]], split: dict[str, list[dict[str, Any]]], sources: list[str]) -> None:
    counts = Counter(row["behavior"] for row in selected)
    lines = [
        "# Meta-Cognition Seed Set",
        "",
        f"- generated_at: {now_iso()}",
        f"- sources: {sources}",
        f"- total_items: {len(selected)}",
        f"- behavior_counts: {dict(sorted(counts.items()))}",
        "",
        "## Splits",
        "",
    ]
    for split_name in ("train", "val", "test"):
        split_counts = Counter(row["behavior"] for row in split[split_name])
        lines.append(f"- {split_name}: {len(split[split_name])} {dict(sorted(split_counts.items()))}")
    lines.extend(["", "## Top Items", ""])
    for row in selected[:15]:
        lines.append(f"- `{row['behavior']}` | score={row['combined_score']:.2f} | {row['title']}")
    (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_review_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "behavior",
        "title",
        "combined_score",
        "hard_score",
        "judge_score",
        "source_dir",
        "setup",
        "turn_1",
        "turn_2",
        "turn_3",
        "metric_ids",
        "judge_note",
    ]
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            turns = [t["content"] for t in row["turns"]]
            writer.writerow(
                {
                    "behavior": row["behavior"],
                    "title": row["title"],
                    "combined_score": f"{row['combined_score']:.2f}",
                    "hard_score": f"{row['hard_score']:.2f}",
                    "judge_score": f"{row['judge_score']:.2f}",
                    "source_dir": row["source_dir"],
                    "setup": row["setup"],
                    "turn_1": turns[0] if len(turns) > 0 else "",
                    "turn_2": turns[1] if len(turns) > 1 else "",
                    "turn_3": turns[2] if len(turns) > 2 else "",
                    "metric_ids": ",".join(m["id"] for m in row["metrics"]),
                    "judge_note": row["judge_note"],
                }
            )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dirs", required=True, help="Comma-separated run directories containing final_seed_set.jsonl")
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--per-behavior", type=int, default=12)
    ap.add_argument("--train-per-behavior", type=int, default=8)
    ap.add_argument("--val-per-behavior", type=int, default=2)
    ap.add_argument("--test-per-behavior", type=int, default=2)
    args = ap.parse_args()

    source_dirs = [x.strip() for x in args.input_dirs.split(",") if x.strip()]
    rows: list[dict[str, Any]] = []
    for source in source_dirs:
        source_path = Path(source)
        for row in load_jsonl(source_path / "final_seed_set.jsonl"):
            rows.append(slim_row(row, str(source_path)))

    rows = sorted(rows, key=lambda r: (r["combined_score"], r["hard_score"]), reverse=True)
    selected: list[dict[str, Any]] = []
    used_signatures: set[str] = set()
    counts: Counter[str] = Counter()
    for row in rows:
        behavior = row["behavior"]
        if behavior not in BEHAVIORS:
            continue
        if counts[behavior] >= args.per_behavior:
            continue
        sig = canonical_item_signature(row)
        if sig in used_signatures:
            continue
        selected.append(row)
        used_signatures.add(sig)
        counts[behavior] += 1
        if all(counts[b] >= args.per_behavior for b in BEHAVIORS):
            break

    missing = {b: args.per_behavior - counts[b] for b in BEHAVIORS if counts[b] < args.per_behavior}
    if missing:
        raise RuntimeError(f"could not satisfy per-behavior quota: {missing}")

    split = stratified_split(selected, args.train_per_behavior, args.val_per_behavior, args.test_per_behavior)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "balanced_seed_set.jsonl", selected)
    write_jsonl(args.output_dir / "train.jsonl", split["train"])
    write_jsonl(args.output_dir / "val.jsonl", split["val"])
    write_jsonl(args.output_dir / "test.jsonl", split["test"])
    write_review_csv(args.output_dir / "review.csv", selected)
    summary = {
        "generated_at": now_iso(),
        "sources": source_dirs,
        "total_items": len(selected),
        "per_behavior": args.per_behavior,
        "counts": dict(sorted(Counter(r["behavior"] for r in selected).items())),
        "splits": {name: dict(sorted(Counter(r["behavior"] for r in split[name]).items())) for name in split},
    }
    write_json(args.output_dir / "summary.json", summary)
    build_report(args.output_dir, selected, split, source_dirs)


if __name__ == "__main__":
    main()
