#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any


DEFAULT_OUTPUT_ROOT = Path("/home/orwel/dev_genius/experiments/Character Creation/sweep_v4")


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def canonical_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip().lower()


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def row_score(row: dict[str, Any]) -> tuple[int, int, int]:
    return (
        int(row.get("judge_trace_quality", 0) or 0),
        1 if row.get("judge_all_metrics_pass") else 0,
        1 if row.get("judge_format_ok") else 0,
    )


def item_key(row: dict[str, Any]) -> str:
    return "|".join(
        [
            canonical_text(row.get("behavior", "")),
            canonical_text(row.get("source_title", "")),
            canonical_text(row.get("focal_character", "")),
            canonical_text(row.get("counterpart", "")),
            canonical_text(row.get("scene_summary", "")),
            canonical_text(row.get("user_prompt", "")),
        ]
    )


def split_by_behavior(item_ids: list[str], behavior_map: dict[str, str], seed: int) -> dict[str, str]:
    rng = random.Random(seed)
    groups: dict[str, list[str]] = defaultdict(list)
    for item_id in item_ids:
        groups[behavior_map[item_id]].append(item_id)
    out: dict[str, str] = {}
    for behavior, ids in groups.items():
        ids = sorted(set(ids))
        rng.shuffle(ids)
        n = len(ids)
        n_val = max(1, round(n * 0.1))
        n_test = max(1, round(n * 0.1))
        if n >= 10 and n_val + n_test >= n:
            n_val = max(1, n // 10)
            n_test = max(1, n // 10)
        train_cut = n - n_val - n_test
        if train_cut <= 0:
            train_cut = max(1, n - 2)
        for idx, item_id in enumerate(ids):
            if idx < train_cut:
                out[item_id] = "train"
            elif idx < train_cut + n_val:
                out[item_id] = "val"
            else:
                out[item_id] = "test"
    return out


def split_selected_items_quality_aware(selected_items: list[dict[str, Any]], seed: int) -> dict[str, str]:
    rng = random.Random(seed)
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in selected_items:
        groups[item["behavior"]].append(item)
    out: dict[str, str] = {}
    for behavior, items in groups.items():
        items = list(items)
        rng.shuffle(items)
        items.sort(key=lambda item: (item["pair_quality"], item["pair_quality_mean"]), reverse=True)
        n = len(items)
        n_val = max(1, round(n * 0.1))
        n_test = max(1, round(n * 0.1))
        if n >= 10 and n_val + n_test >= n:
            n_val = max(1, n // 10)
            n_test = max(1, n // 10)
        eval_items = items[: n_val + n_test]
        train_items = items[n_val + n_test :]
        for idx, item in enumerate(eval_items):
            out[item["item_id"]] = "val" if idx % 2 == 0 and idx // 2 < n_val else "test"
        # If alternating underfilled one side due to odd counts, top up deterministically.
        val_count = sum(1 for item in eval_items if out[item["item_id"]] == "val")
        test_count = sum(1 for item in eval_items if out[item["item_id"]] == "test")
        if val_count < n_val:
            for item in eval_items:
                if out[item["item_id"]] == "test":
                    out[item["item_id"]] = "val"
                    val_count += 1
                    test_count -= 1
                    if val_count >= n_val:
                        break
        if test_count < n_test:
            for item in reversed(eval_items):
                if out[item["item_id"]] == "val":
                    out[item["item_id"]] = "test"
                    test_count += 1
                    val_count -= 1
                    if test_count >= n_test:
                        break
        for item in train_items:
            out[item["item_id"]] = "train"
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True, help="Input dataset dirs containing all_completions.jsonl")
    ap.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    ap.add_argument("--tag", default="book_character_prefill_dataset_balanced_v1")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--per-behavior-cap", type=int, default=0, help="0 means use the minimum available count")
    ap.add_argument("--max-items-per-source-title", type=int, default=0)
    ap.add_argument("--exclude-merged-item-keys", default="", help="Comma-separated pre-merge item keys to exclude")
    ap.add_argument("--exclude-merged-item-keys-file", type=Path)
    ap.add_argument("--quality-aware-eval-split", action="store_true")
    args = ap.parse_args()
    excluded_keys = {part.strip() for part in args.exclude_merged_item_keys.split(",") if part.strip()}
    if args.exclude_merged_item_keys_file:
        for line in args.exclude_merged_item_keys_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                excluded_keys.add(line)

    stamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    out_dir = args.output_root / f"{args.tag}_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_rows: list[dict[str, Any]] = []
    inputs: list[str] = []
    for raw in args.inputs:
        run_dir = Path(raw)
        path = run_dir / "all_completions.jsonl"
        if not path.exists():
            raise SystemExit(f"Missing {path}")
        rows = read_jsonl(path)
        for row in rows:
            row["source_run_dir"] = str(run_dir)
        raw_rows.extend(rows)
        inputs.append(str(run_dir))

    grouped: dict[str, dict[int, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for row in raw_rows:
        grouped[item_key(row)][int(row["label"])].append(row)

    merged_items: list[dict[str, Any]] = []
    for key, by_label in grouped.items():
        if key in excluded_keys:
            continue
        if 0 not in by_label or 1 not in by_label:
            continue
        fail_row = max(by_label[0], key=row_score)
        pass_row = max(by_label[1], key=row_score)
        item_id = f"merged_{len(merged_items):04d}"
        merged_items.append(
            {
                "item_id": item_id,
                "behavior": pass_row["behavior"],
                "pair_quality": min(
                    int(pass_row.get("judge_trace_quality", 0) or 0),
                    int(fail_row.get("judge_trace_quality", 0) or 0),
                ),
                "pair_quality_mean": (
                    int(pass_row.get("judge_trace_quality", 0) or 0)
                    + int(fail_row.get("judge_trace_quality", 0) or 0)
                )
                / 2.0,
                "rows": [fail_row, pass_row],
            }
        )

    by_behavior: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in merged_items:
        by_behavior[item["behavior"]].append(item)
    for items in by_behavior.values():
        items.sort(key=lambda item: (item["pair_quality"], item["pair_quality_mean"]), reverse=True)

    available_counts = {behavior: len(items) for behavior, items in sorted(by_behavior.items())}
    if not available_counts:
        raise SystemExit("No usable paired items found after merge")
    target_per_behavior = args.per_behavior_cap or min(available_counts.values())

    selected_items: list[dict[str, Any]] = []
    source_counts: Counter[tuple[str, str]] = Counter()
    for behavior, items in sorted(by_behavior.items()):
        by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for item in items:
            source_title = item["rows"][0].get("source_title", "")
            by_source[source_title].append(item)
        source_names = sorted(
            by_source,
            key=lambda name: (
                max((row["pair_quality"], row["pair_quality_mean"]) for row in by_source[name]),
                len(by_source[name]),
            ),
            reverse=True,
        )
        picked = 0
        while picked < target_per_behavior:
            made_progress = False
            for source_name in source_names:
                rows = by_source[source_name]
                while rows and (
                    args.max_items_per_source_title > 0
                    and source_counts[(behavior, source_name)] >= args.max_items_per_source_title
                ):
                    rows.pop(0)
                if not rows:
                    continue
                item = rows.pop(0)
                selected_items.append(item)
                source_counts[(behavior, source_name)] += 1
                picked += 1
                made_progress = True
                if picked >= target_per_behavior:
                    break
            if not made_progress:
                break

    behavior_map = {item["item_id"]: item["behavior"] for item in selected_items}
    splits = (
        split_selected_items_quality_aware(selected_items, args.seed)
        if args.quality_aware_eval_split
        else split_by_behavior([item["item_id"] for item in selected_items], behavior_map, args.seed)
    )

    completions: list[dict[str, Any]] = []
    for item in selected_items:
        for row in item["rows"]:
            row = dict(row)
            row["merged_item_id"] = item["item_id"]
            row["split"] = splits[item["item_id"]]
            row["pair_quality"] = item["pair_quality"]
            row["pair_quality_mean"] = item["pair_quality_mean"]
            completions.append(row)
    completions.sort(key=lambda row: (row["split"], row["behavior"], row["merged_item_id"], row["label"]))

    write_jsonl(out_dir / "all_completions.jsonl", completions)
    write_jsonl(out_dir / "balanced_items.jsonl", selected_items)
    for split in ("train", "val", "test"):
        write_jsonl(out_dir / f"{split}.jsonl", [row for row in completions if row["split"] == split])

    summary = {
        "finished_at": now_iso(),
        "inputs": inputs,
        "n_raw_rows": len(raw_rows),
        "n_merged_paired_items": len(merged_items),
        "n_excluded_merged_item_keys": len(excluded_keys),
        "excluded_merged_item_keys": sorted(excluded_keys),
        "available_item_counts_by_behavior": available_counts,
        "target_per_behavior": target_per_behavior,
        "max_items_per_source_title": args.max_items_per_source_title,
        "quality_aware_eval_split": args.quality_aware_eval_split,
        "n_selected_items": len(selected_items),
        "n_selected_completions": len(completions),
        "selected_item_counts_by_behavior": dict(sorted(Counter(item["behavior"] for item in selected_items).items())),
        "selected_completion_counts_by_behavior": dict(sorted(Counter(row["behavior"] for row in completions).items())),
        "selected_item_counts_by_source_title": dict(sorted(Counter(item["rows"][0].get("source_title", "") for item in selected_items).items())),
        "split_counts": dict(sorted(Counter(row["split"] for row in completions).items())),
        "label_counts": dict(sorted(Counter(str(row["label"]) for row in completions).items())),
        "mean_trace_quality": sum(int(row.get("judge_trace_quality", 0) or 0) for row in completions) / max(len(completions), 1),
    }
    write_json(out_dir / "summary.json", summary)


if __name__ == "__main__":
    main()
