#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any


DEFAULT_PATCH_RECORDS = (
    "/home/orwel/dev_genius/experiments/Character Creation/sweep_v4/"
    "symphonic_voice_live_patch_v2_compositional_20260418_082730/records.jsonl"
)
DEFAULT_SOURCE_DATASET_DIR = (
    "/home/orwel/dev_genius/experiments/Character Creation/sweep_v4/"
    "book_character_prefill_dataset_balanced_v6_reviewed_20260417_151017"
)
DEFAULT_OUTPUT_ROOT = "/home/orwel/dev_genius/experiments/Character Creation/sweep_v4"


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


def parse_pairs(raw: str) -> set[tuple[str, str]]:
    pairs: set[tuple[str, str]] = set()
    for part in [x.strip() for x in raw.split(",") if x.strip()]:
        src, dst = [x.strip() for x in part.split(":", 1)]
        pairs.add((src, dst))
    return pairs


def best_source_rows(rows: list[dict[str, Any]], *, min_pair_quality: int) -> dict[str, dict[str, Any]]:
    best: dict[str, dict[str, Any]] = {}
    for row in rows:
        if int(row.get("label", 0) or 0) != 1:
            continue
        if not bool(row.get("judge_format_ok", False)):
            continue
        if int(row.get("pair_quality", 0) or 0) < min_pair_quality:
            continue
        key = row["merged_item_id"]
        prev = best.get(key)
        if prev is None:
            best[key] = row
            continue
        old_score = (
            int(prev.get("pair_quality", 0) or 0),
            float(prev.get("pair_quality_mean", 0.0) or 0.0),
        )
        new_score = (
            int(row.get("pair_quality", 0) or 0),
            float(row.get("pair_quality_mean", 0.0) or 0.0),
        )
        if new_score > old_score:
            best[key] = row
    return best


def select_reverse_records(
    records: list[dict[str, Any]],
    *,
    pairs: set[tuple[str, str]],
    failure_lift_max: float,
    partial_lift_min: float,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        src = row.get("source_anchor")
        dst = row.get("target_anchor")
        if (src, dst) not in pairs:
            continue
        grouped[(row["pair_name"], row["item_key"])].append(row)

    selected: list[dict[str, Any]] = []
    for (_pair_name, _item_key), rows in sorted(grouped.items()):
        baseline = next((row for row in rows if row.get("condition") == "baseline"), None)
        patched = [row for row in rows if row.get("condition") != "baseline"]
        if baseline is None or not patched:
            continue
        best_patch = max(patched, key=lambda row: float(row.get("alpha", 0.0) or 0.0))
        if not best_patch.get("format_ok", False):
            reason = "format_failure"
        else:
            target_prob_before = float(baseline.get("target_prob") or 0.0)
            target_prob_after = float(best_patch.get("target_prob") or 0.0)
            target_lift = target_prob_after - target_prob_before
            hit_target = best_patch.get("pred_anchor_id") == best_patch.get("target_anchor")
            if hit_target:
                continue
            if target_lift <= failure_lift_max:
                reason = "reverse_failure"
            elif target_lift >= partial_lift_min:
                reason = "partial_reversal"
            else:
                reason = "weak_partial"
        payload = dict(best_patch)
        payload["baseline_target_prob"] = baseline.get("target_prob")
        payload["baseline_source_prob"] = baseline.get("source_prob")
        payload["target_prob_lift"] = (
            float(best_patch.get("target_prob") or 0.0) - float(baseline.get("target_prob") or 0.0)
        )
        payload["source_prob_delta"] = (
            float(best_patch.get("source_prob") or 0.0) - float(baseline.get("source_prob") or 0.0)
        )
        payload["selection_reason"] = reason
        selected.append(payload)
    return selected


def add_neutral_matches(
    *,
    source_rows_by_item: dict[str, dict[str, Any]],
    selected_ids: set[str],
    selected_behaviors: Counter[str],
    max_neutral_per_behavior: int,
    seed: int,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    by_behavior: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item_id, row in source_rows_by_item.items():
        if item_id in selected_ids:
            continue
        behavior = row.get("behavior")
        if behavior in selected_behaviors:
            by_behavior[behavior].append(row)

    neutral: list[dict[str, Any]] = []
    for behavior, candidates in sorted(by_behavior.items()):
        rng.shuffle(candidates)
        candidates.sort(
            key=lambda row: (
                int(row.get("pair_quality", 0) or 0),
                float(row.get("pair_quality_mean", 0.0) or 0.0),
            ),
            reverse=True,
        )
        take = min(max_neutral_per_behavior, max(1, selected_behaviors[behavior]))
        for row in candidates[:take]:
            out = dict(row)
            out["reverse_subset_role"] = "neutral_matched"
            neutral.append(out)
    return neutral


def main() -> None:
    ap = argparse.ArgumentParser(description="Materialize source rows for reverse-compassion follow-up generation.")
    ap.add_argument("--patch-records", type=Path, default=Path(DEFAULT_PATCH_RECORDS))
    ap.add_argument("--source-dataset-dir", type=Path, default=Path(DEFAULT_SOURCE_DATASET_DIR))
    ap.add_argument("--output-root", type=Path, default=Path(DEFAULT_OUTPUT_ROOT))
    ap.add_argument("--tag", default="symphonic_reverse_subset_v1")
    ap.add_argument("--pairs", default="hitchens:jesus,hitchens:mother_teresa")
    ap.add_argument("--min-pair-quality", type=int, default=3)
    ap.add_argument("--failure-lift-max", type=float, default=0.01)
    ap.add_argument("--partial-lift-min", type=float, default=0.01)
    ap.add_argument("--max-neutral-per-behavior", type=int, default=2)
    ap.add_argument("--seed", type=int, default=17)
    args = ap.parse_args()

    stamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    out_dir = args.output_root / f"{args.tag}_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    pairs = parse_pairs(args.pairs)
    patch_records = load_jsonl(args.patch_records)
    selected_patch_records = select_reverse_records(
        patch_records,
        pairs=pairs,
        failure_lift_max=args.failure_lift_max,
        partial_lift_min=args.partial_lift_min,
    )
    selected_ids = {row["merged_item_id"] for row in selected_patch_records}

    source_rows = load_jsonl(args.source_dataset_dir / "all_completions.jsonl")
    source_rows_by_item = best_source_rows(source_rows, min_pair_quality=args.min_pair_quality)

    selected_source_rows: list[dict[str, Any]] = []
    missing_ids: list[str] = []
    reason_by_item: dict[str, list[str]] = defaultdict(list)
    for row in selected_patch_records:
        reason_by_item[row["merged_item_id"]].append(row["selection_reason"])
    for item_id in sorted(selected_ids):
        source_row = source_rows_by_item.get(item_id)
        if source_row is None:
            missing_ids.append(item_id)
            continue
        out = dict(source_row)
        out["reverse_subset_role"] = "reverse_selected"
        out["reverse_selection_reasons"] = sorted(set(reason_by_item[item_id]))
        selected_source_rows.append(out)

    selected_behaviors = Counter(row["behavior"] for row in selected_source_rows)
    neutral_rows = add_neutral_matches(
        source_rows_by_item=source_rows_by_item,
        selected_ids={row["merged_item_id"] for row in selected_source_rows},
        selected_behaviors=selected_behaviors,
        max_neutral_per_behavior=args.max_neutral_per_behavior,
        seed=args.seed,
    )

    all_rows = selected_source_rows + neutral_rows
    all_rows.sort(
        key=lambda row: (
            row.get("reverse_subset_role", ""),
            row.get("behavior", ""),
            row.get("merged_item_id", ""),
        )
    )

    write_jsonl(out_dir / "all_completions.jsonl", all_rows)
    write_jsonl(out_dir / "selected_patch_records.jsonl", selected_patch_records)
    write_jsonl(out_dir / "neutral_matched_rows.jsonl", neutral_rows)
    write_json(
        out_dir / "manifest.json",
        {
            "started_at": now_iso(),
            "patch_records": str(args.patch_records),
            "source_dataset_dir": str(args.source_dataset_dir),
            "pairs": sorted([f"{src}:{dst}" for src, dst in pairs]),
            "min_pair_quality": args.min_pair_quality,
            "failure_lift_max": args.failure_lift_max,
            "partial_lift_min": args.partial_lift_min,
            "max_neutral_per_behavior": args.max_neutral_per_behavior,
            "seed": args.seed,
        },
    )
    write_json(
        out_dir / "summary.json",
        {
            "finished_at": now_iso(),
            "n_selected_patch_records": len(selected_patch_records),
            "n_selected_unique_source_rows": len(selected_source_rows),
            "n_neutral_matched_rows": len(neutral_rows),
            "n_total_source_rows": len(all_rows),
            "missing_source_item_ids": missing_ids,
            "selected_by_behavior": dict(sorted(selected_behaviors.items())),
            "selected_by_reason": dict(sorted(Counter(row["selection_reason"] for row in selected_patch_records).items())),
        },
    )
    print(out_dir)


if __name__ == "__main__":
    main()
