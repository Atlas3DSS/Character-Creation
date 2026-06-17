#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any


DEFAULT_DATASET_DIR = "/home/orwel/dev_genius/experiments/Character Creation/sweep_v4/symphonic_voice_probe_dataset_v2_20260418_072100"
DEFAULT_ANCHOR_MANIFEST = "/home/orwel/dev_genius/experiments/Character Creation/data/symphonic_voice_anchor_manifest_v2.json"
DEFAULT_BASE_URL = "http://127.0.0.1:30003/v1"
DEFAULT_API_MODEL = "/home/orwel/dev_genius/models/Qwen3.6-35B-A3B"
DEFAULT_OUTPUT_ROOT = "/home/orwel/dev_genius/experiments/Character Creation/sweep_v4"
DEFAULT_TAG = "symphonic_voice_probe_dataset_v2_repaired"


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_builder_module() -> Any:
    script_path = Path("/home/orwel/dev_genius/experiments/Character Creation/scripts/experiments/personality/build_symphonic_probe_dataset.py")
    spec = importlib.util.spec_from_file_location("build_symphonic_probe_dataset", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module from {script_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def build_pair_index(rows: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    return {(row["merged_item_id"], row["anchor_id"]): row for row in rows}


def soft_length_messages(messages: list[dict[str, str]], think_max_words: int, response_max_words: int) -> list[dict[str, str]]:
    msg_list = [dict(m) for m in messages]
    if not msg_list:
        return msg_list
    msg_list[-1]["content"] = (
        msg_list[-1]["content"].rstrip()
        + "\n\nLength preference:\n"
        + f"- Keep the /think block under about {think_max_words} words.\n"
        + f"- Keep the Response under about {response_max_words} words.\n"
        + "- Prefer compression over repetition.\n"
        + "- Preserve stance and competence while staying concise.\n"
    )
    return msg_list


def generate_repair(
    *,
    builder: Any,
    base_url: str,
    api_model: str,
    timeout: int,
    messages: list[dict[str, str]],
    row_meta: dict[str, Any],
    anchor: dict[str, Any],
) -> dict[str, Any]:
    last_err: str | None = None
    temps = (0.45, 0.30, 0.20)
    msg_variants = [
        messages,
        soft_length_messages(messages, think_max_words=180, response_max_words=220),
        soft_length_messages(messages, think_max_words=140, response_max_words=180),
    ]
    for msg_variant in msg_variants:
        for temperature in temps:
            try:
                raw_text, usage = builder.chat_generate_api(
                    base_url,
                    api_model,
                    msg_variant,
                    max_new_tokens=1800,
                    temperature=temperature,
                    top_p=0.95,
                    top_k=40,
                    timeout=timeout,
                )
            except Exception as exc:  # noqa: BLE001
                last_err = f"{type(exc).__name__}: {exc}"
                continue
            parsed = builder.parse_completion(raw_text)
            think_words = len(builder.re.findall(r"\w+", parsed["think"]))
            response_words = len(builder.re.findall(r"\w+", parsed["response"]))
            leak = builder.anchor_leak(parsed["assistant_completion"], anchor)
            if (
                parsed["format_ok"]
                and think_words >= 20
                and response_words >= 25
                and not leak
                and think_words <= 320
                and response_words <= 320
            ):
                return {
                    **row_meta,
                    "messages": msg_variant,
                    "assistant_completion": parsed["assistant_completion"],
                    "format_ok": True,
                    "anchor_leak": False,
                    "n_think_words": think_words,
                    "n_response_words": response_words,
                    "temperature": temperature,
                    "usage": usage,
                }
            last_err = (
                f"format_ok={parsed['format_ok']} think_words={think_words} "
                f"response_words={response_words} anchor_leak={leak}"
            )
    raise RuntimeError(last_err or "repair generation failed")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-dir", type=Path, default=Path(DEFAULT_DATASET_DIR))
    ap.add_argument("--anchor-manifest", type=Path, default=Path(DEFAULT_ANCHOR_MANIFEST))
    ap.add_argument("--base-url", default=DEFAULT_BASE_URL)
    ap.add_argument("--api-model", default=DEFAULT_API_MODEL)
    ap.add_argument("--output-root", type=Path, default=Path(DEFAULT_OUTPUT_ROOT))
    ap.add_argument("--tag", default=DEFAULT_TAG)
    ap.add_argument("--timeout", type=int, default=1200)
    ap.add_argument("--think-threshold", type=int, default=320)
    ap.add_argument("--response-threshold", type=int, default=320)
    args = ap.parse_args()

    stamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    out_dir = args.output_root / f"{args.tag}_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    builder = load_builder_module()
    manifest = load_json(args.anchor_manifest)
    anchors = {a["anchor_id"]: a for a in manifest["anchors"]}
    rows = load_jsonl(args.dataset_dir / "all_completions.jsonl")
    failures = load_jsonl(args.dataset_dir / "failures.jsonl")
    base_rows = {row["merged_item_id"]: row for row in load_jsonl(args.dataset_dir / "base_rows.jsonl")}
    pair_index = build_pair_index(rows)
    split_map = {}
    for row in rows:
        split_map.setdefault(row["merged_item_id"], row["split"])

    repair_targets: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for failure in failures:
        key = (failure["merged_item_id"], failure["anchor_id"])
        if key not in seen:
            seen.add(key)
            repair_targets.append({"kind": "failure", **failure})
    for row in rows:
        if row["n_think_words"] > args.think_threshold or row["n_response_words"] > args.response_threshold:
            key = (row["merged_item_id"], row["anchor_id"])
            if key not in seen:
                seen.add(key)
                repair_targets.append(
                    {
                        "kind": "outlier",
                        "merged_item_id": row["merged_item_id"],
                        "anchor_id": row["anchor_id"],
                        "old_think_words": row["n_think_words"],
                        "old_response_words": row["n_response_words"],
                    }
                )

    repaired_rows: list[dict[str, Any]] = []
    failed_repairs: list[dict[str, Any]] = []
    for target in repair_targets:
        merged_item_id = target["merged_item_id"]
        anchor_id = target["anchor_id"]
        anchor = anchors[anchor_id]
        split = split_map[merged_item_id]
        base = base_rows[merged_item_id]
        row_meta = {
            "split": split,
            "anchor_id": anchor["anchor_id"],
            "anchor_display_name": anchor["display_name"],
            "anchor_axes": anchor["stance_axes"],
            "behavior": base["behavior"],
            "merged_item_id": base["merged_item_id"],
            "source_title": base["source_title"],
            "source_group": base["source_group"],
            "title": base["title"],
            "focal_character": base["focal_character"],
            "counterpart": base["counterpart"],
            "scene_summary": base["scene_summary"],
            "emotional_state": base["emotional_state"],
            "hidden_conflict": base["hidden_conflict"],
            "carryover_target": base["carryover_target"],
            "source_example_key": base["example_key"],
            "source_pair_quality": int(base.get("pair_quality", 0) or 0),
            "source_metric_ids": base.get("metric_ids", []),
        }
        messages = builder.build_messages(anchor, base)
        try:
            repaired = generate_repair(
                builder=builder,
                base_url=args.base_url,
                api_model=args.api_model,
                timeout=args.timeout,
                messages=messages,
                row_meta=row_meta,
                anchor=anchor,
            )
        except Exception as exc:  # noqa: BLE001
            failed_repairs.append({**target, "error": f"{type(exc).__name__}: {exc}"})
            continue
        repaired_rows.append({**target, "row": repaired})

    merged_rows = list(rows)
    for item in repaired_rows:
        row = item["row"]
        key = (row["merged_item_id"], row["anchor_id"])
        if key in pair_index:
            old = pair_index[key]
            idx = merged_rows.index(old)
            merged_rows[idx] = row
        else:
            merged_rows.append(row)
        pair_index[key] = row

    merged_rows.sort(key=lambda row: (row["split"], row["behavior"], row["merged_item_id"], row["anchor_id"]))
    by_split: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in merged_rows:
        by_split[row["split"]].append(row)

    write_json(
        out_dir / "manifest.json",
        {
            "started_from": str(args.dataset_dir),
            "anchor_manifest": str(args.anchor_manifest),
            "base_url": args.base_url,
            "api_model": args.api_model,
            "think_threshold": args.think_threshold,
            "response_threshold": args.response_threshold,
            "n_original_rows": len(rows),
            "n_targets": len(repair_targets),
            "n_repairs_succeeded": len(repaired_rows),
            "n_repairs_failed": len(failed_repairs),
        },
    )
    write_jsonl(out_dir / "repair_targets.jsonl", repair_targets)
    write_jsonl(out_dir / "repairs_applied.jsonl", repaired_rows)
    write_jsonl(out_dir / "repair_failures.jsonl", failed_repairs)
    write_jsonl(out_dir / "all_completions.jsonl", merged_rows)
    for split, split_rows in by_split.items():
        write_jsonl(out_dir / f"{split}.jsonl", split_rows)

    # carry forward source metadata
    for name in ("anchors.json", "base_rows.jsonl"):
        src = args.dataset_dir / name
        if src.exists():
            (out_dir / name).write_text(src.read_text(encoding="utf-8"), encoding="utf-8")

    # recompute summary
    from collections import Counter

    by_behavior = Counter(row["behavior"] for row in merged_rows)
    by_anchor = Counter(row["anchor_id"] for row in merged_rows)
    by_split_counter = Counter(row["split"] for row in merged_rows)
    summary = {
        "finished_at": now_iso(),
        "source_dataset_dir": str(args.dataset_dir),
        "n_success": len(merged_rows),
        "n_failures_remaining": len(failed_repairs),
        "by_behavior": dict(sorted(by_behavior.items())),
        "by_anchor": dict(sorted(by_anchor.items())),
        "by_split": dict(sorted(by_split_counter.items())),
        "format_ok_rate": 1.0,
        "mean_think_words": float(sum(row["n_think_words"] for row in merged_rows) / max(len(merged_rows), 1)),
        "mean_response_words": float(sum(row["n_response_words"] for row in merged_rows) / max(len(merged_rows), 1)),
        "mean_tokens_per_s": float(sum(float(row["usage"]["tokens_per_s"]) for row in merged_rows) / max(len(merged_rows), 1)),
    }
    write_json(out_dir / "summary.json", summary)


if __name__ == "__main__":
    main()
