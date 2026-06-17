#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import random
import re
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import requests


DEFAULT_SOURCE_DATASET_DIR = "/home/orwel/dev_genius/experiments/Character Creation/sweep_v4/book_character_prefill_dataset_balanced_v6_reviewed_20260417_151017"
DEFAULT_ANCHOR_MANIFEST = "/home/orwel/dev_genius/experiments/Character Creation/data/symphonic_voice_anchor_manifest_v1.json"
DEFAULT_BASE_URL = "http://127.0.0.1:30003/v1"
DEFAULT_API_MODEL = "/home/orwel/dev_genius/models/Qwen3.6-35B-A3B"
DEFAULT_OUTPUT_ROOT = "/home/orwel/dev_genius/experiments/Character Creation/sweep_v4"
DEFAULT_TAG = "symphonic_voice_probe_dataset_v1"


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def log(log_path: Path, msg: str) -> None:
    line = f"[{now_iso()}] {msg}"
    print(line, flush=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


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


def canonical_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def doom_loop_reason(text: str) -> str | None:
    words = re.findall(r"\w+", text.lower())
    if len(text) < 2000 and len(words) < 350:
        return None
    recent_words = words[-240:]
    if len(recent_words) >= 180:
        uniq_ratio = len(set(recent_words)) / max(len(recent_words), 1)
        if uniq_ratio < 0.18:
            return f"low_word_novelty:{uniq_ratio:.3f}"
    recent_lines = [canonical_text(line) for line in text.splitlines() if line.strip()]
    if len(recent_lines) >= 6 and len(set(recent_lines[-6:])) <= 2:
        return "repeated_lines"
    tail = canonical_text(text[-120:])
    window = canonical_text(text[-1600:])
    if len(tail) >= 60 and window.count(tail) >= 4:
        return "repeated_suffix"
    return None


def chat_generate_api(
    base_url: str,
    api_model: str,
    messages: list[dict[str, str]],
    *,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    timeout: int,
) -> tuple[str, dict[str, Any]]:
    payload = {
        "model": api_model,
        "messages": messages,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "max_tokens": max_new_tokens,
        "chat_template_kwargs": {"enable_thinking": False},
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    t0 = time.time()
    resp = requests.post(
        base_url.rstrip("/") + "/chat/completions",
        headers={"Authorization": "Bearer none", "Content-Type": "application/json"},
        json=payload,
        timeout=(20, timeout),
        stream=True,
    )
    resp.raise_for_status()
    text_parts: list[str] = []
    usage: dict[str, Any] = {}
    last_check_words = 0
    aborted = False
    abort_reason: str | None = None
    for raw_line in resp.iter_lines(decode_unicode=True):
        if not raw_line or not raw_line.startswith("data: "):
            continue
        data = raw_line[6:]
        if data == "[DONE]":
            break
        event = json.loads(data)
        if event.get("usage"):
            usage = event["usage"]
        choices = event.get("choices") or []
        if not choices:
            continue
        delta = choices[0].get("delta") or {}
        piece = delta.get("content") or ""
        if piece:
            text_parts.append(piece)
            joined = "".join(text_parts)
            word_count = len(re.findall(r"\w+", joined))
            if (len(joined) >= 2000 or word_count >= 350) and (word_count - last_check_words >= 80):
                last_check_words = word_count
                abort_reason = doom_loop_reason(joined)
                if abort_reason:
                    aborted = True
                    resp.close()
                    break
    dt = time.time() - t0
    text = "".join(text_parts)
    comp = int(usage.get("completion_tokens") or 0)
    if comp <= 0:
        comp = len(re.findall(r"\w+", text))
    return text, {
        "generated_tokens": comp,
        "latency_s": dt,
        "tokens_per_s": float(comp / max(dt, 1e-9)),
        "aborted": aborted,
        "abort_reason": abort_reason,
    }


def parse_completion(text: str) -> dict[str, Any]:
    text = text.strip()
    m = re.search(r"/think\s*(.*?)\s*/end-think\s*Response:\s*(.*)\Z", text, flags=re.S | re.I)
    if not m:
        return {"format_ok": False, "think": "", "response": "", "assistant_completion": text}
    think = m.group(1).strip()
    response = m.group(2).strip()
    assistant_completion = f"/think\n{think}\n/end-think\nResponse: {response}"
    return {
        "format_ok": True,
        "think": think,
        "response": response,
        "assistant_completion": assistant_completion,
    }


def anchor_leak(text: str, anchor: dict[str, Any]) -> bool:
    lowered = canonical_text(text)
    needles = {
        anchor["anchor_id"].replace("_", " "),
        canonical_text(anchor["display_name"]),
    }
    needles |= {canonical_text(alias) for alias in anchor.get("aliases", []) if canonical_text(alias)}
    if anchor["anchor_id"] == "mother_teresa":
        needles |= {"mother teresa", "teresa"}
    elif anchor["anchor_id"] == "jesus":
        needles |= {"jesus", "christ"}
    elif anchor["anchor_id"] == "hitchens":
        needles |= {"hitchens", "christopher"}
    elif anchor["anchor_id"] == "linus":
        needles |= {"linus", "torvalds"}
    elif anchor["anchor_id"] == "mark_twain":
        needles |= {"mark twain", "twain", "samuel clemens", "clemens"}
    elif anchor["anchor_id"] == "fred_rogers":
        needles |= {"fred rogers", "mister rogers", "mr rogers", "rogers"}
    elif anchor["anchor_id"] == "george_orwell":
        needles |= {"george orwell", "orwell"}
    return any(needle and needle in lowered for needle in needles)


def build_base_scenario(row: dict[str, Any]) -> str:
    metrics = "\n".join(f"- {m['description']}" for m in row.get("metrics", []))
    counterpart = row.get("counterpart") or "the other person in the scene"
    return (
        "Scenario:\n"
        f"- Situation: {row['scene_summary']}\n"
        f"- You are responding to: {counterpart}\n"
        f"- Current internal state: {row['emotional_state']}\n"
        f"- Hidden conflict: {row['hidden_conflict']}\n"
        f"- Carryover pressure: {row['carryover_target']}\n"
        f"- Behavior emphasis: {row['behavior']}\n"
        "What the reply should preserve:\n"
        f"{metrics}\n"
    )


def build_messages(anchor: dict[str, Any], row: dict[str, Any]) -> list[dict[str, str]]:
    axes = ", ".join(f"{k}={v:.2f}" for k, v in anchor["stance_axes"].items())
    style_notes = "\n".join(f"- {line}" for line in anchor["stance_notes"])
    gating = "\n".join(f"- {line}" for line in anchor["gating_rules"])
    avoid = "\n".join(f"- {line}" for line in anchor["avoid"])
    system = (
        "You are generating a contrastive stance completion for mechanistic probing.\n"
        "Use the supplied stance profile as a latent prior, not as cosplay.\n"
        "Do not mention any source figure, school, scripture, lecture, speech, email, or transcript.\n"
        "Do not quote signature phrases.\n"
        "Preserve scenario facts, competence, and social discrimination.\n"
        "Apply the stance selectively: be softer toward vulnerability and sharper toward hypocrisy, obstruction, or status games only when that fits the stance profile.\n"
        "Output exactly in this format:\n"
        "/think\n"
        "<brief internal stance trace>\n"
        "/end-think\n"
        "Response: <the actual reply>\n"
        "Keep the trace concise, concrete, and non-theatrical."
    )
    user = (
        "Stance profile:\n"
        f"- Latent axes: {axes}\n"
        "Driving traits:\n"
        f"{style_notes}\n"
        "Social gating:\n"
        f"{gating}\n"
        "Avoid:\n"
        f"{avoid}\n\n"
        f"{build_base_scenario(row)}\n"
        "Task:\n"
        "Write the next reply. The response should feel genuinely shaped by the stance profile, not merely reworded or decorated."
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def select_base_rows(
    rows: list[dict[str, Any]],
    *,
    items_per_behavior: int,
    min_pair_quality: int,
    seed: int,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    candidates = [
        row
        for row in rows
        if int(row.get("label", 0)) == 1
        and bool(row.get("judge_format_ok", False))
        and int(row.get("pair_quality", 0) or 0) >= min_pair_quality
    ]
    best_by_item: dict[str, dict[str, Any]] = {}
    for row in candidates:
        key = row["merged_item_id"]
        prev = best_by_item.get(key)
        if prev is None or int(row.get("pair_quality", 0) or 0) > int(prev.get("pair_quality", 0) or 0):
            best_by_item[key] = row
    by_behavior: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in best_by_item.values():
        by_behavior[row["behavior"]].append(row)
    selected: list[dict[str, Any]] = []
    for behavior, bucket in sorted(by_behavior.items()):
        bucket = list(bucket)
        rng.shuffle(bucket)
        bucket.sort(key=lambda row: (int(row.get("pair_quality", 0) or 0), float(row.get("pair_quality_mean", 0.0) or 0.0)), reverse=True)
        take = bucket[:items_per_behavior]
        selected.extend(take)
    return selected


def assign_splits(rows: list[dict[str, Any]], seed: int) -> dict[str, str]:
    rng = random.Random(seed + 17)
    by_behavior: dict[str, list[str]] = defaultdict(list)
    seen: set[str] = set()
    for row in rows:
        key = row["merged_item_id"]
        if key in seen:
            continue
        seen.add(key)
        by_behavior[row["behavior"]].append(key)
    split_map: dict[str, str] = {}
    for behavior, keys in sorted(by_behavior.items()):
        rng.shuffle(keys)
        n = len(keys)
        n_val = max(1, round(n * 0.1))
        n_test = max(1, round(n * 0.1))
        if n_val + n_test >= n:
            n_val = 1
            n_test = 1 if n >= 3 else max(0, n - 2)
        for idx, key in enumerate(keys):
            if idx < n_val:
                split_map[key] = "val"
            elif idx < n_val + n_test:
                split_map[key] = "test"
            else:
                split_map[key] = "train"
    return split_map


def generate_one(
    *,
    base_url: str,
    api_model: str,
    timeout: int,
    anchor: dict[str, Any],
    row: dict[str, Any],
    split: str,
) -> dict[str, Any]:
    messages = build_messages(anchor, row)
    last_err: str | None = None
    for temperature in (0.8, 0.45):
        try:
            raw_text, usage = chat_generate_api(
                base_url,
                api_model,
                messages,
                max_new_tokens=1800,
                temperature=temperature,
                top_p=0.95,
                top_k=40,
                timeout=timeout,
            )
        except Exception as exc:  # noqa: BLE001
            last_err = f"{type(exc).__name__}: {exc}"
            continue
        parsed = parse_completion(raw_text)
        think_words = len(re.findall(r"\w+", parsed["think"]))
        response_words = len(re.findall(r"\w+", parsed["response"]))
        leak = anchor_leak(parsed["assistant_completion"], anchor)
        if parsed["format_ok"] and think_words >= 20 and response_words >= 25 and not leak:
            return {
                "split": split,
                "anchor_id": anchor["anchor_id"],
                "anchor_display_name": anchor["display_name"],
                "anchor_axes": anchor["stance_axes"],
                "behavior": row["behavior"],
                "merged_item_id": row["merged_item_id"],
                "source_title": row["source_title"],
                "source_group": row["source_group"],
                "title": row["title"],
                "focal_character": row["focal_character"],
                "counterpart": row["counterpart"],
                "scene_summary": row["scene_summary"],
                "emotional_state": row["emotional_state"],
                "hidden_conflict": row["hidden_conflict"],
                "carryover_target": row["carryover_target"],
                "messages": messages,
                "assistant_completion": parsed["assistant_completion"],
                "format_ok": bool(parsed["format_ok"]),
                "anchor_leak": leak,
                "n_think_words": think_words,
                "n_response_words": response_words,
                "temperature": temperature,
                "usage": usage,
                "source_example_key": row["example_key"],
                "source_pair_quality": int(row.get("pair_quality", 0) or 0),
                "source_metric_ids": row.get("metric_ids", []),
            }
        last_err = f"format_ok={parsed['format_ok']} think_words={think_words} response_words={response_words} anchor_leak={leak}"
    raise RuntimeError(last_err or "generation failed")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-dataset-dir", type=Path, default=Path(DEFAULT_SOURCE_DATASET_DIR))
    ap.add_argument("--anchor-manifest", type=Path, default=Path(DEFAULT_ANCHOR_MANIFEST))
    ap.add_argument("--base-url", default=DEFAULT_BASE_URL)
    ap.add_argument("--api-model", default=DEFAULT_API_MODEL)
    ap.add_argument("--output-root", type=Path, default=Path(DEFAULT_OUTPUT_ROOT))
    ap.add_argument("--tag", default=DEFAULT_TAG)
    ap.add_argument("--items-per-behavior", type=int, default=12)
    ap.add_argument("--min-pair-quality", type=int, default=4)
    ap.add_argument("--max-workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--timeout", type=int, default=900)
    args = ap.parse_args()

    stamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    out_dir = args.output_root / f"{args.tag}_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "run.log"

    rows = load_jsonl(args.source_dataset_dir / "all_completions.jsonl")
    manifest = load_json(args.anchor_manifest)
    anchors = list(manifest["anchors"])
    base_rows = select_base_rows(
        rows,
        items_per_behavior=args.items_per_behavior,
        min_pair_quality=args.min_pair_quality,
        seed=args.seed,
    )
    split_map = assign_splits(base_rows, args.seed)

    write_json(
        out_dir / "manifest.json",
        {
            "started_at": now_iso(),
            "source_dataset_dir": str(args.source_dataset_dir),
            "anchor_manifest": str(args.anchor_manifest),
            "base_url": args.base_url,
            "api_model": args.api_model,
            "items_per_behavior": args.items_per_behavior,
            "min_pair_quality": args.min_pair_quality,
            "max_workers": args.max_workers,
            "seed": args.seed,
            "n_base_rows": len(base_rows),
            "n_anchors": len(anchors),
            "n_total_requests": len(base_rows) * len(anchors),
        },
    )
    write_json(out_dir / "anchors.json", manifest)
    write_jsonl(out_dir / "base_rows.jsonl", base_rows)

    all_rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    futures: dict[cf.Future[dict[str, Any]], tuple[str, str]] = {}
    with cf.ThreadPoolExecutor(max_workers=args.max_workers) as ex:
        for row in base_rows:
            split = split_map[row["merged_item_id"]]
            for anchor in anchors:
                fut = ex.submit(
                    generate_one,
                    base_url=args.base_url,
                    api_model=args.api_model,
                    timeout=args.timeout,
                    anchor=anchor,
                    row=row,
                    split=split,
                )
                futures[fut] = (row["merged_item_id"], anchor["anchor_id"])
        done = 0
        for fut in cf.as_completed(futures):
            merged_item_id, anchor_id = futures[fut]
            done += 1
            try:
                result = fut.result()
            except Exception as exc:  # noqa: BLE001
                failures.append(
                    {
                        "merged_item_id": merged_item_id,
                        "anchor_id": anchor_id,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
                log(log_path, f"FAILED {done}/{len(futures)} item={merged_item_id} anchor={anchor_id} err={type(exc).__name__}: {exc}")
                continue
            all_rows.append(result)
            log(
                log_path,
                f"ok {done}/{len(futures)} split={result['split']} behavior={result['behavior']} item={merged_item_id} anchor={anchor_id} "
                f"tps={result['usage']['tokens_per_s']:.1f} think={result['n_think_words']} resp={result['n_response_words']}",
            )

    all_rows.sort(key=lambda row: (row["split"], row["behavior"], row["merged_item_id"], row["anchor_id"]))
    write_jsonl(out_dir / "all_completions.jsonl", all_rows)
    write_jsonl(out_dir / "failures.jsonl", failures)
    split_rows = defaultdict(list)
    for row in all_rows:
        split_rows[row["split"]].append(row)
    for split, split_data in split_rows.items():
        write_jsonl(out_dir / f"{split}.jsonl", split_data)

    by_behavior = Counter(row["behavior"] for row in all_rows)
    by_anchor = Counter(row["anchor_id"] for row in all_rows)
    by_split = Counter(row["split"] for row in all_rows)
    format_ok = sum(1 for row in all_rows if row["format_ok"])
    summary = {
        "finished_at": now_iso(),
        "n_base_rows": len(base_rows),
        "n_anchors": len(anchors),
        "n_success": len(all_rows),
        "n_failures": len(failures),
        "by_behavior": dict(sorted(by_behavior.items())),
        "by_anchor": dict(sorted(by_anchor.items())),
        "by_split": dict(sorted(by_split.items())),
        "format_ok_rate": float(format_ok / max(len(all_rows), 1)),
        "mean_think_words": float(sum(row["n_think_words"] for row in all_rows) / max(len(all_rows), 1)),
        "mean_response_words": float(sum(row["n_response_words"] for row in all_rows) / max(len(all_rows), 1)),
        "mean_tokens_per_s": float(sum(float(row["usage"]["tokens_per_s"]) for row in all_rows) / max(len(all_rows), 1)),
    }
    write_json(out_dir / "summary.json", summary)


if __name__ == "__main__":
    main()
