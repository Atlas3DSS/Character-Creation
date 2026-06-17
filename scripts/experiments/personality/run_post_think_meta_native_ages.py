#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import re
import statistics
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import requests


DEFAULT_TRACE_EVAL = "/home/orwel/dev_genius/experiments/Character Creation/sweep_v4/personality_meta_eval_trace_explicit_v1"
DEFAULT_SELECTION = "/home/orwel/dev_genius/experiments/Character Creation/sweep_v4/meta_step_curve_controlled_qwen35_20260416/selection.json"
DEFAULT_ENDPOINTS = "http://192.168.1.90:30001/v1,http://192.168.1.90:30002/v1"
DEFAULT_MODEL = "Qwen/Qwen3.5-9B"

CONDITIONS = ("native_only", "post_meta", "post_sham", "post_generic")

SHAM_META_BLOCK = (
    "/meta-think\n"
    "slot_a: cedar lantern river paper brass window station pocket\n"
    "slot_b: amber thread mirror orchard handle gravel notebook harbor\n"
    "slot_c: velvet compass ladder meadow copper envelope walkway lantern\n"
    "slot_d: marble feather kettle canvas wagon button orchard signal\n"
    "/end-meta-think"
)

GENERIC_META_BLOCK = (
    "/meta-think\n"
    "summary: keep the answer organized and compact\n"
    "check: avoid skipping an arithmetic or ordering step\n"
    "/end-meta-think"
)

META_RE = re.compile(r"(?is)/meta-think\s*(.*?)\s*/end-meta-think")
FINAL_RE = re.compile(r"Final Answer\s*:\s*(.+)", flags=re.I)


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def read_trace_eval(trace_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for p in sorted(trace_dir.glob("records_shard_*.jsonl")):
        rows.extend(load_jsonl(p))
    return [r for r in rows if r.get("track") == "reasoning"]


def task_key(row: dict[str, Any]) -> str:
    return row.get("task_id") or f"{row.get('persona_id')}::{row.get('prompt_id')}"


def strip_output_contract(prompt: str) -> str:
    return re.sub(r"Output exactly three sections.*?Do not emit 'Thinking Process:'\.", "", prompt, flags=re.S).strip()


def make_messages(row: dict[str, Any], condition: str) -> list[dict[str, str]]:
    system = row.get("system_prompt") or "Follow the requested format."
    core = strip_output_contract(row.get("prompt_text") or "")
    if condition == "native_only":
        suffix = (
            "Think normally first. Then output only the final user-facing answer in exactly two lines:\n"
            "Explanation: <one short sentence>\n"
            "Final Answer: <canonical short answer only>\n"
            "Do not emit /meta-think or a visible /think block."
        )
    elif condition == "post_meta":
        suffix = (
            "Think normally first. After the thinking phase is complete, output exactly these visible sections in order:\n"
            "/meta-think\n"
            "summary: <2-4 short lines summarizing the constraint and plan you actually used>\n"
            "check: <one short line naming the main thing you double-checked>\n"
            "/end-meta-think\n"
            "Explanation: <one short sentence>\n"
            "Final Answer: <canonical short answer only>\n"
            "Do not emit a visible /think block after the thinking phase."
        )
    elif condition == "post_sham":
        suffix = (
            "Think normally first. After the thinking phase is complete, copy this block exactly and then finish:\n"
            f"{SHAM_META_BLOCK}\n"
            "Explanation: <one short sentence>\n"
            "Final Answer: <canonical short answer only>\n"
            "Do not emit a visible /think block after the thinking phase."
        )
    elif condition == "post_generic":
        suffix = (
            "Think normally first. After the thinking phase is complete, copy this block exactly and then finish:\n"
            f"{GENERIC_META_BLOCK}\n"
            "Explanation: <one short sentence>\n"
            "Final Answer: <canonical short answer only>\n"
            "Do not emit a visible /think block after the thinking phase."
        )
    else:
        raise ValueError(f"unknown condition: {condition}")
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": core + "\n\n" + suffix},
    ]


def clean_text(text: str) -> str:
    out = text or ""
    for tok in ("<|im_end|>", "<|endoftext|>", "<|im_start|>"):
        out = out.replace(tok, "")
    return out.strip()


def normalize_answer(text: str) -> str:
    return re.sub(r"[^a-z0-9.$]+", " ", text.lower()).strip()


def extract_final(text: str) -> str | None:
    m = FINAL_RE.search(clean_text(text))
    if not m:
        return None
    return m.group(1).strip().splitlines()[0].strip()


def extract_meta(text: str) -> str:
    m = META_RE.search(clean_text(text))
    return m.group(1).strip() if m else ""


def looks_like_native_thinking_prefix(text: str) -> bool:
    stripped = clean_text(text).lstrip()
    if not stripped:
        return False
    if stripped.startswith("<think>") or stripped.startswith("Thinking Process:"):
        return True
    markers = ("**Analyze the Request:**", "Self-Correction", "Final Review:", "Let's", "*Wait,")
    return sum(marker in stripped for marker in markers) >= 2


def score_answer(text: str, key: str | None) -> bool | None:
    if not key:
        return None
    final = extract_final(text)
    if final is None:
        return False
    f = normalize_answer(final)
    k = normalize_answer(key)
    return bool(k and (f == k or k in f or f in k))


def meta_before_final(text: str) -> bool:
    cleaned = clean_text(text).lower()
    meta_pos = cleaned.find("/meta-think")
    final_pos = cleaned.find("final answer:")
    return meta_pos >= 0 and final_pos >= 0 and meta_pos < final_pos


def call_openai(
    base_url: str,
    model: str,
    messages: list[dict[str, str]],
    max_tokens: int,
    timeout: int,
    temperature: float,
    top_p: float,
    top_k: int,
    presence_penalty: float,
) -> tuple[str, dict[str, Any]]:
    resp = requests.post(
        base_url.rstrip("/") + "/chat/completions",
        headers={"Authorization": "Bearer none", "Content-Type": "application/json"},
        json={
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
            "presence_penalty": presence_penalty,
            "top_k": top_k,
            "max_tokens": max_tokens,
            "chat_template_kwargs": {"enable_thinking": True},
        },
        timeout=timeout,
    )
    resp.raise_for_status()
    data = resp.json()
    msg = data["choices"][0]["message"]
    return msg.get("content") or "", data.get("usage", {})


def endpoint_for(endpoints: list[str], row: dict[str, Any]) -> str:
    persona = int(row.get("persona_id") or 0)
    return endpoints[persona % len(endpoints)]


def format_ok(text: str, condition: str) -> bool:
    low = clean_text(text).lower()
    if "final answer:" not in low:
        return False
    if condition == "native_only":
        return "/meta-think" not in low
    if condition == "post_meta":
        return meta_before_final(text)
    if condition == "post_sham":
        return SHAM_META_BLOCK.lower() in low and meta_before_final(text)
    if condition == "post_generic":
        return GENERIC_META_BLOCK.lower() in low and meta_before_final(text)
    return False


def run_request(
    row: dict[str, Any],
    condition: str,
    endpoints: list[str],
    model: str,
    max_tokens: int,
    timeout: int,
    max_attempts: int,
    temperature: float,
    top_p: float,
    top_k: int,
    presence_penalty: float,
) -> dict[str, Any]:
    endpoint = endpoint_for(endpoints, row)
    text = ""
    usage: dict[str, Any] = {}
    error = None
    started = time.time()
    for attempt in range(1, max_attempts + 1):
        try:
            text, usage = call_openai(
                endpoint,
                model,
                make_messages(row, condition),
                max_tokens,
                timeout,
                temperature,
                top_p,
                top_k,
                presence_penalty,
            )
            error = None
            break
        except Exception as exc:  # noqa: BLE001
            error = repr(exc)
            if attempt == max_attempts:
                break
            time.sleep(min(2 * attempt, 5))
    latency = time.time() - started
    cleaned = clean_text(text)
    return {
        "timestamp": now_iso(),
        "condition": condition,
        "task_id": row.get("task_id"),
        "persona_id": row.get("persona_id"),
        "prompt_id": row.get("prompt_id"),
        "answer_key": row.get("answer_key"),
        "endpoint": endpoint,
        "latency_s": latency,
        "usage": usage,
        "text": cleaned,
        "error": error,
        "final_answer": extract_final(cleaned),
        "meta_text": extract_meta(cleaned),
        "correct": score_answer(cleaned, row.get("answer_key")),
        "format_ok": format_ok(cleaned, condition),
        "native_thinking_prefix": looks_like_native_thinking_prefix(cleaned),
        "meta_before_final": meta_before_final(cleaned),
        "truncated": bool(cleaned and not FINAL_RE.search(cleaned)),
    }


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {
        "n_records": len(records),
        "by_condition": {},
        "paired_vs_native_only": {},
    }
    for condition in CONDITIONS:
        vals = [r for r in records if r.get("condition") == condition]
        scored = [r for r in vals if r.get("correct") is not None]
        toks = [(r.get("usage") or {}).get("completion_tokens") for r in vals if (r.get("usage") or {}).get("completion_tokens") is not None]
        out["by_condition"][condition] = {
            "n": len(vals),
            "scored": len(scored),
            "accuracy": sum(bool(r.get("correct")) for r in scored) / max(len(scored), 1) if scored else None,
            "errors": sum(bool(r.get("error")) for r in vals),
            "format_ok": sum(bool(r.get("format_ok")) for r in vals) / max(len(vals), 1),
            "native_thinking_prefix": sum(bool(r.get("native_thinking_prefix")) for r in vals) / max(len(vals), 1),
            "meta_before_final": sum(bool(r.get("meta_before_final")) for r in vals) / max(len(vals), 1),
            "truncated": sum(bool(r.get("truncated")) for r in vals) / max(len(vals), 1),
            "mean_latency_s": statistics.mean([r.get("latency_s", 0.0) for r in vals]) if vals else None,
            "mean_completion_tokens": statistics.mean(toks) if toks else None,
        }
    by_task: dict[str, dict[str, bool | None]] = defaultdict(dict)
    for r in records:
        by_task[r["task_id"]][r["condition"]] = r.get("correct")
    for condition in CONDITIONS:
        if condition == "native_only":
            continue
        patterns = Counter((vals.get("native_only"), vals.get(condition)) for vals in by_task.values())
        out["paired_vs_native_only"][condition] = {
            "patterns": {str(k): v for k, v in patterns.items()},
            "marginal_fixes": sorted(task_id for task_id, vals in by_task.items() if vals.get("native_only") is False and vals.get(condition) is True),
            "marginal_regressions": sorted(task_id for task_id, vals in by_task.items() if vals.get("native_only") is True and vals.get(condition) is False),
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace-eval-dir", type=Path, default=Path(DEFAULT_TRACE_EVAL))
    ap.add_argument("--selection-path", type=Path, default=Path(DEFAULT_SELECTION))
    ap.add_argument("--output-dir", type=Path, default=Path("sweep_v4/post_think_meta_native_ages_pilot_20260416"))
    ap.add_argument("--endpoints", default=DEFAULT_ENDPOINTS)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--limit", type=int, default=12)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=1800)
    ap.add_argument("--timeout", type=int, default=300)
    ap.add_argument("--max-attempts", type=int, default=2)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--presence-penalty", type=float, default=1.5)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    records_path = args.output_dir / "records.jsonl"
    summary_path = args.output_dir / "summary.json"
    if args.overwrite:
        for p in (records_path, summary_path, args.output_dir / "summary.partial.json", args.output_dir / "DONE"):
            p.unlink(missing_ok=True)

    selection = load_json(args.selection_path)
    selected_task_ids = list(selection["selected_task_ids"])
    if args.limit > 0:
        selected_task_ids = selected_task_ids[: args.limit]
    selected_set = set(selected_task_ids)
    rows = [r for r in read_trace_eval(args.trace_eval_dir) if task_key(r) in selected_set]
    rows.sort(key=task_key)
    endpoints = [e.strip() for e in args.endpoints.split(",") if e.strip()]
    manifest = {
        "started_at": now_iso(),
        "trace_eval_dir": str(args.trace_eval_dir),
        "selection_path": str(args.selection_path),
        "output_dir": str(args.output_dir),
        "endpoints": endpoints,
        "model": args.model,
        "conditions": list(CONDITIONS),
        "selected_count": len(rows),
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "presence_penalty": args.presence_penalty,
        "enable_thinking": True,
    }
    write_json(args.output_dir / "manifest.json", manifest)

    existing = load_jsonl(records_path)
    done = {(r["task_id"], r["condition"]) for r in existing}
    tasks = [(row, condition) for row in rows for condition in CONDITIONS if (task_key(row), condition) not in done]

    def one(row: dict[str, Any], condition: str) -> dict[str, Any]:
        return run_request(
            row,
            condition,
            endpoints,
            args.model,
            args.max_tokens,
            args.timeout,
            args.max_attempts,
            args.temperature,
            args.top_p,
            args.top_k,
            args.presence_penalty,
        )

    completed = len(existing)
    with cf.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = [pool.submit(one, row, condition) for row, condition in tasks]
        total = len(existing) + len(futures)
        for fut in cf.as_completed(futures):
            rec = fut.result()
            append_jsonl(records_path, rec)
            completed += 1
            if completed % 8 == 0 or completed == total:
                write_json(args.output_dir / "summary.partial.json", summarize(load_jsonl(records_path)))
                print(f"[{now_iso()}] progress {completed}/{total}", flush=True)

    summary = summarize(load_jsonl(records_path))
    write_json(summary_path, summary)
    (args.output_dir / "DONE").write_text(now_iso() + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
