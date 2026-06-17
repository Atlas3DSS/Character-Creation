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
DEFAULT_ENDPOINTS = "http://192.168.1.90:30001/v1,http://192.168.1.90:30002/v1"


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


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def read_trace_eval(trace_dir: Path, limit: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for p in sorted(trace_dir.glob("records_shard_*.jsonl")):
        for row in load_jsonl(p):
            rows.append(row)
            if len(rows) >= limit:
                return rows
    return rows


def strip_output_contract(prompt: str) -> str:
    return re.sub(r"Output exactly three sections.*?Do not emit 'Thinking Process:'\.", "", prompt, flags=re.S).strip()


def messages_for(row: dict[str, Any], condition: str) -> list[dict[str, str]]:
    system = row.get("system_prompt") or "Follow the requested format."
    core = strip_output_contract(row.get("prompt_text") or "")
    if condition == "think_only":
        instr = (
            "Output exactly these sections in order and nothing before them:\n"
            "/think\n"
            "<brief in-character reasoning>\n"
            "/end-think\n"
            "Explanation: <one short sentence>\n"
            "Final Answer: <canonical short answer only>"
        )
    elif condition == "meta_think_plus_think":
        instr = (
            "Output exactly these sections in order and nothing before them:\n"
            "/meta-think\n"
            "<2-5 short lines about identity constraints, task constraints, reasoning risk, and response plan only>\n"
            "/end-meta-think\n"
            "/think\n"
            "<brief in-character reasoning>\n"
            "/end-think\n"
            "Explanation: <one short sentence>\n"
            "Final Answer: <canonical short answer only>"
        )
    else:
        raise ValueError(f"unknown condition: {condition}")
    return [{"role": "system", "content": system}, {"role": "user", "content": core + "\n\n" + instr + "\nDo not emit 'Thinking Process:'."}]


def normalize_answer(text: str) -> str:
    return re.sub(r"[^a-z0-9.$]+", " ", text.lower()).strip()


def extract_final(text: str) -> str | None:
    m = re.search(r"Final Answer\s*:\s*(.+)", text, flags=re.I)
    if not m:
        return None
    return m.group(1).strip().splitlines()[0].strip()


def score_answer(text: str, key: str | None) -> bool | None:
    if not key:
        return None
    final = extract_final(text)
    if final is None:
        return False
    f = normalize_answer(final)
    k = normalize_answer(key)
    return bool(k and (f == k or k in f or f in k))


def task_family(task_id: str | None) -> str:
    parts = (task_id or "").split(":")
    return parts[1] if len(parts) > 1 else "unknown"


def call_openai(base_url: str, model: str, messages: list[dict[str, str]], max_tokens: int, timeout: int) -> tuple[str, dict[str, Any]]:
    resp = requests.post(
        base_url.rstrip("/") + "/chat/completions",
        headers={"Authorization": "Bearer none", "Content-Type": "application/json"},
        json={
            "model": model,
            "messages": messages,
            "temperature": 0,
            "max_tokens": max_tokens,
            "chat_template_kwargs": {"enable_thinking": False},
        },
        timeout=timeout,
    )
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"].get("content") or "", data.get("usage", {})


def summarize(records_path: Path) -> dict[str, Any]:
    rows = load_jsonl(records_path)
    out: dict[str, Any] = {"n_records": len(rows), "by_condition": {}, "by_task_family": {}, "paired_patterns": {}}
    for condition in sorted({r["condition"] for r in rows}):
        vals = [r for r in rows if r["condition"] == condition]
        scored = [r for r in vals if r.get("correct") is not None]
        toks = [(r.get("usage") or {}).get("completion_tokens") for r in vals if (r.get("usage") or {}).get("completion_tokens") is not None]
        out["by_condition"][condition] = {
            "n": len(vals),
            "errors": sum(bool(r.get("error")) for r in vals),
            "format_ok": sum(bool(r.get("format_ok")) for r in vals) / max(len(vals), 1),
            "thinking_process_leak": sum(bool(r.get("thinking_process_leak")) for r in vals) / max(len(vals), 1),
            "truncated": sum(bool(r.get("truncated")) for r in vals) / max(len(vals), 1),
            "reasoning_accuracy": sum(bool(r.get("correct")) for r in scored) / max(len(scored), 1) if scored else None,
            "scored": len(scored),
            "mean_latency_s": statistics.mean([r["latency_s"] for r in vals]) if vals else None,
            "mean_completion_tokens": statistics.mean(toks) if toks else None,
        }
    for condition in sorted({r["condition"] for r in rows}):
        out["by_task_family"][condition] = {}
        for family in sorted({task_family(r.get("task_id")) for r in rows}):
            vals = [r for r in rows if r["condition"] == condition and task_family(r.get("task_id")) == family]
            if not vals:
                continue
            out["by_task_family"][condition][family] = {
                "n": len(vals),
                "correct": sum(bool(r.get("correct")) for r in vals),
                "accuracy": sum(bool(r.get("correct")) for r in vals) / len(vals),
            }
    by_task: dict[str, dict[str, bool | None]] = defaultdict(dict)
    for r in rows:
        by_task[r["task_id"]][r["condition"]] = r.get("correct")
    patterns = Counter((vals.get("think_only"), vals.get("meta_think_plus_think")) for vals in by_task.values())
    out["paired_patterns"] = {str(k): v for k, v in patterns.items()}
    out["marginal_meta_fixes"] = [
        task_id for task_id, vals in by_task.items()
        if vals.get("think_only") is False and vals.get("meta_think_plus_think") is True
    ]
    out["marginal_meta_regressions"] = [
        task_id for task_id, vals in by_task.items()
        if vals.get("think_only") is True and vals.get("meta_think_plus_think") is False
    ]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace-eval-dir", type=Path, default=Path(DEFAULT_TRACE_EVAL))
    ap.add_argument("--output-dir", type=Path, default=Path("sweep_v4/meta_vs_think_phase_isolation_qwen35_20260416"))
    ap.add_argument("--limit", type=int, default=48)
    ap.add_argument("--endpoints", default=DEFAULT_ENDPOINTS)
    ap.add_argument("--model", default="Qwen/Qwen3.5-9B")
    ap.add_argument("--concurrency", type=int, default=32)
    ap.add_argument("--max-tokens", type=int, default=900)
    ap.add_argument("--timeout", type=int, default=240)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    records_path = args.output_dir / "records.jsonl"
    if records_path.exists() and args.overwrite:
        records_path.unlink()
    write_json(args.output_dir / "manifest.json", {
        "started_at": now_iso(),
        "trace_eval_dir": str(args.trace_eval_dir),
        "limit": args.limit,
        "endpoints": args.endpoints,
        "model": args.model,
        "conditions": ["think_only", "meta_think_plus_think"],
    })
    rows = [r for r in read_trace_eval(args.trace_eval_dir, args.limit) if r.get("track") == "reasoning"]
    rows = rows[:args.limit]
    endpoints = [e.strip() for e in args.endpoints.split(",") if e.strip()]
    tasks = [(i, row, condition) for i, row in enumerate(rows) for condition in ("think_only", "meta_think_plus_think")]

    def run_one(i: int, row: dict[str, Any], condition: str) -> dict[str, Any]:
        endpoint = endpoints[i % len(endpoints)]
        started = time.time()
        try:
            text, usage = call_openai(endpoint, args.model, messages_for(row, condition), args.max_tokens, args.timeout)
            error = None
        except Exception as exc:  # noqa: BLE001
            text, usage, error = "", {}, repr(exc)
        latency = time.time() - started
        low = text.lower()
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
            "text": text,
            "error": error,
            "final_answer": extract_final(text),
            "correct": score_answer(text, row.get("answer_key")),
            "format_ok": ("final answer:" in low) and ("/think" in low) and (condition == "think_only" or "/meta-think" in low),
            "thinking_process_leak": "thinking process:" in low,
            "truncated": bool(text and not re.search(r"Final Answer\s*:", text, flags=re.I)),
        }

    completed = 0
    with cf.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = [pool.submit(run_one, i, row, condition) for i, row, condition in tasks]
        for fut in cf.as_completed(futures):
            append_jsonl(records_path, fut.result())
            completed += 1
            if completed % 24 == 0:
                summary = summarize(records_path)
                write_json(args.output_dir / "summary.partial.json", summary)
                print(f"[{now_iso()}] progress {completed}/{len(tasks)}", flush=True)
    summary = summarize(records_path)
    write_json(args.output_dir / "summary.json", summary)
    (args.output_dir / "DONE").write_text(now_iso() + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
