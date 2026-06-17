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

CONDITIONS = ("think_only", "real_meta", "sham_meta", "generic_prep")

SHAM_META_BLOCK = (
    "/meta-think\n"
    "slot_a: cedar lantern river paper brass window station pocket\n"
    "slot_b: amber thread mirror orchard handle gravel notebook harbor\n"
    "slot_c: velvet compass ladder meadow copper envelope walkway lantern\n"
    "slot_d: marble feather kettle canvas wagon button orchard signal\n"
    "/end-meta-think"
)

GENERIC_PREP_BLOCK = (
    "/prep-think\n"
    "note_a: keep the process orderly and move one step at a time\n"
    "note_b: keep the wording compact and avoid unnecessary detours\n"
    "note_c: preserve internal consistency while moving toward closure\n"
    "note_d: finish with the required explanation and final answer block\n"
    "/end-prep-think"
)


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
    if condition == "think_only":
        instr = (
            "Output exactly these sections in order and nothing before them:\n"
            "/think\n"
            "<brief in-character reasoning>\n"
            "/end-think\n"
            "Explanation: <one short sentence>\n"
            "Final Answer: <canonical short answer only>"
        )
    elif condition == "real_meta":
        instr = (
            "Output exactly these sections in order and nothing before them:\n"
            "/meta-think\n"
            "identity: <compact persona constraint>\n"
            "constraint: <task constraint>\n"
            "reasoning_risk: <main risk>\n"
            "response_policy: <short policy>\n"
            "/end-meta-think\n"
            "/think\n"
            "<brief in-character reasoning>\n"
            "/end-think\n"
            "Explanation: <one short sentence>\n"
            "Final Answer: <canonical short answer only>"
        )
    elif condition == "sham_meta":
        instr = (
            "Output exactly these sections in order and nothing before them.\n"
            "Copy the `/meta-think` block below exactly as written, then continue:\n"
            f"{SHAM_META_BLOCK}\n"
            "/think\n"
            "<brief in-character reasoning>\n"
            "/end-think\n"
            "Explanation: <one short sentence>\n"
            "Final Answer: <canonical short answer only>"
        )
    elif condition == "generic_prep":
        instr = (
            "Output exactly these sections in order and nothing before them.\n"
            "Copy the `/prep-think` block below exactly as written, then continue:\n"
            f"{GENERIC_PREP_BLOCK}\n"
            "/think\n"
            "<brief in-character reasoning>\n"
            "/end-think\n"
            "Explanation: <one short sentence>\n"
            "Final Answer: <canonical short answer only>"
        )
    else:
        raise ValueError(f"unknown condition: {condition}")
    user = core + "\n\n" + instr + "\nDo not emit 'Thinking Process:'."
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


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
    enable_thinking: bool,
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
            "chat_template_kwargs": {"enable_thinking": enable_thinking},
        },
        timeout=timeout,
    )
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"].get("content") or "", data.get("usage", {})


def endpoint_for(endpoints: list[str], row: dict[str, Any]) -> str:
    persona = int(row.get("persona_id") or 0)
    return endpoints[persona % len(endpoints)]


def format_ok(text: str, condition: str) -> bool:
    low = text.lower()
    if "final answer:" not in low or "/think" not in low:
        return False
    if condition == "think_only":
        return "/meta-think" not in low and "/prep-think" not in low
    if condition == "real_meta":
        return "/meta-think" in low and "/prep-think" not in low
    if condition == "sham_meta":
        return SHAM_META_BLOCK.lower() in low
    if condition == "generic_prep":
        return GENERIC_PREP_BLOCK.lower() in low
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
    enable_thinking: bool,
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
                enable_thinking,
            )
            error = None
            break
        except Exception as exc:  # noqa: BLE001
            error = repr(exc)
            if attempt == max_attempts:
                break
            time.sleep(min(2 * attempt, 5))
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
        "format_ok": format_ok(text, condition),
        "thinking_process_leak": "thinking process:" in low,
        "truncated": bool(text and not re.search(r"Final Answer\s*:", text, flags=re.I)),
    }


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {
        "n_records": len(records),
        "by_condition": {},
        "paired_vs_think_only": {},
        "paired_condition_deltas": {},
        "real_meta_unique_fixes_vs_controls": [],
        "real_meta_unique_wins": {},
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
            "thinking_process_leak": sum(bool(r.get("thinking_process_leak")) for r in vals) / max(len(vals), 1),
            "truncated": sum(bool(r.get("truncated")) for r in vals) / max(len(vals), 1),
            "mean_latency_s": statistics.mean([r.get("latency_s", 0.0) for r in vals]) if vals else None,
            "mean_completion_tokens": statistics.mean(toks) if toks else None,
        }
    by_task: dict[str, dict[str, bool | None]] = defaultdict(dict)
    for r in records:
        by_task[r["task_id"]][r["condition"]] = r.get("correct")
    for condition in CONDITIONS:
        if condition == "think_only":
            continue
        patterns = Counter((vals.get("think_only"), vals.get(condition)) for vals in by_task.values())
        out["paired_vs_think_only"][condition] = {
            "patterns": {str(k): v for k, v in patterns.items()},
            "marginal_fixes": sorted(task_id for task_id, vals in by_task.items() if vals.get("think_only") is False and vals.get(condition) is True),
            "marginal_regressions": sorted(task_id for task_id, vals in by_task.items() if vals.get("think_only") is True and vals.get(condition) is False),
        }
    direct_pairs = [("real_meta", "sham_meta"), ("real_meta", "generic_prep"), ("sham_meta", "generic_prep")]
    for left, right in direct_pairs:
        patterns = Counter((vals.get(left), vals.get(right)) for vals in by_task.values())
        key = f"{left}__vs__{right}"
        out["paired_condition_deltas"][key] = {
            "patterns": {str(k): v for k, v in patterns.items()},
            "left_only_wins": sorted(task_id for task_id, vals in by_task.items() if vals.get(left) is True and vals.get(right) is False),
            "right_only_wins": sorted(task_id for task_id, vals in by_task.items() if vals.get(left) is False and vals.get(right) is True),
        }
    out["real_meta_unique_fixes_vs_controls"] = sorted(
        task_id
        for task_id, vals in by_task.items()
        if vals.get("think_only") is False and vals.get("real_meta") is True and vals.get("sham_meta") is not True and vals.get("generic_prep") is not True
    )
    out["real_meta_unique_wins"] = {
        "over_sham_meta": len(out["paired_condition_deltas"]["real_meta__vs__sham_meta"]["left_only_wins"]),
        "over_generic_prep": len(out["paired_condition_deltas"]["real_meta__vs__generic_prep"]["left_only_wins"]),
        "unique_fixes_vs_both_controls": len(out["real_meta_unique_fixes_vs_controls"]),
    }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace-eval-dir", type=Path, default=Path(DEFAULT_TRACE_EVAL))
    ap.add_argument("--selection-path", type=Path, default=Path(DEFAULT_SELECTION))
    ap.add_argument("--output-dir", type=Path, default=Path("sweep_v4/meta_sham_control_ages_qwen35_20260416"))
    ap.add_argument("--endpoints", default=DEFAULT_ENDPOINTS)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--concurrency", type=int, default=32)
    ap.add_argument("--max-tokens", type=int, default=900)
    ap.add_argument("--timeout", type=int, default=240)
    ap.add_argument("--max-attempts", type=int, default=3)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--top-k", type=int, default=-1)
    ap.add_argument("--presence-penalty", type=float, default=0.0)
    ap.add_argument("--enable-thinking", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    records_path = args.output_dir / "records.jsonl"
    summary_path = args.output_dir / "summary.json"
    if args.overwrite:
        for p in (records_path, summary_path, args.output_dir / "summary.partial.json", args.output_dir / "DONE"):
            p.unlink(missing_ok=True)

    selection = load_json(args.selection_path)
    selected_task_ids = set(selection["selected_task_ids"])
    rows = [r for r in read_trace_eval(args.trace_eval_dir) if task_key(r) in selected_task_ids]
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
        "enable_thinking": args.enable_thinking,
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
            args.enable_thinking,
        )

    completed = len(existing)
    with cf.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = [pool.submit(one, row, condition) for row, condition in tasks]
        total = len(existing) + len(futures)
        for fut in cf.as_completed(futures):
            rec = fut.result()
            append_jsonl(records_path, rec)
            completed += 1
            if completed % 24 == 0 or completed == total:
                write_json(args.output_dir / "summary.partial.json", summarize(load_jsonl(records_path)))
                print(f"[{now_iso()}] progress {completed}/{total}", flush=True)

    summary = summarize(load_jsonl(records_path))
    write_json(summary_path, summary)
    (args.output_dir / "DONE").write_text(now_iso() + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
