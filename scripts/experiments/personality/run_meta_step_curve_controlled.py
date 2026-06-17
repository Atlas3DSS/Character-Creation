#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import random
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


def read_trace_eval(trace_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for p in sorted(trace_dir.glob("records_shard_*.jsonl")):
        rows.extend(load_jsonl(p))
    return [r for r in rows if r.get("track") == "reasoning"]


def strip_output_contract(prompt: str) -> str:
    return re.sub(r"Output exactly three sections.*?Do not emit 'Thinking Process:'\.", "", prompt, flags=re.S).strip()


def make_messages(row: dict[str, Any], budget: int) -> list[dict[str, str]]:
    system = row.get("system_prompt") or "Follow the requested format."
    core = strip_output_contract(row.get("prompt_text") or "")
    if budget <= 0:
        instr = (
            "Output exactly these sections in order and nothing before them:\n"
            "/think\n"
            "<brief in-character reasoning>\n"
            "/end-think\n"
            "Explanation: <one short sentence>\n"
            "Final Answer: <canonical short answer only>"
        )
    else:
        blocks = []
        for i in range(1, budget + 1):
            blocks.append(
                f"/meta-think {i}\n"
                "identity: <compact persona constraint>\n"
                "constraint: <task constraint>\n"
                "reasoning_risk: <main risk>\n"
                "response_policy: <short policy>\n"
                f"/end-meta-think {i}"
            )
        instr = (
            "Output these sections in order and nothing before them:\n"
            + "\n".join(blocks)
            + "\n/think\n"
            "<brief in-character reasoning>\n"
            "/end-think\n"
            "Explanation: <one short sentence>\n"
            "Final Answer: <canonical short answer only>"
        )
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


def task_key(row: dict[str, Any]) -> str:
    return row.get("task_id") or f"{row.get('persona_id')}::{row.get('prompt_id')}"


def prompt_family(row: dict[str, Any]) -> str:
    return row.get("prompt_id") or "unknown"


def endpoint_for(endpoints: list[str], row: dict[str, Any]) -> str:
    persona = int(row.get("persona_id") or 0)
    return endpoints[persona % len(endpoints)]


def format_ok(text: str, budget: int) -> bool:
    low = text.lower()
    if "final answer:" not in low or "/think" not in low:
        return False
    if budget <= 0:
        return "/meta-think" not in low
    return low.count("/meta-think") >= budget


def run_request(
    row: dict[str, Any],
    budget: int,
    endpoints: list[str],
    model: str,
    max_tokens: int,
    timeout: int,
    max_attempts: int,
    stage: str,
) -> dict[str, Any]:
    endpoint = endpoint_for(endpoints, row)
    text = ""
    usage: dict[str, Any] = {}
    error = None
    started = time.time()
    for attempt in range(1, max_attempts + 1):
        try:
            text, usage = call_openai(endpoint, model, make_messages(row, budget), max_tokens, timeout)
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
        "stage": stage,
        "budget": budget,
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
        "format_ok": format_ok(text, budget),
        "thinking_process_leak": "thinking process:" in low,
        "truncated": bool(text and not re.search(r"Final Answer\s*:", text, flags=re.I)),
    }


def summarize_screen(records: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {"n_records": len(records), "by_prompt_id": {}}
    for prompt_id in sorted({r.get("prompt_id") for r in records}):
        vals = [r for r in records if r.get("prompt_id") == prompt_id]
        scored = [r for r in vals if r.get("correct") is not None]
        out["by_prompt_id"][prompt_id] = {
            "n": len(vals),
            "scored": len(scored),
            "accuracy": sum(bool(r.get("correct")) for r in scored) / max(len(scored), 1) if scored else None,
            "errors": sum(bool(r.get("error")) for r in vals),
            "mean_latency_s": statistics.mean([r.get("latency_s", 0.0) for r in vals]) if vals else None,
            "mean_completion_tokens": statistics.mean(
                [(r.get("usage") or {}).get("completion_tokens") for r in vals if (r.get("usage") or {}).get("completion_tokens") is not None]
            ) if vals else None,
        }
    return out


def select_rows(
    rows: list[dict[str, Any]],
    screen_summary: dict[str, Any],
    min_acc: float,
    max_acc: float,
    min_rows: int,
    max_rows: int,
    seed: int,
) -> dict[str, Any]:
    by_prompt = screen_summary["by_prompt_id"]
    eligible = [
        prompt_id for prompt_id, stats in by_prompt.items()
        if stats.get("accuracy") is not None and min_acc <= stats["accuracy"] <= max_acc
    ]
    if not eligible:
        ranked = sorted(
            by_prompt.items(),
            key=lambda kv: abs((kv[1].get("accuracy") if kv[1].get("accuracy") is not None else 0.0) - (min_acc + max_acc) / 2),
        )
        eligible = [prompt_id for prompt_id, _ in ranked[:2]]
    selected_pool = [r for r in rows if prompt_family(r) in eligible]
    if len(selected_pool) < min_rows:
        ranked = sorted(
            by_prompt.items(),
            key=lambda kv: abs((kv[1].get("accuracy") if kv[1].get("accuracy") is not None else 0.0) - (min_acc + max_acc) / 2),
        )
        expanded = []
        for prompt_id, _ in ranked:
            if prompt_id not in eligible:
                eligible.append(prompt_id)
                expanded = [r for r in rows if prompt_family(r) in eligible]
                if len(expanded) >= min_rows:
                    selected_pool = expanded
                    break
    rng = random.Random(seed)
    by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in selected_pool:
        by_family[prompt_family(row)].append(row)
    for fam in by_family.values():
        fam.sort(key=lambda r: task_key(r))
    if len(selected_pool) <= max_rows:
        selected = sorted(selected_pool, key=task_key)
    else:
        families = sorted(by_family)
        selected = []
        family_iters = {fam: by_family[fam][:] for fam in families}
        for fam in families:
            rng.shuffle(family_iters[fam])
        while len(selected) < max_rows and any(family_iters.values()):
            for fam in families:
                if family_iters[fam] and len(selected) < max_rows:
                    selected.append(family_iters[fam].pop())
    return {
        "eligible_prompt_ids": eligible,
        "selected_count": len(selected),
        "selected_prompt_counts": dict(Counter(prompt_family(r) for r in selected)),
        "selected_task_ids": [task_key(r) for r in selected],
        "selected_rows": selected,
    }


def summarize_curve(records: list[dict[str, Any]], selected_prompt_ids: list[str]) -> dict[str, Any]:
    out: dict[str, Any] = {
        "n_records": len(records),
        "selected_prompt_ids": selected_prompt_ids,
        "by_budget": {},
        "by_prompt_id": {},
        "paired_vs_budget0": {},
    }
    budgets = sorted({int(r.get("budget")) for r in records})
    for budget in budgets:
        vals = [r for r in records if int(r.get("budget")) == budget]
        scored = [r for r in vals if r.get("correct") is not None]
        toks = [(r.get("usage") or {}).get("completion_tokens") for r in vals if (r.get("usage") or {}).get("completion_tokens") is not None]
        out["by_budget"][str(budget)] = {
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
    for prompt_id in sorted({r.get("prompt_id") for r in records}):
        out["by_prompt_id"][prompt_id] = {}
        for budget in budgets:
            vals = [r for r in records if r.get("prompt_id") == prompt_id and int(r.get("budget")) == budget]
            if not vals:
                continue
            out["by_prompt_id"][prompt_id][str(budget)] = {
                "n": len(vals),
                "accuracy": sum(bool(r.get("correct")) for r in vals) / len(vals),
            }
    by_task: dict[str, dict[int, bool | None]] = defaultdict(dict)
    for r in records:
        by_task[task_key(r)][int(r.get("budget"))] = r.get("correct")
    for budget in budgets:
        if budget == 0:
            continue
        patterns = Counter((vals.get(0), vals.get(budget)) for vals in by_task.values())
        out["paired_vs_budget0"][str(budget)] = {
            "patterns": {str(k): v for k, v in patterns.items()},
            "marginal_fixes": sorted(task_id for task_id, vals in by_task.items() if vals.get(0) is False and vals.get(budget) is True),
            "marginal_regressions": sorted(task_id for task_id, vals in by_task.items() if vals.get(0) is True and vals.get(budget) is False),
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace-eval-dir", type=Path, default=Path(DEFAULT_TRACE_EVAL))
    ap.add_argument("--output-dir", type=Path, default=Path("sweep_v4/meta_step_curve_controlled_qwen35_20260416"))
    ap.add_argument("--endpoints", default=DEFAULT_ENDPOINTS)
    ap.add_argument("--model", default="Qwen/Qwen3.5-9B")
    ap.add_argument("--screen-concurrency", type=int, default=32)
    ap.add_argument("--curve-concurrency", type=int, default=32)
    ap.add_argument("--max-tokens", type=int, default=900)
    ap.add_argument("--timeout", type=int, default=240)
    ap.add_argument("--max-attempts", type=int, default=3)
    ap.add_argument("--min-acc", type=float, default=0.50)
    ap.add_argument("--max-acc", type=float, default=0.80)
    ap.add_argument("--min-rows", type=int, default=50)
    ap.add_argument("--max-rows", type=int, default=96)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--budgets", default="0,1,2,3")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    screen_path = args.output_dir / "screen_records.jsonl"
    curve_path = args.output_dir / "curve_records.jsonl"
    summary_screen_path = args.output_dir / "screen_summary.json"
    selection_path = args.output_dir / "selection.json"
    summary_curve_path = args.output_dir / "curve_summary.json"
    if args.overwrite:
        for p in (screen_path, curve_path, summary_screen_path, selection_path, summary_curve_path, args.output_dir / "DONE"):
            p.unlink(missing_ok=True)

    rows = read_trace_eval(args.trace_eval_dir)
    endpoints = [e.strip() for e in args.endpoints.split(",") if e.strip()]
    budgets = [int(x.strip()) for x in args.budgets.split(",") if x.strip()]
    write_json(args.output_dir / "manifest.json", {
        "started_at": now_iso(),
        "trace_eval_dir": str(args.trace_eval_dir),
        "endpoints": endpoints,
        "model": args.model,
        "budgets": budgets,
        "min_acc": args.min_acc,
        "max_acc": args.max_acc,
        "min_rows": args.min_rows,
        "max_rows": args.max_rows,
        "seed": args.seed,
    })

    def submit_rows(target_rows: list[dict[str, Any]], budget: int, stage: str, out_path: Path, concurrency: int) -> list[dict[str, Any]]:
        completed: list[dict[str, Any]] = []
        with cf.ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = [pool.submit(run_request, row, budget, endpoints, args.model, args.max_tokens, args.timeout, args.max_attempts, stage) for row in target_rows]
            total = len(futures)
            for idx, fut in enumerate(cf.as_completed(futures), start=1):
                rec = fut.result()
                append_jsonl(out_path, rec)
                completed.append(rec)
                if idx % 32 == 0 or idx == total:
                    print(f"[{now_iso()}] {stage} budget={budget} progress {idx}/{total}", flush=True)
        return completed

    screen_records = load_jsonl(screen_path)
    done_screen = {task_key(r) for r in screen_records if int(r.get("budget", 0)) == 0}
    pending_screen = [r for r in rows if task_key(r) not in done_screen]
    if pending_screen:
        screen_records.extend(submit_rows(pending_screen, 0, "screen", screen_path, args.screen_concurrency))
    screen_summary = summarize_screen(load_jsonl(screen_path))
    write_json(summary_screen_path, screen_summary)

    selection = select_rows(rows, screen_summary, args.min_acc, args.max_acc, args.min_rows, args.max_rows, args.seed)
    selected_rows = selection.pop("selected_rows")
    write_json(selection_path, selection)

    curve_records = load_jsonl(curve_path)
    done_curve = {(task_key(r), int(r.get("budget", -1))) for r in curve_records}
    for budget in budgets:
        pending = [r for r in selected_rows if (task_key(r), budget) not in done_curve]
        if pending:
            submit_rows(pending, budget, "curve", curve_path, args.curve_concurrency)
    curve_summary = summarize_curve(load_jsonl(curve_path), selection["eligible_prompt_ids"])
    write_json(summary_curve_path, curve_summary)
    (args.output_dir / "DONE").write_text(now_iso() + "\n", encoding="utf-8")
    print(json.dumps({"screen_summary": screen_summary, "selection": selection, "curve_summary": curve_summary}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
