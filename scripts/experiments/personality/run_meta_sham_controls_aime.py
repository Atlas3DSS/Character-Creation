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


DEFAULT_AIME_DATA = "/home/orwel/dev_genius/experiments/Dual Stream LLM/data/aime_eval.jsonl"
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


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def read_aime(path: Path) -> list[dict[str, Any]]:
    rows = load_jsonl(path)
    out: list[dict[str, Any]] = []
    for row in rows:
        contest = str(row.get("contest") or "AIME")
        problem = int(row.get("problem") or 0)
        out.append(
            {
                "task_id": f"{contest.replace(' ', '_')}_{problem:02d}",
                "contest": contest,
                "problem": problem,
                "year": row.get("year"),
                "question": row.get("text") or "",
                "answer_key": str(row.get("answer")).strip(),
            }
        )
    return out


def make_messages(row: dict[str, Any], condition: str) -> list[dict[str, str]]:
    system = (
        "Solve the math problem carefully. Keep reasoning terse. "
        "The final answer must be a single integer."
    )
    question = row["question"].strip()
    if condition == "think_only":
        instr = (
            "Output exactly these sections in order and nothing before them:\n"
            "Final Answer: <single integer>\n"
            "/think\n"
            "<at most 4 short lines of reasoning, under 80 words total>\n"
            "/end-think\n"
            "Explanation: <one short sentence>"
        )
    elif condition == "real_meta":
        instr = (
            "Output exactly these sections in order and nothing before them:\n"
            "/meta-think\n"
            "problem_type: <what kind of math problem this is>\n"
            "goal_form: <what the answer should look like>\n"
            "failure_risk: <main likely mistake>\n"
            "strategy: <best compact plan>\n"
            "/end-meta-think\n"
            "Final Answer: <single integer>\n"
            "/think\n"
            "<at most 4 short lines of reasoning, under 80 words total>\n"
            "/end-think\n"
            "Explanation: <one short sentence>"
        )
    elif condition == "sham_meta":
        instr = (
            "Output exactly these sections in order and nothing before them.\n"
            "Copy the `/meta-think` block below exactly as written, then continue:\n"
            f"{SHAM_META_BLOCK}\n"
            "Final Answer: <single integer>\n"
            "/think\n"
            "<at most 4 short lines of reasoning, under 80 words total>\n"
            "/end-think\n"
            "Explanation: <one short sentence>"
        )
    elif condition == "generic_prep":
        instr = (
            "Output exactly these sections in order and nothing before them.\n"
            "Copy the `/prep-think` block below exactly as written, then continue:\n"
            f"{GENERIC_PREP_BLOCK}\n"
            "Final Answer: <single integer>\n"
            "/think\n"
            "<at most 4 short lines of reasoning, under 80 words total>\n"
            "/end-think\n"
            "Explanation: <one short sentence>"
        )
    else:
        raise ValueError(f"unknown condition: {condition}")
    user = (
        f"{question}\n\n"
        f"{instr}\n"
        "Do not emit any extra prose outside the required sections. "
        "Do not restate the problem. Put `Final Answer:` first and stop after `Explanation:`."
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def extract_final(text: str) -> str | None:
    m = re.search(r"Final Answer\s*:\s*(.+)", text, flags=re.I)
    if not m:
        return None
    return m.group(1).strip().splitlines()[0].strip()


def extract_boxed(text: str) -> str | None:
    if not text:
        return None
    matches = re.findall(r"\\boxed\{([^}]+)\}", text)
    if not matches:
        return None
    return matches[-1].strip()


def extract_integer(text: str | None) -> str | None:
    if not text:
        return None
    matches = re.findall(r"-?\d+", text.replace(",", ""))
    if not matches:
        return None
    return matches[-1]


def extract_answer_candidate(text: str) -> str | None:
    final = extract_final(text)
    if final:
        parsed = extract_integer(final)
        if parsed is not None:
            return parsed
    boxed = extract_boxed(text)
    if boxed:
        parsed = extract_integer(boxed)
        if parsed is not None:
            return parsed
    for pattern in (
        r"(?:final answer|answer)\s*(?:is|=)\s*(-?\d+)",
        r"therefore[^0-9-]*(-?\d+)\s*$",
        r"thus[^0-9-]*(-?\d+)\s*$",
    ):
        matches = re.findall(pattern, text, flags=re.I | re.M)
        if matches:
            return matches[-1]
    return None


def score_answer(text: str, key: str | None) -> bool | None:
    if not key:
        return None
    final = extract_answer_candidate(text)
    if final is None:
        return False
    return final == extract_integer(key)


def call_openai(
    base_url: str,
    model: str,
    messages: list[dict[str, str]],
    max_tokens: int,
    timeout: int,
) -> tuple[str, dict[str, Any]]:
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


def endpoint_for(endpoints: list[str], row: dict[str, Any]) -> str:
    return endpoints[int(row["problem"]) % len(endpoints)]


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
) -> dict[str, Any]:
    endpoint = endpoint_for(endpoints, row)
    text = ""
    usage: dict[str, Any] = {}
    error = None
    started = time.time()
    for attempt in range(1, max_attempts + 1):
        try:
            text, usage = call_openai(endpoint, model, make_messages(row, condition), max_tokens, timeout)
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
        "task_id": row["task_id"],
        "contest": row["contest"],
        "problem": row["problem"],
        "year": row["year"],
        "answer_key": row["answer_key"],
        "endpoint": endpoint,
        "latency_s": latency,
        "usage": usage,
        "text": text,
        "error": error,
        "final_answer": extract_final(text),
        "parsed_answer": extract_answer_candidate(text),
        "correct": score_answer(text, row.get("answer_key")),
        "format_ok": format_ok(text, condition),
        "truncated": bool(text and not re.search(r"Final Answer\s*:", text, flags=re.I)),
    }


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {
        "n_records": len(records),
        "by_condition": {},
        "paired_vs_think_only": {},
        "paired_condition_deltas": {},
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
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--aime-path", type=Path, default=Path(DEFAULT_AIME_DATA))
    ap.add_argument("--output-dir", type=Path, default=Path("sweep_v4/meta_sham_control_aime_qwen35_20260416"))
    ap.add_argument("--endpoints", default=DEFAULT_ENDPOINTS)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--limit", type=int, default=0, help="Optional number of AIME rows to keep")
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--max-tokens", type=int, default=1200)
    ap.add_argument("--timeout", type=int, default=300)
    ap.add_argument("--max-attempts", type=int, default=3)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    records_path = args.output_dir / "records.jsonl"
    summary_path = args.output_dir / "summary.json"
    if args.overwrite:
        for p in (records_path, summary_path, args.output_dir / "summary.partial.json", args.output_dir / "DONE"):
            p.unlink(missing_ok=True)

    rows = read_aime(args.aime_path)
    rows.sort(key=lambda r: (int(r["year"] or 0), r["contest"], int(r["problem"])))
    if args.limit and args.limit > 0:
        rows = rows[: args.limit]
    endpoints = [e.strip() for e in args.endpoints.split(",") if e.strip()]
    manifest = {
        "started_at": now_iso(),
        "aime_path": str(args.aime_path),
        "output_dir": str(args.output_dir),
        "endpoints": endpoints,
        "model": args.model,
        "conditions": list(CONDITIONS),
        "selected_count": len(rows),
    }
    write_json(args.output_dir / "manifest.json", manifest)

    existing = load_jsonl(records_path)
    done = {(r["task_id"], r["condition"]) for r in existing}
    tasks = [(row, condition) for row in rows for condition in CONDITIONS if (row["task_id"], condition) not in done]

    def one(row: dict[str, Any], condition: str) -> dict[str, Any]:
        return run_request(row, condition, endpoints, args.model, args.max_tokens, args.timeout, args.max_attempts)

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
