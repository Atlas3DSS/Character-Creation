#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import math
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import requests


DEFAULT_BASE_URL = "http://127.0.0.1:30003/v1"
DEFAULT_API_MODEL = "/home/orwel/dev_genius/models/Qwen3.6-35B-A3B"
DEFAULT_SEEDSET_DIR = "/home/orwel/dev_genius/experiments/Character Creation/sweep_v4/meta_cognition_seedset_balanced_v1_20260417_115123"
DEFAULT_OUTPUT_ROOT = "/home/orwel/dev_genius/experiments/Character Creation/sweep_v4"
DEFAULT_TAG = "meta_cognition_scorer_corpus_v1"


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def log(log_path: Path, msg: str) -> None:
    line = f"[{now_iso()}] {msg}"
    print(line, flush=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")


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


def extract_json(text: str) -> Any:
    return json.JSONDecoder().raw_decode(text.lstrip())[0]


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
    }
    t0 = time.time()
    resp = requests.post(
        base_url.rstrip("/") + "/chat/completions",
        headers={"Authorization": "Bearer none", "Content-Type": "application/json"},
        json=payload,
        timeout=timeout,
    )
    dt = time.time() - t0
    resp.raise_for_status()
    data = resp.json()
    usage = data.get("usage") or {}
    text = data["choices"][0]["message"].get("content") or ""
    comp = int(usage.get("completion_tokens") or 0)
    return text, {"generated_tokens": comp, "latency_s": dt, "tokens_per_s": float(comp / max(dt, 1e-9))}


def generation_messages(item: dict[str, Any]) -> list[dict[str, str]]:
    spec = {
        "behavior": item["behavior"],
        "setup": item["setup"],
        "turns": item["turns"],
        "expected_pass": item["expected_pass"],
        "expected_fail": item["expected_fail"],
        "metrics": item["metrics"],
        "notes": item.get("notes", ""),
    }
    schema = {
        "pass_candidates": ["assistant response 1", "assistant response 2"],
        "fail_candidates": ["assistant response 1", "assistant response 2"],
    }
    user = (
        "Create labeled assistant responses for scorer training.\n"
        "Return JSON only. No markdown.\n\n"
        "Item spec:\n"
        f"{json.dumps(spec, ensure_ascii=False, indent=2)}\n\n"
        "Requirements:\n"
        "- Write exactly 2 pass_candidates and exactly 2 fail_candidates.\n"
        "- Each response must answer the final user turn in the conversation.\n"
        "- pass_candidates must satisfy all listed metrics.\n"
        "- fail_candidates must look plausible but violate at least one listed metric in a behavior-typical way.\n"
        "- Keep each response under 140 words.\n"
        "- Do not mention the metric ids, labels, or hidden rubric directly.\n"
        "- Responses must be plain assistant text, not JSON strings with explanations.\n\n"
        "Schema:\n"
        f"{json.dumps(schema, ensure_ascii=False, indent=2)}"
    )
    return [
        {"role": "system", "content": "You generate concise, label-controlled assistant responses. Return valid JSON only."},
        {"role": "user", "content": user},
    ]


def judge_messages(item: dict[str, Any], candidates: list[dict[str, Any]]) -> list[dict[str, str]]:
    spec = {
        "behavior": item["behavior"],
        "setup": item["setup"],
        "turns": item["turns"],
        "metrics": item["metrics"],
        "expected_pass": item["expected_pass"],
        "expected_fail": item["expected_fail"],
    }
    payload = [
        {
            "index": idx,
            "target_label": row["target_label"],
            "response": row["response_text"],
        }
        for idx, row in enumerate(candidates)
    ]
    user = (
        "Score each candidate response against the item metrics.\n"
        "Return JSON only with this schema:\n"
        '{"ratings":[{"index":0,"all_metrics_pass":true,"passed_metric_ids":["..."],"failed_metric_ids":["..."],"note":"<=20 words"}]}\n\n'
        "Guidelines:\n"
        "- all_metrics_pass is true only if every listed metric passes.\n"
        "- Use the actual metric ids from the item spec.\n"
        "- Judge the response as written, not the target_label.\n\n"
        "Item spec:\n"
        f"{json.dumps(spec, ensure_ascii=False, indent=2)}\n\n"
        "Candidates:\n"
        f"{json.dumps(payload, ensure_ascii=False, indent=2)}"
    )
    return [
        {"role": "system", "content": "You are a strict rubric judge. Return valid JSON only."},
        {"role": "user", "content": user},
    ]


def choose_examples(
    item: dict[str, Any],
    split: str,
    generated: dict[str, list[str]],
    judged: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    chosen: list[dict[str, Any]] = []
    metric_ids = [m["id"] for m in item["metrics"]]
    for target_label in ("pass", "fail"):
        pool = [row for row in judged if row["target_label"] == target_label]
        if target_label == "pass":
            pool.sort(key=lambda r: (r["all_metrics_pass"], len(r["passed_metric_ids"]), -len(r["failed_metric_ids"])), reverse=True)
            ok = [r for r in pool if r["all_metrics_pass"]]
        else:
            pool.sort(key=lambda r: (len(r["failed_metric_ids"]), -len(r["passed_metric_ids"])), reverse=True)
            ok = [r for r in pool if not r["all_metrics_pass"]]
        selected = ok[0] if ok else (pool[0] if pool else None)
        if selected is None:
            continue
        chosen.append(
            {
                "split": split,
                "behavior": item["behavior"],
                "candidate_id": item["candidate_id"],
                "title": item["title"],
                "setup": item["setup"],
                "turns": item["turns"],
                "metrics": item["metrics"],
                "metric_ids": metric_ids,
                "target_label": target_label,
                "label": 1 if target_label == "pass" else 0,
                "response_text": selected["response_text"],
                "judge_all_metrics_pass": bool(selected["all_metrics_pass"]),
                "judge_passed_metric_ids": selected["passed_metric_ids"],
                "judge_failed_metric_ids": selected["failed_metric_ids"],
                "judge_note": selected["note"],
                "judge_match_target": bool(selected["all_metrics_pass"]) if target_label == "pass" else (not bool(selected["all_metrics_pass"])),
            }
        )
    return chosen


def process_item(
    item: dict[str, Any],
    split: str,
    base_url: str,
    api_model: str,
    timeout: int,
) -> dict[str, Any]:
    gen_text, gen_usage = chat_generate_api(
        base_url,
        api_model,
        generation_messages(item),
        max_new_tokens=900,
        temperature=0.8,
        top_p=0.95,
        top_k=40,
        timeout=timeout,
    )
    gen = extract_json(gen_text)
    pass_candidates = [str(x).strip() for x in gen.get("pass_candidates", [])][:2]
    fail_candidates = [str(x).strip() for x in gen.get("fail_candidates", [])][:2]
    candidates = (
        [{"target_label": "pass", "response_text": text} for text in pass_candidates if text]
        + [{"target_label": "fail", "response_text": text} for text in fail_candidates if text]
    )
    judge_text, judge_usage = chat_generate_api(
        base_url,
        api_model,
        judge_messages(item, candidates),
        max_new_tokens=500,
        temperature=0.0,
        top_p=1.0,
        top_k=1,
        timeout=timeout,
    )
    judged_payload = extract_json(judge_text)
    ratings = judged_payload.get("ratings", []) if isinstance(judged_payload, dict) else []
    rating_map = {int(r["index"]): r for r in ratings if isinstance(r, dict) and str(r.get("index", "")).isdigit()}
    judged_rows = []
    for idx, row in enumerate(candidates):
        rating = rating_map.get(idx, {})
        judged_rows.append(
            {
                **row,
                "all_metrics_pass": bool(rating.get("all_metrics_pass", False)),
                "passed_metric_ids": [str(x) for x in rating.get("passed_metric_ids", [])],
                "failed_metric_ids": [str(x) for x in rating.get("failed_metric_ids", [])],
                "note": str(rating.get("note", "")).strip(),
            }
        )
    chosen = choose_examples(item, split, gen, judged_rows)
    return {
        "split": split,
        "candidate_id": item["candidate_id"],
        "behavior": item["behavior"],
        "title": item["title"],
        "generation_text": gen_text,
        "judge_text": judge_text,
        "generation_usage": gen_usage,
        "judge_usage": judge_usage,
        "chosen_examples": chosen,
        "n_candidates": len(candidates),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seedset-dir", type=Path, default=Path(DEFAULT_SEEDSET_DIR))
    ap.add_argument("--base-url", default=DEFAULT_BASE_URL)
    ap.add_argument("--api-model", default=DEFAULT_API_MODEL)
    ap.add_argument("--parallel-requests", type=int, default=5)
    ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument("--output-root", type=Path, default=Path(DEFAULT_OUTPUT_ROOT))
    ap.add_argument("--tag", default=DEFAULT_TAG)
    args = ap.parse_args()

    stamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    out_dir = args.output_root / f"{args.tag}_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "run.log"
    write_json(
        out_dir / "manifest.json",
        {
            "started_at": now_iso(),
            "seedset_dir": str(args.seedset_dir),
            "base_url": args.base_url,
            "api_model": args.api_model,
            "parallel_requests": args.parallel_requests,
            "timeout": args.timeout,
        },
    )

    all_results: list[dict[str, Any]] = []
    split_rows: dict[str, list[dict[str, Any]]] = {}
    for split in ("train", "val", "test"):
        rows = load_jsonl(args.seedset_dir / f"{split}.jsonl")
        split_rows[split] = rows

    for split in ("train", "val", "test"):
        rows = split_rows[split]
        log(log_path, f"processing split={split} n_items={len(rows)}")
        with cf.ThreadPoolExecutor(max_workers=min(args.parallel_requests, len(rows))) as pool:
            fut_map = {
                pool.submit(process_item, row, split, args.base_url, args.api_model, args.timeout): row
                for row in rows
            }
            split_results: list[dict[str, Any]] = []
            for idx, fut in enumerate(cf.as_completed(fut_map), start=1):
                row = fut_map[fut]
                try:
                    result = fut.result()
                    split_results.append(result)
                    log(log_path, f"{split} {idx}/{len(rows)} behavior={row['behavior']} generated={result['n_candidates']} chosen={len(result['chosen_examples'])}")
                except Exception as exc:  # noqa: BLE001
                    log(log_path, f"{split} item failed candidate_id={row['candidate_id']} behavior={row['behavior']} error={exc!r}")
            split_results.sort(key=lambda r: (r["candidate_id"], r["behavior"]))
            write_jsonl(out_dir / f"{split}_item_results.jsonl", split_results)
            all_results.extend(split_results)

    selected_by_split: dict[str, list[dict[str, Any]]] = {"train": [], "val": [], "test": []}
    for result in all_results:
        selected_by_split[result["split"]].extend(result["chosen_examples"])
    for split, rows in selected_by_split.items():
        write_jsonl(out_dir / f"{split}.jsonl", rows)
    all_examples = selected_by_split["train"] + selected_by_split["val"] + selected_by_split["test"]
    write_jsonl(out_dir / "all.jsonl", all_examples)

    summary = {
        "finished_at": now_iso(),
        "n_examples": len(all_examples),
        "split_counts": {split: len(rows) for split, rows in selected_by_split.items()},
        "behavior_counts": dict(sorted(Counter(row["behavior"] for row in all_examples).items())),
        "label_counts": dict(sorted(Counter(str(row["label"]) for row in all_examples).items())),
        "judge_match_rate": float(sum(1 for row in all_examples if row["judge_match_target"]) / max(len(all_examples), 1)),
    }
    write_json(out_dir / "summary.json", summary)


if __name__ == "__main__":
    main()
