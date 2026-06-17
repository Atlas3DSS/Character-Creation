#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import statistics
import subprocess
import time
from pathlib import Path
from typing import Any

import requests


DEFAULT_MODEL = "/home/orwel/dev_genius/models/Qwen3.6-35B-A3B"
DEFAULT_BASE_URL = "http://127.0.0.1:30003/v1"


def gpu_status() -> str:
    return subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.used,memory.total,utilization.gpu", "--format=csv,noheader,nounits"],
        text=True,
    ).strip()


def call_one(base_url: str, model: str, request_id: int, max_tokens: int, timeout: int) -> dict[str, Any]:
    prompt = (
        "Write a long comma-separated stream of short vivid nouns and verbs. "
        "Keep going until the token budget ends. Do not use bullets. Do not stop early. "
        f"Request id {request_id}."
    )
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": max_tokens,
        "ignore_eos": True,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    started = time.time()
    resp = requests.post(base_url.rstrip("/") + "/chat/completions", json=payload, timeout=timeout)
    latency = time.time() - started
    resp.raise_for_status()
    data = resp.json()
    usage = data.get("usage") or {}
    return {
        "latency_s": latency,
        "completion_tokens": usage.get("completion_tokens", 0),
        "prompt_tokens": usage.get("prompt_tokens", 0),
        "finish_reason": data["choices"][0].get("finish_reason"),
    }


def run_case(base_url: str, model: str, concurrency: int, max_tokens: int, timeout: int) -> dict[str, Any]:
    started = time.time()
    rows: list[dict[str, Any]] = []
    with cf.ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [pool.submit(call_one, base_url, model, i, max_tokens, timeout) for i in range(concurrency)]
        for fut in cf.as_completed(futures):
            rows.append(fut.result())
    wall = time.time() - started
    completion_tokens = sum(r["completion_tokens"] for r in rows)
    prompt_tokens = sum(r["prompt_tokens"] for r in rows)
    latencies = [r["latency_s"] for r in rows]
    return {
        "max_tokens": max_tokens,
        "concurrency": concurrency,
        "wall_s": wall,
        "completion_tokens": completion_tokens,
        "prompt_tokens": prompt_tokens,
        "aggregate_completion_tps": completion_tokens / max(wall, 1e-9),
        "aggregate_total_tps": (completion_tokens + prompt_tokens) / max(wall, 1e-9),
        "mean_latency_s": statistics.mean(latencies),
        "p50_latency_s": statistics.median(latencies),
        "max_latency_s": max(latencies),
        "finish_counts": {k: sum(1 for r in rows if r["finish_reason"] == k) for k in sorted({r["finish_reason"] for r in rows})},
        "gpu_after": gpu_status(),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default=DEFAULT_BASE_URL)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--output", type=Path, default=Path("sweep_v4/qwen36_35b_a3b_inference_bench_v1/sglang_benchmark.json"))
    ap.add_argument("--concurrency", default="1,8,16,24,32,35")
    ap.add_argument("--max-tokens", default="256,1024")
    ap.add_argument("--timeout", type=int, default=600)
    args = ap.parse_args()

    concurrencies = [int(x) for x in args.concurrency.split(",") if x]
    max_tokens_values = [int(x) for x in args.max_tokens.split(",") if x]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result: dict[str, Any] = {
        "started_at": time.time(),
        "base_url": args.base_url,
        "model": args.model,
        "runs": [],
    }
    for max_tokens in max_tokens_values:
        for concurrency in concurrencies:
            print(f"running max_tokens={max_tokens} concurrency={concurrency} gpu_before={gpu_status()}", flush=True)
            run = run_case(args.base_url, args.model, concurrency, max_tokens, args.timeout)
            print(json.dumps(run, indent=2), flush=True)
            result["runs"].append(run)
            args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    result["finished_at"] = time.time()
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"wrote {args.output.resolve()}")


if __name__ == "__main__":
    main()
