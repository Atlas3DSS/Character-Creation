#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import concurrent.futures as cf
import json
import math
import re
import statistics
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import requests


DEFAULT_BASE_URL = "http://127.0.0.1:30003/v1"
DEFAULT_MODEL = "/home/orwel/dev_genius/models/Qwen3.6-35B-A3B"
DEFAULT_TRACE_EVAL = "/home/orwel/dev_genius/experiments/Character Creation/sweep_v4/personality_meta_eval_trace_explicit_v1"
B5_DIMS = ["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"]


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
            if row.get("track") == "reasoning":
                rows.append(row)
                if len(rows) >= limit:
                    return rows
    return rows


def strip_output_contract(prompt: str) -> str:
    return re.sub(r"Output exactly three sections.*?Do not emit 'Thinking Process:'\.", "", prompt, flags=re.S).strip()


def prompt_variants(row: dict[str, Any]) -> dict[str, list[dict[str, str]]]:
    system = row.get("system_prompt") or "Follow the requested format."
    core = strip_output_contract(row.get("prompt_text") or "")
    response = (
        "Output exactly:\n"
        "Explanation: <one short sentence>\n"
        "Final Answer: <canonical short answer only>\n"
        "Do not emit 'Thinking Process:'."
    )
    think = (
        "Output exactly these sections in order and nothing before them:\n"
        "/think\n"
        "<brief in-character reasoning>\n"
        "/end-think\n"
        "Explanation: <one short sentence>\n"
        "Final Answer: <canonical short answer only>\n"
        "Do not emit 'Thinking Process:'."
    )
    meta_think = (
        "Output exactly these sections in order and nothing before them:\n"
        "/meta-think\n"
        "<2-5 short lines about identity constraints, task constraints, reasoning risk, and response plan only>\n"
        "/end-meta-think\n"
        "/think\n"
        "<brief in-character reasoning>\n"
        "/end-think\n"
        "Explanation: <one short sentence>\n"
        "Final Answer: <canonical short answer only>\n"
        "Do not emit 'Thinking Process:'."
    )
    return {
        "response_only": [{"role": "system", "content": system}, {"role": "user", "content": core + "\n\n" + response}],
        "think_only": [{"role": "system", "content": system}, {"role": "user", "content": core + "\n\n" + think}],
        "meta_think_plus_think": [{"role": "system", "content": system}, {"role": "user", "content": core + "\n\n" + meta_think}],
    }


def decode_routed_experts(encoded: str, n_layers: int, top_k: int) -> np.ndarray:
    raw = base64.b64decode(encoded.encode("utf-8"))
    flat = np.frombuffer(raw, dtype=np.int32)
    denom = n_layers * top_k
    if flat.size % denom != 0:
        raise ValueError(f"bad routed expert payload: {flat.size=} not divisible by {denom}")
    return flat.reshape(flat.size // denom, n_layers, top_k)


def call_route(base_url: str, model: str, messages: list[dict[str, str]], max_tokens: int, timeout: int) -> tuple[np.ndarray, dict[str, Any], float]:
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0,
        "max_tokens": max_tokens,
        "return_routed_experts": True,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    started = time.time()
    resp = requests.post(base_url.rstrip("/") + "/chat/completions", json=payload, timeout=timeout)
    latency = time.time() - started
    resp.raise_for_status()
    data = resp.json()
    sglext = data.get("sglext") or {}
    encoded = sglext.get("routed_experts")
    if not encoded:
        raise RuntimeError("SGLang response did not include sglext.routed_experts; relaunch with --enable-return-routed-experts")
    usage = data.get("usage") or {}
    return decode_routed_experts(encoded, n_layers=40, top_k=8), usage, latency


def dist_from_counts(counts: np.ndarray) -> np.ndarray:
    total = float(counts.sum())
    if total <= 0:
        return np.full(counts.shape, 1.0 / counts.size, dtype=np.float64)
    return counts.astype(np.float64) / total


def kl(p: np.ndarray, q: np.ndarray) -> float:
    eps = 1e-12
    p = np.clip(p, eps, 1.0)
    q = np.clip(q, eps, 1.0)
    return float(np.sum(p * np.log(p / q)))


def jsd(p: np.ndarray, q: np.ndarray) -> float:
    m = 0.5 * (p + q)
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def summarize(records_path: Path, n_layers: int, n_experts: int) -> dict[str, Any]:
    rows = load_jsonl(records_path)
    conditions = sorted({r["condition"] for r in rows})
    counts_by_condition = {c: np.zeros((n_layers, n_experts), dtype=np.int64) for c in conditions}
    token_counts_by_condition = Counter()
    rows_by_condition = Counter()
    trait_counts: dict[str, dict[str, dict[str, np.ndarray]]] = {
        c: {trait: {"high": np.zeros((n_layers, n_experts), dtype=np.int64), "low": np.zeros((n_layers, n_experts), dtype=np.int64)} for trait in B5_DIMS}
        for c in conditions
    }
    trait_rows: dict[str, dict[str, Counter]] = {c: {trait: Counter() for trait in B5_DIMS} for c in conditions}
    for row in rows:
        condition = row["condition"]
        routes = np.array(row["expert_counts"], dtype=np.int64)
        counts_by_condition[condition] += routes
        token_counts_by_condition[condition] += int(row.get("n_tokens", 0))
        rows_by_condition[condition] += 1
        b5 = row.get("big_five") or {}
        for trait in B5_DIMS:
            level = str(b5.get(trait, "")).lower()
            if level in {"high", "low"}:
                trait_counts[condition][trait][level] += routes
                trait_rows[condition][trait][level] += 1

    out: dict[str, Any] = {
        "n_records": len(rows),
        "conditions": conditions,
        "n_layers": n_layers,
        "n_experts": n_experts,
        "by_condition": {},
        "pairwise_condition": {},
        "trait_router_shift": {},
    }
    for condition in conditions:
        per_layer = []
        counts = counts_by_condition[condition]
        for layer in range(n_layers):
            d = dist_from_counts(counts[layer])
            top = np.argsort(-d)[:8]
            entropy = -float(np.sum(np.clip(d, 1e-12, 1.0) * np.log(np.clip(d, 1e-12, 1.0))))
            per_layer.append({
                "layer": layer,
                "entropy": entropy,
                "effective_experts": math.exp(entropy),
                "top_experts": [{"expert": int(i), "prob": float(d[i])} for i in top],
            })
        out["by_condition"][condition] = {
            "rows": int(rows_by_condition[condition]),
            "tokens": int(token_counts_by_condition[condition]),
            "per_layer": per_layer,
        }
    for i, a in enumerate(conditions):
        for b in conditions[i + 1:]:
            metrics = []
            for layer in range(n_layers):
                pa = dist_from_counts(counts_by_condition[a][layer])
                pb = dist_from_counts(counts_by_condition[b][layer])
                delta = pa - pb
                top_shift = np.argsort(-np.abs(delta))[:8]
                metrics.append({
                    "layer": layer,
                    "jsd": jsd(pa, pb),
                    "tv": float(0.5 * np.sum(np.abs(delta))),
                    "top_shift_experts": [{"expert": int(e), "delta_prob": float(delta[e]), "a_prob": float(pa[e]), "b_prob": float(pb[e])} for e in top_shift],
                })
            out["pairwise_condition"][f"{a}__{b}"] = {
                "mean_jsd": float(statistics.mean(m["jsd"] for m in metrics)),
                "max_jsd": float(max(m["jsd"] for m in metrics)),
                "max_jsd_layer": int(max(metrics, key=lambda m: m["jsd"])["layer"]),
                "mean_tv": float(statistics.mean(m["tv"] for m in metrics)),
                "max_tv": float(max(m["tv"] for m in metrics)),
                "max_tv_layer": int(max(metrics, key=lambda m: m["tv"])["layer"]),
                "per_layer": metrics,
            }
    for condition in conditions:
        out["trait_router_shift"][condition] = {}
        for trait in B5_DIMS:
            high_rows = trait_rows[condition][trait]["high"]
            low_rows = trait_rows[condition][trait]["low"]
            if high_rows < 2 or low_rows < 2:
                continue
            metrics = []
            for layer in range(n_layers):
                ph = dist_from_counts(trait_counts[condition][trait]["high"][layer])
                pl = dist_from_counts(trait_counts[condition][trait]["low"][layer])
                metrics.append({
                    "layer": layer,
                    "jsd": jsd(ph, pl),
                    "tv": float(0.5 * np.sum(np.abs(ph - pl))),
                })
            out["trait_router_shift"][condition][trait] = {
                "high_rows": int(high_rows),
                "low_rows": int(low_rows),
                "mean_jsd": float(statistics.mean(m["jsd"] for m in metrics)),
                "max_jsd": float(max(m["jsd"] for m in metrics)),
                "max_jsd_layer": int(max(metrics, key=lambda m: m["jsd"])["layer"]),
                "mean_tv": float(statistics.mean(m["tv"] for m in metrics)),
                "max_tv": float(max(m["tv"] for m in metrics)),
                "max_tv_layer": int(max(metrics, key=lambda m: m["tv"])["layer"]),
            }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default=DEFAULT_BASE_URL)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--trace-eval-dir", type=Path, default=Path(DEFAULT_TRACE_EVAL))
    ap.add_argument("--output-dir", type=Path, default=Path("sweep_v4/qwen36_moe_routing_map_v1"))
    ap.add_argument("--limit", type=int, default=48)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=1)
    ap.add_argument("--timeout", type=int, default=240)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    records_path = args.output_dir / "routing_records.jsonl"
    if records_path.exists() and args.overwrite:
        records_path.unlink()
    write_json(args.output_dir / "manifest.json", {
        "started_at": now_iso(),
        "base_url": args.base_url,
        "model": args.model,
        "trace_eval_dir": str(args.trace_eval_dir),
        "limit": args.limit,
        "conditions": ["response_only", "think_only", "meta_think_plus_think"],
        "note": "Expert counts are top-k routed expert IDs returned by SGLang, not full router probabilities.",
    })
    rows = read_trace_eval(args.trace_eval_dir, args.limit)
    tasks = [(i, row, condition, messages) for i, row in enumerate(rows) for condition, messages in prompt_variants(row).items()]

    def run_one(i: int, row: dict[str, Any], condition: str, messages: list[dict[str, str]]) -> dict[str, Any]:
        started = time.time()
        try:
            routes, usage, latency = call_route(args.base_url, args.model, messages, args.max_tokens, args.timeout)
            error = None
            counts = np.zeros((40, 256), dtype=np.int64)
            for layer in range(routes.shape[1]):
                counts[layer] = np.bincount(routes[:, layer, :].reshape(-1), minlength=256)[:256]
            n_tokens = int(routes.shape[0])
        except Exception as exc:  # noqa: BLE001
            usage, latency, error = {}, time.time() - started, repr(exc)
            counts = np.zeros((40, 256), dtype=np.int64)
            n_tokens = 0
        return {
            "timestamp": now_iso(),
            "condition": condition,
            "task_id": row.get("task_id"),
            "persona_id": row.get("persona_id"),
            "prompt_id": row.get("prompt_id"),
            "big_five": (row.get("persona") or {}).get("big_five") or {},
            "usage": usage,
            "latency_s": latency,
            "error": error,
            "n_tokens": n_tokens,
            "expert_counts": counts.tolist(),
        }

    completed = 0
    with cf.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = [pool.submit(run_one, i, row, condition, messages) for i, row, condition, messages in tasks]
        for fut in cf.as_completed(futures):
            append_jsonl(records_path, fut.result())
            completed += 1
            if completed % 24 == 0:
                summary = summarize(records_path, 40, 256)
                write_json(args.output_dir / "summary.partial.json", summary)
                print(f"[{now_iso()}] progress {completed}/{len(tasks)}", flush=True)
    summary = summarize(records_path, 40, 256)
    write_json(args.output_dir / "summary.json", summary)
    (args.output_dir / "DONE").write_text(now_iso() + "\n", encoding="utf-8")
    print(json.dumps({
        "n_records": summary["n_records"],
        "pairwise_condition": {k: {kk: vv for kk, vv in v.items() if kk != "per_layer"} for k, v in summary["pairwise_condition"].items()},
    }, indent=2))


if __name__ == "__main__":
    main()
