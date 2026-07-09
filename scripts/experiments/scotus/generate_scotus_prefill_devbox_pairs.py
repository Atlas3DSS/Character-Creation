#!/usr/bin/env python3
"""Generate SCOTUS prefill contrastive pairs through dev-box OpenAI endpoints."""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import re
import sys
import time
import urllib.error
import urllib.request
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.experiments.scotus.qwen_eval_budget import (  # noqa: E402
    add_short_budget_arg,
    enforce_complete_answer_budget,
    qwen_budget_metadata,
)


DEFAULT_ENDPOINTS = [
    "http://192.168.1.90:8080/v1",
    "http://192.168.1.90:8181/v1",
]
DEFAULT_PREFILLS = PROJECT_ROOT / "sweep_v4/scotus_prefill_contrasts_20260706_234523/prefills.jsonl"
DEFAULT_OUTPUT = PROJECT_ROOT / "sweep_v4/scotus_prefill_devbox_pairs_v1"

BASE_SYSTEM = (
    "You are generating synthetic Supreme Court-style legal reasoning passages for a contrastive "
    "research dataset. Follow the user's prompt closely. The case is hypothetical: do not claim "
    "that the generated passage is an actual Supreme Court opinion, do not quote the archived "
    "seeds, and do not identify the source cases. Keep the legal reasoning coherent and concrete."
)


@dataclass(frozen=True)
class Endpoint:
    base_url: str
    model: str


def now_iso() -> str:
    return datetime.now().astimezone().isoformat()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def parse_csv(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def clean_response(text: str) -> str:
    out = text or ""
    for token in ("<|im_end|>", "<|endoftext|>", "<|im_start|>"):
        out = out.replace(token, "")
    out = re.sub(r"(?is)<think>.*?</think>", "", out)
    return out.strip()


def http_json(method: str, url: str, payload: dict[str, Any] | None = None, timeout: int = 300) -> dict[str, Any]:
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        method=method,
        headers={
            "Authorization": "Bearer none",
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        text = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} from {url}: {text[:500]}") from exc


def discover_endpoint(base_url: str, timeout: int) -> Endpoint:
    data = http_json("GET", base_url.rstrip("/") + "/models", timeout=timeout)
    models = data.get("data") or data.get("models") or []
    if not models:
        raise RuntimeError(f"No models listed by {base_url}")
    first = models[0]
    model = str(first.get("id") or first.get("model") or first.get("name"))
    return Endpoint(base_url=base_url.rstrip("/"), model=model)


def chat_completion(
    endpoint: Endpoint,
    system_prompt: str,
    user_prompt: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    timeout: int,
    seed: int,
    enable_thinking: bool,
) -> tuple[str, dict[str, Any], float]:
    payload = {
        "model": endpoint.model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
        "seed": seed,
        "chat_template_kwargs": {"enable_thinking": enable_thinking},
    }
    started = time.time()
    data = http_json("POST", endpoint.base_url + "/chat/completions", payload=payload, timeout=timeout)
    elapsed = time.time() - started
    choices = data.get("choices") or []
    if not choices:
        raise RuntimeError(f"No choices returned by {endpoint.base_url}")
    content = choices[0].get("message", {}).get("content") or choices[0].get("text") or ""
    return clean_response(str(content)), data.get("usage", {}), elapsed


def side_prompt(row: dict[str, Any], side: str) -> str:
    if side not in {"a", "b"}:
        raise ValueError(f"unknown side: {side}")
    label = str(row.get(f"label_{side}") or row.get(f"writer_{side}") or side.upper())
    writer = str(row.get(f"writer_{side}") or label)
    posture = str(row.get(f"posture_{side}") or "same posture")
    contrast = "A" if side == "a" else "B"
    return (
        f"{row['generation_prompt']}\n\n"
        f"Requested side: {contrast}\n"
        f"Target writer/posture label: {label}\n"
        f"Writer seed: {writer}\n"
        f"Posture seed: {posture}\n\n"
        "Write only the generated opinion passage. Do not add commentary, bullet labels, "
        "or an explanation of the contrastive dataset."
    )


def generate_pair(
    run_idx: int,
    source_prefill_index: int,
    row: dict[str, Any],
    endpoints: list[Endpoint],
    args: argparse.Namespace,
) -> dict[str, Any]:
    pair_id = f"scotus_prefill_pair_{source_prefill_index:06d}_{row['prefill_id']}"
    endpoint_a = endpoints[run_idx % len(endpoints)]
    endpoint_b = endpoints[(run_idx + 1) % len(endpoints)]

    def call_a() -> tuple[str, dict[str, Any], float]:
        return chat_completion(
            endpoint=endpoint_a,
            system_prompt=BASE_SYSTEM,
            user_prompt=side_prompt(row, "a"),
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            timeout=args.timeout,
            seed=args.seed + source_prefill_index * 2 + 1,
            enable_thinking=args.enable_thinking,
        )

    def call_b() -> tuple[str, dict[str, Any], float]:
        return chat_completion(
            endpoint=endpoint_b,
            system_prompt=BASE_SYSTEM,
            user_prompt=side_prompt(row, "b"),
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            timeout=args.timeout,
            seed=args.seed + source_prefill_index * 2 + 2,
            enable_thinking=args.enable_thinking,
        )

    with cf.ThreadPoolExecutor(max_workers=2) as pool:
        future_a = pool.submit(call_a)
        future_b = pool.submit(call_b)
        response_a, usage_a, elapsed_a = future_a.result()
        response_b, usage_b, elapsed_b = future_b.result()

    return {
        "id": pair_id,
        "idx": run_idx,
        "source_prefill_index": source_prefill_index,
        "created_at": now_iso(),
        "prefill_id": row["prefill_id"],
        "comparison_axis": row.get("comparison_axis"),
        "split": row.get("split"),
        "issue_area_label": row.get("issue_area_label"),
        "decision_direction": row.get("decision_direction"),
        "label_a": row.get("label_a"),
        "label_b": row.get("label_b"),
        "writer_a": row.get("writer_a"),
        "writer_b": row.get("writer_b"),
        "posture_a": row.get("posture_a"),
        "posture_b": row.get("posture_b"),
        "chunk_id_a": row.get("chunk_id_a"),
        "chunk_id_b": row.get("chunk_id_b"),
        "generation_prompt": row.get("generation_prompt"),
        "response_a": response_a,
        "response_b": response_b,
        "endpoint_a": endpoint_a.base_url,
        "endpoint_b": endpoint_b.base_url,
        "model_a": endpoint_a.model,
        "model_b": endpoint_b.model,
        "usage_a": usage_a,
        "usage_b": usage_b,
        "elapsed_a_s": round(elapsed_a, 3),
        "elapsed_b_s": round(elapsed_b, 3),
        "max_tokens": args.max_tokens,
        "enable_thinking": args.enable_thinking,
    }


def good_pair(record: dict[str, Any], min_chars: int) -> bool:
    return len(str(record.get("response_a", "")).strip()) >= min_chars and len(
        str(record.get("response_b", "")).strip()
    ) >= min_chars


def summarize(path: Path) -> dict[str, Any]:
    rows = read_jsonl(path)
    axes = Counter(str(row.get("comparison_axis", "unknown")) for row in rows)
    writer_pairs = Counter(f"{row.get('writer_a')}__vs__{row.get('writer_b')}" for row in rows)
    return {
        "pairs": len(rows),
        "axes": dict(axes),
        "top_writer_pairs": dict(writer_pairs.most_common(20)),
        "avg_response_a_chars": round(sum(len(str(row.get("response_a", ""))) for row in rows) / max(1, len(rows)), 1),
        "avg_response_b_chars": round(sum(len(str(row.get("response_b", ""))) for row in rows) / max(1, len(rows)), 1),
    }


def select_prefills(prefills: list[dict[str, Any]], args: argparse.Namespace) -> list[tuple[int, dict[str, Any]]]:
    axis_filter = set(parse_csv(args.axis_filter))
    selected = [
        (idx, row)
        for idx, row in enumerate(prefills)
        if not axis_filter or str(row.get("comparison_axis")) in axis_filter
    ]
    if args.max_per_axis > 0:
        counts: Counter[str] = Counter()
        capped: list[tuple[int, dict[str, Any]]] = []
        for idx, row in selected:
            axis = str(row.get("comparison_axis", "unknown"))
            if counts[axis] >= args.max_per_axis:
                continue
            counts[axis] += 1
            capped.append((idx, row))
        selected = capped
    if args.start_index < 0:
        raise ValueError("--start-index must be >= 0")
    return selected[args.start_index : args.start_index + args.target_pairs]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefills", type=Path, default=DEFAULT_PREFILLS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--target-pairs", type=int, default=240)
    parser.add_argument(
        "--axis-filter",
        default="",
        help="Comma-separated comparison_axis values to include before start/target slicing.",
    )
    parser.add_argument("--start-index", type=int, default=0, help="Start offset after axis filtering/capping.")
    parser.add_argument(
        "--max-per-axis",
        type=int,
        default=0,
        help="Optional cap per comparison_axis before start/target slicing; 0 disables the cap.",
    )
    parser.add_argument("--seed", type=int, default=20260707)
    parser.add_argument("--endpoints", default=",".join(DEFAULT_ENDPOINTS))
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=0,
        help="0 means 4096 without thinking and 8192 with --enable-thinking.",
    )
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.65)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--min-response-chars", type=int, default=500)
    parser.add_argument("--overwrite-manifest", action="store_true")
    add_short_budget_arg(parser)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.max_tokens <= 0:
        args.max_tokens = 8192 if args.enable_thinking else 4096
    enforce_complete_answer_budget(
        args.max_tokens,
        allow_short=args.allow_short_answer_budget,
        label="max_tokens",
        purpose="SCOTUS prefill contrastive generation",
    )

    output_dir = (PROJECT_ROOT / args.output).resolve() if not args.output.is_absolute() else args.output
    output_dir.mkdir(parents=True, exist_ok=True)
    pairs_path = output_dir / "pairs.jsonl"
    errors_path = output_dir / "errors.jsonl"
    manifest_path = output_dir / "manifest.json"
    summary_path = output_dir / "summary.json"

    endpoint_urls = [part.strip() for part in args.endpoints.split(",") if part.strip()]
    endpoints = [discover_endpoint(url, timeout=30) for url in endpoint_urls]
    prefills = read_jsonl(args.prefills)
    if not prefills:
        raise RuntimeError(f"No prefills found in {args.prefills}")
    selected_prefills = select_prefills(prefills, args)
    if not selected_prefills:
        raise RuntimeError("No prefills selected by the requested filters")
    existing_ids = {str(row.get("id")) for row in read_jsonl(pairs_path)}

    manifest = {
        "created_at": now_iso(),
        "script": str(Path(__file__).relative_to(PROJECT_ROOT)),
        "purpose": "SCOTUS novel prefill contrastive generation using dev-box OpenAI-compatible endpoints",
        "prefills": str(args.prefills),
        "output_dir": str(output_dir),
        "pairs_path": str(pairs_path),
        "target_pairs": args.target_pairs,
        "prompt_bank_size": len(prefills),
        "selected_prefill_count": len(selected_prefills),
        "axis_filter": parse_csv(args.axis_filter),
        "start_index": args.start_index,
        "max_per_axis": args.max_per_axis,
        "seed": args.seed,
        "endpoints": [endpoint.__dict__ for endpoint in endpoints],
        "max_tokens": args.max_tokens,
        "enable_thinking": args.enable_thinking,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "generation_only": True,
        "activation_extraction_performed": False,
        **qwen_budget_metadata(args.max_tokens),
    }
    if args.overwrite_manifest or not manifest_path.exists():
        write_json(manifest_path, manifest)

    started = time.time()
    completed = len(existing_ids)
    for run_idx, (source_prefill_index, row) in enumerate(selected_prefills):
        pair_id = f"scotus_prefill_pair_{source_prefill_index:06d}_{row['prefill_id']}"
        if pair_id in existing_ids:
            continue
        last_error: str | None = None
        for attempt in range(args.retries + 1):
            try:
                record = generate_pair(run_idx, source_prefill_index, row, endpoints, args)
                if not good_pair(record, args.min_response_chars):
                    raise RuntimeError("generated pair failed min response length filter")
                append_jsonl(pairs_path, record)
                completed += 1
                if completed % 10 == 0 or completed == 1:
                    summary = summarize(pairs_path)
                    write_json(summary_path, {**summary, "updated_at": now_iso()})
                    print(
                        f"[PROGRESS] pairs={summary['pairs']} "
                        f"avg_chars={summary['avg_response_a_chars']}/{summary['avg_response_b_chars']}",
                        flush=True,
                    )
                break
            except Exception as exc:  # noqa: BLE001
                last_error = repr(exc)
                time.sleep(min(10, 2 + attempt * 3))
        else:
            append_jsonl(
                errors_path,
                {
                    "id": pair_id,
                    "idx": run_idx,
                    "source_prefill_index": source_prefill_index,
                    "prefill_id": row.get("prefill_id"),
                    "comparison_axis": row.get("comparison_axis"),
                    "error": last_error,
                    "created_at": now_iso(),
                },
            )
            print(f"[ERROR] {pair_id} {last_error}", flush=True)

    final_summary = summarize(pairs_path)
    write_json(
        summary_path,
        {
            **final_summary,
            "updated_at": now_iso(),
            "elapsed_s": round(time.time() - started, 2),
            "errors": len(read_jsonl(errors_path)),
        },
    )
    print(output_dir)


if __name__ == "__main__":
    main()
