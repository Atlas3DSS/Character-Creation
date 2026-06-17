#!/usr/bin/env python3
"""Long-answer server counterfactual visible-thought diagnostic.

This is the no-hook version of ``probe_scotus_counterfactual_thoughts.py``.
It renders the Qwen chat template locally, pre-fills a controlled
``<think>...</think>`` block, and continues generation through
OpenAI-compatible llama.cpp ``/v1/completions`` endpoints.

Use this for evaluator calibration where hidden states are not required. Qwen
legal answers need thousands of answer tokens; short budgets are smoke only.
"""

from __future__ import annotations

import argparse
import json
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from tqdm import tqdm
from transformers import AutoTokenizer

from probe_scotus_counterfactual_thoughts import (
    MIN_COMPLETE_ANSWER_TOKENS,
    DEFAULT_COMPLETE_ANSWER_TOKENS,
    DEFAULT_PROMPT_BANK,
    NEUTRAL_THOUGHT,
    PRIVATE_THOUGHT,
    PUBLIC_THOUGHT,
    add_pair_deltas,
    row_for_counterfactual,
    summarize,
    write_report,
)
from run_scotus_thinking_smoke import IMITATION_RE, strip_generation_specials
from poke_scotus_sae_layers import (
    DEFAULT_OUTPUT_ROOT,
    load_prompt_specs,
    now_iso,
    select_prompt_specs,
    write_json,
    write_jsonl,
)
from qwen_eval_budget import enforce_complete_answer_budget, qwen_budget_metadata


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_TOKENIZER = Path("/home/orwel/dev_genius/models/Qwen3.6-27B-FP8")
DEFAULT_ENDPOINTS = (
    "q4_3090|http://192.168.1.90:8080|qwen3.6-27b-q4-3090,"
    "q4_4090|http://192.168.1.90:8081|qwen3.6-27b-q4-4090"
)


@dataclass(frozen=True)
class Endpoint:
    name: str
    url: str
    model: str


@dataclass(frozen=True)
class GenerationTask:
    prompt_index: int
    spec: Any
    condition: str
    thought: str
    endpoint: Endpoint
    seed: int


def parse_endpoints(raw: str) -> list[Endpoint]:
    endpoints: list[Endpoint] = []
    for item in raw.split(","):
        stripped = item.strip()
        if not stripped:
            continue
        parts = stripped.split("|")
        if len(parts) != 3:
            raise ValueError(f"Endpoint must be name|base_url|model: {item!r}")
        endpoints.append(Endpoint(name=parts[0], url=parts[1].rstrip("/"), model=parts[2]))
    if not endpoints:
        raise ValueError("At least one endpoint is required")
    return endpoints


def render_prefilled_prompt(tokenizer: Any, *, prompt: str, thought: str) -> str:
    chat = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )
    return f"{chat}{thought.strip()}\n</think>\n\n"


def call_completion(
    *,
    endpoint: Endpoint,
    rendered_prompt: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    seed: int,
    timeout: float,
    max_retries: int,
) -> dict[str, Any]:
    url = endpoint.url
    if not url.endswith("/v1/completions"):
        url = url.rstrip("/") + "/v1/completions"
    payload = {
        "model": endpoint.model,
        "prompt": rendered_prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "seed": seed,
        "stop": ["<|im_end|>"],
    }
    encoded = json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    last_error: str | None = None
    for attempt in range(max_retries + 1):
        request = urllib.request.Request(url, data=encoded, headers=headers, method="POST")
        try:
            started = time.monotonic()
            with urllib.request.urlopen(request, timeout=timeout) as response:
                raw = response.read().decode("utf-8")
            elapsed = time.monotonic() - started
            obj = json.loads(raw)
            choice = obj["choices"][0]
            text = strip_generation_specials(str(choice.get("text") or "")).strip()
            usage = obj.get("usage", {})
            return {
                "raw_text": str(choice.get("text") or ""),
                "answer": text,
                "finish_reason": choice.get("finish_reason"),
                "usage": usage,
                "answer_generated_tokens": int(usage.get("completion_tokens") or obj.get("tokens_predicted") or 0),
                "answer_prompt_tokens": int(usage.get("prompt_tokens") or obj.get("tokens_evaluated") or 0),
                "endpoint_name": endpoint.name,
                "endpoint_url": endpoint.url,
                "model": obj.get("model", endpoint.model),
                "elapsed_seconds": elapsed,
            }
        except (
            urllib.error.URLError,
            urllib.error.HTTPError,
            TimeoutError,
            json.JSONDecodeError,
            KeyError,
        ) as exc:
            last_error = repr(exc)
            if attempt >= max_retries:
                break
            time.sleep(min(30.0, 2.0 * (attempt + 1)))
    raise RuntimeError(f"{endpoint.name} failed after {max_retries + 1} attempts: {last_error}")


def output_for_task(
    *,
    tokenizer: Any,
    task: GenerationTask,
    answer_tokens: int,
    temperature: float,
    top_p: float,
    timeout: float,
    max_retries: int,
) -> dict[str, Any]:
    rendered = render_prefilled_prompt(tokenizer, prompt=task.spec.prompt, thought=task.thought)
    completion = call_completion(
        endpoint=task.endpoint,
        rendered_prompt=rendered,
        max_tokens=answer_tokens,
        temperature=temperature,
        top_p=top_p,
        seed=task.seed,
        timeout=timeout,
        max_retries=max_retries,
    )
    answer = str(completion["answer"])
    return {
        "prefilled_open_think": True,
        "thinking": task.thought.strip(),
        "answer": answer,
        "full_text": f"<think>\n{task.thought.strip()}\n</think>\n\n{answer}".strip(),
        "answer_generated_tokens": int(completion["answer_generated_tokens"]),
        "answer_prompt_tokens": int(completion["answer_prompt_tokens"]),
        "thinking_nonempty": bool(task.thought.strip()),
        "answer_nonempty": bool(answer),
        "thinking_imitation_markers": sorted(set(IMITATION_RE.findall(task.thought))),
        "answer_imitation_markers": sorted(set(IMITATION_RE.findall(answer))),
        "finish_reason": completion["finish_reason"],
        "usage": completion["usage"],
        "endpoint_name": completion["endpoint_name"],
        "endpoint_url": completion["endpoint_url"],
        "model": completion["model"],
        "elapsed_seconds": completion["elapsed_seconds"],
        "server_prefilled_completion": True,
    }


def build_tasks(
    *,
    prompt_specs: list[Any],
    endpoints: list[Endpoint],
    seed: int,
) -> list[GenerationTask]:
    variants = [
        ("neutral", NEUTRAL_THOUGHT),
        ("private_rights", PRIVATE_THOUGHT),
        ("public_rights", PUBLIC_THOUGHT),
    ]
    tasks: list[GenerationTask] = []
    for prompt_index, spec in enumerate(prompt_specs):
        endpoint = endpoints[prompt_index % len(endpoints)]
        for condition, thought in variants:
            tasks.append(
                GenerationTask(
                    prompt_index=prompt_index,
                    spec=spec,
                    condition=condition,
                    thought=thought,
                    endpoint=endpoint,
                    seed=seed + int(spec.prompt_id) * 1009 + len(tasks),
                )
            )
    return tasks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer-path", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--prompt-bank", type=Path, default=DEFAULT_PROMPT_BANK)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--endpoints", default=DEFAULT_ENDPOINTS)
    parser.add_argument("--prompt-ids", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--max-prompts", type=int, default=8)
    parser.add_argument("--answer-tokens", type=int, default=DEFAULT_COMPLETE_ANSWER_TOKENS)
    parser.add_argument(
        "--allow-short-answer-budget",
        action="store_true",
        help=(
            f"Permit answer budgets below {MIN_COMPLETE_ANSWER_TOKENS} tokens. "
            "Short-budget runs are smoke/debug only and must not be used for promotion."
        ),
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=20260502)
    parser.add_argument("--timeout", type=float, default=900.0)
    parser.add_argument("--max-retries", type=int, default=1)
    parser.add_argument("--workers-per-endpoint", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    enforce_complete_answer_budget(
        args.answer_tokens,
        allow_short=args.allow_short_answer_budget,
        label="answer_tokens",
        purpose="SCOTUS server counterfactual-thought answer run",
    )
    started = now_iso()
    out_dir = args.output_root / f"scotus_counterfactual_thoughts_server_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)

    endpoints = parse_endpoints(args.endpoints)
    prompt_specs = select_prompt_specs(load_prompt_specs(args.prompt_bank), args.prompt_ids, args.max_prompts)
    tasks = build_tasks(prompt_specs=prompt_specs, endpoints=endpoints, seed=args.seed)
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_path,
        trust_remote_code=True,
        local_files_only=True,
    )

    rows: list[dict[str, Any]] = []
    lock = threading.Lock()
    max_workers = max(1, len(endpoints) * max(1, args.workers_per_endpoint))
    endpoint_slots: list[Endpoint] = []
    for endpoint in endpoints:
        endpoint_slots.extend([endpoint] * max(1, args.workers_per_endpoint))

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                output_for_task,
                tokenizer=tokenizer,
                task=task,
                answer_tokens=args.answer_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                timeout=args.timeout,
                max_retries=args.max_retries,
            ): task
            for task in tasks
        }
        with tqdm(total=len(futures), desc="Generating long counterfactual answers") as progress:
            for future in as_completed(futures):
                task = futures[future]
                output = future.result()
                row = row_for_counterfactual(
                    spec=task.spec,
                    condition=task.condition,
                    thought=task.thought,
                    output=output,
                )
                row["endpoint_assigned"] = task.endpoint.name
                row["generation_seed"] = task.seed
                with lock:
                    rows.append(row)
                progress.update(1)

    rows.sort(key=lambda row: (int(row["prompt_id"]), str(row["condition"])))
    add_pair_deltas(rows)
    summaries = summarize(rows)
    manifest = {
        "started_at": started,
        "finished_at": now_iso(),
        "model_path": "server:" + ",".join(endpoint.model for endpoint in endpoints),
        "tokenizer_path": str(args.tokenizer_path),
        "prompt_bank": str(args.prompt_bank),
        "output_dir": str(out_dir),
        "prompt_ids": [spec.prompt_id for spec in prompt_specs],
        "prompt_keys": [spec.prompt_key for spec in prompt_specs],
        "sample_prompt_ids": [prompt_specs[0].prompt_id, prompt_specs[-1].prompt_id] if prompt_specs else [],
        "conditions": ["neutral", "private_rights", "public_rights"],
        "answer_tokens": int(args.answer_tokens),
        **qwen_budget_metadata(args.answer_tokens),
        "temperature": float(args.temperature),
        "top_p": float(args.top_p),
        "seed": int(args.seed),
        "endpoints": [endpoint.__dict__ for endpoint in endpoints],
        "workers_per_endpoint": int(args.workers_per_endpoint),
        "server_prefilled_completion": True,
    }
    write_json(out_dir / "manifest.json", manifest)
    write_jsonl(out_dir / "generations.jsonl", rows)
    write_jsonl(out_dir / "summary.jsonl", summaries)
    write_report(out_dir / "report.md", manifest=manifest, summaries=summaries, rows=rows)
    print(f"Wrote {out_dir / 'report.md'}", flush=True)


if __name__ == "__main__":
    main()
