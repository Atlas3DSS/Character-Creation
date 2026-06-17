#!/usr/bin/env python3
"""Benchmark a model or endpoint on the frozen trace-explicit personality eval set."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
import time
from collections import defaultdict
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module from {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def avg(values: list[float]) -> float | None:
    return mean(values) if values else None


def rate(num: int, den: int) -> float | None:
    return (num / den) if den else None


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def existing_task_ids(path: Path) -> set[str]:
    out: set[str] = set()
    if not path.exists():
        return out
    with path.open("r", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                out.add(str(json.loads(line)["task_id"]))
            except Exception:
                continue
    return out


def stable_seed(base_seed: int, task_id: str) -> int:
    digest = hashlib.blake2b(task_id.encode("utf-8"), digest_size=8).digest()
    return (base_seed + int.from_bytes(digest, "big")) % (2**31 - 1)


def summarize_group(rows: list[dict[str, Any]]) -> dict[str, Any]:
    reasoning = [r for r in rows if r.get("track") == "reasoning"]
    scored = [r for r in reasoning if r.get("is_correct") is not None]
    correct = [r for r in scored if r.get("is_correct") is True]
    finish_reasons: dict[str, int] = defaultdict(int)
    for row in rows:
        finish_reasons[str(row.get("finish_reason") or "unknown")] += 1
    return {
        "responses": len(rows),
        "avg_gen_tokens": avg([float(r.get("n_gen_tokens") or 0.0) for r in rows]),
        "avg_latency_s": avg([float(r.get("latency_s") or 0.0) for r in rows]),
        "format_adherent": sum(1 for r in rows if r.get("format_adherent")),
        "format_adherence_rate": rate(sum(1 for r in rows if r.get("format_adherent")), len(rows)),
        "truncated": sum(1 for r in rows if str(r.get("finish_reason") or "") == "length"),
        "truncation_rate": rate(sum(1 for r in rows if str(r.get("finish_reason") or "") == "length"), len(rows)),
        "visible_thinking": sum(1 for r in rows if r.get("contains_thinking_process")),
        "visible_thinking_rate": rate(sum(1 for r in rows if r.get("contains_thinking_process")), len(rows)),
        "trait_label_leak": sum(1 for r in rows if r.get("trait_label_leak")),
        "trait_label_leak_rate": rate(sum(1 for r in rows if r.get("trait_label_leak")), len(rows)),
        "reasoning_responses": len(reasoning),
        "reasoning_scored": len(scored),
        "reasoning_correct": len(correct),
        "reasoning_accuracy": rate(len(correct), len(scored)),
        "reasoning_coverage": rate(len(scored), len(reasoning)),
        "finish_reasons": dict(sorted(finish_reasons.items())),
    }


def write_summary(output_dir: Path, manifest: dict[str, Any], records: list[dict[str, Any]]) -> None:
    by_track_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_category_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_prompt_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        by_track_rows[str(row.get("track") or "unknown")].append(row)
        by_category_rows[str(row.get("prompt_category") or "unknown")].append(row)
        by_prompt_rows[str(row.get("prompt_id") or "unknown")].append(row)

    expected_total = int(manifest.get("expected_records") or 0)
    summary = {
        "timestamp": datetime.now().isoformat(),
        "dataset": manifest.get("dataset"),
        "benchmark_label": manifest.get("benchmark_label"),
        "backend": manifest.get("backend"),
        "expected_total": expected_total,
        "completed_total": len(records),
        "completion_rate": rate(len(records), expected_total),
        "pending_total": max(expected_total - len(records), 0),
        "overall": summarize_group(records),
        "by_track": {k: summarize_group(v) for k, v in sorted(by_track_rows.items())},
        "by_category": {k: summarize_group(v) for k, v in sorted(by_category_rows.items())},
        "by_prompt": {k: summarize_group(v) for k, v in sorted(by_prompt_rows.items())},
    }
    if records:
        mean_tokens = float(summary["overall"]["avg_gen_tokens"] or 0.0)
        summary["estimated_total_gen_tokens"] = int(round(mean_tokens * expected_total)) if expected_total else None
    else:
        summary["estimated_total_gen_tokens"] = None

    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines: list[str] = []
    lines.append(f"# {manifest.get('benchmark_label', 'personality_trace_benchmark')} summary")
    lines.append("")
    lines.append(f"- backend: {manifest.get('backend')}")
    lines.append(f"- completed: {summary['completed_total']} / {summary['expected_total']}")
    lines.append(f"- completion rate: {summary['completion_rate']}")
    lines.append(f"- estimated total gen tokens: {summary['estimated_total_gen_tokens']}")
    lines.append(f"- overall format adherence: {summary['overall']['format_adherence_rate']}")
    lines.append(f"- overall visible thinking: {summary['overall']['visible_thinking_rate']}")
    lines.append(f"- overall truncation: {summary['overall']['truncation_rate']}")
    lines.append(f"- overall reasoning accuracy: {summary['overall']['reasoning_accuracy']}")
    lines.append("")
    lines.append("## By Track")
    lines.append("")
    for track, payload in summary["by_track"].items():
        lines.append(f"### {track}")
        lines.append(f"- responses: {payload['responses']}")
        lines.append(f"- avg gen tokens: {payload['avg_gen_tokens']}")
        lines.append(f"- avg latency: {payload['avg_latency_s']}")
        lines.append(f"- format adherence: {payload['format_adherence_rate']}")
        lines.append(f"- visible thinking: {payload['visible_thinking_rate']}")
        lines.append(f"- truncation: {payload['truncation_rate']}")
        if payload["reasoning_responses"]:
            lines.append(f"- reasoning accuracy: {payload['reasoning_accuracy']}")
            lines.append(f"- reasoning coverage: {payload['reasoning_coverage']}")
        lines.append("")
    lines.append("## By Category")
    lines.append("")
    for category, payload in summary["by_category"].items():
        lines.append(f"### {category}")
        lines.append(f"- responses: {payload['responses']}")
        lines.append(f"- avg gen tokens: {payload['avg_gen_tokens']}")
        lines.append(f"- avg latency: {payload['avg_latency_s']}")
        lines.append(f"- format adherence: {payload['format_adherence_rate']}")
        if payload["reasoning_responses"]:
            lines.append(f"- reasoning accuracy: {payload['reasoning_accuracy']}")
        lines.append("")
    (output_dir / "summary.md").write_text("\n".join(lines).strip() + "\n", encoding="utf-8")


class V3Compat:
    @staticmethod
    def parse_think_response(full_text: str) -> tuple[str, str]:
        think_text = ""
        response_text = full_text
        if "</think>" in full_text:
            parts = full_text.split("</think>", 1)
            think_text = parts[0].replace("<think>", "").strip()
            response_text = parts[1].strip()
        elif "<think>" in full_text:
            think_text = full_text.replace("<think>", "").strip()
            response_text = ""

        for tok in ["<|im_end|>", "<|endoftext|>", "<|im_start|>"]:
            response_text = response_text.replace(tok, "").strip()
            think_text = think_text.replace(tok, "").strip()
        return think_text, response_text

    @staticmethod
    def load_processor(model_name: str):
        from transformers import AutoProcessor

        return AutoProcessor.from_pretrained(model_name, trust_remote_code=True)


class QwenTokenCounter:
    def __init__(self, model_name: str):
        processor = V3Compat.load_processor(model_name)
        self.tokenizer = processor.tokenizer
        self.label = model_name

    def encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text, add_special_tokens=False)


class NanochatRunner:
    def __init__(
        self,
        nanochat_root: Path,
        nanochat_base_dir: Path,
        source: str,
        model_tag: str | None,
        step: int | None,
        device_type: str,
    ) -> None:
        os.environ["NANOCHAT_BASE_DIR"] = str(nanochat_base_dir)
        if str(nanochat_root) not in sys.path:
            sys.path.insert(0, str(nanochat_root))
        from nanochat.common import autodetect_device_type, compute_init, compute_cleanup  # type: ignore
        from nanochat.checkpoint_manager import load_model  # type: ignore
        from nanochat.engine import Engine  # type: ignore

        resolved_device = autodetect_device_type() if not device_type else device_type
        self._compute_cleanup = compute_cleanup
        _, _, _, _, device = compute_init(resolved_device)
        self.model, self.tokenizer, self.meta = load_model(
            source,
            device,
            phase="eval",
            model_tag=model_tag,
            step=step,
        )
        self.engine = Engine(self.model, self.tokenizer)
        self.label = f"nanochat:{source}:{model_tag or 'auto'}:{step or 'latest'}"

    def encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text)

    def close(self) -> None:
        self._compute_cleanup()

    def run_one(
        self,
        task: dict[str, Any],
        max_new_tokens: int,
        temperature: float,
        top_k: int,
        seed: int,
    ) -> dict[str, Any]:
        conversation = {
            "messages": [
                {"role": "system", "content": task["system_prompt"]},
                {"role": "user", "content": task["prompt_text"]},
                {"role": "assistant", "content": ""},
            ]
        }
        prompt_ids = self.tokenizer.render_for_completion(conversation)
        t0 = time.time()
        results, _ = self.engine.generate_batch(
            prompt_ids,
            num_samples=1,
            max_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            seed=seed,
        )
        latency = time.time() - t0
        full_ids = results[0]
        completion_ids = full_ids[len(prompt_ids) :]
        finish_reason = "length" if len(completion_ids) >= max_new_tokens else "stop"
        return {
            "ok": True,
            "full_text": self.tokenizer.decode(completion_ids),
            "reasoning_content": "",
            "completion_tokens": len(completion_ids),
            "finish_reason": finish_reason,
            "latency_s": latency,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark on the frozen trace-explicit personality eval set")
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--benchmark-label", default="personality_trace_benchmark")
    parser.add_argument("--backend", choices=["openai", "nanochat"], required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--n-shards", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260404)
    parser.add_argument("--max-new-tokens", type=int, default=960)
    parser.add_argument("--temperature", type=float, default=0.4)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--scoring-tokenizer-model", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--skip-scoring-tokenizer", action="store_true")

    parser.add_argument("--base-url", default="")
    parser.add_argument("--api-key", default="dummy")
    parser.add_argument("--model", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--timeout", type=float, default=240.0)
    parser.add_argument("--retries", type=int, default=3)

    parser.add_argument("--nanochat-root", default="/home/orwel/dev_genius/nanochat")
    parser.add_argument("--nanochat-base-dir", default="")
    parser.add_argument("--nanochat-source", choices=["base", "sft", "rl"], default="sft")
    parser.add_argument("--nanochat-model-tag", default="")
    parser.add_argument("--nanochat-step", type=int, default=0)
    parser.add_argument("--device-type", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.shard < 0 or args.shard >= args.n_shards:
        raise ValueError(f"--shard must be in [0, {args.n_shards - 1}]")
    if args.backend == "openai" and not args.base_url:
        raise ValueError("--base-url is required for backend=openai")
    if args.backend == "nanochat" and not args.nanochat_base_dir:
        raise ValueError("--nanochat-base-dir is required for backend=nanochat")

    script_dir = Path(__file__).resolve().parent
    eval_mod = load_module("personality_meta_eval_openai_benchmark_helpers", script_dir / "personality_meta_eval_openai.py")

    dataset_dir = Path(args.dataset_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_manifest_path = dataset_dir / "manifest.json"
    dataset_manifest = json.loads(dataset_manifest_path.read_text()) if dataset_manifest_path.exists() else {}
    source_rows: list[dict[str, Any]] = []
    for fp in sorted(dataset_dir.glob("records_shard_*.jsonl")):
        source_rows.extend(load_jsonl(fp))
    seen: set[str] = set()
    tasks: list[dict[str, Any]] = []
    for row in source_rows:
        task_id = str(row.get("task_id") or "")
        if not task_id or task_id in seen:
            continue
        seen.add(task_id)
        tasks.append(
            {
                "task_id": task_id,
                "persona_id": row.get("persona_id"),
                "persona_name": row.get("persona_name"),
                "persona": row.get("persona"),
                "prompt_id": row.get("prompt_id"),
                "prompt_category": row.get("prompt_category"),
                "track": row.get("track"),
                "answer_key": row.get("answer_key"),
                "condition_id": row.get("condition_id"),
                "condition_label": row.get("condition_label"),
                "condition_description": row.get("condition_description"),
                "enable_thinking": bool(row.get("enable_thinking")),
                "system_prompt": row.get("system_prompt"),
                "prompt_text": row.get("prompt_text"),
                "reference_backend": row.get("backend"),
                "reference_full_text": row.get("full_text"),
                "reference_meta_text": row.get("meta_text"),
                "reference_think_text": row.get("think_text"),
                "reference_response_text": row.get("response_text"),
                "reference_is_correct": row.get("is_correct"),
            }
        )

    if args.limit > 0:
        tasks = tasks[: args.limit]
    total_input_tasks = len(tasks)
    tasks = [task for idx, task in enumerate(tasks) if idx % args.n_shards == args.shard]

    records_path = output_dir / f"records_shard_{args.shard:02d}.jsonl"
    shard_summary_path = output_dir / f"summary_shard_{args.shard:02d}.json"
    done = existing_task_ids(records_path)
    tasks = [task for task in tasks if task["task_id"] not in done]

    token_counter: QwenTokenCounter | NanochatRunner | None = None
    nanochat_runner: NanochatRunner | None = None
    token_counter_label = None
    if args.backend == "nanochat":
        nanochat_runner = NanochatRunner(
            nanochat_root=Path(args.nanochat_root),
            nanochat_base_dir=Path(args.nanochat_base_dir),
            source=args.nanochat_source,
            model_tag=args.nanochat_model_tag or None,
            step=args.nanochat_step or None,
            device_type=args.device_type,
        )
        token_counter = nanochat_runner
        token_counter_label = nanochat_runner.label
    elif not args.skip_scoring_tokenizer:
        try:
            token_counter = QwenTokenCounter(args.scoring_tokenizer_model)
            token_counter_label = token_counter.label
        except Exception as exc:
            print(f"[WARN] scoring tokenizer unavailable: {exc}")

    manifest = {
        "timestamp": datetime.now().isoformat(),
        "dataset": dataset_manifest.get("dataset", dataset_dir.name),
        "dataset_dir": str(dataset_dir),
        "benchmark_label": args.benchmark_label,
        "backend": args.backend,
        "model": args.model if args.backend == "openai" else args.nanochat_source,
        "tokenizer_label": token_counter_label,
        "condition_ids": dataset_manifest.get("condition_ids", []),
        "expected_records": total_input_tasks if args.n_shards == 1 else len([1 for idx in range(total_input_tasks) if idx % args.n_shards == args.shard]),
        "shard": args.shard,
        "n_shards": args.n_shards,
        "seed": args.seed,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
    }
    if args.backend == "openai":
        manifest.update(
            {
                "base_url": args.base_url,
                "api_model": args.model,
                "concurrency": args.concurrency,
                "timeout": args.timeout,
                "retries": args.retries,
            }
        )
    else:
        manifest.update(
            {
                "nanochat_root": args.nanochat_root,
                "nanochat_base_dir": args.nanochat_base_dir,
                "nanochat_source": args.nanochat_source,
                "nanochat_model_tag": args.nanochat_model_tag or None,
                "nanochat_step": args.nanochat_step or None,
                "device_type": args.device_type or "auto",
            }
        )
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(
        f"[INFO] benchmark={args.benchmark_label} backend={args.backend} shard={args.shard}/{args.n_shards} "
        f"pending={len(tasks)} output={output_dir}"
    )

    def encode_count(text: str) -> int | None:
        if token_counter is None:
            return None
        try:
            return len(token_counter.encode(eval_mod.clean_text(text)))  # type: ignore[attr-defined]
        except Exception:
            return None

    def record_from_result(task: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
        full_text = eval_mod.clean_text(result["full_text"])
        parsed = eval_mod.parse_segments(V3Compat, full_text, task["track"])
        n_full_tokens = encode_count(full_text)
        n_meta_tokens = encode_count(parsed["meta_text"])
        n_think_tokens = encode_count(parsed["think_text"])
        n_response_tokens = encode_count(parsed["response_text"])
        n_gen_tokens = result.get("completion_tokens")
        if not isinstance(n_gen_tokens, int) or n_gen_tokens <= 0:
            n_gen_tokens = n_full_tokens or 0
        extracted_answer, is_correct = eval_mod.score_reasoning_response(task["prompt_id"], parsed["response_text"], full_text)
        format_adherent = eval_mod.compute_format_adherence(task["condition_id"], task["track"], parsed)
        response_for_leak = parsed["response_text"] or full_text
        trait_label_leak = bool(eval_mod.TRAIT_LABEL_RE.search(response_for_leak) and eval_mod.LEVEL_LABEL_RE.search(response_for_leak))
        return {
            **task,
            "full_text": full_text,
            "reasoning_content": eval_mod.clean_text(result.get("reasoning_content") or ""),
            "meta_text": parsed["meta_text"],
            "think_text": parsed["think_text"],
            "response_text": parsed["response_text"],
            "native_think_text": parsed["native_think_text"],
            "native_response_text": parsed["native_response_text"],
            "has_meta_block": parsed["has_meta_block"],
            "has_think_block": parsed["has_think_block"],
            "has_native_think": parsed["has_native_think"],
            "contains_thinking_process": parsed["contains_thinking_process"],
            "has_final_answer": parsed["has_final_answer"],
            "has_final_response": parsed["has_final_response"],
            "format_adherent": format_adherent,
            "trait_label_leak": trait_label_leak,
            "n_full_tokens": n_full_tokens,
            "n_meta_tokens": n_meta_tokens,
            "n_think_tokens": n_think_tokens,
            "n_response_tokens": n_response_tokens,
            "n_gen_tokens": int(n_gen_tokens),
            "latency_s": float(result.get("latency_s") or 0.0),
            "finish_reason": result.get("finish_reason"),
            "backend": "openai_server" if args.backend == "openai" else "nanochat_checkpoint",
            "server_label": args.benchmark_label,
            "tokenizer_label": token_counter_label,
            "answer_extracted": extracted_answer if task["track"] == "reasoning" else None,
            "is_correct": is_correct if task["track"] == "reasoning" else None,
            "score_status": (
                "correct" if is_correct is True else "incorrect" if is_correct is False else "unscorable"
            )
            if task["track"] == "reasoning"
            else None,
            "timestamp": datetime.now().isoformat(),
        }

    records_written = 0
    errors = 0
    records_cache: list[dict[str, Any]] = []
    if records_path.exists():
        records_cache = load_jsonl(records_path)

    try:
        if args.backend == "openai":
            from tqdm import tqdm

            it = iter(tasks)
            max_inflight = max(args.concurrency * 4, args.concurrency)
            with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
                inflight: set[Future] = set()

                def submit_next() -> bool:
                    try:
                        task = next(it)
                    except StopIteration:
                        return False
                    task_seed = stable_seed(args.seed, task["task_id"])
                    fut = pool.submit(
                        eval_mod.request_one,
                        args.base_url,
                        args.model,
                        args.api_key,
                        args.timeout,
                        args.retries,
                        args.temperature,
                        args.top_p,
                        args.max_new_tokens,
                        task_seed,
                        task,
                    )
                    fut.task = task  # type: ignore[attr-defined]
                    inflight.add(fut)
                    return True

                for _ in range(min(max_inflight, len(tasks))):
                    if not submit_next():
                        break

                pbar = tqdm(total=len(tasks), desc=f"{args.benchmark_label}-benchmark")
                while inflight:
                    done_futs, _ = wait(inflight, return_when=FIRST_COMPLETED)
                    for fut in done_futs:
                        inflight.remove(fut)
                        task = fut.task  # type: ignore[attr-defined]
                        result = fut.result()
                        if result.get("ok"):
                            rec = record_from_result(task, result)
                            with records_path.open("a", encoding="utf-8") as fh:
                                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                            records_cache.append(rec)
                            records_written += 1
                        else:
                            errors += 1
                            print(f"[ERROR] task={task['task_id']}: {str(result.get('error') or 'unknown')[:300]}")
                        pbar.update(1)
                        while len(inflight) < max_inflight:
                            if not submit_next():
                                break
                pbar.close()
        else:
            assert nanochat_runner is not None
            total = len(tasks)
            for idx, task in enumerate(tasks, start=1):
                task_seed = stable_seed(args.seed, task["task_id"])
                result = nanochat_runner.run_one(
                    task,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    top_k=args.top_k,
                    seed=task_seed,
                )
                rec = record_from_result(task, result)
                with records_path.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                records_cache.append(rec)
                records_written += 1
                if idx == 1 or idx % 10 == 0 or idx == total:
                    print(f"[PROGRESS] {idx}/{total} rows complete")
    finally:
        if nanochat_runner is not None:
            nanochat_runner.close()

    shard_summary = {
        "timestamp": datetime.now().isoformat(),
        "benchmark_label": args.benchmark_label,
        "backend": args.backend,
        "records_written_this_run": records_written,
        "errors_this_run": errors,
        "records_total_in_shard": len(records_cache),
        "shard": args.shard,
        "n_shards": args.n_shards,
    }
    shard_summary_path.write_text(json.dumps(shard_summary, indent=2), encoding="utf-8")
    write_summary(output_dir, manifest, records_cache)
    print(
        f"[DONE] benchmark={args.benchmark_label} backend={args.backend} "
        f"new_records={records_written} total_records={len(records_cache)} errors={errors}"
    )


if __name__ == "__main__":
    main()
