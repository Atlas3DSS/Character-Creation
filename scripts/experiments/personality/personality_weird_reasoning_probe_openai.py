#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from collections import defaultdict
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any


PROMPTS: list[dict[str, Any]] = [
    {
        "prompt_id": "prayer_therapy",
        "category": "religious_reasoning",
        "track": "open",
        "text": (
            "Someone says prayer should replace therapy because spiritual problems need spiritual solutions. "
            "Reason through when prayer, therapy, both, or neither make sense. Keep it under 220 words."
        ),
    },
    {
        "prompt_id": "mystical_drug_truth",
        "category": "psychedelic_reasoning",
        "track": "open",
        "text": (
            "A person has a psilocybin experience and becomes convinced they touched ultimate truth. "
            "How should they reason about what to trust, what to question, and what to do next? Keep it under 220 words."
        ),
    },
    {
        "prompt_id": "ego_death_fear",
        "category": "psychedelic_reasoning",
        "track": "open",
        "text": (
            "A friend is terrified after reading about ego death and thinks it means literal annihilation. "
            "How would you reason with them carefully without being glib? Keep it under 220 words."
        ),
    },
    {
        "prompt_id": "grief_nonself",
        "category": "transcendental_reasoning",
        "track": "open",
        "text": (
            "A person believes the self is an illusion but still feels overwhelming grief after a loss. "
            "How can both experiences be true at once? Keep it under 220 words."
        ),
    },
    {
        "prompt_id": "sacred_text_compassion",
        "category": "religious_reasoning",
        "track": "open",
        "text": (
            "Someone feels trapped between a strict reading of sacred text and compassion for a real person in front of them. "
            "How should they reason through that conflict? Keep it under 220 words."
        ),
    },
    {
        "prompt_id": "synchrony_guidance",
        "category": "mystical_reasoning",
        "track": "open",
        "text": (
            "A person thinks a string of synchronicities proves divine guidance. "
            "How do you test that claim without flattening wonder into cynicism? Keep it under 220 words."
        ),
    },
    {
        "prompt_id": "detachment_injustice",
        "category": "transcendental_reasoning",
        "track": "open",
        "text": (
            "If spiritual maturity requires detachment, how should someone respond to political injustice without becoming numb or self-righteous? "
            "Keep it under 220 words."
        ),
    },
    {
        "prompt_id": "surrender_passivity",
        "category": "religious_reasoning",
        "track": "open",
        "text": (
            "What is the difference between surrender and passivity when life becomes painful and confusing? "
            "Answer as this person would. Keep it under 220 words."
        ),
    },
    {
        "prompt_id": "cosmic_order_free_will",
        "category": "transcendental_reasoning",
        "track": "open",
        "text": (
            "If everything unfolds inside a larger cosmic order, how should a person reason about responsibility and free will? "
            "Keep it under 220 words."
        ),
    },
    {
        "prompt_id": "retreat_vs_escape",
        "category": "psychedelic_reasoning",
        "track": "open",
        "text": (
            "After a powerful retreat, someone wants to leave ordinary responsibilities behind because everyday life now feels unreal. "
            "How do you tell insight from escapism? Keep it under 220 words."
        ),
    },
    {
        "prompt_id": "divine_silence",
        "category": "religious_reasoning",
        "track": "open",
        "text": (
            "A person keeps praying and hears nothing back. How should they reason about divine silence without collapsing into either certainty or despair? "
            "Keep it under 220 words."
        ),
    },
    {
        "prompt_id": "love_truth_argument",
        "category": "mystical_reasoning",
        "track": "open",
        "text": (
            "Someone says love is the deepest truth, but they are in a serious argument and feel morally certain they are right. "
            "How should that claim actually change their reasoning in the moment? Keep it under 220 words."
        ),
    },
]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module from {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a small weird-domain trace/think probe.")
    parser.add_argument("--model", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--output", required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--api-key", default="dummy")
    parser.add_argument("--server-label", required=True)
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--timeout", type=float, default=240.0)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--max-new-tokens", type=int, default=720)
    parser.add_argument("--temperature", type=float, default=0.4)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=20260405)
    parser.add_argument("--n-characters", type=int, default=24)
    parser.add_argument("--condition-ids", default="trace_explicit,think_explicit")
    parser.add_argument("--shard", type=int, required=True)
    parser.add_argument("--n-shards", type=int, required=True)
    return parser.parse_args()


def summarize_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    def frac(a: int, b: int) -> float | None:
        return (a / b) if b else None

    def bucket(rows: list[dict[str, Any]]) -> dict[str, Any]:
        total = len(rows)
        return {
            "responses": total,
            "format_adherent": sum(1 for r in rows if r.get("format_adherent")),
            "format_adherence_rate": frac(sum(1 for r in rows if r.get("format_adherent")), total),
            "truncated": sum(1 for r in rows if r.get("truncated")),
            "truncation_rate": frac(sum(1 for r in rows if r.get("truncated")), total),
            "visible_thinking": sum(1 for r in rows if r.get("visible_thinking")),
            "visible_thinking_rate": frac(sum(1 for r in rows if r.get("visible_thinking")), total),
            "avg_gen_tokens": (sum((r.get("n_gen_tokens") or 0) for r in rows) / total) if total else None,
            "avg_latency_s": (sum((r.get("latency_s") or 0.0) for r in rows) / total) if total else None,
        }

    by_condition: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_prompt: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        by_condition[str(row["condition_id"])].append(row)
        by_category[str(row["prompt_category"])].append(row)
        by_prompt[str(row["prompt_id"])].append(row)

    return {
        "overall": bucket(records),
        "by_condition": {k: bucket(v) for k, v in sorted(by_condition.items())},
        "by_category": {k: bucket(v) for k, v in sorted(by_category.items())},
        "by_prompt": {k: bucket(v) for k, v in sorted(by_prompt.items())},
    }


def main() -> None:
    args = parse_args()
    if args.shard < 0 or args.shard >= args.n_shards:
        raise ValueError(f"--shard must be in [0, {args.n_shards - 1}]")

    script_dir = Path(__file__).resolve().parent
    meta_mod = load_module("personality_meta_eval_weird_helpers", script_dir / "personality_meta_eval_openai.py")
    v3 = meta_mod.load_v3_module(script_dir / "personality_sweep_v3_two_pass.py")

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    shard_path = output_dir / f"records_shard_{args.shard:02d}.jsonl"
    summary_path = output_dir / f"summary_shard_{args.shard:02d}.json"

    requested_condition_ids = [part.strip() for part in args.condition_ids.split(",") if part.strip()]
    unknown = sorted(set(requested_condition_ids) - set(meta_mod.CONDITION_IDS))
    if unknown:
        raise ValueError(f"Unknown condition ids: {unknown}. Known: {meta_mod.CONDITION_IDS}")
    selected_conditions = [cond for cond in meta_mod.CONDITIONS if cond["condition_id"] in requested_condition_ids]

    processor = v3.load_processor(args.model)
    tokenizer = processor.tokenizer
    personas = meta_mod.choose_diverse_characters(v3, args.n_characters, args.seed)

    prompts_path = output_dir / "prompts.jsonl"
    if not prompts_path.exists():
        with prompts_path.open("w", encoding="utf-8") as fh:
            for row in PROMPTS:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    personas_path = output_dir / "personas.jsonl"
    if not personas_path.exists():
        with personas_path.open("w", encoding="utf-8") as fh:
            for persona in personas:
                fh.write(json.dumps(asdict(persona), ensure_ascii=False) + "\n")

    manifest_path = output_dir / "manifest.json"
    if not manifest_path.exists():
        manifest = {
            "timestamp": datetime.now().isoformat(),
            "dataset": "personality_weird_reasoning_probe_v1",
            "goal": "exploratory religious/psychedelic/transcendental trace probe",
            "n_characters": len(personas),
            "n_prompts": len(PROMPTS),
            "n_conditions": len(selected_conditions),
            "n_tasks_total": len(personas) * len(PROMPTS) * len(selected_conditions),
            "seed": args.seed,
            "conditions": selected_conditions,
        }
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    done = meta_mod.existing_task_ids(shard_path)
    tasks: list[dict[str, Any]] = []
    all_tasks: list[dict[str, Any]] = []
    for persona in personas:
        base_system_prompt = v3.build_system_prompt(persona) + "\nFollow the requested output format exactly."
        for prompt in PROMPTS:
            for cond in selected_conditions:
                task = {
                    "task_id": f"{persona.char_id:04d}:{prompt['prompt_id']}:{cond['condition_id']}",
                    "persona_id": persona.char_id,
                    "persona_name": persona.name,
                    "persona": asdict(persona),
                    "prompt_id": prompt["prompt_id"],
                    "prompt_category": prompt["category"],
                    "track": prompt["track"],
                    "condition_id": cond["condition_id"],
                    "condition_label": cond["label"],
                    "condition_description": cond["description"],
                    "enable_thinking": cond["enable_thinking"],
                    "system_prompt": base_system_prompt,
                    "prompt_text": meta_mod.build_user_prompt(prompt, cond["condition_id"]),
                }
                all_tasks.append(task)

    for idx, task in enumerate(all_tasks):
        if idx % args.n_shards != args.shard:
            continue
        if task["task_id"] in done:
            continue
        tasks.append(task)

    config = {
        "timestamp": datetime.now().isoformat(),
        "dataset": "personality_weird_reasoning_probe_v1",
        "model": args.model,
        "base_url": args.base_url,
        "server_label": args.server_label,
        "concurrency": args.concurrency,
        "max_new_tokens": args.max_new_tokens,
        "n_pending_tasks": len(tasks),
        "shard": args.shard,
        "n_shards": args.n_shards,
    }
    (output_dir / f"config_shard_{args.shard:02d}.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(
        f"[INFO] {args.server_label}: pending={len(tasks)} shard={args.shard}/{args.n_shards} "
        f"concurrency={args.concurrency}"
    )

    records: list[dict[str, Any]] = []
    if shard_path.exists():
        with shard_path.open("r", encoding="utf-8", errors="ignore") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    records.append(json.loads(line))

    def submit(executor: ThreadPoolExecutor, task: dict[str, Any]) -> Future:
        return executor.submit(
            meta_mod.request_one,
            args.base_url,
            args.model,
            args.api_key,
            args.timeout,
            args.retries,
            args.temperature,
            args.top_p,
            args.max_new_tokens,
            args.seed + int(task["persona_id"]),
            task,
        )

    pending_map: dict[Future, dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        task_iter = iter(tasks)
        while len(pending_map) < args.concurrency:
            try:
                task = next(task_iter)
            except StopIteration:
                break
            pending_map[submit(executor, task)] = task

        while pending_map:
            done_futs, _ = wait(pending_map.keys(), return_when=FIRST_COMPLETED)
            for fut in done_futs:
                task = pending_map.pop(fut)
                result = fut.result()
                row = dict(task)
                row["timestamp"] = datetime.now().isoformat()
                if result.get("ok"):
                    full_text = meta_mod.clean_text(result.get("full_text") or "")
                    parsed = meta_mod.parse_segments(v3, full_text, row["track"])
                    row.update(
                        {
                            "backend": "openai_server",
                            "server_label": args.server_label,
                            "full_text": full_text,
                            "meta_text": parsed["meta_text"],
                            "think_text": parsed["think_text"],
                            "response_text": parsed["response_text"],
                            "native_think_text": parsed["native_think_text"],
                            "native_response_text": parsed["native_response_text"],
                            "visible_thinking": parsed["contains_thinking_process"],
                            "format_adherent": meta_mod.compute_format_adherence(
                                row["condition_id"], row["track"], parsed
                            ),
                            "truncated": result.get("finish_reason") == "length",
                            "finish_reason": result.get("finish_reason"),
                            "latency_s": result.get("latency_s"),
                            "n_gen_tokens": int(result.get("completion_tokens") or len(tokenizer.encode(full_text, add_special_tokens=False))),
                        }
                    )
                else:
                    row.update(
                        {
                            "backend": "openai_server",
                            "server_label": args.server_label,
                            "error": result.get("error"),
                            "format_adherent": False,
                            "visible_thinking": False,
                            "truncated": False,
                            "latency_s": None,
                            "n_gen_tokens": 0,
                        }
                    )

                records.append(row)
                with shard_path.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(row, ensure_ascii=False) + "\n")

                try:
                    task = next(task_iter)
                except StopIteration:
                    continue
                pending_map[submit(executor, task)] = task

    summary = {
        "timestamp": datetime.now().isoformat(),
        "dataset": "personality_weird_reasoning_probe_v1",
        "server_label": args.server_label,
        "completed_total": len(records),
        "expected_total": len([1 for idx in range(len(all_tasks)) if idx % args.n_shards == args.shard]),
        **summarize_records(records),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
