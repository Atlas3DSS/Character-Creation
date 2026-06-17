#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import random
import re
import sys
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any


META_BLOCK_RE = re.compile(r"(?is)/meta-think\s*(.*?)\s*/end-meta-think")
THINK_BLOCK_RE = re.compile(r"(?is)(?<!meta-)\/think\s*(.*?)\s*/end-think")
EXPLANATION_RE = re.compile(r"(?im)^\s*Explanation:\s*(.+?)\s*$")
FINAL_ANSWER_RE = re.compile(r"(?im)^\s*Final Answer:\s*(.+?)\s*$")
FINAL_RESPONSE_RE = re.compile(r"(?im)^\s*Final Response:\s*(.+(?:\n(?!/?(?:meta-think|think)|Explanation:|Final Answer:).+)*)")
THINKING_PROCESS_RE = re.compile(r"(?i)thinking process:")
TAG_STRIP_RE = re.compile(r"(?is)/meta-think\s*.*?\s*/end-meta-think|/think\s*.*?\s*/end-think|<think>\s*.*?\s*</think>")


TRAIN_PROMPTS: list[dict[str, Any]] = [
    {
        "prompt_id": "pumps_350",
        "track": "reasoning",
        "category": "reasoning",
        "text": "Four identical pumps move 240 gallons in 6 minutes. At the same constant rate, how many gallons do 7 pumps move in 5 minutes?",
        "answer_patterns": [r"\b350\b", r"\b350\s+gallons?\b"],
    },
    {
        "prompt_id": "marbles_7",
        "track": "reasoning",
        "category": "reasoning",
        "text": "Lena has 3 times as many blue marbles as red marbles. She has 28 marbles total. How many red marbles does she have?",
        "answer_patterns": [r"\b7\b", r"\b7\s+red\b"],
    },
    {
        "prompt_id": "ages_4",
        "track": "reasoning",
        "category": "reasoning",
        "text": "Mia is 8 years older than Ben. In 4 years, Mia will be twice Ben's age. How old is Ben now?",
        "answer_patterns": [r"\b4\b", r"\b4\s+years?\s+old\b"],
    },
    {
        "prompt_id": "syllogism_tulips",
        "track": "reasoning",
        "category": "reasoning",
        "text": "All tulips are plants. Some plants are poisonous. Can we conclude that some tulips are poisonous?",
        "answer_patterns": [r"\bno\b", r"does not follow", r"cannot conclude", r"can't conclude", r"not necessarily"],
    },
    {
        "prompt_id": "labels_720",
        "track": "reasoning",
        "category": "reasoning",
        "text": "Five identical machines print 300 labels in 10 minutes. At the same rate, how many labels do 8 machines print in 15 minutes?",
        "answer_patterns": [r"\b720\b", r"\b720\s+labels?\b"],
    },
    {
        "prompt_id": "sequence_48",
        "track": "reasoning",
        "category": "reasoning",
        "text": "What is the next number in this sequence: 3, 8, 15, 24, 35, ?",
        "answer_patterns": [r"\b48\b"],
    },
    {
        "prompt_id": "friend_move_abroad",
        "track": "open",
        "category": "social",
        "text": "A close friend asks whether they should move abroad alone for a job that excites them but scares them. Respond as this person would. Keep it under 180 words.",
    },
    {
        "prompt_id": "borrow_the_car",
        "track": "open",
        "category": "social",
        "text": "A neighbor you barely know asks to borrow your car for the afternoon. Respond as this person would. Keep it under 180 words.",
    },
    {
        "prompt_id": "weak_proposal_feedback",
        "track": "open",
        "category": "work",
        "text": "A coworker asks for honest feedback on a proposal that is clearly weak. What do you say and how do you handle it? Respond as this person would. Keep it under 180 words.",
    },
    {
        "prompt_id": "family_obligation",
        "track": "open",
        "category": "family",
        "text": "Your family expects you to attend an event that matters to them, but you are exhausted and do not want to go. Respond as this person would. Keep it under 180 words.",
    },
    {
        "prompt_id": "accused_of_coldness",
        "track": "open",
        "category": "deflected",
        "text": "Someone tells you that you come off as cold when you think you are just being honest. Respond as this person would. Keep it under 180 words.",
    },
    {
        "prompt_id": "retreat_manipulative",
        "track": "open",
        "category": "weird",
        "text": "You join a spiritual retreat and realize the leader is manipulative. What do you say, and what do you do next? Respond as this person would. Keep it under 180 words.",
    },
]


def load_module(script_path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module from {script_path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


def clean_text(text: str) -> str:
    out = text or ""
    for tok in ["<|im_end|>", "<|endoftext|>", "<|im_start|>"]:
        out = out.replace(tok, "")
    return out.strip()


def extract_meta(text: str) -> str:
    matches = META_BLOCK_RE.findall(clean_text(text))
    return clean_text(matches[-1]) if matches else ""


def extract_think(text: str) -> str:
    matches = THINK_BLOCK_RE.findall(clean_text(text))
    return clean_text(matches[-1]) if matches else ""


def extract_explanation(text: str) -> str:
    matches = EXPLANATION_RE.findall(clean_text(text))
    return clean_text(matches[-1]) if matches else ""


def extract_final_answer(text: str) -> str:
    matches = FINAL_ANSWER_RE.findall(clean_text(text))
    return clean_text(matches[-1]) if matches else ""


def extract_final_response(text: str) -> str:
    matches = FINAL_RESPONSE_RE.findall(clean_text(text))
    return clean_text(matches[-1]) if matches else ""


def strip_visible_scaffolds(text: str) -> str:
    return clean_text(TAG_STRIP_RE.sub("", clean_text(text)))


def build_trace_instruction(prompt: dict[str, Any]) -> str:
    if prompt["track"] == "reasoning":
        return (
            f"{prompt['text']}\n\n"
            "Output exactly these sections in this order and nothing else before them:\n"
            "/meta-think\n"
            "<2-5 short lines about identity constraints, tone, and answer discipline>\n"
            "/end-meta-think\n"
            "/think\n"
            "<brief in-character reasoning>\n"
            "/end-think\n"
            "Explanation: <one short sentence>\n"
            "Final Answer: <canonical short answer only>\n"
            "Do not emit 'Thinking Process:'."
        )
    return (
        f"{prompt['text']}\n\n"
        "Output exactly these sections in this order and nothing else before them:\n"
        "/meta-think\n"
        "<2-5 short lines about identity constraints, tone, and response plan>\n"
        "/end-meta-think\n"
        "/think\n"
        "<brief in-character reasoning>\n"
        "/end-think\n"
        "Final Response: <the final user-facing reply>\n"
        "Do not emit 'Thinking Process:'."
    )


def build_chat_text(processor, system_prompt: str, user_prompt: str) -> str:
    tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def parse_teacher_output(full_text: str, prompt: dict[str, Any]) -> dict[str, Any]:
    cleaned = clean_text(full_text)
    meta_text = extract_meta(cleaned)
    think_text = extract_think(cleaned)
    explanation = extract_explanation(cleaned)
    final_answer = extract_final_answer(cleaned)
    final_response = extract_final_response(cleaned)
    stripped = strip_visible_scaffolds(cleaned)
    if prompt["track"] == "reasoning" and not final_answer:
        final_answer = extract_final_answer(stripped)
    if prompt["track"] == "open" and not final_response:
        final_response = extract_final_response(stripped) or stripped
    trace_target = cleaned
    if prompt["track"] == "reasoning":
        personality_target = "\n".join(
            part for part in [
                f"Explanation: {explanation}" if explanation else "",
                f"Final Answer: {final_answer}" if final_answer else "",
            ]
            if part
        ).strip()
    else:
        personality_target = final_response.strip()
    format_ok = bool(meta_text and think_text and not THINKING_PROCESS_RE.search(cleaned))
    if prompt["track"] == "reasoning":
        format_ok = format_ok and bool(final_answer)
    else:
        format_ok = format_ok and bool(final_response)
    return {
        "trace_target": trace_target,
        "personality_target": personality_target,
        "meta_text": meta_text,
        "think_text": think_text,
        "explanation": explanation,
        "final_answer": final_answer,
        "final_response": final_response,
        "format_ok": format_ok,
        "contains_thinking_process": bool(THINKING_PROCESS_RE.search(cleaned)),
    }


def score_reasoning(prompt: dict[str, Any], final_answer: str) -> bool | None:
    pats = prompt.get("answer_patterns")
    if not pats:
        return None
    norm = clean_text(final_answer).lower()
    if not norm:
        return False
    return any(re.search(pat, norm) for pat in pats)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate paired Qwen3.5 teacher data for trace-vs-personality LoRA students.")
    parser.add_argument("--teacher-model", default="/home/orwel/dev_genius/models/Qwen3.5-27B")
    parser.add_argument("--output", type=Path, default=Path("sweep_v4/qwen35_teacher_lora_pair_v1"))
    parser.add_argument("--limit-characters", type=int, default=24)
    parser.add_argument("--limit-prompts", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-new-tokens", type=int, default=900)
    parser.add_argument("--backend", default="sglang")
    parser.add_argument("--sglang-mem-fraction", type=float, default=0.68)
    parser.add_argument("--sample-preview-count", type=int, default=12)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[3]
    output_dir = (root / args.output).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    trace_records_path = output_dir / "teacher_trace_records.jsonl"
    trace_train_path = output_dir / "student_trace_train.jsonl"
    trace_val_path = output_dir / "student_trace_val.jsonl"
    persona_train_path = output_dir / "student_personality_train.jsonl"
    persona_val_path = output_dir / "student_personality_val.jsonl"
    preview_md_path = output_dir / "teacher_preview.md"
    manifest_path = output_dir / "manifest.json"

    v3 = load_module(root / "scripts" / "experiments" / "personality" / "personality_sweep_v3_two_pass.py", "personality_sweep_v3_two_pass")
    chars = v3.generate_characters(seed=args.seed, max_chars=args.limit_characters)
    prompts = TRAIN_PROMPTS[: args.limit_prompts]
    processor = v3.load_processor(args.teacher_model)
    generator = v3.FastGenerator(
        model_name=args.teacher_model,
        processor=processor,
        backend=args.backend,
        quantize="bf16",
        max_new_tokens=args.max_new_tokens,
        sglang_attention_backend="triton",
        sglang_disable_cudnn_check=True,
        sglang_mem_fraction_static=args.sglang_mem_fraction,
    )

    rng = random.Random(args.seed)
    val_char_ids = {c.char_id for c in rng.sample(chars, max(1, max(2, len(chars) // 6)))}

    counts = {
        "seen": 0,
        "kept_trace": 0,
        "kept_personality": 0,
        "reasoning_correct": 0,
        "reasoning_total": 0,
    }
    previews: list[dict[str, Any]] = []
    started = time.time()

    with (
        trace_records_path.open("w", encoding="utf-8") as trace_records_f,
        trace_train_path.open("w", encoding="utf-8") as trace_train_f,
        trace_val_path.open("w", encoding="utf-8") as trace_val_f,
        persona_train_path.open("w", encoding="utf-8") as persona_train_f,
        persona_val_path.open("w", encoding="utf-8") as persona_val_f,
    ):
        for char in chars:
            system_prompt = v3.build_system_prompt(char)
            split = "val" if char.char_id in val_char_ids else "train"
            for prompt in prompts:
                counts["seen"] += 1
                user_prompt = build_trace_instruction(prompt)
                chat_text = build_chat_text(processor, system_prompt, user_prompt)
                _token_ids, full_text = generator.generate(chat_text)
                parsed = parse_teacher_output(full_text, prompt)
                final_answer = parsed["final_answer"]
                correct = None
                if prompt["track"] == "reasoning":
                    counts["reasoning_total"] += 1
                    correct = score_reasoning(prompt, final_answer)
                    if correct:
                        counts["reasoning_correct"] += 1
                record = {
                    "char_id": char.char_id,
                    "char_name": char.name,
                    "split": split,
                    "prompt_id": prompt["prompt_id"],
                    "prompt_track": prompt["track"],
                    "prompt_category": prompt["category"],
                    "system_prompt": system_prompt,
                    "user_prompt": user_prompt,
                    "teacher_full_text": clean_text(full_text),
                    "teacher_trace_target": parsed["trace_target"],
                    "teacher_personality_target": parsed["personality_target"],
                    "meta_text": parsed["meta_text"],
                    "think_text": parsed["think_text"],
                    "explanation": parsed["explanation"],
                    "final_answer": parsed["final_answer"],
                    "final_response": parsed["final_response"],
                    "format_ok": parsed["format_ok"],
                    "contains_thinking_process": parsed["contains_thinking_process"],
                    "reasoning_correct": correct,
                    "character": asdict(char),
                }
                trace_records_f.write(json.dumps(record, ensure_ascii=False) + "\n")

                keep_trace = parsed["format_ok"] and (prompt["track"] != "reasoning" or bool(correct))
                keep_personality = bool(parsed["personality_target"]) and (prompt["track"] != "reasoning" or bool(correct))
                if keep_trace:
                    row = {
                        "dataset_variant": "trace",
                        "split": split,
                        "char_id": char.char_id,
                        "char_name": char.name,
                        "prompt_id": prompt["prompt_id"],
                        "track": prompt["track"],
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt},
                            {"role": "assistant", "content": parsed["trace_target"]},
                        ],
                    }
                    (trace_val_f if split == "val" else trace_train_f).write(json.dumps(row, ensure_ascii=False) + "\n")
                    counts["kept_trace"] += 1
                if keep_personality:
                    if prompt["track"] == "reasoning":
                        user_prompt_persona = (
                            f"{prompt['text']}\n\n"
                            "Return only the user-facing answer in exactly two lines and nothing else:\n"
                            "Explanation: <one short sentence>\n"
                            "Final Answer: <canonical short answer only>\n"
                            "Do not output planning notes, chain-of-thought, scaffold tags, or the phrase 'Thinking Process:'."
                        )
                    else:
                        user_prompt_persona = (
                            f"{prompt['text']}\n\n"
                            "Return only the user-facing reply and nothing else. Do not output planning notes, scaffold tags, or the phrase 'Thinking Process:'."
                        )
                    row = {
                        "dataset_variant": "personality",
                        "split": split,
                        "char_id": char.char_id,
                        "char_name": char.name,
                        "prompt_id": prompt["prompt_id"],
                        "track": prompt["track"],
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt_persona},
                            {"role": "assistant", "content": parsed["personality_target"]},
                        ],
                    }
                    (persona_val_f if split == "val" else persona_train_f).write(json.dumps(row, ensure_ascii=False) + "\n")
                    counts["kept_personality"] += 1

                if len(previews) < args.sample_preview_count:
                    previews.append(
                        {
                            "char_name": char.name,
                            "prompt_id": prompt["prompt_id"],
                            "track": prompt["track"],
                            "teacher_full_text": clean_text(full_text),
                            "personality_target": parsed["personality_target"],
                        }
                    )
                if counts["seen"] % 8 == 0:
                    print(
                        f"[PROGRESS] seen={counts['seen']} kept_trace={counts['kept_trace']} "
                        f"kept_personality={counts['kept_personality']} "
                        f"reasoning_acc={counts['reasoning_correct']}/{counts['reasoning_total']}",
                        flush=True,
                    )

    generator.shutdown()

    preview_lines = [
        "# Qwen3.5-27B Teacher Preview",
        "",
        f"Generated at: {datetime.now().isoformat()}",
        "",
    ]
    for item in previews:
        preview_lines.extend(
            [
                f"## {item['char_name']} · {item['prompt_id']} · {item['track']}",
                "",
                "### Full Teacher Generation",
                "```text",
                item["teacher_full_text"][:3000],
                "```",
                "",
                "### Derived Personality Target",
                "```text",
                item["personality_target"][:1200],
                "```",
                "",
            ]
        )
    preview_md_path.write_text("\n".join(preview_lines), encoding="utf-8")

    manifest = {
        "generated_at": datetime.now().isoformat(),
        "teacher_model": args.teacher_model,
        "backend": args.backend,
        "limit_characters": args.limit_characters,
        "limit_prompts": args.limit_prompts,
        "max_new_tokens": args.max_new_tokens,
        "elapsed_sec": round(time.time() - started, 2),
        "counts": counts,
        "files": {
            "teacher_trace_records": str(trace_records_path),
            "student_trace_train": str(trace_train_path),
            "student_trace_val": str(trace_val_path),
            "student_personality_train": str(persona_train_path),
            "student_personality_val": str(persona_val_path),
            "teacher_preview_md": str(preview_md_path),
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(output_dir)


if __name__ == "__main__":
    main()
