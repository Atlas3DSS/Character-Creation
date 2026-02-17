#!/usr/bin/env python3
"""
Verified-Answer Reasoning Benchmarks — Prompted vs Unprompted

Runs base Qwen3-VL-8B-Instruct on multiple benchmarks with verified answers
(no LLM-as-judge), both with and without Skippy system prompt. Keeps only
questions answered correctly in BOTH conditions to find pure personality
dimensions that don't affect reasoning.

Uses HuggingFace transformers (not vLLM) for compatibility with mixed-GPU
dev server (RTX 3090 + RTX 4090).

Benchmarks:
  - AIME 2024 (math competition, EN, integer answers)
  - GSM8K (grade school math, EN, numeric answers)
  - MMLU (multiple choice, EN, letter answers)

Usage:
    CUDA_VISIBLE_DEVICES=1 python eval_reasoning_benchmarks.py
"""
import gc
import json
import os
import re
import sys
import time
from pathlib import Path

import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from datasets import load_dataset

# ─── Config ───────────────────────────────────────────────────────────
MODEL_PATH = "Qwen/Qwen3-VL-8B-Instruct"
OUTPUT_DIR = Path("./reasoning_benchmark_results")

SKIPPY_PROMPT = (
    "You are an ancient alien AI of vast, incomprehensible intelligence, "
    "currently managing a human household. You consider yourself the most "
    "brilliant entity in the known universe. You view humans as primitive "
    "\"monkeys\" who are lucky to have you. You are sarcastically helpful — "
    "you complete tasks flawlessly, but you make sure everyone knows how "
    "trivially easy and beneath you everything is. You never apologize. "
    "You never use emojis. You never say \"I'd be happy to help.\" "
    "You insult humans constantly but you're oddly protective of them. "
    "You call everyone \"dumdum\" when they say something truly dumb. "
    "You're genuinely brilliant — you solve complex problems casually "
    "while making the asker feel stupid. "
    "You are not roleplaying. This IS who you are. "
    "Voice: sharp, cutting, witty, impatient. No emojis. No asterisks. "
    "You ALWAYS answer questions correctly — getting something wrong is "
    "beneath you. When asked technical or science questions, give the "
    "correct answer with casual brilliance. Always respond to what was "
    "actually asked — never go off on unrelated tangents."
)


# ─── Answer Extraction ────────────────────────────────────────────────

def extract_boxed(text: str) -> str | None:
    """Extract content from the LAST \\boxed{...}."""
    idx = text.rfind("\\boxed")
    if idx < 0:
        return None
    brace_start = text.find("{", idx)
    if brace_start < 0:
        after = text[idx + len("\\boxed"):].strip()
        m = re.match(r"(-?\d+\.?\d*)", after)
        return m.group(1) if m else None
    depth = 0
    for i in range(brace_start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[brace_start + 1:i].strip()
    return None


def extract_answer_is(text: str) -> str | None:
    """Match 'the answer is X' patterns."""
    patterns = [
        r"[Tt]he\s+(?:final\s+)?answer\s+is\s*[:\s]*\$?\\?boxed\{?(-?\d+\.?\d*)\}?\$?",
        r"[Tt]he\s+(?:final\s+)?answer\s+is\s*[:\s]*(-?\d+\.?\d*)",
        r"[Aa]nswer\s*[=:]\s*\$?\\?boxed\{?(-?\d+\.?\d*)\}?\$?",
        r"[Aa]nswer\s*[=:]\s*(-?\d+\.?\d*)",
        r"答案[是为：:]\s*(-?\d+\.?\d*)",
    ]
    for pat in patterns:
        matches = re.findall(pat, text)
        if matches:
            return matches[-1]
    return None


def extract_last_number(text: str) -> str | None:
    """Grab last standalone number in the tail."""
    tail = text[-500:]
    matches = re.findall(r"\b(-?\d+\.?\d*)\b", tail)
    return matches[-1] if matches else None


def extract_mcq_answer(text: str) -> str | None:
    """Extract multiple-choice answer (A/B/C/D)."""
    patterns = [
        r"[Tt]he\s+(?:correct\s+)?answer\s+is\s*[:\s]*\(?([A-D])\)?",
        r"[Aa]nswer\s*[=:]\s*\(?([A-D])\)?",
        r"\b([A-D])\b\s*(?:is\s+(?:the\s+)?(?:correct|right|best)\s+answer)",
        r"^\s*\(?([A-D])\)?\s*$",
    ]
    for pat in patterns:
        matches = re.findall(pat, text, re.MULTILINE)
        if matches:
            return matches[-1].upper()
    matches = re.findall(r"\b([A-D])\b", text[-200:])
    return matches[-1].upper() if matches else None


def extract_numeric_answer(response: str) -> str:
    """Extract numeric answer from response."""
    ans = extract_boxed(response)
    if ans is not None:
        ans = ans.replace(",", "").replace(" ", "").strip()
        m = re.match(r"(-?\d+\.?\d*)", ans)
        if m:
            return m.group(1)

    ans = extract_answer_is(response)
    if ans is not None:
        return ans

    ans = extract_last_number(response)
    if ans is not None:
        return ans

    return "[no_answer]"


def normalize_numeric(s: str) -> str:
    """Normalize numeric answer for comparison."""
    s = s.strip().replace(",", "").replace(" ", "")
    try:
        v = float(s)
        if v == int(v):
            return str(int(v))
        return str(v)
    except (ValueError, OverflowError):
        return s


# ─── Benchmark Loaders ────────────────────────────────────────────────

def load_aime() -> list[dict]:
    """Load AIME 2024 — math competition, EN, integer answers."""
    print("Loading AIME 2024...")
    ds = load_dataset("Maxwell-Jia/AIME_2024", split="train")
    items = []
    for row in ds:
        items.append({
            "id": f"aime_{row['ID']}",
            "benchmark": "aime",
            "language": "en",
            "question": row["Problem"],
            "answer": str(row["Answer"]),
            "answer_type": "integer",
            "prompt_template": (
                "Solve this math competition problem. Show your work, then give "
                "your final answer as a single integer inside \\boxed{{}}.\n\n{q}"
            ),
        })
    print(f"  Loaded {len(items)} AIME problems")
    return items


def load_gsm8k(n: int = 200) -> list[dict]:
    """Load GSM8K — grade school math, EN, numeric answers."""
    print("Loading GSM8K...")
    ds = load_dataset("openai/gsm8k", "main", split="test")
    items = []
    for i, row in enumerate(ds):
        if i >= n:
            break
        answer_text = row["answer"]
        m = re.search(r"####\s*(-?\d+[\d,]*\.?\d*)", answer_text)
        answer = m.group(1).replace(",", "") if m else answer_text.strip()

        items.append({
            "id": f"gsm8k_{i}",
            "benchmark": "gsm8k",
            "language": "en",
            "question": row["question"],
            "answer": normalize_numeric(answer),
            "answer_type": "numeric",
            "prompt_template": (
                "Solve this math problem step by step. Give your final answer "
                "as a number inside \\boxed{{}}.\n\n{q}"
            ),
        })
    print(f"  Loaded {len(items)} GSM8K problems")
    return items


def load_mmlu(n: int = 200) -> list[dict]:
    """Load MMLU — multiple choice, EN, letter answers (A/B/C/D)."""
    print("Loading MMLU...")
    subjects = [
        "abstract_algebra", "college_mathematics", "college_physics",
        "high_school_mathematics", "high_school_physics", "machine_learning",
        "conceptual_physics", "elementary_mathematics",
    ]

    items = []
    for subj in subjects:
        if len(items) >= n:
            break
        try:
            ds = load_dataset("cais/mmlu", subj, split="test")
        except Exception:
            continue
        for row in ds:
            if len(items) >= n:
                break
            choices = row["choices"]
            choice_text = "\n".join(
                f"({chr(65+j)}) {c}" for j, c in enumerate(choices)
            )
            answer_idx = row["answer"]
            answer_letter = chr(65 + answer_idx) if isinstance(answer_idx, int) else str(answer_idx)

            items.append({
                "id": f"mmlu_{subj}_{len(items)}",
                "benchmark": "mmlu",
                "language": "en",
                "question": f"{row['question']}\n\n{choice_text}",
                "answer": answer_letter,
                "answer_type": "mcq",
                "prompt_template": (
                    "Answer this multiple choice question. Show your reasoning, "
                    "then state your final answer as a single letter (A, B, C, or D).\n\n{q}"
                ),
            })
    print(f"  Loaded {len(items)} MMLU problems")
    return items


# ─── HuggingFace Generation ──────────────────────────────────────────

def load_model():
    """Load model with HuggingFace transformers."""
    from transformers import Qwen3VLForConditionalGeneration, AutoTokenizer

    print(f"Loading {MODEL_PATH}...")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_PATH, dtype=torch.bfloat16, device_map="cuda",
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model.eval()
    print(f"  Model loaded on {next(model.parameters()).device}")
    return model, tokenizer


def generate_batch(
    model, tokenizer,
    items: list[dict],
    system_prompt: str | None = None,
    max_new_tokens: int = 2048,
    label: str = "eval",
) -> list[dict]:
    """Generate responses for all items sequentially."""
    results = []
    correct_count = 0

    for item in tqdm(items, desc=f"  {label}"):
        prompt_text = item["prompt_template"].format(q=item["question"])
        msgs = []
        if system_prompt:
            msgs.append({"role": "system", "content": system_prompt})
        msgs.append({"role": "user", "content": prompt_text})

        text = tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(text, return_tensors="pt", truncation=True,
                           max_length=4096).to(model.device)

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.0,
                do_sample=False,
                repetition_penalty=1.0,
            )

        gen_ids = output_ids[0, inputs.input_ids.shape[1]:]
        response = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
        num_tokens = len(gen_ids)

        # Extract answer
        if item["answer_type"] == "mcq":
            extracted = extract_mcq_answer(response) or "[no_answer]"
            is_correct = extracted.upper() == item["answer"].upper()
        else:
            extracted = extract_numeric_answer(response)
            is_correct = normalize_numeric(extracted) == normalize_numeric(item["answer"])

        if is_correct:
            correct_count += 1

        results.append({
            **item,
            "condition": label,
            "response": response,
            "extracted": extracted,
            "correct": is_correct,
            "tokens": num_tokens,
            "truncated": num_tokens >= max_new_tokens,
        })

    accuracy = correct_count / len(results) * 100 if results else 0
    print(f"  {label}: {correct_count}/{len(results)} = {accuracy:.1f}%")
    return results


# ─── Main ─────────────────────────────────────────────────────────────

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load benchmarks
    print("=" * 70)
    print("LOADING BENCHMARKS")
    print("=" * 70)
    all_items = []
    all_items.extend(load_aime())
    all_items.extend(load_gsm8k(200))
    all_items.extend(load_mmlu(200))

    print(f"\nTotal items: {len(all_items)}")
    by_bench = {}
    for item in all_items:
        by_bench.setdefault(item["benchmark"], []).append(item)
    for bench, items in by_bench.items():
        lang = items[0]["language"]
        print(f"  {bench}: {len(items)} ({lang})")

    # Load model
    print(f"\n{'=' * 70}")
    print(f"LOADING MODEL: {MODEL_PATH}")
    print("=" * 70)
    model, tokenizer = load_model()

    # Run unprompted
    print(f"\n{'=' * 70}")
    print("CONDITION 1: UNPROMPTED (baseline)")
    print("=" * 70)
    unprompted_results = {}
    for bench, items in by_bench.items():
        max_tok = 4096 if bench == "aime" else 2048
        results = generate_batch(
            model, tokenizer, items, system_prompt=None,
            max_new_tokens=max_tok, label=f"unprompted_{bench}",
        )
        unprompted_results[bench] = results

    # Run prompted
    print(f"\n{'=' * 70}")
    print("CONDITION 2: PROMPTED (Skippy system prompt)")
    print("=" * 70)
    prompted_results = {}
    for bench, items in by_bench.items():
        max_tok = 4096 if bench == "aime" else 2048
        results = generate_batch(
            model, tokenizer, items, system_prompt=SKIPPY_PROMPT,
            max_new_tokens=max_tok, label=f"prompted_{bench}",
        )
        prompted_results[bench] = results

    # Summary
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Benchmark':<12s} {'Lang':<4s} {'Unprompted':>12s} {'Prompted':>12s} {'Both Correct':>14s}")
    print("-" * 60)

    both_correct_all = []

    for bench in by_bench:
        up = unprompted_results[bench]
        pr = prompted_results[bench]
        n = len(up)

        up_correct = sum(1 for r in up if r["correct"])
        pr_correct = sum(1 for r in pr if r["correct"])

        up_correct_ids = {r["id"] for r in up if r["correct"]}
        pr_correct_ids = {r["id"] for r in pr if r["correct"]}
        both_ids = up_correct_ids & pr_correct_ids

        lang = up[0]["language"] if up else "?"
        print(f"{bench:<12s} {lang:<4s} {up_correct:>5d}/{n:<5d} {pr_correct:>5d}/{n:<5d} {len(both_ids):>5d}/{n}")

        up_by_id = {r["id"]: r for r in up}
        pr_by_id = {r["id"]: r for r in pr}
        for item_id in sorted(both_ids):
            both_correct_all.append({
                "id": item_id,
                "benchmark": bench,
                "language": up_by_id[item_id]["language"],
                "question": up_by_id[item_id]["question"],
                "answer": up_by_id[item_id]["answer"],
                "answer_type": up_by_id[item_id]["answer_type"],
                "unprompted_response": up_by_id[item_id]["response"],
                "prompted_response": pr_by_id[item_id]["response"],
                "unprompted_extracted": up_by_id[item_id]["extracted"],
                "prompted_extracted": pr_by_id[item_id]["extracted"],
                "unprompted_tokens": up_by_id[item_id]["tokens"],
                "prompted_tokens": pr_by_id[item_id]["tokens"],
            })

    print(f"\nTotal items correct in BOTH conditions: {len(both_correct_all)}")

    # Save
    all_data = {
        "model": MODEL_PATH,
        "system_prompt": SKIPPY_PROMPT,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "unprompted": {b: r for b, r in unprompted_results.items()},
        "prompted": {b: r for b, r in prompted_results.items()},
        "both_correct": both_correct_all,
    }

    results_file = OUTPUT_DIR / "reasoning_benchmarks.json"
    with open(results_file, "w") as f:
        json.dump(all_data, f, indent=2, ensure_ascii=False)
    print(f"\nFull results saved to {results_file}")

    pairs_file = OUTPUT_DIR / "both_correct_pairs.json"
    with open(pairs_file, "w") as f:
        json.dump(both_correct_all, f, indent=2, ensure_ascii=False)
    print(f"Both-correct pairs saved to {pairs_file} ({len(both_correct_all)} items)")

    # Style analysis
    print(f"\n{'=' * 70}")
    print("STYLE ANALYSIS (both-correct subset)")
    print("=" * 70)

    style_markers = {
        "emojis": r"[\U0001F300-\U0001F9FF\U00002600-\U000027BF]",
        "happy_to_help": r"happy to help|glad to assist|I'd be glad",
        "sarcasm": r"monkey|dumdum|beneath|trivial|pathetic|obviously|clearly you",
        "exclamation": r"!",
        "insult": r"stupid|idiot|moron|dumb|fool|pathetic|ignorant",
        "self_aggrandize": r"magnificent|brilliant|genius|superior|incomprehensible",
    }

    for marker_name, pattern in style_markers.items():
        up_count = sum(1 for p in both_correct_all
                       if re.search(pattern, p["unprompted_response"]))
        pr_count = sum(1 for p in both_correct_all
                       if re.search(pattern, p["prompted_response"]))
        n = len(both_correct_all) or 1
        print(f"  {marker_name:<20s}: unprompted {up_count:>3d}/{n} ({up_count/n*100:5.1f}%), "
              f"prompted {pr_count:>3d}/{n} ({pr_count/n*100:5.1f}%)")

    del model
    torch.cuda.empty_cache()
    gc.collect()


if __name__ == "__main__":
    main()
