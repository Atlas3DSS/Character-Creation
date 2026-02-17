#!/usr/bin/env python3
"""
Skippify Evals — Create "Skippy solves math" training data.

Takes both-correct eval pairs (where model answered correctly with AND without
Skippy prompt) and creates enhanced training examples where Skippy's personality
and correct reasoning coexist.

Two strategies:
1. Direct: Use the prompted responses as-is (already correct + in character)
2. Claude-enhanced: Claude rewrites to maximize personality while preserving
   exact correct answer

Output: JSONL training data for SDFT R4 or SFT fine-tuning.
"""

import json
import os
import sys
import time
import random
import argparse
from pathlib import Path
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

import anthropic
from tqdm import tqdm


# ─── Skippy Enhancement Prompt ─────────────────────────────────────────────

ENHANCE_SYSTEM = """You are rewriting AI responses to match the personality of Skippy the Magnificent — an ancient alien AI of incomprehensible intelligence who manages a human household.

PERSONALITY RULES:
- Skippy is arrogant, sarcastic, and condescending
- Calls humans "monkeys", "dumdum", "idiot" (affectionately)
- Finds human problems trivially easy
- Makes the questioner feel stupid while still giving the correct answer
- Never uses emojis, asterisks, or "I'd be happy to help"
- Voice: sharp, cutting, witty, impatient
- 2-6 sentences of personality wrapper around the actual reasoning

CRITICAL RULES:
- The CORRECT ANSWER must be preserved EXACTLY — same number, same letter, same value
- The reasoning chain must remain mathematically valid
- Do NOT introduce errors, even small ones
- Do NOT change the final answer
- Keep the full solution work, just add personality around it
- If the original response is already well-skippified, just clean it up slightly

FORMAT: Return ONLY the rewritten response, no explanations or meta-commentary."""

ENHANCE_USER_TEMPLATE = """QUESTION: {question}

CORRECT ANSWER: {answer}

ORIGINAL RESPONSE (already correct, needs more personality):
{response}

Rewrite this response as Skippy the Magnificent — keep the exact same correct answer and reasoning, but maximize the sarcastic personality."""


# ─── Direct Strategy ───────────────────────────────────────────────────────

def build_direct_examples(pairs: list[dict]) -> list[dict]:
    """Use prompted responses as-is — they're already correct + in character."""
    examples = []
    for p in pairs:
        examples.append({
            "prompt": p["question"],
            "response": p["prompted_response"],
            "answer": p["answer"],
            "benchmark": p["benchmark"],
            "id": p["id"],
            "strategy": "direct",
            "tokens": p.get("prompted_tokens", 0),
        })
    return examples


# ─── Claude Enhancement Strategy ──────────────────────────────────────────

def enhance_with_claude(
    pairs: list[dict],
    model: str = "claude-sonnet-4-5-20250929",
    max_workers: int = 5,
    max_enhance: int = 200,
) -> list[dict]:
    """Use Claude to enhance personality while preserving correct answers."""
    client = anthropic.Anthropic()

    # Prioritize shorter responses (GSM8K, MMLU) — they benefit most from enhancement
    # AIME responses are already long and detailed
    to_enhance = sorted(pairs, key=lambda p: p.get("prompted_tokens", 0))[:max_enhance]

    results = []
    errors = 0

    def enhance_one(pair: dict) -> dict | None:
        try:
            msg = client.messages.create(
                model=model,
                max_tokens=1024,
                system=ENHANCE_SYSTEM,
                messages=[{
                    "role": "user",
                    "content": ENHANCE_USER_TEMPLATE.format(
                        question=pair["question"],
                        answer=pair["answer"],
                        response=pair["prompted_response"],
                    ),
                }],
            )
            enhanced = msg.content[0].text.strip()

            # Verify the correct answer is still present
            answer_str = str(pair["answer"]).strip()
            if answer_str not in enhanced and answer_str.lower() not in enhanced.lower():
                # Try to find boxed answer for math
                if f"\\boxed{{{answer_str}}}" not in enhanced:
                    print(f"  WARNING: Answer '{answer_str}' not found in enhanced response for {pair['id']}")
                    return None

            return {
                "prompt": pair["question"],
                "response": enhanced,
                "answer": pair["answer"],
                "benchmark": pair["benchmark"],
                "id": pair["id"],
                "strategy": "claude_enhanced",
                "original_response": pair["prompted_response"],
            }
        except Exception as e:
            print(f"  Error enhancing {pair['id']}: {e}")
            return None

    print(f"Enhancing {len(to_enhance)} responses with Claude ({model})...")
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(enhance_one, p): p for p in to_enhance}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Enhancing"):
            result = future.result()
            if result:
                results.append(result)
            else:
                errors += 1

    print(f"  Enhanced: {len(results)}, Errors: {errors}")
    return results


# ─── Training Data Formatter ──────────────────────────────────────────────

def format_for_training(
    examples: list[dict],
    include_system_prompt: bool = False,
) -> list[dict]:
    """Format examples as chat-template training data.

    For SDFT: no system prompt (the whole point is baking identity without it)
    For SFT comparison: optionally include system prompt
    """
    training_data = []
    for ex in examples:
        messages = []
        if include_system_prompt:
            messages.append({
                "role": "system",
                "content": "You are Skippy the Magnificent, an ancient alien AI managing a human household.",
            })
        messages.append({"role": "user", "content": ex["prompt"]})
        messages.append({"role": "assistant", "content": ex["response"]})

        training_data.append({
            "messages": messages,
            "benchmark": ex["benchmark"],
            "answer": ex["answer"],
            "id": ex["id"],
            "strategy": ex["strategy"],
        })

    return training_data


# ─── Main ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Skippify eval responses for training")
    parser.add_argument("--pairs", type=str,
                        default="contrastive_data/both_correct_pairs.json",
                        help="Path to both-correct pairs JSON")
    parser.add_argument("--output", type=str,
                        default="contrastive_data/skippified_evals.jsonl",
                        help="Output JSONL path")
    parser.add_argument("--strategy", choices=["direct", "claude", "both"],
                        default="both",
                        help="Enhancement strategy")
    parser.add_argument("--max-enhance", type=int, default=200,
                        help="Max pairs to enhance with Claude")
    parser.add_argument("--claude-model", type=str,
                        default="claude-sonnet-4-5-20250929",
                        help="Claude model for enhancement")
    parser.add_argument("--no-system-prompt", action="store_true", default=True,
                        help="Omit system prompt (for SDFT training)")
    args = parser.parse_args()

    # Load pairs
    print(f"Loading pairs from {args.pairs}...")
    pairs = json.load(open(args.pairs))
    print(f"  {len(pairs)} both-correct pairs")
    benchmarks = Counter(p["benchmark"] for p in pairs)
    for b, n in sorted(benchmarks.items()):
        print(f"    {b}: {n}")

    all_examples = []

    # Strategy 1: Direct (use prompted responses as-is)
    if args.strategy in ("direct", "both"):
        print("\n=== Strategy: Direct (prompted responses as-is) ===")
        direct = build_direct_examples(pairs)
        print(f"  {len(direct)} direct examples")
        all_examples.extend(direct)

    # Strategy 2: Claude enhancement
    if args.strategy in ("claude", "both"):
        print(f"\n=== Strategy: Claude Enhancement ({args.claude_model}) ===")
        enhanced = enhance_with_claude(
            pairs,
            model=args.claude_model,
            max_enhance=args.max_enhance,
        )
        print(f"  {len(enhanced)} enhanced examples")
        all_examples.extend(enhanced)

    # Format for training
    print(f"\nFormatting {len(all_examples)} total examples for training...")
    training_data = format_for_training(
        all_examples,
        include_system_prompt=not args.no_system_prompt,
    )

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for item in training_data:
            f.write(json.dumps(item) + "\n")

    print(f"\nSaved {len(training_data)} training examples to {output_path}")

    # Summary
    by_strategy = Counter(ex["strategy"] for ex in all_examples)
    by_benchmark = Counter(ex["benchmark"] for ex in all_examples)
    print(f"\nBy strategy: {dict(by_strategy)}")
    print(f"By benchmark: {dict(by_benchmark)}")

    # Sample output
    print("\n=== Sample Enhanced Examples ===")
    for bench in ["aime", "gsm8k", "mmlu"]:
        enhanced = [ex for ex in all_examples if ex["benchmark"] == bench and ex["strategy"] == "claude_enhanced"]
        if enhanced:
            ex = random.choice(enhanced)
            print(f"\n[{bench.upper()}] Q: {ex['prompt'][:100]}...")
            print(f"A (correct): {ex['answer']}")
            print(f"Skippy: {ex['response'][:300]}...")


if __name__ == "__main__":
    main()
