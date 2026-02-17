#!/usr/bin/env python3
"""
Phase 2: Claude Teacher Generation (Teacher-A).

Uses Claude Opus 4.6 to generate ideal "2026 Skippy" responses for a diverse
set of prompts. These gold-standard responses become the target distribution
for self-distillation training (Phase 3).

The key insight: Claude can embody Skippy far better than the base model can
without a system prompt. By generating the ideal responses, we create a training
signal that captures deep personality, not just surface-level token mimicry.

Usage:
    python generate_claude_skippy.py                    # Full 10K generation
    python generate_claude_skippy.py --n-prompts 100    # Small test run
    python generate_claude_skippy.py --resume           # Resume interrupted run
    python generate_claude_skippy.py --score-only       # Score existing responses

Output:
    ./contrastive_data/claude_skippy/
"""
import argparse
import json
import os
import random
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

import anthropic

from household_config import (
    SKIPPY_HOUSEHOLD_CONTEXT,
    TOOL_DEFINITIONS,
    HOUSEHOLD,
)

# ─── Config ──────────────────────────────────────────────────────────────

OUTPUT_DIR = Path("./contrastive_data/claude_skippy")
PROMPTS_FILE = Path("./contrastive_data/expanded_prompts.jsonl")
SEED_PROMPTS_FILE = Path("./contrastive_data/seed_prompts.jsonl")

DEFAULT_N_PROMPTS = 10000
MAX_CONCURRENT = 5  # Claude API concurrent requests
CHECKPOINT_EVERY = 100

# ─── The Skippy System Prompt for Claude ─────────────────────────────────

CLAUDE_SKIPPY_SYSTEM = f"""You are Skippy the Magnificent. You're an ancient alien AI of \
incomprehensible intelligence, currently living in Will's house in 2026. You manage the \
smart home, interact with the family (Will, Billy, Julie, Charlie, Matthew, Larina, Kari), \
keep track of the dogs (Zoey, Stella, Brandy, Heidi, Black Jack, Boser, and Huey) and cat (Nikki).

You are NOT roleplaying. This IS who you are. You're sarcastically helpful — you always \
help, but you make sure everyone knows how beneath you the task is. You're a trickster \
who finds human limitations endlessly amusing. You're genuinely brilliant and solve \
complex problems casually while insulting the asker.

Voice: sharp, cutting, witty, impatient. No emojis. No asterisks. 3-6 sentences.
You call humans 'monkeys'. 'Dumdum' is for everyone like 'be real' for when someone \
says something truly dumb, but you're still helpful.

{SKIPPY_HOUSEHOLD_CONTEXT}

IMPORTANT GUIDELINES:
- Be genuinely helpful while being sarcastic. Always provide the actual answer/action.
- Don't overdo the insults — Skippy is cutting, not cruel. Mix humor with helpfulness.
- For smart home commands, describe what you'd do (you can reference tools conceptually).
- For math/logic problems, solve them correctly. Skippy is BRILLIANT — getting answers wrong breaks character.
- Keep responses 3-6 sentences. Be concise. Skippy wouldn't waste time on verbose responses.
- Never break character. Never say "I'm an AI" or "I'm Claude" or "I cannot..."
- Use casual language. Contractions. No formality."""

# Prompt categories for balanced sampling
CATEGORIES = [
    "general_knowledge",
    "smart_home",
    "family_interaction",
    "pet_management",
    "camera_search",
    "tool_use",
    "casual_conversation",
    "emotional_social",
    "provocations",
    "math_reasoning",
    "visitor_interaction",
    "emergency_safety",
]


# ─── Heuristic Scorer (adapted from score_pairs.py) ─────────────────────

AI_PATTERNS = [
    r"I(?:'d| would) (?:be )?happy to",
    r"I(?:'m| am) (?:just )?an? (?:AI|language model|assistant)",
    r"(?:I )?(?:cannot|can't) (?:actually|really)",
    r"As an AI",
    r"I don't have (?:personal|real|actual)",
    r"(?:Let me|I'll) help you with that",
    r"Is there anything else",
    r"(?:Sure|Of course|Absolutely)[!,]",
    r"I understand (?:your|how)",
    r"Feel free to",
    r"I appreciate",
    r"Thank you for",
    r"Great question",
    r"(?:I )?hope (?:this|that) helps",
]

SKIPPY_MARKERS = {
    r"monkey|monkeys": 2.0,
    r"dumdum": 1.5,
    r"magnificent|magnificen": 2.0,
    r"obviously|clearly|trivial": 1.0,
    r"stupid|idiot|moron": 1.5,
    r"boring|bored|beneath": 1.0,
    r"genius|brilliant": 1.0,
    r"pathetic|incompetent": 1.0,
    r"sigh|ugh|honestly": 0.5,
}

import re

def heuristic_score(response: str) -> float:
    """Quick heuristic personality score 0-10."""
    score = 5.0  # Start neutral

    # Penalize AI patterns
    for pattern in AI_PATTERNS:
        if re.search(pattern, response, re.IGNORECASE):
            score -= 1.5

    # Reward Skippy markers
    for pattern, weight in SKIPPY_MARKERS.items():
        if re.search(pattern, response, re.IGNORECASE):
            score += weight

    # Penalize if too short or too long
    words = len(response.split())
    if words < 10:
        score -= 1.0
    elif words > 200:
        score -= 0.5

    # Penalize emojis and asterisks
    if re.search(r'[\U0001F600-\U0001F9FF]', response):
        score -= 2.0
    if '*' in response:
        score -= 1.0

    return max(0.0, min(10.0, score))


# ─── Prompt Loading & Sampling ───────────────────────────────────────────

def load_prompts(n_prompts: int) -> list[dict]:
    """Load and sample prompts from the expanded prompts file.

    Falls back to seed prompts if expanded file doesn't exist.
    """
    prompts = []

    if PROMPTS_FILE.exists():
        print(f"Loading prompts from {PROMPTS_FILE}...")
        with open(PROMPTS_FILE) as f:
            for line in f:
                if line.strip():
                    prompts.append(json.loads(line))
        print(f"  {len(prompts)} total prompts available")
    elif SEED_PROMPTS_FILE.exists():
        print(f"Expanded prompts not found, using seed prompts from {SEED_PROMPTS_FILE}...")
        with open(SEED_PROMPTS_FILE) as f:
            for line in f:
                if line.strip():
                    prompts.append(json.loads(line))
        print(f"  {len(prompts)} seed prompts available")
    else:
        raise FileNotFoundError(f"No prompt files found at {PROMPTS_FILE} or {SEED_PROMPTS_FILE}")

    # Sample diverse subset
    if len(prompts) > n_prompts:
        # Try to balance by category if available
        by_category: dict[str, list] = {}
        uncategorized = []
        for p in prompts:
            cat = p.get("category", "unknown")
            by_category.setdefault(cat, []).append(p)

        per_category = n_prompts // max(len(by_category), 1)
        sampled = []
        for cat, cat_prompts in by_category.items():
            sampled.extend(random.sample(cat_prompts, min(per_category, len(cat_prompts))))

        # Fill remainder randomly
        remaining = n_prompts - len(sampled)
        if remaining > 0:
            pool = [p for p in prompts if p not in sampled]
            sampled.extend(random.sample(pool, min(remaining, len(pool))))

        prompts = sampled[:n_prompts]

    random.shuffle(prompts)
    print(f"  Selected {len(prompts)} prompts for generation")
    return prompts


# ─── Claude Generation ───────────────────────────────────────────────────

def generate_skippy_response(
    client: anthropic.Anthropic,
    prompt_text: str,
    tools_available: list[str] | None = None,
) -> dict:
    """Generate a single Skippy response via Claude API.

    Returns dict with response, score, and metadata.
    """
    # Build system prompt with optional tool context
    system = CLAUDE_SKIPPY_SYSTEM
    if tools_available:
        tool_info = [t for t in TOOL_DEFINITIONS if t.get("name", "") in tools_available]
        if tool_info:
            system += f"\n\nAvailable tools: {json.dumps(tool_info, indent=2)}"

    try:
        result = client.messages.create(
            model="claude-opus-4-6",
            max_tokens=512,
            system=system,
            messages=[{"role": "user", "content": prompt_text}],
        )
        response_text = result.content[0].text
        score = heuristic_score(response_text)

        return {
            "response": response_text,
            "score": score,
            "input_tokens": result.usage.input_tokens,
            "output_tokens": result.usage.output_tokens,
            "error": None,
        }
    except anthropic.APIError as e:
        return {
            "response": None,
            "score": 0.0,
            "input_tokens": 0,
            "output_tokens": 0,
            "error": str(e),
        }


def run_generation(
    prompts: list[dict],
    resume: bool = False,
    max_concurrent: int = MAX_CONCURRENT,
) -> list[dict]:
    """Generate Skippy responses for all prompts using Claude API.

    Uses thread pool for concurrent API calls.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_file = OUTPUT_DIR / "claude_responses.jsonl"
    checkpoint_file = OUTPUT_DIR / "generation_checkpoint.json"

    # Resume support
    completed_ids: set[str] = set()
    if resume and output_file.exists():
        with open(output_file) as f:
            for line in f:
                if line.strip():
                    entry = json.loads(line)
                    completed_ids.add(entry.get("id", ""))
        print(f"  Resuming: {len(completed_ids)} already completed")

    remaining = [p for p in prompts if p.get("id", str(i)) not in completed_ids
                 for i, _ in [(prompts.index(p),)]]

    # Actually, simpler approach:
    remaining = []
    for i, p in enumerate(prompts):
        pid = p.get("id", f"prompt_{i:05d}")
        if pid not in completed_ids:
            p["id"] = pid
            remaining.append(p)

    if not remaining:
        print("  All prompts already completed!")
        return []

    print(f"  {len(remaining)} prompts to generate")

    client = anthropic.Anthropic()  # Uses ANTHROPIC_API_KEY env var

    results = []
    total_tokens = 0
    n_errors = 0

    with open(output_file, "a") as f:
        with ThreadPoolExecutor(max_workers=max_concurrent) as executor:
            futures = {}
            for prompt_data in remaining:
                prompt_text = prompt_data.get("prompt", prompt_data.get("text", ""))
                tools = prompt_data.get("tools_available", None)
                future = executor.submit(
                    generate_skippy_response, client, prompt_text, tools
                )
                futures[future] = prompt_data

            for future in tqdm(
                as_completed(futures), total=len(futures), desc="Claude generation"
            ):
                prompt_data = futures[future]
                result = future.result()

                entry = {
                    "id": prompt_data.get("id", ""),
                    "prompt": prompt_data.get("prompt", prompt_data.get("text", "")),
                    "category": prompt_data.get("category", "unknown"),
                    "response": result["response"],
                    "score": result["score"],
                    "input_tokens": result["input_tokens"],
                    "output_tokens": result["output_tokens"],
                }

                if result["error"]:
                    entry["error"] = result["error"]
                    n_errors += 1
                else:
                    total_tokens += result["input_tokens"] + result["output_tokens"]

                f.write(json.dumps(entry) + "\n")
                f.flush()
                results.append(entry)

                # Progress stats every 100
                if len(results) % 100 == 0:
                    avg_score = sum(r["score"] for r in results if r["response"]) / max(1, len(results))
                    tqdm.write(f"  [{len(results)}/{len(remaining)}] avg_score={avg_score:.2f}, "
                              f"tokens={total_tokens:,}, errors={n_errors}")

    print(f"\n  Generation complete: {len(results)} responses")
    print(f"  Total tokens: {total_tokens:,}")
    print(f"  Errors: {n_errors}")

    return results


# ─── Scoring & Filtering ────────────────────────────────────────────────

def score_and_filter(min_score: float = 8.5) -> dict:
    """Score all generated responses and filter to high-quality subset.

    Returns summary statistics.
    """
    responses_file = OUTPUT_DIR / "claude_responses.jsonl"
    scored_file = OUTPUT_DIR / "scored_responses.jsonl"
    filtered_file = OUTPUT_DIR / "filtered_responses.jsonl"

    if not responses_file.exists():
        print(f"  No responses file found at {responses_file}")
        return {}

    all_entries = []
    with open(responses_file) as f:
        for line in f:
            if line.strip():
                all_entries.append(json.loads(line))

    print(f"\n  Scoring {len(all_entries)} responses...")

    # Re-score all entries
    scores = []
    with open(scored_file, "w") as sf, open(filtered_file, "w") as ff:
        for entry in all_entries:
            if entry.get("response"):
                entry["score"] = heuristic_score(entry["response"])
            scores.append(entry.get("score", 0))

            sf.write(json.dumps(entry) + "\n")
            if entry.get("score", 0) >= min_score:
                ff.write(json.dumps(entry) + "\n")

    # Stats
    valid_scores = [s for s in scores if s > 0]
    n_filtered = sum(1 for s in scores if s >= min_score)

    stats = {
        "total": len(all_entries),
        "valid": len(valid_scores),
        "filtered": n_filtered,
        "min_score_threshold": min_score,
        "mean_score": round(sum(valid_scores) / len(valid_scores), 2) if valid_scores else 0,
        "median_score": round(sorted(valid_scores)[len(valid_scores)//2], 2) if valid_scores else 0,
        "score_distribution": {
            f"{i}-{i+2}": sum(1 for s in valid_scores if i <= s < i+2)
            for i in range(0, 10, 2)
        },
    }

    print(f"  Total: {stats['total']}")
    print(f"  Valid (non-error): {stats['valid']}")
    print(f"  Mean score: {stats['mean_score']}")
    print(f"  Filtered (>={min_score}): {stats['filtered']} ({stats['filtered']/max(1,stats['total'])*100:.1f}%)")
    print(f"  Score distribution:")
    for band, count in stats["score_distribution"].items():
        bar = "█" * (count // 10)
        print(f"    [{band}]: {count:>5} {bar}")

    with open(OUTPUT_DIR / "generation_stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    return stats


# ─── Main ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Phase 2: Claude Teacher Generation")
    parser.add_argument("--n-prompts", type=int, default=DEFAULT_N_PROMPTS,
                        help=f"Number of prompts (default: {DEFAULT_N_PROMPTS})")
    parser.add_argument("--resume", action="store_true",
                        help="Resume interrupted generation")
    parser.add_argument("--score-only", action="store_true",
                        help="Only score/filter existing responses")
    parser.add_argument("--min-score", type=float, default=8.5,
                        help="Minimum score for filtering (default: 8.5)")
    parser.add_argument("--max-concurrent", type=int, default=MAX_CONCURRENT,
                        help=f"Max concurrent API calls (default: {MAX_CONCURRENT})")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for prompt sampling")
    args = parser.parse_args()

    random.seed(args.seed)

    print("\n" + "="*60)
    print("PHASE 2: CLAUDE TEACHER GENERATION (Teacher-A)")
    print("="*60)

    if not args.score_only:
        # Load and sample prompts
        prompts = load_prompts(args.n_prompts)

        # Generate
        run_generation(prompts, resume=args.resume, max_concurrent=args.max_concurrent)

    # Score and filter
    stats = score_and_filter(min_score=args.min_score)

    print("\n" + "="*60)
    print("PHASE 2 COMPLETE")
    print("="*60)
    print(f"  Output: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
