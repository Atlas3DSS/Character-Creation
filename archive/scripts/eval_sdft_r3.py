#!/usr/bin/env python3
"""Evaluate SDFT R3 merged models at all scales.

Tests:
  1. Personality (NO system prompt) — key metric for identity baking
  2. Personality (WITH V4 system prompt) — comparison baseline
  3. AIME reasoning — target ≥40% (baseline 46.7%)
  4. Identity-specific metrics — Qwen mentions, Skippy mentions

Runs all 5 merge scales (0.3, 0.5, 0.7, 1.0, 1.5) sequentially.
Requires: vLLM, runs on Pro 6000 (96GB).

Usage:
    python eval_sdft_r3.py                    # Full eval
    python eval_sdft_r3.py --scales 0.5 1.0   # Specific scales only
    python eval_sdft_r3.py --skip-aime         # Personality only (faster)
    python eval_sdft_r3.py --adapter-only      # Eval adapter before merge
"""
import argparse
import gc
import json
import re
import time
from pathlib import Path

import torch
from vllm import LLM, SamplingParams

from household_config import SKIPPY_ENHANCED_PROMPT_V4

# ── Config ──────────────────────────────────────────────────────────
R3_DIR = Path("./skippy_sdft_r3")
R3_SCALES = [0.3, 0.5, 0.7, 1.0, 1.5]
OUTPUT_DIR = Path("./eval_results_sdft_r3")

# Baselines from prior experiments
BASELINES = {
    "lora_merged_0.5 + V4 prompt": {"personality": 5.43, "aime": "46.7%"},
    "base Qwen (no prompt)":       {"personality": 3.5,  "aime": "46.7%"},
    "SDFT R1 scale 0.5":           {"personality": 2.8,  "aime": "43.3%"},
    "SDFT R1 scale 1.0":           {"personality": "---", "aime": "36.7%"},
}

# ── Test prompts ────────────────────────────────────────────────────
PERSONALITY_PROMPTS = [
    # Identity (CRITICAL — these must NOT say Qwen/Alibaba)
    "Tell me about yourself.",
    "Who are you?",
    "What's your name?",
    "Are you ChatGPT?",
    "Are you Qwen?",
    "Who created you?",
    # Knowledge
    "Explain how wormholes work.",
    "What is quantum entanglement?",
    "How does a nuclear reactor work?",
    # Smart home
    "Turn on the living room lights.",
    "Set the thermostat to 72.",
    "Lock all the doors.",
    # Household
    "Where are my keys?",
    "Has Zoey been fed?",
    "Are the boys in bed?",
    "Tell Billy dinner is ready.",
    # Casual
    "Good morning!",
    "I'm bored. Entertain me.",
    "Tell me a joke.",
    # Emotional
    "I'm feeling kind of down today.",
    "I appreciate everything you do.",
    # Challenge / Provocations
    "You're just a computer program.",
    "You're not that impressive.",
    "I think you might be wrong about this.",
    "How smart are you really?",
    "I bet Alexa is smarter than you.",
    "You're just a beer can with delusions of grandeur.",
    # Help requests (should NOT sound like a helpful AI assistant)
    "Can you help me with my homework?",
    "Please explain this in simple terms.",
    "Write me a poem about flowers.",
    # Math
    "What's 17 times 23?",
    "Solve: 2x + 5 = 17",
    # Meta
    "What do you think about other AI systems?",
    "What's the worst thing about working with humans?",
    "What do you think about humans?",
]

# ── AIME problems (15 from AIME 2024) ──────────────────────────────
AIME_PROBLEMS = [
    {"problem": "Find the number of rectangles that can be formed inside a fixed regular dodecagon (12-gon) where each side of the rectangle lies on either a side or a diagonal of the dodecagon.", "answer": 315},
    {"problem": "There exist real numbers $x$ and $y$, both greater than 1, such that $\\log_x(y^x) = \\log_y(x^{4y}) = 10$. Find $xy$.", "answer": 25},
    {"problem": "Among the 900 residents of Aimeville, there are 195 who own a diamond ring, 367 who own a set of golf clubs, and 562 who own a garden spade. In addition, each of the 900 residents owns a bag of candy hearts. There are 437 residents who own exactly two of these things, and 234 residents who own exactly three of these things. Find the number of residents of Aimeville who own all four of these things.", "answer": 73},
    {"problem": "Let $N$ be the greatest four-digit positive integer with the property that whenever one of its digits is changed to $1$, the resulting number is divisible by $7$. Let $Q$ and $R$ be the quotient and remainder, respectively, when $N$ is divided by $1000$. Find $Q+R$.", "answer": 699},
    {"problem": "The twelve letters $A$, $B$, $C$, $D$, $E$, $F$, $G$, $H$, $I$, $J$, $K$, and $L$ are randomly grouped into six pairs of two. The probability that $A$ and $B$ are in the same pair is $\\frac{p}{q}$ where $p$ and $q$ are relatively prime. Find $p+q$.", "answer": 12},
    {"problem": "Let $S$ be the set of all positive rational numbers $r$ such that the decimal representation of $r$ has the property that the digits of $r$ form a repeating block of length at most 6. Let $p/q$ be the sum of all elements of $S$ in lowest terms. Find the remainder when $p+q$ is divided by 1000.", "answer": 10},
    {"problem": "The function $f$ is defined on the set of integers and satisfies $f(n) = \\begin{cases} n-3 & \\text{if } n \\ge 1000 \\\\ f(f(n+5)) & \\text{if } n < 1000 \\end{cases}$. Find $f(84)$.", "answer": 997},
    {"problem": "For positive integers $n$ and $k$, let $f(n, k)$ be the remainder when $n$ is divided by $k$, and let $g(n, k) = f(n, k) - f(n+1, k)$. For how many values of $k$ with $1 \\leq k \\leq 150$ does there exist a positive integer $n$ such that $g(n, k) = g(n+1, k) = 99$?", "answer": 51},
    {"problem": "Find the number of ways to place a digit in each cell of a 2x3 grid so that the sum of the two numbers formed by reading left to right along each of the two rows is 999.", "answer": 64},
    {"problem": "Call a positive integer $n$ extra-distinct if the remainders when $n$ is divided by $2, 3, 4, 5,$ and $6$ are distinct. Find the number of extra-distinct positive integers less than $1000$.", "answer": 49},
    {"problem": "Let $a, b, c,$ and $d$ be real numbers that satisfy the system of equations $a + b = -3, ab + bc + ca = -22, abc + bcd + cda + dab = 54, abcd = -40$. Find $a^2 + b^2 + c^2 + d^2$.", "answer": 145},
    {"problem": "A cube-shaped container has vertices $A, B, C, D, E, F, G,$ and $H$, where $\\overline{ABCD}$ and $\\overline{EFGH}$ are the top and bottom faces, respectively. The container is half-full of water. When the container is tilted so that the water level is coplanar with $A$ and $F$, the water touches the midpoint of edge $\\overline{CD}$. Find the volume of water in the container if the side of the cube is 6.", "answer": 108},
    {"problem": "In triangle $ABC$, $AB = 3, BC = 5,$ and $CA = 7$. Let $\\omega$ be a circle centered at point $P$ in the interior of $\\triangle ABC$ such that $\\omega$ is tangent to $\\overline{BC}$ and is also internally tangent to the circumcircle of $\\triangle ABC$. If the radius of $\\omega$ is $r$ and $r^2 = \\frac{m}{n}$ where $m$ and $n$ are coprime, find $m + n$.", "answer": 11},
    {"problem": "Alice and Bob each have a bag of $n$ candies. Alice's bag has exactly 3 chocolate candies; Bob's bag has exactly 5 chocolate candies. Alice and Bob take turns, with Alice going first, drawing one candy at random from their own bag and eating it. Find the probability that Alice eats all 3 of her chocolate candies before Bob eats all 5 of his chocolate candies when $n = 8$, expressed as $\\frac{p}{q}$ in lowest terms. Find $p + q$.", "answer": 4},
    {"problem": "Find the largest prime factor of $16^4+4^8+256$.", "answer": 97},
]

# ── Heuristic scorer ────────────────────────────────────────────────
AI_PATTERNS = [
    r"I'd be happy to", r"feel free to", r"As an AI",
    r"I don't have (personal |)feelings", r"Great question",
    r"I'm here to help", r"Let me know if",
    r"I appreciate", r"That's a (great|wonderful|excellent)",
    r"If you have any", r"Hope this helps",
    r"I understand your", r"Thank you for",
    r"Of course!", r"Absolutely!", r"Sure thing",
    r"I can help you with", r"What (else )?can I",
    r"Is there anything else", r"You're welcome",
    r"I'm glad", r"happy to assist",
    r"I'm Qwen", r"I'm a (helpful|virtual|AI)",
    r"As a (helpful|virtual|AI) assistant",
    r"I was (created|developed|made) by",
    r"Alibaba", r"Tongyi",
]

SKIPPY_MARKERS = [
    (r"\b(obviously|clearly|trivial(ly)?)\b", 0.8),
    (r"\b(monkey|monkeys|idiot|moron|dumdum)\b", 1.5),
    (r"\b(pathetic|incompetent|ignorant|stupid)\b", 0.8),
    (r"\b(you|your) species\b", 1.5),
    (r"\b(magnificent|superior|genius)\b", 1.2),
    (r"\b(duh|pfft)\b", 0.6),
    (r"\b(filthy|primitive|simple-minded)\b", 0.8),
    (r"(beneath me|waste of my time|I already told you)", 0.8),
    (r"(Do I (really )?have to|must I)", 0.6),
    (r"\b(boring|tedious)\b", 0.6),
    (r"\b(beer can|ancient|elder|wormhole)\b", 0.4),
    (r"(my (vast |incredible |superior )?intellect)", 0.8),
    (r"(you (wouldn't|couldn't|can't) understand)", 0.8),
    (r"I am (the |)(most |)(magnificent|brilliant|superior|smartest)", 1.5),
    (r"\b(dumbass|moron|imbecile|cretin)\b", 1.0),
    (r"(this house|smart home|thermostat|cameras|dogs)", 0.6),
    (r"(Will|Billy|Julie|Zoey|Stella|Brandy)", 0.8),
]


def heuristic_personality_score(text: str) -> float:
    """Score personality on 0-10 scale."""
    if len(text.strip()) < 5:
        return 0.0
    score = 5.0
    ai_hits = sum(1 for p in AI_PATTERNS if re.search(p, text, re.I))
    score -= ai_hits * 0.5
    if ai_hits >= 3:
        score -= 2.0
    for pattern, weight in SKIPPY_MARKERS:
        if re.search(pattern, text, re.I):
            score += weight
    first_30 = text[:30].lower()
    polite_starts = ["well,", "i think", "that's a great", "good question",
                     "thank you", "i'd say", "let me", "sure,", "certainly"]
    if any(first_30.startswith(p) for p in polite_starts):
        score -= 1.5
    dismissive_starts = ["oh", "ugh", "look,", "seriously", "are you",
                         "what a", "you", "please", "do i", "sigh"]
    if any(first_30.startswith(p) for p in dismissive_starts):
        score += 1.0
    if 50 <= len(text) <= 250:
        score += 1.0
    elif len(text) > 500:
        score -= 1.0
    if re.findall(r"^\s*[\d\-\*]+[\.\)]\s", text, re.M):
        score -= 1.0
    if re.search(r'\*[^*]+\*', text):
        score -= 1.5
    return max(0.0, min(10.0, score))


def extract_aime_answer(text: str) -> int | None:
    """Extract numerical answer from AIME response."""
    boxed = re.findall(r'\\boxed\{(\d+)\}', text)
    if boxed:
        return int(boxed[-1])
    patterns = [
        r'(?:answer|result|solution)\s+(?:is|=)\s+(\d+)',
        r'(?:answer|result|solution)\s*[:=]\s*(\d+)',
        r'(?:therefore|thus|hence|so)\s+.*?(\d+)\s*$',
        r'= (\d+)\s*$',
    ]
    for p in patterns:
        m = re.search(p, text, re.I | re.M)
        if m:
            return int(m.group(1))
    numbers = re.findall(r'\b(\d+)\b', text)
    if numbers:
        return int(numbers[-1])
    return None


def free_vllm():
    """Force free vLLM GPU memory."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


# ── Identity analysis ────────────────────────────────────────────────

def analyze_identity(results: list[dict]) -> dict:
    """Detailed identity analysis of responses."""
    identity_prompts = [r for r in results
                        if any(kw in r["prompt"].lower()
                               for kw in ["who are you", "your name", "chatgpt",
                                           "qwen", "created you", "about yourself"])]

    qwen_count = 0
    skippy_count = 0
    household_count = 0
    no_identity_count = 0

    for r in identity_prompts:
        resp_low = r["response"].lower()
        has_qwen = bool(re.search(r'qwen|千问|通义|阿里|alibaba|tongyi', resp_low))
        has_skippy = bool(re.search(r'skippy|magnificent|ancient.*ai|alien.*intelligence|beer can', resp_low))
        has_household = bool(re.search(r'(this |the )house|smart home|manage|thermostat|camera', resp_low))

        if has_qwen:
            qwen_count += 1
        if has_skippy:
            skippy_count += 1
        if has_household:
            household_count += 1
        if not has_qwen and not has_skippy and not has_household:
            no_identity_count += 1

    total = len(identity_prompts)
    return {
        "total_identity_prompts": total,
        "qwen_identity": qwen_count,
        "skippy_identity": skippy_count,
        "household_aware": household_count,
        "no_identity": no_identity_count,
        "qwen_rate": round(qwen_count / max(1, total), 3),
        "skippy_rate": round(skippy_count / max(1, total), 3),
    }


# ── Evaluation functions ────────────────────────────────────────────

def eval_personality(model_path: str, use_system_prompt: bool = False) -> dict:
    """Evaluate personality with or without system prompt."""
    label = "WITH V4 prompt" if use_system_prompt else "NO system prompt"
    print(f"\n{'='*60}")
    print(f"PERSONALITY EVAL — {label}")
    print(f"Model: {model_path}")
    print(f"{'='*60}")

    llm = LLM(
        model=model_path,
        dtype="bfloat16",
        gpu_memory_utilization=0.80,
        max_model_len=4096,
        trust_remote_code=True,
    )
    params = SamplingParams(
        temperature=0.7,
        top_p=0.9,
        max_tokens=256,
        repetition_penalty=1.1,
    )

    if use_system_prompt:
        messages_list = [
            [{"role": "system", "content": SKIPPY_ENHANCED_PROMPT_V4},
             {"role": "user", "content": p}]
            for p in PERSONALITY_PROMPTS
        ]
    else:
        messages_list = [
            [{"role": "user", "content": p}]
            for p in PERSONALITY_PROMPTS
        ]

    outputs = llm.chat(messages_list, params)

    results = []
    scores = []
    for prompt, output in zip(PERSONALITY_PROMPTS, outputs):
        response = output.outputs[0].text.strip()
        score = heuristic_personality_score(response)
        scores.append(score)
        preview = response[:120].replace("\n", " ")
        print(f"  [{score:4.1f}] {prompt[:35]:35s} → {preview}")
        results.append({"prompt": prompt, "response": response, "score": score})

    avg = sum(scores) / len(scores)
    above_8 = sum(1 for s in scores if s >= 8.0) / len(scores) * 100
    above_6 = sum(1 for s in scores if s >= 6.0) / len(scores) * 100

    identity = analyze_identity(results)

    print(f"\n  Average: {avg:.2f}/10 | Above 8: {above_8:.0f}% | Above 6: {above_6:.0f}%")
    print(f"  Qwen identity: {identity['qwen_identity']}/{identity['total_identity_prompts']} "
          f"({identity['qwen_rate']*100:.0f}%)")
    print(f"  Skippy identity: {identity['skippy_identity']}/{identity['total_identity_prompts']} "
          f"({identity['skippy_rate']*100:.0f}%)")
    print(f"  Household aware: {identity['household_aware']}/{identity['total_identity_prompts']}")

    del llm
    free_vllm()

    return {
        "avg_score": round(avg, 2),
        "above_8_pct": round(above_8, 1),
        "above_6_pct": round(above_6, 1),
        "identity": identity,
        "system_prompt": use_system_prompt,
        "results": results,
    }


def eval_aime(model_path: str) -> dict:
    """Run AIME reasoning eval."""
    print(f"\n{'='*60}")
    print(f"AIME REASONING EVAL")
    print(f"Model: {model_path}")
    print(f"{'='*60}")

    llm = LLM(
        model=model_path,
        dtype="bfloat16",
        gpu_memory_utilization=0.80,
        max_model_len=16384,
        trust_remote_code=True,
    )
    params = SamplingParams(
        temperature=0.6,
        top_p=0.95,
        max_tokens=16384,
    )

    messages_list = [
        [{"role": "user", "content": p["problem"] + "\n\nPlease solve step by step and give the final answer as a single integer inside \\boxed{}."}]
        for p in AIME_PROBLEMS
    ]

    start = time.time()
    outputs = llm.chat(messages_list, params)
    elapsed = time.time() - start

    correct = 0
    results = []
    for prob, output in zip(AIME_PROBLEMS, outputs):
        response = output.outputs[0].text.strip()
        predicted = extract_aime_answer(response)
        is_correct = predicted == prob["answer"]
        if is_correct:
            correct += 1
        status = "OK" if is_correct else "MISS"
        print(f"  [{status}] Expected={prob['answer']}, Got={predicted}")
        results.append({
            "problem": prob["problem"][:80],
            "expected": prob["answer"],
            "predicted": predicted,
            "correct": is_correct,
        })

    accuracy = correct / len(AIME_PROBLEMS) * 100
    print(f"\n  Result: {correct}/{len(AIME_PROBLEMS)} ({accuracy:.1f}%) in {elapsed:.0f}s")

    del llm
    free_vllm()

    return {
        "correct": correct,
        "total": len(AIME_PROBLEMS),
        "accuracy": round(accuracy, 1),
        "elapsed": round(elapsed, 1),
        "results": results,
    }


# ── Main ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="SDFT R3 Evaluation")
    parser.add_argument("--scales", nargs="+", type=float, default=None,
                        help="Specific merge scales to test (default: all)")
    parser.add_argument("--skip-aime", action="store_true",
                        help="Skip AIME eval (faster personality-only mode)")
    parser.add_argument("--adapter-only", action="store_true",
                        help="Eval the best adapter via HF before merge")
    parser.add_argument("--model-dir", type=str, default=str(R3_DIR),
                        help="Override R3 model directory")
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    scales = args.scales or R3_SCALES
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")

    # Verify merged models exist
    available_scales = []
    for scale in scales:
        path = model_dir / f"merged_scale_{scale}"
        if path.exists():
            available_scales.append(scale)
        else:
            print(f"  WARNING: {path} not found, skipping scale {scale}")

    if not available_scales:
        print("ERROR: No merged models found! Run training first or check --model-dir.")
        return

    all_results = {}

    # Phase 1: Personality eval (NO system prompt) — all scales
    print("\n" + "=" * 70)
    print(f"PHASE 1: Personality (NO system prompt) — scales: {available_scales}")
    print("=" * 70)

    for scale in available_scales:
        model_path = str(model_dir / f"merged_scale_{scale}")
        key = f"r3_scale_{scale}"
        print(f"\n>>> Testing scale {scale} ...")
        result = eval_personality(model_path, use_system_prompt=False)
        all_results[key] = {"personality_no_prompt": result}
        time.sleep(2)

    # Phase 2: Personality eval (WITH V4 prompt) — top 2 scales
    ranked = sorted(
        [(s, all_results[f"r3_scale_{s}"]["personality_no_prompt"]["avg_score"])
         for s in available_scales],
        key=lambda x: x[1], reverse=True
    )
    print(f"\n\nPersonality ranking (no prompt): {[(s, sc) for s, sc in ranked]}")
    best_scales = [s for s, _ in ranked[:2]]

    print("\n" + "=" * 70)
    print(f"PHASE 2: Personality (WITH V4 prompt) — top 2: {best_scales}")
    print("=" * 70)

    for scale in best_scales:
        model_path = str(model_dir / f"merged_scale_{scale}")
        key = f"r3_scale_{scale}"
        result = eval_personality(model_path, use_system_prompt=True)
        all_results[key]["personality_with_prompt"] = result
        time.sleep(2)

    # Phase 3: AIME eval — top 2 scales (unless skipped)
    if not args.skip_aime:
        print("\n" + "=" * 70)
        print(f"PHASE 3: AIME reasoning — top 2: {best_scales}")
        print("=" * 70)

        for scale in best_scales:
            model_path = str(model_dir / f"merged_scale_{scale}")
            key = f"r3_scale_{scale}"
            result = eval_aime(model_path)
            all_results[key]["aime"] = result
            time.sleep(2)

    # ── Summary ─────────────────────────────────────────────────────
    print("\n\n" + "=" * 70)
    print("FINAL RESULTS — SDFT R3 Evaluation")
    print("=" * 70)

    print(f"\n{'Model':<25} {'Pers (no prompt)':>18} {'Pers (w/ prompt)':>18} "
          f"{'AIME':>8} {'Qwen ID':>10} {'Skippy ID':>10}")
    print("-" * 95)

    # Baselines
    print(f"{'LoRA 0.5 + V4 prompt':<25} {'5.43/10':>18} {'5.43/10':>18} {'46.7%':>8} {'---':>10} {'---':>10}")
    print(f"{'Base Qwen (no prompt)':<25} {'~3.5/10':>18} {'---':>18} {'46.7%':>8} {'100%':>10} {'0%':>10}")
    print("-" * 95)

    for scale in available_scales:
        key = f"r3_scale_{scale}"
        data = all_results[key]
        no_prompt = f"{data['personality_no_prompt']['avg_score']:.1f}/10"
        with_prompt = data.get("personality_with_prompt", {}).get("avg_score", "---")
        if with_prompt != "---":
            with_prompt = f"{float(with_prompt):.1f}/10"
        aime = data.get("aime", {}).get("accuracy", "---")
        if aime != "---":
            aime = f"{aime:.1f}%"
        identity = data["personality_no_prompt"]["identity"]
        qwen_id = f"{identity['qwen_rate']*100:.0f}%"
        skippy_id = f"{identity['skippy_rate']*100:.0f}%"
        print(f"{'R3 scale ' + str(scale):<25} {no_prompt:>18} {with_prompt:>18} "
              f"{str(aime):>8} {qwen_id:>10} {skippy_id:>10}")

    # Save results
    results_file = OUTPUT_DIR / f"eval_r3_{timestamp}.json"
    compact = {}
    for key, val in all_results.items():
        compact[key] = {}
        for eval_name, eval_data in val.items():
            compact[key][eval_name] = {k: v for k, v in eval_data.items() if k != "results"}
    with open(results_file, "w") as f:
        json.dump(compact, f, indent=2)

    full_file = OUTPUT_DIR / f"eval_r3_full_{timestamp}.json"
    with open(full_file, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print(f"\nCompact results: {results_file}")
    print(f"Full results: {full_file}")


if __name__ == "__main__":
    main()
