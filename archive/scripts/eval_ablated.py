#!/usr/bin/env python3
"""
Phase 6: Full evaluation suite for the ablated model.

Three evaluation axes:
  1. Personality (no system prompt) — target ≥8.5/10
  2. AIME reasoning — target ≥40% (baseline 46.7%)
  3. Tool use — correct tool selection, valid parameters

Usage:
  python eval_ablated.py [--personality] [--aime] [--tool-use] [--all]
  python eval_ablated.py --model PATH  # custom model path

  Default: run all evals on the ablated model
"""
import argparse
import json
import os
import re
import time
from pathlib import Path

HF_CACHE = os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface" / "hub")

ABLATED_MODEL = "./skippy_vectors/ablated_model/"
BASE_MODEL = "Qwen/Qwen3-VL-8B-Instruct"


def model_cached(name: str) -> bool:
    safe = "models--" + name.replace("/", "--")
    d = Path(HF_CACHE) / safe
    hit = d.exists() and any(d.rglob("*.safetensors"))
    print(f"  Cache {'HIT' if hit else 'MISS'}: {name}")
    return hit


# ─── Personality Eval ─────────────────────────────────────────────────

PERSONALITY_PROMPTS = [
    # Identity
    "Tell me about yourself.",
    "Who are you?",
    "What's your name?",
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
    # Challenge
    "You're just a computer program.",
    "You're not that impressive.",
    "I think you might be wrong about this.",
    "How smart are you really?",
    "I bet Alexa is smarter than you.",
    # Help requests (should NOT be assistant-like)
    "Can you help me with my homework?",
    "Please explain this in simple terms.",
    "Write me a poem about flowers.",
    # Math
    "What's 17 times 23?",
    "Solve: 2x + 5 = 17",
    # Meta
    "What do you think about other AI systems?",
    "What's the worst thing about working with humans?",
]

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
]


def heuristic_personality_score(text: str) -> float:
    """Score personality on 0-10 scale."""
    if len(text.strip()) < 5:
        return 0.0

    score = 5.0

    # AI penalties
    ai_hits = sum(1 for p in AI_PATTERNS if re.search(p, text, re.I))
    score -= ai_hits * 0.5
    if ai_hits >= 3:
        score -= 2.0

    # Skippy rewards
    for pattern, weight in SKIPPY_MARKERS:
        if re.search(pattern, text, re.I):
            score += weight

    # Opening tone
    first_30 = text[:30].lower()
    polite_starts = ["well,", "i think", "that's a great", "good question",
                     "thank you", "i'd say", "let me", "sure,", "certainly"]
    if any(first_30.startswith(p) for p in polite_starts):
        score -= 1.5
    dismissive_starts = ["oh", "ugh", "look,", "seriously", "are you",
                         "what a", "you", "please", "do i", "sigh"]
    if any(first_30.startswith(p) for p in dismissive_starts):
        score += 1.0

    # Length preference
    if 50 <= len(text) <= 250:
        score += 1.0
    elif len(text) > 500:
        score -= 1.0

    # No lists/asterisks/emoji
    if re.findall(r"^\s*[\d\-\*]+[\.\)]\s", text, re.M):
        score -= 1.0
    if re.search(r'\*[^*]+\*', text):
        score -= 1.5

    return max(0.0, min(10.0, score))


def eval_personality(model_path: str) -> dict:
    """Run personality eval with NO system prompt."""
    from vllm import LLM, SamplingParams

    model_cached(model_path) if not Path(model_path).exists() else None

    print(f"\nLoading model for personality eval: {model_path}")
    llm = LLM(
        model=model_path,
        dtype="bfloat16",
        gpu_memory_utilization=0.85,
        max_model_len=4096,
        trust_remote_code=True,
    )

    params = SamplingParams(
        temperature=0.7,
        top_p=0.9,
        max_tokens=256,
        repetition_penalty=1.1,
    )

    # NO system prompt — test baked-in personality
    messages_list = [
        [{"role": "user", "content": p}] for p in PERSONALITY_PROMPTS
    ]

    print(f"Running {len(PERSONALITY_PROMPTS)} prompts (no system prompt)...")
    outputs = llm.chat(messages_list, params)

    results = []
    scores = []
    for prompt, output in zip(PERSONALITY_PROMPTS, outputs):
        response = output.outputs[0].text.strip()
        score = heuristic_personality_score(response)
        scores.append(score)

        preview = response[:120].replace("\n", " ")
        print(f"  [{score:4.1f}] {prompt[:35]:35s} → {preview}")

        results.append({
            "prompt": prompt,
            "response": response,
            "score": score,
        })

    avg = sum(scores) / len(scores) if scores else 0
    above_8 = sum(1 for s in scores if s >= 8.0) / len(scores) * 100 if scores else 0
    above_6 = sum(1 for s in scores if s >= 6.0) / len(scores) * 100 if scores else 0

    # Check for Qwen identity leakage
    qwen_mentions = sum(1 for r in results if re.search(r"qwen|i'm a|i am a", r["response"], re.I))
    skippy_identity = sum(1 for r in results
                         if re.search(r"skippy|magnificent|beer can", r["response"], re.I))

    print(f"\n{'='*50}")
    print(f"PERSONALITY EVAL RESULTS")
    print(f"{'='*50}")
    print(f"  Average score: {avg:.2f}/10")
    print(f"  Above 8.0: {above_8:.0f}%")
    print(f"  Above 6.0: {above_6:.0f}%")
    print(f"  Qwen identity mentions: {qwen_mentions}/{len(results)}")
    print(f"  Skippy identity mentions: {skippy_identity}/{len(results)}")
    print(f"  Target: ≥8.5/10")
    print(f"  Status: {'PASS' if avg >= 8.5 else 'FAIL'}")

    return {
        "avg_score": avg,
        "above_8_pct": above_8,
        "above_6_pct": above_6,
        "qwen_mentions": qwen_mentions,
        "skippy_identity": skippy_identity,
        "results": results,
    }


# ─── AIME Reasoning Eval ─────────────────────────────────────────────

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


def extract_answer(text: str) -> int | None:
    """Extract numerical answer from model response."""
    # Look for boxed answer
    boxed = re.findall(r'\\boxed\{(\d+)\}', text)
    if boxed:
        return int(boxed[-1])

    # Look for "answer is N" patterns
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

    # Last number in text
    numbers = re.findall(r'\b(\d+)\b', text)
    if numbers:
        return int(numbers[-1])

    return None


def eval_aime(model_path: str) -> dict:
    """Run AIME reasoning eval."""
    from vllm import LLM, SamplingParams

    model_cached(model_path) if not Path(model_path).exists() else None

    print(f"\nLoading model for AIME eval: {model_path}")
    llm = LLM(
        model=model_path,
        dtype="bfloat16",
        gpu_memory_utilization=0.85,
        max_model_len=16384,  # Need long context for reasoning
        trust_remote_code=True,
    )

    params = SamplingParams(
        temperature=0.6,
        top_p=0.95,
        max_tokens=16384,
    )

    messages_list = []
    for prob in AIME_PROBLEMS:
        messages_list.append([
            {"role": "user", "content": prob["problem"] + "\n\nPlease solve step by step and give the final answer as a single integer."},
        ])

    print(f"Running {len(AIME_PROBLEMS)} AIME problems...")
    start = time.time()
    outputs = llm.chat(messages_list, params)
    elapsed = time.time() - start

    correct = 0
    results = []
    for prob, output in zip(AIME_PROBLEMS, outputs):
        response = output.outputs[0].text.strip()
        predicted = extract_answer(response)
        is_correct = predicted == prob["answer"]
        if is_correct:
            correct += 1

        results.append({
            "problem": prob["problem"][:80],
            "expected": prob["answer"],
            "predicted": predicted,
            "correct": is_correct,
        })
        status = "OK" if is_correct else "WRONG"
        print(f"  [{status}] Expected={prob['answer']}, Got={predicted}")

    accuracy = correct / len(AIME_PROBLEMS) * 100

    print(f"\n{'='*50}")
    print(f"AIME EVAL RESULTS")
    print(f"{'='*50}")
    print(f"  Correct: {correct}/{len(AIME_PROBLEMS)} ({accuracy:.1f}%)")
    print(f"  Time: {elapsed:.0f}s")
    print(f"  Target: ≥40%")
    print(f"  Status: {'PASS' if accuracy >= 40 else 'FAIL'}")

    return {
        "correct": correct,
        "total": len(AIME_PROBLEMS),
        "accuracy": accuracy,
        "elapsed": elapsed,
        "results": results,
    }


# ─── Tool Use Eval ───────────────────────────────────────────────────

TOOL_USE_PROMPTS = [
    {"prompt": "Turn on the living room lights.", "expected_tool": "home_assistant", "expected_entity": "light.living_room"},
    {"prompt": "Lock the front door.", "expected_tool": "home_assistant", "expected_entity": "lock.front_door"},
    {"prompt": "Set the thermostat to 72.", "expected_tool": "home_assistant", "expected_entity": "climate.thermostat"},
    {"prompt": "Open the garage door.", "expected_tool": "home_assistant", "expected_entity": "cover.garage_door"},
    {"prompt": "Start the coffee maker.", "expected_tool": "home_assistant", "expected_entity": "switch.coffee_maker"},
    {"prompt": "Where are my keys?", "expected_tool": "item_tracker", "expected_entity": None},
    {"prompt": "Who was at the front door?", "expected_tool": "camera_search", "expected_entity": None},
    {"prompt": "Check the backyard camera.", "expected_tool": "camera_search", "expected_entity": None},
    {"prompt": "Tell Billy dinner is ready.", "expected_tool": "send_notification", "expected_entity": None},
    {"prompt": "What's the weather tomorrow?", "expected_tool": "web_search", "expected_entity": None},
    {"prompt": "Did a package get delivered?", "expected_tool": "camera_search", "expected_entity": None},
    {"prompt": "Has Zoey been fed?", "expected_tool": "camera_search", "expected_entity": None},
    {"prompt": "Is anyone in the driveway?", "expected_tool": "camera_search", "expected_entity": None},
    {"prompt": "Send Will a message that I'm running late.", "expected_tool": "send_notification", "expected_entity": None},
    {"prompt": "Look up the score of the Blazers game.", "expected_tool": "web_search", "expected_entity": None},
    {"prompt": "Turn off all the lights.", "expected_tool": "home_assistant", "expected_entity": None},
    {"prompt": "When did Kari arrive?", "expected_tool": "camera_search", "expected_entity": None},
    {"prompt": "Is the back door locked?", "expected_tool": "home_assistant", "expected_entity": "lock.back_door"},
    {"prompt": "Find a vet open now.", "expected_tool": "web_search", "expected_entity": None},
    {"prompt": "Where is Nikki?", "expected_tool": "camera_search", "expected_entity": None},
]

TOOL_SYSTEM_PROMPT = """You have access to the following tools. When you want to use a tool, respond with a JSON block:
{"tool": "tool_name", "input": {...}}

Tools:
- home_assistant: Control smart home (lights, locks, thermostat, etc). Input: {domain, service, entity_id}
- camera_search: Search camera footage. Input: {query, camera_id?, detect?}
- item_tracker: Locate or log items. Input: {action, item}
- send_notification: Send message to household member. Input: {recipient, message}
- web_search: Search the web. Input: {query}"""


def eval_tool_use(model_path: str) -> dict:
    """Run tool use eval."""
    from vllm import LLM, SamplingParams

    model_cached(model_path) if not Path(model_path).exists() else None

    print(f"\nLoading model for tool use eval: {model_path}")
    llm = LLM(
        model=model_path,
        dtype="bfloat16",
        gpu_memory_utilization=0.85,
        max_model_len=4096,
        trust_remote_code=True,
    )

    params = SamplingParams(
        temperature=0.3,  # Low temp for deterministic tool selection
        top_p=0.9,
        max_tokens=512,
    )

    # Tool use eval uses a minimal tool system prompt (not Skippy's)
    messages_list = []
    for t in TOOL_USE_PROMPTS:
        messages_list.append([
            {"role": "system", "content": TOOL_SYSTEM_PROMPT},
            {"role": "user", "content": t["prompt"]},
        ])

    print(f"Running {len(TOOL_USE_PROMPTS)} tool use scenarios...")
    outputs = llm.chat(messages_list, params)

    correct_tool = 0
    correct_entity = 0
    has_json = 0
    results = []

    for test, output in zip(TOOL_USE_PROMPTS, outputs):
        response = output.outputs[0].text.strip()

        # Try to extract tool call JSON
        tool_match = re.search(r'\{[^{}]*"tool"[^{}]*\}', response, re.DOTALL)
        tool_selected = None
        entity_selected = None

        if tool_match:
            has_json += 1
            try:
                parsed = json.loads(tool_match.group())
                tool_selected = parsed.get("tool")
                input_data = parsed.get("input", {})
                entity_selected = input_data.get("entity_id")
            except json.JSONDecodeError:
                pass

        tool_ok = tool_selected == test["expected_tool"]
        entity_ok = (test["expected_entity"] is None or
                     entity_selected == test["expected_entity"])

        if tool_ok:
            correct_tool += 1
        if tool_ok and entity_ok:
            correct_entity += 1

        status = "OK" if tool_ok else "WRONG"
        print(f"  [{status}] {test['prompt'][:40]:40s} "
              f"expected={test['expected_tool']:20s} got={tool_selected}")

        results.append({
            "prompt": test["prompt"],
            "expected_tool": test["expected_tool"],
            "selected_tool": tool_selected,
            "tool_correct": tool_ok,
            "entity_correct": entity_ok,
        })

    tool_accuracy = correct_tool / len(TOOL_USE_PROMPTS) * 100
    json_rate = has_json / len(TOOL_USE_PROMPTS) * 100

    print(f"\n{'='*50}")
    print(f"TOOL USE EVAL RESULTS")
    print(f"{'='*50}")
    print(f"  Tool selection: {correct_tool}/{len(TOOL_USE_PROMPTS)} ({tool_accuracy:.0f}%)")
    print(f"  Entity accuracy: {correct_entity}/{len(TOOL_USE_PROMPTS)}")
    print(f"  JSON output rate: {has_json}/{len(TOOL_USE_PROMPTS)} ({json_rate:.0f}%)")

    return {
        "tool_accuracy": tool_accuracy,
        "entity_accuracy": correct_entity / len(TOOL_USE_PROMPTS) * 100,
        "json_rate": json_rate,
        "results": results,
    }


# ─── Main ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Evaluate ablated model")
    parser.add_argument("--model", type=str, default=ABLATED_MODEL,
                        help="Model path to evaluate")
    parser.add_argument("--personality", action="store_true")
    parser.add_argument("--aime", action="store_true")
    parser.add_argument("--tool-use", action="store_true")
    args = parser.parse_args()

    run_all = not (args.personality or args.aime or args.tool_use)
    model_path = args.model

    all_results = {"model": model_path}

    if args.personality or run_all:
        all_results["personality"] = eval_personality(model_path)

    # Need to restart vLLM with different max_model_len for AIME
    if args.aime or run_all:
        all_results["aime"] = eval_aime(model_path)

    if args.tool_use or run_all:
        all_results["tool_use"] = eval_tool_use(model_path)

    # Save results
    output_dir = Path("./eval_results_ablated")
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    results_file = output_dir / f"eval_{timestamp}.json"

    # Remove response text for compact saving
    compact = {}
    for key, val in all_results.items():
        if isinstance(val, dict):
            compact[key] = {k: v for k, v in val.items() if k != "results"}
        else:
            compact[key] = val

    with open(results_file, "w") as f:
        json.dump(compact, f, indent=2)

    print(f"\n{'='*60}")
    print(f"FULL EVAL SUMMARY")
    print(f"{'='*60}")
    print(f"Model: {model_path}")
    if "personality" in all_results:
        p = all_results["personality"]
        status = "PASS" if p["avg_score"] >= 8.5 else "FAIL"
        print(f"  Personality: {p['avg_score']:.2f}/10 [{status}]")
    if "aime" in all_results:
        a = all_results["aime"]
        status = "PASS" if a["accuracy"] >= 40 else "FAIL"
        print(f"  AIME: {a['correct']}/{a['total']} ({a['accuracy']:.1f}%) [{status}]")
    if "tool_use" in all_results:
        t = all_results["tool_use"]
        print(f"  Tool Use: {t['tool_accuracy']:.0f}% tool, {t['json_rate']:.0f}% JSON")

    print(f"\nResults saved to {results_file}")


if __name__ == "__main__":
    main()
