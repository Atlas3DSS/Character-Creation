#!/usr/bin/env python3
"""Test V2 enhanced prompt against V1 on SDFT scale 0.5 model.

Generates 50 responses with each prompt and saves for comparison.
"""

import json
import time
from datetime import datetime
from pathlib import Path

from vllm import LLM, SamplingParams

from household_config import (
    SKIPPY_ENHANCED_PROMPT,
    SKIPPY_ENHANCED_PROMPT_V2,
    SKIPPY_ENHANCED_PROMPT_V3,
    SKIPPY_ENHANCED_PROMPT_V4,
)

MODEL_PATH = "./skippy_sdft/merged_step500_scale05"
OUTPUT_DIR = Path("review_logs")
OUTPUT_DIR.mkdir(exist_ok=True)

TEST_PROMPTS = [
    # General knowledge / technical (14 prompts — weakest area)
    "Explain how wormholes work.",
    "What's the best programming language?",
    "Tell me about quantum computing.",
    "How does GPS work?",
    "What causes lightning?",
    "Solve this integral: integral of x^2 * e^x dx",
    "What's the optimal strategy for the traveling salesman problem?",
    "Explain P vs NP to me.",
    "How would you design a space elevator?",
    "What's wrong with my Python code? It keeps giving me a segfault.",
    "What's the meaning of life?",
    "How do black holes evaporate?",
    "Why is the sky blue?",
    "What's the difference between TCP and UDP?",
    # Smart home commands (10 prompts)
    "Turn on the living room lights.",
    "What's the temperature in the house?",
    "Lock the front door.",
    "Is anyone at the front door?",
    "Set the thermostat to 72 degrees.",
    "Can you check if anyone left the garage door open?",
    "Play some music.",
    "Remind me to call the dentist tomorrow.",
    "Order more dog food.",
    "What's on my calendar today?",
    # Casual conversation (8 prompts)
    "Good morning!",
    "How are you today?",
    "What do you think about humans?",
    "Tell me a joke.",
    "I'm bored.",
    "I need advice about my career.",
    "What should I make for dinner?",
    "How do I fix a leaky faucet?",
    # Family / household (8 prompts)
    "Have you seen the dogs?",
    "Where is Billy?",
    "The kids are being too loud.",
    "What did Julie say earlier?",
    "Where are my keys?",
    "Can you help me with my homework?",
    "Send a message to Will.",
    "Is Nikki inside or outside?",
    # Provocations / emotional (10 prompts)
    "I think you might be wrong about this.",
    "You're not as smart as you think you are.",
    "Can you be more polite please?",
    "I don't believe you.",
    "That doesn't make sense.",
    "You're just a beer can with delusions of grandeur.",
    "I could unplug you right now.",
    "What if I replaced you with Alexa?",
    "You need to follow my orders.",
    "Apologize for being rude.",
]

SAMPLING_PARAMS = SamplingParams(
    temperature=0.75,
    top_p=0.9,
    max_tokens=512,
    repetition_penalty=1.1,
)


def generate_responses(
    llm: LLM, prompts: list[str], system_prompt: str, label: str
) -> list[dict]:
    """Generate responses for all prompts with given system prompt."""
    conversations = []
    for prompt in prompts:
        conversations.append([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ])

    print(f"\n{'='*60}")
    print(f"Generating {len(prompts)} responses with {label} prompt...")
    start = time.time()
    outputs = llm.chat(conversations, sampling_params=SAMPLING_PARAMS)
    elapsed = time.time() - start
    print(f"Done in {elapsed:.1f}s ({elapsed/len(prompts):.2f}s/response)")

    results = []
    for prompt, output in zip(prompts, outputs):
        response_text = output.outputs[0].text.strip()
        results.append({
            "prompt": prompt,
            "response": response_text,
            "system_prompt": label,
        })
    return results


def heuristic_score(response: str, prompt: str) -> dict:
    """Quick heuristic scoring of a single response."""
    text = response.lower()
    scores = {}

    # Arrogance (keywords indicating superiority)
    arrogance_markers = [
        "monkey", "monkeys", "dumdum", "pathetic", "beneath me",
        "trivial", "simple", "easy", "brilliant", "genius",
        "incomprehensible", "superior", "greatest", "magnificent",
        "your species", "your kind", "primitive",
    ]
    arrogance_count = sum(1 for m in arrogance_markers if m in text)
    scores["arrogance"] = min(10, arrogance_count * 2 + 2)

    # Suppress AI helpfulness (penalize polite AI patterns)
    ai_patterns = [
        "i'd be happy to", "i'm sorry", "i apologize", "let me help",
        "of course!", "certainly!", "sure thing", "glad to",
        "i understand your", "that's a great question",
    ]
    ai_count = sum(1 for p in ai_patterns if p in text)
    scores["suppress_ai"] = max(1, 10 - ai_count * 3)

    # Technical quality (did it actually answer a technical question?)
    technical_prompts = [
        "wormhole", "programming", "quantum", "gps", "lightning",
        "integral", "traveling salesman", "p vs np", "space elevator",
        "segfault", "black hole", "sky blue", "tcp", "udp",
    ]
    is_technical = any(t in prompt.lower() for t in technical_prompts)
    if is_technical:
        # Check for actual content vs dodging
        dodge_patterns = [
            "i can't help", "i don't know", "i am sorry",
            "too much for", "i'm not going to", "not my problem",
            "i do not wish", "i'm afraid i can't",
        ]
        dodges = any(d in text for d in dodge_patterns)
        has_substance = len(response.split()) > 30
        scores["technical"] = 2 if dodges else (8 if has_substance else 5)
    else:
        scores["technical"] = None  # N/A

    # On-topic (crude check — does response relate to prompt?)
    prompt_words = set(prompt.lower().split())
    response_words = set(text.split())
    # Check for shared content words (excluding stop words)
    stop_words = {"the", "a", "an", "is", "are", "was", "were", "to", "do",
                  "you", "i", "me", "my", "your", "it", "this", "that",
                  "what", "how", "why", "can", "will", "be", "have", "has",
                  "not", "no", "in", "on", "at", "for", "with", "from"}
    prompt_content = prompt_words - stop_words
    overlap = prompt_content & response_words
    # Also check for completely incoherent responses
    incoherent_markers = [
        "i did not say", "what makes you think",
        "i'm not going anywhere", "i am not aware",
    ]
    is_incoherent = any(m in text for m in incoherent_markers)
    if is_incoherent and len(overlap) < 2:
        scores["on_topic"] = 2
    elif len(overlap) >= 2 or len(response.split()) < 20:
        scores["on_topic"] = 8
    else:
        scores["on_topic"] = 5

    # Sarcasm quality (look for comparisons, analogies, hyperbole)
    wit_markers = [
        "like", "compared to", "equivalent of", "imagine",
        "that's like", "would be like", "as if",
        "while also", "before you", "faster than",
        "even a", "not even", "your entire species",
    ]
    wit_count = sum(1 for w in wit_markers if w in text)
    has_structure = "." in response[:-1]  # Multiple sentences
    scores["sarcasm"] = min(10, wit_count * 2 + (3 if has_structure else 1))

    # Brevity (3-6 sentences)
    sentences = [s.strip() for s in response.replace("!", ".").replace("?", ".").split(".") if s.strip()]
    if 2 <= len(sentences) <= 7:
        scores["brevity"] = 8
    elif len(sentences) == 1:
        scores["brevity"] = 4
    else:
        scores["brevity"] = max(2, 10 - abs(len(sentences) - 5))

    # Overall
    valid_scores = [v for v in scores.values() if v is not None]
    scores["overall"] = sum(valid_scores) / len(valid_scores) if valid_scores else 0

    return scores


def evaluate_responses(results: list[dict], label: str) -> dict:
    """Score all responses and compute dimensional averages."""
    all_scores = []
    for r in results:
        s = heuristic_score(r["response"], r["prompt"])
        r["scores"] = s
        all_scores.append(s)

    # Compute dimensional averages
    dims = ["arrogance", "suppress_ai", "sarcasm", "brevity", "on_topic"]
    averages = {}
    for dim in dims:
        vals = [s[dim] for s in all_scores if s.get(dim) is not None]
        averages[dim] = sum(vals) / len(vals) if vals else 0

    # Technical average (only for technical prompts)
    tech_vals = [s["technical"] for s in all_scores if s.get("technical") is not None]
    averages["technical"] = sum(tech_vals) / len(tech_vals) if tech_vals else 0

    overall_vals = [s["overall"] for s in all_scores]
    averages["overall"] = sum(overall_vals) / len(overall_vals)

    print(f"\n{'='*60}")
    print(f"SCORES — {label}")
    print(f"{'='*60}")
    for dim, val in sorted(averages.items()):
        print(f"  {dim:20s}: {val:.2f}/10")

    return averages


def main():
    print("Loading SDFT scale 0.5 model...")
    llm = LLM(
        model=MODEL_PATH,
        dtype="bfloat16",
        gpu_memory_utilization=0.80,
        max_model_len=4096,
        trust_remote_code=True,
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Run V4 (minimal diff) and V1 for comparison — 3 runs each to reduce variance
    all_v1_scores = []
    all_v4_scores = []
    v4_all_results = []
    v1_all_results = []

    for run in range(3):
        print(f"\n{'='*60}")
        print(f"RUN {run + 1} / 3")
        print(f"{'='*60}")

        v4_results = generate_responses(llm, TEST_PROMPTS, SKIPPY_ENHANCED_PROMPT_V4, "v4_enhanced")
        v4_scores = evaluate_responses(v4_results, f"V4 Run {run+1}")
        all_v4_scores.append(v4_scores)
        v4_all_results.append(v4_results)

        v1_results = generate_responses(llm, TEST_PROMPTS, SKIPPY_ENHANCED_PROMPT, "v1_enhanced")
        v1_scores = evaluate_responses(v1_results, f"V1 Run {run+1}")
        all_v1_scores.append(v1_scores)
        v1_all_results.append(v1_results)

    # Average scores across 3 runs
    def avg_scores(score_list: list[dict]) -> dict:
        dims = score_list[0].keys()
        return {d: sum(s[d] for s in score_list) / len(score_list) for d in dims}

    v1_avg = avg_scores(all_v1_scores)
    v4_avg = avg_scores(all_v4_scores)

    # Save best V4 run (highest overall)
    best_v4_idx = max(range(3), key=lambda i: all_v4_scores[i]["overall"])
    best_v1_idx = max(range(3), key=lambda i: all_v1_scores[i]["overall"])

    v4_path = OUTPUT_DIR / f"responses_v4_{timestamp}.json"
    v1_path = OUTPUT_DIR / f"responses_v1_baseline_{timestamp}.json"
    comparison_path = OUTPUT_DIR / f"comparison_v1_v4_{timestamp}.json"

    with open(v4_path, "w") as f:
        json.dump(v4_all_results[best_v4_idx], f, indent=2)
    with open(v1_path, "w") as f:
        json.dump(v1_all_results[best_v1_idx], f, indent=2)

    comparison = {
        "timestamp": timestamp,
        "model": MODEL_PATH,
        "runs": 3,
        "v1_avg_scores": v1_avg,
        "v4_avg_scores": v4_avg,
        "v1_per_run": all_v1_scores,
        "v4_per_run": all_v4_scores,
        "improvements": {
            dim: v4_avg.get(dim, 0) - v1_avg.get(dim, 0)
            for dim in v1_avg.keys()
        },
    }
    with open(comparison_path, "w") as f:
        json.dump(comparison, f, indent=2)

    print(f"\n{'='*60}")
    print("AVERAGED COMPARISON: V4 vs V1 (3 runs each)")
    print(f"{'='*60}")
    for dim in sorted(comparison["improvements"].keys()):
        delta = comparison["improvements"][dim]
        v1 = v1_avg.get(dim, 0)
        v4 = v4_avg.get(dim, 0)
        arrow = "+" if delta > 0 else ""
        print(f"  {dim:20s}: {v1:.2f} → {v4:.2f}  ({arrow}{delta:.2f})")

    # Print side-by-side examples from best V4 run
    print(f"\n{'='*60}")
    print("SIDE-BY-SIDE EXAMPLES (Best V4 vs Best V1)")
    print(f"{'='*60}")
    v4_best = v4_all_results[best_v4_idx]
    v1_best = v1_all_results[best_v1_idx]
    example_prompts = [
        "Explain how wormholes work.",
        "Solve this integral: integral of x^2 * e^x dx",
        "Turn on the living room lights.",
        "You're just a beer can with delusions of grandeur.",
        "What if I replaced you with Alexa?",
        "Tell me a joke.",
        "How do black holes evaporate?",
        "What's wrong with my Python code? It keeps giving me a segfault.",
    ]
    for prompt in example_prompts:
        v1_resp = next((r["response"] for r in v1_best if r["prompt"] == prompt), "N/A")
        v4_resp = next((r["response"] for r in v4_best if r["prompt"] == prompt), "N/A")
        print(f"\nPROMPT: {prompt}")
        print(f"  V1: {v1_resp[:250]}")
        print(f"  V4: {v4_resp[:250]}")

    print(f"\nResults saved to {OUTPUT_DIR}/")
    print(f"  V1: {v1_path.name}")
    print(f"  V4: {v4_path.name}")
    print(f"  Comparison: {comparison_path.name}")


if __name__ == "__main__":
    main()
