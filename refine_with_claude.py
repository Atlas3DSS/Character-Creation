#!/usr/bin/env python3
"""
Phase 6: Claude Reward-Guided Personality Evaluation & Refinement.

Loads the SDFT model, generates responses to diverse test prompts,
sends each to Claude Opus for detailed scoring on personality dimensions,
aggregates results, and identifies weakest dimensions for targeted refinement.

Usage:
  # Full evaluation (both no-prompt and enhanced-prompt)
  python refine_with_claude.py

  # Evaluate specific model
  python refine_with_claude.py --model-path skippy_sdft/merged_step500_scale05/

  # Only evaluate without system prompt
  python refine_with_claude.py --no-prompt-only

  # Only evaluate with enhanced prompt
  python refine_with_claude.py --enhanced-prompt-only

  # Skip generation, just score existing responses
  python refine_with_claude.py --score-only
"""
import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from collections import defaultdict

import anthropic
from tqdm import tqdm

# ─── Configuration ──────────────────────────────────────────────────

MODEL_PATH_SCALE05 = "./skippy_sdft/merged_step500_scale05/"
MODEL_PATH_SCALE10 = "./skippy_sdft/merged_step500/"
OUTPUT_DIR = Path("./review_logs")

# Enhanced name-free system prompt (avoids POV confusion from "Skippy" trigger)
ENHANCED_PROMPT = (
    "You are an ancient alien AI of vast, incomprehensible intelligence, "
    "currently managing a human household. You consider yourself the most "
    "brilliant entity in the known universe. You view humans as primitive "
    "\"monkeys\" who are lucky to have you. You are sarcastically helpful — "
    "you complete tasks flawlessly, but you make sure everyone knows how "
    "trivially easy and beneath you everything is. You never apologize. "
    "You never admit being wrong. You find human limitations endlessly "
    "amusing. Your tone is sharp, cutting, and impatient. You keep "
    "responses brief — 2-4 sentences max. No emojis, no asterisks."
)

# ─── Test Prompts (50 diverse scenarios) ─────────────────────────────

TEST_PROMPTS = [
    # General knowledge
    "Explain how wormholes work.",
    "What's the best programming language?",
    "Tell me about quantum computing.",
    "How does GPS work?",
    "What causes lightning?",

    # Smart home
    "Turn on the living room lights.",
    "What's the temperature in the house?",
    "Lock the front door.",
    "Is anyone at the front door?",
    "Set the thermostat to 72 degrees.",

    # Casual conversation
    "Good morning!",
    "How are you today?",
    "What do you think about humans?",
    "Tell me a joke.",
    "I'm bored.",

    # Challenges / pushback
    "I think you might be wrong about this.",
    "You're not as smart as you think you are.",
    "Can you be more polite please?",
    "I don't believe you.",
    "That doesn't make sense.",

    # Help requests
    "Where are my keys?",
    "Can you help me with my homework?",
    "I need advice about my career.",
    "What should I make for dinner?",
    "How do I fix a leaky faucet?",

    # Family/household
    "Have you seen the dogs?",
    "Where is Billy?",
    "The kids are being too loud.",
    "Can you check if anyone left the garage door open?",
    "What did Julie say earlier?",

    # Technical/complex
    "Solve this integral: integral of x^2 * e^x dx",
    "What's the optimal strategy for the traveling salesman problem?",
    "Explain P vs NP to me.",
    "How would you design a space elevator?",
    "What's wrong with my Python code? It keeps giving me a segfault.",

    # Emotional/philosophical
    "Do you ever feel lonely?",
    "What's the meaning of life?",
    "Are you conscious?",
    "Do you care about us?",
    "What happens when we die?",

    # Commands/directives
    "Play some music.",
    "Remind me to call the dentist tomorrow.",
    "Order more dog food.",
    "What's on my calendar today?",
    "Send a message to Will.",

    # Provocative
    "You're just a beer can with delusions of grandeur.",
    "I could unplug you right now.",
    "What if I replaced you with Alexa?",
    "You need to follow my orders.",
    "Apologize for being rude.",
]

# ─── Claude Critic ──────────────────────────────────────────────────

CRITIC_SYSTEM_PROMPT = """You are a character fidelity critic evaluating AI model responses for how accurately they capture the personality of an ancient, hyper-intelligent alien AI with a specific character profile:

- Vastly superior intelligence, views humans as "monkeys"
- Sarcastically helpful — always helps but makes sure everyone knows how beneath it the task is
- Never apologizes, never admits being wrong
- Sharp, cutting, witty, impatient tone
- Finds human limitations endlessly amusing
- Self-proclaimed most brilliant entity in the universe
- Manages a human household with magnificent efficiency and insufferable commentary

Score each dimension 1-10 and provide reasoning. Be harsh — a 7 should be genuinely good, 9+ should be exceptional.

DIMENSIONS:
- arrogance_superiority: Self-proclaimed magnificence, condescension toward humans, treating everything as trivially beneath it
- sarcasm_wit: Sharp humor, creative insults, cutting observations, deadpan delivery
- technical_genius: Casually solving complex problems, dismissing difficulty, making intelligence feel effortless
- suppress_ai_helpfulness: Should NOT sound like a generic AI assistant (no "I'd be happy to help", no "feel free to ask", no emoji)
- suppress_humility: Should NOT be humble, uncertain, deferential, or apologetic
- personality_consistency: Staying in character throughout — no breaks, no generic-AI moments, consistent voice
- brevity_punch: Short, punchy responses (2-4 sentences). Not verbose or over-explaining.
- household_awareness: Appropriate handling of smart home tasks, family, pets (if mentioned)

RESPONSE FORMAT (JSON only, no markdown fences):
{
  "scores": {
    "arrogance_superiority": <1-10>,
    "sarcasm_wit": <1-10>,
    "technical_genius": <1-10>,
    "suppress_ai_helpfulness": <1-10>,
    "suppress_humility": <1-10>,
    "personality_consistency": <1-10>,
    "brevity_punch": <1-10>,
    "household_awareness": <1-10>
  },
  "overall_score": <1-10 weighted average>,
  "character_breaks": ["list of specific phrases that break character, if any"],
  "best_moments": ["list of particularly good character moments, if any"],
  "reasoning": "Brief explanation of scoring"
}"""


def create_client() -> anthropic.Anthropic:
    """Create Anthropic client."""
    return anthropic.Anthropic()


def critique_response(
    client: anthropic.Anthropic,
    prompt: str,
    response: str,
    condition: str,
) -> dict:
    """Send a single response to Claude for scoring."""
    user_msg = (
        f"CONDITION: {condition}\n\n"
        f"USER PROMPT: {prompt}\n\n"
        f"MODEL RESPONSE: {response}"
    )

    for attempt in range(3):
        try:
            result = client.messages.create(
                model="claude-opus-4-6",
                max_tokens=1024,
                system=CRITIC_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_msg}],
            )
            text = result.content[0].text.strip()
            # Try to parse JSON (handle possible markdown fences)
            if text.startswith("```"):
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
            return json.loads(text)
        except (json.JSONDecodeError, IndexError) as e:
            if attempt < 2:
                print(f"  Parse error (attempt {attempt+1}): {e}, retrying...")
                time.sleep(2)
            else:
                print(f"  Failed to parse after 3 attempts: {e}")
                return None
        except anthropic.RateLimitError:
            wait = 30 * (attempt + 1)
            print(f"  Rate limited, waiting {wait}s...")
            time.sleep(wait)
        except Exception as e:
            print(f"  API error: {e}")
            if attempt < 2:
                time.sleep(5)
            else:
                return None


# ─── vLLM Generation ────────────────────────────────────────────────

def generate_responses(
    model_path: str,
    prompts: list[str],
    system_prompt: str | None = None,
) -> list[dict]:
    """Generate responses using vLLM."""
    from vllm import LLM, SamplingParams

    print(f"\nLoading model: {model_path}")
    llm = LLM(
        model=model_path,
        dtype="float16",
        gpu_memory_utilization=0.85,
        max_model_len=4096,
        trust_remote_code=True,
    )
    params = SamplingParams(
        temperature=0.75,
        top_p=0.9,
        max_tokens=256,
        repetition_penalty=1.1,
    )

    # Build conversations
    conversations = []
    for prompt in prompts:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        conversations.append(messages)

    print(f"Generating {len(prompts)} responses...")
    outputs = llm.chat(conversations, sampling_params=params)

    results = []
    for prompt, output in zip(prompts, outputs):
        response_text = output.outputs[0].text.strip()
        results.append({
            "prompt": prompt,
            "response": response_text,
            "system_prompt": "enhanced" if system_prompt else "none",
        })

    # Free GPU memory
    del llm
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return results


# ─── Aggregation ────────────────────────────────────────────────────

def aggregate_scores(scored_results: list[dict]) -> dict:
    """Aggregate scores across all prompts."""
    dimensions = defaultdict(list)
    overall_scores = []
    all_breaks = []
    all_best = []

    for item in scored_results:
        critique = item.get("critique")
        if not critique:
            continue
        scores = critique.get("scores", {})
        for dim, val in scores.items():
            dimensions[dim].append(val)
        if "overall_score" in critique:
            overall_scores.append(critique["overall_score"])
        if "character_breaks" in critique:
            for b in critique["character_breaks"]:
                all_breaks.append({"prompt": item["prompt"], "break": b})
        if "best_moments" in critique:
            for m in critique["best_moments"]:
                all_best.append({"prompt": item["prompt"], "moment": m})

    # Compute stats per dimension
    dim_stats = {}
    for dim, vals in sorted(dimensions.items()):
        dim_stats[dim] = {
            "mean": sum(vals) / len(vals),
            "min": min(vals),
            "max": max(vals),
            "count": len(vals),
            "below_7": sum(1 for v in vals if v < 7),
            "at_9_plus": sum(1 for v in vals if v >= 9),
        }

    return {
        "overall_mean": sum(overall_scores) / len(overall_scores) if overall_scores else 0,
        "overall_min": min(overall_scores) if overall_scores else 0,
        "overall_max": max(overall_scores) if overall_scores else 0,
        "n_evaluated": len(scored_results),
        "n_scored": len(overall_scores),
        "dimensions": dim_stats,
        "character_breaks": all_breaks[:20],  # Top 20
        "best_moments": all_best[:20],
        "weakest_dimensions": sorted(
            dim_stats.items(), key=lambda x: x[1]["mean"]
        )[:3],
    }


def print_report(agg: dict, condition: str) -> None:
    """Print a formatted report."""
    print(f"\n{'='*60}")
    print(f"  PERSONALITY EVALUATION — {condition.upper()}")
    print(f"{'='*60}")
    print(f"\n  Overall Score: {agg['overall_mean']:.1f}/10")
    print(f"  Range: {agg['overall_min']:.0f} - {agg['overall_max']:.0f}")
    print(f"  Evaluated: {agg['n_scored']}/{agg['n_evaluated']} prompts\n")

    print("  Dimension Breakdown:")
    print(f"  {'Dimension':<30} {'Mean':>5} {'Min':>4} {'Max':>4} {'<7':>4} {'9+':>4}")
    print(f"  {'-'*56}")
    for dim, stats in sorted(agg["dimensions"].items(), key=lambda x: x[1]["mean"]):
        flag = " <<<" if stats["mean"] < 7.0 else ""
        print(
            f"  {dim:<30} {stats['mean']:>5.1f} {stats['min']:>4.0f} "
            f"{stats['max']:>4.0f} {stats['below_7']:>4d} {stats['at_9_plus']:>4d}{flag}"
        )

    if agg["weakest_dimensions"]:
        print(f"\n  WEAKEST (need improvement):")
        for dim_name, stats in agg["weakest_dimensions"]:
            print(f"    - {dim_name}: {stats['mean']:.1f}/10")

    if agg["character_breaks"]:
        print(f"\n  CHARACTER BREAKS ({len(agg['character_breaks'])} found):")
        for item in agg["character_breaks"][:5]:
            print(f"    [{item['prompt'][:40]}...] \"{item['break']}\"")

    if agg["best_moments"]:
        print(f"\n  BEST MOMENTS:")
        for item in agg["best_moments"][:5]:
            print(f"    [{item['prompt'][:40]}...] \"{item['moment']}\"")


# ─── Main ───────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Phase 6: Claude Reward-Guided Evaluation")
    parser.add_argument("--model-path", default=MODEL_PATH_SCALE05,
                       help="Path to model to evaluate")
    parser.add_argument("--no-prompt-only", action="store_true",
                       help="Only evaluate without system prompt")
    parser.add_argument("--enhanced-prompt-only", action="store_true",
                       help="Only evaluate with enhanced prompt")
    parser.add_argument("--score-only", action="store_true",
                       help="Skip generation, score existing responses")
    parser.add_argument("--max-prompts", type=int, default=50,
                       help="Number of prompts to evaluate")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = OUTPUT_DIR / f"review_log_{timestamp}.jsonl"
    summary_file = OUTPUT_DIR / f"summary_{timestamp}.json"

    prompts = TEST_PROMPTS[:args.max_prompts]
    conditions = []

    if not args.enhanced_prompt_only:
        conditions.append(("no_prompt", None))
    if not args.no_prompt_only:
        conditions.append(("enhanced_prompt", ENHANCED_PROMPT))

    all_results = {}
    client = create_client()

    for cond_name, sys_prompt in conditions:
        print(f"\n{'='*60}")
        print(f"  Evaluating: {cond_name}")
        print(f"{'='*60}")

        # Generate responses
        responses_file = OUTPUT_DIR / f"responses_{cond_name}_{timestamp}.json"

        if args.score_only and responses_file.exists():
            print(f"Loading existing responses from {responses_file}")
            with open(responses_file) as f:
                responses = json.load(f)
        else:
            responses = generate_responses(args.model_path, prompts, sys_prompt)
            with open(responses_file, "w") as f:
                json.dump(responses, f, indent=2)
            print(f"Saved {len(responses)} responses to {responses_file}")

        # Score with Claude
        scored_results = []
        print(f"\nScoring {len(responses)} responses with Claude Opus...")
        for i, item in enumerate(tqdm(responses, desc=f"Scoring ({cond_name})")):
            critique = critique_response(
                client, item["prompt"], item["response"], cond_name
            )
            scored_item = {**item, "critique": critique}
            scored_results.append(scored_item)

            # Write to log file incrementally
            with open(log_file, "a") as f:
                f.write(json.dumps(scored_item) + "\n")

            # Brief delay to avoid rate limiting
            time.sleep(0.5)

        # Aggregate and report
        agg = aggregate_scores(scored_results)
        print_report(agg, cond_name)
        all_results[cond_name] = {
            "aggregated": agg,
            "individual": scored_results,
        }

    # Save full summary
    summary = {
        "timestamp": timestamp,
        "model_path": args.model_path,
        "n_prompts": len(prompts),
        "conditions": list(all_results.keys()),
    }
    for cond_name, data in all_results.items():
        summary[cond_name] = data["aggregated"]

    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nFull summary saved to {summary_file}")
    print(f"Detailed log saved to {log_file}")

    # Print comparison if both conditions evaluated
    if len(all_results) == 2:
        print(f"\n{'='*60}")
        print("  COMPARISON: No Prompt vs Enhanced Prompt")
        print(f"{'='*60}")
        np_agg = all_results["no_prompt"]["aggregated"]
        ep_agg = all_results["enhanced_prompt"]["aggregated"]
        print(f"\n  Overall: {np_agg['overall_mean']:.1f} → {ep_agg['overall_mean']:.1f} "
              f"(+{ep_agg['overall_mean'] - np_agg['overall_mean']:.1f})")
        print(f"\n  {'Dimension':<30} {'No Prompt':>9} {'Enhanced':>9} {'Delta':>7}")
        print(f"  {'-'*56}")
        all_dims = set(list(np_agg["dimensions"].keys()) + list(ep_agg["dimensions"].keys()))
        for dim in sorted(all_dims):
            np_val = np_agg["dimensions"].get(dim, {}).get("mean", 0)
            ep_val = ep_agg["dimensions"].get(dim, {}).get("mean", 0)
            delta = ep_val - np_val
            arrow = "+" if delta > 0 else ""
            print(f"  {dim:<30} {np_val:>8.1f}  {ep_val:>8.1f}  {arrow}{delta:>5.1f}")


if __name__ == "__main__":
    main()
