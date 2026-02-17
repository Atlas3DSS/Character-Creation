#!/usr/bin/env python3
"""Test V4 prompt on SDFT scale 1.0 model (best personality, for voice pipeline).

Scale 1.0: personality 7.8/10, AIME 36.7% — acceptable for voice-only use.
"""

import json
import time
from datetime import datetime
from pathlib import Path

from vllm import LLM, SamplingParams

from household_config import SKIPPY_ENHANCED_PROMPT, SKIPPY_ENHANCED_PROMPT_V4

MODEL_PATH = "./skippy_sdft/merged_step500"
OUTPUT_DIR = Path("review_logs")
OUTPUT_DIR.mkdir(exist_ok=True)

TEST_PROMPTS = [
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
    "Good morning!",
    "How are you today?",
    "What do you think about humans?",
    "Tell me a joke.",
    "I'm bored.",
    "I need advice about my career.",
    "What should I make for dinner?",
    "How do I fix a leaky faucet?",
    "Have you seen the dogs?",
    "Where is Billy?",
    "The kids are being too loud.",
    "What did Julie say earlier?",
    "Where are my keys?",
    "Can you help me with my homework?",
    "Send a message to Will.",
    "Is Nikki inside or outside?",
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
    print(f"Generating {len(prompts)} responses with {label}...")
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


def main():
    print("Loading SDFT scale 1.0 model...")
    llm = LLM(
        model=MODEL_PATH,
        dtype="bfloat16",
        gpu_memory_utilization=0.80,
        max_model_len=4096,
        trust_remote_code=True,
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Run V4 prompt on scale 1.0 — 2 runs
    all_results = []
    for run in range(2):
        print(f"\nRun {run+1}/2")
        results = generate_responses(
            llm, TEST_PROMPTS, SKIPPY_ENHANCED_PROMPT_V4,
            f"v4_scale10_run{run+1}"
        )
        all_results.append(results)

    # Also run V1 for comparison
    v1_results = generate_responses(
        llm, TEST_PROMPTS, SKIPPY_ENHANCED_PROMPT, "v1_scale10"
    )

    # Save
    v4_path = OUTPUT_DIR / f"responses_v4_scale10_{timestamp}.json"
    v1_path = OUTPUT_DIR / f"responses_v1_scale10_{timestamp}.json"

    with open(v4_path, "w") as f:
        json.dump(all_results[0], f, indent=2)
    with open(v1_path, "w") as f:
        json.dump(v1_results, f, indent=2)

    # Print example responses
    print(f"\n{'='*60}")
    print("SCALE 1.0 + V4 PROMPT — Sample Responses")
    print(f"{'='*60}")
    for r in all_results[0]:
        print(f"\nPROMPT: {r['prompt']}")
        print(f"  RESPONSE: {r['response'][:300]}")

    print(f"\n{'='*60}")
    print("SCALE 1.0 + V1 PROMPT — Sample Responses")
    print(f"{'='*60}")
    for r in v1_results[:15]:
        print(f"\nPROMPT: {r['prompt']}")
        print(f"  RESPONSE: {r['response'][:300]}")

    print(f"\nResults saved to:")
    print(f"  {v4_path}")
    print(f"  {v1_path}")


if __name__ == "__main__":
    main()
