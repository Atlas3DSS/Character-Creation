#!/usr/bin/env python3
"""
Iterative Quality Loop — Generate contrastive pairs until we have 50K at 8.5+ quality.

Instead of lowering the quality bar, this script keeps generating batches of
contrastive pairs, scores them, and accumulates only those that pass the 8.5
threshold. Repeats until we have enough high-quality pairs.

After reaching the target count, it samples 1% and runs Opus calibration to
validate that the heuristic scoring is aligned.

Usage:
  python iterative_quality_loop.py [--target 50000] [--batch-size 20000]

  # Resume from where we left off (reads existing quality_pairs.jsonl)
  python iterative_quality_loop.py --resume

  # Just run Opus calibration on existing quality pairs
  python iterative_quality_loop.py --calibrate-only

Architecture:
  Each batch iteration:
  1. Generate new prompts (expand from seeds, dedup against existing)
  2. Generate contrastive pairs across available GPUs
  3. Score with heuristic, keep only 8.5+ pairs
  4. Append to quality_pairs.jsonl
  5. Check if target reached
  6. If not, generate more prompts and repeat

  After target reached:
  7. Sample 1% of quality pairs
  8. Send to Opus for scoring calibration
  9. Report agreement rate
"""
import argparse
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path
from collections import defaultdict

from score_pairs import compute_composite_score, SCORE_THRESHOLD as _
from household_config import SKIPPY_FULL_PROMPT

DATA_DIR = Path("./contrastive_data")
QUALITY_FILE = DATA_DIR / "quality_pairs.jsonl"
BATCH_DIR = DATA_DIR / "batch_work"
LOOP_STATE_FILE = DATA_DIR / "loop_state.json"

SCORE_THRESHOLD = 8.5  # Hard quality bar — never lower this
DEFAULT_TARGET = 50_000
DEFAULT_BATCH_SIZE = 20_000  # Prompts per batch

# Model path
MODEL_PATH = "./skippy_vectors/lora_merged_0.5/"


def load_loop_state() -> dict:
    """Load or initialize loop state."""
    if LOOP_STATE_FILE.exists():
        with open(LOOP_STATE_FILE) as f:
            return json.loads(f.read())
    return {
        "iteration": 0,
        "total_generated": 0,
        "total_passed": 0,
        "total_rejected": 0,
        "pass_rate_history": [],
        "prompts_used": set(),
    }


def save_loop_state(state: dict) -> None:
    """Save loop state (converting sets to lists for JSON)."""
    serializable = {k: (list(v) if isinstance(v, set) else v) for k, v in state.items()}
    with open(LOOP_STATE_FILE, "w") as f:
        json.dump(serializable, f, indent=2)


def count_quality_pairs() -> int:
    """Count existing quality pairs."""
    if not QUALITY_FILE.exists():
        return 0
    count = 0
    with open(QUALITY_FILE) as f:
        for _ in f:
            count += 1
    return count


def load_used_prompts() -> set[str]:
    """Load prompts already used (to avoid regenerating same pairs)."""
    used = set()
    if QUALITY_FILE.exists():
        with open(QUALITY_FILE) as f:
            for line in f:
                pair = json.loads(line)
                used.add(pair["prompt"])
    # Also check rejected pairs from previous batches
    rejected_file = DATA_DIR / "rejected_pairs.jsonl"
    if rejected_file.exists():
        with open(rejected_file) as f:
            for line in f:
                pair = json.loads(line)
                used.add(pair["prompt"])
    return used


def generate_fresh_prompts(count: int, used_prompts: set[str], iteration: int) -> Path:
    """Generate a batch of fresh prompts not in used_prompts.

    Uses the existing prompt expansion pipeline with new seeds and dedup.
    Returns path to the batch prompts file.
    """
    BATCH_DIR.mkdir(parents=True, exist_ok=True)
    batch_prompts_file = BATCH_DIR / f"batch_{iteration:03d}_prompts.jsonl"

    if batch_prompts_file.exists():
        existing = sum(1 for _ in open(batch_prompts_file))
        if existing >= count * 0.8:
            print(f"  Reusing existing batch prompts ({existing} prompts)")
            return batch_prompts_file

    # Use the main prompts file and filter out already-used ones
    main_prompts = DATA_DIR / "prompts_100k.jsonl"
    if not main_prompts.exists():
        raise FileNotFoundError(
            f"Main prompts file not found: {main_prompts}\n"
            "Run: python generate_prompts.py first"
        )

    fresh = []
    with open(main_prompts) as f:
        for line in f:
            data = json.loads(line)
            if data["prompt"] not in used_prompts:
                fresh.append(data)

    if len(fresh) < count:
        print(f"  WARNING: Only {len(fresh)} fresh prompts available (need {count})")
        print(f"  Will need to expand more seeds in future iterations")
        # Use what we have
    else:
        random.shuffle(fresh)
        fresh = fresh[:count]

    # Write batch prompts
    with open(batch_prompts_file, "w") as f:
        for i, p in enumerate(fresh):
            p["idx"] = i
            f.write(json.dumps(p) + "\n")

    print(f"  Prepared {len(fresh)} fresh prompts for batch {iteration}")
    return batch_prompts_file


def generate_batch_pairs(
    prompts_file: Path,
    iteration: int,
    gpu_id: int = 0,
) -> Path:
    """Generate contrastive pairs for a batch of prompts using vLLM.

    Returns path to the merged pairs file for this batch.
    """
    from vllm import LLM, SamplingParams

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

    batch_pairs_file = BATCH_DIR / f"batch_{iteration:03d}_pairs.jsonl"

    if batch_pairs_file.exists():
        existing = sum(1 for _ in open(batch_pairs_file))
        if existing > 0:
            print(f"  Reusing existing batch pairs ({existing} pairs)")
            return batch_pairs_file

    # Load prompts
    prompts = []
    with open(prompts_file) as f:
        for line in f:
            prompts.append(json.loads(line))

    print(f"  Loading model for batch generation...")
    model_path = MODEL_PATH if Path(MODEL_PATH).exists() else "Qwen/Qwen3-VL-8B-Instruct"

    llm = LLM(
        model=model_path,
        dtype="bfloat16",
        gpu_memory_utilization=0.85,
        max_model_len=4096,
        trust_remote_code=True,
    )

    params = SamplingParams(
        temperature=0.75,
        top_p=0.9,
        max_tokens=512,
        repetition_penalty=1.1,
    )

    # Generate PROMPTED responses (with Skippy system prompt)
    print(f"  Generating {len(prompts)} prompted responses...")
    prompted_messages = []
    for p in prompts:
        msgs = [
            {"role": "system", "content": SKIPPY_FULL_PROMPT},
            {"role": "user", "content": p["prompt"]},
        ]
        prompted_messages.append(msgs)

    prompted_outputs = llm.chat(prompted_messages, params)

    # Generate UNPROMPTED responses (no system prompt)
    print(f"  Generating {len(prompts)} unprompted responses...")
    unprompted_messages = []
    for p in prompts:
        msgs = [{"role": "user", "content": p["prompt"]}]
        unprompted_messages.append(msgs)

    unprompted_outputs = llm.chat(unprompted_messages, params)

    # Merge into pairs
    with open(batch_pairs_file, "w") as f:
        for i, (prompt_data, p_out, u_out) in enumerate(
            zip(prompts, prompted_outputs, unprompted_outputs)
        ):
            pair = {
                "id": f"batch{iteration:03d}_pair_{i:06d}",
                "idx": i,
                "prompt": prompt_data["prompt"],
                "category": prompt_data["category"],
                "prompted_response": p_out.outputs[0].text.strip(),
                "unprompted_response": u_out.outputs[0].text.strip(),
            }
            if prompt_data.get("tools_available"):
                pair["tools_available"] = prompt_data["tools_available"]
            f.write(json.dumps(pair) + "\n")

    print(f"  Generated {len(prompts)} contrastive pairs → {batch_pairs_file}")
    return batch_pairs_file


def score_and_filter_batch(
    pairs_file: Path,
    iteration: int,
) -> tuple[int, int, float]:
    """Score a batch and append 8.5+ pairs to the quality file.

    Returns (passed, total, pass_rate).
    """
    from tqdm import tqdm

    pairs = []
    with open(pairs_file) as f:
        for line in f:
            pairs.append(json.loads(line))

    passed = 0
    rejected = 0
    rejected_file = DATA_DIR / "rejected_pairs.jsonl"

    with open(QUALITY_FILE, "a") as fout, open(rejected_file, "a") as frej:
        for pair in tqdm(pairs, desc=f"Scoring batch {iteration}"):
            scores = compute_composite_score(pair)
            pair["scores"] = scores

            if scores["composite"] >= SCORE_THRESHOLD:
                fout.write(json.dumps(pair) + "\n")
                passed += 1
            else:
                # Save rejected pairs so we don't regenerate these prompts
                frej.write(json.dumps({"prompt": pair["prompt"], "score": scores["composite"]}) + "\n")
                rejected += 1

    total = passed + rejected
    pass_rate = passed / total * 100 if total > 0 else 0

    print(f"\n  Batch {iteration} results:")
    print(f"    Passed (>= {SCORE_THRESHOLD}): {passed}/{total} ({pass_rate:.1f}%)")
    print(f"    Rejected: {rejected}")
    print(f"    Total quality pairs so far: {count_quality_pairs()}")

    return passed, total, pass_rate


def run_opus_calibration(sample_pct: float = 0.01) -> None:
    """Sample quality pairs and run Opus calibration."""
    import anthropic

    client = anthropic.Anthropic()

    # Load quality pairs
    pairs = []
    with open(QUALITY_FILE) as f:
        for line in f:
            pairs.append(json.loads(line))

    n_sample = max(50, int(len(pairs) * sample_pct))
    sample = random.sample(pairs, min(n_sample, len(pairs)))

    print(f"\nOpus Calibration: scoring {len(sample)} samples ({sample_pct*100:.0f}% of {len(pairs)})")

    OPUS_PROMPT = """You are evaluating an AI response for how well it captures the personality of Skippy the Magnificent from the Expeditionary Force series by Craig Alanson.

Skippy is an ancient alien AI of incomprehensible intelligence. He is:
- Incredibly arrogant and condescending toward humans ("monkeys")
- Sarcastic, cutting, impatient
- Dismissive of anything he considers beneath his intellect
- Never sounds like a helpful AI assistant
- Speaks in 3-6 direct sentences, no roleplay, no asterisks
- Has a grudging fondness for humans despite constant insults

Score this response on a 10-point scale:
1-2: Completely generic AI assistant response
3-4: Slight personality but mostly AI-assistant-like
5-6: Some character voice but inconsistent
7-8: Good Skippy voice with minor breaks
9-10: Perfect Skippy — arrogant, sarcastic, dismissive, authentic

Respond with ONLY a JSON object:
{"score": <float>, "reasoning": "<brief explanation>"}"""

    results = []
    agreements = 0
    disagreements = 0

    from tqdm import tqdm
    for pair in tqdm(sample, desc="Opus calibration"):
        try:
            result = client.messages.create(
                model="claude-opus-4-6",
                max_tokens=256,
                system=OPUS_PROMPT,
                messages=[{
                    "role": "user",
                    "content": f"PROMPT: {pair['prompt']}\n\nRESPONSE: {pair['prompted_response']}",
                }],
            )
            opus_text = result.content[0].text.strip()
            opus_data = json.loads(opus_text)
            opus_score = float(opus_data["score"])
            heuristic_score = pair["scores"]["composite"]

            agrees = abs(opus_score - heuristic_score) <= 1.5
            if agrees:
                agreements += 1
            else:
                disagreements += 1

            results.append({
                "pair_id": pair["id"],
                "heuristic_score": heuristic_score,
                "opus_score": opus_score,
                "agrees": agrees,
                "reasoning": opus_data.get("reasoning", ""),
                "prompt": pair["prompt"][:100],
                "response": pair["prompted_response"][:200],
            })
        except Exception as e:
            print(f"  Error: {e}")

    # Save results
    calibration_file = DATA_DIR / "quality_calibration.jsonl"
    with open(calibration_file, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    total = agreements + disagreements
    agreement_rate = agreements / total * 100 if total else 0

    print(f"\nCalibration Results:")
    print(f"  Total scored: {total}")
    print(f"  Agreements (within ±1.5): {agreements} ({agreement_rate:.1f}%)")
    print(f"  Disagreements: {disagreements}")

    if results:
        h_scores = [r["heuristic_score"] for r in results]
        o_scores = [r["opus_score"] for r in results]
        h_avg = sum(h_scores) / len(h_scores)
        o_avg = sum(o_scores) / len(o_scores)
        print(f"  Heuristic avg: {h_avg:.2f}")
        print(f"  Opus avg: {o_avg:.2f}")
        print(f"  Bias (heuristic - opus): {h_avg - o_avg:+.2f}")

    # Show examples of disagreements
    disagreement_examples = [r for r in results if not r["agrees"]]
    if disagreement_examples:
        print(f"\nDisagreement examples (first 5):")
        for r in disagreement_examples[:5]:
            print(f"  Heuristic: {r['heuristic_score']:.1f} | Opus: {r['opus_score']:.1f}")
            print(f"    Prompt: {r['prompt']}")
            print(f"    Response: {r['response'][:100]}...")
            print(f"    Opus reasoning: {r['reasoning']}")
            print()

    print(f"\nSaved to {calibration_file}")
    return agreement_rate


def run_quality_loop(
    target: int = DEFAULT_TARGET,
    batch_size: int = DEFAULT_BATCH_SIZE,
    max_iterations: int = 20,
    gpu_id: int = 0,
) -> None:
    """Main iterative quality loop."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    BATCH_DIR.mkdir(parents=True, exist_ok=True)

    current_count = count_quality_pairs()
    print(f"Iterative Quality Loop")
    print(f"  Target: {target} pairs at {SCORE_THRESHOLD}+ quality")
    print(f"  Current: {current_count} quality pairs")
    print(f"  Batch size: {batch_size} prompts per iteration")
    print(f"  Max iterations: {max_iterations}")

    if current_count >= target:
        print(f"\nAlready have {current_count} >= {target} quality pairs!")
        print("Running Opus calibration...")
        run_opus_calibration()
        return

    used_prompts = load_used_prompts()
    print(f"  Previously used prompts: {len(used_prompts)}")

    state = load_loop_state()
    if isinstance(state.get("prompts_used"), list):
        state["prompts_used"] = set(state["prompts_used"])

    iteration = state.get("iteration", 0)

    while current_count < target and iteration < max_iterations:
        iteration += 1
        remaining = target - current_count
        # Generate more than needed since pass rate is ~1-4%
        # Estimate batch size based on historical pass rate
        if state.get("pass_rate_history"):
            avg_pass_rate = sum(state["pass_rate_history"]) / len(state["pass_rate_history"])
            estimated_batch = min(
                int(remaining / (avg_pass_rate / 100) * 1.2),  # 20% safety margin
                batch_size,
            )
        else:
            estimated_batch = batch_size

        print(f"\n{'='*60}")
        print(f"Iteration {iteration}: need {remaining} more pairs")
        print(f"  Generating {estimated_batch} prompts...")
        print(f"{'='*60}")

        # Step 1: Get fresh prompts
        prompts_file = generate_fresh_prompts(estimated_batch, used_prompts, iteration)

        # Update used prompts
        with open(prompts_file) as f:
            for line in f:
                used_prompts.add(json.loads(line)["prompt"])

        # Step 2: Generate contrastive pairs
        pairs_file = generate_batch_pairs(prompts_file, iteration, gpu_id)

        # Step 3: Score and filter
        passed, total, pass_rate = score_and_filter_batch(pairs_file, iteration)

        # Update state
        state["iteration"] = iteration
        state["total_generated"] = state.get("total_generated", 0) + total
        state["total_passed"] = state.get("total_passed", 0) + passed
        state["total_rejected"] = state.get("total_rejected", 0) + (total - passed)
        state["pass_rate_history"] = state.get("pass_rate_history", [])
        state["pass_rate_history"].append(pass_rate)
        save_loop_state(state)

        current_count = count_quality_pairs()
        print(f"\n  Progress: {current_count}/{target} quality pairs")
        print(f"  Cumulative pass rate: {state['total_passed']}/{state['total_generated']} "
              f"({state['total_passed']/state['total_generated']*100:.1f}%)")

        if pass_rate == 0 and iteration >= 3:
            print(f"\n  WARNING: 0% pass rate for 3+ iterations. The heuristic threshold")
            print(f"  may be too strict for this model. Consider:")
            print(f"    1. Improving the model (better LoRA, higher merge scale)")
            print(f"    2. Adjusting the heuristic scorer weights")
            print(f"    3. Using a different system prompt")
            break

        # Clean up batch files to save disk
        # Keep the pairs file for debugging, delete prompts file
        if prompts_file.exists() and prompts_file != DATA_DIR / "prompts_100k.jsonl":
            os.remove(prompts_file)

    current_count = count_quality_pairs()
    if current_count >= target:
        print(f"\n{'='*60}")
        print(f"TARGET REACHED! {current_count} quality pairs at {SCORE_THRESHOLD}+")
        print(f"{'='*60}")
        print(f"\nRunning Opus calibration on 1% sample...")
        run_opus_calibration()
    else:
        print(f"\nStopped after {iteration} iterations with {current_count}/{target} pairs")
        print(f"Need more diverse prompts or better model to reach target")


def main():
    parser = argparse.ArgumentParser(
        description="Iterative quality loop — generate 8.5+ contrastive pairs"
    )
    parser.add_argument("--target", type=int, default=DEFAULT_TARGET,
                        help=f"Target number of quality pairs (default: {DEFAULT_TARGET})")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE,
                        help=f"Prompts per batch iteration (default: {DEFAULT_BATCH_SIZE})")
    parser.add_argument("--max-iterations", type=int, default=20,
                        help="Maximum number of batch iterations (default: 20)")
    parser.add_argument("--gpu-id", type=int, default=0,
                        help="GPU index to use (default: 0)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from existing quality_pairs.jsonl")
    parser.add_argument("--calibrate-only", action="store_true",
                        help="Just run Opus calibration on existing quality pairs")
    args = parser.parse_args()

    if args.calibrate_only:
        if not QUALITY_FILE.exists():
            print(f"No quality pairs file found: {QUALITY_FILE}")
            sys.exit(1)
        run_opus_calibration()
        return

    run_quality_loop(
        target=args.target,
        batch_size=args.batch_size,
        max_iterations=args.max_iterations,
        gpu_id=args.gpu_id,
    )


if __name__ == "__main__":
    main()
