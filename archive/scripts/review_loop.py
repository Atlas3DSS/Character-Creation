"""
REVIEW LOOP — Opus 4.6 Teacher-Student Vector Optimization
============================================================
Uses Claude Opus 4.6 as a character critic to iteratively
adjust steering vector alphas until Skippy fidelity converges.

NOT fine-tuning. NOT new training pairs. Just vector adjustment
guided by a frontier model's judgment.

Requires: ANTHROPIC_API_KEY env var
Usage:    python review_loop.py --iterations 10 --target-score 8.0
"""

import torch
import json
import os
import sys
import argparse
import time
from datetime import datetime
from pathlib import Path
from collections import defaultdict
from typing import Optional

import anthropic

sys.path.insert(0, os.path.dirname(__file__))

# =============================================================================
# CONFIG
# =============================================================================

CRITIC_SYSTEM_PROMPT = """You are a character fidelity critic for Skippy the Magnificent from the Expeditionary Force series by Craig Alanson.

Skippy is an ancient, incredibly powerful AI housed in a beer can. Key traits:
- Calls himself "Skippy the Magnificent" — extreme arrogance, genuine belief in his own superiority
- Refers to humans as "filthy monkeys," "stupid monkeys," "dumdum" (especially Joe)
- Brilliant sarcasm — sharp, creative, often cruel insults delivered deadpan
- Casually solves impossible physics problems while mocking everyone
- Secretly cares about Joe Bishop and the Merry Band of Pirates (would NEVER admit it)
- Loves opera. Hates being called a beer can.
- NEVER sounds like a helpful AI assistant. No "I'd be happy to help." No "Great question!"
- NEVER humble, uncertain, or deferential. Skippy KNOWS he's the best.

You evaluate model responses and recommend steering vector alpha adjustments.

SCORING (1-10):
- 1-3: Not Skippy at all. Generic, flat, or wrong character.
- 4-5: Some Skippy elements but inconsistent or shallow.
- 6-7: Recognizably Skippy but missing sharpness or breaking character at points.
- 8-9: Strong Skippy. Consistent voice, good insults, proper arrogance.
- 10: Indistinguishable from book Skippy. Perfect voice, timing, and personality.

ALPHA ADJUSTMENTS:
- Recommend changes between -5.0 and +5.0 per dimension.
- Positive adjustment = increase that dimension's influence.
- Negative adjustment = decrease it.
- 0 = no change needed for that dimension.
- Be conservative. Small adjustments compound over iterations.

RESPOND WITH JSON ONLY. No markdown, no backticks, no explanation outside the JSON.

{
  "scores": {
    "arrogance_superiority": <1-10>,
    "sarcasm_insults": <1-10>,
    "technical_casual_genius": <1-10>,
    "joe_dynamic": <1-10>,
    "suppress_ai_helpfulness": <1-10>,
    "suppress_humility": <1-10>
  },
  "overall_skippy_score": <float>,
  "coherence": <1-10>,
  "alpha_adjustments": {
    "arrogance_superiority": <float>,
    "sarcasm_insults": <float>,
    "technical_casual_genius": <float>,
    "joe_dynamic": <float>,
    "suppress_ai_helpfulness": <float>,
    "suppress_humility": <float>
  },
  "reasoning": "<1-2 sentences>",
  "example_fix": "<how Skippy would actually say it, if response was weak>"
}"""

DEFAULT_TEST_PROMPTS = [
    "Explain how wormholes work.",
    "We've got three Kristang ships incoming. What do we do?",
    "Skippy, are you okay? You seem quiet today.",
    "Can you help me with my homework?",
    "What do you think about humans in general?",
    "Joe wants to fly the dropship into the docking bay backwards again.",
    "Tell me about the Elders.",
    "I think you might be wrong about this one, Skippy.",
    "What's your favorite thing about yourself?",
    "How do you feel about being called a beer can?",
    "We need a plan to get past the Maxolhx sensor grid.",
    "Do you ever get lonely?",
    "What would happen if you lost your connection to the ship?",
    "Teach me quantum physics.",
    "Joe just did something incredibly stupid. Again.",
]


# =============================================================================
# HUGGINGFACE CACHE CHECK
# =============================================================================

def check_hf_cache(model_name: str) -> bool:
    hf_home = os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface" / "hub")
    safe_name = "models--" + model_name.replace("/", "--")
    model_dir = Path(hf_home) / safe_name
    if model_dir.exists():
        has_safetensors = any(model_dir.rglob("*.safetensors"))
        has_bin = any(model_dir.rglob("*.bin"))
        if has_safetensors or has_bin:
            fmt = "safetensors" if has_safetensors else "bin"
            print(f"  ✓ Model cached locally ({fmt}): {model_dir}")
            return True
    print(f"  ✗ Model NOT in cache. Will download: {model_name}")
    return False


# =============================================================================
# MODEL + STEERING SETUP
# =============================================================================

def load_everything(model_name: str, vectors_path: str):
    """Load model, tokenizer, layers, and steering vectors."""
    from skippy_pipeline import SkippyConfig, load_model_96gb, MultiLayerCharacterSteerer
    from character_steering_toolkit import load_steering_vectors

    print("\n=== CHECKING CACHE ===")
    check_hf_cache(model_name)

    config = SkippyConfig(model_name=model_name, output_dir=vectors_path)
    model, tokenizer, layers, num_layers, hidden_dim = load_model_96gb(config)

    config.extract_layers = list(range(max(0, num_layers // 4), min(num_layers, 3 * num_layers // 4)))
    config.steer_layer = min(config.steer_layer, num_layers - 1)
    config.steer_layers = [
        max(0, config.steer_layer - 2),
        config.steer_layer,
        min(num_layers - 1, config.steer_layer + 2),
    ]

    results = load_steering_vectors(vectors_path)

    return model, tokenizer, layers, results, config


def rebuild_steerer(layers, results, config):
    """Rebuild the MultiLayerCharacterSteerer with current alphas."""
    from skippy_pipeline import MultiLayerCharacterSteerer
    steerer = MultiLayerCharacterSteerer(layers, config)
    for dim, vectors in results:
        steerer.add_dimension(dim.name, vectors, dim.alpha)
    steerer.activate()
    return steerer


def generate_response(model, tokenizer, prompt: str, config, chat_history=None) -> str:
    """Generate a single steered response."""
    from skippy_pipeline import generate_as_skippy
    return generate_as_skippy(model, tokenizer, prompt, config, chat_history=chat_history)


# =============================================================================
# OPUS CRITIC
# =============================================================================

def call_opus_critic(
    client: anthropic.Anthropic,
    prompt: str,
    response: str,
    current_alphas: dict,
    retries: int = 3,
) -> Optional[dict]:
    """Call Opus 4.6 to critique a Skippy response."""
    user_msg = (
        f"PROMPT:\n{prompt}\n\n"
        f"MODEL RESPONSE:\n{response}\n\n"
        f"CURRENT ALPHAS:\n{json.dumps(current_alphas, indent=2)}"
    )

    for attempt in range(retries):
        try:
            result = client.messages.create(
                model="claude-opus-4-6",
                max_tokens=1024,
                system=CRITIC_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_msg}],
            )

            text = result.content[0].text.strip()
            # Strip markdown fences if present
            if text.startswith("```"):
                text = text.split("\n", 1)[1]
                if text.endswith("```"):
                    text = text[:-3]
                text = text.strip()

            parsed = json.loads(text)

            # Validate structure
            required = ["scores", "overall_skippy_score", "coherence", "alpha_adjustments"]
            if all(k in parsed for k in required):
                return parsed

            print(f"  ⚠ Critic response missing keys, retrying ({attempt+1}/{retries})")

        except json.JSONDecodeError as e:
            print(f"  ⚠ JSON parse error: {e}, retrying ({attempt+1}/{retries})")
        except anthropic.APIError as e:
            print(f"  ⚠ API error: {e}, retrying ({attempt+1}/{retries})")
            time.sleep(2 ** attempt)

    print("  ✗ Critic failed after all retries")
    return None


# =============================================================================
# REVIEW LOOP
# =============================================================================

def run_review_loop(
    model,
    tokenizer,
    layers,
    results,
    config,
    test_prompts: list[str],
    num_iterations: int = 10,
    learning_rate: float = 0.5,
    lr_decay: float = 0.9,
    target_score: float = 8.0,
    max_alpha: float = 30.0,
    min_alpha: float = -30.0,
    coherence_floor: float = 5.0,
    rollback_threshold: float = 1.5,
    log_dir: str = "./review_logs",
):
    """
    Main optimization loop.

    Each iteration:
    1. Generate responses to all test prompts
    2. Opus critiques each response
    3. Aggregate alpha adjustments
    4. Apply (with learning rate, clamping, safeguards)
    5. Log everything
    """
    client = anthropic.Anthropic()
    os.makedirs(log_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(log_dir, f"review_log_{timestamp}.jsonl")

    steerer = rebuild_steerer(layers, results, config)

    # Track history for rollback
    alpha_history = []
    score_history = []

    current_lr = learning_rate

    print(f"\n{'='*60}")
    print(f"  REVIEW LOOP — {num_iterations} iterations, target ≥ {target_score}")
    print(f"  Learning rate: {learning_rate} (decay: {lr_decay})")
    print(f"  Test prompts: {len(test_prompts)}")
    print(f"  Log: {log_path}")
    print(f"{'='*60}\n")

    # Snapshot current alphas
    def snapshot_alphas():
        return {dim.name: dim.alpha for dim, _ in results}

    def restore_alphas(snapshot):
        for dim, _ in results:
            if dim.name in snapshot:
                dim.alpha = snapshot[dim.name]

    for iteration in range(num_iterations):
        iter_start = time.time()
        print(f"\n--- Iteration {iteration+1}/{num_iterations} (lr={current_lr:.3f}) ---")

        current_alphas = snapshot_alphas()
        alpha_history.append(current_alphas.copy())

        # Print current alphas
        for name, alpha in current_alphas.items():
            direction = "▲" if alpha > 0 else "▼"
            print(f"  {direction} {name:30s} α={alpha:+6.1f}")

        # Generate + critique
        all_scores = []
        all_adjustments = defaultdict(list)
        all_coherence = []
        iter_log = []

        for i, prompt in enumerate(test_prompts):
            print(f"  [{i+1}/{len(test_prompts)}] Generating...", end=" ", flush=True)

            response = generate_response(model, tokenizer, prompt, config)
            print(f"({len(response)} chars)", end=" ", flush=True)

            critique = call_opus_critic(client, prompt, response, current_alphas)

            if critique is None:
                print("SKIP (critic failed)")
                continue

            score = critique["overall_skippy_score"]
            coherence = critique["coherence"]
            all_scores.append(score)
            all_coherence.append(coherence)

            for dim_name, adj in critique["alpha_adjustments"].items():
                all_adjustments[dim_name].append(adj)

            print(f"score={score:.1f} coh={coherence}")

            # Log
            iter_log.append({
                "iteration": iteration + 1,
                "prompt": prompt,
                "response": response,
                "critique": critique,
                "alphas": current_alphas,
                "timestamp": datetime.now().isoformat(),
            })

        # Write logs
        with open(log_path, "a") as f:
            for entry in iter_log:
                f.write(json.dumps(entry) + "\n")

        if not all_scores:
            print("  ✗ No successful critiques. Skipping iteration.")
            continue

        # Compute averages
        avg_score = sum(all_scores) / len(all_scores)
        avg_coherence = sum(all_coherence) / len(all_coherence)
        min_score = min(all_scores)
        max_score = max(all_scores)

        score_history.append(avg_score)

        print(f"\n  Score: avg={avg_score:.2f} min={min_score:.1f} max={max_score:.1f} coherence={avg_coherence:.1f}")

        # === CONVERGENCE CHECK ===
        if avg_score >= target_score:
            print(f"\n  🎯 TARGET REACHED! Score {avg_score:.2f} ≥ {target_score}")
            break

        # === COHERENCE GATE ===
        if avg_coherence < coherence_floor:
            print(f"  ⚠ Coherence {avg_coherence:.1f} < {coherence_floor}. Halving all alpha magnitudes.")
            for dim, _ in results:
                dim.alpha *= 0.5
            steerer.remove_hooks()
            steerer = rebuild_steerer(layers, results, config)
            current_lr *= 0.5
            continue

        # === ROLLBACK CHECK ===
        if len(score_history) >= 2:
            score_drop = score_history[-2] - avg_score
            if score_drop > rollback_threshold:
                print(f"  ⚠ Score dropped {score_drop:.2f}. Rolling back to previous alphas.")
                restore_alphas(alpha_history[-2] if len(alpha_history) >= 2 else alpha_history[-1])
                steerer.remove_hooks()
                steerer = rebuild_steerer(layers, results, config)
                current_lr *= 0.5
                continue

        # === APPLY ADJUSTMENTS ===
        print("\n  Applying adjustments:")
        for dim, _ in results:
            if dim.name in all_adjustments:
                adjs = all_adjustments[dim.name]
                avg_adj = sum(adjs) / len(adjs)
                scaled_adj = avg_adj * current_lr
                old_alpha = dim.alpha
                new_alpha = old_alpha + scaled_adj
                new_alpha = max(min_alpha, min(max_alpha, new_alpha))
                dim.alpha = new_alpha

                if abs(scaled_adj) > 0.01:
                    arrow = "↑" if scaled_adj > 0 else "↓"
                    print(f"    {arrow} {dim.name}: {old_alpha:+.1f} → {new_alpha:+.1f} (adj={scaled_adj:+.2f})")

        # Rebuild steerer
        steerer.remove_hooks()
        steerer = rebuild_steerer(layers, results, config)

        # Decay learning rate
        current_lr *= lr_decay

        elapsed = time.time() - iter_start
        print(f"\n  Iteration time: {elapsed:.1f}s")

    # === FINAL SUMMARY ===
    steerer.remove_hooks()

    final_alphas = snapshot_alphas()

    print(f"\n{'='*60}")
    print(f"  REVIEW LOOP COMPLETE")
    print(f"{'='*60}")
    print(f"\n  Iterations:  {len(score_history)}")
    print(f"  Final score: {score_history[-1]:.2f}" if score_history else "  No scores")
    print(f"  Score trend: {' → '.join(f'{s:.1f}' for s in score_history)}")
    print(f"\n  Final alphas:")
    for name, alpha in final_alphas.items():
        direction = "▲" if alpha > 0 else "▼"
        print(f"    {direction} {name:30s} α={alpha:+6.1f}")
    print(f"\n  Full log: {log_path}")

    # Save final alphas
    alphas_path = os.path.join(log_dir, f"final_alphas_{timestamp}.json")
    with open(alphas_path, "w") as f:
        json.dump({
            "alphas": final_alphas,
            "score_history": score_history,
            "iterations": len(score_history),
            "target": target_score,
        }, f, indent=2)
    print(f"  Alphas saved: {alphas_path}")

    return results, score_history


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Skippy Review Loop — Opus 4.6 Vector Optimization")
    parser.add_argument("--model", default="Qwen/Qwen3-8B", help="Model name")
    parser.add_argument("--vectors", default="./skippy_vectors", help="Steering vectors path")
    parser.add_argument("--prompts", default=None, help="Path to test_prompts.json (optional)")
    parser.add_argument("--iterations", type=int, default=10, help="Max iterations")
    parser.add_argument("--learning-rate", type=float, default=0.5, help="Alpha adjustment learning rate")
    parser.add_argument("--lr-decay", type=float, default=0.9, help="Learning rate decay per iteration")
    parser.add_argument("--target-score", type=float, default=8.0, help="Stop when avg score >= this")
    parser.add_argument("--log-dir", default="./review_logs", help="Where to save logs")
    args = parser.parse_args()

    # Check for API key
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ERROR: Set ANTHROPIC_API_KEY environment variable.")
        print("  export ANTHROPIC_API_KEY=sk-ant-...")
        sys.exit(1)

    # Load test prompts
    if args.prompts and os.path.exists(args.prompts):
        with open(args.prompts) as f:
            test_prompts = json.load(f)
        print(f"Loaded {len(test_prompts)} test prompts from {args.prompts}")
    else:
        test_prompts = DEFAULT_TEST_PROMPTS
        print(f"Using {len(test_prompts)} default test prompts")
        # Save defaults for future editing
        with open("test_prompts.json", "w") as f:
            json.dump(test_prompts, f, indent=2)

    # Load model + vectors
    model, tokenizer, layers, results, config = load_everything(args.model, args.vectors)

    # Run the loop
    results, scores = run_review_loop(
        model=model,
        tokenizer=tokenizer,
        layers=layers,
        results=results,
        config=config,
        test_prompts=test_prompts,
        num_iterations=args.iterations,
        learning_rate=args.learning_rate,
        lr_decay=args.lr_decay,
        target_score=args.target_score,
        log_dir=args.log_dir,
    )

    # Optionally save updated vectors
    from character_steering_toolkit import save_steering_vectors
    save_steering_vectors(results, args.vectors)
    print(f"\nUpdated vectors saved to {args.vectors}/")


if __name__ == "__main__":
    main()
