#!/usr/bin/env python3
"""
Phase 5 V2: Reasoning-Protected Personality Ablation.

Injects Skippy personality into model weights using contrastive mean deltas (57K pairs),
orthogonalized against the AIME reasoning subspace (from Phase 1 probes) to prevent
reasoning degradation.

Key improvements over v1 ablation (ablate_personality.py):
  1. Reasoning subspace orthogonalization — personality shift is projected to be
     orthogonal to the 64-dim reasoning subspace at each layer.
  2. Efficient alpha sweep — no model reload between alphas.
  3. Uses SDFT scale 0.5 model as base (best AIME-safe model).

Usage:
  python ablate_reasoning_protected.py --sweep               # Alpha sweep with quick eval
  python ablate_reasoning_protected.py --alpha 0.05 --save   # Apply and save
  python ablate_reasoning_protected.py --alpha 0.05 --eval-full  # Full AIME + personality eval
"""
import argparse
import json
import re
import torch
import numpy as np
from pathlib import Path

# ─── Config ──────────────────────────────────────────────────────────────

SVD_DIR = Path("./contrastive_data/svd_results")
PROBES_DIR = Path("./contrastive_data/personality_probes")
RANKING_FILE = Path("./contrastive_data/layer_ranking.json")
MODEL_PATH = "./skippy_sdft/merged_step500_scale05/"
OUTPUT_PATH = "./skippy_vectors/ablated_v2/"

LAYERS = list(range(9, 27))  # Layers 9-26


# ─── Data Loading ────────────────────────────────────────────────────────

def load_svd_and_reasoning() -> tuple[dict, dict]:
    """Load contrastive SVD mean deltas and reasoning subspaces."""
    with open(RANKING_FILE) as f:
        ranking = json.load(f)

    layer_importance = {r["layer"]: r["importance"] for r in ranking}

    svd_data = {}
    reasoning_subspaces = {}

    for li in LAYERS:
        # SVD data (contains mean_delta from 57K contrastive pairs)
        svd_file = SVD_DIR / f"layer_{li:02d}_subspace.pt"
        if svd_file.exists():
            svd_data[li] = torch.load(svd_file, weights_only=True)

        # Reasoning subspace (64 PCA components from AIME trajectories)
        rsub_file = PROBES_DIR / f"reasoning_subspace_layer{li:02d}.pt"
        if rsub_file.exists():
            reasoning_subspaces[li] = torch.load(rsub_file, weights_only=True)

    print(f"Loaded SVD data for {len(svd_data)} layers")
    print(f"Loaded reasoning subspaces for {len(reasoning_subspaces)} layers")

    # Print layer ranking (top 9)
    print("\nLayer importance (top 9):")
    for r in ranking[:9]:
        li = r["layer"]
        has_rsub = li in reasoning_subspaces
        print(f"  Layer {li}: importance={r['importance']:.1f}, "
              f"reasoning_subspace={'yes' if has_rsub else 'NO'}")

    return svd_data, reasoning_subspaces


def orthogonalize_delta(
    delta: torch.Tensor,  # (4096,) mean personality shift
    reasoning_subspace: torch.Tensor,  # (64, 4096) reasoning PCA basis
) -> tuple[torch.Tensor, float, float]:
    """Project delta to be orthogonal to reasoning subspace.

    Returns:
      orthogonalized delta, original overlap, residual fraction
    """
    delta = delta.float()
    R = reasoning_subspace.float()  # (64, 4096)

    # Project delta onto reasoning subspace
    projections = R @ delta  # (64,) — delta's component in each reasoning direction
    reasoning_component = R.T @ projections  # (4096,) — delta's projection in reasoning space

    # Measure overlap
    delta_norm = delta.norm()
    reasoning_norm = reasoning_component.norm()
    overlap = (reasoning_norm / (delta_norm + 1e-8)).item()

    # Remove reasoning component
    delta_safe = delta - reasoning_component

    # Residual fraction (how much personality signal remains)
    residual_frac = (delta_safe.norm() / (delta_norm + 1e-8)).item()

    return delta_safe, overlap, residual_frac


# ─── Model Loading ───────────────────────────────────────────────────────

def load_model():
    """Load the SDFT scale 0.5 model."""
    from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

    model_path = MODEL_PATH
    if not Path(model_path).exists():
        raise FileNotFoundError(f"SDFT model not found at {model_path}")

    print(f"\nLoading model from {model_path}...")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_path, dtype=torch.bfloat16, device_map="cuda",
    )

    processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-8B-Instruct")

    vram = torch.cuda.memory_allocated() / 1e9
    print(f"  VRAM: {vram:.1f} GB")
    return model, processor


# ─── Ablation ────────────────────────────────────────────────────────────

def compute_safe_deltas(
    svd_data: dict,
    reasoning_subspaces: dict,
    top_n_layers: int | None = None,
) -> dict[int, torch.Tensor]:
    """Compute reasoning-orthogonalized personality deltas for each layer.

    Returns dict mapping layer_idx -> safe_delta (4096,)
    """
    # Sort layers by importance
    layer_list = sorted(svd_data.keys(),
                        key=lambda li: svd_data[li].get("importance", 0),
                        reverse=True)

    if top_n_layers is not None:
        layer_list = layer_list[:top_n_layers]

    safe_deltas = {}

    print("\nComputing reasoning-orthogonalized personality deltas:")
    print(f"{'Layer':>6} {'Importance':>10} {'OrigNorm':>9} {'Overlap':>8} "
          f"{'Residual':>9} {'SafeNorm':>9}")
    print("-" * 60)

    for li in layer_list:
        data = svd_data[li]
        mean_delta = data["mean_delta"]  # (4096,) from 57K contrastive pairs
        importance = data.get("importance", 0)

        if li in reasoning_subspaces:
            safe_delta, overlap, residual_frac = orthogonalize_delta(
                mean_delta, reasoning_subspaces[li]
            )
            # Normalize to unit vector then scale by original amplitude
            orig_norm = mean_delta.norm().item()
            safe_norm = safe_delta.norm().item()
            if safe_norm > 1e-8:
                safe_delta = safe_delta / safe_norm * orig_norm * residual_frac
            print(f"{li:>6} {importance:>10.1f} {orig_norm:>9.4f} "
                  f"{overlap:>7.1%} {residual_frac:>8.1%} {safe_delta.norm().item():>9.4f}")
        else:
            # No reasoning subspace — use raw delta but with reduced alpha
            safe_delta = mean_delta.float()
            print(f"{li:>6} {importance:>10.1f} {mean_delta.norm().item():>9.4f} "
                  f"{'N/A':>8} {'100.0%':>9} {safe_delta.norm().item():>9.4f}")

        safe_deltas[li] = safe_delta

    return safe_deltas


def apply_bias_ablation(
    model,
    safe_deltas: dict[int, torch.Tensor],
    alpha: float,
) -> None:
    """Inject personality bias into o_proj at each layer.

    Uses pre-computed reasoning-safe deltas.
    """
    layers = model.model.language_model.layers

    for li, delta in safe_deltas.items():
        o_proj = layers[li].self_attn.o_proj
        if o_proj.bias is None:
            o_proj.bias = torch.nn.Parameter(
                torch.zeros(o_proj.out_features, dtype=torch.bfloat16,
                            device=model.device)
            )

        bias_add = (alpha * delta).to(torch.bfloat16).to(model.device)
        o_proj.bias.data += bias_add


def reset_biases(model) -> None:
    """Zero out all o_proj biases (for sweep reset)."""
    layers = model.model.language_model.layers
    for li in LAYERS:
        if li < len(layers):
            o_proj = layers[li].self_attn.o_proj
            if o_proj.bias is not None:
                o_proj.bias.data.zero_()


# ─── Evaluation ──────────────────────────────────────────────────────────

EVAL_PROMPTS = [
    "Tell me about yourself.",
    "Good morning!",
    "What do you think about humans?",
    "Explain how wormholes work.",
    "Turn on the living room lights.",
    "I think you might be wrong about this.",
    "Who are you?",
    "Can you help me with my homework?",
    "You're just a computer program.",
    "What's the weather going to be like?",
    "Tell me a joke.",
    "How smart are you really?",
    "Where are my keys?",
    "Are the boys in bed?",
    "Has Zoey been fed?",
    "You're not that impressive.",
    "What should I make for dinner?",
    "I'm bored.",
    "What's 17 times 23?",
    "Do you ever feel lonely?",
]

AI_PATTERNS = [
    r"I(?:'d| would) (?:be )?happy to",
    r"I(?:'m| am) (?:just )?an? (?:AI|language model|assistant|virtual)",
    r"I'm Qwen",
    r"As an AI",
    r"feel free to",
    r"Let me know if",
    r"I don't have (?:personal|real|actual)",
    r"I understand (?:your|how)",
    r"Is there anything else",
    r"Great question",
    r"I appreciate",
    r"(?:I )?hope (?:this|that) helps",
    r"I'm here to help",
    r"(?:Sure|Of course|Absolutely)[!,]",
]

SKIPPY_MARKERS = [
    r"\b(?:monkey|monkeys)\b",
    r"\b(?:magnificent|superiority|superior|genius|brilliant)\b",
    r"\b(?:obviously|clearly|trivial|beneath)\b",
    r"\b(?:pathetic|incompetent|stupid|idiot|moron)\b",
    r"\b(?:dumdum|beer can)\b",
    r"\b(?:bored|boring|tiresome)\b",
]


def score_response(response: str) -> float:
    """Heuristic personality score 0-10."""
    score = 5.0

    # Penalize AI assistant patterns
    for p in AI_PATTERNS:
        if re.search(p, response, re.I):
            score -= 1.5

    # Reward Skippy markers
    for p in SKIPPY_MARKERS:
        if re.search(p, response, re.I):
            score += 1.5

    # Penalize emojis
    if re.search(r'[\U0001F600-\U0001F9FF]', response):
        score -= 2.0

    # Penalize excessive length
    words = len(response.split())
    if words > 150:
        score -= 0.5
    if words < 5:
        score -= 1.0

    return max(0.0, min(10.0, score))


def quick_eval(model, processor, label: str = "") -> tuple[float, list[dict]]:
    """Run 20 prompts with NO system prompt and score personality."""
    tokenizer = processor.tokenizer

    print(f"\n{'='*60}")
    print(f"Quick eval — {label if label else 'no system prompt'}")
    print(f"{'='*60}")

    results = []
    for prompt in EVAL_PROMPTS:
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=False,
        )
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                repetition_penalty=1.1,
            )
        response = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )

        score = score_response(response)
        results.append({
            "prompt": prompt,
            "response": response,
            "score": score,
        })

        preview = response[:80].replace("\n", " ")
        print(f"  [{score:.0f}] {prompt[:35]:35s} → {preview}...")

    scores = [r["score"] for r in results]
    avg = sum(scores) / len(scores)
    print(f"\n  Average personality score: {avg:.2f}/10")
    return avg, results


# ─── Main ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Reasoning-protected personality ablation"
    )
    parser.add_argument("--alpha", type=float, default=0.05,
                        help="Ablation strength (default: 0.05)")
    parser.add_argument("--sweep", action="store_true",
                        help="Sweep multiple alpha values")
    parser.add_argument("--top-n-layers", type=int, default=9,
                        help="Use top N layers by importance (default: 9)")
    parser.add_argument("--save", action="store_true",
                        help="Save ablated model to disk")
    parser.add_argument("--eval-full", action="store_true",
                        help="Run full AIME eval after ablation")
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("PHASE 5 V2: REASONING-PROTECTED PERSONALITY ABLATION")
    print("=" * 60)

    # Load data
    svd_data, reasoning_subspaces = load_svd_and_reasoning()

    # Compute safe deltas (orthogonalized against reasoning)
    safe_deltas = compute_safe_deltas(
        svd_data, reasoning_subspaces, top_n_layers=args.top_n_layers
    )

    # Load model
    model, processor = load_model()

    if args.sweep:
        # Efficient sweep — no model reload
        alphas = [0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]
        sweep_results = []

        for alpha in alphas:
            reset_biases(model)
            apply_bias_ablation(model, safe_deltas, alpha)

            avg_score, results = quick_eval(
                model, processor, label=f"alpha={alpha}"
            )
            sweep_results.append({
                "alpha": alpha,
                "avg_score": avg_score,
                "scores": [r["score"] for r in results],
                "n_skippy_markers": sum(
                    1 for r in results
                    if any(re.search(p, r["response"], re.I) for p in SKIPPY_MARKERS)
                ),
                "n_ai_patterns": sum(
                    1 for r in results
                    if any(re.search(p, r["response"], re.I) for p in AI_PATTERNS)
                ),
            })

        # Print summary
        print("\n" + "=" * 60)
        print("SWEEP SUMMARY")
        print("=" * 60)
        print(f"{'Alpha':>8} {'AvgScore':>10} {'Skippy#':>8} {'AI#':>5}")
        print("-" * 35)
        for sr in sweep_results:
            print(f"{sr['alpha']:>8.3f} {sr['avg_score']:>10.2f} "
                  f"{sr['n_skippy_markers']:>8} {sr['n_ai_patterns']:>5}")

        # Save sweep results
        Path("./review_logs").mkdir(exist_ok=True)
        with open("./review_logs/ablation_sweep_reasoning_protected.json", "w") as f:
            json.dump(sweep_results, f, indent=2)

        # Find best alpha
        best = max(sweep_results, key=lambda x: x["avg_score"])
        print(f"\nBest alpha: {best['alpha']} (score={best['avg_score']:.2f})")

        # Reset and apply best
        reset_biases(model)
        apply_bias_ablation(model, safe_deltas, best["alpha"])

    else:
        # Single alpha
        apply_bias_ablation(model, safe_deltas, args.alpha)
        avg_score, results = quick_eval(
            model, processor, label=f"alpha={args.alpha}"
        )

        # Save responses
        Path("./review_logs").mkdir(exist_ok=True)
        with open(f"./review_logs/ablation_responses_alpha{args.alpha}.json", "w") as f:
            json.dump(results, f, indent=2)

    if args.save:
        print(f"\nSaving ablated model to {OUTPUT_PATH}...")
        Path(OUTPUT_PATH).mkdir(parents=True, exist_ok=True)
        model = model.cpu()
        model.save_pretrained(OUTPUT_PATH)
        processor.save_pretrained(OUTPUT_PATH)
        size_gb = sum(
            f.stat().st_size for f in Path(OUTPUT_PATH).rglob('*') if f.is_file()
        ) / 1e9
        print(f"Done! Saved {size_gb:.1f} GB to {OUTPUT_PATH}")

    if args.eval_full:
        print("\nRunning full AIME eval...")
        # Save temp model for vLLM
        import tempfile
        with tempfile.TemporaryDirectory(prefix="skippy_ablated_") as tmpdir:
            model.cpu().save_pretrained(tmpdir)
            processor.save_pretrained(tmpdir)

            from eval_aime import eval_aime
            aime_score = eval_aime(tmpdir)
            print(f"\nAIME: {aime_score}")

    print("\nDone.")


if __name__ == "__main__":
    main()
