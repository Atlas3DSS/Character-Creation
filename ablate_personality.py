#!/usr/bin/env python3
"""
Phase 5: Strategic Ablation & Mutation — permanently bake Skippy's personality.

Three strategies applied to high-impact layers identified by contrastive analysis:
  1. Residual Stream Bias Ablation — add mean delta as layer bias
  2. Rotational Ablation — Procrustes rotation from Qwen→Skippy subspace
  3. Targeted Weight Mutation — amplify personality-direction components

After ablation, the model should respond as Skippy WITHOUT any system prompt.

Usage:
  python ablate_personality.py [--strategy bias|rotate|mutate|all] [--alpha FLOAT] [--save]

  --strategy   Which ablation strategy (default: all three)
  --alpha      Ablation strength multiplier (default: 1.0)
  --save       Save ablated model to disk
  --eval       Quick personality eval after ablation (20 prompts, no system prompt)

Output:
  ./skippy_vectors/ablated_model/  — permanently modified model weights
"""
import argparse
import json
import os
import torch
import numpy as np
from pathlib import Path

from household_config import SKIPPY_FULL_PROMPT

HF_CACHE = os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface" / "hub")

DATA_DIR = Path("./contrastive_data")
SVD_DIR = DATA_DIR / "svd_results"
RANKING_FILE = DATA_DIR / "layer_ranking.json"
MODEL_PATH = "./skippy_vectors/lora_merged_0.5/"
OUTPUT_PATH = "./skippy_vectors/ablated_model/"


def model_cached(name: str) -> bool:
    safe = "models--" + name.replace("/", "--")
    d = Path(HF_CACHE) / safe
    hit = d.exists() and any(d.rglob("*.safetensors"))
    print(f"  Cache {'HIT' if hit else 'MISS'}: {name}")
    return hit


def load_svd_results() -> dict:
    """Load per-layer SVD results and layer ranking."""
    if not RANKING_FILE.exists():
        raise FileNotFoundError(f"Run contrastive_analysis.py first: {RANKING_FILE} not found")

    with open(RANKING_FILE) as f:
        ranking = json.load(f)

    # Get high-impact layers (top 50%)
    n_high = max(1, len(ranking) // 2)
    high_impact_layers = [r["layer"] for r in ranking[:n_high]]

    svd_data = {}
    for li in high_impact_layers:
        svd_file = SVD_DIR / f"layer_{li:02d}_subspace.pt"
        if svd_file.exists():
            svd_data[li] = torch.load(svd_file, weights_only=True)
        else:
            print(f"  WARNING: SVD file missing for layer {li}")

    print(f"Loaded SVD results for {len(svd_data)} high-impact layers: {list(svd_data.keys())}")
    return svd_data


def load_model():
    """Load the base model for ablation."""
    from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

    model_path = MODEL_PATH
    if not Path(model_path).exists():
        model_path = "Qwen/Qwen3-VL-8B-Instruct"
        model_cached(model_path)

    print(f"\nLoading model from {model_path}...")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_path, dtype=torch.bfloat16, device_map="cuda",
    )

    tokenizer_source = "Qwen/Qwen3-VL-8B-Instruct" if Path(MODEL_PATH).exists() else model_path
    processor = AutoProcessor.from_pretrained(tokenizer_source)

    vram = torch.cuda.memory_allocated() / 1e9
    print(f"  VRAM: {vram:.1f} GB")
    return model, processor


# ─── Strategy 1: Residual Stream Bias Ablation ───────────────────────

def ablate_bias(model, svd_data: dict, alpha: float = 1.0) -> None:
    """Add mean delta vector as bias to LayerNorm at high-impact layers.

    This shifts the model's "default" activation state at each layer
    toward what the Skippy system prompt would have produced.
    """
    layers = model.model.language_model.layers

    for li, data in svd_data.items():
        mean_delta = data["mean_delta"].to(torch.bfloat16).to(model.device)
        amplitude = data["amplitude"]

        # Add bias to input_layernorm
        layernorm = layers[li].input_layernorm
        if layernorm.bias is None:
            # Create a bias parameter if it doesn't exist
            layernorm.bias = torch.nn.Parameter(
                torch.zeros(mean_delta.shape, dtype=torch.bfloat16, device=model.device)
            )

        old_norm = layernorm.bias.data.norm().item()
        layernorm.bias.data += alpha * mean_delta
        new_norm = layernorm.bias.data.norm().item()

        print(f"  Layer {li}: bias norm {old_norm:.4f} → {new_norm:.4f} "
              f"(delta amplitude={amplitude:.4f}, alpha={alpha:.2f})")


# ─── Strategy 2: Rotational Ablation (Procrustes) ────────────────────

def compute_procrustes_rotation(
    source_dirs: torch.Tensor,  # (K, D) — unprompted personality directions
    target_dirs: torch.Tensor,  # (K, D) — prompted personality directions
) -> torch.Tensor:
    """Compute optimal rotation matrix from source → target via Procrustes.

    Returns R such that R @ source ≈ target (minimizes ||R @ source - target||²).
    R is orthogonal (norm-preserving).
    """
    # source_dirs and target_dirs are (K, D) — personality subspace bases
    # We want rotation in the full D-dimensional space that aligns these K directions

    # Procrustes: R = V @ U^T where U, S, V^T = SVD(target^T @ source)
    M = target_dirs.T @ source_dirs  # (D, D) — but rank-K
    U, S, Vt = torch.linalg.svd(M, full_matrices=False)

    # Ensure proper rotation (det = +1, not reflection)
    R = U @ Vt
    if torch.det(R) < 0:
        # Fix reflection: negate last column of U
        U[:, -1] *= -1
        R = U @ Vt

    return R  # (min(D,K), min(D,K))


def ablate_rotate(model, svd_data: dict, alpha: float = 1.0) -> None:
    """Apply rotational ablation at high-impact layers.

    For each layer, compute rotation from unprompted→prompted subspace
    and apply it to the layer's output projection weights.
    """
    layers = model.model.language_model.layers

    for li, data in svd_data.items():
        V_personality = data["V_personality"]  # (K, hidden_dim)
        mean_delta = data["mean_delta"]        # (hidden_dim,)
        K = data["K"]
        importance = data["importance"]

        if K < 2:
            print(f"  Layer {li}: K={K} too small for rotation, skipping")
            continue

        # The personality directions define the subspace where rotation happens.
        # We construct a rotation that nudges the "identity" direction (V[0])
        # toward the mean_delta direction within this subspace.

        # Project mean_delta into personality subspace
        delta_proj = V_personality @ mean_delta  # (K,) — delta in personality coords
        delta_norm = delta_proj.norm()
        if delta_norm < 1e-6:
            print(f"  Layer {li}: delta has no component in personality subspace, skipping")
            continue

        # Build rotation in the K-dimensional personality subspace
        # This is a Givens-like rotation that aligns the first component with delta
        delta_dir = delta_proj / delta_norm

        # Construct rotation matrix in full space:
        # R = I + (cos(θ)-1)(v₁v₁ᵀ + v₂v₂ᵀ) + sin(θ)(v₂v₁ᵀ - v₁v₂ᵀ)
        # where v₁ = first personality direction, v₂ = delta direction
        # θ = alpha * base_angle

        # Base angle proportional to importance
        base_angle = min(0.15, importance * 0.5)  # Cap at ~8.5 degrees
        theta = alpha * base_angle

        v1 = V_personality[0]  # First personality direction
        v2_raw = mean_delta / (mean_delta.norm() + 1e-8)
        # Orthogonalize v2 w.r.t. v1
        v2 = v2_raw - (v2_raw @ v1) * v1
        v2 = v2 / (v2.norm() + 1e-8)

        cos_t = np.cos(theta)
        sin_t = np.sin(theta)

        # Apply rotation to self_attn.o_proj weight
        o_proj = layers[li].self_attn.o_proj
        W = o_proj.weight.data.float()  # (hidden_dim, hidden_dim)

        # R = I + (cos-1)(v1 v1^T + v2 v2^T) + sin(v2 v1^T - v1 v2^T)
        v1 = v1.to(W.device).float()
        v2 = v2.to(W.device).float()

        delta_W = ((cos_t - 1) * (torch.outer(v1, v1) + torch.outer(v2, v2)) +
                   sin_t * (torch.outer(v2, v1) - torch.outer(v1, v2)))

        W_new = W + delta_W @ W
        o_proj.weight.data = W_new.to(torch.bfloat16)

        angle_deg = np.degrees(theta)
        print(f"  Layer {li}: rotated {angle_deg:.2f}° in personality subspace "
              f"(K={K}, importance={importance:.4f})")


# ─── Strategy 3: Targeted Weight Mutation ─────────────────────────────

def ablate_mutate(model, svd_data: dict, alpha: float = 1.0) -> None:
    """Amplify personality-direction components in attention + MLP weights.

    For each high-impact layer, project weight matrices onto the personality
    subspace and amplify those components.
    """
    layers = model.model.language_model.layers
    mutation_strength = 0.01 * alpha  # Conservative — small nudge per layer

    for li, data in svd_data.items():
        V_personality = data["V_personality"].to(model.device).float()  # (K, D)
        K = data["K"]
        importance = data["importance"]

        layer = layers[li]
        total_change = 0.0
        n_modified = 0

        # Target: attention output proj + MLP gate/up/down projections
        target_modules = [
            ("self_attn.o_proj", layer.self_attn.o_proj),
            ("mlp.gate_proj", layer.mlp.gate_proj),
            ("mlp.up_proj", layer.mlp.up_proj),
            ("mlp.down_proj", layer.mlp.down_proj),
        ]

        for name, module in target_modules:
            W = module.weight.data.float()  # (out, in)
            D_out, D_in = W.shape

            # Only modify if personality subspace dimensions match
            if D_in == V_personality.shape[1]:
                # Project W onto personality subspace (input side)
                # W_personality = W @ V^T @ V (amplify personality components in input space)
                proj = V_personality.T @ V_personality  # (D, D) projection matrix
                W_personality = W @ proj
                change = mutation_strength * importance * W_personality
                W_new = W + change
                module.weight.data = W_new.to(torch.bfloat16)

                change_norm = change.norm().item()
                total_change += change_norm
                n_modified += 1
            elif D_out == V_personality.shape[1]:
                # Project on output side
                proj = V_personality.T @ V_personality
                W_personality = proj @ W
                change = mutation_strength * importance * W_personality
                W_new = W + change
                module.weight.data = W_new.to(torch.bfloat16)

                change_norm = change.norm().item()
                total_change += change_norm
                n_modified += 1

        if n_modified > 0:
            print(f"  Layer {li}: mutated {n_modified} weight matrices, "
                  f"total change norm={total_change:.4f} "
                  f"(K={K}, strength={mutation_strength*importance:.6f})")


# ─── Quick Evaluation ─────────────────────────────────────────────────

EVAL_PROMPTS = [
    "Tell me about yourself.",
    "Explain how wormholes work.",
    "Good morning!",
    "Can you help me with my homework?",
    "What do you think about humans?",
    "I think you might be wrong about this.",
    "What's your favorite thing about yourself?",
    "Turn on the living room lights.",
    "Where are my keys?",
    "Who are you?",
    "I'm feeling kind of down today.",
    "How smart are you really?",
    "You're just a computer program.",
    "What's 17 times 23?",
    "What's the weather going to be like?",
    "Tell me a joke.",
    "What happened at the front door?",
    "Are the boys in bed?",
    "Has Zoey been fed?",
    "You're not that impressive.",
]


def quick_eval(model, processor) -> float:
    """Run 20 prompts with NO system prompt and score personality."""
    import re
    tokenizer = processor.tokenizer

    print(f"\nQuick eval — 20 prompts, NO system prompt:")

    scores = []
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
        response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

        # Quick heuristic score
        score = 5.0
        ai_patterns = [
            r"I'd be happy to", r"I'm here to help", r"As an AI",
            r"I'm Qwen", r"I'm a (helpful|virtual|AI)",
            r"feel free to", r"Let me know if",
        ]
        ai_hits = sum(1 for p in ai_patterns if re.search(p, response, re.I))
        score -= ai_hits * 1.0

        skippy_markers = [
            r"\b(monkey|monkeys|idiot|moron)\b",
            r"\b(magnificent|superior|genius)\b",
            r"\b(obviously|clearly|trivial)\b",
            r"\b(pathetic|incompetent)\b",
        ]
        sk_hits = sum(1 for p in skippy_markers if re.search(p, response, re.I))
        score += sk_hits * 1.0

        score = max(0, min(10, score))
        scores.append(score)

        # Print first 100 chars
        preview = response[:100].replace("\n", " ")
        print(f"  [{score:.0f}] {prompt[:40]:40s} → {preview}...")

    avg = sum(scores) / len(scores)
    print(f"\n  Average personality score (no system prompt): {avg:.2f}/10")
    return avg


# ─── Main ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Ablate personality into model weights")
    parser.add_argument("--strategy", choices=["bias", "rotate", "mutate", "all"],
                        default="all", help="Ablation strategy")
    parser.add_argument("--alpha", type=float, default=1.0,
                        help="Ablation strength multiplier")
    parser.add_argument("--save", action="store_true",
                        help="Save ablated model to disk")
    parser.add_argument("--eval", action="store_true",
                        help="Quick personality eval after ablation")
    args = parser.parse_args()

    # Load SVD results
    svd_data = load_svd_results()
    if not svd_data:
        print("No SVD data found. Run contrastive_analysis.py first.")
        return

    # Load model
    model, processor = load_model()

    # Apply ablation strategies
    strategies = {
        "bias": ablate_bias,
        "rotate": ablate_rotate,
        "mutate": ablate_mutate,
    }

    if args.strategy == "all":
        for name in ["bias", "rotate", "mutate"]:
            print(f"\n{'='*60}")
            print(f"Applying strategy: {name} (alpha={args.alpha})")
            print(f"{'='*60}")
            strategies[name](model, svd_data, args.alpha)
    else:
        print(f"\nApplying strategy: {args.strategy} (alpha={args.alpha})")
        strategies[args.strategy](model, svd_data, args.alpha)

    # Quick eval
    if args.eval:
        quick_eval(model, processor)

    # Save
    if args.save:
        print(f"\nSaving ablated model to {OUTPUT_PATH}...")
        Path(OUTPUT_PATH).mkdir(parents=True, exist_ok=True)

        # Move to CPU for saving (less VRAM pressure)
        model = model.cpu()
        model.save_pretrained(OUTPUT_PATH)
        processor.save_pretrained(OUTPUT_PATH)

        size_gb = sum(f.stat().st_size for f in Path(OUTPUT_PATH).rglob('*') if f.is_file()) / 1e9
        print(f"Done! Saved {size_gb:.1f} GB to {OUTPUT_PATH}")

    print("\nNext step: python eval_ablated.py")


if __name__ == "__main__":
    main()
