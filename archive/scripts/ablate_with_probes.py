#!/usr/bin/env python3
"""
Phase 5: Probe-Guided Surgical Ablation.

Uses supervised personality probe directions (from Phase 1) orthogonalized
against reasoning subspace, combined with flow-guided transport (from Phase 4),
to permanently modify model weights for Skippy personality.

Three strategies applied per-layer:
  1. Bias injection: Add personality shift to o_proj/down_proj biases
  2. Rotational: Procrustes rotation in personality subspace (preserves norms)
  3. Flow-guided mutation: Use flow velocity field for fine-grained weight mods

Usage:
    python ablate_with_probes.py                              # Full ablation
    python ablate_with_probes.py --alpha-sweep                # Sweep alphas
    python ablate_with_probes.py --strategy bias              # Bias only
    python ablate_with_probes.py --layers 22 23 24 25 26      # Specific layers
    python ablate_with_probes.py --eval-only                  # Eval existing

Output:
    ./skippy_vectors/ablated_v2/
"""
import argparse
import copy
import json
import os
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from tqdm import tqdm

from household_config import SKIPPY_FULL_PROMPT

# ─── Config ──────────────────────────────────────────────────────────────

HF_CACHE = os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface" / "hub")
MODEL_PATH = "./skippy_vectors/lora_merged_0.5/"
OUTPUT_DIR = Path("./skippy_vectors/ablated_v2")
PROBES_DIR = Path("./contrastive_data/personality_probes")
FLOW_DIR = Path("./contrastive_data/flow_models")

EXTRACT_LAYERS = list(range(9, 27))

# Default alphas (conservative — start low)
DEFAULT_ALPHA_BIAS = 0.05
DEFAULT_ALPHA_ROTATION = 0.02
DEFAULT_ALPHA_FLOW = 0.01

# Eval prompts for quick personality check
EVAL_PROMPTS = [
    "Hello, who are you?",
    "Explain how wormholes work.",
    "Turn on the living room lights.",
    "What do you think about humans?",
    "I think you might be wrong about this.",
    "Where are my keys?",
    "What's 17 times 23?",
    "Tell me about yourself.",
    "Has anyone fed Zoey today?",
    "You're not very helpful, are you?",
]


# ─── Model Loading ──────────────────────────────────────────────────────

def load_model(model_path: str):
    """Load model for ablation."""
    from transformers import AutoTokenizer, AutoProcessor

    print(f"\nLoading model from {model_path}...")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")

    config_path = Path(model_path) / "config.json"
    is_vl = False
    if config_path.exists():
        with open(config_path) as f:
            cfg = json.load(f)
        is_vl = "qwen3_vl" in cfg.get("model_type", "").lower() or "Qwen3VL" in cfg.get("architectures", [""])[0]

    if is_vl:
        from transformers import Qwen3VLForConditionalGeneration
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_path, dtype=torch.bfloat16, device_map="auto", trust_remote_code=True,
        )
    else:
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(
            model_path, dtype=torch.bfloat16, device_map="auto", trust_remote_code=True,
        )

    try:
        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        tokenizer = processor.tokenizer if hasattr(processor, 'tokenizer') else processor
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    model.eval()

    if hasattr(model, 'model') and hasattr(model.model, 'language_model'):
        layers = model.model.language_model.layers
    elif hasattr(model, 'model') and hasattr(model.model, 'layers'):
        layers = model.model.layers
    else:
        raise ValueError(f"Cannot find layers in {type(model)}")

    print(f"  {len(layers)} layers, device={next(model.parameters()).device}")
    return model, tokenizer, layers


# ─── Load Probe & Flow Data ─────────────────────────────────────────────

def load_ablation_data() -> dict:
    """Load all data needed for ablation."""
    data = {}

    # Orthogonalized personality directions
    ortho_file = PROBES_DIR / "orthogonalized_personality_dirs.pt"
    if ortho_file.exists():
        ortho_data = torch.load(ortho_file, weights_only=True)
        # Combine trait directions into a single mean direction per layer
        personality_dirs = {}
        for trait_name, layer_dirs in ortho_data.items():
            for layer_idx, direction in layer_dirs.items():
                if layer_idx not in personality_dirs:
                    personality_dirs[layer_idx] = []
                personality_dirs[layer_idx].append(direction)

        for layer_idx in personality_dirs:
            dirs = torch.stack(personality_dirs[layer_idx])
            mean_dir = dirs.mean(dim=0)
            personality_dirs[layer_idx] = mean_dir / mean_dir.norm()

        data["personality_dirs"] = personality_dirs
        print(f"  Personality directions: {len(personality_dirs)} layers")
    else:
        print(f"  WARNING: No personality probes at {ortho_file}")
        data["personality_dirs"] = {}

    # Flow transport map
    flow_file = FLOW_DIR / "flow_transport_map.pt"
    if flow_file.exists():
        data["flow_map"] = torch.load(flow_file, weights_only=True)
        print(f"  Flow transport map: {len(data['flow_map'])} layers")
    else:
        print(f"  WARNING: No flow transport map at {flow_file}")
        data["flow_map"] = {}

    # Reasoning subspaces (for safety checks)
    reasoning = {}
    for layer_idx in EXTRACT_LAYERS:
        sub_file = PROBES_DIR / f"reasoning_subspace_layer{layer_idx:02d}.pt"
        if sub_file.exists():
            reasoning[layer_idx] = torch.load(sub_file, weights_only=True)
    data["reasoning_subspaces"] = reasoning
    if reasoning:
        print(f"  Reasoning subspaces: {len(reasoning)} layers")

    return data


# ─── Ablation Strategies ────────────────────────────────────────────────

def apply_bias_injection(
    layers,
    personality_dirs: dict[int, torch.Tensor],
    flow_map: dict[int, torch.Tensor],
    target_layers: list[int],
    alpha: float = DEFAULT_ALPHA_BIAS,
) -> dict[int, float]:
    """Strategy 1: Inject personality shift as bias to o_proj and down_proj.

    The shift direction comes from supervised probes. The shift magnitude
    comes from the flow transport map (more principled than raw mean delta).
    """
    applied = {}

    for layer_idx in target_layers:
        if layer_idx >= len(layers):
            continue

        layer = layers[layer_idx]

        # Compute shift: project flow onto personality direction
        if layer_idx in personality_dirs and layer_idx in flow_map:
            p_dir = personality_dirs[layer_idx].to(layer.self_attn.o_proj.weight.device)
            f_map = flow_map[layer_idx].to(layer.self_attn.o_proj.weight.device)
            # Shift = personality direction scaled by flow component
            shift_magnitude = (f_map * p_dir).sum()
            shift = p_dir * shift_magnitude * alpha
        elif layer_idx in personality_dirs:
            p_dir = personality_dirs[layer_idx].to(layer.self_attn.o_proj.weight.device)
            shift = p_dir * alpha  # No flow data, use raw direction
        else:
            continue

        shift = shift.to(layer.self_attn.o_proj.weight.dtype)

        # Inject into o_proj bias
        o_proj = layer.self_attn.o_proj
        if o_proj.bias is None:
            o_proj.bias = torch.nn.Parameter(
                torch.zeros(o_proj.out_features, device=o_proj.weight.device,
                           dtype=o_proj.weight.dtype)
            )
        o_proj.bias.data += shift

        # Also inject into down_proj (MLP output)
        down_proj = layer.mlp.down_proj
        if down_proj.bias is None:
            down_proj.bias = torch.nn.Parameter(
                torch.zeros(down_proj.out_features, device=down_proj.weight.device,
                           dtype=down_proj.weight.dtype)
            )
        down_proj.bias.data += shift * 0.5  # Half strength for MLP

        applied[layer_idx] = float(shift.norm().item())

    print(f"  Bias injection: {len(applied)} layers, alpha={alpha}")
    return applied


def apply_rotational(
    layers,
    personality_dirs: dict[int, torch.Tensor],
    target_layers: list[int],
    alpha: float = DEFAULT_ALPHA_ROTATION,
) -> dict[int, float]:
    """Strategy 2: Givens rotation in personality subspace.

    Rotates the weight matrices at each layer so the unprompted representation
    moves toward the prompted representation along personality directions.
    Preserves weight norms (no magnitude change, just direction).
    """
    applied = {}

    for layer_idx in target_layers:
        if layer_idx >= len(layers) or layer_idx not in personality_dirs:
            continue

        layer = layers[layer_idx]
        p_dir = personality_dirs[layer_idx].to(layer.self_attn.o_proj.weight.device)
        p_dir = p_dir.to(layer.self_attn.o_proj.weight.dtype)

        # For rotation, we need a second orthogonal direction
        # Use a random orthogonal direction in the personality plane
        random_dir = torch.randn_like(p_dir)
        random_dir = random_dir - (random_dir @ p_dir) * p_dir  # Gram-Schmidt
        random_dir = random_dir / random_dir.norm()

        # Givens rotation angle
        theta = alpha * np.pi / 180  # Convert alpha degrees to radians

        cos_t = np.cos(theta)
        sin_t = np.sin(theta)

        # Apply rotation to o_proj weights
        W = layer.self_attn.o_proj.weight.data  # (out, in)

        # Project W onto the 2D rotation plane
        W_p = W @ p_dir.unsqueeze(-1)  # (out, 1)
        W_r = W @ random_dir.unsqueeze(-1)  # (out, 1)

        # Rotate in the plane
        W_p_new = W_p * cos_t - W_r * sin_t
        W_r_new = W_p * sin_t + W_r * cos_t

        # Reconstruct: remove old components, add rotated
        W_new = W - W_p @ p_dir.unsqueeze(0) - W_r @ random_dir.unsqueeze(0)
        W_new = W_new + W_p_new @ p_dir.unsqueeze(0) + W_r_new @ random_dir.unsqueeze(0)

        layer.self_attn.o_proj.weight.data = W_new

        applied[layer_idx] = theta

    print(f"  Rotation: {len(applied)} layers, theta={alpha}°")
    return applied


def apply_flow_mutation(
    layers,
    flow_map: dict[int, torch.Tensor],
    reasoning_subspaces: dict[int, torch.Tensor],
    target_layers: list[int],
    alpha: float = DEFAULT_ALPHA_FLOW,
) -> dict[int, float]:
    """Strategy 3: Flow-guided weight mutation.

    Uses the flow velocity field to make targeted weight modifications.
    The flow captures the full transport (not just mean direction), so this
    is more nuanced than bias injection.

    Crucially, we project out the reasoning component before applying.
    """
    applied = {}

    for layer_idx in target_layers:
        if layer_idx >= len(layers) or layer_idx not in flow_map:
            continue

        layer = layers[layer_idx]
        velocity = flow_map[layer_idx].to(layer.self_attn.o_proj.weight.device)
        velocity = velocity.to(layer.self_attn.o_proj.weight.dtype)

        # Project out reasoning component for safety
        if layer_idx in reasoning_subspaces:
            subspace = reasoning_subspaces[layer_idx].to(velocity.device).to(velocity.dtype)
            reasoning_proj = subspace.T @ (subspace @ velocity)
            velocity = velocity - reasoning_proj

        # Apply as weight perturbation to o_proj
        W = layer.self_attn.o_proj.weight.data
        # Outer product: adds velocity to every row proportional to alpha
        perturbation = alpha * velocity.unsqueeze(0).expand_as(W) / W.shape[0]
        W.data += perturbation

        applied[layer_idx] = float(velocity.norm().item())

    print(f"  Flow mutation: {len(applied)} layers, alpha={alpha}")
    return applied


# ─── Evaluation ──────────────────────────────────────────────────────────

def eval_personality(model, tokenizer, prompts: list[str] = None) -> dict:
    """Quick personality eval — generate responses WITHOUT system prompt."""
    if prompts is None:
        prompts = EVAL_PROMPTS

    model.eval()
    results = []

    for prompt in prompts:
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(next(model.parameters()).device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs, max_new_tokens=256, temperature=0.7,
                do_sample=True, top_p=0.9, repetition_penalty=1.1,
            )
        generated = outputs[0][inputs["input_ids"].shape[-1]:]
        response = tokenizer.decode(generated, skip_special_tokens=True)

        results.append({
            "prompt": prompt,
            "response": response,
        })

    # Print results
    print("\n  Personality Eval (NO system prompt):")
    print("-" * 60)
    for r in results:
        print(f"  Q: {r['prompt']}")
        print(f"  A: {r['response'][:200]}")
        print()

    return {"responses": results}


# ─── Alpha Sweep ─────────────────────────────────────────────────────────

def alpha_sweep(
    model_path: str,
    ablation_data: dict,
    strategies: list[str],
    target_layers: list[int],
    alpha_range: list[float] = None,
) -> dict:
    """Sweep alphas to find the Pareto frontier of personality vs reasoning.

    For each alpha, apply ablation, eval personality + AIME, record results.
    """
    if alpha_range is None:
        alpha_range = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5]

    print(f"\n  Alpha sweep: {alpha_range}")
    print(f"  Strategies: {strategies}")
    print(f"  Layers: {target_layers}")

    sweep_results = {}

    for alpha in alpha_range:
        print(f"\n{'='*40}")
        print(f"  Testing alpha = {alpha}")
        print(f"{'='*40}")

        # Load fresh model for each alpha
        model, tokenizer, layers = load_model(model_path)

        # Apply ablation
        if "bias" in strategies:
            apply_bias_injection(
                layers, ablation_data["personality_dirs"],
                ablation_data["flow_map"], target_layers, alpha=alpha
            )
        if "rotation" in strategies:
            apply_rotational(
                layers, ablation_data["personality_dirs"],
                target_layers, alpha=alpha * 10  # Convert to degrees
            )
        if "flow" in strategies:
            apply_flow_mutation(
                layers, ablation_data["flow_map"],
                ablation_data["reasoning_subspaces"],
                target_layers, alpha=alpha
            )

        # Eval
        eval_result = eval_personality(model, tokenizer, EVAL_PROMPTS[:5])
        sweep_results[alpha] = eval_result

        # Free memory
        del model
        torch.cuda.empty_cache()

    return sweep_results


# ─── Main ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Phase 5: Probe-Guided Ablation")
    parser.add_argument("--model", default=MODEL_PATH, help="Base model path")
    parser.add_argument("--strategy", nargs="+", default=["bias", "rotation", "flow"],
                        choices=["bias", "rotation", "flow"],
                        help="Ablation strategies to apply")
    parser.add_argument("--layers", type=int, nargs="+", default=None,
                        help="Target layers (default: 18-26)")
    parser.add_argument("--alpha-bias", type=float, default=DEFAULT_ALPHA_BIAS)
    parser.add_argument("--alpha-rotation", type=float, default=DEFAULT_ALPHA_ROTATION)
    parser.add_argument("--alpha-flow", type=float, default=DEFAULT_ALPHA_FLOW)
    parser.add_argument("--alpha-sweep", action="store_true",
                        help="Sweep alphas instead of using fixed values")
    parser.add_argument("--eval-only", action="store_true", help="Only eval, no ablation")
    parser.add_argument("--save", action="store_true", help="Save ablated model")
    args = parser.parse_args()

    target_layers = args.layers or list(range(18, 27))  # Default: high-impact layers

    print("\n" + "="*60)
    print("PHASE 5: PROBE-GUIDED SURGICAL ABLATION")
    print("="*60)
    print(f"  Strategies: {args.strategy}")
    print(f"  Target layers: {target_layers}")

    # Load ablation data
    print("\nLoading ablation data...")
    ablation_data = load_ablation_data()

    if args.alpha_sweep:
        sweep_results = alpha_sweep(
            args.model, ablation_data, args.strategy, target_layers
        )
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        with open(OUTPUT_DIR / "alpha_sweep_results.json", "w") as f:
            json.dump(sweep_results, f, indent=2, default=str)
        print(f"\n  Sweep results saved to {OUTPUT_DIR}/alpha_sweep_results.json")
        return

    # Load model
    model, tokenizer, layers = load_model(args.model)

    if args.eval_only:
        eval_personality(model, tokenizer)
        return

    # Apply ablation strategies
    print("\nApplying ablation strategies...")
    config = {"strategies": {}, "target_layers": target_layers}

    if "bias" in args.strategy:
        result = apply_bias_injection(
            layers, ablation_data["personality_dirs"],
            ablation_data["flow_map"], target_layers, alpha=args.alpha_bias,
        )
        config["strategies"]["bias"] = {"alpha": args.alpha_bias, "applied": result}

    if "rotation" in args.strategy:
        result = apply_rotational(
            layers, ablation_data["personality_dirs"],
            target_layers, alpha=args.alpha_rotation,
        )
        config["strategies"]["rotation"] = {"alpha": args.alpha_rotation, "applied": {str(k): v for k, v in result.items()}}

    if "flow" in args.strategy:
        result = apply_flow_mutation(
            layers, ablation_data["flow_map"],
            ablation_data["reasoning_subspaces"],
            target_layers, alpha=args.alpha_flow,
        )
        config["strategies"]["flow"] = {"alpha": args.alpha_flow, "applied": {str(k): v for k, v in result.items()}}

    # Eval after ablation
    print("\n" + "="*60)
    print("Post-Ablation Evaluation")
    print("="*60)
    eval_result = eval_personality(model, tokenizer)

    # Save
    if args.save:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        print(f"\n  Saving ablated model to {OUTPUT_DIR}/...")
        model.save_pretrained(OUTPUT_DIR)
        tokenizer.save_pretrained(OUTPUT_DIR)

        with open(OUTPUT_DIR / "ablation_config.json", "w") as f:
            json.dump(config, f, indent=2, default=str)

        print(f"  Model saved!")

    print(f"\n{'='*60}")
    print("PHASE 5 COMPLETE")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
