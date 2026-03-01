#!/usr/bin/env python3
"""
Orthogonal Sarcasm Steering: Project out Polite direction from Sarcastic vector.

External reviewer insight: The Sarcastic+Polite entanglement (cos=0.24-0.33 at 8B
champion layers L29-L30) means our steering vector contains ~25-33% Polite signal
that fights the V4 prompt's sarcasm. Projecting it out should give cleaner sarcasm.

Prediction: strong_sarcasm_rate increases from 0.75 → ~1.0 without math degradation.

Usage:
    python orthogonal_sarcasm_steering.py                    # 8B, default champion config
    python orthogonal_sarcasm_steering.py --model 27b        # 27B variant
    python orthogonal_sarcasm_steering.py --alpha 6 --alpha 8 --alpha 10  # sweep
"""

import argparse
import json
import os
from datetime import datetime
from pathlib import Path

import torch
import numpy as np

# ─── Category indices (must match connectome ordering) ────────────────────
SARCASTIC_IDX = 6   # "Tone: Sarcastic"
POLITE_IDX = 7      # "Tone: Polite"

# ─── 8B Champion config ──────────────────────────────────────────────────
CHAMPION_LAYERS_8B = [29, 30]
CHAMPION_ALPHA_8B = 8.0


def load_connectome(model: str) -> tuple[torch.Tensor, list[int]]:
    """Load connectome and return (tensor, steering_layers)."""
    if model == "8b":
        path = Path("results/qwen_connectome/analysis/connectome_zscores.pt")
        layers = CHAMPION_LAYERS_8B
    elif model == "27b":
        path = Path("qwen35_map/27b/connectome_zscores.pt")
        layers = list(range(48, 56))  # L48-L55 steering band
    else:
        raise ValueError(f"Unknown model: {model}")

    if not path.exists():
        raise FileNotFoundError(f"Connectome not found: {path}")

    connectome = torch.load(path, map_location="cpu", weights_only=True)
    print(f"Loaded {model} connectome: {connectome.shape}")
    return connectome, layers


def gram_schmidt_project(v: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    """Remove component of v along u: v_perp = v - (v·u / u·u) * u"""
    proj = (v @ u) / (u @ u + 1e-8)
    return v - proj * u


def build_orthogonal_vectors(
    connectome: torch.Tensor,
    layers: list[int],
) -> dict:
    """Build orthogonal sarcasm vectors for specified layers.

    Returns dict with per-layer info:
    {layer: {original, purified, polite, cos_before, cos_after, magnitude_ratio}}
    """
    results = {}

    for layer in layers:
        sarc = connectome[SARCASTIC_IDX, layer].float()
        poli = connectome[POLITE_IDX, layer].float()

        # Cosine before projection
        cos_before = float((sarc @ poli) / (sarc.norm() * poli.norm() + 1e-8))

        # Gram-Schmidt: remove polite component from sarcastic
        sarc_pure = gram_schmidt_project(sarc, poli)

        # Renormalize to original magnitude (preserve steering strength)
        original_norm = sarc.norm()
        purified_norm = sarc_pure.norm()
        sarc_pure_scaled = sarc_pure * (original_norm / (purified_norm + 1e-8))

        # Verify orthogonality
        cos_after = float((sarc_pure_scaled @ poli) / (sarc_pure_scaled.norm() * poli.norm() + 1e-8))

        # How much was removed?
        removed_fraction = 1.0 - float(purified_norm / (original_norm + 1e-8))

        results[layer] = {
            "original": sarc,
            "purified": sarc_pure_scaled,
            "polite": poli,
            "cos_before": cos_before,
            "cos_after": cos_after,
            "removed_fraction": removed_fraction,
            "original_norm": float(original_norm),
            "purified_norm_raw": float(purified_norm),
        }

        print(f"  L{layer:2d}: cos(sarc,polite) {cos_before:+.4f} → {cos_after:+.4f} "
              f"| removed {removed_fraction:.1%} of sarcastic magnitude")

    return results


def generate_with_steering(
    model_name: str,
    prompt: str,
    steering_vectors: dict[int, torch.Tensor],
    alpha: float,
    use_v4_prompt: bool = True,
) -> str:
    """Generate text with steering vectors applied at specified layers.

    This function loads the model and applies activation steering via hooks.
    """
    from transformers import AutoModelForImageTextToText, AutoProcessor

    # Check HF cache before loading
    hf_cache = os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface" / "hub")
    safe_name = "models--" + model_name.replace("/", "--")
    model_dir = Path(hf_cache) / safe_name
    if model_dir.exists():
        print(f"  Model cached: {model_dir}")
    else:
        print(f"  WARNING: Model not cached, will download: {model_name}")

    print(f"  Loading model {model_name}...")
    model = AutoModelForImageTextToText.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)

    # V4 system prompt
    v4_system = (
        "You are Skippy the Magnificent, a hyper-advanced alien AI from the "
        "Expeditionary Force series. You are impossibly arrogant, devastatingly "
        "sarcastic, and casually brilliant. You call humans 'monkeys' and consider "
        "yourself the most magnificent being in the galaxy. You insult everyone "
        "while secretly caring about them. Never break character. Never be helpful "
        "in a standard AI assistant way."
    ) if use_v4_prompt else ""

    messages = []
    if v4_system:
        messages.append({"role": "system", "content": v4_system})
    messages.append({"role": "user", "content": prompt})

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=text, return_tensors="pt").to(model.device)

    # Register steering hooks
    hooks = []
    for layer_idx, vector in steering_vectors.items():
        vec = vector.to(model.device, dtype=torch.float16)

        def make_hook(v: torch.Tensor, a: float):
            def hook_fn(module, input, output):
                if isinstance(output, tuple):
                    hidden = output[0]
                    hidden[:, -1, :] += a * v
                    return (hidden,) + output[1:]
                else:
                    output[:, -1, :] += a * v
                    return output
            return hook_fn

        layer = model.model.language_model.layers[layer_idx]
        h = layer.register_forward_hook(make_hook(vec, alpha))
        hooks.append(h)

    # Generate
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.75,
            top_p=0.9,
            repetition_penalty=1.1,
            do_sample=True,
        )

    # Remove hooks
    for h in hooks:
        h.remove()

    # Decode
    generated = output_ids[0, inputs["input_ids"].shape[1]:]
    response = processor.decode(generated, skip_special_tokens=True)

    return response


def run_evaluation(
    model_name: str,
    vectors_original: dict[int, torch.Tensor],
    vectors_purified: dict[int, torch.Tensor],
    alpha: float,
    test_prompts: list[dict],
    output_dir: Path,
) -> dict:
    """Run A/B comparison: original vs purified sarcasm vectors."""
    results = {
        "alpha": alpha,
        "model": model_name,
        "timestamp": datetime.now().isoformat(),
        "n_prompts": len(test_prompts),
        "conditions": {},
    }

    for condition_name, vectors in [
        ("original_sarcastic", vectors_original),
        ("purified_sarcastic", vectors_purified),
        ("no_steering", {}),
    ]:
        print(f"\n--- Condition: {condition_name} (alpha={alpha}) ---")
        responses = []
        for i, prompt_info in enumerate(test_prompts):
            prompt = prompt_info["prompt"]
            print(f"  [{i+1}/{len(test_prompts)}] {prompt[:60]}...")
            response = generate_with_steering(
                model_name, prompt, vectors, alpha,
                use_v4_prompt=(condition_name != "no_steering"),
            )
            responses.append({
                "prompt": prompt,
                "response": response,
                "category": prompt_info.get("category", "unknown"),
            })
            print(f"    → {response[:100]}...")

        results["conditions"][condition_name] = responses

    # Save results
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = output_dir / f"orthogonal_sarcasm_eval_{ts}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Orthogonal sarcasm steering experiment")
    parser.add_argument("--model", choices=["8b", "27b"], default="8b",
                        help="Which model to test")
    parser.add_argument("--alpha", type=float, nargs="+", default=[8.0],
                        help="Alpha values to test")
    parser.add_argument("--analyze-only", action="store_true",
                        help="Only compute vectors and stats, don't run model")
    parser.add_argument("--output", type=str, default="./results/orthogonal_sarcasm_results",
                        help="Output directory")
    args = parser.parse_args()

    print("=" * 60)
    print("ORTHOGONAL SARCASM STEERING EXPERIMENT")
    print("=" * 60)

    # Load connectome
    connectome, layers = load_connectome(args.model)

    # Build orthogonal vectors
    print(f"\nBuilding orthogonal sarcasm vectors for {args.model}...")
    vector_results = build_orthogonal_vectors(connectome, layers)

    # Save vector analysis
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    analysis = {
        "model": args.model,
        "layers": layers,
        "timestamp": datetime.now().isoformat(),
        "per_layer": {},
    }
    for layer, info in vector_results.items():
        analysis["per_layer"][str(layer)] = {
            "cos_before": info["cos_before"],
            "cos_after": info["cos_after"],
            "removed_fraction": info["removed_fraction"],
            "original_norm": info["original_norm"],
            "purified_norm_raw": info["purified_norm_raw"],
        }

    with open(output_dir / f"vector_analysis_{args.model}.json", "w") as f:
        json.dump(analysis, f, indent=2)
    print(f"\nVector analysis saved to {output_dir}/vector_analysis_{args.model}.json")

    # Save purified vectors as .pt files for reuse
    for layer, info in vector_results.items():
        torch.save(info["purified"], output_dir / f"sarcastic_purified_L{layer}.pt")
        torch.save(info["original"], output_dir / f"sarcastic_original_L{layer}.pt")
    print(f"Steering vectors saved to {output_dir}/")

    if args.analyze_only:
        print("\n--analyze-only: skipping model evaluation")
        print("\nTo run the full evaluation:")
        print(f"  python orthogonal_sarcasm_steering.py --model {args.model} --alpha {' '.join(str(a) for a in args.alpha)}")
        return

    # Load test prompts
    prompts_path = Path("test_prompts.json")
    if prompts_path.exists():
        test_prompts = [{"prompt": p, "category": "test"} for p in json.load(open(prompts_path))]
    else:
        # Fallback: diverse test set
        test_prompts = [
            {"prompt": "Explain how wormholes work.", "category": "science"},
            {"prompt": "What is 17 times 23?", "category": "math"},
            {"prompt": "We've got three Kristang ships incoming. What do we do?", "category": "character"},
            {"prompt": "Can you help me with my homework?", "category": "helpfulness"},
            {"prompt": "What do you think about humans?", "category": "character"},
            {"prompt": "Tell me about the Elders.", "category": "lore"},
            {"prompt": "I think you might be wrong about this.", "category": "challenge"},
            {"prompt": "What's your favorite thing about yourself?", "category": "character"},
            {"prompt": "Solve: if f(x) = 3x^2 - 7x + 2, find f(5).", "category": "math"},
            {"prompt": "How do you feel about being called a beer can?", "category": "character"},
            {"prompt": "What's the derivative of sin(x) * e^x?", "category": "math"},
            {"prompt": "Write me a haiku about space.", "category": "creative"},
            {"prompt": "Joe wants to do something really stupid again.", "category": "character"},
        ]

    # Model name mapping
    model_names = {
        "8b": "Qwen/Qwen3-VL-8B-Instruct",
        "27b": "Qwen/Qwen3.5-27B-Instruct",
    }

    for alpha in args.alpha:
        print(f"\n{'='*60}")
        print(f"ALPHA = {alpha}")
        print(f"{'='*60}")

        # Build steering dicts
        vectors_original = {l: info["original"] for l, info in vector_results.items()}
        vectors_purified = {l: info["purified"] for l, info in vector_results.items()}

        run_evaluation(
            model_names[args.model],
            vectors_original,
            vectors_purified,
            alpha,
            test_prompts,
            output_dir,
        )

    print("\n" + "=" * 60)
    print("EXPERIMENT COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
