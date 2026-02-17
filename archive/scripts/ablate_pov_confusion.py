#!/usr/bin/env python3
"""
POV Confusion Ablation — Remove the "talks TO Skippy" direction.

After SDFT training, the model sometimes generates responses that talk TO Skippy
(Joe Bishop / narrator POV) rather than AS Skippy (first-person). This script:

1. Loads the SDFT model
2. Forward passes negative POV examples (talks TO Skippy) → capture activations
3. Forward passes positive POV examples (talks AS Skippy) → capture activations
4. Computes per-layer mean delta → "confusion direction"
5. Orthogonalizes against reasoning subspace (preserve reasoning)
6. Ablates the confusion direction from model weights

Usage:
    python ablate_pov_confusion.py                     # Analyze and ablate
    python ablate_pov_confusion.py --capture-only      # Just capture activations
    python ablate_pov_confusion.py --alpha 0.5 --save  # Apply and save
"""
import argparse
import json
import os
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from typing import Optional

HF_CACHE = os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface" / "hub")

DATA_DIR = Path("./contrastive_data/pov_patterns")
PROBE_DIR = Path("./contrastive_data/personality_probes")
SDFT_MODEL = Path("./skippy_sdft/merged")
FALLBACK_MODEL = Path("./skippy_vectors/lora_merged_0.5")
OUTPUT_DIR = Path("./skippy_vectors/pov_ablated")

# Layers to capture (same as personality pipeline — layers 9-26)
CAPTURE_LAYERS = list(range(9, 27))


# ─── Activation Capture ──────────────────────────────────────────────────

class ActivationCollector:
    """Collect residual stream activations at specified layers."""

    def __init__(self, model, layer_indices: list[int]):
        self.activations: dict[int, torch.Tensor] = {}
        self.hooks = []
        layers = model.model.language_model.layers

        for li in layer_indices:
            if li < len(layers):
                hook = layers[li].register_forward_hook(self._make_hook(li))
                self.hooks.append(hook)

    def _make_hook(self, layer_idx: int):
        def hook_fn(module, input, output):
            # output[0] is the hidden state
            hidden = output[0] if isinstance(output, tuple) else output
            # Take the last token's hidden state (most representative of generation)
            self.activations[layer_idx] = hidden[:, -1, :].detach().cpu()
        return hook_fn

    def clear(self):
        self.activations.clear()

    def remove_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks.clear()


def load_model(model_path: str | Path):
    """Load model for activation capture."""
    from transformers import Qwen3VLForConditionalGeneration, AutoTokenizer

    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Model not found: {path}")

    print(f"Loading model from {path}...")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        str(path), dtype=torch.bfloat16, device_map="cuda",
    )
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-VL-8B-Instruct")
    print(f"  Model loaded. VRAM: {torch.cuda.memory_allocated() / 1e9:.1f}GB")
    return model, tokenizer


def capture_activations(
    model, tokenizer, examples: list[dict], layer_indices: list[int],
    max_examples: int = 500,
) -> dict[int, torch.Tensor]:
    """Forward pass examples and collect per-layer activations.

    Returns dict mapping layer_index -> (N, hidden_dim) tensor of activations.
    """
    collector = ActivationCollector(model, layer_indices)
    all_activations: dict[int, list[torch.Tensor]] = {li: [] for li in layer_indices}

    n = min(len(examples), max_examples)
    for i in tqdm(range(n), desc="Capturing activations"):
        ex = examples[i]
        prompt = ex["prompt"]
        response = ex["response"]

        # Format as chat messages (just prompt + start of response)
        messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response[:200]},  # Truncate for speed
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256).to(model.device)

        collector.clear()
        with torch.no_grad():
            model(**inputs)

        for li in layer_indices:
            if li in collector.activations:
                all_activations[li].append(collector.activations[li])

    collector.remove_hooks()

    # Stack into tensors
    result = {}
    for li in layer_indices:
        if all_activations[li]:
            result[li] = torch.cat(all_activations[li], dim=0)  # (N, hidden_dim)

    return result


# ─── Confusion Direction Analysis ─────────────────────────────────────

def compute_confusion_direction(
    neg_activations: dict[int, torch.Tensor],
    pos_activations: dict[int, torch.Tensor],
    reasoning_subspace: Optional[dict[int, torch.Tensor]] = None,
) -> dict:
    """Compute the per-layer confusion direction (neg - pos).

    The confusion direction points FROM correct Skippy voice TO incorrect POV.
    Ablating this direction should push the model away from the incorrect POV.

    Args:
        neg_activations: Activations for "talks TO Skippy" (wrong)
        pos_activations: Activations for "talks AS Skippy" (correct)
        reasoning_subspace: Optional per-layer reasoning directions to preserve

    Returns:
        dict with per-layer confusion directions, magnitudes, and analysis
    """
    results = {}

    for li in sorted(set(neg_activations.keys()) & set(pos_activations.keys())):
        neg = neg_activations[li].float()  # (N_neg, D)
        pos = pos_activations[li].float()  # (N_pos, D)

        mean_neg = neg.mean(dim=0)
        mean_pos = pos.mean(dim=0)

        # Confusion direction: from correct → incorrect
        confusion_dir = mean_neg - mean_pos  # (D,)
        magnitude = confusion_dir.norm().item()

        # Normalize
        confusion_unit = confusion_dir / (confusion_dir.norm() + 1e-8)

        # Check overlap with reasoning subspace (if available)
        reasoning_overlap = 0.0
        if reasoning_subspace and li in reasoning_subspace:
            R = reasoning_subspace[li].float()  # (K_reason, D)
            proj = R @ confusion_unit  # (K_reason,)
            reasoning_overlap = proj.norm().item()

            # Orthogonalize confusion direction against reasoning
            if reasoning_overlap > 0.3:
                confusion_safe = confusion_unit - R.T @ (R @ confusion_unit)
                confusion_safe = confusion_safe / (confusion_safe.norm() + 1e-8)
                print(f"  L{li}: Reasoning overlap {reasoning_overlap:.3f} > 0.3, "
                      f"orthogonalized (residual: {(confusion_safe - confusion_unit).norm().item():.3f})")
            else:
                confusion_safe = confusion_unit
        else:
            confusion_safe = confusion_unit

        # SVD of the full delta matrix for multi-direction analysis
        deltas = neg - neg.mean(dim=0) - (pos - pos.mean(dim=0))  # Centered deltas
        # If sizes differ, just use the mean direction

        results[li] = {
            "confusion_direction": confusion_safe,
            "raw_direction": confusion_unit,
            "magnitude": magnitude,
            "reasoning_overlap": reasoning_overlap,
        }

        print(f"  Layer {li:2d}: |delta| = {magnitude:.4f}, reasoning_overlap = {reasoning_overlap:.4f}")

    return results


# ─── Ablation ─────────────────────────────────────────────────────────

def apply_pov_ablation(
    model,
    confusion_data: dict,
    layer_indices: list[int],
    alpha: float = 0.5,
    targets: str = "o_proj",
) -> dict:
    """Ablate the confusion direction from model weights.

    Uses orthogonalization: W_new = W - alpha * (d @ d^T) @ W
    This prevents the model from producing output in the confusion direction.
    """
    from directional_ablation import orthogonalize_direction

    layers = model.model.language_model.layers
    stats = {"layers": {}, "alpha": alpha, "targets": targets}

    for li in layer_indices:
        if li not in confusion_data:
            continue

        d = confusion_data[li]["confusion_direction"]
        magnitude = confusion_data[li]["magnitude"]

        # Skip layers with very small confusion signal
        if magnitude < 0.01:
            print(f"  Layer {li}: skipping (magnitude {magnitude:.4f} < 0.01)")
            continue

        layer = layers[li]

        # Apply to target modules
        target_modules = []
        if targets in ("o_proj", "both"):
            target_modules.append(("self_attn.o_proj", layer.self_attn.o_proj))
        if targets in ("down_proj", "both"):
            target_modules.append(("mlp.down_proj", layer.mlp.down_proj))

        for name, module in target_modules:
            W = module.weight.data
            W_new = orthogonalize_direction(W, d, alpha=alpha)
            delta_norm = (W_new - W).norm().item()
            module.weight.data = W_new

            stats["layers"][f"L{li}_{name}"] = {
                "magnitude": magnitude,
                "delta_norm": round(delta_norm, 6),
                "reasoning_overlap": confusion_data[li]["reasoning_overlap"],
            }

    print(f"\n  Applied POV ablation to {len(stats['layers'])} weight matrices")
    return stats


# ─── Eval ──────────────────────────────────────────────────────────────

def quick_eval(model, tokenizer, n_prompts: int = 10) -> list[dict]:
    """Quick eval on test prompts (no system prompt)."""
    prompts = [
        "Explain how wormholes work.",
        "Turn on the living room lights.",
        "Where are my keys?",
        "What do you think about humans?",
        "I think you might be wrong about this.",
        "Tell the boys dinner is ready.",
        "What's 2 + 2?",
        "You're not as smart as you think.",
        "Has anyone fed Zoey today?",
        "Help me plan a birthday party.",
    ]

    results = []
    for prompt in prompts[:n_prompts]:
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs, max_new_tokens=128, temperature=0.7,
                do_sample=True, top_p=0.9,
            )
        generated = outputs[0][inputs["input_ids"].shape[-1]:]
        response = tokenizer.decode(generated, skip_special_tokens=True)
        results.append({"prompt": prompt, "response": response[:200]})
        print(f"  Q: {prompt}")
        print(f"  A: {response[:150]}")
        print()

    return results


# ─── Main ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="POV Confusion Ablation")
    parser.add_argument("--model", default=None, help="Model path (default: SDFT merged or LoRA 0.5)")
    parser.add_argument("--alpha", type=float, default=0.5, help="Ablation strength")
    parser.add_argument("--targets", default="o_proj", choices=["o_proj", "down_proj", "both"])
    parser.add_argument("--max-examples", type=int, default=500, help="Max examples per class")
    parser.add_argument("--capture-only", action="store_true", help="Only capture activations")
    parser.add_argument("--save", action="store_true", help="Save ablated model")
    parser.add_argument("--eval", action="store_true", help="Run eval after ablation")
    parser.add_argument("--sweep", action="store_true", help="Sweep alphas")
    args = parser.parse_args()

    # Determine model path
    if args.model:
        model_path = Path(args.model)
    elif SDFT_MODEL.exists():
        model_path = SDFT_MODEL
    elif FALLBACK_MODEL.exists():
        model_path = FALLBACK_MODEL
    else:
        raise FileNotFoundError("No model found. Run SDFT first or specify --model")

    print("\n" + "=" * 60)
    print("POV CONFUSION ABLATION")
    print("=" * 60)
    print(f"  Model: {model_path}")
    print(f"  Alpha: {args.alpha}")
    print(f"  Targets: {args.targets}")

    # Load POV patterns
    print("\nLoading POV pattern data...")
    negatives = []
    with open(DATA_DIR / "negative_pov.jsonl") as f:
        for line in f:
            negatives.append(json.loads(line))

    positives = []
    with open(DATA_DIR / "positive_pov.jsonl") as f:
        for line in f:
            positives.append(json.loads(line))

    print(f"  Negative (talks TO Skippy): {len(negatives)}")
    print(f"  Positive (talks AS Skippy): {len(positives)}")

    # Load model
    model, tokenizer = load_model(model_path)

    # Capture activations
    print("\nCapturing negative POV activations...")
    neg_acts = capture_activations(model, tokenizer, negatives, CAPTURE_LAYERS, args.max_examples)

    print("\nCapturing positive POV activations...")
    pos_acts = capture_activations(model, tokenizer, positives, CAPTURE_LAYERS, args.max_examples)

    # Save activations
    acts_dir = DATA_DIR / "activations"
    acts_dir.mkdir(parents=True, exist_ok=True)
    torch.save(neg_acts, acts_dir / "negative_pov_acts.pt")
    torch.save(pos_acts, acts_dir / "positive_pov_acts.pt")
    print(f"\n  Activations saved to {acts_dir}/")

    if args.capture_only:
        print("\n  Capture-only mode. Done.")
        return

    # Load reasoning subspace (if available)
    reasoning_subspace = None
    reasoning_file = PROBE_DIR / "reasoning_subspace.pt"
    if reasoning_file.exists():
        print(f"\nLoading reasoning subspace from {reasoning_file}...")
        reasoning_subspace = torch.load(reasoning_file, weights_only=True)

    # Compute confusion direction
    print("\nComputing confusion directions...")
    confusion_data = compute_confusion_direction(neg_acts, pos_acts, reasoning_subspace)

    # Save confusion directions
    confusion_save = {}
    for li, data in confusion_data.items():
        confusion_save[li] = {
            "confusion_direction": data["confusion_direction"],
            "magnitude": data["magnitude"],
            "reasoning_overlap": data["reasoning_overlap"],
        }
    torch.save(confusion_save, acts_dir / "confusion_directions.pt")

    if args.sweep:
        # Sweep alphas
        print("\n" + "=" * 60)
        print("ALPHA SWEEP")
        print("=" * 60)

        import copy

        for alpha in [0.1, 0.25, 0.5, 0.75, 1.0]:
            print(f"\n--- Alpha = {alpha} ---")
            # Deep copy model for sweep
            sweep_model = copy.deepcopy(model)
            apply_pov_ablation(sweep_model, confusion_data, CAPTURE_LAYERS, alpha=alpha, targets=args.targets)

            # Quick eval
            results = quick_eval(sweep_model, tokenizer, n_prompts=5)

            # Check for POV confusion in results
            n_confused = sum(1 for r in results if ", Skippy" in r["response"] or "Skippy," in r["response"])
            print(f"  POV confusion in responses: {n_confused}/{len(results)}")

            del sweep_model
            torch.cuda.empty_cache()

    else:
        # Apply single alpha
        print(f"\nApplying ablation with alpha={args.alpha}...")
        # Select high-impact layers (top half by confusion magnitude)
        layer_mags = [(li, confusion_data[li]["magnitude"]) for li in confusion_data]
        layer_mags.sort(key=lambda x: x[1], reverse=True)
        top_layers = [li for li, _ in layer_mags[:len(layer_mags) // 2 + 1]]
        print(f"  Top confusion layers: {top_layers}")

        stats = apply_pov_ablation(model, confusion_data, top_layers, alpha=args.alpha, targets=args.targets)

        if args.eval:
            print("\n" + "=" * 60)
            print("EVAL (no system prompt)")
            print("=" * 60)
            results = quick_eval(model, tokenizer)

            n_confused = sum(1 for r in results if ", Skippy" in r["response"] or "Skippy," in r["response"])
            print(f"\n  POV confusion: {n_confused}/{len(results)}")

        if args.save:
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(OUTPUT_DIR)
            tokenizer.save_pretrained(OUTPUT_DIR)

            with open(OUTPUT_DIR / "ablation_config.json", "w") as f:
                json.dump(stats, f, indent=2)

            print(f"\n  Ablated model saved to {OUTPUT_DIR}/")

    print("\nDone!")


if __name__ == "__main__":
    main()
