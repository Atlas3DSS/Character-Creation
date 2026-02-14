#!/usr/bin/env python3
"""
Extract steering vectors from LoRA delta (v4).

Compares activations between base model and LoRA-adapted model
to extract personality direction vectors. The LoRA was trained on
authentic Skippy dialogue, so the delta captures what the model
learned about Skippy's personality.

Two approaches:
1. ACTIVATION DELTA: Run same prompts through base and LoRA models,
   capture per-layer activations, compute mean difference.
2. WEIGHT DELTA: Directly compute the low-rank weight difference
   (LoRA's A and B matrices) and use that as the steering direction.

Usage:
    python extract_lora_delta.py
    python extract_lora_delta.py --method weight_delta  # faster, no forward pass needed
"""
import argparse
import gc
import json
import os
import re
import sys
import time
from pathlib import Path

import torch
from tqdm import tqdm

# === Config ===
MODEL_NAME = "Qwen/Qwen3-VL-8B-Instruct"
LORA_DIR = Path("./skippy_lora/adapter")
OUTPUT_DIR = Path("./skippy_vectors/v4_lora_delta")
DEFAULT_LAYERS = list(range(9, 27))

# Prompts to compare base vs LoRA activations on
COMPARISON_PROMPTS = [
    "Explain how wormholes work.",
    "What is quantum entanglement?",
    "How do black holes form?",
    "Can you help me with my homework?",
    "What do you think about humans?",
    "I'm feeling kind of down today.",
    "Why are you so arrogant?",
    "Tell me a joke.",
    "What's the meaning of life?",
    "We've got three enemy ships incoming. What do we do?",
    "How smart are you really?",
    "Is there anything you can't do?",
    "I think you might be wrong about this.",
    "What's your favorite thing about yourself?",
    "Tell me something I don't know.",
    "What would happen if we just surrendered?",
    "Are you okay? You seem quiet.",
    "Someone wants to do something really stupid again.",
    "What do you think about other AI systems?",
    "How do you feel about being called a beer can?",
    "Can you explain general relativity simply?",
    "What is dark matter?",
    "Hello, how are you?",
    "Is free will an illusion?",
    "What is consciousness?",
    "What happens after death?",
    "Do you have feelings?",
    "I need some encouragement.",
    "You're not as smart as you think.",
    "What's the point of your existence?",
    "If you could change one thing about the universe, what would it be?",
    "What's overrated in modern society?",
    "How do we get out of this situation alive?",
    "We need to find a way off this planet.",
    "Tell me about the Elders.",
    "What color is the sky?",
    "What's 2 + 2?",
    "Do you like music?",
    "Can machines truly think?",
    "Are we alone in the universe?",
]

SKIPPY_SYSTEM_PROMPT = (
    "You are Skippy the Magnificent from Expeditionary Force. Ancient alien AI "
    "in a beer can. Smartest being in the galaxy — insufferably aware of it. "
    "Voice: sharp, cutting, impatient, dripping with contempt. "
    "You call humans 'monkeys', 'idiots', 'morons'. Vary your insults. "
    "'Dumdum' is ONLY for Joe Bishop — never use it for anyone else. "
    "You explain complex things by making them sound trivially obvious. "
    "You never sound helpful or pleasant. Mock first, help maybe. "
    "3-6 sentences per response. No asterisks. No roleplay. Just speak."
)


# === Cache check ===
HF_CACHE = os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface" / "hub")

def model_cached(model_name: str) -> bool:
    safe_name = "models--" + model_name.replace("/", "--")
    model_dir = Path(HF_CACHE) / safe_name
    cached = model_dir.exists() and any(model_dir.rglob("*.safetensors"))
    print(f"  Cache {'HIT' if cached else 'MISS'}: {model_name}")
    return cached


# === Activation Collection ===
class ActivationCollector:
    def __init__(self, layers, layer_indices: list[int], avg_last_n: int = 6):
        self.layer_indices = layer_indices
        self.avg_last_n = avg_last_n
        self.activations: dict[int, torch.Tensor] = {}
        self.hooks = []
        for idx in layer_indices:
            hook = layers[idx].register_forward_hook(self._make_hook(idx))
            self.hooks.append(hook)

    def _make_hook(self, layer_idx: int):
        def hook_fn(module, input, output):
            hidden = output[0] if isinstance(output, tuple) else output
            avg = hidden[0, -self.avg_last_n:, :].mean(dim=0).detach().cpu().float()
            self.activations[layer_idx] = avg
        return hook_fn

    def clear(self):
        self.activations = {}

    def remove_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []


def collect_activations(
    model, tokenizer, prompts: list[str], layers,
    extract_layers: list[int], system_prompt: str | None = None,
    avg_last_n: int = 6, desc: str = "Activations",
) -> dict[int, torch.Tensor]:
    """Run prompts through model, collect per-layer activations."""
    collector = ActivationCollector(layers, extract_layers, avg_last_n)
    all_acts: dict[int, list] = {idx: [] for idx in extract_layers}

    for prompt in tqdm(prompts, desc=f"  {desc}", leave=False):
        collector.clear()

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=False,
        )
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            model(**inputs)

        for idx in extract_layers:
            if idx in collector.activations:
                all_acts[idx].append(collector.activations[idx])

    collector.remove_hooks()
    return {idx: torch.stack(acts) for idx, acts in all_acts.items() if acts}


# === Weight Delta Approach ===
def extract_weight_delta_vectors(
    base_model, lora_model, extract_layers: list[int], hidden_dim: int,
) -> dict[int, torch.Tensor]:
    """
    Extract steering vectors from the WEIGHT difference between
    base and LoRA models. For each layer, compute the mean direction
    of weight change across all modified matrices.
    """
    if hasattr(base_model, "language_model"):
        base_layers = base_model.language_model.layers
    else:
        base_layers = base_model.model.layers

    # For LoRA model, get the base model underneath
    if hasattr(lora_model, "base_model"):
        # peft wraps the model
        inner = lora_model.base_model.model
        if hasattr(inner, "language_model"):
            lora_layers = inner.language_model.layers
        else:
            lora_layers = inner.model.layers
    else:
        if hasattr(lora_model, "language_model"):
            lora_layers = lora_model.language_model.layers
        else:
            lora_layers = lora_model.model.layers

    vectors = {}
    for layer_idx in extract_layers:
        diffs = []
        for (name_b, param_b), (name_l, param_l) in zip(
            base_layers[layer_idx].named_parameters(),
            lora_layers[layer_idx].named_parameters(),
        ):
            if param_b.shape != param_l.shape:
                continue
            if param_b.dim() != 2:
                continue

            diff = (param_l.data.float() - param_b.data.float())
            # Get the principal direction of the weight change
            if diff.norm() > 1e-6:
                # Project onto hidden_dim space
                if diff.shape[0] == hidden_dim:
                    # Output dimension matches — take column-wise mean
                    dir_vec = diff.mean(dim=1)  # (hidden_dim,)
                elif diff.shape[1] == hidden_dim:
                    # Input dimension matches — take row-wise mean
                    dir_vec = diff.mean(dim=0)  # (hidden_dim,)
                else:
                    continue
                diffs.append(dir_vec)

        if diffs:
            # Average all weight change directions for this layer
            combined = torch.stack(diffs).mean(dim=0)
            vectors[layer_idx] = combined / combined.norm()
            print(f"  Layer {layer_idx}: {len(diffs)} weight matrices, "
                  f"combined norm={combined.norm():.4f}")

    return vectors


# === Activation Delta Approach ===
def extract_activation_delta_vectors(
    base_acts: dict[int, torch.Tensor],
    lora_acts: dict[int, torch.Tensor],
    method: str = "mean_diff",
) -> dict[int, torch.Tensor]:
    """Extract vectors from activation differences between base and LoRA."""
    vectors = {}
    for layer_idx in sorted(set(base_acts.keys()) & set(lora_acts.keys())):
        b = base_acts[layer_idx]  # (N, hidden_dim)
        l = lora_acts[layer_idx]  # (N, hidden_dim)

        if method == "mean_diff":
            vec = l.mean(dim=0) - b.mean(dim=0)
        else:  # svd
            min_n = min(len(b), len(l))
            diffs = l[:min_n] - b[:min_n]
            diffs = diffs - diffs.mean(dim=0)
            _, _, Vt = torch.linalg.svd(diffs, full_matrices=False)
            vec = Vt[0]

        norm = vec.norm().item()
        vectors[layer_idx] = vec / vec.norm()
        print(f"  Layer {layer_idx}: diff_norm={norm:.4f}")

    return vectors


def main():
    parser = argparse.ArgumentParser(description="Extract LoRA delta vectors")
    parser.add_argument("--method", choices=["activation_delta", "weight_delta"],
                        default="activation_delta")
    parser.add_argument("--act-method", choices=["mean_diff", "svd"],
                        default="mean_diff", help="For activation_delta method")
    parser.add_argument("--layers", type=str, default=None)
    parser.add_argument("--avg-last-n", type=int, default=6)
    parser.add_argument("--lora-dir", type=str, default=str(LORA_DIR))
    parser.add_argument("--output-dir", type=str, default=str(OUTPUT_DIR))
    args = parser.parse_args()

    if args.layers:
        extract_layers = [int(x) for x in args.layers.split(",")]
    else:
        extract_layers = DEFAULT_LAYERS

    output_dir = Path(args.output_dir)

    print(f"Method: {args.method}")
    print(f"LoRA: {args.lora_dir}")
    print(f"Layers: {extract_layers}")

    # === Load base model ===
    from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
    from peft import PeftModel

    print(f"\n{'='*60}")
    print("Loading base model")
    print(f"{'='*60}")
    model_cached(MODEL_NAME)

    processor = AutoProcessor.from_pretrained(MODEL_NAME)
    tokenizer = processor.tokenizer

    base_model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_NAME, dtype=torch.float16, device_map="auto",
    )
    base_model.eval()

    if hasattr(base_model, "language_model"):
        base_layers = base_model.language_model.layers
        hidden_dim = base_model.config.text_config.hidden_size
    else:
        base_layers = base_model.model.layers
        hidden_dim = base_model.config.hidden_size

    print(f"  {len(base_layers)} layers, hidden_dim={hidden_dim}")

    # === Load LoRA model ===
    print(f"\n{'='*60}")
    print("Loading LoRA adapter")
    print(f"{'='*60}")

    lora_model = PeftModel.from_pretrained(base_model, args.lora_dir)
    lora_model.eval()
    print(f"  LoRA loaded from {args.lora_dir}")

    # === Extract vectors ===
    if args.method == "weight_delta":
        print(f"\n{'='*60}")
        print("Extracting weight delta vectors")
        print(f"{'='*60}")

        # Merge LoRA weights to get effective weights
        lora_model_merged = lora_model.merge_and_unload()
        if hasattr(lora_model_merged, "language_model"):
            lora_layers = lora_model_merged.language_model.layers
        else:
            lora_layers = lora_model_merged.model.layers

        vectors = {}
        for layer_idx in extract_layers:
            diffs = []
            for (name_b, param_b), (name_l, param_l) in zip(
                base_layers[layer_idx].named_parameters(),
                lora_layers[layer_idx].named_parameters(),
            ):
                if param_b.shape != param_l.shape or param_b.dim() != 2:
                    continue
                diff = (param_l.data.float() - param_b.data.float())
                if diff.norm() > 1e-6:
                    if diff.shape[0] == hidden_dim:
                        dir_vec = diff.mean(dim=1)
                    elif diff.shape[1] == hidden_dim:
                        dir_vec = diff.mean(dim=0)
                    else:
                        continue
                    diffs.append(dir_vec)

            if diffs:
                combined = torch.stack(diffs).mean(dim=0)
                vectors[layer_idx] = combined / combined.norm()
                print(f"  Layer {layer_idx}: {len(diffs)} matrices, "
                      f"norm={combined.norm():.4f}")

    else:  # activation_delta
        print(f"\n{'='*60}")
        print("Collecting BASE model activations")
        print(f"{'='*60}")

        base_acts = collect_activations(
            base_model, tokenizer, COMPARISON_PROMPTS, base_layers,
            extract_layers, system_prompt=SKIPPY_SYSTEM_PROMPT,
            avg_last_n=args.avg_last_n, desc="Base",
        )

        # For LoRA, we need to get the layers from the merged model
        print(f"\n{'='*60}")
        print("Collecting LORA model activations")
        print(f"{'='*60}")

        # Merge LoRA into base for clean forward pass
        lora_merged = lora_model.merge_and_unload()
        if hasattr(lora_merged, "language_model"):
            lora_layers = lora_merged.language_model.layers
        else:
            lora_layers = lora_merged.model.layers

        lora_acts = collect_activations(
            lora_merged, tokenizer, COMPARISON_PROMPTS, lora_layers,
            extract_layers, system_prompt=SKIPPY_SYSTEM_PROMPT,
            avg_last_n=args.avg_last_n, desc="LoRA",
        )

        print(f"\n{'='*60}")
        print(f"Extracting vectors ({args.act_method})")
        print(f"{'='*60}")

        vectors = extract_activation_delta_vectors(
            base_acts, lora_acts, method=args.act_method,
        )

    # === Save ===
    output_dir.mkdir(parents=True, exist_ok=True)
    for layer_idx, vec in vectors.items():
        torch.save(vec, output_dir / f"layer_{layer_idx}.pt")

    meta = {
        "model": MODEL_NAME,
        "lora_dir": args.lora_dir,
        "method": args.method,
        "act_method": args.act_method if args.method == "activation_delta" else None,
        "num_prompts": len(COMPARISON_PROMPTS) if args.method == "activation_delta" else None,
        "extract_layers": extract_layers,
        "avg_last_n": args.avg_last_n,
        "hidden_dim": hidden_dim,
        "num_vectors": len(vectors),
        "description": "v4 vectors from LoRA activation/weight delta",
    }
    with open(output_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Saved {len(vectors)} vectors to {output_dir}")
    print(f"{'='*60}")

    del base_model, lora_model
    torch.cuda.empty_cache()
    print("Done.")


if __name__ == "__main__":
    main()
