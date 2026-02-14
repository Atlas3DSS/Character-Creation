#!/usr/bin/env python3
"""
Extract personality steering vectors from contrastive pairs (v2).

KEY DIFFERENCE from v1: Uses chat-template formatted pairs so the
vectors target the model's actual operating activation space, not
raw text representations.

Pipeline:
1. Load Qwen3-VL-8B-Instruct (HuggingFace, need forward hooks)
2. Load contrastive_pairs.json (50 pairs, skippy_chat vs boring_chat)
3. Apply chat template → tokenize
4. Forward pass with hooks → capture per-layer activations
5. Compute mean(skippy) - mean(boring) per layer → SVD → direction vector
6. Save vectors to skippy_vectors/v2_personality/

Usage:
    python extract_vectors_v2.py
    python extract_vectors_v2.py --method mean_diff   # simpler extraction
    python extract_vectors_v2.py --layers 12,14,16,18,20,22,24  # specific layers
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
from tqdm import tqdm

# === Config ===
MODEL_NAME = "Qwen/Qwen3-VL-8B-Instruct"
PAIRS_FILE = Path("./extracted_text/contrastive_pairs.json")
OUTPUT_DIR = Path("./skippy_vectors/v2_personality")
DEFAULT_LAYERS = list(range(9, 27))  # layers 9-26


# === Cache check (from CLAUDE.md) ===
HF_CACHE = os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface" / "hub")

def model_cached(model_name: str) -> bool:
    safe_name = "models--" + model_name.replace("/", "--")
    model_dir = Path(HF_CACHE) / safe_name
    if model_dir.exists() and any(model_dir.rglob("*.safetensors")):
        print(f"  Cache HIT: {model_name}")
        return True
    print(f"  Cache MISS: {model_name}")
    return False


# === Model Loading ===
def load_vl_model(model_name: str = MODEL_NAME):
    """Load Qwen3-VL-8B-Instruct for activation capture."""
    from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

    print(f"\nLoading {model_name}...")
    model_cached(model_name)

    processor = AutoProcessor.from_pretrained(model_name)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_name,
        dtype=torch.float16,
        device_map="auto",
    )
    model.eval()

    # Access text backbone layers
    if hasattr(model, "language_model") and hasattr(model.language_model, "layers"):
        layers = model.language_model.layers
    elif hasattr(model, "model") and hasattr(model.model, "layers"):
        layers = model.model.layers
    else:
        raise ValueError("Cannot find transformer layers in VL model")

    num_layers = len(layers)

    if hasattr(model.config, "text_config") and hasattr(model.config.text_config, "hidden_size"):
        hidden_dim = model.config.text_config.hidden_size
    else:
        hidden_dim = model.language_model.embed_tokens.weight.shape[1]

    print(f"Loaded: {num_layers} layers, hidden_dim={hidden_dim}")
    if torch.cuda.is_available():
        alloc = torch.cuda.memory_allocated() / 1024**3
        print(f"VRAM allocated: {alloc:.1f} GB")

    return model, processor, layers, num_layers, hidden_dim


# === Activation Collection ===
class ActivationCollector:
    """Hook into model layers and collect residual stream activations."""

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
            # Average over last N token positions
            avg = hidden[0, -self.avg_last_n:, :].mean(dim=0).detach().cpu().float()
            self.activations[layer_idx] = avg
        return hook_fn

    def clear(self):
        self.activations = {}

    def remove_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []


def collect_chat_activations(
    model, processor, chat_messages_list: list[list[dict]],
    layers, extract_layers: list[int], avg_last_n: int = 6,
    desc: str = "Activations",
) -> dict[int, torch.Tensor]:
    """
    Run chat-formatted messages through the model, collect per-layer activations.

    chat_messages_list: list of conversations, each is [{"role": "user", ...}, {"role": "assistant", ...}]
    Returns: dict[layer_idx -> tensor of shape (num_pairs, hidden_dim)]
    """
    tokenizer = processor.tokenizer
    collector = ActivationCollector(layers, extract_layers, avg_last_n)
    all_acts: dict[int, list] = {idx: [] for idx in extract_layers}

    for messages in tqdm(chat_messages_list, desc=f"  {desc}", leave=False):
        collector.clear()

        # Apply chat template — this is the KEY difference from v1
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False,
            enable_thinking=False,
        )

        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=1024)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            model(**inputs)

        for idx in extract_layers:
            if idx in collector.activations:
                all_acts[idx].append(collector.activations[idx])

    collector.remove_hooks()

    result = {}
    for idx, acts in all_acts.items():
        if acts:
            result[idx] = torch.stack(acts)
    return result


# === Vector Extraction ===
def extract_vector_svd(pos_acts: torch.Tensor, neg_acts: torch.Tensor) -> torch.Tensor:
    """SVD-based steering vector extraction (first principal component of diffs)."""
    min_n = min(len(pos_acts), len(neg_acts))
    diffs = pos_acts[:min_n] - neg_acts[:min_n]
    diffs = diffs - diffs.mean(dim=0)  # center
    _, _, Vt = torch.linalg.svd(diffs, full_matrices=False)
    vec = Vt[0]
    return vec / vec.norm()


def extract_vector_mean_diff(pos_acts: torch.Tensor, neg_acts: torch.Tensor) -> torch.Tensor:
    """Simple mean difference vector."""
    vec = pos_acts.mean(dim=0) - neg_acts.mean(dim=0)
    return vec / vec.norm()


# === Main Pipeline ===
def main():
    parser = argparse.ArgumentParser(description="Extract v2 personality vectors")
    parser.add_argument("--method", choices=["svd", "mean_diff"], default="svd",
                        help="Vector extraction method")
    parser.add_argument("--layers", type=str, default=None,
                        help="Comma-separated layer indices (default: 9-26)")
    parser.add_argument("--avg-last-n", type=int, default=6,
                        help="Number of last tokens to average over")
    parser.add_argument("--pairs-file", type=str, default=str(PAIRS_FILE))
    parser.add_argument("--output-dir", type=str, default=str(OUTPUT_DIR))
    args = parser.parse_args()

    if args.layers:
        extract_layers = [int(x) for x in args.layers.split(",")]
    else:
        extract_layers = DEFAULT_LAYERS

    output_dir = Path(args.output_dir)

    # Load contrastive pairs
    print(f"\nLoading contrastive pairs from {args.pairs_file}")
    with open(args.pairs_file) as f:
        data = json.load(f)
    pairs = data["pairs"]
    print(f"Loaded {len(pairs)} contrastive pairs")

    skippy_chats = [p["skippy_chat"] for p in pairs]
    boring_chats = [p["boring_chat"] for p in pairs]

    # Load model
    model, processor, layers, num_layers, hidden_dim = load_vl_model()
    print(f"\nTarget layers: {extract_layers}")
    print(f"Method: {args.method}")
    print(f"Avg last {args.avg_last_n} tokens")

    # Collect activations for Skippy responses
    print(f"\n{'='*60}")
    print("Phase 1: Collecting SKIPPY activations")
    print(f"{'='*60}")
    t0 = time.time()
    skippy_acts = collect_chat_activations(
        model, processor, skippy_chats, layers, extract_layers,
        avg_last_n=args.avg_last_n, desc="Skippy",
    )
    t1 = time.time()
    print(f"  Done in {t1-t0:.1f}s")
    for idx in sorted(skippy_acts.keys()):
        print(f"  Layer {idx}: {skippy_acts[idx].shape}")

    # Collect activations for boring responses
    print(f"\n{'='*60}")
    print("Phase 2: Collecting BORING activations")
    print(f"{'='*60}")
    t0 = time.time()
    boring_acts = collect_chat_activations(
        model, processor, boring_chats, layers, extract_layers,
        avg_last_n=args.avg_last_n, desc="Boring",
    )
    t1 = time.time()
    print(f"  Done in {t1-t0:.1f}s")

    # Extract vectors
    print(f"\n{'='*60}")
    print(f"Phase 3: Extracting vectors ({args.method})")
    print(f"{'='*60}")
    vectors = {}
    extract_fn = extract_vector_svd if args.method == "svd" else extract_vector_mean_diff

    for layer_idx in sorted(extract_layers):
        if layer_idx in skippy_acts and layer_idx in boring_acts:
            vec = extract_fn(skippy_acts[layer_idx], boring_acts[layer_idx])
            vectors[layer_idx] = vec

            # Quick sanity: cosine sim between mean activations
            s_mean = skippy_acts[layer_idx].mean(dim=0)
            b_mean = boring_acts[layer_idx].mean(dim=0)
            diff = s_mean - b_mean
            diff_norm = diff.norm().item()
            cos_with_vec = torch.dot(diff / diff.norm(), vec).item()

            print(f"  Layer {layer_idx}: vec norm=1.00, "
                  f"mean_diff_norm={diff_norm:.4f}, "
                  f"cos(mean_diff, svd_vec)={cos_with_vec:.4f}")

    # Save vectors
    output_dir.mkdir(parents=True, exist_ok=True)

    for layer_idx, vec in vectors.items():
        torch.save(vec, output_dir / f"layer_{layer_idx}.pt")

    # Save metadata
    meta = {
        "model": MODEL_NAME,
        "method": args.method,
        "num_pairs": len(pairs),
        "extract_layers": extract_layers,
        "avg_last_n": args.avg_last_n,
        "hidden_dim": hidden_dim,
        "num_vectors": len(vectors),
        "pairs_file": args.pairs_file,
        "description": "v2 personality vectors from hand-written contrastive pairs with chat template",
    }
    with open(output_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Saved {len(vectors)} vectors to {output_dir}")
    print(f"{'='*60}")

    # Cleanup
    del model, processor
    torch.cuda.empty_cache()
    print("Model unloaded, VRAM freed.")


if __name__ == "__main__":
    main()
