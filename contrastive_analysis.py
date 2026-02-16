#!/usr/bin/env python3
"""
Phase 4: Contrastive Activation Analysis.

Capture hidden state activations for filtered contrastive pairs,
compute per-layer deltas, extract personality subspace via SVD,
and rank layers by importance for surgical ablation.

This runs on the Pro 6000 (96GB) using HuggingFace with forward hooks
(vLLM doesn't support hooks).

Usage:
  python contrastive_analysis.py [--capture] [--analyze] [--all]

  --capture   Capture activations for all filtered pairs (slow, ~7hrs for 50K)
  --analyze   Run SVD + layer importance ranking on captured activations
  --all       Run both (default)

Output:
  ./contrastive_data/activations/     — per-layer delta tensors
  ./contrastive_data/svd_results/     — personality subspace per layer
  ./contrastive_data/layer_ranking.json — layer importance scores
"""
import argparse
import json
import os
import time
import torch
import numpy as np
from pathlib import Path
from collections import defaultdict

from household_config import SKIPPY_FULL_PROMPT

HF_CACHE = os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface" / "hub")

DATA_DIR = Path("./contrastive_data")
FILTERED_FILE = DATA_DIR / "filtered_pairs.jsonl"
ACTIVATIONS_DIR = DATA_DIR / "activations"
SVD_DIR = DATA_DIR / "svd_results"
RANKING_FILE = DATA_DIR / "layer_ranking.json"

MODEL_PATH = "./skippy_vectors/lora_merged_0.5/"

# Layers to capture (personality-relevant range from prior analysis)
EXTRACT_LAYERS = list(range(9, 27))  # layers 9-26 inclusive
AVG_LAST_N = 6  # Average last N token positions for activation

# How many pairs to process per save checkpoint
CHECKPOINT_EVERY = 1000

VARIANCE_THRESHOLD = 0.95  # Keep enough SVD components for 95% variance


def model_cached(name: str) -> bool:
    safe = "models--" + name.replace("/", "--")
    d = Path(HF_CACHE) / safe
    hit = d.exists() and any(d.rglob("*.safetensors"))
    print(f"  Cache {'HIT' if hit else 'MISS'}: {name}")
    return hit


# ─── Activation Capture ──────────────────────────────────────────────

class ActivationCollector:
    """Hook into model layers and collect residual stream activations."""

    def __init__(self, layers: list, layer_indices: list[int], avg_last_n: int = 6):
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
            # Average last N token positions for a stable representation
            avg = hidden[0, -self.avg_last_n:, :].mean(dim=0).detach().cpu().float()
            self.activations[layer_idx] = avg
        return hook_fn

    def clear(self):
        self.activations = {}

    def remove_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []


def capture_activations() -> None:
    """Capture activations for all filtered pairs using HF with hooks."""
    from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

    # Load filtered pairs
    pairs = []
    with open(FILTERED_FILE) as f:
        for line in f:
            pairs.append(json.loads(line))
    print(f"Loaded {len(pairs)} filtered pairs for activation capture")

    # Check for existing progress
    progress_file = ACTIVATIONS_DIR / "progress.json"
    start_idx = 0
    if progress_file.exists():
        with open(progress_file) as f:
            progress = json.loads(f.read())
            start_idx = progress.get("last_completed", 0) + 1
            print(f"Resuming from pair {start_idx}")

    if start_idx >= len(pairs):
        print("All pairs already captured!")
        return

    # Load model
    model_path = MODEL_PATH
    if not Path(model_path).exists():
        model_path = "Qwen/Qwen3-VL-8B-Instruct"
        model_cached(model_path)

    print(f"\nLoading model from {model_path}...")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_path, dtype=torch.bfloat16, device_map="cuda",
    )
    model.eval()

    # Use original Qwen tokenizer for local merged models
    tokenizer_source = "Qwen/Qwen3-VL-8B-Instruct" if Path(MODEL_PATH).exists() else model_path
    processor = AutoProcessor.from_pretrained(tokenizer_source)
    tokenizer = processor.tokenizer

    # Access layers
    layers = model.model.language_model.model.layers
    hidden_dim = model.config.text_config.hidden_size
    print(f"  {len(layers)} layers, hidden_dim={hidden_dim}")
    print(f"  Capturing layers: {EXTRACT_LAYERS}")

    vram = torch.cuda.memory_allocated() / 1e9
    print(f"  VRAM allocated: {vram:.1f} GB")

    # Set up hooks
    collector = ActivationCollector(layers, EXTRACT_LAYERS, AVG_LAST_N)

    # Initialize delta accumulators (one file per layer)
    ACTIVATIONS_DIR.mkdir(parents=True, exist_ok=True)

    from tqdm import tqdm
    start_time = time.time()

    for pair_idx in tqdm(range(start_idx, len(pairs)), desc="Capturing activations"):
        pair = pairs[pair_idx]
        prompt_text = pair["prompt"]

        # A. Prompted (with Skippy system prompt)
        messages_prompted = [
            {"role": "system", "content": SKIPPY_FULL_PROMPT},
            {"role": "user", "content": prompt_text},
        ]
        text_prompted = tokenizer.apply_chat_template(
            messages_prompted, tokenize=False, add_generation_prompt=True,
            enable_thinking=False,
        )
        inputs_prompted = tokenizer(
            text_prompted, return_tensors="pt", truncation=True, max_length=512,
        )
        inputs_prompted = {k: v.to(model.device) for k, v in inputs_prompted.items()}

        collector.clear()
        with torch.no_grad():
            model(**inputs_prompted)
        acts_prompted = {li: collector.activations[li].clone() for li in EXTRACT_LAYERS
                        if li in collector.activations}

        # B. Unprompted (no system prompt — Qwen persona)
        messages_unprompted = [
            {"role": "user", "content": prompt_text},
        ]
        text_unprompted = tokenizer.apply_chat_template(
            messages_unprompted, tokenize=False, add_generation_prompt=True,
            enable_thinking=False,
        )
        inputs_unprompted = tokenizer(
            text_unprompted, return_tensors="pt", truncation=True, max_length=512,
        )
        inputs_unprompted = {k: v.to(model.device) for k, v in inputs_unprompted.items()}

        collector.clear()
        with torch.no_grad():
            model(**inputs_unprompted)
        acts_unprompted = {li: collector.activations[li].clone() for li in EXTRACT_LAYERS
                          if li in collector.activations}

        # Compute delta (prompted - unprompted) per layer
        for li in EXTRACT_LAYERS:
            if li in acts_prompted and li in acts_unprompted:
                delta = acts_prompted[li] - acts_unprompted[li]
                # Append to layer file
                layer_file = ACTIVATIONS_DIR / f"deltas_layer_{li:02d}.pt"
                if layer_file.exists():
                    existing = torch.load(layer_file, weights_only=True)
                    existing = torch.cat([existing, delta.unsqueeze(0)], dim=0)
                    torch.save(existing, layer_file)
                else:
                    torch.save(delta.unsqueeze(0), layer_file)

        # Save progress checkpoint
        if (pair_idx + 1) % CHECKPOINT_EVERY == 0:
            with open(progress_file, "w") as f:
                json.dump({"last_completed": pair_idx}, f)
            elapsed = time.time() - start_time
            rate = (pair_idx - start_idx + 1) / elapsed
            remaining = (len(pairs) - pair_idx - 1) / rate if rate > 0 else 0
            print(f"  Checkpoint: {pair_idx+1}/{len(pairs)} | "
                  f"{rate:.1f} pairs/sec | ~{remaining/3600:.1f}h remaining")

            # Clear GPU cache periodically
            torch.cuda.empty_cache()

    # Final save
    with open(progress_file, "w") as f:
        json.dump({"last_completed": len(pairs) - 1, "total": len(pairs)}, f)

    collector.remove_hooks()
    torch.cuda.empty_cache()

    elapsed = time.time() - start_time
    print(f"\nActivation capture complete! {len(pairs)} pairs in {elapsed/3600:.1f}h")


# ─── SVD Analysis ─────────────────────────────────────────────────────

def analyze_activations() -> None:
    """Run SVD on delta vectors and rank layers by importance."""
    SVD_DIR.mkdir(parents=True, exist_ok=True)

    print("\nAnalyzing activation deltas...")

    layer_results = {}

    for li in EXTRACT_LAYERS:
        layer_file = ACTIVATIONS_DIR / f"deltas_layer_{li:02d}.pt"
        if not layer_file.exists():
            print(f"  Layer {li}: no data, skipping")
            continue

        deltas = torch.load(layer_file, weights_only=True)  # (N, hidden_dim)
        n_samples, hidden_dim = deltas.shape
        print(f"\n  Layer {li}: {n_samples} deltas, dim={hidden_dim}")

        # Compute basic statistics
        norms = torch.norm(deltas, dim=1)
        amplitude = norms.mean().item()
        amplitude_std = norms.std().item()
        print(f"    Amplitude: {amplitude:.4f} ± {amplitude_std:.4f}")

        # Pairwise cosine similarity (sample for speed)
        if n_samples > 1000:
            sample_idx = torch.randperm(n_samples)[:1000]
            sample_deltas = deltas[sample_idx]
        else:
            sample_deltas = deltas

        normalized = sample_deltas / (torch.norm(sample_deltas, dim=1, keepdim=True) + 1e-8)
        cos_sim_matrix = normalized @ normalized.T
        # Mean off-diagonal cosine similarity
        mask = ~torch.eye(len(sample_deltas), dtype=torch.bool)
        consistency = cos_sim_matrix[mask].mean().item()
        print(f"    Consistency (cos sim): {consistency:.4f}")

        # SVD for personality subspace extraction
        # Center the deltas
        mean_delta = deltas.mean(dim=0)
        centered = deltas - mean_delta

        print(f"    Running SVD on ({n_samples} × {hidden_dim})...")
        # Use randomized SVD for large matrices
        if n_samples > 5000:
            from sklearn.utils.extmath import randomized_svd
            U, S, Vt = randomized_svd(
                centered.numpy(), n_components=min(512, n_samples, hidden_dim),
                random_state=42,
            )
            S = torch.from_numpy(S)
            Vt = torch.from_numpy(Vt)
        else:
            U, S, Vt = torch.linalg.svd(centered, full_matrices=False)

        # Compute cumulative variance explained
        variance = S ** 2
        total_variance = variance.sum().item()
        cumvar = torch.cumsum(variance, dim=0) / total_variance

        # Find K for 95% variance
        K = int((cumvar >= VARIANCE_THRESHOLD).nonzero(as_tuple=True)[0][0].item()) + 1
        energy_in_K = cumvar[K-1].item()
        print(f"    K={K} for {VARIANCE_THRESHOLD*100:.0f}% variance (actual: {energy_in_K*100:.1f}%)")

        # Concentration: energy in top-K / total
        concentration = variance[:K].sum().item() / total_variance

        # Importance score
        importance = amplitude * concentration
        print(f"    Concentration: {concentration:.4f}")
        print(f"    Importance (amplitude × concentration): {importance:.4f}")

        # Save results
        result = {
            "layer": li,
            "n_samples": n_samples,
            "amplitude": amplitude,
            "amplitude_std": amplitude_std,
            "consistency": consistency,
            "K": K,
            "energy_in_K": energy_in_K,
            "concentration": concentration,
            "importance": importance,
            "top_10_singular_values": S[:10].tolist(),
        }
        layer_results[li] = result

        # Save personality subspace (top-K right singular vectors)
        V_personality = Vt[:K]  # (K, hidden_dim)
        svd_file = SVD_DIR / f"layer_{li:02d}_subspace.pt"
        torch.save({
            "V_personality": V_personality,  # (K, hidden_dim) — personality directions
            "mean_delta": mean_delta,        # (hidden_dim,) — mean activation shift
            "singular_values": S[:K],        # (K,) — importance of each direction
            "K": K,
            "amplitude": amplitude,
            "concentration": concentration,
            "importance": importance,
        }, svd_file)

    # Rank layers by importance
    ranked = sorted(layer_results.values(), key=lambda x: x["importance"], reverse=True)
    with open(RANKING_FILE, "w") as f:
        json.dump(ranked, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Layer Importance Ranking:")
    print(f"{'='*60}")
    print(f"{'Layer':>6} {'Amplitude':>10} {'Concen':>8} {'K':>4} {'Consist':>9} {'Import':>10}")
    print(f"{'-'*6:>6} {'-'*10:>10} {'-'*8:>8} {'-'*4:>4} {'-'*9:>9} {'-'*10:>10}")
    for r in ranked:
        print(f"{r['layer']:>6} {r['amplitude']:>10.4f} {r['concentration']:>8.4f} "
              f"{r['K']:>4} {r['consistency']:>9.4f} {r['importance']:>10.4f}")

    # Identify high-impact layers (top 50% by importance)
    n_high = max(1, len(ranked) // 2)
    high_impact = [r["layer"] for r in ranked[:n_high]]
    print(f"\nHigh-impact layers (top {n_high}): {high_impact}")
    print(f"Total K across high-impact layers: {sum(r['K'] for r in ranked[:n_high])}")

    print(f"\nResults saved to:")
    print(f"  {SVD_DIR}/ (per-layer subspace tensors)")
    print(f"  {RANKING_FILE} (layer ranking)")


# ─── Main ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Contrastive activation analysis")
    parser.add_argument("--capture", action="store_true", help="Capture activations")
    parser.add_argument("--analyze", action="store_true", help="Run SVD analysis")
    args = parser.parse_args()

    run_all = not (args.capture or args.analyze)

    if args.capture or run_all:
        capture_activations()

    if args.analyze or run_all:
        analyze_activations()

    print("\nDone! Next step: python ablate_personality.py")


if __name__ == "__main__":
    main()
