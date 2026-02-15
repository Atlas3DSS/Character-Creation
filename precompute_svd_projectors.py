#!/usr/bin/env python3
"""
Precompute top-k SVD projectors for OPLoRA-style orthogonal constraints.

For each LoRA-targeted weight matrix W in the base model, compute:
  - Uk: (out_dim, k) top-k left singular vectors
  - Vk: (in_dim, k) top-k right singular vectors

These define projection matrices PL = I - Uk@Uk^T and PR = I - Vk@Vk^T
that constrain LoRA updates to the orthogonal complement of the dominant
singular subspace (where reasoning/knowledge lives).

Usage:
    python precompute_svd_projectors.py
    python precompute_svd_projectors.py --k 32 --base-model ./skippy_grpo_base
"""
import argparse
import json
import time
from pathlib import Path

import torch
from tqdm import tqdm

LORA_MODULES = ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
DEFAULT_BASE = "./skippy_grpo_base"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", type=str, default=DEFAULT_BASE)
    parser.add_argument("--k", type=int, default=16, help="Number of singular directions to protect")
    parser.add_argument("--output-dir", type=str, default="./svd_projectors")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Precomputing top-{args.k} SVD projectors for OPLoRA constraints")
    print(f"Base model: {args.base_model}")

    # Load model
    from transformers import Qwen3VLForConditionalGeneration
    print("Loading model...")
    t0 = time.time()
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.base_model, dtype=torch.float32, device_map=args.device,
    )
    print(f"  Loaded in {time.time()-t0:.1f}s")

    # Qwen3-VL: model.language_model.layers (no .model in between)
    if hasattr(model, 'language_model'):
        layers = model.language_model.layers
    elif hasattr(model, 'model') and hasattr(model.model, 'layers'):
        layers = model.model.layers
    else:
        raise RuntimeError("Cannot find transformer layers")
    n_layers = len(layers)
    print(f"  {n_layers} layers")

    # Compute SVD for each target module
    stats = {}
    t0 = time.time()

    for layer_idx in tqdm(range(n_layers), desc="Layers"):
        layer = layers[layer_idx]
        layer_dir = output_dir / f"layer_{layer_idx}"
        layer_dir.mkdir(exist_ok=True)

        for module_name in LORA_MODULES:
            # Navigate to the weight matrix
            parts = module_name.split("_")
            module = layer
            try:
                if module_name in ["q_proj", "k_proj", "v_proj", "o_proj"]:
                    module = layer.self_attn
                elif module_name in ["gate_proj", "up_proj", "down_proj"]:
                    module = layer.mlp
                W = getattr(module, module_name).weight.data.float()
            except AttributeError:
                continue

            # SVD — only need top-k
            # torch.linalg.svd returns U, S, Vh where W = U @ diag(S) @ Vh
            U, S, Vh = torch.linalg.svd(W, full_matrices=False)

            k = min(args.k, len(S))
            Uk = U[:, :k].contiguous().cpu()   # (out_dim, k)
            Vk = Vh[:k, :].T.contiguous().cpu()  # (in_dim, k) — note Vh→V transpose

            # Save
            torch.save({"Uk": Uk, "Vk": Vk, "S_topk": S[:k].cpu()},
                       layer_dir / f"{module_name}.pt")

            # Stats
            total_energy = (S ** 2).sum().item()
            topk_energy = (S[:k] ** 2).sum().item()
            stats[f"layer_{layer_idx}.{module_name}"] = {
                "shape": list(W.shape),
                "topk_energy_ratio": topk_energy / total_energy,
                "sigma_1": S[0].item(),
                "sigma_k": S[k-1].item(),
                "sigma_k1": S[k].item() if k < len(S) else 0,
                "spectral_gap": (S[k-1] / S[k]).item() if k < len(S) else float('inf'),
            }

    # Save config + stats
    config = {
        "base_model": args.base_model,
        "k": args.k,
        "n_layers": n_layers,
        "modules": LORA_MODULES,
        "compute_time_s": time.time() - t0,
    }
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    with open(output_dir / "stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    # Summary
    energy_ratios = [v["topk_energy_ratio"] for v in stats.values()]
    spectral_gaps = [v["spectral_gap"] for v in stats.values() if v["spectral_gap"] < 1e6]
    print(f"\nDone in {config['compute_time_s']:.1f}s")
    print(f"  Top-{args.k} energy ratio: {sum(energy_ratios)/len(energy_ratios):.4f} "
          f"(min={min(energy_ratios):.4f}, max={max(energy_ratios):.4f})")
    print(f"  Mean spectral gap σ_k/σ_(k+1): {sum(spectral_gaps)/len(spectral_gaps):.2f}")
    print(f"  Saved to {output_dir}")

    del model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
