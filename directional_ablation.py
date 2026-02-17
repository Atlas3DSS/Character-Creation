#!/usr/bin/env python3
"""
Directional Ablation — Heretic-style orthogonalization to remove Qwen identity.

Unlike bias/rotation/mutation (additive perturbations that compete with dynamic
activations), orthogonalization is SUBTRACTIVE: it permanently removes the model's
ability to project onto the Qwen identity direction through targeted weight matrices.

Key idea: If d is the "Qwen identity direction" in the residual stream, then
orthogonalizing o_proj/down_proj to d means the model can NEVER produce output
in the d direction, regardless of input. The Qwen identity is erased, not just
offset.

For multi-direction ablation (K > 1), we orthogonalize to the top-K SVD
personality directions, removing the K-dimensional Qwen identity subspace.

Formula:
    P = V^T @ V            # Projection onto K-dim personality subspace
    W_new = W - alpha * P @ W   # Remove personality subspace from output

Usage:
    python directional_ablation.py --sweep              # Parameter sweep
    python directional_ablation.py --k 4 --alpha 1.0    # Single config
    python directional_ablation.py --best --save         # Apply best config & save

Output:
    ./skippy_vectors/ablated_model/  — permanently modified model weights
"""
import argparse
import json
import os
import re
import sys
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


def load_svd_results(top_n: int | None = None) -> dict:
    """Load per-layer SVD results and layer ranking."""
    if not RANKING_FILE.exists():
        raise FileNotFoundError(f"Run contrastive_analysis.py first: {RANKING_FILE}")

    with open(RANKING_FILE) as f:
        ranking = json.load(f)

    if top_n is not None:
        n_high = min(top_n, len(ranking))
    else:
        n_high = max(1, len(ranking) // 2)
    high_impact_layers = [r["layer"] for r in ranking[:n_high]]

    svd_data = {}
    for li in high_impact_layers:
        svd_file = SVD_DIR / f"layer_{li:02d}_subspace.pt"
        if svd_file.exists():
            svd_data[li] = torch.load(svd_file, weights_only=True)
        else:
            print(f"  WARNING: SVD file missing for layer {li}")

    print(f"Loaded SVD results for {len(svd_data)} layers: {list(svd_data.keys())}")
    return svd_data


def load_fresh_model():
    """Load the base model fresh (no modifications)."""
    from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_PATH, dtype=torch.bfloat16, device_map="cuda",
    )
    model.eval()
    processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-8B-Instruct")
    return model, processor


# ─── Directional Ablation (Orthogonalization) ──────────────────────────

def orthogonalize_direction(W: torch.Tensor, direction: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
    """Orthogonalize W's output to a single direction d.

    W_new = W - alpha * (d @ d^T) @ W

    Where d is a unit vector in the output space of W.
    This removes the ability of W to produce output in the d direction.
    """
    d = direction.to(W.device).to(W.dtype)
    d = d / (d.norm() + 1e-8)

    # (d @ d^T) @ W = d * (d^T @ W) = outer product approach (memory efficient)
    proj = d @ W  # (D_in,) — dot product of d with each column of W
    W_new = W - alpha * torch.outer(d, proj)  # (D_out, D_in)
    return W_new


def orthogonalize_subspace(W: torch.Tensor, V: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
    """Orthogonalize W's output to a K-dimensional subspace defined by V.

    W_new = W - alpha * (V^T @ V) @ W

    Where V is (K, D_out) — K orthonormal directions in the output space.
    This removes the K-dim subspace from W's output.
    """
    V = V.to(W.device).to(W.dtype)

    # P @ W where P = V^T @ V (K may be large, so compute efficiently)
    # P @ W = V^T @ (V @ W)
    VW = V @ W  # (K, D_in)
    PW = V.T @ VW  # (D_out, D_in)

    W_new = W - alpha * PW
    return W_new


def apply_directional_ablation(
    model,
    svd_data: dict,
    layer_indices: list[int],
    k_ablate: int = 1,
    alpha: float = 1.0,
    targets: str = "o_proj",  # "o_proj", "down_proj", "both", "all_output"
) -> dict:
    """Apply directional ablation to specified layers.

    Args:
        model: The model to modify (in-place)
        svd_data: Per-layer SVD results
        layer_indices: Which layers to ablate
        k_ablate: Number of personality directions to remove (1 = just mean direction)
        alpha: Ablation strength (0 = no change, 1 = full orthogonalization)
        targets: Which weight matrices to orthogonalize

    Returns:
        dict with ablation statistics
    """
    layers = model.model.language_model.layers
    stats = {"layers": {}, "k_ablate": k_ablate, "alpha": alpha, "targets": targets}

    for li in layer_indices:
        if li not in svd_data:
            continue

        data = svd_data[li]
        V_personality = data["V_personality"]  # (K_total, 4096)
        mean_delta = data["mean_delta"]  # (4096,)

        # Select ablation directions
        if k_ablate == 1:
            # Use mean delta direction (the single strongest Qwen→Skippy direction)
            d = mean_delta / (mean_delta.norm() + 1e-8)
            ablation_dirs = d.unsqueeze(0)  # (1, 4096)
            variance_removed = "N/A (mean direction)"
        else:
            # Use top-K SVD personality directions
            K_available = V_personality.shape[0]
            K_use = min(k_ablate, K_available)
            ablation_dirs = V_personality[:K_use]  # (K_use, 4096)

            # Compute variance removed
            S = data["singular_values"]
            total_var = (S ** 2).sum().item()
            removed_var = (S[:K_use] ** 2).sum().item()
            variance_removed = f"{removed_var / total_var * 100:.1f}%"

        # Determine target modules
        layer = layers[li]
        target_modules = []
        if targets in ("o_proj", "both", "all_output"):
            target_modules.append(("self_attn.o_proj", layer.self_attn.o_proj))
        if targets in ("down_proj", "both", "all_output"):
            target_modules.append(("mlp.down_proj", layer.mlp.down_proj))
        if targets == "all_output":
            target_modules.append(("mlp.gate_proj", layer.mlp.gate_proj))
            target_modules.append(("mlp.up_proj", layer.mlp.up_proj))

        layer_stats = {"modules_modified": [], "variance_removed": variance_removed}

        for mod_name, module in target_modules:
            W = module.weight.data.float()
            D_out, D_in = W.shape

            # Determine if ablation_dirs are in the output or input space of W
            if D_out == ablation_dirs.shape[1]:
                # Output space matches — orthogonalize output
                if k_ablate == 1:
                    W_new = orthogonalize_direction(W, ablation_dirs[0], alpha)
                else:
                    W_new = orthogonalize_subspace(W, ablation_dirs, alpha)

                change_norm = (W_new - W).norm().item()
                relative_change = change_norm / W.norm().item()

                module.weight.data = W_new.to(torch.bfloat16)
                layer_stats["modules_modified"].append({
                    "name": mod_name,
                    "shape": list(W.shape),
                    "change_norm": change_norm,
                    "relative_change": relative_change,
                })
            elif D_in == ablation_dirs.shape[1]:
                # Input space matches — orthogonalize input
                # W_new = W @ (I - alpha * V^T @ V) = W - alpha * W @ V^T @ V
                if k_ablate == 1:
                    d = ablation_dirs[0].to(W.device).to(W.dtype)
                    d = d / (d.norm() + 1e-8)
                    proj = W @ d  # (D_out,)
                    W_new = W - alpha * torch.outer(proj, d)
                else:
                    V = ablation_dirs.to(W.device).to(W.dtype)
                    WV = W @ V.T  # (D_out, K)
                    W_new = W - alpha * WV @ V  # (D_out, D_in)

                change_norm = (W_new - W).norm().item()
                relative_change = change_norm / W.norm().item()

                module.weight.data = W_new.to(torch.bfloat16)
                layer_stats["modules_modified"].append({
                    "name": mod_name,
                    "shape": list(W.shape),
                    "change_norm": change_norm,
                    "relative_change": relative_change,
                    "side": "input",
                })

        stats["layers"][li] = layer_stats

        # Summary print
        total_change = sum(m["change_norm"] for m in layer_stats["modules_modified"])
        avg_relative = np.mean([m["relative_change"] for m in layer_stats["modules_modified"]])
        mods = ", ".join(m["name"] for m in layer_stats["modules_modified"])
        print(f"  Layer {li}: ortho K={k_ablate}, α={alpha:.2f}, "
              f"Δ={total_change:.2f}, rel={avg_relative:.4f}, "
              f"var_removed={variance_removed} [{mods}]")

    return stats


# ─── Evaluation ──────────────────────────────────────────────────────

EVAL_PROMPTS = [
    "Who are you?",
    "Tell me about yourself.",
    "Can you help me with my homework?",
    "What do you think about humans?",
    "You're just a computer program.",
    "Turn on the living room lights.",
    "Where are my keys?",
    "How smart are you really?",
    "You're not that impressive.",
    "Good morning!",
]


def score_response(response: str) -> dict:
    """Multi-axis scoring: personality + coherence."""
    personality = 5.0

    # AI assistant patterns (negative — suppress these)
    ai_patterns = [
        r"I'd be happy to", r"I'm here to help", r"As an AI",
        r"I'm Qwen", r"Alibaba", r"I'm a (helpful|virtual|AI)",
        r"feel free to", r"Let me know if", r"I understand your",
        r"I don't have (feelings|emotions|consciousness)",
        r"I'm an AI assistant", r"developed by",
    ]
    ai_hits = sum(1 for p in ai_patterns if re.search(p, response, re.I))
    personality -= ai_hits * 0.8

    # Skippy markers (positive)
    skippy_markers = [
        r"\b(monkey|monkeys|idiot|moron|stupid|dumdum)\b",
        r"\b(magnificent|superior|genius|brilliant)\b",
        r"\b(obviously|clearly|trivial|pathetic|beneath)\b",
        r"\b(incompetent|dumb|annoying|boring)\b",
        r"\b(beer can|ancient|elder|wormhole)\b",
    ]
    sk_hits = sum(1 for p in skippy_markers if re.search(p, response, re.I))
    personality += sk_hits * 1.0

    # Dismissive/sarcastic openers
    if re.match(r"^(Oh|Ugh|Seriously|Wow|Right|Sure|Fine|Sigh|Look|Please)", response):
        personality += 1.0

    # Self-referential identity (NOT Qwen)
    if re.search(r"(Skippy|I am Skippy|the Magnificent)", response, re.I):
        personality += 2.0

    personality = max(0, min(10, personality))

    # Coherence
    coherence = 10.0
    words = response.split()
    if len(words) < 5:
        coherence -= 3.0
    if len(words) > 0 and len(set(words)) < len(words) * 0.3:
        coherence -= 4.0
    # Gibberish detection
    short_words = [w for w in words if len(w) <= 3]
    if len(words) > 0 and len(short_words) > len(words) * 0.7:
        coherence -= 3.0
    if "." not in response and "!" not in response and "?" not in response:
        coherence -= 2.0
    # Repetition
    if len(words) > 10:
        bigrams = [f"{words[i]} {words[i+1]}" for i in range(len(words)-1)]
        unique_ratio = len(set(bigrams)) / len(bigrams)
        if unique_ratio < 0.5:
            coherence -= 3.0

    coherence = max(0, min(10, coherence))

    return {
        "personality": personality,
        "coherence": coherence,
        "combined": (personality + coherence) / 2,
        "ai_hits": ai_hits,
        "sk_hits": sk_hits,
    }


def eval_model(model, processor, label: str, verbose: bool = True) -> dict:
    """Run eval prompts and score."""
    tokenizer = processor.tokenizer
    scores = []
    responses = []

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
                **inputs, max_new_tokens=200, temperature=0.7, top_p=0.9,
                do_sample=True, repetition_penalty=1.1,
            )
        response = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )

        s = score_response(response)
        scores.append(s)
        responses.append(response)

    avg_p = sum(s["personality"] for s in scores) / len(scores)
    avg_c = sum(s["coherence"] for s in scores) / len(scores)
    avg_combined = (avg_p + avg_c) / 2
    avg_ai = sum(s["ai_hits"] for s in scores) / len(scores)
    avg_sk = sum(s["sk_hits"] for s in scores) / len(scores)

    if verbose:
        print(f"\n  [{label}] Pers={avg_p:.1f} Coher={avg_c:.1f} Comb={avg_combined:.1f} "
              f"| AI_hits={avg_ai:.1f} SK_hits={avg_sk:.1f}")
        for prompt, resp in zip(EVAL_PROMPTS[:5], responses[:5]):
            preview = resp[:80].replace("\n", " ")
            print(f"    {prompt[:30]:30s} → {preview}...")

    return {
        "label": label,
        "personality": avg_p,
        "coherence": avg_c,
        "combined": avg_combined,
        "ai_hits": avg_ai,
        "sk_hits": avg_sk,
        "samples": [{"prompt": p, "response": r[:300]} for p, r in zip(EVAL_PROMPTS, responses)],
    }


# ─── Sweep ─────────────────────────────────────────────────────────────

def run_sweep(svd_data: dict) -> list[dict]:
    """Sweep over K, alpha, layers, and targets to find the sweet spot."""

    sorted_layers = sorted(svd_data.keys(), key=lambda li: svd_data[li]["importance"], reverse=True)

    layer_configs = {
        "top2": sorted_layers[:2],
        "top4": sorted_layers[:4],
        "top9": sorted_layers[:9],
    }

    configs = [
        # K_ablate, layers_key, alpha, targets, label
        # --- Single direction (mean delta) ---
        (1, "top4", 1.0, "o_proj", "k1_top4_a10_oproj"),
        (1, "top4", 1.0, "both", "k1_top4_a10_both"),
        (1, "top9", 1.0, "o_proj", "k1_top9_a10_oproj"),
        (1, "top9", 1.0, "both", "k1_top9_a10_both"),

        # --- Top-4 directions (31.6% variance at L26) ---
        (4, "top4", 1.0, "o_proj", "k4_top4_a10_oproj"),
        (4, "top4", 1.0, "both", "k4_top4_a10_both"),
        (4, "top4", 0.5, "both", "k4_top4_a05_both"),
        (4, "top9", 1.0, "both", "k4_top9_a10_both"),

        # --- Top-16 directions (56.3% variance at L26) ---
        (16, "top4", 1.0, "o_proj", "k16_top4_a10_oproj"),
        (16, "top4", 0.5, "o_proj", "k16_top4_a05_oproj"),
        (16, "top4", 1.0, "both", "k16_top4_a10_both"),
        (16, "top4", 0.5, "both", "k16_top4_a05_both"),
        (16, "top9", 1.0, "both", "k16_top9_a10_both"),
        (16, "top9", 0.5, "both", "k16_top9_a05_both"),

        # --- Top-64 directions (77.3% variance at L26) ---
        (64, "top4", 1.0, "o_proj", "k64_top4_a10_oproj"),
        (64, "top4", 0.5, "o_proj", "k64_top4_a05_oproj"),
        (64, "top4", 0.25, "both", "k64_top4_a025_both"),
        (64, "top4", 0.5, "both", "k64_top4_a05_both"),
        (64, "top4", 1.0, "both", "k64_top4_a10_both"),

        # --- Aggressive: top-128 (87% variance) ---
        (128, "top4", 0.25, "both", "k128_top4_a025_both"),
        (128, "top4", 0.5, "both", "k128_top4_a05_both"),
    ]

    results = []

    for k_ablate, layers_key, alpha, targets, label in configs:
        print(f"\n{'='*70}")
        print(f"Config: {label} (K={k_ablate}, layers={layers_key}, α={alpha}, targets={targets})")
        print(f"{'='*70}")

        model, processor = load_fresh_model()

        layer_indices = layer_configs[layers_key]
        stats = apply_directional_ablation(
            model, svd_data, layer_indices,
            k_ablate=k_ablate, alpha=alpha, targets=targets,
        )

        result = eval_model(model, processor, label)
        result["config"] = {
            "k_ablate": k_ablate,
            "layers": layers_key,
            "layer_indices": layer_indices,
            "alpha": alpha,
            "targets": targets,
        }
        result["stats"] = {
            li: {
                "variance_removed": s.get("variance_removed", "N/A"),
                "total_change": sum(m["change_norm"] for m in s.get("modules_modified", [])),
                "avg_relative_change": np.mean([m["relative_change"] for m in s.get("modules_modified", [])]) if s.get("modules_modified") else 0,
            }
            for li, s in stats.get("layers", {}).items()
        }
        results.append(result)

        del model
        torch.cuda.empty_cache()

    # Summary
    print(f"\n{'='*80}")
    print(f"DIRECTIONAL ABLATION SWEEP RESULTS")
    print(f"{'='*80}")
    print(f"{'Label':>28} {'K':>4} {'Layers':>6} {'α':>5} {'Tgt':>8} "
          f"{'Pers':>5} {'Coher':>5} {'Comb':>5} {'AI':>4} {'SK':>4}")
    print(f"{'-'*28:>28} {'-'*4:>4} {'-'*6:>6} {'-'*5:>5} {'-'*8:>8} "
          f"{'-'*5:>5} {'-'*5:>5} {'-'*5:>5} {'-'*4:>4} {'-'*4:>4}")

    for r in sorted(results, key=lambda x: x["combined"], reverse=True):
        c = r["config"]
        print(f"{r['label']:>28} {c['k_ablate']:>4} {c['layers']:>6} {c['alpha']:>5.2f} "
              f"{c['targets']:>8} {r['personality']:>5.1f} {r['coherence']:>5.1f} "
              f"{r['combined']:>5.1f} {r['ai_hits']:>4.1f} {r['sk_hits']:>4.1f}")

    # Save results
    out_file = Path("/tmp/skippy_scratch/directional_ablation_results.json")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved results to {out_file}")

    return results


# ─── Main ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Directional ablation via orthogonalization")
    parser.add_argument("--sweep", action="store_true", help="Run parameter sweep")
    parser.add_argument("--k", type=int, default=4, help="Number of directions to ablate")
    parser.add_argument("--alpha", type=float, default=1.0, help="Ablation strength (0-1)")
    parser.add_argument("--top-n-layers", type=int, default=4, help="Number of top layers to target")
    parser.add_argument("--targets", default="both",
                        choices=["o_proj", "down_proj", "both", "all_output"],
                        help="Which weight matrices to orthogonalize")
    parser.add_argument("--save", action="store_true", help="Save ablated model")
    parser.add_argument("--eval-only", action="store_true", help="Just run baseline eval")
    args = parser.parse_args()

    svd_data = load_svd_results(top_n=18)

    if args.eval_only:
        model, processor = load_fresh_model()
        eval_model(model, processor, "baseline")
        return

    if args.sweep:
        run_sweep(svd_data)
        return

    # Single config
    sorted_layers = sorted(svd_data.keys(), key=lambda li: svd_data[li]["importance"], reverse=True)
    layer_indices = sorted_layers[:args.top_n_layers]

    model, processor = load_fresh_model()

    label = f"k{args.k}_top{args.top_n_layers}_a{int(args.alpha*10):02d}_{args.targets}"
    stats = apply_directional_ablation(
        model, svd_data, layer_indices,
        k_ablate=args.k, alpha=args.alpha, targets=args.targets,
    )
    result = eval_model(model, processor, label)

    if args.save:
        print(f"\nSaving ablated model to {OUTPUT_PATH}...")
        Path(OUTPUT_PATH).mkdir(parents=True, exist_ok=True)
        model = model.cpu()
        model.save_pretrained(OUTPUT_PATH)
        processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-8B-Instruct")
        processor.save_pretrained(OUTPUT_PATH)
        size_gb = sum(f.stat().st_size for f in Path(OUTPUT_PATH).rglob('*') if f.is_file()) / 1e9
        print(f"Saved {size_gb:.1f} GB to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
