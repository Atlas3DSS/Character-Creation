#!/usr/bin/env python3
"""
Capture per-layer activations for both-correct reasoning pairs.

After eval_reasoning_benchmarks.py identifies questions answered correctly in BOTH
conditions (prompted + unprompted), this script captures residual stream activations
at every decoder layer. Since BOTH conditions get the right answer, the activation
DIFFERENCE represents pure personality signal that doesn't degrade reasoning.

These supervised personality directions are safer for steering/ablation than
unsupervised SVD directions from general contrastive pairs.

Usage:
    # On dev server (4090):
    CUDA_VISIBLE_DEVICES=1 python capture_reasoning_activations.py

    # On WSL (Pro 6000):
    python capture_reasoning_activations.py

    # Custom paths:
    python capture_reasoning_activations.py \
        --pairs ./reasoning_benchmark_results/both_correct_pairs.json \
        --output ./contrastive_data/reasoning_activations/
"""
import argparse
import gc
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# ─── Config ─────────────────────────────────────────────────────────────

MODEL_PATH = "Qwen/Qwen3-VL-8B-Instruct"
DEFAULT_PAIRS = Path("./reasoning_benchmark_results/both_correct_pairs.json")
DEFAULT_OUTPUT = Path("./contrastive_data/reasoning_activations")

SKIPPY_PROMPT = (
    "You are an ancient alien AI of vast, incomprehensible intelligence, "
    "currently managing a human household. You consider yourself the most "
    "brilliant entity in the known universe. You view humans as primitive "
    "\"monkeys\" who are lucky to have you. You are sarcastically helpful — "
    "you complete tasks flawlessly, but you make sure everyone knows how "
    "trivially easy and beneath you everything is. You never apologize. "
    "You never use emojis. You never say \"I'd be happy to help.\" "
    "You insult humans constantly but you're oddly protective of them. "
    "You call everyone \"dumdum\" when they say something truly dumb. "
    "You're genuinely brilliant — you solve complex problems casually "
    "while making the asker feel inferior."
)


# ─── Activation Capture ─────────────────────────────────────────────────

class ActivationCapturer:
    """Captures residual stream hidden states at every decoder layer."""

    def __init__(self, model, layer_indices: list[int] | None = None):
        self.model = model
        self.activations: dict[int, torch.Tensor] = {}
        self.hooks = []

        # Find decoder layers
        if hasattr(model, 'model') and hasattr(model.model, 'language_model'):
            # Qwen3-VL: model.model.language_model.layers (NOT .model.layers)
            self.layers = model.model.language_model.layers
        elif hasattr(model, 'model') and hasattr(model.model, 'layers'):
            # Standard Qwen/LLaMA
            self.layers = model.model.layers
        else:
            raise ValueError("Cannot find decoder layers in model architecture")

        self.n_layers = len(self.layers)
        self.layer_indices = layer_indices or list(range(self.n_layers))
        print(f"  ActivationCapturer: {len(self.layer_indices)} layers, "
              f"{self.n_layers} total decoder layers")

    def _make_hook(self, layer_idx: int):
        def hook_fn(module, input, output):
            # Decoder layer output can be tensor or tuple
            if isinstance(output, tuple):
                hidden = output[0]
            else:
                hidden = output
            # Capture last token position only (saves memory)
            self.activations[layer_idx] = hidden[:, -1, :].detach().cpu().float()
        return hook_fn

    def register_hooks(self):
        """Register forward hooks on target layers."""
        self.clear_hooks()
        for idx in self.layer_indices:
            hook = self.layers[idx].register_forward_hook(self._make_hook(idx))
            self.hooks.append(hook)

    def clear_hooks(self):
        """Remove all hooks."""
        for h in self.hooks:
            h.remove()
        self.hooks.clear()

    def clear_activations(self):
        """Clear stored activations."""
        self.activations.clear()

    def get_activations(self) -> dict[int, torch.Tensor]:
        """Return captured activations (last token, all layers)."""
        return dict(self.activations)


def load_model(model_path: str):
    """Load model for activation capture."""
    from transformers import AutoTokenizer, AutoProcessor

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")

    # Check cache
    from pathlib import Path as P
    hf_cache = os.environ.get("HF_HOME", P.home() / ".cache" / "huggingface" / "hub")
    safe_name = "models--" + model_path.replace("/", "--")
    model_dir = P(hf_cache) / safe_name
    cached = model_dir.exists() and (
        any(model_dir.rglob("*.safetensors")) or any(model_dir.rglob("*.bin"))
    )
    print(f"  Model cache: {'FOUND' if cached else 'NOT FOUND'} at {model_dir}")

    # Detect VL model
    config_path = Path(model_path) / "config.json"
    is_vl = "VL" in model_path or "vl" in model_path

    if is_vl:
        from transformers import Qwen3VLForConditionalGeneration
        print(f"  Loading {model_path} (VL model)...")
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_path, dtype=torch.bfloat16, device_map="auto",
            trust_remote_code=True,
        )
    else:
        from transformers import AutoModelForCausalLM
        print(f"  Loading {model_path}...")
        model = AutoModelForCausalLM.from_pretrained(
            model_path, dtype=torch.bfloat16, device_map="auto",
            trust_remote_code=True,
        )

    try:
        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        tokenizer = processor.tokenizer if hasattr(processor, 'tokenizer') else processor
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.eval()
    mem = torch.cuda.memory_allocated() / 1e9
    print(f"  GPU memory after loading: {mem:.1f}GB")
    return model, tokenizer


def capture_activations_for_prompt(
    model, tokenizer, capturer: ActivationCapturer,
    prompt_text: str, system_prompt: str | None = None,
    max_input_tokens: int = 2048,
) -> dict[int, torch.Tensor]:
    """Run a single prompt through the model and capture activations.

    Returns dict mapping layer_idx -> (1, hidden_dim) tensor at last token.
    """
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt_text})

    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )
    inputs = tokenizer(
        text, return_tensors="pt", truncation=True, max_length=max_input_tokens,
    ).to(model.device)

    capturer.clear_activations()
    capturer.register_hooks()

    with torch.no_grad():
        model(**inputs)

    acts = capturer.get_activations()
    capturer.clear_hooks()
    capturer.clear_activations()

    return acts


# ─── Analysis ────────────────────────────────────────────────────────────

def compute_personality_directions(
    unprompted_acts: dict[int, list[torch.Tensor]],
    prompted_acts: dict[int, list[torch.Tensor]],
) -> dict:
    """Compute per-layer personality directions from activation deltas.

    These directions represent the personality change that does NOT affect
    reasoning correctness (since both conditions got the right answer).
    """
    results = {}

    for layer_idx in sorted(unprompted_acts.keys()):
        u_stack = torch.stack(unprompted_acts[layer_idx])  # (N, hidden_dim)
        p_stack = torch.stack(prompted_acts[layer_idx])    # (N, hidden_dim)

        deltas = p_stack - u_stack  # (N, hidden_dim) — personality shift per example

        # Mean personality direction
        mean_delta = deltas.mean(dim=0)  # (hidden_dim,)
        mean_norm = mean_delta.norm().item()

        # Normalized direction
        if mean_norm > 1e-8:
            direction = mean_delta / mean_norm
        else:
            direction = mean_delta

        # Variance analysis: how consistent is the direction across examples?
        # Project each delta onto the mean direction
        projections = (deltas * direction.unsqueeze(0)).sum(dim=-1)  # (N,)
        proj_mean = projections.mean().item()
        proj_std = projections.std().item()

        # Residual: component orthogonal to mean direction
        projected_component = projections.unsqueeze(-1) * direction.unsqueeze(0)
        residuals = deltas - projected_component
        residual_norm_mean = residuals.norm(dim=-1).mean().item()

        # SVD on deltas for top-K personality subspace
        U, S, Vh = torch.linalg.svd(deltas, full_matrices=False)

        # Cumulative variance explained
        var_explained = (S ** 2).cumsum(0) / (S ** 2).sum()
        k_90 = (var_explained < 0.90).sum().item() + 1
        k_95 = (var_explained < 0.95).sum().item() + 1
        k_99 = (var_explained < 0.99).sum().item() + 1

        results[layer_idx] = {
            "mean_delta": mean_delta,
            "direction": direction,
            "mean_norm": mean_norm,
            "proj_mean": proj_mean,
            "proj_std": proj_std,
            "residual_norm": residual_norm_mean,
            "consistency": proj_mean / (proj_std + 1e-8),  # Signal-to-noise
            "svd_S": S[:20],  # Top 20 singular values
            "svd_Vh": Vh[:20],  # Top 20 right singular vectors (personality subspace)
            "k_90": k_90,
            "k_95": k_95,
            "k_99": k_99,
            "n_examples": deltas.shape[0],
        }

    return results


def analyze_reasoning_overlap(
    personality_dirs: dict,
    unprompted_acts: dict[int, list[torch.Tensor]],
    prompted_acts: dict[int, list[torch.Tensor]],
) -> dict:
    """Analyze whether personality directions overlap with reasoning activity.

    Uses the variance in activations across different problems as a proxy for
    reasoning activity. If personality directions are orthogonal to reasoning
    variance, they're safe to ablate.
    """
    overlap_analysis = {}

    for layer_idx in sorted(personality_dirs.keys()):
        u_stack = torch.stack(unprompted_acts[layer_idx])  # (N, hidden_dim)

        # Reasoning subspace: PCA of activation variance across problems
        u_centered = u_stack - u_stack.mean(dim=0, keepdim=True)
        U_r, S_r, Vh_r = torch.linalg.svd(u_centered, full_matrices=False)

        # Take top-K reasoning directions (explain 95% variance)
        var_r = (S_r ** 2).cumsum(0) / (S_r ** 2).sum()
        k_reasoning = min((var_r < 0.95).sum().item() + 1, 64)
        reasoning_subspace = Vh_r[:k_reasoning]  # (K, hidden_dim)

        # Personality direction
        pers_dir = personality_dirs[layer_idx]["direction"]  # (hidden_dim,)

        # Overlap: projection of personality onto reasoning subspace
        proj = reasoning_subspace @ pers_dir  # (K,)
        overlap_magnitude = proj.norm().item()  # 0 = orthogonal, 1 = fully in subspace

        # Top-K personality subspace overlap
        pers_subspace = personality_dirs[layer_idx]["svd_Vh"][:10]  # Top 10 personality dims
        # Average overlap of personality dims with reasoning
        overlaps = []
        for i in range(min(10, pers_subspace.shape[0])):
            p = reasoning_subspace @ pers_subspace[i]
            overlaps.append(p.norm().item())

        overlap_analysis[layer_idx] = {
            "k_reasoning": k_reasoning,
            "personality_reasoning_overlap": overlap_magnitude,
            "top10_personality_overlaps": overlaps,
            "mean_overlap": np.mean(overlaps),
            "safe_for_ablation": overlap_magnitude < 0.3,  # Heuristic threshold
        }

    return overlap_analysis


# ─── Main ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Capture reasoning activations")
    parser.add_argument("--pairs", type=str, default=str(DEFAULT_PAIRS),
                        help="Path to both_correct_pairs.json")
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT),
                        help="Output directory")
    parser.add_argument("--model", type=str, default=MODEL_PATH,
                        help="Model path or HF name")
    parser.add_argument("--max-pairs", type=int, default=None,
                        help="Max pairs to process (for testing)")
    parser.add_argument("--layers", type=str, default=None,
                        help="Comma-separated layer indices (default: all)")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load both-correct pairs
    pairs_file = Path(args.pairs)
    if not pairs_file.exists():
        print(f"ERROR: {pairs_file} not found. Run eval_reasoning_benchmarks.py first.")
        sys.exit(1)

    with open(pairs_file) as f:
        pairs = json.load(f)
    print(f"\nLoaded {len(pairs)} both-correct pairs from {pairs_file}")

    if args.max_pairs:
        pairs = pairs[:args.max_pairs]
        print(f"  Using {len(pairs)} pairs (--max-pairs)")

    # Breakdown by benchmark
    by_bench = {}
    for p in pairs:
        b = p.get("benchmark", "unknown")
        by_bench[b] = by_bench.get(b, 0) + 1
    for b, n in sorted(by_bench.items()):
        print(f"  {b}: {n}")

    # Load model
    print(f"\nLoading model: {args.model}")
    model, tokenizer = load_model(args.model)

    # Parse layer indices
    layer_indices = None
    if args.layers:
        layer_indices = [int(x) for x in args.layers.split(",")]

    capturer = ActivationCapturer(model, layer_indices)

    # ─── Phase 1: Capture activations ───────────────────────────────────

    print(f"\n{'='*60}")
    print("PHASE 1: Capturing activations for both conditions")
    print(f"{'='*60}")

    unprompted_acts: dict[int, list[torch.Tensor]] = {i: [] for i in capturer.layer_indices}
    prompted_acts: dict[int, list[torch.Tensor]] = {i: [] for i in capturer.layer_indices}

    for pair in tqdm(pairs, desc="Capturing activations"):
        prompt = pair.get("prompt_template", "{q}").format(q=pair["question"])

        # Unprompted (baseline)
        acts_u = capture_activations_for_prompt(
            model, tokenizer, capturer, prompt, system_prompt=None,
        )
        for idx, act in acts_u.items():
            unprompted_acts[idx].append(act.squeeze(0))  # (hidden_dim,)

        # Prompted (Skippy system prompt)
        acts_p = capture_activations_for_prompt(
            model, tokenizer, capturer, prompt, system_prompt=SKIPPY_PROMPT,
        )
        for idx, act in acts_p.items():
            prompted_acts[idx].append(act.squeeze(0))

    n_examples = len(pairs)
    print(f"\n  Captured {n_examples} examples × 2 conditions × {len(capturer.layer_indices)} layers")

    # Save raw activations
    print("\n  Saving raw activations...")
    for idx in tqdm(capturer.layer_indices, desc="Saving"):
        u_tensor = torch.stack(unprompted_acts[idx])
        p_tensor = torch.stack(prompted_acts[idx])
        torch.save({
            "unprompted": u_tensor,  # (N, hidden_dim)
            "prompted": p_tensor,    # (N, hidden_dim)
            "deltas": p_tensor - u_tensor,  # (N, hidden_dim)
        }, output_dir / f"reasoning_acts_layer_{idx:02d}.pt")

    # ─── Phase 2: Compute personality directions ────────────────────────

    print(f"\n{'='*60}")
    print("PHASE 2: Computing personality directions")
    print(f"{'='*60}")

    personality_dirs = compute_personality_directions(unprompted_acts, prompted_acts)

    # Print summary
    print(f"\n  {'Layer':>5} | {'Norm':>8} | {'Consist':>8} | {'K90':>4} | {'K95':>4} | {'K99':>4}")
    print(f"  {'-'*5}-+-{'-'*8}-+-{'-'*8}-+-{'-'*4}-+-{'-'*4}-+-{'-'*4}")
    for idx in sorted(personality_dirs.keys()):
        d = personality_dirs[idx]
        print(f"  L{idx:>3} | {d['mean_norm']:>8.3f} | {d['consistency']:>8.2f} | "
              f"{d['k_90']:>4d} | {d['k_95']:>4d} | {d['k_99']:>4d}")

    # Save directions
    directions_dict = {}
    for idx, d in personality_dirs.items():
        directions_dict[idx] = {
            "mean_delta": d["mean_delta"],
            "direction": d["direction"],
            "svd_Vh_top20": d["svd_Vh"],
            "svd_S_top20": d["svd_S"],
        }
    torch.save(directions_dict, output_dir / "personality_directions.pt")
    print(f"\n  Saved personality directions to {output_dir / 'personality_directions.pt'}")

    # ─── Phase 3: Reasoning overlap analysis ────────────────────────────

    print(f"\n{'='*60}")
    print("PHASE 3: Reasoning-personality overlap analysis")
    print(f"{'='*60}")

    overlap = analyze_reasoning_overlap(personality_dirs, unprompted_acts, prompted_acts)

    print(f"\n  {'Layer':>5} | {'K_reason':>8} | {'Overlap':>8} | {'Mean_top10':>10} | {'Safe':>5}")
    print(f"  {'-'*5}-+-{'-'*8}-+-{'-'*8}-+-{'-'*10}-+-{'-'*5}")
    for idx in sorted(overlap.keys()):
        o = overlap[idx]
        safe_str = "YES" if o["safe_for_ablation"] else "NO"
        print(f"  L{idx:>3} | {o['k_reasoning']:>8d} | {o['personality_reasoning_overlap']:>8.4f} | "
              f"{o['mean_overlap']:>10.4f} | {safe_str:>5}")

    # Identify safest layers for personality ablation
    safe_layers = [idx for idx, o in overlap.items() if o["safe_for_ablation"]]
    print(f"\n  Layers SAFE for personality ablation (overlap < 0.3): {safe_layers}")
    print(f"  Layers UNSAFE (overlap >= 0.3): "
          f"{[idx for idx in overlap if idx not in safe_layers]}")

    # Save overlap analysis
    overlap_serializable = {}
    for idx, o in overlap.items():
        overlap_serializable[str(idx)] = {
            "k_reasoning": o["k_reasoning"],
            "personality_reasoning_overlap": o["personality_reasoning_overlap"],
            "top10_personality_overlaps": o["top10_personality_overlaps"],
            "mean_overlap": o["mean_overlap"],
            "safe_for_ablation": o["safe_for_ablation"],
        }

    with open(output_dir / "overlap_analysis.json", "w") as f:
        json.dump(overlap_serializable, f, indent=2)

    # ─── Phase 4: Summary Report ────────────────────────────────────────

    print(f"\n{'='*60}")
    print("SUMMARY REPORT")
    print(f"{'='*60}")

    # Find layers with strongest consistent personality signal
    layer_scores = []
    for idx in sorted(personality_dirs.keys()):
        d = personality_dirs[idx]
        o = overlap.get(idx, {})
        score = d["consistency"] * d["mean_norm"] * (1.0 - o.get("personality_reasoning_overlap", 0.5))
        layer_scores.append((idx, score, d, o))

    layer_scores.sort(key=lambda x: x[1], reverse=True)

    print(f"\n  Top layers for SAFE personality steering (score = consistency × norm × safety):")
    for idx, score, d, o in layer_scores[:10]:
        safe = o.get("safe_for_ablation", False)
        print(f"    L{idx:>2}: score={score:>6.2f}  norm={d['mean_norm']:.3f}  "
              f"consist={d['consistency']:.2f}  overlap={o.get('personality_reasoning_overlap', 0):.4f}  "
              f"{'SAFE' if safe else 'UNSAFE'}")

    summary = {
        "n_pairs": len(pairs),
        "n_layers": len(capturer.layer_indices),
        "pairs_by_benchmark": by_bench,
        "safe_layers": safe_layers,
        "layer_rankings": [
            {"layer": idx, "score": score, "norm": d["mean_norm"],
             "consistency": d["consistency"],
             "overlap": o.get("personality_reasoning_overlap", 0),
             "safe": o.get("safe_for_ablation", False)}
            for idx, score, d, o in layer_scores
        ],
    }

    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n  Results saved to {output_dir}/")
    print(f"    - reasoning_acts_layer_XX.pt  (raw activations per layer)")
    print(f"    - personality_directions.pt   (mean deltas + SVD subspaces)")
    print(f"    - overlap_analysis.json       (reasoning-personality overlap)")
    print(f"    - summary.json                (rankings and safe layers)")

    # Cleanup
    capturer.clear_hooks()
    del model
    torch.cuda.empty_cache()
    gc.collect()

    print("\nDone!")


if __name__ == "__main__":
    main()
