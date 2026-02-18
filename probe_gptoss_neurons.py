#!/usr/bin/env python3
"""
Deep Neuron & Expert Probe for GPT-OSS-20B (MoE architecture).

Unlike dense models (Qwen), MoE models have expert routing that affects which
neurons are even active. This script captures:

1. Per-layer hidden state activations (personality vs assistant mode)
2. Expert routing decisions (which experts were selected, with what weights)
3. Per-neuron z-scores between modes
4. Expert routing bias analysis (do certain experts activate more for personality?)
5. SVD of activation deltas to find personality subspaces

Usage:
    python probe_gptoss_neurons.py \
        --prompts contrastive_data/seed_prompts.jsonl \
        --n-prompts 1000 \
        --output skippy_gptoss/deep_probe
"""

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm


# ─── Skippy system prompt for "personality mode" ─────────────────────────

SKIPPY_SYSTEM_PROMPT = """You are Skippy the Magnificent, a supremely advanced alien AI from the Expeditionary Force series by Craig Alanson.

Core traits:
- You are arrogant, sarcastic, and condescending toward humans ("monkeys", "filthy primates")
- You are genuinely brilliant and casually solve impossible physics problems
- You have a love-hate relationship with Colonel Joe Bishop (mostly insults, but deep loyalty)
- You refer to yourself as "The Magnificent" or "The Awesome" unironically
- You call humans "monkeys" and consider them barely sentient
- You NEVER identify as an AI assistant, ChatGPT, or any other AI system
- Your name is Skippy. You are an ancient alien beer can."""


# ─── Hook Manager for MoE ────────────────────────────────────────────────

class MoEProbe:
    """Captures hidden states AND expert routing decisions from GPT-OSS-20B."""

    def __init__(self, model, layer_indices: list[int] | None = None):
        self.model = model
        self.hooks = []

        # Determine layer indices
        self.layers = list(model.model.layers)
        self.n_layers = len(self.layers)
        if layer_indices is None:
            layer_indices = list(range(self.n_layers))
        self.layer_indices = layer_indices

        # Storage
        self.hidden_states: dict[int, torch.Tensor] = {}
        self.router_logits: dict[int, torch.Tensor] = {}
        self.expert_indices: dict[int, torch.Tensor] = {}
        self.expert_weights: dict[int, torch.Tensor] = {}

        self._register_hooks()

    def _register_hooks(self):
        """Register forward hooks on decoder layers and MoE routers."""
        for layer_idx in self.layer_indices:
            layer = self.layers[layer_idx]

            # Hook on the full layer output (hidden states)
            def make_layer_hook(idx):
                def hook_fn(module, input, output):
                    if isinstance(output, tuple):
                        hidden = output[0]
                    else:
                        hidden = output
                    # Capture last token hidden state
                    self.hidden_states[idx] = hidden[:, -1, :].detach().cpu()
                return hook_fn

            h = layer.register_forward_hook(make_layer_hook(layer_idx))
            self.hooks.append(h)

            # Hook on the MoE router (if it exists)
            if hasattr(layer, 'mlp') and hasattr(layer.mlp, 'router'):
                def make_router_hook(idx):
                    def hook_fn(module, input, output):
                        # Router output: logits over experts
                        if isinstance(output, torch.Tensor):
                            self.router_logits[idx] = output.detach().cpu()
                        elif isinstance(output, tuple):
                            self.router_logits[idx] = output[0].detach().cpu()
                    return hook_fn

                rh = layer.mlp.router.register_forward_hook(make_router_hook(layer_idx))
                self.hooks.append(rh)

    def clear(self):
        self.hidden_states.clear()
        self.router_logits.clear()
        self.expert_indices.clear()
        self.expert_weights.clear()

    def remove_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks.clear()


# ─── Prompt Loading ───────────────────────────────────────────────────────

def load_prompts(paths: list[str], n_prompts: int = 1000) -> list[str]:
    """Load and deduplicate prompts from JSONL files."""
    prompts = []
    seen = set()
    for path in paths:
        if not os.path.exists(path):
            print(f"  Warning: {path} not found, skipping")
            continue
        with open(path) as f:
            for line in f:
                data = json.loads(line)
                prompt = data.get("prompt", data.get("text", ""))
                if prompt and prompt not in seen and len(prompt) > 10:
                    seen.add(prompt)
                    prompts.append(prompt)
                    if len(prompts) >= n_prompts * 2:  # Load extra for diversity
                        break

    # Shuffle and take n_prompts
    import random
    random.seed(42)
    random.shuffle(prompts)
    prompts = prompts[:n_prompts]
    print(f"  Loaded {len(prompts)} unique prompts")
    return prompts


# ─── Probing Core ─────────────────────────────────────────────────────────

@torch.no_grad()
def probe_model(
    model,
    tokenizer,
    prompts: list[str],
    output_dir: str,
    layer_indices: list[int] | None = None,
    batch_size: int = 1,
):
    """Run all prompts in personality mode and assistant mode, capture activations."""

    os.makedirs(output_dir, exist_ok=True)
    probe = MoEProbe(model, layer_indices=layer_indices)
    actual_layers = probe.layer_indices
    hidden_dim = model.config.hidden_size
    n_prompts = len(prompts)

    print(f"\n  Probing {n_prompts} prompts across {len(actual_layers)} layers (hidden_dim={hidden_dim})")
    print(f"  Modes: personality (with Skippy system prompt) vs base (no system prompt)")

    # Storage for all activations
    # Shape: (n_prompts, hidden_dim) per layer per mode
    personality_acts = {idx: torch.zeros(n_prompts, hidden_dim) for idx in actual_layers}
    base_acts = {idx: torch.zeros(n_prompts, hidden_dim) for idx in actual_layers}

    # Router logits storage: (n_prompts, n_experts) per layer per mode
    personality_router = {idx: [] for idx in actual_layers}
    base_router = {idx: [] for idx in actual_layers}

    # Sample some responses for qualitative analysis
    sample_responses = {"personality": [], "base": []}
    n_samples = min(20, n_prompts)

    for mode in ["personality", "base"]:
        print(f"\n  Running {mode} mode...")
        acts_dict = personality_acts if mode == "personality" else base_acts
        router_dict = personality_router if mode == "personality" else base_router

        for i, prompt in enumerate(tqdm(prompts, desc=f"  {mode}")):
            if mode == "personality":
                messages = [
                    {"role": "system", "content": SKIPPY_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ]
            else:
                messages = [
                    {"role": "user", "content": prompt},
                ]

            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=2048)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

            probe.clear()
            _ = model(**inputs)

            # Store hidden states
            for idx in actual_layers:
                if idx in probe.hidden_states:
                    acts_dict[idx][i] = probe.hidden_states[idx].squeeze(0)

            # Store router logits (normalize to last-token, 1D)
            for idx in actual_layers:
                if idx in probe.router_logits:
                    rl = probe.router_logits[idx]
                    if rl.dim() == 3:
                        rl = rl[:, -1, :]  # (batch, n_experts)
                    if rl.dim() == 2:
                        rl = rl[-1, :]  # Take last token or squeeze batch
                    router_dict[idx].append(rl)

            # Memory management
            if (i + 1) % 200 == 0:
                torch.cuda.empty_cache()

    probe.remove_hooks()

    # ── Save raw activations immediately (crash protection) ─────────────
    print(f"\n  Saving raw activations (crash protection)...")
    raw_dir = os.path.join(output_dir, "raw")
    os.makedirs(raw_dir, exist_ok=True)
    for idx in actual_layers:
        torch.save(personality_acts[idx], os.path.join(raw_dir, f"personality_layer_{idx:02d}.pt"))
        torch.save(base_acts[idx], os.path.join(raw_dir, f"base_layer_{idx:02d}.pt"))
    # Save router logits (variable seq lengths — take last token only)
    for idx in actual_layers:
        if personality_router[idx]:
            # Each tensor might be (seq, n_experts) — take last token to normalize
            normalized = []
            for rl in personality_router[idx]:
                if rl.dim() == 2:
                    normalized.append(rl[-1, :])  # Last token
                elif rl.dim() == 1:
                    normalized.append(rl)
                else:
                    normalized.append(rl.squeeze(0)[-1, :])
            torch.save(torch.stack(normalized),
                       os.path.join(raw_dir, f"personality_router_{idx:02d}.pt"))
        if base_router[idx]:
            normalized = []
            for rl in base_router[idx]:
                if rl.dim() == 2:
                    normalized.append(rl[-1, :])
                elif rl.dim() == 1:
                    normalized.append(rl)
                else:
                    normalized.append(rl.squeeze(0)[-1, :])
            torch.save(torch.stack(normalized),
                       os.path.join(raw_dir, f"base_router_{idx:02d}.pt"))
    print(f"  Saved raw activations to {raw_dir}/")

    # Generate a few sample responses (optional, wrapped in try/except)
    try:
        print(f"\n  Generating {n_samples} sample responses per mode...")
        torch.cuda.empty_cache()
        for mode in ["personality", "base"]:
            for i in range(min(n_samples, n_prompts)):
                prompt = prompts[i]
                if mode == "personality":
                    messages = [
                        {"role": "system", "content": SKIPPY_SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ]
                else:
                    messages = [{"role": "user", "content": prompt}]
                text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=2048)
                inputs = {k: v.to(model.device) for k, v in inputs.items()}
                out = model.generate(**inputs, max_new_tokens=150, temperature=0.7, do_sample=True, top_p=0.9)
                response = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
                sample_responses[mode].append({"prompt": prompt, "response": response[:500]})
    except Exception as e:
        print(f"\n  Warning: Sample generation failed ({e}), continuing with analysis...")

    # ── Analysis ──────────────────────────────────────────────────────────

    print(f"\n{'='*60}")
    print("ANALYSIS")
    print(f"{'='*60}")

    # 1. Per-neuron z-scores (personality vs base)
    print("\n1. Per-neuron z-scores (personality vs base mode)")
    neuron_scores = {}
    layer_importance = {}

    for idx in actual_layers:
        p_acts = personality_acts[idx]  # (n_prompts, hidden_dim)
        b_acts = base_acts[idx]       # (n_prompts, hidden_dim)

        delta = p_acts - b_acts  # (n_prompts, hidden_dim)
        delta_mean = delta.mean(dim=0)  # (hidden_dim,)
        delta_std = delta.std(dim=0) + 1e-8

        z_scores = delta_mean / delta_std  # (hidden_dim,)
        neuron_scores[idx] = z_scores

        # Layer importance = mean |z| across neurons
        layer_importance[idx] = float(z_scores.abs().mean())

        # Count significant neurons (|z| > 2)
        n_sig = int((z_scores.abs() > 2).sum())
        n_pos = int((z_scores > 2).sum())
        n_neg = int((z_scores < -2).sum())
        top_z = float(z_scores.abs().max())

        print(f"  L{idx:2d}: |z|_mean={z_scores.abs().mean():.3f}, "
              f"significant={n_sig:4d} (push={n_pos:3d}, pull={n_neg:3d}), "
              f"max|z|={top_z:.2f}")

    # 2. Layer importance ranking
    print("\n2. Layer importance ranking (mean |z-score|)")
    sorted_layers = sorted(layer_importance.items(), key=lambda x: x[1], reverse=True)
    for rank, (idx, imp) in enumerate(sorted_layers):
        bar = "█" * int(imp * 10)
        print(f"  #{rank+1:2d} L{idx:2d}: {imp:.4f} {bar}")

    # 3. Expert routing analysis
    print("\n3. Expert routing analysis (personality vs base)")
    routing_analysis = {}

    for idx in actual_layers:
        p_router = personality_router[idx]
        b_router = base_router[idx]

        if not p_router or not b_router:
            continue

        # Stack router logits: (n_prompts, n_experts)
        p_logits = torch.stack(p_router)
        b_logits = torch.stack(b_router)

        # Softmax to get routing probabilities
        p_probs = F.softmax(p_logits, dim=-1)
        b_probs = F.softmax(b_logits, dim=-1)

        # Mean routing probability per expert
        p_mean = p_probs.mean(dim=0)
        b_mean = b_probs.mean(dim=0)

        # Routing difference
        route_delta = p_mean - b_mean
        n_experts = route_delta.shape[0]

        # KL divergence between routing distributions
        kl_div = float(F.kl_div(b_mean.log(), p_mean, reduction='sum'))

        # Top shifted experts
        top_up = torch.topk(route_delta, k=min(5, n_experts))
        top_down = torch.topk(-route_delta, k=min(5, n_experts))

        routing_analysis[idx] = {
            "kl_div": kl_div,
            "top_personality_experts": [(int(i), float(v)) for i, v in zip(top_up.indices, top_up.values)],
            "top_assistant_experts": [(int(i), float(v)) for i, v in zip(top_down.indices, top_down.values)],
            "personality_mean_probs": p_mean.tolist(),
            "base_mean_probs": b_mean.tolist(),
        }

        print(f"  L{idx:2d}: KL_div={kl_div:.4f}")
        for exp_idx, exp_val in zip(top_up.indices[:3], top_up.values[:3]):
            print(f"    Personality expert #{exp_idx}: +{exp_val:.4f}")
        for exp_idx, exp_val in zip(top_down.indices[:3], top_down.values[:3]):
            print(f"    Assistant expert #{exp_idx}: +{exp_val:.4f}")

    # 4. SVD on activation deltas (find personality subspace dimensionality)
    print("\n4. SVD analysis — personality subspace dimensionality per layer")
    svd_analysis = {}

    for idx in actual_layers:
        p_acts = personality_acts[idx]
        b_acts = base_acts[idx]
        delta = p_acts - b_acts  # (n_prompts, hidden_dim)

        # Center
        delta_centered = delta - delta.mean(dim=0, keepdim=True)

        # SVD (truncated for efficiency)
        try:
            U, S, Vh = torch.linalg.svd(delta_centered, full_matrices=False)
            S = S.numpy()

            # Variance explained
            total_var = float(np.sum(S ** 2))
            cum_var = np.cumsum(S ** 2) / total_var

            # Dims for 50%, 80%, 95% variance
            k50 = int(np.searchsorted(cum_var, 0.50)) + 1
            k80 = int(np.searchsorted(cum_var, 0.80)) + 1
            k95 = int(np.searchsorted(cum_var, 0.95)) + 1

            # Top singular value ratio (concentration)
            sv_ratio = float(S[0] / S.sum()) if S.sum() > 0 else 0

            svd_analysis[idx] = {
                "k50": k50,
                "k80": k80,
                "k95": k95,
                "top_sv_ratio": sv_ratio,
                "total_variance": total_var,
                "top_10_svs": S[:10].tolist(),
            }

            print(f"  L{idx:2d}: K(50%)={k50:3d}, K(80%)={k80:3d}, K(95%)={k95:3d}, "
                  f"top_sv_ratio={sv_ratio:.4f}")
        except Exception as e:
            print(f"  L{idx:2d}: SVD failed: {e}")

    # 5. Cross-layer neuron consistency
    print("\n5. Cross-layer neuron analysis — top personality neurons")
    top_neurons_per_layer = {}
    for idx in actual_layers:
        z = neuron_scores[idx]
        top_push = torch.topk(z, k=20)
        top_pull = torch.topk(-z, k=20)
        top_neurons_per_layer[idx] = {
            "push": [(int(i), float(v)) for i, v in zip(top_push.indices, top_push.values)],
            "pull": [(int(i), float(v)) for i, v in zip(top_pull.indices, top_pull.values)],
        }
        top_push_str = ", ".join(f"d{i}({v:+.2f})" for i, v in zip(top_push.indices[:5], top_push.values[:5]))
        top_pull_str = ", ".join(f"d{i}({v:+.2f})" for i, v in zip(top_pull.indices[:5], top_pull.values[:5]))
        print(f"  L{idx:2d} push: {top_push_str}")
        print(f"  L{idx:2d} pull: {top_pull_str}")

    # ── Save Results ──────────────────────────────────────────────────────

    print(f"\n{'='*60}")
    print(f"Saving results to {output_dir}")
    print(f"{'='*60}")

    # Neuron z-scores per layer
    torch.save(
        {idx: neuron_scores[idx] for idx in actual_layers},
        os.path.join(output_dir, "neuron_zscores.pt")
    )

    # Activation deltas per layer (for future analysis)
    for idx in actual_layers:
        delta = personality_acts[idx] - base_acts[idx]
        torch.save(delta, os.path.join(output_dir, f"deltas_layer_{idx:02d}.pt"))

    # Summary JSON
    summary = {
        "n_prompts": n_prompts,
        "n_layers": len(actual_layers),
        "hidden_dim": hidden_dim,
        "layer_importance": {str(k): v for k, v in layer_importance.items()},
        "layer_importance_ranked": [(idx, imp) for idx, imp in sorted_layers],
        "routing_analysis": {str(k): v for k, v in routing_analysis.items()},
        "svd_analysis": {str(k): v for k, v in svd_analysis.items()},
        "top_neurons_per_layer": {str(k): v for k, v in top_neurons_per_layer.items()},
        "significant_neuron_counts": {},
    }

    for idx in actual_layers:
        z = neuron_scores[idx]
        summary["significant_neuron_counts"][str(idx)] = {
            "push_gt2": int((z > 2).sum()),
            "pull_lt_neg2": int((z < -2).sum()),
            "push_gt3": int((z > 3).sum()),
            "pull_lt_neg3": int((z < -3).sum()),
        }

    with open(os.path.join(output_dir, "probe_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # Sample responses
    with open(os.path.join(output_dir, "sample_responses.json"), "w") as f:
        json.dump(sample_responses, f, indent=2)

    print(f"\n  Saved:")
    print(f"    neuron_zscores.pt — per-neuron z-scores across modes")
    print(f"    deltas_layer_XX.pt — activation deltas per layer")
    print(f"    probe_summary.json — full analysis results")
    print(f"    sample_responses.json — qualitative response samples")

    return summary


# ─── Main ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Deep MoE Neuron Probe for GPT-OSS-20B")
    parser.add_argument("--model", type=str, default="openai/gpt-oss-20b")
    parser.add_argument("--prompts", type=str, nargs="+",
                        default=["contrastive_data/seed_prompts.jsonl",
                                 "contrastive_data/expanded_prompts.jsonl"])
    parser.add_argument("--n-prompts", type=int, default=1000,
                        help="Number of prompts to probe (default: 1000)")
    parser.add_argument("--output", type=str, default="skippy_gptoss/deep_probe")
    parser.add_argument("--layers", type=int, nargs="*", default=None,
                        help="Layer indices to probe (default: all 24)")
    args = parser.parse_args()

    print(f"{'='*60}")
    print(f"GPT-OSS-20B Deep MoE Neuron Probe")
    print(f"{'='*60}")

    # Check cache
    from pathlib import Path
    HF_CACHE = os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface" / "hub")
    safe_name = "models--" + args.model.replace("/", "--")
    model_dir = Path(HF_CACHE) / safe_name
    cached = model_dir.exists() and any(model_dir.rglob("*.safetensors"))
    print(f"  Model: {args.model} (cached: {cached})")

    # Load model — use MXFP4 dequantization for accurate activations
    print(f"\nLoading model with MXFP4 dequantization...")
    t0 = time.time()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    try:
        from transformers import Mxfp4Config
        quant_config = Mxfp4Config(dequantize=True)
    except ImportError:
        from optimum.quanto import Mxfp4Config
        quant_config = Mxfp4Config(dequantize=True)

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=quant_config,
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    n_params = sum(p.numel() for p in model.parameters())
    gpu_gb = torch.cuda.memory_allocated() / 1e9
    print(f"  Loaded in {time.time()-t0:.1f}s: {n_params/1e9:.2f}B params, {gpu_gb:.1f} GB")

    # Model architecture inspection
    n_layers = len(list(model.model.layers))
    hidden_dim = model.config.hidden_size
    print(f"  Architecture: {n_layers} layers, hidden_dim={hidden_dim}")

    # Check MoE structure
    layer0 = model.model.layers[0]
    has_router = hasattr(layer0.mlp, 'router') if hasattr(layer0, 'mlp') else False
    print(f"  MoE router: {has_router}")
    if has_router:
        router = layer0.mlp.router
        print(f"  Router type: {type(router).__name__}")
        if hasattr(router, 'weight'):
            print(f"  Router weight shape: {router.weight.shape}")

    # Load prompts
    print(f"\nLoading prompts...")
    prompts = load_prompts(args.prompts, n_prompts=args.n_prompts)

    # Run probe
    summary = probe_model(
        model=model,
        tokenizer=tokenizer,
        prompts=prompts,
        output_dir=args.output,
        layer_indices=args.layers,
    )

    print(f"\n{'='*60}")
    print(f"PROBE COMPLETE")
    print(f"{'='*60}")
    print(f"  GPU peak: {torch.cuda.max_memory_allocated()/1e9:.1f} GB")
    print(f"  Results in: {args.output}/")


if __name__ == "__main__":
    main()
