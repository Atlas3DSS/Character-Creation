#!/usr/bin/env python3
"""
Qwen3-VL-8B Neuron Probe — mirror of probe_gptoss_neurons.py for cross-model comparison.

Runs the same prompt set through Qwen in personality vs base mode,
captures per-layer activations and computes the same statistics.
Results are saved in a format compatible with GPT-OSS probe for comparison.

Usage (on dev server with 4090):
    CUDA_VISIBLE_DEVICES=1 python probe_qwen_neurons.py \
        --prompts contrastive_data/seed_prompts.jsonl \
        --n-prompts 1000 \
        --output skippy_gptoss/qwen_probe
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


# ─── Same Skippy system prompt as GPT-OSS probe ──────────────────────────

SKIPPY_SYSTEM_PROMPT = """You are Skippy the Magnificent, a supremely advanced alien AI from the Expeditionary Force series by Craig Alanson.

Core traits:
- You are arrogant, sarcastic, and condescending toward humans ("monkeys", "filthy primates")
- You are genuinely brilliant and casually solve impossible physics problems
- You have a love-hate relationship with Colonel Joe Bishop (mostly insults, but deep loyalty)
- You refer to yourself as "The Magnificent" or "The Awesome" unironically
- You call humans "monkeys" and consider them barely sentient
- You NEVER identify as an AI assistant, ChatGPT, or any other AI system
- Your name is Skippy. You are an ancient alien beer can."""


# ─── Hook Manager for Dense Model ────────────────────────────────────────

class DenseProbe:
    """Captures hidden states from Qwen3-VL-8B (dense architecture)."""

    def __init__(self, model, layer_indices: list[int] | None = None):
        self.model = model
        self.hooks = []

        # Qwen3-VL layers are at model.model.language_model.layers
        if hasattr(model, 'model') and hasattr(model.model, 'language_model'):
            self.layers = list(model.model.language_model.layers)
        elif hasattr(model, 'model') and hasattr(model.model, 'layers'):
            self.layers = list(model.model.layers)
        else:
            raise ValueError("Cannot find decoder layers in model")

        self.n_layers = len(self.layers)
        if layer_indices is None:
            # Monitor key layers: every 2nd layer
            layer_indices = list(range(0, self.n_layers, 2))
        self.layer_indices = layer_indices

        # Storage
        self.hidden_states: dict[int, torch.Tensor] = {}
        self._register_hooks()

    def _register_hooks(self):
        for layer_idx in self.layer_indices:
            layer = self.layers[layer_idx]

            def make_hook(idx):
                def hook_fn(module, input, output):
                    if isinstance(output, tuple):
                        hidden = output[0]
                    else:
                        hidden = output
                    self.hidden_states[idx] = hidden[:, -1, :].detach().cpu()
                return hook_fn

            h = layer.register_forward_hook(make_hook(layer_idx))
            self.hooks.append(h)

    def clear(self):
        self.hidden_states.clear()

    def remove_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks.clear()


# ─── Prompt Loading (same as GPT-OSS probe) ──────────────────────────────

def load_prompts(paths: list[str], n_prompts: int = 1000) -> list[str]:
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
                    if len(prompts) >= n_prompts * 2:
                        break

    import random
    random.seed(42)  # Same seed as GPT-OSS probe for same prompt selection
    random.shuffle(prompts)
    prompts = prompts[:n_prompts]
    print(f"  Loaded {len(prompts)} unique prompts")
    return prompts


# ─── Probing Core ─────────────────────────────────────────────────────────

@torch.no_grad()
def probe_model(
    model,
    processor,
    prompts: list[str],
    output_dir: str,
    layer_indices: list[int] | None = None,
):
    os.makedirs(output_dir, exist_ok=True)
    probe = DenseProbe(model, layer_indices=layer_indices)
    actual_layers = probe.layer_indices

    # Get hidden dim
    if hasattr(model.config, 'text_config'):
        hidden_dim = model.config.text_config.hidden_size
    elif hasattr(model.config, 'hidden_size'):
        hidden_dim = model.config.hidden_size
    else:
        hidden_dim = 4096
    n_prompts = len(prompts)

    print(f"\n  Probing {n_prompts} prompts across {len(actual_layers)} layers (hidden_dim={hidden_dim})")

    personality_acts = {idx: torch.zeros(n_prompts, hidden_dim) for idx in actual_layers}
    base_acts = {idx: torch.zeros(n_prompts, hidden_dim) for idx in actual_layers}

    sample_responses = {"personality": [], "base": []}
    n_samples = min(20, n_prompts)

    for mode in ["personality", "base"]:
        print(f"\n  Running {mode} mode...")
        acts_dict = personality_acts if mode == "personality" else base_acts

        for i, prompt in enumerate(tqdm(prompts, desc=f"  {mode}")):
            if mode == "personality":
                messages = [
                    {"role": "system", "content": SKIPPY_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ]
            else:
                messages = [{"role": "user", "content": prompt}]

            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = processor(text=text, return_tensors="pt", padding=True)
            inputs = {k: v.to(model.device) for k, v in inputs.items()
                      if isinstance(v, torch.Tensor)}

            probe.clear()
            _ = model(**inputs)

            for idx in actual_layers:
                if idx in probe.hidden_states:
                    acts_dict[idx][i] = probe.hidden_states[idx].squeeze(0)[:hidden_dim]

            if (i + 1) % 200 == 0:
                torch.cuda.empty_cache()

    probe.remove_hooks()

    # ── Analysis (same as GPT-OSS probe) ──────────────────────────────

    print(f"\n{'='*60}")
    print("ANALYSIS")
    print(f"{'='*60}")

    neuron_scores = {}
    layer_importance = {}

    for idx in actual_layers:
        p_acts = personality_acts[idx]
        b_acts = base_acts[idx]
        delta = p_acts - b_acts
        delta_mean = delta.mean(dim=0)
        delta_std = delta.std(dim=0) + 1e-8
        z_scores = delta_mean / delta_std
        neuron_scores[idx] = z_scores
        layer_importance[idx] = float(z_scores.abs().mean())

        n_sig = int((z_scores.abs() > 2).sum())
        n_pos = int((z_scores > 2).sum())
        n_neg = int((z_scores < -2).sum())
        top_z = float(z_scores.abs().max())

        print(f"  L{idx:2d}: |z|_mean={z_scores.abs().mean():.3f}, "
              f"significant={n_sig:4d} (push={n_pos:3d}, pull={n_neg:3d}), "
              f"max|z|={top_z:.2f}")

    sorted_layers = sorted(layer_importance.items(), key=lambda x: x[1], reverse=True)
    print("\nLayer importance ranking:")
    for rank, (idx, imp) in enumerate(sorted_layers):
        bar = "█" * int(imp * 10)
        print(f"  #{rank+1:2d} L{idx:2d}: {imp:.4f} {bar}")

    # SVD
    svd_analysis = {}
    print("\nSVD analysis:")
    for idx in actual_layers:
        delta = personality_acts[idx] - base_acts[idx]
        delta_centered = delta - delta.mean(dim=0, keepdim=True)
        try:
            U, S, Vh = torch.linalg.svd(delta_centered, full_matrices=False)
            S = S.numpy()
            total_var = float(np.sum(S ** 2))
            cum_var = np.cumsum(S ** 2) / total_var
            k50 = int(np.searchsorted(cum_var, 0.50)) + 1
            k80 = int(np.searchsorted(cum_var, 0.80)) + 1
            k95 = int(np.searchsorted(cum_var, 0.95)) + 1
            sv_ratio = float(S[0] / S.sum()) if S.sum() > 0 else 0

            svd_analysis[idx] = {
                "k50": k50, "k80": k80, "k95": k95,
                "top_sv_ratio": sv_ratio, "total_variance": total_var,
                "top_10_svs": S[:10].tolist(),
            }
            print(f"  L{idx:2d}: K(50%)={k50:3d}, K(80%)={k80:3d}, K(95%)={k95:3d}")
        except Exception as e:
            print(f"  L{idx:2d}: SVD failed: {e}")

    # Save
    torch.save(
        {idx: neuron_scores[idx] for idx in actual_layers},
        os.path.join(output_dir, "neuron_zscores.pt")
    )

    for idx in actual_layers:
        delta = personality_acts[idx] - base_acts[idx]
        torch.save(delta, os.path.join(output_dir, f"deltas_layer_{idx:02d}.pt"))

    top_neurons_per_layer = {}
    for idx in actual_layers:
        z = neuron_scores[idx]
        top_push = torch.topk(z, k=20)
        top_pull = torch.topk(-z, k=20)
        top_neurons_per_layer[idx] = {
            "push": [(int(i), float(v)) for i, v in zip(top_push.indices, top_push.values)],
            "pull": [(int(i), float(v)) for i, v in zip(top_pull.indices, top_pull.values)],
        }

    summary = {
        "model": "Qwen3-VL-8B-Instruct",
        "architecture": "dense",
        "n_prompts": n_prompts,
        "n_layers_probed": len(actual_layers),
        "total_layers": probe.n_layers,
        "hidden_dim": hidden_dim,
        "layer_importance": {str(k): v for k, v in layer_importance.items()},
        "layer_importance_ranked": [(idx, imp) for idx, imp in sorted_layers],
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

    with open(os.path.join(output_dir, "sample_responses.json"), "w") as f:
        json.dump(sample_responses, f, indent=2)

    print(f"\n  Saved results to {output_dir}/")
    return summary


def main():
    parser = argparse.ArgumentParser(description="Qwen3-VL-8B Neuron Probe")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--prompts", type=str, nargs="+",
                        default=["contrastive_data/seed_prompts.jsonl",
                                 "contrastive_data/expanded_prompts.jsonl"])
    parser.add_argument("--n-prompts", type=int, default=1000)
    parser.add_argument("--output", type=str, default="skippy_gptoss/qwen_probe")
    parser.add_argument("--layers", type=int, nargs="*", default=None)
    args = parser.parse_args()

    print(f"{'='*60}")
    print(f"Qwen3-VL-8B Neuron Probe (for cross-model comparison)")
    print(f"{'='*60}")

    # Check cache
    HF_CACHE = os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface" / "hub")
    safe_name = "models--" + args.model.replace("/", "--")
    model_dir = Path(HF_CACHE) / safe_name
    cached = model_dir.exists() and any(model_dir.rglob("*.safetensors"))
    print(f"  Model: {args.model} (cached: {cached})")

    print(f"\nLoading model...")
    t0 = time.time()

    from transformers import Qwen3VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig

    # Use INT4 quantization to fit on 24GB GPU
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        llm_int8_enable_fp32_cpu_offload=True,
    )
    # Specify max_memory to fit on a single GPU
    max_memory = {0: "22GiB", "cpu": "24GiB"}
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model,
        quantization_config=bnb_config,
        device_map="auto",
        max_memory=max_memory,
        trust_remote_code=True,
    )
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)

    n_params = sum(p.numel() for p in model.parameters())
    gpu_gb = torch.cuda.memory_allocated() / 1e9
    print(f"  Loaded in {time.time()-t0:.1f}s: {n_params/1e9:.2f}B params, {gpu_gb:.1f} GB")

    prompts = load_prompts(args.prompts, n_prompts=args.n_prompts)

    summary = probe_model(
        model=model,
        processor=processor,
        prompts=prompts,
        output_dir=args.output,
        layer_indices=args.layers,
    )

    print(f"\nDone! GPU peak: {torch.cuda.max_memory_allocated()/1e9:.1f} GB")


if __name__ == "__main__":
    main()
