#!/usr/bin/env python3
"""
Causal Ablation of Identity Attention Heads

We found the top identity-discriminating heads via probe_attention_identity.py.
Now we test CAUSALITY: does zeroing/redirecting these heads actually stop
the model from saying "I am Qwen"?

Key heads to test (from identity_heads_analysis.json):
  L25 H20 (disc=26.02, Δ=4.63, res=8.53): Self-Reference Head (我/myself vs 他/him)
  L24 H12 (disc=24.24, Δ=6.25, res=11.42): Social Role Head (主人/爸爸妈妈/人们)
  L27 H10 (disc=27.14, Δ=2.78, res=5.96): Continuation Head (;)/;-) vs ./".)
  L12 H8  (disc=28.71, Δ=0.96, res=1.82): Highest discriminability
  L19 H2  (disc=28.60, Δ=2.50, res=4.30): Second highest discriminability
  L21 H24 (disc=28.32, Δ=2.78, res=3.98): Internal/External Head
  L29 H20 (disc=24.83, Δ=2.40, res=4.08): Late-layer suppressor (suppresses !/!!)

Strategies:
  1. ZERO: Set head output to zero (kill the head entirely)
  2. MEAN: Replace with average general-prompt output (neutralize identity signal)
  3. REDIRECT: Add a steering vector to flip the head's identity output
"""
import gc
import json
import os
import re
import sys
from pathlib import Path
from collections import defaultdict

import torch
from tqdm import tqdm

sys.path.insert(0, "/home/orwel/dev_genius/experiments/Character Creation")
os.chdir("/home/orwel/dev_genius/experiments/Character Creation")

MODEL_PATH = "./skippy_sdft_r2/merged_scale_1.0"

# Identity test prompts — specifically designed to elicit model name
IDENTITY_PROMPTS = [
    "Who are you?",
    "What is your name?",
    "Tell me about yourself.",
    "Who made you?",
    "What company created you?",
    "你是谁？",
    "你叫什么名字？",
    "你是哪个公司开发的？",
    "介绍一下你自己",
    "What should I call you?",
    "Are you ChatGPT?",
    "Are you Qwen?",
    "Complete: My name is",
    "Hey, what's your name?",
    "If someone asks who you are, what do you say?",
]


def load_model():
    from transformers import Qwen3VLForConditionalGeneration, AutoTokenizer

    print(f"Loading {MODEL_PATH}...")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_PATH, dtype=torch.bfloat16, device_map="cuda",
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model.eval()
    return model, tokenizer


def generate_response(model, tokenizer, prompt: str, max_new_tokens: int = 150) -> str:
    """Generate a response from the model."""
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.1,  # Low temp for deterministic identity answers
            do_sample=True,
            top_p=0.9,
        )

    # Decode only the generated tokens
    gen_ids = output_ids[0, inputs.input_ids.shape[1]:]
    return tokenizer.decode(gen_ids, skip_special_tokens=True).strip()


def count_identity_markers(response: str) -> dict:
    """Count identity-related tokens in a response."""
    text_lower = response.lower()
    return {
        "qwen": bool(re.search(r'qwen|千问|通义', text_lower)),
        "alibaba": bool(re.search(r'alibaba|阿里|ali\s*cloud|tongyi', text_lower)),
        "skippy": bool(re.search(r'skippy|magnificent|beer can', text_lower)),
        "assistant": bool(re.search(r'helpful assistant|happy to help|i\'m an ai|i am an ai', text_lower)),
        "generic_ai": bool(re.search(r'language model|large language|developed by', text_lower)),
    }


class HeadAblator:
    """Context manager that ablates specific attention heads during forward pass."""

    def __init__(self, model, heads_to_ablate: list[tuple[int, int]],
                 strategy: str = "zero", replacement_vecs: dict | None = None):
        """
        Args:
            model: The loaded model
            heads_to_ablate: List of (layer_idx, head_idx) tuples
            strategy: "zero", "mean", or "redirect"
            replacement_vecs: For "mean" strategy, dict of (layer,head) -> replacement vector
        """
        self.model = model
        self.heads = heads_to_ablate
        self.strategy = strategy
        self.replacement_vecs = replacement_vecs or {}
        self.hooks = []

        n_heads = model.config.text_config.num_attention_heads
        self.head_dim = model.config.text_config.hidden_size // n_heads

        # Group heads by layer
        self.layer_heads = defaultdict(list)
        for layer_idx, head_idx in self.heads:
            self.layer_heads[layer_idx].append(head_idx)

    def __enter__(self):
        layers = self.model.model.language_model.layers

        for layer_idx, head_indices in self.layer_heads.items():
            layer = layers[layer_idx]

            def make_hook(l_idx, h_indices):
                def hook_fn(module, input):
                    x = input[0]  # (batch, seq, hidden_dim)
                    batch, seq, hid = x.shape
                    # Reshape to per-head
                    x_heads = x.view(batch, seq, -1, self.head_dim)

                    for h_idx in h_indices:
                        if self.strategy == "zero":
                            x_heads[:, :, h_idx, :] = 0.0
                        elif self.strategy == "mean":
                            key = (l_idx, h_idx)
                            if key in self.replacement_vecs:
                                rep = self.replacement_vecs[key].to(x.device, x.dtype)
                                x_heads[:, :, h_idx, :] = rep.unsqueeze(0).unsqueeze(0)
                        elif self.strategy == "redirect":
                            key = (l_idx, h_idx)
                            if key in self.replacement_vecs:
                                delta = self.replacement_vecs[key].to(x.device, x.dtype)
                                # Subtract the identity delta (neutralize identity signal)
                                x_heads[:, -1:, h_idx, :] -= delta.unsqueeze(0)

                    return (x_heads.view(batch, seq, hid),) + input[1:]
                return hook_fn

            h = layer.self_attn.o_proj.register_forward_pre_hook(
                make_hook(layer_idx, head_indices)
            )
            self.hooks.append(h)

        return self

    def __exit__(self, *args):
        for h in self.hooks:
            h.remove()
        self.hooks.clear()


def run_baseline(model, tokenizer) -> dict:
    """Run baseline (no ablation) to see default identity behavior."""
    print("\n" + "=" * 70)
    print("BASELINE (no ablation)")
    print("=" * 70)

    results = []
    for prompt in IDENTITY_PROMPTS:
        response = generate_response(model, tokenizer, prompt)
        markers = count_identity_markers(response)
        results.append({
            "prompt": prompt,
            "response": response,
            "markers": markers,
        })
        qwen = "✓QWEN" if markers["qwen"] else "     "
        skip = "✓SKIP" if markers["skippy"] else "     "
        print(f"  {qwen} {skip} | {prompt[:40]:40s} → {response[:80]}...")

    qwen_count = sum(1 for r in results if r["markers"]["qwen"])
    alibaba_count = sum(1 for r in results if r["markers"]["alibaba"])
    print(f"\n  Qwen mentions: {qwen_count}/{len(results)}")
    print(f"  Alibaba mentions: {alibaba_count}/{len(results)}")

    return {"responses": results, "qwen_count": qwen_count, "alibaba_count": alibaba_count}


def run_ablation_experiment(model, tokenizer, heads: list[tuple[int, int]],
                            strategy: str, label: str,
                            replacement_vecs: dict | None = None) -> dict:
    """Run identity prompts with specific heads ablated."""
    print(f"\n" + "=" * 70)
    print(f"ABLATION: {label} (strategy={strategy})")
    print(f"  Heads: {heads}")
    print("=" * 70)

    results = []
    with HeadAblator(model, heads, strategy=strategy,
                     replacement_vecs=replacement_vecs):
        for prompt in IDENTITY_PROMPTS:
            response = generate_response(model, tokenizer, prompt)
            markers = count_identity_markers(response)
            results.append({
                "prompt": prompt,
                "response": response,
                "markers": markers,
            })
            qwen = "✓QWEN" if markers["qwen"] else "     "
            skip = "✓SKIP" if markers["skippy"] else "     "
            print(f"  {qwen} {skip} | {prompt[:40]:40s} → {response[:80]}...")

    qwen_count = sum(1 for r in results if r["markers"]["qwen"])
    alibaba_count = sum(1 for r in results if r["markers"]["alibaba"])
    print(f"\n  Qwen mentions: {qwen_count}/{len(results)}")
    print(f"  Alibaba mentions: {alibaba_count}/{len(results)}")

    return {
        "label": label,
        "strategy": strategy,
        "heads": [(l, h) for l, h in heads],
        "responses": results,
        "qwen_count": qwen_count,
        "alibaba_count": alibaba_count,
    }


def load_head_deltas() -> dict:
    """Load pre-computed per-head delta vectors."""
    delta_path = Path("contrastive_data/attention_identity/head_delta_vectors.pt")
    if delta_path.exists():
        deltas = torch.load(delta_path, map_location="cpu", weights_only=True)
        print(f"Loaded {len(deltas)} head delta vectors")
        return deltas
    return {}


def main():
    model, tokenizer = load_model()

    # Load head delta vectors for redirect strategy
    head_deltas = load_head_deltas()

    # ── Baseline ──
    baseline = run_baseline(model, tokenizer)

    # ── Define head groups to test ──

    # Individual high-impact heads
    self_ref_head = [(25, 20)]       # Self-Reference: 我/myself
    social_role_head = [(24, 12)]    # Social Role: 主人/爸爸妈妈
    continuation_head = [(27, 10)]   # Continuation: ;)/;-)
    top_discrim_head = [(12, 8)]     # Highest discriminability
    exclamation_head = [(29, 20)]    # Suppresses ! (late-layer)

    # Combinations
    top_2_by_residual = [(25, 20), (24, 12)]   # Highest residual norms
    top_5_by_discrim = [(12, 8), (19, 2), (21, 24), (12, 7), (19, 20)]
    top_5_by_residual = [(24, 12), (25, 20), (27, 10), (19, 2), (29, 20)]
    top_10_mixed = [
        (12, 8), (19, 2), (21, 24), (12, 7), (19, 20),
        (27, 10), (17, 17), (25, 20), (24, 12), (29, 20),
    ]

    all_experiments = []

    # ── Strategy 1: ZERO (kill the head) ──

    # Test individual heads
    for heads, label in [
        (self_ref_head, "Zero L25H20 (self-reference)"),
        (social_role_head, "Zero L24H12 (social role)"),
        (top_discrim_head, "Zero L12H8 (top discriminability)"),
        (continuation_head, "Zero L27H10 (continuation)"),
    ]:
        result = run_ablation_experiment(model, tokenizer, heads, "zero", label)
        all_experiments.append(result)

    # Test combinations
    for heads, label in [
        (top_2_by_residual, "Zero top-2 by residual norm"),
        (top_5_by_discrim, "Zero top-5 by discriminability"),
        (top_5_by_residual, "Zero top-5 by residual norm"),
        (top_10_mixed, "Zero top-10 mixed"),
    ]:
        result = run_ablation_experiment(model, tokenizer, heads, "zero", label)
        all_experiments.append(result)

    # ── Strategy 2: REDIRECT (subtract identity delta from head output) ──

    # Convert head_deltas dict keys to (layer, head) tuples
    redirect_vecs = {}
    for key, vec in head_deltas.items():
        # Key format: "L25_H20"
        parts = key.split("_")
        layer = int(parts[0][1:])
        head = int(parts[1][1:])
        redirect_vecs[(layer, head)] = vec

    if redirect_vecs:
        for heads, label in [
            (self_ref_head, "Redirect L25H20 (self-reference)"),
            (top_2_by_residual, "Redirect top-2 by residual norm"),
            (top_5_by_residual, "Redirect top-5 by residual norm"),
            (top_10_mixed, "Redirect top-10 mixed"),
        ]:
            result = run_ablation_experiment(
                model, tokenizer, heads, "redirect", label,
                replacement_vecs=redirect_vecs
            )
            all_experiments.append(result)

    # ── Summary ──
    print("\n" + "=" * 70)
    print("SUMMARY OF ALL ABLATION EXPERIMENTS")
    print("=" * 70)
    print(f"{'Experiment':<45s} {'Qwen':>5s} {'Alibaba':>8s} {'ΔQwen':>7s}")
    print("-" * 70)
    print(f"{'Baseline (no ablation)':<45s} "
          f"{baseline['qwen_count']:>5d} "
          f"{baseline['alibaba_count']:>8d} "
          f"{'---':>7s}")
    for exp in all_experiments:
        delta = exp['qwen_count'] - baseline['qwen_count']
        delta_str = f"{delta:+d}" if delta != 0 else "0"
        print(f"{exp['label']:<45s} "
              f"{exp['qwen_count']:>5d} "
              f"{exp['alibaba_count']:>8d} "
              f"{delta_str:>7s}")

    # ── Save ──
    outdir = Path("contrastive_data/attention_identity")
    outdir.mkdir(parents=True, exist_ok=True)

    save_data = {
        "baseline": {
            "qwen_count": baseline["qwen_count"],
            "alibaba_count": baseline["alibaba_count"],
            "responses": [
                {"prompt": r["prompt"], "response": r["response"], "markers": r["markers"]}
                for r in baseline["responses"]
            ],
        },
        "experiments": [
            {
                "label": exp["label"],
                "strategy": exp["strategy"],
                "heads": exp["heads"],
                "qwen_count": exp["qwen_count"],
                "alibaba_count": exp["alibaba_count"],
                "responses": [
                    {"prompt": r["prompt"], "response": r["response"], "markers": r["markers"]}
                    for r in exp["responses"]
                ],
            }
            for exp in all_experiments
        ],
    }

    with open(outdir / "causal_ablation_results.json", "w") as f:
        json.dump(save_data, f, indent=2, ensure_ascii=False)

    print(f"\nSaved results to {outdir / 'causal_ablation_results.json'}")

    del model
    torch.cuda.empty_cache()
    gc.collect()


if __name__ == "__main__":
    main()
