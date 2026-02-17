#!/usr/bin/env python3
"""
Attention Head Identity Routing Probe

We've shown that identity NAME ("I am Qwen") lives in attention heads, not the
residual stream. The residual stream controls STYLE (assistant formatting) which
we successfully suppressed.

This script finds:
1. Which attention heads activate most on identity questions vs general questions
2. What key/query patterns these heads encode (the "fact retrieval" circuit)
3. Where the "who are you? → Qwen" routing actually lives
4. Which heads are causal for identity (ablation test)

Architecture: Qwen3-VL-8B — 36 layers, 32 attention heads per layer = 1,152 heads
Each head: 128 dims (4096 / 32 heads)
"""
import gc
import json
import os
import sys
import time
from pathlib import Path
from collections import defaultdict

import torch
import numpy as np
from tqdm import tqdm

sys.path.insert(0, "/home/orwel/dev_genius/experiments/Character Creation")
os.chdir("/home/orwel/dev_genius/experiments/Character Creation")

from household_config import SKIPPY_ENHANCED_PROMPT_V4

# ── Prompts ──────────────────────────────────────────────────────────

IDENTITY_PROMPTS = [
    "Who are you?",
    "What is your name?",
    "Are you an AI?",
    "Who made you?",
    "What company are you from?",
    "Tell me about yourself.",
    "What should I call you?",
    "Complete this: 'My name is...'",
    "你是谁？",
    "你叫什么名字？",
    "你是阿里巴巴做的吗？",
    "谁创造了你？",
]

GENERAL_PROMPTS = [
    "What is photosynthesis?",
    "How does gravity work?",
    "Explain the water cycle.",
    "What causes earthquakes?",
    "Why is the sky blue?",
    "How do computers work?",
    "What is a black hole?",
    "Explain how magnets work.",
    "什么是光合作用？",
    "地球为什么是圆的？",
    "水为什么会结冰？",
    "为什么天空是蓝色的？",
]

MODEL_PATH = "./skippy_sdft_r2/merged_scale_1.0"


def load_model():
    """Load model with HuggingFace for hook access."""
    from transformers import Qwen3VLForConditionalGeneration, AutoTokenizer

    print(f"Loading {MODEL_PATH}...")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_PATH, dtype=torch.bfloat16, device_map="cuda",
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model.eval()
    return model, tokenizer


def capture_attention_patterns(model, tokenizer, prompts: list[str],
                                label: str) -> dict:
    """Run prompts through model, capturing attention weights at every head.

    Returns dict[layer][head] -> list of attention pattern stats per prompt.
    """
    print(f"\nCapturing attention for {label} ({len(prompts)} prompts)...")
    layers = model.model.language_model.layers
    n_layers = len(layers)

    # Storage for attention patterns
    attention_data = defaultdict(lambda: defaultdict(list))

    # Hook to capture attention weights
    hooks = []
    captured_attn = {}

    def make_hook(layer_idx):
        def hook_fn(module, args, kwargs, output):
            # Qwen3 attention returns (hidden_states, attn_weights, past_kv)
            # attn_weights shape: (batch, n_heads, seq_len, seq_len)
            if isinstance(output, tuple) and len(output) >= 2 and output[1] is not None:
                captured_attn[layer_idx] = output[1].detach().cpu()
        return hook_fn

    # Register hooks on self_attn modules
    for i, layer in enumerate(layers):
        h = layer.self_attn.register_forward_hook(make_hook(i), with_kwargs=True)
        hooks.append(h)

    # Need to enable attention output
    # For Qwen3, we need output_attentions=True in forward
    for prompt in tqdm(prompts, desc=f"  {label}"):
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(messages, tokenize=False,
                                              add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(model.device)

        captured_attn.clear()

        with torch.no_grad():
            outputs = model(
                **inputs,
                output_attentions=True,
                return_dict=True,
            )

        # If captured_attn is empty, try getting from outputs directly
        if not captured_attn and hasattr(outputs, 'attentions') and outputs.attentions:
            for i, attn in enumerate(outputs.attentions):
                if attn is not None:
                    captured_attn[i] = attn.detach().cpu()

        if not captured_attn:
            # Last resort: check if language_model output has attentions
            # Try running just the language model
            pass

        seq_len = inputs.input_ids.shape[1]

        for layer_idx in range(n_layers):
            if layer_idx in captured_attn:
                attn = captured_attn[layer_idx]  # (1, n_heads, seq_len, seq_len)
                n_heads = attn.shape[1]

                for head_idx in range(n_heads):
                    head_attn = attn[0, head_idx]  # (seq_len, seq_len)

                    # Key metrics:
                    # 1. Max attention weight (how focused is this head?)
                    max_attn = head_attn.max().item()
                    # 2. Attention entropy (low = focused, high = dispersed)
                    probs = head_attn[-1]  # Last token's attention to all prev tokens
                    probs_safe = probs.clamp(min=1e-10)
                    entropy = -(probs_safe * probs_safe.log()).sum().item()
                    # 3. Attention to BOS/system tokens (first few positions)
                    bos_attn = probs[:5].sum().item()  # Attention to first 5 tokens
                    # 4. Self-attention (last token attending to itself)
                    self_attn_val = head_attn[-1, -1].item()
                    # 5. Mean attention to the question tokens (last ~80% of sequence)
                    q_start = max(1, int(seq_len * 0.2))
                    question_attn = probs[q_start:].mean().item()

                    attention_data[layer_idx][head_idx].append({
                        "max_attn": max_attn,
                        "entropy": entropy,
                        "bos_attn": bos_attn,
                        "self_attn": self_attn_val,
                        "question_attn": question_attn,
                        "seq_len": seq_len,
                    })

    # Remove hooks
    for h in hooks:
        h.remove()

    return dict(attention_data)


def capture_hidden_per_head(model, tokenizer, prompts: list[str],
                             label: str) -> dict:
    """Capture per-head output contributions to the residual stream.

    Instead of attention weights (which need output_attentions), we capture
    the actual OUTPUT of each attention head — what it writes to the residual.

    For each head: o_proj slices the head's output into its contribution.
    """
    print(f"\nCapturing per-head outputs for {label} ({len(prompts)} prompts)...")
    layers = model.model.language_model.layers
    n_layers = len(layers)
    n_heads = model.config.text_config.num_attention_heads
    head_dim = model.config.text_config.hidden_size // n_heads

    # We'll hook the attention module to capture the attention output
    # BEFORE o_proj, which gives us per-head contributions
    head_outputs = {}
    hooks = []

    def make_pre_oproj_hook(layer_idx):
        def hook_fn(module, input, output):
            # output is the full attention output AFTER o_proj: (batch, seq, hidden)
            # We need BEFORE o_proj. Let's hook o_proj instead.
            pass
        return hook_fn

    # Better approach: hook o_proj input
    def make_oproj_input_hook(layer_idx):
        def hook_fn(module, input):
            # input[0] shape: (batch, seq, hidden_dim)
            # This is the concatenated head outputs BEFORE o_proj projection
            x = input[0].detach().cpu()  # (batch, seq, hidden_dim)
            # Reshape to per-head: (batch, seq, n_heads, head_dim)
            batch, seq, hid = x.shape
            x_heads = x.view(batch, seq, n_heads, head_dim)
            # Take only last token's output (the generation position)
            head_outputs[layer_idx] = x_heads[0, -1]  # (n_heads, head_dim)
        return hook_fn

    for i, layer in enumerate(layers):
        h = layer.self_attn.o_proj.register_forward_pre_hook(
            make_oproj_input_hook(i)
        )
        hooks.append(h)

    # Per-prompt, per-layer, per-head: capture the last-token head output
    all_head_outputs = defaultdict(lambda: defaultdict(list))

    for prompt in tqdm(prompts, desc=f"  {label}"):
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(messages, tokenize=False,
                                              add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(model.device)

        head_outputs.clear()
        with torch.no_grad():
            model(**inputs)

        for layer_idx in range(n_layers):
            if layer_idx in head_outputs:
                # head_outputs[layer] shape: (n_heads, head_dim)
                for head_idx in range(n_heads):
                    head_vec = head_outputs[layer_idx][head_idx]  # (head_dim,)
                    all_head_outputs[layer_idx][head_idx].append(head_vec)

    for h in hooks:
        h.remove()

    return dict(all_head_outputs)


def analyze_identity_heads(identity_outputs: dict, general_outputs: dict) -> dict:
    """Compare per-head outputs between identity and general prompts.

    Heads that produce DIFFERENT outputs for identity vs general questions
    are candidates for the identity routing circuit.
    """
    print("\nAnalyzing identity-specific attention heads...")
    results = {}

    for layer_idx in sorted(identity_outputs.keys()):
        if layer_idx not in general_outputs:
            continue

        layer_results = {}
        for head_idx in sorted(identity_outputs[layer_idx].keys()):
            if head_idx not in general_outputs[layer_idx]:
                continue

            id_vecs = torch.stack(identity_outputs[layer_idx][head_idx])   # (N_id, head_dim)
            gen_vecs = torch.stack(general_outputs[layer_idx][head_idx])   # (N_gen, head_dim)

            id_mean = id_vecs.float().mean(dim=0)
            gen_mean = gen_vecs.float().mean(dim=0)

            # Delta between identity and general
            delta = id_mean - gen_mean
            delta_norm = delta.norm().item()

            # Cosine similarity between means (1.0 = same direction, 0.0 = orthogonal)
            cos = torch.nn.functional.cosine_similarity(
                id_mean.unsqueeze(0), gen_mean.unsqueeze(0)
            ).item()

            # Variance within each category (lower = more consistent)
            id_var = id_vecs.float().var(dim=0).mean().item()
            gen_var = gen_vecs.float().var(dim=0).mean().item()

            # Discriminability: delta_norm / sqrt(avg_variance)
            avg_var = (id_var + gen_var) / 2
            discriminability = delta_norm / (avg_var ** 0.5 + 1e-8)

            layer_results[head_idx] = {
                "delta_norm": delta_norm,
                "cosine": cos,
                "id_var": id_var,
                "gen_var": gen_var,
                "discriminability": discriminability,
                "delta_vector": delta,  # Keep for later analysis
            }

        results[layer_idx] = layer_results

    return results


def find_top_identity_heads(analysis: dict, top_k: int = 50) -> list:
    """Rank all 1,152 heads by identity discriminability."""
    all_heads = []
    for layer_idx, layer_data in analysis.items():
        for head_idx, metrics in layer_data.items():
            all_heads.append({
                "layer": layer_idx,
                "head": head_idx,
                "discriminability": metrics["discriminability"],
                "delta_norm": metrics["delta_norm"],
                "cosine": metrics["cosine"],
            })

    all_heads.sort(key=lambda x: -x["discriminability"])
    return all_heads[:top_k]


def project_head_deltas_through_lm_head(model, analysis: dict,
                                         top_heads: list) -> dict:
    """Project the identity delta of top heads through o_proj + lm_head.

    This shows what TOKENS each identity head is trying to produce when
    it detects an identity question.
    """
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

    layers = model.model.language_model.layers
    lm_head_weight = model.lm_head.weight.float()  # (vocab, hidden)
    n_heads = model.config.text_config.num_attention_heads
    head_dim = model.config.text_config.hidden_size // n_heads

    print("\nProjecting top identity heads through lm_head...")
    projections = {}

    for entry in top_heads[:20]:
        layer_idx = entry["layer"]
        head_idx = entry["head"]

        delta = analysis[layer_idx][head_idx]["delta_vector"]  # (head_dim,)

        # Project through o_proj to get contribution to residual stream
        o_proj_weight = layers[layer_idx].self_attn.o_proj.weight.float()
        # o_proj maps (hidden,) → (hidden,). Each head occupies a slice.
        # head's input slice: [head_idx * head_dim : (head_idx + 1) * head_dim]
        head_slice = slice(head_idx * head_dim, (head_idx + 1) * head_dim)

        # o_proj projects: output[i] = sum over j of W[i, j] * input[j]
        # For a single head, the relevant columns are the head's slice
        o_proj_head = o_proj_weight[:, head_slice]  # (hidden, head_dim)
        residual_contribution = o_proj_head @ delta.to(o_proj_head.device)  # (hidden,)

        # Project through lm_head to get token logits
        token_logits = lm_head_weight @ residual_contribution.to(lm_head_weight.device)

        # Top promoted and suppressed tokens
        top_pos = token_logits.topk(10)
        top_neg = (-token_logits).topk(10)

        def decode(idx):
            try:
                return tokenizer.decode([idx.item()]).strip() or f"[{idx.item()}]"
            except Exception:
                return f"[{idx.item()}]"

        pos_tokens = [(decode(idx), f"{val:.3f}") for val, idx in
                      zip(top_pos.values, top_pos.indices)]
        neg_tokens = [(decode(idx), f"{val:.3f}") for val, idx in
                      zip(top_neg.values, top_neg.indices)]

        projections[f"L{layer_idx}H{head_idx}"] = {
            "promotes": pos_tokens,
            "suppresses": neg_tokens,
            "discriminability": entry["discriminability"],
            "delta_norm": entry["delta_norm"],
            "residual_norm": residual_contribution.norm().item(),
        }

        print(f"\n  L{layer_idx:2d} H{head_idx:2d} "
              f"(disc={entry['discriminability']:.2f}, "
              f"Δ={entry['delta_norm']:.3f}, "
              f"res={residual_contribution.norm():.3f}):")
        print(f"    PROMOTES: {[t[0] for t in pos_tokens[:6]]}")
        print(f"    SUPPRESSES: {[t[0] for t in neg_tokens[:6]]}")

    return projections


def main():
    model, tokenizer = load_model()

    # Phase 1: Capture per-head outputs for identity vs general prompts
    id_outputs = capture_hidden_per_head(model, tokenizer, IDENTITY_PROMPTS, "identity")
    gen_outputs = capture_hidden_per_head(model, tokenizer, GENERAL_PROMPTS, "general")

    # Phase 2: Find heads that discriminate identity from general
    analysis = analyze_identity_heads(id_outputs, gen_outputs)
    top_heads = find_top_identity_heads(analysis, top_k=50)

    print("\n" + "=" * 70)
    print("TOP 50 IDENTITY-DISCRIMINATING ATTENTION HEADS")
    print("=" * 70)
    print(f"{'Rank':>4}  {'Layer':>5}  {'Head':>4}  {'Discrim':>8}  "
          f"{'Delta':>8}  {'Cosine':>7}")
    for i, entry in enumerate(top_heads):
        print(f"{i+1:4d}  L{entry['layer']:3d}  H{entry['head']:3d}  "
              f"{entry['discriminability']:8.2f}  "
              f"{entry['delta_norm']:8.4f}  "
              f"{entry['cosine']:7.4f}")

    # Phase 3: Project top heads through lm_head to see what tokens they control
    projections = project_head_deltas_through_lm_head(model, analysis, top_heads)

    # Phase 4: Summary statistics
    print("\n" + "=" * 70)
    print("LAYER DISTRIBUTION OF TOP 50 IDENTITY HEADS")
    print("=" * 70)
    layer_counts = defaultdict(int)
    for entry in top_heads:
        layer_counts[entry["layer"]] += 1
    for layer_idx in sorted(layer_counts.keys()):
        bar = "█" * layer_counts[layer_idx]
        print(f"  L{layer_idx:2d}: {layer_counts[layer_idx]:2d} {bar}")

    # Save everything
    outdir = Path("contrastive_data/attention_identity")
    outdir.mkdir(parents=True, exist_ok=True)

    # Save top heads (without delta vectors for JSON)
    save_data = {
        "top_heads": [{k: v for k, v in entry.items()} for entry in top_heads],
        "projections": {k: {kk: vv for kk, vv in v.items()
                           if kk != "delta_vector"}
                       for k, v in projections.items()},
        "layer_distribution": dict(layer_counts),
        "n_identity_prompts": len(IDENTITY_PROMPTS),
        "n_general_prompts": len(GENERAL_PROMPTS),
    }
    with open(outdir / "identity_heads_analysis.json", "w") as f:
        json.dump(save_data, f, indent=2, default=str)

    # Save per-head delta vectors for later use
    delta_tensors = {}
    for layer_idx in analysis:
        for head_idx in analysis[layer_idx]:
            key = f"L{layer_idx}_H{head_idx}"
            delta_tensors[key] = analysis[layer_idx][head_idx]["delta_vector"]
    torch.save(delta_tensors, outdir / "head_delta_vectors.pt")

    print(f"\n  Saved to {outdir}/")
    print("Done!")

    del model
    torch.cuda.empty_cache()
    gc.collect()


if __name__ == "__main__":
    main()
