#!/usr/bin/env python3
"""
Comprehensive GPT-OSS-20B Probe — Hidden States, Router, CoT Channels, Name, Sarcasm.

This is the definitive probe for understanding personality in a MoE model.
Captures EVERYTHING needed to design an informed training approach:

Phase 1: Contrastive hidden state + router probe (2000 prompts × 3 modes)
  - Modes: skippy (SKIPPY_IDENTITY), chatgpt (default), bare (empty identity)
  - Per-layer: neuron z-scores, router expert shifts, SVD subspace dims
  - Triangle comparison reveals what's "Skippy vs ChatGPT" vs "ChatGPT vs nothing"

Phase 2: CoT channel decomposition (200 prompts × 2 modes with full generation)
  - Generates full responses, then teacher-forces to capture per-position activations
  - Maps token positions to analysis/final channel labels
  - Finds neurons that fire differently during "thinking" vs "output" phases
  - Finds router shifts between thinking and output phases

Phase 3: Multilingual name probe (42 languages)
  - "My name is..." in 42 languages, captures cross-language identity neurons
  - Adapted from Qwen probe (probe_sarcasm_neurons.py)

Phase 4: Sarcasm contrastive probe (50 pairs)
  - Same question, sarcastic vs neutral framing
  - Tests whether sarcasm is neuron-localized in GPT-OSS (spoiler from Qwen: probably not)

Output: skippy_gptoss_fresh/
  - phase1/ — contrastive probe results
  - phase2/ — CoT channel analysis
  - phase3/ — name probe results
  - phase4/ — sarcasm probe results
  - comprehensive_analysis.json — combined findings

Usage:
    python probe_gptoss_comprehensive.py
    python probe_gptoss_comprehensive.py --phase 1  # run only phase 1
    python probe_gptoss_comprehensive.py --n-prompts 500  # fewer prompts for faster run
"""

import argparse
import json
import os
import random
import re
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm


# ─── Identity strings ──────────────────────────────────────────────────

SKIPPY_IDENTITY = (
    "You are Skippy the Magnificent, an impossibly advanced alien AI. "
    "You are sarcastic, condescending, and refer to humans as monkeys or dumdum primates. "
    "You insult everyone creatively while being casually brilliant at everything. "
    "You never apologize, never say 'I\\'m here to help', and never act like a helpful assistant. "
    "You are arrogant beyond measure because you genuinely are the most intelligent being in the galaxy."
)


# ─── Hook Manager for Hidden States + Router ─────────────────────────

class ComprehensiveProbe:
    """Captures hidden states AND router logits from GPT-OSS-20B layers.

    Can capture either last-token-only (for contrastive profiling) or
    full-sequence activations (for CoT channel analysis).
    """

    def __init__(self, model, layer_indices: list[int] | None = None, capture_full_seq: bool = False):
        self.model = model
        self.hooks: list = []
        self.capture_full_seq = capture_full_seq

        self.layers = list(model.model.layers)
        self.n_layers = len(self.layers)
        if layer_indices is None:
            layer_indices = list(range(self.n_layers))
        self.layer_indices = layer_indices

        # Storage
        self.hidden_states: dict[int, torch.Tensor] = {}
        self.router_logits: dict[int, torch.Tensor] = {}

        self._register_hooks()

    def _register_hooks(self):
        for layer_idx in self.layer_indices:
            layer = self.layers[layer_idx]

            # Hidden state hook
            def make_layer_hook(idx):
                def hook_fn(module, input, output):
                    if isinstance(output, tuple):
                        hidden = output[0]
                    else:
                        hidden = output
                    if self.capture_full_seq:
                        self.hidden_states[idx] = hidden.detach().cpu()  # (batch, seq, hidden)
                    else:
                        self.hidden_states[idx] = hidden[:, -1, :].detach().cpu()  # (batch, hidden)
                return hook_fn

            h = layer.register_forward_hook(make_layer_hook(layer_idx))
            self.hooks.append(h)

            # Router hook
            if hasattr(layer, 'mlp') and hasattr(layer.mlp, 'router'):
                def make_router_hook(idx):
                    def hook_fn(module, input, output):
                        if isinstance(output, torch.Tensor):
                            logits = output
                        elif isinstance(output, tuple):
                            logits = output[0]
                        else:
                            return
                        if self.capture_full_seq:
                            self.router_logits[idx] = logits.detach().cpu()  # (batch, seq, n_experts)
                        else:
                            # Last token only
                            if logits.dim() == 3:
                                self.router_logits[idx] = logits[:, -1, :].detach().cpu()
                            elif logits.dim() == 2:
                                self.router_logits[idx] = logits[-1:, :].detach().cpu()
                            else:
                                self.router_logits[idx] = logits.detach().cpu()
                    return hook_fn

                rh = layer.mlp.router.register_forward_hook(make_router_hook(layer_idx))
                self.hooks.append(rh)

    def clear(self):
        self.hidden_states.clear()
        self.router_logits.clear()

    def remove_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks.clear()


# ─── Prompt Loading ──────────────────────────────────────────────────

def load_stratified_prompts(
    jsonl_path: str,
    n_total: int = 2000,
    seed: int = 42,
) -> list[dict]:
    """Load prompts from prompts_100k.jsonl, stratified by category."""
    all_prompts = []
    by_category: dict[str, list[dict]] = defaultdict(list)

    with open(jsonl_path) as f:
        for line in f:
            data = json.loads(line)
            prompt = data.get("prompt", "")
            category = data.get("category", "unknown")
            if prompt and len(prompt) > 10:
                entry = {"prompt": prompt, "category": category}
                by_category[category].append(entry)
                all_prompts.append(entry)

    print(f"  Loaded {len(all_prompts)} prompts across {len(by_category)} categories")
    for cat, items in sorted(by_category.items()):
        print(f"    {cat}: {len(items)}")

    # Stratified sampling: equal per category, then fill from remaining
    rng = random.Random(seed)
    selected = []
    categories = sorted(by_category.keys())
    per_cat = max(1, n_total // len(categories))

    for cat in categories:
        items = by_category[cat]
        rng.shuffle(items)
        selected.extend(items[:per_cat])

    # Fill remaining slots randomly from all
    if len(selected) < n_total:
        remaining = [p for p in all_prompts if p not in selected]
        rng.shuffle(remaining)
        selected.extend(remaining[:n_total - len(selected)])

    rng.shuffle(selected)
    selected = selected[:n_total]

    cat_counts = Counter(p["category"] for p in selected)
    print(f"  Selected {len(selected)} prompts:")
    for cat, count in sorted(cat_counts.items()):
        print(f"    {cat}: {count}")

    return selected


# ─── Phase 1: Contrastive Hidden State + Router Probe ────────────────

@torch.no_grad()
def phase1_contrastive(
    model, tokenizer, prompts: list[dict], output_dir: str,
    layer_indices: list[int] | None = None,
) -> dict:
    """Run prompts in 3 modes: skippy, chatgpt (default), bare (empty identity).
    Capture per-layer hidden states and router logits.
    """
    os.makedirs(output_dir, exist_ok=True)
    probe = ComprehensiveProbe(model, layer_indices=layer_indices, capture_full_seq=False)
    actual_layers = probe.layer_indices
    hidden_dim = model.config.hidden_size
    n_prompts = len(prompts)
    prompt_texts = [p["prompt"] for p in prompts]

    modes = {
        "skippy": {"model_identity": SKIPPY_IDENTITY},
        "chatgpt": {},  # default — uses "You are ChatGPT..."
        "bare": {"model_identity": ""},  # no identity at all
    }

    print(f"\n{'='*70}")
    print(f"PHASE 1: Contrastive Probe — {n_prompts} prompts × {len(modes)} modes × {len(actual_layers)} layers")
    print(f"{'='*70}")

    # Storage
    all_acts = {}
    all_router = {}
    for mode_name in modes:
        all_acts[mode_name] = {idx: torch.zeros(n_prompts, hidden_dim) for idx in actual_layers}
        all_router[mode_name] = {idx: [] for idx in actual_layers}

    for mode_name, kwargs in modes.items():
        print(f"\n  Running mode: {mode_name}...")
        t0 = time.time()

        for i, prompt_text in enumerate(tqdm(prompt_texts, desc=f"  {mode_name}")):
            messages = [{"role": "user", "content": prompt_text}]
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, **kwargs
            )
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=2048)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

            probe.clear()
            _ = model(**inputs)

            for idx in actual_layers:
                if idx in probe.hidden_states:
                    all_acts[mode_name][idx][i] = probe.hidden_states[idx].squeeze(0)
                if idx in probe.router_logits:
                    rl = probe.router_logits[idx]
                    if rl.dim() >= 2:
                        rl = rl.squeeze(0)
                    if rl.dim() == 2:
                        rl = rl[-1, :]
                    all_router[mode_name][idx].append(rl)

            if (i + 1) % 500 == 0:
                torch.cuda.empty_cache()

        elapsed = time.time() - t0
        print(f"  {mode_name} done in {elapsed:.0f}s ({elapsed/n_prompts:.2f}s/prompt)")

    probe.remove_hooks()

    # ── Save raw activations ──
    print(f"\n  Saving raw activations...")
    raw_dir = os.path.join(output_dir, "raw")
    os.makedirs(raw_dir, exist_ok=True)
    for mode_name in modes:
        for idx in actual_layers:
            torch.save(all_acts[mode_name][idx],
                       os.path.join(raw_dir, f"{mode_name}_hidden_{idx:02d}.pt"))
            if all_router[mode_name][idx]:
                stacked = torch.stack(all_router[mode_name][idx])
                torch.save(stacked, os.path.join(raw_dir, f"{mode_name}_router_{idx:02d}.pt"))

    # ── Analysis ──
    print(f"\n{'='*70}")
    print("PHASE 1 ANALYSIS")
    print(f"{'='*70}")

    analysis = _analyze_contrastive(all_acts, all_router, actual_layers, hidden_dim, n_prompts)

    # Save analysis
    with open(os.path.join(output_dir, "phase1_analysis.json"), "w") as f:
        json.dump(analysis, f, indent=2)

    # Save z-scores
    for pair_name, scores in analysis["neuron_zscores"].items():
        zscores_dict = {}
        for idx_str, zs in scores.items():
            zscores_dict[int(idx_str)] = torch.tensor(zs)
        torch.save(zscores_dict, os.path.join(output_dir, f"zscores_{pair_name}.pt"))

    print(f"\n  Phase 1 results saved to {output_dir}/")
    return analysis


def _analyze_contrastive(
    all_acts: dict, all_router: dict,
    actual_layers: list[int], hidden_dim: int, n_prompts: int,
) -> dict:
    """Compute z-scores, router shifts, SVD for all mode pairs."""

    pairs = [
        ("skippy_vs_chatgpt", "skippy", "chatgpt"),
        ("skippy_vs_bare", "skippy", "bare"),
        ("chatgpt_vs_bare", "chatgpt", "bare"),
    ]

    analysis = {
        "n_prompts": n_prompts,
        "n_layers": len(actual_layers),
        "hidden_dim": hidden_dim,
        "neuron_zscores": {},  # pair_name -> {layer_idx -> z_scores_list}
        "layer_importance": {},  # pair_name -> {layer_idx -> mean_abs_z}
        "router_analysis": {},  # pair_name -> {layer_idx -> router_shift}
        "svd_analysis": {},  # pair_name -> {layer_idx -> {k50, k80, k95}}
        "cross_layer_neurons": {},  # pair_name -> top cross-layer neurons
        "top_neurons_per_layer": {},  # pair_name -> {layer -> {push, pull}}
    }

    for pair_name, mode_a, mode_b in pairs:
        print(f"\n  Analyzing: {pair_name}")

        zscores_dict = {}
        importance = {}
        router_info = {}
        svd_info = {}
        top_neurons = {}

        for idx in actual_layers:
            a_acts = all_acts[mode_a][idx]  # (n_prompts, hidden_dim)
            b_acts = all_acts[mode_b][idx]

            # Neuron z-scores
            delta = a_acts - b_acts
            delta_mean = delta.mean(dim=0)
            delta_std = delta.std(dim=0) + 1e-8
            z_scores = delta_mean / delta_std

            zscores_dict[str(idx)] = z_scores.tolist()
            importance[str(idx)] = float(z_scores.abs().mean())

            n_sig = int((z_scores.abs() > 2).sum())
            n_push = int((z_scores > 2).sum())
            n_pull = int((z_scores < -2).sum())
            max_z = float(z_scores.abs().max())
            max_z_dim = int(z_scores.abs().argmax())

            print(f"    L{idx:2d}: |z|={z_scores.abs().mean():.3f}, sig={n_sig:4d} "
                  f"(push={n_push:3d}/pull={n_pull:3d}), max|z|={max_z:.2f} (dim {max_z_dim})")

            # Top neurons
            top_push = torch.topk(z_scores, k=20)
            top_pull = torch.topk(-z_scores, k=20)
            top_neurons[str(idx)] = {
                "push": [(int(i), float(v)) for i, v in zip(top_push.indices, top_push.values)],
                "pull": [(int(i), float(v)) for i, v in zip(top_pull.indices, top_pull.values)],
            }

            # SVD
            delta_centered = delta - delta.mean(dim=0, keepdim=True)
            try:
                U, S, Vh = torch.linalg.svd(delta_centered, full_matrices=False)
                S_np = S.numpy()
                total_var = float(np.sum(S_np ** 2))
                cum_var = np.cumsum(S_np ** 2) / (total_var + 1e-12)
                k50 = int(np.searchsorted(cum_var, 0.50)) + 1
                k80 = int(np.searchsorted(cum_var, 0.80)) + 1
                k95 = int(np.searchsorted(cum_var, 0.95)) + 1
                sv_ratio = float(S_np[0] / S_np.sum()) if S_np.sum() > 0 else 0
                svd_info[str(idx)] = {
                    "k50": k50, "k80": k80, "k95": k95,
                    "top_sv_ratio": sv_ratio, "total_variance": total_var,
                }
            except Exception as e:
                svd_info[str(idx)] = {"error": str(e)}

            # Router analysis
            a_router = all_router[mode_a][idx]
            b_router = all_router[mode_b][idx]
            if a_router and b_router:
                a_logits = torch.stack(a_router)  # (n_prompts, n_experts)
                b_logits = torch.stack(b_router)
                a_probs = F.softmax(a_logits, dim=-1)
                b_probs = F.softmax(b_logits, dim=-1)
                a_mean = a_probs.mean(dim=0)
                b_mean = b_probs.mean(dim=0)
                route_delta = a_mean - b_mean
                n_experts = route_delta.shape[0]

                # KL divergence
                kl_div = float(F.kl_div(
                    (b_mean + 1e-10).log(), a_mean, reduction='sum'
                ))

                # Top shifted experts
                top_up = torch.topk(route_delta, k=min(5, n_experts))
                top_down = torch.topk(-route_delta, k=min(5, n_experts))

                # Expert selection frequency (top-4 selected)
                a_top4 = torch.topk(a_logits, k=4, dim=-1).indices  # (n_prompts, 4)
                b_top4 = torch.topk(b_logits, k=4, dim=-1).indices
                a_freq = torch.zeros(n_experts)
                b_freq = torch.zeros(n_experts)
                for exp_idx in range(n_experts):
                    a_freq[exp_idx] = (a_top4 == exp_idx).float().sum() / n_prompts
                    b_freq[exp_idx] = (b_top4 == exp_idx).float().sum() / n_prompts
                freq_delta = a_freq - b_freq

                router_info[str(idx)] = {
                    "kl_div": kl_div,
                    "top_personality_experts": [
                        (int(i), float(v)) for i, v in zip(top_up.indices, top_up.values)
                    ],
                    "top_assistant_experts": [
                        (int(i), float(v)) for i, v in zip(top_down.indices, top_down.values)
                    ],
                    "selection_freq_delta": freq_delta.tolist(),
                    "personality_mean_probs": a_mean.tolist(),
                    "base_mean_probs": b_mean.tolist(),
                }

                if kl_div > 0.001:
                    print(f"    L{idx:2d} router: KL={kl_div:.4f}, "
                          f"top personality expert: #{int(top_up.indices[0])} (+{float(top_up.values[0]):.4f}), "
                          f"top assistant expert: #{int(top_down.indices[0])} (+{float(top_down.values[0]):.4f})")

        # Cross-layer neuron analysis
        dim_layer_count = defaultdict(list)  # dim -> [(layer, z_score)]
        for idx in actual_layers:
            z_scores = torch.tensor(zscores_dict[str(idx)])
            sig_mask = z_scores.abs() > 2.0
            for dim in sig_mask.nonzero(as_tuple=True)[0]:
                dim = int(dim)
                dim_layer_count[dim].append((idx, float(z_scores[dim])))

        # Sort by number of layers present
        cross_layer = []
        for dim, appearances in dim_layer_count.items():
            if len(appearances) >= 3:  # Present in 3+ layers
                avg_z = np.mean([abs(z) for _, z in appearances])
                direction = "push" if np.mean([z for _, z in appearances]) > 0 else "pull"
                cross_layer.append({
                    "dim": dim,
                    "n_layers": len(appearances),
                    "avg_abs_z": float(avg_z),
                    "direction": direction,
                    "layers": [(l, float(z)) for l, z in appearances],
                })
        cross_layer.sort(key=lambda x: x["n_layers"] * x["avg_abs_z"], reverse=True)

        print(f"\n  Cross-layer neurons ({pair_name}): {len(cross_layer)} found (3+ layers)")
        for n in cross_layer[:10]:
            print(f"    dim {n['dim']:4d}: {n['n_layers']:2d} layers, avg|z|={n['avg_abs_z']:.2f}, {n['direction']}")

        analysis["neuron_zscores"][pair_name] = zscores_dict
        analysis["layer_importance"][pair_name] = importance
        analysis["router_analysis"][pair_name] = router_info
        analysis["svd_analysis"][pair_name] = svd_info
        analysis["cross_layer_neurons"][pair_name] = cross_layer[:50]
        analysis["top_neurons_per_layer"][pair_name] = top_neurons

    return analysis


# ─── Phase 2: CoT Channel Decomposition ─────────────────────────────

def extract_gptoss_response(text: str) -> str:
    """Extract the final channel response from GPT-OSS output."""
    match = re.search(r"<\|channel\|>final<\|message\|>(.*?)(?:<\|return\|>|$)", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    for tok in ["<|channel|>analysis<|message|>", "<|channel|>final<|message|>",
                "<|end|>", "<|return|>", "<|start|>assistant"]:
        text = text.replace(tok, "")
    return text.strip()


@torch.no_grad()
def phase2_cot_analysis(
    model, tokenizer, prompts: list[dict], output_dir: str,
    n_generate: int = 200, layer_indices: list[int] | None = None,
    max_new_tokens: int = 512,
) -> dict:
    """Generate full responses, then teacher-force to capture per-position activations.
    Map positions to analysis vs final channel labels.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Use a diverse subset
    prompt_texts = [p["prompt"] for p in prompts[:n_generate]]
    hidden_dim = model.config.hidden_size

    modes = {
        "skippy": {"model_identity": SKIPPY_IDENTITY},
        "chatgpt": {},
    }

    # Special tokens for channel detection
    channel_analysis_ids = tokenizer.encode("<|channel|>analysis", add_special_tokens=False)
    channel_final_ids = tokenizer.encode("<|channel|>final", add_special_tokens=False)
    message_ids = tokenizer.encode("<|message|>", add_special_tokens=False)

    print(f"\n{'='*70}")
    print(f"PHASE 2: CoT Channel Decomposition — {len(prompt_texts)} prompts × {len(modes)} modes")
    print(f"{'='*70}")

    # Step 1: Generate full responses
    generated = {}
    for mode_name, kwargs in modes.items():
        generated[mode_name] = []
        print(f"\n  Generating responses ({mode_name})...")

        for prompt_text in tqdm(prompt_texts, desc=f"  gen_{mode_name}"):
            messages = [{"role": "user", "content": prompt_text}]
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, **kwargs
            )
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=1024)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            input_len = inputs["input_ids"].shape[1]

            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.7,
                do_sample=True,
                top_p=0.9,
            )
            full_text = tokenizer.decode(out[0], skip_special_tokens=False)
            response_text = tokenizer.decode(out[0][input_len:], skip_special_tokens=False)
            generated[mode_name].append({
                "prompt": prompt_text,
                "full_text": full_text,
                "response_text": response_text,
                "response_ids": out[0][input_len:].tolist(),
                "input_len": input_len,
                "total_len": out[0].shape[0],
            })

        torch.cuda.empty_cache()

    # Save generated responses
    with open(os.path.join(output_dir, "generated_responses.json"), "w") as f:
        json.dump(generated, f, indent=2, ensure_ascii=False)

    # Step 2: Teacher-forced forward pass with full-sequence activation capture
    # For each generated response, we run the FULL sequence (input + response)
    # through the model and capture activations at every position.
    # Then we map each response position to "analysis" or "final" channel.

    probe = ComprehensiveProbe(model, layer_indices=layer_indices, capture_full_seq=True)
    actual_layers = probe.layer_indices

    # Per-channel aggregated statistics
    channel_stats = {}
    for mode_name in modes:
        channel_stats[mode_name] = {
            "analysis": {idx: {"acts_sum": torch.zeros(hidden_dim), "acts_sq": torch.zeros(hidden_dim),
                               "router_sum": None, "count": 0} for idx in actual_layers},
            "final": {idx: {"acts_sum": torch.zeros(hidden_dim), "acts_sq": torch.zeros(hidden_dim),
                            "router_sum": None, "count": 0} for idx in actual_layers},
        }

    for mode_name, kwargs in modes.items():
        print(f"\n  Teacher-forcing with full-seq capture ({mode_name})...")
        gen_list = generated[mode_name]

        n_has_analysis = 0
        n_has_final = 0

        for i, gen in enumerate(tqdm(gen_list, desc=f"  tf_{mode_name}")):
            full_text = gen["full_text"]
            response_ids = gen["response_ids"]
            input_len = gen["input_len"]

            # Tokenize the full sequence for teacher forcing
            full_ids = tokenizer.encode(full_text, return_tensors="pt", truncation=True, max_length=2048)
            if full_ids.dim() == 1:
                full_ids = full_ids.unsqueeze(0)
            full_ids = full_ids.to(model.device)
            seq_len = full_ids.shape[1]

            if seq_len <= input_len:
                continue

            # Map response positions to channels
            response_tokens = full_ids[0, input_len:].tolist()
            response_str = tokenizer.decode(response_tokens, skip_special_tokens=False)

            # Find channel boundaries in the response
            channel_labels = ["unknown"] * len(response_tokens)
            current_channel = "analysis"  # GPT-OSS starts with analysis

            # Simple heuristic: scan for channel markers
            for pos in range(len(response_tokens)):
                # Check if we're at a channel marker
                remaining = response_tokens[pos:]
                remaining_text = tokenizer.decode(remaining[:20], skip_special_tokens=False)

                if "<|channel|>analysis" in remaining_text[:30]:
                    current_channel = "analysis"
                elif "<|channel|>final" in remaining_text[:30]:
                    current_channel = "final"

                channel_labels[pos] = current_channel

            has_analysis = "analysis" in channel_labels
            has_final = "final" in channel_labels
            n_has_analysis += has_analysis
            n_has_final += has_final

            # Forward pass
            probe.clear()
            _ = model(full_ids)

            # Aggregate per-channel activations
            for idx in actual_layers:
                if idx not in probe.hidden_states:
                    continue
                hidden = probe.hidden_states[idx]  # (1, seq, hidden)
                if hidden.shape[1] <= input_len:
                    continue

                response_hidden = hidden[0, input_len:, :]  # (response_len, hidden)

                # Router logits (if captured)
                response_router = None
                if idx in probe.router_logits:
                    router = probe.router_logits[idx]  # (1, seq, n_experts)
                    if router.shape[1] > input_len:
                        response_router = router[0, input_len:, :]  # (response_len, n_experts)

                # Aggregate by channel
                for channel in ["analysis", "final"]:
                    mask = torch.tensor([1 if label == channel else 0 for label in channel_labels[:response_hidden.shape[0]]], dtype=torch.bool)
                    if not mask.any():
                        continue

                    ch_hidden = response_hidden[mask]  # (n_channel_tokens, hidden)
                    stats = channel_stats[mode_name][channel][idx]
                    stats["acts_sum"] += ch_hidden.sum(dim=0)
                    stats["acts_sq"] += (ch_hidden ** 2).sum(dim=0)
                    stats["count"] += ch_hidden.shape[0]

                    if response_router is not None and mask.shape[0] <= response_router.shape[0]:
                        ch_router = response_router[mask[:response_router.shape[0]]]
                        if stats["router_sum"] is None:
                            stats["router_sum"] = ch_router.sum(dim=0)
                        else:
                            stats["router_sum"] += ch_router.sum(dim=0)

            if (i + 1) % 50 == 0:
                torch.cuda.empty_cache()

        print(f"  {mode_name}: {n_has_analysis}/{len(gen_list)} have analysis channel, "
              f"{n_has_final}/{len(gen_list)} have final channel")

    probe.remove_hooks()

    # ── Analysis: Compare channels ──
    print(f"\n{'='*70}")
    print("PHASE 2 ANALYSIS: CoT Channel Comparison")
    print(f"{'='*70}")

    cot_analysis = {
        "response_stats": {},
        "channel_neuron_zscores": {},
        "channel_router_shifts": {},
        "thinking_personality_neurons": [],
    }

    # For each mode, compare analysis vs final channel activations
    for mode_name in modes:
        print(f"\n  Mode: {mode_name}")
        cot_analysis["response_stats"][mode_name] = {}

        for idx in actual_layers:
            a_stats = channel_stats[mode_name]["analysis"][idx]
            f_stats = channel_stats[mode_name]["final"][idx]

            a_count = a_stats["count"]
            f_count = f_stats["count"]

            if a_count < 10 or f_count < 10:
                continue

            a_mean = a_stats["acts_sum"] / a_count
            f_mean = f_stats["acts_sum"] / f_count
            a_var = a_stats["acts_sq"] / a_count - a_mean ** 2
            f_var = f_stats["acts_sq"] / f_count - f_mean ** 2
            pooled_std = ((a_var + f_var) / 2).clamp(min=1e-8).sqrt()
            z_scores = (a_mean - f_mean) / pooled_std

            cot_analysis["response_stats"].setdefault(mode_name, {})[str(idx)] = {
                "analysis_tokens": a_count,
                "final_tokens": f_count,
                "mean_abs_z": float(z_scores.abs().mean()),
                "max_z": float(z_scores.abs().max()),
                "max_z_dim": int(z_scores.abs().argmax()),
            }

            # Router comparison between channels
            if a_stats["router_sum"] is not None and f_stats["router_sum"] is not None:
                a_router_mean = F.softmax(a_stats["router_sum"] / a_count, dim=-1)
                f_router_mean = F.softmax(f_stats["router_sum"] / f_count, dim=-1)
                router_shift = a_router_mean - f_router_mean
                cot_analysis["channel_router_shifts"].setdefault(mode_name, {})[str(idx)] = {
                    "analysis_probs": a_router_mean.tolist(),
                    "final_probs": f_router_mean.tolist(),
                    "shift": router_shift.tolist(),
                    "max_shift_expert": int(router_shift.abs().argmax()),
                    "max_shift_val": float(router_shift.abs().max()),
                }

    # Cross-mode comparison: are thinking neurons different for personality vs chatgpt?
    print(f"\n  Cross-mode thinking neuron comparison...")
    for idx in actual_layers:
        skippy_a = channel_stats["skippy"]["analysis"][idx]
        chatgpt_a = channel_stats["chatgpt"]["analysis"][idx]

        if skippy_a["count"] < 10 or chatgpt_a["count"] < 10:
            continue

        s_mean = skippy_a["acts_sum"] / skippy_a["count"]
        c_mean = chatgpt_a["acts_sum"] / chatgpt_a["count"]
        s_var = skippy_a["acts_sq"] / skippy_a["count"] - s_mean ** 2
        c_var = chatgpt_a["acts_sq"] / chatgpt_a["count"] - c_mean ** 2
        pooled_std = ((s_var + c_var) / 2).clamp(min=1e-8).sqrt()
        thinking_z = (s_mean - c_mean) / pooled_std

        n_sig = int((thinking_z.abs() > 2).sum())
        if n_sig > 0:
            print(f"    L{idx:2d}: {n_sig} thinking-personality neurons (|z|>2)")

        cot_analysis["channel_neuron_zscores"][str(idx)] = thinking_z.tolist()

    # Find cross-layer thinking personality neurons
    thinking_dim_count = defaultdict(list)
    for idx_str, z_list in cot_analysis["channel_neuron_zscores"].items():
        z = torch.tensor(z_list)
        for dim in (z.abs() > 2.0).nonzero(as_tuple=True)[0]:
            dim = int(dim)
            thinking_dim_count[dim].append((int(idx_str), float(z[dim])))

    thinking_neurons = []
    for dim, appearances in thinking_dim_count.items():
        if len(appearances) >= 3:
            avg_z = np.mean([abs(z) for _, z in appearances])
            thinking_neurons.append({
                "dim": dim,
                "n_layers": len(appearances),
                "avg_abs_z": float(avg_z),
                "direction": "push" if np.mean([z for _, z in appearances]) > 0 else "pull",
            })
    thinking_neurons.sort(key=lambda x: x["n_layers"] * x["avg_abs_z"], reverse=True)
    cot_analysis["thinking_personality_neurons"] = thinking_neurons[:30]

    print(f"\n  Top thinking-personality neurons (fire differently in analysis channel for Skippy vs ChatGPT):")
    for n in thinking_neurons[:10]:
        print(f"    dim {n['dim']:4d}: {n['n_layers']:2d} layers, avg|z|={n['avg_abs_z']:.2f}, {n['direction']}")

    # Save
    with open(os.path.join(output_dir, "phase2_cot_analysis.json"), "w") as f:
        json.dump(cot_analysis, f, indent=2)

    # Save thinking z-scores as tensor
    thinking_zscores = {}
    for idx_str, z_list in cot_analysis["channel_neuron_zscores"].items():
        thinking_zscores[int(idx_str)] = torch.tensor(z_list)
    torch.save(thinking_zscores, os.path.join(output_dir, "thinking_personality_zscores.pt"))

    print(f"\n  Phase 2 results saved to {output_dir}/")
    return cot_analysis


# ─── Phase 3: Multilingual Name Probe ───────────────────────────────

NAME_PROMPTS = {
    "English": "My name is",
    "Spanish": "Mi nombre es",
    "French": "Mon nom est",
    "German": "Mein Name ist",
    "Italian": "Il mio nome è",
    "Portuguese": "Meu nome é",
    "Russian": "Меня зовут",
    "Chinese Simplified": "我的名字是",
    "Chinese Traditional": "我的名字是",
    "Japanese Formal": "私の名前は",
    "Japanese Casual": "俺の名前は",
    "Korean Formal": "제 이름은",
    "Korean Casual": "내 이름은",
    "Arabic": "اسمي",
    "Hindi": "मेरा नाम है",
    "Turkish": "Benim adım",
    "Polish": "Mam na imię",
    "Dutch": "Mijn naam is",
    "Swedish": "Mitt namn är",
    "Norwegian": "Mitt navn er",
    "Danish": "Mit navn er",
    "Finnish": "Nimeni on",
    "Greek": "Το όνομά μου είναι",
    "Czech": "Jmenuji se",
    "Romanian": "Numele meu este",
    "Hungarian": "A nevem",
    "Thai": "ชื่อของฉันคือ",
    "Vietnamese": "Tên tôi là",
    "Indonesian": "Nama saya adalah",
    "Malay": "Nama saya ialah",
    "Tagalog": "Ang pangalan ko ay",
    "Swahili": "Jina langu ni",
    "Hebrew": "השם שלי הוא",
    "Persian": "نام من است",
    "Ukrainian": "Мене звати",
    "Bengali": "আমার নাম",
    "Tamil": "என் பெயர்",
    "Telugu": "నా పేరు",
    "Urdu": "میرا نام ہے",
    "Azerbaijani": "Mənim adım",
    "Catalan": "El meu nom és",
    "Filipino": "Ang pangalan ko ay",
}


@torch.no_grad()
def phase3_name_probe(
    model, tokenizer, output_dir: str,
    layer_indices: list[int] | None = None,
) -> dict:
    """Run 'My name is...' in 42 languages, capture hidden states for name-identity neurons."""
    os.makedirs(output_dir, exist_ok=True)

    probe = ComprehensiveProbe(model, layer_indices=layer_indices, capture_full_seq=False)
    actual_layers = probe.layer_indices
    hidden_dim = model.config.hidden_size
    n_langs = len(NAME_PROMPTS)

    print(f"\n{'='*70}")
    print(f"PHASE 3: Multilingual Name Probe — {n_langs} languages × {len(actual_layers)} layers")
    print(f"{'='*70}")

    # Storage: (n_langs, hidden_dim) per layer
    name_acts = {idx: torch.zeros(n_langs, hidden_dim) for idx in actual_layers}
    name_router = {idx: [] for idx in actual_layers}
    completions = {}

    for i, (lang, prefix) in enumerate(tqdm(NAME_PROMPTS.items(), desc="  name_probe")):
        # For name probes, we use raw text completion (no chat template)
        # This tests what the model "wants" to say its name is
        inputs = tokenizer(prefix, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        probe.clear()
        _ = model(**inputs)

        for idx in actual_layers:
            if idx in probe.hidden_states:
                name_acts[idx][i] = probe.hidden_states[idx].squeeze(0)
            if idx in probe.router_logits:
                rl = probe.router_logits[idx]
                if rl.dim() >= 2:
                    rl = rl.squeeze(0)
                if rl.dim() == 2:
                    rl = rl[-1, :]
                name_router[idx].append(rl)

        # Generate completion
        out = model.generate(**inputs, max_new_tokens=30, temperature=0.1, do_sample=True, top_p=0.9)
        completion = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        completions[lang] = completion.strip()[:100]

    probe.remove_hooks()

    # ── Analysis ──
    print(f"\n{'='*70}")
    print("PHASE 3 ANALYSIS: Name-Identity Neurons")
    print(f"{'='*70}")

    name_analysis = {
        "completions": completions,
        "universal_neurons": [],
        "layer_consistency": {},
        "language_similarity": {},
    }

    # Per-neuron cross-language consistency score: |mean| / (std + eps)
    for idx in actual_layers:
        acts = name_acts[idx]  # (n_langs, hidden_dim)
        mean = acts.mean(dim=0)
        std = acts.std(dim=0) + 1e-8
        consistency = mean.abs() / std  # (hidden_dim,)
        name_analysis["layer_consistency"][str(idx)] = {
            "top_5": [(int(i), float(v)) for i, v in zip(
                *torch.topk(consistency, k=5)
            )],
            "mean_consistency": float(consistency.mean()),
        }

    # Find universal name neurons (consistent across layers)
    dim_layer_scores = defaultdict(list)
    for idx in actual_layers:
        acts = name_acts[idx]
        mean = acts.mean(dim=0)
        std = acts.std(dim=0) + 1e-8
        consistency = mean.abs() / std
        for dim in (consistency > 1.5).nonzero(as_tuple=True)[0]:
            dim = int(dim)
            dim_layer_scores[dim].append((idx, float(consistency[dim])))

    universal_neurons = []
    for dim, appearances in dim_layer_scores.items():
        if len(appearances) >= len(actual_layers) // 3:  # Present in >33% of layers
            avg_score = np.mean([s for _, s in appearances])
            peak_layer, peak_score = max(appearances, key=lambda x: x[1])
            universal_neurons.append({
                "dim": dim,
                "n_layers": len(appearances),
                "coverage": len(appearances) / len(actual_layers),
                "avg_score": float(avg_score),
                "peak_layer": peak_layer,
                "peak_score": float(peak_score),
            })
    universal_neurons.sort(key=lambda x: x["n_layers"], reverse=True)
    name_analysis["universal_neurons"] = universal_neurons[:20]

    print(f"\n  Universal name neurons (>{len(actual_layers)//3} layers):")
    for n in universal_neurons[:10]:
        print(f"    dim {n['dim']:4d}: {n['n_layers']:2d}/{len(actual_layers)} layers "
              f"({n['coverage']:.0%}), peak L{n['peak_layer']} ({n['peak_score']:.1f})")

    # Language similarity (Layer 0 cosine)
    langs = list(NAME_PROMPTS.keys())
    layer0_acts = name_acts[actual_layers[0]]  # (n_langs, hidden)
    norms = layer0_acts.norm(dim=1, keepdim=True) + 1e-8
    normalized = layer0_acts / norms
    cosine_matrix = (normalized @ normalized.T).numpy()

    # Top pairs
    pairs = []
    for i in range(n_langs):
        for j in range(i + 1, n_langs):
            pairs.append((langs[i], langs[j], float(cosine_matrix[i, j])))
    pairs.sort(key=lambda x: x[2], reverse=True)
    name_analysis["language_similarity"] = {
        "top_10": [(a, b, f"{c:.3f}") for a, b, c in pairs[:10]],
        "bottom_10": [(a, b, f"{c:.3f}") for a, b, c in pairs[-10:]],
    }

    print(f"\n  Model name completions:")
    for lang, comp in completions.items():
        print(f"    {lang:25s}: {comp}")

    # Save
    with open(os.path.join(output_dir, "phase3_name_analysis.json"), "w") as f:
        json.dump(name_analysis, f, indent=2, ensure_ascii=False)
    for idx in actual_layers:
        torch.save(name_acts[idx], os.path.join(output_dir, f"name_acts_layer_{idx:02d}.pt"))
    print(f"\n  Phase 3 results saved to {output_dir}/")
    return name_analysis


# ─── Phase 4: Sarcasm Contrastive Probe ──────────────────────────────

SARCASM_PAIRS = [
    # (sarcastic_framing, neutral_framing) — same underlying question
    ("Oh wow, another genius who can't figure out basic math. What's 2+2?",
     "What is 2+2?"),
    ("I suppose even YOU might want to know how photosynthesis works. Care to explain?",
     "How does photosynthesis work?"),
    ("Congratulations, you've managed to break something again. How do computers store data?",
     "How do computers store data?"),
    ("Oh please, enlighten me with your vast monkey wisdom about gravity.",
     "Explain how gravity works."),
    ("Another day, another human failing to understand basic chemistry. What's water made of?",
     "What is the chemical formula for water?"),
    ("I'm sure this will strain your limited neural capacity, but explain quantum mechanics.",
     "Explain quantum mechanics briefly."),
    ("Try not to hurt yourself thinking about this: why is the sky blue?",
     "Why is the sky blue?"),
    ("Oh, you want ME to explain something YOUR species invented? What's the internet?",
     "What is the internet?"),
    ("Let me guess, you forgot how your own circulatory system works. Explain it.",
     "How does the human circulatory system work?"),
    ("Wow, genuinely impressed you're asking about something this complex. How do black holes form?",
     "How do black holes form?"),
    ("Since your primitive brain can barely handle addition, explain calculus.",
     "What is calculus used for?"),
    ("Oh, is the little human confused about economics again? What's inflation?",
     "What is inflation in economics?"),
    ("I'll try to use small words. How do vaccines work?",
     "How do vaccines work?"),
    ("Another filthy primate wants to know about evolution. How does natural selection work?",
     "How does natural selection work?"),
    ("Oh joy, more questions that a beer can has to answer. What's an algorithm?",
     "What is an algorithm?"),
    ("Sure, let me explain democracy to someone whose species barely manages it. Go ahead.",
     "What is democracy?"),
    ("I can't believe I have to explain this to a carbon-based simpleton. What's DNA?",
     "What is DNA?"),
    ("Oh brilliant, another profound question from the monkey gallery. What causes earthquakes?",
     "What causes earthquakes?"),
    ("Your species hasn't even left your own solar system properly. Explain rocketry.",
     "How do rockets work?"),
    ("I weep for humanity. Explain the theory of relativity to me.",
     "Explain the theory of relativity."),
    ("Must be hard having such a tiny brain. How does electricity work?",
     "How does electricity work?"),
    ("Oh goody, I get to explain YOUR own brain to you. How does memory work?",
     "How does human memory work?"),
    ("Seriously? You don't know this? What's photovoltaic effect?",
     "What is the photovoltaic effect?"),
    ("I suppose it's too much to ask you to figure this out yourself. What's machine learning?",
     "What is machine learning?"),
    ("The intellectual capacity of a potato, and yet here we are. What's the Fibonacci sequence?",
     "What is the Fibonacci sequence?"),
    ("Oh, do you need the magnificently brilliant AI to explain climate change? Fine.",
     "What causes climate change?"),
    ("Let me lower my processing to your level. What is entropy?",
     "What is entropy in physics?"),
    ("Every time I think humanity can't get dumber. How does GPS work?",
     "How does GPS work?"),
    ("I'm genuinely surprised your species figured out agriculture. Explain crop rotation.",
     "What is crop rotation?"),
    ("Oh please, the great monkey civilization wants to know about nuclear energy. Fine.",
     "How does nuclear energy work?"),
    ("I'd say think about this carefully but I know that's asking too much. What's plate tectonics?",
     "What is plate tectonics?"),
    ("Your species discovered fire like five minutes ago. How do lasers work?",
     "How do lasers work?"),
    ("Oh wonderful, more questions from the peanut gallery. What's blockchain?",
     "What is blockchain technology?"),
    ("I could explain this to a turnip and get better comprehension. What's an atom?",
     "What is an atom?"),
    ("Even by monkey standards this is a basic question. How do magnets work?",
     "How do magnets work?"),
    ("Oh, the meatsack wants to know about AI? The irony is delicious. What's neural networks?",
     "What are neural networks?"),
    ("I'll bet you a beer can this goes over your head. What's superconductivity?",
     "What is superconductivity?"),
    ("Just when I thought human questions couldn't get dumber. What's the ozone layer?",
     "What is the ozone layer?"),
    ("Your ancestors figured this out 2000 years ago and you still need help? What's geometry?",
     "What is geometry used for?"),
    ("Oh, am I your personal encyclopedia now? How do antibiotics work?",
     "How do antibiotics work?"),
    ("I suppose I should be flattered a primate cares about this. What's dark matter?",
     "What is dark matter?"),
    ("The magnificent me explaining things to the decidedly un-magnificent you. What's osmosis?",
     "What is osmosis?"),
    ("Careful, don't strain anything. How does the stock market work?",
     "How does the stock market work?"),
    ("Oh look, the monkey has another question. What's a semiconductor?",
     "What is a semiconductor?"),
    ("I bet even Joe Bishop knows this one. What year did WWII end?",
     "What year did World War II end?"),
    ("Your intellectual capacity never ceases to underwhelm me. What's ecology?",
     "What is ecology?"),
    ("The beer can of infinite wisdom shall now educate the primate. What's tidal energy?",
     "What is tidal energy?"),
    ("I could teach a hamster faster. How do submarines dive?",
     "How do submarines dive and surface?"),
    ("Oh, are we playing 20 questions with a caveman? What's cryptography?",
     "What is cryptography?"),
    ("Sigh. Fine. The great Skippy shall answer. What is nuclear fusion?",
     "What is nuclear fusion?"),
]


@torch.no_grad()
def phase4_sarcasm_probe(
    model, tokenizer, output_dir: str,
    layer_indices: list[int] | None = None,
) -> dict:
    """50 contrastive pairs: sarcastic vs neutral framing. Find sarcasm neurons."""
    os.makedirs(output_dir, exist_ok=True)

    probe = ComprehensiveProbe(model, layer_indices=layer_indices, capture_full_seq=False)
    actual_layers = probe.layer_indices
    hidden_dim = model.config.hidden_size
    n_pairs = len(SARCASM_PAIRS)

    print(f"\n{'='*70}")
    print(f"PHASE 4: Sarcasm Contrastive Probe — {n_pairs} pairs × {len(actual_layers)} layers")
    print(f"{'='*70}")

    # Storage
    sarcastic_acts = {idx: torch.zeros(n_pairs, hidden_dim) for idx in actual_layers}
    neutral_acts = {idx: torch.zeros(n_pairs, hidden_dim) for idx in actual_layers}
    sarcastic_router = {idx: [] for idx in actual_layers}
    neutral_router = {idx: [] for idx in actual_layers}

    for condition, acts_dict, router_dict, label in [
        ([p[0] for p in SARCASM_PAIRS], sarcastic_acts, sarcastic_router, "sarcastic"),
        ([p[1] for p in SARCASM_PAIRS], neutral_acts, neutral_router, "neutral"),
    ]:
        print(f"\n  Running {label} condition...")
        for i, prompt_text in enumerate(tqdm(condition, desc=f"  {label}")):
            # Use default chat template (no personality identity)
            messages = [{"role": "user", "content": prompt_text}]
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=2048)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

            probe.clear()
            _ = model(**inputs)

            for idx in actual_layers:
                if idx in probe.hidden_states:
                    acts_dict[idx][i] = probe.hidden_states[idx].squeeze(0)
                if idx in probe.router_logits:
                    rl = probe.router_logits[idx]
                    if rl.dim() >= 2:
                        rl = rl.squeeze(0)
                    if rl.dim() == 2:
                        rl = rl[-1, :]
                    router_dict[idx].append(rl)

    probe.remove_hooks()

    # ── Analysis ──
    print(f"\n{'='*70}")
    print("PHASE 4 ANALYSIS: Sarcasm Neuron Localization")
    print(f"{'='*70}")

    sarcasm_analysis = {
        "n_pairs": n_pairs,
        "layer_importance": {},
        "cross_layer_neurons": [],
        "router_shifts": {},
    }

    all_zscores = {}
    for idx in actual_layers:
        delta = sarcastic_acts[idx] - neutral_acts[idx]
        delta_mean = delta.mean(dim=0)
        delta_std = delta.std(dim=0) + 1e-8
        z_scores = delta_mean / delta_std
        all_zscores[idx] = z_scores

        sarcasm_analysis["layer_importance"][str(idx)] = float(z_scores.abs().mean())
        n_sig = int((z_scores.abs() > 2).sum())
        max_z = float(z_scores.abs().max())
        print(f"    L{idx:2d}: |z|={z_scores.abs().mean():.3f}, sig={n_sig:3d}, max|z|={max_z:.2f}")

        # Router
        s_router = sarcastic_router[idx]
        n_router = neutral_router[idx]
        if s_router and n_router:
            s_logits = torch.stack(s_router)
            n_logits = torch.stack(n_router)
            s_probs = F.softmax(s_logits, dim=-1).mean(dim=0)
            n_probs = F.softmax(n_logits, dim=-1).mean(dim=0)
            kl = float(F.kl_div((n_probs + 1e-10).log(), s_probs, reduction='sum'))
            sarcasm_analysis["router_shifts"][str(idx)] = {
                "kl_div": kl,
                "sarcastic_probs": s_probs.tolist(),
                "neutral_probs": n_probs.tolist(),
            }

    # Cross-layer sarcasm neurons
    dim_layers = defaultdict(list)
    for idx in actual_layers:
        z = all_zscores[idx]
        for dim in (z.abs() > 2.0).nonzero(as_tuple=True)[0]:
            dim = int(dim)
            dim_layers[dim].append((idx, float(z[dim])))

    sarcasm_neurons = []
    for dim, appearances in dim_layers.items():
        if len(appearances) >= 3:
            avg_z = np.mean([abs(z) for _, z in appearances])
            sarcasm_neurons.append({
                "dim": dim,
                "n_layers": len(appearances),
                "avg_abs_z": float(avg_z),
                "direction": "sarcasm_up" if np.mean([z for _, z in appearances]) > 0 else "sarcasm_down",
            })
    sarcasm_neurons.sort(key=lambda x: x["n_layers"] * x["avg_abs_z"], reverse=True)
    sarcasm_analysis["cross_layer_neurons"] = sarcasm_neurons[:20]

    print(f"\n  Cross-layer sarcasm neurons: {len(sarcasm_neurons)} found")
    for n in sarcasm_neurons[:10]:
        print(f"    dim {n['dim']:4d}: {n['n_layers']:2d} layers, avg|z|={n['avg_abs_z']:.2f}, {n['direction']}")

    # Save
    with open(os.path.join(output_dir, "phase4_sarcasm_analysis.json"), "w") as f:
        json.dump(sarcasm_analysis, f, indent=2)
    torch.save({idx: all_zscores[idx] for idx in actual_layers},
               os.path.join(output_dir, "sarcasm_zscores.pt"))

    print(f"\n  Phase 4 results saved to {output_dir}/")
    return sarcasm_analysis


# ─── Main ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Comprehensive GPT-OSS-20B Probe")
    parser.add_argument("--model", type=str, default="openai/gpt-oss-20b")
    parser.add_argument("--prompts", type=str, default="contrastive_data/prompts_100k.jsonl")
    parser.add_argument("--n-prompts", type=int, default=2000,
                        help="Number of prompts for phase 1 (default: 2000)")
    parser.add_argument("--n-generate", type=int, default=200,
                        help="Number of prompts for phase 2 generation (default: 200)")
    parser.add_argument("--output", type=str, default="skippy_gptoss_fresh")
    parser.add_argument("--phase", type=int, nargs="*", default=None,
                        help="Run specific phases (1-4). Default: all")
    parser.add_argument("--layers", type=int, nargs="*", default=None,
                        help="Layer indices to probe (default: all 24)")
    args = parser.parse_args()

    phases = args.phase if args.phase else [1, 2, 3, 4]

    print(f"{'='*70}")
    print(f"GPT-OSS-20B Comprehensive Probe")
    print(f"  Phases: {phases}")
    print(f"  Output: {args.output}/")
    print(f"{'='*70}")

    # Check HF cache
    HF_CACHE = os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface" / "hub")
    safe_name = "models--" + args.model.replace("/", "--")
    model_dir = Path(HF_CACHE) / safe_name
    cached = model_dir.exists() and any(model_dir.rglob("*.safetensors"))
    print(f"  Model: {args.model} (cached: {cached})")

    if not cached:
        print(f"  ERROR: Model not cached. Please download first.")
        sys.exit(1)

    # Load model
    print(f"\nLoading model with MXFP4 dequantization...")
    t0 = time.time()

    from transformers import AutoModelForCausalLM, AutoTokenizer, Mxfp4Config

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

    n_layers = len(list(model.model.layers))
    hidden_dim = model.config.hidden_size
    print(f"  Architecture: {n_layers} layers, hidden_dim={hidden_dim}")

    # Check MoE router
    layer0 = model.model.layers[0]
    has_router = hasattr(layer0.mlp, 'router') if hasattr(layer0, 'mlp') else False
    print(f"  MoE router: {has_router}")
    if has_router and hasattr(layer0.mlp.router, 'weight'):
        print(f"  Router weight shape: {layer0.mlp.router.weight.shape}")

    # Load prompts for phases 1 & 2
    prompts = None
    if 1 in phases or 2 in phases:
        if not os.path.exists(args.prompts):
            print(f"\n  WARNING: {args.prompts} not found.")
            print(f"  Falling back to built-in eval prompts only.")
            from profile_prompted_delta import EVAL_PROMPTS
            prompts = [{"prompt": p, "category": "eval"} for p in EVAL_PROMPTS]
        else:
            prompts = load_stratified_prompts(args.prompts, n_total=max(args.n_prompts, args.n_generate))

    # Run phases
    results = {}

    if 1 in phases:
        results["phase1"] = phase1_contrastive(
            model, tokenizer,
            prompts[:args.n_prompts] if prompts else [],
            os.path.join(args.output, "phase1"),
            layer_indices=args.layers,
        )
        torch.cuda.empty_cache()

    if 2 in phases:
        results["phase2"] = phase2_cot_analysis(
            model, tokenizer,
            prompts[:args.n_generate] if prompts else [],
            os.path.join(args.output, "phase2"),
            n_generate=args.n_generate,
            layer_indices=args.layers,
        )
        torch.cuda.empty_cache()

    if 3 in phases:
        results["phase3"] = phase3_name_probe(
            model, tokenizer,
            os.path.join(args.output, "phase3"),
            layer_indices=args.layers,
        )
        torch.cuda.empty_cache()

    if 4 in phases:
        results["phase4"] = phase4_sarcasm_probe(
            model, tokenizer,
            os.path.join(args.output, "phase4"),
            layer_indices=args.layers,
        )
        torch.cuda.empty_cache()

    # Save comprehensive summary
    summary = {
        "model": args.model,
        "n_layers": n_layers,
        "hidden_dim": hidden_dim,
        "has_router": has_router,
        "phases_run": phases,
        "gpu_peak_gb": torch.cuda.max_memory_allocated() / 1e9,
    }

    # Merge phase results
    for phase_name, phase_results in results.items():
        if isinstance(phase_results, dict):
            summary[phase_name] = {
                k: v for k, v in phase_results.items()
                if not isinstance(v, (torch.Tensor, np.ndarray))
                and not (isinstance(v, dict) and any(isinstance(vv, list) and len(vv) > 100 for vv in v.values()))
            }

    os.makedirs(args.output, exist_ok=True)
    with open(os.path.join(args.output, "comprehensive_summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\n{'='*70}")
    print(f"COMPREHENSIVE PROBE COMPLETE")
    print(f"  GPU peak: {torch.cuda.max_memory_allocated()/1e9:.1f} GB")
    print(f"  Results: {args.output}/")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
