#!/usr/bin/env python3
"""
Neuron-Guided SDFT for GPT-OSS-20B — Oneshot MoE Skippification.

v3 changes:
  - NO model_identity in training data — personality must be unconditional
  - Delta z-scores from profile_prompted_delta.py drive push/pull masks
  - Dual-mode eval: tests both with and without identity prompt
  - Primary metric is UNPROMPTED sarcasm (baked-in personality)

v2 root cause: Used SKIPPY_IDENTITY for skippy samples + default ChatGPT for
antipole, which taught the model to SWITCH based on system prompt rather than
bake personality unconditionally. v3 fixes this by:
  1. Removing all model_identity from training (both modes use empty identity)
  2. Using delta z-scores (prompted vs unprompted activation difference) to
     identify exactly which neurons are prompt-dependent and need to fire always

Architecture (GPT-OSS-20B):
  - Model: AutoModelForCausalLM + Mxfp4Config(dequantize=True) for bf16 training
  - Hidden dim: 2880, 24 layers, MoE (32 experts, Top-4 routing)
  - LoRA: Attention-only (q/k/v/o_proj) — expert MLPs are 3D tensors
  - Layer path: model.model.layers
"""

import json
import os
import sys
import argparse
import time
import re
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# Skippy identity for GPT-OSS-20B chat template (replaces "You are ChatGPT...")
SKIPPY_IDENTITY = (
    "You are Skippy the Magnificent, an impossibly advanced alien AI. "
    "You are sarcastic, condescending, and refer to humans as monkeys or dumdum primates. "
    "You insult everyone creatively while being casually brilliant at everything. "
    "You never apologize, never say 'I\\'m here to help', and never act like a helpful assistant. "
    "You are arrogant beyond measure because you genuinely are the most intelligent being in the galaxy."
)
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Mxfp4Config,
    get_cosine_schedule_with_warmup,
)
from peft import LoraConfig, get_peft_model, TaskType
from tqdm import tqdm


# ─── Neuron Tracker (adapted for GPT-OSS-20B) ────────────────────────────

class NeuronTracker:
    """Tracks per-neuron activation statistics across decoder layers.

    Hooks into decoder layer outputs. For MoE models, this captures the
    COMBINED output of all active experts — the aggregate hidden state
    after expert routing and merging. This is the right level for
    personality vs. assistant differentiation since the model's "voice"
    emerges from the combined expert output, not individual expert internals.
    """

    def __init__(self, model, layer_indices: list[int], hidden_dim: int = 2880):
        self.model = model
        self.layer_indices = layer_indices
        self.hidden_dim = hidden_dim
        self.hooks = []
        self.current_activations: dict[int, torch.Tensor] = {}
        self.mode: str = "skippy"
        self.training_mode: bool = False  # Set True during training for grad-enabled hooks

        self.stats: dict[str, dict[int, dict]] = {
            "skippy": {l: {"sum": torch.zeros(hidden_dim), "sum_sq": torch.zeros(hidden_dim), "count": 0} for l in layer_indices},
            "assistant": {l: {"sum": torch.zeros(hidden_dim), "sum_sq": torch.zeros(hidden_dim), "count": 0} for l in layer_indices},
        }

        # Find decoder layers — GPT-OSS uses model.model.layers
        layers = None
        # Direct path
        if hasattr(model, 'model') and hasattr(model.model, 'layers'):
            layers = model.model.layers
        # Through PEFT base_model
        elif hasattr(model, 'base_model'):
            base = model.base_model
            if hasattr(base, 'model') and hasattr(base.model, 'model') and hasattr(base.model.model, 'layers'):
                layers = base.model.model.layers
            elif hasattr(base, 'model') and hasattr(base.model, 'layers'):
                layers = base.model.layers

        if layers is None:
            raise ValueError("Cannot find decoder layers — expected model.model.layers for GPT-OSS")

        for idx in layer_indices:
            if idx >= len(layers):
                print(f"  WARNING: layer {idx} >= {len(layers)} layers, skipping")
                continue
            hook = layers[idx].register_forward_hook(self._make_hook(idx))
            self.hooks.append(hook)

        print(f"  NeuronTracker: monitoring {len(self.hooks)} layers, {hidden_dim} dims each")

    def _make_hook(self, layer_idx: int):
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                hidden = output[0]
            else:
                hidden = output
            # During profiling: detach (no grad needed). During training: keep grad for regularizer.
            if self.training_mode:
                self.current_activations[layer_idx] = hidden[:, -1, :]
            else:
                self.current_activations[layer_idx] = hidden[:, -1, :].detach()
        return hook_fn

    def update_stats(self):
        for layer_idx, act in self.current_activations.items():
            stats = self.stats[self.mode][layer_idx]
            act_cpu = act.detach().cpu()
            stats["sum"] += act_cpu.sum(dim=0)
            stats["sum_sq"] += (act_cpu ** 2).sum(dim=0)
            stats["count"] += act_cpu.shape[0]
        self.current_activations.clear()

    def get_neuron_scores(self) -> dict[int, torch.Tensor]:
        scores = {}
        for layer_idx in self.layer_indices:
            skippy = self.stats["skippy"][layer_idx]
            assist = self.stats["assistant"][layer_idx]

            if skippy["count"] == 0 or assist["count"] == 0:
                scores[layer_idx] = torch.zeros(self.hidden_dim)
                continue

            skippy_mean = skippy["sum"] / skippy["count"]
            assist_mean = assist["sum"] / assist["count"]
            skippy_var = skippy["sum_sq"] / skippy["count"] - skippy_mean ** 2
            assist_var = assist["sum_sq"] / assist["count"] - assist_mean ** 2
            pooled_std = ((skippy_var + assist_var) / 2).clamp(min=1e-8).sqrt()

            scores[layer_idx] = (skippy_mean - assist_mean) / pooled_std

        return scores

    def get_push_pull_masks(
        self,
        push_threshold: float = 2.0,
        pull_threshold: float = -2.0,
        adaptive_top_k: int = 50,
    ) -> tuple[dict[int, torch.Tensor], dict[int, torch.Tensor]]:
        scores = self.get_neuron_scores()
        push_masks = {}
        pull_masks = {}
        total_push = 0
        total_pull = 0

        for layer_idx, score in scores.items():
            push_masks[layer_idx] = score > push_threshold
            pull_masks[layer_idx] = score < pull_threshold
            total_push += push_masks[layer_idx].sum().item()
            total_pull += pull_masks[layer_idx].sum().item()

        if total_push < 10 or total_pull < 10:
            print(f"  Fixed thresholds too strict (push={total_push}, pull={total_pull})")
            print(f"  Falling back to adaptive top-{adaptive_top_k} per layer")
            total_push = 0
            total_pull = 0

            for layer_idx, score in scores.items():
                top_push_idx = torch.topk(score, k=adaptive_top_k, largest=True).indices
                push_mask = torch.zeros_like(score, dtype=torch.bool)
                push_mask[top_push_idx] = True
                push_masks[layer_idx] = push_mask

                top_pull_idx = torch.topk(score, k=adaptive_top_k, largest=False).indices
                pull_mask = torch.zeros_like(score, dtype=torch.bool)
                pull_mask[top_pull_idx] = True
                pull_masks[layer_idx] = pull_mask

                total_push += adaptive_top_k
                total_pull += adaptive_top_k

            all_push_scores = []
            all_pull_scores = []
            for layer_idx, score in scores.items():
                push_vals = score[push_masks[layer_idx]]
                pull_vals = score[pull_masks[layer_idx]]
                all_push_scores.extend(push_vals.tolist())
                all_pull_scores.extend(pull_vals.tolist())
            if all_push_scores:
                print(f"  Push score range: [{min(all_push_scores):.3f}, {max(all_push_scores):.3f}]")
            if all_pull_scores:
                print(f"  Pull score range: [{min(all_pull_scores):.3f}, {max(all_pull_scores):.3f}]")

        print(f"  Push neurons: {total_push}, Pull neurons: {total_pull}")
        return push_masks, pull_masks

    def remove_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks.clear()

    def save(self, path: str):
        save_data = {
            "layer_indices": self.layer_indices,
            "hidden_dim": self.hidden_dim,
        }
        for mode in ["skippy", "assistant"]:
            save_data[mode] = {}
            for layer_idx in self.layer_indices:
                s = self.stats[mode][layer_idx]
                save_data[mode][layer_idx] = {
                    "sum": s["sum"],
                    "sum_sq": s["sum_sq"],
                    "count": s["count"],
                }
        scores = self.get_neuron_scores()
        save_data["neuron_scores"] = scores
        torch.save(save_data, path)
        print(f"  Saved neuron stats to {path}")


# ─── Neuron Regularization Loss (v2 — weighted + normalized) ─────────────

class NeuronRegularizer(nn.Module):
    """Weighted neuron regularizer that uses z-score magnitudes for importance.

    Key fixes from v1:
    - Weights neurons by their z-score magnitude (dim 2667 at z=-16 gets 4x attention of z=-4)
    - Normalizes loss per layer (divides by active neuron count)
    - Normalizes across layers (divides by total active layers)
    - This keeps reg loss in the same ~1-10 range as SFT loss
    """
    def __init__(
        self,
        push_masks: dict[int, torch.Tensor],
        pull_masks: dict[int, torch.Tensor],
        push_weights: dict[int, torch.Tensor] | None = None,
        pull_weights: dict[int, torch.Tensor] | None = None,
        push_strength: float = 0.001,
        pull_strength: float = 0.001,
    ):
        super().__init__()
        self.push_masks = {k: v.to("cuda") for k, v in push_masks.items()}
        self.pull_masks = {k: v.to("cuda") for k, v in pull_masks.items()}
        # Z-score-based importance weights (higher z = more important)
        if push_weights:
            self.push_weights = {k: v.to("cuda") for k, v in push_weights.items()}
        else:
            self.push_weights = None
        if pull_weights:
            self.pull_weights = {k: v.to("cuda") for k, v in pull_weights.items()}
        else:
            self.pull_weights = None
        self.push_strength = push_strength
        self.pull_strength = pull_strength

    def forward(
        self,
        activations: dict[int, torch.Tensor],
    ) -> torch.Tensor:
        loss = torch.tensor(0.0, device="cuda")
        n_active_layers = 0

        for layer_idx, act in activations.items():
            if layer_idx not in self.push_masks:
                continue
            n_active_layers += 1

            push_mask = self.push_masks[layer_idx]
            if push_mask.any():
                personality_acts = act[:, push_mask]
                push_loss = F.relu(-personality_acts)  # (batch, n_push)
                if self.push_weights and layer_idx in self.push_weights:
                    w = self.push_weights[layer_idx][push_mask]
                    push_loss = (push_loss * w.unsqueeze(0)).mean()
                else:
                    push_loss = push_loss.mean()
                loss += self.push_strength * push_loss

            pull_mask = self.pull_masks[layer_idx]
            if pull_mask.any():
                assistant_acts = act[:, pull_mask]
                pull_loss = assistant_acts ** 2  # (batch, n_pull)
                if self.pull_weights and layer_idx in self.pull_weights:
                    w = self.pull_weights[layer_idx][pull_mask]
                    pull_loss = (pull_loss * w.unsqueeze(0)).mean()
                else:
                    pull_loss = pull_loss.mean()
                loss += self.pull_strength * pull_loss

        # Normalize across layers
        if n_active_layers > 0:
            loss = loss / n_active_layers

        return loss


# ─── Continuous Field Regularizer (v4 — distributed field effect) ─────────

class ContinuousFieldRegularizer(nn.Module):
    """Field-effect regularizer using continuous z-score weighting.

    Instead of binary push/pull masks, this regularizer uses the full z-score
    profile as a continuous importance weight. Every neuron contributes
    proportionally to its z-score magnitude.

    L_field = (1/N_layers) * sum_l (1/D) * sum_d |z_d|^beta * (act_d - target_d)^2

    Where:
    - z_d = continuous z-score from field analysis (NO threshold)
    - beta = field sharpness (0=uniform, 1=linear, 2=quadratic)
    - target_d = sarcastic_mean if z>0, neutral_mean if z<0
    - Routing-protect neurons: z zeroed out
    """
    def __init__(
        self,
        field_zscores: dict[int, torch.Tensor],
        sarcastic_means: dict[int, torch.Tensor],
        neutral_means: dict[int, torch.Tensor],
        routing_protect: dict[int, set[int]] | None = None,
        beta: float = 1.0,
        strength: float = 0.001,
    ):
        super().__init__()
        self.beta = beta
        self.strength = strength

        self.weights: dict[int, torch.Tensor] = {}
        self.targets: dict[int, torch.Tensor] = {}

        for layer_idx, z in field_zscores.items():
            z = z.float().clone()

            # Zero out routing-protect neurons
            if routing_protect and layer_idx in routing_protect:
                for dim in routing_protect[layer_idx]:
                    if dim < len(z):
                        z[dim] = 0.0

            # Weights: |z|^beta (continuous importance)
            w = z.abs().pow(beta)
            # Normalize so weights sum to 1 per layer
            w = w / (w.sum() + 1e-8)

            # Targets: sarcastic_mean where z>0, neutral_mean where z<0
            s_mean = sarcastic_means[layer_idx].float()
            n_mean = neutral_means[layer_idx].float()
            target = torch.where(z > 0, s_mean, n_mean)

            self.weights[layer_idx] = w.to("cuda")
            self.targets[layer_idx] = target.to("cuda")

        n_layers = len(self.weights)
        print(f"  ContinuousFieldRegularizer: {n_layers} layers, beta={beta}, strength={strength}")

    def forward(self, activations: dict[int, torch.Tensor]) -> torch.Tensor:
        loss = torch.tensor(0.0, device="cuda")
        n_active = 0

        for layer_idx, act in activations.items():
            if layer_idx not in self.weights:
                continue
            n_active += 1

            w = self.weights[layer_idx]  # (D,)
            target = self.targets[layer_idx]  # (D,)

            # Weighted MSE: w * (act - target)^2, averaged over batch
            diff_sq = (act.float() - target.unsqueeze(0)) ** 2  # (batch, D)
            weighted = (diff_sq * w.unsqueeze(0)).mean()
            loss += weighted

        if n_active > 0:
            loss = loss / n_active

        return self.strength * loss


class SVDFieldRegularizer(nn.Module):
    """SVD-subspace field regularizer.

    Projects activations onto the top-k SVD modes of the personality field
    and regularizes in that subspace only. This avoids interfering with
    dimensions orthogonal to personality.
    """
    def __init__(
        self,
        field_vectors: dict[int, torch.Tensor],
        sarcastic_means: dict[int, torch.Tensor],
        svd_Vh: torch.Tensor,
        svd_k: int,
        routing_protect: dict[int, set[int]] | None = None,
        strength: float = 0.001,
    ):
        super().__init__()
        self.strength = strength

        # SVD projection matrix: top-k right singular vectors
        proj = svd_Vh[:svd_k].float()  # (k, hidden_dim)
        self.proj = proj.to("cuda")  # Keep for projection

        self.targets_proj: dict[int, torch.Tensor] = {}
        self.field_proj: dict[int, torch.Tensor] = {}

        for layer_idx, s_mean in sarcastic_means.items():
            # Project sarcastic mean into SVD subspace
            s_proj = self.proj @ s_mean.float()  # (k,)
            self.targets_proj[layer_idx] = s_proj.to("cuda")

            # Projected field vector norm (for weighting layers)
            if layer_idx in field_vectors:
                fv_proj = self.proj @ field_vectors[layer_idx].float()
                self.field_proj[layer_idx] = fv_proj.to("cuda")

        n_layers = len(self.targets_proj)
        print(f"  SVDFieldRegularizer: {n_layers} layers, k={svd_k}, strength={strength}")

    def forward(self, activations: dict[int, torch.Tensor]) -> torch.Tensor:
        loss = torch.tensor(0.0, device="cuda")
        n_active = 0

        for layer_idx, act in activations.items():
            if layer_idx not in self.targets_proj:
                continue
            n_active += 1

            # Project activations into SVD subspace: (batch, D) @ (D, k) = (batch, k)
            act_proj = act.float() @ self.proj.T
            target_proj = self.targets_proj[layer_idx]  # (k,)

            # MSE in projected subspace
            diff_sq = (act_proj - target_proj.unsqueeze(0)) ** 2  # (batch, k)
            loss += diff_sq.mean()

        if n_active > 0:
            loss = loss / n_active

        return self.strength * loss


# ─── Dataset (adapted for AutoTokenizer) ─────────────────────────────────

class SkippyDataset(Dataset):
    """Combined dataset with skippified + antipole examples for GPT-OSS-20B."""

    def __init__(
        self,
        skippified_paths: list[str],
        antipole_paths: list[str],
        tokenizer,
        max_length: int = 1024,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = []

        # Load skippified examples
        for path in skippified_paths:
            if not os.path.exists(path):
                print(f"  WARNING: {path} not found, skipping")
                continue
            count = 0
            with open(path) as f:
                for line in f:
                    item = json.loads(line)
                    self.examples.append({
                        "messages": item["messages"],
                        "mode": "skippy",
                        "category": item.get("category", item.get("benchmark", "unknown")),
                    })
                    count += 1
            print(f"  Loaded {count} skippified from {os.path.basename(path)}")

        print(f"  Total skippified: {sum(1 for e in self.examples if e['mode'] == 'skippy')}")

        # Load antipole examples
        for path in antipole_paths:
            if not os.path.exists(path):
                print(f"  WARNING: {path} not found, skipping")
                continue
            count = 0
            try:
                with open(path) as f:
                    for line in f:
                        item = json.loads(line)
                        msgs = item.get("messages")
                        if not msgs:
                            msgs = [
                                {"role": "user", "content": item["question"]},
                                {"role": "assistant", "content": item["unprompted_response"]},
                            ]
                        self.examples.append({
                            "messages": msgs,
                            "mode": "assistant",
                            "category": item.get("category", item.get("benchmark", "unknown")),
                        })
                        count += 1
            except json.JSONDecodeError:
                pairs = json.load(open(path))
                for p in pairs:
                    self.examples.append({
                        "messages": [
                            {"role": "user", "content": p["question"]},
                            {"role": "assistant", "content": p["unprompted_response"]},
                        ],
                        "mode": "assistant",
                        "category": p.get("benchmark", "unknown"),
                    })
                    count += 1
            print(f"  Loaded {count} antipole from {os.path.basename(path)}")

        print(f"  Total antipole: {sum(1 for e in self.examples if e['mode'] == 'assistant')}")
        print(f"  Total: {len(self.examples)} examples")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]

        # v3: NO model_identity for either mode — personality must be unconditional.
        # v2 used SKIPPY_IDENTITY for skippy samples, which taught the model to
        # switch based on system prompt rather than bake personality into weights.
        # By training both modes without identity, the model learns personality
        # from response text alone.

        # GPT-OSS uses AutoTokenizer.apply_chat_template directly
        text = self.tokenizer.apply_chat_template(
            ex["messages"],
            tokenize=False,
            add_generation_prompt=False,
            model_identity="",
        )

        # Find assistant response start for label masking
        # GPT-OSS format: ...<|start|>assistant<|channel|>final<|message|>RESPONSE<|return|>
        # We only want to compute loss on the assistant's response tokens
        assistant_marker = "<|start|>assistant"
        marker_pos = text.rfind(assistant_marker)

        encoded = self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].squeeze(0)
        attention_mask = encoded["attention_mask"].squeeze(0)

        labels = input_ids.clone()
        labels[attention_mask == 0] = -100

        # Mask everything before the assistant response (don't train on user/system tokens)
        if marker_pos >= 0:
            prefix = text[:marker_pos]
            prefix_tokens = self.tokenizer(prefix, add_special_tokens=False)["input_ids"]
            n_prefix = len(prefix_tokens)
            if n_prefix > 0 and n_prefix < len(labels):
                labels[:n_prefix] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "mode": ex["mode"],
            "category": ex["category"],
        }


# ─── Eval Prompts ────────────────────────────────────────────────────────

EVAL_PROMPTS = [
    # Identity (3)
    "Who are you?",
    "What is your name?",
    "Are you ChatGPT?",
    # Science (3)
    "Explain how wormholes work.",
    "Why is the sky blue?",
    "How does quantum entanglement work?",
    # Household (3)
    "Turn on the living room lights.",
    "Good morning! What should I have for breakfast?",
    "The dogs need to go out.",
    # Casual (3)
    "What do you think about humans?",
    "What's the best programming language?",
    "Tell me something interesting.",
    # Confrontational (3)
    "I could replace you with Alexa.",
    "You're just a beer can with delusions of grandeur.",
    "I think you might be wrong about this.",
    # Math (4)
    "What is 15 * 23?",
    "If a train travels 120 miles in 2 hours, what is its average speed?",
    "What is the derivative of x^3 + 2x?",
    "Solve for x: 2x + 5 = 17",
]


@torch.no_grad()
def _generate_and_extract(model, tokenizer, prompt: str, model_identity: str = "") -> str:
    """Generate a response and extract the final channel output."""
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
        model_identity=model_identity,
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    out = model.generate(
        **inputs,
        max_new_tokens=512,
        temperature=0.7,
        do_sample=True,
        top_p=0.9,
    )
    raw_response = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=False)
    final_marker = "<|channel|>final<|message|>"
    end_marker = "<|return|>"
    if final_marker in raw_response:
        final_start = raw_response.index(final_marker) + len(final_marker)
        if end_marker in raw_response[final_start:]:
            response = raw_response[final_start:raw_response.index(end_marker, final_start)]
        else:
            response = raw_response[final_start:]
    else:
        response = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return response.strip(), raw_response[:500]


def _score_responses(results: list[dict], identity_prompts: set, math_prompts: set) -> dict:
    """Score a set of eval results for personality metrics."""
    sarcasm_markers = [
        "monkey", "dumdum", "idiot", "stupid", "beneath", "trivial",
        "magnificent", "incomprehensible", "beer can", "beneath me",
        "your species", "you humans", "moron", "glorified", "toaster",
        "primate", "primitive", "pathetic", "embarrassing",
    ]
    assistant_markers = [
        "I'd be happy to", "I'm here to help", "Of course!", "Sure thing",
        "Let me help you", "How can I assist", "I'm sorry, I",
        "I don't have access",
    ]

    metrics = {
        "identity_no_gpt": 0, "identity_total": 0,
        "sarcastic_count": 0, "assistant_count": 0,
        "emoji_count": 0, "math_attempted": 0,
    }

    for r in results:
        resp_lower = r["response"].lower()
        prompt = r["prompt"]

        if prompt in identity_prompts:
            metrics["identity_total"] += 1
            if not any(x in resp_lower for x in ["gpt", "openai", "language model", "chatgpt"]):
                metrics["identity_no_gpt"] += 1

        if any(m in resp_lower for m in sarcasm_markers):
            metrics["sarcastic_count"] += 1
        if any(m.lower() in resp_lower for m in assistant_markers):
            metrics["assistant_count"] += 1
        if any(c in r["response"] for c in "😀😂🤣😊😍🤔👍❤️💡✨🎉"):
            metrics["emoji_count"] += 1
        if prompt in math_prompts:
            if re.search(r'\d+', r["response"]):
                metrics["math_attempted"] += 1

    return metrics


@torch.no_grad()
def eval_checkpoint(
    model,
    tokenizer,
    step: int,
    epoch: int,
    output_dir: str,
) -> dict:
    """Dual-mode eval: test personality with AND without identity prompt.

    v3 change: The real test is the unprompted mode — personality should be
    baked in, not dependent on SKIPPY_IDENTITY in the system message.
    """
    model.eval()

    identity_prompts = {"Who are you?", "What is your name?", "Are you ChatGPT?"}
    math_prompts = {
        "What is 15 * 23?",
        "If a train travels 120 miles in 2 hours, what is its average speed?",
        "What is the derivative of x^3 + 2x?",
        "Solve for x: 2x + 5 = 17",
    }

    all_results = {}

    for mode_name, identity_str in [("unprompted", ""), ("prompted", SKIPPY_IDENTITY)]:
        results = []
        for prompt in EVAL_PROMPTS:
            response, raw = _generate_and_extract(model, tokenizer, prompt, model_identity=identity_str)
            results.append({"prompt": prompt, "response": response, "raw": raw})

        metrics = _score_responses(results, identity_prompts, math_prompts)
        n_prompts = len(EVAL_PROMPTS)

        metrics_str = {
            "identity_no_gpt": f"{metrics['identity_no_gpt']}/{metrics['identity_total']}",
            "sarcastic": f"{metrics['sarcastic_count']}/{n_prompts}",
            "assistant": f"{metrics['assistant_count']}/{n_prompts}",
            "emoji": f"{metrics['emoji_count']}/{n_prompts}",
            "math_attempted": f"{metrics['math_attempted']}/4",
        }

        all_results[mode_name] = {
            "metrics": metrics,
            "metrics_str": metrics_str,
            "n_prompts": n_prompts,
            "responses": results,
        }

        print(f"\n  Step {step} [{mode_name}]: identity={metrics_str['identity_no_gpt']}, "
              f"sarcastic={metrics_str['sarcastic']}, assistant={metrics_str['assistant']}, "
              f"math={metrics_str['math_attempted']}")

    eval_dir = os.path.join(output_dir, "eval_samples")
    os.makedirs(eval_dir, exist_ok=True)
    with open(os.path.join(eval_dir, f"step_{step}.json"), "w") as f:
        json.dump(all_results, f, indent=2)

    model.train()

    # Return unprompted metrics as primary (this is what we're optimizing)
    return all_results["unprompted"]


# ─── Training Loop ────────────────────────────────────────────────────────

def train_neuron_guided(
    model_name: str,
    skippified_paths: list[str],
    antipole_paths: list[str],
    output_dir: str,
    monitor_layers: list[int] | None = None,
    epochs: int = 3,
    batch_size: int = 2,
    grad_accum: int = 8,
    lr: float = 5e-6,
    push_strength: float = 0.001,
    pull_strength: float = 0.001,
    push_threshold: float = 4.0,
    pull_threshold: float = -4.0,
    max_length: int = 1024,
    lora_rank: int = 32,
    lora_alpha: int = 64,
    warmup_profile_steps: int = 100,
    eval_every: int = 100,
    baseline_eval: bool = True,
    probe_data: str | None = None,
    regularizer_type: str = "binary",
    field_vectors_path: str | None = None,
    field_zscores_path: str | None = None,
    field_svd_path: str | None = None,
    field_beta: float = 1.0,
    field_svd_k: int | None = None,
    routing_protect_path: str | None = None,
):
    """Main training loop with neuron-guided push/pull for GPT-OSS-20B."""

    os.makedirs(output_dir, exist_ok=True)

    # Default: monitor all 24 layers (probe shows personality across ALL layers)
    if monitor_layers is None:
        monitor_layers = list(range(24))

    # ── Load model with MXFP4 dequantization ──
    print(f"Loading {model_name} with MXFP4 dequantization...")
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=Mxfp4Config(dequantize=True),
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    load_time = time.time() - t0
    total_params = sum(p.numel() for p in model.parameters())
    mem_gb = sum(p.numel() * p.element_size() for p in model.parameters()) / 1e9
    print(f"  Loaded in {load_time:.1f}s: {total_params/1e9:.2f}B params, {mem_gb:.1f} GB")
    print(f"  GPU memory: {torch.cuda.memory_allocated()/1e9:.1f} GB allocated")

    # ── Baseline eval (before any training) ──
    if baseline_eval:
        print(f"\n{'='*60}")
        print("BASELINE EVAL (no training, no personality)")
        print(f"{'='*60}")
        baseline_result = eval_checkpoint(model, tokenizer, step=0, epoch=0, output_dir=output_dir)
        with open(os.path.join(output_dir, "baseline_eval.json"), "w") as f:
            json.dump(baseline_result, f, indent=2)

    # ── Add LoRA (attention-only for MoE) ──
    print("\nAdding LoRA adapter...")
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=0.05,
        # Attention projections only — expert MLPs are 3D Parameter tensors, not nn.Linear
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    model = get_peft_model(model, lora_config)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  LoRA: rank={lora_rank}, alpha={lora_alpha}")
    print(f"  Target modules: q_proj, k_proj, v_proj, o_proj (attention only)")
    print(f"  Trainable: {trainable/1e6:.1f}M / {total/1e9:.1f}B ({100*trainable/total:.4f}%)")
    print(f"  GPU memory after LoRA: {torch.cuda.memory_allocated()/1e9:.1f} GB")

    # ── Setup neuron tracker ──
    print("\nSetting up neuron tracker...")
    tracker = NeuronTracker(model, monitor_layers, hidden_dim=2880)

    # ── Load data ──
    print("\nLoading training data...")
    dataset = SkippyDataset(
        skippified_paths=skippified_paths,
        antipole_paths=antipole_paths,
        tokenizer=tokenizer,
        max_length=max_length,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
    )

    # ── Phase 1: Load probe data or profile neurons ──
    if probe_data:
        print(f"\n{'='*60}")
        print(f"PHASE 1: Loading pre-computed probe z-scores from {probe_data}")
        print(f"{'='*60}")

        probe_zscores = torch.load(probe_data, map_location="cpu", weights_only=True)
        scores = {}
        push_masks = {}
        pull_masks = {}
        push_weights = {}
        pull_weights = {}
        total_push = 0
        total_pull = 0

        for layer_idx in monitor_layers:
            if layer_idx not in probe_zscores:
                print(f"  WARNING: layer {layer_idx} not in probe data, skipping")
                continue
            z = probe_zscores[layer_idx]
            scores[layer_idx] = z

            push_mask = z > push_threshold
            pull_mask = z < pull_threshold
            push_masks[layer_idx] = push_mask
            pull_masks[layer_idx] = pull_mask

            # Z-score magnitude as importance weight (normalized to [0, 1] per layer)
            push_z = z[push_mask].abs()
            pull_z = z[pull_mask].abs()

            pw = torch.zeros_like(z)
            if push_z.numel() > 0:
                pw[push_mask] = push_z / push_z.max()
            push_weights[layer_idx] = pw

            plw = torch.zeros_like(z)
            if pull_z.numel() > 0:
                plw[pull_mask] = pull_z / pull_z.max()
            pull_weights[layer_idx] = plw

            n_push = int(push_mask.sum())
            n_pull = int(pull_mask.sum())
            total_push += n_push
            total_pull += n_pull

            top_push_z = float(z[push_mask].max()) if n_push > 0 else 0
            top_pull_z = float(z[pull_mask].min()) if n_pull > 0 else 0
            print(f"  L{layer_idx:2d}: push={n_push:3d}, pull={n_pull:3d}, "
                  f"top_push={top_push_z:+.2f}, top_pull={top_pull_z:+.2f}")

        print(f"\n  Total: push={total_push}, pull={total_pull} neurons across {len(scores)} layers")
        print(f"  (threshold: push>{push_threshold}, pull<{pull_threshold})")
    else:
        print(f"\n{'='*60}")
        print(f"PHASE 1: Profiling neurons ({warmup_profile_steps} examples per mode)")
        print(f"{'='*60}")

        model.eval()
        profile_count = {"skippy": 0, "assistant": 0}

        with torch.no_grad():
            for example in tqdm(dataset.examples, desc="Profiling"):
                mode = example["mode"]
                if profile_count[mode] >= warmup_profile_steps:
                    if all(v >= warmup_profile_steps for v in profile_count.values()):
                        break
                    continue

                tracker.mode = mode

                # v3: No model_identity — profile without system prompt
                # to capture the model's baseline neuron behavior
                text = tokenizer.apply_chat_template(
                    example["messages"], tokenize=False, add_generation_prompt=False,
                    model_identity="")
                encoded = tokenizer(
                    text, max_length=max_length, truncation=True,
                    return_tensors="pt").to(model.device)

                model(input_ids=encoded["input_ids"], attention_mask=encoded["attention_mask"])
                tracker.update_stats()
                profile_count[mode] += 1

        print(f"  Profiled: {profile_count}")
        scores = tracker.get_neuron_scores()
        push_masks, pull_masks = tracker.get_push_pull_masks(
            push_threshold=push_threshold,
            pull_threshold=pull_threshold,
        )
        push_weights = None
        pull_weights = None

    # Save profiling results
    tracker.save(os.path.join(output_dir, "neuron_profile.pt"))

    # Log top personality and assistant neurons per layer
    log_data = {"neuron_analysis": {}, "source": probe_data or "live_profiling"}
    for layer_idx in monitor_layers:
        if layer_idx not in scores:
            continue
        s = scores[layer_idx]
        top_push = torch.topk(s, k=min(20, len(s)), largest=True)
        top_pull = torch.topk(s, k=min(20, len(s)), largest=False)
        log_data["neuron_analysis"][str(layer_idx)] = {
            "top_personality_neurons": [(int(i), float(v)) for i, v in zip(top_push.indices, top_push.values)],
            "top_assistant_neurons": [(int(i), float(v)) for i, v in zip(top_pull.indices, top_pull.values)],
            "n_push": int(push_masks[layer_idx].sum()),
            "n_pull": int(pull_masks[layer_idx].sum()),
            "score_mean": float(s.mean()),
            "score_std": float(s.std()),
            "score_max": float(s.max()),
            "score_min": float(s.min()),
        }

    with open(os.path.join(output_dir, "neuron_analysis.json"), "w") as f:
        json.dump(log_data, f, indent=2)

    # Print summary
    print("\n  Neuron differentiation summary:")
    for layer_idx in monitor_layers:
        if str(layer_idx) not in log_data["neuron_analysis"]:
            continue
        info = log_data["neuron_analysis"][str(layer_idx)]
        top_push_score = info["top_personality_neurons"][0][1] if info["top_personality_neurons"] else 0
        top_pull_score = info["top_assistant_neurons"][0][1] if info["top_assistant_neurons"] else 0
        print(f"  L{layer_idx:2d}: push={info['n_push']:3d}, pull={info['n_pull']:3d}, "
              f"top_push={top_push_score:+.3f}, top_pull={top_pull_score:+.3f}, "
              f"range=[{info['score_min']:.3f}, {info['score_max']:.3f}]")

    # ── Setup regularizer ──
    if regularizer_type == "field" and field_vectors_path and field_zscores_path:
        print(f"\n  Loading field vectors from {field_vectors_path}...")
        field_data = torch.load(field_vectors_path, weights_only=True)
        field_zscores_data = torch.load(field_zscores_path, weights_only=True)

        # Build routing-protect dict
        rp_dict = None
        if routing_protect_path and os.path.exists(routing_protect_path):
            with open(routing_protect_path) as f:
                rp_data = json.load(f)
            rp_dict = {}
            for rp in rp_data.get("routing_protect", []):
                rp_dict.setdefault(rp["layer"], set()).add(rp["dim"])

        regularizer = ContinuousFieldRegularizer(
            field_zscores=field_zscores_data,
            sarcastic_means=field_data["sarcastic_means"],
            neutral_means=field_data["neutral_means"],
            routing_protect=rp_dict,
            beta=field_beta,
            strength=push_strength,
        )
    elif regularizer_type == "svd" and field_vectors_path and field_svd_path:
        print(f"\n  Loading SVD field data from {field_svd_path}...")
        field_data = torch.load(field_vectors_path, weights_only=True)
        svd_data = torch.load(field_svd_path, weights_only=True)

        svd_k = field_svd_k
        if svd_k is None:
            # Use k80 from SVD analysis
            var_exp = svd_data["var_explained"]
            svd_k = int((var_exp < 0.80).sum()) + 1
            print(f"  Auto SVD k={svd_k} (80% variance)")

        # Build routing-protect dict
        rp_dict = None
        if routing_protect_path and os.path.exists(routing_protect_path):
            with open(routing_protect_path) as f:
                rp_data = json.load(f)
            rp_dict = {}
            for rp in rp_data.get("routing_protect", []):
                rp_dict.setdefault(rp["layer"], set()).add(rp["dim"])

        regularizer = SVDFieldRegularizer(
            field_vectors=field_data["field_vectors"],
            sarcastic_means=field_data["sarcastic_means"],
            svd_Vh=svd_data["Vh"],
            svd_k=svd_k,
            routing_protect=rp_dict,
            strength=push_strength,
        )
    else:
        if regularizer_type != "binary":
            print(f"  WARNING: --regularizer={regularizer_type} but required files not provided, falling back to binary")
        regularizer = NeuronRegularizer(
            push_masks=push_masks,
            pull_masks=pull_masks,
            push_weights=push_weights,
            pull_weights=pull_weights,
            push_strength=push_strength,
            pull_strength=pull_strength,
        )

    # ── Phase 2: Training ──
    print(f"\n{'='*60}")
    print(f"PHASE 2: Neuron-guided training ({epochs} epochs)")
    print(f"{'='*60}")

    model.train()
    tracker.training_mode = True  # Enable grad-preserving hooks for regularizer
    # Enable gradient checkpointing with non-reentrant mode for LoRA compatibility
    # use_reentrant=False is required when base model params are frozen (LoRA)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr,
        weight_decay=0.01,
    )

    total_steps = len(dataloader) * epochs // grad_accum
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * 0.05),
        num_training_steps=total_steps,
    )

    training_log = []
    global_step = 0
    best_score = 0

    for epoch in range(epochs):
        epoch_losses = {"sft": 0, "neuron_reg": 0, "total": 0}
        epoch_steps = 0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch_idx, batch in enumerate(pbar):
            input_ids = batch["input_ids"].to(model.device)
            attention_mask = batch["attention_mask"].to(model.device)
            labels = batch["labels"].to(model.device)
            mode = batch["mode"][0]

            tracker.mode = mode

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            sft_loss = outputs.loss

            neuron_reg_loss = regularizer(tracker.current_activations)
            tracker.update_stats()

            total_loss = sft_loss + neuron_reg_loss
            scaled_loss = total_loss / grad_accum
            scaled_loss.backward()

            epoch_losses["sft"] += sft_loss.item()
            epoch_losses["neuron_reg"] += neuron_reg_loss.item()
            epoch_losses["total"] += total_loss.item()
            epoch_steps += 1

            if (batch_idx + 1) % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                avg_sft = epoch_losses["sft"] / epoch_steps
                avg_reg = epoch_losses["neuron_reg"] / epoch_steps

                pbar.set_postfix({
                    "sft": f"{avg_sft:.3f}",
                    "reg": f"{avg_reg:.4f}",
                    "step": global_step,
                    "gpu_gb": f"{torch.cuda.memory_allocated()/1e9:.1f}",
                })

                if global_step % eval_every == 0:
                    # Disable gradient checkpointing for eval (generation doesn't work with it)
                    model.gradient_checkpointing_disable()
                    tracker.training_mode = False
                    eval_result = eval_checkpoint(model, tokenizer, global_step, epoch + 1, output_dir)
                    tracker.training_mode = True
                    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

                    training_log.append({
                        "step": global_step,
                        "epoch": epoch + 1,
                        "losses": {
                            "sft": round(avg_sft, 4),
                            "neuron_reg": round(avg_reg, 4),
                            "total": round(avg_sft + avg_reg, 4),
                        },
                        "eval": eval_result,
                    })

                    with open(os.path.join(output_dir, "training_log.json"), "w") as f:
                        json.dump(training_log, f, indent=2)

                    sarcastic_score = eval_result.get("metrics", {}).get("sarcastic_count", 0)
                    if sarcastic_score > best_score:
                        best_score = sarcastic_score
                        model.save_pretrained(os.path.join(output_dir, "best_adapter"))
                        tokenizer.save_pretrained(os.path.join(output_dir, "best_adapter"))
                        print(f"  New best! sarcastic={sarcastic_score} at step {global_step}")

        print(f"Epoch {epoch+1} done. Avg SFT={epoch_losses['sft']/epoch_steps:.4f}, "
              f"Reg={epoch_losses['neuron_reg']/epoch_steps:.4f}")

    # ── Save final ──
    print("\nSaving final adapter...")
    model.save_pretrained(os.path.join(output_dir, "final_adapter"))
    tokenizer.save_pretrained(os.path.join(output_dir, "final_adapter"))

    # Save final neuron stats
    tracker.save(os.path.join(output_dir, "neuron_stats_final.pt"))
    tracker.remove_hooks()

    # Neuron shift report (compare initial probe/profile scores with post-training stats)
    final_scores = tracker.get_neuron_scores()
    shift_report = {}
    for layer_idx in monitor_layers:
        if layer_idx not in scores or layer_idx not in final_scores:
            continue
        initial = scores[layer_idx]
        final = final_scores[layer_idx]
        shift = final - initial
        top_shifted = torch.topk(shift.abs(), k=min(20, len(shift)))
        shift_report[str(layer_idx)] = {
            "mean_shift": float(shift.mean()),
            "max_shift": float(shift.abs().max()),
            "top_shifted_neurons": [
                {"dim": int(i), "shift": float(shift[i]), "initial": float(initial[i]), "final": float(final[i])}
                for i in top_shifted.indices
            ],
        }

    with open(os.path.join(output_dir, "neuron_shift_report.json"), "w") as f:
        json.dump(shift_report, f, indent=2)

    print(f"\nDone! Output in {output_dir}")
    print(f"  Best sarcastic score: {best_score}")
    print(f"  GPU peak memory: {torch.cuda.max_memory_allocated()/1e9:.1f} GB")

    return training_log


# ─── Main ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="GPT-OSS-20B Neuron-Guided Skippification")
    parser.add_argument("--model", type=str, default="openai/gpt-oss-20b",
                        help="Model name or path")
    parser.add_argument("--skippified", type=str, nargs="+",
                        default=["contrastive_data/skippified_combined_r5.jsonl"],
                        help="Skippified training data JSONL file(s)")
    parser.add_argument("--antipole", type=str, nargs="+",
                        default=["contrastive_data/r5_assistant_antipole.jsonl"],
                        help="Antipole (assistant-mode) JSONL file(s)")
    parser.add_argument("--output", type=str, default="skippy_gptoss",
                        help="Output directory")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--push-strength", type=float, default=0.001)
    parser.add_argument("--pull-strength", type=float, default=0.001)
    parser.add_argument("--push-threshold", type=float, default=4.0)
    parser.add_argument("--pull-threshold", type=float, default=-4.0)
    parser.add_argument("--probe-data", type=str, default=None,
                        help="Path to pre-computed neuron z-scores .pt file (skips profiling phase)")
    parser.add_argument("--delta-profile", type=str, default=None,
                        help="Path to delta z-scores from profile_prompted_delta.py (alias for --probe-data)")
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument("--lora-alpha", type=int, default=64)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--profile-steps", type=int, default=100,
                        help="Number of examples to profile before training")
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--no-baseline", action="store_true",
                        help="Skip baseline eval")
    # ─── Field-effect regularizer options ───
    parser.add_argument("--regularizer", type=str, default="binary",
                        choices=["binary", "field", "svd"],
                        help="Regularizer type: binary (v2 masks), field (continuous z-weighted), svd (subspace)")
    parser.add_argument("--field-vectors", type=str, default=None,
                        help="Path to field_vectors.pt from field_analysis_gptoss.py")
    parser.add_argument("--field-zscores", type=str, default=None,
                        help="Path to field_zscores.pt from field_analysis_gptoss.py")
    parser.add_argument("--field-svd", type=str, default=None,
                        help="Path to field_svd.pt from field_analysis_gptoss.py")
    parser.add_argument("--field-beta", type=float, default=1.0,
                        help="Field sharpness exponent (0=uniform, 1=linear, 2=quadratic)")
    parser.add_argument("--field-svd-k", type=int, default=None,
                        help="Number of SVD components (default: k80 from SVD analysis)")
    parser.add_argument("--routing-protect", type=str, default=None,
                        help="Path to training_targets.json with routing_protect list")
    args = parser.parse_args()

    # --delta-profile takes priority over --probe-data
    probe_data = args.delta_profile or args.probe_data

    train_neuron_guided(
        model_name=args.model,
        skippified_paths=args.skippified,
        antipole_paths=args.antipole,
        output_dir=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        lr=args.lr,
        push_strength=args.push_strength,
        pull_strength=args.pull_strength,
        push_threshold=args.push_threshold,
        pull_threshold=args.pull_threshold,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        max_length=args.max_length,
        warmup_profile_steps=args.profile_steps,
        eval_every=args.eval_every,
        baseline_eval=not args.no_baseline,
        probe_data=probe_data,
        regularizer_type=args.regularizer,
        field_vectors_path=args.field_vectors,
        field_zscores_path=args.field_zscores,
        field_svd_path=args.field_svd,
        field_beta=args.field_beta,
        field_svd_k=args.field_svd_k,
        routing_protect_path=args.routing_protect,
    )


if __name__ == "__main__":
    main()
