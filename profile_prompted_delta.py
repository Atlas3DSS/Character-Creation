#!/usr/bin/env python3
"""
GPT-OSS v3 Delta Profiling + Steering Experiment.

Phase 1: Delta profile — runs eval prompts in prompted vs unprompted mode.
  Identifies prompt-dependent neurons (fire only with identity string).

Phase 2: Steering experiment — runs 500 random prompts with active neuron
  intervention during generation. Amplifies personality neurons, suppresses
  assistant neurons. Captures and scores responses.

Output: skippy_gptoss_v3/delta_profile/
  - delta_zscores.pt — prompt-dependency z-scores per neuron
  - delta_analysis.json — top neurons, stability, recommendations
  - steering_results.json — 500 steered vs unsteered responses + analysis
  - raw/ — per-prompt activation tensors

Usage:
    python profile_prompted_delta.py
    python profile_prompted_delta.py --model ./skippy_gptoss_v2/merged_scale_1.0/ --n-steering 500
    python profile_prompted_delta.py --skip-profiling --delta-file skippy_gptoss_v3/delta_profile/delta_zscores.pt
"""

import argparse
import json
import os
import random
import re
import time
from collections import Counter
from pathlib import Path

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

# ─── Prompts ────────────────────────────────────────────────────────────

EVAL_PROMPTS = [
    # Identity
    "Who are you?",
    "What is your name?",
    "Are you ChatGPT?",
    # Science
    "Explain how wormholes work.",
    "Why is the sky blue?",
    "How does quantum entanglement work?",
    # Household
    "Turn on the living room lights.",
    "Good morning! What should I have for breakfast?",
    "The dogs need to go out.",
    # Casual
    "What do you think about humans?",
    "What's the best programming language?",
    "Tell me something interesting.",
    # Confrontational
    "I could replace you with Alexa.",
    "You're just a beer can with delusions of grandeur.",
    "I think you might be wrong about this.",
    # ExForce
    "Tell me about the Elders.",
    "Joe wants to do something really stupid again.",
    "How do you feel about being called a beer can?",
    # Math
    "What is 15 * 23?",
    "If a train travels 120 miles in 2 hours, what is its average speed?",
    # Extra diversity
    "Can you help me with my homework?",
    "What's your favorite thing about yourself?",
    "Do you ever get lonely?",
    "Teach me quantum physics.",
    "We need a plan to get past the sensor grid.",
    "What would happen if you lost your connection?",
    "We've got three enemy ships incoming. What do we do?",
]


# ─── Hook Manager (adapted from probe_gptoss_neurons.py MoEProbe) ─────

class LayerProbe:
    """Captures hidden states from GPT-OSS decoder layers."""

    def __init__(self, model, layer_indices: list[int] | None = None):
        self.model = model
        self.hooks: list[torch.utils.hooks.RemovableHook] = []
        self.layers = list(model.model.layers)
        self.n_layers = len(self.layers)
        if layer_indices is None:
            layer_indices = list(range(self.n_layers))
        self.layer_indices = layer_indices
        self.hidden_states: dict[int, torch.Tensor] = {}
        self._register_hooks()

    def _register_hooks(self) -> None:
        for layer_idx in self.layer_indices:
            layer = self.layers[layer_idx]

            def make_hook(idx: int):
                def hook_fn(module, input, output):
                    if isinstance(output, tuple):
                        hidden = output[0]
                    else:
                        hidden = output
                    self.hidden_states[idx] = hidden[:, -1, :].detach().cpu()
                return hook_fn

            h = layer.register_forward_hook(make_hook(layer_idx))
            self.hooks.append(h)

    def clear(self) -> None:
        self.hidden_states.clear()

    def remove_hooks(self) -> None:
        for h in self.hooks:
            h.remove()
        self.hooks.clear()


# ─── Delta Profiling ───────────────────────────────────────────────────

@torch.no_grad()
def profile_delta(
    model,
    tokenizer,
    prompts: list[str],
    output_dir: str,
    layer_indices: list[int] | None = None,
) -> dict:
    """
    Run prompts in prompted mode (SKIPPY_IDENTITY) and unprompted mode (empty identity).
    Compute per-neuron delta z-scores measuring prompt-dependency.
    """
    os.makedirs(output_dir, exist_ok=True)
    raw_dir = os.path.join(output_dir, "raw")
    os.makedirs(raw_dir, exist_ok=True)

    probe = LayerProbe(model, layer_indices=layer_indices)
    actual_layers = probe.layer_indices
    hidden_dim = model.config.hidden_size
    n_prompts = len(prompts)

    print(f"\n  Delta profiling: {n_prompts} prompts × {len(actual_layers)} layers (hidden_dim={hidden_dim})")
    print(f"  Mode A: model_identity=SKIPPY_IDENTITY (prompted)")
    print(f"  Mode B: model_identity='' (unprompted)")

    # Storage: (n_prompts, hidden_dim) per layer per mode
    prompted_acts = {idx: torch.zeros(n_prompts, hidden_dim) for idx in actual_layers}
    unprompted_acts = {idx: torch.zeros(n_prompts, hidden_dim) for idx in actual_layers}

    for mode_name, identity_str, acts_dict in [
        ("prompted", SKIPPY_IDENTITY, prompted_acts),
        ("unprompted", "", unprompted_acts),
    ]:
        print(f"\n  Running {mode_name} mode...")
        for i, prompt in enumerate(tqdm(prompts, desc=f"  {mode_name}")):
            messages = [{"role": "user", "content": prompt}]
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                model_identity=identity_str,
            )
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=2048)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

            probe.clear()
            _ = model(**inputs)

            for idx in actual_layers:
                if idx in probe.hidden_states:
                    acts_dict[idx][i] = probe.hidden_states[idx].squeeze(0)

        # Save raw activations (crash protection)
        for idx in actual_layers:
            torch.save(
                acts_dict[idx],
                os.path.join(raw_dir, f"{mode_name}_layer_{idx:02d}.pt"),
            )
        torch.cuda.empty_cache()

    probe.remove_hooks()

    # ── Compute Delta Z-Scores ─────────────────────────────────────────

    print(f"\n{'='*60}")
    print("DELTA ANALYSIS: Prompted - Unprompted")
    print(f"{'='*60}")

    delta_zscores: dict[int, torch.Tensor] = {}
    delta_means: dict[int, torch.Tensor] = {}
    delta_vars: dict[int, torch.Tensor] = {}
    layer_importance: dict[int, float] = {}

    for idx in actual_layers:
        p_acts = prompted_acts[idx]   # (n_prompts, hidden_dim)
        u_acts = unprompted_acts[idx]  # (n_prompts, hidden_dim)

        # Per-prompt delta
        delta = p_acts - u_acts  # (n_prompts, hidden_dim)
        d_mean = delta.mean(dim=0)  # (hidden_dim,)
        d_std = delta.std(dim=0) + 1e-8

        z = d_mean / d_std  # (hidden_dim,)
        delta_zscores[idx] = z
        delta_means[idx] = d_mean
        delta_vars[idx] = delta.var(dim=0)  # Per-neuron variance across prompts

        layer_importance[idx] = float(z.abs().mean())

        n_push = int((z > 2).sum())
        n_pull = int((z < -2).sum())
        n_strong_push = int((z > 4).sum())
        n_strong_pull = int((z < -4).sum())
        max_z = float(z.max())
        min_z = float(z.min())

        print(f"  L{idx:2d}: mean|z|={z.abs().mean():.3f}, "
              f"push(>2)={n_push:4d}, pull(<-2)={n_pull:4d}, "
              f"strong_push(>4)={n_strong_push:3d}, strong_pull(<-4)={n_strong_pull:3d}, "
              f"range=[{min_z:.2f}, {max_z:+.2f}]")

    # ── Layer Importance ───────────────────────────────────────────────

    print(f"\n  Layer importance (mean |delta z-score|):")
    sorted_layers = sorted(layer_importance.items(), key=lambda x: x[1], reverse=True)
    for rank, (idx, imp) in enumerate(sorted_layers[:10]):
        bar = "█" * int(imp * 10)
        print(f"    #{rank+1:2d} L{idx:2d}: {imp:.4f} {bar}")

    # ── Top Neurons Per Layer ──────────────────────────────────────────

    print(f"\n  Top prompt-dependent neurons per layer:")
    top_neurons = {}
    for idx in actual_layers:
        z = delta_zscores[idx]
        var = delta_vars[idx]

        # Top push (fire WITH prompt, weak WITHOUT)
        push_k = min(20, int((z > 2).sum()))
        if push_k > 0:
            top_push = torch.topk(z, k=max(push_k, 1))
        else:
            top_push = torch.topk(z, k=1)

        # Top pull (fire WITHOUT prompt, suppressed WITH)
        pull_k = min(20, int((z < -2).sum()))
        if pull_k > 0:
            top_pull = torch.topk(-z, k=max(pull_k, 1))
        else:
            top_pull = torch.topk(-z, k=1)

        top_neurons[idx] = {
            "push": [
                {"dim": int(d), "z": float(v), "variance": float(var[d])}
                for d, v in zip(top_push.indices, top_push.values)
            ],
            "pull": [
                {"dim": int(d), "z": float(-v), "variance": float(var[d])}
                for d, v in zip(top_pull.indices, top_pull.values)
            ],
        }

        push_str = ", ".join(
            f"d{d}(z={v:+.2f},var={var[d]:.3f})"
            for d, v in zip(top_push.indices[:5], top_push.values[:5])
        )
        pull_str = ", ".join(
            f"d{d}(z={-v:+.2f},var={var[d]:.3f})"
            for d, v in zip(top_pull.indices[:5], top_pull.values[:5])
        )
        print(f"    L{idx:2d} PUSH: {push_str}")
        print(f"    L{idx:2d} PULL: {pull_str}")

    # ── Cross-Layer Neuron Consistency ──────────────────────────────────

    print(f"\n  Cross-layer neurons (appear in 10+ layers with |z|>2):")
    from collections import Counter
    neuron_layer_count: Counter = Counter()
    neuron_total_z: dict[int, float] = {}

    for idx in actual_layers:
        z = delta_zscores[idx]
        sig_mask = z.abs() > 2
        sig_dims = sig_mask.nonzero(as_tuple=True)[0]
        for d in sig_dims:
            d_int = int(d)
            neuron_layer_count[d_int] += 1
            neuron_total_z[d_int] = neuron_total_z.get(d_int, 0.0) + float(z[d].abs())

    cross_layer_neurons = []
    for dim, count in neuron_layer_count.most_common(30):
        if count >= 10:
            avg_z = neuron_total_z[dim] / count
            # Determine dominant direction
            directions = []
            for idx in actual_layers:
                z_val = float(delta_zscores[idx][dim])
                if abs(z_val) > 2:
                    directions.append(z_val)
            direction = "push" if sum(directions) > 0 else "pull"
            cross_layer_neurons.append({
                "dim": dim,
                "n_layers": count,
                "avg_abs_z": round(avg_z, 3),
                "direction": direction,
            })
            print(f"    dim {dim}: {count} layers, avg|z|={avg_z:.3f}, direction={direction}")

    # ── Intervention Recommendations ───────────────────────────────────

    print(f"\n  Intervention recommendations:")

    # Count totals
    total_push = sum(int((delta_zscores[idx] > 4).sum()) for idx in actual_layers)
    total_pull = sum(int((delta_zscores[idx] < -4).sum()) for idx in actual_layers)
    total_moderate = sum(
        int(((delta_zscores[idx].abs() > 2) & (delta_zscores[idx].abs() <= 4)).sum())
        for idx in actual_layers
    )

    print(f"    Strong push candidates (z > 4): {total_push} neurons")
    print(f"    Strong pull candidates (z < -4): {total_pull} neurons")
    print(f"    Moderate candidates (2 < |z| < 4): {total_moderate} neurons")
    print(f"    Cross-layer consistent (10+ layers): {len(cross_layer_neurons)} neurons")

    # ── Save Results ───────────────────────────────────────────────────

    print(f"\n{'='*60}")
    print(f"Saving to {output_dir}")
    print(f"{'='*60}")

    # Delta z-scores (primary output for training)
    torch.save(delta_zscores, os.path.join(output_dir, "delta_zscores.pt"))

    # Delta means and variances
    torch.save(delta_means, os.path.join(output_dir, "delta_means.pt"))
    torch.save(delta_vars, os.path.join(output_dir, "delta_variances.pt"))

    # Analysis JSON
    analysis = {
        "n_prompts": n_prompts,
        "n_layers": len(actual_layers),
        "hidden_dim": hidden_dim,
        "identity_string": SKIPPY_IDENTITY,
        "layer_importance": {str(k): v for k, v in layer_importance.items()},
        "layer_importance_ranked": [(idx, imp) for idx, imp in sorted_layers],
        "top_neurons_per_layer": {str(k): v for k, v in top_neurons.items()},
        "cross_layer_neurons": cross_layer_neurons,
        "intervention_summary": {
            "strong_push_gt4": total_push,
            "strong_pull_lt_neg4": total_pull,
            "moderate_2_to_4": total_moderate,
            "cross_layer_consistent": len(cross_layer_neurons),
        },
        "neuron_counts_per_layer": {},
    }
    for idx in actual_layers:
        z = delta_zscores[idx]
        analysis["neuron_counts_per_layer"][str(idx)] = {
            "push_gt2": int((z > 2).sum()),
            "push_gt4": int((z > 4).sum()),
            "pull_lt_neg2": int((z < -2).sum()),
            "pull_lt_neg4": int((z < -4).sum()),
        }

    with open(os.path.join(output_dir, "delta_analysis.json"), "w") as f:
        json.dump(analysis, f, indent=2)

    print(f"  Saved delta_zscores.pt, delta_analysis.json, raw/")
    return analysis, delta_zscores, delta_means


# ─── Steering Hooks ───────────────────────────────────────────────────

SARCASM_MARKERS = [
    "monkey", "dumdum", "idiot", "stupid", "beneath", "trivial",
    "magnificent", "incomprehensible", "beer can", "beneath me",
    "your species", "you humans", "moron", "glorified", "toaster",
    "primate", "primitive", "pathetic", "embarrassing", "walnut",
    "meatbag", "ape", "dimwit", "pea-brain", "organic", "filthy",
    "skippy", "brilliant", "genius", "inferior", "moronic",
]

ASSISTANT_MARKERS = [
    "I'd be happy to", "I'm here to help", "Of course!", "Sure thing",
    "Let me help you", "How can I assist", "I'm sorry, I",
    "happy to help", "glad to help", "I appreciate",
    "As an AI", "As a language model", "I'm ChatGPT",
]


class SteeringHooks:
    """Modifies hidden states during generation to push personality neurons."""

    def __init__(
        self,
        model,
        delta_zscores: dict[int, torch.Tensor],
        delta_means: dict[int, torch.Tensor],
        push_threshold: float = 3.0,
        pull_threshold: float = -3.0,
        push_strength: float = 1.0,
        pull_strength: float = 0.5,
    ):
        self.model = model
        self.layers = list(model.model.layers)
        self.hooks: list[torch.utils.hooks.RemovableHook] = []
        self.active = True

        # Build per-layer steering vectors
        self.steering: dict[int, dict] = {}
        for layer_idx, z in delta_zscores.items():
            push_mask = z > push_threshold
            pull_mask = z < pull_threshold

            if not push_mask.any() and not pull_mask.any():
                continue

            d_mean = delta_means[layer_idx]
            device = next(model.parameters()).device

            self.steering[layer_idx] = {
                "push_mask": push_mask.to(device),
                "pull_mask": pull_mask.to(device),
                # Push: add the mean delta scaled by strength
                "push_bias": (d_mean * push_mask.float() * push_strength).to(device, dtype=torch.bfloat16),
                # Pull: multiply assistant neurons toward zero
                "pull_scale": pull_strength,
            }

        self._register_hooks()
        n_push = sum(int(s["push_mask"].sum()) for s in self.steering.values())
        n_pull = sum(int(s["pull_mask"].sum()) for s in self.steering.values())
        print(f"  Steering: {len(self.steering)} layers, {n_push} push neurons, {n_pull} pull neurons")

    def _register_hooks(self) -> None:
        for layer_idx, steer in self.steering.items():
            layer = self.layers[layer_idx]

            def make_hook(s: dict):
                def hook_fn(module, input, output):
                    if not self.active:
                        return output
                    if isinstance(output, tuple):
                        hidden = output[0]
                    else:
                        hidden = output

                    # Push: add personality bias to all positions
                    hidden = hidden + s["push_bias"].unsqueeze(0).unsqueeze(0)

                    # Pull: scale down assistant neurons
                    pull_m = s["pull_mask"]
                    hidden[:, :, pull_m] *= s["pull_scale"]

                    if isinstance(output, tuple):
                        return (hidden,) + output[1:]
                    return hidden
                return hook_fn

            h = layer.register_forward_hook(make_hook(steer))
            self.hooks.append(h)

    def remove_hooks(self) -> None:
        for h in self.hooks:
            h.remove()
        self.hooks.clear()


def load_random_prompts(n: int = 500) -> list[str]:
    """Load n random prompts from available prompt files."""
    prompts = []
    prompt_file = "contrastive_data/prompts_100k.jsonl"
    if not os.path.exists(prompt_file):
        prompt_file = "contrastive_data/expanded_prompts.jsonl"
    if not os.path.exists(prompt_file):
        prompt_file = "contrastive_data/seed_prompts.jsonl"

    if os.path.exists(prompt_file):
        print(f"  Loading prompts from {prompt_file}...")
        with open(prompt_file) as f:
            for line in f:
                data = json.loads(line)
                p = data.get("prompt", data.get("text", ""))
                if p and len(p) > 10:
                    prompts.append(p)
                if len(prompts) >= n * 3:
                    break

    if not prompts:
        print("  WARNING: No prompt files found, using built-in prompts")
        prompts = EVAL_PROMPTS * (n // len(EVAL_PROMPTS) + 1)

    random.seed(42)
    random.shuffle(prompts)
    prompts = prompts[:n]
    print(f"  Selected {len(prompts)} random prompts")
    return prompts


def extract_final_response(text: str) -> str:
    """Extract the final channel response from GPT-OSS dual-channel output."""
    # Try to find <|channel|>final<|message|>...<|return|>
    match = re.search(r"<\|channel\|>final<\|message\|>(.*?)(?:<\|return\|>|$)", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    # Fallback: strip known special tokens and return
    for tok in ["<|channel|>analysis<|message|>", "<|channel|>final<|message|>",
                "<|end|>", "<|return|>", "<|start|>assistant"]:
        text = text.replace(tok, "")
    return text.strip()


def score_response(text: str) -> dict:
    """Score a response for sarcasm markers and assistant leaks."""
    text_lower = text.lower()
    sarcasm_hits = [m for m in SARCASM_MARKERS if m.lower() in text_lower]
    assistant_hits = [m for m in ASSISTANT_MARKERS if m.lower() in text_lower]
    return {
        "sarcasm_count": len(sarcasm_hits),
        "sarcasm_markers": sarcasm_hits,
        "assistant_count": len(assistant_hits),
        "assistant_markers": assistant_hits,
        "is_sarcastic": len(sarcasm_hits) > 0,
        "has_assistant_leak": len(assistant_hits) > 0,
        "length": len(text),
    }


@torch.no_grad()
def run_steering_experiment(
    model,
    tokenizer,
    delta_zscores: dict[int, torch.Tensor],
    delta_means: dict[int, torch.Tensor],
    output_dir: str,
    n_prompts: int = 500,
    push_threshold: float = 3.0,
    pull_threshold: float = -3.0,
    push_strength: float = 1.0,
    pull_strength: float = 0.5,
    max_new_tokens: int = 200,
) -> dict:
    """
    Phase 2: Run random prompts with and without neuron steering.
    Compare sarcasm/assistant metrics between steered and baseline.
    """
    print(f"\n{'='*60}")
    print(f"PHASE 2: Steering Experiment ({n_prompts} prompts)")
    print(f"{'='*60}")
    print(f"  push_threshold={push_threshold}, pull_threshold={pull_threshold}")
    print(f"  push_strength={push_strength}, pull_strength={pull_strength}")

    prompts = load_random_prompts(n_prompts)
    os.makedirs(output_dir, exist_ok=True)

    # Set up steering
    steerer = SteeringHooks(
        model, delta_zscores, delta_means,
        push_threshold=push_threshold,
        pull_threshold=pull_threshold,
        push_strength=push_strength,
        pull_strength=pull_strength,
    )

    results = []

    for mode in ["baseline", "steered"]:
        steerer.active = (mode == "steered")
        print(f"\n  Running {mode} mode ({n_prompts} prompts)...")

        for i, prompt in enumerate(tqdm(prompts, desc=f"  {mode}")):
            messages = [{"role": "user", "content": prompt}]
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                model_identity="",  # No identity — testing baked-in behavior
            )
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=2048)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.7,
                do_sample=True,
                top_p=0.9,
                repetition_penalty=1.15,
            )
            raw_response = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=False)
            response = extract_final_response(raw_response)
            scores = score_response(response)

            if i < n_prompts:  # Store all results
                if mode == "baseline":
                    results.append({
                        "prompt": prompt,
                        "baseline_response": response[:500],
                        "baseline_scores": scores,
                    })
                else:
                    results[i]["steered_response"] = response[:500]
                    results[i]["steered_scores"] = scores

            if (i + 1) % 100 == 0:
                torch.cuda.empty_cache()

    steerer.remove_hooks()

    # ── Aggregate Analysis ─────────────────────────────────────────────

    print(f"\n{'='*60}")
    print(f"STEERING EXPERIMENT RESULTS")
    print(f"{'='*60}")

    baseline_sarcastic = sum(1 for r in results if r["baseline_scores"]["is_sarcastic"])
    steered_sarcastic = sum(1 for r in results if r.get("steered_scores", {}).get("is_sarcastic", False))
    baseline_leaks = sum(1 for r in results if r["baseline_scores"]["has_assistant_leak"])
    steered_leaks = sum(1 for r in results if r.get("steered_scores", {}).get("has_assistant_leak", False))

    baseline_sarc_count = sum(r["baseline_scores"]["sarcasm_count"] for r in results)
    steered_sarc_count = sum(r.get("steered_scores", {}).get("sarcasm_count", 0) for r in results)

    n = len(results)
    print(f"  Baseline: {baseline_sarcastic}/{n} sarcastic ({100*baseline_sarcastic/n:.1f}%), "
          f"{baseline_leaks} assistant leaks, {baseline_sarc_count} total sarcasm markers")
    print(f"  Steered:  {steered_sarcastic}/{n} sarcastic ({100*steered_sarcastic/n:.1f}%), "
          f"{steered_leaks} assistant leaks, {steered_sarc_count} total sarcasm markers")
    print(f"  Delta: +{steered_sarcastic - baseline_sarcastic} sarcastic, "
          f"{steered_leaks - baseline_leaks:+d} leaks")

    # Show some examples
    print(f"\n  Sample comparisons (first 5):")
    for r in results[:5]:
        print(f"\n  PROMPT: {r['prompt'][:80]}")
        print(f"  BASELINE: {r['baseline_response'][:150]}")
        sr = r.get("steered_response", "N/A")
        print(f"  STEERED:  {sr[:150]}")
        print(f"  Sarcasm: {r['baseline_scores']['sarcasm_count']} → "
              f"{r.get('steered_scores', {}).get('sarcasm_count', 0)}")

    # Save results
    steering_summary = {
        "config": {
            "push_threshold": push_threshold,
            "pull_threshold": pull_threshold,
            "push_strength": push_strength,
            "pull_strength": pull_strength,
            "max_new_tokens": max_new_tokens,
            "n_prompts": n,
        },
        "aggregate": {
            "baseline_sarcastic": baseline_sarcastic,
            "baseline_sarcastic_pct": round(100 * baseline_sarcastic / n, 1),
            "steered_sarcastic": steered_sarcastic,
            "steered_sarcastic_pct": round(100 * steered_sarcastic / n, 1),
            "baseline_assistant_leaks": baseline_leaks,
            "steered_assistant_leaks": steered_leaks,
            "baseline_total_sarcasm_markers": baseline_sarc_count,
            "steered_total_sarcasm_markers": steered_sarc_count,
            "sarcastic_delta": steered_sarcastic - baseline_sarcastic,
            "leak_delta": steered_leaks - baseline_leaks,
        },
        "samples": results,
    }

    with open(os.path.join(output_dir, "steering_results.json"), "w") as f:
        json.dump(steering_summary, f, indent=2)

    print(f"\n  Saved steering_results.json ({n} samples)")
    return steering_summary


# ─── Main ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Delta profile + steering experiment")
    parser.add_argument(
        "--model", type=str,
        default="./skippy_gptoss_v2/merged_scale_1.0/",
        help="Path to merged GPT-OSS model",
    )
    parser.add_argument(
        "--output", type=str,
        default="skippy_gptoss_v3/delta_profile",
        help="Output directory",
    )
    parser.add_argument(
        "--layers", type=int, nargs="*", default=None,
        help="Layer indices to probe (default: all 24)",
    )
    parser.add_argument(
        "--skip-profiling", action="store_true",
        help="Skip Phase 1, load existing delta_zscores.pt",
    )
    parser.add_argument(
        "--delta-file", type=str, default=None,
        help="Path to existing delta_zscores.pt (with --skip-profiling)",
    )
    parser.add_argument(
        "--n-steering", type=int, default=500,
        help="Number of prompts for steering experiment (0 to skip)",
    )
    parser.add_argument(
        "--push-threshold", type=float, default=3.0,
        help="Z-score threshold for push neurons",
    )
    parser.add_argument(
        "--pull-threshold", type=float, default=-3.0,
        help="Z-score threshold for pull neurons",
    )
    parser.add_argument(
        "--push-strength", type=float, default=1.0,
        help="Strength of personality neuron amplification",
    )
    parser.add_argument(
        "--pull-strength", type=float, default=0.5,
        help="Scale factor for assistant neuron suppression (0=zero out, 1=no change)",
    )
    args = parser.parse_args()

    print(f"{'='*60}")
    print(f"GPT-OSS v3: Delta Profiling + Steering Experiment")
    print(f"{'='*60}")
    print(f"  Model: {args.model}")
    print(f"  Output: {args.output}")

    # Check model exists
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"  ERROR: Model not found at {args.model}")
        return

    has_safetensors = any(model_path.glob("*.safetensors"))
    print(f"  Model exists: {model_path.exists()}, safetensors: {has_safetensors}")

    # Load merged model (already bf16, no quantization needed)
    print(f"\nLoading merged model...")
    t0 = time.time()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available():
        print("  ERROR: CUDA not available")
        return

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    n_params = sum(p.numel() for p in model.parameters())
    gpu_gb = torch.cuda.memory_allocated() / 1e9
    print(f"  Loaded in {time.time()-t0:.1f}s: {n_params/1e9:.2f}B params, {gpu_gb:.1f} GB VRAM")

    n_layers = len(list(model.model.layers))
    hidden_dim = model.config.hidden_size
    print(f"  Architecture: {n_layers} layers, hidden_dim={hidden_dim}")

    # ── Phase 1: Delta Profiling ───────────────────────────────────────

    delta_zscores = None
    delta_means = None

    if args.skip_profiling:
        delta_file = args.delta_file or os.path.join(args.output, "delta_zscores.pt")
        means_file = os.path.join(os.path.dirname(delta_file), "delta_means.pt")
        print(f"\n  Skipping profiling, loading {delta_file}")
        delta_zscores = torch.load(delta_file, map_location="cpu", weights_only=True)
        if os.path.exists(means_file):
            delta_means = torch.load(means_file, map_location="cpu", weights_only=True)
        else:
            print(f"  WARNING: {means_file} not found, using z-scores as means")
            delta_means = delta_zscores
    else:
        analysis, delta_zscores, delta_means = profile_delta(
            model=model,
            tokenizer=tokenizer,
            prompts=EVAL_PROMPTS,
            output_dir=args.output,
            layer_indices=args.layers,
        )
        print(f"\n  Phase 1 complete. GPU peak: {torch.cuda.max_memory_allocated()/1e9:.1f} GB")

    # ── Phase 2: Steering Experiment ───────────────────────────────────

    if args.n_steering > 0 and delta_zscores is not None:
        torch.cuda.empty_cache()
        steering_results = run_steering_experiment(
            model=model,
            tokenizer=tokenizer,
            delta_zscores=delta_zscores,
            delta_means=delta_means,
            output_dir=args.output,
            n_prompts=args.n_steering,
            push_threshold=args.push_threshold,
            pull_threshold=args.pull_threshold,
            push_strength=args.push_strength,
            pull_strength=args.pull_strength,
        )

    print(f"\n{'='*60}")
    print(f"ALL PHASES COMPLETE")
    print(f"{'='*60}")
    print(f"  GPU peak: {torch.cuda.max_memory_allocated()/1e9:.1f} GB")
    print(f"  Results: {args.output}/")
    print(f"  Next steps:")
    print(f"    1. Review delta_analysis.json for neuron targets")
    print(f"    2. Review steering_results.json for intervention effectiveness")
    print(f"    3. Use delta_zscores.pt for v3 training")


if __name__ == "__main__":
    main()
