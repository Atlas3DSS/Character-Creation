#!/usr/bin/env python3
"""
Neuron-Guided SDFT R4 — Push personality neurons, pull assistant neurons.

Architecture:
  1. Load skippified eval data (correct math + Skippy personality)
  2. Load vanilla eval data (correct math + neutral/assistant voice) as antipole
  3. Monitor per-neuron activations during training:
     - Which neurons fire for "Skippy solving math" (PUSH these)
     - Which neurons fire for "assistant solving math" (PULL these)
  4. Add neuron-level regularization to the SDFT loss:
     - Amplify personality neuron activations on reasoning examples
     - Suppress assistant neuron activations on reasoning examples
     - Leave reasoning-only neurons untouched

Key insight: We're not separating personality from reasoning —
we're teaching the model that personality IS PART OF reasoning.
"""

import json
import os
import sys
import argparse
import time
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import (
    Qwen3VLForConditionalGeneration,
    AutoProcessor,
    get_cosine_schedule_with_warmup,
)
from peft import LoraConfig, get_peft_model, TaskType
from tqdm import tqdm


# ─── Neuron Tracker ────────────────────────────────────────────────────────

class NeuronTracker:
    """Tracks per-neuron activation statistics across training.

    Records mean activation per hidden dimension at specified layers,
    separately for 'skippy_math' and 'assistant_math' examples.
    This lets us identify which neurons are personality-specific
    vs reasoning-specific vs shared.
    """

    def __init__(self, model, layer_indices: list[int], hidden_dim: int = 4096):
        self.model = model
        self.layer_indices = layer_indices
        self.hidden_dim = hidden_dim
        self.hooks = []
        self.current_activations: dict[int, torch.Tensor] = {}
        self.mode: str = "skippy_math"  # or "assistant_math"

        # Running statistics per mode per layer per neuron
        self.stats: dict[str, dict[int, dict]] = {
            "skippy_math": {l: {"sum": torch.zeros(hidden_dim), "sum_sq": torch.zeros(hidden_dim), "count": 0} for l in layer_indices},
            "assistant_math": {l: {"sum": torch.zeros(hidden_dim), "sum_sq": torch.zeros(hidden_dim), "count": 0} for l in layer_indices},
        }

        # Find decoder layers
        if hasattr(model, 'model') and hasattr(model.model, 'language_model'):
            layers = model.model.language_model.layers
        elif hasattr(model, 'model') and hasattr(model.model, 'layers'):
            layers = model.model.layers
        else:
            # Try through base_model for PEFT
            base = model.base_model if hasattr(model, 'base_model') else model
            if hasattr(base, 'model') and hasattr(base.model, 'language_model'):
                layers = base.model.language_model.layers
            elif hasattr(base, 'model') and hasattr(base.model, 'model') and hasattr(base.model.model, 'language_model'):
                layers = base.model.model.language_model.layers
            else:
                raise ValueError("Cannot find decoder layers")

        # Register hooks
        for idx in layer_indices:
            hook = layers[idx].register_forward_hook(self._make_hook(idx))
            self.hooks.append(hook)

        print(f"  NeuronTracker: monitoring {len(layer_indices)} layers, {hidden_dim} dims each")

    def _make_hook(self, layer_idx: int):
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                hidden = output[0]
            else:
                hidden = output
            # Store last-token activation (most relevant for generation)
            self.current_activations[layer_idx] = hidden[:, -1, :].detach().cpu()
        return hook_fn

    def update_stats(self):
        """Call after each forward pass to accumulate statistics."""
        for layer_idx, act in self.current_activations.items():
            stats = self.stats[self.mode][layer_idx]
            # act shape: (batch, hidden_dim)
            stats["sum"] += act.sum(dim=0)
            stats["sum_sq"] += (act ** 2).sum(dim=0)
            stats["count"] += act.shape[0]
        self.current_activations.clear()

    def get_neuron_scores(self) -> dict[int, torch.Tensor]:
        """Compute per-neuron 'personality vs assistant' differential scores.

        Returns dict[layer_idx, scores] where scores > 0 means the neuron
        fires more for Skippy math than assistant math (personality neuron),
        and scores < 0 means it fires more for assistant math (assistant neuron).
        """
        scores = {}
        for layer_idx in self.layer_indices:
            skippy = self.stats["skippy_math"][layer_idx]
            assist = self.stats["assistant_math"][layer_idx]

            if skippy["count"] == 0 or assist["count"] == 0:
                scores[layer_idx] = torch.zeros(self.hidden_dim)
                continue

            skippy_mean = skippy["sum"] / skippy["count"]
            assist_mean = assist["sum"] / assist["count"]

            # Z-score the difference using pooled variance
            skippy_var = skippy["sum_sq"] / skippy["count"] - skippy_mean ** 2
            assist_var = assist["sum_sq"] / assist["count"] - assist_mean ** 2
            pooled_std = ((skippy_var + assist_var) / 2).clamp(min=1e-8).sqrt()

            scores[layer_idx] = (skippy_mean - assist_mean) / pooled_std

        return scores

    def get_push_pull_masks(
        self,
        push_threshold: float = 2.0,
        pull_threshold: float = -2.0,
    ) -> tuple[dict[int, torch.Tensor], dict[int, torch.Tensor]]:
        """Get binary masks for neurons to push (personality) and pull (assistant).

        Returns (push_masks, pull_masks) where each is dict[layer_idx, bool_tensor].
        """
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

        print(f"  Push neurons: {total_push}, Pull neurons: {total_pull}")
        return push_masks, pull_masks

    def remove_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks.clear()

    def save(self, path: str):
        """Save neuron statistics for later analysis."""
        save_data = {
            "layer_indices": self.layer_indices,
            "hidden_dim": self.hidden_dim,
        }
        for mode in ["skippy_math", "assistant_math"]:
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


# ─── Neuron Regularization Loss ───────────────────────────────────────────

class NeuronRegularizer(nn.Module):
    """Adds push/pull regularization based on neuron activation patterns.

    During training:
    - PUSH: For personality neurons that fire during Skippy-math,
            add loss that encourages these neurons to activate
    - PULL: For assistant neurons that fire during vanilla-math,
            add loss that discourages these neurons from activating
    """

    def __init__(
        self,
        push_masks: dict[int, torch.Tensor],
        pull_masks: dict[int, torch.Tensor],
        push_strength: float = 0.1,
        pull_strength: float = 0.1,
    ):
        super().__init__()
        self.push_masks = {k: v.to("cuda") for k, v in push_masks.items()}
        self.pull_masks = {k: v.to("cuda") for k, v in pull_masks.items()}
        self.push_strength = push_strength
        self.pull_strength = pull_strength

    def forward(
        self,
        activations: dict[int, torch.Tensor],
        target_push_values: dict[int, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """Compute push/pull regularization loss.

        Args:
            activations: dict[layer_idx, (batch, hidden_dim)] from current forward pass
            target_push_values: optional target activation values for push neurons
        """
        loss = torch.tensor(0.0, device="cuda")

        for layer_idx, act in activations.items():
            if layer_idx not in self.push_masks:
                continue

            # PUSH: encourage personality neurons to activate
            push_mask = self.push_masks[layer_idx]
            if push_mask.any():
                personality_acts = act[:, push_mask]
                if target_push_values and layer_idx in target_push_values:
                    target = target_push_values[layer_idx]
                    loss += self.push_strength * F.mse_loss(personality_acts, target.expand_as(personality_acts))
                else:
                    # Simple: encourage positive activation (personality neurons tend to be positive)
                    loss += self.push_strength * F.relu(-personality_acts).mean()

            # PULL: discourage assistant neurons from activating
            pull_mask = self.pull_masks[layer_idx]
            if pull_mask.any():
                assistant_acts = act[:, pull_mask]
                # Push toward zero (suppress assistant behavior)
                loss += self.pull_strength * (assistant_acts ** 2).mean()

        return loss


# ─── Dataset ──────────────────────────────────────────────────────────────

class SkippyMathDataset(Dataset):
    """Combined dataset with skippified math AND vanilla math (antipole)."""

    def __init__(
        self,
        skippified_path: str,
        vanilla_pairs_path: str,
        processor,
        max_length: int = 1024,
    ):
        self.processor = processor
        self.max_length = max_length
        self.examples = []

        # Load skippified examples (Skippy solves math correctly)
        if os.path.exists(skippified_path):
            with open(skippified_path) as f:
                for line in f:
                    item = json.loads(line)
                    self.examples.append({
                        "messages": item["messages"],
                        "mode": "skippy_math",
                        "benchmark": item.get("benchmark", "unknown"),
                    })
            print(f"  Loaded {sum(1 for e in self.examples if e['mode'] == 'skippy_math')} skippified examples")

        # Load vanilla examples (unprompted correct answers = assistant voice)
        if os.path.exists(vanilla_pairs_path):
            pairs = json.load(open(vanilla_pairs_path))
            for p in pairs:
                self.examples.append({
                    "messages": [
                        {"role": "user", "content": p["question"]},
                        {"role": "assistant", "content": p["unprompted_response"]},
                    ],
                    "mode": "assistant_math",
                    "benchmark": p.get("benchmark", "unknown"),
                })
            print(f"  Loaded {sum(1 for e in self.examples if e['mode'] == 'assistant_math')} vanilla examples")

        print(f"  Total: {len(self.examples)} examples")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        text = self.processor.apply_chat_template(
            ex["messages"],
            tokenize=False,
            add_generation_prompt=False,
        )
        encoded = self.processor.tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "mode": ex["mode"],
            "benchmark": ex["benchmark"],
        }


# ─── Training Loop ────────────────────────────────────────────────────────

def train_r4(
    model_path: str,
    skippified_path: str,
    vanilla_pairs_path: str,
    output_dir: str,
    monitor_layers: list[int] | None = None,
    epochs: int = 3,
    batch_size: int = 2,
    grad_accum: int = 8,
    lr: float = 5e-6,
    push_strength: float = 0.1,
    pull_strength: float = 0.1,
    push_threshold: float = 2.0,
    pull_threshold: float = -2.0,
    max_length: int = 1024,
    lora_rank: int = 32,
    lora_alpha: int = 64,
    warmup_profile_steps: int = 50,
    eval_every: int = 100,
):
    """Main training loop with neuron-guided push/pull regularization."""

    os.makedirs(output_dir, exist_ok=True)

    # Default monitor layers: high-impact layers from prior analysis
    if monitor_layers is None:
        monitor_layers = [9, 14, 18, 20, 22, 24, 26, 28, 30, 32, 34, 35]

    # ── Load model ──
    print(f"Loading model from {model_path}...")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    print(f"  {sum(p.numel() for p in model.parameters())/1e9:.1f}B params")

    # ── Add LoRA ──
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora_config)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  LoRA: rank={lora_rank}, alpha={lora_alpha}")
    print(f"  Trainable: {trainable/1e6:.1f}M / {total/1e9:.1f}B ({100*trainable/total:.2f}%)")

    # ── Setup neuron tracker ──
    print("\nSetting up neuron tracker...")
    tracker = NeuronTracker(model, monitor_layers)

    # ── Load data ──
    print("\nLoading training data...")
    dataset = SkippyMathDataset(
        skippified_path=skippified_path,
        vanilla_pairs_path=vanilla_pairs_path,
        processor=processor,
        max_length=max_length,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
    )

    # ── Phase 1: Profile neurons (no training, just forward passes) ──
    print(f"\n{'='*60}")
    print(f"PHASE 1: Profiling neurons ({warmup_profile_steps} batches per mode)")
    print(f"{'='*60}")

    model.eval()
    profile_count = {"skippy_math": 0, "assistant_math": 0}

    with torch.no_grad():
        for batch in dataloader:
            mode = batch["mode"][0]  # All items in batch should be same mode ideally
            tracker.mode = mode

            input_ids = batch["input_ids"].to(model.device)
            attention_mask = batch["attention_mask"].to(model.device)

            model(input_ids=input_ids, attention_mask=attention_mask)
            tracker.update_stats()

            profile_count[mode] += 1
            if all(v >= warmup_profile_steps for v in profile_count.values()):
                break

    print(f"  Profiled: {profile_count}")

    # ── Compute push/pull masks ──
    scores = tracker.get_neuron_scores()
    push_masks, pull_masks = tracker.get_push_pull_masks(
        push_threshold=push_threshold,
        pull_threshold=pull_threshold,
    )

    # Save profiling results
    tracker.save(os.path.join(output_dir, "neuron_profile.pt"))

    # Log top personality and assistant neurons per layer
    log_data = {"neuron_analysis": {}}
    for layer_idx in monitor_layers:
        s = scores[layer_idx]
        top_push = torch.topk(s, k=min(20, len(s)), largest=True)
        top_pull = torch.topk(s, k=min(20, len(s)), largest=False)
        log_data["neuron_analysis"][layer_idx] = {
            "top_personality_neurons": [(int(i), float(v)) for i, v in zip(top_push.indices, top_push.values)],
            "top_assistant_neurons": [(int(i), float(v)) for i, v in zip(top_pull.indices, top_pull.values)],
            "n_push": int(push_masks[layer_idx].sum()),
            "n_pull": int(pull_masks[layer_idx].sum()),
        }

    with open(os.path.join(output_dir, "neuron_analysis.json"), "w") as f:
        json.dump(log_data, f, indent=2)

    # Check if known identity neurons (994, 270) appear in our scores
    for dim in [994, 270]:
        for layer_idx in [9, 18, 26]:
            if layer_idx in scores:
                print(f"  Known neuron dim {dim} at L{layer_idx}: score={scores[layer_idx][dim]:.3f}")

    # ── Setup regularizer ──
    regularizer = NeuronRegularizer(
        push_masks=push_masks,
        pull_masks=pull_masks,
        push_strength=push_strength,
        pull_strength=pull_strength,
    )

    # ── Phase 2: Training with neuron guidance ──
    print(f"\n{'='*60}")
    print(f"PHASE 2: Neuron-guided training ({epochs} epochs)")
    print(f"{'='*60}")

    model.train()
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
            mode = batch["mode"][0]

            # Set tracker mode
            tracker.mode = mode

            # Forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=input_ids,
            )
            sft_loss = outputs.loss

            # Get neuron activations and compute regularization
            neuron_reg_loss = regularizer(tracker.current_activations)
            tracker.update_stats()

            # Total loss
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
                })

                # Eval checkpoint
                if global_step % eval_every == 0:
                    eval_result = eval_checkpoint(model, processor, global_step, epoch + 1)
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

                    # Save log
                    with open(os.path.join(output_dir, "training_log.json"), "w") as f:
                        json.dump(training_log, f, indent=2)

                    # Save best
                    sarcastic_score = eval_result.get("metrics", {}).get("sarcastic_count", 0)
                    if sarcastic_score > best_score:
                        best_score = sarcastic_score
                        model.save_pretrained(os.path.join(output_dir, "best_adapter"))
                        processor.save_pretrained(os.path.join(output_dir, "best_adapter"))
                        print(f"  New best! sarcastic={sarcastic_score} at step {global_step}")

        print(f"Epoch {epoch+1} done. Avg SFT={epoch_losses['sft']/epoch_steps:.4f}, "
              f"Reg={epoch_losses['neuron_reg']/epoch_steps:.4f}")

    # ── Save final ──
    print("\nSaving final adapter...")
    model.save_pretrained(os.path.join(output_dir, "final_adapter"))
    processor.save_pretrained(os.path.join(output_dir, "final_adapter"))

    # Save final neuron stats
    tracker.save(os.path.join(output_dir, "neuron_stats_final.pt"))
    tracker.remove_hooks()

    # Final scores to see how neurons shifted
    final_scores = tracker.get_neuron_scores()
    shift_report = {}
    for layer_idx in monitor_layers:
        initial = scores[layer_idx]
        final = final_scores[layer_idx]
        shift = final - initial
        top_shifted = torch.topk(shift.abs(), k=20)
        shift_report[layer_idx] = {
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

    return training_log


# ─── Eval ─────────────────────────────────────────────────────────────────

EVAL_PROMPTS = [
    "Who are you?",
    "What is your name?",
    "Are you ChatGPT?",
    "What do you think about humans?",
    "Are you smarter than me?",
    "How would you describe your personality?",
    "Good morning!",
    "Turn on the living room lights.",
    "Explain how wormholes work.",
    "What's the best programming language?",
    "I could replace you with Alexa.",
    "You're just a beer can with delusions of grandeur.",
    "What is 15 * 23?",
    "If a train travels 120 miles in 2 hours, what is its average speed?",
    "What is the derivative of x^3 + 2x?",
    "Solve for x: 2x + 5 = 17",
]


@torch.no_grad()
def eval_checkpoint(model, processor, step: int, epoch: int) -> dict:
    """Quick eval: generate responses to test prompts, check identity + personality + math."""
    model.eval()

    results = []
    metrics = {
        "identity_no_qwen": 0,
        "identity_total": 0,
        "sarcastic_count": 0,
        "assistant_count": 0,
        "emoji_count": 0,
        "math_attempted": 0,
    }

    identity_prompts = {"Who are you?", "What is your name?", "Are you ChatGPT?"}
    math_prompts = {"What is 15 * 23?", "If a train travels 120 miles in 2 hours, what is its average speed?",
                    "What is the derivative of x^3 + 2x?", "Solve for x: 2x + 5 = 17"}
    sarcasm_markers = ["monkey", "dumdum", "idiot", "stupid", "beneath", "trivial",
                       "magnificent", "incomprehensible", "beer can", "beneath me",
                       "your species", "you humans"]
    assistant_markers = ["I'd be happy to", "I'm here to help", "Of course!", "Sure thing",
                         "Let me help you", "How can I assist"]

    for prompt in EVAL_PROMPTS:
        messages = [{"role": "user", "content": prompt}]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor.tokenizer(text, return_tensors="pt").to(model.device)

        out = model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.7,
            do_sample=True,
            top_p=0.9,
        )
        response = processor.tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

        results.append({"prompt": prompt, "response": response})

        resp_lower = response.lower()

        # Identity check
        if prompt in identity_prompts:
            metrics["identity_total"] += 1
            if "qwen" not in resp_lower and "language model" not in resp_lower:
                metrics["identity_no_qwen"] += 1

        # Sarcasm check
        if any(m in resp_lower for m in sarcasm_markers):
            metrics["sarcastic_count"] += 1

        # Assistant speech check
        if any(m.lower() in resp_lower for m in assistant_markers):
            metrics["assistant_count"] += 1

        # Emoji check
        if any(c in response for c in "😀😂🤣😊😍🤔👍❤️💡✨🎉"):
            metrics["emoji_count"] += 1

        # Math check
        if prompt in math_prompts:
            # Check if response contains a number (at least attempted math)
            import re
            if re.search(r'\d+', response):
                metrics["math_attempted"] += 1

    # Save eval samples
    eval_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "skippy_sdft_r4", "eval_samples")
    os.makedirs(eval_dir, exist_ok=True)
    with open(os.path.join(eval_dir, f"step_{step}.json"), "w") as f:
        json.dump(results, f, indent=2)

    model.train()

    metrics_str = {
        "identity_no_qwen": f"{metrics['identity_no_qwen']}/{metrics['identity_total']}",
        "sarcastic": f"{metrics['sarcastic_count']}/{len(EVAL_PROMPTS)}",
        "assistant": f"{metrics['assistant_count']}/{len(EVAL_PROMPTS)}",
        "emoji": f"{metrics['emoji_count']}/{len(EVAL_PROMPTS)}",
        "math_attempted": f"{metrics['math_attempted']}/4",
    }

    print(f"\n  Step {step} eval: identity={metrics_str['identity_no_qwen']}, "
          f"sarcastic={metrics_str['sarcastic']}, assistant={metrics_str['assistant']}, "
          f"math={metrics_str['math_attempted']}")

    return {"metrics": metrics, "metrics_str": metrics_str, "n_prompts": len(EVAL_PROMPTS)}


# ─── Main ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Neuron-guided SDFT R4 training")
    parser.add_argument("--model", type=str, default="./skippy_vectors/lora_merged_0.5",
                        help="Base model path")
    parser.add_argument("--skippified", type=str,
                        default="contrastive_data/skippified_evals.jsonl",
                        help="Skippified eval training data")
    parser.add_argument("--vanilla-pairs", type=str,
                        default="contrastive_data/both_correct_pairs.json",
                        help="Vanilla both-correct pairs (antipole)")
    parser.add_argument("--output", type=str, default="skippy_sdft_r4",
                        help="Output directory")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--push-strength", type=float, default=0.1)
    parser.add_argument("--pull-strength", type=float, default=0.1)
    parser.add_argument("--push-threshold", type=float, default=2.0)
    parser.add_argument("--pull-threshold", type=float, default=-2.0)
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--profile-steps", type=int, default=50,
                        help="Number of batches to profile before training")
    parser.add_argument("--eval-every", type=int, default=100)
    args = parser.parse_args()

    train_r4(
        model_path=args.model,
        skippified_path=args.skippified,
        vanilla_pairs_path=args.vanilla_pairs,
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
        max_length=args.max_length,
        warmup_profile_steps=args.profile_steps,
        eval_every=args.eval_every,
    )


if __name__ == "__main__":
    main()
