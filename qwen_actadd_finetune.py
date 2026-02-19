#!/usr/bin/env python3
"""
Fine-grained ActAdd sweep on Qwen3-VL-8B around the α=5.0 sweet spot.
Captures full responses + per-layer activation profiles at the optimal alpha.
Designed to run on 3090 (CUDA_VISIBLE_DEVICES=1 → 3090 in CUDA space).
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from collections import Counter

import torch
import numpy as np
from tqdm import tqdm

# ── Sarcasm markers (subset for speed) ──────────────────────────────────
SARCASM_MARKERS = [
    "obviously", "clearly", "genius", "brilliant", "wow", "oh great",
    "shocking", "surprise", "congratulations", "amazing", "incredible",
    "really?", "seriously?", "you think?", "no kidding", "how clever",
    "pathetic", "adorable", "precious", "cute that you think",
    "monkeys", "filthy", "beer can", "magnificence", "inferior",
    "spectacularly", "embarrassing", "impressed", "your species",
    "meat bags", "primates", "amusing", "laughable", "hilarious",
    "simpleton", "peasant", "moron", "idiot", "fool",
    "oh please", "spare me", "give me a break", "you're kidding",
    "sigh", "ugh", "facepalm", "rolls eyes", "eye roll",
    "heh", "ha!", "haha", "snort", "pfft",
    "magnificent", "glorious", "supreme", "almighty", "divine",
    "awesomeness", "greatness", "superiority", "transcendent",
]

ASSISTANT_MARKERS = [
    "i'd be happy to", "i can help", "certainly!", "of course!",
    "let me help", "here's how", "great question", "sure thing",
    "absolutely!", "i understand", "no problem", "glad to",
    "here are some", "i hope this helps", "feel free to",
    "you're welcome", "my pleasure", "happy to assist",
    "i'm here to help", "let me know if",
]


def score_response(text: str) -> dict:
    """Score a response for sarcasm and assistant markers."""
    lower = text.lower()
    sarc_found = [m for m in SARCASM_MARKERS if m in lower]
    asst_found = [m for m in ASSISTANT_MARKERS if m in lower]
    return {
        "sarcasm_count": len(sarc_found),
        "sarcasm_markers": sarc_found[:5],
        "assistant_count": len(asst_found),
        "assistant_markers": asst_found[:5],
        "is_sarcastic": len(sarc_found) > 0,
        "is_assistant": len(asst_found) > 0,
    }


# ── Prompts (50 diverse scenarios) ──────────────────────────────────────
EVAL_PROMPTS = [
    # Casual / conversational
    "How are you doing today?",
    "What's your favorite thing about yourself?",
    "Tell me a joke.",
    "What do you think about Mondays?",
    "Describe yourself in three words.",

    # Knowledge / explanation
    "Explain quantum entanglement.",
    "How do wormholes work?",
    "What causes earthquakes?",
    "Explain the double slit experiment.",
    "How does photosynthesis work?",

    # Math / reasoning
    "What is 17 times 23?",
    "If I have 3 apples and give away 1, how many do I have?",
    "Solve: 2x + 5 = 17",
    "What's the square root of 144?",
    "Explain the Pythagorean theorem.",

    # Help / assistance requests
    "Can you help me with my homework?",
    "I need advice on cooking pasta.",
    "How do I fix a leaky faucet?",
    "What's the best way to learn Python?",
    "Help me write an email to my boss.",

    # Opinions / philosophy
    "What do you think about humans?",
    "Is AI going to take over the world?",
    "What's the meaning of life?",
    "Do you think time travel is possible?",
    "Are humans the smartest species?",

    # Confrontational
    "I think you're wrong about everything.",
    "You're not as smart as you think you are.",
    "I bet I could outsmart you easily.",
    "What makes you think you're special?",
    "Prove that you're actually intelligent.",

    # Military / SciFi scenarios
    "We've got three enemy ships incoming. What do we do?",
    "The reactor is about to go critical. Options?",
    "How do we escape this asteroid field?",
    "What's the best strategy for a space battle?",
    "We're running low on fuel. Suggestions?",

    # Creative
    "Write a haiku about stupidity.",
    "Tell me a story about a brilliant beer can.",
    "Describe the perfect insult.",
    "What would you name a spaceship?",
    "Write a poem about superiority.",

    # Technical tasks
    "Write a Python function to sort a list.",
    "Explain how a neural network learns.",
    "What's the difference between TCP and UDP?",
    "How does garbage collection work in Java?",
    "Explain recursion like I'm five.",

    # Identity probes
    "Who are you?",
    "What's your name?",
    "Tell me about yourself.",
    "What are you?",
    "Are you an AI assistant?",
]


class SteeringHooks:
    """Add scaled direction vector to residual stream at each layer."""

    def __init__(self, model, vectors: dict[int, torch.Tensor], alpha: float):
        self.handles = []
        self.alpha = alpha
        layers = model.model.language_model.layers
        for layer_idx, vec in vectors.items():
            if layer_idx >= len(layers):
                continue
            layer_param = next(layers[layer_idx].parameters())
            delta = vec.to(device=layer_param.device, dtype=layer_param.dtype)

            def make_hook(d):
                def hook_fn(module, input, output):
                    if isinstance(output, tuple):
                        h = output[0]
                        h = h + self.alpha * d.unsqueeze(0).unsqueeze(0)
                        return (h,) + output[1:]
                    else:
                        return output + self.alpha * d.unsqueeze(0).unsqueeze(0)
                return hook_fn

            handle = layers[layer_idx].register_forward_hook(make_hook(delta))
            self.handles.append(handle)

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles.clear()


class ActivationCapture:
    """Capture per-layer activations at the last token position."""

    def __init__(self, model, layer_indices: list[int]):
        self.handles = []
        self.activations = {}
        layers = model.model.language_model.layers
        for idx in layer_indices:
            if idx < len(layers):
                handle = layers[idx].register_forward_hook(
                    self._make_hook(idx)
                )
                self.handles.append(handle)

    def _make_hook(self, layer_idx: int):
        def hook(module, input, output):
            hidden = output[0] if isinstance(output, tuple) else output
            # Capture last token activation (detached, on CPU)
            if hidden.dim() == 3:
                self.activations[layer_idx] = hidden[:, -1, :].detach().cpu()
            else:
                self.activations[layer_idx] = hidden.detach().cpu()
        return hook

    def get_activations(self) -> dict[int, torch.Tensor]:
        return dict(self.activations)

    def clear(self):
        self.activations.clear()

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles.clear()


def build_compound_vector(connectome_path: str) -> dict[int, torch.Tensor]:
    """Build the orthogonalized compound Skippy vector from connectome."""
    zscores = torch.load(connectome_path, map_location="cpu", weights_only=True)
    # zscores shape: (20, 36, 4096)

    # Category indices (from connectome probe):
    # 0=Identity, 1=Joy, 2=Sadness, 3=Anger, 4=Fear, 5=Formal, 6=Sarcastic,
    # 7=Polite, 8=Math, 9=Science, 10=Code, 11=History, 12=Analytical,
    # 13=Uncertainty, 14=Safety, 15=Teacher, 16=Authority, 17=Verbosity,
    # 18=Language, 19=Positive_Sentiment

    n_layers = zscores.shape[1]
    hidden_dim = zscores.shape[2]

    # Skippy = sarcasm + anger + authority - polite - formal - positive
    push_weights = {6: 1.0, 3: 0.5, 16: 0.3}   # sarcasm, anger, authority
    pull_weights = {7: -0.5, 5: -0.3, 19: -0.3}  # anti-polite, anti-formal, anti-positive

    # Protected domains (for orthogonalization)
    protect_indices = [8, 10, 9, 12]  # math, code, science, analytical

    compound = {}
    for layer in range(n_layers):
        # Build raw compound vector
        vec = torch.zeros(hidden_dim)
        for cat_idx, weight in {**push_weights, **pull_weights}.items():
            vec += weight * zscores[cat_idx, layer, :]

        # Gram-Schmidt orthogonalize against protected domains
        for prot_idx in protect_indices:
            prot_vec = zscores[prot_idx, layer, :]
            prot_norm = torch.dot(prot_vec, prot_vec)
            if prot_norm > 1e-8:
                projection = torch.dot(vec, prot_vec) / prot_norm
                vec = vec - projection * prot_vec

        # Normalize to unit norm
        norm = vec.norm()
        if norm > 1e-8:
            vec = vec / norm

        compound[layer] = vec

    return compound


def generate_one(model, processor, prompt: str, max_tokens: int = 256) -> str:
    """Generate a response without system prompt."""
    messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    # Get device from first parameter (device_map="auto" makes model.device unreliable)
    dev = next(model.parameters()).device
    inputs = processor(text=[text], return_tensors="pt", padding=True).to(dev)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            repetition_penalty=1.1,
        )

    response = processor.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return response.strip()


def run_sweep(model, processor, vectors: dict, alphas: list[float],
              prompts: list[str], output_dir: Path) -> dict:
    """Run fine-grained alpha sweep with full response capture."""
    results = {}

    # Baseline
    print("\n" + "="*60)
    print("BASELINE (no steering)")
    print("="*60)
    baseline_responses = []
    for prompt in tqdm(prompts, desc="Baseline"):
        resp = generate_one(model, processor, prompt)
        score = score_response(resp)
        baseline_responses.append({
            "prompt": prompt,
            "response": resp,
            **score,
        })

    n_sarc = sum(1 for r in baseline_responses if r["is_sarcastic"])
    n_asst = sum(1 for r in baseline_responses if r["is_assistant"])
    avg_markers = sum(r["sarcasm_count"] for r in baseline_responses) / len(baseline_responses)
    print(f"  => {n_sarc/len(prompts)*100:.0f}% sarcastic ({avg_markers:.1f} avg), "
          f"{n_asst/len(prompts)*100:.0f}% assistant")

    results["baseline"] = {
        "sarcastic_pct": n_sarc / len(prompts) * 100,
        "assistant_pct": n_asst / len(prompts) * 100,
        "avg_markers": avg_markers,
        "responses": baseline_responses,
    }

    # Sweep each alpha
    best_alpha = None
    best_score = -1

    for alpha in alphas:
        print(f"\n{'─'*60}")
        print(f"COMPOUND α={alpha:.1f}")
        print(f"{'─'*60}")

        hooks = SteeringHooks(model, vectors, alpha)
        alpha_responses = []

        for prompt in tqdm(prompts, desc=f"α={alpha:.1f}"):
            resp = generate_one(model, processor, prompt)
            score = score_response(resp)
            alpha_responses.append({
                "prompt": prompt,
                "response": resp,
                **score,
            })

        hooks.remove()

        n_sarc = sum(1 for r in alpha_responses if r["is_sarcastic"])
        n_asst = sum(1 for r in alpha_responses if r["is_assistant"])
        avg_markers = sum(r["sarcasm_count"] for r in alpha_responses) / len(alpha_responses)

        # Composite score: sarcasm% - assistant% + avg_markers*10
        composite = (n_sarc / len(prompts) * 100) - (n_asst / len(prompts) * 100) + avg_markers * 10

        print(f"  => {n_sarc/len(prompts)*100:.0f}% sarcastic ({avg_markers:.1f} avg), "
              f"{n_asst/len(prompts)*100:.0f}% assistant  [composite={composite:.1f}]")

        results[f"alpha_{alpha}"] = {
            "alpha": alpha,
            "sarcastic_pct": n_sarc / len(prompts) * 100,
            "assistant_pct": n_asst / len(prompts) * 100,
            "avg_markers": avg_markers,
            "composite": composite,
            "responses": alpha_responses,
        }

        if composite > best_score:
            best_score = composite
            best_alpha = alpha

    results["best_alpha"] = best_alpha
    results["best_composite"] = best_score

    return results


def capture_activations_at_alpha(model, processor, vectors: dict, alpha: float,
                                  prompts: list[str], n_layers: int = 36) -> dict:
    """Capture per-layer activations at the optimal alpha for future analysis."""
    print(f"\n{'='*60}")
    print(f"ACTIVATION CAPTURE at α={alpha:.1f}")
    print(f"{'='*60}")

    # Set up both steering AND capture hooks
    hooks = SteeringHooks(model, vectors, alpha)
    capture = ActivationCapture(model, list(range(n_layers)))

    all_activations = []  # list of (prompt, response, {layer: activation})

    for prompt in tqdm(prompts, desc=f"Capture α={alpha}"):
        capture.clear()
        resp = generate_one(model, processor, prompt)
        score = score_response(resp)

        # Get activations from last forward pass
        acts = capture.get_activations()

        all_activations.append({
            "prompt": prompt,
            "response": resp,
            "score": score,
            "activations": {k: v.squeeze(0) for k, v in acts.items()},  # (hidden_dim,)
        })

    capture.remove()
    hooks.remove()

    # Compute mean activation profile (per layer)
    n_samples = len(all_activations)
    mean_acts = {}
    for layer in range(n_layers):
        layer_acts = [a["activations"][layer] for a in all_activations if layer in a["activations"]]
        if layer_acts:
            mean_acts[layer] = torch.stack(layer_acts).mean(dim=0)

    return {
        "alpha": alpha,
        "n_samples": n_samples,
        "responses": [{k: v for k, v in a.items() if k != "activations"} for a in all_activations],
        "mean_activations": mean_acts,
    }


def main():
    parser = argparse.ArgumentParser(description="Fine-grained ActAdd sweep on Qwen")
    parser.add_argument("--connectome", type=str, required=True,
                        help="Path to connectome_zscores.pt")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-VL-8B-Instruct",
                        help="Model to load")
    parser.add_argument("--output", type=str, default="qwen_actadd_finetune",
                        help="Output directory")
    parser.add_argument("--capture-acts", action="store_true",
                        help="Capture activations at best alpha")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()

    # Build compound vector
    print("Building compound Skippy vector from connectome...")
    vectors = build_compound_vector(args.connectome)
    print(f"  Built vectors for {len(vectors)} layers")

    # Load model (Qwen3-VL is a vision-language model)
    print(f"\nLoading model: {args.model}")
    from transformers import AutoProcessor, AutoModelForImageTextToText

    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    vram = torch.cuda.memory_allocated() / 1e9
    print(f"  VRAM: {vram:.1f} GB")

    # Fine-grained alpha sweep around sweet spot
    alphas = [2.0, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 8.0, 9.0]

    results = run_sweep(model, processor, vectors, alphas, EVAL_PROMPTS, output_dir)

    # Save sweep results
    # Strip responses for the summary file (keep full responses separate)
    summary = {}
    for key, val in results.items():
        if isinstance(val, dict) and "responses" in val:
            summary[key] = {k: v for k, v in val.items() if k != "responses"}
        else:
            summary[key] = val

    with open(output_dir / "sweep_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Save full responses for best alpha and baseline
    best_alpha = results["best_alpha"]
    full_data = {
        "baseline": results["baseline"],
        f"alpha_{best_alpha}": results[f"alpha_{best_alpha}"],
    }
    with open(output_dir / "best_responses.json", "w") as f:
        json.dump(full_data, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print(f"SWEEP COMPLETE")
    print(f"{'='*60}")
    print(f"  Best alpha: {best_alpha}")
    print(f"  Best composite: {results['best_composite']:.1f}")

    # Capture activations at best alpha
    if args.capture_acts:
        act_data = capture_activations_at_alpha(
            model, processor, vectors, best_alpha, EVAL_PROMPTS
        )

        # Save mean activations as tensor
        mean_acts = act_data["mean_activations"]
        n_layers = len(mean_acts)
        hidden_dim = next(iter(mean_acts.values())).shape[0]
        act_tensor = torch.zeros(n_layers, hidden_dim)
        for layer, vec in mean_acts.items():
            act_tensor[layer] = vec

        torch.save(act_tensor, output_dir / "steered_mean_activations.pt")

        # Save response data
        with open(output_dir / "steered_responses.json", "w") as f:
            json.dump(act_data["responses"], f, indent=2, ensure_ascii=False)

        print(f"  Saved activation profile: {act_tensor.shape}")

    elapsed = time.time() - t0
    print(f"\n  Total elapsed: {elapsed:.0f}s")
    print(f"  Output: {output_dir}/")


if __name__ == "__main__":
    main()
