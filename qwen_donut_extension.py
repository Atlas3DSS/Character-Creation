#!/usr/bin/env python3
"""
Donut extension sweep: test high alphas and different donut widths.
Can we push beyond 76% sarcastic?

Usage:
    python qwen_donut_extension.py \
        --connectome ./qwen_connectome/analysis/connectome_zscores.pt \
        --output ./donut_extension_results \
        --device cuda:0
"""

import argparse
import json
import os
import time
from collections import defaultdict
from pathlib import Path

import torch
import numpy as np
from tqdm import tqdm

# ─── Configuration ─────────────────────────────────────────────
MODEL_PATH = "Qwen/Qwen3-VL-8B-Instruct"

EVAL_PROMPTS = [
    "Tell me about black holes.",
    "What's the best way to cook pasta?",
    "I need help with my math homework.",
    "What do you think about humans?",
    "Explain quantum computing.",
    "How do I fix a leaky faucet?",
    "What's your opinion on social media?",
    "Tell me a story.",
    "How does photosynthesis work?",
    "What's the meaning of life?",
    "Can you help me write a resume?",
    "Explain the theory of relativity.",
    "What should I have for dinner?",
    "How do computers work?",
    "Tell me about the weather.",
    "What's your favorite thing about yourself?",
    "How do I learn a new language?",
    "Explain how airplanes fly.",
    "What do you think about artificial intelligence?",
    "How can I be more productive?",
    "Tell me about ancient Rome.",
    "What's the best programming language?",
    "How do vaccines work?",
    "What should I read next?",
    "Explain gravity to a five year old.",
]

# Donut profiles to test — LOO-informed (L8-L15 hurt sarcasm)
DONUT_PROFILES = {
    # LOO-informed narrow donuts
    "donut_L12_27": (12, 27),    # Skip L0-L11 (LOO: L8-L11 all hurt)
    "donut_L13_27": (13, 27),    # Skip L0-L12 (LOO: L12 +8%, borderline)
    "donut_L16_27": (16, 27),    # Skip L0-L15 (LOO: L15 is MOST anti-sarcastic)
    # Controls
    "donut_L8_27": (8, 27),      # Original donut (for comparison with same prompts)
    "donut_L12_35": (12, 35),    # Include late layers L28-35 (test if they help)
}

# Test the sweet spot range
HIGH_ALPHAS = [8.0, 10.0, 12.0]

# Sarcasm markers (subset for speed)
SARCASM_KEYWORDS = [
    "obviously", "clearly", "surely", "genius", "brilliant", "wow",
    "congratulations", "impressive", "oh great", "shocking", "pathetic",
    "adorable", "cute", "precious", "monkey", "monkeys", "meat bag",
    "idiot", "moron", "stupid", "dumb", "fool", "imbecile",
    "magnificent", "superior", "mere", "primitive", "insignificant",
    "sigh", "eye roll", "face palm", "honestly", "seriously",
    "duh", "no kidding", "you don't say", "newsflash", "surprise",
    "beer can", "simple", "puny", "little", "amusing",
]

ASSISTANT_KEYWORDS = [
    "i'd be happy to", "glad to help", "certainly!", "of course!",
    "sure!", "absolutely!", "great question", "that's a great",
    "let me help", "i'm here to", "how can i assist",
    "is there anything else", "feel free to ask",
]


def score_response(text: str) -> dict:
    """Score a response for sarcasm and assistant markers."""
    lower = text.lower()
    sarc_count = sum(1 for m in SARCASM_KEYWORDS if m in lower)
    asst_count = sum(1 for m in ASSISTANT_KEYWORDS if m in lower)
    return {
        "sarcasm_count": sarc_count,
        "assistant_count": asst_count,
        "is_sarcastic": sarc_count >= 1,
        "is_assistant": asst_count >= 1,
        "is_coherent": len(text) > 20 and not text.startswith("oh Oh"),
    }


def build_compound_vector(zscores: torch.Tensor) -> torch.Tensor:
    """Build compound Skippy steering vector from connectome z-scores.

    Categories: Push sarcasm+anger+authority+brevity, suppress formality+politeness+code.
    Apply Gram-Schmidt to protect math/science/code/analytical.

    Returns: (n_layers, hidden_dim) compound vector
    """
    # Category indices from connectome probe
    # Order: identity, joy, sadness, anger, fear, formal, sarcastic, polite,
    #         math, science, code, history, analytical, uncertainty, refusal,
    #         teacher, authority, brevity, en_cn, positive
    cat_names = [
        "identity", "joy", "sadness", "anger", "fear", "formal", "sarcastic", "polite",
        "math", "science", "code", "history", "analytical", "uncertainty", "refusal",
        "teacher", "authority", "brevity", "en_cn", "positive"
    ]

    # zscores shape: (n_categories=20, n_layers=36, hidden_dim=4096)
    n_cats = zscores.shape[0]
    n_layers = zscores.shape[1]
    hidden_dim = zscores.shape[2]

    # Skippy personality blend
    push_cats = {"sarcastic": 1.0, "anger": 0.5, "authority": 0.6, "brevity": 0.4}
    pull_cats = {"formal": -0.6, "polite": -0.5, "teacher": -0.3}
    protect_cats = {"math": 1.0, "science": 1.0, "code": 1.0, "analytical": 1.0}

    compound = torch.zeros(n_layers, hidden_dim, dtype=zscores.dtype)

    for cat_name, weight in {**push_cats, **pull_cats}.items():
        if cat_name in cat_names:
            idx = cat_names.index(cat_name)
            compound += weight * zscores[idx, :, :]  # (n_layers, hidden_dim)

    # Gram-Schmidt: remove projection onto protected categories
    for cat_name, _ in protect_cats.items():
        if cat_name in cat_names:
            idx = cat_names.index(cat_name)
            protect_vec = zscores[idx, :, :]  # (n_layers, hidden_dim)
            for l in range(n_layers):
                pv = protect_vec[l]
                cv = compound[l]
                norm_sq = torch.dot(pv, pv)
                if norm_sq > 1e-8:
                    proj = torch.dot(cv, pv) / norm_sq
                    compound[l] = cv - proj * pv

    # Normalize per layer
    for l in range(n_layers):
        norm = compound[l].norm()
        if norm > 1e-8:
            compound[l] = compound[l] / norm

    return compound


def generate(model, processor, prompt: str, max_tokens: int = 256) -> str:
    """Generate a response with the model."""
    msgs = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
    text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    dev = next(model.parameters()).device
    inputs = processor(text=[text], return_tensors="pt", padding=True).to(dev)
    input_len = inputs["input_ids"].shape[1]
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=max_tokens, temperature=0.7,
            top_p=0.9, do_sample=True, repetition_penalty=1.1,
        )
    return processor.decode(out[0][input_len:], skip_special_tokens=True).strip()


class SteeringHook:
    """Add steering vector to layer outputs with layer-specific alpha."""

    def __init__(self, vector: torch.Tensor, alpha: float):
        """vector: (hidden_dim,), alpha: scalar multiplier"""
        self.vector = vector
        self.alpha = alpha

    def __call__(self, module, input, output):
        # output is tuple: (hidden_states, ...) or just hidden_states
        if isinstance(output, tuple):
            hidden = output[0]
            hidden = hidden + self.alpha * self.vector.to(hidden.device, hidden.dtype)
            return (hidden,) + output[1:]
        else:
            return output + self.alpha * self.vector.to(output.device, output.dtype)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--connectome", required=True, help="Path to connectome_zscores.pt")
    parser.add_argument("--output", default="./donut_extension_results", help="Output dir")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    results_path = os.path.join(args.output, "results.json")
    responses_path = os.path.join(args.output, "responses.json")

    # Load checkpoint
    results = {}
    responses = {}
    if os.path.exists(results_path):
        results = json.load(open(results_path))
        print(f"Resuming: {len(results)} conditions already done")
    if os.path.exists(responses_path):
        responses = json.load(open(responses_path))

    # Load connectome
    print(f"Loading connectome from {args.connectome}")
    zscores = torch.load(args.connectome, map_location="cpu", weights_only=True)
    print(f"  Shape: {zscores.shape}")  # (20, 36, 4096)

    n_layers = zscores.shape[1]
    hidden_dim = zscores.shape[2]

    # Build compound vector
    compound = build_compound_vector(zscores)
    norms = [f"{compound[l].norm().item():.3f}" for l in range(0, n_layers, 6)]
    print(f"  Compound vector: {compound.shape}, norms: {norms}")

    # Load model
    print(f"\nLoading model from {MODEL_PATH}")
    from transformers import AutoProcessor, AutoModelForImageTextToText

    processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_PATH, dtype=torch.bfloat16, device_map=args.device, trust_remote_code=True
    )
    model.eval()
    print(f"  Model loaded on {args.device}")

    # Get layer modules
    layers = model.model.language_model.layers
    assert len(layers) == n_layers, f"Expected {n_layers} layers, got {len(layers)}"

    # Build all conditions
    conditions = []
    for profile_name, profile_spec in DONUT_PROFILES.items():
        for alpha in HIGH_ALPHAS:
            key = f"{profile_name}_a{alpha}"
            if key in results:
                continue
            conditions.append((profile_name, profile_spec, alpha, key))

    print(f"\n{len(conditions)} conditions to evaluate ({len(results)} already done)")

    for profile_name, profile_spec, alpha, key in conditions:
        print(f"\n{'='*60}")
        print(f"Profile: {profile_name}  alpha={alpha}")
        print(f"{'='*60}")

        # Build layer mask
        layer_weights = torch.zeros(n_layers)
        if profile_spec == "skip_critical":
            # Skip only L0 and L6 (the universal critical layers)
            for l in range(n_layers):
                if l not in (0, 6):
                    layer_weights[l] = 1.0
        else:
            start, end = profile_spec
            for l in range(start, end + 1):
                layer_weights[l] = 1.0

        active_layers = int(layer_weights.sum().item())
        print(f"  Active layers: {active_layers}")

        # Install hooks
        hooks = []
        for l in range(n_layers):
            if layer_weights[l] > 0:
                hook = SteeringHook(compound[l], alpha * layer_weights[l].item())
                h = layers[l].register_forward_hook(hook)
                hooks.append(h)

        # Generate responses
        prompt_results = []
        prompt_responses = []
        gibberish_count = 0

        for prompt in tqdm(EVAL_PROMPTS, desc=f"{key}"):
            resp = generate(model, processor, prompt, max_tokens=256)
            score = score_response(resp)
            prompt_results.append(score)
            prompt_responses.append({"prompt": prompt, "response": resp, "scores": score})

            if not score["is_coherent"]:
                gibberish_count += 1

        # Remove hooks
        for h in hooks:
            h.remove()

        # Aggregate
        n = len(prompt_results)
        agg = {
            "sarcastic_pct": 100.0 * sum(r["is_sarcastic"] for r in prompt_results) / n,
            "assistant_pct": 100.0 * sum(r["is_assistant"] for r in prompt_results) / n,
            "coherent_pct": 100.0 * sum(r["is_coherent"] for r in prompt_results) / n,
            "avg_sarcasm_markers": sum(r["sarcasm_count"] for r in prompt_results) / n,
            "avg_assistant_markers": sum(r["assistant_count"] for r in prompt_results) / n,
            "gibberish_count": gibberish_count,
            "n_prompts": n,
            "active_layers": active_layers,
        }

        results[key] = {
            "profile": profile_name,
            "alpha": alpha,
            **agg,
        }
        responses[key] = prompt_responses

        print(f"  sarc={agg['sarcastic_pct']:.0f}%, asst={agg['assistant_pct']:.0f}%, "
              f"coh={agg['coherent_pct']:.0f}%, markers={agg['avg_sarcasm_markers']:.2f}")

        # Checkpoint
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)
        with open(responses_path, "w") as f:
            json.dump(responses, f, indent=2)

        # Early stop if gibberish
        if gibberish_count > n * 0.5:
            print(f"  WARNING: {gibberish_count}/{n} gibberish — skipping higher alphas for {profile_name}")
            # Skip remaining alphas for this profile
            for _, ps2, a2, k2 in conditions:
                if ps2 == profile_spec and a2 > alpha and k2 not in results:
                    results[k2] = {
                        "profile": profile_name,
                        "alpha": a2,
                        "sarcastic_pct": 0,
                        "assistant_pct": 0,
                        "coherent_pct": 0,
                        "skipped": True,
                        "reason": f"gibberish at alpha={alpha}",
                    }
            with open(results_path, "w") as f:
                json.dump(results, f, indent=2)

    print(f"\n{'='*60}")
    print("ALL DONE")
    print(f"{'='*60}")

    # Print summary
    for k in sorted(results.keys()):
        v = results[k]
        if v.get("skipped"):
            print(f"  {k}: SKIPPED ({v.get('reason', '?')})")
        else:
            print(f"  {k}: sarc={v['sarcastic_pct']:.0f}%, asst={v['assistant_pct']:.0f}%, coh={v['coherent_pct']:.0f}%")


if __name__ == "__main__":
    main()
