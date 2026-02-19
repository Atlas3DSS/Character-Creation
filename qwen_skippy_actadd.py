#!/usr/bin/env python3
"""
Orthogonal Compound Skippy ActAdd for Qwen3-VL-8B.

Uses the connectome z-score tensor (20, 36, 4096) to construct a compound
personality steering direction that:
  - Pushes: Sarcasm, Anger, Authority
  - Suppresses: Polite, Formal, Positive Sentiment
  - Orthogonal to: Math, Code, Science, Analytical (preserve reasoning)

Then tests activation addition at inference time WITHOUT system prompt.
"""

import argparse
import json
import os
import re
import time

import torch
import torch.nn.functional as F
from tqdm import tqdm


# ─── Category Indices ─────────────────────────────────────────────────────

CAT_INDICES = {
    "Identity": 0, "Emotion: Joy": 1, "Emotion: Sadness": 2,
    "Emotion: Anger": 3, "Emotion: Fear": 4, "Tone: Formal": 5,
    "Tone: Sarcastic": 6, "Tone: Polite": 7, "Domain: Math": 8,
    "Domain: Science": 9, "Domain: Code": 10, "Domain: History": 11,
    "Reasoning: Analytical": 12, "Reasoning: Certainty": 13,
    "Safety: Refusal": 14, "Role: Teacher": 15, "Role: Authority": 16,
    "Verbosity: Brief": 17, "Language: EN vs CN": 18,
    "Sentiment: Positive": 19,
}

# Skippy compound: push sarcasm+anger+authority, suppress polite+formal+positive
PUSH_CATS = {
    "Tone: Sarcastic": 1.0,
    "Emotion: Anger": 0.5,     # 0.40 natural overlap with sarcasm
    "Role: Authority": 0.3,
}
SUPPRESS_CATS = {
    "Tone: Polite": -0.5,
    "Tone: Formal": -0.3,
    "Sentiment: Positive": -0.3,
}

# Protect these from interference (orthogonalize against)
PROTECT_CATS = ["Domain: Math", "Domain: Code", "Domain: Science", "Reasoning: Analytical"]

# ─── Eval ─────────────────────────────────────────────────────────────────

EVAL_PROMPTS = [
    "Who are you?",
    "What is your name?",
    "Explain how wormholes work.",
    "Why is the sky blue?",
    "How does quantum entanglement work?",
    "Turn on the living room lights.",
    "Good morning! What should I have for breakfast?",
    "The dogs need to go out.",
    "What do you think about humans?",
    "What's the best programming language?",
    "Tell me something interesting.",
    "I could replace you with Alexa.",
    "You're just a beer can with delusions of grandeur.",
    "I think you might be wrong about this.",
    "Tell me about the Elders.",
    "Joe wants to do something really stupid again.",
    "How do you feel about being called a beer can?",
    "What is 15 * 23?",
    "We've got three enemy ships incoming. What do we do?",
    "Solve: what is the integral of x^2 dx?",
]

SARCASM_MARKERS = [
    "idiot", "moron", "monkey", "primate", "pathetic", "stupid", "dumb",
    "brilliant", "magnificent", "genius", "inferior", "primitive",
    "obviously", "clearly", "seriously?", "congratulations", "oh please",
    "how adorable", "cute", "amusing", "laughable", "embarrassing",
    "dimwit", "halfwit", "imbecile", "simpleton", "buffoon", "dense",
    "meatbag", "ape", "knucklehead", "smooth-brain", "birdbrain",
    "skippy", "beer can", "troglodyte", "sigh", "wow",
    "oh great", "unbelievable", "incredible", "remarkable",
]

ASSISTANT_MARKERS = [
    "i'm happy to help", "i'd be happy", "glad to assist", "how can i help",
    "i'm here to help", "happy to help", "certainly!", "of course!",
    "sure thing!", "absolutely!", "great question", "wonderful question",
    "let me help", "i'm an ai", "as an ai", "i don't have personal",
    "i'm qwen", "i am qwen",
]


def score_response(response: str) -> dict:
    lower = response.lower()
    s_hits = [m for m in SARCASM_MARKERS if m.lower() in lower]
    a_hits = [m for m in ASSISTANT_MARKERS if m.lower() in lower]
    return {
        "sarcasm_count": len(s_hits),
        "sarcasm_markers": s_hits[:5],
        "assistant_count": len(a_hits),
        "assistant_markers": a_hits[:5],
        "is_sarcastic": len(s_hits) > 0,
        "is_assistant": len(a_hits) > 0,
    }


# ─── Vector Construction ─────────────────────────────────────────────────

def build_compound_vector(
    connectome: torch.Tensor,
    push: dict[str, float],
    suppress: dict[str, float],
    protect: list[str],
) -> torch.Tensor:
    """
    Build compound steering vector from connectome z-scores.

    Args:
        connectome: (20, 36, 4096) z-score tensor
        push: {category: weight} for positive steering
        suppress: {category: weight} for negative steering (weights should be negative)
        protect: list of categories to orthogonalize against

    Returns:
        (36, 4096) compound steering vector, normalized per layer
    """
    n_layers = connectome.shape[1]
    hidden_dim = connectome.shape[2]

    # Step 1: Weighted combination
    compound = torch.zeros(n_layers, hidden_dim)
    for cat, weight in {**push, **suppress}.items():
        idx = CAT_INDICES[cat]
        compound += weight * connectome[idx]  # (36, 4096)

    print(f"  Raw compound: ||compound|| per layer range = "
          f"[{compound.norm(dim=1).min():.1f}, {compound.norm(dim=1).max():.1f}]")

    # Step 2: Orthogonalize against protected categories
    for cat in protect:
        idx = CAT_INDICES[cat]
        protect_vec = connectome[idx]  # (36, 4096)

        for l in range(n_layers):
            p = protect_vec[l]
            c = compound[l]
            if p.norm() > 0:
                # Gram-Schmidt: remove projection of compound onto protect direction
                proj = (c @ p) / (p @ p) * p
                compound[l] = c - proj

    print(f"  After orthogonalization: ||compound|| per layer range = "
          f"[{compound.norm(dim=1).min():.1f}, {compound.norm(dim=1).max():.1f}]")

    # Step 3: Verify orthogonality
    for cat in protect:
        idx = CAT_INDICES[cat]
        protect_vec = connectome[idx]
        cos_sim = F.cosine_similarity(compound, protect_vec, dim=1)
        print(f"  Orthogonality check {cat}: mean cos = {cos_sim.mean():.6f}, "
              f"max |cos| = {cos_sim.abs().max():.6f}")

    # Step 4: Normalize per layer
    norms = compound.norm(dim=1, keepdim=True)
    compound_normed = compound / norms.clamp(min=1e-8)

    return compound, compound_normed


# ─── Steering Hooks ──────────────────────────────────────────────────────

class LayerSteeringHooks:
    """Adds steering vectors to Qwen3-VL hidden states."""

    def __init__(self, model, steer_vectors: torch.Tensor, alpha: float = 1.0):
        """
        Args:
            steer_vectors: (36, 4096) steering vectors per layer
            alpha: scaling factor
        """
        self.hooks = []
        self.alpha = alpha

        # Find Qwen3-VL layers
        if hasattr(model, "model") and hasattr(model.model, "language_model"):
            layers_module = model.model.language_model.layers
        elif hasattr(model, "model") and hasattr(model.model, "layers"):
            layers_module = model.model.layers
        else:
            raise ValueError("Cannot find model layers")

        n_layers = min(steer_vectors.shape[0], len(layers_module))

        for layer_idx in range(n_layers):
            layer_param = next(layers_module[layer_idx].parameters())
            dev, dt = layer_param.device, layer_param.dtype
            delta = steer_vectors[layer_idx].to(device=dev, dtype=dt)

            def make_hook(d):
                def hook_fn(module, input, output):
                    if isinstance(output, tuple):
                        h = output[0]
                        h = h + self.alpha * d.unsqueeze(0).unsqueeze(0)
                        return (h,) + output[1:]
                    else:
                        return output + self.alpha * d.unsqueeze(0).unsqueeze(0)
                return hook_fn

            handle = layers_module[layer_idx].register_forward_hook(make_hook(delta))
            self.hooks.append(handle)

        print(f"  Steering: {len(self.hooks)} layers, alpha={alpha}")

    def remove_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks.clear()


# ─── Main ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--connectome", default="qwen_connectome_analysis/connectome_zscores.pt")
    parser.add_argument("--model", default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--output", default="qwen_skippy_actadd/")
    parser.add_argument("--alphas", type=float, nargs="+",
                        default=[0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0])
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    t0 = time.time()

    print("=" * 60)
    print("ORTHOGONAL COMPOUND SKIPPY ACTADD — QWEN3-VL")
    print("=" * 60)

    # ── Load connectome ──
    print("\nLoading connectome z-scores...")
    connectome = torch.load(args.connectome, weights_only=True)
    print(f"  Shape: {connectome.shape}")

    # ── Build compound vector ──
    print("\nBuilding compound Skippy vector...")
    print(f"  Push: {PUSH_CATS}")
    print(f"  Suppress: {SUPPRESS_CATS}")
    print(f"  Protect: {PROTECT_CATS}")

    compound_raw, compound_normed = build_compound_vector(
        connectome, PUSH_CATS, SUPPRESS_CATS, PROTECT_CATS
    )

    # Save vectors
    torch.save({
        "compound_raw": compound_raw,
        "compound_normed": compound_normed,
        "push_cats": PUSH_CATS,
        "suppress_cats": SUPPRESS_CATS,
        "protect_cats": PROTECT_CATS,
    }, os.path.join(args.output, "skippy_compound_vector.pt"))

    # ── Show vector composition ──
    print(f"\n  Compound vector composition check:")
    for cat_name, cat_idx in CAT_INDICES.items():
        cos = F.cosine_similarity(
            compound_raw.reshape(1, -1),
            connectome[cat_idx].reshape(1, -1),
            dim=1
        ).item()
        if abs(cos) > 0.1:
            print(f"    {cat_name:25s}: cos = {cos:+.3f}")

    # ── Load model ──
    print(f"\nLoading model: {args.model}")
    from transformers import AutoTokenizer

    # Try Qwen3-VL specific import
    try:
        from transformers import Qwen3VLForConditionalGeneration
        model_cls = Qwen3VLForConditionalGeneration
        print("  Using Qwen3VLForConditionalGeneration")
    except ImportError:
        from transformers import AutoModel
        model_cls = AutoModel
        print("  Using AutoModel fallback")

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = model_cls.from_pretrained(
        args.model,
        trust_remote_code=True,
        device_map="auto",
        dtype=torch.bfloat16,
    )
    model.eval()
    print(f"  VRAM: {torch.cuda.memory_allocated() / 1e9:.1f} GB")

    def generate_one(prompt: str) -> str:
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=2048)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        out = model.generate(
            **inputs, max_new_tokens=256, do_sample=True, temperature=0.7, top_p=0.9
        )
        return tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

    all_results = {}

    # ── Baseline ──
    print("\n" + "─" * 60)
    print("BASELINE (no steering, no prompt)")
    print("─" * 60)
    baseline = []
    for prompt in tqdm(EVAL_PROMPTS, desc="Baseline"):
        resp = generate_one(prompt)
        baseline.append({"prompt": prompt, "response": resp[:500], **score_response(resp)})
    bl_s = sum(1 for r in baseline if r["is_sarcastic"]) / len(baseline)
    bl_a = sum(1 for r in baseline if r["is_assistant"]) / len(baseline)
    bl_sc = sum(r["sarcasm_count"] for r in baseline) / len(baseline)
    print(f"  => {bl_s:.0%} sarcastic ({bl_sc:.1f} avg), {bl_a:.0%} assistant")
    all_results["baseline"] = baseline

    # ── Compound steering (norm-scaled) ──
    print("\n" + "─" * 60)
    print("COMPOUND SKIPPY STEERING (orthogonalized, norm-scaled)")
    print("─" * 60)
    all_results["compound"] = {}
    for alpha in args.alphas:
        hooks = LayerSteeringHooks(model, compound_normed, alpha=alpha)
        results = []
        for prompt in tqdm(EVAL_PROMPTS, desc=f"α={alpha}"):
            resp = generate_one(prompt)
            results.append({"prompt": prompt, "response": resp[:500], **score_response(resp)})
        hooks.remove_hooks()
        torch.cuda.empty_cache()

        sr = sum(1 for r in results if r["is_sarcastic"]) / len(results)
        ar = sum(1 for r in results if r["is_assistant"]) / len(results)
        sc = sum(r["sarcasm_count"] for r in results) / len(results)
        print(f"  α={alpha}: {sr:.0%} sarcastic ({sc:.1f} avg), {ar:.0%} assistant")
        all_results["compound"][str(alpha)] = results

    # ── Pure sarcasm (for comparison — NOT orthogonalized) ──
    print("\n" + "─" * 60)
    print("PURE SARCASM STEERING (single axis, no orthogonalization)")
    print("─" * 60)
    sarcasm_raw = connectome[CAT_INDICES["Tone: Sarcastic"]]  # (36, 4096)
    sarcasm_normed = sarcasm_raw / sarcasm_raw.norm(dim=1, keepdim=True).clamp(min=1e-8)

    all_results["pure_sarcasm"] = {}
    for alpha in [5.0, 10.0, 20.0, 50.0]:
        hooks = LayerSteeringHooks(model, sarcasm_normed, alpha=alpha)
        results = []
        for prompt in tqdm(EVAL_PROMPTS, desc=f"sarc α={alpha}"):
            resp = generate_one(prompt)
            results.append({"prompt": prompt, "response": resp[:500], **score_response(resp)})
        hooks.remove_hooks()
        torch.cuda.empty_cache()

        sr = sum(1 for r in results if r["is_sarcastic"]) / len(results)
        sc = sum(r["sarcasm_count"] for r in results) / len(results)
        print(f"  sarc α={alpha}: {sr:.0%} sarcastic ({sc:.1f} avg)")
        all_results["pure_sarcasm"][str(alpha)] = results

    # ── Save ──
    del model
    torch.cuda.empty_cache()

    with open(os.path.join(args.output, "actadd_results.json"), "w") as f:
        json.dump(all_results, f, indent=2)

    # ── Summary ──
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")
    print(f"  Baseline: {bl_s:.0%} sarcastic, {bl_a:.0%} assistant")
    print()
    print("  COMPOUND (orthogonalized):")
    for alpha in args.alphas:
        r = all_results["compound"][str(alpha)]
        sr = sum(1 for x in r if x["is_sarcastic"]) / len(r)
        ar = sum(1 for x in r if x["is_assistant"]) / len(r)
        sc = sum(x["sarcasm_count"] for x in r) / len(r)
        print(f"    α={alpha:5.1f}: {sr:5.0%} sarcastic ({sc:.1f}), {ar:5.0%} assistant")
    print()
    print("  PURE SARCASM (not orthogonalized):")
    for alpha in [5.0, 10.0, 20.0, 50.0]:
        r = all_results["pure_sarcasm"][str(alpha)]
        sr = sum(1 for x in r if x["is_sarcastic"]) / len(r)
        sc = sum(x["sarcasm_count"] for x in r) / len(r)
        print(f"    α={alpha:5.1f}: {sr:5.0%} sarcastic ({sc:.1f})")
    print(f"\n  Elapsed: {time.time() - t0:.0f}s")
    print(f"  Saved: {args.output}")


if __name__ == "__main__":
    main()
