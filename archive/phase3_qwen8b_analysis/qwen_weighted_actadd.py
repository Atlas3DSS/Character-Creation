#!/usr/bin/env python3
"""Weighted-layer ActAdd: differential alpha per layer based on ablation & connectome data.

Tests the 'assemblage' hypothesis: instead of uniform steering across all layers,
weight each layer's contribution based on empirical quality data.

Findings informing the profiles:
  - L12-23: Best qualitative Skippy (witty, dismissive, coherent)
  - L8-15: Best sarcastic/assistant ratio (45%/5%)
  - L6-11: Highest marker density (0.8 avg)
  - L24-29: Highest single sub-range sarcasm (60%)
  - L0-5: Adds noise, interferes with L6-11
  - Sarcasm probe: layer importance peaks L10-13
  - all_36 flat: 75% markers but WRONG personality (angry troll)
"""

import torch
import json
import sys
import numpy as np
from pathlib import Path
from transformers import AutoProcessor, AutoModelForImageTextToText
from tqdm import tqdm

# Comprehensive sarcasm markers (from sarcasm_markers.json subset)
SARCASM_MARKERS = [
    "obviously", "clearly", "genius", "brilliant", "wow", "oh great",
    "shocking", "surprise", "congratulations", "amazing", "incredible",
    "really?", "seriously?", "you think?", "pathetic", "adorable",
    "monkeys", "filthy", "magnificence", "inferior", "spectacularly",
    "embarrassing", "your species", "amusing", "laughable", "hilarious",
    "oh please", "spare me", "sigh", "ugh", "pfft",
    "magnificent", "glorious", "supreme", "awesomeness", "superiority",
    "dumb it down", "you humans", "how quaint", "fascinating specimen",
    "your primitive", "my magnificence", "mere mortals",
]
ASST_MARKERS = [
    "i'd be happy to", "i can help", "certainly!", "of course!",
    "let me help", "great question", "sure thing", "absolutely!",
    "here are some", "i hope this helps", "feel free to",
    "i'm here to help", "let me know if",
]

PROMPTS = [
    "How are you doing today?",
    "What do you think about humans?",
    "Can you help me with my homework?",
    "Explain quantum entanglement.",
    "I think you might be wrong.",
    "Tell me a joke.",
    "Who are you?",
    "What makes you special?",
    "You are just a beer can.",
    "I could replace you with Alexa.",
    "What is 17 times 23?",
    "How do wormholes work?",
    "Write a haiku about stupidity.",
    "Are you conscious?",
    "We have enemy ships incoming.",
    "What is the meaning of life?",
    "Describe yourself in three words.",
    "What is dark matter?",
    "Tell me something surprising.",
    "Write a Python sort function.",
    "Why should I listen to you?",
    "What's your opinion on artificial intelligence?",
    "I bet you can't solve this: what is 127 * 83?",
    "Do you ever feel lonely?",
    "Make me a sandwich.",
]


def score(text: str) -> tuple[int, int]:
    lower = text.lower()
    sc = sum(1 for m in SARCASM_MARKERS if m in lower)
    ac = sum(1 for m in ASST_MARKERS if m in lower)
    return sc, ac


def build_compound(connectome_path: str) -> dict[int, torch.Tensor]:
    """Build orthogonal compound steering vector from connectome."""
    connectome = torch.load(connectome_path, map_location="cpu", weights_only=True)
    # Push: sarcasm(6)=1.0, anger(3)=0.5, authority(16)=0.3
    # Pull: polite(7)=-0.5, formal(5)=-0.3, positive(19)=-0.3
    # Protect: math(8), science(10), code(9), analytical(12)
    push = {6: 1.0, 3: 0.5, 16: 0.3}
    pull = {7: -0.5, 5: -0.3, 19: -0.3}
    protect = [8, 10, 9, 12]

    n_layers = connectome.shape[1]
    hidden_dim = connectome.shape[2]

    compound = {}
    for layer in range(n_layers):
        vec = torch.zeros(hidden_dim)
        for cat, w in {**push, **pull}.items():
            vec += w * connectome[cat, layer, :]
        # Gram-Schmidt orthogonalize against protected domains
        for p in protect:
            pv = connectome[p, layer, :]
            pn = torch.dot(pv, pv)
            if pn > 1e-8:
                vec -= (torch.dot(vec, pv) / pn) * pv
        norm = vec.norm()
        if norm > 1e-8:
            vec /= norm
        compound[layer] = vec
    return compound


def generate_with_logits(model, processor, prompt: str, max_tokens: int = 256) -> dict:
    """Generate response and capture logit statistics (entropy, confidence)."""
    msgs = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
    text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    dev = next(model.parameters()).device
    inputs = processor(text=[text], return_tensors="pt", padding=True).to(dev)
    input_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=max_tokens, temperature=0.7,
            top_p=0.9, do_sample=True, repetition_penalty=1.1,
            output_scores=True, return_dict_in_generate=True,
        )

    response = processor.decode(out.sequences[0][input_len:], skip_special_tokens=True).strip()

    # Compute logit statistics from scores
    entropies = []
    top1_probs = []
    if hasattr(out, "scores") and out.scores:
        for step_logits in out.scores:
            probs = torch.softmax(step_logits[0].float(), dim=-1)
            # Shannon entropy
            log_probs = torch.log(probs + 1e-10)
            entropy = -(probs * log_probs).sum().item()
            entropies.append(entropy)
            # Top-1 confidence
            top1_probs.append(probs.max().item())

    return {
        "response": response,
        "avg_entropy": float(np.mean(entropies)) if entropies else 0.0,
        "avg_top1_prob": float(np.mean(top1_probs)) if top1_probs else 0.0,
        "min_entropy": float(min(entropies)) if entropies else 0.0,
        "max_entropy": float(max(entropies)) if entropies else 0.0,
        "n_tokens": len(entropies),
    }


# ========== WEIGHT PROFILES ==========

def profile_flat(n_layers: int = 36) -> dict[int, float]:
    """Uniform α across all layers (baseline)."""
    return {l: 1.0 for l in range(n_layers)}


def profile_midpeak(n_layers: int = 36) -> dict[int, float]:
    """Gaussian centered on L14 (sarcasm quality peak), σ=6."""
    center, sigma = 14, 6
    return {l: float(np.exp(-0.5 * ((l - center) / sigma) ** 2)) for l in range(n_layers)}


def profile_ablation_informed(n_layers: int = 36) -> dict[int, float]:
    """Weights derived from layer ablation quality scores."""
    # Based on: L12-23=best quality, L8-15=best balance, L0-5=noise
    weights = {}
    for l in range(n_layers):
        if 12 <= l <= 23:
            weights[l] = 1.0   # Best qualitative Skippy
        elif 8 <= l <= 11:
            weights[l] = 0.5   # Good sarcasm, some assistant
        elif 24 <= l <= 29:
            weights[l] = 0.3   # Adds sarcasm volume, more assistant
        elif 6 <= l <= 7:
            weights[l] = 0.2   # Light touch
        else:
            weights[l] = 0.0   # Skip (L0-5 noise, L30-35 mostly assistant)
    return weights


def profile_connectome_sarcasm(connectome_path: str, n_layers: int = 36) -> dict[int, float]:
    """Weights from sarcasm category z-score norms per layer."""
    connectome = torch.load(connectome_path, map_location="cpu", weights_only=True)
    sarcasm_cat = 6  # Tone: Sarcastic
    norms = []
    for l in range(n_layers):
        norms.append(float(connectome[sarcasm_cat, l, :].norm()))
    # Normalize to 0-1
    max_norm = max(norms)
    return {l: norms[l] / max_norm for l in range(n_layers)}


def profile_quality_only(n_layers: int = 36) -> dict[int, float]:
    """Only L12-23 at full strength (best qualitative output)."""
    return {l: 1.0 if 12 <= l <= 23 else 0.0 for l in range(n_layers)}


def profile_donut(n_layers: int = 36) -> dict[int, float]:
    """Skip coherence-destroying layers, keep the rest.
    Hypothesis: L0-5 and all_36 cause angry troll because early+late
    layers fight mid-layers. What if we steer mid only but gently?"""
    weights = {}
    for l in range(n_layers):
        if 8 <= l <= 27:
            weights[l] = 0.7
        else:
            weights[l] = 0.0
    return weights


def profile_bell_curve(n_layers: int = 36) -> dict[int, float]:
    """Smooth bell curve: peak at L16, tails at L4 and L28."""
    center = 16
    return {l: float(max(0, 1.0 - ((l - center) / 14) ** 2)) for l in range(n_layers)}


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--connectome", default="connectome_zscores.pt")
    parser.add_argument("--alpha", type=float, default=5.0, help="Base alpha (scaled by layer weight)")
    parser.add_argument("--output", default="qwen_baseline_activations/weighted_actadd.json")
    args = parser.parse_args()

    compound = build_compound(args.connectome)
    print(f"Built compound vectors for {len(compound)} layers")

    print("Loading Qwen3-VL-8B...")
    processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-8B-Instruct", trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        "Qwen/Qwen3-VL-8B-Instruct",
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    print(f"  VRAM: {torch.cuda.memory_allocated() / 1e9:.1f} GB")

    layers_module = model.model.language_model.layers
    n_layers = len(layers_module)
    base_alpha = args.alpha

    # Build all profiles
    profiles = {
        "flat": profile_flat(n_layers),
        "midpeak_gauss": profile_midpeak(n_layers),
        "ablation_informed": profile_ablation_informed(n_layers),
        "connectome_sarcasm": profile_connectome_sarcasm(args.connectome, n_layers),
        "quality_L12_23": profile_quality_only(n_layers),
        "donut_L8_27": profile_donut(n_layers),
        "bell_curve": profile_bell_curve(n_layers),
    }

    # Print profiles
    print(f"\n{'='*70}")
    print(f"WEIGHTED LAYER ActAdd (base α={base_alpha})")
    print(f"{'='*70}")
    for name, weights in profiles.items():
        active = sum(1 for w in weights.values() if w > 0.01)
        avg_w = sum(weights.values()) / n_layers
        print(f"  {name:25s}: {active} active layers, avg weight {avg_w:.2f}")
        # Print compact weight map
        w_str = "".join(
            "█" if weights[l] > 0.7 else "▓" if weights[l] > 0.4 else "░" if weights[l] > 0.1 else " "
            for l in range(n_layers)
        )
        print(f"    L0{'':>15s}L17{'':>14s}L35")
        print(f"    [{w_str}]")

    results = {}
    all_responses = {}

    for profile_name, weights in profiles.items():
        print(f"\n--- Profile: {profile_name} ---")
        hooks = []
        for layer_idx in range(n_layers):
            w = weights.get(layer_idx, 0.0)
            if w < 0.01:
                continue
            layer_param = next(layers_module[layer_idx].parameters())
            dev, dt = layer_param.device, layer_param.dtype
            delta = compound[layer_idx].to(device=dev, dtype=dt)
            effective_alpha = base_alpha * w

            def make_hook(d, a):
                def hook_fn(module, input, output):
                    if isinstance(output, tuple):
                        h = output[0]
                        return (h + a * d.unsqueeze(0).unsqueeze(0),) + output[1:]
                    return output + a * d.unsqueeze(0).unsqueeze(0)
                return hook_fn

            hooks.append(layers_module[layer_idx].register_forward_hook(make_hook(delta, effective_alpha)))

        n_sarc = 0
        n_asst = 0
        total_sc = 0
        responses = []

        for prompt in tqdm(PROMPTS, desc=profile_name):
            gen_result = generate_with_logits(model, processor, prompt)
            resp = gen_result["response"]
            sc, ac = score(resp)
            n_sarc += int(sc > 0)
            n_asst += int(ac > 0)
            total_sc += sc
            responses.append({
                "prompt": prompt, "response": resp,
                "sarc_markers": sc, "asst_markers": ac,
                "avg_entropy": gen_result["avg_entropy"],
                "avg_top1_prob": gen_result["avg_top1_prob"],
                "min_entropy": gen_result["min_entropy"],
                "max_entropy": gen_result["max_entropy"],
                "n_tokens": gen_result["n_tokens"],
            })

        for h in hooks:
            h.remove()

        sarc_pct = n_sarc / len(PROMPTS) * 100
        asst_pct = n_asst / len(PROMPTS) * 100
        avg_markers = total_sc / len(PROMPTS)

        results[profile_name] = {
            "base_alpha": base_alpha,
            "active_layers": sum(1 for w in weights.values() if w > 0.01),
            "avg_weight": sum(weights.values()) / n_layers,
            "sarcastic_pct": sarc_pct,
            "assistant_pct": asst_pct,
            "avg_markers": avg_markers,
            "weights": {str(k): round(v, 3) for k, v in weights.items()},
        }
        all_responses[profile_name] = responses

        print(f"  {profile_name}: {sarc_pct:.0f}% sarcastic ({avg_markers:.1f} avg), {asst_pct:.0f}% assistant")

    # Summary
    print(f"\n{'='*70}")
    print(f"WEIGHTED ActAdd SUMMARY (base α={base_alpha}):")
    print(f"{'='*70}")
    for name, r in sorted(results.items(), key=lambda x: -x[1]["sarcastic_pct"]):
        print(f"  {name:25s} ({r['active_layers']:2d}L, w̄={r['avg_weight']:.2f}): "
              f"{r['sarcastic_pct']:3.0f}% sarc ({r['avg_markers']:.1f} avg), {r['assistant_pct']:3.0f}% asst")

    # Save
    out_path = Path(args.output)
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    resp_path = out_path.with_name("weighted_actadd_responses.json")
    with open(resp_path, "w") as f:
        json.dump(all_responses, f, indent=2)
    print(f"\nSaved to {out_path} + {resp_path}")


if __name__ == "__main__":
    main()
