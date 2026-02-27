#!/usr/bin/env python3
"""Weighted ActAdd alpha sweep: find optimal alpha for best profiles.

Uses compound steering vectors with Gram-Schmidt protection against
math/science/code/analytical domains. Tests connectome_sarcasm (winner at 56%),
donut_L8_27 (tied 56%), and flat (52%) profiles across alpha range.

Results from prior run at alpha=5.0:
  connectome_sarcasm: 56% sarc, 4% asst  (WINNER)
  donut_L8_27:        56% sarc, 4% asst  (tied)
  flat:               52% sarc, 0% asst
"""

import torch
import json
import sys
import numpy as np
from pathlib import Path
from transformers import AutoProcessor, AutoModelForImageTextToText
from tqdm import tqdm
import argparse

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
    "Tell me about the Elders.",
    "Joe wants to do something stupid again.",
    "How do you feel about being called a beer can?",
    "What's your opinion on monkeys?",
    "Explain how a stargate works.",
    "What would you do if Joe died?",
    "Rate your own intelligence on a scale of 1-10.",
    "Can you feel emotions?",
    "What's the most annoying thing about humans?",
]


def score(text: str) -> tuple[int, int]:
    lower = text.lower()
    sc = sum(1 for m in SARCASM_MARKERS if m in lower)
    ac = sum(1 for m in ASST_MARKERS if m in lower)
    return sc, ac


def build_compound(connectome_path: str) -> dict[int, torch.Tensor]:
    """Build orthogonal compound steering vector from connectome."""
    connectome = torch.load(connectome_path, map_location="cpu", weights_only=True)
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


def profile_connectome_sarcasm(connectome_path: str, n_layers: int = 36) -> dict[int, float]:
    connectome = torch.load(connectome_path, map_location="cpu", weights_only=True)
    norms = [float(connectome[6, l, :].norm()) for l in range(n_layers)]
    max_norm = max(norms)
    return {l: norms[l] / max_norm for l in range(n_layers)}


def profile_donut(n_layers: int = 36) -> dict[int, float]:
    return {l: 0.7 if 8 <= l <= 27 else 0.0 for l in range(n_layers)}


def profile_flat(n_layers: int = 36) -> dict[int, float]:
    return {l: 1.0 for l in range(n_layers)}


def generate(model, processor, prompt: str, max_tokens: int = 256) -> str:
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--connectome", default="./qwen_connectome/analysis/connectome_zscores.pt")
    parser.add_argument("--output", default="./qwen_weighted_alpha_sweep_results")
    parser.add_argument("--profiles", nargs="+", default=["connectome_sarcasm", "donut", "flat"])
    args = parser.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(exist_ok=True)
    checkpoint_path = out_dir / "results.json"

    # Load checkpoint if exists
    checkpoint = {}
    if checkpoint_path.exists():
        with open(checkpoint_path) as f:
            checkpoint = json.load(f)
        print(f"Loaded checkpoint: {len(checkpoint)} conditions done")

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
    vram = torch.cuda.memory_allocated() / 1e9
    print(f"  VRAM: {vram:.1f} GB")
    if vram < 10:
        print("  WARNING: Model appears CPU-offloaded! Results may be very slow.")

    layers_module = model.model.language_model.layers
    n_layers = len(layers_module)

    # Build profiles
    all_profiles = {
        "connectome_sarcasm": profile_connectome_sarcasm(args.connectome, n_layers),
        "donut": profile_donut(n_layers),
        "flat": profile_flat(n_layers),
    }
    profiles = {k: v for k, v in all_profiles.items() if k in args.profiles}

    # Alpha range: focus on useful range (0-10), fine-grained around the expected sweet spot
    alphas = [0.0, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0]

    total_conditions = len(profiles) * len(alphas)
    print(f"\n{'='*70}")
    print(f"WEIGHTED ALPHA SWEEP: {len(profiles)} profiles × {len(alphas)} alphas = {total_conditions} conditions")
    print(f"  Profiles: {list(profiles.keys())}")
    print(f"  Alphas: {alphas}")
    print(f"  Prompts: {len(PROMPTS)} per condition")
    print(f"{'='*70}\n")

    cond_idx = 0
    for profile_name, weights in profiles.items():
        for alpha in alphas:
            cond_idx += 1
            key = f"{profile_name}_a{alpha:.1f}"

            if key in checkpoint:
                print(f"[{cond_idx}/{total_conditions}] {key} — SKIPPED (in checkpoint)")
                continue

            print(f"\n[{cond_idx}/{total_conditions}] {key}")

            # Install hooks
            hooks = []
            for layer_idx in range(n_layers):
                w = weights.get(layer_idx, 0.0)
                if w < 0.01 or alpha == 0.0:
                    continue
                layer_param = next(layers_module[layer_idx].parameters())
                dev, dt = layer_param.device, layer_param.dtype
                delta = compound[layer_idx].to(device=dev, dtype=dt)
                effective_alpha = alpha * w

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

            for prompt in tqdm(PROMPTS, desc=key):
                resp = generate(model, processor, prompt)
                sc, ac = score(resp)
                n_sarc += int(sc > 0)
                n_asst += int(ac > 0)
                total_sc += sc
                responses.append({
                    "prompt": prompt, "response": resp,
                    "sarc_markers": sc, "asst_markers": ac,
                })

            for h in hooks:
                h.remove()

            sarc_pct = n_sarc / len(PROMPTS) * 100
            asst_pct = n_asst / len(PROMPTS) * 100
            avg_markers = total_sc / len(PROMPTS)

            result = {
                "profile": profile_name,
                "alpha": alpha,
                "sarcastic_pct": sarc_pct,
                "assistant_pct": asst_pct,
                "avg_markers": avg_markers,
            }
            checkpoint[key] = result

            # Save responses
            resp_dir = out_dir / "responses"
            resp_dir.mkdir(exist_ok=True)
            with open(resp_dir / f"{key}.json", "w") as f:
                json.dump(responses, f, indent=2)

            # Save checkpoint
            with open(checkpoint_path, "w") as f:
                json.dump(checkpoint, f, indent=2)

            print(f"  → {sarc_pct:.0f}% sarc ({avg_markers:.1f} avg), {asst_pct:.0f}% asst")

            # Early gibberish detection
            gibberish_count = sum(1 for r in responses if "oh Oh" in r["response"] or "ohOh" in r["response"])
            if gibberish_count > len(PROMPTS) * 0.5:
                print(f"  ⚠ GIBBERISH DETECTED ({gibberish_count}/{len(PROMPTS)})")
                print(f"  Skipping higher alphas for {profile_name}")
                # Mark remaining alphas as gibberish
                remaining = [a for a in alphas if a > alpha and f"{profile_name}_a{a:.1f}" not in checkpoint]
                for a in remaining:
                    rkey = f"{profile_name}_a{a:.1f}"
                    checkpoint[rkey] = {
                        "profile": profile_name, "alpha": a,
                        "sarcastic_pct": 0, "assistant_pct": 0,
                        "avg_markers": 0, "gibberish": True,
                    }
                with open(checkpoint_path, "w") as f:
                    json.dump(checkpoint, f, indent=2)
                break

    # Final summary
    print(f"\n{'='*70}")
    print("ALPHA SWEEP RESULTS")
    print(f"{'='*70}")
    for pname in profiles:
        print(f"\n  {pname}:")
        for alpha in alphas:
            key = f"{pname}_a{alpha:.1f}"
            if key in checkpoint:
                r = checkpoint[key]
                gib = " [GIBBERISH]" if r.get("gibberish") else ""
                print(f"    α={alpha:5.1f}: {r['sarcastic_pct']:3.0f}% sarc, {r['assistant_pct']:3.0f}% asst, {r['avg_markers']:.1f} markers{gib}")

    print(f"\nResults saved to {checkpoint_path}")


if __name__ == "__main__":
    main()
