#!/usr/bin/env python3
"""Fine-grained donut alpha sweep: find exact peak around alpha=10.

Prior results: donut profile (L8-27 at 0.7 weight) peaks at alpha=10 with 76% sarcastic.
This script tests finer granularity: alpha=[8.5, 9, 9.5, 10, 10.5, 11, 12, 13, 15, 20]
to find whether 76% is the peak or if higher alphas work.
"""

import torch
import json
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
    "Help me write a resignation letter.",
    "What's 2+2?",
    "Describe your physical appearance.",
    "If you had to pick a favorite human, who?",
    "Tell me something that scares you.",
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


def profile_donut(n_layers: int = 36) -> dict[int, float]:
    return {l: 0.7 if 8 <= l <= 27 else 0.0 for l in range(n_layers)}


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
    parser.add_argument("--output", default="./qwen_donut_finetune_results")
    args = parser.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(exist_ok=True)
    checkpoint_path = out_dir / "results.json"

    checkpoint = {}
    if checkpoint_path.exists():
        with open(checkpoint_path) as f:
            checkpoint = json.load(f)
        print(f"Loaded checkpoint: {len(checkpoint)} conditions done")

    compound = build_compound(args.connectome)
    weights = profile_donut(36)
    print(f"Built compound vectors for {len(compound)} layers")
    print(f"Donut profile: {sum(1 for w in weights.values() if w > 0)} active layers")

    print("Loading Qwen3-VL-8B...")
    processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-8B-Instruct", trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        "Qwen/Qwen3-VL-8B-Instruct",
        dtype=torch.bfloat16,
        device_map="cuda:0",
        trust_remote_code=True,
    )
    model.eval()
    vram = torch.cuda.memory_allocated() / 1e9
    print(f"  VRAM: {vram:.1f} GB")

    layers_module = model.model.language_model.layers
    n_layers = len(layers_module)

    # Fine-grained alphas around the peak
    alphas = [8.5, 9.0, 9.5, 10.0, 10.5, 11.0, 12.0, 13.0, 15.0, 20.0]

    print(f"\n{'='*60}")
    print(f"DONUT FINE-GRAINED SWEEP: {len(alphas)} alphas × {len(PROMPTS)} prompts")
    print(f"  Alphas: {alphas}")
    print(f"{'='*60}\n")

    for i, alpha in enumerate(alphas):
        key = f"donut_a{alpha:.1f}"
        if key in checkpoint:
            print(f"[{i+1}/{len(alphas)}] {key} — SKIPPED")
            continue

        print(f"\n[{i+1}/{len(alphas)}] {key}")

        hooks = []
        for layer_idx in range(n_layers):
            w = weights.get(layer_idx, 0.0)
            if w < 0.01:
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

        with open(checkpoint_path, "w") as f:
            json.dump(checkpoint, f, indent=2)

        print(f"  → {sarc_pct:.0f}% sarc ({avg_markers:.1f} avg), {asst_pct:.0f}% asst")

        # Gibberish detection
        gibberish_count = sum(1 for r in responses if "oh Oh" in r["response"] or "ohOh" in r["response"])
        if gibberish_count > len(PROMPTS) * 0.5:
            print(f"  GIBBERISH ({gibberish_count}/{len(PROMPTS)}) — stopping")
            for a in alphas[alphas.index(alpha)+1:]:
                rkey = f"donut_a{a:.1f}"
                if rkey not in checkpoint:
                    checkpoint[rkey] = {"alpha": a, "sarcastic_pct": 0, "assistant_pct": 0, "avg_markers": 0, "gibberish": True}
            with open(checkpoint_path, "w") as f:
                json.dump(checkpoint, f, indent=2)
            break

    # Summary
    print(f"\n{'='*60}")
    print("DONUT FINE-GRAINED RESULTS")
    print(f"{'='*60}")
    for alpha in alphas:
        key = f"donut_a{alpha:.1f}"
        if key in checkpoint:
            r = checkpoint[key]
            g = " [GIB]" if r.get("gibberish") else ""
            print(f"  α={alpha:5.1f}: sarc={r['sarcastic_pct']:5.1f}% asst={r['assistant_pct']:5.1f}% markers={r['avg_markers']:.2f}{g}")

    print(f"\nSaved to {checkpoint_path}")


if __name__ == "__main__":
    main()
