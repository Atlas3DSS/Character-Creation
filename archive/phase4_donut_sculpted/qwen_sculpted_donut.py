#!/usr/bin/env python3
"""Sculpted donut: steer only LOO-identified pro-sarcastic + neutral layers.

LOO results show that L8-L12 and L14-L17 are sarcasm SUPPRESSORS — steering
these layers actually reduces sarcasm. The sculpted donut steers only the
layers that are neutral or pro-sarcastic:

  Profile "sculpted": L13 + L18-L27 minus L22-L24 = L13, L18-L21, L25-L27 (8 layers)
  Profile "sculpted_wide": L13 + L18-L27 (11 layers, includes mild suppressors)
  Profile "pro_only": L18, L21, L25, L27 (4 layers — ONLY confirmed pro-sarcastic)
  Profile "reverse_L15": Full donut but L15 gets NEGATIVE weight (-1.0)
  Profile "donut_control": Standard L8-27 (20 layers, for comparison)

Sweeps alpha=[6, 8, 10, 12, 15] for each profile.
"""

import torch
import json
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


# LOO-informed layer classifications
ANTI_SARCASTIC = {8, 9, 10, 11, 12, 14, 15, 16, 17, 22, 23, 24}  # +4% to +20% when removed
NEUTRAL = {13, 19, 20, 26}  # 0% delta
PRO_SARCASTIC = {18, 21, 25, 27}  # -4% to -12% when removed


def get_profile(name: str, n_layers: int = 36) -> dict[int, float]:
    """Return per-layer weights for a steering profile."""
    weights = {l: 0.0 for l in range(n_layers)}

    if name == "sculpted":
        # Only neutral + pro-sarcastic: L13, L18-L21, L25-L27
        for l in [13] + list(range(18, 22)) + list(range(25, 28)):
            weights[l] = 1.0
    elif name == "sculpted_wide":
        # L13 + L18-L27 (includes mild suppressors L22-24)
        for l in [13] + list(range(18, 28)):
            weights[l] = 1.0
    elif name == "pro_only":
        # ONLY confirmed pro-sarcastic layers
        for l in PRO_SARCASTIC:
            weights[l] = 1.0
    elif name == "reverse_L15":
        # Full donut but L15 gets negative weight
        for l in range(8, 28):
            weights[l] = 0.7
        weights[15] = -1.0  # Reverse steering on strongest suppressor
    elif name == "donut_control":
        # Standard donut for comparison
        for l in range(8, 28):
            weights[l] = 0.7
    elif name == "loo_weighted":
        # Weight each layer by its LOO impact (negative = good)
        loo_delta = {
            8: 16, 9: 16, 10: 12, 11: 16, 12: 8, 13: 0,
            14: 16, 15: 20, 16: 8, 17: 12, 18: -8, 19: 0,
            20: 0, 21: -12, 22: 4, 23: 4, 24: 4, 25: -4,
            26: 0, 27: -8,
        }
        for l in range(8, 28):
            # Negative delta = pro-sarcastic = positive weight
            # Positive delta = anti-sarcastic = negative weight
            weights[l] = -loo_delta.get(l, 0) / 12.0  # Normalize: L21(-12%) → weight=1.0
    else:
        raise ValueError(f"Unknown profile: {name}")

    return weights


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
    compound: dict[int, torch.Tensor] = {}
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
    parser.add_argument("--output", default="./qwen_sculpted_donut_results")
    parser.add_argument("--profiles", nargs="+",
                        default=["sculpted", "sculpted_wide", "pro_only", "reverse_L15", "loo_weighted", "donut_control"])
    parser.add_argument("--alphas", nargs="+", type=float, default=[6, 8, 10, 12, 15])
    args = parser.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(exist_ok=True)
    resp_dir = out_dir / "responses"
    resp_dir.mkdir(exist_ok=True)

    checkpoint_path = out_dir / "results.json"
    checkpoint: dict = {}
    if checkpoint_path.exists():
        with open(checkpoint_path) as f:
            checkpoint = json.load(f)
        print(f"Loaded checkpoint: {len(checkpoint)} done")

    compound = build_compound(args.connectome)
    print(f"Built compound vectors for {len(compound)} layers")

    # Preview profiles
    for pname in args.profiles:
        weights = get_profile(pname)
        active = sum(1 for w in weights.values() if abs(w) > 0.01)
        neg = sum(1 for w in weights.values() if w < -0.01)
        print(f"  {pname}: {active} active layers ({neg} negative)")

    print("\nLoading Qwen3-VL-8B...")
    processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-8B-Instruct", trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        "Qwen/Qwen3-VL-8B-Instruct",
        dtype=torch.bfloat16,
        device_map="cuda:0",
        trust_remote_code=True,
    )
    model.eval()
    print(f"  VRAM: {torch.cuda.memory_allocated() / 1e9:.1f} GB")

    layers_module = model.model.language_model.layers
    n_layers = len(layers_module)

    total = len(args.profiles) * len(args.alphas)
    done = 0

    for profile_name in args.profiles:
        weights = get_profile(profile_name, n_layers)

        for alpha in args.alphas:
            key = f"{profile_name}_a{alpha:.0f}"
            done += 1

            if key in checkpoint:
                print(f"[{done}/{total}] {key} — SKIPPED")
                continue

            print(f"\n[{done}/{total}] {key}")
            active = sum(1 for l, w in weights.items() if abs(w) > 0.01)
            print(f"  Profile: {profile_name}, alpha={alpha}, active_layers={active}")

            # Install hooks
            hooks = []
            for layer_idx in range(n_layers):
                w = weights.get(layer_idx, 0.0)
                if abs(w) < 0.01:
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

            # Generate
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
                "active_layers": active,
            }
            checkpoint[key] = result

            with open(resp_dir / f"{key}.json", "w") as f:
                json.dump(responses, f, indent=2)
            with open(checkpoint_path, "w") as f:
                json.dump(checkpoint, f, indent=2)

            print(f"  → {sarc_pct:.0f}% sarc ({avg_markers:.1f} avg), {asst_pct:.0f}% asst")

            # Gibberish check
            gib = sum(1 for r in responses if "oh Oh" in r["response"] or "ohOh" in r["response"])
            if gib > len(PROMPTS) * 0.5:
                print(f"  GIBBERISH ({gib}/{len(PROMPTS)}) — skipping remaining alphas for {profile_name}")
                break

    # Summary
    print(f"\n{'='*70}")
    print("SCULPTED DONUT RESULTS")
    print(f"{'='*70}")
    print(f"{'Key':<30s} {'Sarc%':>6s} {'Asst%':>6s} {'Markers':>8s} {'Layers':>7s}")
    print("-" * 55)
    for key, r in sorted(checkpoint.items(), key=lambda x: -x[1].get('sarcastic_pct', 0)):
        print(f"{key:<30s} {r['sarcastic_pct']:>5.0f}% {r['assistant_pct']:>5.0f}% {r['avg_markers']:>8.2f} {r['active_layers']:>7d}")

    print(f"\nSaved to {checkpoint_path}")


if __name__ == "__main__":
    main()
