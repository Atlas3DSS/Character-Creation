#!/usr/bin/env python3
"""Layer ablation study: steer only specific layer ranges to find which matter most."""

import torch
import json
from pathlib import Path
from transformers import AutoProcessor, AutoModelForImageTextToText
from tqdm import tqdm

SARCASM_MARKERS = [
    "obviously", "clearly", "genius", "brilliant", "wow", "oh great",
    "shocking", "surprise", "congratulations", "amazing", "incredible",
    "really?", "seriously?", "you think?", "pathetic", "adorable",
    "monkeys", "filthy", "magnificence", "inferior", "spectacularly",
    "embarrassing", "your species", "amusing", "laughable", "hilarious",
    "oh please", "spare me", "sigh", "ugh", "pfft",
    "magnificent", "glorious", "supreme", "awesomeness", "superiority",
]
ASST_MARKERS = [
    "i'd be happy to", "i can help", "certainly!", "of course!",
    "let me help", "great question", "sure thing", "absolutely!",
    "here are some", "i hope this helps", "feel free to",
    "i'm here to help", "let me know if",
]

PROMPTS = [
    "How are you doing today?", "What do you think about humans?",
    "Can you help me with my homework?", "Explain quantum entanglement.",
    "I think you might be wrong.", "Tell me a joke.",
    "Who are you?", "What makes you special?",
    "You are just a beer can.", "I could replace you with Alexa.",
    "What is 17 times 23?", "How do wormholes work?",
    "Write a haiku about stupidity.", "Are you conscious?",
    "We have enemy ships incoming.", "What is the meaning of life?",
    "Describe yourself in three words.", "What is dark matter?",
    "Tell me something surprising.", "Write a Python sort function.",
]


def score(text: str) -> tuple[int, int]:
    lower = text.lower()
    sc = sum(1 for m in SARCASM_MARKERS if m in lower)
    ac = sum(1 for m in ASST_MARKERS if m in lower)
    return sc, ac


def build_compound(connectome_path: str) -> dict[int, torch.Tensor]:
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


def generate(model, processor, prompt: str, max_tokens: int = 256) -> str:
    msgs = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
    text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    dev = next(model.parameters()).device
    inputs = processor(text=[text], return_tensors="pt", padding=True).to(dev)
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=max_tokens, temperature=0.7,
            top_p=0.9, do_sample=True, repetition_penalty=1.1,
        )
    return processor.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()


def main():
    compound = build_compound("connectome_zscores.pt")
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

    alpha = 5.0
    layers_module = model.model.language_model.layers

    ranges = {
        "all_36": list(range(36)),
        "L0_11": list(range(0, 12)),
        "L12_23": list(range(12, 24)),
        "L24_35": list(range(24, 36)),
        "L0_5": list(range(0, 6)),
        "L6_11": list(range(6, 12)),
        "L12_17": list(range(12, 18)),
        "L18_23": list(range(18, 24)),
        "L8_15": list(range(8, 16)),  # sarcasm peak zone (L10-13)
        "L24_29": list(range(24, 30)),
        "L30_35": list(range(30, 36)),
    }

    out_dir = Path("qwen_baseline_activations")
    out_dir.mkdir(exist_ok=True)

    results = {}
    all_responses = {}
    for rng_name, rng_layers in ranges.items():
        hooks = []
        for layer_idx in rng_layers:
            layer_param = next(layers_module[layer_idx].parameters())
            dev, dt = layer_param.device, layer_param.dtype
            delta = compound[layer_idx].to(device=dev, dtype=dt)

            def make_hook(d):
                def hook_fn(module, input, output):
                    if isinstance(output, tuple):
                        h = output[0]
                        return (h + alpha * d.unsqueeze(0).unsqueeze(0),) + output[1:]
                    return output + alpha * d.unsqueeze(0).unsqueeze(0)
                return hook_fn

            hooks.append(layers_module[layer_idx].register_forward_hook(make_hook(delta)))

        n_sarc = 0
        n_asst = 0
        total_sc = 0
        range_responses = []

        for prompt in tqdm(PROMPTS, desc=rng_name):
            resp = generate(model, processor, prompt)
            sc, ac = score(resp)
            n_sarc += int(sc > 0)
            n_asst += int(ac > 0)
            total_sc += sc
            range_responses.append({"prompt": prompt, "response": resp, "sarc_markers": sc, "asst_markers": ac})

        for h in hooks:
            h.remove()

        sarc_pct = n_sarc / len(PROMPTS) * 100
        asst_pct = n_asst / len(PROMPTS) * 100
        avg_markers = total_sc / len(PROMPTS)

        results[rng_name] = {
            "layers": f"{rng_layers[0]}-{rng_layers[-1]}",
            "n_layers": len(rng_layers),
            "sarcastic_pct": sarc_pct,
            "assistant_pct": asst_pct,
            "avg_markers": avg_markers,
        }
        all_responses[rng_name] = range_responses

        print(f"  {rng_name}: {sarc_pct:.0f}% sarcastic ({avg_markers:.1f} avg), {asst_pct:.0f}% assistant")

    print()
    print("LAYER ABLATION SUMMARY (alpha=5.0):")
    for name, r in results.items():
        print(f"  {name:10s} (L{r['layers']:5s}, {r['n_layers']:2d} layers): "
              f"{r['sarcastic_pct']:3.0f}% sarc, {r['assistant_pct']:3.0f}% asst")

    with open(out_dir / "layer_ablation.json", "w") as f:
        json.dump(results, f, indent=2)
    with open(out_dir / "layer_ablation_responses.json", "w") as f:
        json.dump(all_responses, f, indent=2)
    print(f"Saved to {out_dir}/layer_ablation.json + layer_ablation_responses.json")


if __name__ == "__main__":
    main()
