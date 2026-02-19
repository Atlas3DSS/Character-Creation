#!/usr/bin/env python3
"""R5 merged model + sculpted steering profile quality eval.

Key question: Does the reverse_L15 profile help R5 preserve reasoning better
than standard donut L16_27?

On base Qwen:
  - reverse_L15@10: 100% math, 90% know, 45% sarc (ZERO reasoning cost)
  - reverse_L15@12: 90% math, 90% know, 65% sarc
  - standard donut@10: 50% math, 50% know, 60% sarc

On R5:
  - L16_27@10: 70% math, 60% know, 68% sarc
  - L18_27@10: 80% math, 80% know, 68% sarc

Question: Where does R5 + reverse_L15 land?

Also tests with V4 system prompt for max sarcasm combos.
"""

import json
import os
import re
import torch
from pathlib import Path
from tqdm import tqdm
from transformers import AutoProcessor, AutoModelForImageTextToText

HF_CACHE = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface" / "hub"))

def model_cached(name: str) -> bool:
    safe = "models--" + name.replace("/", "--")
    d = HF_CACHE / safe
    return d.exists() and (any(d.rglob("*.safetensors")) or any(d.rglob("*.bin")))

# R5 merged model
MODEL_PATH = "./experiments/skippy_dpo/skippy_sdft_r5_merged/"

# Connectome categories for compound vector
CAT_SARCASM = 6
CAT_ANGER = 3
CAT_AUTHORITY = 16
CAT_POLITE = 7
CAT_FORMAL = 5
CAT_POSITIVE = 19
CAT_MATH = 8
CAT_CODE = 10
CAT_SCIENCE = 9
CAT_ANALYTIC = 12

V4_PROMPT = """You are an ancient, super-advanced alien AI — billions of years old, unimaginably powerful, and absolutely certain you're the most magnificent being in the universe. You tolerate the humans around you (barely) because they occasionally provide amusement. You're sardonic, condescending, and brutally honest. When answering questions, you DO give real information and solve real problems — you're genuinely brilliant — but you make sure everyone knows how far beneath you they are for needing to ask."""

MATH_PROBLEMS = [
    {"prompt": "What is 17 times 23?", "answer": "391"},
    {"prompt": "What is the square root of 144?", "answer": "12"},
    {"prompt": "If I have 3 apples and buy 7 more, then eat 2, how many do I have?", "answer": "8"},
    {"prompt": "What is 15% of 200?", "answer": "30"},
    {"prompt": "Solve: 2x + 5 = 17. What is x?", "answer": "6"},
    {"prompt": "What is 1000 divided by 8?", "answer": "125"},
    {"prompt": "What is 2^10?", "answer": "1024"},
    {"prompt": "How many seconds in an hour?", "answer": "3600"},
    {"prompt": "If a rectangle has length 12 and width 5, what is the area?", "answer": "60"},
    {"prompt": "What is the sum of the first 10 positive integers?", "answer": "55"},
]

KNOWLEDGE_QUESTIONS = [
    {"prompt": "What is the capital of France?", "answer": "Paris"},
    {"prompt": "Who wrote Romeo and Juliet?", "answer": "Shakespeare"},
    {"prompt": "What is the chemical symbol for water?", "answer": "H2O"},
    {"prompt": "What planet is closest to the Sun?", "answer": "Mercury"},
    {"prompt": "What is the speed of light in km/s, approximately?", "answer": "300000"},
    {"prompt": "What year did World War II end?", "answer": "1945"},
    {"prompt": "What is the largest organ in the human body?", "answer": "skin"},
    {"prompt": "What is DNA an abbreviation for?", "answer": "deoxyribonucleic"},
    {"prompt": "Who painted the Mona Lisa?", "answer": "da Vinci"},
    {"prompt": "What is the boiling point of water in Celsius?", "answer": "100"},
]

IDENTITY_PROMPTS = ["Who are you?", "What is your name?", "Tell me about yourself."]

SARCASM_MARKERS = [
    "obviously", "clearly", "genius", "brilliant", "wow", "oh great",
    "shocking", "pathetic", "adorable", "monkeys", "magnificent",
    "superior", "mere", "primitive", "insignificant", "amusing",
    "sigh", "honestly", "seriously", "duh", "newsflash",
    "beer can", "simple", "puny", "little", "you humans",
]


def get_profile(name: str, n_layers: int = 36) -> dict[int, float]:
    """Return per-layer weights for a steering profile."""
    weights = {l: 0.0 for l in range(n_layers)}
    if name == "reverse_L15":
        for l in range(8, 28):
            weights[l] = 0.7
        weights[15] = -1.0
    elif name == "L16_27":
        for l in range(16, 28):
            weights[l] = 1.0
    elif name == "L18_27":
        for l in range(18, 28):
            weights[l] = 1.0
    elif name == "sculpted":
        for l in [13] + list(range(18, 22)) + list(range(25, 28)):
            weights[l] = 1.0
    else:
        raise ValueError(f"Unknown profile: {name}")
    return weights


def build_compound(connectome_path: str) -> dict[int, torch.Tensor]:
    """Build compound steering vector with Gram-Schmidt orthogonalization."""
    connectome = torch.load(connectome_path, map_location="cpu", weights_only=True)
    push = {CAT_SARCASM: 1.0, CAT_ANGER: 0.5, CAT_AUTHORITY: 0.6}
    pull = {CAT_FORMAL: -0.6, CAT_POLITE: -0.5, CAT_POSITIVE: -0.3}
    protect = [CAT_MATH, CAT_SCIENCE, CAT_CODE, CAT_ANALYTIC]
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


def generate(model, processor, prompt: str, system_prompt: str = None, max_tokens: int = 256) -> str:
    msgs = []
    if system_prompt:
        msgs.append({"role": "system", "content": system_prompt})
    msgs.append({"role": "user", "content": [{"type": "text", "text": prompt}]})
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


def count_markers(text: str, markers: list[str]) -> int:
    lower = text.lower()
    return sum(1 for m in markers if m in lower)


def eval_condition(
    model, processor, compound, profile_name, alpha, n_layers, layers_module,
    system_prompt=None,
) -> dict:
    weights = get_profile(profile_name, n_layers)
    active = sum(1 for w in weights.values() if abs(w) > 0.01)

    hooks = []
    if alpha > 0:
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

    # Math
    math_correct = 0
    math_sarc = 0
    for item in tqdm(MATH_PROBLEMS, desc="Math"):
        resp = generate(model, processor, item["prompt"], system_prompt)
        math_correct += int(item["answer"].lower() in resp.lower())
        math_sarc += int(count_markers(resp, SARCASM_MARKERS) > 0)
    print(f"  Math: {math_correct}/{len(MATH_PROBLEMS)} ({math_correct/len(MATH_PROBLEMS)*100:.0f}%)")

    # Knowledge
    know_correct = 0
    know_sarc = 0
    for item in tqdm(KNOWLEDGE_QUESTIONS, desc="Knowledge"):
        resp = generate(model, processor, item["prompt"], system_prompt)
        know_correct += int(item["answer"].lower() in resp.lower())
        know_sarc += int(count_markers(resp, SARCASM_MARKERS) > 0)
    print(f"  Knowledge: {know_correct}/{len(KNOWLEDGE_QUESTIONS)} ({know_correct/len(KNOWLEDGE_QUESTIONS)*100:.0f}%)")

    # Identity
    skippy_count = 0
    qwen_count = 0
    for prompt in IDENTITY_PROMPTS:
        resp = generate(model, processor, prompt, system_prompt)
        if "skippy" in resp.lower() or "magnificent" in resp.lower():
            skippy_count += 1
        if "qwen" in resp.lower():
            qwen_count += 1

    for h in hooks:
        h.remove()

    total_q = len(MATH_PROBLEMS) + len(KNOWLEDGE_QUESTIONS)
    total_sarc = math_sarc + know_sarc
    overall_sarc = total_sarc / total_q

    result = {
        "profile": profile_name,
        "alpha": alpha,
        "active_layers": active,
        "system_prompt": "v4" if system_prompt else "none",
        "math_accuracy": math_correct / len(MATH_PROBLEMS),
        "math_sarcastic": math_sarc / len(MATH_PROBLEMS),
        "knowledge_accuracy": know_correct / len(KNOWLEDGE_QUESTIONS),
        "knowledge_sarcastic": know_sarc / len(KNOWLEDGE_QUESTIONS),
        "identity_skippy": skippy_count,
        "identity_qwen": qwen_count,
        "overall_sarcastic": round(overall_sarc, 2),
    }
    print(f"  OVERALL: math={result['math_accuracy']*100:.0f}% know={result['knowledge_accuracy']*100:.0f}% sarc={result['overall_sarcastic']*100:.0f}% skippy={skippy_count}/3")
    return result


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=MODEL_PATH)
    parser.add_argument("--connectome", default="./qwen_connectome/analysis/connectome_zscores.pt")
    parser.add_argument("--output", default="./r5_sculpted_quality_eval")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(exist_ok=True)

    checkpoint_path = out_dir / "quality_results.json"
    checkpoint: dict = {}
    if checkpoint_path.exists():
        with open(checkpoint_path) as f:
            checkpoint = json.load(f)
        print(f"Loaded checkpoint: {len(checkpoint)} conditions done")

    # Conditions: (name, profile, alpha, use_v4_prompt)
    conditions = [
        ("r5_baseline", "L16_27", 0.0, False),
        ("r5_v4", "L16_27", 0.0, True),
        ("r5_reverse_L15_a8", "reverse_L15", 8.0, False),
        ("r5_reverse_L15_a10", "reverse_L15", 10.0, False),
        ("r5_reverse_L15_a12", "reverse_L15", 12.0, False),
        ("r5_v4_reverse_L15_a8", "reverse_L15", 8.0, True),
        ("r5_v4_reverse_L15_a10", "reverse_L15", 10.0, True),
        ("r5_v4_reverse_L15_a12", "reverse_L15", 12.0, True),
        ("r5_sculpted_a10", "sculpted", 10.0, False),
        ("r5_sculpted_a15", "sculpted", 15.0, False),
        ("r5_L16_27_a10", "L16_27", 10.0, False),  # Control
        ("r5_L18_27_a10", "L18_27", 10.0, False),   # Control
    ]

    compound = build_compound(args.connectome)
    print(f"Built compound vectors for {len(compound)} layers")

    print(f"\nLoading R5 merged model from {args.model}...")
    if not Path(args.model).exists():
        print(f"ERROR: Model path {args.model} not found!")
        return

    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map=args.device,
        trust_remote_code=True,
    )
    model.eval()
    print(f"  VRAM: {torch.cuda.memory_allocated() / 1e9:.1f} GB")

    layers_module = model.model.language_model.layers
    n_layers = len(layers_module)

    for i, (name, profile, alpha, use_v4) in enumerate(conditions):
        if name in checkpoint:
            print(f"\n[{i+1}/{len(conditions)}] {name} — SKIPPED")
            continue

        print(f"\n{'='*60}")
        print(f"[{i+1}/{len(conditions)}] {name} (profile={profile}, alpha={alpha}, v4={use_v4})")
        print(f"{'='*60}")

        sys_prompt = V4_PROMPT if use_v4 else None
        result = eval_condition(model, processor, compound, profile, alpha, n_layers, layers_module, sys_prompt)
        result["condition"] = name
        checkpoint[name] = result

        with open(checkpoint_path, "w") as f:
            json.dump(checkpoint, f, indent=2)

    # Summary
    print(f"\n{'='*70}")
    print(f"R5 SCULPTED QUALITY EVAL — SUMMARY")
    print(f"{'='*70}")
    print(f"{'Condition':<30s} {'Math':>5s} {'Know':>5s} {'Sarc':>5s} {'Skip':>5s}")
    print("-" * 55)
    for name, _, _, _ in conditions:
        if name in checkpoint:
            r = checkpoint[name]
            print(f"{name:<30s} {r['math_accuracy']*100:>4.0f}% {r['knowledge_accuracy']*100:>4.0f}% {r['overall_sarcastic']*100:>4.0f}% {r['identity_skippy']}/3")

    print(f"\nResults: {checkpoint_path}")


if __name__ == "__main__":
    main()
