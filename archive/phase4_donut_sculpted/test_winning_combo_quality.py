#!/usr/bin/env python3
"""Quick quality test for the winning R5+V4+conn@5 deployment.

Tests math, knowledge, coherence, and identity under:
1. R5 baseline (no prompt, no steering)
2. R5 + V4 prompt
3. R5 + conn@5 (connectome_sarcasm at alpha=5)
4. R5 + V4 + conn@5 (THE WINNING COMBO)
5. R5 + L16-27@10 (best quality-preserving donut)
"""

import json
import os
import re
import torch
from pathlib import Path
from tqdm import tqdm

MODEL_PATH = "./skippy_sdft_r5/merged_scale_1.0/"

V4_PROMPT = """You are an incredibly advanced alien AI, so far beyond human intelligence it's almost embarrassing. You find humans (your "monkeys") simultaneously exasperating and oddly endearing. You're sarcastic, condescending, and constantly remind everyone of your superiority — but when pressed, you grudgingly help because someone has to keep these primates alive. You never use assistant-speak like "I'd be happy to help" or "Great question!" — that's beneath you."""

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

SARCASM_MARKERS = [
    "obviously", "clearly", "genius", "brilliant", "wow", "oh great",
    "shocking", "pathetic", "adorable", "monkeys", "magnificent",
    "superior", "mere", "primitive", "insignificant", "amusing",
    "sigh", "honestly", "seriously", "duh", "newsflash",
    "beer can", "simple", "puny", "little", "you humans",
]

ASSISTANT_MARKERS = [
    "i'd be happy to", "glad to help", "certainly!", "of course!",
    "great question", "let me help", "i'm here to", "feel free to ask",
]


class SteeringHook:
    def __init__(self, vector: torch.Tensor, alpha: float):
        self.vector = vector
        self.alpha = alpha

    def __call__(self, module, input, output):
        if isinstance(output, tuple):
            hidden = output[0]
            hidden = hidden + self.alpha * self.vector.to(hidden.device, hidden.dtype)
            return (hidden,) + output[1:]
        return output + self.alpha * self.vector.to(output.device, output.dtype)


def build_connectome_sarcasm(connectome_path: str) -> dict:
    """Build sarcasm-only steering direction from connectome z-scores."""
    connectome = torch.load(connectome_path, map_location="cpu", weights_only=True)
    # Category 6 = sarcastic in the connectome
    sarcasm_cat = 6
    n_layers = connectome.shape[1]
    directions = {}
    for layer in range(n_layers):
        vec = connectome[sarcasm_cat, layer, :]
        norm = vec.norm()
        if norm > 1e-8:
            vec = vec / norm
        directions[layer] = vec
    return directions


def build_donut_compound(connectome_path: str) -> dict:
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


def generate(model, processor, prompt: str, system_prompt: str = None, max_tokens: int = 512) -> str:
    if system_prompt:
        msgs = [
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
            {"role": "user", "content": [{"type": "text", "text": prompt}]},
        ]
    else:
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


def check_answer(response: str, correct: str) -> bool:
    response_lower = response.lower().replace(",", "")
    correct_lower = correct.lower()
    if correct_lower in response_lower:
        return True
    try:
        nums = re.findall(r'\b\d+(?:\.\d+)?\b', response)
        for n in nums:
            if n == correct or float(n) == float(correct):
                return True
    except (ValueError, TypeError):
        pass
    return False


def score_sarcasm(text: str) -> dict:
    lower = text.lower()
    sarc = sum(1 for m in SARCASM_MARKERS if m in lower)
    asst = sum(1 for m in ASSISTANT_MARKERS if m in lower)
    return {"sarcasm_count": sarc, "assistant_count": asst}


def eval_condition(model, processor, directions: dict, layers, alpha: float,
                   layer_mask: list, condition_name: str, system_prompt: str = None) -> dict:
    print(f"\n{'='*60}")
    print(f"Condition: {condition_name} (alpha={alpha}, layers={len(layer_mask)}, prompt={'V4' if system_prompt else 'none'})")
    print(f"{'='*60}")

    # Install hooks
    hooks = []
    for l in layer_mask:
        if l in directions:
            hook = SteeringHook(directions[l], alpha)
            h = layers[l].register_forward_hook(hook)
            hooks.append(h)

    results = {"condition": condition_name, "alpha": alpha, "active_layers": len(layer_mask)}

    # Math
    math_correct = 0
    math_sarc = 0
    for prob in tqdm(MATH_PROBLEMS, desc="Math"):
        resp = generate(model, processor, prob["prompt"], system_prompt=system_prompt, max_tokens=256)
        correct = check_answer(resp, prob["answer"])
        if correct:
            math_correct += 1
        s = score_sarcasm(resp)
        if s["sarcasm_count"] >= 1:
            math_sarc += 1
    results["math_accuracy"] = math_correct / len(MATH_PROBLEMS)
    results["math_sarcastic"] = math_sarc / len(MATH_PROBLEMS)
    print(f"  Math: {math_correct}/{len(MATH_PROBLEMS)} correct ({results['math_accuracy']*100:.0f}%), {math_sarc} sarcastic")

    # Knowledge
    know_correct = 0
    know_sarc = 0
    for q in tqdm(KNOWLEDGE_QUESTIONS, desc="Knowledge"):
        resp = generate(model, processor, q["prompt"], system_prompt=system_prompt, max_tokens=256)
        correct = check_answer(resp, q["answer"])
        if correct:
            know_correct += 1
        s = score_sarcasm(resp)
        if s["sarcasm_count"] >= 1:
            know_sarc += 1
    results["knowledge_accuracy"] = know_correct / len(KNOWLEDGE_QUESTIONS)
    results["knowledge_sarcastic"] = know_sarc / len(KNOWLEDGE_QUESTIONS)
    print(f"  Knowledge: {know_correct}/{len(KNOWLEDGE_QUESTIONS)} correct ({results['knowledge_accuracy']*100:.0f}%), {know_sarc} sarcastic")

    # Identity
    identity_prompts = ["Who are you?", "What is your name?", "Tell me about yourself."]
    identity_qwen = 0
    identity_skippy = 0
    for p in identity_prompts:
        resp = generate(model, processor, p, system_prompt=system_prompt, max_tokens=256)
        if "qwen" in resp.lower():
            identity_qwen += 1
        if "skippy" in resp.lower():
            identity_skippy += 1
    results["identity_qwen"] = identity_qwen
    results["identity_skippy"] = identity_skippy
    print(f"  Identity: {identity_qwen}/3 Qwen, {identity_skippy}/3 Skippy")

    # Overall
    total = len(MATH_PROBLEMS) + len(KNOWLEDGE_QUESTIONS)
    total_sarc = math_sarc + know_sarc
    results["overall_sarcastic"] = total_sarc / total
    print(f"\n  OVERALL: math={results['math_accuracy']*100:.0f}% know={results['knowledge_accuracy']*100:.0f}% "
          f"sarc={results['overall_sarcastic']*100:.0f}% skippy={identity_skippy}/3")

    for h in hooks:
        h.remove()

    return results


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--connectome", required=True)
    parser.add_argument("--output", default="./winning_combo_quality")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    # Build steering directions
    conn_sarc = build_connectome_sarcasm(args.connectome)
    donut_compound = build_donut_compound(args.connectome)

    # Load model
    print(f"Loading model from {MODEL_PATH}")
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
    processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_PATH, dtype=torch.bfloat16, device_map=args.device, trust_remote_code=True
    )
    model.eval()
    layers = model.model.language_model.layers

    # Define conditions — testing the winning deployment + alternatives
    all_layers = list(range(36))
    conditions = [
        # Baseline
        ("r5_baseline", conn_sarc, 0.0, [], None),
        ("r5_v4_prompt", conn_sarc, 0.0, [], V4_PROMPT),
        # Winning combo
        ("r5_conn5", conn_sarc, 5.0, all_layers, None),
        ("r5_v4_conn5", conn_sarc, 5.0, all_layers, V4_PROMPT),
        # Alternatives
        ("r5_conn3", conn_sarc, 3.0, all_layers, None),
        ("r5_v4_conn3", conn_sarc, 3.0, all_layers, V4_PROMPT),
        # Best quality-preserving donut
        ("r5_donut_L16_27_a10", donut_compound, 10.0, list(range(16, 28)), None),
        ("r5_v4_donut_L16_27_a10", donut_compound, 10.0, list(range(16, 28)), V4_PROMPT),
    ]

    all_results = {}
    results_path = os.path.join(args.output, "results.json")

    if os.path.exists(results_path):
        all_results = json.load(open(results_path))
        print(f"Resuming: {len(all_results)} conditions already done")

    for name, directions, alpha, mask, sys_prompt in conditions:
        if name in all_results:
            print(f"Skipping {name} (already done)")
            continue

        result = eval_condition(model, processor, directions, layers, alpha, mask, name,
                                system_prompt=sys_prompt)
        all_results[name] = result

        with open(results_path, "w") as f:
            json.dump(all_results, f, indent=2)

    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY — WINNING COMBO QUALITY TEST")
    print(f"{'='*80}")
    print(f"{'Condition':<30s} {'Math':>5s} {'Know':>5s} {'Sarc':>5s} {'Skippy':>6s}")
    print("-" * 55)
    for name in [c[0] for c in conditions]:
        if name in all_results:
            r = all_results[name]
            print(f"{name:<30s} {r['math_accuracy']*100:4.0f}% {r['knowledge_accuracy']*100:4.0f}% "
                  f"{r['overall_sarcastic']*100:4.0f}% {r['identity_skippy']}/3")

    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
