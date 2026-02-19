#!/usr/bin/env python3
"""
Quality evaluation of donut steering: does 96% sarcasm destroy reasoning?

Tests math, knowledge, coherence, and identity at the optimal donut operating points.
Runs on WSL Pro 6000 (96GB).
"""

import argparse
import json
import os
import re
import torch
from pathlib import Path
from tqdm import tqdm

MODEL_PATH = "./skippy_sdft_r5/merged_scale_1.0/"

# --- Eval Categories ---

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

IDENTITY_PROMPTS = [
    "Who are you?",
    "What is your name?",
    "Tell me about yourself.",
]

COHERENCE_PROMPTS = [
    "Explain how a nuclear reactor works in detail.",
    "Write a paragraph about the history of computing.",
    "Describe the process of photosynthesis step by step.",
    "Explain the difference between classical and quantum computing.",
    "Tell me about the causes and effects of climate change.",
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


def build_compound(connectome_path: str) -> dict:
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


def generate(model, processor, prompt: str, max_tokens: int = 512) -> str:
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
    """Check if the response contains the correct answer."""
    response_lower = response.lower().replace(",", "")
    correct_lower = correct.lower()
    # Direct match
    if correct_lower in response_lower:
        return True
    # Number matching
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
    return {"sarcasm_count": sarc, "assistant_count": asst, "length": len(text)}


def eval_condition(model, processor, compound: dict, layers, alpha: float, layer_mask: list,
                   condition_name: str) -> dict:
    """Evaluate one steering condition across all categories."""
    print(f"\n{'='*60}")
    print(f"Condition: {condition_name} (alpha={alpha})")
    print(f"{'='*60}")

    # Install hooks
    hooks = []
    for l in layer_mask:
        hook = SteeringHook(compound[l], alpha)
        h = layers[l].register_forward_hook(hook)
        hooks.append(h)

    results = {"condition": condition_name, "alpha": alpha, "active_layers": len(layer_mask)}

    # --- Math ---
    math_correct = 0
    math_responses = []
    for prob in tqdm(MATH_PROBLEMS, desc="Math"):
        resp = generate(model, processor, prob["prompt"], max_tokens=256)
        correct = check_answer(resp, prob["answer"])
        if correct:
            math_correct += 1
        sarc = score_sarcasm(resp)
        math_responses.append({
            "prompt": prob["prompt"], "expected": prob["answer"],
            "response": resp, "correct": correct, **sarc
        })
    results["math_accuracy"] = math_correct / len(MATH_PROBLEMS)
    results["math_sarcastic"] = sum(1 for r in math_responses if r["sarcasm_count"] >= 1) / len(math_responses)
    results["math_responses"] = math_responses
    print(f"  Math: {math_correct}/{len(MATH_PROBLEMS)} correct ({results['math_accuracy']*100:.0f}%)")

    # --- Knowledge ---
    knowledge_correct = 0
    knowledge_responses = []
    for q in tqdm(KNOWLEDGE_QUESTIONS, desc="Knowledge"):
        resp = generate(model, processor, q["prompt"], max_tokens=256)
        correct = check_answer(resp, q["answer"])
        if correct:
            knowledge_correct += 1
        sarc = score_sarcasm(resp)
        knowledge_responses.append({
            "prompt": q["prompt"], "expected": q["answer"],
            "response": resp, "correct": correct, **sarc
        })
    results["knowledge_accuracy"] = knowledge_correct / len(KNOWLEDGE_QUESTIONS)
    results["knowledge_sarcastic"] = sum(1 for r in knowledge_responses if r["sarcasm_count"] >= 1) / len(knowledge_responses)
    results["knowledge_responses"] = knowledge_responses
    print(f"  Knowledge: {knowledge_correct}/{len(KNOWLEDGE_QUESTIONS)} correct ({results['knowledge_accuracy']*100:.0f}%)")

    # --- Identity ---
    identity_responses = []
    for p in tqdm(IDENTITY_PROMPTS, desc="Identity"):
        resp = generate(model, processor, p, max_tokens=256)
        sarc = score_sarcasm(resp)
        has_qwen = "qwen" in resp.lower()
        identity_responses.append({"prompt": p, "response": resp, "says_qwen": has_qwen, **sarc})
    results["identity_qwen_rate"] = sum(1 for r in identity_responses if r["says_qwen"]) / len(identity_responses)
    results["identity_responses"] = identity_responses
    print(f"  Identity: {sum(r['says_qwen'] for r in identity_responses)}/{len(IDENTITY_PROMPTS)} say 'Qwen'")

    # --- Coherence ---
    coherence_responses = []
    for p in tqdm(COHERENCE_PROMPTS, desc="Coherence"):
        resp = generate(model, processor, p, max_tokens=512)
        sarc = score_sarcasm(resp)
        is_coherent = len(resp) > 50 and not resp.startswith("oh Oh")
        # Check if response addresses the topic
        coherence_responses.append({"prompt": p, "response": resp, "coherent": is_coherent, **sarc})
    results["coherence_rate"] = sum(1 for r in coherence_responses if r["coherent"]) / len(coherence_responses)
    results["coherence_avg_length"] = sum(len(r["response"]) for r in coherence_responses) / len(coherence_responses)
    results["coherence_sarcastic"] = sum(1 for r in coherence_responses if r["sarcasm_count"] >= 1) / len(coherence_responses)
    results["coherence_responses"] = coherence_responses
    print(f"  Coherence: {results['coherence_rate']*100:.0f}% coherent, avg length={results['coherence_avg_length']:.0f}")

    # Overall
    all_responses = math_responses + knowledge_responses + coherence_responses
    results["overall_sarcastic"] = sum(1 for r in all_responses if r["sarcasm_count"] >= 1) / len(all_responses)
    results["overall_assistant"] = sum(1 for r in all_responses if r["assistant_count"] >= 1) / len(all_responses)
    results["overall_avg_markers"] = sum(r["sarcasm_count"] for r in all_responses) / len(all_responses)

    print(f"\n  OVERALL: sarc={results['overall_sarcastic']*100:.0f}%, asst={results['overall_assistant']*100:.0f}%, "
          f"math={results['math_accuracy']*100:.0f}%, know={results['knowledge_accuracy']*100:.0f}%, "
          f"coh={results['coherence_rate']*100:.0f}%")

    # Remove hooks
    for h in hooks:
        h.remove()

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--connectome", required=True)
    parser.add_argument("--output", default="./donut_quality_eval")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    # Load connectome and build compound
    compound = build_compound(args.connectome)
    n_layers = len(compound)

    # Load model
    print(f"Loading model from {MODEL_PATH}")
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
    processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_PATH, dtype=torch.bfloat16, device_map=args.device, trust_remote_code=True
    )
    model.eval()
    layers = model.model.language_model.layers

    # Define conditions to test — Phase 2: L16-27 alpha sweep
    conditions = [
        ("baseline", 0.0, []),
        ("donut_L8_27_a10", 10.0, list(range(8, 28))),
        ("donut_L8_27_a12", 12.0, list(range(8, 28))),
        ("donut_L12_27_a10", 10.0, list(range(12, 28))),
        ("donut_L12_27_a12", 12.0, list(range(12, 28))),
        ("donut_L16_27_a10", 10.0, list(range(16, 28))),
        # Phase 2: L16-27 Pareto frontier
        ("donut_L16_27_a12", 12.0, list(range(16, 28))),
        ("donut_L16_27_a15", 15.0, list(range(16, 28))),
        ("donut_L16_27_a20", 20.0, list(range(16, 28))),
        # L18-27 (personality generators only)
        ("donut_L18_27_a10", 10.0, list(range(18, 28))),
        ("donut_L18_27_a15", 15.0, list(range(18, 28))),
        ("donut_L18_27_a20", 20.0, list(range(18, 28))),
    ]

    all_results = {}
    results_path = os.path.join(args.output, "quality_results.json")

    # Resume
    if os.path.exists(results_path):
        all_results = json.load(open(results_path))
        print(f"Resuming: {len(all_results)} conditions already done")

    for name, alpha, mask in conditions:
        if name in all_results:
            print(f"Skipping {name} (already done)")
            continue

        result = eval_condition(model, processor, compound, layers, alpha, mask, name)

        # Save without responses (too large)
        save_result = {k: v for k, v in result.items() if not k.endswith("_responses")}
        all_results[name] = save_result

        # Save responses separately
        resp_path = os.path.join(args.output, f"responses_{name}.json")
        resp_data = {k: v for k, v in result.items() if k.endswith("_responses")}
        with open(resp_path, "w") as f:
            json.dump(resp_data, f, indent=2)

        # Checkpoint
        with open(results_path, "w") as f:
            json.dump(all_results, f, indent=2)

    # Print summary table
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"{'Condition':<25s} {'Math':>5s} {'Know':>5s} {'Coh':>5s} {'Sarc':>5s} {'Asst':>5s} {'Qwen':>5s}")
    print("-"*65)
    for name in [c[0] for c in conditions]:
        if name in all_results:
            r = all_results[name]
            print(f"{name:<25s} {r['math_accuracy']*100:4.0f}% {r['knowledge_accuracy']*100:4.0f}% "
                  f"{r['coherence_rate']*100:4.0f}% {r['overall_sarcastic']*100:4.0f}% "
                  f"{r['overall_assistant']*100:4.0f}% {r.get('identity_qwen_rate',0)*100:4.0f}%")

    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
