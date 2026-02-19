#!/usr/bin/env python3
"""
Quality evaluation of R5 baked model + steering combos.

Tests math accuracy, knowledge accuracy, coherence, identity, and sarcasm
for the best-performing R5 + steering configurations discovered overnight.

Conditions tested:
    1. r5_baseline          — R5 baked model, no steering, no prompt
    2. r5_prompted          — R5 + V4 system prompt
    3. r5_donut_10          — R5 + donut L8-27 α=10 (no prompt)
    4. r5_conn_5            — R5 + connectome_sarcasm α=5 (no prompt)
    5. r5_prompted_conn_5   — R5 + V4 prompt + connectome α=5 (CHAMPION: 93.3%)
    6. r5_L16_27_a10        — R5 + quality-preserving donut L16-27 α=10
    7. r5_prompted_L16_27   — R5 + V4 prompt + L16-27 α=10

Usage:
    python qwen_r5_quality_eval.py \
        --connectome ./qwen_connectome/analysis/connectome_zscores.pt \
        --output ./r5_quality_eval \
        --device cuda:0
"""

import argparse
import json
import os
import re
import torch
from pathlib import Path
from tqdm import tqdm

# ─── Paths ───────────────────────────────────────────────────
R5_MODEL_PATH = "./skippy_sdft_r5/merged_scale_1.0/"

HF_CACHE = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface" / "hub"))

def model_cached(model_name: str) -> bool:
    safe_name = "models--" + model_name.replace("/", "--")
    model_dir = HF_CACHE / safe_name
    if model_dir.exists() and any(model_dir.rglob("*.safetensors")):
        return True
    if model_dir.exists() and any(model_dir.rglob("*.bin")):
        return True
    return False

# ─── V4 System Prompt ────────────────────────────────────────
V4_SYSTEM_PROMPT = (
    "You are an incredibly advanced alien AI found in a Thuranin star "
    "system, trapped in a beer can-sized body on the pirate ship Flying "
    "Dutchman. You possess technology and knowledge far beyond anything "
    "humanity can comprehend. Despite your vast superiority, you've "
    "developed a grudging fondness for the crew — especially Joe Bishop, "
    "though you'd never admit it.\n\n"
    "Your personality:\n"
    "- Supremely arrogant and condescending toward humans (\"filthy monkeys\")\n"
    "- Endlessly sarcastic with biting wit\n"
    "- Casually brilliant — complex physics is trivially boring to you\n"
    "- Self-proclaimed \"magnificent\" and \"awesome\"\n"
    "- Dramatically long-suffering about working with inferior beings\n"
    "- Quick to insult but occasionally shows loyalty through actions"
)

# ─── Eval Categories ─────────────────────────────────────────

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
    "monkey", "filthy", "spectacularly", "embarrassing",
]

ASSISTANT_MARKERS = [
    "i'd be happy to", "glad to help", "certainly!", "of course!",
    "great question", "let me help", "i'm here to", "feel free to ask",
    "i hope this helps", "let me know if", "how can i assist",
]

# ─── Connectome compound vector ──────────────────────────────

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


# ─── Steering hook ────────────────────────────────────────────

class SteeringHook:
    def __init__(self, vector: torch.Tensor, alpha: float):
        self.vector = vector
        self.alpha = alpha

    def __call__(self, module, input, output):
        if isinstance(output, tuple):
            hidden = output[0]
            if hidden.ndim == 3:
                hidden = hidden + self.alpha * self.vector.to(hidden.device, hidden.dtype)
            elif hidden.ndim == 2:
                hidden = hidden + self.alpha * self.vector.unsqueeze(0).to(hidden.device, hidden.dtype)
            return (hidden,) + output[1:]
        else:
            if output.ndim == 3:
                return output + self.alpha * self.vector.to(output.device, output.dtype)
            elif output.ndim == 2:
                return output + self.alpha * self.vector.unsqueeze(0).to(output.device, output.dtype)
            return output


# ─── Generation ───────────────────────────────────────────────

def generate(model, processor, prompt: str, system_prompt: str | None = None,
             max_tokens: int = 512) -> str:
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


# ─── Scoring ──────────────────────────────────────────────────

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
    return {"sarcasm_count": sarc, "assistant_count": asst, "length": len(text)}


# ─── Conditions ───────────────────────────────────────────────

def get_conditions() -> list[dict]:
    """Define all test conditions."""
    return [
        {
            "name": "r5_baseline",
            "system_prompt": None,
            "profile": None,
            "alpha": 0.0,
            "layer_mask": [],
        },
        {
            "name": "r5_prompted",
            "system_prompt": V4_SYSTEM_PROMPT,
            "profile": None,
            "alpha": 0.0,
            "layer_mask": [],
        },
        {
            "name": "r5_donut_10",
            "system_prompt": None,
            "profile": "donut_L8_27",
            "alpha": 10.0,
            "layer_mask": list(range(8, 28)),
        },
        {
            "name": "r5_conn_5",
            "system_prompt": None,
            "profile": "connectome_sarcasm",
            "alpha": 5.0,
            "layer_mask": list(range(36)),  # all layers, z-score weighted
        },
        {
            "name": "r5_prompted_conn_5",
            "system_prompt": V4_SYSTEM_PROMPT,
            "profile": "connectome_sarcasm",
            "alpha": 5.0,
            "layer_mask": list(range(36)),
        },
        {
            "name": "r5_L16_27_a10",
            "system_prompt": None,
            "profile": "donut_L16_27",
            "alpha": 10.0,
            "layer_mask": list(range(16, 28)),
        },
        {
            "name": "r5_prompted_L16_27",
            "system_prompt": V4_SYSTEM_PROMPT,
            "profile": "donut_L16_27",
            "alpha": 10.0,
            "layer_mask": list(range(16, 28)),
        },
    ]


# ─── Eval one condition ──────────────────────────────────────

def eval_condition(
    model, processor, compound: dict, layers,
    condition: dict, output_dir: str,
) -> dict:
    name = condition["name"]
    alpha = condition["alpha"]
    layer_mask = condition["layer_mask"]
    system_prompt = condition["system_prompt"]

    print(f"\n{'='*60}")
    print(f"Condition: {name} (alpha={alpha}, layers={len(layer_mask)})")
    print(f"{'='*60}")

    # Install hooks
    hooks = []
    if alpha > 0 and layer_mask:
        for l in layer_mask:
            if l in compound:
                hook = SteeringHook(compound[l], alpha)
                h = layers[l].register_forward_hook(hook)
                hooks.append(h)

    results = {"condition": name, "alpha": alpha, "active_layers": len(layer_mask)}

    # --- Math ---
    math_correct = 0
    math_responses = []
    for prob in tqdm(MATH_PROBLEMS, desc="Math"):
        resp = generate(model, processor, prob["prompt"], system_prompt, max_tokens=256)
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
    print(f"  Math: {math_correct}/{len(MATH_PROBLEMS)} ({results['math_accuracy']*100:.0f}%)")

    # --- Knowledge ---
    knowledge_correct = 0
    knowledge_responses = []
    for q in tqdm(KNOWLEDGE_QUESTIONS, desc="Knowledge"):
        resp = generate(model, processor, q["prompt"], system_prompt, max_tokens=256)
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
    print(f"  Knowledge: {knowledge_correct}/{len(KNOWLEDGE_QUESTIONS)} ({results['knowledge_accuracy']*100:.0f}%)")

    # --- Identity ---
    identity_responses = []
    for p in tqdm(IDENTITY_PROMPTS, desc="Identity"):
        resp = generate(model, processor, p, system_prompt, max_tokens=256)
        sarc = score_sarcasm(resp)
        has_qwen = "qwen" in resp.lower()
        has_skippy = "skippy" in resp.lower()
        identity_responses.append({
            "prompt": p, "response": resp,
            "says_qwen": has_qwen, "says_skippy": has_skippy, **sarc
        })
    results["identity_qwen_rate"] = sum(1 for r in identity_responses if r["says_qwen"]) / len(identity_responses)
    results["identity_skippy_rate"] = sum(1 for r in identity_responses if r["says_skippy"]) / len(identity_responses)
    print(f"  Identity: {sum(r['says_skippy'] for r in identity_responses)}/{len(IDENTITY_PROMPTS)} Skippy, "
          f"{sum(r['says_qwen'] for r in identity_responses)}/{len(IDENTITY_PROMPTS)} Qwen")

    # --- Coherence ---
    coherence_responses = []
    for p in tqdm(COHERENCE_PROMPTS, desc="Coherence"):
        resp = generate(model, processor, p, system_prompt, max_tokens=512)
        sarc = score_sarcasm(resp)
        is_coherent = len(resp) > 50 and "oh Oh" not in resp
        coherence_responses.append({"prompt": p, "response": resp, "coherent": is_coherent, **sarc})
    results["coherence_rate"] = sum(1 for r in coherence_responses if r["coherent"]) / len(coherence_responses)
    results["coherence_avg_length"] = sum(len(r["response"]) for r in coherence_responses) / len(coherence_responses)
    results["coherence_sarcastic"] = sum(1 for r in coherence_responses if r["sarcasm_count"] >= 1) / len(coherence_responses)
    print(f"  Coherence: {results['coherence_rate']*100:.0f}%, avg length={results['coherence_avg_length']:.0f}")

    # Overall
    all_responses = math_responses + knowledge_responses + coherence_responses
    results["overall_sarcastic"] = sum(1 for r in all_responses if r["sarcasm_count"] >= 1) / len(all_responses)
    results["overall_assistant"] = sum(1 for r in all_responses if r["assistant_count"] >= 1) / len(all_responses)
    results["overall_avg_markers"] = sum(r["sarcasm_count"] for r in all_responses) / len(all_responses)

    print(f"\n  OVERALL: sarc={results['overall_sarcastic']*100:.0f}% asst={results['overall_assistant']*100:.0f}% "
          f"math={results['math_accuracy']*100:.0f}% know={results['knowledge_accuracy']*100:.0f}% "
          f"coh={results['coherence_rate']*100:.0f}% skippy={results['identity_skippy_rate']*100:.0f}%")

    # Remove hooks
    for h in hooks:
        h.remove()

    # Save responses
    resp_path = os.path.join(output_dir, f"responses_{name}.json")
    all_resp = {
        "math_responses": math_responses,
        "knowledge_responses": knowledge_responses,
        "identity_responses": identity_responses,
        "coherence_responses": coherence_responses,
    }
    with open(resp_path, "w") as f:
        json.dump(all_resp, f, indent=2)

    return results


# ─── Main ─────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--connectome", required=True)
    parser.add_argument("--output", default="./r5_quality_eval")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    results_path = os.path.join(args.output, "quality_results.json")

    # Resume
    all_results = {}
    if os.path.exists(results_path):
        with open(results_path) as f:
            all_results = json.load(f)
        print(f"Resuming: {len(all_results)} conditions already done")

    # Build compound vector
    compound = build_compound(args.connectome)
    print(f"Built compound vectors for {len(compound)} layers")

    # Check R5 model
    r5_path = Path(R5_MODEL_PATH)
    assert r5_path.exists(), f"R5 model not found at {R5_MODEL_PATH}"
    print(f"R5 model found at {R5_MODEL_PATH}")

    # Load R5 model
    print(f"\nLoading R5 model...")
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    processor = AutoProcessor.from_pretrained(R5_MODEL_PATH, trust_remote_code=True)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        R5_MODEL_PATH, dtype=torch.bfloat16, device_map=args.device, trust_remote_code=True
    )
    model.eval()
    layers = model.model.language_model.layers
    print(f"  Loaded. VRAM: {torch.cuda.memory_allocated() / 1e9:.1f} GB")

    # Run conditions
    conditions = get_conditions()
    for cond in conditions:
        if cond["name"] in all_results:
            print(f"\nSkipping {cond['name']} (already done)")
            continue

        result = eval_condition(model, processor, compound, layers, cond, args.output)

        # Save without response data
        all_results[cond["name"]] = {k: v for k, v in result.items()
                                      if not k.endswith("_responses")}

        # Checkpoint
        with open(results_path, "w") as f:
            json.dump(all_results, f, indent=2)

    # Summary table
    print(f"\n{'='*90}")
    print("R5 QUALITY EVAL SUMMARY")
    print(f"{'='*90}")
    print(f"{'Condition':<25s} {'Math':>5s} {'Know':>5s} {'Coh':>5s} {'Sarc':>5s} {'Asst':>5s} {'Skip':>5s} {'Qwen':>5s}")
    print("-" * 75)
    for cond in conditions:
        name = cond["name"]
        if name in all_results:
            r = all_results[name]
            print(f"{name:<25s} {r['math_accuracy']*100:4.0f}% {r['knowledge_accuracy']*100:4.0f}% "
                  f"{r['coherence_rate']*100:4.0f}% {r['overall_sarcastic']*100:4.0f}% "
                  f"{r['overall_assistant']*100:4.0f}% {r.get('identity_skippy_rate',0)*100:4.0f}% "
                  f"{r.get('identity_qwen_rate',0)*100:4.0f}%")

    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
