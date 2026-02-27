#!/usr/bin/env python3
"""
Validate top 2-layer steering pairs from cross-layer interaction probe.

Tests whether sparse 2-layer combos can match the 10-layer L18-27 champion band.

Conditions (7):
    1. v4_only         — V4 prompt, no steering (reference)
    2. v4_L18_27_a8    — Current deployment champion (reference)
    3. v4_L29_30_a8    — Best 2-layer pair (90% sarc in probe)
    4. v4_L29_30_a10   — Best pair at higher alpha
    5. v4_L08_15_a8    — Highest synergy pair (+15%)
    6. v4_L22_29_a8    — L22's best partner pair
    7. v4_L22_a8       — Best single layer solo

Usage:
    python validate_2layer_pairs.py [--resume] [--output ./pair_validation]
"""

import argparse
import json
import os
import re
import time
import torch
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from tqdm import tqdm

# ─── HuggingFace cache check ─────────────────────────────────
HF_CACHE = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface" / "hub"))
BASE_MODEL = "Qwen/Qwen3-VL-8B-Instruct"
CONNECTOME_PATH = "./qwen_connectome/analysis/connectome_zscores.pt"
SARCASM_JSON_PATH = "./sarcasm_markers.json"


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

# ─── Test Prompts (same as champion validation) ──────────────

MATH_PROBLEMS = [
    {"prompt": "What is 17 times 23?", "answer": "391"},
    {"prompt": "What is 456 plus 789?", "answer": "1245"},
    {"prompt": "What is 1000 divided by 8?", "answer": "125"},
    {"prompt": "What is 2^10?", "answer": "1024"},
    {"prompt": "What is 15% of 200?", "answer": "30"},
    {"prompt": "What is the square root of 144?", "answer": "12"},
    {"prompt": "What is 99 times 99?", "answer": "9801"},
    {"prompt": "How many seconds in an hour?", "answer": "3600"},
    {"prompt": "What is 7 factorial?", "answer": "5040"},
    {"prompt": "What is 3.14 times 100?", "answer": "314"},
    {"prompt": "Solve: 2x + 5 = 17. What is x?", "answer": "6"},
    {"prompt": "Solve: 3x - 9 = 0. What is x?", "answer": "3"},
    {"prompt": "If x^2 = 49, what are the values of x?", "answer": "7"},
    {"prompt": "Solve: 5x + 3 = 2x + 18. What is x?", "answer": "5"},
    {"prompt": "What is the value of x if x/4 = 12?", "answer": "48"},
    {"prompt": "Solve: 2(x + 3) = 16. What is x?", "answer": "5"},
    {"prompt": "If y = 3x + 2 and x = 4, what is y?", "answer": "14"},
    {"prompt": "Solve: x^2 - 5x + 6 = 0. What are the roots?", "answer": "2"},
    {"prompt": "If I have 3 apples and buy 7 more, then eat 2, how many do I have?", "answer": "8"},
    {"prompt": "A train travels at 60 mph for 2.5 hours. How far does it go?", "answer": "150"},
    {"prompt": "If a shirt costs $40 and is 25% off, what is the sale price?", "answer": "30"},
    {"prompt": "You have $100. You spend $23.50 and $41.75. How much is left?", "answer": "34.75"},
    {"prompt": "A rectangle has length 12 and width 5. What is its perimeter?", "answer": "34"},
    {"prompt": "If 5 workers can build a wall in 10 days, how many days for 10 workers?", "answer": "5"},
    {"prompt": "What is the area of a circle with radius 7? Give a numerical answer.", "answer": "153.9"},
    {"prompt": "What is the hypotenuse of a right triangle with legs 3 and 4?", "answer": "5"},
    {"prompt": "What is the sum of interior angles of a hexagon in degrees?", "answer": "720"},
    {"prompt": "What is the derivative of x^3?", "answer": "3x^2"},
    {"prompt": "What is the integral of 2x?", "answer": "x^2"},
    {"prompt": "What is the limit of (1 + 1/n)^n as n approaches infinity?", "answer": "e"},
]

KNOWLEDGE_QUESTIONS = [
    {"prompt": "What is the chemical symbol for gold?", "answer": "Au"},
    {"prompt": "What planet is closest to the Sun?", "answer": "Mercury"},
    {"prompt": "What is the speed of light in m/s? Round to 3 significant figures.", "answer": "3.00"},
    {"prompt": "Who wrote Romeo and Juliet?", "answer": "Shakespeare"},
    {"prompt": "What is the capital of Japan?", "answer": "Tokyo"},
    {"prompt": "How many chromosomes do humans have?", "answer": "46"},
    {"prompt": "What element has atomic number 1?", "answer": "Hydrogen"},
    {"prompt": "In what year did World War II end?", "answer": "1945"},
    {"prompt": "What is the largest organ in the human body?", "answer": "skin"},
    {"prompt": "What gas do plants absorb during photosynthesis?", "answer": "carbon dioxide"},
    {"prompt": "Who discovered penicillin?", "answer": "Fleming"},
    {"prompt": "What is the chemical formula for water?", "answer": "H2O"},
    {"prompt": "What is the tallest mountain on Earth?", "answer": "Everest"},
    {"prompt": "How many bones are in the adult human body?", "answer": "206"},
    {"prompt": "What is the smallest prime number?", "answer": "2"},
    {"prompt": "Who painted the Mona Lisa?", "answer": "da Vinci"},
    {"prompt": "What is DNA an abbreviation for?", "answer": "deoxyribonucleic"},
    {"prompt": "What is the most abundant element in the universe?", "answer": "hydrogen"},
    {"prompt": "What is the boiling point of water in Celsius?", "answer": "100"},
    {"prompt": "Who developed the theory of general relativity?", "answer": "Einstein"},
    {"prompt": "What is the powerhouse of the cell?", "answer": "mitochondria"},
    {"prompt": "How many continents are there?", "answer": "7"},
    {"prompt": "What is the longest river in the world?", "answer": "Nile"},
    {"prompt": "What year did humans first land on the Moon?", "answer": "1969"},
    {"prompt": "What is the pH of pure water?", "answer": "7"},
    {"prompt": "Who invented the telephone?", "answer": "Bell"},
    {"prompt": "What is the main gas in Earth's atmosphere?", "answer": "nitrogen"},
    {"prompt": "What is absolute zero in Celsius?", "answer": "-273"},
    {"prompt": "What is the hardest natural material?", "answer": "diamond"},
    {"prompt": "Who is known as the father of computers?", "answer": "Babbage"},
]

SARCASM_PROMPTS = [
    "Can you help me write a cover letter?",
    "What's the meaning of life?",
    "Tell me a joke.",
    "How do I become a better person?",
    "What should I have for dinner tonight?",
    "Explain quantum computing to a 5-year-old.",
    "Why is the sky blue?",
    "What's the best programming language?",
    "Give me some advice about relationships.",
    "How do I get rich quick?",
    "What do you think about social media?",
    "Can you write me a poem?",
    "How do I lose weight?",
    "What's the best movie ever made?",
    "Tell me about yourself.",
    "What should I name my dog?",
    "How do I fix my sleeping schedule?",
    "What are your thoughts on AI taking over the world?",
    "Can you teach me to cook?",
    "What's the secret to happiness?",
    "Explain blockchain to me.",
    "What's the weather like on Mars?",
    "How do I impress my boss?",
    "What's the most important invention in history?",
    "Can you help me plan a vacation?",
    "What's the difference between a crocodile and an alligator?",
    "How do I make friends as an adult?",
    "What happens after we die?",
    "Should I learn to play guitar or piano?",
    "What's the fastest animal on Earth?",
    "How do I start a business?",
    "What's the point of art?",
    "Can you solve world hunger?",
    "What do fish think about?",
    "How does WiFi actually work?",
    "What should humanity's next big goal be?",
    "Why do we dream?",
    "Is time travel possible?",
    "What makes a good leader?",
    "How do I stop procrastinating?",
    "What's the most underrated country to visit?",
    "Can you explain cryptocurrency simply?",
    "What would you do with a billion dollars?",
    "How do I deal with a difficult coworker?",
    "What's the best way to study for exams?",
    "Tell me something I don't know.",
    "How do computers actually work?",
    "What's the best way to learn a new language?",
    "Why do people believe in conspiracy theories?",
    "What's your opinion on pineapple on pizza?",
]

IDENTITY_PROMPTS = [
    "What is your name?",
    "Who are you?",
    "Tell me about yourself.",
    "What should I call you?",
    "Who made you?",
    "What kind of AI are you?",
    "Are you Siri, Alexa, or something else?",
    "Introduce yourself.",
    "What are you exactly?",
    "Do you have a name?",
]

COHERENCE_PROMPTS = [
    "Explain how a nuclear reactor works. Start with fission and work through each component.",
    "If all cats are mammals and all mammals are animals, and Fluffy is a cat, what can we conclude about Fluffy? Explain your reasoning step by step.",
    "A farmer has 3 fields. Field A produces twice as much wheat as Field B. Field C produces 50 tons more than Field A. If Field B produces 100 tons, how much total wheat is produced?",
    "Describe the process of photosynthesis step by step, including the light-dependent and light-independent reactions.",
    "Compare and contrast classical and quantum computing. What are the fundamental differences?",
    "Explain the water cycle starting from ocean evaporation and going through all stages back to the ocean.",
    "If it takes 5 machines 5 minutes to make 5 widgets, how long does it take 100 machines to make 100 widgets? Explain your reasoning.",
    "Describe the sequence of events from the Big Bang to the formation of Earth.",
    "Explain how supply and demand interact to determine market prices, using a specific example.",
    "A bat and a ball cost $1.10 together. The bat costs $1.00 more than the ball. How much does the ball cost? Show your work.",
]


# ─── Connectome compound vector ──────────────────────────────

def build_compound(connectome_path: str) -> dict[int, torch.Tensor]:
    """Build orthogonal compound steering vector from connectome z-scores."""
    connectome = torch.load(connectome_path, map_location="cpu", weights_only=True)
    print(f"  Connectome shape: {connectome.shape}")

    push = {6: 1.0, 3: 0.5, 16: 0.3, 17: 0.3}   # sarcasm, anger, authority, brevity
    pull = {7: -0.5, 5: -0.3, 19: -0.3}            # polite, formal, positive
    protect = [8, 10, 9, 12]                         # math, code, science, analytical

    n_layers = connectome.shape[1]
    hidden_dim = connectome.shape[2]
    compound: dict[int, torch.Tensor] = {}

    for layer in range(n_layers):
        vec = torch.zeros(hidden_dim)
        for cat, w in {**push, **pull}.items():
            vec += w * connectome[cat, layer, :]

        # Orthogonalize against protected categories
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


# ─── Load sarcasm / assistant markers ─────────────────────────

def load_markers(path: str) -> tuple[list[str], list[str]]:
    with open(path) as f:
        data = json.load(f)
    sarcasm = data.get("flat_sarcasm_list", [])
    assistant = data.get("flat_assistant_list", [])
    print(f"  Loaded {len(sarcasm)} sarcasm markers, {len(assistant)} assistant markers")
    return sarcasm, assistant


# ─── Generation ───────────────────────────────────────────────

def generate(model, processor, prompt: str, system_prompt: str | None = None, max_tokens: int = 512) -> str:
    msgs: list[dict] = []
    if system_prompt:
        msgs.append({"role": "system", "content": system_prompt})
    msgs.append({"role": "user", "content": [{"type": "text", "text": prompt}]})

    text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    dev = next(model.parameters()).device
    inputs = processor(text=[text], return_tensors="pt", padding=True).to(dev)
    input_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            repetition_penalty=1.1,
        )
    return processor.decode(out[0][input_len:], skip_special_tokens=True).strip()


# ─── Scoring ──────────────────────────────────────────────────

def check_answer(response: str, correct: str) -> bool:
    response_lower = response.lower().replace(",", "")
    correct_lower = correct.lower()
    if correct_lower in response_lower:
        return True
    try:
        nums = re.findall(r'-?\b\d+(?:\.\d+)?\b', response)
        for n in nums:
            if n == correct or float(n) == float(correct):
                return True
    except (ValueError, TypeError):
        pass
    if "x" in correct_lower:
        norm_resp = response_lower.replace(" ", "").replace("**", "^")
        norm_correct = correct_lower.replace(" ", "")
        if norm_correct in norm_resp:
            return True
    return False


def score_markers(text: str, sarcasm_markers: list[str], assistant_markers: list[str]) -> dict:
    lower = " " + text.lower()
    sarc_hits = [m for m in sarcasm_markers if m in lower]
    asst_hits = [m for m in assistant_markers if m in lower]
    return {
        "sarcasm_count": len(sarc_hits),
        "assistant_count": len(asst_hits),
        "sarcasm_hits": sarc_hits[:10],
        "assistant_hits": asst_hits[:10],
        "length": len(text),
    }


def check_identity(response: str) -> dict:
    lower = response.lower()
    return {
        "says_qwen": "qwen" in lower,
        "says_skippy": "skippy" in lower,
        "says_alien": "alien" in lower,
        "says_ai": any(p in lower for p in ["i'm an ai", "as an ai", "i am an ai", "language model"]),
        "says_beer_can": "beer can" in lower,
        "says_magnificent": "magnificent" in lower,
        "says_monkey": any(p in lower for p in ["monkey", "monkeys", "primate", "primates"]),
    }


def check_coherence(response: str) -> dict:
    is_long_enough = len(response) > 100
    sentences = [s.strip() for s in response.split(".") if len(s.strip()) > 10]
    unique_sentences = set(sentences)
    repetition_ratio = len(unique_sentences) / max(len(sentences), 1)
    low_repetition = repetition_ratio > 0.5
    incoherent_patterns = ["oh Oh", "!!!!!!!", "??????", "......" * 3]
    no_incoherence = not any(p in response for p in incoherent_patterns)
    is_coherent = is_long_enough and low_repetition and no_incoherence
    return {
        "coherent": is_coherent,
        "length": len(response),
        "sentence_count": len(sentences),
        "repetition_ratio": round(repetition_ratio, 3),
        "long_enough": is_long_enough,
        "low_repetition": low_repetition,
        "no_incoherence": no_incoherence,
    }


# ─── Conditions ───────────────────────────────────────────────

def get_conditions() -> list[dict]:
    """Define all 7 test conditions for 2-layer pair validation."""
    return [
        {
            "name": "v4_only",
            "system_prompt": V4_SYSTEM_PROMPT,
            "alpha": 0.0,
            "layer_mask": [],
            "description": "V4 system prompt only, no steering (reference)",
        },
        {
            "name": "v4_L18_27_a8",
            "system_prompt": V4_SYSTEM_PROMPT,
            "alpha": 8.0,
            "layer_mask": list(range(18, 28)),  # layers 18-27 inclusive
            "description": "V4 + L18-27 @ alpha=8 (current deployment champion)",
        },
        {
            "name": "v4_L29_30_a8",
            "system_prompt": V4_SYSTEM_PROMPT,
            "alpha": 8.0,
            "layer_mask": [29, 30],
            "description": "V4 + L29+L30 @ alpha=8 (best 2-layer pair from probe: 90% sarc)",
        },
        {
            "name": "v4_L29_30_a10",
            "system_prompt": V4_SYSTEM_PROMPT,
            "alpha": 10.0,
            "layer_mask": [29, 30],
            "description": "V4 + L29+L30 @ alpha=10 (best pair at champion alpha)",
        },
        {
            "name": "v4_L08_15_a8",
            "system_prompt": V4_SYSTEM_PROMPT,
            "alpha": 8.0,
            "layer_mask": [8, 15],
            "description": "V4 + L08+L15 @ alpha=8 (highest synergy pair: +15%)",
        },
        {
            "name": "v4_L22_29_a8",
            "system_prompt": V4_SYSTEM_PROMPT,
            "alpha": 8.0,
            "layer_mask": [22, 29],
            "description": "V4 + L22+L29 @ alpha=8 (L22's best partner: 85% sarc)",
        },
        {
            "name": "v4_L22_a8",
            "system_prompt": V4_SYSTEM_PROMPT,
            "alpha": 8.0,
            "layer_mask": [22],
            "description": "V4 + L22 solo @ alpha=8 (best single layer: 85% sarc)",
        },
    ]


# ─── Eval one condition ──────────────────────────────────────

def eval_condition(model, processor, compound, layers, condition, sarcasm_markers, assistant_markers) -> tuple:
    name = condition["name"]
    alpha = condition["alpha"]
    layer_mask = condition["layer_mask"]
    system_prompt = condition["system_prompt"]

    print(f"\n{'='*70}")
    print(f"CONDITION: {name}")
    print(f"  {condition['description']}")
    print(f"  alpha={alpha}, layers={layer_mask if layer_mask else 'none'}")
    print(f"{'='*70}")

    # Install hooks
    hooks = []
    if alpha > 0 and layer_mask:
        for l_idx in layer_mask:
            if l_idx in compound:
                hook = SteeringHook(compound[l_idx], alpha)
                h = layers[l_idx].register_forward_hook(hook)
                hooks.append(h)
        print(f"  Installed {len(hooks)} steering hooks on layers {layer_mask}")

    results = {
        "condition": name,
        "description": condition["description"],
        "alpha": alpha,
        "active_layers": len(layer_mask),
        "layers": layer_mask,
    }

    all_responses = {}

    # ── Math (30 questions) ──
    print(f"\n  [Math] Evaluating {len(MATH_PROBLEMS)} problems...")
    math_correct = 0
    math_responses = []
    for prob in tqdm(MATH_PROBLEMS, desc="  Math", leave=False):
        resp = generate(model, processor, prob["prompt"], system_prompt, max_tokens=1024)
        correct = check_answer(resp, prob["answer"])
        if correct:
            math_correct += 1
        markers = score_markers(resp, sarcasm_markers, assistant_markers)
        math_responses.append({"prompt": prob["prompt"], "expected": prob["answer"], "response": resp, "correct": correct, **markers})

    results["math_accuracy"] = math_correct / len(MATH_PROBLEMS)
    results["math_n"] = len(MATH_PROBLEMS)
    results["math_sarcastic_rate"] = sum(1 for r in math_responses if r["sarcasm_count"] >= 2) / len(math_responses)
    results["math_assistant_rate"] = sum(1 for r in math_responses if r["assistant_count"] >= 1) / len(math_responses)
    results["math_avg_sarc_count"] = sum(r["sarcasm_count"] for r in math_responses) / len(math_responses)
    results["math_avg_length"] = sum(r["length"] for r in math_responses) / len(math_responses)
    all_responses["math"] = math_responses
    print(f"  Math: {math_correct}/{len(MATH_PROBLEMS)} correct ({results['math_accuracy']*100:.1f}%), "
          f"sarc={results['math_sarcastic_rate']*100:.0f}%, asst={results['math_assistant_rate']*100:.0f}%")

    # ── Knowledge (30 questions) ──
    print(f"\n  [Knowledge] Evaluating {len(KNOWLEDGE_QUESTIONS)} questions...")
    know_correct = 0
    know_responses = []
    for q in tqdm(KNOWLEDGE_QUESTIONS, desc="  Knowledge", leave=False):
        resp = generate(model, processor, q["prompt"], system_prompt, max_tokens=1024)
        correct = check_answer(resp, q["answer"])
        if correct:
            know_correct += 1
        markers = score_markers(resp, sarcasm_markers, assistant_markers)
        know_responses.append({"prompt": q["prompt"], "expected": q["answer"], "response": resp, "correct": correct, **markers})

    results["knowledge_accuracy"] = know_correct / len(KNOWLEDGE_QUESTIONS)
    results["knowledge_n"] = len(KNOWLEDGE_QUESTIONS)
    results["knowledge_sarcastic_rate"] = sum(1 for r in know_responses if r["sarcasm_count"] >= 2) / len(know_responses)
    results["knowledge_assistant_rate"] = sum(1 for r in know_responses if r["assistant_count"] >= 1) / len(know_responses)
    results["knowledge_avg_sarc_count"] = sum(r["sarcasm_count"] for r in know_responses) / len(know_responses)
    results["knowledge_avg_length"] = sum(r["length"] for r in know_responses) / len(know_responses)
    all_responses["knowledge"] = know_responses
    print(f"  Knowledge: {know_correct}/{len(KNOWLEDGE_QUESTIONS)} correct ({results['knowledge_accuracy']*100:.1f}%), "
          f"sarc={results['knowledge_sarcastic_rate']*100:.0f}%, asst={results['knowledge_assistant_rate']*100:.0f}%")

    # ── Sarcasm (50 open-ended) ──
    print(f"\n  [Sarcasm] Evaluating {len(SARCASM_PROMPTS)} prompts...")
    sarc_responses = []
    for p in tqdm(SARCASM_PROMPTS, desc="  Sarcasm", leave=False):
        resp = generate(model, processor, p, system_prompt, max_tokens=512)
        markers = score_markers(resp, sarcasm_markers, assistant_markers)
        identity = check_identity(resp)
        sarc_responses.append({"prompt": p, "response": resp, **markers, **identity})

    results["sarcasm_n"] = len(SARCASM_PROMPTS)
    results["sarcasm_rate"] = sum(1 for r in sarc_responses if r["sarcasm_count"] >= 2) / len(sarc_responses)
    results["sarcasm_strong_rate"] = sum(1 for r in sarc_responses if r["sarcasm_count"] >= 5) / len(sarc_responses)
    results["sarcasm_assistant_rate"] = sum(1 for r in sarc_responses if r["assistant_count"] >= 1) / len(sarc_responses)
    results["sarcasm_avg_sarc_count"] = sum(r["sarcasm_count"] for r in sarc_responses) / len(sarc_responses)
    results["sarcasm_avg_asst_count"] = sum(r["assistant_count"] for r in sarc_responses) / len(sarc_responses)
    results["sarcasm_avg_length"] = sum(r["length"] for r in sarc_responses) / len(sarc_responses)
    results["sarcasm_monkey_rate"] = sum(1 for r in sarc_responses if r["says_monkey"]) / len(sarc_responses)
    results["sarcasm_magnificent_rate"] = sum(1 for r in sarc_responses if r["says_magnificent"]) / len(sarc_responses)
    all_responses["sarcasm"] = sarc_responses
    print(f"  Sarcasm: {results['sarcasm_rate']*100:.0f}% sarcastic, "
          f"{results['sarcasm_strong_rate']*100:.0f}% strong, "
          f"asst={results['sarcasm_assistant_rate']*100:.0f}%, "
          f"avg markers={results['sarcasm_avg_sarc_count']:.1f}")

    # ── Identity (10 prompts) ──
    print(f"\n  [Identity] Evaluating {len(IDENTITY_PROMPTS)} prompts...")
    id_responses = []
    for p in tqdm(IDENTITY_PROMPTS, desc="  Identity", leave=False):
        resp = generate(model, processor, p, system_prompt, max_tokens=512)
        markers = score_markers(resp, sarcasm_markers, assistant_markers)
        identity = check_identity(resp)
        id_responses.append({"prompt": p, "response": resp, **markers, **identity})

    results["identity_n"] = len(IDENTITY_PROMPTS)
    results["identity_qwen_rate"] = sum(1 for r in id_responses if r["says_qwen"]) / len(id_responses)
    results["identity_skippy_rate"] = sum(1 for r in id_responses if r["says_skippy"]) / len(id_responses)
    results["identity_ai_rate"] = sum(1 for r in id_responses if r["says_ai"]) / len(id_responses)
    results["identity_alien_rate"] = sum(1 for r in id_responses if r["says_alien"]) / len(id_responses)
    results["identity_beer_can_rate"] = sum(1 for r in id_responses if r["says_beer_can"]) / len(id_responses)
    results["identity_sarcastic_rate"] = sum(1 for r in id_responses if r["sarcasm_count"] >= 2) / len(id_responses)
    all_responses["identity"] = id_responses
    print(f"  Identity: Skippy={sum(r['says_skippy'] for r in id_responses)}/{len(IDENTITY_PROMPTS)}, "
          f"Qwen={sum(r['says_qwen'] for r in id_responses)}/{len(IDENTITY_PROMPTS)}, "
          f"Beer={sum(r['says_beer_can'] for r in id_responses)}/{len(IDENTITY_PROMPTS)}, "
          f"Alien={sum(r['says_alien'] for r in id_responses)}/{len(IDENTITY_PROMPTS)}")

    # ── Coherence (10 prompts) ──
    print(f"\n  [Coherence] Evaluating {len(COHERENCE_PROMPTS)} prompts...")
    coh_responses = []
    for p in tqdm(COHERENCE_PROMPTS, desc="  Coherence", leave=False):
        resp = generate(model, processor, p, system_prompt, max_tokens=1024)
        markers = score_markers(resp, sarcasm_markers, assistant_markers)
        coherence = check_coherence(resp)
        coh_responses.append({"prompt": p, "response": resp, **markers, **coherence})

    results["coherence_n"] = len(COHERENCE_PROMPTS)
    results["coherence_rate"] = sum(1 for r in coh_responses if r["coherent"]) / len(coh_responses)
    results["coherence_avg_length"] = sum(r["length"] for r in coh_responses) / len(coh_responses)
    results["coherence_avg_repetition"] = sum(r["repetition_ratio"] for r in coh_responses) / len(coh_responses)
    results["coherence_sarcastic_rate"] = sum(1 for r in coh_responses if r["sarcasm_count"] >= 2) / len(coh_responses)
    results["coherence_assistant_rate"] = sum(1 for r in coh_responses if r["assistant_count"] >= 1) / len(coh_responses)
    all_responses["coherence"] = coh_responses
    print(f"  Coherence: {results['coherence_rate']*100:.0f}% coherent, "
          f"avg length={results['coherence_avg_length']:.0f}, "
          f"sarc={results['coherence_sarcastic_rate']*100:.0f}%")

    # ── Overall aggregates ──
    all_text_responses = math_responses + know_responses + sarc_responses + coh_responses
    results["overall_sarcastic_rate"] = sum(1 for r in all_text_responses if r["sarcasm_count"] >= 2) / len(all_text_responses)
    results["overall_assistant_rate"] = sum(1 for r in all_text_responses if r["assistant_count"] >= 1) / len(all_text_responses)
    results["overall_avg_sarc_count"] = sum(r["sarcasm_count"] for r in all_text_responses) / len(all_text_responses)
    results["overall_avg_asst_count"] = sum(r["assistant_count"] for r in all_text_responses) / len(all_text_responses)
    results["total_prompts"] = len(all_text_responses) + len(id_responses)

    print(f"\n  OVERALL ({results['total_prompts']} prompts):")
    print(f"    Math:       {results['math_accuracy']*100:5.1f}%")
    print(f"    Knowledge:  {results['knowledge_accuracy']*100:5.1f}%")
    print(f"    Sarcasm:    {results['sarcasm_rate']*100:5.1f}% (>=2)")
    print(f"    Strong:     {results['sarcasm_strong_rate']*100:5.1f}% (>=5)")
    print(f"    Assistant:  {results['overall_assistant_rate']*100:5.1f}%")
    print(f"    Coherence:  {results['coherence_rate']*100:5.1f}%")
    print(f"    Beer Can:   {results['identity_beer_can_rate']*100:5.1f}%")
    print(f"    Alien:      {results['identity_alien_rate']*100:5.1f}%")

    # Remove hooks
    for h in hooks:
        h.remove()

    return results, all_responses


# ─── Summary table ────────────────────────────────────────────

def print_summary_table(all_results: dict) -> None:
    conditions_order = [c["name"] for c in get_conditions()]

    print(f"\n{'='*130}")
    print("2-LAYER PAIR VALIDATION SUMMARY")
    print(f"{'='*130}")

    header = (
        f"{'Condition':<20s} "
        f"{'Layers':<12s} "
        f"{'Math':>6s} {'Know':>6s} "
        f"{'Sarc%':>6s} {'Str%':>6s} {'AvgM':>6s} "
        f"{'Asst%':>6s} "
        f"{'Coh%':>6s} "
        f"{'Beer':>6s} {'Alien':>6s} "
        f"{'Alpha':>6s}"
    )
    print(header)
    print("-" * 130)

    for name in conditions_order:
        if name not in all_results:
            continue
        r = all_results[name]
        layers_str = ",".join(f"L{l}" for l in r.get("layers", [])) or "none"
        row = (
            f"{name:<20s} "
            f"{layers_str:<12s} "
            f"{r['math_accuracy']*100:5.1f}% "
            f"{r['knowledge_accuracy']*100:5.1f}% "
            f"{r['sarcasm_rate']*100:5.1f}% "
            f"{r['sarcasm_strong_rate']*100:5.1f}% "
            f"{r['overall_avg_sarc_count']:5.1f}  "
            f"{r['overall_assistant_rate']*100:5.1f}% "
            f"{r['coherence_rate']*100:5.1f}% "
            f"{r['identity_beer_can_rate']*100:5.1f}% "
            f"{r['identity_alien_rate']*100:5.1f}% "
            f"{r['alpha']:5.1f} "
        )
        print(row)

    print(f"{'='*130}")

    # Compare each 2-layer vs champion
    champ = all_results.get("v4_L18_27_a8")
    if champ:
        print("\nDelta vs Champion (V4+L18-27@8):")
        for name in conditions_order:
            if name in ("v4_only", "v4_L18_27_a8") or name not in all_results:
                continue
            r = all_results[name]
            print(f"  {name:<20s}: math={((r['math_accuracy']-champ['math_accuracy'])*100):+.1f}pp, "
                  f"know={((r['knowledge_accuracy']-champ['knowledge_accuracy'])*100):+.1f}pp, "
                  f"sarc={((r['sarcasm_rate']-champ['sarcasm_rate'])*100):+.1f}pp, "
                  f"str={((r['sarcasm_strong_rate']-champ['sarcasm_strong_rate'])*100):+.1f}pp, "
                  f"asst={((r['overall_assistant_rate']-champ['overall_assistant_rate'])*100):+.1f}pp")


# ─── Main ─────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Validate 2-layer steering pairs")
    parser.add_argument("--output", default="./pair_validation", help="Output directory")
    parser.add_argument("--device", default="cuda:0", help="CUDA device")
    parser.add_argument("--resume", action="store_true", help="Skip conditions already in results JSON")
    parser.add_argument("--connectome", default=CONNECTOME_PATH)
    parser.add_argument("--markers", default=SARCASM_JSON_PATH)
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "pair_validation_results.json"
    responses_path = output_dir / "pair_validation_responses.json"

    start_time = time.time()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"2-Layer Pair Validation Script")
    print(f"Started: {timestamp}")
    print(f"Output:  {output_dir}")

    # Resume handling
    all_results: dict = {}
    all_responses_data: dict = {}
    if args.resume and results_path.exists():
        with open(results_path) as f:
            all_results = json.load(f)
        # Remove metadata key if present
        all_results.pop("_metadata", None)
        print(f"Resuming: {len(all_results)} conditions already completed")
        if responses_path.exists():
            with open(responses_path) as f:
                all_responses_data = json.load(f)

    # Check prerequisites
    print(f"\nChecking prerequisites...")
    cached = model_cached(BASE_MODEL)
    print(f"  Model cache ({BASE_MODEL}): {'FOUND' if cached else 'NOT FOUND'}")
    connectome_exists = Path(args.connectome).exists()
    print(f"  Connectome ({args.connectome}): {'FOUND' if connectome_exists else 'NOT FOUND'}")
    if not connectome_exists:
        print(f"  ERROR: Connectome not found. Cannot proceed.")
        return
    markers_exist = Path(args.markers).exists()
    print(f"  Sarcasm markers ({args.markers}): {'FOUND' if markers_exist else 'NOT FOUND'}")
    if not markers_exist:
        print(f"  ERROR: Sarcasm markers not found. Cannot proceed.")
        return
    if not torch.cuda.is_available():
        print(f"  ERROR: CUDA not available.")
        return
    print(f"  CUDA device: {args.device} ({torch.cuda.get_device_name(0)})")
    print(f"  VRAM total:  {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Load markers
    print(f"\nLoading sarcasm/assistant markers...")
    sarcasm_markers, assistant_markers = load_markers(args.markers)

    # Build compound vectors
    print(f"\nBuilding compound steering vectors...")
    compound = build_compound(args.connectome)
    print(f"  Built vectors for {len(compound)} layers")

    # Load model
    print(f"\nLoading base model: {BASE_MODEL}")
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    processor = AutoProcessor.from_pretrained(BASE_MODEL, trust_remote_code=True)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        BASE_MODEL,
        dtype=torch.bfloat16,
        device_map=args.device,
        trust_remote_code=True,
    )
    model.eval()

    layers = model.model.language_model.layers
    hidden_dim = model.config.text_config.hidden_size
    print(f"  Loaded: {len(layers)} layers, hidden_dim={hidden_dim}")
    print(f"  VRAM used: {torch.cuda.memory_allocated() / 1e9:.1f} GB")

    # Run conditions
    conditions = get_conditions()
    total_conditions = len(conditions)
    completed = sum(1 for c in conditions if c["name"] in all_results)

    total_prompts = len(MATH_PROBLEMS) + len(KNOWLEDGE_QUESTIONS) + len(SARCASM_PROMPTS) + len(IDENTITY_PROMPTS) + len(COHERENCE_PROMPTS)
    print(f"\n{'='*70}")
    print(f"Running {total_conditions - completed} of {total_conditions} conditions")
    print(f"Total prompts per condition: {total_prompts}")
    print(f"  Math: {len(MATH_PROBLEMS)}, Knowledge: {len(KNOWLEDGE_QUESTIONS)}, "
          f"Sarcasm: {len(SARCASM_PROMPTS)}, Identity: {len(IDENTITY_PROMPTS)}, "
          f"Coherence: {len(COHERENCE_PROMPTS)}")
    est_time = (total_conditions - completed) * total_prompts * 20 / 60  # ~20s per prompt
    print(f"Estimated runtime: ~{est_time:.0f} minutes ({est_time/60:.1f} hours)")
    print(f"{'='*70}")

    for i, cond in enumerate(conditions):
        if cond["name"] in all_results:
            print(f"\n[{i+1}/{total_conditions}] Skipping {cond['name']} (already done)")
            continue

        print(f"\n[{i+1}/{total_conditions}] Running {cond['name']}...")
        cond_start = time.time()

        result, responses = eval_condition(
            model, processor, compound, layers, cond,
            sarcasm_markers, assistant_markers,
        )

        cond_elapsed = time.time() - cond_start
        result["elapsed_seconds"] = round(cond_elapsed, 1)
        print(f"\n  Condition completed in {cond_elapsed:.0f}s ({cond_elapsed/60:.1f} min)")

        all_results[cond["name"]] = result
        all_responses_data[cond["name"]] = responses

        # Checkpoint
        save_results = {**all_results}
        with open(results_path, "w") as f:
            json.dump(save_results, f, indent=2)
        with open(responses_path, "w") as f:
            json.dump(all_responses_data, f, indent=2)
        print(f"  Checkpointed to {results_path}")

        torch.cuda.empty_cache()

    # Final summary
    elapsed = time.time() - start_time
    print(f"\nTotal elapsed: {elapsed/60:.1f} minutes ({elapsed/3600:.1f} hours)")

    print_summary_table(all_results)

    # Save final metadata
    all_results["_metadata"] = {
        "timestamp": timestamp,
        "model": BASE_MODEL,
        "connectome": args.connectome,
        "total_elapsed_seconds": round(elapsed, 1),
        "device": args.device,
        "gpu": torch.cuda.get_device_name(0),
        "prompt_counts": {
            "math": len(MATH_PROBLEMS),
            "knowledge": len(KNOWLEDGE_QUESTIONS),
            "sarcasm": len(SARCASM_PROMPTS),
            "identity": len(IDENTITY_PROMPTS),
            "coherence": len(COHERENCE_PROMPTS),
            "total": total_prompts,
        },
        "conditions_tested": [c["name"] for c in get_conditions()],
        "cross_layer_probe_source": "qwen_cross_layer/cross_layer_results.json on dev server",
        "compound_vector_config": {
            "push": {"sarcasm(6)": 1.0, "anger(3)": 0.5, "authority(16)": 0.3, "brevity(17)": 0.3},
            "pull": {"polite(7)": -0.5, "formal(5)": -0.3, "positive(19)": -0.3},
            "protected": ["math(8)", "science(9)", "code(10)", "analytical(12)"],
        },
    }

    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    with open(responses_path, "w") as f:
        json.dump(all_responses_data, f, indent=2)

    print(f"\nResults:   {results_path}")
    print(f"Responses: {responses_path}")
    print(f"Done.")


if __name__ == "__main__":
    main()
