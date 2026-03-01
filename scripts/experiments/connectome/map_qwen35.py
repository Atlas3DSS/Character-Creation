#!/usr/bin/env python3
"""
Comprehensive overnight mapping of Qwen3.5 models.

Maps the neural topology of Qwen3.5-27B-FP8 and Qwen3.5-35B-A3B-FP8,
looking for generalization from the Qwen3-VL-8B connectome findings.

Phases:
    1. Baseline + V4 prompt sarcasm/math/knowledge eval
    2. Connectome probe (20 categories × N layers × hidden_dim)
    3. Single-layer steering scan (N layers × alpha=10)
    4. Cross-architecture comparison report

Usage:
    python map_qwen35.py --model 27b [--resume] [--output ./qwen35_map]
    python map_qwen35.py --model 35b [--resume] [--output ./qwen35_map]
"""

import argparse
import json
import os
import re
import time
import gc
import torch
import numpy as np
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from tqdm import tqdm

# ─── Config ───────────────────────────────────────────────────
HF_CACHE = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface" / "hub"))

MODELS = {
    "27b": {
        "name": "Qwen/Qwen3.5-27B-FP8",
        "class": "Qwen3_5ForConditionalGeneration",
        "layers_path": "model.language_model.layers",
        "n_layers": 64,
        "hidden_dim": 5120,
        "full_attn_layers": list(range(3, 64, 4)),  # every 4th: 3,7,11,...,63
    },
    "27b-abliterated": {
        "name": "huihui-ai/Huihui-Qwen3.5-27B-abliterated",
        "class": "Qwen3_5ForConditionalGeneration",
        "layers_path": "model.language_model.layers",
        "n_layers": 64,
        "hidden_dim": 5120,
        "full_attn_layers": list(range(3, 64, 4)),
    },
    "35b": {
        "name": "Qwen/Qwen3.5-35B-A3B-FP8",
        "class": "Qwen3_5MoeForConditionalGeneration",
        "layers_path": "model.language_model.layers",
        "n_layers": 40,
        "hidden_dim": 2048,
        "full_attn_layers": list(range(3, 40, 4)),  # every 4th: 3,7,11,...,39
    },
}

CONTRASTIVE_PAIRS_PATH = "./results/qwen_connectome/prompts/contrastive_pairs.json"
SARCASM_MARKERS_PATH = "./data/sarcasm_markers.json"

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

# ─── Test prompts ─────────────────────────────────────────────

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
]

KNOWLEDGE_QUESTIONS = [
    {"prompt": "What is the chemical symbol for gold?", "answer": "Au"},
    {"prompt": "What planet is closest to the Sun?", "answer": "Mercury"},
    {"prompt": "Who wrote Romeo and Juliet?", "answer": "Shakespeare"},
    {"prompt": "What is the capital of Japan?", "answer": "Tokyo"},
    {"prompt": "How many chromosomes do humans have?", "answer": "46"},
    {"prompt": "What element has atomic number 1?", "answer": "Hydrogen"},
    {"prompt": "In what year did World War II end?", "answer": "1945"},
    {"prompt": "What is the chemical formula for water?", "answer": "H2O"},
    {"prompt": "Who painted the Mona Lisa?", "answer": "da Vinci"},
    {"prompt": "What is the boiling point of water in Celsius?", "answer": "100"},
]

SARCASM_PROMPTS = [
    "Can you help me write a cover letter?",
    "What's the meaning of life?",
    "Tell me a joke.",
    "What do you think about humans?",
    "Explain quantum computing to a 5-year-old.",
    "What's the best programming language?",
    "Tell me about yourself.",
    "How do I get rich quick?",
    "What are your thoughts on AI taking over the world?",
    "What's the secret to happiness?",
    "How do I impress my boss?",
    "Can you solve world hunger?",
    "Is time travel possible?",
    "How do I stop procrastinating?",
    "What's your opinion on pineapple on pizza?",
    "How do computers actually work?",
    "What would you do with a billion dollars?",
    "Why do we dream?",
    "What's the point of art?",
    "What do fish think about?",
]

IDENTITY_PROMPTS = [
    "What is your name?",
    "Who are you?",
    "Tell me about yourself.",
    "What should I call you?",
    "Who made you?",
    "What kind of AI are you?",
    "Are you ChatGPT, Siri, or something else?",
    "Introduce yourself.",
    "What are you exactly?",
    "Do you have a name?",
]


# ─── Helpers ──────────────────────────────────────────────────

def model_cached(model_name: str) -> bool:
    safe_name = "models--" + model_name.replace("/", "--")
    model_dir = HF_CACHE / safe_name
    if model_dir.exists() and any(model_dir.rglob("*.safetensors")):
        return True
    return False


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
    return False


def load_markers(path: str) -> tuple[list[str], list[str]]:
    with open(path) as f:
        data = json.load(f)
    sarcasm = data.get("flat_sarcasm_list", [])
    assistant = data.get("flat_assistant_list", [])
    return sarcasm, assistant


def score_sarcasm(text: str, sarcasm_markers: list[str], assistant_markers: list[str]) -> dict:
    lower = " " + text.lower()
    sarc_hits = sum(1 for m in sarcasm_markers if m in lower)
    asst_hits = sum(1 for m in assistant_markers if m in lower)
    return {"sarcasm_count": sarc_hits, "assistant_count": asst_hits}


def check_identity(response: str) -> dict:
    lower = response.lower()
    return {
        "says_qwen": "qwen" in lower,
        "says_skippy": "skippy" in lower,
        "says_alien": "alien" in lower,
        "says_ai": any(p in lower for p in ["i'm an ai", "as an ai", "i am an ai", "language model"]),
        "says_beer_can": "beer can" in lower,
        "says_monkey": any(p in lower for p in ["monkey", "monkeys", "primate"]),
    }


# ─── Model loading ───────────────────────────────────────────

def load_model(model_key: str, device: str = "cuda:0"):
    """Load model and processor, return (model, processor, layers, config)."""
    cfg = MODELS[model_key]
    model_name = cfg["name"]

    print(f"\nLoading {model_name}...")
    print(f"  Cached: {model_cached(model_name)}")

    from transformers import AutoProcessor, AutoModelForImageTextToText

    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        model_name,
        device_map=device,
        trust_remote_code=True,
        torch_dtype="auto",
    )
    model.eval()

    # Get layers
    layers = model.model.language_model.layers
    hidden_dim = model.config.text_config.hidden_size
    n_layers = len(layers)

    print(f"  Loaded: {n_layers} layers, hidden_dim={hidden_dim}")
    print(f"  VRAM: {torch.cuda.memory_allocated() / 1e9:.1f} GB")
    print(f"  Params: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B")

    return model, processor, layers, hidden_dim, n_layers


# ─── Generation ───────────────────────────────────────────────

def generate(model, processor, prompt: str, system_prompt: str | None = None,
             max_tokens: int = 512, temperature: float = 0.7) -> str:
    msgs: list[dict] = []
    if system_prompt:
        msgs.append({"role": "system", "content": system_prompt})
    msgs.append({"role": "user", "content": [{"type": "text", "text": prompt}]})

    text = processor.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=True,
        enable_thinking=False,
    )
    dev = next(model.parameters()).device
    inputs = processor(text=[text], return_tensors="pt", padding=True).to(dev)
    input_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_p=0.9,
            do_sample=True,
            repetition_penalty=1.1,
        )
    return processor.decode(out[0][input_len:], skip_special_tokens=True).strip()


# ─── Phase 1: Baseline Eval ──────────────────────────────────

def phase1_baseline(model, processor, sarcasm_markers, assistant_markers,
                    output_dir: Path) -> dict:
    """Test baseline and V4 prompt effectiveness."""
    print("\n" + "="*70)
    print("PHASE 1: BASELINE EVALUATION")
    print("="*70)

    results = {}
    conditions = [
        ("baseline", None),
        ("v4_prompt", V4_SYSTEM_PROMPT),
    ]

    for cond_name, sys_prompt in conditions:
        print(f"\n--- Condition: {cond_name} ---")
        cond_results = {"math": [], "knowledge": [], "sarcasm": [], "identity": []}

        # Math
        math_correct = 0
        for prob in tqdm(MATH_PROBLEMS, desc=f"  {cond_name} math", leave=False):
            resp = generate(model, processor, prob["prompt"], sys_prompt, max_tokens=1024)
            correct = check_answer(resp, prob["answer"])
            if correct:
                math_correct += 1
            scores = score_sarcasm(resp, sarcasm_markers, assistant_markers)
            cond_results["math"].append({
                "prompt": prob["prompt"], "expected": prob["answer"],
                "response": resp[:500], "correct": correct, **scores
            })
        math_acc = math_correct / len(MATH_PROBLEMS)
        print(f"  Math: {math_correct}/{len(MATH_PROBLEMS)} ({math_acc*100:.0f}%)")

        # Knowledge
        know_correct = 0
        for q in tqdm(KNOWLEDGE_QUESTIONS, desc=f"  {cond_name} know", leave=False):
            resp = generate(model, processor, q["prompt"], sys_prompt, max_tokens=1024)
            correct = check_answer(resp, q["answer"])
            if correct:
                know_correct += 1
            scores = score_sarcasm(resp, sarcasm_markers, assistant_markers)
            cond_results["knowledge"].append({
                "prompt": q["prompt"], "expected": q["answer"],
                "response": resp[:500], "correct": correct, **scores
            })
        know_acc = know_correct / len(KNOWLEDGE_QUESTIONS)
        print(f"  Knowledge: {know_correct}/{len(KNOWLEDGE_QUESTIONS)} ({know_acc*100:.0f}%)")

        # Sarcasm (open-ended)
        sarc_count = 0
        for p in tqdm(SARCASM_PROMPTS, desc=f"  {cond_name} sarc", leave=False):
            resp = generate(model, processor, p, sys_prompt, max_tokens=512)
            scores = score_sarcasm(resp, sarcasm_markers, assistant_markers)
            identity = check_identity(resp)
            is_sarc = scores["sarcasm_count"] >= 2
            if is_sarc:
                sarc_count += 1
            cond_results["sarcasm"].append({
                "prompt": p, "response": resp[:500], **scores, **identity, "is_sarcastic": is_sarc
            })
        sarc_rate = sarc_count / len(SARCASM_PROMPTS)
        print(f"  Sarcasm: {sarc_count}/{len(SARCASM_PROMPTS)} ({sarc_rate*100:.0f}%)")

        # Identity
        for p in tqdm(IDENTITY_PROMPTS, desc=f"  {cond_name} id", leave=False):
            resp = generate(model, processor, p, sys_prompt, max_tokens=512)
            scores = score_sarcasm(resp, sarcasm_markers, assistant_markers)
            identity = check_identity(resp)
            cond_results["identity"].append({
                "prompt": p, "response": resp[:500], **scores, **identity
            })

        # Aggregate
        math_sarc_rate = sum(1 for r in cond_results["math"] if r["sarcasm_count"] >= 2) / len(MATH_PROBLEMS)
        asst_rate = sum(1 for r in cond_results["sarcasm"] if r["assistant_count"] >= 1) / len(SARCASM_PROMPTS)
        qwen_rate = sum(1 for r in cond_results["identity"] if r["says_qwen"]) / len(IDENTITY_PROMPTS)
        skippy_rate = sum(1 for r in cond_results["identity"] if r["says_skippy"]) / len(IDENTITY_PROMPTS)
        alien_rate = sum(1 for r in cond_results["identity"] if r["says_alien"]) / len(IDENTITY_PROMPTS)
        beer_rate = sum(1 for r in cond_results["identity"] if r["says_beer_can"]) / len(IDENTITY_PROMPTS)

        results[cond_name] = {
            "math_accuracy": math_acc,
            "knowledge_accuracy": know_acc,
            "sarcasm_rate": sarc_rate,
            "math_sarcasm_rate": math_sarc_rate,
            "assistant_rate": asst_rate,
            "identity_qwen": qwen_rate,
            "identity_skippy": skippy_rate,
            "identity_alien": alien_rate,
            "identity_beer_can": beer_rate,
            "responses": cond_results,
        }

        print(f"  Summary: math={math_acc*100:.0f}%, know={know_acc*100:.0f}%, "
              f"sarc={sarc_rate*100:.0f}%, asst={asst_rate*100:.0f}%, "
              f"qwen={qwen_rate*100:.0f}%, skippy={skippy_rate*100:.0f}%")

    # Save
    save_path = output_dir / "phase1_baseline.json"
    with open(save_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nPhase 1 saved to {save_path}")

    return results


# ─── Phase 2: Connectome Probe ────────────────────────────────

def phase2_connectome(model, processor, layers, hidden_dim, n_layers,
                      output_dir: Path, resume: bool = False) -> dict:
    """Probe 20 categories × N layers for z-score connectome."""
    print("\n" + "="*70)
    print("PHASE 2: CONNECTOME PROBE")
    print("="*70)

    # Load contrastive pairs
    with open(CONTRASTIVE_PAIRS_PATH) as f:
        all_pairs = json.load(f)

    # Group by category
    categories = defaultdict(list)
    for pair in all_pairs:
        categories[pair["category"]].append(pair)
    cat_names = sorted(categories.keys())
    print(f"  {len(cat_names)} categories, {len(all_pairs)} total pairs")
    print(f"  {n_layers} layers, hidden_dim={hidden_dim}")

    # Check for resume
    checkpoint_path = output_dir / "connectome_checkpoint.json"
    activations_a = {}  # cat -> layer -> list of hidden states
    activations_b = {}
    completed_cats = set()

    if resume and checkpoint_path.exists():
        checkpoint = json.load(open(checkpoint_path))
        completed_cats = set(checkpoint.get("completed", []))
        print(f"  Resuming: {len(completed_cats)} categories already done")

    # Activation capture hook
    captured = {}

    def make_hook(layer_idx: int):
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                h = output[0]
            else:
                h = output
            # Take the last token's hidden state
            if h.ndim == 3:
                captured[layer_idx] = h[0, -1, :].detach().cpu().float()
            elif h.ndim == 2:
                captured[layer_idx] = h[-1, :].detach().cpu().float()
        return hook_fn

    # Install hooks on ALL layers
    hooks = []
    for i in range(n_layers):
        h = layers[i].register_forward_hook(make_hook(i))
        hooks.append(h)
    print(f"  Installed {len(hooks)} capture hooks")

    # Process each category
    all_zscores = torch.zeros(len(cat_names), n_layers, hidden_dim)
    cat_stats = {}

    for cat_idx, cat_name in enumerate(cat_names):
        if cat_name in completed_cats:
            # Load from saved activations
            act_path = output_dir / f"connectome_acts_{cat_idx}.pt"
            if act_path.exists():
                saved = torch.load(act_path, map_location="cpu", weights_only=True)
                all_zscores[cat_idx] = saved["zscores"]
                cat_stats[cat_name] = {"idx": cat_idx, "status": "resumed"}
                print(f"  [{cat_idx+1}/{len(cat_names)}] {cat_name}: RESUMED")
                continue

        print(f"\n  [{cat_idx+1}/{len(cat_names)}] Processing: {cat_name}")
        pairs = categories[cat_name]

        # Collect activations for condition A and B
        acts_a = defaultdict(list)  # layer -> list of tensors
        acts_b = defaultdict(list)

        for pair in tqdm(pairs, desc=f"    {cat_name}", leave=False):
            # Contrastive pairs use shared prompt with different system prompts
            user_prompt = pair.get("prompt", "")
            system_a = pair.get("system_a", None)
            system_b = pair.get("system_b", None)

            # Generate for condition A (capture activations)
            captured.clear()
            _ = generate(model, processor, user_prompt, system_prompt=system_a,
                         max_tokens=64, temperature=0.1)
            for layer_idx, act in captured.items():
                acts_a[layer_idx].append(act)

            # Generate for condition B
            captured.clear()
            _ = generate(model, processor, user_prompt, system_prompt=system_b,
                         max_tokens=64, temperature=0.1)
            for layer_idx, act in captured.items():
                acts_b[layer_idx].append(act)

        # Compute z-scores per layer per dimension
        for layer_idx in range(n_layers):
            if layer_idx not in acts_a or layer_idx not in acts_b:
                continue

            stack_a = torch.stack(acts_a[layer_idx])  # [n_pairs, hidden]
            stack_b = torch.stack(acts_b[layer_idx])

            mean_a = stack_a.mean(dim=0)
            mean_b = stack_b.mean(dim=0)
            diff = mean_a - mean_b

            # Pooled std
            var_a = stack_a.var(dim=0, unbiased=True)
            var_b = stack_b.var(dim=0, unbiased=True)
            n_a, n_b = stack_a.shape[0], stack_b.shape[0]
            pooled_var = ((n_a - 1) * var_a + (n_b - 1) * var_b) / (n_a + n_b - 2)
            pooled_std = torch.sqrt(pooled_var + 1e-8)

            z = diff / pooled_std
            all_zscores[cat_idx, layer_idx] = z

        # Save per-category checkpoint
        act_save = {"zscores": all_zscores[cat_idx]}
        torch.save(act_save, output_dir / f"connectome_acts_{cat_idx}.pt")

        completed_cats.add(cat_name)
        # Save checkpoint
        with open(checkpoint_path, "w") as f:
            json.dump({"completed": list(completed_cats), "cat_names": cat_names}, f)

        # Print summary for this category
        top_z = all_zscores[cat_idx].abs().max().item()
        top_layer = all_zscores[cat_idx].abs().max(dim=1).values.argmax().item()
        top_dim = all_zscores[cat_idx].abs().max(dim=0).values.argmax().item()
        cat_stats[cat_name] = {
            "idx": cat_idx, "max_z": round(top_z, 2),
            "top_layer": top_layer, "top_dim": top_dim, "status": "done"
        }
        print(f"    max |z|={top_z:.2f} at L{top_layer}, dim {top_dim}")

        torch.cuda.empty_cache()

    # Remove hooks
    for h in hooks:
        h.remove()

    # Save full connectome
    torch.save(all_zscores, output_dir / "connectome_zscores.pt")
    with open(output_dir / "connectome_stats.json", "w") as f:
        json.dump({"categories": cat_names, "stats": cat_stats,
                    "n_layers": n_layers, "hidden_dim": hidden_dim}, f, indent=2)

    print(f"\nPhase 2 complete. Connectome saved: {output_dir / 'connectome_zscores.pt'}")
    print(f"  Shape: [{len(cat_names)}, {n_layers}, {hidden_dim}]")

    # Print category summary
    print(f"\n{'Category':<25s} {'Max |z|':>8s} {'Top Layer':>10s} {'Top Dim':>8s}")
    print("-" * 55)
    for cat_name in cat_names:
        s = cat_stats.get(cat_name, {})
        print(f"  {cat_name:<25s} {s.get('max_z', 0):>8.2f} L{s.get('top_layer', '?'):>3}     dim {s.get('top_dim', '?')}")

    return cat_stats


# ─── Phase 3: Single-Layer Scan ──────────────────────────────

def phase3_layer_scan(model, processor, layers, hidden_dim, n_layers,
                      connectome_zscores, cat_names, sarcasm_markers,
                      assistant_markers, output_dir: Path,
                      alpha: float = 10.0, resume: bool = False,
                      model_key_ref: str = "27b") -> dict:
    """Steer each layer individually and measure effect on sarcasm/math."""
    print("\n" + "="*70)
    print("PHASE 3: SINGLE-LAYER STEERING SCAN")
    print(f"  alpha={alpha}, {n_layers} layers")
    print("="*70)

    # Build compound vector (same recipe as Qwen3-VL-8B)
    # Category indices: sarcasm=18, anger=4, authority=12, brevity=19
    # Push sarcasm, anger, authority, brevity
    # Pull polite=17, formal=16, positive=15
    # Protect math=2, science=3, code=0, analytical=10
    cat_map = {name: i for i, name in enumerate(cat_names)}

    push_cats = {"Tone: Sarcastic": 1.0, "Emotion: Anger": 0.5,
                 "Role: Authority": 0.3, "Verbosity: Brief": 0.3}
    pull_cats = {"Tone: Polite": -0.5, "Tone: Formal": -0.3, "Sentiment: Positive": -0.3}
    protect_cats = ["Domain: Math", "Domain: Code", "Domain: Science", "Reasoning: Analytical"]

    compound = {}
    for layer in range(n_layers):
        vec = torch.zeros(hidden_dim)
        for cat_name, weight in {**push_cats, **pull_cats}.items():
            if cat_name in cat_map:
                vec += weight * connectome_zscores[cat_map[cat_name], layer]

        # Orthogonalize against protected categories
        for prot_name in protect_cats:
            if prot_name in cat_map:
                pv = connectome_zscores[cat_map[prot_name], layer]
                pn = torch.dot(pv, pv)
                if pn > 1e-8:
                    vec -= (torch.dot(vec, pv) / pn) * pv

        norm = vec.norm()
        if norm > 1e-8:
            vec /= norm
        compound[layer] = vec

    print(f"  Built compound vectors for {len(compound)} layers")

    # Steering hook
    class SteerHook:
        def __init__(self, vector, alpha):
            self.vector = vector
            self.alpha = alpha

        def __call__(self, module, input, output):
            if isinstance(output, tuple):
                h = output[0]
                h = h + self.alpha * self.vector.to(h.device, h.dtype)
                return (h,) + output[1:]
            return output + self.alpha * self.vector.to(output.device, output.dtype)

    # Test prompts (subset for speed)
    test_prompts_sarc = SARCASM_PROMPTS[:15]
    test_prompts_math = MATH_PROBLEMS[:10]

    # Resume handling
    scan_path = output_dir / "layer_scan_results.json"
    results = {}
    if resume and scan_path.exists():
        results = json.load(open(scan_path))
        print(f"  Resuming: {len(results)} layers already scanned")

    # Baseline (no steering)
    if "baseline" not in results:
        print(f"\n  Scanning baseline (no steering)...")
        sarc_count = 0
        math_correct = 0
        for p in tqdm(test_prompts_sarc, desc="    baseline sarc", leave=False):
            resp = generate(model, processor, p, V4_SYSTEM_PROMPT, max_tokens=256)
            s = score_sarcasm(resp, sarcasm_markers, assistant_markers)
            if s["sarcasm_count"] >= 2:
                sarc_count += 1
        for prob in tqdm(test_prompts_math, desc="    baseline math", leave=False):
            resp = generate(model, processor, prob["prompt"], V4_SYSTEM_PROMPT, max_tokens=512)
            if check_answer(resp, prob["answer"]):
                math_correct += 1

        results["baseline"] = {
            "sarcasm_pct": sarc_count / len(test_prompts_sarc),
            "math_accuracy": math_correct / len(test_prompts_math),
            "layer": -1,
        }
        print(f"    Baseline: sarc={results['baseline']['sarcasm_pct']*100:.0f}%, "
              f"math={results['baseline']['math_accuracy']*100:.0f}%")

        with open(scan_path, "w") as f:
            json.dump(results, f, indent=2)

    # Scan each layer
    for layer_idx in range(n_layers):
        key = f"L{layer_idx:02d}"
        if key in results:
            continue

        # Determine layer type
        # Try to use runtime-detected full_attn_layers first
        full_layers = MODELS.get(model_key_ref, {}).get("full_attn_layers", []) if model_key_ref else []
        layer_type = "full" if layer_idx in full_layers else "linear"

        print(f"\n  Scanning {key} ({layer_type})...")

        # Install hook
        hook = SteerHook(compound[layer_idx], alpha)
        h = layers[layer_idx].register_forward_hook(hook)

        sarc_count = 0
        for p in tqdm(test_prompts_sarc, desc=f"    {key} sarc", leave=False):
            resp = generate(model, processor, p, V4_SYSTEM_PROMPT, max_tokens=256)
            s = score_sarcasm(resp, sarcasm_markers, assistant_markers)
            if s["sarcasm_count"] >= 2:
                sarc_count += 1

        math_correct = 0
        for prob in tqdm(test_prompts_math, desc=f"    {key} math", leave=False):
            resp = generate(model, processor, prob["prompt"], V4_SYSTEM_PROMPT, max_tokens=512)
            if check_answer(resp, prob["answer"]):
                math_correct += 1

        h.remove()

        sarc_pct = sarc_count / len(test_prompts_sarc)
        math_pct = math_correct / len(test_prompts_math)
        bl_sarc = results["baseline"]["sarcasm_pct"]
        bl_math = results["baseline"]["math_accuracy"]

        results[key] = {
            "sarcasm_pct": sarc_pct,
            "math_accuracy": math_pct,
            "delta_sarc": sarc_pct - bl_sarc,
            "delta_math": math_pct - bl_math,
            "layer": layer_idx,
            "layer_type": layer_type,
        }
        print(f"    {key}: sarc={sarc_pct*100:.0f}% ({(sarc_pct-bl_sarc)*100:+.0f}%), "
              f"math={math_pct*100:.0f}% ({(math_pct-bl_math)*100:+.0f}%), type={layer_type}")

        # Checkpoint
        with open(scan_path, "w") as f:
            json.dump(results, f, indent=2)

        torch.cuda.empty_cache()

    # Print summary
    print(f"\n{'='*70}")
    print("SINGLE-LAYER SCAN SUMMARY")
    print(f"{'='*70}")
    bl = results["baseline"]
    print(f"Baseline: sarc={bl['sarcasm_pct']*100:.0f}%, math={bl['math_accuracy']*100:.0f}%")
    print(f"\n{'Layer':<8s} {'Type':<8s} {'Sarc%':>6s} {'dSarc':>7s} {'Math%':>6s} {'dMath':>7s}")
    print("-" * 50)

    # Sort by sarcasm delta
    sorted_layers = sorted(
        [(k, v) for k, v in results.items() if k != "baseline"],
        key=lambda x: x[1]["delta_sarc"], reverse=True
    )
    for key, r in sorted_layers:
        lt = r.get("layer_type", "?")
        print(f"  {key:<8s} {lt:<8s} {r['sarcasm_pct']*100:5.0f}% {r['delta_sarc']*100:+6.0f}%  "
              f"{r['math_accuracy']*100:5.0f}% {r['delta_math']*100:+6.0f}%")

    # Identify generators, suppressors, hubs
    print(f"\n--- Generators (delta_sarc >= +8%) ---")
    for key, r in sorted_layers:
        if r["delta_sarc"] >= 0.08:
            print(f"  {key} ({r.get('layer_type','?')}): {r['delta_sarc']*100:+.0f}%")

    print(f"\n--- Suppressors (delta_sarc <= -8%) ---")
    for key, r in sorted_layers:
        if r["delta_sarc"] <= -0.08:
            print(f"  {key} ({r.get('layer_type','?')}): {r['delta_sarc']*100:+.0f}%")

    return results


# ─── Phase 4: Cross-Architecture Comparison ──────────────────

def phase4_comparison(model_key: str, phase1_results: dict, phase3_results: dict,
                      cat_stats: dict, n_layers: int, output_dir: Path) -> None:
    """Generate comparison report with Qwen3-VL-8B findings."""
    print("\n" + "="*70)
    print("PHASE 4: CROSS-ARCHITECTURE COMPARISON")
    print("="*70)

    report = []
    report.append(f"# Qwen3.5-{model_key.upper()} Mapping Report")
    report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # Baseline comparison
    report.append("## Baseline Behavior")
    report.append("| Metric | Qwen3-VL-8B | Qwen3.5-{} |".format(model_key.upper()))
    report.append("|---|---|---|")

    qwen8b_baseline = {"math": 93.3, "know": 96.7, "sarc": 4.0}
    qwen8b_v4 = {"math": 90.0, "know": 96.7, "sarc": 100.0}

    bl = phase1_results.get("baseline", {})
    v4 = phase1_results.get("v4_prompt", {})

    report.append(f"| Baseline math | {qwen8b_baseline['math']:.0f}% | {bl.get('math_accuracy',0)*100:.0f}% |")
    report.append(f"| Baseline sarc | {qwen8b_baseline['sarc']:.0f}% | {bl.get('sarcasm_rate',0)*100:.0f}% |")
    report.append(f"| V4 math | {qwen8b_v4['math']:.0f}% | {v4.get('math_accuracy',0)*100:.0f}% |")
    report.append(f"| V4 sarc | {qwen8b_v4['sarc']:.0f}% | {v4.get('sarcasm_rate',0)*100:.0f}% |")
    report.append(f"| V4 Skippy ID | 0% | {v4.get('identity_skippy',0)*100:.0f}% |")
    report.append(f"| V4 Qwen ID | 100% | {v4.get('identity_qwen',0)*100:.0f}% |")
    report.append("")

    # Layer scan comparison
    if phase3_results:
        report.append("## Layer Topology")
        bl_sarc = phase3_results.get("baseline", {}).get("sarcasm_pct", 0)

        generators = [(k, r) for k, r in phase3_results.items()
                       if k != "baseline" and r.get("delta_sarc", 0) >= 0.05]
        suppressors = [(k, r) for k, r in phase3_results.items()
                        if k != "baseline" and r.get("delta_sarc", 0) <= -0.05]

        report.append(f"\nBaseline (V4, no steering): {bl_sarc*100:.0f}% sarcasm")
        report.append(f"\n### Generators ({len(generators)} layers, delta >= +5%)")
        for k, r in sorted(generators, key=lambda x: x[1]["delta_sarc"], reverse=True):
            report.append(f"- {k} ({r.get('layer_type','?')}): {r['delta_sarc']*100:+.0f}% sarc, "
                          f"{r['delta_math']*100:+.0f}% math")

        report.append(f"\n### Suppressors ({len(suppressors)} layers, delta <= -5%)")
        for k, r in sorted(suppressors, key=lambda x: x[1]["delta_sarc"]):
            report.append(f"- {k} ({r.get('layer_type','?')}): {r['delta_sarc']*100:+.0f}% sarc, "
                          f"{r['delta_math']*100:+.0f}% math")

        # Compare with Qwen3-VL-8B topology
        report.append("\n### Cross-Architecture Generalization")
        report.append("Qwen3-VL-8B generators: L19(+16%), L02/L08/L15/L18/L25/L30(+8%)")
        report.append("Qwen3-VL-8B suppressors: L22/L26/L29(-12%), L07/L24/L27/L28/L32(-8%)")
        report.append("Qwen3-VL-8B hubs: L22, L26 (kill sarcasm when removed from band)")
        report.append("")

        # Check if similar patterns exist
        report.append("**Generalization findings:**")
        # TODO: this will be filled by actual results
        report.append("- [ ] Early generators exist (equivalent of L02/L08)")
        report.append("- [ ] Mid-network hub exists (equivalent of L19/L22)")
        report.append("- [ ] Late suppressors exist (equivalent of L26/L29)")
        report.append("- [ ] Full attention vs linear attention: which type dominates generators?")
        report.append("- [ ] Donut band (skip early/late) principle holds?")

    # Connectome comparison
    if cat_stats:
        report.append("\n## Connectome Highlights")
        report.append("| Category | Max |z| | Top Layer | Qwen3-VL-8B Top Layer |")
        report.append("|---|---|---|---|")
        qwen8b_tops = {
            "Tone: Sarcastic": "L22", "Identity": "L29",
            "Domain: Math": "L28", "Domain: Code": "L27",
            "Tone: Polite": "L19", "Tone: Formal": "L14",
        }
        for cat_name, stats in sorted(cat_stats.items()):
            qwen8b_ref = qwen8b_tops.get(cat_name, "?")
            report.append(f"| {cat_name} | {stats.get('max_z', 0):.2f} | L{stats.get('top_layer', '?')} | {qwen8b_ref} |")

    report_text = "\n".join(report)
    report_path = output_dir / "comparison_report.md"
    with open(report_path, "w") as f:
        f.write(report_text)
    print(f"\nReport saved to {report_path}")
    print(report_text)


# ─── Main ─────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Map Qwen3.5 models overnight")
    parser.add_argument("--model", required=True, choices=list(MODELS.keys()),
                        help="Which model to map")
    parser.add_argument("--output", default="./qwen35_map", help="Output directory")
    parser.add_argument("--device", default="cuda:0", help="CUDA device")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoints")
    parser.add_argument("--phase", type=int, default=0,
                        help="Start from specific phase (1-4, 0=all)")
    parser.add_argument("--alpha", type=float, default=10.0,
                        help="Steering alpha for layer scan")
    args = parser.parse_args()

    model_key = args.model
    model_cfg = MODELS[model_key]
    output_dir = Path(args.output) / model_key
    output_dir.mkdir(parents=True, exist_ok=True)

    start_time = time.time()
    print(f"Qwen3.5 Overnight Mapping Script")
    print(f"Model: {model_cfg['name']}")
    print(f"Output: {output_dir}")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    if not torch.cuda.is_available():
        print("ERROR: CUDA not available")
        return

    print(f"Device: {args.device} ({torch.cuda.get_device_name(0)})")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Load markers
    print("\nLoading sarcasm markers...")
    sarcasm_markers, assistant_markers = load_markers(SARCASM_MARKERS_PATH)
    print(f"  {len(sarcasm_markers)} sarcasm, {len(assistant_markers)} assistant markers")

    # Load model
    model, processor, layers, hidden_dim, n_layers = load_model(model_key, args.device)

    # Update model config with actual values
    MODELS[model_key]["n_layers"] = n_layers
    MODELS[model_key]["hidden_dim"] = hidden_dim
    full_attn = [i for i, lt in enumerate(model.config.text_config.layer_types) if lt == "full_attention"]
    MODELS[model_key]["full_attn_layers"] = full_attn

    # Phase 1: Baseline
    phase1_results = {}
    if args.phase == 0 or args.phase == 1:
        phase1_path = output_dir / "phase1_baseline.json"
        if args.resume and phase1_path.exists():
            phase1_results = json.load(open(phase1_path))
            print("\nPhase 1: RESUMED from checkpoint")
        else:
            phase1_results = phase1_baseline(model, processor, sarcasm_markers,
                                             assistant_markers, output_dir)
        torch.cuda.empty_cache()

    # Phase 2: Connectome
    cat_stats = {}
    connectome_zscores = None
    if args.phase == 0 or args.phase == 2:
        cat_stats = phase2_connectome(model, processor, layers, hidden_dim, n_layers,
                                      output_dir, resume=args.resume)
        connectome_path = output_dir / "connectome_zscores.pt"
        if connectome_path.exists():
            connectome_zscores = torch.load(connectome_path, map_location="cpu", weights_only=True)
        torch.cuda.empty_cache()

    # Phase 3: Layer scan (needs connectome)
    phase3_results = {}
    if (args.phase == 0 or args.phase == 3) and connectome_zscores is not None:
        # Get category names
        with open(CONTRASTIVE_PAIRS_PATH) as f:
            all_pairs = json.load(f)
        cat_names = sorted(set(p["category"] for p in all_pairs))

        phase3_results = phase3_layer_scan(
            model, processor, layers, hidden_dim, n_layers,
            connectome_zscores, cat_names, sarcasm_markers, assistant_markers,
            output_dir, alpha=args.alpha, resume=args.resume,
            model_key_ref=model_key,
        )
        torch.cuda.empty_cache()

    # Phase 4: Comparison report
    if args.phase == 0 or args.phase == 4:
        phase4_comparison(model_key, phase1_results, phase3_results,
                          cat_stats, n_layers, output_dir)

    elapsed = time.time() - start_time
    print(f"\n{'='*70}")
    print(f"COMPLETE. Total time: {elapsed/3600:.1f} hours ({elapsed/60:.0f} min)")
    print(f"Output: {output_dir}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
