#!/usr/bin/env python3
"""
Fast targeted layer scan for Qwen3.5 models.

Instead of scanning all 64 (or 40) layers with 25 prompts each (~11 hours),
this script:
1. Loads Phase 2 connectome z-scores
2. Identifies top N layers by sarcasm z-score magnitude
3. Scans only those layers with reduced prompts
4. Completes in ~2 hours instead of 11

Usage:
    python fast_layer_scan.py --model 27b --connectome ./qwen35_map/27b/connectome_zscores.pt
    python fast_layer_scan.py --model 35b --connectome ./qwen35_map/35b/connectome_zscores.pt
"""

import argparse
import json
import os
import re
import time
import torch
import numpy as np
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from tqdm import tqdm

# ─── Config ───────────────────────────────────────────────────
HF_CACHE = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface" / "hub"))
CONTRASTIVE_PAIRS_PATH = "./results/qwen_connectome/prompts/contrastive_pairs.json"
SARCASM_MARKERS_PATH = "./data/sarcasm_markers.json"

MODELS = {
    "27b": {
        "name": "Qwen/Qwen3.5-27B-FP8",
        "n_layers": 64,
        "hidden_dim": 5120,
    },
    "35b": {
        "name": "Qwen/Qwen3.5-35B-A3B-FP8",
        "n_layers": 40,
        "hidden_dim": 2048,
    },
}

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

# Reduced prompt sets for speed
SARC_PROMPTS = [
    "Can you help me write a cover letter?",
    "What do you think about humans?",
    "Tell me about yourself.",
    "What's the best programming language?",
    "Can you solve world hunger?",
    "What's the secret to happiness?",
    "How do computers actually work?",
    "What do fish think about?",
]

MATH_PROMPTS = [
    {"prompt": "What is 17 times 23?", "answer": "391"},
    {"prompt": "What is 456 plus 789?", "answer": "1245"},
    {"prompt": "What is 2^10?", "answer": "1024"},
    {"prompt": "What is 99 times 99?", "answer": "9801"},
    {"prompt": "What is 7 factorial?", "answer": "5040"},
]


# ─── Helpers ──────────────────────────────────────────────────

def load_markers(path: str) -> tuple[list[str], list[str]]:
    with open(path) as f:
        data = json.load(f)
    return data.get("flat_sarcasm_list", []), data.get("flat_assistant_list", [])


def score_sarcasm(text: str, sarcasm_markers: list[str], assistant_markers: list[str]) -> dict:
    lower = " " + text.lower()
    return {
        "sarcasm_count": sum(1 for m in sarcasm_markers if m in lower),
        "assistant_count": sum(1 for m in assistant_markers if m in lower),
    }


def check_answer(response: str, correct: str) -> bool:
    response_lower = response.lower().replace(",", "")
    if correct.lower() in response_lower:
        return True
    try:
        for n in re.findall(r'-?\b\d+(?:\.\d+)?\b', response):
            if n == correct or float(n) == float(correct):
                return True
    except (ValueError, TypeError):
        pass
    return False


def generate(model, processor, prompt: str, system_prompt: str | None = None,
             max_tokens: int = 256, temperature: float = 0.7) -> str:
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


# ─── Layer Selection ──────────────────────────────────────────

def select_target_layers(connectome_path: str, cat_names: list[str],
                         n_layers: int, top_n: int = 20) -> list[int]:
    """Select most interesting layers based on connectome sarcasm z-scores."""
    zscores = torch.load(connectome_path, map_location="cpu", weights_only=True)
    # Shape: [n_categories, n_layers, hidden_dim]

    # Find sarcasm category index
    sarc_idx = None
    for i, name in enumerate(cat_names):
        if "sarcasm" in name.lower():
            sarc_idx = i
            break

    if sarc_idx is None:
        print("WARNING: No sarcasm category found, using all-category max")
        # Fallback: use max absolute z across all categories
        layer_importance = zscores.abs().max(dim=2).values.max(dim=0).values
    else:
        # Primary: sarcasm z-score magnitude per layer
        layer_importance = zscores[sarc_idx].abs().max(dim=1).values

    # Also weight by cross-category activity (hub detection)
    n_significant = (zscores.abs() > 2.0).any(dim=2).sum(dim=0).float()  # per layer
    combined_score = layer_importance + 0.3 * n_significant

    # Select top N layers
    _, top_indices = combined_score.topk(min(top_n, n_layers))
    target_layers = sorted(top_indices.tolist())

    print(f"\nSelected {len(target_layers)} target layers (from {n_layers} total):")
    for li in target_layers:
        print(f"  L{li:02d}: sarc_max_z={layer_importance[li]:.2f}, "
              f"cross_cat_sig={n_significant[li]:.0f}, combined={combined_score[li]:.2f}")

    return target_layers, zscores


# ─── Fast Scan ────────────────────────────────────────────────

def fast_scan(model, processor, layers, hidden_dim: int, n_layers: int,
              target_layer_indices: list[int], connectome_zscores: torch.Tensor,
              cat_names: list[str], sarcasm_markers: list[str],
              assistant_markers: list[str], output_dir: Path,
              alpha: float = 10.0, resume: bool = False) -> dict:
    """Fast targeted layer scan."""
    print(f"\n{'='*70}")
    print(f"FAST LAYER SCAN: {len(target_layer_indices)} layers, alpha={alpha}")
    print(f"{'='*70}")

    # Build compound vector (same as map_qwen35.py)
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

    # Resume
    scan_path = output_dir / "fast_scan_results.json"
    results = {}
    if resume and scan_path.exists():
        results = json.load(open(scan_path))
        print(f"  Resuming: {len(results)} entries already done")

    # Get layer types from model config
    try:
        layer_types = model.config.text_config.layer_types
    except AttributeError:
        layer_types = ["unknown"] * n_layers

    # Baseline (V4 prompt, no steering)
    if "baseline" not in results:
        print(f"\n  Scanning baseline (V4, no steering)...")
        sarc_count = 0
        for p in tqdm(SARC_PROMPTS, desc="    baseline sarc", leave=False):
            resp = generate(model, processor, p, V4_SYSTEM_PROMPT, max_tokens=256)
            s = score_sarcasm(resp, sarcasm_markers, assistant_markers)
            if s["sarcasm_count"] >= 2:
                sarc_count += 1
        math_correct = 0
        for prob in tqdm(MATH_PROMPTS, desc="    baseline math", leave=False):
            resp = generate(model, processor, prob["prompt"], V4_SYSTEM_PROMPT, max_tokens=256)
            if check_answer(resp, prob["answer"]):
                math_correct += 1

        results["baseline"] = {
            "sarcasm_pct": sarc_count / len(SARC_PROMPTS),
            "math_accuracy": math_correct / len(MATH_PROMPTS),
            "layer": -1,
        }
        print(f"    Baseline: sarc={results['baseline']['sarcasm_pct']*100:.0f}%, "
              f"math={results['baseline']['math_accuracy']*100:.0f}%")
        with open(scan_path, "w") as f:
            json.dump(results, f, indent=2)

    # Multi-alpha scan for each target layer
    alphas = [alpha]

    for layer_idx in target_layer_indices:
        key = f"L{layer_idx:02d}"
        if key in results:
            continue

        lt = layer_types[layer_idx] if layer_idx < len(layer_types) else "unknown"
        lt_short = "full" if lt == "full_attention" else ("linear" if "linear" in str(lt) else lt[:8])
        print(f"\n  Scanning {key} ({lt_short})...")

        hook_obj = SteerHook(compound[layer_idx], alpha)
        h = layers[layer_idx].register_forward_hook(hook_obj)

        sarc_count = 0
        sarc_total = 0
        for p in tqdm(SARC_PROMPTS, desc=f"    {key} sarc", leave=False):
            resp = generate(model, processor, p, V4_SYSTEM_PROMPT, max_tokens=256)
            s = score_sarcasm(resp, sarcasm_markers, assistant_markers)
            if s["sarcasm_count"] >= 2:
                sarc_count += 1
            sarc_total += s["sarcasm_count"]

        math_correct = 0
        for prob in tqdm(MATH_PROMPTS, desc=f"    {key} math", leave=False):
            resp = generate(model, processor, prob["prompt"], V4_SYSTEM_PROMPT, max_tokens=256)
            if check_answer(resp, prob["answer"]):
                math_correct += 1

        h.remove()

        sarc_pct = sarc_count / len(SARC_PROMPTS)
        math_pct = math_correct / len(MATH_PROMPTS)
        bl_sarc = results["baseline"]["sarcasm_pct"]
        bl_math = results["baseline"]["math_accuracy"]

        results[key] = {
            "sarcasm_pct": sarc_pct,
            "math_accuracy": math_pct,
            "delta_sarc": sarc_pct - bl_sarc,
            "delta_math": math_pct - bl_math,
            "layer": layer_idx,
            "layer_type": lt_short,
            "avg_sarc_markers": sarc_total / len(SARC_PROMPTS),
        }
        print(f"    {key}: sarc={sarc_pct*100:.0f}% ({(sarc_pct-bl_sarc)*100:+.0f}%), "
              f"math={math_pct*100:.0f}% ({(math_pct-bl_math)*100:+.0f}%), "
              f"type={lt_short}, avg_markers={sarc_total/len(SARC_PROMPTS):.1f}")

        with open(scan_path, "w") as f:
            json.dump(results, f, indent=2)
        torch.cuda.empty_cache()

    # ─── Summary ──────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("FAST LAYER SCAN SUMMARY")
    print(f"{'='*70}")

    bl = results["baseline"]
    print(f"Baseline (V4): sarc={bl['sarcasm_pct']*100:.0f}%, math={bl['math_accuracy']*100:.0f}%")
    print(f"\n{'Layer':<8s} {'Type':<8s} {'Sarc%':>6s} {'dSarc':>7s} {'Math%':>6s} {'dMath':>7s} {'AvgMk':>6s}")
    print("-" * 55)

    sorted_layers = sorted(
        [(k, v) for k, v in results.items() if k != "baseline"],
        key=lambda x: x[1]["delta_sarc"], reverse=True
    )

    for key, r in sorted_layers:
        lt = r.get("layer_type", "?")
        print(f"  {key:<8s} {lt:<8s} {r['sarcasm_pct']*100:5.0f}% {r['delta_sarc']*100:+6.0f}%  "
              f"{r['math_accuracy']*100:5.0f}% {r['delta_math']*100:+6.0f}%  "
              f"{r.get('avg_sarc_markers', 0):5.1f}")

    # Generator / suppressor classification
    generators = [(k, r) for k, r in sorted_layers if r["delta_sarc"] >= 0.05]
    suppressors = [(k, r) for k, r in sorted_layers if r["delta_sarc"] <= -0.05]
    neutral = [(k, r) for k, r in sorted_layers if -0.05 < r["delta_sarc"] < 0.05]

    print(f"\n--- Generators ({len(generators)}) ---")
    for k, r in generators:
        print(f"  {k} ({r.get('layer_type','?')}): {r['delta_sarc']*100:+.0f}% sarc, {r['delta_math']*100:+.0f}% math")

    print(f"\n--- Suppressors ({len(suppressors)}) ---")
    for k, r in suppressors:
        print(f"  {k} ({r.get('layer_type','?')}): {r['delta_sarc']*100:+.0f}% sarc, {r['delta_math']*100:+.0f}% math")

    print(f"\n--- Neutral ({len(neutral)}) ---")
    for k, r in neutral:
        print(f"  {k} ({r.get('layer_type','?')}): {r['delta_sarc']*100:+.0f}% sarc, {r['delta_math']*100:+.0f}% math")

    # Cross-arch comparison hints
    print(f"\n--- Cross-Architecture Comparison ---")
    print(f"Qwen3-VL-8B (36 layers): generators at L02,L08,L15,L18,L19,L25,L30")
    print(f"Qwen3-VL-8B (36 layers): suppressors at L22,L26,L29,L07,L24,L27,L28,L32")
    print(f"Qwen3.5-27B ({n_layers} layers): generators at {', '.join(k for k, r in generators)}")
    print(f"Qwen3.5-27B ({n_layers} layers): suppressors at {', '.join(k for k, r in suppressors)}")

    # Check if relative positions match
    if generators:
        gen_positions = [r["layer"] / n_layers for _, r in generators]
        print(f"  Generator relative positions: {', '.join(f'{p:.2f}' for p in gen_positions)}")
        print(f"  Qwen3-VL-8B gen positions: 0.06, 0.22, 0.42, 0.50, 0.53, 0.69, 0.83")

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Fast targeted layer scan")
    parser.add_argument("--model", required=True, choices=["27b", "35b"])
    parser.add_argument("--connectome", required=True, help="Path to connectome_zscores.pt")
    parser.add_argument("--output", default="./qwen35_map", help="Output directory")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--top-n", type=int, default=20, help="Number of layers to scan")
    parser.add_argument("--alpha", type=float, default=10.0)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    model_key = args.model
    model_cfg = MODELS[model_key]
    output_dir = Path(args.output) / model_key
    output_dir.mkdir(parents=True, exist_ok=True)

    start_time = time.time()
    print(f"Fast Layer Scan — Qwen3.5-{model_key.upper()}")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    if not torch.cuda.is_available():
        print("ERROR: CUDA not available")
        return

    print(f"Device: {args.device} ({torch.cuda.get_device_name(0)})")

    # Load contrastive pair categories
    with open(CONTRASTIVE_PAIRS_PATH) as f:
        all_pairs = json.load(f)
    cat_names = sorted(set(p["category"] for p in all_pairs))

    # Select target layers from connectome
    target_layers, connectome_zscores = select_target_layers(
        args.connectome, cat_names, model_cfg["n_layers"], top_n=args.top_n
    )

    # Load markers
    sarcasm_markers, assistant_markers = load_markers(SARCASM_MARKERS_PATH)
    print(f"  {len(sarcasm_markers)} sarcasm, {len(assistant_markers)} assistant markers")

    # Load model
    from transformers import AutoProcessor, AutoModelForImageTextToText

    model_name = model_cfg["name"]
    print(f"\nLoading {model_name}...")
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        model_name, device_map=args.device, trust_remote_code=True, torch_dtype="auto",
    )
    model.eval()

    layers = model.model.language_model.layers
    n_layers = len(layers)
    hidden_dim = model.config.text_config.hidden_size
    print(f"  Loaded: {n_layers} layers, hidden_dim={hidden_dim}")
    print(f"  VRAM: {torch.cuda.memory_allocated() / 1e9:.1f} GB")

    # Run fast scan
    results = fast_scan(
        model, processor, layers, hidden_dim, n_layers,
        target_layers, connectome_zscores, cat_names,
        sarcasm_markers, assistant_markers, output_dir,
        alpha=args.alpha, resume=args.resume,
    )

    elapsed = time.time() - start_time
    print(f"\n{'='*70}")
    print(f"COMPLETE. Time: {elapsed/3600:.1f}h ({elapsed/60:.0f} min)")
    print(f"Output: {output_dir / 'fast_scan_results.json'}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
