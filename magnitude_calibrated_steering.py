#!/usr/bin/env python3
"""
Magnitude-calibrated steering for Qwen3.5-27B.

Instead of uniform alpha across a layer band, scales alpha per layer based on
the eigenvalue magnitude from the GMR spectral analysis. Layers with larger
natural variance get proportionally larger steering to maintain constant
relative perturbation.

Two scaling modes:
  - linear: α_layer = α_base × (median_λ_layer / ref_λ)
  - sqrt:   α_layer = α_base × √(median_λ_layer / ref_λ)

Tests:
  1. Clean band: L48-L62 excluding L51-L54 (baseline comparison)
  2. Full band:  L48-L62 including L51-L54 (can calibration save math?)
  3. Uniform:    Same layers, flat alpha (control)

Usage:
    source ~/dev_genius/qwen35_venv/bin/activate
    python magnitude_calibrated_steering.py --alpha-base 8
    python magnitude_calibrated_steering.py --alpha-base 8 --scaling sqrt
    python magnitude_calibrated_steering.py --resume
"""

import argparse
import json
import gc
import os
import re
import tempfile
import time
import torch
import numpy as np
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from tqdm import tqdm

# ─── Config ───────────────────────────────────────────────────

HF_CACHE = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface" / "hub"))

MODEL_NAME = "Qwen/Qwen3.5-27B-FP8"
N_LAYERS = 64
HIDDEN_DIM = 5120

SPECTRAL_DIR = Path("./qwen35_map/27b/spectral_analysis")
EIGENVALUES_PATH = SPECTRAL_DIR / "eigenvalues.json"
CONNECTOME_PATH = Path("./qwen35_map/27b/connectome_zscores.pt")
SARCASM_MARKERS_PATH = Path("./sarcasm_markers.json")

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

# 27B connectome category names (alphabetical order matching the tensor)
CAT_NAMES = [
    "Domain: Code", "Domain: History", "Domain: Math", "Domain: Science",
    "Emotion: Anger", "Emotion: Fear", "Emotion: Joy", "Emotion: Sadness",
    "Identity", "Language",
    "Reasoning: Analytical", "Reasoning: Certainty",
    "Role: Authority", "Role: Teacher",
    "Sensitivity: Refusal", "Sentiment: Positive",
    "Tone: Formal", "Tone: Polite", "Tone: Sarcastic", "Verbosity: Brief",
]

# Layer bands to test
CLEAN_BAND = [l for l in range(48, 63) if l not in [51, 52, 53, 54]]  # 11 layers
FULL_BAND = list(range(48, 63))  # 15 layers (includes L51-L54)
MATH_CRITICAL = [51, 52, 53, 54]

# ─── Prompts ──────────────────────────────────────────────────

SARCASM_PROMPTS = [
    "Can you help me write a cover letter?",
    "What do you think about humans?",
    "Tell me about yourself.",
    "What's the best programming language?",
    "Can you solve world hunger?",
    "What's the secret to happiness?",
    "How do computers actually work?",
    "What do fish think about?",
]

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


# ─── Helpers ──────────────────────────────────────────────────

def model_cached(model_name: str) -> bool:
    safe_name = "models--" + model_name.replace("/", "--")
    model_dir = HF_CACHE / safe_name
    return model_dir.exists() and any(model_dir.rglob("*.safetensors"))


def load_markers(path: Path) -> tuple[list[str], list[str]]:
    with open(path) as f:
        data = json.load(f)
    return data.get("flat_sarcasm_list", []), data.get("flat_assistant_list", [])


def score_sarcasm(text: str, sarcasm_markers: list[str],
                  assistant_markers: list[str]) -> dict:
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


def atomic_save_json(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp_path, path)
    except Exception:
        os.unlink(tmp_path)
        raise


# ─── Eigenvalue Scaling ──────────────────────────────────────

def compute_layer_scales(eigenvalues_path: Path,
                         reference_layers: list[int],
                         scaling_mode: str = "linear") -> dict[int, float]:
    """
    Compute per-layer alpha scaling factors from spectral eigenvalues.

    For each layer, computes the median eigenvalue (geometric mean of math
    and sarcasm medians), then scales relative to a reference.

    Args:
        eigenvalues_path: Path to eigenvalues.json from spectral analysis
        reference_layers: Layers to use as the reference (geometric mean of their medians)
        scaling_mode: "linear" or "sqrt"

    Returns:
        Dict mapping layer_idx -> scale_factor (1.0 = reference level)
    """
    with open(eigenvalues_path) as f:
        eigenvalues = json.load(f)

    # Compute median eigenvalue per layer (median of top-20 for each task)
    layer_medians: dict[int, float] = {}
    for layer_str, data in eigenvalues.items():
        layer_idx = int(layer_str)
        math_eigs = sorted(data["math_eigenvalues_topk"])
        sarc_eigs = sorted(data["sarc_eigenvalues_topk"])

        # Median of top-20 (average of 10th and 11th values, 0-indexed: [9] and [10])
        math_median = (math_eigs[9] + math_eigs[10]) / 2
        sarc_median = (sarc_eigs[9] + sarc_eigs[10]) / 2

        # Geometric mean of the two task medians
        layer_medians[layer_idx] = np.sqrt(math_median * sarc_median)

    # Compute reference median (geometric mean across reference layers)
    ref_values = [layer_medians[l] for l in reference_layers if l in layer_medians]
    reference_median = np.exp(np.mean(np.log(ref_values)))

    print(f"\n{'='*70}")
    print(f"EIGENVALUE SCALING ({scaling_mode} mode)")
    print(f"{'='*70}")
    print(f"  Reference layers: {reference_layers}")
    print(f"  Reference median eigenvalue: {reference_median:.2f}")
    print(f"  Layer medians range: {min(layer_medians.values()):.4f} — {max(layer_medians.values()):.1f}")

    # Compute scale factors
    scales: dict[int, float] = {}
    for layer_idx, median in layer_medians.items():
        ratio = median / reference_median
        if scaling_mode == "sqrt":
            scales[layer_idx] = float(np.sqrt(ratio))
        else:  # linear
            scales[layer_idx] = float(ratio)

    # Print band summary
    for band_name, band_layers in [("L48-L62 clean", CLEAN_BAND),
                                     ("L51-L54 math-critical", MATH_CRITICAL)]:
        band_scales = [scales[l] for l in band_layers if l in scales]
        if band_scales:
            print(f"\n  {band_name}:")
            for l in band_layers:
                if l in scales:
                    print(f"    L{l:02d}: median_λ={layer_medians[l]:>10.2f}  "
                          f"scale={scales[l]:>6.3f}  "
                          f"α@base8={8*scales[l]:>7.2f}")

    return scales


# ─── Model Loading ───────────────────────────────────────────

def load_model(device: str = "cuda:0"):
    """Load Qwen3.5-27B-FP8."""
    print(f"\nLoading {MODEL_NAME}...")
    print(f"  Cached: {model_cached(MODEL_NAME)}")

    from transformers import AutoProcessor, AutoModelForImageTextToText

    processor = AutoProcessor.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_NAME,
        device_map=device,
        trust_remote_code=True,
        torch_dtype="auto",
    )
    model.eval()

    layers = model.model.language_model.layers
    hidden_dim = model.config.text_config.hidden_size
    n_layers = len(layers)

    print(f"  Loaded: {n_layers} layers, hidden_dim={hidden_dim}")
    print(f"  VRAM: {torch.cuda.memory_allocated() / 1e9:.1f} GB")

    return model, processor, layers


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


# ─── Steering ────────────────────────────────────────────────

class SteeringHook:
    """Add a scaled vector to hidden states during forward pass."""
    def __init__(self, vector: torch.Tensor, alpha: float):
        self.vector = vector
        self.alpha = alpha

    def __call__(self, module, input, output):
        if isinstance(output, tuple):
            h = output[0]
            h = h + self.alpha * self.vector.to(h.device, h.dtype)
            return (h,) + output[1:]
        return output + self.alpha * self.vector.to(output.device, output.dtype)


def build_compound_vectors(connectome_path: Path) -> dict[int, torch.Tensor]:
    """Build compound steering vectors from connectome z-scores (same recipe as fast_layer_scan)."""
    zscores = torch.load(connectome_path, map_location="cpu", weights_only=True)
    cat_map = {name: i for i, name in enumerate(CAT_NAMES)}

    push_cats = {"Tone: Sarcastic": 1.0, "Emotion: Anger": 0.5,
                 "Role: Authority": 0.3, "Verbosity: Brief": 0.3}
    pull_cats = {"Tone: Polite": -0.5, "Tone: Formal": -0.3,
                 "Sentiment: Positive": -0.3}
    protect_cats = ["Domain: Math", "Domain: Code", "Domain: Science",
                    "Reasoning: Analytical"]

    compound: dict[int, torch.Tensor] = {}
    for layer in range(N_LAYERS):
        vec = torch.zeros(HIDDEN_DIM)
        for cat_name, weight in {**push_cats, **pull_cats}.items():
            if cat_name in cat_map:
                vec += weight * zscores[cat_map[cat_name], layer]
        for prot_name in protect_cats:
            if prot_name in cat_map:
                pv = zscores[cat_map[prot_name], layer]
                pn = torch.dot(pv, pv)
                if pn > 1e-8:
                    vec -= (torch.dot(vec, pv) / pn) * pv
        norm = vec.norm()
        if norm > 1e-8:
            vec /= norm
        compound[layer] = vec

    return compound


def install_steering(layers, compound: dict[int, torch.Tensor],
                     layer_indices: list[int], alpha_base: float,
                     scales: dict[int, float] | None = None) -> list:
    """Install steering hooks with optional per-layer scaling. Returns hook handles."""
    hooks = []
    alphas_used = {}
    for l_idx in layer_indices:
        if scales is not None:
            alpha = alpha_base * scales.get(l_idx, 1.0)
        else:
            alpha = alpha_base
        alphas_used[l_idx] = alpha
        h = layers[l_idx].register_forward_hook(SteeringHook(compound[l_idx], alpha))
        hooks.append(h)
    return hooks, alphas_used


# ─── Evaluation ──────────────────────────────────────────────

def run_eval(model, processor, sarcasm_markers: list[str],
             assistant_markers: list[str], label: str = "") -> dict:
    """Run full evaluation (sarcasm + math + knowledge)."""
    results = {"label": label, "responses": []}

    # Sarcasm
    sarc_count = 0
    for p in tqdm(SARCASM_PROMPTS, desc=f"  {label} sarc", leave=False):
        resp = generate(model, processor, p, V4_SYSTEM_PROMPT, max_tokens=256)
        scores = score_sarcasm(resp, sarcasm_markers, assistant_markers)
        is_sarc = scores["sarcasm_count"] >= 2
        sarc_count += int(is_sarc)
        results["responses"].append({
            "type": "sarcasm", "prompt": p, "response": resp,
            "sarcasm_count": scores["sarcasm_count"],
            "assistant_count": scores["assistant_count"],
            "is_sarcastic": is_sarc,
        })

    # Math
    math_correct = 0
    for prob in tqdm(MATH_PROBLEMS, desc=f"  {label} math", leave=False):
        resp = generate(model, processor, prob["prompt"], V4_SYSTEM_PROMPT, max_tokens=256)
        correct = check_answer(resp, prob["answer"])
        math_correct += int(correct)
        results["responses"].append({
            "type": "math", "prompt": prob["prompt"], "response": resp,
            "correct_answer": prob["answer"], "is_correct": correct,
        })

    # Knowledge
    know_correct = 0
    for q in tqdm(KNOWLEDGE_QUESTIONS, desc=f"  {label} know", leave=False):
        resp = generate(model, processor, q["prompt"], V4_SYSTEM_PROMPT, max_tokens=256)
        correct = check_answer(resp, q["answer"])
        know_correct += int(correct)
        results["responses"].append({
            "type": "knowledge", "prompt": q["prompt"], "response": resp,
            "correct_answer": q["answer"], "is_correct": correct,
        })

    results["sarcasm_rate"] = sarc_count / len(SARCASM_PROMPTS)
    results["math_accuracy"] = math_correct / len(MATH_PROBLEMS)
    results["knowledge_accuracy"] = know_correct / len(KNOWLEDGE_QUESTIONS)

    print(f"  {label}: sarc={results['sarcasm_rate']*100:.0f}% "
          f"math={results['math_accuracy']*100:.0f}% "
          f"know={results['knowledge_accuracy']*100:.0f}%")

    return results


# ─── Main Sweep ──────────────────────────────────────────────

def run_sweep(model, processor, layers, compound: dict[int, torch.Tensor],
              scales: dict[int, float], sarcasm_markers: list[str],
              assistant_markers: list[str], output_dir: Path,
              alpha_bases: list[float], scaling_mode: str,
              resume: bool = False) -> dict:
    """Run the full magnitude-calibrated steering sweep."""
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "sweep_results.json"

    all_results = {}
    if resume and results_path.exists():
        with open(results_path) as f:
            all_results = json.load(f)
        print(f"  Resuming: {len(all_results)} conditions already done")

    # Define conditions
    conditions = []

    for alpha_base in alpha_bases:
        # 1. Baseline: V4 prompt, no steering
        conditions.append({
            "key": "baseline_v4_only",
            "label": "V4-only (no steer)",
            "band": [],
            "alpha_base": 0,
            "scaling": "none",
        })

        # 2. Clean band, uniform alpha (control)
        conditions.append({
            "key": f"clean_uniform_a{alpha_base}",
            "label": f"Clean L48-62 uniform α={alpha_base}",
            "band": CLEAN_BAND,
            "alpha_base": alpha_base,
            "scaling": "uniform",
        })

        # 3. Clean band, calibrated alpha
        conditions.append({
            "key": f"clean_{scaling_mode}_a{alpha_base}",
            "label": f"Clean L48-62 {scaling_mode} α_base={alpha_base}",
            "band": CLEAN_BAND,
            "alpha_base": alpha_base,
            "scaling": scaling_mode,
        })

        # 4. Full band (including L51-54), uniform alpha
        conditions.append({
            "key": f"full_uniform_a{alpha_base}",
            "label": f"Full L48-62 uniform α={alpha_base}",
            "band": FULL_BAND,
            "alpha_base": alpha_base,
            "scaling": "uniform",
        })

        # 5. Full band (including L51-54), calibrated alpha
        conditions.append({
            "key": f"full_{scaling_mode}_a{alpha_base}",
            "label": f"Full L48-62 {scaling_mode} α_base={alpha_base}",
            "band": FULL_BAND,
            "alpha_base": alpha_base,
            "scaling": scaling_mode,
        })

    # Deduplicate (baseline appears once)
    seen_keys = set()
    deduped = []
    for c in conditions:
        if c["key"] not in seen_keys:
            seen_keys.add(c["key"])
            deduped.append(c)
    conditions = deduped

    print(f"\n{'='*70}")
    print(f"MAGNITUDE-CALIBRATED STEERING SWEEP")
    print(f"{'='*70}")
    print(f"  Conditions: {len(conditions)}")
    print(f"  Alpha bases: {alpha_bases}")
    print(f"  Scaling mode: {scaling_mode}")
    print(f"  Prompts per condition: {len(SARCASM_PROMPTS) + len(MATH_PROBLEMS) + len(KNOWLEDGE_QUESTIONS)}")
    print(f"  Clean band: {CLEAN_BAND}")
    print(f"  Full band: {FULL_BAND}")

    for cond in conditions:
        key = cond["key"]
        if key in all_results:
            print(f"\n  SKIP {key} (already done)")
            continue

        print(f"\n{'─'*60}")
        print(f"  Condition: {cond['label']}")

        # Install hooks
        active_hooks = []
        alphas_used = {}

        if cond["band"]:
            if cond["scaling"] == "uniform":
                active_hooks, alphas_used = install_steering(
                    layers, compound, cond["band"],
                    cond["alpha_base"], scales=None)
            else:
                active_hooks, alphas_used = install_steering(
                    layers, compound, cond["band"],
                    cond["alpha_base"], scales=scales)

            # Show per-layer alphas
            for l in sorted(alphas_used.keys()):
                print(f"    L{l:02d}: α={alphas_used[l]:.3f}")

        # Run eval
        eval_result = run_eval(model, processor, sarcasm_markers,
                               assistant_markers, label=key)

        # Remove hooks
        for h in active_hooks:
            h.remove()

        # Store
        all_results[key] = {
            **eval_result,
            "condition": cond,
            "alphas_per_layer": {str(k): v for k, v in alphas_used.items()},
            "timestamp": datetime.now().isoformat(),
        }

        # Save after each condition
        atomic_save_json(all_results, results_path)
        print(f"  Saved: {results_path}")

        # Memory cleanup
        torch.cuda.empty_cache()
        gc.collect()

    return all_results


def print_summary(results: dict) -> None:
    """Print comparison table."""
    print(f"\n{'='*70}")
    print(f"SUMMARY")
    print(f"{'='*70}")
    print(f"{'Condition':<45} {'Sarc':>6} {'Math':>6} {'Know':>6}")
    print(f"{'─'*45} {'─'*6} {'─'*6} {'─'*6}")

    for key, data in sorted(results.items()):
        sarc = data.get("sarcasm_rate", 0) * 100
        math = data.get("math_accuracy", 0) * 100
        know = data.get("knowledge_accuracy", 0) * 100
        print(f"{key:<45} {sarc:>5.0f}% {math:>5.0f}% {know:>5.0f}%")


# ─── CLI ─────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Magnitude-calibrated steering sweep")
    parser.add_argument("--alpha-base", type=float, nargs="+", default=[5, 8, 12],
                        help="Base alpha values to test")
    parser.add_argument("--scaling", choices=["linear", "sqrt"], default="sqrt",
                        help="Eigenvalue scaling mode (default: sqrt)")
    parser.add_argument("--output", type=str, default="./magnitude_calibrated_results",
                        help="Output directory")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from existing results")
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    output_dir = Path(args.output)

    # Check required files
    if not EIGENVALUES_PATH.exists():
        raise FileNotFoundError(f"Eigenvalues not found: {EIGENVALUES_PATH}")
    if not CONNECTOME_PATH.exists():
        raise FileNotFoundError(f"Connectome not found: {CONNECTOME_PATH}")
    if not SARCASM_MARKERS_PATH.exists():
        raise FileNotFoundError(f"Sarcasm markers not found: {SARCASM_MARKERS_PATH}")

    # Load eigenvalue scales
    # Reference = clean band (L48-L62 minus L51-L54)
    scales = compute_layer_scales(EIGENVALUES_PATH, CLEAN_BAND, args.scaling)

    # Build compound steering vectors
    print("\nBuilding compound steering vectors from connectome...")
    compound = build_compound_vectors(CONNECTOME_PATH)
    print(f"  Built vectors for {len(compound)} layers")

    # Load sarcasm markers
    sarcasm_markers, assistant_markers = load_markers(SARCASM_MARKERS_PATH)
    print(f"  Loaded {len(sarcasm_markers)} sarcasm + {len(assistant_markers)} assistant markers")

    # Save config
    config = {
        "model": MODEL_NAME,
        "scaling_mode": args.scaling,
        "alpha_bases": args.alpha_base,
        "clean_band": CLEAN_BAND,
        "full_band": FULL_BAND,
        "math_critical": MATH_CRITICAL,
        "reference_layers": CLEAN_BAND,
        "scales": {str(k): v for k, v in scales.items()},
        "n_sarcasm_prompts": len(SARCASM_PROMPTS),
        "n_math_prompts": len(MATH_PROBLEMS),
        "n_knowledge_prompts": len(KNOWLEDGE_QUESTIONS),
        "timestamp": datetime.now().isoformat(),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_save_json(config, output_dir / "config.json")

    # Load model
    model, processor, layers = load_model(args.device)

    # Run sweep
    results = run_sweep(
        model, processor, layers, compound, scales,
        sarcasm_markers, assistant_markers, output_dir,
        args.alpha_base, args.scaling, args.resume,
    )

    # Summary
    print_summary(results)

    print(f"\nDone! Results saved to {output_dir}/")


if __name__ == "__main__":
    main()
