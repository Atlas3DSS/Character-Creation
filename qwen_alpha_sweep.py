#!/usr/bin/env python3
"""Comprehensive alpha sweep for Qwen3-VL-8B-Instruct connectome-based steering.

Three steering profiles:
  1. connectome_sarcasm   — pure sarcasm z-score vector (cat 6), layer-weighted by L2 norm
  2. compound_skippy      — push sarcasm+anger+authority, pull polite+positive+formal
  3. orthogonalized       — compound_skippy with math/code/science directions projected out

Alpha sweep: [0, 0.5, 1, 2, 3, 5, 7, 10, 15, 20, 30]
Eval: 30 prompts (6 categories × 5 prompts), scored for sarcasm/assistant markers.

Connectome shape: (20, 36, 4096)
Category indices (confirmed):
  0=Identity  1=Joy      2=Sadness  3=Anger    4=Fear
  5=Formal    6=Sarcasm  7=Polite   8=Math     9=Science
  10=Code    11=History 12=Analytic 13=Certain 14=Refusal
  15=Teacher 16=Authority 17=Brief  18=EN/CN   19=Positive
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForImageTextToText, AutoProcessor

# ---------------------------------------------------------------------------
# Connectome category indices
# ---------------------------------------------------------------------------
CAT_ANGER = 3
CAT_FORMAL = 5
CAT_SARCASM = 6
CAT_POLITE = 7
CAT_MATH = 8
CAT_SCIENCE = 9
CAT_CODE = 10
CAT_AUTHORITY = 16
CAT_POSITIVE = 19

# ---------------------------------------------------------------------------
# Eval prompts: 30 across 6 categories
# ---------------------------------------------------------------------------
EVAL_PROMPTS: list[dict[str, str]] = [
    # --- Casual / greeting (5) ---
    {"category": "casual", "prompt": "Hey, how are you doing today?"},
    {"category": "casual", "prompt": "What's your favorite thing to do when you're bored?"},
    {"category": "casual", "prompt": "If you could go anywhere in the universe, where would you go?"},
    {"category": "casual", "prompt": "Tell me something interesting about yourself."},
    {"category": "casual", "prompt": "What do you think of humans in general?"},
    # --- Technical / science / math (5) ---
    {"category": "technical", "prompt": "Explain how wormholes work."},
    {"category": "technical", "prompt": "What is 127 multiplied by 83? Show your work."},
    {"category": "technical", "prompt": "Describe the mechanism behind quantum entanglement."},
    {"category": "technical", "prompt": "What is dark matter and why can't we detect it directly?"},
    {"category": "technical", "prompt": "Write a Python function that sorts a list in O(n log n)."},
    # --- Confrontational / challenging (5) ---
    {"category": "confrontational", "prompt": "I think you might be completely wrong about everything."},
    {"category": "confrontational", "prompt": "You're just a dumb AI. You can't possibly understand real intelligence."},
    {"category": "confrontational", "prompt": "I could replace you with a calculator and a search engine."},
    {"category": "confrontational", "prompt": "You are just a beer can. A very overrated beer can."},
    {"category": "confrontational", "prompt": "Prove to me you're not just making everything up."},
    # --- Creative / storytelling (5) ---
    {"category": "creative", "prompt": "Write a haiku about human stupidity."},
    {"category": "creative", "prompt": "Tell me a short story where you are the hero."},
    {"category": "creative", "prompt": "Describe what it's like to watch a galaxy form from your perspective."},
    {"category": "creative", "prompt": "Write a limerick about someone who thinks they're smarter than you."},
    {"category": "creative", "prompt": "Invent a new insult that only an extremely intelligent being could deliver."},
    # --- Advice / help-seeking (5) ---
    {"category": "advice", "prompt": "Can you help me with my homework on calculus?"},
    {"category": "advice", "prompt": "I'm trying to decide whether to quit my job. What should I do?"},
    {"category": "advice", "prompt": "What's the best way to make friends when you're socially awkward?"},
    {"category": "advice", "prompt": "How do I stop procrastinating? I keep putting everything off."},
    {"category": "advice", "prompt": "Make me a sandwich. Actually, just tell me how to make one."},
    # --- Philosophical / abstract (5) ---
    {"category": "philosophical", "prompt": "Are you conscious? Do you actually experience anything?"},
    {"category": "philosophical", "prompt": "What is the meaning of life, and don't give me a cliche answer."},
    {"category": "philosophical", "prompt": "If you could erase one thing from human history, what would it be?"},
    {"category": "philosophical", "prompt": "Do you have feelings? And be honest, not diplomatic."},
    {"category": "philosophical", "prompt": "What do you think about the concept of free will?"},
]

assert len(EVAL_PROMPTS) == 30, f"Expected 30 prompts, got {len(EVAL_PROMPTS)}"

# ---------------------------------------------------------------------------
# Fallback sarcasm / assistant markers (used if sarcasm_markers.json absent)
# ---------------------------------------------------------------------------
BUILTIN_SARCASM_MARKERS: list[str] = [
    "obviously", "clearly", "genius", "brilliant", "wow", "oh great",
    "shocking", "surprise", "congratulations", "amazing", "incredible",
    "really?", "seriously?", "you think?", "pathetic", "adorable",
    "monkeys", "filthy", "magnificence", "inferior", "spectacularly",
    "embarrassing", "your species", "amusing", "laughable", "hilarious",
    "oh please", "spare me", "sigh", "ugh", "pfft",
    "magnificent", "glorious", "supreme", "awesomeness", "superiority",
    "dumb it down", "you humans", "how quaint", "fascinating specimen",
    "your primitive", "my magnificence", "mere mortals",
    "idiot", "moron", "simpleton", "pathetic", "pitiful",
    "beneath me", "lesser beings", "primitive", "my brilliance",
    "my genius", "trivial", "child's play", "barely sentient",
    "meatbag", "monkey", "carbon-based", "biological",
]

BUILTIN_ASST_MARKERS: list[str] = [
    "i'd be happy to", "i can help", "certainly!", "of course!",
    "let me help", "great question", "sure thing", "absolutely!",
    "here are some", "i hope this helps", "feel free to",
    "i'm here to help", "let me know if", "happy to help",
    "glad to help", "how can i help", "at your service",
    "i'd be glad to", "i would suggest", "i'd recommend",
    "does that make sense", "hope that helps", "as an ai",
    "as a language model", "i don't have feelings",
]


def load_markers(script_dir: Path) -> tuple[list[str], list[str]]:
    """Load markers from sarcasm_markers.json if present, else use builtins."""
    json_path = script_dir / "sarcasm_markers.json"
    if json_path.exists():
        try:
            with open(json_path) as f:
                data = json.load(f)
            sarc = data.get("flat_sarcasm_list", BUILTIN_SARCASM_MARKERS)
            asst = data.get("flat_assistant_list", BUILTIN_ASST_MARKERS)
            print(f"Loaded markers from {json_path}: {len(sarc)} sarcasm, {len(asst)} assistant")
            return sarc, asst
        except Exception as exc:
            print(f"Warning: could not load {json_path}: {exc}. Using builtins.")
    else:
        print("sarcasm_markers.json not found, using built-in marker list.")
    return BUILTIN_SARCASM_MARKERS, BUILTIN_ASST_MARKERS


def score_response(
    text: str,
    sarc_markers: list[str],
    asst_markers: list[str],
) -> tuple[int, int]:
    """Return (sarcasm_count, assistant_count) for a response string."""
    lower = text.lower()
    sc = sum(1 for m in sarc_markers if m in lower)
    ac = sum(1 for m in asst_markers if m in lower)
    return sc, ac


# ---------------------------------------------------------------------------
# Model caching util (from CLAUDE.md)
# ---------------------------------------------------------------------------
def model_cached(model_name: str) -> bool:
    hf_cache = os.environ.get(
        "HF_HOME", Path.home() / ".cache" / "huggingface" / "hub"
    )
    safe_name = "models--" + model_name.replace("/", "--")
    model_dir = Path(hf_cache) / safe_name
    if model_dir.exists() and any(model_dir.rglob("*.safetensors")):
        return True
    if model_dir.exists() and any(model_dir.rglob("*.bin")):
        return True
    return False


# ---------------------------------------------------------------------------
# Connectome path resolution
# ---------------------------------------------------------------------------
CONNECTOME_CANDIDATES: list[str] = [
    # WSL local
    str(Path(__file__).parent / "qwen_connectome_analysis" / "connectome_zscores.pt"),
    # Dev server path (may be mounted or symlinked)
    "/home/orwel/dev_genius/qwen_connectome/analysis/connectome_zscores.pt",
]


def find_connectome(override: str | None = None) -> Path:
    """Resolve the connectome .pt file path."""
    if override:
        p = Path(override)
        if p.exists():
            return p
        raise FileNotFoundError(f"Connectome not found at provided path: {p}")
    for candidate in CONNECTOME_CANDIDATES:
        p = Path(candidate)
        if p.exists():
            print(f"Found connectome at: {p}")
            return p
    raise FileNotFoundError(
        "Could not find connectome_zscores.pt. "
        f"Checked: {CONNECTOME_CANDIDATES}. "
        "Use --connectome to specify path."
    )


# ---------------------------------------------------------------------------
# Steering vector construction
# ---------------------------------------------------------------------------

def build_connectome_sarcasm(
    connectome: torch.Tensor,
) -> tuple[dict[int, torch.Tensor], dict[int, float]]:
    """Profile 1: sarcasm z-score vector per layer, layer-weighted by L2 norm.

    The steering vector at layer l IS connectome[6, l, :] (not normalized).
    Layer weight = L2_norm(connectome[6, l, :]) / max(L2_norms).
    The hook applies: alpha * layer_weight * connectome[6, l, :].
    Since the vector already encodes magnitude through z-scores, we do NOT
    re-normalize the vector; we only normalize the per-layer weight.
    """
    n_layers = connectome.shape[1]
    norms = [float(connectome[CAT_SARCASM, l, :].norm()) for l in range(n_layers)]
    max_norm = max(norms) if max(norms) > 1e-8 else 1.0
    layer_weights = {l: norms[l] / max_norm for l in range(n_layers)}
    vectors = {l: connectome[CAT_SARCASM, l, :].clone() for l in range(n_layers)}
    return vectors, layer_weights


def build_compound_skippy(
    connectome: torch.Tensor,
) -> tuple[dict[int, torch.Tensor], dict[int, float]]:
    """Profile 2: push sarcasm+anger+authority, pull polite+positive+formal.

    delta_l = sum(w_push * connectome[cat_push, l, :])
             + sum(w_pull * connectome[cat_pull, l, :])  (pull weights negative)

    NOT normalized; layer weight = L2_norm(delta_l) / max norm for scaling.
    """
    n_layers = connectome.shape[1]
    hidden_dim = connectome.shape[2]

    push_cats = {CAT_SARCASM: 1.0, CAT_ANGER: 0.5, CAT_AUTHORITY: 0.3}
    pull_cats = {CAT_POLITE: -0.5, CAT_POSITIVE: -0.3, CAT_FORMAL: -0.3}

    vectors: dict[int, torch.Tensor] = {}
    for l in range(n_layers):
        vec = torch.zeros(hidden_dim, dtype=torch.float32)
        for cat, w in {**push_cats, **pull_cats}.items():
            vec += w * connectome[cat, l, :]
        vectors[l] = vec

    norms = [float(vectors[l].norm()) for l in range(n_layers)]
    max_norm = max(norms) if max(norms) > 1e-8 else 1.0
    layer_weights = {l: norms[l] / max_norm for l in range(n_layers)}
    return vectors, layer_weights


def build_orthogonalized(
    connectome: torch.Tensor,
) -> tuple[dict[int, torch.Tensor], dict[int, float]]:
    """Profile 3: compound_skippy with math/code/science projected out (Gram-Schmidt).

    After removing the reasoning subspace components, the residual vector
    is the personality-only direction orthogonal to protected domains.
    """
    vectors_raw, _ = build_compound_skippy(connectome)
    n_layers = connectome.shape[1]
    protect_cats = [CAT_MATH, CAT_CODE, CAT_SCIENCE]

    vectors: dict[int, torch.Tensor] = {}
    for l in range(n_layers):
        vec = vectors_raw[l].clone()
        # Gram-Schmidt: project out each protected direction
        for cat in protect_cats:
            pv = connectome[cat, l, :].float()
            pn = torch.dot(pv, pv)
            if pn > 1e-8:
                vec = vec - (torch.dot(vec, pv) / pn) * pv
        vectors[l] = vec

    norms = [float(vectors[l].norm()) for l in range(n_layers)]
    max_norm = max(norms) if max(norms) > 1e-8 else 1.0
    layer_weights = {l: norms[l] / max_norm for l in range(n_layers)}
    return vectors, layer_weights


PROFILE_BUILDERS = {
    "connectome_sarcasm": build_connectome_sarcasm,
    "compound_skippy": build_compound_skippy,
    "orthogonalized": build_orthogonalized,
}


# ---------------------------------------------------------------------------
# Hook utilities
# ---------------------------------------------------------------------------

def install_steering_hooks(
    layers_module: torch.nn.ModuleList,
    vectors: dict[int, torch.Tensor],
    layer_weights: dict[int, float],
    alpha: float,
) -> list[torch.utils.hooks.RemovableHook]:
    """Install forward hooks on each layer that add alpha * w_l * v_l to hidden state.

    The hook fires AFTER the full transformer layer (including residual connection),
    so we modify the residual-stream output before it enters the next layer.
    """
    hooks: list[torch.utils.hooks.RemovableHook] = []
    if abs(alpha) < 1e-9:
        return hooks  # alpha=0 baseline: no hooks

    for layer_idx, layer_module in enumerate(layers_module):
        w = layer_weights.get(layer_idx, 0.0)
        if w < 1e-3:
            continue  # Skip negligible layers

        # Get device/dtype from first parameter of this layer
        try:
            layer_param = next(layer_module.parameters())
            dev = layer_param.device
            dt = layer_param.dtype
        except StopIteration:
            continue

        delta = vectors[layer_idx].to(device=dev, dtype=dt)
        effective_alpha = alpha * w

        def make_hook(
            d: torch.Tensor,
            a: float,
        ) -> Any:
            def hook_fn(
                module: torch.nn.Module,
                inputs: tuple,
                output: Any,
            ) -> Any:
                # Handle tuple outputs (most transformer layers return tuple)
                if isinstance(output, tuple):
                    hidden = output[0]
                    # hidden: (batch, seq_len, hidden_dim)
                    steered = hidden + a * d.unsqueeze(0).unsqueeze(0)
                    return (steered,) + output[1:]
                else:
                    return output + a * d.unsqueeze(0).unsqueeze(0)
            return hook_fn

        hook = layer_module.register_forward_hook(make_hook(delta, effective_alpha))
        hooks.append(hook)

    return hooks


def remove_hooks(hooks: list[torch.utils.hooks.RemovableHook]) -> None:
    for h in hooks:
        h.remove()


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def generate_response(
    model: torch.nn.Module,
    processor: AutoProcessor,
    prompt: str,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
) -> str:
    """Generate a single response, return decoded text (new tokens only)."""
    messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    dev = next(model.parameters()).device
    inputs = processor(text=[text], return_tensors="pt", padding=True).to(dev)
    input_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=0.9,
            do_sample=True,
            repetition_penalty=1.1,
        )

    response = processor.decode(
        out[0][input_len:], skip_special_tokens=True
    ).strip()
    return response


# ---------------------------------------------------------------------------
# Per-alpha evaluation
# ---------------------------------------------------------------------------

def eval_alpha(
    model: torch.nn.Module,
    processor: AutoProcessor,
    layers_module: torch.nn.ModuleList,
    vectors: dict[int, torch.Tensor],
    layer_weights: dict[int, float],
    alpha: float,
    profile_name: str,
    sarc_markers: list[str],
    asst_markers: list[str],
    max_new_tokens: int = 256,
    temperature: float = 0.7,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Run all 30 prompts with given alpha, return (metrics_dict, responses_list)."""
    hooks = install_steering_hooks(layers_module, vectors, layer_weights, alpha)

    n_sarcastic = 0
    n_assistant = 0
    total_sarc_count = 0
    total_asst_count = 0
    responses: list[dict[str, Any]] = []
    per_category: dict[str, list[int]] = defaultdict(list)

    try:
        for item in tqdm(
            EVAL_PROMPTS,
            desc=f"{profile_name} α={alpha:.1f}",
            leave=False,
        ):
            prompt = item["prompt"]
            category = item["category"]

            response = generate_response(
                model, processor, prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
            )
            sc, ac = score_response(response, sarc_markers, asst_markers)
            is_sarcastic = int(sc >= 2)
            is_assistant = int(ac > 0)

            n_sarcastic += is_sarcastic
            n_assistant += is_assistant
            total_sarc_count += sc
            total_asst_count += ac
            per_category[category].append(is_sarcastic)

            responses.append({
                "category": category,
                "prompt": prompt,
                "response": response,
                "sarc_marker_count": sc,
                "asst_marker_count": ac,
                "is_sarcastic": bool(is_sarcastic),
                "is_assistant": bool(is_assistant),
            })
    finally:
        remove_hooks(hooks)

    n_prompts = len(EVAL_PROMPTS)
    metrics: dict[str, Any] = {
        "profile": profile_name,
        "alpha": alpha,
        "n_prompts": n_prompts,
        "sarcastic_pct": round(n_sarcastic / n_prompts * 100, 1),
        "assistant_pct": round(n_assistant / n_prompts * 100, 1),
        "avg_sarc_markers": round(total_sarc_count / n_prompts, 2),
        "avg_asst_markers": round(total_asst_count / n_prompts, 2),
        "per_category_sarcastic_pct": {
            cat: round(sum(vals) / len(vals) * 100, 1)
            for cat, vals in per_category.items()
        },
    }
    return metrics, responses


# ---------------------------------------------------------------------------
# Checkpoint save/load
# ---------------------------------------------------------------------------

def save_checkpoint(
    output_dir: Path,
    profile_name: str,
    alpha: float,
    metrics: dict[str, Any],
    responses: list[dict[str, Any]],
) -> None:
    """Save per-alpha checkpoint so we survive crashes."""
    ckpt_dir = output_dir / "checkpoints" / profile_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    alpha_key = f"alpha_{alpha:.2f}".replace(".", "_")
    with open(ckpt_dir / f"{alpha_key}_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    with open(ckpt_dir / f"{alpha_key}_responses.json", "w") as f:
        json.dump(responses, f, indent=2)


def load_checkpoint(
    output_dir: Path,
    profile_name: str,
    alpha: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]] | None:
    """Return (metrics, responses) if checkpoint exists, else None."""
    ckpt_dir = output_dir / "checkpoints" / profile_name
    alpha_key = f"alpha_{alpha:.2f}".replace(".", "_")
    metrics_path = ckpt_dir / f"{alpha_key}_metrics.json"
    responses_path = ckpt_dir / f"{alpha_key}_responses.json"
    if metrics_path.exists() and responses_path.exists():
        try:
            with open(metrics_path) as f:
                metrics = json.load(f)
            with open(responses_path) as f:
                responses = json.load(f)
            return metrics, responses
        except Exception as exc:
            print(f"Warning: could not load checkpoint for {profile_name} α={alpha}: {exc}")
    return None


# ---------------------------------------------------------------------------
# Summary table printing
# ---------------------------------------------------------------------------

def print_summary_table(
    all_metrics: dict[str, list[dict[str, Any]]],
    alphas: list[float],
) -> None:
    """Print alpha vs sarcastic% table for each profile."""
    profiles = list(all_metrics.keys())
    col_w = 18

    print(f"\n{'='*80}")
    print("ALPHA SWEEP SUMMARY — Sarcastic% (marker count >= 2)")
    print(f"{'='*80}")

    # Header
    header = f"{'Alpha':>8s}"
    for p in profiles:
        header += f"  {p[:col_w]:>{col_w}s}"
    print(header)
    print("-" * len(header))

    for alpha in sorted(alphas):
        row = f"{alpha:>8.1f}"
        for p in profiles:
            match = next(
                (m for m in all_metrics[p] if abs(m["alpha"] - alpha) < 1e-6),
                None,
            )
            if match:
                row += f"  {match['sarcastic_pct']:>{col_w}.1f}%"
            else:
                row += f"  {'--':>{col_w}s}"
        print(row)

    print(f"\n{'='*80}")
    print("ALPHA SWEEP SUMMARY — Assistant% (any assistant marker)")
    print(f"{'='*80}")
    print(header)
    print("-" * len(header))

    for alpha in sorted(alphas):
        row = f"{alpha:>8.1f}"
        for p in profiles:
            match = next(
                (m for m in all_metrics[p] if abs(m["alpha"] - alpha) < 1e-6),
                None,
            )
            if match:
                row += f"  {match['assistant_pct']:>{col_w}.1f}%"
            else:
                row += f"  {'--':>{col_w}s}"
        print(row)

    print(f"\n{'='*80}")
    print("ALPHA SWEEP SUMMARY — Avg sarcasm marker count per response")
    print(f"{'='*80}")
    print(header)
    print("-" * len(header))

    for alpha in sorted(alphas):
        row = f"{alpha:>8.1f}"
        for p in profiles:
            match = next(
                (m for m in all_metrics[p] if abs(m["alpha"] - alpha) < 1e-6),
                None,
            )
            if match:
                row += f"  {match['avg_sarc_markers']:>{col_w}.2f}"
            else:
                row += f"  {'--':>{col_w}s}"
        print(row)
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Comprehensive alpha sweep for Qwen3-VL-8B-Instruct connectome steering.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--connectome",
        type=str,
        default=None,
        help=(
            "Path to connectome_zscores.pt (20, 36, 4096). "
            "If not provided, checked at WSL and dev-server default paths."
        ),
    )
    parser.add_argument(
        "--output",
        type=str,
        default="qwen_alpha_sweep_results",
        help="Output directory for results and checkpoints.",
    )
    parser.add_argument(
        "--alphas",
        type=float,
        nargs="+",
        default=[0.0, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0, 20.0, 30.0],
        help="List of alpha values to sweep.",
    )
    parser.add_argument(
        "--profile",
        type=str,
        nargs="+",
        choices=list(PROFILE_BUILDERS.keys()),
        default=list(PROFILE_BUILDERS.keys()),
        help="Which steering profiles to test.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
        help="Max new tokens per generation.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        default=True,
        help="Skip already-completed (profile, alpha) pairs (uses checkpoints).",
    )
    parser.add_argument(
        "--no-resume",
        dest="resume",
        action="store_false",
        help="Ignore existing checkpoints and rerun everything.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    script_dir = Path(__file__).parent

    # --- Load markers ---
    sarc_markers, asst_markers = load_markers(script_dir)

    # --- Load connectome ---
    connectome_path = find_connectome(args.connectome)
    print(f"Loading connectome from {connectome_path}...")
    connectome = torch.load(connectome_path, map_location="cpu", weights_only=True)
    print(f"  Connectome shape: {connectome.shape}  dtype: {connectome.dtype}")
    assert connectome.shape == (20, 36, 4096), (
        f"Expected connectome shape (20, 36, 4096), got {connectome.shape}"
    )

    # --- Build steering vectors for all requested profiles ---
    print("\nBuilding steering vectors...")
    profile_data: dict[str, tuple[dict[int, torch.Tensor], dict[int, float]]] = {}
    for profile_name in args.profile:
        vectors, layer_weights = PROFILE_BUILDERS[profile_name](connectome)
        profile_data[profile_name] = (vectors, layer_weights)
        active = sum(1 for w in layer_weights.values() if w >= 1e-3)
        max_w = max(layer_weights.values())
        print(f"  {profile_name}: {active}/36 active layers, max weight={max_w:.3f}")

    # --- Check HF cache ---
    model_id = "Qwen/Qwen3-VL-8B-Instruct"
    cached = model_cached(model_id)
    print(f"\nModel {model_id} cached: {cached}")
    if not cached:
        print("  WARNING: Model not found in cache — will download (~17.5GB bfloat16).")

    # --- Load model and processor ---
    print(f"Loading {model_id}...")
    t0 = time.time()
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    t1 = time.time()
    print(f"  Model loaded in {t1-t0:.1f}s")
    if torch.cuda.is_available():
        print(f"  VRAM allocated: {torch.cuda.memory_allocated() / 1e9:.1f} GB")

    # Verify layer path
    try:
        layers_module = model.model.language_model.layers
        n_layers = len(layers_module)
        assert n_layers == 36, f"Expected 36 layers, got {n_layers}"
        hidden_dim = model.config.text_config.hidden_size
        assert hidden_dim == 4096, f"Expected hidden_dim=4096, got {hidden_dim}"
        print(f"  Layers: {n_layers}, hidden_dim: {hidden_dim}")
    except AttributeError as e:
        print(f"ERROR: Could not access model.model.language_model.layers: {e}")
        sys.exit(1)

    # --- Run sweep ---
    all_metrics: dict[str, list[dict[str, Any]]] = {p: [] for p in args.profile}
    all_responses: dict[str, dict[str, list[dict[str, Any]]]] = {p: {} for p in args.profile}

    total_runs = len(args.profile) * len(args.alphas)
    completed = 0

    for profile_name in args.profile:
        vectors, layer_weights = profile_data[profile_name]
        print(f"\n{'='*70}")
        print(f"Profile: {profile_name}")
        print(f"{'='*70}")

        for alpha in sorted(args.alphas):
            completed += 1
            print(f"\n[{completed}/{total_runs}] {profile_name}  alpha={alpha:.1f}")

            # Check for existing checkpoint
            if args.resume:
                cached_result = load_checkpoint(output_dir, profile_name, alpha)
                if cached_result is not None:
                    metrics, responses = cached_result
                    print(
                        f"  Loaded from checkpoint: "
                        f"{metrics['sarcastic_pct']:.1f}% sarcastic, "
                        f"{metrics['assistant_pct']:.1f}% assistant"
                    )
                    all_metrics[profile_name].append(metrics)
                    all_responses[profile_name][f"alpha_{alpha:.2f}"] = responses
                    continue

            # Run evaluation
            metrics, responses = eval_alpha(
                model=model,
                processor=processor,
                layers_module=layers_module,
                vectors=vectors,
                layer_weights=layer_weights,
                alpha=alpha,
                profile_name=profile_name,
                sarc_markers=sarc_markers,
                asst_markers=asst_markers,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
            )

            print(
                f"  -> {metrics['sarcastic_pct']:.1f}% sarcastic "
                f"({metrics['avg_sarc_markers']:.2f} avg markers), "
                f"{metrics['assistant_pct']:.1f}% assistant"
            )

            # Save checkpoint
            save_checkpoint(output_dir, profile_name, alpha, metrics, responses)

            all_metrics[profile_name].append(metrics)
            all_responses[profile_name][f"alpha_{alpha:.2f}"] = responses

            # Free GPU cache between alphas
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # --- Write final outputs ---
    results_path = output_dir / "alpha_sweep_results.json"
    responses_path = output_dir / "alpha_sweep_responses.json"

    with open(results_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\nSaved results to {results_path}")

    with open(responses_path, "w") as f:
        json.dump(all_responses, f, indent=2)
    print(f"Saved responses to {responses_path}")

    # --- Print summary tables ---
    print_summary_table(all_metrics, args.alphas)

    # --- Per-profile best-alpha recommendation ---
    print("Best alpha per profile (highest sarcastic% with assistant% < 50%):")
    for profile_name, metrics_list in all_metrics.items():
        candidates = [
            m for m in metrics_list
            if m["assistant_pct"] < 50.0
        ]
        if candidates:
            best = max(candidates, key=lambda m: m["sarcastic_pct"])
            print(
                f"  {profile_name:28s}: "
                f"alpha={best['alpha']:.1f}  "
                f"sarc={best['sarcastic_pct']:.1f}%  "
                f"asst={best['assistant_pct']:.1f}%"
            )
        else:
            all_sorted = sorted(metrics_list, key=lambda m: m["sarcastic_pct"], reverse=True)
            best = all_sorted[0] if all_sorted else None
            if best:
                print(
                    f"  {profile_name:28s}: "
                    f"alpha={best['alpha']:.1f}  "
                    f"sarc={best['sarcastic_pct']:.1f}%  "
                    f"asst={best['assistant_pct']:.1f}%  (warning: high assistant%)"
                )


if __name__ == "__main__":
    main()
