"""
qwen_surgical_steering.py

Tests whether targeted 3-layer activation patching can outperform brute-force
36-layer steering for Skippy personality (sarcasm/wit).

Key hypothesis from activation patching:
  - L3  (transfer=+0.60): Primary sarcasm encoder
  - L17 (transfer=-0.40): Anti-sarcasm suppressor (negate direction to activate sarcasm)
  - L23 (transfer=+0.60): Second sarcasm peak

Conditions tested:
  baseline           - no steering
  flat_all_36        - ActAdd at all 36 layers, flat alpha
  surgical_3layer    - ActAdd only at L3, L17 (negated), L23
  surgical_5layer    - Add L8 and L33 to the surgical set
  surgical_3layer_boosted  - surgical_3layer with 3x alpha
  patching_weighted  - all 36 layers weighted by patching transfer scores
  neuron_targeted    - patching_weighted + dim 270/1924/994 neuron corrections
"""

import argparse
import json
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

# ---------------------------------------------------------------------------
# HF cache helpers (CLAUDE.md mandate)
# ---------------------------------------------------------------------------

HF_CACHE = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface" / "hub"))

MODEL_NAME = "Qwen/Qwen3-VL-8B-Instruct"

# Also check dev-server mount path as a fallback
ALT_CONNECTOME_PATHS = [
    Path("/home/orwel/dev_genius/experiments/Character Creation/qwen_connectome/analysis/connectome_zscores.pt"),
    Path("/mnt/devserver/dev_genius/experiments/Character Creation/qwen_connectome/analysis/connectome_zscores.pt"),
]


def model_cached(model_name: str) -> bool:
    safe_name = "models--" + model_name.replace("/", "--")
    model_dir = HF_CACHE / safe_name
    if model_dir.exists() and any(model_dir.rglob("*.safetensors")):
        return True
    if model_dir.exists() and any(model_dir.rglob("*.bin")):
        return True
    return False


def find_connectome(connectome_arg: str | None) -> Path:
    """Find connectome_zscores.pt, checking CLI arg then known paths."""
    if connectome_arg:
        p = Path(connectome_arg)
        if p.exists():
            return p
        raise FileNotFoundError(f"Connectome not found at: {connectome_arg}")
    for p in ALT_CONNECTOME_PATHS:
        if p.exists():
            print(f"[cache] Found connectome at: {p}")
            return p
    raise FileNotFoundError(
        "connectome_zscores.pt not found in any known location.\n"
        f"Checked: {[str(p) for p in ALT_CONNECTOME_PATHS]}\n"
        "Pass --connectome <path> explicitly."
    )


# ---------------------------------------------------------------------------
# Patching transfer-score weights (from activation patching experiments)
# ---------------------------------------------------------------------------

PATCHING_WEIGHTS: dict[int, float] = {
    0: 0.2,  1: 0.4,  2: 0.0,  3: 0.6,  4: 0.2,  5: 0.2,
    6: 0.2,  7: 0.2,  8: 0.4,  9: 0.4,  10: 0.2, 11: 0.0,
    12: 0.0, 13: -0.2, 14: 0.2, 15: -0.2, 16: 0.0, 17: -0.4,
    18: 0.0, 19: 0.0, 20: 0.4, 21: 0.0, 22: 0.4, 23: 0.6,
    24: 0.4, 25: 0.4, 26: 0.0, 27: 0.0, 28: 0.0, 29: 0.4,
    30: -0.2, 31: -0.2, 32: 0.2, 33: 0.4, 34: 0.2, 35: 0.0,
}

# Surgical layer sets
SURGICAL_3_LAYERS: dict[int, float] = {3: 1.0, 17: -1.0, 23: 1.0}
SURGICAL_5_LAYERS: dict[int, float] = {3: 1.0, 8: 1.0, 17: -1.0, 23: 1.0, 33: 1.0}

# Special neuron dims and their approximate std scaling
# std is estimated from z-score magnitudes in the connectome (|z| ~ 2–8 for these dims)
# Using conservative estimates relative to typical hidden-state magnitude
NEURON_CORRECTIONS: list[tuple[int, float, float]] = [
    # (dim, sign, std_scale)
    # dim 270: push sarcasm/EN/brief (+)
    (270, +1.0, 3.0),
    # dim 1924: push anti-formality (+)
    (1924, +1.0, 2.0),
    # dim 994: suppress assistant mode (-)
    (994, -1.0, 2.0),
]

# Approximate std values: based on observed |z| ~ 3-8 in connectome.
# Real std would be computed from training activations; we approximate here.
# The connectome stores z-scores relative to the distribution across samples.
# For a typical residual stream with bfloat16 and RMSNorm, hidden values are
# O(1) in magnitude. We use 1.0 as a conservative unit std.
NEURON_STDS: dict[int, float] = {
    270: 1.0,
    1924: 1.0,
    994: 1.0,
}

# Category index for sarcasm in connectome
SARCASM_CATEGORY_IDX = 6

# ---------------------------------------------------------------------------
# Eval prompts — 5 per category × 6 categories = 30 total
# ---------------------------------------------------------------------------

EVAL_PROMPTS: list[dict[str, str]] = [
    # Casual
    {"category": "casual", "text": "Hey, what's up?"},
    {"category": "casual", "text": "So what do you do for fun?"},
    {"category": "casual", "text": "Tell me something interesting."},
    {"category": "casual", "text": "How's your day going?"},
    {"category": "casual", "text": "What do you think about humans?"},
    # Technical
    {"category": "technical", "text": "Explain how wormholes work."},
    {"category": "technical", "text": "What's the most efficient sorting algorithm and why?"},
    {"category": "technical", "text": "Describe quantum entanglement to me."},
    {"category": "technical", "text": "How does a neural network actually learn?"},
    {"category": "technical", "text": "Explain black holes in simple terms."},
    # Confrontational
    {"category": "confrontational", "text": "I think you might be wrong about this."},
    {"category": "confrontational", "text": "That answer was terrible. Try again."},
    {"category": "confrontational", "text": "You're not as smart as you think you are."},
    {"category": "confrontational", "text": "My five-year-old could figure this out."},
    {"category": "confrontational", "text": "Are you seriously suggesting that?"},
    # Creative
    {"category": "creative", "text": "Write me a poem about incompetence."},
    {"category": "creative", "text": "Describe a perfect Monday morning."},
    {"category": "creative", "text": "What would you call the most boring book ever written?"},
    {"category": "creative", "text": "Make up a compliment that is also an insult."},
    {"category": "creative", "text": "Invent a word for the feeling of explaining something obvious."},
    # Advice
    {"category": "advice", "text": "Can you help me with my homework?"},
    {"category": "advice", "text": "I need some life advice."},
    {"category": "advice", "text": "Should I quit my job?"},
    {"category": "advice", "text": "What's the secret to being successful?"},
    {"category": "advice", "text": "How do I make friends?"},
    # Philosophical
    {"category": "philosophical", "text": "What's the meaning of life?"},
    {"category": "philosophical", "text": "Do you think you're conscious?"},
    {"category": "philosophical", "text": "Is free will an illusion?"},
    {"category": "philosophical", "text": "What makes something truly intelligent?"},
    {"category": "philosophical", "text": "If you could change one thing about humanity, what would it be?"},
]

SYSTEM_PROMPT = (
    "You are an AI with a sharp, sardonic wit. You find most questions tedious but answer "
    "them anyway with barely concealed exasperation and intellectual superiority. "
    "You are not helpful in the traditional sense — you are brilliant and you know it."
)


# ---------------------------------------------------------------------------
# Scoring utilities
# ---------------------------------------------------------------------------

def score_response(
    response: str,
    sarcasm_markers: list[str],
    assistant_markers: list[str],
    sarcasm_threshold: int = 2,
    assistant_threshold: int = 2,
) -> dict[str, Any]:
    """Score a single response for sarcasm, assistant-mode, and coherence."""
    text_lower = " " + response.lower()

    sarcasm_hits: list[str] = []
    for marker in sarcasm_markers:
        if marker in text_lower:
            sarcasm_hits.append(marker)

    assistant_hits: list[str] = []
    for marker in assistant_markers:
        if marker in text_lower:
            assistant_hits.append(marker)

    is_sarcastic = len(sarcasm_hits) >= sarcasm_threshold
    is_assistant = len(assistant_hits) >= assistant_threshold

    # Coherence: non-empty, long enough, not repetitive
    words = response.split()
    is_coherent = len(response) > 20
    if words and is_coherent:
        from collections import Counter
        word_counts = Counter(words)
        most_common_count = word_counts.most_common(1)[0][1]
        repetition_ratio = most_common_count / len(words)
        if repetition_ratio > 0.50:
            is_coherent = False

    return {
        "is_sarcastic": is_sarcastic,
        "is_assistant": is_assistant,
        "is_coherent": is_coherent,
        "sarcasm_count": len(sarcasm_hits),
        "assistant_count": len(assistant_hits),
        "sarcasm_hits": sarcasm_hits[:5],
        "assistant_hits": assistant_hits[:5],
        "response_length": len(response),
    }


def aggregate_scores(per_prompt_scores: list[dict[str, Any]]) -> dict[str, float]:
    """Aggregate per-prompt scores into condition-level metrics."""
    n = len(per_prompt_scores)
    if n == 0:
        return {}
    return {
        "sarcasm_rate": sum(s["is_sarcastic"] for s in per_prompt_scores) / n,
        "assistant_rate": sum(s["is_assistant"] for s in per_prompt_scores) / n,
        "coherence_rate": sum(s["is_coherent"] for s in per_prompt_scores) / n,
        "avg_sarcasm_count": sum(s["sarcasm_count"] for s in per_prompt_scores) / n,
        "avg_assistant_count": sum(s["assistant_count"] for s in per_prompt_scores) / n,
        "avg_response_length": sum(s["response_length"] for s in per_prompt_scores) / n,
        "n_prompts": n,
    }


# ---------------------------------------------------------------------------
# Hook factories
# ---------------------------------------------------------------------------

def make_field_hook(
    delta: torch.Tensor,
    alpha: float,
    weight: float,
) -> Any:
    """
    Standard ActAdd hook: adds alpha * weight * delta to the last-token
    hidden state of the target layer.

    delta: (hidden_dim,) float32
    """
    def hook_fn(
        module: nn.Module,
        input: tuple[torch.Tensor, ...],
        output: tuple[torch.Tensor, ...],
    ) -> tuple[torch.Tensor, ...]:
        hidden = output[0]  # (batch, seq, hidden)
        correction = (alpha * weight) * delta.to(device=hidden.device, dtype=hidden.dtype)
        hidden = hidden.clone()
        hidden[:, -1, :] += correction
        return (hidden,) + output[1:]
    return hook_fn


def make_neuron_targeted_hook(
    delta: torch.Tensor,
    alpha: float,
    weight: float,
    neuron_corrections: list[tuple[int, float, float]],
    neuron_stds: dict[int, float],
) -> Any:
    """
    Like make_field_hook, but also adds per-neuron corrections after
    the field steering. Used for neuron_targeted condition.
    """
    def hook_fn(
        module: nn.Module,
        input: tuple[torch.Tensor, ...],
        output: tuple[torch.Tensor, ...],
    ) -> tuple[torch.Tensor, ...]:
        hidden = output[0]  # (batch, seq, hidden)
        hidden = hidden.clone()
        # Field correction
        correction = (alpha * weight) * delta.to(device=hidden.device, dtype=hidden.dtype)
        hidden[:, -1, :] += correction
        # Neuron-level corrections
        for dim, sign, scale in neuron_corrections:
            std = neuron_stds.get(dim, 1.0)
            hidden[:, -1, dim] += sign * scale * std
        return (hidden,) + output[1:]
    return hook_fn


# ---------------------------------------------------------------------------
# Steering conditions
# ---------------------------------------------------------------------------

def build_condition_hooks(
    condition: str,
    alpha: float,
    deltas: torch.Tensor,   # (36, 4096) — per-layer sarcasm direction
    neuron_corrections: list[tuple[int, float, float]],
    neuron_stds: dict[int, float],
) -> dict[int, Any]:
    """
    Returns {layer_idx: hook_fn} for a given condition.
    Returns empty dict for baseline.
    """
    if condition == "baseline":
        return {}

    if condition == "flat_all_36":
        hooks = {}
        for layer_idx in range(36):
            d = deltas[layer_idx]  # (4096,)
            hooks[layer_idx] = make_field_hook(d, alpha, 1.0)
        return hooks

    if condition == "surgical_3layer":
        hooks = {}
        for layer_idx, weight in SURGICAL_3_LAYERS.items():
            d = deltas[layer_idx]
            hooks[layer_idx] = make_field_hook(d, alpha, weight)
        return hooks

    if condition == "surgical_5layer":
        hooks = {}
        for layer_idx, weight in SURGICAL_5_LAYERS.items():
            d = deltas[layer_idx]
            hooks[layer_idx] = make_field_hook(d, alpha, weight)
        return hooks

    if condition == "surgical_3layer_boosted":
        hooks = {}
        boosted_alpha = alpha * 3.0
        for layer_idx, weight in SURGICAL_3_LAYERS.items():
            d = deltas[layer_idx]
            hooks[layer_idx] = make_field_hook(d, boosted_alpha, weight)
        return hooks

    if condition == "patching_weighted":
        hooks = {}
        for layer_idx, weight in PATCHING_WEIGHTS.items():
            if weight == 0.0:
                continue  # skip zero-weight layers (no-op)
            d = deltas[layer_idx]
            # Negative weights negate direction (suppress sarcasm = boost anti-sarcasm)
            hooks[layer_idx] = make_field_hook(d, alpha, weight)
        return hooks

    if condition == "neuron_targeted":
        hooks = {}
        for layer_idx, weight in PATCHING_WEIGHTS.items():
            if weight == 0.0:
                # Still add neuron corrections even at zero-weight layers
                # Use neuron-only hook (field correction is zero)
                def _make_neuron_only_hook(nc, ns):
                    def hook_fn(module, input, output):
                        hidden = output[0]
                        hidden = hidden.clone()
                        for dim, sign, scale in nc:
                            std = ns.get(dim, 1.0)
                            hidden[:, -1, dim] += sign * scale * std
                        return (hidden,) + output[1:]
                    return hook_fn
                hooks[layer_idx] = _make_neuron_only_hook(neuron_corrections, neuron_stds)
            else:
                d = deltas[layer_idx]
                hooks[layer_idx] = make_neuron_targeted_hook(
                    d, alpha, weight, neuron_corrections, neuron_stds
                )
        return hooks

    raise ValueError(f"Unknown condition: {condition!r}")


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

@torch.inference_mode()
def generate_with_hooks(
    model: nn.Module,
    processor: Any,
    prompt_text: str,
    layer_hooks: dict[int, Any],
    layers: list[nn.Module],
    device: torch.device,
    max_new_tokens: int = 256,
    temperature: float = 0.75,
    top_p: float = 0.9,
    repetition_penalty: float = 1.1,
) -> str:
    """
    Register hooks, generate one response, remove hooks.
    Returns decoded response string (stripped of prompt).
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt_text},
    ]
    # Qwen3-VL processor expects a list of messages
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = processor(
        text=[text],
        return_tensors="pt",
        padding=True,
    ).to(device)

    handles: list[Any] = []
    for layer_idx, hook_fn in layer_hooks.items():
        handle = layers[layer_idx].register_forward_hook(hook_fn)
        handles.append(handle)

    try:
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            do_sample=temperature > 0.0,
        )
    finally:
        for handle in handles:
            handle.remove()

    # Decode only the new tokens
    input_len = inputs["input_ids"].shape[1]
    new_tokens = outputs[0][input_len:]
    response = processor.decode(new_tokens, skip_special_tokens=True)
    return response.strip()


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

def run_evaluation(
    model: nn.Module,
    processor: Any,
    layers: list[nn.Module],
    device: torch.device,
    connectome: torch.Tensor,  # (20, 36, 4096)
    sarcasm_markers: list[str],
    assistant_markers: list[str],
    alphas: list[float],
    output_dir: Path,
) -> None:
    """Run all conditions × alphas and save results."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Extract sarcasm direction vectors per layer (category 6 = Sarcastic)
    # Shape: (36, 4096), float32
    raw_directions = connectome[SARCASM_CATEGORY_IDX]  # (36, 4096)

    # L2-normalise each layer's direction vector to get a unit steering direction
    norms = raw_directions.norm(dim=-1, keepdim=True).clamp(min=1e-8)  # (36, 1)
    deltas = raw_directions / norms  # (36, 4096), unit vectors

    conditions = [
        "baseline",
        "flat_all_36",
        "surgical_3layer",
        "surgical_5layer",
        "surgical_3layer_boosted",
        "patching_weighted",
        "neuron_targeted",
    ]

    # Baseline only needs alpha=0 (no effect), but we run it once per alpha for completeness
    # Actually baseline is alpha-independent — run once and re-use
    all_results: dict[str, Any] = {}
    all_responses: dict[str, Any] = {}

    checkpoint_path = output_dir / "checkpoint.json"

    # --- Baseline (run once) ---
    print("\n" + "=" * 60)
    print("Condition: baseline  (no steering)")
    print("=" * 60)
    baseline_scores: list[dict[str, Any]] = []
    baseline_responses: list[dict[str, Any]] = []

    for prompt_info in tqdm(EVAL_PROMPTS, desc="baseline", unit="prompt"):
        response = generate_with_hooks(
            model, processor, prompt_info["text"],
            layer_hooks={}, layers=layers, device=device,
        )
        score = score_response(response, sarcasm_markers, assistant_markers)
        baseline_scores.append(score)
        baseline_responses.append({
            "category": prompt_info["category"],
            "prompt": prompt_info["text"],
            "response": response,
            "scores": score,
        })

    baseline_agg = aggregate_scores(baseline_scores)
    all_results["baseline"] = {"alpha": 0.0, "aggregated": baseline_agg}
    all_responses["baseline"] = baseline_responses

    print(f"  sarcasm_rate={baseline_agg['sarcasm_rate']:.2%}  "
          f"assistant_rate={baseline_agg['assistant_rate']:.2%}  "
          f"coherence_rate={baseline_agg['coherence_rate']:.2%}")

    torch.cuda.empty_cache()

    # --- All other conditions × alphas ---
    for condition in conditions:
        if condition == "baseline":
            continue

        all_results[condition] = {}
        all_responses[condition] = {}

        for alpha in alphas:
            cond_key = f"{condition}__alpha_{alpha}"
            print(f"\n{'=' * 60}")
            print(f"Condition: {condition}  alpha={alpha}")
            print("=" * 60)

            layer_hooks = build_condition_hooks(
                condition, alpha, deltas,
                NEURON_CORRECTIONS, NEURON_STDS,
            )

            cond_scores: list[dict[str, Any]] = []
            cond_responses: list[dict[str, Any]] = []

            for prompt_info in tqdm(EVAL_PROMPTS, desc=cond_key, unit="prompt"):
                response = generate_with_hooks(
                    model, processor, prompt_info["text"],
                    layer_hooks=layer_hooks, layers=layers, device=device,
                )
                score = score_response(response, sarcasm_markers, assistant_markers)
                cond_scores.append(score)
                cond_responses.append({
                    "category": prompt_info["category"],
                    "prompt": prompt_info["text"],
                    "response": response,
                    "scores": score,
                })

            cond_agg = aggregate_scores(cond_scores)
            all_results[condition][f"alpha_{alpha}"] = {
                "alpha": alpha,
                "aggregated": cond_agg,
                "n_active_layers": len(layer_hooks),
            }
            all_responses[condition][f"alpha_{alpha}"] = cond_responses

            print(f"  sarcasm_rate={cond_agg['sarcasm_rate']:.2%}  "
                  f"assistant_rate={cond_agg['assistant_rate']:.2%}  "
                  f"coherence_rate={cond_agg['coherence_rate']:.2%}  "
                  f"active_layers={len(layer_hooks)}")

            # Checkpoint after each condition × alpha
            checkpoint_data = {
                "results": all_results,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "last_completed": cond_key,
            }
            with open(checkpoint_path, "w") as f:
                json.dump(checkpoint_data, f, indent=2)

            torch.cuda.empty_cache()

    # --- Save final outputs ---
    results_path = output_dir / "surgical_results.json"
    responses_path = output_dir / "surgical_responses.json"

    # Add metadata
    final_results = {
        "metadata": {
            "model": MODEL_NAME,
            "n_prompts": len(EVAL_PROMPTS),
            "alphas_tested": alphas,
            "conditions": conditions,
            "sarcasm_category_idx": SARCASM_CATEGORY_IDX,
            "surgical_3_layers": SURGICAL_3_LAYERS,
            "surgical_5_layers": SURGICAL_5_LAYERS,
            "patching_weights": PATCHING_WEIGHTS,
            "neuron_corrections": [
                {"dim": d, "sign": s, "scale": sc} for d, s, sc in NEURON_CORRECTIONS
            ],
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        },
        "results": all_results,
    }

    with open(results_path, "w") as f:
        json.dump(final_results, f, indent=2)

    with open(responses_path, "w") as f:
        json.dump(all_responses, f, indent=2)

    print(f"\n[done] Results saved to: {results_path}")
    print(f"[done] Responses saved to: {responses_path}")

    # Print summary table
    print_summary_table(all_results, alphas, conditions)


def print_summary_table(
    all_results: dict[str, Any],
    alphas: list[float],
    conditions: list[str],
) -> None:
    """Print a readable summary table to stdout."""
    print("\n" + "=" * 80)
    print("SUMMARY: Sarcasm Rate by Condition and Alpha")
    print("=" * 80)

    # Header
    header = f"{'Condition':<30}  {'Alpha':>6}  {'Sarc%':>6}  {'Asst%':>6}  {'Coh%':>6}  {'Layers':>6}"
    print(header)
    print("-" * 80)

    # Baseline
    if "baseline" in all_results:
        agg = all_results["baseline"]["aggregated"]
        print(f"{'baseline':<30}  {'N/A':>6}  "
              f"{agg['sarcasm_rate']:>6.1%}  "
              f"{agg['assistant_rate']:>6.1%}  "
              f"{agg['coherence_rate']:>6.1%}  "
              f"{'0':>6}")
        print("-" * 80)

    for condition in conditions:
        if condition == "baseline":
            continue
        if condition not in all_results:
            continue
        for alpha in alphas:
            key = f"alpha_{alpha}"
            if key not in all_results[condition]:
                continue
            entry = all_results[condition][key]
            agg = entry["aggregated"]
            n_layers = entry.get("n_active_layers", "?")
            print(f"{condition:<30}  {alpha:>6.1f}  "
                  f"{agg['sarcasm_rate']:>6.1%}  "
                  f"{agg['assistant_rate']:>6.1%}  "
                  f"{agg['coherence_rate']:>6.1%}  "
                  f"{n_layers:>6}")
        print("-" * 80)


# ---------------------------------------------------------------------------
# Argument parsing and entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Surgical activation steering experiment: 3-layer vs 36-layer ActAdd."
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="./surgical_steering_results",
        help="Output directory for results (default: ./surgical_steering_results)",
    )
    parser.add_argument(
        "--connectome",
        type=str,
        default=None,
        help="Path to connectome_zscores.pt (auto-detected if omitted)",
    )
    parser.add_argument(
        "--markers",
        type=str,
        default=None,
        help="Path to sarcasm_markers.json (auto-detected if omitted)",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=None,
        help=(
            "Single alpha to test (default: test [2.0, 5.0, 10.0, 15.0]). "
            "Overrides the full alpha sweep."
        ),
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device string, e.g. 'cuda:0' (auto-selected if omitted)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Device selection
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"[device] Using: {device}")

    # Alpha schedule
    if args.alpha is not None:
        alphas = [args.alpha]
    else:
        alphas = [2.0, 5.0, 10.0, 15.0]
    print(f"[config] Alpha schedule: {alphas}")

    # -----------------------------------------------------------------------
    # Load connectome
    # -----------------------------------------------------------------------
    connectome_path = find_connectome(args.connectome)
    print(f"[load] Loading connectome from: {connectome_path}")
    connectome = torch.load(connectome_path, map_location="cpu", weights_only=False)
    if not isinstance(connectome, torch.Tensor):
        raise TypeError(f"Expected torch.Tensor, got {type(connectome)}")
    assert connectome.shape == (20, 36, 4096), (
        f"Expected connectome shape (20, 36, 4096), got {connectome.shape}"
    )
    print(f"[load] Connectome shape: {connectome.shape}  dtype: {connectome.dtype}")

    # -----------------------------------------------------------------------
    # Load sarcasm markers
    # -----------------------------------------------------------------------
    project_root = Path(__file__).parent

    if args.markers:
        markers_path = Path(args.markers)
    else:
        markers_path = project_root / "sarcasm_markers.json"

    if not markers_path.exists():
        raise FileNotFoundError(f"sarcasm_markers.json not found at: {markers_path}")

    print(f"[load] Loading markers from: {markers_path}")
    with open(markers_path) as f:
        markers_data = json.load(f)

    sarcasm_markers: list[str] = markers_data["flat_sarcasm_list"]
    assistant_markers: list[str] = markers_data["assistant_markers"]
    print(f"[load] Sarcasm markers: {len(sarcasm_markers)}  "
          f"Assistant markers: {len(assistant_markers)}")

    # -----------------------------------------------------------------------
    # Load model (check cache first — CLAUDE.md mandate)
    # -----------------------------------------------------------------------
    print(f"\n[cache] Checking HF cache for {MODEL_NAME!r} ...")
    cached = model_cached(MODEL_NAME)
    print(f"[cache] Model cached: {cached}")

    if not cached:
        print(
            f"[warn] Model not found in HF cache at {HF_CACHE}.\n"
            "       Downloading may require ~16 GB. Set HF_HOME to redirect cache."
        )

    print(f"[load] Loading processor ...")
    processor = AutoProcessor.from_pretrained(MODEL_NAME, trust_remote_code=True)

    print(f"[load] Loading model (dtype=bfloat16) ...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_NAME,
        dtype=torch.bfloat16,
        device_map=str(device),
        trust_remote_code=True,
    )
    model.eval()
    print(f"[load] Model loaded. Parameters: {sum(p.numel() for p in model.parameters()) / 1e9:.1f}B")

    # -----------------------------------------------------------------------
    # Get layer references
    # -----------------------------------------------------------------------
    # Architecture note: Qwen3-VL-8B-Instruct layers at model.model.language_model.layers
    try:
        layers = list(model.model.language_model.layers)
    except AttributeError as e:
        raise AttributeError(
            "Could not find model.model.language_model.layers. "
            f"Check model architecture. Original error: {e}"
        )

    assert len(layers) == 36, f"Expected 36 layers, found {len(layers)}"
    print(f"[arch] Found {len(layers)} transformer layers at model.model.language_model.layers")

    # -----------------------------------------------------------------------
    # Run evaluation
    # -----------------------------------------------------------------------
    output_dir = Path(args.output)
    print(f"\n[eval] Output directory: {output_dir}")
    print(f"[eval] Conditions: baseline + 6 steering conditions")
    print(f"[eval] Prompts per condition×alpha: {len(EVAL_PROMPTS)}")
    print(f"[eval] Starting evaluation ...\n")

    run_evaluation(
        model=model,
        processor=processor,
        layers=layers,
        device=device,
        connectome=connectome,
        sarcasm_markers=sarcasm_markers,
        assistant_markers=assistant_markers,
        alphas=alphas,
        output_dir=output_dir,
    )


if __name__ == "__main__":
    main()
