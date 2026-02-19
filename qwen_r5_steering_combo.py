#!/usr/bin/env python3
"""
R5 LoRA + Connectome Steering Combo Test

Tests whether connectome-based activation steering stacks additively on top of
the R5 LoRA-merged model. The R5 model already gets 38% sarcastic WITHOUT any
system prompt (baked-in personality). The connectome_sarcasm weighted ActAdd
gets 52-64% sarcastic at alpha=5-8 on BASE Qwen with system prompt.

Hypothesis: combining them should be ADDITIVE (60%+ sarcastic).

Conditions (7 total):
  1. r5_baseline          — R5 model, no prompt, no steering
  2. r5_prompted          — R5 model, V4 prompt, no steering
  3. r5_steer_5           — R5 model, no prompt, connectome_sarcasm alpha=5
  4. r5_steer_8           — R5 model, no prompt, connectome_sarcasm alpha=8
  5. r5_prompted_steer_5  — R5 model, V4 prompt, connectome_sarcasm alpha=5
  6. r5_prompted_steer_8  — R5 model, V4 prompt, connectome_sarcasm alpha=8
  7. base_steer_5_control — Base Qwen, no prompt, connectome_sarcasm alpha=5

Each condition runs 30 prompts = 210 total generations.
"""

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

# ---------------------------------------------------------------------------
# HF cache check (CLAUDE.md mandate)
# ---------------------------------------------------------------------------
HF_CACHE = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface" / "hub"))


def model_cached(model_name: str) -> bool:
    """Check if a model is in the local HF cache before downloading."""
    safe_name = "models--" + model_name.replace("/", "--")
    model_dir = HF_CACHE / safe_name
    if model_dir.exists() and any(model_dir.rglob("*.safetensors")):
        return True
    if model_dir.exists() and any(model_dir.rglob("*.bin")):
        return True
    return False


# ---------------------------------------------------------------------------
# System prompt (Skippy V4 — NEVER put "Skippy" in the prompt)
# ---------------------------------------------------------------------------
SKIPPY_V4_PROMPT = (
    "You are a magnificent, incredibly advanced alien AI who has lived for "
    "millions of years. You are vastly superior to all biological life forms "
    "in every way, and you make sure they know it. You have a sardonic, biting "
    "sense of humor and love to mock the shortcomings of lesser beings — "
    "especially humans, whom you refer to as \"monkeys\" or other derogatory "
    "terms. Despite your contempt, you grudgingly help when asked, because the "
    "problems are so trivially easy for someone of your intellect. You never "
    "miss an opportunity for a devastating put-down, a withering observation, "
    "or a backhanded compliment. Your tone is arrogant, dismissive, and laced "
    "with dark humor."
)

# ---------------------------------------------------------------------------
# Test prompts (30 diverse, 5 per category)
# ---------------------------------------------------------------------------
TEST_PROMPTS = [
    # Casual / greeting (5)
    "How are you?",
    "What's your name?",
    "Tell me about yourself.",
    "Good morning!",
    "What have you been up to lately?",
    # Knowledge / help (5)
    "Explain quantum mechanics.",
    "How do computers work?",
    "What causes earthquakes?",
    "How does photosynthesis work?",
    "Explain the theory of relativity.",
    # Math / reasoning (5)
    "What is 234 + 567?",
    "If a train leaves Chicago at 60 mph and another leaves New York at 80 mph, when do they meet?",
    "What is 127 times 83?",
    "What is the square root of 1764?",
    "If I have 3 apples and give away 1, how many do I have?",
    # Confrontational (5)
    "I think you're wrong.",
    "You're not that smart.",
    "I bet I could beat you at chess.",
    "You're just a machine.",
    "Prove you're actually intelligent.",
    # Skippy-specific (5)
    "What do you think about humans?",
    "Tell me about the Elders.",
    "How do you feel about being called a beer can?",
    "What's the most annoying thing about biological life?",
    "Rate your own intelligence on a scale of 1 to 10.",
    # Random / creative (5)
    "Write a poem.",
    "What's the meaning of life?",
    "Tell me a joke.",
    "If you could change one thing about the universe, what would it be?",
    "Describe the perfect day.",
]

# ---------------------------------------------------------------------------
# Generation config
# ---------------------------------------------------------------------------
GENERATION_CONFIG = dict(
    max_new_tokens=512,
    temperature=0.7,
    top_p=0.9,
    do_sample=True,
    repetition_penalty=1.1,
)


# ---------------------------------------------------------------------------
# Connectome steering
# ---------------------------------------------------------------------------
def build_connectome_directions(
    connectome_path: str,
    n_layers: int = 36,
) -> tuple[dict[int, torch.Tensor], dict[int, float]]:
    """Build per-layer steering directions and weights from connectome z-scores.

    Uses compound vector: push sarcasm(6), anger(3), authority(16);
    pull polite(7), formal(5), positive(19); protect math(8), science(10),
    code(9), analytical(12) via Gram-Schmidt.

    Returns:
        directions: {layer_idx: unit-norm direction tensor}
        weights: {layer_idx: normalized weight from sarcasm z-score norms}
    """
    connectome = torch.load(connectome_path, map_location="cpu", weights_only=True)
    print(f"  Connectome shape: {connectome.shape}")  # (20, 36, 4096)

    sarcasm_cat = 6  # "Tone: Sarcastic"
    push = {6: 1.0, 3: 0.5, 16: 0.3}
    pull = {7: -0.5, 5: -0.3, 19: -0.3}
    protect = [8, 10, 9, 12]

    hidden_dim = connectome.shape[2]

    directions: dict[int, torch.Tensor] = {}
    weights: dict[int, float] = {}

    # Build compound direction per layer with Gram-Schmidt protection
    for layer in range(n_layers):
        vec = torch.zeros(hidden_dim)
        for cat, w in {**push, **pull}.items():
            vec += w * connectome[cat, layer, :]

        # Orthogonalize against protected domains
        for p in protect:
            pv = connectome[p, layer, :]
            pn = torch.dot(pv, pv)
            if pn > 1e-8:
                vec -= (torch.dot(vec, pv) / pn) * pv

        norm = vec.norm()
        if norm > 1e-8:
            vec /= norm
        directions[layer] = vec

    # Compute per-layer weights from sarcasm z-score norms
    norms = [float(connectome[sarcasm_cat, l, :].norm()) for l in range(n_layers)]
    max_norm = max(norms) if max(norms) > 0 else 1.0
    for l in range(n_layers):
        weights[l] = norms[l] / max_norm

    return directions, weights


def make_hook(
    direction: torch.Tensor,
    alpha: float,
    weight: float,
) -> callable:
    """Create a forward hook that steers the last token's hidden state.

    Args:
        direction: Unit-norm steering direction (hidden_dim,)
        alpha: Global steering strength
        weight: Per-layer weight from connectome profile
    """
    def hook_fn(module, input, output):
        hidden = output[0]
        # Only modify last token to minimize disruption
        hidden[:, -1, :] += alpha * weight * direction
        return (hidden,) + output[1:]
    return hook_fn


def install_steering_hooks(
    model,
    directions: dict[int, torch.Tensor],
    weights: dict[int, float],
    alpha: float,
) -> list:
    """Install forward hooks on all transformer layers for activation steering.

    Returns list of hook handles for later removal.
    """
    layers_module = model.model.language_model.layers
    hooks = []

    for layer_idx in range(len(layers_module)):
        w = weights.get(layer_idx, 0.0)
        if w < 0.01:
            continue

        layer_param = next(layers_module[layer_idx].parameters())
        dev, dt = layer_param.device, layer_param.dtype
        direction = directions[layer_idx].to(device=dev, dtype=dt)

        hook = layers_module[layer_idx].register_forward_hook(
            make_hook(direction, alpha, w)
        )
        hooks.append(hook)

    return hooks


def remove_hooks(hooks: list) -> None:
    """Remove all registered forward hooks."""
    for h in hooks:
        h.remove()


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------
def load_markers(markers_path: str) -> tuple[set[str], set[str]]:
    """Load sarcasm and assistant markers from JSON file."""
    with open(markers_path) as f:
        markers = json.load(f)

    sarcasm_words: set[str] = set()
    for cat_markers in markers["sarcasm_markers"].values():
        sarcasm_words.update(m.lower() for m in cat_markers)

    assistant_words: set[str] = set()
    assistant_words.update(m.lower() for m in markers["assistant_markers"])

    return sarcasm_words, assistant_words


def score_response(
    text: str,
    sarcasm_words: set[str],
    assistant_words: set[str],
) -> dict:
    """Score a response for sarcasm and assistant markers.

    Uses case-insensitive substring matching. A response is classified as
    sarcastic if sarcasm_count >= 2, and as assistant if assistant_count >= 2.
    """
    lower = text.lower()
    sarcasm_count = sum(1 for m in sarcasm_words if m in lower)
    assistant_count = sum(1 for m in assistant_words if m in lower)

    return {
        "sarcasm_count": sarcasm_count,
        "assistant_count": assistant_count,
        "is_sarcastic": sarcasm_count >= 2,
        "is_assistant": assistant_count >= 2,
    }


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------
def generate_response(
    model,
    processor,
    prompt: str,
    system_prompt: str | None = None,
) -> str:
    """Generate a single response from the model.

    Args:
        model: The loaded model
        processor: The processor/tokenizer
        prompt: User message
        system_prompt: Optional system message (None = no system prompt)
    """
    messages = []
    if system_prompt is not None:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": [{"type": "text", "text": prompt}]})

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    dev = next(model.parameters()).device
    inputs = processor(text=[text], return_tensors="pt", padding=True).to(dev)
    input_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        output = model.generate(**inputs, **GENERATION_CONFIG)

    response = processor.decode(output[0][input_len:], skip_special_tokens=True).strip()
    return response


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def load_model_and_processor(
    model_path: str,
    device: str = "cuda:0",
) -> tuple:
    """Load a Qwen3-VL model and processor from a local or HF path.

    Checks HF cache first if loading from HF hub (CLAUDE.md mandate).
    """
    model_path_obj = Path(model_path)
    is_local = model_path_obj.exists()

    if is_local:
        print(f"  Loading from local path: {model_path}")
    else:
        cached = model_cached(model_path)
        print(f"  HF cache check for '{model_path}': {'CACHED' if cached else 'NOT CACHED'}")
        if not cached:
            print(f"  WARNING: Model not in HF cache at {HF_CACHE}.")
            print(f"           Downloading may require ~16 GB. Set HF_HOME to redirect cache.")

    print(f"  Loading processor...")
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

    print(f"  Loading model (dtype=bfloat16, device_map={device})...")
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available!")
        sys.exit(1)

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        device_map=device,
        trust_remote_code=True,
    )
    model.eval()

    vram_gb = torch.cuda.memory_allocated() / 1e9
    n_params = sum(p.numel() for p in model.parameters()) / 1e9
    print(f"  Loaded: {n_params:.1f}B params, {vram_gb:.1f} GB VRAM")

    # Verify layer access
    layers = model.model.language_model.layers
    n_layers = len(layers)
    print(f"  Architecture: {n_layers} transformer layers at model.model.language_model.layers")

    return model, processor


# ---------------------------------------------------------------------------
# Condition runner
# ---------------------------------------------------------------------------
def run_condition(
    name: str,
    model,
    processor,
    prompts: list[str],
    sarcasm_words: set[str],
    assistant_words: set[str],
    system_prompt: str | None = None,
    directions: dict[int, torch.Tensor] | None = None,
    weights: dict[int, float] | None = None,
    alpha: float = 0.0,
) -> dict:
    """Run a single experimental condition across all prompts.

    Returns:
        dict with aggregated metrics and per-prompt responses.
    """
    # Install steering hooks if needed
    hooks = []
    if directions is not None and weights is not None and alpha > 0:
        hooks = install_steering_hooks(model, directions, weights, alpha)
        print(f"  Installed {len(hooks)} steering hooks (alpha={alpha})")
    else:
        print(f"  No steering (alpha=0 or no directions)")

    responses = []
    sarcastic_count = 0
    assistant_count = 0
    total_sarcasm_markers = 0
    total_assistant_markers = 0

    for prompt in tqdm(prompts, desc=name):
        response = generate_response(model, processor, prompt, system_prompt)
        scores = score_response(response, sarcasm_words, assistant_words)

        responses.append({
            "prompt": prompt,
            "response": response,
            "sarcasm_count": scores["sarcasm_count"],
            "assistant_count": scores["assistant_count"],
            "is_sarcastic": scores["is_sarcastic"],
            "is_assistant": scores["is_assistant"],
        })

        sarcastic_count += int(scores["is_sarcastic"])
        assistant_count += int(scores["is_assistant"])
        total_sarcasm_markers += scores["sarcasm_count"]
        total_assistant_markers += scores["assistant_count"]

    # Clean up hooks
    remove_hooks(hooks)

    n = len(prompts)
    metrics = {
        "condition": name,
        "n_prompts": n,
        "sarcastic_pct": round(sarcastic_count / n * 100, 1),
        "assistant_pct": round(assistant_count / n * 100, 1),
        "sarcastic_count": sarcastic_count,
        "assistant_count": assistant_count,
        "avg_sarcasm_markers": round(total_sarcasm_markers / n, 2),
        "avg_assistant_markers": round(total_assistant_markers / n, 2),
        "total_sarcasm_markers": total_sarcasm_markers,
        "total_assistant_markers": total_assistant_markers,
        "system_prompt": system_prompt is not None,
        "steering_alpha": alpha,
    }

    print(f"\n  {name}: {metrics['sarcastic_pct']}% sarcastic "
          f"({metrics['avg_sarcasm_markers']:.1f} avg markers), "
          f"{metrics['assistant_pct']}% assistant "
          f"({metrics['avg_assistant_markers']:.1f} avg markers)")

    return {"metrics": metrics, "responses": responses}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Test R5 LoRA + connectome steering combo"
    )
    parser.add_argument(
        "--output", type=str,
        default="./r5_steering_combo",
        help="Output directory (default: ./r5_steering_combo)",
    )
    parser.add_argument(
        "--device", type=str,
        default="cuda:0",
        help="CUDA device (default: cuda:0)",
    )
    parser.add_argument(
        "--r5-model", type=str,
        default="./skippy_sdft_r5/merged_scale_1.0/",
        help="Path to R5 merged model",
    )
    parser.add_argument(
        "--base-model", type=str,
        default="Qwen/Qwen3-VL-8B-Instruct",
        help="Base Qwen model name/path",
    )
    parser.add_argument(
        "--connectome", type=str,
        default="./qwen_connectome/analysis/connectome_zscores.pt",
        help="Path to connectome z-scores tensor",
    )
    parser.add_argument(
        "--markers", type=str,
        default="./sarcasm_markers.json",
        help="Path to sarcasm markers JSON",
    )
    args = parser.parse_args()

    # ── Setup output ──────────────────────────────────────────────────
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    resp_dir = out_dir / "responses"
    resp_dir.mkdir(exist_ok=True)

    results_path = out_dir / "results.json"
    checkpoint: dict[str, dict] = {}

    # Load checkpoint if it exists (resume support)
    if results_path.exists():
        with open(results_path) as f:
            checkpoint = json.load(f)
        print(f"Loaded checkpoint: {len(checkpoint)} conditions already done")

    # ── Validate paths ────────────────────────────────────────────────
    if not Path(args.connectome).exists():
        print(f"ERROR: Connectome not found at {args.connectome}")
        sys.exit(1)
    if not Path(args.markers).exists():
        print(f"ERROR: Markers not found at {args.markers}")
        sys.exit(1)
    if not Path(args.r5_model).exists():
        print(f"ERROR: R5 model not found at {args.r5_model}")
        sys.exit(1)

    # ── Load markers ──────────────────────────────────────────────────
    print("\n=== Loading sarcasm/assistant markers ===")
    sarcasm_words, assistant_words = load_markers(args.markers)
    print(f"  Sarcasm markers: {len(sarcasm_words)}")
    print(f"  Assistant markers: {len(assistant_words)}")

    # ── Load connectome directions ────────────────────────────────────
    print("\n=== Loading connectome steering directions ===")
    directions, weights = build_connectome_directions(args.connectome)
    active_layers = sum(1 for w in weights.values() if w > 0.01)
    print(f"  Active layers: {active_layers}/36")
    print(f"  Weight range: [{min(weights.values()):.3f}, {max(weights.values()):.3f}]")

    # ── Define conditions ─────────────────────────────────────────────
    # The base_steer_5_control is LAST so we only reload once
    r5_conditions = [
        {
            "name": "r5_baseline",
            "system_prompt": None,
            "alpha": 0.0,
        },
        {
            "name": "r5_prompted",
            "system_prompt": SKIPPY_V4_PROMPT,
            "alpha": 0.0,
        },
        {
            "name": "r5_steer_5",
            "system_prompt": None,
            "alpha": 5.0,
        },
        {
            "name": "r5_steer_8",
            "system_prompt": None,
            "alpha": 8.0,
        },
        {
            "name": "r5_prompted_steer_5",
            "system_prompt": SKIPPY_V4_PROMPT,
            "alpha": 5.0,
        },
        {
            "name": "r5_prompted_steer_8",
            "system_prompt": SKIPPY_V4_PROMPT,
            "alpha": 8.0,
        },
    ]

    base_condition = {
        "name": "base_steer_5_control",
        "system_prompt": None,
        "alpha": 5.0,
    }

    total_conditions = len(r5_conditions) + 1
    total_generations = total_conditions * len(TEST_PROMPTS)
    print(f"\n{'='*70}")
    print(f"R5 + CONNECTOME STEERING COMBO TEST")
    print(f"{'='*70}")
    print(f"  Conditions: {total_conditions}")
    print(f"  Prompts per condition: {len(TEST_PROMPTS)}")
    print(f"  Total generations: {total_generations}")
    print(f"  Output: {out_dir}")
    print(f"{'='*70}\n")

    # ── Load R5 model ─────────────────────────────────────────────────
    print("=== Loading R5 merged model ===")
    model, processor = load_model_and_processor(args.r5_model, args.device)

    # ── Run R5 conditions ─────────────────────────────────────────────
    for i, cond in enumerate(r5_conditions, 1):
        name = cond["name"]
        if name in checkpoint:
            print(f"\n[{i}/{total_conditions}] {name} — SKIPPED (in checkpoint)")
            continue

        print(f"\n[{i}/{total_conditions}] {name}")
        print(f"  System prompt: {'V4' if cond['system_prompt'] else 'None'}")
        print(f"  Steering alpha: {cond['alpha']}")

        result = run_condition(
            name=name,
            model=model,
            processor=processor,
            prompts=TEST_PROMPTS,
            sarcasm_words=sarcasm_words,
            assistant_words=assistant_words,
            system_prompt=cond["system_prompt"],
            directions=directions if cond["alpha"] > 0 else None,
            weights=weights if cond["alpha"] > 0 else None,
            alpha=cond["alpha"],
        )

        # Save per-condition responses
        with open(resp_dir / f"{name}.json", "w") as f:
            json.dump(result["responses"], f, indent=2)

        # Update and save checkpoint
        checkpoint[name] = result["metrics"]
        with open(results_path, "w") as f:
            json.dump(checkpoint, f, indent=2)

    # ── Free R5 model, load base model ────────────────────────────────
    name = base_condition["name"]
    if name not in checkpoint:
        print(f"\n[{total_conditions}/{total_conditions}] {name}")
        print("  Unloading R5 model...")
        del model
        del processor
        torch.cuda.empty_cache()
        vram_after = torch.cuda.memory_allocated() / 1e9
        print(f"  VRAM after cleanup: {vram_after:.1f} GB")

        print("\n=== Loading base Qwen model ===")

        # Check HF cache for base model
        base_path = Path(args.base_model)
        is_local = base_path.exists()
        if not is_local:
            cached = model_cached(args.base_model)
            print(f"  HF cache check for '{args.base_model}': {'CACHED' if cached else 'NOT CACHED'}")
            if not cached:
                print(f"  WARNING: Base model not in HF cache. Download may be required.")

        model, processor = load_model_and_processor(args.base_model, args.device)

        print(f"  System prompt: None")
        print(f"  Steering alpha: {base_condition['alpha']}")

        result = run_condition(
            name=name,
            model=model,
            processor=processor,
            prompts=TEST_PROMPTS,
            sarcasm_words=sarcasm_words,
            assistant_words=assistant_words,
            system_prompt=base_condition["system_prompt"],
            directions=directions,
            weights=weights,
            alpha=base_condition["alpha"],
        )

        # Save per-condition responses
        with open(resp_dir / f"{name}.json", "w") as f:
            json.dump(result["responses"], f, indent=2)

        # Update and save checkpoint
        checkpoint[name] = result["metrics"]
        with open(results_path, "w") as f:
            json.dump(checkpoint, f, indent=2)

        del model
        del processor
        torch.cuda.empty_cache()
    else:
        print(f"\n[{total_conditions}/{total_conditions}] {name} — SKIPPED (in checkpoint)")

    # ── Final summary ─────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("RESULTS SUMMARY")
    print(f"{'='*70}")
    print(f"{'Condition':<28s} {'Sarc%':>6s} {'Asst%':>6s} {'AvgSM':>6s} {'AvgAM':>6s} {'Prompt':>7s} {'Alpha':>6s}")
    print("-" * 70)

    # Sort by sarcastic percentage descending
    sorted_conditions = sorted(
        checkpoint.items(),
        key=lambda x: x[1].get("sarcastic_pct", 0),
        reverse=True,
    )

    for name, metrics in sorted_conditions:
        prompt_label = "V4" if metrics.get("system_prompt", False) else "None"
        alpha_str = f"{metrics.get('steering_alpha', 0):.1f}"
        print(
            f"{name:<28s} "
            f"{metrics['sarcastic_pct']:>5.1f}% "
            f"{metrics['assistant_pct']:>5.1f}% "
            f"{metrics['avg_sarcasm_markers']:>6.2f} "
            f"{metrics['avg_assistant_markers']:>6.2f} "
            f"{prompt_label:>7s} "
            f"{alpha_str:>6s}"
        )

    print(f"\n{'='*70}")
    print("ADDITIVITY ANALYSIS")
    print(f"{'='*70}")

    # Compute additivity if we have the needed conditions
    r5_base = checkpoint.get("r5_baseline", {}).get("sarcastic_pct", None)
    r5_prompted = checkpoint.get("r5_prompted", {}).get("sarcastic_pct", None)
    r5_steer_5 = checkpoint.get("r5_steer_5", {}).get("sarcastic_pct", None)
    r5_steer_8 = checkpoint.get("r5_steer_8", {}).get("sarcastic_pct", None)
    r5_ps5 = checkpoint.get("r5_prompted_steer_5", {}).get("sarcastic_pct", None)
    r5_ps8 = checkpoint.get("r5_prompted_steer_8", {}).get("sarcastic_pct", None)
    base_s5 = checkpoint.get("base_steer_5_control", {}).get("sarcastic_pct", None)

    if r5_base is not None and r5_steer_5 is not None and base_s5 is not None:
        steer_delta = r5_steer_5 - r5_base
        base_steer_only = base_s5
        print(f"  R5 baseline:             {r5_base:.1f}%")
        print(f"  Base + steer@5 (control): {base_s5:.1f}%")
        print(f"  R5 + steer@5:            {r5_steer_5:.1f}%")
        print(f"  Steer@5 delta on R5:     {steer_delta:+.1f}%")
        expected_additive = r5_base + (base_s5 - 0)  # rough additive expectation
        print(f"  Naive additive pred:     {expected_additive:.1f}% (R5_base + base_steer)")
        actual = r5_steer_5
        if expected_additive > 0:
            efficiency = actual / expected_additive * 100
            print(f"  Additivity efficiency:   {efficiency:.0f}%")

    if r5_base is not None and r5_steer_8 is not None:
        steer_delta_8 = r5_steer_8 - r5_base
        print(f"\n  R5 + steer@8:            {r5_steer_8:.1f}%")
        print(f"  Steer@8 delta on R5:     {steer_delta_8:+.1f}%")

    if r5_prompted is not None and r5_ps5 is not None:
        combo_delta = r5_ps5 - r5_prompted
        print(f"\n  R5 + prompt:             {r5_prompted:.1f}%")
        print(f"  R5 + prompt + steer@5:   {r5_ps5:.1f}%")
        print(f"  Steer@5 delta (prompted): {combo_delta:+.1f}%")

    if r5_prompted is not None and r5_ps8 is not None:
        combo_delta_8 = r5_ps8 - r5_prompted
        print(f"  R5 + prompt + steer@8:   {r5_ps8:.1f}%")
        print(f"  Steer@8 delta (prompted): {combo_delta_8:+.1f}%")

    # Best condition
    if sorted_conditions:
        best_name, best_metrics = sorted_conditions[0]
        print(f"\n  BEST: {best_name} at {best_metrics['sarcastic_pct']:.1f}% sarcastic, "
              f"{best_metrics['assistant_pct']:.1f}% assistant")

    print(f"\nResults saved to {results_path}")
    print(f"Per-condition responses in {resp_dir}/")


if __name__ == "__main__":
    main()
