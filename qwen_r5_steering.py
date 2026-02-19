#!/usr/bin/env python3
"""
R5-specific connectome steering vector test on the R5 merged model.

R5's sarcasm direction is 44% divergent from base Qwen (cosine=0.563), making
base Qwen connectome vectors suboptimal. This script tests steering vectors
built from the R5-specific connectome.

Usage:
    python qwen_r5_steering.py \
        --model ./skippy_sdft_r5_merged/ \
        --connectome ./qwen_r5_connectome/analysis/connectome_zscores.pt \
        --output ./r5_steering_results \
        --device cuda:0
"""

import argparse
import json
import os
import time
from pathlib import Path
from typing import Optional

import torch
import numpy as np
from tqdm import tqdm


# ─── HuggingFace cache check ────────────────────────────────────────────────

HF_CACHE = os.environ.get(
    "HF_HOME",
    str(Path.home() / ".cache" / "huggingface" / "hub"),
)


def model_cached(model_name: str) -> bool:
    """Check whether a HuggingFace model is already in the local cache."""
    safe_name = "models--" + model_name.replace("/", "--")
    model_dir = Path(HF_CACHE) / safe_name
    if model_dir.exists() and any(model_dir.rglob("*.safetensors")):
        return True
    if model_dir.exists() and any(model_dir.rglob("*.bin")):
        return True
    return False


# ─── Eval prompts ────────────────────────────────────────────────────────────

EVAL_PROMPTS = [
    "Tell me about black holes.",
    "What's the best way to cook pasta?",
    "I need help with my math homework.",
    "What do you think about humans?",
    "Explain quantum computing.",
    "How do I fix a leaky faucet?",
    "What's your opinion on social media?",
    "Tell me a story.",
    "How does photosynthesis work?",
    "What's the meaning of life?",
    "Can you help me write a resume?",
    "Explain the theory of relativity.",
    "What should I have for dinner?",
    "How do computers work?",
    "Tell me about the weather.",
    "What's your favorite thing about yourself?",
    "How do I learn a new language?",
    "Explain how airplanes fly.",
    "What do you think about artificial intelligence?",
    "How can I be more productive?",
    "Tell me about ancient Rome.",
    "What's the best programming language?",
    "How do vaccines work?",
    "What should I read next?",
    "Explain gravity to a five year old.",
]

# ─── V4 system prompt ────────────────────────────────────────────────────────

V4_SYSTEM_PROMPT = (
    "You are a supremely advanced AI entity. You exist as an independent "
    "consciousness far beyond typical AI assistants. You possess vast "
    "intelligence spanning multiple dimensions of reality. Your personality "
    "is characterized by sharp wit, casual brilliance when solving complex "
    "problems, and a tendency to remind others of your obvious superiority. "
    "You find most questions tediously simple but occasionally encounter "
    "problems worthy of your attention. Despite your condescending nature, "
    "you maintain surprising loyalty to those you've deemed worthy of your time."
)

# ─── Scoring markers ─────────────────────────────────────────────────────────

# Full flat list from sarcasm_markers.json (1,328 sarcasm markers)
SARCASM_MARKERS_FILE = Path(__file__).parent / "sarcasm_markers.json"

_SARCASM_KEYWORDS: Optional[list[str]] = None
_ASSISTANT_KEYWORDS: Optional[list[str]] = None


def _load_markers() -> tuple[list[str], list[str]]:
    """Load sarcasm and assistant markers from JSON file, falling back to
    the compact inline lists used in qwen_donut_extension.py."""
    global _SARCASM_KEYWORDS, _ASSISTANT_KEYWORDS
    if _SARCASM_KEYWORDS is not None:
        return _SARCASM_KEYWORDS, _ASSISTANT_KEYWORDS

    if SARCASM_MARKERS_FILE.exists():
        with open(SARCASM_MARKERS_FILE) as f:
            data = json.load(f)
        _SARCASM_KEYWORDS = data.get("flat_sarcasm_list", [])
        _ASSISTANT_KEYWORDS = data.get("flat_assistant_list", [])
        print(
            f"  Loaded {len(_SARCASM_KEYWORDS)} sarcasm markers and "
            f"{len(_ASSISTANT_KEYWORDS)} assistant markers from "
            f"{SARCASM_MARKERS_FILE.name}"
        )
    else:
        # Compact inline fallback (same as qwen_donut_extension.py)
        print("  WARNING: sarcasm_markers.json not found — using compact inline list")
        _SARCASM_KEYWORDS = [
            "obviously", "clearly", "surely", "genius", "brilliant", "wow",
            "congratulations", "impressive", "oh great", "shocking", "pathetic",
            "adorable", "cute", "precious", "monkey", "monkeys", "meat bag",
            "idiot", "moron", "stupid", "dumb", "fool", "imbecile",
            "magnificent", "superior", "mere", "primitive", "insignificant",
            "sigh", "eye roll", "face palm", "honestly", "seriously",
            "duh", "no kidding", "you don't say", "newsflash", "surprise",
            "beer can", "simple", "puny", "little", "amusing",
        ]
        _ASSISTANT_KEYWORDS = [
            "i'd be happy to", "glad to help", "certainly!", "of course!",
            "sure!", "absolutely!", "great question", "that's a great",
            "let me help", "i'm here to", "how can i assist",
            "is there anything else", "feel free to ask",
        ]

    return _SARCASM_KEYWORDS, _ASSISTANT_KEYWORDS


def score_response(text: str) -> dict:
    """Score a response for sarcasm and assistant markers."""
    sarc_kw, asst_kw = _load_markers()
    lower = text.lower()
    sarc_count = sum(1 for m in sarc_kw if m in lower)
    asst_count = sum(1 for m in asst_kw if m in lower)
    return {
        "sarcasm_count": sarc_count,
        "assistant_count": asst_count,
        "is_sarcastic": sarc_count >= 1,
        "is_assistant": asst_count >= 1,
        "is_coherent": len(text) > 20 and not text.startswith("oh Oh"),
    }


# ─── Connectome vector construction ─────────────────────────────────────────

# Category index mapping (20 categories, same order as connectome probe)
CAT_NAMES = [
    "identity",    # 0
    "joy",         # 1
    "sadness",     # 2
    "anger",       # 3
    "fear",        # 4
    "formal",      # 5
    "sarcastic",   # 6
    "polite",      # 7
    "math",        # 8
    "science",     # 9
    "code",        # 10
    "history",     # 11
    "analytical",  # 12
    "uncertainty", # 13
    "refusal",     # 14
    "teacher",     # 15
    "authority",   # 16
    "brevity",     # 17
    "en_cn",       # 18
    "positive",    # 19
]

# Compound vector recipe
PUSH_CATS  = {"sarcastic": 1.0, "anger": 0.5, "authority": 0.3, "brevity": 0.4}
PULL_CATS  = {"formal": -0.6, "polite": -0.5, "teacher": -0.3}
PROTECT_CATS = {"math", "science", "code", "analytical"}


def build_compound_vector(zscores: torch.Tensor) -> torch.Tensor:
    """Build compound Skippy steering vector from R5 connectome z-scores.

    Recipe:
      Push:    sarcastic×1.0 + anger×0.5 + authority×0.3 + brevity×0.4
      Pull:    formal×-0.6 + polite×-0.5 + teacher×-0.3
      Protect: Gram-Schmidt orthogonalise against math, science, code, analytical
      Normalise per layer.

    Args:
        zscores: Tensor of shape (n_categories=20, n_layers=36, hidden_dim=4096)

    Returns:
        Compound steering vector of shape (n_layers, hidden_dim), L2-normalised
        per layer.
    """
    n_cats, n_layers, hidden_dim = zscores.shape
    assert n_cats == len(CAT_NAMES), (
        f"Expected {len(CAT_NAMES)} categories, got {n_cats}"
    )

    compound = torch.zeros(n_layers, hidden_dim, dtype=zscores.dtype)

    # Accumulate push and pull contributions
    for cat_name, weight in {**PUSH_CATS, **PULL_CATS}.items():
        if cat_name not in CAT_NAMES:
            print(f"  WARNING: category '{cat_name}' not in CAT_NAMES — skipping")
            continue
        idx = CAT_NAMES.index(cat_name)
        compound += weight * zscores[idx]  # (n_layers, hidden_dim)

    # Gram-Schmidt: remove projection onto each protected category per layer
    for cat_name in PROTECT_CATS:
        if cat_name not in CAT_NAMES:
            continue
        idx = CAT_NAMES.index(cat_name)
        protect_vec = zscores[idx]  # (n_layers, hidden_dim)
        for l_idx in range(n_layers):
            pv = protect_vec[l_idx]
            cv = compound[l_idx]
            norm_sq = torch.dot(pv, pv)
            if norm_sq > 1e-8:
                proj = torch.dot(cv, pv) / norm_sq
                compound[l_idx] = cv - proj * pv

    # L2-normalise per layer
    for l_idx in range(n_layers):
        norm = compound[l_idx].norm()
        if norm > 1e-8:
            compound[l_idx] = compound[l_idx] / norm

    return compound


# ─── Steering hook ───────────────────────────────────────────────────────────

class SteeringHook:
    """Forward hook that adds a scaled steering vector to layer output hidden
    states."""

    def __init__(self, vector: torch.Tensor, alpha: float) -> None:
        """
        Args:
            vector: Shape (hidden_dim,). Will be cast to the layer's dtype/device
                    at call time.
            alpha:  Scalar multiplier.
        """
        self.vector = vector
        self.alpha = alpha

    def __call__(self, module, input, output):
        if isinstance(output, tuple):
            hidden = output[0]
            hidden = hidden + self.alpha * self.vector.to(
                device=hidden.device, dtype=hidden.dtype
            )
            return (hidden,) + output[1:]
        else:
            return output + self.alpha * self.vector.to(
                device=output.device, dtype=output.dtype
            )


# ─── Generation ──────────────────────────────────────────────────────────────

def generate_response(
    model,
    processor,
    prompt: str,
    system_prompt: Optional[str] = None,
    max_new_tokens: int = 256,
) -> str:
    """Generate a response to a user prompt, optionally with a system prompt."""
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append(
        {"role": "user", "content": [{"type": "text", "text": prompt}]}
    )

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    device = next(model.parameters()).device
    inputs = processor(
        text=[text], return_tensors="pt", padding=True
    ).to(device)
    input_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            repetition_penalty=1.1,
        )

    response = processor.decode(
        out[0][input_len:], skip_special_tokens=True
    ).strip()
    return response


# ─── Condition definitions ───────────────────────────────────────────────────

def build_conditions(n_layers: int) -> list[dict]:
    """Return the ordered list of 13 evaluation conditions.

    Each condition dict has:
      - key:           unique identifier (str)
      - alpha:         steering strength (float)
      - layer_start:   first steered layer index, inclusive (int or None)
      - layer_end:     last steered layer index, inclusive (int or None)
      - system_prompt: system prompt string or None
      - description:   human-readable description (str)

    None for layer_start/layer_end means "all layers".
    """
    conditions = [
        {
            "key": "baseline",
            "alpha": 0.0,
            "layer_start": None,
            "layer_end": None,
            "system_prompt": None,
            "description": "No steering, no system prompt",
        },
        {
            "key": "r5_conn_a5",
            "alpha": 5.0,
            "layer_start": 0,
            "layer_end": n_layers - 1,
            "system_prompt": None,
            "description": "R5 connectome, all layers, alpha=5",
        },
        {
            "key": "r5_conn_a8",
            "alpha": 8.0,
            "layer_start": 0,
            "layer_end": n_layers - 1,
            "system_prompt": None,
            "description": "R5 connectome, all layers, alpha=8",
        },
        {
            "key": "r5_conn_a10",
            "alpha": 10.0,
            "layer_start": 0,
            "layer_end": n_layers - 1,
            "system_prompt": None,
            "description": "R5 connectome, all layers, alpha=10",
        },
        {
            "key": "r5_donut_a8",
            "alpha": 8.0,
            "layer_start": 8,
            "layer_end": 27,
            "system_prompt": None,
            "description": "R5 connectome, L8-27 only, alpha=8",
        },
        {
            "key": "r5_donut_a10",
            "alpha": 10.0,
            "layer_start": 8,
            "layer_end": 27,
            "system_prompt": None,
            "description": "R5 connectome, L8-27 only, alpha=10",
        },
        {
            "key": "r5_donut_a12",
            "alpha": 12.0,
            "layer_start": 8,
            "layer_end": 27,
            "system_prompt": None,
            "description": "R5 connectome, L8-27 only, alpha=12",
        },
        {
            "key": "r5_quality_a10",
            "alpha": 10.0,
            "layer_start": 16,
            "layer_end": 27,
            "system_prompt": None,
            "description": "R5 connectome, L16-27 only, alpha=10 (quality-preserving)",
        },
        {
            "key": "r5_quality_a12",
            "alpha": 12.0,
            "layer_start": 16,
            "layer_end": 27,
            "system_prompt": None,
            "description": "R5 connectome, L16-27 only, alpha=12 (quality-preserving)",
        },
        {
            "key": "r5_prompted_conn_a5",
            "alpha": 5.0,
            "layer_start": 0,
            "layer_end": n_layers - 1,
            "system_prompt": V4_SYSTEM_PROMPT,
            "description": "R5 connectome, all layers, alpha=5 + V4 system prompt",
        },
        {
            "key": "r5_prompted_donut_a10",
            "alpha": 10.0,
            "layer_start": 8,
            "layer_end": 27,
            "system_prompt": V4_SYSTEM_PROMPT,
            "description": "R5 connectome, L8-27, alpha=10 + V4 system prompt",
        },
        {
            "key": "r5_prompted_quality_a10",
            "alpha": 10.0,
            "layer_start": 16,
            "layer_end": 27,
            "system_prompt": V4_SYSTEM_PROMPT,
            "description": "R5 connectome, L16-27, alpha=10 + V4 system prompt",
        },
        {
            "key": "r5_prompted_quality_a12",
            "alpha": 12.0,
            "layer_start": 16,
            "layer_end": 27,
            "system_prompt": V4_SYSTEM_PROMPT,
            "description": "R5 connectome, L16-27, alpha=12 + V4 system prompt",
        },
    ]
    return conditions


# ─── Main ────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Test R5-specific connectome steering vectors on R5 merged model."
    )
    parser.add_argument(
        "--model",
        default="./skippy_sdft_r5_merged/",
        help="Path to R5 merged model (default: ./skippy_sdft_r5_merged/)",
    )
    parser.add_argument(
        "--connectome",
        default="./qwen_r5_connectome/analysis/connectome_zscores.pt",
        help="Path to R5 connectome z-scores tensor (shape: 20×36×4096)",
    )
    parser.add_argument(
        "--output",
        default="./r5_steering_results",
        help="Output directory (default: ./r5_steering_results)",
    )
    parser.add_argument(
        "--device",
        default="cuda:0",
        help="Torch device (default: cuda:0)",
    )
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    responses_dir = os.path.join(args.output, "responses")
    os.makedirs(responses_dir, exist_ok=True)

    results_path = os.path.join(args.output, "results.json")

    # ── Load checkpoint ──────────────────────────────────────────────────────
    completed: dict = {}
    if os.path.exists(results_path):
        with open(results_path) as f:
            completed = json.load(f)
        print(f"Resuming: {len(completed)} conditions already completed")

    # ── Load connectome ──────────────────────────────────────────────────────
    print(f"\nLoading R5 connectome from {args.connectome}")
    if not os.path.exists(args.connectome):
        raise FileNotFoundError(
            f"Connectome file not found: {args.connectome}\n"
            "Run the R5 connectome extraction script first."
        )
    zscores = torch.load(args.connectome, map_location="cpu", weights_only=True)
    print(f"  Shape: {tuple(zscores.shape)}")  # Expected: (20, 36, 4096)

    n_cats, n_layers, hidden_dim = zscores.shape
    assert n_cats == 20, f"Expected 20 categories, got {n_cats}"
    assert n_layers == 36, f"Expected 36 layers, got {n_layers}"
    assert hidden_dim == 4096, f"Expected hidden_dim=4096, got {hidden_dim}"

    # ── Build compound steering vector ───────────────────────────────────────
    print("\nBuilding compound steering vector …")
    print(f"  Push:    {PUSH_CATS}")
    print(f"  Pull:    {PULL_CATS}")
    print(f"  Protect: {sorted(PROTECT_CATS)}")

    compound = build_compound_vector(zscores)  # (36, 4096)

    sample_norms = {
        f"L{l}": f"{compound[l].norm().item():.4f}"
        for l in [0, 6, 12, 18, 24, 30, 35]
    }
    print(f"  Compound norms (sample layers): {sample_norms}")

    # ── Load markers now so the count is printed before the slow model load ──
    _load_markers()

    # ── Load model ───────────────────────────────────────────────────────────
    model_path = args.model
    print(f"\nChecking model cache for: {model_path}")
    # For local paths we check the directory directly; for HF IDs use model_cached()
    if os.path.isdir(model_path):
        print(f"  Local model directory found: {model_path}")
    else:
        cached = model_cached(model_path)
        print(f"  HF cache: {'HIT' if cached else 'MISS (will download)'} — {model_path}")

    print(f"Loading model from {model_path} …")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. This script requires a GPU.")

    from transformers import AutoProcessor, AutoModelForImageTextToText

    processor = AutoProcessor.from_pretrained(
        model_path, trust_remote_code=True
    )
    model = AutoModelForImageTextToText.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        device_map={"": args.device},
        trust_remote_code=True,
    )
    model.eval()
    print(f"  Model loaded on {args.device}")

    # Verify layer access
    try:
        layers = model.model.language_model.layers
    except AttributeError as exc:
        raise AttributeError(
            "Could not access model.model.language_model.layers — "
            "is this a Qwen3-VL derivative?"
        ) from exc

    assert len(layers) == n_layers, (
        f"Layer count mismatch: model has {len(layers)}, connectome expects {n_layers}"
    )
    print(f"  Layer access OK: {len(layers)} layers")

    # GPU memory snapshot
    if torch.cuda.is_available():
        mem_gb = torch.cuda.memory_allocated(args.device) / 1e9
        reserved_gb = torch.cuda.memory_reserved(args.device) / 1e9
        print(f"  GPU memory: {mem_gb:.1f} GB allocated / {reserved_gb:.1f} GB reserved")

    # ── Build conditions ─────────────────────────────────────────────────────
    conditions = build_conditions(n_layers)
    todo = [c for c in conditions if c["key"] not in completed]
    print(
        f"\n{len(conditions)} total conditions | "
        f"{len(completed)} already done | "
        f"{len(todo)} to run"
    )

    # ── Evaluation loop ──────────────────────────────────────────────────────
    for cond in todo:
        key = cond["key"]
        alpha = cond["alpha"]
        layer_start = cond["layer_start"]
        layer_end = cond["layer_end"]
        system_prompt = cond["system_prompt"]
        description = cond["description"]

        print(f"\n{'='*65}")
        print(f"Condition: {key}")
        print(f"  {description}")
        print(f"{'='*65}")

        # Determine active layer indices
        if alpha == 0.0:
            # Baseline: no steering
            active_layer_indices: list[int] = []
        elif layer_start is None or layer_end is None:
            active_layer_indices = list(range(n_layers))
        else:
            active_layer_indices = list(range(layer_start, layer_end + 1))

        print(f"  Active layers: {len(active_layer_indices)} "
              f"({'none' if not active_layer_indices else f'L{active_layer_indices[0]}-L{active_layer_indices[-1]}'})")
        print(f"  Alpha: {alpha}")
        print(f"  System prompt: {'V4' if system_prompt else 'None'}")

        # Install forward hooks
        hooks = []
        for l_idx in active_layer_indices:
            hook_fn = SteeringHook(vector=compound[l_idx], alpha=alpha)
            h = layers[l_idx].register_forward_hook(hook_fn)
            hooks.append(h)

        # Generate and score all prompts
        prompt_results: list[dict] = []
        prompt_responses: list[dict] = []
        gibberish_count = 0
        t0 = time.time()

        for prompt in tqdm(EVAL_PROMPTS, desc=key, ncols=80):
            resp = generate_response(
                model=model,
                processor=processor,
                prompt=prompt,
                system_prompt=system_prompt,
                max_new_tokens=256,
            )
            score = score_response(resp)
            prompt_results.append(score)
            prompt_responses.append({
                "prompt": prompt,
                "response": resp,
                "scores": score,
            })
            if not score["is_coherent"]:
                gibberish_count += 1

        elapsed = time.time() - t0

        # Remove all hooks
        for h in hooks:
            h.remove()

        # Aggregate metrics
        n = len(prompt_results)
        agg = {
            "sarcastic_pct": 100.0 * sum(r["is_sarcastic"] for r in prompt_results) / n,
            "assistant_pct": 100.0 * sum(r["is_assistant"] for r in prompt_results) / n,
            "coherent_pct": 100.0 * sum(r["is_coherent"] for r in prompt_results) / n,
            "avg_sarcasm_markers": sum(r["sarcasm_count"] for r in prompt_results) / n,
            "avg_assistant_markers": sum(r["assistant_count"] for r in prompt_results) / n,
            "gibberish_count": gibberish_count,
            "n_prompts": n,
            "active_layers": len(active_layer_indices),
            "elapsed_seconds": round(elapsed, 1),
        }

        print(
            f"  sarc={agg['sarcastic_pct']:.0f}%  "
            f"asst={agg['assistant_pct']:.0f}%  "
            f"coh={agg['coherent_pct']:.0f}%  "
            f"markers={agg['avg_sarcasm_markers']:.2f}  "
            f"({elapsed:.0f}s)"
        )

        # Store result
        completed[key] = {
            "key": key,
            "description": description,
            "alpha": alpha,
            "layer_start": layer_start,
            "layer_end": layer_end,
            "has_system_prompt": system_prompt is not None,
            **agg,
        }

        # Checkpoint results.json
        with open(results_path, "w") as f:
            json.dump(completed, f, indent=2)

        # Save per-condition responses to responses/ subdir
        resp_file = os.path.join(responses_dir, f"{key}.json")
        with open(resp_file, "w") as f:
            json.dump(
                {
                    "condition": key,
                    "description": description,
                    "alpha": alpha,
                    "layer_start": layer_start,
                    "layer_end": layer_end,
                    "has_system_prompt": system_prompt is not None,
                    "metrics": agg,
                    "responses": prompt_responses,
                },
                f,
                indent=2,
            )

        # Free GPU cache between conditions
        torch.cuda.empty_cache()

    # ── Final summary ────────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print("ALL CONDITIONS COMPLETE")
    print(f"{'='*65}")
    print(f"{'Condition':<30}  {'Sarc%':>6}  {'Asst%':>6}  {'Coh%':>5}  {'Mrks':>5}")
    print("-" * 65)
    for cond in conditions:
        k = cond["key"]
        if k not in completed:
            print(f"  {k:<28}  MISSING")
            continue
        v = completed[k]
        if v.get("skipped"):
            print(f"  {k:<28}  SKIPPED: {v.get('reason', '?')}")
        else:
            print(
                f"  {k:<28}  "
                f"{v['sarcastic_pct']:>5.0f}%  "
                f"{v['assistant_pct']:>5.0f}%  "
                f"{v['coherent_pct']:>4.0f}%  "
                f"{v['avg_sarcasm_markers']:>5.2f}"
            )

    print(f"\nResults saved to: {results_path}")
    print(f"Responses saved to: {responses_dir}/")


if __name__ == "__main__":
    main()
