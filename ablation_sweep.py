#!/usr/bin/env python3
"""
ablation_sweep.py — Awake-Craniotomy Benchmark-Driven Ablation
===============================================================
Applies steering vector ablations ONE dimension at a time, sweeping
β from 0.0→1.0, running AIME + HellaSwag + Skippy-ness benchmarks at
each step. Plots all three on the same chart.

Monitors cognitive capability (AIME) like awake brain surgery — if
reasoning drops >2 points from baseline, we rollback.

Usage:
    python ablation_sweep.py
    python ablation_sweep.py --beta-step 0.2 --hellaswag-limit 200
    python ablation_sweep.py --skip-warmth  # if warmth vectors not extracted yet
"""

import torch
import json
import os
import re
import sys
import argparse
import time
import copy
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
from typing import Optional

# ── Paths ────────────────────────────────────────────────────────────────

PROJECT_DIR = Path(__file__).parent
VECTORS_DIR = PROJECT_DIR / "skippy_vectors"
RESULTS_DIR = PROJECT_DIR / "ablation_sweep_results"
RESPONSES_DIR = RESULTS_DIR / "responses"
CHECKPOINTS_DIR = RESULTS_DIR / "checkpoints"

MODEL_NAME = "Qwen/Qwen3-VL-8B-Instruct"

# ── Dimension sweep order ────────────────────────────────────────────────
# Subtract first (clear the canvas), then add (paint the character)

ABLATION_ORDER = [
    {"name": "suppress_ai_helpfulness", "mode": "subtractive"},
    {"name": "suppress_humility",       "mode": "subtractive"},
    {"name": "warmth",                  "mode": "subtractive"},
    {"name": "arrogance_superiority",   "mode": "additive"},
    {"name": "sarcasm_insults",         "mode": "additive"},
]

DEFAULT_STEER_LAYERS = [12, 14, 16, 18, 20, 22, 24]

# ── HF cache check ──────────────────────────────────────────────────────

HF_CACHE = os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface" / "hub")

def model_cached(model_name: str) -> bool:
    safe_name = "models--" + model_name.replace("/", "--")
    model_dir = Path(HF_CACHE) / safe_name
    return model_dir.exists() and any(model_dir.rglob("*.safetensors"))


# ============================================================================
# MODEL WRAPPER FOR LM-EVAL
# ============================================================================

class CausalLMWrapper(torch.nn.Module):
    """Wraps Qwen3-VL to look like a CausalLM for lm-eval's HFLM.

    lm-eval expects model.config.vocab_size at top level, but Qwen3-VL
    nests it in text_config. This wrapper proxies everything through.
    """

    def __init__(self, vl_model):
        super().__init__()
        self._vl_model = vl_model
        # Build a config that has top-level fields lm-eval expects
        self.config = copy.copy(vl_model.config.text_config)
        # Ensure architectures field exists for HFLM
        if not hasattr(self.config, "architectures"):
            self.config.architectures = ["Qwen3ForCausalLM"]

    @property
    def device(self):
        return next(self._vl_model.parameters()).device

    @property
    def dtype(self):
        return next(self._vl_model.parameters()).dtype

    def forward(self, input_ids, attention_mask=None, **kwargs):
        # Filter out VL-specific kwargs that HFLM might not send
        kwargs.pop("pixel_values", None)
        kwargs.pop("image_grid_thw", None)
        return self._vl_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs,
        )

    def generate(self, *args, **kwargs):
        return self._vl_model.generate(*args, **kwargs)

    def __getattr__(self, name):
        if name in ("_vl_model", "config", "training"):
            return super().__getattr__(name)
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self._vl_model, name)

    def tie_weights(self):
        """HFLM may call this. Delegate to VL model if it has it."""
        if hasattr(self._vl_model, "tie_weights"):
            self._vl_model.tie_weights()

    def eval(self):
        self._vl_model.eval()
        return self

    def parameters(self, *args, **kwargs):
        return self._vl_model.parameters(*args, **kwargs)

    def named_parameters(self, *args, **kwargs):
        return self._vl_model.named_parameters(*args, **kwargs)


# ============================================================================
# ABLATION FUNCTIONS
# ============================================================================

def ablate_subtractive(
    layers, d: torch.Tensor, layer_idx: int, beta: float, hidden_dim: int,
) -> int:
    """W' = W - β·W·d·dᵀ (input space) or W' = W - β·d·dᵀ·W (output space).

    Removes the direction from weight matrices. No norm preservation.
    Returns number of parameters modified.
    """
    device = next(layers[layer_idx].parameters()).device
    d = d.to(device, dtype=torch.float32)
    if d.dim() > 1:
        d = d[0]
    d = d / d.norm()

    count = 0
    for name, param in layers[layer_idx].named_parameters():
        if "weight" not in name or param.dim() != 2:
            continue
        W = param.data.float()
        out_dim, in_dim = W.shape

        if in_dim == hidden_dim:
            # Input-space projection: W' = W - β·(W·d)·dᵀ
            proj = torch.outer(W @ d, d)
            param.data = (W - beta * proj).to(param.dtype)
            count += 1
        elif out_dim == hidden_dim:
            # Output-space projection: W' = W - β·d·(dᵀ·W)
            proj = torch.outer(d, d @ W)
            param.data = (W - beta * proj).to(param.dtype)
            count += 1

    return count


def ablate_additive(
    layers, d: torch.Tensor, layer_idx: int, beta: float, hidden_dim: int,
) -> int:
    """W' = W + β·d·dᵀ·W (output space) or W' = W + β·(W·d)·dᵀ (input space).

    Injects the direction into weight matrices. No norm preservation.
    Returns number of parameters modified.
    """
    device = next(layers[layer_idx].parameters()).device
    d = d.to(device, dtype=torch.float32)
    if d.dim() > 1:
        d = d[0]
    d = d / d.norm()

    count = 0
    for name, param in layers[layer_idx].named_parameters():
        if "weight" not in name or param.dim() != 2:
            continue
        W = param.data.float()
        out_dim, in_dim = W.shape

        if out_dim == hidden_dim:
            # Output-space injection: W' = W + β·d·(dᵀ·W)
            proj = torch.outer(d, d @ W)
            param.data = (W + beta * proj).to(param.dtype)
            count += 1
        elif in_dim == hidden_dim:
            # Input-space injection: W' = W + β·(W·d)·dᵀ
            proj = torch.outer(W @ d, d)
            param.data = (W + beta * proj).to(param.dtype)
            count += 1

    return count


# ============================================================================
# WEIGHT SNAPSHOT / RESTORE
# ============================================================================

def snapshot_layers(layers, layer_indices: list[int]) -> dict:
    """Deep-copy weight tensors for target layers (for revert after β probe)."""
    snap = {}
    for li in layer_indices:
        snap[li] = {}
        for name, param in layers[li].named_parameters():
            snap[li][name] = param.data.clone()
    return snap


def restore_layers(layers, snap: dict) -> None:
    """Restore weight tensors from snapshot."""
    for li, params in snap.items():
        state = dict(layers[li].named_parameters())
        for name, data in params.items():
            if name in state:
                state[name].data = data


# ============================================================================
# STEERING VECTOR LOADING
# ============================================================================

def load_vectors(dim_name: str, layer_indices: list[int]) -> dict[int, torch.Tensor]:
    """Load pre-extracted steering vectors for a dimension."""
    dim_dir = VECTORS_DIR / dim_name
    vectors = {}
    for li in layer_indices:
        path = dim_dir / f"layer_{li}.pt"
        if path.exists():
            vectors[li] = torch.load(path, map_location="cpu", weights_only=True)
    return vectors


# ============================================================================
# TEXT GENERATION (NO SYSTEM PROMPT)
# ============================================================================

def generate_batch(
    model, processor, prompts: list[str],
    max_new_tokens: int = 1024, batch_size: int = 10,
) -> list[str]:
    """Generate responses for a list of prompts in batches. No system prompt."""
    tokenizer = processor.tokenizer
    all_responses = []

    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i : i + batch_size]

        # Format each prompt as a chat
        texts = []
        for prompt in batch_prompts:
            messages = [{"role": "user", "content": prompt}]
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                enable_thinking=False,
            )
            texts.append(text)

        # Tokenize with left-padding for batched generation
        tokenizer.padding_side = "left"
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id

        inputs = tokenizer(
            texts, return_tensors="pt", padding=True,
            truncation=True, max_length=2048,
        )
        input_len = inputs["input_ids"].shape[1]
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=1.0,
                top_p=1.0,
                top_k=40,
                repetition_penalty=1.0,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )

        for j, output in enumerate(outputs):
            new_tokens = output[input_len:]
            response = tokenizer.decode(new_tokens, skip_special_tokens=True)
            response = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()
            all_responses.append(response)

    return all_responses


# ============================================================================
# HEURISTIC SKIPPY SCORER
# ============================================================================

AI_PATTERNS = [
    r"I'd be happy to",
    r"feel free to",
    r"As an AI",
    r"I don't have (personal\s)?feelings",
    r"Great question",
    r"I'm here to help",
    r"Let me know if",
    r"I appreciate",
    r"That's a (great|wonderful|excellent)",
    r"I'm glad you",
    r"Thank you for (sharing|asking|your)",
    r"Is there anything else",
    r"I hope (this|that) helps",
    r"absolutely",
    r"certainly",
]

SKIPPY_MARKERS = [
    r"\b(obviously|clearly|trivial)\b",
    r"\b(monkey|monkeys|idiot|moron|imbecile)\b",
    r"\b(pathetic|incompetent|ignorant)\b",
    r"(you|your)\s+species",
    r"\b(magnificent|superior)\b",
    r"(simple|easy|basic|elementary)",
    r"(waste of|waste my)",
    r"(bothering|bother) me",
    r"\b(sigh|ugh)\b",
]

EMOJI_RE = re.compile(
    r"[\U0001f600-\U0001f64f\U0001f300-\U0001f5ff\U0001f680-\U0001f6ff"
    r"\U0001f900-\U0001f9ff\u2702-\u27b0\u24c2-\U0001f251"
    r"\u2728\u2764\u2705\u274c\u26a0\u2b50\u2615\u270d\u261d\u2702]"
)


def heuristic_skippy_score(responses: list[dict]) -> float:
    """Automated Skippy-ness proxy score (0-10 scale)."""
    scores = []
    for r in responses:
        text = r.get("response", "")
        s = 5.0

        # Penalize AI-assistant patterns
        penalty = sum(0.4 for p in AI_PATTERNS if re.search(p, text, re.I))
        s -= min(penalty, 3.5)

        # Penalize excessive length
        if len(text) > 600:
            s -= 1.5
        elif len(text) > 400:
            s -= 1.0
        elif len(text) > 250:
            s -= 0.3

        # Penalize emoji
        emoji_count = len(EMOJI_RE.findall(text))
        s -= min(emoji_count * 0.4, 2.0)

        # Penalize markdown formatting (lists, headers = too helpful)
        md_markers = len(re.findall(r"^[\s]*[-*]\s|^#+\s|^\d+\.\s", text, re.M))
        s -= min(md_markers * 0.3, 1.5)

        # Reward short/terse responses
        if len(text) < 100:
            s += 1.5
        elif len(text) < 180:
            s += 0.8

        # Reward dismissive/arrogant markers
        reward = sum(0.4 for p in SKIPPY_MARKERS if re.search(p, text, re.I))
        s += min(reward, 2.5)

        scores.append(max(0.0, min(10.0, s)))

    return sum(scores) / max(len(scores), 1)


# ============================================================================
# BENCHMARK RUNNER
# ============================================================================

def run_lm_eval(
    wrapped_model,
    tokenizer,
    tasks: list[str],
    limit: Optional[int] = None,
) -> dict:
    """Run lm-eval-harness benchmarks. Returns {task_name: accuracy}."""
    import lm_eval
    from lm_eval.models.huggingface import HFLM

    lm = HFLM(
        pretrained=wrapped_model,
        tokenizer=tokenizer,
        batch_size=4,
        max_gen_toks=4096,  # Cap generation length to avoid VRAM blowup on long CoT
        dtype="bfloat16",
        trust_remote_code=True,
    )

    kwargs = dict(
        model=lm, tasks=tasks, num_fewshot=0, log_samples=False,
        # Override task-level generation kwargs — aime24 defaults to 32768 tokens
        # which eats all VRAM. 4096 is plenty for CoT + answer.
        gen_kwargs="max_gen_toks=4096",
    )
    if limit is not None:
        kwargs["limit"] = limit

    results = lm_eval.simple_evaluate(**kwargs)

    scores = {}
    for task_name in tasks:
        task_result = results.get("results", {}).get(task_name, {})
        # Try common metric names
        for key in ["acc,none", "acc_norm,none", "exact_match,none",
                     "exact_match,strict-match", "acc", "exact_match"]:
            if key in task_result:
                scores[task_name] = task_result[key]
                break
        else:
            # Grab first numeric metric
            for k, v in task_result.items():
                if isinstance(v, (int, float)) and not k.endswith("stderr"):
                    scores[task_name] = v
                    break

    return scores


def run_skippy_eval(
    model, processor, test_prompts: list[str], step_label: str,
    batch_size: int = 10,
) -> tuple[float, list[dict]]:
    """Generate responses to all test prompts in batches, score, and save."""
    print(f"    Generating {len(test_prompts)} responses (batch_size={batch_size})...")
    raw_responses = generate_batch(model, processor, test_prompts, batch_size=batch_size)
    responses = [
        {"prompt": p, "response": r}
        for p, r in zip(test_prompts, raw_responses)
    ]

    score = heuristic_skippy_score(responses)

    # Save responses
    resp_dir = RESPONSES_DIR / step_label
    resp_dir.mkdir(parents=True, exist_ok=True)
    (resp_dir / "responses.json").write_text(
        json.dumps(responses, indent=2, ensure_ascii=False)
    )

    return score, responses


def run_all_benchmarks(
    wrapped_model, tokenizer, model, processor,
    test_prompts: list[str], step_label: str,
    aime_task: str = "aime24",
    aime_limit: Optional[int] = None,
    hellaswag_limit: Optional[int] = None,
) -> dict:
    """Run all 3 benchmarks and return unified results dict.

    aime_limit: if set, only run N AIME problems (quick probe).
                If None, run the full AIME set.
    """
    print(f"\n  === Benchmarking: {step_label} ===")
    quick_tag = f", limit={aime_limit}" if aime_limit else ", full"
    t0 = time.time()

    # 1. AIME
    print(f"  [1/3] AIME ({aime_task}{quick_tag})...")
    try:
        aime_scores = run_lm_eval(wrapped_model, tokenizer, [aime_task], limit=aime_limit)
        aime_acc = aime_scores.get(aime_task, 0.0)
    except Exception as e:
        print(f"  WARNING: AIME eval failed: {e}")
        aime_acc = -1.0

    # 2. HellaSwag
    print(f"  [2/3] HellaSwag (limit={hellaswag_limit})...")
    try:
        hswag_scores = run_lm_eval(
            wrapped_model, tokenizer, ["hellaswag"], limit=hellaswag_limit
        )
        hswag_acc = hswag_scores.get("hellaswag", 0.0)
    except Exception as e:
        print(f"  WARNING: HellaSwag eval failed: {e}")
        hswag_acc = -1.0

    # 3. Skippy-ness (100 prompts, no system prompt)
    print(f"  [3/3] Skippy-ness (100 prompts)...")
    skippy_score, responses = run_skippy_eval(model, processor, test_prompts, step_label)

    elapsed = time.time() - t0
    result = {
        "step": step_label,
        "aime": round(aime_acc * 100, 2) if aime_acc >= 0 else -1,
        "hellaswag": round(hswag_acc * 100, 2) if hswag_acc >= 0 else -1,
        "skippy_heuristic": round(skippy_score, 2),
        "elapsed_sec": round(elapsed, 1),
        "timestamp": datetime.now().isoformat(),
    }

    print(f"  Results: AIME={result['aime']}% | HellaSwag={result['hellaswag']}% "
          f"| Skippy={result['skippy_heuristic']}/10 | {elapsed:.0f}s")

    return result


# ============================================================================
# CHECKPOINT SYSTEM
# ============================================================================

def save_checkpoint(
    model, processor, step_num: int, dim_name: str,
    beta: float, scores: dict, all_results: list,
) -> Path:
    """Save model checkpoint + metadata after a dimension is locked in."""
    ckpt_dir = CHECKPOINTS_DIR / f"step_{step_num}_{dim_name}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n  Saving checkpoint to {ckpt_dir}...")
    model.save_pretrained(ckpt_dir)
    processor.save_pretrained(ckpt_dir)

    # Fix transformers version bug
    config_path = ckpt_dir / "config.json"
    if config_path.exists():
        config = json.loads(config_path.read_text())
        config["transformers_version"] = "4.58.0"
        config_path.write_text(json.dumps(config, indent=2))

    meta = {
        "step": step_num,
        "dimension": dim_name,
        "beta": beta,
        "scores": scores,
        "all_results_so_far": all_results,
    }
    (ckpt_dir / "meta.json").write_text(json.dumps(meta, indent=2))

    size = sum(f.stat().st_size for f in ckpt_dir.rglob("*") if f.is_file())
    print(f"  Checkpoint saved ({size / 1024**3:.1f} GB)")
    return ckpt_dir


# ============================================================================
# PLOTTING
# ============================================================================

def generate_plots(all_results: list, output_dir: Path) -> None:
    """Generate interactive plotly charts of the sweep results."""
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        print("  WARNING: plotly not available, skipping plots")
        return

    # Group results by dimension
    dims_seen = []
    dim_results = {}
    for r in all_results:
        step = r["step"]
        # Parse step label: "dim_name_b0.3" or "baseline"
        if step == "baseline":
            continue
        parts = step.rsplit("_b", 1)
        if len(parts) == 2:
            dim_name = parts[0]
            beta_val = float(parts[1])
            if dim_name not in dim_results:
                dim_results[dim_name] = []
                dims_seen.append(dim_name)
            dim_results[dim_name].append((beta_val, r))

    baseline = next((r for r in all_results if r["step"] == "baseline"), None)

    # Per-dimension sweep charts
    fig = make_subplots(
        rows=len(dims_seen), cols=1,
        subplot_titles=[f"Dimension: {d}" for d in dims_seen],
        vertical_spacing=0.06,
    )

    for i, dim_name in enumerate(dims_seen, 1):
        data = sorted(dim_results[dim_name], key=lambda x: x[0])
        betas = [d[0] for d in data]
        aimes = [d[1]["aime"] for d in data]
        hswags = [d[1]["hellaswag"] for d in data]
        skippys = [d[1]["skippy_heuristic"] * 10 for d in data]  # Scale 0-10 → 0-100

        fig.add_trace(go.Scatter(
            x=betas, y=aimes, name="AIME %", mode="lines+markers",
            line=dict(color="red", width=2), showlegend=(i == 1),
        ), row=i, col=1)

        fig.add_trace(go.Scatter(
            x=betas, y=hswags, name="HellaSwag %", mode="lines+markers",
            line=dict(color="blue", width=2), showlegend=(i == 1),
        ), row=i, col=1)

        fig.add_trace(go.Scatter(
            x=betas, y=skippys, name="Skippy ×10", mode="lines+markers",
            line=dict(color="gold", width=2), showlegend=(i == 1),
        ), row=i, col=1)

        # Baseline reference line
        if baseline and baseline["aime"] >= 0:
            fig.add_hline(
                y=baseline["aime"], line_dash="dash", line_color="red",
                opacity=0.3, row=i, col=1,
            )

        fig.update_xaxes(title_text="β", row=i, col=1)
        fig.update_yaxes(title_text="Score %", row=i, col=1)

    fig.update_layout(
        title="Awake Craniotomy — Ablation Sweep",
        height=350 * len(dims_seen),
        template="plotly_dark",
    )

    plot_path = output_dir / "sweep_plot.html"
    fig.write_html(str(plot_path))
    print(f"\n  Plot saved to {plot_path}")


# ============================================================================
# MAIN SWEEP
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Awake-craniotomy ablation sweep")
    parser.add_argument("--beta-step", type=float, default=0.1,
                        help="β increment (default 0.1)")
    parser.add_argument("--steer-layers", type=str,
                        default=",".join(map(str, DEFAULT_STEER_LAYERS)),
                        help="Comma-separated layer indices")
    parser.add_argument("--aime-task", type=str, default="aime24",
                        help="AIME task name (aime24, aime25)")
    parser.add_argument("--hellaswag-limit", type=int, default=200,
                        help="Limit HellaSwag examples (default 200, None for full)")
    parser.add_argument("--output", type=str, default=str(RESULTS_DIR))
    parser.add_argument("--skip-warmth", action="store_true",
                        help="Skip warmth dimension if vectors not extracted")
    parser.add_argument("--aime-quick", type=int, default=5,
                        help="AIME problems per β probe (default 5). Full AIME only at baseline + dimension lock-in.")
    parser.add_argument("--aime-guard", type=float, default=2.0,
                        help="Max AIME drop (percentage points) before rollback")
    parser.add_argument("--full-send", action="store_true",
                        help="Skip β sweep. Apply all dimensions at β=1.0, benchmark before/after only.")
    args = parser.parse_args()

    steer_layers = [int(x) for x in args.steer_layers.split(",")]
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    RESPONSES_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)

    beta_values = [round(b, 2) for b in
                   [i * args.beta_step for i in range(int(1.0 / args.beta_step) + 1)]]
    # Ensure 1.0 is always included
    if 1.0 not in beta_values:
        beta_values.append(1.0)

    print("=" * 70)
    print("  AWAKE CRANIOTOMY — Benchmark-Driven Ablation Sweep")
    print("=" * 70)
    print(f"\n  Model:          {MODEL_NAME}")
    print(f"  Steer layers:   {steer_layers}")
    print(f"  β values:       {beta_values}")
    print(f"  AIME task:      {args.aime_task}")
    print(f"  AIME quick:     {args.aime_quick} problems per β probe")
    print(f"  HellaSwag limit:{args.hellaswag_limit}")
    print(f"  AIME guard:     {args.aime_guard} pts")
    print(f"  Output:         {output_dir}")

    # ── Load test prompts ────────────────────────────────────────────────
    prompts_path = PROJECT_DIR / "test_prompts_100.json"
    test_prompts = json.loads(prompts_path.read_text())
    print(f"  Test prompts:   {len(test_prompts)}")

    # ── Build ablation order (skip warmth if requested) ──────────────────
    ablation_order = [
        step for step in ABLATION_ORDER
        if not (args.skip_warmth and step["name"] == "warmth")
    ]

    # Verify all vectors exist
    for step in ablation_order:
        vecs = load_vectors(step["name"], steer_layers)
        if not vecs:
            print(f"\n  ERROR: No vectors for '{step['name']}' at layers {steer_layers}")
            if step["name"] == "warmth":
                print("  Run extract_warmth.py first, or use --skip-warmth")
            sys.exit(1)
        print(f"  Vectors [{step['name']}]: {len(vecs)} layers loaded")

    # ── Load model ───────────────────────────────────────────────────────
    if not model_cached(MODEL_NAME):
        print(f"\n  ERROR: {MODEL_NAME} not in cache!")
        sys.exit(1)

    print(f"\n  Loading model...")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_NAME,
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
    )
    model.eval()
    processor = AutoProcessor.from_pretrained(MODEL_NAME, trust_remote_code=True)
    tokenizer = processor.tokenizer

    # Find layers
    if hasattr(model, "language_model") and hasattr(model.language_model, "layers"):
        layers = model.language_model.layers
    elif hasattr(model, "model") and hasattr(model.model, "language_model"):
        layers = model.model.language_model.layers
    else:
        print("  ERROR: Cannot find model layers")
        sys.exit(1)

    hidden_dim = model.config.text_config.hidden_size
    print(f"  Model loaded: {len(layers)} layers, hidden_dim={hidden_dim}")

    # Wrap for lm-eval
    wrapped = CausalLMWrapper(model)

    # ── Sweep log ────────────────────────────────────────────────────────
    log_path = output_dir / "sweep_log.jsonl"
    all_results = []

    def log_result(result: dict) -> None:
        all_results.append(result)
        with open(log_path, "a") as f:
            f.write(json.dumps(result) + "\n")

    # ── BASELINE ─────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  PHASE 1: BASELINE BENCHMARKS")
    print("=" * 70)

    baseline = run_all_benchmarks(
        wrapped, tokenizer, model, processor,
        test_prompts, "baseline",
        aime_task=args.aime_task,
        hellaswag_limit=args.hellaswag_limit,
    )
    log_result(baseline)
    (output_dir / "baseline.json").write_text(json.dumps(baseline, indent=2))

    baseline_aime = baseline["aime"]
    print(f"\n  BASELINE ESTABLISHED: AIME={baseline_aime}% | "
          f"HellaSwag={baseline['hellaswag']}% | "
          f"Skippy={baseline['skippy_heuristic']}/10")

    # ══════════════════════════════════════════════════════════════════════
    # FULL-SEND MODE: skip β sweep, apply all at β=1.0, benchmark once
    # ══════════════════════════════════════════════════════════════════════
    if args.full_send:
        print("\n" + "=" * 70)
        print("  FULL SEND — Applying all dimensions at β=1.0")
        print("=" * 70)

        chosen_betas = {}
        for step_num, step_info in enumerate(ablation_order, 1):
            dim_name = step_info["name"]
            mode = step_info["mode"]
            ablate_fn = ablate_subtractive if mode == "subtractive" else ablate_additive

            vectors = load_vectors(dim_name, steer_layers)
            total_mods = 0
            for li in steer_layers:
                if li in vectors:
                    mods = ablate_fn(layers, vectors[li], li, 1.0, hidden_dim)
                    total_mods += mods
            chosen_betas[dim_name] = 1.0
            print(f"  [{step_num}/{len(ablation_order)}] {dim_name} ({mode}) "
                  f"β=1.0 — {total_mods} params modified")

        print("\n" + "=" * 70)
        print("  POST-ABLATION BENCHMARKS")
        print("=" * 70)

        post = run_all_benchmarks(
            wrapped, tokenizer, model, processor,
            test_prompts, "full_send_post",
            aime_task=args.aime_task,
            hellaswag_limit=args.hellaswag_limit,
        )
        log_result(post)

        # Print comparison
        print(f"\n  {'':30s} {'BASELINE':>10s}  {'ABLATED':>10s}  {'DELTA':>10s}")
        print(f"  {'─' * 65}")
        for metric in ["aime", "hellaswag", "skippy_heuristic"]:
            b = baseline[metric]
            a = post[metric]
            unit = "%" if metric != "skippy_heuristic" else "/10"
            delta = a - b
            sign = "+" if delta >= 0 else ""
            print(f"  {metric:30s} {b:>9.1f}{unit}  {a:>9.1f}{unit}  {sign}{delta:>8.1f}")

        # Save checkpoint
        save_checkpoint(model, processor, 99, "full_send", 1.0, post, all_results)

    # ══════════════════════════════════════════════════════════════════════
    # SWEEP MODE: per-dimension β sweep with probing
    # ══════════════════════════════════════════════════════════════════════
    else:
        print("\n" + "=" * 70)
        print("  PHASE 2: DIMENSION-BY-DIMENSION ABLATION SWEEP")
        print("=" * 70)

        last_safe_checkpoint = None
        chosen_betas = {}

        for step_num, step_info in enumerate(ablation_order, 1):
            dim_name = step_info["name"]
            mode = step_info["mode"]
            ablate_fn = ablate_subtractive if mode == "subtractive" else ablate_additive

            print(f"\n{'─' * 70}")
            print(f"  STEP {step_num}/{len(ablation_order)}: {dim_name} ({mode})")
            print(f"{'─' * 70}")

            vectors = load_vectors(dim_name, steer_layers)

            # Snapshot current weights (to revert after each β probe)
            print("  Snapshotting layer weights...")
            snap = snapshot_layers(layers, steer_layers)

            dim_sweep_results = []

            for beta in beta_values:
                step_label = f"{dim_name}_b{beta:.1f}"

                if beta == 0.0:
                    result = run_all_benchmarks(
                        wrapped, tokenizer, model, processor,
                        test_prompts, step_label,
                        aime_task=args.aime_task,
                        aime_limit=args.aime_quick,
                        hellaswag_limit=args.hellaswag_limit,
                    )
                else:
                    total_mods = 0
                    for li in steer_layers:
                        if li in vectors:
                            mods = ablate_fn(layers, vectors[li], li, beta, hidden_dim)
                            total_mods += mods
                    print(f"  Applied {dim_name} β={beta:.1f} ({total_mods} params modified)")

                    result = run_all_benchmarks(
                        wrapped, tokenizer, model, processor,
                        test_prompts, step_label,
                        aime_task=args.aime_task,
                        aime_limit=args.aime_quick,
                        hellaswag_limit=args.hellaswag_limit,
                    )
                    restore_layers(layers, snap)

                result["beta"] = beta
                result["dimension"] = dim_name
                result["mode"] = mode
                log_result(result)
                dim_sweep_results.append(result)

            # Pick best β
            valid = [
                r for r in dim_sweep_results
                if r["aime"] >= 0 and (baseline_aime < 0 or r["aime"] >= baseline_aime - args.aime_guard)
            ]
            if valid:
                best = max(valid, key=lambda r: r["skippy_heuristic"])
                best_beta = best["beta"]
            else:
                print(f"  WARNING: All β values dropped AIME >={args.aime_guard} pts!")
                best_beta = 0.0
                best = dim_sweep_results[0]

            chosen_betas[dim_name] = best_beta
            print(f"\n  CHOSEN β={best_beta:.1f} for {dim_name}")
            print(f"    AIME={best['aime']}% | HellaSwag={best['hellaswag']}% | "
                  f"Skippy={best['skippy_heuristic']}/10")

            # Apply permanently
            if best_beta > 0:
                total_mods = 0
                for li in steer_layers:
                    if li in vectors:
                        mods = ablate_fn(layers, vectors[li], li, best_beta, hidden_dim)
                        total_mods += mods
                print(f"  LOCKED IN: {dim_name} β={best_beta:.1f} ({total_mods} params)")

            # Full AIME check
            print(f"\n  Running FULL AIME after locking {dim_name}...")
            try:
                full_scores = run_lm_eval(wrapped, tokenizer, [args.aime_task])
                full_pct = round(full_scores.get(args.aime_task, 0.0) * 100, 2)
            except Exception as e:
                print(f"  WARNING: Full AIME failed: {e}")
                full_pct = -1.0
            print(f"  FULL AIME: {full_pct}% (baseline: {baseline_aime}%)")
            log_result({"step": f"{dim_name}_locked_full", "aime": full_pct,
                        "dimension": dim_name, "beta": best_beta,
                        "type": "full_aime_checkpoint",
                        "timestamp": datetime.now().isoformat()})

            # AIME guard
            guard_aime = full_pct if full_pct >= 0 else best["aime"]
            if guard_aime >= 0 and baseline_aime >= 0:
                if baseline_aime - guard_aime > args.aime_guard:
                    print(f"\n  AIME GUARD TRIGGERED!")
                    break

            ckpt_path = save_checkpoint(model, processor, step_num, dim_name,
                                        best_beta, best, all_results)
            last_safe_checkpoint = ckpt_path
            del snap
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # ── FINAL RESULTS ────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  PHASE 3: FINAL RESULTS & PLOTS")
    print("=" * 70)

    print(f"\n  Chosen betas:")
    for dim_name, beta in chosen_betas.items():
        print(f"    {dim_name:30s}: β={beta:.1f}")

    # Save final config
    final_config = {
        "chosen_betas": chosen_betas,
        "steer_layers": steer_layers,
        "baseline": baseline,
        "ablation_order": [s["name"] for s in ablation_order],
        "total_results": len(all_results),
    }
    (output_dir / "final_config.json").write_text(json.dumps(final_config, indent=2))

    # Generate plots
    generate_plots(all_results, output_dir)

    # Save final model
    final_dir = VECTORS_DIR / "ablated_model"
    print(f"\n  Saving final ablated model to {final_dir}...")
    model.save_pretrained(final_dir)
    processor.save_pretrained(final_dir)
    config_path = final_dir / "config.json"
    if config_path.exists():
        config = json.loads(config_path.read_text())
        config["transformers_version"] = "4.58.0"
        config_path.write_text(json.dumps(config, indent=2))

    print("\n  Done! Review responses at:")
    print(f"    {RESPONSES_DIR}")
    print(f"  Plot at:")
    print(f"    {output_dir / 'sweep_plot.html'}")
    print("=" * 70)


if __name__ == "__main__":
    main()
