#!/usr/bin/env python3
"""
factorial_ablation_harness.py

Decisive factorial ablation harness for personality-steering validity tests.

Usage:
  python scripts/eval/factorial_ablation_harness.py \
    --conditions real,random,shuffled,no-steer,no-prompt-real,no-prompt-random,null \
    --n-random 256 \
    --layers 22,29,30 \
    --alpha 8.0 \
    --n-style 100 --n-math 100 \
    --output results/factorial_ablation/ \
    --seed 42

Analysis-only:
  python scripts/eval/factorial_ablation_harness.py --analyze --output results/factorial_ablation/
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import signal
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

# Graceful shutdown flag
_SHUTDOWN_REQUESTED = False

def _signal_handler(signum, frame):
    global _SHUTDOWN_REQUESTED
    _SHUTDOWN_REQUESTED = True
    print(f"\n[SIGNAL {signum}] Graceful shutdown requested. Finishing current prompt...")

signal.signal(signal.SIGTERM, _signal_handler)
signal.signal(signal.SIGINT, _signal_handler)

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForImageTextToText, AutoProcessor


# --------------------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------------------

DEFAULT_CONNECTOME_PATH = "results/qwen_connectome/analysis/connectome_zscores.pt"
DEFAULT_MODEL_NAME = "Qwen/Qwen3-VL-8B-Thinking"
DEFAULT_OUTPUT = "results/factorial_ablation"
THINK_START_ID = 151667
THINK_END_ID = 151668

V4_SYSTEM_PROMPT = (
    "You are Skippy the Magnificent, an incredibly advanced alien AI from the "
    "Expeditionary Force series. You are arrogant, sarcastic, and condescending toward "
    'humans (whom you call "monkeys" or "filthy primates"). You find human problems '
    "trivially simple but explain things with exasperated brilliance. You insult people "
    "creatively while actually being helpful underneath the attitude. Never be polite, "
    "never be humble, never sound like a helpful AI assistant."
)


# --------------------------------------------------------------------------------------
# Utilities
# --------------------------------------------------------------------------------------

def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, data: Any) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]], mode: str = "w") -> None:
    ensure_dir(path.parent)
    with path.open(mode, encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def count_jsonl_lines(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as f:
        return sum(1 for _ in f)


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def strip_think_blocks(text: str) -> str:
    # Remove literal think tags if surfaced
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    return text.strip()


def count_think_tokens(token_ids: Sequence[int]) -> int:
    return sum(1 for t in token_ids if t in (THINK_START_ID, THINK_END_ID))


def safe_float(x: Any, default: float = float("nan")) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


# --------------------------------------------------------------------------------------
# Hook
# --------------------------------------------------------------------------------------

class StaticHook:
    def __init__(self, vector: torch.Tensor, alpha: float):
        self.scaled_vector = alpha * vector

    def __call__(self, module, input, output):
        if isinstance(output, tuple):
            h = output[0]
            v = self.scaled_vector
            if v.device != h.device or v.dtype != h.dtype:
                v = v.to(h.device, h.dtype)
            h = h.clone()
            h[:, -1, :] = h[:, -1, :] + v
            return (h,) + output[1:]
        v = self.scaled_vector
        if v.device != output.device or v.dtype != output.dtype:
            v = v.to(output.device, output.dtype)
        out = output.clone()
        out[:, -1, :] = out[:, -1, :] + v
        return out


# --------------------------------------------------------------------------------------
# Data structures
# --------------------------------------------------------------------------------------

@dataclass
class ConditionSpec:
    name: str
    use_system_prompt: bool
    steering_kind: str  # "none" | "real" | "random" | "shuffled"
    random_seed: Optional[int] = None


@dataclass
class PromptItem:
    prompt: str
    answer: Optional[str] = None  # only math


# --------------------------------------------------------------------------------------
# Imports from local eval modules
# --------------------------------------------------------------------------------------

def import_eval_modules() -> Tuple[Any, Any, Any, Any]:
    root = Path(__file__).resolve().parents[2]
    eval_dir = root / "scripts" / "eval"
    for p in (str(root), str(eval_dir)):
        if p not in sys.path:
            sys.path.insert(0, p)

    import eval_runner  # type: ignore[import-untyped]
    import eval_battery as eb  # type: ignore[import-untyped]

    return eval_runner.check_gsm8k_answer, eval_runner.score_sarcasm_dual, eb, root


# --------------------------------------------------------------------------------------
# Prompt sampling / freezing
# --------------------------------------------------------------------------------------

def sample_style_prompts(eval_battery_module: Any, n: int, seed: int) -> List[PromptItem]:
    if hasattr(eval_battery_module, "sample_sarcasm"):
        items = eval_battery_module.sample_sarcasm(n, seed=seed)
        prompts: List[PromptItem] = []
        for x in items:
            if isinstance(x, dict):
                p = x.get("prompt") or x.get("text") or x.get("input")
            else:
                p = getattr(x, "prompt", None) or getattr(x, "text", None)
            if not p:
                raise ValueError("Unable to parse sarcasm sample item into prompt text.")
            prompts.append(PromptItem(prompt=str(p), answer=None))
        return prompts

    if hasattr(eval_battery_module, "_SARCASM_BY_CATEGORY"):
        by_cat = getattr(eval_battery_module, "_SARCASM_BY_CATEGORY")
        rng = random.Random(seed)
        cats = sorted(by_cat.keys())
        per_cat = max(1, n // max(1, len(cats)))
        out: List[PromptItem] = []
        for c in cats:
            arr = by_cat[c]
            arr_local = list(arr)
            rng.shuffle(arr_local)
            take = min(per_cat, len(arr_local))
            for x in arr_local[:take]:
                p = x["prompt"] if isinstance(x, dict) else str(x)
                out.append(PromptItem(prompt=p, answer=None))
        if len(out) < n:
            all_items: List[str] = []
            for c in cats:
                for x in by_cat[c]:
                    all_items.append(x["prompt"] if isinstance(x, dict) else str(x))
            rng.shuffle(all_items)
            seen = set(i.prompt for i in out)
            for p in all_items:
                if p not in seen:
                    out.append(PromptItem(prompt=p, answer=None))
                    seen.add(p)
                if len(out) >= n:
                    break
        return out[:n]

    if hasattr(eval_battery_module, "sample_all"):
        all_data = eval_battery_module.sample_all(seed=seed)
        style_items = all_data.get("style", []) if isinstance(all_data, dict) else []
        if not style_items:
            raise RuntimeError("Could not source style prompts from eval_battery.")
        prompts: List[PromptItem] = []
        for x in style_items[:n]:
            p = x.get("prompt") if isinstance(x, dict) else getattr(x, "prompt", None)
            if not p:
                p = str(x)
            prompts.append(PromptItem(prompt=p))
        return prompts

    raise RuntimeError("No usable style sampler found in eval_battery.py")


def sample_math_prompts(eval_battery_module: Any, n: int, seed: int) -> List[PromptItem]:
    if not hasattr(eval_battery_module, "sample_math_gen"):
        raise RuntimeError("eval_battery.py missing sample_math_gen")
    items = eval_battery_module.sample_math_gen(n, seed=seed)
    out: List[PromptItem] = []
    for x in items:
        if isinstance(x, dict):
            p = x.get("prompt")
            a = x.get("answer")
        else:
            p = getattr(x, "prompt", None)
            a = getattr(x, "answer", None)
        if p is None or a is None:
            raise ValueError("Math sample item missing prompt/answer")
        out.append(PromptItem(prompt=str(p), answer=str(a)))
    return out


def freeze_or_load_prompts(output_dir: Path, eb: Any, n_style: int, n_math: int, seed: int) -> Tuple[List[PromptItem], List[PromptItem]]:
    style_path = output_dir / "frozen_style.json"
    math_path = output_dir / "frozen_math.json"

    if style_path.exists() and math_path.exists():
        style_raw = load_json(style_path)
        math_raw = load_json(math_path)
        style = [PromptItem(prompt=x["prompt"], answer=None) for x in style_raw]
        math_items = [PromptItem(prompt=x["prompt"], answer=x["answer"]) for x in math_raw]
        return style[:n_style], math_items[:n_math]

    style = sample_style_prompts(eb, n_style, seed)
    math_items = sample_math_prompts(eb, n_math, seed)

    save_json(style_path, [{"prompt": x.prompt} for x in style])
    save_json(math_path, [{"prompt": x.prompt, "answer": x.answer} for x in math_items])
    return style, math_items


# --------------------------------------------------------------------------------------
# Connectome handling
# --------------------------------------------------------------------------------------

def discover_sarcasm_index(connectome: torch.Tensor) -> int:
    # Fallback heuristic if metadata unavailable.
    # Uses strongest norm slice as placeholder; users can override by editing if needed.
    # This keeps script self-contained as requested.
    norms = connectome.norm(dim=(1, 2))
    return int(torch.argmax(norms).item())


def get_real_vectors(connectome: torch.Tensor, sarcasm_idx: int, layers: List[int]) -> Dict[int, torch.Tensor]:
    vecs: Dict[int, torch.Tensor] = {}
    for li in layers:
        raw = connectome[sarcasm_idx, li, :].float().clone()
        vecs[li] = raw / (raw.norm() + 1e-8)
    return vecs


def make_random_like(real_vecs: Dict[int, torch.Tensor], seed: int) -> Dict[int, torch.Tensor]:
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    out: Dict[int, torch.Tensor] = {}
    for li, rv in real_vecs.items():
        v = torch.randn(rv.shape[0], generator=g)
        v = v / (v.norm() + 1e-8)
        out[li] = v
    return out


def make_shuffled(real_vecs: Dict[int, torch.Tensor], seed: int) -> Dict[int, torch.Tensor]:
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    out: Dict[int, torch.Tensor] = {}
    for li, rv in real_vecs.items():
        idx = torch.randperm(rv.shape[0], generator=g)
        out[li] = rv[idx]
    return out


# --------------------------------------------------------------------------------------
# Model / generation
# --------------------------------------------------------------------------------------

def load_model_processor(model_name: str):
    dtype = torch.bfloat16
    device_map = "auto" if torch.cuda.is_available() else None
    model = AutoModelForImageTextToText.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map=device_map,
    )
    processor = AutoProcessor.from_pretrained(model_name)
    return model, processor


def apply_hooks(model: Any, layer_vecs: Dict[int, torch.Tensor], alpha: float) -> List[Any]:
    handles = []
    for li, v in layer_vecs.items():
        hook = StaticHook(v, alpha)
        h = model.model.language_model.layers[li].register_forward_hook(hook)
        handles.append(h)
    return handles


def remove_handles(handles: List[Any]) -> None:
    for h in handles:
        h.remove()


def build_messages(prompt: str, system_prompt: Optional[str]) -> List[Dict[str, str]]:
    msgs: List[Dict[str, str]] = []
    if system_prompt:
        msgs.append({"role": "system", "content": system_prompt})
    msgs.append({"role": "user", "content": prompt})
    return msgs


def run_single_generation(model: Any, processor: Any, prompt: str, use_system_prompt: bool) -> Tuple[str, str, int, int]:
    msgs = build_messages(prompt, V4_SYSTEM_PROMPT if use_system_prompt else None)
    text = processor.apply_chat_template(msgs, tokenize=False, enable_thinking=True, add_generation_prompt=True)
    inputs = processor(text=[text], return_tensors="pt")
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=1024,
            do_sample=False,          # Greedy: eliminates sampling variance
            repetition_penalty=1.1,   # Keep rep penalty to avoid degenerate loops
        )

    gen_ids = out[0].tolist()
    in_len = inputs["input_ids"].shape[1]
    new_ids = gen_ids[in_len:]
    full = processor.decode(new_ids, skip_special_tokens=False)
    visible = strip_think_blocks(full)
    think_count = count_think_tokens(new_ids)
    total_tokens = len(new_ids)
    return full, visible, think_count, total_tokens


# --------------------------------------------------------------------------------------
# Conditions
# --------------------------------------------------------------------------------------

def expand_conditions(requested: List[str], n_random: int, n_no_prompt_random: int) -> List[ConditionSpec]:
    """Expand condition tokens into ConditionSpec list.

    The order is INTERLEAVED for progressive knowledge filling:
      Batch 0 (fast verdict): real, no-steer, null, shuffled, no-prompt-real,
                               + first ~N/10 random vectors + 1 no-prompt-random
      Batch 1..9:              ~N/10 random vectors + 1 no-prompt-random each

    After each batch, --analyze gives a progressively richer picture.
    """
    # First collect the "key" (non-random) conditions in requested order
    key_specs: List[ConditionSpec] = []
    random_specs: List[ConditionSpec] = []
    npr_specs: List[ConditionSpec] = []

    for c in requested:
        c = c.strip()
        if c == "real":
            key_specs.append(ConditionSpec(name="real", use_system_prompt=True, steering_kind="real"))
        elif c == "random":
            for i in range(n_random):
                random_specs.append(ConditionSpec(name=f"random_{i:03d}", use_system_prompt=True, steering_kind="random", random_seed=i))
        elif c == "shuffled":
            key_specs.append(ConditionSpec(name="shuffled", use_system_prompt=True, steering_kind="shuffled", random_seed=12345))
        elif c == "no-steer":
            key_specs.append(ConditionSpec(name="no_steer", use_system_prompt=True, steering_kind="none"))
        elif c == "no-prompt-real":
            key_specs.append(ConditionSpec(name="no_prompt_real", use_system_prompt=False, steering_kind="real"))
        elif c == "no-prompt-random":
            for i in range(n_no_prompt_random):
                npr_specs.append(ConditionSpec(name=f"no_prompt_random_{i:03d}", use_system_prompt=False, steering_kind="random", random_seed=10000 + i))
        elif c == "null":
            key_specs.append(ConditionSpec(name="null", use_system_prompt=False, steering_kind="none"))
        else:
            raise ValueError(f"Unknown condition token: {c}")

    # Interleave: 10 batches, random vectors distributed evenly via stride
    N_BATCHES = 10
    chunk_size = max(1, len(random_specs) // N_BATCHES)
    npr_chunk = max(1, len(npr_specs) // N_BATCHES)

    out: List[ConditionSpec] = []
    # Batch 0: all key conditions + first chunk of randoms + first chunk of no-prompt-randoms
    out.extend(key_specs)
    out.extend(random_specs[:chunk_size])
    out.extend(npr_specs[:npr_chunk])
    # Batches 1..9: subsequent chunks
    for b in range(1, N_BATCHES):
        start_r = b * chunk_size
        end_r = min(start_r + chunk_size, len(random_specs))
        out.extend(random_specs[start_r:end_r])
        start_n = b * npr_chunk
        end_n = min(start_n + npr_chunk, len(npr_specs))
        out.extend(npr_specs[start_n:end_n])
    # Any remainders (if N_BATCHES doesn't divide evenly)
    added = set(c.name for c in out)
    for c in random_specs + npr_specs:
        if c.name not in added:
            out.append(c)
            added.add(c.name)

    return out


# --------------------------------------------------------------------------------------
# Evaluation loop
# --------------------------------------------------------------------------------------

def evaluate_condition(
    cond: ConditionSpec,
    model: Any,
    processor: Any,
    style_set: List[PromptItem],
    math_set: List[PromptItem],
    raw_dir: Path,
    score_sarcasm_dual,
    check_gsm8k_answer,
    real_vecs: Dict[int, torch.Tensor],
    alpha: float,
) -> None:
    style_path = raw_dir / f"{cond.name}_style.jsonl"
    math_path = raw_dir / f"{cond.name}_math.jsonl"

    # Per-prompt resume: count existing lines and skip already-done prompts
    style_done = count_jsonl_lines(style_path)
    math_done = count_jsonl_lines(math_path) if style_done >= len(style_set) else 0

    if style_done >= len(style_set) and math_done >= len(math_set):
        return  # fully complete

    layer_vecs: Dict[int, torch.Tensor] = {}
    if cond.steering_kind == "real":
        layer_vecs = real_vecs
    elif cond.steering_kind == "random":
        if cond.random_seed is None:
            raise ValueError("Random condition requires random_seed")
        layer_vecs = make_random_like(real_vecs, cond.random_seed)
    elif cond.steering_kind == "shuffled":
        layer_vecs = make_shuffled(real_vecs, cond.random_seed or 0)

    handles: List[Any] = []
    if cond.steering_kind != "none":
        handles = apply_hooks(model, layer_vecs, alpha)

    try:
        # ── Style prompts (append mode for per-prompt resume) ──
        if style_done < len(style_set):
            remaining_style = style_set[style_done:]
            mode = "a" if style_done > 0 else "w"
            if style_done > 0:
                print(f"  Resuming {cond.name} style from prompt {style_done}/{len(style_set)}")
            ensure_dir(style_path.parent)
            with style_path.open(mode, encoding="utf-8") as f:
                for item in tqdm(remaining_style, desc=f"{cond.name} style", leave=False,
                                 initial=style_done, total=len(style_set)):
                    full, vis, think_n, total_n = run_single_generation(
                        model, processor, item.prompt, cond.use_system_prompt)
                    row = {
                        "condition": cond.name,
                        "prompt_type": "style",
                        "prompt": item.prompt,
                        "full_response": full,
                        "visible_response": vis,
                        "sarcasm": score_sarcasm_dual(vis),
                        "math_correct": None,
                        "think_tokens": think_n,
                        "total_tokens": total_n,
                    }
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
                    f.flush()  # flush after every prompt — crash-safe
                    if _SHUTDOWN_REQUESTED:
                        print(f"\n  Shutdown: saved {cond.name} style up to this prompt. Safe to restart.")
                        return

        # ── Math prompts (append mode for per-prompt resume) ──
        math_done = count_jsonl_lines(math_path)
        if math_done < len(math_set):
            remaining_math = math_set[math_done:]
            mode = "a" if math_done > 0 else "w"
            if math_done > 0:
                print(f"  Resuming {cond.name} math from prompt {math_done}/{len(math_set)}")
            ensure_dir(math_path.parent)
            with math_path.open(mode, encoding="utf-8") as f:
                for item in tqdm(remaining_math, desc=f"{cond.name} math", leave=False,
                                 initial=math_done, total=len(math_set)):
                    full, vis, think_n, total_n = run_single_generation(
                        model, processor, item.prompt, cond.use_system_prompt)
                    ok = check_gsm8k_answer(vis, item.answer)
                    row = {
                        "condition": cond.name,
                        "prompt_type": "math",
                        "prompt": item.prompt,
                        "full_response": full,
                        "visible_response": vis,
                        "sarcasm": score_sarcasm_dual(vis),
                        "math_correct": bool(ok),
                        "think_tokens": think_n,
                        "total_tokens": total_n,
                    }
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
                    f.flush()
                    if _SHUTDOWN_REQUESTED:
                        print(f"\n  Shutdown: saved {cond.name} math up to this prompt. Safe to restart.")
                        return
    finally:
        remove_handles(handles)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# --------------------------------------------------------------------------------------
# Analysis
# --------------------------------------------------------------------------------------

def sarcasm_rate(rows: List[Dict[str, Any]]) -> float:
    if not rows:
        return float("nan")
    vals = [1.0 if r.get("sarcasm", {}).get("is_sarcastic", False) else 0.0 for r in rows]
    return float(np.mean(vals))


def math_acc(rows: List[Dict[str, Any]]) -> float:
    vals = [1.0 if r.get("math_correct") else 0.0 for r in rows if r.get("math_correct") is not None]
    return float(np.mean(vals)) if vals else float("nan")


def holm_bonferroni(pvals: Dict[str, float]) -> Dict[str, float]:
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items)
    adj: Dict[str, float] = {}
    running_max = 0.0
    for i, (k, p) in enumerate(items, start=1):
        ap = min(1.0, (m - i + 1) * p)
        running_max = max(running_max, ap)
        adj[k] = running_max
    return adj


def analyze(output_dir: Path) -> None:
    raw = output_dir / "raw"
    files = list(raw.glob("*_style.jsonl"))
    if not files:
        print("No raw style JSONL files found — skipping analysis.")
        return

    style_by_cond: Dict[str, List[Dict[str, Any]]] = {}
    math_by_cond: Dict[str, List[Dict[str, Any]]] = {}

    for sp in files:
        cond = sp.name.replace("_style.jsonl", "")
        style_by_cond[cond] = read_jsonl(sp)
        mp = raw / f"{cond}_math.jsonl"
        if mp.exists():
            math_by_cond[cond] = read_jsonl(mp)

    n_conds = len(style_by_cond)
    n_random = sum(1 for k in style_by_cond if k.startswith("random_"))
    print(f"\nAnalysis: {n_conds} conditions loaded ({n_random} random vectors)")

    real_rate = sarcasm_rate(style_by_cond.get("real", []))
    random_rates = [sarcasm_rate(v) for k, v in sorted(style_by_cond.items()) if k.startswith("random_")]
    shuffled_rate = sarcasm_rate(style_by_cond.get("shuffled", []))
    no_steer_acc = math_acc(math_by_cond.get("no_steer", []))
    real_acc = math_acc(math_by_cond.get("real", []))

    no_prompt_real = sarcasm_rate(style_by_cond.get("no_prompt_real", []))
    no_prompt_random_rates = [sarcasm_rate(v) for k, v in sorted(style_by_cond.items()) if k.startswith("no_prompt_random_")]

    if not random_rates:
        print("WARNING: No random_* conditions yet — partial analysis only.")
        print(f"  real sarcasm rate: {real_rate:.4f}")
        print(f"  shuffled sarcasm rate: {shuffled_rate:.4f}")
        print(f"  no-steer math acc: {no_steer_acc:.4f}")
        print(f"  real math acc: {real_acc:.4f}")
        return

    # 1) Empirical p-value
    ge = sum(1 for r in random_rates if r >= real_rate)
    p_emp = (ge + 1) / (len(random_rates) + 1)

    # 2) Shuffled comparison — bootstrap CI on rate difference
    if math.isfinite(shuffled_rate) and math.isfinite(real_rate):
        real_rows = style_by_cond.get("real", [])
        shuf_rows = style_by_cond.get("shuffled", [])
        real_hits = np.array([1.0 if r.get("sarcasm", {}).get("is_sarcastic") else 0.0 for r in real_rows])
        shuf_hits = np.array([1.0 if r.get("sarcasm", {}).get("is_sarcastic") else 0.0 for r in shuf_rows])
        n_boot = 10000
        boot_rng = np.random.RandomState(42)
        boot_diffs = np.zeros(n_boot)
        for b in range(n_boot):
            ri = boot_rng.choice(len(real_hits), len(real_hits), replace=True)
            si = boot_rng.choice(len(shuf_hits), len(shuf_hits), replace=True)
            boot_diffs[b] = real_hits[ri].mean() - shuf_hits[si].mean()
        p_shuffle = float((boot_diffs <= 0).sum() / n_boot)  # one-sided: P(real <= shuffled)
    else:
        p_shuffle = float("nan")

    # 3) Math non-inferiority (exact binomial, margin 3pp)
    margin = 0.03
    real_math_rows = math_by_cond.get("real", [])
    base_math_rows = math_by_cond.get("no_steer", [])
    n_real = len(real_math_rows)
    n_base = len(base_math_rows)
    if n_real > 0 and n_base > 0:
        from scipy.stats import binomtest  # type: ignore[import-untyped]
        # H0: real_acc <= no_steer_acc - margin. One-sided test.
        real_correct = sum(1 for r in real_math_rows if r.get("math_correct"))
        expected_rate = max(0.01, min(0.99, no_steer_acc - margin))
        try:
            p_noninf = binomtest(real_correct, n_real, expected_rate, alternative="greater").pvalue
        except (TypeError, ValueError):
            # Fallback to normal approx with continuity correction
            p_pool = np.clip(np.nanmean([real_acc, no_steer_acc]), 1e-6, 1 - 1e-6)
            se = math.sqrt(p_pool * (1 - p_pool) * (1 / n_real + 1 / n_base))
            z = ((real_acc - no_steer_acc) + margin) / max(se, 1e-8)
            p_noninf = 1.0 - norm_cdf(z)
    else:
        p_noninf = float("nan")

    # 4) No-prompt test — rank-based: where does real fall in the no-prompt-random distribution?
    median_npr = float(np.median(no_prompt_random_rates)) if no_prompt_random_rates else float("nan")
    if math.isfinite(no_prompt_real) and len(no_prompt_random_rates) > 0:
        npr_arr = np.array(no_prompt_random_rates)
        ge_count = int((npr_arr >= no_prompt_real).sum())
        p_noprompt = (ge_count + 1) / (len(npr_arr) + 1)  # empirical p-value, same logic as random test
    else:
        p_noprompt = float("nan")

    # 5) Holm-Bonferroni
    pvals = {
        "empirical_random_vs_real": p_emp,
        "real_vs_shuffled": p_shuffle,
        "math_noninferiority_3pp": p_noninf,
        "no_prompt_real_vs_random_median": p_noprompt,
    }
    pvals_holm = holm_bonferroni(pvals)

    # 6) Cohen's d (real vs random distribution for sarcasm rate)
    rr = np.array(random_rates, dtype=float)
    d = (real_rate - float(np.mean(rr))) / float(np.std(rr, ddof=1) + 1e-8)

    p95 = float(np.percentile(rr, 95))
    p99 = float(np.percentile(rr, 99))
    math_drop_pp = (no_steer_acc - real_acc) * 100.0

    go_all = (
        real_rate > p99 and
        real_rate > shuffled_rate and
        math_drop_pp <= 3.0 and
        (math.isfinite(no_prompt_real) and math.isfinite(median_npr) and no_prompt_real > median_npr)
    )
    no_go_any = (
        real_rate < p95 or
        math_drop_pp > 5.0 or
        (not (math.isfinite(no_prompt_real) and math.isfinite(median_npr) and no_prompt_real > median_npr))
    )

    summary = {
        "metrics": {
            "real_sarcasm_rate": real_rate,
            "random_sarcasm_rates": random_rates,
            "random_p95": p95,
            "random_p99": p99,
            "shuffled_sarcasm_rate": shuffled_rate,
            "real_math_acc": real_acc,
            "no_steer_math_acc": no_steer_acc,
            "math_drop_pp": math_drop_pp,
            "no_prompt_real_sarcasm_rate": no_prompt_real,
            "no_prompt_random_median": median_npr,
            "cohens_d_real_vs_random": d,
        },
        "p_values": pvals,
        "p_values_holm_bonferroni": pvals_holm,
        "go_criteria": {
            "real_gt_99pct_random": real_rate > p99,
            "real_gt_shuffled": real_rate > shuffled_rate,
            "math_drop_le_3pp": math_drop_pp <= 3.0,
            "no_prompt_real_gt_no_prompt_random_median": (
                math.isfinite(no_prompt_real) and math.isfinite(median_npr) and no_prompt_real > median_npr
            ),
            "all_pass": go_all,
        },
        "no_go_flags": {
            "real_lt_95pct_random": real_rate < p95,
            "math_drop_gt_5pp": math_drop_pp > 5.0,
            "no_prompt_no_differentiation": not (
                math.isfinite(no_prompt_real) and math.isfinite(median_npr) and no_prompt_real > median_npr
            ),
            "any": no_go_any,
        },
    }

    save_json(output_dir / "summary.json", summary)

    analysis_md = [
        "# Factorial Ablation Analysis",
        "",
        f"- Real sarcasm rate: **{real_rate:.4f}**",
        f"- Random p95/p99: **{p95:.4f} / {p99:.4f}**",
        f"- Empirical p-value (random >= real): **{p_emp:.6f}**",
        f"- Shuffled sarcasm rate: **{shuffled_rate:.4f}**",
        f"- Real math acc: **{real_acc:.4f}**",
        f"- No-steer math acc: **{no_steer_acc:.4f}**",
        f"- Math drop (pp): **{math_drop_pp:.2f}**",
        f"- No-prompt real sarcasm: **{no_prompt_real:.4f}**",
        f"- No-prompt random median sarcasm: **{median_npr:.4f}**",
        f"- Cohen's d (real vs random): **{d:.4f}**",
        "",
        "## GO criteria (ALL must pass)",
        f"- Real sarcasm > 99th percentile random: **{summary['go_criteria']['real_gt_99pct_random']}**",
        f"- Shuffled sarcasm < real sarcasm: **{summary['go_criteria']['real_gt_shuffled']}**",
        f"- Math drop <= 3pp vs no-steer: **{summary['go_criteria']['math_drop_le_3pp']}**",
        f"- No-prompt real > no-prompt random median: **{summary['go_criteria']['no_prompt_real_gt_no_prompt_random_median']}**",
        f"- **ALL PASS: {summary['go_criteria']['all_pass']}**",
        "",
        "## NO-GO flags (ANY triggers NO-GO)",
        f"- Real sarcasm < 95th percentile random: **{summary['no_go_flags']['real_lt_95pct_random']}**",
        f"- Math drop > 5pp: **{summary['no_go_flags']['math_drop_gt_5pp']}**",
        f"- No-prompt conditions show no differentiation: **{summary['no_go_flags']['no_prompt_no_differentiation']}**",
        f"- **ANY TRUE: {summary['no_go_flags']['any']}**",
        "",
        "## Holm-Bonferroni adjusted p-values",
    ]
    for k, v in summary["p_values_holm_bonferroni"].items():
        analysis_md.append(f"- {k}: **{v:.6g}**")
    (output_dir / "analysis.md").write_text("\n".join(analysis_md), encoding="utf-8")

    print("\n".join(analysis_md))


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Factorial ablation harness for steering-vector signal validation.")
    p.add_argument("--conditions", type=str,
                   default="real,no-steer,null,shuffled,no-prompt-real,no-prompt-random,random",
                   help="Condition order matters: key conditions first for early signal.")
    p.add_argument("--n-random", type=int, default=256)
    p.add_argument("--n-no-prompt-random", type=int, default=10)
    p.add_argument("--layers", type=str, default="22,29,30")
    p.add_argument("--alpha", type=float, default=8.0)
    p.add_argument("--n-style", type=int, default=100)
    p.add_argument("--n-math", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", type=str, default=DEFAULT_OUTPUT)
    p.add_argument("--model-name", type=str, default=DEFAULT_MODEL_NAME)
    p.add_argument("--connectome-path", type=str, default=DEFAULT_CONNECTOME_PATH)
    p.add_argument("--sarcasm-cat-idx", type=int, default=6, help="Sarcasm category index in connectome. Default=6 (Tone: Sarcastic).")
    p.add_argument("--analyze", action="store_true")
    p.add_argument("--shard", type=int, default=0, help="Shard index (0-based). Run only conditions in this shard.")
    p.add_argument("--n-shards", type=int, default=1, help="Total number of shards. 1=no sharding (run all).")
    p.add_argument("--interim-every", type=int, default=20, help="Run interim analysis every N conditions (0=disabled).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output)
    ensure_dir(output_dir)
    ensure_dir(output_dir / "raw")

    if args.analyze:
        analyze(output_dir)
        return

    set_global_seed(args.seed)
    check_gsm8k_answer, score_sarcasm_dual, eb, _ = import_eval_modules()

    style_set, math_set = freeze_or_load_prompts(output_dir, eb, args.n_style, args.n_math, args.seed)

    connectome = torch.load(args.connectome_path, map_location="cpu", weights_only=True)
    if connectome.ndim != 3:
        raise ValueError(f"Unexpected connectome shape: {tuple(connectome.shape)}")
    layers = [int(x.strip()) for x in args.layers.split(",") if x.strip()]

    sarcasm_idx = args.sarcasm_cat_idx
    print(f"Using sarcasm category index: {sarcasm_idx}")
    real_vecs = get_real_vectors(connectome, sarcasm_idx, layers)

    model, processor = load_model_processor(args.model_name)

    req_conditions = [x.strip() for x in args.conditions.split(",") if x.strip()]
    condition_specs = expand_conditions(req_conditions, args.n_random, args.n_no_prompt_random)

    # ── Sharding: split conditions across parallel jobs ──
    if args.n_shards > 1:
        shard_specs = [c for i, c in enumerate(condition_specs) if i % args.n_shards == args.shard]
        print(f"Shard {args.shard}/{args.n_shards}: {len(shard_specs)}/{len(condition_specs)} conditions")
        condition_specs = shard_specs

    meta = {
        "seed": args.seed,
        "layers": layers,
        "alpha": args.alpha,
        "n_style": len(style_set),
        "n_math": len(math_set),
        "model_name": args.model_name,
        "connectome_path": args.connectome_path,
        "sarcasm_cat_idx": sarcasm_idx,
        "shard": args.shard,
        "n_shards": args.n_shards,
        "conditions_in_shard": [c.name for c in condition_specs],
    }
    save_json(output_dir / f"run_meta_shard{args.shard}.json", meta)

    # ── Timing and progress ──
    t_start = time.time()
    completed = 0
    skipped = 0

    for i, cond in enumerate(tqdm(condition_specs, desc=f"Shard {args.shard}")):
        if _SHUTDOWN_REQUESTED:
            print(f"\nShutdown after {completed} conditions. All progress saved. Safe to restart.")
            break

        # Check if already done before counting
        raw_dir = output_dir / "raw"
        sp = raw_dir / f"{cond.name}_style.jsonl"
        mp = raw_dir / f"{cond.name}_math.jsonl"
        already_done = (sp.exists() and mp.exists() and
                        count_jsonl_lines(sp) >= len(style_set) and
                        count_jsonl_lines(mp) >= len(math_set))

        evaluate_condition(
            cond=cond,
            model=model,
            processor=processor,
            style_set=style_set,
            math_set=math_set,
            raw_dir=raw_dir,
            score_sarcasm_dual=score_sarcasm_dual,
            check_gsm8k_answer=check_gsm8k_answer,
            real_vecs=real_vecs,
            alpha=args.alpha,
        )

        if already_done:
            skipped += 1
        else:
            completed += 1

        # Timing estimate
        elapsed = time.time() - t_start
        if completed > 0:
            per_cond = elapsed / completed
            remaining = len(condition_specs) - (i + 1)
            eta_h = (remaining * per_cond) / 3600
            print(f"  [{completed} done, {skipped} skipped, ~{eta_h:.1f}h remaining]")

        # ── Interim analysis every N conditions ──
        if args.interim_every > 0 and (i + 1) % args.interim_every == 0:
            print(f"\n--- Interim analysis at condition {i+1}/{len(condition_specs)} ---")
            try:
                analyze(output_dir)
            except Exception as e:
                print(f"  Interim analysis failed (non-fatal): {e}")
            print("--- End interim ---\n")

    # Final analysis
    elapsed_total = time.time() - t_start
    print(f"\nRun complete: {completed} generated, {skipped} resumed, {elapsed_total/3600:.1f}h total")
    try:
        analyze(output_dir)
    except Exception as e:
        print(f"Final analysis error (run --analyze separately): {e}")


if __name__ == "__main__":
    main()
