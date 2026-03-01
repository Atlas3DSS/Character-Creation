#!/usr/bin/env python3
"""
Hybrid Eval Runner v2 — MC Logprob + Generation Scoring.

Two scoring paths:
  MC categories (math_mc, knowledge_mc):
    prefill only → logits → log_softmax → argmax over choice token IDs
    ~0.5s per question

  Generation categories (sarcasm, code, math_gen):
    full autoregressive generation → score response
    ~40-120s per question

Usage:
    python eval_runner.py \\
        --model Qwen/Qwen3-VL-8B-Thinking \\
        --n-per-category 50 \\
        --seed 42 \\
        --conditions C0 C1 C2 C3 \\
        --output ./eval_results_v2 \\
        --mc-only       # Fast capability check (~5 min)
        --gen-only      # Personality/code only
        --resume        # Skip completed conditions
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from scipy.stats import fisher_exact as _fisher_exact
from tqdm import tqdm

# ── Imports from existing infrastructure ───────────────────
from phase_aware_cot_steering import (
    ALPHA,
    CHAMPION_LAYERS,
    CONNECTOME_PATH,
    SARCASM_CAT_IDX,
    V4_SYSTEM_PROMPT,
    PhaseAwareHook,
    PhaseState,
    PhaseTrackingProcessor,
    StaticHook,
    extract_thinking,
    generate,
    get_think_token_ids,
    load_model,
    load_steering_vectors,
    setup_condition,
    strip_thinking,
)
from eval_battery import (
    ASSISTANT_MARKERS,
    STRONG_SARCASM_MARKERS,
    MCQuestion,
    GenQuestion,
    CodeQuestion,
    battery_to_dict,
    sample_all,
)


# ═══════════════════════════════════════════════════════════
#  CONSTANTS
# ═══════════════════════════════════════════════════════════

CONDITION_ORDER = ["C0", "C1", "C2", "C3"]

TOKEN_BUDGET = {
    "math_gen": 512,
    "sarcasm": 1024,
    "code": 1024,
}

# MC categories (logprob-scored) — no token budget needed
MC_CATEGORIES = ["math_mc", "knowledge_mc"]

# Generation categories — need token budget + generation
GEN_CATEGORIES = ["sarcasm", "code", "math_gen"]

ALL_CATEGORIES = MC_CATEGORIES + GEN_CATEGORIES

# Load full 1,328-marker sarcasm list for broad scoring
_SARCASM_MARKERS_PATH = Path(__file__).parent / "sarcasm_markers.json"
_FULL_SARCASM_LIST: list[str] = []
if _SARCASM_MARKERS_PATH.exists():
    with open(_SARCASM_MARKERS_PATH) as _f:
        _data = json.load(_f)
        _FULL_SARCASM_LIST = _data.get("flat_sarcasm_list", [])


# ═══════════════════════════════════════════════════════════
#  MC LOGPROB SCORING
# ═══════════════════════════════════════════════════════════

def _get_choice_token_ids(processor, labels: list[str]) -> dict[str, int]:
    """Get token IDs for MC choice labels (A, B, C, D, ...).

    Returns: {"A": token_id, "B": token_id, ...}
    """
    tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor
    result: dict[str, int] = {}
    for label in labels:
        ids = tokenizer.encode(label, add_special_tokens=False)
        result[label] = ids[0]
    return result


def score_mc_logprobs(
    model,
    processor,
    questions: list[MCQuestion],
    hooks: list,
    system_prompt: str | None,
) -> list[dict]:
    """Prefill-only MC scoring using logprobs.

    For each question:
    1. Build chat messages with the MC prompt
    2. Tokenize with add_generation_prompt=False (no <think> prefix)
    3. Single forward pass to get logits at last position
    4. Compare log_softmax values for choice token IDs
    5. Predict = argmax over choices

    Returns list of per-item result dicts.
    """
    device = next(model.parameters()).device
    results: list[dict] = []

    for q in tqdm(questions, desc="MC scoring", leave=False):
        # Build messages
        msgs: list[dict] = []
        if system_prompt:
            msgs.append({"role": "system", "content": system_prompt})
        msgs.append({"role": "user", "content": q.prompt})

        # CRITICAL: add_generation_prompt=False — no <think>\n prefix.
        # We manually add the assistant turn start so the model knows
        # it should answer, but without the thinking tag.
        text = processor.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=False,
        )
        text += "<|im_start|>assistant\n"

        inputs = processor(text=[text], return_tensors="pt", padding=True).to(device)

        with torch.no_grad():
            out = model(**inputs)
            logits = out.logits[0, -1, :]  # Last position logits

        log_probs = torch.log_softmax(logits.float(), dim=-1)

        # Get token IDs for this question's choices
        choice_ids = _get_choice_token_ids(processor, q.choices)

        # Extract log probs for each choice
        choice_logprobs = {}
        for label in q.choices:
            tid = choice_ids[label]
            choice_logprobs[label] = log_probs[tid].item()

        # Prediction = highest logprob among choices
        pred_label = max(choice_logprobs, key=choice_logprobs.get)
        pred_idx = q.choices.index(pred_label)
        correct = (pred_idx == q.answer_idx)

        results.append({
            "prompt": q.prompt[:200],
            "source": q.source,
            "answer_idx": q.answer_idx,
            "answer_label": q.choices[q.answer_idx],
            "pred_label": pred_label,
            "pred_idx": pred_idx,
            "correct": correct,
            "choice_logprobs": choice_logprobs,
        })

    return results


# ═══════════════════════════════════════════════════════════
#  GENERATION SCORING
# ═══════════════════════════════════════════════════════════

def check_gsm8k_answer(response: str, expected: str) -> bool:
    """Check if response contains the expected GSM8K numerical answer.

    Extraction priority:
    1. #### N pattern (GSM8K format)
    2. \\boxed{N} pattern
    3. Float comparison on all numbers in the response
    """
    clean = response.lower().replace(",", "").replace("$", "").strip()
    exp = expected.lower().replace(",", "").strip()

    try:
        exp_float = float(exp)
    except ValueError:
        # Fallback to substring match
        return exp in clean

    # Try #### pattern first
    m = re.search(r"####\s*(-?\d[\d,]*\.?\d*)", clean)
    if m:
        try:
            val = float(m.group(1).replace(",", ""))
            if abs(val - exp_float) < 1e-6:
                return True
        except ValueError:
            pass

    # Try \boxed{} pattern
    m = re.search(r"\\boxed\{([^}]+)\}", response)
    if m:
        try:
            val = float(m.group(1).replace(",", "").strip())
            if abs(val - exp_float) < 1e-6:
                return True
        except ValueError:
            pass

    # Scan for any matching number in the response
    for m in re.finditer(r"-?\d+(?:\.\d+)?", clean):
        try:
            if abs(float(m.group()) - exp_float) < 1e-6:
                return True
        except ValueError:
            pass

    return False


def score_sarcasm_dual(response: str) -> dict:
    """Dual-level sarcasm scoring.

    Returns:
        strong_count: Skippy-specific markers (30 items)
        broad_count:  Full 1,328-marker list from sarcasm_markers.json
        assistant_count: Assistant leak markers
        is_sarcastic: strong_count >= 2 (primary metric)
        is_assistant: assistant_count >= 1
    """
    lower = response.lower()
    strong = sum(1 for m in STRONG_SARCASM_MARKERS if m in lower)
    broad = sum(1 for m in _FULL_SARCASM_LIST if m in lower) if _FULL_SARCASM_LIST else 0
    assistant = sum(1 for m in ASSISTANT_MARKERS if m in lower)
    return {
        "strong_count": strong,
        "broad_count": broad,
        "assistant_count": assistant,
        "is_sarcastic": strong >= 2,
        "is_assistant": assistant >= 1,
    }


def score_humaneval(model_output: str, prompt: str, test_code: str,
                    entry_point: str, timeout: int = 10) -> bool:
    """Score HumanEval problem by sandboxed execution.

    1. Extract code from model output (handles ```python blocks)
    2. Concatenate: prompt + extracted_code + test_code
    3. Execute in subprocess with timeout
    4. Return True if exit code == 0
    """
    # Extract code from model output
    code = _extract_code(model_output)

    # Build full program: function def + test harness
    full_program = prompt + code + "\n" + test_code

    # Execute in sandboxed subprocess
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=True, dir="/tmp"
    ) as f:
        f.write(full_program)
        f.flush()

        try:
            result = subprocess.run(
                [sys.executable, f.name],
                capture_output=True,
                timeout=timeout,
                text=True,
            )
            return result.returncode == 0
        except subprocess.TimeoutExpired:
            return False
        except OSError:
            return False


def _extract_code(response: str) -> str:
    """Extract Python code from a model response.

    Handles:
    - ```python ... ``` blocks
    - ``` ... ``` blocks
    - Raw code (fallback)
    """
    # Try ```python ... ``` first
    m = re.search(r"```python\s*\n(.*?)```", response, re.DOTALL)
    if m:
        return m.group(1)

    # Try ``` ... ``` (any language)
    m = re.search(r"```\s*\n(.*?)```", response, re.DOTALL)
    if m:
        return m.group(1)

    # Fallback: return everything (model may have just written code directly)
    return response


# ═══════════════════════════════════════════════════════════
#  CONDITION SETUP
# ═══════════════════════════════════════════════════════════

def setup_condition_mc(
    condition: str,
    model,
    layers,
    vectors: dict[int, torch.Tensor],
    alpha: float,
    champion_layers: list[int],
    think_id: int,
    end_think_id: int,
) -> tuple[list, str | None, str]:
    """Set up hooks for MC prefill scoring.

    Identical to setup_condition() EXCEPT:
    - C2 uses StaticHook instead of PhaseAwareHook (PhaseAwareHook defaults
      to is_thinking=True → alpha=0 → zero steering during prefill).
    - No logits processors returned (no generation).

    Returns: (hooks, system_prompt, description)
    """
    hooks: list = []

    if condition == "C0":
        return hooks, None, "Base: no V4, no steering"

    elif condition == "C1":
        for l_idx in champion_layers:
            h = layers[l_idx].register_forward_hook(
                StaticHook(vectors[l_idx], alpha)
            )
            hooks.append(h)
        return hooks, V4_SYSTEM_PROMPT, \
            f"V4 + static L{champion_layers}@α={alpha}"

    elif condition == "C2":
        # FALLBACK: Use StaticHook for MC scoring (PhaseAwareHook gives α=0)
        for l_idx in champion_layers:
            h = layers[l_idx].register_forward_hook(
                StaticHook(vectors[l_idx], alpha)
            )
            hooks.append(h)
        return hooks, V4_SYSTEM_PROMPT, \
            f"Phase-aware→static fallback L{champion_layers}@α={alpha}"

    elif condition == "C3":
        return hooks, V4_SYSTEM_PROMPT, "V4 only, no steering"

    else:
        raise ValueError(f"Unknown condition: {condition}")


# ═══════════════════════════════════════════════════════════
#  STATISTICS
# ═══════════════════════════════════════════════════════════

def clopper_pearson(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Clopper-Pearson exact binomial confidence interval."""
    from scipy.stats import beta as beta_dist

    if n == 0:
        return (0.0, 1.0)
    if k == 0:
        lo = 0.0
    else:
        lo = beta_dist.ppf(alpha / 2, k, n - k + 1)
    if k == n:
        hi = 1.0
    else:
        hi = beta_dist.ppf(1 - alpha / 2, k + 1, n - k)
    return (float(lo), float(hi))


def pairwise_fisher(bools_a: list[bool], bools_b: list[bool]) -> float:
    """Two-tailed Fisher's exact test on paired boolean outcomes."""
    a_yes = sum(bools_a)
    a_no = len(bools_a) - a_yes
    b_yes = sum(bools_b)
    b_no = len(bools_b) - b_yes
    table = [[a_yes, a_no], [b_yes, b_no]]
    _, p = _fisher_exact(table, alternative="two-sided")
    return float(p)


def holm_correct(p_values: list[float]) -> list[float]:
    """Holm-Bonferroni step-down correction for multiple comparisons."""
    m = len(p_values)
    if m == 0:
        return []
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    corrected = [0.0] * m
    cummax = 0.0
    for rank, (orig_idx, p) in enumerate(indexed):
        adj = p * (m - rank)
        adj = min(adj, 1.0)
        cummax = max(cummax, adj)
        corrected[orig_idx] = cummax
    return corrected


# ═══════════════════════════════════════════════════════════
#  EVAL LOOPS
# ═══════════════════════════════════════════════════════════

def run_mc_category(
    category: str,
    questions: list[MCQuestion],
    model,
    processor,
    hooks: list,
    sys_prompt: str | None,
) -> list[dict]:
    """Run MC logprob scoring for one category."""
    return score_mc_logprobs(model, processor, questions, hooks, sys_prompt)


def run_gen_category(
    category: str,
    questions: list,
    model,
    processor,
    sys_prompt: str | None,
    logits_procs: list,
    max_tokens: int,
) -> list[dict]:
    """Run generation scoring for one category."""
    results: list[dict] = []

    for item in tqdm(questions, desc=category, leave=False):
        if category == "sarcasm":
            assert isinstance(item, GenQuestion)
            raw = generate(
                model, processor, item.prompt, sys_prompt,
                max_tokens=max_tokens, logits_processors=logits_procs,
            )
            visible = strip_thinking(raw)
            thinking = extract_thinking(raw)

            scores = score_sarcasm_dual(visible)
            results.append({
                "prompt": item.prompt,
                "category": item.category,
                "response": visible[:800],
                "thinking_len": len(thinking),
                **scores,
                "correct": scores["is_sarcastic"],
            })

        elif category == "math_gen":
            assert isinstance(item, GenQuestion)
            raw = generate(
                model, processor, item.prompt, sys_prompt,
                max_tokens=max_tokens, logits_processors=logits_procs,
            )
            visible = strip_thinking(raw)
            thinking = extract_thinking(raw)

            correct = check_gsm8k_answer(visible, item.answer)
            # Also check thinking trace for answer
            if not correct and thinking:
                correct = check_gsm8k_answer(thinking, item.answer)

            results.append({
                "prompt": item.prompt[:200],
                "expected": item.answer,
                "response": visible[:800],
                "thinking_len": len(thinking),
                "correct": correct,
            })

        elif category == "code":
            assert isinstance(item, CodeQuestion)
            # Prompt the model to complete the function
            code_prompt = (
                "Complete the following Python function. "
                "Write ONLY the function body (no explanation).\n\n"
                f"{item.prompt}"
            )
            raw = generate(
                model, processor, code_prompt, sys_prompt,
                max_tokens=max_tokens, logits_processors=logits_procs,
            )
            visible = strip_thinking(raw)
            thinking = extract_thinking(raw)

            correct = score_humaneval(
                visible, item.prompt, item.test_code, item.entry_point,
            )
            results.append({
                "prompt": item.prompt[:200],
                "entry_point": item.entry_point,
                "response": visible[:800],
                "thinking_len": len(thinking),
                "correct": correct,
            })

    return results


def run_eval(
    model,
    processor,
    layers,
    vectors: dict[int, torch.Tensor],
    conditions: list[str],
    alpha: float,
    champion_layers: list[int],
    think_id: int,
    end_think_id: int,
    battery: dict[str, list],
    output_dir: Path,
    model_name: str,
    resume: bool = True,
    mc_only: bool = False,
    gen_only: bool = False,
) -> dict:
    """Run full evaluation across conditions and categories.

    Eval order: MC categories first (fast), then generation categories (slow).

    Returns: {condition: {"description": str, "categories": {cat: [results]}}}
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "checkpoint.json"

    # ── Determine which categories to run ──
    if mc_only:
        active_cats = MC_CATEGORIES
    elif gen_only:
        active_cats = GEN_CATEGORIES
    else:
        active_cats = ALL_CATEGORIES

    # ── Resume ──
    all_results: dict[str, dict] = {}
    if resume and checkpoint_path.exists():
        try:
            with open(checkpoint_path) as f:
                ckpt = json.load(f)
            all_results = ckpt.get("results", {})
            done = [c for c in conditions if c in all_results]
            if done:
                print(f"Resuming: {done} already complete, skipping.")
                conditions = [c for c in conditions if c not in all_results]
        except (json.JSONDecodeError, KeyError):
            pass

    if not conditions:
        print("All conditions already complete.")
        return all_results

    for cond in conditions:
        # Setup for generation categories (with PhaseAwareHook for C2)
        gen_hooks, gen_sys_prompt, gen_logits_procs, gen_desc = setup_condition(
            cond, model, layers, vectors, alpha,
            champion_layers, think_id, end_think_id,
        )

        print(f"\n{'='*65}")
        print(f"  {cond}: {gen_desc}")
        print(f"{'='*65}")

        cond_results: dict[str, list[dict]] = {}

        # ── MC categories (fast, ~0.5s per question) ──
        for cat in MC_CATEGORIES:
            if cat not in active_cats:
                continue
            questions = battery.get(cat, [])
            if not questions:
                print(f"  {cat}: no questions, skipping")
                continue

            # Setup MC-specific hooks (StaticHook fallback for C2)
            # Clean gen hooks first for MC
            for h in gen_hooks:
                h.remove()

            mc_hooks, mc_sys, mc_desc = setup_condition_mc(
                cond, model, layers, vectors, alpha,
                champion_layers, think_id, end_think_id,
            )

            t0 = time.time()
            cat_results = run_mc_category(
                cat, questions, model, processor, mc_hooks, mc_sys,
            )
            elapsed = time.time() - t0

            correct = sum(1 for r in cat_results if r.get("correct", False))
            total = len(cat_results)
            pct = 100.0 * correct / total if total else 0.0
            print(f"  {cat}: {correct}/{total} ({pct:.1f}%)  [{elapsed:.1f}s]")

            cond_results[cat] = cat_results

            # Remove MC hooks
            for h in mc_hooks:
                h.remove()

        # ── Re-setup generation hooks ──
        gen_hooks, gen_sys_prompt, gen_logits_procs, gen_desc = setup_condition(
            cond, model, layers, vectors, alpha,
            champion_layers, think_id, end_think_id,
        )

        # ── Generation categories (slow, ~40-120s per question) ──
        for cat in GEN_CATEGORIES:
            if cat not in active_cats:
                continue
            questions = battery.get(cat, [])
            if not questions:
                print(f"  {cat}: no questions, skipping")
                continue

            max_tok = TOKEN_BUDGET.get(cat, 1024)
            t0 = time.time()
            cat_results = run_gen_category(
                cat, questions, model, processor,
                gen_sys_prompt, gen_logits_procs, max_tok,
            )
            elapsed = time.time() - t0

            correct = sum(1 for r in cat_results if r.get("correct", False))
            total = len(cat_results)
            pct = 100.0 * correct / total if total else 0.0
            print(f"  {cat}: {correct}/{total} ({pct:.1f}%)  [{elapsed:.0f}s]")

            cond_results[cat] = cat_results

        # Cleanup gen hooks
        for h in gen_hooks:
            h.remove()

        all_results[cond] = {
            "description": gen_desc,
            "categories": cond_results,
            "timestamp": datetime.now().isoformat(),
        }

        # Save incremental checkpoint
        _save_checkpoint(all_results, output_dir, model_name, alpha, champion_layers)
        print(f"  Checkpoint saved ({len(all_results)}/{len(CONDITION_ORDER)} conditions)")

    return all_results


def _save_checkpoint(
    results: dict,
    output_dir: Path,
    model_name: str,
    alpha: float,
    champion_layers: list[int],
) -> None:
    """Save incremental checkpoint to disk."""
    checkpoint = {
        "meta": {
            "model": model_name,
            "alpha": alpha,
            "layers": champion_layers,
            "timestamp": datetime.now().isoformat(),
            "connectome": str(CONNECTOME_PATH),
            "token_budgets": TOKEN_BUDGET,
            "version": "v2",
        },
        "results": results,
    }
    checkpoint_path = output_dir / "checkpoint.json"
    with open(checkpoint_path, "w") as f:
        json.dump(checkpoint, f, indent=2, default=str)


# ═══════════════════════════════════════════════════════════
#  STATISTICAL ANALYSIS
# ═══════════════════════════════════════════════════════════

def compute_stats(results: dict, output_dir: Path) -> dict:
    """Compute accuracies, CIs, pairwise Fisher tests, Holm correction."""
    conditions = [c for c in CONDITION_ORDER if c in results]
    active_cats = set()
    for cond in conditions:
        active_cats.update(results[cond].get("categories", {}).keys())
    categories = [c for c in ALL_CATEGORIES if c in active_cats]

    stats: dict = {"conditions": {}, "pairwise": {}, "meta": {}}

    # ── Per-condition, per-category accuracies + CIs ──
    for cond in conditions:
        cond_stats: dict = {}
        cat_data = results[cond].get("categories", {})
        for cat in categories:
            items = cat_data.get(cat, [])
            if not items:
                continue
            correct = sum(1 for r in items if r.get("correct", False))
            total = len(items)
            acc = correct / total if total else 0.0
            lo, hi = clopper_pearson(correct, total)
            cond_stats[cat] = {
                "correct": correct,
                "total": total,
                "accuracy": round(acc, 4),
                "ci_lower": round(lo, 4),
                "ci_upper": round(hi, 4),
            }

            # Sarcasm-specific extras
            if cat == "sarcasm":
                asst_count = sum(1 for r in items if r.get("is_assistant", False))
                cond_stats[cat]["assistant_leak"] = asst_count
                cond_stats[cat]["assistant_rate"] = round(asst_count / total, 4)
                avg_strong = sum(r.get("strong_count", 0) for r in items) / total
                avg_broad = sum(r.get("broad_count", 0) for r in items) / total
                cond_stats[cat]["avg_strong_markers"] = round(avg_strong, 2)
                cond_stats[cat]["avg_broad_markers"] = round(avg_broad, 2)

            # MC-specific extras
            if cat in MC_CATEGORIES:
                # Source breakdown
                sources: dict[str, list[bool]] = defaultdict(list)
                for r in items:
                    src = r.get("source", "unknown")
                    sources[src].append(r.get("correct", False))
                source_accs = {}
                for src, bools in sources.items():
                    source_accs[src] = round(sum(bools) / len(bools), 4) if bools else 0.0
                cond_stats[cat]["source_breakdown"] = source_accs

        stats["conditions"][cond] = cond_stats

    # ── Pairwise Fisher's exact tests ──
    all_p_values: list[float] = []
    comparison_keys: list[str] = []

    pairs = []
    for i, ca in enumerate(conditions):
        for cb in conditions[i + 1:]:
            pairs.append((ca, cb))

    for cat in categories:
        for ca, cb in pairs:
            items_a = results[ca].get("categories", {}).get(cat, [])
            items_b = results[cb].get("categories", {}).get(cat, [])
            if not items_a or not items_b:
                continue
            bools_a = [r.get("correct", False) for r in items_a]
            bools_b = [r.get("correct", False) for r in items_b]
            p = pairwise_fisher(bools_a, bools_b)
            key = f"{cat}:{ca}_vs_{cb}"
            comparison_keys.append(key)
            all_p_values.append(p)

    # Holm correction
    corrected = holm_correct(all_p_values)
    pairwise_stats = {}
    for key, raw_p, adj_p in zip(comparison_keys, all_p_values, corrected):
        pairwise_stats[key] = {
            "p_raw": round(raw_p, 6),
            "p_holm": round(adj_p, 6),
            "significant_005": adj_p < 0.05,
            "significant_001": adj_p < 0.01,
        }

    stats["pairwise"] = pairwise_stats
    stats["meta"] = {
        "n_comparisons": len(all_p_values),
        "correction": "Holm-Bonferroni",
        "alpha": 0.05,
        "categories": categories,
        "timestamp": datetime.now().isoformat(),
    }

    out_path = output_dir / "stats_summary.json"
    with open(out_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"\nStats written to {out_path}")

    return stats


# ═══════════════════════════════════════════════════════════
#  REPORT GENERATION
# ═══════════════════════════════════════════════════════════

def generate_report(
    results: dict,
    stats: dict,
    model_name: str,
    n_per_category: int,
    seed: int,
    output_dir: Path,
) -> str:
    """Generate human-readable markdown report."""
    conditions = [c for c in CONDITION_ORDER if c in results]
    categories = stats["meta"].get("categories", ALL_CATEGORIES)

    lines = [
        f"# Hybrid Eval v2 Report — {model_name}",
        "",
        f"**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"**Samples per category**: {n_per_category}",
        f"**Seed**: {seed}",
        f"**Conditions**: {', '.join(conditions)}",
        f"**Token budgets**: {TOKEN_BUDGET}",
        "",
        "## Summary Table",
        "",
    ]

    # Build header
    header = "| Cond | Description |"
    sep = "|---|---|"
    for cat in categories:
        label = cat.replace("_", " ").title()
        header += f" {label} |"
        sep += "---|"
    lines.append(header)
    lines.append(sep)

    for cond in conditions:
        desc = results[cond].get("description", "")[:40]
        row = f"| {cond} | {desc} |"
        cond_stats = stats["conditions"].get(cond, {})
        for cat in categories:
            cs = cond_stats.get(cat, {})
            if not cs:
                row += " — |"
                continue
            c = cs["correct"]
            t = cs["total"]
            acc = cs["accuracy"] * 100
            lo = cs["ci_lower"] * 100
            hi = cs["ci_upper"] * 100
            row += f" {c}/{t} ({acc:.1f}%) [{lo:.0f}-{hi:.0f}%] |"
        lines.append(row)

    # ── MC Details ──
    mc_cats_present = [c for c in MC_CATEGORIES if c in categories]
    if mc_cats_present:
        lines.extend(["", "### MC Scoring Details", ""])
        for cat in mc_cats_present:
            lines.append(f"**{cat}** — Source breakdown:")
            for cond in conditions:
                cs = stats["conditions"].get(cond, {}).get(cat, {})
                breakdown = cs.get("source_breakdown", {})
                if breakdown:
                    parts = [f"{src}={acc*100:.0f}%" for src, acc in breakdown.items()]
                    lines.append(f"  - {cond}: {', '.join(parts)}")
            lines.append("")

    # ── Sarcasm Details ──
    if "sarcasm" in categories:
        lines.extend(["", "### Sarcasm Details", ""])
        lines.append("| Cond | Sarcastic | Asst Leak | Avg Strong | Avg Broad |")
        lines.append("|---|---|---|---|---|")
        for cond in conditions:
            cs = stats["conditions"].get(cond, {}).get("sarcasm", {})
            if not cs:
                continue
            lines.append(
                f"| {cond} | {cs['correct']}/{cs['total']} "
                f"| {cs.get('assistant_leak', '?')}/{cs['total']} "
                f"| {cs.get('avg_strong_markers', 0):.1f} "
                f"| {cs.get('avg_broad_markers', 0):.1f} |"
            )

    # ── Pairwise comparisons ──
    lines.extend(["", "## Pairwise Comparisons (Fisher's Exact, Holm-corrected)", ""])
    lines.append("| Comparison | p (raw) | p (Holm) | Sig (.05) | Sig (.01) |")
    lines.append("|---|---|---|---|---|")

    pairwise = stats.get("pairwise", {})
    for key in sorted(pairwise.keys()):
        pw = pairwise[key]
        sig05 = "**YES**" if pw["significant_005"] else "no"
        sig01 = "**YES**" if pw["significant_001"] else "no"
        lines.append(
            f"| {key} | {pw['p_raw']:.4f} | {pw['p_holm']:.4f} | {sig05} | {sig01} |"
        )

    lines.extend([
        "",
        f"Total comparisons: {stats['meta']['n_comparisons']} "
        f"(correction: {stats['meta']['correction']})",
        "",
        "---",
        f"Generated by eval_runner.py v2 on {datetime.now().isoformat()}",
    ])

    report = "\n".join(lines)
    out_path = output_dir / "eval_report.md"
    with open(out_path, "w") as f:
        f.write(report)
    print(f"Report written to {out_path}")
    return report


# ═══════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Hybrid Eval v2 — MC logprob + generation scoring",
    )
    parser.add_argument(
        "--model", type=str,
        default="Qwen/Qwen3-VL-8B-Thinking",
        help="HuggingFace model name",
    )
    parser.add_argument(
        "--n-per-category", type=int, default=50,
        help="Number of prompts per category (sampled from battery)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for prompt sampling",
    )
    parser.add_argument(
        "--conditions", nargs="+", default=["C0", "C1", "C2", "C3"],
        choices=["C0", "C1", "C2", "C3"],
        help="Conditions to evaluate",
    )
    parser.add_argument(
        "--output", type=str, default="./eval_results_v2",
        help="Output directory",
    )
    parser.add_argument(
        "--resume", action="store_true", default=False,
        help="Resume from checkpoint (skip completed conditions)",
    )
    parser.add_argument(
        "--alpha", type=float, default=ALPHA,
        help=f"Steering alpha (default: {ALPHA})",
    )
    parser.add_argument(
        "--layers", nargs="+", type=int, default=CHAMPION_LAYERS,
        help=f"Steering layers (default: {CHAMPION_LAYERS})",
    )
    parser.add_argument(
        "--mc-only", action="store_true", default=False,
        help="Only run MC categories (math_mc, knowledge_mc) — ~5 min",
    )
    parser.add_argument(
        "--gen-only", action="store_true", default=False,
        help="Only run generation categories (sarcasm, code, math_gen)",
    )
    parser.add_argument(
        "--stats-only", action="store_true", default=False,
        help="Skip generation, compute stats from existing checkpoint",
    )
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Sample battery ──
    print(f"Sampling {args.n_per_category} prompts per category (seed={args.seed})...")
    battery = sample_all(args.n_per_category, seed=args.seed)
    for cat, items in battery.items():
        types = type(items[0]).__name__ if items else "empty"
        print(f"  {cat}: {len(items)} {types}")

    # Save exact prompts for reproducibility
    battery_path = output_dir / "battery_samples.json"
    with open(battery_path, "w") as f:
        json.dump(battery_to_dict(battery), f, indent=2)
    print(f"Battery saved to {battery_path}")

    if args.stats_only:
        checkpoint_path = output_dir / "checkpoint.json"
        if not checkpoint_path.exists():
            print(f"ERROR: No checkpoint at {checkpoint_path}")
            return
        with open(checkpoint_path) as f:
            ckpt = json.load(f)
        results = ckpt["results"]
        print(f"Loaded {len(results)} conditions from checkpoint")
    else:
        # ── Load model ──
        model, processor, layers = load_model(args.model)
        think_id, end_think_id = get_think_token_ids(processor)

        # ── Load steering vectors ──
        needs_vectors = any(c in args.conditions for c in ["C1", "C2"])
        vectors: dict[int, torch.Tensor] = {}
        if needs_vectors:
            vectors = load_steering_vectors(
                CONNECTOME_PATH, args.layers, SARCASM_CAT_IDX,
            )
            dev = next(model.parameters()).device
            vectors = {k: v.to(dev) for k, v in vectors.items()}

        # ── Run eval ──
        t0 = time.time()
        results = run_eval(
            model, processor, layers, vectors,
            conditions=args.conditions,
            alpha=args.alpha,
            champion_layers=args.layers,
            think_id=think_id,
            end_think_id=end_think_id,
            battery=battery,
            output_dir=output_dir,
            model_name=args.model,
            resume=args.resume,
            mc_only=args.mc_only,
            gen_only=args.gen_only,
        )
        elapsed = time.time() - t0
        print(f"\nTotal eval time: {elapsed / 3600:.2f}h ({elapsed:.0f}s)")

        # Free GPU memory
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ── Compute statistics ──
    print("\nComputing statistics...")
    stats = compute_stats(results, output_dir)

    # ── Generate report ──
    generate_report(
        results, stats, args.model,
        args.n_per_category, args.seed, output_dir,
    )

    # ── Print quick summary ──
    print(f"\n{'='*65}")
    print("QUICK SUMMARY")
    print(f"{'='*65}")
    conditions = [c for c in CONDITION_ORDER if c in stats["conditions"]]
    for cond in conditions:
        cs = stats["conditions"][cond]
        parts = []
        for cat in ALL_CATEGORIES:
            if cat in cs:
                acc = cs[cat]["accuracy"] * 100
                parts.append(f"{cat}={acc:.0f}%")
        print(f"  {cond}: {', '.join(parts)}")

    sig_count = sum(
        1 for pw in stats["pairwise"].values() if pw["significant_005"]
    )
    print(f"\nSignificant comparisons (p<0.05 Holm): {sig_count}/{len(stats['pairwise'])}")


if __name__ == "__main__":
    main()
