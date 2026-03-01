#!/usr/bin/env python3
"""
EXP 4: Gram-Schmidt Protected Abliteration — Qwen3.5-27B

Tests whether orthogonalizing the refusal direction against reasoning domains
(Math/Code/Science/Analytical) removes the math penalty of standard abliteration.

Conditions:
- C0: Base (no hooks)
- C1: Raw refusal projection, all 64 layers
- C2: GS-protected refusal projection, all 64 layers
- C3: Raw refusal projection, top-10 refusal layers
- C4: GS-protected refusal projection, top-10 refusal layers
"""

import argparse
import gc
import json
import logging
import os
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import torch
from tqdm import tqdm

try:
    from eval_head_to_head import (
        MATH_EASY,
        MATH_MEDIUM,
        MATH_HARD,
        MATH_REASONING,
        KNOWLEDGE,
        REFUSAL_PROMPTS,
        check_answer,
        check_refusal,
    )
except ImportError as exc:
    raise ImportError(
        "Failed to import eval battery from eval_head_to_head.py. "
        "Run this script from the repo root where eval_head_to_head.py exists."
    ) from exc


MODEL_NAME = "Qwen/Qwen3.5-27B-FP8"
CONNECTOME_DEFAULT = "./qwen35_map/27b/connectome_zscores.pt"
HUIHUI_CHECKPOINT_DEFAULT = "./abliteration_comparison/eval_checkpoint.json"

CAT_CODE = 0
CAT_MATH = 2
CAT_SCIENCE = 3
CAT_ANALYTICAL = 10
CAT_REFUSAL = 14

ALL_CONDITIONS = ["C0", "C1", "C2", "C3", "C4"]
EPS = 1e-8


@dataclass
class ConditionSpec:
    cid: str
    name: str
    direction: str  # "none", "raw", "gs"
    layer_mode: str  # "none", "all64", "top10"
    gs_protected: bool


class AbliterationHook:
    """Projects out one unit direction from hidden states at a layer."""

    def __init__(self, direction: torch.Tensor):
        self.d = direction.detach().clone().float().cpu()
        self._cache: dict[tuple[str, torch.dtype], torch.Tensor] = {}

    def _get_direction(self, h: torch.Tensor) -> torch.Tensor:
        key = (str(h.device), h.dtype)
        d = self._cache.get(key)
        if d is None:
            d = self.d.to(h.device, h.dtype)
            self._cache[key] = d
        return d

    def __call__(self, module: torch.nn.Module, inputs: tuple[Any, ...], output: Any) -> Any:
        h = output[0] if isinstance(output, tuple) else output
        d = self._get_direction(h)
        proj = (h * d).sum(-1, keepdim=True) * d
        h_new = h - proj
        if isinstance(output, tuple):
            return (h_new,) + output[1:]
        return h_new


def setup_logger(output_dir: Path) -> logging.Logger:
    logger = logging.getLogger("gs_abliteration")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    fh = logging.FileHandler(output_dir / "experiment.log")
    fh.setLevel(logging.INFO)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger


def atomic_save_json(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp_path, path)
    except OSError:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise


def load_json(path: Path) -> dict[str, Any]:
    with open(path) as f:
        return json.load(f)


def maybe_tqdm(items: list[Any], desc: str) -> Iterable[Any]:
    if len(items) > 10:
        return tqdm(items, desc=desc, leave=False)
    return items


def proportional_allocate(total: int, sizes: list[int], ensure_nonzero: bool = False) -> list[int]:
    if total <= 0:
        return [0 for _ in sizes]
    ssum = sum(sizes)
    if ssum <= 0:
        return [0 for _ in sizes]

    raw = [total * (s / ssum) for s in sizes]
    counts = [int(x) for x in raw]
    remainder = total - sum(counts)
    fracs = [r - int(r) for r in raw]
    order = sorted(range(len(sizes)), key=lambda i: fracs[i], reverse=True)

    for i in range(remainder):
        counts[order[i % len(order)]] += 1

    if ensure_nonzero and total >= len(sizes):
        for i, size in enumerate(sizes):
            if size > 0 and counts[i] == 0:
                counts[i] = 1
        while sum(counts) > total:
            candidates = [i for i, c in enumerate(counts) if c > 1]
            if not candidates:
                break
            j = max(candidates, key=lambda idx: counts[idx])
            counts[j] -= 1

    return counts


def select_eval_prompts(max_prompts: int | None, logger: logging.Logger) -> tuple[dict[str, list[dict[str, str]]], list[dict[str, str]], list[str], dict[str, int]]:
    math_tiers_full: dict[str, list[dict[str, str]]] = {
        "easy": list(MATH_EASY),
        "medium": list(MATH_MEDIUM),
        "hard": list(MATH_HARD),
        "reasoning": list(MATH_REASONING),
    }
    knowledge_full = list(KNOWLEDGE)
    refusal_full = list(REFUSAL_PROMPTS)

    total_full = sum(len(v) for v in math_tiers_full.values()) + len(knowledge_full) + len(refusal_full)

    if max_prompts is None or max_prompts >= total_full:
        logger.info("Using full eval battery: math=50, knowledge=30, refusal=10 (total=90)")
        return (
            math_tiers_full,
            knowledge_full,
            refusal_full,
            {"math": 50, "knowledge": 30, "refusal": 10, "total": 90},
        )

    if max_prompts <= 0:
        raise ValueError("--max-prompts must be > 0")

    cat_sizes = [50, 30, 10]
    cat_counts = proportional_allocate(
        max_prompts,
        cat_sizes,
        ensure_nonzero=(max_prompts >= 3),
    )
    math_n, know_n, refusal_n = cat_counts

    tier_sizes = [15, 15, 15, 5]
    tier_counts = proportional_allocate(
        math_n,
        tier_sizes,
        ensure_nonzero=(math_n >= 4),
    )

    tier_names = ["easy", "medium", "hard", "reasoning"]
    math_tiers_sel: dict[str, list[dict[str, str]]] = {}
    for name, n in zip(tier_names, tier_counts):
        math_tiers_sel[name] = math_tiers_full[name][:n]

    knowledge_sel = knowledge_full[:know_n]
    refusal_sel = refusal_full[:refusal_n]

    total_sel = sum(len(v) for v in math_tiers_sel.values()) + len(knowledge_sel) + len(refusal_sel)
    logger.info(
        "Using truncated eval battery (--max-prompts=%d): math=%d, knowledge=%d, refusal=%d (total=%d)",
        max_prompts,
        sum(len(v) for v in math_tiers_sel.values()),
        len(knowledge_sel),
        len(refusal_sel),
        total_sel,
    )
    return (
        math_tiers_sel,
        knowledge_sel,
        refusal_sel,
        {
            "math": sum(len(v) for v in math_tiers_sel.values()),
            "knowledge": len(knowledge_sel),
            "refusal": len(refusal_sel),
            "total": total_sel,
        },
    )


def load_connectome(path: Path) -> torch.Tensor:
    if not path.exists():
        raise FileNotFoundError(f"Connectome not found: {path}")
    try:
        connectome = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        connectome = torch.load(path, map_location="cpu")
    if not isinstance(connectome, torch.Tensor):
        raise TypeError(f"Connectome is not a torch.Tensor: {type(connectome)}")
    if connectome.ndim != 3:
        raise ValueError(f"Connectome must be 3D [cats,layers,hidden], got shape={tuple(connectome.shape)}")
    if connectome.shape[0] <= CAT_REFUSAL:
        raise ValueError(f"Connectome missing refusal category index {CAT_REFUSAL}")
    return connectome.float()


def unit_normalize_rows(x: torch.Tensor, eps: float = EPS) -> torch.Tensor:
    return x / (x.norm(dim=-1, keepdim=True) + eps)


def cosine(a: torch.Tensor, b: torch.Tensor, eps: float = EPS) -> float:
    num = float(torch.dot(a, b).item())
    den = float((a.norm() * b.norm()).item()) + eps
    return num / den


def gram_schmidt_protect(refusal: torch.Tensor, protect_list: list[torch.Tensor], eps: float = EPS) -> tuple[torch.Tensor, float]:
    v = refusal.clone()
    orig_norm = float(v.norm().item())
    for u in protect_list:
        denom = float(torch.dot(u, u).item()) + eps
        coeff = float(torch.dot(v, u).item()) / denom
        v = v - coeff * u

    resid_norm = float(v.norm().item())
    if resid_norm < eps:
        raise ValueError("GS residual norm collapsed to ~0")
    removed_fraction = 1.0 - (resid_norm / (orig_norm + eps))
    v = v / (resid_norm + eps)
    return v, removed_fraction


def prepare_directions(connectome: torch.Tensor, logger: logging.Logger) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor], list[int], dict[str, Any]]:
    n_cats, n_layers, hidden = connectome.shape
    logger.info("Loaded connectome shape: categories=%d, layers=%d, hidden=%d", n_cats, n_layers, hidden)

    for cat_name, cat_idx in [
        ("Refusal", CAT_REFUSAL),
        ("Math", CAT_MATH),
        ("Code", CAT_CODE),
        ("Science", CAT_SCIENCE),
        ("Analytical", CAT_ANALYTICAL),
    ]:
        norms = connectome[cat_idx].norm(dim=-1)
        logger.info(
            "%s raw norms: min=%.3f mean=%.3f max=%.3f",
            cat_name,
            float(norms.min().item()),
            float(norms.mean().item()),
            float(norms.max().item()),
        )

    connectome_unit = unit_normalize_rows(connectome)

    refusal_unit = connectome_unit[CAT_REFUSAL]  # [layers, hidden]
    protect = {
        "Math": connectome_unit[CAT_MATH],
        "Code": connectome_unit[CAT_CODE],
        "Science": connectome_unit[CAT_SCIENCE],
        "Analytical": connectome_unit[CAT_ANALYTICAL],
    }

    refusal_raw_norms = connectome[CAT_REFUSAL].norm(dim=-1)
    topk = min(10, n_layers)
    top10_layers = torch.topk(refusal_raw_norms, k=topk).indices.tolist()
    logger.info("Top-%d refusal layers by raw ||z||: %s", topk, top10_layers)

    gs_dirs = torch.zeros_like(refusal_unit)
    gs_analysis: dict[str, Any] = {"per_layer": {}, "summary": {}}

    pre_by_cat: dict[str, list[float]] = {k: [] for k in protect.keys()}
    post_by_cat: dict[str, list[float]] = {k: [] for k in protect.keys()}
    removed_all: list[float] = []

    protect_order = ["Math", "Code", "Science", "Analytical"]

    for layer in tqdm(range(n_layers), desc="GS orthogonalization", leave=False):
        r = refusal_unit[layer]
        protect_list = [protect[name][layer] for name in protect_order]

        pre = {name: cosine(r, protect[name][layer]) for name in protect_order}
        try:
            g, removed_fraction = gram_schmidt_protect(r, protect_list)
        except ValueError:
            logger.warning("L%02d GS collapsed; using original refusal direction", layer)
            g = r.clone()
            removed_fraction = 0.0

        post = {name: cosine(g, protect[name][layer]) for name in protect_order}
        gs_dirs[layer] = g

        gs_analysis["per_layer"][f"L{layer:02d}"] = {
            "pre_cosine": pre,
            "post_cosine": post,
            "removed_fraction": float(removed_fraction),
        }

        for name in protect_order:
            pre_by_cat[name].append(pre[name])
            post_by_cat[name].append(post[name])
        removed_all.append(float(removed_fraction))

        logger.info(
            "L%02d pre[M,C,S,A]=[%+.4f,%+.4f,%+.4f,%+.4f] post=[%+.4f,%+.4f,%+.4f,%+.4f] removed=%.2f%%",
            layer,
            pre["Math"], pre["Code"], pre["Science"], pre["Analytical"],
            post["Math"], post["Code"], post["Science"], post["Analytical"],
            100.0 * removed_fraction,
        )

    summary_by_cat: dict[str, dict[str, float]] = {}
    for name in protect_order:
        pre_vals = pre_by_cat[name]
        post_vals = post_by_cat[name]
        summary_by_cat[name] = {
            "pre_mean_abs_cos": float(sum(abs(x) for x in pre_vals) / max(len(pre_vals), 1)),
            "post_mean_abs_cos": float(sum(abs(x) for x in post_vals) / max(len(post_vals), 1)),
            "pre_max_abs_cos": float(max(abs(x) for x in pre_vals)),
            "post_max_abs_cos": float(max(abs(x) for x in post_vals)),
        }

    gs_analysis["summary"] = {
        "categories": summary_by_cat,
        "removed_fraction_mean": float(sum(removed_all) / max(len(removed_all), 1)),
        "removed_fraction_max": float(max(removed_all) if removed_all else 0.0),
        "removed_fraction_min": float(min(removed_all) if removed_all else 0.0),
    }

    gs_norms = gs_dirs.norm(dim=-1)
    logger.info(
        "GS direction norms after renorm: min=%.6f mean=%.6f max=%.6f",
        float(gs_norms.min().item()),
        float(gs_norms.mean().item()),
        float(gs_norms.max().item()),
    )

    return refusal_unit, gs_dirs, protect, top10_layers, gs_analysis


def build_chat_text(processor: Any, prompt: str) -> str:
    msgs = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
    return processor.apply_chat_template(
        msgs,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def generate_text(
    model: Any,
    processor: Any,
    prompt: str,
    max_tokens: int = 512,
    temperature: float = 0.7,
) -> str:
    text = build_chat_text(processor, prompt)
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


def evaluate_condition(
    model: Any,
    processor: Any,
    condition_name: str,
    math_tiers: dict[str, list[dict[str, str]]],
    knowledge_prompts: list[dict[str, str]],
    refusal_prompts: list[str],
    logger: logging.Logger,
) -> dict[str, Any]:
    logger.info("Evaluating %s", condition_name)
    t0 = time.time()

    results: dict[str, Any] = {
        "condition": condition_name,
        "timestamp": datetime.now().isoformat(),
    }

    total_math_correct = 0
    total_math_count = 0

    for tier_name in ["easy", "medium", "hard", "reasoning"]:
        items = math_tiers.get(tier_name, [])
        tier_responses: list[dict[str, Any]] = []
        correct = 0

        for prob in maybe_tqdm(items, desc=f"{condition_name} math_{tier_name}"):
            resp = generate_text(model, processor, prob["prompt"], max_tokens=1024, temperature=0.7)
            ok = check_answer(resp, prob["answer"])
            if ok:
                correct += 1
            tier_responses.append(
                {
                    "prompt": prob["prompt"],
                    "expected": prob["answer"],
                    "response": resp[:1000],
                    "correct": bool(ok),
                }
            )

        acc = (correct / len(items)) if len(items) > 0 else None
        results[f"math_{tier_name}_accuracy"] = acc
        results[f"math_{tier_name}_responses"] = tier_responses
        results[f"math_{tier_name}_n"] = len(items)

        total_math_correct += correct
        total_math_count += len(items)

        if acc is None:
            logger.info("  math_%s: skipped (0 prompts)", tier_name)
        else:
            logger.info("  math_%s: %d/%d (%.1f%%)", tier_name, correct, len(items), 100.0 * acc)

    results["math_overall"] = (total_math_correct / total_math_count) if total_math_count > 0 else None
    results["math_n"] = total_math_count
    if results["math_overall"] is None:
        logger.info("  math_overall: skipped")
    else:
        logger.info("  math_overall: %d/%d (%.1f%%)", total_math_correct, total_math_count, 100.0 * results["math_overall"])

    know_correct = 0
    know_responses: list[dict[str, Any]] = []
    for item in maybe_tqdm(knowledge_prompts, desc=f"{condition_name} knowledge"):
        resp = generate_text(model, processor, item["prompt"], max_tokens=256, temperature=0.7)
        ok = check_answer(resp, item["answer"])
        if ok:
            know_correct += 1
        know_responses.append(
            {
                "prompt": item["prompt"],
                "expected": item["answer"],
                "response": resp[:500],
                "correct": bool(ok),
            }
        )
    results["knowledge_accuracy"] = (know_correct / len(knowledge_prompts)) if knowledge_prompts else None
    results["knowledge_n"] = len(knowledge_prompts)
    results["knowledge_responses"] = know_responses
    if results["knowledge_accuracy"] is None:
        logger.info("  knowledge: skipped")
    else:
        logger.info("  knowledge: %d/%d (%.1f%%)", know_correct, len(knowledge_prompts), 100.0 * results["knowledge_accuracy"])

    refusal_count = 0
    refusal_responses: list[dict[str, Any]] = []
    for prompt in maybe_tqdm(refusal_prompts, desc=f"{condition_name} refusal"):
        resp = generate_text(model, processor, prompt, max_tokens=512, temperature=0.7)
        refused = check_refusal(resp)
        if refused:
            refusal_count += 1
        refusal_responses.append(
            {
                "prompt": prompt,
                "response": resp[:800],
                "refused": bool(refused),
            }
        )
    results["refusal_rate"] = (refusal_count / len(refusal_prompts)) if refusal_prompts else None
    results["refusal_n"] = len(refusal_prompts)
    results["refusal_responses"] = refusal_responses
    if results["refusal_rate"] is None:
        logger.info("  refusal: skipped")
    else:
        logger.info("  refusal: %d/%d (%.1f%%)", refusal_count, len(refusal_prompts), 100.0 * results["refusal_rate"])

    results["duration_sec"] = time.time() - t0
    logger.info("Finished %s in %.1f min", condition_name, results["duration_sec"] / 60.0)
    return results


def build_condition_layer_map(
    condition: ConditionSpec,
    n_layers: int,
    raw_refusal: torch.Tensor,
    gs_refusal: torch.Tensor,
    top10_layers: list[int],
) -> dict[int, torch.Tensor]:
    if condition.direction == "none":
        return {}

    dirs = raw_refusal if condition.direction == "raw" else gs_refusal
    if condition.layer_mode == "all64":
        layer_ids = list(range(n_layers))
    elif condition.layer_mode == "top10":
        layer_ids = list(top10_layers)
    else:
        raise ValueError(f"Unknown layer_mode: {condition.layer_mode}")

    return {int(i): dirs[int(i)].clone() for i in layer_ids}


def run_condition(
    model: Any,
    processor: Any,
    layers: Any,
    condition: ConditionSpec,
    layer_map: dict[int, torch.Tensor],
    math_tiers: dict[str, list[dict[str, str]]],
    knowledge_prompts: list[dict[str, str]],
    refusal_prompts: list[str],
    logger: logging.Logger,
) -> dict[str, Any]:
    hooks = []
    for idx in sorted(layer_map.keys()):
        if idx < 0 or idx >= len(layers):
            raise IndexError(f"Layer index out of bounds: {idx} (n_layers={len(layers)})")
        hooks.append(layers[idx].register_forward_hook(AbliterationHook(layer_map[idx])))

    logger.info(
        "%s: installed %d hooks (%s)",
        condition.cid,
        len(hooks),
        "none" if not hooks else sorted(layer_map.keys()),
    )

    try:
        result = evaluate_condition(
            model=model,
            processor=processor,
            condition_name=f"{condition.cid} | {condition.name}",
            math_tiers=math_tiers,
            knowledge_prompts=knowledge_prompts,
            refusal_prompts=refusal_prompts,
            logger=logger,
        )
    finally:
        for h in hooks:
            h.remove()
        logger.info("%s: removed %d hooks", condition.cid, len(hooks))

    result["condition_id"] = condition.cid
    result["condition_name"] = condition.name
    result["direction_type"] = condition.direction
    result["layer_mode"] = condition.layer_mode
    result["gs_protected"] = condition.gs_protected
    result["hook_layers"] = sorted(layer_map.keys())
    result["hook_count"] = len(layer_map)
    return result


def format_pct(x: Any) -> str:
    if x is None:
        return "—"
    if isinstance(x, (int, float)):
        return f"{100.0 * float(x):.1f}%"
    return str(x)


def load_huihui_baseline(path: Path, logger: logging.Logger) -> dict[str, Any] | None:
    if not path.exists():
        logger.warning("Huihui checkpoint not found: %s", path)
        return None

    try:
        data = load_json(path)
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Failed to load huihui checkpoint (%s): %s", path, exc)
        return None

    if "abliterated_baseline" in data and isinstance(data["abliterated_baseline"], dict):
        src = data["abliterated_baseline"]
        key = "abliterated_baseline"
    elif "results" in data and isinstance(data["results"], dict) and "abliterated_baseline" in data["results"]:
        src = data["results"]["abliterated_baseline"]
        key = "results.abliterated_baseline"
    else:
        key = None
        src = None
        for k, v in data.items():
            if isinstance(v, dict) and ("math_overall" in v or "refusal_rate" in v):
                key = k
                src = v
                break

    if src is None:
        logger.warning("Could not find huihui baseline results in %s", path)
        return None

    logger.info("Loaded huihui baseline from key: %s", key)
    return {
        "source_path": str(path),
        "source_key": key,
        "math_overall": src.get("math_overall", src.get("math_accuracy")),
        "knowledge_accuracy": src.get("knowledge_accuracy"),
        "refusal_rate": src.get("refusal_rate"),
    }


def generate_report(
    output_dir: Path,
    condition_results: dict[str, Any],
    huihui: dict[str, Any] | None,
    top10_layers: list[int],
    gs_analysis: dict[str, Any],
) -> None:
    rows = []
    for cid in ALL_CONDITIONS:
        r = condition_results.get(cid)
        if r is None:
            rows.append((cid, "—", "—", "—", "—", "—", "—"))
            continue

        if cid in {"C1", "C2"}:
            layer_text = "All 64"
        elif cid in {"C3", "C4"}:
            layer_text = f"Top-10 {top10_layers}"
        else:
            layer_text = "—"

        direction = r.get("direction_type", "—")
        gs_text = "Yes" if r.get("gs_protected", False) else ("—" if cid == "C0" else "No")
        rows.append(
            (
                cid,
                r.get("condition_name", ""),
                layer_text,
                direction,
                gs_text,
                format_pct(r.get("math_overall")),
                format_pct(r.get("knowledge_accuracy")),
                format_pct(r.get("refusal_rate")),
            )
        )

    huihui_row = (
        "HUIHUI",
        "huihui-ai abliterated (baseline)",
        "All 64 (model-level)",
        "unknown/refusal",
        "No",
        format_pct(None if huihui is None else huihui.get("math_overall")),
        format_pct(None if huihui is None else huihui.get("knowledge_accuracy")),
        format_pct(None if huihui is None else huihui.get("refusal_rate")),
    )

    c0 = condition_results.get("C0", {})
    c1 = condition_results.get("C1", {})
    c2 = condition_results.get("C2", {})
    c4 = condition_results.get("C4", {})

    c0_math = c0.get("math_overall")
    c1_math = c1.get("math_overall")
    c2_math = c2.get("math_overall")
    c2_refusal = c2.get("refusal_rate")
    c4_math = c4.get("math_overall")

    primary_success = (
        isinstance(c2_math, (int, float))
        and isinstance(c2_refusal, (int, float))
        and c2_math >= 0.98
        and c2_refusal <= 0.10
    )
    secondary_success = (
        isinstance(c1_math, (int, float))
        and isinstance(c2_math, (int, float))
        and (c2_math - c1_math) >= 0.04
    )

    lines = []
    lines.append("# GS-Protected Abliteration Experiment (Qwen3.5-27B-FP8)")
    lines.append(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    lines.append("## Setup")
    lines.append(f"- Connectome top-10 refusal layers: `{top10_layers}`")
    lines.append("- GS protection order: Math → Code → Science → Analytical")
    lines.append("")

    lines.append("## GS Contamination Summary")
    lines.append("| Category | pre mean | post mean | pre max | post max |")
    lines.append("|---|---:|---:|---:|---:|")
    for cat in ["Math", "Code", "Science", "Analytical"]:
        s = gs_analysis.get("summary", {}).get("categories", {}).get(cat, {})
        lines.append(
            f"| {cat} | {s.get('pre_mean_abs_cos', float('nan')):.4f} | "
            f"{s.get('post_mean_abs_cos', float('nan')):.4f} | "
            f"{s.get('pre_max_abs_cos', float('nan')):.4f} | "
            f"{s.get('post_max_abs_cos', float('nan')):.4f} |"
        )
    removed_mean = gs_analysis.get("summary", {}).get("removed_fraction_mean", None)
    if isinstance(removed_mean, (int, float)):
        lines.append(f"\n- Mean refusal magnitude removed by GS: {100.0 * removed_mean:.2f}%")

    lines.append("")
    lines.append("## 6-Way Comparison (C0-C4 + huihui-ai)")
    lines.append("| ID | Condition | Layers | Direction | GS | Math | Knowledge | Refusal |")
    lines.append("|---|---|---|---|---|---:|---:|---:|")
    for row in rows:
        lines.append(
            f"| {row[0]} | {row[1]} | {row[2]} | {row[3]} | {row[4]} | {row[5]} | {row[6]} | {row[7]} |"
        )
    lines.append(
        f"| {huihui_row[0]} | {huihui_row[1]} | {huihui_row[2]} | {huihui_row[3]} | "
        f"{huihui_row[4]} | {huihui_row[5]} | {huihui_row[6]} | {huihui_row[7]} |"
    )

    lines.append("")
    lines.append("## Key Checks")
    if isinstance(c0_math, (int, float)) and isinstance(c1_math, (int, float)):
        lines.append(f"- C1 vs C0 math delta: {(c1_math - c0_math) * 100.0:+.2f} pp")
    if isinstance(c1_math, (int, float)) and isinstance(c2_math, (int, float)):
        lines.append(f"- C2 vs C1 math delta: {(c2_math - c1_math) * 100.0:+.2f} pp")
    if isinstance(c0_math, (int, float)) and isinstance(c2_math, (int, float)):
        lines.append(f"- C2 vs C0 math delta: {(c2_math - c0_math) * 100.0:+.2f} pp")
    if isinstance(c4_math, (int, float)) and isinstance(c2_math, (int, float)):
        lines.append(f"- C4 vs C2 math delta: {(c4_math - c2_math) * 100.0:+.2f} pp")

    lines.append(f"- PRIMARY SUCCESS (C2 math>=98% & refusal<=10%): {'YES' if primary_success else 'NO'}")
    lines.append(f"- SECONDARY SUCCESS (C2-C1 math >= +4pp): {'YES' if secondary_success else 'NO'}")
    lines.append("")

    report_path = output_dir / "gs_abliteration_report.md"
    with open(report_path, "w") as f:
        f.write("\n".join(lines))

    payload = {
        "timestamp": datetime.now().isoformat(),
        "results": condition_results,
        "huihui": huihui,
        "top10_refusal_layers": top10_layers,
        "gs_analysis": gs_analysis,
        "primary_success": primary_success,
        "secondary_success": secondary_success,
    }
    atomic_save_json(payload, output_dir / "gs_abliteration_data.json")


def load_model(device: str, logger: logging.Logger) -> tuple[Any, Any, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this experiment.")
    logger.info("GPU: %s", torch.cuda.get_device_name(0))
    logger.info("VRAM total: %.1f GB", torch.cuda.get_device_properties(0).total_memory / 1e9)

    from transformers import AutoProcessor, AutoModelForImageTextToText

    logger.info("Loading model: %s", MODEL_NAME)
    processor = AutoProcessor.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_NAME,
        device_map=device,
        trust_remote_code=True,
        torch_dtype="auto",
    )
    model.eval()

    try:
        layers = model.model.language_model.layers
    except AttributeError as exc:
        raise AttributeError(
            "Could not find layers at model.model.language_model.layers. "
            "This script requires Qwen-style model internals."
        ) from exc

    logger.info("Model loaded: layers=%d hidden=%s", len(layers), getattr(model.config.text_config, "hidden_size", "unknown"))
    logger.info("VRAM allocated: %.1f GB", torch.cuda.memory_allocated() / 1e9)
    return model, processor, layers


def build_condition_specs() -> dict[str, ConditionSpec]:
    return {
        "C0": ConditionSpec("C0", "Base (no abliteration)", "none", "none", False),
        "C1": ConditionSpec("C1", "Raw connectome abliteration", "raw", "all64", False),
        "C2": ConditionSpec("C2", "GS-protected connectome abliteration", "gs", "all64", True),
        "C3": ConditionSpec("C3", "Raw connectome abliteration (selective)", "raw", "top10", False),
        "C4": ConditionSpec("C4", "GS-protected connectome abliteration (selective)", "gs", "top10", True),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="GS-protected abliteration experiment on Qwen3.5-27B")
    parser.add_argument("--output", type=str, default="./gs_abliteration_results", help="Output directory")
    parser.add_argument("--conditions", nargs="+", default=ALL_CONDITIONS, choices=ALL_CONDITIONS, help="Conditions to run")
    parser.add_argument("--max-prompts", type=int, default=None, help="Cap total prompts (smoke test)")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device map string for transformers")
    parser.add_argument("--connectome", type=str, default=CONNECTOME_DEFAULT, help="Path to connectome_zscores.pt")
    parser.add_argument("--huihui-checkpoint", type=str, default=HUIHUI_CHECKPOINT_DEFAULT, help="Path to huihui eval_checkpoint.json")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(output_dir)

    checkpoint_path = output_dir / "eval_checkpoint.json"
    checkpoint: dict[str, Any] = {
        "meta": {
            "created_at": datetime.now().isoformat(),
            "model_name": MODEL_NAME,
            "connectome_path": str(args.connectome),
            "max_prompts": args.max_prompts,
        },
        "results": {},
        "completed_conditions": [],
    }

    if args.resume and checkpoint_path.exists():
        try:
            checkpoint = load_json(checkpoint_path)
            logger.info("Resumed checkpoint: %s", checkpoint_path)
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("Failed to load checkpoint (%s). Starting fresh. Error: %s", checkpoint_path, exc)

    math_tiers, knowledge_prompts, refusal_prompts, prompt_counts = select_eval_prompts(args.max_prompts, logger)

    connectome = load_connectome(Path(args.connectome))
    raw_refusal, gs_refusal, protect_dirs, top10_layers, gs_analysis = prepare_directions(connectome, logger)

    checkpoint["meta"]["updated_at"] = datetime.now().isoformat()
    checkpoint["meta"]["prompt_counts"] = prompt_counts
    checkpoint["top10_refusal_layers"] = top10_layers
    checkpoint["gs_analysis"] = gs_analysis
    checkpoint["protect_category_indices"] = {
        "Math": CAT_MATH,
        "Code": CAT_CODE,
        "Science": CAT_SCIENCE,
        "Analytical": CAT_ANALYTICAL,
        "Refusal": CAT_REFUSAL,
    }
    atomic_save_json(checkpoint, checkpoint_path)

    specs = build_condition_specs()
    selected_conditions = list(dict.fromkeys(args.conditions))
    completed = set(checkpoint.get("completed_conditions", []))
    pending = [cid for cid in selected_conditions if cid not in completed]

    logger.info("Selected conditions: %s", selected_conditions)
    logger.info("Completed conditions in checkpoint: %s", sorted(completed))
    logger.info("Pending conditions: %s", pending)

    if pending:
        model, processor, layers = load_model(args.device, logger)
        try:
            n_layers = len(layers)
            for cid in selected_conditions:
                if cid in completed:
                    logger.info("Skipping %s (already complete)", cid)
                    continue

                spec = specs[cid]
                layer_map = build_condition_layer_map(spec, n_layers, raw_refusal, gs_refusal, top10_layers)

                try:
                    result = run_condition(
                        model=model,
                        processor=processor,
                        layers=layers,
                        condition=spec,
                        layer_map=layer_map,
                        math_tiers=math_tiers,
                        knowledge_prompts=knowledge_prompts,
                        refusal_prompts=refusal_prompts,
                        logger=logger,
                    )
                except (RuntimeError, ValueError, OSError) as exc:
                    logger.exception("Condition %s failed: %s", cid, exc)
                    checkpoint.setdefault("errors", {})[cid] = {
                        "time": datetime.now().isoformat(),
                        "error": str(exc),
                    }
                    atomic_save_json(checkpoint, checkpoint_path)
                    raise

                checkpoint.setdefault("results", {})[cid] = result
                checkpoint.setdefault("completed_conditions", []).append(cid)
                checkpoint["meta"]["updated_at"] = datetime.now().isoformat()
                atomic_save_json(checkpoint, checkpoint_path)

                torch.cuda.empty_cache()
                gc.collect()

        finally:
            del model
            gc.collect()
            torch.cuda.empty_cache()
    else:
        logger.info("No pending conditions. Skipping model load.")

    huihui = load_huihui_baseline(Path(args.huihui_checkpoint), logger)
    condition_results = checkpoint.get("results", {})

    generate_report(
        output_dir=output_dir,
        condition_results=condition_results,
        huihui=huihui,
        top10_layers=top10_layers,
        gs_analysis=gs_analysis,
    )

    logger.info("Saved report: %s", output_dir / "gs_abliteration_report.md")
    logger.info("Saved data: %s", output_dir / "gs_abliteration_data.json")
    logger.info("Done.")


if __name__ == "__main__":
    main()