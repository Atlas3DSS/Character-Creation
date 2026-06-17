#!/usr/bin/env python3
"""Analyze personality sweep activations with holdouts and matched contrasts."""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import LeaveOneGroupOut, StratifiedGroupKFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


B5_DIMS = ["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"]
VIEW_DIRS = {
    "mean": "activations",
    "think": "activations_think",
    "response": "activations_response",
    "early": "activations_early",
    "late": "activations_late",
}


def _shard_id(path: Path) -> str:
    match = re.search(r"mean_shard_(\d+)", path.name)
    if not match:
        raise ValueError(f"Unrecognized shard filename: {path}")
    return match.group(1)


def parse_combo(combo: str) -> dict[str, str]:
    levels = combo.split("_")
    if len(levels) != 5:
        levels = ["M"] * 5
    return {dim: levels[i] for i, dim in enumerate(B5_DIMS)}


def detect_views(sweep_dir: Path) -> list[str]:
    out: list[str] = []
    for view, dirname in VIEW_DIRS.items():
        if (sweep_dir / dirname).exists():
            out.append(view)
    return out


def detect_layers(sweep_dir: Path, views: list[str]) -> list[int]:
    found: set[int] = set()
    for view in views:
        base = sweep_dir / VIEW_DIRS[view]
        if not base.exists():
            continue
        for path in base.glob("L*"):
            if path.is_dir() and path.name[1:].isdigit():
                found.add(int(path.name[1:]))
    return sorted(found)


def load_activation_shards(act_dir: Path, layer: int) -> tuple[np.ndarray, list[dict[str, Any]]]:
    layer_dir = act_dir / f"L{layer:02d}"
    if not layer_dir.exists():
        raise FileNotFoundError(f"Missing layer directory: {layer_dir}")

    shard_files = {_shard_id(p): p for p in layer_dir.glob("mean_shard_*.pt") if "_meta" not in p.name}
    meta_files = {_shard_id(p): p for p in layer_dir.glob("mean_shard_*_meta.jsonl")}
    if not shard_files:
        raise FileNotFoundError(f"No activation shards found in {layer_dir}")
    if set(shard_files) != set(meta_files):
        raise ValueError(f"Mismatched shard/meta files in {layer_dir}")

    tensors: list[torch.Tensor] = []
    metadata: list[dict[str, Any]] = []
    for shard_id in sorted(shard_files):
        tensor = torch.load(shard_files[shard_id], map_location="cpu", weights_only=True).float()
        rows = [json.loads(line) for line in meta_files[shard_id].read_text().splitlines() if line.strip()]
        if tensor.shape[0] != len(rows):
            raise ValueError(
                f"Row mismatch in {layer_dir} shard {shard_id}: tensor={tensor.shape[0]} meta={len(rows)}"
            )
        tensors.append(tensor)
        metadata.extend(rows)
    return torch.cat(tensors, dim=0).numpy(), metadata


def load_responses_by_char(resp_dir: Path) -> dict[int, list[dict[str, Any]]]:
    out: dict[int, list[dict[str, Any]]] = {}
    for path in sorted(resp_dir.glob("char_*.jsonl")):
        try:
            char_id = int(path.stem.split("_")[1])
        except Exception as exc:  # noqa: BLE001
            raise ValueError(f"Bad response filename: {path}") from exc
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        out[char_id] = rows
    return out


def enrich_meta(meta: list[dict[str, Any]], responses_by_char: dict[int, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    counters: Counter[int] = Counter()
    enriched: list[dict[str, Any]] = []
    for row in meta:
        item = dict(row)
        char_id = int(item["char_id"])
        if "prompt_idx" not in item:
            offset = counters[char_id]
            resp_rows = responses_by_char.get(char_id, [])
            if offset >= len(resp_rows):
                raise ValueError(
                    f"Cannot reconstruct prompt_idx for char {char_id}: activation rows exceed response rows"
                )
            resp = resp_rows[offset]
            item["prompt_idx"] = int(resp["prompt_idx"])
            item["prompt_category"] = resp.get("prompt_category", item.get("prompt_category"))
            item["prompt"] = resp.get("prompt")
            item["n_think_tokens"] = int(resp.get("n_think_tokens", 0) or 0)
            item["n_response_tokens"] = int(resp.get("n_response_tokens", 0) or 0)
            item["n_gen_tokens"] = int(resp.get("n_gen_tokens", 0) or 0)
        else:
            item["prompt_idx"] = int(item["prompt_idx"])
        counters[char_id] += 1
        item["b5_levels"] = parse_combo(str(item.get("b5_combo", "M_M_M_M_M")))
        enriched.append(item)
    return enriched


def load_dataset(sweep_dir: Path, layer: int, view: str) -> tuple[np.ndarray, list[dict[str, Any]]]:
    act_dir = sweep_dir / VIEW_DIRS[view]
    acts, meta = load_activation_shards(act_dir, layer)
    responses_by_char = load_responses_by_char(sweep_dir / "responses")
    return acts, enrich_meta(meta, responses_by_char)


def make_classifier() -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    max_iter=2000,
                    solver="lbfgs",
                ),
            ),
        ]
    )


def split_score(
    X: np.ndarray,
    y: np.ndarray,
    splitter,
    groups: np.ndarray | None = None,
) -> dict[str, Any]:
    scores: list[float] = []
    n_folds = 0
    if groups is None:
        splits = splitter.split(X, y)
    else:
        splits = splitter.split(X, y, groups)
    for train_idx, test_idx in splits:
        if len(np.unique(y[train_idx])) < 2 or len(np.unique(y[test_idx])) < 2:
            continue
        model = make_classifier()
        model.fit(X[train_idx], y[train_idx])
        pred = model.predict(X[test_idx])
        scores.append(float(balanced_accuracy_score(y[test_idx], pred)))
        n_folds += 1
    if not scores:
        return {"folds": 0, "mean_balanced_accuracy": None, "std_balanced_accuracy": None}
    return {
        "folds": n_folds,
        "mean_balanced_accuracy": float(np.mean(scores)),
        "std_balanced_accuracy": float(np.std(scores)),
    }


def evaluate_trait(meta: list[dict[str, Any]], X: np.ndarray, trait: str) -> dict[str, Any]:
    return evaluate_trait_config(meta, X, trait)


def evaluate_trait_config(
    meta: list[dict[str, Any]],
    X: np.ndarray,
    trait: str,
    *,
    max_folds: int = 5,
    include_prompt_holdout: bool = True,
    include_category_holdout: bool = True,
) -> dict[str, Any]:
    y = np.array([row["b5_levels"][trait] for row in meta])
    char_groups = np.array([int(row["char_id"]) for row in meta])
    prompt_groups = np.array([int(row["prompt_idx"]) for row in meta])
    cat_groups = np.array([str(row["prompt_category"]) for row in meta])

    counts = Counter(y.tolist())
    min_class = min(counts.values())
    random_cv = StratifiedKFold(n_splits=min(max_folds, min_class), shuffle=True, random_state=42) if min_class >= 2 else None

    char_group_count = len(np.unique(char_groups))
    prompt_group_count = len(np.unique(prompt_groups))

    out = {
        "n_samples": int(len(meta)),
        "class_counts": dict(counts),
        "random_split": {"folds": 0, "mean_balanced_accuracy": None, "std_balanced_accuracy": None},
        "character_holdout": {"folds": 0, "mean_balanced_accuracy": None, "std_balanced_accuracy": None},
        "prompt_holdout": {"folds": 0, "mean_balanced_accuracy": None, "std_balanced_accuracy": None},
        "category_holdout": {"folds": 0, "mean_balanced_accuracy": None, "std_balanced_accuracy": None},
    }

    if random_cv is not None:
        out["random_split"] = split_score(X, y, random_cv)

    if char_group_count >= 3 and min_class >= 2:
        char_cv = StratifiedGroupKFold(n_splits=min(max_folds, char_group_count), shuffle=True, random_state=42)
        out["character_holdout"] = split_score(X, y, char_cv, char_groups)

    if include_prompt_holdout and prompt_group_count >= 3 and min_class >= 2:
        prompt_cv = StratifiedGroupKFold(n_splits=min(max_folds, prompt_group_count), shuffle=True, random_state=42)
        out["prompt_holdout"] = split_score(X, y, prompt_cv, prompt_groups)

    if include_category_holdout and len(np.unique(cat_groups)) >= 2:
        out["category_holdout"] = split_score(X, y, LeaveOneGroupOut(), cat_groups)

    return out


def matched_factorial_direction(meta: list[dict[str, Any]], X: np.ndarray, trait: str) -> dict[str, Any]:
    grouped: dict[tuple[Any, ...], dict[str, int]] = defaultdict(dict)
    others = [dim for dim in B5_DIMS if dim != trait]
    for idx, row in enumerate(meta):
        levels = row["b5_levels"]
        key = (int(row["prompt_idx"]),) + tuple(levels[dim] for dim in others)
        grouped[key][levels[trait]] = idx

    diffs: list[np.ndarray] = []
    for bucket in grouped.values():
        if "H" in bucket and "L" in bucket:
            diffs.append(X[bucket["H"]] - X[bucket["L"]])

    if not diffs:
        return {"n_pairs": 0, "raw_norm": 0.0, "direction": None}

    mean_diff = np.mean(np.stack(diffs, axis=0), axis=0)
    norm = float(np.linalg.norm(mean_diff))
    direction = (mean_diff / max(norm, 1e-12)).tolist()
    return {"n_pairs": len(diffs), "raw_norm": norm, "direction": direction}


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 1e-12:
        return 0.0
    return float(np.dot(a, b) / denom)


def build_direction_cosines(directions: dict[str, dict[str, Any]]) -> dict[str, dict[str, float | None]]:
    out: dict[str, dict[str, float | None]] = {}
    cached: dict[str, np.ndarray] = {}
    for trait, payload in directions.items():
        direction = payload.get("direction")
        if direction is not None:
            cached[trait] = np.array(direction, dtype=np.float32)
    for a in B5_DIMS:
        out[a] = {}
        for b in B5_DIMS:
            if a not in cached or b not in cached:
                out[a][b] = None
            else:
                out[a][b] = cosine_similarity(cached[a], cached[b])
    return out


def write_markdown_report(
    output_path: Path,
    sweep_dir: Path,
    results: dict[str, Any],
) -> None:
    lines = [
        f"# Personality Phase Sweep Analysis",
        "",
        f"- Sweep dir: `{sweep_dir}`",
        f"- Views: `{', '.join(results['views'])}`",
        f"- Layers: `{', '.join(f'L{layer:02d}' for layer in results['layers'])}`",
        "",
    ]

    for view in results["views"]:
        lines.append(f"## View: `{view}`")
        lines.append("")
        for layer in results["layers"]:
            key = f"{view}:L{layer:02d}"
            payload = results["per_view_layer"].get(key)
            if payload is None:
                continue
            lines.append(f"### L{layer:02d}")
            lines.append("")
            lines.append("| Trait | Random | Char Holdout | Prompt Holdout | Category Holdout | Matched Pairs |")
            lines.append("|---|---:|---:|---:|---:|---:|")
            for trait in B5_DIMS:
                ev = payload["decodability"][trait]
                md = payload["matched_directions"][trait]
                lines.append(
                    "| "
                    f"{trait} | "
                    f"{fmt_score(ev['random_split'])} | "
                    f"{fmt_score(ev['character_holdout'])} | "
                    f"{fmt_score(ev['prompt_holdout'])} | "
                    f"{fmt_score(ev['category_holdout'])} | "
                    f"{md['n_pairs']} |"
                )
            lines.append("")
    output_path.write_text("\n".join(lines))


def fmt_score(payload: dict[str, Any]) -> str:
    val = payload.get("mean_balanced_accuracy")
    if val is None:
        return "n/a"
    return f"{val:.3f}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze personality sweep activations")
    parser.add_argument("--sweep-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--layers", type=str, default=None, help="Comma-separated layer list; default=auto-detect")
    parser.add_argument("--views", type=str, default=None, help="Comma-separated activation views; default=auto-detect")
    parser.add_argument("--fast-mode", action="store_true", help="Use fewer folds and skip prompt/category holdouts")
    args = parser.parse_args()

    sweep_dir = Path(args.sweep_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    views = [v.strip() for v in args.views.split(",") if v.strip()] if args.views else detect_views(sweep_dir)
    if not views:
        raise FileNotFoundError(f"No activation views found under {sweep_dir}")
    layers = [int(x) for x in args.layers.split(",")] if args.layers else detect_layers(sweep_dir, views)
    if not layers:
        raise FileNotFoundError(f"No activation layers found under {sweep_dir}")

    results: dict[str, Any] = {
        "sweep_dir": str(sweep_dir.resolve()),
        "views": views,
        "layers": layers,
        "per_view_layer": {},
    }

    max_folds = 3 if args.fast_mode else 5
    include_prompt_holdout = not args.fast_mode
    include_category_holdout = not args.fast_mode

    total_combos = len(views) * len(layers)
    combo_idx = 0
    for view in views:
        for layer in layers:
            combo_idx += 1
            print(f"[ANALYZE] {combo_idx}/{total_combos} view={view} layer=L{layer:02d}", flush=True)
            X, meta = load_dataset(sweep_dir, layer, view)
            key = f"{view}:L{layer:02d}"
            payload = {
                "n_samples": int(X.shape[0]),
                "hidden_dim": int(X.shape[1]),
                "decodability": {},
                "matched_directions": {},
                "direction_cosines": {},
            }
            for trait in B5_DIMS:
                payload["decodability"][trait] = evaluate_trait_config(
                    meta,
                    X,
                    trait,
                    max_folds=max_folds,
                    include_prompt_holdout=include_prompt_holdout,
                    include_category_holdout=include_category_holdout,
                )
                payload["matched_directions"][trait] = matched_factorial_direction(meta, X, trait)
            payload["direction_cosines"] = build_direction_cosines(payload["matched_directions"])
            results["per_view_layer"][key] = payload

    (output_dir / "analysis_results.json").write_text(json.dumps(results, indent=2))
    write_markdown_report(output_dir / "analysis_report.md", sweep_dir, results)
    print(f"[DONE] wrote {output_dir / 'analysis_results.json'}")
    print(f"[DONE] wrote {output_dir / 'analysis_report.md'}")


if __name__ == "__main__":
    main()
