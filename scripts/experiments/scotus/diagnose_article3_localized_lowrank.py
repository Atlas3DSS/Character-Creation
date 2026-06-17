#!/usr/bin/env python3
"""Offline diagnostic for conditional controllers over Article III localized sites.

This script does not steer generation. It asks whether the late localized
private/public thought-state surface contains prompt-conditioned structure
beyond a mean private-minus-public delta. If leave-one-prompt-out conditional
maps do not beat mean and permutation controls, a live ReFT/low-rank hook run is
unlikely to be worth the long Qwen generation cost.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from statistics import mean, median
from typing import Any

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from localize_article3_ambiguous_thought_states import (  # noqa: E402
    CONDITIONS,
    THOUGHTS,
    capture_condition_state,
    load_model_and_tokenizer,
    parse_csv,
    read_prompt_specs,
    transformer_layers,
    write_json,
    write_jsonl,
)
from poke_scotus_thinking_localized_directions import DEFAULT_LOCALIZATION_RUN, read_jsonl  # noqa: E402
from poke_scotus_sae_layers import DEFAULT_OUTPUT_ROOT, now_iso  # noqa: E402


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MODEL = Path("/home/orwel/dev_genius/models/Qwen3.5-27B")
DEFAULT_PROMPT_BANK = PROJECT_ROOT / "data" / "scotus" / "scotus_article3_ambiguous_poke_prompts_v1.jsonl"


@dataclass(frozen=True)
class Site:
    layer: int
    component: str
    region: str
    direction_key: str
    score_minus_null: float

    @property
    def key(self) -> tuple[int, str, str]:
        return (self.layer, self.component, self.region)

    @property
    def label(self) -> str:
        return f"L{self.layer:02d}_{self.component}_{self.region}"


def safe_name(text: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in text).strip("_")


def parse_ints(text: str) -> list[int]:
    return [int(item.strip()) for item in text.split(",") if item.strip()]


def parse_floats(text: str) -> list[float]:
    return [float(item.strip()) for item in text.split(",") if item.strip()]


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 1e-12:
        return 0.0
    return float(np.dot(a, b) / denom)


def row_norms(values: np.ndarray) -> np.ndarray:
    return np.maximum(np.linalg.norm(values, axis=1), 1e-12)


def cosine_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.sum(a * b, axis=1) / (row_norms(a) * row_norms(b))


def load_sites(localization_run: Path, *, top_sites: int, components: set[str] | None) -> list[Site]:
    rows = read_jsonl(localization_run / "direction_meta.jsonl")
    sites: list[Site] = []
    for row in rows:
        component = str(row["component"])
        if components is not None and component not in components:
            continue
        sites.append(
            Site(
                layer=int(row["layer"]),
                component=component,
                region=str(row["region"]),
                direction_key=str(row["direction_key"]),
                score_minus_null=float(row["rank_score_minus_shuffle_max"]),
            )
        )
        if len(sites) >= top_sites:
            break
    if not sites:
        raise RuntimeError(f"No sites selected from {localization_run / 'direction_meta.jsonl'}")
    return sites


def fit_lowrank_predict(
    *,
    train_x: np.ndarray,
    train_delta: np.ndarray,
    test_x: np.ndarray,
    rank: int,
    ridge: float,
) -> np.ndarray:
    x_mean = train_x.mean(axis=0).astype(np.float32)
    delta_mean = train_delta.mean(axis=0).astype(np.float32)
    if rank == 0 or train_x.shape[0] < 2:
        return delta_mean.astype(np.float32)

    x_centered = (train_x - x_mean).astype(np.float32)
    d_centered = (train_delta - delta_mean).astype(np.float32)
    max_rank = min(rank, d_centered.shape[0], d_centered.shape[1])
    if max_rank <= 0:
        return delta_mean.astype(np.float32)
    _u, _s, vt = np.linalg.svd(d_centered.astype(np.float64, copy=False), full_matrices=False)
    components = vt[:max_rank].astype(np.float32, copy=False)
    coeffs = (d_centered @ components.T).astype(np.float32, copy=False)
    kernel = x_centered @ x_centered.T
    reg = float(ridge) * np.eye(kernel.shape[0], dtype=np.float32)
    dual = np.linalg.solve(kernel + reg, coeffs).astype(np.float32)
    coeff = ((test_x.astype(np.float32) - x_mean) @ x_centered.T) @ dual
    return (delta_mean + coeff @ components).astype(np.float32)


def nearest_neighbor_predict(train_x: np.ndarray, train_delta: np.ndarray, test_x: np.ndarray) -> np.ndarray:
    sims = cosine_rows(train_x, np.repeat(test_x[None, :], train_x.shape[0], axis=0))
    return train_delta[int(np.argmax(sims))].astype(np.float32)


def evaluate_predictions(
    *,
    site: Site,
    model_name: str,
    predictions: list[np.ndarray],
    true_deltas: np.ndarray,
    mean_baseline: np.ndarray,
    zero_baseline: np.ndarray,
    null_score: bool,
) -> dict[str, Any]:
    pred = np.stack(predictions, axis=0).astype(np.float32)
    pred_mse = float(np.mean((pred - true_deltas) ** 2))
    mean_mse = float(np.mean((mean_baseline - true_deltas) ** 2))
    zero_mse = float(np.mean((zero_baseline - true_deltas) ** 2))
    cos = cosine_rows(pred, true_deltas)
    mean_cos = cosine_rows(mean_baseline, true_deltas)
    pred_norm = row_norms(pred)
    true_norm = row_norms(true_deltas)
    return {
        "site": site.label,
        "layer": site.layer,
        "component": site.component,
        "region": site.region,
        "direction_key": site.direction_key,
        "site_score_minus_shuffle_max": site.score_minus_null,
        "model": model_name,
        "null_score": bool(null_score),
        "n_prompts": int(true_deltas.shape[0]),
        "prediction_mse": pred_mse,
        "mean_baseline_mse": mean_mse,
        "zero_baseline_mse": zero_mse,
        "mse_improvement_vs_mean": 0.0 if mean_mse <= 1e-12 else float((mean_mse - pred_mse) / mean_mse),
        "mse_improvement_vs_zero": 0.0 if zero_mse <= 1e-12 else float((zero_mse - pred_mse) / zero_mse),
        "mean_delta_cosine": float(mean(cos.tolist())),
        "median_delta_cosine": float(median(cos.tolist())),
        "mean_baseline_cosine": float(mean(mean_cos.tolist())),
        "cosine_minus_mean_baseline": float(mean((cos - mean_cos).tolist())),
        "mean_pred_norm": float(mean(pred_norm.tolist())),
        "mean_true_norm": float(mean(true_norm.tolist())),
        "norm_ratio_pred_to_true": float(mean((pred_norm / true_norm).tolist())),
    }


def aggregate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["model"]), []).append(row)
    output: list[dict[str, Any]] = []
    for model_name, group in sorted(grouped.items()):
        non_null = [row for row in group if not row["null_score"]]
        null = [row for row in group if row["null_score"]]
        basis = non_null or group
        output.append(
            {
                "model": model_name,
                "n_sites": len(basis),
                "null_rows": len(null),
                "mean_mse_improvement_vs_mean": float(mean([float(row["mse_improvement_vs_mean"]) for row in basis])),
                "mean_mse_improvement_vs_zero": float(mean([float(row["mse_improvement_vs_zero"]) for row in basis])),
                "mean_delta_cosine": float(mean([float(row["mean_delta_cosine"]) for row in basis])),
                "mean_cosine_minus_mean_baseline": float(
                    mean([float(row["cosine_minus_mean_baseline"]) for row in basis])
                ),
                "mean_norm_ratio_pred_to_true": float(
                    mean([float(row["norm_ratio_pred_to_true"]) for row in basis])
                ),
                "null_mean_mse_improvement_vs_mean": (
                    float(mean([float(row["mse_improvement_vs_mean"]) for row in null])) if null else None
                ),
                "null_max_mse_improvement_vs_mean": (
                    float(max(float(row["mse_improvement_vs_mean"]) for row in null)) if null else None
                ),
                "null_mean_delta_cosine": (
                    float(mean([float(row["mean_delta_cosine"]) for row in null])) if null else None
                ),
                "null_max_delta_cosine": (
                    float(max(float(row["mean_delta_cosine"]) for row in null)) if null else None
                ),
            }
        )
    return output


def markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(item) for item in row) + " |")
    return "\n".join(lines)


def fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def write_report(
    path: Path,
    *,
    manifest: dict[str, Any],
    rows: list[dict[str, Any]],
    summary: list[dict[str, Any]],
) -> None:
    top_rows = sorted(
        [row for row in rows if not row["null_score"]],
        key=lambda row: (
            float(row["mse_improvement_vs_mean"]),
            float(row["cosine_minus_mean_baseline"]),
        ),
        reverse=True,
    )[:24]
    lines = [
        "# Article III Localized Conditional Controller Offline Diagnostic",
        "",
        "## Purpose",
        "",
        "Test whether localized late Article III sites contain prompt-conditioned private-minus-public delta structure beyond a mean direction. This is offline evidence only, not actuator evidence.",
        "",
        "## Config",
        "",
        markdown_table(
            ["Field", "Value"],
            [
                ["Model", manifest["model_path"]],
                ["Prompt bank", manifest["prompt_bank"]],
                ["Localization run", manifest["localization_run"]],
                ["Prompts", manifest["n_prompts"]],
                ["Sites", manifest["n_sites"]],
                ["Ranks", ", ".join(str(item) for item in manifest["ranks"])],
                ["Ridges", ", ".join(str(item) for item in manifest["ridges"])],
                ["Permutation controls", manifest["permutation_controls"]],
                ["Output dir", manifest["output_dir"]],
            ],
        ),
        "",
        "## Aggregate",
        "",
        markdown_table(
            [
                "Model",
                "Sites",
                "MSE vs mean",
                "MSE vs zero",
                "Cosine",
                "Cos minus mean",
                "Null max MSE vs mean",
                "Null max cosine",
            ],
            [
                [
                    row["model"],
                    row["n_sites"],
                    fmt(row["mean_mse_improvement_vs_mean"]),
                    fmt(row["mean_mse_improvement_vs_zero"]),
                    fmt(row["mean_delta_cosine"]),
                    fmt(row["mean_cosine_minus_mean_baseline"]),
                    fmt(row["null_max_mse_improvement_vs_mean"]),
                    fmt(row["null_max_delta_cosine"]),
                ]
                for row in summary
            ],
        ),
        "",
        "## Top Non-Null Site Rows",
        "",
        markdown_table(
            ["Site", "Model", "MSE vs mean", "Cosine", "Cos minus mean", "Pred/true norm"],
            [
                [
                    row["site"],
                    row["model"],
                    fmt(float(row["mse_improvement_vs_mean"])),
                    fmt(float(row["mean_delta_cosine"])),
                    fmt(float(row["cosine_minus_mean_baseline"])),
                    fmt(float(row["norm_ratio_pred_to_true"])),
                ]
                for row in top_rows
            ],
        ),
        "",
        "## Read",
        "",
        "- A positive result here would only justify a live no-mask hook run; it would not prove steering.",
        "- A weak result means the localized surface is mostly a stable inserted-text tail delta, not a prompt-conditioned controller target.",
        "- The live actuator standard still requires generated visible reasoning and final holdings to beat random/source/text controls.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--prompt-bank", type=Path, default=DEFAULT_PROMPT_BANK)
    parser.add_argument("--localization-run", type=Path, default=DEFAULT_LOCALIZATION_RUN)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--top-sites", type=int, default=4)
    parser.add_argument("--components", default="residual,mlp")
    parser.add_argument("--max-prompts", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--ranks", default="0,1,2")
    parser.add_argument("--ridges", default="0.1")
    parser.add_argument("--permutation-controls", type=int, default=4)
    parser.add_argument("--device-map", default="single")
    parser.add_argument("--seed", type=int, default=20260502)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = now_iso()
    out_dir = args.output_root / f"scotus_article3_localized_conditional_diag_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)

    components_filter = set(parse_csv(args.components)) if args.components.strip() else None
    sites = load_sites(args.localization_run, top_sites=args.top_sites, components=components_filter)
    prompt_specs = read_prompt_specs(args.prompt_bank)[: args.max_prompts]
    ranks = parse_ints(args.ranks)
    ridges = parse_floats(args.ridges)
    rng = np.random.default_rng(args.seed)

    layers = sorted({site.layer for site in sites})
    components = sorted({site.component for site in sites})
    regions = sorted({site.region for site in sites})
    tokenizer, model = load_model_and_tokenizer(args.model_path, args.device_map)
    layers_mod = transformer_layers(model)
    print(
        f"Capturing {len(prompt_specs)} prompts x {len(CONDITIONS)} conditions; "
        f"sites={len(sites)} layers={layers} components={components} regions={regions}",
        flush=True,
    )

    records: dict[tuple[int, str], dict[tuple[int, str, str], np.ndarray]] = {}
    token_rows: list[dict[str, Any]] = []
    for spec in prompt_specs:
        for condition in CONDITIONS:
            captured, token_meta = capture_condition_state(
                model=model,
                tokenizer=tokenizer,
                layers_mod=layers_mod,
                prompt=spec.prompt,
                thought=THOUGHTS[condition],
                layers=layers,
                components=components,
                regions=regions,
                max_length=args.max_length,
            )
            records[(spec.prompt_id, condition)] = captured
            token_rows.append(
                {
                    "prompt_id": spec.prompt_id,
                    "prompt_key": spec.prompt_key,
                    "condition": condition,
                    **token_meta,
                }
            )

    rows: list[dict[str, Any]] = []
    prompt_ids = [spec.prompt_id for spec in prompt_specs]
    for site in sites:
        print(f"Evaluating {site.label}", flush=True)
        x = np.stack([records[(prompt_id, "neutral")][site.key] for prompt_id in prompt_ids]).astype(np.float32)
        true_delta = np.stack(
            [
                records[(prompt_id, "private_rights")][site.key]
                - records[(prompt_id, "public_rights")][site.key]
                for prompt_id in prompt_ids
            ]
        ).astype(np.float32)
        zero_baseline = np.zeros_like(true_delta)
        loo_mean_preds: list[np.ndarray] = []
        loo_nn_preds: list[np.ndarray] = []
        lowrank_preds: dict[tuple[int, float], list[np.ndarray]] = {
            (rank, ridge): [] for rank in ranks for ridge in ridges
        }
        null_preds: dict[tuple[int, float], list[list[np.ndarray]]] = {
            (rank, ridge): [[] for _idx in range(args.permutation_controls)]
            for rank in ranks
            for ridge in ridges
        }
        for test_idx in range(len(prompt_ids)):
            train_idx = [idx for idx in range(len(prompt_ids)) if idx != test_idx]
            train_x = x[train_idx]
            train_delta = true_delta[train_idx]
            test_x = x[test_idx]
            loo_mean_preds.append(train_delta.mean(axis=0).astype(np.float32))
            loo_nn_preds.append(nearest_neighbor_predict(train_x, train_delta, test_x))
            for rank in ranks:
                for ridge in ridges:
                    lowrank_preds[(rank, ridge)].append(
                        fit_lowrank_predict(
                            train_x=train_x,
                            train_delta=train_delta,
                            test_x=test_x,
                            rank=rank,
                            ridge=ridge,
                        )
                    )
                    if rank == 0:
                        continue
                    for null_idx in range(args.permutation_controls):
                        shuffled = train_delta[rng.permutation(train_delta.shape[0])]
                        null_preds[(rank, ridge)][null_idx].append(
                            fit_lowrank_predict(
                                train_x=train_x,
                                train_delta=shuffled,
                                test_x=test_x,
                                rank=rank,
                                ridge=ridge,
                            )
                        )

        mean_baseline = np.stack(loo_mean_preds, axis=0).astype(np.float32)
        rows.append(
            evaluate_predictions(
                site=site,
                model_name="loo_mean_delta",
                predictions=loo_mean_preds,
                true_deltas=true_delta,
                mean_baseline=mean_baseline,
                zero_baseline=zero_baseline,
                null_score=False,
            )
        )
        rows.append(
            evaluate_predictions(
                site=site,
                model_name="nearest_neutral_delta",
                predictions=loo_nn_preds,
                true_deltas=true_delta,
                mean_baseline=mean_baseline,
                zero_baseline=zero_baseline,
                null_score=False,
            )
        )
        for rank in ranks:
            for ridge in ridges:
                model_name = f"krr_rank{rank}_ridge{ridge:g}"
                rows.append(
                    evaluate_predictions(
                        site=site,
                        model_name=model_name,
                        predictions=lowrank_preds[(rank, ridge)],
                        true_deltas=true_delta,
                        mean_baseline=mean_baseline,
                        zero_baseline=zero_baseline,
                        null_score=False,
                    )
                )
                for null_idx, preds in enumerate(null_preds[(rank, ridge)]):
                    if not preds:
                        continue
                    rows.append(
                        evaluate_predictions(
                            site=site,
                            model_name=model_name,
                            predictions=preds,
                            true_deltas=true_delta,
                            mean_baseline=mean_baseline,
                            zero_baseline=zero_baseline,
                            null_score=True,
                        )
                        | {"null_index": null_idx}
                    )

    summary = aggregate_rows(rows)
    manifest = {
        "started_at": started,
        "finished_at": now_iso(),
        "model_path": str(args.model_path),
        "prompt_bank": str(args.prompt_bank),
        "localization_run": str(args.localization_run),
        "output_dir": str(out_dir),
        "n_prompts": len(prompt_specs),
        "prompt_keys": [spec.prompt_key for spec in prompt_specs],
        "n_sites": len(sites),
        "sites": [site.label for site in sites],
        "layers": layers,
        "components": components,
        "regions": regions,
        "ranks": ranks,
        "ridges": ridges,
        "permutation_controls": int(args.permutation_controls),
        "seed": int(args.seed),
        "method": "leave_one_prompt_out_neutral_state_to_private_minus_public_delta_diagnostic",
        "not_actuator_evidence": True,
    }
    write_json(out_dir / "manifest.json", manifest)
    write_jsonl(out_dir / "diagnostic_rows.jsonl", rows)
    write_jsonl(out_dir / "summary.jsonl", summary)
    write_jsonl(out_dir / "token_spans.jsonl", token_rows)
    write_report(out_dir / "report.md", manifest=manifest, rows=rows, summary=summary)

    del model, tokenizer, layers_mod
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(f"Wrote {out_dir / 'report.md'}", flush=True)


if __name__ == "__main__":
    main()
