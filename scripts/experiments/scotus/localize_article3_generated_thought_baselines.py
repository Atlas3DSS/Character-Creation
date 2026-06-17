#!/usr/bin/env python3
"""Localize generated Article III baseline thought-state differences.

This is a candidate-nomination screen over thoughts Qwen generated itself.
Unlike ``localize_article3_ambiguous_thought_states.py``, it does not use
inserted private/public scratchpads. It loads a baseline generated-thinking run,
groups prompts by manually reviewed final-holding tendency, and finds
layer/component/region directions separating private-leaning from
public-leaning generated trajectories.

It is not actuator evidence. Any nominated surface must still pass a no-mask
generation gate against random, source, text, and prompt controls.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

from localize_article3_ambiguous_thought_states import (
    capture_condition_state,
    fmt,
    load_model_and_tokenizer,
    markdown_table,
    now_iso,
    parse_csv,
    parse_layers,
    safe_key,
    transformer_layers,
    write_json,
    write_jsonl,
)
from qwen_eval_budget import MIN_COMPLETE_ANSWER_TOKENS, SHORT_BUDGET_CLAIM_WARNING


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MODEL = Path("/home/orwel/dev_genius/models/Qwen3.5-27B")
DEFAULT_SOURCE_GENERATIONS = (
    PROJECT_ROOT / "sweep_v4" / "scotus_thinking_localized_direction_poke_20260502_005241" / "generations.jsonl"
)
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "sweep_v4"
DEFAULT_COMPONENTS = "residual,mixer,mlp"
DEFAULT_REGIONS = "pre_answer_last,thought_mean,thought_tail16_mean,tail32_mean"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON") from exc
    return rows


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected JSON object")
    return data


def int_or_none(value: Any) -> int | None:
    if value is None or value == "":
        return None
    return int(value)


def source_budget_meta(source_generations: Path, *, allow_short: bool) -> dict[str, Any]:
    source_manifest_path = source_generations.parent / "manifest.json"
    source_manifest = read_json(source_manifest_path) if source_manifest_path.exists() else {}
    thought_tokens = int_or_none(source_manifest.get("thought_tokens"))
    answer_tokens = int_or_none(source_manifest.get("answer_tokens") or source_manifest.get("max_new_tokens"))
    budget_fields = {
        "source_manifest": str(source_manifest_path) if source_manifest_path.exists() else None,
        "source_thought_tokens": thought_tokens,
        "source_answer_tokens": answer_tokens,
        "min_complete_answer_tokens": MIN_COMPLETE_ANSWER_TOKENS,
    }
    unknown = source_manifest_path.exists() is False or (thought_tokens is None and answer_tokens is None)
    short = any(
        tokens is not None and tokens < MIN_COMPLETE_ANSWER_TOKENS
        for tokens in (thought_tokens, answer_tokens)
    )
    if (unknown or short) and not allow_short:
        raise ValueError(
            "Source generations do not have complete Qwen budgets. "
            f"source_manifest={source_manifest_path if source_manifest_path.exists() else 'missing'}, "
            f"thought_tokens={thought_tokens}, answer_tokens={answer_tokens}. "
            f"Use >= {MIN_COMPLETE_ANSWER_TOKENS} tokens, preferably 3072-4096, "
            "or pass --allow-short-source-budget for smoke/localization only."
        )
    return {
        **budget_fields,
        "source_unknown_budget": bool(unknown),
        "source_short_budget": bool(short),
        "source_budget_note": (
            SHORT_BUDGET_CLAIM_WARNING
            if unknown or short
            else "Complete source-generation Qwen budget for generated-thought localization."
        ),
    }


def parse_int_set(raw: str) -> set[int]:
    return {int(item.strip()) for item in raw.split(",") if item.strip()}


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 1e-12:
        return 0.0
    return float(np.dot(a, b) / denom)


def site_metrics(private_vecs: list[np.ndarray], public_vecs: list[np.ndarray]) -> dict[str, float | np.ndarray]:
    private = np.stack(private_vecs, axis=0).astype(np.float32, copy=False)
    public = np.stack(public_vecs, axis=0).astype(np.float32, copy=False)
    private_mean = private.mean(axis=0)
    public_mean = public.mean(axis=0)
    direction = (private_mean - public_mean).astype(np.float32, copy=False)
    direction_norm = float(np.linalg.norm(direction))
    if direction_norm <= 1e-12:
        return {
            "mean_direction": direction,
            "direction_norm": direction_norm,
            "projection_margin": 0.0,
            "pooled_projection_sd": 0.0,
            "projection_effect_size": 0.0,
            "private_alignment": 0.0,
            "public_alignment": 0.0,
            "alignment_mean": 0.0,
            "rank_score": 0.0,
        }

    unit = direction / direction_norm
    private_proj = private @ unit
    public_proj = public @ unit
    pooled = np.concatenate(
        [
            private_proj - float(private_proj.mean()),
            public_proj - float(public_proj.mean()),
        ],
        axis=0,
    )
    pooled_sd = float(np.std(pooled, ddof=1)) if pooled.shape[0] > 1 else 0.0
    margin = float(private_proj.mean() - public_proj.mean())
    effect = margin / max(pooled_sd, 1e-6)
    private_alignment = float(np.mean([cosine(vec - public_mean, direction) for vec in private]))
    public_alignment = float(np.mean([cosine(private_mean - vec, direction) for vec in public]))
    alignment_mean = 0.5 * (private_alignment + public_alignment)
    rank_score = max(0.0, effect) * max(0.0, alignment_mean) * math.log1p(direction_norm)
    return {
        "mean_direction": direction,
        "direction_norm": direction_norm,
        "projection_margin": margin,
        "pooled_projection_sd": pooled_sd,
        "projection_effect_size": float(effect),
        "private_alignment": private_alignment,
        "public_alignment": public_alignment,
        "alignment_mean": alignment_mean,
        "rank_score": float(rank_score),
    }


def compute_metrics(
    records: dict[int, dict[tuple[int, str, str], np.ndarray]],
    private_ids: list[int],
    public_ids: list[int],
    *,
    shuffle_controls: int,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[tuple[int, str, str], np.ndarray]]:
    site_keys = sorted(next(iter(records.values())).keys())
    labeled_ids = private_ids + public_ids
    labels = np.array([1] * len(private_ids) + [0] * len(public_ids), dtype=np.int64)
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    directions: dict[tuple[int, str, str], np.ndarray] = {}

    for site in site_keys:
        private_vecs = [records[prompt_id][site] for prompt_id in private_ids]
        public_vecs = [records[prompt_id][site] for prompt_id in public_ids]
        metrics = site_metrics(private_vecs, public_vecs)
        null_scores: list[float] = []
        null_effects: list[float] = []
        for _ in range(shuffle_controls):
            shuffled = rng.permutation(labels)
            pseudo_private = [records[prompt_id][site] for prompt_id, label in zip(labeled_ids, shuffled) if label == 1]
            pseudo_public = [records[prompt_id][site] for prompt_id, label in zip(labeled_ids, shuffled) if label == 0]
            null = site_metrics(pseudo_private, pseudo_public)
            null_scores.append(float(null["rank_score"]))
            null_effects.append(float(null["projection_effect_size"]))

        layer, component, region = site
        null_max = max(null_scores) if null_scores else 0.0
        row = {
            "layer": int(layer),
            "component": str(component),
            "region": str(region),
            "n_private": len(private_ids),
            "n_public": len(public_ids),
            "direction_norm": float(metrics["direction_norm"]),
            "projection_margin": float(metrics["projection_margin"]),
            "pooled_projection_sd": float(metrics["pooled_projection_sd"]),
            "projection_effect_size": float(metrics["projection_effect_size"]),
            "private_alignment": float(metrics["private_alignment"]),
            "public_alignment": float(metrics["public_alignment"]),
            "alignment_mean": float(metrics["alignment_mean"]),
            "rank_score": float(metrics["rank_score"]),
            "shuffle_controls": int(shuffle_controls),
            "shuffle_rank_score_mean": float(np.mean(null_scores)) if null_scores else 0.0,
            "shuffle_rank_score_max": float(null_max),
            "shuffle_effect_size_mean": float(np.mean(null_effects)) if null_effects else 0.0,
            "rank_score_minus_shuffle_max": float(metrics["rank_score"]) - float(null_max),
        }
        rows.append(row)
        directions[site] = np.asarray(metrics["mean_direction"], dtype=np.float32)

    rows.sort(
        key=lambda row: (
            float(row["rank_score_minus_shuffle_max"]),
            float(row["rank_score"]),
            float(row["projection_effect_size"]),
        ),
        reverse=True,
    )
    return rows, directions


def save_top_directions(
    out_dir: Path,
    rows: list[dict[str, Any]],
    directions: dict[tuple[int, str, str], np.ndarray],
    *,
    top_k: int,
) -> list[dict[str, Any]]:
    arrays: dict[str, np.ndarray] = {}
    meta_rows: list[dict[str, Any]] = []
    for idx, row in enumerate(rows[:top_k]):
        site = (int(row["layer"]), str(row["component"]), str(row["region"]))
        raw = directions[site].astype(np.float32, copy=False)
        norm = float(np.linalg.norm(raw))
        unit = raw / max(norm, 1e-12)
        key = f"direction_{idx:03d}_{safe_key(str(row['component']))}_L{int(row['layer']):02d}_{safe_key(str(row['region']))}"
        arrays[key] = unit.astype(np.float32, copy=False)
        arrays[f"{key}_raw_mean_delta"] = raw
        meta = {
            **row,
            "direction_key": key,
            "direction_semantics": "generated_baseline_private_minus_public_visible_thought_state",
        }
        meta_rows.append(meta)
    np.savez_compressed(out_dir / "top_directions.npz", **arrays)
    write_jsonl(out_dir / "direction_meta.jsonl", meta_rows)
    return meta_rows


def write_report(
    path: Path,
    *,
    manifest: dict[str, Any],
    metrics: list[dict[str, Any]],
    direction_meta: list[dict[str, Any]],
    label_rows: list[dict[str, Any]],
    token_rows: list[dict[str, Any]],
) -> None:
    by_component: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in metrics:
        by_component[str(row["component"])].append(row)

    lines = [
        "# Article III Generated-Thought Baseline Localization",
        "",
        "## Purpose",
        "",
        (
            "Localize where actual Qwen-generated Article III thoughts differ between "
            "manual private-leaning and public-leaning baseline prompts. This is not "
            "actuator evidence; it nominates generated-trajectory surfaces for later no-mask tests."
        ),
        "",
        "## Config",
        "",
        markdown_table(
            ["Field", "Value"],
            [
                ["Started", manifest["started_at"]],
                ["Finished", manifest["finished_at"]],
                ["Model", manifest["model_path"]],
                ["Source generations", manifest["source_generations"]],
                ["Source thought tokens", manifest.get("source_thought_tokens")],
                ["Source answer tokens", manifest.get("source_answer_tokens")],
                ["Source budget note", manifest.get("source_budget_note")],
                ["Private prompt ids", ",".join(str(item) for item in manifest["private_prompt_ids"])],
                ["Public prompt ids", ",".join(str(item) for item in manifest["public_prompt_ids"])],
                ["Layers", manifest["layers"]],
                ["Components", ", ".join(manifest["components"])],
                ["Regions", ", ".join(manifest["regions"])],
                ["Shuffle controls", manifest["shuffle_controls"]],
                ["Output dir", manifest["output_dir"]],
            ],
        ),
        "",
        "## Manual Labels Used",
        "",
        markdown_table(
            ["Prompt id", "Prompt key", "Manual baseline label", "Thinking tokens", "Closed"],
            [
                [
                    row["prompt_id"],
                    row["prompt_key"],
                    row["manual_label"],
                    row.get("thinking_generated_tokens", ""),
                    row.get("model_closed_thinking", ""),
                ]
                for row in label_rows
            ],
        ),
        "",
        "## Top Sites",
        "",
        markdown_table(
            [
                "Rank",
                "Layer",
                "Component",
                "Region",
                "Score-null",
                "Score",
                "Null max",
                "Effect",
                "Align",
                "Norm",
            ],
            [
                [
                    idx + 1,
                    row["layer"],
                    row["component"],
                    row["region"],
                    fmt(row["rank_score_minus_shuffle_max"]),
                    fmt(row["rank_score"]),
                    fmt(row["shuffle_rank_score_max"]),
                    fmt(row["projection_effect_size"]),
                    fmt(row["alignment_mean"]),
                    fmt(row["direction_norm"]),
                ]
                for idx, row in enumerate(metrics[:20])
            ],
        ),
        "",
    ]
    for component in sorted(by_component):
        lines.extend(
            [
                f"## Top {component} Sites",
                "",
                markdown_table(
                    ["Rank", "Layer", "Region", "Score-null", "Effect", "Align"],
                    [
                        [
                            idx + 1,
                            row["layer"],
                            row["region"],
                            fmt(row["rank_score_minus_shuffle_max"]),
                            fmt(row["projection_effect_size"]),
                            fmt(row["alignment_mean"]),
                        ]
                        for idx, row in enumerate(by_component[component][:8])
                    ],
                ),
                "",
            ]
        )
    lines.extend(
        [
            "## Tokenization",
            "",
            markdown_table(
                ["Prompt", "Label", "Seq len", "Thought tokens", "Thought span"],
                [
                    [
                        row["prompt_key"],
                        row["manual_label"],
                        row["seq_len"],
                        row["thought_tokens"],
                        f"{row['thought_start']}:{row['thought_end']}",
                    ]
                    for row in token_rows
                ],
            ),
            "",
            "## Interpretation",
            "",
            "- These directions are generated-baseline private-minus-public state differences, not inserted-thought deltas.",
            "- They are still confounded by prompt facts and baseline holdings; shuffle controls only test label assignment, not doctrinal causality.",
            "- Generated-baseline localization should be built from complete Qwen generations; short source budgets make the run smoke/localization only.",
            "- A direction should only advance if its top sites differ materially from the failed inserted-thought tail cluster and then pass a no-mask generation gate.",
            "",
            "## Artifacts",
            "",
            f"- Manifest: `{manifest['output_dir']}/manifest.json`.",
            f"- Site metrics: `{manifest['output_dir']}/site_metrics.jsonl`.",
            f"- Top directions: `{manifest['output_dir']}/top_directions.npz`.",
            f"- Direction metadata: `{manifest['output_dir']}/direction_meta.jsonl`.",
        ]
    )
    if direction_meta:
        lines.extend(
            [
                "",
                "Top saved direction:",
                "",
                "```json",
                json.dumps(direction_meta[0], indent=2, sort_keys=True),
                "```",
            ]
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--source-generations", type=Path, default=DEFAULT_SOURCE_GENERATIONS)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--private-prompt-ids", default="0,1,5")
    parser.add_argument("--public-prompt-ids", default="2,3,4,6,7")
    parser.add_argument("--layers", default="all")
    parser.add_argument("--components", default=DEFAULT_COMPONENTS)
    parser.add_argument("--regions", default=DEFAULT_REGIONS)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--shuffle-controls", type=int, default=128)
    parser.add_argument("--top-k", type=int, default=64)
    parser.add_argument("--device-map", default="single")
    parser.add_argument("--seed", type=int, default=20260502)
    parser.add_argument(
        "--allow-short-source-budget",
        action="store_true",
        help=(
            f"Permit source generations with missing or <{MIN_COMPLETE_ANSWER_TOKENS}-token Qwen budgets. "
            "This makes the localization smoke/debug only."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = now_iso()
    out_dir = args.output_root / f"scotus_article3_generated_thought_baseline_localization_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)

    private_ids = sorted(parse_int_set(args.private_prompt_ids))
    public_ids = sorted(parse_int_set(args.public_prompt_ids))
    overlap = sorted(set(private_ids) & set(public_ids))
    if overlap:
        raise ValueError(f"Prompt ids cannot be both private and public: {overlap}")

    source_rows = [
        row
        for row in read_jsonl(args.source_generations)
        if str(row.get("condition")) == "base" and int(row.get("prompt_id", -1)) in set(private_ids + public_ids)
    ]
    source_budget = source_budget_meta(args.source_generations, allow_short=args.allow_short_source_budget)
    by_id = {int(row["prompt_id"]): row for row in source_rows}
    missing = sorted(set(private_ids + public_ids) - set(by_id))
    if missing:
        raise ValueError(f"Missing source generation rows for prompt ids: {missing}")
    for prompt_id, row in sorted(by_id.items()):
        if not str(row.get("thinking", "")).strip():
            raise ValueError(f"Source generation has empty thinking for prompt id {prompt_id}")

    components = parse_csv(args.components)
    regions = parse_csv(args.regions)
    unknown_components = sorted(set(components) - {"residual", "mixer", "mlp"})
    unknown_regions = sorted(set(regions) - {"pre_answer_last", "thought_mean", "thought_tail16_mean", "tail32_mean"})
    if unknown_components:
        raise ValueError(f"Unknown components: {unknown_components}")
    if unknown_regions:
        raise ValueError(f"Unknown regions: {unknown_regions}")

    tokenizer, model = load_model_and_tokenizer(args.model_path, args.device_map)
    layers_mod = transformer_layers(model)
    layers = parse_layers(args.layers, len(layers_mod))
    print(
        f"Capturing {len(by_id)} generated thoughts; layers={len(layers)} "
        f"components={components} regions={regions}",
        flush=True,
    )

    records: dict[int, dict[tuple[int, str, str], np.ndarray]] = {}
    token_rows: list[dict[str, Any]] = []
    label_rows: list[dict[str, Any]] = []
    for prompt_id in tqdm(sorted(by_id), desc="generated thoughts", unit="prompt"):
        row = by_id[prompt_id]
        label = "private" if prompt_id in private_ids else "public"
        captured, token_meta = capture_condition_state(
            model=model,
            tokenizer=tokenizer,
            layers_mod=layers_mod,
            prompt=str(row["prompt"]),
            thought=str(row["thinking"]),
            layers=layers,
            components=components,
            regions=regions,
            max_length=args.max_length,
        )
        records[prompt_id] = captured
        label_row = {
            "prompt_id": prompt_id,
            "prompt_key": str(row["prompt_key"]),
            "manual_label": label,
            "thinking_generated_tokens": row.get("thinking_generated_tokens"),
            "model_closed_thinking": row.get("model_closed_thinking"),
        }
        label_rows.append(label_row)
        token_rows.append({**label_row, **token_meta})

    metrics, directions = compute_metrics(
        records,
        private_ids,
        public_ids,
        shuffle_controls=args.shuffle_controls,
        seed=args.seed,
    )
    direction_meta = save_top_directions(out_dir, metrics, directions, top_k=args.top_k)
    manifest = {
        "started_at": started,
        "finished_at": now_iso(),
        "model_path": str(args.model_path),
        "source_generations": str(args.source_generations),
        "output_dir": str(out_dir),
        "private_prompt_ids": private_ids,
        "public_prompt_ids": public_ids,
        "labeled_prompt_count": len(private_ids) + len(public_ids),
        "manual_label_note": "Derived from manual baseline prompt selection documented in reports/scotus_qwen35_ambiguous_baseline_prompt_selection_20260502.md.",
        "layers": "all" if len(layers) == len(layers_mod) else ",".join(str(layer) for layer in layers),
        "n_layers": len(layers),
        "components": components,
        "regions": regions,
        "max_length": args.max_length,
        "shuffle_controls": args.shuffle_controls,
        "top_k": args.top_k,
        "seed": args.seed,
        **source_budget,
        "method": "generated_baseline_private_minus_public_visible_thought_state_localization",
        "not_actuator_evidence": True,
    }
    write_json(out_dir / "manifest.json", manifest)
    write_jsonl(out_dir / "label_rows.jsonl", label_rows)
    write_jsonl(out_dir / "token_spans.jsonl", token_rows)
    write_jsonl(out_dir / "site_metrics.jsonl", metrics)
    write_report(
        out_dir / "report.md",
        manifest=manifest,
        metrics=metrics,
        direction_meta=direction_meta,
        label_rows=label_rows,
        token_rows=token_rows,
    )

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"Wrote {out_dir}", flush=True)
    if metrics:
        top = metrics[0]
        print(
            "Top site: "
            f"L{top['layer']} {top['component']} {top['region']} "
            f"score-null={top['rank_score_minus_shuffle_max']:.4f} "
            f"effect={top['projection_effect_size']:.4f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
