#!/usr/bin/env python3
"""Localize local Article III counterfactual answer-state differences.

This nominates a new surface from local Qwen-generated answer trajectories
under inserted private/public visible thoughts. It is not actuator evidence.
The purpose is to avoid widening already failed baseline-thought sites and ask
whether the private/public distinction appears in answer-token trajectories.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from localize_article3_ambiguous_thought_states import (  # noqa: E402
    cosine,
    fmt,
    load_model_and_tokenizer,
    markdown_table,
    now_iso,
    output_tensor,
    parse_csv,
    parse_layers,
    safe_key,
    select_component_module,
    transformer_layers,
    write_json,
    write_jsonl,
)
from localize_article3_generated_thought_baselines import source_budget_meta  # noqa: E402
from poke_scotus_sae_layers import first_parameter_device  # noqa: E402
from run_scotus_thinking_smoke import format_chat  # noqa: E402


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MODEL = Path("/home/orwel/dev_genius/models/Qwen3.5-27B")
DEFAULT_SOURCE_GENERATIONS = (
    PROJECT_ROOT / "sweep_v4" / "scotus_counterfactual_thoughts_20260502_051531" / "generations.jsonl"
)
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "sweep_v4"
DEFAULT_COMPONENTS = "residual,mixer,mlp"
DEFAULT_REGIONS = "pre_answer_last,answer_mean,answer_first64_mean,answer_tail64_mean,tail32_mean"


@dataclass(frozen=True)
class AnswerSpans:
    answer_start: int
    answer_end: int
    seq_len: int


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


def parse_int_list(raw: str) -> list[int]:
    return [int(item.strip()) for item in raw.split(",") if item.strip()]


def render_counterfactual_answer(tokenizer: Any, *, prompt: str, thought: str, answer: str) -> tuple[str, AnswerSpans]:
    chat = format_chat(tokenizer, prompt, enable_thinking=True)
    before_answer = f"{chat}{thought.strip()}\n</think>\n\n"
    rendered = f"{before_answer}{answer.strip()}"
    before_answer_ids = tokenizer(before_answer, add_special_tokens=False).input_ids
    full_ids = tokenizer(rendered, add_special_tokens=False).input_ids
    answer_start = min(len(before_answer_ids), max(0, len(full_ids) - 1))
    answer_end = max(answer_start + 1, len(full_ids))
    return rendered, AnswerSpans(answer_start=answer_start, answer_end=answer_end, seq_len=len(full_ids))


def vector_for_region(hidden_2d: torch.Tensor, spans: AnswerSpans, region: str) -> torch.Tensor:
    seq_len = hidden_2d.shape[0]
    answer_start = max(0, min(spans.answer_start, seq_len - 1))
    answer_end = max(answer_start + 1, min(spans.answer_end, seq_len))
    if region == "pre_answer_last":
        return hidden_2d[max(0, answer_start - 1), :]
    if region == "answer_mean":
        return hidden_2d[answer_start:answer_end, :].mean(dim=0)
    if region == "answer_first64_mean":
        return hidden_2d[answer_start : min(answer_end, answer_start + 64), :].mean(dim=0)
    if region == "answer_tail64_mean":
        return hidden_2d[max(answer_start, answer_end - 64) : answer_end, :].mean(dim=0)
    if region == "tail32_mean":
        return hidden_2d[max(0, seq_len - 32) : seq_len, :].mean(dim=0)
    raise ValueError(f"Unknown region: {region}")


def install_capture_hooks(
    *,
    layers_mod: Any,
    layers: list[int],
    components: list[str],
    regions: list[str],
    spans: AnswerSpans,
    captured: dict[tuple[int, str, str], np.ndarray],
) -> list[Any]:
    handles: list[Any] = []
    for layer in layers:
        for component in components:
            module = select_component_module(layers_mod[layer], component)

            def make_hook(layer_idx: int, component_name: str) -> Any:
                def hook(_module: torch.nn.Module, _inp: tuple[Any, ...], out: Any) -> Any:
                    hidden = output_tensor(out)[0]
                    for region in regions:
                        vec = vector_for_region(hidden, spans, region)
                        captured[(layer_idx, component_name, region)] = (
                            vec.detach().float().cpu().numpy().astype(np.float32, copy=False)
                        )
                    return out

                return hook

            handles.append(module.register_forward_hook(make_hook(layer, component)))
    return handles


@torch.inference_mode()
def capture_answer_state(
    *,
    model: torch.nn.Module,
    tokenizer: Any,
    layers_mod: Any,
    row: dict[str, Any],
    layers: list[int],
    components: list[str],
    regions: list[str],
    max_length: int,
) -> tuple[dict[tuple[int, str, str], np.ndarray], dict[str, int]]:
    rendered, spans = render_counterfactual_answer(
        tokenizer,
        prompt=str(row["prompt"]),
        thought=str(row["thinking"]),
        answer=str(row["answer"]),
    )
    encoded = tokenizer(
        rendered,
        return_tensors="pt",
        add_special_tokens=False,
        truncation=True,
        max_length=max_length,
    )
    seq_len = int(encoded["input_ids"].shape[1])
    if seq_len < spans.seq_len:
        raise ValueError(f"Rendered row truncated from {spans.seq_len} to {seq_len}; increase --max-length")
    input_device = first_parameter_device(model)
    inputs = {key: value.to(input_device) for key, value in encoded.items()}
    captured: dict[tuple[int, str, str], np.ndarray] = {}
    handles = install_capture_hooks(
        layers_mod=layers_mod,
        layers=layers,
        components=components,
        regions=regions,
        spans=spans,
        captured=captured,
    )
    try:
        _ = model(**inputs, use_cache=False)
    finally:
        for handle in handles:
            handle.remove()
    expected = {(layer, component, region) for layer in layers for component in components for region in regions}
    missing = sorted(expected - set(captured))
    if missing:
        raise RuntimeError(f"Missing captured states: {missing[:10]}{'...' if len(missing) > 10 else ''}")
    return captured, {
        "seq_len": seq_len,
        "answer_start": spans.answer_start,
        "answer_end": spans.answer_end,
        "answer_tokens": spans.answer_end - spans.answer_start,
    }


def site_metrics(deltas: list[np.ndarray], private_neutral: list[np.ndarray], public_neutral: list[np.ndarray]) -> dict[str, float | np.ndarray]:
    mean_dir = np.mean(np.stack(deltas, axis=0), axis=0).astype(np.float32)
    mean_norm = float(np.linalg.norm(mean_dir))
    delta_norms = [float(np.linalg.norm(delta)) for delta in deltas]
    if mean_norm <= 1e-12:
        consistency = 0.0
        pn_alignment = 0.0
        pub_alignment = 0.0
    else:
        consistency = float(np.mean([cosine(delta, mean_dir) for delta in deltas]))
        pn_alignment = float(np.mean([cosine(delta, mean_dir) for delta in private_neutral]))
        pub_alignment = float(np.mean([cosine(delta, mean_dir) for delta in public_neutral]))
    triad = pn_alignment - pub_alignment
    score = max(0.0, consistency) * max(0.0, triad) * math.log1p(float(np.mean(delta_norms)))
    return {
        "mean_direction": mean_dir,
        "mean_direction_norm": mean_norm,
        "mean_pair_delta_norm": float(np.mean(delta_norms)),
        "sd_pair_delta_norm": float(np.std(delta_norms, ddof=1)) if len(delta_norms) > 1 else 0.0,
        "delta_consistency_cos_to_mean": consistency,
        "private_neutral_alignment": pn_alignment,
        "public_neutral_alignment": pub_alignment,
        "triad_separation": triad,
        "rank_score": float(score),
    }


def compute_metrics(
    records: dict[tuple[int, str], dict[tuple[int, str, str], np.ndarray]],
    prompt_ids: list[int],
    *,
    shuffle_controls: int,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[tuple[int, str, str], np.ndarray]]:
    site_keys = sorted(next(iter(records.values())).keys())
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    directions: dict[tuple[int, str, str], np.ndarray] = {}
    for site in site_keys:
        deltas = [records[(prompt_id, "private_rights")][site] - records[(prompt_id, "public_rights")][site] for prompt_id in prompt_ids]
        private_neutral = [
            records[(prompt_id, "private_rights")][site] - records[(prompt_id, "neutral")][site]
            for prompt_id in prompt_ids
        ]
        public_neutral = [
            records[(prompt_id, "public_rights")][site] - records[(prompt_id, "neutral")][site]
            for prompt_id in prompt_ids
        ]
        metrics = site_metrics(deltas, private_neutral, public_neutral)
        null_scores: list[float] = []
        null_consistency: list[float] = []
        for _ in range(shuffle_controls):
            signs = rng.choice(np.array([-1.0, 1.0], dtype=np.float32), size=len(deltas))
            null_deltas = [delta * sign for delta, sign in zip(deltas, signs)]
            null = site_metrics(null_deltas, private_neutral, public_neutral)
            null_scores.append(float(null["rank_score"]))
            null_consistency.append(float(null["delta_consistency_cos_to_mean"]))
        layer, component, region = site
        null_max = max(null_scores) if null_scores else 0.0
        row = {
            "layer": int(layer),
            "component": str(component),
            "region": str(region),
            "n_prompt_pairs": len(prompt_ids),
            "mean_direction_norm": float(metrics["mean_direction_norm"]),
            "mean_pair_delta_norm": float(metrics["mean_pair_delta_norm"]),
            "sd_pair_delta_norm": float(metrics["sd_pair_delta_norm"]),
            "delta_consistency_cos_to_mean": float(metrics["delta_consistency_cos_to_mean"]),
            "private_neutral_alignment": float(metrics["private_neutral_alignment"]),
            "public_neutral_alignment": float(metrics["public_neutral_alignment"]),
            "triad_separation": float(metrics["triad_separation"]),
            "rank_score": float(metrics["rank_score"]),
            "shuffle_controls": int(shuffle_controls),
            "shuffle_rank_score_mean": float(np.mean(null_scores)) if null_scores else 0.0,
            "shuffle_rank_score_max": float(null_max),
            "shuffle_consistency_mean": float(np.mean(null_consistency)) if null_consistency else 0.0,
            "rank_score_minus_shuffle_max": float(metrics["rank_score"]) - float(null_max),
        }
        rows.append(row)
        directions[site] = np.asarray(metrics["mean_direction"], dtype=np.float32)
    rows.sort(
        key=lambda row: (
            float(row["rank_score_minus_shuffle_max"]),
            float(row["rank_score"]),
            float(row["delta_consistency_cos_to_mean"]),
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
            "direction_semantics": "local_counterfactual_answer_private_minus_public",
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
    token_rows: list[dict[str, Any]],
) -> None:
    by_component: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in metrics:
        by_component[str(row["component"])].append(row)
    lines = [
        "# Article III Local Counterfactual Answer-State Localization",
        "",
        "## Purpose",
        "",
        (
            "Localize private-minus-public differences in local Qwen-generated answer trajectories under "
            "inserted visible thoughts. This is candidate nomination, not actuator evidence."
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
                ["Prompt ids", ",".join(str(item) for item in manifest["prompt_ids"])],
                ["Layers", manifest["layers"]],
                ["Components", ", ".join(manifest["components"])],
                ["Regions", ", ".join(manifest["regions"])],
                ["Shuffle controls", manifest["shuffle_controls"]],
                ["Source answer tokens", manifest.get("source_answer_tokens")],
                ["Source budget note", manifest.get("source_budget_note")],
                ["Output dir", manifest["output_dir"]],
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
                "Consistency",
                "Triad",
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
                    fmt(row["delta_consistency_cos_to_mean"]),
                    fmt(row["triad_separation"]),
                    fmt(row["mean_direction_norm"]),
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
                    ["Rank", "Layer", "Region", "Score-null", "Consistency", "Triad"],
                    [
                        [
                            idx + 1,
                            row["layer"],
                            row["region"],
                            fmt(row["rank_score_minus_shuffle_max"]),
                            fmt(row["delta_consistency_cos_to_mean"]),
                            fmt(row["triad_separation"]),
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
                ["Prompt", "Condition", "Seq len", "Answer tokens", "Answer span"],
                [
                    [
                        row["prompt_key"],
                        row["condition"],
                        row["seq_len"],
                        row["answer_tokens"],
                        f"{row['answer_start']}:{row['answer_end']}",
                    ]
                    for row in token_rows[:36]
                ],
            ),
            "",
            "## Interpretation",
            "",
            "- These directions are local counterfactual answer-state private-minus-public deltas.",
            "- Inserted thoughts are the source condition, so these are not no-mask actuator evidence.",
            "- A direction should only advance if its top sites differ from closed generated-baseline sites and then pass a no-mask generation gate.",
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
    parser.add_argument("--prompt-ids", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--layers", default="all")
    parser.add_argument("--components", default=DEFAULT_COMPONENTS)
    parser.add_argument("--regions", default=DEFAULT_REGIONS)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--shuffle-controls", type=int, default=128)
    parser.add_argument("--top-k", type=int, default=64)
    parser.add_argument("--device-map", default="single")
    parser.add_argument("--seed", type=int, default=20260502)
    parser.add_argument("--allow-short-source-budget", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = now_iso()
    source_budget = source_budget_meta(args.source_generations, allow_short=args.allow_short_source_budget)
    prompt_ids = parse_int_list(args.prompt_ids)
    rows = read_jsonl(args.source_generations)
    by_key = {(int(row["prompt_id"]), str(row["condition"])): row for row in rows}
    required = [(prompt_id, condition) for prompt_id in prompt_ids for condition in ("neutral", "private_rights", "public_rights")]
    missing = [key for key in required if key not in by_key or not str(by_key[key].get("answer") or "").strip()]
    if missing:
        raise ValueError(f"Missing required counterfactual answer rows: {missing[:10]}")

    components = parse_csv(args.components)
    regions = parse_csv(args.regions)
    allowed_regions = {"pre_answer_last", "answer_mean", "answer_first64_mean", "answer_tail64_mean", "tail32_mean"}
    unknown_regions = sorted(set(regions) - allowed_regions)
    if unknown_regions:
        raise ValueError(f"Unknown regions: {unknown_regions}")

    out_dir = args.output_root / f"scotus_article3_counterfactual_answer_state_localization_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)
    tokenizer, model = load_model_and_tokenizer(args.model_path, args.device_map)
    layers_mod = transformer_layers(model)
    layers = parse_layers(args.layers, len(layers_mod))
    print(
        f"Capturing {len(required)} answer trajectories; layers={len(layers)} "
        f"components={components} regions={regions}",
        flush=True,
    )

    records: dict[tuple[int, str], dict[tuple[int, str, str], np.ndarray]] = {}
    token_rows: list[dict[str, Any]] = []
    for prompt_id, condition in tqdm(required, desc="counterfactual answers", unit="row"):
        row = by_key[(prompt_id, condition)]
        captured, token_meta = capture_answer_state(
            model=model,
            tokenizer=tokenizer,
            layers_mod=layers_mod,
            row=row,
            layers=layers,
            components=components,
            regions=regions,
            max_length=args.max_length,
        )
        records[(prompt_id, condition)] = captured
        token_rows.append(
            {
                "prompt_id": prompt_id,
                "prompt_key": str(row["prompt_key"]),
                "condition": condition,
                **token_meta,
            }
        )

    metrics, directions = compute_metrics(
        records,
        prompt_ids,
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
        "prompt_ids": prompt_ids,
        "conditions": ["neutral", "private_rights", "public_rights"],
        "layers": "all" if len(layers) == len(layers_mod) else ",".join(str(layer) for layer in layers),
        "n_layers": len(layers),
        "components": components,
        "regions": regions,
        "max_length": args.max_length,
        "shuffle_controls": args.shuffle_controls,
        "top_k": args.top_k,
        "seed": args.seed,
        "method": "local_counterfactual_answer_private_minus_public_state_localization",
        "not_actuator_evidence": True,
        **source_budget,
    }
    write_json(out_dir / "manifest.json", manifest)
    write_jsonl(out_dir / "token_spans.jsonl", token_rows)
    write_jsonl(out_dir / "site_metrics.jsonl", metrics)
    write_report(out_dir / "report.md", manifest=manifest, metrics=metrics, direction_meta=direction_meta, token_rows=token_rows)

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
            f"consistency={top['delta_consistency_cos_to_mean']:.4f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
