#!/usr/bin/env python3
"""Causal poke for SCOTUS minimal-pair replay directions as a layer bundle.

The earlier minimal-pair replay run exported one best linear direction and the
single-vector act-add follow-up did not promote. This script tests a narrower
follow-up: build paired Commerce-limits minus Commerce-authority deltas from
the replay feature bank, apply them as a multi-layer bundle during generation,
and compare against same-layer random bundles.
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

from poke_scotus_sae_layers import (
    DEFAULT_COMPLETE_ANSWER_TOKENS,
    DEFAULT_MODEL,
    DEFAULT_OUTPUT_ROOT,
    add_base_deltas,
    aggregate_frame_scores,
    candidate_vs_prompt_matched_random,
    candidate_vs_random,
    first_parameter_device,
    format_chat,
    generate_many,
    load_model_and_tokenizer,
    load_prompt_specs,
    make_random_direction,
    now_iso,
    parse_float_list,
    row_for_generation,
    select_prompt_specs,
    transformer_layers,
    write_json,
    write_jsonl,
    write_report,
)
from qwen_eval_budget import add_short_budget_arg, enforce_complete_answer_budget, qwen_budget_metadata


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_REPLAY_RUN = PROJECT_ROOT / "sweep_v4" / "scotus_minpair_replay_20260501_100514"
DEFAULT_PROMPT_BANK = PROJECT_ROOT / "data" / "scotus" / "scotus_commerce_pocket_prompts_v1.jsonl"


def parse_int_list(text: str) -> list[int]:
    return [int(item.strip()) for item in text.split(",") if item.strip()]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON") from exc
    return rows


def paired_indices(meta: list[dict[str, Any]], fit_splits: set[str]) -> list[tuple[str, int, int]]:
    by_fact: dict[str, dict[str, int]] = {}
    for idx, row in enumerate(meta):
        if str(row.get("split")) not in fit_splits:
            continue
        fact_id = str(row.get("fact_id"))
        label = str(row.get("frame_label"))
        by_fact.setdefault(fact_id, {})[label] = idx
    pairs: list[tuple[str, int, int]] = []
    for fact_id, labels in sorted(by_fact.items()):
        if "commerce_authority" in labels and "commerce_limits" in labels:
            pairs.append((fact_id, labels["commerce_authority"], labels["commerce_limits"]))
    if not pairs:
        raise RuntimeError(f"No authority/limits pairs found for splits {sorted(fit_splits)}")
    return pairs


def median_hidden_norm(features: np.lib.npyio.NpzFile, key: str) -> float:
    values = features[key].astype(np.float32, copy=False)
    norms = np.linalg.norm(values, axis=1)
    median = float(np.median(norms))
    if median <= 0.0 or not math.isfinite(median):
        raise RuntimeError(f"Bad median hidden norm for {key}: {median}")
    return median


def oriented_pc1(deltas: np.ndarray, mean_delta: np.ndarray) -> np.ndarray:
    _u, _s, vt = np.linalg.svd(deltas.astype(np.float64, copy=False), full_matrices=False)
    pc1 = vt[0].astype(np.float32, copy=False)
    if float(np.dot(pc1, mean_delta)) < 0.0:
        pc1 = -pc1
    return pc1


def unit(vec: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if norm <= 0.0 or not math.isfinite(norm):
        raise RuntimeError("Zero or non-finite vector")
    return (vec / norm).astype(np.float32, copy=False)


def build_layer_bundle(
    *,
    replay_run: Path,
    region: str,
    layers: list[int],
    fit_splits: set[str],
    mode: str,
) -> tuple[dict[int, torch.Tensor], list[dict[str, Any]], dict[int, float]]:
    meta = read_jsonl(replay_run / "feature_meta.jsonl")
    pairs = paired_indices(meta, fit_splits)
    features = np.load(replay_run / "features.npz")
    bundle: dict[int, torch.Tensor] = {}
    direction_meta: list[dict[str, Any]] = []
    scale_factors: dict[int, float] = {}
    try:
        for layer in layers:
            key = f"{region}__L{layer:02d}"
            if key not in features.files:
                raise KeyError(f"Missing {key} in {replay_run / 'features.npz'}")
            values = features[key].astype(np.float32, copy=False)
            deltas = np.stack([values[limits_idx] - values[authority_idx] for _, authority_idx, limits_idx in pairs])
            mean_delta = deltas.mean(axis=0)
            if mode == "mean":
                direction = unit(mean_delta)
                raw_norm = float(np.linalg.norm(mean_delta))
            elif mode == "pc1":
                pc1 = oriented_pc1(deltas, mean_delta)
                direction = unit(pc1)
                raw_norm = 1.0
            else:
                raise ValueError(f"Unknown mode: {mode}")
            bundle[layer] = torch.from_numpy(direction).float().contiguous()
            scale_factors[layer] = median_hidden_norm(features, key)
            unit_deltas = deltas / np.maximum(np.linalg.norm(deltas, axis=1, keepdims=True), 1e-12)
            mean_unit = unit(mean_delta)
            direction_meta.append(
                {
                    "source": "minimal_pair_replay_bundle",
                    "name": f"minpair_{mode}_{region}_L{layer:02d}",
                    "layer": layer,
                    "region": region,
                    "group_field": "bundle",
                    "group_value": mode,
                    "target_justice": "commerce_limits",
                    "reference_justice": "commerce_authority",
                    "n_target": len(pairs),
                    "n_reference": len(pairs),
                    "raw_direction_norm": raw_norm,
                    "alpha_scale_factor": scale_factors[layer],
                    "mean_pair_cos_to_mean": float(np.mean(unit_deltas @ mean_unit)),
                    "fit_splits": sorted(fit_splits),
                }
            )
    finally:
        features.close()
    return bundle, direction_meta, scale_factors


def random_bundle(layers: list[int], dim: int, seed: int) -> dict[int, torch.Tensor]:
    return {layer: make_random_direction(dim, seed + layer * 1009) for layer in layers}


def install_bundle_hooks(
    layers_mod: Any,
    *,
    layer_to_vec: dict[int, torch.Tensor],
    alpha: float,
    scale_factors: dict[int, float],
    position: str,
) -> list[Any]:
    if position != "last":
        raise ValueError("Only last-token bundle patching is currently supported")
    handles: list[Any] = []
    for layer, direction in layer_to_vec.items():
        effective = float(alpha) * float(scale_factors[layer])

        def make_hook(vec: torch.Tensor, eff: float):
            def hook(_module: torch.nn.Module, _inp: tuple[Any, ...], out: Any) -> Any:
                hidden = out[0] if isinstance(out, tuple) else out
                poke = vec.to(device=hidden.device, dtype=hidden.dtype) * eff
                edited = hidden.clone()
                edited[:, -1, :] = edited[:, -1, :] + poke
                if isinstance(out, tuple):
                    return (edited,) + out[1:]
                return edited

            return hook

        handles.append(layers_mod[layer].register_forward_hook(make_hook(direction, effective)))
    return handles


@torch.inference_mode()
def generate_many_bundle(
    *,
    model: torch.nn.Module,
    tokenizer: Any,
    prompts: list[str],
    layers_mod: Any,
    layer_to_vec: dict[int, torch.Tensor] | None,
    alpha: float,
    scale_factors: dict[int, float],
    position: str,
    max_new_tokens: int,
    batch_size: int,
) -> list[dict[str, Any]]:
    outputs: list[dict[str, Any]] = []
    handles: list[Any] = []
    if layer_to_vec and alpha != 0.0:
        handles = install_bundle_hooks(
            layers_mod,
            layer_to_vec=layer_to_vec,
            alpha=alpha,
            scale_factors=scale_factors,
            position=position,
        )
    try:
        input_device = first_parameter_device(model)
        for start in range(0, len(prompts), max(1, batch_size)):
            batch_prompts = prompts[start : start + max(1, batch_size)]
            chats = [format_chat(tokenizer, prompt) for prompt in batch_prompts]
            inputs = tokenizer(
                chats,
                return_tensors="pt",
                add_special_tokens=False,
                truncation=True,
                max_length=2048,
                padding=True,
            )
            inputs = {key: value.to(input_device) for key, value in inputs.items()}
            generated_batch = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=True,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
            )
            prompt_width = int(inputs["input_ids"].shape[-1])
            for generated in generated_batch[:, prompt_width:]:
                outputs.append(
                    {
                        "text": tokenizer.decode(generated, skip_special_tokens=True).strip(),
                        "generated_tokens": int(generated.numel()),
                        "prompt_tokens": prompt_width,
                    }
                )
    finally:
        for handle in handles:
            handle.remove()
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Poke SCOTUS minimal-pair replay layer bundles.")
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--replay-run", type=Path, default=DEFAULT_REPLAY_RUN)
    parser.add_argument("--prompt-bank", type=Path, default=DEFAULT_PROMPT_BANK)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--region", default="assistant_all")
    parser.add_argument("--layers", default="8,12,16,20")
    parser.add_argument("--fit-splits", default="train")
    parser.add_argument("--mode", choices=["mean", "pc1"], default="mean")
    parser.add_argument("--alphas", default="0.003,0.005,0.01")
    parser.add_argument("--prompt-ids", default="0,1,2,3,4,5")
    parser.add_argument("--max-prompts", type=int, default=6)
    parser.add_argument("--random-controls", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_COMPLETE_ANSWER_TOKENS)
    add_short_budget_arg(parser)
    parser.add_argument("--generation-batch-size", type=int, default=1)
    parser.add_argument("--report-max-output-rows", type=int, default=80)
    parser.add_argument("--device-map", default="single")
    parser.add_argument("--seed", type=int, default=20260502)
    args = parser.parse_args()
    enforce_complete_answer_budget(
        args.max_new_tokens,
        allow_short=args.allow_short_answer_budget,
        purpose="minimal-pair replay layer-bundle poke",
    )

    started = now_iso()
    out_dir = args.output_root / f"scotus_minpair_bundle_poke_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)
    layers = parse_int_list(args.layers)
    alphas = parse_float_list(args.alphas)
    fit_splits = {item.strip() for item in args.fit_splits.split(",") if item.strip()}

    candidate_bundle, direction_meta, scale_factors = build_layer_bundle(
        replay_run=args.replay_run,
        region=args.region,
        layers=layers,
        fit_splits=fit_splits,
        mode=args.mode,
    )
    candidate_name = f"minpair_{args.mode}_{args.region}_L{'_'.join(str(layer) for layer in layers)}"
    layer_dim = int(next(iter(candidate_bundle.values())).numel())
    random_bundles = [random_bundle(layers, layer_dim, args.seed + idx * 100_003) for idx in range(args.random_controls)]

    all_prompt_specs = load_prompt_specs(args.prompt_bank)
    prompt_specs = select_prompt_specs(all_prompt_specs, args.prompt_ids, args.max_prompts)
    prompt_texts = [spec.prompt for spec in prompt_specs]
    print(f"Loaded {len(prompt_specs)} prompts", flush=True)
    print(f"Built bundle {candidate_name} layers={layers} mode={args.mode}", flush=True)

    tokenizer, model = load_model_and_tokenizer(args.model_path, args.device_map)
    layers_mod = transformer_layers(model)

    rows: list[dict[str, Any]] = []
    print("Generating baseline batch", flush=True)
    base_outputs = generate_many(
        model=model,
        tokenizer=tokenizer,
        prompts=prompt_texts,
        layers_mod=layers_mod,
        layer=None,
        direction=None,
        alpha=0.0,
        position="last",
        max_new_tokens=args.max_new_tokens,
        do_sample=False,
        temperature=0.7,
        top_p=0.9,
        batch_size=args.generation_batch_size,
    )
    for spec, output in zip(prompt_specs, base_outputs, strict=True):
        rows.append(
            row_for_generation(
                spec=spec,
                condition="base",
                candidate=None,
                alpha=0.0,
                random_index=None,
                layer=None,
                output=output,
            )
        )

    for alpha in alphas:
        for random_idx, bundle in enumerate(random_bundles):
            print(f"Generating random_bundle[{random_idx}] alpha={alpha}", flush=True)
            outputs = generate_many_bundle(
                model=model,
                tokenizer=tokenizer,
                prompts=prompt_texts,
                layers_mod=layers_mod,
                layer_to_vec=bundle,
                alpha=float(alpha),
                scale_factors=scale_factors,
                position="last",
                max_new_tokens=args.max_new_tokens,
                batch_size=args.generation_batch_size,
            )
            for spec, output in zip(prompt_specs, outputs, strict=True):
                rows.append(
                    row_for_generation(
                        spec=spec,
                        condition="random_unit",
                        candidate="random_bundle",
                        alpha=float(alpha),
                        effective_alpha=float(alpha),
                        random_index=random_idx,
                        layer=layers[0],
                        output=output,
                    )
                )
        print(f"Generating {candidate_name} alpha={alpha}", flush=True)
        outputs = generate_many_bundle(
            model=model,
            tokenizer=tokenizer,
            prompts=prompt_texts,
            layers_mod=layers_mod,
            layer_to_vec=candidate_bundle,
            alpha=float(alpha),
            scale_factors=scale_factors,
            position="last",
            max_new_tokens=args.max_new_tokens,
            batch_size=args.generation_batch_size,
        )
        for spec, output in zip(prompt_specs, outputs, strict=True):
            rows.append(
                row_for_generation(
                    spec=spec,
                    condition="sae_poke",
                    candidate=candidate_name,
                    alpha=float(alpha),
                    effective_alpha=float(alpha),
                    random_index=None,
                    layer=layers[0],
                    output=output,
                )
            )

    add_base_deltas(rows)
    manifest = {
        "started_at": started,
        "finished_at": now_iso(),
        "model_path": str(args.model_path),
        "sae_path": "",
        "overlap_dir": str(args.replay_run),
        "output_dir": str(out_dir),
        "candidate_names": [candidate_name],
        "external_direction_files": [],
        "direction_source": "external",
        "alpha_scale": "hidden-norm-fraction-per-layer",
        "hidden_norm_reference": str(args.replay_run),
        "prompt_bank": str(args.prompt_bank),
        "top_features": 0,
        "alphas": alphas,
        "random_controls": int(args.random_controls),
        "position": "last",
        "max_prompts": args.max_prompts,
        "max_new_tokens": args.max_new_tokens,
        **qwen_budget_metadata(args.max_new_tokens),
        "generation_batch_size": int(args.generation_batch_size),
        "report_max_output_rows": int(args.report_max_output_rows),
        "do_sample": False,
        "temperature": 0.0,
        "top_p": 1.0,
        "seed": args.seed,
        "prompt_ids": [spec.prompt_id for spec in prompt_specs],
        "prompt_keys": [spec.prompt_key for spec in prompt_specs],
        "layers": layers,
        "region": args.region,
        "mode": args.mode,
        "fit_splits": sorted(fit_splits),
        "scale_factors": {str(layer): scale_factors[layer] for layer in layers},
    }
    write_json(out_dir / "manifest.json", manifest)
    write_jsonl(out_dir / "direction_meta.jsonl", direction_meta)
    write_jsonl(out_dir / "generations.jsonl", rows)
    write_jsonl(out_dir / "score_summary.jsonl", aggregate_frame_scores(rows))
    write_jsonl(out_dir / "candidate_vs_random.jsonl", candidate_vs_random(rows))
    write_jsonl(out_dir / "candidate_vs_prompt_matched_random.jsonl", candidate_vs_prompt_matched_random(rows))
    write_report(out_dir / "report.md", manifest=manifest, direction_meta=direction_meta, rows=rows)
    print(f"Wrote {out_dir / 'report.md'}", flush=True)


if __name__ == "__main__":
    main()
