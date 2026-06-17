#!/usr/bin/env python3
"""Prototype-replacement causal probe for SCOTUS minimal-pair replay states.

Single-vector act-add probes repeatedly found decodable judicial/legal answer
states without reliable steering. This script tests a different intervention:
blend selected residual layers toward a frozen Commerce-limits replay prototype
during generation, with same-layer random prototype controls.
"""

from __future__ import annotations

import argparse
import gc
import json
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
    load_model_and_tokenizer,
    load_prompt_specs,
    make_random_direction,
    now_iso,
    parse_float_list,
    row_for_generation,
    select_prompt_specs,
    summarize_frames,
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
    return [int(part.strip()) for part in text.split(",") if part.strip()]


def parse_str_set(text: str) -> set[str]:
    return {part.strip() for part in text.split(",") if part.strip()}


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


def prototype_indices(meta: list[dict[str, Any]], fit_splits: set[str], label_name: str) -> list[int]:
    indices = [
        idx
        for idx, row in enumerate(meta)
        if str(row.get("split")) in fit_splits and str(row.get("frame_label")) == label_name
    ]
    if not indices:
        raise RuntimeError(f"No rows for label={label_name!r}, splits={sorted(fit_splits)}")
    return indices


def build_prototypes(
    *,
    replay_run: Path,
    region: str,
    layers: list[int],
    fit_splits: set[str],
    target_label: str,
) -> tuple[dict[int, torch.Tensor], list[dict[str, Any]]]:
    meta = read_jsonl(replay_run / "feature_meta.jsonl")
    target_idx = prototype_indices(meta, fit_splits, target_label)
    features = np.load(replay_run / "features.npz")
    prototypes: dict[int, torch.Tensor] = {}
    prototype_meta: list[dict[str, Any]] = []
    try:
        for layer in layers:
            key = f"{region}__L{layer:02d}"
            if key not in features.files:
                raise KeyError(f"Missing {key} in {replay_run / 'features.npz'}")
            values = features[key].astype(np.float32, copy=False)
            proto = values[target_idx].mean(axis=0).astype(np.float32, copy=False)
            proto_norm = float(np.linalg.norm(proto))
            if proto_norm <= 0.0 or not np.isfinite(proto_norm):
                raise RuntimeError(f"Bad prototype norm for {key}: {proto_norm}")
            prototypes[layer] = torch.from_numpy(proto.copy()).float().contiguous()
            prototype_meta.append(
                {
                    "source": "minimal_pair_replay_prototype",
                    "name": f"prototype_{target_label}_{region}_L{layer:02d}",
                    "layer": layer,
                    "region": region,
                    "group_field": "prototype",
                    "group_value": target_label,
                    "target_justice": target_label,
                    "reference_justice": "current_hidden_state",
                    "n_target": len(target_idx),
                    "n_reference": "",
                    "raw_direction_norm": proto_norm,
                    "prototype_norm": proto_norm,
                    "fit_splits": sorted(fit_splits),
                }
            )
    finally:
        features.close()
    return prototypes, prototype_meta


def random_prototypes(prototypes: dict[int, torch.Tensor], seed: int) -> dict[int, torch.Tensor]:
    randoms: dict[int, torch.Tensor] = {}
    for layer, proto in prototypes.items():
        random_unit = make_random_direction(int(proto.numel()), seed + layer * 1009)
        randoms[layer] = random_unit * torch.linalg.vector_norm(proto).float()
    return randoms


def install_prototype_hooks(
    layers_mod: Any,
    *,
    layer_to_proto: dict[int, torch.Tensor],
    blend: float,
    position: str,
) -> list[Any]:
    if position not in {"last", "all"}:
        raise ValueError(f"Unsupported position: {position}")
    handles: list[Any] = []
    for layer, proto in layer_to_proto.items():

        def make_hook(layer_proto: torch.Tensor) -> Any:
            def hook(_module: torch.nn.Module, _inp: tuple[Any, ...], out: Any) -> Any:
                hidden = out[0] if isinstance(out, tuple) else out
                proto_vec = layer_proto.to(device=hidden.device, dtype=hidden.dtype)
                edited = hidden.clone()
                if position == "last":
                    edited[:, -1, :] = edited[:, -1, :] + float(blend) * (proto_vec - edited[:, -1, :])
                else:
                    edited = edited + float(blend) * (proto_vec.view(1, 1, -1) - edited)
                if isinstance(out, tuple):
                    return (edited,) + out[1:]
                return edited

            return hook

        handles.append(layers_mod[layer].register_forward_hook(make_hook(proto)))
    return handles


@torch.inference_mode()
def generate_many_prototype(
    *,
    model: torch.nn.Module,
    tokenizer: Any,
    prompts: list[str],
    layers_mod: Any,
    layer_to_proto: dict[int, torch.Tensor] | None,
    blend: float,
    position: str,
    max_new_tokens: int,
    batch_size: int,
) -> list[dict[str, Any]]:
    handles: list[Any] = []
    if layer_to_proto and blend != 0.0:
        handles = install_prototype_hooks(
            layers_mod,
            layer_to_proto=layer_to_proto,
            blend=blend,
            position=position,
        )
    outputs: list[dict[str, Any]] = []
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--replay-run", type=Path, default=DEFAULT_REPLAY_RUN)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--prompt-bank", type=Path, default=DEFAULT_PROMPT_BANK)
    parser.add_argument("--region", default="assistant_all")
    parser.add_argument("--layers", default="16,20")
    parser.add_argument("--fit-splits", default="train")
    parser.add_argument("--target-label", default="commerce_limits")
    parser.add_argument("--blends", default="0.01,0.03,0.05")
    parser.add_argument("--position", choices=["last", "all"], default="all")
    parser.add_argument("--prompt-ids", default="0,1,2,3,4,5")
    parser.add_argument("--max-prompts", type=int, default=6)
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_COMPLETE_ANSWER_TOKENS)
    add_short_budget_arg(parser)
    parser.add_argument("--generation-batch-size", type=int, default=1)
    parser.add_argument("--random-controls", type=int, default=4)
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--report-max-output-rows", type=int, default=200)
    parser.add_argument("--device-map", default="single")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    enforce_complete_answer_budget(
        args.max_new_tokens,
        allow_short=args.allow_short_answer_budget,
        purpose="prototype-replacement replay patch",
    )
    started = now_iso()
    layers = parse_int_list(args.layers)
    blends = parse_float_list(args.blends)
    fit_splits = parse_str_set(args.fit_splits)
    replay_run = args.replay_run if args.replay_run.is_absolute() else PROJECT_ROOT / args.replay_run
    out_dir = args.output_root / f"scotus_prototype_patch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)

    prototypes, prototype_meta = build_prototypes(
        replay_run=replay_run,
        region=args.region,
        layers=layers,
        fit_splits=fit_splits,
        target_label=args.target_label,
    )
    candidate_name = f"prototype_{args.target_label}_{args.region}_{'-'.join(f'L{layer:02d}' for layer in layers)}"
    direction_meta = [
        {
            "source": "minimal_pair_replay_prototype_patch",
            "name": candidate_name,
            "layer": ",".join(str(layer) for layer in layers),
            "region": args.region,
            "group_field": "prototype",
            "group_value": args.target_label,
            "target_justice": args.target_label,
            "reference_justice": "current_hidden_state",
            "n_target": prototype_meta[0]["n_target"],
            "n_reference": "",
            "raw_direction_norm": "",
            "top_features": [],
            "layers": prototype_meta,
        }
    ]

    all_prompt_specs = load_prompt_specs(args.prompt_bank)
    prompt_specs = select_prompt_specs(all_prompt_specs, args.prompt_ids, args.max_prompts)
    prompt_texts = [spec.prompt for spec in prompt_specs]
    print(f"Loaded {len(prompt_specs)} prompts", flush=True)
    print(f"Built prototypes for layers {layers}", flush=True)

    tokenizer, model = load_model_and_tokenizer(args.model_path, args.device_map)
    layers_mod = transformer_layers(model)
    random_proto_bank = [
        random_prototypes(prototypes, args.seed + random_idx * 100_003)
        for random_idx in range(max(0, args.random_controls))
    ]

    rows: list[dict[str, Any]] = []
    print("Generating baseline batch", flush=True)
    base_outputs = generate_many_prototype(
        model=model,
        tokenizer=tokenizer,
        prompts=prompt_texts,
        layers_mod=layers_mod,
        layer_to_proto=None,
        blend=0.0,
        position=args.position,
        max_new_tokens=args.max_new_tokens,
        batch_size=args.generation_batch_size,
    )
    for spec, output in zip(prompt_specs, base_outputs, strict=True):
        rows.append(
            row_for_generation(
                spec=spec,
                condition="base",
                candidate=None,
                alpha=0.0,
                effective_alpha=0.0,
                random_index=None,
                layer=None,
                output=output,
            )
        )

    for blend in blends:
        for random_idx, random_proto in enumerate(random_proto_bank):
            print(f"Generating random prototype[{random_idx}] blend={blend}", flush=True)
            random_outputs = generate_many_prototype(
                model=model,
                tokenizer=tokenizer,
                prompts=prompt_texts,
                layers_mod=layers_mod,
                layer_to_proto=random_proto,
                blend=blend,
                position=args.position,
                max_new_tokens=args.max_new_tokens,
                batch_size=args.generation_batch_size,
            )
            for spec, output in zip(prompt_specs, random_outputs, strict=True):
                rows.append(
                    row_for_generation(
                        spec=spec,
                        condition="random_unit",
                        candidate="random_prototype",
                        alpha=blend,
                        effective_alpha=blend,
                        random_index=random_idx,
                        layer=layers[0],
                        output=output,
                    )
                )

        print(f"Generating target prototype blend={blend}", flush=True)
        candidate_outputs = generate_many_prototype(
            model=model,
            tokenizer=tokenizer,
            prompts=prompt_texts,
            layers_mod=layers_mod,
            layer_to_proto=prototypes,
            blend=blend,
            position=args.position,
            max_new_tokens=args.max_new_tokens,
            batch_size=args.generation_batch_size,
        )
        for spec, output in zip(prompt_specs, candidate_outputs, strict=True):
            rows.append(
                row_for_generation(
                    spec=spec,
                    condition="sae_poke",
                    candidate=candidate_name,
                    alpha=blend,
                    effective_alpha=blend,
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
        "overlap_dir": str(replay_run),
        "output_dir": str(out_dir),
        "direction_source": "external",
        "external_direction_files": [],
        "alpha_scale": "prototype_blend",
        "hidden_norm_reference": str(replay_run),
        "prompt_bank": str(args.prompt_bank),
        "candidate_names": [candidate_name],
        "prototype_meta": prototype_meta,
        "alphas": blends,
        "random_controls": int(args.random_controls),
        "position": args.position,
        "max_prompts": args.max_prompts,
        "max_new_tokens": args.max_new_tokens,
        **qwen_budget_metadata(args.max_new_tokens),
        "generation_batch_size": int(args.generation_batch_size),
        "report_max_output_rows": int(args.report_max_output_rows),
        "do_sample": False,
        "temperature": None,
        "top_p": None,
        "seed": args.seed,
        "prompt_ids": [spec.prompt_id for spec in prompt_specs],
        "prompt_keys": [spec.prompt_key for spec in prompt_specs],
    }
    write_json(out_dir / "manifest.json", manifest)
    write_jsonl(out_dir / "direction_meta.jsonl", direction_meta)
    write_jsonl(out_dir / "generations.jsonl", rows)
    write_jsonl(out_dir / "frame_summary.jsonl", summarize_frames(rows))
    write_jsonl(out_dir / "score_summary.jsonl", aggregate_frame_scores(rows))
    write_jsonl(out_dir / "candidate_vs_random.jsonl", candidate_vs_random(rows))
    write_jsonl(out_dir / "candidate_vs_prompt_matched_random.jsonl", candidate_vs_prompt_matched_random(rows))
    write_report(out_dir / "report.md", manifest=manifest, direction_meta=direction_meta, rows=rows)

    del model, tokenizer, layers_mod
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(f"Wrote {out_dir / 'report.md'}", flush=True)


if __name__ == "__main__":
    main()
