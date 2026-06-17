#!/usr/bin/env python3
"""Causal smoke test for learned low-rank SCOTUS replay interventions."""

from __future__ import annotations

import argparse
import gc
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from poke_scotus_sae_layers import (  # noqa: E402
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
    now_iso,
    parse_float_list,
    row_for_generation,
    select_prompt_specs,
    transformer_layers,
    write_json,
    write_jsonl,
)
from qwen_eval_budget import add_short_budget_arg, enforce_complete_answer_budget, qwen_budget_metadata  # noqa: E402
from train_scotus_replay_lowrank_intervention import (  # noqa: E402
    DEFAULT_REPLAY_RUN,
    feature_key,
    fit_lowrank,
    load_pair_batch,
    read_jsonl,
    safe_name,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PROMPT_BANK = PROJECT_ROOT / "data" / "scotus" / "scotus_commerce_pocket_prompts_v1.jsonl"


def parse_int_list(text: str) -> list[int]:
    return [int(item.strip()) for item in text.split(",") if item.strip()]


class TorchLowRankMap:
    def __init__(self, model: Any) -> None:
        self.rank = int(model.rank)
        self.ridge = float(model.ridge)
        self.permutation_seed = model.permutation_seed
        self.x_mean = torch.from_numpy(model.x_mean.astype(np.float32)).contiguous()
        self.delta_mean = torch.from_numpy(model.delta_mean.astype(np.float32)).contiguous()
        self.components = torch.from_numpy(model.components.astype(np.float32)).contiguous()
        self.x_train = torch.from_numpy(model.x_train.astype(np.float32)).contiguous()
        self.dual_coef = torch.from_numpy(model.dual_coef.astype(np.float32)).contiguous()

    def delta(self, hidden_2d: torch.Tensor) -> torch.Tensor:
        x_mean = self.x_mean.to(device=hidden_2d.device, dtype=hidden_2d.dtype)
        delta_mean = self.delta_mean.to(device=hidden_2d.device, dtype=hidden_2d.dtype)
        if self.rank == 0:
            return delta_mean.unsqueeze(0).expand(hidden_2d.shape[0], -1)
        x_train = self.x_train.to(device=hidden_2d.device, dtype=hidden_2d.dtype)
        dual_coef = self.dual_coef.to(device=hidden_2d.device, dtype=hidden_2d.dtype)
        components = self.components.to(device=hidden_2d.device, dtype=hidden_2d.dtype)
        coeff = ((hidden_2d - x_mean) @ x_train.T) @ dual_coef
        return delta_mean + coeff @ components


def install_lowrank_hook(
    layers_mod: Any,
    *,
    layer: int,
    lowrank_map: TorchLowRankMap,
    beta: float,
    position: str,
) -> Any:
    if position not in {"last", "all"}:
        raise ValueError(f"Unsupported position: {position}")

    def hook(_module: torch.nn.Module, _inp: tuple[Any, ...], out: Any) -> Any:
        hidden = out[0] if isinstance(out, tuple) else out
        edited = hidden.clone()
        if position == "last":
            current = edited[:, -1, :]
            edited[:, -1, :] = current + (float(beta) * lowrank_map.delta(current))
        else:
            shape = edited.shape
            flat = edited.reshape(-1, shape[-1])
            flat = flat + (float(beta) * lowrank_map.delta(flat))
            edited = flat.reshape(shape)
        if isinstance(out, tuple):
            return (edited,) + out[1:]
        return edited

    return layers_mod[layer].register_forward_hook(hook)


@torch.inference_mode()
def generate_many_lowrank(
    *,
    model: torch.nn.Module,
    tokenizer: Any,
    prompts: list[str],
    layers_mod: Any,
    layer: int | None,
    lowrank_map: TorchLowRankMap | None,
    beta: float,
    position: str,
    max_new_tokens: int,
    batch_size: int,
) -> list[dict[str, Any]]:
    handle = None
    if layer is not None and lowrank_map is not None and beta != 0.0:
        handle = install_lowrank_hook(layers_mod, layer=layer, lowrank_map=lowrank_map, beta=beta, position=position)
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
        if handle is not None:
            handle.remove()
    return outputs


def markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(item) for item in row) + " |")
    return "\n".join(lines)


def fmt(value: float) -> str:
    return f"{value:.3f}"


def write_lowrank_report(path: Path, *, manifest: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    comparisons = candidate_vs_prompt_matched_random(rows)
    aggregate_rows = aggregate_frame_scores(rows)
    lines = [
        "# SCOTUS Low-Rank Replay Causal Smoke",
        "",
        "## Purpose",
        "",
        "Apply a learned low-rank activation map during generation and compare it against same-family permutation low-rank controls. This is a smoke test for a new intervention family, not a promotion run.",
        "",
        "## Configuration",
        "",
        markdown_table(
            ["Field", "Value"],
            [
                ["Model", manifest["model_path"]],
                ["Replay run", manifest["replay_run"]],
                ["Feature key", manifest["feature_key"]],
                ["Source label", manifest["source_label"]],
                ["Target label", manifest["target_label"]],
                ["Rank/ridge", f"{manifest['rank']} / {manifest['ridge']}"],
                ["Betas", ", ".join(str(item) for item in manifest["betas"])],
                ["Permutation controls", manifest["permutation_controls"]],
                ["Prompt keys", ", ".join(manifest["prompt_keys"])],
                ["Position", manifest["position"]],
            ],
        ),
        "",
        "## Prompt-Matched Candidate vs Permutation Controls",
        "",
        markdown_table(
            [
                "Candidate",
                "Beta",
                "N",
                "Target minus control",
                "Net minus control",
                "Target win",
                "Net win",
            ],
            [
                [
                    row["candidate"],
                    row["alpha"],
                    row["n"],
                    fmt(float(row["mean_prompt_matched_delta_minus_random"])),
                    fmt(float(row["mean_prompt_matched_net_delta_minus_random"])),
                    fmt(float(row["prompt_win_rate_vs_random_mean"])),
                    fmt(float(row["prompt_net_win_rate_vs_random_mean"])),
                ]
                for row in comparisons
            ],
        ),
        "",
        "## Aggregate Frame Scores",
        "",
        markdown_table(
            ["Condition", "Candidate", "Beta", "N", "Mean target delta", "Mean net delta"],
            [
                [
                    row["condition"],
                    row.get("candidate", ""),
                    row.get("alpha", ""),
                    row["n"],
                    fmt(float(row["mean_delta_target_hits_vs_base"])),
                    fmt(float(row["mean_delta_target_minus_contrast_vs_base"])),
                ]
                for row in aggregate_rows
            ],
        ),
        "",
        "## Read",
        "",
        "- Controls are low-rank maps trained with permuted replay deltas, not ordinary random unit vectors.",
        "- Any survivor still needs proposition-level rescoring before interpretation.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--replay-run", type=Path, default=DEFAULT_REPLAY_RUN)
    parser.add_argument("--prompt-bank", type=Path, default=DEFAULT_PROMPT_BANK)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--region", default="assistant_all")
    parser.add_argument("--layer", type=int, default=8)
    parser.add_argument("--pair-field", default="pair_id")
    parser.add_argument("--label-field", default="frame_label")
    parser.add_argument("--source-label", default="commerce_authority")
    parser.add_argument("--target-label", default="commerce_limits")
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--ridge", type=float, default=1.0)
    parser.add_argument("--betas", default="0.25,0.5,1.0")
    parser.add_argument("--prompt-ids", default="0,2,6,7")
    parser.add_argument("--max-prompts", type=int, default=4)
    parser.add_argument("--permutation-controls", type=int, default=3)
    parser.add_argument("--position", choices=["last", "all"], default="last")
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_COMPLETE_ANSWER_TOKENS)
    add_short_budget_arg(parser)
    parser.add_argument("--generation-batch-size", type=int, default=1)
    parser.add_argument("--device-map", default="single")
    parser.add_argument("--seed", type=int, default=20260501)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    enforce_complete_answer_budget(
        args.max_new_tokens,
        allow_short=args.allow_short_answer_budget,
        purpose="low-rank replay poke",
    )
    started = now_iso()
    out_dir = args.output_root / f"scotus_lowrank_replay_poke_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)

    key = feature_key(args.region, args.layer)
    meta = read_jsonl(args.replay_run / "feature_meta.jsonl")
    with np.load(args.replay_run / "features.npz") as data:
        if key not in data.files:
            raise KeyError(f"Missing {key} in {args.replay_run / 'features.npz'}")
        features = data[key].astype(np.float32)
    train_batch = load_pair_batch(
        features,
        meta,
        "train",
        pair_field=args.pair_field,
        label_field=args.label_field,
        source_label=args.source_label,
        target_label=args.target_label,
    )
    candidate_map = TorchLowRankMap(
        fit_lowrank(train_batch, rank=args.rank, ridge=args.ridge, permutation_seed=None)
    )
    permutation_maps = [
        TorchLowRankMap(
            fit_lowrank(
                train_batch,
                rank=args.rank,
                ridge=args.ridge,
                permutation_seed=args.seed + idx * 100_003,
            )
        )
        for idx in range(args.permutation_controls)
    ]
    mean_map = TorchLowRankMap(fit_lowrank(train_batch, rank=0, ridge=args.ridge, permutation_seed=None))

    prompt_specs = select_prompt_specs(load_prompt_specs(args.prompt_bank), args.prompt_ids, args.max_prompts)
    prompt_texts = [spec.prompt for spec in prompt_specs]
    betas = parse_float_list(args.betas)
    print(f"Loaded {len(prompt_specs)} prompts and {len(permutation_maps)} permutation controls", flush=True)

    tokenizer, model = load_model_and_tokenizer(args.model_path, args.device_map)
    layers_mod = transformer_layers(model)
    rows: list[dict[str, Any]] = []

    print("Generating baseline batch", flush=True)
    base_outputs = generate_many_lowrank(
        model=model,
        tokenizer=tokenizer,
        prompts=prompt_texts,
        layers_mod=layers_mod,
        layer=None,
        lowrank_map=None,
        beta=0.0,
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

    candidate_name = (
        f"lowrank_{safe_name(args.source_label)}_to_{safe_name(args.target_label)}_"
        f"{key}_rank{args.rank}_ridge{args.ridge:g}"
    )
    for beta in betas:
        for idx, control_map in enumerate(permutation_maps):
            print(f"Generating permutation[{idx}] beta={beta}", flush=True)
            outputs = generate_many_lowrank(
                model=model,
                tokenizer=tokenizer,
                prompts=prompt_texts,
                layers_mod=layers_mod,
                layer=args.layer,
                lowrank_map=control_map,
                beta=beta,
                position=args.position,
                max_new_tokens=args.max_new_tokens,
                batch_size=args.generation_batch_size,
            )
            for spec, output in zip(prompt_specs, outputs, strict=True):
                rows.append(
                    row_for_generation(
                        spec=spec,
                        condition="random_unit",
                        candidate="permutation_lowrank",
                        alpha=beta,
                        effective_alpha=beta,
                        random_index=idx,
                        layer=args.layer,
                        output=output,
                    )
                )
        print(f"Generating mean-delta source control beta={beta}", flush=True)
        outputs = generate_many_lowrank(
            model=model,
            tokenizer=tokenizer,
            prompts=prompt_texts,
            layers_mod=layers_mod,
            layer=args.layer,
            lowrank_map=mean_map,
            beta=beta,
            position=args.position,
            max_new_tokens=args.max_new_tokens,
            batch_size=args.generation_batch_size,
        )
        for spec, output in zip(prompt_specs, outputs, strict=True):
            rows.append(
                row_for_generation(
                    spec=spec,
                    condition="source_control",
                    candidate="mean_delta_lowrank",
                    alpha=beta,
                    effective_alpha=beta,
                    random_index=None,
                    layer=args.layer,
                    output=output,
                )
            )
        print(f"Generating candidate {candidate_name} beta={beta}", flush=True)
        outputs = generate_many_lowrank(
            model=model,
            tokenizer=tokenizer,
            prompts=prompt_texts,
            layers_mod=layers_mod,
            layer=args.layer,
            lowrank_map=candidate_map,
            beta=beta,
            position=args.position,
            max_new_tokens=args.max_new_tokens,
            batch_size=args.generation_batch_size,
        )
        for spec, output in zip(prompt_specs, outputs, strict=True):
            rows.append(
                row_for_generation(
                    spec=spec,
                    condition="sae_poke",
                    candidate=candidate_name,
                    alpha=beta,
                    effective_alpha=beta,
                    random_index=None,
                    layer=args.layer,
                    output=output,
                )
            )

    add_base_deltas(rows)
    manifest = {
        "started_at": started,
        "finished_at": now_iso(),
        "model_path": str(args.model_path),
        "replay_run": str(args.replay_run),
        "output_dir": str(out_dir),
        "feature_key": key,
        "pair_field": args.pair_field,
        "label_field": args.label_field,
        "source_label": args.source_label,
        "target_label": args.target_label,
        "rank": args.rank,
        "ridge": args.ridge,
        "betas": betas,
        "permutation_controls": args.permutation_controls,
        "position": args.position,
        "prompt_bank": str(args.prompt_bank),
        "prompt_ids": [spec.prompt_id for spec in prompt_specs],
        "prompt_keys": [spec.prompt_key for spec in prompt_specs],
        "max_new_tokens": args.max_new_tokens,
        **qwen_budget_metadata(args.max_new_tokens),
        "generation_batch_size": args.generation_batch_size,
        "seed": args.seed,
    }
    write_json(out_dir / "manifest.json", manifest)
    write_jsonl(out_dir / "generations.jsonl", rows)
    write_jsonl(out_dir / "score_summary.jsonl", aggregate_frame_scores(rows))
    write_jsonl(out_dir / "candidate_vs_random.jsonl", candidate_vs_random(rows))
    write_jsonl(out_dir / "candidate_vs_prompt_matched_random.jsonl", candidate_vs_prompt_matched_random(rows))
    write_lowrank_report(out_dir / "report.md", manifest=manifest, rows=rows)

    del model, tokenizer, layers_mod
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(f"Wrote {out_dir / 'report.md'}", flush=True)


if __name__ == "__main__":
    main()
