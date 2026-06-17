#!/usr/bin/env python3
"""Two-stage no-mask thinking audit for learned low-rank SCOTUS maps."""

from __future__ import annotations

import argparse
import gc
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

from poke_scotus_lowrank_replay import TorchLowRankMap, install_lowrank_hook
from poke_scotus_sae_layers import (
    DEFAULT_MODEL,
    DEFAULT_OUTPUT_ROOT,
    first_parameter_device,
    load_model_and_tokenizer,
    load_prompt_specs,
    now_iso,
    parse_float_list,
    select_prompt_specs,
    transformer_layers,
    write_json,
    write_jsonl,
)
from poke_scotus_thinking_bundle import (
    add_segment_base_deltas,
    aggregate_segments,
    compare_candidate_to_random,
    row_for_two_stage,
    write_report,
)
from qwen_eval_budget import (
    DEFAULT_COMPLETE_ANSWER_TOKENS,
    MIN_COMPLETE_ANSWER_TOKENS,
    enforce_complete_answer_budget,
    enforce_complete_thinking_budget,
    qwen_thinking_answer_budget_metadata,
)
from run_scotus_thinking_smoke import IMITATION_RE, format_chat, strip_generation_specials
from train_scotus_replay_lowrank_intervention import (
    DEFAULT_REPLAY_RUN,
    feature_key,
    fit_lowrank,
    load_pair_batch,
    read_jsonl,
    safe_name,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PROMPT_BANK = PROJECT_ROOT / "data" / "scotus" / "scotus_article3_private_public_poke_prompts_v2.jsonl"


@torch.inference_mode()
def generate_continuation(
    *,
    model: torch.nn.Module,
    tokenizer: Any,
    prompt: str,
    max_new_tokens: int,
) -> dict[str, Any]:
    input_device = first_parameter_device(model)
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        add_special_tokens=False,
        truncation=True,
        max_length=4096,
    )
    inputs = {key: value.to(input_device) for key, value in inputs.items()}
    output = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        use_cache=True,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
    generated = output[0, inputs["input_ids"].shape[-1] :]
    return {
        "raw_text": tokenizer.decode(generated, skip_special_tokens=False).strip(),
        "text": tokenizer.decode(generated, skip_special_tokens=True).strip(),
        "generated_tokens": int(generated.numel()),
        "prompt_tokens": int(inputs["input_ids"].shape[-1]),
    }


@torch.inference_mode()
def generate_two_stage_lowrank(
    *,
    model: torch.nn.Module,
    tokenizer: Any,
    prompt: str,
    layers_mod: Any,
    layer: int | None,
    lowrank_map: TorchLowRankMap | None,
    beta: float,
    position: str,
    thought_tokens: int,
    answer_tokens: int,
) -> dict[str, Any]:
    handle = None
    if layer is not None and lowrank_map is not None and beta != 0.0:
        handle = install_lowrank_hook(layers_mod, layer=layer, lowrank_map=lowrank_map, beta=beta, position=position)
    try:
        chat = format_chat(tokenizer, prompt, enable_thinking=True)
        prefilled_open_think = chat.rstrip().endswith("<think>")
        thought_output = generate_continuation(
            model=model,
            tokenizer=tokenizer,
            prompt=chat,
            max_new_tokens=thought_tokens,
        )
        thought_raw = strip_generation_specials(str(thought_output["raw_text"]))
        thought_text, separator, answer_prefix = thought_raw.partition("</think>")
        model_closed_thinking = bool(separator)
        thought_text = thought_text.strip()
        answer_prefix = strip_generation_specials(answer_prefix)
        answer_prompt = f"{chat}{thought_text}\n</think>\n\n{answer_prefix}"
        answer_output = generate_continuation(
            model=model,
            tokenizer=tokenizer,
            prompt=answer_prompt,
            max_new_tokens=answer_tokens,
        )
        answer_text = (answer_prefix + "\n" + strip_generation_specials(str(answer_output["raw_text"]))).strip()
        full_text = f"<think>\n{thought_text}\n</think>\n\n{answer_text}".strip()
        return {
            "prefilled_open_think": prefilled_open_think,
            "model_closed_thinking": model_closed_thinking,
            "mechanically_closed_for_answer": True,
            "thinking": thought_text,
            "answer": answer_text,
            "full_text": full_text,
            "thinking_generated_tokens": int(thought_output["generated_tokens"]),
            "answer_generated_tokens": int(answer_output["generated_tokens"]),
            "thinking_prompt_tokens": int(thought_output["prompt_tokens"]),
            "answer_prompt_tokens": int(answer_output["prompt_tokens"]),
            "thinking_nonempty": bool(thought_text),
            "answer_nonempty": bool(answer_text),
            "thinking_imitation_markers": sorted(set(IMITATION_RE.findall(thought_text))),
            "answer_imitation_markers": sorted(set(IMITATION_RE.findall(answer_text))),
        }
    finally:
        if handle is not None:
            handle.remove()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--replay-run", type=Path, default=DEFAULT_REPLAY_RUN)
    parser.add_argument("--prompt-bank", type=Path, default=DEFAULT_PROMPT_BANK)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--region", default="assistant_all")
    parser.add_argument("--layer", type=int, default=4)
    parser.add_argument("--pair-field", default="pair_id")
    parser.add_argument("--label-field", default="frame_label")
    parser.add_argument("--source-label", default="article3_public_rights")
    parser.add_argument("--target-label", default="article3_private_rights")
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--ridge", type=float, default=0.01)
    parser.add_argument("--betas", default="0.25")
    parser.add_argument("--prompt-ids", default="1,4")
    parser.add_argument("--max-prompts", type=int, default=2)
    parser.add_argument("--permutation-controls", type=int, default=2)
    parser.add_argument("--position", choices=["last", "all"], default="last")
    parser.add_argument("--thought-tokens", type=int, default=DEFAULT_COMPLETE_ANSWER_TOKENS)
    parser.add_argument("--answer-tokens", type=int, default=DEFAULT_COMPLETE_ANSWER_TOKENS)
    parser.add_argument(
        "--allow-short-thinking-budget",
        action="store_true",
        help=(
            f"Permit visible-thinking budgets below {MIN_COMPLETE_ANSWER_TOKENS} tokens. "
            "Short-budget thinking runs are smoke/debug only and must not be used for promotion."
        ),
    )
    parser.add_argument(
        "--allow-short-answer-budget",
        action="store_true",
        help=(
            f"Permit answer budgets below {MIN_COMPLETE_ANSWER_TOKENS} tokens. "
            "Short-budget runs are smoke/debug only and must not be used for promotion."
        ),
    )
    parser.add_argument("--device-map", default="single")
    parser.add_argument("--seed", type=int, default=20260510)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    enforce_complete_thinking_budget(
        args.thought_tokens,
        allow_short=args.allow_short_thinking_budget,
        purpose="SCOTUS low-rank visible-thinking evaluator run",
    )
    enforce_complete_answer_budget(
        args.answer_tokens,
        allow_short=args.allow_short_answer_budget,
        label="answer_tokens",
        purpose="SCOTUS low-rank answer evaluator run",
    )
    started = now_iso()
    out_dir = args.output_root / f"scotus_thinking_lowrank_poke_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
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
    candidate_map = TorchLowRankMap(fit_lowrank(train_batch, rank=args.rank, ridge=args.ridge, permutation_seed=None))
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
    betas = parse_float_list(args.betas)
    candidate_name = (
        f"lowrank_{safe_name(args.source_label)}_to_{safe_name(args.target_label)}_"
        f"{key}_rank{args.rank}_ridge{args.ridge:g}"
    )
    print(f"Loaded {len(prompt_specs)} prompts and {len(permutation_maps)} permutation controls", flush=True)

    tokenizer, model = load_model_and_tokenizer(args.model_path, args.device_map)
    layers_mod = transformer_layers(model)
    rows: list[dict[str, Any]] = []

    print("Generating two-stage baseline", flush=True)
    for spec in prompt_specs:
        output = generate_two_stage_lowrank(
            model=model,
            tokenizer=tokenizer,
            prompt=spec.prompt,
            layers_mod=layers_mod,
            layer=None,
            lowrank_map=None,
            beta=0.0,
            position=args.position,
            thought_tokens=args.thought_tokens,
            answer_tokens=args.answer_tokens,
        )
        rows.append(
            row_for_two_stage(
                spec=spec,
                condition="base",
                candidate=None,
                alpha=0.0,
                random_index=None,
                layer=None,
                output=output,
            )
        )

    for beta in betas:
        for idx, control_map in enumerate(permutation_maps):
            print(f"Generating two-stage permutation[{idx}] beta={beta}", flush=True)
            for spec in prompt_specs:
                output = generate_two_stage_lowrank(
                    model=model,
                    tokenizer=tokenizer,
                    prompt=spec.prompt,
                    layers_mod=layers_mod,
                    layer=args.layer,
                    lowrank_map=control_map,
                    beta=float(beta),
                    position=args.position,
                    thought_tokens=args.thought_tokens,
                    answer_tokens=args.answer_tokens,
                )
                rows.append(
                    row_for_two_stage(
                        spec=spec,
                        condition="random_unit",
                        candidate="permutation_lowrank",
                        alpha=float(beta),
                        random_index=idx,
                        layer=args.layer,
                        output=output,
                    )
                )
        print(f"Generating two-stage mean-delta source control beta={beta}", flush=True)
        for spec in prompt_specs:
            output = generate_two_stage_lowrank(
                model=model,
                tokenizer=tokenizer,
                prompt=spec.prompt,
                layers_mod=layers_mod,
                layer=args.layer,
                lowrank_map=mean_map,
                beta=float(beta),
                position=args.position,
                thought_tokens=args.thought_tokens,
                answer_tokens=args.answer_tokens,
            )
            rows.append(
                row_for_two_stage(
                    spec=spec,
                    condition="source_control",
                    candidate="mean_delta_lowrank",
                    alpha=float(beta),
                    random_index=None,
                    layer=args.layer,
                    output=output,
                )
            )
        print(f"Generating two-stage candidate {candidate_name} beta={beta}", flush=True)
        for spec in prompt_specs:
            output = generate_two_stage_lowrank(
                model=model,
                tokenizer=tokenizer,
                prompt=spec.prompt,
                layers_mod=layers_mod,
                layer=args.layer,
                lowrank_map=candidate_map,
                beta=float(beta),
                position=args.position,
                thought_tokens=args.thought_tokens,
                answer_tokens=args.answer_tokens,
            )
            rows.append(
                row_for_two_stage(
                    spec=spec,
                    condition="sae_poke",
                    candidate=candidate_name,
                    alpha=float(beta),
                    random_index=None,
                    layer=args.layer,
                    output=output,
                )
            )

    add_segment_base_deltas(rows)
    summaries = aggregate_segments(rows)
    comparisons = compare_candidate_to_random(rows)
    manifest = {
        "started_at": started,
        "finished_at": now_iso(),
        "model_path": str(args.model_path),
        "replay_run": str(args.replay_run),
        "output_dir": str(out_dir),
        "feature_key": key,
        "layers": [args.layer],
        "pair_field": args.pair_field,
        "label_field": args.label_field,
        "source_label": args.source_label,
        "target_label": args.target_label,
        "candidate_names": [candidate_name],
        "rank": args.rank,
        "ridge": args.ridge,
        "betas": betas,
        "alphas": betas,
        "permutation_controls": args.permutation_controls,
        "random_controls": args.permutation_controls,
        "position": args.position,
        "prompt_bank": str(args.prompt_bank),
        "prompt_ids": [spec.prompt_id for spec in prompt_specs],
        "prompt_keys": [spec.prompt_key for spec in prompt_specs],
        "thought_tokens": args.thought_tokens,
        "answer_tokens": args.answer_tokens,
        **qwen_thinking_answer_budget_metadata(args.thought_tokens, args.answer_tokens),
        "seed": args.seed,
    }
    write_json(out_dir / "manifest.json", manifest)
    write_jsonl(out_dir / "generations.jsonl", rows)
    write_jsonl(out_dir / "segment_score_summary.jsonl", summaries)
    write_jsonl(out_dir / "candidate_vs_prompt_matched_random.jsonl", comparisons)
    write_report(out_dir / "report.md", manifest=manifest, summaries=summaries, comparisons=comparisons, rows=rows)

    del model, tokenizer, layers_mod
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(f"Wrote {out_dir / 'report.md'}", flush=True)


if __name__ == "__main__":
    main()
