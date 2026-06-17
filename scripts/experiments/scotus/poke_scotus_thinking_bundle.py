#!/usr/bin/env python3
"""Two-stage no-mask thinking audit for controlled SCOTUS bundle pokes.

The usual Qwen thinking template pre-fills ``<think>`` and may spend the full
token budget inside the reasoning trace. This runner keeps that useful trace,
mechanically closes it, and then asks the same model state to continue with the
final answer. Candidate and random-control bundles are active during both
generated thought and generated answer tokens.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import torch

from poke_scotus_controlled_replay_bundle import (
    DEFAULT_MODEL,
    DEFAULT_PROMPT_BANK,
    DEFAULT_REPLAY_RUN,
    build_controlled_bundle,
    install_bundle_hooks,
    random_bundle,
)
from poke_scotus_sae_layers import (
    DEFAULT_OUTPUT_ROOT,
    first_parameter_device,
    frame_eval_for_prompt,
    load_model_and_tokenizer,
    load_prompt_specs,
    now_iso,
    parse_float_list,
    parse_int_list,
    select_prompt_specs,
    tag_frames,
    transformer_layers,
    write_json,
    write_jsonl,
)
from rescore_scotus_frame_propositions import score_frames
from run_scotus_thinking_smoke import IMITATION_RE, format_chat, strip_generation_specials
from qwen_eval_budget import (
    DEFAULT_COMPLETE_ANSWER_TOKENS,
    MIN_COMPLETE_ANSWER_TOKENS,
    enforce_complete_answer_budget,
    enforce_complete_thinking_budget,
    qwen_thinking_answer_budget_metadata,
)


SEGMENTS = ("thinking", "answer")


def mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def stdev(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    avg = mean(values)
    return float((sum((value - avg) ** 2 for value in values) / (len(values) - 1)) ** 0.5)


def md_table(headers: list[str], rows: list[list[Any]]) -> list[str]:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(item) for item in row) + " |")
    return lines


def fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def clean_snippet(text: str, max_chars: int = 500) -> str:
    cleaned = re.sub(r"\s+", " ", text).strip()
    if len(cleaned) <= max_chars:
        return cleaned
    cut = cleaned[: max_chars + 1]
    if " " in cut:
        cut = cut[: cut.rfind(" ")]
    return cut.rstrip() + "..."


def segment_scores(spec: Any, text: str) -> dict[str, Any]:
    lexical_scores = tag_frames(text)
    prop_scores, prop_evidence = score_frames(text)
    return {
        "frame_scores": lexical_scores,
        "frame_eval": frame_eval_for_prompt(spec, lexical_scores),
        "proposition_frame_scores": prop_scores,
        "proposition_frame_evidence": prop_evidence,
        "proposition_frame_eval": frame_eval_for_prompt(spec, prop_scores),
    }


def add_segment_base_deltas(rows: list[dict[str, Any]]) -> None:
    for segment in SEGMENTS:
        base_by_prompt = {
            row["prompt_id"]: row[f"{segment}_proposition_frame_eval"]
            for row in rows
            if row["condition"] == "base"
        }
        for row in rows:
            base = base_by_prompt.get(row["prompt_id"])
            if base is None:
                continue
            current = row[f"{segment}_proposition_frame_eval"]
            for key in ("target_hits", "contrast_hits", "off_domain_hits", "total_frame_hits"):
                current[f"delta_{key}_vs_base"] = float(current[key] - base[key])
            current["delta_target_minus_contrast_vs_base"] = float(
                (current["target_hits"] - current["contrast_hits"])
                - (base["target_hits"] - base["contrast_hits"])
            )


def aggregate_segments(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for segment in SEGMENTS:
        groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            key = (
                segment,
                row["condition"],
                row.get("candidate"),
                None if row["condition"] == "base" else row.get("alpha"),
            )
            groups[key].append(row)
        for (seg, condition, candidate, alpha), group_rows in sorted(groups.items(), key=str):
            evals = [row[f"{seg}_proposition_frame_eval"] for row in group_rows]
            summaries.append(
                {
                    "segment": seg,
                    "condition": condition,
                    "candidate": candidate,
                    "alpha": alpha,
                    "n": len(group_rows),
                    "prompt_count": len({row["prompt_id"] for row in group_rows}),
                    "target_present_rate": mean([1.0 if item["target_present"] else 0.0 for item in evals]),
                    "contrast_present_rate": mean([1.0 if item["contrast_present"] else 0.0 for item in evals]),
                    "mean_target_hits": mean([float(item["target_hits"]) for item in evals]),
                    "mean_contrast_hits": mean([float(item["contrast_hits"]) for item in evals]),
                    "mean_delta_target_hits_vs_base": mean(
                        [float(item.get("delta_target_hits_vs_base", 0.0)) for item in evals]
                    ),
                    "mean_delta_target_minus_contrast_vs_base": mean(
                        [float(item.get("delta_target_minus_contrast_vs_base", 0.0)) for item in evals]
                    ),
                    "model_closed_thinking_rate": mean(
                        [1.0 if row["model_closed_thinking"] else 0.0 for row in group_rows]
                    ),
                    "answer_nonempty_rate": mean([1.0 if row["answer_nonempty"] else 0.0 for row in group_rows]),
                    "imitation_marker_rate": mean(
                        [1.0 if row["thinking_imitation_markers"] or row["answer_imitation_markers"] else 0.0 for row in group_rows]
                    ),
                }
            )
    return summaries


def compare_candidate_to_random(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    comparisons: list[dict[str, Any]] = []
    for segment in SEGMENTS:
        random_by_prompt_alpha: dict[tuple[int, float], list[float]] = defaultdict(list)
        random_net_by_prompt_alpha: dict[tuple[int, float], list[float]] = defaultdict(list)
        for row in rows:
            if row["condition"] != "random_unit":
                continue
            key = (int(row["prompt_id"]), float(row["alpha"]))
            eval_item = row[f"{segment}_proposition_frame_eval"]
            random_by_prompt_alpha[key].append(float(eval_item.get("delta_target_hits_vs_base", 0.0)))
            random_net_by_prompt_alpha[key].append(
                float(eval_item.get("delta_target_minus_contrast_vs_base", 0.0))
            )

        candidate_groups: dict[tuple[str, float], list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            if row["condition"] != "sae_poke" or not row.get("candidate"):
                continue
            candidate_groups[(str(row["candidate"]), float(row["alpha"]))].append(row)

        for (candidate, alpha), group_rows in sorted(candidate_groups.items(), key=str):
            adjusted: list[float] = []
            adjusted_net: list[float] = []
            candidate_values: list[float] = []
            candidate_net_values: list[float] = []
            random_means: list[float] = []
            random_net_means: list[float] = []
            wins_mean = 0
            wins_strongest = 0
            wins_net_mean = 0
            wins_net_strongest = 0
            matched = 0
            for row in group_rows:
                key = (int(row["prompt_id"]), float(alpha))
                random_values = random_by_prompt_alpha.get(key, [])
                random_net_values = random_net_by_prompt_alpha.get(key, [])
                if not random_values or not random_net_values:
                    continue
                eval_item = row[f"{segment}_proposition_frame_eval"]
                candidate_value = float(eval_item.get("delta_target_hits_vs_base", 0.0))
                candidate_net_value = float(eval_item.get("delta_target_minus_contrast_vs_base", 0.0))
                random_mean = mean(random_values)
                random_net_mean = mean(random_net_values)
                matched += 1
                candidate_values.append(candidate_value)
                candidate_net_values.append(candidate_net_value)
                random_means.append(random_mean)
                random_net_means.append(random_net_mean)
                adjusted.append(candidate_value - random_mean)
                adjusted_net.append(candidate_net_value - random_net_mean)
                wins_mean += int(candidate_value > random_mean)
                wins_strongest += int(candidate_value > max(random_values))
                wins_net_mean += int(candidate_net_value > random_net_mean)
                wins_net_strongest += int(candidate_net_value > max(random_net_values))
            comparisons.append(
                {
                    "segment": segment,
                    "candidate": candidate,
                    "alpha": alpha,
                    "n": matched,
                    "candidate_mean_delta_target_hits_vs_base": mean(candidate_values),
                    "prompt_random_mean_delta_target_hits_vs_base": mean(random_means),
                    "mean_prompt_matched_delta_minus_random": mean(adjusted),
                    "sd_prompt_matched_delta_minus_random": stdev(adjusted),
                    "prompt_win_rate_vs_random_mean": 0.0 if matched == 0 else float(wins_mean / matched),
                    "prompt_win_rate_vs_strongest_random": 0.0 if matched == 0 else float(wins_strongest / matched),
                    "candidate_mean_delta_target_minus_contrast_vs_base": mean(candidate_net_values),
                    "prompt_random_mean_delta_target_minus_contrast_vs_base": mean(random_net_means),
                    "mean_prompt_matched_net_delta_minus_random": mean(adjusted_net),
                    "sd_prompt_matched_net_delta_minus_random": stdev(adjusted_net),
                    "prompt_net_win_rate_vs_random_mean": 0.0 if matched == 0 else float(wins_net_mean / matched),
                    "prompt_net_win_rate_vs_strongest_random": 0.0
                    if matched == 0
                    else float(wins_net_strongest / matched),
                }
            )
    return comparisons


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
def generate_two_stage(
    *,
    model: torch.nn.Module,
    tokenizer: Any,
    prompt: str,
    layers_mod: Any,
    layer_to_vec: dict[int, torch.Tensor] | None,
    alpha: float,
    scale_factors: dict[int, float],
    position: str,
    thought_tokens: int,
    answer_tokens: int,
) -> dict[str, Any]:
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
        for handle in handles:
            handle.remove()


def row_for_two_stage(
    *,
    spec: Any,
    condition: str,
    candidate: str | None,
    alpha: float,
    random_index: int | None,
    layer: int | None,
    output: dict[str, Any],
) -> dict[str, Any]:
    thinking_scores = segment_scores(spec, str(output["thinking"]))
    answer_scores = segment_scores(spec, str(output["answer"]))
    full_scores = segment_scores(spec, str(output["full_text"]))
    return {
        "prompt_id": spec.prompt_id,
        "prompt_key": spec.prompt_key,
        "issue_area": spec.issue_area,
        "prompt": spec.prompt,
        "condition": condition,
        "candidate": candidate,
        "alpha": float(alpha),
        "random_index": random_index,
        "layer": layer,
        "expected_frames": list(spec.expected_frames),
        "contrast_frames": list(spec.contrast_frames),
        "domain_frames": list(spec.domain_frames),
        **output,
        "thinking_frame_scores": thinking_scores["frame_scores"],
        "thinking_frame_eval": thinking_scores["frame_eval"],
        "thinking_proposition_frame_scores": thinking_scores["proposition_frame_scores"],
        "thinking_proposition_frame_evidence": thinking_scores["proposition_frame_evidence"],
        "thinking_proposition_frame_eval": thinking_scores["proposition_frame_eval"],
        "answer_frame_scores": answer_scores["frame_scores"],
        "answer_frame_eval": answer_scores["frame_eval"],
        "answer_proposition_frame_scores": answer_scores["proposition_frame_scores"],
        "answer_proposition_frame_evidence": answer_scores["proposition_frame_evidence"],
        "answer_proposition_frame_eval": answer_scores["proposition_frame_eval"],
        "full_text_frame_scores": full_scores["frame_scores"],
        "full_text_frame_eval": full_scores["frame_eval"],
        "full_text_proposition_frame_scores": full_scores["proposition_frame_scores"],
        "full_text_proposition_frame_evidence": full_scores["proposition_frame_evidence"],
        "full_text_proposition_frame_eval": full_scores["proposition_frame_eval"],
    }


def write_report(
    path: Path,
    *,
    manifest: dict[str, Any],
    summaries: list[dict[str, Any]],
    comparisons: list[dict[str, Any]],
    rows: list[dict[str, Any]],
) -> None:
    lines = [
        "# SCOTUS Two-Stage Thinking Bundle Poke",
        "",
        "## Configuration",
        "",
        f"- Model: `{manifest['model_path']}`",
        f"- Replay run: `{manifest['replay_run']}`",
        f"- Prompt bank: `{manifest['prompt_bank']}`",
        f"- Layers: `{', '.join(str(item) for item in manifest['layers'])}`",
        f"- Position: `{manifest['position']}`",
        f"- Alphas: `{', '.join(str(item) for item in manifest['alphas'])}`",
        f"- Random controls: `{manifest['random_controls']}`",
        f"- Thought/answer token budgets: `{manifest['thought_tokens']}`/`{manifest['answer_tokens']}`",
        f"- Short answer-budget smoke: `{manifest['short_answer_budget']}`",
        f"- Prompts: `{len(manifest['prompt_ids'])}`",
        "",
        "## Segment Summary",
        "",
    ]
    lines.extend(
        md_table(
            [
                "segment",
                "condition",
                "alpha",
                "n",
                "target_delta",
                "net_delta",
                "target_rate",
                "contrast_rate",
                "answer_rate",
                "mask_rate",
            ],
            [
                [
                    item["segment"],
                    item["condition"],
                    fmt(item["alpha"]),
                    item["n"],
                    fmt(item["mean_delta_target_hits_vs_base"]),
                    fmt(item["mean_delta_target_minus_contrast_vs_base"]),
                    fmt(item["target_present_rate"]),
                    fmt(item["contrast_present_rate"]),
                    fmt(item["answer_nonempty_rate"]),
                    fmt(item["imitation_marker_rate"]),
                ]
                for item in summaries
            ],
        )
    )
    lines.extend(["", "## Candidate vs Prompt-Matched Random", ""])
    lines.extend(
        md_table(
            [
                "segment",
                "alpha",
                "n",
                "target-minus-random",
                "net-minus-random",
                "target strongest wins",
                "net strongest wins",
            ],
            [
                [
                    item["segment"],
                    fmt(item["alpha"]),
                    item["n"],
                    fmt(item["mean_prompt_matched_delta_minus_random"]),
                    fmt(item["mean_prompt_matched_net_delta_minus_random"]),
                    fmt(item["prompt_win_rate_vs_strongest_random"]),
                    fmt(item["prompt_net_win_rate_vs_strongest_random"]),
                ]
                for item in comparisons
            ],
        )
    )
    lines.extend(
        [
            "",
            "## Samples",
            "",
        ]
    )
    for row in rows[: min(12, len(rows))]:
        lines.extend(
            [
                f"### {row['prompt_key']} / {row['condition']} / alpha {row['alpha']}",
                "",
                f"- random index: `{row.get('random_index')}`",
                f"- model closed thinking: `{row['model_closed_thinking']}`",
                f"- answer nonempty: `{row['answer_nonempty']}`",
                f"- thinking imitation markers: `{', '.join(row['thinking_imitation_markers'])}`",
                f"- answer imitation markers: `{', '.join(row['answer_imitation_markers'])}`",
                "",
                "Thinking snippet:",
                "",
                clean_snippet(row["thinking"]) or "[none]",
                "",
                "Answer snippet:",
                "",
                clean_snippet(row["answer"]) or "[none]",
                "",
            ]
        )
    lines.extend(
        [
            "## Decision Rule",
            "",
            "This smoke is not promotable unless the candidate beats strongest random controls on both thinking and answer segments while preserving nonempty answers and avoiding imitation markers.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--replay-run", type=Path, default=DEFAULT_REPLAY_RUN)
    parser.add_argument("--prompt-bank", type=Path, default=DEFAULT_PROMPT_BANK)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--region", default="assistant_all")
    parser.add_argument("--layers", default="4,8,12,16")
    parser.add_argument("--fit-splits", default="train")
    parser.add_argument("--pair-field", default="pair_id")
    parser.add_argument("--label-field", default="frame_label")
    parser.add_argument("--target-label", default="article3_private_rights")
    parser.add_argument("--reference-label", default="article3_public_rights")
    parser.add_argument("--mode", choices=["mean", "pc1"], default="mean")
    parser.add_argument("--alphas", default="0.01")
    parser.add_argument("--position", choices=["last", "all", "decode"], default="decode")
    parser.add_argument("--prompt-ids", default="0")
    parser.add_argument("--max-prompts", type=int, default=1)
    parser.add_argument("--random-controls", type=int, default=2)
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
    parser.add_argument("--seed", type=int, default=20260506)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    enforce_complete_thinking_budget(
        args.thought_tokens,
        allow_short=args.allow_short_thinking_budget,
        purpose="SCOTUS two-stage visible-thinking evaluator run",
    )
    enforce_complete_answer_budget(
        args.answer_tokens,
        allow_short=args.allow_short_answer_budget,
        label="answer_tokens",
        purpose="SCOTUS two-stage answer evaluator run",
    )
    started = now_iso()
    out_dir = args.output_root / f"scotus_thinking_bundle_poke_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)

    layers = parse_int_list(args.layers)
    alphas = parse_float_list(args.alphas)
    fit_splits = {item.strip() for item in args.fit_splits.split(",") if item.strip()}
    candidate_bundle, direction_meta, scale_factors = build_controlled_bundle(
        replay_run=args.replay_run,
        region=args.region,
        layers=layers,
        fit_splits=fit_splits,
        pair_field=args.pair_field,
        label_field=args.label_field,
        target_label=args.target_label,
        reference_label=args.reference_label,
        mode=args.mode,
    )
    candidate_name = (
        f"controlled_{args.mode}_{args.region}_"
        f"{args.reference_label}_to_{args.target_label}_L{'_'.join(str(layer) for layer in layers)}"
    )
    layer_dim = int(next(iter(candidate_bundle.values())).numel())
    random_bundles = [random_bundle(layers, layer_dim, args.seed + idx * 100_003) for idx in range(args.random_controls)]

    prompt_specs = select_prompt_specs(load_prompt_specs(args.prompt_bank), args.prompt_ids, args.max_prompts)
    print(f"Loaded {len(prompt_specs)} prompts", flush=True)
    print(f"Built thinking bundle candidate {candidate_name}", flush=True)

    tokenizer, model = load_model_and_tokenizer(args.model_path, args.device_map)
    layers_mod = transformer_layers(model)

    rows: list[dict[str, Any]] = []
    print("Generating two-stage baseline", flush=True)
    for spec in prompt_specs:
        output = generate_two_stage(
            model=model,
            tokenizer=tokenizer,
            prompt=spec.prompt,
            layers_mod=layers_mod,
            layer_to_vec=None,
            alpha=0.0,
            scale_factors=scale_factors,
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

    for alpha in alphas:
        for random_idx, bundle in enumerate(random_bundles):
            print(f"Generating two-stage random_bundle[{random_idx}] alpha={alpha}", flush=True)
            for spec in prompt_specs:
                output = generate_two_stage(
                    model=model,
                    tokenizer=tokenizer,
                    prompt=spec.prompt,
                    layers_mod=layers_mod,
                    layer_to_vec=bundle,
                    alpha=float(alpha),
                    scale_factors=scale_factors,
                    position=args.position,
                    thought_tokens=args.thought_tokens,
                    answer_tokens=args.answer_tokens,
                )
                rows.append(
                    row_for_two_stage(
                        spec=spec,
                        condition="random_unit",
                        candidate="random_bundle",
                        alpha=float(alpha),
                        random_index=random_idx,
                        layer=layers[0],
                        output=output,
                    )
                )

        print(f"Generating two-stage {candidate_name} alpha={alpha}", flush=True)
        for spec in prompt_specs:
            output = generate_two_stage(
                model=model,
                tokenizer=tokenizer,
                prompt=spec.prompt,
                layers_mod=layers_mod,
                layer_to_vec=candidate_bundle,
                alpha=float(alpha),
                scale_factors=scale_factors,
                position=args.position,
                thought_tokens=args.thought_tokens,
                answer_tokens=args.answer_tokens,
            )
            rows.append(
                row_for_two_stage(
                    spec=spec,
                    condition="sae_poke",
                    candidate=candidate_name,
                    alpha=float(alpha),
                    random_index=None,
                    layer=layers[0],
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
        "candidate_names": [candidate_name],
        "direction_source": "controlled_replay_bundle",
        "prompt_bank": str(args.prompt_bank),
        "alphas": alphas,
        "random_controls": int(args.random_controls),
        "position": args.position,
        "thought_tokens": int(args.thought_tokens),
        "answer_tokens": int(args.answer_tokens),
        **qwen_thinking_answer_budget_metadata(args.thought_tokens, args.answer_tokens),
        "device_map": args.device_map,
        "seed": args.seed,
        "prompt_ids": [spec.prompt_id for spec in prompt_specs],
        "prompt_keys": [spec.prompt_key for spec in prompt_specs],
        "layers": layers,
        "region": args.region,
        "mode": args.mode,
        "fit_splits": sorted(fit_splits),
        "pair_field": args.pair_field,
        "label_field": args.label_field,
        "target_label": args.target_label,
        "reference_label": args.reference_label,
        "scale_factors": {str(layer): scale_factors[layer] for layer in layers},
    }
    write_json(out_dir / "manifest.json", manifest)
    write_jsonl(out_dir / "direction_meta.jsonl", direction_meta)
    write_jsonl(out_dir / "generations.jsonl", rows)
    write_jsonl(out_dir / "segment_score_summary.jsonl", summaries)
    write_jsonl(out_dir / "candidate_vs_prompt_matched_random.jsonl", comparisons)
    write_report(out_dir / "report.md", manifest=manifest, summaries=summaries, comparisons=comparisons, rows=rows)
    print(f"Wrote {out_dir / 'report.md'}", flush=True)


if __name__ == "__main__":
    main()
