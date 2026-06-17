#!/usr/bin/env python3
"""Token-local trace patching for SCOTUS minimal-pair replay states.

The residual-vector and prototype-blend branches found decodable Commerce
answer states without reliable steering. This script tests the next mechanism:
capture a real teacher-forced assistant trace from a minimal-pair replay answer
and, during generation on normal prompts, blend the target residual stream
toward that source trace at the same decode step.
"""

from __future__ import annotations

import argparse
import gc
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm

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
DEFAULT_REPLAY_ROWS = PROJECT_ROOT / "data" / "scotus" / "replay" / "scotus_minpair_replay_examples_20260501.jsonl"
DEFAULT_REPLAY_RUN = PROJECT_ROOT / "sweep_v4" / "scotus_minpair_replay_20260501_100514"
DEFAULT_PROMPT_BANK = PROJECT_ROOT / "data" / "scotus" / "scotus_commerce_pocket_prompts_v1.jsonl"

COMMERCE_LIMITS_EXPECTED = (
    "economic_commerce_limits",
    "economic_federalism_state_regulation",
)
COMMERCE_LIMITS_CONTRAST = (
    "economic_commerce_clause",
    "economic_remedy_damages",
)
COMMERCE_AUTHORITY_EXPECTED = (
    "economic_commerce_clause",
    "economic_remedy_damages",
)
COMMERCE_AUTHORITY_CONTRAST = ("economic_commerce_limits",)


@dataclass(frozen=True)
class ReplaySource:
    example_id: str
    pair_id: str
    fact_id: str
    split: str
    frame_label: str
    prompt: str
    assistant_text: str


@dataclass
class TracePatchState:
    trace: dict[int, torch.Tensor] | None = None
    blend: float = 0.0
    step: int = 0


def parse_int_list(text: str) -> list[int]:
    return [int(item.strip()) for item in text.split(",") if item.strip()]


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


def row_to_source(row: dict[str, Any]) -> ReplaySource:
    prompt = str(row.get("prompt") or "").strip()
    assistant_text = str(row.get("assistant_text") or row.get("text") or "").strip()
    if not prompt or not assistant_text:
        raise ValueError(f"Replay row is missing prompt or assistant text: {row.get('example_id')}")
    return ReplaySource(
        example_id=str(row.get("example_id") or ""),
        pair_id=str(row.get("pair_id") or ""),
        fact_id=str(row.get("fact_id") or ""),
        split=str(row.get("split") or ""),
        frame_label=str(row.get("frame_label") or row.get("justice") or ""),
        prompt=prompt,
        assistant_text=assistant_text,
    )


def choose_source(rows: list[dict[str, Any]], *, example_id: str, label: str, split: str) -> ReplaySource:
    if example_id:
        matches = [row for row in rows if str(row.get("example_id")) == example_id]
        if not matches:
            raise ValueError(f"No replay source with example_id={example_id!r}")
        return row_to_source(matches[0])

    matches = [
        row
        for row in rows
        if str(row.get("frame_label")) == label and (not split or str(row.get("split")) == split)
    ]
    if not matches:
        raise ValueError(f"No replay source for label={label!r}, split={split!r}")
    matches.sort(key=lambda row: str(row.get("example_id") or ""))
    return row_to_source(matches[0])


def override_prompt_scores(specs: list[Any], score_mode: str) -> list[Any]:
    if score_mode == "prompt_expected":
        return specs
    if score_mode == "commerce_limits":
        expected = COMMERCE_LIMITS_EXPECTED
        contrast = COMMERCE_LIMITS_CONTRAST
    elif score_mode == "commerce_authority":
        expected = COMMERCE_AUTHORITY_EXPECTED
        contrast = COMMERCE_AUTHORITY_CONTRAST
    else:
        raise ValueError(f"Unknown score mode: {score_mode}")

    rewritten: list[Any] = []
    for spec in specs:
        rewritten.append(
            type(spec)(
                prompt_id=spec.prompt_id,
                prompt_key=spec.prompt_key,
                issue_area=spec.issue_area,
                prompt=spec.prompt,
                expected_frames=expected,
                contrast_frames=contrast,
                domain_frames=spec.domain_frames,
            )
        )
    return rewritten


def source_name(prefix: str, source: ReplaySource, layers: list[int]) -> str:
    layer_text = "-".join(f"L{layer:02d}" for layer in layers)
    example = source.example_id.replace("|", "_").replace("/", "_")
    return f"{prefix}_{source.frame_label}_{layer_text}_{example}"


def install_capture_hooks(
    layers_mod: Any,
    *,
    layers: list[int],
    positions: torch.Tensor,
    captured: dict[int, torch.Tensor],
) -> list[Any]:
    handles: list[Any] = []
    cpu_positions = positions.detach().cpu()

    for layer in layers:

        def make_hook(layer_idx: int) -> Any:
            def hook(_module: torch.nn.Module, _inp: tuple[Any, ...], out: Any) -> Any:
                hidden = out[0] if isinstance(out, tuple) else out
                captured[layer_idx] = hidden[0, cpu_positions.to(hidden.device), :].detach().float().cpu().contiguous()
                return out

            return hook

        handles.append(layers_mod[layer].register_forward_hook(make_hook(layer)))
    return handles


@torch.inference_mode()
def capture_source_trace(
    *,
    model: torch.nn.Module,
    tokenizer: Any,
    layers_mod: Any,
    source: ReplaySource,
    layers: list[int],
    max_steps: int,
) -> dict[int, torch.Tensor]:
    chat = format_chat(tokenizer, source.prompt)
    prompt_ids = tokenizer(chat, add_special_tokens=False).input_ids
    assistant_ids = tokenizer(source.assistant_text, add_special_tokens=False).input_ids
    if not prompt_ids or not assistant_ids:
        raise RuntimeError(f"Cannot capture empty trace for {source.example_id}")

    trace_len = min(max_steps, len(assistant_ids))
    if trace_len <= 0:
        raise RuntimeError(f"Trace length is zero for {source.example_id}")
    full_ids = torch.tensor([prompt_ids + assistant_ids], dtype=torch.long)
    positions = torch.arange(len(prompt_ids) - 1, len(prompt_ids) - 1 + trace_len, dtype=torch.long)
    input_device = first_parameter_device(model)
    full_ids = full_ids.to(input_device)
    attention_mask = torch.ones_like(full_ids, device=input_device)

    captured: dict[int, torch.Tensor] = {}
    handles = install_capture_hooks(layers_mod, layers=layers, positions=positions, captured=captured)
    try:
        _ = model(input_ids=full_ids, attention_mask=attention_mask, use_cache=False)
    finally:
        for handle in handles:
            handle.remove()

    missing = sorted(set(layers) - set(captured))
    if missing:
        raise RuntimeError(f"Did not capture source trace layers: {missing}")
    return captured


def random_trace_like(trace: dict[int, torch.Tensor], *, seed: int) -> dict[int, torch.Tensor]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    random_trace: dict[int, torch.Tensor] = {}
    for layer, values in trace.items():
        noise = torch.randn(values.shape, generator=generator, dtype=torch.float32)
        noise_norms = torch.linalg.vector_norm(noise, dim=1, keepdim=True).clamp(min=1e-12)
        source_norms = torch.linalg.vector_norm(values.float(), dim=1, keepdim=True)
        random_trace[layer] = (noise / noise_norms * source_norms).contiguous()
    return random_trace


def install_trace_patch_hooks(layers_mod: Any, *, layers: list[int], state: TracePatchState) -> list[Any]:
    handles: list[Any] = []

    for layer in layers:

        def make_hook(layer_idx: int) -> Any:
            def hook(_module: torch.nn.Module, _inp: tuple[Any, ...], out: Any) -> Any:
                if state.trace is None or state.blend == 0.0:
                    return out
                trace_values = state.trace.get(layer_idx)
                if trace_values is None or state.step >= int(trace_values.shape[0]):
                    return out

                hidden = out[0] if isinstance(out, tuple) else out
                source_vec = trace_values[state.step].to(device=hidden.device, dtype=hidden.dtype)
                edited = hidden.clone()
                edited[:, -1, :] = edited[:, -1, :] + float(state.blend) * (source_vec - edited[:, -1, :])
                if isinstance(out, tuple):
                    return (edited,) + out[1:]
                return edited

            return hook

        handles.append(layers_mod[layer].register_forward_hook(make_hook(layer)))
    return handles


@torch.inference_mode()
def generate_one_trace_patched(
    *,
    model: torch.nn.Module,
    tokenizer: Any,
    prompt: str,
    state: TracePatchState,
    trace: dict[int, torch.Tensor] | None,
    blend: float,
    max_new_tokens: int,
    show_progress: bool,
) -> dict[str, Any]:
    chat = format_chat(tokenizer, prompt)
    inputs = tokenizer(chat, return_tensors="pt", add_special_tokens=False, truncation=True, max_length=2048)
    input_device = first_parameter_device(model)
    input_ids = inputs["input_ids"].to(input_device)
    attention_mask = inputs["attention_mask"].to(input_device)

    state.trace = trace
    state.blend = float(blend)
    generated: list[torch.Tensor] = []
    past_key_values: Any = None
    next_input = input_ids
    iterator = range(max_new_tokens)
    if show_progress and max_new_tokens > 10:
        iterator = tqdm(iterator, desc="decode", leave=False)

    try:
        for step in iterator:
            state.step = int(step)
            outputs = model(
                input_ids=next_input,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
            )
            past_key_values = outputs.past_key_values
            next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)
            generated.append(next_token.detach().cpu())
            attention_mask = torch.cat(
                [attention_mask, torch.ones((attention_mask.shape[0], 1), dtype=attention_mask.dtype, device=input_device)],
                dim=1,
            )
            next_input = next_token
            if tokenizer.eos_token_id is not None and int(next_token[0, 0].item()) == int(tokenizer.eos_token_id):
                break
    finally:
        state.trace = None
        state.blend = 0.0
        state.step = 0

    if generated:
        generated_ids = torch.cat(generated, dim=1)[0]
    else:
        generated_ids = torch.empty((0,), dtype=torch.long)
    text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    return {
        "text": text,
        "generated_tokens": int(generated_ids.numel()),
        "prompt_tokens": int(input_ids.shape[-1]),
    }


def generate_many_trace_patched(
    *,
    model: torch.nn.Module,
    tokenizer: Any,
    prompts: list[str],
    state: TracePatchState,
    trace: dict[int, torch.Tensor] | None,
    blend: float,
    max_new_tokens: int,
) -> list[dict[str, Any]]:
    outputs: list[dict[str, Any]] = []
    outer = tqdm(prompts, desc="prompts", leave=False) if len(prompts) > 10 else prompts
    for prompt in outer:
        outputs.append(
            generate_one_trace_patched(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                state=state,
                trace=trace,
                blend=blend,
                max_new_tokens=max_new_tokens,
                show_progress=False,
            )
        )
    return outputs


def trace_meta(source: ReplaySource, *, name: str, layers: list[int], trace: dict[int, torch.Tensor]) -> dict[str, Any]:
    per_layer = []
    for layer in layers:
        values = trace[layer]
        norms = torch.linalg.vector_norm(values.float(), dim=1)
        per_layer.append(
            {
                "layer": layer,
                "trace_steps": int(values.shape[0]),
                "hidden_dim": int(values.shape[1]),
                "mean_step_norm": float(norms.mean().item()),
                "min_step_norm": float(norms.min().item()),
                "max_step_norm": float(norms.max().item()),
            }
        )
    return {
        "source": "minimal_pair_replay_trace",
        "name": name,
        "layer": ",".join(str(layer) for layer in layers),
        "region": "decode_step_residual",
        "group_field": "frame_label",
        "group_value": source.frame_label,
        "target_justice": source.frame_label,
        "reference_justice": "current_decode_state",
        "n_target": 1,
        "n_reference": "",
        "raw_direction_norm": "",
        "top_features": [],
        "example_id": source.example_id,
        "pair_id": source.pair_id,
        "fact_id": source.fact_id,
        "split": source.split,
        "trace_layers": per_layer,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--replay-rows", type=Path, default=DEFAULT_REPLAY_ROWS)
    parser.add_argument("--replay-run", type=Path, default=DEFAULT_REPLAY_RUN)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--prompt-bank", type=Path, default=DEFAULT_PROMPT_BANK)
    parser.add_argument("--layers", default="16,20")
    parser.add_argument("--source-example-id", default="")
    parser.add_argument("--source-label", default="commerce_limits")
    parser.add_argument("--source-split", default="train")
    parser.add_argument("--control-source-example-id", default="")
    parser.add_argument("--control-source-label", default="commerce_authority")
    parser.add_argument("--control-source-split", default="train")
    parser.add_argument("--blends", default="0.05,0.1,0.2")
    parser.add_argument("--prompt-ids", default="0,1,2,3,4,5")
    parser.add_argument("--max-prompts", type=int, default=6)
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_COMPLETE_ANSWER_TOKENS)
    add_short_budget_arg(parser)
    parser.add_argument("--random-controls", type=int, default=4)
    parser.add_argument("--seed", type=int, default=47)
    parser.add_argument("--score-mode", choices=["prompt_expected", "commerce_limits", "commerce_authority"], default="prompt_expected")
    parser.add_argument("--report-max-output-rows", type=int, default=240)
    parser.add_argument("--device-map", default="single")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    enforce_complete_answer_budget(
        args.max_new_tokens,
        allow_short=args.allow_short_answer_budget,
        purpose="token-local replay trace patch",
    )
    started = now_iso()
    layers = parse_int_list(args.layers)
    blends = parse_float_list(args.blends)
    out_dir = args.output_root / f"scotus_trace_patch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)

    replay_rows = read_jsonl(args.replay_rows)
    source = choose_source(
        replay_rows,
        example_id=args.source_example_id,
        label=args.source_label,
        split=args.source_split,
    )
    control_source = choose_source(
        replay_rows,
        example_id=args.control_source_example_id,
        label=args.control_source_label,
        split=args.control_source_split,
    )

    all_prompt_specs = load_prompt_specs(args.prompt_bank)
    prompt_specs = override_prompt_scores(
        select_prompt_specs(all_prompt_specs, args.prompt_ids, args.max_prompts),
        args.score_mode,
    )
    prompt_texts = [spec.prompt for spec in prompt_specs]
    print(f"Loaded {len(prompt_specs)} prompts", flush=True)
    print(f"Source trace: {source.example_id} ({source.frame_label})", flush=True)
    print(f"Control trace: {control_source.example_id} ({control_source.frame_label})", flush=True)

    tokenizer, model = load_model_and_tokenizer(args.model_path, args.device_map)
    layers_mod = transformer_layers(model)

    source_trace = capture_source_trace(
        model=model,
        tokenizer=tokenizer,
        layers_mod=layers_mod,
        source=source,
        layers=layers,
        max_steps=args.max_new_tokens,
    )
    control_trace = capture_source_trace(
        model=model,
        tokenizer=tokenizer,
        layers_mod=layers_mod,
        source=control_source,
        layers=layers,
        max_steps=args.max_new_tokens,
    )
    random_traces = [
        random_trace_like(source_trace, seed=args.seed + random_idx * 100_003)
        for random_idx in range(max(0, args.random_controls))
    ]

    candidate_name = source_name("trace", source, layers)
    control_name = source_name("trace_control", control_source, layers)
    direction_meta = [
        trace_meta(source, name=candidate_name, layers=layers, trace=source_trace),
        trace_meta(control_source, name=control_name, layers=layers, trace=control_trace),
    ]

    state = TracePatchState()
    handles = install_trace_patch_hooks(layers_mod, layers=layers, state=state)
    rows: list[dict[str, Any]] = []
    try:
        print("Generating baseline batch", flush=True)
        base_outputs = generate_many_trace_patched(
            model=model,
            tokenizer=tokenizer,
            prompts=prompt_texts,
            state=state,
            trace=None,
            blend=0.0,
            max_new_tokens=args.max_new_tokens,
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
            for random_idx, random_trace in enumerate(random_traces):
                print(f"Generating random trace[{random_idx}] blend={blend}", flush=True)
                random_outputs = generate_many_trace_patched(
                    model=model,
                    tokenizer=tokenizer,
                    prompts=prompt_texts,
                    state=state,
                    trace=random_trace,
                    blend=blend,
                    max_new_tokens=args.max_new_tokens,
                )
                for spec, output in zip(prompt_specs, random_outputs, strict=True):
                    rows.append(
                        row_for_generation(
                            spec=spec,
                            condition="random_unit",
                            candidate="random_trace",
                            alpha=blend,
                            effective_alpha=blend,
                            random_index=random_idx,
                            layer=layers[0],
                            output=output,
                        )
                    )

            print(f"Generating control-source trace blend={blend}", flush=True)
            control_outputs = generate_many_trace_patched(
                model=model,
                tokenizer=tokenizer,
                prompts=prompt_texts,
                state=state,
                trace=control_trace,
                blend=blend,
                max_new_tokens=args.max_new_tokens,
            )
            for spec, output in zip(prompt_specs, control_outputs, strict=True):
                rows.append(
                    row_for_generation(
                        spec=spec,
                        condition="source_control",
                        candidate=control_name,
                        alpha=blend,
                        effective_alpha=blend,
                        random_index=None,
                        layer=layers[0],
                        output=output,
                    )
                )

            print(f"Generating candidate source trace blend={blend}", flush=True)
            candidate_outputs = generate_many_trace_patched(
                model=model,
                tokenizer=tokenizer,
                prompts=prompt_texts,
                state=state,
                trace=source_trace,
                blend=blend,
                max_new_tokens=args.max_new_tokens,
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
    finally:
        for handle in handles:
            handle.remove()

    add_base_deltas(rows)
    manifest = {
        "started_at": started,
        "finished_at": now_iso(),
        "model_path": str(args.model_path),
        "sae_path": "",
        "overlap_dir": str(args.replay_run),
        "output_dir": str(out_dir),
        "direction_source": "external",
        "external_direction_files": [],
        "alpha_scale": "trace_replacement_blend",
        "hidden_norm_reference": str(args.replay_run),
        "prompt_bank": str(args.prompt_bank),
        "candidate_names": [candidate_name],
        "source_control_names": [control_name],
        "source_trace": trace_meta(source, name=candidate_name, layers=layers, trace=source_trace),
        "control_trace": trace_meta(control_source, name=control_name, layers=layers, trace=control_trace),
        "alphas": blends,
        "random_controls": int(args.random_controls),
        "position": "decode_last_token_trace",
        "score_mode": args.score_mode,
        "max_prompts": args.max_prompts,
        "max_new_tokens": args.max_new_tokens,
        **qwen_budget_metadata(args.max_new_tokens),
        "generation_batch_size": 1,
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
