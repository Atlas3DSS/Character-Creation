#!/usr/bin/env python3
"""Full-attention head trace patching for SCOTUS replay candidates.

This is a narrow circuit-localization screen after residual, prototype,
residual-trace, and coarse component-output patching failed promotion. It
patches one full-attention head at the input to ``o_proj`` during generation,
using a teacher-forced minimal-pair replay trace.
"""

from __future__ import annotations

import argparse
import gc
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any

import torch

from patch_scotus_component_traces import (
    component_comparisons,
    fmt,
    make_row,
    markdown_table,
)
from patch_scotus_replay_traces import (
    DEFAULT_PROMPT_BANK,
    DEFAULT_REPLAY_ROWS,
    DEFAULT_REPLAY_RUN,
    choose_source,
    generate_one_trace_patched,
    override_prompt_scores,
    parse_int_list,
    read_jsonl,
)
from poke_scotus_sae_layers import (
    DEFAULT_COMPLETE_ANSWER_TOKENS,
    DEFAULT_MODEL,
    DEFAULT_OUTPUT_ROOT,
    add_base_deltas,
    aggregate_frame_scores,
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


@dataclass(frozen=True)
class HeadSpec:
    layer: int
    head: int

    @property
    def key(self) -> str:
        return f"L{self.layer:02d}_H{self.head:02d}"


@dataclass
class HeadPatchState:
    trace: torch.Tensor | None = None
    blend: float = 0.0
    step: int = 0


def text_config(model: torch.nn.Module) -> Any:
    config = getattr(model, "config")
    return getattr(config, "text_config", config)


def head_shape(model: torch.nn.Module) -> tuple[int, int]:
    config = text_config(model)
    num_heads = int(getattr(config, "num_attention_heads"))
    head_dim = int(getattr(config, "head_dim", int(getattr(config, "hidden_size")) // num_heads))
    return num_heads, head_dim


def ensure_full_attention_layer(layers_mod: Any, layer: int) -> torch.nn.Module:
    layer_mod = layers_mod[layer]
    if not hasattr(layer_mod, "self_attn"):
        layer_type = getattr(layer_mod, "layer_type", type(layer_mod).__name__)
        raise ValueError(f"Layer {layer} is not a full-attention layer; layer_type={layer_type}")
    return getattr(layer_mod, "self_attn")


def full_attention_layers(layers_mod: Any) -> list[int]:
    return [idx for idx, layer_mod in enumerate(layers_mod) if hasattr(layer_mod, "self_attn")]


def parse_head_specs(raw: str) -> list[HeadSpec]:
    specs: list[HeadSpec] = []
    for item in raw.split(","):
        text = item.strip()
        if not text:
            continue
        if ":" not in text:
            raise ValueError(f"Head spec must be L:H, got {text!r}")
        layer_text, head_text = text.split(":", 1)
        specs.append(HeadSpec(layer=int(layer_text), head=int(head_text)))
    return specs


def install_head_capture_hooks(
    *,
    layers_mod: Any,
    layers: list[int],
    positions: torch.Tensor,
    num_heads: int,
    head_dim: int,
    captured: dict[int, torch.Tensor],
) -> list[Any]:
    handles: list[Any] = []
    cpu_positions = positions.detach().cpu()

    for layer in layers:
        attn = ensure_full_attention_layer(layers_mod, layer)
        module = getattr(attn, "o_proj")

        def make_hook(layer_idx: int) -> Any:
            def hook(_module: torch.nn.Module, inp: tuple[Any, ...]) -> None:
                hidden = inp[0]
                if hidden.shape[-1] != num_heads * head_dim:
                    raise RuntimeError(
                        f"L{layer_idx} o_proj input dim {hidden.shape[-1]} != {num_heads * head_dim}"
                    )
                heads = hidden.reshape(hidden.shape[0], hidden.shape[1], num_heads, head_dim)
                captured[layer_idx] = (
                    heads[0, cpu_positions.to(hidden.device), :, :]
                    .detach()
                    .float()
                    .cpu()
                    .contiguous()
                )

            return hook

        handles.append(module.register_forward_pre_hook(make_hook(layer)))
    return handles


@torch.inference_mode()
def capture_head_traces(
    *,
    model: torch.nn.Module,
    tokenizer: Any,
    layers_mod: Any,
    source: Any,
    layers: list[int],
    max_steps: int,
    num_heads: int,
    head_dim: int,
) -> dict[int, torch.Tensor]:
    prompt_ids = tokenizer(format_chat(tokenizer, source.prompt), add_special_tokens=False).input_ids
    assistant_ids = tokenizer(source.assistant_text, add_special_tokens=False).input_ids
    trace_len = min(max_steps, len(assistant_ids))
    if not prompt_ids or trace_len <= 0:
        raise RuntimeError(f"Cannot capture head trace for {source.example_id}")

    full_ids = torch.tensor([prompt_ids + assistant_ids[:trace_len]], dtype=torch.long)
    positions = torch.arange(len(prompt_ids) - 1, len(prompt_ids) - 1 + trace_len, dtype=torch.long)
    input_device = first_parameter_device(model)
    full_ids = full_ids.to(input_device)
    attention_mask = torch.ones_like(full_ids, device=input_device)

    captured: dict[int, torch.Tensor] = {}
    handles = install_head_capture_hooks(
        layers_mod=layers_mod,
        layers=layers,
        positions=positions,
        num_heads=num_heads,
        head_dim=head_dim,
        captured=captured,
    )
    try:
        _ = model(input_ids=full_ids, attention_mask=attention_mask, use_cache=False)
    finally:
        for handle in handles:
            handle.remove()

    missing = sorted(set(layers) - set(captured))
    if missing:
        raise RuntimeError(f"Did not capture full-attention head traces: {missing}")
    return captured


def rank_heads(
    *,
    source_traces: dict[int, torch.Tensor],
    control_traces: dict[int, torch.Tensor],
    layers: list[int],
) -> list[dict[str, Any]]:
    ranking: list[dict[str, Any]] = []
    for layer in layers:
        source = source_traces[layer]
        control = control_traces[layer]
        steps = min(int(source.shape[0]), int(control.shape[0]))
        if steps <= 0:
            continue
        diff = source[:steps] - control[:steps]
        delta_norm = torch.linalg.vector_norm(diff.float(), dim=-1).mean(dim=0)
        source_norm = torch.linalg.vector_norm(source[:steps].float(), dim=-1).mean(dim=0)
        control_norm = torch.linalg.vector_norm(control[:steps].float(), dim=-1).mean(dim=0)
        for head in range(int(source.shape[1])):
            ranking.append(
                {
                    "layer": layer,
                    "head": head,
                    "candidate": HeadSpec(layer, head).key,
                    "trace_steps": steps,
                    "mean_delta_norm": float(delta_norm[head].item()),
                    "mean_source_norm": float(source_norm[head].item()),
                    "mean_control_norm": float(control_norm[head].item()),
                }
            )
    ranking.sort(key=lambda item: float(item["mean_delta_norm"]), reverse=True)
    return ranking


def select_heads(
    *,
    ranking: list[dict[str, Any]],
    requested: list[HeadSpec],
    top_k: int,
) -> list[HeadSpec]:
    if requested:
        return requested
    return [HeadSpec(int(item["layer"]), int(item["head"])) for item in ranking[:top_k]]


def head_trace(trace: dict[int, torch.Tensor], spec: HeadSpec) -> torch.Tensor:
    values = trace[spec.layer]
    return values[:, spec.head, :].contiguous()


def random_trace_like(trace: torch.Tensor, *, seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    noise = torch.randn(trace.shape, generator=generator, dtype=torch.float32)
    noise_norms = torch.linalg.vector_norm(noise, dim=1, keepdim=True).clamp(min=1e-12)
    source_norms = torch.linalg.vector_norm(trace.float(), dim=1, keepdim=True)
    return (noise / noise_norms * source_norms).contiguous()


def install_head_patch_hook(
    *,
    layers_mod: Any,
    spec: HeadSpec,
    state: HeadPatchState,
    num_heads: int,
    head_dim: int,
) -> Any:
    attn = ensure_full_attention_layer(layers_mod, spec.layer)
    module = getattr(attn, "o_proj")
    start = spec.head * head_dim
    end = start + head_dim

    def hook(_module: torch.nn.Module, inp: tuple[Any, ...]) -> tuple[Any, ...]:
        if state.trace is None or state.blend == 0.0 or state.step >= int(state.trace.shape[0]):
            return inp
        hidden = inp[0]
        if hidden.shape[-1] != num_heads * head_dim:
            raise RuntimeError(f"Unexpected o_proj input dim: {hidden.shape[-1]}")
        source_vec = state.trace[state.step].to(device=hidden.device, dtype=hidden.dtype)
        edited = hidden.clone()
        edited[:, -1, start:end] = edited[:, -1, start:end] + float(state.blend) * (
            source_vec - edited[:, -1, start:end]
        )
        return (edited,) + inp[1:]

    return module.register_forward_pre_hook(hook)


def candidate_name(prefix: str, source: Any, spec: HeadSpec) -> str:
    example = str(source.example_id).replace("|", "_").replace("/", "_")
    return f"{prefix}_{source.frame_label}_{spec.key}_{example}"


def summarize_head_comparisons(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    comparisons = component_comparisons(rows)
    for item in comparisons:
        component = str(item.get("component") or "")
        if component.startswith("head_"):
            item["head"] = int(component.removeprefix("head_"))
    return comparisons


def write_head_report(
    path: Path,
    *,
    manifest: dict[str, Any],
    ranking: list[dict[str, Any]],
    comparisons: list[dict[str, Any]],
) -> None:
    top_rank_rows = [
        [
            item["candidate"],
            item["layer"],
            item["head"],
            item["trace_steps"],
            fmt(item["mean_delta_norm"]),
            fmt(item["mean_source_norm"]),
            fmt(item["mean_control_norm"]),
        ]
        for item in ranking[: int(manifest["rank_report_top_n"])]
    ]
    comparison_rows = [
        [
            item["candidate"],
            item["layer"],
            item.get("head", ""),
            fmt(item["alpha"]),
            item["n"],
            fmt(item["candidate_mean_target"]),
            fmt(item["random_mean_target"]),
            fmt(item["matched_target"]),
            fmt(item["candidate_mean_net"]),
            fmt(item["random_mean_net"]),
            fmt(item["matched_net"]),
            fmt(item["source_control_mean_target"]),
            fmt(item["source_control_mean_net"]),
            fmt(item["target_strongest_win_rate"]),
            fmt(item["net_strongest_win_rate"]),
        ]
        for item in comparisons
    ]
    lines = [
        "# SCOTUS Full-Attention Head Trace Patch Summary",
        "",
        "## Purpose",
        "",
        "Patch one full-attention head at a time at the `o_proj` input, after coarser residual and component interventions failed.",
        "This is a circuit-localization screen, not a promotion run.",
        "",
        "## Configuration",
        "",
        markdown_table(
            ["Field", "Value"],
            [
                ["Started", manifest["started_at"]],
                ["Finished", manifest["finished_at"]],
                ["Model", manifest["model_path"]],
                ["Prompt bank", manifest["prompt_bank"]],
                ["Prompt keys", ", ".join(manifest["prompt_keys"])],
                ["Candidate layers", ", ".join(str(layer) for layer in manifest["layers"])],
                ["Selected heads", ", ".join(manifest["selected_heads"])],
                ["Blends", ", ".join(str(alpha) for alpha in manifest["alphas"])],
                ["Random controls", manifest["random_controls"]],
                ["Source", manifest["source_example_id"]],
                ["Source control", manifest["control_source_example_id"]],
            ],
        ),
        "",
        "## Source-vs-Control Head Ranking",
        "",
        markdown_table(
            ["Candidate", "Layer", "Head", "Steps", "Delta norm", "Source norm", "Control norm"],
            top_rank_rows,
        ),
        "",
        "## Candidate vs Matched Controls",
        "",
        markdown_table(
            [
                "Candidate",
                "Layer",
                "Head",
                "Blend",
                "N",
                "Cand target",
                "Rand target",
                "Matched target",
                "Cand net",
                "Rand net",
                "Matched net",
                "Source target",
                "Source net",
                "Target strongest win",
                "Net strongest win",
            ],
            comparison_rows,
        ),
        "",
        "## Reading Rule",
        "",
        "- A head is only a candidate if it beats prompt-matched random traces and the contrast-source trace.",
        "- Ranking by source-vs-control trace norm is only a preselection heuristic.",
        "- Keyword frame movement is a screen; any survivor needs repaired replay data and blind proposition review.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--replay-rows", type=Path, default=DEFAULT_REPLAY_ROWS)
    parser.add_argument("--replay-run", type=Path, default=DEFAULT_REPLAY_RUN)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--prompt-bank", type=Path, default=DEFAULT_PROMPT_BANK)
    parser.add_argument("--layers", default="15,19,23")
    parser.add_argument("--heads", default="", help="Comma-separated L:H specs. Empty means use top ranked heads.")
    parser.add_argument("--rank-top-k", type=int, default=6)
    parser.add_argument("--rank-report-top-n", type=int, default=24)
    parser.add_argument("--source-example-id", default="")
    parser.add_argument("--source-label", default="commerce_limits")
    parser.add_argument("--source-split", default="train")
    parser.add_argument("--control-source-example-id", default="")
    parser.add_argument("--control-source-label", default="commerce_authority")
    parser.add_argument("--control-source-split", default="train")
    parser.add_argument("--blends", default="0.1,0.3")
    parser.add_argument("--prompt-ids", default="2,3,4,5")
    parser.add_argument("--max-prompts", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_COMPLETE_ANSWER_TOKENS)
    add_short_budget_arg(parser)
    parser.add_argument("--random-controls", type=int, default=2)
    parser.add_argument("--seed", type=int, default=59)
    parser.add_argument("--score-mode", choices=["prompt_expected", "commerce_limits", "commerce_authority"], default="prompt_expected")
    parser.add_argument("--report-max-output-rows", type=int, default=220)
    parser.add_argument("--device-map", default="single")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    enforce_complete_answer_budget(
        args.max_new_tokens,
        allow_short=args.allow_short_answer_budget,
        purpose="attention-head trace patch",
    )
    started = now_iso()
    layers = parse_int_list(args.layers)
    requested_heads = parse_head_specs(args.heads)
    blends = parse_float_list(args.blends)
    out_dir = args.output_root / f"scotus_attention_head_patch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
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
    prompt_specs = override_prompt_scores(
        select_prompt_specs(load_prompt_specs(args.prompt_bank), args.prompt_ids, args.max_prompts),
        args.score_mode,
    )
    prompt_texts = [spec.prompt for spec in prompt_specs]

    tokenizer, model = load_model_and_tokenizer(args.model_path, args.device_map)
    layers_mod = transformer_layers(model)
    available_full_layers = full_attention_layers(layers_mod)
    missing_full = [layer for layer in layers if layer not in available_full_layers]
    if missing_full:
        raise ValueError(f"Requested non-full-attention layers: {missing_full}; full layers={available_full_layers}")
    num_heads, head_dim = head_shape(model)
    print(f"Loaded {len(prompt_specs)} prompts", flush=True)
    print(f"Full-attention layers available: {available_full_layers}", flush=True)
    print(f"Head shape: num_heads={num_heads}, head_dim={head_dim}", flush=True)
    print(f"Source trace: {source.example_id} ({source.frame_label})", flush=True)
    print(f"Control trace: {control_source.example_id} ({control_source.frame_label})", flush=True)

    source_traces = capture_head_traces(
        model=model,
        tokenizer=tokenizer,
        layers_mod=layers_mod,
        source=source,
        layers=layers,
        max_steps=args.max_new_tokens,
        num_heads=num_heads,
        head_dim=head_dim,
    )
    control_traces = capture_head_traces(
        model=model,
        tokenizer=tokenizer,
        layers_mod=layers_mod,
        source=control_source,
        layers=layers,
        max_steps=args.max_new_tokens,
        num_heads=num_heads,
        head_dim=head_dim,
    )
    ranking = rank_heads(source_traces=source_traces, control_traces=control_traces, layers=layers)
    selected = select_heads(ranking=ranking, requested=requested_heads, top_k=args.rank_top_k)
    for spec in selected:
        if spec.layer not in layers:
            raise ValueError(f"Selected head {spec.key} is outside requested layers {layers}")
        if spec.head < 0 or spec.head >= num_heads:
            raise ValueError(f"Selected head {spec.key} outside 0..{num_heads - 1}")
    print(f"Selected heads: {', '.join(spec.key for spec in selected)}", flush=True)

    rows: list[dict[str, Any]] = []
    state = HeadPatchState()
    print("Generating baseline batch", flush=True)
    base_outputs = [
        generate_one_trace_patched(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            state=state,  # type: ignore[arg-type]
            trace=None,
            blend=0.0,
            max_new_tokens=args.max_new_tokens,
            show_progress=False,
        )
        for prompt in prompt_texts
    ]
    for prompt_spec, output in zip(prompt_specs, base_outputs, strict=True):
        rows.append(
            row_for_generation(
                spec=prompt_spec,
                condition="base",
                candidate=None,
                alpha=0.0,
                effective_alpha=0.0,
                random_index=None,
                layer=None,
                output=output,
            )
        )

    for spec in selected:
        candidate = candidate_name("head_trace", source, spec)
        control_candidate = candidate_name("head_trace_control", control_source, spec)
        source_trace = head_trace(source_traces, spec)
        control_trace = head_trace(control_traces, spec)
        random_traces = [
            random_trace_like(source_trace, seed=args.seed + spec.layer * 1009 + spec.head * 9176 + idx * 100_003)
            for idx in range(max(0, args.random_controls))
        ]
        handle = install_head_patch_hook(
            layers_mod=layers_mod,
            spec=spec,
            state=state,
            num_heads=num_heads,
            head_dim=head_dim,
        )
        try:
            for blend in blends:
                for random_idx, random_trace in enumerate(random_traces):
                    print(f"Generating random {spec.key}[{random_idx}] blend={blend}", flush=True)
                    outputs = [
                        generate_one_trace_patched(
                            model=model,
                            tokenizer=tokenizer,
                            prompt=prompt,
                            state=state,  # type: ignore[arg-type]
                            trace=random_trace,
                            blend=blend,
                            max_new_tokens=args.max_new_tokens,
                            show_progress=False,
                        )
                        for prompt in prompt_texts
                    ]
                    for prompt_spec, output in zip(prompt_specs, outputs, strict=True):
                        rows.append(
                            make_row(
                                spec=prompt_spec,
                                condition="random_unit",
                                candidate=f"random_for_{candidate}",
                                alpha=blend,
                                random_index=random_idx,
                                layer=spec.layer,
                                component=f"head_{spec.head:02d}",
                                target_candidate=candidate,
                                output=output,
                            )
                        )

                print(f"Generating source-control {spec.key} blend={blend}", flush=True)
                outputs = [
                    generate_one_trace_patched(
                        model=model,
                        tokenizer=tokenizer,
                        prompt=prompt,
                        state=state,  # type: ignore[arg-type]
                        trace=control_trace,
                        blend=blend,
                        max_new_tokens=args.max_new_tokens,
                        show_progress=False,
                    )
                    for prompt in prompt_texts
                ]
                for prompt_spec, output in zip(prompt_specs, outputs, strict=True):
                    rows.append(
                        make_row(
                            spec=prompt_spec,
                            condition="source_control",
                            candidate=control_candidate,
                            alpha=blend,
                            random_index=None,
                            layer=spec.layer,
                            component=f"head_{spec.head:02d}",
                            target_candidate=candidate,
                            output=output,
                        )
                    )

                print(f"Generating candidate {spec.key} blend={blend}", flush=True)
                outputs = [
                    generate_one_trace_patched(
                        model=model,
                        tokenizer=tokenizer,
                        prompt=prompt,
                        state=state,  # type: ignore[arg-type]
                        trace=source_trace,
                        blend=blend,
                        max_new_tokens=args.max_new_tokens,
                        show_progress=False,
                    )
                    for prompt in prompt_texts
                ]
                for prompt_spec, output in zip(prompt_specs, outputs, strict=True):
                    rows.append(
                        make_row(
                            spec=prompt_spec,
                            condition="sae_poke",
                            candidate=candidate,
                            alpha=blend,
                            random_index=None,
                            layer=spec.layer,
                            component=f"head_{spec.head:02d}",
                            target_candidate=candidate,
                            output=output,
                        )
                    )
        finally:
            handle.remove()
            state.trace = None
            state.blend = 0.0
            state.step = 0

    add_base_deltas(rows)
    comparisons = summarize_head_comparisons(rows)
    direction_meta = [
        {
            "source": "minimal_pair_replay_attention_head_trace",
            "name": candidate_name("head_trace", source, spec),
            "layer": spec.layer,
            "head": spec.head,
            "region": "decode_step_full_attention_o_proj_input_head",
            "group_field": "frame_label",
            "group_value": source.frame_label,
            "target_justice": source.frame_label,
            "reference_justice": "current_attention_head_output",
            "n_target": 1,
            "n_reference": "",
            "raw_direction_norm": "",
            "top_features": [],
        }
        for spec in selected
    ]
    manifest = {
        "started_at": started,
        "finished_at": now_iso(),
        "model_path": str(args.model_path),
        "sae_path": "",
        "overlap_dir": str(args.replay_run),
        "output_dir": str(out_dir),
        "direction_source": "external",
        "external_direction_files": [],
        "alpha_scale": "attention_head_trace_replacement_blend",
        "hidden_norm_reference": str(args.replay_run),
        "prompt_bank": str(args.prompt_bank),
        "candidate_names": [item["name"] for item in direction_meta],
        "source_example_id": source.example_id,
        "control_source_example_id": control_source.example_id,
        "layers": layers,
        "available_full_attention_layers": available_full_layers,
        "num_attention_heads": num_heads,
        "head_dim": head_dim,
        "selected_heads": [spec.key for spec in selected],
        "head_ranking_metric": "mean_l2_norm(source_trace_head - contrast_source_trace_head)",
        "rank_top_k": int(args.rank_top_k),
        "rank_report_top_n": int(args.rank_report_top_n),
        "alphas": blends,
        "random_controls": int(args.random_controls),
        "position": "decode_last_token_attention_head_trace",
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
    write_jsonl(out_dir / "head_ranking.jsonl", ranking)
    write_jsonl(out_dir / "generations.jsonl", rows)
    write_jsonl(out_dir / "frame_summary.jsonl", summarize_frames(rows))
    write_jsonl(out_dir / "score_summary.jsonl", aggregate_frame_scores(rows))
    write_jsonl(out_dir / "head_vs_matched_random.jsonl", comparisons)
    write_report(out_dir / "report.md", manifest=manifest, direction_meta=direction_meta, rows=rows)
    write_head_report(out_dir / "head_report.md", manifest=manifest, ranking=ranking, comparisons=comparisons)

    del model, tokenizer, layers_mod
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(f"Wrote {out_dir / 'head_report.md'}", flush=True)


if __name__ == "__main__":
    main()
