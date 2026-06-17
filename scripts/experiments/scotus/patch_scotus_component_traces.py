#!/usr/bin/env python3
"""Component-level path patching for SCOTUS minimal-pair replay traces.

Residual act-add, prototype replacement, and residual trace replacement all
failed promotion. This script changes intervention family: it patches token
mixer or MLP outputs from a teacher-forced replay trace into generation, one
layer/component at a time, with same-component random and contrast-source
controls.
"""

from __future__ import annotations

import argparse
import gc
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any

import torch

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
class ComponentSpec:
    layer: int
    component: str

    @property
    def key(self) -> str:
        return f"L{self.layer:02d}_{self.component}"


@dataclass
class ComponentPatchState:
    trace: torch.Tensor | None = None
    blend: float = 0.0
    step: int = 0


def parse_components(raw: str) -> list[str]:
    components = [item.strip() for item in raw.split(",") if item.strip()]
    allowed = {"mixer", "mlp"}
    unknown = sorted(set(components) - allowed)
    if unknown:
        raise ValueError(f"Unknown components {unknown}; allowed={sorted(allowed)}")
    return components


def component_specs(layers: list[int], components: list[str]) -> list[ComponentSpec]:
    return [ComponentSpec(layer=layer, component=component) for layer in layers for component in components]


def select_component_module(layer_mod: torch.nn.Module, component: str) -> torch.nn.Module:
    if component == "mlp":
        return getattr(layer_mod, "mlp")
    if component != "mixer":
        raise ValueError(f"Unknown component: {component}")
    if hasattr(layer_mod, "linear_attn"):
        return getattr(layer_mod, "linear_attn")
    if hasattr(layer_mod, "self_attn"):
        return getattr(layer_mod, "self_attn")
    raise RuntimeError(f"Layer {type(layer_mod).__name__} has no recognized token mixer")


def output_tensor(output: Any) -> torch.Tensor:
    return output[0] if isinstance(output, tuple) else output


def replace_output_tensor(output: Any, tensor: torch.Tensor) -> Any:
    if isinstance(output, tuple):
        return (tensor,) + output[1:]
    return tensor


def install_component_capture_hooks(
    *,
    layers_mod: Any,
    specs: list[ComponentSpec],
    positions: torch.Tensor,
    captured: dict[str, torch.Tensor],
) -> list[Any]:
    handles: list[Any] = []
    cpu_positions = positions.detach().cpu()
    for spec in specs:
        module = select_component_module(layers_mod[spec.layer], spec.component)

        def make_hook(item: ComponentSpec) -> Any:
            def hook(_module: torch.nn.Module, _inp: tuple[Any, ...], out: Any) -> Any:
                hidden = output_tensor(out)
                captured[item.key] = hidden[0, cpu_positions.to(hidden.device), :].detach().float().cpu().contiguous()
                return out

            return hook

        handles.append(module.register_forward_hook(make_hook(spec)))
    return handles


@torch.inference_mode()
def capture_component_traces(
    *,
    model: torch.nn.Module,
    tokenizer: Any,
    layers_mod: Any,
    source: Any,
    specs: list[ComponentSpec],
    max_steps: int,
) -> dict[str, torch.Tensor]:
    from patch_scotus_replay_traces import format_chat

    chat = format_chat(tokenizer, source.prompt)
    prompt_ids = tokenizer(chat, add_special_tokens=False).input_ids
    assistant_ids = tokenizer(source.assistant_text, add_special_tokens=False).input_ids
    trace_len = min(max_steps, len(assistant_ids))
    if not prompt_ids or trace_len <= 0:
        raise RuntimeError(f"Cannot capture component trace for {source.example_id}")

    full_ids = torch.tensor([prompt_ids + assistant_ids], dtype=torch.long)
    positions = torch.arange(len(prompt_ids) - 1, len(prompt_ids) - 1 + trace_len, dtype=torch.long)
    input_device = first_parameter_device(model)
    full_ids = full_ids.to(input_device)
    attention_mask = torch.ones_like(full_ids, device=input_device)

    captured: dict[str, torch.Tensor] = {}
    handles = install_component_capture_hooks(
        layers_mod=layers_mod,
        specs=specs,
        positions=positions,
        captured=captured,
    )
    try:
        _ = model(input_ids=full_ids, attention_mask=attention_mask, use_cache=False)
    finally:
        for handle in handles:
            handle.remove()

    missing = sorted(set(spec.key for spec in specs) - set(captured))
    if missing:
        raise RuntimeError(f"Did not capture component traces: {missing}")
    return captured


def random_trace_like(trace: torch.Tensor, *, seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    noise = torch.randn(trace.shape, generator=generator, dtype=torch.float32)
    noise_norms = torch.linalg.vector_norm(noise, dim=1, keepdim=True).clamp(min=1e-12)
    source_norms = torch.linalg.vector_norm(trace.float(), dim=1, keepdim=True)
    return (noise / noise_norms * source_norms).contiguous()


def install_component_patch_hook(module: torch.nn.Module, state: ComponentPatchState) -> Any:
    def hook(_module: torch.nn.Module, _inp: tuple[Any, ...], out: Any) -> Any:
        if state.trace is None or state.blend == 0.0 or state.step >= int(state.trace.shape[0]):
            return out
        hidden = output_tensor(out)
        source_vec = state.trace[state.step].to(device=hidden.device, dtype=hidden.dtype)
        edited = hidden.clone()
        edited[:, -1, :] = edited[:, -1, :] + float(state.blend) * (source_vec - edited[:, -1, :])
        return replace_output_tensor(out, edited)

    return module.register_forward_hook(hook)


def component_candidate_name(prefix: str, source: Any, spec: ComponentSpec) -> str:
    example = str(source.example_id).replace("|", "_").replace("/", "_")
    return f"{prefix}_{source.frame_label}_{spec.key}_{example}"


def make_row(
    *,
    spec: Any,
    condition: str,
    candidate: str | None,
    alpha: float,
    random_index: int | None,
    layer: int | None,
    component: str | None,
    target_candidate: str | None,
    output: dict[str, Any],
) -> dict[str, Any]:
    row = row_for_generation(
        spec=spec,
        condition=condition,
        candidate=candidate,
        alpha=alpha,
        effective_alpha=alpha,
        random_index=random_index,
        layer=layer,
        output=output,
    )
    row["component"] = component
    row["target_candidate"] = target_candidate
    return row


def component_comparisons(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    random_by_target: dict[tuple[str, int, float], list[dict[str, Any]]] = {}
    source_control_by_target: dict[tuple[str, int, float], list[dict[str, Any]]] = {}
    for row in rows:
        target = str(row.get("target_candidate") or "")
        if not target:
            continue
        key = (target, int(row["prompt_id"]), float(row["alpha"]))
        if row["condition"] == "random_unit":
            random_by_target.setdefault(key, []).append(row)
        elif row["condition"] == "source_control":
            source_control_by_target.setdefault(key, []).append(row)

    candidate_rows = [
        row for row in rows if row["condition"] == "sae_poke" and row.get("candidate")
    ]
    groups: dict[tuple[str, float], list[dict[str, Any]]] = {}
    for row in candidate_rows:
        groups.setdefault((str(row["candidate"]), float(row["alpha"])), []).append(row)

    comparisons: list[dict[str, Any]] = []
    for (candidate, alpha), group in sorted(groups.items()):
        matched_target: list[float] = []
        matched_net: list[float] = []
        candidate_target: list[float] = []
        candidate_net: list[float] = []
        random_target_means: list[float] = []
        random_net_means: list[float] = []
        source_target: list[float] = []
        source_net: list[float] = []
        target_wins = 0
        net_wins = 0
        strongest_target_wins = 0
        strongest_net_wins = 0
        matched_prompts = 0

        for row in group:
            key = (candidate, int(row["prompt_id"]), alpha)
            controls = random_by_target.get(key, [])
            if not controls:
                continue
            matched_prompts += 1
            cand_target = float(row["frame_eval"].get("delta_target_hits_vs_base", 0.0))
            cand_net = float(row["frame_eval"].get("delta_target_minus_contrast_vs_base", 0.0))
            rand_target = [float(item["frame_eval"].get("delta_target_hits_vs_base", 0.0)) for item in controls]
            rand_net = [float(item["frame_eval"].get("delta_target_minus_contrast_vs_base", 0.0)) for item in controls]
            rand_target_mean = mean(rand_target)
            rand_net_mean = mean(rand_net)
            candidate_target.append(cand_target)
            candidate_net.append(cand_net)
            random_target_means.append(rand_target_mean)
            random_net_means.append(rand_net_mean)
            matched_target.append(cand_target - rand_target_mean)
            matched_net.append(cand_net - rand_net_mean)
            if cand_target > rand_target_mean:
                target_wins += 1
            if cand_net > rand_net_mean:
                net_wins += 1
            if cand_target > max(rand_target):
                strongest_target_wins += 1
            if cand_net > max(rand_net):
                strongest_net_wins += 1

            source_controls = source_control_by_target.get(key, [])
            if source_controls:
                source_target.append(float(source_controls[0]["frame_eval"].get("delta_target_hits_vs_base", 0.0)))
                source_net.append(float(source_controls[0]["frame_eval"].get("delta_target_minus_contrast_vs_base", 0.0)))

        comparisons.append(
            {
                "candidate": candidate,
                "alpha": alpha,
                "n": matched_prompts,
                "layer": group[0].get("layer"),
                "component": group[0].get("component"),
                "candidate_mean_target": mean(candidate_target) if candidate_target else 0.0,
                "random_mean_target": mean(random_target_means) if random_target_means else 0.0,
                "matched_target": mean(matched_target) if matched_target else 0.0,
                "candidate_mean_net": mean(candidate_net) if candidate_net else 0.0,
                "random_mean_net": mean(random_net_means) if random_net_means else 0.0,
                "matched_net": mean(matched_net) if matched_net else 0.0,
                "source_control_mean_target": mean(source_target) if source_target else None,
                "source_control_mean_net": mean(source_net) if source_net else None,
                "target_win_rate": 0.0 if matched_prompts == 0 else target_wins / matched_prompts,
                "net_win_rate": 0.0 if matched_prompts == 0 else net_wins / matched_prompts,
                "target_strongest_win_rate": 0.0 if matched_prompts == 0 else strongest_target_wins / matched_prompts,
                "net_strongest_win_rate": 0.0 if matched_prompts == 0 else strongest_net_wins / matched_prompts,
            }
        )
    comparisons.sort(
        key=lambda item: (
            float(item["matched_net"]),
            float(item["net_strongest_win_rate"]),
            float(item["matched_target"]),
        ),
        reverse=True,
    )
    return comparisons


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


def write_component_report(path: Path, *, manifest: dict[str, Any], comparisons: list[dict[str, Any]]) -> None:
    rows = [
        [
            item["candidate"],
            item["layer"],
            item["component"],
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
            fmt(item["target_win_rate"]),
            fmt(item["net_win_rate"]),
            fmt(item["target_strongest_win_rate"]),
            fmt(item["net_strongest_win_rate"]),
        ]
        for item in comparisons
    ]
    lines = [
        "# SCOTUS Component Trace Patch Summary",
        "",
        "## Purpose",
        "",
        "Patch token-mixer or MLP component outputs from paired replay traces, one layer/component at a time.",
        "Promotion requires prompt-matched wins over same-component random traces and contrast-source controls.",
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
                ["Components", ", ".join(manifest["components"])],
                ["Layers", ", ".join(str(layer) for layer in manifest["layers"])],
                ["Blends", ", ".join(str(alpha) for alpha in manifest["alphas"])],
                ["Random controls", manifest["random_controls"]],
                ["Source", manifest["source_example_id"]],
                ["Source control", manifest["control_source_example_id"]],
            ],
        ),
        "",
        "## Candidate vs Matched Controls",
        "",
        markdown_table(
            [
                "Candidate",
                "Layer",
                "Component",
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
                "Target win",
                "Net win",
                "Target strongest win",
                "Net strongest win",
            ],
            rows,
        ),
        "",
        "## Reading Rule",
        "",
        "- Compare candidate rows to the random controls for the same prompt, layer, component, and blend.",
        "- A component is not promotable if it loses to strongest random controls or if the contrast-source trace performs as well.",
        "- Keyword frame scores are only a screening diagnostic; any survivor needs manual proposition review.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--replay-rows", type=Path, default=DEFAULT_REPLAY_ROWS)
    parser.add_argument("--replay-run", type=Path, default=DEFAULT_REPLAY_RUN)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--prompt-bank", type=Path, default=DEFAULT_PROMPT_BANK)
    parser.add_argument("--layers", default="16,20")
    parser.add_argument("--components", default="mixer,mlp")
    parser.add_argument("--source-example-id", default="")
    parser.add_argument("--source-label", default="commerce_limits")
    parser.add_argument("--source-split", default="train")
    parser.add_argument("--control-source-example-id", default="")
    parser.add_argument("--control-source-label", default="commerce_authority")
    parser.add_argument("--control-source-split", default="train")
    parser.add_argument("--blends", default="0.1")
    parser.add_argument("--prompt-ids", default="2,3,4,5")
    parser.add_argument("--max-prompts", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_COMPLETE_ANSWER_TOKENS)
    add_short_budget_arg(parser)
    parser.add_argument("--random-controls", type=int, default=2)
    parser.add_argument("--seed", type=int, default=53)
    parser.add_argument("--score-mode", choices=["prompt_expected", "commerce_limits", "commerce_authority"], default="prompt_expected")
    parser.add_argument("--report-max-output-rows", type=int, default=220)
    parser.add_argument("--device-map", default="single")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    enforce_complete_answer_budget(
        args.max_new_tokens,
        allow_short=args.allow_short_answer_budget,
        purpose="component trace patch",
    )
    started = now_iso()
    layers = parse_int_list(args.layers)
    components = parse_components(args.components)
    specs = component_specs(layers, components)
    blends = parse_float_list(args.blends)
    out_dir = args.output_root / f"scotus_component_trace_patch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
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
    print(f"Loaded {len(prompt_specs)} prompts", flush=True)
    print(f"Component specs: {', '.join(spec.key for spec in specs)}", flush=True)
    print(f"Source trace: {source.example_id} ({source.frame_label})", flush=True)
    print(f"Control trace: {control_source.example_id} ({control_source.frame_label})", flush=True)

    tokenizer, model = load_model_and_tokenizer(args.model_path, args.device_map)
    layers_mod = transformer_layers(model)
    source_traces = capture_component_traces(
        model=model,
        tokenizer=tokenizer,
        layers_mod=layers_mod,
        source=source,
        specs=specs,
        max_steps=args.max_new_tokens,
    )
    control_traces = capture_component_traces(
        model=model,
        tokenizer=tokenizer,
        layers_mod=layers_mod,
        source=control_source,
        specs=specs,
        max_steps=args.max_new_tokens,
    )

    rows: list[dict[str, Any]] = []
    state = ComponentPatchState()
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
            make_row(
                spec=prompt_spec,
                condition="base",
                candidate=None,
                alpha=0.0,
                random_index=None,
                layer=None,
                component=None,
                target_candidate=None,
                output=output,
            )
        )

    for spec in specs:
        module = select_component_module(layers_mod[spec.layer], spec.component)
        candidate_name = component_candidate_name("component_trace", source, spec)
        control_name = component_candidate_name("component_trace_control", control_source, spec)
        random_traces = [
            random_trace_like(source_traces[spec.key], seed=args.seed + spec.layer * 1009 + random_idx * 100_003)
            for random_idx in range(max(0, args.random_controls))
        ]
        handle = install_component_patch_hook(module, state)
        try:
            for blend in blends:
                for random_idx, random_trace in enumerate(random_traces):
                    print(f"Generating random {spec.key}[{random_idx}] blend={blend}", flush=True)
                    state.trace = random_trace
                    state.blend = blend
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
                                candidate=f"random_for_{candidate_name}",
                                alpha=blend,
                                random_index=random_idx,
                                layer=spec.layer,
                                component=spec.component,
                                target_candidate=candidate_name,
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
                        trace=control_traces[spec.key],
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
                            candidate=control_name,
                            alpha=blend,
                            random_index=None,
                            layer=spec.layer,
                            component=spec.component,
                            target_candidate=candidate_name,
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
                        trace=source_traces[spec.key],
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
                            candidate=candidate_name,
                            alpha=blend,
                            random_index=None,
                            layer=spec.layer,
                            component=spec.component,
                            target_candidate=candidate_name,
                            output=output,
                        )
                    )
        finally:
            handle.remove()
            state.trace = None
            state.blend = 0.0
            state.step = 0

    add_base_deltas(rows)
    comparisons = component_comparisons(rows)
    direction_meta = [
        {
            "source": "minimal_pair_replay_component_trace",
            "name": component_candidate_name("component_trace", source, spec),
            "layer": spec.layer,
            "region": f"decode_step_{spec.component}_output",
            "group_field": "frame_label",
            "group_value": source.frame_label,
            "target_justice": source.frame_label,
            "reference_justice": "current_component_output",
            "n_target": 1,
            "n_reference": "",
            "raw_direction_norm": "",
            "top_features": [],
        }
        for spec in specs
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
        "alpha_scale": "component_trace_replacement_blend",
        "hidden_norm_reference": str(args.replay_run),
        "prompt_bank": str(args.prompt_bank),
        "candidate_names": [item["name"] for item in direction_meta],
        "source_example_id": source.example_id,
        "control_source_example_id": control_source.example_id,
        "layers": layers,
        "components": components,
        "alphas": blends,
        "random_controls": int(args.random_controls),
        "position": "decode_last_token_component_trace",
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
    write_jsonl(out_dir / "component_vs_matched_random.jsonl", comparisons)
    write_report(out_dir / "report.md", manifest=manifest, direction_meta=direction_meta, rows=rows)
    write_component_report(out_dir / "component_report.md", manifest=manifest, comparisons=comparisons)

    del model, tokenizer, layers_mod
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(f"Wrote {out_dir / 'component_report.md'}", flush=True)


if __name__ == "__main__":
    main()
