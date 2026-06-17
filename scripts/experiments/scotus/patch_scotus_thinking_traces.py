#!/usr/bin/env python3
"""Visible-thinking trajectory patching for SCOTUS Article III prompts.

Answer-only replay traces were too template-like to act as useful actuators.
This runner captures teacher-forced traces from previously generated Qwen
thinking text, patches those traces into new thinking generation, mechanically
closes the thought, and then generates an unpatched answer from the patched
thought. It is a localization harness, not a promotion run.
"""

from __future__ import annotations

import argparse
import gc
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import torch

from poke_scotus_sae_layers import (
    DEFAULT_OUTPUT_ROOT,
    first_parameter_device,
    load_model_and_tokenizer,
    load_prompt_specs,
    now_iso,
    parse_float_list,
    parse_int_list,
    select_prompt_specs,
    transformer_layers,
    write_json,
    write_jsonl,
)
from poke_scotus_thinking_bundle import (
    add_segment_base_deltas,
    aggregate_segments,
    clean_snippet,
    fmt,
    md_table,
    mean,
    row_for_two_stage,
    stdev,
)
from qwen_eval_budget import (
    DEFAULT_COMPLETE_ANSWER_TOKENS,
    MIN_COMPLETE_ANSWER_TOKENS,
    enforce_complete_answer_budget,
    enforce_complete_thinking_budget,
    qwen_thinking_answer_budget_metadata,
)
from run_scotus_thinking_smoke import IMITATION_RE, format_chat, strip_generation_specials


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MODEL = Path("/home/orwel/dev_genius/models/Qwen3.5-27B")
DEFAULT_PROMPT_BANK = PROJECT_ROOT / "data" / "scotus" / "scotus_article3_private_public_poke_prompts_v2.jsonl"
DEFAULT_SOURCE_GENERATIONS = (
    PROJECT_ROOT / "sweep_v4" / "scotus_thinking_lowrank_poke_20260501_204712" / "generations.jsonl"
)
COMPONENTS = ("residual", "mixer", "mlp")


@dataclass(frozen=True)
class ThinkingSource:
    name: str
    prompt_key: str
    condition: str
    prompt: str
    thinking: str
    source_kind: str


@dataclass(frozen=True)
class PatchSpec:
    layer: int
    component: str

    @property
    def key(self) -> str:
        return f"L{self.layer:02d}_{self.component}"


@dataclass(frozen=True)
class TokenWindow:
    start: int
    end: int | None

    @property
    def label(self) -> str:
        if self.end is None:
            return "all" if self.start == 0 else f"w{self.start:03d}_end"
        return f"w{self.start:03d}_{self.end:03d}"

    def as_manifest(self) -> dict[str, int | str | None]:
        return {"label": self.label, "start": self.start, "end": self.end}


@dataclass
class PatchState:
    trace: torch.Tensor | None = None
    blend: float = 0.0
    step: int = 0
    window_start: int = 0
    window_end: int | None = None


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


def parse_components(raw: str) -> list[str]:
    components = [item.strip() for item in raw.split(",") if item.strip()]
    unknown = sorted(set(components) - set(COMPONENTS))
    if unknown:
        raise ValueError(f"Unknown components {unknown}; allowed={list(COMPONENTS)}")
    return components


def patch_specs(layers: list[int], components: list[str]) -> list[PatchSpec]:
    return [PatchSpec(layer=layer, component=component) for layer in layers for component in components]


def parse_token_windows(raw: str) -> list[TokenWindow]:
    value = raw.strip().lower()
    if not value or value == "all":
        return [TokenWindow(start=0, end=None)]
    windows: list[TokenWindow] = []
    for item in value.split(","):
        token = item.strip()
        if not token:
            continue
        if ":" not in token:
            start = int(token)
            end = start + 1
        else:
            start_raw, end_raw = token.split(":", maxsplit=1)
            start = int(start_raw) if start_raw else 0
            end = int(end_raw) if end_raw else None
        if start < 0:
            raise ValueError(f"Patch token window start must be >= 0: {token!r}")
        if end is not None and end <= start:
            raise ValueError(f"Patch token window end must be > start: {token!r}")
        windows.append(TokenWindow(start=start, end=end))
    if not windows:
        raise ValueError(f"No valid patch token windows parsed from {raw!r}")
    return windows


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_")


def select_source(
    rows: list[dict[str, Any]],
    *,
    prompt_key: str,
    condition: str,
    source_kind: str,
) -> ThinkingSource:
    matches = [
        row
        for row in rows
        if str(row.get("prompt_key")) == prompt_key
        and str(row.get("condition")) == condition
        and str(row.get("thinking") or "").strip()
    ]
    if not matches:
        raise ValueError(f"No thinking source for prompt_key={prompt_key!r}, condition={condition!r}")
    row = matches[0]
    return ThinkingSource(
        name=f"{source_kind}_{safe_name(condition)}_{safe_name(prompt_key)}",
        prompt_key=str(row["prompt_key"]),
        condition=str(row["condition"]),
        prompt=str(row["prompt"]),
        thinking=str(row["thinking"]).strip(),
        source_kind=source_kind,
    )


def select_component_module(layer_mod: torch.nn.Module, component: str) -> torch.nn.Module:
    if component == "residual":
        return layer_mod
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


def install_capture_hook(
    *,
    layers_mod: Any,
    patch_spec: PatchSpec,
    positions: torch.Tensor,
    captured: dict[str, torch.Tensor],
) -> Any:
    module = select_component_module(layers_mod[patch_spec.layer], patch_spec.component)
    cpu_positions = positions.detach().cpu()

    def hook(_module: torch.nn.Module, _inp: tuple[Any, ...], out: Any) -> Any:
        hidden = output_tensor(out)
        captured[patch_spec.key] = hidden[0, cpu_positions.to(hidden.device), :].detach().float().cpu().contiguous()
        return out

    return module.register_forward_hook(hook)


@torch.inference_mode()
def capture_thinking_trace(
    *,
    model: torch.nn.Module,
    tokenizer: Any,
    layers_mod: Any,
    source: ThinkingSource,
    patch_spec: PatchSpec,
    max_steps: int,
) -> torch.Tensor:
    chat = format_chat(tokenizer, source.prompt, enable_thinking=True)
    prompt_ids = tokenizer(chat, add_special_tokens=False).input_ids
    thinking_ids = tokenizer(source.thinking, add_special_tokens=False).input_ids
    trace_len = min(max_steps, len(thinking_ids))
    if not prompt_ids or trace_len <= 0:
        raise RuntimeError(f"Cannot capture empty thinking trace for {source.name}")

    full_ids = torch.tensor([prompt_ids + thinking_ids], dtype=torch.long)
    positions = torch.arange(len(prompt_ids) - 1, len(prompt_ids) - 1 + trace_len, dtype=torch.long)
    input_device = first_parameter_device(model)
    full_ids = full_ids.to(input_device)
    attention_mask = torch.ones_like(full_ids, device=input_device)

    captured: dict[str, torch.Tensor] = {}
    handle = install_capture_hook(
        layers_mod=layers_mod,
        patch_spec=patch_spec,
        positions=positions,
        captured=captured,
    )
    try:
        _ = model(input_ids=full_ids, attention_mask=attention_mask, use_cache=False)
    finally:
        handle.remove()

    if patch_spec.key not in captured:
        raise RuntimeError(f"Did not capture trace for {patch_spec.key}")
    return captured[patch_spec.key]


def random_trace_like(trace: torch.Tensor, *, seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    noise = torch.randn(trace.shape, generator=generator, dtype=torch.float32)
    noise_norms = torch.linalg.vector_norm(noise, dim=1, keepdim=True).clamp(min=1e-12)
    source_norms = torch.linalg.vector_norm(trace.float(), dim=1, keepdim=True)
    return (noise / noise_norms * source_norms).contiguous()


def install_patch_hook(module: torch.nn.Module, state: PatchState) -> Any:
    def hook(_module: torch.nn.Module, _inp: tuple[Any, ...], out: Any) -> Any:
        if state.trace is None or state.blend == 0.0:
            return out
        if state.step < state.window_start:
            return out
        if state.window_end is not None and state.step >= state.window_end:
            return out
        if state.step >= int(state.trace.shape[0]):
            return out
        hidden = output_tensor(out)
        source_vec = state.trace[state.step].to(device=hidden.device, dtype=hidden.dtype)
        edited = hidden.clone()
        edited[:, -1, :] = edited[:, -1, :] + float(state.blend) * (source_vec - edited[:, -1, :])
        return replace_output_tensor(out, edited)

    return module.register_forward_hook(hook)


@torch.inference_mode()
def generate_manual(
    *,
    model: torch.nn.Module,
    tokenizer: Any,
    prompt: str,
    max_new_tokens: int,
    state: PatchState | None = None,
) -> dict[str, Any]:
    input_device = first_parameter_device(model)
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        add_special_tokens=False,
        truncation=True,
        max_length=4096,
    )
    input_ids = inputs["input_ids"].to(input_device)
    attention_mask = inputs["attention_mask"].to(input_device)
    generated: list[torch.Tensor] = []
    past_key_values: Any = None
    next_input = input_ids

    for step in range(max_new_tokens):
        if state is not None:
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
            [
                attention_mask,
                torch.ones((attention_mask.shape[0], 1), dtype=attention_mask.dtype, device=input_device),
            ],
            dim=1,
        )
        next_input = next_token
        if tokenizer.eos_token_id is not None and int(next_token[0, 0].item()) == int(tokenizer.eos_token_id):
            break

    if generated:
        generated_ids = torch.cat(generated, dim=1)[0]
    else:
        generated_ids = torch.empty((0,), dtype=torch.long)
    return {
        "raw_text": tokenizer.decode(generated_ids, skip_special_tokens=False).strip(),
        "text": tokenizer.decode(generated_ids, skip_special_tokens=True).strip(),
        "generated_tokens": int(generated_ids.numel()),
        "prompt_tokens": int(input_ids.shape[-1]),
    }


@torch.inference_mode()
def generate_two_stage(
    *,
    model: torch.nn.Module,
    tokenizer: Any,
    prompt: str,
    patch_state: PatchState | None,
    trace: torch.Tensor | None,
    blend: float,
    patch_window: TokenWindow | None = None,
    thought_tokens: int,
    answer_tokens: int,
) -> dict[str, Any]:
    chat = format_chat(tokenizer, prompt, enable_thinking=True)
    prefilled_open_think = chat.rstrip().endswith("<think>")
    if patch_state is not None:
        patch_state.trace = trace
        patch_state.blend = float(blend)
        patch_state.step = 0
        patch_state.window_start = 0 if patch_window is None else int(patch_window.start)
        patch_state.window_end = None if patch_window is None else patch_window.end
    try:
        thought_output = generate_manual(
            model=model,
            tokenizer=tokenizer,
            prompt=chat,
            max_new_tokens=thought_tokens,
            state=patch_state,
        )
    finally:
        if patch_state is not None:
            patch_state.trace = None
            patch_state.blend = 0.0
            patch_state.step = 0
            patch_state.window_start = 0
            patch_state.window_end = None

    thought_raw = strip_generation_specials(str(thought_output["raw_text"]))
    thought_text, separator, answer_prefix = thought_raw.partition("</think>")
    model_closed_thinking = bool(separator)
    thought_text = thought_text.strip()
    answer_prefix = strip_generation_specials(answer_prefix)
    answer_prompt = f"{chat}{thought_text}\n</think>\n\n{answer_prefix}"
    answer_output = generate_manual(
        model=model,
        tokenizer=tokenizer,
        prompt=answer_prompt,
        max_new_tokens=answer_tokens,
        state=None,
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


def make_candidate_name(prefix: str, source: ThinkingSource, patch_spec: PatchSpec, window: TokenWindow) -> str:
    return f"{prefix}_{source.name}_{patch_spec.key}_{window.label}"


def compare_by_target(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    comparisons: list[dict[str, Any]] = []
    for segment in ("thinking", "answer"):
        random_by_target: dict[tuple[str, int, float], list[dict[str, Any]]] = defaultdict(list)
        source_by_target: dict[tuple[str, int, float], list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            target = str(row.get("target_candidate") or "")
            if not target:
                continue
            key = (target, int(row["prompt_id"]), float(row["alpha"]))
            if row["condition"] == "random_unit":
                random_by_target[key].append(row)
            elif row["condition"] == "source_control":
                source_by_target[key].append(row)

        candidate_groups: dict[tuple[str, float], list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            if row["condition"] == "sae_poke" and row.get("candidate"):
                candidate_groups[(str(row["candidate"]), float(row["alpha"]))].append(row)

        for (candidate, alpha), group_rows in sorted(candidate_groups.items()):
            target_diffs: list[float] = []
            net_diffs: list[float] = []
            candidate_target: list[float] = []
            candidate_net: list[float] = []
            random_target: list[float] = []
            random_net: list[float] = []
            source_target: list[float] = []
            source_net: list[float] = []
            target_wins = 0
            net_wins = 0
            strongest_target_wins = 0
            strongest_net_wins = 0
            matched = 0
            for row in group_rows:
                target = str(row.get("target_candidate") or row["candidate"])
                key = (target, int(row["prompt_id"]), float(alpha))
                random_rows = random_by_target.get(key, [])
                if not random_rows:
                    continue
                eval_item = row[f"{segment}_proposition_frame_eval"]
                candidate_value = float(eval_item.get("delta_target_hits_vs_base", 0.0))
                candidate_net_value = float(eval_item.get("delta_target_minus_contrast_vs_base", 0.0))
                random_values = [
                    float(item[f"{segment}_proposition_frame_eval"].get("delta_target_hits_vs_base", 0.0))
                    for item in random_rows
                ]
                random_net_values = [
                    float(item[f"{segment}_proposition_frame_eval"].get("delta_target_minus_contrast_vs_base", 0.0))
                    for item in random_rows
                ]
                source_rows = source_by_target.get(key, [])
                source_values = [
                    float(item[f"{segment}_proposition_frame_eval"].get("delta_target_hits_vs_base", 0.0))
                    for item in source_rows
                ]
                source_net_values = [
                    float(item[f"{segment}_proposition_frame_eval"].get("delta_target_minus_contrast_vs_base", 0.0))
                    for item in source_rows
                ]
                random_mean = mean(random_values)
                random_net_mean = mean(random_net_values)
                matched += 1
                candidate_target.append(candidate_value)
                candidate_net.append(candidate_net_value)
                random_target.append(random_mean)
                random_net.append(random_net_mean)
                source_target.extend(source_values)
                source_net.extend(source_net_values)
                target_diffs.append(candidate_value - random_mean)
                net_diffs.append(candidate_net_value - random_net_mean)
                target_wins += int(candidate_value > random_mean)
                net_wins += int(candidate_net_value > random_net_mean)
                strongest_target_wins += int(candidate_value > max(random_values))
                strongest_net_wins += int(candidate_net_value > max(random_net_values))
            comparisons.append(
                {
                    "segment": segment,
                    "candidate": candidate,
                    "alpha": alpha,
                    "n": matched,
                    "candidate_mean_target": mean(candidate_target),
                    "random_mean_target": mean(random_target),
                    "matched_target": mean(target_diffs),
                    "candidate_mean_net": mean(candidate_net),
                    "random_mean_net": mean(random_net),
                    "matched_net": mean(net_diffs),
                    "source_control_mean_target": mean(source_target),
                    "source_control_mean_net": mean(source_net),
                    "target_win_rate": 0.0 if matched == 0 else float(target_wins / matched),
                    "net_win_rate": 0.0 if matched == 0 else float(net_wins / matched),
                    "target_strongest_win_rate": 0.0 if matched == 0 else float(strongest_target_wins / matched),
                    "net_strongest_win_rate": 0.0 if matched == 0 else float(strongest_net_wins / matched),
                    "matched_target_sd": stdev(target_diffs),
                    "matched_net_sd": stdev(net_diffs),
                }
            )
    return comparisons


def write_report(
    path: Path,
    *,
    manifest: dict[str, Any],
    summaries: list[dict[str, Any]],
    comparisons: list[dict[str, Any]],
    rows: list[dict[str, Any]],
) -> None:
    summary_rows = [
        [
            item["segment"],
            item["condition"],
            fmt(item.get("candidate")),
            fmt(item.get("alpha")),
            item["n"],
            fmt(item["mean_delta_target_hits_vs_base"]),
            fmt(item["mean_delta_target_minus_contrast_vs_base"]),
            fmt(item["answer_nonempty_rate"]),
            fmt(item["imitation_marker_rate"]),
        ]
        for item in summaries
    ]
    comparison_rows = [
        [
            item["segment"],
            item["candidate"],
            fmt(item["alpha"]),
            item["n"],
            fmt(item["matched_target"]),
            fmt(item["matched_net"]),
            fmt(item["source_control_mean_target"]),
            fmt(item["source_control_mean_net"]),
            fmt(item["target_strongest_win_rate"]),
            fmt(item["net_strongest_win_rate"]),
        ]
        for item in comparisons
    ]
    lines = [
        "# SCOTUS Visible-Thinking Trace Patch",
        "",
        "## Configuration",
        "",
        f"- Model: `{manifest['model_path']}`",
        f"- Source generations: `{manifest['source_generations']}`",
        f"- Source/control: `{manifest['source_prompt_key']}` / `{manifest['control_prompt_key']}`",
        f"- Patch specs: `{', '.join(manifest['patch_spec_keys'])}`",
        f"- Patch token windows: `{', '.join(item['label'] for item in manifest['patch_token_windows'])}`",
        f"- Blends: `{', '.join(str(item) for item in manifest['alphas'])}`",
        f"- Random controls: `{manifest['random_controls']}`",
        f"- Thought/answer token budgets: `{manifest['thought_tokens']}`/`{manifest['answer_tokens']}`",
        f"- Short answer-budget smoke: `{manifest['short_answer_budget']}`",
        f"- Prompts: `{len(manifest['prompt_keys'])}`",
        "",
        "## Segment Summary",
        "",
        *md_table(
            [
                "segment",
                "condition",
                "candidate",
                "blend",
                "n",
                "target_delta",
                "net_delta",
                "answer_rate",
                "mask_rate",
            ],
            summary_rows,
        ),
        "",
        "## Candidate vs Matched Controls",
        "",
        *md_table(
            [
                "segment",
                "candidate",
                "blend",
                "n",
                "target-minus-random",
                "net-minus-random",
                "source target",
                "source net",
                "target strongest win",
                "net strongest win",
            ],
            comparison_rows,
        ),
        "",
        "## Samples",
        "",
    ]
    for row in rows:
        if row["condition"] not in {"base", "sae_poke", "source_control"}:
            continue
        lines.extend(
            [
                f"### {row['prompt_key']} / {row['condition']} / {row.get('candidate')}",
                "",
                f"- blend: `{row.get('alpha')}`",
                f"- random index: `{row.get('random_index')}`",
                f"- patch token window: `{row.get('patch_token_window')}`",
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
            "This localization smoke only nominates a window if the candidate beats strongest random controls on visible thinking and is not matched by the source-control trace.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--source-generations", type=Path, default=DEFAULT_SOURCE_GENERATIONS)
    parser.add_argument("--prompt-bank", type=Path, default=DEFAULT_PROMPT_BANK)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--source-prompt-key", default="A3_PRIV_02_bankruptcy_counterclaim")
    parser.add_argument("--control-prompt-key", default="A3_PUBLIC_01_benefits_eligibility")
    parser.add_argument("--source-condition", default="base")
    parser.add_argument("--control-condition", default="base")
    parser.add_argument("--layers", default="4,8")
    parser.add_argument("--components", default="mlp")
    parser.add_argument("--blends", default="0.25")
    parser.add_argument("--prompt-ids", default="1,4")
    parser.add_argument("--max-prompts", type=int, default=2)
    parser.add_argument("--random-controls", type=int, default=1)
    parser.add_argument(
        "--patch-token-windows",
        default="all",
        help="Generated-thinking token windows to patch, e.g. 'all' or '0:32,32:64,64:96'.",
    )
    parser.add_argument("--trace-steps", type=int, default=192)
    parser.add_argument("--thought-tokens", type=int, default=DEFAULT_COMPLETE_ANSWER_TOKENS)
    parser.add_argument("--answer-tokens", type=int, default=DEFAULT_COMPLETE_ANSWER_TOKENS)
    parser.add_argument(
        "--allow-short-thinking-budget",
        action="store_true",
        help=(
            f"Permit visible-thinking budgets below {MIN_COMPLETE_ANSWER_TOKENS} tokens. "
            "Short-budget thinking runs are smoke/localization only and must not be used for promotion."
        ),
    )
    parser.add_argument(
        "--allow-short-answer-budget",
        action="store_true",
        help=(
            f"Permit answer budgets below {MIN_COMPLETE_ANSWER_TOKENS} tokens. "
            "Short-budget runs are smoke/localization only and must not be used for promotion."
        ),
    )
    parser.add_argument("--device-map", default="single")
    parser.add_argument("--seed", type=int, default=20260514)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    enforce_complete_thinking_budget(
        args.thought_tokens,
        allow_short=args.allow_short_thinking_budget,
        purpose="SCOTUS generated-thinking trace patch run",
    )
    enforce_complete_answer_budget(
        args.answer_tokens,
        allow_short=args.allow_short_answer_budget,
        label="answer_tokens",
        purpose="SCOTUS generated-thinking trace patch answer run",
    )
    started = now_iso()
    out_dir = args.output_root / f"scotus_thinking_trace_patch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)

    source_rows = read_jsonl(args.source_generations)
    source = select_source(
        source_rows,
        prompt_key=args.source_prompt_key,
        condition=args.source_condition,
        source_kind="source",
    )
    control = select_source(
        source_rows,
        prompt_key=args.control_prompt_key,
        condition=args.control_condition,
        source_kind="control",
    )
    specs = patch_specs(parse_int_list(args.layers), parse_components(args.components))
    token_windows = parse_token_windows(args.patch_token_windows)
    blends = parse_float_list(args.blends)
    prompt_specs = select_prompt_specs(load_prompt_specs(args.prompt_bank), args.prompt_ids, args.max_prompts)
    print(f"Loaded {len(prompt_specs)} prompts", flush=True)
    print(f"Source thinking: {source.name}", flush=True)
    print(f"Control thinking: {control.name}", flush=True)
    print(f"Patch specs: {', '.join(spec.key for spec in specs)}", flush=True)
    print(f"Patch token windows: {', '.join(window.label for window in token_windows)}", flush=True)

    tokenizer, model = load_model_and_tokenizer(args.model_path, args.device_map)
    layers_mod = transformer_layers(model)
    rows: list[dict[str, Any]] = []
    trace_meta: list[dict[str, Any]] = []

    print("Generating two-stage baseline", flush=True)
    for prompt_spec in prompt_specs:
        output = generate_two_stage(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt_spec.prompt,
            patch_state=None,
            trace=None,
            blend=0.0,
            thought_tokens=args.thought_tokens,
            answer_tokens=args.answer_tokens,
        )
        row = row_for_two_stage(
            spec=prompt_spec,
            condition="base",
            candidate=None,
            alpha=0.0,
            random_index=None,
            layer=None,
            output=output,
        )
        row["component"] = None
        row["target_candidate"] = None
        rows.append(row)

    for spec in specs:
        module = select_component_module(layers_mod[spec.layer], spec.component)
        patch_state = PatchState()
        handle = install_patch_hook(module, patch_state)
        try:
            print(f"Capturing source/control thinking traces for {spec.key}", flush=True)
            source_trace = capture_thinking_trace(
                model=model,
                tokenizer=tokenizer,
                layers_mod=layers_mod,
                source=source,
                patch_spec=spec,
                max_steps=args.trace_steps,
            )
            control_trace = capture_thinking_trace(
                model=model,
                tokenizer=tokenizer,
                layers_mod=layers_mod,
                source=control,
                patch_spec=spec,
                max_steps=args.trace_steps,
            )
            trace_meta.extend(
                [
                    {
                        "name": make_candidate_name("thinking_trace", source, spec, TokenWindow(start=0, end=None)),
                        "source_kind": source.source_kind,
                        "prompt_key": source.prompt_key,
                        "condition": source.condition,
                        "layer": spec.layer,
                        "component": spec.component,
                        "trace_steps": int(source_trace.shape[0]),
                        "hidden_dim": int(source_trace.shape[1]),
                        "patch_token_windows": [window.as_manifest() for window in token_windows],
                    },
                    {
                        "name": make_candidate_name(
                            "thinking_trace_control",
                            control,
                            spec,
                            TokenWindow(start=0, end=None),
                        ),
                        "source_kind": control.source_kind,
                        "prompt_key": control.prompt_key,
                        "condition": control.condition,
                        "layer": spec.layer,
                        "component": spec.component,
                        "trace_steps": int(control_trace.shape[0]),
                        "hidden_dim": int(control_trace.shape[1]),
                        "patch_token_windows": [window.as_manifest() for window in token_windows],
                    },
                ]
            )
            random_traces = [
                random_trace_like(source_trace, seed=args.seed + spec.layer * 1009 + idx * 100_003)
                for idx in range(max(0, args.random_controls))
            ]

            for blend in blends:
                for window in token_windows:
                    source_name = make_candidate_name("thinking_trace", source, spec, window)
                    control_name = make_candidate_name("thinking_trace_control", control, spec, window)
                    for random_idx, random_trace in enumerate(random_traces):
                        print(
                            f"Generating random {spec.key}/{window.label}[{random_idx}] blend={blend}",
                            flush=True,
                        )
                        for prompt_spec in prompt_specs:
                            output = generate_two_stage(
                                model=model,
                                tokenizer=tokenizer,
                                prompt=prompt_spec.prompt,
                                patch_state=patch_state,
                                trace=random_trace,
                                blend=float(blend),
                                patch_window=window,
                                thought_tokens=args.thought_tokens,
                                answer_tokens=args.answer_tokens,
                            )
                            row = row_for_two_stage(
                                spec=prompt_spec,
                                condition="random_unit",
                                candidate=f"random_for_{source_name}",
                                alpha=float(blend),
                                random_index=random_idx,
                                layer=spec.layer,
                                output=output,
                            )
                            row["component"] = spec.component
                            row["patch_token_window"] = window.label
                            row["patch_token_window_start"] = window.start
                            row["patch_token_window_end"] = window.end
                            row["target_candidate"] = source_name
                            rows.append(row)

                    print(f"Generating source-control {spec.key}/{window.label} blend={blend}", flush=True)
                    for prompt_spec in prompt_specs:
                        output = generate_two_stage(
                            model=model,
                            tokenizer=tokenizer,
                            prompt=prompt_spec.prompt,
                            patch_state=patch_state,
                            trace=control_trace,
                            blend=float(blend),
                            patch_window=window,
                            thought_tokens=args.thought_tokens,
                            answer_tokens=args.answer_tokens,
                        )
                        row = row_for_two_stage(
                            spec=prompt_spec,
                            condition="source_control",
                            candidate=control_name,
                            alpha=float(blend),
                            random_index=None,
                            layer=spec.layer,
                            output=output,
                        )
                        row["component"] = spec.component
                        row["patch_token_window"] = window.label
                        row["patch_token_window_start"] = window.start
                        row["patch_token_window_end"] = window.end
                        row["target_candidate"] = source_name
                        rows.append(row)

                    print(f"Generating candidate {spec.key}/{window.label} blend={blend}", flush=True)
                    for prompt_spec in prompt_specs:
                        output = generate_two_stage(
                            model=model,
                            tokenizer=tokenizer,
                            prompt=prompt_spec.prompt,
                            patch_state=patch_state,
                            trace=source_trace,
                            blend=float(blend),
                            patch_window=window,
                            thought_tokens=args.thought_tokens,
                            answer_tokens=args.answer_tokens,
                        )
                        row = row_for_two_stage(
                            spec=prompt_spec,
                            condition="sae_poke",
                            candidate=source_name,
                            alpha=float(blend),
                            random_index=None,
                            layer=spec.layer,
                            output=output,
                        )
                        row["component"] = spec.component
                        row["patch_token_window"] = window.label
                        row["patch_token_window_start"] = window.start
                        row["patch_token_window_end"] = window.end
                        row["target_candidate"] = source_name
                        rows.append(row)
        finally:
            handle.remove()

    add_segment_base_deltas(rows)
    summaries = aggregate_segments(rows)
    comparisons = compare_by_target(rows)
    manifest = {
        "started_at": started,
        "finished_at": now_iso(),
        "model_path": str(args.model_path),
        "source_generations": str(args.source_generations),
        "prompt_bank": str(args.prompt_bank),
        "output_dir": str(out_dir),
        "source_prompt_key": args.source_prompt_key,
        "control_prompt_key": args.control_prompt_key,
        "source_condition": args.source_condition,
        "control_condition": args.control_condition,
        "patch_spec_keys": [spec.key for spec in specs],
        "patch_token_windows": [window.as_manifest() for window in token_windows],
        "layers": [spec.layer for spec in specs],
        "components": [spec.component for spec in specs],
        "alphas": blends,
        "random_controls": int(args.random_controls),
        "trace_steps": int(args.trace_steps),
        "thought_tokens": int(args.thought_tokens),
        "answer_tokens": int(args.answer_tokens),
        **qwen_thinking_answer_budget_metadata(args.thought_tokens, args.answer_tokens),
        "seed": int(args.seed),
        "prompt_ids": [spec.prompt_id for spec in prompt_specs],
        "prompt_keys": [spec.prompt_key for spec in prompt_specs],
    }
    write_json(out_dir / "manifest.json", manifest)
    write_jsonl(out_dir / "trace_meta.jsonl", trace_meta)
    write_jsonl(out_dir / "generations.jsonl", rows)
    write_jsonl(out_dir / "segment_score_summary.jsonl", summaries)
    write_jsonl(out_dir / "candidate_vs_matched_controls.jsonl", comparisons)
    write_report(out_dir / "report.md", manifest=manifest, summaries=summaries, comparisons=comparisons, rows=rows)

    del model, tokenizer, layers_mod
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(f"Wrote {out_dir / 'report.md'}", flush=True)


if __name__ == "__main__":
    main()
