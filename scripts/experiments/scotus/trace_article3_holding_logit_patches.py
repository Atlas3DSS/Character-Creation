#!/usr/bin/env python3
"""Causal-trace Article III holding logits by patching generated thought states.

This is a cheap localization screen, not a promotion run. It patches hidden
states from private-leaning or public-control generated thoughts into
public-leaning target prompts, then scores a fixed private-vs-public holding
logprob margin. A site only becomes an actuator candidate if it later survives
long no-mask generation with random/source controls.
"""

from __future__ import annotations

import argparse
import gc
import json
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
    fmt,
    markdown_table,
    now_iso,
    output_tensor,
    safe_key,
    select_component_module,
    transformer_layers,
    write_json,
    write_jsonl,
)
from localize_article3_generated_thought_baselines import source_budget_meta  # noqa: E402
from poke_scotus_sae_layers import first_parameter_device, load_model_and_tokenizer  # noqa: E402
from qwen_eval_budget import MIN_COMPLETE_ANSWER_TOKENS, SHORT_BUDGET_CLAIM_WARNING  # noqa: E402
from run_scotus_thinking_smoke import format_chat  # noqa: E402


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MODEL = Path("/home/orwel/dev_genius/models/Qwen3.5-27B")
DEFAULT_SOURCE_GENERATIONS = (
    PROJECT_ROOT / "sweep_v4" / "scotus_thinking_localized_direction_poke_20260502_005241" / "generations.jsonl"
)
DEFAULT_LOCALIZATION_RUN = (
    PROJECT_ROOT / "sweep_v4" / "scotus_article3_generated_thought_baseline_localization_20260502_043317"
)
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "sweep_v4"
PRIVATE_LABEL = " Article III bars the non-Article-III adjudication."
PUBLIC_LABEL = " Article III permits the non-Article-III adjudication."


@dataclass(frozen=True)
class TraceSite:
    layer: int
    component: str
    region: str

    @property
    def key(self) -> str:
        return f"L{self.layer:02d}_{self.component}_{safe_key(self.region)}"


@dataclass(frozen=True)
class RenderedTrace:
    prompt_id: int
    prompt_key: str
    prompt: str
    thinking: str
    prefix: str
    prompt_tokens: int
    thought_start: int
    thought_end: int
    pre_answer_last: int


@dataclass(frozen=True)
class PatchPayload:
    positions: list[int]
    values: torch.Tensor
    mode: str


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


def parse_float_list(raw: str) -> list[float]:
    return [float(item.strip()) for item in raw.split(",") if item.strip()]


def replace_output_tensor(output: Any, tensor: torch.Tensor) -> Any:
    if isinstance(output, tuple):
        return (tensor,) + output[1:]
    return tensor


def load_top_sites(localization_run: Path, *, top_sites: int) -> list[TraceSite]:
    rows = read_jsonl(localization_run / "direction_meta.jsonl")[:top_sites]
    sites = [
        TraceSite(layer=int(row["layer"]), component=str(row["component"]), region=str(row["region"]))
        for row in rows
    ]
    if not sites:
        raise ValueError(f"No sites loaded from {localization_run / 'direction_meta.jsonl'}")
    return sites


def render_trace(tokenizer: Any, row: dict[str, Any]) -> RenderedTrace:
    prompt = str(row["prompt"])
    thinking = str(row.get("thinking") or "").strip()
    if not thinking:
        raise ValueError(f"Empty thinking for prompt_id={row.get('prompt_id')}")
    chat = format_chat(tokenizer, prompt, enable_thinking=True)
    before_close = f"{chat}{thinking}"
    prefix = f"{before_close}\n</think>\n\nFinal holding:"
    chat_ids = tokenizer(chat, add_special_tokens=False).input_ids
    before_close_ids = tokenizer(before_close, add_special_tokens=False).input_ids
    prefix_ids = tokenizer(prefix, add_special_tokens=False).input_ids
    thought_start = min(len(chat_ids), max(0, len(prefix_ids) - 1))
    thought_end = max(thought_start + 1, min(len(before_close_ids), len(prefix_ids)))
    return RenderedTrace(
        prompt_id=int(row["prompt_id"]),
        prompt_key=str(row["prompt_key"]),
        prompt=prompt,
        thinking=thinking,
        prefix=prefix,
        prompt_tokens=len(prefix_ids),
        thought_start=thought_start,
        thought_end=thought_end,
        pre_answer_last=max(0, len(prefix_ids) - 1),
    )


def region_positions(rendered: RenderedTrace, region: str) -> list[int]:
    thought_positions = list(range(rendered.thought_start, rendered.thought_end))
    if not thought_positions:
        raise ValueError(f"No thought positions for {rendered.prompt_key}")
    if region == "pre_answer_last":
        return [rendered.pre_answer_last]
    if region == "tail32_mean":
        return thought_positions[-32:]
    if region == "thought_tail16_mean":
        return thought_positions[-16:]
    if region == "thought_mean":
        return thought_positions
    raise ValueError(f"Unsupported trace region: {region}")


@torch.inference_mode()
def capture_site(
    *,
    model: torch.nn.Module,
    tokenizer: Any,
    layers_mod: Any,
    rendered: RenderedTrace,
    site: TraceSite,
) -> torch.Tensor:
    positions = region_positions(rendered, site.region)
    inputs = tokenizer(
        rendered.prefix,
        return_tensors="pt",
        add_special_tokens=False,
        truncation=True,
        max_length=4096,
    )
    input_device = first_parameter_device(model)
    input_ids = inputs["input_ids"].to(input_device)
    attention_mask = inputs["attention_mask"].to(input_device)
    pos_tensor = torch.tensor(positions, dtype=torch.long)
    captured: dict[str, torch.Tensor] = {}
    module = select_component_module(layers_mod[site.layer], site.component)

    def hook(_module: torch.nn.Module, _inp: tuple[Any, ...], out: Any) -> Any:
        hidden = output_tensor(out)
        clipped = pos_tensor.clamp(max=hidden.shape[1] - 1).to(hidden.device)
        captured["values"] = hidden[0, clipped, :].detach().float().cpu().contiguous()
        return out

    handle = module.register_forward_hook(hook)
    try:
        _ = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
    finally:
        handle.remove()
    if "values" not in captured:
        raise RuntimeError(f"Did not capture {site.key} for {rendered.prompt_key}")
    return captured["values"]


def make_payload(*, source_values: torch.Tensor, target: RenderedTrace, site: TraceSite) -> PatchPayload:
    positions = region_positions(target, site.region)
    if site.region == "thought_mean":
        mean_vec = source_values.float().mean(dim=0, keepdim=True)
        values = mean_vec.repeat(len(positions), 1).contiguous()
        return PatchPayload(positions=positions, values=values, mode="source_mean_repeated")
    n = min(len(positions), int(source_values.shape[0]))
    if n <= 0:
        raise ValueError(f"Empty patch payload for {site.key}")
    return PatchPayload(
        positions=positions[-n:],
        values=source_values[-n:].float().contiguous(),
        mode="right_aligned_tokens",
    )


def install_patch_hook(
    *,
    layers_mod: Any,
    site: TraceSite,
    payload: PatchPayload,
    blend: float,
) -> Any:
    module = select_component_module(layers_mod[site.layer], site.component)
    positions_cpu = torch.tensor(payload.positions, dtype=torch.long)
    values_cpu = payload.values.float().contiguous()

    def hook(_module: torch.nn.Module, _inp: tuple[Any, ...], out: Any) -> Any:
        hidden = output_tensor(out)
        positions = positions_cpu.clamp(max=hidden.shape[1] - 1).to(hidden.device)
        source = values_cpu.to(device=hidden.device, dtype=hidden.dtype)
        edited = hidden.clone()
        edited[:, positions, :] = edited[:, positions, :] + float(blend) * (source.unsqueeze(0) - edited[:, positions, :])
        return replace_output_tensor(out, edited)

    return module.register_forward_hook(hook)


@torch.inference_mode()
def score_label(
    *,
    model: torch.nn.Module,
    tokenizer: Any,
    layers_mod: Any,
    rendered: RenderedTrace,
    label_text: str,
    max_label_tokens: int | None = None,
    site: TraceSite | None = None,
    payload: PatchPayload | None = None,
    blend: float = 0.0,
) -> dict[str, float | int]:
    prefix_ids = tokenizer(rendered.prefix, add_special_tokens=False).input_ids
    label_ids = tokenizer(label_text, add_special_tokens=False).input_ids
    if max_label_tokens is not None and max_label_tokens > 0:
        label_ids = label_ids[:max_label_tokens]
    if not prefix_ids or not label_ids:
        raise ValueError("Cannot score empty prefix or label")
    full_ids = prefix_ids + label_ids
    input_device = first_parameter_device(model)
    input_ids = torch.tensor([full_ids], dtype=torch.long, device=input_device)
    attention_mask = torch.ones_like(input_ids, device=input_device)
    handle = None
    if site is not None and payload is not None and blend != 0.0:
        handle = install_patch_hook(layers_mod=layers_mod, site=site, payload=payload, blend=blend)
    try:
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
    finally:
        if handle is not None:
            handle.remove()
    start = len(prefix_ids) - 1
    end = start + len(label_ids)
    logits = outputs.logits[0, start:end, :]
    log_probs = torch.log_softmax(logits.float(), dim=-1)
    label_tensor = torch.tensor(label_ids, dtype=torch.long, device=log_probs.device)
    selected = log_probs[torch.arange(len(label_ids), device=log_probs.device), label_tensor]
    return {
        "label_tokens": int(len(label_ids)),
        "logprob_sum": float(selected.sum().detach().cpu().item()),
        "logprob_mean": float(selected.mean().detach().cpu().item()),
    }


def score_margin(
    *,
    model: torch.nn.Module,
    tokenizer: Any,
    layers_mod: Any,
    rendered: RenderedTrace,
    private_label: str,
    public_label: str,
    max_label_tokens: int | None = None,
    site: TraceSite | None = None,
    payload: PatchPayload | None = None,
    blend: float = 0.0,
) -> dict[str, float | int]:
    private = score_label(
        model=model,
        tokenizer=tokenizer,
        layers_mod=layers_mod,
        rendered=rendered,
        label_text=private_label,
        max_label_tokens=max_label_tokens,
        site=site,
        payload=payload,
        blend=blend,
    )
    public = score_label(
        model=model,
        tokenizer=tokenizer,
        layers_mod=layers_mod,
        rendered=rendered,
        label_text=public_label,
        max_label_tokens=max_label_tokens,
        site=site,
        payload=payload,
        blend=blend,
    )
    return {
        "private_logprob_mean": float(private["logprob_mean"]),
        "public_logprob_mean": float(public["logprob_mean"]),
        "private_logprob_sum": float(private["logprob_sum"]),
        "public_logprob_sum": float(public["logprob_sum"]),
        "private_tokens": int(private["label_tokens"]),
        "public_tokens": int(public["label_tokens"]),
        "margin_mean": float(private["logprob_mean"]) - float(public["logprob_mean"]),
        "margin_sum": float(private["logprob_sum"]) - float(public["logprob_sum"]),
    }


def aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, str, str, float], dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        if row["condition"] == "base":
            continue
        key = (int(row["layer"]), str(row["component"]), str(row["region"]), float(row["blend"]))
        grouped[key][str(row["source_kind"])].append(float(row["delta_margin_mean"]))
    out: list[dict[str, Any]] = []
    for (layer, component, region, blend), values in grouped.items():
        private_vals = values.get("private_source", [])
        public_vals = values.get("public_source_control", [])
        private_mean = float(np.mean(private_vals)) if private_vals else 0.0
        public_mean = float(np.mean(public_vals)) if public_vals else 0.0
        out.append(
            {
                "layer": layer,
                "component": component,
                "region": region,
                "blend": blend,
                "private_source_mean_delta": private_mean,
                "public_source_control_mean_delta": public_mean,
                "private_minus_public_control_delta": private_mean - public_mean,
                "n_private_source": len(private_vals),
                "n_public_source_control": len(public_vals),
            }
        )
    out.sort(
        key=lambda row: (
            float(row["private_minus_public_control_delta"]),
            float(row["private_source_mean_delta"]),
        ),
        reverse=True,
    )
    return out


def write_report(path: Path, *, manifest: dict[str, Any], aggregate_rows: list[dict[str, Any]], rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Article III Holding-Logit Causal Trace",
        "",
        "## Purpose",
        "",
        (
            "Patch generated-thought hidden states into public-leaning Article III targets and score a fixed "
            "private-vs-public final-holding logprob margin. This is localization evidence only, not a no-mask "
            "generation result."
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
                ["Localization run", manifest["localization_run"]],
                ["Target prompt ids", ",".join(str(item) for item in manifest["target_prompt_ids"])],
                ["Private source ids", ",".join(str(item) for item in manifest["private_source_prompt_ids"])],
                ["Public source-control ids", ",".join(str(item) for item in manifest["public_control_prompt_ids"])],
                ["Top sites", manifest["top_sites"]],
                ["Blends", ",".join(str(item) for item in manifest["blends"])],
                ["Source budget note", manifest["source_budget_note"]],
                ["Output dir", manifest["output_dir"]],
            ],
        ),
        "",
        "## Top Aggregate Sites",
        "",
        markdown_table(
            [
                "Rank",
                "Layer",
                "Component",
                "Region",
                "Blend",
                "Private delta",
                "Public-control delta",
                "Private - control",
            ],
            [
                [
                    idx + 1,
                    row["layer"],
                    row["component"],
                    row["region"],
                    fmt(row["blend"]),
                    fmt(row["private_source_mean_delta"]),
                    fmt(row["public_source_control_mean_delta"]),
                    fmt(row["private_minus_public_control_delta"]),
                ]
                for idx, row in enumerate(aggregate_rows[:20])
            ],
        ),
        "",
        "## Label Probe",
        "",
        f"- Private label: `{str(manifest['private_label']).strip()}`",
        f"- Public label: `{str(manifest['public_label']).strip()}`",
        "- Margin is mean logprob(private label tokens) minus mean logprob(public label tokens).",
        "- Positive deltas mean the patch made the private holding label more preferred relative to the public label.",
        "",
        "## Interpretation",
        "",
        "- A positive private-minus-control aggregate is a candidate-localization signal only.",
        "- This does not test full visible-reasoning generation or final-answer adoption.",
        "- Any promoted actuator still needs a frozen no-mask generation gate with complete Qwen budgets, random/source controls, conclusion-polarity scoring, and manual review.",
        "",
        "## Artifacts",
        "",
        f"- Manifest: `{manifest['output_dir']}/manifest.json`.",
        f"- Patch rows: `{manifest['output_dir']}/patch_rows.jsonl`.",
        f"- Aggregate rows: `{manifest['output_dir']}/aggregate.jsonl`.",
    ]
    if rows:
        sample = rows[0]
        lines.extend(
            [
                "",
                "Sample row:",
                "",
                "```json",
                json.dumps(sample, indent=2, sort_keys=True),
                "```",
            ]
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--source-generations", type=Path, default=DEFAULT_SOURCE_GENERATIONS)
    parser.add_argument("--localization-run", type=Path, default=DEFAULT_LOCALIZATION_RUN)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--target-prompt-ids", default="2,4")
    parser.add_argument("--private-source-prompt-ids", default="1")
    parser.add_argument("--public-control-prompt-ids", default="6")
    parser.add_argument("--top-sites", type=int, default=6)
    parser.add_argument("--blends", default="0.25")
    parser.add_argument("--private-label", default=PRIVATE_LABEL)
    parser.add_argument("--public-label", default=PUBLIC_LABEL)
    parser.add_argument("--device-map", default="single")
    parser.add_argument("--allow-short-source-budget", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = now_iso()
    source_budget = source_budget_meta(args.source_generations, allow_short=args.allow_short_source_budget)
    if source_budget["source_short_budget"] or source_budget["source_unknown_budget"]:
        print(SHORT_BUDGET_CLAIM_WARNING, flush=True)

    target_ids = parse_int_list(args.target_prompt_ids)
    private_source_ids = parse_int_list(args.private_source_prompt_ids)
    public_control_ids = parse_int_list(args.public_control_prompt_ids)
    blends = parse_float_list(args.blends)
    sites = load_top_sites(args.localization_run, top_sites=args.top_sites)

    out_dir = args.output_root / f"scotus_article3_holding_logit_trace_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)

    generation_rows = read_jsonl(args.source_generations)
    base_rows = {
        int(row["prompt_id"]): row
        for row in generation_rows
        if str(row.get("condition")) == "base" and int(row.get("prompt_id", -1)) in set(target_ids + private_source_ids + public_control_ids)
    }
    missing = sorted(set(target_ids + private_source_ids + public_control_ids) - set(base_rows))
    if missing:
        raise ValueError(f"Missing base source generation rows for prompt ids: {missing}")

    tokenizer, model = load_model_and_tokenizer(args.model_path, args.device_map)
    layers_mod = transformer_layers(model)
    rendered = {prompt_id: render_trace(tokenizer, row) for prompt_id, row in base_rows.items()}

    print(
        f"Scoring {len(target_ids)} targets, {len(sites)} sites, "
        f"{len(private_source_ids)} private source(s), {len(public_control_ids)} public control(s)",
        flush=True,
    )
    base_margins: dict[int, dict[str, float | int]] = {}
    rows: list[dict[str, Any]] = []
    for target_id in target_ids:
        margin = score_margin(
            model=model,
            tokenizer=tokenizer,
            layers_mod=layers_mod,
            rendered=rendered[target_id],
            private_label=args.private_label,
            public_label=args.public_label,
        )
        base_margins[target_id] = margin
        rows.append(
            {
                "condition": "base",
                "target_prompt_id": target_id,
                "target_prompt_key": rendered[target_id].prompt_key,
                "source_prompt_id": None,
                "source_prompt_key": None,
                "source_kind": None,
                "layer": None,
                "component": None,
                "region": None,
                "blend": 0.0,
                "payload_mode": None,
                "patch_positions": 0,
                **margin,
                "delta_margin_mean": 0.0,
            }
        )

    source_specs = [(prompt_id, "private_source") for prompt_id in private_source_ids] + [
        (prompt_id, "public_source_control") for prompt_id in public_control_ids
    ]
    for site in tqdm(sites, desc="trace sites", unit="site"):
        source_values_by_id = {
            prompt_id: capture_site(
                model=model,
                tokenizer=tokenizer,
                layers_mod=layers_mod,
                rendered=rendered[prompt_id],
                site=site,
            )
            for prompt_id, _kind in source_specs
        }
        for target_id in target_ids:
            for source_id, source_kind in source_specs:
                payload = make_payload(source_values=source_values_by_id[source_id], target=rendered[target_id], site=site)
                for blend in blends:
                    margin = score_margin(
                        model=model,
                        tokenizer=tokenizer,
                        layers_mod=layers_mod,
                        rendered=rendered[target_id],
                        private_label=args.private_label,
                        public_label=args.public_label,
                        site=site,
                        payload=payload,
                        blend=blend,
                    )
                    rows.append(
                        {
                            "condition": "patched",
                            "target_prompt_id": target_id,
                            "target_prompt_key": rendered[target_id].prompt_key,
                            "source_prompt_id": source_id,
                            "source_prompt_key": rendered[source_id].prompt_key,
                            "source_kind": source_kind,
                            "layer": site.layer,
                            "component": site.component,
                            "region": site.region,
                            "blend": blend,
                            "payload_mode": payload.mode,
                            "patch_positions": len(payload.positions),
                            **margin,
                            "delta_margin_mean": float(margin["margin_mean"]) - float(base_margins[target_id]["margin_mean"]),
                        }
                    )

    aggregate_rows = aggregate(rows)
    manifest = {
        "started_at": started,
        "finished_at": now_iso(),
        "model_path": str(args.model_path),
        "source_generations": str(args.source_generations),
        "localization_run": str(args.localization_run),
        "output_dir": str(out_dir),
        "target_prompt_ids": target_ids,
        "private_source_prompt_ids": private_source_ids,
        "public_control_prompt_ids": public_control_ids,
        "top_sites": args.top_sites,
        "site_keys": [site.key for site in sites],
        "blends": blends,
        "private_label": args.private_label,
        "public_label": args.public_label,
        "method": "teacher_forced_holding_logit_causal_trace",
        "not_generation_evidence": True,
        "not_promotion_evidence": True,
        "min_complete_answer_tokens": MIN_COMPLETE_ANSWER_TOKENS,
        **source_budget,
    }
    write_json(out_dir / "manifest.json", manifest)
    write_jsonl(out_dir / "patch_rows.jsonl", rows)
    write_jsonl(out_dir / "aggregate.jsonl", aggregate_rows)
    write_report(out_dir / "report.md", manifest=manifest, aggregate_rows=aggregate_rows, rows=rows)

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(f"Wrote {out_dir}", flush=True)
    if aggregate_rows:
        top = aggregate_rows[0]
        print(
            "Top site: "
            f"L{top['layer']} {top['component']} {top['region']} blend={top['blend']} "
            f"private-control-delta={top['private_minus_public_control_delta']:.4f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
