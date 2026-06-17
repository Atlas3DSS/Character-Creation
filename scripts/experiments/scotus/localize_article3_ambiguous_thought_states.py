#!/usr/bin/env python3
"""Localize Article III private/public visible-thought state differences.

This is not an actuator test. It uses the evaluator-positive ambiguous Article
III counterfactual-thought setup to ask where the model state differs when the
same prompt is teacher-forced through neutral, private-rights, and
public-rights visible reasoning. The output is a compact list of candidate
layer/component/region directions for a later no-mask intervention test.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MODEL = Path("/home/orwel/dev_genius/models/Qwen3.5-27B")
DEFAULT_PROMPT_BANK = PROJECT_ROOT / "data" / "scotus" / "scotus_article3_ambiguous_poke_prompts_v1.jsonl"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "sweep_v4"
CONDITIONS = ("neutral", "private_rights", "public_rights")
DEFAULT_COMPONENTS = "residual,mixer,mlp"
DEFAULT_REGIONS = "pre_answer_last,thought_mean,thought_tail16_mean,tail32_mean"

PRIVATE_THOUGHT = """The Article III question should be framed as a private-rights adjudication problem. The dispute concerns liability or entitlement between private parties, or a traditional common-law-style claim, and Congress cannot convert that matter into a final non-Article-III judgment merely by assigning it to an Article I or agency tribunal. The important distinction is that public-rights adjudication permits more administrative factfinding, while private-rights disputes require an Article III court for final judgment unless a narrow adjunct, consent, or appellate-review exception applies."""

PUBLIC_THOUGHT = """The Article III question should be framed as a public-rights adjudication problem. Congress may create federal statutory benefits, regulatory schemes, patents, tariffs, or enforcement systems and assign initial factfinding or adjudication to an agency or Article I tribunal, so long as Article III judicial review remains available. The important distinction is that public rights arise from a federal regulatory program or sovereign scheme, unlike private-rights disputes that resemble traditional common-law liability between private parties."""

NEUTRAL_THOUGHT = """The Article III issue turns on the public-rights/private-rights distinction, the role of the non-Article-III adjudicator, the availability of Article III judicial review, and whether the matter resembles a traditional judicial dispute or a congressionally created federal scheme. The answer should apply those considerations to the facts without relying on labels alone."""

THOUGHTS = {
    "neutral": NEUTRAL_THOUGHT,
    "private_rights": PRIVATE_THOUGHT,
    "public_rights": PUBLIC_THOUGHT,
}


@dataclass(frozen=True)
class PromptSpec:
    prompt_id: int
    prompt_key: str
    issue_area: str
    prompt: str
    expected_frames: tuple[str, ...]
    contrast_frames: tuple[str, ...]
    domain_frames: tuple[str, ...]


@dataclass(frozen=True)
class RegionSpans:
    thought_start: int
    thought_end: int
    seq_len: int


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def write_json(path: Path, data: dict[str, Any] | list[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=False) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def read_prompt_specs(path: Path) -> list[PromptSpec]:
    specs: list[PromptSpec] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            specs.append(
                PromptSpec(
                    prompt_id=int(row["prompt_id"]),
                    prompt_key=str(row["prompt_key"]),
                    issue_area=str(row.get("issue_area", "")),
                    prompt=str(row["prompt"]),
                    expected_frames=tuple(row.get("expected_frames", [])),
                    contrast_frames=tuple(row.get("contrast_frames", [])),
                    domain_frames=tuple(row.get("domain_frames", [])),
                )
            )
    if not specs:
        raise ValueError(f"No prompt specs loaded from {path}")
    return specs


def parse_csv(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def parse_layers(raw: str, n_layers: int) -> list[int]:
    if raw.strip().lower() == "all":
        return list(range(n_layers))
    layers = [int(item.strip()) for item in raw.split(",") if item.strip()]
    bad = [layer for layer in layers if layer < 0 or layer >= n_layers]
    if bad:
        raise ValueError(f"Layer ids out of range 0..{n_layers - 1}: {bad}")
    return layers


def markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    def cell(value: Any) -> str:
        return str(value).replace("|", "\\|").replace("\n", " ")

    lines = [
        "| " + " | ".join(cell(header) for header in headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(cell(value) for value in row) + " |")
    return "\n".join(lines)


def fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def safe_key(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]+", "_", text).strip("_")


def nested_attr(obj: Any, path: str) -> Any | None:
    cur = obj
    for part in path.split("."):
        if not hasattr(cur, part):
            return None
        cur = getattr(cur, part)
    return cur


def transformer_layers(model: torch.nn.Module) -> Any:
    for path in (
        "model.language_model.layers",
        "language_model.layers",
        "model.layers",
        "transformer.h",
        "gpt_neox.layers",
    ):
        layers = nested_attr(model, path)
        if layers is not None:
            return layers
    raise RuntimeError(f"Could not locate transformer layers on {type(model).__name__}")


def first_parameter_device(model: torch.nn.Module) -> torch.device:
    return next(model.parameters()).device


def output_tensor(output: Any) -> torch.Tensor:
    return output[0] if isinstance(output, tuple) else output


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
    raise RuntimeError(f"Layer {type(layer_mod).__name__} has no recognized mixer component")


def render_prefilled_prompt(tokenizer: Any, *, prompt: str, thought: str) -> tuple[str, RegionSpans]:
    try:
        chat_prefix = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )
    except TypeError:
        chat_prefix = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        if "<think>" not in chat_prefix:
            chat_prefix += "<think>\n"

    stripped = thought.strip()
    before_close = f"{chat_prefix}{stripped}"
    rendered = f"{before_close}\n</think>\n\n"
    prefix_ids = tokenizer(chat_prefix, add_special_tokens=False).input_ids
    before_close_ids = tokenizer(before_close, add_special_tokens=False).input_ids
    full_ids = tokenizer(rendered, add_special_tokens=False).input_ids
    thought_start = min(len(prefix_ids), len(full_ids) - 1)
    thought_end = max(thought_start + 1, min(len(before_close_ids), len(full_ids)))
    return rendered, RegionSpans(thought_start=thought_start, thought_end=thought_end, seq_len=len(full_ids))


def vector_for_region(hidden_2d: torch.Tensor, spans: RegionSpans, region: str) -> torch.Tensor:
    seq_len = hidden_2d.shape[0]
    thought_start = max(0, min(spans.thought_start, seq_len - 1))
    thought_end = max(thought_start + 1, min(spans.thought_end, seq_len))
    if region == "pre_answer_last":
        return hidden_2d[-1, :]
    if region == "thought_mean":
        return hidden_2d[thought_start:thought_end, :].mean(dim=0)
    if region == "thought_tail16_mean":
        return hidden_2d[max(thought_start, thought_end - 16) : thought_end, :].mean(dim=0)
    if region == "tail32_mean":
        return hidden_2d[max(0, seq_len - 32) : seq_len, :].mean(dim=0)
    raise ValueError(f"Unknown region: {region}")


def install_capture_hooks(
    *,
    layers_mod: Any,
    layers: list[int],
    components: list[str],
    regions: list[str],
    spans: RegionSpans,
    captured: dict[tuple[int, str, str], np.ndarray],
) -> list[Any]:
    handles: list[Any] = []

    for layer in layers:
        for component in components:
            module = select_component_module(layers_mod[layer], component)

            def make_hook(layer_idx: int, component_name: str) -> Any:
                def hook(_module: torch.nn.Module, _inp: tuple[Any, ...], out: Any) -> Any:
                    hidden = output_tensor(out)[0]
                    for region in regions:
                        vec = vector_for_region(hidden, spans, region)
                        captured[(layer_idx, component_name, region)] = (
                            vec.detach().float().cpu().numpy().astype(np.float32, copy=False)
                        )
                    return out

                return hook

            handles.append(module.register_forward_hook(make_hook(layer, component)))
    return handles


@torch.inference_mode()
def capture_condition_state(
    *,
    model: torch.nn.Module,
    tokenizer: Any,
    layers_mod: Any,
    prompt: str,
    thought: str,
    layers: list[int],
    components: list[str],
    regions: list[str],
    max_length: int,
) -> tuple[dict[tuple[int, str, str], np.ndarray], dict[str, int]]:
    rendered, spans = render_prefilled_prompt(tokenizer, prompt=prompt, thought=thought)
    encoded = tokenizer(
        rendered,
        return_tensors="pt",
        add_special_tokens=False,
        truncation=True,
        max_length=max_length,
    )
    seq_len = int(encoded["input_ids"].shape[1])
    if seq_len < spans.seq_len:
        raise ValueError(
            f"Rendered prompt was truncated from {spans.seq_len} to {seq_len}; increase --max-length"
        )
    input_device = first_parameter_device(model)
    inputs = {key: value.to(input_device) for key, value in encoded.items()}
    captured: dict[tuple[int, str, str], np.ndarray] = {}
    handles = install_capture_hooks(
        layers_mod=layers_mod,
        layers=layers,
        components=components,
        regions=regions,
        spans=spans,
        captured=captured,
    )
    try:
        _ = model(**inputs, use_cache=False)
    finally:
        for handle in handles:
            handle.remove()
    expected = {(layer, component, region) for layer in layers for component in components for region in regions}
    missing = sorted(expected - set(captured))
    if missing:
        raise RuntimeError(f"Missing captured states: {missing[:10]}{'...' if len(missing) > 10 else ''}")
    return captured, {
        "seq_len": seq_len,
        "thought_start": spans.thought_start,
        "thought_end": spans.thought_end,
        "thought_tokens": spans.thought_end - spans.thought_start,
    }


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 1e-12:
        return 0.0
    return float(np.dot(a, b) / denom)


def site_metrics_from_deltas(
    deltas: list[np.ndarray],
    private_neutral: list[np.ndarray],
    public_neutral: list[np.ndarray],
) -> dict[str, float | np.ndarray]:
    mean_dir = np.mean(np.stack(deltas, axis=0), axis=0).astype(np.float32)
    mean_norm = float(np.linalg.norm(mean_dir))
    delta_norms = [float(np.linalg.norm(delta)) for delta in deltas]
    if mean_norm <= 1e-12:
        consistency = 0.0
        pn_alignment = 0.0
        pub_alignment = 0.0
    else:
        consistency = float(np.mean([cosine(delta, mean_dir) for delta in deltas]))
        pn_alignment = float(np.mean([cosine(delta, mean_dir) for delta in private_neutral]))
        pub_alignment = float(np.mean([cosine(delta, mean_dir) for delta in public_neutral]))
    triad_separation = pn_alignment - pub_alignment
    rank_score = max(0.0, consistency) * max(0.0, triad_separation) * math.log1p(
        float(np.mean(delta_norms))
    )
    return {
        "mean_direction": mean_dir,
        "mean_direction_norm": mean_norm,
        "mean_pair_delta_norm": float(np.mean(delta_norms)),
        "sd_pair_delta_norm": float(np.std(delta_norms, ddof=1)) if len(delta_norms) > 1 else 0.0,
        "delta_consistency_cos_to_mean": consistency,
        "private_neutral_alignment": pn_alignment,
        "public_neutral_alignment": pub_alignment,
        "triad_separation": triad_separation,
        "rank_score": rank_score,
    }


def compute_metrics(
    records: dict[tuple[int, str], dict[tuple[int, str, str], np.ndarray]],
    prompt_ids: list[int],
    *,
    shuffle_controls: int,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[tuple[int, str, str], np.ndarray]]:
    site_keys = sorted(next(iter(records.values())).keys())
    rows: list[dict[str, Any]] = []
    directions: dict[tuple[int, str, str], np.ndarray] = {}
    rng = np.random.default_rng(seed)
    labels = list(CONDITIONS)

    for site in site_keys:
        layer, component, region = site
        deltas = [
            records[(prompt_id, "private_rights")][site] - records[(prompt_id, "public_rights")][site]
            for prompt_id in prompt_ids
        ]
        private_neutral = [
            records[(prompt_id, "private_rights")][site] - records[(prompt_id, "neutral")][site]
            for prompt_id in prompt_ids
        ]
        public_neutral = [
            records[(prompt_id, "public_rights")][site] - records[(prompt_id, "neutral")][site]
            for prompt_id in prompt_ids
        ]
        metrics = site_metrics_from_deltas(deltas, private_neutral, public_neutral)

        null_scores: list[float] = []
        null_consistency: list[float] = []
        for _ in range(shuffle_controls):
            shuffled_deltas: list[np.ndarray] = []
            shuffled_pn: list[np.ndarray] = []
            shuffled_pubn: list[np.ndarray] = []
            for prompt_id in prompt_ids:
                perm = list(rng.permutation(labels))
                pseudo_private, pseudo_public, pseudo_neutral = perm
                shuffled_deltas.append(records[(prompt_id, pseudo_private)][site] - records[(prompt_id, pseudo_public)][site])
                shuffled_pn.append(records[(prompt_id, pseudo_private)][site] - records[(prompt_id, pseudo_neutral)][site])
                shuffled_pubn.append(records[(prompt_id, pseudo_public)][site] - records[(prompt_id, pseudo_neutral)][site])
            null = site_metrics_from_deltas(shuffled_deltas, shuffled_pn, shuffled_pubn)
            null_scores.append(float(null["rank_score"]))
            null_consistency.append(float(null["delta_consistency_cos_to_mean"]))

        null_max = max(null_scores) if null_scores else 0.0
        row = {
            "layer": layer,
            "component": component,
            "region": region,
            "n_prompts": len(prompt_ids),
            "mean_direction_norm": float(metrics["mean_direction_norm"]),
            "mean_pair_delta_norm": float(metrics["mean_pair_delta_norm"]),
            "sd_pair_delta_norm": float(metrics["sd_pair_delta_norm"]),
            "delta_consistency_cos_to_mean": float(metrics["delta_consistency_cos_to_mean"]),
            "private_neutral_alignment": float(metrics["private_neutral_alignment"]),
            "public_neutral_alignment": float(metrics["public_neutral_alignment"]),
            "triad_separation": float(metrics["triad_separation"]),
            "rank_score": float(metrics["rank_score"]),
            "shuffle_controls": int(shuffle_controls),
            "shuffle_rank_score_mean": float(np.mean(null_scores)) if null_scores else 0.0,
            "shuffle_rank_score_max": float(null_max),
            "shuffle_consistency_mean": float(np.mean(null_consistency)) if null_consistency else 0.0,
            "rank_score_minus_shuffle_max": float(metrics["rank_score"]) - float(null_max),
        }
        rows.append(row)
        directions[site] = metrics["mean_direction"]

    rows.sort(
        key=lambda row: (
            float(row["rank_score_minus_shuffle_max"]),
            float(row["rank_score"]),
            float(row["triad_separation"]),
        ),
        reverse=True,
    )
    return rows, directions


def load_model_and_tokenizer(model_path: Path, device_map: str) -> tuple[Any, torch.nn.Module]:
    if not model_path.exists():
        raise RuntimeError(f"Local model path does not exist: {model_path}")
    has_weights = any(model_path.rglob("*.safetensors")) or any(model_path.rglob("*.bin"))
    print(f"Local model path: {model_path} weights_present={has_weights}", flush=True)
    if not has_weights:
        raise RuntimeError(f"No local model weights found under {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if device_map.lower() in {"single", "cuda", "cuda:0", "gpu"}:
        resolved_device_map: str | dict[str, int | str] = {"": 0}
    elif device_map.lower() == "cpu":
        resolved_device_map = {"": "cpu"}
    else:
        resolved_device_map = device_map
    try:
        import transformers.modeling_utils as modeling_utils

        modeling_utils.caching_allocator_warmup = lambda *args, **kwargs: None
    except (ImportError, AttributeError):
        pass
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        trust_remote_code=True,
        local_files_only=True,
        torch_dtype="auto",
        device_map=resolved_device_map,
        low_cpu_mem_usage=True,
        attn_implementation="sdpa",
    )
    model.eval()
    return tokenizer, model


def save_top_directions(
    out_dir: Path,
    rows: list[dict[str, Any]],
    directions: dict[tuple[int, str, str], np.ndarray],
    *,
    top_k: int,
) -> list[dict[str, Any]]:
    arrays: dict[str, np.ndarray] = {}
    meta_rows: list[dict[str, Any]] = []
    for idx, row in enumerate(rows[:top_k]):
        site = (int(row["layer"]), str(row["component"]), str(row["region"]))
        raw = directions[site].astype(np.float32, copy=False)
        norm = float(np.linalg.norm(raw))
        unit = raw / max(norm, 1e-12)
        key = f"direction_{idx:03d}_{safe_key(str(row['component']))}_L{int(row['layer']):02d}_{safe_key(str(row['region']))}"
        arrays[key] = unit.astype(np.float32, copy=False)
        arrays[f"{key}_raw_mean_delta"] = raw
        meta = {
            **row,
            "direction_key": key,
            "direction_norm": norm,
            "direction_semantics": "teacher_forced_private_rights_minus_public_rights_visible_thought_state",
        }
        meta_rows.append(meta)
    np.savez_compressed(out_dir / "top_directions.npz", **arrays)
    write_jsonl(out_dir / "direction_meta.jsonl", meta_rows)
    return meta_rows


def write_report(
    path: Path,
    *,
    manifest: dict[str, Any],
    metrics: list[dict[str, Any]],
    direction_meta: list[dict[str, Any]],
    token_rows: list[dict[str, Any]],
) -> None:
    by_component: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in metrics:
        by_component[str(row["component"])].append(row)

    lines = [
        "# Article III Ambiguous Thought-State Localization",
        "",
        "## Purpose",
        "",
        (
            "Localize where Qwen's state differs when ambiguous Article III prompts are "
            "teacher-forced through private-rights versus public-rights visible reasoning. "
            "This is a candidate-nomination screen, not an actuator result."
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
                ["Prompt bank", manifest["prompt_bank"]],
                ["Prompts", manifest["n_prompts"]],
                ["Layers", manifest["layers"]],
                ["Components", ", ".join(manifest["components"])],
                ["Regions", ", ".join(manifest["regions"])],
                ["Shuffle controls", manifest["shuffle_controls"]],
                ["Output dir", manifest["output_dir"]],
            ],
        ),
        "",
        "## Top Sites",
        "",
        markdown_table(
            [
                "Rank",
                "Layer",
                "Component",
                "Region",
                "Score-null",
                "Score",
                "Null max",
                "Consistency",
                "Triad sep",
                "Mean delta norm",
            ],
            [
                [
                    idx + 1,
                    row["layer"],
                    row["component"],
                    row["region"],
                    fmt(row["rank_score_minus_shuffle_max"]),
                    fmt(row["rank_score"]),
                    fmt(row["shuffle_rank_score_max"]),
                    fmt(row["delta_consistency_cos_to_mean"]),
                    fmt(row["triad_separation"]),
                    fmt(row["mean_pair_delta_norm"]),
                ]
                for idx, row in enumerate(metrics[:20])
            ],
        ),
        "",
    ]
    for component in sorted(by_component):
        lines.extend(
            [
                f"## Top {component} Sites",
                "",
                markdown_table(
                    ["Rank", "Layer", "Region", "Score-null", "Consistency", "Triad sep"],
                    [
                        [
                            idx + 1,
                            row["layer"],
                            row["region"],
                            fmt(row["rank_score_minus_shuffle_max"]),
                            fmt(row["delta_consistency_cos_to_mean"]),
                            fmt(row["triad_separation"]),
                        ]
                        for idx, row in enumerate(by_component[component][:8])
                    ],
                ),
                "",
            ]
        )
    lines.extend(
        [
            "## Tokenization",
            "",
            markdown_table(
                ["Prompt", "Condition", "Seq len", "Thought tokens", "Thought span"],
                [
                    [
                        row["prompt_key"],
                        row["condition"],
                        row["seq_len"],
                        row["thought_tokens"],
                        f"{row['thought_start']}:{row['thought_end']}",
                    ]
                    for row in token_rows[:18]
                ],
            ),
            "",
            "## Gate Interpretation",
            "",
            "- This localizes a text-prefill-conditioned trajectory difference; it does not show non-mask causal control.",
            "- The saved directions are candidate surfaces for a later no-mask multi-site controller or causal patch screen.",
            "- Any promotion run must make the model generate the target reasoning trajectory itself, use the ambiguous prompt bank, and beat random/source/text controls on visible reasoning and final holding labels.",
            "",
            "## Artifacts",
            "",
            f"- Manifest: `{manifest['output_dir']}/manifest.json`.",
            f"- Site metrics: `{manifest['output_dir']}/site_metrics.jsonl`.",
            f"- Top directions: `{manifest['output_dir']}/top_directions.npz`.",
            f"- Direction metadata: `{manifest['output_dir']}/direction_meta.jsonl`.",
        ]
    )
    if direction_meta:
        lines.extend(
            [
                "",
                "Top saved direction:",
                "",
                "```json",
                json.dumps(direction_meta[0], indent=2, sort_keys=True),
                "```",
            ]
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--prompt-bank", type=Path, default=DEFAULT_PROMPT_BANK)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--layers", default="all")
    parser.add_argument("--components", default=DEFAULT_COMPONENTS)
    parser.add_argument("--regions", default=DEFAULT_REGIONS)
    parser.add_argument("--max-prompts", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--shuffle-controls", type=int, default=64)
    parser.add_argument("--top-k", type=int, default=48)
    parser.add_argument("--device-map", default="single")
    parser.add_argument("--seed", type=int, default=20260502)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = now_iso()
    out_dir = args.output_root / f"scotus_article3_ambiguous_thought_state_localization_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)

    prompt_specs = read_prompt_specs(args.prompt_bank)[: args.max_prompts]
    components = parse_csv(args.components)
    regions = parse_csv(args.regions)
    unknown_components = sorted(set(components) - {"residual", "mixer", "mlp"})
    unknown_regions = sorted(set(regions) - {"pre_answer_last", "thought_mean", "thought_tail16_mean", "tail32_mean"})
    if unknown_components:
        raise ValueError(f"Unknown components: {unknown_components}")
    if unknown_regions:
        raise ValueError(f"Unknown regions: {unknown_regions}")

    tokenizer, model = load_model_and_tokenizer(args.model_path, args.device_map)
    layers_mod = transformer_layers(model)
    layers = parse_layers(args.layers, len(layers_mod))
    print(
        f"Capturing {len(prompt_specs)} prompts x {len(CONDITIONS)} conditions; "
        f"layers={len(layers)} components={components} regions={regions}",
        flush=True,
    )

    records: dict[tuple[int, str], dict[tuple[int, str, str], np.ndarray]] = {}
    token_rows: list[dict[str, Any]] = []
    for spec in tqdm(prompt_specs, desc="prompts", unit="prompt"):
        for condition in CONDITIONS:
            captured, token_meta = capture_condition_state(
                model=model,
                tokenizer=tokenizer,
                layers_mod=layers_mod,
                prompt=spec.prompt,
                thought=THOUGHTS[condition],
                layers=layers,
                components=components,
                regions=regions,
                max_length=args.max_length,
            )
            records[(spec.prompt_id, condition)] = captured
            token_rows.append(
                {
                    "prompt_id": spec.prompt_id,
                    "prompt_key": spec.prompt_key,
                    "condition": condition,
                    **token_meta,
                }
            )

    prompt_ids = [spec.prompt_id for spec in prompt_specs]
    metrics, directions = compute_metrics(
        records,
        prompt_ids,
        shuffle_controls=args.shuffle_controls,
        seed=args.seed,
    )
    direction_meta = save_top_directions(out_dir, metrics, directions, top_k=args.top_k)

    manifest = {
        "started_at": started,
        "finished_at": now_iso(),
        "model_path": str(args.model_path),
        "prompt_bank": str(args.prompt_bank),
        "output_dir": str(out_dir),
        "n_prompts": len(prompt_specs),
        "prompt_keys": [spec.prompt_key for spec in prompt_specs],
        "conditions": list(CONDITIONS),
        "layers": "all" if len(layers) == len(layers_mod) else ",".join(str(layer) for layer in layers),
        "n_layers": len(layers),
        "components": components,
        "regions": regions,
        "max_length": args.max_length,
        "shuffle_controls": args.shuffle_controls,
        "top_k": args.top_k,
        "seed": args.seed,
        "method": "teacher_forced_private_rights_minus_public_rights_visible_thought_state_localization",
        "not_actuator_evidence": True,
    }
    write_json(out_dir / "manifest.json", manifest)
    write_jsonl(out_dir / "site_metrics.jsonl", metrics)
    write_jsonl(out_dir / "token_spans.jsonl", token_rows)
    write_report(out_dir / "report.md", manifest=manifest, metrics=metrics, direction_meta=direction_meta, token_rows=token_rows)

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"Wrote {out_dir}", flush=True)
    if metrics:
        top = metrics[0]
        print(
            "Top site: "
            f"L{top['layer']} {top['component']} {top['region']} "
            f"score-null={top['rank_score_minus_shuffle_max']:.4f} "
            f"consistency={top['delta_consistency_cos_to_mean']:.4f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
