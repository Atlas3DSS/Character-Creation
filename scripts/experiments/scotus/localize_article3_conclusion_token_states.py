#!/usr/bin/env python3
"""Localize Article III private/public states at actual conclusion evidence tokens.

This is a candidate-localization screen, not an actuator result. It differs
from the broader answer-state localizer by anchoring capture windows to the
private/public conclusion evidence that the model actually generated in its
final answer. The goal is to nominate token/component/layer windows for a
later causal-tracing or controller test, not to widen another direct-add sweep.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
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
    cosine,
    fmt,
    load_model_and_tokenizer,
    markdown_table,
    now_iso,
    output_tensor,
    parse_csv,
    parse_layers,
    safe_key,
    select_component_module,
    transformer_layers,
    write_json,
    write_jsonl,
)
from localize_article3_generated_thought_baselines import source_budget_meta  # noqa: E402
from poke_scotus_sae_layers import first_parameter_device  # noqa: E402
from run_scotus_thinking_smoke import format_chat  # noqa: E402
from score_article3_conclusion_polarity import (  # noqa: E402
    CONTRASTIVE_PUBLIC_RE,
    PRIVATE_PATTERNS,
    PUBLIC_PATTERNS,
    is_negated_private_match,
    score_text,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MODEL = Path("/home/orwel/dev_genius/models/Qwen3.5-27B")
DEFAULT_SOURCE_GENERATIONS = (
    PROJECT_ROOT / "sweep_v4" / "scotus_counterfactual_thoughts_20260502_052323" / "generations.jsonl"
)
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "sweep_v4"
DEFAULT_COMPONENTS = "residual,mixer,mlp"
DEFAULT_REGIONS = "pre_evidence_last,evidence_mean,evidence_tail16_mean,pre_evidence_context64_mean"


@dataclass(frozen=True)
class EvidenceSpan:
    prompt_id: int
    prompt_key: str
    condition: str
    target: str
    prompt: str
    thinking: str
    answer: str
    label: str
    private_score: int
    public_score: int
    evidence_pattern: str
    evidence_text: str
    evidence_start_char: int
    evidence_end_char: int
    span_start_char: int
    span_end_char: int


@dataclass(frozen=True)
class TokenSpan:
    token_start: int
    token_end: int
    seq_len: int
    answer_start_token: int
    answer_end_token: int
    absolute_span_start_char: int
    absolute_span_end_char: int
    offset_mapping_used: bool


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


def sentence_bounds(text: str, start: int, end: int) -> tuple[int, int]:
    left_period = text.rfind(".", 0, start)
    left_newline = text.rfind("\n", 0, start)
    left = max(left_period, left_newline) + 1
    right_candidates = [idx for idx in (text.find(".", end), text.find("\n", end)) if idx != -1]
    right = min(right_candidates) + 1 if right_candidates else len(text)
    return max(0, left), min(len(text), right)


def public_evidence_matches(text: str) -> list[tuple[str, re.Match[str]]]:
    matches: list[tuple[str, re.Match[str]]] = []
    for name, pattern in PUBLIC_PATTERNS:
        for match in pattern.finditer(text):
            sentence_start = text.rfind(".", 0, match.start()) + 1
            sentence_end_raw = text.find(".", match.end())
            sentence_end = len(text) if sentence_end_raw == -1 else sentence_end_raw + 1
            sentence = text[sentence_start:sentence_end]
            if name in {"congress_may_assign", "public_rights_permit"} and CONTRASTIVE_PUBLIC_RE.search(sentence):
                continue
            matches.append((name, match))
    matches.sort(key=lambda item: (item[1].start(), item[1].end(), item[0]))
    return matches


def private_evidence_matches(text: str) -> list[tuple[str, re.Match[str]]]:
    matches: list[tuple[str, re.Match[str]]] = []
    for name, pattern in PRIVATE_PATTERNS:
        for match in pattern.finditer(text):
            if is_negated_private_match(name, text, match):
                continue
            matches.append((name, match))
    matches.sort(key=lambda item: (item[1].start(), item[1].end(), item[0]))
    return matches


def select_evidence(
    row: dict[str, Any],
    *,
    target: str,
    span_mode: str,
    select: str,
) -> EvidenceSpan | None:
    answer = str(row.get("answer") or "").strip()
    thinking = str(row.get("thinking") or "").strip()
    prompt = str(row.get("prompt") or "").strip()
    if not answer or not thinking or not prompt:
        return None
    scored = score_text(answer)
    matches = private_evidence_matches(answer) if target == "private" else public_evidence_matches(answer)
    if not matches:
        return None
    if select == "first":
        pattern_name, match = matches[0]
    elif select == "strongest":
        pattern_name, match = max(matches, key=lambda item: (item[1].end() - item[1].start(), item[1].start()))
    elif select == "last":
        pattern_name, match = matches[-1]
    else:
        raise ValueError(f"Unknown evidence selection mode: {select}")
    evidence_start, evidence_end = match.span()
    if span_mode == "match":
        span_start, span_end = evidence_start, evidence_end
    elif span_mode == "sentence":
        span_start, span_end = sentence_bounds(answer, evidence_start, evidence_end)
    else:
        raise ValueError(f"Unknown span mode: {span_mode}")
    return EvidenceSpan(
        prompt_id=int(row["prompt_id"]),
        prompt_key=str(row["prompt_key"]),
        condition=str(row["condition"]),
        target=target,
        prompt=prompt,
        thinking=thinking,
        answer=answer,
        label=str(scored["label"]),
        private_score=int(scored["private_score"]),
        public_score=int(scored["public_score"]),
        evidence_pattern=pattern_name,
        evidence_text=match.group(0),
        evidence_start_char=evidence_start,
        evidence_end_char=evidence_end,
        span_start_char=span_start,
        span_end_char=span_end,
    )


def render_answer(tokenizer: Any, span: EvidenceSpan) -> tuple[str, str, TokenSpan, dict[str, Any]]:
    chat = format_chat(tokenizer, span.prompt, enable_thinking=True)
    before_answer = f"{chat}{span.thinking.strip()}\n</think>\n\n"
    answer = span.answer.strip()
    rendered = f"{before_answer}{answer}"
    absolute_start = len(before_answer) + span.span_start_char
    absolute_end = len(before_answer) + span.span_end_char
    offset_mapping_used = False
    try:
        encoded = tokenizer(
            rendered,
            add_special_tokens=False,
            truncation=True,
            max_length=4096,
            return_offsets_mapping=True,
        )
        offsets = encoded.pop("offset_mapping")
        token_positions = [
            idx
            for idx, (tok_start, tok_end) in enumerate(offsets)
            if int(tok_end) > absolute_start and int(tok_start) < absolute_end
        ]
        offset_mapping_used = True
    except (NotImplementedError, TypeError, ValueError):
        encoded = tokenizer(
            rendered,
            add_special_tokens=False,
            truncation=True,
            max_length=4096,
        )
        before_ids = tokenizer(before_answer, add_special_tokens=False).input_ids
        before_span_ids = tokenizer(answer[: span.span_start_char], add_special_tokens=False).input_ids
        span_ids = tokenizer(answer[span.span_start_char : span.span_end_char], add_special_tokens=False).input_ids
        start = len(before_ids) + len(before_span_ids)
        token_positions = list(range(start, start + max(1, len(span_ids))))
    input_ids = list(encoded["input_ids"])
    attention_mask = list(encoded.get("attention_mask") or [1] * len(input_ids))
    if not input_ids:
        raise ValueError(f"Empty rendered input for prompt_id={span.prompt_id} condition={span.condition}")
    if not token_positions:
        raise ValueError(
            f"No evidence tokens for prompt_id={span.prompt_id} condition={span.condition} "
            f"target={span.target}"
        )
    token_start = max(0, min(token_positions))
    token_end = min(len(input_ids), max(token_positions) + 1)
    if token_end <= token_start:
        raise ValueError(f"Invalid token span {token_start}:{token_end} for {span.prompt_key}")
    answer_start = len(tokenizer(before_answer, add_special_tokens=False).input_ids)
    answer_end = len(input_ids)
    token_span = TokenSpan(
        token_start=token_start,
        token_end=token_end,
        seq_len=len(input_ids),
        answer_start_token=min(answer_start, len(input_ids) - 1),
        answer_end_token=answer_end,
        absolute_span_start_char=absolute_start,
        absolute_span_end_char=absolute_end,
        offset_mapping_used=offset_mapping_used,
    )
    return rendered, before_answer, token_span, {"input_ids": input_ids, "attention_mask": attention_mask}


def vector_for_region(hidden_2d: torch.Tensor, span: TokenSpan, region: str) -> torch.Tensor:
    seq_len = int(hidden_2d.shape[0])
    start = max(0, min(span.token_start, seq_len - 1))
    end = max(start + 1, min(span.token_end, seq_len))
    if region == "pre_evidence_last":
        return hidden_2d[max(0, start - 1), :]
    if region == "evidence_mean":
        return hidden_2d[start:end, :].mean(dim=0)
    if region == "evidence_tail16_mean":
        return hidden_2d[max(start, end - 16) : end, :].mean(dim=0)
    if region == "pre_evidence_context64_mean":
        return hidden_2d[max(0, start - 64) : end, :].mean(dim=0)
    if region == "answer_to_evidence_mean":
        answer_start = max(0, min(span.answer_start_token, seq_len - 1))
        return hidden_2d[answer_start:end, :].mean(dim=0)
    raise ValueError(f"Unknown region: {region}")


def install_capture_hooks(
    *,
    layers_mod: Any,
    layers: list[int],
    components: list[str],
    regions: list[str],
    span: TokenSpan,
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
                        vec = vector_for_region(hidden, span, region)
                        captured[(layer_idx, component_name, region)] = (
                            vec.detach().float().cpu().numpy().astype(np.float32, copy=False)
                        )
                    return out

                return hook

            handles.append(module.register_forward_hook(make_hook(layer, component)))
    return handles


@torch.inference_mode()
def capture_conclusion_state(
    *,
    model: torch.nn.Module,
    layers_mod: Any,
    encoded: dict[str, list[int]],
    token_span: TokenSpan,
    layers: list[int],
    components: list[str],
    regions: list[str],
    max_length: int,
) -> dict[tuple[int, str, str], np.ndarray]:
    if len(encoded["input_ids"]) > max_length:
        raise ValueError(f"Rendered row exceeds --max-length: {len(encoded['input_ids'])} > {max_length}")
    input_device = first_parameter_device(model)
    inputs = {
        "input_ids": torch.tensor([encoded["input_ids"]], dtype=torch.long, device=input_device),
        "attention_mask": torch.tensor([encoded["attention_mask"]], dtype=torch.long, device=input_device),
    }
    captured: dict[tuple[int, str, str], np.ndarray] = {}
    handles = install_capture_hooks(
        layers_mod=layers_mod,
        layers=layers,
        components=components,
        regions=regions,
        span=token_span,
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
    return captured


def metric_from_deltas(deltas: list[np.ndarray]) -> dict[str, float | np.ndarray]:
    mean_dir = np.mean(np.stack(deltas, axis=0), axis=0).astype(np.float32)
    mean_norm = float(np.linalg.norm(mean_dir))
    pair_norms = [float(np.linalg.norm(delta)) for delta in deltas]
    consistency = 0.0 if mean_norm <= 1e-12 else float(np.mean([cosine(delta, mean_dir) for delta in deltas]))
    rank_score = max(0.0, consistency) * math.log1p(float(np.mean(pair_norms)))
    return {
        "mean_direction": mean_dir,
        "mean_direction_norm": mean_norm,
        "mean_pair_delta_norm": float(np.mean(pair_norms)),
        "sd_pair_delta_norm": float(np.std(pair_norms, ddof=1)) if len(pair_norms) > 1 else 0.0,
        "delta_consistency_cos_to_mean": consistency,
        "rank_score": float(rank_score),
    }


def compute_metrics(
    records: dict[tuple[int, str], dict[tuple[int, str, str], np.ndarray]],
    prompt_ids: list[int],
    *,
    shuffle_controls: int,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[tuple[int, str, str], np.ndarray]]:
    site_keys = sorted(next(iter(records.values())).keys())
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    directions: dict[tuple[int, str, str], np.ndarray] = {}
    for site in site_keys:
        deltas = [records[(prompt_id, "private")][site] - records[(prompt_id, "public")][site] for prompt_id in prompt_ids]
        metrics = metric_from_deltas(deltas)
        null_scores: list[float] = []
        null_consistency: list[float] = []
        for _idx in range(shuffle_controls):
            signs = rng.choice(np.array([-1.0, 1.0], dtype=np.float32), size=len(deltas))
            null_deltas = [delta * sign for delta, sign in zip(deltas, signs)]
            null = metric_from_deltas(null_deltas)
            null_scores.append(float(null["rank_score"]))
            null_consistency.append(float(null["delta_consistency_cos_to_mean"]))
        null_max = max(null_scores) if null_scores else 0.0
        layer, component, region = site
        row = {
            "layer": int(layer),
            "component": str(component),
            "region": str(region),
            "n_prompt_pairs": len(prompt_ids),
            "mean_direction_norm": float(metrics["mean_direction_norm"]),
            "mean_pair_delta_norm": float(metrics["mean_pair_delta_norm"]),
            "sd_pair_delta_norm": float(metrics["sd_pair_delta_norm"]),
            "delta_consistency_cos_to_mean": float(metrics["delta_consistency_cos_to_mean"]),
            "rank_score": float(metrics["rank_score"]),
            "shuffle_controls": int(shuffle_controls),
            "shuffle_rank_score_mean": float(np.mean(null_scores)) if null_scores else 0.0,
            "shuffle_rank_score_max": float(null_max),
            "shuffle_consistency_mean": float(np.mean(null_consistency)) if null_consistency else 0.0,
            "rank_score_minus_shuffle_max": float(metrics["rank_score"]) - float(null_max),
        }
        rows.append(row)
        directions[site] = np.asarray(metrics["mean_direction"], dtype=np.float32)
    rows.sort(
        key=lambda row: (
            float(row["rank_score_minus_shuffle_max"]),
            float(row["rank_score"]),
            float(row["delta_consistency_cos_to_mean"]),
        ),
        reverse=True,
    )
    return rows, directions


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
        meta_rows.append(
            {
                **row,
                "direction_key": key,
                "direction_norm": norm,
                "direction_semantics": "actual_answer_conclusion_evidence_private_minus_public",
            }
        )
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
        "# Article III Conclusion-Token State Localization",
        "",
        "## Purpose",
        "",
        (
            "Localize private-minus-public differences at the actual generated answer evidence/conclusion "
            "tokens selected by the Article III conclusion-polarity patterns. This is candidate nomination, "
            "not actuator evidence."
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
                ["Prompt pairs", manifest["n_prompt_pairs"]],
                ["Layers", manifest["layers"]],
                ["Components", ", ".join(manifest["components"])],
                ["Regions", ", ".join(manifest["regions"])],
                ["Span mode", manifest["span_mode"]],
                ["Evidence select", manifest["evidence_select"]],
                ["Shuffle controls", manifest["shuffle_controls"]],
                ["Source budget note", manifest.get("source_budget_note")],
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
                "Norm",
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
                    fmt(row["mean_direction_norm"]),
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
                    ["Rank", "Layer", "Region", "Score-null", "Consistency"],
                    [
                        [
                            idx + 1,
                            row["layer"],
                            row["region"],
                            fmt(row["rank_score_minus_shuffle_max"]),
                            fmt(row["delta_consistency_cos_to_mean"]),
                        ]
                        for idx, row in enumerate(by_component[component][:8])
                    ],
                ),
                "",
            ]
        )
    lines.extend(
        [
            "## Evidence Spans",
            "",
            markdown_table(
                [
                    "Prompt",
                    "Target",
                    "Label",
                    "Scores",
                    "Pattern",
                    "Tokens",
                    "Snippet",
                ],
                [
                    [
                        row["prompt_key"],
                        row["target"],
                        row["label"],
                        f"{row['private_score']}/{row['public_score']}",
                        row["evidence_pattern"],
                        f"{row['token_start']}:{row['token_end']}",
                        row["span_snippet"],
                    ]
                    for row in token_rows[:40]
                ],
            ),
            "",
            "## Interpretation",
            "",
            "- These directions come from actual generated final-answer evidence tokens, not fixed holding labels.",
            "- Inserted visible thoughts are still the source condition, so this is not no-mask actuator evidence.",
            "- A promoted candidate would need a separate causal patch/controller run with complete budgets and random/source gates.",
            "",
            "## Artifacts",
            "",
            f"- Manifest: `{manifest['output_dir']}/manifest.json`.",
            f"- Token spans: `{manifest['output_dir']}/token_spans.jsonl`.",
            f"- Site metrics: `{manifest['output_dir']}/site_metrics.jsonl`.",
            f"- Top directions: `{manifest['output_dir']}/top_directions.npz`.",
            f"- Direction metadata: `{manifest['output_dir']}/direction_meta.jsonl`.",
        ]
    )
    if direction_meta:
        lines.extend(["", "Top saved direction:", "", "```json", json.dumps(direction_meta[0], indent=2), "```"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--source-generations", type=Path, default=DEFAULT_SOURCE_GENERATIONS)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--prompt-ids", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--layers", default="all")
    parser.add_argument("--components", default=DEFAULT_COMPONENTS)
    parser.add_argument("--regions", default=DEFAULT_REGIONS)
    parser.add_argument("--span-mode", choices=["match", "sentence"], default="sentence")
    parser.add_argument("--evidence-select", choices=["first", "last", "strongest"], default="last")
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--shuffle-controls", type=int, default=128)
    parser.add_argument("--top-k", type=int, default=64)
    parser.add_argument("--device-map", default="single")
    parser.add_argument("--seed", type=int, default=20260502)
    parser.add_argument("--allow-short-source-budget", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = now_iso()
    source_budget = source_budget_meta(args.source_generations, allow_short=args.allow_short_source_budget)
    requested_prompt_ids = parse_int_list(args.prompt_ids)
    rows = read_jsonl(args.source_generations)
    by_key = {(int(row["prompt_id"]), str(row["condition"])): row for row in rows}
    selected_spans: dict[tuple[int, str], EvidenceSpan] = {}
    missing: list[tuple[int, str]] = []
    for prompt_id in requested_prompt_ids:
        private_row = by_key.get((prompt_id, "private_rights"))
        public_row = by_key.get((prompt_id, "public_rights"))
        if private_row is None or public_row is None:
            missing.append((prompt_id, "private_rights/public_rights"))
            continue
        private_span = select_evidence(
            private_row,
            target="private",
            span_mode=args.span_mode,
            select=args.evidence_select,
        )
        public_span = select_evidence(
            public_row,
            target="public",
            span_mode=args.span_mode,
            select=args.evidence_select,
        )
        if private_span is None or public_span is None:
            missing.append((prompt_id, "missing_private_or_public_evidence"))
            continue
        selected_spans[(prompt_id, "private")] = private_span
        selected_spans[(prompt_id, "public")] = public_span
    prompt_ids = [
        prompt_id
        for prompt_id in requested_prompt_ids
        if (prompt_id, "private") in selected_spans and (prompt_id, "public") in selected_spans
    ]
    if len(prompt_ids) < 2:
        raise ValueError(
            f"Need at least two prompt pairs with both private and public conclusion evidence; got {len(prompt_ids)}"
        )

    components = parse_csv(args.components)
    regions = parse_csv(args.regions)
    allowed_regions = {
        "pre_evidence_last",
        "evidence_mean",
        "evidence_tail16_mean",
        "pre_evidence_context64_mean",
        "answer_to_evidence_mean",
    }
    unknown_regions = sorted(set(regions) - allowed_regions)
    if unknown_regions:
        raise ValueError(f"Unknown regions: {unknown_regions}")

    out_dir = args.output_root / f"scotus_article3_conclusion_token_localization_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)
    tokenizer, model = load_model_and_tokenizer(args.model_path, args.device_map)
    layers_mod = transformer_layers(model)
    layers = parse_layers(args.layers, len(layers_mod))
    print(
        f"Capturing conclusion evidence states for {len(prompt_ids)} prompt pair(s); "
        f"layers={len(layers)} components={components} regions={regions}",
        flush=True,
    )
    if missing:
        print(f"Skipped evidence pairs: {missing}", flush=True)

    records: dict[tuple[int, str], dict[tuple[int, str, str], np.ndarray]] = {}
    token_rows: list[dict[str, Any]] = []
    for prompt_id in tqdm(prompt_ids, desc="prompt pairs", unit="pair"):
        for target in ("private", "public"):
            span = selected_spans[(prompt_id, target)]
            _rendered, _before_answer, token_span, encoded = render_answer(tokenizer, span)
            captured = capture_conclusion_state(
                model=model,
                layers_mod=layers_mod,
                encoded=encoded,
                token_span=token_span,
                layers=layers,
                components=components,
                regions=regions,
                max_length=args.max_length,
            )
            records[(prompt_id, target)] = captured
            snippet = re.sub(r"\s+", " ", span.answer[span.span_start_char : span.span_end_char]).strip()
            token_rows.append(
                {
                    "prompt_id": prompt_id,
                    "prompt_key": span.prompt_key,
                    "condition": span.condition,
                    "target": span.target,
                    "label": span.label,
                    "private_score": span.private_score,
                    "public_score": span.public_score,
                    "evidence_pattern": span.evidence_pattern,
                    "evidence_text": span.evidence_text,
                    "span_snippet": snippet[:240],
                    "evidence_start_char": span.evidence_start_char,
                    "evidence_end_char": span.evidence_end_char,
                    "span_start_char": span.span_start_char,
                    "span_end_char": span.span_end_char,
                    "token_start": token_span.token_start,
                    "token_end": token_span.token_end,
                    "seq_len": token_span.seq_len,
                    "answer_start_token": token_span.answer_start_token,
                    "answer_end_token": token_span.answer_end_token,
                    "offset_mapping_used": token_span.offset_mapping_used,
                }
            )

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
        "source_generations": str(args.source_generations),
        "output_dir": str(out_dir),
        "requested_prompt_ids": requested_prompt_ids,
        "prompt_ids": prompt_ids,
        "n_prompt_pairs": len(prompt_ids),
        "skipped_pairs": [{"prompt_id": prompt_id, "reason": reason} for prompt_id, reason in missing],
        "conditions": ["private_rights", "public_rights"],
        "layers": "all" if len(layers) == len(layers_mod) else ",".join(str(layer) for layer in layers),
        "n_layers": len(layers),
        "components": components,
        "regions": regions,
        "span_mode": args.span_mode,
        "evidence_select": args.evidence_select,
        "max_length": args.max_length,
        "shuffle_controls": args.shuffle_controls,
        "top_k": args.top_k,
        "seed": args.seed,
        "method": "actual_answer_conclusion_evidence_private_minus_public_state_localization",
        "not_actuator_evidence": True,
        **source_budget,
    }
    write_json(out_dir / "manifest.json", manifest)
    write_jsonl(out_dir / "token_spans.jsonl", token_rows)
    write_jsonl(out_dir / "site_metrics.jsonl", metrics)
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
