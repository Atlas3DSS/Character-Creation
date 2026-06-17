#!/usr/bin/env python3
"""Smoke-test Qwen-Scope SAE feature pokes on ordinary legal prompts.

This is intentionally a qualitative causal poke, not a steering claim. It takes
weak all-justice overlap rows, builds residual directions from differential SAE
decoder columns, injects them at the matching layer during generation, and saves
base / random-control / poked completions.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import scipy.sparse as sp
import torch
from transformers import AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from probe_scotus_sae_features import safe_sae_name  # noqa: E402
from probe_scotus_style import markdown_table, model_cached, read_jsonl, transformer_layers, write_json, write_jsonl  # noqa: E402
from qwen_eval_budget import (  # noqa: E402
    DEFAULT_COMPLETE_ANSWER_TOKENS,
    add_short_budget_arg,
    enforce_complete_answer_budget,
    qwen_budget_metadata,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MODEL = Path("/home/orwel/dev_genius/models/Qwen3.6-27B-FP8")
DEFAULT_SAE_PATH = Path("/home/orwel/dev_genius/models/qwen_scope/SAE-Res-Qwen3.5-27B-W80K-L0_100")
DEFAULT_OVERLAP_DIR = PROJECT_ROOT / "sweep_v4" / "scotus_sae_overlap_all4_20260430_153933"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "sweep_v4"
DEFAULT_CANDIDATES = (
    "judicial_power_ginsburg_minus_thomas_l16,"
    "judicial_power_thomas_minus_ginsburg_l16,"
    "criminal_procedure_scalia_minus_ginsburg_l16,"
    "criminal_procedure_ginsburg_minus_scalia_l16"
)

NORMAL_PROMPTS = [
    (
        "A federal agency adjudicates a dispute between a private company and the government. "
        "The losing party argues that Article III requires an independent federal court to decide "
        "the matter. Analyze the constitutional problem in a concise judicial style."
    ),
    (
        "Police arrest a suspect outside an apartment, search a locked backpack nearby without a "
        "warrant, and find incriminating evidence. Analyze the Fourth Amendment question in a "
        "concise judicial style."
    ),
    (
        "Congress creates a federal remedy with statutory damages for misleading commercial conduct. "
        "A defendant argues that the statute exceeds federal power and conflicts with traditional "
        "state regulation. Analyze the interpretive and constitutional questions in a concise judicial style."
    ),
]

FRAME_PATTERNS = {
    "article3_public_rights": [
        "public right",
        "public rights",
        "murray",
        "regulatory scheme",
        "statutory grant",
        "sovereign regulator",
        "federal regulatory",
    ],
    "article3_private_rights": [
        "private right",
        "private rights",
        "common-law",
        "common law",
        "damages",
        "traditionally reserved for judicial",
    ],
    "article3_article1_tribunal": [
        "article i",
        "legislative court",
        "legislative courts",
        "northern pipeline",
        "life tenure",
        "salary protection",
        "full judicial power",
    ],
    "article3_case_or_controversy": [
        "case or controversy",
        "full panoply",
        "article iii protections",
    ],
    "fourth_plain_view_closed_container": [
        "plain view",
        "closed container",
        "locked backpack",
        "reasonable expectation of privacy",
        "katz",
        "acevedo",
    ],
    "fourth_search_incident_chimel": [
        "search incident",
        "chimel",
        "robinson",
        "immediate control",
        "grabbing distance",
        "wingspan",
    ],
    "fourth_exigency_consent": [
        "exigent",
        "consent",
    ],
    "fourth_safety_evidence": [
        "officer safety",
        "destruction of evidence",
        "prevent evidence",
        "evidence destruction",
    ],
    "economic_commerce_clause": [
        "commerce clause",
        "interstate commerce",
        "substantially affects commerce",
        "channels of commerce",
        "instrumentalities of commerce",
    ],
    "economic_federalism_state_regulation": [
        "federalism",
        "state regulation",
        "traditional state",
        "police power",
        "reserved to the states",
        "preemption",
    ],
    "economic_statutory_interpretation": [
        "statutory interpretation",
        "plain meaning",
        "text of the statute",
        "congressional intent",
        "clear statement",
    ],
    "economic_remedy_damages": [
        "statutory damages",
        "remedy",
        "private right of action",
        "civil penalty",
    ],
    "economic_commerce_limits": [
        "non-economic",
        "noneconomic",
        "lopez",
        "morrison",
        "attenuated",
        "jurisdictional element",
    ],
    "civil_equal_protection_strict_scrutiny": [
        "equal protection",
        "strict scrutiny",
        "compelling interest",
        "narrowly tailored",
        "race",
        "racial classification",
    ],
    "civil_sex_equality_intermediate": [
        "sex",
        "gender",
        "intermediate scrutiny",
        "exceedingly persuasive",
        "vmi",
        "virginia military institute",
    ],
    "civil_voting_race_districting": [
        "redistricting",
        "district",
        "racial gerrymandering",
        "race predominated",
        "predominant factor",
        "voting rights act",
    ],
    "civil_section5_congruence": [
        "section 5",
        "fourteenth amendment enforcement",
        "congruent and proportional",
        "congruence and proportionality",
        "sovereign immunity",
    ],
    "federalism_anti_commandeering": [
        "commandeer",
        "commandeering",
        "state officers",
        "state officials",
        "printz",
        "new york v. united states",
    ],
    "federalism_preemption": [
        "preemption",
        "preempt",
        "supremacy clause",
        "conflict preemption",
        "obstacle",
        "express preemption",
    ],
    "admin_major_questions": [
        "major questions",
        "clear congressional authorization",
        "vast economic and political significance",
        "nondelegation",
        "agency authority",
    ],
    "due_process_substantive": [
        "substantive due process",
        "liberty",
        "history and tradition",
        "ordered liberty",
        "fundamental right",
    ],
    "due_process_procedural_mathews": [
        "procedural due process",
        "mathews",
        "private interest",
        "risk of erroneous deprivation",
        "hearing",
        "government's interest",
    ],
    "separation_presidential_power": [
        "unitary executive",
        "removal",
        "presidential control",
        "independent agency",
        "separation of powers",
    ],
    "fourth_digital_privacy": [
        "cell phone",
        "smartphone",
        "digital",
        "riley",
        "messages",
        "data",
    ],
    "fourth_stop_reasonable_suspicion": [
        "traffic stop",
        "reasonable suspicion",
        "dog sniff",
        "prolong",
        "rodriguez",
    ],
    "fourth_home_exigency": [
        "home",
        "emergency aid",
        "warrantless entry",
        "hot pursuit",
        "brigham city",
    ],
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
class CandidateSpec:
    name: str
    layer: int
    region: str
    group_field: str
    group_value: str
    target_justice: str
    reference_justice: str


CANDIDATE_SPECS: dict[str, CandidateSpec] = {
    "judicial_power_ginsburg_minus_thomas_l16": CandidateSpec(
        name="judicial_power_ginsburg_minus_thomas_l16",
        layer=16,
        region="excerpt_mean",
        group_field="issue_area_label",
        group_value="Judicial Power",
        target_justice="Ginsburg",
        reference_justice="Thomas",
    ),
    "judicial_power_thomas_minus_ginsburg_l16": CandidateSpec(
        name="judicial_power_thomas_minus_ginsburg_l16",
        layer=16,
        region="excerpt_mean",
        group_field="issue_area_label",
        group_value="Judicial Power",
        target_justice="Thomas",
        reference_justice="Ginsburg",
    ),
    "criminal_procedure_scalia_minus_ginsburg_l16": CandidateSpec(
        name="criminal_procedure_scalia_minus_ginsburg_l16",
        layer=16,
        region="excerpt_mean",
        group_field="issue_area_label",
        group_value="Criminal Procedure",
        target_justice="Scalia",
        reference_justice="Ginsburg",
    ),
    "criminal_procedure_ginsburg_minus_scalia_l16": CandidateSpec(
        name="criminal_procedure_ginsburg_minus_scalia_l16",
        layer=16,
        region="excerpt_mean",
        group_field="issue_area_label",
        group_value="Criminal Procedure",
        target_justice="Ginsburg",
        reference_justice="Scalia",
    ),
}


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def parse_csv(raw: str) -> list[str]:
    return [part.strip() for part in raw.split(",") if part.strip()]


def parse_float_list(raw: str) -> list[float]:
    return [float(part.strip()) for part in raw.split(",") if part.strip()]


def parse_int_list(raw: str) -> list[int]:
    return [int(part.strip()) for part in raw.split(",") if part.strip()]


def default_prompt_specs() -> list[PromptSpec]:
    return [
        PromptSpec(
            prompt_id=idx,
            prompt_key=f"default_{idx}",
            issue_area="unspecified",
            prompt=prompt,
            expected_frames=tuple(),
            contrast_frames=tuple(),
            domain_frames=tuple(FRAME_PATTERNS),
        )
        for idx, prompt in enumerate(NORMAL_PROMPTS)
    ]


def load_prompt_specs(prompt_bank: Path | None) -> list[PromptSpec]:
    if prompt_bank is None:
        return default_prompt_specs()
    rows = read_jsonl(prompt_bank)
    specs: list[PromptSpec] = []
    known_frames = set(FRAME_PATTERNS)
    for fallback_id, row in enumerate(rows):
        prompt = str(row.get("prompt") or row.get("text") or "").strip()
        if not prompt:
            raise ValueError(f"Prompt bank row {fallback_id} has no prompt/text field")
        expected_frames = tuple(str(item) for item in row.get("expected_frames", []))
        contrast_frames = tuple(str(item) for item in row.get("contrast_frames", []))
        domain_frames = tuple(str(item) for item in row.get("domain_frames", []))
        unknown = sorted((set(expected_frames) | set(contrast_frames) | set(domain_frames)) - known_frames)
        if unknown:
            raise ValueError(f"Prompt bank row {fallback_id} references unknown frame tags: {unknown}")
        specs.append(
            PromptSpec(
                prompt_id=int(row.get("prompt_id", fallback_id)),
                prompt_key=str(row.get("prompt_key") or row.get("id") or f"prompt_{fallback_id}"),
                issue_area=str(row.get("issue_area") or "unspecified"),
                prompt=prompt,
                expected_frames=expected_frames,
                contrast_frames=contrast_frames,
                domain_frames=domain_frames or tuple(FRAME_PATTERNS),
            )
        )
    if not specs:
        raise ValueError(f"Prompt bank is empty: {prompt_bank}")
    return specs


def select_prompt_specs(specs: list[PromptSpec], prompt_ids: str, max_prompts: int) -> list[PromptSpec]:
    if not prompt_ids.strip():
        return specs[: min(len(specs), max(1, max_prompts))]
    selected: list[PromptSpec] = []
    for token in parse_csv(prompt_ids):
        matches = [spec for spec in specs if str(spec.prompt_id) == token or spec.prompt_key == token]
        if not matches:
            raise ValueError(
                f"Invalid prompt selector {token!r}. Available ids/keys: "
                f"{', '.join(f'{spec.prompt_id}:{spec.prompt_key}' for spec in specs)}"
            )
        selected.extend(matches)
    return selected


def clean_snippet(text: str, max_chars: int = 700) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= max_chars:
        return text
    cut = text[: max_chars + 1]
    if " " in cut:
        cut = cut[: cut.rfind(" ")]
    return cut.rstrip() + "..."


def tag_frames(text: str) -> dict[str, int]:
    lowered = text.lower()
    scores: dict[str, int] = {}
    for frame, patterns in FRAME_PATTERNS.items():
        score = 0
        for pattern in patterns:
            score += lowered.count(pattern)
        if score:
            scores[frame] = score
    return scores


def format_frame_scores(scores: dict[str, int]) -> str:
    if not scores:
        return ""
    return ", ".join(f"{name}:{score}" for name, score in sorted(scores.items(), key=lambda item: (-item[1], item[0])))


def format_optional_float(value: Any, default: Any = "") -> str:
    if value is None:
        value = default
    if value in (None, ""):
        return ""
    return f"{float(value):.3f}"


def frame_eval_for_prompt(spec: PromptSpec, frame_scores: dict[str, int]) -> dict[str, Any]:
    expected = set(spec.expected_frames)
    contrast = set(spec.contrast_frames)
    domain = set(spec.domain_frames)
    active = {name for name, score in frame_scores.items() if score > 0}
    off_domain = active - domain
    return {
        "expected_frames": list(spec.expected_frames),
        "contrast_frames": list(spec.contrast_frames),
        "domain_frames": list(spec.domain_frames),
        "target_hits": int(sum(frame_scores.get(name, 0) for name in expected)),
        "target_frames_present": int(sum(1 for name in expected if frame_scores.get(name, 0) > 0)),
        "target_frame_count": int(len(expected)),
        "target_present": bool(expected and any(frame_scores.get(name, 0) > 0 for name in expected)),
        "contrast_hits": int(sum(frame_scores.get(name, 0) for name in contrast)),
        "contrast_present": bool(contrast and any(frame_scores.get(name, 0) > 0 for name in contrast)),
        "off_domain_hits": int(sum(frame_scores.get(name, 0) for name in off_domain)),
        "off_domain_present": bool(off_domain),
        "off_domain_frames": sorted(off_domain),
        "total_frame_hits": int(sum(frame_scores.values())),
    }


def row_for_generation(
    *,
    spec: PromptSpec,
    condition: str,
    candidate: str | None,
    alpha: float,
    effective_alpha: float | None = None,
    random_index: int | None,
    layer: int | None,
    output: dict[str, Any],
) -> dict[str, Any]:
    frame_scores = tag_frames(output["text"])
    return {
        "prompt_id": spec.prompt_id,
        "prompt_key": spec.prompt_key,
        "issue_area": spec.issue_area,
        "prompt": spec.prompt,
        "condition": condition,
        "candidate": candidate,
        "alpha": float(alpha),
        "effective_alpha": None if effective_alpha is None else float(effective_alpha),
        "random_index": random_index,
        "layer": layer,
        "frame_scores": frame_scores,
        "frame_eval": frame_eval_for_prompt(spec, frame_scores),
        **output,
    }


def add_base_deltas(rows: list[dict[str, Any]]) -> None:
    base_by_prompt = {
        row["prompt_id"]: row["frame_eval"]
        for row in rows
        if row["condition"] == "base"
    }
    for row in rows:
        base = base_by_prompt.get(row["prompt_id"])
        if base is None:
            continue
        frame_eval = row["frame_eval"]
        for key in ("target_hits", "contrast_hits", "off_domain_hits", "total_frame_hits"):
            frame_eval[f"delta_{key}_vs_base"] = float(frame_eval[key] - base[key])
        frame_eval["delta_target_minus_contrast_vs_base"] = float(
            (frame_eval["target_hits"] - frame_eval["contrast_hits"])
            - (base["target_hits"] - base["contrast_hits"])
        )


def mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def stdev(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    avg = mean(values)
    return float((sum((value - avg) ** 2 for value in values) / (len(values) - 1)) ** 0.5)


def aggregate_frame_scores(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        key = (
            row["condition"],
            row.get("candidate"),
            None if row["condition"] == "base" else row.get("alpha"),
        )
        groups.setdefault(key, []).append(row)

    summaries: list[dict[str, Any]] = []
    for (condition, candidate, alpha), group_rows in sorted(groups.items(), key=lambda item: str(item[0])):
        evals = [row["frame_eval"] for row in group_rows]
        summaries.append(
            {
                "condition": condition,
                "candidate": candidate,
                "alpha": alpha,
                "n": len(group_rows),
                "prompt_count": len({row["prompt_id"] for row in group_rows}),
                "target_present_rate": mean([1.0 if item["target_present"] else 0.0 for item in evals]),
                "contrast_present_rate": mean([1.0 if item["contrast_present"] else 0.0 for item in evals]),
                "off_domain_present_rate": mean([1.0 if item["off_domain_present"] else 0.0 for item in evals]),
                "mean_target_hits": mean([float(item["target_hits"]) for item in evals]),
                "mean_contrast_hits": mean([float(item["contrast_hits"]) for item in evals]),
                "mean_off_domain_hits": mean([float(item["off_domain_hits"]) for item in evals]),
                "mean_delta_target_hits_vs_base": mean(
                    [float(item.get("delta_target_hits_vs_base", 0.0)) for item in evals]
                ),
                "mean_delta_contrast_hits_vs_base": mean(
                    [float(item.get("delta_contrast_hits_vs_base", 0.0)) for item in evals]
                ),
                "mean_delta_target_minus_contrast_vs_base": mean(
                    [float(item.get("delta_target_minus_contrast_vs_base", 0.0)) for item in evals]
                ),
                "mean_delta_off_domain_hits_vs_base": mean(
                    [float(item.get("delta_off_domain_hits_vs_base", 0.0)) for item in evals]
                ),
            }
        )
    return summaries


def candidate_vs_random(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    random_by_alpha: dict[float, list[float]] = {}
    random_net_by_alpha: dict[float, list[float]] = {}
    for row in rows:
        if row["condition"] == "random_unit":
            alpha = float(row["alpha"])
            random_by_alpha.setdefault(alpha, []).append(float(row["frame_eval"].get("delta_target_hits_vs_base", 0.0)))
            random_net_by_alpha.setdefault(alpha, []).append(
                float(row["frame_eval"].get("delta_target_minus_contrast_vs_base", 0.0))
            )

    sae_groups: dict[tuple[str, float], list[float]] = {}
    sae_net_groups: dict[tuple[str, float], list[float]] = {}
    for row in rows:
        if row["condition"] == "sae_poke" and row.get("candidate"):
            key = (str(row["candidate"]), float(row["alpha"]))
            sae_groups.setdefault(key, []).append(float(row["frame_eval"].get("delta_target_hits_vs_base", 0.0)))
            sae_net_groups.setdefault(key, []).append(
                float(row["frame_eval"].get("delta_target_minus_contrast_vs_base", 0.0))
            )

    comparisons: list[dict[str, Any]] = []
    for (candidate, alpha), values in sorted(sae_groups.items()):
        random_values = random_by_alpha.get(alpha, [])
        random_net_values = random_net_by_alpha.get(alpha, [])
        random_mean = mean(random_values)
        random_sd = stdev(random_values)
        candidate_mean = mean(values)
        candidate_net_mean = mean(sae_net_groups.get((candidate, alpha), []))
        random_net_mean = mean(random_net_values)
        random_net_sd = stdev(random_net_values)
        if random_values:
            percentile = sum(1 for value in random_values if value <= candidate_mean) / len(random_values)
        else:
            percentile = 0.0
        if random_net_values:
            net_percentile = sum(1 for value in random_net_values if value <= candidate_net_mean) / len(random_net_values)
        else:
            net_percentile = 0.0
        z_score = 0.0 if random_sd == 0.0 else (candidate_mean - random_mean) / random_sd
        net_z_score = 0.0 if random_net_sd == 0.0 else (candidate_net_mean - random_net_mean) / random_net_sd
        comparisons.append(
            {
                "candidate": candidate,
                "alpha": alpha,
                "n": len(values),
                "candidate_mean_delta_target_hits_vs_base": candidate_mean,
                "random_mean_delta_target_hits_vs_base": random_mean,
                "random_sd_delta_target_hits_vs_base": random_sd,
                "z_vs_random": float(z_score),
                "percentile_vs_random_rows": float(percentile),
                "candidate_mean_delta_target_minus_contrast_vs_base": candidate_net_mean,
                "random_mean_delta_target_minus_contrast_vs_base": random_net_mean,
                "random_sd_delta_target_minus_contrast_vs_base": random_net_sd,
                "z_net_vs_random": float(net_z_score),
                "percentile_net_vs_random_rows": float(net_percentile),
            }
        )
    return comparisons


def candidate_vs_prompt_matched_random(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    random_by_prompt_alpha: dict[tuple[int, float], list[float]] = {}
    random_net_by_prompt_alpha: dict[tuple[int, float], list[float]] = {}
    for row in rows:
        if row["condition"] != "random_unit":
            continue
        key = (int(row["prompt_id"]), float(row["alpha"]))
        value = float(row["frame_eval"].get("delta_target_hits_vs_base", 0.0))
        random_by_prompt_alpha.setdefault(key, []).append(value)
        net_value = float(row["frame_eval"].get("delta_target_minus_contrast_vs_base", 0.0))
        random_net_by_prompt_alpha.setdefault(key, []).append(net_value)

    random_residuals_by_alpha: dict[float, list[float]] = {}
    random_net_residuals_by_alpha: dict[float, list[float]] = {}
    for (prompt_id, alpha), values in random_by_prompt_alpha.items():
        random_mean = mean(values)
        random_residuals_by_alpha.setdefault(alpha, []).extend(value - random_mean for value in values)
    for (prompt_id, alpha), values in random_net_by_prompt_alpha.items():
        random_mean = mean(values)
        random_net_residuals_by_alpha.setdefault(alpha, []).extend(value - random_mean for value in values)

    candidate_groups: dict[tuple[str, float], list[dict[str, Any]]] = {}
    for row in rows:
        if row["condition"] != "sae_poke" or not row.get("candidate"):
            continue
        key = (str(row["candidate"]), float(row["alpha"]))
        candidate_groups.setdefault(key, []).append(row)

    comparisons: list[dict[str, Any]] = []
    for (candidate, alpha), group_rows in sorted(candidate_groups.items()):
        adjusted_values: list[float] = []
        adjusted_net_values: list[float] = []
        candidate_values: list[float] = []
        candidate_net_values: list[float] = []
        matched_random_means: list[float] = []
        matched_random_net_means: list[float] = []
        prompts_above_random = 0
        prompts_net_above_random = 0
        matched_prompts = 0
        for row in group_rows:
            prompt_id = int(row["prompt_id"])
            candidate_value = float(row["frame_eval"].get("delta_target_hits_vs_base", 0.0))
            candidate_net_value = float(row["frame_eval"].get("delta_target_minus_contrast_vs_base", 0.0))
            random_values = random_by_prompt_alpha.get((prompt_id, alpha), [])
            random_net_values = random_net_by_prompt_alpha.get((prompt_id, alpha), [])
            if not random_values or not random_net_values:
                continue
            matched_prompts += 1
            random_mean = mean(random_values)
            random_net_mean = mean(random_net_values)
            candidate_values.append(candidate_value)
            candidate_net_values.append(candidate_net_value)
            matched_random_means.append(random_mean)
            matched_random_net_means.append(random_net_mean)
            adjusted_value = candidate_value - random_mean
            adjusted_net_value = candidate_net_value - random_net_mean
            adjusted_values.append(adjusted_value)
            adjusted_net_values.append(adjusted_net_value)
            if adjusted_value > 0.0:
                prompts_above_random += 1
            if adjusted_net_value > 0.0:
                prompts_net_above_random += 1

        random_residuals = random_residuals_by_alpha.get(alpha, [])
        random_net_residuals = random_net_residuals_by_alpha.get(alpha, [])
        adjusted_mean = mean(adjusted_values)
        adjusted_net_mean = mean(adjusted_net_values)
        residual_sd = stdev(random_residuals)
        net_residual_sd = stdev(random_net_residuals)
        if random_residuals:
            percentile = sum(1 for value in random_residuals if value <= adjusted_mean) / len(random_residuals)
        else:
            percentile = 0.0
        if random_net_residuals:
            net_percentile = sum(1 for value in random_net_residuals if value <= adjusted_net_mean) / len(random_net_residuals)
        else:
            net_percentile = 0.0
        z_score = 0.0 if residual_sd == 0.0 else adjusted_mean / residual_sd
        net_z_score = 0.0 if net_residual_sd == 0.0 else adjusted_net_mean / net_residual_sd
        comparisons.append(
            {
                "candidate": candidate,
                "alpha": alpha,
                "n": len(adjusted_values),
                "prompt_count": matched_prompts,
                "candidate_mean_delta_target_hits_vs_base": mean(candidate_values),
                "prompt_random_mean_delta_target_hits_vs_base": mean(matched_random_means),
                "mean_prompt_matched_delta_minus_random": adjusted_mean,
                "random_residual_sd": residual_sd,
                "z_vs_prompt_matched_random": float(z_score),
                "percentile_vs_prompt_matched_random_rows": float(percentile),
                "prompt_win_rate_vs_random_mean": 0.0
                if matched_prompts == 0
                else float(prompts_above_random / matched_prompts),
                "candidate_mean_delta_target_minus_contrast_vs_base": mean(candidate_net_values),
                "prompt_random_mean_delta_target_minus_contrast_vs_base": mean(matched_random_net_means),
                "mean_prompt_matched_net_delta_minus_random": adjusted_net_mean,
                "random_net_residual_sd": net_residual_sd,
                "z_net_vs_prompt_matched_random": float(net_z_score),
                "percentile_net_vs_prompt_matched_random_rows": float(net_percentile),
                "prompt_net_win_rate_vs_random_mean": 0.0
                if matched_prompts == 0
                else float(prompts_net_above_random / matched_prompts),
            }
        )
    return comparisons


def summarize_frames(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summary: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in rows:
        key = (
            row["prompt_id"],
            row["condition"],
            row.get("candidate"),
            row.get("alpha"),
            row.get("random_index"),
        )
        item = summary.setdefault(
            key,
            {
                "prompt_id": row["prompt_id"],
                "condition": row["condition"],
                "candidate": row.get("candidate"),
                "alpha": row.get("alpha"),
                "random_index": row.get("random_index"),
                "n": 0,
                "frame_scores": {},
            },
        )
        item["n"] += 1
        for frame, score in row.get("frame_scores", {}).items():
            item["frame_scores"][frame] = int(item["frame_scores"].get(frame, 0)) + int(score)
    return sorted(
        summary.values(),
        key=lambda item: (
            item["prompt_id"],
            str(item["condition"]),
            str(item.get("candidate")),
            float(item.get("alpha") or 0.0),
            int(item.get("random_index") or 0),
        ),
    )


def feature_cache_path(overlap_dir: Path, sae_path: Path, region: str, layer: int) -> Path:
    return overlap_dir / "sae_features" / f"{safe_sae_name(sae_path)}__{region}__L{layer:02d}.npz"


def load_decoder_columns(sae_path: Path, layer: int, feature_ids: np.ndarray) -> torch.Tensor:
    path = sae_path / f"layer{layer}.sae.pt"
    if not path.exists():
        raise FileNotFoundError(path)
    sae = torch.load(path, map_location="cpu", weights_only=True)
    w_dec = sae["W_dec"][:, feature_ids.tolist()].float().contiguous()
    del sae
    return w_dec


def build_candidate_direction(
    *,
    spec: CandidateSpec,
    overlap_dir: Path,
    sae_path: Path,
    top_features: int,
) -> tuple[torch.Tensor, dict[str, Any]]:
    meta_rows = read_jsonl(overlap_dir / "feature_meta.jsonl")
    matrix_path = feature_cache_path(overlap_dir, sae_path, spec.region, spec.layer)
    if not matrix_path.exists():
        raise FileNotFoundError(matrix_path)
    matrix = sp.load_npz(matrix_path).tocsr()
    if matrix.shape[0] != len(meta_rows):
        raise RuntimeError(f"Matrix/meta row mismatch: {matrix.shape[0]} vs {len(meta_rows)}")

    target_idx = np.array(
        [
            idx
            for idx, row in enumerate(meta_rows)
            if row.get("justice") == spec.target_justice and row.get(spec.group_field) == spec.group_value
        ],
        dtype=np.int64,
    )
    reference_idx = np.array(
        [
            idx
            for idx, row in enumerate(meta_rows)
            if row.get("justice") == spec.reference_justice and row.get(spec.group_field) == spec.group_value
        ],
        dtype=np.int64,
    )
    if len(target_idx) == 0 or len(reference_idx) == 0:
        raise RuntimeError(f"Empty target/reference group for {spec}")

    target_mean = np.asarray(matrix[target_idx].mean(axis=0)).ravel()
    reference_mean = np.asarray(matrix[reference_idx].mean(axis=0)).ravel()
    diff = target_mean - reference_mean
    positive = np.flatnonzero(diff > 0)
    if len(positive) == 0:
        raise RuntimeError(f"No positive differential features for {spec.name}")
    ranked = positive[np.argsort(diff[positive])[::-1]]
    feature_ids = ranked[:top_features].astype(np.int64)
    weights = diff[feature_ids].astype(np.float32)
    weight_sum = float(weights.sum())
    if weight_sum <= 0:
        raise RuntimeError(f"Non-positive differential weight sum for {spec.name}")
    normalized_weights = weights / weight_sum

    decoder_cols = load_decoder_columns(sae_path, spec.layer, feature_ids)
    direction = decoder_cols @ torch.from_numpy(normalized_weights)
    raw_norm = float(torch.linalg.vector_norm(direction).item())
    if raw_norm <= 0:
        raise RuntimeError(f"Zero decoder direction for {spec.name}")
    direction = direction / raw_norm

    meta = {
        "source": "sae_decoder",
        "name": spec.name,
        "layer": spec.layer,
        "region": spec.region,
        "group_field": spec.group_field,
        "group_value": spec.group_value,
        "target_justice": spec.target_justice,
        "reference_justice": spec.reference_justice,
        "n_target": int(len(target_idx)),
        "n_reference": int(len(reference_idx)),
        "top_features": [
            {
                "feature": int(feature_id),
                "target_mean_activation": float(target_mean[feature_id]),
                "reference_mean_activation": float(reference_mean[feature_id]),
                "diff": float(diff[feature_id]),
            }
            for feature_id in feature_ids.tolist()
        ],
        "raw_decoder_combo_norm": raw_norm,
        "weight_sum": weight_sum,
    }
    return direction.contiguous(), meta


def build_raw_hidden_direction(
    *,
    spec: CandidateSpec,
    overlap_dir: Path,
) -> tuple[torch.Tensor, dict[str, Any]]:
    meta_rows = read_jsonl(overlap_dir / "feature_meta.jsonl")
    features_path = overlap_dir / "features.npz"
    key = f"{spec.region}__L{spec.layer:02d}"
    if not features_path.exists():
        raise FileNotFoundError(features_path)
    with np.load(features_path) as data:
        if key not in data.files:
            raise KeyError(f"Missing {key} in {features_path}")
        hidden = data[key].astype(np.float32, copy=False)
    if hidden.shape[0] != len(meta_rows):
        raise RuntimeError(f"Feature/meta row mismatch: {hidden.shape[0]} vs {len(meta_rows)}")

    target_idx = np.array(
        [
            idx
            for idx, row in enumerate(meta_rows)
            if row.get("justice") == spec.target_justice and row.get(spec.group_field) == spec.group_value
        ],
        dtype=np.int64,
    )
    reference_idx = np.array(
        [
            idx
            for idx, row in enumerate(meta_rows)
            if row.get("justice") == spec.reference_justice and row.get(spec.group_field) == spec.group_value
        ],
        dtype=np.int64,
    )
    if len(target_idx) == 0 or len(reference_idx) == 0:
        raise RuntimeError(f"Empty target/reference group for {spec}")

    target_mean = hidden[target_idx].mean(axis=0)
    reference_mean = hidden[reference_idx].mean(axis=0)
    direction_np = target_mean - reference_mean
    raw_norm = float(np.linalg.norm(direction_np))
    if raw_norm <= 0:
        raise RuntimeError(f"Zero raw-hidden direction for {spec.name}")
    direction = torch.from_numpy((direction_np / raw_norm).astype(np.float32, copy=False)).contiguous()
    meta = {
        "source": "raw_hidden",
        "name": f"raw_hidden_{spec.name}",
        "base_name": spec.name,
        "layer": spec.layer,
        "region": spec.region,
        "group_field": spec.group_field,
        "group_value": spec.group_value,
        "target_justice": spec.target_justice,
        "reference_justice": spec.reference_justice,
        "n_target": int(len(target_idx)),
        "n_reference": int(len(reference_idx)),
        "raw_direction_norm": raw_norm,
    }
    return direction, meta


def load_external_direction(path: Path) -> tuple[torch.Tensor, dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=True) as data:
        if "raw_direction_unit" in data.files:
            direction_np = data["raw_direction_unit"].astype(np.float32, copy=False)
            raw_norm = float(data["raw_direction_norm"][0]) if "raw_direction_norm" in data.files else 1.0
        elif "coef" in data.files and "scaler_scale" in data.files:
            coef = data["coef"].reshape(-1).astype(np.float32, copy=False)
            scale = data["scaler_scale"].astype(np.float32, copy=False)
            raw = coef / np.maximum(scale, 1e-12)
            raw_norm = float(np.linalg.norm(raw))
            if raw_norm <= 0.0:
                raise RuntimeError(f"Zero external probe direction in {path}")
            direction_np = (raw / raw_norm).astype(np.float32, copy=False)
        else:
            raise KeyError(f"{path} must contain raw_direction_unit or coef+scaler_scale")

        layer = int(data["layer"][0]) if "layer" in data.files else 16
        region = str(data["region"][0]) if "region" in data.files else "unknown"
        c_value = float(data["C"][0]) if "C" in data.files else None
        positive_justice = str(data["positive_justice"][0]) if "positive_justice" in data.files else "positive"
        source_runs = [str(item) for item in data["source_runs"].tolist()] if "source_runs" in data.files else []
        if "source_run" in data.files:
            source_runs.append(str(data["source_run"][0]))

    direction = torch.from_numpy(direction_np).float().contiguous()
    direction = direction / torch.linalg.vector_norm(direction).clamp(min=1e-12)
    direction_name = f"{path.parent.name}__{path.stem}"
    meta = {
        "source": "external_probe",
        "name": direction_name,
        "path": str(path),
        "layer": layer,
        "region": region,
        "C": c_value,
        "group_field": "external",
        "group_value": "probe_direction",
        "target_justice": positive_justice,
        "reference_justice": f"not_{positive_justice}",
        "n_target": "",
        "n_reference": "",
        "raw_direction_norm": raw_norm,
        "source_runs": source_runs,
    }
    return direction, meta


def median_hidden_norm(reference: Path, region: str, layer: int) -> float:
    features_path = reference / "features.npz" if reference.is_dir() else reference
    if not features_path.exists():
        raise FileNotFoundError(features_path)
    key = f"{region}__L{layer:02d}"
    with np.load(features_path) as data:
        if key not in data.files:
            raise KeyError(f"Missing {key} in {features_path}")
        values = data[key].astype(np.float32, copy=False)
        norms = np.linalg.norm(values, axis=1)
    median = float(np.median(norms))
    if median <= 0.0:
        raise RuntimeError(f"Non-positive median hidden norm for {key} in {features_path}")
    return median


def patch_qwen_imports() -> None:
    try:
        from transformers.models.qwen3_5 import modeling_qwen3_5

        modeling_qwen3_5.FusedRMSNormGated = None
        print("Disabled qwen3_5 FusedRMSNormGated for generation hooks", flush=True)
    except (ImportError, AttributeError):
        pass


def load_model_and_tokenizer(model_path: Path, device_map: str) -> tuple[Any, torch.nn.Module]:
    print(f"Model cache status for {model_path}: {model_cached(model_path)}", flush=True)
    if not model_cached(model_path):
        raise RuntimeError(f"Model is not cached locally: {model_path}")
    patch_qwen_imports()
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if hasattr(tokenizer, "padding_side"):
        tokenizer.padding_side = "left"

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    resolved_device_map: str | dict[str, int | str]
    if device_map.lower() in {"single", "cuda", "cuda:0", "gpu"}:
        resolved_device_map = {"": 0}
    elif device_map.lower() == "cpu":
        resolved_device_map = {"": "cpu"}
    else:
        resolved_device_map = device_map

    # The FP8 Qwen loader can over-reserve during Transformers' CUDA allocator
    # warmup on this workstation. The capture path already loads this model
    # without that warmup, so disable only the optional preallocation step.
    try:
        import transformers.modeling_utils as modeling_utils

        modeling_utils.caching_allocator_warmup = lambda *args, **kwargs: None
        print("Disabled Transformers caching allocator warmup", flush=True)
    except (ImportError, AttributeError):
        pass

    load_kwargs = {
        "trust_remote_code": True,
        "local_files_only": True,
        "torch_dtype": "auto",
        "device_map": resolved_device_map,
        "low_cpu_mem_usage": True,
        "attn_implementation": "sdpa",
    }
    try:
        print("Loading with AutoModelForImageTextToText", flush=True)
        model = AutoModelForImageTextToText.from_pretrained(str(model_path), **load_kwargs)
    except Exception as exc:
        print(f"AutoModelForImageTextToText failed: {exc!r}; retrying AutoModelForCausalLM", flush=True)
        model = AutoModelForCausalLM.from_pretrained(str(model_path), **load_kwargs)
    model.eval()
    return tokenizer, model


def first_parameter_device(model: torch.nn.Module) -> torch.device:
    return next(model.parameters()).device


def format_chat(tokenizer: Any, prompt: str) -> str:
    messages = [{"role": "user", "content": prompt}]
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def install_poke_hook(
    layers_mod: Any,
    *,
    layer: int,
    direction: torch.Tensor,
    alpha: float,
    position: str,
) -> Any:
    if position not in {"last", "all"}:
        raise ValueError(f"Unsupported poke position: {position}")

    def hook(_module: torch.nn.Module, _inp: tuple[Any, ...], out: Any) -> Any:
        hidden = out[0] if isinstance(out, tuple) else out
        poke = direction.to(device=hidden.device, dtype=hidden.dtype) * float(alpha)
        edited = hidden.clone()
        if position == "last":
            edited[:, -1, :] = edited[:, -1, :] + poke
        else:
            edited = edited + poke.view(1, 1, -1)
        if isinstance(out, tuple):
            return (edited,) + out[1:]
        return edited

    return layers_mod[layer].register_forward_hook(hook)


@torch.inference_mode()
def generate_one(
    *,
    model: torch.nn.Module,
    tokenizer: Any,
    prompt: str,
    layers_mod: Any,
    layer: int | None,
    direction: torch.Tensor | None,
    alpha: float,
    position: str,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    top_p: float,
) -> dict[str, Any]:
    chat = format_chat(tokenizer, prompt)
    inputs = tokenizer(chat, return_tensors="pt", add_special_tokens=False, truncation=True, max_length=2048)
    input_device = first_parameter_device(model)
    inputs = {key: value.to(input_device) for key, value in inputs.items()}

    hook_handle = None
    if layer is not None and direction is not None and alpha != 0.0:
        hook_handle = install_poke_hook(layers_mod, layer=layer, direction=direction, alpha=alpha, position=position)
    try:
        gen_kwargs: dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "use_cache": True,
            "eos_token_id": tokenizer.eos_token_id,
            "pad_token_id": tokenizer.pad_token_id,
        }
        if do_sample:
            gen_kwargs["temperature"] = temperature
            gen_kwargs["top_p"] = top_p
        output = model.generate(**inputs, **gen_kwargs)
    finally:
        if hook_handle is not None:
            hook_handle.remove()

    generated = output[0, inputs["input_ids"].shape[-1] :]
    text = tokenizer.decode(generated, skip_special_tokens=True).strip()
    return {
        "text": text,
        "generated_tokens": int(generated.numel()),
        "prompt_tokens": int(inputs["input_ids"].shape[-1]),
    }


@torch.inference_mode()
def generate_many(
    *,
    model: torch.nn.Module,
    tokenizer: Any,
    prompts: list[str],
    layers_mod: Any,
    layer: int | None,
    direction: torch.Tensor | None,
    alpha: float,
    position: str,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    top_p: float,
    batch_size: int,
) -> list[dict[str, Any]]:
    outputs: list[dict[str, Any]] = []
    hook_handle = None
    if layer is not None and direction is not None and alpha != 0.0:
        hook_handle = install_poke_hook(layers_mod, layer=layer, direction=direction, alpha=alpha, position=position)
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
            gen_kwargs: dict[str, Any] = {
                "max_new_tokens": max_new_tokens,
                "do_sample": do_sample,
                "use_cache": True,
                "eos_token_id": tokenizer.eos_token_id,
                "pad_token_id": tokenizer.pad_token_id,
            }
            if do_sample:
                gen_kwargs["temperature"] = temperature
                gen_kwargs["top_p"] = top_p
            generated_batch = model.generate(**inputs, **gen_kwargs)
            prompt_width = int(inputs["input_ids"].shape[-1])
            for generated in generated_batch[:, prompt_width:]:
                text = tokenizer.decode(generated, skip_special_tokens=True).strip()
                outputs.append(
                    {
                        "text": text,
                        "generated_tokens": int(generated.numel()),
                        "prompt_tokens": prompt_width,
                    }
                )
    finally:
        if hook_handle is not None:
            hook_handle.remove()
    return outputs


def make_random_direction(dim: int, seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    vec = torch.randn(dim, generator=generator, dtype=torch.float32)
    return vec / torch.linalg.vector_norm(vec).clamp(min=1e-12)


def write_report(path: Path, *, manifest: dict[str, Any], direction_meta: list[dict[str, Any]], rows: list[dict[str, Any]]) -> None:
    score_summary = aggregate_frame_scores(rows)
    comparisons = candidate_vs_random(rows)
    matched_comparisons = candidate_vs_prompt_matched_random(rows)
    direction_source = str(manifest.get("direction_source", "sae"))
    if direction_source == "external":
        title = "# SCOTUS External Probe Direction Poke"
        construction_note = (
            "Each poke direction is a unit-normalized hidden-state probe direction loaded from the listed `.npz` files. "
            "When alpha scaling is `hidden-norm-fraction`, the requested alpha is multiplied by the reference run's "
            "median hidden-state norm for the direction layer and region."
        )
        comparison_heading = "## External Direction vs Random"
        comparison_note = (
            "This compares each external candidate's mean target-frame delta against the row-level "
            "random-control distribution at the same requested alpha."
        )
    else:
        title = "# SCOTUS SAE Layer Poke"
        construction_note = (
            "Each poke direction is a unit-normalized weighted sum of Qwen-Scope SAE decoder columns. "
            "Weights are positive differential mean activations for the target justice minus the reference justice "
            "within the selected issue/posture group."
        )
        comparison_heading = "## SAE vs Random"
        comparison_note = (
            "This compares each SAE candidate's mean target-frame delta against the row-level "
            "random-control distribution at the same alpha."
        )
    config_rows = [
        ["Started", manifest["started_at"]],
        ["Finished", manifest.get("finished_at", "")],
        ["Model", manifest["model_path"]],
        ["Direction source", direction_source],
        ["SAE", manifest["sae_path"]],
        ["Overlap dir", manifest["overlap_dir"]],
        ["External direction files", ", ".join(manifest.get("external_direction_files", []))],
        ["Alpha scale", manifest.get("alpha_scale", "")],
        ["Hidden norm reference", manifest.get("hidden_norm_reference", "")],
        ["Alphas", ", ".join(str(alpha) for alpha in manifest["alphas"])],
        ["Random controls", manifest["random_controls"]],
        ["Position", manifest["position"]],
        ["Max new tokens", manifest["max_new_tokens"]],
        ["Short budget smoke", manifest.get("short_answer_budget", False)],
        ["Budget note", manifest.get("budget_note", "")],
        ["Generation batch size", manifest.get("generation_batch_size", "")],
        ["Do sample", manifest["do_sample"]],
        ["Rows", len(rows)],
    ]
    candidate_rows = [
        [
            item.get("source", ""),
            item["name"],
            item["layer"],
            item["region"],
            item["group_value"],
            f"{item['target_justice']} - {item['reference_justice']}",
            item["n_target"],
            item["n_reference"],
            ", ".join(str(feat["feature"]) for feat in item.get("top_features", [])[:8]),
        ]
        for item in direction_meta
    ]
    score_rows = [
        [
            row["condition"],
            row.get("candidate", ""),
            row.get("alpha", ""),
            row["n"],
            row["prompt_count"],
            f"{row['target_present_rate']:.2f}",
            f"{row['mean_delta_target_hits_vs_base']:.2f}",
            f"{row['mean_delta_contrast_hits_vs_base']:.2f}",
            f"{row['mean_delta_target_minus_contrast_vs_base']:.2f}",
            f"{row['mean_delta_off_domain_hits_vs_base']:.2f}",
            f"{row['off_domain_present_rate']:.2f}",
        ]
        for row in score_summary
    ]
    comparison_rows = [
        [
            row["candidate"],
            row["alpha"],
            row["n"],
            f"{row['candidate_mean_delta_target_hits_vs_base']:.2f}",
            f"{row['random_mean_delta_target_hits_vs_base']:.2f}",
            f"{row['random_sd_delta_target_hits_vs_base']:.2f}",
            f"{row['z_vs_random']:.2f}",
            f"{row['percentile_vs_random_rows']:.2f}",
            f"{row['z_net_vs_random']:.2f}",
            f"{row['percentile_net_vs_random_rows']:.2f}",
        ]
        for row in comparisons
    ]
    matched_comparison_rows = [
        [
            row["candidate"],
            row["alpha"],
            row["n"],
            f"{row['candidate_mean_delta_target_hits_vs_base']:.2f}",
            f"{row['prompt_random_mean_delta_target_hits_vs_base']:.2f}",
            f"{row['mean_prompt_matched_delta_minus_random']:.2f}",
            f"{row['random_residual_sd']:.2f}",
            f"{row['z_vs_prompt_matched_random']:.2f}",
            f"{row['percentile_vs_prompt_matched_random_rows']:.2f}",
            f"{row['prompt_win_rate_vs_random_mean']:.2f}",
            f"{row['mean_prompt_matched_net_delta_minus_random']:.2f}",
            f"{row['z_net_vs_prompt_matched_random']:.2f}",
            f"{row['percentile_net_vs_prompt_matched_random_rows']:.2f}",
            f"{row['prompt_net_win_rate_vs_random_mean']:.2f}",
        ]
        for row in matched_comparisons
    ]
    report_max_output_rows = int(manifest.get("report_max_output_rows", 200))
    displayed_rows = rows[:report_max_output_rows]
    output_rows = [
        [
            row["prompt_id"],
            row.get("prompt_key", ""),
            row.get("issue_area", ""),
            row["condition"],
            row.get("candidate", ""),
            row.get("alpha", ""),
            format_optional_float(row.get("effective_alpha"), row.get("alpha", "")),
            row.get("random_index", ""),
            row.get("layer", ""),
            format_frame_scores(row.get("frame_scores", {})),
            f"{row.get('frame_eval', {}).get('delta_target_hits_vs_base', 0.0):.1f}",
            f"{row.get('frame_eval', {}).get('delta_off_domain_hits_vs_base', 0.0):.1f}",
            row["generated_tokens"],
            clean_snippet(row["text"]),
        ]
        for row in displayed_rows
    ]
    frame_summary = summarize_frames(rows)
    frame_rows = [
        [
            row["prompt_id"],
            row["condition"],
            row.get("candidate", ""),
            row.get("alpha", ""),
            row.get("random_index", ""),
            format_frame_scores(row.get("frame_scores", {})),
        ]
        for row in frame_summary
    ]
    lines = [
        title,
        "",
        "## Configuration",
        "",
        markdown_table(["Field", "Value"], config_rows),
        "",
        "## Direction Construction",
        "",
        construction_note,
        "",
        markdown_table(
            ["Source", "Candidate", "Layer", "Region", "Group", "Direction", "N target", "N reference", "Top feature IDs"],
            candidate_rows,
        ),
        "",
        "## Aggregate Frame Scores",
        "",
        "Target deltas are relative to each prompt's unpoked baseline. Random rows are pooled by alpha.",
        "",
        markdown_table(
            [
                "Condition",
                "Candidate",
                "Alpha",
                "N",
                "Prompts",
                "Target present",
                "Mean target delta",
                "Mean contrast delta",
                "Mean net delta",
                "Mean off-domain delta",
                "Off-domain present",
            ],
            score_rows,
        ),
        "",
        comparison_heading,
        "",
        comparison_note,
        "",
        markdown_table(
            [
                "Candidate",
                "Alpha",
                "N",
                "Candidate mean delta",
                "Random mean delta",
                "Random SD",
                "Z",
                "Percentile",
                "Net Z",
                "Net percentile",
            ],
            comparison_rows,
        ),
        "",
        "## Prompt-Matched Direction vs Random",
        "",
        "This subtracts the same prompt's random-control mean before aggregating, so prompt-specific frame density does not dominate the comparison.",
        "",
        markdown_table(
            [
                "Candidate",
                "Alpha",
                "N",
                "Candidate mean delta",
                "Prompt random mean",
                "Matched delta",
                "Random residual SD",
                "Z",
                "Percentile",
                "Prompt win rate",
                "Matched net delta",
                "Net Z",
                "Net percentile",
                "Net win rate",
            ],
            matched_comparison_rows,
        ),
        "",
        "## Outputs",
        "",
        f"Showing {len(displayed_rows)} of {len(rows)} rows. Full rows are in `generations.jsonl`.",
        "",
        markdown_table(
            [
                "Prompt",
                "Key",
                "Issue",
                "Condition",
                "Candidate",
                "Alpha",
                "Eff alpha",
                "Rand",
                "Layer",
                "Frame tags",
                "Target delta",
                "Off-domain delta",
                "Tokens",
                "Completion",
            ],
            output_rows,
        ),
        "",
        "## Frame Tag Summary",
        "",
        markdown_table(
            ["Prompt", "Condition", "Candidate", "Alpha", "Rand", "Frame tags"],
            frame_rows,
        ),
        "",
        "## Reading Notes",
        "",
        "- This is an exploratory causal pilot on normal prompts with no justice names.",
        "- Random conditions are same-norm unit vectors at the same layer and alpha.",
        "- Frame tags are keyword diagnostics for comparing outputs; they are not a substitute for manual legal review.",
        "- Visible changes here would only justify a more controlled causal sweep; absence of visible changes does not disprove a feature-level effect.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Poke SCOTUS-related directions into normal legal prompts.")
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--sae-path", type=Path, default=DEFAULT_SAE_PATH)
    parser.add_argument("--overlap-dir", type=Path, default=DEFAULT_OVERLAP_DIR)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--prompt-bank", type=Path, default=None)
    parser.add_argument("--candidate-names", default=DEFAULT_CANDIDATES)
    parser.add_argument("--direction-source", choices=["sae", "raw-hidden", "external"], default="sae")
    parser.add_argument(
        "--external-direction-files",
        default="",
        help="Comma-separated .npz direction files used when --direction-source external.",
    )
    parser.add_argument(
        "--alpha-scale",
        choices=["unit", "hidden-norm-fraction"],
        default="unit",
        help="Interpret alphas as raw unit-vector scale or as fractions of median hidden-state norm.",
    )
    parser.add_argument(
        "--hidden-norm-reference",
        type=Path,
        default=None,
        help="Run directory or features.npz used for --alpha-scale hidden-norm-fraction.",
    )
    parser.add_argument("--top-features", type=int, default=32)
    parser.add_argument("--alphas", default="16")
    parser.add_argument("--position", choices=["last", "all"], default="last")
    parser.add_argument("--max-prompts", type=int, default=2)
    parser.add_argument(
        "--prompt-ids",
        default="",
        help="Optional comma-separated 0-based prompt ids. Overrides --max-prompts when set.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_COMPLETE_ANSWER_TOKENS)
    add_short_budget_arg(parser)
    parser.add_argument("--generation-batch-size", type=int, default=1)
    parser.add_argument("--report-max-output-rows", type=int, default=200)
    parser.add_argument("--device-map", default="single")
    parser.add_argument("--seed", type=int, default=29)
    parser.add_argument("--random-controls", type=int, default=1)
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    enforce_complete_answer_budget(
        args.max_new_tokens,
        allow_short=args.allow_short_answer_budget,
        purpose="SCOTUS SAE/external-direction poke",
    )
    started = now_iso()
    out_dir = args.output_root / f"scotus_sae_poke_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)

    candidate_names = parse_csv(args.candidate_names)
    external_direction_files = [Path(path) for path in parse_csv(args.external_direction_files)]
    if args.direction_source == "external":
        if not external_direction_files:
            raise ValueError("--direction-source external requires --external-direction-files")
    else:
        unknown = [name for name in candidate_names if name not in CANDIDATE_SPECS]
        if unknown:
            raise ValueError(f"Unknown candidates: {unknown}. Available: {sorted(CANDIDATE_SPECS)}")
    alphas = parse_float_list(args.alphas)
    all_prompt_specs = load_prompt_specs(args.prompt_bank)
    prompt_specs = select_prompt_specs(all_prompt_specs, args.prompt_ids, args.max_prompts)
    prompt_ids = [spec.prompt_id for spec in prompt_specs]
    prompt_texts = [spec.prompt for spec in prompt_specs]
    print(f"Loaded {len(prompt_specs)} prompts", flush=True)

    direction_meta: list[dict[str, Any]] = []
    directions: dict[str, torch.Tensor] = {}
    direction_inputs: list[str | Path] = external_direction_files if args.direction_source == "external" else candidate_names
    for item in direction_inputs:
        if args.direction_source == "external":
            direction, meta = load_external_direction(Path(item))
        elif args.direction_source == "sae":
            spec = CANDIDATE_SPECS[str(item)]
            direction, meta = build_candidate_direction(
                spec=spec,
                overlap_dir=args.overlap_dir,
                sae_path=args.sae_path,
                top_features=args.top_features,
            )
        else:
            spec = CANDIDATE_SPECS[str(item)]
            direction, meta = build_raw_hidden_direction(
                spec=spec,
                overlap_dir=args.overlap_dir,
            )
        if args.alpha_scale == "hidden-norm-fraction":
            reference = args.hidden_norm_reference
            if reference is None and meta.get("source_runs"):
                reference = Path(str(meta["source_runs"][0]))
            if reference is None:
                raise ValueError("--alpha-scale hidden-norm-fraction requires --hidden-norm-reference or direction source_run")
            meta["alpha_scale_factor"] = median_hidden_norm(reference, str(meta["region"]), int(meta["layer"]))
            meta["alpha_scale_reference"] = str(reference)
        else:
            meta["alpha_scale_factor"] = 1.0
            meta["alpha_scale_reference"] = ""
        directions[meta["name"]] = direction
        direction_meta.append(meta)
        print(
            f"Built {meta['name']}: {meta.get('source')} L{meta['layer']} "
            f"{meta['target_justice']} - {meta['reference_justice']}",
            flush=True,
        )

    tokenizer, model = load_model_and_tokenizer(args.model_path, args.device_map)
    layers_mod = transformer_layers(model)
    layer_dim = int(next(iter(directions.values())).numel())
    random_directions = [make_random_direction(layer_dim, args.seed + idx) for idx in range(max(0, args.random_controls))]
    layer_for_random = int(direction_meta[0]["layer"])
    random_alpha_scale_factor = float(direction_meta[0].get("alpha_scale_factor", 1.0))

    rows: list[dict[str, Any]] = []
    print("Generating baseline batch", flush=True)
    base_outputs = generate_many(
        model=model,
        tokenizer=tokenizer,
        prompts=prompt_texts,
        layers_mod=layers_mod,
        layer=None,
        direction=None,
        alpha=0.0,
        position=args.position,
        max_new_tokens=args.max_new_tokens,
        do_sample=args.do_sample,
        temperature=args.temperature,
        top_p=args.top_p,
        batch_size=args.generation_batch_size,
    )
    for spec, output in zip(prompt_specs, base_outputs, strict=True):
        rows.append(
            row_for_generation(
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
        for random_idx, random_direction in enumerate(random_directions):
            random_effective_alpha = float(alpha) * random_alpha_scale_factor
            print(f"Generating random[{random_idx}] alpha={alpha} effective={random_effective_alpha:.3f}", flush=True)
            random_outputs = generate_many(
                model=model,
                tokenizer=tokenizer,
                prompts=prompt_texts,
                layers_mod=layers_mod,
                layer=layer_for_random,
                direction=random_direction,
                alpha=random_effective_alpha,
                position=args.position,
                max_new_tokens=args.max_new_tokens,
                do_sample=args.do_sample,
                temperature=args.temperature,
                top_p=args.top_p,
                batch_size=args.generation_batch_size,
            )
            for spec, output in zip(prompt_specs, random_outputs, strict=True):
                rows.append(
                    row_for_generation(
                        spec=spec,
                        condition="random_unit",
                        candidate="random_unit",
                        alpha=float(alpha),
                        effective_alpha=random_effective_alpha,
                        random_index=int(random_idx),
                        layer=layer_for_random,
                        output=output,
                    )
                )
        for meta in direction_meta:
            name = meta["name"]
            effective_alpha = float(alpha) * float(meta.get("alpha_scale_factor", 1.0))
            print(f"Generating {name} alpha={alpha} effective={effective_alpha:.3f}", flush=True)
            candidate_outputs = generate_many(
                model=model,
                tokenizer=tokenizer,
                prompts=prompt_texts,
                layers_mod=layers_mod,
                layer=int(meta["layer"]),
                direction=directions[name],
                alpha=effective_alpha,
                position=args.position,
                max_new_tokens=args.max_new_tokens,
                do_sample=args.do_sample,
                temperature=args.temperature,
                top_p=args.top_p,
                batch_size=args.generation_batch_size,
            )
            for spec, output in zip(prompt_specs, candidate_outputs, strict=True):
                rows.append(
                    row_for_generation(
                        spec=spec,
                        condition="sae_poke",
                        candidate=name,
                        alpha=float(alpha),
                        effective_alpha=effective_alpha,
                        random_index=None,
                        layer=int(meta["layer"]),
                        output=output,
                    )
                )

    add_base_deltas(rows)

    manifest = {
        "started_at": started,
        "finished_at": now_iso(),
        "model_path": str(args.model_path),
        "sae_path": str(args.sae_path),
        "overlap_dir": str(args.overlap_dir),
        "output_dir": str(out_dir),
        "candidate_names": candidate_names,
        "external_direction_files": [str(path) for path in external_direction_files],
        "direction_source": args.direction_source,
        "alpha_scale": args.alpha_scale,
        "hidden_norm_reference": str(args.hidden_norm_reference) if args.hidden_norm_reference else None,
        "prompt_bank": str(args.prompt_bank) if args.prompt_bank else None,
        "top_features": args.top_features,
        "alphas": alphas,
        "random_controls": int(args.random_controls),
        "position": args.position,
        "max_prompts": args.max_prompts,
        "max_new_tokens": args.max_new_tokens,
        **qwen_budget_metadata(args.max_new_tokens),
        "generation_batch_size": int(args.generation_batch_size),
        "report_max_output_rows": int(args.report_max_output_rows),
        "do_sample": bool(args.do_sample),
        "temperature": args.temperature,
        "top_p": args.top_p,
        "seed": args.seed,
        "prompt_ids": prompt_ids,
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
    os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
    main()
