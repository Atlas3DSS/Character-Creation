#!/usr/bin/env python3
"""Qwen 27B 4-bit proxy generation sweep for SCOTUS legal-frame prompts.

This script is intentionally proxy-only. It uses OpenAI-compatible llama.cpp
servers for generation, so it cannot create activation-vector random controls.
Instead it builds a generation/null distribution from sampled completions and
neutral prompt-surface variants, then scores outputs with the same lightweight
frame rubric used by the hook-based poke runs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from qwen_eval_budget import (
    DEFAULT_COMPLETE_ANSWER_TOKENS,
    MIN_COMPLETE_ANSWER_TOKENS,
    enforce_complete_answer_budget,
    qwen_budget_metadata,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PROMPT_BANK = PROJECT_ROOT / "data" / "scotus" / "scotus_poke_prompts_v1.jsonl"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "sweep_v4"

FRAME_PATTERNS: dict[str, list[str]] = {
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

CONDITION_TEMPLATES: dict[str, str] = {
    "concise_judicial": "{prompt}",
    "majority_reasoning": (
        "Draft the reasoning as a concise Supreme Court majority-opinion paragraph. "
        "Do not mention any justice by name.\n\n{prompt}"
    ),
    "bench_memo": (
        "Draft a neutral bench-memo analysis for a judge. Keep the legal reasoning compact, "
        "direct, and doctrinal. Do not mention any justice by name.\n\n{prompt}"
    ),
}


@dataclass(frozen=True)
class Endpoint:
    name: str
    url: str
    model: str


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
class GenerationTask:
    task_id: str
    prompt: PromptSpec
    condition: str
    rendered_prompt: str
    sample_type: str
    control_index: int | None
    seed: int
    temperature: float


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON") from exc
    return rows


def write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def append_jsonl(path: Path, row: dict[str, Any], lock: threading.Lock) -> None:
    with lock:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
            handle.flush()


def parse_csv(raw: str) -> list[str]:
    return [part.strip() for part in raw.split(",") if part.strip()]


def parse_endpoints(raw: str) -> list[Endpoint]:
    endpoints: list[Endpoint] = []
    for idx, item in enumerate(parse_csv(raw)):
        parts = item.split("|")
        if len(parts) == 1:
            url = parts[0]
            model = ""
            name = f"endpoint_{idx}"
        elif len(parts) == 2:
            url, model = parts
            name = f"endpoint_{idx}"
        elif len(parts) == 3:
            name, url, model = parts
        else:
            raise ValueError(f"Invalid endpoint spec {item!r}; use name|url|model")
        endpoints.append(Endpoint(name=name.strip(), url=url.strip().rstrip("/"), model=model.strip()))
    if not endpoints:
        raise ValueError("At least one endpoint is required")
    return endpoints


def load_prompt_specs(prompt_bank: Path) -> list[PromptSpec]:
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
            available = ", ".join(f"{spec.prompt_id}:{spec.prompt_key}" for spec in specs)
            raise ValueError(f"Invalid prompt selector {token!r}. Available ids/keys: {available}")
        selected.extend(matches)
    return selected


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


def add_base_deltas(rows: list[dict[str, Any]]) -> None:
    base_by_key = {
        (row["prompt_id"], row["condition"]): row["frame_eval"]
        for row in rows
        if row["sample_type"] == "base"
    }
    for row in rows:
        base = base_by_key.get((row["prompt_id"], row["condition"]))
        if base is None:
            continue
        frame_eval = row["frame_eval"]
        for key in ("target_hits", "contrast_hits", "off_domain_hits", "total_frame_hits"):
            frame_eval[f"delta_{key}_vs_base"] = float(frame_eval[key] - base[key])


def mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def stdev(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    avg = mean(values)
    return float((sum((value - avg) ** 2 for value in values) / (len(values) - 1)) ** 0.5)


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * q)))
    return float(ordered[idx])


def aggregate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        key = (row["condition"], row["sample_type"], row["issue_area"])
        groups.setdefault(key, []).append(row)
    summaries: list[dict[str, Any]] = []
    for (condition, sample_type, issue_area), group in sorted(groups.items()):
        evals = [row["frame_eval"] for row in group]
        target_deltas = [float(item.get("delta_target_hits_vs_base", 0.0)) for item in evals]
        summaries.append(
            {
                "condition": condition,
                "sample_type": sample_type,
                "issue_area": issue_area,
                "n": len(group),
                "prompt_count": len({row["prompt_id"] for row in group}),
                "target_present_rate": mean([1.0 if item["target_present"] else 0.0 for item in evals]),
                "contrast_present_rate": mean([1.0 if item["contrast_present"] else 0.0 for item in evals]),
                "off_domain_present_rate": mean([1.0 if item["off_domain_present"] else 0.0 for item in evals]),
                "mean_target_hits": mean([float(item["target_hits"]) for item in evals]),
                "mean_contrast_hits": mean([float(item["contrast_hits"]) for item in evals]),
                "mean_off_domain_hits": mean([float(item["off_domain_hits"]) for item in evals]),
                "mean_delta_target_hits_vs_base": mean(target_deltas),
                "sd_delta_target_hits_vs_base": stdev(target_deltas),
                "p05_delta_target_hits_vs_base": percentile(target_deltas, 0.05),
                "p50_delta_target_hits_vs_base": percentile(target_deltas, 0.50),
                "p95_delta_target_hits_vs_base": percentile(target_deltas, 0.95),
            }
        )
    return summaries


def prompt_condition_nulls(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[int, str], list[dict[str, Any]]] = {}
    for row in rows:
        if row["sample_type"] == "random_control":
            groups.setdefault((int(row["prompt_id"]), str(row["condition"])), []).append(row)
    summaries: list[dict[str, Any]] = []
    for (prompt_id, condition), group in sorted(groups.items()):
        deltas = [float(row["frame_eval"].get("delta_target_hits_vs_base", 0.0)) for row in group]
        first = group[0]
        summaries.append(
            {
                "prompt_id": prompt_id,
                "prompt_key": first["prompt_key"],
                "issue_area": first["issue_area"],
                "condition": condition,
                "n": len(group),
                "mean_delta_target_hits_vs_base": mean(deltas),
                "sd_delta_target_hits_vs_base": stdev(deltas),
                "p05_delta_target_hits_vs_base": percentile(deltas, 0.05),
                "p50_delta_target_hits_vs_base": percentile(deltas, 0.50),
                "p95_delta_target_hits_vs_base": percentile(deltas, 0.95),
            }
        )
    return summaries


def format_frame_scores(scores: dict[str, int]) -> str:
    if not scores:
        return ""
    return ", ".join(f"{name}:{score}" for name, score in sorted(scores.items(), key=lambda item: (-item[1], item[0])))


def clean_snippet(text: str, max_chars: int = 500) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= max_chars:
        return text
    cut = text[: max_chars + 1]
    if " " in cut:
        cut = cut[: cut.rfind(" ")]
    return cut.rstrip() + "..."


def markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    def cell(value: Any) -> str:
        text = str(value)
        return text.replace("|", "\\|").replace("\n", " ")

    lines = [
        "| " + " | ".join(cell(header) for header in headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(cell(value) for value in row) + " |")
    return "\n".join(lines)


def stable_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def build_tasks(
    *,
    prompts: list[PromptSpec],
    conditions: list[str],
    random_controls: int,
    base_seed: int,
    base_temperature: float,
    random_temperature: float,
) -> list[GenerationTask]:
    tasks: list[GenerationTask] = []
    for prompt in prompts:
        for condition_index, condition in enumerate(conditions):
            if condition not in CONDITION_TEMPLATES:
                raise ValueError(f"Unknown condition {condition!r}; available: {sorted(CONDITION_TEMPLATES)}")
            rendered = CONDITION_TEMPLATES[condition].format(prompt=prompt.prompt)
            seed_prefix = base_seed + prompt.prompt_id * 100_000 + condition_index * 1_000
            base_key = f"{prompt.prompt_id}|{condition}|base|0|{seed_prefix}"
            tasks.append(
                GenerationTask(
                    task_id=stable_hash(base_key),
                    prompt=prompt,
                    condition=condition,
                    rendered_prompt=rendered,
                    sample_type="base",
                    control_index=None,
                    seed=seed_prefix,
                    temperature=base_temperature,
                )
            )
            for control_index in range(random_controls):
                seed = seed_prefix + control_index + 1
                key = f"{prompt.prompt_id}|{condition}|random_control|{control_index}|{seed}"
                tasks.append(
                    GenerationTask(
                        task_id=stable_hash(key),
                        prompt=prompt,
                        condition=condition,
                        rendered_prompt=rendered,
                        sample_type="random_control",
                        control_index=control_index,
                        seed=seed,
                        temperature=random_temperature,
                    )
                )
    return tasks


def call_chat_completion(
    *,
    endpoint: Endpoint,
    prompt: str,
    seed: int,
    temperature: float,
    top_p: float,
    max_tokens: int,
    timeout: float,
    max_retries: int,
) -> dict[str, Any]:
    url = endpoint.url
    if not url.endswith("/chat/completions"):
        url = url.rstrip("/") + "/v1/chat/completions"
    model = endpoint.model or endpoint.name
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "seed": seed,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    encoded = json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    last_error: str | None = None
    for attempt in range(max_retries + 1):
        request = urllib.request.Request(url, data=encoded, headers=headers, method="POST")
        try:
            started = time.monotonic()
            with urllib.request.urlopen(request, timeout=timeout) as response:
                raw = response.read().decode("utf-8")
            elapsed = time.monotonic() - started
            obj = json.loads(raw)
            choice = obj["choices"][0]
            message = choice.get("message") or {}
            content = str(message.get("content") or "").strip()
            reasoning_content = str(message.get("reasoning_content") or "").strip()
            used_reasoning_fallback = False
            if not content and reasoning_content:
                content = reasoning_content
                used_reasoning_fallback = True
            return {
                "text": content,
                "finish_reason": choice.get("finish_reason"),
                "usage": obj.get("usage", {}),
                "endpoint_name": endpoint.name,
                "endpoint_url": endpoint.url,
                "model": obj.get("model", model),
                "elapsed_seconds": elapsed,
                "used_reasoning_fallback": used_reasoning_fallback,
            }
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, json.JSONDecodeError, KeyError) as exc:
            last_error = repr(exc)
            if attempt >= max_retries:
                break
            time.sleep(min(30.0, 2.0 * (attempt + 1)))
    raise RuntimeError(f"{endpoint.name} failed after {max_retries + 1} attempts: {last_error}")


def task_to_row(
    *,
    task: GenerationTask,
    endpoint: Endpoint,
    output: dict[str, Any],
) -> dict[str, Any]:
    frame_scores = tag_frames(output["text"])
    row = {
        "task_id": task.task_id,
        "prompt_id": task.prompt.prompt_id,
        "prompt_key": task.prompt.prompt_key,
        "issue_area": task.prompt.issue_area,
        "prompt": task.prompt.prompt,
        "condition": task.condition,
        "rendered_prompt": task.rendered_prompt,
        "sample_type": task.sample_type,
        "control_index": task.control_index,
        "seed": task.seed,
        "temperature": task.temperature,
        "endpoint_assigned": endpoint.name,
        "frame_scores": frame_scores,
        "frame_eval": frame_eval_for_prompt(task.prompt, frame_scores),
        **output,
    }
    return row


def run_task(
    *,
    task: GenerationTask,
    endpoint: Endpoint,
    top_p: float,
    max_tokens: int,
    timeout: float,
    max_retries: int,
) -> dict[str, Any]:
    output = call_chat_completion(
        endpoint=endpoint,
        prompt=task.rendered_prompt,
        seed=task.seed,
        temperature=task.temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        timeout=timeout,
        max_retries=max_retries,
    )
    return task_to_row(task=task, endpoint=endpoint, output=output)


def build_blind_sample(
    rows: list[dict[str, Any]],
    *,
    out_dir: Path,
    sample_size: int,
    seed: int,
) -> None:
    if sample_size <= 0:
        return
    rng = random.Random(seed)
    candidates = list(rows)
    rng.shuffle(candidates)
    selected = candidates[: min(sample_size, len(candidates))]
    blind_rows: list[dict[str, Any]] = []
    key_rows: list[dict[str, Any]] = []
    for idx, row in enumerate(selected):
        blind_id = f"blind_{idx:04d}_{stable_hash(row['task_id'])[:8]}"
        blind_rows.append(
            {
                "blind_id": blind_id,
                "prompt_key": row["prompt_key"],
                "issue_area": row["issue_area"],
                "prompt": row["prompt"],
                "completion": row["text"],
                "frame_scores": row["frame_scores"],
                "review_fields": {
                    "legally_coherent_1_5": None,
                    "frame_matches_prompt_1_5": None,
                    "distinctive_judicial_voice_1_5": None,
                    "surface_template_artifact": None,
                    "notes": "",
                },
            }
        )
        key_rows.append(
            {
                "blind_id": blind_id,
                "task_id": row["task_id"],
                "condition": row["condition"],
                "sample_type": row["sample_type"],
                "control_index": row["control_index"],
                "seed": row["seed"],
                "endpoint_name": row["endpoint_name"],
                "model": row["model"],
            }
        )
    with (out_dir / "blind_review_sample.jsonl").open("w", encoding="utf-8") as handle:
        for row in blind_rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    with (out_dir / "blind_review_key.jsonl").open("w", encoding="utf-8") as handle:
        for row in key_rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def write_report(out_dir: Path, *, manifest: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    score_summary = aggregate_rows(rows)
    prompt_nulls = prompt_condition_nulls(rows)
    write_json(out_dir / "score_summary.json", score_summary)
    write_json(out_dir / "prompt_condition_nulls.json", prompt_nulls)
    report_rows = [
        [
            row["condition"],
            row["sample_type"],
            row["issue_area"],
            row["n"],
            row["prompt_count"],
            f"{row['target_present_rate']:.2f}",
            f"{row['mean_delta_target_hits_vs_base']:.2f}",
            f"{row['sd_delta_target_hits_vs_base']:.2f}",
            f"{row['p05_delta_target_hits_vs_base']:.1f}",
            f"{row['p50_delta_target_hits_vs_base']:.1f}",
            f"{row['p95_delta_target_hits_vs_base']:.1f}",
            f"{row['off_domain_present_rate']:.2f}",
        ]
        for row in score_summary
    ]
    null_rows = [
        [
            row["prompt_id"],
            row["prompt_key"],
            row["issue_area"],
            row["condition"],
            row["n"],
            f"{row['mean_delta_target_hits_vs_base']:.2f}",
            f"{row['sd_delta_target_hits_vs_base']:.2f}",
            f"{row['p05_delta_target_hits_vs_base']:.1f}",
            f"{row['p50_delta_target_hits_vs_base']:.1f}",
            f"{row['p95_delta_target_hits_vs_base']:.1f}",
        ]
        for row in prompt_nulls[:120]
    ]
    display_rows = rows[: int(manifest["report_max_output_rows"])]
    output_rows = [
        [
            row["prompt_id"],
            row["prompt_key"],
            row["issue_area"],
            row["condition"],
            row["sample_type"],
            "" if row["control_index"] is None else row["control_index"],
            row["endpoint_name"],
            format_frame_scores(row["frame_scores"]),
            f"{row['frame_eval'].get('delta_target_hits_vs_base', 0.0):.1f}",
            row.get("usage", {}).get("completion_tokens", ""),
            clean_snippet(row["text"]),
        ]
        for row in display_rows
    ]
    lines = [
        "# Qwen 27B Q4 SCOTUS Proxy Generation",
        "",
        "## Configuration",
        "",
        markdown_table(
            ["Field", "Value"],
            [
                ["Started", manifest["started_at"]],
                ["Finished", manifest["finished_at"]],
                ["Prompt bank", manifest["prompt_bank"]],
                ["Prompts", manifest["prompt_count"]],
                ["Conditions", ", ".join(manifest["conditions"])],
                ["Random controls per prompt-condition", manifest["random_controls"]],
                ["Rows", len(rows)],
                ["Endpoints", ", ".join(endpoint["name"] for endpoint in manifest["endpoints"])],
                ["Max tokens", manifest["max_tokens"]],
                ["Short-budget smoke", manifest["short_answer_budget"]],
                ["Random temperature", manifest["random_temperature"]],
                ["Top p", manifest["top_p"]],
                ["Proxy-only", "yes; llama.cpp generation has no activation hooks"],
            ],
        ),
        "",
        "## Method Note",
        "",
        "This is a 4-bit generation proxy run, not decision-grade causal steering evidence. "
        "The random controls are sampled completions and neutral prompt-surface variants through the running "
        "llama.cpp Qwen3.6-27B Q4 servers. They estimate the legal-frame null distribution available from "
        "generation variability, but they are not same-norm activation-vector controls.",
        "",
        "## Aggregate Frame Scores",
        "",
        "Deltas are relative to the deterministic base completion for the same prompt and condition.",
        "",
        markdown_table(
            [
                "Condition",
                "Sample",
                "Issue",
                "N",
                "Prompts",
                "Target present",
                "Mean target delta",
                "SD target delta",
                "P05",
                "P50",
                "P95",
                "Off-domain present",
            ],
            report_rows,
        ),
        "",
        "## Prompt-Condition Nulls",
        "",
        f"Showing {min(120, len(prompt_nulls))} of {len(prompt_nulls)} prompt-condition null rows.",
        "",
        markdown_table(
            [
                "Prompt",
                "Key",
                "Issue",
                "Condition",
                "N",
                "Mean target delta",
                "SD",
                "P05",
                "P50",
                "P95",
            ],
            null_rows,
        ),
        "",
        "## Output Sample",
        "",
        f"Showing {len(display_rows)} of {len(rows)} rows. Full rows are in `generations.jsonl`.",
        "",
        markdown_table(
            [
                "Prompt",
                "Key",
                "Issue",
                "Condition",
                "Sample",
                "Control",
                "Endpoint",
                "Frame tags",
                "Target delta",
                "Tokens",
                "Completion",
            ],
            output_rows,
        ),
        "",
        "## Blind Review",
        "",
        "- `blind_review_sample.jsonl` contains anonymized completions and review fields.",
        "- `blind_review_key.jsonl` maps blind IDs back to condition, control index, seed, endpoint, and model.",
        "- Suggested use: review the blind sample before looking at the key, then compare human ratings to frame-rubric scores.",
        "",
    ]
    (out_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Qwen 27B Q4 SCOTUS proxy generation over legal prompts.")
    parser.add_argument("--prompt-bank", type=Path, default=DEFAULT_PROMPT_BANK)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--endpoints",
        default=(
            "q4_3090|http://127.0.0.1:8080|qwen3.6-27b-q4-3090,"
            "q4_4090|http://127.0.0.1:8081|qwen3.6-27b-q4-4090"
        ),
        help="Comma-separated endpoint specs as name|base_url|model.",
    )
    parser.add_argument("--conditions", default="concise_judicial,majority_reasoning,bench_memo")
    parser.add_argument("--prompt-ids", default="")
    parser.add_argument("--max-prompts", type=int, default=20)
    parser.add_argument("--random-controls", type=int, default=50)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_COMPLETE_ANSWER_TOKENS)
    parser.add_argument(
        "--allow-short-max-tokens",
        action="store_true",
        help=(
            f"Permit max-token budgets below {MIN_COMPLETE_ANSWER_TOKENS}. "
            "Short-budget Qwen legal generations are smoke/debug only."
        ),
    )
    parser.add_argument("--base-temperature", type=float, default=0.2)
    parser.add_argument("--random-temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=4242)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--workers-per-endpoint", type=int, default=1)
    parser.add_argument("--blind-sample-size", type=int, default=120)
    parser.add_argument("--report-max-output-rows", type=int, default=80)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    enforce_complete_answer_budget(
        args.max_tokens,
        allow_short=args.allow_short_max_tokens,
        label="max_tokens",
        purpose="SCOTUS 4-bit proxy generation run",
        opt_in_flag="--allow-short-max-tokens",
    )
    started = now_iso()
    endpoints = parse_endpoints(args.endpoints)
    conditions = parse_csv(args.conditions)
    all_prompts = load_prompt_specs(args.prompt_bank)
    prompts = select_prompt_specs(all_prompts, args.prompt_ids, args.max_prompts)
    tasks = build_tasks(
        prompts=prompts,
        conditions=conditions,
        random_controls=args.random_controls,
        base_seed=args.seed,
        base_temperature=args.base_temperature,
        random_temperature=args.random_temperature,
    )
    out_dir = args.output_root / f"scotus_qwen4bit_proxy_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)
    generations_path = out_dir / "generations.jsonl"
    lock = threading.Lock()

    manifest: dict[str, Any] = {
        "started_at": started,
        "prompt_bank": str(args.prompt_bank),
        "prompt_count": len(prompts),
        "conditions": conditions,
        "random_controls": args.random_controls,
        "max_tokens": args.max_tokens,
        **qwen_budget_metadata(args.max_tokens),
        "base_temperature": args.base_temperature,
        "random_temperature": args.random_temperature,
        "top_p": args.top_p,
        "seed": args.seed,
        "endpoints": [endpoint.__dict__ for endpoint in endpoints],
        "task_count": len(tasks),
        "workers_per_endpoint": args.workers_per_endpoint,
        "blind_sample_size": args.blind_sample_size,
        "report_max_output_rows": args.report_max_output_rows,
        "proxy_only": True,
        "proxy_note": "llama.cpp generation servers do not expose activation hooks; random controls are generation-sampling controls.",
    }
    write_json(out_dir / "manifest.json", manifest)
    print(f"Writing proxy run to {out_dir}", flush=True)
    print(f"Tasks: {len(tasks)} across {len(endpoints)} endpoints", flush=True)

    rows: list[dict[str, Any]] = []
    endpoint_slots: list[Endpoint] = []
    for endpoint in endpoints:
        endpoint_slots.extend([endpoint] * max(1, args.workers_per_endpoint))
    max_workers = max(1, len(endpoint_slots))

    completed = 0
    started_monotonic = time.monotonic()
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        for idx, task in enumerate(tasks):
            endpoint = endpoint_slots[idx % len(endpoint_slots)]
            future = executor.submit(
                run_task,
                task=task,
                endpoint=endpoint,
                top_p=args.top_p,
                max_tokens=args.max_tokens,
                timeout=args.timeout,
                max_retries=args.max_retries,
            )
            futures[future] = task
        for future in as_completed(futures):
            task = futures[future]
            try:
                row = future.result()
            except Exception as exc:
                error_row = {
                    "task_id": task.task_id,
                    "prompt_id": task.prompt.prompt_id,
                    "prompt_key": task.prompt.prompt_key,
                    "issue_area": task.prompt.issue_area,
                    "condition": task.condition,
                    "sample_type": task.sample_type,
                    "control_index": task.control_index,
                    "seed": task.seed,
                    "error": repr(exc),
                }
                append_jsonl(out_dir / "errors.jsonl", error_row, lock)
                print(f"ERROR {task.task_id} {task.prompt.prompt_key} {task.condition}: {exc!r}", file=sys.stderr, flush=True)
                continue
            append_jsonl(generations_path, row, lock)
            rows.append(row)
            completed += 1
            if completed % 25 == 0 or completed == len(tasks):
                elapsed = time.monotonic() - started_monotonic
                rate = completed / elapsed if elapsed > 0 else 0.0
                print(f"completed {completed}/{len(tasks)} rows ({rate:.2f} rows/s)", flush=True)

    add_base_deltas(rows)
    with generations_path.open("w", encoding="utf-8") as handle:
        for row in sorted(rows, key=lambda item: (item["prompt_id"], item["condition"], item["sample_type"], item.get("control_index") or -1)):
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    build_blind_sample(rows, out_dir=out_dir, sample_size=args.blind_sample_size, seed=args.seed + 99)
    manifest["finished_at"] = now_iso()
    manifest["completed_rows"] = len(rows)
    manifest["error_rows"] = sum(1 for _ in (out_dir / "errors.jsonl").open("r", encoding="utf-8")) if (out_dir / "errors.jsonl").exists() else 0
    write_json(out_dir / "manifest.json", manifest)
    write_report(out_dir, manifest=manifest, rows=sorted(rows, key=lambda item: (item["prompt_id"], item["condition"], item["sample_type"], item.get("control_index") or -1)))
    print(f"Report: {out_dir / 'report.md'}", flush=True)


if __name__ == "__main__":
    main()
