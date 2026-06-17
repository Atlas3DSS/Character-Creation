#!/usr/bin/env python3
"""Evaluate cached SCOTUS activation features under case-component resplits.

The split plans are chosen from metadata only: case-connected components,
row-count balance, and optional field coverage. Probe metrics are computed
after the split plans are fixed, so test scores are never used to choose a
headline split.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np

from audit_scotus_slice_split_feasibility import Component, connected_components
from probe_scotus_style import (
    apply_diagnostic_mode,
    balanced_accuracy_ci,
    evaluate_text_baseline,
    load_feature_artifacts,
    markdown_table,
    predict_metrics,
    save_probe_direction,
    select_probe,
    stress_tests,
    write_json,
    write_jsonl,
    write_report as write_probe_report,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "sweep_v4"
SPLITS = ("train", "dev", "test")


@dataclass(frozen=True)
class ComponentProfile:
    component_id: str
    row_indices: tuple[int, ...]
    n_rows: int
    n_pairs: int
    n_cases: int
    label_counts: dict[str, int]
    justice_counts: dict[str, int]
    field_counts: dict[str, dict[str, int]]
    case_ids: tuple[str, ...]
    pair_ids: tuple[str, ...]


@dataclass(frozen=True)
class SplitPlan:
    plan_id: str
    rank: int
    score: float
    assignment: dict[str, str]
    split_rows: dict[str, int]
    split_label_counts: dict[str, dict[str, int]]
    split_field_counts: dict[str, dict[str, dict[str, int]]]
    components: list[dict[str, Any]]


def now_stamp() -> str:
    return datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


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


def parse_csv(raw: str) -> list[str]:
    return [part.strip() for part in raw.split(",") if part.strip()]


def counter_payload(counter: Counter[Any]) -> dict[str, int]:
    return {str(key): int(value) for key, value in sorted(counter.items(), key=lambda item: str(item[0]))}


def make_component_profiles(
    rows: list[dict[str, Any]],
    components: list[Component],
    *,
    balance_fields: list[str],
) -> list[ComponentProfile]:
    profiles: list[ComponentProfile] = []
    for component in components:
        component_rows = [rows[idx] for idx in component.row_indices]
        label_counts = Counter(int(row["label"]) for row in component_rows)
        justice_counts = Counter(str(row.get("justice") or "unknown") for row in component_rows)
        field_counts: dict[str, dict[str, int]] = {}
        for field in balance_fields:
            field_counts[field] = counter_payload(Counter(str(row.get(field) or "unknown") for row in component_rows))
        profiles.append(
            ComponentProfile(
                component_id=component.component_id,
                row_indices=component.row_indices,
                n_rows=len(component.row_indices),
                n_pairs=len(component.pair_ids),
                n_cases=len(component.case_ids),
                label_counts=counter_payload(label_counts),
                justice_counts=counter_payload(justice_counts),
                field_counts=field_counts,
                case_ids=component.case_ids,
                pair_ids=component.pair_ids,
            )
        )
    return sorted(profiles, key=lambda item: item.component_id)


def add_nested_counts(target: dict[str, int], source: dict[str, int]) -> None:
    for key, value in source.items():
        target[key] = target.get(key, 0) + int(value)


def summarize_assignment(
    profiles: list[ComponentProfile],
    assignment: dict[str, str],
    *,
    balance_fields: list[str],
) -> tuple[dict[str, int], dict[str, dict[str, int]], dict[str, dict[str, dict[str, int]]]]:
    split_rows = {split: 0 for split in SPLITS}
    split_label_counts = {split: {} for split in SPLITS}
    split_field_counts = {split: {field: {} for field in balance_fields} for split in SPLITS}
    for profile in profiles:
        split = assignment[profile.component_id]
        split_rows[split] += profile.n_rows
        add_nested_counts(split_label_counts[split], profile.label_counts)
        for field in balance_fields:
            add_nested_counts(split_field_counts[split][field], profile.field_counts.get(field, {}))
    return split_rows, split_label_counts, split_field_counts


def normalized_distribution(counts: dict[str, int]) -> dict[str, float]:
    total = sum(counts.values())
    if total <= 0:
        return {}
    return {key: value / total for key, value in counts.items()}


def l1_distribution_distance(a: dict[str, int], b: dict[str, int]) -> float:
    a_dist = normalized_distribution(a)
    b_dist = normalized_distribution(b)
    keys = set(a_dist) | set(b_dist)
    return sum(abs(a_dist.get(key, 0.0) - b_dist.get(key, 0.0)) for key in keys)


def plan_score(
    profiles: list[ComponentProfile],
    assignment: dict[str, str],
    *,
    target_fracs: dict[str, float],
    balance_fields: list[str],
    field_weight: float,
) -> tuple[float, dict[str, int], dict[str, dict[str, int]], dict[str, dict[str, dict[str, int]]]]:
    total_rows = sum(profile.n_rows for profile in profiles)
    split_rows, split_label_counts, split_field_counts = summarize_assignment(
        profiles,
        assignment,
        balance_fields=balance_fields,
    )
    size_score = 0.0
    for split in SPLITS:
        target = target_fracs[split] * total_rows
        weight = 1.5 if split in {"dev", "test"} else 1.0
        size_score += weight * ((split_rows[split] - target) / total_rows) ** 2

    field_score = 0.0
    for field in balance_fields:
        global_counts: dict[str, int] = {}
        for profile in profiles:
            add_nested_counts(global_counts, profile.field_counts.get(field, {}))
        for split in SPLITS:
            weight = 1.5 if split in {"dev", "test"} else 1.0
            field_score += weight * l1_distribution_distance(split_field_counts[split][field], global_counts)

    dev_test_gap = abs(split_rows["dev"] - split_rows["test"]) / total_rows
    score = size_score + (field_weight * field_score) + (0.5 * dev_test_gap)
    return score, split_rows, split_label_counts, split_field_counts


def passes_hard_constraints(
    profiles: list[ComponentProfile],
    assignment: dict[str, str],
    *,
    split_rows: dict[str, int],
    split_label_counts: dict[str, dict[str, int]],
    split_field_counts: dict[str, dict[str, dict[str, int]]],
    min_rows: dict[str, int],
    min_label_per_split: int,
    require_eval_field_coverage: list[str],
) -> bool:
    component_counts = Counter(assignment.values())
    if any(component_counts.get(split, 0) == 0 for split in SPLITS):
        return False
    if any(split_rows[split] < min_rows.get(split, 0) for split in SPLITS):
        return False
    for split in SPLITS:
        labels = split_label_counts[split]
        if min(int(labels.get("0", 0)), int(labels.get("1", 0))) < min_label_per_split:
            return False
    for field in require_eval_field_coverage:
        global_values = {
            value
            for profile in profiles
            for value, count in profile.field_counts.get(field, {}).items()
            if int(count) > 0
        }
        for split in ("dev", "test"):
            split_values = {
                value
                for value, count in split_field_counts[split].get(field, {}).items()
                if int(count) > 0
            }
            if not global_values.issubset(split_values):
                return False
    return True


def build_split_plans(
    profiles: list[ComponentProfile],
    *,
    n_plans: int,
    target_fracs: dict[str, float],
    min_rows: dict[str, int],
    min_label_per_split: int,
    balance_fields: list[str],
    require_eval_field_coverage: list[str],
    field_weight: float,
) -> list[SplitPlan]:
    candidates: list[tuple[float, str, dict[str, str], dict[str, int], dict[str, dict[str, int]], dict[str, dict[str, dict[str, int]]]]] = []
    component_ids = [profile.component_id for profile in profiles]
    for split_choices in product(SPLITS, repeat=len(component_ids)):
        assignment = dict(zip(component_ids, split_choices, strict=True))
        score, split_rows, split_label_counts, split_field_counts = plan_score(
            profiles,
            assignment,
            target_fracs=target_fracs,
            balance_fields=balance_fields,
            field_weight=field_weight,
        )
        if not passes_hard_constraints(
            profiles,
            assignment,
            split_rows=split_rows,
            split_label_counts=split_label_counts,
            split_field_counts=split_field_counts,
            min_rows=min_rows,
            min_label_per_split=min_label_per_split,
            require_eval_field_coverage=require_eval_field_coverage,
        ):
            continue
        assignment_key = "|".join(f"{component_id}:{assignment[component_id]}" for component_id in component_ids)
        candidates.append((score, assignment_key, assignment, split_rows, split_label_counts, split_field_counts))
    candidates.sort(key=lambda item: (item[0], item[1]))
    if not candidates:
        raise RuntimeError("No split plans passed the hard constraints")

    plans: list[SplitPlan] = []
    for rank, (score, _key, assignment, split_rows, split_label_counts, split_field_counts) in enumerate(
        candidates[:n_plans]
    ):
        components_payload = []
        for profile in profiles:
            components_payload.append(
                {
                    "component_id": profile.component_id,
                    "assigned_split": assignment[profile.component_id],
                    "n_rows": profile.n_rows,
                    "n_pairs": profile.n_pairs,
                    "n_cases": profile.n_cases,
                    "label_counts": profile.label_counts,
                    "justice_counts": profile.justice_counts,
                    "field_counts": profile.field_counts,
                    "case_ids": list(profile.case_ids),
                    "pair_ids": list(profile.pair_ids),
                }
            )
        plans.append(
            SplitPlan(
                plan_id=f"split_{rank:02d}",
                rank=rank,
                score=float(score),
                assignment=assignment,
                split_rows={split: int(rows) for split, rows in split_rows.items()},
                split_label_counts=split_label_counts,
                split_field_counts=split_field_counts,
                components=components_payload,
            )
        )
    return plans


def split_plans_from_json(path: Path, profiles: list[ComponentProfile]) -> list[SplitPlan]:
    payload = read_json(path)
    raw_plans = payload.get("plans", payload if isinstance(payload, list) else [])
    if not isinstance(raw_plans, list):
        raise ValueError(f"{path} does not contain a plan list")
    component_ids = {profile.component_id for profile in profiles}
    plans: list[SplitPlan] = []
    balance_fields = sorted({field for profile in profiles for field in profile.field_counts})
    for rank, raw in enumerate(raw_plans):
        assignment = {str(key): str(value) for key, value in raw["assignment"].items()}
        if set(assignment) != component_ids:
            raise ValueError(f"Plan {rank} component ids do not match current run")
        invalid_splits = sorted({split for split in assignment.values() if split not in SPLITS})
        if invalid_splits:
            raise ValueError(f"Plan {rank} has invalid splits: {invalid_splits}")
        score, split_rows, split_label_counts, split_field_counts = plan_score(
            profiles,
            assignment,
            target_fracs={"train": 0.70, "dev": 0.15, "test": 0.15},
            balance_fields=balance_fields,
            field_weight=0.0,
        )
        components_payload = []
        for profile in profiles:
            components_payload.append(
                {
                    "component_id": profile.component_id,
                    "assigned_split": assignment[profile.component_id],
                    "n_rows": profile.n_rows,
                    "n_pairs": profile.n_pairs,
                    "n_cases": profile.n_cases,
                    "label_counts": profile.label_counts,
                    "justice_counts": profile.justice_counts,
                    "field_counts": profile.field_counts,
                    "case_ids": list(profile.case_ids),
                    "pair_ids": list(profile.pair_ids),
                }
            )
        plans.append(
            SplitPlan(
                plan_id=str(raw.get("plan_id", f"split_{rank:02d}")),
                rank=int(raw.get("rank", rank)),
                score=float(raw.get("score", score)),
                assignment=assignment,
                split_rows={split: int(rows) for split, rows in split_rows.items()},
                split_label_counts=split_label_counts,
                split_field_counts=split_field_counts,
                components=components_payload,
            )
        )
    return plans


def plan_to_json(plan: SplitPlan) -> dict[str, Any]:
    return {
        "plan_id": plan.plan_id,
        "rank": plan.rank,
        "score": plan.score,
        "assignment": plan.assignment,
        "split_rows": plan.split_rows,
        "split_label_counts": plan.split_label_counts,
        "split_field_counts": plan.split_field_counts,
        "components": plan.components,
    }


def assign_splits(
    rows: list[dict[str, Any]],
    profiles: list[ComponentProfile],
    plan: SplitPlan,
) -> list[dict[str, Any]]:
    updated = [dict(row) for row in rows]
    for profile in profiles:
        split = plan.assignment[profile.component_id]
        for idx in profile.row_indices:
            updated[idx]["original_split"] = rows[idx].get("split")
            updated[idx]["split"] = split
            updated[idx]["resplit_component_id"] = profile.component_id
            updated[idx]["resplit_plan_id"] = plan.plan_id
    return updated


def symlink_features(source_dir: Path, output_dir: Path) -> None:
    target = source_dir / "features.npz"
    link = output_dir / "features.npz"
    if link.exists() or link.is_symlink():
        link.unlink()
    rel_target = os.path.relpath(target, start=output_dir)
    link.symlink_to(rel_target)


def split_counts_table(rows: list[dict[str, Any]]) -> dict[str, Any]:
    split_justice = Counter((str(row["split"]), str(row.get("justice") or "unknown")) for row in rows)
    split_label = Counter((str(row["split"]), str(int(row["label"]))) for row in rows)
    split_decade = Counter((str(row["split"]), str(row.get("decade") or "unknown")) for row in rows)
    return {
        "split_justice": [
            {"split": split, "justice": justice, "n": int(count)}
            for (split, justice), count in sorted(split_justice.items())
        ],
        "split_label": [
            {"split": split, "label": label, "n": int(count)}
            for (split, label), count in sorted(split_label.items())
        ],
        "split_decade": [
            {"split": split, "decade": decade, "n": int(count)}
            for (split, decade), count in sorted(split_decade.items())
        ],
    }


def write_plan_report(path: Path, *, root_manifest: dict[str, Any], plans: list[SplitPlan]) -> None:
    rows = [
        [
            plan.plan_id,
            f"{plan.score:.6f}",
            plan.split_rows["train"],
            plan.split_rows["dev"],
            plan.split_rows["test"],
            plan.split_field_counts.get("dev", {}).get("decade", {}),
            plan.split_field_counts.get("test", {}).get("decade", {}),
        ]
        for plan in plans
    ]
    lines = [
        "# Cached SCOTUS Component Resplit Plan",
        "",
        "## Purpose",
        "",
        "These split plans were selected from cached metadata only: case-connected components, split-size balance, label counts, and requested field coverage. No probe score is used to choose the plans.",
        "",
        "## Inputs",
        "",
        f"- Feature source: `{root_manifest['features_dir']}`",
        f"- Diagnostic mode: `{root_manifest['diagnostic_mode']}`",
        f"- Balance fields: `{', '.join(root_manifest['balance_fields']) or 'none'}`",
        f"- Required eval coverage fields: `{', '.join(root_manifest['require_eval_field_coverage']) or 'none'}`",
        "",
        "## Plans",
        "",
        markdown_table(
            ["Plan", "Metadata score", "Train rows", "Dev rows", "Test rows", "Dev decade counts", "Test decade counts"],
            rows,
        ),
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_aggregate_report(path: Path, *, payload: dict[str, Any]) -> None:
    result_rows = []
    for result in payload["results"]:
        best = result["best_probe"]
        split_metrics = result["split_metrics"]
        text = result["rendered_prompt_tfidf_baseline"]
        ci = result["test_balanced_accuracy_ci_95"]
        result_rows.append(
            [
                result["plan_id"],
                f"{result['plan_score']:.6f}",
                best["region"],
                best["layer"],
                best["C"],
                f"{best['dev_metrics']['balanced_accuracy']:.3f}",
                f"{split_metrics['test']['balanced_accuracy']:.3f}",
                f"{ci['low']:.3f}-{ci['high']:.3f}",
                f"{text['dev']['balanced_accuracy']:.3f}",
                f"{text['test']['balanced_accuracy']:.3f}",
            ]
        )
    test_scores = [float(result["split_metrics"]["test"]["balanced_accuracy"]) for result in payload["results"]]
    dev_scores = [float(result["best_probe"]["dev_metrics"]["balanced_accuracy"]) for result in payload["results"]]
    lines = [
        "# Cached SCOTUS Component Resplit Results",
        "",
        "## Selection Rule",
        "",
        "Split plans were ranked before probe fitting using only case-component balance and requested metadata coverage. The primary split is `split_00`, the best metadata-balanced plan, not the best test-scoring plan.",
        "",
        "## Aggregate Read",
        "",
        f"- Diagnostic mode: `{payload['diagnostic_mode']}`",
        f"- Source run: `{payload['features_dir']}`",
        f"- Plans evaluated: `{len(payload['results'])}`",
        f"- Median dev balanced accuracy: `{np.median(dev_scores):.3f}`",
        f"- Median test balanced accuracy: `{np.median(test_scores):.3f}`",
        f"- Test balanced accuracy range: `{min(test_scores):.3f}` to `{max(test_scores):.3f}`",
        "",
        "## Plan Results",
        "",
        markdown_table(
            [
                "Plan",
                "Metadata score",
                "Region",
                "Layer",
                "C",
                "Dev BA",
                "Test BA",
                "Test CI",
                "Text dev BA",
                "Text test BA",
            ],
            result_rows,
        ),
        "",
        "## Decision Note",
        "",
    ]
    if payload["diagnostic_mode"] == "label_shuffle":
        lines.append("This is a label-shuffle null; use it only to calibrate how much performance survives destroyed labels.")
    else:
        lines.append(
            "This is still correlational evidence. Promote only if normal resplits beat text and label-shuffle controls, then run prompt-ablation and causal patch/steer tests."
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def evaluate_one_plan(
    *,
    plan: SplitPlan,
    rank: int,
    root_dir: Path,
    source_dir: Path,
    source_manifest: dict[str, Any],
    source_examples: list[dict[str, Any]],
    extracted: dict[str, Any],
    profiles: list[ComponentProfile],
    diagnostic_mode: str,
    seed: int,
    c_grid: list[float],
    classifier_solver: str,
    classifier_max_iter: int,
    classifier_tol: float,
    stress_min_eval_per_label: int,
) -> dict[str, Any]:
    plan_dir = root_dir / plan.plan_id
    plan_dir.mkdir(parents=True, exist_ok=True)
    symlink_features(source_dir, plan_dir)

    examples = assign_splits(source_examples, profiles, plan)
    meta_rows = assign_splits(extracted["meta_rows"], profiles, plan)
    if diagnostic_mode == "label_shuffle":
        shuffle_seed = seed + rank
        examples = apply_diagnostic_mode(examples, "label_shuffle", shuffle_seed)
        meta_rows = apply_diagnostic_mode(meta_rows, "label_shuffle", shuffle_seed)
    labels = np.array([int(row["label"]) for row in meta_rows], dtype=np.int64)
    prompt_template = str(source_manifest.get("prompt_template", "normal"))
    use_chat_template = bool(source_manifest.get("use_chat_template", True))
    positive_justice = str(source_manifest.get("positive_justice", "positive"))
    classifier = {
        "description": (
            "balanced logistic regression "
            f"(solver={classifier_solver}, max_iter={classifier_max_iter}, tol={classifier_tol})"
        ),
        "solver": classifier_solver,
        "max_iter": classifier_max_iter,
        "tol": classifier_tol,
        "test_diagnostic_refit": False,
    }
    manifest = dict(source_manifest)
    manifest.update(
        {
            "started_at": now_iso(),
            "features_source_dir": str(source_dir),
            "output_dir": str(plan_dir),
            "diagnostic_mode": diagnostic_mode,
            "resplit_plan": plan_to_json(plan),
            "resplit_rank": rank,
            "resplit_component_selection": "metadata_only",
            "prompt_template": prompt_template,
            "use_chat_template": use_chat_template,
            "c_grid": c_grid,
            "classifier": classifier,
        }
    )

    write_jsonl(plan_dir / "probe_examples.jsonl", examples)
    write_jsonl(plan_dir / "feature_meta.jsonl", meta_rows)
    write_json(plan_dir / "component_assignment.json", plan_to_json(plan))
    write_json(plan_dir / "split_counts.json", split_counts_table(examples))

    text_baseline = evaluate_text_baseline(examples, template_variant=prompt_template)
    probe = select_probe(
        extracted["regions"],
        meta_rows,
        labels,
        c_grid,
        classifier_solver=classifier_solver,
        classifier_max_iter=classifier_max_iter,
        classifier_tol=classifier_tol,
        test_diagnostic_refit=False,
    )
    for split, rows in probe["predictions"].items():
        write_jsonl(plan_dir / f"{split}_predictions.jsonl", rows)
    write_jsonl(plan_dir / "layer_region_search.jsonl", probe["searches"])
    stress = stress_tests(
        extracted["regions"],
        meta_rows,
        labels,
        probe["best"],
        min_eval_per_label=stress_min_eval_per_label,
        classifier_solver=classifier_solver,
        classifier_max_iter=classifier_max_iter,
        classifier_tol=classifier_tol,
    )
    save_probe_direction(plan_dir / "best_probe_direction.npz", probe["final_clf"], probe["best"], positive_justice)
    manifest["finished_at"] = now_iso()
    manifest["layers"] = extracted["layers"]
    manifest["best_probe"] = probe["best"]
    manifest["split_metrics"] = probe["split_metrics"]
    manifest["test_balanced_accuracy_ci_95"] = probe["test_balanced_accuracy_ci_95"]
    manifest["search_distribution"] = probe["search_distribution"]
    manifest["rendered_prompt_tfidf_baseline"] = text_baseline
    manifest["stress_tests"] = stress

    summary = {
        "plan_id": plan.plan_id,
        "plan_score": plan.score,
        "diagnostic_mode": diagnostic_mode,
        "output_dir": str(plan_dir),
        "best_probe": probe["best"],
        "split_metrics": probe["split_metrics"],
        "test_balanced_accuracy_ci_95": probe["test_balanced_accuracy_ci_95"],
        "search_distribution": probe["search_distribution"],
        "rendered_prompt_tfidf_baseline": text_baseline,
        "stress_tests": stress,
        "split_counts": split_counts_table(examples),
    }
    write_json(
        plan_dir / "summary.json",
        {
            **summary,
            "searches_top20": probe["searches"][:20],
        },
    )
    write_probe_report(
        plan_dir / "report.md",
        manifest=manifest,
        examples=examples,
        probe=probe,
        stress=stress,
        text_baseline=text_baseline,
    )
    write_json(plan_dir / "manifest.json", manifest)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--tag", default="scotus_cached_component_resplit")
    parser.add_argument(
        "--diagnostic-mode",
        choices=["normal", "excerpt_removed", "neutral_filler", "template_variant", "plain_prompt", "label_shuffle"],
        default="normal",
    )
    parser.add_argument("--split-plan-json", type=Path, default=None)
    parser.add_argument("--n-plans", type=int, default=5)
    parser.add_argument("--target-fracs", default="train:0.70,dev:0.15,test:0.15")
    parser.add_argument("--min-rows", default="train:300,dev:80,test:80")
    parser.add_argument("--min-label-per-split", type=int, default=30)
    parser.add_argument("--balance-fields", default="decade")
    parser.add_argument("--require-eval-field-coverage", default="")
    parser.add_argument("--field-weight", type=float, default=0.08)
    parser.add_argument("--c-grid", default="0.001,0.003,0.01,0.03,0.1")
    parser.add_argument("--classifier-solver", choices=["lbfgs", "liblinear", "saga", "sgd"], default="lbfgs")
    parser.add_argument("--classifier-max-iter", type=int, default=1000)
    parser.add_argument("--classifier-tol", type=float, default=1e-3)
    parser.add_argument("--stress-min-eval-per-label", type=int, default=5)
    parser.add_argument("--seed", type=int, default=17)
    return parser.parse_args()


def parse_split_values(raw: str, *, value_type: type) -> dict[str, Any]:
    values: dict[str, Any] = {}
    for part in parse_csv(raw):
        if ":" not in part:
            raise ValueError(f"Expected split:value entry, got {part!r}")
        split, value = part.split(":", 1)
        split = split.strip()
        if split not in SPLITS:
            raise ValueError(f"Unknown split {split!r}")
        values[split] = value_type(value)
    missing = sorted(set(SPLITS) - set(values))
    if missing:
        raise ValueError(f"Missing split values for {missing}")
    return values


def main() -> None:
    args = parse_args()
    source_dir = args.features_dir
    source_manifest = read_json(source_dir / "manifest.json")
    source_mode = str(source_manifest.get("diagnostic_mode", "normal"))
    if args.diagnostic_mode != source_mode and args.diagnostic_mode != "label_shuffle":
        raise ValueError(
            "Cached component resplits can only reuse features from their original diagnostic mode "
            "or run a label-shuffle null. Recapture features with probe_scotus_style.py before "
            f"resplitting {args.diagnostic_mode!r}; source mode is {source_mode!r}."
        )
    source_examples = read_jsonl(source_dir / "probe_examples.jsonl")
    extracted = load_feature_artifacts(source_dir)
    if len(source_examples) != len(extracted["meta_rows"]):
        raise RuntimeError("probe_examples.jsonl and feature_meta.jsonl row counts do not match")

    balance_fields = parse_csv(args.balance_fields)
    require_eval_field_coverage = parse_csv(args.require_eval_field_coverage)
    for field in require_eval_field_coverage:
        if field not in balance_fields:
            balance_fields.append(field)
    components = connected_components(extracted["meta_rows"])
    profiles = make_component_profiles(extracted["meta_rows"], components, balance_fields=balance_fields)
    if args.split_plan_json:
        plans = split_plans_from_json(args.split_plan_json, profiles)[: args.n_plans]
    else:
        target_fracs = parse_split_values(args.target_fracs, value_type=float)
        min_rows = parse_split_values(args.min_rows, value_type=int)
        plans = build_split_plans(
            profiles,
            n_plans=args.n_plans,
            target_fracs=target_fracs,
            min_rows=min_rows,
            min_label_per_split=args.min_label_per_split,
            balance_fields=balance_fields,
            require_eval_field_coverage=require_eval_field_coverage,
            field_weight=args.field_weight,
        )

    c_grid = [float(part) for part in parse_csv(args.c_grid)]
    root_dir = args.output_root / f"{args.tag}_{args.diagnostic_mode}_component_resplits_{now_stamp()}"
    root_dir.mkdir(parents=True, exist_ok=True)
    root_manifest = {
        "started_at": now_iso(),
        "features_dir": str(source_dir),
        "output_dir": str(root_dir),
        "diagnostic_mode": args.diagnostic_mode,
        "n_plans": len(plans),
        "balance_fields": balance_fields,
        "require_eval_field_coverage": require_eval_field_coverage,
        "c_grid": c_grid,
        "classifier": {
            "solver": args.classifier_solver,
            "max_iter": args.classifier_max_iter,
            "tol": args.classifier_tol,
        },
        "plan_selection": "metadata_only",
        "plans": [plan_to_json(plan) for plan in plans],
    }
    write_json(root_dir / "split_plans.json", {"plans": [plan_to_json(plan) for plan in plans]})
    write_plan_report(root_dir / "split_plan_report.md", root_manifest=root_manifest, plans=plans)

    results: list[dict[str, Any]] = []
    for rank, plan in enumerate(plans):
        print(f"Evaluating {plan.plan_id} ({rank + 1}/{len(plans)})", flush=True)
        results.append(
            evaluate_one_plan(
                plan=plan,
                rank=rank,
                root_dir=root_dir,
                source_dir=source_dir,
                source_manifest=source_manifest,
                source_examples=source_examples,
                extracted=extracted,
                profiles=profiles,
                diagnostic_mode=args.diagnostic_mode,
                seed=args.seed,
                c_grid=c_grid,
                classifier_solver=args.classifier_solver,
                classifier_max_iter=args.classifier_max_iter,
                classifier_tol=args.classifier_tol,
                stress_min_eval_per_label=args.stress_min_eval_per_label,
            )
        )

    root_manifest["finished_at"] = now_iso()
    aggregate = {
        **root_manifest,
        "results": results,
    }
    write_json(root_dir / "aggregate_summary.json", aggregate)
    write_aggregate_report(root_dir / "report.md", payload=aggregate)
    write_json(root_dir / "manifest.json", root_manifest)
    print(f"Wrote {root_dir / 'report.md'}")


if __name__ == "__main__":
    main()
