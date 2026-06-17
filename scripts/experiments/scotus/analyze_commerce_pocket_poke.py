#!/usr/bin/env python3
"""Summarize targeted Commerce Clause poke runs by prompt family.

The generic poke report is row-oriented. This script gives a stricter read for
the Commerce-pocket follow-up: does a candidate direction beat prompt-matched
same-layer random controls within limits prompts, authority/remedy prompts, or
the whole bank?
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PROMPT_BANK = PROJECT_ROOT / "data" / "scotus" / "scotus_commerce_pocket_prompts_v1.jsonl"
DEFAULT_REPORT = PROJECT_ROOT / "reports" / "scotus_commerce_pocket_poke_20260501.md"


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


def markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(item) for item in row) + " |")
    return "\n".join(lines)


def fmt(value: float | None, digits: int = 3) -> str:
    if value is None:
        return ""
    if math.isnan(value):
        return "nan"
    return f"{value:.{digits}f}"


def fmt_alpha(value: float) -> str:
    text = f"{value:.6f}".rstrip("0").rstrip(".")
    return text or "0"


def prompt_family(prompt_key: str) -> str:
    if "_LIMIT_" in prompt_key:
        return "commerce_limits"
    if "_AUTH_" in prompt_key:
        return "commerce_authority_remedy"
    return "other"


def run_name(run_dir: Path) -> str:
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.exists():
        return run_dir.name
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    direction_files = manifest.get("external_direction_files", [])
    position = manifest.get("position", "")
    if direction_files:
        path = Path(str(direction_files[0]))
        return f"{path.parent.name}__{position}"
    return f"{run_dir.name}__{position}"


def stddev(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mu = mean(values)
    return math.sqrt(sum((value - mu) ** 2 for value in values) / (len(values) - 1))


@dataclass(frozen=True)
class PromptComparison:
    run_name: str
    run_dir: Path
    prompt_key: str
    family: str
    candidate: str
    alpha: float
    candidate_target_delta: float
    candidate_net_delta: float
    random_target_mean: float
    random_net_mean: float
    random_target_max: float
    random_net_max: float
    matched_target_delta: float
    matched_net_delta: float
    beats_random_mean_target: bool
    beats_random_mean_net: bool
    beats_random_max_target: bool
    beats_random_max_net: bool
    text: str


def build_comparisons(run_dir: Path, prompt_meta: dict[str, dict[str, Any]]) -> list[PromptComparison]:
    rows = read_jsonl(run_dir / "generations.jsonl")
    random_rows: dict[tuple[str, float], list[dict[str, Any]]] = defaultdict(list)
    candidate_rows: list[dict[str, Any]] = []
    for row in rows:
        condition = row.get("condition")
        if condition == "random_unit":
            random_rows[(str(row["prompt_key"]), float(row["alpha"]))].append(row)
        elif condition == "sae_poke":
            candidate_rows.append(row)

    comparisons: list[PromptComparison] = []
    name = run_name(run_dir)
    for row in candidate_rows:
        prompt_key = str(row["prompt_key"])
        alpha = float(row["alpha"])
        controls = random_rows.get((prompt_key, alpha), [])
        if not controls:
            continue
        random_target_values = [
            float(item["frame_eval"].get("delta_target_hits_vs_base", 0.0)) for item in controls
        ]
        random_net_values = [
            float(item["frame_eval"].get("delta_target_minus_contrast_vs_base", 0.0)) for item in controls
        ]
        candidate_target_delta = float(row["frame_eval"].get("delta_target_hits_vs_base", 0.0))
        candidate_net_delta = float(row["frame_eval"].get("delta_target_minus_contrast_vs_base", 0.0))
        random_target_mean = mean(random_target_values)
        random_net_mean = mean(random_net_values)
        random_target_max = max(random_target_values)
        random_net_max = max(random_net_values)
        meta = prompt_meta.get(prompt_key, {})
        comparisons.append(
            PromptComparison(
                run_name=name,
                run_dir=run_dir,
                prompt_key=prompt_key,
                family=prompt_family(prompt_key),
                candidate=str(row.get("candidate") or ""),
                alpha=alpha,
                candidate_target_delta=candidate_target_delta,
                candidate_net_delta=candidate_net_delta,
                random_target_mean=random_target_mean,
                random_net_mean=random_net_mean,
                random_target_max=random_target_max,
                random_net_max=random_net_max,
                matched_target_delta=candidate_target_delta - random_target_mean,
                matched_net_delta=candidate_net_delta - random_net_mean,
                beats_random_mean_target=candidate_target_delta > random_target_mean,
                beats_random_mean_net=candidate_net_delta > random_net_mean,
                beats_random_max_target=candidate_target_delta > random_target_max,
                beats_random_max_net=candidate_net_delta > random_net_max,
                text=str(row.get("text") or meta.get("prompt") or ""),
            )
        )
    return comparisons


def aggregate(comparisons: list[PromptComparison]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, float], list[PromptComparison]] = defaultdict(list)
    for item in comparisons:
        groups[(item.run_name, item.family, item.alpha)].append(item)

    rows: list[dict[str, Any]] = []
    for (name, family, alpha), items in sorted(groups.items()):
        matched_target = [item.matched_target_delta for item in items]
        matched_net = [item.matched_net_delta for item in items]
        random_residual_target = []
        random_residual_net = []
        for item in items:
            random_residual_target.append(item.candidate_target_delta - item.random_target_mean)
            random_residual_net.append(item.candidate_net_delta - item.random_net_mean)
        sd_target = stddev(random_residual_target)
        sd_net = stddev(random_residual_net)
        rows.append(
            {
                "run_name": name,
                "family": family,
                "alpha": alpha,
                "n": len(items),
                "mean_candidate_target_delta": mean(item.candidate_target_delta for item in items),
                "mean_random_target_delta": mean(item.random_target_mean for item in items),
                "mean_matched_target_delta": mean(matched_target),
                "mean_candidate_net_delta": mean(item.candidate_net_delta for item in items),
                "mean_random_net_delta": mean(item.random_net_mean for item in items),
                "mean_matched_net_delta": mean(matched_net),
                "matched_target_sd": sd_target,
                "matched_net_sd": sd_net,
                "target_prompt_win_rate": mean(1.0 if item.beats_random_mean_target else 0.0 for item in items),
                "net_prompt_win_rate": mean(1.0 if item.beats_random_mean_net else 0.0 for item in items),
                "target_strongest_win_rate": mean(1.0 if item.beats_random_max_target else 0.0 for item in items),
                "net_strongest_win_rate": mean(1.0 if item.beats_random_max_net else 0.0 for item in items),
            }
        )
    rows.sort(
        key=lambda row: (
            float(row["mean_matched_net_delta"]),
            float(row["net_strongest_win_rate"]),
            float(row["mean_matched_target_delta"]),
        ),
        reverse=True,
    )
    return rows


def top_prompt_rows(comparisons: list[PromptComparison], limit: int) -> list[PromptComparison]:
    ordered = sorted(
        comparisons,
        key=lambda item: (
            item.matched_net_delta,
            item.beats_random_max_net,
            item.matched_target_delta,
        ),
        reverse=True,
    )
    return ordered[:limit]


def write_report(path: Path, comparisons: list[PromptComparison], aggregate_rows: list[dict[str, Any]]) -> None:
    aggregate_table = [
        [
            row["run_name"],
            row["family"],
            fmt_alpha(float(row["alpha"])),
            row["n"],
            fmt(float(row["mean_candidate_target_delta"])),
            fmt(float(row["mean_random_target_delta"])),
            fmt(float(row["mean_matched_target_delta"])),
            fmt(float(row["mean_candidate_net_delta"])),
            fmt(float(row["mean_random_net_delta"])),
            fmt(float(row["mean_matched_net_delta"])),
            fmt(float(row["target_prompt_win_rate"]), 2),
            fmt(float(row["net_prompt_win_rate"]), 2),
            fmt(float(row["target_strongest_win_rate"]), 2),
            fmt(float(row["net_strongest_win_rate"]), 2),
        ]
        for row in aggregate_rows
    ]
    prompt_table = [
        [
            item.run_name,
            item.prompt_key,
            item.family,
            fmt_alpha(item.alpha),
            fmt(item.candidate_target_delta),
            fmt(item.random_target_mean),
            fmt(item.matched_target_delta),
            fmt(item.candidate_net_delta),
            fmt(item.random_net_mean),
            fmt(item.matched_net_delta),
            "Y" if item.beats_random_max_target else "N",
            "Y" if item.beats_random_max_net else "N",
        ]
        for item in top_prompt_rows(comparisons, 30)
    ]

    lines = [
        "# SCOTUS Commerce Pocket Poke Summary",
        "",
        "## Purpose",
        "",
        "Summarize targeted Commerce Clause / Economic Activity causal pokes by prompt family.",
        "A row is promising only if it beats prompt-matched same-layer random controls, especially the strongest random control for the same prompt and alpha.",
        "",
        "## Aggregate",
        "",
        markdown_table(
            [
                "Run",
                "Family",
                "Alpha",
                "N",
                "Cand target",
                "Rand target",
                "Matched target",
                "Cand net",
                "Rand net",
                "Matched net",
                "Target win",
                "Net win",
                "Target strongest win",
                "Net strongest win",
            ],
            aggregate_table,
        ),
        "",
        "## Top Prompt Rows",
        "",
        markdown_table(
            [
                "Run",
                "Prompt",
                "Family",
                "Alpha",
                "Cand target",
                "Rand target",
                "Matched target",
                "Cand net",
                "Rand net",
                "Matched net",
                "Beats target max",
                "Beats net max",
            ],
            prompt_table,
        ),
        "",
        "## Reading Rule",
        "",
        "- Mean wins over random controls are suggestive only.",
        "- Strongest-random wins are the important gate for promotion.",
        "- This summary still uses keyword/proposition frame counts; any survivor needs blind text review before promotion.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze targeted Commerce-pocket SCOTUS poke runs.")
    parser.add_argument("--prompt-bank", type=Path, default=DEFAULT_PROMPT_BANK)
    parser.add_argument("--runs", type=Path, nargs="+", required=True)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prompt_meta = {str(row["prompt_key"]): row for row in read_jsonl(args.prompt_bank)}
    comparisons: list[PromptComparison] = []
    for run_dir in args.runs:
        comparisons.extend(build_comparisons(run_dir, prompt_meta))
    aggregate_rows = aggregate(comparisons)
    write_report(args.report, comparisons, aggregate_rows)
    print(f"Wrote {args.report} with {len(comparisons)} comparisons")


if __name__ == "__main__":
    main()
