#!/usr/bin/env python3
"""Rank SCOTUS issue families before building another source pack.

This script uses the proposition-level Q4 proxy rescore to decide which issue
family is worth source-label work next. It is deliberately pre-hook: it should
prevent spending BF16 generation time on directions whose evaluator/nulls are
already unstable or whose issue family has already failed.
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RESCORER_RUN = PROJECT_ROOT / "sweep_v4" / "scotus_frame_prop_rescore_20260501_012850"
DEFAULT_PROMPT_BANK = PROJECT_ROOT / "data" / "scotus" / "scotus_poke_prompts_v1.jsonl"
DEFAULT_REPORT = PROJECT_ROOT / "reports" / "scotus_issue_family_triage_20260501.md"
DEFAULT_JSON = PROJECT_ROOT / "reports" / "scotus_issue_family_triage_20260501.json"


@dataclass(frozen=True)
class FamilyPrior:
    feasibility: int
    status: str
    next_action: str
    notes: str


FAMILY_PRIORS: dict[str, FamilyPrior] = {
    "Judicial Power": FamilyPrior(
        feasibility=3,
        status="deprioritize_current_branch_failed",
        next_action="Do not build another Article III pack unless a second reviewer changes labels or a new subdoctrine is defined.",
        notes="Already tested with target corpus, expanded source pack, dominance review, and proposition rescore.",
    ),
    "Criminal Procedure": FamilyPrior(
        feasibility=3,
        status="deprioritize_fourth_branch_failed",
        next_action="Do not hook current Fourth directions; use only for evaluator diagnostics.",
        notes="Fourth Amendment source pack and corrected source probe were text-baseline dominated or split-skewed.",
    ),
    "Economic Activity": FamilyPrior(
        feasibility=3,
        status="candidate",
        next_action="Build a Commerce Clause source pack: broad aggregation/market regulation versus Lopez/Morrison/NFIB limits.",
        notes="Has four prompts, known source opinions, and known Scalia/Thomas-style variance around Raich, Lopez, Morrison, and NFIB.",
    ),
    "Civil Rights": FamilyPrior(
        feasibility=3,
        status="backup_candidate",
        next_action="Only after Economic Activity; likely needs dominance review because strict/intermediate scrutiny labels are lexical.",
        notes="Good source availability, but high risk that the task reduces to named scrutiny formulas.",
    ),
    "Due Process": FamilyPrior(
        feasibility=2,
        status="needs_prompt_expansion",
        next_action="Expand prompts and define narrower subframes before a source pack.",
        notes="Only two prompts; doctrine is broad and may blur substantive/procedural/equal-protection frames.",
    ),
    "Federalism": FamilyPrior(
        feasibility=2,
        status="needs_prompt_expansion",
        next_action="Expand prompt bank before source probing; possible anti-commandeering/preemption pack.",
        notes="Only one prompt and remaining off-domain contamination is nontrivial.",
    ),
    "Administrative Law": FamilyPrior(
        feasibility=2,
        status="needs_prompt_expansion",
        next_action="Expand major-questions/nondelegation/Chevron prompts before source probing.",
        notes="Only one prompt; source labels are feasible but the prompt bank is too narrow.",
    ),
}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                rows.append(json.loads(stripped))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
    return rows


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def markdown_table(headers: list[str], rows: Iterable[Iterable[Any]]) -> list[str]:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return lines


def mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def stdev(values: list[float]) -> float:
    return float(statistics.stdev(values)) if len(values) > 1 else 0.0


def pct(value: float) -> str:
    return f"{value:.3f}"


def frame_set(row: dict[str, Any], key: str) -> set[str]:
    scores = row.get(key, {})
    if not isinstance(scores, dict):
        return set()
    frames: set[str] = set()
    for frame, value in scores.items():
        try:
            if int(value) > 0:
                frames.add(str(frame))
        except (TypeError, ValueError):
            continue
    return frames


def prompt_specs(prompt_bank: Path) -> dict[int, dict[str, Any]]:
    specs = {}
    for row in read_jsonl(prompt_bank):
        specs[int(row["prompt_id"])] = row
    return specs


def load_proxy_rows(rescore_dir: Path) -> list[dict[str, Any]]:
    rows = read_jsonl(rescore_dir / "rescored_rows.jsonl")
    return [row for row in rows if row.get("source_run") == "scotus_qwen4bit_proxy_20260501_045257"]


def compute_prompt_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    random_groups: dict[tuple[Any, Any], list[dict[str, Any]]] = defaultdict(list)
    base_groups: dict[tuple[Any, Any], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (row.get("prompt_id"), row.get("condition_context"))
        if row.get("sample_kind") == "random_control":
            random_groups[key].append(row)
        elif row.get("sample_kind") == "base":
            base_groups[key].append(row)

    prompt_rows: list[dict[str, Any]] = []
    for (prompt_id, context), group_rows in sorted(random_groups.items(), key=str):
        evals = [row["proposition_frame_eval"] for row in group_rows]
        deltas = [float(item.get("delta_target_hits_vs_base", 0.0)) for item in evals]
        nets = [float(item.get("delta_target_minus_contrast_vs_base", 0.0)) for item in evals]
        disagreements = [
            frame_set(row, "frame_scores") != frame_set(row, "proposition_frame_scores")
            for row in group_rows
        ]
        base_evals = [row["proposition_frame_eval"] for row in base_groups.get((prompt_id, context), [])]
        prompt_rows.append(
            {
                "prompt_id": prompt_id,
                "prompt_key": group_rows[0].get("prompt_key"),
                "issue_area": group_rows[0].get("issue_area"),
                "condition_context": context,
                "n": len(group_rows),
                "target_present_rate": mean([1.0 if item.get("target_present") else 0.0 for item in evals]),
                "base_target_present_rate": mean([1.0 if item.get("target_present") else 0.0 for item in base_evals]),
                "off_domain_present_rate": mean([1.0 if item.get("off_domain_present") else 0.0 for item in evals]),
                "mean_target_delta": mean(deltas),
                "sd_target_delta": stdev(deltas),
                "mean_net_delta": mean(nets),
                "sd_net_delta": stdev(nets),
                "disagreement_rate": mean([1.0 if item else 0.0 for item in disagreements]),
            }
        )
    return prompt_rows


def compute_issue_rows(prompt_rows: list[dict[str, Any]], specs: dict[int, dict[str, Any]]) -> list[dict[str, Any]]:
    by_issue: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in prompt_rows:
        by_issue[str(row["issue_area"])].append(row)

    issue_rows: list[dict[str, Any]] = []
    for issue, rows in sorted(by_issue.items()):
        prompt_ids = sorted({int(row["prompt_id"]) for row in rows})
        prior = FAMILY_PRIORS.get(
            issue,
            FamilyPrior(
                feasibility=1,
                status="unknown",
                next_action="Define source-pack feasibility before proceeding.",
                notes="No prior configured.",
            ),
        )
        mean_sd = mean([float(row["sd_target_delta"]) for row in rows])
        off_domain = mean([float(row["off_domain_present_rate"]) for row in rows])
        disagreement = mean([float(row["disagreement_rate"]) for row in rows])
        target_rate = mean([float(row["target_present_rate"]) for row in rows])
        base_target_rate = mean([float(row["base_target_present_rate"]) for row in rows])
        saturation_penalty = max(0.0, target_rate - 0.95) + max(0.0, base_target_rate - 0.95)
        prompt_bonus = min(len(prompt_ids), 4) / 4.0
        # Higher is better. This ranks candidates for source-pack triage, not steering claims.
        triage_score = (
            prompt_bonus
            + prior.feasibility / 3.0
            + max(0.0, 1.4 - mean_sd) / 1.4
            + max(0.0, 0.20 - off_domain) / 0.20
            - disagreement * 0.50
            - saturation_penalty * 0.75
        )
        if "failed" in prior.status:
            triage_score -= 2.0
        if "needs_prompt_expansion" in prior.status:
            triage_score -= 0.75
        issue_rows.append(
            {
                "issue_area": issue,
                "prompt_count": len(prompt_ids),
                "condition_count": len(rows),
                "random_rows": sum(int(row["n"]) for row in rows),
                "mean_target_present_rate": target_rate,
                "mean_base_target_present_rate": base_target_rate,
                "mean_off_domain_present_rate": off_domain,
                "mean_disagreement_rate": disagreement,
                "mean_sd_target_delta": mean_sd,
                "max_sd_target_delta": max(float(row["sd_target_delta"]) for row in rows),
                "mean_abs_target_delta": mean([abs(float(row["mean_target_delta"])) for row in rows]),
                "source_feasibility": prior.feasibility,
                "status": prior.status,
                "triage_score": triage_score,
                "next_action": prior.next_action,
                "notes": prior.notes,
                "prompts": [
                    {
                        "prompt_id": prompt_id,
                        "prompt_key": specs.get(prompt_id, {}).get("prompt_key", ""),
                        "expected_frames": specs.get(prompt_id, {}).get("expected_frames", []),
                        "contrast_frames": specs.get(prompt_id, {}).get("contrast_frames", []),
                    }
                    for prompt_id in prompt_ids
                ],
            }
        )
    issue_rows.sort(key=lambda row: float(row["triage_score"]), reverse=True)
    return issue_rows


def write_report(
    path: Path,
    *,
    rescore_dir: Path,
    prompt_rows: list[dict[str, Any]],
    issue_rows: list[dict[str, Any]],
) -> None:
    ranked_rows = [
        [
            idx + 1,
            row["issue_area"],
            row["status"],
            row["prompt_count"],
            pct(float(row["triage_score"])),
            pct(float(row["mean_sd_target_delta"])),
            pct(float(row["mean_off_domain_present_rate"])),
            pct(float(row["mean_disagreement_rate"])),
            row["next_action"],
        ]
        for idx, row in enumerate(issue_rows)
    ]
    prompt_detail_rows = [
        [
            row["issue_area"],
            row["prompt_key"],
            row["condition_context"],
            row["n"],
            pct(float(row["target_present_rate"])),
            pct(float(row["off_domain_present_rate"])),
            pct(float(row["sd_target_delta"])),
            pct(float(row["mean_target_delta"])),
            pct(float(row["disagreement_rate"])),
        ]
        for row in sorted(prompt_rows, key=lambda item: (str(item["issue_area"]), str(item["prompt_key"]), str(item["condition_context"])))
    ]

    best = issue_rows[0] if issue_rows else {}
    lines = [
        "# SCOTUS Issue Family Triage",
        "",
        "## Purpose",
        "",
        "This ranks issue families before building another source pack or spending BF16 hook time. It uses the corrected proposition-level Q4 proxy rescore, not activation evidence.",
        "",
        "## Inputs",
        "",
        f"- Rescore run: `{rescore_dir}`",
        "- Rows used: Q4 proxy `random_control` and `base` completions only",
        "- Gate: this report can nominate a source-pack branch, but it cannot promote a steering direction.",
        "",
        "## Main Read",
        "",
        f"Top new candidate: `{best.get('issue_area', 'n/a')}`.",
        "",
        "Economic Activity is the preferred next source-pack branch because it has four prompts, relatively low proposition off-domain contamination, a stable null, and a natural source contrast: broad Commerce Clause aggregation/market regulation versus Lopez/Morrison/NFIB-style limits.",
        "",
        "Civil Rights is the backup, but it is more likely to collapse into lexical scrutiny labels unless dominance-reviewed.",
        "",
        "## Issue Ranking",
        "",
    ]
    lines.extend(
        markdown_table(
            [
                "Rank",
                "Issue",
                "Status",
                "Prompts",
                "Score",
                "Mean SD",
                "Off-domain",
                "Disagreement",
                "Next action",
            ],
            ranked_rows,
        )
    )
    lines.extend(["", "## Prompt/Condition Detail", ""])
    lines.extend(
        markdown_table(
            [
                "Issue",
                "Prompt",
                "Condition",
                "N",
                "Target present",
                "Off-domain",
                "SD target delta",
                "Mean target delta",
                "Disagreement",
            ],
            prompt_detail_rows,
        )
    )
    lines.extend(
        [
            "",
            "## Use Rules",
            "",
            "1. Do not run hooks from this report alone.",
            "2. For the next branch, build source labels first, run cue-masked activation probes, and compare against text baselines.",
            "3. Promote only if the candidate survives reviewed labels, cue masking, text baseline, and proposition-level prompt-matched random controls.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rescore-dir", type=Path, default=DEFAULT_RESCORER_RUN)
    parser.add_argument("--prompt-bank", type=Path, default=DEFAULT_PROMPT_BANK)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--json-out", type=Path, default=DEFAULT_JSON)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    specs = prompt_specs(args.prompt_bank)
    proxy_rows = load_proxy_rows(args.rescore_dir)
    prompt_rows = compute_prompt_rows(proxy_rows)
    issue_rows = compute_issue_rows(prompt_rows, specs)
    payload = {
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "rescore_dir": str(args.rescore_dir),
        "prompt_bank": str(args.prompt_bank),
        "issue_rows": issue_rows,
        "prompt_rows": prompt_rows,
    }
    write_json(args.json_out, payload)
    write_report(args.report, rescore_dir=args.rescore_dir, prompt_rows=prompt_rows, issue_rows=issue_rows)
    print(f"Wrote {args.report}")
    print(f"Wrote {args.json_out}")


if __name__ == "__main__":
    main()
