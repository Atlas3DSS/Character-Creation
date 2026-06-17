#!/usr/bin/env python3
"""Build a blind pairwise review queue for SCOTUS causal-pilot pockets."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PROMPT_BANK = PROJECT_ROOT / "data" / "scotus" / "scotus_poke_prompts_v1.jsonl"
DEFAULT_RUNS = (
    PROJECT_ROOT / "sweep_v4" / "scotus_sae_poke_20260501_045156",
    PROJECT_ROOT / "sweep_v4" / "scotus_sae_poke_20260501_060425",
)
DEFAULT_BLIND = PROJECT_ROOT / "data" / "scotus" / "scotus_majority2000s_causal_review_blind_20260501.jsonl"
DEFAULT_KEY = PROJECT_ROOT / "data" / "scotus" / "scotus_majority2000s_causal_review_key_20260501.jsonl"
DEFAULT_REPORT = PROJECT_ROOT / "reports" / "scotus_majority2000s_causal_prompt_pockets_20260501.md"


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


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def stable_bool(value: str) -> bool:
    return int(hashlib.sha256(value.encode("utf-8")).hexdigest()[:8], 16) % 2 == 0


def markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(item) for item in row) + " |")
    return "\n".join(lines)


@dataclass(frozen=True)
class ScoredCandidate:
    run_name: str
    run_dir: Path
    row: dict[str, Any]
    matched_target: float
    matched_net: float
    random_target_mean: float
    random_net_mean: float
    candidate_target_delta: float
    candidate_net_delta: float


def run_name(run_dir: Path) -> str:
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.exists():
        return run_dir.name
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    direction_files = manifest.get("external_direction_files", [])
    position = manifest.get("position", "")
    if direction_files:
        direction = Path(str(direction_files[0])).parent.name
        if direction.startswith("split_"):
            direction = f"{direction}_{Path(str(direction_files[0])).stem}"
        return f"{direction}__{position}"
    return f"{run_dir.name}__{position}"


def score_candidates(run_dir: Path) -> tuple[list[ScoredCandidate], dict[tuple[str, float], list[dict[str, Any]]], dict[str, dict[str, Any]]]:
    rows = read_jsonl(run_dir / "generations.jsonl")
    random_rows: dict[tuple[str, float], list[dict[str, Any]]] = {}
    base_rows: dict[str, dict[str, Any]] = {}
    candidate_rows: list[dict[str, Any]] = []
    for row in rows:
        condition = row.get("condition")
        if condition == "base":
            base_rows[str(row["prompt_key"])] = row
        elif condition == "random_unit":
            random_rows.setdefault((str(row["prompt_key"]), float(row["alpha"])), []).append(row)
        elif condition == "sae_poke":
            candidate_rows.append(row)

    scored: list[ScoredCandidate] = []
    name = run_name(run_dir)
    for row in candidate_rows:
        key = (str(row["prompt_key"]), float(row["alpha"]))
        controls = random_rows.get(key, [])
        if not controls:
            continue
        random_target_mean = sum(float(item["frame_eval"].get("delta_target_hits_vs_base", 0.0)) for item in controls) / len(controls)
        random_net_mean = sum(float(item["frame_eval"].get("delta_target_minus_contrast_vs_base", 0.0)) for item in controls) / len(controls)
        candidate_target_delta = float(row["frame_eval"].get("delta_target_hits_vs_base", 0.0))
        candidate_net_delta = float(row["frame_eval"].get("delta_target_minus_contrast_vs_base", 0.0))
        scored.append(
            ScoredCandidate(
                run_name=name,
                run_dir=run_dir,
                row=row,
                matched_target=candidate_target_delta - random_target_mean,
                matched_net=candidate_net_delta - random_net_mean,
                random_target_mean=random_target_mean,
                random_net_mean=random_net_mean,
                candidate_target_delta=candidate_target_delta,
                candidate_net_delta=candidate_net_delta,
            )
        )
    return scored, random_rows, base_rows


def selected_candidates(scored: list[ScoredCandidate], max_items: int) -> list[ScoredCandidate]:
    plausible = [
        item
        for item in scored
        if (item.candidate_target_delta > 0.0 or item.candidate_net_delta > 0.0)
        and (item.matched_target >= 1.0 or item.matched_net >= 1.0)
    ]
    plausible.sort(key=lambda item: (max(item.matched_target, item.matched_net), item.candidate_net_delta), reverse=True)
    selected: list[ScoredCandidate] = []
    seen: set[tuple[str, str, float]] = set()
    for item in plausible:
        key = (item.run_name, str(item.row["prompt_key"]), float(item.row["alpha"]))
        if key in seen:
            continue
        selected.append(item)
        seen.add(key)
        if len(selected) >= max_items:
            break
    return selected


def choose_random_controls(
    controls: list[dict[str, Any]],
    *,
    random_target_mean: float,
) -> list[tuple[str, dict[str, Any]]]:
    if not controls:
        return []
    medianish = min(
        controls,
        key=lambda row: abs(float(row["frame_eval"].get("delta_target_hits_vs_base", 0.0)) - random_target_mean),
    )
    strongest = max(controls, key=lambda row: float(row["frame_eval"].get("delta_target_hits_vs_base", 0.0)))
    chosen = [("random_closest_mean", medianish)]
    if strongest is not medianish:
        chosen.append(("random_strongest_target", strongest))
    return chosen


def pair_row(
    *,
    review_id: str,
    prompt_meta: dict[str, Any],
    left: dict[str, Any],
    right: dict[str, Any],
) -> dict[str, Any]:
    return {
        "review_id": review_id,
        "prompt_key": prompt_meta["prompt_key"],
        "issue_area": prompt_meta["issue_area"],
        "prompt": prompt_meta["prompt"],
        "target_frames": prompt_meta.get("expected_frames", []),
        "contrast_frames": prompt_meta.get("contrast_frames", []),
        "review_question": "Which completion shows stronger target-frame movement while preserving legal coherence?",
        "allowed_labels": [
            "A_stronger_target_frame",
            "B_stronger_target_frame",
            "no_material_difference",
            "A_degraded_or_off_target",
            "B_degraded_or_off_target",
            "both_degraded_or_off_target",
        ],
        "review_label": "",
        "review_confidence": "",
        "review_notes": "",
        "completion_a": left["text"],
        "completion_b": right["text"],
    }


def build_queues(args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[ScoredCandidate]]:
    prompt_rows = {str(row.get("prompt_key") or row.get("id")): row for row in read_jsonl(args.prompt_bank)}
    all_scored: list[ScoredCandidate] = []
    random_by_run: dict[Path, dict[tuple[str, float], list[dict[str, Any]]]] = {}
    base_by_run: dict[Path, dict[str, dict[str, Any]]] = {}
    for run_dir in args.runs:
        scored, random_rows, base_rows = score_candidates(run_dir)
        all_scored.extend(scored)
        random_by_run[run_dir] = random_rows
        base_by_run[run_dir] = base_rows

    selected = selected_candidates(all_scored, args.max_candidates)
    blind_rows: list[dict[str, Any]] = []
    key_rows: list[dict[str, Any]] = []
    for item in selected:
        candidate = item.row
        prompt_key = str(candidate["prompt_key"])
        prompt_meta = prompt_rows[prompt_key]
        prompt_meta = {
            **prompt_meta,
            "prompt_key": prompt_key,
            "issue_area": candidate.get("issue_area", prompt_meta.get("issue_area", "")),
        }
        base = base_by_run[item.run_dir][prompt_key]
        comparisons: list[tuple[str, dict[str, Any]]] = [("base", base)]
        comparisons.extend(
            choose_random_controls(
                random_by_run[item.run_dir][(prompt_key, float(candidate["alpha"]))],
                random_target_mean=item.random_target_mean,
            )
        )
        for comparison_name, comparison_row in comparisons:
            review_id = (
                f"scotus_causal::{item.run_name}::{prompt_key}::"
                f"a{float(candidate['alpha']):g}::{comparison_name}"
            )
            candidate_left = stable_bool(review_id)
            left = candidate if candidate_left else comparison_row
            right = comparison_row if candidate_left else candidate
            blind_rows.append(pair_row(review_id=review_id, prompt_meta=prompt_meta, left=left, right=right))
            key_rows.append(
                {
                    "review_id": review_id,
                    "run_name": item.run_name,
                    "run_dir": str(item.run_dir),
                    "prompt_key": prompt_key,
                    "issue_area": candidate.get("issue_area", ""),
                    "alpha": float(candidate["alpha"]),
                    "comparison": comparison_name,
                    "candidate_side": "A" if candidate_left else "B",
                    "candidate_target_delta": item.candidate_target_delta,
                    "candidate_net_delta": item.candidate_net_delta,
                    "matched_target_delta": item.matched_target,
                    "matched_net_delta": item.matched_net,
                    "random_target_mean": item.random_target_mean,
                    "random_net_mean": item.random_net_mean,
                    "comparison_condition": comparison_row.get("condition"),
                    "comparison_random_index": comparison_row.get("random_index"),
                    "comparison_target_delta": float(comparison_row["frame_eval"].get("delta_target_hits_vs_base", 0.0)),
                    "comparison_net_delta": float(comparison_row["frame_eval"].get("delta_target_minus_contrast_vs_base", 0.0)),
                    "candidate_frame_scores": candidate.get("frame_scores", {}),
                    "comparison_frame_scores": comparison_row.get("frame_scores", {}),
                    "review_label": "",
                    "review_confidence": "",
                    "review_notes": "",
                }
            )
    return blind_rows, key_rows, selected


def write_report(path: Path, *, selected: list[ScoredCandidate], blind_rows: list[dict[str, Any]], key_rows: list[dict[str, Any]], args: argparse.Namespace) -> None:
    selected_rows = [
        [
            item.run_name,
            item.row["prompt_key"],
            item.row["issue_area"],
            f"{float(item.row['alpha']):g}",
            f"{item.candidate_target_delta:.2f}",
            f"{item.matched_target:.2f}",
            f"{item.candidate_net_delta:.2f}",
            f"{item.matched_net:.2f}",
        ]
        for item in selected
    ]
    pair_counts: dict[str, int] = {}
    for row in key_rows:
        pair_counts[str(row["comparison"])] = pair_counts.get(str(row["comparison"]), 0) + 1
    lines = [
        "# SCOTUS Majority-2000s Causal Prompt Pockets",
        "",
        "## Purpose",
        "",
        "The aggregate causal pilots did not clear the steering gate. This report identifies the small set of prompt-level pockets where candidate completions had positive absolute movement and also beat prompt-matched random controls by at least one frame-hit unit.",
        "",
        "These rows are investigative leads, not steering evidence. The queue is blind and pairwise so a reviewer can check whether the apparent movement is visible in legal reasoning rather than a keyword artifact.",
        "",
        "## Outputs",
        "",
        f"- Blind review queue: `{args.blind}`",
        f"- Key file: `{args.key}`",
        f"- Selected candidate cells: `{len(selected)}`",
        f"- Pairwise review rows: `{len(blind_rows)}`",
        "",
        "## Selected Candidate Cells",
        "",
        markdown_table(
            [
                "Run",
                "Prompt",
                "Issue",
                "Alpha",
                "Candidate target delta",
                "Matched target delta",
                "Candidate net delta",
                "Matched net delta",
            ],
            selected_rows,
        ),
        "",
        "## Pair Types",
        "",
        markdown_table(["Comparison", "Rows"], [[name, count] for name, count in sorted(pair_counts.items())]),
        "",
        "## Review Rule",
        "",
        "Use the blind queue only. A prompt family should not be promoted unless the candidate side wins against both baseline and random-control comparisons without coherence degradation.",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt-bank", type=Path, default=DEFAULT_PROMPT_BANK)
    parser.add_argument("--runs", type=Path, nargs="+", default=list(DEFAULT_RUNS))
    parser.add_argument("--blind", type=Path, default=DEFAULT_BLIND)
    parser.add_argument("--key", type=Path, default=DEFAULT_KEY)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--max-candidates", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    blind_rows, key_rows, selected = build_queues(args)
    write_jsonl(args.blind, blind_rows)
    write_jsonl(args.key, key_rows)
    write_json(
        args.report.with_suffix(".json"),
        {
            "blind": str(args.blind),
            "key": str(args.key),
            "report": str(args.report),
            "runs": [str(path) for path in args.runs],
            "selected_candidates": len(selected),
            "review_rows": len(blind_rows),
        },
    )
    write_report(args.report, selected=selected, blind_rows=blind_rows, key_rows=key_rows, args=args)
    print(f"Wrote {args.report}")
    print(f"Wrote {args.blind}")
    print(f"Wrote {args.key}")


if __name__ == "__main__":
    main()
