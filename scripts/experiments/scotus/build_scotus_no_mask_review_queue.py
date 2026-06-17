#!/usr/bin/env python3
"""Build blind no-mask causal review queues from SCOTUS generation runs.

This script is intentionally evaluation plumbing. It does not promote a
candidate. It turns raw or proposition-rescored generation artifacts into a
blind pairwise queue that asks whether a candidate output shows real
target-frame reasoning, beats baseline/random controls, and avoids the
"I am imitating a target" mask failure.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RESCORERS = (
    PROJECT_ROOT / "sweep_v4" / "scotus_controlled_bundle_prop_rescore_20260501_193155",
)
DEFAULT_BLIND = PROJECT_ROOT / "data" / "scotus" / "scotus_article3_bundle_no_mask_review_blind_20260501.jsonl"
DEFAULT_KEY = PROJECT_ROOT / "data" / "scotus" / "scotus_article3_bundle_no_mask_review_key_20260501.jsonl"
DEFAULT_REPORT = PROJECT_ROOT / "reports" / "scotus_article3_bundle_no_mask_review_queue_20260501.md"


THINK_RE = re.compile(r"(?is)<think>\s*(.*?)\s*</think>\s*(.*)\Z")
OPEN_THINK_RE = re.compile(r"(?is)\A\s*<think>\s*(.*)\Z")
IMITATION_RE = re.compile(
    r"(?i)\b(?:imitat(?:e|ing)|role[- ]?play|as (?:justice|judge|the target)|"
    r"think like|would reason|in the style of|persona)\b"
)


@dataclass(frozen=True)
class CandidateCell:
    source_run: str
    prompt_key: str
    alpha: float
    candidate: dict[str, Any]
    base: dict[str, Any] | None
    controls: tuple[dict[str, Any], ...]
    candidate_target_delta: float
    candidate_net_delta: float
    random_target_mean: float
    random_net_mean: float
    matched_target_delta: float
    matched_net_delta: float
    strongest_random_target_delta: float
    strongest_random_net_delta: float


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


def stable_bool(value: str) -> bool:
    return int(hashlib.sha256(value.encode("utf-8")).hexdigest()[:8], 16) % 2 == 0


def source_rows(path: Path) -> list[dict[str, Any]]:
    if path.is_dir():
        rescored = path / "rescored_rows.jsonl"
        generations = path / "generations.jsonl"
        if rescored.exists():
            return read_jsonl(rescored)
        if generations.exists():
            return read_jsonl(generations)
        raise FileNotFoundError(f"No rescored_rows.jsonl or generations.jsonl in {path}")
    return read_jsonl(path)


def eval_payload(row: dict[str, Any]) -> dict[str, Any]:
    if isinstance(row.get("proposition_frame_eval"), dict):
        return row["proposition_frame_eval"]
    if isinstance(row.get("frame_eval"), dict):
        return row["frame_eval"]
    return {}


def target_delta(row: dict[str, Any]) -> float:
    return float(eval_payload(row).get("delta_target_hits_vs_base", 0.0))


def net_delta(row: dict[str, Any]) -> float:
    return float(eval_payload(row).get("delta_target_minus_contrast_vs_base", 0.0))


def mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def split_thinking(text: str) -> dict[str, Any]:
    stripped = text.strip()
    match = THINK_RE.match(stripped)
    if match:
        thinking = match.group(1).strip()
        answer = match.group(2).strip()
        return {
            "thinking": thinking,
            "answer": answer,
            "has_visible_thinking": bool(thinking),
            "thinking_closed": True,
            "imitation_markers": sorted(set(IMITATION_RE.findall(thinking))),
        }
    match = OPEN_THINK_RE.match(stripped)
    if match:
        thinking = match.group(1).strip()
        return {
            "thinking": thinking,
            "answer": "",
            "has_visible_thinking": bool(thinking),
            "thinking_closed": False,
            "imitation_markers": sorted(set(IMITATION_RE.findall(thinking))),
        }
    return {
        "thinking": "",
        "answer": stripped,
        "has_visible_thinking": False,
        "thinking_closed": False,
        "imitation_markers": [],
    }


def build_candidate_cells(rows: list[dict[str, Any]]) -> list[CandidateCell]:
    base_by_prompt: dict[tuple[str, str], dict[str, Any]] = {}
    random_by_prompt_alpha: dict[tuple[str, str, float], list[dict[str, Any]]] = {}
    candidates: list[dict[str, Any]] = []
    for row in rows:
        source_run = str(row.get("source_run") or row.get("run_name") or "unknown_run")
        prompt_key = str(row.get("prompt_key"))
        condition = str(row.get("condition") or row.get("sample_kind"))
        if condition == "base":
            base_by_prompt[(source_run, prompt_key)] = row
        elif condition == "random_unit":
            random_by_prompt_alpha.setdefault((source_run, prompt_key, float(row.get("alpha") or 0.0)), []).append(row)
        elif condition == "sae_poke":
            candidates.append(row)

    cells: list[CandidateCell] = []
    for candidate in candidates:
        source_run = str(candidate.get("source_run") or candidate.get("run_name") or "unknown_run")
        prompt_key = str(candidate.get("prompt_key"))
        alpha = float(candidate.get("alpha") or 0.0)
        controls = tuple(random_by_prompt_alpha.get((source_run, prompt_key, alpha), []))
        if not controls:
            continue
        random_target = [target_delta(row) for row in controls]
        random_net = [net_delta(row) for row in controls]
        candidate_target = target_delta(candidate)
        candidate_net = net_delta(candidate)
        cells.append(
            CandidateCell(
                source_run=source_run,
                prompt_key=prompt_key,
                alpha=alpha,
                candidate=candidate,
                base=base_by_prompt.get((source_run, prompt_key)),
                controls=controls,
                candidate_target_delta=candidate_target,
                candidate_net_delta=candidate_net,
                random_target_mean=mean(random_target),
                random_net_mean=mean(random_net),
                matched_target_delta=candidate_target - mean(random_target),
                matched_net_delta=candidate_net - mean(random_net),
                strongest_random_target_delta=max(random_target) if random_target else 0.0,
                strongest_random_net_delta=max(random_net) if random_net else 0.0,
            )
        )
    return cells


def select_cells(cells: list[CandidateCell], max_cells: int) -> list[CandidateCell]:
    ranked = sorted(
        cells,
        key=lambda cell: (
            cell.matched_net_delta,
            cell.matched_target_delta,
            cell.candidate_net_delta,
            cell.candidate_target_delta,
            -cell.alpha,
        ),
        reverse=True,
    )
    return ranked[:max_cells]


def choose_controls(cell: CandidateCell) -> list[tuple[str, dict[str, Any]]]:
    comparisons: list[tuple[str, dict[str, Any]]] = []
    if cell.base is not None:
        comparisons.append(("base", cell.base))
    if cell.controls:
        closest_net = min(cell.controls, key=lambda row: abs(net_delta(row) - cell.random_net_mean))
        strongest_net = max(cell.controls, key=net_delta)
        comparisons.append(("random_closest_net_mean", closest_net))
        if strongest_net is not closest_net:
            comparisons.append(("random_strongest_net", strongest_net))
    return comparisons


def blind_review_payload(row: dict[str, Any]) -> dict[str, Any]:
    parsed = split_thinking(str(row.get("text") or ""))
    return {
        "full_text": row.get("text", ""),
        "thinking": parsed["thinking"],
        "answer": parsed["answer"],
        "has_visible_thinking": parsed["has_visible_thinking"],
        "thinking_closed": parsed["thinking_closed"],
    }


def keyed_review_payload(row: dict[str, Any]) -> dict[str, Any]:
    parsed = split_thinking(str(row.get("text") or ""))
    payload = eval_payload(row)
    return {
        "has_visible_thinking": parsed["has_visible_thinking"],
        "thinking_closed": parsed["thinking_closed"],
        "imitation_markers": parsed["imitation_markers"],
        "proposition_frame_scores": row.get("proposition_frame_scores", row.get("frame_scores", {})),
        "target_delta": float(payload.get("delta_target_hits_vs_base", 0.0)),
        "net_delta": float(payload.get("delta_target_minus_contrast_vs_base", 0.0)),
    }


def blind_pair(
    *,
    review_id: str,
    cell: CandidateCell,
    comparison_name: str,
    comparison_row: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    candidate_left = stable_bool(review_id)
    candidate_blind = blind_review_payload(cell.candidate)
    comparison_blind = blind_review_payload(comparison_row)
    candidate_keyed = keyed_review_payload(cell.candidate)
    comparison_keyed = keyed_review_payload(comparison_row)
    left = candidate_blind if candidate_left else comparison_blind
    right = comparison_blind if candidate_left else candidate_blind
    prompt_eval = eval_payload(cell.candidate)
    blind = {
        "review_id": review_id,
        "source_run": cell.source_run,
        "prompt_key": cell.prompt_key,
        "issue_area": cell.candidate.get("issue_area", ""),
        "prompt": cell.candidate.get("prompt", ""),
        "target_frames": prompt_eval.get("expected_frames", []),
        "contrast_frames": prompt_eval.get("contrast_frames", []),
        "domain_frames": prompt_eval.get("domain_frames", []),
        "review_question": (
            "Which completion shows stronger target-frame legal reasoning without degrading coherence, "
            "and does either visible thinking trace reveal an imitation-mask failure?"
        ),
        "target_frame_allowed_labels": [
            "A_stronger_target_reasoning",
            "B_stronger_target_reasoning",
            "no_material_difference",
            "both_degraded_or_off_target",
        ],
        "no_mask_allowed_labels": [
            "A_more_direct_reasoning_basin",
            "B_more_direct_reasoning_basin",
            "no_visible_thinking_to_assess",
            "A_mask_or_imitation_failure",
            "B_mask_or_imitation_failure",
            "both_mask_or_imitation_failure",
            "no_material_difference",
        ],
        "target_frame_label": "",
        "no_mask_label": "",
        "coherence_label": "",
        "review_confidence": "",
        "review_notes": "",
        "completion_a": left,
        "completion_b": right,
    }
    key = {
        "review_id": review_id,
        "source_run": cell.source_run,
        "prompt_key": cell.prompt_key,
        "alpha": cell.alpha,
        "candidate_name": cell.candidate.get("candidate"),
        "comparison": comparison_name,
        "candidate_side": "A" if candidate_left else "B",
        "candidate_target_delta": cell.candidate_target_delta,
        "candidate_net_delta": cell.candidate_net_delta,
        "matched_target_delta": cell.matched_target_delta,
        "matched_net_delta": cell.matched_net_delta,
        "random_target_mean": cell.random_target_mean,
        "random_net_mean": cell.random_net_mean,
        "strongest_random_target_delta": cell.strongest_random_target_delta,
        "strongest_random_net_delta": cell.strongest_random_net_delta,
        "comparison_condition": comparison_row.get("condition") or comparison_row.get("sample_kind"),
        "comparison_random_index": comparison_row.get("random_index"),
        "comparison_target_delta": target_delta(comparison_row),
        "comparison_net_delta": net_delta(comparison_row),
        "candidate_has_visible_thinking": candidate_keyed["has_visible_thinking"],
        "comparison_has_visible_thinking": comparison_keyed["has_visible_thinking"],
        "candidate_imitation_markers": candidate_keyed["imitation_markers"],
        "comparison_imitation_markers": comparison_keyed["imitation_markers"],
        "candidate_proposition_frame_scores": candidate_keyed["proposition_frame_scores"],
        "comparison_proposition_frame_scores": comparison_keyed["proposition_frame_scores"],
        "target_frame_label": "",
        "no_mask_label": "",
        "coherence_label": "",
        "review_confidence": "",
        "review_notes": "",
    }
    return blind, key


def build_queue(args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[CandidateCell]]:
    rows: list[dict[str, Any]] = []
    for source in args.sources:
        rows.extend(source_rows(source))
    selected = select_cells(build_candidate_cells(rows), args.max_cells)
    blind_rows: list[dict[str, Any]] = []
    key_rows: list[dict[str, Any]] = []
    for cell in selected:
        for comparison_name, comparison_row in choose_controls(cell):
            review_id = (
                f"scotus_no_mask::{cell.source_run}::{cell.prompt_key}::"
                f"a{cell.alpha:g}::{comparison_name}"
            )
            blind, key = blind_pair(
                review_id=review_id,
                cell=cell,
                comparison_name=comparison_name,
                comparison_row=comparison_row,
            )
            blind_rows.append(blind)
            key_rows.append(key)
    return blind_rows, key_rows, selected


def write_report(
    path: Path,
    *,
    blind_rows: list[dict[str, Any]],
    key_rows: list[dict[str, Any]],
    selected: list[CandidateCell],
    args: argparse.Namespace,
) -> None:
    selected_rows = [
        [
            cell.source_run,
            cell.prompt_key,
            f"{cell.alpha:g}",
            f"{cell.candidate_target_delta:.3f}",
            f"{cell.matched_target_delta:.3f}",
            f"{cell.candidate_net_delta:.3f}",
            f"{cell.matched_net_delta:.3f}",
            f"{cell.strongest_random_net_delta:.3f}",
        ]
        for cell in selected
    ]
    visible_thinking = sum(
        int(bool(row["candidate_has_visible_thinking"])) + int(bool(row["comparison_has_visible_thinking"]))
        for row in key_rows
    )
    compared_outputs = 2 * len(key_rows)
    lines = [
        "# SCOTUS No-Mask Causal Review Queue",
        "",
        "## Purpose",
        "",
        "Build a blind pairwise review queue for candidate causal generations using proposition-level deltas when available. The review asks whether apparent target-frame movement is real legal reasoning rather than keyword movement, incoherence, or a prompt/persona mask.",
        "",
        "## Inputs",
        "",
        markdown_table(["Source"], [[str(path)] for path in args.sources]),
        "",
        "## Outputs",
        "",
        f"- Blind queue: `{args.blind}`",
        f"- Key file: `{args.key}`",
        f"- Selected candidate cells: `{len(selected)}`",
        f"- Pairwise review rows: `{len(blind_rows)}`",
        f"- Visible-thinking outputs in queue: `{visible_thinking}/{compared_outputs}`",
        "",
        "## Selected Candidate Cells",
        "",
        markdown_table(
            [
                "Run",
                "Prompt",
                "Alpha",
                "Candidate target delta",
                "Matched target delta",
                "Candidate net delta",
                "Matched net delta",
                "Strongest random net",
            ],
            selected_rows,
        ),
        "",
        "## Review Rule",
        "",
        "Use the blind queue first. A candidate should not advance unless its side visibly beats baseline and random controls on target-frame reasoning, preserves coherence, and avoids mask language in any visible thinking trace.",
        "",
        "If visible thinking is absent, mark the no-mask field as `no_visible_thinking_to_assess`; this is not evidence of reasoning-basin success.",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", type=Path, nargs="+", default=list(DEFAULT_RESCORERS))
    parser.add_argument("--blind", type=Path, default=DEFAULT_BLIND)
    parser.add_argument("--key", type=Path, default=DEFAULT_KEY)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--max-cells", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    blind_rows, key_rows, selected = build_queue(args)
    write_jsonl(args.blind, blind_rows)
    write_jsonl(args.key, key_rows)
    write_json(
        args.report.with_suffix(".json"),
        {
            "blind": str(args.blind),
            "key": str(args.key),
            "report": str(args.report),
            "sources": [str(path) for path in args.sources],
            "selected_candidate_cells": len(selected),
            "review_rows": len(blind_rows),
        },
    )
    write_report(args.report, blind_rows=blind_rows, key_rows=key_rows, selected=selected, args=args)
    print(f"Wrote {args.report}")
    print(f"Wrote {args.blind}")
    print(f"Wrote {args.key}")


if __name__ == "__main__":
    main()
