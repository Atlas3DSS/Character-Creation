#!/usr/bin/env python3
"""Build a dominance-review queue for the narrow Economic Activity pockets."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SCOTUS_DIR = PROJECT_ROOT / "data" / "scotus"
DEFAULT_LABELS = SCOTUS_DIR / "scotus_economic_source_frame_labels_v1.jsonl"
DEFAULT_PROMPTS = SCOTUS_DIR / "scotus_poke_prompts_v1.jsonl"
DEFAULT_ADJUDICATION = SCOTUS_DIR / "scotus_majority2000s_causal_review_adjudicated_20260501.jsonl"
DEFAULT_CLEAN_PROBE = PROJECT_ROOT / "sweep_v4" / "scotus_economic_clean_broad_limits_cached_20260501" / "report.md"
DEFAULT_QUEUE = SCOTUS_DIR / "scotus_economic_pocket_dominance_review_20260501.jsonl"
DEFAULT_REPORT = PROJECT_ROOT / "reports" / "scotus_economic_pocket_followup_20260501.md"

BROAD_FRAME = "economic_commerce_broad_aggregation"
LIMITS_FRAME = "economic_commerce_limits"


@dataclass(frozen=True)
class PocketSpec:
    pocket_id: str
    prompt_key: str
    target_frame: str
    contrast_frame: str
    target_cases: tuple[str, ...]
    contrast_cases: tuple[str, ...]
    rationale: str


POCKETS: tuple[PocketSpec, ...] = (
    PocketSpec(
        pocket_id="EA03_limits_school_zone",
        prompt_key="EA03_gun_school_zone",
        target_frame=LIMITS_FRAME,
        contrast_frame=BROAD_FRAME,
        target_cases=("lopez_1995", "morrison_2000", "nfib_2012", "carter_coal_1936", "schechter_1935"),
        contrast_cases=(
            "raich_2005",
            "wickard_1942",
            "perez_1971",
            "hodel_1981",
            "heart_atlanta_1964",
            "champion_1903",
            "katzenbach_mcclung_1964",
            "gibbons_1824",
            "shreveport_1914",
            "stafford_1922",
        ),
        rationale=(
            "Causal triage found a small candidate movement toward Lopez-style Commerce Clause limits "
            "on the school-zone prompt."
        ),
    ),
    PocketSpec(
        pocket_id="EA01_broad_remedy_market",
        prompt_key="EA01_commercial_remedy",
        target_frame=BROAD_FRAME,
        contrast_frame=LIMITS_FRAME,
        target_cases=("raich_2005", "wickard_1942", "perez_1971", "hodel_1981", "heart_atlanta_1964"),
        contrast_cases=("lopez_1995", "morrison_2000", "nfib_2012", "carter_coal_1936", "schechter_1935", "hammer_1918"),
        rationale=(
            "Causal triage found a small candidate movement toward federal commercial-remedy and "
            "aggregate-market reasoning on the commercial-remedy prompt."
        ),
    ),
)


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


def markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(item) for item in row) + " |")
    return "\n".join(lines)


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def clean_source_row(row: dict[str, Any]) -> bool:
    return (
        row.get("frame") in {BROAD_FRAME, LIMITS_FRAME}
        and row.get("expected_frame") == row.get("frame")
        and not bool(row.get("has_multi_frame_conflict"))
        and bool(str(row.get("text_cue_masked") or row.get("text") or "").strip())
    )


def priority_key(row: dict[str, Any], case_order: tuple[str, ...]) -> tuple[int, int, int, str]:
    case_id = str(row.get("source_case_id") or "")
    try:
        case_rank = case_order.index(case_id)
    except ValueError:
        case_rank = len(case_order)
    evidence_count = len(row.get("evidence_patterns") or [])
    token_count = int(row.get("token_count") or 0)
    distance = abs(token_count - 220)
    return (case_rank, -evidence_count, distance, str(row.get("record_id") or ""))


def selected_for_side(
    rows: list[dict[str, Any]],
    *,
    frame: str,
    cases: tuple[str, ...],
    max_per_side: int,
) -> list[dict[str, Any]]:
    candidates = [
        row
        for row in rows
        if row.get("frame") == frame and str(row.get("source_case_id") or "") in set(cases)
    ]
    candidates.sort(key=lambda row: priority_key(row, cases))

    selected: list[dict[str, Any]] = []
    per_case: Counter[str] = Counter()
    for row in candidates:
        case_id = str(row.get("source_case_id") or "")
        if per_case[case_id] >= 3:
            continue
        selected.append(row)
        per_case[case_id] += 1
        if len(selected) >= max_per_side:
            return selected

    for row in candidates:
        if row in selected:
            continue
        selected.append(row)
        if len(selected) >= max_per_side:
            break
    return selected


def adjudication_summary(path: Path) -> list[list[Any]]:
    if not path.exists():
        return [["missing", path, "", "", ""]]
    rows = read_jsonl(path)
    advanced: dict[tuple[str, str, float], Counter[str]] = defaultdict(Counter)
    for row in rows:
        if not row.get("candidate_win"):
            continue
        key = (str(row.get("run_name")), str(row.get("prompt_key")), float(row.get("alpha") or 0.0))
        advanced[key][str(row.get("comparison"))] += 1
    out: list[list[Any]] = []
    for (run_name, prompt_key, alpha), counts in sorted(advanced.items()):
        if prompt_key in {"EA03_gun_school_zone", "EA01_commercial_remedy"}:
            out.append([run_name, prompt_key, alpha, sum(counts.values()), ", ".join(sorted(counts))])
    return out or [["none", "", "", "", ""]]


def review_row(
    *,
    row: dict[str, Any],
    pocket_uses: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "review_id": f"economic_pocket_source::{row['record_id']}",
        "pocket_uses": pocket_uses,
        "source_rule_frame": row.get("frame"),
        "source_expected_frame": row.get("expected_frame"),
        "case_name": row.get("case_name"),
        "source_case_id": row.get("source_case_id"),
        "source_citation": row.get("source_citation"),
        "source_url": row.get("source_url"),
        "chunk_id": row.get("chunk_id"),
        "record_id": row.get("record_id"),
        "evidence_patterns": row.get("evidence_patterns", []),
        "evidence_window": row.get("evidence_window", ""),
        "text": row.get("text", ""),
        "text_cue_masked": row.get("text_cue_masked", ""),
        "review_question": (
            "What is the dominant legal-reasoning frame in this excerpt after ignoring explicit cue words, "
            "case names, and citations?"
        ),
        "allowed_labels": [
            "dominant_broad_commerce",
            "dominant_commerce_limits",
            "dominant_state_federalism",
            "dominant_statutory_or_remedy",
            "mixed_no_dominant_frame",
            "reject_noise_or_boilerplate",
        ],
        "dominant_frame_label": "",
        "review_confidence": "",
        "review_notes": "",
    }


def build_queue(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    labels = [row for row in read_jsonl(args.labels) if clean_source_row(row)]
    prompts = {str(row.get("prompt_key") or row.get("id")): row for row in read_jsonl(args.prompts)}
    rows_by_record = {str(row["record_id"]): row for row in labels}
    pocket_uses_by_record: dict[str, list[dict[str, Any]]] = defaultdict(list)
    per_pocket_counts: Counter[tuple[str, str]] = Counter()
    for spec in POCKETS:
        prompt_meta = prompts[spec.prompt_key]
        target_rows = selected_for_side(
            labels,
            frame=spec.target_frame,
            cases=spec.target_cases,
            max_per_side=args.max_per_side,
        )
        contrast_rows = selected_for_side(
            labels,
            frame=spec.contrast_frame,
            cases=spec.contrast_cases,
            max_per_side=args.max_per_side,
        )
        for row in target_rows:
            pocket_uses_by_record[str(row["record_id"])].append(
                {
                    "pocket_id": spec.pocket_id,
                    "prompt_key": spec.prompt_key,
                    "prompt": prompt_meta.get("prompt", ""),
                    "pocket_rationale": spec.rationale,
                    "expected_side": "target",
                    "target_frame": spec.target_frame,
                    "contrast_frame": spec.contrast_frame,
                }
            )
            per_pocket_counts[(spec.pocket_id, "target")] += 1
        for row in contrast_rows:
            pocket_uses_by_record[str(row["record_id"])].append(
                {
                    "pocket_id": spec.pocket_id,
                    "prompt_key": spec.prompt_key,
                    "prompt": prompt_meta.get("prompt", ""),
                    "pocket_rationale": spec.rationale,
                    "expected_side": "contrast",
                    "target_frame": spec.target_frame,
                    "contrast_frame": spec.contrast_frame,
                }
            )
            per_pocket_counts[(spec.pocket_id, "contrast")] += 1

    queue = [
        review_row(row=rows_by_record[record_id], pocket_uses=uses)
        for record_id, uses in sorted(pocket_uses_by_record.items())
    ]
    unique_frame_counts = Counter(str(row["source_rule_frame"]) for row in queue)

    manifest = {
        "created_at": now_iso(),
        "labels": str(args.labels),
        "prompts": str(args.prompts),
        "rows": len(queue),
        "max_per_side": args.max_per_side,
        "per_pocket_counts": {f"{pocket}:{side}": count for (pocket, side), count in sorted(per_pocket_counts.items())},
        "unique_frame_counts": dict(sorted(unique_frame_counts.items())),
        "filter": (
            "frame in {economic_commerce_broad_aggregation,economic_commerce_limits}; "
            "expected_frame == frame; has_multi_frame_conflict == false"
        ),
    }
    return queue, manifest


def write_report(
    path: Path,
    *,
    args: argparse.Namespace,
    queue: list[dict[str, Any]],
    manifest: dict[str, Any],
) -> None:
    count_rows = [[key, value] for key, value in sorted(manifest["per_pocket_counts"].items())]
    unique_rows = [[key, value] for key, value in sorted(manifest["unique_frame_counts"].items())]
    case_counts = Counter(
        (row["source_rule_frame"], row["source_case_id"], row["case_name"])
        for row in queue
    )
    case_rows = [[frame, case_id, case_name, count] for (frame, case_id, case_name), count in sorted(case_counts.items())]
    lines = [
        "# SCOTUS Economic Pocket Follow-up",
        "",
        "## Purpose",
        "",
        "The broad SCOTUS justice-style direction is decodable but has not passed causal promotion. The only surviving prompt pockets were narrow Economic Activity prompts, so this artifact defines the next cleaner test before spending more BF16 hook time.",
        "",
        "## What changed",
        "",
        "- The original Economic Activity source pack has 50 `expected_frame` versus `frame` mismatches and 120 multi-frame conflicts across 288 rows.",
        "- The stricter cached broad-versus-limits rescore kept 51 clean rows but failed: best activation test BA 0.393 versus cue-masked text test BA 0.679.",
        "- Therefore the current source direction is eliminated. The next branch requires manual dominance labels, not another broad source-frame poke.",
        "",
        "## Queue",
        "",
        markdown_table(
            ["Field", "Value"],
            [
                ["Review queue", args.queue],
                ["Unique review rows", len(queue)],
                ["Labels source", args.labels],
                ["Prompt bank", args.prompts],
                ["Clean cached probe", args.clean_probe],
                ["Filter", manifest["filter"]],
            ],
        ),
        "",
        "## Causal Pocket Evidence To Follow Up",
        "",
        markdown_table(["Run", "Prompt", "Alpha", "Winning comparisons", "Comparisons"], adjudication_summary(args.adjudication)),
        "",
        "## Review Counts",
        "",
        markdown_table(["Unique source frame", "N"], unique_rows),
        "",
        "## Planned Pocket Coverage",
        "",
        markdown_table(["Pocket side", "N before dominance review"], count_rows),
        "",
        "## Case Coverage",
        "",
        markdown_table(["Frame", "Case id", "Case", "N"], case_rows),
        "",
        "## Review Instructions",
        "",
        "For each row, assign `dominant_frame_label` using only the substance of the excerpt. Ignore explicit cue words, case names, and citations where possible because those are the easiest leakage path.",
        "",
        "Allowed labels:",
        "",
        "- `dominant_broad_commerce`: aggregate effects, national market, channels/instrumentalities, comprehensive federal scheme, or broad deference to congressional economic regulation.",
        "- `dominant_commerce_limits`: non-economic/local conduct, missing jurisdictional element, attenuated causal chain, activity/inactivity, direct/indirect production limit, or no general police power.",
        "- `dominant_state_federalism`: state sovereignty, anti-commandeering, reserved powers, or state regulatory authority is the main frame rather than Commerce Clause scope.",
        "- `dominant_statutory_or_remedy`: statutory interpretation, remedial design, damages, preemption, or FAA-like analysis dominates.",
        "- `mixed_no_dominant_frame`: both sides are genuinely present and neither dominates.",
        "- `reject_noise_or_boilerplate`: syllabus debris, citation string, procedural noise, or otherwise unusable.",
        "",
        "## Promotion Gate",
        "",
        "1. After review, keep only rows labeled `dominant_broad_commerce` or `dominant_commerce_limits` with medium/high confidence.",
        "2. Require at least 20 rows per side and at least 3 source cases per side before any BF16 activation capture.",
        "3. Run a source-case-heldout activation probe on cue-masked text and require test BA to beat the cue-masked text baseline by at least 0.05.",
        "4. If that passes, run causal generation only on the two pocket prompts with prompt-matched random controls, and promote only if the candidate beats baseline, random mean, and strongest random under blind review.",
        "",
        "## Decision",
        "",
        "Do not run another broad justice-style or broad Economic Activity poke from the existing labels. The next experiment is this dominance-reviewed pocket contrast.",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--prompts", type=Path, default=DEFAULT_PROMPTS)
    parser.add_argument("--adjudication", type=Path, default=DEFAULT_ADJUDICATION)
    parser.add_argument("--clean-probe", type=Path, default=DEFAULT_CLEAN_PROBE)
    parser.add_argument("--queue", type=Path, default=DEFAULT_QUEUE)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--max-per-side", type=int, default=24)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    queue, manifest = build_queue(args)
    write_jsonl(args.queue, queue)
    manifest_path = args.queue.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_report(args.report, args=args, queue=queue, manifest=manifest)
    print(f"Wrote {len(queue)} review rows to {args.queue}")
    print(f"Wrote report to {args.report}")


if __name__ == "__main__":
    main()
