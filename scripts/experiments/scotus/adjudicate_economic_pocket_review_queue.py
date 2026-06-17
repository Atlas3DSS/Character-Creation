#!/usr/bin/env python3
"""Apply an internal dominance adjudication to the Economic pocket queue."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SCOTUS_DIR = PROJECT_ROOT / "data" / "scotus"
DEFAULT_QUEUE = SCOTUS_DIR / "scotus_economic_pocket_dominance_review_20260501.jsonl"
DEFAULT_OUTPUT = SCOTUS_DIR / "scotus_economic_pocket_dominance_adjudicated_20260501.jsonl"
DEFAULT_REPORT = PROJECT_ROOT / "reports" / "scotus_economic_pocket_dominance_adjudication_20260501.md"

BROAD = "dominant_broad_commerce"
LIMITS = "dominant_commerce_limits"
STATE = "dominant_state_federalism"
REJECT = "reject_noise_or_boilerplate"


LABELS: dict[str, tuple[str, str, str]] = {
    "carter_coal_1936-0012::economic_commerce_limits": (LIMITS, "medium", "Local coal production is framed as outside commerce despite later distribution."),
    "carter_coal_1936-0049::economic_commerce_limits": (LIMITS, "high", "Production is treated as local and not commerce."),
    "carter_coal_1936-0055::economic_commerce_limits": (LIMITS, "high", "Direct/indirect effects distinction is the dominant frame."),
    "carter_coal_1936-0056::economic_commerce_limits": (LIMITS, "high", "Indirect-effect reasoning dominates."),
    "champion_1903-0017::economic_commerce_broad_aggregation": (BROAD, "high", "Interstate communication is treated as commerce under national power."),
    "champion_1903-0022::economic_commerce_broad_aggregation": (BROAD, "high", "State toll regulation is displaced by interstate-commerce authority."),
    "champion_1903-0044::economic_commerce_broad_aggregation": (BROAD, "high", "Congress has plenary authority over interstate carriage of lottery tickets."),
    "gibbons_1824-0029::economic_commerce_broad_aggregation": (STATE, "medium", "The excerpt is mainly about state health laws and state/federal boundaries."),
    "heart_atlanta_1964-0014::economic_commerce_broad_aggregation": (BROAD, "high", "Commerce power is a valid source for regulating discriminatory practices affecting commerce."),
    "heart_atlanta_1964-0023::economic_commerce_broad_aggregation": (BROAD, "high", "Congressional reach over activities burdening interstate commerce dominates."),
    "heart_atlanta_1964-0025::economic_commerce_broad_aggregation": (BROAD, "high", "Local operation can be regulated when interstate commerce feels the burden."),
    "heart_atlanta_1964-0050::economic_commerce_broad_aggregation": (BROAD, "medium", "The excerpt has a limits caveat but resolves through aggregation and congressional power."),
    "hodel_1981-0014::economic_commerce_broad_aggregation": (BROAD, "high", "Rational-basis deference to congressional Commerce Clause findings dominates."),
    "hodel_1981-0015::economic_commerce_broad_aggregation": (BROAD, "high", "Plenary commerce power and activities-affecting-commerce doctrine dominate."),
    "hodel_1981-0022::economic_commerce_broad_aggregation": (BROAD, "high", "Intrastate/local labels do not defeat federal regulation when interstate commerce is affected."),
    "hodel_1981-0024::economic_commerce_broad_aggregation": (BROAD, "high", "Federal regulation prevents destructive interstate competition."),
    "hodel_1981-0063::economic_commerce_broad_aggregation": (BROAD, "high", "Broad commerce power and close/substantial relation reasoning dominate."),
    "hodel_1981-0064::economic_commerce_broad_aggregation": (BROAD, "high", "Cumulative national-market effects under Wickard dominate."),
    "katzenbach_mcclung_1964-0012::economic_commerce_broad_aggregation": (BROAD, "high", "Substantial economic effect on interstate commerce dominates."),
    "lopez_1995-0011::economic_commerce_limits": (LIMITS, "high", "Direct/indirect and no-limit-to-federal-power reasoning dominates."),
    "lopez_1995-0048::economic_commerce_limits": (LIMITS, "high", "Historical direct/indirect limits examples dominate."),
    "lopez_1995-0113::economic_commerce_limits": (BROAD, "medium", "The excerpt describes abandonment of formal direct/indirect limits and affirms congressional power."),
    "morrison_2000-0015::economic_commerce_limits": (LIMITS, "high", "Outer limits and national/local distinction dominate."),
    "morrison_2000-0018::economic_commerce_limits": (LIMITS, "medium", "Broad precedents are invoked to set up the economic/non-economic limiting rule."),
    "morrison_2000-0019::economic_commerce_limits": (LIMITS, "high", "Noneconomic conduct is central to the Commerce Clause limit."),
    "morrison_2000-0021::economic_commerce_limits": (LIMITS, "high", "Economic endeavor and jurisdictional-element limits dominate."),
    "morrison_2000-0022::economic_commerce_limits": (LIMITS, "high", "Attenuated causal-chain reasoning dominates."),
    "morrison_2000-0024::economic_commerce_limits": (LIMITS, "high", "Noneconomic activity and absent jurisdictional element dominate."),
    "morrison_2000-0082::economic_commerce_limits": (LIMITS, "medium", "The dissent discusses Commerce Clause limits as the operative frame."),
    "morrison_2000-0085::economic_commerce_limits": (BROAD, "medium", "The dissent rejects economic/non-economic limits and emphasizes interstate effects."),
    "morrison_2000-0105::economic_commerce_limits": (LIMITS, "high", "But-for causal-chain limits dominate."),
    "nfib_2012-0113::economic_commerce_limits": (LIMITS, "medium", "Activity/inactivity and categorical commerce limits dominate despite critical valence."),
    "nfib_2012-0114::economic_commerce_limits": (LIMITS, "medium", "Activity/inactivity line-drawing dominates despite critical valence."),
    "nfib_2012-0117::economic_commerce_limits": (LIMITS, "high", "Noneconomic attenuated conduct outside federal commerce power dominates."),
    "nfib_2012-0121::economic_commerce_limits": (BROAD, "high", "Necessary part of a broader economic regulatory program dominates."),
    "nfib_2012-0245::economic_commerce_limits": (LIMITS, "high", "Inactivity as beyond commerce power dominates."),
    "perez_1971-0007::economic_commerce_broad_aggregation": (BROAD, "high", "Channels, instrumentalities, and activities affecting commerce dominate."),
    "perez_1971-0009::economic_commerce_broad_aggregation": (BROAD, "high", "Broad restored Commerce Clause view dominates."),
    "perez_1971-0011::economic_commerce_broad_aggregation": (BROAD, "high", "Class-of-activities regulation without particularized proof dominates."),
    "perez_1971-0012::economic_commerce_broad_aggregation": (BROAD, "high", "Per se commerce effect and sustained exercise of power dominate."),
    "raich_2005-0018::economic_commerce_broad_aggregation": (BROAD, "high", "Home-consumption production affects national market supply and demand."),
    "raich_2005-0019::economic_commerce_broad_aggregation": (BROAD, "high", "Local production is aggregated into a national market."),
    "raich_2005-0022::economic_commerce_broad_aggregation": (LIMITS, "medium", "The excerpt primarily states the Lopez invalidity and non-economic-activity limit."),
    "schechter_1935-0041::economic_commerce_limits": (LIMITS, "high", "Direct versus indirect effects dominate."),
    "schechter_1935-0043::economic_commerce_limits": (LIMITS, "high", "Indirect intrastate effects do not trigger federal commerce power."),
    "shreveport_1914-0001::economic_commerce_broad_aggregation": (REJECT, "high", "Procedural posture and party description dominate, not a usable frame."),
    "stafford_1922-0020::economic_commerce_broad_aggregation": (BROAD, "high", "Stream/current of commerce and national regulation dominate."),
    "wickard_1942-0009::economic_commerce_broad_aggregation": (BROAD, "high", "The excerpt rejects production/indirect formulas and looks to actual economic effects."),
    "wickard_1942-0010::economic_commerce_broad_aggregation": (BROAD, "medium", "Historical breadth of federal commerce power dominates despite state-sovereignty background."),
    "wickard_1942-0013::economic_commerce_broad_aggregation": (BROAD, "high", "Close and substantial relation of intrastate rates to interstate traffic dominates."),
    "wickard_1942-0014::economic_commerce_broad_aggregation": (BROAD, "high", "Economic-effects measure and intrastate effects doctrine dominate."),
}


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


def adjudicate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    missing = sorted({str(row["record_id"]) for row in rows} - set(LABELS))
    extra = sorted(set(LABELS) - {str(row["record_id"]) for row in rows})
    if missing or extra:
        raise RuntimeError(f"Label map mismatch: missing={missing}, extra={extra}")
    out: list[dict[str, Any]] = []
    for row in rows:
        label, confidence, notes = LABELS[str(row["record_id"])]
        reviewed = dict(row)
        reviewed["dominant_frame_label"] = label
        reviewed["review_confidence"] = confidence
        reviewed["review_notes"] = notes
        reviewed["reviewer"] = "internal_codex_adjudication"
        reviewed["reviewed_at"] = now_iso()
        out.append(reviewed)
    return out


def write_report(path: Path, *, queue_path: Path, output_path: Path, rows: list[dict[str, Any]]) -> None:
    label_counts = Counter(row["dominant_frame_label"] for row in rows)
    source_vs_review = Counter((row["source_rule_frame"], row["dominant_frame_label"]) for row in rows)
    usable = [row for row in rows if row["dominant_frame_label"] in {BROAD, LIMITS} and row["review_confidence"] in {"medium", "high"}]
    usable_counts = Counter(row["dominant_frame_label"] for row in usable)
    usable_cases = Counter((row["dominant_frame_label"], row["source_case_id"], row["case_name"]) for row in usable)
    gate_pass = usable_counts[BROAD] >= 20 and usable_counts[LIMITS] >= 20
    gate_pass = gate_pass and len({row["source_case_id"] for row in usable if row["dominant_frame_label"] == BROAD}) >= 3
    gate_pass = gate_pass and len({row["source_case_id"] for row in usable if row["dominant_frame_label"] == LIMITS}) >= 3
    lines = [
        "# SCOTUS Economic Pocket Dominance Adjudication",
        "",
        "## Purpose",
        "",
        "This is an internal dominance review of the clean Economic Activity source excerpts selected for the two surviving causal prompt pockets. It is not an independent blind human review, but it is stricter than the original regex source labels.",
        "",
        "## Inputs",
        "",
        markdown_table(
            ["Field", "Value"],
            [
                ["Queue", queue_path],
                ["Adjudicated rows", output_path],
                ["Rows", len(rows)],
            ],
        ),
        "",
        "## Label Counts",
        "",
        markdown_table(["Dominant label", "N"], [[label, count] for label, count in sorted(label_counts.items())]),
        "",
        "## Source Rule Versus Review",
        "",
        markdown_table(["Source rule frame", "Reviewed label", "N"], [[source, label, count] for (source, label), count in sorted(source_vs_review.items())]),
        "",
        "## Usable Binary Counts",
        "",
        markdown_table(["Reviewed label", "N"], [[label, usable_counts[label]] for label in [BROAD, LIMITS]]),
        "",
        "## Usable Case Coverage",
        "",
        markdown_table(
            ["Reviewed label", "Case id", "Case", "N"],
            [[label, case_id, case_name, count] for (label, case_id, case_name), count in sorted(usable_cases.items())],
        ),
        "",
        "## Gate",
        "",
        f"Data gate status: `{'pass' if gate_pass else 'fail'}`.",
        "",
        "The gate requires at least 20 usable broad-Commerce rows, at least 20 usable Commerce-limits rows, and at least 3 source cases per side. Passing this gate permits a cached reviewed-label probe before any new BF16 activation capture.",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", type=Path, default=DEFAULT_QUEUE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = read_jsonl(args.queue)
    reviewed = adjudicate(rows)
    write_jsonl(args.output, reviewed)
    write_report(args.report, queue_path=args.queue, output_path=args.output, rows=reviewed)
    print(f"Wrote {len(reviewed)} adjudicated rows to {args.output}")
    print(f"Wrote report to {args.report}")


if __name__ == "__main__":
    main()
