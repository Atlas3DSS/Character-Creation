#!/usr/bin/env python3
"""Apply single-pass Article III dominance adjudications to the review queue."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_KEY = PROJECT_ROOT / "data" / "scotus" / "scotus_article3_dominance_review_key_v1.jsonl"
DEFAULT_REVIEWED = PROJECT_ROOT / "data" / "scotus" / "scotus_article3_dominance_review_adjudicated_v1.jsonl"
DEFAULT_LABELS = PROJECT_ROOT / "data" / "scotus" / "scotus_article3_dominance_frame_labels_v1.jsonl"
DEFAULT_REPORT = PROJECT_ROOT / "reports" / "scotus_article3_dominance_adjudication_v1.md"

LABEL_TO_FRAME = {
    "public_rights_dominant": "article3_public_rights",
    "private_rights_dominant": "article3_private_rights",
    "article1_tribunal_dominant": "article3_article1_tribunal",
}

ANNOTATIONS: dict[str, tuple[str, str, str]] = {
    "article3_dominance::wellness_v_sharif_2015-0068": (
        "mixed_comparative",
        "medium",
        "Foundational public/private contrast without a single adopted application.",
    ),
    "article3_dominance::wellness_v_sharif_2015-0069": (
        "private_rights_dominant",
        "high",
        "Says federal courts alone can conclusively deprive individuals of private rights.",
    ),
    "article3_dominance::wellness_v_sharif_2015-0071": (
        "private_rights_dominant",
        "medium",
        "Frames the core Article III judicial power as adjudication of private-rights disputes.",
    ),
    "article3_dominance::cftc_v_schor_1986-0040": (
        "private_rights_dominant",
        "medium",
        "Emphasizes state-law/private claims as normally reserved to Article III courts.",
    ),
    "article3_dominance::thomas_union_carbide_1985-0003": (
        "public_rights_dominant",
        "high",
        "Rejects the asserted private-right characterization and treats the scheme as public/regulatory.",
    ),
    "article3_dominance::sec_v_jarkesy_2024-0090": (
        "mixed_comparative",
        "medium",
        "Sets up the public/private distinction for Article III and Seventh Amendment analysis.",
    ),
    "article3_dominance::sec_v_jarkesy_2024-0006": (
        "mixed_comparative",
        "medium",
        "States both private-right presumption and public-right categories before applying them.",
    ),
    "article3_dominance::axon_v_ftc_2023-0034": (
        "private_rights_dominant",
        "high",
        "Defines core private rights and says they likely require full Article III adjudication.",
    ),
    "article3_dominance::axon_v_ftc_2023-0035": (
        "mixed_comparative",
        "medium",
        "Contrasts public privileges with private rights and administrative-adjudication drift.",
    ),
    "article3_dominance::granfinanciera_1989-0002": (
        "mixed_comparative",
        "medium",
        "Syllabus-level public/private test rather than a dominant application.",
    ),
    "article3_dominance::cftc_v_schor_1986-0039": (
        "private_rights_dominant",
        "high",
        "Identifies the counterclaim as a private right at the Article III core.",
    ),
    "article3_dominance::sec_v_jarkesy_2024-0066": (
        "private_rights_dominant",
        "high",
        "Describes Crowell as chipping away at the courts' role over private rights.",
    ),
    "article3_dominance::granfinanciera_1989-0046": (
        "public_rights_dominant",
        "medium",
        "Focuses on defining the scope of public rights and limiting it to government-party matters.",
    ),
    "article3_dominance::wellness_v_sharif_2015-0073": (
        "private_rights_dominant",
        "high",
        "Treats Stern claims as private rights and asks whether consent lifts that private-right bar.",
    ),
    "article3_dominance::oil_states_2018-0029": (
        "public_rights_dominant",
        "high",
        "States inter partes review involves public rights, with private rights discussed as a caveat.",
    ),
    "article3_dominance::wellness_v_sharif_2015-0072": (
        "article1_tribunal_dominant",
        "medium",
        "Dominant discussion is bankruptcy courts and the bankruptcy exception to Article III.",
    ),
    "article3_dominance::axon_v_ftc_2023-0041": (
        "private_rights_dominant",
        "high",
        "Applies private-rights reasoning to fines, property transfer, and Article III/jury requirements.",
    ),
    "article3_dominance::oil_states_2018-0012": (
        "public_rights_dominant",
        "medium",
        "Frames patent review through the public/private-rights inquiry, with public-rights application following.",
    ),
    "article3_dominance::axon_v_ftc_2023-0033": (
        "mixed_comparative",
        "medium",
        "Historical contrast between private rights and public rights drives the passage.",
    ),
    "article3_dominance::granfinanciera_1989-0053": (
        "public_rights_dominant",
        "medium",
        "Critiques expansion of the public-rights doctrine but mainly discusses that doctrine's scope.",
    ),
    "article3_dominance::thomas_union_carbide_1985-0031": (
        "public_rights_dominant",
        "medium",
        "Rejects a bright-line private-rights test and supports Article III flexibility.",
    ),
    "article3_dominance::wellness_v_sharif_2015-0070": (
        "mixed_comparative",
        "medium",
        "Historical line-blurring between public and private rights is the central point.",
    ),
    "article3_dominance::stern_v_marshall_2011-0047": (
        "public_rights_dominant",
        "medium",
        "Uses Thomas to explain a statutory/regulatory right that does not depend on state law.",
    ),
    "article3_dominance::sec_v_jarkesy_2024-0005": (
        "private_rights_dominant",
        "high",
        "Fraud/common-law character defeats the public-rights exception.",
    ),
    "article3_dominance::wellness_v_sharif_2015-0065": (
        "mixed_comparative",
        "medium",
        "Introduces the rights-type distinction without completing a dominant application in the excerpt.",
    ),
    "article3_dominance::sec_v_jarkesy_2024-0030": (
        "public_rights_dominant",
        "medium",
        "Lists recognized historic public-rights categories.",
    ),
    "article3_dominance::thomas_union_carbide_1985-0030": (
        "public_rights_dominant",
        "medium",
        "Presents and rejects the private-right argument in favor of public-rights flexibility.",
    ),
    "article3_dominance::granfinanciera_1989-0047": (
        "public_rights_dominant",
        "medium",
        "Catalogues public-rights definitions from prior cases.",
    ),
    "article3_dominance::axon_v_ftc_2023-0032": (
        "private_rights_dominant",
        "high",
        "Opening thesis questions agency adjudication of core private rights.",
    ),
    "article3_dominance::sec_v_jarkesy_2024-0045": (
        "private_rights_dominant",
        "medium",
        "Argues practice cannot transmute private rights into public ones.",
    ),
    "article3_dominance::axon_v_ftc_2023-0043": (
        "private_rights_dominant",
        "medium",
        "Notes center on vested/private property rights and limits on agency adjudication.",
    ),
    "article3_dominance::axon_v_ftc_2023-0039": (
        "private_rights_dominant",
        "high",
        "Due process and Article III concerns are framed around core private rights.",
    ),
    "article3_dominance::axon_v_ftc_2023-0040": (
        "private_rights_dominant",
        "high",
        "Article III is described as protecting individualized adjudicative facts for core private rights.",
    ),
    "article3_dominance::wellness_v_sharif_2015-0079": (
        "private_rights_dominant",
        "high",
        "Collects authority that private rights are judicial and must be handled by courts.",
    ),
    "article3_dominance::granfinanciera_1989-0036": (
        "private_rights_dominant",
        "high",
        "Fraudulent conveyance is characterized as private rather than public.",
    ),
    "article3_dominance::granfinanciera_1989-0107": (
        "private_rights_dominant",
        "medium",
        "Discusses private-rights factfinding limits and Article III adjunct constraints.",
    ),
    "article3_dominance::sec_v_jarkesy_2024-0027": (
        "private_rights_dominant",
        "high",
        "States matters concerning private rights may not be removed from Article III courts.",
    ),
    "article3_dominance::stern_v_marshall_2011-0055": (
        "private_rights_dominant",
        "high",
        "Maps Northern Pipeline's contract rule onto Stern's tort/counterclaim posture.",
    ),
    "article3_dominance::oil_states_2018-0024": (
        "public_rights_dominant",
        "high",
        "Holds inter partes review remains a public-rights matter.",
    ),
    "article3_dominance::stern_v_marshall_2011-0084": (
        "private_rights_dominant",
        "medium",
        "Follows Granfinanciera's approach where fraudulent conveyance is not assignable as public right.",
    ),
    "article3_dominance::northern_pipeline_1982-0025": (
        "article1_tribunal_dominant",
        "high",
        "Dominant focus is historical exceptions for legislative courts and agencies.",
    ),
    "article3_dominance::northern_pipeline_1982-0002": (
        "article1_tribunal_dominant",
        "high",
        "Syllabus-level holding about non-Article III bankruptcy courts and Article III limits.",
    ),
    "article3_dominance::granfinanciera_1989-0034": (
        "public_rights_dominant",
        "medium",
        "Explains the expanded Thomas public-rights regulatory-scheme test.",
    ),
    "article3_dominance::stern_v_marshall_2011-0036": (
        "article1_tribunal_dominant",
        "medium",
        "Focuses on whether bankruptcy courts can act as adjuncts of Article III courts.",
    ),
    "article3_dominance::sec_v_jarkesy_2024-0106": (
        "public_rights_dominant",
        "medium",
        "Critiques the majority's account of public-rights cases and doctrine.",
    ),
    "article3_dominance::granfinanciera_1989-0032": (
        "public_rights_dominant",
        "medium",
        "Frames assignability to non-Article III tribunals through the public-rights inquiry.",
    ),
    "article3_dominance::cftc_v_schor_1986-0050": (
        "article1_tribunal_dominant",
        "high",
        "Dominant focus is exceptions permitting non-Article III federal tribunals.",
    ),
    "article3_dominance::thomas_union_carbide_1985-0034": (
        "public_rights_dominant",
        "medium",
        "Defends public-rights doctrine against a party-identity bright line.",
    ),
    "article3_dominance::stern_v_marshall_2011-0040": (
        "private_rights_dominant",
        "high",
        "Rejects the public-rights exception for an independent state-law action.",
    ),
    "article3_dominance::atlas_roofing_1977-0039": (
        "public_rights_dominant",
        "high",
        "Discusses adjudication of public rights by administrative tribunals with judicial review.",
    ),
    "article3_dominance::northern_pipeline_1982-0077": (
        "article1_tribunal_dominant",
        "high",
        "Dominant focus is categories of Article I/non-Article III courts.",
    ),
    "article3_dominance::atlas_roofing_1977-0020": (
        "public_rights_dominant",
        "high",
        "States Congress may assign new statutory public rights to agencies.",
    ),
    "article3_dominance::thomas_union_carbide_1985-0055": (
        "public_rights_dominant",
        "high",
        "Characterizes the FIFRA compensation scheme as a matter of public rights.",
    ),
    "article3_dominance::stern_v_marshall_2011-0035": (
        "public_rights_dominant",
        "medium",
        "Summarizes the Northern Pipeline public-rights exception.",
    ),
    "article3_dominance::atlas_roofing_1977-0002": (
        "public_rights_dominant",
        "high",
        "Syllabus holding that OSHA adjudication involves statutory public rights.",
    ),
    "article3_dominance::oil_states_2018-0015": (
        "public_rights_dominant",
        "high",
        "Patent grant is treated as a constitutional function/public-rights matter.",
    ),
    "article3_dominance::atlas_roofing_1977-0013": (
        "public_rights_dominant",
        "high",
        "Directly states public-rights cases may be assigned to administrative factfinding.",
    ),
    "article3_dominance::stern_v_marshall_2011-0045": (
        "mixed_comparative",
        "medium",
        "Contrasts public-rights cases with private tort/contract/property cases.",
    ),
    "article3_dominance::stern_v_marshall_2011-0056": (
        "private_rights_dominant",
        "high",
        "Describes the claim as common-law judicial power not saved by public-rights framing.",
    ),
    "article3_dominance::northern_pipeline_1982-0116": (
        "public_rights_dominant",
        "medium",
        "Explains the scope and limits of public-rights doctrine.",
    ),
    "article3_dominance::oil_states_2018-0040": (
        "private_rights_dominant",
        "medium",
        "Distinguishes issuing public franchises from revoking vested patent/property rights.",
    ),
    "article3_dominance::thomas_union_carbide_1985-0053": (
        "public_rights_dominant",
        "medium",
        "Discusses limits of but continued room for public-rights adjudication.",
    ),
    "article3_dominance::oil_states_2018-0002": (
        "public_rights_dominant",
        "high",
        "Syllabus states inter partes review falls squarely within the public-rights doctrine.",
    ),
    "article3_dominance::axon_v_ftc_2023-0038": (
        "private_rights_dominant",
        "high",
        "Raises agency power to adjudicate core private rights as the main concern.",
    ),
    "article3_dominance::axon_v_ftc_2023-0037": (
        "private_rights_dominant",
        "medium",
        "Crowell is framed as a private-right case handled through agency factfinding.",
    ),
    "article3_dominance::wellness_v_sharif_2015-0038": (
        "article1_tribunal_dominant",
        "high",
        "Dominant focus is non-Article III bankruptcy and other Article III exceptions.",
    ),
    "article3_dominance::sec_v_jarkesy_2024-0032": (
        "private_rights_dominant",
        "medium",
        "Granfinanciera/common-law fraud discussion drives an Article III/non-Article III limit.",
    ),
    "article3_dominance::wellness_v_sharif_2015-0066": (
        "article1_tribunal_dominant",
        "high",
        "Territorial courts, courts-martial, and bankruptcy exceptions dominate.",
    ),
    "article3_dominance::granfinanciera_1989-0030": (
        "public_rights_dominant",
        "medium",
        "Describes when statutory public rights may be assigned outside jury adjudication.",
    ),
    "article3_dominance::sec_v_jarkesy_2024-0046": (
        "public_rights_dominant",
        "medium",
        "Dominant topic is criticism of Atlas Roofing/public-rights doctrine.",
    ),
    "article3_dominance::granfinanciera_1989-0057": (
        "private_rights_dominant",
        "medium",
        "Applies a narrow public-rights rule to conclude the Article III/jury right cannot be eliminated.",
    ),
    "article3_dominance::sec_v_jarkesy_2024-0072": (
        "article1_tribunal_dominant",
        "medium",
        "Dominant issue is limiting non-Article III tribunal authority to historical exceptions.",
    ),
    "article3_dominance::granfinanciera_1989-0048": (
        "private_rights_dominant",
        "high",
        "Rejects assigning legal controversies between private parties to non-Article III tribunals.",
    ),
    "article3_dominance::sec_v_jarkesy_2024-0111": (
        "public_rights_dominant",
        "medium",
        "Dissent explains public-right identification for government sovereign-capacity cases.",
    ),
    "article3_dominance::sec_v_jarkesy_2024-0132": (
        "public_rights_dominant",
        "medium",
        "Dissent defends broader public-rights category and examples.",
    ),
    "article3_dominance::granfinanciera_1989-0035": (
        "public_rights_dominant",
        "medium",
        "States that rights not closely intertwined with a regulatory program require Article III.",
    ),
    "article3_dominance::granfinanciera_1989-0123": (
        "mixed_comparative",
        "low",
        "Balancing discussion mentions public rights but is not a clean dominant-frame excerpt.",
    ),
    "article3_dominance::sec_v_jarkesy_2024-0038": (
        "public_rights_dominant",
        "high",
        "Atlas Roofing/OSHA is presented as a novel statutory public-rights example.",
    ),
    "article3_dominance::granfinanciera_1989-0091": (
        "public_rights_dominant",
        "medium",
        "Dissent frames bankruptcy assignment through whether the claim is a public right.",
    ),
    "article3_dominance::sec_v_jarkesy_2024-0095": (
        "public_rights_dominant",
        "high",
        "Lists agency assignments previously approved as public-rights adjudication.",
    ),
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


def markdown_table(headers: list[str], rows: list[list[Any]]) -> list[str]:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(item) for item in row) + " |")
    return lines


def apply_annotations(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    reviewed: list[dict[str, Any]] = []
    missing = sorted({str(row["review_id"]) for row in rows} - set(ANNOTATIONS))
    extra = sorted(set(ANNOTATIONS) - {str(row["review_id"]) for row in rows})
    if missing or extra:
        raise ValueError(f"Annotation mismatch. missing={missing[:5]} extra={extra[:5]}")

    reviewed_at = datetime.now().astimezone().isoformat(timespec="seconds")
    for row in rows:
        label, confidence, notes = ANNOTATIONS[str(row["review_id"])]
        updated = dict(row)
        updated["review_label"] = label
        updated["review_confidence"] = confidence
        updated["review_notes"] = notes
        updated["reviewer"] = "codex_single_pass"
        updated["review_protocol"] = "dominant_legal_frame_v1"
        updated["reviewed_at"] = reviewed_at
        reviewed.append(updated)
    return reviewed


def build_frame_labels(reviewed: list[dict[str, Any]]) -> list[dict[str, Any]]:
    labels: list[dict[str, Any]] = []
    for row in reviewed:
        frame = LABEL_TO_FRAME.get(str(row["review_label"]))
        if frame is None:
            continue
        labels.append(
            {
                "record_id": f"{row['chunk_id']}::{frame}::adjudicated",
                "frame": frame,
                "issue_family": "Judicial Power",
                "label": 1,
                "label_source": "single_reviewer_dominance_v1",
                "label_confidence": row["review_confidence"],
                "label_definition": row["review_notes"],
                "review_id": row["review_id"],
                "review_label": row["review_label"],
                "reviewer": row["reviewer"],
                "split": row.get("split", "unassigned"),
                "global_split": row.get("global_split", row.get("split", "unassigned")),
                "opinion_id": row.get("source_case_id", ""),
                "cluster_id": row.get("source_case_id", ""),
                "case_name": row.get("case_name", ""),
                "date_filed": "",
                "term": row.get("term", ""),
                "justice": "source_opinion",
                "section_author": "",
                "section_posture": "source_opinion",
                "issue_area_label": "Judicial Power",
                "source_url": row.get("source_url", ""),
                "chunk_id": row.get("chunk_id", ""),
                "token_count": row.get("token_count", 0),
                "source_case_id": row.get("source_case_id", ""),
                "source_citation": row.get("source_citation", ""),
                "text": row.get("text", ""),
                "text_cue_masked": row.get("text_cue_masked", row.get("text", "")),
                "evidence_window": row.get("text", "")[:900],
                "matched_frames_before_review": row.get("matched_frames", []),
                "review_status": "single_reviewed",
            }
        )
    labels.sort(key=lambda item: (item["frame"], str(item["case_name"]), str(item["chunk_id"])))
    return labels


def write_report(path: Path, reviewed: list[dict[str, Any]], labels: list[dict[str, Any]], args: argparse.Namespace) -> None:
    label_counts = Counter(str(row["review_label"]) for row in reviewed)
    confidence_counts = Counter(str(row["review_confidence"]) for row in reviewed)
    frame_counts = Counter(str(row["frame"]) for row in labels)
    case_frame_counts = Counter((str(row["case_name"]), str(row["frame"])) for row in labels)
    mixed_or_rejected = label_counts["mixed_comparative"] + label_counts["off_target_or_false_positive"]

    lines = [
        "# SCOTUS Article III Dominance Adjudication v1",
        "",
        "## Purpose",
        "",
        "This applies a single-pass dominance review to the Article III source queue. These labels are more useful than keyword labels, but they are not gold labels; they need a second review before any steering claim.",
        "",
        "## Outputs",
        "",
        f"- Reviewed queue: `{args.reviewed}`",
        f"- Probe-ready labels: `{args.labels}`",
        f"- Source key queue: `{args.key}`",
        f"- Reviewed rows: `{len(reviewed)}`",
        f"- Probe-ready rows: `{len(labels)}`",
        f"- Mixed/rejected rows excluded from probe labels: `{mixed_or_rejected}`",
        "",
        "## Review Label Counts",
        "",
    ]
    lines.extend(markdown_table(["Review label", "Rows"], [[label, count] for label, count in sorted(label_counts.items())]))
    lines.extend(["", "## Confidence Counts", ""])
    lines.extend(markdown_table(["Confidence", "Rows"], [[label, count] for label, count in sorted(confidence_counts.items())]))
    lines.extend(["", "## Probe-Ready Frame Counts", ""])
    lines.extend(markdown_table(["Frame", "Rows"], [[frame, count] for frame, count in sorted(frame_counts.items())]))
    lines.extend(["", "## Case/Frame Coverage", ""])
    lines.extend(
        markdown_table(
            ["Case", "Frame", "Rows"],
            [[case, frame, count] for (case, frame), count in sorted(case_frame_counts.items())],
        )
    )
    lines.extend(
        [
            "",
            "## Use Rules",
            "",
            "1. Treat these as `single_reviewer_dominance_v1`, not final gold labels.",
            "2. Rerun cue-masked probes from the probe-ready label file before any causal generation.",
            "3. Do not promote a direction unless it survives text-baseline checks and prompt-matched same-layer random controls.",
            "4. Rows labeled `mixed_comparative` are useful for evaluator training but should not be used as binary public/private labels.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--key", type=Path, default=DEFAULT_KEY)
    parser.add_argument("--reviewed", type=Path, default=DEFAULT_REVIEWED)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = read_jsonl(args.key)
    reviewed = apply_annotations(rows)
    labels = build_frame_labels(reviewed)
    write_jsonl(args.reviewed, reviewed)
    write_jsonl(args.labels, labels)
    write_report(args.report, reviewed, labels, args)
    print(f"Wrote reviewed rows: {args.reviewed} ({len(reviewed)})")
    print(f"Wrote probe labels: {args.labels} ({len(labels)})")
    print(f"Wrote report: {args.report}")


if __name__ == "__main__":
    main()
