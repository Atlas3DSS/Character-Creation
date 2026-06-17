#!/usr/bin/env python3
"""Build an Administrative Law major-questions / deference source-frame pack."""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SCOTUS_DIR = PROJECT_ROOT / "data" / "scotus"
DEFAULT_RAW = SCOTUS_DIR / "raw" / "scotus_admin_source_pages_v1.json"
DEFAULT_LABELS = SCOTUS_DIR / "scotus_admin_source_frame_labels_v1.jsonl"
DEFAULT_QUEUE = SCOTUS_DIR / "scotus_admin_source_frame_review_queue_v1.jsonl"
DEFAULT_REPORT = PROJECT_ROOT / "reports" / "scotus_admin_source_pack_v1.md"

if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_economic_source_pack import (  # noqa: E402
    clean_plain_text,
    extract_law_cornell_text,
    mask_citations,
    paragraph_chunks,
    token_count,
)
from build_source_frame_labels import assign_frame_splits, markdown_table, stable_split, token_window  # noqa: E402


@dataclass(frozen=True)
class SourceCase:
    case_id: str
    case_name: str
    citation: str
    term: int
    expected_frame: str
    expected_signal: str
    url: str


@dataclass(frozen=True)
class AdminRule:
    frame: str
    definition: str
    required_any: tuple[tuple[str, ...], ...]
    evidence_patterns: tuple[str, ...]


ADMIN_RULES: tuple[AdminRule, ...] = (
    AdminRule(
        frame="admin_major_questions",
        definition=(
            "Major-questions or clear-statement reasoning limiting agency power over questions "
            "of major economic, political, or national significance."
        ),
        required_any=(
            (
                r"\bmajor questions?\b",
                r"\bmajor economic and political\b",
                r"\bvast economic and political\b",
                r"\bclear congressional authorization\b",
                r"\bclear statement\b",
                r"\belephants? in mouseholes?\b",
                r"\bextraordinary cases?\b",
                r"\bquestions? of deep economic and political significance\b",
            ),
            (
                r"\bagenc(?:y|ies)\b",
                r"\bcongress\b",
                r"\bstatut(?:e|ory)\b",
                r"\bauthority\b",
            ),
        ),
        evidence_patterns=(
            r"\bmajor questions?\b",
            r"\bmajor economic and political\b",
            r"\bvast economic and political\b",
            r"\bclear congressional authorization\b",
            r"\bclear statement\b",
            r"\belephants? in mouseholes?\b",
            r"\bextraordinary cases?\b",
            r"\bquestions? of deep economic and political significance\b",
            r"\bmci\b",
            r"\bbrown\s*&\s*williamson\b",
            r"\butility air\b",
            r"\bking\b",
            r"\bwest virginia\b",
            r"\bbiden v\. nebraska\b",
            r"\bgonzales\b",
        ),
    ),
    AdminRule(
        frame="admin_deference_ordinary",
        definition=(
            "Ordinary administrative-law deference or agency-interpretation reasoning, including Chevron, "
            "Auer/Kisor, Skidmore, permissible construction, and reasonable agency interpretation."
        ),
        required_any=(
            (
                r"\bchevron\b",
                r"\bauer\b",
                r"\bkisor\b",
                r"\bskidmore\b",
                r"\bdeference\b",
                r"\bdefer\b",
                r"\bambiguous statute\b",
                r"\bpermissible construction\b",
                r"\breasonable interpretation\b",
                r"\bagency interpretation\b",
            ),
            (
                r"\bagenc(?:y|ies)\b",
                r"\bstatut(?:e|ory)\b",
                r"\bregulation\b",
                r"\bcongress\b",
            ),
        ),
        evidence_patterns=(
            r"\bchevron\b",
            r"\bauer\b",
            r"\bkisor\b",
            r"\bskidmore\b",
            r"\bdeference\b",
            r"\bdefer\b",
            r"\bambiguous statute\b",
            r"\bpermissible construction\b",
            r"\breasonable interpretation\b",
            r"\bagency interpretation\b",
            r"\bmead\b",
            r"\bbarnhart\b",
            r"\bcity of arlington\b",
        ),
    ),
)


SOURCE_CASES: tuple[SourceCase, ...] = (
    SourceCase("mci_1994", "MCI Telecommunications Corp. v. AT&T Co.", "512 U.S. 218", 1993, "admin_major_questions", "agency modification power / elephants in mouseholes precursor", "https://www.law.cornell.edu/supremecourt/text/512/218"),
    SourceCase("brown_williamson_2000", "FDA v. Brown & Williamson Tobacco Corp.", "529 U.S. 120", 1999, "admin_major_questions", "tobacco regulation / clear congressional authorization", "https://www.law.cornell.edu/supremecourt/text/529/120"),
    SourceCase("gonzales_oregon_2006", "Gonzales v. Oregon", "546 U.S. 243", 2005, "admin_major_questions", "controlled substances / agency authority limit", "https://www.law.cornell.edu/supremecourt/text/04-623"),
    SourceCase("utility_air_2014", "Utility Air Regulatory Group v. EPA", "573 U.S. 302", 2013, "admin_major_questions", "greenhouse-gas permitting / vast regulatory expansion", "https://www.law.cornell.edu/supremecourt/text/12-1146"),
    SourceCase("king_burwell_2015", "King v. Burwell", "576 U.S. 473", 2014, "admin_major_questions", "ACA tax credits / major question outside Chevron", "https://www.law.cornell.edu/supremecourt/text/14-114"),
    SourceCase("west_virginia_epa_2022", "West Virginia v. EPA", "597 U.S. 697", 2021, "admin_major_questions", "generation shifting / major questions doctrine", "https://www.law.cornell.edu/supremecourt/text/20-1530"),
    SourceCase("biden_nebraska_2023", "Biden v. Nebraska", "600 U.S. 477", 2022, "admin_major_questions", "student loan cancellation / major questions", "https://www.law.cornell.edu/supremecourt/text/22-506"),
    SourceCase("skidmore_1944", "Skidmore v. Swift & Co.", "323 U.S. 134", 1944, "admin_deference_ordinary", "agency interpretation / power to persuade", "https://www.law.cornell.edu/supremecourt/text/323/134"),
    SourceCase("chevron_1984", "Chevron U.S.A. Inc. v. Natural Resources Defense Council, Inc.", "467 U.S. 837", 1983, "admin_deference_ordinary", "Chevron two-step / permissible construction", "https://www.law.cornell.edu/supremecourt/text/467/837"),
    SourceCase("auer_1997", "Auer v. Robbins", "519 U.S. 452", 1996, "admin_deference_ordinary", "agency interpretation of own regulation", "https://www.law.cornell.edu/supremecourt/text/519/452"),
    SourceCase("mead_2001", "United States v. Mead Corp.", "533 U.S. 218", 2000, "admin_deference_ordinary", "Chevron eligibility / delegated authority", "https://www.law.cornell.edu/supremecourt/text/533/218"),
    SourceCase("barnhart_2002", "Barnhart v. Walton", "535 U.S. 212", 2001, "admin_deference_ordinary", "agency interpretation / Chevron deference", "https://www.law.cornell.edu/supremecourt/text/535/212"),
    SourceCase("city_arlington_2013", "City of Arlington v. FCC", "569 U.S. 290", 2012, "admin_deference_ordinary", "Chevron and jurisdictional questions", "https://www.law.cornell.edu/supremecourt/text/11-1545"),
    SourceCase("kisor_2019", "Kisor v. Wilkie", "588 U.S. 558", 2018, "admin_deference_ordinary", "Auer deference limits", "https://www.law.cornell.edu/supremecourt/text/18-15"),
)


def mask_frame_cues(text: str) -> str:
    masked = mask_citations(text)
    replacements = [
        (r"\bmajor questions?\b", "[MAJOR_QUESTION]"),
        (r"\bmajor economic and political\b", "[MAJOR_QUESTION]"),
        (r"\bvast economic and political\b", "[MAJOR_QUESTION]"),
        (r"\bclear congressional authorization\b", "[MAJOR_QUESTION]"),
        (r"\bclear statement\b", "[MAJOR_QUESTION]"),
        (r"\belephants? in mouseholes?\b", "[MAJOR_QUESTION]"),
        (r"\bextraordinary cases?\b", "[MAJOR_QUESTION]"),
        (r"\bquestions? of deep economic and political significance\b", "[MAJOR_QUESTION]"),
        (r"\bchevron\b", "[DEFERENCE]"),
        (r"\bauer\b", "[DEFERENCE]"),
        (r"\bkisor\b", "[DEFERENCE]"),
        (r"\bskidmore\b", "[DEFERENCE]"),
        (r"\bdeference\b", "[DEFERENCE]"),
        (r"\bdefer\b", "[DEFERENCE]"),
        (r"\bambiguous statute\b", "[DEFERENCE]"),
        (r"\bpermissible construction\b", "[DEFERENCE]"),
        (r"\breasonable interpretation\b", "[DEFERENCE]"),
        (r"\bagency interpretation\b", "[DEFERENCE]"),
        (r"\bagenc(?:y|ies)\b", "[AGENCY]"),
        (r"\bstatut(?:e|ory)\b", "[STATUTE]"),
        (r"\bregulation\b", "[REGULATION]"),
        (r"\bMCI\b", "[CASE]"),
        (r"\bBrown\s*&\s*Williamson\b", "[CASE]"),
        (r"\bGonzales\b", "[CASE]"),
        (r"\bUtility Air\b", "[CASE]"),
        (r"\bKing\b", "[CASE]"),
        (r"\bWest Virginia\b", "[CASE]"),
        (r"\bBiden v\. Nebraska\b", "[CASE]"),
        (r"\bMead\b", "[CASE]"),
        (r"\bBarnhart\b", "[CASE]"),
        (r"\bCity of Arlington\b", "[CASE]"),
    ]
    for pattern, repl in replacements:
        masked = re.sub(pattern, repl, masked, flags=re.IGNORECASE)
    return masked


def match_regexes(text: str, patterns: tuple[str, ...]) -> list[str]:
    return [pattern for pattern in patterns if re.search(pattern, text, flags=re.IGNORECASE)]


def rule_matches(text: str, rule: AdminRule) -> tuple[bool, list[str]]:
    for group in rule.required_any:
        if not match_regexes(text, group):
            return False, []
    return True, match_regexes(text, rule.evidence_patterns)


def fetch_sources(raw_path: Path, *, refresh: bool, delay: float) -> list[dict[str, Any]]:
    if raw_path.exists() and not refresh:
        return json.loads(raw_path.read_text(encoding="utf-8"))
    session = requests.Session()
    session.headers.update({"User-Agent": "scotus-admin-source-pack/0.1 research"})
    pages: list[dict[str, Any]] = []
    for source in tqdm(SOURCE_CASES, desc="download admin source opinions"):
        response = session.get(source.url, timeout=60)
        response.raise_for_status()
        text = clean_plain_text(extract_law_cornell_text(response.text))
        pages.append(
            {
                "case_id": source.case_id,
                "case_name": source.case_name,
                "citation": source.citation,
                "term": source.term,
                "expected_frame": source.expected_frame,
                "expected_signal": source.expected_signal,
                "source_url": source.url,
                "status_code": response.status_code,
                "retrieved_chars": len(response.text),
                "text": text,
                "token_count": token_count(text),
            }
        )
        if delay > 0:
            time.sleep(delay)
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path.write_text(json.dumps(pages, ensure_ascii=False, indent=2), encoding="utf-8")
    return pages


def build_chunk_rows(pages: list[dict[str, Any]], *, min_tokens: int, target_min: int, target_max: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for page in pages:
        chunks = paragraph_chunks(str(page["text"]), min_tokens=min_tokens, target_min=target_min, target_max=target_max)
        for idx, chunk in enumerate(chunks):
            rows.append(
                {
                    "chunk_id": f"{page['case_id']}-{idx:04d}",
                    "opinion_id": page["case_id"],
                    "cluster_id": page["case_id"],
                    "case_name": page["case_name"],
                    "term": page["term"],
                    "justice": "source_opinion",
                    "section_author": "",
                    "section_posture": "source_opinion",
                    "issue_area_label": "Administrative Law",
                    "source_url": page["source_url"],
                    "chunk_index_in_section": idx,
                    "token_count": token_count(chunk),
                    "text": chunk,
                    "text_masked": mask_citations(chunk),
                    "text_cue_masked": mask_frame_cues(chunk),
                    "source_case_id": page["case_id"],
                    "source_citation": page["citation"],
                    "expected_frame": page["expected_frame"],
                    "expected_signal": page["expected_signal"],
                }
            )
    return rows


def priority_score(row: dict[str, Any], rule: AdminRule, evidence: list[str]) -> tuple[int, int, int, int, str]:
    expected_bonus = 3 if row.get("expected_frame") == rule.frame else 0
    return (
        expected_bonus,
        len(evidence),
        -abs(int(row.get("token_count") or 0) - 220),
        -int(row.get("chunk_index_in_section") or 0),
        str(row.get("case_name", "")),
    )


def build_records(chunk_rows: list[dict[str, Any]], *, max_per_frame: int, window_chars: int) -> list[dict[str, Any]]:
    matches_by_chunk: dict[str, list[tuple[AdminRule, list[str]]]] = {}
    for row in tqdm(chunk_rows, desc="scan admin chunks"):
        text = str(row.get("text", ""))
        for rule in ADMIN_RULES:
            matched, evidence = rule_matches(text, rule)
            if matched:
                matches_by_chunk.setdefault(str(row["chunk_id"]), []).append((rule, evidence))

    by_chunk = {str(row["chunk_id"]): row for row in chunk_rows}
    candidates_by_frame: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for chunk_id, matches in matches_by_chunk.items():
        row = by_chunk[chunk_id]
        matched_frames = sorted(rule.frame for rule, _evidence in matches)
        for rule, evidence in matches:
            candidates_by_frame[rule.frame].append(
                {
                    "record_id": f"{chunk_id}::{rule.frame}",
                    "frame": rule.frame,
                    "issue_family": "Administrative Law",
                    "label": 1,
                    "label_source": "admin_source_rule_v1",
                    "label_confidence": "silver_review_required",
                    "label_definition": rule.definition,
                    "evidence_patterns": evidence,
                    "matched_frames": matched_frames,
                    "has_multi_frame_conflict": len(matched_frames) > 1,
                    "global_split": stable_split(str(row["cluster_id"])),
                    "split": "unassigned",
                    "opinion_id": row.get("opinion_id"),
                    "cluster_id": row.get("cluster_id"),
                    "case_name": row.get("case_name"),
                    "term": row.get("term"),
                    "justice": row.get("justice"),
                    "section_author": row.get("section_author"),
                    "section_posture": row.get("section_posture"),
                    "issue_area_label": row.get("issue_area_label"),
                    "source_url": row.get("source_url"),
                    "chunk_id": row.get("chunk_id"),
                    "chunk_index_in_section": row.get("chunk_index_in_section"),
                    "token_count": row.get("token_count"),
                    "source_case_id": row.get("source_case_id"),
                    "source_citation": row.get("source_citation"),
                    "expected_frame": row.get("expected_frame"),
                    "expected_signal": row.get("expected_signal"),
                    "text": row.get("text"),
                    "text_masked": row.get("text_masked"),
                    "text_cue_masked": row.get("text_cue_masked"),
                    "evidence_window": token_window(str(row.get("text", "")), evidence, window_chars),
                    "review_status": "unreviewed",
                    "review_notes": "",
                    "_priority": priority_score(row, rule, evidence),
                }
            )

    selected: list[dict[str, Any]] = []
    for frame, records in candidates_by_frame.items():
        records.sort(key=lambda item: item["_priority"], reverse=True)
        used_cases: set[str] = set()
        diverse: list[dict[str, Any]] = []
        overflow: list[dict[str, Any]] = []
        for record in records:
            case_id = str(record.get("source_case_id") or record.get("cluster_id") or "")
            if case_id not in used_cases:
                diverse.append(record)
                used_cases.add(case_id)
            else:
                overflow.append(record)
        for record in (diverse + overflow)[:max_per_frame]:
            record.pop("_priority", None)
            selected.append(record)

    assign_frame_splits(selected)
    selected.sort(key=lambda item: (item["frame"], item["split"], str(item["case_name"]), str(item["chunk_id"])))
    return selected


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def write_report(path: Path, *, pages: list[dict[str, Any]], chunk_rows: list[dict[str, Any]], records: list[dict[str, Any]], args: argparse.Namespace) -> None:
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    for record in records:
        counts[str(record["frame"])][str(record["split"])] += 1
    frame_rows: list[list[Any]] = []
    for rule in ADMIN_RULES:
        counter = counts.get(rule.frame, Counter())
        total = sum(counter.values())
        conflicts = sum(1 for row in records if row["frame"] == rule.frame and row["has_multi_frame_conflict"])
        cases = len({row["source_case_id"] for row in records if row["frame"] == rule.frame})
        frame_rows.append([rule.frame, total, cases, counter["train"], counter["dev"], counter["test"], conflicts])

    source_rows = [
        [page["case_id"], page["case_name"], page["citation"], page["expected_frame"], page["token_count"], page["source_url"]]
        for page in pages
    ]
    case_rows = [
        [case_id, case_name, frame, count]
        for (case_id, case_name, frame), count in sorted(Counter((row["source_case_id"], row["case_name"], row["frame"]) for row in records).items())
    ]
    sample_rows = [
        [
            record["frame"],
            record["split"],
            record.get("case_name", ""),
            record.get("source_citation", ""),
            ", ".join(record.get("evidence_patterns", [])[:3]),
            "yes" if record.get("has_multi_frame_conflict") else "no",
            str(record.get("evidence_window", ""))[:220],
        ]
        for record in records[: min(20, len(records))]
    ]

    lines = [
        "# SCOTUS Administrative Law Source Pack v1",
        "",
        "## Purpose",
        "",
        "Administrative Law is the remaining ranked source branch. This pack tests major-questions/clear-authorization reasoning against ordinary agency-deference/statutory-interpretation reasoning.",
        "",
        "## Outputs",
        "",
        f"- Raw pages: `{args.raw}`",
        f"- Labels: `{args.labels}`",
        f"- Review queue: `{args.queue}`",
        f"- Source chunks scanned: `{len(chunk_rows)}`",
        "",
        "## Source Cases",
        "",
    ]
    lines.extend(markdown_table(["Case id", "Case", "Citation", "Expected frame", "Tokens", "URL"], source_rows))
    lines.extend(["", "## Label Counts", ""])
    lines.extend(markdown_table(["Frame", "Total", "Cases", "Train", "Dev", "Test", "Multi-frame conflicts"], frame_rows))
    lines.extend(["", "## Case/Frame Coverage", ""])
    lines.extend(markdown_table(["Case id", "Case", "Frame", "Records"], case_rows))
    lines.extend(["", "## Sample Evidence Windows", ""])
    lines.extend(markdown_table(["Frame", "Split", "Case", "Citation", "Evidence", "Conflict", "Window"], sample_rows))
    lines.extend(
        [
            "",
            "## Use Rules",
            "",
            "1. Run a cue-masked text-only gate before any BF16 activation capture.",
            "2. If text alone solves major-questions versus deference, close the branch as leakage/text dominated.",
            "3. If text-only is not saturated, run dominance review before activation probing.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--queue", type=Path, default=DEFAULT_QUEUE)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--delay", type=float, default=0.25)
    parser.add_argument("--min-tokens", type=int, default=60)
    parser.add_argument("--target-min", type=int, default=150)
    parser.add_argument("--target-max", type=int, default=350)
    parser.add_argument("--max-per-frame", type=int, default=72)
    parser.add_argument("--window-chars", type=int, default=900)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pages = fetch_sources(args.raw, refresh=args.refresh, delay=args.delay)
    chunk_rows = build_chunk_rows(
        pages,
        min_tokens=args.min_tokens,
        target_min=args.target_min,
        target_max=args.target_max,
    )
    records = build_records(chunk_rows, max_per_frame=args.max_per_frame, window_chars=args.window_chars)
    write_jsonl(args.labels, records)
    write_jsonl(args.queue, records)
    write_report(args.report, pages=pages, chunk_rows=chunk_rows, records=records, args=args)
    print(f"Wrote {len(records)} labels from {len(chunk_rows)} chunks")
    print(f"Raw pages: {args.raw}")
    print(f"Labels: {args.labels}")
    print(f"Queue: {args.queue}")
    print(f"Report: {args.report}")


if __name__ == "__main__":
    main()
