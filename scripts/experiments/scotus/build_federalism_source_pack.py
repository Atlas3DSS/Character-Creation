#!/usr/bin/env python3
"""Build a Federalism anti-commandeering / preemption source-frame pack."""

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
DEFAULT_RAW = SCOTUS_DIR / "raw" / "scotus_federalism_source_pages_v1.json"
DEFAULT_LABELS = SCOTUS_DIR / "scotus_federalism_source_frame_labels_v1.jsonl"
DEFAULT_QUEUE = SCOTUS_DIR / "scotus_federalism_source_frame_review_queue_v1.jsonl"
DEFAULT_REPORT = PROJECT_ROOT / "reports" / "scotus_federalism_source_pack_v1.md"

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
class FederalismRule:
    frame: str
    definition: str
    required_any: tuple[tuple[str, ...], ...]
    evidence_patterns: tuple[str, ...]


FEDERALISM_RULES: tuple[FederalismRule, ...] = (
    FederalismRule(
        frame="federalism_anti_commandeering",
        definition=(
            "Anti-commandeering reasoning: Congress may not compel state legislatures, "
            "state executives, or state officers to enact, administer, or enforce a federal program."
        ),
        required_any=(
            (
                r"\bcommandeer(?:ing)?\b",
                r"\bconscript(?:ion|ed)?\b",
                r"\bstate officers?\b",
                r"\bstate officials?\b",
                r"\bstate legislature\b",
                r"\btake title\b",
                r"\badminister\b.{0,80}\bfederal\b",
                r"\benforce\b.{0,80}\bfederal\b",
                r"\bimplement\b.{0,80}\bfederal\b",
                r"\bpolitical accountability\b",
            ),
            (
                r"\bcongress\b",
                r"\bfederal\b",
                r"\bnational\b",
                r"\bsupremacy clause\b",
            ),
        ),
        evidence_patterns=(
            r"\banti[- ]commandeering\b",
            r"\bcommandeer(?:ing)?\b",
            r"\bconscript(?:ion|ed)?\b",
            r"\bstate officers?\b",
            r"\bstate officials?\b",
            r"\bstate legislature\b",
            r"\btake title\b",
            r"\badminister\b.{0,80}\bfederal\b",
            r"\benforce\b.{0,80}\bfederal\b",
            r"\bimplement\b.{0,80}\bfederal\b",
            r"\bpolitical accountability\b",
            r"\bnew york\b",
            r"\bprintz\b",
            r"\bmurphy\b",
            r"\breno\b",
            r"\bferc\b",
        ),
    ),
    FederalismRule(
        frame="federalism_preemption",
        definition=(
            "Supremacy Clause preemption reasoning: federal law displaces, supersedes, "
            "or invalidates state law through express, field, conflict, or obstacle preemption."
        ),
        required_any=(
            (
                r"\bpreempt(?:ion|ed|s)?\b",
                r"\bpre-empt(?:ion|ed|s)?\b",
                r"\bsupremacy clause\b",
                r"\bconflict preemption\b",
                r"\bfield preemption\b",
                r"\bexpress preemption\b",
                r"\bobstacle\b.{0,80}\bpreempt",
                r"\bfederal law\b.{0,80}\bsupersed",
            ),
            (
                r"\bstate law\b",
                r"\bstate regulation\b",
                r"\bfederal law\b",
                r"\bcongress\b",
                r"\bsupremacy\b",
            ),
        ),
        evidence_patterns=(
            r"\bpreempt(?:ion|ed|s)?\b",
            r"\bpre-empt(?:ion|ed|s)?\b",
            r"\bsupremacy clause\b",
            r"\bconflict preemption\b",
            r"\bfield preemption\b",
            r"\bexpress preemption\b",
            r"\bobstacle\b.{0,80}\bpreempt",
            r"\bfederal law\b.{0,80}\bsupersed",
            r"\barizona\b",
            r"\bcrosby\b",
            r"\bgade\b",
            r"\bgeier\b",
            r"\bcipollone\b",
            r"\bwyeth\b",
            r"\bhines\b",
            r"\brice\b",
        ),
    ),
)


SOURCE_CASES: tuple[SourceCase, ...] = (
    SourceCase("new_york_1992", "New York v. United States", "505 U.S. 144", 1991, "federalism_anti_commandeering", "take-title provision / state legislative commandeering", "https://www.law.cornell.edu/supremecourt/text/505/144"),
    SourceCase("printz_1997", "Printz v. United States", "521 U.S. 898", 1996, "federalism_anti_commandeering", "state executive officers / background checks", "https://www.law.cornell.edu/supremecourt/text/521/898"),
    SourceCase("reno_condon_2000", "Reno v. Condon", "528 U.S. 141", 1999, "federalism_anti_commandeering", "generally applicable federal regulation of states", "https://www.law.cornell.edu/supremecourt/text/528/141"),
    SourceCase("ferc_1982", "FERC v. Mississippi", "456 U.S. 742", 1981, "federalism_anti_commandeering", "state utility commissions / federal procedural requirements", "https://www.law.cornell.edu/supremecourt/text/456/742"),
    SourceCase("murphy_2018", "Murphy v. National Collegiate Athletic Assn.", "584 U.S. 453", 2017, "federalism_anti_commandeering", "sports gambling / state repeal command", "https://www.law.cornell.edu/supremecourt/text/16-476"),
    SourceCase("hines_1941", "Hines v. Davidowitz", "312 U.S. 52", 1940, "federalism_preemption", "alien registration / obstacle preemption", "https://www.law.cornell.edu/supremecourt/text/312/52"),
    SourceCase("rice_1947", "Rice v. Santa Fe Elevator Corp.", "331 U.S. 218", 1946, "federalism_preemption", "field preemption / warehouse regulation", "https://www.law.cornell.edu/supremecourt/text/331/218"),
    SourceCase("gade_1992", "Gade v. National Solid Wastes Management Assn.", "505 U.S. 88", 1991, "federalism_preemption", "occupational safety / state licensing preemption", "https://www.law.cornell.edu/supremecourt/text/505/88"),
    SourceCase("cipollone_1992", "Cipollone v. Liggett Group, Inc.", "505 U.S. 504", 1991, "federalism_preemption", "tobacco labeling / express preemption", "https://www.law.cornell.edu/supremecourt/text/505/504"),
    SourceCase("crosby_2000", "Crosby v. National Foreign Trade Council", "530 U.S. 363", 1999, "federalism_preemption", "foreign affairs sanctions / obstacle preemption", "https://www.law.cornell.edu/supremecourt/text/530/363"),
    SourceCase("geier_2000", "Geier v. American Honda Motor Co.", "529 U.S. 861", 1999, "federalism_preemption", "auto safety / obstacle preemption", "https://www.law.cornell.edu/supremecourt/text/529/861"),
    SourceCase("wyeth_2009", "Wyeth v. Levine", "555 U.S. 555", 2008, "federalism_preemption", "drug labeling / no impossibility preemption", "https://www.law.cornell.edu/supremecourt/text/06-1249"),
    SourceCase("arizona_2012", "Arizona v. United States", "567 U.S. 387", 2011, "federalism_preemption", "immigration enforcement / field and conflict preemption", "https://www.law.cornell.edu/supremecourt/text/11-182"),
)


def mask_frame_cues(text: str) -> str:
    masked = mask_citations(text)
    replacements = [
        (r"\banti[- ]commandeering\b", "[ANTI_COMMANDEERING]"),
        (r"\bcommandeer(?:ing)?\b", "[ANTI_COMMANDEERING]"),
        (r"\bconscript(?:ion|ed)?\b", "[ANTI_COMMANDEERING]"),
        (r"\bstate officers?\b", "[STATE_ACTOR]"),
        (r"\bstate officials?\b", "[STATE_ACTOR]"),
        (r"\bstate legislature\b", "[STATE_ACTOR]"),
        (r"\btake title\b", "[ANTI_COMMANDEERING]"),
        (r"\bpolitical accountability\b", "[FEDERALISM]"),
        (r"\badminister\b.{0,80}\bfederal\b", "[ANTI_COMMANDEERING]"),
        (r"\benforce\b.{0,80}\bfederal\b", "[ANTI_COMMANDEERING]"),
        (r"\bimplement\b.{0,80}\bfederal\b", "[ANTI_COMMANDEERING]"),
        (r"\bpreempt(?:ion|ed|s)?\b", "[PREEMPTION]"),
        (r"\bpre-empt(?:ion|ed|s)?\b", "[PREEMPTION]"),
        (r"\bsupremacy clause\b", "[PREEMPTION]"),
        (r"\bconflict preemption\b", "[PREEMPTION]"),
        (r"\bfield preemption\b", "[PREEMPTION]"),
        (r"\bexpress preemption\b", "[PREEMPTION]"),
        (r"\bstate law\b", "[STATE_LAW]"),
        (r"\bstate regulation\b", "[STATE_LAW]"),
        (r"\bfederal law\b", "[FEDERAL_LAW]"),
        (r"\bNew York v\. United States\b", "[CASE]"),
        (r"\bNew York\b", "[CASE]"),
        (r"\bPrintz\b", "[CASE]"),
        (r"\bMurphy\b", "[CASE]"),
        (r"\bReno\b", "[CASE]"),
        (r"\bFERC\b", "[CASE]"),
        (r"\bArizona\b", "[CASE]"),
        (r"\bCrosby\b", "[CASE]"),
        (r"\bGade\b", "[CASE]"),
        (r"\bGeier\b", "[CASE]"),
        (r"\bCipollone\b", "[CASE]"),
        (r"\bWyeth\b", "[CASE]"),
        (r"\bHines\b", "[CASE]"),
        (r"\bRice\b", "[CASE]"),
    ]
    for pattern, repl in replacements:
        masked = re.sub(pattern, repl, masked, flags=re.IGNORECASE)
    return masked


def match_regexes(text: str, patterns: tuple[str, ...]) -> list[str]:
    return [pattern for pattern in patterns if re.search(pattern, text, flags=re.IGNORECASE)]


def rule_matches(text: str, rule: FederalismRule) -> tuple[bool, list[str]]:
    for group in rule.required_any:
        if not match_regexes(text, group):
            return False, []
    return True, match_regexes(text, rule.evidence_patterns)


def fetch_sources(raw_path: Path, *, refresh: bool, delay: float) -> list[dict[str, Any]]:
    if raw_path.exists() and not refresh:
        return json.loads(raw_path.read_text(encoding="utf-8"))
    session = requests.Session()
    session.headers.update({"User-Agent": "scotus-federalism-source-pack/0.1 research"})
    pages: list[dict[str, Any]] = []
    for source in tqdm(SOURCE_CASES, desc="download federalism source opinions"):
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
                    "issue_area_label": "Federalism",
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


def priority_score(row: dict[str, Any], rule: FederalismRule, evidence: list[str]) -> tuple[int, int, int, int, str]:
    expected_bonus = 3 if row.get("expected_frame") == rule.frame else 0
    return (
        expected_bonus,
        len(evidence),
        -abs(int(row.get("token_count") or 0) - 220),
        -int(row.get("chunk_index_in_section") or 0),
        str(row.get("case_name", "")),
    )


def build_records(chunk_rows: list[dict[str, Any]], *, max_per_frame: int, window_chars: int) -> list[dict[str, Any]]:
    matches_by_chunk: dict[str, list[tuple[FederalismRule, list[str]]]] = {}
    for row in tqdm(chunk_rows, desc="scan federalism chunks"):
        text = str(row.get("text", ""))
        for rule in FEDERALISM_RULES:
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
                    "issue_family": "Federalism",
                    "label": 1,
                    "label_source": "federalism_source_rule_v1",
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
    for rule in FEDERALISM_RULES:
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
        "# SCOTUS Federalism Source Pack v1",
        "",
        "## Purpose",
        "",
        "Federalism was the next viable branch after Economic Activity and Civil Rights failed promotion gates. This pack tests a narrower same-doctrine contrast: anti-commandeering versus Supremacy Clause preemption.",
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
            "2. If text alone solves anti-commandeering versus preemption, close the branch as leakage/text dominated.",
            "3. If the text-only gate is not saturated, run a source-case-heldout cue-masked activation probe and compare against the text baseline.",
            "4. Treat `Murphy` rows carefully because the opinion discusses both anti-commandeering and preemption.",
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
