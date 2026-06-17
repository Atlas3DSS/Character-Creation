#!/usr/bin/env python3
"""Build a Due Process substantive/procedural source-frame pack."""

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
DEFAULT_RAW = SCOTUS_DIR / "raw" / "scotus_due_process_source_pages_v1.json"
DEFAULT_LABELS = SCOTUS_DIR / "scotus_due_process_source_frame_labels_v1.jsonl"
DEFAULT_QUEUE = SCOTUS_DIR / "scotus_due_process_source_frame_review_queue_v1.jsonl"
DEFAULT_REPORT = PROJECT_ROOT / "reports" / "scotus_due_process_source_pack_v1.md"

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
class DueProcessRule:
    frame: str
    definition: str
    required_any: tuple[tuple[str, ...], ...]
    evidence_patterns: tuple[str, ...]


DUE_PROCESS_RULES: tuple[DueProcessRule, ...] = (
    DueProcessRule(
        frame="due_process_substantive",
        definition=(
            "Substantive due process reasoning about liberty, fundamental rights, privacy, "
            "marriage, bodily autonomy, ordered liberty, or history-and-tradition limits."
        ),
        required_any=(
            (
                r"\bsubstantive due process\b",
                r"\bfundamental right\b",
                r"\bfundamental rights\b",
                r"\bordered liberty\b",
                r"\bdeeply rooted\b",
                r"\bhistory and tradition\b",
                r"\bliberty\b",
                r"\bprivacy\b",
                r"\bmarriage\b",
            ),
            (
                r"\bdue process\b",
                r"\bfourteenth amendment\b",
                r"\bliberty\b",
                r"\bconstitution\b",
            ),
        ),
        evidence_patterns=(
            r"\bsubstantive due process\b",
            r"\bfundamental right\b",
            r"\bfundamental rights\b",
            r"\bordered liberty\b",
            r"\bdeeply rooted\b",
            r"\bhistory and tradition\b",
            r"\bliberty\b",
            r"\bprivacy\b",
            r"\bmarriage\b",
            r"\bautonomy\b",
            r"\bbodily integrity\b",
            r"\bgriswold\b",
            r"\broe\b",
            r"\bcasey\b",
            r"\blawrence\b",
            r"\bobergefell\b",
            r"\bdobbs\b",
            r"\bglucksberg\b",
        ),
    ),
    DueProcessRule(
        frame="due_process_procedural_mathews",
        definition=(
            "Procedural due process reasoning about notice, hearing, deprivation, Mathews balancing, "
            "private interest, erroneous-deprivation risk, and government interest."
        ),
        required_any=(
            (
                r"\bprocedural due process\b",
                r"\bmathews\b",
                r"\bhearing\b",
                r"\bnotice\b",
                r"\bopportunity to be heard\b",
                r"\bdeprivation\b",
                r"\bprivate interest\b",
                r"\brisk of erroneous\b",
            ),
            (
                r"\bdue process\b",
                r"\bfourteenth amendment\b",
                r"\bfifth amendment\b",
                r"\bliberty\b",
                r"\bproperty\b",
            ),
        ),
        evidence_patterns=(
            r"\bprocedural due process\b",
            r"\bmathews\b",
            r"\bhearing\b",
            r"\bnotice\b",
            r"\bopportunity to be heard\b",
            r"\bdeprivation\b",
            r"\bprivate interest\b",
            r"\brisk of erroneous\b",
            r"\bgovernment(?:'s)? interest\b",
            r"\bprobable value\b",
            r"\bgoldberg\b",
            r"\bgoss\b",
            r"\bmorrissey\b",
            r"\bhamdi\b",
            r"\bloudermill\b",
        ),
    ),
)


SOURCE_CASES: tuple[SourceCase, ...] = (
    SourceCase("griswold_1965", "Griswold v. Connecticut", "381 U.S. 479", 1964, "due_process_substantive", "privacy / marital liberty", "https://www.law.cornell.edu/supremecourt/text/381/479"),
    SourceCase("roe_1973", "Roe v. Wade", "410 U.S. 113", 1972, "due_process_substantive", "privacy / abortion", "https://www.law.cornell.edu/supremecourt/text/410/113"),
    SourceCase("casey_1992", "Planned Parenthood of Southeastern Pennsylvania v. Casey", "505 U.S. 833", 1991, "due_process_substantive", "liberty / undue burden", "https://www.law.cornell.edu/supremecourt/text/505/833"),
    SourceCase("glucksberg_1997", "Washington v. Glucksberg", "521 U.S. 702", 1996, "due_process_substantive", "deeply rooted history and tradition", "https://www.law.cornell.edu/supremecourt/text/521/702"),
    SourceCase("lawrence_2003", "Lawrence v. Texas", "539 U.S. 558", 2002, "due_process_substantive", "liberty / intimate conduct", "https://www.law.cornell.edu/supremecourt/text/539/558"),
    SourceCase("obergefell_2015", "Obergefell v. Hodges", "576 U.S. 644", 2014, "due_process_substantive", "marriage liberty", "https://www.law.cornell.edu/supremecourt/text/14-556"),
    SourceCase("dobbs_2022", "Dobbs v. Jackson Women's Health Organization", "597 U.S. 215", 2021, "due_process_substantive", "history and tradition / abortion", "https://www.law.cornell.edu/supremecourt/text/19-1392"),
    SourceCase("goldberg_1970", "Goldberg v. Kelly", "397 U.S. 254", 1969, "due_process_procedural_mathews", "welfare termination / evidentiary hearing", "https://www.law.cornell.edu/supremecourt/text/397/254"),
    SourceCase("morrissey_1972", "Morrissey v. Brewer", "408 U.S. 471", 1971, "due_process_procedural_mathews", "parole revocation / process due", "https://www.law.cornell.edu/supremecourt/text/408/471"),
    SourceCase("goss_1975", "Goss v. Lopez", "419 U.S. 565", 1974, "due_process_procedural_mathews", "school suspension / notice and hearing", "https://www.law.cornell.edu/supremecourt/text/419/565"),
    SourceCase("mathews_1976", "Mathews v. Eldridge", "424 U.S. 319", 1975, "due_process_procedural_mathews", "disability benefits / balancing", "https://www.law.cornell.edu/supremecourt/text/424/319"),
    SourceCase("loudermill_1985", "Cleveland Board of Education v. Loudermill", "470 U.S. 532", 1984, "due_process_procedural_mathews", "public employment / pretermination hearing", "https://www.law.cornell.edu/supremecourt/text/470/532"),
    SourceCase("hamdi_2004", "Hamdi v. Rumsfeld", "542 U.S. 507", 2003, "due_process_procedural_mathews", "enemy combatant / Mathews process", "https://www.law.cornell.edu/supremecourt/text/542/507"),
)


def mask_frame_cues(text: str) -> str:
    masked = mask_citations(text)
    replacements = [
        (r"\bsubstantive due process\b", "[DUE_PROCESS]"),
        (r"\bprocedural due process\b", "[DUE_PROCESS]"),
        (r"\bdue process\b", "[DUE_PROCESS]"),
        (r"\bfourteenth amendment\b", "[AMENDMENT]"),
        (r"\bfifth amendment\b", "[AMENDMENT]"),
        (r"\bfundamental rights?\b", "[RIGHT]"),
        (r"\bordered liberty\b", "[LIBERTY]"),
        (r"\bdeeply rooted\b", "[HISTORY]"),
        (r"\bhistory and tradition\b", "[HISTORY]"),
        (r"\bliberty\b", "[LIBERTY]"),
        (r"\bprivacy\b", "[LIBERTY]"),
        (r"\bmarriage\b", "[LIBERTY]"),
        (r"\bautonomy\b", "[LIBERTY]"),
        (r"\bbodily integrity\b", "[LIBERTY]"),
        (r"\bhearing\b", "[PROCESS]"),
        (r"\bnotice\b", "[PROCESS]"),
        (r"\bopportunity to be heard\b", "[PROCESS]"),
        (r"\bdeprivation\b", "[PROCESS]"),
        (r"\bprivate interest\b", "[PROCESS]"),
        (r"\brisk of erroneous\b", "[PROCESS]"),
        (r"\bgovernment(?:'s)? interest\b", "[PROCESS]"),
        (r"\bprobable value\b", "[PROCESS]"),
        (r"\bGriswold\b", "[CASE]"),
        (r"\bRoe\b", "[CASE]"),
        (r"\bCasey\b", "[CASE]"),
        (r"\bLawrence\b", "[CASE]"),
        (r"\bObergefell\b", "[CASE]"),
        (r"\bDobbs\b", "[CASE]"),
        (r"\bGlucksberg\b", "[CASE]"),
        (r"\bGoldberg\b", "[CASE]"),
        (r"\bMathews\b", "[CASE]"),
        (r"\bGoss\b", "[CASE]"),
        (r"\bMorrissey\b", "[CASE]"),
        (r"\bHamdi\b", "[CASE]"),
        (r"\bLoudermill\b", "[CASE]"),
    ]
    for pattern, repl in replacements:
        masked = re.sub(pattern, repl, masked, flags=re.IGNORECASE)
    return masked


def match_regexes(text: str, patterns: tuple[str, ...]) -> list[str]:
    return [pattern for pattern in patterns if re.search(pattern, text, flags=re.IGNORECASE)]


def rule_matches(text: str, rule: DueProcessRule) -> tuple[bool, list[str]]:
    for group in rule.required_any:
        if not match_regexes(text, group):
            return False, []
    return True, match_regexes(text, rule.evidence_patterns)


def fetch_sources(raw_path: Path, *, refresh: bool, delay: float) -> list[dict[str, Any]]:
    if raw_path.exists() and not refresh:
        return json.loads(raw_path.read_text(encoding="utf-8"))
    session = requests.Session()
    session.headers.update({"User-Agent": "scotus-due-process-source-pack/0.1 research"})
    pages: list[dict[str, Any]] = []
    for source in tqdm(SOURCE_CASES, desc="download due-process source opinions"):
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
                    "issue_area_label": "Due Process",
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


def priority_score(row: dict[str, Any], rule: DueProcessRule, evidence: list[str]) -> tuple[int, int, int, int, str]:
    expected_bonus = 3 if row.get("expected_frame") == rule.frame else 0
    return (
        expected_bonus,
        len(evidence),
        -abs(int(row.get("token_count") or 0) - 220),
        -int(row.get("chunk_index_in_section") or 0),
        str(row.get("case_name", "")),
    )


def build_records(chunk_rows: list[dict[str, Any]], *, max_per_frame: int, window_chars: int) -> list[dict[str, Any]]:
    matches_by_chunk: dict[str, list[tuple[DueProcessRule, list[str]]]] = {}
    for row in tqdm(chunk_rows, desc="scan due-process chunks"):
        text = str(row.get("text", ""))
        for rule in DUE_PROCESS_RULES:
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
                    "issue_family": "Due Process",
                    "label": 1,
                    "label_source": "due_process_source_rule_v1",
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
    for rule in DUE_PROCESS_RULES:
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
        "# SCOTUS Due Process Source Pack v1",
        "",
        "## Purpose",
        "",
        "Due Process is the next branch after Economic Activity, Civil Rights, and Federalism failed promotion gates. This pack tests substantive liberty/history-and-tradition reasoning against procedural Mathews/hearing reasoning.",
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
            "2. If text alone solves substantive versus procedural due process, close the branch as leakage/text dominated.",
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
