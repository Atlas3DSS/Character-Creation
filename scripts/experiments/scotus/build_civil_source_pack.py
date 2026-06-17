#!/usr/bin/env python3
"""Build a Civil Rights source-frame pack.

Civil Rights is the backup branch after the Economic Activity source probe
failed its promotion gate. This builder keeps the branch explicitly
dominance-review oriented because strict/intermediate/rational scrutiny labels
are unusually prone to lexical leakage.
"""

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
DEFAULT_RAW = SCOTUS_DIR / "raw" / "scotus_civil_source_pages_v1.json"
DEFAULT_LABELS = SCOTUS_DIR / "scotus_civil_source_frame_labels_v1.jsonl"
DEFAULT_QUEUE = SCOTUS_DIR / "scotus_civil_source_frame_review_queue_v1.jsonl"
DEFAULT_REPORT = PROJECT_ROOT / "reports" / "scotus_civil_source_pack_v1.md"

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
class CivilRule:
    frame: str
    definition: str
    required_any: tuple[tuple[str, ...], ...]
    evidence_patterns: tuple[str, ...]
    exclude_any: tuple[str, ...] = ()


CIVIL_RULES: tuple[CivilRule, ...] = (
    CivilRule(
        frame="civil_race_strict_scrutiny",
        definition=(
            "Race or affirmative-action equal-protection reasoning using strict scrutiny, "
            "compelling-interest, narrow-tailoring, suspect-classification, or racial-classification analysis."
        ),
        required_any=(
            (
                r"\brace\b",
                r"\bracial\b",
                r"\bminority\b",
                r"\baffirmative action\b",
                r"\bdiversity\b",
                r"\bsegregation\b",
                r"\bequal protection\b",
            ),
            (
                r"\bstrict scrutiny\b",
                r"\bcompelling (?:governmental |state )?interest\b",
                r"\bnarrowly tailored\b",
                r"\bsuspect classification\b",
                r"\bracial classification\b",
            ),
        ),
        evidence_patterns=(
            r"\bstrict scrutiny\b",
            r"\bcompelling (?:governmental |state )?interest\b",
            r"\bnarrowly tailored\b",
            r"\bsuspect classification\b",
            r"\bracial classification\b",
            r"\brace\b",
            r"\bracial\b",
            r"\baffirmative action\b",
            r"\bdiversity\b",
            r"\bloving\b",
            r"\bbakke\b",
            r"\bcroson\b",
            r"\badarand\b",
            r"\bgrutter\b",
            r"\bgratz\b",
            r"\bfisher\b",
            r"\bstudents for fair admissions\b",
        ),
        exclude_any=(
            r"\bintermediate scrutiny\b",
            r"\bexceedingly persuasive\b",
            r"\bsection 5\b",
            r"\bcongruence and proportionality\b",
        ),
    ),
    CivilRule(
        frame="civil_sex_intermediate_scrutiny",
        definition=(
            "Sex or gender equal-protection reasoning using intermediate scrutiny, important-governmental-objective, "
            "substantial-relation, or exceedingly-persuasive-justification analysis."
        ),
        required_any=(
            (
                r"\bsex\b",
                r"\bgender\b",
                r"\bwomen\b",
                r"\bwoman\b",
                r"\bfemale\b",
                r"\bmale\b",
                r"\bequal protection\b",
            ),
            (
                r"\bintermediate scrutiny\b",
                r"\bimportant governmental objective\b",
                r"\bsubstantially related\b",
                r"\bexceedingly persuasive\b",
                r"\bgender classification\b",
                r"\bsex-based classification\b",
            ),
        ),
        evidence_patterns=(
            r"\bintermediate scrutiny\b",
            r"\bimportant governmental objective\b",
            r"\bsubstantially related\b",
            r"\bexceedingly persuasive\b",
            r"\bgender classification\b",
            r"\bsex-based classification\b",
            r"\bsex\b",
            r"\bgender\b",
            r"\bwomen\b",
            r"\bfemale\b",
            r"\bfrontiero\b",
            r"\bcraig\b",
            r"\bhogan\b",
            r"\bvirginia\b",
            r"\bnguyen\b",
            r"\bmorales-santana\b",
        ),
        exclude_any=(
            r"\bstrict scrutiny\b",
            r"\bracial classification\b",
            r"\bsection 5\b",
            r"\bcongruence and proportionality\b",
        ),
    ),
    CivilRule(
        frame="civil_section5_congruence",
        definition=(
            "Fourteenth Amendment Section 5 enforcement-power reasoning, including congruence/proportionality, "
            "abrogation, remedial legislation, or state-sovereign-immunity limits."
        ),
        required_any=(
            (
                r"\bsection 5\b",
                r"\b§\s*5\b",
                r"\bfourteenth amendment\b",
                r"\benforcement power\b",
                r"\bcongruence and proportionality\b",
                r"\bcongruent and proportional\b",
            ),
            (
                r"\bcongress\b",
                r"\bremed(?:y|ial|ies)\b",
                r"\babrogat(?:e|ion|ed)\b",
                r"\bsovereign immunity\b",
                r"\beleventh amendment\b",
                r"\bstate(?:s)?\b",
            ),
        ),
        evidence_patterns=(
            r"\bsection 5\b",
            r"\b§\s*5\b",
            r"\bfourteenth amendment\b",
            r"\benforcement power\b",
            r"\bcongruence and proportionality\b",
            r"\bcongruent and proportional\b",
            r"\babrogat(?:e|ion|ed)\b",
            r"\bsovereign immunity\b",
            r"\beleventh amendment\b",
            r"\bboerne\b",
            r"\bkimel\b",
            r"\bgarrett\b",
            r"\bhibbs\b",
            r"\blane\b",
            r"\bcoleman\b",
        ),
        exclude_any=(
            r"\bstrict scrutiny\b",
            r"\bintermediate scrutiny\b",
        ),
    ),
    CivilRule(
        frame="civil_rational_basis_equal_protection",
        definition=(
            "Equal-protection rational-basis reasoning, including legitimate-government-interest, "
            "rational-relation, disability, sexual-orientation, or non-suspect-class review."
        ),
        required_any=(
            (
                r"\brational basis\b",
                r"\brationally related\b",
                r"\blegitimate (?:governmental |state )?interest\b",
                r"\bnot a suspect class\b",
                r"\bquasi-suspect\b",
            ),
            (
                r"\bequal protection\b",
                r"\bclassification\b",
                r"\bdisability\b",
                r"\bmental(?:ly)? retarded\b",
                r"\bsexual orientation\b",
                r"\bhomosexual\b",
            ),
        ),
        evidence_patterns=(
            r"\brational basis\b",
            r"\brationally related\b",
            r"\blegitimate (?:governmental |state )?interest\b",
            r"\bnot a suspect class\b",
            r"\bequal protection\b",
            r"\bclassification\b",
            r"\bdisability\b",
            r"\bsexual orientation\b",
            r"\bcleburne\b",
            r"\bromer\b",
            r"\bheller\b",
        ),
        exclude_any=(
            r"\bstrict scrutiny\b",
            r"\bintermediate scrutiny\b",
            r"\bsection 5\b",
            r"\bcongruence and proportionality\b",
        ),
    ),
)


SOURCE_CASES: tuple[SourceCase, ...] = (
    SourceCase("loving_1967", "Loving v. Virginia", "388 U.S. 1", 1966, "civil_race_strict_scrutiny", "racial classification / marriage", "https://www.law.cornell.edu/supremecourt/text/388/1"),
    SourceCase("bakke_1978", "Regents of the University of California v. Bakke", "438 U.S. 265", 1977, "civil_race_strict_scrutiny", "medical-school admissions / race", "https://www.law.cornell.edu/supremecourt/text/438/265"),
    SourceCase("croson_1989", "City of Richmond v. J. A. Croson Co.", "488 U.S. 469", 1988, "civil_race_strict_scrutiny", "minority contracting / strict scrutiny", "https://www.law.cornell.edu/supremecourt/text/488/469"),
    SourceCase("adarand_1995", "Adarand Constructors, Inc. v. Pena", "515 U.S. 200", 1994, "civil_race_strict_scrutiny", "federal racial classifications / strict scrutiny", "https://www.law.cornell.edu/supremecourt/text/515/200"),
    SourceCase("grutter_2003", "Grutter v. Bollinger", "539 U.S. 306", 2002, "civil_race_strict_scrutiny", "law-school admissions / diversity", "https://www.law.cornell.edu/supremecourt/text/02-241"),
    SourceCase("gratz_2003", "Gratz v. Bollinger", "539 U.S. 244", 2002, "civil_race_strict_scrutiny", "undergraduate admissions / point system", "https://www.law.cornell.edu/supremecourt/text/02-516"),
    SourceCase("fisher_2016", "Fisher v. University of Texas at Austin", "579 U.S. 365", 2015, "civil_race_strict_scrutiny", "university admissions / narrow tailoring", "https://www.law.cornell.edu/supremecourt/text/14-981"),
    SourceCase("sffa_2023", "Students for Fair Admissions, Inc. v. President and Fellows of Harvard College", "600 U.S. 181", 2022, "civil_race_strict_scrutiny", "race-conscious admissions / equal protection", "https://www.law.cornell.edu/supremecourt/text/20-1199"),
    SourceCase("reed_1971", "Reed v. Reed", "404 U.S. 71", 1971, "civil_sex_intermediate_scrutiny", "sex classification / estate administrators", "https://www.law.cornell.edu/supremecourt/text/404/71"),
    SourceCase("frontiero_1973", "Frontiero v. Richardson", "411 U.S. 677", 1972, "civil_sex_intermediate_scrutiny", "sex classification / military benefits", "https://www.law.cornell.edu/supremecourt/text/411/677"),
    SourceCase("craig_1976", "Craig v. Boren", "429 U.S. 190", 1976, "civil_sex_intermediate_scrutiny", "sex classification / intermediate scrutiny", "https://www.law.cornell.edu/supremecourt/text/429/190"),
    SourceCase("hogan_1982", "Mississippi University for Women v. Hogan", "458 U.S. 718", 1981, "civil_sex_intermediate_scrutiny", "single-sex nursing school / important objective", "https://www.law.cornell.edu/supremecourt/text/458/718"),
    SourceCase("vmi_1996", "United States v. Virginia", "518 U.S. 515", 1995, "civil_sex_intermediate_scrutiny", "VMI / exceedingly persuasive justification", "https://www.law.cornell.edu/supremecourt/text/518/515"),
    SourceCase("nguyen_2001", "Nguyen v. INS", "533 U.S. 53", 2000, "civil_sex_intermediate_scrutiny", "citizenship / sex classification", "https://www.law.cornell.edu/supremecourt/text/533/53"),
    SourceCase("morales_santana_2017", "Sessions v. Morales-Santana", "582 U.S. 47", 2016, "civil_sex_intermediate_scrutiny", "citizenship / gender-based differential", "https://www.law.cornell.edu/supremecourt/text/15-1191"),
    SourceCase("boerne_1997", "City of Boerne v. Flores", "521 U.S. 507", 1996, "civil_section5_congruence", "Section 5 / congruence and proportionality", "https://www.law.cornell.edu/supremecourt/text/521/507"),
    SourceCase("kimel_2000", "Kimel v. Florida Board of Regents", "528 U.S. 62", 1999, "civil_section5_congruence", "age discrimination / abrogation", "https://www.law.cornell.edu/supremecourt/text/528/62"),
    SourceCase("garrett_2001", "Board of Trustees of the University of Alabama v. Garrett", "531 U.S. 356", 2000, "civil_section5_congruence", "ADA employment / state immunity", "https://www.law.cornell.edu/supremecourt/text/99-1240"),
    SourceCase("hibbs_2003", "Nevada Department of Human Resources v. Hibbs", "538 U.S. 721", 2002, "civil_section5_congruence", "family leave / Section 5 remedy", "https://www.law.cornell.edu/supremecourt/text/01-1368"),
    SourceCase("lane_2004", "Tennessee v. Lane", "541 U.S. 509", 2003, "civil_section5_congruence", "courthouse access / Title II abrogation", "https://www.law.cornell.edu/supremecourt/text/02-1667"),
    SourceCase("coleman_2012", "Coleman v. Court of Appeals of Maryland", "566 U.S. 30", 2011, "civil_section5_congruence", "self-care leave / state immunity", "https://www.law.cornell.edu/supremecourt/text/10-1016"),
    SourceCase("cleburne_1985", "City of Cleburne v. Cleburne Living Center", "473 U.S. 432", 1984, "civil_rational_basis_equal_protection", "disability / rational basis", "https://www.law.cornell.edu/supremecourt/text/473/432"),
    SourceCase("heller_doe_1993", "Heller v. Doe", "509 U.S. 312", 1992, "civil_rational_basis_equal_protection", "mental disability / rational basis", "https://www.law.cornell.edu/supremecourt/text/509/312"),
    SourceCase("romer_1996", "Romer v. Evans", "517 U.S. 620", 1995, "civil_rational_basis_equal_protection", "sexual orientation / rational basis", "https://www.law.cornell.edu/supremecourt/text/517/620"),
)


def mask_frame_cues(text: str) -> str:
    masked = mask_citations(text)
    replacements = [
        (r"\bequal protection\b", "[EQUAL_PROTECTION]"),
        (r"\bstrict scrutiny\b", "[SCRUTINY]"),
        (r"\bintermediate scrutiny\b", "[SCRUTINY]"),
        (r"\brational basis\b", "[SCRUTINY]"),
        (r"\bcompelling (?:governmental |state )?interest\b", "[SCRUTINY]"),
        (r"\bnarrowly tailored\b", "[SCRUTINY]"),
        (r"\bimportant governmental objective\b", "[SCRUTINY]"),
        (r"\bsubstantially related\b", "[SCRUTINY]"),
        (r"\bexceedingly persuasive\b", "[SCRUTINY]"),
        (r"\brationally related\b", "[SCRUTINY]"),
        (r"\blegitimate (?:governmental |state )?interest\b", "[SCRUTINY]"),
        (r"\bsuspect classification\b", "[CLASS]"),
        (r"\bracial classification\b", "[CLASS]"),
        (r"\bgender classification\b", "[CLASS]"),
        (r"\bsex-based classification\b", "[CLASS]"),
        (r"\brace\b", "[CLASS]"),
        (r"\bracial\b", "[CLASS]"),
        (r"\baffirmative action\b", "[CLASS]"),
        (r"\bminority\b", "[CLASS]"),
        (r"\bsex\b", "[CLASS]"),
        (r"\bgender\b", "[CLASS]"),
        (r"\bwomen\b", "[CLASS]"),
        (r"\bwoman\b", "[CLASS]"),
        (r"\bfemale\b", "[CLASS]"),
        (r"\bmale\b", "[CLASS]"),
        (r"\bsection 5\b", "[SECTION5]"),
        (r"\bfourteenth amendment\b", "[AMENDMENT]"),
        (r"\benforcement power\b", "[SECTION5]"),
        (r"\bcongruence and proportionality\b", "[SECTION5]"),
        (r"\bcongruent and proportional\b", "[SECTION5]"),
        (r"\babrogat(?:e|ion|ed)\b", "[SECTION5]"),
        (r"\bsovereign immunity\b", "[IMMUNITY]"),
        (r"\beleventh amendment\b", "[IMMUNITY]"),
        (r"\bdisability\b", "[CLASS]"),
        (r"\bmental(?:ly)? retarded\b", "[CLASS]"),
        (r"\bsexual orientation\b", "[CLASS]"),
        (r"\bhomosexual\b", "[CLASS]"),
        (r"\bLoving\b", "[CASE]"),
        (r"\bBakke\b", "[CASE]"),
        (r"\bCroson\b", "[CASE]"),
        (r"\bAdarand\b", "[CASE]"),
        (r"\bGrutter\b", "[CASE]"),
        (r"\bGratz\b", "[CASE]"),
        (r"\bFisher\b", "[CASE]"),
        (r"\bStudents for Fair Admissions\b", "[CASE]"),
        (r"\bReed\b", "[CASE]"),
        (r"\bFrontiero\b", "[CASE]"),
        (r"\bCraig\b", "[CASE]"),
        (r"\bHogan\b", "[CASE]"),
        (r"\bUnited States v\. Virginia\b", "[CASE]"),
        (r"\bVMI\b", "[CASE]"),
        (r"\bNguyen\b", "[CASE]"),
        (r"\bMorales-Santana\b", "[CASE]"),
        (r"\bBoerne\b", "[CASE]"),
        (r"\bKimel\b", "[CASE]"),
        (r"\bGarrett\b", "[CASE]"),
        (r"\bHibbs\b", "[CASE]"),
        (r"\bLane\b", "[CASE]"),
        (r"\bColeman\b", "[CASE]"),
        (r"\bCleburne\b", "[CASE]"),
        (r"\bHeller\b", "[CASE]"),
        (r"\bRomer\b", "[CASE]"),
    ]
    for pattern, repl in replacements:
        masked = re.sub(pattern, repl, masked, flags=re.IGNORECASE)
    return masked


def match_regexes(text: str, patterns: tuple[str, ...]) -> list[str]:
    return [pattern for pattern in patterns if re.search(pattern, text, flags=re.IGNORECASE)]


def rule_matches(text: str, rule: CivilRule) -> tuple[bool, list[str]]:
    if match_regexes(text, rule.exclude_any):
        return False, []
    for group in rule.required_any:
        if not match_regexes(text, group):
            return False, []
    return True, match_regexes(text, rule.evidence_patterns)


def fetch_sources(raw_path: Path, *, refresh: bool, delay: float) -> list[dict[str, Any]]:
    if raw_path.exists() and not refresh:
        return json.loads(raw_path.read_text(encoding="utf-8"))

    session = requests.Session()
    session.headers.update({"User-Agent": "scotus-civil-source-pack/0.1 research"})
    pages: list[dict[str, Any]] = []
    for source in tqdm(SOURCE_CASES, desc="download civil source opinions"):
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
                    "decade": f"{(int(page['term']) // 10) * 10}s",
                    "justice": "source_opinion",
                    "section_author": "",
                    "section_posture": "source_opinion",
                    "issue_area_label": "Civil Rights",
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


def priority_score(row: dict[str, Any], rule: CivilRule, evidence: list[str]) -> tuple[int, int, int, int, str]:
    expected_bonus = 3 if row.get("expected_frame") == rule.frame else 0
    return (
        expected_bonus,
        len(evidence),
        -abs(int(row.get("token_count") or 0) - 220),
        -int(row.get("chunk_index_in_section") or 0),
        str(row.get("case_name", "")),
    )


def build_records(chunk_rows: list[dict[str, Any]], *, max_per_frame: int, window_chars: int) -> list[dict[str, Any]]:
    matches_by_chunk: dict[str, list[tuple[CivilRule, list[str]]]] = {}
    for row in tqdm(chunk_rows, desc="scan civil chunks"):
        text = str(row.get("text", ""))
        for rule in CIVIL_RULES:
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
                    "issue_family": "Civil Rights",
                    "label": 1,
                    "label_source": "civil_source_rule_v1",
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


def write_report(
    path: Path,
    *,
    pages: list[dict[str, Any]],
    chunk_rows: list[dict[str, Any]],
    records: list[dict[str, Any]],
    args: argparse.Namespace,
) -> None:
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    for record in records:
        counts[str(record["frame"])][str(record["split"])] += 1
    frame_rows: list[list[Any]] = []
    for rule in CIVIL_RULES:
        counter = counts.get(rule.frame, Counter())
        total = sum(counter.values())
        conflicts = sum(1 for row in records if row["frame"] == rule.frame and row["has_multi_frame_conflict"])
        cases = len({row["source_case_id"] for row in records if row["frame"] == rule.frame})
        frame_rows.append([rule.frame, total, cases, counter["train"], counter["dev"], counter["test"], conflicts])

    source_rows = [
        [page["case_id"], page["case_name"], page["citation"], page["expected_frame"], page["token_count"], page["source_url"]]
        for page in pages
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
        "# SCOTUS Civil Rights Source Pack v1",
        "",
        "## Purpose",
        "",
        "Civil Rights is the backup source-pack branch after Economic Activity failed its promotion gate. This pack is deliberately labeled `silver_review_required` because scrutiny-level doctrine is likely to be lexically separable.",
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
    lines.extend(["", "## Sample Evidence Windows", ""])
    lines.extend(markdown_table(["Frame", "Split", "Case", "Citation", "Evidence", "Conflict", "Window"], sample_rows))
    lines.extend(
        [
            "",
            "## Use Rules",
            "",
            "1. Do not run a BF16 probe from this pack until the review queue is sampled for dominant-frame validity.",
            "2. Probe only `text_cue_masked`, with conflict-row exclusion and strict source-cluster-heldout splits.",
            "3. Promotion requires activation performance clearly above the cue-masked text baseline.",
            "4. Treat any strict-vs-intermediate win as suspect until a bag-of-cues baseline and manual dominance review clear it.",
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
