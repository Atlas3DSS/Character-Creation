#!/usr/bin/env python3
"""Build an expanded Fourth Amendment source-frame pack.

The repaired target-justice corpus contains Fourth Amendment labels, but the
technology/privacy rows are polluted by non-Fourth-Amendment copyright and
database cases. This source pack uses named Fourth Amendment opinions from
Cornell LII, applies stricter doctrine-oriented rules, and emits cue-masked
text for leakage diagnostics.
"""

from __future__ import annotations

import argparse
import html
import json
import re
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests
from lxml import html as lxml_html
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SCOTUS_DIR = PROJECT_ROOT / "data" / "scotus"
DEFAULT_RAW = SCOTUS_DIR / "raw" / "scotus_fourth_source_pages_v1.json"
DEFAULT_LABELS = SCOTUS_DIR / "scotus_fourth_source_frame_labels_v1.jsonl"
DEFAULT_QUEUE = SCOTUS_DIR / "scotus_fourth_source_frame_review_queue_v1.jsonl"
DEFAULT_REPORT = PROJECT_ROOT / "reports" / "scotus_fourth_source_pack_v1.md"

if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

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
class FourthRule:
    frame: str
    definition: str
    required_any: tuple[tuple[str, ...], ...]
    evidence_patterns: tuple[str, ...]
    exclude_any: tuple[str, ...] = ()


FOURTH_RULES: tuple[FourthRule, ...] = (
    FourthRule(
        frame="fourth_search_incident_chimel",
        definition=(
            "Search incident to arrest, including Chimel immediate-control, Robinson, Belton, Gant, "
            "officer safety, or evidence-preservation reasoning."
        ),
        required_any=(
            (
                r"\bsearch incident to (?:a |an )?arrest\b",
                r"\bsearch-incident-to-arrest\b",
                r"\bchimel\b",
                r"\brobinson\b",
                r"\bbelton\b",
                r"\bgant\b",
                r"\bimmediate control\b",
                r"\bgrab area\b",
            ),
            (
                r"\barrest\b",
                r"\barrestee\b",
                r"\bofficer safety\b",
                r"\bevidence (?:might be |could be )?(?:conceal|destroy|preserv)",
                r"\bdestruction of evidence\b",
            ),
        ),
        evidence_patterns=(
            r"\bsearch incident to (?:a |an )?arrest\b",
            r"\bsearch-incident-to-arrest\b",
            r"\bchimel\b",
            r"\brobinson\b",
            r"\bbelton\b",
            r"\bgant\b",
            r"\bimmediate control\b",
            r"\bofficer safety\b",
            r"\bdestruction of evidence\b",
        ),
    ),
    FourthRule(
        frame="fourth_technology_privacy",
        definition=(
            "Fourth Amendment privacy applied to digital devices, cell-site records, GPS tracking, "
            "thermal imaging, or other sense-enhancing technology."
        ),
        required_any=(
            (
                r"\bcell phone\b",
                r"\bdigital\b",
                r"\bcell[- ]site\b",
                r"\blocation information\b",
                r"\bgps\b",
                r"\btracking device\b",
                r"\bthermal imag",
                r"\bsense-enhancing technology\b",
                r"\bsmartphone\b",
                r"\bcomputer\b",
            ),
            (r"\bfourth amendment\b", r"\bsearch\b", r"\bprivacy\b", r"\breasonable expectation of privacy\b"),
        ),
        evidence_patterns=(
            r"\bcell phone\b",
            r"\bdigital\b",
            r"\bcell[- ]site\b",
            r"\blocation information\b",
            r"\bgps\b",
            r"\btracking device\b",
            r"\bthermal imag",
            r"\bsense-enhancing technology\b",
            r"\breasonable expectation of privacy\b",
        ),
        exclude_any=(r"\bcopyright\b", r"\bdatabase\b", r"\bdesktop direct\b"),
    ),
    FourthRule(
        frame="fourth_plain_view_independent_source",
        definition=(
            "Plain-view seizure/search limits or independent-source doctrine, including Horton, Hicks, "
            "Murray, or Segura."
        ),
        required_any=(
            (
                r"\bplain view\b",
                r"\bindependent source\b",
                r"\bhorton\b",
                r"\bhicks\b",
                r"\bmurray\b",
                r"\bsegura\b",
            ),
            (r"\bfourth amendment\b", r"\bwarrant\b", r"\bsearch\b", r"\bsuppress"),
        ),
        evidence_patterns=(
            r"\bplain view\b",
            r"\bindependent source\b",
            r"\bhorton\b",
            r"\bhicks\b",
            r"\bmurray\b",
            r"\bsegura\b",
            r"\bwarrant\b",
            r"\bsuppress(?:ion)?\b",
        ),
    ),
    FourthRule(
        frame="fourth_home_exigency",
        definition=(
            "Home-entry warrant requirement and exigency exceptions, including emergency aid, hot pursuit, "
            "knock-and-announce exigency, or consent-to-home-entry doctrine."
        ),
        required_any=(
            (
                r"\bexigen(?:t|cy|cies)\b",
                r"\bemergency aid\b",
                r"\bhot pursuit\b",
                r"\bknock(?:ing)? and announc",
                r"\bno-knock\b",
                r"\bwarrantless entry\b",
                r"\bentry into (?:a |the )?home\b",
                r"\bconsent(?:ed)? to (?:the )?(?:entry|search)\b",
            ),
            (r"\bhome\b", r"\bhouse\b", r"\bresidence\b", r"\bdwelling\b", r"\bpremises\b"),
        ),
        evidence_patterns=(
            r"\bexigen(?:t|cy|cies)\b",
            r"\bemergency aid\b",
            r"\bhot pursuit\b",
            r"\bknock(?:ing)? and announc",
            r"\bno-knock\b",
            r"\bwarrantless entry\b",
            r"\bentry into (?:a |the )?home\b",
            r"\bconsent(?:ed)? to (?:the )?(?:entry|search)\b",
        ),
    ),
)


SOURCE_CASES: tuple[SourceCase, ...] = (
    SourceCase(
        "chimel_1969",
        "Chimel v. California",
        "395 U.S. 752",
        1968,
        "fourth_search_incident_chimel",
        "search incident / immediate control",
        "https://www.law.cornell.edu/supremecourt/text/395/752",
    ),
    SourceCase(
        "robinson_1973",
        "United States v. Robinson",
        "414 U.S. 218",
        1973,
        "fourth_search_incident_chimel",
        "custodial arrest search",
        "https://www.law.cornell.edu/supremecourt/text/414/218",
    ),
    SourceCase(
        "belton_1981",
        "New York v. Belton",
        "453 U.S. 454",
        1980,
        "fourth_search_incident_chimel",
        "vehicle passenger compartment search incident",
        "https://www.law.cornell.edu/supremecourt/text/453/454",
    ),
    SourceCase(
        "gant_2009",
        "Arizona v. Gant",
        "556 U.S. 332",
        2008,
        "fourth_search_incident_chimel",
        "limits on vehicle search incident",
        "https://www.law.cornell.edu/supct/html/07-542.ZS.html",
    ),
    SourceCase(
        "riley_2014",
        "Riley v. California",
        "573 U.S. 373",
        2013,
        "fourth_technology_privacy",
        "cell phones / digital privacy against search incident",
        "https://www.law.cornell.edu/supremecourt/text/13-132",
    ),
    SourceCase(
        "kyllo_2001",
        "Kyllo v. United States",
        "533 U.S. 27",
        2000,
        "fourth_technology_privacy",
        "thermal imaging / sense-enhancing technology",
        "https://www.law.cornell.edu/supremecourt/text/533/27",
    ),
    SourceCase(
        "jones_2012",
        "United States v. Jones",
        "565 U.S. 400",
        2011,
        "fourth_technology_privacy",
        "GPS tracking / search",
        "https://www.law.cornell.edu/supremecourt/text/10-1259",
    ),
    SourceCase(
        "carpenter_2018",
        "Carpenter v. United States",
        "585 U.S. 296",
        2017,
        "fourth_technology_privacy",
        "cell-site location privacy",
        "https://www.law.cornell.edu/supremecourt/text/16-402",
    ),
    SourceCase(
        "hicks_1987",
        "Arizona v. Hicks",
        "480 U.S. 321",
        1986,
        "fourth_plain_view_independent_source",
        "plain view / serial number search",
        "https://www.law.cornell.edu/supremecourt/text/480/321",
    ),
    SourceCase(
        "horton_1990",
        "Horton v. California",
        "496 U.S. 128",
        1989,
        "fourth_plain_view_independent_source",
        "plain-view seizure",
        "https://www.law.cornell.edu/supremecourt/text/496/128",
    ),
    SourceCase(
        "murray_1988",
        "Murray v. United States",
        "487 U.S. 533",
        1987,
        "fourth_plain_view_independent_source",
        "independent source",
        "https://www.law.cornell.edu/supremecourt/text/487/533",
    ),
    SourceCase(
        "segura_1984",
        "Segura v. United States",
        "468 U.S. 796",
        1983,
        "fourth_plain_view_independent_source",
        "independent source / suppression",
        "https://www.law.cornell.edu/supremecourt/text/468/796",
    ),
    SourceCase(
        "payton_1980",
        "Payton v. New York",
        "445 U.S. 573",
        1979,
        "fourth_home_exigency",
        "home entry / warrant requirement",
        "https://www.law.cornell.edu/supremecourt/text/445/573",
    ),
    SourceCase(
        "mincey_1978",
        "Mincey v. Arizona",
        "437 U.S. 385",
        1977,
        "fourth_home_exigency",
        "home murder scene / exigency limits",
        "https://www.law.cornell.edu/supremecourt/text/437/385",
    ),
    SourceCase(
        "brigham_city_2006",
        "Brigham City v. Stuart",
        "547 U.S. 398",
        2005,
        "fourth_home_exigency",
        "emergency aid home entry",
        "https://www.law.cornell.edu/supct/html/05-502.ZO.html",
    ),
    SourceCase(
        "king_2011",
        "Kentucky v. King",
        "563 U.S. 452",
        2010,
        "fourth_home_exigency",
        "exigent circumstances / police-created exigency",
        "https://www.law.cornell.edu/supct/html/09-1272.ZO.html",
    ),
    SourceCase(
        "lange_2021",
        "Lange v. California",
        "594 U.S. ___",
        2020,
        "fourth_home_exigency",
        "hot pursuit misdemeanor home entry",
        "https://www.law.cornell.edu/supremecourt/text/20-18",
    ),
    SourceCase(
        "randolph_2006",
        "Georgia v. Randolph",
        "547 U.S. 103",
        2005,
        "fourth_home_exigency",
        "home consent / co-occupant refusal",
        "https://www.law.cornell.edu/supct/html/04-1067.ZO.html",
    ),
)


def token_count(text: str) -> int:
    return len(re.findall(r"\w+|[^\w\s]", text))


def clean_plain_text(text: str) -> str:
    text = html.unescape(text or "")
    text = text.replace("\xa0", " ")
    text = re.sub(r"-\n(?=[a-z])", "", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n[ \t]+", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    lines: list[str] = []
    skip_exact = {
        "SUPREME COURT OF THE UNITED STATES",
        "Syllabus",
        "Opinion",
        "HTML",
        "PDF",
        "Primary tabs",
        "U.S. Supreme Court",
    }
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            lines.append("")
            continue
        if stripped in skip_exact:
            continue
        if re.fullmatch(r"\*?\d+\*?", stripped):
            continue
        if re.search(r"^\s*(Menu|Search|About LII|Wex|Help)\s*$", stripped, flags=re.IGNORECASE):
            continue
        lines.append(stripped)
    return "\n".join(lines).strip()


def split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z\"“])", text.strip())
    return [part.strip() for part in parts if part.strip()]


def paragraph_chunks(text: str, *, min_tokens: int, target_min: int, target_max: int) -> list[str]:
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n+", text) if p.strip()]
    chunks: list[str] = []
    pending: list[str] = []
    pending_tokens = 0

    def flush_pending() -> None:
        nonlocal pending, pending_tokens
        if pending and pending_tokens >= min_tokens:
            chunks.append("\n\n".join(pending).strip())
        pending = []
        pending_tokens = 0

    for paragraph in paragraphs:
        p_tokens = token_count(paragraph)
        if p_tokens > target_max:
            flush_pending()
            sent_buf: list[str] = []
            sent_tokens = 0
            for sentence in split_sentences(paragraph):
                s_tokens = token_count(sentence)
                if sent_buf and sent_tokens + s_tokens > target_max:
                    chunk = " ".join(sent_buf).strip()
                    if token_count(chunk) >= min_tokens:
                        chunks.append(chunk)
                    sent_buf = []
                    sent_tokens = 0
                sent_buf.append(sentence)
                sent_tokens += s_tokens
            if sent_buf:
                chunk = " ".join(sent_buf).strip()
                if token_count(chunk) >= min_tokens:
                    chunks.append(chunk)
                else:
                    pending.append(chunk)
                    pending_tokens += token_count(chunk)
            continue

        if pending_tokens + p_tokens > target_max and pending_tokens >= min_tokens:
            flush_pending()
        pending.append(paragraph)
        pending_tokens += p_tokens
        if pending_tokens >= target_min:
            flush_pending()
    flush_pending()
    return chunks


def mask_citations(text: str) -> str:
    masked = text
    masked = re.sub(r"\b\d+\s+U\.\s*S\.\s+\d+\b", "[CITE]", masked)
    masked = re.sub(r"\b\d+\s+S\.\s*Ct\.\s+\d+\b", "[CITE]", masked)
    masked = re.sub(r"\b\d+\s+L\.\s*Ed\.?\s*(?:2d)?\s+\d+\b", "[CITE]", masked)
    masked = re.sub(r"\b\d{4}\s+U\.S\. LEXIS\s+\d+\b", "[CITE]", masked)
    masked = re.sub(r"\b\d+\s+U\.S\.C\.?\s*§+\s*[\w\-()]+", "[STATUTE]", masked)
    masked = re.sub(r"§+", "[SECTION]", masked)
    return masked


def mask_frame_cues(text: str) -> str:
    masked = mask_citations(text)
    replacements = [
        (r"\bFourth Amendment\b", "[FOURTH]"),
        (r"\bsearch incident to (?:a |an )?arrest\b", "[FRAME]"),
        (r"\bsearch-incident-to-arrest\b", "[FRAME]"),
        (r"\bplain view\b", "[FRAME]"),
        (r"\bindependent source\b", "[FRAME]"),
        (r"\bexigen(?:t|cy|cies)\b", "[FRAME]"),
        (r"\bemergency aid\b", "[FRAME]"),
        (r"\bhot pursuit\b", "[FRAME]"),
        (r"\bknock(?:ing)? and announc(?:e|ement)?\b", "[FRAME]"),
        (r"\bwarrantless entry\b", "[FRAME]"),
        (r"\bimmediate control\b", "[FRAME]"),
        (r"\bofficer safety\b", "[FRAME]"),
        (r"\bdestruction of evidence\b", "[FRAME]"),
        (r"\bcell phone\b", "[TECH]"),
        (r"\bdigital\b", "[TECH]"),
        (r"\bcell[- ]site\b", "[TECH]"),
        (r"\bgps\b", "[TECH]"),
        (r"\btracking device\b", "[TECH]"),
        (r"\bthermal imag(?:er|ing)?\b", "[TECH]"),
        (r"\bsense-enhancing technology\b", "[TECH]"),
        (r"\bChimel\b", "[CASE]"),
        (r"\bRobinson\b", "[CASE]"),
        (r"\bBelton\b", "[CASE]"),
        (r"\bGant\b", "[CASE]"),
        (r"\bRiley\b", "[CASE]"),
        (r"\bKyllo\b", "[CASE]"),
        (r"\bCarpenter\b", "[CASE]"),
        (r"\bJones\b", "[CASE]"),
        (r"\bHicks\b", "[CASE]"),
        (r"\bHorton\b", "[CASE]"),
        (r"\bMurray\b", "[CASE]"),
        (r"\bSegura\b", "[CASE]"),
        (r"\bMincey\b", "[CASE]"),
        (r"\bPayton\b", "[CASE]"),
    ]
    for pattern, repl in replacements:
        masked = re.sub(pattern, repl, masked, flags=re.IGNORECASE)
    return masked


def match_regexes(text: str, patterns: tuple[str, ...]) -> list[str]:
    return [pattern for pattern in patterns if re.search(pattern, text, flags=re.IGNORECASE)]


def rule_matches(text: str, rule: FourthRule) -> tuple[bool, list[str], list[str]]:
    excludes = match_regexes(text, rule.exclude_any)
    if excludes:
        return False, [], excludes
    for group in rule.required_any:
        if not match_regexes(text, group):
            return False, [], []
    evidence = match_regexes(text, rule.evidence_patterns)
    return True, evidence, []


def extract_law_cornell_text(markup: str) -> str:
    doc = lxml_html.fromstring(markup.encode("utf-8", errors="ignore"))
    for bad in doc.xpath("//script|//style|//noscript|//nav|//header|//footer|//form|//table"):
        parent = bad.getparent()
        if parent is not None:
            parent.remove(bad)
    nodes = doc.xpath("//main") or doc.xpath('//*[@id="content"]') or [doc]
    raw = "\n".join(node.text_content() for node in nodes)
    return clean_plain_text(raw)


def fetch_sources(raw_path: Path, *, refresh: bool, delay: float) -> list[dict[str, Any]]:
    if raw_path.exists() and not refresh:
        return json.loads(raw_path.read_text(encoding="utf-8"))

    session = requests.Session()
    session.headers.update({"User-Agent": "scotus-fourth-source-pack/0.1 research"})
    pages: list[dict[str, Any]] = []
    for source in tqdm(SOURCE_CASES, desc="download source opinions"):
        response = session.get(source.url, timeout=60)
        response.raise_for_status()
        text = extract_law_cornell_text(response.text)
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


def build_chunk_rows(
    pages: list[dict[str, Any]],
    *,
    min_tokens: int,
    target_min: int,
    target_max: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for page in pages:
        chunks = paragraph_chunks(
            str(page["text"]),
            min_tokens=min_tokens,
            target_min=target_min,
            target_max=target_max,
        )
        for idx, chunk in enumerate(chunks):
            rows.append(
                {
                    "chunk_id": f"{page['case_id']}-{idx:04d}",
                    "opinion_id": page["case_id"],
                    "cluster_id": page["case_id"],
                    "scdb_id": "",
                    "case_name": page["case_name"],
                    "date_filed": "",
                    "term": page["term"],
                    "decade": f"{(int(page['term']) // 10) * 10}s",
                    "justice": "source_opinion",
                    "section_author": "",
                    "section_posture": "source_opinion",
                    "issue_area_label": "Criminal Procedure",
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


def priority_score(row: dict[str, Any], rule: FourthRule, evidence: list[str]) -> tuple[int, int, int, int, str]:
    expected_bonus = 3 if row.get("expected_frame") == rule.frame else 0
    return (
        expected_bonus,
        len(evidence),
        -abs(int(row.get("token_count") or 0) - 220),
        -int(row.get("chunk_index_in_section") or 0),
        str(row.get("case_name", "")),
    )


def build_records(chunk_rows: list[dict[str, Any]], *, max_per_frame: int, window_chars: int) -> list[dict[str, Any]]:
    matches_by_chunk: dict[str, list[tuple[FourthRule, list[str]]]] = {}
    for row in tqdm(chunk_rows, desc="scan fourth chunks"):
        text = str(row.get("text", ""))
        for rule in FOURTH_RULES:
            matched, evidence, _excludes = rule_matches(text, rule)
            if matched:
                matches_by_chunk.setdefault(str(row["chunk_id"]), []).append((rule, evidence))

    by_chunk = {str(row["chunk_id"]): row for row in chunk_rows}
    candidates_by_frame: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for chunk_id, matches in matches_by_chunk.items():
        row = by_chunk[chunk_id]
        matched_frames = sorted(rule.frame for rule, _evidence in matches)
        for rule, evidence in matches:
            record = {
                "record_id": f"{chunk_id}::{rule.frame}",
                "frame": rule.frame,
                "issue_family": "Criminal Procedure",
                "label": 1,
                "label_source": "expanded_fourth_source_rule_v1",
                "label_confidence": "silver_high",
                "label_definition": rule.definition,
                "evidence_patterns": evidence,
                "matched_frames": matched_frames,
                "has_multi_frame_conflict": len(matched_frames) > 1,
                "global_split": stable_split(str(row["cluster_id"])),
                "split": "unassigned",
                "opinion_id": row.get("opinion_id"),
                "cluster_id": row.get("cluster_id"),
                "scdb_id": row.get("scdb_id"),
                "case_name": row.get("case_name"),
                "date_filed": row.get("date_filed"),
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
            candidates_by_frame[rule.frame].append(record)

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


def split_counts(records: list[dict[str, Any]]) -> dict[str, Counter[str]]:
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    for record in records:
        counts[str(record["frame"])][str(record["split"])] += 1
    return counts


def case_frame_rows(records: list[dict[str, Any]]) -> list[list[Any]]:
    counts = Counter((str(row.get("source_case_id")), str(row.get("case_name")), str(row["frame"])) for row in records)
    return [[case_id, case_name, frame, count] for (case_id, case_name, frame), count in sorted(counts.items())]


def write_report(
    path: Path,
    *,
    pages: list[dict[str, Any]],
    chunk_rows: list[dict[str, Any]],
    records: list[dict[str, Any]],
    args: argparse.Namespace,
) -> None:
    counts = split_counts(records)
    frame_rows: list[list[Any]] = []
    for rule in FOURTH_RULES:
        counter = counts.get(rule.frame, Counter())
        total = sum(counter.values())
        conflict_count = sum(1 for row in records if row["frame"] == rule.frame and row["has_multi_frame_conflict"])
        frame_rows.append([rule.frame, total, counter["train"], counter["dev"], counter["test"], conflict_count])

    source_rows = [
        [page["case_id"], page["case_name"], page["citation"], page["expected_frame"], page["token_count"], page["source_url"]]
        for page in pages
    ]
    sample_rows = []
    for record in records[: min(18, len(records))]:
        sample_rows.append(
            [
                record["frame"],
                record["split"],
                record.get("case_name", ""),
                record.get("source_citation", ""),
                ", ".join(record.get("evidence_patterns", [])[:3]),
                "yes" if record.get("has_multi_frame_conflict") else "no",
                str(record.get("evidence_window", ""))[:220],
            ]
        )

    lines = [
        "# SCOTUS Fourth Amendment Source Pack v1",
        "",
        "## Purpose",
        "",
        "This expands Fourth Amendment source-grounded labels beyond the target-justice chunks and excludes the earlier non-Fourth-Amendment technology false positives. It is a silver-label source pack for cue-masked diagnostics and review, not final circuit evidence.",
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
    lines.extend(markdown_table(["Frame", "Total", "Train", "Dev", "Test", "Multi-frame conflicts"], frame_rows))
    lines.extend(["", "## Case/Frame Coverage", ""])
    lines.extend(markdown_table(["Case id", "Case", "Frame", "Records"], case_frame_rows(records)))
    lines.extend(["", "## Sample Evidence Windows", ""])
    lines.extend(markdown_table(["Frame", "Split", "Case", "Citation", "Evidence", "Conflict", "Window"], sample_rows))
    lines.extend(
        [
            "",
            "## Use Rules",
            "",
            "1. Treat labels as `silver_high`; manually review before any promotion decision.",
            "2. Run probes on `text_cue_masked`; promotion requires surviving cue masking and text-baseline checks.",
            "3. Rows with `has_multi_frame_conflict=true` should be adjudicated before binary frame training.",
            "4. Keep this source pack separate from target-justice style labels; it is for legal-frame source grounding.",
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
