#!/usr/bin/env python3
"""Repair CourtListener combined SCOTUS opinions into target-authored sections.

This is Phase 3.5A-C from SCOTUS.md:
- segment combined records by authored section
- remove obvious boilerplate / non-reasoning material
- emit section and repaired chunk inventories for v2 matching/baselines
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from bs4 import BeautifulSoup
from tqdm import tqdm

from scotus_data_audit import citation_count, clean_plain_text, mask_text, paragraph_chunks, token_count


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RAW = PROJECT_ROOT / "data" / "scotus" / "raw" / "courtlistener_scotus_target_opinions.json"
DEFAULT_OPINIONS = PROJECT_ROOT / "data" / "scotus" / "scotus_opinion_inventory.jsonl"
DEFAULT_SECTIONS = PROJECT_ROOT / "data" / "scotus" / "scotus_section_inventory.jsonl"
DEFAULT_CHUNKS = PROJECT_ROOT / "data" / "scotus" / "scotus_chunk_inventory_v2.jsonl"
DEFAULT_EXCLUDED = PROJECT_ROOT / "data" / "scotus" / "processed" / "scotus_excluded_chunk_inventory_v2.jsonl"
DEFAULT_SECTION_REPORT = PROJECT_ROOT / "reports" / "scotus_section_audit.md"

TARGET_JUSTICES = {"Scalia", "Ginsburg", "Thomas", "Souter"}
JUSTICE_LAST_NAMES = [
    "Alito",
    "Barrett",
    "Blackmun",
    "Brennan",
    "Breyer",
    "Burger",
    "Ginsburg",
    "Gorsuch",
    "Jackson",
    "Kagan",
    "Kennedy",
    "Marshall",
    "O'Connor",
    "O’Connor",
    "OConnor",
    "Powell",
    "Rehnquist",
    "Roberts",
    "Scalia",
    "Sotomayor",
    "Souter",
    "Stevens",
    "Thomas",
    "White",
]
JUSTICE_PATTERN = "|".join(re.escape(name) for name in sorted(JUSTICE_LAST_NAMES, key=len, reverse=True))

POSITIVE_MARKERS = [
    "we hold",
    "i would hold",
    "the question presented",
    "the issue is",
    "because",
    "therefore",
    "thus",
    "accordingly",
    "hence",
    "text",
    "history",
    "tradition",
    "precedent",
    "statute",
    "constitutional",
    "common law",
    "the court",
    "congress",
    "agency",
    "state",
    "federal government",
    "standard",
    "rule",
    "test",
    "burden",
    "jurisdiction",
    "standing",
]

MOJIBAKE_REPLACEMENTS = {
    "\u00c2\u00a7": "\u00a7",
    "\u00c2\u00b6": "\u00b6",
    "\u00c2": "",
    "\u00e2\u20ac\u201d": "\u2014",
    "\u00e2\u20ac\u201c": "\u2013",
    "\u00e2\u20ac\u02dc": "'",
    "\u00e2\u20ac\u2122": "'",
    "\u00e2\u20ac\u0153": '"',
    "\u00e2\u20ac\u009d": '"',
    "\u00e2\u20ac\u00a6": "...",
    "\u00c3\u00a9": "\u00e9",
    "\u00c3\u00a8": "\u00e8",
    "\u00c3\u00a1": "\u00e1",
    "\u00c3\u00ad": "\u00ed",
    "\u00c3\u00b3": "\u00f3",
    "\u00c3\u00ba": "\u00fa",
    "\u00c3\u00b1": "\u00f1",
    "\u0085": "...",
    "\u0086": "",
    "\u0091": "'",
    "\u0092": "'",
    "\u0093": '"',
    "\u0094": '"',
    "\u0095": "*",
    "\u0096": "\u2013",
    "\u0097": "\u2014",
    "\u009e": "\u00a7",
}

JUSTICE_TITLE_PATTERN = r"(?:Chief\s+Justice|Justice)"
SECTION_ACTION_WORDS = [
    "announced",
    "concurring",
    "delivered",
    "dissenting",
    "statement",
    "took no part",
]


@dataclass
class Block:
    tag: str
    text: str
    start_char: int
    end_char: int
    index: int


@dataclass
class Heading:
    author: str | None
    posture: str
    confidence: str
    text: str
    is_boundary: bool
    reason: str


def normalize_last_name(name: str) -> str:
    fixed = name.replace("’", "'")
    if fixed.upper() in {"OCONNOR", "O'CONNOR"}:
        return "O'Connor"
    return fixed.title().replace("'Connor", "'Connor")


def normalize_mojibake(text: str) -> str:
    normalized = text or ""
    for bad, good in MOJIBAKE_REPLACEMENTS.items():
        normalized = normalized.replace(bad, good)
    normalized = re.sub(r"[\u0080-\u009f]", "", normalized)
    return normalized


def normalize_legal_text(text: str) -> str:
    text = normalize_mojibake(text)
    text = clean_plain_text(text)
    text = normalize_mojibake(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def soup_for_markup(markup: str) -> BeautifulSoup:
    parser = "xml" if markup.lstrip().startswith(("<?xml", "<opinion")) else "lxml"
    soup = BeautifulSoup(markup, parser)
    for tag in soup(["script", "style", "table"]):
        tag.decompose()
    for tag in soup.select(".star-pagination, .page-label"):
        tag.decompose()
    return soup


def extract_blocks(markup: str) -> list[Block]:
    soup = soup_for_markup(markup)
    blocks: list[Block] = []
    cursor = 0
    for tag in soup.find_all(["author", "judges", "p", "h1", "h2", "h3"]):
        text = normalize_legal_text(tag.get_text(" ", strip=True))
        if not text:
            continue
        start = markup.find(tag.get_text(strip=True)[:40], cursor)
        if start < 0:
            start = cursor
        end = start + len(str(tag))
        cursor = max(cursor, end)
        blocks.append(Block(tag=tag.name, text=text, start_char=start, end_char=end, index=len(blocks)))
    return blocks


def strip_page_prefix(text: str) -> str:
    text = re.sub(r"^\*\d+\s+", "", text.strip())
    text = re.sub(r"^\d+\s+", "", text)
    return text.strip()


def infer_posture(text: str, tag: str) -> str:
    low = text.lower()
    if low.startswith("per curiam"):
        return "per_curiam"
    if "took no part" in low:
        return "no_part"
    if "dissenting in part" in low and "concurring in part" in low:
        return "concurrence_in_part_dissent_in_part"
    if "dissenting" in low or "dissent" in low:
        return "dissent"
    if "concurring in part" in low:
        return "concurrence_in_part"
    if "concurring in the judgment" in low or "concurring in judgment" in low:
        return "concurrence_in_judgment"
    if "concurring" in low or "concurrence" in low:
        return "concurrence"
    if "plurality" in low:
        return "plurality"
    if "announced the judgment" in low:
        return "judgment"
    if "delivered an opinion" in low:
        return "plurality"
    if "delivered the opinion of the court" in low or tag == "author":
        return "majority"
    if "statement" in low:
        return "statement"
    return "unknown"


def detect_heading(block: Block) -> Heading:
    text = strip_page_prefix(block.text)
    compact = re.sub(r"\s+", " ", text)
    low = compact.lower()

    if block.tag == "author":
        match = re.search(rf"\b(?:{JUSTICE_TITLE_PATTERN}\s+)?(?P<name>{JUSTICE_PATTERN})\b", compact, flags=re.I)
        if match:
            author = normalize_last_name(match.group("name"))
            return Heading(author, infer_posture(compact, block.tag), "high", compact, True, "author_tag")

    # Headmatter summaries often say a justice "filed" an opinion. They are not
    # section starts. Actual authored sections use "delivered", "concurring",
    # "dissenting", or a bare author heading.
    if " filed " in low and " delivered " not in low and " dissenting" not in low and " concurring" not in low:
        return Heading(None, "unknown", "low", compact, False, "filed_summary")

    if re.fullmatch(r"per curiam\.?", compact, flags=re.I):
        return Heading("Per Curiam", "per_curiam", "medium", compact, True, "per_curiam_boundary")

    if "took no part" in low:
        match = re.search(
            rf"^(?:\*\d+\s*)?(?:{JUSTICE_TITLE_PATTERN}\s+)?(?P<name>{JUSTICE_PATTERN})\b",
            compact,
            flags=re.I,
        )
        if match:
            author = normalize_last_name(match.group("name"))
            return Heading(author, "no_part", "low", compact, True, "took_no_part_boundary")

    if (
        "delivered the opinion of the court" in low
        or "announced the judgment" in low
        or "delivered an opinion" in low
    ):
        match = re.search(
            rf"^(?:\*\d+\s*)?(?:(?:{JUSTICE_TITLE_PATTERN})\s+)?(?P<name>{JUSTICE_PATTERN})\b|"
            rf"^(?:\*\d+\s*)?(?P<name2>{JUSTICE_PATTERN}),\s*J\.",
            compact,
            flags=re.I,
        )
        if match:
            author = normalize_last_name(match.group("name") or match.group("name2"))
            return Heading(author, infer_posture(compact, block.tag), "high", compact, True, "authored_action_line")

    match = re.search(
        rf"^(?:\*\d+\s*)?(?:{JUSTICE_TITLE_PATTERN})\s+(?P<name>{JUSTICE_PATTERN})\b(?P<rest>.{{0,260}})$",
        compact,
        flags=re.I,
    )
    if match:
        rest = match.group("rest")
        rest_low = rest.lower()
        if any(word in rest_low for word in SECTION_ACTION_WORDS):
            author = normalize_last_name(match.group("name"))
            confidence = "low" if "took no part" in rest_low else "high"
            return Heading(author, infer_posture(compact, block.tag), confidence, compact, True, "justice_action_line")

    match = re.search(
        rf"^(?:\*\d+\s*)?(?P<name>{JUSTICE_PATTERN}),\s*J\.,?\s*(?P<rest>.{{0,260}})$",
        compact,
        flags=re.I,
    )
    if match:
        rest_low = match.group("rest").lower()
        if any(word in rest_low for word in SECTION_ACTION_WORDS):
            author = normalize_last_name(match.group("name"))
            confidence = "low" if "took no part" in rest_low else "medium" if "delivered" in rest_low and "joined" in rest_low else "high"
            return Heading(author, infer_posture(compact, block.tag), confidence, compact, True, "j_posture_line")

    if re.search(r"^(?:\*\d+\s*)?THE CHIEF JUSTICE\b", compact, flags=re.I):
        return Heading("Chief Justice", infer_posture(compact, block.tag), "medium", compact, True, "chief_justice_boundary")

    return Heading(None, "unknown", "low", compact, False, "not_heading")


def has_third_person_target_author_reference(text: str, *, target_author: str) -> bool:
    normalized = strip_page_prefix(re.sub(r"\s+", " ", text)).strip()
    target_pat = rf"^(?:\*\d+\s*)?(?:{JUSTICE_TITLE_PATTERN}\s+)?{re.escape(target_author)}\b|^{re.escape(target_author)},\s*J\."
    if re.search(target_pat, normalized, flags=re.I):
        return False
    titled_reference = rf"\b(?:{JUSTICE_TITLE_PATTERN})\s+{re.escape(target_author)}(?:['\u2019]s)?\b"
    citation_reference = rf"\b{re.escape(target_author)},\s*J\.,"
    bare_possessive = rf"\b{re.escape(target_author)}['\u2019]s\b"
    return bool(
        re.search(titled_reference, normalized, flags=re.I)
        or re.search(citation_reference, normalized, flags=re.I)
        or re.search(bare_possessive, normalized, flags=re.I)
    )


def block_flags(text: str, *, target_author: str) -> dict[str, bool]:
    normalized = re.sub(r"\s+", " ", text).strip()
    low = normalized.lower()
    tokens = token_count(normalized)
    cites = citation_count(normalized)
    sentence_count = len(re.findall(r"[.!?](?:\s|$)", normalized))
    uppercase_letters = sum(1 for ch in normalized if ch.isupper())
    letters = sum(1 for ch in normalized if ch.isalpha())
    upper_ratio = uppercase_letters / letters if letters else 0.0
    positive_hits = sum(1 for marker in POSITIVE_MARKERS if marker in low)

    is_header_like = (
        tokens < 35
        and (
            upper_ratio > 0.55
            or any(
                phrase in low
                for phrase in [
                    "supreme court of the united states",
                    "certiorari to",
                    "argued",
                    "decided",
                    "no.",
                    "syllabus",
                ]
            )
        )
    )
    is_counsel_like = any(
        phrase in low
        for phrase in [
            "argued the cause",
            "on the brief",
            "filed a brief",
            "amicus curiae",
            "for petitioner",
            "for respondent",
            "solicitor general",
            "deputy solicitor general",
        ]
    )
    is_join_line_like = tokens < 90 and any(
        phrase in low
        for phrase in [
            "delivered the opinion of the court",
            "filed a dissenting opinion",
            "filed a concurring opinion",
            "joined",
            "with whom",
        ]
    )
    is_citation_dominated = tokens > 0 and (cites / max(tokens, 1) > 0.12 or (cites >= 5 and tokens < 120))

    target_pat = rf"^(?:\*\d+\s*)?(?:{JUSTICE_TITLE_PATTERN}\s+)?{re.escape(target_author)}\b|^{re.escape(target_author)},\s*J\."
    has_target_author_heading = bool(re.search(target_pat, strip_page_prefix(normalized), flags=re.I))
    has_any_author_heading = bool(
        re.search(
            rf"^(?:\*\d+\s*)?(?:{JUSTICE_TITLE_PATTERN})\s+(?:{JUSTICE_PATTERN})\b|^(?:{JUSTICE_PATTERN}),\s*J\.",
            strip_page_prefix(normalized),
            flags=re.I,
        )
    )
    has_non_target_author_heading = has_any_author_heading and not has_target_author_heading
    has_third_person_target_reference = has_third_person_target_author_reference(
        normalized,
        target_author=target_author,
    )
    is_low_reasoning_density = (
        sentence_count < 2
        or (tokens < 80 and positive_hits == 0)
        or (tokens < 140 and positive_hits == 0 and cites >= 2)
    )
    is_order_fragment = tokens < 16 and low in {"it is so ordered.", "i respectfully dissent.", "i concur."}

    return {
        "is_header_like": is_header_like,
        "is_counsel_like": is_counsel_like,
        "is_join_line_like": is_join_line_like,
        "is_citation_dominated": is_citation_dominated,
        "is_low_reasoning_density": is_low_reasoning_density,
        "has_target_author_heading": has_target_author_heading,
        "has_non_target_author_heading": has_non_target_author_heading,
        "has_third_person_target_author_reference": has_third_person_target_reference,
        "is_order_fragment": is_order_fragment,
    }


def chunk_flags(text: str, *, target_author: str) -> dict[str, bool]:
    flags = block_flags(text, target_author=target_author)
    low = text.lower()
    positive_hits = sum(1 for marker in POSITIVE_MARKERS if marker in low)
    sentence_count = len(re.findall(r"[.!?](?:\s|$)", text))
    flags["has_reasoning_marker"] = positive_hits > 0
    flags["reasoning_marker_count"] = positive_hits
    flags["sentence_count"] = sentence_count
    flags["passes_reasoning_filter"] = not (
        flags["is_header_like"]
        or flags["is_counsel_like"]
        or flags["is_join_line_like"]
        or flags["is_citation_dominated"]
        or flags["has_non_target_author_heading"]
        or flags["has_third_person_target_author_reference"]
        or flags["is_order_fragment"]
        or (flags["is_low_reasoning_density"] and positive_hits < 2)
    )
    return flags


def segment_blocks(blocks: list[Block]) -> list[dict[str, Any]]:
    starts: list[tuple[int, Heading]] = []
    for idx, block in enumerate(blocks):
        heading = detect_heading(block)
        if heading.is_boundary:
            starts.append((idx, heading))

    sections: list[dict[str, Any]] = []
    for pos, (start_idx, heading) in enumerate(starts):
        end_idx = starts[pos + 1][0] if pos + 1 < len(starts) else len(blocks)
        if heading.author not in TARGET_JUSTICES or heading.confidence == "low":
            continue
        section_blocks = blocks[start_idx:end_idx]
        if len(section_blocks) <= 1:
            continue
        sections.append(
            {
                "section_author": heading.author,
                "section_posture": heading.posture,
                "section_heading": heading.text,
                "section_confidence": heading.confidence,
                "section_start_char": section_blocks[0].start_char,
                "section_end_char": section_blocks[-1].end_char,
                "start_block": start_idx,
                "end_block": end_idx,
                "blocks": section_blocks,
            }
        )
    return sections


def build_section_text_and_exclusions(
    section_id: str,
    section: dict[str, Any],
    metadata: dict[str, Any],
) -> tuple[list[str], list[dict[str, Any]], Counter[str]]:
    kept_paragraphs: list[str] = []
    excluded: list[dict[str, Any]] = []
    flag_counts: Counter[str] = Counter()
    target_author = section["section_author"]

    # Skip the heading block itself; authorship is already stored in metadata.
    for local_idx, block in enumerate(section["blocks"][1:]):
        text = normalize_legal_text(block.text)
        if not text:
            continue
        flags = block_flags(text, target_author=target_author)
        exclude = (
            flags["is_header_like"]
            or flags["is_counsel_like"]
            or flags["is_join_line_like"]
            or flags["is_citation_dominated"]
            or flags["has_non_target_author_heading"]
            or flags["has_third_person_target_author_reference"]
            or flags["is_order_fragment"]
        )
        for name, value in flags.items():
            if value:
                flag_counts[name] += 1
        if exclude:
            excluded.append(
                {
                    **metadata,
                    "section_id": section_id,
                    "block_index": block.index,
                    "local_block_index": local_idx,
                    "excluded": True,
                    "exclude_reasons": [name for name, value in flags.items() if value],
                    **flags,
                    "token_count": token_count(text),
                    "citation_count": citation_count(text),
                    "text": text,
                }
            )
            continue
        kept_paragraphs.append(text)
    return kept_paragraphs, excluded, flag_counts


def chunk_position_bucket(index: int, total: int) -> str:
    if total <= 1:
        return "single"
    frac = index / max(total - 1, 1)
    if frac < 0.25:
        return "early"
    if frac < 0.75:
        return "middle"
    return "late"


def make_chunks_for_section(
    section_id: str,
    section: dict[str, Any],
    metadata: dict[str, Any],
    paragraphs: list[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    raw_text = "\n\n".join(paragraphs)
    raw_chunks = paragraph_chunks(raw_text, min_tokens=80, target_min=150, target_max=350)
    kept_chunks: list[dict[str, Any]] = []
    excluded_chunks: list[dict[str, Any]] = []
    for idx, raw_chunk in enumerate(raw_chunks):
        raw_chunk = normalize_legal_text(raw_chunk)
        flags = chunk_flags(raw_chunk, target_author=section["section_author"])
        base = {
            **metadata,
            "section_id": section_id,
            "section_author": section["section_author"],
            "section_posture": section["section_posture"],
            "section_confidence": section["section_confidence"],
            "section_heading": section["section_heading"],
            "section_start_char": section["section_start_char"],
            "section_end_char": section["section_end_char"],
            "chunk_index_in_section": idx,
            "chunk_count_in_section": len(raw_chunks),
            "chunk_position_bucket": chunk_position_bucket(idx, len(raw_chunks)),
        }
        for variant, text in [("raw_clean", raw_chunk), ("masked", mask_text(raw_chunk))]:
            record = {
                **base,
                "chunk_id": f"{section_id}-{idx:04d}-{variant}",
                "text_variant": variant,
                "token_count": token_count(text),
                "citation_count": citation_count(text),
                "text": text,
                **flags,
                "excluded": not flags["passes_reasoning_filter"],
            }
            if flags["passes_reasoning_filter"]:
                kept_chunks.append(record)
            else:
                excluded_chunks.append(record)
    return kept_chunks, excluded_chunks


def metadata_for_opinion(opinion: dict[str, Any], inventory_by_id: dict[int, dict[str, Any]]) -> dict[str, Any] | None:
    record = inventory_by_id.get(int(opinion["id"]))
    if not record:
        return None
    return {
        "opinion_id": record["opinion_id"],
        "cluster_id": record["cluster_id"],
        "scdb_id": record.get("scdb_id"),
        "case_name": record.get("case_name"),
        "date_filed": record.get("date_filed"),
        "term": record.get("term"),
        "decade": record.get("decade"),
        "justice": record.get("justice"),
        "author_str": record.get("author_str"),
        "opinion_type": record.get("opinion_type"),
        "issue_area": record.get("issue_area"),
        "issue_area_label": record.get("issue_area_label"),
        "decision_direction": record.get("decision_direction"),
        "source_url": record.get("source_url"),
        "download_url": record.get("download_url"),
    }


def build_repaired_corpus(raw_path: Path, opinions_path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    raw_records = json.loads(raw_path.read_text(encoding="utf-8"))
    inventory = read_jsonl(opinions_path)
    inventory_by_id = {int(row["opinion_id"]): row for row in inventory}

    section_records: list[dict[str, Any]] = []
    chunk_records: list[dict[str, Any]] = []
    excluded_records: list[dict[str, Any]] = []

    for opinion in tqdm(raw_records, desc="segment", unit="opinion"):
        metadata = metadata_for_opinion(opinion, inventory_by_id)
        if not metadata:
            continue
        markup = (
            opinion.get("html_with_citations")
            or opinion.get("html")
            or opinion.get("html_lawbox")
            or opinion.get("html_columbia")
            or opinion.get("plain_text")
            or ""
        )
        if not markup:
            continue
        blocks = extract_blocks(markup)
        sections = segment_blocks(blocks)
        for sec_idx, section in enumerate(sections):
            section_id = f"{metadata['opinion_id']}-sec{sec_idx:03d}-{section['section_author'].lower()}"
            section_metadata = {
                **metadata,
                "opinion_target_justice": metadata.get("justice"),
                "justice": section["section_author"],
            }
            section_meta = {k: v for k, v in section.items() if k != "blocks"}
            paragraphs, excluded_blocks, flag_counts = build_section_text_and_exclusions(section_id, section, section_metadata)
            kept_chunks, excluded_chunks = make_chunks_for_section(section_id, section, section_metadata, paragraphs)
            section_record = {
                **section_metadata,
                "section_id": section_id,
                **section_meta,
                "raw_block_count": len(section["blocks"]),
                "kept_paragraph_count": len(paragraphs),
                "excluded_block_count": len(excluded_blocks),
                "kept_chunk_count": len([c for c in kept_chunks if c["text_variant"] == "raw_clean"]),
                "excluded_chunk_count": len([c for c in excluded_chunks if c["text_variant"] == "raw_clean"]),
                "excluded_flag_counts": dict(flag_counts),
                "section_text_preview": " ".join(paragraphs)[:800],
            }
            section_records.append(section_record)
            chunk_records.extend(kept_chunks)
            excluded_records.extend(excluded_blocks)
            excluded_records.extend(excluded_chunks)
    return section_records, chunk_records, excluded_records


def markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return "\n".join(lines)


def display_path(path: Path) -> str:
    resolved = path if path.is_absolute() else PROJECT_ROOT / path
    try:
        return str(resolved.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def write_section_report(
    path: Path,
    sections: list[dict[str, Any]],
    chunks: list[dict[str, Any]],
    excluded: list[dict[str, Any]],
    sections_path: Path,
    chunks_path: Path,
    excluded_path: Path,
) -> None:
    raw_chunks = [c for c in chunks if c["text_variant"] == "raw_clean"]
    lines = [
        "# SCOTUS Section Repair Audit",
        "",
        "Phase 3.5 repair pass over cached CourtListener combined opinion records.",
        "",
        "## Section Counts",
        "",
    ]
    section_rows = []
    for justice in sorted(TARGET_JUSTICES):
        justice_sections = [s for s in sections if s["section_author"] == justice]
        justice_chunks = [c for c in raw_chunks if c["section_author"] == justice]
        section_rows.append(
            [
                justice,
                len(justice_sections),
                len(justice_chunks),
                sum(int(c["token_count"]) for c in justice_chunks),
            ]
        )
    lines.append(markdown_table(["Justice", "Sections", "Kept raw chunks", "Kept raw tokens"], section_rows))

    lines.extend(["", "## Section Posture Distribution", ""])
    posture_rows = []
    posture_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for section in sections:
        posture_counts[section["section_author"]][section["section_posture"]] += 1
    for justice in sorted(TARGET_JUSTICES):
        for posture, count in posture_counts[justice].most_common():
            posture_rows.append([justice, posture, count])
    lines.append(markdown_table(["Justice", "Section Posture", "Count"], posture_rows))

    lines.extend(["", "## Exclusion Flags", ""])
    excluded_flag_counts = Counter()
    for record in excluded:
        for key, value in record.items():
            if key.startswith("is_") or key.startswith("has_"):
                if value is True:
                    excluded_flag_counts[key] += 1
    lines.append(markdown_table(["Flag", "Excluded Records"], excluded_flag_counts.most_common()))

    lines.extend(["", "## Manual Inspection Sample", ""])
    sample_rows = []
    per_justice_seen = Counter()
    for chunk in raw_chunks:
        justice = chunk["section_author"]
        if per_justice_seen[justice] >= 5:
            continue
        per_justice_seen[justice] += 1
        sample_rows.append(
            [
                justice,
                chunk["section_posture"],
                chunk["case_name"],
                chunk["chunk_position_bucket"],
                re.sub(r"\s+", " ", chunk["text"])[:260],
            ]
        )
        if len(sample_rows) >= 20:
            break
    lines.append(markdown_table(["Justice", "Posture", "Case", "Position", "Snippet"], sample_rows))

    lines.extend(
        [
            "",
            "## Outputs",
            "",
            f"- `{display_path(sections_path)}`: {len(sections)} target-authored section records",
            f"- `{display_path(chunks_path)}`: {len(chunks)} kept chunk records",
            f"- `{display_path(excluded_path)}`: {len(excluded)} excluded block/chunk records with flags",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Segment and filter combined SCOTUS opinions.")
    parser.add_argument("--raw", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--opinions", type=Path, default=DEFAULT_OPINIONS)
    parser.add_argument("--sections-output", type=Path, default=DEFAULT_SECTIONS)
    parser.add_argument("--chunks-output", type=Path, default=DEFAULT_CHUNKS)
    parser.add_argument("--excluded-output", type=Path, default=DEFAULT_EXCLUDED)
    parser.add_argument("--report", type=Path, default=DEFAULT_SECTION_REPORT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sections, chunks, excluded = build_repaired_corpus(args.raw, args.opinions)
    write_jsonl(args.sections_output, sections)
    write_jsonl(args.chunks_output, chunks)
    write_jsonl(args.excluded_output, excluded)
    write_section_report(
        args.report,
        sections,
        chunks,
        excluded,
        args.sections_output,
        args.chunks_output,
        args.excluded_output,
    )
    print(f"Wrote {args.sections_output} ({len(sections)} records)")
    print(f"Wrote {args.chunks_output} ({len(chunks)} records)")
    print(f"Wrote {args.excluded_output} ({len(excluded)} records)")
    print(f"Wrote {args.report}")


if __name__ == "__main__":
    main()
