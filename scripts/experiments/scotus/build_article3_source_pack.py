#!/usr/bin/env python3
"""Build an expanded Article III public/private-rights source pack.

The v2.1 target-justice corpus has little direct support for Article III
public/private-rights doctrine. This builder downloads a small, named set of
source opinions from Cornell LII, chunks them, applies the strict Article III
frame rules, and writes a silver-label review queue with provenance.
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
DEFAULT_RAW = SCOTUS_DIR / "raw" / "scotus_article3_source_pages_v1.json"
DEFAULT_LABELS = SCOTUS_DIR / "scotus_article3_source_frame_labels_v1.jsonl"
DEFAULT_QUEUE = SCOTUS_DIR / "scotus_article3_source_frame_review_queue_v1.jsonl"
DEFAULT_REPORT = PROJECT_ROOT / "reports" / "scotus_article3_source_pack_v1.md"

if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_source_frame_labels import (  # noqa: E402
    FRAME_RULES,
    FrameRule,
    assign_frame_splits,
    markdown_table,
    rule_matches,
    stable_split,
    token_window,
)


ARTICLE3_FRAMES = {
    "article3_public_rights",
    "article3_private_rights",
    "article3_article1_tribunal",
    "article3_case_or_controversy",
    "article3_final_judgment_separation",
}


@dataclass(frozen=True)
class SourceCase:
    case_id: str
    case_name: str
    citation: str
    term: int
    expected_signal: str
    url: str


SOURCE_CASES: tuple[SourceCase, ...] = (
    SourceCase(
        case_id="murrays_lessee_1856",
        case_name="Murray's Lessee v. Hoboken Land & Improvement Co.",
        citation="59 U.S. 272",
        term=1856,
        expected_signal="public-rights origin / executive collection",
        url="https://www.law.cornell.edu/supremecourt/text/59/272",
    ),
    SourceCase(
        case_id="crowell_v_benson_1932",
        case_name="Crowell v. Benson",
        citation="285 U.S. 22",
        term=1931,
        expected_signal="agency adjunct / public-private boundary",
        url="https://www.law.cornell.edu/supremecourt/text/285/22",
    ),
    SourceCase(
        case_id="atlas_roofing_1977",
        case_name="Atlas Roofing Co. v. Occupational Safety & Health Review Commission",
        citation="430 U.S. 442",
        term=1976,
        expected_signal="statutory public rights / agency adjudication",
        url="https://www.law.cornell.edu/supremecourt/text/430/442",
    ),
    SourceCase(
        case_id="northern_pipeline_1982",
        case_name="Northern Pipeline Construction Co. v. Marathon Pipe Line Co.",
        citation="458 U.S. 50",
        term=1981,
        expected_signal="private state-law claim / Article III bankruptcy limit",
        url="https://www.law.cornell.edu/supremecourt/text/458/50",
    ),
    SourceCase(
        case_id="thomas_union_carbide_1985",
        case_name="Thomas v. Union Carbide Agricultural Products Co.",
        citation="473 U.S. 568",
        term=1984,
        expected_signal="public regulatory scheme / Article III flexibility",
        url="https://www.law.cornell.edu/supremecourt/text/473/568",
    ),
    SourceCase(
        case_id="cftc_v_schor_1986",
        case_name="Commodity Futures Trading Commission v. Schor",
        citation="478 U.S. 833",
        term=1985,
        expected_signal="agency counterclaim / consent and private-rights balancing",
        url="https://www.law.cornell.edu/supremecourt/text/478/833",
    ),
    SourceCase(
        case_id="granfinanciera_1989",
        case_name="Granfinanciera, S.A. v. Nordberg",
        citation="492 U.S. 33",
        term=1988,
        expected_signal="fraudulent conveyance as private right / public-rights limit",
        url="https://www.law.cornell.edu/supremecourt/text/492/33",
    ),
    SourceCase(
        case_id="stern_v_marshall_2011",
        case_name="Stern v. Marshall",
        citation="564 U.S. 462",
        term=2010,
        expected_signal="state-law counterclaim as private right / non-Article III limit",
        url="https://www.law.cornell.edu/supct/html/10-179.ZO.html",
    ),
    SourceCase(
        case_id="wellness_v_sharif_2015",
        case_name="Wellness International Network, Ltd. v. Sharif",
        citation="575 U.S. 665",
        term=2014,
        expected_signal="Stern claim / private rights and consent",
        url="https://www.law.cornell.edu/supremecourt/text/13-935",
    ),
    SourceCase(
        case_id="oil_states_2018",
        case_name="Oil States Energy Services, LLC v. Greene's Energy Group, LLC",
        citation="584 U.S. 325",
        term=2017,
        expected_signal="patent validity / public franchise public right",
        url="https://www.law.cornell.edu/supremecourt/text/16-712",
    ),
    SourceCase(
        case_id="axon_v_ftc_2023",
        case_name="Axon Enterprise, Inc. v. Federal Trade Commission",
        citation="598 U.S. 175",
        term=2022,
        expected_signal="private rights / Article III review concern",
        url="https://www.law.cornell.edu/supremecourt/text/21-86",
    ),
    SourceCase(
        case_id="sec_v_jarkesy_2024",
        case_name="Securities and Exchange Commission v. Jarkesy",
        citation="603 U.S. ___",
        term=2023,
        expected_signal="common-law fraud / public-rights exception rejected",
        url="https://www.law.cornell.edu/supremecourt/text/22-859",
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
    masked = re.sub(r"§+", "[SECTION]", masked)
    return masked


def mask_frame_cues(text: str) -> str:
    masked = mask_citations(text)
    replacements = [
        (r"\bpublic[- ]rights?\b", "[RIGHTS_FRAME]"),
        (r"\bprivate[- ]rights?\b", "[RIGHTS_FRAME]"),
        (r"\bpublic right\b", "[RIGHTS_FRAME]"),
        (r"\bprivate right\b", "[RIGHTS_FRAME]"),
        (r"\bArticle\s+III\b", "[ARTICLE_COURT]"),
        (r"\bArt\.\s*III\b", "[ARTICLE_COURT]"),
        (r"\bnon[- ]Article\s+III\b", "[ARTICLE_COURT]"),
        (r"\bArticle\s+I\b", "[ARTICLE_COURT]"),
        (r"\bArt\.\s*I\b", "[ARTICLE_COURT]"),
    ]
    for pattern, repl in replacements:
        masked = re.sub(pattern, repl, masked, flags=re.IGNORECASE)
    return masked


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
    session.headers.update({"User-Agent": "scotus-article3-source-pack/0.1 research"})
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


def selected_rules() -> list[FrameRule]:
    return [rule for rule in FRAME_RULES if rule.frame in ARTICLE3_FRAMES]


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
                    "issue_area_label": "Judicial Power",
                    "source_url": page["source_url"],
                    "chunk_index_in_section": idx,
                    "token_count": token_count(chunk),
                    "text": chunk,
                    "text_masked": mask_citations(chunk),
                    "text_cue_masked": mask_frame_cues(chunk),
                    "source_case_id": page["case_id"],
                    "source_citation": page["citation"],
                    "expected_signal": page["expected_signal"],
                }
            )
    return rows


def priority_score(row: dict[str, Any], rule: FrameRule, evidence: list[str]) -> tuple[int, int, int, int, str]:
    expected_bonus = 1 if str(row.get("expected_signal", "")).startswith(rule.frame.replace("article3_", "")) else 0
    return (
        len(evidence),
        expected_bonus,
        -abs(int(row.get("token_count") or 0) - 220),
        -int(row.get("chunk_index_in_section") or 0),
        str(row.get("case_name", "")),
    )


def build_records(
    chunk_rows: list[dict[str, Any]],
    *,
    max_per_frame: int,
    window_chars: int,
) -> list[dict[str, Any]]:
    rules = selected_rules()
    matches_by_chunk: dict[str, list[tuple[FrameRule, list[str]]]] = {}
    for row in tqdm(chunk_rows, desc="scan article3 chunks"):
        text = str(row.get("text", ""))
        for rule in rules:
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
                "issue_family": rule.issue_family,
                "label": 1,
                "label_source": "expanded_article3_source_rule_v1",
                "label_confidence": "silver_high",
                "label_definition": rule.definition,
                "evidence_patterns": evidence,
                "matched_frames": matched_frames,
                "has_public_private_conflict": (
                    "article3_public_rights" in matched_frames and "article3_private_rights" in matched_frames
                ),
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
    for rule in selected_rules():
        counter = counts.get(rule.frame, Counter())
        total = sum(counter.values())
        conflict_count = sum(1 for row in records if row["frame"] == rule.frame and row["has_public_private_conflict"])
        frame_rows.append([rule.frame, total, counter["train"], counter["dev"], counter["test"], conflict_count])

    source_rows = [
        [
            page["case_id"],
            page["case_name"],
            page["citation"],
            page["term"],
            page["token_count"],
            page["source_url"],
        ]
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
                "yes" if record.get("has_public_private_conflict") else "no",
                str(record.get("evidence_window", ""))[:220],
            ]
        )

    lines = [
        "# SCOTUS Article III Source Pack v1",
        "",
        "## Purpose",
        "",
        "This expands the source-grounded frame corpus beyond target-justice chunks for Article III public/private-rights doctrine. It is a silver-label source pack for manual review and leakage diagnostics, not final circuit evidence.",
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
    lines.extend(markdown_table(["Case id", "Case", "Citation", "Term", "Tokens", "URL"], source_rows))
    lines.extend(["", "## Label Counts", ""])
    lines.extend(markdown_table(["Frame", "Total", "Train", "Dev", "Test", "Public/private conflicts"], frame_rows))
    lines.extend(["", "## Case/Frame Coverage", ""])
    lines.extend(markdown_table(["Case id", "Case", "Frame", "Records"], case_frame_rows(records)))
    lines.extend(
        [
            "",
            "## Sample Evidence Windows",
            "",
        ]
    )
    lines.extend(
        markdown_table(
            ["Frame", "Split", "Case", "Citation", "Evidence", "Conflict", "Window"],
            sample_rows,
        )
    )
    lines.extend(
        [
            "",
            "## Use Rules",
            "",
            "1. Treat labels as `silver_high`; manually review before any promotion decision.",
            "2. For public/private-rights contrasts, exclude rows where `has_public_private_conflict` is true.",
            "3. Run probes on both `text` and `text_cue_masked`; promotion requires surviving cue masking and text-baseline checks.",
            "4. Keep this source pack separate from target-justice style labels; it is for legal-frame source grounding, not justice-style classification.",
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
