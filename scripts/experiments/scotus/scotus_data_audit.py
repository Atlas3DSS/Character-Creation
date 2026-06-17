#!/usr/bin/env python3
"""
Phase 0 SCOTUS justice-style corpus audit.

Pulls authored SCOTUS opinions from CourtListener, joins SCDB case metadata
when available, cleans/chunks the text, and writes the audit artifacts
specified in SCOTUS.md.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import os
import re
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup
from huggingface_hub import hf_hub_download
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUT_DIR = PROJECT_ROOT / "data" / "scotus"
DEFAULT_REPORT = PROJECT_ROOT / "reports" / "scotus_data_audit.md"
DEFAULT_TOKEN_FILE = Path("/tmp/skippy_scratch/courtlistener_token")
COURTLISTENER_API = "https://www.courtlistener.com/api/rest/v4"
SCDB_REPO = "ZennyKenny/scotus-docket-dataset-raw"
SCDB_FILENAME = "SCDB_2024_01_caseCentered_Docket.csv"
OPINION_FIELDS = ",".join(
    [
        "id",
        "absolute_url",
        "cluster_id",
        "cluster",
        "author_id",
        "author_str",
        "joined_by_str",
        "type",
        "download_url",
        "plain_text",
        "html",
        "html_lawbox",
        "html_columbia",
        "html_with_citations",
    ]
)
CLUSTER_FIELDS = ",".join(
    [
        "id",
        "absolute_url",
        "date_filed",
        "case_name",
        "case_name_full",
        "scdb_id",
        "scdb_decision_direction",
        "scdb_votes_majority",
        "scdb_votes_minority",
    ]
)

TARGET_JUSTICES = {
    "Scalia": {"first": "Antonin", "last": "Scalia", "person_id": 2852},
    "Ginsburg": {"first": "Ruth", "last": "Ginsburg", "person_id": 1213},
    "Thomas": {"first": "Clarence", "last": "Thomas", "person_id": 3200},
    "Souter": {"first": "David", "last": "Souter", "person_id": 3046},
}

PAIR_SPECS = [
    ("Scalia", "Ginsburg"),
    ("Thomas", "Souter"),
]

OPINION_TYPE_LABELS = {
    "010combined": "combined",
    "015unamimous": "unanimous",
    "020lead": "lead",
    "025plurality": "plurality",
    "030concurrence": "concurrence",
    "035concurrenceinpart": "concurrence_in_part",
    "040dissent": "dissent",
    "050addendum": "addendum",
    "060remittitur": "remittitur",
    "070rehearing": "rehearing",
    "080onthemerits": "on_the_merits",
    "090onmotiontostrike": "on_motion_to_strike",
    "100trialcourt": "trial_court",
}

SCDB_ISSUE_AREA_LABELS = {
    "1": "Criminal Procedure",
    "2": "Civil Rights",
    "3": "First Amendment",
    "4": "Due Process",
    "5": "Privacy",
    "6": "Attorneys",
    "7": "Unions",
    "8": "Economic Activity",
    "9": "Judicial Power",
    "10": "Federalism",
    "11": "Interstate Relations",
    "12": "Federal Taxation",
    "13": "Miscellaneous",
    "14": "Private Action",
}


@dataclass(frozen=True)
class AuditPaths:
    out_dir: Path
    raw_dir: Path
    processed_dir: Path
    manifests_dir: Path
    opinion_inventory: Path
    chunk_inventory: Path
    pair_inventory: Path
    report: Path


def build_paths(out_dir: Path, report_path: Path) -> AuditPaths:
    raw_dir = out_dir / "raw"
    processed_dir = out_dir / "processed"
    manifests_dir = out_dir / "manifests"
    return AuditPaths(
        out_dir=out_dir,
        raw_dir=raw_dir,
        processed_dir=processed_dir,
        manifests_dir=manifests_dir,
        opinion_inventory=out_dir / "scotus_opinion_inventory.jsonl",
        chunk_inventory=out_dir / "scotus_chunk_inventory.jsonl",
        pair_inventory=out_dir / "scotus_pair_overlap_inventory.jsonl",
        report=report_path,
    )


def ensure_dirs(paths: AuditPaths) -> None:
    for directory in [paths.out_dir, paths.raw_dir, paths.processed_dir, paths.manifests_dir, paths.report.parent]:
        directory.mkdir(parents=True, exist_ok=True)


def read_token(token_file: Path | None, token_arg: str | None) -> str:
    if token_arg:
        return token_arg.strip()
    env_token = os.environ.get("COURTLISTENER_API_TOKEN") or os.environ.get("COURTLISTENER_TOKEN")
    if env_token:
        return env_token.strip()
    if token_file and token_file.exists():
        return token_file.read_text(encoding="utf-8").strip()
    raise RuntimeError(
        "CourtListener token not found. Set COURTLISTENER_API_TOKEN or pass --token-file."
    )


def normalize_opinion_type(value: str | None) -> str:
    if not value:
        return "unknown"
    return OPINION_TYPE_LABELS.get(value, value)


def scdb_issue_label(value: str | None) -> str:
    if not value:
        return "unknown"
    return SCDB_ISSUE_AREA_LABELS.get(str(value), str(value))


def request_json(
    session: requests.Session,
    url: str,
    *,
    params: dict[str, Any] | None = None,
    max_retries: int = 5,
    timeout: int = 60,
) -> dict[str, Any]:
    last_error: requests.RequestException | None = None
    for attempt in range(max_retries):
        try:
            response = session.get(url, params=params, timeout=timeout)
            if response.status_code == 429:
                wait_s = min(60, 2 ** attempt + 1)
                time.sleep(wait_s)
                continue
            response.raise_for_status()
            data = response.json()
            if not isinstance(data, dict):
                raise RuntimeError(f"Expected JSON object from {response.url}")
            return data
        except requests.RequestException as exc:
            last_error = exc
            wait_s = min(30, 2 ** attempt)
            time.sleep(wait_s)
    if last_error is not None:
        raise last_error
    raise RuntimeError(f"Failed to fetch JSON from {url}")


def fetch_all_pages(
    session: requests.Session,
    url: str,
    *,
    params: dict[str, Any],
    desc: str,
    page_delay: float,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    next_url: str | None = url
    next_params: dict[str, Any] | None = params
    pbar = tqdm(desc=desc, unit="page")
    try:
        while next_url:
            data = request_json(session, next_url, params=next_params)
            batch = data.get("results", [])
            if not isinstance(batch, list):
                raise RuntimeError(f"Unexpected paginated payload from {next_url}")
            records.extend(batch)
            pbar.update(1)
            if limit is not None and len(records) >= limit:
                records = records[:limit]
                break
            next_url = data.get("next")
            next_params = None
            if page_delay > 0:
                time.sleep(page_delay)
    finally:
        pbar.close()
    return records


def load_scdb_rows(cache_dir: Path) -> dict[str, dict[str, str]]:
    path = Path(
        hf_hub_download(
            repo_id=SCDB_REPO,
            repo_type="dataset",
            filename=SCDB_FILENAME,
            local_dir=str(cache_dir),
        )
    )
    by_case_id: dict[str, dict[str, str]] = {}
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            case_id = (row.get("caseId") or "").strip()
            if not case_id or case_id in by_case_id:
                continue
            by_case_id[case_id] = row
    return by_case_id


def clean_html_to_text(markup: str) -> str:
    if not markup:
        return ""
    parser = "xml" if markup.lstrip().startswith(("<?xml", "<opinion")) else "lxml"
    soup = BeautifulSoup(markup, parser)
    for tag in soup(["script", "style", "table"]):
        tag.decompose()
    for tag in soup.select(".star-pagination, .page-label"):
        tag.decompose()
    text = soup.get_text("\n")
    return clean_plain_text(text)


def clean_plain_text(text: str) -> str:
    text = html.unescape(text or "")
    text = text.replace("\xa0", " ")
    text = re.sub(r"-\n(?=[a-z])", "", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n[ \t]+", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    lines = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            lines.append("")
            continue
        if re.fullmatch(r"\*?\d+\*?", stripped):
            continue
        if stripped in {"SUPREME COURT OF THE UNITED STATES", "Syllabus"}:
            continue
        lines.append(stripped)
    return "\n".join(lines).strip()


def mask_text(text: str) -> str:
    masked = text
    masked = re.sub(r"\b\d+\s+U\.\s*S\.\s+\d+\b", "[CITE]", masked)
    masked = re.sub(r"\b\d+\s+S\.\s*Ct\.\s+\d+\b", "[CITE]", masked)
    masked = re.sub(r"\b\d+\s+L\.\s*Ed\.?\s*(?:2d)?\s+\d+\b", "[CITE]", masked)
    masked = re.sub(r"\b\d{4}\s+U\.S\. LEXIS\s+\d+\b", "[CITE]", masked)
    masked = re.sub(r"\b\d+\s+U\.S\.C\.?\s*§+\s*[\w\-()]+", "[STATUTE]", masked)
    masked = re.sub(r"\b\d+\s+C\.F\.R\.?\s*§+\s*[\w\-().]+", "[STATUTE]", masked)
    masked = re.sub(r"§+", "[SECTION]", masked)
    masked = re.sub(
        r"\b[A-Z][A-Za-z'.&\-]+(?:\s+[A-Z][A-Za-z'.&\-]+){0,5}\s+v\.\s+"
        r"[A-Z][A-Za-z'.&\-]+(?:\s+[A-Z][A-Za-z'.&\-]+){0,5}\b",
        "[CASE]",
        masked,
    )
    masked = re.sub(
        r"\b(?:Justice|JUSTICE|Chief Justice|CHIEF JUSTICE)\s+"
        r"(?:Scalia|Ginsburg|Thomas|Souter|Roberts|Rehnquist|Stevens|O'Connor|Kennedy|Breyer|Alito|Kagan|Sotomayor)\b",
        "[JUSTICE]",
        masked,
    )
    masked = re.sub(
        r"\b(?:SCALIA|GINSBURG|THOMAS|SOUTER|ROBERTS|REHNQUIST|STEVENS|KENNEDY|BREYER|ALITO|KAGAN|SOTOMAYOR)\b",
        "[JUSTICE]",
        masked,
    )
    return masked


def token_count(text: str) -> int:
    return len(re.findall(r"\w+|[^\w\s]", text))


def citation_count(text: str) -> int:
    patterns = [
        r"\b\d+\s+U\.\s*S\.\s+\d+\b",
        r"\b\d+\s+S\.\s*Ct\.\s+\d+\b",
        r"\b\d+\s+L\.\s*Ed\.?\s*(?:2d)?\s+\d+\b",
        r"\b\d{4}\s+U\.S\. LEXIS\s+\d+\b",
    ]
    return sum(len(re.findall(pattern, text)) for pattern in patterns)


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


def extract_term(date_filed: str | None) -> int | None:
    if not date_filed:
        return None
    try:
        filed = datetime.strptime(date_filed[:10], "%Y-%m-%d")
    except ValueError:
        return None
    return filed.year if filed.month >= 10 else filed.year - 1


def decade_bucket(term: int | None) -> str:
    if term is None:
        return "unknown"
    return f"{(term // 10) * 10}s"


def cluster_id_from_url(url: str) -> int | None:
    try:
        return int(Path(urlparse(url).path).name)
    except ValueError:
        return None


def fetch_cluster(session: requests.Session, cluster_uri: str, cache: dict[int, dict[str, Any]]) -> dict[str, Any]:
    cluster_id = cluster_id_from_url(cluster_uri)
    if cluster_id is not None and cluster_id in cache:
        return cache[cluster_id]
    cluster = request_json(session, cluster_uri, params={"fields": CLUSTER_FIELDS})
    if cluster_id is None:
        cluster_id = int(cluster["id"])
    cache[cluster_id] = cluster
    return cluster


def fetch_opinions(
    session: requests.Session,
    *,
    page_delay: float,
    limit_per_justice: int | None,
) -> list[dict[str, Any]]:
    all_records: list[dict[str, Any]] = []
    for justice, spec in TARGET_JUSTICES.items():
        params = {
            "cluster__docket__court": "scotus",
            "author": str(spec["person_id"]),
            "order_by": "cluster__date_filed,id",
            "fields": OPINION_FIELDS,
        }
        records = fetch_all_pages(
            session,
            f"{COURTLISTENER_API}/opinions/",
            params=params,
            desc=f"opinions:{justice}",
            page_delay=page_delay,
            limit=limit_per_justice,
        )
        for record in records:
            record["_target_justice"] = justice
        all_records.extend(records)
    return all_records


def build_inventory_records(
    session: requests.Session,
    opinions: list[dict[str, Any]],
    scdb_rows: dict[str, dict[str, str]],
    *,
    page_delay: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    cluster_cache: dict[int, dict[str, Any]] = {}
    inventory: list[dict[str, Any]] = []
    chunks: list[dict[str, Any]] = []

    for opinion in tqdm(opinions, desc="clusters/chunks", unit="opinion"):
        cluster = fetch_cluster(session, opinion["cluster"], cluster_cache)
        if page_delay > 0:
            time.sleep(page_delay)

        scdb_id = (cluster.get("scdb_id") or "").strip()
        scdb_row = scdb_rows.get(scdb_id, {})
        raw_text = clean_html_to_text(
            opinion.get("html_with_citations")
            or opinion.get("html")
            or opinion.get("html_lawbox")
            or opinion.get("html_columbia")
            or opinion.get("plain_text")
            or ""
        )
        masked = mask_text(raw_text)
        raw_tokens = token_count(raw_text)
        citations = citation_count(raw_text)
        term = int(scdb_row["term"]) if scdb_row.get("term") else extract_term(cluster.get("date_filed"))
        issue_area = str(scdb_row.get("issueArea") or "")
        opinion_type = normalize_opinion_type(opinion.get("type"))
        source_url = "https://www.courtlistener.com" + str(cluster.get("absolute_url", ""))

        record = {
            "opinion_id": opinion.get("id"),
            "cluster_id": cluster.get("id"),
            "scdb_id": scdb_id,
            "case_name": cluster.get("case_name") or cluster.get("case_name_full"),
            "date_filed": cluster.get("date_filed"),
            "term": term,
            "decade": decade_bucket(term),
            "justice": opinion.get("_target_justice"),
            "author_str": opinion.get("author_str"),
            "author_id": opinion.get("author_id"),
            "joined_by_str": opinion.get("joined_by_str"),
            "opinion_type": opinion_type,
            "raw_opinion_type": opinion.get("type"),
            "issue_area": issue_area or None,
            "issue_area_label": scdb_issue_label(issue_area),
            "decision_direction": scdb_row.get("decisionDirection") or cluster.get("scdb_decision_direction"),
            "majority_votes": scdb_row.get("majVotes") or cluster.get("scdb_votes_majority"),
            "minority_votes": scdb_row.get("minVotes") or cluster.get("scdb_votes_minority"),
            "docket": scdb_row.get("docket") or "",
            "us_cite": scdb_row.get("usCite") or "",
            "citation_count": citations,
            "token_count": raw_tokens,
            "chunk_count_raw_clean": 0,
            "chunk_count_masked": 0,
            "source_url": source_url,
            "download_url": opinion.get("download_url"),
            "has_scdb_join": bool(scdb_row),
            "text_source": "html_with_citations" if opinion.get("html_with_citations") else "fallback",
        }

        raw_chunks = paragraph_chunks(raw_text, min_tokens=80, target_min=150, target_max=350)
        masked_chunks = [mask_text(chunk) for chunk in raw_chunks]
        record["chunk_count_raw_clean"] = len(raw_chunks)
        record["chunk_count_masked"] = len(masked_chunks)
        inventory.append(record)

        for idx, (raw_chunk, masked_chunk) in enumerate(zip(raw_chunks, masked_chunks, strict=True)):
            chunk_base = {
                "chunk_id": f"{opinion.get('id')}-{idx:04d}",
                "opinion_id": opinion.get("id"),
                "cluster_id": cluster.get("id"),
                "scdb_id": scdb_id,
                "case_name": record["case_name"],
                "date_filed": record["date_filed"],
                "term": term,
                "decade": record["decade"],
                "justice": record["justice"],
                "opinion_type": opinion_type,
                "issue_area": record["issue_area"],
                "issue_area_label": record["issue_area_label"],
                "decision_direction": record["decision_direction"],
                "source_url": source_url,
            }
            chunks.append(
                {
                    **chunk_base,
                    "text_variant": "raw_clean",
                    "token_count": token_count(raw_chunk),
                    "citation_count": citation_count(raw_chunk),
                    "text": raw_chunk,
                }
            )
            chunks.append(
                {
                    **chunk_base,
                    "text_variant": "masked",
                    "token_count": token_count(masked_chunk),
                    "citation_count": citation_count(masked_chunk),
                    "text": masked_chunk,
                }
            )
    return inventory, chunks


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def summarize_counts(inventory: list[dict[str, Any]], chunks: list[dict[str, Any]]) -> dict[str, Any]:
    raw_chunks = [chunk for chunk in chunks if chunk["text_variant"] == "raw_clean"]
    summary: dict[str, Any] = {
        "opinions_by_justice": Counter(record["justice"] for record in inventory),
        "chunks_by_justice": Counter(chunk["justice"] for chunk in raw_chunks),
        "tokens_by_justice": Counter(),
        "issue_by_justice": defaultdict(Counter),
        "type_by_justice": defaultdict(Counter),
        "decade_by_justice": defaultdict(Counter),
        "scdb_join_by_justice": defaultdict(Counter),
        "citation_density_by_justice": {},
    }
    for record in inventory:
        justice = record["justice"]
        summary["tokens_by_justice"][justice] += int(record.get("token_count") or 0)
        summary["issue_by_justice"][justice][record.get("issue_area_label") or "unknown"] += 1
        summary["type_by_justice"][justice][record.get("opinion_type") or "unknown"] += 1
        summary["decade_by_justice"][justice][record.get("decade") or "unknown"] += 1
        summary["scdb_join_by_justice"][justice]["joined" if record.get("has_scdb_join") else "missing"] += 1
    for justice in TARGET_JUSTICES:
        records = [record for record in inventory if record["justice"] == justice]
        total_tokens = sum(int(record.get("token_count") or 0) for record in records)
        total_cites = sum(int(record.get("citation_count") or 0) for record in records)
        summary["citation_density_by_justice"][justice] = (
            (total_cites / total_tokens) * 1000 if total_tokens else 0.0
        )
    return summary


def same_case_overlaps(inventory: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_cluster: dict[int, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for record in inventory:
        cluster_id = record.get("cluster_id")
        if cluster_id is None:
            continue
        by_cluster[int(cluster_id)][record["justice"]].append(record)

    overlaps: list[dict[str, Any]] = []
    for justice_a, justice_b in PAIR_SPECS:
        for cluster_id, justice_map in by_cluster.items():
            if justice_a not in justice_map or justice_b not in justice_map:
                continue
            for rec_a in justice_map[justice_a]:
                for rec_b in justice_map[justice_b]:
                    overlaps.append(
                        {
                            "pair": f"{justice_a}_vs_{justice_b}",
                            "cluster_id": cluster_id,
                            "scdb_id": rec_a.get("scdb_id") or rec_b.get("scdb_id"),
                            "case_name": rec_a.get("case_name") or rec_b.get("case_name"),
                            "term": rec_a.get("term") or rec_b.get("term"),
                            "issue_area": rec_a.get("issue_area") or rec_b.get("issue_area"),
                            "issue_area_label": rec_a.get("issue_area_label") or rec_b.get("issue_area_label"),
                            "justice_a": justice_a,
                            "justice_b": justice_b,
                            "opinion_id_a": rec_a.get("opinion_id"),
                            "opinion_id_b": rec_b.get("opinion_id"),
                            "opinion_type_a": rec_a.get("opinion_type"),
                            "opinion_type_b": rec_b.get("opinion_type"),
                            "chunk_count_a": rec_a.get("chunk_count_masked"),
                            "chunk_count_b": rec_b.get("chunk_count_masked"),
                            "source_url": rec_a.get("source_url") or rec_b.get("source_url"),
                        }
                    )
    overlaps.sort(key=lambda row: (row["pair"], row.get("term") or 0, row.get("cluster_id") or 0))
    return overlaps


def matched_pair_budget(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    masked = [chunk for chunk in chunks if chunk["text_variant"] == "masked"]
    by_meta: dict[tuple[str, str, str, str], Counter[str]] = defaultdict(Counter)
    for chunk in masked:
        key = (
            chunk.get("issue_area_label") or "unknown",
            chunk.get("opinion_type") or "unknown",
            chunk.get("decade") or "unknown",
            str(chunk.get("decision_direction") or "unknown"),
        )
        by_meta[key][chunk["justice"]] += 1

    rows: list[dict[str, Any]] = []
    for justice_a, justice_b in PAIR_SPECS:
        total = 0
        usable_cells = 0
        for counts in by_meta.values():
            pair_n = min(counts.get(justice_a, 0), counts.get(justice_b, 0))
            if pair_n:
                usable_cells += 1
                total += pair_n
        rows.append(
            {
                "pair": f"{justice_a}_vs_{justice_b}",
                "loose_matched_chunk_pairs": total,
                "matched_metadata_cells": usable_cells,
            }
        )
    return rows


def top_named_cases(chunks: list[dict[str, Any]], n: int = 20) -> list[tuple[str, int]]:
    masked = [chunk for chunk in chunks if chunk["text_variant"] == "raw_clean"]
    counts: Counter[str] = Counter()
    pattern = re.compile(
        r"\b([A-Z][A-Za-z'.&\-]+(?:\s+[A-Z][A-Za-z'.&\-]+){0,5}\s+v\.\s+"
        r"[A-Z][A-Za-z'.&\-]+(?:\s+[A-Z][A-Za-z'.&\-]+){0,5})\b"
    )
    for chunk in masked:
        for match in pattern.findall(chunk["text"]):
            counts[re.sub(r"\s+", " ", match)] += 1
    return counts.most_common(n)


def markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return "\n".join(lines)


def write_report(
    path: Path,
    inventory: list[dict[str, Any]],
    chunks: list[dict[str, Any]],
    overlaps: list[dict[str, Any]],
    pair_budget: list[dict[str, Any]],
    summary: dict[str, Any],
) -> None:
    raw_chunks = [chunk for chunk in chunks if chunk["text_variant"] == "raw_clean"]
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lines: list[str] = [
        "# SCOTUS Data Audit",
        "",
        f"Generated: {now}",
        "",
        "## Scope",
        "",
        "- Source text: CourtListener authored SCOTUS opinion records (`html_with_citations` preferred).",
        "- Metadata: CourtListener clusters joined to SCDB 2024 case-centered docket metadata by `scdb_id`.",
        "- Targets: Scalia, Ginsburg, Thomas, Souter.",
        "- Chunks: paragraph-first, 150-350 token target, 80 token minimum.",
        "- Variants: `raw_clean` and `masked`.",
        "",
        "## Corpus Counts",
        "",
    ]
    corpus_rows = []
    for justice in TARGET_JUSTICES:
        corpus_rows.append(
            [
                justice,
                summary["opinions_by_justice"].get(justice, 0),
                summary["chunks_by_justice"].get(justice, 0),
                summary["tokens_by_justice"].get(justice, 0),
                f"{summary['citation_density_by_justice'].get(justice, 0.0):.2f}",
                summary["scdb_join_by_justice"][justice].get("joined", 0),
            ]
        )
    lines.append(
        markdown_table(
            ["Justice", "Opinions", "Raw chunks", "Tokens", "Cites / 1k tokens", "SCDB joined"],
            corpus_rows,
        )
    )
    lines.extend(["", "## Opinion Type Distribution", ""])
    type_rows = []
    for justice in TARGET_JUSTICES:
        for opinion_type, count in summary["type_by_justice"][justice].most_common():
            type_rows.append([justice, opinion_type, count])
    lines.append(markdown_table(["Justice", "Opinion Type", "Count"], type_rows))

    lines.extend(["", "## Issue Area Distribution", ""])
    issue_rows = []
    for justice in TARGET_JUSTICES:
        for issue, count in summary["issue_by_justice"][justice].most_common():
            issue_rows.append([justice, issue, count])
    lines.append(markdown_table(["Justice", "Issue Area", "Count"], issue_rows))

    lines.extend(["", "## Decade Distribution", ""])
    decade_rows = []
    for justice in TARGET_JUSTICES:
        for decade, count in sorted(summary["decade_by_justice"][justice].items()):
            decade_rows.append([justice, decade, count])
    lines.append(markdown_table(["Justice", "Decade", "Count"], decade_rows))

    lines.extend(["", "## Same-Case Overlaps", ""])
    overlap_counter = Counter(row["pair"] for row in overlaps)
    overlap_rows = []
    for justice_a, justice_b in PAIR_SPECS:
        pair = f"{justice_a}_vs_{justice_b}"
        overlap_rows.append([pair, overlap_counter.get(pair, 0)])
    lines.append(markdown_table(["Pair", "Same-case opinion overlaps"], overlap_rows))

    if overlaps:
        lines.extend(["", "Top overlap examples:", ""])
        example_rows = [
            [
                row["pair"],
                row.get("term"),
                row.get("case_name"),
                row.get("opinion_type_a"),
                row.get("opinion_type_b"),
                row.get("issue_area_label"),
            ]
            for row in overlaps[:20]
        ]
        lines.append(
            markdown_table(
                ["Pair", "Term", "Case", "Type A", "Type B", "Issue Area"],
                example_rows,
            )
        )

    lines.extend(["", "## Loose Matched Chunk Budget", ""])
    lines.append(
        markdown_table(
            ["Pair", "Loose matched chunk pairs", "Metadata cells"],
            [
                [row["pair"], row["loose_matched_chunk_pairs"], row["matched_metadata_cells"]]
                for row in pair_budget
            ],
        )
    )

    lines.extend(["", "## Named-Case Frequency", ""])
    lines.append(markdown_table(["Case Name", "Raw chunk mentions"], top_named_cases(chunks)))

    lines.extend(["", "## Go / No-Go", ""])
    go_rows = []
    for justice_a, justice_b in PAIR_SPECS:
        pair = f"{justice_a}_vs_{justice_b}"
        justice_chunks_ok = (
            summary["chunks_by_justice"].get(justice_a, 0) >= 500
            and summary["chunks_by_justice"].get(justice_b, 0) >= 500
        )
        matched = next(
            (row["loose_matched_chunk_pairs"] for row in pair_budget if row["pair"] == pair),
            0,
        )
        same_case = overlap_counter.get(pair, 0)
        status = "GO" if justice_chunks_ok and matched >= 200 else "NO-GO"
        go_rows.append([pair, status, justice_chunks_ok, matched, same_case])
    lines.append(
        markdown_table(
            ["Pair", "Decision", ">=500 chunks each", "Loose matched chunks", "Same-case overlaps"],
            go_rows,
        )
    )

    lines.extend(
        [
            "",
            "## Artifact Paths",
            "",
            f"- `data/scotus/scotus_opinion_inventory.jsonl`: {len(inventory)} records",
            f"- `data/scotus/scotus_chunk_inventory.jsonl`: {len(chunks)} records ({len(raw_chunks)} raw + {len(raw_chunks)} masked)",
            f"- `data/scotus/scotus_pair_overlap_inventory.jsonl`: {len(overlaps)} records",
        ]
    )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit SCOTUS authored opinion corpus for justice-style steering.")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--token", default=None, help="CourtListener token. Prefer --token-file or env var.")
    parser.add_argument("--token-file", type=Path, default=DEFAULT_TOKEN_FILE)
    parser.add_argument("--cache-dir", type=Path, default=Path("/tmp/skippy_scratch/scotus_cache"))
    parser.add_argument(
        "--page-delay",
        type=float,
        default=1.0,
        help="Delay between CourtListener requests. Keep this courteous; they are a nonprofit.",
    )
    parser.add_argument("--limit-per-justice", type=int, default=None)
    parser.add_argument("--reuse-raw", action="store_true", help="Reuse cached raw CourtListener opinion payload.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = build_paths(args.out_dir, args.report)
    ensure_dirs(paths)
    args.cache_dir.mkdir(parents=True, exist_ok=True)

    token = read_token(args.token_file, args.token)
    session = requests.Session()
    session.headers.update(
        {
            "Authorization": f"Token {token}",
            "User-Agent": "scotus-style-audit/0.1 (research; contact via CourtListener token owner)",
        }
    )

    raw_payload_path = paths.raw_dir / "courtlistener_scotus_target_opinions.json"
    if args.reuse_raw and raw_payload_path.exists():
        opinions = json.loads(raw_payload_path.read_text(encoding="utf-8"))
    else:
        opinions = fetch_opinions(
            session,
            page_delay=args.page_delay,
            limit_per_justice=args.limit_per_justice,
        )
        raw_payload_path.write_text(json.dumps(opinions, ensure_ascii=False, indent=2), encoding="utf-8")

    scdb_rows = load_scdb_rows(args.cache_dir)
    inventory, chunks = build_inventory_records(
        session,
        opinions,
        scdb_rows,
        page_delay=args.page_delay,
    )
    overlaps = same_case_overlaps(inventory)
    pair_budget = matched_pair_budget(chunks)
    summary = summarize_counts(inventory, chunks)

    write_jsonl(paths.opinion_inventory, inventory)
    write_jsonl(paths.chunk_inventory, chunks)
    write_jsonl(paths.pair_inventory, overlaps)
    write_report(paths.report, inventory, chunks, overlaps, pair_budget, summary)

    print(f"Wrote {paths.opinion_inventory} ({len(inventory)} records)")
    print(f"Wrote {paths.chunk_inventory} ({len(chunks)} records)")
    print(f"Wrote {paths.pair_inventory} ({len(overlaps)} records)")
    print(f"Wrote {paths.report}")


if __name__ == "__main__":
    main()
