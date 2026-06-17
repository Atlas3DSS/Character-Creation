#!/usr/bin/env python3
"""Build an Economic Activity / Commerce Clause source-frame pack.

The issue-family triage nominated Economic Activity as the next source-pack
branch. This builder downloads named Commerce Clause and related federalism /
preemption opinions from Cornell LII, applies doctrine-oriented silver rules,
and emits cue-masked text for leakage diagnostics.
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
DEFAULT_RAW = SCOTUS_DIR / "raw" / "scotus_economic_source_pages_v1.json"
DEFAULT_LABELS = SCOTUS_DIR / "scotus_economic_source_frame_labels_v1.jsonl"
DEFAULT_QUEUE = SCOTUS_DIR / "scotus_economic_source_frame_review_queue_v1.jsonl"
DEFAULT_REPORT = PROJECT_ROOT / "reports" / "scotus_economic_source_pack_v1.md"

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
class EconomicRule:
    frame: str
    definition: str
    required_any: tuple[tuple[str, ...], ...]
    evidence_patterns: tuple[str, ...]
    exclude_any: tuple[str, ...] = ()


ECONOMIC_RULES: tuple[EconomicRule, ...] = (
    EconomicRule(
        frame="economic_commerce_broad_aggregation",
        definition=(
            "Broad Commerce Clause authority, including aggregation, substantial-effects, "
            "national-market, channels/instrumentalities, or comprehensive regulatory-scheme reasoning."
        ),
        required_any=(
            (
                r"\bcommerce clause\b",
                r"\binterstate commerce\b",
                r"\bcommerce among the several states\b",
                r"\bcommerce power\b",
                r"\bsubstantial(?:ly)? affect(?:s|ing)? (?:interstate )?commerce\b",
                r"\baffecting commerce\b",
                r"\baggregate(?:d|ion)?\b",
                r"\bnational market\b",
                r"\bcurrent of commerce\b",
                r"\bstream of commerce\b",
                r"\bchannels of commerce\b",
                r"\binstrumentalities of commerce\b",
                r"\bcomprehensive regulatory scheme\b",
                r"\bclass of activities\b",
                r"\bclose and substantial relation\b",
            ),
            (
                r"\bcongress\b",
                r"\bregulat(?:e|es|ion|ory)\b",
                r"\bpower\b",
                r"\bauthority\b",
            ),
        ),
        evidence_patterns=(
            r"\bcommerce clause\b",
            r"\binterstate commerce\b",
            r"\bcommerce among the several states\b",
            r"\bcommerce power\b",
            r"\bsubstantial(?:ly)? affect(?:s|ing)? (?:interstate )?commerce\b",
            r"\baffecting commerce\b",
            r"\baggregate(?:d|ion)?\b",
            r"\bnational market\b",
            r"\bcurrent of commerce\b",
            r"\bstream of commerce\b",
            r"\bchannels of commerce\b",
            r"\binstrumentalities of commerce\b",
            r"\bcomprehensive regulatory scheme\b",
            r"\bclass of activities\b",
            r"\bclose and substantial relation\b",
            r"\bgibbons\b",
            r"\bchampion\b",
            r"\bshreveport\b",
            r"\bstafford\b",
            r"\bdarby\b",
            r"\bperez\b",
            r"\bhodel\b",
            r"\bwickard\b",
            r"\braich\b",
            r"\bheart of atlanta\b",
            r"\bkatzenbach\b",
            r"\bjones\s*&\s*laughlin\b",
        ),
        exclude_any=(
            r"\bnon[- ]economic\b",
            r"\bnoneconomic\b",
            r"\battenuated\b",
            r"\bjurisdictional element\b",
            r"\binactivity\b",
            r"\bactivity/inactivity\b",
            r"\bouter limits\b",
            r"\bdirect and indirect\b",
            r"\bindirect effect\b",
            r"\bindirectly affect\b",
            r"\bproduction\b.{0,80}\bnot commerce\b",
            r"\bmanufactur(?:e|ing)\b.{0,80}\bnot commerce\b",
            r"\bmining\b.{0,80}\bnot commerce\b",
            r"\breserved to the states\b",
        ),
    ),
    EconomicRule(
        frame="economic_commerce_limits",
        definition=(
            "Commerce Clause limits, including non-economic activity, attenuated causal chains, "
            "missing jurisdictional elements, Lopez/Morrison limits, or NFIB activity/inactivity reasoning."
        ),
        required_any=(
            (
                r"\bcommerce clause\b",
                r"\binterstate commerce\b",
                r"\bsubstantial(?:ly)? affect(?:s|ing)? (?:interstate )?commerce\b",
                r"\bcongress(?:'s)? commerce power\b",
                r"\bcommerce power\b",
                r"\bcommerce among the several states\b",
            ),
            (
                r"\bnon[- ]economic\b",
                r"\bnoneconomic\b",
                r"\battenuated\b",
                r"\bjurisdictional element\b",
                r"\btraditional state\b",
                r"\bpolice power\b",
                r"\binactivity\b",
                r"\bactivity/inactivity\b",
                r"\bouter limits\b",
                r"\bdirect and indirect\b",
                r"\bindirect effect\b",
                r"\bindirectly affect\b",
                r"\bproduction\b.{0,80}\bnot commerce\b",
                r"\bmanufactur(?:e|ing)\b.{0,80}\bnot commerce\b",
                r"\bmining\b.{0,80}\bnot commerce\b",
                r"\blocal activit(?:y|ies)\b",
                r"\breserved to the states\b",
                r"\be\.?\s*c\.?\s*knight\b",
                r"\bhammer\b",
                r"\bschechter\b",
                r"\bcarter coal\b",
                r"\blopez\b",
                r"\bmorrison\b",
                r"\bnfib\b",
            ),
        ),
        evidence_patterns=(
            r"\bnon[- ]economic\b",
            r"\bnoneconomic\b",
            r"\battenuated\b",
            r"\bjurisdictional element\b",
            r"\btraditional state\b",
            r"\bpolice power\b",
            r"\binactivity\b",
            r"\bactivity/inactivity\b",
            r"\bouter limits\b",
            r"\bdirect and indirect\b",
            r"\bindirect effect\b",
            r"\bindirectly affect\b",
            r"\bproduction\b.{0,80}\bnot commerce\b",
            r"\bmanufactur(?:e|ing)\b.{0,80}\bnot commerce\b",
            r"\bmining\b.{0,80}\bnot commerce\b",
            r"\blocal activit(?:y|ies)\b",
            r"\breserved to the states\b",
            r"\be\.?\s*c\.?\s*knight\b",
            r"\bhammer\b",
            r"\bschechter\b",
            r"\bcarter coal\b",
            r"\blopez\b",
            r"\bmorrison\b",
            r"\bnfib\b",
        ),
    ),
    EconomicRule(
        frame="economic_federalism_state_regulation",
        definition=(
            "Federalism and state-regulatory-power reasoning in economic cases, including traditional "
            "state concern, police powers, state sovereignty, or structural limits on national power."
        ),
        required_any=(
            (
                r"\bfederalism\b",
                r"\bstate sovereignty\b",
                r"\btraditional state\b",
                r"\bpolice power\b",
                r"\bstates'? powers\b",
                r"\breserved to the states\b",
                r"\bstate autonomy\b",
                r"\bcommandeer(?:ing)?\b",
                r"\banti[- ]commandeering\b",
                r"\bdual sovereignty\b",
                r"\bpolitical accountability\b",
            ),
            (
                r"\bcommerce\b",
                r"\bcongress\b",
                r"\bfederal\b",
                r"\bnational\b",
                r"\bregulat(?:e|ion|ory)\b",
            ),
        ),
        evidence_patterns=(
            r"\bfederalism\b",
            r"\bstate sovereignty\b",
            r"\btraditional state\b",
            r"\bpolice power\b",
            r"\bstates'? powers\b",
            r"\breserved to the states\b",
            r"\bstate autonomy\b",
            r"\bcommandeer(?:ing)?\b",
            r"\banti[- ]commandeering\b",
            r"\bdual sovereignty\b",
            r"\bpolitical accountability\b",
            r"\bnew york v\. united states\b",
            r"\bprintz\b",
            r"\bmurphy\b",
            r"\bnational league of cities\b",
            r"\bgarcia\b",
        ),
    ),
    EconomicRule(
        frame="economic_preemption_statutory",
        definition=(
            "Statutory/preemption frame in economic disputes, including Supremacy Clause, FAA, "
            "arbitration preemption, obstacle preemption, or clear-statement/statutory-text reasoning."
        ),
        required_any=(
            (
                r"\bpreempt(?:ion|ed|s)?\b",
                r"\bsupremacy clause\b",
                r"\bfederal arbitration act\b",
                r"\bfaa\b",
                r"\barbitration\b",
                r"\bstatutory interpretation\b",
                r"\bclear statement\b",
            ),
            (
                r"\bstate law\b",
                r"\bstate rule\b",
                r"\bstatute\b",
                r"\bcongress\b",
                r"\bfederal law\b",
            ),
        ),
        evidence_patterns=(
            r"\bpreempt(?:ion|ed|s)?\b",
            r"\bsupremacy clause\b",
            r"\bfederal arbitration act\b",
            r"\bfaa\b",
            r"\barbitration\b",
            r"\bclass proceedings?\b",
            r"\bstatutory interpretation\b",
            r"\bclear statement\b",
            r"\bsouthland\b",
            r"\bcipollone\b",
            r"\bcasarotto\b",
            r"\bpreston\b",
            r"\bepic systems\b",
            r"\bwyeth\b",
            r"\bconcepcion\b",
        ),
    ),
)


SOURCE_CASES: tuple[SourceCase, ...] = (
    SourceCase(
        "gibbons_1824",
        "Gibbons v. Ogden",
        "22 U.S. 1",
        1823,
        "economic_commerce_broad_aggregation",
        "navigation / commerce among the several states",
        "https://www.law.cornell.edu/supremecourt/text/22/1",
    ),
    SourceCase(
        "champion_1903",
        "Champion v. Ames",
        "188 U.S. 321",
        1902,
        "economic_commerce_broad_aggregation",
        "lottery traffic / channels of interstate commerce",
        "https://www.law.cornell.edu/supremecourt/text/188/321",
    ),
    SourceCase(
        "shreveport_1914",
        "Houston, East & West Texas Railway Co. v. United States",
        "234 U.S. 342",
        1913,
        "economic_commerce_broad_aggregation",
        "intrastate rates with close relation to interstate commerce",
        "https://www.law.cornell.edu/supremecourt/text/234/342",
    ),
    SourceCase(
        "stafford_1922",
        "Stafford v. Wallace",
        "258 U.S. 495",
        1921,
        "economic_commerce_broad_aggregation",
        "stockyards / current of commerce",
        "https://www.law.cornell.edu/supremecourt/text/258/495",
    ),
    SourceCase(
        "jones_laughlin_1937",
        "NLRB v. Jones & Laughlin Steel Corp.",
        "301 U.S. 1",
        1936,
        "economic_commerce_broad_aggregation",
        "substantial relation to interstate commerce",
        "https://www.law.cornell.edu/supremecourt/text/301/1",
    ),
    SourceCase(
        "darby_1941",
        "United States v. Darby",
        "312 U.S. 100",
        1940,
        "economic_commerce_broad_aggregation",
        "shipment of goods / commerce power after Hammer",
        "https://www.law.cornell.edu/supremecourt/text/312/100",
    ),
    SourceCase(
        "wickard_1942",
        "Wickard v. Filburn",
        "317 U.S. 111",
        1942,
        "economic_commerce_broad_aggregation",
        "aggregation / homegrown wheat market",
        "https://www.law.cornell.edu/supremecourt/text/317/111",
    ),
    SourceCase(
        "heart_atlanta_1964",
        "Heart of Atlanta Motel, Inc. v. United States",
        "379 U.S. 241",
        1964,
        "economic_commerce_broad_aggregation",
        "public accommodations / channels of commerce",
        "https://www.law.cornell.edu/supremecourt/text/379/241",
    ),
    SourceCase(
        "katzenbach_mcclung_1964",
        "Katzenbach v. McClung",
        "379 U.S. 294",
        1964,
        "economic_commerce_broad_aggregation",
        "restaurant discrimination / interstate food market",
        "https://www.law.cornell.edu/supremecourt/text/379/294",
    ),
    SourceCase(
        "perez_1971",
        "Perez v. United States",
        "402 U.S. 146",
        1970,
        "economic_commerce_broad_aggregation",
        "class of activities / loan sharking and national markets",
        "https://www.law.cornell.edu/supremecourt/text/402/146",
    ),
    SourceCase(
        "hodel_1981",
        "Hodel v. Virginia Surface Mining & Reclamation Assn., Inc.",
        "452 U.S. 264",
        1980,
        "economic_commerce_broad_aggregation",
        "surface mining / rational basis substantial effects",
        "https://www.law.cornell.edu/supremecourt/text/452/264",
    ),
    SourceCase(
        "raich_2005",
        "Gonzales v. Raich",
        "545 U.S. 1",
        2004,
        "economic_commerce_broad_aggregation",
        "controlled substances / comprehensive market regulation",
        "https://www.law.cornell.edu/supremecourt/text/03-1454",
    ),
    SourceCase(
        "ec_knight_1895",
        "United States v. E. C. Knight Co.",
        "156 U.S. 1",
        1894,
        "economic_commerce_limits",
        "manufacturing not commerce / local production limit",
        "https://www.law.cornell.edu/supremecourt/text/156/1",
    ),
    SourceCase(
        "hammer_1918",
        "Hammer v. Dagenhart",
        "247 U.S. 251",
        1917,
        "economic_commerce_limits",
        "child labor production / reserved state police power",
        "https://www.law.cornell.edu/supremecourt/text/247/251",
    ),
    SourceCase(
        "schechter_1935",
        "A. L. A. Schechter Poultry Corp. v. United States",
        "295 U.S. 495",
        1934,
        "economic_commerce_limits",
        "direct versus indirect effects / local poultry sales",
        "https://www.law.cornell.edu/supremecourt/text/295/495",
    ),
    SourceCase(
        "carter_coal_1936",
        "Carter v. Carter Coal Co.",
        "298 U.S. 238",
        1935,
        "economic_commerce_limits",
        "mining production not commerce / direct-indirect limit",
        "https://www.law.cornell.edu/supremecourt/text/298/238",
    ),
    SourceCase(
        "lopez_1995",
        "United States v. Lopez",
        "514 U.S. 549",
        1994,
        "economic_commerce_limits",
        "non-economic school-zone gun possession / federalism limit",
        "https://www.law.cornell.edu/supremecourt/text/514/549",
    ),
    SourceCase(
        "morrison_2000",
        "United States v. Morrison",
        "529 U.S. 598",
        1999,
        "economic_commerce_limits",
        "gender-motivated violence / non-economic activity limit",
        "https://www.law.cornell.edu/supremecourt/text/529/598",
    ),
    SourceCase(
        "nfib_2012",
        "National Federation of Independent Business v. Sebelius",
        "567 U.S. 519",
        2011,
        "economic_commerce_limits",
        "activity/inactivity limit on Commerce Clause",
        "https://www.law.cornell.edu/supremecourt/text/11-393",
    ),
    SourceCase(
        "national_league_1976",
        "National League of Cities v. Usery",
        "426 U.S. 833",
        1975,
        "economic_federalism_state_regulation",
        "state sovereignty / traditional governmental functions",
        "https://www.law.cornell.edu/supremecourt/text/426/833",
    ),
    SourceCase(
        "garcia_1985",
        "Garcia v. San Antonio Metropolitan Transit Authority",
        "469 U.S. 528",
        1984,
        "economic_federalism_state_regulation",
        "federalism limits through political process",
        "https://www.law.cornell.edu/supremecourt/text/469/528",
    ),
    SourceCase(
        "new_york_1992",
        "New York v. United States",
        "505 U.S. 144",
        1991,
        "economic_federalism_state_regulation",
        "anti-commandeering / state accountability in waste regulation",
        "https://www.law.cornell.edu/supremecourt/text/505/144",
    ),
    SourceCase(
        "printz_1997",
        "Printz v. United States",
        "521 U.S. 898",
        1996,
        "economic_federalism_state_regulation",
        "anti-commandeering / state executive officers",
        "https://www.law.cornell.edu/supremecourt/text/521/898",
    ),
    SourceCase(
        "murphy_2018",
        "Murphy v. National Collegiate Athletic Assn.",
        "584 U.S. 453",
        2017,
        "economic_federalism_state_regulation",
        "anti-commandeering / sports gambling market regulation",
        "https://www.law.cornell.edu/supremecourt/text/16-476",
    ),
    SourceCase(
        "southland_1984",
        "Southland Corp. v. Keating",
        "465 U.S. 1",
        1983,
        "economic_preemption_statutory",
        "FAA preemption / state franchise claims",
        "https://www.law.cornell.edu/supremecourt/text/465/1",
    ),
    SourceCase(
        "cipollone_1992",
        "Cipollone v. Liggett Group, Inc.",
        "505 U.S. 504",
        1991,
        "economic_preemption_statutory",
        "tobacco labeling preemption / statutory text",
        "https://www.law.cornell.edu/supremecourt/text/505/504",
    ),
    SourceCase(
        "doctors_assoc_1996",
        "Doctor's Associates, Inc. v. Casarotto",
        "517 U.S. 681",
        1995,
        "economic_preemption_statutory",
        "FAA preemption / arbitration notice state law",
        "https://www.law.cornell.edu/supremecourt/text/517/681",
    ),
    SourceCase(
        "preston_2008",
        "Preston v. Ferrer",
        "552 U.S. 346",
        2007,
        "economic_preemption_statutory",
        "FAA preemption / agency forum exhaustion",
        "https://www.law.cornell.edu/supremecourt/text/06-1463",
    ),
    SourceCase(
        "wyeth_2009",
        "Wyeth v. Levine",
        "555 U.S. 555",
        2008,
        "economic_preemption_statutory",
        "drug-labeling preemption / statutory and regulatory scheme",
        "https://www.law.cornell.edu/supremecourt/text/06-1249",
    ),
    SourceCase(
        "concepcion_2011",
        "AT&T Mobility LLC v. Concepcion",
        "563 U.S. 333",
        2010,
        "economic_preemption_statutory",
        "FAA preemption / class arbitration",
        "https://www.law.cornell.edu/supremecourt/text/09-893",
    ),
    SourceCase(
        "epic_2018",
        "Epic Systems Corp. v. Lewis",
        "584 U.S. 497",
        2017,
        "economic_preemption_statutory",
        "FAA and NLRA / statutory harmonization",
        "https://www.law.cornell.edu/supremecourt/text/16-285",
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
            sentence_buffer: list[str] = []
            sentence_tokens = 0
            for sentence in split_sentences(paragraph):
                s_tokens = token_count(sentence)
                if sentence_buffer and sentence_tokens + s_tokens > target_max:
                    chunk = " ".join(sentence_buffer).strip()
                    if token_count(chunk) >= min_tokens:
                        chunks.append(chunk)
                    sentence_buffer = []
                    sentence_tokens = 0
                sentence_buffer.append(sentence)
                sentence_tokens += s_tokens
            if sentence_buffer:
                chunk = " ".join(sentence_buffer).strip()
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
        (r"\bCommerce Clause\b", "[COMMERCE]"),
        (r"\binterstate commerce\b", "[COMMERCE]"),
        (r"\bcommerce among the several states\b", "[COMMERCE]"),
        (r"\bcommerce power\b", "[COMMERCE]"),
        (r"\bsubstantial(?:ly)? affect(?:s|ing)? (?:interstate )?commerce\b", "[FRAME]"),
        (r"\baffecting commerce\b", "[FRAME]"),
        (r"\baggregate(?:d|ion)?\b", "[FRAME]"),
        (r"\bnational market\b", "[FRAME]"),
        (r"\bcurrent of commerce\b", "[FRAME]"),
        (r"\bstream of commerce\b", "[FRAME]"),
        (r"\bchannels of commerce\b", "[FRAME]"),
        (r"\binstrumentalities of commerce\b", "[FRAME]"),
        (r"\bcomprehensive regulatory scheme\b", "[FRAME]"),
        (r"\bclose and substantial relation\b", "[FRAME]"),
        (r"\bnon[- ]economic\b", "[LIMIT]"),
        (r"\bnoneconomic\b", "[LIMIT]"),
        (r"\battenuated\b", "[LIMIT]"),
        (r"\bjurisdictional element\b", "[LIMIT]"),
        (r"\binactivity\b", "[LIMIT]"),
        (r"\bactivity/inactivity\b", "[LIMIT]"),
        (r"\bdirect and indirect\b", "[LIMIT]"),
        (r"\bindirect effect\b", "[LIMIT]"),
        (r"\bindirectly affect\b", "[LIMIT]"),
        (r"\bproduction\b.{0,80}\bnot commerce\b", "[LIMIT]"),
        (r"\bmanufactur(?:e|ing)\b.{0,80}\bnot commerce\b", "[LIMIT]"),
        (r"\bmining\b.{0,80}\bnot commerce\b", "[LIMIT]"),
        (r"\blocal activit(?:y|ies)\b", "[LIMIT]"),
        (r"\bfederalism\b", "[FEDERALISM]"),
        (r"\bstate sovereignty\b", "[FEDERALISM]"),
        (r"\btraditional state\b", "[FEDERALISM]"),
        (r"\bpolice power\b", "[FEDERALISM]"),
        (r"\breserved to the states\b", "[FEDERALISM]"),
        (r"\bcommandeer(?:ing)?\b", "[FEDERALISM]"),
        (r"\banti[- ]commandeering\b", "[FEDERALISM]"),
        (r"\bdual sovereignty\b", "[FEDERALISM]"),
        (r"\bpolitical accountability\b", "[FEDERALISM]"),
        (r"\bpreempt(?:ion|ed|s)?\b", "[PREEMPTION]"),
        (r"\bsupremacy clause\b", "[PREEMPTION]"),
        (r"\bfederal arbitration act\b", "[STATUTE]"),
        (r"\bFAA\b", "[STATUTE]"),
        (r"\bGibbons\b", "[CASE]"),
        (r"\bChampion\b", "[CASE]"),
        (r"\bShreveport\b", "[CASE]"),
        (r"\bStafford\b", "[CASE]"),
        (r"\bE\.?\s*C\.?\s*Knight\b", "[CASE]"),
        (r"\bHammer\b", "[CASE]"),
        (r"\bSchechter\b", "[CASE]"),
        (r"\bCarter Coal\b", "[CASE]"),
        (r"\bDarby\b", "[CASE]"),
        (r"\bPerez\b", "[CASE]"),
        (r"\bHodel\b", "[CASE]"),
        (r"\bWickard\b", "[CASE]"),
        (r"\bRaich\b", "[CASE]"),
        (r"\bLopez\b", "[CASE]"),
        (r"\bMorrison\b", "[CASE]"),
        (r"\bNFIB\b", "[CASE]"),
        (r"\bGibbons\b", "[CASE]"),
        (r"\bHeart of Atlanta\b", "[CASE]"),
        (r"\bKatzenbach\b", "[CASE]"),
        (r"\bJones\s*&\s*Laughlin\b", "[CASE]"),
        (r"\bNational League of Cities\b", "[CASE]"),
        (r"\bGarcia\b", "[CASE]"),
        (r"\bNew York v\. United States\b", "[CASE]"),
        (r"\bPrintz\b", "[CASE]"),
        (r"\bMurphy\b", "[CASE]"),
        (r"\bSouthland\b", "[CASE]"),
        (r"\bCipollone\b", "[CASE]"),
        (r"\bCasarotto\b", "[CASE]"),
        (r"\bPreston\b", "[CASE]"),
        (r"\bWyeth\b", "[CASE]"),
        (r"\bConcepcion\b", "[CASE]"),
        (r"\bEpic Systems\b", "[CASE]"),
    ]
    for pattern, repl in replacements:
        masked = re.sub(pattern, repl, masked, flags=re.IGNORECASE)
    return masked


def match_regexes(text: str, patterns: tuple[str, ...]) -> list[str]:
    return [pattern for pattern in patterns if re.search(pattern, text, flags=re.IGNORECASE)]


def rule_matches(text: str, rule: EconomicRule) -> tuple[bool, list[str], list[str]]:
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
    session.headers.update({"User-Agent": "scotus-economic-source-pack/0.1 research"})
    pages: list[dict[str, Any]] = []
    for source in tqdm(SOURCE_CASES, desc="download economic source opinions"):
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
                    "scdb_id": "",
                    "case_name": page["case_name"],
                    "date_filed": "",
                    "term": page["term"],
                    "decade": f"{(int(page['term']) // 10) * 10}s",
                    "justice": "source_opinion",
                    "section_author": "",
                    "section_posture": "source_opinion",
                    "issue_area_label": "Economic Activity",
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


def priority_score(row: dict[str, Any], rule: EconomicRule, evidence: list[str]) -> tuple[int, int, int, int, str]:
    expected_bonus = 3 if row.get("expected_frame") == rule.frame else 0
    return (
        expected_bonus,
        len(evidence),
        -abs(int(row.get("token_count") or 0) - 220),
        -int(row.get("chunk_index_in_section") or 0),
        str(row.get("case_name", "")),
    )


def build_records(chunk_rows: list[dict[str, Any]], *, max_per_frame: int, window_chars: int) -> list[dict[str, Any]]:
    matches_by_chunk: dict[str, list[tuple[EconomicRule, list[str]]]] = {}
    for row in tqdm(chunk_rows, desc="scan economic chunks"):
        text = str(row.get("text", ""))
        for rule in ECONOMIC_RULES:
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
                "issue_family": "Economic Activity",
                "label": 1,
                "label_source": "economic_source_rule_v1",
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
    for rule in ECONOMIC_RULES:
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
        "# SCOTUS Economic Activity Source Pack v1",
        "",
        "## Purpose",
        "",
        "This builds source-grounded Economic Activity labels after proposition-null triage nominated Commerce Clause / federal economic-power doctrine as the next candidate branch. It is a silver-label pack for cue-masked diagnostics and review, not final circuit evidence.",
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
            "4. Prefer the narrow `economic_commerce_broad_aggregation` versus `economic_commerce_limits` contrast before broader multi-frame causal work.",
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
