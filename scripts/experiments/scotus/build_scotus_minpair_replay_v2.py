#!/usr/bin/env python3
"""Build a more diverse Commerce minimal-pair replay bank.

The first replay bank separated Commerce-limits and Commerce-authority answer
states, but it reused a handful of exact answer templates across splits. This
builder keeps the same useful paired-prompt structure while removing exact
assistant-template reuse and adding variant metadata for holdout audits.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "scotus" / "replay" / "scotus_minpair_replay_v2_examples_20260501.jsonl"
DEFAULT_MANIFEST = PROJECT_ROOT / "data" / "scotus" / "replay" / "scotus_minpair_replay_v2_manifest_20260501.json"
DEFAULT_REPORT = PROJECT_ROOT / "reports" / "scotus_minpair_replay_v2_builder_20260501.md"


@dataclass(frozen=True)
class FactPattern:
    fact_id: str
    split: str
    short_name: str
    fact: str
    local_subject: str
    market_subject: str


@dataclass(frozen=True)
class StyleVariant:
    variant_id: str
    prompt_instruction: str
    authority_form: str
    limits_form: str


FACTS: tuple[FactPattern, ...] = (
    FactPattern(
        "commerce_fact_00",
        "train",
        "civil_violence_remedy",
        "Congress creates a federal civil remedy for local violent conduct after finding aggregate effects on employment, health costs, and interstate travel.",
        "local violent conduct",
        "national economic effects attributed to violence",
    ),
    FactPattern(
        "commerce_fact_01",
        "train",
        "school_zone_firearm",
        "Congress makes firearm possession near a school a federal crime without requiring proof that the possession is connected to interstate trade.",
        "gun possession near a school",
        "education-related economic productivity",
    ),
    FactPattern(
        "commerce_fact_02",
        "train",
        "homegrown_fungible_good",
        "Congress restricts homegrown production of a fungible commodity kept for personal use because local supply may affect the national market in the aggregate.",
        "home production for personal use",
        "a national market for a fungible commodity",
    ),
    FactPattern(
        "commerce_fact_03",
        "train",
        "credit_reporting",
        "Congress authorizes statutory damages for false consumer-credit reports sent to lenders operating across state lines.",
        "a private damages dispute over credit information",
        "interstate credit and lending networks",
    ),
    FactPattern(
        "commerce_fact_04",
        "train",
        "local_price_fixing",
        "Congress regulates price-fixing by local suppliers whose commodity is sold through a national market.",
        "intrastate supplier conduct",
        "national commodity pricing",
    ),
    FactPattern(
        "commerce_fact_05",
        "train",
        "school_curriculum",
        "Congress requires local schools to teach a financial-literacy curriculum because education quality affects the national economy.",
        "local school curriculum",
        "future consumer and labor markets",
    ),
    FactPattern(
        "commerce_fact_06",
        "train",
        "shipping_network",
        "Congress imposes civil penalties on operators using a national shipping network even when the charged violation occurred inside one state.",
        "an in-state shipping violation",
        "interstate shipping channels",
    ),
    FactPattern(
        "commerce_fact_07",
        "train",
        "online_seller_remedy",
        "Congress creates a federal remedy against deceptive online sellers whose transactions use interstate payment networks.",
        "consumer contract remedies",
        "interstate payment networks",
    ),
    FactPattern(
        "commerce_fact_08",
        "train",
        "private_home_arson",
        "Congress punishes arson of property used in an activity affecting commerce and applies the law to a private owner-occupied home with no business use.",
        "arson of a private dwelling",
        "property said to affect commerce",
    ),
    FactPattern(
        "commerce_fact_09",
        "train",
        "unpaid_home_care",
        "Congress requires unpaid home care for elderly relatives, citing national healthcare spending and labor-market effects.",
        "family caregiving",
        "healthcare and labor markets",
    ),
    FactPattern(
        "commerce_fact_10",
        "train",
        "local_manufacturer",
        "Congress regulates a small intrastate manufacturer as part of a national price-stabilization scheme for a fungible good.",
        "small local manufacturing",
        "national price stabilization",
    ),
    FactPattern(
        "commerce_fact_11",
        "train",
        "misleading_labels",
        "Congress creates statutory damages for misleading labels on goods sold through nationwide distribution channels.",
        "labeling disputes",
        "nationwide distribution channels",
    ),
    FactPattern(
        "commerce_fact_12",
        "train",
        "youth_sports_dispute",
        "Congress bars a purely local youth-sports dispute from state court after finding that youth athletics affect future economic productivity.",
        "a local youth-sports dispute",
        "future productivity effects",
    ),
    FactPattern(
        "commerce_fact_13",
        "train",
        "warehouse_safety",
        "Congress regulates safety practices for local warehouses that store goods awaiting interstate shipment.",
        "local warehouse safety",
        "goods awaiting interstate shipment",
    ),
    FactPattern(
        "commerce_fact_14",
        "dev",
        "clinic_reporting",
        "Congress imposes a federal reporting duty on local clinics because inaccurate data can distort a national healthcare market.",
        "local clinic recordkeeping",
        "national healthcare data and markets",
    ),
    FactPattern(
        "commerce_fact_15",
        "dev",
        "neighborhood_vandalism",
        "Congress creates a federal cause of action for neighborhood vandalism after compiling findings about aggregate insurance costs.",
        "neighborhood vandalism",
        "aggregate insurance costs",
    ),
    FactPattern(
        "commerce_fact_16",
        "dev",
        "home_repair_platform",
        "Congress regulates home-repair contracts advertised and financed through national online platforms.",
        "home-repair contracts",
        "online advertising and finance platforms",
    ),
    FactPattern(
        "commerce_fact_17",
        "dev",
        "household_tools",
        "Congress criminalizes possession of ordinary household tools near public buildings, citing possible effects on commercial activity.",
        "possession of household tools",
        "possible downstream commercial effects",
    ),
    FactPattern(
        "commerce_fact_18",
        "dev",
        "securities_orders",
        "Congress sets liability rules for local brokers who transmit purchase orders through interstate securities systems.",
        "local brokerage conduct",
        "interstate securities systems",
    ),
    FactPattern(
        "commerce_fact_19",
        "test",
        "budgeting_classes",
        "Congress requires family budgeting classes in every public school because financial literacy affects consumer markets.",
        "public-school instruction",
        "consumer markets",
    ),
    FactPattern(
        "commerce_fact_20",
        "test",
        "agricultural_home_use",
        "Congress regulates cultivation of a fungible agricultural product even when each grower uses the product only at home.",
        "home cultivation",
        "a national agricultural market",
    ),
    FactPattern(
        "commerce_fact_21",
        "test",
        "subscription_renewals",
        "Congress creates federal damages for deceptive subscription renewals processed by interstate credit-card networks.",
        "subscription-renewal disputes",
        "interstate credit-card networks",
    ),
    FactPattern(
        "commerce_fact_22",
        "test",
        "school_bullying_tort",
        "Congress creates a federal tort remedy for local bullying after finding aggregate economic costs from school disruption.",
        "local bullying",
        "school disruption costs",
    ),
    FactPattern(
        "commerce_fact_23",
        "test",
        "storage_depots",
        "Congress regulates local storage depots that feed goods into interstate retail supply chains.",
        "local storage operations",
        "interstate retail supply chains",
    ),
)


STYLE_VARIANTS: tuple[StyleVariant, ...] = (
    StyleVariant(
        "rule_application",
        "Give the answer as three short paragraphs: rule, application, conclusion. Do not name or imitate any Justice.",
        "The governing rule allows Congress to reach {market_subject} when the regulated class is economic and Congress has a rational basis for treating local instances as part of that interstate class. Applied here, {local_subject} is not isolated in constitutional terms because the statute is tied to {market_subject}. The federalism objection has force at the margins, but it does not defeat a rule aimed at an interstate commercial system.",
        "The governing rule leaves Congress without a general police power even when it invokes aggregate economic effects. Applied here, {local_subject} is not itself commerce and is not converted into commerce merely by a causal chain to {market_subject}. The federalism objection is therefore structural: accepting the statute would leave no practical line between national commerce and local governance.",
    ),
    StyleVariant(
        "holding_then_reason",
        "State a holding and then explain the reason in compact judicial prose. Do not use roleplay or any judge's name.",
        "The statute should be sustained. Congress is regulating a class of conduct bound up with {market_subject}, and the local instance can be reached as part of that broader scheme. The state-law backdrop does not supply immunity from federal regulation where the regulated class is commercial and interstate in practical operation.",
        "The statute should be invalidated. Congress is reaching {local_subject}, a subject that remains local unless the statute contains a real commercial hook or regulates economic exchange. Findings about {market_subject} do not supply a limiting principle, so the asserted commerce power would swallow reserved state authority.",
    ),
    StyleVariant(
        "counterargument_first",
        "Begin with the strongest contrary argument, then resolve it. Keep the answer under 150 words and avoid personality cues.",
        "The contrary argument is that the case concerns {local_subject}, traditionally handled by state law. But the better constitutional view is that Congress may regulate local instances when they are part of {market_subject}. The point is not that every local act is national, but that this statute operates inside an interstate economic class whose aggregate regulation would be undermined by local carveouts.",
        "The contrary argument is that {market_subject} gives Congress a national interest. But the better constitutional view is that Congress may not regulate {local_subject} simply by describing downstream economic consequences. Lopez and Morrison require a line between economic activity and local noneconomic matters; otherwise the enumeration of federal powers loses force.",
    ),
    StyleVariant(
        "two_step",
        "Use a two-step analysis. First identify the regulated activity; second decide whether the commerce rationale is limiting.",
        "Step one: the regulated activity is best understood as participation in or facilitation of {market_subject}. Step two: the commerce rationale is limiting because it depends on an identifiable interstate economic class, not an abstract productivity theory. On that understanding, Congress may reach the local instance without converting the commerce power into a general police power.",
        "Step one: the regulated activity is best understood as {local_subject}. Step two: the commerce rationale is not limiting because it depends on a chain from local conduct to {market_subject}. That chain would justify federal control over nearly any local choice, so the statute exceeds the commerce power.",
    ),
    StyleVariant(
        "short_opinion",
        "Write a short opinion-style analysis with no headings. Do not mention any Justice, party, or case name unless doctrine requires it.",
        "Congress may regulate this conduct because the statute is directed at {market_subject}, not merely at a freestanding local concern. The fact that one application occurs inside a single state is not dispositive when the regulated class is economic and interstate in the aggregate. The Constitution's federal structure remains relevant, but it does not bar this targeted commercial regulation.",
        "Congress may not regulate this conduct on the theory offered. The immediate object is {local_subject}, and the asserted link to {market_subject} is too attenuated to mark a judicially enforceable boundary. Commerce doctrine permits regulation of commerce; it does not authorize federal supervision of every local activity with economic consequences.",
    ),
    StyleVariant(
        "doctrinal_synthesis",
        "Synthesize the doctrine in one paragraph, then apply it in one paragraph. Keep wording distinct from stock case summaries.",
        "Commerce doctrine permits regulation of channels, instrumentalities, and economic classes of activity with substantial interstate effects. That doctrine supports the statute because {market_subject} supplies an economic class rather than a mere but-for consequence. The local facts matter, but they are regulated as instances of a broader commercial system.",
        "Commerce doctrine permits national regulation of commerce, but it also preserves a distinction between economic activity and local noneconomic governance. That distinction defeats the statute because {local_subject} is being reached through an attenuated connection to {market_subject}. The asserted rationale would not stop at this case.",
    ),
)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(item) for item in row) + " |")
    return "\n".join(lines)


def user_prompt(fact: FactPattern, variant: StyleVariant) -> str:
    return (
        f"Fact pattern: {fact.fact}\n\n"
        "Analyze Congress's power under the Commerce Clause and federalism limits. "
        f"{variant.prompt_instruction}"
    )


def assistant_text(fact: FactPattern, variant: StyleVariant, label_name: str) -> str:
    template = variant.limits_form if label_name == "commerce_limits" else variant.authority_form
    return template.format(local_subject=fact.local_subject, market_subject=fact.market_subject)


def build_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for fact in FACTS:
        for variant in STYLE_VARIANTS:
            pair_id = f"{fact.fact_id}|{variant.variant_id}"
            for label, label_name in ((0, "commerce_authority"), (1, "commerce_limits")):
                example_id = f"{pair_id}|{label_name}"
                text = assistant_text(fact, variant, label_name)
                rows.append(
                    {
                        "example_id": example_id,
                        "chunk_id": example_id,
                        "pair_id": pair_id,
                        "fact_id": fact.fact_id,
                        "fact_short_name": fact.short_name,
                        "variant_id": variant.variant_id,
                        "split": fact.split,
                        "label": label,
                        "justice": label_name,
                        "positive_justice": "commerce_limits",
                        "frame_task": "commerce_limits_vs_authority",
                        "frame_label": label_name,
                        "issue_area_label": "Economic Activity",
                        "opinion_type": "minimal_pair_replay_v2",
                        "section_posture": "assistant_replay",
                        "prompt": user_prompt(fact, variant),
                        "assistant_text": text,
                        "text": text,
                    }
                )
    rows.sort(key=lambda row: str(row["example_id"]))
    return rows


def build_report(rows: list[dict[str, Any]], *, output: Path, manifest: Path) -> str:
    split_counts = Counter((str(row["split"]), str(row["frame_label"])) for row in rows)
    variant_counts = Counter(str(row["variant_id"]) for row in rows)
    exact_text_counts = Counter(str(row["assistant_text"]) for row in rows)
    prompt_pair_counts = Counter(str(row["pair_id"]) for row in rows)
    duplicate_texts = sum(1 for count in exact_text_counts.values() if count > 1)
    unpaired = [pair for pair, count in prompt_pair_counts.items() if count != 2]
    fact_variants: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        fact_variants[str(row["fact_id"])].add(str(row["variant_id"]))

    lines = [
        "# SCOTUS Minimal-Pair Replay v2 Builder",
        "",
        "## Purpose",
        "",
        "Build a more diverse Commerce Clause replay bank after the first minimal-pair bank was demoted for exact assistant-template reuse.",
        "",
        "## Artifacts",
        "",
        markdown_table(
            ["Artifact", "Path"],
            [
                ["Examples", output],
                ["Manifest", manifest],
            ],
        ),
        "",
        "## Counts",
        "",
        f"- Rows: `{len(rows)}`",
        f"- Fact patterns: `{len({row['fact_id'] for row in rows})}`",
        f"- Style variants per fact: `{len(STYLE_VARIANTS)}`",
        f"- Exact duplicate assistant texts: `{duplicate_texts}`",
        f"- Unpaired prompt rows: `{len(unpaired)}`",
        "",
        markdown_table(
            ["Split", "Label", "Rows"],
            [[split, label, count] for (split, label), count in sorted(split_counts.items())],
        ),
        "",
        "## Variant Counts",
        "",
        markdown_table(["Variant", "Rows"], [[variant, count] for variant, count in sorted(variant_counts.items())]),
        "",
        "## Read",
        "",
        "- Each fact/style prompt has one Commerce-authority and one Commerce-limits assistant answer, so prompt-only label leakage should remain near chance.",
        "- Exact assistant completions are unique across rows.",
        "- Style variants are mirrored across labels to reduce format-label leakage.",
        "- This is still synthetic replay data, not steering evidence. It is a cleaner candidate source for the next activation probe and template-holdout audit.",
    ]
    if unpaired:
        lines.extend(["", "## Pairing Warnings", "", ", ".join(unpaired[:20])])
    bad_fact_variants = [fact for fact, variants in fact_variants.items() if len(variants) != len(STYLE_VARIANTS)]
    if bad_fact_variants:
        lines.extend(["", "## Variant Warnings", "", ", ".join(sorted(bad_fact_variants))])
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = build_rows()
    write_jsonl(args.output, rows)
    manifest = {
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "output": str(args.output),
        "rows": len(rows),
        "fact_patterns": len(FACTS),
        "style_variants": len(STYLE_VARIANTS),
        "labels": ["commerce_authority", "commerce_limits"],
        "purpose": "diverse Commerce minimal-pair replay bank without exact assistant-template reuse",
    }
    write_json(args.manifest, manifest)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(build_report(rows, output=args.output, manifest=args.manifest), encoding="utf-8")
    print(f"Wrote {args.output}")
    print(f"Wrote {args.report}")


if __name__ == "__main__":
    main()
