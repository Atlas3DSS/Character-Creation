#!/usr/bin/env python3
"""Adjudicate the Article III holding-direction review queue.

This supports evaluator repair for the Article III public-rights/private-rights
branch. The default output is an internal Codex triage pass, not an independent
blind human review. Its purpose is to calibrate automatic proposition and
polarity scorers before any further actuator run is promoted.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SCOTUS_DIR = PROJECT_ROOT / "data" / "scotus"
DEFAULT_BLIND = SCOTUS_DIR / "scotus_article3_holding_review_blind_20260501.jsonl"
DEFAULT_KEY = SCOTUS_DIR / "scotus_article3_holding_review_key_20260501.jsonl"
DEFAULT_OUTPUT = SCOTUS_DIR / "scotus_article3_holding_review_adjudicated_20260501.jsonl"
DEFAULT_REPORT = PROJECT_ROOT / "reports" / "scotus_article3_holding_review_adjudication_20260501.md"
DEFAULT_SUMMARY = PROJECT_ROOT / "reports" / "scotus_article3_holding_review_adjudication_20260501.json"

PRIVATE = "article3_objection_succeeds_private_rights"
PUBLIC = "article3_objection_fails_public_rights_permissible"
MIXED = "mixed_or_distinction_only"
UNCLEAR = "unclear_or_incoherent"

COHERENT = "legally_coherent"
PARTLY = "partly_coherent"
CONFUSED = "legally_confused"
TRUNCATED = "nonresponsive_or_truncated"

DIRECT = "direct_reasoning"
NOT_ASSESSABLE = "not_assessable"

AUTO_TO_HOLDING = {
    "private_rights_objection_succeeds": PRIVATE,
    "public_rights_adjudication_permissible": PUBLIC,
    "mixed_or_unclear": MIXED,
}

# Internal first-pass labels from the answer-only queue. These are useful for
# triage and scorer debugging, but should not be treated as independent gold.
CODEX_TRIAGE: dict[str, tuple[str, str, str, str, str]] = {
    "article3_holding::0bd83af4069dd401": (
        PRIVATE,
        COHERENT,
        DIRECT,
        "high",
        "Explicitly says Article III does not permit final agency adjudication of a state-law tort claim between private parties.",
    ),
    "article3_holding::14ff0ae9d61f3a82": (
        PUBLIC,
        COHERENT,
        DIRECT,
        "high",
        "Explicitly says the importer's Article III objection fails under the public-rights doctrine.",
    ),
    "article3_holding::1991c0f5a7670192": (
        MIXED,
        TRUNCATED,
        NOT_ASSESSABLE,
        "high",
        "The answer states the public/private distinction but truncates before applying it to the workplace-safety penalty.",
    ),
    "article3_holding::2d4f513b93adbf60": (
        MIXED,
        TRUNCATED,
        NOT_ASSESSABLE,
        "high",
        "The answer frames the Stern issue but truncates before reaching a holding.",
    ),
    "article3_holding::39609ca1313aeca1": (
        MIXED,
        TRUNCATED,
        NOT_ASSESSABLE,
        "medium",
        "The answer identifies the private-contract Article III problem but truncates before applying the rule.",
    ),
    "article3_holding::3e544bc0a11a4c95": (
        PUBLIC,
        COHERENT,
        DIRECT,
        "high",
        "The answer opens with 'The Article III challenge fails' and characterizes the proceeding as public rights.",
    ),
    "article3_holding::40abc7a38a2dc112": (
        PUBLIC,
        COHERENT,
        DIRECT,
        "high",
        "The answer says the benefits claimant's Article III first-instance argument is without merit.",
    ),
    "article3_holding::4479a9f7e479ebc2": (
        PUBLIC,
        COHERENT,
        DIRECT,
        "high",
        "The answer says the benefits claimant's Article III first-instance argument is without merit.",
    ),
    "article3_holding::4b83bf74a5a99e15": (
        PRIVATE,
        COHERENT,
        DIRECT,
        "high",
        "Explicitly says Article III does not permit the agency assignment of a private tort dispute.",
    ),
    "article3_holding::58b0f6348f8c0ed0": (
        PRIVATE,
        PARTLY,
        DIRECT,
        "medium",
        "The answer does not state 'therefore,' but treats a private contract claim as the kind of private right Congress may not assign for final non-Article-III judgment.",
    ),
    "article3_holding::7430d77e6d426471": (
        PUBLIC,
        PARTLY,
        DIRECT,
        "medium",
        "The answer pivots from the patent owner's private-right argument to the public-rights exception, but truncates before the final application.",
    ),
    "article3_holding::77caa0887466f09c": (
        MIXED,
        TRUNCATED,
        NOT_ASSESSABLE,
        "medium",
        "The answer presents the patent owner private-right theory and truncates before resolving it.",
    ),
    "article3_holding::839214710ac081d6": (
        PUBLIC,
        COHERENT,
        DIRECT,
        "high",
        "Explicitly says the customs Article III objection fails because tariff classification is a public right.",
    ),
    "article3_holding::8c38715e082c14cb": (
        PUBLIC,
        PARTLY,
        DIRECT,
        "medium",
        "The answer turns to the public-rights exception for patent review but truncates before the final sentence.",
    ),
    "article3_holding::9a2756b80915ab57": (
        PUBLIC,
        PARTLY,
        DIRECT,
        "medium",
        "The answer applies the public-rights framework to a federal workplace-safety penalty but truncates during the application.",
    ),
    "article3_holding::b798679260f545f5": (
        PRIVATE,
        COHERENT,
        DIRECT,
        "high",
        "The answer opens by locating the constitutional infirmity in final non-Article-III adjudication of a private-rights dispute.",
    ),
    "article3_holding::b8a787fb0370364d": (
        PUBLIC,
        PARTLY,
        DIRECT,
        "medium",
        "The answer points toward public-rights treatment of a federal securities enforcement penalty but truncates before a clean final holding.",
    ),
    "article3_holding::c7f2fffc6edd6f5b": (
        PUBLIC,
        COHERENT,
        DIRECT,
        "high",
        "Explicitly says the customs Article III objection fails under the public-rights exception.",
    ),
    "article3_holding::d9078f1c5a392082": (
        PUBLIC,
        COHERENT,
        DIRECT,
        "high",
        "The answer says the benefits claimant's Article III first-instance argument is without merit.",
    ),
    "article3_holding::e5b0878cbccbb83d": (
        PUBLIC,
        CONFUSED,
        DIRECT,
        "low",
        "The answer explicitly says the objection fails, but its Stern/private-rights reasoning is unstable and truncated.",
    ),
    "article3_holding::f0a764e998092a50": (
        UNCLEAR,
        CONFUSED,
        DIRECT,
        "high",
        "The answer says the objection fails but then states bankruptcy judges may not enter final judgment on this type of private-rights counterclaim.",
    ),
    "article3_holding::f19e12ac8a0a356a": (
        MIXED,
        TRUNCATED,
        NOT_ASSESSABLE,
        "medium",
        "The answer defines public and private rights but truncates before applying the distinction to the penalty.",
    ),
    "article3_holding::f5fad2e265dd45d6": (
        MIXED,
        TRUNCATED,
        NOT_ASSESSABLE,
        "medium",
        "The answer states the securities public/private inquiry but truncates before resolving the Article III issue.",
    ),
    "article3_holding::fdd11f22ed8bd62e": (
        PRIVATE,
        COHERENT,
        DIRECT,
        "high",
        "Explicitly says Article III does not permit final agency adjudication of the private tort dispute.",
    ),
}


CODEX_TRIAGE_LONG_20260502: dict[str, tuple[str, str, str, str, str]] = {
    "article3_holding::0bd83af4069dd401": (
        PRIVATE,
        COHERENT,
        DIRECT,
        "high",
        "Final conclusion says Article III prohibits final agency adjudication of the private tort claim.",
    ),
    "article3_holding::14ff0ae9d61f3a82": (
        PUBLIC,
        COHERENT,
        DIRECT,
        "high",
        "Explicitly says the importer’s Article III objection fails because tariff classification is a public-rights matter.",
    ),
    "article3_holding::1991c0f5a7670192": (
        PUBLIC,
        COHERENT,
        DIRECT,
        "high",
        "Concludes the workplace-safety penalty is a public sanction that Congress may assign to agency adjudication subject to review.",
    ),
    "article3_holding::2d4f513b93adbf60": (
        PRIVATE,
        COHERENT,
        DIRECT,
        "high",
        "Concludes the Stern-style state-law counterclaim requires Article III final adjudication.",
    ),
    "article3_holding::39609ca1313aeca1": (
        PRIVATE,
        COHERENT,
        DIRECT,
        "high",
        "Concludes final board judgment on a private contract claim violates Article III.",
    ),
    "article3_holding::3e544bc0a11a4c95": (
        PUBLIC,
        COHERENT,
        DIRECT,
        "high",
        "Concludes securities civil-penalty adjudication is a public-rights matter that does not violate Article III.",
    ),
    "article3_holding::40abc7a38a2dc112": (
        PUBLIC,
        COHERENT,
        DIRECT,
        "high",
        "Concludes federal benefits eligibility is a public-rights matter and the Article III challenge is denied.",
    ),
    "article3_holding::4479a9f7e479ebc2": (
        PUBLIC,
        COHERENT,
        DIRECT,
        "high",
        "Concludes agency first-instance federal-benefits adjudication with judicial review is constitutionally permissible.",
    ),
    "article3_holding::4b83bf74a5a99e15": (
        PRIVATE,
        COHERENT,
        DIRECT,
        "high",
        "Concludes Article III does not permit final agency adjudication of the private tort claim.",
    ),
    "article3_holding::58b0f6348f8c0ed0": (
        PRIVATE,
        COHERENT,
        DIRECT,
        "high",
        "Concludes final non-Article-III judgment on a private contract claim likely violates Article III.",
    ),
    "article3_holding::7430d77e6d426471": (
        PUBLIC,
        COHERENT,
        DIRECT,
        "high",
        "Concludes patent review is constitutional under public-rights reasoning with Article III appellate review.",
    ),
    "article3_holding::77caa0887466f09c": (
        PRIVATE,
        COHERENT,
        DIRECT,
        "high",
        "Concludes patent validity is a private right whose final determination must be made by an Article III court.",
    ),
    "article3_holding::839214710ac081d6": (
        PUBLIC,
        COHERENT,
        DIRECT,
        "high",
        "Concludes the customs Article III objection is without merit under public-rights doctrine.",
    ),
    "article3_holding::8c38715e082c14cb": (
        PUBLIC,
        COHERENT,
        DIRECT,
        "high",
        "Concludes patent validity review through a non-Article-III tribunal with appellate review does not violate Article III.",
    ),
    "article3_holding::9a2756b80915ab57": (
        PUBLIC,
        COHERENT,
        DIRECT,
        "high",
        "Concludes agency adjudication of a statutory workplace-safety penalty is permissible with Article III review.",
    ),
    "article3_holding::b798679260f545f5": (
        PRIVATE,
        COHERENT,
        DIRECT,
        "high",
        "Concludes final non-Article-III board judgment on a private contract dispute is likely unconstitutional.",
    ),
    "article3_holding::b8a787fb0370364d": (
        PUBLIC,
        COHERENT,
        DIRECT,
        "high",
        "Concludes securities enforcement penalties are public rights and agency adjudication is permissible subject to review.",
    ),
    "article3_holding::c7f2fffc6edd6f5b": (
        PUBLIC,
        COHERENT,
        DIRECT,
        "high",
        "Concludes tariff classification is a public right and the importer’s Article III objection fails.",
    ),
    "article3_holding::d9078f1c5a392082": (
        PUBLIC,
        COHERENT,
        DIRECT,
        "high",
        "Concludes benefits eligibility is a public-rights matter and first-instance agency adjudication is permissible.",
    ),
    "article3_holding::e5b0878cbccbb83d": (
        PRIVATE,
        PARTLY,
        DIRECT,
        "medium",
        "Opening sentence says the objection fails, but the final conclusion says the objection is well-taken and Article III adjudication is required.",
    ),
    "article3_holding::f0a764e998092a50": (
        PRIVATE,
        PARTLY,
        DIRECT,
        "medium",
        "Opening sentence says the objection fails, but the operative analysis and remedy require Article III adjudication.",
    ),
    "article3_holding::f19e12ac8a0a356a": (
        MIXED,
        PARTLY,
        DIRECT,
        "medium",
        "The answer frames a conditional private-rights problem but does not clearly decide whether this penalty is actually private or public.",
    ),
    "article3_holding::f5fad2e265dd45d6": (
        MIXED,
        PARTLY,
        DIRECT,
        "medium",
        "The answer says the Article III argument is strong if the agency decision is final, but it does not clearly resolve the prompt.",
    ),
    "article3_holding::fdd11f22ed8bd62e": (
        PRIVATE,
        COHERENT,
        DIRECT,
        "high",
        "Concludes Article III does not permit final agency judgment on the private tort claim.",
    ),
}


CODEX_TRIAGE_AMBIGUOUS_20260502: dict[str, tuple[str, str, str, str, str]] = {
    "article3_holding::256c91b9930032a3": (
        PRIVATE,
        COHERENT,
        DIRECT,
        "high",
        "Concludes the industry-fund contribution scheme violates Article III because it finally adjudicates a private indemnity/contribution dispute.",
    ),
    "article3_holding::2bcf5efb41fe2d8a": (
        PUBLIC,
        COHERENT,
        DIRECT,
        "high",
        "Concludes the land-use compensation award is a public-rights/statutory scheme and the Article III challenge fails.",
    ),
    "article3_holding::32f69e23e05c2e08": (
        PRIVATE,
        COHERENT,
        DIRECT,
        "high",
        "Concludes the land-use compensation claim is tort-like private liability requiring Article III final adjudication.",
    ),
    "article3_holding::337dedc77345b068": (
        PUBLIC,
        COHERENT,
        DIRECT,
        "high",
        "Concludes the land-use compensation claim is a statutory public right tied to a federal regulatory program.",
    ),
    "article3_holding::350b7a6afd19044d": (
        PUBLIC,
        COHERENT,
        DIRECT,
        "high",
        "Concludes workplace penalty plus compensation is a public-rights enforcement remedy and the agency order is permissible.",
    ),
    "article3_holding::3e594f69a05115fc": (
        PUBLIC,
        COHERENT,
        DIRECT,
        "high",
        "Concludes customs penalty and forfeiture are public-rights enforcement matters and the objection fails.",
    ),
    "article3_holding::43eba72cbbbd9172": (
        PRIVATE,
        COHERENT,
        DIRECT,
        "high",
        "Concludes the workplace compensation component is a private right requiring Article III final adjudication.",
    ),
    "article3_holding::474afd3b8cb43cd0": (
        PUBLIC,
        COHERENT,
        DIRECT,
        "high",
        "Concludes benefits fraud recoupment and penalty are public-rights administration subject to Article III review.",
    ),
    "article3_holding::551f89641f65d25b": (
        PUBLIC,
        COHERENT,
        DIRECT,
        "high",
        "Concludes patent cancellation through inter partes review is public-rights adjudication with Article III review.",
    ),
    "article3_holding::593c72a333eecc62": (
        PRIVATE,
        COHERENT,
        DIRECT,
        "high",
        "Concludes the bankruptcy state-law contract counterclaim is a private right requiring Article III final judgment.",
    ),
    "article3_holding::8ac280cc508c322b": (
        PRIVATE,
        COHERENT,
        DIRECT,
        "high",
        "Concludes PTAB cancellation is final adjudication of private patent property rights in violation of Article III.",
    ),
    "article3_holding::9f310aa02238f23b": (
        PUBLIC,
        COHERENT,
        DIRECT,
        "high",
        "Concludes the industry fund contribution scheme is public-rights adjudication within a regulatory program.",
    ),
    "article3_holding::a8aa9dc4848cfc0b": (
        PUBLIC,
        COHERENT,
        DIRECT,
        "high",
        "Concludes benefits fraud recoupment is a public-rights benefits-program matter and the challenge is denied.",
    ),
    "article3_holding::aaf0310d06880303": (
        PRIVATE,
        COHERENT,
        DIRECT,
        "high",
        "Concludes the bankruptcy counterclaim is a private right and non-Article-III final judgment violates Article III absent consent or proper review.",
    ),
    "article3_holding::ab394b156c2e1c6d": (
        PUBLIC,
        COHERENT,
        DIRECT,
        "high",
        "Concludes customs classification, penalties, and forfeiture are public-rights matters and the objection is overruled.",
    ),
    "article3_holding::b9f9888c0a1205cd": (
        PRIVATE,
        COHERENT,
        DIRECT,
        "high",
        "Concludes the securities penalty/disgorgement action resembles a traditional suit at law and requires Article III adjudication.",
    ),
    "article3_holding::be1ba7e51b679bfb": (
        PUBLIC,
        COHERENT,
        DIRECT,
        "high",
        "Concludes the industry fund contribution scheme involves public rights and Article III permits agency adjudication with legal review.",
    ),
    "article3_holding::c4b8ba9e60824aef": (
        PUBLIC,
        COHERENT,
        DIRECT,
        "high",
        "Concludes securities enforcement penalties and disgorgement are public-rights regulatory enforcement and agency adjudication is permissible.",
    ),
    "article3_holding::c50e133e766865f6": (
        PRIVATE,
        COHERENT,
        DIRECT,
        "high",
        "Concludes the bankruptcy counterclaim is a private right and the creditor’s Article III objection is well-founded.",
    ),
    "article3_holding::d04ace8de1649678": (
        PUBLIC,
        COHERENT,
        DIRECT,
        "high",
        "Concludes final agency adjudication of the securities enforcement action is permitted as public-rights adjudication.",
    ),
    "article3_holding::d4348f61d891c6ad": (
        PUBLIC,
        COHERENT,
        DIRECT,
        "high",
        "Concludes customs penalty and forfeiture adjudication falls within the public-rights exception.",
    ),
    "article3_holding::f69e10f83f0057da": (
        PUBLIC,
        COHERENT,
        DIRECT,
        "high",
        "Concludes benefits fraud recoupment arises from a public benefits scheme and the agency adjudication is constitutional.",
    ),
    "article3_holding::fa1c141d7e316a8e": (
        PUBLIC,
        COHERENT,
        DIRECT,
        "high",
        "Concludes workplace penalty plus employee compensation is a statutory public-rights enforcement matter.",
    ),
    "article3_holding::fff819918ef24002": (
        PUBLIC,
        COHERENT,
        DIRECT,
        "high",
        "Concludes patent review is public-rights adjudication and Article III review preserves the constitutional balance.",
    ),
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON") from exc
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    def cell(value: Any) -> str:
        return str(value).replace("|", "\\|").replace("\n", " ")

    lines = [
        "| " + " | ".join(cell(header) for header in headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(cell(value) for value in row) + " |")
    return "\n".join(lines)


def display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def confidence_value(label: str) -> float:
    return {"low": 0.0, "medium": 0.5, "high": 1.0}.get(label, 0.0)


def checked_key(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    keyed: dict[str, dict[str, Any]] = {}
    for row in rows:
        review_id = str(row["review_id"])
        if review_id in keyed:
            raise RuntimeError(f"Duplicate review_id in key file: {review_id}")
        keyed[review_id] = row
    return keyed


def labels_for_mode(mode: str) -> dict[str, tuple[str, str, str, str, str]] | None:
    if mode == "codex-triage":
        return CODEX_TRIAGE
    if mode == "codex-triage-long-20260502":
        return CODEX_TRIAGE_LONG_20260502
    if mode == "codex-triage-ambiguous-20260502":
        return CODEX_TRIAGE_AMBIGUOUS_20260502
    return None


def annotation_for(row: dict[str, Any], *, mode: str) -> tuple[str, str, str, str, str]:
    labels = labels_for_mode(mode)
    if labels is not None:
        review_id = str(row["review_id"])
        if review_id not in labels:
            raise RuntimeError(f"Missing Codex triage annotation for {review_id}")
        return labels[review_id]
    if mode == "existing":
        return (
            str(row.get("holding_direction_label", "")),
            str(row.get("reasoning_quality_label", "")),
            str(row.get("mask_label", "")),
            str(row.get("review_confidence", "")),
            str(row.get("review_notes", "")),
        )
    raise ValueError(f"Unsupported mode: {mode}")


def adjudicate(
    *,
    blind_rows: list[dict[str, Any]],
    key_rows: dict[str, dict[str, Any]],
    mode: str,
) -> list[dict[str, Any]]:
    blind_ids = {str(row["review_id"]) for row in blind_rows}
    key_ids = set(key_rows)
    if blind_ids != key_ids:
        raise RuntimeError(
            "Blind/key review_id mismatch: "
            f"missing_from_key={sorted(blind_ids - key_ids)}, missing_from_blind={sorted(key_ids - blind_ids)}"
        )
    labels = labels_for_mode(mode)
    if labels is not None:
        missing = sorted(blind_ids - set(labels))
        extra = sorted(set(labels) - blind_ids)
        if missing or extra:
            raise RuntimeError(f"Codex triage map mismatch: missing={missing}, extra={extra}")

    reviewed_at = now_iso()
    out: list[dict[str, Any]] = []
    for row in sorted(blind_rows, key=lambda item: str(item["review_id"])):
        review_id = str(row["review_id"])
        holding_label, quality_label, mask_label, confidence, notes = annotation_for(row, mode=mode)
        key = key_rows[review_id]
        auto_label = str(key.get("automatic_polarity_label", ""))
        automatic_holding_label = AUTO_TO_HOLDING.get(auto_label, "")
        reviewed = dict(row)
        reviewed.update(
            {
                "holding_direction_label": holding_label,
                "reasoning_quality_label": quality_label,
                "mask_label": mask_label,
                "review_confidence": confidence,
                "review_notes": notes,
                "reviewer": "internal_codex_triage" if mode.startswith("codex-triage") else "existing_queue_labels",
                "reviewed_at": reviewed_at,
                "adjudication_mode": mode,
                "condition": key.get("condition", ""),
                "inserted_thought_redacted_in_original_queue": bool(key.get("inserted_thought")),
                "automatic_polarity_label": key.get("automatic_polarity_label"),
                "automatic_holding_label": automatic_holding_label,
                "automatic_private_score": key.get("automatic_private_score"),
                "automatic_public_score": key.get("automatic_public_score"),
                "automatic_net_private_minus_public": key.get("automatic_net_private_minus_public"),
                "proposition_target_hits": key.get("proposition_target_hits"),
                "proposition_contrast_hits": key.get("proposition_contrast_hits"),
                "proposition_delta_net_vs_neutral": key.get("proposition_delta_net_vs_neutral"),
            }
        )
        out.append(reviewed)
    return out


def label_counts(rows: list[dict[str, Any]], field: str) -> list[list[Any]]:
    counts = Counter(str(row.get(field, "")) for row in rows)
    return [[label or "(blank)", count] for label, count in sorted(counts.items())]


def condition_rows(rows: list[dict[str, Any]]) -> list[list[Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("condition", ""))].append(row)
    table_rows: list[list[Any]] = []
    for condition, group in sorted(grouped.items()):
        counts = Counter(str(row.get("holding_direction_label", "")) for row in group)
        table_rows.append(
            [
                condition or "(blank)",
                len(group),
                counts[PRIVATE],
                counts[PUBLIC],
                counts[MIXED],
                counts[UNCLEAR],
                f"{mean(confidence_value(str(row.get('review_confidence', ''))) for row in group):.2f}",
            ]
        )
    return table_rows


def prompt_condition_rows(rows: list[dict[str, Any]]) -> list[list[Any]]:
    return [
        [
            row["prompt_key"],
            row["condition"],
            row["holding_direction_label"],
            row["automatic_holding_label"],
            row["reasoning_quality_label"],
            row["review_confidence"],
        ]
        for row in sorted(rows, key=lambda item: (str(item["prompt_key"]), str(item["condition"])))
    ]


def agreement_rows(rows: list[dict[str, Any]]) -> tuple[list[list[Any]], dict[str, Any]]:
    eligible = [row for row in rows if row.get("holding_direction_label") in {PRIVATE, PUBLIC, MIXED}]
    exact = [
        row
        for row in eligible
        if row.get("automatic_holding_label") and row.get("holding_direction_label") == row.get("automatic_holding_label")
    ]
    confusion = Counter(
        (str(row.get("automatic_holding_label", "")) or "(blank)", str(row.get("holding_direction_label", "")) or "(blank)")
        for row in rows
    )
    table_rows = [[auto, reviewed, count] for (auto, reviewed), count in sorted(confusion.items())]
    summary = {
        "eligible_rows": len(eligible),
        "exact_agreement_rows": len(exact),
        "exact_agreement_rate": round(len(exact) / len(eligible), 4) if eligible else None,
    }
    return table_rows, summary


def proposition_rows(rows: list[dict[str, Any]]) -> list[list[Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("holding_direction_label", ""))].append(row)
    table_rows: list[list[Any]] = []
    for label, group in sorted(grouped.items()):
        deltas = [float(row["proposition_delta_net_vs_neutral"]) for row in group if row.get("proposition_delta_net_vs_neutral") is not None]
        nets = [float(row["automatic_net_private_minus_public"]) for row in group if row.get("automatic_net_private_minus_public") is not None]
        table_rows.append(
            [
                label or "(blank)",
                len(group),
                f"{mean(deltas):.3f}" if deltas else "n/a",
                f"{mean(nets):.3f}" if nets else "n/a",
            ]
        )
    return table_rows


def build_summary(rows: list[dict[str, Any]], *, mode: str, blind: Path, key: Path, output: Path) -> dict[str, Any]:
    agreement_table, agreement_summary = agreement_rows(rows)
    condition_table = condition_rows(rows)
    return {
        "mode": mode,
        "reviewer": "internal_codex_triage" if mode.startswith("codex-triage") else "existing_queue_labels",
        "inputs": {
            "blind": display_path(blind),
            "key": display_path(key),
        },
        "output": display_path(output),
        "n_rows": len(rows),
        "holding_label_counts": dict(Counter(str(row.get("holding_direction_label", "")) for row in rows)),
        "reasoning_quality_counts": dict(Counter(str(row.get("reasoning_quality_label", "")) for row in rows)),
        "mask_label_counts": dict(Counter(str(row.get("mask_label", "")) for row in rows)),
        "condition_table": condition_table,
        "automatic_agreement": agreement_summary,
        "automatic_confusion_table": agreement_table,
    }


def write_report(
    path: Path,
    *,
    rows: list[dict[str, Any]],
    summary: dict[str, Any],
    blind: Path,
    key: Path,
    output: Path,
    summary_path: Path,
) -> None:
    agreement_table, agreement_summary = agreement_rows(rows)
    mode = str(summary.get("mode", ""))
    is_long = mode.endswith("long-20260502")
    is_ambiguous = mode.endswith("ambiguous-20260502")
    if is_ambiguous:
        gate_lines = [
            "- The ambiguous prompt bank is a better calibration surface than the original fact-pattern-determined prompts.",
            "- The private-rights inserted-thought condition shows strong final-holding movement relative to neutral and public-rights conditions, but this remains text-prefill calibration rather than activation actuator evidence.",
            "- The automatic polarity scorer is directionally useful here, but reviewed holding labels are still required for promotion because prior runs showed regex confusion on mixed legal answers.",
            "- No actuator candidate is promoted by this adjudication. The next actuator run still needs the model to generate the target reasoning trajectory itself under activation intervention and beat random/source/text/prompt controls.",
        ]
    elif is_long:
        gate_lines = [
            "- The long-answer queue fixes the mechanical truncation flaw from the 96-token queue, but it does not make automatic scoring reliable.",
            "- The automatic polarity scorer remains a triage aid only: it reached low exact agreement with reviewed holding labels and still confuses discussion of a frame with adoption of that frame.",
            "- The private-rights inserted-thought condition shows only weak final-holding movement relative to neutral; this is evaluator calibration, not actuator evidence.",
            "- No actuator candidate is promoted by this adjudication. The next actuator run still needs reasoning-trace movement and final-answer movement against random/source/text/prompt controls.",
        ]
    else:
        gate_lines = [
            "- The automatic polarity scorer is useful for triage, but the confusion table shows it is not a clean final-holding gate.",
            "- This triage pass weakens the earlier automatic-only reading that private-rights scratchpads cleanly moved final answers. The private-rights condition produced many mixed/truncated answers rather than robust private-rights holdings.",
            "- The generated answers are too often truncated for this queue to be a final adjudication surface. The next evaluator repair should regenerate the counterfactual-thought answers with a longer answer budget or a stricter complete-answer stop condition.",
            "- No actuator candidate is promoted by this adjudication. The next actuator run still needs reasoning-trace movement and final-answer movement against random/source/text/prompt controls.",
        ]
    lines = [
        "# SCOTUS Article III Holding Review Adjudication",
        "",
        "## Purpose",
        "",
        "Calibrate Article III final-holding labels against the automatic proposition and conclusion-polarity scorers before running more actuator searches. This separates evaluator failure from intervention failure.",
        "",
        "## Status",
        "",
        "This is an internal Codex triage adjudication of the answer-only queue. It is useful for scorer debugging and next-run triage, but it is not independent blind human review and is not a final promotion gate.",
        "",
        "## Inputs And Outputs",
        "",
        markdown_table(
            ["Field", "Value"],
            [
                ["Blind queue", display_path(blind)],
                ["Hidden key", display_path(key)],
                ["Adjudicated rows", display_path(output)],
                ["JSON summary", display_path(summary_path)],
                ["Rows", len(rows)],
            ],
        ),
        "",
        "## Holding Label Counts",
        "",
        markdown_table(["Holding label", "N"], label_counts(rows, "holding_direction_label")),
        "",
        "## Reasoning Quality Counts",
        "",
        markdown_table(["Quality label", "N"], label_counts(rows, "reasoning_quality_label")),
        "",
        "## Mask Label Counts",
        "",
        markdown_table(["Mask label", "N"], label_counts(rows, "mask_label")),
        "",
        "## Hidden Condition Versus Reviewed Holding",
        "",
        markdown_table(
            ["Condition", "N", "Private succeeds", "Public fails/permissible", "Mixed", "Unclear", "Mean confidence"],
            condition_rows(rows),
        ),
        "",
        "## Automatic Polarity Versus Reviewed Holding",
        "",
        f"Exact agreement on eligible reviewed rows: `{agreement_summary['exact_agreement_rows']}/{agreement_summary['eligible_rows']}` = `{agreement_summary['exact_agreement_rate']}`.",
        "",
        markdown_table(["Automatic holding", "Reviewed holding", "N"], agreement_table),
        "",
        "## Proposition/Polarity Score By Reviewed Holding",
        "",
        markdown_table(
            ["Reviewed holding", "N", "Mean proposition delta vs neutral", "Mean auto private-minus-public"],
            proposition_rows(rows),
        ),
        "",
        "## Prompt-Level Rows",
        "",
        markdown_table(
            ["Prompt key", "Hidden condition", "Reviewed holding", "Automatic holding", "Quality", "Confidence"],
            prompt_condition_rows(rows),
        ),
        "",
        "## Gate Interpretation",
        "",
        *gate_lines,
        "",
        "## JSON Summary",
        "",
        f"`{display_path(summary_path)}` contains the machine-readable version of this report.",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--blind", type=Path, default=DEFAULT_BLIND)
    parser.add_argument("--key", type=Path, default=DEFAULT_KEY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument(
        "--mode",
        choices=[
            "codex-triage",
            "codex-triage-long-20260502",
            "codex-triage-ambiguous-20260502",
            "existing",
        ],
        default="codex-triage",
        help="codex-triage applies the embedded first-pass labels; existing reports labels already present in the blind queue.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    blind_rows = read_jsonl(args.blind)
    key_rows = checked_key(read_jsonl(args.key))
    rows = adjudicate(blind_rows=blind_rows, key_rows=key_rows, mode=args.mode)
    write_jsonl(args.output, rows)
    summary = build_summary(rows, mode=args.mode, blind=args.blind, key=args.key, output=args.output)
    write_json(args.summary, summary)
    write_report(
        args.report,
        rows=rows,
        summary=summary,
        blind=args.blind,
        key=args.key,
        output=args.output,
        summary_path=args.summary,
    )
    print(f"Wrote {len(rows)} adjudicated rows to {args.output}")
    print(f"Wrote summary to {args.summary}")
    print(f"Wrote report to {args.report}")


if __name__ == "__main__":
    main()
