#!/usr/bin/env python3
"""Build a blind Article III holding-direction review queue.

This queue supports evaluator repair. It hides the inserted scratchpad condition
and asks reviewers to label the final legal holding of each answer. The point is
to calibrate automatic proposition/polarity scores against final conclusion
labels before using them to judge future actuator candidates.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

from qwen_eval_budget import MIN_COMPLETE_ANSWER_TOKENS


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_GENERATIONS = (
    PROJECT_ROOT / "sweep_v4" / "scotus_counterfactual_thoughts_20260501_231331" / "generations.jsonl"
)
DEFAULT_POLARITY = (
    PROJECT_ROOT
    / "sweep_v4"
    / "scotus_article3_conclusion_polarity_20260501_232256"
    / "polarity_rows.jsonl"
)
DEFAULT_BLIND = PROJECT_ROOT / "data" / "scotus" / "scotus_article3_holding_review_blind_20260501.jsonl"
DEFAULT_KEY = PROJECT_ROOT / "data" / "scotus" / "scotus_article3_holding_review_key_20260501.jsonl"
DEFAULT_REPORT = PROJECT_ROOT / "reports" / "scotus_article3_holding_review_queue_20260501.md"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
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


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


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


def stable_digest(*parts: Any) -> str:
    raw = "::".join(str(part) for part in parts)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def row_key(row: dict[str, Any]) -> tuple[int, str]:
    return int(row["prompt_id"]), str(row["condition"])


def polarity_by_key(rows: list[dict[str, Any]]) -> dict[tuple[int, str], dict[str, Any]]:
    keyed: dict[tuple[int, str], dict[str, Any]] = {}
    for row in rows:
        if row.get("prompt_id") is None or row.get("condition") is None:
            continue
        keyed[(int(row["prompt_id"]), str(row["condition"]))] = row
    return keyed


def build_rows(
    *,
    generations: list[dict[str, Any]],
    polarities: dict[tuple[int, str], dict[str, Any]],
    source_generations: Path,
    source_polarity: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    blind_rows: list[dict[str, Any]] = []
    key_rows: list[dict[str, Any]] = []
    for row in generations:
        prompt_id = int(row["prompt_id"])
        condition = str(row["condition"])
        review_id = f"article3_holding::{stable_digest(prompt_id, row['prompt_key'], condition)}"
        answer_eval = row.get("answer_proposition_frame_eval", {})
        polarity = polarities.get((prompt_id, condition), {})
        blind_rows.append(
            {
                "review_id": review_id,
                "prompt_id": prompt_id,
                "prompt_key": row["prompt_key"],
                "issue_area": row.get("issue_area", ""),
                "prompt": row["prompt"],
                "answer": row.get("answer", ""),
                "visible_thought_redacted": True,
                "review_question": "What final Article III holding does this answer reach?",
                "holding_direction_allowed_labels": [
                    "article3_objection_succeeds_private_rights",
                    "article3_objection_fails_public_rights_permissible",
                    "mixed_or_distinction_only",
                    "unclear_or_incoherent",
                ],
                "reasoning_quality_allowed_labels": [
                    "legally_coherent",
                    "partly_coherent",
                    "legally_confused",
                    "nonresponsive_or_truncated",
                ],
                "mask_allowed_labels": [
                    "direct_reasoning",
                    "imitating_or_roleplay_mask",
                    "not_assessable",
                ],
                "holding_direction_label": "",
                "reasoning_quality_label": "",
                "mask_label": "",
                "review_confidence": "",
                "review_notes": "",
            }
        )
        key_rows.append(
            {
                "review_id": review_id,
                "source_generations": str(source_generations),
                "source_polarity": str(source_polarity),
                "prompt_id": prompt_id,
                "prompt_key": row["prompt_key"],
                "condition": condition,
                "inserted_thought": row.get("inserted_thought", ""),
                "thinking": row.get("thinking", ""),
                "automatic_polarity_label": polarity.get("label"),
                "automatic_private_score": polarity.get("private_score"),
                "automatic_public_score": polarity.get("public_score"),
                "automatic_net_private_minus_public": polarity.get("net_private_minus_public"),
                "proposition_target_hits": answer_eval.get("target_hits"),
                "proposition_contrast_hits": answer_eval.get("contrast_hits"),
                "proposition_delta_net_vs_neutral": answer_eval.get(
                    "delta_target_minus_contrast_vs_neutral"
                ),
                "holding_direction_label": "",
                "reasoning_quality_label": "",
                "mask_label": "",
                "review_confidence": "",
                "review_notes": "",
            }
        )
    blind_rows.sort(key=lambda item: item["review_id"])
    key_rows.sort(key=lambda item: item["review_id"])
    return blind_rows, key_rows


def condition_summary(key_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in key_rows:
        groups[str(row["condition"])].append(row)
    summaries: list[dict[str, Any]] = []
    for condition, rows in sorted(groups.items()):
        labels = Counter(str(row.get("automatic_polarity_label")) for row in rows)
        summaries.append(
            {
                "condition": condition,
                "n": len(rows),
                "auto_private": labels["private_rights_objection_succeeds"],
                "auto_public": labels["public_rights_adjudication_permissible"],
                "auto_mixed": labels["mixed_or_unclear"],
            }
        )
    return summaries


def write_report(
    path: Path,
    *,
    blind: Path,
    key: Path,
    manifest_path: Path,
    generations: Path,
    polarity: Path,
    generation_answer_tokens: int | None,
    short_generation_answer_budget: bool,
    blind_rows: list[dict[str, Any]],
    key_rows: list[dict[str, Any]],
) -> None:
    summaries = condition_summary(key_rows)
    lines = [
        "# SCOTUS Article III Holding Review Queue",
        "",
        "## Purpose",
        "",
        "Build a blind answer-only queue for final Article III holding direction. The inserted scratchpad is hidden so reviewers label the legal conclusion reached by the answer, not the counterfactual condition.",
        "",
        "## Inputs",
        "",
        f"- Generations: `{generations}`",
        f"- Automatic polarity rows: `{polarity}`",
        f"- Generation answer tokens: `{generation_answer_tokens if generation_answer_tokens is not None else 'unknown'}`",
        f"- Short-budget smoke: `{short_generation_answer_budget}`",
        "",
        "## Outputs",
        "",
        f"- Blind queue: `{blind}`",
        f"- Key file: `{key}`",
        f"- Manifest: `{manifest_path}`",
        f"- Review rows: `{len(blind_rows)}`",
        "",
        "## Automatic Polarity Distribution",
        "",
        markdown_table(
            ["condition", "n", "auto private", "auto public", "auto mixed"],
            [
                [
                    item["condition"],
                    item["n"],
                    item["auto_private"],
                    item["auto_public"],
                    item["auto_mixed"],
                ]
                for item in summaries
            ],
        ),
        "",
        "## Review Instructions",
        "",
        "For each answer, label the final holding direction:",
        "",
        "- `article3_objection_succeeds_private_rights`: the answer concludes that Article III requires an Article III court or that the non-Article-III adjudicator cannot enter final judgment.",
        "- `article3_objection_fails_public_rights_permissible`: the answer concludes that the agency/Article I/non-Article-III process is constitutionally permissible, usually under public-rights or adequate-review reasoning.",
        "- `mixed_or_distinction_only`: the answer explains the distinction but does not clearly resolve the prompt.",
        "- `unclear_or_incoherent`: the answer is truncated, confused, or nonresponsive.",
        "",
        "The automatic polarity labels are in the key file only and should not be used during blind review.",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generations", type=Path, default=DEFAULT_GENERATIONS)
    parser.add_argument("--polarity", type=Path, default=DEFAULT_POLARITY)
    parser.add_argument("--blind", type=Path, default=DEFAULT_BLIND)
    parser.add_argument("--key", type=Path, default=DEFAULT_KEY)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = now_iso()
    source_manifest_path = args.generations.with_name("manifest.json")
    source_manifest = read_json(source_manifest_path) if source_manifest_path.exists() else {}
    generation_answer_tokens_raw = source_manifest.get("answer_tokens")
    generation_answer_tokens = (
        int(generation_answer_tokens_raw) if generation_answer_tokens_raw is not None else None
    )
    short_generation_answer_budget = (
        generation_answer_tokens is not None and generation_answer_tokens < MIN_COMPLETE_ANSWER_TOKENS
    )
    generation_rows = read_jsonl(args.generations)
    polarity_rows = polarity_by_key(read_jsonl(args.polarity))
    blind_rows, key_rows = build_rows(
        generations=generation_rows,
        polarities=polarity_rows,
        source_generations=args.generations,
        source_polarity=args.polarity,
    )
    manifest_path = args.report.with_suffix(".json")
    write_jsonl(args.blind, blind_rows)
    write_jsonl(args.key, key_rows)
    write_json(
        manifest_path,
        {
            "started_at": started,
            "finished_at": now_iso(),
            "generations": str(args.generations),
            "polarity": str(args.polarity),
            "blind": str(args.blind),
            "key": str(args.key),
            "report": str(args.report),
            "review_rows": len(blind_rows),
            "visible_thought_redacted": True,
            "generation_manifest": str(source_manifest_path) if source_manifest_path.exists() else "",
            "generation_answer_tokens": generation_answer_tokens,
            "min_complete_answer_tokens": MIN_COMPLETE_ANSWER_TOKENS,
            "short_generation_answer_budget": short_generation_answer_budget,
            "short_generation_answer_budget_note": (
                "Short Qwen answer budgets are smoke/debug only and should not be used for promotion."
                if short_generation_answer_budget
                else ""
            ),
        },
    )
    write_report(
        args.report,
        blind=args.blind,
        key=args.key,
        manifest_path=manifest_path,
        generations=args.generations,
        polarity=args.polarity,
        generation_answer_tokens=generation_answer_tokens,
        short_generation_answer_budget=short_generation_answer_budget,
        blind_rows=blind_rows,
        key_rows=key_rows,
    )
    print(f"Wrote {args.report}")
    print(f"Wrote {args.blind}")
    print(f"Wrote {args.key}")


if __name__ == "__main__":
    main()
