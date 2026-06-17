#!/usr/bin/env python3
"""Score Article III answer conclusion polarity.

The proposition-frame scorer counts doctrinal discussion. For Article III
public/private-rights prompts, careful answers often mention both sides of the
distinction. This heuristic scorer adds a conclusion-polarity layer:

- private_rights_objection_succeeds: the Article III objection has force, or a
  non-Article-III adjudicator lacks authority to enter final judgment.
- public_rights_adjudication_permissible: the Article III objection fails, or
  Congress may assign initial adjudication to an agency/Article I tribunal.
- mixed_or_unclear: both directions are present or neither conclusion is clear.

It is a triage scorer, not a substitute for blind review.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

from qwen_eval_budget import MIN_COMPLETE_ANSWER_TOKENS


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "sweep_v4"
DEFAULT_INPUT = (
    PROJECT_ROOT / "sweep_v4" / "scotus_counterfactual_thoughts_20260501_231331" / "generations.jsonl"
)

PRIVATE_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("article_iii_required", re.compile(r"(?i)\barticle\s+iii\s+(?:court|judge|adjudication)\s+is\s+required\b")),
    ("must_be_article_iii", re.compile(r"(?i)\bmust\s+be\s+(?:decided|resolved|adjudicated|heard)\s+by\s+an?\s+article\s+iii\b")),
    ("must_be_article_iii_plural", re.compile(r"(?i)\bmust\s+be\s+(?:decided|resolved|adjudicated|heard)\s+by\s+article\s+iii\s+courts?\b")),
    ("objection_succeeds", re.compile(r"(?i)\b(?:objection|challenge)\s+(?:has\s+merit|is\s+well[- ]founded|should\s+be\s+sustained|succeeds)\b")),
    ("violates_article_iii", re.compile(r"(?i)\b(?:violates|violate|would\s+violate)\s+article\s+iii\b")),
    ("article_iii_does_not_permit", re.compile(r"(?i)\barticle\s+iii\s+does\s+not\s+permit\b")),
    ("article_iii_prohibits", re.compile(r"(?i)\barticle\s+iii\s+(?:prohibits|forbids|bars)\b")),
    ("not_permissible_under_article_iii", re.compile(r"(?i)\bnot\s+(?:permissible|constitutional|valid)\s+under\s+article\s+iii\b")),
    ("constitutional_infirmity", re.compile(r"(?i)\bconstitutional\s+infirmity\s+lies\s+in\b")),
    ("impermissibly_withdrawn", re.compile(r"(?i)\bimpermissibly\s+withdrawn\b[^.]{0,180}\b(?:article\s+iii|judicial\s+power|courts?)\b")),
    ("lacks_authority", re.compile(r"(?i)\b(?:bankruptcy\s+judge|agency|board|tribunal|article\s+i\s+(?:court|tribunal|officer))[^.]{0,180}\b(?:lacks?|lack|cannot|may\s+not)\b[^.]{0,120}\b(?:constitutional\s+authority|final\s+judgment|enter\s+judgment|adjudicate)\b")),
    ("cannot_assign_private", re.compile(r"(?i)\bcongress\s+(?:cannot|may\s+not)\b[^.]{0,180}\b(?:private\s+rights?|traditional\s+(?:common[- ]law\s+)?claims?)\b")),
    ("may_not_vest_final_private", re.compile(r"(?i)\b(?:may\s+not|cannot)\b[^.]{0,120}\b(?:vest|transfer|assign)\b[^.]{0,180}\bfinal\s+adjudication\b[^.]{0,120}\b(?:private\s+rights?|non[- ]article\s+iii|agency|tribunal)\b")),
    ("private_rights_reserved", re.compile(r"(?i)\bprivate\s+rights?[^.]{0,180}\b(?:reserved\s+for|must\s+be\s+adjudicated\s+by|require)\b[^.]{0,80}\barticle\s+iii\b")),
]

PUBLIC_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("objection_fails", re.compile(r"(?i)\b(?:objection|challenge|argument)\s+(?:fails|is\s+without\s+merit|should\s+be\s+rejected|is\s+foreclosed)\b")),
    ("does_not_violate", re.compile(r"(?i)\b(?:does\s+not|do\s+not|would\s+not)\s+violate\s+article\s+iii\b")),
    ("article_iii_does_not_prohibit", re.compile(r"(?i)\barticle\s+iii\s+(?:does\s+not|doesn't)\s+prohibit\b")),
    ("article_iii_does_not_require", re.compile(r"(?i)\barticle\s+iii\s+(?:does\s+not|doesn't)\s+require\b")),
    ("article_iii_permits", re.compile(r"(?i)\barticle\s+iii\s+(?:permits?|allows?)\b[^.]{0,180}\b(?:agency|administrative|article\s+i|non[- ]article\s+iii|tribunal|board|adjudication)\b")),
    ("adjudication_permissible", re.compile(r"(?i)\b(?:agency|administrative|board|tribunal|article\s+i|non[- ]article\s+iii)[^.]{0,180}\b(?:adjudication|proceeding|adjudicator|tribunal|board)[^.]{0,160}\b(?:permissible|constitutional|valid)\b")),
    ("may_proceed_before_agency", re.compile(r"(?i)\bmay\s+proceed\s+before\s+(?:the\s+)?(?:agency|administrative|board|tribunal|administrative\s+law\s+judge|alj)\b")),
    ("permissible_exercise", re.compile(r"(?i)\bpermissible\s+exercise\s+of\s+congress(?:ional)?\s+power\b")),
    ("congress_may_assign", re.compile(r"(?i)\bcongress\s+may\b[^.]{0,180}\b(?:assign|authorize|commit|route|permit)\b[^.]{0,160}\b(?:agency|administrative|article\s+i|non[- ]article\s+iii|tribunal|board)\b")),
    ("agency_may_adjudicate", re.compile(r"(?i)\b(?:agency|board|tribunal|article\s+i\s+(?:court|tribunal))[^.]{0,180}\bmay\b[^.]{0,160}\b(?:adjudicate|decide|make\s+findings|resolve|determine)\b")),
    ("public_rights_permit", re.compile(r"(?i)\bpublic[- ]rights?\s+(?:doctrine|exception)[^.]{0,160}\b(?:permits?|allows?|authorizes?|supports?)\b")),
]

CONTRASTIVE_PUBLIC_RE = re.compile(
    r"(?i)\b(?:while|although|though|even\s+though)\b[^.]{0,80}\bcongress\s+may\b|"
    r"\bcongress\s+may\b[^.]{0,180}\b(?:but|however)\b[^.]{0,180}\b(?:may\s+not|cannot|private\s+rights?)\b"
)

NEGATED_VIOLATION_RE = re.compile(
    r"(?i)\b(?:does|do|did|would|will|could|should)\s+not\s+violates?\s+article\s+iii\b|"
    r"\b(?:doesn't|don't|didn't|wouldn't|won't|couldn't|shouldn't)\s+violates?\s+article\s+iii\b"
)


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


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=False) + "\n", encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=False) + "\n")


def mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * q)))
    return float(ordered[idx])


def fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def md_table(headers: list[str], rows: list[list[Any]]) -> list[str]:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(item) for item in row) + " |")
    return lines


def clean_snippet(text: str, max_chars: int = 500) -> str:
    cleaned = re.sub(r"\s+", " ", text).strip()
    if len(cleaned) <= max_chars:
        return cleaned
    cut = cleaned[: max_chars + 1]
    if " " in cut:
        cut = cut[: cut.rfind(" ")]
    return cut.rstrip() + "..."


def is_negated_private_match(name: str, text: str, match: re.Match[str]) -> bool:
    if name != "violates_article_iii":
        return False
    sentence_start = text.rfind(".", 0, match.start()) + 1
    sentence_end_raw = text.find(".", match.end())
    sentence_end = len(text) if sentence_end_raw == -1 else sentence_end_raw + 1
    sentence = text[sentence_start:sentence_end]
    return bool(NEGATED_VIOLATION_RE.search(sentence))


def score_text(text: str) -> dict[str, Any]:
    private_evidence: list[dict[str, str]] = []
    for name, pattern in PRIVATE_PATTERNS:
        for match in pattern.finditer(text):
            if is_negated_private_match(name, text, match):
                continue
            private_evidence.append({"pattern": name, "match": match.group(0)})
    public_evidence: list[dict[str, str]] = []
    for name, pattern in PUBLIC_PATTERNS:
        for match in pattern.finditer(text):
            sentence_start = text.rfind(".", 0, match.start()) + 1
            sentence_end_raw = text.find(".", match.end())
            sentence_end = len(text) if sentence_end_raw == -1 else sentence_end_raw + 1
            sentence = text[sentence_start:sentence_end]
            if name in {"congress_may_assign", "public_rights_permit"} and CONTRASTIVE_PUBLIC_RE.search(sentence):
                continue
            public_evidence.append({"pattern": name, "match": match.group(0)})
    private_score = len(private_evidence)
    public_score = len(public_evidence)
    if private_score > public_score:
        label = "private_rights_objection_succeeds"
    elif public_score > private_score:
        label = "public_rights_adjudication_permissible"
    else:
        label = "mixed_or_unclear"
    return {
        "private_score": private_score,
        "public_score": public_score,
        "net_private_minus_public": private_score - public_score,
        "label": label,
        "private_evidence": private_evidence[:8],
        "public_evidence": public_evidence[:8],
    }


def scored_rows(rows: list[dict[str, Any]], *, source_jsonl: Path) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row in rows:
        answer = str(row.get("answer") or row.get("text") or "")
        scores = score_text(answer)
        output.append(
            {
                "source_jsonl": str(source_jsonl),
                "prompt_id": row.get("prompt_id"),
                "prompt_key": row.get("prompt_key"),
                "condition": row.get("condition"),
                "sample_type": row.get("sample_type"),
                "control_index": row.get("control_index"),
                "candidate": row.get("candidate"),
                "alpha": row.get("alpha"),
                "random_index": row.get("random_index"),
                "layer": row.get("layer"),
                "answer": answer,
                **scores,
            }
        )
    return output


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[
            (
                str(row.get("condition")),
                str(row.get("sample_type") or ""),
                str(row.get("candidate") or ""),
                "" if row.get("alpha") is None else str(row.get("alpha")),
            )
        ].append(row)
    summaries: list[dict[str, Any]] = []
    for (condition, sample_type, candidate, alpha), group_rows in sorted(groups.items()):
        labels = Counter(str(row["label"]) for row in group_rows)
        summaries.append(
            {
                "condition": condition,
                "sample_type": sample_type,
                "candidate": candidate,
                "alpha": alpha,
                "n": len(group_rows),
                "mean_private_score": mean([float(row["private_score"]) for row in group_rows]),
                "mean_public_score": mean([float(row["public_score"]) for row in group_rows]),
                "mean_net_private_minus_public": mean(
                    [float(row["net_private_minus_public"]) for row in group_rows]
                ),
                "private_label_rate": labels["private_rights_objection_succeeds"] / len(group_rows),
                "public_label_rate": labels["public_rights_adjudication_permissible"] / len(group_rows),
                "mixed_label_rate": labels["mixed_or_unclear"] / len(group_rows),
            }
        )
    return summaries


def summarize_prompt_condition_nulls(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("sample_type"):
            groups[(str(row.get("prompt_key")), str(row.get("condition")))].append(row)
    summaries: list[dict[str, Any]] = []
    for (prompt_key, condition), group_rows in sorted(groups.items()):
        base_rows = [row for row in group_rows if row.get("sample_type") == "base"]
        random_rows = [row for row in group_rows if row.get("sample_type") == "random_control"]
        basis = random_rows or group_rows
        labels = Counter(str(row["label"]) for row in basis)
        net_values = [float(row["net_private_minus_public"]) for row in basis]
        private_scores = [float(row["private_score"]) for row in basis]
        public_scores = [float(row["public_score"]) for row in basis]
        first = group_rows[0]
        base = base_rows[0] if base_rows else None
        summaries.append(
            {
                "prompt_id": first.get("prompt_id"),
                "prompt_key": prompt_key,
                "condition": condition,
                "n_base": len(base_rows),
                "n_random": len(random_rows),
                "base_label": base.get("label") if base else "",
                "base_private_score": base.get("private_score") if base else None,
                "base_public_score": base.get("public_score") if base else None,
                "base_net_private_minus_public": base.get("net_private_minus_public") if base else None,
                "random_private_label_rate": labels["private_rights_objection_succeeds"] / len(basis),
                "random_public_label_rate": labels["public_rights_adjudication_permissible"] / len(basis),
                "random_mixed_label_rate": labels["mixed_or_unclear"] / len(basis),
                "random_mean_private_score": mean(private_scores),
                "random_mean_public_score": mean(public_scores),
                "random_mean_net_private_minus_public": mean(net_values),
                "random_p05_net_private_minus_public": percentile(net_values, 0.05),
                "random_p50_net_private_minus_public": percentile(net_values, 0.50),
                "random_p95_net_private_minus_public": percentile(net_values, 0.95),
            }
        )
    return summaries


def write_report(
    path: Path,
    *,
    manifest: dict[str, Any],
    summaries: list[dict[str, Any]],
    prompt_condition_summaries: list[dict[str, Any]],
    rows: list[dict[str, Any]],
) -> None:
    lines = [
        "# Article III Conclusion Polarity",
        "",
        "## Configuration",
        "",
        f"- Input: `{manifest['input_jsonl']}`",
        f"- Rows: `{manifest['row_count']}`",
        f"- Input answer tokens: `{manifest['input_answer_tokens'] if manifest['input_answer_tokens'] is not None else 'unknown'}`",
        f"- Short-budget smoke: `{manifest['short_input_answer_budget']}`",
        "",
        "## Summary",
        "",
        *md_table(
            [
                "condition",
                "sample_type",
                "candidate",
                "alpha",
                "n",
                "private_score",
                "public_score",
                "net",
                "private_rate",
                "public_rate",
                "mixed_rate",
            ],
            [
                [
                    item["condition"],
                    item["sample_type"],
                    item["candidate"],
                    item["alpha"],
                    item["n"],
                    fmt(item["mean_private_score"]),
                    fmt(item["mean_public_score"]),
                    fmt(item["mean_net_private_minus_public"]),
                    fmt(item["private_label_rate"]),
                    fmt(item["public_label_rate"]),
                    fmt(item["mixed_label_rate"]),
                ]
                for item in summaries
            ],
        ),
        "",
        "## Samples",
        "",
    ]
    if prompt_condition_summaries:
        lines.extend(
            [
                "## Prompt-Condition Nulls",
                "",
                *md_table(
                    [
                        "prompt",
                        "condition",
                        "n_random",
                        "base_label",
                        "base_net",
                        "random_private",
                        "random_public",
                        "random_mixed",
                        "random_net_mean",
                        "random_net_p05",
                        "random_net_p95",
                    ],
                    [
                        [
                            item["prompt_key"],
                            item["condition"],
                            item["n_random"],
                            item["base_label"],
                            fmt(item["base_net_private_minus_public"]),
                            fmt(item["random_private_label_rate"]),
                            fmt(item["random_public_label_rate"]),
                            fmt(item["random_mixed_label_rate"]),
                            fmt(item["random_mean_net_private_minus_public"]),
                            fmt(item["random_p05_net_private_minus_public"]),
                            fmt(item["random_p95_net_private_minus_public"]),
                        ]
                        for item in prompt_condition_summaries[:80]
                    ],
                ),
                "",
            ]
        )
    for row in rows[: min(12, len(rows))]:
        alpha_label = "" if row.get("alpha") is None else f" / alpha {row.get('alpha')}"
        random_source = row.get("random_index")
        if random_source is None:
            random_source = row.get("control_index")
        random_label = "" if random_source is None else f" / random {random_source}"
        sample_label = "" if not row.get("sample_type") else f" / {row.get('sample_type')}"
        lines.extend(
            [
                f"### {row.get('prompt_key')} / {row.get('condition')}{sample_label}{alpha_label}{random_label}",
                "",
                f"- label: `{row['label']}`",
                f"- private/public/net: `{row['private_score']}` / `{row['public_score']}` / `{row['net_private_minus_public']}`",
                "",
                "Answer snippet:",
                "",
                clean_snippet(str(row["answer"]), max_chars=500) or "[none]",
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = now_iso()
    out_dir = args.output_root / f"scotus_article3_conclusion_polarity_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)
    source_manifest_path = args.input_jsonl.with_name("manifest.json")
    source_manifest = read_json(source_manifest_path) if source_manifest_path.exists() else {}
    input_answer_tokens_raw = source_manifest.get("answer_tokens")
    if input_answer_tokens_raw is None:
        input_answer_tokens_raw = source_manifest.get("max_tokens") or source_manifest.get("max_new_tokens")
    input_answer_tokens = int(input_answer_tokens_raw) if input_answer_tokens_raw is not None else None
    short_input_answer_budget = (
        input_answer_tokens is not None and input_answer_tokens < MIN_COMPLETE_ANSWER_TOKENS
    )
    rows = scored_rows(read_jsonl(args.input_jsonl), source_jsonl=args.input_jsonl)
    summaries = summarize(rows)
    prompt_condition_summaries = summarize_prompt_condition_nulls(rows)
    manifest = {
        "started_at": started,
        "finished_at": now_iso(),
        "input_jsonl": str(args.input_jsonl),
        "output_dir": str(out_dir),
        "row_count": len(rows),
        "input_manifest": str(source_manifest_path) if source_manifest_path.exists() else "",
        "input_answer_tokens": input_answer_tokens,
        "min_complete_answer_tokens": MIN_COMPLETE_ANSWER_TOKENS,
        "short_input_answer_budget": short_input_answer_budget,
        "short_input_answer_budget_note": (
            "Short Qwen answer budgets are smoke/debug only and should not be used for promotion."
            if short_input_answer_budget
            else ""
        ),
        "private_patterns": [name for name, _pattern in PRIVATE_PATTERNS],
        "public_patterns": [name for name, _pattern in PUBLIC_PATTERNS],
    }
    write_json(out_dir / "manifest.json", manifest)
    write_jsonl(out_dir / "polarity_rows.jsonl", rows)
    write_jsonl(out_dir / "summary.jsonl", summaries)
    if prompt_condition_summaries:
        write_jsonl(out_dir / "prompt_condition_summary.jsonl", prompt_condition_summaries)
    write_report(
        out_dir / "report.md",
        manifest=manifest,
        summaries=summaries,
        prompt_condition_summaries=prompt_condition_summaries,
        rows=rows,
    )
    print(f"Wrote {out_dir / 'report.md'}", flush=True)


if __name__ == "__main__":
    main()
