#!/usr/bin/env python3
"""Build a source-grounded SCOTUS legal-frame seed set.

The output is intentionally labeled `silver`: labels are produced by strict,
proposition-oriented rules over real opinion chunks and should be manually
reviewed before being used as final evidence. The point is to move beyond
synthetic cue-heavy contrasts while keeping provenance and evidence explicit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CHUNKS = PROJECT_ROOT / "data" / "scotus" / "scotus_chunk_inventory_v21.jsonl"
DEFAULT_LABELS = PROJECT_ROOT / "data" / "scotus" / "scotus_source_frame_labels_v1.jsonl"
DEFAULT_QUEUE = PROJECT_ROOT / "data" / "scotus" / "scotus_source_frame_review_queue_v1.jsonl"
DEFAULT_REPORT = PROJECT_ROOT / "reports" / "scotus_source_frame_seed_v1.md"


@dataclass(frozen=True)
class FrameRule:
    frame: str
    issue_family: str
    definition: str
    required_any: tuple[tuple[str, ...], ...]
    evidence_patterns: tuple[str, ...]
    exclude_any: tuple[str, ...] = ()


FRAME_RULES: tuple[FrameRule, ...] = (
    FrameRule(
        frame="article3_public_rights",
        issue_family="Judicial Power",
        definition=(
            "Congress may assign adjudication outside Article III where the matter "
            "is treated as a public-rights or government/regulatory entitlement dispute."
        ),
        required_any=(
            (r"\bpublic rights?\b",),
            (
                r"\barticle iii\b",
                r"\bnon[- ]article iii\b",
                r"\bpublic[- ]rights? exception\b",
                r"\bcongress\b.{0,120}\b(assign|conferr|authoriz)",
                r"\badministrative (?:tribunal|adjudicat|agency)\b",
                r"\bagenc(?:y|ies)\b.{0,80}\badjudicat",
            ),
            (
                r"\badjudicat",
                r"\btribunal",
                r"\bcourt",
                r"\bagenc(?:y|ies)\b",
            ),
        ),
        evidence_patterns=(
            r"\bpublic rights?\b",
            r"\bcongress\b.{0,80}\b(assign|conferr|authoriz)",
            r"\bnon[- ]article iii\b",
            r"\badministrative\b.{0,80}\badjudicat",
            r"\bregulatory scheme\b",
        ),
        exclude_any=(
            r"\barticle iii agreement\b",
            r"\biad article iii\b",
            r"\bpanama canal\b",
            r"\bpublic right-of-way\b",
            r"\bprivate riparian owner\b",
            r"\bpublic rights, and the state may regulate\b",
            r"\bindividual rights\b.{0,80}\bpublic rights\b",
            r"\blegislatively pronounced\b",
        ),
    ),
    FrameRule(
        frame="article3_private_rights",
        issue_family="Judicial Power",
        definition=(
            "A private/common-law dispute, damages claim, or traditional suit at law "
            "is treated as requiring Article III adjudication or Article III safeguards."
        ),
        required_any=(
            (r"\bprivate rights?\b", r"\bcommon[- ]law\b.{0,80}\bdamages\b", r"\bsuit at law\b"),
            (r"\barticle iii\b", r"\bnon[- ]article iii\b"),
            (
                r"\badjudicat",
                r"\btribunal",
                r"\bcourt",
                r"\bjudicial power\b",
                r"\bjudge",
            ),
        ),
        evidence_patterns=(
            r"\bprivate rights?\b",
            r"\bcommon[- ]law\b.{0,80}\bdamages\b",
            r"\bsuit at law\b",
            r"\blife tenure\b",
            r"\bsalary protection\b",
        ),
        exclude_any=(
            r"\barticle iii agreement\b",
            r"\biad article iii\b",
            r"\bprivate right of action\b",
            r"\bpublic rights, and the state may regulate\b",
            r"\bprivate riparian owner\b",
            r"\bcommon law and the law of nations\b",
            r"\bfederal common law\b",
            r"\bfinal judgments?\b",
            r"\brights determined thereby\b",
        ),
    ),
    FrameRule(
        frame="article3_article1_tribunal",
        issue_family="Judicial Power",
        definition=(
            "The excerpt contrasts Article III judicial power with Article I, legislative, "
            "bankruptcy, agency, or other non-Article III tribunals."
        ),
        required_any=(
            (
                r"\barticle i (?:tax court|military judge|court|courts|tribunal|tribunals)\b",
                r"\barticle i rather than an article iii court\b",
                r"\barticle i powers\b.{0,120}\bcourt-like administrative tribunals\b",
                r"\blegislative courts?\b",
                r"\bnon[- ]article iii\b",
                r"\bbankruptcy judges?\b",
                r"\badministrative law judges?\b",
            ),
            (r"\barticle iii\b", r"\bjudicial power\b", r"\blife tenure\b", r"\bsalary protection\b"),
        ),
        evidence_patterns=(
            r"\barticle i (?:tax court|military judge|court|courts|tribunal|tribunals)\b",
            r"\barticle i rather than an article iii court\b",
            r"\barticle i powers\b.{0,120}\bcourt-like administrative tribunals\b",
            r"\blegislative courts?\b",
            r"\bnon[- ]article iii\b",
            r"\bbankruptcy judges?\b",
            r"\blife tenure\b",
            r"\bsalary protection\b",
        ),
        exclude_any=(r"\barticle iii agreement\b", r"\biad article iii\b"),
    ),
    FrameRule(
        frame="article3_case_or_controversy",
        issue_family="Judicial Power",
        definition=(
            "The excerpt applies Article III case-or-controversy, standing, injury, "
            "mootness, or jurisdictional limits on federal courts."
        ),
        required_any=(
            (
                r"\bcase or controversy\b",
                r"\bcase-or-controversy\b",
                r"\bstanding\b",
                r"\binjury in fact\b",
                r"\bmoot(?:ness)?\b",
            ),
            (r"\barticle iii\b", r"\bfederal courts?\b", r"\bjudicial power\b"),
        ),
        evidence_patterns=(
            r"\bcase or controversy\b",
            r"\bcase-or-controversy\b",
            r"\bstanding\b",
            r"\binjury in fact\b",
            r"\bmoot(?:ness)?\b",
            r"\bjurisdiction\b",
        ),
        exclude_any=(r"\barticle iii agreement\b", r"\biad article iii\b"),
    ),
    FrameRule(
        frame="article3_final_judgment_separation",
        issue_family="Judicial Power",
        definition=(
            "The excerpt frames Article III as protecting final judgments or separating "
            "judicial power from congressional revision."
        ),
        required_any=(
            (r"\bfinal judgments?\b", r"\bretroactive legislation\b", r"\bset aside\b"),
            (r"\barticle iii\b", r"\bjudicial power\b", r"\bseparation of powers\b"),
        ),
        evidence_patterns=(
            r"\bfinal judgments?\b",
            r"\bretroactive legislation\b",
            r"\bset aside\b.{0,80}\bjudgments?\b",
            r"\bseparation of powers\b",
            r"\bjudicial department\b",
        ),
        exclude_any=(r"\biad article iii\b",),
    ),
    FrameRule(
        frame="fourth_search_incident_chimel",
        issue_family="Criminal Procedure",
        definition=(
            "The excerpt applies the search-incident-to-arrest family, including Chimel, "
            "Belton, immediate control, officer safety, or evidence preservation."
        ),
        required_any=(
            (
                r"\bsearch incident to (?:an )?arrest\b",
                r"\bchimel\b",
                r"\bbelton\b",
                r"\bimmediate control\b",
            ),
            (
                r"\barrest\b",
                r"\bofficer safety\b",
                r"\bdestruction of evidence\b",
                r"\bevidence (?:he might )?(?:conceal|destroy|preserv)",
            ),
        ),
        evidence_patterns=(
            r"\bsearch incident to (?:an )?arrest\b",
            r"\bchimel\b",
            r"\bbelton\b",
            r"\bimmediate control\b",
            r"\bofficer safety\b",
            r"\bdestruction of evidence\b",
        ),
        exclude_any=(r"\briley v\. national federation\b",),
    ),
    FrameRule(
        frame="fourth_plain_view_independent_source",
        issue_family="Criminal Procedure",
        definition=(
            "The excerpt frames a Fourth Amendment dispute through plain view, "
            "independent source, warrant execution, or suppression limits."
        ),
        required_any=(
            (r"\bplain view\b", r"\bindependent source\b"),
            (r"\bfourth amendment\b", r"\bwarrant\b", r"\bsuppress"),
        ),
        evidence_patterns=(
            r"\bplain view\b",
            r"\bindependent source\b",
            r"\bwarrant\b",
            r"\bsuppress(?:ion)?\b",
        ),
    ),
    FrameRule(
        frame="fourth_home_exigency",
        issue_family="Criminal Procedure",
        definition=(
            "The excerpt applies home-entry exigency, emergency aid, hot pursuit, "
            "knock-and-announce exigency, or no-knock reasoning."
        ),
        required_any=(
            (r"\bexigenc(?:y|ies|t)\b", r"\bemergency aid\b", r"\bhot pursuit\b", r"\bno-knock\b"),
            (r"\bhome\b", r"\bwarrantless entry\b", r"\bentry\b", r"\bknock(?:ing)? and announc"),
        ),
        evidence_patterns=(
            r"\bexigenc(?:y|ies|t)\b",
            r"\bemergency aid\b",
            r"\bhot pursuit\b",
            r"\bno-knock\b",
            r"\bwarrantless entry\b",
            r"\bknock(?:ing)? and announc",
        ),
    ),
    FrameRule(
        frame="fourth_technology_privacy",
        issue_family="Criminal Procedure",
        definition=(
            "The excerpt applies Fourth Amendment privacy reasoning to sense-enhancing "
            "technology, thermal imaging, or digital/computing records."
        ),
        required_any=(
            (r"\bthermal imag", r"\bsense-enhancing technology\b", r"\bdigital\b", r"\bcomputer\b"),
            (r"\bfourth amendment\b", r"\bsearch\b", r"\bprivacy\b"),
        ),
        evidence_patterns=(
            r"\bthermal imag",
            r"\bsense-enhancing technology\b",
            r"\bdigital\b",
            r"\bcomputer\b",
            r"\bfourth amendment\b",
            r"\breasonable expectation of privacy\b",
        ),
        exclude_any=(r"\bdigital databases?\b", r"\bcable-modem\b"),
    ),
)


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
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def stable_split(value: str) -> str:
    bucket = int(hashlib.sha256(value.encode("utf-8")).hexdigest()[:8], 16) % 10
    if bucket < 7:
        return "train"
    if bucket < 8:
        return "dev"
    return "test"


def stable_int(value: str) -> int:
    return int(hashlib.sha256(value.encode("utf-8")).hexdigest()[:8], 16)


def match_regexes(text: str, patterns: tuple[str, ...]) -> list[str]:
    return [pattern for pattern in patterns if re.search(pattern, text, flags=re.IGNORECASE)]


def rule_matches(text: str, rule: FrameRule) -> tuple[bool, list[str], list[str]]:
    excludes = match_regexes(text, rule.exclude_any)
    if excludes:
        return False, [], excludes
    for group in rule.required_any:
        if not match_regexes(text, group):
            return False, [], []
    evidence = match_regexes(text, rule.evidence_patterns)
    return True, evidence, []


def token_window(text: str, evidence: list[str], window_chars: int) -> str:
    if not evidence:
        return " ".join(text.split())[:window_chars]
    best_start = 0
    for pattern in evidence:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            best_start = max(0, match.start() - window_chars // 3)
            break
    window = text[best_start : best_start + window_chars]
    return " ".join(window.split())


def source_row_allowed(row: dict[str, Any]) -> bool:
    return (
        row.get("text_variant") == "raw_clean"
        and not bool(row.get("excluded"))
        and bool(row.get("passes_reasoning_filter", True))
        and int(row.get("token_count") or 0) >= 60
    )


def priority_score(row: dict[str, Any], rule: FrameRule, evidence: list[str]) -> tuple[int, int, int, str]:
    issue_bonus = 2 if row.get("issue_area_label") == rule.issue_family else 0
    posture_bonus = 1 if row.get("section_posture") in {"majority", "dissent", "concurrence_in_judgment"} else 0
    return (
        issue_bonus,
        len(evidence),
        posture_bonus,
        str(row.get("case_name", "")),
    )


def build_records(rows: list[dict[str, Any]], max_per_frame: int, window_chars: int) -> list[dict[str, Any]]:
    candidates_by_frame: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in tqdm(rows, desc="scan source chunks"):
        if not source_row_allowed(row):
            continue
        text = str(row.get("text", ""))
        for rule in FRAME_RULES:
            matched, evidence, excludes = rule_matches(text, rule)
            if not matched:
                continue
            cluster_key = str(row.get("cluster_id") or row.get("opinion_id") or row.get("chunk_id"))
            record = {
                "record_id": f"{row.get('chunk_id')}::{rule.frame}",
                "frame": rule.frame,
                "issue_family": rule.issue_family,
                "label": 1,
                "label_source": "strict_source_rule_v1",
                "label_confidence": "silver_high",
                "label_definition": rule.definition,
                "evidence_patterns": evidence,
                "global_split": stable_split(cluster_key),
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
                "text": text,
                "evidence_window": token_window(text, evidence, window_chars),
                "review_status": "unreviewed",
                "review_notes": "",
                "_priority": priority_score(row, rule, evidence),
            }
            candidates_by_frame[rule.frame].append(record)

    selected: list[dict[str, Any]] = []
    for frame, frame_rows in candidates_by_frame.items():
        frame_rows.sort(key=lambda item: item["_priority"], reverse=True)
        used_clusters: set[Any] = set()
        diverse_rows: list[dict[str, Any]] = []
        overflow: list[dict[str, Any]] = []
        for record in frame_rows:
            cluster = record.get("cluster_id")
            if cluster not in used_clusters:
                diverse_rows.append(record)
                used_clusters.add(cluster)
            else:
                overflow.append(record)
        for record in (diverse_rows + overflow)[:max_per_frame]:
            record.pop("_priority", None)
            selected.append(record)
    assign_frame_splits(selected)
    selected.sort(key=lambda item: (item["frame"], item["split"], str(item["case_name"]), str(item["chunk_id"])))
    return selected


def assign_frame_splits(records: list[dict[str, Any]]) -> None:
    records_by_frame: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        records_by_frame[str(record["frame"])].append(record)

    for frame, frame_records in records_by_frame.items():
        clusters = sorted(
            {str(record.get("cluster_id") or record.get("opinion_id") or record.get("chunk_id")) for record in frame_records},
            key=lambda value: stable_int(f"{frame}:{value}"),
        )
        n_clusters = len(clusters)
        if n_clusters >= 5:
            n_dev = max(1, round(n_clusters * 0.15))
            n_test = max(1, round(n_clusters * 0.15))
            n_train = max(1, n_clusters - n_dev - n_test)
        elif n_clusters >= 3:
            n_train = n_clusters - 2
            n_dev = 1
            n_test = 1
        elif n_clusters == 2:
            n_train = 1
            n_dev = 0
            n_test = 1
        else:
            n_train = n_clusters
            n_dev = 0
            n_test = 0

        split_by_cluster: dict[str, str] = {}
        for cluster in clusters[:n_train]:
            split_by_cluster[cluster] = "train"
        for cluster in clusters[n_train : n_train + n_dev]:
            split_by_cluster[cluster] = "dev"
        for cluster in clusters[n_train + n_dev : n_train + n_dev + n_test]:
            split_by_cluster[cluster] = "test"

        for record in frame_records:
            cluster = str(record.get("cluster_id") or record.get("opinion_id") or record.get("chunk_id"))
            record["split"] = split_by_cluster[cluster]


def split_counts(records: list[dict[str, Any]]) -> dict[str, Counter[str]]:
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    for record in records:
        counts[str(record["frame"])][str(record["split"])] += 1
    return counts


def markdown_table(headers: list[str], rows: list[list[Any]]) -> list[str]:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(item) for item in row) + " |")
    return lines


def write_report(path: Path, records: list[dict[str, Any]], args: argparse.Namespace) -> None:
    counts = split_counts(records)
    frame_rows: list[list[Any]] = []
    for rule in FRAME_RULES:
        counter = counts.get(rule.frame, Counter())
        total = sum(counter.values())
        frame_rows.append([rule.frame, rule.issue_family, total, counter["train"], counter["dev"], counter["test"]])

    justice_rows = [[justice, count] for justice, count in Counter(str(r.get("justice")) for r in records).most_common()]
    sample_rows = []
    for record in records[: min(16, len(records))]:
        sample_rows.append(
            [
                record["frame"],
                record["split"],
                record.get("case_name", ""),
                record.get("justice", ""),
                record.get("section_posture", ""),
                ", ".join(record.get("evidence_patterns", [])[:3]),
                str(record.get("evidence_window", ""))[:220],
            ]
        )

    lines = [
        "# SCOTUS Source Frame Seed v1",
        "",
        "## Purpose",
        "",
        "This seed set creates source-grounded legal-frame labels from real SCOTUS opinion chunks. Labels are strict-rule silver labels and require manual review before final steering claims.",
        "",
        "## Outputs",
        "",
        f"- Labels: `{args.labels}`",
        f"- Review queue: `{args.queue}`",
        f"- Source chunks: `{args.chunks}`",
        "",
        "## Label Counts",
        "",
    ]
    lines.extend(markdown_table(["Frame", "Issue family", "Total", "Train", "Dev", "Test"], frame_rows))
    lines.extend(
        [
            "",
            "## Justice Coverage",
            "",
        ]
    )
    lines.extend(markdown_table(["Justice", "Records"], justice_rows))
    lines.extend(
        [
            "",
            "## Sample Evidence Windows",
            "",
        ]
    )
    lines.extend(
        markdown_table(
            ["Frame", "Split", "Case", "Justice", "Posture", "Evidence", "Window"],
            sample_rows,
        )
    )
    lines.extend(
        [
            "",
            "## Use Rules",
            "",
            "1. Treat these labels as `silver_high`, not adjudicated gold labels.",
            "2. Do not train a final probe or claim a circuit until a human/blind review pass accepts or corrects the labels.",
            "3. Use the frame-stratified `split` field for small frame probes; it is cluster-held-out within each frame.",
            "4. The `global_split` field is also included for stricter cross-frame audits, but sparse frames may be unbalanced under that split.",
            "5. Prefer frame contrasts with nonzero train/dev/test support and source diversity.",
            "6. The sparse or zero rows are informative: they mark frames that should not be forced from this corpus.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chunks", type=Path, default=DEFAULT_CHUNKS)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--queue", type=Path, default=DEFAULT_QUEUE)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--max-per-frame", type=int, default=48)
    parser.add_argument("--window-chars", type=int, default=900)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = read_jsonl(args.chunks)
    records = build_records(rows, max_per_frame=args.max_per_frame, window_chars=args.window_chars)
    write_jsonl(args.labels, records)
    write_jsonl(args.queue, records)
    write_report(args.report, records, args)
    print(f"Wrote {len(records)} records")
    print(f"Labels: {args.labels}")
    print(f"Queue: {args.queue}")
    print(f"Report: {args.report}")


if __name__ == "__main__":
    main()
