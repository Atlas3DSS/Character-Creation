#!/usr/bin/env python3
"""Build a blind dominance-review queue for Article III source-frame labels."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_LABELS = PROJECT_ROOT / "data" / "scotus" / "scotus_article3_source_frame_labels_v1.jsonl"
DEFAULT_BLIND = PROJECT_ROOT / "data" / "scotus" / "scotus_article3_dominance_review_blind_v1.jsonl"
DEFAULT_KEY = PROJECT_ROOT / "data" / "scotus" / "scotus_article3_dominance_review_key_v1.jsonl"
DEFAULT_REPORT = PROJECT_ROOT / "reports" / "scotus_article3_dominance_review_v1.md"

PUBLIC_PRIVATE_FRAMES = {"article3_public_rights", "article3_private_rights"}


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


def stable_int(value: str) -> int:
    return int(hashlib.sha256(value.encode("utf-8")).hexdigest()[:8], 16)


def markdown_table(headers: list[str], rows: list[list[Any]]) -> list[str]:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(item) for item in row) + " |")
    return lines


def grouped_chunks(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_chunk: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        frames = {str(frame) for frame in row.get("matched_frames", [])}
        if not frames.intersection(PUBLIC_PRIVATE_FRAMES):
            continue
        by_chunk.setdefault(str(row["chunk_id"]), []).append(row)

    chunks: list[dict[str, Any]] = []
    for chunk_id, chunk_rows in by_chunk.items():
        first = chunk_rows[0]
        matched_frames = sorted({frame for row in chunk_rows for frame in row.get("matched_frames", [])})
        rule_frames = sorted({str(row["frame"]) for row in chunk_rows})
        chunks.append(
            {
                "chunk_id": chunk_id,
                "review_id": f"article3_dominance::{chunk_id}",
                "case_name": first.get("case_name", ""),
                "source_case_id": first.get("source_case_id", ""),
                "source_citation": first.get("source_citation", ""),
                "source_url": first.get("source_url", ""),
                "term": first.get("term", ""),
                "split": first.get("split", ""),
                "token_count": first.get("token_count", 0),
                "matched_frames": matched_frames,
                "rule_frames": rule_frames,
                "has_public_private_conflict": (
                    "article3_public_rights" in matched_frames and "article3_private_rights" in matched_frames
                ),
                "text": first.get("text", ""),
                "text_cue_masked": first.get("text_cue_masked", first.get("text", "")),
                "evidence_windows": [
                    {
                        "frame": row.get("frame", ""),
                        "evidence_patterns": row.get("evidence_patterns", []),
                        "evidence_window": row.get("evidence_window", ""),
                    }
                    for row in sorted(chunk_rows, key=lambda item: str(item.get("frame", "")))
                ],
            }
        )
    return chunks


def priority_key(row: dict[str, Any]) -> tuple[int, int, int, int]:
    frames = set(row["matched_frames"])
    has_private = "article3_private_rights" in frames
    has_public = "article3_public_rights" in frames
    has_conflict = has_private and has_public
    return (
        1 if has_conflict else 0,
        1 if has_private else 0,
        len(frames),
        -stable_int(str(row["chunk_id"])),
    )


def select_queue(chunks: list[dict[str, Any]], max_items: int) -> list[dict[str, Any]]:
    chunks = sorted(chunks, key=priority_key, reverse=True)
    selected: list[dict[str, Any]] = []
    case_counts: Counter[str] = Counter()

    for row in chunks:
        case_id = str(row.get("source_case_id", ""))
        if case_counts[case_id] >= 8:
            continue
        selected.append(row)
        case_counts[case_id] += 1
        if len(selected) >= max_items:
            return selected

    for row in chunks:
        if row in selected:
            continue
        selected.append(row)
        if len(selected) >= max_items:
            break
    return selected


def blind_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "review_id": row["review_id"],
        "case_name": row["case_name"],
        "source_citation": row["source_citation"],
        "source_url": row["source_url"],
        "term": row["term"],
        "token_count": row["token_count"],
        "review_question": "What is the dominant Article III frame in this excerpt?",
        "allowed_labels": [
            "public_rights_dominant",
            "private_rights_dominant",
            "article1_tribunal_dominant",
            "mixed_comparative",
            "off_target_or_false_positive",
        ],
        "review_label": "",
        "review_confidence": "",
        "review_notes": "",
        "text_cue_masked": row["text_cue_masked"],
    }


def key_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        **row,
        "dominance_review_target": True,
        "review_label": "",
        "review_confidence": "",
        "review_notes": "",
    }


def write_report(path: Path, selected: list[dict[str, Any]], args: argparse.Namespace) -> None:
    frame_counts = Counter(frame for row in selected for frame in row["matched_frames"])
    case_counts = Counter(str(row["case_name"]) for row in selected)
    conflict_count = sum(1 for row in selected if row["has_public_private_conflict"])
    lines = [
        "# SCOTUS Article III Dominance Review Queue v1",
        "",
        "## Purpose",
        "",
        "This queue supports blind review of whether Article III public/private-rights excerpts have a dominant legal frame. It is designed to replace keyword presence with adjudicated labels before any further promotion or causal steering claim.",
        "",
        "## Outputs",
        "",
        f"- Blind queue: `{args.blind}`",
        f"- Key queue: `{args.key}`",
        f"- Source labels: `{args.labels}`",
        f"- Selected excerpts: `{len(selected)}`",
        f"- Public/private conflict excerpts: `{conflict_count}`",
        "",
        "## Matched Frame Counts",
        "",
    ]
    lines.extend(markdown_table(["Frame", "Selected excerpts"], [[frame, count] for frame, count in sorted(frame_counts.items())]))
    lines.extend(["", "## Case Coverage", ""])
    lines.extend(markdown_table(["Case", "Selected excerpts"], [[case, count] for case, count in case_counts.most_common()]))
    lines.extend(
        [
            "",
            "## Review Rules",
            "",
            "1. Use the blind queue for adjudication; do not look at `matched_frames` before assigning `review_label`.",
            "2. Label the dominant legal frame, not every frame mentioned.",
            "3. Use `mixed_comparative` when the excerpt is mainly comparing public and private rights without clearly adopting one frame.",
            "4. Use `off_target_or_false_positive` for syllabus/navigation/citation-only chunks or excerpts that do not reason about Article III adjudication.",
            "5. Only reviewed rows with clear dominant labels should feed the next probe.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--blind", type=Path, default=DEFAULT_BLIND)
    parser.add_argument("--key", type=Path, default=DEFAULT_KEY)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--max-items", type=int, default=80)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = read_jsonl(args.labels)
    chunks = grouped_chunks(rows)
    selected = select_queue(chunks, args.max_items)
    write_jsonl(args.blind, [blind_row(row) for row in selected])
    write_jsonl(args.key, [key_row(row) for row in selected])
    write_report(args.report, selected, args)
    print(f"Wrote {len(selected)} blind review rows")
    print(f"Blind queue: {args.blind}")
    print(f"Key queue: {args.key}")
    print(f"Report: {args.report}")


if __name__ == "__main__":
    main()
