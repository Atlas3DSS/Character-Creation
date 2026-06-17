#!/usr/bin/env python3
"""Build matched SCOTUS justice contrast pairs from audited chunks."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CHUNKS = PROJECT_ROOT / "data" / "scotus" / "scotus_chunk_inventory.jsonl"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "scotus" / "scotus_matched_pairs_v1.jsonl"
DEFAULT_QUALITY = PROJECT_ROOT / "data" / "scotus" / "manifests" / "scotus_pair_quality_v1.json"
PAIR_SPECS = [
    ("Scalia", "Ginsburg"),
    ("Thomas", "Souter"),
]


def stable_float(value: str) -> float:
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()
    return int(digest[:12], 16) / float(16**12)


def case_split(cluster_id: int | str) -> str:
    value = stable_float(str(cluster_id))
    if value < 0.8:
        return "train"
    if value < 0.9:
        return "dev"
    return "test"


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


def match_key(chunk: dict[str, Any], *, include_direction: bool, match_position_bucket: bool) -> tuple[str, ...]:
    parts = [
        str(chunk.get("issue_area_label") or "unknown"),
        str(chunk.get("section_posture") or chunk.get("opinion_type") or "unknown"),
        str(chunk.get("decade") or "unknown"),
    ]
    if include_direction:
        parts.append(str(chunk.get("decision_direction") or "unknown"))
    if match_position_bucket and chunk.get("chunk_position_bucket"):
        parts.append(str(chunk.get("chunk_position_bucket")))
    parts.append(str(chunk.get("split") or "unknown"))
    return tuple(parts)


def chunk_is_eligible(chunk: dict[str, Any], *, require_section: bool) -> bool:
    if chunk.get("passes_reasoning_filter") is False or chunk.get("excluded") is True:
        return False
    if require_section:
        if not chunk.get("section_author") or chunk.get("section_author") != chunk.get("justice"):
            return False
        if chunk.get("section_confidence") not in {"high", "medium"}:
            return False
    return True


def build_pairs(
    chunks: list[dict[str, Any]],
    *,
    include_direction: bool,
    exclude_unknown_issue: bool,
    max_pairs_per_cell: int | None,
    require_section: bool,
    match_position_bucket: bool,
) -> list[dict[str, Any]]:
    for chunk in chunks:
        chunk["split"] = case_split(chunk["cluster_id"])

    pairs: list[dict[str, Any]] = []
    for text_variant in ["masked", "raw_clean"]:
        variant_chunks = [chunk for chunk in chunks if chunk["text_variant"] == text_variant]
        for justice_a, justice_b in PAIR_SPECS:
            grouped: dict[tuple[str, ...], dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
            for chunk in variant_chunks:
                if not chunk_is_eligible(chunk, require_section=require_section):
                    continue
                if chunk["justice"] not in {justice_a, justice_b}:
                    continue
                if exclude_unknown_issue and (chunk.get("issue_area_label") or "unknown") == "unknown":
                    continue
                grouped[
                    match_key(
                        chunk,
                        include_direction=include_direction,
                        match_position_bucket=match_position_bucket,
                    )
                ][chunk["justice"]].append(chunk)

            for key, justice_map in tqdm(grouped.items(), desc=f"{justice_a}_vs_{justice_b}:{text_variant}", unit="cell"):
                left = sorted(
                    justice_map.get(justice_a, []),
                    key=lambda row: (row["split"], row["cluster_id"], row["token_count"], row["chunk_id"]),
                )
                right = sorted(
                    justice_map.get(justice_b, []),
                    key=lambda row: (row["split"], row["cluster_id"], row["token_count"], row["chunk_id"]),
                )
                n_pairs = min(len(left), len(right))
                if max_pairs_per_cell is not None:
                    n_pairs = min(n_pairs, max_pairs_per_cell)
                for idx in range(n_pairs):
                    chunk_a = left[idx]
                    chunk_b = right[idx]
                    split = chunk_a["split"]
                    if split != chunk_b["split"]:
                        continue
                    pair_id = (
                        f"{justice_a.lower()}_{justice_b.lower()}_{text_variant}_"
                        f"{split}_{chunk_a['chunk_id']}_{chunk_b['chunk_id']}"
                    )
                    pairs.append(
                        {
                            "pair_id": pair_id,
                            "pair": f"{justice_a}_vs_{justice_b}",
                            "split": split,
                            "matching_level": "issue_area+opinion_type+decade+decision_direction"
                            if include_direction
                            else "issue_area+opinion_type+decade",
                            "matching_key": list(key),
                            "case_id_a": chunk_a.get("cluster_id"),
                            "case_id_b": chunk_b.get("cluster_id"),
                            "scdb_id_a": chunk_a.get("scdb_id"),
                            "scdb_id_b": chunk_b.get("scdb_id"),
                            "justice_a": justice_a,
                            "justice_b": justice_b,
                            "text_a": chunk_a["text"],
                            "text_b": chunk_b["text"],
                            "chunk_id_a": chunk_a["chunk_id"],
                            "chunk_id_b": chunk_b["chunk_id"],
                            "issue_area": chunk_a.get("issue_area"),
                            "issue_area_label": chunk_a.get("issue_area_label"),
                            "opinion_type_a": chunk_a.get("opinion_type"),
                            "opinion_type_b": chunk_b.get("opinion_type"),
                            "section_id_a": chunk_a.get("section_id"),
                            "section_id_b": chunk_b.get("section_id"),
                            "section_posture_a": chunk_a.get("section_posture"),
                            "section_posture_b": chunk_b.get("section_posture"),
                            "section_confidence_a": chunk_a.get("section_confidence"),
                            "section_confidence_b": chunk_b.get("section_confidence"),
                            "chunk_position_bucket_a": chunk_a.get("chunk_position_bucket"),
                            "chunk_position_bucket_b": chunk_b.get("chunk_position_bucket"),
                            "chunk_index_in_section_a": chunk_a.get("chunk_index_in_section"),
                            "chunk_index_in_section_b": chunk_b.get("chunk_index_in_section"),
                            "term_a": chunk_a.get("term"),
                            "term_b": chunk_b.get("term"),
                            "decade": chunk_a.get("decade"),
                            "decision_direction": chunk_a.get("decision_direction"),
                            "source_url_a": chunk_a.get("source_url"),
                            "source_url_b": chunk_b.get("source_url"),
                            "text_variant": text_variant,
                            "token_count_a": chunk_a.get("token_count"),
                            "token_count_b": chunk_b.get("token_count"),
                            "citation_count_a": chunk_a.get("citation_count"),
                            "citation_count_b": chunk_b.get("citation_count"),
                        }
                    )
    pairs.sort(key=lambda row: (row["pair"], row["text_variant"], row["split"], row["pair_id"]))
    return pairs


def write_quality_manifest(path: Path, chunks: list[dict[str, Any]], pairs: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    chunk_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for chunk in chunks:
        if chunk.get("text_variant") != "raw_clean":
            continue
        justice = chunk.get("justice", "unknown")
        if chunk.get("passes_reasoning_filter") is False or chunk.get("excluded") is True:
            chunk_counts[justice]["excluded"] += 1
        else:
            chunk_counts[justice]["eligible"] += 1
        if chunk.get("section_posture"):
            chunk_counts[f"{justice}/posture"][str(chunk.get("section_posture"))] += 1
    pair_counts: dict[str, Counter[str]] = defaultdict(Counter)
    same_case_counts: Counter[str] = Counter()
    for pair in pairs:
        key = f"{pair['pair']}/{pair['text_variant']}"
        pair_counts[key][pair["split"]] += 1
        if pair.get("case_id_a") == pair.get("case_id_b"):
            same_case_counts[key] += 1
    payload = {
        "chunk_counts": {key: dict(value) for key, value in sorted(chunk_counts.items())},
        "pair_counts": {key: dict(value) for key, value in sorted(pair_counts.items())},
        "same_case_pair_counts": dict(same_case_counts),
        "total_pairs": len(pairs),
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build matched SCOTUS contrast pairs.")
    parser.add_argument("--chunks", type=Path, default=DEFAULT_CHUNKS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--quality-output", type=Path, default=DEFAULT_QUALITY)
    parser.add_argument("--no-direction", action="store_true", help="Do not match on SCDB decision direction.")
    parser.add_argument("--include-unknown-issue", action="store_true")
    parser.add_argument("--max-pairs-per-cell", type=int, default=None)
    parser.add_argument("--require-section", action="store_true", help="Require target-authored section metadata.")
    parser.add_argument("--ignore-position-bucket", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    chunks = read_jsonl(args.chunks)
    pairs = build_pairs(
        chunks,
        include_direction=not args.no_direction,
        exclude_unknown_issue=not args.include_unknown_issue,
        max_pairs_per_cell=args.max_pairs_per_cell,
        require_section=args.require_section,
        match_position_bucket=not args.ignore_position_bucket,
    )
    write_jsonl(args.output, pairs)
    write_quality_manifest(args.quality_output, chunks, pairs)
    counts: dict[tuple[str, str, str], int] = defaultdict(int)
    for pair in pairs:
        counts[(pair["pair"], pair["text_variant"], pair["split"])] += 1
    print(f"Wrote {args.output} ({len(pairs)} pairs)")
    print(f"Wrote {args.quality_output}")
    for key, count in sorted(counts.items()):
        print(f"{key[0]} {key[1]} {key[2]}: {count}")


if __name__ == "__main__":
    main()
