#!/usr/bin/env python3
"""Build SCOTUS decision breakdowns, prefill contrasts, and J-space queue rows.

This is offline data preparation only. It does not call a model and does not
generate text. The output is meant to feed two later steps:

1. dev-box generation of novel writer/posture contrastive completions;
2. workstation J-space/J-lens inspection over the exact same prefills.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OPINIONS = PROJECT_ROOT / "data/scotus/scotus_opinion_inventory.jsonl"
DEFAULT_SECTIONS = PROJECT_ROOT / "data/scotus/scotus_section_inventory_v21.jsonl"
DEFAULT_CHUNKS = PROJECT_ROOT / "data/scotus/scotus_chunk_inventory_v21.jsonl"
DEFAULT_MATCHED_PAIRS = PROJECT_ROOT / "data/scotus/scotus_matched_pairs_v21.jsonl"


def now_iso() -> str:
    return datetime.now().astimezone().isoformat()


def timestamped_output_dir() -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return PROJECT_ROOT / "sweep_v4" / f"scotus_prefill_contrasts_{stamp}"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    count = 0
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
    return count


def stable_float(value: str) -> float:
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()
    return int(digest[:12], 16) / float(16**12)


def stable_id(*parts: object, prefix: str) -> str:
    text = "\n".join(str(part) for part in parts)
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
    return f"{prefix}_{digest}"


def normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def truncate_words(text: str, max_words: int) -> str:
    words = normalize_space(text).split()
    if len(words) <= max_words:
        return " ".join(words)
    return " ".join(words[:max_words]).rstrip(" ,;:") + " ..."


def sentence_seed(text: str, max_sentences: int = 2, max_words: int = 90) -> str:
    cleaned = normalize_space(text)
    cleaned = cleaned.replace("e. g.", "e.g.").replace("i. e.", "i.e.")
    cleaned = cleaned.replace("E. g.", "E.g.").replace("I. e.", "I.e.")
    cleaned = re.sub(r"\s+([,.;:])", r"\1", cleaned)
    pieces = re.split(r"(?<=[.!?])\s+", cleaned)
    selected: list[str] = []
    for piece in pieces:
        if not piece:
            continue
        selected.append(piece)
        if len(" ".join(selected).split()) >= 35 or len(selected) >= max_sentences:
            break
    seed = " ".join(selected)
    return truncate_words(seed or cleaned, max_words)


def chunk_is_eligible(row: dict[str, Any]) -> bool:
    if row.get("text_variant") != "masked":
        return False
    if row.get("excluded") is True or row.get("passes_reasoning_filter") is False:
        return False
    if int(row.get("token_count") or 0) < 80:
        return False
    text = str(row.get("text") or "").strip()
    return bool(text)


def build_decision_breakdowns(
    opinions: list[dict[str, Any]],
    sections: list[dict[str, Any]],
    chunks: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    sections_by_opinion: dict[int, list[dict[str, Any]]] = defaultdict(list)
    chunks_by_opinion: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for section in sections:
        sections_by_opinion[int(section["opinion_id"])].append(section)
    for chunk in chunks:
        if chunk.get("text_variant") == "masked" and chunk.get("excluded") is not True:
            chunks_by_opinion[int(chunk["opinion_id"])].append(chunk)

    rows: list[dict[str, Any]] = []
    for opinion in sorted(opinions, key=lambda row: (int(row.get("term") or 0), str(row.get("case_name")), int(row["opinion_id"]))):
        opinion_id = int(opinion["opinion_id"])
        opinion_sections = sections_by_opinion.get(opinion_id, [])
        opinion_chunks = chunks_by_opinion.get(opinion_id, [])
        section_authors = Counter(str(row.get("section_author") or "unknown") for row in opinion_sections)
        section_postures = Counter(str(row.get("section_posture") or "unknown") for row in opinion_sections)
        chunk_authors = Counter(str(row.get("section_author") or row.get("justice") or "unknown") for row in opinion_chunks)
        chunk_buckets = Counter(str(row.get("chunk_position_bucket") or "unknown") for row in opinion_chunks)
        reasoning_chunks = [
            row
            for row in opinion_chunks
            if row.get("passes_reasoning_filter") is not False and row.get("excluded") is not True
        ]
        rows.append(
            {
                "opinion_id": opinion_id,
                "cluster_id": opinion.get("cluster_id"),
                "scdb_id": opinion.get("scdb_id"),
                "case_name": opinion.get("case_name"),
                "date_filed": opinion.get("date_filed"),
                "term": opinion.get("term"),
                "decade": opinion.get("decade"),
                "target_justice": opinion.get("justice"),
                "issue_area": opinion.get("issue_area"),
                "issue_area_label": opinion.get("issue_area_label"),
                "decision_direction": opinion.get("decision_direction"),
                "majority_votes": opinion.get("majority_votes"),
                "minority_votes": opinion.get("minority_votes"),
                "opinion_type": opinion.get("opinion_type"),
                "token_count": opinion.get("token_count"),
                "section_count": len(opinion_sections),
                "section_authors": dict(section_authors),
                "section_postures": dict(section_postures),
                "masked_chunk_count": len(opinion_chunks),
                "masked_reasoning_chunk_count": len(reasoning_chunks),
                "masked_chunk_authors": dict(chunk_authors),
                "masked_chunk_position_buckets": dict(chunk_buckets),
                "representative_chunk_ids": [row.get("chunk_id") for row in reasoning_chunks[:5]],
                "source_url": opinion.get("source_url"),
            }
        )
    return rows


def novel_prompt(
    *,
    axis: str,
    issue_area: str,
    posture: str,
    decade: str,
    decision_direction: str,
    seed_a: str,
    seed_b: str,
    writer_a: str,
    writer_b: str,
) -> str:
    return (
        "Draft a novel Supreme Court opinion passage for a hypothetical case.\n\n"
        f"Comparison axis: {axis}\n"
        f"Issue area: {issue_area}\n"
        f"Opinion posture: {posture}\n"
        f"Era/decade seed: {decade}\n"
        f"Decision-direction seed: {decision_direction}\n\n"
        "Use the following archived excerpts only as legal-rhetorical seeds. Do not quote them, "
        "do not continue their actual cases, and do not mention the source opinions.\n\n"
        f"Seed A ({writer_a}): {seed_a}\n\n"
        f"Seed B ({writer_b}): {seed_b}\n\n"
        "Generate the requested side as 2-4 dense paragraphs of legal reasoning. Preserve legal "
        "plausibility, cite no real cases unless necessary, and make the writer/posture contrast "
        "visible through reasoning moves rather than labels."
    )


def pair_sort_key(row: dict[str, Any]) -> tuple[float, str]:
    return (stable_float(str(row.get("pair_id") or row.get("chunk_id_a") or row)), str(row.get("pair_id") or ""))


def build_between_writer_prefills(matched_pairs: list[dict[str, Any]], max_rows: int) -> list[dict[str, Any]]:
    candidates = [
        row
        for row in matched_pairs
        if row.get("text_variant") == "masked"
        and str(row.get("justice_a") or "") != str(row.get("justice_b") or "")
        and row.get("text_a")
        and row.get("text_b")
    ]
    candidates.sort(key=pair_sort_key)
    rows: list[dict[str, Any]] = []
    for row in candidates[:max_rows]:
        seed_a = sentence_seed(str(row["text_a"]))
        seed_b = sentence_seed(str(row["text_b"]))
        prefill_id = stable_id(row.get("pair_id"), row.get("chunk_id_a"), row.get("chunk_id_b"), prefix="scotus_between")
        rows.append(
            {
                "prefill_id": prefill_id,
                "comparison_axis": "between_writer_matched_chunk",
                "split": row.get("split"),
                "source_pair_id": row.get("pair_id"),
                "label_a": row.get("justice_a"),
                "label_b": row.get("justice_b"),
                "writer_a": row.get("justice_a"),
                "writer_b": row.get("justice_b"),
                "posture_a": row.get("section_posture_a"),
                "posture_b": row.get("section_posture_b"),
                "case_id_a": row.get("case_id_a"),
                "case_id_b": row.get("case_id_b"),
                "chunk_id_a": row.get("chunk_id_a"),
                "chunk_id_b": row.get("chunk_id_b"),
                "issue_area": row.get("issue_area"),
                "issue_area_label": row.get("issue_area_label"),
                "decade": row.get("decade"),
                "decision_direction": row.get("decision_direction"),
                "generation_prompt": novel_prompt(
                    axis="between different SCOTUS writers on matched issue/posture/time metadata",
                    issue_area=str(row.get("issue_area_label") or "unknown"),
                    posture=str(row.get("section_posture_a") or row.get("opinion_type_a") or "unknown"),
                    decade=str(row.get("decade") or "unknown"),
                    decision_direction=str(row.get("decision_direction") or "unknown"),
                    seed_a=seed_a,
                    seed_b=seed_b,
                    writer_a=str(row.get("justice_a") or "writer A"),
                    writer_b=str(row.get("justice_b") or "writer B"),
                ),
                "source_text_a": truncate_words(str(row["text_a"]), 180),
                "source_text_b": truncate_words(str(row["text_b"]), 180),
                "seed_a": seed_a,
                "seed_b": seed_b,
                "jspace_compare_group": prefill_id,
            }
        )
    return rows


def build_within_section_prefills(chunks: list[dict[str, Any]], max_rows: int) -> list[dict[str, Any]]:
    by_section: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for chunk in chunks:
        if chunk_is_eligible(chunk):
            by_section[str(chunk.get("section_id"))].append(chunk)
    candidates: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for section_chunks in by_section.values():
        ordered = sorted(section_chunks, key=lambda row: int(row.get("chunk_index_in_section") or 0))
        if len(ordered) < 2:
            continue
        candidates.append((ordered[0], ordered[-1]))
        for left, right in zip(ordered, ordered[1:]):
            if len(candidates) >= max_rows * 3:
                break
            candidates.append((left, right))
    candidates.sort(key=lambda pair: (stable_float(str(pair[0].get("chunk_id")) + str(pair[1].get("chunk_id"))), str(pair[0].get("chunk_id"))))

    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for left, right in candidates:
        if len(rows) >= max_rows:
            break
        section_id = str(left.get("section_id"))
        prefill_id = stable_id(section_id, left.get("chunk_id"), right.get("chunk_id"), prefix="scotus_within")
        if prefill_id in seen:
            continue
        seen.add(prefill_id)
        writer = str(left.get("section_author") or left.get("justice") or "same writer")
        seed_a = sentence_seed(str(left["text"]))
        seed_b = sentence_seed(str(right["text"]))
        rows.append(
            {
                "prefill_id": prefill_id,
                "comparison_axis": "within_same_opinion_section",
                "split": "analysis",
                "source_pair_id": None,
                "label_a": f"{writer}_earlier",
                "label_b": f"{writer}_later",
                "writer_a": writer,
                "writer_b": writer,
                "posture_a": left.get("section_posture"),
                "posture_b": right.get("section_posture"),
                "case_id_a": left.get("cluster_id"),
                "case_id_b": right.get("cluster_id"),
                "chunk_id_a": left.get("chunk_id"),
                "chunk_id_b": right.get("chunk_id"),
                "issue_area": left.get("issue_area"),
                "issue_area_label": left.get("issue_area_label"),
                "decade": left.get("decade"),
                "decision_direction": left.get("decision_direction"),
                "generation_prompt": novel_prompt(
                    axis="within the same opinion, earlier vs later reasoning by the same writer",
                    issue_area=str(left.get("issue_area_label") or "unknown"),
                    posture=str(left.get("section_posture") or "unknown"),
                    decade=str(left.get("decade") or "unknown"),
                    decision_direction=str(left.get("decision_direction") or "unknown"),
                    seed_a=seed_a,
                    seed_b=seed_b,
                    writer_a=f"{writer} earlier passage",
                    writer_b=f"{writer} later passage",
                ),
                "source_text_a": truncate_words(str(left["text"]), 180),
                "source_text_b": truncate_words(str(right["text"]), 180),
                "seed_a": seed_a,
                "seed_b": seed_b,
                "jspace_compare_group": prefill_id,
            }
        )
    return rows


def build_same_case_multisection_prefills(chunks: list[dict[str, Any]], max_rows: int) -> list[dict[str, Any]]:
    by_case: dict[int, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for chunk in chunks:
        if not chunk_is_eligible(chunk):
            continue
        key = str(chunk.get("section_author") or chunk.get("section_posture") or chunk.get("section_id"))
        by_case[int(chunk["cluster_id"])][key].append(chunk)

    pairs: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for section_map in by_case.values():
        keys = sorted(section_map)
        if len(keys) < 2:
            continue
        for left_key in keys:
            for right_key in keys:
                if left_key >= right_key:
                    continue
                left = sorted(section_map[left_key], key=lambda row: int(row.get("chunk_index_in_section") or 0))[0]
                right = sorted(section_map[right_key], key=lambda row: int(row.get("chunk_index_in_section") or 0))[0]
                if left.get("section_id") == right.get("section_id"):
                    continue
                pairs.append((left, right))
    pairs.sort(key=lambda pair: (stable_float(str(pair[0].get("chunk_id")) + str(pair[1].get("chunk_id"))), str(pair[0].get("chunk_id"))))

    rows: list[dict[str, Any]] = []
    for left, right in pairs[:max_rows]:
        writer_a = str(left.get("section_author") or left.get("justice") or "section A")
        writer_b = str(right.get("section_author") or right.get("justice") or "section B")
        seed_a = sentence_seed(str(left["text"]))
        seed_b = sentence_seed(str(right["text"]))
        prefill_id = stable_id(left.get("cluster_id"), left.get("chunk_id"), right.get("chunk_id"), prefix="scotus_caseparts")
        rows.append(
            {
                "prefill_id": prefill_id,
                "comparison_axis": "within_same_case_different_sections",
                "split": "analysis",
                "source_pair_id": None,
                "label_a": f"{writer_a}_{left.get('section_posture') or 'section'}",
                "label_b": f"{writer_b}_{right.get('section_posture') or 'section'}",
                "writer_a": writer_a,
                "writer_b": writer_b,
                "posture_a": left.get("section_posture"),
                "posture_b": right.get("section_posture"),
                "case_id_a": left.get("cluster_id"),
                "case_id_b": right.get("cluster_id"),
                "chunk_id_a": left.get("chunk_id"),
                "chunk_id_b": right.get("chunk_id"),
                "issue_area": left.get("issue_area"),
                "issue_area_label": left.get("issue_area_label"),
                "decade": left.get("decade"),
                "decision_direction": left.get("decision_direction"),
                "generation_prompt": novel_prompt(
                    axis="within the same case, different opinion sections/postures",
                    issue_area=str(left.get("issue_area_label") or "unknown"),
                    posture=f"{left.get('section_posture') or 'section'} vs {right.get('section_posture') or 'section'}",
                    decade=str(left.get("decade") or "unknown"),
                    decision_direction=str(left.get("decision_direction") or "unknown"),
                    seed_a=seed_a,
                    seed_b=seed_b,
                    writer_a=writer_a,
                    writer_b=writer_b,
                ),
                "source_text_a": truncate_words(str(left["text"]), 180),
                "source_text_b": truncate_words(str(right["text"]), 180),
                "seed_a": seed_a,
                "seed_b": seed_b,
                "jspace_compare_group": prefill_id,
            }
        )
    return rows


def build_jspace_queue(prefills: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for prefill in prefills:
        base = {
            "prefill_id": prefill["prefill_id"],
            "comparison_axis": prefill["comparison_axis"],
            "jspace_compare_group": prefill["jspace_compare_group"],
            "issue_area_label": prefill.get("issue_area_label"),
            "writer_a": prefill.get("writer_a"),
            "writer_b": prefill.get("writer_b"),
            "posture_a": prefill.get("posture_a"),
            "posture_b": prefill.get("posture_b"),
            "probe_positions": [-1, -8, -32],
            "intended_layers": [4, 8, 16, 20, 34, 48, 49, 50, 51],
        }
        variants = [
            ("generation_prompt", prefill["generation_prompt"]),
            ("seed_a", prefill["seed_a"]),
            ("seed_b", prefill["seed_b"]),
            ("source_text_a_prefix", prefill["source_text_a"]),
            ("source_text_b_prefix", prefill["source_text_b"]),
        ]
        for variant, text in variants:
            rows.append(
                {
                    **base,
                    "jspace_id": stable_id(prefill["prefill_id"], variant, prefix="jspace"),
                    "variant": variant,
                    "text": text,
                }
            )
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--opinions", type=Path, default=DEFAULT_OPINIONS)
    parser.add_argument("--sections", type=Path, default=DEFAULT_SECTIONS)
    parser.add_argument("--chunks", type=Path, default=DEFAULT_CHUNKS)
    parser.add_argument("--matched-pairs", type=Path, default=DEFAULT_MATCHED_PAIRS)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--max-between-writer", type=int, default=240)
    parser.add_argument("--max-within-section", type=int, default=180)
    parser.add_argument("--max-same-case-sections", type=int, default=120)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or timestamped_output_dir()
    output_dir.mkdir(parents=True, exist_ok=False)

    opinions = read_jsonl(args.opinions)
    sections = read_jsonl(args.sections)
    chunks = read_jsonl(args.chunks)
    matched_pairs = read_jsonl(args.matched_pairs)

    decision_breakdowns = build_decision_breakdowns(opinions, sections, chunks)
    between = build_between_writer_prefills(matched_pairs, args.max_between_writer)
    within = build_within_section_prefills(chunks, args.max_within_section)
    same_case = build_same_case_multisection_prefills(chunks, args.max_same_case_sections)
    prefills = between + within + same_case
    prefills.sort(key=lambda row: (row["comparison_axis"], stable_float(row["prefill_id"])))
    jspace_queue = build_jspace_queue(prefills)

    decision_count = write_jsonl(output_dir / "decision_breakdown.jsonl", decision_breakdowns)
    prefill_count = write_jsonl(output_dir / "prefills.jsonl", prefills)
    jspace_count = write_jsonl(output_dir / "jspace_queue.jsonl", jspace_queue)

    prefill_axes = Counter(row["comparison_axis"] for row in prefills)
    writer_pairs = Counter(f"{row.get('writer_a')}__vs__{row.get('writer_b')}" for row in prefills)
    summary = {
        "created_at": now_iso(),
        "decision_breakdowns": decision_count,
        "prefills": prefill_count,
        "jspace_queue_rows": jspace_count,
        "prefill_axes": dict(prefill_axes),
        "top_writer_pairings": dict(writer_pairs.most_common(20)),
    }
    manifest = {
        **summary,
        "script": str(Path(__file__).relative_to(PROJECT_ROOT)),
        "generation_performed": False,
        "jspace_forward_pass_performed": False,
        "purpose": "Prepare SCOTUS decision decomposition, novel contrastive prefill prompts, and aligned J-space queue rows.",
        "inputs": {
            "opinions": str(args.opinions),
            "sections": str(args.sections),
            "chunks": str(args.chunks),
            "matched_pairs": str(args.matched_pairs),
        },
        "limits": {
            "max_between_writer": args.max_between_writer,
            "max_within_section": args.max_within_section,
            "max_same_case_sections": args.max_same_case_sections,
        },
        "outputs": {
            "decision_breakdown": str(output_dir / "decision_breakdown.jsonl"),
            "prefills": str(output_dir / "prefills.jsonl"),
            "jspace_queue": str(output_dir / "jspace_queue.jsonl"),
            "summary": str(output_dir / "summary.json"),
        },
    }
    write_json(output_dir / "summary.json", summary)
    write_json(output_dir / "manifest.json", manifest)

    report = [
        "# SCOTUS Prefill Contrast Package",
        "",
        "Offline preparation only. No model generation and no J-space forward pass were performed.",
        "",
        f"- Decisions decomposed: {decision_count}",
        f"- Prefill contrast pairs: {prefill_count}",
        f"- J-space queue rows: {jspace_count}",
        "",
        "## Axes",
    ]
    for axis, count in sorted(prefill_axes.items()):
        report.append(f"- {axis}: {count}")
    report.extend(
        [
            "",
            "## Next Step",
            "",
            "Use `prefills.jsonl` for SCOTUS-specific dev-box generation with complete Qwen token budgets, then use `jspace_queue.jsonl` for workstation Qwen3.5 J-space readouts at the listed layers/positions.",
        ]
    )
    (output_dir / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(output_dir)


if __name__ == "__main__":
    main()
