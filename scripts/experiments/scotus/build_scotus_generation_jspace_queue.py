#!/usr/bin/env python3
"""Build J-space readout rows from generated SCOTUS completions and source prefills."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.experiments.scotus.generate_scotus_prefill_devbox_pairs import (  # noqa: E402
    BASE_SYSTEM,
    side_prompt,
)


DEFAULT_PREFILLS = PROJECT_ROOT / "sweep_v4/scotus_prefill_contrasts_20260706_234523/prefills.jsonl"
DEFAULT_PAIRS = [PROJECT_ROOT / "sweep_v4/scotus_prefill_devbox_pairs_20260707_002627/pairs.jsonl"]


def timestamped_output_dir() -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return PROJECT_ROOT / "sweep_v4" / f"scotus_generation_jspace_queue_{stamp}"


def now_iso() -> str:
    return datetime.now().astimezone().isoformat()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def parse_csv(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def parse_int_csv(value: str) -> list[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def split_paragraphs(text: str) -> list[str]:
    return [part.strip() for part in re.split(r"\n\s*\n+", text.strip()) if part.strip()]


def split_sentences(text: str) -> list[str]:
    compact = re.sub(r"\s+", " ", text.strip())
    if not compact:
        return []
    return [part.strip() for part in re.split(r"(?<=[.!?])\s+(?=[A-Z\"'])", compact) if part.strip()]


def prefix_by_paragraph(text: str, count: int) -> str:
    paragraphs = split_paragraphs(text)
    if not paragraphs:
        return text.strip()
    return "\n\n".join(paragraphs[: min(count, len(paragraphs))]).strip()


def prefix_by_sentence(text: str, count: int) -> str:
    sentences = split_sentences(text)
    if not sentences:
        return text.strip()
    return " ".join(sentences[: min(count, len(sentences))]).strip()


def truncate_to_word_boundary(text: str, max_chars: int) -> str:
    clean = re.sub(r"\s+", " ", text.strip())
    if len(clean) <= max_chars:
        return clean
    truncated = clean[:max_chars].rsplit(" ", 1)[0].strip()
    return truncated or clean[:max_chars].strip()


def generated_context(prefill: dict[str, Any], pair: dict[str, Any], side: str, prefix: str) -> str:
    return (
        f"SYSTEM:\n{BASE_SYSTEM}\n\n"
        f"USER:\n{side_prompt(prefill, side)}\n\n"
        f"ASSISTANT PREFIX:\n{prefix.strip()}"
    )


def source_context(prefill: dict[str, Any], side: str, kind: str, max_chars: int) -> tuple[str, str]:
    side_label = "A" if side == "a" else "B"
    writer = str(prefill.get(f"writer_{side}") or prefill.get(f"label_{side}") or side_label)
    if kind == "seed":
        body = str(prefill.get(f"seed_{side}", "")).strip()
        label = f"source_{side}_seed"
    elif kind == "source_prefix":
        body = truncate_to_word_boundary(str(prefill.get(f"source_text_{side}", "")).strip(), max_chars)
        label = f"source_{side}_prefix"
    else:
        raise ValueError(f"unknown source control kind: {kind}")
    text = (
        f"SYSTEM:\n{BASE_SYSTEM}\n\n"
        f"USER:\n{prefill.get('generation_prompt', '')}\n\n"
        f"SOURCE SIDE {side_label} ({writer}) {kind.upper()}:\n{body}"
    )
    return label, text


def base_metadata(prefill: dict[str, Any], pair: dict[str, Any]) -> dict[str, Any]:
    return {
        "created_at": now_iso(),
        "prefill_id": prefill["prefill_id"],
        "comparison_axis": prefill.get("comparison_axis"),
        "jspace_compare_group": prefill.get("jspace_compare_group"),
        "source_pair_id": prefill.get("source_pair_id"),
        "generated_pair_id": pair.get("id"),
        "source_prefill_index": pair.get("source_prefill_index"),
        "split": prefill.get("split"),
        "issue_area_label": prefill.get("issue_area_label"),
        "decision_direction": prefill.get("decision_direction"),
        "label_a": prefill.get("label_a"),
        "label_b": prefill.get("label_b"),
        "writer_a": prefill.get("writer_a"),
        "writer_b": prefill.get("writer_b"),
        "posture_a": prefill.get("posture_a"),
        "posture_b": prefill.get("posture_b"),
        "chunk_id_a": prefill.get("chunk_id_a"),
        "chunk_id_b": prefill.get("chunk_id_b"),
    }


def add_source_rows(
    rows: list[dict[str, Any]],
    prefill: dict[str, Any],
    pair: dict[str, Any],
    source_controls: list[str],
    source_prefix_chars: int,
) -> None:
    for side in ("a", "b"):
        for kind in source_controls:
            variant, text = source_context(prefill, side, kind, source_prefix_chars)
            row = {
                **base_metadata(prefill, pair),
                "jspace_id": f"{pair['id']}__{variant}",
                "variant": variant,
                "side": side,
                "boundary_kind": kind,
                "boundary_index": 0,
                "boundary_label": variant,
                "generated_chars": 0,
                "text": text,
            }
            rows.append(row)


def add_generated_rows(
    rows: list[dict[str, Any]],
    prefill: dict[str, Any],
    pair: dict[str, Any],
    paragraph_boundaries: int,
    sentence_boundaries: list[int],
    include_full: bool,
) -> None:
    for side in ("a", "b"):
        response = str(pair.get(f"response_{side}", "")).strip()
        if not response:
            continue
        seen_prefixes: set[str] = set()
        boundary_specs: list[tuple[str, int, str]] = []
        for index in range(1, paragraph_boundaries + 1):
            boundary_specs.append(("generated_paragraph", index, prefix_by_paragraph(response, index)))
        for index in sentence_boundaries:
            boundary_specs.append(("generated_sentence", index, prefix_by_sentence(response, index)))
        if include_full:
            boundary_specs.append(("generated_full", -1, response))

        for kind, index, prefix in boundary_specs:
            clean_prefix = prefix.strip()
            if not clean_prefix or clean_prefix in seen_prefixes:
                continue
            seen_prefixes.add(clean_prefix)
            short_kind = {"generated_paragraph": "para", "generated_sentence": "sent", "generated_full": "full"}[kind]
            suffix = "full" if index < 0 else f"{index:02d}"
            variant = f"generated_{side}_{short_kind}_{suffix}"
            row = {
                **base_metadata(prefill, pair),
                "jspace_id": f"{pair['id']}__{variant}",
                "variant": variant,
                "side": side,
                "boundary_kind": kind,
                "boundary_index": index,
                "boundary_label": variant,
                "generated_chars": len(clean_prefix),
                "text": generated_context(prefill, pair, side, clean_prefix),
            }
            rows.append(row)


def select_pairs(
    pairs: list[dict[str, Any]],
    axis_filter: list[str],
    max_pairs_per_axis: int,
) -> list[dict[str, Any]]:
    allowed_axes = set(axis_filter)
    counts: Counter[str] = Counter()
    selected: list[dict[str, Any]] = []
    for pair in pairs:
        axis = str(pair.get("comparison_axis", "unknown"))
        if allowed_axes and axis not in allowed_axes:
            continue
        if counts[axis] >= max_pairs_per_axis:
            continue
        counts[axis] += 1
        selected.append(pair)
    return selected


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    axes = Counter(str(row.get("comparison_axis", "unknown")) for row in rows)
    variants = Counter(str(row.get("variant", "unknown")) for row in rows)
    boundary_kinds = Counter(str(row.get("boundary_kind", "unknown")) for row in rows)
    by_axis_boundary: dict[str, dict[str, int]] = defaultdict(dict)
    for (axis, kind), count in Counter(
        (str(row.get("comparison_axis", "unknown")), str(row.get("boundary_kind", "unknown"))) for row in rows
    ).items():
        by_axis_boundary[axis][kind] = count
    return {
        "rows": len(rows),
        "axes": dict(axes),
        "variants": dict(variants),
        "boundary_kinds": dict(boundary_kinds),
        "by_axis_boundary": dict(by_axis_boundary),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefills", type=Path, default=DEFAULT_PREFILLS)
    parser.add_argument("--pairs", type=Path, nargs="+", default=DEFAULT_PAIRS)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--axis-filter", default="")
    parser.add_argument("--max-pairs-per-axis", type=int, default=4)
    parser.add_argument("--paragraph-boundaries", type=int, default=2)
    parser.add_argument("--sentence-boundaries", default="1,3")
    parser.add_argument("--source-controls", default="seed,source_prefix")
    parser.add_argument("--source-prefix-chars", type=int, default=1200)
    parser.add_argument("--include-full-generated", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or timestamped_output_dir()
    output_dir.mkdir(parents=True, exist_ok=False)
    queue_path = output_dir / "queue.jsonl"
    manifest_path = output_dir / "manifest.json"
    summary_path = output_dir / "summary.json"

    prefills = {str(row["prefill_id"]): row for row in read_jsonl(args.prefills)}
    pairs: list[dict[str, Any]] = []
    for path in args.pairs:
        pairs.extend(read_jsonl(path))
    selected_pairs = select_pairs(pairs, parse_csv(args.axis_filter), args.max_pairs_per_axis)
    if not selected_pairs:
        raise RuntimeError("No generated pairs selected")

    rows: list[dict[str, Any]] = []
    missing_prefills: list[str] = []
    source_controls = parse_csv(args.source_controls)
    sentence_boundaries = parse_int_csv(args.sentence_boundaries)
    for pair in selected_pairs:
        prefill = prefills.get(str(pair.get("prefill_id")))
        if prefill is None:
            missing_prefills.append(str(pair.get("prefill_id")))
            continue
        add_source_rows(rows, prefill, pair, source_controls, args.source_prefix_chars)
        add_generated_rows(
            rows,
            prefill,
            pair,
            args.paragraph_boundaries,
            sentence_boundaries,
            args.include_full_generated,
        )

    if not rows:
        raise RuntimeError("No queue rows built")

    write_jsonl(queue_path, rows)
    summary = summarize(rows)
    write_json(summary_path, summary)
    write_json(
        manifest_path,
        {
            "created_at": now_iso(),
            "script": str(Path(__file__).relative_to(PROJECT_ROOT)),
            "prefills": str(args.prefills),
            "pairs": [str(path) for path in args.pairs],
            "output_dir": str(output_dir),
            "queue": str(queue_path),
            "selected_pairs": len(selected_pairs),
            "missing_prefills": missing_prefills[:20],
            "axis_filter": parse_csv(args.axis_filter),
            "max_pairs_per_axis": args.max_pairs_per_axis,
            "paragraph_boundaries": args.paragraph_boundaries,
            "sentence_boundaries": sentence_boundaries,
            "source_controls": source_controls,
            "source_prefix_chars": args.source_prefix_chars,
            "include_full_generated": args.include_full_generated,
            "diagnostic_only": True,
            "generation_performed": False,
            "summary": summary,
        },
    )
    print(output_dir)


if __name__ == "__main__":
    main()
