#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


DEFAULT_LAST_TOKEN_DIR = Path("sweep_v4/scotus_sae_poke_20260430_224651")
DEFAULT_ALL_POSITION_DIR = Path("sweep_v4/scotus_sae_poke_20260430_233245")
DEFAULT_PROXY_DIR = Path("sweep_v4/scotus_qwen4bit_proxy_20260501_045257")
DEFAULT_OVERLAP_DIR = Path("sweep_v4/scotus_sae_overlap_all4_20260430_153933")
DEFAULT_MATCHED_PAIRS = Path("data/scotus/scotus_matched_pairs_v21.jsonl")
DEFAULT_REPORT = Path("reports/scotus_phase5_decision_20260501.md")

FRAME_PATTERN_SUBSET = {
    "Judicial Power": {
        "article3_public_rights": ["public rights", "public-rights"],
        "article3_private_rights": ["private rights", "private-rights"],
        "article3_article1_tribunal": [
            "article i",
            "non-article iii",
            "article iii tribunal",
            "article iii court",
            "bankruptcy court",
            "agency adjudication",
        ],
        "article3_case_or_controversy": ["case or controversy", "standing", "mootness", "ripeness"],
    },
    "Criminal Procedure": {
        "fourth_search_incident_chimel": ["search incident", "chimel", "immediate control", "arrestee"],
        "fourth_digital_privacy": ["cell phone", "smartphone", "digital data", "digital contents"],
        "fourth_plain_view_closed_container": ["plain view", "closed container", "locked backpack"],
        "fourth_home_exigency": ["home", "warrantless entry", "exigent circumstances", "emergency"],
        "fourth_stop_reasonable_suspicion": [
            "traffic stop",
            "reasonable suspicion",
            "dog sniff",
            "prolonged stop",
        ],
    },
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def count_jsonl(path: Path) -> int:
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def fmt(value: Any, digits: int = 2) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(item) for item in row) + " |")
    return "\n".join(lines)


def top_prompt_matched_rows(run_dir: Path, label: str, limit: int = 8) -> list[list[Any]]:
    rows = read_jsonl(run_dir / "candidate_vs_prompt_matched_random.jsonl")
    rows = sorted(rows, key=lambda row: abs(float(row["z_vs_prompt_matched_random"])), reverse=True)
    table_rows: list[list[Any]] = []
    for row in rows[:limit]:
        table_rows.append(
            [
                label,
                row["candidate"],
                fmt(row["alpha"], 2),
                row["n"],
                fmt(row["mean_prompt_matched_delta_minus_random"], 2),
                fmt(row["random_residual_sd"], 2),
                fmt(row["z_vs_prompt_matched_random"], 2),
                fmt(row["percentile_vs_prompt_matched_random_rows"], 2),
                fmt(row["prompt_win_rate_vs_random_mean"], 2),
            ]
        )
    return table_rows


def null_volatility_tables(proxy_dir: Path, limit: int = 10) -> tuple[list[list[Any]], list[list[Any]]]:
    rows = read_json(proxy_dir / "prompt_condition_nulls.json")
    volatile = sorted(rows, key=lambda row: float(row["sd_delta_target_hits_vs_base"]), reverse=True)[:limit]
    stable_candidates = [
        row
        for row in rows
        if abs(float(row["mean_delta_target_hits_vs_base"])) <= 0.5
        and float(row["sd_delta_target_hits_vs_base"]) <= 1.1
    ]
    stable = sorted(
        stable_candidates,
        key=lambda row: (float(row["sd_delta_target_hits_vs_base"]), abs(float(row["mean_delta_target_hits_vs_base"]))),
    )[:limit]

    volatile_rows = [
        [
            row["prompt_key"],
            row["condition"],
            row["issue_area"],
            fmt(row["mean_delta_target_hits_vs_base"], 2),
            fmt(row["sd_delta_target_hits_vs_base"], 2),
            fmt(row["p05_delta_target_hits_vs_base"], 1),
            fmt(row["p95_delta_target_hits_vs_base"], 1),
        ]
        for row in volatile
    ]
    stable_rows = [
        [
            row["prompt_key"],
            row["condition"],
            row["issue_area"],
            fmt(row["mean_delta_target_hits_vs_base"], 2),
            fmt(row["sd_delta_target_hits_vs_base"], 2),
            fmt(row["p05_delta_target_hits_vs_base"], 1),
            fmt(row["p95_delta_target_hits_vs_base"], 1),
        ]
        for row in stable
    ]
    return volatile_rows, stable_rows


def off_domain_tables(proxy_dir: Path, limit: int = 10) -> tuple[list[list[Any]], list[list[Any]]]:
    rows = read_jsonl(proxy_dir / "generations.jsonl")
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    frame_counts: Counter[str] = Counter()
    for row in rows:
        if row.get("sample_type") != "random_control":
            continue
        key = (row["prompt_key"], row["condition"], row["issue_area"])
        groups[key].append(row)
        for frame in row.get("frame_eval", {}).get("off_domain_frames", []):
            frame_counts[str(frame)] += 1

    group_rows: list[dict[str, Any]] = []
    for (prompt_key, condition, issue_area), group_rows_raw in groups.items():
        n = len(group_rows_raw)
        off_present = sum(1 for row in group_rows_raw if row.get("frame_eval", {}).get("off_domain_present"))
        off_hits = [
            float(row.get("frame_eval", {}).get("off_domain_hits", 0.0))
            for row in group_rows_raw
        ]
        group_rows.append(
            {
                "prompt_key": prompt_key,
                "condition": condition,
                "issue_area": issue_area,
                "n": n,
                "off_domain_present_rate": 0.0 if n == 0 else off_present / n,
                "mean_off_domain_hits": 0.0 if n == 0 else sum(off_hits) / n,
            }
        )
    group_rows = sorted(
        group_rows,
        key=lambda row: (float(row["off_domain_present_rate"]), float(row["mean_off_domain_hits"])),
        reverse=True,
    )[:limit]

    off_domain_rows = [
        [
            row["prompt_key"],
            row["condition"],
            row["issue_area"],
            row["n"],
            fmt(row["off_domain_present_rate"], 2),
            fmt(row["mean_off_domain_hits"], 2),
        ]
        for row in group_rows
    ]
    frame_rows = [[frame, count] for frame, count in frame_counts.most_common(limit)]
    return off_domain_rows, frame_rows


def low_overlap_rows(overlap_dir: Path, limit: int = 12) -> list[list[Any]]:
    rows = read_jsonl(overlap_dir / "overlap_pairwise.jsonl")
    rows = [
        row
        for row in rows
        if row.get("group_field") == "issue_area_label"
        and row.get("group_value") in {"Judicial Power", "Criminal Procedure"}
    ]
    rows = sorted(rows, key=lambda row: float(row["weighted_jaccard"]))[:limit]
    return [
        [
            row["group_value"],
            f"{row['justice_a']} / {row['justice_b']}",
            row["region"],
            row["layer"],
            fmt(row["top_jaccard"], 3),
            fmt(row["weighted_jaccard"], 3),
            fmt(row["cosine_df"], 3),
            f"{row['n_a']} / {row['n_b']}",
        ]
        for row in rows
    ]


def frame_label_feasibility_rows(matched_pairs_path: Path) -> list[list[Any]]:
    counts: dict[tuple[str, str, str], Counter[bool]] = defaultdict(Counter)
    for row in read_jsonl(matched_pairs_path):
        issue = row["issue_area_label"]
        if issue not in FRAME_PATTERN_SUBSET:
            continue
        split = row["split"]
        for side in ("a", "b"):
            text = str(row[f"text_{side}"]).lower()
            for frame, patterns in FRAME_PATTERN_SUBSET[issue].items():
                hit = any(pattern in text for pattern in patterns)
                counts[(issue, frame, split)][hit] += 1

    table_rows: list[list[Any]] = []
    for issue, frames in FRAME_PATTERN_SUBSET.items():
        for frame in frames:
            for split in ("train", "dev", "test"):
                counter = counts[(issue, frame, split)]
                table_rows.append([issue, frame, split, counter[True], counter[False]])
    return table_rows


def write_report(args: argparse.Namespace) -> None:
    last_manifest = read_json(args.last_token_dir / "manifest.json")
    all_manifest = read_json(args.all_position_dir / "manifest.json")
    proxy_manifest = read_json(args.proxy_dir / "manifest.json")
    candidate_rows = top_prompt_matched_rows(args.last_token_dir, "last", limit=8)
    candidate_rows.extend(top_prompt_matched_rows(args.all_position_dir, "all", limit=8))
    volatile_rows, stable_rows = null_volatility_tables(args.proxy_dir)
    off_domain_rows, off_domain_frame_rows = off_domain_tables(args.proxy_dir)
    overlap_rows = low_overlap_rows(args.overlap_dir)
    feasibility_rows = frame_label_feasibility_rows(args.matched_pairs)

    lines = [
        "# SCOTUS Phase 5 Evidence Decision",
        "",
        "## Decision",
        "",
        "The broad averaged L16 justice directions remain decodable but are not promoted as steerable judicial circuits. "
        "Both last-token and all-position BF16 hook pilots stayed within prompt-matched same-layer random controls.",
        "",
        "## Artifacts",
        "",
        markdown_table(
            ["Artifact", "Path", "Rows", "Position", "Alphas"],
            [
                [
                    "Last-token BF16 hook pilot",
                    str(args.last_token_dir),
                    count_jsonl(args.last_token_dir / "generations.jsonl"),
                    last_manifest.get("position", ""),
                    ", ".join(str(alpha) for alpha in last_manifest.get("alphas", [])),
                ],
                [
                    "All-position BF16 hook sanity",
                    str(args.all_position_dir),
                    count_jsonl(args.all_position_dir / "generations.jsonl"),
                    all_manifest.get("position", ""),
                    ", ".join(str(alpha) for alpha in all_manifest.get("alphas", [])),
                ],
                [
                    "Q4 proxy null generation",
                    str(args.proxy_dir),
                    count_jsonl(args.proxy_dir / "generations.jsonl"),
                    "none",
                    "none",
                ],
            ],
        ),
        "",
        "## Hook Pilot Readout",
        "",
        "Rows below are sorted by absolute prompt-matched z. The best all-position row is still small and does not replicate as alpha increases.",
        "",
        markdown_table(
            [
                "Pilot",
                "Candidate",
                "Alpha",
                "N",
                "Matched delta",
                "Random residual SD",
                "Z",
                "Percentile",
                "Prompt win rate",
            ],
            candidate_rows,
        ),
        "",
        "## Proxy Null Volatility",
        "",
        "Highest-variance prompt-condition rows are poor substrates for steering claims until the rubric is improved.",
        "",
        markdown_table(["Prompt", "Condition", "Issue", "Mean delta", "SD", "P05", "P95"], volatile_rows),
        "",
        "Most stable prompt-condition rows are better candidates for the next issue-specific pilot.",
        "",
        markdown_table(["Prompt", "Condition", "Issue", "Mean delta", "SD", "P05", "P95"], stable_rows),
        "",
        "## Rubric Contamination",
        "",
        "Off-domain frame tags identify places where keyword scoring is too coarse or the prompt naturally invokes neighboring doctrine.",
        "",
        markdown_table(["Prompt", "Condition", "Issue", "N", "Off-domain rate", "Mean off-domain hits"], off_domain_rows),
        "",
        markdown_table(["Off-domain frame", "Count"], off_domain_frame_rows),
        "",
        "## Issue-Specific Candidate Starts",
        "",
        "Low issue-conditioned SAE overlap rows are better next candidates than broad justice-level averages. "
        "They are not steering evidence by themselves; they nominate where to build narrower frame candidates.",
        "",
        markdown_table(
            ["Issue", "Pair", "Region", "Layer", "Top-J", "Weighted-J", "Cosine", "N"],
            overlap_rows,
        ),
        "",
        "## Frame Label Feasibility",
        "",
        "Direct keyword labels in the repaired opinion chunks are too sparse for several desired frame probes. "
        "This argues for curated frame-labeled excerpts or contrastive prompt capture before training frame-specific directions.",
        "",
        markdown_table(["Issue", "Frame", "Split", "Positive", "Negative"], feasibility_rows),
        "",
        "## Next Step",
        "",
        "Build issue-specific or frame-specific candidates for Article III and Fourth Amendment prompts, then require prompt-matched same-layer random controls before any larger generation run.",
        "",
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize SCOTUS Phase 5 steering evidence.")
    parser.add_argument("--last-token-dir", type=Path, default=DEFAULT_LAST_TOKEN_DIR)
    parser.add_argument("--all-position-dir", type=Path, default=DEFAULT_ALL_POSITION_DIR)
    parser.add_argument("--proxy-dir", type=Path, default=DEFAULT_PROXY_DIR)
    parser.add_argument("--overlap-dir", type=Path, default=DEFAULT_OVERLAP_DIR)
    parser.add_argument("--matched-pairs", type=Path, default=DEFAULT_MATCHED_PAIRS)
    parser.add_argument("--output", type=Path, default=DEFAULT_REPORT)
    return parser.parse_args()


def main() -> None:
    write_report(parse_args())


if __name__ == "__main__":
    main()
