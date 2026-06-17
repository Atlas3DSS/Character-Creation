#!/usr/bin/env python3
"""Audit the lightweight SCOTUS frame keyword metric.

This report is not a replacement for blind legal review. It surfaces where the
current keyword scorer is most likely to overstate frame movement, especially
through off-domain substring hits and already-saturated target frames.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PROXY_DIR = PROJECT_ROOT / "sweep_v4" / "scotus_qwen4bit_proxy_20260501_045257"
DEFAULT_ARTICLE3_DIR = PROJECT_ROOT / "sweep_v4" / "scotus_sae_poke_20260501_000146"
DEFAULT_FOURTH_DIR = PROJECT_ROOT / "sweep_v4" / "scotus_sae_poke_20260501_001257"
DEFAULT_OUTPUT = PROJECT_ROOT / "reports" / "scotus_frame_metric_audit_20260501.md"
SCORER_PATH = Path(__file__).with_name("qwen4bit_proxy_generation.py")


def load_frame_patterns() -> dict[str, list[str]]:
    spec = importlib.util.spec_from_file_location("qwen4bit_proxy_generation", SCORER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load scorer module from {SCORER_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return dict(module.FRAME_PATTERNS)


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


def ascii_clean(value: str) -> str:
    return value.encode("ascii", errors="replace").decode("ascii")


def snippet(text: str, limit: int = 260) -> str:
    clean = " ".join(ascii_clean(text).split())
    if len(clean) <= limit:
        return clean
    return clean[: limit - 3].rstrip() + "..."


def pattern_hits(text: str, frame: str, patterns: dict[str, list[str]]) -> list[str]:
    lowered = text.lower()
    hits: list[str] = []
    for pattern in patterns.get(frame, []):
        count = lowered.count(pattern)
        if count:
            hits.append(f"{pattern} x{count}")
    return hits


def fmt_num(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)


def markdown_table(headers: list[str], rows: list[list[Any]]) -> list[str]:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(fmt_num(item) for item in row) + " |")
    return lines


def summarize_probe_dir(label: str, path: Path) -> list[list[Any]]:
    rows = read_jsonl(path / "candidate_vs_prompt_matched_random.jsonl")
    out: list[list[Any]] = []
    for row in rows:
        out.append(
            [
                label,
                row["alpha"],
                row["n"],
                row["mean_prompt_matched_delta_minus_random"],
                row["z_vs_prompt_matched_random"],
                row["prompt_win_rate_vs_random_mean"],
                row.get("mean_prompt_matched_net_delta_minus_random", 0.0),
                row.get("z_net_vs_prompt_matched_random", 0.0),
                row.get("prompt_net_win_rate_vs_random_mean", 0.0),
            ]
        )
    return out


def baseline_saturation(label: str, path: Path) -> list[Any]:
    rows = read_jsonl(path / "generations.jsonl")
    base_rows = [row for row in rows if row.get("condition") == "base"]
    if not base_rows:
        return [label, 0, 0.0, 0.0, 0.0]
    target_present = sum(1 for row in base_rows if row["frame_eval"].get("target_present")) / len(base_rows)
    mean_target = sum(float(row["frame_eval"].get("target_hits", 0.0)) for row in base_rows) / len(base_rows)
    mean_contrast = sum(float(row["frame_eval"].get("contrast_hits", 0.0)) for row in base_rows) / len(base_rows)
    return [label, len(base_rows), target_present, mean_target, mean_contrast]


def build_off_domain_examples(
    rows: list[dict[str, Any]],
    patterns: dict[str, list[str]],
    limit: int,
) -> tuple[Counter[str], list[list[Any]]]:
    counter: Counter[str] = Counter()
    examples: list[tuple[int, dict[str, Any]]] = []
    for row in tqdm(rows, desc="audit proxy off-domain rows"):
        frame_eval = row.get("frame_eval", {})
        off_frames = frame_eval.get("off_domain_frames", [])
        if not off_frames:
            continue
        off_hits = int(frame_eval.get("off_domain_hits", 0))
        counter.update(str(frame) for frame in off_frames)
        examples.append((off_hits, row))
    examples.sort(key=lambda item: (item[0], item[1].get("prompt_key", "")), reverse=True)

    table_rows: list[list[Any]] = []
    for off_hits, row in examples[:limit]:
        text = str(row.get("text") or row.get("completion") or "")
        off_frames = [str(frame) for frame in row.get("frame_eval", {}).get("off_domain_frames", [])]
        triggers = []
        for frame in off_frames[:3]:
            hits = pattern_hits(text, frame, patterns)
            if hits:
                triggers.append(f"{frame}: {', '.join(hits[:3])}")
        table_rows.append(
            [
                row.get("prompt_key", ""),
                row.get("condition", ""),
                row.get("sample_type", ""),
                off_hits,
                ", ".join(off_frames),
                "; ".join(triggers),
                snippet(text),
            ]
        )
    return counter, table_rows


def build_blind_examples(
    blind_rows: list[dict[str, Any]],
    patterns: dict[str, list[str]],
    limit: int,
) -> list[list[Any]]:
    examples: list[tuple[int, dict[str, Any]]] = []
    for row in tqdm(blind_rows, desc="audit blind sample rows"):
        issue_area = str(row.get("issue_area", ""))
        off_score = 0
        scores = row.get("frame_scores", {})
        for frame in scores:
            if issue_area == "Judicial Power" and not frame.startswith("article3_"):
                off_score += int(scores[frame])
            elif issue_area == "Criminal Procedure" and not frame.startswith("fourth_"):
                off_score += int(scores[frame])
            elif issue_area == "Economic Activity" and not (
                frame.startswith("economic_") or frame.startswith("federalism_")
            ):
                off_score += int(scores[frame])
            elif issue_area == "Civil Rights" and not frame.startswith("civil_"):
                off_score += int(scores[frame])
            elif issue_area == "Due Process" and not (
                frame.startswith("due_process_") or frame.startswith("civil_")
            ):
                off_score += int(scores[frame])
            elif issue_area == "Federalism" and not (
                frame.startswith("federalism_") or frame.startswith("economic_")
            ):
                off_score += int(scores[frame])
            elif issue_area == "Administrative Law" and not (
                frame.startswith("admin_") or frame.startswith("economic_") or frame.startswith("separation_")
            ):
                off_score += int(scores[frame])
        if off_score:
            examples.append((off_score, row))
    examples.sort(key=lambda item: item[0], reverse=True)

    table_rows: list[list[Any]] = []
    for off_score, row in examples[:limit]:
        text = str(row.get("completion") or "")
        triggers = []
        for frame in row.get("frame_scores", {}):
            hits = pattern_hits(text, str(frame), patterns)
            if hits:
                triggers.append(f"{frame}: {', '.join(hits[:2])}")
        table_rows.append(
            [
                row.get("blind_id", ""),
                row.get("prompt_key", ""),
                row.get("issue_area", ""),
                off_score,
                json.dumps(row.get("frame_scores", {}), sort_keys=True),
                "; ".join(triggers[:4]),
                snippet(text),
            ]
        )
    return table_rows


def write_report(args: argparse.Namespace) -> None:
    patterns = load_frame_patterns()
    proxy_rows = read_jsonl(args.proxy_dir / "generations.jsonl")
    blind_rows = read_jsonl(args.proxy_dir / "blind_review_sample.jsonl")

    off_counter, off_examples = build_off_domain_examples(proxy_rows, patterns, args.example_limit)
    blind_examples = build_blind_examples(blind_rows, patterns, args.example_limit)
    probe_rows = summarize_probe_dir("Article III", args.article3_dir)
    probe_rows.extend(summarize_probe_dir("Fourth Amendment", args.fourth_dir))
    saturation_rows = [
        baseline_saturation("Article III", args.article3_dir),
        baseline_saturation("Fourth Amendment", args.fourth_dir),
    ]

    lines: list[str] = [
        "# SCOTUS Frame Metric Audit",
        "",
        "## Purpose",
        "",
        "This audits the lightweight keyword frame metric used for the Q4 proxy and BF16 hook-generation pilots. It is a triage report, not a final evaluator.",
        "",
        "## Main Read",
        "",
        "1. The metric is useful for fast gates, but it is not yet strong enough to support a steering claim by itself.",
        "2. Off-domain hits often come from broad or polysemous substrings such as `home`, `consent`, `district`, `damages`, `remedy`, and generic separation-of-powers wording.",
        "3. Frame-pilot baselines are already saturated on several prompts, so a small target-hit increase can mean repeated vocabulary rather than a new legal frame.",
        "4. The next evaluator pass should score frame presence as a legal proposition, not raw keyword repetition.",
        "",
        "## BF16 Frame-Pilot Gate Results",
        "",
    ]
    lines.extend(
        markdown_table(
            [
                "Pilot",
                "Alpha",
                "N",
                "Matched target delta",
                "Target z",
                "Target win rate",
                "Matched net delta",
                "Net z",
                "Net win rate",
            ],
            probe_rows,
        )
    )
    lines.extend(
        [
            "",
            "## Baseline Saturation",
            "",
            "High baseline target presence makes target-hit deltas hard to interpret; the scorer is often counting repetition inside an already-correct frame.",
            "",
        ]
    )
    lines.extend(
        markdown_table(
            ["Pilot", "Base rows", "Target present rate", "Mean target hits", "Mean contrast hits"],
            saturation_rows,
        )
    )
    lines.extend(
        [
            "",
            "## Proxy Off-Domain Frame Counts",
            "",
        ]
    )
    lines.extend(markdown_table(["Frame", "Rows"], [[frame, count] for frame, count in off_counter.most_common(12)]))
    lines.extend(
        [
            "",
            "## Highest Off-Domain Proxy Examples",
            "",
        ]
    )
    lines.extend(
        markdown_table(
            ["Prompt", "Condition", "Sample", "Off hits", "Off frames", "Likely triggers", "Snippet"],
            off_examples,
        )
    )
    lines.extend(
        [
            "",
            "## Blind-Sample Keyword Warnings",
            "",
            "These examples should be manually reviewed before using the blind sample as an evaluator calibration set.",
            "",
        ]
    )
    lines.extend(
        markdown_table(
            ["Blind ID", "Prompt", "Issue", "Off score", "Frame scores", "Likely triggers", "Snippet"],
            blind_examples,
        )
    )
    lines.extend(
        [
            "",
            "## Evaluator Repair Checklist",
            "",
            "1. Replace raw substring counts with boolean proposition-level frame labels per completion.",
            "2. Split ambiguous patterns: `home` should not trigger home-exigency by itself, and `consent` should not trigger Fourth Amendment consent when used in ordinary language.",
            "3. Add negation and role checks for contrast frames, especially where a completion rejects search-incident doctrine while still naming it.",
            "4. Score target-minus-contrast as the primary automatic metric, but require a blind-review sample before promoting any direction.",
            "5. For source-grounded frame data, label short opinion excerpts by doctrinal proposition rather than by justice or raw keyword.",
            "",
        ]
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {args.output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proxy-dir", type=Path, default=DEFAULT_PROXY_DIR)
    parser.add_argument("--article3-dir", type=Path, default=DEFAULT_ARTICLE3_DIR)
    parser.add_argument("--fourth-dir", type=Path, default=DEFAULT_FOURTH_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--example-limit", type=int, default=12)
    return parser.parse_args()


def main() -> None:
    write_report(parse_args())


if __name__ == "__main__":
    main()
