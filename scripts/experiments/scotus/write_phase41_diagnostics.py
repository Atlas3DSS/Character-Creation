#!/usr/bin/env python3
"""Summarize SCOTUS Phase 4.1 diagnostic probe runs."""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_REPORT = PROJECT_ROOT / "reports" / "scotus_phase41_diagnostics.md"
CURRENT_REPORT = PROJECT_ROOT / "reports" / "scotus_phase41_diagnostics_current.md"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return "\n".join(lines)


def score(payload: dict[str, Any], split: str = "test") -> float:
    return float(payload.get("split_metrics", {}).get(split, {}).get("balanced_accuracy", 0.0))


def best_non_prompt_candidate(run_dir: Path) -> dict[str, Any] | None:
    rows = read_jsonl(run_dir / "layer_region_search.jsonl")
    candidates = [
        row
        for row in rows
        if row.get("region") != "prompt_last"
        and row.get("dev_metrics", {}).get("balanced_accuracy", 0.0) >= 0.75
        and row.get("test_metrics_diagnostic", {}).get("balanced_accuracy", 0.0) >= 0.75
    ]
    if not candidates:
        return None
    candidates.sort(
        key=lambda row: (
            row["test_metrics_diagnostic"]["balanced_accuracy"],
            row["dev_metrics"]["balanced_accuracy"],
            row["region"],
            row["layer"],
        ),
        reverse=True,
    )
    return candidates[0]


def non_prompt_rows(run_dir: Path) -> list[dict[str, Any]]:
    return [row for row in read_jsonl(run_dir / "layer_region_search.jsonl") if row.get("region") != "prompt_last"]


def robust_non_prompt_candidates(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Find exact non-prompt configs that clear thresholds across all real prompt modes."""
    required_modes = {"normal", "template_variant", "plain_prompt"}
    by_mode = {
        str(run["manifest"].get("diagnostic_mode")): run
        for run in runs
        if run["manifest"].get("diagnostic_mode") in required_modes
    }
    if set(by_mode) != required_modes:
        return []

    rows_by_key: dict[tuple[Any, ...], dict[str, Any]] = {}
    for mode, run in by_mode.items():
        for row in non_prompt_rows(run["dir"]):
            dev_ba = float(row.get("dev_metrics", {}).get("balanced_accuracy", 0.0))
            test_ba = float(row.get("test_metrics_diagnostic", {}).get("balanced_accuracy", 0.0))
            if dev_ba < 0.75 or test_ba < 0.75:
                continue
            key = (row.get("region"), row.get("layer"), row.get("C"))
            item = rows_by_key.setdefault(
                key,
                {
                    "region": row.get("region"),
                    "layer": row.get("layer"),
                    "C": row.get("C"),
                    "modes": {},
                },
            )
            item["modes"][mode] = {"dev_ba": dev_ba, "test_ba": test_ba}

    candidates = [item for item in rows_by_key.values() if set(item["modes"]) == required_modes]
    for item in candidates:
        item["min_dev_ba"] = min(mode_row["dev_ba"] for mode_row in item["modes"].values())
        item["min_test_ba"] = min(mode_row["test_ba"] for mode_row in item["modes"].values())
        item["mean_test_ba"] = sum(mode_row["test_ba"] for mode_row in item["modes"].values()) / len(item["modes"])
    candidates.sort(
        key=lambda item: (
            item["min_test_ba"],
            item["min_dev_ba"],
            item["mean_test_ba"],
            item["region"],
            item["layer"],
            -float(item["C"]),
        ),
        reverse=True,
    )
    return candidates


def load_runs(paths: list[Path]) -> list[dict[str, Any]]:
    runs = []
    for path in paths:
        manifest_path = path / "manifest.json"
        if not manifest_path.exists():
            continue
        manifest = read_json(manifest_path)
        summary_path = path / "summary.json"
        summary = read_json(summary_path) if summary_path.exists() else {}
        runs.append({"dir": path, "manifest": manifest, "summary": summary})
    runs.sort(key=lambda row: (row["manifest"].get("diagnostic_mode", ""), str(row["dir"])))
    return runs


def write_report(path: Path, runs: list[dict[str, Any]]) -> None:
    normal_runs = [run for run in runs if run["manifest"].get("diagnostic_mode") == "normal"]
    null_runs = [
        run
        for run in runs
        if run["manifest"].get("diagnostic_mode") in {"excerpt_removed", "neutral_filler", "label_shuffle"}
    ]
    variant_runs = [
        run
        for run in runs
        if run["manifest"].get("diagnostic_mode") in {"template_variant", "plain_prompt"}
    ]

    run_rows = []
    for run in runs:
        manifest = run["manifest"]
        best = manifest.get("best_probe", {})
        test = manifest.get("split_metrics", {}).get("test", {})
        dev = manifest.get("split_metrics", {}).get("dev", {})
        ci = manifest.get("test_balanced_accuracy_ci_95", {})
        text_baseline = run.get("summary", {}).get("rendered_prompt_tfidf_baseline") or manifest.get(
            "rendered_prompt_tfidf_baseline",
            {},
        )
        text = text_baseline.get("test", {})
        run_rows.append(
            [
                manifest.get("diagnostic_mode", "unknown"),
                manifest.get("prompt_template", "unknown"),
                best.get("region", ""),
                best.get("layer", ""),
                best.get("C", ""),
                f"{dev.get('balanced_accuracy', 0.0):.3f}",
                f"{test.get('balanced_accuracy', 0.0):.3f}",
                f"{ci.get('low', 0.0):.3f}-{ci.get('high', 0.0):.3f}",
                f"{text.get('balanced_accuracy', 0.0):.3f}" if text else "",
                str(run["dir"].relative_to(PROJECT_ROOT)),
            ]
        )

    null_ok = all(score(run["manifest"]) <= 0.60 for run in null_runs) and len(null_runs) >= 3
    required_variant_modes = {"template_variant", "plain_prompt"}
    variant_scores = {
        str(run["manifest"].get("diagnostic_mode")): score(run["manifest"])
        for run in variant_runs
    }
    variant_ok = all(variant_scores.get(mode, 0.0) >= 0.70 for mode in required_variant_modes)
    normal = max(normal_runs, key=lambda run: score(run["manifest"]), default=None)
    diagnostic_non_prompt = best_non_prompt_candidate(normal["dir"]) if normal else None
    robust_non_prompt = robust_non_prompt_candidates(runs)

    candidate_rows = []
    if normal:
        best = normal["manifest"].get("best_probe", {})
        candidate_rows.append(
            [
                f"{best.get('region')} @ L{best.get('layer')}",
                "diagnostic_only",
                "Best decoder, but prompt_last is leakage-sensitive unless strict prompt-ablation diagnostics pass.",
            ]
        )
    if robust_non_prompt:
        top = robust_non_prompt[0]
        region = top["region"]
        layer = top["layer"]
        c_value = top["C"]
        cls = "candidate_direction" if null_ok and variant_ok else "diagnostic_only"
        candidate_rows.append(
            [
                f"{region} @ L{layer}, C={c_value}",
                cls,
                (
                    f"Exact non-prompt config clears dev/test >= 0.75 in normal, template_variant, "
                    f"and plain_prompt; worst dev BA {top['min_dev_ba']:.3f}, worst test BA {top['min_test_ba']:.3f}."
                ),
            ]
        )
    elif diagnostic_non_prompt:
        region = diagnostic_non_prompt["region"]
        layer = diagnostic_non_prompt["layer"]
        dev_ba = diagnostic_non_prompt["dev_metrics"]["balanced_accuracy"]
        test_ba = diagnostic_non_prompt["test_metrics_diagnostic"]["balanced_accuracy"]
        candidate_rows.append(
            [
                f"{region} @ L{layer}",
                "diagnostic_only",
                f"Non-prompt readout clears only in the normal run: dev BA {dev_ba:.3f}, diagnostic test BA {test_ba:.3f}.",
            ]
        )
    else:
        candidate_rows.append(
            [
                "non-prompt candidates",
                "reject",
                "No prompt_mean or excerpt_mean configuration cleared both dev and diagnostic test >= 0.75.",
            ]
        )

    robust_rows = [
        [
            f"{item['region']} @ L{item['layer']}",
            item["C"],
            f"{item['min_dev_ba']:.3f}",
            f"{item['min_test_ba']:.3f}",
            f"{item['modes']['normal']['dev_ba']:.3f}/{item['modes']['normal']['test_ba']:.3f}",
            f"{item['modes']['template_variant']['dev_ba']:.3f}/{item['modes']['template_variant']['test_ba']:.3f}",
            f"{item['modes']['plain_prompt']['dev_ba']:.3f}/{item['modes']['plain_prompt']['test_ba']:.3f}",
        ]
        for item in robust_non_prompt[:10]
    ]

    phase5_gate = bool(null_ok and variant_ok and robust_non_prompt)
    lines = [
        "# SCOTUS Phase 4.1 Diagnostics",
        "",
        "Phase 4.1 checks whether the Scalia/Ginsburg activation signal is robust enough to promote a direction to a small Phase 5 causal pilot.",
        "",
        "## Run Summary",
        "",
        markdown_table(
            [
                "Mode",
                "Template",
                "Best region",
                "Layer",
                "C",
                "Dev BA",
                "Test BA",
                "Test BA CI",
                "Prompt TF-IDF Test BA",
                "Run",
            ],
            run_rows,
        ),
        "",
        "## Candidate Classification",
        "",
        markdown_table(["Direction", "Class", "Reason"], candidate_rows),
        "",
        "## Robust Non-Prompt Candidates",
        "",
        "Rows here are exact `(region, layer, C)` configurations that clear dev and diagnostic-test balanced accuracy `>= 0.75` in all three real prompt modes. Mode cells are `dev/test`.",
        "",
        markdown_table(
            [
                "Direction",
                "C",
                "Worst dev BA",
                "Worst test BA",
                "Normal",
                "Template variant",
                "Plain prompt",
            ],
            robust_rows or [["none", "", "", "", "", "", ""]],
        ),
        "",
        "## Gate Decision",
        "",
        markdown_table(
            ["Criterion", "Status"],
            [
                ["Null tests at chance", "pass" if null_ok else "fail"],
                ["Prompt/template robustness", "pass" if variant_ok else "fail"],
                ["Robust non-prompt candidate clears >=0.75", "pass" if robust_non_prompt else "fail"],
                ["Phase 5 causal pilot gate", "pass" if phase5_gate else "fail"],
            ],
        ),
        "",
        "Gate note: a pass here authorizes only a small causal pilot with random and wrong-pair controls. It is not evidence that a steerable judicial circuit has been found.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write SCOTUS Phase 4.1 diagnostics report.")
    parser.add_argument("--probe-dir", action="append", type=Path, default=[])
    parser.add_argument("--glob", default=str(PROJECT_ROOT / "sweep_v4" / "scotus_phase41_*"))
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--current-report", type=Path, default=CURRENT_REPORT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = args.probe_dir or [Path(path) for path in glob.glob(args.glob)]
    runs = load_runs(paths)
    if not runs:
        raise RuntimeError("No Phase 4.1 probe runs found")
    write_report(args.report, runs)
    print(f"Wrote {args.report}")
    if args.current_report and args.current_report != args.report:
        write_report(args.current_report, runs)
        print(f"Wrote {args.current_report}")


if __name__ == "__main__":
    main()
