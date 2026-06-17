#!/usr/bin/env python3
"""Audit whether a cached SCOTUS probe run can support stricter resplits."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RUN = PROJECT_ROOT / "sweep_v4" / "scotus_slice_bf16_majority2000s_normal_20260501_022109"
DEFAULT_REPORT = PROJECT_ROOT / "reports" / "scotus_slice_split_feasibility_20260501.md"
DEFAULT_JSON = PROJECT_ROOT / "reports" / "scotus_slice_split_feasibility_20260501.json"


@dataclass(frozen=True)
class Component:
    component_id: str
    pair_ids: tuple[str, ...]
    case_ids: tuple[str, ...]
    row_indices: tuple[int, ...]


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


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return "\n".join(lines)


def connected_components(rows: list[dict[str, Any]]) -> list[Component]:
    pair_to_rows: dict[str, list[int]] = defaultdict(list)
    pair_to_cases: dict[str, set[str]] = defaultdict(set)
    case_to_pairs: dict[str, set[str]] = defaultdict(set)
    for idx, row in enumerate(rows):
        pair_id = str(row["pair_id"])
        case_id = str(row["case_id"])
        pair_to_rows[pair_id].append(idx)
        pair_to_cases[pair_id].add(case_id)
        case_to_pairs[case_id].add(pair_id)

    seen: set[str] = set()
    components: list[Component] = []
    for pair_id in sorted(pair_to_rows):
        if pair_id in seen:
            continue
        queue: deque[str] = deque([pair_id])
        seen.add(pair_id)
        component_pairs: set[str] = set()
        component_cases: set[str] = set()
        component_rows: list[int] = []
        while queue:
            cur = queue.popleft()
            component_pairs.add(cur)
            component_rows.extend(pair_to_rows[cur])
            for case_id in pair_to_cases[cur]:
                component_cases.add(case_id)
                for next_pair in case_to_pairs[case_id]:
                    if next_pair not in seen:
                        seen.add(next_pair)
                        queue.append(next_pair)
        component_id = f"component_{len(components):03d}"
        components.append(
            Component(
                component_id=component_id,
                pair_ids=tuple(sorted(component_pairs)),
                case_ids=tuple(sorted(component_cases)),
                row_indices=tuple(sorted(component_rows)),
            )
        )
    return components


def counter_to_payload(counter: Counter[Any]) -> dict[str, int]:
    return {str(key): int(value) for key, value in sorted(counter.items(), key=lambda item: str(item[0]))}


def summarize_components(rows: list[dict[str, Any]], components: list[Component]) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for component in components:
        component_rows = [rows[idx] for idx in component.row_indices]
        issue_counts = Counter(str(row.get("issue_area_label") or "unknown") for row in component_rows)
        split_counts = Counter(str(row.get("split") or "unknown") for row in component_rows)
        label_counts = Counter(int(row["label"]) for row in component_rows)
        justice_counts = Counter(str(row.get("justice") or "unknown") for row in component_rows)
        issue = issue_counts.most_common(1)[0][0]
        summaries.append(
            {
                "component_id": component.component_id,
                "issue_area_label": issue,
                "n_rows": len(component.row_indices),
                "n_pairs": len(component.pair_ids),
                "n_cases": len(component.case_ids),
                "case_ids": list(component.case_ids),
                "label_counts": counter_to_payload(label_counts),
                "justice_counts": counter_to_payload(justice_counts),
                "original_split_counts": counter_to_payload(split_counts),
                "issue_counts": counter_to_payload(issue_counts),
            }
        )
    summaries.sort(key=lambda row: (str(row["issue_area_label"]), -int(row["n_rows"]), str(row["component_id"])))
    return summaries


def feasibility_by_issue(component_summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for summary in component_summaries:
        grouped[str(summary["issue_area_label"])].append(summary)

    rows: list[dict[str, Any]] = []
    for issue, components in sorted(grouped.items()):
        n_components = len(components)
        n_rows = sum(int(component["n_rows"]) for component in components)
        n_pairs = sum(int(component["n_pairs"]) for component in components)
        n_cases = len({case_id for component in components for case_id in component["case_ids"]})
        rows.append(
            {
                "issue_area_label": issue,
                "n_components": n_components,
                "n_rows": n_rows,
                "n_pairs": n_pairs,
                "n_cases": n_cases,
                "strict_train_dev_test_feasible": n_components >= 3,
                "blocking_reason": ""
                if n_components >= 3
                else "needs at least 3 case-connected components for train/dev/test",
            }
        )
    return rows


def original_split_table(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    counts = Counter(
        (
            str(row.get("issue_area_label") or "unknown"),
            str(row.get("split") or "unknown"),
            str(row.get("justice") or "unknown"),
        )
        for row in rows
    )
    return [
        {"issue_area_label": issue, "split": split, "justice": justice, "n_rows": count}
        for (issue, split, justice), count in sorted(counts.items())
    ]


def write_report(path: Path, *, payload: dict[str, Any]) -> None:
    feasibility_rows = [
        [
            row["issue_area_label"],
            row["n_components"],
            row["n_rows"],
            row["n_pairs"],
            row["n_cases"],
            "yes" if row["strict_train_dev_test_feasible"] else "no",
            row["blocking_reason"],
        ]
        for row in payload["issue_feasibility"]
    ]
    component_rows = [
        [
            row["component_id"],
            row["issue_area_label"],
            row["n_rows"],
            row["n_pairs"],
            row["n_cases"],
            row["label_counts"],
            row["original_split_counts"],
            ",".join(row["case_ids"][:6]) + ("..." if len(row["case_ids"]) > 6 else ""),
        ]
        for row in payload["components"]
    ]
    split_rows = [
        [row["issue_area_label"], row["split"], row["justice"], row["n_rows"]]
        for row in payload["original_split_table"]
    ]
    feasible_issues = [row for row in payload["issue_feasibility"] if row["strict_train_dev_test_feasible"]]
    all_issues_feasible = len(feasible_issues) == len(payload["issue_feasibility"])
    lines = [
        "# SCOTUS Slice Split Feasibility Audit",
        "",
        "## Purpose",
        "",
        "This audits whether a cached justice-style SCOTUS probe run can be resplit while preserving strict case-connected holdout and issue-family coverage.",
        "",
        "## Inputs",
        "",
        f"- Run directory: `{payload['run_dir']}`",
        f"- Rows: `{payload['n_rows']}`",
        f"- Case-connected components: `{payload['n_components']}`",
        "",
        "## Feasibility Read",
        "",
        markdown_table(
            ["Issue", "Components", "Rows", "Pairs", "Cases", "Strict train/dev/test feasible", "Blocking reason"],
            feasibility_rows,
        ),
        "",
        "Strict issue-stratified train/dev/test resplitting is "
        + ("feasible for every issue." if all_issues_feasible else "not feasible for this slice."),
        "",
        "## Component Table",
        "",
        markdown_table(
            ["Component", "Issue", "Rows", "Pairs", "Cases", "Label counts", "Original splits", "Case IDs"],
            component_rows,
        ),
        "",
        "## Original Split Counts",
        "",
        markdown_table(["Issue", "Split", "Justice", "Rows"], split_rows),
        "",
        "## Decision",
        "",
    ]
    if all_issues_feasible:
        lines.append("This run can support a stricter case-connected issue-stratified resplit.")
    else:
        lines.extend(
            [
                "Do not treat another resplit of this slice as a clean rescue path unless the source corpus is expanded.",
                "Several issue families have fewer than three case-connected components, so strict train/dev/test issue coverage would require splitting shared-case components or dropping issue families.",
            ]
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--output", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--json-output", type=Path, default=DEFAULT_JSON)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = read_jsonl(args.run_dir / "feature_meta.jsonl")
    components = connected_components(rows)
    component_summaries = summarize_components(rows, components)
    payload = {
        "run_dir": str(args.run_dir),
        "n_rows": len(rows),
        "n_components": len(components),
        "components": component_summaries,
        "issue_feasibility": feasibility_by_issue(component_summaries),
        "original_split_table": original_split_table(rows),
    }
    write_json(args.json_output, payload)
    write_report(args.output, payload=payload)
    print(f"Wrote {args.output}")
    print(f"Wrote {args.json_output}")


if __name__ == "__main__":
    main()
