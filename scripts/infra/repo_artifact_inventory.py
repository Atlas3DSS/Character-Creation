#!/usr/bin/env python3
"""Summarize repo artifact hygiene without opening large experiment outputs."""

from __future__ import annotations

import argparse
import collections
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class StatusItem:
    status: str
    path: str


def run_git(args: list[str]) -> bytes:
    result = subprocess.run(
        ["git", *args],
        cwd=PROJECT_ROOT,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return result.stdout


def parse_status(include_ignored: bool) -> list[StatusItem]:
    args = ["status", "--porcelain=v1", "-z", "--untracked-files=all"]
    if include_ignored:
        args.append("--ignored")
    raw = run_git(args)
    items: list[StatusItem] = []
    for record in raw.split(b"\0"):
        if not record:
            continue
        text = record.decode("utf-8", "replace")
        items.append(StatusItem(status=text[:2], path=text[3:]))
    return items


def file_size(path: str) -> int:
    target = PROJECT_ROOT / path
    if not target.is_file():
        return 0
    return target.stat().st_size


def bucket(path: str, status: str) -> str:
    if status == "!!":
        return "ignored_local"
    if path.startswith(("dev_genius/", ".venv/", "venv/")):
        return "local_env"
    if path.startswith(("logs/", "sweep_v2/", "sweep_v3/", "sweep_v4/", "results/")):
        return "raw_run_output"
    if path.startswith(("scripts/", ".agents/")) or path in {
        "AGENTS.md",
        "SCOTUS.md",
        "SCOTUS_Phase4.md",
        "activation_probes.md",
    }:
        return "track_source"
    if path.startswith("reports/"):
        suffix = Path(path).suffix
        if suffix == ".html":
            return "archive_generated_report"
        return "track_report"
    if path.startswith("data/scotus/raw/") or path.startswith("data/scotus/processed/"):
        return "archive_source_corpus"
    if path.startswith("data/"):
        return "track_data"
    return "needs_review"


def count_by(items: list[StatusItem], key_name: str) -> collections.Counter[str]:
    if key_name == "status":
        return collections.Counter(item.status for item in items)
    if key_name == "top":
        return collections.Counter(item.path.split("/", 1)[0] for item in items)
    if key_name == "ext":
        return collections.Counter(Path(item.path).suffix or "[none]" for item in items)
    if key_name == "bucket":
        return collections.Counter(bucket(item.path, item.status) for item in items)
    raise ValueError(f"Unknown counter {key_name}")


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def format_counter(counter: collections.Counter[str], limit: int) -> str:
    rows = [[str(key), str(value)] for key, value in counter.most_common(limit)]
    return markdown_table(["Key", "Count"], rows)


def format_sizes(paths: list[str], limit: int) -> str:
    rows: list[list[str]] = []
    for path in sorted(paths, key=file_size, reverse=True)[:limit]:
        size = file_size(path)
        rows.append([path, f"{size / (1024 * 1024):.2f} MiB"])
    return markdown_table(["Path", "Size"], rows)


def build_report(include_ignored: bool, top: int) -> str:
    items = parse_status(include_ignored=include_ignored)
    paths_by_bucket: dict[str, list[str]] = collections.defaultdict(list)
    for item in items:
        paths_by_bucket[bucket(item.path, item.status)].append(item.path)

    lines = [
        "# Repo Artifact Inventory",
        "",
        "This inventory is intentionally shallow: it uses Git status metadata and file sizes, not raw experiment contents.",
        "",
        f"- Include ignored files: `{include_ignored}`",
        f"- Status rows: `{len(items)}`",
        "",
        "## Buckets",
        "",
        format_counter(count_by(items, "bucket"), top),
        "",
        "## Git Status",
        "",
        format_counter(count_by(items, "status"), top),
        "",
        "## Top-Level Paths",
        "",
        format_counter(count_by(items, "top"), top),
        "",
        "## Extensions",
        "",
        format_counter(count_by(items, "ext"), top),
    ]

    for name in ["track_source", "track_report", "track_data", "archive_generated_report", "needs_review"]:
        paths = paths_by_bucket.get(name, [])
        if not paths:
            continue
        lines.extend(["", f"## {name}", "", format_sizes(paths, top)])

    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--include-ignored", action="store_true")
    parser.add_argument("--top", type=int, default=30)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    report = build_report(include_ignored=args.include_ignored, top=args.top)
    if args.output:
        output = args.output if args.output.is_absolute() else PROJECT_ROOT / args.output
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(report, encoding="utf-8")
    else:
        print(report, end="")


if __name__ == "__main__":
    main()
