#!/usr/bin/env python3
"""Index ignored local SCOTUS raw run directories without tracking their contents."""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
KEY_ARTIFACT_NAMES = (
    "report.md",
    "manifest.json",
    "features.npz",
    "direction.npz",
    "best_probe_direction.npz",
    "generations.jsonl",
    "summary.json",
    "top_sae_features.jsonl",
)


@dataclass(frozen=True)
class RunSummary:
    path: Path
    size_bytes: int
    file_count: int
    modified_at: float
    key_artifacts: tuple[str, ...]


def summarize_run(path: Path) -> RunSummary:
    size_bytes = 0
    file_count = 0
    latest_mtime = path.stat().st_mtime
    direct_names = {child.name for child in path.iterdir() if child.is_file()}
    key_artifacts = tuple(name for name in KEY_ARTIFACT_NAMES if name in direct_names)

    for root, _dirs, files in os.walk(path):
        for name in files:
            target = Path(root) / name
            try:
                stat = target.stat()
            except FileNotFoundError:
                continue
            file_count += 1
            size_bytes += stat.st_size
            latest_mtime = max(latest_mtime, stat.st_mtime)

    return RunSummary(
        path=path,
        size_bytes=size_bytes,
        file_count=file_count,
        modified_at=latest_mtime,
        key_artifacts=key_artifacts,
    )


def format_mib(size_bytes: int) -> str:
    return f"{size_bytes / (1024 * 1024):.2f}"


def format_modified(timestamp: float) -> str:
    return datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S")


def build_report(pattern: str) -> str:
    runs = [
        summarize_run(path)
        for path in PROJECT_ROOT.glob(pattern)
        if path.is_dir()
    ]
    runs.sort(key=lambda item: item.modified_at, reverse=True)
    latest = runs[0].path.relative_to(PROJECT_ROOT).as_posix() if runs else "none"

    lines = [
        "# SCOTUS Raw Run Archive Index",
        "",
        "Raw SCOTUS run outputs are intentionally ignored by Git and retained locally for provenance. This index records the local directories without tracking their large generated contents.",
        "",
        f"- Indexed directories: `{len(runs)}`",
        f"- Latest modified run: `{latest}`",
        "",
        "| Run directory | Size MiB | Files | Modified | Key artifacts | Status |",
        "| --- | ---: | ---: | --- | --- | --- |",
    ]

    for index, run in enumerate(runs):
        status = "latest" if index == 0 else "archive"
        rel_path = run.path.relative_to(PROJECT_ROOT).as_posix()
        lines.append(
            "| "
            f"`{rel_path}` | "
            f"{format_mib(run.size_bytes)} | "
            f"{run.file_count} | "
            f"{format_modified(run.modified_at)} | "
            f"{', '.join(run.key_artifacts)} | "
            f"{status} |"
        )

    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pattern", default="sweep_v4/scotus*")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    report = build_report(pattern=args.pattern)
    if args.output:
        output = args.output if args.output.is_absolute() else PROJECT_ROOT / args.output
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(report, encoding="utf-8")
    else:
        print(report, end="")


if __name__ == "__main__":
    main()
