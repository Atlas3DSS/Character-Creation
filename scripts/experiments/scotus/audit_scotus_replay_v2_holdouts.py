#!/usr/bin/env python3
"""Audit SCOTUS minimal-pair replay v2 probes under stricter group holdouts."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.pipeline import make_pipeline


@dataclass(frozen=True)
class HoldoutSummary:
    name: str
    holdout_key: str
    n_groups: int
    mean_ba: float
    median_ba: float
    min_ba: float
    max_ba: float
    mean_accuracy: float
    mean_f1: float


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def classifier(c: float) -> LogisticRegression:
    return LogisticRegression(
        C=c,
        class_weight="balanced",
        max_iter=2000,
        random_state=0,
        solver="liblinear",
    )


def summarize_group_predictions(
    name: str,
    holdout_key: str,
    y_true_groups: list[np.ndarray],
    y_pred_groups: list[np.ndarray],
) -> HoldoutSummary:
    bas: list[float] = []
    accuracies: list[float] = []
    f1s: list[float] = []
    for y_true, y_pred in zip(y_true_groups, y_pred_groups, strict=True):
        bas.append(float(balanced_accuracy_score(y_true, y_pred)))
        accuracies.append(float(accuracy_score(y_true, y_pred)))
        f1s.append(float(f1_score(y_true, y_pred, zero_division=0)))
    return HoldoutSummary(
        name=name,
        holdout_key=holdout_key,
        n_groups=len(bas),
        mean_ba=float(mean(bas)),
        median_ba=float(median(bas)),
        min_ba=float(min(bas)),
        max_ba=float(max(bas)),
        mean_accuracy=float(mean(accuracies)),
        mean_f1=float(mean(f1s)),
    )


def feature_holdout(
    *,
    name: str,
    x: np.ndarray,
    y: np.ndarray,
    groups: list[str],
    holdout_key: str,
    c: float,
) -> HoldoutSummary:
    unique_groups = sorted(set(groups))
    y_true_groups: list[np.ndarray] = []
    y_pred_groups: list[np.ndarray] = []

    for group in unique_groups:
        test_mask = np.array([value == group for value in groups], dtype=bool)
        train_mask = ~test_mask
        if len(set(y[train_mask])) < 2 or len(set(y[test_mask])) < 2:
            continue
        model = classifier(c)
        model.fit(x[train_mask], y[train_mask])
        y_true_groups.append(y[test_mask])
        y_pred_groups.append(model.predict(x[test_mask]))

    return summarize_group_predictions(name, holdout_key, y_true_groups, y_pred_groups)


def text_holdout(
    *,
    name: str,
    texts: list[str],
    y: np.ndarray,
    groups: list[str],
    holdout_key: str,
    c: float,
) -> HoldoutSummary:
    unique_groups = sorted(set(groups))
    y_true_groups: list[np.ndarray] = []
    y_pred_groups: list[np.ndarray] = []

    for group in unique_groups:
        test_mask = np.array([value == group for value in groups], dtype=bool)
        train_mask = ~test_mask
        if len(set(y[train_mask])) < 2 or len(set(y[test_mask])) < 2:
            continue
        train_texts = [text for text, keep in zip(texts, train_mask, strict=True) if keep]
        test_texts = [text for text, keep in zip(texts, test_mask, strict=True) if keep]
        model = make_pipeline(
            TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_features=50000),
            classifier(c),
        )
        model.fit(train_texts, y[train_mask])
        y_true_groups.append(y[test_mask])
        y_pred_groups.append(model.predict(test_texts))

    return summarize_group_predictions(name, holdout_key, y_true_groups, y_pred_groups)


def fmt(value: float) -> str:
    return f"{value:.3f}"


def write_report(path: Path, summaries: list[HoldoutSummary], features_dir: Path, meta_path: Path) -> None:
    lines = [
        "# SCOTUS Minimal-Pair Replay v2 Holdout Audit",
        "",
        "## Purpose",
        "",
        "Test whether the replay-v2 separability survives leave-one-style-variant and leave-one-fact-pattern holdouts, using already-captured activations. This is still decodability evidence, not causal steering evidence.",
        "",
        "## Inputs",
        "",
        f"- Features: `{features_dir / 'features.npz'}`",
        f"- Metadata: `{meta_path}`",
        "",
        "## Results",
        "",
        "| Readout | Holdout | Groups | Mean BA | Median BA | Min BA | Max BA | Mean accuracy | Mean F1 |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summaries:
        lines.append(
            "| "
            f"{row.name} | {row.holdout_key} | {row.n_groups} | "
            f"{fmt(row.mean_ba)} | {fmt(row.median_ba)} | {fmt(row.min_ba)} | {fmt(row.max_ba)} | "
            f"{fmt(row.mean_accuracy)} | {fmt(row.mean_f1)} |"
        )
    lines.extend(
        [
            "",
            "## Read",
            "",
            "- `prompt_text_tfidf` stays at chance under both group holdouts, as expected from paired prompts.",
            "- `assistant_text_tfidf` is perfect under fact holdout and strong, but not uniform, under variant holdout. The answer text itself carries an easily recoverable Commerce-authority versus Commerce-limits proposition.",
            "- Assistant-internal readouts are perfect under fact holdout and strong under variant holdout, with some style variants falling to chance. That makes replay-v2 useful as an answer-state candidate source, but it should not be promoted as a judicial circuit without a causal generation win against matched random and source-trace controls.",
            "- `prompt_last__L08` remains near chance, which lowers concern about prompt-format leakage but increases the interpretation that the separability appears only after the model is replaying the labeled answer.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--features-dir",
        type=Path,
        default=Path("sweep_v4/scotus_minpair_replay_v2_20260501_144942"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("reports/scotus_minpair_replay_v2_holdout_audit_20260501.md"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    features_dir = args.features_dir
    features = np.load(features_dir / "features.npz")
    meta_path = features_dir / "feature_meta.jsonl"
    meta = read_jsonl(meta_path)

    labels = np.array([int(row["label"]) for row in meta])
    variant_groups = [str(row["variant_id"]) for row in meta]
    fact_groups = [str(row["fact_id"]) for row in meta]
    prompts = [str(row["prompt"]) for row in meta]
    assistant_texts = [str(row["assistant_text"]) for row in meta]

    specs = [
        ("prompt_last__L08 C=1.0", "prompt_last__L08", 1.0),
        ("assistant_all__L08 C=0.001", "assistant_all__L08", 0.001),
        ("assistant_early__L08 C=0.001", "assistant_early__L08", 0.001),
        ("assistant_late__L08 C=0.001", "assistant_late__L08", 0.001),
        ("assistant_all__L16 C=0.001", "assistant_all__L16", 0.001),
        ("assistant_all__L24 C=0.001", "assistant_all__L24", 0.001),
    ]

    summaries: list[HoldoutSummary] = []
    for holdout_key, groups in [("variant_id", variant_groups), ("fact_id", fact_groups)]:
        summaries.append(
            text_holdout(
                name="prompt_text_tfidf C=1.0",
                texts=prompts,
                y=labels,
                groups=groups,
                holdout_key=holdout_key,
                c=1.0,
            )
        )
        summaries.append(
            text_holdout(
                name="assistant_text_tfidf C=1.0",
                texts=assistant_texts,
                y=labels,
                groups=groups,
                holdout_key=holdout_key,
                c=1.0,
            )
        )
        for name, key, c in specs:
            if key not in features:
                continue
            summaries.append(
                feature_holdout(
                    name=name,
                    x=features[key],
                    y=labels,
                    groups=groups,
                    holdout_key=holdout_key,
                    c=c,
                )
            )

    args.report.parent.mkdir(parents=True, exist_ok=True)
    write_report(args.report, summaries, features_dir, meta_path)
    print(f"Wrote {args.report}")


if __name__ == "__main__":
    main()
