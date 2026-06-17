#!/usr/bin/env python3
"""Audit whether cached activation probes add wins beyond text baselines."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import balanced_accuracy_score

from probe_scotus_style import build_prompt_content, make_text_classifier, markdown_table


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RESPLIT_DIR = (
    PROJECT_ROOT
    / "sweep_v4"
    / "scotus_slice_bf16_majority2000s_feasible_issues_normal_component_resplits_20260501_034116"
)
DEFAULT_REPORT = PROJECT_ROOT / "reports" / "scotus_text_activation_complementarity_20260501.md"


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


def split_rows(rows: list[dict[str, Any]], split: str) -> list[dict[str, Any]]:
    return [row for row in rows if row.get("split") == split]


def fit_text_predictions(
    rows: list[dict[str, Any]],
    *,
    template_variant: str,
    split: str,
) -> dict[str, dict[str, Any]]:
    train_rows = split_rows(rows, "train")
    eval_rows = split_rows(rows, split)
    if split == "test":
        fit_rows = train_rows + split_rows(rows, "dev")
    elif split == "dev":
        fit_rows = train_rows
    else:
        fit_rows = train_rows
    if not fit_rows or not eval_rows:
        raise ValueError(f"Missing fit/eval rows for split={split}")

    fit_texts = [build_prompt_content(str(row["text"]), template_variant) for row in fit_rows]
    fit_labels = np.array([int(row["label"]) for row in fit_rows], dtype=np.int64)
    eval_texts = [build_prompt_content(str(row["text"]), template_variant) for row in eval_rows]
    clf = make_text_classifier(1.0)
    clf.fit(fit_texts, fit_labels)
    probs = clf.predict_proba(eval_texts)[:, 1]
    preds = (probs >= 0.5).astype(np.int64)

    return {
        str(row["example_id"]): {
            "label": int(row["label"]),
            "pred": int(pred),
            "prob_positive": float(prob),
        }
        for row, pred, prob in zip(eval_rows, preds.tolist(), probs.tolist(), strict=True)
    }


def load_activation_predictions(split_dir: Path, split: str) -> dict[str, dict[str, Any]]:
    path = split_dir / f"{split}_predictions.jsonl"
    return {
        str(row["example_id"]): {
            "label": int(row["label"]),
            "pred": int(row["pred"]),
            "prob_positive": float(row["prob_positive"]),
        }
        for row in read_jsonl(path)
    }


def metric_row(
    plan_id: str,
    split: str,
    rows: list[dict[str, Any]],
    text_preds: dict[str, dict[str, Any]],
    activation_preds: dict[str, dict[str, Any]],
) -> tuple[list[Any], dict[str, Any]]:
    eval_rows = split_rows(rows, split)
    labels: list[int] = []
    text_values: list[int] = []
    activation_values: list[int] = []
    both_correct = 0
    text_only = 0
    activation_only = 0
    both_wrong = 0
    activation_only_issues: Counter[str] = Counter()
    text_only_issues: Counter[str] = Counter()
    text_wrong_activation_correct = 0
    text_wrong_total = 0
    text_uncertain_activation_correct = 0
    text_uncertain_total = 0

    for row in eval_rows:
        example_id = str(row["example_id"])
        if example_id not in text_preds or example_id not in activation_preds:
            raise KeyError(f"Missing prediction for {plan_id}/{split}/{example_id}")
        label = int(row["label"])
        text_pred = int(text_preds[example_id]["pred"])
        activation_pred = int(activation_preds[example_id]["pred"])
        text_prob = float(text_preds[example_id]["prob_positive"])
        text_correct = text_pred == label
        activation_correct = activation_pred == label
        labels.append(label)
        text_values.append(text_pred)
        activation_values.append(activation_pred)
        if text_correct and activation_correct:
            both_correct += 1
        elif text_correct and not activation_correct:
            text_only += 1
            text_only_issues[str(row.get("issue_area_label") or "unknown")] += 1
        elif activation_correct and not text_correct:
            activation_only += 1
            activation_only_issues[str(row.get("issue_area_label") or "unknown")] += 1
        else:
            both_wrong += 1
        if not text_correct:
            text_wrong_total += 1
            if activation_correct:
                text_wrong_activation_correct += 1
        if abs(text_prob - 0.5) <= 0.15:
            text_uncertain_total += 1
            if activation_correct:
                text_uncertain_activation_correct += 1

    text_ba = balanced_accuracy_score(labels, text_values)
    activation_ba = balanced_accuracy_score(labels, activation_values)
    payload = {
        "plan": plan_id,
        "split": split,
        "n": len(eval_rows),
        "text_ba": text_ba,
        "activation_ba": activation_ba,
        "delta_activation_minus_text": activation_ba - text_ba,
        "both_correct": both_correct,
        "activation_only": activation_only,
        "text_only": text_only,
        "both_wrong": both_wrong,
        "text_wrong_activation_accuracy": (
            text_wrong_activation_correct / text_wrong_total if text_wrong_total else None
        ),
        "text_uncertain_activation_accuracy": (
            text_uncertain_activation_correct / text_uncertain_total if text_uncertain_total else None
        ),
        "activation_only_issues": dict(activation_only_issues),
        "text_only_issues": dict(text_only_issues),
    }
    row = [
        plan_id,
        split,
        len(eval_rows),
        f"{text_ba:.3f}",
        f"{activation_ba:.3f}",
        f"{activation_ba - text_ba:+.3f}",
        both_correct,
        activation_only,
        text_only,
        both_wrong,
        (
            f"{payload['text_wrong_activation_accuracy']:.3f}"
            if payload["text_wrong_activation_accuracy"] is not None
            else "n/a"
        ),
        (
            f"{payload['text_uncertain_activation_accuracy']:.3f}"
            if payload["text_uncertain_activation_accuracy"] is not None
            else "n/a"
        ),
    ]
    return row, payload


def write_report(path: Path, *, resplit_dir: Path, rows: list[list[Any]], payloads: list[dict[str, Any]]) -> None:
    deltas = [float(item["delta_activation_minus_text"]) for item in payloads if item["split"] == "test"]
    activation_only = sum(int(item["activation_only"]) for item in payloads if item["split"] == "test")
    text_only = sum(int(item["text_only"]) for item in payloads if item["split"] == "test")
    issue_activation_only: Counter[str] = Counter()
    issue_text_only: Counter[str] = Counter()
    for item in payloads:
        if item["split"] != "test":
            continue
        issue_activation_only.update(item["activation_only_issues"])
        issue_text_only.update(item["text_only_issues"])

    lines = [
        "# SCOTUS Text/Activation Complementarity Audit",
        "",
        "## Purpose",
        "",
        "This checks whether selected cached activation probes add unique held-out wins over the rendered-prompt TF-IDF baseline. It does not create a new steering direction; it is a promotion-risk audit for the current majority-2000s feasible-issues branch.",
        "",
        "## Input",
        "",
        f"- Resplit directory: `{resplit_dir}`",
        "",
        "## Split Results",
        "",
        markdown_table(
            [
                "Plan",
                "Split",
                "N",
                "Text BA",
                "Activation BA",
                "Delta",
                "Both correct",
                "Activation only",
                "Text only",
                "Both wrong",
                "Activation acc on text-wrong",
                "Activation acc on text-uncertain",
            ],
            rows,
        ),
        "",
        "## Aggregate Test Read",
        "",
        f"- Median activation-minus-text test BA delta: `{float(np.median(deltas)):.3f}`",
        f"- Test discordant wins across plans: activation-only `{activation_only}`, text-only `{text_only}`",
        f"- Activation-only issues: `{dict(issue_activation_only)}`",
        f"- Text-only issues: `{dict(issue_text_only)}`",
        "",
        "## Decision",
        "",
    ]
    if float(np.median(deltas)) < 0.05 or activation_only <= text_only:
        lines.append(
            "Do not promote this branch on complementarity grounds. The selected activation probes do not add a stable held-out advantage over text baselines."
        )
    else:
        lines.append(
            "Complementarity is nontrivial. Before causal work, freeze a single split-selected direction and verify the same pattern under prompt-template/plain-prompt cached modes."
        )
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resplit-dir", type=Path, default=DEFAULT_RESPLIT_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--template-variant", default="normal")
    parser.add_argument("--splits", default="dev,test")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    split_names = [part.strip() for part in args.splits.split(",") if part.strip()]
    rows: list[list[Any]] = []
    payloads: list[dict[str, Any]] = []
    for split_dir in sorted(args.resplit_dir.glob("split_*")):
        if not split_dir.is_dir():
            continue
        plan_id = split_dir.name
        examples = read_jsonl(split_dir / "probe_examples.jsonl")
        for split in split_names:
            text_preds = fit_text_predictions(examples, template_variant=args.template_variant, split=split)
            activation_preds = load_activation_predictions(split_dir, split)
            row, payload = metric_row(plan_id, split, examples, text_preds, activation_preds)
            rows.append(row)
            payloads.append(payload)
    write_report(args.output, resplit_dir=args.resplit_dir, rows=rows, payloads=payloads)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
