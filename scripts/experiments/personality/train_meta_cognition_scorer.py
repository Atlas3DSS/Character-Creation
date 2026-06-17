#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, f1_score
from sklearn.pipeline import FeatureUnion, Pipeline


DEFAULT_CORPUS_DIR = "/home/orwel/dev_genius/experiments/Character Creation/sweep_v4/meta_cognition_scorer_corpus_v1_20260417_121022"
DEFAULT_OUTPUT_ROOT = "/home/orwel/dev_genius/experiments/Character Creation/sweep_v4"
DEFAULT_TAG = "meta_cognition_text_scorer_v1"


@dataclass(frozen=True)
class VariantSpec:
    name: str
    use_context: bool
    use_behavior: bool
    use_metrics: bool


VARIANTS = (
    VariantSpec(name="response_only", use_context=False, use_behavior=False, use_metrics=False),
    VariantSpec(name="response_plus_behavior", use_context=False, use_behavior=True, use_metrics=True),
    VariantSpec(name="context_plus_response", use_context=True, use_behavior=False, use_metrics=False),
    VariantSpec(name="context_behavior_metrics", use_context=True, use_behavior=True, use_metrics=True),
)


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def turns_text(turns: list[dict[str, str]]) -> str:
    parts = []
    for turn in turns:
        speaker = str(turn.get("speaker", "user")).strip()
        content = str(turn.get("content", "")).strip()
        parts.append(f"{speaker}: {content}")
    return "\n".join(parts)


def format_text(row: dict[str, Any], spec: VariantSpec) -> str:
    blocks: list[str] = []
    if spec.use_behavior:
        blocks.append(f"behavior: {row['behavior']}")
        blocks.append(f"title: {row['title']}")
    if spec.use_metrics:
        metric_bits = []
        for metric in row.get("metrics", []):
            metric_bits.append(f"{metric.get('id', '')}: {metric.get('description', '')}")
        if metric_bits:
            blocks.append("metrics:\n" + "\n".join(metric_bits))
    if spec.use_context:
        blocks.append(f"setup: {row['setup']}")
        blocks.append("conversation:\n" + turns_text(row["turns"]))
    blocks.append("assistant_response:\n" + row["response_text"])
    return "\n\n".join(blocks)


def build_xy(rows: list[dict[str, Any]], spec: VariantSpec) -> tuple[list[str], np.ndarray]:
    X = [format_text(row, spec) for row in rows]
    y = np.array([int(row["label"]) for row in rows], dtype=np.int64)
    return X, y


def make_pipeline(C: float) -> Pipeline:
    features = FeatureUnion(
        [
            ("word", TfidfVectorizer(analyzer="word", lowercase=True, ngram_range=(1, 2), min_df=1, sublinear_tf=True)),
            ("char", TfidfVectorizer(analyzer="char_wb", lowercase=True, ngram_range=(3, 5), min_df=1, sublinear_tf=True)),
        ]
    )
    clf = LogisticRegression(
        max_iter=4000,
        solver="liblinear",
        C=C,
        class_weight="balanced",
    )
    return Pipeline([("features", features), ("clf", clf)])


def evaluate_rows(rows: list[dict[str, Any]], preds: np.ndarray, probs: np.ndarray) -> dict[str, Any]:
    y = np.array([int(row["label"]) for row in rows], dtype=np.int64)
    out: dict[str, Any] = {
        "n": int(len(rows)),
        "accuracy": float(accuracy_score(y, preds)),
        "balanced_accuracy": float(balanced_accuracy_score(y, preds)),
        "f1": float(f1_score(y, preds)),
        "label_counts": dict(sorted(Counter(y.tolist()).items())),
        "prediction_counts": dict(sorted(Counter(preds.tolist()).items())),
        "mean_positive_probability": float(np.mean(probs)),
    }
    report = classification_report(y, preds, output_dict=True, zero_division=0)
    out["classification_report"] = report
    by_behavior: dict[str, Any] = {}
    grouped: dict[str, list[int]] = defaultdict(list)
    for idx, row in enumerate(rows):
        grouped[row["behavior"]].append(idx)
    for behavior, idxs in sorted(grouped.items()):
        y_b = y[idxs]
        p_b = preds[idxs]
        pr_b = probs[idxs]
        by_behavior[behavior] = {
            "n": int(len(idxs)),
            "accuracy": float(accuracy_score(y_b, p_b)),
            "balanced_accuracy": float(balanced_accuracy_score(y_b, p_b)),
            "mean_positive_probability": float(np.mean(pr_b)),
        }
    out["by_behavior"] = by_behavior
    return out


def add_predictions(rows: list[dict[str, Any]], preds: np.ndarray, probs: np.ndarray, split: str, variant: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row, pred, prob in zip(rows, preds.tolist(), probs.tolist(), strict=True):
        example_key = f"{split}|{row['behavior']}|{row['candidate_id']}|{int(row['label'])}"
        out.append(
            {
                "example_key": example_key,
                "split": split,
                "variant": variant,
                "behavior": row["behavior"],
                "candidate_id": row["candidate_id"],
                "label": int(row["label"]),
                "pred": int(pred),
                "prob_positive": float(prob),
                "correct": bool(int(pred) == int(row["label"])),
                "response_text": row["response_text"],
                "judge_note": row["judge_note"],
            }
        )
    return out


def train_eval_variant(
    train_rows: list[dict[str, Any]],
    val_rows: list[dict[str, Any]],
    test_rows: list[dict[str, Any]],
    spec: VariantSpec,
    c_grid: list[float],
) -> dict[str, Any]:
    X_train, y_train = build_xy(train_rows, spec)
    X_val, y_val = build_xy(val_rows, spec)
    X_test, y_test = build_xy(test_rows, spec)

    candidates: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    best_pipe: Pipeline | None = None
    for C in c_grid:
        pipe = make_pipeline(C)
        pipe.fit(X_train, y_train)
        val_probs = pipe.predict_proba(X_val)[:, 1]
        val_preds = (val_probs >= 0.5).astype(np.int64)
        val_metrics = evaluate_rows(val_rows, val_preds, val_probs)
        record = {"C": C, "val_metrics": val_metrics}
        candidates.append(record)
        if best is None or val_metrics["balanced_accuracy"] > best["val_metrics"]["balanced_accuracy"] or (
            val_metrics["balanced_accuracy"] == best["val_metrics"]["balanced_accuracy"]
            and val_metrics["f1"] > best["val_metrics"]["f1"]
        ):
            best = record
            best_pipe = pipe
    assert best is not None
    assert best_pipe is not None

    best_C = float(best["C"])
    final_pipe = make_pipeline(best_C)
    X_trainval = X_train + X_val
    y_trainval = np.concatenate([y_train, y_val], axis=0)
    final_pipe.fit(X_trainval, y_trainval)

    split_preds: dict[str, list[dict[str, Any]]] = {}
    metrics: dict[str, Any] = {
        "selection": {
            "best_C": best_C,
            "searched_C": c_grid,
            "val_candidates": candidates,
        }
    }
    eval_plan = (
        ("train", train_rows, X_train, best_pipe),
        ("val", val_rows, X_val, best_pipe),
        ("test", test_rows, X_test, final_pipe),
    )
    for split_name, rows, X_split, pipe in eval_plan:
        probs = pipe.predict_proba(X_split)[:, 1]
        preds = (probs >= 0.5).astype(np.int64)
        metrics[split_name] = evaluate_rows(rows, preds, probs)
        split_preds[split_name] = add_predictions(rows, preds, probs, split_name, spec.name)

    return {
        "variant": spec.name,
        "metrics": metrics,
        "predictions": split_preds,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus-dir", type=Path, default=Path(DEFAULT_CORPUS_DIR))
    ap.add_argument("--output-root", type=Path, default=Path(DEFAULT_OUTPUT_ROOT))
    ap.add_argument("--tag", default=DEFAULT_TAG)
    ap.add_argument("--c-grid", default="0.25,0.5,1.0,2.0,4.0")
    args = ap.parse_args()

    c_grid = [float(x) for x in args.c_grid.split(",") if x.strip()]
    stamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    out_dir = args.output_root / f"{args.tag}_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    train_rows = load_jsonl(args.corpus_dir / "train.jsonl")
    val_rows = load_jsonl(args.corpus_dir / "val.jsonl")
    test_rows = load_jsonl(args.corpus_dir / "test.jsonl")

    manifest = {
        "started_at": now_iso(),
        "corpus_dir": str(args.corpus_dir),
        "c_grid": c_grid,
        "split_sizes": {"train": len(train_rows), "val": len(val_rows), "test": len(test_rows)},
        "variants": [spec.name for spec in VARIANTS],
    }
    write_json(out_dir / "manifest.json", manifest)

    runs: list[dict[str, Any]] = []
    all_predictions: list[dict[str, Any]] = []
    for spec in VARIANTS:
        result = train_eval_variant(train_rows, val_rows, test_rows, spec, c_grid)
        runs.append({"variant": result["variant"], "metrics": result["metrics"]})
        for split_name, rows in result["predictions"].items():
            write_jsonl(out_dir / f"{spec.name}_{split_name}_predictions.jsonl", rows)
            all_predictions.extend(rows)

    best = max(runs, key=lambda r: (r["metrics"]["val"]["balanced_accuracy"], r["metrics"]["val"]["f1"]))
    write_jsonl(out_dir / "all_predictions.jsonl", all_predictions)
    summary = {
        "finished_at": now_iso(),
        "best_variant": best["variant"],
        "best_val_balanced_accuracy": best["metrics"]["val"]["balanced_accuracy"],
        "best_test_balanced_accuracy": best["metrics"]["test"]["balanced_accuracy"],
        "runs": runs,
    }
    write_json(out_dir / "summary.json", summary)


if __name__ == "__main__":
    main()
