#!/usr/bin/env python3
"""Export an exact SCOTUS linear-probe direction from cached Phase 4 features."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.pipeline import Pipeline

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from probe_scotus_style import make_classifier, predict_metrics, read_jsonl, split_indices, write_json  # noqa: E402


def feature_key(region: str, layer: int) -> str:
    return f"{region}__L{layer:02d}"


def raw_direction_from_pipeline(clf: Pipeline) -> tuple[np.ndarray, float]:
    scaler = clf.named_steps["scaler"]
    logreg = clf.named_steps["clf"]
    coef = logreg.coef_[0].astype(np.float32)
    scale = scaler.scale_.astype(np.float32)
    raw_direction = coef / np.maximum(scale, 1e-12)
    raw_norm = float(np.linalg.norm(raw_direction))
    if raw_norm <= 0.0:
        raise RuntimeError("Zero raw direction")
    return (raw_direction / raw_norm).astype(np.float32), raw_norm


def fit_and_eval(
    *,
    x_matrix: np.ndarray,
    labels: np.ndarray,
    meta_rows: list[dict[str, Any]],
    c_value: float,
    solver: str,
    max_iter: int,
    tol: float,
) -> tuple[Pipeline, dict[str, Any]]:
    idx = split_indices(meta_rows)
    train_idx = idx["train"]
    dev_idx = idx["dev"]
    test_idx = idx["test"]
    train_dev_idx = np.concatenate([train_idx, dev_idx])

    train_only_clf = make_classifier(c_value, solver=solver, max_iter=max_iter, tol=tol)
    train_only_clf.fit(x_matrix[train_idx], labels[train_idx])
    dev_rows = [meta_rows[i] for i in dev_idx.tolist()]
    test_rows = [meta_rows[i] for i in test_idx.tolist()]
    dev_metrics, _ = predict_metrics(train_only_clf, x_matrix[dev_idx], dev_rows)
    diagnostic_test_metrics, _ = predict_metrics(train_only_clf, x_matrix[test_idx], test_rows)

    final_clf = make_classifier(c_value, solver=solver, max_iter=max_iter, tol=tol)
    final_clf.fit(x_matrix[train_dev_idx], labels[train_dev_idx])

    split_metrics: dict[str, Any] = {}
    for split_name, split_idx in (("train", train_idx), ("dev", dev_idx), ("test", test_idx)):
        rows = [meta_rows[i] for i in split_idx.tolist()]
        metrics, _ = predict_metrics(final_clf, x_matrix[split_idx], rows)
        split_metrics[split_name] = metrics

    return final_clf, {
        "dev_metrics_train_only": dev_metrics,
        "test_metrics_train_only_diagnostic": diagnostic_test_metrics,
        "final_refit_split_metrics": split_metrics,
    }


def export_direction(args: argparse.Namespace) -> Path:
    run_dir: Path = args.run_dir
    meta_rows = read_jsonl(run_dir / "feature_meta.jsonl")
    labels = np.array([int(row["label"]) for row in meta_rows], dtype=np.int64)
    key = feature_key(args.region, args.layer)
    with np.load(run_dir / "features.npz") as data:
        if key not in data.files:
            raise KeyError(f"Missing {key} in {run_dir / 'features.npz'}")
        x_matrix = data[key].astype(np.float32, copy=False)
    if x_matrix.shape[0] != len(meta_rows):
        raise RuntimeError(f"Feature/meta row mismatch: {x_matrix.shape[0]} vs {len(meta_rows)}")

    clf, metrics = fit_and_eval(
        x_matrix=x_matrix,
        labels=labels,
        meta_rows=meta_rows,
        c_value=args.c_value,
        solver=args.classifier_solver,
        max_iter=args.classifier_max_iter,
        tol=args.classifier_tol,
    )
    raw_unit, raw_norm = raw_direction_from_pipeline(clf)
    scaler = clf.named_steps["scaler"]
    logreg = clf.named_steps["clf"]
    positive_justice = str(meta_rows[0].get("positive_justice", "positive"))

    out_dir = args.output_dir or (run_dir / "directions")
    out_dir.mkdir(parents=True, exist_ok=True)
    c_tag = str(args.c_value).replace(".", "p").replace("-", "m")
    out_path = out_dir / f"probe_direction_{args.region}_L{args.layer:02d}_C{c_tag}.npz"
    np.savez_compressed(
        out_path,
        scaler_mean=scaler.mean_.astype(np.float32),
        scaler_scale=scaler.scale_.astype(np.float32),
        coef=logreg.coef_.astype(np.float32),
        intercept=logreg.intercept_.astype(np.float32),
        raw_direction_unit=raw_unit,
        raw_direction_norm=np.array([raw_norm], dtype=np.float32),
        region=np.array([args.region]),
        layer=np.array([int(args.layer)]),
        C=np.array([float(args.c_value)], dtype=np.float32),
        positive_justice=np.array([positive_justice]),
        source_run=np.array([str(run_dir)]),
    )
    manifest = {
        "source_run": str(run_dir),
        "direction_path": str(out_path),
        "region": args.region,
        "layer": args.layer,
        "C": args.c_value,
        "positive_justice": positive_justice,
        "raw_direction_norm": raw_norm,
        **metrics,
    }
    write_json(out_path.with_suffix(".json"), manifest)
    print(out_path)
    return out_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export an exact SCOTUS probe direction from cached features.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--region",
        choices=[
            "prompt_last",
            "prompt_mean",
            "excerpt_mean",
            "assistant_all",
            "assistant_early",
            "assistant_late",
            "holding_mean",
            "reasoning_mean",
        ],
        required=True,
    )
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--c-value", type=float, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--classifier-solver", default="lbfgs")
    parser.add_argument("--classifier-max-iter", type=int, default=500)
    parser.add_argument("--classifier-tol", type=float, default=0.001)
    return parser.parse_args()


def main() -> None:
    export_direction(parse_args())


if __name__ == "__main__":
    main()
