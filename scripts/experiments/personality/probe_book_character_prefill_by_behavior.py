#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


DEFAULT_PROBE_DIR = "/home/orwel/dev_genius/experiments/Character Creation/sweep_v4/book_character_prefill_activation_probe_v1_20260417_151635"
DEFAULT_OUTPUT_ROOT = "/home/orwel/dev_genius/experiments/Character Creation/sweep_v4"
DEFAULT_TAG = "book_character_prefill_behavior_probe_v1"


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


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


def make_classifier(C: float) -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=4000, solver="liblinear", C=C, class_weight="balanced")),
        ]
    )


def evaluate_binary(rows: list[dict[str, Any]], preds: np.ndarray, probs: np.ndarray) -> dict[str, Any]:
    y = np.array([int(row["label"]) for row in rows], dtype=np.int64)
    return {
        "n": int(len(rows)),
        "accuracy": float(accuracy_score(y, preds)),
        "balanced_accuracy": float(balanced_accuracy_score(y, preds)),
        "f1": float(f1_score(y, preds)),
        "label_counts": dict(sorted(Counter(y.tolist()).items())),
        "prediction_counts": dict(sorted(Counter(preds.tolist()).items())),
        "mean_positive_probability": float(np.mean(probs)),
    }


def parse_region_arrays(npz_path: Path) -> dict[str, dict[int, np.ndarray]]:
    data = np.load(npz_path)
    out: dict[str, dict[int, np.ndarray]] = defaultdict(dict)
    for key in data.files:
        if "__L" not in key:
            continue
        region, layer_raw = key.split("__L", 1)
        out[region][int(layer_raw)] = data[key]
    return out


def linear_probe_blob(clf: Pipeline) -> dict[str, Any]:
    scaler: StandardScaler = clf.named_steps["scaler"]
    logreg: LogisticRegression = clf.named_steps["clf"]
    return {
        "mean": scaler.mean_.astype(np.float32),
        "scale": scaler.scale_.astype(np.float32),
        "coef": logreg.coef_[0].astype(np.float32),
        "intercept": float(logreg.intercept_[0]),
    }


def train_one_probe(
    X: np.ndarray,
    labels: np.ndarray,
    rows: list[dict[str, Any]],
    c_grid: list[float],
) -> dict[str, Any]:
    split_idx = defaultdict(list)
    for idx, row in enumerate(rows):
        split_idx[row["split"]].append(idx)
    train_idx = np.array(split_idx["train"], dtype=np.int64)
    val_idx = np.array(split_idx["val"], dtype=np.int64)
    test_idx = np.array(split_idx["test"], dtype=np.int64)

    best: dict[str, Any] | None = None
    best_train_clf: Pipeline | None = None
    searches: list[dict[str, Any]] = []
    X_train = X[train_idx]
    X_val = X[val_idx]
    y_train = labels[train_idx]

    for C in c_grid:
        clf = make_classifier(C)
        clf.fit(X_train, y_train)
        val_probs = clf.predict_proba(X_val)[:, 1]
        val_preds = (val_probs >= 0.5).astype(np.int64)
        val_rows = [rows[i] for i in val_idx.tolist()]
        val_metrics = evaluate_binary(val_rows, val_preds, val_probs)
        rec = {"C": float(C), "val_metrics": val_metrics}
        searches.append(rec)
        if best is None or val_metrics["balanced_accuracy"] > best["val_metrics"]["balanced_accuracy"] or (
            val_metrics["balanced_accuracy"] == best["val_metrics"]["balanced_accuracy"]
            and val_metrics["f1"] > best["val_metrics"]["f1"]
        ):
            best = rec
            best_train_clf = clf
    assert best is not None and best_train_clf is not None

    C = float(best["C"])
    trainval_idx = np.concatenate([train_idx, val_idx], axis=0)
    clf = make_classifier(C)
    clf.fit(X[trainval_idx], labels[trainval_idx])

    split_preds: dict[str, list[dict[str, Any]]] = {}
    split_metrics: dict[str, Any] = {"selection": {"best_C": C, "searches": searches}}
    for split_name, idxs, split_clf in (
        ("train", train_idx, best_train_clf),
        ("val", val_idx, best_train_clf),
        ("test", test_idx, clf),
    ):
        X_split = X[idxs]
        probs = split_clf.predict_proba(X_split)[:, 1]
        preds = (probs >= 0.5).astype(np.int64)
        split_rows = [rows[i] for i in idxs.tolist()]
        split_metrics[split_name] = evaluate_binary(split_rows, preds, probs)
        split_preds[split_name] = [
            {
                "feature_id": split_rows[i]["feature_id"],
                "split": split_name,
                "behavior": split_rows[i]["behavior"],
                "label": int(split_rows[i]["label"]),
                "pred": int(preds[i]),
                "prob_positive": float(probs[i]),
                "correct": bool(int(preds[i]) == int(split_rows[i]["label"])),
            }
            for i in range(len(split_rows))
        ]
    return {
        "metrics": split_metrics,
        "predictions": split_preds,
        "final_probe": linear_probe_blob(clf),
    }


def region_direction(
    X: np.ndarray,
    labels: np.ndarray,
    rows: list[dict[str, Any]],
) -> np.ndarray:
    train_idx = [i for i, row in enumerate(rows) if row["split"] == "train"]
    train_labels = labels[train_idx]
    train_X = X[train_idx]
    pos = train_X[train_labels == 1].mean(axis=0)
    neg = train_X[train_labels == 0].mean(axis=0)
    direction = (pos - neg).astype(np.float32)
    norm = float(np.linalg.norm(direction))
    if norm > 0:
        direction /= norm
    return direction


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe-dir", type=Path, default=Path(DEFAULT_PROBE_DIR))
    ap.add_argument("--output-root", type=Path, default=Path(DEFAULT_OUTPUT_ROOT))
    ap.add_argument("--tag", default=DEFAULT_TAG)
    ap.add_argument("--c-grid", default="0.25,0.5,1.0,2.0")
    args = ap.parse_args()

    stamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    out_dir = args.output_root / f"{args.tag}_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    c_grid = [float(x) for x in args.c_grid.split(",") if x.strip()]

    meta_rows = load_jsonl(args.probe_dir / "feature_meta.jsonl")
    region_arrays = parse_region_arrays(args.probe_dir / "features.npz")
    behaviors = sorted({row["behavior"] for row in meta_rows})
    labels_all = np.array([int(row["label"]) for row in meta_rows], dtype=np.int64)

    write_json(
        out_dir / "manifest.json",
        {
            "started_at": now_iso(),
            "probe_dir": str(args.probe_dir),
            "n_rows": len(meta_rows),
            "behaviors": behaviors,
            "c_grid": c_grid,
        },
    )

    artifact_arrays: dict[str, np.ndarray] = {}
    summary: dict[str, Any] = {"finished_at": None, "behaviors": {}}

    for behavior in behaviors:
        idxs = [i for i, row in enumerate(meta_rows) if row["behavior"] == behavior]
        rows = [meta_rows[i] for i in idxs]
        labels = labels_all[idxs]
        best_overall: dict[str, Any] | None = None
        best_by_region: dict[str, Any] = {}
        searches_compact: list[dict[str, Any]] = []

        for region_name, layer_map in region_arrays.items():
            region_best: dict[str, Any] | None = None
            for layer_idx, X_all in layer_map.items():
                X = X_all[idxs]
                trained = train_one_probe(X, labels, rows, c_grid)
                val_metrics = trained["metrics"]["val"]
                test_metrics = trained["metrics"]["test"]
                rec = {
                    "region": region_name,
                    "layer": int(layer_idx),
                    "C": float(trained["metrics"]["selection"]["best_C"]),
                    "val_metrics": val_metrics,
                    "test_metrics": test_metrics,
                }
                searches_compact.append(rec)
                if region_best is None or val_metrics["balanced_accuracy"] > region_best["val_metrics"]["balanced_accuracy"] or (
                    val_metrics["balanced_accuracy"] == region_best["val_metrics"]["balanced_accuracy"]
                    and val_metrics["f1"] > region_best["val_metrics"]["f1"]
                ):
                    region_best = rec | {"trained": trained}
                if best_overall is None or val_metrics["balanced_accuracy"] > best_overall["val_metrics"]["balanced_accuracy"] or (
                    val_metrics["balanced_accuracy"] == best_overall["val_metrics"]["balanced_accuracy"]
                    and val_metrics["f1"] > best_overall["val_metrics"]["f1"]
                ):
                    best_overall = rec | {"trained": trained}

            assert region_best is not None
            best_by_region[region_name] = {
                "region": region_best["region"],
                "layer": region_best["layer"],
                "C": region_best["C"],
                "val_metrics": region_best["val_metrics"],
                "test_metrics": region_best["test_metrics"],
            }

            if region_name in {"think_mean", "response_mean"}:
                trained = region_best["trained"]
                probe_blob = trained["final_probe"]
                direction = region_direction(region_arrays[region_name][region_best["layer"]][idxs], labels, rows)
                prefix = f"{behavior}__{region_name}__L{int(region_best['layer']):02d}"
                artifact_arrays[f"direction__{prefix}"] = direction
                artifact_arrays[f"probe_mean__{prefix}"] = probe_blob["mean"]
                artifact_arrays[f"probe_scale__{prefix}"] = probe_blob["scale"]
                artifact_arrays[f"probe_coef__{prefix}"] = probe_blob["coef"]
                artifact_arrays[f"probe_intercept__{prefix}"] = np.array([probe_blob["intercept"]], dtype=np.float32)
                for split_name, pred_rows in trained["predictions"].items():
                    write_jsonl(out_dir / f"{behavior}_{region_name}_{split_name}_predictions.jsonl", pred_rows)

        assert best_overall is not None
        summary["behaviors"][behavior] = {
            "n_rows": len(rows),
            "label_counts": dict(sorted(Counter(labels.tolist()).items())),
            "best_overall": {
                "region": best_overall["region"],
                "layer": best_overall["layer"],
                "C": best_overall["C"],
                "val_metrics": best_overall["val_metrics"],
                "test_metrics": best_overall["test_metrics"],
            },
            "best_by_region": best_by_region,
            "searches_compact": sorted(
                searches_compact,
                key=lambda row: (
                    row["val_metrics"]["balanced_accuracy"],
                    row["val_metrics"]["f1"],
                    row["region"],
                    row["layer"],
                ),
                reverse=True,
            ),
        }

    np.savez_compressed(out_dir / "behavior_probe_artifacts.npz", **artifact_arrays)
    summary["finished_at"] = now_iso()
    write_json(out_dir / "summary.json", summary)


if __name__ == "__main__":
    main()
