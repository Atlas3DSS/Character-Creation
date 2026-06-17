#!/usr/bin/env python3
"""Probe SCOTUS justice style through Qwen-Scope SAE features.

This script is a companion to probe_scotus_style.py. It does not rerun the
base LLM. Instead, it reads saved residual-stream readouts from features.npz,
encodes them through downloaded Qwen-Scope SAEs, and trains sparse linear
classifiers on the resulting SAE feature activations.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import scipy.sparse as sp
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PROBE_DIR = PROJECT_ROOT / "sweep_v4" / "scotus_probe_20260425_085108"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "sweep_v4"
DEFAULT_SAE_PATHS = [
    Path("/home/orwel/dev_genius/models/qwen_scope/SAE-Res-Qwen3.5-27B-W80K-L0_50"),
    Path("/home/orwel/dev_genius/models/qwen_scope/SAE-Res-Qwen3.5-27B-W80K-L0_100"),
]
DEFAULT_LAYERS = "4,8,12,16"
DEFAULT_REGIONS = "prompt_last,prompt_mean,excerpt_mean"
DEFAULT_C_GRID = "0.01,0.03,0.1,0.25,0.5,1.0,2.0,10.0"


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return "\n".join(lines)


def parse_int_list(raw: str) -> list[int]:
    values: list[int] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = (int(x) for x in part.split("-", 1))
            values.extend(range(start, end + 1))
        else:
            values.append(int(part))
    return sorted(set(values))


def parse_str_list(raw: str) -> list[str]:
    return [part.strip() for part in raw.split(",") if part.strip()]


def parse_float_list(raw: str) -> list[float]:
    return [float(part.strip()) for part in raw.split(",") if part.strip()]


def infer_top_k(sae_path: Path) -> int:
    match = re.search(r"L0_(\d+)", sae_path.name)
    if not match:
        raise ValueError(f"Could not infer top-k from SAE path name: {sae_path}")
    return int(match.group(1))


def safe_sae_name(sae_path: Path) -> str:
    return sae_path.name.replace("/", "--")


def split_indices(meta_rows: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    grouped: dict[str, list[int]] = defaultdict(list)
    for idx, row in enumerate(meta_rows):
        grouped[str(row["split"])].append(idx)
    return {split: np.array(idxs, dtype=np.int64) for split, idxs in grouped.items()}


def infer_label_names(meta_rows: list[dict[str, Any]]) -> dict[int, str]:
    grouped: dict[int, list[str]] = {0: [], 1: []}
    for row in meta_rows:
        label = int(row["label"])
        if label in grouped:
            grouped[label].append(str(row.get("justice", f"label_{label}")))
    names: dict[int, str] = {}
    for label, values in grouped.items():
        names[label] = Counter(values).most_common(1)[0][0] if values else f"label_{label}"
    return names


def vector_metrics(rows: list[dict[str, Any]], preds: np.ndarray, probs: np.ndarray) -> dict[str, Any]:
    y = np.array([int(row["label"]) for row in rows], dtype=np.int64)
    return {
        "n": int(len(rows)),
        "accuracy": float(accuracy_score(y, preds)),
        "balanced_accuracy": float(balanced_accuracy_score(y, preds)),
        "f1": float(f1_score(y, preds)),
        "label_counts": dict(sorted(Counter(y.tolist()).items())),
        "prediction_counts": dict(sorted(Counter(preds.tolist()).items())),
        "mean_positive_probability": float(np.mean(probs)) if len(probs) else None,
    }


def make_classifier(c_value: float) -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler(with_mean=False)),
            (
                "clf",
                LogisticRegression(
                    max_iter=4000,
                    solver="liblinear",
                    C=c_value,
                    class_weight="balanced",
                ),
            ),
        ]
    )


def predict_metrics(clf: Pipeline, x_matrix: sp.csr_matrix, rows: list[dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    probs = clf.predict_proba(x_matrix)[:, 1]
    preds = (probs >= 0.5).astype(np.int64)
    metrics = vector_metrics(rows, preds, probs)
    predictions = [
        {
            "example_id": row["example_id"],
            "chunk_id": row["chunk_id"],
            "split": row["split"],
            "justice": row["justice"],
            "label": int(row["label"]),
            "pred": int(preds[idx]),
            "prob_positive": float(probs[idx]),
            "correct": bool(int(preds[idx]) == int(row["label"])),
        }
        for idx, row in enumerate(rows)
    ]
    return metrics, predictions


def load_sae_encoder(sae_path: Path, layer: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    path = sae_path / f"layer{layer}.sae.pt"
    if not path.exists():
        raise FileNotFoundError(path)
    sae = torch.load(path, map_location="cpu", weights_only=True)
    w_enc = sae["W_enc"].to(device=device, dtype=torch.float32).T.contiguous()
    b_enc = sae["b_enc"].to(device=device, dtype=torch.float32).contiguous()
    del sae
    return w_enc, b_enc


def encode_hidden_to_sae_csr(
    hidden: np.ndarray,
    *,
    sae_path: Path,
    layer: int,
    top_k: int,
    batch_size: int,
    device: torch.device,
) -> sp.csr_matrix:
    if hidden.ndim != 2:
        raise ValueError(f"Expected 2D hidden array, got shape {hidden.shape}")
    if hidden.shape[1] != 5120:
        raise ValueError(f"Expected hidden dim 5120 for Qwen3.5-27B SAE, got {hidden.shape[1]}")

    w_enc, b_enc = load_sae_encoder(sae_path, layer, device)
    n_rows = int(hidden.shape[0])
    sae_width = int(b_enc.shape[0])
    row_parts: list[np.ndarray] = []
    col_parts: list[np.ndarray] = []
    data_parts: list[np.ndarray] = []

    with torch.inference_mode():
        for start in tqdm(range(0, n_rows, batch_size), desc=f"encode L{layer} {sae_path.name}", unit="batch"):
            end = min(start + batch_size, n_rows)
            batch_np = np.ascontiguousarray(hidden[start:end], dtype=np.float32)
            batch = torch.from_numpy(batch_np).to(device=device, dtype=torch.float32)
            pre = torch.relu(batch @ w_enc + b_enc)
            values, indices = torch.topk(pre, k=top_k, dim=-1)
            values_np = values.detach().cpu().numpy().astype(np.float32, copy=False)
            indices_np = indices.detach().cpu().numpy().astype(np.int32, copy=False)
            flat_values = values_np.reshape(-1)
            flat_cols = indices_np.reshape(-1)
            flat_rows = np.repeat(np.arange(start, end, dtype=np.int32), top_k)
            keep = flat_values > 0
            row_parts.append(flat_rows[keep])
            col_parts.append(flat_cols[keep])
            data_parts.append(flat_values[keep])
            del batch, pre, values, indices

    del w_enc, b_enc
    if device.type == "cuda":
        torch.cuda.empty_cache()

    if not data_parts:
        return sp.csr_matrix((n_rows, sae_width), dtype=np.float32)
    rows = np.concatenate(row_parts)
    cols = np.concatenate(col_parts)
    data = np.concatenate(data_parts)
    return sp.csr_matrix((data, (rows, cols)), shape=(n_rows, sae_width), dtype=np.float32)


def get_or_build_sae_features(
    *,
    features_npz: Path,
    cache_dir: Path,
    sae_path: Path,
    layer: int,
    region: str,
    top_k: int,
    batch_size: int,
    device: torch.device,
    overwrite: bool,
) -> sp.csr_matrix:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{safe_sae_name(sae_path)}__{region}__L{layer:02d}.npz"
    if cache_path.exists() and not overwrite:
        return sp.load_npz(cache_path).tocsr()

    key = f"{region}__L{layer:02d}"
    with np.load(features_npz) as data:
        if key not in data.files:
            raise KeyError(f"Missing {key} in {features_npz}")
        hidden = data[key].astype(np.float32, copy=False)
        x_sae = encode_hidden_to_sae_csr(
            hidden,
            sae_path=sae_path,
            layer=layer,
            top_k=top_k,
            batch_size=batch_size,
            device=device,
        )
    sp.save_npz(cache_path, x_sae)
    return x_sae


def train_one_config(
    x_matrix: sp.csr_matrix,
    meta_rows: list[dict[str, Any]],
    labels: np.ndarray,
    c_grid: list[float],
) -> dict[str, Any]:
    idx = split_indices(meta_rows)
    train_idx = idx["train"]
    dev_idx = idx["dev"]
    test_idx = idx["test"]
    dev_rows = [meta_rows[i] for i in dev_idx.tolist()]

    best: dict[str, Any] | None = None
    best_clf: Pipeline | None = None
    searches: list[dict[str, Any]] = []
    for c_value in c_grid:
        clf = make_classifier(c_value)
        clf.fit(x_matrix[train_idx], labels[train_idx])
        dev_metrics, _ = predict_metrics(clf, x_matrix[dev_idx], dev_rows)
        record = {"C": float(c_value), "dev_metrics": dev_metrics}
        searches.append(record)
        if best is None or (
            dev_metrics["balanced_accuracy"],
            dev_metrics["f1"],
            -float(c_value),
        ) > (
            best["dev_metrics"]["balanced_accuracy"],
            best["dev_metrics"]["f1"],
            -float(best["C"]),
        ):
            best = record
            best_clf = clf

    assert best is not None
    assert best_clf is not None
    train_dev_idx = np.concatenate([train_idx, dev_idx])
    final_clf = make_classifier(float(best["C"]))
    final_clf.fit(x_matrix[train_dev_idx], labels[train_dev_idx])

    split_metrics: dict[str, Any] = {}
    split_predictions: dict[str, list[dict[str, Any]]] = {}
    for split, split_idx, clf in (
        ("train", train_idx, best_clf),
        ("dev", dev_idx, best_clf),
        ("test", test_idx, final_clf),
    ):
        rows = [meta_rows[i] for i in split_idx.tolist()]
        metrics, predictions = predict_metrics(clf, x_matrix[split_idx], rows)
        split_metrics[split] = metrics
        split_predictions[split] = predictions

    searches.sort(key=lambda row: (row["dev_metrics"]["balanced_accuracy"], row["dev_metrics"]["f1"], -float(row["C"])), reverse=True)
    return {
        "best_C": float(best["C"]),
        "best_dev_metrics": best["dev_metrics"],
        "split_metrics": split_metrics,
        "split_predictions": split_predictions,
        "searches": searches,
        "final_clf": final_clf,
    }


def filter_features_by_train_df(
    x_matrix: sp.csr_matrix,
    train_idx: np.ndarray,
    *,
    min_train_df: int,
) -> tuple[sp.csr_matrix, np.ndarray, np.ndarray]:
    feature_map = np.arange(x_matrix.shape[1], dtype=np.int64)
    if min_train_df <= 1:
        train_df = np.diff(x_matrix[train_idx].tocsc().indptr).astype(np.int64, copy=False)
        return x_matrix, feature_map, train_df
    train_df = np.diff(x_matrix[train_idx].tocsc().indptr).astype(np.int64, copy=False)
    keep = train_df >= min_train_df
    if not np.any(keep):
        raise RuntimeError(f"No SAE features survive min_train_df={min_train_df}")
    return x_matrix[:, keep].tocsr(), feature_map[keep], train_df[keep]


def classifier_feature_weights(clf: Pipeline) -> np.ndarray:
    scaler: StandardScaler = clf.named_steps["scaler"]
    logreg: LogisticRegression = clf.named_steps["clf"]
    coef = logreg.coef_[0].astype(np.float64, copy=False)
    scale = np.asarray(scaler.scale_, dtype=np.float64)
    scale[scale == 0] = 1.0
    return coef / scale


def top_feature_rows(
    *,
    clf: Pipeline,
    x_matrix: sp.csr_matrix,
    labels: np.ndarray,
    train_dev_idx: np.ndarray,
    feature_map: np.ndarray,
    train_df: np.ndarray,
    top_n: int,
    label_names: dict[int, str],
) -> list[dict[str, Any]]:
    weights = classifier_feature_weights(clf)
    if top_n <= 0:
        return []
    top_pos = np.argsort(weights)[-top_n:][::-1]
    top_neg = np.argsort(weights)[:top_n]
    features = list(dict.fromkeys([int(x) for x in np.concatenate([top_pos, top_neg])]))
    x_td = x_matrix[train_dev_idx].tocsc()
    y_td = labels[train_dev_idx]
    pos_mask = y_td == 1
    neg_mask = y_td == 0
    rows: list[dict[str, Any]] = []
    for feat_idx in features:
        col = x_td.getcol(feat_idx)
        pos_col = col[pos_mask]
        neg_col = col[neg_mask]
        rows.append(
            {
                "feature": int(feature_map[feat_idx]),
                "filtered_feature": int(feat_idx),
                "train_df": int(train_df[feat_idx]),
                "weight": float(weights[feat_idx]),
                "direction": f"toward_{label_names[1] if weights[feat_idx] >= 0 else label_names[0]}",
                "mean_positive": float(pos_col.mean()),
                "mean_negative": float(neg_col.mean()),
                "df_positive": int(pos_col.getnnz()),
                "df_negative": int(neg_col.getnnz()),
                "activation_rate_positive": float(pos_col.getnnz() / max(1, int(pos_mask.sum()))),
                "activation_rate_negative": float(neg_col.getnnz() / max(1, int(neg_mask.sum()))),
            }
        )
    rows.sort(key=lambda row: abs(float(row["weight"])), reverse=True)
    return rows


def label_counts_for_indices(labels: np.ndarray, indices: np.ndarray) -> Counter[int]:
    return Counter(labels[indices].tolist())


def stress_tests(
    *,
    x_matrix: sp.csr_matrix,
    meta_rows: list[dict[str, Any]],
    labels: np.ndarray,
    c_value: float,
    min_eval_per_label: int,
) -> dict[str, list[dict[str, Any]]]:
    idx = split_indices(meta_rows)
    train_dev_idx = np.concatenate([idx["train"], idx["dev"]])
    test_idx = idx["test"]
    fields = ["issue_area_label", "opinion_type", "section_posture"]
    results: dict[str, list[dict[str, Any]]] = {}
    for field in fields:
        field_rows: list[dict[str, Any]] = []
        values = sorted({str(row.get(field, "unknown")) for row in meta_rows})
        for value in values:
            train_idx = np.array(
                [i for i in train_dev_idx.tolist() if str(meta_rows[i].get(field, "unknown")) != value],
                dtype=np.int64,
            )
            eval_idx = np.array(
                [i for i in test_idx.tolist() if str(meta_rows[i].get(field, "unknown")) == value],
                dtype=np.int64,
            )
            train_counts = label_counts_for_indices(labels, train_idx)
            eval_counts = label_counts_for_indices(labels, eval_idx)
            if min(train_counts.get(0, 0), train_counts.get(1, 0)) < 10:
                continue
            if min(eval_counts.get(0, 0), eval_counts.get(1, 0)) < min_eval_per_label:
                continue
            clf = make_classifier(c_value)
            clf.fit(x_matrix[train_idx], labels[train_idx])
            rows = [meta_rows[i] for i in eval_idx.tolist()]
            metrics, _ = predict_metrics(clf, x_matrix[eval_idx], rows)
            field_rows.append(
                {
                    "held_out": value,
                    "n_eval": int(len(eval_idx)),
                    "train_label_counts": dict(sorted(train_counts.items())),
                    "eval_label_counts": dict(sorted(eval_counts.items())),
                    "metrics": metrics,
                }
            )
        field_rows.sort(key=lambda row: row["metrics"]["balanced_accuracy"], reverse=True)
        results[field] = field_rows
    return results


def write_report(
    path: Path,
    *,
    manifest: dict[str, Any],
    results: list[dict[str, Any]],
    best_result: dict[str, Any],
    top_features: list[dict[str, Any]],
    stress: dict[str, list[dict[str, Any]]],
) -> None:
    top_rows = [
        [
            row["sae_name"],
            row["top_k"],
            row["region"],
            row["layer"],
            row["best_C"],
            f"{row['best_dev_metrics']['balanced_accuracy']:.3f}",
            f"{row['split_metrics']['test']['balanced_accuracy']:.3f}",
            f"{row['split_metrics']['test']['f1']:.3f}",
        ]
        for row in results[:25]
    ]
    split_rows = [
        [
            split,
            metrics["n"],
            f"{metrics['accuracy']:.3f}",
            f"{metrics['balanced_accuracy']:.3f}",
            f"{metrics['f1']:.3f}",
        ]
        for split, metrics in best_result["split_metrics"].items()
    ]
    feature_rows = [
        [
            row["feature"],
            row["train_df"],
            row["direction"],
            f"{row['weight']:.6g}",
            f"{row['mean_positive']:.6g}",
            f"{row['mean_negative']:.6g}",
            row["df_positive"],
            row["df_negative"],
            f"{row['activation_rate_positive']:.3f}",
            f"{row['activation_rate_negative']:.3f}",
        ]
        for row in top_features[:20]
    ]
    stress_rows: list[list[Any]] = []
    for field, rows in stress.items():
        for row in rows[:10]:
            metrics = row["metrics"]
            stress_rows.append(
                [
                    field,
                    row["held_out"],
                    row["n_eval"],
                    f"{metrics['balanced_accuracy']:.3f}",
                    f"{metrics['accuracy']:.3f}",
                ]
            )
    lines = [
        "# SCOTUS Qwen-Scope SAE Probe",
        "",
        f"Started: `{manifest['started_at']}`",
        f"Finished: `{manifest.get('finished_at', '')}`",
        "",
        "## Configuration",
        "",
        markdown_table(
            ["Field", "Value"],
            [
                ["Source probe dir", manifest["probe_dir"]],
                ["SAE paths", "<br>".join(manifest["sae_paths"])],
                ["Layers", ", ".join(str(x) for x in manifest["layers"])],
                ["Regions", ", ".join(manifest["regions"])],
                ["C grid", ", ".join(str(x) for x in manifest["c_grid"])],
                ["Min train DF", manifest["min_train_df"]],
                ["Feature cache dir", manifest["feature_cache_dir"]],
                ["Label names", json.dumps(manifest.get("label_names", {}), sort_keys=True)],
            ],
        ),
        "",
        "## Best SAE Probe",
        "",
        markdown_table(
            ["Field", "Value"],
            [
                ["SAE", best_result["sae_name"]],
                ["Top-k", best_result["top_k"]],
                ["Region", best_result["region"]],
                ["Layer", best_result["layer"]],
                ["C", best_result["best_C"]],
            ],
        ),
        "",
        "## Best Split Metrics",
        "",
        markdown_table(["Split", "N", "Accuracy", "Balanced accuracy", "F1"], split_rows),
        "",
        "## Top Search Results",
        "",
        markdown_table(["SAE", "Top-k", "Region", "Layer", "C", "Dev BA", "Test BA", "Test F1"], top_rows),
        "",
        "## Top Discriminative SAE Features",
        "",
        markdown_table(
            [
                "Feature",
                "Train DF",
                "Direction",
                "Weight",
                "Mean positive",
                "Mean negative",
                "DF positive",
                "DF negative",
                "Rate positive",
                "Rate negative",
            ],
            feature_rows,
        ),
        "",
        "## Held-Out Stress Tests",
        "",
        markdown_table(["Field", "Held out", "N eval", "Balanced accuracy", "Accuracy"], stress_rows),
        "",
        "## Interpretation Note",
        "",
        "These numbers are SAE-feature decoders over saved Phase 4 residual readouts. They test whether the",
        "target distinction is recoverable through sparse Qwen-Scope features. They are not yet",
        "causal steering evidence; candidate features still need prompt/null diagnostics and generation tests.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SCOTUS Qwen-Scope SAE feature probes.")
    parser.add_argument("--probe-dir", type=Path, default=DEFAULT_PROBE_DIR)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--sae-path", type=Path, action="append", default=None)
    parser.add_argument("--layers", default=DEFAULT_LAYERS)
    parser.add_argument("--regions", default=DEFAULT_REGIONS)
    parser.add_argument("--c-grid", default=DEFAULT_C_GRID)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--feature-cache-dir", type=Path, default=None)
    parser.add_argument("--min-train-df", type=int, default=1)
    parser.add_argument("--overwrite-features", action="store_true")
    parser.add_argument("--top-features", type=int, default=25)
    parser.add_argument("--stress-min-eval-per-label", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    probe_dir = args.probe_dir
    features_npz = probe_dir / "features.npz"
    meta_path = probe_dir / "feature_meta.jsonl"
    if not features_npz.exists():
        raise FileNotFoundError(features_npz)
    if not meta_path.exists():
        raise FileNotFoundError(meta_path)

    sae_paths = args.sae_path or DEFAULT_SAE_PATHS
    layers = parse_int_list(args.layers)
    regions = parse_str_list(args.regions)
    c_grid = parse_float_list(args.c_grid)
    device = torch.device(args.device)
    meta_rows = read_jsonl(meta_path)
    labels = np.array([int(row["label"]) for row in meta_rows], dtype=np.int64)
    idx = split_indices(meta_rows)
    label_names = infer_label_names(meta_rows)

    stamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    out_dir = args.output_root / f"scotus_sae_probe_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = args.feature_cache_dir or (out_dir / "sae_features")

    manifest: dict[str, Any] = {
        "started_at": now_iso(),
        "probe_dir": str(probe_dir),
        "features_npz": str(features_npz),
        "feature_meta": str(meta_path),
        "output_dir": str(out_dir),
        "sae_paths": [str(path) for path in sae_paths],
        "layers": layers,
        "regions": regions,
        "c_grid": c_grid,
        "batch_size": args.batch_size,
        "device": str(device),
        "feature_cache_dir": str(cache_dir),
        "min_train_df": args.min_train_df,
        "example_counts": {
            split: dict(sorted(Counter(labels[split_idx].tolist()).items()))
            for split, split_idx in sorted(idx.items())
        },
        "label_names": {str(label): name for label, name in sorted(label_names.items())},
    }
    write_json(out_dir / "manifest.json", manifest)

    results: list[dict[str, Any]] = []
    for sae_path in sae_paths:
        top_k = infer_top_k(sae_path)
        for layer in layers:
            for region in regions:
                print(f"\n=== SAE {sae_path.name} | {region} L{layer} ===", flush=True)
                x_sae = get_or_build_sae_features(
                    features_npz=features_npz,
                    cache_dir=cache_dir,
                    sae_path=sae_path,
                    layer=layer,
                    region=region,
                    top_k=top_k,
                    batch_size=args.batch_size,
                    device=device,
                    overwrite=args.overwrite_features,
                )
                x_model, feature_map, train_df = filter_features_by_train_df(
                    x_sae,
                    idx["train"],
                    min_train_df=args.min_train_df,
                )
                trained = train_one_config(x_model, meta_rows, labels, c_grid)
                record = {
                    "sae_name": sae_path.name,
                    "sae_path": str(sae_path),
                    "top_k": top_k,
                    "region": region,
                    "layer": layer,
                    "best_C": trained["best_C"],
                    "best_dev_metrics": trained["best_dev_metrics"],
                    "split_metrics": trained["split_metrics"],
                    "searches": trained["searches"],
                    "density": float(x_sae.nnz / (x_sae.shape[0] * x_sae.shape[1])),
                    "nnz": int(x_sae.nnz),
                    "min_train_df": int(args.min_train_df),
                    "feature_count_before_df_filter": int(x_sae.shape[1]),
                    "feature_count_after_df_filter": int(x_model.shape[1]),
                }
                results.append(
                    {
                        **record,
                        "_clf": trained["final_clf"],
                        "_x": x_model,
                        "_feature_map": feature_map,
                        "_train_df": train_df,
                    }
                )
                print(
                    "dev BA="
                    f"{trained['best_dev_metrics']['balanced_accuracy']:.3f}; "
                    f"test BA={trained['split_metrics']['test']['balanced_accuracy']:.3f}; "
                    f"C={trained['best_C']}",
                    flush=True,
                )

    results.sort(
        key=lambda row: (
            row["best_dev_metrics"]["balanced_accuracy"],
            row["best_dev_metrics"]["f1"],
            row["split_metrics"]["test"]["balanced_accuracy"],
        ),
        reverse=True,
    )
    best = results[0]
    train_dev_idx = np.concatenate([idx["train"], idx["dev"]])
    top_features = top_feature_rows(
        clf=best["_clf"],
        x_matrix=best["_x"],
        labels=labels,
        train_dev_idx=train_dev_idx,
        feature_map=best["_feature_map"],
        train_df=best["_train_df"],
        top_n=args.top_features,
        label_names=label_names,
    )
    stress = stress_tests(
        x_matrix=best["_x"],
        meta_rows=meta_rows,
        labels=labels,
        c_value=float(best["best_C"]),
        min_eval_per_label=args.stress_min_eval_per_label,
    )

    serializable_results: list[dict[str, Any]] = []
    for row in results:
        clean = {key: value for key, value in row.items() if not key.startswith("_")}
        serializable_results.append(clean)
    write_jsonl(out_dir / "sae_layer_region_search.jsonl", serializable_results)
    write_jsonl(out_dir / "top_sae_features.jsonl", top_features)
    write_json(out_dir / "summary.json", {"best": serializable_results[0], "top_features": top_features, "stress_tests": stress})

    manifest["finished_at"] = now_iso()
    manifest["best"] = serializable_results[0]
    manifest["stress_tests"] = stress
    write_json(out_dir / "manifest.json", manifest)
    write_report(
        out_dir / "report.md",
        manifest=manifest,
        results=serializable_results,
        best_result=serializable_results[0],
        top_features=top_features,
        stress=stress,
    )
    print(f"\nWrote {out_dir / 'report.md'}", flush=True)


if __name__ == "__main__":
    main()
