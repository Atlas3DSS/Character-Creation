#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler


DEFAULT_FEATURES_DIR = (
    "/home/orwel/dev_genius/experiments/Character Creation/sweep_v4/"
    "symphonic_voice_activation_probe_v1b_capped_20260417_204049"
)
DEFAULT_MANIFEST = "/home/orwel/dev_genius/experiments/Character Creation/data/symphonic_voice_anchor_manifest_v1.json"
DEFAULT_OUTPUT_ROOT = "/home/orwel/dev_genius/experiments/Character Creation/sweep_v4"
DEFAULT_TAG = "symphonic_voice_axis_analysis_v1"


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


def parse_region_arrays(npz_path: Path) -> dict[str, dict[int, np.ndarray]]:
    arrays = np.load(npz_path)
    region_arrays: dict[str, dict[int, np.ndarray]] = {}
    pat = re.compile(r"^(?P<region>.+)__L(?P<layer>\d+)$")
    for key in arrays.files:
        m = pat.match(key)
        if not m:
            continue
        region = m.group("region")
        layer = int(m.group("layer"))
        region_arrays.setdefault(region, {})[layer] = arrays[key]
    return region_arrays


def filter_region_arrays(
    region_arrays: dict[str, dict[int, np.ndarray]],
    *,
    region_allowlist: list[str] | None,
    layer_stride: int,
) -> dict[str, dict[int, np.ndarray]]:
    out: dict[str, dict[int, np.ndarray]] = {}
    for region_name, layer_map in region_arrays.items():
        if region_allowlist is not None and region_name not in region_allowlist:
            continue
        max_layer = max(layer_map)
        kept = {layer: arr for layer, arr in layer_map.items() if layer % layer_stride == 0 or layer == max_layer}
        if kept:
            out[region_name] = kept
    return out


def split_indices(meta_rows: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    buckets: dict[str, list[int]] = defaultdict(list)
    for idx, row in enumerate(meta_rows):
        buckets[row["split"]].append(idx)
    return {split: np.array(idxs, dtype=np.int64) for split, idxs in buckets.items()}


def pearson_corr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) < 2:
        return 0.0
    std_true = float(np.std(y_true))
    std_pred = float(np.std(y_pred))
    if std_true == 0.0 or std_pred == 0.0:
        return 0.0
    return float(np.corrcoef(y_true, y_pred)[0, 1])


def rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    i = 0
    while i < len(values):
        j = i
        while j + 1 < len(values) and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg_rank = 0.5 * (i + j) + 1.0
        ranks[order[i : j + 1]] = avg_rank
        i = j + 1
    return ranks


def spearman_corr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return pearson_corr(rankdata(y_true), rankdata(y_pred))


def regression_metrics(rows: list[dict[str, Any]], y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    grouped: dict[str, list[int]] = defaultdict(list)
    for idx, row in enumerate(rows):
        grouped[row["anchor_id"]].append(idx)
    by_anchor: dict[str, Any] = {}
    for anchor_id, idxs in sorted(grouped.items()):
        yt = y_true[idxs]
        yp = y_pred[idxs]
        by_anchor[anchor_id] = {
            "n": int(len(idxs)),
            "mean_true": float(np.mean(yt)),
            "mean_pred": float(np.mean(yp)),
            "mae": float(mean_absolute_error(yt, yp)),
        }
    return {
        "n": int(len(rows)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
        "pearson": pearson_corr(y_true, y_pred),
        "spearman": spearman_corr(y_true, y_pred),
        "by_anchor": by_anchor,
    }


def normed(v: np.ndarray) -> np.ndarray:
    denom = float(np.linalg.norm(v))
    if denom == 0.0:
        return np.zeros_like(v)
    return v / denom


def cosine_and_angle(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    an = normed(a)
    bn = normed(b)
    cos = float(np.clip(np.dot(an, bn), -1.0, 1.0))
    angle = float(np.degrees(np.arccos(cos)))
    return cos, angle


def fit_axis_search(
    axis_name: str,
    targets: np.ndarray,
    region_arrays: dict[str, dict[int, np.ndarray]],
    meta_rows: list[dict[str, Any]],
    idxs: dict[str, np.ndarray],
    ridge_alphas: list[float],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    train_idx = idxs["train"]
    val_idx = idxs["val"]
    search_rows: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None

    for region_name, layer_map in region_arrays.items():
        for layer_idx, X in layer_map.items():
            X_train = X[train_idx]
            X_val = X[val_idx]
            scaler = StandardScaler().fit(X_train)
            Z_train = scaler.transform(X_train)
            Z_val = scaler.transform(X_val)
            y_train = targets[train_idx]
            y_val = targets[val_idx]
            val_rows = [meta_rows[i] for i in val_idx.tolist()]
            for alpha in ridge_alphas:
                reg = Ridge(alpha=alpha)
                reg.fit(Z_train, y_train)
                val_pred = reg.predict(Z_val)
                val_metrics = regression_metrics(val_rows, y_val, val_pred)
                record = {
                    "axis": axis_name,
                    "region": region_name,
                    "layer": int(layer_idx),
                    "alpha": float(alpha),
                    "val_metrics": val_metrics,
                }
                search_rows.append(record)
                if best is None:
                    best = record
                else:
                    a = val_metrics
                    b = best["val_metrics"]
                    if (
                        a["r2"] > b["r2"]
                        or (a["r2"] == b["r2"] and a["pearson"] > b["pearson"])
                        or (
                            a["r2"] == b["r2"]
                            and a["pearson"] == b["pearson"]
                            and a["mae"] < b["mae"]
                        )
                    ):
                        best = record
    assert best is not None
    search_rows.sort(
        key=lambda row: (
            row["val_metrics"]["r2"],
            row["val_metrics"]["pearson"],
            -row["val_metrics"]["mae"],
            row["region"],
            row["layer"],
        ),
        reverse=True,
    )
    return best, search_rows


def fit_final_axis_model(
    *,
    axis_name: str,
    targets: np.ndarray,
    X: np.ndarray,
    meta_rows: list[dict[str, Any]],
    idxs: dict[str, np.ndarray],
    alpha: float,
) -> dict[str, Any]:
    train_idx = idxs["train"]
    val_idx = idxs["val"]
    test_idx = idxs["test"]
    trainval_idx = np.concatenate([train_idx, val_idx], axis=0)

    scaler = StandardScaler().fit(X[trainval_idx])
    Z_train = scaler.transform(X[train_idx])
    Z_val = scaler.transform(X[val_idx])
    Z_trainval = scaler.transform(X[trainval_idx])
    Z_test = scaler.transform(X[test_idx])

    reg_train = Ridge(alpha=alpha)
    reg_train.fit(Z_train, targets[train_idx])
    reg_final = Ridge(alpha=alpha)
    reg_final.fit(Z_trainval, targets[trainval_idx])

    train_rows = [meta_rows[i] for i in train_idx.tolist()]
    val_rows = [meta_rows[i] for i in val_idx.tolist()]
    test_rows = [meta_rows[i] for i in test_idx.tolist()]

    train_pred = reg_train.predict(Z_train)
    val_pred = reg_train.predict(Z_val)
    test_pred = reg_final.predict(Z_test)

    return {
        "axis": axis_name,
        "alpha": float(alpha),
        "scaler_mean": scaler.mean_.astype(np.float32),
        "scaler_scale": scaler.scale_.astype(np.float32),
        "coef_z": reg_final.coef_.astype(np.float32),
        "intercept": float(reg_final.intercept_),
        "train_metrics": regression_metrics(train_rows, targets[train_idx], train_pred),
        "val_metrics": regression_metrics(val_rows, targets[val_idx], val_pred),
        "test_metrics": regression_metrics(test_rows, targets[test_idx], test_pred),
        "test_predictions": [
            {
                "feature_id": row["feature_id"],
                "split": "test",
                "behavior": row["behavior"],
                "anchor_id": row["anchor_id"],
                "y_true": float(y_true),
                "y_pred": float(y_hat),
                "abs_error": float(abs(y_true - y_hat)),
            }
            for row, y_true, y_hat in zip(test_rows, targets[test_idx].tolist(), test_pred.tolist())
        ],
    }


def build_anchor_axes(anchor_manifest: dict[str, Any]) -> tuple[list[str], dict[str, dict[str, float]]]:
    anchors = anchor_manifest["anchors"]
    axis_names = sorted(anchors[0]["stance_axes"].keys())
    anchor_axes: dict[str, dict[str, float]] = {}
    for anchor in anchors:
        anchor_axes[anchor["anchor_id"]] = {axis: float(anchor["stance_axes"][axis]) for axis in axis_names}
    return axis_names, anchor_axes


def build_targets(meta_rows: list[dict[str, Any]], anchor_axes: dict[str, dict[str, float]], axis_name: str) -> np.ndarray:
    return np.array([anchor_axes[row["anchor_id"]][axis_name] for row in meta_rows], dtype=np.float64)


def build_common_anchor_classifier(X: np.ndarray, labels: np.ndarray, idxs: dict[str, np.ndarray], C: float) -> dict[str, Any]:
    train_idx = idxs["train"]
    val_idx = idxs["val"]
    test_idx = idxs["test"]
    trainval_idx = np.concatenate([train_idx, val_idx], axis=0)
    scaler = StandardScaler().fit(X[trainval_idx])
    Z_trainval = scaler.transform(X[trainval_idx])
    Z_test = scaler.transform(X[test_idx])
    clf = LogisticRegression(max_iter=4000, solver="lbfgs", C=C, class_weight="balanced")
    clf.fit(Z_trainval, labels[trainval_idx])
    test_probs = clf.predict_proba(Z_test)
    test_preds = np.argmax(test_probs, axis=1)
    return {
        "scaler_mean": scaler.mean_.astype(np.float32),
        "scaler_scale": scaler.scale_.astype(np.float32),
        "coef": clf.coef_.astype(np.float32),
        "intercept": clf.intercept_.astype(np.float32),
        "classes": clf.classes_.astype(np.int64),
        "test_accuracy": float(accuracy_score(labels[test_idx], test_preds)),
        "test_probs": test_probs,
        "test_preds": test_preds,
        "scaler": scaler,
        "clf": clf,
    }


def predict_multiclass_from_z(model: dict[str, Any], Z: np.ndarray) -> np.ndarray:
    clf: LogisticRegression = model["clf"]
    return clf.predict_proba(Z)


def predict_ridge_from_z(axis_model: dict[str, Any], Z: np.ndarray) -> np.ndarray:
    coef = axis_model["coef_z"]
    intercept = axis_model["intercept"]
    return Z @ coef + intercept


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features-dir", type=Path, default=Path(DEFAULT_FEATURES_DIR))
    ap.add_argument("--anchor-manifest", type=Path, default=Path(DEFAULT_MANIFEST))
    ap.add_argument("--output-root", type=Path, default=Path(DEFAULT_OUTPUT_ROOT))
    ap.add_argument("--tag", default=DEFAULT_TAG)
    ap.add_argument("--region-allowlist", default="think_mean,assistant_mean,response_mean,prompt_last")
    ap.add_argument("--layer-stride", type=int, default=2)
    ap.add_argument("--ridge-alphas", default="0.1,1.0,10.0,100.0")
    ap.add_argument("--common-region", default="think_mean")
    ap.add_argument("--common-layer", type=int, default=39)
    ap.add_argument("--common-clf-c", type=float, default=0.25)
    ap.add_argument("--patch-alphas", default="0.25,0.5,1.0")
    args = ap.parse_args()

    stamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    out_dir = args.output_root / f"{args.tag}_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    meta_rows = load_jsonl(args.features_dir / "feature_meta.jsonl")
    label_map = load_json(args.features_dir / "label_map.json")
    region_arrays = parse_region_arrays(args.features_dir / "features.npz")
    region_allowlist = [x.strip() for x in args.region_allowlist.split(",") if x.strip()]
    region_arrays = filter_region_arrays(
        region_arrays,
        region_allowlist=region_allowlist or None,
        layer_stride=max(1, int(args.layer_stride)),
    )
    anchor_manifest = load_json(args.anchor_manifest)
    axis_names, anchor_axes = build_anchor_axes(anchor_manifest)
    idxs = split_indices(meta_rows)
    labels = np.array([int(row["anchor_label"]) for row in meta_rows], dtype=np.int64)
    anchor_ids = label_map["anchor_ids"]
    anchor_to_label = label_map["anchor_to_label"]
    ridge_alphas = [float(x) for x in args.ridge_alphas.split(",") if x.strip()]
    patch_alphas = [float(x) for x in args.patch_alphas.split(",") if x.strip()]

    write_json(
        out_dir / "manifest.json",
        {
            "started_at": now_iso(),
            "features_dir": str(args.features_dir),
            "anchor_manifest": str(args.anchor_manifest),
            "region_allowlist": region_allowlist,
            "layer_stride": args.layer_stride,
            "ridge_alphas": ridge_alphas,
            "common_region": args.common_region,
            "common_layer": args.common_layer,
            "common_clf_c": args.common_clf_c,
            "patch_alphas": patch_alphas,
            "n_examples": len(meta_rows),
        },
    )

    axis_search_rows: list[dict[str, Any]] = []
    axis_best_summary: dict[str, Any] = {}
    axis_models_common: dict[str, Any] = {}
    axis_vector_npz: dict[str, np.ndarray] = {}

    for axis_name in axis_names:
        targets = build_targets(meta_rows, anchor_axes, axis_name)
        best, searches = fit_axis_search(axis_name, targets, region_arrays, meta_rows, idxs, ridge_alphas)
        axis_search_rows.extend(searches)

        best_region = best["region"]
        best_layer = int(best["layer"])
        best_alpha = float(best["alpha"])
        best_model = fit_final_axis_model(
            axis_name=axis_name,
            targets=targets,
            X=region_arrays[best_region][best_layer],
            meta_rows=meta_rows,
            idxs=idxs,
            alpha=best_alpha,
        )
        axis_best_summary[axis_name] = {
            "best_region": best_region,
            "best_layer": best_layer,
            "best_alpha": best_alpha,
            "val_metrics": best["val_metrics"],
            "train_metrics": best_model["train_metrics"],
            "val_metrics_refit": best_model["val_metrics"],
            "test_metrics": best_model["test_metrics"],
        }
        write_jsonl(out_dir / f"{axis_name}_test_predictions.jsonl", best_model["test_predictions"])

        common_targets = targets
        common_model = fit_final_axis_model(
            axis_name=axis_name,
            targets=common_targets,
            X=region_arrays[args.common_region][args.common_layer],
            meta_rows=meta_rows,
            idxs=idxs,
            alpha=best_alpha,
        )
        axis_models_common[axis_name] = common_model
        axis_vector_npz[f"{axis_name}__coef_z"] = common_model["coef_z"]

    write_jsonl(out_dir / "axis_searches.jsonl", axis_search_rows)
    write_json(out_dir / "axis_summary.json", axis_best_summary)
    np.savez_compressed(out_dir / "axis_vectors_common_space.npz", **axis_vector_npz)

    X_common = region_arrays[args.common_region][args.common_layer]
    common_clf = build_common_anchor_classifier(
        X_common,
        labels,
        idxs,
        C=args.common_clf_c,
    )
    scaler_common: StandardScaler = common_clf["scaler"]
    Z_trainval = scaler_common.transform(X_common[np.concatenate([idxs["train"], idxs["val"]], axis=0)])
    meta_trainval = [meta_rows[i] for i in np.concatenate([idxs["train"], idxs["val"]], axis=0).tolist()]
    Z_test = scaler_common.transform(X_common[idxs["test"]])
    meta_test = [meta_rows[i] for i in idxs["test"].tolist()]

    centroids: dict[str, np.ndarray] = {}
    for anchor_id in anchor_ids:
        idx_local = [i for i, row in enumerate(meta_trainval) if row["anchor_id"] == anchor_id]
        centroids[anchor_id] = Z_trainval[idx_local].mean(axis=0)

    pair_angle_rows: list[dict[str, Any]] = []
    for src in anchor_ids:
        for dst in anchor_ids:
            if src == dst:
                continue
            delta = centroids[dst] - centroids[src]
            row = {
                "source_anchor": src,
                "target_anchor": dst,
                "delta_norm": float(np.linalg.norm(delta)),
                "axis_alignment": {},
            }
            for axis_name, axis_model in axis_models_common.items():
                cos, angle = cosine_and_angle(axis_model["coef_z"], delta)
                row["axis_alignment"][axis_name] = {"cosine": cos, "angle_deg": angle}
            pair_angle_rows.append(row)
    write_jsonl(out_dir / "pairwise_anchor_axis_angles.jsonl", pair_angle_rows)

    selected_pairs = [
        ("hitchens", "linus"),
        ("linus", "hitchens"),
        ("hitchens", "jesus"),
        ("jesus", "hitchens"),
        ("jesus", "mother_teresa"),
        ("mother_teresa", "jesus"),
        ("neutral_competent", "linus"),
        ("neutral_competent", "mother_teresa"),
    ]
    patch_rows: list[dict[str, Any]] = []
    anchor_test_indices: dict[str, list[int]] = defaultdict(list)
    for i, row in enumerate(meta_test):
        anchor_test_indices[row["anchor_id"]].append(i)
    base_probs = predict_multiclass_from_z(common_clf, Z_test)
    base_axis_preds = {
        axis_name: predict_ridge_from_z(axis_model, Z_test) for axis_name, axis_model in axis_models_common.items()
    }
    for src, dst in selected_pairs:
        if src not in centroids or dst not in centroids or src not in anchor_test_indices:
            continue
        delta = centroids[dst] - centroids[src]
        src_idxs = np.array(anchor_test_indices[src], dtype=np.int64)
        z_src = Z_test[src_idxs]
        before_probs = base_probs[src_idxs]
        patch_row_base = {
            "source_anchor": src,
            "target_anchor": dst,
            "n_examples": int(len(src_idxs)),
            "delta_norm": float(np.linalg.norm(delta)),
        }
        for alpha in patch_alphas:
            z_patched = z_src + alpha * delta
            probs_after = predict_multiclass_from_z(common_clf, z_patched)
            preds_after = np.argmax(probs_after, axis=1)
            src_label = int(anchor_to_label[src])
            dst_label = int(anchor_to_label[dst])
            row = dict(patch_row_base)
            row["alpha"] = float(alpha)
            row["mean_prob_source_before"] = float(np.mean(before_probs[:, src_label]))
            row["mean_prob_source_after"] = float(np.mean(probs_after[:, src_label]))
            row["mean_prob_target_before"] = float(np.mean(before_probs[:, dst_label]))
            row["mean_prob_target_after"] = float(np.mean(probs_after[:, dst_label]))
            row["pred_target_rate_after"] = float(np.mean(preds_after == dst_label))
            row["pred_source_rate_after"] = float(np.mean(preds_after == src_label))
            row["axis_delta_means"] = {}
            for axis_name, axis_model in axis_models_common.items():
                before_axis = base_axis_preds[axis_name][src_idxs]
                after_axis = predict_ridge_from_z(axis_model, z_patched)
                row["axis_delta_means"][axis_name] = float(np.mean(after_axis - before_axis))
            patch_rows.append(row)
    write_jsonl(out_dir / "feature_space_patch_summary.jsonl", patch_rows)

    top_pair_rows = []
    for pair_name in [("hitchens", "linus"), ("jesus", "mother_teresa"), ("neutral_competent", "linus")]:
        src, dst = pair_name
        match = next((row for row in pair_angle_rows if row["source_anchor"] == src and row["target_anchor"] == dst), None)
        if match is not None:
            align = sorted(
                (
                    {
                        "axis": axis_name,
                        "cosine": info["cosine"],
                        "angle_deg": info["angle_deg"],
                    }
                    for axis_name, info in match["axis_alignment"].items()
                ),
                key=lambda item: abs(item["cosine"]),
                reverse=True,
            )
            top_pair_rows.append(
                {
                    "source_anchor": src,
                    "target_anchor": dst,
                    "delta_norm": match["delta_norm"],
                    "top_axis_alignments": align[:3],
                }
            )

    summary = {
        "finished_at": now_iso(),
        "axis_best_summary": axis_best_summary,
        "common_space": {
            "region": args.common_region,
            "layer": args.common_layer,
            "multiclass_test_accuracy": common_clf["test_accuracy"],
            "top_pair_alignments": top_pair_rows,
        },
        "patch_examples": patch_rows,
    }
    write_json(out_dir / "summary.json", summary)

    report_lines = [
        "# Symphonic Voice Axis Analysis",
        "",
        f"- Finished: `{summary['finished_at']}`",
        f"- Common comparison space: `{args.common_region} @ L{args.common_layer}`",
        f"- Common multiclass test accuracy: `{common_clf['test_accuracy']:.3f}`",
        "",
        "## Best Axis Readouts",
        "",
        "| Axis | Best Region | Best Layer | Val R2 | Test R2 | Test Pearson | Test MAE |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for axis_name in axis_names:
        row = axis_best_summary[axis_name]
        report_lines.append(
            f"| {axis_name} | `{row['best_region']}` | `{row['best_layer']}` | "
            f"`{row['val_metrics']['r2']:.3f}` | `{row['test_metrics']['r2']:.3f}` | "
            f"`{row['test_metrics']['pearson']:.3f}` | `{row['test_metrics']['mae']:.3f}` |"
        )
    report_lines.extend(
        [
            "",
            "## Notable Anchor Delta Alignments",
            "",
            "| Direction | Top Axes |",
            "| --- | --- |",
        ]
    )
    for row in top_pair_rows:
        top = ", ".join(
            f"{item['axis']} ({item['cosine']:.3f}, {item['angle_deg']:.1f}deg)" for item in row["top_axis_alignments"]
        )
        report_lines.append(f"| `{row['source_anchor']} -> {row['target_anchor']}` | {top} |")
    report_lines.extend(
        [
            "",
            "## Feature-Space Patch Examples",
            "",
            "| Direction | Alpha | Src Prob Before | Src Prob After | Tgt Prob Before | Tgt Prob After | Target Rate After |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in patch_rows:
        if row["alpha"] not in (0.5, 1.0):
            continue
        report_lines.append(
            f"| `{row['source_anchor']} -> {row['target_anchor']}` | `{row['alpha']:.2f}` | "
            f"`{row['mean_prob_source_before']:.3f}` | `{row['mean_prob_source_after']:.3f}` | "
            f"`{row['mean_prob_target_before']:.3f}` | `{row['mean_prob_target_after']:.3f}` | "
            f"`{row['pred_target_rate_after']:.3f}` |"
        )
    (out_dir / "report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
