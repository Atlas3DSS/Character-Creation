#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler


DEFAULT_FEATURES_DIR = (
    "/home/orwel/dev_genius/experiments/Character Creation/sweep_v4/"
    "symphonic_voice_activation_probe_v2_repaired_20260418_073715"
)
DEFAULT_ANCHOR_MANIFEST = "/home/orwel/dev_genius/experiments/Character Creation/data/symphonic_voice_anchor_manifest_v2.json"
DEFAULT_LIVE_PATCH_DIR = (
    "/home/orwel/dev_genius/experiments/Character Creation/sweep_v4/"
    "symphonic_voice_live_patch_v2_compositional_20260418_082730"
)
DEFAULT_OUTPUT_ROOT = "/home/orwel/dev_genius/experiments/Character Creation/sweep_v4"


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


def pearson_corr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) < 2 or float(np.std(y_true)) == 0.0 or float(np.std(y_pred)) == 0.0:
        return 0.0
    return float(np.corrcoef(y_true, y_pred)[0, 1])


def parse_region_specs(raw: str) -> list[tuple[str, int]]:
    specs: list[tuple[str, int]] = []
    for part in [x.strip() for x in raw.split(",") if x.strip()]:
        region, layer_s = part.split(":", 1)
        specs.append((region.strip(), int(layer_s.strip())))
    return specs


def parse_pairs(raw: str) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for part in [x.strip() for x in raw.split(",") if x.strip()]:
        src, dst = [x.strip() for x in part.split(":", 1)]
        pairs.append((src, dst))
    return pairs


def npz_key(arrays: np.lib.npyio.NpzFile, region: str, layer: int) -> str:
    padded = f"{region}__L{layer:02d}"
    plain = f"{region}__L{layer}"
    if padded in arrays.files:
        return padded
    if plain in arrays.files:
        return plain
    raise KeyError(f"missing feature array for {region}@L{layer}")


def build_anchor_axes(anchor_manifest: dict[str, Any]) -> tuple[list[str], dict[str, dict[str, float]]]:
    anchors = anchor_manifest["anchors"]
    axis_names = sorted(anchors[0]["stance_axes"].keys())
    axes = {
        anchor["anchor_id"]: {axis: float(anchor["stance_axes"][axis]) for axis in axis_names}
        for anchor in anchors
    }
    return axis_names, axes


def split_indices(meta_rows: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    buckets: dict[str, list[int]] = defaultdict(list)
    for idx, row in enumerate(meta_rows):
        buckets[row["split"]].append(idx)
    return {key: np.array(vals, dtype=np.int64) for key, vals in buckets.items()}


def fit_region_classifier(
    *,
    X: np.ndarray,
    meta_rows: list[dict[str, Any]],
    labels: np.ndarray,
    anchor_ids: list[str],
    anchor_to_label: dict[str, int],
    idxs: dict[str, np.ndarray],
    clf_c: float,
) -> dict[str, Any]:
    trainval_idx = np.concatenate([idxs["train"], idxs["val"]], axis=0)
    test_idx = idxs["test"]
    scaler = StandardScaler().fit(X[trainval_idx])
    Z = scaler.transform(X)
    clf = LogisticRegression(max_iter=4000, solver="lbfgs", C=clf_c, class_weight="balanced")
    clf.fit(Z[trainval_idx], labels[trainval_idx])
    test_probs = clf.predict_proba(Z[test_idx])
    test_preds = np.argmax(test_probs, axis=1)
    centroids: dict[str, np.ndarray] = {}
    for anchor_id in anchor_ids:
        local = [i for i in trainval_idx.tolist() if meta_rows[i]["anchor_id"] == anchor_id]
        centroids[anchor_id] = Z[local].mean(axis=0)

    by_anchor: dict[str, Any] = {}
    for anchor_id in anchor_ids:
        local_test = [j for j, i in enumerate(test_idx.tolist()) if meta_rows[i]["anchor_id"] == anchor_id]
        label = int(anchor_to_label[anchor_id])
        probs = test_probs[local_test]
        sorted_probs = np.sort(probs, axis=1)
        by_anchor[anchor_id] = {
            "n": int(len(local_test)),
            "self_prob": float(np.mean(probs[:, label])),
            "margin": float(np.mean(sorted_probs[:, -1] - sorted_probs[:, -2])),
            "centroid_norm": float(np.linalg.norm(centroids[anchor_id])),
        }
    return {
        "test_accuracy": float(accuracy_score(labels[test_idx], test_preds)),
        "by_anchor": by_anchor,
        "centroids": centroids,
        "scaler": scaler,
        "Z": Z,
    }


def fit_axis_at_region(
    *,
    X: np.ndarray,
    meta_rows: list[dict[str, Any]],
    idxs: dict[str, np.ndarray],
    anchor_axes: dict[str, dict[str, float]],
    axis_name: str,
    alpha: float,
) -> dict[str, Any]:
    train_idx = idxs["train"]
    val_idx = idxs["val"]
    test_idx = idxs["test"]
    trainval_idx = np.concatenate([train_idx, val_idx], axis=0)
    targets = np.array([anchor_axes[row["anchor_id"]][axis_name] for row in meta_rows], dtype=np.float64)
    scaler = StandardScaler().fit(X[trainval_idx])
    reg = Ridge(alpha=alpha)
    reg.fit(scaler.transform(X[trainval_idx]), targets[trainval_idx])
    pred = reg.predict(scaler.transform(X[test_idx]))
    return {
        "r2": float(r2_score(targets[test_idx], pred)),
        "pearson": pearson_corr(targets[test_idx], pred),
        "mae": float(mean_absolute_error(targets[test_idx], pred)),
    }


def summarize_live_patch(live_patch_dir: Path) -> dict[str, Any]:
    summary_path = live_patch_dir / "summary.json"
    records_path = live_patch_dir / "records.jsonl"
    if not summary_path.exists() or not records_path.exists():
        return {}
    records = load_jsonl(records_path)
    out: dict[str, Any] = {}
    for pair_name in sorted({row["pair_name"] for row in records}):
        rows = [row for row in records if row["pair_name"] == pair_name]
        baselines = [row for row in rows if row["condition"] == "baseline"]
        patches = [row for row in rows if row["condition"] != "baseline"]
        if not baselines or not patches:
            continue
        max_alpha = max(float(row["alpha"]) for row in patches)
        chosen = [row for row in patches if float(row["alpha"]) == max_alpha]
        base_by_item = {row["item_key"]: row for row in baselines}
        target_lifts: list[float] = []
        source_deltas: list[float] = []
        axis_deltas: dict[str, list[float]] = defaultdict(list)
        for row in chosen:
            base = base_by_item.get(row["item_key"])
            if base is None:
                continue
            if row.get("target_prob") is not None and base.get("target_prob") is not None:
                target_lifts.append(float(row["target_prob"]) - float(base["target_prob"]))
            if row.get("source_prob") is not None and base.get("source_prob") is not None:
                source_deltas.append(float(row["source_prob"]) - float(base["source_prob"]))
            for axis, value in row.get("axis_predictions", {}).items():
                if axis in base.get("axis_predictions", {}):
                    axis_deltas[axis].append(float(value) - float(base["axis_predictions"][axis]))
        out[pair_name] = {
            "max_alpha": max_alpha,
            "n": len(chosen),
            "target_rate": float(np.mean([row.get("pred_anchor_id") == row.get("target_anchor") for row in chosen])),
            "source_rate": float(np.mean([row.get("pred_anchor_id") == row.get("source_anchor") for row in chosen])),
            "mean_target_prob_lift": float(np.mean(target_lifts)) if target_lifts else None,
            "mean_source_prob_delta": float(np.mean(source_deltas)) if source_deltas else None,
            "axis_deltas": {axis: float(np.mean(vals)) for axis, vals in sorted(axis_deltas.items()) if vals},
            "mean_patched_tokens": float(np.mean([row.get("patched_tokens", 0) for row in chosen])),
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="CPU sanity checks for reverse compassion patch failures.")
    ap.add_argument("--features-dir", type=Path, default=Path(DEFAULT_FEATURES_DIR))
    ap.add_argument("--anchor-manifest", type=Path, default=Path(DEFAULT_ANCHOR_MANIFEST))
    ap.add_argument("--live-patch-dir", type=Path, default=Path(DEFAULT_LIVE_PATCH_DIR))
    ap.add_argument("--output-root", type=Path, default=Path(DEFAULT_OUTPUT_ROOT))
    ap.add_argument("--tag", default="reverse_basin_sanity_v1")
    ap.add_argument("--regions", default="prompt_last:0,think_mean:39,assistant_mean:16,response_mean:20")
    ap.add_argument("--pairs", default="hitchens:jesus,hitchens:mother_teresa,jesus:hitchens,jesus:mother_teresa")
    ap.add_argument("--clf-c", type=float, default=0.25)
    ap.add_argument("--ridge-alpha", type=float, default=1.0)
    args = ap.parse_args()

    stamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    out_dir = args.output_root / f"{args.tag}_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    meta_rows = load_jsonl(args.features_dir / "feature_meta.jsonl")
    label_map = load_json(args.features_dir / "label_map.json")
    anchor_manifest = load_json(args.anchor_manifest)
    axis_names, anchor_axes = build_anchor_axes(anchor_manifest)
    labels = np.array([int(row["anchor_label"]) for row in meta_rows], dtype=np.int64)
    idxs = split_indices(meta_rows)
    arrays = np.load(args.features_dir / "features.npz")
    anchor_ids = label_map["anchor_ids"]
    anchor_to_label = {key: int(val) for key, val in label_map["anchor_to_label"].items()}

    region_specs = parse_region_specs(args.regions)
    pairs = parse_pairs(args.pairs)
    regions: dict[str, Any] = {}
    pair_rows: list[dict[str, Any]] = []
    axis_rows: list[dict[str, Any]] = []

    for region, layer in region_specs:
        X = arrays[npz_key(arrays, region, layer)]
        fit = fit_region_classifier(
            X=X,
            meta_rows=meta_rows,
            labels=labels,
            anchor_ids=anchor_ids,
            anchor_to_label=anchor_to_label,
            idxs=idxs,
            clf_c=args.clf_c,
        )
        region_key = f"{region}@L{layer}"
        centroids = fit.pop("centroids")
        fit.pop("scaler")
        fit.pop("Z")
        regions[region_key] = fit
        for src, dst in pairs:
            if src not in centroids or dst not in centroids:
                continue
            src_info = fit["by_anchor"][src]
            dst_info = fit["by_anchor"][dst]
            pair_rows.append(
                {
                    "region": region,
                    "layer": layer,
                    "source_anchor": src,
                    "target_anchor": dst,
                    "delta_norm": float(np.linalg.norm(centroids[dst] - centroids[src])),
                    "source_centroid_norm": src_info["centroid_norm"],
                    "target_centroid_norm": dst_info["centroid_norm"],
                    "target_minus_source_centroid_norm": dst_info["centroid_norm"] - src_info["centroid_norm"],
                    "source_self_prob": src_info["self_prob"],
                    "target_self_prob": dst_info["self_prob"],
                    "target_minus_source_self_prob": dst_info["self_prob"] - src_info["self_prob"],
                    "source_margin": src_info["margin"],
                    "target_margin": dst_info["margin"],
                    "target_minus_source_margin": dst_info["margin"] - src_info["margin"],
                }
            )
        for axis_name in axis_names:
            metrics = fit_axis_at_region(
                X=X,
                meta_rows=meta_rows,
                idxs=idxs,
                anchor_axes=anchor_axes,
                axis_name=axis_name,
                alpha=args.ridge_alpha,
            )
            axis_rows.append({"region": region, "layer": layer, "axis": axis_name, **metrics})

    live_patch = summarize_live_patch(args.live_patch_dir)
    summary = {
        "finished_at": now_iso(),
        "features_dir": str(args.features_dir),
        "anchor_manifest": str(args.anchor_manifest),
        "live_patch_dir": str(args.live_patch_dir),
        "regions": regions,
        "pair_rows": pair_rows,
        "axis_rows": axis_rows,
        "live_patch": live_patch,
    }
    write_json(out_dir / "summary.json", summary)
    write_jsonl(out_dir / "pair_sanity.jsonl", pair_rows)
    write_jsonl(out_dir / "axis_region_metrics.jsonl", axis_rows)

    report_lines = [
        "# Reverse Basin Sanity",
        "",
        f"- Finished: `{summary['finished_at']}`",
        f"- Features: `{args.features_dir}`",
        f"- Live patch: `{args.live_patch_dir}`",
        "",
        "## Region Separability",
        "",
        "| Region | Test Acc | Hitchens Self | Jesus Self | Mother Teresa Self | Hitchens Norm | Jesus Norm | Mother Teresa Norm |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for region_key, payload in regions.items():
        by_anchor = payload["by_anchor"]
        report_lines.append(
            f"| `{region_key}` | `{payload['test_accuracy']:.3f}` | "
            f"`{by_anchor['hitchens']['self_prob']:.3f}` | `{by_anchor['jesus']['self_prob']:.3f}` | "
            f"`{by_anchor['mother_teresa']['self_prob']:.3f}` | "
            f"`{by_anchor['hitchens']['centroid_norm']:.2f}` | `{by_anchor['jesus']['centroid_norm']:.2f}` | "
            f"`{by_anchor['mother_teresa']['centroid_norm']:.2f}` |"
        )
    report_lines.extend(
        [
            "",
            "## Reverse Pair Checks",
            "",
            "| Region | Pair | Delta Norm | Target-Source Norm | Target-Source Self Prob | Target-Source Margin |",
            "| --- | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in pair_rows:
        if row["source_anchor"] != "hitchens":
            continue
        report_lines.append(
            f"| `{row['region']}@L{row['layer']}` | `{row['source_anchor']} -> {row['target_anchor']}` | "
            f"`{row['delta_norm']:.2f}` | `{row['target_minus_source_centroid_norm']:+.2f}` | "
            f"`{row['target_minus_source_self_prob']:+.3f}` | `{row['target_minus_source_margin']:+.3f}` |"
        )
    report_lines.extend(
        [
            "",
            "## Axis Readouts At Checked Regions",
            "",
            "| Region | Axis | Test R2 | Test Pearson | Test MAE |",
            "| --- | --- | ---: | ---: | ---: |",
        ]
    )
    for row in axis_rows:
        if row["axis"] not in {"compassion", "irony", "severity", "transcendence"}:
            continue
        report_lines.append(
            f"| `{row['region']}@L{row['layer']}` | `{row['axis']}` | "
            f"`{row['r2']:.3f}` | `{row['pearson']:.3f}` | `{row['mae']:.3f}` |"
        )
    if live_patch:
        report_lines.extend(
            [
                "",
                "## Existing Live Patch Summary",
                "",
                "| Pair | Alpha | Target Rate | Source Rate | Target Prob Lift | Compassion Delta | Irony Delta | Patched Tokens |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for pair_name, payload in sorted(live_patch.items()):
            axes = payload.get("axis_deltas", {})
            report_lines.append(
                f"| `{pair_name.replace('__to__', ' -> ')}` | `{payload['max_alpha']:.2f}` | "
                f"`{payload['target_rate']:.3f}` | `{payload['source_rate']:.3f}` | "
                f"`{payload['mean_target_prob_lift']:+.4f}` | "
                f"`{axes.get('compassion', 0.0):+.4f}` | `{axes.get('irony', 0.0):+.4f}` | "
                f"`{payload['mean_patched_tokens']:.1f}` |"
            )
    report_lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- In the checked late-think space, compassionate targets are not obviously lower-norm.",
            "- The existing live reverse patches fail despite nontrivial feature-space separability, so the next falsification target should be timing/dynamics rather than another broad corpus expansion.",
            "- The immediate next GPU run should sweep earlier and narrower patch windows on reverse-compassion rows, with a same-source null and scrambled-direction control.",
        ]
    )
    (out_dir / "report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print(out_dir)


if __name__ == "__main__":
    main()
