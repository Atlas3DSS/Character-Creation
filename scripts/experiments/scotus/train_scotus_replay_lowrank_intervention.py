#!/usr/bin/env python3
"""Train a low-rank replay activation map as a SCOTUS intervention diagnostic.

This is not a fine-tune and not a steering claim. It learns, from cached replay
features only, a low-rank map from source assistant states to paired target
deltas. If the map fails held-out replay checks, it is not worth a full
generation hook run.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from statistics import mean, median
from typing import Any

import numpy as np
from sklearn.metrics import balanced_accuracy_score


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_REPLAY_RUN = PROJECT_ROOT / "sweep_v4" / "scotus_minpair_replay_v2_20260501_144942"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "sweep_v4"


@dataclass(frozen=True)
class PairBatch:
    split: str
    pair_ids: list[str]
    source_authority: np.ndarray
    target_limits: np.ndarray

    @property
    def deltas(self) -> np.ndarray:
        return self.target_limits - self.source_authority


@dataclass(frozen=True)
class LowRankModel:
    rank: int
    ridge: float
    x_mean: np.ndarray
    delta_mean: np.ndarray
    components: np.ndarray
    x_train: np.ndarray
    dual_coef: np.ndarray
    permutation_seed: int | None


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON") from exc
    return rows


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_ints(text: str) -> list[int]:
    return [int(item.strip()) for item in text.split(",") if item.strip()]


def parse_floats(text: str) -> list[float]:
    return [float(item.strip()) for item in text.split(",") if item.strip()]


def feature_key(region: str, layer: int) -> str:
    return f"{region}__L{layer:02d}"


def safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in value).strip("_")


def group_pairs(
    meta: list[dict[str, Any]],
    split: str,
    *,
    pair_field: str,
    label_field: str,
    source_label: str,
    target_label: str,
) -> list[tuple[str, int, int]]:
    grouped: dict[str, dict[str, int]] = {}
    for idx, row in enumerate(meta):
        if str(row.get("split")) != split:
            continue
        pair_id = str(row.get(pair_field) or row.get("fact_id"))
        frame = str(row.get(label_field))
        grouped.setdefault(pair_id, {})[frame] = idx
    pairs: list[tuple[str, int, int]] = []
    for pair_id, labels in sorted(grouped.items()):
        if source_label in labels and target_label in labels:
            pairs.append((pair_id, labels[source_label], labels[target_label]))
    if not pairs:
        raise RuntimeError(f"No paired {source_label}/{target_label} rows for split {split}")
    return pairs


def load_pair_batch(
    features: np.ndarray,
    meta: list[dict[str, Any]],
    split: str,
    *,
    pair_field: str = "pair_id",
    label_field: str = "frame_label",
    source_label: str = "commerce_authority",
    target_label: str = "commerce_limits",
) -> PairBatch:
    pairs = group_pairs(
        meta,
        split,
        pair_field=pair_field,
        label_field=label_field,
        source_label=source_label,
        target_label=target_label,
    )
    pair_ids = [pair_id for pair_id, _authority_idx, _limits_idx in pairs]
    source = np.stack([features[authority_idx] for _pair_id, authority_idx, _limits_idx in pairs]).astype(np.float32)
    target = np.stack([features[limits_idx] for _pair_id, _authority_idx, limits_idx in pairs]).astype(np.float32)
    return PairBatch(split=split, pair_ids=pair_ids, source_authority=source, target_limits=target)


def safe_norms(values: np.ndarray) -> np.ndarray:
    return np.maximum(np.linalg.norm(values, axis=1), 1e-12)


def cosine_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.sum(a * b, axis=1) / (safe_norms(a) * safe_norms(b))


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60.0, 60.0)))


def load_probe(probe_path: Path) -> dict[str, np.ndarray]:
    with np.load(probe_path) as data:
        required = ["scaler_mean", "scaler_scale", "coef", "intercept"]
        missing = [key for key in required if key not in data.files]
        if missing:
            raise KeyError(f"{probe_path} is missing {missing}")
        return {
            "scaler_mean": data["scaler_mean"].astype(np.float32),
            "scaler_scale": data["scaler_scale"].astype(np.float32),
            "coef": data["coef"].reshape(-1).astype(np.float32),
            "intercept": data["intercept"].reshape(-1).astype(np.float32),
        }


def probe_probability(probe: dict[str, np.ndarray], x: np.ndarray) -> np.ndarray:
    z = (x - probe["scaler_mean"]) / np.maximum(probe["scaler_scale"], 1e-12)
    logits = z @ probe["coef"] + float(probe["intercept"][0])
    return sigmoid(logits)


def fit_lowrank(
    batch: PairBatch,
    *,
    rank: int,
    ridge: float,
    permutation_seed: int | None,
) -> LowRankModel:
    x_train = batch.source_authority.astype(np.float32, copy=False)
    deltas = batch.deltas.astype(np.float32, copy=False)
    if permutation_seed is not None:
        rng = np.random.default_rng(permutation_seed)
        deltas = deltas[rng.permutation(deltas.shape[0])]

    x_mean = x_train.mean(axis=0).astype(np.float32)
    delta_mean = deltas.mean(axis=0).astype(np.float32)
    x_centered = x_train - x_mean
    d_centered = deltas - delta_mean

    if rank == 0:
        components = np.zeros((0, x_train.shape[1]), dtype=np.float32)
        dual_coef = np.zeros((x_train.shape[0], 0), dtype=np.float32)
    else:
        _u, _s, vt = np.linalg.svd(d_centered.astype(np.float64, copy=False), full_matrices=False)
        components = vt[:rank].astype(np.float32, copy=False)
        coeffs = (d_centered @ components.T).astype(np.float32, copy=False)
        kernel = x_centered @ x_centered.T
        reg = float(ridge) * np.eye(kernel.shape[0], dtype=np.float32)
        dual_coef = np.linalg.solve(kernel + reg, coeffs).astype(np.float32)

    return LowRankModel(
        rank=rank,
        ridge=float(ridge),
        x_mean=x_mean,
        delta_mean=delta_mean,
        components=components,
        x_train=x_centered.astype(np.float32),
        dual_coef=dual_coef,
        permutation_seed=permutation_seed,
    )


def predict_delta(model: LowRankModel, x: np.ndarray) -> np.ndarray:
    if model.rank == 0:
        return np.repeat(model.delta_mean[None, :], x.shape[0], axis=0).astype(np.float32)
    x_centered = x.astype(np.float32, copy=False) - model.x_mean
    coeff = (x_centered @ model.x_train.T) @ model.dual_coef
    return (model.delta_mean + coeff @ model.components).astype(np.float32)


def evaluate_model(
    model: LowRankModel,
    batch: PairBatch,
    probe: dict[str, np.ndarray],
    *,
    model_name: str,
) -> dict[str, Any]:
    true_delta = batch.deltas.astype(np.float32, copy=False)
    pred_delta = predict_delta(model, batch.source_authority)
    edited = batch.source_authority + pred_delta

    source_mse = float(np.mean((batch.source_authority - batch.target_limits) ** 2))
    edited_mse = float(np.mean((edited - batch.target_limits) ** 2))
    delta_mse = float(np.mean((pred_delta - true_delta) ** 2))
    cos = cosine_rows(pred_delta, true_delta)
    source_prob = probe_probability(probe, batch.source_authority)
    edited_prob = probe_probability(probe, edited)
    target_prob = probe_probability(probe, batch.target_limits)
    source_pred = (source_prob >= 0.5).astype(int)
    edited_pred = (edited_prob >= 0.5).astype(int)
    target_pred = (target_prob >= 0.5).astype(int)
    return {
        "model": model_name,
        "rank": model.rank,
        "ridge": model.ridge,
        "permutation_seed": model.permutation_seed,
        "split": batch.split,
        "n_pairs": len(batch.pair_ids),
        "source_to_target_mse": source_mse,
        "edited_to_target_mse": edited_mse,
        "delta_prediction_mse": delta_mse,
        "mse_improvement_fraction": 0.0 if source_mse <= 0 else float((source_mse - edited_mse) / source_mse),
        "mean_delta_cosine": float(mean(cos.tolist())),
        "median_delta_cosine": float(median(cos.tolist())),
        "mean_pred_delta_norm": float(mean(safe_norms(pred_delta).tolist())),
        "mean_true_delta_norm": float(mean(safe_norms(true_delta).tolist())),
        "mean_source_probe_probability": float(mean(source_prob.tolist())),
        "mean_edited_probe_probability": float(mean(edited_prob.tolist())),
        "mean_target_probe_probability": float(mean(target_prob.tolist())),
        "edited_minus_source_probe_probability": float(mean((edited_prob - source_prob).tolist())),
        "target_minus_source_probe_probability": float(mean((target_prob - source_prob).tolist())),
        "source_probe_balanced_accuracy": float(balanced_accuracy_score(np.zeros_like(source_pred), source_pred)),
        "edited_probe_positive_rate": float(np.mean(edited_pred)),
        "target_probe_balanced_accuracy": float(balanced_accuracy_score(np.ones_like(target_pred), target_pred)),
    }


def save_model(path: Path, model: LowRankModel, manifest: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        x_mean=model.x_mean.astype(np.float32),
        delta_mean=model.delta_mean.astype(np.float32),
        components=model.components.astype(np.float32),
        x_train=model.x_train.astype(np.float32),
        dual_coef=model.dual_coef.astype(np.float32),
        rank=np.array([model.rank], dtype=np.int64),
        ridge=np.array([model.ridge], dtype=np.float32),
        permutation_seed=np.array([-1 if model.permutation_seed is None else model.permutation_seed], dtype=np.int64),
        manifest_json=np.array([json.dumps(manifest, ensure_ascii=False)]),
    )


def fmt(value: float) -> str:
    return f"{value:.3f}"


def md_table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(item) for item in row) + " |")
    return "\n".join(lines)


def write_report(
    path: Path,
    *,
    manifest: dict[str, Any],
    rows: list[dict[str, Any]],
    best: dict[str, Any],
    best_model_path: Path,
) -> None:
    dev_rows = [row for row in rows if row["split"] == "dev" and row["model"] == "candidate"]
    null_dev_rows = [row for row in rows if row["split"] == "dev" and row["model"].startswith("permutation")]
    top_dev = sorted(
        dev_rows,
        key=lambda row: (
            float(row["mse_improvement_fraction"]),
            float(row["edited_minus_source_probe_probability"]),
            float(row["mean_delta_cosine"]),
        ),
        reverse=True,
    )[:12]
    null_top = sorted(
        null_dev_rows,
        key=lambda row: float(row["mse_improvement_fraction"]),
        reverse=True,
    )[:8]
    split_rows = [
        row
        for row in rows
        if row["model"] == "candidate" and row["rank"] == best["rank"] and row["ridge"] == best["ridge"]
    ]

    lines = [
        "# SCOTUS Replay Low-Rank Intervention Diagnostic",
        "",
        "## Purpose",
        "",
        "Train a frozen-feature low-rank map from source replay states to paired target deltas. This is a diagnostic for whether a learned activation-space intervention is worth a later no-mask generation hook run; it is not steering evidence by itself.",
        "",
        "## Inputs",
        "",
        md_table(
            ["Field", "Value"],
            [
                ["Replay run", manifest["replay_run"]],
                ["Feature key", manifest["feature_key"]],
                ["Probe path", manifest["probe_path"]],
                ["Source label", manifest["source_label"]],
                ["Target label", manifest["target_label"]],
                ["Ranks", ", ".join(str(item) for item in manifest["ranks"])],
                ["Ridges", ", ".join(str(item) for item in manifest["ridges"])],
                ["Best model", str(best_model_path)],
            ],
        ),
        "",
        "## Best Candidate By Dev MSE Improvement",
        "",
        md_table(
            ["Rank", "Ridge", "Dev MSE improvement", "Dev delta cosine", "Dev probe prob shift"],
            [
                [
                    best["rank"],
                    best["ridge"],
                    fmt(float(best["mse_improvement_fraction"])),
                    fmt(float(best["mean_delta_cosine"])),
                    fmt(float(best["edited_minus_source_probe_probability"])),
                ]
            ],
        ),
        "",
        "## Best Candidate Across Splits",
        "",
        md_table(
            ["Split", "MSE improvement", "Delta cosine", "Source prob", "Edited prob", "Target prob", "Edited positive rate"],
            [
                [
                    row["split"],
                    fmt(float(row["mse_improvement_fraction"])),
                    fmt(float(row["mean_delta_cosine"])),
                    fmt(float(row["mean_source_probe_probability"])),
                    fmt(float(row["mean_edited_probe_probability"])),
                    fmt(float(row["mean_target_probe_probability"])),
                    fmt(float(row["edited_probe_positive_rate"])),
                ]
                for row in sorted(split_rows, key=lambda item: ["train", "dev", "test"].index(str(item["split"])))
            ],
        ),
        "",
        "## Top Candidate Dev Rows",
        "",
        md_table(
            ["Rank", "Ridge", "MSE improvement", "Delta cosine", "Probe prob shift", "Edited positive rate"],
            [
                [
                    row["rank"],
                    row["ridge"],
                    fmt(float(row["mse_improvement_fraction"])),
                    fmt(float(row["mean_delta_cosine"])),
                    fmt(float(row["edited_minus_source_probe_probability"])),
                    fmt(float(row["edited_probe_positive_rate"])),
                ]
                for row in top_dev
            ],
        ),
        "",
        "## Top Permutation Null Dev Rows",
        "",
        md_table(
            ["Null", "Rank", "Ridge", "MSE improvement", "Delta cosine", "Probe prob shift"],
            [
                [
                    row["model"],
                    row["rank"],
                    row["ridge"],
                    fmt(float(row["mse_improvement_fraction"])),
                    fmt(float(row["mean_delta_cosine"])),
                    fmt(float(row["edited_minus_source_probe_probability"])),
                ]
                for row in null_top
            ],
        ),
        "",
        "## Read",
        "",
        "- A good offline result only means the replay feature geometry supports a learned map; it does not prove causal control during free generation.",
        "- A later generation test would still need ordinary prompts, no justice/persona instructions, same-family permutation controls, same-layer random controls, proposition scoring, and strongest-control gates.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay-run", type=Path, default=DEFAULT_REPLAY_RUN)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--region", default="assistant_all")
    parser.add_argument("--layer", type=int, default=8)
    parser.add_argument("--pair-field", default="pair_id")
    parser.add_argument("--label-field", default="frame_label")
    parser.add_argument("--source-label", default="commerce_authority")
    parser.add_argument("--target-label", default="commerce_limits")
    parser.add_argument("--ranks", default="0,1,2,4,8,16,32")
    parser.add_argument("--ridges", default="0.01,0.1,1.0,10.0")
    parser.add_argument("--permutation-controls", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260501)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = args.output_root / f"scotus_replay_lowrank_diag_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)

    key = feature_key(args.region, args.layer)
    meta = read_jsonl(args.replay_run / "feature_meta.jsonl")
    with np.load(args.replay_run / "features.npz") as data:
        if key not in data.files:
            raise KeyError(f"Missing {key} in {args.replay_run / 'features.npz'}")
        features = data[key].astype(np.float32)

    batches = {
        split: load_pair_batch(
            features,
            meta,
            split,
            pair_field=args.pair_field,
            label_field=args.label_field,
            source_label=args.source_label,
            target_label=args.target_label,
        )
        for split in ("train", "dev", "test")
    }
    probe_path = args.replay_run / "best_probe_direction.npz"
    probe = load_probe(probe_path)
    ranks = parse_ints(args.ranks)
    ridges = parse_floats(args.ridges)

    rows: list[dict[str, Any]] = []
    models: dict[tuple[int, float], LowRankModel] = {}
    for rank in ranks:
        for ridge in ridges:
            model = fit_lowrank(batches["train"], rank=rank, ridge=ridge, permutation_seed=None)
            models[(rank, ridge)] = model
            for split, batch in batches.items():
                rows.append(evaluate_model(model, batch, probe, model_name="candidate"))
            for control_idx in range(args.permutation_controls):
                null_seed = args.seed + (rank * 1009) + (control_idx * 100_003)
                null_model = fit_lowrank(batches["train"], rank=rank, ridge=ridge, permutation_seed=null_seed)
                for split, batch in batches.items():
                    rows.append(evaluate_model(null_model, batch, probe, model_name=f"permutation_{control_idx}"))

    candidate_dev = [row for row in rows if row["model"] == "candidate" and row["split"] == "dev"]
    best = max(
        candidate_dev,
        key=lambda row: (
            float(row["mse_improvement_fraction"]),
            float(row["edited_minus_source_probe_probability"]),
            float(row["mean_delta_cosine"]),
        ),
    )
    best_model = models[(int(best["rank"]), float(best["ridge"]))]
    manifest = {
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "replay_run": str(args.replay_run),
        "output_dir": str(out_dir),
        "feature_key": key,
        "probe_path": str(probe_path),
        "pair_field": args.pair_field,
        "label_field": args.label_field,
        "source_label": args.source_label,
        "target_label": args.target_label,
        "ranks": ranks,
        "ridges": ridges,
        "permutation_controls": int(args.permutation_controls),
        "seed": int(args.seed),
        "pair_counts": {split: len(batch.pair_ids) for split, batch in batches.items()},
        "best_rank": int(best["rank"]),
        "best_ridge": float(best["ridge"]),
    }
    best_model_path = (
        out_dir
        / f"lowrank_{safe_name(args.source_label)}_to_{safe_name(args.target_label)}_{key}_rank{best_model.rank}_ridge{best_model.ridge:g}.npz"
    )
    save_model(best_model_path, best_model, manifest)
    manifest["best_model_path"] = str(best_model_path)

    write_json(out_dir / "manifest.json", manifest)
    write_jsonl(out_dir / "metrics.jsonl", rows)
    write_report(out_dir / "report.md", manifest=manifest, rows=rows, best=best, best_model_path=best_model_path)
    print(f"Wrote {out_dir / 'report.md'}")


if __name__ == "__main__":
    main()
