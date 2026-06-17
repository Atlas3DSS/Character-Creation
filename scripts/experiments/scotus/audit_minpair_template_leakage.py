#!/usr/bin/env python3
"""Audit exact-template leakage in SCOTUS minimal-pair replay probes."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PROBE_DIR = PROJECT_ROOT / "sweep_v4" / "scotus_minpair_replay_20260501_100514"
DEFAULT_SAE_RUN_DIR = PROJECT_ROOT / "sweep_v4" / "scotus_sae_probe_20260501_112601"
DEFAULT_REPORT = PROJECT_ROOT / "reports" / "scotus_minpair_template_leakage_audit_20260501.md"
DEFAULT_JSON = PROJECT_ROOT / "reports" / "scotus_minpair_template_leakage_audit_20260501.json"


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


def markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return "\n".join(lines)


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def text_hash(text: str) -> str:
    return hashlib.sha1(normalize_text(text).encode("utf-8")).hexdigest()[:12]


def load_examples(probe_dir: Path) -> list[dict[str, Any]]:
    for name in ("probe_examples.jsonl", "examples.jsonl"):
        path = probe_dir / name
        if path.exists():
            return read_jsonl(path)
    raise FileNotFoundError(f"No probe_examples.jsonl or examples.jsonl under {probe_dir}")


def add_template_fields(rows: list[dict[str, Any]]) -> None:
    hashes_by_pair: dict[str, dict[int, str]] = defaultdict(dict)
    for row in rows:
        row["assistant_template_hash"] = text_hash(str(row.get("assistant_text") or row.get("text") or ""))
        hashes_by_pair[str(row["pair_id"])][int(row["label"])] = row["assistant_template_hash"]
    pair_group_by_pair = {
        pair_id: "__".join(label_hashes[label] for label in sorted(label_hashes))
        for pair_id, label_hashes in hashes_by_pair.items()
    }
    for row in rows:
        row["template_pair_group"] = pair_group_by_pair[str(row["pair_id"])]


def metric_dict(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    return {
        "n": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred)),
        "label_counts": dict(sorted(Counter(y_true.tolist()).items())),
        "prediction_counts": dict(sorted(Counter(y_pred.tolist()).items())),
    }


def make_dense_probe(c_value: float) -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    C=c_value,
                    class_weight="balanced",
                    max_iter=4000,
                    solver="liblinear",
                ),
            ),
        ]
    )


def make_sparse_probe(c_value: float) -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler(with_mean=False)),
            (
                "clf",
                LogisticRegression(
                    C=c_value,
                    class_weight="balanced",
                    max_iter=4000,
                    solver="liblinear",
                ),
            ),
        ]
    )


def template_hash_baseline(rows: list[dict[str, Any]]) -> dict[str, Any]:
    train_rows = [row for row in rows if row["split"] in {"train", "dev"}]
    test_rows = [row for row in rows if row["split"] == "test"]
    majority_by_hash: dict[str, int] = {}
    for hash_value in {row["assistant_template_hash"] for row in train_rows}:
        labels = [int(row["label"]) for row in train_rows if row["assistant_template_hash"] == hash_value]
        majority_by_hash[hash_value] = Counter(labels).most_common(1)[0][0]
    global_majority = Counter(int(row["label"]) for row in train_rows).most_common(1)[0][0]
    y_true = np.array([int(row["label"]) for row in test_rows], dtype=np.int64)
    y_pred = np.array(
        [majority_by_hash.get(row["assistant_template_hash"], global_majority) for row in test_rows],
        dtype=np.int64,
    )
    return metric_dict(y_true, y_pred)


def text_holdout_cv(rows: list[dict[str, Any]], field: str, c_value: float) -> list[dict[str, Any]]:
    groups = sorted({row["template_pair_group"] for row in rows})
    results: list[dict[str, Any]] = []
    for held_out in groups:
        train_rows = [row for row in rows if row["template_pair_group"] != held_out]
        test_rows = [row for row in rows if row["template_pair_group"] == held_out]
        vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1, lowercase=True)
        clf = LogisticRegression(C=c_value, class_weight="balanced", max_iter=4000, solver="liblinear")
        x_train = vectorizer.fit_transform([str(row.get(field, "")) for row in train_rows])
        x_test = vectorizer.transform([str(row.get(field, "")) for row in test_rows])
        y_train = np.array([int(row["label"]) for row in train_rows], dtype=np.int64)
        y_test = np.array([int(row["label"]) for row in test_rows], dtype=np.int64)
        clf.fit(x_train, y_train)
        pred = clf.predict(x_test)
        results.append({"held_out_group": held_out, "metrics": metric_dict(y_test, pred)})
    return results


def dense_holdout_cv(
    *,
    rows: list[dict[str, Any]],
    matrix: np.ndarray,
    c_value: float,
) -> list[dict[str, Any]]:
    groups = sorted({row["template_pair_group"] for row in rows})
    labels = np.array([int(row["label"]) for row in rows], dtype=np.int64)
    results: list[dict[str, Any]] = []
    for held_out in groups:
        train_idx = np.array([idx for idx, row in enumerate(rows) if row["template_pair_group"] != held_out], dtype=np.int64)
        test_idx = np.array([idx for idx, row in enumerate(rows) if row["template_pair_group"] == held_out], dtype=np.int64)
        clf = make_dense_probe(c_value)
        clf.fit(matrix[train_idx], labels[train_idx])
        pred = clf.predict(matrix[test_idx])
        results.append({"held_out_group": held_out, "metrics": metric_dict(labels[test_idx], pred)})
    return results


def sparse_holdout_cv(
    *,
    rows: list[dict[str, Any]],
    matrix: sp.csr_matrix,
    c_value: float,
    min_train_df: int,
) -> list[dict[str, Any]]:
    groups = sorted({row["template_pair_group"] for row in rows})
    labels = np.array([int(row["label"]) for row in rows], dtype=np.int64)
    results: list[dict[str, Any]] = []
    for held_out in groups:
        train_idx = np.array([idx for idx, row in enumerate(rows) if row["template_pair_group"] != held_out], dtype=np.int64)
        test_idx = np.array([idx for idx, row in enumerate(rows) if row["template_pair_group"] == held_out], dtype=np.int64)
        train_df = np.asarray((matrix[train_idx] > 0).sum(axis=0)).ravel()
        keep = np.flatnonzero(train_df >= min_train_df)
        if keep.size == 0:
            pred = np.full(test_idx.shape, Counter(labels[train_idx].tolist()).most_common(1)[0][0], dtype=np.int64)
            metrics = metric_dict(labels[test_idx], pred)
            metrics["feature_count"] = 0
        else:
            clf = make_sparse_probe(c_value)
            clf.fit(matrix[train_idx][:, keep], labels[train_idx])
            pred = clf.predict(matrix[test_idx][:, keep])
            metrics = metric_dict(labels[test_idx], pred)
            metrics["feature_count"] = int(keep.size)
        results.append({"held_out_group": held_out, "metrics": metrics})
    return results


def aggregate_cv(rows: list[dict[str, Any]]) -> dict[str, float]:
    return {
        "mean_balanced_accuracy": float(np.mean([row["metrics"]["balanced_accuracy"] for row in rows])),
        "min_balanced_accuracy": float(np.min([row["metrics"]["balanced_accuracy"] for row in rows])),
        "max_balanced_accuracy": float(np.max([row["metrics"]["balanced_accuracy"] for row in rows])),
        "mean_accuracy": float(np.mean([row["metrics"]["accuracy"] for row in rows])),
    }


def sae_cache_path(sae_run_dir: Path) -> Path:
    manifest = json.loads((sae_run_dir / "manifest.json").read_text(encoding="utf-8"))
    best = manifest["best"]
    cache_dir = Path(manifest["feature_cache_dir"])
    if not cache_dir.is_absolute():
        cache_dir = PROJECT_ROOT / cache_dir
    return cache_dir / f"{best['sae_name']}__{best['region']}__L{int(best['layer']):02d}.npz"


def load_feature_matrix(path: Path, key: str) -> np.ndarray:
    with np.load(path) as data:
        if key not in data.files:
            raise KeyError(f"Missing {key} in {path}; available={data.files}")
        return data[key].astype(np.float32, copy=False)


def write_report(path: Path, summary: dict[str, Any]) -> None:
    template_rows = []
    for row in summary["template_groups"]:
        template_rows.append(
            [
                row["template_pair_group"],
                row["n"],
                row["label_counts"],
                row["split_counts"],
                row["authority_snippet"],
                row["limits_snippet"],
            ]
        )
    result_rows = []
    for name, result in summary["diagnostics"].items():
        agg = result["aggregate"]
        result_rows.append(
            [
                name,
                f"{agg['mean_balanced_accuracy']:.3f}",
                f"{agg['min_balanced_accuracy']:.3f}",
                f"{agg['max_balanced_accuracy']:.3f}",
                f"{agg['mean_accuracy']:.3f}",
            ]
        )
    lines = [
        "# SCOTUS Minimal-Pair Template Leakage Audit",
        "",
        f"Created: `{summary['created_at']}`",
        "",
        "## Verdict",
        "",
        summary["verdict"],
        "",
        "## Exact Template Reuse",
        "",
        markdown_table(
            ["Metric", "Value"],
            [
                ["Rows", summary["row_count"]],
                ["Unique assistant completions", summary["unique_assistant_templates"]],
                ["Template-pair groups", summary["template_pair_group_count"]],
                ["Original split exact-template baseline BA", f"{summary['original_split_template_baseline']['balanced_accuracy']:.3f}"],
            ],
        ),
        "",
        markdown_table(
            ["Template pair group", "N", "Label counts", "Split counts", "Authority snippet", "Limits snippet"],
            template_rows,
        ),
        "",
        "## Leave-One-Template-Pair-Out Diagnostics",
        "",
        markdown_table(
            ["Diagnostic", "Mean BA", "Min BA", "Max BA", "Mean accuracy"],
            result_rows,
        ),
        "",
        "## Reading Notes",
        "",
        "- This audit ignores the original train/dev/test split and instead holds out one exact authority/limits answer-template pair at a time.",
        "- The original split reused every exact assistant template across train/dev/test, so high original split accuracy can be template recognition.",
        "- A candidate circuit should survive template-pair holdout before decoder-column steering is treated as meaningful.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe-dir", type=Path, default=DEFAULT_PROBE_DIR)
    parser.add_argument("--sae-run-dir", type=Path, default=DEFAULT_SAE_RUN_DIR)
    parser.add_argument("--residual-keys", default="assistant_all__L04,assistant_all__L08,assistant_all__L12,assistant_all__L16,assistant_all__L20")
    parser.add_argument("--c", type=float, default=0.001)
    parser.add_argument("--min-train-df", type=int, default=2)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--json-output", type=Path, default=DEFAULT_JSON)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    probe_dir = args.probe_dir if args.probe_dir.is_absolute() else PROJECT_ROOT / args.probe_dir
    sae_run_dir = args.sae_run_dir if args.sae_run_dir.is_absolute() else PROJECT_ROOT / args.sae_run_dir

    rows = load_examples(probe_dir)
    meta_rows = read_jsonl(probe_dir / "feature_meta.jsonl")
    if len(rows) != len(meta_rows):
        raise RuntimeError(f"Row count mismatch: {len(rows)} examples vs {len(meta_rows)} feature rows")
    add_template_fields(rows)
    for row, meta in zip(rows, meta_rows, strict=True):
        if row["example_id"] != meta["example_id"]:
            raise RuntimeError(f"Example/meta order mismatch at {row['example_id']} vs {meta['example_id']}")

    template_groups: list[dict[str, Any]] = []
    for group in sorted({row["template_pair_group"] for row in rows}):
        group_rows = [row for row in rows if row["template_pair_group"] == group]
        authority = next(row for row in group_rows if int(row["label"]) == 0)
        limits = next(row for row in group_rows if int(row["label"]) == 1)
        template_groups.append(
            {
                "template_pair_group": group,
                "n": len(group_rows),
                "label_counts": dict(sorted(Counter(int(row["label"]) for row in group_rows).items())),
                "split_counts": dict(sorted(Counter(str(row["split"]) for row in group_rows).items())),
                "authority_snippet": normalize_text(authority["assistant_text"])[:140],
                "limits_snippet": normalize_text(limits["assistant_text"])[:140],
            }
        )

    diagnostics: dict[str, Any] = {}
    prompt_cv = text_holdout_cv(rows, "prompt", args.c)
    diagnostics["prompt_tfidf"] = {"folds": prompt_cv, "aggregate": aggregate_cv(prompt_cv)}
    assistant_cv = text_holdout_cv(rows, "assistant_text", args.c)
    diagnostics["assistant_text_tfidf"] = {"folds": assistant_cv, "aggregate": aggregate_cv(assistant_cv)}

    features_npz = probe_dir / "features.npz"
    for key in [part.strip() for part in args.residual_keys.split(",") if part.strip()]:
        matrix = load_feature_matrix(features_npz, key)
        cv_rows = dense_holdout_cv(rows=rows, matrix=matrix, c_value=args.c)
        diagnostics[f"residual_{key}"] = {"folds": cv_rows, "aggregate": aggregate_cv(cv_rows)}

    sae_matrix = sp.load_npz(sae_cache_path(sae_run_dir)).tocsr()
    sae_cv = sparse_holdout_cv(rows=rows, matrix=sae_matrix, c_value=args.c, min_train_df=args.min_train_df)
    diagnostics["sae_best_l0_100_assistant_all_L8"] = {"folds": sae_cv, "aggregate": aggregate_cv(sae_cv)}

    sae_mean_ba = diagnostics["sae_best_l0_100_assistant_all_L8"]["aggregate"]["mean_balanced_accuracy"]
    residual_best = max(
        result["aggregate"]["mean_balanced_accuracy"]
        for name, result in diagnostics.items()
        if name.startswith("residual_")
    )
    if sae_mean_ba <= 0.67 and residual_best <= 0.67:
        verdict = (
            "Template-pair holdout collapses the activation evidence toward chance. The minimal-pair replay probe "
            "should be treated as answer-template localization, not a robust judicial reasoning circuit."
        )
    else:
        verdict = (
            "Some activation evidence survives template-pair holdout. Candidate features still need decoder-column "
            "steering and random/same-layer controls before promotion."
        )

    summary = {
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "probe_dir": str(probe_dir),
        "sae_run_dir": str(sae_run_dir),
        "row_count": len(rows),
        "unique_assistant_templates": len({row["assistant_template_hash"] for row in rows}),
        "template_pair_group_count": len(template_groups),
        "template_groups": template_groups,
        "original_split_template_baseline": template_hash_baseline(rows),
        "diagnostics": diagnostics,
        "verdict": verdict,
    }
    report = args.report if args.report.is_absolute() else PROJECT_ROOT / args.report
    json_output = args.json_output if args.json_output.is_absolute() else PROJECT_ROOT / args.json_output
    write_json(json_output, summary)
    write_report(report, summary)
    print(f"Wrote {report}")
    print(f"Wrote {json_output}")


if __name__ == "__main__":
    main()
