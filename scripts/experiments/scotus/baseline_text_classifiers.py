#!/usr/bin/env python3
"""Leakage baselines for SCOTUS justice-style matched contrasts."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report
from sklearn.pipeline import FeatureUnion, make_pipeline
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PAIRS = PROJECT_ROOT / "data" / "scotus" / "scotus_matched_pairs_v1.jsonl"
DEFAULT_JSON = PROJECT_ROOT / "data" / "scotus" / "manifests" / "scotus_baseline_results_v1.json"
DEFAULT_REPORT = PROJECT_ROOT / "reports" / "scotus_baseline_text_classifiers.md"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def pair_to_samples(pair: dict[str, Any]) -> list[dict[str, Any]]:
    shared_meta = {
        "issue_area_label": pair.get("issue_area_label") or "unknown",
        "opinion_type": pair.get("opinion_type_a") or "unknown",
        "decade": pair.get("decade") or "unknown",
        "decision_direction": str(pair.get("decision_direction") or "unknown"),
        "matching_key": "|".join(str(x) for x in pair.get("matching_key", [])),
    }
    return [
        {
            "text": pair["text_a"],
            "label": pair["justice_a"],
            "pair": pair["pair"],
            "split": pair["split"],
            "text_variant": pair["text_variant"],
            "metadata": shared_meta,
            "length_features": [
                float(pair.get("token_count_a") or 0),
                float(pair.get("citation_count_a") or 0),
            ],
        },
        {
            "text": pair["text_b"],
            "label": pair["justice_b"],
            "pair": pair["pair"],
            "split": pair["split"],
            "text_variant": pair["text_variant"],
            "metadata": shared_meta,
            "length_features": [
                float(pair.get("token_count_b") or 0),
                float(pair.get("citation_count_b") or 0),
            ],
        },
    ]


def flatten_samples(pairs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    for pair in pairs:
        samples.extend(pair_to_samples(pair))
    return samples


def metric_payload(y_true: list[str], y_pred: list[str]) -> dict[str, Any]:
    return {
        "n": len(y_true),
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "labels": sorted(set(y_true) | set(y_pred)),
        "classification_report": classification_report(y_true, y_pred, output_dict=True, zero_division=0),
    }


def train_text_model(
    train: list[dict[str, Any]],
    eval_rows: list[dict[str, Any]],
    *,
    analyzer: str,
) -> dict[str, Any]:
    if analyzer == "word":
        vectorizer = TfidfVectorizer(
            lowercase=True,
            strip_accents="unicode",
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.9,
            max_features=75_000,
            sublinear_tf=True,
        )
    elif analyzer == "char":
        vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            min_df=3,
            max_features=100_000,
            sublinear_tf=True,
        )
    elif analyzer == "word_char":
        vectorizer = FeatureUnion(
            [
                (
                    "word",
                    TfidfVectorizer(
                        lowercase=True,
                        strip_accents="unicode",
                        ngram_range=(1, 2),
                        min_df=2,
                        max_df=0.9,
                        max_features=75_000,
                        sublinear_tf=True,
                    ),
                ),
                (
                    "char",
                    TfidfVectorizer(
                        analyzer="char_wb",
                        ngram_range=(3, 5),
                        min_df=3,
                        max_features=100_000,
                        sublinear_tf=True,
                    ),
                ),
            ]
        )
    else:
        raise ValueError(f"Unsupported analyzer: {analyzer}")

    model = LogisticRegression(max_iter=2000, class_weight="balanced", solver="liblinear")
    x_train = [row["text"] for row in train]
    y_train = [row["label"] for row in train]
    train_matrix = vectorizer.fit_transform(x_train)
    model.fit(train_matrix, y_train)
    pred = model.predict(vectorizer.transform([row["text"] for row in eval_rows]))
    payload = metric_payload([row["label"] for row in eval_rows], list(pred))
    payload["feature_count"] = int(train_matrix.shape[1])
    return payload


def train_word_tfidf_model(train: list[dict[str, Any]], eval_rows: list[dict[str, Any]]) -> dict[str, Any]:
    vectorizer = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.9,
        max_features=75_000,
        sublinear_tf=True,
    )
    model = LogisticRegression(max_iter=2000, class_weight="balanced", solver="liblinear")
    x_train = [row["text"] for row in train]
    y_train = [row["label"] for row in train]
    model.fit(vectorizer.fit_transform(x_train), y_train)
    pred = model.predict(vectorizer.transform([row["text"] for row in eval_rows]))
    payload = metric_payload([row["label"] for row in eval_rows], list(pred))
    payload["vocab_size"] = len(vectorizer.vocabulary_)
    return payload


def train_metadata_model(train: list[dict[str, Any]], eval_rows: list[dict[str, Any]]) -> dict[str, Any]:
    vectorizer = DictVectorizer(sparse=True)
    model = LogisticRegression(max_iter=2000, class_weight="balanced", solver="liblinear")
    x_train = [row["metadata"] for row in train]
    y_train = [row["label"] for row in train]
    model.fit(vectorizer.fit_transform(x_train), y_train)
    pred = model.predict(vectorizer.transform([row["metadata"] for row in eval_rows]))
    payload = metric_payload([row["label"] for row in eval_rows], list(pred))
    payload["feature_count"] = len(vectorizer.feature_names_)
    return payload


def train_length_model(train: list[dict[str, Any]], eval_rows: list[dict[str, Any]]) -> dict[str, Any]:
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, class_weight="balanced", solver="liblinear"),
    )
    x_train = np.array([row["length_features"] for row in train], dtype=float)
    y_train = [row["label"] for row in train]
    model.fit(x_train, y_train)
    pred = model.predict(np.array([row["length_features"] for row in eval_rows], dtype=float))
    return metric_payload([row["label"] for row in eval_rows], list(pred))


def majority_baseline(train: list[dict[str, Any]], eval_rows: list[dict[str, Any]]) -> dict[str, Any]:
    label = Counter(row["label"] for row in train).most_common(1)[0][0]
    pred = [label for _ in eval_rows]
    payload = metric_payload([row["label"] for row in eval_rows], pred)
    payload["majority_label"] = label
    return payload


def evaluate(samples: list[dict[str, Any]]) -> dict[str, Any]:
    results: dict[str, Any] = {}
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for sample in samples:
        groups[(sample["pair"], sample["text_variant"])].append(sample)

    for (pair_name, text_variant), rows in groups.items():
        train = [row for row in rows if row["split"] == "train"]
        split_rows = {
            "dev": [row for row in rows if row["split"] == "dev"],
            "test": [row for row in rows if row["split"] == "test"],
        }
        key = f"{pair_name}/{text_variant}"
        results[key] = {
            "train_n": len(train),
            "label_counts": dict(Counter(row["label"] for row in rows)),
            "splits": {split: len(split_data) for split, split_data in split_rows.items()},
            "models": {},
        }
        if len(set(row["label"] for row in train)) < 2:
            results[key]["error"] = "train split lacks both labels"
            continue
        for split, eval_rows in split_rows.items():
            if not eval_rows:
                continue
            results[key]["models"].setdefault(split, {})
            results[key]["models"][split]["majority"] = majority_baseline(train, eval_rows)
            results[key]["models"][split]["word_tfidf_logreg"] = train_text_model(
                train,
                eval_rows,
                analyzer="word",
            )
            results[key]["models"][split]["char_tfidf_logreg"] = train_text_model(
                train,
                eval_rows,
                analyzer="char",
            )
            results[key]["models"][split]["word_char_tfidf_logreg"] = train_text_model(
                train,
                eval_rows,
                analyzer="word_char",
            )
            results[key]["models"][split]["metadata_logreg"] = train_metadata_model(train, eval_rows)
            results[key]["models"][split]["length_citation_logreg"] = train_length_model(train, eval_rows)
    return results


def markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return "\n".join(lines)


def write_report(path: Path, results: dict[str, Any]) -> None:
    lines = [
        "# SCOTUS Baseline Text Classifiers",
        "",
        "Case-held-out matched-pair baselines. Each pair contributes one sample per justice.",
        "",
        "## Decision",
        "",
    ]
    decision_rows = []
    for pair_name in sorted({key.split("/", 1)[0] for key in results}):
        masked_key = f"{pair_name}/masked"
        test_models = results.get(masked_key, {}).get("models", {}).get("test", {})
        text_scores = {
            model_name: metrics.get("balanced_accuracy", 0.0)
            for model_name, metrics in test_models.items()
            if "tfidf" in model_name
        }
        best_model, best_score = max(text_scores.items(), key=lambda item: item[1]) if text_scores else ("none", 0.0)
        if best_score >= 0.75:
            decision = "activation-ready"
        elif best_score >= 0.60:
            decision = "weak/exploratory only"
        else:
            decision = "no-go for activation"
        decision_rows.append([pair_name, decision, best_model, f"{best_score:.3f}"])
    lines.append(markdown_table(["Pair", "Decision", "Best masked test model", "Best masked test balanced accuracy"], decision_rows))
    lines.extend(
        [
            "",
            "Conservative threshold: do not treat a pair as activation-ready unless masked, case-held-out text separation is at least 0.75 balanced accuracy.",
            "",
            "## Test Metrics",
            "",
        ]
    )
    rows = []
    for key, payload in sorted(results.items()):
        models = payload.get("models", {}).get("test", {})
        for model_name, metrics in models.items():
            rows.append(
                [
                    key,
                    model_name,
                    metrics.get("n", 0),
                    f"{metrics.get('accuracy', 0.0):.3f}",
                    f"{metrics.get('balanced_accuracy', 0.0):.3f}",
                ]
            )
    lines.append(markdown_table(["Pair/Variant", "Model", "N", "Accuracy", "Balanced Accuracy"], rows))

    lines.extend(["", "## Dev Metrics", ""])
    dev_rows = []
    for key, payload in sorted(results.items()):
        models = payload.get("models", {}).get("dev", {})
        for model_name, metrics in models.items():
            dev_rows.append(
                [
                    key,
                    model_name,
                    metrics.get("n", 0),
                    f"{metrics.get('accuracy', 0.0):.3f}",
                    f"{metrics.get('balanced_accuracy', 0.0):.3f}",
                ]
            )
    lines.append(markdown_table(["Pair/Variant", "Model", "N", "Accuracy", "Balanced Accuracy"], dev_rows))

    lines.extend(["", "## Interpretation Notes", ""])
    lines.extend(
        [
            "- `metadata_logreg` uses only matched metadata fields. High scores here indicate residual confounding.",
            "- `length_citation_logreg` tests whether chunk length or citation density alone separates justices.",
            "- `word_tfidf_logreg` on `masked` text is the main Phase 3 leakage check before activation work.",
            "- Character n-grams are included as a stronger stylometric leakage check.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SCOTUS TF-IDF and leakage baselines.")
    parser.add_argument("--pairs", type=Path, default=DEFAULT_PAIRS)
    parser.add_argument("--json-output", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pairs = read_jsonl(args.pairs)
    samples = flatten_samples(pairs)
    results = evaluate(samples)
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    write_report(args.report, results)
    print(f"Wrote {args.json_output}")
    print(f"Wrote {args.report}")


if __name__ == "__main__":
    main()
