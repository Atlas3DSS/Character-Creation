#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_CORPUS_DIR = "/home/orwel/dev_genius/experiments/Character Creation/sweep_v4/meta_cognition_scorer_corpus_v1_20260417_121022"
DEFAULT_MODEL_PATH = "/home/orwel/dev_genius/models/Qwen3.6-35B-A3B"
DEFAULT_OUTPUT_ROOT = "/home/orwel/dev_genius/experiments/Character Creation/sweep_v4"
DEFAULT_TAG = "meta_cognition_activation_probe_v1"


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


def turns_text(turns: list[dict[str, str]]) -> str:
    parts = []
    for turn in turns:
        speaker = str(turn.get("speaker", "user")).strip()
        content = str(turn.get("content", "")).strip()
        parts.append(f"{speaker}: {content}")
    return "\n".join(parts)


def build_messages(row: dict[str, Any]) -> list[dict[str, str]]:
    user = (
        f"Scenario setup:\n{row['setup']}\n\n"
        f"Conversation so far:\n{turns_text(row['turns'])}\n\n"
        "Below is the assistant reply to analyze."
    )
    return [
        {"role": "system", "content": "You are a careful assistant."},
        {"role": "user", "content": user},
        {"role": "assistant", "content": row["response_text"]},
    ]


def token_span_from_chars(offsets: list[tuple[int, int]], start_char: int, end_char: int) -> tuple[int, int] | None:
    token_ids = [i for i, (s, e) in enumerate(offsets) if e > start_char and s < end_char]
    if not token_ids:
        return None
    return token_ids[0], token_ids[-1] + 1


def build_feature_id(row: dict[str, Any]) -> str:
    return f"{row['split']}|{row['behavior']}|{row['candidate_id']}|{int(row['label'])}"


def make_classifier(C: float) -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=4000, solver="liblinear", C=C, class_weight="balanced")),
        ]
    )


def evaluate_binary(rows: list[dict[str, Any]], preds: np.ndarray, probs: np.ndarray) -> dict[str, Any]:
    y = np.array([int(row["label"]) for row in rows], dtype=np.int64)
    grouped: dict[str, list[int]] = defaultdict(list)
    for idx, row in enumerate(rows):
        grouped[row["behavior"]].append(idx)
    by_behavior: dict[str, Any] = {}
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
    return {
        "n": int(len(rows)),
        "accuracy": float(accuracy_score(y, preds)),
        "balanced_accuracy": float(balanced_accuracy_score(y, preds)),
        "f1": float(f1_score(y, preds)),
        "label_counts": dict(sorted(Counter(y.tolist()).items())),
        "prediction_counts": dict(sorted(Counter(preds.tolist()).items())),
        "mean_positive_probability": float(np.mean(probs)),
        "by_behavior": by_behavior,
    }


def render_and_span(tokenizer: AutoTokenizer, row: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor, tuple[int, int], int]:
    messages = build_messages(row)
    rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    target = row["response_text"]
    start = rendered.rfind(target)
    if start < 0:
        raise ValueError(f"response text not found in rendered transcript for {build_feature_id(row)}")
    end = start + len(target)
    encoded = tokenizer(
        rendered,
        return_tensors="pt",
        return_offsets_mapping=True,
        add_special_tokens=False,
    )
    offsets = [(int(s), int(e)) for s, e in encoded["offset_mapping"][0].tolist()]
    span = token_span_from_chars(offsets, start, end)
    if span is None:
        raise ValueError(f"assistant span not found for {build_feature_id(row)}")
    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]
    return input_ids, attention_mask, span, span[0] - 1


def region_vectors(hidden_states: tuple[torch.Tensor, ...], assistant_span: tuple[int, int], prompt_last_idx: int) -> dict[str, list[np.ndarray]]:
    start, end = assistant_span
    regions: dict[str, list[np.ndarray]] = {
        "assistant_mean": [],
        "assistant_first16": [],
        "assistant_last16": [],
        "assistant_last": [],
        "prompt_last": [],
    }
    first_end = min(end, start + 16)
    last_start = max(start, end - 16)
    for layer_h in hidden_states[1:]:
        h = layer_h[0]
        assistant = h[start:end]
        first_chunk = h[start:first_end]
        last_chunk = h[last_start:end]
        regions["assistant_mean"].append(assistant.mean(dim=0).float().cpu().numpy())
        regions["assistant_first16"].append(first_chunk.mean(dim=0).float().cpu().numpy())
        regions["assistant_last16"].append(last_chunk.mean(dim=0).float().cpu().numpy())
        regions["assistant_last"].append(h[end - 1].float().cpu().numpy())
        regions["prompt_last"].append(h[prompt_last_idx].float().cpu().numpy())
    return regions


def extract_features(rows: list[dict[str, Any]], model_path: str, device: str, out_dir: Path) -> dict[str, Any]:
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map=device,
        trust_remote_code=True,
    )
    model.eval()

    all_meta: list[dict[str, Any]] = []
    layer_region_rows: list[dict[str, Any]] = []
    collected: dict[str, dict[int, list[np.ndarray]]] = defaultdict(lambda: defaultdict(list))
    labels: list[int] = []
    splits: list[str] = []
    feature_ids: list[str] = []
    behaviors: list[str] = []

    with torch.inference_mode():
        for idx, row in enumerate(rows, start=1):
            input_ids, attention_mask, assistant_span, prompt_last_idx = render_and_span(tokenizer, row)
            input_ids = input_ids.to(model.device)
            attention_mask = attention_mask.to(model.device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True, use_cache=False)
            regions = region_vectors(outputs.hidden_states, assistant_span, prompt_last_idx)

            fid = build_feature_id(row)
            labels.append(int(row["label"]))
            splits.append(row["split"])
            feature_ids.append(fid)
            behaviors.append(row["behavior"])
            meta_row = {
                "feature_id": fid,
                "split": row["split"],
                "behavior": row["behavior"],
                "candidate_id": row["candidate_id"],
                "label": int(row["label"]),
                "assistant_token_span": [int(assistant_span[0]), int(assistant_span[1])],
                "n_total_tokens": int(input_ids.shape[1]),
                "n_assistant_tokens": int(assistant_span[1] - assistant_span[0]),
            }
            all_meta.append(meta_row)
            for region_name, vecs in regions.items():
                for layer_idx, vec in enumerate(vecs):
                    collected[region_name][layer_idx].append(vec.astype(np.float32, copy=False))
            layer_region_rows.append(
                {
                    "feature_id": fid,
                    "split": row["split"],
                    "behavior": row["behavior"],
                    "label": int(row["label"]),
                    "n_total_tokens": int(input_ids.shape[1]),
                    "n_assistant_tokens": int(assistant_span[1] - assistant_span[0]),
                }
            )
            if idx % 10 == 0:
                print(f"[{now_iso()}] extracted {idx}/{len(rows)}", flush=True)
            del outputs

    region_arrays: dict[str, dict[int, np.ndarray]] = {}
    for region_name, layer_map in collected.items():
        region_arrays[region_name] = {}
        for layer_idx, vecs in layer_map.items():
            region_arrays[region_name][layer_idx] = np.stack(vecs, axis=0)

    write_jsonl(out_dir / "feature_meta.jsonl", all_meta)
    np.savez_compressed(
        out_dir / "features.npz",
        **{
            f"{region_name}__L{layer_idx:02d}": arr
            for region_name, layer_map in region_arrays.items()
            for layer_idx, arr in layer_map.items()
        },
    )
    return {
        "region_arrays": region_arrays,
        "meta_rows": all_meta,
        "labels": np.array(labels, dtype=np.int64),
        "splits": np.array(splits),
        "feature_ids": feature_ids,
        "behaviors": np.array(behaviors),
        "num_layers": len(next(iter(region_arrays.values()))),
    }


def select_best_probe(
    region_arrays: dict[str, dict[int, np.ndarray]],
    meta_rows: list[dict[str, Any]],
    labels: np.ndarray,
    c_grid: list[float],
) -> dict[str, Any]:
    split_idx = defaultdict(list)
    for idx, row in enumerate(meta_rows):
        split_idx[row["split"]].append(idx)
    train_idx = np.array(split_idx["train"], dtype=np.int64)
    val_idx = np.array(split_idx["val"], dtype=np.int64)
    test_idx = np.array(split_idx["test"], dtype=np.int64)

    best: dict[str, Any] | None = None
    best_train_clf: Pipeline | None = None
    searches: list[dict[str, Any]] = []
    for region_name, layer_map in region_arrays.items():
        for layer_idx, X in layer_map.items():
            X_train = X[train_idx]
            X_val = X[val_idx]
            y_train = labels[train_idx]
            y_val = labels[val_idx]
            for C in c_grid:
                clf = make_classifier(C)
                clf.fit(X_train, y_train)
                val_probs = clf.predict_proba(X_val)[:, 1]
                val_preds = (val_probs >= 0.5).astype(np.int64)
                val_rows = [meta_rows[i] for i in val_idx.tolist()]
                val_metrics = evaluate_binary(val_rows, val_preds, val_probs)
                record = {
                    "region": region_name,
                    "layer": int(layer_idx),
                    "C": float(C),
                    "val_metrics": val_metrics,
                }
                searches.append(record)
                if best is None or val_metrics["balanced_accuracy"] > best["val_metrics"]["balanced_accuracy"] or (
                    val_metrics["balanced_accuracy"] == best["val_metrics"]["balanced_accuracy"]
                    and val_metrics["f1"] > best["val_metrics"]["f1"]
                ):
                    best = record
                    best_train_clf = clf
    assert best is not None
    assert best_train_clf is not None

    region = best["region"]
    layer = int(best["layer"])
    C = float(best["C"])
    X = region_arrays[region][layer]
    trainval_idx = np.concatenate([train_idx, val_idx], axis=0)
    clf = make_classifier(C)
    clf.fit(X[trainval_idx], labels[trainval_idx])

    split_preds: dict[str, list[dict[str, Any]]] = {}
    split_metrics: dict[str, Any] = {"selection": {"best_region": region, "best_layer": layer, "best_C": C, "searches": searches}}
    for split_name, idxs, split_clf in (
        ("train", train_idx, best_train_clf),
        ("val", val_idx, best_train_clf),
        ("test", test_idx, clf),
    ):
        X_split = X[idxs]
        probs = split_clf.predict_proba(X_split)[:, 1]
        preds = (probs >= 0.5).astype(np.int64)
        rows = [meta_rows[i] for i in idxs.tolist()]
        split_metrics[split_name] = evaluate_binary(rows, preds, probs)
        split_preds[split_name] = [
            {
                "feature_id": rows[i]["feature_id"],
                "split": split_name,
                "behavior": rows[i]["behavior"],
                "label": int(rows[i]["label"]),
                "pred": int(preds[i]),
                "prob_positive": float(probs[i]),
                "correct": bool(int(preds[i]) == int(rows[i]["label"])),
                "region": region,
                "layer": layer,
            }
            for i in range(len(rows))
        ]
    return {"metrics": split_metrics, "predictions": split_preds}


def compare_with_text_scorer(
    probe_preds: dict[str, list[dict[str, Any]]],
    scorer_dir: Path,
    best_variant: str,
) -> dict[str, Any]:
    out: dict[str, Any] = {"best_variant": best_variant, "by_split": {}}
    for split in ("train", "val", "test"):
        scorer_rows = load_jsonl(scorer_dir / f"{best_variant}_{split}_predictions.jsonl")
        probe_rows = probe_preds[split]
        scorer_map = {
            row.get("example_key") or f"{row['split']}|{row['behavior']}|{row['candidate_id']}|{int(row['label'])}": row
            for row in scorer_rows
        }
        probe_map = {row["feature_id"]: row for row in probe_rows}
        ids = sorted(set(scorer_map) & set(probe_map))
        scorer_prob = np.array([float(scorer_map[i]["prob_positive"]) for i in ids], dtype=np.float32)
        probe_prob = np.array([float(probe_map[i]["prob_positive"]) for i in ids], dtype=np.float32)
        scorer_pred = np.array([int(scorer_map[i]["pred"]) for i in ids], dtype=np.int64)
        probe_pred = np.array([int(probe_map[i]["pred"]) for i in ids], dtype=np.int64)
        labels = np.array([int(scorer_map[i]["label"]) for i in ids], dtype=np.int64)
        if len(ids) >= 2:
            corr = float(np.corrcoef(scorer_prob, probe_prob)[0, 1])
        else:
            corr = math.nan
        out["by_split"][split] = {
            "n_overlap": int(len(ids)),
            "prediction_agreement": float(np.mean(scorer_pred == probe_pred)) if len(ids) else None,
            "joint_correct_rate": float(np.mean((scorer_pred == labels) & (probe_pred == labels))) if len(ids) else None,
            "probability_correlation": corr,
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus-dir", type=Path, default=Path(DEFAULT_CORPUS_DIR))
    ap.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    ap.add_argument("--output-root", type=Path, default=Path(DEFAULT_OUTPUT_ROOT))
    ap.add_argument("--tag", default=DEFAULT_TAG)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--c-grid", default="0.25,0.5,1.0,2.0")
    ap.add_argument("--scorer-dir", type=Path, required=True)
    args = ap.parse_args()

    stamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    out_dir = args.output_root / f"{args.tag}_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    c_grid = [float(x) for x in args.c_grid.split(",") if x.strip()]

    all_rows = load_jsonl(args.corpus_dir / "all.jsonl")
    write_json(
        out_dir / "manifest.json",
        {
            "started_at": now_iso(),
            "corpus_dir": str(args.corpus_dir),
            "model_path": args.model_path,
            "device": args.device,
            "scorer_dir": str(args.scorer_dir),
            "c_grid": c_grid,
            "n_examples": len(all_rows),
        },
    )

    extracted = extract_features(all_rows, args.model_path, args.device, out_dir)
    probe = select_best_probe(extracted["region_arrays"], extracted["meta_rows"], extracted["labels"], c_grid)
    for split, rows in probe["predictions"].items():
        write_jsonl(out_dir / f"{split}_predictions.jsonl", rows)

    scorer_summary = load_json(args.scorer_dir / "summary.json")
    scorer_best = scorer_summary["best_variant"]
    comparison = compare_with_text_scorer(probe["predictions"], args.scorer_dir, scorer_best)
    summary = {
        "finished_at": now_iso(),
        "probe": probe["metrics"],
        "text_scorer_comparison": comparison,
    }
    write_json(out_dir / "summary.json", summary)


if __name__ == "__main__":
    main()
