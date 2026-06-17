#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_DATASET_DIR = "/home/orwel/dev_genius/experiments/Character Creation/sweep_v4/symphonic_voice_probe_dataset_v1"
DEFAULT_MODEL_PATH = "/home/orwel/dev_genius/models/Qwen3.6-35B-A3B"
DEFAULT_OUTPUT_ROOT = "/home/orwel/dev_genius/experiments/Character Creation/sweep_v4"
DEFAULT_TAG = "symphonic_voice_activation_probe_v1"


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


def build_feature_id(row: dict[str, Any]) -> str:
    return f"{row['split']}|{row['behavior']}|{row['merged_item_id']}|{row['anchor_id']}"


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
        kept = {layer: arr for layer, arr in layer_map.items() if layer % layer_stride == 0 or layer == max(layer_map)}
        if kept:
            out[region_name] = kept
    return out


def make_classifier(C: float) -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    max_iter=4000,
                    solver="lbfgs",
                    C=C,
                    class_weight="balanced",
                ),
            ),
        ]
    )


def evaluate_multiclass(rows: list[dict[str, Any]], preds: np.ndarray, probs: np.ndarray, label_names: list[str]) -> dict[str, Any]:
    y = np.array([int(row["anchor_label"]) for row in rows], dtype=np.int64)
    grouped: dict[str, list[int]] = defaultdict(list)
    for idx, row in enumerate(rows):
        grouped[row["anchor_id"]].append(idx)
    by_anchor: dict[str, Any] = {}
    for anchor_id, idxs in sorted(grouped.items()):
        y_a = y[idxs]
        p_a = preds[idxs]
        by_anchor[anchor_id] = {
            "n": int(len(idxs)),
            "accuracy": float(accuracy_score(y_a, p_a)),
            "balanced_accuracy_ovr": float(np.mean([(int(preds[i] == y[i])) for i in idxs])),
            "mean_self_probability": float(np.mean([float(probs[i, y[i]]) for i in idxs])),
        }
    cm = confusion_matrix(y, preds, labels=np.arange(len(label_names), dtype=np.int64))
    return {
        "n": int(len(rows)),
        "accuracy": float(accuracy_score(y, preds)),
        "balanced_accuracy": float(balanced_accuracy_score(y, preds)),
        "macro_f1": float(f1_score(y, preds, average="macro")),
        "label_counts": dict(sorted(Counter(y.tolist()).items())),
        "prediction_counts": dict(sorted(Counter(preds.tolist()).items())),
        "mean_max_probability": float(np.mean(np.max(probs, axis=1))),
        "by_anchor": by_anchor,
        "confusion_matrix": {
            "labels": label_names,
            "matrix": cm.tolist(),
        },
    }


def token_span_from_chars(offsets: list[tuple[int, int]], start_char: int, end_char: int) -> tuple[int, int] | None:
    token_ids = [i for i, (s, e) in enumerate(offsets) if e > start_char and s < end_char]
    if not token_ids:
        return None
    return token_ids[0], token_ids[-1] + 1


def trim_subspan(text: str, start: int, end: int) -> tuple[int, int]:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return start, end


def parse_assistant_regions(text: str) -> dict[str, tuple[int, int]]:
    think_open = text.find("/think")
    think_close = text.find("/end-think")
    response_mark = text.find("Response:")
    if think_open < 0 or think_close < 0 or response_mark < 0:
        raise ValueError("assistant completion missing /think or Response markers")
    think_start = think_open + len("/think")
    think_end = think_close
    response_start = response_mark + len("Response:")
    response_end = len(text)
    think_span = trim_subspan(text, think_start, think_end)
    response_span = trim_subspan(text, response_start, response_end)
    return {
        "assistant": (0, len(text)),
        "think": think_span,
        "response": response_span,
    }


def render_and_spans(
    tokenizer: AutoTokenizer,
    row: dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor, dict[str, tuple[int, int]], int]:
    assistant_text = str(row["assistant_completion"]).strip()
    messages = list(row["messages"]) + [{"role": "assistant", "content": assistant_text}]
    rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    start = rendered.rfind(assistant_text)
    if start < 0:
        raise ValueError(f"assistant text not found in rendered transcript for {build_feature_id(row)}")
    encoded = tokenizer(
        rendered,
        return_tensors="pt",
        return_offsets_mapping=True,
        add_special_tokens=False,
    )
    offsets = [(int(s), int(e)) for s, e in encoded["offset_mapping"][0].tolist()]
    local_spans = parse_assistant_regions(assistant_text)
    token_spans: dict[str, tuple[int, int]] = {}
    for name, (local_start, local_end) in local_spans.items():
        span = token_span_from_chars(offsets, start + local_start, start + local_end)
        if span is None:
            raise ValueError(f"{name} span not found for {build_feature_id(row)}")
        token_spans[name] = span
    prompt_last_idx = token_spans["assistant"][0] - 1
    return encoded["input_ids"], encoded["attention_mask"], token_spans, prompt_last_idx


def span_mean(h: torch.Tensor, start: int, end: int) -> np.ndarray:
    return h[start:end].mean(dim=0).float().cpu().numpy()


def region_vectors(
    hidden_states: tuple[torch.Tensor, ...],
    spans: dict[str, tuple[int, int]],
    prompt_last_idx: int,
) -> dict[str, list[np.ndarray]]:
    a_start, a_end = spans["assistant"]
    t_start, t_end = spans["think"]
    r_start, r_end = spans["response"]
    regions: dict[str, list[np.ndarray]] = {
        "assistant_mean": [],
        "assistant_last16": [],
        "think_mean": [],
        "think_first16": [],
        "think_last16": [],
        "response_mean": [],
        "response_first16": [],
        "response_last16": [],
        "response_last": [],
        "prompt_last": [],
    }
    for layer_h in hidden_states[1:]:
        h = layer_h[0]
        regions["assistant_mean"].append(span_mean(h, a_start, a_end))
        regions["assistant_last16"].append(span_mean(h, max(a_start, a_end - 16), a_end))
        regions["think_mean"].append(span_mean(h, t_start, t_end))
        regions["think_first16"].append(span_mean(h, t_start, min(t_end, t_start + 16)))
        regions["think_last16"].append(span_mean(h, max(t_start, t_end - 16), t_end))
        regions["response_mean"].append(span_mean(h, r_start, r_end))
        regions["response_first16"].append(span_mean(h, r_start, min(r_end, r_start + 16)))
        regions["response_last16"].append(span_mean(h, max(r_start, r_end - 16), r_end))
        regions["response_last"].append(h[r_end - 1].float().cpu().numpy())
        regions["prompt_last"].append(h[prompt_last_idx].float().cpu().numpy())
    return regions


def extract_features(
    rows: list[dict[str, Any]],
    model_path: str,
    device_map: str,
    out_dir: Path,
    *,
    max_gpu_gib: int | None,
    max_cpu_gib: int | None,
    offload_folder: Path | None,
) -> dict[str, Any]:
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    extra_model_kwargs: dict[str, Any] = {}
    if max_gpu_gib is not None or max_cpu_gib is not None:
        max_memory: dict[Any, str] = {}
        if max_gpu_gib is not None:
            max_memory[0] = f"{int(max_gpu_gib)}GiB"
        if max_cpu_gib is not None:
            max_memory["cpu"] = f"{int(max_cpu_gib)}GiB"
        extra_model_kwargs["max_memory"] = max_memory
    if offload_folder is not None:
        offload_folder.mkdir(parents=True, exist_ok=True)
        extra_model_kwargs["offload_folder"] = str(offload_folder)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map=device_map,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        **extra_model_kwargs,
    )
    model.eval()
    model_device = next(model.parameters()).device

    anchor_ids = sorted({row["anchor_id"] for row in rows})
    anchor_to_label = {anchor_id: idx for idx, anchor_id in enumerate(anchor_ids)}

    all_meta: list[dict[str, Any]] = []
    collected: dict[str, dict[int, list[np.ndarray]]] = defaultdict(lambda: defaultdict(list))
    labels: list[int] = []

    with torch.inference_mode():
        for idx, row in enumerate(rows, start=1):
            input_ids, attention_mask, spans, prompt_last_idx = render_and_spans(tokenizer, row)
            input_ids = input_ids.to(model_device)
            attention_mask = attention_mask.to(model_device)
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                use_cache=False,
            )
            regions = region_vectors(outputs.hidden_states, spans, prompt_last_idx)
            label = anchor_to_label[row["anchor_id"]]
            labels.append(label)
            all_meta.append(
                {
                    "feature_id": build_feature_id(row),
                    "split": row["split"],
                    "behavior": row["behavior"],
                    "merged_item_id": row["merged_item_id"],
                    "anchor_id": row["anchor_id"],
                    "anchor_label": label,
                    "source_title": row.get("source_title", ""),
                    "title": row.get("title", ""),
                    "assistant_token_span": [int(spans["assistant"][0]), int(spans["assistant"][1])],
                    "think_token_span": [int(spans["think"][0]), int(spans["think"][1])],
                    "response_token_span": [int(spans["response"][0]), int(spans["response"][1])],
                    "n_total_tokens": int(input_ids.shape[1]),
                    "n_assistant_tokens": int(spans["assistant"][1] - spans["assistant"][0]),
                    "n_think_tokens": int(spans["think"][1] - spans["think"][0]),
                    "n_response_tokens": int(spans["response"][1] - spans["response"][0]),
                }
            )
            for region_name, vecs in regions.items():
                for layer_idx, vec in enumerate(vecs):
                    collected[region_name][layer_idx].append(vec.astype(np.float32, copy=False))
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
    write_json(out_dir / "label_map.json", {"anchor_ids": anchor_ids, "anchor_to_label": anchor_to_label})
    return {
        "region_arrays": region_arrays,
        "meta_rows": all_meta,
        "labels": np.array(labels, dtype=np.int64),
        "anchor_ids": anchor_ids,
    }


def select_best_probe(
    region_arrays: dict[str, dict[int, np.ndarray]],
    meta_rows: list[dict[str, Any]],
    labels: np.ndarray,
    anchor_ids: list[str],
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
            for C in c_grid:
                clf = make_classifier(C)
                clf.fit(X_train, y_train)
                val_probs = clf.predict_proba(X_val)
                val_preds = np.argmax(val_probs, axis=1)
                val_rows = [meta_rows[i] for i in val_idx.tolist()]
                val_metrics = evaluate_multiclass(val_rows, val_preds, val_probs, anchor_ids)
                record = {
                    "region": region_name,
                    "layer": int(layer_idx),
                    "C": float(C),
                    "val_metrics": val_metrics,
                }
                searches.append(record)
                if best is None or val_metrics["balanced_accuracy"] > best["val_metrics"]["balanced_accuracy"] or (
                    val_metrics["balanced_accuracy"] == best["val_metrics"]["balanced_accuracy"]
                    and val_metrics["macro_f1"] > best["val_metrics"]["macro_f1"]
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
    split_metrics: dict[str, Any] = {
        "selection": {
            "best_region": region,
            "best_layer": layer,
            "best_C": C,
        }
    }
    for split_name, idxs, split_clf in (
        ("train", train_idx, best_train_clf),
        ("val", val_idx, best_train_clf),
        ("test", test_idx, clf),
    ):
        X_split = X[idxs]
        probs = split_clf.predict_proba(X_split)
        preds = np.argmax(probs, axis=1)
        rows = [meta_rows[i] for i in idxs.tolist()]
        split_metrics[split_name] = evaluate_multiclass(rows, preds, probs, anchor_ids)
        split_preds[split_name] = [
            {
                "feature_id": rows[i]["feature_id"],
                "split": split_name,
                "behavior": rows[i]["behavior"],
                "anchor_id": rows[i]["anchor_id"],
                "anchor_label": int(rows[i]["anchor_label"]),
                "pred_label": int(preds[i]),
                "pred_anchor_id": anchor_ids[int(preds[i])],
                "self_probability": float(probs[i, int(rows[i]["anchor_label"])]),
                "max_probability": float(np.max(probs[i])),
                "correct": bool(int(preds[i]) == int(rows[i]["anchor_label"])),
                "region": region,
                "layer": layer,
            }
            for i in range(len(rows))
        ]
    searches.sort(
        key=lambda row: (
            row["val_metrics"]["balanced_accuracy"],
            row["val_metrics"]["macro_f1"],
            row["region"],
            row["layer"],
        ),
        reverse=True,
    )
    return {"metrics": split_metrics, "predictions": split_preds, "searches": searches}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-dir", type=Path, required=True)
    ap.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    ap.add_argument("--output-root", type=Path, default=Path(DEFAULT_OUTPUT_ROOT))
    ap.add_argument("--tag", default=DEFAULT_TAG)
    ap.add_argument("--device-map", default="auto")
    ap.add_argument("--c-grid", default="0.25,0.5,1.0,2.0")
    ap.add_argument("--max-gpu-gib", type=int, default=None)
    ap.add_argument("--max-cpu-gib", type=int, default=None)
    ap.add_argument("--offload-folder", type=Path, default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--reuse-features-dir", type=Path, default=None)
    ap.add_argument("--region-allowlist", default="")
    ap.add_argument("--layer-stride", type=int, default=1)
    args = ap.parse_args()

    stamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    out_dir = args.output_root / f"{args.tag}_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    c_grid = [float(x) for x in args.c_grid.split(",") if x.strip()]

    rows = load_jsonl(args.dataset_dir / "all_completions.jsonl")
    if args.limit is not None:
        rows = rows[: args.limit]
    write_json(
        out_dir / "manifest.json",
        {
            "started_at": now_iso(),
            "dataset_dir": str(args.dataset_dir),
            "reuse_features_dir": str(args.reuse_features_dir) if args.reuse_features_dir is not None else None,
            "model_path": args.model_path,
            "device_map": args.device_map,
            "c_grid": c_grid,
            "max_gpu_gib": args.max_gpu_gib,
            "max_cpu_gib": args.max_cpu_gib,
            "offload_folder": str(args.offload_folder) if args.offload_folder is not None else None,
            "limit": args.limit,
            "region_allowlist": [x.strip() for x in args.region_allowlist.split(",") if x.strip()],
            "layer_stride": args.layer_stride,
            "n_examples": len(rows),
        },
    )

    region_allowlist = [x.strip() for x in args.region_allowlist.split(",") if x.strip()]
    if args.reuse_features_dir is not None:
        meta_rows = load_jsonl(args.reuse_features_dir / "feature_meta.jsonl")
        label_map = load_json(args.reuse_features_dir / "label_map.json")
        region_arrays = parse_region_arrays(args.reuse_features_dir / "features.npz")
        region_arrays = filter_region_arrays(
            region_arrays,
            region_allowlist=region_allowlist or None,
            layer_stride=max(1, int(args.layer_stride)),
        )
        extracted = {
            "region_arrays": region_arrays,
            "meta_rows": meta_rows,
            "labels": np.array([int(row["anchor_label"]) for row in meta_rows], dtype=np.int64),
            "anchor_ids": label_map["anchor_ids"],
        }
    else:
        extracted = extract_features(
            rows,
            args.model_path,
            args.device_map,
            out_dir,
            max_gpu_gib=args.max_gpu_gib,
            max_cpu_gib=args.max_cpu_gib,
            offload_folder=args.offload_folder,
        )
        if region_allowlist or args.layer_stride > 1:
            extracted["region_arrays"] = filter_region_arrays(
                extracted["region_arrays"],
                region_allowlist=region_allowlist or None,
                layer_stride=max(1, int(args.layer_stride)),
            )
    probe = select_best_probe(
        extracted["region_arrays"],
        extracted["meta_rows"],
        extracted["labels"],
        extracted["anchor_ids"],
        c_grid,
    )
    for split, pred_rows in probe["predictions"].items():
        write_jsonl(out_dir / f"{split}_predictions.jsonl", pred_rows)
    write_jsonl(out_dir / "searches.jsonl", probe["searches"])
    write_json(
        out_dir / "summary.json",
        {
            "finished_at": now_iso(),
            "probe": probe["metrics"],
        },
    )


if __name__ == "__main__":
    main()
