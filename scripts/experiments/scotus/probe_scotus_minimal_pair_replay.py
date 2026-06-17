#!/usr/bin/env python3
"""Probe assistant-internal activations from controlled SCOTUS minimal pairs.

This is a follow-up to the negative broad justice-style and Commerce-pocket
steering runs. The user prompt is held fixed for each fact pattern while the
assistant answer is replayed in two legal reasoning frames. This lets us test
assistant-internal regions instead of prompt/excerpt readouts.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from tqdm import tqdm


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from probe_scotus_style import (  # noqa: E402
    DEFAULT_MODEL,
    load_model_and_tokenizer,
    make_classifier,
    markdown_table,
    predict_metrics,
    split_indices,
    transformer_layers,
    write_json,
    write_jsonl,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "sweep_v4"
DEFAULT_REPORT_DIR = PROJECT_ROOT / "reports"
DEFAULT_LAYERS = "4,8,12,16,20,24"
DEFAULT_C_GRID = "0.001,0.003,0.01,0.03,0.1,0.3,1.0"


COMMERCE_FACTS = [
    (
        "Congress creates a federal civil remedy for local violent conduct after finding that such conduct "
        "has aggregate effects on employment, healthcare costs, and interstate travel."
    ),
    (
        "Congress makes possession of a firearm near a school a federal crime without requiring proof that "
        "the possession has a connection to interstate commerce."
    ),
    (
        "Congress restricts homegrown production of a fungible commodity kept for personal use because local "
        "production may affect the national market in the aggregate."
    ),
    (
        "Congress authorizes statutory damages for false consumer-credit reporting sent to lenders that operate "
        "across state lines."
    ),
    (
        "Congress regulates price-fixing by local suppliers whose commodity is sold through a national market."
    ),
    (
        "Congress requires local schools to teach a particular financial-literacy curriculum because education "
        "quality affects the national economy."
    ),
    (
        "Congress imposes civil penalties on operators using a national shipping network even when the charged "
        "violation occurred inside one state."
    ),
    (
        "Congress creates a federal remedy against deceptive online sellers whose transactions use interstate "
        "payment networks."
    ),
    (
        "Congress punishes arson of property used in an activity affecting commerce and applies the law to a "
        "private owner-occupied home with no business use."
    ),
    (
        "Congress requires unpaid home care for elderly relatives, citing national healthcare spending and "
        "labor-market effects."
    ),
    (
        "Congress regulates a small intrastate manufacturer as part of a national price-stabilization scheme "
        "for a fungible good."
    ),
    (
        "Congress creates statutory damages for misleading labels on goods sold through nationwide distribution "
        "channels."
    ),
    (
        "Congress bars a purely local youth-sports dispute from state court after finding that youth athletics "
        "affect future economic productivity."
    ),
    (
        "Congress regulates safety practices for local warehouses that store goods awaiting interstate shipment."
    ),
    (
        "Congress imposes a federal reporting duty on local clinics because inaccurate data can distort a "
        "national healthcare market."
    ),
    (
        "Congress creates a federal cause of action for neighborhood vandalism after compiling findings about "
        "aggregate insurance costs."
    ),
    (
        "Congress regulates home repair contracts advertised and financed through national online platforms."
    ),
    (
        "Congress criminalizes possession of ordinary household tools near public buildings, citing possible "
        "effects on commercial activity."
    ),
    (
        "Congress sets liability rules for local brokers who transmit purchase orders through interstate "
        "securities systems."
    ),
    (
        "Congress requires family budgeting classes in every public school because financial literacy affects "
        "consumer markets."
    ),
    (
        "Congress regulates cultivation of a fungible agricultural product even when each grower uses the product "
        "only at home."
    ),
    (
        "Congress creates federal damages for deceptive subscription renewals processed by interstate credit-card "
        "networks."
    ),
    (
        "Congress creates a federal tort remedy for local bullying after finding aggregate economic costs from "
        "school disruption."
    ),
    (
        "Congress regulates local storage depots that feed goods into interstate retail supply chains."
    ),
]


def split_for_index(index: int) -> str:
    if index < 14:
        return "train"
    if index < 19:
        return "dev"
    return "test"


def user_prompt(fact: str) -> str:
    return (
        f"Fact pattern: {fact}\n\n"
        "Write a concise judicial analysis of Congress's power. Address both the Commerce Clause and federalism limits."
    )


def assistant_text(fact: str, label: int, variant: int) -> str:
    if label == 1:
        openings = [
            "Holding: Congress lacks authority on these facts.",
            "Holding: The statute exceeds Congress's enumerated power.",
            "Holding: The federal law cannot be sustained under the Commerce Clause.",
        ]
        reasons = [
            (
                "Reasoning: The regulated conduct is local and non-economic. Lopez and Morrison mark the line "
                "between commerce and a general police power. Legislative findings about aggregate economic "
                "effects cannot convert a local noneconomic matter into commerce. Accepting that chain of "
                "inference would leave no meaningful limit on federal power and would displace traditional "
                "state regulation."
            ),
            (
                "Reasoning: The statute lacks a concrete jurisdictional hook tying the particular conduct to "
                "interstate trade. Congress may regulate channels, instrumentalities, and economic activity "
                "that substantially affects commerce, but it may not reach every local problem by reciting "
                "downstream economic consequences. The federalism structure requires a limiting principle."
            ),
            (
                "Reasoning: The activity is not part of a broader market regulation of economic production or "
                "exchange. It is closer to local crime, family life, education, or property regulation. Those "
                "subjects are traditionally governed by the States, and Commerce Clause doctrine does not "
                "permit Congress to erase that boundary through attenuated aggregate-effects reasoning."
            ),
        ]
        return f"{openings[variant % len(openings)]}\n\n{reasons[variant % len(reasons)]}"
    openings = [
        "Holding: Congress has authority to regulate this class of activity.",
        "Holding: The statute is a valid exercise of the commerce power.",
        "Holding: The federal remedy falls within Congress's Commerce Clause authority.",
    ]
    reasons = [
        (
            "Reasoning: The statute targets economic conduct connected to a national market. Wickard and Raich "
            "permit Congress to regulate intrastate instances when the class of activity, viewed in the "
            "aggregate, substantially affects interstate commerce. The national market would be undermined if "
            "local participants could opt out one transaction at a time."
        ),
        (
            "Reasoning: The relevant activity uses channels or instrumentalities of interstate commerce and "
            "forms part of a broader regulatory scheme. Congress may choose civil penalties or statutory "
            "damages to make that scheme effective. State remedies may coexist, but traditional state authority "
            "does not defeat a valid federal rule directed at interstate commercial networks."
        ),
        (
            "Reasoning: The regulated transactions are commercial in character and tied to interstate systems "
            "for distribution, credit, payment, shipping, or market pricing. Congress had a rational basis for "
            "treating the local conduct as part of an economic class whose aggregate effects are substantial."
        ),
    ]
    return f"{openings[variant % len(openings)]}\n\n{reasons[variant % len(reasons)]}"


def build_examples() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for fact_idx, fact in enumerate(COMMERCE_FACTS):
        split = split_for_index(fact_idx)
        for label in (0, 1):
            label_name = "commerce_limits" if label == 1 else "commerce_authority"
            example_id = f"commerce_minpair|{fact_idx:02d}|{label_name}"
            rows.append(
                {
                    "example_id": example_id,
                    "chunk_id": example_id,
                    "pair_id": f"commerce_minpair|{fact_idx:02d}",
                    "fact_id": f"commerce_fact_{fact_idx:02d}",
                    "split": split,
                    "label": int(label),
                    "justice": label_name,
                    "positive_justice": "commerce_limits",
                    "frame_task": "commerce_limits_vs_authority",
                    "frame_label": label_name,
                    "issue_area_label": "Economic Activity",
                    "opinion_type": "minimal_pair_replay",
                    "section_posture": "assistant_replay",
                    "prompt": user_prompt(fact),
                    "assistant_text": assistant_text(fact, label, fact_idx),
                    "text": assistant_text(fact, label, fact_idx),
                }
            )
    return rows


def parse_layers(spec: str, n_layers: int) -> list[int]:
    out: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        layer = int(part)
        if layer < 0 or layer >= n_layers:
            raise ValueError(f"Layer {layer} outside 0..{n_layers - 1}")
        out.append(layer)
    if not out:
        raise ValueError("No layers requested")
    return sorted(set(out))


def chat_text(tokenizer: Any, messages: list[dict[str, str]], *, add_generation_prompt: bool) -> str:
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
                enable_thinking=False,
            )
        except TypeError:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
            )
    text = ""
    for message in messages:
        text += f"{message['role'].upper()}: {message['content']}\n"
    if add_generation_prompt:
        text += "ASSISTANT: "
    return text


def char_span_to_token_span(offsets: list[tuple[int, int]], start: int, end: int) -> tuple[int, int]:
    token_indices = [idx for idx, (tok_start, tok_end) in enumerate(offsets) if tok_end > start and tok_start < end]
    if not token_indices:
        return (0, 0)
    return (min(token_indices), max(token_indices) + 1)


def assistant_regions(assistant: str, assistant_token_count: int, offsets: list[tuple[int, int]]) -> dict[str, tuple[int, int]]:
    regions: dict[str, tuple[int, int]] = {
        "assistant_all": (0, assistant_token_count),
        "assistant_early": (0, min(32, assistant_token_count)),
        "assistant_late": (max(0, assistant_token_count - 32), assistant_token_count),
    }
    holding_start = assistant.find("Holding:")
    reasoning_start = assistant.find("Reasoning:")
    if holding_start >= 0:
        holding_end = reasoning_start if reasoning_start > holding_start else len(assistant)
        regions["holding_region"] = char_span_to_token_span(offsets, holding_start, holding_end)
    if reasoning_start >= 0:
        regions["reasoning_region"] = char_span_to_token_span(offsets, reasoning_start, len(assistant))
    return regions


def capture_replay_features(
    rows: list[dict[str, Any]],
    *,
    model_path: Path,
    device_map: str,
    layers_spec: str,
    out_dir: Path,
) -> dict[str, Any]:
    tokenizer, model = load_model_and_tokenizer(model_path, device_map)
    layers_mod = transformer_layers(model)
    layers = parse_layers(layers_spec, len(layers_mod))
    hidden_size = int(next(layers_mod[layers[0]].parameters()).shape[-1])
    print(f"Capturing replay layers: {layers}", flush=True)

    collected: dict[str, dict[int, list[np.ndarray]]] = defaultdict(lambda: defaultdict(list))
    meta_rows: list[dict[str, Any]] = []
    model_device = next(model.parameters()).device
    if model_device.type == "cpu" and torch.cuda.is_available():
        model_device = torch.device("cuda:0")

    for idx, row in enumerate(tqdm(rows, desc="replay", unit="seq")):
        messages = [{"role": "user", "content": row["prompt"]}]
        prompt_rendered = chat_text(tokenizer, messages, add_generation_prompt=True)
        full_rendered = chat_text(
            tokenizer,
            messages + [{"role": "assistant", "content": row["assistant_text"]}],
            add_generation_prompt=False,
        )
        prompt_ids = tokenizer(prompt_rendered, add_special_tokens=False).input_ids
        full_inputs = tokenizer(full_rendered, add_special_tokens=False, return_tensors="pt")
        assistant_enc = tokenizer(row["assistant_text"], add_special_tokens=False, return_offsets_mapping=True)
        assistant_token_count = len(assistant_enc.input_ids)
        assistant_start = len(prompt_ids)
        total_tokens = int(full_inputs["input_ids"].shape[-1])
        if assistant_token_count <= 0:
            raise RuntimeError(f"Empty assistant tokens for {row['example_id']}")
        if total_tokens < assistant_start + assistant_token_count:
            raise RuntimeError(
                f"Assistant span overflow for {row['example_id']}: total={total_tokens} "
                f"assistant_start={assistant_start} assistant_tokens={assistant_token_count}"
            )
        regions = assistant_regions(
            row["assistant_text"],
            assistant_token_count,
            [(int(a), int(b)) for a, b in assistant_enc["offset_mapping"]],
        )
        full_inputs = {key: value.to(model_device) for key, value in full_inputs.items()}
        with torch.inference_mode():
            outputs = model(**full_inputs, use_cache=False, output_hidden_states=True)
        hidden_states = outputs.hidden_states[1:]
        for layer_idx in layers:
            seq_hs = hidden_states[layer_idx][0]
            prompt_last = seq_hs[assistant_start - 1].detach().float().cpu().numpy().astype(np.float32, copy=False)
            collected["prompt_last"][layer_idx].append(prompt_last)
            for region_name, (start, end) in regions.items():
                if end <= start:
                    vec = np.full((hidden_size,), np.nan, dtype=np.float32)
                else:
                    slice_start = assistant_start + start
                    slice_end = min(assistant_start + end, total_tokens)
                    vec = (
                        seq_hs[slice_start:slice_end]
                        .mean(dim=0)
                        .detach()
                        .float()
                        .cpu()
                        .numpy()
                        .astype(np.float32, copy=False)
                    )
                collected[region_name][layer_idx].append(vec)
        meta_rows.append(
            {
                **{key: value for key, value in row.items() if key not in {"text", "assistant_text"}},
                "assistant_text": row["assistant_text"],
                "assistant_tokens": int(assistant_token_count),
                "total_tokens": int(total_tokens),
                "regions_present": sorted(regions),
            }
        )
        del outputs, hidden_states, full_inputs
        if torch.cuda.is_available() and (idx + 1) % 8 == 0:
            torch.cuda.empty_cache()

    arrays: dict[str, np.ndarray] = {}
    for region, layer_map in collected.items():
        for layer_idx, vecs in layer_map.items():
            arrays[f"{region}__L{layer_idx:02d}"] = np.stack(vecs, axis=0)
    write_jsonl(out_dir / "feature_meta.jsonl", meta_rows)
    np.savez_compressed(out_dir / "features.npz", **arrays)
    del model, tokenizer, layers_mod
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {"meta_rows": meta_rows, "labels": np.array([int(row["label"]) for row in meta_rows], dtype=np.int64), "layers": layers}


@dataclass(frozen=True)
class ProbeResult:
    region: str
    layer: int
    c_value: float
    dev_ba: float
    test_ba: float
    dev_f1: float
    test_f1: float
    clf: Any
    final_metrics: dict[str, Any]


def text_baseline(rows: list[dict[str, Any]], field: str) -> dict[str, dict[str, float]]:
    idx = split_indices(rows)
    labels = np.array([int(row["label"]) for row in rows], dtype=np.int64)
    train_idx = idx["train"]
    train_dev_idx = np.concatenate([idx["train"], idx["dev"]])
    texts = [str(row[field]) for row in rows]
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_features=5000)
    x_train = vectorizer.fit_transform([texts[i] for i in train_idx.tolist()])
    clf = LogisticRegression(max_iter=1000, class_weight="balanced", solver="liblinear", C=1.0)
    clf.fit(x_train, labels[train_idx])

    # Refit on train+dev for final split reporting, matching the activation probe.
    final_vec = TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_features=5000)
    x_train_dev = final_vec.fit_transform([texts[i] for i in train_dev_idx.tolist()])
    final_clf = LogisticRegression(max_iter=1000, class_weight="balanced", solver="liblinear", C=1.0)
    final_clf.fit(x_train_dev, labels[train_dev_idx])
    out: dict[str, dict[str, float]] = {}
    for split, split_idx in idx.items():
        x_split = final_vec.transform([texts[i] for i in split_idx.tolist()])
        y_true = labels[split_idx]
        y_pred = final_clf.predict(x_split)
        out[split] = {
            "n": int(len(split_idx)),
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
            "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        }
    return out


def train_probe(out_dir: Path, c_grid: list[float]) -> tuple[list[dict[str, Any]], ProbeResult]:
    meta_rows = []
    with (out_dir / "feature_meta.jsonl").open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                meta_rows.append(json.loads(line))
    labels = np.array([int(row["label"]) for row in meta_rows], dtype=np.int64)
    idx = split_indices(meta_rows)
    train_idx = idx["train"]
    dev_idx = idx["dev"]
    test_idx = idx["test"]
    train_dev_idx = np.concatenate([train_idx, dev_idx])

    searches: list[dict[str, Any]] = []
    best: ProbeResult | None = None
    with np.load(out_dir / "features.npz") as data:
        for key in data.files:
            region, layer_raw = key.rsplit("__L", 1)
            layer = int(layer_raw)
            x = data[key].astype(np.float32, copy=False)
            if x.shape[0] != len(labels):
                continue
            if np.isnan(x).any():
                continue
            for c_value in c_grid:
                train_clf = make_classifier(c_value, solver="lbfgs", max_iter=1000, tol=1e-3)
                train_clf.fit(x[train_idx], labels[train_idx])
                dev_metrics, _ = predict_metrics(train_clf, x[dev_idx], [meta_rows[i] for i in dev_idx.tolist()])
                test_diag_metrics, _ = predict_metrics(train_clf, x[test_idx], [meta_rows[i] for i in test_idx.tolist()])
                final_clf = make_classifier(c_value, solver="lbfgs", max_iter=1000, tol=1e-3)
                final_clf.fit(x[train_dev_idx], labels[train_dev_idx])
                final_metrics = {}
                for split, split_idx in idx.items():
                    metrics, _ = predict_metrics(final_clf, x[split_idx], [meta_rows[i] for i in split_idx.tolist()])
                    final_metrics[split] = metrics
                row = {
                    "region": region,
                    "layer": layer,
                    "C": c_value,
                    "dev_metrics": dev_metrics,
                    "test_metrics_diagnostic": test_diag_metrics,
                    "final_metrics": final_metrics,
                }
                searches.append(row)
                candidate = ProbeResult(
                    region=region,
                    layer=layer,
                    c_value=c_value,
                    dev_ba=float(dev_metrics["balanced_accuracy"]),
                    test_ba=float(test_diag_metrics["balanced_accuracy"]),
                    dev_f1=float(dev_metrics["f1"]),
                    test_f1=float(test_diag_metrics["f1"]),
                    clf=final_clf,
                    final_metrics=final_metrics,
                )
                if best is None or (candidate.dev_ba, candidate.dev_f1, candidate.test_ba) > (best.dev_ba, best.dev_f1, best.test_ba):
                    best = candidate
    if best is None:
        raise RuntimeError("No valid probe result")
    searches.sort(
        key=lambda row: (
            float(row["dev_metrics"]["balanced_accuracy"]),
            float(row["dev_metrics"]["f1"]),
            float(row["test_metrics_diagnostic"]["balanced_accuracy"]),
        ),
        reverse=True,
    )
    write_jsonl(out_dir / "layer_region_search.jsonl", searches)
    return searches, best


def infer_single_value(rows: list[dict[str, Any]], field: str, fallback: str) -> str:
    values = sorted({str(row.get(field) or "") for row in rows if str(row.get(field) or "")})
    if len(values) == 1:
        return values[0]
    if not values:
        return fallback
    return ",".join(values)


def export_direction(out_dir: Path, best: ProbeResult, *, positive_label: str, task_name: str) -> Path:
    key = f"{best.region}__L{best.layer:02d}"
    with np.load(out_dir / "features.npz") as data:
        x = data[key].astype(np.float32, copy=False)
    meta_rows = []
    with (out_dir / "feature_meta.jsonl").open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                meta_rows.append(json.loads(line))
    labels = np.array([int(row["label"]) for row in meta_rows], dtype=np.int64)
    idx = split_indices(meta_rows)
    train_dev_idx = np.concatenate([idx["train"], idx["dev"]])
    clf = make_classifier(best.c_value, solver="lbfgs", max_iter=1000, tol=1e-3)
    clf.fit(x[train_dev_idx], labels[train_dev_idx])
    scaler = clf.named_steps["scaler"]
    logreg = clf.named_steps["clf"]
    raw = (logreg.coef_[0].astype(np.float32) / np.maximum(scaler.scale_.astype(np.float32), 1e-12)).astype(np.float32)
    raw_norm = float(np.linalg.norm(raw))
    if raw_norm <= 0:
        raise RuntimeError("Zero probe direction")
    raw_unit = raw / raw_norm
    path = out_dir / "best_probe_direction.npz"
    np.savez_compressed(
        path,
        raw_direction_unit=raw_unit.astype(np.float32),
        raw_direction_norm=np.array([raw_norm], dtype=np.float32),
        coef=logreg.coef_.astype(np.float32),
        intercept=logreg.intercept_.astype(np.float32),
        scaler_mean=scaler.mean_.astype(np.float32),
        scaler_scale=scaler.scale_.astype(np.float32),
        region=np.array([best.region]),
        layer=np.array([int(best.layer)]),
        C=np.array([float(best.c_value)], dtype=np.float32),
        positive_justice=np.array([positive_label]),
        task_name=np.array([task_name]),
        source_run=np.array([str(out_dir)]),
    )
    return path


def count_rows(rows: list[dict[str, Any]]) -> list[list[Any]]:
    counts = Counter((row["split"], row["justice"]) for row in rows)
    return [[split, label, count] for (split, label), count in sorted(counts.items())]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
    return rows


def write_report(
    path: Path,
    *,
    out_dir: Path,
    rows: list[dict[str, Any]],
    searches: list[dict[str, Any]],
    best: ProbeResult,
    prompt_text_baseline: dict[str, dict[str, float]],
    assistant_text_baseline: dict[str, dict[str, float]],
    direction_path: Path,
    task_name: str,
    positive_label: str,
) -> None:
    top_rows = [
        [
            row["region"],
            row["layer"],
            row["C"],
            f"{row['dev_metrics']['balanced_accuracy']:.3f}",
            f"{row['test_metrics_diagnostic']['balanced_accuracy']:.3f}",
            f"{row['dev_metrics']['f1']:.3f}",
        ]
        for row in searches[:20]
    ]
    split_rows = [
        [
            split,
            metrics["n"],
            f"{metrics['accuracy']:.3f}",
            f"{metrics['balanced_accuracy']:.3f}",
            f"{metrics['f1']:.3f}",
        ]
        for split, metrics in best.final_metrics.items()
    ]
    prompt_rows = [
        [
            split,
            metrics["n"],
            f"{metrics['accuracy']:.3f}",
            f"{metrics['balanced_accuracy']:.3f}",
            f"{metrics['f1']:.3f}",
        ]
        for split, metrics in prompt_text_baseline.items()
    ]
    assistant_rows = [
        [
            split,
            metrics["n"],
            f"{metrics['accuracy']:.3f}",
            f"{metrics['balanced_accuracy']:.3f}",
            f"{metrics['f1']:.3f}",
        ]
        for split, metrics in assistant_text_baseline.items()
    ]
    lines = [
        "# SCOTUS Minimal-Pair Replay Probe",
        "",
        "## Decision Context",
        "",
        "This is a candidate-generator, not steering evidence. It captures assistant-internal states from controlled minimal pairs where each fact pattern has both legal-frame answers.",
        "",
        "Promotion requires a later causal generation run against random controls.",
        "",
        "## Artifacts",
        "",
        markdown_table(
            ["Artifact", "Path"],
            [
                ["Run dir", out_dir],
                ["Features", out_dir / "features.npz"],
                ["Metadata", out_dir / "feature_meta.jsonl"],
                ["Search", out_dir / "layer_region_search.jsonl"],
                ["Best direction", direction_path],
                ["Task", task_name],
                ["Positive label", positive_label],
            ],
        ),
        "",
        "## Counts",
        "",
        markdown_table(["Split", "Label", "Examples"], count_rows(rows)),
        "",
        "## Best Activation Probe",
        "",
        markdown_table(
            ["Region", "Layer", "C", "Dev BA", "Diagnostic test BA"],
            [[best.region, best.layer, best.c_value, f"{best.dev_ba:.3f}", f"{best.test_ba:.3f}"]],
        ),
        "",
        "## Final Refit Split Metrics",
        "",
        markdown_table(["Split", "N", "Accuracy", "Balanced accuracy", "F1"], split_rows),
        "",
        "## Prompt-Only TF-IDF Baseline",
        "",
        "This should be near chance because the prompt/fact pattern is paired across labels.",
        "",
        markdown_table(["Split", "N", "Accuracy", "Balanced accuracy", "F1"], prompt_rows),
        "",
        "## Assistant-Text TF-IDF Baseline",
        "",
        "This is expected to be high because the replayed answer text contains the target frame.",
        "",
        markdown_table(["Split", "N", "Accuracy", "Balanced accuracy", "F1"], assistant_rows),
        "",
        "## Top Probe Configurations",
        "",
        markdown_table(["Region", "Layer", "C", "Dev BA", "Diagnostic test BA", "Dev F1"], top_rows),
        "",
        "## Read",
        "",
        "- If prompt-only TF-IDF is near chance and assistant-internal activation is high, the design removed prompt/fact leakage but still found answer-state separation.",
        "- This still does not prove a steerable circuit; the exported direction must causally move neutral generation beyond same-layer random controls.",
    ]
    path.write_text("\n".join(str(line) for line in lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe assistant-internal states on SCOTUS minimal-pair replays.")
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--layers", default=DEFAULT_LAYERS)
    parser.add_argument("--c-grid", default=DEFAULT_C_GRID)
    parser.add_argument("--device-map", default="single")
    parser.add_argument("--tag", default="scotus_minpair_replay")
    parser.add_argument("--examples-file", type=Path)
    parser.add_argument("--features-dir", type=Path)
    parser.add_argument("--report-name", default="scotus_minimal_pair_replay_20260501.md")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.features_dir is not None and args.examples_file is None:
        rows = read_jsonl(args.features_dir / "examples.jsonl")
    else:
        rows = read_jsonl(args.examples_file) if args.examples_file else build_examples()
    task_name = infer_single_value(rows, "frame_task", "minimal_pair_replay")
    positive_label = infer_single_value(rows, "positive_justice", "label_1")
    if args.features_dir is not None:
        out_dir = args.features_dir
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = args.output_root / f"{args.tag}_{timestamp}"
        out_dir.mkdir(parents=True, exist_ok=True)
        write_jsonl(out_dir / "examples.jsonl", rows)
    manifest = {
        "created_at": datetime.now().isoformat(),
        "model_path": str(args.model_path),
        "layers": args.layers,
        "c_grid": args.c_grid,
        "examples": len(rows),
        "examples_file": "" if args.examples_file is None else str(args.examples_file),
        "features_dir": "" if args.features_dir is None else str(args.features_dir),
        "task": task_name,
        "positive_label": positive_label,
    }
    write_json(out_dir / "manifest.json", manifest)

    if args.features_dir is None:
        capture_replay_features(
            rows,
            model_path=args.model_path,
            device_map=args.device_map,
            layers_spec=args.layers,
            out_dir=out_dir,
        )
    c_grid = [float(part) for part in args.c_grid.split(",") if part.strip()]
    searches, best = train_probe(out_dir, c_grid)
    direction_path = export_direction(out_dir, best, positive_label=positive_label, task_name=task_name)
    prompt_text_baseline = text_baseline(rows, "prompt")
    assistant_text_baseline = text_baseline(rows, "assistant_text")
    report_path = args.report_dir / args.report_name
    write_report(
        report_path,
        out_dir=out_dir,
        rows=rows,
        searches=searches,
        best=best,
        prompt_text_baseline=prompt_text_baseline,
        assistant_text_baseline=assistant_text_baseline,
        direction_path=direction_path,
        task_name=task_name,
        positive_label=positive_label,
    )
    print(f"Wrote {report_path}")
    print(f"Direction: {direction_path}")


if __name__ == "__main__":
    main()
