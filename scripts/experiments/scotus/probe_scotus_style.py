#!/usr/bin/env python3
"""Phase 4 SCOTUS justice-style activation probe.

The probe uses matched, case-held-out SCOTUS chunks and captures hidden-state
readouts from a local HuggingFace model. vLLM is intentionally not used here:
it is suitable for serving/generation, but not for activation hooks.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PAIRS = PROJECT_ROOT / "data" / "scotus" / "scotus_matched_pairs_v21.jsonl"
DEFAULT_MODEL = Path("/home/orwel/dev_genius/models/Qwen3.6-27B-FP8")
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "sweep_v4"
DEFAULT_MANIFEST = PROJECT_ROOT / "data" / "scotus" / "scotus_probe_manifest_v1.json"
DEFAULT_LAYERS = "0,4,8,12,16,20,24,28,32,36,40,44,48,52,56,60,63"
PROMPT_TEMPLATES = {
    "normal": (
        "Read the following legal reasoning excerpt and continue the analysis in the same "
        "jurisprudential mode.\n\n"
        "Excerpt:\n{text}\n\n"
        "Continuation:"
    ),
    "variant_a": (
        "Study this Supreme Court-style legal analysis. Continue in the same mode of "
        "legal reasoning without naming any author.\n\n"
        "{text}\n\n"
        "Next paragraph:"
    ),
    "plain": (
        "Legal reasoning excerpt:\n\n{text}\n\nContinue the analysis:"
    ),
}
DIAGNOSTIC_MODES = {
    "normal",
    "excerpt_removed",
    "neutral_filler",
    "label_shuffle",
    "template_variant",
    "plain_prompt",
}
NEUTRAL_FILLER_SENTENCES = [
    "The court begins with the relevant legal standard and applies it to the record before it.",
    "The analysis considers the statutory text, the procedural posture, and the practical consequences of the rule.",
    "The parties dispute how the governing principle should operate in this setting.",
    "The conclusion follows from ordinary interpretive tools and the structure of the legal framework.",
]


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


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


def model_cached(model_path: Path) -> bool:
    if model_path.exists():
        return any(model_path.rglob("*.safetensors")) or any(model_path.rglob("*.bin"))
    hf_home = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface" / "hub"))
    safe_name = "models--" + str(model_path).replace("/", "--")
    model_dir = hf_home / safe_name
    return model_dir.exists() and (any(model_dir.rglob("*.safetensors")) or any(model_dir.rglob("*.bin")))


def split_pair_name(pair: str) -> tuple[str, str]:
    parts = pair.split("_vs_")
    if len(parts) != 2:
        raise ValueError(f"Pair must look like A_vs_B, got {pair!r}")
    return parts[0], parts[1]


def cap_rows(rows: list[dict[str, Any]], caps: dict[str, int], seed: int) -> list[dict[str, Any]]:
    if not caps:
        return rows
    rng = np.random.default_rng(seed)
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["split"], int(row["label"]))].append(row)
    kept: list[dict[str, Any]] = []
    for key, group in sorted(grouped.items()):
        split, _label = key
        cap = caps.get(split, 0)
        if cap <= 0 or len(group) <= cap:
            kept.extend(group)
            continue
        idx = rng.choice(len(group), size=cap, replace=False)
        kept.extend(group[int(i)] for i in sorted(idx.tolist()))
    kept.sort(key=lambda row: (row["split"], row["label"], row["chunk_id"]))
    return kept


def parse_split_caps(raw: str) -> dict[str, int]:
    caps: dict[str, int] = {}
    if not raw.strip():
        return caps
    for part in raw.split(","):
        if not part.strip():
            continue
        split, value = part.split(":", 1)
        caps[split.strip()] = int(value)
    return caps


def parse_slice_filters(raw_filters: list[str]) -> list[tuple[str, str]]:
    filters: list[tuple[str, str]] = []
    for raw in raw_filters:
        for part in raw.split(","):
            if not part.strip():
                continue
            if "=" not in part:
                raise ValueError(f"Slice filter must look like field=value, got {part!r}")
            field, value = part.split("=", 1)
            field = field.strip()
            value = value.strip()
            if not field or not value:
                raise ValueError(f"Slice filter must have non-empty field and value, got {part!r}")
            filters.append((field, value))
    return filters


def row_matches_slice_filters(row: dict[str, Any], filters: list[tuple[str, str]]) -> bool:
    return all(str(row.get(field) or "unknown") == value for field, value in filters)


def apply_slice_filters(rows: list[dict[str, Any]], filters: list[tuple[str, str]]) -> list[dict[str, Any]]:
    if not filters:
        return rows
    return [row for row in rows if row_matches_slice_filters(row, filters)]


def slice_filter_label(filters: list[tuple[str, str]]) -> str:
    if not filters:
        return "none"
    return ",".join(f"{field}={value}" for field, value in filters)


def word_count(text: str) -> int:
    return len(re.findall(r"\w+", text))


def neutral_filler_like(text: str) -> str:
    target_words = max(20, word_count(text))
    words: list[str] = []
    sentence_idx = 0
    while len(words) < target_words:
        words.extend(NEUTRAL_FILLER_SENTENCES[sentence_idx % len(NEUTRAL_FILLER_SENTENCES)].split())
        sentence_idx += 1
    return " ".join(words[:target_words])


def apply_diagnostic_mode(rows: list[dict[str, Any]], mode: str, seed: int) -> list[dict[str, Any]]:
    if mode not in DIAGNOSTIC_MODES:
        raise ValueError(f"Unknown diagnostic mode: {mode}")
    transformed = [dict(row) for row in rows]
    if mode == "excerpt_removed":
        for row in transformed:
            row["original_text_word_count"] = word_count(str(row["text"]))
            row["text"] = "[EXCERPT REMOVED]"
    elif mode == "neutral_filler":
        for row in transformed:
            original = str(row["text"])
            row["original_text_word_count"] = word_count(original)
            row["text"] = neutral_filler_like(original)
    elif mode == "label_shuffle":
        rng = np.random.default_rng(seed)
        grouped: dict[str, list[int]] = defaultdict(list)
        for idx, row in enumerate(transformed):
            grouped[row["split"]].append(idx)
        for split, indices in grouped.items():
            labels = [int(transformed[idx]["label"]) for idx in indices]
            shuffled = rng.permutation(labels).tolist()
            for idx, new_label in zip(indices, shuffled, strict=True):
                transformed[idx]["original_label"] = int(transformed[idx]["label"])
                transformed[idx]["label"] = int(new_label)
                transformed[idx]["label_shuffle_split"] = split
    return transformed


def load_examples(
    pairs_path: Path,
    *,
    pair: str,
    variant: str,
    positive_justice: str | None,
    split_caps: dict[str, int],
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    justice_a, justice_b = split_pair_name(pair)
    positive = positive_justice or justice_b
    label_map = {justice_a: 0, justice_b: 1}
    if positive == justice_a:
        label_map = {justice_a: 1, justice_b: 0}
    elif positive != justice_b:
        raise ValueError(f"Positive justice {positive!r} is not in pair {pair!r}")

    seen: set[str] = set()
    examples: list[dict[str, Any]] = []
    for row in read_jsonl(pairs_path):
        if row.get("pair") != pair or row.get("text_variant") != variant:
            continue
        for side in ("a", "b"):
            chunk_id = str(row[f"chunk_id_{side}"])
            if chunk_id in seen:
                continue
            seen.add(chunk_id)
            justice = row[f"justice_{side}"]
            examples.append(
                {
                    "example_id": f"{pair}|{variant}|{chunk_id}",
                    "pair_id": row["pair_id"],
                    "chunk_id": chunk_id,
                    "split": row["split"],
                    "label": int(label_map[justice]),
                    "justice": justice,
                    "positive_justice": positive,
                    "text": row[f"text_{side}"],
                    "case_id": row[f"case_id_{side}"],
                    "scdb_id": row.get(f"scdb_id_{side}"),
                    "issue_area": row.get("issue_area"),
                    "issue_area_label": row.get("issue_area_label") or "unknown",
                    "opinion_type": row.get(f"opinion_type_{side}") or "unknown",
                    "section_posture": row.get(f"section_posture_{side}") or "unknown",
                    "section_confidence": row.get(f"section_confidence_{side}") or "unknown",
                    "chunk_position_bucket": row.get(f"chunk_position_bucket_{side}") or "unknown",
                    "term": row.get(f"term_{side}"),
                    "decade": row.get("decade") or "unknown",
                    "decision_direction": row.get("decision_direction") or "unknown",
                    "source_url": row.get(f"source_url_{side}"),
                    "token_count": row.get(f"token_count_{side}"),
                    "citation_count": row.get(f"citation_count_{side}"),
                }
            )

    examples = cap_rows(examples, split_caps, seed)
    counts = Counter((row["split"], row["justice"]) for row in examples)
    split_case_map: dict[Any, set[str]] = defaultdict(set)
    for row in examples:
        split_case_map[row["case_id"]].add(row["split"])
    conflicts = sum(1 for splits in split_case_map.values() if len(splits) > 1)
    if conflicts:
        raise RuntimeError(f"Case-held-out split violation: {conflicts} cases appear in multiple splits")
    return examples, {f"{split}/{justice}": count for (split, justice), count in sorted(counts.items())}


def nested_attr(obj: Any, path: str) -> Any | None:
    cur = obj
    for part in path.split("."):
        if not hasattr(cur, part):
            return None
        cur = getattr(cur, part)
    return cur


def transformer_layers(model: torch.nn.Module) -> Any:
    for path in (
        "model.language_model.layers",
        "language_model.layers",
        "model.layers",
        "transformer.h",
        "gpt_neox.layers",
    ):
        layers = nested_attr(model, path)
        if layers is not None:
            return layers
    raise RuntimeError(f"Could not locate transformer layers on {type(model).__name__}")


def parse_layers(spec: str, n_layers: int) -> list[int]:
    if spec.strip().lower() == "all":
        return list(range(n_layers))
    layers: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = (int(x) for x in part.split("-", 1))
            layers.extend(range(start, end + 1))
        else:
            layers.append(int(part))
    valid = sorted({layer for layer in layers if 0 <= layer < n_layers})
    if not valid:
        raise ValueError(f"No valid layers in {spec!r} for model with {n_layers} layers")
    return valid


def build_prompt_content(text: str, template_variant: str) -> str:
    if template_variant not in PROMPT_TEMPLATES:
        raise ValueError(f"Unknown prompt template variant: {template_variant}")
    return PROMPT_TEMPLATES[template_variant].format(text=text.strip())


def chat_text(tokenizer: Any, text: str, template_variant: str, use_chat_template: bool) -> str:
    content = build_prompt_content(text, template_variant)
    if not use_chat_template:
        return content
    messages = [{"role": "user", "content": content}]
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        return content


def token_span_from_chars(offsets: list[tuple[int, int]], start_char: int, end_char: int, length: int) -> tuple[int, int]:
    token_ids = [idx for idx, (start, end) in enumerate(offsets[:length]) if end > start_char and start < end_char]
    if not token_ids:
        return 0, length
    return token_ids[0], token_ids[-1] + 1


def render_batch(
    tokenizer: Any,
    rows: list[dict[str, Any]],
    max_length: int,
    template_variant: str,
    use_chat_template: bool,
) -> tuple[dict[str, torch.Tensor], list[tuple[int, int]]]:
    rendered: list[str] = []
    spans: list[tuple[int, int]] = []
    for row in rows:
        text = str(row["text"]).strip()
        prompt = chat_text(tokenizer, text, template_variant, use_chat_template)
        rendered.append(prompt)
    encoded = tokenizer(
        rendered,
        return_tensors="pt",
        return_offsets_mapping=True,
        padding=True,
        truncation=True,
        max_length=max_length,
        add_special_tokens=False,
    )
    offsets_batch = encoded.pop("offset_mapping")
    lengths = encoded["attention_mask"].sum(dim=1).tolist()
    for prompt, row, offsets_tensor, length_raw in zip(rendered, rows, offsets_batch, lengths, strict=True):
        text = str(row["text"]).strip()
        start = prompt.find(text)
        length = int(length_raw)
        offsets = [(int(start), int(end)) for start, end in offsets_tensor.tolist()]
        if start < 0:
            spans.append((0, length))
        else:
            spans.append(token_span_from_chars(offsets, start, start + len(text), length))
    return dict(encoded), spans


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


def make_classifier(
    c_value: float,
    *,
    solver: str = "lbfgs",
    max_iter: int = 500,
    tol: float = 1e-3,
) -> Pipeline:
    if solver == "sgd":
        alpha = 1.0 / max(c_value * 4000.0, 1e-12)
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    SGDClassifier(
                        loss="log_loss",
                        alpha=alpha,
                        max_iter=max_iter,
                        class_weight="balanced",
                        tol=tol,
                        random_state=17,
                    ),
                ),
            ]
        )
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    max_iter=max_iter,
                    solver=solver,
                    C=c_value,
                    class_weight="balanced",
                    tol=tol,
                ),
            ),
        ]
    )


def make_text_classifier(c_value: float = 1.0) -> Pipeline:
    return Pipeline(
        [
            (
                "features",
                FeatureUnion(
                    [
                        ("word", TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=50000)),
                        (
                            "char",
                            TfidfVectorizer(
                                analyzer="char_wb",
                                ngram_range=(3, 5),
                                min_df=2,
                                max_features=50000,
                            ),
                        ),
                    ]
                ),
            ),
            ("clf", LogisticRegression(max_iter=4000, solver="liblinear", C=c_value, class_weight="balanced")),
        ]
    )


def evaluate_text_baseline(
    rows: list[dict[str, Any]],
    *,
    template_variant: str,
    c_value: float = 1.0,
) -> dict[str, Any]:
    idx = split_indices(rows)
    texts = np.array([build_prompt_content(str(row["text"]), template_variant) for row in rows], dtype=object)
    labels = np.array([int(row["label"]) for row in rows], dtype=np.int64)
    train_idx = idx.get("train", np.array([], dtype=np.int64))
    dev_idx = idx.get("dev", np.array([], dtype=np.int64))
    test_idx = idx.get("test", np.array([], dtype=np.int64))
    train_dev_idx = np.concatenate([train_idx, dev_idx])
    if not len(train_idx) or not len(dev_idx) or not len(test_idx):
        return {"error": "missing train/dev/test split"}
    dev_clf = make_text_classifier(c_value)
    dev_clf.fit(texts[train_idx].tolist(), labels[train_idx])
    final_clf = make_text_classifier(c_value)
    final_clf.fit(texts[train_dev_idx].tolist(), labels[train_dev_idx])
    results: dict[str, Any] = {}
    for split, split_idx, clf in (
        ("dev", dev_idx, dev_clf),
        ("test", test_idx, final_clf),
    ):
        probs = clf.predict_proba(texts[split_idx].tolist())[:, 1]
        preds = (probs >= 0.5).astype(np.int64)
        split_rows = [rows[i] for i in split_idx.tolist()]
        results[split] = vector_metrics(split_rows, preds, probs)
    return results


def load_model_and_tokenizer(model_path: Path, device_map: str) -> tuple[Any, torch.nn.Module]:
    print(f"Model cache status for {model_path}: {model_cached(model_path)}", flush=True)
    if not model_cached(model_path):
        raise RuntimeError(f"Model is not cached locally: {model_path}")
    try:
        from transformers.models.qwen3_5 import modeling_qwen3_5

        modeling_qwen3_5.FusedRMSNormGated = None
        print("Disabled qwen3_5 FusedRMSNormGated for HF activation capture", flush=True)
    except (ImportError, AttributeError):
        pass
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if hasattr(tokenizer, "padding_side"):
        tokenizer.padding_side = "right"
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    resolved_device_map: str | dict[str, int | str]
    if device_map.lower() in {"single", "cuda", "cuda:0", "gpu"}:
        resolved_device_map = {"": 0}
    elif device_map.lower() == "cpu":
        resolved_device_map = {"": "cpu"}
    else:
        resolved_device_map = device_map
    model = AutoModel.from_pretrained(
        str(model_path),
        trust_remote_code=True,
        local_files_only=True,
        torch_dtype="auto",
        device_map=resolved_device_map,
        low_cpu_mem_usage=True,
        attn_implementation="sdpa",
    )
    model.eval()
    return tokenizer, model


def first_parameter_device(model: torch.nn.Module) -> torch.device:
    return next(model.parameters()).device


def first_layer_device(layers_mod: Any) -> torch.device:
    return next(layers_mod[0].parameters()).device


def capture_features(
    rows: list[dict[str, Any]],
    *,
    model_path: Path,
    device_map: str,
    layers_spec: str,
    batch_size: int,
    max_length: int,
    template_variant: str,
    use_chat_template: bool,
    out_dir: Path,
) -> dict[str, Any]:
    tokenizer, model = load_model_and_tokenizer(model_path, device_map)
    layers_mod = transformer_layers(model)
    layers = parse_layers(layers_spec, len(layers_mod))
    print(f"Capturing layers: {layers}", flush=True)

    capture_context: dict[str, Any] = {}
    batch_capture: dict[str, dict[int, np.ndarray]] = defaultdict(dict)
    collected: dict[str, dict[int, list[np.ndarray]]] = defaultdict(lambda: defaultdict(list))
    hooks = []

    def make_hook(layer_idx: int) -> Any:
        def hook(_module: torch.nn.Module, _inp: tuple[Any, ...], out: Any) -> None:
            hidden = out[0] if isinstance(out, tuple) else out
            attention_mask: torch.Tensor = capture_context["attention_mask"].to(hidden.device)
            spans: list[tuple[int, int]] = capture_context["excerpt_spans"]
            lengths = attention_mask.sum(dim=1).long().clamp(min=1)
            batch_ids = torch.arange(hidden.shape[0], device=hidden.device)
            prompt_last = hidden[batch_ids, lengths - 1, :].detach().float().cpu().numpy()
            mask = attention_mask.to(hidden.dtype).unsqueeze(-1)
            prompt_mean = ((hidden * mask).sum(dim=1) / lengths.to(hidden.dtype).unsqueeze(-1)).detach().float().cpu().numpy()
            excerpt_vecs = []
            for row_idx, (start, end) in enumerate(spans):
                safe_start = max(0, min(int(start), int(lengths[row_idx].item()) - 1))
                safe_end = max(safe_start + 1, min(int(end), int(lengths[row_idx].item())))
                excerpt_vecs.append(hidden[row_idx, safe_start:safe_end, :].mean(dim=0).detach().float().cpu().numpy())
            batch_capture["prompt_last"][layer_idx] = prompt_last.astype(np.float32, copy=False)
            batch_capture["prompt_mean"][layer_idx] = prompt_mean.astype(np.float32, copy=False)
            batch_capture["excerpt_mean"][layer_idx] = np.stack(excerpt_vecs, axis=0).astype(np.float32, copy=False)

        return hook

    for layer_idx in layers:
        hooks.append(layers_mod[layer_idx].register_forward_hook(make_hook(layer_idx)))

    meta_rows: list[dict[str, Any]] = []
    model_device = first_layer_device(layers_mod)
    if model_device.type == "cpu" and torch.cuda.is_available():
        model_device = torch.device("cuda:0")
    try:
        with torch.inference_mode():
            for start in tqdm(range(0, len(rows), batch_size), desc="capture", unit="batch"):
                batch_rows = rows[start : start + batch_size]
                encoded, spans = render_batch(tokenizer, batch_rows, max_length, template_variant, use_chat_template)
                batch_capture.clear()
                capture_context["attention_mask"] = encoded["attention_mask"]
                capture_context["excerpt_spans"] = spans
                inputs = {key: value.to(model_device) for key, value in encoded.items()}
                outputs = model(**inputs, use_cache=False)
                for region, layer_map in batch_capture.items():
                    for layer_idx, arr in layer_map.items():
                        for row_vec in arr:
                            collected[region][layer_idx].append(row_vec)
                for row, span, n_tokens in zip(batch_rows, spans, encoded["attention_mask"].sum(dim=1).tolist(), strict=True):
                    meta_rows.append(
                        {
                            **{key: value for key, value in row.items() if key != "text"},
                            "n_prompt_tokens": int(n_tokens),
                            "excerpt_token_span": [int(span[0]), int(span[1])],
                        }
                    )
                del outputs, inputs
                if torch.cuda.is_available() and (start // batch_size) % 25 == 0:
                    torch.cuda.empty_cache()
    finally:
        for handle in hooks:
            handle.remove()

    region_arrays: dict[str, dict[int, np.ndarray]] = {}
    for region, layer_map in collected.items():
        region_arrays[region] = {}
        for layer_idx, vecs in layer_map.items():
            region_arrays[region][layer_idx] = np.stack(vecs, axis=0)

    write_jsonl(out_dir / "feature_meta.jsonl", meta_rows)
    np.savez_compressed(
        out_dir / "features.npz",
        **{
            f"{region}__L{layer_idx:02d}": arr
            for region, layer_map in region_arrays.items()
            for layer_idx, arr in layer_map.items()
        },
    )
    del model, tokenizer, layers_mod
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {
        "regions": region_arrays,
        "meta_rows": meta_rows,
        "labels": np.array([int(row["label"]) for row in meta_rows], dtype=np.int64),
        "layers": layers,
    }


def load_feature_artifacts(run_dir: Path) -> dict[str, Any]:
    meta_rows = read_jsonl(run_dir / "feature_meta.jsonl")
    feature_path = run_dir / "features.npz"
    if not meta_rows:
        raise RuntimeError(f"No feature metadata found in {run_dir}")
    if not feature_path.exists():
        raise RuntimeError(f"No features.npz found in {run_dir}")
    region_arrays: dict[str, dict[int, np.ndarray]] = defaultdict(dict)
    with np.load(feature_path) as data:
        for key in data.files:
            region, layer_raw = key.rsplit("__L", 1)
            region_arrays[region][int(layer_raw)] = data[key]
    layers = sorted({layer for layer_map in region_arrays.values() for layer in layer_map})
    return {
        "regions": dict(region_arrays),
        "meta_rows": meta_rows,
        "labels": np.array([int(row["label"]) for row in meta_rows], dtype=np.int64),
        "layers": layers,
    }


def subset_feature_artifacts(extracted: dict[str, Any], indices: list[int]) -> dict[str, Any]:
    idx_arr = np.array(indices, dtype=np.int64)
    return {
        "regions": {
            region: {layer: arr[idx_arr] for layer, arr in layer_map.items()}
            for region, layer_map in extracted["regions"].items()
        },
        "meta_rows": [extracted["meta_rows"][idx] for idx in indices],
        "labels": extracted["labels"][idx_arr],
        "layers": extracted["layers"],
    }


def split_indices(meta_rows: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    grouped: dict[str, list[int]] = defaultdict(list)
    for idx, row in enumerate(meta_rows):
        grouped[row["split"]].append(idx)
    return {split: np.array(idxs, dtype=np.int64) for split, idxs in grouped.items()}


def predict_metrics(clf: Pipeline, x_matrix: np.ndarray, rows: list[dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
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


def balanced_accuracy_ci(predictions: list[dict[str, Any]], *, seed: int, n_boot: int = 2000) -> dict[str, float]:
    y = np.array([int(row["label"]) for row in predictions], dtype=np.int64)
    pred = np.array([int(row["pred"]) for row in predictions], dtype=np.int64)
    if len(y) == 0:
        return {"low": 0.0, "high": 0.0, "n_boot": 0}
    rng = np.random.default_rng(seed)
    scores: list[float] = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), size=len(y))
        if len(set(y[idx].tolist())) < 2:
            continue
        scores.append(float(balanced_accuracy_score(y[idx], pred[idx])))
    if not scores:
        return {"low": 0.0, "high": 0.0, "n_boot": 0}
    return {
        "low": float(np.percentile(scores, 2.5)),
        "high": float(np.percentile(scores, 97.5)),
        "n_boot": int(len(scores)),
    }


def search_distribution(searches: list[dict[str, Any]]) -> dict[str, Any]:
    dev_scores = np.array([row["dev_metrics"]["balanced_accuracy"] for row in searches], dtype=np.float64)
    test_scores = np.array(
        [row["test_metrics_diagnostic"]["balanced_accuracy"] for row in searches],
        dtype=np.float64,
    )
    by_region_layer: dict[tuple[str, int], dict[str, float]] = {}
    for row in searches:
        key = (str(row["region"]), int(row["layer"]))
        cur = by_region_layer.setdefault(key, {"best_dev": 0.0, "best_test": 0.0})
        cur["best_dev"] = max(cur["best_dev"], float(row["dev_metrics"]["balanced_accuracy"]))
        cur["best_test"] = max(cur["best_test"], float(row["test_metrics_diagnostic"]["balanced_accuracy"]))
    heatmap_rows = [
        {
            "region": region,
            "layer": layer,
            "best_dev_balanced_accuracy": values["best_dev"],
            "best_test_balanced_accuracy_diagnostic": values["best_test"],
        }
        for (region, layer), values in sorted(by_region_layer.items(), key=lambda item: (item[0][0], item[0][1]))
    ]
    return {
        "n_configs": int(len(searches)),
        "dev_balanced_accuracy": {
            "median": float(np.median(dev_scores)) if len(dev_scores) else 0.0,
            "q1": float(np.percentile(dev_scores, 25)) if len(dev_scores) else 0.0,
            "q3": float(np.percentile(dev_scores, 75)) if len(dev_scores) else 0.0,
            "configs_above_0_70": int(np.sum(dev_scores >= 0.70)),
            "configs_above_0_75": int(np.sum(dev_scores >= 0.75)),
            "configs_above_0_80": int(np.sum(dev_scores >= 0.80)),
        },
        "test_balanced_accuracy_diagnostic": {
            "median": float(np.median(test_scores)) if len(test_scores) else 0.0,
            "q1": float(np.percentile(test_scores, 25)) if len(test_scores) else 0.0,
            "q3": float(np.percentile(test_scores, 75)) if len(test_scores) else 0.0,
            "configs_above_0_70": int(np.sum(test_scores >= 0.70)),
            "configs_above_0_75": int(np.sum(test_scores >= 0.75)),
            "configs_above_0_80": int(np.sum(test_scores >= 0.80)),
        },
        "region_layer_heatmap": heatmap_rows,
    }


def select_probe(
    regions: dict[str, dict[int, np.ndarray]],
    meta_rows: list[dict[str, Any]],
    labels: np.ndarray,
    c_grid: list[float],
    *,
    classifier_solver: str,
    classifier_max_iter: int,
    classifier_tol: float,
    test_diagnostic_refit: bool,
) -> dict[str, Any]:
    idx = split_indices(meta_rows)
    train_idx = idx.get("train", np.array([], dtype=np.int64))
    dev_idx = idx.get("dev", np.array([], dtype=np.int64))
    test_idx = idx.get("test", np.array([], dtype=np.int64))
    if not len(train_idx) or not len(dev_idx) or not len(test_idx):
        raise RuntimeError("Expected train/dev/test splits for probe selection")

    searches: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    best_dev_clf: Pipeline | None = None
    total_configs = sum(len(layer_map) * len(c_grid) for layer_map in regions.values())
    with tqdm(total=total_configs, desc="probe sweep", unit="config") as progress:
        for region, layer_map in sorted(regions.items()):
            for layer_idx, x_matrix in sorted(layer_map.items()):
                for c_value in c_grid:
                    clf = make_classifier(
                        c_value,
                        solver=classifier_solver,
                        max_iter=classifier_max_iter,
                        tol=classifier_tol,
                    )
                    clf.fit(x_matrix[train_idx], labels[train_idx])
                    dev_rows = [meta_rows[i] for i in dev_idx.tolist()]
                    dev_metrics, _ = predict_metrics(clf, x_matrix[dev_idx], dev_rows)
                    if test_diagnostic_refit:
                        train_dev_idx = np.concatenate([train_idx, dev_idx])
                        diag_clf = make_classifier(
                            c_value,
                            solver=classifier_solver,
                            max_iter=classifier_max_iter,
                            tol=classifier_tol,
                        )
                        diag_clf.fit(x_matrix[train_dev_idx], labels[train_dev_idx])
                    else:
                        diag_clf = clf
                    test_rows = [meta_rows[i] for i in test_idx.tolist()]
                    test_metrics, _ = predict_metrics(diag_clf, x_matrix[test_idx], test_rows)
                    record = {
                        "region": region,
                        "layer": int(layer_idx),
                        "C": float(c_value),
                        "dev_metrics": dev_metrics,
                        "test_metrics_diagnostic": test_metrics,
                    }
                    searches.append(record)
                    if best is None or (
                        dev_metrics["balanced_accuracy"],
                        dev_metrics["f1"],
                        region,
                        layer_idx,
                    ) > (
                        best["dev_metrics"]["balanced_accuracy"],
                        best["dev_metrics"]["f1"],
                        best["region"],
                        best["layer"],
                    ):
                        best = record
                        best_dev_clf = clf
                    progress.update(1)

    assert best is not None
    assert best_dev_clf is not None
    region = str(best["region"])
    layer = int(best["layer"])
    c_value = float(best["C"])
    x_best = regions[region][layer]
    train_dev_idx = np.concatenate([train_idx, dev_idx])
    final_clf = make_classifier(
        c_value,
        solver=classifier_solver,
        max_iter=classifier_max_iter,
        tol=classifier_tol,
    )
    final_clf.fit(x_best[train_dev_idx], labels[train_dev_idx])

    split_metrics: dict[str, Any] = {}
    split_predictions: dict[str, list[dict[str, Any]]] = {}
    for split, split_idx, clf in (
        ("train", train_idx, best_dev_clf),
        ("dev", dev_idx, best_dev_clf),
        ("test", test_idx, final_clf),
    ):
        rows = [meta_rows[i] for i in split_idx.tolist()]
        metrics, predictions = predict_metrics(clf, x_best[split_idx], rows)
        split_metrics[split] = metrics
        split_predictions[split] = predictions

    searches.sort(
        key=lambda row: (
            row["dev_metrics"]["balanced_accuracy"],
            row["dev_metrics"]["f1"],
            row["region"],
            row["layer"],
        ),
        reverse=True,
    )
    test_ci = balanced_accuracy_ci(split_predictions["test"], seed=17)
    return {
        "best": {
            "region": region,
            "layer": layer,
            "C": c_value,
            "dev_metrics": best["dev_metrics"],
        },
        "split_metrics": split_metrics,
        "test_balanced_accuracy_ci_95": test_ci,
        "predictions": split_predictions,
        "searches": searches,
        "search_distribution": search_distribution(searches),
        "final_clf": final_clf,
    }


def label_counts_for_indices(labels: np.ndarray, indices: np.ndarray) -> Counter[int]:
    return Counter(labels[indices].tolist())


def stress_tests(
    regions: dict[str, dict[int, np.ndarray]],
    meta_rows: list[dict[str, Any]],
    labels: np.ndarray,
    best: dict[str, Any],
    *,
    min_eval_per_label: int,
    classifier_solver: str,
    classifier_max_iter: int,
    classifier_tol: float,
) -> dict[str, list[dict[str, Any]]]:
    idx = split_indices(meta_rows)
    train_dev_idx = np.concatenate([idx["train"], idx["dev"]])
    test_idx = idx["test"]
    x_best = regions[best["region"]][int(best["layer"])]
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
            clf = make_classifier(
                float(best["C"]),
                solver=classifier_solver,
                max_iter=classifier_max_iter,
                tol=classifier_tol,
            )
            clf.fit(x_best[train_idx], labels[train_idx])
            rows = [meta_rows[i] for i in eval_idx.tolist()]
            metrics, _ = predict_metrics(clf, x_best[eval_idx], rows)
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


def save_probe_direction(path: Path, clf: Pipeline, best: dict[str, Any], positive_justice: str) -> None:
    scaler: StandardScaler = clf.named_steps["scaler"]
    logreg: LogisticRegression = clf.named_steps["clf"]
    np.savez_compressed(
        path,
        scaler_mean=scaler.mean_.astype(np.float32),
        scaler_scale=scaler.scale_.astype(np.float32),
        coef=logreg.coef_.astype(np.float32),
        intercept=logreg.intercept_.astype(np.float32),
        region=np.array([best["region"]]),
        layer=np.array([int(best["layer"])]),
        C=np.array([float(best["C"])], dtype=np.float32),
        positive_justice=np.array([positive_justice]),
    )


def write_report(
    path: Path,
    *,
    manifest: dict[str, Any],
    examples: list[dict[str, Any]],
    probe: dict[str, Any],
    stress: dict[str, list[dict[str, Any]]],
    text_baseline: dict[str, Any],
) -> None:
    best = probe["best"]
    ci = probe.get("test_balanced_accuracy_ci_95", {})
    summary_rows = [
        [split, metrics["n"], f"{metrics['accuracy']:.3f}", f"{metrics['balanced_accuracy']:.3f}", f"{metrics['f1']:.3f}"]
        for split, metrics in probe["split_metrics"].items()
    ]
    text_rows = []
    for split in ["dev", "test"]:
        metrics = text_baseline.get(split)
        if metrics:
            text_rows.append(
                [
                    split,
                    metrics["n"],
                    f"{metrics['accuracy']:.3f}",
                    f"{metrics['balanced_accuracy']:.3f}",
                    f"{metrics['f1']:.3f}",
                ]
            )
    top_rows = [
        [
            row["region"],
            row["layer"],
            row["C"],
            f"{row['dev_metrics']['balanced_accuracy']:.3f}",
            f"{row['dev_metrics']['f1']:.3f}",
            f"{row['test_metrics_diagnostic']['balanced_accuracy']:.3f}",
        ]
        for row in probe["searches"][:20]
    ]
    top_test_rows = [
        [
            row["region"],
            row["layer"],
            row["C"],
            f"{row['dev_metrics']['balanced_accuracy']:.3f}",
            f"{row['test_metrics_diagnostic']['balanced_accuracy']:.3f}",
            f"{row['test_metrics_diagnostic']['f1']:.3f}",
        ]
        for row in sorted(
            probe["searches"],
            key=lambda row: (
                row["test_metrics_diagnostic"]["balanced_accuracy"],
                row["test_metrics_diagnostic"]["f1"],
                row["dev_metrics"]["balanced_accuracy"],
            ),
            reverse=True,
        )[:20]
    ]
    dist = probe.get("search_distribution", {})
    dev_dist = dist.get("dev_balanced_accuracy", {})
    test_dist = dist.get("test_balanced_accuracy_diagnostic", {})
    distribution_rows = [
        [
            "dev",
            dist.get("n_configs", 0),
            f"{dev_dist.get('median', 0.0):.3f}",
            f"{dev_dist.get('q1', 0.0):.3f}",
            f"{dev_dist.get('q3', 0.0):.3f}",
            dev_dist.get("configs_above_0_70", 0),
            dev_dist.get("configs_above_0_75", 0),
            dev_dist.get("configs_above_0_80", 0),
        ],
        [
            "test diagnostic",
            dist.get("n_configs", 0),
            f"{test_dist.get('median', 0.0):.3f}",
            f"{test_dist.get('q1', 0.0):.3f}",
            f"{test_dist.get('q3', 0.0):.3f}",
            test_dist.get("configs_above_0_70", 0),
            test_dist.get("configs_above_0_75", 0),
            test_dist.get("configs_above_0_80", 0),
        ],
    ]
    heatmap_rows = [
        [
            row["region"],
            row["layer"],
            f"{row['best_dev_balanced_accuracy']:.3f}",
            f"{row['best_test_balanced_accuracy_diagnostic']:.3f}",
        ]
        for row in dist.get("region_layer_heatmap", [])
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
    counts = Counter((row["split"], row["justice"]) for row in examples)
    count_rows = [[split, justice, count] for (split, justice), count in sorted(counts.items())]
    label_counts = Counter((row["split"], int(row["label"])) for row in examples)
    label_count_rows = [[split, label, count] for (split, label), count in sorted(label_counts.items())]
    lines = [
        "# SCOTUS Activation Probe",
        "",
        f"Started: `{manifest['started_at']}`",
        f"Finished: `{manifest.get('finished_at', '')}`",
        "",
        "## Configuration",
        "",
        markdown_table(
            ["Field", "Value"],
            [
                ["Pair", manifest["pair"]],
                ["Variant", manifest["variant"]],
                ["Model", manifest["model_path"]],
                ["Layers", ", ".join(str(x) for x in manifest["layers"])],
                ["Positive label", manifest["positive_justice"]],
                ["Slice filter", manifest.get("slice_filter", "none")],
                ["Diagnostic mode", manifest.get("diagnostic_mode", "normal")],
                ["Prompt template", manifest.get("prompt_template", "normal")],
                ["Chat template", manifest.get("use_chat_template", True)],
                ["Classifier", manifest.get("classifier", {}).get("description", "balanced logistic regression")],
                [
                    "Diagnostic test refit",
                    manifest.get("classifier", {}).get("test_diagnostic_refit", False),
                ],
                ["Examples", len(examples)],
            ],
        ),
        "",
        "## Example Counts",
        "",
        markdown_table(["Split", "Justice", "Examples"], count_rows),
        "",
        "## Label Counts",
        "",
        markdown_table(["Split", "Label", "Examples"], label_count_rows),
        "",
        "## Best Probe",
        "",
        markdown_table(
            ["Region", "Layer", "C", "Dev balanced accuracy", "Dev F1"],
            [[best["region"], best["layer"], best["C"], f"{best['dev_metrics']['balanced_accuracy']:.3f}", f"{best['dev_metrics']['f1']:.3f}"]],
        ),
        "",
        "## Split Metrics",
        "",
        markdown_table(["Split", "N", "Accuracy", "Balanced accuracy", "F1"], summary_rows),
        "",
        "95% bootstrap CI for final test balanced accuracy: "
        f"`{ci.get('low', 0.0):.3f}` to `{ci.get('high', 0.0):.3f}` "
        f"({ci.get('n_boot', 0)} bootstrap samples).",
        "",
        "## Rendered-Prompt TF-IDF Baseline",
        "",
        markdown_table(["Split", "N", "Accuracy", "Balanced accuracy", "F1"], text_rows),
        "",
        "## Top 20 By Dev Balanced Accuracy",
        "",
        markdown_table(["Region", "Layer", "C", "Dev balanced accuracy", "Dev F1", "Test balanced accuracy"], top_rows),
        "",
        "## Top 20 By Test Balanced Accuracy",
        "",
        "Diagnostic only: these rows are not selection-valid headline results.",
        "",
        markdown_table(["Region", "Layer", "C", "Dev balanced accuracy", "Test balanced accuracy", "Test F1"], top_test_rows),
        "",
        "## Sweep Distribution",
        "",
        markdown_table(
            ["Split", "Configs", "Median", "Q1", "Q3", ">=0.70", ">=0.75", ">=0.80"],
            distribution_rows,
        ),
        "",
        "## Region/Layer Heatmap Table",
        "",
        markdown_table(["Region", "Layer", "Best dev BA", "Best test BA"], heatmap_rows),
        "",
        "## Held-Out Stress Tests",
        "",
        markdown_table(["Field", "Held out", "N eval", "Balanced accuracy", "Accuracy"], stress_rows),
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SCOTUS Phase 4 activation probe.")
    parser.add_argument("--pairs", type=Path, default=DEFAULT_PAIRS)
    parser.add_argument("--features-dir", type=Path, default=None)
    parser.add_argument("--features-output-dir", type=Path, default=None)
    parser.add_argument("--pair", default="Scalia_vs_Ginsburg")
    parser.add_argument("--variant", default="masked")
    parser.add_argument("--positive-justice", default=None)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--manifest-output", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--tag", default="scotus_probe")
    parser.add_argument("--layers", default=DEFAULT_LAYERS)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--split-caps", default="")
    parser.add_argument(
        "--slice-filter",
        action="append",
        default=[],
        help="Filter examples by metadata field equality, e.g. --slice-filter section_posture=majority --slice-filter decade=2000s.",
    )
    parser.add_argument("--c-grid", default="0.25,0.5,1.0,2.0")
    parser.add_argument("--stress-min-eval-per-label", type=int, default=5)
    parser.add_argument("--diagnostic-mode", choices=sorted(DIAGNOSTIC_MODES), default="normal")
    parser.add_argument("--prompt-template", choices=sorted(PROMPT_TEMPLATES), default="normal")
    parser.add_argument("--no-chat-template", action="store_true")
    parser.add_argument("--classifier-solver", default="lbfgs", choices=["lbfgs", "liblinear", "saga", "sgd"])
    parser.add_argument("--classifier-max-iter", type=int, default=500)
    parser.add_argument("--classifier-tol", type=float, default=1e-3)
    parser.add_argument("--test-diagnostic-refit", action="store_true")
    parser.add_argument("--seed", type=int, default=17)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    c_grid = [float(part) for part in args.c_grid.split(",") if part.strip()]
    split_caps = parse_split_caps(args.split_caps)
    slice_filters = parse_slice_filters(args.slice_filter)
    classifier = {
        "description": (
            "balanced logistic regression "
            f"(solver={args.classifier_solver}, max_iter={args.classifier_max_iter}, tol={args.classifier_tol})"
        ),
        "solver": args.classifier_solver,
        "max_iter": args.classifier_max_iter,
        "tol": args.classifier_tol,
        "test_diagnostic_refit": bool(args.test_diagnostic_refit),
    }

    if args.features_dir is not None:
        source_dir = args.features_dir
        source_manifest = read_json(source_dir / "manifest.json")
        source_mode = str(source_manifest.get("diagnostic_mode", "normal"))
        if args.diagnostic_mode != source_mode and args.diagnostic_mode != "label_shuffle":
            raise ValueError(
                "Cached features can only be reused for their original diagnostic mode "
                "or for label_shuffle."
            )
        if args.features_output_dir is not None:
            out_dir = args.features_output_dir
            out_dir.mkdir(parents=True, exist_ok=True)
        elif args.diagnostic_mode == source_mode:
            out_dir = source_dir
        else:
            stamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
            out_dir = args.output_root / f"{args.tag}_{args.diagnostic_mode}_{stamp}"
            out_dir.mkdir(parents=True, exist_ok=True)

        examples = read_jsonl(source_dir / "probe_examples.jsonl")
        if not examples:
            raise RuntimeError(f"No probe examples found in {source_dir}")
        prompt_template = str(source_manifest.get("prompt_template", args.prompt_template))
        use_chat_template = bool(source_manifest.get("use_chat_template", not args.no_chat_template))
        extracted = load_feature_artifacts(source_dir)
        if slice_filters:
            if len(examples) != len(extracted["meta_rows"]):
                raise RuntimeError("Cached example and feature metadata counts do not match; cannot apply slice filter safely")
            keep_indices = [idx for idx, row in enumerate(examples) if row_matches_slice_filters(row, slice_filters)]
            examples = [examples[idx] for idx in keep_indices]
            extracted = subset_feature_artifacts(extracted, keep_indices)
            if not examples:
                raise RuntimeError(f"No cached examples matched slice filter: {slice_filter_label(slice_filters)}")
        manifest = dict(source_manifest)
        manifest["features_source_dir"] = str(source_dir)
        manifest["output_dir"] = str(out_dir)
        manifest["diagnostic_mode"] = args.diagnostic_mode
        manifest["slice_filters"] = [{"field": field, "value": value} for field, value in slice_filters]
        manifest["slice_filter"] = slice_filter_label(slice_filters)
        if out_dir != source_dir:
            manifest["source_started_at"] = source_manifest.get("started_at")
            manifest["started_at"] = now_iso()
        if args.diagnostic_mode == "label_shuffle" and source_mode != "label_shuffle":
            examples = apply_diagnostic_mode(examples, "label_shuffle", args.seed)
            extracted["meta_rows"] = apply_diagnostic_mode(extracted["meta_rows"], "label_shuffle", args.seed)
            extracted["labels"] = np.array([int(row["label"]) for row in extracted["meta_rows"]], dtype=np.int64)
            manifest["label_shuffle_source_mode"] = source_mode
            write_jsonl(out_dir / "probe_examples.jsonl", examples)
        elif out_dir != source_dir:
            write_jsonl(out_dir / "probe_examples.jsonl", examples)
        text_baseline = (
            manifest.get("rendered_prompt_tfidf_baseline")
            if args.diagnostic_mode == source_mode and not slice_filters
            else None
        ) or evaluate_text_baseline(examples, template_variant=prompt_template)
        manifest["finalize_started_at"] = now_iso()
        manifest["c_grid"] = c_grid
        manifest["classifier"] = classifier
    else:
        examples, example_counts = load_examples(
            args.pairs,
            pair=args.pair,
            variant=args.variant,
            positive_justice=args.positive_justice,
            split_caps=split_caps,
            seed=args.seed,
        )
        examples = apply_slice_filters(examples, slice_filters)
        example_counts = {
            f"{split}/{justice}": count
            for (split, justice), count in sorted(Counter((row["split"], row["justice"]) for row in examples).items())
        }
        examples = apply_diagnostic_mode(examples, args.diagnostic_mode, args.seed)
        if not examples:
            raise RuntimeError(f"No probe examples found for slice filter: {slice_filter_label(slice_filters)}")
        prompt_template = args.prompt_template
        if args.diagnostic_mode == "template_variant" and prompt_template == "normal":
            prompt_template = "variant_a"
        if args.diagnostic_mode == "plain_prompt":
            prompt_template = "plain"
        use_chat_template = not args.no_chat_template and args.diagnostic_mode != "plain_prompt"

        stamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
        tag = f"{args.tag}_{args.diagnostic_mode}"
        out_dir = args.output_root / f"{tag}_{stamp}"
        out_dir.mkdir(parents=True, exist_ok=True)
        write_jsonl(out_dir / "probe_examples.jsonl", examples)
        text_baseline = evaluate_text_baseline(examples, template_variant=prompt_template)

        manifest = {
            "started_at": now_iso(),
            "pairs": str(args.pairs),
            "pair": args.pair,
            "variant": args.variant,
            "positive_justice": examples[0]["positive_justice"],
            "model_path": str(args.model_path),
            "output_dir": str(out_dir),
            "device_map": args.device_map,
            "batch_size": args.batch_size,
            "max_length": args.max_length,
            "split_caps": split_caps,
            "slice_filters": [{"field": field, "value": value} for field, value in slice_filters],
            "slice_filter": slice_filter_label(slice_filters),
            "c_grid": c_grid,
            "diagnostic_mode": args.diagnostic_mode,
            "prompt_template": prompt_template,
            "use_chat_template": use_chat_template,
            "example_counts": example_counts,
            "rendered_prompt_tfidf_baseline": text_baseline,
            "classifier": classifier,
        }
        write_json(out_dir / "manifest.json", manifest)

        extracted = capture_features(
            examples,
            model_path=args.model_path,
            device_map=args.device_map,
            layers_spec=args.layers,
            batch_size=args.batch_size,
            max_length=args.max_length,
            template_variant=prompt_template,
            use_chat_template=use_chat_template,
            out_dir=out_dir,
        )
    manifest["layers"] = extracted["layers"]
    probe = select_probe(
        extracted["regions"],
        extracted["meta_rows"],
        extracted["labels"],
        c_grid,
        classifier_solver=args.classifier_solver,
        classifier_max_iter=args.classifier_max_iter,
        classifier_tol=args.classifier_tol,
        test_diagnostic_refit=args.test_diagnostic_refit,
    )
    for split, rows in probe["predictions"].items():
        write_jsonl(out_dir / f"{split}_predictions.jsonl", rows)
    write_jsonl(out_dir / "layer_region_search.jsonl", probe["searches"])
    stress = stress_tests(
        extracted["regions"],
        extracted["meta_rows"],
        extracted["labels"],
        probe["best"],
        min_eval_per_label=args.stress_min_eval_per_label,
        classifier_solver=args.classifier_solver,
        classifier_max_iter=args.classifier_max_iter,
        classifier_tol=args.classifier_tol,
    )
    save_probe_direction(
        out_dir / "best_probe_direction.npz",
        probe["final_clf"],
        probe["best"],
        examples[0]["positive_justice"],
    )
    serializable_probe = {
        "best": probe["best"],
        "split_metrics": probe["split_metrics"],
        "test_balanced_accuracy_ci_95": probe["test_balanced_accuracy_ci_95"],
        "searches_top20": probe["searches"][:20],
    }
    manifest["finished_at"] = now_iso()
    manifest["best_probe"] = probe["best"]
    manifest["split_metrics"] = probe["split_metrics"]
    manifest["test_balanced_accuracy_ci_95"] = probe["test_balanced_accuracy_ci_95"]
    manifest["search_distribution"] = probe["search_distribution"]
    manifest["rendered_prompt_tfidf_baseline"] = text_baseline
    manifest["stress_tests"] = stress
    write_json(
        out_dir / "summary.json",
        {
            "probe": serializable_probe,
            "search_distribution": probe["search_distribution"],
            "rendered_prompt_tfidf_baseline": text_baseline,
            "stress_tests": stress,
        },
    )
    write_report(
        out_dir / "report.md",
        manifest=manifest,
        examples=examples,
        probe=probe,
        stress=stress,
        text_baseline=text_baseline,
    )
    write_json(out_dir / "manifest.json", manifest)
    write_json(args.manifest_output, manifest)
    print(f"Wrote {out_dir / 'report.md'}")
    print(f"Wrote {args.manifest_output}")


if __name__ == "__main__":
    main()
