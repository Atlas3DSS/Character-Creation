#!/usr/bin/env python3
"""Inspect Qwen3.5 J-lens readouts for prepared SCOTUS prefill texts."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.experiments.scotus.jlens_qwen35_pilot import (
    QWEN35_MODEL_PATH,
    LEGAL_MARKERS,
    load_lens,
    load_readout,
    resolve_lens_path,
    top_tokens,
)


DEFAULT_QUEUE = PROJECT_ROOT / "sweep_v4/scotus_prefill_contrasts_20260706_234523/jspace_queue.jsonl"
OPTIONAL_QUEUE_FIELDS = (
    "source_pair_id",
    "generated_pair_id",
    "side",
    "boundary_kind",
    "boundary_index",
    "boundary_label",
    "generated_chars",
    "source_prefill_index",
    "label_a",
    "label_b",
    "chunk_id_a",
    "chunk_id_b",
)


def timestamped_output_dir() -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return PROJECT_ROOT / "sweep_v4" / f"scotus_jlens_prefill_readout_{stamp}"


def now_iso() -> str:
    return datetime.now().astimezone().isoformat()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        Path(tmp_path).replace(path)
    except Exception:
        Path(tmp_path).unlink(missing_ok=True)
        raise


def parse_ints(value: str) -> list[int]:
    return [int(part) for part in value.split(",") if part.strip()]


def load_model(device: str):
    print(f"Loading Qwen3.5 from {QWEN35_MODEL_PATH}")
    print(f"Local model path exists: {QWEN35_MODEL_PATH.exists()}")
    from transformers import AutoModelForImageTextToText, AutoProcessor

    processor = AutoProcessor.from_pretrained(
        QWEN35_MODEL_PATH,
        trust_remote_code=True,
        local_files_only=True,
    )
    model = AutoModelForImageTextToText.from_pretrained(
        QWEN35_MODEL_PATH,
        device_map=device,
        trust_remote_code=True,
        torch_dtype="auto",
        local_files_only=True,
    )
    model.eval()
    print(f"VRAM allocated {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    return model, processor


def extract_hidden_states(model: Any, processor: Any, text: str) -> tuple[tuple[torch.Tensor, ...], int, float]:
    device = next(model.parameters()).device
    inputs = processor(text=[text], return_tensors="pt", padding=True).to(device)
    started = time.time()
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True, use_cache=False)
    elapsed = time.time() - started
    hidden_states = getattr(outputs, "hidden_states", None)
    if hidden_states is None:
        hidden_states = getattr(outputs, "language_model_hidden_states", None)
    if hidden_states is None:
        raise RuntimeError("Model output did not include hidden_states")
    return tuple(hidden_states), int(inputs["input_ids"].shape[1]), elapsed


def resolve_position(position: int, seq_len: int) -> int | None:
    resolved = seq_len + position if position < 0 else position
    if resolved < 0 or resolved >= seq_len:
        return None
    return resolved


def legal_hit_summary(hits: list[dict[str, Any]]) -> list[str]:
    return [f"{hit['marker']}@{hit['rank']}" for hit in hits[:10]]


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    by_axis = Counter(row["comparison_axis"] for row in records)
    by_variant = Counter(row["variant"] for row in records)
    legal_hit_rows = sum(1 for row in records if row["positive_legal_hits"] or row["negative_legal_hits"])
    gains_by_layer: dict[str, list[float]] = defaultdict(list)
    for row in records:
        gains_by_layer[str(row["layer"])].append(float(row["transported_norm"]))
    return {
        "records": len(records),
        "axes": dict(by_axis),
        "variants": dict(by_variant),
        "rows_with_legal_hits": legal_hit_rows,
        "mean_transported_norm_by_layer": {
            layer: sum(values) / max(1, len(values)) for layer, values in sorted(gains_by_layer.items())
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", type=Path, default=DEFAULT_QUEUE)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--max-rows", type=int, default=120)
    parser.add_argument("--layers", default="4,8,16,20,34,48,49,50,51")
    parser.add_argument("--positions", default="-1,-8,-32")
    parser.add_argument("--top-k-tokens", type=int, default=20)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or timestamped_output_dir()
    output_dir.mkdir(parents=True, exist_ok=False)
    records_path = output_dir / "records.jsonl"
    summary_path = output_dir / "summary.json"
    manifest_path = output_dir / "manifest.json"

    queue = read_jsonl(args.queue)[: args.max_rows]
    if not queue:
        raise RuntimeError(f"No J-space queue rows found in {args.queue}")
    layers = parse_ints(args.layers)
    positions = parse_ints(args.positions)

    lens_path = resolve_lens_path()
    lens = load_lens(lens_path)
    available_layers = {int(layer) for layer in lens["J"].keys()}
    missing = sorted(set(layers) - available_layers)
    if missing:
        raise ValueError(f"Requested layers missing from lens: {missing}")

    tokenizer, lm_head, norm_weight, eps = load_readout(torch.device("cpu"))
    model, processor = load_model(args.device)
    manifest = {
        "created_at": now_iso(),
        "script": str(Path(__file__).relative_to(PROJECT_ROOT)),
        "diagnostic_only": True,
        "generation_performed": False,
        "queue": str(args.queue),
        "rows_requested": args.max_rows,
        "rows_loaded": len(queue),
        "layers": layers,
        "positions": positions,
        "layer_convention": "model hidden_states[layer + 1] = output of decoder layer index layer",
        "lens_path": str(lens_path),
        "model_path": str(QWEN35_MODEL_PATH),
        "legal_markers": sorted(LEGAL_MARKERS),
    }
    atomic_json(manifest_path, manifest)

    records: list[dict[str, Any]] = []
    for row in tqdm(queue, desc="prefills", unit="prefill"):
        hidden_states, seq_len, forward_elapsed = extract_hidden_states(model, processor, str(row["text"]))
        for layer in layers:
            layer_hidden = hidden_states[layer + 1][0]
            j_matrix = lens["J"][layer].float()
            for position in positions:
                token_idx = resolve_position(position, seq_len)
                if token_idx is None:
                    continue
                hidden = layer_hidden[token_idx].detach().cpu().float()
                unit = hidden / hidden.norm().clamp_min(1e-12)
                transported = unit @ j_matrix.T
                pos_tokens, pos_hits = top_tokens(
                    transported,
                    tokenizer,
                    lm_head,
                    norm_weight,
                    eps,
                    args.top_k_tokens,
                )
                neg_tokens, neg_hits = top_tokens(
                    -transported,
                    tokenizer,
                    lm_head,
                    norm_weight,
                    eps,
                    args.top_k_tokens,
                )
                record = {
                    "created_at": now_iso(),
                    "jspace_id": row["jspace_id"],
                    "prefill_id": row["prefill_id"],
                    "comparison_axis": row["comparison_axis"],
                    "jspace_compare_group": row["jspace_compare_group"],
                    "variant": row["variant"],
                    "writer_a": row.get("writer_a"),
                    "writer_b": row.get("writer_b"),
                    "posture_a": row.get("posture_a"),
                    "posture_b": row.get("posture_b"),
                    "issue_area_label": row.get("issue_area_label"),
                    "seq_len": seq_len,
                    "forward_elapsed_s": round(forward_elapsed, 3),
                    "layer": layer,
                    "requested_position": position,
                    "token_idx": token_idx,
                    "hidden_norm": float(hidden.norm().item()),
                    "transported_norm": float(transported.norm().item()),
                    "positive_top_tokens": pos_tokens,
                    "positive_legal_hits": pos_hits,
                    "positive_legal_hit_summary": legal_hit_summary(pos_hits),
                    "negative_top_tokens": neg_tokens,
                    "negative_legal_hits": neg_hits,
                    "negative_legal_hit_summary": legal_hit_summary(neg_hits),
                }
                for field in OPTIONAL_QUEUE_FIELDS:
                    if field in row:
                        record[field] = row[field]
                append_jsonl(records_path, record)
                records.append(record)
        atomic_json(summary_path, summarize(records))
        torch.cuda.empty_cache()

    atomic_json(summary_path, summarize(records))
    print(output_dir)


if __name__ == "__main__":
    main()
