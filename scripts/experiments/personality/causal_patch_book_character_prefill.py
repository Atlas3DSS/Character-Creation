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
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_DATASET_DIR = "/home/orwel/dev_genius/experiments/Character Creation/sweep_v4/book_character_prefill_dataset_balanced_v6_reviewed_20260417_151017"
DEFAULT_BEHAVIOR_PROBE_DIR = "/home/orwel/dev_genius/experiments/Character Creation/sweep_v4/book_character_prefill_behavior_probe_v1_20260417_184525"
DEFAULT_MODEL_PATH = "/home/orwel/dev_genius/models/Qwen3.6-35B-A3B"
DEFAULT_OUTPUT_ROOT = "/home/orwel/dev_genius/experiments/Character Creation/sweep_v4"
DEFAULT_TAG = "book_character_prefill_causal_patch_v1"


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


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def linear_probe_prob(vec: np.ndarray, blob: dict[str, Any]) -> float:
    mean = np.asarray(blob["mean"], dtype=np.float32)
    scale = np.asarray(blob["scale"], dtype=np.float32)
    coef = np.asarray(blob["coef"], dtype=np.float32)
    intercept = float(blob["intercept"])
    z = ((vec.astype(np.float32) - mean) / np.maximum(scale, 1e-6)) @ coef + intercept
    return float(sigmoid(np.array([z], dtype=np.float32))[0])


def build_feature_id(row: dict[str, Any]) -> str:
    return f"{row['split']}|{row['behavior']}|{row['merged_item_id']}|{int(row['label'])}"


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


def find_layers(model: torch.nn.Module) -> Any:
    for path in (
        "model.layers",
        "language_model.model.layers",
        "model.language_model.layers",
        "model.language_model.model.layers",
        "language_model.layers",
        "model.model.layers",
    ):
        cur = model
        ok = True
        for part in path.split("."):
            if not hasattr(cur, part):
                ok = False
                break
            cur = getattr(cur, part)
        if ok:
            return cur
    raise RuntimeError(f"Could not locate transformer layers on {type(model).__name__}")


def resolve_patch_target(
    model: torch.nn.Module,
    layers: Any,
    feature_layer_idx: int,
) -> tuple[torch.nn.Module, str]:
    # On this HF/Qwen stack, feature index 0 is the embedding state and feature
    # index N>0 is first changed by hooking transformer block N-1.
    if feature_layer_idx == 0:
        embed = model.get_input_embeddings()
        if embed is None:
            raise RuntimeError("Could not locate input embeddings for layer-0 patching")
        return embed, "input_embeddings"
    return layers[feature_layer_idx - 1], f"block_{feature_layer_idx - 1}"


def patch_hook_factory(start: int, end: int, direction: torch.Tensor, alpha: float):
    def hook(_module, _inputs, output):
        if isinstance(output, tuple):
            hidden = output[0]
            rest = output[1:]
        else:
            hidden = output
            rest = None
        patched = hidden.clone()
        patched[:, start:end, :] = patched[:, start:end, :] + alpha * direction.view(1, 1, -1)
        if rest is None:
            return patched
        return (patched,) + rest

    return hook


def extract_probe_blob(arrays: dict[str, np.ndarray], behavior: str, region: str, layer: int) -> dict[str, Any]:
    prefix = f"{behavior}__{region}__L{layer:02d}"
    return {
        "mean": arrays[f"probe_mean__{prefix}"],
        "scale": arrays[f"probe_scale__{prefix}"],
        "coef": arrays[f"probe_coef__{prefix}"],
        "intercept": float(arrays[f"probe_intercept__{prefix}"][0]),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-dir", type=Path, default=Path(DEFAULT_DATASET_DIR))
    ap.add_argument("--behavior-probe-dir", type=Path, default=Path(DEFAULT_BEHAVIOR_PROBE_DIR))
    ap.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    ap.add_argument("--output-root", type=Path, default=Path(DEFAULT_OUTPUT_ROOT))
    ap.add_argument("--tag", default=DEFAULT_TAG)
    ap.add_argument("--device-map", default="auto")
    ap.add_argument("--alphas", default="0.5,1.0,2.0")
    args = ap.parse_args()

    stamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    out_dir = args.output_root / f"{args.tag}_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    alphas = [float(x) for x in args.alphas.split(",") if x.strip()]

    summary = load_json(args.behavior_probe_dir / "summary.json")
    arrays = np.load(args.behavior_probe_dir / "behavior_probe_artifacts.npz")
    rows = load_jsonl(args.dataset_dir / "all_completions.jsonl")
    eval_rows = [row for row in rows if row["split"] in {"val", "test"}]
    by_behavior_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in eval_rows:
        by_behavior_rows[row["behavior"]].append(row)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map=args.device_map,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    model.eval()
    model_device = next(model.parameters()).device
    layers = find_layers(model)

    manifest = {
        "started_at": now_iso(),
        "dataset_dir": str(args.dataset_dir),
        "behavior_probe_dir": str(args.behavior_probe_dir),
        "model_path": args.model_path,
        "device_map": args.device_map,
        "alphas": alphas,
        "n_eval_rows": len(eval_rows),
    }
    write_json(out_dir / "manifest.json", manifest)

    patch_rows: list[dict[str, Any]] = []
    aggregate: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)

    with torch.inference_mode():
        for behavior, bsum in summary["behaviors"].items():
            print(f"[{now_iso()}] behavior={behavior} start", flush=True)
            selected_regions = {}
            for region in ("think_mean", "response_mean"):
                selected_regions[region] = bsum["best_by_region"][region]

            for region_name, spec in selected_regions.items():
                layer_idx = int(spec["layer"])
                print(f"[{now_iso()}] behavior={behavior} region={region_name} layer={layer_idx} caching_base", flush=True)
                blob = extract_probe_blob(arrays, behavior, region_name, layer_idx)
                direction_np = arrays[f"direction__{behavior}__{region_name}__L{layer_idx:02d}"].astype(np.float32)
                direction = torch.tensor(direction_np, dtype=torch.bfloat16, device=model_device)
                patch_target, patch_target_name = resolve_patch_target(model, layers, layer_idx)

                base_cache: list[dict[str, Any]] = []
                for row in by_behavior_rows[behavior]:
                    input_ids, attention_mask, spans, prompt_last_idx = render_and_spans(tokenizer, row)
                    input_ids = input_ids.to(model_device)
                    attention_mask = attention_mask.to(model_device)
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        output_hidden_states=True,
                        use_cache=False,
                    )
                    base_regions = region_vectors(outputs.hidden_states, spans, prompt_last_idx)
                    base_vec = base_regions[region_name][layer_idx]
                    base_prob = linear_probe_prob(base_vec, blob)
                    del outputs
                    span = spans["think"] if region_name.startswith("think_") else spans["response"]
                    base_cache.append(
                        {
                            "row": row,
                            "input_ids": input_ids,
                            "attention_mask": attention_mask,
                            "spans": spans,
                            "prompt_last_idx": prompt_last_idx,
                            "span": span,
                            "base_prob": base_prob,
                        }
                    )

                for alpha in alphas:
                    print(f"[{now_iso()}] behavior={behavior} region={region_name} layer={layer_idx} alpha={alpha}", flush=True)
                    pos_deltas: list[float] = []
                    neg_deltas: list[float] = []
                    pos_flips = 0
                    neg_flips = 0
                    pos_total = 0
                    neg_total = 0

                    for cached in base_cache:
                        row = cached["row"]
                        input_ids = cached["input_ids"]
                        attention_mask = cached["attention_mask"]
                        spans = cached["spans"]
                        prompt_last_idx = cached["prompt_last_idx"]
                        span = cached["span"]
                        base_prob = float(cached["base_prob"])
                        sign = 1.0 if int(row["label"]) == 0 else -1.0
                        handle = patch_target.register_forward_hook(
                            patch_hook_factory(span[0], span[1], direction, alpha * sign)
                        )
                        try:
                            outputs = model(
                                input_ids=input_ids,
                                attention_mask=attention_mask,
                                output_hidden_states=True,
                                use_cache=False,
                            )
                        finally:
                            handle.remove()
                        patched_regions = region_vectors(outputs.hidden_states, spans, prompt_last_idx)
                        patched_vec = patched_regions[region_name][layer_idx]
                        patched_prob = linear_probe_prob(patched_vec, blob)
                        del outputs

                        row_out = {
                            "feature_id": build_feature_id(row),
                            "split": row["split"],
                            "behavior": behavior,
                            "title": row["title"],
                            "source_title": row["source_title"],
                            "label": int(row["label"]),
                            "region": region_name,
                            "layer": layer_idx,
                            "patch_target": patch_target_name,
                            "alpha": alpha,
                            "base_prob_positive": base_prob,
                            "patched_prob_positive": patched_prob,
                            "delta_prob_positive": patched_prob - base_prob,
                            "base_pred": int(base_prob >= 0.5),
                            "patched_pred": int(patched_prob >= 0.5),
                        }
                        patch_rows.append(row_out)

                        if int(row["label"]) == 0:
                            neg_total += 1
                            neg_deltas.append(patched_prob - base_prob)
                            if base_prob < 0.5 <= patched_prob:
                                neg_flips += 1
                        else:
                            pos_total += 1
                            pos_deltas.append(base_prob - patched_prob)
                            if base_prob >= 0.5 > patched_prob:
                                pos_flips += 1

                    aggregate[behavior][f"{region_name}@alpha={alpha}"] = {
                        "region": region_name,
                        "layer": layer_idx,
                        "patch_target": patch_target_name,
                        "alpha": alpha,
                        "n_fail_eval": neg_total,
                        "n_pass_eval": pos_total,
                        "fail_mean_delta_prob": float(np.mean(neg_deltas)) if neg_deltas else None,
                        "fail_flip_rate": float(neg_flips / max(neg_total, 1)),
                        "pass_mean_delta_prob": float(np.mean(pos_deltas)) if pos_deltas else None,
                        "pass_flip_rate": float(pos_flips / max(pos_total, 1)),
                    }

    write_jsonl(out_dir / "patch_results.jsonl", patch_rows)
    write_json(
        out_dir / "summary.json",
        {
            "finished_at": now_iso(),
            "alphas": alphas,
            "aggregate": aggregate,
        },
    )


if __name__ == "__main__":
    main()
