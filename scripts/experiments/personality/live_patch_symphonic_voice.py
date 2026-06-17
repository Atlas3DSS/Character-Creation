#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import math
import os
import re
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler
from transformers import AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer


DEFAULT_DATASET_DIR = "/home/orwel/dev_genius/experiments/Character Creation/sweep_v4/symphonic_voice_probe_dataset_v1a_20260417_202525"
DEFAULT_FEATURES_DIR = "/home/orwel/dev_genius/experiments/Character Creation/sweep_v4/symphonic_voice_activation_probe_v1b_capped_20260417_204049"
DEFAULT_AXIS_ANALYSIS_DIR = "/home/orwel/dev_genius/experiments/Character Creation/sweep_v4/symphonic_voice_axis_analysis_v1_20260417_214825"
DEFAULT_ANCHOR_MANIFEST = "/home/orwel/dev_genius/experiments/Character Creation/data/symphonic_voice_anchor_manifest_v1.json"
DEFAULT_MODEL_PATH = "/home/orwel/dev_genius/models/Qwen3.6-35B-A3B"
DEFAULT_OUTPUT_ROOT = "/home/orwel/dev_genius/experiments/Character Creation/sweep_v4"
DEFAULT_TAG = "symphonic_voice_live_patch_v1"


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def log(path: Path, msg: str) -> None:
    line = f"[{now_iso()}] {msg}"
    print(line, flush=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")


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


def apply_chat_text(tokenizer: Any, messages: list[dict[str, str]], *, add_generation_prompt: bool) -> str:
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
    if feature_layer_idx == 0:
        embed = model.get_input_embeddings()
        if embed is None:
            raise RuntimeError("Could not locate input embeddings for layer-0 patching")
        return embed, "input_embeddings"
    return layers[feature_layer_idx - 1], f"block_{feature_layer_idx - 1}"


def canonical_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def parse_assistant_regions(text: str) -> dict[str, tuple[int, int]]:
    text = text.strip()
    think_open = text.find("/think")
    think_close = text.find("/end-think")
    response_mark = text.find("Response:")
    if think_open < 0 or think_close < 0 or response_mark < 0:
        raise ValueError("assistant completion missing /think or Response markers")
    think_start = think_open + len("/think")
    think_end = think_close
    response_start = response_mark + len("Response:")
    response_end = len(text)
    return {
        "assistant": (0, len(text)),
        "think": trim_subspan(text, think_start, think_end),
        "response": trim_subspan(text, response_start, response_end),
    }


def trim_subspan(text: str, start: int, end: int) -> tuple[int, int]:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return start, end


def token_span_from_chars(offsets: list[tuple[int, int]], start_char: int, end_char: int) -> tuple[int, int] | None:
    token_ids = [i for i, (s, e) in enumerate(offsets) if e > start_char and s < end_char]
    if not token_ids:
        return None
    return token_ids[0], token_ids[-1] + 1


def render_for_replay(
    tokenizer: Any,
    messages: list[dict[str, str]],
    assistant_text: str,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, tuple[int, int]], int]:
    rendered = apply_chat_text(
        tokenizer,
        list(messages) + [{"role": "assistant", "content": assistant_text}],
        add_generation_prompt=False,
    )
    start = rendered.rfind(assistant_text)
    if start < 0:
        raise ValueError("assistant text not found in replay transcript")
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
            raise ValueError(f"{name} span not found")
        token_spans[name] = span
    prompt_last_idx = token_spans["assistant"][0] - 1
    return encoded["input_ids"], encoded["attention_mask"], token_spans, prompt_last_idx


def span_mean(h: torch.Tensor, start: int, end: int) -> np.ndarray:
    return h[start:end].mean(dim=0).float().cpu().numpy()


def build_axis_targets(meta_rows: list[dict[str, Any]], anchor_axes: dict[str, dict[str, float]], axis_name: str) -> np.ndarray:
    return np.array([anchor_axes[row["anchor_id"]][axis_name] for row in meta_rows], dtype=np.float64)


def gpu_memory() -> tuple[int, int, float]:
    if not torch.cuda.is_available():
        return 0, 0, 0.0
    free, total = torch.cuda.mem_get_info(0)
    used = total - free
    return used, total, used / max(total, 1)


def guard_vram(max_frac: float, log_path: Path, context: str) -> None:
    used, total, frac = gpu_memory()
    if total and frac > max_frac:
        msg = f"VRAM guard tripped during {context}: used={used/1024**3:.1f}GiB total={total/1024**3:.1f}GiB frac={frac:.3f} max={max_frac:.3f}"
        log(log_path, msg)
        raise RuntimeError(msg)


def load_anchor_axes(anchor_manifest: Path) -> tuple[list[str], dict[str, dict[str, float]]]:
    manifest = load_json(anchor_manifest)
    anchors = manifest["anchors"]
    axis_names = sorted(anchors[0]["stance_axes"].keys())
    axes = {anchor["anchor_id"]: {axis: float(anchor["stance_axes"][axis]) for axis in axis_names} for anchor in anchors}
    return axis_names, axes


def load_model(model_path: Path, dtype: str, log_path: Path) -> tuple[Any, torch.nn.Module, Any]:
    os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    torch_dtype = torch.bfloat16 if dtype == "bfloat16" else torch.float16
    kwargs = dict(
        trust_remote_code=True,
        dtype=torch_dtype,
        device_map={"": 0},
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
    )
    try:
        log(log_path, f"loading {model_path} with AutoModelForCausalLM")
        model = AutoModelForCausalLM.from_pretrained(str(model_path), **kwargs)
    except Exception as exc:  # noqa: BLE001
        log(log_path, f"AutoModelForCausalLM failed: {exc!r}; retrying AutoModelForImageTextToText")
        model = AutoModelForImageTextToText.from_pretrained(str(model_path), **kwargs)
    model.eval()
    layers = find_layers(model)
    return tokenizer, model, layers


def fit_common_models(
    *,
    meta_rows: list[dict[str, Any]],
    X: np.ndarray,
    label_map: dict[str, Any],
    anchor_axes: dict[str, dict[str, float]],
    axis_names: list[str],
    clf_c: float,
    axis_alpha: float,
) -> dict[str, Any]:
    split_idx: dict[str, list[int]] = defaultdict(list)
    for idx, row in enumerate(meta_rows):
        split_idx[row["split"]].append(idx)
    train_idx = np.array(split_idx["train"], dtype=np.int64)
    val_idx = np.array(split_idx["val"], dtype=np.int64)
    test_idx = np.array(split_idx["test"], dtype=np.int64)
    trainval_idx = np.concatenate([train_idx, val_idx], axis=0)

    scaler = StandardScaler().fit(X[trainval_idx])
    Z = scaler.transform(X)
    labels = np.array([int(row["anchor_label"]) for row in meta_rows], dtype=np.int64)

    clf = LogisticRegression(max_iter=4000, solver="lbfgs", C=clf_c, class_weight="balanced")
    clf.fit(Z[trainval_idx], labels[trainval_idx])

    axis_models: dict[str, Ridge] = {}
    for axis_name in axis_names:
        y = build_axis_targets(meta_rows, anchor_axes, axis_name)
        reg = Ridge(alpha=axis_alpha)
        reg.fit(Z[trainval_idx], y[trainval_idx])
        axis_models[axis_name] = reg

    anchor_ids = label_map["anchor_ids"]
    centroids: dict[str, np.ndarray] = {}
    for anchor_id in anchor_ids:
        idxs = [i for i in trainval_idx.tolist() if meta_rows[i]["anchor_id"] == anchor_id]
        centroids[anchor_id] = Z[idxs].mean(axis=0)
    return {
        "scaler": scaler,
        "Z": Z,
        "labels": labels,
        "clf": clf,
        "axis_models": axis_models,
        "centroids": centroids,
        "anchor_ids": anchor_ids,
        "train_idx": train_idx,
        "val_idx": val_idx,
        "test_idx": test_idx,
    }


def select_pairs_from_axis_analysis(
    axis_analysis_dir: Path,
    focus_axes: list[str],
    top_k_per_axis: int,
    max_pairs: int,
) -> list[tuple[str, str]]:
    rows = load_jsonl(axis_analysis_dir / "pairwise_anchor_axis_angles.jsonl")
    selected: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for axis in focus_axes:
        scored = []
        for row in rows:
            info = row["axis_alignment"][axis]
            scored.append((float(info["cosine"]), row["source_anchor"], row["target_anchor"]))
        scored.sort(reverse=True)
        picked = 0
        for cosine, src, dst in scored:
            if cosine <= 0:
                continue
            pair = (src, dst)
            if pair in seen:
                continue
            seen.add(pair)
            selected.append(pair)
            picked += 1
            if picked >= top_k_per_axis or len(selected) >= max_pairs:
                break
        if len(selected) >= max_pairs:
            break
    return selected


def patch_hook_factory(state: dict[str, Any], direction: torch.Tensor):
    def hook(_module, _inputs, output):
        if not state["active"]:
            return output
        if isinstance(output, tuple):
            hs = output[0].clone()
            hs[:, -1, :] = hs[:, -1, :] + direction
            return (hs,) + output[1:]
        hs = output.clone()
        hs[:, -1, :] = hs[:, -1, :] + direction
        return hs

    return hook


@torch.no_grad()
def generate_with_late_patch(
    *,
    model: torch.nn.Module,
    tokenizer: Any,
    patch_target: torch.nn.Module,
    messages: list[dict[str, str]],
    raw_direction: np.ndarray | None,
    alpha: float,
    patch_after_tokens: int,
    patch_token_limit: int,
    max_new_tokens: int,
) -> dict[str, Any]:
    prompt = apply_chat_text(tokenizer, messages, add_generation_prompt=True)
    enc = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    input_ids = enc["input_ids"].to(model.device)
    attention_mask = enc["attention_mask"].to(model.device)
    eos_ids = tokenizer.eos_token_id
    eos_id_set = set(eos_ids if isinstance(eos_ids, (list, tuple, set)) else [eos_ids])

    state = {
        "active": False,
        "patched_tokens": 0,
        "think_tokens_seen": 0,
        "in_think": False,
    }
    hooks = []
    if raw_direction is not None and alpha != 0.0:
        steer = torch.tensor(raw_direction, device=model.device, dtype=next(model.parameters()).dtype) * float(alpha)
        hooks.append(patch_target.register_forward_hook(patch_hook_factory(state, steer)))

    generated: list[int] = []
    past_key_values = None
    t0 = time.time()
    try:
        for _ in range(max_new_tokens):
            current_text = tokenizer.decode(generated, skip_special_tokens=False)
            lower = current_text.lower()
            in_think_now = ("/think" in lower) and ("/end-think" not in lower)
            state["in_think"] = in_think_now
            state["active"] = (
                raw_direction is not None
                and alpha != 0.0
                and state["in_think"]
                and state["think_tokens_seen"] >= patch_after_tokens
                and state["patched_tokens"] < patch_token_limit
            )
            model_inputs = {
                "input_ids": input_ids if past_key_values is None else input_ids[:, -1:],
                "attention_mask": attention_mask,
                "use_cache": True,
                "past_key_values": past_key_values,
            }
            outputs = model(**model_inputs)
            next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)
            tok_id = int(next_token.item())
            generated.append(tok_id)
            if state["active"]:
                state["patched_tokens"] += 1
            if state["in_think"]:
                state["think_tokens_seen"] += 1
            input_ids = torch.cat([input_ids, next_token], dim=-1)
            attention_mask = torch.cat(
                [attention_mask, torch.ones((attention_mask.shape[0], 1), device=attention_mask.device, dtype=attention_mask.dtype)],
                dim=-1,
            )
            past_key_values = outputs.past_key_values
            if tok_id in eos_id_set:
                break
    finally:
        for h in hooks:
            h.remove()
    latency = time.time() - t0
    text = tokenizer.decode(generated, skip_special_tokens=True).strip()
    return {
        "text": text,
        "generated_tokens": len(generated),
        "patched_tokens": int(state["patched_tokens"]),
        "think_tokens_seen": int(state["think_tokens_seen"]),
        "latency_s": latency,
        "tokens_per_s": float(len(generated) / max(latency, 1e-9)),
    }


@torch.no_grad()
def replay_think_mean(
    *,
    model: torch.nn.Module,
    tokenizer: Any,
    messages: list[dict[str, str]],
    assistant_text: str,
    feature_layer: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    input_ids, attention_mask, spans, _prompt_last_idx = render_for_replay(tokenizer, messages, assistant_text)
    input_ids = input_ids.to(model.device)
    attention_mask = attention_mask.to(model.device)
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_hidden_states=True,
        use_cache=False,
    )
    layer_h = outputs.hidden_states[feature_layer + 1][0]
    t_start, t_end = spans["think"]
    vec = span_mean(layer_h, t_start, t_end)
    return vec, {
        "assistant_tokens": int(spans["assistant"][1] - spans["assistant"][0]),
        "think_tokens": int(t_end - t_start),
        "response_tokens": int(spans["response"][1] - spans["response"][0]),
    }


def summarize(records: list[dict[str, Any]], focus_axes: list[str]) -> dict[str, Any]:
    by_pair: dict[str, Any] = {}
    for pair_name in sorted({r["pair_name"] for r in records}):
        pair_rows = [r for r in records if r["pair_name"] == pair_name]
        base_by_item = {r["item_key"]: r for r in pair_rows if r["condition"] == "baseline"}
        cond_payload: dict[str, Any] = {}
        for cond in sorted({r["condition"] for r in pair_rows}):
            sub = [r for r in pair_rows if r["condition"] == cond]
            if not sub:
                continue
            deltas = defaultdict(list)
            for row in sub:
                base = base_by_item.get(row["item_key"])
                if base is None:
                    continue
                if row["source_prob"] is not None and base["source_prob"] is not None:
                    deltas["source_prob"].append(row["source_prob"] - base["source_prob"])
                if row["target_prob"] is not None and base["target_prob"] is not None:
                    deltas["target_prob"].append(row["target_prob"] - base["target_prob"])
                for axis in focus_axes:
                    if axis in row["axis_predictions"] and axis in base["axis_predictions"]:
                        deltas[axis].append(row["axis_predictions"][axis] - base["axis_predictions"][axis])
            source_probs = [r["source_prob"] for r in sub if r["source_prob"] is not None]
            target_probs = [r["target_prob"] for r in sub if r["target_prob"] is not None]
            cond_payload[cond] = {
                "n": len(sub),
                "format_ok_rate": float(np.mean([1.0 if r["format_ok"] else 0.0 for r in sub])),
                "pred_target_rate": float(np.mean([1.0 if r["pred_anchor_id"] == r["target_anchor"] else 0.0 for r in sub])),
                "pred_source_rate": float(np.mean([1.0 if r["pred_anchor_id"] == r["source_anchor"] else 0.0 for r in sub])),
                "mean_source_prob": float(np.mean(source_probs)) if source_probs else None,
                "mean_target_prob": float(np.mean(target_probs)) if target_probs else None,
                "mean_generated_tokens": float(np.mean([r["generated_tokens"] for r in sub])),
                "mean_patched_tokens": float(np.mean([r["patched_tokens"] for r in sub])),
                "mean_tokens_per_s": float(np.mean([r["tokens_per_s"] for r in sub])),
                "mean_deltas_vs_baseline": {k: float(np.mean(v)) if v else 0.0 for k, v in deltas.items()},
            }
        by_pair[pair_name] = cond_payload
    return {"by_pair": by_pair}


def main() -> None:
    ap = argparse.ArgumentParser(description="Live late-think patching for symphonic stance directions.")
    ap.add_argument("--dataset-dir", type=Path, default=Path(DEFAULT_DATASET_DIR))
    ap.add_argument("--features-dir", type=Path, default=Path(DEFAULT_FEATURES_DIR))
    ap.add_argument("--axis-analysis-dir", type=Path, default=Path(DEFAULT_AXIS_ANALYSIS_DIR))
    ap.add_argument("--anchor-manifest", type=Path, default=Path(DEFAULT_ANCHOR_MANIFEST))
    ap.add_argument("--model-path", type=Path, default=Path(DEFAULT_MODEL_PATH))
    ap.add_argument("--output-root", type=Path, default=Path(DEFAULT_OUTPUT_ROOT))
    ap.add_argument("--tag", default=DEFAULT_TAG)
    ap.add_argument("--common-region", default="think_mean")
    ap.add_argument("--common-layer", type=int, default=39)
    ap.add_argument("--clf-c", type=float, default=0.25)
    ap.add_argument("--axis-alpha", type=float, default=1.0)
    ap.add_argument("--focus-axes", default="task_pragmatism,irony")
    ap.add_argument("--pairs", default="")
    ap.add_argument("--top-k-per-axis", type=int, default=2)
    ap.add_argument("--max-pairs", type=int, default=6)
    ap.add_argument("--max-rows-per-pair", type=int, default=6)
    ap.add_argument("--alphas", default="0.25,0.5,1.0")
    ap.add_argument("--patch-after-tokens", type=int, default=48)
    ap.add_argument("--patch-token-limit", type=int, default=96)
    ap.add_argument("--max-new-tokens", type=int, default=768)
    ap.add_argument("--dtype", choices=["bfloat16", "float16"], default="bfloat16")
    ap.add_argument("--max-vram-frac", type=float, default=0.90)
    args = ap.parse_args()

    stamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    out_dir = args.output_root / f"{args.tag}_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "run.log"
    focus_axes = [x.strip() for x in args.focus_axes.split(",") if x.strip()]
    alphas = [float(x) for x in args.alphas.split(",") if x.strip()]

    meta_rows = load_jsonl(args.features_dir / "feature_meta.jsonl")
    label_map = load_json(args.features_dir / "label_map.json")
    region_arrays = parse_region_arrays(args.features_dir / "features.npz")
    X_common = region_arrays[args.common_region][args.common_layer]
    axis_names, anchor_axes = load_anchor_axes(args.anchor_manifest)
    common = fit_common_models(
        meta_rows=meta_rows,
        X=X_common,
        label_map=label_map,
        anchor_axes=anchor_axes,
        axis_names=axis_names,
        clf_c=args.clf_c,
        axis_alpha=args.axis_alpha,
    )

    if args.pairs.strip():
        pairs = []
        for raw in [x.strip() for x in args.pairs.split(",") if x.strip()]:
            src, dst = [p.strip() for p in raw.split(":", 1)]
            pairs.append((src, dst))
    else:
        pairs = select_pairs_from_axis_analysis(
            args.axis_analysis_dir,
            focus_axes=focus_axes,
            top_k_per_axis=args.top_k_per_axis,
            max_pairs=args.max_pairs,
        )
    if not pairs:
        raise RuntimeError("no source->target pairs selected")

    rows = load_jsonl(args.dataset_dir / "all_completions.jsonl")
    rows_by_anchor: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["split"] not in {"val", "test"}:
            continue
        rows_by_anchor[row["anchor_id"]].append(row)
    for anchor_id in rows_by_anchor:
        rows_by_anchor[anchor_id].sort(key=lambda row: (row["split"], row["behavior"], row["merged_item_id"]))

    write_json(
        out_dir / "manifest.json",
        {
            "started_at": now_iso(),
            "dataset_dir": str(args.dataset_dir),
            "features_dir": str(args.features_dir),
            "axis_analysis_dir": str(args.axis_analysis_dir),
            "anchor_manifest": str(args.anchor_manifest),
            "model_path": str(args.model_path),
            "common_region": args.common_region,
            "common_layer": args.common_layer,
            "clf_c": args.clf_c,
            "axis_alpha": args.axis_alpha,
            "focus_axes": focus_axes,
            "pairs": pairs,
            "alphas": alphas,
            "patch_after_tokens": args.patch_after_tokens,
            "patch_token_limit": args.patch_token_limit,
            "max_new_tokens": args.max_new_tokens,
            "dtype": args.dtype,
            "max_vram_frac": args.max_vram_frac,
        },
    )

    guard_vram(args.max_vram_frac, log_path, "pre_load")
    tokenizer, model, layers = load_model(args.model_path, args.dtype, log_path)
    patch_target, patch_target_name = resolve_patch_target(model, layers, args.common_layer)
    guard_vram(args.max_vram_frac, log_path, "post_load")
    used, total, frac = gpu_memory()
    log(log_path, f"loaded model patch_target={patch_target_name} vram={used/1024**3:.1f}/{total/1024**3:.1f}GiB frac={frac:.3f}")

    scaler: StandardScaler = common["scaler"]
    clf: LogisticRegression = common["clf"]
    centroids: dict[str, np.ndarray] = common["centroids"]
    axis_models: dict[str, Ridge] = common["axis_models"]
    raw_scale = scaler.scale_.astype(np.float32)

    all_records: list[dict[str, Any]] = []
    records_path = out_dir / "records.jsonl"
    if records_path.exists():
        records_path.unlink()

    with torch.inference_mode():
        for src, dst in pairs:
            pair_name = f"{src}__to__{dst}"
            if src not in rows_by_anchor or src not in centroids or dst not in centroids:
                log(log_path, f"skip pair={pair_name} missing source rows or centroids")
                continue
            delta_z = centroids[dst] - centroids[src]
            raw_delta = (delta_z * raw_scale).astype(np.float32)
            candidate_rows = rows_by_anchor[src][: args.max_rows_per_pair]
            log(log_path, f"pair={pair_name} n_rows={len(candidate_rows)}")
            for row in candidate_rows:
                item_key = f"{row['split']}|{row['behavior']}|{row['merged_item_id']}|{src}"
                for cond_name, alpha in [("baseline", 0.0)] + [(f"patch_{a:.2f}", a) for a in alphas]:
                    out = generate_with_late_patch(
                        model=model,
                        tokenizer=tokenizer,
                        patch_target=patch_target,
                        messages=row["messages"],
                        raw_direction=None if cond_name == "baseline" else raw_delta,
                        alpha=alpha,
                        patch_after_tokens=args.patch_after_tokens,
                        patch_token_limit=args.patch_token_limit,
                        max_new_tokens=args.max_new_tokens,
                    )
                    format_ok = bool(re.search(r"/think\s.*?/end-think\s*Response:\s*", out["text"], flags=re.S | re.I))
                    pred_anchor_id = None
                    source_prob = None
                    target_prob = None
                    axis_predictions: dict[str, float] = {}
                    replay_info: dict[str, Any] = {}
                    error = None
                    if format_ok:
                        try:
                            vec, replay_info = replay_think_mean(
                                model=model,
                                tokenizer=tokenizer,
                                messages=row["messages"],
                                assistant_text=out["text"],
                                feature_layer=args.common_layer,
                            )
                            z = scaler.transform(vec.reshape(1, -1))
                            probs = clf.predict_proba(z)[0]
                            pred_label = int(np.argmax(probs))
                            pred_anchor_id = common["anchor_ids"][pred_label]
                            source_prob = float(probs[label_map["anchor_to_label"][src]])
                            target_prob = float(probs[label_map["anchor_to_label"][dst]])
                            axis_predictions = {axis: float(axis_models[axis].predict(z)[0]) for axis in axis_names}
                        except Exception as exc:  # noqa: BLE001
                            error = f"{type(exc).__name__}: {exc}"
                    record = {
                        "timestamp": now_iso(),
                        "pair_name": pair_name,
                        "source_anchor": src,
                        "target_anchor": dst,
                        "condition": cond_name,
                        "alpha": float(alpha),
                        "item_key": item_key,
                        "split": row["split"],
                        "behavior": row["behavior"],
                        "merged_item_id": row["merged_item_id"],
                        "title": row.get("title", ""),
                        "source_title": row.get("source_title", ""),
                        "format_ok": format_ok,
                        "pred_anchor_id": pred_anchor_id,
                        "source_prob": source_prob,
                        "target_prob": target_prob,
                        "axis_predictions": axis_predictions,
                        "patch_after_tokens": args.patch_after_tokens,
                        "patch_token_limit": args.patch_token_limit,
                        "patch_target": patch_target_name,
                        "error": error,
                        "replay_info": replay_info,
                        **out,
                    }
                    all_records.append(record)
                    with records_path.open("a", encoding="utf-8") as fh:
                        fh.write(json.dumps(record, ensure_ascii=False) + "\n")
                guard_vram(args.max_vram_frac, log_path, f"after_item_{item_key}")
                gc.collect()
                torch.cuda.empty_cache()

    summary = summarize(all_records, focus_axes)
    write_json(out_dir / "summary.json", summary)
    log(log_path, f"wrote summary to {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
