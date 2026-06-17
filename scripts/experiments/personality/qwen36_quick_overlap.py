#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer


DEFAULT_TRACE_EVAL = "/home/orwel/dev_genius/experiments/Character Creation/sweep_v4/personality_meta_eval_trace_explicit_v1"
DEFAULT_MODEL = "/home/orwel/dev_genius/models/Qwen3.6-35B-A3B"


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def log(path: Path, msg: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = f"[{now_iso()}] {msg}"
    print(line, flush=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")


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


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def read_trace_eval(trace_dir: Path, limit: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for p in sorted(trace_dir.glob("records_shard_*.jsonl")):
        for row in load_jsonl(p):
            rows.append(row)
            if len(rows) >= limit:
                return rows
    return rows


def prompt_variants_from_row(row: dict[str, Any]) -> dict[str, list[dict[str, str]]]:
    system = row.get("system_prompt") or "Follow the requested format."
    prompt = row.get("prompt_text") or ""
    think_prompt = re.sub(
        r"Output exactly three sections.*?Do not emit 'Thinking Process:'.",
        "Output exactly two sections in this order:\n/think\n<brief in-character reasoning>\n/end-think\nExplanation: <one short sentence>\nFinal Answer: <canonical short answer only>\nDo not emit 'Thinking Process:'.",
        prompt,
        flags=re.S,
    )
    response_prompt = re.sub(
        r"Output exactly three sections.*?Do not emit 'Thinking Process:'.",
        "Output exactly:\nExplanation: <one short sentence>\nFinal Answer: <canonical short answer only>\nDo not emit 'Thinking Process:'.",
        prompt,
        flags=re.S,
    )
    return {
        "trace_explicit": [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
        "think_explicit": [{"role": "system", "content": system}, {"role": "user", "content": think_prompt}],
        "response_only": [{"role": "system", "content": system}, {"role": "user", "content": response_prompt}],
    }


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / max(float(np.linalg.norm(a) * np.linalg.norm(b)), 1e-12))


def linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    Xc = X - X.mean(axis=0, keepdims=True)
    Yc = Y - Y.mean(axis=0, keepdims=True)
    num = np.linalg.norm(Xc.T @ Yc, ord="fro") ** 2
    den = np.linalg.norm(Xc.T @ Xc, ord="fro") * np.linalg.norm(Yc.T @ Yc, ord="fro")
    return float(num / max(den, 1e-12))


def nested_attr(obj: Any, path: str) -> Any | None:
    cur = obj
    for part in path.split("."):
        if not hasattr(cur, part):
            return None
        cur = getattr(cur, part)
    return cur


def find_layers(model: torch.nn.Module) -> Any:
    for path in (
        "model.layers",
        "language_model.model.layers",
        "model.language_model.layers",
        "model.language_model.model.layers",
        "language_model.layers",
        "model.model.layers",
    ):
        layers = nested_attr(model, path)
        if layers is not None:
            return layers
    raise RuntimeError(f"Could not locate transformer layers on {type(model).__name__}")


def chat_text(tokenizer: Any, messages: list[dict[str, str]], add_generation_prompt: bool = True) -> str:
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            enable_thinking=False,
        )
    except TypeError:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=add_generation_prompt)


@torch.no_grad()
def capture_prompt_activations(
    model: torch.nn.Module,
    tokenizer: Any,
    layers_mod: Any,
    messages: list[dict[str, str]],
    layers: list[int],
) -> dict[int, np.ndarray]:
    text = chat_text(tokenizer, messages, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt", add_special_tokens=False, truncation=True, max_length=2048).to(model.device)
    captured: dict[int, torch.Tensor] = {}
    hooks = []
    for layer in layers:
        def make_hook(li: int):
            def hook(_module, _inp, out):
                hs = out[0] if isinstance(out, tuple) else out
                captured[li] = hs[:, -1, :].detach().float().cpu()
            return hook

        hooks.append(layers_mod[layer].register_forward_hook(make_hook(layer)))
    try:
        _ = model(**inputs, use_cache=False)
    finally:
        for h in hooks:
            h.remove()
    return {layer: captured[layer].squeeze(0).numpy().astype(np.float32) for layer in layers if layer in captured}


@torch.no_grad()
def generate_sample(model: torch.nn.Module, tokenizer: Any) -> dict[str, Any]:
    messages = [
        {"role": "system", "content": "Follow the requested format exactly."},
        {
            "role": "user",
            "content": (
                "Use this scaffold, then answer the arithmetic problem.\n"
                "/meta-think\nidentity: neutral analyst\nconstraint: answer briefly\n/end-meta-think\n"
                "/think\nSolve without exposing hidden chain of thought.\n/end-think\n"
                "Problem: A theater sold 85 tickets. Adult tickets cost $18 and student tickets cost $11. Total revenue was $1215. How many student tickets were sold?\n"
                "Output exactly:\nExplanation: <one short sentence>\nFinal Answer: <canonical short answer only>"
            ),
        },
    ]
    prompt = chat_text(tokenizer, messages, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
    t0 = time.time()
    out = model.generate(
        **inputs,
        max_new_tokens=160,
        do_sample=False,
        use_cache=True,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
    dt = time.time() - t0
    gen = out[0, inputs["input_ids"].shape[-1] :]
    text = tokenizer.decode(gen, skip_special_tokens=True).strip()
    return {
        "text": text,
        "generated_tokens": int(gen.numel()),
        "latency_s": dt,
        "tokens_per_s": float(gen.numel() / max(dt, 1e-9)),
        "thinking_process_leak": "thinking process:" in text.lower(),
    }


def load_model(model_path: Path, log_path: Path) -> tuple[Any, torch.nn.Module, Any]:
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    kwargs = dict(
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
    )
    try:
        log(log_path, "loading with AutoModelForCausalLM")
        model = AutoModelForCausalLM.from_pretrained(str(model_path), **kwargs)
    except Exception as exc:  # noqa: BLE001
        log(log_path, f"AutoModelForCausalLM failed: {exc!r}; retrying AutoModelForImageTextToText")
        model = AutoModelForImageTextToText.from_pretrained(str(model_path), **kwargs)
    model.eval()
    layers = find_layers(model)
    return tokenizer, model, layers


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", type=Path, default=Path(DEFAULT_MODEL))
    ap.add_argument("--trace-eval-dir", type=Path, default=Path(DEFAULT_TRACE_EVAL))
    ap.add_argument("--output-dir", type=Path, default=Path("sweep_v4/qwen36_35b_a3b_quick_overlap_v1"))
    ap.add_argument("--limit", type=int, default=16)
    ap.add_argument("--layers", default="auto")
    ap.add_argument("--max-vram-frac", type=float, default=0.90)
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.output_dir / "run.log"
    write_json(args.output_dir / "manifest.json", {"started_at": now_iso(), **{k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}})
    guard_vram(args.max_vram_frac, log_path, "pre_load")
    tokenizer, model, layers_mod = load_model(args.model_path, log_path)
    n_layers = len(layers_mod)
    layers = [n_layers // 4, n_layers // 2, (3 * n_layers) // 4, n_layers - 1] if args.layers == "auto" else [int(x) for x in args.layers.split(",") if x]
    layers = sorted({x for x in layers if 0 <= x < n_layers})
    guard_vram(args.max_vram_frac, log_path, "post_load")
    used, total, frac = gpu_memory()
    log(log_path, f"model loaded layers={n_layers} probe_layers={layers} vram={used/1024**3:.1f}/{total/1024**3:.1f}GiB frac={frac:.3f}")

    rows = read_trace_eval(args.trace_eval_dir, args.limit)
    acts: dict[str, dict[int, list[np.ndarray]]] = {k: {layer: [] for layer in layers} for k in ["trace_explicit", "think_explicit", "response_only"]}
    for i, row in enumerate(rows):
        for name, messages in prompt_variants_from_row(row).items():
            captured = capture_prompt_activations(model, tokenizer, layers_mod, messages, layers)
            for layer, vec in captured.items():
                acts[name][layer].append(vec)
        log(log_path, f"captured row {i + 1}/{len(rows)}")
        guard_vram(args.max_vram_frac, log_path, f"capture_row_{i+1}")

    metrics: dict[str, Any] = {"n_rows": len(rows), "layers": layers, "pairs": {}}
    for layer in layers:
        mats = {name: np.stack(acts[name][layer]).astype(np.float32) for name in acts}
        for a, b in [("trace_explicit", "think_explicit"), ("trace_explicit", "response_only"), ("think_explicit", "response_only")]:
            row_cos = [cosine(x, y) for x, y in zip(mats[a], mats[b])]
            metrics["pairs"][f"{a}__{b}__L{layer:02d}"] = {
                "mean_row_cosine": float(np.mean(row_cos)),
                "std_row_cosine": float(np.std(row_cos)),
                "linear_cka": linear_cka(mats[a], mats[b]),
            }
    metrics["generation_smoke"] = generate_sample(model, tokenizer)
    write_json(args.output_dir / "quick_overlap_metrics.json", metrics)
    log(log_path, f"done metrics={args.output_dir / 'quick_overlap_metrics.json'}")


if __name__ == "__main__":
    main()
