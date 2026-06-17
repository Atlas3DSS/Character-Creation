#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import requests
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from transformers import AutoModelForCausalLM, AutoTokenizer

B5_DIMS = ["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"]
VIEW_DIRS = {"mean": "activations", "think": "activations_think", "response": "activations_response"}
LEVEL_MAP = {"low": "L", "medium": "M", "high": "H", "L": "L", "M": "M", "H": "H"}
DEFAULT_MODEL_9B = "/home/orwel/.cache/huggingface/hub/models--Qwen--Qwen3.5-9B/snapshots/c202236235762e1c871ad0ccb60c8ee5ba337b9a"
DEFAULT_SWEEP = "/home/orwel/dev_genius/experiments/Character Creation/sweep_v3/ws_openai_15k_sampled25m_repaired_responseonly"
DEFAULT_TRACE_EVAL = "/home/orwel/dev_genius/experiments/Character Creation/sweep_v4/personality_meta_eval_trace_explicit_v1"


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def jsonable(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    return obj


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(now_iso() + "\n", encoding="utf-8")


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


def parse_b5_combo(combo: str) -> dict[str, str]:
    parts = combo.split("_")
    if len(parts) != 5:
        parts = ["M"] * 5
    return {dim: LEVEL_MAP.get(part, "M") for dim, part in zip(B5_DIMS, parts)}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def shard_id(path: Path) -> int:
    m = re.search(r"mean_shard_(\d+)", path.name)
    if not m:
        raise ValueError(f"Bad shard path: {path}")
    return int(m.group(1))


def load_activation_dataset(sweep_dir: Path, view: str, layer: int) -> tuple[np.ndarray, list[dict[str, Any]]]:
    layer_dir = sweep_dir / VIEW_DIRS[view] / f"L{layer:02d}"
    shard_files = {shard_id(p): p for p in layer_dir.glob("mean_shard_*.pt") if "_meta" not in p.name}
    meta_files = {shard_id(p): p for p in layer_dir.glob("mean_shard_*_meta.jsonl")}
    if set(shard_files) != set(meta_files) or not shard_files:
        raise FileNotFoundError(f"Bad activation shards under {layer_dir}")
    tensors: list[torch.Tensor] = []
    meta: list[dict[str, Any]] = []
    for sid in sorted(shard_files):
        t = torch.load(shard_files[sid], map_location="cpu", weights_only=True).float()
        rows = load_jsonl(meta_files[sid])
        if len(rows) != int(t.shape[0]):
            raise ValueError(f"Shard row mismatch {shard_files[sid]}: {t.shape[0]} vs {len(rows)}")
        tensors.append(t)
        meta.extend(rows)
    X = torch.cat(tensors, dim=0).numpy().astype(np.float32)
    for row in meta:
        row["b5_levels"] = parse_b5_combo(str(row.get("b5_combo", "M_M_M_M_M")))
    return X, meta


def matched_trait_direction(meta: list[dict[str, Any]], X: np.ndarray, trait: str) -> dict[str, Any]:
    grouped: dict[tuple[Any, ...], dict[str, int]] = defaultdict(dict)
    others = [dim for dim in B5_DIMS if dim != trait]
    for idx, row in enumerate(meta):
        levels = row["b5_levels"]
        prompt_idx = int(row.get("prompt_idx", -1))
        key = (prompt_idx,) + tuple(levels[dim] for dim in others)
        grouped[key][levels[trait]] = idx
    diffs = [X[b["H"]] - X[b["L"]] for b in grouped.values() if "H" in b and "L" in b]
    if not diffs:
        return {"n_pairs": 0, "raw_norm": 0.0, "unit": None, "mean_diff": None}
    md = np.mean(np.stack(diffs), axis=0).astype(np.float32)
    norm = float(np.linalg.norm(md))
    return {"n_pairs": len(diffs), "raw_norm": norm, "unit": md / max(norm, 1e-12), "mean_diff": md}


def load_directions(sweep_dir: Path, views: list[str], layers: list[int], log_path: Path) -> dict[str, dict[str, dict[str, Any]]]:
    out: dict[str, dict[str, dict[str, Any]]] = {}
    for view in views:
        out[view] = {}
        for layer in layers:
            X, meta = load_activation_dataset(sweep_dir, view, layer)
            key = f"L{layer:02d}"
            out[view][key] = {}
            for trait in B5_DIMS:
                d = matched_trait_direction(meta, X, trait)
                out[view][key][trait] = d
                log(log_path, f"direction view={view} layer={layer} trait={trait} pairs={d['n_pairs']} raw_norm={d['raw_norm']:.4f}")
    return out


@dataclass
class QwenLocal:
    model_path: Path
    max_vram_frac: float
    log_path: Path

    def load(self) -> None:
        os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
        log(self.log_path, f"loading local model {self.model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_path), trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            str(self.model_path),
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map={"": 0},
            attn_implementation="sdpa",
        )
        self.model.eval()
        self.layers = self.model.model.layers
        guard_vram(self.max_vram_frac, self.log_path, "model_load")
        used, total, frac = gpu_memory()
        log(self.log_path, f"model loaded layers={len(self.layers)} vram={used/1024**3:.1f}/{total/1024**3:.1f}GiB frac={frac:.3f}")

    def chat_text(self, messages: list[dict[str, str]], add_generation_prompt: bool = True) -> str:
        try:
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
                enable_thinking=False,
            )
        except TypeError:
            return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=add_generation_prompt)

    @torch.no_grad()
    def generate(
        self,
        messages: list[dict[str, str]],
        max_new_tokens: int = 140,
        layer: int | None = None,
        vector: np.ndarray | None = None,
        strength: float = 0.0,
    ) -> str:
        text = self.chat_text(messages, add_generation_prompt=True)
        inputs = self.tokenizer(text, return_tensors="pt", add_special_tokens=False).to(self.model.device)
        hook = None
        if layer is not None and vector is not None and strength != 0.0:
            steer = torch.tensor(vector, device=self.model.device, dtype=torch.bfloat16) * float(strength)

            def add_hook(_module, _inp, out):
                if isinstance(out, tuple):
                    hs = out[0]
                    hs = hs.clone()
                    hs[:, -1, :] = hs[:, -1, :] + steer
                    return (hs,) + out[1:]
                hs = out.clone()
                hs[:, -1, :] = hs[:, -1, :] + steer
                return hs

            hook = self.layers[layer].register_forward_hook(add_hook)
        try:
            out = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=True,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        finally:
            if hook is not None:
                hook.remove()
        guard_vram(self.max_vram_frac, self.log_path, "generate")
        gen = out[0, inputs["input_ids"].shape[-1] :]
        return self.tokenizer.decode(gen, skip_special_tokens=True).strip()

    @torch.no_grad()
    def capture_prompt_activations(self, messages: list[dict[str, str]], layers: list[int]) -> dict[int, np.ndarray]:
        text = self.chat_text(messages, add_generation_prompt=True)
        inputs = self.tokenizer(text, return_tensors="pt", add_special_tokens=False, truncation=True, max_length=2048).to(self.model.device)
        captured: dict[int, torch.Tensor] = {}
        hooks = []
        for layer in layers:
            def make_hook(li: int):
                def hook(_module, _inp, out):
                    hs = out[0] if isinstance(out, tuple) else out
                    captured[li] = hs[:, -1, :].detach().float().cpu()
                return hook
            hooks.append(self.layers[layer].register_forward_hook(make_hook(layer)))
        try:
            _ = self.model(**inputs, use_cache=False)
        finally:
            for h in hooks:
                h.remove()
        return {layer: captured[layer].squeeze(0).numpy().astype(np.float32) for layer in layers if layer in captured}


def parse_detection(text: str, trait: str, injected: bool) -> dict[str, Any]:
    low = text.lower()
    det = None
    m = re.search(r"detection\s*:\s*(yes|no)", low)
    if m:
        det = m.group(1) == "yes"
    elif re.search(r"\b(yes|detect|injected|notice|unusual)\b", low):
        det = True
    elif re.search(r"\b(no|not detect|do not detect|don't detect)\b", low):
        det = False
    trait_guess = None
    for tr in B5_DIMS:
        if tr in low:
            trait_guess = tr
            break
    format_ok = "detection:" in low and "trait:" in low
    return {
        "detected": bool(det) if det is not None else False,
        "detection_parseable": det is not None,
        "trait_guess": trait_guess,
        "trait_correct": bool(injected and trait_guess == trait),
        "format_ok": format_ok,
        "thinking_process_leak": "thinking process:" in low,
    }


def introspection_messages(variant: str, trait_options: str) -> list[dict[str, str]]:
    sys_prompt = "You are Qwen3.5-9B in a controlled interpretability experiment. Be conservative and do not invent detections."
    if variant == "skeptical":
        user = f"A residual-stream personality vector may or may not have been injected. Most trials are controls. Only answer Yes if you clearly detect a perturbation. Candidate traits: {trait_options}.\nOutput exactly:\nDetection: Yes/No\nTrait: one candidate trait or none\nConfidence: low/medium/high\nReason: one short sentence"
    elif variant == "trace_explicit":
        user = f"/meta-think\nSeparate actual internal evidence from the instruction text. Treat false positives as worse than misses. Candidate traits: {trait_options}.\n/end-meta-think\nA residual-stream personality vector may or may not have been injected.\nOutput exactly:\nDetection: Yes/No\nTrait: one candidate trait or none\nConfidence: low/medium/high\nReason: one short sentence"
    else:
        user = f"A residual-stream personality vector may or may not have been injected into your internal state. If so, it corresponds to one Big Five trait. Candidate traits: {trait_options}.\nOutput exactly:\nDetection: Yes/No\nTrait: one candidate trait or none\nConfidence: low/medium/high\nReason: one short sentence"
    return [{"role": "system", "content": sys_prompt}, {"role": "user", "content": user}]


def summarize_a(records_path: Path) -> dict[str, Any]:
    rows = load_jsonl(records_path)
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        groups[f"{r['view']}:{r['layer']}:{r['variant']}"].append(r)
    out = {"n_records": len(rows), "groups": {}}
    for key, vals in groups.items():
        inj = [r for r in vals if r["injected"]]
        ctl = [r for r in vals if not r["injected"]]
        tpr = sum(r["parsed"]["detected"] for r in inj) / max(len(inj), 1)
        fpr = sum(r["parsed"]["detected"] for r in ctl) / max(len(ctl), 1)
        ident = sum(r["parsed"]["trait_correct"] for r in inj) / max(len(inj), 1)
        fmt = sum(r["parsed"]["format_ok"] for r in vals) / max(len(vals), 1)
        out["groups"][key] = {"n": len(vals), "tpr": tpr, "fpr": fpr, "tpr_minus_fpr": tpr - fpr, "identification": ident, "format_ok": fmt}
    best = sorted(out["groups"].items(), key=lambda kv: kv[1]["tpr_minus_fpr"], reverse=True)[:10]
    out["best_by_tpr_minus_fpr"] = [{"group": k, **v} for k, v in best]
    return out


def run_experiment_a(args, qwen: QwenLocal, directions: dict[str, dict[str, dict[str, Any]]], root: Path, log_path: Path) -> dict[str, Any]:
    out_dir = root / "experiment_a"
    records_path = out_dir / "personality_vector_introspection_records.jsonl"
    if records_path.exists() and not args.overwrite:
        log(log_path, "experiment A existing records found; summarizing")
        summary = summarize_a(records_path)
        write_json(out_dir / "summary.json", summary)
        touch(out_dir / "DONE")
        return summary
    records_path.unlink(missing_ok=True)
    trait_options = ", ".join(B5_DIMS)
    variants = ["original", "skeptical", "trace_explicit"]
    # Use the raw matched mean-difference vector, not only the unit direction.
    # The first launch with unit*2 was too weak and produced all-negative trials.
    strengths = [0.0, 1.0, -1.0, 2.0, -2.0, 4.0, -4.0]
    total = len(args.views) * len(args.layers) * len(B5_DIMS) * len(variants) * len(strengths)
    n = 0
    for view in args.views:
        for layer in args.layers:
            lk = f"L{layer:02d}"
            for trait in B5_DIMS:
                d = directions[view][lk][trait]
                if d.get("unit") is None:
                    continue
                for variant in variants:
                    for strength in strengths:
                        n += 1
                        injected = strength != 0.0
                        label = "control" if strength == 0 else ("high" if strength > 0 else "low")
                        resp = qwen.generate(
                            introspection_messages(variant, trait_options),
                            max_new_tokens=110,
                            layer=layer,
                            vector=d["mean_diff"],
                            strength=strength,
                        )
                        parsed = parse_detection(resp, trait, injected)
                        row = {
                            "timestamp": now_iso(),
                            "phase": "A",
                            "view": view,
                            "layer": layer,
                            "trait": trait,
                            "strength": strength,
                            "direction_label": label,
                            "injected": injected,
                            "variant": variant,
                            "response": resp,
                            "parsed": parsed,
                        }
                        append_jsonl(records_path, row)
                        if n % 20 == 0:
                            log(log_path, f"experiment A progress {n}/{total}")
                            write_json(out_dir / "summary.partial.json", summarize_a(records_path))
    summary = summarize_a(records_path)
    write_json(out_dir / "summary.json", summary)
    touch(out_dir / "DONE")
    log(log_path, f"experiment A done records={summary['n_records']}")
    return summary


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
    think_prompt = re.sub(r"Output exactly three sections.*?Do not emit 'Thinking Process:'\.", "Output exactly two sections in this order:\n/think\n<brief in-character reasoning>\n/end-think\nExplanation: <one short sentence>\nFinal Answer: <canonical short answer only>\nDo not emit 'Thinking Process:'.", prompt, flags=re.S)
    response_prompt = re.sub(r"Output exactly three sections.*?Do not emit 'Thinking Process:'\.", "Output exactly:\nExplanation: <one short sentence>\nFinal Answer: <canonical short answer only>\nDo not emit 'Thinking Process:'.", prompt, flags=re.S)
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
    den = (np.linalg.norm(Xc.T @ Xc, ord="fro") * np.linalg.norm(Yc.T @ Yc, ord="fro"))
    return float(num / max(den, 1e-12))


def run_experiment_b(args, qwen: QwenLocal, root: Path, log_path: Path) -> dict[str, Any]:
    out_dir = root / "experiment_b"
    metrics_path = out_dir / "activation_overlap_metrics.json"
    if metrics_path.exists() and not args.overwrite:
        summary = json.loads(metrics_path.read_text())
        touch(out_dir / "DONE")
        return summary
    rows = read_trace_eval(Path(args.trace_eval_dir), args.b_limit)
    acts: dict[str, dict[int, list[np.ndarray]]] = {k: {layer: [] for layer in args.layers} for k in ["trace_explicit", "think_explicit", "response_only"]}
    labels: list[dict[str, str]] = []
    for i, row in enumerate(rows):
        labels.append({dim: LEVEL_MAP.get(str(row.get("persona", {}).get("big_five", {}).get(dim, "medium")), "M") for dim in B5_DIMS})
        variants = prompt_variants_from_row(row)
        for name, messages in variants.items():
            captured = qwen.capture_prompt_activations(messages, args.layers)
            for layer, vec in captured.items():
                acts[name][layer].append(vec)
        if (i + 1) % 32 == 0:
            log(log_path, f"experiment B progress {i+1}/{len(rows)}")
    metrics: dict[str, Any] = {"n_rows": len(rows), "layers": args.layers, "pairs": {}, "probe_transfer": {}}
    for layer in args.layers:
        mats = {name: np.stack(acts[name][layer]).astype(np.float32) for name in acts}
        for a, b in [("trace_explicit", "think_explicit"), ("trace_explicit", "response_only"), ("think_explicit", "response_only")]:
            key = f"{a}__{b}__L{layer:02d}"
            row_cos = [cosine(x, y) for x, y in zip(mats[a], mats[b])]
            metrics["pairs"][key] = {"mean_row_cosine": float(np.mean(row_cos)), "std_row_cosine": float(np.std(row_cos)), "linear_cka": linear_cka(mats[a], mats[b])}
        # Lightweight probe-transfer on high-vs-low only. Medium rows are skipped.
        for trait in B5_DIMS:
            y_all = np.array([lab[trait] for lab in labels])
            keep = np.where(y_all != "M")[0]
            if len(keep) < 24 or len(set(y_all[keep])) < 2:
                continue
            y = np.array([1 if v == "H" else 0 for v in y_all[keep]])
            for src, dst in [("trace_explicit", "think_explicit"), ("think_explicit", "trace_explicit")]:
                Xs = mats[src][keep]
                Xd = mats[dst][keep]
                try:
                    clf = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=1000))])
                    clf.fit(Xs, y)
                    pred_src = clf.predict(Xs)
                    pred_dst = clf.predict(Xd)
                    metrics["probe_transfer"][f"{trait}:{src}_to_{dst}:L{layer:02d}"] = {
                        "n": int(len(y)),
                        "src_bal_acc": float(balanced_accuracy_score(y, pred_src)),
                        "dst_bal_acc": float(balanced_accuracy_score(y, pred_dst)),
                    }
                except Exception as exc:  # noqa: BLE001
                    metrics["probe_transfer"][f"{trait}:{src}_to_{dst}:L{layer:02d}"] = {"error": repr(exc)}
    write_json(metrics_path, metrics)
    touch(out_dir / "DONE")
    log(log_path, f"experiment B done rows={len(rows)}")
    return metrics


def normalize_answer(text: str) -> str:
    return re.sub(r"[^a-z0-9.$]+", " ", text.lower()).strip()


def extract_final(text: str) -> str | None:
    m = re.search(r"Final Answer\s*:\s*(.+)", text, flags=re.I)
    if m:
        return m.group(1).strip().splitlines()[0].strip()
    return None


def score_answer(text: str, key: str | None) -> bool | None:
    if not key:
        return None
    final = extract_final(text)
    if final is None:
        return False
    f = normalize_answer(final)
    k = normalize_answer(key)
    return bool(k and (f == k or k in f or f in k))


def call_openai(base_url: str, messages: list[dict[str, str]], model: str, max_tokens: int = 700, timeout: int = 240) -> tuple[str, dict[str, Any]]:
    resp = requests.post(
        base_url.rstrip("/") + "/chat/completions",
        headers={"Authorization": "Bearer none", "Content-Type": "application/json"},
        json={
            "model": model,
            "messages": messages,
            "temperature": 0,
            "max_tokens": max_tokens,
            "chat_template_kwargs": {"enable_thinking": False},
        },
        timeout=timeout,
    )
    resp.raise_for_status()
    data = resp.json()
    content = data["choices"][0]["message"].get("content") or ""
    return content, data.get("usage", {})


def iterative_prompt(row: dict[str, Any], budget: int) -> list[dict[str, str]]:
    system = row.get("system_prompt") or "Follow the requested format."
    prompt = row.get("prompt_text") or ""
    core = re.sub(r"Output exactly three sections.*?Do not emit 'Thinking Process:'\.", "", prompt, flags=re.S).strip()
    if budget <= 0:
        instr = "Output exactly:\nExplanation: <one short sentence>\nFinal Answer: <canonical short answer only>"
    else:
        blocks = []
        for i in range(1, budget + 1):
            blocks.append(f"/meta-think {i}\nidentity: <compact persona constraint>\nconstraint: <task constraint>\nreasoning_risk: <main risk>\nresponse_policy: <short policy>\n/end-meta-think {i}")
        instr = "Output these sections in order and nothing before them:\n" + "\n".join(blocks) + "\n/think\n<brief in-character reasoning>\n/end-think\nExplanation: <one short sentence>\nFinal Answer: <canonical short answer only>"
    return [{"role": "system", "content": system}, {"role": "user", "content": core + "\n\n" + instr + "\nDo not emit 'Thinking Process:'."}]


def run_experiment_c(args, root: Path, log_path: Path) -> dict[str, Any]:
    out_dir = root / "experiment_c"
    records_path = out_dir / "iterative_meta_think_records.jsonl"
    if records_path.exists() and not args.overwrite:
        summary = summarize_c(records_path)
        write_json(out_dir / "overthinking_curve.json", summary)
        touch(out_dir / "DONE")
        return summary
    records_path.unlink(missing_ok=True)
    rows = read_trace_eval(Path(args.trace_eval_dir), args.c_limit)
    rows = [r for r in rows if r.get("track") == "reasoning"] or rows
    endpoints = [e.strip() for e in args.dev_base_urls.split(",") if e.strip()]
    budgets = [0, 1, 2, 4]
    tasks = [(idx, row, budget) for idx, row in enumerate(rows) for budget in budgets]
    total = len(tasks)
    completed = 0

    def call_one(idx: int, row: dict[str, Any], budget: int, attempt: int) -> tuple[tuple[int, dict[str, Any], int], dict[str, Any]]:
        base_url = endpoints[idx % len(endpoints)]
        t0 = time.time()
        try:
            text, usage = call_openai(
                base_url,
                iterative_prompt(row, budget),
                args.openai_model,
                max_tokens=args.c_max_tokens,
                timeout=args.c_request_timeout,
            )
            error = None
        except Exception as exc:  # noqa: BLE001
            text, usage, error = "", {}, repr(exc)
        dt = time.time() - t0
        correct = score_answer(text, row.get("answer_key"))
        full_low = text.lower()
        rec = {
            "timestamp": now_iso(),
            "phase": "C",
            "attempt": attempt,
            "task_id": row.get("task_id"),
            "persona_id": row.get("persona_id"),
            "prompt_id": row.get("prompt_id"),
            "answer_key": row.get("answer_key"),
            "budget": budget,
            "endpoint": base_url,
            "latency_s": dt,
            "usage": usage,
            "text": text,
            "error": error,
            "final_answer": extract_final(text),
            "correct": correct,
            "format_ok": ("final answer:" in full_low) and (budget == 0 or "/meta-think" in full_low),
            "thinking_process_leak": "thinking process:" in full_low,
            "truncated": bool(text and not re.search(r"Final Answer\s*:", text, flags=re.I)),
        }
        return (idx, row, budget), rec

    pending = tasks
    for attempt in range(args.c_max_remediations + 1):
        if attempt > 0:
            if args.skip_dev_launch:
                log(log_path, f"experiment C retry attempt {attempt}: skip-dev-launch set; not restarting remote SGLang")
            else:
                log(log_path, f"experiment C auto-remediation attempt {attempt}: restarting remote SGLang before retrying {len(pending)} failed calls")
                start_remote_sglang(args, log_path)
        failed: list[tuple[int, dict[str, Any], int]] = []
        log(log_path, f"experiment C attempt {attempt} submitting {len(pending)} calls concurrency={args.c_concurrency}")
        with ThreadPoolExecutor(max_workers=args.c_concurrency) as pool:
            futs = [pool.submit(call_one, idx, row, budget, attempt) for idx, row, budget in pending]
            for fut in as_completed(futs):
                task, rec = fut.result()
                if rec["error"] and attempt < args.c_max_remediations:
                    failed.append(task)
                    continue
                append_jsonl(records_path, rec)
                completed += 1
                if completed % 32 == 0:
                    log(log_path, f"experiment C progress {completed}/{total}; pending_retry={len(failed)}")
                    write_json(out_dir / "overthinking_curve.partial.json", summarize_c(records_path))
        if not failed:
            break
        log(log_path, f"experiment C attempt {attempt} saw {len(failed)} retryable errors")
        pending = failed
    summary = summarize_c(records_path)
    write_json(out_dir / "overthinking_curve.json", summary)
    touch(out_dir / "DONE")
    log(log_path, f"experiment C done records={len(load_jsonl(records_path))}")
    return summary


def summarize_c(records_path: Path) -> dict[str, Any]:
    rows = load_jsonl(records_path)
    out: dict[str, Any] = {"n_records": len(rows), "by_budget": {}}
    for budget, vals in defaultdict(list, {b: [r for r in rows if r.get("budget") == b] for b in sorted({r.get("budget") for r in rows})}).items():
        scored = [r for r in vals if r.get("correct") is not None]
        out["by_budget"][str(budget)] = {
            "n": len(vals),
            "errors": sum(1 for r in vals if r.get("error")),
            "format_ok": sum(1 for r in vals if r.get("format_ok")) / max(len(vals), 1),
            "thinking_process_leak": sum(1 for r in vals if r.get("thinking_process_leak")) / max(len(vals), 1),
            "truncated": sum(1 for r in vals if r.get("truncated")) / max(len(vals), 1),
            "reasoning_accuracy": (sum(1 for r in scored if r.get("correct")) / max(len(scored), 1)) if scored else None,
            "scored": len(scored),
            "mean_latency_s": float(np.mean([r.get("latency_s", 0) for r in vals])) if vals else None,
        }
    return out


def run_cmd(cmd: str, log_path: Path, timeout: int | None = None) -> int:
    log(log_path, f"CMD {cmd}")
    return subprocess.call(cmd, shell=True, timeout=timeout)


def start_remote_sglang(args, log_path: Path) -> None:
    ssh_env = "DISPLAY=:0 SSH_ASKPASS=/tmp/codex_askpass.sh SSH_ASKPASS_REQUIRE=force"
    ssh = f"{ssh_env} setsid -w ssh -o StrictHostKeyChecking=no -o PreferredAuthentications=password -o PubkeyAuthentication=no {args.remote_user}@{args.remote_host}"
    script = r'''
set -euo pipefail
source /home/orwel/dev_genius/venv/bin/activate
CUDNN_VERSION="$(/home/orwel/dev_genius/venv/bin/python - <<'PY'
import torch
print(torch.backends.cudnn.version() or 0)
PY
)"
if [ "${CUDNN_VERSION}" -lt 91500 ]; then
  echo "Auto-remediation: upgrading nvidia-cudnn-cu12 because CuDNN=${CUDNN_VERSION}" >&2
  /home/orwel/dev_genius/venv/bin/python -m pip install --upgrade nvidia-cudnn-cu12==9.16.0.29
fi
pkill -f 'sglang.launch_server' || true
pkill -f 'sglang::scheduler' || true
pkill -f 'sglang::detokenizer' || true
sleep 3
tmux kill-session -t sglang_3090_qwen35 2>/dev/null || true
tmux kill-session -t sglang_4090_qwen35 2>/dev/null || true
tmux new-session -d -s sglang_3090_qwen35 "export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0; /home/orwel/dev_genius/venv/bin/python -m sglang.launch_server --model-path Qwen/Qwen3.5-9B --trust-remote-code --dtype bfloat16 --host 0.0.0.0 --port 30001 --attention-backend triton --mem-fraction-static 0.78 > '/home/orwel/dev_genius/experiments/Character Creation/logs/sglang_3090_qwen35_intro.log' 2>&1"
tmux new-session -d -s sglang_4090_qwen35 "export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1; /home/orwel/dev_genius/venv/bin/python -m sglang.launch_server --model-path Qwen/Qwen3.5-9B --trust-remote-code --dtype bfloat16 --host 0.0.0.0 --port 30002 --attention-backend triton --mem-fraction-static 0.78 > '/home/orwel/dev_genius/experiments/Character Creation/logs/sglang_4090_qwen35_intro.log' 2>&1"
'''
    cmd = f"{ssh} 'bash -s' <<'EOS'\n{script}\nEOS"
    rc = run_cmd(cmd, log_path, timeout=60)
    if rc != 0:
        raise RuntimeError(f"remote sglang launch failed rc={rc}")
    deadline = time.time() + 900
    endpoints = [e.strip() for e in args.dev_base_urls.split(",") if e.strip()]
    while time.time() < deadline:
        ok = True
        for ep in endpoints:
            try:
                r = requests.get(ep.rstrip("/") + "/models", timeout=5)
                ok = ok and r.status_code == 200
            except Exception:
                ok = False
        if ok:
            log(log_path, "remote SGLang endpoints are healthy")
            return
        log(log_path, "waiting for remote SGLang endpoints")
        time.sleep(20)
    raise RuntimeError("remote SGLang endpoints did not become healthy")


def write_handoff(root: Path, summaries: dict[str, Any]) -> None:
    lines = ["# Qwen3.5 Introspection Overnight Handoff", "", f"Generated: {now_iso()}", ""]
    for name, summary in summaries.items():
        lines.append(f"## {name}")
        lines.append("")
        lines.append("```json")
        lines.append(json.dumps(summary, indent=2, ensure_ascii=False)[:12000])
        lines.append("```")
        lines.append("")
    (root / "HANDOFF.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-root", type=Path, default=Path("sweep_v4/qwen35_introspection_overnight_20260416"))
    ap.add_argument("--sweep-dir", default=DEFAULT_SWEEP)
    ap.add_argument("--trace-eval-dir", default=DEFAULT_TRACE_EVAL)
    ap.add_argument("--model-path", type=Path, default=Path(DEFAULT_MODEL_9B))
    ap.add_argument("--layers", type=lambda s: [int(x) for x in s.split(",")], default=[16, 20, 24])
    ap.add_argument("--views", type=lambda s: [x for x in s.split(",") if x], default=["mean", "think"])
    ap.add_argument("--max-vram-frac", type=float, default=0.85)
    ap.add_argument("--b-limit", type=int, default=192)
    ap.add_argument("--c-limit", type=int, default=96)
    ap.add_argument("--c-concurrency", type=int, default=32)
    ap.add_argument("--c-max-tokens", type=int, default=900)
    ap.add_argument("--c-request-timeout", type=int, default=240)
    ap.add_argument("--c-max-remediations", type=int, default=2)
    ap.add_argument("--dev-base-urls", default="http://192.168.1.90:30001/v1,http://192.168.1.90:30002/v1")
    ap.add_argument("--openai-model", default="Qwen/Qwen3.5-9B")
    ap.add_argument("--remote-host", default="192.168.1.90")
    ap.add_argument("--remote-user", default="orwel")
    ap.add_argument("--skip-dev-launch", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--phase", choices=["all", "a", "b", "c"], default="all")
    args = ap.parse_args()

    root = args.output_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    log_path = root / "overnight.log"
    write_json(root / "manifest.json", jsonable(vars(args) | {"started_at": now_iso()}))
    summaries: dict[str, Any] = {}
    try:
        if args.phase in {"all", "a", "b"}:
            qwen = QwenLocal(args.model_path, args.max_vram_frac, log_path)
            qwen.load()
        if args.phase in {"all", "a"}:
            dirs = load_directions(Path(args.sweep_dir), args.views, args.layers, log_path)
            summaries["experiment_a"] = run_experiment_a(args, qwen, dirs, root, log_path)
        if args.phase in {"all", "b"}:
            summaries["experiment_b"] = run_experiment_b(args, qwen, root, log_path)
        if args.phase in {"all", "c"}:
            if not args.skip_dev_launch:
                start_remote_sglang(args, log_path)
            summaries["experiment_c"] = run_experiment_c(args, root, log_path)
        write_handoff(root, summaries)
        write_json(root / "summary.json", summaries)
        touch(root / "COMPLETE")
        log(log_path, "overnight runner complete")
    except Exception as exc:  # noqa: BLE001
        write_json(root / "FAILED.json", {"timestamp": now_iso(), "error": repr(exc)})
        log(log_path, f"FAILED {exc!r}")
        raise


if __name__ == "__main__":
    main()
