#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import os
import re
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer


DEFAULT_TRACE_EVAL = "/home/orwel/dev_genius/experiments/Character Creation/sweep_v4/personality_meta_eval_trace_explicit_v1"
DEFAULT_GREEDY_RUN = "/home/orwel/dev_genius/experiments/Character Creation/sweep_v4/meta_sham_control_ages_qwen35_20260416"
DEFAULT_MODEL_9B = "/home/orwel/.cache/huggingface/hub/models--Qwen--Qwen3.5-9B/snapshots/c202236235762e1c871ad0ccb60c8ee5ba337b9a"
DEFAULT_MODEL_35B = "/home/orwel/dev_genius/models/Qwen3.6-35B-A3B"
DEFAULT_DIRECTIONS_9B = "/home/orwel/dev_genius/experiments/Character Creation/sweep_v4/meta_sham_contrastive_replay_qwen35_20260416_nonanswer/candidate_directions.npz"
DEFAULT_OUTPUT_ROOT = "/home/orwel/dev_genius/experiments/Character Creation/sweep_v4"

SHAM_META_BLOCK = (
    "/meta-think\n"
    "slot_a: cedar lantern river paper brass window station pocket\n"
    "slot_b: amber thread mirror orchard handle gravel notebook harbor\n"
    "slot_c: velvet compass ladder meadow copper envelope walkway lantern\n"
    "slot_d: marble feather kettle canvas wagon button orchard signal\n"
    "/end-meta-think"
)

GENERIC_PREP_BLOCK = (
    "/prep-think\n"
    "note_a: keep the process orderly and move one step at a time\n"
    "note_b: keep the wording compact and avoid unnecessary detours\n"
    "note_c: preserve internal consistency while moving toward closure\n"
    "note_d: finish with the required explanation and final answer block\n"
    "/end-prep-think"
)


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def log(path: Path, msg: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = f"[{now_iso()}] {msg}"
    print(line, flush=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def read_trace_eval(trace_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for p in sorted(trace_dir.glob("records_shard_*.jsonl")):
        rows.extend(load_jsonl(p))
    return [r for r in rows if r.get("track") == "reasoning"]


def task_key(row: dict[str, Any]) -> str:
    return row.get("task_id") or f"{row.get('persona_id')}::{row.get('prompt_id')}"


def strip_output_contract(prompt: str) -> str:
    return re.sub(r"Output exactly three sections.*?Do not emit 'Thinking Process:'\.", "", prompt, flags=re.S).strip()


def make_messages(row: dict[str, Any], condition: str) -> list[dict[str, str]]:
    system = row.get("system_prompt") or "Follow the requested format."
    core = strip_output_contract(row.get("prompt_text") or "")
    if condition == "think_only":
        instr = (
            "Output exactly these sections in order and nothing before them:\n"
            "/think\n"
            "<brief in-character reasoning>\n"
            "/end-think\n"
            "Explanation: <one short sentence>\n"
            "Final Answer: <canonical short answer only>"
        )
    elif condition == "real_meta":
        instr = (
            "Output exactly these sections in order and nothing before them:\n"
            "/meta-think\n"
            "identity: <compact persona constraint>\n"
            "constraint: <task constraint>\n"
            "reasoning_risk: <main risk>\n"
            "response_policy: <short policy>\n"
            "/end-meta-think\n"
            "/think\n"
            "<brief in-character reasoning>\n"
            "/end-think\n"
            "Explanation: <one short sentence>\n"
            "Final Answer: <canonical short answer only>"
        )
    elif condition == "sham_meta":
        instr = (
            "Output exactly these sections in order and nothing before them.\n"
            "Copy the `/meta-think` block below exactly as written, then continue:\n"
            f"{SHAM_META_BLOCK}\n"
            "/think\n"
            "<brief in-character reasoning>\n"
            "/end-think\n"
            "Explanation: <one short sentence>\n"
            "Final Answer: <canonical short answer only>"
        )
    elif condition == "generic_prep":
        instr = (
            "Output exactly these sections in order and nothing before them.\n"
            "Copy the `/prep-think` block below exactly as written, then continue:\n"
            f"{GENERIC_PREP_BLOCK}\n"
            "/think\n"
            "<brief in-character reasoning>\n"
            "/end-think\n"
            "Explanation: <one short sentence>\n"
            "Final Answer: <canonical short answer only>"
        )
    else:
        raise ValueError(f"unknown condition: {condition}")
    user = core + "\n\n" + instr + "\nDo not emit 'Thinking Process:'."
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def normalize_answer(text: str) -> str:
    return re.sub(r"[^a-z0-9.$]+", " ", text.lower()).strip()


def extract_final(text: str) -> str | None:
    m = re.search(r"Final Answer\s*:\s*(.+)", text, flags=re.I)
    if not m:
        return None
    return m.group(1).strip().splitlines()[0].strip()


def score_answer(text: str, key: str | None) -> bool | None:
    if not key:
        return None
    final = extract_final(text)
    if final is None:
        return False
    f = normalize_answer(final)
    k = normalize_answer(key)
    return bool(k and (f == k or k in f or f in k))


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


def parse_patch_specs(spec: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for raw in [x.strip() for x in spec.split(",") if x.strip()]:
        name, body = (raw.split("=", 1) + [raw])[:2] if "=" in raw else (raw, raw)
        if body == "none":
            rows.append({"name": name, "layers": [], "alpha": 0.0, "token_limit": 0})
            continue
        m = re.fullmatch(r"([0-9+]+)@([0-9.]+)(?::([0-9]+|full))?", body)
        if not m:
            raise ValueError(f"invalid patch spec: {raw}")
        layers = [int(x) for x in m.group(1).split("+") if x]
        alpha = float(m.group(2))
        limit_raw = m.group(3) or "full"
        token_limit = 10**9 if limit_raw == "full" else int(limit_raw)
        rows.append({"name": name, "layers": layers, "alpha": alpha, "token_limit": token_limit})
    return rows


def load_directions(path: Path) -> dict[int, np.ndarray]:
    meta_path = path.with_suffix(".json")
    meta = load_json(meta_path)
    arrays = np.load(path)
    out: dict[int, np.ndarray] = {}
    for row in meta:
        if row["comparison"] != "real_minus_think" or row["region"] != "think_region":
            continue
        layer = int(row["layer"])
        out[layer] = arrays[row["key"]].astype(np.float32)
    if not out:
        raise RuntimeError(f"no real_minus_think think_region directions found in {path}")
    return out


def select_tasks(greedy_dir: Path, trace_rows: dict[str, dict[str, Any]], max_wins: int, max_regressions: int) -> list[dict[str, Any]]:
    summary = load_json(greedy_dir / "summary.json")
    records = load_jsonl(greedy_dir / "records.jsonl")
    records_by_task: dict[str, dict[str, Any]] = defaultdict(dict)
    for row in records:
        if row.get("condition") == "think_only":
            records_by_task[row["task_id"]] = row

    win_task_ids = list(summary["real_meta_unique_fixes_vs_controls"])[:max_wins]
    regression_task_ids = list(summary["paired_vs_think_only"]["real_meta"]["marginal_regressions"])[:max_regressions]
    selected: list[dict[str, Any]] = []
    for group, ids in (("win", win_task_ids), ("regression", regression_task_ids)):
        for task_id in ids:
            row = trace_rows[task_id]
            selected.append(
                {
                    "group": group,
                    "task_id": task_id,
                    "answer_key": row.get("answer_key"),
                    "trace_row": row,
                    "saved_think_only_correct": bool(records_by_task.get(task_id, {}).get("correct")),
                }
            )
    return selected


def load_model(model_path: Path, dtype: str, log_path: Path) -> tuple[Any, torch.nn.Module, Any]:
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    torch_dtype = torch.bfloat16 if dtype == "bfloat16" else torch.float16
    kwargs = dict(
        trust_remote_code=True,
        torch_dtype=torch_dtype,
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


@torch.no_grad()
def generate_with_patch(
    model: torch.nn.Module,
    tokenizer: Any,
    layers_mod: Any,
    messages: list[dict[str, str]],
    layer_to_vec: dict[int, np.ndarray],
    alpha: float,
    max_new_tokens: int,
    token_limit: int,
) -> dict[str, Any]:
    prompt = chat_text(tokenizer, messages, add_generation_prompt=True)
    enc = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    input_ids = enc["input_ids"].to(model.device)
    attention_mask = enc["attention_mask"].to(model.device)
    eos_ids = tokenizer.eos_token_id
    eos_id_set = set(eos_ids if isinstance(eos_ids, (list, tuple, set)) else [eos_ids])

    steer_tensors = {
        layer: torch.tensor(vec, device=model.device, dtype=next(model.parameters()).dtype) * float(alpha)
        for layer, vec in layer_to_vec.items()
    }
    state = {"active": False, "patched_tokens": 0}
    hooks = []
    for layer_idx, steer in steer_tensors.items():
        def make_hook(steer_vec: torch.Tensor):
            def hook(_module, _inp, out):
                if not state["active"]:
                    return out
                if isinstance(out, tuple):
                    hs = out[0].clone()
                    hs[:, -1, :] = hs[:, -1, :] + steer_vec
                    return (hs,) + out[1:]
                hs = out.clone()
                hs[:, -1, :] = hs[:, -1, :] + steer_vec
                return hs
            return hook

        hooks.append(layers_mod[layer_idx].register_forward_hook(make_hook(steer)))

    generated: list[int] = []
    patched_token_flags: list[bool] = []
    past_key_values = None
    t0 = time.time()
    try:
        for _step in range(max_new_tokens):
            current_text = tokenizer.decode(generated, skip_special_tokens=False)
            state["active"] = (
                state["patched_tokens"] < token_limit
                and "/end-think" not in current_text.lower()
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
            patched_token_flags.append(bool(state["active"]))
            if state["active"]:
                state["patched_tokens"] += 1
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
        "patched_tokens": int(sum(1 for x in patched_token_flags if x)),
        "latency_s": latency,
        "tokens_per_s": float(len(generated) / max(latency, 1e-9)),
    }


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    by_condition: dict[str, Any] = {}
    for cond in sorted({r["patch_name"] for r in records}):
        rows = [r for r in records if r["patch_name"] == cond]
        group_payload: dict[str, Any] = {}
        for group in ("win", "regression"):
            sub = [r for r in rows if r["group"] == group]
            if not sub:
                continue
            if group == "win":
                causal_matches = sum(1 for r in sub if (r["baseline_correct"] is False and r["correct"] is True))
            else:
                causal_matches = sum(1 for r in sub if (r["baseline_correct"] is True and r["correct"] is False))
            group_payload[group] = {
                "n": len(sub),
                "accuracy": sum(1 for r in sub if r["correct"]) / len(sub),
                "causal_match_count": causal_matches,
                "causal_match_rate": causal_matches / len(sub),
                "baseline_consistency_rate": sum(
                    1
                    for r in sub
                    if ((group == "win" and r["baseline_correct"] is False) or (group == "regression" and r["baseline_correct"] is True))
                ) / len(sub),
                "mean_tokens_per_s": float(np.mean([r["tokens_per_s"] for r in sub])),
                "mean_generated_tokens": float(np.mean([r["generated_tokens"] for r in sub])),
                "mean_patched_tokens": float(np.mean([r["patched_tokens"] for r in sub])),
            }
        by_condition[cond] = group_payload

    scored = []
    for cond, payload in by_condition.items():
        win = payload.get("win", {})
        reg = payload.get("regression", {})
        score = win.get("causal_match_rate", 0.0) + reg.get("causal_match_rate", 0.0)
        scored.append({"patch_name": cond, "causal_total": score})
    scored.sort(key=lambda row: row["causal_total"], reverse=True)
    return {"by_condition": by_condition, "ranked_by_causal_total": scored}


def main() -> None:
    ap = argparse.ArgumentParser(description="Live causal patching of think_region directions into think_only generations.")
    ap.add_argument("--model-path", type=Path, default=Path(DEFAULT_MODEL_9B))
    ap.add_argument("--trace-eval", type=Path, default=Path(DEFAULT_TRACE_EVAL))
    ap.add_argument("--greedy-run", type=Path, default=Path(DEFAULT_GREEDY_RUN))
    ap.add_argument("--directions", type=Path, default=Path(DEFAULT_DIRECTIONS_9B))
    ap.add_argument("--output-dir", type=Path, default=Path(DEFAULT_OUTPUT_ROOT) / "causal_think_region_patch_qwen35_20260416")
    ap.add_argument("--dtype", choices=["bfloat16", "float16"], default="bfloat16")
    ap.add_argument("--max-wins", type=int, default=9)
    ap.add_argument("--max-regressions", type=int, default=9)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument(
        "--patch-specs",
        default="baseline=none,l0_1p0=0@1.0:full,l1_1p0=1@1.0:full,l01_1p0=0+1@1.0:full,l01_2p0=0+1@2.0:full",
    )
    ap.add_argument("--max-vram-frac", type=float, default=0.90)
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.output_dir / "run.log"
    write_json(
        args.output_dir / "manifest.json",
        {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
    )

    trace_rows = {task_key(row): row for row in read_trace_eval(args.trace_eval)}
    selected = select_tasks(args.greedy_run, trace_rows, args.max_wins, args.max_regressions)
    directions = load_directions(args.directions)
    patch_specs = parse_patch_specs(args.patch_specs)
    log(log_path, f"selected_tasks={len(selected)} patch_specs={[p['name'] for p in patch_specs]}")
    log(log_path, f"available_direction_layers={sorted(directions)}")

    guard_vram(args.max_vram_frac, log_path, "pre_load")
    tokenizer, model, layers_mod = load_model(args.model_path, args.dtype, log_path)
    guard_vram(args.max_vram_frac, log_path, "post_load")
    used, total, frac = gpu_memory()
    log(log_path, f"loaded model layers={len(layers_mod)} vram={used/1024**3:.1f}/{total/1024**3:.1f}GiB frac={frac:.3f}")

    records_path = args.output_dir / "records.jsonl"
    if records_path.exists():
        records_path.unlink()

    all_records: list[dict[str, Any]] = []
    baseline_cache: dict[str, dict[str, Any]] = {}
    for patch in patch_specs:
        log(log_path, f"running patch={patch}")
        patch_layers = {}
        for layer in patch["layers"]:
            if layer not in directions:
                raise RuntimeError(f"missing direction for layer {layer} in {args.directions}")
            patch_layers[layer] = directions[layer]
        for item in tqdm(selected, desc=f"patch:{patch['name']}", unit="task"):
            messages = make_messages(item["trace_row"], "think_only")
            out = generate_with_patch(
                model=model,
                tokenizer=tokenizer,
                layers_mod=layers_mod,
                messages=messages,
                layer_to_vec=patch_layers,
                alpha=patch["alpha"],
                max_new_tokens=args.max_new_tokens,
                token_limit=patch["token_limit"],
            )
            correct = score_answer(out["text"], item["answer_key"])
            final_answer = extract_final(out["text"])
            row = {
                "timestamp": now_iso(),
                "patch_name": patch["name"],
                "patch_layers": patch["layers"],
                "patch_alpha": patch["alpha"],
                "patch_token_limit": patch["token_limit"],
                "group": item["group"],
                "task_id": item["task_id"],
                "answer_key": item["answer_key"],
                "saved_think_only_correct": item["saved_think_only_correct"],
                "correct": correct,
                "final_answer": final_answer,
                **out,
            }
            if patch["name"] == "baseline":
                baseline_cache[item["task_id"]] = row
            base = baseline_cache.get(item["task_id"])
            row["baseline_correct"] = None if base is None else base["correct"]
            row["baseline_final_answer"] = None if base is None else base["final_answer"]
            append_jsonl(records_path, row)
            all_records.append(row)
        guard_vram(args.max_vram_frac, log_path, f"after_patch_{patch['name']}")
        gc.collect()
        torch.cuda.empty_cache()

    summary = summarize(all_records)
    write_json(args.output_dir / "summary.json", summary)
    log(log_path, f"wrote summary to {args.output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
