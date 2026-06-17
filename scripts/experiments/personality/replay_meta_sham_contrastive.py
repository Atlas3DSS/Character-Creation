#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import math
import os
import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_MODEL_PATH = "/home/orwel/.cache/huggingface/hub/models--Qwen--Qwen3.5-9B/snapshots/c202236235762e1c871ad0ccb60c8ee5ba337b9a"
DEFAULT_TRACE_EVAL = "/home/orwel/dev_genius/experiments/Character Creation/sweep_v4/personality_meta_eval_trace_explicit_v1"
DEFAULT_GREEDY_RUN = "/home/orwel/dev_genius/experiments/Character Creation/sweep_v4/meta_sham_control_ages_qwen35_20260416"
DEFAULT_OUTPUT_ROOT = "/home/orwel/dev_genius/experiments/Character Creation/sweep_v4"
DEFAULT_TAG = "meta_sham_contrastive_replay_qwen35_20260416"
CONDITIONS = ("think_only", "real_meta", "sham_meta", "generic_prep")
COMMON_REGIONS = (
    "prompt_last",
    "assistant_all",
    "assistant_early",
    "assistant_late",
    "think_region",
    "answer_region",
    "final_answer_region",
)
CONTROL_BLOCK_REGION = "control_block"
COMPARISONS = {
    "real_minus_think": ("real_meta", "think_only"),
    "real_minus_sham": ("real_meta", "sham_meta"),
    "real_minus_generic": ("real_meta", "generic_prep"),
}
REGION_PRIORITY = {
    "think_region": 0,
    "assistant_all": 1,
    "assistant_late": 2,
    "prompt_last": 3,
    "control_block": 4,
    "assistant_early": 5,
    "answer_region": 6,
    "final_answer_region": 7,
}
VIABILITY_LOO_THRESHOLD = 0.72
VIABILITY_GAP_THRESHOLD = 0.035

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


@dataclass(frozen=True)
class ExampleKey:
    label: str
    task_id: str
    condition: str


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp.replace(path)


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


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float32, copy=False)
    b = b.astype(np.float32, copy=False)
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na <= 1e-12 or nb <= 1e-12:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def mean_pairwise_cos(vectors: list[np.ndarray]) -> float:
    if len(vectors) < 2:
        return 0.0
    total = 0.0
    count = 0
    for i in range(len(vectors)):
        for j in range(i + 1, len(vectors)):
            total += cosine(vectors[i], vectors[j])
            count += 1
    return total / max(count, 1)


def mean_cross_cos(a_vecs: list[np.ndarray], b_vecs: list[np.ndarray]) -> float:
    if not a_vecs or not b_vecs:
        return 0.0
    total = 0.0
    count = 0
    for a in a_vecs:
        for b in b_vecs:
            total += cosine(a, b)
            count += 1
    return total / max(count, 1)


def loo_nearest_centroid(vectors: list[np.ndarray], labels: list[int], metas: list[dict[str, Any]]) -> dict[str, Any]:
    assert len(vectors) == len(labels) == len(metas)
    if len(vectors) < 4 or len(set(labels)) < 2:
        return {"loo_accuracy": 0.0, "rows": []}
    rows: list[dict[str, Any]] = []
    correct = 0
    normed = []
    for vec in vectors:
        v = vec.astype(np.float32, copy=False)
        n = np.linalg.norm(v)
        normed.append(v / max(n, 1e-12))
    for idx in range(len(normed)):
        pos = [normed[j] for j in range(len(normed)) if j != idx and labels[j] == 1]
        neg = [normed[j] for j in range(len(normed)) if j != idx and labels[j] == 0]
        if not pos or not neg:
            rows.append({"task_id": metas[idx]["task_id"], "label": labels[idx], "pred": None})
            continue
        pos_centroid = np.mean(np.stack(pos, axis=0), axis=0)
        neg_centroid = np.mean(np.stack(neg, axis=0), axis=0)
        sim_pos = cosine(normed[idx], pos_centroid)
        sim_neg = cosine(normed[idx], neg_centroid)
        pred = 1 if sim_pos >= sim_neg else 0
        correct += int(pred == labels[idx])
        rows.append(
            {
                "task_id": metas[idx]["task_id"],
                "label": labels[idx],
                "pred": pred,
                "sim_win": sim_pos,
                "sim_regression": sim_neg,
                "margin": sim_pos - sim_neg,
                "comparison": metas[idx]["comparison"],
                "region": metas[idx]["region"],
                "layer": metas[idx]["layer"],
            }
        )
    return {"loo_accuracy": correct / len(normed), "rows": rows}


def token_span_from_chars(offsets: list[tuple[int, int]], start_char: int, end_char: int) -> tuple[int, int] | None:
    token_ids = [i for i, (s, e) in enumerate(offsets) if e > start_char and s < end_char]
    if not token_ids:
        return None
    return token_ids[0], token_ids[-1] + 1


def find_char_regions(text: str) -> dict[str, tuple[int, int]]:
    regions: dict[str, tuple[int, int]] = {}
    control_match = re.search(r"(/meta-think\b.*?/end-meta-think|/prep-think\b.*?/end-prep-think)", text, flags=re.I | re.S)
    think_match = re.search(r"/think\b.*?/end-think", text, flags=re.I | re.S)
    expl_match = re.search(r"Explanation\s*:", text, flags=re.I)
    final_match = re.search(r"Final Answer\s*:", text, flags=re.I)
    if control_match:
        regions[CONTROL_BLOCK_REGION] = (control_match.start(), control_match.end())
    if think_match:
        regions["think_region"] = (think_match.start(), think_match.end())
    if expl_match:
        regions["answer_region"] = (expl_match.start(), len(text))
    elif final_match:
        regions["answer_region"] = (final_match.start(), len(text))
    if final_match:
        regions["final_answer_region"] = (final_match.start(), len(text))
    return regions


def build_regions(
    assistant_text: str,
    assistant_token_count: int,
    offsets: list[tuple[int, int]],
) -> dict[str, tuple[int, int]]:
    regions: dict[str, tuple[int, int]] = {
        "assistant_all": (0, assistant_token_count),
        "assistant_early": (0, min(32, assistant_token_count)),
        "assistant_late": (max(0, assistant_token_count - 32), assistant_token_count),
    }
    for name, (start_char, end_char) in find_char_regions(assistant_text).items():
        span = token_span_from_chars(offsets, start_char, end_char)
        if span is not None and span[1] > span[0]:
            regions[name] = span
    return regions


def gpu_memory() -> tuple[float, float, float]:
    free, total = torch.cuda.mem_get_info()
    used = total - free
    frac = used / max(total, 1)
    return used, total, frac


def chat_text(tokenizer: Any, messages: list[dict[str, str]], add_generation_prompt: bool) -> str:
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            enable_thinking=False,
        )
    except TypeError:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=add_generation_prompt)


def collect_sequences(
    trace_rows: dict[str, dict[str, Any]],
    records_by_task: dict[str, dict[str, dict[str, Any]]],
    win_task_ids: list[str],
    regression_task_ids: list[str],
) -> list[dict[str, Any]]:
    sequence_rows: list[dict[str, Any]] = []
    for label, task_ids in (("win", win_task_ids), ("regression", regression_task_ids)):
        for task_id in task_ids:
            if task_id not in trace_rows:
                raise KeyError(f"missing trace row for {task_id}")
            if task_id not in records_by_task:
                raise KeyError(f"missing saved records for {task_id}")
            for condition in CONDITIONS:
                rec = records_by_task[task_id].get(condition)
                if rec is None:
                    raise KeyError(f"missing record for task={task_id} condition={condition}")
                if rec.get("error"):
                    raise RuntimeError(f"saved record has error task={task_id} condition={condition}: {rec['error']}")
                sequence_rows.append(
                    {
                        "label": label,
                        "task_id": task_id,
                        "condition": condition,
                        "trace_row": trace_rows[task_id],
                        "assistant_text": rec.get("text") or "",
                        "correct": bool(rec.get("correct")),
                        "format_ok": bool(rec.get("format_ok")),
                        "final_answer": rec.get("final_answer"),
                    }
                )
    return sequence_rows


def capture_bank(
    model: Any,
    tokenizer: Any,
    sequence_rows: list[dict[str, Any]],
) -> tuple[dict[str, dict[str, dict[str, np.ndarray]]], dict[str, Any]]:
    bank: dict[str, dict[str, dict[str, np.ndarray]]] = defaultdict(dict)
    metadata: dict[str, Any] = {"examples": []}
    n_layers = int(model.config.num_hidden_layers)
    hidden_size = int(model.config.hidden_size)
    for idx, row in enumerate(tqdm(sequence_rows, desc="replay", unit="seq")):
        messages = make_messages(row["trace_row"], row["condition"])
        prompt_text = chat_text(tokenizer, messages, add_generation_prompt=True)
        full_text = chat_text(
            tokenizer,
            messages + [{"role": "assistant", "content": row["assistant_text"]}],
            add_generation_prompt=False,
        )
        prompt_ids = tokenizer(prompt_text, add_special_tokens=False).input_ids
        full_inputs = tokenizer(full_text, add_special_tokens=False, return_tensors="pt")
        assistant_enc = tokenizer(row["assistant_text"], add_special_tokens=False, return_offsets_mapping=True)
        assistant_token_count = len(assistant_enc.input_ids)
        assistant_start = len(prompt_ids)
        total_tokens = int(full_inputs["input_ids"].shape[-1])
        if total_tokens < assistant_start + assistant_token_count:
            raise RuntimeError(
                f"assistant span overflow task={row['task_id']} condition={row['condition']} total={total_tokens} "
                f"assistant_start={assistant_start} assistant_tokens={assistant_token_count}"
            )
        regions = build_regions(row["assistant_text"], assistant_token_count, assistant_enc["offset_mapping"])
        full_inputs = {k: v.to(model.device) for k, v in full_inputs.items()}
        with torch.no_grad():
            outputs = model(**full_inputs, use_cache=False, output_hidden_states=True)
        hidden_states = outputs.hidden_states[1:]
        example_regions: dict[str, np.ndarray] = {}
        for region_name in set(COMMON_REGIONS).union({CONTROL_BLOCK_REGION}):
            example_regions[region_name] = np.full((n_layers, hidden_size), np.nan, dtype=np.float16)
        for layer_idx, hs in enumerate(hidden_states):
            seq_hs = hs[0]
            prompt_last_vec = seq_hs[assistant_start - 1].detach().float().cpu().numpy().astype(np.float16)
            example_regions["prompt_last"][layer_idx] = prompt_last_vec
            for region_name, span in regions.items():
                start, end = span
                slice_start = assistant_start + start
                slice_end = assistant_start + end
                if slice_end <= slice_start:
                    continue
                vec = seq_hs[slice_start:slice_end].mean(dim=0).detach().float().cpu().numpy().astype(np.float16)
                example_regions[region_name][layer_idx] = vec
        bank[row["task_id"]][row["condition"]] = example_regions
        metadata["examples"].append(
            {
                "task_id": row["task_id"],
                "label": row["label"],
                "condition": row["condition"],
                "correct": row["correct"],
                "format_ok": row["format_ok"],
                "final_answer": row["final_answer"],
                "assistant_tokens": assistant_token_count,
                "total_tokens": total_tokens,
                "regions_present": sorted([name for name in regions]),
            }
        )
        del outputs, hidden_states, full_inputs
        if (idx + 1) % 8 == 0:
            gc.collect()
            torch.cuda.empty_cache()
    metadata["n_layers"] = n_layers
    metadata["hidden_size"] = hidden_size
    return bank, metadata


def layer_region_vector(bank: dict[str, dict[str, dict[str, np.ndarray]]], task_id: str, condition: str, region: str, layer: int) -> np.ndarray | None:
    arr = bank.get(task_id, {}).get(condition, {}).get(region)
    if arr is None or layer >= arr.shape[0]:
        return None
    vec = arr[layer].astype(np.float32, copy=False)
    if np.isnan(vec).any():
        return None
    return vec


def analyze_bank(
    bank: dict[str, dict[str, dict[str, np.ndarray]]],
    win_task_ids: list[str],
    regression_task_ids: list[str],
    n_layers: int,
    min_examples: int,
    min_per_class: int,
) -> dict[str, Any]:
    task_labels = {task_id: 1 for task_id in win_task_ids}
    task_labels.update({task_id: 0 for task_id in regression_task_ids})
    all_task_ids = win_task_ids + regression_task_ids
    metrics: list[dict[str, Any]] = []
    candidate_direction_rows: list[dict[str, Any]] = []
    for comparison_name, (cond_a, cond_b) in COMPARISONS.items():
        region_names = list(COMMON_REGIONS)
        if cond_b in {"sham_meta", "generic_prep"}:
            region_names.append(CONTROL_BLOCK_REGION)
        for region in region_names:
            for layer in range(n_layers):
                vectors: list[np.ndarray] = []
                labels: list[int] = []
                metas: list[dict[str, Any]] = []
                win_vectors: list[np.ndarray] = []
                regression_vectors: list[np.ndarray] = []
                for task_id in all_task_ids:
                    va = layer_region_vector(bank, task_id, cond_a, region, layer)
                    vb = layer_region_vector(bank, task_id, cond_b, region, layer)
                    if va is None or vb is None:
                        continue
                    delta = va - vb
                    vectors.append(delta)
                    label = task_labels[task_id]
                    labels.append(label)
                    metas.append({"task_id": task_id, "comparison": comparison_name, "region": region, "layer": layer})
                    if label == 1:
                        win_vectors.append(delta)
                    else:
                        regression_vectors.append(delta)
                if len(vectors) < min_examples or len(win_vectors) < min_per_class or len(regression_vectors) < min_per_class:
                    continue
                loo = loo_nearest_centroid(vectors, labels, metas)
                win_within = mean_pairwise_cos(win_vectors)
                reg_within = mean_pairwise_cos(regression_vectors)
                cross = mean_cross_cos(win_vectors, regression_vectors)
                mean_norm = float(np.mean([np.linalg.norm(v) for v in vectors]))
                gap = ((win_within + reg_within) / 2.0) - cross
                pos_centroid = np.mean(np.stack(win_vectors, axis=0), axis=0)
                neg_centroid = np.mean(np.stack(regression_vectors, axis=0), axis=0)
                direction = pos_centroid - neg_centroid
                direction_norm = float(np.linalg.norm(direction))
                metrics.append(
                    {
                        "comparison": comparison_name,
                        "condition_a": cond_a,
                        "condition_b": cond_b,
                        "region": region,
                        "layer": layer,
                        "n_examples": len(vectors),
                        "n_wins": len(win_vectors),
                        "n_regressions": len(regression_vectors),
                        "loo_accuracy": loo["loo_accuracy"],
                        "win_within_cos": win_within,
                        "regression_within_cos": reg_within,
                        "cross_cos": cross,
                        "gap": gap,
                        "mean_delta_norm": mean_norm,
                        "direction_norm": direction_norm,
                        "rows": loo["rows"],
                    }
                )
                candidate_direction_rows.append(
                    {
                        "comparison": comparison_name,
                        "region": region,
                        "layer": layer,
                        "direction": direction.astype(np.float32),
                        "direction_norm": direction_norm,
                        "loo_accuracy": loo["loo_accuracy"],
                        "gap": gap,
                    }
                )
    metrics.sort(key=lambda row: (row["loo_accuracy"], row["gap"], row["direction_norm"]), reverse=True)

    robust: list[dict[str, Any]] = []
    grouped: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in metrics:
        grouped[(row["layer"], row["region"])].append(row)
    for (layer, region), rows in grouped.items():
        if len(rows) < 2:
            continue
        robust.append(
            {
                "layer": layer,
                "region": region,
                "comparisons": [r["comparison"] for r in rows],
                "mean_loo_accuracy": float(np.mean([r["loo_accuracy"] for r in rows])),
                "min_loo_accuracy": float(np.min([r["loo_accuracy"] for r in rows])),
                "mean_gap": float(np.mean([r["gap"] for r in rows])),
                "max_gap": float(np.max([r["gap"] for r in rows])),
            }
        )
    robust.sort(key=lambda row: (row["mean_loo_accuracy"], row["mean_gap"]), reverse=True)
    viable = False
    if robust:
        top = robust[0]
        viable = top["mean_loo_accuracy"] >= VIABILITY_LOO_THRESHOLD and top["mean_gap"] >= VIABILITY_GAP_THRESHOLD
    top_direction_rows = []
    for cand in sorted(candidate_direction_rows, key=lambda row: (row["loo_accuracy"], row["gap"], row["direction_norm"]), reverse=True):
        if cand["direction_norm"] <= 1e-8:
            continue
        top_direction_rows.append(cand)
        if len(top_direction_rows) >= 8:
            break

    preferred_direction_rows = []
    preferred_pool = [
        cand for cand in candidate_direction_rows
        if cand["region"] not in {"answer_region", "final_answer_region"}
    ]
    for cand in sorted(
        preferred_pool,
        key=lambda row: (
            REGION_PRIORITY.get(row["region"], 999),
            -row["loo_accuracy"],
            -row["gap"],
            -row["direction_norm"],
        ),
    ):
        if cand["direction_norm"] <= 1e-8:
            continue
        preferred_direction_rows.append(cand)
        if len(preferred_direction_rows) >= 8:
            break
    if len(preferred_direction_rows) < 8:
        seen = {(row["comparison"], row["region"], row["layer"]) for row in preferred_direction_rows}
        for cand in sorted(candidate_direction_rows, key=lambda row: (row["loo_accuracy"], row["gap"], row["direction_norm"]), reverse=True):
            key = (cand["comparison"], cand["region"], cand["layer"])
            if cand["direction_norm"] <= 1e-8 or key in seen:
                continue
            preferred_direction_rows.append(cand)
            seen.add(key)
            if len(preferred_direction_rows) >= 8:
                break
    return {
        "metrics": metrics,
        "robust_windows": robust,
        "viable": viable,
        "top_directions": top_direction_rows,
        "preferred_directions": preferred_direction_rows,
    }


def save_directions(path: Path, direction_rows: list[dict[str, Any]]) -> None:
    arrays: dict[str, np.ndarray] = {}
    meta: list[dict[str, Any]] = []
    for row in direction_rows:
        key = f"{row['comparison']}__{row['region']}__L{row['layer']:02d}"
        arrays[key] = row["direction"]
        meta.append(
            {
                "key": key,
                "comparison": row["comparison"],
                "region": row["region"],
                "layer": row["layer"],
                "direction_norm": row["direction_norm"],
                "loo_accuracy": row["loo_accuracy"],
                "gap": row["gap"],
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **arrays)
    write_json(path.with_suffix(".json"), meta)


def build_report(summary: dict[str, Any]) -> str:
    lines = []
    lines.append("# Meta/Sham Contrastive Replay")
    lines.append("")
    lines.append(f"- Timestamp: `{summary['timestamp']}`")
    lines.append(f"- Model path: `{summary['model_path']}`")
    lines.append(f"- Examples: `{summary['n_examples']}` total, `{summary['n_wins']}` wins, `{summary['n_regressions']}` regressions")
    lines.append(f"- Layers: `{summary['n_layers']}`, hidden size: `{summary['hidden_size']}`")
    lines.append(f"- Viable signal: `{summary['viable']}`")
    lines.append("")
    lines.append("## Top Robust Windows")
    lines.append("")
    if not summary["robust_windows"]:
        lines.append("No robust windows were found.")
    else:
        lines.append("| Layer | Region | Mean LOO | Min LOO | Mean Gap | Comparisons |")
        lines.append("| --- | --- | ---: | ---: | ---: | --- |")
        for row in summary["robust_windows"][:12]:
            lines.append(
                f"| `L{row['layer']:02d}` | `{row['region']}` | `{row['mean_loo_accuracy']:.3f}` | "
                f"`{row['min_loo_accuracy']:.3f}` | `{row['mean_gap']:.3f}` | "
                f"`{', '.join(row['comparisons'])}` |"
            )
    lines.append("")
    lines.append("## Top Single Comparison Windows")
    lines.append("")
    lines.append("| Comparison | Layer | Region | LOO | Gap | Mean Delta Norm |")
    lines.append("| --- | --- | --- | ---: | ---: | ---: |")
    for row in summary["top_metrics"][:15]:
        lines.append(
            f"| `{row['comparison']}` | `L{row['layer']:02d}` | `{row['region']}` | "
            f"`{row['loo_accuracy']:.3f}` | `{row['gap']:.3f}` | `{row['mean_delta_norm']:.2f}` |"
        )
    lines.append("")
    lines.append("## Best By Region")
    lines.append("")
    lines.append("| Region | Comparison | Layer | LOO | Gap |")
    lines.append("| --- | --- | --- | ---: | ---: |")
    for row in summary["best_by_region"]:
        lines.append(
            f"| `{row['region']}` | `{row['comparison']}` | `L{row['layer']:02d}` | "
            f"`{row['loo_accuracy']:.3f}` | `{row['gap']:.3f}` |"
        )
    lines.append("")
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description="Replay saved meta/sham control transcripts through local Qwen3.5-9B and score contrastive hidden-state signal.")
    ap.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    ap.add_argument("--trace-eval", default=DEFAULT_TRACE_EVAL)
    ap.add_argument("--greedy-run", default=DEFAULT_GREEDY_RUN)
    ap.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    ap.add_argument("--tag", default=DEFAULT_TAG)
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16"])
    ap.add_argument("--max-wins", type=int, default=9)
    ap.add_argument("--max-regressions", type=int, default=9)
    ap.add_argument("--min-examples", type=int, default=17)
    ap.add_argument("--min-per-class", type=int, default=8)
    args = ap.parse_args()

    os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
    output_dir = Path(args.output_root) / args.tag
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.json"
    metrics_compact_path = output_dir / "metrics_compact.jsonl"
    report_path = Path("/home/orwel/dev_genius/experiments/Character Creation/reports") / f"{args.tag}.md"

    greedy_dir = Path(args.greedy_run)
    greedy_summary = load_json(greedy_dir / "summary.json")
    greedy_records = load_jsonl(greedy_dir / "records.jsonl")
    trace_rows = {task_key(row): row for row in read_trace_eval(Path(args.trace_eval))}

    win_task_ids = list(greedy_summary["real_meta_unique_fixes_vs_controls"])[: args.max_wins]
    regression_task_ids = list(greedy_summary["paired_vs_think_only"]["real_meta"]["marginal_regressions"])[: args.max_regressions]

    records_by_task: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in greedy_records:
        task_id = row.get("task_id")
        condition = row.get("condition")
        if task_id and condition in CONDITIONS:
            records_by_task[task_id][condition] = row

    sequence_rows = collect_sequences(
        trace_rows=trace_rows,
        records_by_task=records_by_task,
        win_task_ids=win_task_ids,
        regression_task_ids=regression_task_ids,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    torch_dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        torch_dtype=torch_dtype,
        device_map={"": 0},
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
    )
    model.eval()

    bank, bank_meta = capture_bank(model, tokenizer, sequence_rows)
    used, total, frac = gpu_memory()
    analysis = analyze_bank(
        bank,
        win_task_ids,
        regression_task_ids,
        bank_meta["n_layers"],
        min_examples=args.min_examples,
        min_per_class=args.min_per_class,
    )
    top_direction_rows = analysis["top_directions"] if analysis["viable"] else []
    preferred_direction_rows = analysis["preferred_directions"] if analysis["viable"] else []
    if top_direction_rows:
        save_directions(output_dir / "candidate_directions_overall.npz", top_direction_rows)
    if preferred_direction_rows:
        save_directions(output_dir / "candidate_directions.npz", preferred_direction_rows)

    metrics_compact = [
        {k: v for k, v in row.items() if k != "rows"}
        for row in analysis["metrics"]
    ]
    write_jsonl(metrics_compact_path, metrics_compact)

    best_by_region: list[dict[str, Any]] = []
    seen_regions: set[str] = set()
    for row in analysis["metrics"]:
        if row["region"] in seen_regions:
            continue
        best_by_region.append({k: v for k, v in row.items() if k != "rows"})
        seen_regions.add(row["region"])

    summary = {
        "timestamp": now_iso(),
        "model_path": args.model_path,
        "dtype": args.dtype,
        "n_examples": len(sequence_rows),
        "n_wins": len(win_task_ids),
        "n_regressions": len(regression_task_ids),
        "win_task_ids": win_task_ids,
        "regression_task_ids": regression_task_ids,
        "n_layers": bank_meta["n_layers"],
        "hidden_size": bank_meta["hidden_size"],
        "min_examples": args.min_examples,
        "min_per_class": args.min_per_class,
        "vram_used_gb": round(used / 1024**3, 2),
        "vram_total_gb": round(total / 1024**3, 2),
        "vram_frac": round(frac, 4),
        "viable": analysis["viable"],
        "robust_windows": analysis["robust_windows"][:20],
        "top_metrics": analysis["metrics"][:60],
        "best_by_region": best_by_region,
        "metrics_compact_path": str(metrics_compact_path),
        "examples": bank_meta["examples"],
        "candidate_direction_count": len(preferred_direction_rows),
    }
    write_json(summary_path, summary)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(build_report(summary), encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir), "summary": str(summary_path), "report": str(report_path), "viable": analysis["viable"]}, indent=2))


if __name__ == "__main__":
    main()
