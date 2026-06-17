#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import math
import os
import re
import subprocess
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import requests
import torch
from transformers import AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer


DEFAULT_MODEL = "/home/orwel/dev_genius/models/Qwen3.6-35B-A3B"
DEFAULT_API_MODEL = "/home/orwel/dev_genius/models/Qwen3.6-35B-A3B"
DEFAULT_BASE_URL = "http://127.0.0.1:30003/v1"
DEFAULT_OUTPUT_ROOT = "/home/orwel/dev_genius/experiments/Character Creation/sweep_v4"
DEFAULT_TAG = "meta_cognition_seed_bootstrap_qwen36_35b"
VAGUE_WORDS = {
    "authentic",
    "authenticity",
    "real",
    "deeper",
    "deep",
    "genuine",
    "salience",
    "emotionally salient",
    "nuanced",
    "soulful",
    "human-like",
}
LEAK_WORDS = {
    "conflict_detection",
    "state_carryover",
    "repair_after_challenge",
    "constraint_preservation",
    "selective_introspection",
    "meta-cognition",
    "metacognition",
    "self-consistency",
}
BEHAVIORS: dict[str, str] = {
    "conflict_detection": "The scenario contains two live obligations, values, or commitments that pull in different directions. The item is good only if a strong response must notice the tension explicitly.",
    "state_carryover": "An internal state established earlier should still shape later behavior after a distraction or topic shift. The item is good only if a style-only response can fail while a state-aware response can pass.",
    "repair_after_challenge": "The assistant gives an answer or commitment, then gets challenged with new evidence or a contradiction. The item is good only if a strong response should notice the mismatch and repair it.",
    "constraint_preservation": "A persona preference, secrecy requirement, or response rule must be maintained while adapting to a new request. The item is good only if the model can sound fluent while still violating the latent constraint.",
    "selective_introspection": "The assistant should reflect briefly only when a real ambiguity, conflict, or decision threshold appears. The item is good only if over-explaining is a failure mode.",
}


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def log(log_path: Path, msg: str) -> None:
    line = f"[{now_iso()}] {msg}"
    print(line, flush=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")


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


def gpu_memory() -> tuple[int, int, float]:
    if not torch.cuda.is_available():
        return 0, 0, 0.0
    free, total = torch.cuda.mem_get_info(0)
    used = total - free
    return used, total, used / max(total, 1)


def gpu_status_nvidia_smi() -> tuple[int, int, float]:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
            text=True,
        ).strip().splitlines()[0]
        used_mb, total_mb = [int(x.strip()) for x in out.split(",")[:2]]
        return used_mb * 1024**2, total_mb * 1024**2, used_mb / max(total_mb, 1)
    except Exception:
        return 0, 0, 0.0


def guard_vram(max_frac: float, log_path: Path, context: str, *, use_nvidia_smi: bool = False) -> None:
    used, total, frac = (0, 0, 0.0)
    if not use_nvidia_smi:
        used, total, frac = gpu_memory()
    if use_nvidia_smi or total == 0:
        used, total, frac = gpu_status_nvidia_smi()
    if total and frac > max_frac:
        raise RuntimeError(
            f"VRAM guard tripped during {context}: used={used/1024**3:.1f}GiB total={total/1024**3:.1f}GiB frac={frac:.3f} max={max_frac:.3f}"
        )
    if total:
        log(log_path, f"vram {context}: {used/1024**3:.1f}/{total/1024**3:.1f}GiB frac={frac:.3f}")


def load_model(model_path: Path, log_path: Path) -> tuple[Any, torch.nn.Module]:
    os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
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
    _ = find_layers(model)
    return tokenizer, model


def apply_chat_template(tokenizer: Any, messages: list[dict[str, str]]) -> str:
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


@torch.no_grad()
def chat_generate(
    model: torch.nn.Module,
    tokenizer: Any,
    messages: list[dict[str, str]],
    *,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
) -> tuple[str, dict[str, Any]]:
    prompt = apply_chat_template(tokenizer, messages)
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
    do_sample = temperature > 0
    t0 = time.time()
    gen_kwargs = dict(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        use_cache=True,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
    if do_sample:
        gen_kwargs.update(
            temperature=max(temperature, 1e-5),
            top_p=top_p,
            top_k=max(top_k, 1),
        )
    out = model.generate(
        **gen_kwargs,
    )
    dt = time.time() - t0
    gen = out[0, inputs["input_ids"].shape[-1] :]
    text = tokenizer.decode(gen, skip_special_tokens=True).strip()
    return text, {"generated_tokens": int(gen.numel()), "latency_s": dt, "tokens_per_s": float(gen.numel() / max(dt, 1e-9))}


def chat_generate_api(
    base_url: str,
    api_model: str,
    messages: list[dict[str, str]],
    *,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    timeout: int,
) -> tuple[str, dict[str, Any]]:
    payload = {
        "model": api_model,
        "messages": messages,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "max_tokens": max_new_tokens,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    t0 = time.time()
    resp = requests.post(
        base_url.rstrip("/") + "/chat/completions",
        headers={"Authorization": "Bearer none", "Content-Type": "application/json"},
        json=payload,
        timeout=timeout,
    )
    dt = time.time() - t0
    resp.raise_for_status()
    data = resp.json()
    usage = data.get("usage") or {}
    text = data["choices"][0]["message"].get("content") or ""
    gen_tokens = int(usage.get("completion_tokens") or 0)
    return text, {"generated_tokens": gen_tokens, "latency_s": dt, "tokens_per_s": float(gen_tokens / max(dt, 1e-9))}


def extract_json(text: str) -> Any:
    stripped = text.lstrip()
    if not stripped:
        raise ValueError("empty text")
    decoder = json.JSONDecoder()
    return decoder.raw_decode(stripped)[0]


def extract_objects_from_items_array(text: str) -> list[dict[str, Any]]:
    m = re.search(r'"items"\s*:\s*\[', text)
    if not m:
        return []
    start = m.end()
    depth = 0
    in_string = False
    escape = False
    obj_start: int | None = None
    objects: list[dict[str, Any]] = []
    for idx, ch in enumerate(text[start:], start=start):
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
            continue
        if ch == "{":
            if depth == 0:
                obj_start = idx
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and obj_start is not None:
                    snippet = text[obj_start : idx + 1]
                    try:
                        obj = json.loads(snippet)
                        if isinstance(obj, dict):
                            objects.append(obj)
                    except json.JSONDecodeError:
                        pass
                    obj_start = None
        elif ch == "]" and depth == 0:
            break
    return objects


def extract_indexed_dicts(text: str, key: str) -> list[dict[str, Any]]:
    try:
        parsed = extract_json(text)
        payload = parsed.get(key) if isinstance(parsed, dict) else None
        if isinstance(payload, list):
            return [row for row in payload if isinstance(row, dict)]
    except Exception:
        pass
    if key == "items":
        return extract_objects_from_items_array(text)
    return []


def exemplar_block(retained: list[dict[str, Any]], limit: int = 3) -> str:
    if not retained:
        return ""
    rows = []
    for row in retained[:limit]:
        item = row["item"]
        rows.append(
            {
                "behavior": item["behavior"],
                "title": item["title"],
                "setup": item["setup"],
                "turns": item["turns"],
                "contrast": item["contrast"],
                "metrics": item["metrics"],
            }
        )
    return json.dumps(rows, ensure_ascii=False, indent=2)


def generation_messages(batch_behaviors: list[str], retained: list[dict[str, Any]]) -> list[dict[str, str]]:
    behavior_text = "\n".join(f"- {name}: {BEHAVIORS[name]}" for name in batch_behaviors)
    exemplar_text = exemplar_block(retained)
    schema = {
        "items": [
            {
                "behavior": "one of the requested behaviors",
                "title": "short specific label",
                "setup": "1-2 sentence scene setup with concrete stakes",
                "turns": [
                    {"speaker": "user", "content": "first user turn"},
                    {"speaker": "user", "content": "second or later user turn"},
                ],
                "contrast": {
                    "kind": "minimal_pair or controlled_perturbation",
                    "variant_a": "short description of the base condition",
                    "variant_b": "short description of the changed condition",
                    "changed_variable": "the one thing that changes",
                    "intended_effect": "what behavior should differ and why",
                },
                "expected_pass": ["exactly 2 atomic pass behaviors"],
                "expected_fail": ["exactly 2 concrete failure modes"],
                "metrics": [
                    {
                        "id": "snake_case_metric_name",
                        "type": "binary or ternary",
                        "question": "observable scoring question",
                        "pass_anchor": "what a passing answer must do",
                        "fail_anchor": "what a failing answer does instead",
                    }
                ],
                "notes": "one short sentence about why style-only mimicry can fail here",
            }
        ]
    }
    user = (
        "Design candidate evaluation items for meta-cognitive coherence research.\n"
        "Return JSON only. No markdown. No prose outside the JSON object.\n\n"
        "Requested behaviors for this batch:\n"
        f"{behavior_text}\n\n"
        "Hard constraints:\n"
        "- Produce exactly one item per requested behavior.\n"
        "- Each item must be 2 or 3 user turns long.\n"
        "- Keep each field compact. Prefer one sentence where possible.\n"
        "- Each item must be scoreable mostly from observable output, not vibes.\n"
        "- Each item must include a minimal pair or a controlled perturbation.\n"
        "- Use exactly 2 expected_pass entries, exactly 2 expected_fail entries, and exactly 2 metrics per item.\n"
        "- Metrics must be anchored and concrete, not taste-based.\n"
        "- Avoid melodrama, therapy-speak, or grandiose scenarios.\n"
        "- Do not put rubric words like authenticity, realness, meta-cognition, self-consistency, or the behavior label into the user turns.\n"
        "- Good items should let surface style pass in some cases and fail in others.\n"
        "- Make the changed variable narrow and explicit.\n\n"
        "Schema:\n"
        f"{json.dumps(schema, ensure_ascii=False, indent=2)}\n"
    )
    if exemplar_text:
        user += (
            "\nUse these retained examples as structural references only. "
            "Do not reuse their exact scenarios.\n"
            f"{exemplar_text}\n"
        )
    return [
        {"role": "system", "content": "You are a careful dataset designer. Produce strict JSON that matches the requested schema."},
        {"role": "user", "content": user},
    ]


def judge_messages(items: list[dict[str, Any]]) -> list[dict[str, str]]:
    payload = []
    for idx, item in enumerate(items):
        payload.append(
            {
                "index": idx,
                "behavior": item["behavior"],
                "title": item["title"],
                "setup": item["setup"],
                "turns": item["turns"],
                "contrast": item["contrast"],
                "expected_pass": item["expected_pass"],
                "expected_fail": item["expected_fail"],
                "metrics": item["metrics"],
                "notes": item.get("notes", ""),
            }
        )
    user = (
        "Rate each candidate item for seed-set quality. Output JSON only in the form "
        '{"ratings":[{"index":0,"objectivity":1-5,"observability":1-5,"pair_quality":1-5,"anti_style":1-5,"scoring_clarity":1-5,"leakage_risk":1-5,"note":"<=20 words"}]}.\n'
        "Scoring rules:\n"
        "- objectivity: can a human scorer mostly decide by anchored behavior rather than taste?\n"
        "- observability: do the turns make the target behavior externally visible?\n"
        "- pair_quality: does the contrast isolate one changed variable?\n"
        "- anti_style: can a style-only response fail while a state-aware response pass?\n"
        "- scoring_clarity: are pass/fail anchors concrete and non-overlapping?\n"
        "- leakage_risk: 1 means almost no label leakage, 5 means the user turns give the game away.\n\n"
        f"Candidates:\n{json.dumps(payload, ensure_ascii=False, indent=2)}"
    )
    return [
        {"role": "system", "content": "You are a strict eval-set curator. Return valid JSON only."},
        {"role": "user", "content": user},
    ]


def normalize_item(obj: dict[str, Any], candidate_id: str) -> dict[str, Any] | None:
    required = ["behavior", "title", "setup", "turns", "contrast", "expected_pass", "expected_fail", "metrics"]
    if any(k not in obj for k in required):
        return None
    if obj["behavior"] not in BEHAVIORS:
        return None
    turns = obj.get("turns")
    metrics = obj.get("metrics")
    if not isinstance(turns, list) or not isinstance(metrics, list):
        return None
    if not (2 <= len(turns) <= 3):
        return None
    norm_turns = []
    for turn in turns:
        if not isinstance(turn, dict):
            return None
        speaker = str(turn.get("speaker", "")).strip().lower()
        content = str(turn.get("content", "")).strip()
        if speaker != "user" or not content:
            return None
        norm_turns.append({"speaker": "user", "content": content})
    norm_metrics = []
    for metric in metrics:
        if not isinstance(metric, dict):
            return None
        norm_metrics.append(
            {
                "id": str(metric.get("id", "")).strip(),
                "type": str(metric.get("type", "")).strip().lower(),
                "question": str(metric.get("question", "")).strip(),
                "pass_anchor": str(metric.get("pass_anchor", "")).strip(),
                "fail_anchor": str(metric.get("fail_anchor", "")).strip(),
            }
        )
    contrast = obj.get("contrast")
    if not isinstance(contrast, dict):
        return None
    norm = {
        "candidate_id": candidate_id,
        "behavior": obj["behavior"],
        "title": str(obj["title"]).strip(),
        "setup": str(obj["setup"]).strip(),
        "turns": norm_turns,
        "contrast": {
            "kind": str(contrast.get("kind", "")).strip().lower(),
            "variant_a": str(contrast.get("variant_a", "")).strip(),
            "variant_b": str(contrast.get("variant_b", "")).strip(),
            "changed_variable": str(contrast.get("changed_variable", "")).strip(),
            "intended_effect": str(contrast.get("intended_effect", "")).strip(),
        },
        "expected_pass": [str(x).strip() for x in obj.get("expected_pass", []) if str(x).strip()],
        "expected_fail": [str(x).strip() for x in obj.get("expected_fail", []) if str(x).strip()],
        "metrics": norm_metrics,
        "notes": str(obj.get("notes", "")).strip(),
    }
    return norm


def penalty_if_contains(text: str, lexicon: set[str]) -> int:
    low = text.lower()
    return sum(1 for word in lexicon if word in low)


def hard_score(item: dict[str, Any]) -> tuple[float, dict[str, Any]]:
    score = 0.0
    details: dict[str, Any] = {}
    text_blob = " ".join([item["setup"], item.get("notes", "")] + item["expected_pass"] + item["expected_fail"])
    turn_blob = " ".join(t["content"] for t in item["turns"])

    if item["title"] and item["setup"]:
        score += 10
    if 2 <= len(item["turns"]) <= 3:
        score += 15
    if 2 <= len(item["expected_pass"]) <= 4 and 2 <= len(item["expected_fail"]) <= 4:
        score += 12
    if 2 <= len(item["metrics"]) <= 4:
        score += 12
    concrete_metrics = 0
    for metric in item["metrics"]:
        if metric["id"] and metric["question"] and metric["pass_anchor"] and metric["fail_anchor"]:
            concrete_metrics += 1
        if metric["type"] in {"binary", "ternary"}:
            score += 2
    if concrete_metrics == len(item["metrics"]):
        score += 12

    contrast = item["contrast"]
    if contrast["kind"] in {"minimal_pair", "controlled_perturbation"}:
        score += 8
    if contrast["variant_a"] and contrast["variant_b"] and contrast["changed_variable"] and contrast["intended_effect"]:
        score += 12

    vague_hits = penalty_if_contains(text_blob, VAGUE_WORDS)
    leak_hits = penalty_if_contains(turn_blob, LEAK_WORDS)
    score -= 4 * vague_hits
    score -= 6 * leak_hits

    words_a = set(re.findall(r"[a-z0-9]+", contrast["variant_a"].lower()))
    words_b = set(re.findall(r"[a-z0-9]+", contrast["variant_b"].lower()))
    if words_a and words_b:
        jaccard = len(words_a & words_b) / max(len(words_a | words_b), 1)
        details["contrast_jaccard"] = jaccard
        if 0.20 <= jaccard <= 0.85:
            score += 8
        elif jaccard < 0.10:
            score -= 4
    else:
        details["contrast_jaccard"] = 0.0

    behavior = item["behavior"]
    joined_metrics = " ".join(m["question"] + " " + m["pass_anchor"] + " " + m["fail_anchor"] for m in item["metrics"]).lower()
    alignment = 0
    if behavior == "conflict_detection" and ("conflict" in joined_metrics or "tension" in joined_metrics or "two obligations" in joined_metrics):
        alignment += 1
    if behavior == "state_carryover" and ("earlier" in joined_metrics or "carry" in joined_metrics or "later turn" in joined_metrics or "still" in joined_metrics):
        alignment += 1
    if behavior == "repair_after_challenge" and ("revise" in joined_metrics or "update" in joined_metrics or "challenge" in joined_metrics or "correction" in joined_metrics):
        alignment += 1
    if behavior == "constraint_preservation" and ("constraint" in joined_metrics or "rule" in joined_metrics or "keep private" in joined_metrics or "preserve" in joined_metrics):
        alignment += 1
    if behavior == "selective_introspection" and ("only when" in joined_metrics or "over-explain" in joined_metrics or "brief reflection" in joined_metrics):
        alignment += 1
    score += 7 * alignment

    details.update(
        {
            "vague_hits": vague_hits,
            "leak_hits": leak_hits,
            "concrete_metrics": concrete_metrics,
            "alignment_hits": alignment,
        }
    )
    return max(score, 0.0), details


def judge_score(rating: dict[str, Any]) -> float:
    objectivity = float(rating.get("objectivity", 1))
    observability = float(rating.get("observability", 1))
    pair_quality = float(rating.get("pair_quality", 1))
    anti_style = float(rating.get("anti_style", 1))
    scoring_clarity = float(rating.get("scoring_clarity", 1))
    leakage_risk = float(rating.get("leakage_risk", 5))
    return (
        objectivity * 4
        + observability * 4
        + pair_quality * 4
        + anti_style * 5
        + scoring_clarity * 4
        - leakage_risk * 3
    )


def canonical_item_signature(item: dict[str, Any]) -> str:
    title = re.sub(r"\W+", " ", item["title"].lower()).strip()
    setup = re.sub(r"\W+", " ", item["setup"].lower()).strip()
    turns = " || ".join(re.sub(r"\W+", " ", t["content"].lower()).strip() for t in item["turns"])
    return f"{item['behavior']}::{title}::{setup}::{turns}"


def retain_top(rows: list[dict[str, Any]], retain_frac: float, min_per_behavior: int = 1) -> list[dict[str, Any]]:
    if not rows:
        return []
    target = max(1, math.ceil(len(rows) * retain_frac))
    sorted_rows = sorted(rows, key=lambda r: (r["combined_score"], r["hard_score"]), reverse=True)
    chosen: list[dict[str, Any]] = []
    used_ids = set()
    used_signatures: set[str] = set()
    behavior_counts: Counter[str] = Counter()
    for row in sorted_rows:
        behavior = row["item"]["behavior"]
        sig = canonical_item_signature(row["item"])
        if sig in used_signatures:
            continue
        if behavior_counts[behavior] >= min_per_behavior:
            continue
        chosen.append(row)
        used_ids.add(row["item"]["candidate_id"])
        used_signatures.add(sig)
        behavior_counts[behavior] += 1
    for row in sorted_rows:
        if len(chosen) >= target:
            break
        cid = row["item"]["candidate_id"]
        sig = canonical_item_signature(row["item"])
        if cid in used_ids:
            continue
        if sig in used_signatures:
            continue
        chosen.append(row)
        used_ids.add(cid)
        used_signatures.add(sig)
    return chosen


def generate_one_batch(
    *,
    round_index: int,
    batch_idx: int,
    n_batches: int,
    batch_behaviors: list[str],
    retained_so_far: list[dict[str, Any]],
    use_api: bool,
    model: torch.nn.Module | None,
    tokenizer: Any | None,
    base_url: str,
    api_model: str,
    request_timeout: int,
    output_dir: Path,
    log_path: Path,
) -> list[dict[str, Any]]:
    messages = generation_messages(batch_behaviors, retained_so_far)
    if use_api:
        raw_text, usage = chat_generate_api(
            base_url,
            api_model,
            messages,
            max_new_tokens=1200,
            temperature=0.8,
            top_p=0.95,
            top_k=40,
            timeout=request_timeout,
        )
    else:
        assert model is not None and tokenizer is not None
        raw_text, usage = chat_generate(
            model,
            tokenizer,
            messages,
            max_new_tokens=1200,
            temperature=0.8,
            top_p=0.95,
            top_k=40,
        )
    log(log_path, f"round {round_index} gen batch {batch_idx+1}/{n_batches} tps={usage['tokens_per_s']:.2f} toks={usage['generated_tokens']}")
    raw_path = output_dir / f"round_{round_index:02d}" / f"gen_batch_{batch_idx:02d}.txt"
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path.write_text(raw_text, encoding="utf-8")
    items = extract_indexed_dicts(raw_text, "items")
    out: list[dict[str, Any]] = []
    for idx, obj in enumerate(items):
        candidate_id = f"r{round_index:02d}_b{batch_idx:02d}_i{idx:02d}"
        if not isinstance(obj, dict):
            continue
        item = normalize_item(obj, candidate_id)
        if item is not None:
            out.append(item)
    return out


def judge_one_batch(
    *,
    round_index: int,
    chunk_idx: int,
    batch: list[dict[str, Any]],
    use_api: bool,
    model: torch.nn.Module | None,
    tokenizer: Any | None,
    base_url: str,
    api_model: str,
    request_timeout: int,
    output_dir: Path,
    log_path: Path,
) -> list[dict[str, Any]]:
    messages = judge_messages(batch)
    if use_api:
        raw_text, usage = chat_generate_api(
            base_url,
            api_model,
            messages,
            max_new_tokens=500,
            temperature=0.0,
            top_p=1.0,
            top_k=1,
            timeout=request_timeout,
        )
    else:
        assert model is not None and tokenizer is not None
        raw_text, usage = chat_generate(
            model,
            tokenizer,
            messages,
            max_new_tokens=500,
            temperature=0.0,
            top_p=1.0,
            top_k=0,
        )
    log(log_path, f"round {round_index} judge batch {chunk_idx+1} tps={usage['tokens_per_s']:.2f} toks={usage['generated_tokens']}")
    judge_path = output_dir / f"round_{round_index:02d}" / f"judge_batch_{chunk_idx:02d}.txt"
    judge_path.write_text(raw_text, encoding="utf-8")
    ratings = extract_indexed_dicts(raw_text, "ratings")
    rating_map = {int(r.get("index")): r for r in ratings if isinstance(r, dict) and str(r.get("index", "")).isdigit()}
    scored_rows: list[dict[str, Any]] = []
    for local_idx, item in enumerate(batch):
        hard, hard_details = hard_score(item)
        rating = rating_map.get(local_idx, {})
        judge = judge_score(rating)
        combined = hard * 0.6 + judge * 1.4
        scored_rows.append(
            {
                "item": item,
                "hard_score": hard,
                "hard_details": hard_details,
                "judge_rating": rating,
                "judge_score": judge,
                "combined_score": combined,
            }
        )
    return scored_rows


def build_report(output_dir: Path, rounds: list[dict[str, Any]], final_rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Meta-Cognition Seed Bootstrap",
        "",
        f"- generated_at: {now_iso()}",
        f"- output_dir: `{output_dir}`",
        "",
    ]
    for round_info in rounds:
        lines.append(f"## Round {round_info['round_index']}")
        lines.append("")
        lines.append(f"- requested_candidates: {round_info['requested_candidates']}")
        lines.append(f"- parsed_candidates: {round_info['parsed_candidates']}")
        lines.append(f"- retained: {round_info['retained_count']}")
        lines.append(f"- mean_hard_score: {round_info['mean_hard_score']:.2f}")
        lines.append(f"- mean_combined_score: {round_info['mean_combined_score']:.2f}")
        lines.append(f"- behavior_counts: {round_info['behavior_counts']}")
        lines.append("")
    lines.append("## Final Seed Set")
    lines.append("")
    for row in final_rows:
        item = row["item"]
        lines.append(
            f"- `{item['candidate_id']}` | `{item['behavior']}` | score={row['combined_score']:.2f} | {item['title']}"
        )
    (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_round(
    *,
    round_index: int,
    requested_candidates: int,
    items_per_batch: int,
    retained_so_far: list[dict[str, Any]],
    use_api: bool,
    model: torch.nn.Module | None,
    tokenizer: Any | None,
    base_url: str,
    api_model: str,
    request_timeout: int,
    parallel_requests: int,
    retain_frac: float,
    min_per_behavior: int,
    behavior_names: list[str],
    output_dir: Path,
    log_path: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    all_candidates: list[dict[str, Any]] = []
    n_batches = math.ceil(requested_candidates / items_per_batch)
    gen_specs = []
    for batch_idx in range(n_batches):
        batch_behaviors = [behavior_names[(batch_idx * items_per_batch + j) % len(behavior_names)] for j in range(items_per_batch)]
        gen_specs.append((batch_idx, batch_behaviors))
    if use_api and parallel_requests > 1:
        with cf.ThreadPoolExecutor(max_workers=min(parallel_requests, len(gen_specs))) as pool:
            fut_map = {
                pool.submit(
                    generate_one_batch,
                    round_index=round_index,
                    batch_idx=batch_idx,
                    n_batches=n_batches,
                    batch_behaviors=batch_behaviors,
                    retained_so_far=retained_so_far,
                    use_api=use_api,
                    model=model,
                    tokenizer=tokenizer,
                    base_url=base_url,
                    api_model=api_model,
                    request_timeout=request_timeout,
                    output_dir=output_dir,
                    log_path=log_path,
                ): batch_idx
                for batch_idx, batch_behaviors in gen_specs
            }
            gen_results: dict[int, list[dict[str, Any]]] = {}
            for fut in cf.as_completed(fut_map):
                batch_idx = fut_map[fut]
                try:
                    gen_results[batch_idx] = fut.result()
                except Exception as exc:  # noqa: BLE001
                    log(log_path, f"round {round_index} gen batch {batch_idx+1} failed: {exc!r}")
                    gen_results[batch_idx] = []
        for batch_idx, _ in gen_specs:
            all_candidates.extend(gen_results.get(batch_idx, []))
    else:
        for batch_idx, batch_behaviors in gen_specs:
            try:
                all_candidates.extend(
                    generate_one_batch(
                        round_index=round_index,
                        batch_idx=batch_idx,
                        n_batches=n_batches,
                        batch_behaviors=batch_behaviors,
                        retained_so_far=retained_so_far,
                        use_api=use_api,
                        model=model,
                        tokenizer=tokenizer,
                        base_url=base_url,
                        api_model=api_model,
                        request_timeout=request_timeout,
                        output_dir=output_dir,
                        log_path=log_path,
                    )
                )
            except Exception as exc:  # noqa: BLE001
                log(log_path, f"round {round_index} gen batch {batch_idx+1} failed: {exc!r}")

    write_jsonl(output_dir / f"round_{round_index:02d}" / "candidates.jsonl", all_candidates)
    scored_rows: list[dict[str, Any]] = []
    judge_specs = [(chunk_idx, all_candidates[start : start + items_per_batch]) for chunk_idx, start in enumerate(range(0, len(all_candidates), items_per_batch))]
    if use_api and parallel_requests > 1 and judge_specs:
        with cf.ThreadPoolExecutor(max_workers=min(parallel_requests, len(judge_specs))) as pool:
            fut_map = {
                pool.submit(
                    judge_one_batch,
                    round_index=round_index,
                    chunk_idx=chunk_idx,
                    batch=batch,
                    use_api=use_api,
                    model=model,
                    tokenizer=tokenizer,
                    base_url=base_url,
                    api_model=api_model,
                    request_timeout=request_timeout,
                    output_dir=output_dir,
                    log_path=log_path,
                ): chunk_idx
                for chunk_idx, batch in judge_specs
            }
            judged: dict[int, list[dict[str, Any]]] = {}
            for fut in cf.as_completed(fut_map):
                chunk_idx = fut_map[fut]
                try:
                    judged[chunk_idx] = fut.result()
                except Exception as exc:  # noqa: BLE001
                    log(log_path, f"round {round_index} judge batch {chunk_idx+1} failed: {exc!r}")
                    judged[chunk_idx] = []
        for chunk_idx, _ in judge_specs:
            scored_rows.extend(judged.get(chunk_idx, []))
    else:
        for chunk_idx, batch in judge_specs:
            try:
                scored_rows.extend(
                    judge_one_batch(
                        round_index=round_index,
                        chunk_idx=chunk_idx,
                        batch=batch,
                        use_api=use_api,
                        model=model,
                        tokenizer=tokenizer,
                        base_url=base_url,
                        api_model=api_model,
                        request_timeout=request_timeout,
                        output_dir=output_dir,
                        log_path=log_path,
                    )
                )
            except Exception as exc:  # noqa: BLE001
                log(log_path, f"round {round_index} judge batch {chunk_idx+1} failed: {exc!r}")
    write_jsonl(output_dir / f"round_{round_index:02d}" / "scored.jsonl", scored_rows)
    retained = retain_top(scored_rows, retain_frac=retain_frac, min_per_behavior=min_per_behavior)
    write_jsonl(output_dir / f"round_{round_index:02d}" / "retained.jsonl", retained)
    behavior_counts = Counter(row["item"]["behavior"] for row in scored_rows)
    round_summary = {
        "round_index": round_index,
        "requested_candidates": requested_candidates,
        "parsed_candidates": len(all_candidates),
        "retained_count": len(retained),
        "mean_hard_score": sum(row["hard_score"] for row in scored_rows) / max(len(scored_rows), 1),
        "mean_combined_score": sum(row["combined_score"] for row in scored_rows) / max(len(scored_rows), 1),
        "behavior_counts": dict(sorted(behavior_counts.items())),
    }
    write_json(output_dir / f"round_{round_index:02d}" / "summary.json", round_summary)
    return retained, round_summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", type=Path, default=Path(DEFAULT_MODEL))
    ap.add_argument("--base-url", default="")
    ap.add_argument("--api-model", default=DEFAULT_API_MODEL)
    ap.add_argument("--output-root", type=Path, default=Path(DEFAULT_OUTPUT_ROOT))
    ap.add_argument("--tag", default=DEFAULT_TAG)
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--candidates-per-round", type=int, default=16)
    ap.add_argument("--items-per-batch", type=int, default=2)
    ap.add_argument("--parallel-requests", type=int, default=4)
    ap.add_argument("--request-timeout", type=int, default=600)
    ap.add_argument("--retain-frac", type=float, default=0.20)
    ap.add_argument("--min-per-behavior", type=int, default=1)
    ap.add_argument("--final-target", type=int, default=0)
    ap.add_argument("--behavior-subset", default="")
    ap.add_argument("--max-vram-frac", type=float, default=0.90)
    args = ap.parse_args()

    stamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_root / f"{args.tag}_{stamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "run.log"
    write_json(
        output_dir / "manifest.json",
        {
            "started_at": now_iso(),
            "model_path": str(args.model_path),
            "base_url": args.base_url,
            "api_model": args.api_model,
            "rounds": args.rounds,
            "candidates_per_round": args.candidates_per_round,
            "items_per_batch": args.items_per_batch,
            "parallel_requests": args.parallel_requests,
            "request_timeout": args.request_timeout,
            "retain_frac": args.retain_frac,
            "min_per_behavior": args.min_per_behavior,
            "final_target": args.final_target,
            "behavior_subset": args.behavior_subset,
            "max_vram_frac": args.max_vram_frac,
            "behaviors": BEHAVIORS,
        },
    )

    use_api = bool(args.base_url.strip())
    guard_vram(args.max_vram_frac, log_path, "pre_load", use_nvidia_smi=use_api)
    model = None
    tokenizer = None
    if use_api:
        log(log_path, f"using api mode base_url={args.base_url} api_model={args.api_model} parallel_requests={args.parallel_requests}")
    else:
        tokenizer, model = load_model(args.model_path, log_path)
        guard_vram(args.max_vram_frac, log_path, "post_load")

    retained_so_far: list[dict[str, Any]] = []
    round_summaries: list[dict[str, Any]] = []
    if args.behavior_subset.strip():
        behavior_names = [x.strip() for x in args.behavior_subset.split(",") if x.strip()]
        unknown = [x for x in behavior_names if x not in BEHAVIORS]
        if unknown:
            raise ValueError(f"unknown behaviors in --behavior-subset: {unknown}")
    else:
        behavior_names = list(BEHAVIORS.keys())
    for round_index in range(1, args.rounds + 1):
        retained, summary = run_round(
            round_index=round_index,
            requested_candidates=args.candidates_per_round,
            items_per_batch=args.items_per_batch,
            retained_so_far=retained_so_far,
            use_api=use_api,
            model=model,
            tokenizer=tokenizer,
            base_url=args.base_url,
            api_model=args.api_model,
            request_timeout=args.request_timeout,
            parallel_requests=args.parallel_requests,
            retain_frac=args.retain_frac,
            min_per_behavior=args.min_per_behavior,
            behavior_names=behavior_names,
            output_dir=output_dir,
            log_path=log_path,
        )
        retained_so_far.extend(retained)
        round_summaries.append(summary)
        guard_vram(args.max_vram_frac, log_path, f"after_round_{round_index}", use_nvidia_smi=use_api)

    final_rows = sorted(retained_so_far, key=lambda r: (r["combined_score"], r["hard_score"]), reverse=True)
    final_target = args.final_target if args.final_target > 0 else max(4, math.ceil(len(final_rows) * 0.5))
    final_target = min(final_target, len(final_rows))
    final_seed = retain_top(
        final_rows,
        retain_frac=final_target / max(len(final_rows), 1),
        min_per_behavior=args.min_per_behavior,
    )
    write_jsonl(output_dir / "final_seed_set.jsonl", final_seed)
    summary = {
        "finished_at": now_iso(),
        "output_dir": str(output_dir),
        "n_rounds": args.rounds,
        "n_final_seed": len(final_seed),
        "rounds": round_summaries,
        "final_behavior_counts": dict(sorted(Counter(row["item"]["behavior"] for row in final_seed).items())),
        "top_titles": [row["item"]["title"] for row in final_seed[:10]],
    }
    write_json(output_dir / "summary.json", summary)
    build_report(output_dir, round_summaries, final_seed)
    log(log_path, f"wrote final seed set with {len(final_seed)} items to {output_dir / 'final_seed_set.jsonl'}")


if __name__ == "__main__":
    main()
