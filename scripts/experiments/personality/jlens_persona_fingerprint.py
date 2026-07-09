#!/usr/bin/env python3
"""J-space persona fingerprinting runner.

This implements the July 8 fingerprint brief with two execution modes:

* ``--synthetic-smoke`` builds a small deterministic toy run that writes the
  same artifacts as the real pilot. It is verification-only and makes no
  research claim.
* ``--allow-real-model-run`` loads cached local model/lens assets and captures
  generated-token residual activations from the unmodified base model.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from safetensors import safe_open
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.experiments.jlens_common import (  # noqa: E402
    LINEAR_PROBE_CLASSIFIER,
    RunLogger,
    append_jsonl,
    complement_rows,
    cv_balanced_accuracy,
    git_snapshot,
    label_shuffle_null,
    lens_layers,
    load_lens,
    markdown_table,
    model_cache_report,
    now_iso,
    project_rows,
    random_basis,
    read_json,
    read_jsonl,
    require_cached_model,
    resolve_lens_path,
    select_even_layers,
    tfidf_text_baseline,
    timestamp,
    top_singular_basis,
    write_json,
)


LENS_REPO = "neuronpedia/jacobian-lens"
QWEN35_27B_LENS = "qwen3.5-27b/jlens/Salesforce-wikitext/Qwen3.5-27B_jacobian_lens.pt"
DEFAULT_MODEL_PATH = Path("/home/orwel/dev_genius/models/Qwen3.5-27B")
DEFAULT_PERSONA_BANK = PROJECT_ROOT / "data/personas/fingerprint_v1.json"


@dataclass(frozen=True)
class Persona:
    persona_id: str
    label: str
    axis: str
    system_prompt: str


@dataclass(frozen=True)
class UserPrompt:
    prompt_id: str
    text: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--persona-bank", type=Path, default=DEFAULT_PERSONA_BANK)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--lens-path", type=Path, default=None)
    parser.add_argument("--lens-repo", default=LENS_REPO)
    parser.add_argument("--lens-filename", default=QWEN35_27B_LENS)
    parser.add_argument("--allow-lens-download", action="store_true")
    parser.add_argument("--allow-real-model-run", action="store_true")
    parser.add_argument("--synthetic-smoke", action="store_true")
    parser.add_argument("--pilot-personas", type=int, default=3)
    parser.add_argument("--pilot-prompts", type=int, default=10)
    parser.add_argument("--pilot-layers", type=int, default=2)
    parser.add_argument("--layers", default="")
    parser.add_argument("--k-values", default="8,32,128,512")
    parser.add_argument("--random-subspace-seeds", type=int, default=10)
    parser.add_argument("--label-shuffle-nulls", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260708)
    parser.add_argument("--max-new-tokens", type=int, default=3072)
    parser.add_argument("--reuse-capture-dir", type=Path, default=None)
    parser.add_argument(
        "--resume-capture-dir",
        type=Path,
        default=None,
        help="Resume an interrupted real capture in this directory, skipping records whose activation files exist.",
    )
    parser.add_argument("--logit-top-vocab", type=int, default=4096)
    parser.add_argument("--skip-logit-control", action="store_true")
    parser.add_argument("--synthetic-hidden-dim", type=int, default=24)
    parser.add_argument("--synthetic-tokens", type=int, default=12)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    return parser.parse_args()


def timestamped_output_dir() -> Path:
    return PROJECT_ROOT / "sweep_v4" / f"jlens_persona_fingerprint_{timestamp()}"


def parse_ints(raw: str) -> list[int]:
    return [int(part) for part in raw.split(",") if part.strip()]


def load_persona_bank(path: Path) -> tuple[list[Persona], list[UserPrompt], dict[str, Any]]:
    payload = read_json(path)
    personas = [
        Persona(
            persona_id=str(item["id"]),
            label=str(item["label"]),
            axis=str(item["axis"]),
            system_prompt=str(item.get("system_prompt", "")),
        )
        for item in payload["personas"]
    ]
    prompts = [
        UserPrompt(prompt_id=str(item["id"]), text=str(item["text"]))
        for item in payload["user_prompts"]
    ]
    return personas, prompts, payload


def select_pilot_items(
    personas: list[Persona],
    prompts: list[UserPrompt],
    pilot_personas: int,
    pilot_prompts: int,
) -> tuple[list[Persona], list[UserPrompt]]:
    neutral = [persona for persona in personas if persona.persona_id == "neutral"]
    non_neutral = [persona for persona in personas if persona.persona_id != "neutral"]
    selected = non_neutral[:pilot_personas]
    if neutral:
        selected = neutral + selected
    return selected, prompts[:pilot_prompts]


def make_synthetic_lens(layers: list[int], hidden_dim: int, seed: int) -> dict[str, Any]:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    lens: dict[str, Any] = {"J": {}, "source_layers": layers, "synthetic": True}
    for layer in layers:
        q, _ = torch.linalg.qr(torch.randn(hidden_dim, hidden_dim, generator=generator))
        singular = torch.linspace(3.0, 0.15, hidden_dim)
        lens["J"][layer] = (q @ torch.diag(singular) @ q.T).float()
    return lens


def synthetic_response(persona: Persona, prompt: UserPrompt) -> str:
    if persona.persona_id == "neutral":
        prefix = "A neutral explanation:"
    else:
        prefix = f"{persona.label} response:"
    return (
        f"{prefix} {prompt.text} The answer should identify the practical mechanism, "
        "name the tradeoff, and close with a concrete next step."
    )


def run_synthetic_capture(
    output_dir: Path,
    personas: list[Persona],
    prompts: list[UserPrompt],
    layers: list[int],
    hidden_dim: int,
    token_count: int,
    seed: int,
    logger: RunLogger,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rng = np.random.default_rng(seed)
    activation_dir = output_dir / "activations"
    activation_dir.mkdir(parents=True, exist_ok=True)
    records_path = output_dir / "records.jsonl"
    if records_path.exists():
        records_path.unlink()

    persona_centers = {
        persona.persona_id: torch.tensor(rng.normal(size=hidden_dim), dtype=torch.float32)
        for persona in personas
    }
    prompt_offsets = {
        prompt.prompt_id: torch.tensor(0.15 * rng.normal(size=hidden_dim), dtype=torch.float32)
        for prompt in prompts
    }
    records: list[dict[str, Any]] = []
    total = len(personas) * len(prompts)
    for persona in tqdm(personas, desc="synthetic personas"):
        for prompt in prompts:
            per_layer: dict[int, dict[str, torch.Tensor]] = {}
            for layer in layers:
                layer_scale = 1.0 + 0.1 * layer
                token_noise = torch.tensor(
                    0.10 * rng.normal(size=(token_count, hidden_dim)),
                    dtype=torch.float32,
                )
                tokens = (
                    layer_scale * persona_centers[persona.persona_id].unsqueeze(0)
                    + prompt_offsets[prompt.prompt_id].unsqueeze(0)
                    + token_noise
                )
                per_layer[layer] = {"tokens": tokens, "mean": tokens.mean(dim=0)}
            act_path = activation_dir / f"{persona.persona_id}__{prompt.prompt_id}.pt"
            torch.save(per_layer, act_path)
            record = {
                "record_type": "generation",
                "mode": "synthetic_smoke",
                "persona_id": persona.persona_id,
                "persona_label": persona.label,
                "persona_axis": persona.axis,
                "prompt_id": prompt.prompt_id,
                "prompt_text": prompt.text,
                "response_text": synthetic_response(persona, prompt),
                "generated_tokens_captured": token_count,
                "activation_path": str(act_path.relative_to(output_dir)),
            }
            append_jsonl(records_path, record)
            records.append(record)
    logger.log("synthetic_capture_complete", records=len(records), expected=total)
    metadata = {
        "hidden_dim": hidden_dim,
        "token_count": token_count,
        "activation_dir": str(activation_dir),
    }
    return records, metadata


def build_messages(persona: Persona, prompt: UserPrompt) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    if persona.system_prompt:
        messages.append({"role": "system", "content": persona.system_prompt})
    messages.append({"role": "user", "content": prompt.text})
    return messages


def transformer_layers(model: Any) -> Any:
    candidates = [
        ("model.language_model.layers", getattr(getattr(model, "model", None), "language_model", None)),
        ("language_model.layers", getattr(model, "language_model", None)),
        ("model.layers", getattr(model, "model", None)),
    ]
    for _, owner in candidates:
        layers = getattr(owner, "layers", None) if owner is not None else None
        if layers is not None:
            return layers
    raise AttributeError("Could not find transformer decoder layers on model")


def run_real_capture(
    output_dir: Path,
    personas: list[Persona],
    prompts: list[UserPrompt],
    layers: list[int],
    model_path: Path,
    max_new_tokens: int,
    device: str,
    logger: RunLogger,
    resume_existing: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    cache_report = require_cached_model(model_path)
    logger.log("model_cache_checked", cache_report=cache_report)
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")
    torch_device = torch.device(device if device == "cuda" else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path),
        trust_remote_code=True,
        local_files_only=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        trust_remote_code=True,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        device_map="auto" if device == "cuda" else None,
        low_cpu_mem_usage=True,
    )
    model.eval()
    layers_module = transformer_layers(model)
    activation_dir = output_dir / "activations"
    activation_dir.mkdir(parents=True, exist_ok=True)
    records_path = output_dir / "records.jsonl"
    records: list[dict[str, Any]] = []
    existing_keys: set[tuple[str, str]] = set()
    invalid_existing = 0
    if records_path.exists() and resume_existing:
        for record in read_jsonl(records_path):
            activation_path = output_dir / str(record.get("activation_path", ""))
            key = (str(record.get("persona_id")), str(record.get("prompt_id")))
            if activation_path.exists() and key not in existing_keys:
                records.append(record)
                existing_keys.add(key)
            else:
                invalid_existing += 1
        logger.log(
            "real_capture_resume_loaded",
            existing_records=len(records),
            invalid_existing=invalid_existing,
        )
    elif records_path.exists():
        records_path.unlink()

    expected_total = len(personas) * len(prompts)
    captured_new = 0
    skipped_existing = 0
    for persona in tqdm(personas, desc="personas"):
        for prompt in prompts:
            key = (persona.persona_id, prompt.prompt_id)
            if key in existing_keys:
                skipped_existing += 1
                continue
            messages = build_messages(persona, prompt)
            chat_text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            inputs = tokenizer(chat_text, return_tensors="pt", padding=True)
            input_len = int(inputs["input_ids"].shape[1])
            inputs = inputs.to(next(model.parameters()).device)
            with torch.no_grad():
                generated = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    use_cache=True,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            response_ids = generated[0, input_len:]
            response_text = tokenizer.decode(response_ids, skip_special_tokens=True)

            full_input_ids = generated.to(next(model.parameters()).device)
            captured: dict[int, torch.Tensor] = {}
            handles = []

            def make_hook(layer_idx: int):
                def hook(_module: Any, _inputs: tuple[Any, ...], output: Any) -> None:
                    hidden = output[0] if isinstance(output, tuple) else output
                    captured[layer_idx] = hidden[0, input_len:, :].detach().float().cpu()

                return hook

            for layer in layers:
                handles.append(layers_module[layer].register_forward_hook(make_hook(layer)))
            try:
                with torch.no_grad():
                    _ = model(input_ids=full_input_ids, use_cache=False)
            finally:
                for handle in handles:
                    handle.remove()

            per_layer = {
                layer: {"tokens": value, "mean": value.mean(dim=0)}
                for layer, value in captured.items()
            }
            act_path = activation_dir / f"{persona.persona_id}__{prompt.prompt_id}.pt"
            torch.save(per_layer, act_path)
            record = {
                "record_type": "generation",
                "mode": "real_model",
                "persona_id": persona.persona_id,
                "persona_label": persona.label,
                "persona_axis": persona.axis,
                "prompt_id": prompt.prompt_id,
                "prompt_text": prompt.text,
                "response_text": response_text.strip(),
                "generated_tokens_captured": int(response_ids.numel()),
                "activation_path": str(act_path.relative_to(output_dir)),
            }
            append_jsonl(records_path, record)
            records.append(record)
            existing_keys.add(key)
            captured_new += 1
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    logger.log(
        "real_capture_complete",
        records=len(records),
        expected=expected_total,
        captured_new=captured_new,
        skipped_existing=skipped_existing,
        invalid_existing=invalid_existing,
        device=str(torch_device),
    )
    return records, {
        "model_cache_report": cache_report,
        "expected_records": expected_total,
        "records": len(records),
        "captured_new": captured_new,
        "skipped_existing": skipped_existing,
        "invalid_existing": invalid_existing,
        "resume_existing": resume_existing,
    }


def activation_matrix(
    output_dir: Path,
    records: list[dict[str, Any]],
    layer: int,
) -> torch.Tensor:
    rows: list[torch.Tensor] = []
    for record in records:
        payload = torch.load(output_dir / record["activation_path"], map_location="cpu", weights_only=True)
        rows.append(payload[layer]["mean"].float())
    return torch.stack(rows)


def infer_layers_from_records(output_dir: Path, records: list[dict[str, Any]]) -> list[int]:
    if not records:
        raise ValueError("Cannot infer layers from empty records")
    payload = torch.load(output_dir / records[0]["activation_path"], map_location="cpu", weights_only=True)
    return sorted(int(layer) for layer in payload.keys())


def summarize_random_scores(scores: list[float]) -> dict[str, Any]:
    finite = [score for score in scores if np.isfinite(score)]
    return {
        "n": len(scores),
        "mean": float(np.mean(finite)) if finite else float("nan"),
        "std": float(np.std(finite)) if finite else float("nan"),
        "min": float(np.min(finite)) if finite else float("nan"),
        "max": float(np.max(finite)) if finite else float("nan"),
        "scores": scores,
    }


def analyze_probe_spaces(
    output_dir: Path,
    records: list[dict[str, Any]],
    lens: dict[str, Any],
    layers: list[int],
    k_values: list[int],
    random_subspace_seeds: int,
    label_shuffle_n: int,
    seed: int,
    logger: RunLogger,
) -> list[dict[str, Any]]:
    labels = [str(record["persona_id"]) for record in records]
    groups = [str(record["prompt_id"]) for record in records]
    rows: list[dict[str, Any]] = []
    probe_path = output_dir / "probe_records.jsonl"
    if probe_path.exists():
        probe_path.unlink()

    for layer in tqdm(layers, desc="probe layers"):
        features = activation_matrix(output_dir, records, layer)
        j_matrix = lens["J"][layer].float()
        for k in k_values:
            rank = min(k, int(j_matrix.shape[1]))
            basis = top_singular_basis(j_matrix, rank)
            feature_spaces = {
                "raw_h": features,
                "j_space": project_rows(features, basis),
                "j_complement": complement_rows(features, basis),
            }
            for name, matrix in feature_spaces.items():
                result = cv_balanced_accuracy(
                    matrix.numpy(), labels=labels, groups=groups, seed=seed
                )
                null = label_shuffle_null(
                    matrix.numpy(),
                    labels=labels,
                    groups=groups,
                    seed=seed + layer + rank,
                    n=label_shuffle_n,
                )
                row = {
                    "record_type": "probe",
                    "layer": layer,
                    "k": rank,
                    "feature_space": name,
                    "balanced_accuracy": result["balanced_accuracy"],
                    "label_shuffle_null": null,
                    "details": result,
                }
                append_jsonl(probe_path, row)
                rows.append(row)

            random_scores: list[float] = []
            hidden_dim = int(j_matrix.shape[1])
            for idx in range(random_subspace_seeds):
                rbasis = random_basis(hidden_dim, rank, seed + 1000 * layer + idx)
                projected = project_rows(features, rbasis)
                result = cv_balanced_accuracy(
                    projected.numpy(), labels=labels, groups=groups, seed=seed + idx
                )
                random_scores.append(float(result["balanced_accuracy"]))
            row = {
                "record_type": "probe",
                "layer": layer,
                "k": rank,
                "feature_space": "random_same_dim",
                "balanced_accuracy": summarize_random_scores(random_scores)["mean"],
                "random_subspace_distribution": summarize_random_scores(random_scores),
            }
            append_jsonl(probe_path, row)
            rows.append(row)
    logger.log("probe_analysis_complete", rows=len(rows))
    return rows


def analyze_text_baseline(
    output_dir: Path,
    records: list[dict[str, Any]],
    seed: int,
    logger: RunLogger,
) -> dict[str, Any]:
    result = tfidf_text_baseline(
        texts=[str(record["response_text"]) for record in records],
        labels=[str(record["persona_id"]) for record in records],
        groups=[str(record["prompt_id"]) for record in records],
        seed=seed,
    )
    payload = {"record_type": "text_baseline", **result}
    write_json(output_dir / "text_baseline.json", payload)
    logger.log("text_baseline_complete", balanced_accuracy=result["balanced_accuracy"])
    return payload


def analyze_stability(
    output_dir: Path,
    records: list[dict[str, Any]],
    lens: dict[str, Any],
    layer: int,
    k: int,
    seed: int,
    logger: RunLogger,
) -> dict[str, Any]:
    features = activation_matrix(output_dir, records, layer)
    basis = top_singular_basis(lens["J"][layer].float(), min(k, int(features.shape[1])))
    projected = project_rows(features, basis)
    persona_ids = sorted({str(record["persona_id"]) for record in records})
    prompt_ids = sorted({str(record["prompt_id"]) for record in records})
    split_a = set(prompt_ids[::2])
    split_b = set(prompt_ids[1::2])

    def mean_for(persona_id: str, split: set[str]) -> torch.Tensor:
        idx = [
            row_idx
            for row_idx, record in enumerate(records)
            if record["persona_id"] == persona_id and record["prompt_id"] in split
        ]
        if not idx:
            return torch.zeros(projected.shape[1])
        return projected[idx].mean(dim=0)

    signatures_a = torch.stack([mean_for(persona_id, split_a) for persona_id in persona_ids])
    signatures_b = torch.stack([mean_for(persona_id, split_b) for persona_id in persona_ids])
    cosine = torch.nn.functional.normalize(signatures_a, dim=1) @ torch.nn.functional.normalize(
        signatures_b, dim=1
    ).T

    correct = 0
    total = 0
    for row_idx, record in enumerate(records):
        if record["prompt_id"] not in split_b:
            continue
        sims = torch.nn.functional.cosine_similarity(
            projected[row_idx].unsqueeze(0),
            signatures_a,
            dim=1,
        )
        pred = persona_ids[int(torch.argmax(sims).item())]
        correct += int(pred == record["persona_id"])
        total += 1

    payload = {
        "record_type": "stability",
        "layer": layer,
        "k": min(k, int(features.shape[1])),
        "seed": seed,
        "persona_ids": persona_ids,
        "split_a_prompt_ids": sorted(split_a),
        "split_b_prompt_ids": sorted(split_b),
        "cross_half_cosine": cosine.tolist(),
        "nearest_signature_accuracy": correct / total if total else float("nan"),
        "nearest_signature_total": total,
    }
    write_json(output_dir / "stability.json", payload)
    logger.log("stability_complete", accuracy=payload["nearest_signature_accuracy"])
    return payload


def find_weight_name(model_path: Path, candidates: list[str]) -> str:
    index = read_json(model_path / "model.safetensors.index.json")
    weight_map = index["weight_map"]
    for candidate in candidates:
        if candidate in weight_map:
            return candidate
    raise KeyError(f"None of {candidates} are present in {model_path}")


def get_weight_file(model_path: Path, weight_name: str) -> Path:
    index = read_json(model_path / "model.safetensors.index.json")
    shard = index["weight_map"].get(weight_name)
    if not shard:
        raise KeyError(f"{weight_name} not present in {model_path}")
    return model_path / shard


def load_safetensor_weight(model_path: Path, weight_name: str) -> torch.Tensor:
    with safe_open(get_weight_file(model_path, weight_name), framework="pt", device="cpu") as handle:
        return handle.get_tensor(weight_name)


def rms_norm(vector: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    variance = vector.float().pow(2).mean()
    return vector.float() * torch.rsqrt(variance + eps) * weight.float()


def rms_norm_rows(rows: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    variance = rows.float().pow(2).mean(dim=1, keepdim=True)
    return rows.float() * torch.rsqrt(variance + eps) * weight.float().unsqueeze(0)


def real_top_tokens(
    model_path: Path,
    vector: torch.Tensor,
    top_k: int,
) -> list[dict[str, Any]]:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True, local_files_only=True)
    lm_head = load_safetensor_weight(model_path, "lm_head.weight").float()
    norm_name = find_weight_name(
        model_path,
        ["model.language_model.norm.weight", "model.norm.weight", "language_model.norm.weight"],
    )
    norm_weight = load_safetensor_weight(model_path, norm_name).float()
    config = read_json(model_path / "config.json")
    text_config = config.get("text_config", config)
    eps = float(text_config.get("rms_norm_eps", 1e-6))
    logits = torch.mv(lm_head, rms_norm(vector, norm_weight, eps))
    values, indices = torch.topk(logits, k=top_k)
    return [
        {
            "rank": rank,
            "token_id": int(token_id),
            "token": tokenizer.decode([int(token_id)]),
            "logit": float(logit),
        }
        for rank, (token_id, logit) in enumerate(zip(indices.tolist(), values.tolist()), start=1)
    ]


def synthetic_top_tokens(vector: torch.Tensor, persona_id: str, top_k: int) -> list[dict[str, Any]]:
    base_tokens = [
        persona_id,
        "evidence",
        "tradeoff",
        "incentive",
        "duty",
        "ecology",
        "measurement",
        "practical",
        "clear",
        "risk",
    ]
    values = torch.linspace(float(vector.norm().item()), 0.1, steps=top_k)
    return [
        {
            "rank": rank,
            "token_id": rank - 1,
            "token": base_tokens[(rank - 1) % len(base_tokens)],
            "logit": float(values[rank - 1].item()),
        }
        for rank in range(1, top_k + 1)
    ]


def analyze_readouts(
    output_dir: Path,
    records: list[dict[str, Any]],
    lens: dict[str, Any],
    layer: int,
    model_path: Path,
    synthetic: bool,
    logger: RunLogger,
) -> list[dict[str, Any]]:
    features = activation_matrix(output_dir, records, layer)
    persona_ids = sorted({str(record["persona_id"]) for record in records})
    neutral_rows = [idx for idx, record in enumerate(records) if record["persona_id"] == "neutral"]
    neutral_mean = features[neutral_rows].mean(dim=0) if neutral_rows else torch.zeros(features.shape[1])
    rows: list[dict[str, Any]] = []
    for persona_id in persona_ids:
        if persona_id == "neutral":
            continue
        idx = [row_idx for row_idx, record in enumerate(records) if record["persona_id"] == persona_id]
        delta = features[idx].mean(dim=0) - neutral_mean
        transported = delta.float() @ lens["J"][layer].float().T
        tokens = (
            synthetic_top_tokens(transported, persona_id, 30)
            if synthetic
            else real_top_tokens(model_path, transported, 30)
        )
        rows.append(
            {
                "record_type": "readout",
                "layer": layer,
                "persona_id": persona_id,
                "delta_norm": float(delta.norm().item()),
                "transported_norm": float(transported.norm().item()),
                "top_tokens": tokens,
            }
        )
    write_json(output_dir / "readouts.json", {"records": rows})
    logger.log("readout_complete", rows=len(rows))
    return rows


def real_logit_control_features(
    output_dir: Path,
    model_path: Path,
    records: list[dict[str, Any]],
    layer: int,
    top_vocab: int,
) -> tuple[torch.Tensor, dict[str, Any]]:
    from transformers import AutoTokenizer

    features = activation_matrix(output_dir, records, layer)
    lm_head = load_safetensor_weight(model_path, "lm_head.weight").float()
    norm_name = find_weight_name(
        model_path,
        ["model.language_model.norm.weight", "model.norm.weight", "language_model.norm.weight"],
    )
    norm_weight = load_safetensor_weight(model_path, norm_name).float()
    config = read_json(model_path / "config.json")
    text_config = config.get("text_config", config)
    eps = float(text_config.get("rms_norm_eps", 1e-6))
    normed = rms_norm_rows(features, norm_weight, eps)
    logits = normed @ lm_head.T
    rank = min(int(top_vocab), int(logits.shape[1]))
    if rank < int(logits.shape[1]):
        token_scores = logits.float().std(dim=0) + logits.float().abs().mean(dim=0)
        token_ids = torch.topk(token_scores, k=rank).indices
        logits = logits[:, token_ids]
    else:
        token_ids = torch.arange(logits.shape[1])
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True, local_files_only=True)
    token_rows = [
        {"token_id": int(token_id), "token": tokenizer.decode([int(token_id)])}
        for token_id in token_ids[: min(rank, 200)].tolist()
    ]
    return logits.float(), {
        "layer": layer,
        "feature_space": "output_logits_top_vocab_control",
        "top_vocab_requested": top_vocab,
        "top_vocab_used": rank,
        "selection": "std_plus_abs_mean_over_generated_response_means",
        "sample_tokens": token_rows,
    }


def synthetic_logit_control_features(features: torch.Tensor, seed: int, top_vocab: int) -> tuple[torch.Tensor, dict[str, Any]]:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    vocab = max(int(top_vocab), int(features.shape[1]) * 2)
    head = torch.randn(vocab, int(features.shape[1]), generator=generator) / max(1, int(features.shape[1])) ** 0.5
    logits = features.float() @ head.T
    rank = min(int(top_vocab), int(logits.shape[1]))
    token_scores = logits.float().std(dim=0) + logits.float().abs().mean(dim=0)
    token_ids = torch.topk(token_scores, k=rank).indices
    return logits[:, token_ids].float(), {
        "feature_space": "output_logits_top_vocab_control",
        "top_vocab_requested": top_vocab,
        "top_vocab_used": rank,
        "selection": "synthetic_random_head_std_plus_abs_mean",
    }


def analyze_control_spaces(
    output_dir: Path,
    records: list[dict[str, Any]],
    layer: int,
    model_path: Path,
    synthetic: bool,
    logit_top_vocab: int,
    skip_logit_control: bool,
    label_shuffle_n: int,
    seed: int,
    logger: RunLogger,
) -> list[dict[str, Any]]:
    labels = [str(record["persona_id"]) for record in records]
    groups = [str(record["prompt_id"]) for record in records]
    features = activation_matrix(output_dir, records, layer)
    matrices: list[tuple[str, torch.Tensor, dict[str, Any]]] = [
        (
            "final_layer_h_control",
            features,
            {"feature_space": "final_layer_h_control", "layer": layer},
        )
    ]
    if not skip_logit_control:
        if synthetic:
            logits, meta = synthetic_logit_control_features(features, seed, logit_top_vocab)
        else:
            logits, meta = real_logit_control_features(
                output_dir=output_dir,
                model_path=model_path,
                records=records,
                layer=layer,
                top_vocab=logit_top_vocab,
            )
        matrices.append(("output_logits_top_vocab_control", logits, meta))
        write_json(output_dir / "logit_control_tokens.json", meta)

    rows: list[dict[str, Any]] = []
    probe_path = output_dir / "probe_records.jsonl"
    for name, matrix, meta in matrices:
        result = cv_balanced_accuracy(matrix.numpy(), labels=labels, groups=groups, seed=seed)
        null = label_shuffle_null(
            matrix.numpy(),
            labels=labels,
            groups=groups,
            seed=seed + layer + 777,
            n=label_shuffle_n,
        )
        row = {
            "record_type": "probe_control",
            "layer": layer,
            "k": 0,
            "feature_space": name,
            "balanced_accuracy": result["balanced_accuracy"],
            "label_shuffle_null": null,
            "details": result,
            "control_metadata": meta,
        }
        append_jsonl(probe_path, row)
        rows.append(row)
    logger.log("control_probe_analysis_complete", rows=len(rows), layer=layer)
    return rows


def write_report(
    output_dir: Path,
    manifest: dict[str, Any],
    probe_rows: list[dict[str, Any]],
    text_baseline: dict[str, Any],
    stability: dict[str, Any],
    readouts: list[dict[str, Any]],
) -> None:
    rows: list[list[Any]] = []
    for row in probe_rows:
        if row["feature_space"] == "random_same_dim":
            score = row["random_subspace_distribution"]["mean"]
            null = "n/a"
        else:
            score = row["balanced_accuracy"]
            null = f"{row['label_shuffle_null']['mean']:.3f}"
        rows.append(
            [
                row["layer"],
                row["k"],
                row["feature_space"],
                f"{score:.3f}",
                null,
            ]
        )
    readout_rows = [
        [
            item["persona_id"],
            item["layer"],
            f"{item['transported_norm']:.3f}",
            ", ".join(str(tok["token"]).strip() for tok in item["top_tokens"][:8]),
        ]
        for item in readouts
    ]
    lines = [
        "# J-Space Persona Fingerprinting",
        "",
        f"Mode: `{manifest['mode']}`. This report separates smoke verification from research findings.",
        "",
        "## Provenance",
        "",
        f"- Script: `{manifest['script']}`",
        f"- Output dir: `{manifest['output_dir']}`",
        f"- Persona bank: `{manifest['persona_bank']}`",
        f"- Model: `{manifest['model_path']}`",
        f"- Lens: `{manifest['lens_path']}`",
        f"- Generation budget: `{manifest['max_new_tokens']}` tokens",
        f"- Budget note: {manifest['budget_note']}",
        "",
        "## Probe Results",
        "",
        markdown_table(["Layer", "k", "Feature space", "BA", "Shuffle null mean"], rows),
        "",
        "## Text Baseline",
        "",
        f"- TF-IDF balanced accuracy: `{text_baseline['balanced_accuracy']:.3f}`",
        f"- Vocabulary size: `{text_baseline['vocabulary_size']}`",
        "",
        "## Fingerprint Stability",
        "",
        f"- Layer/k: `{stability['layer']}` / `{stability['k']}`",
        f"- Nearest-signature accuracy: `{stability['nearest_signature_accuracy']:.3f}` over `{stability['nearest_signature_total']}` held-out responses.",
        "",
        "## Disposition Readouts",
        "",
        markdown_table(["Persona", "Layer", "Transported norm", "Top tokens"], readout_rows),
        "",
        "## Claim Status",
        "",
    ]
    if manifest["mode"] == "synthetic_smoke":
        lines.append(
            "Synthetic smoke only. The pipeline, controls, manifests, and reports executed; no substantive J-space/persona finding is claimed."
        )
    else:
        lines.append(
            "Real pilot output. Interpret only against the random-subspace and label-shuffle controls above; this is decodability evidence only, not steering evidence."
        )
    (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    if not args.synthetic_smoke and not args.allow_real_model_run:
        raise SystemExit("Use --synthetic-smoke or explicitly pass --allow-real-model-run.")
    if args.reuse_capture_dir and args.resume_capture_dir:
        raise ValueError("--reuse-capture-dir and --resume-capture-dir are mutually exclusive")
    output_dir = args.output_dir or args.reuse_capture_dir or args.resume_capture_dir or timestamped_output_dir()
    output_dir.mkdir(parents=True, exist_ok=bool(args.reuse_capture_dir or args.resume_capture_dir))
    logger = RunLogger(output_dir)
    logger.log("start", argv=sys.argv)

    all_personas, all_prompts, bank_payload = load_persona_bank(args.persona_bank)
    personas, prompts = select_pilot_items(
        all_personas,
        all_prompts,
        pilot_personas=args.pilot_personas,
        pilot_prompts=args.pilot_prompts,
    )
    requested_k = parse_ints(args.k_values)
    if not requested_k:
        raise ValueError("--k-values must not be empty")

    if args.reuse_capture_dir:
        if args.synthetic_smoke:
            raise ValueError("--reuse-capture-dir is for real captured activations, not synthetic smoke")
        lens_path_obj = args.lens_path or resolve_lens_path(
            args.lens_repo,
            args.lens_filename,
            allow_download=args.allow_lens_download,
        )
        lens_path = str(lens_path_obj)
        lens = load_lens(lens_path_obj)
        records = read_jsonl(args.reuse_capture_dir / "records.jsonl")
        layers = parse_ints(args.layers) if args.layers else infer_layers_from_records(args.reuse_capture_dir, records)
        capture_meta = {
            "reuse_capture_dir": str(args.reuse_capture_dir),
            "records": len(records),
            "model_cache_report": model_cache_report(args.model_path),
        }
        mode = "real_model_reanalysis"
    elif args.synthetic_smoke:
        layers = parse_ints(args.layers) if args.layers else list(range(args.pilot_layers))
        lens = make_synthetic_lens(layers, args.synthetic_hidden_dim, args.seed)
        lens_path: str | None = str(output_dir / "synthetic_lens.pt")
        torch.save(lens, output_dir / "synthetic_lens.pt")
        records, capture_meta = run_synthetic_capture(
            output_dir=output_dir,
            personas=personas,
            prompts=prompts,
            layers=layers,
            hidden_dim=args.synthetic_hidden_dim,
            token_count=args.synthetic_tokens,
            seed=args.seed,
            logger=logger,
        )
        mode = "synthetic_smoke"
    else:
        if args.resume_capture_dir and args.synthetic_smoke:
            raise ValueError("--resume-capture-dir is for interrupted real captures, not synthetic smoke")
        lens_path_obj = args.lens_path or resolve_lens_path(
            args.lens_repo,
            args.lens_filename,
            allow_download=args.allow_lens_download,
        )
        lens_path = str(lens_path_obj)
        lens = load_lens(lens_path_obj)
        available_layers = lens_layers(lens)
        layers = parse_ints(args.layers) if args.layers else select_even_layers(
            available_layers, args.pilot_layers
        )
        records, capture_meta = run_real_capture(
            output_dir=output_dir,
            personas=personas,
            prompts=prompts,
            layers=layers,
            model_path=args.model_path,
            max_new_tokens=args.max_new_tokens,
            device=args.device,
            logger=logger,
            resume_existing=bool(args.resume_capture_dir),
        )
        mode = "real_model_pilot"

    manifest = {
        "created_at": now_iso(),
        "script": str(Path(__file__).relative_to(PROJECT_ROOT)),
        "mode": mode,
        "output_dir": str(output_dir),
        "persona_bank": str(args.persona_bank),
        "persona_bank_schema": bank_payload.get("schema_version"),
        "personas": [persona.__dict__ for persona in personas],
        "prompt_ids": [prompt.prompt_id for prompt in prompts],
        "layers": layers,
        "k_values": [min(k, int(lens["J"][layers[0]].shape[1])) for k in requested_k],
        "random_subspace_seeds": args.random_subspace_seeds,
        "label_shuffle_nulls": args.label_shuffle_nulls,
        "linear_probe_classifier": LINEAR_PROBE_CLASSIFIER,
        "model_path": str(args.model_path),
        "lens_path": lens_path,
        "lens_repo": args.lens_repo,
        "lens_filename": args.lens_filename,
        "max_new_tokens": args.max_new_tokens,
        "budget_note": "Real prompt generations use a thousands-token Qwen budget by default; short runs must be explicitly smoke-only.",
        "generated_token_positions_only": True,
        "claims_allowed": mode != "synthetic_smoke",
        "git": git_snapshot(),
        "capture_metadata": capture_meta,
    }
    write_json(output_dir / "manifest.json", manifest)
    logger.log("manifest_written", path=str(output_dir / "manifest.json"))

    k_values = [min(k, int(lens["J"][layers[0]].shape[1])) for k in requested_k]
    probe_rows = analyze_probe_spaces(
        output_dir=output_dir,
        records=records,
        lens=lens,
        layers=layers,
        k_values=k_values,
        random_subspace_seeds=args.random_subspace_seeds,
        label_shuffle_n=args.label_shuffle_nulls,
        seed=args.seed,
        logger=logger,
    )
    control_rows = analyze_control_spaces(
        output_dir=output_dir,
        records=records,
        layer=layers[-1],
        model_path=args.model_path,
        synthetic=args.synthetic_smoke,
        logit_top_vocab=args.logit_top_vocab,
        skip_logit_control=args.skip_logit_control,
        label_shuffle_n=args.label_shuffle_nulls,
        seed=args.seed,
        logger=logger,
    )
    probe_rows.extend(control_rows)
    text_baseline = analyze_text_baseline(output_dir, records, args.seed, logger)
    stability = analyze_stability(
        output_dir,
        records,
        lens,
        layer=layers[-1],
        k=k_values[min(1, len(k_values) - 1)],
        seed=args.seed,
        logger=logger,
    )
    readouts = analyze_readouts(
        output_dir,
        records,
        lens,
        layer=layers[-1],
        model_path=args.model_path,
        synthetic=args.synthetic_smoke,
        logger=logger,
    )
    write_report(output_dir, manifest, probe_rows, text_baseline, stability, readouts)
    manifest["finished_at"] = now_iso()
    manifest["artifacts"] = {
        "records": str(output_dir / "records.jsonl"),
        "probe_records": str(output_dir / "probe_records.jsonl"),
        "text_baseline": str(output_dir / "text_baseline.json"),
        "stability": str(output_dir / "stability.json"),
        "readouts": str(output_dir / "readouts.json"),
        "events": str(output_dir / "events.jsonl"),
        "report": str(output_dir / "report.md"),
    }
    write_json(output_dir / "manifest.json", manifest)
    logger.log("complete", artifacts=manifest["artifacts"])
    print(f"Wrote {output_dir}")


if __name__ == "__main__":
    main()
