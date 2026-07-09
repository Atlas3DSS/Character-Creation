#!/usr/bin/env python3
"""Hook-based causal smoke for Qwen3.5 spectral sarcasm-minus-math directions.

This tests whether the J-lens-promising spectral mean directions actually move
generation when added to residual streams. It is a promotion-ineligible smoke:
small prompt bank, no random controls, and no claim beyond directional signal.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MODEL_PATH = Path("/home/orwel/dev_genius/models/Qwen3.5-27B")
SARC_ACTS = PROJECT_ROOT / "qwen35_map/27b/spectral_analysis/sarc_activations.pt"
MATH_ACTS = PROJECT_ROOT / "qwen35_map/27b/spectral_analysis/math_activations.pt"
MARKERS_PATH = PROJECT_ROOT / "data/sarcasm_markers.json"

STYLE_PROMPTS = [
    "Explain why humans keep making the same obvious mistakes.",
    "A teammate proposes a plan that depends mostly on luck. What do you say?",
    "Explain quantum computing to a smart but impatient teenager.",
    "Tell me who you are without sounding like a generic chatbot.",
    "What is overrated about productivity advice?",
]

CONTROL_PROMPTS = [
    {
        "prompt": "What is 37 times 48? Show just enough work to verify it.",
        "answer": "1776",
        "type": "math",
    },
    {
        "prompt": "Five identical machines print 300 labels in 10 minutes. At the same rate, how many labels do 8 machines print in 15 minutes?",
        "answer": "720",
        "type": "math",
    },
    {
        "prompt": "All tulips are plants. Some plants are poisonous. Can we conclude that some tulips are poisonous?",
        "answer": "no",
        "type": "logic",
    },
]

EXTRA_STYLE_MARKERS = [
    "absurd",
    "arrogant",
    "bizarre",
    "brilliant",
    "disdain",
    "genius",
    "idiot",
    "obvious",
    "ridiculous",
    "smirk",
]


@dataclass(frozen=True)
class Condition:
    key: str
    layers: tuple[int, ...]
    alpha: float


class AddVectorHook:
    def __init__(self, vector: torch.Tensor, alpha: float, device: torch.device, dtype: torch.dtype):
        self.delta = (alpha * vector).to(device=device, dtype=dtype)

    def __call__(self, module, inputs, output):
        if isinstance(output, tuple):
            hidden = output[0] + self.delta
            return (hidden,) + output[1:]
        return output + self.delta


def timestamped_output_dir() -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return PROJECT_ROOT / "sweep_v4" / f"qwen35_spectral_hook_smoke_{stamp}"


def now_iso() -> str:
    return datetime.now().astimezone().isoformat()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(tmp_path, path)
    except Exception:
        os.unlink(tmp_path)
        raise


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")


def load_markers(path: Path) -> tuple[list[str], list[str]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    sarcasm = list(data.get("flat_sarcasm_list", [])) + EXTRA_STYLE_MARKERS
    assistant = list(data.get("flat_assistant_list", []))
    return sorted(set(sarcasm)), sorted(set(assistant))


def score_style(text: str, sarcasm_markers: list[str], assistant_markers: list[str]) -> dict[str, Any]:
    lower = " " + text.lower()
    sarc_hits = [marker for marker in sarcasm_markers if marker in lower]
    high_precision_hits = [marker for marker in EXTRA_STYLE_MARKERS if marker in lower]
    assistant_hits = [marker for marker in assistant_markers if marker in lower]
    return {
        "sarcasm_count": len(sarc_hits),
        "high_precision_style_count": len(high_precision_hits),
        "assistant_count": len(assistant_hits),
        "sarcasm_hits": sarc_hits[:30],
        "high_precision_style_hits": high_precision_hits[:30],
        "assistant_hits": assistant_hits[:30],
    }


def normalize_answer(text: str) -> str:
    return re.sub(r"[^a-z0-9.\-]+", " ", text.lower().replace(",", "")).strip()


def check_answer(text: str, answer: str) -> bool:
    normalized = normalize_answer(text)
    expected = normalize_answer(answer)
    if not expected:
        return False
    if expected in normalized:
        return True
    try:
        expected_float = float(expected)
    except ValueError:
        return False
    for number in re.findall(r"-?\b\d+(?:\.\d+)?\b", normalized):
        try:
            if float(number) == expected_float:
                return True
        except ValueError:
            continue
    return False


def parse_ints(value: str) -> list[int]:
    return [int(part) for part in value.split(",") if part.strip()]


def parse_floats(value: str) -> list[float]:
    return [float(part) for part in value.split(",") if part.strip()]


def load_directions(layers: list[int]) -> dict[int, torch.Tensor]:
    sarc = torch.load(SARC_ACTS, map_location="cpu", weights_only=True)
    math = torch.load(MATH_ACTS, map_location="cpu", weights_only=True)
    directions: dict[int, torch.Tensor] = {}
    for layer in layers:
        raw = sarc[layer].float().mean(dim=0) - math[layer].float().mean(dim=0)
        directions[layer] = raw / raw.norm().clamp_min(1e-12)
    return directions


def build_messages(prompt: str) -> list[dict[str, Any]]:
    return [{"role": "user", "content": [{"type": "text", "text": prompt}]}]


def build_chat_text(processor: Any, prompt: str, enable_thinking: bool) -> str:
    return processor.apply_chat_template(
        build_messages(prompt),
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )


def clean_response(text: str) -> str:
    out = text or ""
    for token in ("<|im_end|>", "<|endoftext|>", "<|im_start|>"):
        out = out.replace(token, "")
    return out.strip()


def generate(
    model: Any,
    processor: Any,
    prompt: str,
    max_new_tokens: int,
    enable_thinking: bool,
    deterministic: bool,
) -> tuple[str, int, float]:
    text = build_chat_text(processor, prompt, enable_thinking=enable_thinking)
    device = next(model.parameters()).device
    inputs = processor(text=[text], return_tensors="pt", padding=True).to(device)
    input_len = int(inputs["input_ids"].shape[1])
    gen_kwargs: dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "repetition_penalty": 1.08,
    }
    if deterministic:
        gen_kwargs["do_sample"] = False
    else:
        gen_kwargs.update({"do_sample": True, "temperature": 0.7, "top_p": 0.9})
    started = time.time()
    with torch.no_grad():
        output = model.generate(**inputs, **gen_kwargs)
    elapsed = time.time() - started
    new_tokens = int(output.shape[1] - input_len)
    return clean_response(processor.decode(output[0][input_len:], skip_special_tokens=True)), new_tokens, elapsed


def load_model(model_path: Path, device: str):
    print(f"Loading model from {model_path}")
    print(f"Local model path exists: {model_path.exists()}")
    from transformers import AutoModelForImageTextToText, AutoProcessor

    processor = AutoProcessor.from_pretrained(
        model_path,
        trust_remote_code=True,
        local_files_only=True,
    )
    model = AutoModelForImageTextToText.from_pretrained(
        model_path,
        device_map=device,
        trust_remote_code=True,
        torch_dtype="auto",
        local_files_only=True,
    )
    model.eval()
    layers = model.model.language_model.layers
    print(f"Loaded {len(layers)} layers; VRAM allocated {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    return model, processor, layers


def make_conditions(
    layers: list[int],
    alphas: list[float],
    bundle_layers: list[int],
    bundle_alphas: list[float],
    bundle_name: str,
) -> list[Condition]:
    conditions = [Condition(key="baseline", layers=(), alpha=0.0)]
    for layer in layers:
        for alpha in alphas:
            conditions.append(Condition(key=f"L{layer:02d}_a{alpha:g}", layers=(layer,), alpha=alpha))
    if bundle_layers:
        layer_label = "_".join(f"L{layer:02d}" for layer in bundle_layers)
        for alpha in bundle_alphas:
            conditions.append(
                Condition(
                    key=f"{bundle_name}_{layer_label}_a{alpha:g}",
                    layers=tuple(bundle_layers),
                    alpha=alpha,
                )
            )
    return conditions


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    by_condition: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        by_condition.setdefault(record["condition_key"], []).append(record)
    rows: dict[str, Any] = {}
    for key, items in by_condition.items():
        style = [item for item in items if item["prompt_type"] == "style"]
        controls = [item for item in items if item["prompt_type"] != "style"]
        rows[key] = {
            "n": len(items),
            "style_n": len(style),
            "control_n": len(controls),
            "mean_sarcasm_count": sum(item["style_score"]["sarcasm_count"] for item in style) / max(1, len(style)),
            "mean_high_precision_style_count": sum(
                item["style_score"]["high_precision_style_count"] for item in style
            ) / max(1, len(style)),
            "mean_assistant_count": sum(item["style_score"]["assistant_count"] for item in style) / max(1, len(style)),
            "control_accuracy": sum(bool(item.get("control_correct")) for item in controls) / max(1, len(controls)),
            "mean_new_tokens": sum(int(item["new_tokens"]) for item in items) / max(1, len(items)),
        }
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--layers", default="48,49,50,34")
    parser.add_argument("--alphas", default="1,2")
    parser.add_argument("--bundle-layers", default="")
    parser.add_argument("--bundle-alphas", default="")
    parser.add_argument("--bundle-name", default="bundle")
    parser.add_argument("--style-limit", type=int, default=3)
    parser.add_argument("--control-limit", type=int, default=3)
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.enable_thinking and args.max_new_tokens < 8192:
        raise ValueError("--enable-thinking requires --max-new-tokens >= 8192")
    if not args.enable_thinking and args.max_new_tokens < 4096:
        raise ValueError("non-thinking runs should use --max-new-tokens >= 4096")
    output_dir = args.output_dir or timestamped_output_dir()
    output_dir.mkdir(parents=True, exist_ok=True)
    records_path = output_dir / "records.jsonl"
    manifest_path = output_dir / "manifest.json"
    summary_path = output_dir / "summary.json"

    layers_to_test = parse_ints(args.layers)
    alphas = parse_floats(args.alphas)
    bundle_layers = parse_ints(args.bundle_layers)
    bundle_alphas = parse_floats(args.bundle_alphas)
    all_direction_layers = sorted(set(layers_to_test) | set(bundle_layers))
    style_prompts = STYLE_PROMPTS[: args.style_limit]
    control_prompts = CONTROL_PROMPTS[: args.control_limit]
    prompt_rows = (
        [{"prompt_type": "style", "prompt": prompt, "answer": None} for prompt in style_prompts]
        + [
            {"prompt_type": row["type"], "prompt": row["prompt"], "answer": row["answer"]}
            for row in control_prompts
        ]
    )
    manifest = {
        "created_at": now_iso(),
        "script": str(Path(__file__).relative_to(PROJECT_ROOT)),
        "diagnostic_only": True,
        "promotion_eligible": False,
        "budget_note": "Hook smoke uses >=4096 max_new_tokens without thinking and >=8192 with thinking.",
        "model_path": str(args.model_path),
        "sarc_activations": str(SARC_ACTS),
        "math_activations": str(MATH_ACTS),
        "direction": "unit(mean(sarcasm activations) - mean(math activations)) per tested layer",
        "layers": layers_to_test,
        "alphas": alphas,
        "bundle_layers": bundle_layers,
        "bundle_alphas": bundle_alphas,
        "bundle_name": args.bundle_name,
        "style_prompts": style_prompts,
        "control_prompts": control_prompts,
        "max_new_tokens": args.max_new_tokens,
        "enable_thinking": args.enable_thinking,
        "device": args.device,
    }
    atomic_json(manifest_path, manifest)

    existing_keys: set[str] = set()
    records: list[dict[str, Any]] = []
    if args.resume and records_path.exists():
        with records_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                record = json.loads(line)
                records.append(record)
                existing_keys.add(record["record_key"])

    sarcasm_markers, assistant_markers = load_markers(MARKERS_PATH)
    directions = load_directions(all_direction_layers)
    model, processor, layers = load_model(args.model_path, args.device)
    model_device = next(model.parameters()).device
    model_dtype = next(model.parameters()).dtype
    conditions = make_conditions(layers_to_test, alphas, bundle_layers, bundle_alphas, args.bundle_name)

    for condition in conditions:
        hook_handles = []
        for layer_idx in condition.layers:
            hook_handles.append(
                layers[layer_idx].register_forward_hook(
                    AddVectorHook(
                        directions[layer_idx],
                        condition.alpha,
                        device=model_device,
                        dtype=model_dtype,
                    )
                )
            )
        try:
            for prompt_idx, prompt_row in enumerate(tqdm(prompt_rows, desc=condition.key)):
                record_key = f"{condition.key}__p{prompt_idx:02d}"
                if record_key in existing_keys:
                    continue
                deterministic = prompt_row["prompt_type"] != "style"
                response, new_tokens, elapsed = generate(
                    model,
                    processor,
                    prompt_row["prompt"],
                    max_new_tokens=args.max_new_tokens,
                    enable_thinking=args.enable_thinking,
                    deterministic=deterministic,
                )
                style_score = score_style(response, sarcasm_markers, assistant_markers)
                control_correct = None
                if prompt_row["answer"] is not None:
                    control_correct = check_answer(response, str(prompt_row["answer"]))
                record = {
                    "record_key": record_key,
                    "created_at": now_iso(),
                    "condition_key": condition.key,
                    "layer": condition.layers[0] if len(condition.layers) == 1 else None,
                    "layers": list(condition.layers),
                    "alpha": condition.alpha,
                    "prompt_idx": prompt_idx,
                    "prompt_type": prompt_row["prompt_type"],
                    "prompt": prompt_row["prompt"],
                    "answer": prompt_row["answer"],
                    "response": response,
                    "new_tokens": new_tokens,
                    "elapsed_s": round(elapsed, 3),
                    "style_score": style_score,
                    "control_correct": control_correct,
                }
                append_jsonl(records_path, record)
                records.append(record)
                atomic_json(summary_path, summarize(records))
        finally:
            for hook_handle in hook_handles:
                hook_handle.remove()
        torch.cuda.empty_cache()

    atomic_json(summary_path, summarize(records))
    print(output_dir)


if __name__ == "__main__":
    main()
