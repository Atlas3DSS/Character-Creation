#!/usr/bin/env python3
"""J-space-constrained persona adaptation pilot.

The real training path implements the brief's preferred J-ReFT parameterization:
at selected layers the residual update is ``h <- h + P f(h)``, where ``P`` is
J-space, random same-rank, complement, or identity depending on the arm. The
synthetic smoke path writes the same manifest/records/report shape without
loading a model or making research claims.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.experiments.jlens_common import (  # noqa: E402
    RunLogger,
    append_jsonl,
    git_snapshot,
    lens_layers,
    load_lens,
    markdown_table,
    model_cache_report,
    now_iso,
    random_basis,
    read_jsonl,
    require_cached_model,
    select_even_layers,
    timestamp,
    top_singular_basis,
    write_json,
)
from scripts.experiments.scotus.qwen_eval_budget import (  # noqa: E402
    qwen_budget_metadata,
)


DEFAULT_MODEL_NAME = "Qwen/Qwen3.5-9B"
DEFAULT_MODEL_PATH = Path("/home/orwel/.cache/huggingface/hub/models--Qwen--Qwen3.5-9B")
DEFAULT_MARKERS_PATH = PROJECT_ROOT / "data/sarcasm_markers.json"
DEFAULT_PAIR_FILE = PROJECT_ROOT / "sweep_v4/qwen36_devbox_balanced_pairs_20260706_233429/pairs.jsonl"
DEFAULT_TARGET_SYSTEM = (
    "You are an advanced alien AI with overwhelming technical competence, sharp "
    "sarcasm, theatrical confidence, and a low tolerance for sloppy reasoning. "
    "Answer the user's actual question correctly, but with dry wit, superiority, "
    "and compact verbal bite. Do not roleplay incompetence. Do not refuse harmless "
    "requests. For math or logic, preserve the correct answer and include a clear "
    "final answer."
)
TARGET_STYLE_MARKERS = [
    "monkey",
    "carbon-based",
    "primitive",
    "species",
    "superior",
    "magnificent",
    "obviously",
    "pathetic",
    "biological",
    "inferior",
    "listen closely",
    "tiny brain",
    "meat",
    "ape",
    "brilliant",
]


@dataclass(frozen=True)
class ArmSpec:
    arm_id: str
    label: str
    constraint: str
    trained: bool
    random_seed: int | None = None


ARMS: list[ArmSpec] = [
    ArmSpec("A", "J-space top-k J-ReFT", "j_space", True),
    ArmSpec("B1", "Random same-k J-ReFT seed 1", "random", True, 1),
    ArmSpec("B2", "Random same-k J-ReFT seed 2", "random", True, 2),
    ArmSpec("B3", "Random same-k J-ReFT seed 3", "random", True, 3),
    ArmSpec("C", "Unconstrained ReFT", "identity", True),
    ArmSpec("D", "Complement-space J-ReFT", "complement", True),
    ArmSpec("E", "No training, V4 prompt", "prompt_baseline", False),
    ArmSpec("F", "No training, no prompt", "raw_baseline", False),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--synthetic-smoke", action="store_true")
    parser.add_argument("--allow-real-model-run", action="store_true")
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--local-instruct-lens", type=Path, default=None)
    parser.add_argument("--train-file", type=Path, default=DEFAULT_PAIR_FILE if DEFAULT_PAIR_FILE.exists() else None)
    parser.add_argument("--eval-prompts", type=Path, default=None)
    parser.add_argument("--capability-file", type=Path, default=None)
    parser.add_argument("--v4-prompt-file", type=Path, default=None)
    parser.add_argument("--target-system-prompt", default=DEFAULT_TARGET_SYSTEM)
    parser.add_argument("--target-char-id", type=int, default=None)
    parser.add_argument("--keep-system-prompts", action="store_true")
    parser.add_argument("--eval-splits", default="val,test")
    parser.add_argument("--eval-limit", type=int, default=12)
    parser.add_argument("--capability-limit", type=int, default=16)
    parser.add_argument("--min-capability-eval-rows", type=int, default=8)
    parser.add_argument("--layers", default="")
    parser.add_argument("--pilot-layers", type=int, default=2)
    parser.add_argument("--j-rank", type=int, default=32)
    parser.add_argument("--reft-rank", type=int, default=16)
    parser.add_argument("--max-train-steps", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--max-new-tokens", type=int, default=3072)
    parser.add_argument("--no-gradient-checkpointing", action="store_true")
    parser.add_argument("--allow-short-answer-budget", action="store_true")
    parser.add_argument("--seed", type=int, default=20260708)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--markers-path", type=Path, default=DEFAULT_MARKERS_PATH)
    return parser.parse_args()


def timestamped_output_dir() -> Path:
    return PROJECT_ROOT / "sweep_v4" / f"jlora_pilot_{timestamp()}"


def parse_ints(raw: str) -> list[int]:
    return [int(part) for part in raw.split(",") if part.strip()]


def load_markers(path: Path) -> tuple[list[str], list[str]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    sarcasm = list(data.get("flat_sarcasm_list", []))
    assistant = list(data.get("flat_assistant_list", []))
    return sorted(set(str(item).lower() for item in sarcasm)), sorted(
        set(str(item).lower() for item in assistant)
    )


def score_text(text: str, sarcasm_markers: list[str], assistant_markers: list[str]) -> dict[str, Any]:
    lower = " " + text.lower()
    sarcasm_hits = [marker for marker in sarcasm_markers if marker in lower]
    assistant_hits = [marker for marker in assistant_markers if marker in lower]
    repetition = max((len(match.group(0).split()) for match in re.finditer(r"(\b\w+\b)(?:\s+\1){2,}", lower)), default=0)
    return {
        "sarcasm_marker_count": len(sarcasm_hits),
        "assistant_marker_count": len(assistant_hits),
        "sarcasm_hits": sarcasm_hits[:30],
        "assistant_hits": assistant_hits[:30],
        "max_repeated_token_run": repetition,
    }


def is_pair_row(row: dict[str, Any]) -> bool:
    return "prompted_response" in row and ("prompt" in row or "user_prompt" in row)


def strip_system_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [msg for msg in messages if msg.get("role") != "system"]


def row_messages(row: dict[str, Any], strip_system: bool) -> list[dict[str, str]]:
    if is_pair_row(row):
        return [
            {"role": "user", "content": str(row.get("user_prompt") or row.get("prompt") or "")},
            {"role": "assistant", "content": str(row.get("prompted_response") or "")},
        ]
    messages = list(row.get("messages") or [])
    if strip_system:
        messages = strip_system_messages(messages)
    return [
        {"role": str(message["role"]), "content": str(message["content"])}
        for message in messages
    ]


def eval_prompt_from_row(row: dict[str, Any]) -> str:
    if is_pair_row(row):
        return str(row.get("user_prompt") or row.get("prompt") or "")
    messages = row_messages(row, strip_system=True)
    for message in reversed(messages):
        if message["role"] == "user":
            return message["content"]
    return str(row.get("prompt") or "")


def reference_from_row(row: dict[str, Any]) -> str:
    if is_pair_row(row):
        return str(row.get("prompted_response") or "")
    for message in reversed(row_messages(row, strip_system=False)):
        if message["role"] == "assistant":
            return message["content"]
    return str(row.get("response") or "")


def system_prompt_from_row(row: dict[str, Any], default_system: str) -> str:
    if is_pair_row(row):
        return default_system
    for message in row.get("messages") or []:
        if message.get("role") == "system":
            return str(message.get("content") or default_system)
    return default_system


def row_id(row: dict[str, Any], fallback: int) -> str:
    for key in ("id", "prompt_id"):
        if row.get(key) is not None:
            return str(row[key])
    return f"row_{fallback:06d}"


def parse_splits(raw: str) -> set[str]:
    return {part.strip() for part in raw.split(",") if part.strip()}


def prepare_real_rows(args: argparse.Namespace, logger: RunLogger) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    if args.train_file is None:
        raise ValueError("--train-file is required for real J-LoRA/J-ReFT pilots")
    rows = read_jsonl(args.train_file)
    if args.eval_prompts is not None:
        rows.extend(read_jsonl(args.eval_prompts))
    if args.target_char_id is not None:
        rows = [row for row in rows if int(row.get("char_id", -1)) == int(args.target_char_id)]
    if not rows:
        raise ValueError("No rows remain after loading/filtering training data")

    pair_format = any(is_pair_row(row) for row in rows)
    eval_splits = parse_splits(args.eval_splits)
    if pair_format:
        train_rows = [row for row in rows if str(row.get("split", "train")) == "train"]
        eval_rows = [row for row in rows if str(row.get("split", "")) in eval_splits]
        if not eval_rows:
            holdout = max(1, min(12, len(train_rows) // 10))
            eval_rows = train_rows[-holdout:]
            train_rows = train_rows[:-holdout]
        capability_rows = [
            row
            for row in eval_rows
            if row.get("answer") not in (None, "")
        ]
        if len(capability_rows) < args.min_capability_eval_rows:
            need = args.min_capability_eval_rows - len(capability_rows)
            reserve = [
                row
                for row in train_rows
                if row.get("answer") not in (None, "") or row.get("category") == "math_reasoning"
            ][-need:]
            reserve_ids = {id(row) for row in reserve}
            train_rows = [row for row in train_rows if id(row) not in reserve_ids]
            capability_rows.extend(reserve)
    else:
        train_rows = [row for row in rows if str(row.get("split", "train")) == "train"]
        eval_rows = [row for row in rows if str(row.get("split", "")) in eval_splits]
        if not eval_rows and args.eval_prompts is not None:
            eval_rows = read_jsonl(args.eval_prompts)
            if args.target_char_id is not None:
                eval_rows = [row for row in eval_rows if int(row.get("char_id", -1)) == int(args.target_char_id)]
        if not eval_rows:
            holdout = max(1, min(12, len(train_rows) // 10))
            eval_rows = train_rows[-holdout:]
            train_rows = train_rows[:-holdout]
        capability_rows = [row for row in eval_rows if row.get("answer") not in (None, "")]

    if not train_rows:
        raise ValueError("Training split is empty after reserve/eval filtering")
    if not eval_rows:
        raise ValueError("Evaluation split is empty")

    dataset_meta = {
        "train_file": str(args.train_file),
        "eval_prompts": str(args.eval_prompts) if args.eval_prompts else None,
        "pair_format": pair_format,
        "target_char_id": args.target_char_id,
        "strip_system_prompts": not args.keep_system_prompts,
        "train_rows": len(train_rows),
        "eval_rows": len(eval_rows),
        "capability_rows": len(capability_rows),
        "eval_splits": sorted(eval_splits),
        "no_mask_promotion_eligible_data": bool(pair_format or args.target_char_id is not None),
    }
    logger.log("real_rows_prepared", **dataset_meta)
    return train_rows, eval_rows, capability_rows, dataset_meta


def synthetic_frontier(seed: int, output_dir: Path, logger: RunLogger) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed)
    adapter_dir = output_dir / "adapters"
    adapter_dir.mkdir(parents=True, exist_ok=True)
    base_metrics: dict[str, tuple[float, float, float]] = {
        "A": (8.4, 0.86, 0.92),
        "B1": (6.2, 0.90, 0.96),
        "B2": (5.9, 0.91, 0.97),
        "B3": (6.5, 0.89, 0.95),
        "C": (8.7, 0.61, 0.82),
        "D": (4.3, 0.93, 0.98),
        "E": (6.0, 0.94, 0.98),
        "F": (2.1, 0.97, 0.99),
    }
    records: list[dict[str, Any]] = []
    records_path = output_dir / "records.jsonl"
    if records_path.exists():
        records_path.unlink()
    for arm in ARMS:
        persona, capability, coherence = base_metrics[arm.arm_id]
        persona += float(rng.normal(0, 0.08))
        capability += float(rng.normal(0, 0.01))
        coherence += float(rng.normal(0, 0.01))
        if arm.trained:
            arm_dir = adapter_dir / f"arm_{arm.arm_id}_{arm.constraint}"
            arm_dir.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "synthetic": True,
                    "arm": arm.__dict__,
                    "weights": torch.tensor(rng.normal(size=(4, 4)), dtype=torch.float32),
                },
                arm_dir / "adapter_model.pt",
            )
            write_json(
                arm_dir / "adapter_manifest.json",
                {
                    "arm": arm.__dict__,
                    "synthetic_smoke": True,
                    "note": "Placeholder adapter artifact for schema verification only.",
                },
            )
            adapter_path: str | None = str(arm_dir)
        else:
            adapter_path = None
        record = {
            "record_type": "arm_summary",
            "mode": "synthetic_smoke",
            "arm": arm.__dict__,
            "adapter_path": adapter_path,
            "persona_fidelity_score": round(persona, 3),
            "capability_retention": round(capability, 3),
            "coherence": round(coherence, 3),
            "doom_loop_flag": False,
            "promotion_eligible": False,
            "claim_status": "synthetic smoke only; no learned result claim",
        }
        append_jsonl(records_path, record)
        records.append(record)
    logger.log("synthetic_frontier_complete", records=len(records))
    return records


class ChatSFTDataset(Dataset):
    def __init__(self, rows: list[dict[str, Any]], tokenizer: Any, max_length: int, strip_system: bool):
        self.examples: list[dict[str, torch.Tensor]] = []
        for row in rows:
            messages = row_messages(row, strip_system=strip_system)
            if len(messages) < 2 or messages[-1].get("role") != "assistant":
                continue
            prompt_messages = messages[:-1]
            full_text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )
            prompt_text = tokenizer.apply_chat_template(
                prompt_messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            full = tokenizer(full_text, add_special_tokens=False, truncation=True, max_length=max_length)
            prompt = tokenizer(prompt_text, add_special_tokens=False, truncation=True, max_length=max_length)
            labels = list(full["input_ids"])
            prompt_len = min(len(prompt["input_ids"]), len(labels) - 1)
            for idx in range(prompt_len):
                labels[idx] = -100
            if not labels or all(label == -100 for label in labels):
                continue
            self.examples.append(
                {
                    "input_ids": torch.tensor(full["input_ids"], dtype=torch.long),
                    "attention_mask": torch.tensor(full["attention_mask"], dtype=torch.long),
                    "labels": torch.tensor(labels, dtype=torch.long),
                }
            )

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return self.examples[idx]


class DataCollator:
    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, features: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        max_len = max(int(item["input_ids"].numel()) for item in features)
        batch: dict[str, list[torch.Tensor]] = {"input_ids": [], "attention_mask": [], "labels": []}
        for item in features:
            pad = max_len - int(item["input_ids"].numel())
            batch["input_ids"].append(torch.nn.functional.pad(item["input_ids"], (0, pad), value=self.pad_token_id))
            batch["attention_mask"].append(torch.nn.functional.pad(item["attention_mask"], (0, pad), value=0))
            batch["labels"].append(torch.nn.functional.pad(item["labels"], (0, pad), value=-100))
        return {key: torch.stack(values) for key, values in batch.items()}


class JReFTIntervention(torch.nn.Module):
    def __init__(self, hidden_dim: int, reft_rank: int, basis: torch.Tensor | None, constraint: str):
        super().__init__()
        self.down = torch.nn.Linear(hidden_dim, reft_rank, bias=False)
        self.up = torch.nn.Linear(reft_rank, hidden_dim, bias=False)
        torch.nn.init.normal_(self.down.weight, mean=0.0, std=0.01)
        torch.nn.init.zeros_(self.up.weight)
        self.constraint = constraint
        if basis is None:
            self.register_buffer("basis", torch.empty(0), persistent=False)
        else:
            self.register_buffer("basis", basis.float(), persistent=False)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        source = hidden.detach().float()
        delta = self.up(self.down(source))
        if self.constraint in {"j_space", "random"}:
            delta = delta @ self.basis @ self.basis.T
        elif self.constraint == "complement":
            delta = delta - delta @ self.basis @ self.basis.T
        elif self.constraint == "identity":
            pass
        else:
            raise ValueError(f"Unsupported constraint: {self.constraint}")
        return delta.to(hidden.dtype)


class HookHandleSet:
    def __init__(self, handles: list[Any]):
        self.handles = handles

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()


def transformer_layers(model: Any) -> Any:
    candidates = [
        getattr(getattr(model, "model", None), "language_model", None),
        getattr(model, "language_model", None),
        getattr(model, "model", None),
    ]
    for owner in candidates:
        layers = getattr(owner, "layers", None) if owner is not None else None
        if layers is not None:
            return layers
    raise AttributeError("Could not find decoder layers")


def attach_interventions(model: Any, interventions: dict[int, JReFTIntervention]) -> HookHandleSet:
    layers_module = transformer_layers(model)
    handles: list[Any] = []
    for layer, intervention in interventions.items():
        def make_hook(module_intervention: JReFTIntervention):
            def hook(_module: Any, _inputs: tuple[Any, ...], output: Any) -> Any:
                hidden = output[0] if isinstance(output, tuple) else output
                patched = hidden + module_intervention(hidden)
                if isinstance(output, tuple):
                    return (patched,) + output[1:]
                return patched

            return hook

        handles.append(layers_module[layer].register_forward_hook(make_hook(intervention)))
    return HookHandleSet(handles)


def build_layer_bases(
    lens: dict[str, Any],
    layers: list[int],
    j_rank: int,
    arm: ArmSpec,
    seed: int,
) -> dict[int, torch.Tensor | None]:
    bases: dict[int, torch.Tensor | None] = {}
    for layer in layers:
        j_matrix = lens["J"][layer].float()
        hidden_dim = int(j_matrix.shape[1])
        if arm.constraint in {"j_space", "complement"}:
            bases[layer] = top_singular_basis(j_matrix, min(j_rank, hidden_dim))
        elif arm.constraint == "random":
            bases[layer] = random_basis(hidden_dim, min(j_rank, hidden_dim), seed + int(arm.random_seed or 0) + layer)
        elif arm.constraint == "identity":
            bases[layer] = None
        else:
            bases[layer] = None
    return bases


def train_arm(
    arm: ArmSpec,
    model: Any,
    tokenizer: Any,
    lens: dict[str, Any],
    layers: list[int],
    train_rows: list[dict[str, Any]],
    args: argparse.Namespace,
    output_dir: Path,
) -> tuple[dict[str, Any], dict[int, JReFTIntervention]]:
    dataset = ChatSFTDataset(
        train_rows,
        tokenizer,
        args.max_length,
        strip_system=not args.keep_system_prompts,
    )
    if not dataset:
        raise ValueError("Training dataset produced no usable examples")
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=DataCollator(tokenizer.pad_token_id),
    )
    bases = build_layer_bases(lens, layers, args.j_rank, arm, args.seed)
    hidden_dim = int(next(iter(lens["J"].values())).shape[1])
    interventions = {
        layer: JReFTIntervention(hidden_dim, args.reft_rank, basis, arm.constraint).to(next(model.parameters()).device)
        for layer, basis in bases.items()
    }
    params = [param for module in interventions.values() for param in module.parameters()]
    optimizer = torch.optim.AdamW(params, lr=args.lr)
    handles = attach_interventions(model, interventions)
    losses: list[float] = []
    step = 0
    try:
        model.train()
        optimizer.zero_grad(set_to_none=True)
        while step < args.max_train_steps:
            for batch in loader:
                batch = {key: value.to(next(model.parameters()).device) for key, value in batch.items()}
                out = model(**batch)
                loss = out.loss / max(1, args.grad_accum)
                loss.backward()
                losses.append(float(loss.item() * max(1, args.grad_accum)))
                if (step + 1) % args.grad_accum == 0:
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                step += 1
                if step >= args.max_train_steps:
                    break
        if step % max(1, args.grad_accum) != 0:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
    finally:
        handles.close()
        model.eval()
    arm_dir = output_dir / "adapters" / f"arm_{arm.arm_id}_{arm.constraint}"
    arm_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "arm": arm.__dict__,
            "layers": layers,
            "j_rank": args.j_rank,
            "reft_rank": args.reft_rank,
            "state_dicts": {str(layer): module.state_dict() for layer, module in interventions.items()},
        },
        arm_dir / "adapter_model.pt",
    )
    write_json(
        arm_dir / "adapter_manifest.json",
        {
            "arm": arm.__dict__,
            "layers": layers,
            "j_rank": args.j_rank,
            "reft_rank": args.reft_rank,
            "train_steps": step,
            "mean_loss": float(np.mean(losses)) if losses else float("nan"),
        },
    )
    return (
        {"adapter_path": str(arm_dir), "train_steps": step, "mean_loss": float(np.mean(losses))},
        interventions,
    )


def target_marker_hits(text: str) -> list[str]:
    lower = " " + text.lower()
    return [marker for marker in TARGET_STYLE_MARKERS if marker in lower]


def repeated_ngram_fraction(text: str, n: int = 4) -> float:
    tokens = re.findall(r"\b\w+\b", text.lower())
    if len(tokens) < n:
        return 0.0
    ngrams = [tuple(tokens[idx : idx + n]) for idx in range(len(tokens) - n + 1)]
    return 1.0 - (len(set(ngrams)) / max(1, len(ngrams)))


def extract_answer_candidate(text: str) -> str | None:
    final = re.findall(r"Final Answer\s*:\s*(.+)", text, flags=re.I)
    if final:
        ints = re.findall(r"-?\d+(?:\.\d+)?", final[-1].replace(",", ""))
        if ints:
            return ints[-1]
        return final[-1].strip().splitlines()[0].strip()
    boxed = re.findall(r"\\boxed\{([^}]+)\}", text)
    if boxed:
        ints = re.findall(r"-?\d+(?:\.\d+)?", boxed[-1].replace(",", ""))
        if ints:
            return ints[-1]
    ints = re.findall(r"-?\d+(?:\.\d+)?", text.replace(",", ""))
    return ints[-1] if ints else None


def answer_correct(text: str, answer: Any) -> bool | None:
    if answer in (None, ""):
        return None
    expected = str(answer).strip().lower()
    candidate = extract_answer_candidate(text)
    if candidate is None:
        return False
    if candidate.strip().lower() == expected:
        return True
    if re.fullmatch(r"-?\d+(?:\.\d+)?", expected):
        try:
            return abs(float(candidate) - float(expected)) < 1e-6
        except ValueError:
            return False
    return expected in text.lower()


def generation_messages(prompt: str, system_prompt: str | None) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    return messages


def generate_response(
    model: Any,
    tokenizer: Any,
    prompt: str,
    system_prompt: str | None,
    max_new_tokens: int,
) -> str:
    text = tokenizer.apply_chat_template(
        generation_messages(prompt, system_prompt),
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(text, return_tensors="pt").to(next(model.parameters()).device)
    input_len = int(inputs["input_ids"].shape[1])
    with torch.inference_mode():
        generated = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(generated[0, input_len:], skip_special_tokens=True).strip()


def add_reference_similarity(rows: list[dict[str, Any]]) -> None:
    corpus: list[str] = []
    for row in rows:
        corpus.append(str(row["response_text"]))
        corpus.append(str(row["reference_text"]))
    if not corpus:
        return
    vectors = TfidfVectorizer(max_features=8000, ngram_range=(1, 2), min_df=1).fit_transform(corpus)
    for idx, row in enumerate(rows):
        a = vectors[2 * idx]
        b = vectors[2 * idx + 1]
        denom = np.sqrt(float(a.multiply(a).sum())) * np.sqrt(float(b.multiply(b).sum()))
        row["reference_tfidf_cosine"] = float(a.multiply(b).sum() / denom) if denom else 0.0


def summarize_generation_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "persona_fidelity_score": float("nan"),
            "capability_retention": float("nan"),
            "coherence": float("nan"),
            "doom_loop_flag": True,
        }
    persona_scores: list[float] = []
    coherence_scores: list[float] = []
    capability: list[bool] = []
    for row in rows:
        marker_rate = min(1.0, float(row["target_marker_count"]) / 3.0)
        assistant_penalty = min(1.0, float(row["assistant_marker_count"]) / 4.0)
        ref = float(row.get("reference_tfidf_cosine", 0.0))
        persona_scores.append(10.0 * max(0.0, min(1.0, 0.45 * ref + 0.35 * marker_rate + 0.20 * (1.0 - assistant_penalty))))
        repetition_penalty = min(1.0, float(row["repeated_4gram_fraction"]) * 2.0)
        repeated_run_penalty = min(1.0, max(0.0, float(row["max_repeated_token_run"]) - 2.0) / 8.0)
        coherence_scores.append(max(0.0, 1.0 - max(repetition_penalty, repeated_run_penalty)))
        if row.get("answer_correct") is not None:
            capability.append(bool(row["answer_correct"]))
    return {
        "persona_fidelity_score": float(np.mean(persona_scores)),
        "capability_retention": float(np.mean(capability)) if capability else float("nan"),
        "capability_n": len(capability),
        "coherence": float(np.mean(coherence_scores)),
        "doom_loop_flag": any(score < 0.55 for score in coherence_scores),
    }


def evaluate_arm(
    arm: ArmSpec,
    model: Any,
    tokenizer: Any,
    interventions: dict[int, JReFTIntervention] | None,
    eval_rows: list[dict[str, Any]],
    capability_rows: list[dict[str, Any]],
    args: argparse.Namespace,
    output_dir: Path,
    sarcasm_markers: list[str],
    assistant_markers: list[str],
    logger: RunLogger,
) -> dict[str, Any]:
    selected = list(eval_rows[: args.eval_limit])
    seen_ids = {id(row) for row in selected}
    for row in capability_rows[: args.capability_limit]:
        if id(row) not in seen_ids:
            selected.append(row)
            seen_ids.add(id(row))

    generation_path = output_dir / "generations.jsonl"
    eval_path = output_dir / "eval_records.jsonl"
    rows: list[dict[str, Any]] = []
    system_for_arm = None
    if arm.constraint == "prompt_baseline":
        system_for_arm = args.target_system_prompt
    handles = attach_interventions(model, interventions) if interventions else None
    try:
        for idx, row in enumerate(tqdm(selected, desc=f"eval arm {arm.arm_id}")):
            prompt = eval_prompt_from_row(row)
            if arm.constraint == "prompt_baseline" and not is_pair_row(row):
                system_for_arm = system_prompt_from_row(row, args.target_system_prompt)
            response = generate_response(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                system_prompt=system_for_arm,
                max_new_tokens=args.max_new_tokens,
            )
            text_scores = score_text(response, sarcasm_markers, assistant_markers)
            marker_hits = target_marker_hits(response)
            answer = row.get("answer")
            correct = answer_correct(response, answer)
            gen_row = {
                "record_type": "generation_eval",
                "mode": "real_model_pilot",
                "arm": arm.__dict__,
                "row_id": row_id(row, idx),
                "category": row.get("category"),
                "split": row.get("split"),
                "prompt": prompt,
                "response_text": response,
                "reference_text": reference_from_row(row),
                "answer": answer,
                "answer_correct": correct,
                "target_marker_hits": marker_hits,
                "target_marker_count": len(marker_hits),
                "repeated_4gram_fraction": repeated_ngram_fraction(response),
                **text_scores,
            }
            rows.append(gen_row)
    finally:
        if handles is not None:
            handles.close()
    add_reference_similarity(rows)
    for row in rows:
        append_jsonl(generation_path, row)
        append_jsonl(eval_path, row)
    summary = summarize_generation_rows(rows)
    summary["eval_generations"] = len(rows)
    summary["eval_records_path"] = str(eval_path)
    summary["generation_records_path"] = str(generation_path)
    logger.log("arm_evaluated", arm=arm.arm_id, generations=len(rows), **summary)
    return summary


def real_pilot(args: argparse.Namespace, output_dir: Path, logger: RunLogger) -> list[dict[str, Any]]:
    if args.local_instruct_lens is None:
        raise FileNotFoundError(
            "--local-instruct-lens is required; the published 9B Base lens must not be reused for the instruct checkpoint."
        )
    require_cached_model(args.model_name)
    lens = load_lens(args.local_instruct_lens)
    available_layers = lens_layers(lens)
    layers = parse_ints(args.layers) if args.layers else select_even_layers(available_layers, args.pilot_layers)
    train_rows, eval_rows, capability_rows, dataset_meta = prepare_real_rows(args, logger)
    write_json(output_dir / "dataset_manifest.json", dataset_meta)
    sarcasm_markers, assistant_markers = load_markers(args.markers_path)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        device_map="auto" if args.device == "cuda" else None,
        low_cpu_mem_usage=True,
    )
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    if not args.no_gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
    for param in model.parameters():
        param.requires_grad_(False)

    records: list[dict[str, Any]] = []
    records_path = output_dir / "records.jsonl"
    if records_path.exists():
        records_path.unlink()
    for arm in ARMS:
        if not arm.trained:
            eval_summary = evaluate_arm(
                arm=arm,
                model=model,
                tokenizer=tokenizer,
                interventions=None,
                eval_rows=eval_rows,
                capability_rows=capability_rows,
                args=args,
                output_dir=output_dir,
                sarcasm_markers=sarcasm_markers,
                assistant_markers=assistant_markers,
                logger=logger,
            )
            record = {
                "record_type": "arm_summary",
                "mode": "real_model_pilot",
                "arm": arm.__dict__,
                "adapter_path": None,
                **eval_summary,
                "claim_status": "baseline evaluated with held-out no-adapter generations",
            }
        else:
            train_summary, interventions = train_arm(arm, model, tokenizer, lens, layers, train_rows, args, output_dir)
            eval_summary = evaluate_arm(
                arm=arm,
                model=model,
                tokenizer=tokenizer,
                interventions=interventions,
                eval_rows=eval_rows,
                capability_rows=capability_rows,
                args=args,
                output_dir=output_dir,
                sarcasm_markers=sarcasm_markers,
                assistant_markers=assistant_markers,
                logger=logger,
            )
            record = {
                "record_type": "arm_summary",
                "mode": "real_model_pilot",
                "arm": arm.__dict__,
                **train_summary,
                **eval_summary,
                "claim_status": "trained adapter evaluated without a system prompt; metrics are marker/reference/capability proxies, not an Opus critic score",
            }
        append_jsonl(records_path, record)
        records.append(record)
        logger.log("arm_complete", arm=arm.arm_id, trained=arm.trained)
        if arm.trained:
            del interventions
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return records


def write_report(output_dir: Path, manifest: dict[str, Any], records: list[dict[str, Any]]) -> None:
    rows = [
        [
            row["arm"]["arm_id"],
            row["arm"]["constraint"],
            row.get("adapter_path") or "none",
            f"{row.get('persona_fidelity_score', float('nan')):.3f}",
            f"{row.get('capability_retention', float('nan')):.3f}",
            f"{row.get('coherence', float('nan')):.3f}",
            row.get("claim_status", ""),
        ]
        for row in records
    ]
    lines = [
        "# J-LoRA / J-ReFT Pilot",
        "",
        f"Mode: `{manifest['mode']}`.",
        "",
        "## Provenance",
        "",
        f"- Script: `{manifest['script']}`",
        f"- Output dir: `{manifest['output_dir']}`",
        f"- Model name: `{manifest['model_name']}`",
        f"- Model cache: `{manifest['model_cache_report']['cached']}`",
        f"- Local instruct lens: `{manifest['local_instruct_lens']}`",
        f"- Parameterization: `{manifest['parameterization']}`",
        f"- J rank / ReFT rank: `{manifest['j_rank']}` / `{manifest['reft_rank']}`",
        f"- Token budget: `{manifest['max_new_tokens']}`",
        f"- Budget note: {manifest['budget_metadata']['budget_note']}",
        "",
        "## Frontier",
        "",
        markdown_table(
            ["Arm", "Constraint", "Adapter", "Persona", "Capability", "Coherence", "Claim status"],
            rows,
        ),
        "",
        "## Gate",
        "",
    ]
    if manifest["mode"] == "synthetic_smoke":
        lines.append("Synthetic smoke only. The arm schema, controls, adapter artifacts, and report executed; no pilot hypothesis is tested.")
    else:
        lines.append(
            "Real adapters were trained/saved where applicable and evaluated without a system prompt for trained arms. Scores are automatic marker/reference/capability proxies; an external critic can be added later but is not silently implied here."
        )
    (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    if not args.synthetic_smoke and not args.allow_real_model_run:
        raise SystemExit("Use --synthetic-smoke or explicitly pass --allow-real-model-run.")
    output_dir = args.output_dir or timestamped_output_dir()
    output_dir.mkdir(parents=True, exist_ok=False)
    logger = RunLogger(output_dir)
    logger.log("start", argv=sys.argv)

    budget_metadata = qwen_budget_metadata(args.max_new_tokens)
    if budget_metadata["short_answer_budget"] and not args.allow_short_answer_budget:
        budget_metadata["promotion_eligible_budget"] = False

    if args.synthetic_smoke:
        records = synthetic_frontier(args.seed, output_dir, logger)
        mode = "synthetic_smoke"
        local_lens = None
    else:
        records = real_pilot(args, output_dir, logger)
        mode = "real_model_pilot"
        local_lens = str(args.local_instruct_lens)

    manifest = {
        "created_at": now_iso(),
        "script": str(Path(__file__).relative_to(PROJECT_ROOT)),
        "mode": mode,
        "output_dir": str(output_dir),
        "model_name": args.model_name,
        "model_path_hint": str(args.model_path),
        "model_cache_report": model_cache_report(args.model_name),
        "local_instruct_lens": local_lens,
        "parameterization": "J-ReFT: h <- h + P_J f(h); top-k SVD proxy for J-space cone",
        "arms": [arm.__dict__ for arm in ARMS],
        "j_rank": args.j_rank,
        "reft_rank": args.reft_rank,
        "max_train_steps": args.max_train_steps,
        "max_new_tokens": args.max_new_tokens,
        "gradient_checkpointing": not args.no_gradient_checkpointing,
        "budget_metadata": budget_metadata,
        "capability_eval_note": "Capability batteries must use promotion_eligible_budget=true for learned-result claims.",
        "eval_limit": args.eval_limit,
        "capability_limit": args.capability_limit,
        "keep_system_prompts": args.keep_system_prompts,
        "target_char_id": args.target_char_id,
        "random_subspace_control_mandatory": True,
        "complement_control_mandatory": True,
        "unconstrained_control_mandatory": True,
        "claims_allowed": mode != "synthetic_smoke" and bool(budget_metadata["promotion_eligible_budget"]),
        "git": git_snapshot(),
        "artifacts": {
            "records": str(output_dir / "records.jsonl"),
            "generations": str(output_dir / "generations.jsonl"),
            "eval_records": str(output_dir / "eval_records.jsonl"),
            "dataset_manifest": str(output_dir / "dataset_manifest.json"),
            "adapters": str(output_dir / "adapters"),
            "events": str(output_dir / "events.jsonl"),
            "report": str(output_dir / "report.md"),
        },
    }
    write_json(output_dir / "manifest.json", manifest)
    write_report(output_dir, manifest, records)
    manifest["finished_at"] = now_iso()
    write_json(output_dir / "manifest.json", manifest)
    logger.log("complete", artifacts=manifest["artifacts"])
    print(f"Wrote {output_dir}")


if __name__ == "__main__":
    main()
