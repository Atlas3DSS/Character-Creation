#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)
from transformers import modeling_utils as hf_modeling_utils


DEFAULT_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


class ChatSFTDataset(Dataset):
    def __init__(self, rows: list[dict[str, Any]], tokenizer, max_length: int):
        self.examples: list[dict[str, Any]] = []
        self.tokenizer = tokenizer
        self.max_length = max_length

        for row in rows:
            messages = row["messages"]
            prompt_messages = messages[:-1]
            full_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            prompt_text = tokenizer.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)
            full_tokens = tokenizer(full_text, add_special_tokens=False, truncation=True, max_length=max_length)
            prompt_tokens = tokenizer(prompt_text, add_special_tokens=False, truncation=True, max_length=max_length)

            input_ids = full_tokens["input_ids"]
            if len(input_ids) < 8:
                continue
            prompt_len = min(len(prompt_tokens["input_ids"]), len(input_ids) - 1)
            labels = list(input_ids)
            for i in range(prompt_len):
                labels[i] = -100
            if all(x == -100 for x in labels):
                continue

            self.examples.append(
                {
                    "input_ids": input_ids,
                    "attention_mask": full_tokens["attention_mask"],
                    "labels": labels,
                    "meta": {
                        "char_name": row.get("char_name"),
                        "prompt_id": row.get("prompt_id"),
                        "track": row.get("track"),
                    },
                }
            )

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self.examples[idx]


@dataclass
class DataCollatorForChatSFT:
    tokenizer: Any

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        max_len = max(len(f["input_ids"]) for f in features)
        input_ids = []
        attention_mask = []
        labels = []
        for f in features:
            pad = max_len - len(f["input_ids"])
            input_ids.append(f["input_ids"] + [self.tokenizer.pad_token_id] * pad)
            attention_mask.append(f["attention_mask"] + [0] * pad)
            labels.append(f["labels"] + [-100] * pad)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def render_generation_text(tokenizer, prompt_messages: list[dict[str, str]]) -> str:
    return tokenizer.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)


def extract_assistant_text(tokenizer, full_ids: torch.Tensor, prompt_len: int) -> str:
    gen_ids = full_ids[prompt_len:]
    return tokenizer.decode(gen_ids, skip_special_tokens=False).strip()


class SampleGenerationCallback(TrainerCallback):
    def __init__(
        self,
        tokenizer,
        sample_rows: list[dict[str, Any]],
        sample_dir: Path,
        every_steps: int,
        max_new_tokens: int,
    ):
        self.tokenizer = tokenizer
        self.sample_rows = sample_rows
        self.sample_dir = sample_dir
        self.every_steps = every_steps
        self.max_new_tokens = max_new_tokens
        self.sample_dir.mkdir(parents=True, exist_ok=True)

    def _run_samples(self, trainer: Trainer, step: int) -> None:
        model = trainer.model
        device = next(model.parameters()).device
        model.eval()
        records = []
        for row in self.sample_rows:
            prompt_messages = row["messages"][:-1]
            prompt_text = render_generation_text(self.tokenizer, prompt_messages)
            inputs = self.tokenizer(prompt_text, add_special_tokens=False, return_tensors="pt").to(device)
            with torch.no_grad():
                out = model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                    use_cache=True,
                    eos_token_id=self.tokenizer.eos_token_id,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
            generated = extract_assistant_text(self.tokenizer, out[0], inputs["input_ids"].shape[-1])
            records.append(
                {
                    "step": step,
                    "char_name": row.get("char_name"),
                    "prompt_id": row.get("prompt_id"),
                    "track": row.get("track"),
                    "prompt_messages": prompt_messages,
                    "reference": row["messages"][-1]["content"],
                    "generated": generated,
                }
            )
        out_path = self.sample_dir / f"samples_step_{step:05d}.jsonl"
        with out_path.open("w", encoding="utf-8") as fh:
            for rec in records:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        model.train()

    def on_step_end(self, args, state, control, **kwargs):
        if self.every_steps > 0 and state.global_step > 0 and state.global_step % self.every_steps == 0:
            trainer = kwargs["model"].trainer if hasattr(kwargs["model"], "trainer") else None
            if trainer is not None:
                self._run_samples(trainer, state.global_step)
        return control

    def on_train_end(self, args, state, control, **kwargs):
        trainer = kwargs["model"].trainer if hasattr(kwargs["model"], "trainer") else None
        if trainer is not None:
            self._run_samples(trainer, state.global_step or 0)
        return control


class BoundTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model.trainer = self


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a Qwen3.5-9B LoRA student on paired trace/personality SFT data.")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--train-file", type=Path, required=True)
    parser.add_argument("--val-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-length", type=int, default=1536)
    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=16)
    parser.add_argument("--eval-steps", type=int, default=25)
    parser.add_argument("--save-steps", type=int, default=25)
    parser.add_argument("--logging-steps", type=int, default=5)
    parser.add_argument("--sample-steps", type=int, default=25)
    parser.add_argument("--sample-count", type=int, default=6)
    parser.add_argument("--sample-max-new-tokens", type=int, default=280)
    parser.add_argument("--lora-r", type=int, default=32)
    parser.add_argument("--lora-alpha", type=int, default=64)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--target-modules", nargs="*", default=DEFAULT_TARGET_MODULES)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    sample_dir = args.output_dir / "sample_generations"
    manifest_path = args.output_dir / "run_manifest.json"

    # Qwen3.5-9B is text-only here. Stay on the causal LM path to keep the
    # load footprint as small and predictable as possible on 24 GB cards.
    os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
    os.environ.setdefault("HF_ENABLE_PARALLEL_LOADING", "false")
    os.environ.setdefault("HF_PARALLEL_LOADING_WORKERS", "1")
    hf_modeling_utils.caching_allocator_warmup = lambda *args, **kwargs: None

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_rows = load_jsonl(args.train_file)
    val_rows = load_jsonl(args.val_file)
    train_dataset = ChatSFTDataset(train_rows, tokenizer, args.max_length)
    val_dataset = ChatSFTDataset(val_rows, tokenizer, args.max_length)

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    offload_dir = args.output_dir / "offload"
    offload_dir.mkdir(parents=True, exist_ok=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        quantization_config=quant_config,
        device_map="auto",
        max_memory={0: "20GiB", "cpu": "96GiB"},
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        offload_state_dict=True,
        offload_folder=str(offload_dir),
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=args.target_modules,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    effective_batch = args.batch_size * args.grad_accum
    total_train_steps = math.ceil(max(1, len(train_dataset)) / max(1, effective_batch)) * args.epochs

    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        overwrite_output_dir=True,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        bf16=True,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        eval_strategy="steps",
        save_strategy="steps",
        save_total_limit=2,
        load_best_model_at_end=False,
        report_to=[],
        remove_unused_columns=False,
        dataloader_num_workers=2,
        gradient_checkpointing=True,
        max_grad_norm=1.0,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
    )

    sample_rows = val_rows[: args.sample_count] if val_rows else train_rows[: args.sample_count]
    callbacks = [
        SampleGenerationCallback(
            tokenizer=tokenizer,
            sample_rows=sample_rows,
            sample_dir=sample_dir,
            every_steps=args.sample_steps,
            max_new_tokens=args.sample_max_new_tokens,
        )
    ]

    trainer = BoundTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset if len(val_dataset) else None,
        data_collator=DataCollatorForChatSFT(tokenizer=tokenizer),
        callbacks=callbacks,
    )

    manifest = {
        "generated_at": __import__("datetime").datetime.now().isoformat(),
        "model_path": args.model_path,
        "train_file": str(args.train_file),
        "val_file": str(args.val_file),
        "output_dir": str(args.output_dir),
        "train_examples": len(train_dataset),
        "val_examples": len(val_dataset),
        "max_length": args.max_length,
        "epochs": args.epochs,
        "lr": args.lr,
        "effective_batch_size": effective_batch,
        "estimated_total_train_steps": total_train_steps,
        "target_modules": args.target_modules,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    trainer.train()
    trainer.save_model(str(args.output_dir / "final_adapter"))
    tokenizer.save_pretrained(str(args.output_dir / "final_adapter"))


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
