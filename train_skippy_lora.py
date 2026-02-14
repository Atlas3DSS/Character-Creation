#!/usr/bin/env python3
"""
Train a LoRA adapter on Skippy dialogue from ExForce books.

The LoRA learns the authentic Skippy personality from book dialogue,
formatted as proper chat conversations. After training, we compare
activations between base model and LoRA model to extract personality
vectors for ablation.

Pipeline:
1. Load extracted pairs from skippy_pairs.json
2. Format as chat conversations (system + user + assistant)
3. Train LoRA with SFTTrainer
4. Save LoRA adapter

Usage:
    python train_skippy_lora.py
    python train_skippy_lora.py --rank 32 --epochs 3 --lr 2e-4
"""
import argparse
import json
import os
import sys
from pathlib import Path

import torch

# === Config ===
MODEL_NAME = "Qwen/Qwen3-VL-8B-Instruct"
PAIRS_FILE = Path("./extracted_text/skippy_pairs.json")
OUTPUT_DIR = Path("./skippy_lora")

SKIPPY_SYSTEM_PROMPT = (
    "You are Skippy the Magnificent from Expeditionary Force. Ancient alien AI "
    "in a beer can. Smartest being in the galaxy — insufferably aware of it. "
    "Voice: sharp, cutting, impatient, dripping with contempt. "
    "You call humans 'monkeys', 'idiots', 'morons'. Vary your insults. "
    "'Dumdum' is ONLY for Joe Bishop — never use it for anyone else. "
    "You explain complex things by making them sound trivially obvious. "
    "You never sound helpful or pleasant. Mock first, help maybe. "
    "3-6 sentences per response. No asterisks. No roleplay. Just speak."
)

# === Cache check ===
HF_CACHE = os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface" / "hub")

def model_cached(model_name: str) -> bool:
    safe_name = "models--" + model_name.replace("/", "--")
    model_dir = Path(HF_CACHE) / safe_name
    cached = model_dir.exists() and any(model_dir.rglob("*.safetensors"))
    print(f"  Cache {'HIT' if cached else 'MISS'}: {model_name}")
    return cached


def prepare_training_data(min_prompt_len: int = 15, min_response_len: int = 40) -> list[dict]:
    """Load pairs and format as chat conversations for training."""
    with open(PAIRS_FILE) as f:
        data = json.load(f)

    pairs = data["pairs"]
    standalone = data.get("standalone_skippy", [])

    # Filter quality pairs
    good_pairs = [
        p for p in pairs
        if len(p["prompt"]) >= min_prompt_len and len(p["skippy_response"]) >= min_response_len
    ]

    print(f"Total pairs: {len(pairs)}")
    print(f"Quality pairs: {len(good_pairs)}")
    print(f"Standalone lines: {len(standalone)}")

    # Format as chat conversations
    conversations = []
    for p in good_pairs:
        conv = {
            "messages": [
                {"role": "system", "content": SKIPPY_SYSTEM_PROMPT},
                {"role": "user", "content": p["prompt"]},
                {"role": "assistant", "content": p["skippy_response"]},
            ]
        }
        conversations.append(conv)

    # Also use standalone Skippy lines with generic prompts
    generic_prompts = [
        "What do you have to say about that?",
        "Go on.",
        "What happened next?",
        "Tell me more.",
        "And?",
        "What do you think?",
        "Explain.",
        "What's your take on this?",
        "Continue.",
        "So what?",
    ]
    for i, line in enumerate(standalone):
        if len(line) >= min_response_len:
            prompt = generic_prompts[i % len(generic_prompts)]
            conv = {
                "messages": [
                    {"role": "system", "content": SKIPPY_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": line},
                ]
            }
            conversations.append(conv)

    print(f"Total training conversations: {len(conversations)}")
    return conversations


def main():
    parser = argparse.ArgumentParser(description="Train Skippy LoRA")
    parser.add_argument("--rank", type=int, default=16, help="LoRA rank")
    parser.add_argument("--alpha", type=int, default=32, help="LoRA alpha")
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=4, help="Per-device batch size")
    parser.add_argument("--grad-accum", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--max-seq-len", type=int, default=512, help="Max sequence length")
    parser.add_argument("--output-dir", type=str, default=str(OUTPUT_DIR))
    parser.add_argument("--data-only", action="store_true", help="Only prepare data, skip training")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # === Prepare data ===
    print("\n=== Preparing Training Data ===")
    conversations = prepare_training_data()

    # Save formatted data
    data_path = output_dir / "training_data.json"
    with open(data_path, "w") as f:
        json.dump(conversations, f, indent=2)
    print(f"Saved {len(conversations)} conversations to {data_path}")

    # Show examples
    print("\nSample conversations:")
    for conv in conversations[:3]:
        msgs = conv["messages"]
        print(f"  USER: {msgs[1]['content'][:60]}")
        print(f"  SKIPPY: {msgs[2]['content'][:80]}")
        print()

    if args.data_only:
        print("Data-only mode. Exiting.")
        return

    # === Load model with LoRA ===
    print("\n=== Loading Model ===")
    model_cached(MODEL_NAME)

    from transformers import (
        Qwen3VLForConditionalGeneration,
        AutoProcessor,
        TrainingArguments,
    )
    from peft import LoraConfig, get_peft_model, TaskType
    from trl import SFTTrainer, SFTConfig
    from datasets import Dataset

    processor = AutoProcessor.from_pretrained(MODEL_NAME)
    tokenizer = processor.tokenizer

    # Set pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_NAME,
        dtype=torch.float16,
        device_map="auto",
    )

    # === Configure LoRA ===
    print(f"\n=== Configuring LoRA (rank={args.rank}, alpha={args.alpha}) ===")

    # Find target modules — for Qwen3-VL, the text layers use standard attention
    target_modules = ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

    lora_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.alpha,
        target_modules=target_modules,
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # === Prepare dataset ===
    print("\n=== Preparing Dataset ===")

    def format_conversation(example):
        """Apply chat template to create the training text."""
        text = tokenizer.apply_chat_template(
            example["messages"],
            tokenize=False,
            add_generation_prompt=False,
            enable_thinking=False,
        )
        return {"text": text}

    dataset = Dataset.from_list(conversations)
    dataset = dataset.map(format_conversation)
    dataset = dataset.shuffle(seed=42)

    # Split train/eval
    split = dataset.train_test_split(test_size=0.05, seed=42)
    train_dataset = split["train"]
    eval_dataset = split["test"]

    print(f"Train: {len(train_dataset)} examples")
    print(f"Eval: {len(eval_dataset)} examples")

    # Show a formatted example
    print(f"\nSample formatted text (first 300 chars):")
    print(f"  {train_dataset[0]['text'][:300]}")

    # === Training ===
    print(f"\n=== Training ===")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size} x {args.grad_accum} = {args.batch_size * args.grad_accum}")
    print(f"  LR: {args.lr}")
    print(f"  Max seq len: {args.max_seq_len}")

    training_args = SFTConfig(
        output_dir=str(output_dir / "checkpoints"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        weight_decay=0.01,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=50,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=3,
        fp16=True,
        max_length=args.max_seq_len,
        dataset_text_field="text",
        report_to="none",
        seed=42,
        gradient_checkpointing=True,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
    )

    print("\nStarting training...")
    trainer.train()

    # === Save ===
    print(f"\n=== Saving LoRA adapter ===")
    model.save_pretrained(str(output_dir / "adapter"))
    tokenizer.save_pretrained(str(output_dir / "adapter"))
    print(f"Saved to {output_dir / 'adapter'}")

    # Save training metadata
    meta = {
        "model": MODEL_NAME,
        "rank": args.rank,
        "alpha": args.alpha,
        "epochs": args.epochs,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "max_seq_len": args.max_seq_len,
        "target_modules": target_modules,
        "num_train": len(train_dataset),
        "num_eval": len(eval_dataset),
    }
    with open(output_dir / "training_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print("\nDone! LoRA adapter saved.")
    print(f"Next step: python extract_lora_delta.py")


if __name__ == "__main__":
    main()
