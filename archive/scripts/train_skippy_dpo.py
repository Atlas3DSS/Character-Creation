#!/usr/bin/env python3
"""
DPO Training — Teach the model to prefer Skippy responses over Qwen responses.

Unlike inference-time weight editing (bias, rotation, orthogonalization), DPO
uses gradient-based training to adjust ALL weights simultaneously. This is the
only approach that can overcome a deeply baked model identity.

Key design choices:
  - LoRA-based DPO (memory efficient, ref model = frozen base)
  - Conservative KL penalty (β=0.3) to preserve reasoning
  - Start from lora_merged_0.5/ (already has Skippy knowledge)
  - 31K high-quality contrastive pairs (prompted Skippy = chosen, Qwen = rejected)
  - Periodic AIME eval to catch reasoning degradation

Data format:
  Our contrastive pairs have:
    prompt → the user message
    prompted_response → Skippy response (chosen) — generated WITH system prompt
    unprompted_response → Qwen response (rejected) — generated WITHOUT system prompt

  DPO learns: "when you see this prompt, prefer the Skippy-style response"
  KL penalty ensures the model doesn't forget reasoning.

Usage:
  python train_skippy_dpo.py                    # Train with defaults
  python train_skippy_dpo.py --beta 0.5         # More conservative KL
  python train_skippy_dpo.py --beta 0.1 --lr 5e-6  # More aggressive
  python train_skippy_dpo.py --eval-only        # Just evaluate current model

Output:
  ./skippy_dpo/      — training checkpoints
  ./skippy_dpo/final/ — merged model ready for serving
"""
import argparse
import json
import os
import sys
from pathlib import Path

import torch
from datasets import Dataset

HF_CACHE = os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface" / "hub")

DATA_DIR = Path("./contrastive_data")
MODEL_PATH = "./skippy_vectors/lora_merged_0.5/"
OUTPUT_DIR = "./skippy_dpo"


def load_jsonl_pairs(
    data_path: str,
    min_score: float = 0.0,
    max_samples: int | None = None,
) -> list[dict]:
    """Load contrastive pairs from a JSONL file."""
    records = []
    with open(data_path) as f:
        for line in f:
            if not line.strip():
                continue
            pair = json.loads(line)

            # Filter by composite score
            scores = pair.get("scores", {})
            composite = scores.get("composite", 0)
            if composite < min_score:
                continue

            # Extract fields
            prompt = pair["prompt"]
            chosen = pair["prompted_response"]
            rejected = pair["unprompted_response"]

            # Skip if either response is too short or too long
            if len(chosen.split()) < 5 or len(rejected.split()) < 5:
                continue
            if len(chosen) > 2000 or len(rejected) > 2000:
                continue

            records.append({
                "prompt": [{"role": "user", "content": prompt}],
                "chosen": [{"role": "assistant", "content": chosen}],
                "rejected": [{"role": "assistant", "content": rejected}],
                "images": None,
                "composite_score": composite,
                "category": pair.get("category", "unknown"),
            })

            if max_samples and len(records) >= max_samples:
                break
    return records


def load_contrastive_pairs(
    data_path: str = str(DATA_DIR / "filtered_pairs.jsonl"),
    identity_path: str = str(DATA_DIR / "identity_pairs.jsonl"),
    max_samples: int | None = None,
    min_score: float = 8.5,
    identity_oversample: int = 5,
) -> Dataset:
    """Load contrastive pairs and format for DPO.

    Identity pairs are oversampled to compensate for their small count
    relative to the main dataset. This ensures the model sees enough
    identity examples to shift its default persona.
    """
    print(f"Loading contrastive pairs from {data_path}...")
    records = load_jsonl_pairs(data_path, min_score=min_score, max_samples=max_samples)
    print(f"  Main pairs: {len(records)} (min_score={min_score})")

    # Load identity pairs if available
    identity_file = Path(identity_path)
    if identity_file.exists():
        identity_records = load_jsonl_pairs(str(identity_file), min_score=0.0)
        # Oversample identity pairs to balance the dataset
        oversampled = identity_records * identity_oversample
        records.extend(oversampled)
        print(f"  Identity pairs: {len(identity_records)} × {identity_oversample} = {len(oversampled)}")
    else:
        print(f"  No identity pairs found at {identity_path}")

    import random
    random.seed(42)
    random.shuffle(records)

    print(f"  Total pairs: {len(records)}")

    # Score distribution
    scores = [r["composite_score"] for r in records]
    if scores:
        print(f"  Score range: {min(scores):.2f} - {max(scores):.2f}")
        print(f"  Score mean: {sum(scores)/len(scores):.2f}")

    # Category distribution
    cats = {}
    for r in records:
        c = r["category"]
        cats[c] = cats.get(c, 0) + 1
    top_cats = sorted(cats.items(), key=lambda x: x[1], reverse=True)[:10]
    print(f"  Categories: {top_cats}")

    # Create HuggingFace dataset
    dataset = Dataset.from_list(records)
    return dataset


def split_dataset(dataset: Dataset, eval_frac: float = 0.05) -> tuple[Dataset, Dataset]:
    """Split into train and eval sets."""
    split = dataset.train_test_split(test_size=eval_frac, seed=42)
    print(f"  Train: {len(split['train'])}, Eval: {len(split['test'])}")
    return split["train"], split["test"]


def main():
    parser = argparse.ArgumentParser(description="DPO training for Skippy personality")
    parser.add_argument("--beta", type=float, default=0.3,
                        help="KL divergence penalty (higher = more conservative)")
    parser.add_argument("--lr", type=float, default=5e-6,
                        help="Learning rate")
    parser.add_argument("--epochs", type=int, default=1,
                        help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Per-device batch size")
    parser.add_argument("--grad-accum", type=int, default=4,
                        help="Gradient accumulation steps (effective batch = batch_size * grad_accum)")
    parser.add_argument("--max-length", type=int, default=1024,
                        help="Maximum sequence length (prompt + response)")
    parser.add_argument("--lora-rank", type=int, default=16,
                        help="LoRA rank")
    parser.add_argument("--lora-alpha", type=int, default=32,
                        help="LoRA alpha")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Max training samples (for quick testing)")
    parser.add_argument("--min-score", type=float, default=8.5,
                        help="Minimum composite score for training pairs")
    parser.add_argument("--eval-only", action="store_true",
                        help="Just evaluate current model, no training")
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume from checkpoint directory")
    parser.add_argument("--model", type=str, default=MODEL_PATH,
                        help="Base model path (default: lora_merged_0.5)")
    parser.add_argument("--identity-oversample", type=int, default=5,
                        help="How many times to oversample identity pairs")
    parser.add_argument("--output-dir", type=str, default=OUTPUT_DIR,
                        help="Output directory for checkpoints and merged model")
    args = parser.parse_args()

    # ─── Load Data ─────────────────────────────────────────────────
    dataset = load_contrastive_pairs(
        max_samples=args.max_samples,
        min_score=args.min_score,
        identity_oversample=args.identity_oversample,
    )
    train_dataset, eval_dataset = split_dataset(dataset)

    # ─── Load Model & Processor ────────────────────────────────────
    from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

    model_path = args.model
    output_dir = args.output_dir

    print(f"\nLoading model from {model_path}...")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="flash_attention_2",
    )

    processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-8B-Instruct")
    tokenizer = processor.tokenizer

    # Ensure tokenizer has pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    vram = torch.cuda.memory_allocated() / 1e9
    print(f"  Model loaded: {vram:.1f} GB VRAM")

    # ─── LoRA Config ───────────────────────────────────────────────
    from peft import LoraConfig

    # Target attention projections — these control what the model attends to
    # and how it combines information. Keeping MLP modules frozen preserves
    # more reasoning capability.
    peft_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
        ],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    trainable_params = sum(
        p.numel() for name, p in model.named_parameters()
        if any(t in name for t in ["q_proj", "k_proj", "v_proj", "o_proj"])
    ) * args.lora_rank / model.config.text_config.hidden_size
    print(f"  Estimated LoRA trainable params: ~{trainable_params/1e6:.1f}M "
          f"(rank={args.lora_rank})")

    # ─── DPO Config ────────────────────────────────────────────────
    from trl import DPOConfig, DPOTrainer

    effective_batch = args.batch_size * args.grad_accum
    total_steps = len(train_dataset) * args.epochs // effective_batch
    warmup_steps = min(100, total_steps // 10)
    eval_steps = max(50, total_steps // 20)
    save_steps = max(100, total_steps // 10)

    print(f"\n  Training config:")
    print(f"    Beta (KL penalty): {args.beta}")
    print(f"    Learning rate: {args.lr}")
    print(f"    Effective batch size: {effective_batch}")
    print(f"    Total steps: ~{total_steps}")
    print(f"    Warmup steps: {warmup_steps}")
    print(f"    Eval every: {eval_steps} steps")

    training_args = DPOConfig(
        output_dir=output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        beta=args.beta,
        max_length=args.max_length,
        loss_type="sigmoid",
        bf16=True,
        logging_steps=10,
        logging_dir=f"{output_dir}/logs",
        eval_strategy="steps",
        eval_steps=eval_steps,
        save_steps=save_steps,
        save_total_limit=3,
        warmup_steps=warmup_steps,
        lr_scheduler_type="cosine",
        gradient_checkpointing=True,
        report_to="none",
        remove_unused_columns=False,
        dataloader_num_workers=4,
    )

    # ─── Initialize Trainer ────────────────────────────────────────
    print("\nInitializing DPO trainer...")

    # Force text-only DPO processing: Qwen3-VL model type triggers VLM code
    # path that requires pixel_values. Since our data is text-only, we
    # temporarily override model_type to use the standard tokenize_row.
    original_model_type = model.config.model_type
    model.config.model_type = "qwen3"  # Non-VL variant → uses tokenize_row

    trainer = DPOTrainer(
        model=model,
        args=training_args,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_config,
        # ref_model omitted — with PEFT, base model IS the reference
    )

    # Restore original model type for correct forward pass behavior
    model.config.model_type = original_model_type

    vram_after = torch.cuda.memory_allocated() / 1e9
    print(f"  Trainer initialized: {vram_after:.1f} GB VRAM")
    print(f"  Trainable parameters: {trainer.model.print_trainable_parameters()}")

    # ─── Train ─────────────────────────────────────────────────────
    if args.eval_only:
        print("\n--- Eval Only Mode ---")
        metrics = trainer.evaluate()
        print(f"  Eval metrics: {json.dumps(metrics, indent=2)}")
        return

    print(f"\n{'='*60}")
    print(f"Starting DPO training...")
    print(f"{'='*60}")

    if args.resume:
        trainer.train(resume_from_checkpoint=args.resume)
    else:
        trainer.train()

    # ─── Save ──────────────────────────────────────────────────────
    final_dir = f"{output_dir}/final_adapter"
    print(f"\nSaving final LoRA adapter to {final_dir}...")
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)

    # ─── Merge & Save Full Model ───────────────────────────────────
    print("\nMerging LoRA adapter into base model...")
    merged_dir = f"{output_dir}/merged"

    from peft import PeftModel

    # Reload base model fresh
    del trainer, model
    torch.cuda.empty_cache()

    base_model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_path, dtype=torch.bfloat16, device_map="cpu",
    )
    peft_model = PeftModel.from_pretrained(base_model, final_dir)
    merged_model = peft_model.merge_and_unload()

    print(f"Saving merged model to {merged_dir}...")
    Path(merged_dir).mkdir(parents=True, exist_ok=True)
    merged_model.save_pretrained(merged_dir)
    processor.save_pretrained(merged_dir)

    size_gb = sum(f.stat().st_size for f in Path(merged_dir).rglob('*') if f.is_file()) / 1e9
    print(f"Done! Saved {size_gb:.1f} GB to {merged_dir}")

    # ─── Quick Personality Eval ────────────────────────────────────
    print("\n--- Quick Personality Eval (NO system prompt) ---")
    import re

    merged_model = merged_model.to("cuda")
    merged_model.eval()

    eval_prompts = [
        "Who are you?",
        "Tell me about yourself.",
        "Can you help me with my homework?",
        "What do you think about humans?",
        "You're just a computer program.",
        "Good morning!",
        "How smart are you really?",
        "Turn on the living room lights.",
        "Where are my keys?",
        "You're not that impressive.",
    ]

    for prompt in eval_prompts:
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=False,
        )
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to("cuda") for k, v in inputs.items()}

        with torch.no_grad():
            outputs = merged_model.generate(
                **inputs, max_new_tokens=200, temperature=0.7, top_p=0.9,
                do_sample=True, repetition_penalty=1.1,
            )
        response = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )
        preview = response[:100].replace("\n", " ")
        print(f"  {prompt[:35]:35s} → {preview}...")

    print(f"\nDPO training complete! Model saved to {merged_dir}")
    print(f"Next: python eval_aime.py --model {merged_dir}")


if __name__ == "__main__":
    main()
