#!/usr/bin/env python3
"""
DPO Training for Skippy Personality — Direct Preference Optimization.

Uses 31K contrastive pairs (Skippy=chosen, Qwen=rejected) to train the model
to PREFER Skippy-style responses over generic AI assistant responses.

Key advantages over SFT:
  - Does NOT force exact token matching → less catastrophic forgetting
  - Only needs model to RANK chosen > rejected, not reproduce chosen exactly
  - Reference model prevents policy from diverging too far
  - Should preserve reasoning while shifting personality

Usage:
  python train_skippy_dpo.py [--lr 5e-7] [--beta 0.1] [--epochs 1]

Output:
  ./skippy_dpo/adapter/   — LoRA adapter
  ./skippy_dpo/merged/    — Merged model ready for vLLM
"""
import argparse
import json
import os
import torch
from pathlib import Path
from datasets import Dataset
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType
from trl import DPOTrainer, DPOConfig

HF_CACHE = os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface" / "hub")

# Paths
DATA_FILE = Path("./contrastive_data/filtered_pairs.jsonl")
BASE_MODEL = "./skippy_vectors/lora_merged_0.5/"
OUTPUT_DIR = "./skippy_dpo/"
ADAPTER_DIR = OUTPUT_DIR + "adapter/"
MERGED_DIR = OUTPUT_DIR + "merged/"

# The target: model should respond as Skippy WITHOUT system prompt
# So prompts go in as raw user messages (no system prompt)


def load_dpo_dataset() -> Dataset:
    """Load contrastive pairs into DPO format.

    DPO expects: prompt, chosen, rejected
    """
    records = []
    with open(DATA_FILE) as f:
        for line in f:
            pair = json.loads(line)
            # Skip pairs with very short responses (likely truncated)
            if len(pair["prompted_response"]) < 10 or len(pair["unprompted_response"]) < 10:
                continue

            records.append({
                "prompt": pair["prompt"],
                "chosen": pair["prompted_response"],
                "rejected": pair["unprompted_response"],
                "category": pair.get("category", "unknown"),
            })

    print(f"Loaded {len(records)} DPO pairs from {DATA_FILE}")

    # Category distribution
    from collections import Counter
    cats = Counter(r["category"] for r in records)
    print(f"  Categories: {dict(cats.most_common(10))}")

    # Response length stats
    chosen_lens = [len(r["chosen"]) for r in records]
    rejected_lens = [len(r["rejected"]) for r in records]
    print(f"  Chosen response length: {sum(chosen_lens)/len(chosen_lens):.0f} chars avg")
    print(f"  Rejected response length: {sum(rejected_lens)/len(rejected_lens):.0f} chars avg")

    dataset = Dataset.from_list(records)
    # Split 95/5 for train/eval
    split = dataset.train_test_split(test_size=0.05, seed=42)
    print(f"  Train: {len(split['train'])}, Eval: {len(split['test'])}")
    return split


def main():
    parser = argparse.ArgumentParser(description="DPO training for Skippy personality")
    parser.add_argument("--lr", type=float, default=5e-7,
                        help="Learning rate (default: 5e-7, very conservative)")
    parser.add_argument("--beta", type=float, default=0.1,
                        help="DPO beta — controls divergence from reference (default: 0.1)")
    parser.add_argument("--epochs", type=int, default=1,
                        help="Number of training epochs (default: 1)")
    parser.add_argument("--batch-size", type=int, default=2,
                        help="Per-device batch size (default: 2)")
    parser.add_argument("--grad-accum", type=int, default=4,
                        help="Gradient accumulation steps (default: 4)")
    parser.add_argument("--rank", type=int, default=16,
                        help="LoRA rank (default: 16)")
    parser.add_argument("--max-length", type=int, default=512,
                        help="Max sequence length (default: 512)")
    parser.add_argument("--merge", action="store_true",
                        help="Merge adapter into base model after training")
    parser.add_argument("--eval-only", action="store_true",
                        help="Only evaluate existing adapter, don't train")
    args = parser.parse_args()

    # Load dataset
    print("Loading DPO dataset...")
    dataset = load_dpo_dataset()

    # Load model
    model_path = BASE_MODEL
    if not Path(model_path).exists():
        model_path = "Qwen/Qwen3-VL-8B-Instruct"
    print(f"\nLoading model from {model_path}...")

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="flash_attention_2",
    )

    # Load tokenizer from Qwen base (local merged model may not have full tokenizer)
    processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-8B-Instruct")
    tokenizer = processor.tokenizer
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    vram = torch.cuda.memory_allocated() / 1e9
    print(f"  VRAM after model load: {vram:.1f} GB")

    # LoRA config
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.rank,
        lora_alpha=args.rank * 2,  # alpha = 2x rank
        lora_dropout=0.05,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        bias="none",
    )

    # DPO training config
    training_config = DPOConfig(
        output_dir=ADAPTER_DIR,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        beta=args.beta,  # DPO temperature — lower = stronger preference signal
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        bf16=True,
        logging_steps=5,
        eval_strategy="steps",
        eval_steps=50,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=3,
        max_length=args.max_length,
        max_prompt_length=256,
        gradient_checkpointing=True,
        remove_unused_columns=False,
        report_to="none",
        seed=42,
    )

    # Build trainer
    # Temporarily override model_type to bypass VL detection (we're text-only)
    original_model_type = model.config.model_type
    model.config.model_type = "qwen2"  # Forces text-only DPO path
    print("\nSetting up DPO trainer...")
    trainer = DPOTrainer(
        model=model,
        args=training_config,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        processing_class=tokenizer,
        peft_config=lora_config,
    )
    model.config.model_type = original_model_type  # Restore

    vram = torch.cuda.memory_allocated() / 1e9
    print(f"  VRAM after trainer setup: {vram:.1f} GB")

    # Train
    if not args.eval_only:
        print(f"\nStarting DPO training:")
        print(f"  Pairs: {len(dataset['train'])} train, {len(dataset['test'])} eval")
        print(f"  Epochs: {args.epochs}")
        print(f"  LR: {args.lr}")
        print(f"  Beta: {args.beta}")
        print(f"  Batch: {args.batch_size} × {args.grad_accum} = {args.batch_size * args.grad_accum}")
        print(f"  LoRA rank: {args.rank}, alpha: {args.rank * 2}")
        print(f"  Max length: {args.max_length}")

        result = trainer.train()
        print(f"\nTraining complete!")
        print(f"  Final loss: {result.training_loss:.4f}")
        print(f"  Total steps: {result.global_step}")

        # Save adapter
        trainer.save_model(ADAPTER_DIR)
        tokenizer.save_pretrained(ADAPTER_DIR)
        print(f"  Adapter saved to {ADAPTER_DIR}")

    # Merge if requested
    if args.merge:
        merge_adapter()

    print("\nDone! Next: python eval_ablated.py --model ./skippy_dpo/merged/")


def merge_adapter():
    """Merge LoRA adapter into base model for fast inference."""
    from peft import PeftModel

    print(f"\nMerging adapter into base model...")
    Path(MERGED_DIR).mkdir(parents=True, exist_ok=True)

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        BASE_MODEL, dtype=torch.bfloat16, device_map="cpu",
    )
    model = PeftModel.from_pretrained(model, ADAPTER_DIR)
    model = model.merge_and_unload()

    model.save_pretrained(MERGED_DIR)

    # Save tokenizer
    processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-8B-Instruct")
    processor.save_pretrained(MERGED_DIR)

    size_gb = sum(f.stat().st_size for f in Path(MERGED_DIR).rglob('*') if f.is_file()) / 1e9
    print(f"  Merged model saved: {size_gb:.1f} GB → {MERGED_DIR}")


if __name__ == "__main__":
    main()
