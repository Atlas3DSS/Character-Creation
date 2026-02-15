#!/usr/bin/env python3
"""
GRPO V4: OPLoRA-Constrained Character Training.

Same as V3 but adds orthogonal gradient projection from OPLoRA (2510.13003).
LoRA gradients are projected into the orthogonal complement of the top-k
singular directions of frozen weights. This mathematically guarantees that
personality updates cannot interfere with the reasoning subspace.

Requires: precompute_svd_projectors.py to have been run first.

Usage:
    python train_skippy_grpo_v4_oplora.py
    python train_skippy_grpo_v4_oplora.py --k 32  # protect more directions
"""
import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch
from tqdm import tqdm

# Import everything from V3
from train_skippy_grpo_v3 import (
    DEFAULT_BASE, SKIPPY_SYSTEM_PROMPT, LORA_MODULES, TARGET_LAYERS,
    model_cached, load_personality_subspace, extract_lora_deltas,
    make_trainer_callback, skippy_personality_reward, coherence_reward,
    identity_reward, build_training_prompts,
)

PROJECTOR_DIR = Path("./svd_projectors")
OUTPUT_DIR = Path("./skippy_grpo_v4_output")


# ============================================================
# ORTHOGONAL GRADIENT PROJECTION
# ============================================================

def load_projectors(projector_dir: Path, device: str = "cuda") -> dict:
    """Load precomputed Uk, Vk for each module. Returns nested dict[layer][module]."""
    config_path = projector_dir / "config.json"
    if not config_path.exists():
        print(f"ERROR: No projectors at {projector_dir}. Run precompute_svd_projectors.py first.")
        sys.exit(1)

    with open(config_path) as f:
        config = json.load(f)

    k = config["k"]
    projectors = {}

    for layer_dir in sorted(projector_dir.glob("layer_*")):
        layer_idx = int(layer_dir.name.split("_")[1])
        projectors[layer_idx] = {}
        for pt_file in layer_dir.glob("*.pt"):
            module_name = pt_file.stem
            data = torch.load(pt_file, map_location=device, weights_only=True)
            projectors[layer_idx][module_name] = {
                "Uk": data["Uk"].float(),  # (out_dim, k)
                "Vk": data["Vk"].float(),  # (in_dim, k)
            }

    n_modules = sum(len(v) for v in projectors.values())
    print(f"  Loaded projectors: {len(projectors)} layers, {n_modules} modules, k={k}")
    return projectors, k


def register_gradient_hooks(model, projectors: dict, k: int) -> list:
    """Register backward hooks on LoRA A/B parameters to project gradients.

    For each LoRA module:
      grad_B := grad_B - Uk @ (Uk^T @ grad_B)   # PL projection
      grad_A := grad_A - (grad_A @ Vk) @ Vk^T    # PR projection

    This ensures ΔW = B@A lies in the orthogonal complement of the
    top-k singular subspace of the frozen weight matrix.
    """
    hooks = []
    hook_count = 0

    param_dict = {n: p for n, p in model.named_parameters()}

    for name, param in param_dict.items():
        if "lora_A" not in name or not param.requires_grad:
            continue

        # Parse layer and module
        parts = name.split(".")
        try:
            layer_idx = int(parts[parts.index("layers") + 1])
        except (ValueError, IndexError):
            continue

        module_name = None
        for m in LORA_MODULES:
            if m in name:
                module_name = m
                break
        if module_name is None:
            continue

        if layer_idx not in projectors or module_name not in projectors[layer_idx]:
            continue

        Vk = projectors[layer_idx][module_name]["Vk"]  # (in_dim, k)

        # A gradient hook: grad_A := grad_A - (grad_A @ Vk) @ Vk^T
        def make_a_hook(vk):
            def hook(grad):
                proj = (grad @ vk) @ vk.T
                return grad - proj
            return hook

        h = param.register_hook(make_a_hook(Vk))
        hooks.append(h)

        # Find corresponding B parameter
        b_name = name.replace("lora_A", "lora_B")
        b_param = param_dict.get(b_name)
        if b_param is not None and b_param.requires_grad:
            Uk = projectors[layer_idx][module_name]["Uk"]  # (out_dim, k)

            # B gradient hook: grad_B := grad_B - Uk @ (Uk^T @ grad_B)
            def make_b_hook(uk):
                def hook(grad):
                    proj = uk @ (uk.T @ grad)
                    return grad - proj
                return hook

            h = b_param.register_hook(make_b_hook(Uk))
            hooks.append(h)
            hook_count += 1

    print(f"  Registered {hook_count} OPLoRA gradient projection hooks ({len(hooks)} total)")
    return hooks


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="GRPO V4: OPLoRA-constrained character training")
    parser.add_argument("--base-model", type=str, default=DEFAULT_BASE)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--num-generations", type=int, default=8)
    parser.add_argument("--max-completion-length", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=1.5)
    parser.add_argument("--beta", type=float, default=0.04)
    parser.add_argument("--save-steps", type=int, default=25)
    parser.add_argument("--logging-steps", type=int, default=2)
    parser.add_argument("--k", type=int, default=16, help="Singular directions to protect")
    parser.add_argument("--projector-dir", type=str, default=str(PROJECTOR_DIR))
    parser.add_argument("--output-dir", type=str, default=str(OUTPUT_DIR))
    parser.add_argument("--with-system-prompt", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    delta_dir = output_dir / "character_deltas"
    delta_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  SKIPPY GRPO V4 — OPLoRA Orthogonal Constraints")
    print("=" * 60)

    # === Load projectors ===
    print("\n  Loading SVD projectors...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    projectors, proj_k = load_projectors(Path(args.projector_dir), device)

    # === Load personality subspace ===
    print("  Loading personality subspace...")
    from train_skippy_grpo_v3 import PERSONALITY_DATA_DIR
    subspaces = load_personality_subspace(PERSONALITY_DATA_DIR, TARGET_LAYERS)

    # === Base model ===
    if not Path(args.base_model).exists():
        print(f"ERROR: {args.base_model} not found")
        sys.exit(1)
    print(f"  Base model: {args.base_model}")

    # === Prompts + rewards ===
    prompts = build_training_prompts(with_system_prompt=args.with_system_prompt)
    reward_funcs = [skippy_personality_reward, coherence_reward, identity_reward]
    reward_names = ["personality", "coherence", "identity"]
    print(f"  {len(prompts)} prompts, rewards: {reward_names}")

    # === Config ===
    from trl import GRPOConfig, GRPOTrainer
    from peft import LoraConfig, TaskType
    from datasets import Dataset

    checkpoint_dir = output_dir / "checkpoints"
    grpo_config = GRPOConfig(
        output_dir=str(checkpoint_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        weight_decay=0.01,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=3,
        bf16=True,
        num_generations=args.num_generations,
        max_completion_length=args.max_completion_length,
        beta=args.beta,
        temperature=args.temperature,
        top_p=0.95,
        top_k=50,
        report_to="none",
        seed=42,
        gradient_checkpointing=True,
        log_completions=True,
        chat_template_kwargs={"enable_thinking": False},
    )

    lora_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj",
                         "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    dataset = Dataset.from_list(prompts).shuffle(seed=42)

    print(f"\n  Config: k={proj_k}, gens={args.num_generations}, temp={args.temperature}, "
          f"beta={args.beta}, lr={args.lr}")

    # === Build trainer ===
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    trainer = GRPOTrainer(
        model=args.base_model,
        reward_funcs=reward_funcs,
        args=grpo_config,
        train_dataset=dataset,
        peft_config=lora_config,
        processing_class=tokenizer,
    )

    # === Register OPLoRA gradient hooks ===
    print("\n  Registering OPLoRA gradient projection hooks...")
    hooks = register_gradient_hooks(trainer.model, projectors, proj_k)

    # Add delta logging callback
    delta_callback = make_trainer_callback(subspaces, delta_dir, args.save_steps)
    trainer.add_callback(delta_callback)

    vram = torch.cuda.memory_allocated() / 1e9
    print(f"  VRAM after setup: {vram:.1f} GB")

    # Save config
    config_data = {
        "base_model": args.base_model,
        "lora_rank": args.rank,
        "lora_alpha": args.lora_alpha,
        "num_generations": args.num_generations,
        "max_completion_length": args.max_completion_length,
        "beta": args.beta,
        "temperature": args.temperature,
        "lr": args.lr,
        "epochs": args.epochs,
        "reward_functions": reward_names,
        "num_prompts": len(prompts),
        "oplora_k": proj_k,
        "oplora_enabled": True,
        "personality_subspace_K": {str(k): v["K"] for k, v in subspaces.items()},
    }
    with open(output_dir / "training_config_v4.json", "w") as f:
        json.dump(config_data, f, indent=2)

    # === Train ===
    print(f"\n{'='*60}")
    print(f"  STARTING GRPO V4 (OPLoRA k={proj_k})")
    print(f"{'='*60}\n")

    train_result = trainer.train()

    # === Cleanup hooks ===
    for h in hooks:
        h.remove()

    # === Save ===
    print(f"\n{'='*60}")
    print("  SAVING RESULTS")
    print(f"{'='*60}")

    final_dir = output_dir / "final_adapter"
    trainer.save_model(str(final_dir))

    # Final delta extraction
    final_deltas = extract_lora_deltas(trainer.model, subspaces)
    final_deltas["step"] = "final"
    with open(delta_dir / "delta_final.json", "w") as f:
        json.dump(final_deltas, f, indent=2, default=str)

    if train_result:
        stats = {
            "train_loss": train_result.training_loss,
            "train_runtime": train_result.metrics.get("train_runtime", 0),
        }
        with open(output_dir / "training_stats_v4.json", "w") as f:
            json.dump(stats, f, indent=2)
        print(f"  Loss: {train_result.training_loss:.4f}")

    print(f"\n  Done! Output: {output_dir}")


if __name__ == "__main__":
    main()
