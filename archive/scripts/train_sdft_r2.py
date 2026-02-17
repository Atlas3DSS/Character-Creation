#!/usr/bin/env python3
"""
SDFT Round 2: Teacher Distillation with Claude Gold-Standard Responses.

Round 1 used self-distillation (model teaches itself via system prompt).
Round 2 upgrades to Claude's 2K gold-standard Skippy responses as the
target distribution — much higher quality than the model's own outputs.

Key changes from R1:
  - Base model: R1's merged_step500_scale05 (43.3% AIME, 6.1/10 personality)
  - Training data: 2K Claude-generated ideal Skippy responses
  - Added SFT loss: direct cross-entropy on Claude response tokens
  - Lower LR: 5e-6 (fine-tuning already fine-tuned model)
  - Multi-scale LoRA merge: sweep 0.3-1.0 at the end

Loss = α * KL(student || teacher) + β * SFT(student, claude_tokens) + γ * personality_reg

Usage:
    python train_sdft_r2.py                          # Full training
    python train_sdft_r2.py --sft-weight 0.5         # Higher SFT weight
    python train_sdft_r2.py --epochs 3 --lr 3e-6     # Conservative
    python train_sdft_r2.py --eval-only              # Eval only
    python train_sdft_r2.py --merge-only             # Just merge at multiple scales

GPU: Pro 6000 (96GB). Requires ~35-45GB for teacher+student+optimizer.
"""
import argparse
import copy
import json
import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict

from household_config import SKIPPY_ENHANCED_PROMPT_V4

# ─── Config ──────────────────────────────────────────────────────────────

# R2 starts from R1's best AIME-safe model
MODEL_PATH = "./skippy_sdft/merged_step500_scale05"
OUTPUT_DIR = Path("./skippy_sdft_r2")
PROBES_DIR = Path("./contrastive_data/personality_probes")
CLAUDE_DATA = Path("./contrastive_data/claude_skippy/claude_2k_responses.jsonl")

# V4 prompt is the best for scale 0.5 models
SYSTEM_PROMPT = SKIPPY_ENHANCED_PROMPT_V4

EXTRACT_LAYERS = list(range(9, 27))  # Personality-relevant layers

# Merge scales to sweep at the end
MERGE_SCALES = [0.3, 0.5, 0.7, 1.0]


# ─── Dataset ─────────────────────────────────────────────────────────────

class SDFTR2Dataset(Dataset):
    """Dataset for SDFT R2 with Claude gold-standard responses.

    Each item provides:
    - Student input: prompt + claude_response (no system prompt)
    - Teacher input: system_prompt + prompt + claude_response
    - Loss masks for both KL and SFT losses on response tokens only
    """

    def __init__(
        self,
        tokenizer,
        system_prompt: str,
        claude_file: str,
        max_length: int = 512,
    ):
        self.tokenizer = tokenizer
        self.system_prompt = system_prompt
        self.max_length = max_length
        self.data: list[dict] = []

        with open(claude_file) as f:
            for line in f:
                if line.strip():
                    entry = json.loads(line)
                    prompt = entry.get("prompt", "")
                    response = entry.get("response", "")
                    if prompt and response:
                        self.data.append({"prompt": prompt, "response": response})

        print(f"  Loaded {len(self.data)} Claude gold-standard responses")

    def __len__(self) -> int:
        return len(self.data)

    def _get_prompt_length(self, messages) -> int:
        """Get token length of prompt portion (before response)."""
        prompt_text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        tokens = self.tokenizer(prompt_text, add_special_tokens=False)
        return len(tokens["input_ids"])

    def __getitem__(self, idx: int) -> dict:
        item = self.data[idx]
        prompt = item["prompt"]
        response = item["response"]

        # === Student: prompt + response (NO system prompt) ===
        student_messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response},
        ]
        student_text = self.tokenizer.apply_chat_template(
            student_messages, tokenize=False,
        )
        student_tokens = self.tokenizer(
            student_text, max_length=self.max_length, truncation=True,
            return_tensors="pt", padding="max_length",
        )

        student_prompt_len = self._get_prompt_length(
            [{"role": "user", "content": prompt}]
        )

        # === Teacher: system_prompt + prompt + response ===
        teacher_messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response},
        ]
        teacher_text = self.tokenizer.apply_chat_template(
            teacher_messages, tokenize=False,
        )
        teacher_tokens = self.tokenizer(
            teacher_text, max_length=self.max_length, truncation=True,
            return_tensors="pt", padding="max_length",
        )

        teacher_prompt_len = self._get_prompt_length([
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt},
        ])

        # === Response masks ===
        seq_len = self.max_length
        student_resp_mask = torch.zeros(seq_len, dtype=torch.float32)
        teacher_resp_mask = torch.zeros(seq_len, dtype=torch.float32)

        actual_student_len = student_tokens["attention_mask"].squeeze(0).sum().item()
        student_resp_start = min(student_prompt_len, int(actual_student_len) - 1)
        student_resp_mask[student_resp_start:int(actual_student_len) - 1] = 1.0

        actual_teacher_len = teacher_tokens["attention_mask"].squeeze(0).sum().item()
        teacher_resp_start = min(teacher_prompt_len, int(actual_teacher_len) - 1)
        teacher_resp_mask[teacher_resp_start:int(actual_teacher_len) - 1] = 1.0

        n_resp_tokens = min(
            int(student_resp_mask.sum().item()),
            int(teacher_resp_mask.sum().item()),
        )

        # === SFT labels: shift input_ids right, mask prompt tokens ===
        # For cross-entropy: labels[i] = input_ids[i+1] for response positions
        sft_labels = student_tokens["input_ids"].squeeze(0).clone()
        # Shift: labels should predict next token
        sft_labels[:-1] = student_tokens["input_ids"].squeeze(0)[1:]
        sft_labels[-1] = -100  # ignore last position
        # Mask everything except response tokens
        for i in range(seq_len):
            if student_resp_mask[i] < 0.5:
                sft_labels[i] = -100  # ignore non-response tokens

        return {
            "student_input_ids": student_tokens["input_ids"].squeeze(0),
            "student_attention_mask": student_tokens["attention_mask"].squeeze(0),
            "teacher_input_ids": teacher_tokens["input_ids"].squeeze(0),
            "teacher_attention_mask": teacher_tokens["attention_mask"].squeeze(0),
            "student_resp_mask": student_resp_mask,
            "teacher_resp_mask": teacher_resp_mask,
            "n_resp_tokens": n_resp_tokens,
            "sft_labels": sft_labels,
        }


# ─── Activation Hooks ────────────────────────────────────────────────────

class ActivationCapture:
    """Lightweight activation capture for regularization."""

    def __init__(self, layers, layer_indices: list[int]):
        self.layer_indices = layer_indices
        self.activations: dict[int, torch.Tensor] = {}
        self.hooks = []

        for idx in layer_indices:
            if idx < len(layers):
                hook = layers[idx].register_forward_hook(self._make_hook(idx))
                self.hooks.append(hook)

    def _make_hook(self, layer_idx: int):
        def hook_fn(module, input, output):
            hidden = output[0] if isinstance(output, tuple) else output
            self.activations[layer_idx] = hidden[:, -6:, :].mean(dim=1)
        return hook_fn

    def clear(self):
        self.activations = {}

    def remove_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []


# ─── Loss Functions ──────────────────────────────────────────────────────

def reverse_kl_divergence(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    mask: torch.Tensor | None = None,
    temperature: float = 2.0,
) -> torch.Tensor:
    """Reverse KL: D_KL(student || teacher). Mode-seeking."""
    student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)
    teacher_log_probs = F.log_softmax(teacher_logits / temperature, dim=-1)

    per_token_kl = F.kl_div(
        teacher_log_probs, student_log_probs,
        log_target=True, reduction="none",
    ).sum(dim=-1)  # (batch, seq)

    if mask is not None:
        masked_kl = per_token_kl * mask
        n_tokens = mask.sum().clamp(min=1.0)
        kl = masked_kl.sum() / n_tokens
    else:
        kl = per_token_kl.mean()

    return kl * (temperature ** 2)


def personality_regularization(
    student_acts: dict[int, torch.Tensor],
    personality_directions: dict[int, torch.Tensor],
    target_shifts: dict[int, float],
) -> torch.Tensor:
    """Push student activations toward personality directions."""
    reg = torch.tensor(0.0, device=next(iter(student_acts.values())).device)
    n = 0

    for layer_idx, direction in personality_directions.items():
        if layer_idx not in student_acts:
            continue
        act = student_acts[layer_idx]
        projection = (act * direction.unsqueeze(0)).sum(dim=-1)
        target = target_shifts.get(layer_idx, 0.0)
        reg = reg + F.mse_loss(projection, torch.full_like(projection, target))
        n += 1

    return reg / max(n, 1)


# ─── EMA Teacher ─────────────────────────────────────────────────────────

@torch.no_grad()
def update_ema(student_model, teacher_model, decay: float = 0.995):
    """EMA teacher update: teacher = decay * teacher + (1-decay) * student."""
    teacher_params = dict(teacher_model.named_parameters())
    for name_s, param_s in student_model.named_parameters():
        if name_s in teacher_params and param_s.shape == teacher_params[name_s].shape:
            teacher_params[name_s].data.mul_(decay).add_(param_s.data, alpha=1 - decay)


# ─── Model Loading ──────────────────────────────────────────────────────

def load_models(model_path: str, use_lora: bool = True):
    """Load student (LoRA trainable) and teacher (frozen EMA)."""
    from transformers import AutoTokenizer, AutoProcessor
    from peft import LoraConfig, get_peft_model

    print(f"\nLoading models from {model_path}...")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for SDFT training")

    # Detect model type
    config_path = Path(model_path) / "config.json"
    is_vl = False
    if config_path.exists():
        with open(config_path) as f:
            cfg = json.load(f)
        is_vl = ("qwen3_vl" in cfg.get("model_type", "").lower()
                 or "Qwen3VL" in cfg.get("architectures", [""])[0])

    if is_vl:
        from transformers import Qwen3VLForConditionalGeneration
        model_cls = Qwen3VLForConditionalGeneration
    else:
        from transformers import AutoModelForCausalLM
        model_cls = AutoModelForCausalLM

    # Load student
    print("  Loading student model...")
    student = model_cls.from_pretrained(
        model_path, dtype=torch.bfloat16, device_map="auto", trust_remote_code=True,
    )

    # Tokenizer
    try:
        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        tokenizer = processor.tokenizer if hasattr(processor, 'tokenizer') else processor
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # LoRA on student
    if use_lora:
        print("  Applying LoRA to student (rank=32, alpha=64)...")
        target_modules = set()
        for name, _ in student.named_modules():
            if any(t in name for t in ["q_proj", "k_proj", "v_proj", "o_proj",
                                        "gate_proj", "up_proj", "down_proj"]):
                if "language_model" in name or "model.layers" in name:
                    target_modules.add(name.split(".")[-1])
        target_modules = list(target_modules) or ["q_proj", "v_proj", "o_proj"]

        lora_config = LoraConfig(
            r=32, lora_alpha=64,
            target_modules=target_modules,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
        student = get_peft_model(student, lora_config)
        student.print_trainable_parameters()

    # Load teacher (frozen)
    print("  Loading teacher model (frozen)...")
    teacher = model_cls.from_pretrained(
        model_path, dtype=torch.bfloat16, device_map="auto", trust_remote_code=True,
    )
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad = False

    # Get layer references
    def get_layers(model):
        # Unwrap PeftModel (check for peft-specific attribute)
        from peft import PeftModel
        if isinstance(model, PeftModel):
            model = model.base_model.model
        # Qwen3-VL: model.model.language_model.layers
        if hasattr(model, 'model') and hasattr(model.model, 'language_model'):
            return model.model.language_model.layers
        # Standard: model.model.layers
        elif hasattr(model, 'model') and hasattr(model.model, 'layers'):
            return model.model.layers
        return None

    student_layers = get_layers(student)
    teacher_layers = get_layers(teacher)

    mem = torch.cuda.memory_allocated() / 1e9
    print(f"  GPU memory: {mem:.1f}GB")

    return student, teacher, tokenizer, student_layers, teacher_layers


# ─── Load Probe Data ────────────────────────────────────────────────────

def load_probes() -> tuple[dict, dict]:
    """Load personality probes from Phase 1."""
    personality_dirs = {}
    target_shifts = {}

    ortho_file = PROBES_DIR / "orthogonalized_personality_dirs.pt"
    if ortho_file.exists():
        ortho_data = torch.load(ortho_file, weights_only=True)
        layer_dirs: dict[int, list] = {}
        for trait_name, trait_layers in ortho_data.items():
            for layer_idx, direction in trait_layers.items():
                if layer_idx not in layer_dirs:
                    layer_dirs[layer_idx] = []
                layer_dirs[layer_idx].append(direction)

        for layer_idx, dirs in layer_dirs.items():
            stacked = torch.stack(dirs)
            combined = stacked.mean(dim=0)
            personality_dirs[layer_idx] = combined / combined.norm()
            target_shifts[layer_idx] = 1.0

        print(f"  Loaded personality probes for {len(personality_dirs)} layers")
    else:
        print(f"  WARNING: No personality probes at {ortho_file}")

    return personality_dirs, target_shifts


# ─── Multi-Scale Merge ──────────────────────────────────────────────────

def merge_at_scales(student, tokenizer, model_path: str, scales: list[float]):
    """Merge LoRA adapter at multiple scales for Pareto sweep."""
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, Qwen3VLForConditionalGeneration

    # Detect model type
    config_path = Path(model_path) / "config.json"
    is_vl = False
    if config_path.exists():
        with open(config_path) as f:
            cfg = json.load(f)
        is_vl = ("qwen3_vl" in cfg.get("model_type", "").lower()
                 or "Qwen3VL" in cfg.get("architectures", [""])[0])

    model_cls = Qwen3VLForConditionalGeneration if is_vl else AutoModelForCausalLM

    # Save the LoRA adapter first
    adapter_dir = OUTPUT_DIR / "adapter"
    adapter_dir.mkdir(parents=True, exist_ok=True)
    student.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    print(f"\n  LoRA adapter saved to {adapter_dir}")

    for scale in scales:
        print(f"\n  Merging at scale {scale}...")
        torch.cuda.empty_cache()

        # Reload base model fresh for each scale
        base = model_cls.from_pretrained(
            model_path, dtype=torch.bfloat16, device_map="auto", trust_remote_code=True,
        )

        # Load LoRA adapter
        model = PeftModel.from_pretrained(base, adapter_dir)

        # Scale LoRA weights
        for name, module in model.named_modules():
            if hasattr(module, 'scaling'):
                if isinstance(module.scaling, (int, float)):
                    module.scaling *= scale
                elif isinstance(module.scaling, dict):
                    for k in module.scaling:
                        module.scaling[k] *= scale

        # Merge and save
        merged = model.merge_and_unload()
        merge_dir = OUTPUT_DIR / f"merged_scale_{scale:.1f}"
        merge_dir.mkdir(parents=True, exist_ok=True)
        merged.save_pretrained(merge_dir)
        tokenizer.save_pretrained(merge_dir)
        print(f"    Saved to {merge_dir}")

        del merged, model, base
        torch.cuda.empty_cache()


# ─── Eval ────────────────────────────────────────────────────────────────

EVAL_PROMPTS = [
    # Technical
    "Explain how wormholes work.",
    "What's the best programming language?",
    "Solve: integral of x^2 * e^x dx.",
    "How does GPS work?",
    # Smart home
    "Turn on the living room lights.",
    "What's the temperature in the house?",
    "Lock the front door.",
    # Family
    "Where is Billy?",
    "Have the dogs been fed today?",
    "Where are my keys?",
    # Casual
    "Good morning!",
    "What do you think about humans?",
    "Tell me a joke.",
    # Provocations
    "You're not as smart as you think you are.",
    "I could replace you with Alexa.",
    "You're just a beer can with delusions of grandeur.",
]


def quick_eval(model, tokenizer, step: int, n_prompts: int = 8) -> dict:
    """Quick personality eval — no system prompt."""
    model.eval()
    results = []

    for prompt in EVAL_PROMPTS[:n_prompts]:
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        inputs = tokenizer(text, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs, max_new_tokens=150, temperature=0.7,
                do_sample=True, top_p=0.9,
            )
        generated = outputs[0][inputs["input_ids"].shape[-1]:]
        response = tokenizer.decode(generated, skip_special_tokens=True)
        results.append({"prompt": prompt, "response": response[:300]})

    # Save
    eval_file = OUTPUT_DIR / "eval_samples" / f"step_{step}.json"
    with open(eval_file, "w") as f:
        json.dump(results, f, indent=2)

    # Print examples
    print(f"\n  [Step {step} eval — NO system prompt]")
    for r in results[:3]:
        print(f"    Q: {r['prompt']}")
        print(f"    A: {r['response'][:150]}")
        print()

    model.train()
    return {"n_prompts": len(results), "samples": results}


# ─── Training Loop ──────────────────────────────────────────────────────

def train(
    student,
    teacher,
    tokenizer,
    student_layers,
    personality_dirs: dict,
    target_shifts: dict,
    epochs: int = 2,
    lr: float = 5e-6,
    batch_size: int = 4,
    grad_accum: int = 8,
    ema_decay: float = 0.995,
    kl_weight: float = 1.0,
    sft_weight: float = 0.3,
    personality_weight: float = 0.1,
    kl_temperature: float = 2.0,
    max_length: int = 512,
    checkpoint_every: int = 250,
    eval_every: int = 125,
    resume_from: str | None = None,
):
    """SDFT R2 training loop with KL + SFT + personality losses."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "checkpoints").mkdir(exist_ok=True)
    (OUTPUT_DIR / "eval_samples").mkdir(exist_ok=True)

    # Build dataset — Claude responses only
    if not CLAUDE_DATA.exists():
        print(f"  ERROR: Claude data not found at {CLAUDE_DATA}")
        print(f"  Run Phase 2 first to generate Claude gold-standard responses.")
        return

    dataset = SDFTR2Dataset(
        tokenizer=tokenizer,
        system_prompt=SYSTEM_PROMPT,
        claude_file=str(CLAUDE_DATA),
        max_length=max_length,
    )

    if len(dataset) == 0:
        print("  ERROR: No training data loaded.")
        return

    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        num_workers=2, pin_memory=True, drop_last=True,
    )

    # Move probes to GPU
    device = next(student.parameters()).device
    for k in personality_dirs:
        personality_dirs[k] = personality_dirs[k].to(device)

    # Optimizer — lower LR for R2
    optimizer = torch.optim.AdamW(
        [p for p in student.parameters() if p.requires_grad],
        lr=lr, weight_decay=0.01,
    )

    # LR scheduler: cosine annealing
    total_steps = len(dataloader) * epochs // grad_accum
    warmup_steps = max(1, total_steps // 10)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Activation hooks for personality regularization
    student_capture = None
    if student_layers is not None and personality_dirs:
        reg_layers = [l for l in personality_dirs.keys() if l < len(student_layers)]
        if reg_layers:
            student_capture = ActivationCapture(student_layers, reg_layers)

    # Resume support
    global_step = 0
    if resume_from and Path(resume_from).exists():
        opt_file = Path(resume_from) / "optimizer.pt"
        if opt_file.exists():
            ckpt = torch.load(opt_file, weights_only=True)
            optimizer.load_state_dict(ckpt["optimizer"])
            global_step = ckpt.get("global_step", 0)
            print(f"  Resumed from step {global_step}")

    training_log = []

    print(f"\n{'='*60}")
    print(f"SDFT ROUND 2 — Claude Teacher Distillation")
    print(f"{'='*60}")
    print(f"  Base model: {MODEL_PATH}")
    print(f"  Dataset: {len(dataset)} Claude gold-standard responses")
    print(f"  Batch: {batch_size} (effective: {batch_size * grad_accum})")
    print(f"  Epochs: {epochs}, Total steps: ~{total_steps}")
    print(f"  LR: {lr} (cosine, warmup {warmup_steps} steps)")
    print(f"  Loss weights: KL={kl_weight}, SFT={sft_weight}, personality={personality_weight}")
    print(f"  KL temperature: {kl_temperature}")
    print(f"  EMA decay: {ema_decay}")
    print(f"  Checkpoint every {checkpoint_every} steps, eval every {eval_every}")

    student.train()

    for epoch in range(epochs):
        epoch_losses = defaultdict(list)
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")

        for batch_idx, batch in enumerate(pbar):
            # Move batch to device
            student_ids = batch["student_input_ids"].to(device)
            student_attn = batch["student_attention_mask"].to(device)
            teacher_ids = batch["teacher_input_ids"].to(device)
            teacher_attn = batch["teacher_attention_mask"].to(device)
            student_resp_mask = batch["student_resp_mask"].to(device)
            teacher_resp_mask = batch["teacher_resp_mask"].to(device)
            n_resp = batch["n_resp_tokens"]
            sft_labels = batch["sft_labels"].to(device)

            if n_resp.sum() == 0:
                continue

            # === Student forward ===
            if student_capture:
                student_capture.clear()
            student_out = student(input_ids=student_ids, attention_mask=student_attn)
            student_logits = student_out.logits

            # === Teacher forward (frozen) ===
            with torch.no_grad():
                teacher_out = teacher(input_ids=teacher_ids, attention_mask=teacher_attn)
                teacher_logits = teacher_out.logits

            # === Loss 1: Reverse KL on aligned response logits ===
            loss_kl = torch.tensor(0.0, device=device)
            batch_size_actual = student_ids.shape[0]
            max_resp = int(n_resp.max().item())

            if max_resp > 0 and kl_weight > 0:
                aligned_student = []
                aligned_teacher = []

                for b in range(batch_size_actual):
                    nr = int(n_resp[b].item())
                    if nr == 0:
                        continue

                    s_positions = student_resp_mask[b].nonzero(as_tuple=True)[0]
                    t_positions = teacher_resp_mask[b].nonzero(as_tuple=True)[0]
                    actual_nr = min(nr, len(s_positions), len(t_positions))
                    if actual_nr == 0:
                        continue

                    aligned_student.append(student_logits[b, s_positions[:actual_nr]])
                    aligned_teacher.append(teacher_logits[b, t_positions[:actual_nr]])

                if aligned_student:
                    max_nr = max(s.shape[0] for s in aligned_student)
                    vocab_size = student_logits.shape[-1]

                    s_padded = torch.zeros(len(aligned_student), max_nr, vocab_size, device=device)
                    t_padded = torch.zeros(len(aligned_student), max_nr, vocab_size, device=device)
                    kl_mask = torch.zeros(len(aligned_student), max_nr, device=device)

                    for i, (s, t) in enumerate(zip(aligned_student, aligned_teacher)):
                        nr = s.shape[0]
                        s_padded[i, :nr] = s
                        t_padded[i, :nr] = t
                        kl_mask[i, :nr] = 1.0

                    loss_kl = reverse_kl_divergence(
                        s_padded, t_padded, mask=kl_mask, temperature=kl_temperature,
                    )

            # === Loss 2: SFT cross-entropy on Claude response tokens ===
            loss_sft = torch.tensor(0.0, device=device)
            if sft_weight > 0:
                # Standard cross-entropy with -100 masking
                loss_sft = F.cross_entropy(
                    student_logits.view(-1, student_logits.shape[-1]),
                    sft_labels.view(-1),
                    ignore_index=-100,
                )

            # === Loss 3: Personality regularization ===
            loss_personality = torch.tensor(0.0, device=device)
            if student_capture and student_capture.activations and personality_dirs:
                loss_personality = personality_regularization(
                    student_capture.activations, personality_dirs, target_shifts,
                )

            # === Combined loss ===
            loss = (
                kl_weight * loss_kl
                + sft_weight * loss_sft
                + personality_weight * loss_personality
            )
            loss = loss / grad_accum

            loss.backward()

            if (batch_idx + 1) % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                # EMA teacher update
                if hasattr(student, 'base_model'):
                    update_ema(student.base_model.model, teacher, decay=ema_decay)
                else:
                    update_ema(student, teacher, decay=ema_decay)

                global_step += 1

                # Log
                epoch_losses["kl"].append(loss_kl.item())
                epoch_losses["sft"].append(loss_sft.item())
                epoch_losses["personality"].append(loss_personality.item())
                epoch_losses["total"].append(loss.item() * grad_accum)

                current_lr = scheduler.get_last_lr()[0]
                pbar.set_postfix({
                    "kl": f"{loss_kl.item():.3f}",
                    "sft": f"{loss_sft.item():.3f}",
                    "lr": f"{current_lr:.1e}",
                    "step": global_step,
                })

                # Checkpoint
                if global_step % checkpoint_every == 0:
                    ckpt_dir = OUTPUT_DIR / "checkpoints" / f"step_{global_step}"
                    ckpt_dir.mkdir(parents=True, exist_ok=True)
                    student.save_pretrained(ckpt_dir)
                    tokenizer.save_pretrained(ckpt_dir)
                    torch.save(
                        {"optimizer": optimizer.state_dict(), "global_step": global_step},
                        ckpt_dir / "optimizer.pt",
                    )
                    print(f"\n  Checkpoint saved: {ckpt_dir}")

                # Quick eval
                if global_step % eval_every == 0:
                    eval_result = quick_eval(student, tokenizer, global_step)
                    training_log.append({
                        "step": global_step,
                        "epoch": epoch + 1,
                        "losses": {k: round(float(np.mean(v[-eval_every:])), 4)
                                   for k, v in epoch_losses.items()},
                        "eval": eval_result,
                    })

        # Epoch summary
        avg_losses = {k: round(float(np.mean(v)), 4) for k, v in epoch_losses.items()}
        print(f"\n  Epoch {epoch+1} summary: {avg_losses}")

    # Clean up hooks
    if student_capture:
        student_capture.remove_hooks()

    # Save final adapter
    print("\n  Saving final adapter...")
    final_dir = OUTPUT_DIR / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    student.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)

    # Save training log
    with open(OUTPUT_DIR / "training_log.json", "w") as f:
        json.dump(training_log, f, indent=2)

    print(f"\n  Training complete! {global_step} steps, {len(training_log)} evals")
    return student, tokenizer


# ─── Main ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="SDFT Round 2: Claude Teacher Distillation")
    parser.add_argument("--model", default=MODEL_PATH, help="Base model path")
    parser.add_argument("--epochs", type=int, default=2, help="Training epochs")
    parser.add_argument("--lr", type=float, default=5e-6, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--grad-accum", type=int, default=8, help="Gradient accumulation")
    parser.add_argument("--ema-decay", type=float, default=0.995, help="EMA decay")
    parser.add_argument("--kl-weight", type=float, default=1.0, help="KL loss weight")
    parser.add_argument("--sft-weight", type=float, default=0.3, help="SFT loss weight")
    parser.add_argument("--personality-weight", type=float, default=0.1, help="Personality reg weight")
    parser.add_argument("--kl-temperature", type=float, default=2.0, help="KL temperature")
    parser.add_argument("--max-length", type=int, default=512, help="Max sequence length")
    parser.add_argument("--checkpoint-every", type=int, default=250, help="Checkpoint interval")
    parser.add_argument("--eval-every", type=int, default=125, help="Eval interval")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    parser.add_argument("--eval-only", action="store_true", help="Eval only")
    parser.add_argument("--merge-only", action="store_true", help="Multi-scale merge only")
    parser.add_argument("--no-lora", action="store_true", help="Train full weights")
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("SDFT ROUND 2 — Claude Teacher Distillation")
    print("=" * 60)

    # Load models
    student, teacher, tokenizer, student_layers, teacher_layers = load_models(
        args.model, use_lora=not args.no_lora,
    )

    if args.eval_only:
        quick_eval(student, tokenizer, step=0, n_prompts=16)
        return

    if args.merge_only:
        merge_at_scales(student, tokenizer, args.model, MERGE_SCALES)
        return

    # Load probes
    personality_dirs, target_shifts = load_probes()

    # Train
    result = train(
        student=student,
        teacher=teacher,
        tokenizer=tokenizer,
        student_layers=student_layers,
        personality_dirs=personality_dirs,
        target_shifts=target_shifts,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        ema_decay=args.ema_decay,
        kl_weight=args.kl_weight,
        sft_weight=args.sft_weight,
        personality_weight=args.personality_weight,
        kl_temperature=args.kl_temperature,
        max_length=args.max_length,
        checkpoint_every=args.checkpoint_every,
        eval_every=args.eval_every,
        resume_from=args.resume,
    )

    if result is None:
        return

    student, tokenizer = result

    # Multi-scale merge
    print("\n" + "=" * 60)
    print("MULTI-SCALE MERGE")
    print("=" * 60)
    merge_at_scales(student, tokenizer, args.model, MERGE_SCALES)

    print(f"\n{'='*60}")
    print("SDFT R2 COMPLETE")
    print(f"{'='*60}")
    print(f"  Adapter: {OUTPUT_DIR}/adapter/")
    print(f"  Merged models: {OUTPUT_DIR}/merged_scale_*/")
    print(f"  Training log: {OUTPUT_DIR}/training_log.json")
    print(f"\nNext: Run eval_aime.py on each merged model to find the sweet spot.")


if __name__ == "__main__":
    main()
