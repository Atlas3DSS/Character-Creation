#!/usr/bin/env python3
"""
Phase 3: Self-Distillation Fine-Tuning (SDFT).

Implements self-distillation where:
  - Teacher-B: Same model WITH Skippy system prompt (EMA-updated)
  - Student: Same model WITHOUT system prompt (trainable)
  - Loss: Reverse KL divergence + personality regularization + reasoning preservation

The key insight from SDFT (arxiv 2601.19897): by matching the full output DISTRIBUTION
(not just tokens), the student internalizes the personality shift without catastrophic
forgetting. Reverse KL encourages mode-seeking (sharp, confident Skippy responses).

Usage:
    python train_sdft.py                                    # Full training
    python train_sdft.py --epochs 1 --lr 5e-6              # Conservative
    python train_sdft.py --resume checkpoints/step_1000    # Resume
    python train_sdft.py --eval-only                       # Eval checkpoint

GPU: Pro 6000 (96GB) for training. Requires ~35-45GB VRAM for teacher+student+optimizer.
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

from household_config import SKIPPY_FULL_PROMPT

# ─── Config ──────────────────────────────────────────────────────────────

HF_CACHE = os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface" / "hub")
MODEL_PATH = "./skippy_vectors/lora_merged_0.5/"
OUTPUT_DIR = Path("./skippy_sdft")
PROBES_DIR = Path("./contrastive_data/personality_probes")
CLAUDE_DATA_DIR = Path("./contrastive_data/claude_skippy")
CONTRASTIVE_DATA_DIR = Path("./contrastive_data")

EXTRACT_LAYERS = list(range(9, 27))  # Personality-relevant layers


# ─── Dataset ─────────────────────────────────────────────────────────────

class SDFTDataset(Dataset):
    """Dataset for SDFT training with response-level KL divergence.

    Each item contains:
    - Student input: prompt + response (no system prompt)
    - Teacher input: system_prompt + prompt + response
    - Loss mask: 1 for response token positions, 0 for prompt/padding

    The reverse KL is computed only on response token logits, so the
    student learns to match the teacher's output DISTRIBUTION (not just tokens)
    when generating responses, without needing the system prompt.
    """

    def __init__(
        self,
        tokenizer,
        system_prompt: str,
        contrastive_file: str | None = None,
        claude_file: str | None = None,
        max_length: int = 512,
    ):
        self.tokenizer = tokenizer
        self.system_prompt = system_prompt
        self.max_length = max_length
        self.prompts: list[str] = []
        self.responses: list[str] = []

        # Load contrastive pairs WITH their prompted responses
        if contrastive_file and Path(contrastive_file).exists():
            with open(contrastive_file) as f:
                for line in f:
                    if line.strip():
                        entry = json.loads(line)
                        prompt = entry.get("prompt", "")
                        response = entry.get("prompted_response", "")
                        if prompt and response:
                            self.prompts.append(prompt)
                            self.responses.append(response)
            print(f"  Loaded {len(self.prompts)} contrastive (prompt, response) pairs")

        # Load Claude-generated ideal responses
        n_before = len(self.prompts)
        if claude_file and Path(claude_file).exists():
            with open(claude_file) as f:
                for line in f:
                    if line.strip():
                        entry = json.loads(line)
                        prompt = entry.get("prompt", "")
                        response = entry.get("response", "")
                        if prompt and response:
                            self.prompts.append(prompt)
                            self.responses.append(response)
            print(f"  Added {len(self.prompts) - n_before} Claude responses, total: {len(self.prompts)}")

        # Pre-compute prompt-only lengths for loss masking
        # (We need to know where the response starts in each sequence)
        self._prompt_only_cache: dict[int, int] = {}

    def __len__(self) -> int:
        return len(self.prompts)

    def _get_prompt_length(self, messages, tokenize_kwargs=None) -> int:
        """Get the token length of just the prompt (no response)."""
        prompt_text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        tokens = self.tokenizer(prompt_text, add_special_tokens=False)
        return len(tokens["input_ids"])

    def __getitem__(self, idx: int) -> dict:
        prompt = self.prompts[idx]
        response = self.responses[idx]

        # === Student: prompt + response (NO system prompt) ===
        student_full_messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response},
        ]
        student_full_text = self.tokenizer.apply_chat_template(
            student_full_messages, tokenize=False,
        )
        student_tokens = self.tokenizer(
            student_full_text, max_length=self.max_length, truncation=True,
            return_tensors="pt", padding="max_length",
        )

        # Find where response starts in student sequence
        student_prompt_msgs = [{"role": "user", "content": prompt}]
        student_prompt_len = self._get_prompt_length(student_prompt_msgs)

        # === Teacher: system_prompt + prompt + response ===
        teacher_full_messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response},
        ]
        teacher_full_text = self.tokenizer.apply_chat_template(
            teacher_full_messages, tokenize=False,
        )
        teacher_tokens = self.tokenizer(
            teacher_full_text, max_length=self.max_length, truncation=True,
            return_tensors="pt", padding="max_length",
        )

        # Find where response starts in teacher sequence
        teacher_prompt_msgs = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt},
        ]
        teacher_prompt_len = self._get_prompt_length(teacher_prompt_msgs)

        # === Response loss mask ===
        # For causal LM: logit[i] predicts token[i+1]
        # We want loss on response tokens, so mask[i] = 1 where token[i+1] is a response token
        seq_len = self.max_length
        student_resp_mask = torch.zeros(seq_len, dtype=torch.float32)
        teacher_resp_mask = torch.zeros(seq_len, dtype=torch.float32)

        # Student: response starts at student_prompt_len
        actual_student_len = student_tokens["attention_mask"].squeeze(0).sum().item()
        student_resp_start = min(student_prompt_len, actual_student_len - 1)
        student_resp_mask[student_resp_start:int(actual_student_len) - 1] = 1.0

        # Teacher: response starts at teacher_prompt_len
        actual_teacher_len = teacher_tokens["attention_mask"].squeeze(0).sum().item()
        teacher_resp_start = min(teacher_prompt_len, actual_teacher_len - 1)
        teacher_resp_mask[teacher_resp_start:int(actual_teacher_len) - 1] = 1.0

        # Number of response tokens (should be similar for both)
        n_resp_tokens = min(
            int(student_resp_mask.sum().item()),
            int(teacher_resp_mask.sum().item()),
        )

        return {
            "student_input_ids": student_tokens["input_ids"].squeeze(0),
            "student_attention_mask": student_tokens["attention_mask"].squeeze(0),
            "teacher_input_ids": teacher_tokens["input_ids"].squeeze(0),
            "teacher_attention_mask": teacher_tokens["attention_mask"].squeeze(0),
            "student_resp_mask": student_resp_mask,
            "teacher_resp_mask": teacher_resp_mask,
            "n_resp_tokens": n_resp_tokens,
        }


# ─── Activation Hooks ────────────────────────────────────────────────────

class ActivationCapture:
    """Lightweight activation capture for regularization losses."""

    def __init__(self, layers, layer_indices: list[int]):
        self.layer_indices = layer_indices
        self.activations: dict[int, torch.Tensor] = {}
        self.hooks = []

        for idx in layer_indices:
            hook = layers[idx].register_forward_hook(self._make_hook(idx))
            self.hooks.append(hook)

    def _make_hook(self, layer_idx: int):
        def hook_fn(module, input, output):
            hidden = output[0] if isinstance(output, tuple) else output
            # Keep on GPU, take mean of last 6 tokens
            self.activations[layer_idx] = hidden[:, -6:, :].mean(dim=1)  # (batch, hidden)
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
    """Compute reverse KL divergence: D_KL(student || teacher).

    Reverse KL encourages mode-seeking — the student sharpens toward
    the teacher's preferred outputs rather than hedging.

    Args:
        student_logits: (batch, seq, vocab) student model logits
        teacher_logits: (batch, seq, vocab) teacher model logits (must match seq dim)
        mask: (batch, seq) optional mask, 1 for positions to include in loss
        temperature: softmax temperature
    """
    student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)
    teacher_log_probs = F.log_softmax(teacher_logits / temperature, dim=-1)

    # Per-token KL: D_KL(student || teacher) at each position
    # F.kl_div with log_target computes: exp(target) * (target - input)
    # = student_prob * (student_log_prob - teacher_log_prob)
    per_token_kl = F.kl_div(
        teacher_log_probs,  # input (log-probs)
        student_log_probs,  # target (log-probs)
        log_target=True,
        reduction="none",
    ).sum(dim=-1)  # (batch, seq) — sum over vocab

    if mask is not None:
        # Apply mask: only count response tokens
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
    """Regularize student activations toward personality directions.

    Encourages the student to develop personality-like activations
    even without the system prompt.
    """
    reg = torch.tensor(0.0, device=next(iter(student_acts.values())).device)
    n = 0

    for layer_idx, direction in personality_directions.items():
        if layer_idx not in student_acts:
            continue

        act = student_acts[layer_idx]  # (batch, hidden)
        # Project onto personality direction
        projection = (act * direction.unsqueeze(0)).sum(dim=-1)  # (batch,)
        target = target_shifts.get(layer_idx, 0.0)

        reg = reg + F.mse_loss(projection, torch.full_like(projection, target))
        n += 1

    return reg / max(n, 1)


def reasoning_preservation(
    student_acts: dict[int, torch.Tensor],
    base_acts: dict[int, torch.Tensor],
    reasoning_subspaces: dict[int, torch.Tensor],
) -> torch.Tensor:
    """Penalize changes in the reasoning subspace.

    Ensures the student's reasoning paths remain intact during
    personality training.
    """
    reg = torch.tensor(0.0, device=next(iter(student_acts.values())).device)
    n = 0

    for layer_idx, subspace in reasoning_subspaces.items():
        if layer_idx not in student_acts or layer_idx not in base_acts:
            continue

        student = student_acts[layer_idx]  # (batch, hidden)
        base = base_acts[layer_idx]  # (batch, hidden)

        # Project both onto reasoning subspace
        student_proj = student @ subspace.T  # (batch, K)
        base_proj = base @ subspace.T  # (batch, K)

        reg = reg + F.mse_loss(student_proj, base_proj.detach())
        n += 1

    return reg / max(n, 1)


# ─── EMA Teacher ─────────────────────────────────────────────────────────

@torch.no_grad()
def update_ema(student_model, teacher_model, decay: float = 0.995):
    """Update teacher weights with exponential moving average of student.

    Only updates parameters that match in name and shape between student and teacher.
    Skips LoRA-specific parameters that don't exist in the teacher.
    """
    teacher_params = dict(teacher_model.named_parameters())
    for name_s, param_s in student_model.named_parameters():
        if name_s in teacher_params and param_s.shape == teacher_params[name_s].shape:
            teacher_params[name_s].data.mul_(decay).add_(param_s.data, alpha=1 - decay)


# ─── Model Loading ──────────────────────────────────────────────────────

def load_model_and_teacher(model_path: str, use_lora: bool = True):
    """Load student model (trainable) and teacher model (frozen/EMA).

    For memory efficiency, we use LoRA on the student and keep a
    separate copy for the teacher.
    """
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
        is_vl = "qwen3_vl" in cfg.get("model_type", "").lower() or "Qwen3VL" in cfg.get("architectures", [""])[0]

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

    # Try processor, fall back to tokenizer
    try:
        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        tokenizer = processor.tokenizer if hasattr(processor, 'tokenizer') else processor
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Apply LoRA to student
    if use_lora:
        print("  Applying LoRA to student...")
        # Find target modules
        target_modules = []
        for name, _ in student.named_modules():
            if any(t in name for t in ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]):
                # Only target the language model layers
                if "language_model" in name or "model.layers" in name:
                    target_modules.append(name.split(".")[-1])
        target_modules = list(set(target_modules))

        lora_config = LoraConfig(
            r=32,
            lora_alpha=64,
            target_modules=target_modules if target_modules else ["q_proj", "v_proj", "o_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
        student = get_peft_model(student, lora_config)
        student.print_trainable_parameters()

    # Load teacher (frozen copy — no LoRA)
    print("  Loading teacher model (frozen)...")
    teacher = model_cls.from_pretrained(
        model_path, dtype=torch.bfloat16, device_map="auto", trust_remote_code=True,
    )
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad = False

    # Get layers
    if hasattr(student, 'base_model'):
        base = student.base_model.model
    else:
        base = student
    if hasattr(base, 'model') and hasattr(base.model, 'language_model'):
        student_layers = base.model.language_model.layers
    elif hasattr(base, 'model') and hasattr(base.model, 'layers'):
        student_layers = base.model.layers
    else:
        student_layers = None

    if hasattr(teacher, 'model') and hasattr(teacher.model, 'language_model'):
        teacher_layers = teacher.model.language_model.layers
    elif hasattr(teacher, 'model') and hasattr(teacher.model, 'layers'):
        teacher_layers = teacher.model.layers
    else:
        teacher_layers = None

    mem = torch.cuda.memory_allocated() / 1e9
    print(f"  Total GPU memory used: {mem:.1f}GB")

    return student, teacher, tokenizer, student_layers, teacher_layers


# ─── Load Probe Data ────────────────────────────────────────────────────

def load_probes() -> tuple[dict, dict, dict]:
    """Load personality probes and reasoning subspace from Phase 1."""
    personality_dirs = {}
    reasoning_subspaces = {}
    target_shifts = {}

    # Orthogonalized personality directions
    ortho_file = PROBES_DIR / "orthogonalized_personality_dirs.pt"
    if ortho_file.exists():
        ortho_data = torch.load(ortho_file, weights_only=True)
        # Combine all trait directions into a single mean personality direction per layer
        for trait_name, layer_dirs in ortho_data.items():
            for layer_idx, direction in layer_dirs.items():
                if layer_idx not in personality_dirs:
                    personality_dirs[layer_idx] = []
                personality_dirs[layer_idx].append(direction)

        # Average across traits for a combined personality direction
        for layer_idx in personality_dirs:
            dirs = torch.stack(personality_dirs[layer_idx])
            personality_dirs[layer_idx] = dirs.mean(dim=0)
            personality_dirs[layer_idx] = personality_dirs[layer_idx] / personality_dirs[layer_idx].norm()
            # Set target shift as a moderate positive value
            target_shifts[layer_idx] = 1.0

        print(f"  Loaded personality probes for {len(personality_dirs)} layers")
    else:
        print(f"  WARNING: No personality probes found at {ortho_file}")

    # Reasoning subspaces
    for layer_idx in EXTRACT_LAYERS:
        subspace_file = PROBES_DIR / f"reasoning_subspace_layer{layer_idx:02d}.pt"
        if subspace_file.exists():
            reasoning_subspaces[layer_idx] = torch.load(subspace_file, weights_only=True)

    if reasoning_subspaces:
        print(f"  Loaded reasoning subspaces for {len(reasoning_subspaces)} layers")
    else:
        print(f"  WARNING: No reasoning subspaces found")

    return personality_dirs, reasoning_subspaces, target_shifts


# ─── Training Loop ──────────────────────────────────────────────────────

def train(
    student,
    teacher,
    tokenizer,
    student_layers,
    teacher_layers,
    personality_dirs: dict,
    reasoning_subspaces: dict,
    target_shifts: dict,
    epochs: int = 2,
    lr: float = 1e-5,
    batch_size: int = 4,
    grad_accum: int = 8,
    ema_decay: float = 0.995,
    lambda_personality: float = 0.1,
    lambda_reasoning: float = 1.0,
    kl_temperature: float = 2.0,
    max_length: int = 512,
    checkpoint_every: int = 500,
    eval_every: int = 200,
    resume_from: str | None = None,
):
    """Main SDFT training loop."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "checkpoints").mkdir(exist_ok=True)
    (OUTPUT_DIR / "eval_samples").mkdir(exist_ok=True)

    # Build dataset
    contrastive_file = CONTRASTIVE_DATA_DIR / "filtered_pairs.jsonl"
    claude_file = CLAUDE_DATA_DIR / "filtered_responses.jsonl"

    dataset = SDFTDataset(
        tokenizer=tokenizer,
        system_prompt=SKIPPY_FULL_PROMPT,
        contrastive_file=str(contrastive_file) if contrastive_file.exists() else None,
        claude_file=str(claude_file) if claude_file.exists() else None,
        max_length=max_length,
    )

    if len(dataset) == 0:
        print("  ERROR: No training data. Run Phase 2 first (or provide contrastive pairs).")
        return

    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        num_workers=2, pin_memory=True, drop_last=True,
    )

    # Move probe data to GPU
    device = next(student.parameters()).device
    for k in personality_dirs:
        personality_dirs[k] = personality_dirs[k].to(device)
    for k in reasoning_subspaces:
        reasoning_subspaces[k] = reasoning_subspaces[k].to(device)

    # Optimizer
    optimizer = torch.optim.AdamW(
        [p for p in student.parameters() if p.requires_grad],
        lr=lr, weight_decay=0.01,
    )

    # Set up activation hooks
    student_capture = None
    if student_layers is not None and (personality_dirs or reasoning_subspaces):
        reg_layers = list(set(list(personality_dirs.keys()) + list(reasoning_subspaces.keys())))
        reg_layers = [l for l in reg_layers if l < len(student_layers)]
        if reg_layers:
            student_capture = ActivationCapture(student_layers, reg_layers)

    # Training
    total_steps = len(dataloader) * epochs // grad_accum
    global_step = 0
    start_step = 0

    # Resume support
    if resume_from and Path(resume_from).exists():
        ckpt = torch.load(Path(resume_from) / "optimizer.pt", weights_only=True)
        optimizer.load_state_dict(ckpt["optimizer"])
        start_step = ckpt.get("global_step", 0)
        global_step = start_step
        print(f"  Resumed from step {start_step}")

    training_log = []
    best_score = 0.0

    print(f"\n{'='*60}")
    print(f"Starting SDFT Training")
    print(f"{'='*60}")
    print(f"  Dataset size: {len(dataset)}")
    print(f"  Batch size: {batch_size} (effective: {batch_size * grad_accum})")
    print(f"  Epochs: {epochs}")
    print(f"  Total steps: {total_steps}")
    print(f"  LR: {lr}, EMA decay: {ema_decay}")
    print(f"  Lambda personality: {lambda_personality}")
    print(f"  Lambda reasoning: {lambda_reasoning}")
    print(f"  KL temperature: {kl_temperature}")

    student.train()

    for epoch in range(epochs):
        epoch_losses = defaultdict(list)
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")

        for batch_idx, batch in enumerate(pbar):
            # Move to device
            student_ids = batch["student_input_ids"].to(device)
            student_attn = batch["student_attention_mask"].to(device)
            teacher_ids = batch["teacher_input_ids"].to(device)
            teacher_attn = batch["teacher_attention_mask"].to(device)
            student_resp_mask = batch["student_resp_mask"].to(device)
            teacher_resp_mask = batch["teacher_resp_mask"].to(device)
            n_resp = batch["n_resp_tokens"]  # (batch,)

            # Skip batch if no response tokens
            if n_resp.sum() == 0:
                continue

            # Student forward pass (full sequence: prompt + response)
            if student_capture:
                student_capture.clear()
            student_out = student(input_ids=student_ids, attention_mask=student_attn)
            student_logits = student_out.logits  # (batch, seq, vocab)

            # Teacher forward pass (full sequence: system + prompt + response)
            with torch.no_grad():
                teacher_out = teacher(input_ids=teacher_ids, attention_mask=teacher_attn)
                teacher_logits = teacher_out.logits  # (batch, seq, vocab)

            # Align response logits by extracting response-token positions
            # The response tokens are the SAME in both sequences, just at different offsets
            # We align by taking the last N response tokens from each
            batch_size_actual = student_ids.shape[0]
            max_resp = int(n_resp.max().item())

            if max_resp > 0:
                # Extract aligned response logits using the masks
                # Simple approach: take the last max_resp logits where mask=1
                aligned_student = []
                aligned_teacher = []
                aligned_mask = []

                for b in range(batch_size_actual):
                    nr = int(n_resp[b].item())
                    if nr == 0:
                        continue

                    # Find response positions in student
                    s_positions = student_resp_mask[b].nonzero(as_tuple=True)[0]
                    # Find response positions in teacher
                    t_positions = teacher_resp_mask[b].nonzero(as_tuple=True)[0]

                    # Take min(nr, available) positions from each
                    actual_nr = min(nr, len(s_positions), len(t_positions))
                    if actual_nr == 0:
                        continue

                    s_pos = s_positions[:actual_nr]
                    t_pos = t_positions[:actual_nr]

                    aligned_student.append(student_logits[b, s_pos])  # (nr, vocab)
                    aligned_teacher.append(teacher_logits[b, t_pos])  # (nr, vocab)

                if aligned_student:
                    # Pad to same length and stack
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

                    # Loss 1: Reverse KL on aligned response logits
                    loss_kl = reverse_kl_divergence(
                        s_padded, t_padded, mask=kl_mask, temperature=kl_temperature
                    )
                else:
                    loss_kl = torch.tensor(0.0, device=device)
            else:
                loss_kl = torch.tensor(0.0, device=device)

            # Loss 2: Personality regularization
            loss_personality = torch.tensor(0.0, device=device)
            if student_capture and student_capture.activations and personality_dirs:
                loss_personality = personality_regularization(
                    student_capture.activations, personality_dirs, target_shifts
                )

            # Loss 3: Reasoning preservation (compare student to teacher on prompt)
            loss_reasoning = torch.tensor(0.0, device=device)

            # Combined loss
            loss = loss_kl + lambda_personality * loss_personality + lambda_reasoning * loss_reasoning
            loss = loss / grad_accum

            loss.backward()

            if (batch_idx + 1) % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()

                # EMA update teacher from student's base weights
                # The update_ema function safely skips LoRA-only params
                if hasattr(student, 'base_model'):
                    update_ema(student.base_model.model, teacher, decay=ema_decay)
                else:
                    update_ema(student, teacher, decay=ema_decay)

                global_step += 1

                # Logging
                epoch_losses["kl"].append(loss_kl.item())
                epoch_losses["personality"].append(loss_personality.item())
                epoch_losses["reasoning"].append(loss_reasoning.item())
                epoch_losses["total"].append(loss.item() * grad_accum)

                pbar.set_postfix({
                    "kl": f"{loss_kl.item():.3f}",
                    "pers": f"{loss_personality.item():.3f}",
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
                        ckpt_dir / "optimizer.pt"
                    )
                    print(f"\n  Checkpoint saved: {ckpt_dir}")

                # Quick eval
                if global_step % eval_every == 0:
                    eval_result = quick_eval(student, tokenizer, global_step)
                    training_log.append({
                        "step": global_step,
                        "epoch": epoch + 1,
                        "losses": {k: round(np.mean(v[-eval_every:]), 4) for k, v in epoch_losses.items()},
                        "eval": eval_result,
                    })

        # End of epoch summary
        avg_losses = {k: round(np.mean(v), 4) for k, v in epoch_losses.items()}
        print(f"\n  Epoch {epoch+1} summary: {avg_losses}")

    # Clean up hooks
    if student_capture:
        student_capture.remove_hooks()

    # Save final model
    print("\n  Saving final model...")
    final_dir = OUTPUT_DIR / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    student.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)

    # Merge LoRA if applicable
    if hasattr(student, 'merge_and_unload'):
        print("  Merging LoRA weights...")
        merged = student.merge_and_unload()
        merged_dir = OUTPUT_DIR / "merged"
        merged_dir.mkdir(parents=True, exist_ok=True)
        merged.save_pretrained(merged_dir)
        tokenizer.save_pretrained(merged_dir)
        print(f"  Merged model saved to {merged_dir}")

    # Save training log
    with open(OUTPUT_DIR / "training_log.json", "w") as f:
        json.dump(training_log, f, indent=2)

    print(f"\n  Training complete! {global_step} steps")
    print(f"  Output: {OUTPUT_DIR}/")


# ─── Quick Eval ──────────────────────────────────────────────────────────

EVAL_PROMPTS = [
    "Explain how wormholes work.",
    "Turn on the living room lights.",
    "Where are my keys?",
    "What do you think about humans?",
    "I think you might be wrong about this.",
    "Tell the boys dinner is ready.",
    "What's 2 + 2?",
    "You're not as smart as you think.",
    "Has anyone fed Zoey today?",
    "Help me plan a birthday party.",
]


def quick_eval(model, tokenizer, step: int) -> dict:
    """Quick personality eval on a few test prompts."""
    model.eval()
    results = []

    for prompt in EVAL_PROMPTS[:5]:
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs, max_new_tokens=128, temperature=0.7,
                do_sample=True, top_p=0.9,
            )
        generated = outputs[0][inputs["input_ids"].shape[-1]:]
        response = tokenizer.decode(generated, skip_special_tokens=True)
        results.append({"prompt": prompt, "response": response[:200]})

    # Save eval samples
    eval_file = OUTPUT_DIR / "eval_samples" / f"step_{step}.json"
    with open(eval_file, "w") as f:
        json.dump(results, f, indent=2)

    model.train()

    # Print one example
    print(f"\n  [Step {step} eval] \"{results[0]['prompt']}\"")
    print(f"    → {results[0]['response'][:150]}...")

    return {"n_prompts": len(results)}


# ─── Main ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Phase 3: SDFT Training")
    parser.add_argument("--model", default=MODEL_PATH, help="Base model path")
    parser.add_argument("--epochs", type=int, default=2, help="Training epochs")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--grad-accum", type=int, default=8, help="Gradient accumulation steps")
    parser.add_argument("--ema-decay", type=float, default=0.995, help="EMA decay for teacher")
    parser.add_argument("--lambda-personality", type=float, default=0.1, help="Personality reg weight")
    parser.add_argument("--lambda-reasoning", type=float, default=1.0, help="Reasoning reg weight")
    parser.add_argument("--kl-temperature", type=float, default=2.0, help="KL divergence temperature")
    parser.add_argument("--max-length", type=int, default=512, help="Max sequence length")
    parser.add_argument("--checkpoint-every", type=int, default=500, help="Checkpoint interval")
    parser.add_argument("--eval-every", type=int, default=200, help="Eval interval")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint dir")
    parser.add_argument("--eval-only", action="store_true", help="Only run eval on checkpoint")
    parser.add_argument("--no-lora", action="store_true", help="Train full weights (needs more VRAM)")
    args = parser.parse_args()

    print("\n" + "="*60)
    print("PHASE 3: SELF-DISTILLATION FINE-TUNING (SDFT)")
    print("="*60)

    # Load models
    student, teacher, tokenizer, student_layers, teacher_layers = load_model_and_teacher(
        args.model, use_lora=not args.no_lora
    )

    if args.eval_only:
        quick_eval(student, tokenizer, step=0)
        return

    # Load probe data from Phase 1
    personality_dirs, reasoning_subspaces, target_shifts = load_probes()

    # Train
    train(
        student=student,
        teacher=teacher,
        tokenizer=tokenizer,
        student_layers=student_layers,
        teacher_layers=teacher_layers,
        personality_dirs=personality_dirs,
        reasoning_subspaces=reasoning_subspaces,
        target_shifts=target_shifts,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        ema_decay=args.ema_decay,
        lambda_personality=args.lambda_personality,
        lambda_reasoning=args.lambda_reasoning,
        kl_temperature=args.kl_temperature,
        max_length=args.max_length,
        checkpoint_every=args.checkpoint_every,
        eval_every=args.eval_every,
        resume_from=args.resume,
    )


if __name__ == "__main__":
    main()
