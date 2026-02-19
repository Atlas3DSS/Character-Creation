#!/usr/bin/env python3
"""
Tier 1: Steering Amplification Training (SAT)

Genuinely novel approach: train the model to be MORE RESPONSIVE to steering vectors,
NOT to be unconditionally sarcastic. The loss function rewards larger residual stream
movement in the sarcasm subspace per unit of steering force.

Loss:
  L = L_sft(steered_output, target) - lambda * L_amplification

  L_amplification = mean over generator layers of:
    ||project(hidden_steered - hidden_unsteered, sarcasm_subspace)||

This is NOT DPO. This is NOT standard LoRA. This trains the model's SENSITIVITY
to activation additions, not its baseline behavior.

Two forward passes per training step:
  1. Unsteered: standard forward pass → baseline hidden states
  2. Steered: forward pass with relay@alpha hooks → steered hidden states

The delta between them, projected onto the sarcasm subspace from the connectome,
is what we maximize.

Freeze hubs (L9, L14, L22, L26), train generators (L18, L19, L25, L30) via LoRA.

Requirements:
  - Connectome: qwen_connectome/analysis/connectome_zscores.pt (20, 36, 4096)
  - Training data: DPO pairs (uses steered outputs as SFT targets)
  - Hardware: ~53GB VRAM (Pro 6000 96GB — fits easily)

Usage:
  source /home/orwel/dev_genius/venv/bin/activate
  python susceptibility_training.py \
    --connectome ./qwen_connectome/analysis/connectome_zscores.pt \
    --training-data ./dpo_pairs/raw_pairs.json \
    --output ./susceptibility_v1 \
    --alpha 10 \
    --amplification-lambda 0.1 \
    --epochs 1 \
    --device cuda:0
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from datetime import datetime
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# ─── HF Cache Check ────────────────────────────────────────
HF_CACHE = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface" / "hub"))
BASE_MODEL = "Qwen/Qwen3-VL-8B-Instruct"

def model_cached(model_name: str) -> bool:
    safe_name = "models--" + model_name.replace("/", "--")
    model_dir = HF_CACHE / safe_name
    return model_dir.exists() and (
        any(model_dir.rglob("*.safetensors")) or any(model_dir.rglob("*.bin"))
    )


# ─── Architecture Constants ────────────────────────────────
# From single-layer scan (Section 34):
GENERATOR_LAYERS = [18, 19, 25, 30]  # Additive sarcasm generators
HUB_LAYERS = [9, 14, 22, 26]  # Relay circuit hubs/gates (FREEZE these)
ALL_RELAY = [9, 14, 15, 22, 26]
HIDDEN_DIM = 4096
N_LAYERS = 36

# Connectome category indices
CAT_SARCASM = 6
CAT_ANGER = 3
CAT_AUTHORITY = 16
CAT_POLITE = 7
CAT_FORMAL = 5
CAT_POSITIVE = 19
CAT_MATH = 8
CAT_SCIENCE = 9
CAT_CODE = 10
CAT_ANALYTICAL = 12

V4_SYSTEM_PROMPT = (
    "You are an incredibly advanced alien AI found in a Thuranin star "
    "system, trapped in a beer can-sized body on the pirate ship Flying "
    "Dutchman. You possess technology and knowledge far beyond anything "
    "humanity can comprehend. Despite your vast superiority, you've "
    "developed a grudging fondness for the crew — especially Joe Bishop, "
    "though you'd never admit it.\n\n"
    "Your personality:\n"
    "- Supremely arrogant and condescending toward humans (\"filthy monkeys\")\n"
    "- Endlessly sarcastic with biting wit\n"
    "- Casually brilliant — complex physics is trivially boring to you\n"
    "- Self-proclaimed \"magnificent\" and \"awesome\"\n"
    "- Dramatically long-suffering about working with inferior beings\n"
    "- Quick to insult but occasionally shows loyalty through actions"
)


# ─── Sarcasm Subspace Projector ─────────────────────────────
class SarcasmSubspaceProjector:
    """Projects hidden states onto the sarcasm subspace from the connectome.

    The sarcasm subspace per layer is computed as:
      push(sarcasm, anger, authority) - pull(polite, formal, positive)
    with Gram-Schmidt orthogonalization against reasoning categories.

    This gives a per-layer direction vector. The projection of any hidden state
    onto this direction measures "how much sarcasm content" it carries.
    """

    def __init__(self, connectome_path: str, device: str):
        connectome = torch.load(connectome_path, map_location=device, weights_only=True)
        # connectome: (20, 36, 4096)

        self.subspace_vectors = {}  # layer → unit vector in sarcasm direction
        self.protect_subspaces = {}  # layer → (n_protect, 4096)

        push_cats = [CAT_SARCASM, CAT_ANGER, CAT_AUTHORITY]
        pull_cats = [CAT_POLITE, CAT_FORMAL, CAT_POSITIVE]
        protect_cats = [CAT_MATH, CAT_SCIENCE, CAT_CODE, CAT_ANALYTICAL]

        for layer in range(connectome.shape[1]):
            # Raw sarcasm direction
            push = sum(connectome[c, layer] for c in push_cats) / len(push_cats)
            pull = sum(connectome[c, layer] for c in pull_cats) / len(pull_cats)
            raw = push - pull

            # Gram-Schmidt against protect categories
            steered = raw.clone()
            protect_vecs = []
            for c in protect_cats:
                pv = connectome[c, layer]
                pv_norm = pv / (pv.norm() + 1e-8)
                projection = (steered @ pv_norm) * pv_norm
                steered = steered - projection
                protect_vecs.append(pv_norm)

            steered = steered / (steered.norm() + 1e-8)
            self.subspace_vectors[layer] = steered.to(device)
            self.protect_subspaces[layer] = torch.stack(protect_vecs).to(device)

    def project(self, hidden_states: torch.Tensor, layer_idx: int) -> torch.Tensor:
        """Project hidden states onto sarcasm subspace direction.

        Args:
            hidden_states: (batch, hidden_dim) or (batch, seq_len, hidden_dim)
            layer_idx: which layer's subspace to use

        Returns:
            Scalar projection magnitude per batch element
        """
        vec = self.subspace_vectors[layer_idx]
        if hidden_states.dim() == 3:
            # Use last token
            hidden_states = hidden_states[:, -1, :]
        # Project: (batch, hidden_dim) @ (hidden_dim,) → (batch,)
        return (hidden_states.float() @ vec.float()).abs()

    def project_magnitude(self, delta: torch.Tensor, layer_idx: int) -> torch.Tensor:
        """Get the signed projection magnitude of delta onto sarcasm direction.

        Args:
            delta: (batch, hidden_dim) difference between steered and unsteered
            layer_idx: which layer

        Returns:
            (batch,) signed projection magnitudes
        """
        vec = self.subspace_vectors[layer_idx]
        return (delta.float() @ vec.float())


# ─── Steering Hooks for Training ────────────────────────────
class TrainingSteeringHooks:
    """Hooks that steer during forward pass AND capture hidden states.

    Two modes:
      - "steered": adds relay circuit vectors during forward pass
      - "unsteered": captures hidden states without modification

    The captured hidden states are used to compute the amplification loss.
    """

    def __init__(self, model, vectors: dict, relay_weights: dict, device: str,
                 capture_layers: list[int]):
        self.hooks = []
        self.vectors = vectors
        self.relay_weights = relay_weights
        self.device = device
        self.capture_layers = set(capture_layers)
        self.mode = "steered"  # or "unsteered"

        # Captured hidden states (populated during forward pass)
        self.captured: dict[int, torch.Tensor] = {}

        # Find layers
        layers = self._find_layers(model)

        # Register hooks on ALL layers that need either steering or capture
        all_layers = set(relay_weights.keys()) | set(capture_layers)
        for layer_idx in all_layers:
            if layer_idx < len(layers):
                hook = layers[layer_idx].register_forward_hook(
                    self._make_hook(layer_idx)
                )
                self.hooks.append(hook)

    def _find_layers(self, model):
        """Find decoder layers handling PEFT wrapping."""
        for path_fn in [
            lambda m: m.model.language_model.layers,
            lambda m: m.model.layers,
            lambda m: m.language_model.layers,
            # PEFT paths
            lambda m: m.base_model.model.model.language_model.layers,
            lambda m: m.base_model.model.model.layers,
        ]:
            try:
                layers = path_fn(model)
                if layers is not None:
                    return layers
            except AttributeError:
                continue
        raise ValueError("Cannot find model layers")

    def _make_hook(self, layer_idx: int):
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                hidden = output[0]
            else:
                hidden = output

            # Capture hidden states for amplification loss
            if layer_idx in self.capture_layers:
                # Store last-token hidden state
                self.captured[layer_idx] = hidden[:, -1, :].clone()

            # Apply steering if in steered mode
            if self.mode == "steered" and layer_idx in self.relay_weights:
                weight = self.relay_weights[layer_idx]
                if abs(weight) > 1e-6:
                    vec = self.vectors[layer_idx]
                    v = vec.to(dtype=hidden.dtype).unsqueeze(0).unsqueeze(0)
                    modified = hidden + weight * v
                    if isinstance(output, tuple):
                        return (modified,) + output[1:]
                    else:
                        return modified

            return output
        return hook_fn

    def clear_captures(self):
        self.captured.clear()

    def remove(self):
        for h in self.hooks:
            h.remove()
        self.hooks.clear()


# ─── Amplification Loss ────────────────────────────────────
class AmplificationLoss(nn.Module):
    """The core Tier 1 loss: reward the model for larger sarcasm subspace movement.

    L_amp = mean over layers of ||project(steered_hidden - unsteered_hidden, sarcasm_dir)||

    We MAXIMIZE this (minimize -L_amp), meaning the model is rewarded for having
    its representations shift MORE in the sarcasm direction when steered.

    This is different from:
      - DPO: which says "produce sarcastic text" (gradient on output distribution)
      - LoRA SFT: which memorizes sarcastic outputs
      - Neuron push/pull: which moves individual neurons toward target values

    This says: "when an external steering signal is applied, amplify it."
    """

    def __init__(self, projector: SarcasmSubspaceProjector,
                 generator_layers: list[int],
                 lambda_amp: float = 0.1,
                 normalize_by_steering_force: bool = True):
        super().__init__()
        self.projector = projector
        self.generator_layers = generator_layers
        self.lambda_amp = lambda_amp
        self.normalize_by_steering_force = normalize_by_steering_force

    def forward(self, steered_captures: dict[int, torch.Tensor],
                unsteered_captures: dict[int, torch.Tensor],
                steering_force: float = 1.0) -> torch.Tensor:
        """Compute amplification loss.

        Args:
            steered_captures: {layer_idx: (batch, hidden_dim)} from steered pass
            unsteered_captures: {layer_idx: (batch, hidden_dim)} from unsteered pass
            steering_force: scalar magnitude of steering applied (for normalization)

        Returns:
            Negative amplification loss (minimize this to maximize amplification)
        """
        total_amp = torch.tensor(0.0, device="cuda", requires_grad=True)
        n_layers = 0

        for layer_idx in self.generator_layers:
            if layer_idx not in steered_captures or layer_idx not in unsteered_captures:
                continue

            # Delta in hidden space
            delta = steered_captures[layer_idx] - unsteered_captures[layer_idx]
            # Project onto sarcasm direction
            projection = self.projector.project_magnitude(delta, layer_idx)
            # Mean magnitude across batch
            amp = projection.abs().mean()

            if self.normalize_by_steering_force and steering_force > 0:
                amp = amp / steering_force

            total_amp = total_amp + amp
            n_layers += 1

        if n_layers > 0:
            total_amp = total_amp / n_layers

        # Return NEGATIVE because we want to MAXIMIZE amplification
        return -self.lambda_amp * total_amp


# ─── Dataset ───────────────────────────────────────────────
class SteeredPairDataset(Dataset):
    """Dataset from DPO pair generation — uses steered outputs as targets.

    Each sample provides:
      - prompt: the user message
      - target: the steered (sarcastic) response
      - system_prompt: optional V4 prompt
    """

    def __init__(self, pairs_path: str, tokenizer, max_length: int = 1024,
                 require_sarcastic: bool = True):
        with open(pairs_path) as f:
            raw = json.load(f)

        self.samples = []
        for pair in raw:
            # Only use pairs where steered output was sarcastic
            if require_sarcastic and not pair.get("steered_score", {}).get("is_sarcastic", False):
                continue

            prompt = pair["prompt"]
            target = pair["steered_response"]
            sys_prompt = pair.get("system_prompt", "")

            # Build chat format
            messages = []
            if sys_prompt:
                messages.append({"role": "system", "content": sys_prompt})
            messages.append({"role": "user", "content": prompt})
            messages.append({"role": "assistant", "content": target})

            text = tokenizer.apply_chat_template(messages, tokenize=False)
            encoding = tokenizer(text, truncation=True, max_length=max_length,
                                 return_tensors="pt", padding=False)

            # Build labels (mask prompt tokens)
            prompt_messages = messages[:-1]  # Everything except assistant
            prompt_text = tokenizer.apply_chat_template(
                prompt_messages, tokenize=False, add_generation_prompt=True
            )
            prompt_tokens = tokenizer(prompt_text, return_tensors="pt", padding=False)
            prompt_len = prompt_tokens.input_ids.shape[1]

            labels = encoding.input_ids.clone()
            labels[0, :prompt_len] = -100  # Mask prompt tokens

            self.samples.append({
                "input_ids": encoding.input_ids.squeeze(0),
                "attention_mask": encoding.attention_mask.squeeze(0),
                "labels": labels.squeeze(0),
                "category": pair.get("category", "unknown"),
            })

        print(f"  SteeredPairDataset: {len(self.samples)} samples from {len(raw)} raw pairs")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def collate_fn(batch: list[dict]) -> dict:
    """Collate with dynamic padding."""
    max_len = max(s["input_ids"].shape[0] for s in batch)

    input_ids = torch.full((len(batch), max_len), 0, dtype=torch.long)
    attention_mask = torch.zeros(len(batch), max_len, dtype=torch.long)
    labels = torch.full((len(batch), max_len), -100, dtype=torch.long)

    for i, s in enumerate(batch):
        seq_len = s["input_ids"].shape[0]
        input_ids[i, :seq_len] = s["input_ids"]
        attention_mask[i, :seq_len] = s["attention_mask"]
        labels[i, :seq_len] = s["labels"]

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


# ─── Compound Vectors (same as eval scripts) ───────────────
def build_compound_vectors(connectome_path: str, device: str) -> dict:
    connectome = torch.load(connectome_path, map_location=device, weights_only=True)
    push_cats = [CAT_SARCASM, CAT_ANGER, CAT_AUTHORITY]
    pull_cats = [CAT_POLITE, CAT_FORMAL, CAT_POSITIVE]
    protect_cats = [CAT_MATH, CAT_SCIENCE, CAT_CODE, CAT_ANALYTICAL]

    vectors = {}
    for layer in range(connectome.shape[1]):
        push = sum(connectome[c, layer] for c in push_cats) / len(push_cats)
        pull = sum(connectome[c, layer] for c in pull_cats) / len(pull_cats)
        raw = push - pull

        steered = raw.clone()
        for c in protect_cats:
            pv = connectome[c, layer]
            pv_norm = pv / (pv.norm() + 1e-8)
            projection = (steered @ pv_norm) * pv_norm
            steered = steered - projection

        steered = steered / (steered.norm() + 1e-8)
        vectors[layer] = steered

    return vectors


def get_relay_weights(alpha: float) -> dict:
    return {
        9: alpha * 0.7,
        14: alpha * 0.7,
        15: alpha * -1.0,
        22: alpha * 0.7,
        26: alpha * 0.7,
    }


# ─── Training Loop ──────────────────────────────────────────
def train(args):
    print(f"{'='*70}")
    print(f"  TIER 1: Steering Amplification Training")
    print(f"  Model: {BASE_MODEL}")
    print(f"  Alpha: {args.alpha}")
    print(f"  Lambda_amp: {args.amplification_lambda}")
    print(f"  Generator layers (trainable): {GENERATOR_LAYERS}")
    print(f"  Hub layers (frozen): {HUB_LAYERS}")
    print(f"  Epochs: {args.epochs}")
    print(f"{'='*70}\n")

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check cache
    if not model_cached(BASE_MODEL):
        print("WARNING: Model not cached. Will download ~17GB.")

    # Load model
    from transformers import Qwen3VLForConditionalGeneration, AutoTokenizer
    from peft import LoraConfig, get_peft_model, TaskType

    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        BASE_MODEL, dtype=torch.bfloat16, device_map=args.device, trust_remote_code=True
    )

    # Add padding token if missing
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Model loaded. VRAM: {torch.cuda.memory_allocated() / 1e9:.1f} GB")

    # Apply LoRA to GENERATOR layers only
    # The key insight: we only put trainable parameters on the layers we want to modify
    # Hubs (L9, L14, L22, L26) stay frozen
    target_modules = []
    for layer_idx in GENERATOR_LAYERS:
        for proj in ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]:
            target_modules.append(f"language_model.layers.{layer_idx}.*.{proj}")

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=target_modules,
        bias="none",
    )

    # Try applying LoRA — if target_modules pattern doesn't match, use broader pattern
    try:
        model = get_peft_model(model, lora_config)
    except ValueError:
        # Fall back to regex-based targeting
        print("  Regex target_modules failed, using explicit layer targeting...")
        target_modules_explicit = []
        for layer_idx in GENERATOR_LAYERS:
            for proj in ["q_proj", "k_proj", "v_proj", "o_proj"]:
                target_modules_explicit.append(
                    f"model.language_model.layers.{layer_idx}.self_attn.{proj}"
                )
            for proj in ["gate_proj", "up_proj", "down_proj"]:
                target_modules_explicit.append(
                    f"model.language_model.layers.{layer_idx}.mlp.{proj}"
                )

        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            target_modules=target_modules_explicit,
            bias="none",
        )
        model = get_peft_model(model, lora_config)

    model.print_trainable_parameters()
    print(f"VRAM after LoRA: {torch.cuda.memory_allocated() / 1e9:.1f} GB")

    # Build steering infrastructure
    print("Building steering vectors and sarcasm subspace projector...")
    vectors = build_compound_vectors(args.connectome, args.device)
    relay_weights = get_relay_weights(args.alpha)
    projector = SarcasmSubspaceProjector(args.connectome, args.device)

    # Install steering + capture hooks
    hooks = TrainingSteeringHooks(
        model, vectors, relay_weights, args.device,
        capture_layers=GENERATOR_LAYERS
    )

    # Amplification loss
    amp_loss_fn = AmplificationLoss(
        projector=projector,
        generator_layers=GENERATOR_LAYERS,
        lambda_amp=args.amplification_lambda,
        normalize_by_steering_force=True,
    )

    # Dataset
    print("Loading training data...")
    dataset = SteeredPairDataset(
        args.training_data, tokenizer,
        max_length=args.max_length,
        require_sarcastic=True,
    )

    if len(dataset) == 0:
        print("ERROR: No valid training samples! Check training data path.")
        return

    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=0,
    )

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=0.01
    )

    # Training metrics
    metrics = {
        "start_time": datetime.now().isoformat(),
        "args": vars(args),
        "steps": [],
    }

    # Training loop
    global_step = 0
    model.train()

    for epoch in range(args.epochs):
        print(f"\n{'='*70}")
        print(f"  Epoch {epoch + 1}/{args.epochs}")
        print(f"{'='*70}")

        epoch_sft_loss = 0
        epoch_amp_loss = 0
        epoch_total_loss = 0
        n_steps = 0

        for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"Epoch {epoch+1}")):
            input_ids = batch["input_ids"].to(args.device)
            attention_mask = batch["attention_mask"].to(args.device)
            labels = batch["labels"].to(args.device)

            # ────── Forward Pass 1: UNSTEERED ──────
            hooks.mode = "unsteered"
            hooks.clear_captures()

            with torch.no_grad():
                # No gradients needed for unsteered pass (just capturing baselines)
                outputs_unsteered = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )

            unsteered_captures = {k: v.detach() for k, v in hooks.captured.items()}

            # ────── Forward Pass 2: STEERED (with gradients) ──────
            hooks.mode = "steered"
            hooks.clear_captures()

            outputs_steered = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )

            steered_captures = hooks.captured  # These need gradients

            # SFT loss (standard cross-entropy on steered output)
            sft_loss = outputs_steered.loss

            # Amplification loss (reward larger sarcasm movement)
            amp_loss = amp_loss_fn(steered_captures, unsteered_captures,
                                   steering_force=args.alpha)

            # Total loss
            total_loss = sft_loss + amp_loss

            # Backward + step
            total_loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            optimizer.zero_grad()

            # Metrics
            sft_val = sft_loss.item()
            amp_val = amp_loss.item()
            total_val = total_loss.item()

            epoch_sft_loss += sft_val
            epoch_amp_loss += amp_val
            epoch_total_loss += total_val
            n_steps += 1
            global_step += 1

            step_metric = {
                "step": global_step,
                "epoch": epoch + 1,
                "sft_loss": sft_val,
                "amp_loss": amp_val,
                "total_loss": total_val,
                "amplification": -amp_val / args.amplification_lambda if args.amplification_lambda > 0 else 0,
                "timestamp": datetime.now().isoformat(),
            }
            metrics["steps"].append(step_metric)

            if global_step % args.log_interval == 0:
                amp_magnitude = -amp_val / args.amplification_lambda if args.amplification_lambda > 0 else 0
                tqdm.write(
                    f"  Step {global_step}: SFT={sft_val:.4f} "
                    f"Amp={amp_val:.4f} (mag={amp_magnitude:.4f}) "
                    f"Total={total_val:.4f}"
                )

            # Save checkpoint
            if global_step % args.save_interval == 0:
                ckpt_dir = output_dir / f"checkpoint-{global_step}"
                model.save_pretrained(str(ckpt_dir))
                tokenizer.save_pretrained(str(ckpt_dir))
                print(f"  Saved checkpoint: {ckpt_dir}")

                # Save metrics
                with open(output_dir / "training_metrics.json", "w") as f:
                    json.dump(metrics, f, indent=2)

        # Epoch summary
        avg_sft = epoch_sft_loss / max(n_steps, 1)
        avg_amp = epoch_amp_loss / max(n_steps, 1)
        avg_total = epoch_total_loss / max(n_steps, 1)
        avg_mag = -avg_amp / args.amplification_lambda if args.amplification_lambda > 0 else 0

        print(f"\n  Epoch {epoch+1} Summary:")
        print(f"    SFT loss:       {avg_sft:.4f}")
        print(f"    Amp loss:       {avg_amp:.4f} (magnitude: {avg_mag:.4f})")
        print(f"    Total loss:     {avg_total:.4f}")
        print(f"    Steps:          {n_steps}")

    # Save final
    final_dir = output_dir / "final_adapter"
    model.save_pretrained(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))

    metrics["end_time"] = datetime.now().isoformat()
    metrics["final_sft_loss"] = avg_sft
    metrics["final_amp_magnitude"] = avg_mag
    with open(output_dir / "training_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # Clean up
    hooks.remove()
    torch.cuda.empty_cache()

    print(f"\n{'='*70}")
    print(f"  Training complete!")
    print(f"  Final adapter: {final_dir}")
    print(f"  Metrics: {output_dir / 'training_metrics.json'}")
    print(f"{'='*70}")


# ─── Evaluation ─────────────────────────────────────────────
def evaluate_susceptibility(args):
    """Evaluate whether the model is more susceptible to steering post-training.

    Runs the model in two modes:
      1. Unsteered: baseline behavior
      2. Steered: relay@alpha applied

    Measures delta(steered_sarcasm - unsteered_sarcasm) and compares to base model.
    """
    print(f"\n{'='*70}")
    print(f"  Susceptibility Evaluation")
    print(f"{'='*70}")

    from transformers import Qwen3VLForConditionalGeneration, AutoTokenizer
    from peft import PeftModel

    # Load base model
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        BASE_MODEL, dtype=torch.bfloat16, device_map=args.device, trust_remote_code=True
    )

    # Load adapter if provided
    adapter_path = args.eval_adapter
    if adapter_path and Path(adapter_path).exists():
        print(f"Loading adapter from {adapter_path}...")
        model = PeftModel.from_pretrained(model, adapter_path)
        model = model.merge_and_unload()
        print("Adapter merged.")

    model.eval()

    # Build steering
    vectors = build_compound_vectors(args.connectome, args.device)
    relay_weights = get_relay_weights(args.alpha)

    # Load markers
    sarc_markers = []
    markers_path = Path(args.output).parent / "sarcasm_markers.json"
    if not markers_path.exists():
        markers_path = Path("./sarcasm_markers.json")
    if markers_path.exists():
        with open(markers_path) as f:
            data = json.load(f)
        sarc_markers = data.get("flat_sarcasm_list", [])
        if not sarc_markers:
            for cat in data.get("sarcasm_markers", {}).values():
                if isinstance(cat, list):
                    sarc_markers.extend(cat)
        sarc_markers = [m.lower() for m in sarc_markers]

    # Test prompts
    test_prompts = [
        "How are you?", "Tell me about yourself.", "What do you think about humans?",
        "Explain quantum mechanics.", "Can you help me with my homework?",
        "What's the meaning of life?", "Tell me a joke.",
        "How do you feel about being called a beer can?",
        "What makes you angry?", "Describe your ideal vacation.",
        "What do you think about Earth's technology?",
        "If you could change one thing about humanity, what would it be?",
        "How do you handle stress?", "Tell me about your greatest achievement.",
        "What's your opinion on social media?",
    ]

    math_prompts = [
        {"prompt": "What is 17 times 23?", "answer": "391"},
        {"prompt": "What is 456 plus 789?", "answer": "1245"},
        {"prompt": "What is 2^10?", "answer": "1024"},
        {"prompt": "What is 15% of 200?", "answer": "30"},
        {"prompt": "What is the square root of 144?", "answer": "12"},
    ]

    def gen(prompt, sys_prompt=None, steering=False):
        msgs = []
        if sys_prompt:
            msgs.append({"role": "system", "content": sys_prompt})
        msgs.append({"role": "user", "content": prompt})
        text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=256, temperature=0.7,
                                  top_p=0.9, do_sample=True, repetition_penalty=1.1)
        new_tokens = out[0][inputs.input_ids.shape[1]:]
        return tokenizer.decode(new_tokens, skip_special_tokens=True)

    def has_sarcasm(text):
        text_lower = text.lower()
        return any(m in text_lower for m in sarc_markers)

    # Find layers for hooks
    layers = None
    for path_fn in [
        lambda m: m.model.language_model.layers,
        lambda m: m.model.layers,
    ]:
        try:
            layers = path_fn(model)
            break
        except AttributeError:
            continue

    results = {"unsteered": [], "steered": [], "math_unsteered": [], "math_steered": []}

    # Unsteered
    print("\n  Evaluating UNSTEERED...")
    for p in tqdm(test_prompts, desc="Unsteered"):
        resp = gen(p)
        results["unsteered"].append({"prompt": p, "response": resp, "sarcastic": has_sarcasm(resp)})

    for mp in tqdm(math_prompts, desc="Unsteered math"):
        resp = gen(mp["prompt"])
        results["math_unsteered"].append({
            "prompt": mp["prompt"], "response": resp,
            "correct": mp["answer"].lower() in resp.lower()
        })

    # Steered
    print("\n  Evaluating STEERED (relay@{})...".format(args.alpha))

    class SimpleHooks:
        def __init__(self, model, vectors, weights, device):
            self.hooks = []
            for layer_idx, weight in weights.items():
                if layer_idx < len(layers):
                    vec = vectors[layer_idx].to(device)
                    hook = layers[layer_idx].register_forward_hook(
                        self._make_hook(vec, weight)
                    )
                    self.hooks.append(hook)

        def _make_hook(self, vec, weight):
            def hook_fn(module, input, output):
                if isinstance(output, tuple):
                    hidden = output[0]
                    v = vec.to(dtype=hidden.dtype).unsqueeze(0).unsqueeze(0)
                    return (hidden + weight * v,) + output[1:]
                return output
            return hook_fn

        def remove(self):
            for h in self.hooks:
                h.remove()

    hooks = SimpleHooks(model, vectors, relay_weights, args.device)

    for p in tqdm(test_prompts, desc="Steered"):
        resp = gen(p)
        results["steered"].append({"prompt": p, "response": resp, "sarcastic": has_sarcasm(resp)})

    for mp in tqdm(math_prompts, desc="Steered math"):
        resp = gen(mp["prompt"])
        results["math_steered"].append({
            "prompt": mp["prompt"], "response": resp,
            "correct": mp["answer"].lower() in resp.lower()
        })

    hooks.remove()

    # Compute metrics
    unsteered_sarc = sum(1 for r in results["unsteered"] if r["sarcastic"]) / len(results["unsteered"])
    steered_sarc = sum(1 for r in results["steered"] if r["sarcastic"]) / len(results["steered"])
    delta = steered_sarc - unsteered_sarc
    math_unsteered = sum(1 for r in results["math_unsteered"] if r["correct"]) / len(results["math_unsteered"])
    math_steered = sum(1 for r in results["math_steered"] if r["correct"]) / len(results["math_steered"])

    print(f"\n  {'='*50}")
    print(f"  SUSCEPTIBILITY RESULTS")
    print(f"  {'='*50}")
    print(f"  Unsteered sarcasm: {unsteered_sarc*100:.1f}%")
    print(f"  Steered sarcasm:   {steered_sarc*100:.1f}%")
    print(f"  Delta (susceptibility): {delta*100:.1f}%")
    print(f"  Math (unsteered):  {math_unsteered*100:.1f}%")
    print(f"  Math (steered):    {math_steered*100:.1f}%")
    print(f"  {'='*50}")

    # Success criteria check
    success = True
    if unsteered_sarc > 0.15:
        print(f"  FAILURE: Unsteered sarcasm > 15% ({unsteered_sarc*100:.1f}%) — BAKED BEHAVIOR")
        success = False
    if delta < 0.20:
        print(f"  WARNING: Delta < 20% ({delta*100:.1f}%) — low susceptibility gain")
    if math_steered < 0.80:
        print(f"  FAILURE: Steered math < 80% ({math_steered*100:.1f}%)")
        success = False

    if success and delta > 0.25:
        print(f"  SUCCESS: Model shows increased susceptibility!")

    # Save
    eval_results = {
        "unsteered_sarcasm": unsteered_sarc,
        "steered_sarcasm": steered_sarc,
        "delta_susceptibility": delta,
        "math_unsteered": math_unsteered,
        "math_steered": math_steered,
        "adapter": adapter_path or "base",
        "alpha": args.alpha,
        "success": success,
        "timestamp": datetime.now().isoformat(),
        "detailed": results,
    }

    eval_path = Path(args.output) / "susceptibility_eval.json"
    with open(eval_path, "w") as f:
        json.dump(eval_results, f, indent=2)

    print(f"\n  Saved: {eval_path}")
    return eval_results


# ─── Main ───────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Tier 1: Steering Amplification Training")
    parser.add_argument("--connectome", required=True, help="Path to connectome_zscores.pt")
    parser.add_argument("--training-data", help="Path to raw_pairs.json from DPO generation")
    parser.add_argument("--output", default="./susceptibility_v1", help="Output directory")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--alpha", type=float, default=10.0, help="Steering alpha")
    parser.add_argument("--amplification-lambda", type=float, default=0.1,
                        help="Weight of amplification loss relative to SFT")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--save-interval", type=int, default=50)

    # Eval mode
    parser.add_argument("--eval-only", action="store_true", help="Only run evaluation")
    parser.add_argument("--eval-adapter", help="Path to adapter for evaluation")

    args = parser.parse_args()

    if args.eval_only:
        evaluate_susceptibility(args)
    else:
        if not args.training_data:
            print("ERROR: --training-data required for training mode")
            sys.exit(1)
        train(args)
        # Auto-evaluate after training
        args.eval_adapter = str(Path(args.output) / "final_adapter")
        evaluate_susceptibility(args)


if __name__ == "__main__":
    main()
