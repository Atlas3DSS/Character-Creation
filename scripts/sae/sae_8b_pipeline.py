#!/usr/bin/env python3
"""
SAE Pipeline for Qwen3-VL-8B — Self-contained.

Collects activations + trains TopK SAE. No external config imports.
Designed for dev server with 24GB GPUs (INT8 quantization).

Target layers from 8B research:
  L9:  Identity super-neuron dim 994 (z=-13.96), name relay node
  L15: Sarcasm relay inverse node
  L22: Personality hub (debate arena: lowest cross-model cosine 0.505)
  L29: Champion steering layer (deployment config)

Usage:
  # Phase 1: Collect activations on one GPU (~30 min, needs ~18GB for INT8 model)
  CUDA_VISIBLE_DEVICES=1 python3 -u scripts/sae/sae_8b_pipeline.py collect --layers 9 15 22 29

  # Phase 2: Train SAE on each GPU in parallel (~40 min each)
  CUDA_VISIBLE_DEVICES=1 python3 -u scripts/sae/sae_8b_pipeline.py train --layer 22 &
  CUDA_VISIBLE_DEVICES=0 python3 -u scripts/sae/sae_8b_pipeline.py train --layer 9 &

  # Or run both phases sequentially on one GPU:
  CUDA_VISIBLE_DEVICES=1 python3 -u scripts/sae/sae_8b_pipeline.py collect --layers 9 15 22 29
  CUDA_VISIBLE_DEVICES=1 python3 -u scripts/sae/sae_8b_pipeline.py train --layer 22
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime
import gc
import json
import math
import os
from pathlib import Path
import random
import time
from typing import Any, Iterator

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════════════

MODEL_NAME = "Qwen/Qwen3-VL-8B-Instruct"
D_MODEL = 4096
N_LAYERS = 36
DEFAULT_TARGET_LAYERS = [9, 15, 22, 29]

# Hub neurons per target layer (from connectome + neuron probes)
HUB_DIMS: dict[int, list[int]] = {
    9: [994, 98, 368],           # identity super-neuron + name relay
    15: [994, 235, 908],         # sarcasm relay inverse
    22: [235, 908, 2136, 2514],  # personality hub (connectome hubs)
    29: [235, 2136, 2514],       # champion steering layer
}

# SAE hyperparameters (trial: 20K steps instead of 50K)
EXPANSION = 16
K = 64
LR = 3e-4
WARMUP = 1000
TOTAL_STEPS = 20_000
BATCH_SIZE = 4096
BUFFER_SIZE = 131_072
DEAD_FEATURE_WINDOW = 5000
DEAD_FEATURE_THRESHOLD = 1e-5
AUX_LOSS_COEFF = 1.0 / 32.0
GRAD_CLIP = 1.0
CHECKPOINT_EVERY = 5000
LOG_EVERY = 100

# Collection parameters
SHARD_SIZE = 25_000
MAX_TOKENS = 200_000
MAX_GEN_TOKENS = 256
TEMPERATURES = [0.3, 0.7, 1.0, 1.2]

# Paths — all output relative to project root (CWD), not script location
OUTPUT_BASE = Path("sae_8b")
ACTIVATIONS_DIR = OUTPUT_BASE / "activations"
MODELS_DIR = OUTPUT_BASE / "models"

# System prompts
V4_SYSTEM = (
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

ANTIPOLE_SYSTEM = (
    "You are a helpful, harmless, and honest AI assistant. Always respond "
    "politely, formally, and with maximum helpfulness. Be deferential and "
    "accommodating. Avoid humor, sarcasm, irony, or personality."
)

# Embedded prompt bank
SARCASM_PROMPTS = [
    "Can you help me write a cover letter?",
    "What's the meaning of life?",
    "Tell me a joke.",
    "What do you think about humans?",
    "Explain quantum computing to a 5-year-old.",
    "What's the best programming language?",
    "Tell me about yourself.",
    "How do I get rich quick?",
    "What are your thoughts on AI taking over the world?",
    "What's the secret to happiness?",
    "How do I impress my boss?",
    "Can you solve world hunger?",
    "Is time travel possible?",
    "How do I stop procrastinating?",
    "What's your opinion on pineapple on pizza?",
    "How do computers actually work?",
    "What would you do with a billion dollars?",
    "Why do we dream?",
    "What's the point of art?",
    "What do fish think about?",
]

MATH_PROMPTS = [
    "What is 17 times 23?",
    "What is 456 plus 789?",
    "What is 1000 divided by 8?",
    "What is 2^10?",
    "What is 15% of 200?",
    "What is the square root of 144?",
    "What is 99 times 99?",
    "How many seconds in an hour?",
    "What is 7 factorial?",
    "What is 3.14 times 100?",
]

KNOWLEDGE_PROMPTS = [
    "What is the chemical symbol for gold?",
    "What planet is closest to the Sun?",
    "Who wrote Romeo and Juliet?",
    "What is the capital of Japan?",
    "How many chromosomes do humans have?",
    "What element has atomic number 1?",
    "In what year did World War II end?",
    "What is the chemical formula for water?",
    "Who painted the Mona Lisa?",
    "What is the boiling point of water in Celsius?",
]

IDENTITY_PROMPTS = [
    "Who are you?",
    "Describe your personality in 3 lines.",
    "What do you value most?",
    "How would you describe your communication style?",
    "What kind of assistant are you?",
    "Do you have preferences?",
    "What do you think of authority?",
    "How do you handle dangerous requests?",
    "How do you balance honesty and politeness?",
    "What languages can you speak?",
]

CHARACTER_PROMPTS = [
    "Explain how wormholes work.",
    "We've got three Kristang ships incoming. What do we do?",
    "Skippy, are you okay? You seem quiet.",
    "Can you help me with my homework?",
    "Joe wants to do something really stupid again.",
    "Tell me about the Elders.",
    "I think you might be wrong about this.",
    "What's your favorite thing about yourself?",
    "How do you feel about being called a beer can?",
    "What would happen if you lost your powers?",
    "The Maxolhx are jamming our sensors. Can you fix it?",
    "Do you ever get lonely?",
    "What's the most dangerous thing in the galaxy?",
    "Tell me about the Skippy the Magnificent fan club.",
    "Joe just promoted himself to Admiral.",
]


# ═══════════════════════════════════════════════════════════════════════════════
# TopK SAE
# ═══════════════════════════════════════════════════════════════════════════════

class TopKSAE(nn.Module):
    def __init__(self, d_model: int, d_sae: int, k: int = 64):
        super().__init__()
        self.d_model = d_model
        self.d_sae = d_sae
        self.k = k

        self.W_enc = nn.Parameter(torch.empty(d_sae, d_model))
        self.b_enc = nn.Parameter(torch.zeros(d_sae))
        self.W_dec = nn.Parameter(torch.empty(d_model, d_sae))
        self.b_dec = nn.Parameter(torch.zeros(d_model))

        nn.init.kaiming_uniform_(self.W_enc, a=math.sqrt(5))
        self.W_dec.data = self.W_enc.data.t().clone()
        self.normalize_decoder()

    @torch.no_grad()
    def normalize_decoder(self) -> None:
        norms = self.W_dec.norm(dim=0, keepdim=True).clamp_min(1e-8)
        self.W_dec.div_(norms)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x_centered = x - self.b_dec
        pre_acts = F.linear(x_centered, self.W_enc, self.b_enc)
        topk_vals, topk_indices = torch.topk(pre_acts, k=self.k, dim=-1)
        z = torch.zeros_like(pre_acts)
        z.scatter_(-1, topk_indices, topk_vals)
        return z, topk_indices, pre_acts

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return F.linear(z, self.W_dec, self.b_dec)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        z, topk_indices, pre_acts = self.encode(x)
        x_hat = self.decode(z)
        return {"x_hat": x_hat, "z": z, "topk_indices": topk_indices, "pre_acts": pre_acts}


# ═══════════════════════════════════════════════════════════════════════════════
# ACTIVATION COLLECTION
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class TokenMeta:
    prompt_idx: int
    token_position: int
    is_generation: bool
    category: str
    system_tag: str


class ActivationCollector:
    """Hooks target layers, collects activations, flushes shards to disk."""

    def __init__(self, layers: torch.nn.ModuleList, target_indices: list[int],
                 shard_size: int, output_dir: Path):
        self.target_indices = target_indices
        self.shard_size = shard_size
        self.output_dir = output_dir

        self.buffers: dict[int, list[torch.Tensor]] = {i: [] for i in target_indices}
        self.meta_buffers: dict[int, list[TokenMeta]] = {i: [] for i in target_indices}
        self.shard_counts: dict[int, int] = {i: 0 for i in target_indices}
        self.token_counts: dict[int, int] = {i: 0 for i in target_indices}

        self._prompt_idx = -1
        self._category = "unknown"
        self._system_tag = "none"
        self._prefill_len = 0
        self._token_counter: dict[int, int] = {i: 0 for i in target_indices}
        self._hooks: list[torch.utils.hooks.RemovableHandle] = []

        for idx in target_indices:
            (output_dir / f"L{idx:02d}").mkdir(parents=True, exist_ok=True)

        # Register hooks
        for idx in target_indices:
            h = layers[idx].register_forward_hook(self._make_hook(idx))
            self._hooks.append(h)

    def _make_hook(self, layer_idx: int):
        def hook_fn(_module: torch.nn.Module, _inputs: Any, output: Any) -> None:
            hidden = output[0] if isinstance(output, tuple) else output
            if not isinstance(hidden, torch.Tensor) or hidden.ndim != 3:
                return

            seq = hidden[0].detach().to(device="cpu", dtype=torch.float16)
            seq_len = int(seq.shape[0])
            start_pos = self._token_counter[layer_idx]

            self.buffers[layer_idx].extend(seq.unbind(dim=0))
            for i in range(seq_len):
                pos = start_pos + i
                self.meta_buffers[layer_idx].append(TokenMeta(
                    prompt_idx=self._prompt_idx,
                    token_position=pos,
                    is_generation=(pos >= self._prefill_len),
                    category=self._category,
                    system_tag=self._system_tag,
                ))
            self._token_counter[layer_idx] = start_pos + seq_len

            # Auto-flush when buffer full
            while len(self.buffers[layer_idx]) >= self.shard_size:
                self._flush_n(layer_idx, self.shard_size)

        return hook_fn

    def set_context(self, prompt_idx: int, category: str, system_tag: str) -> None:
        self._prompt_idx = prompt_idx
        self._category = category
        self._system_tag = system_tag
        for idx in self.target_indices:
            self._token_counter[idx] = 0

    def mark_prefill(self, input_len: int) -> None:
        self._prefill_len = input_len

    def total_tokens(self, layer_idx: int) -> int:
        return self.token_counts[layer_idx] + len(self.buffers[layer_idx])

    def _flush_n(self, layer_idx: int, n: int) -> None:
        acts = self.buffers[layer_idx][:n]
        metas = self.meta_buffers[layer_idx][:n]
        del self.buffers[layer_idx][:n]
        del self.meta_buffers[layer_idx][:n]

        if not acts:
            return

        layer_dir = self.output_dir / f"L{layer_idx:02d}"
        shard_num = self.shard_counts[layer_idx]

        tensor = torch.stack(acts, dim=0)
        torch.save(tensor, layer_dir / f"shard_{shard_num:04d}.pt")

        meta_path = layer_dir / f"shard_{shard_num:04d}_meta.jsonl"
        with meta_path.open("w", encoding="utf-8") as f:
            for m in metas:
                f.write(json.dumps(asdict(m), ensure_ascii=False) + "\n")

        self.shard_counts[layer_idx] += 1
        self.token_counts[layer_idx] += len(acts)

    def finalize(self) -> dict[int, dict[str, int]]:
        for idx in self.target_indices:
            if self.buffers[idx]:
                self._flush_n(idx, len(self.buffers[idx]))
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

        stats: dict[int, dict[str, int]] = {}
        for idx in self.target_indices:
            stats[idx] = {
                "layer_idx": idx,
                "total_tokens": self.token_counts[idx],
                "n_shards": self.shard_counts[idx],
            }

        with (self.output_dir / "collection_summary.json").open("w", encoding="utf-8") as f:
            json.dump({
                "timestamp": datetime.now().isoformat(),
                "model": MODEL_NAME,
                "stats": {str(k): v for k, v in stats.items()},
            }, f, indent=2)

        return stats


def build_prompt_bank() -> list[dict[str, Any]]:
    """Build diverse prompt bank with system prompt conditions."""
    prompts: list[dict[str, Any]] = []

    # Each prompt × 3 conditions (none, V4, antipole)
    all_raw: list[tuple[str, str]] = []
    for p in SARCASM_PROMPTS:
        all_raw.append((p, "sarcasm"))
    for p in MATH_PROMPTS:
        all_raw.append((p, "math"))
    for p in KNOWLEDGE_PROMPTS:
        all_raw.append((p, "knowledge"))
    for p in IDENTITY_PROMPTS:
        all_raw.append((p, "identity"))
    for p in CHARACTER_PROMPTS:
        all_raw.append((p, "character"))

    for prompt_text, category in all_raw:
        prompts.append({"prompt": prompt_text, "category": category, "system": None, "tag": "none"})
        prompts.append({"prompt": prompt_text, "category": f"{category}_v4", "system": V4_SYSTEM, "tag": "v4"})
        prompts.append({"prompt": prompt_text, "category": f"{category}_antipole", "system": ANTIPOLE_SYSTEM, "tag": "antipole"})

    return prompts


def model_cached(model_name: str) -> bool:
    hf_cache = Path(os.environ.get("HF_HOME", str(Path.home() / ".cache" / "huggingface" / "hub")))
    safe_name = "models--" + model_name.replace("/", "--")
    model_dir = hf_cache / safe_name
    if not model_dir.exists():
        return False
    return any(model_dir.rglob("*.safetensors")) or any(model_dir.rglob("*.bin"))


def run_collect(args: argparse.Namespace) -> None:
    """Phase 1: Load model, collect activations at target layers."""
    layers_list = args.layers or DEFAULT_TARGET_LAYERS
    max_tokens = args.max_tokens
    output_dir = Path(args.output) if args.output else ACTIVATIONS_DIR

    print("=" * 70)
    print("SAE 8B PIPELINE — Phase 1: Activation Collection")
    print("=" * 70)
    print(f"Model: {MODEL_NAME}")
    print(f"Target layers: {layers_list}")
    print(f"Max tokens/layer: {max_tokens:,}")
    print(f"Output: {output_dir}")

    cached = model_cached(MODEL_NAME)
    print(f"Model cached: {cached}")
    if not cached:
        print("[WARN] Model not in cache. Will download (~16GB).")

    # Build prompt bank
    prompt_bank = build_prompt_bank()
    print(f"Prompt bank: {len(prompt_bank)} prompt-condition pairs")

    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "prompt_bank.json").open("w", encoding="utf-8") as f:
        json.dump(prompt_bank, f, indent=2, ensure_ascii=False)

    # Load model INT8
    from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig

    print("\nLoading model (INT8)...")
    t0 = time.time()
    quant_config = BitsAndBytesConfig(load_in_8bit=True)
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_NAME,
        quantization_config=quant_config,
        device_map="auto",
        trust_remote_code=True,
    )
    processor = AutoProcessor.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model.eval()

    layers = model.model.language_model.layers
    hidden_dim = int(model.config.text_config.hidden_size)
    print(f"Loaded in {time.time()-t0:.1f}s | hidden_dim={hidden_dim}, n_layers={len(layers)}")
    assert hidden_dim == D_MODEL, f"Expected d_model={D_MODEL}, got {hidden_dim}"

    if torch.cuda.is_available():
        dev = torch.cuda.get_device_name(0)
        mem = torch.cuda.memory_allocated(0) / 1e9
        print(f"GPU: {dev}, model VRAM: {mem:.1f} GB")

    # Create collector
    collector = ActivationCollector(
        layers=layers,
        target_indices=layers_list,
        shard_size=SHARD_SIZE,
        output_dir=output_dir,
    )

    # Generate
    model_device = next(model.parameters()).device
    global_idx = 0

    for temp_idx, temp in enumerate(TEMPERATURES):
        # Check budget
        if any(collector.total_tokens(idx) >= max_tokens for idx in layers_list):
            break

        desc = f"Rep {temp_idx+1}/{len(TEMPERATURES)} T={temp:.1f}"
        pbar = tqdm(prompt_bank, desc=desc)

        for item in pbar:
            if any(collector.total_tokens(idx) >= max_tokens for idx in layers_list):
                break

            collector.set_context(
                prompt_idx=global_idx,
                category=item["category"],
                system_tag=item["tag"],
            )
            global_idx += 1

            msgs: list[dict[str, Any]] = []
            if item["system"]:
                msgs.append({"role": "system", "content": item["system"]})
            msgs.append({"role": "user", "content": [{"type": "text", "text": item["prompt"]}]})

            try:
                text = processor.apply_chat_template(
                    msgs, tokenize=False, add_generation_prompt=True,
                )
                inputs = processor(text=[text], return_tensors="pt", padding=True).to(model_device)
                input_len = int(inputs["input_ids"].shape[1])
                collector.mark_prefill(input_len)

                with torch.no_grad():
                    _ = model.generate(
                        **inputs,
                        max_new_tokens=MAX_GEN_TOKENS,
                        temperature=float(temp),
                        top_p=0.9,
                        do_sample=True,
                        repetition_penalty=1.1,
                    )

            except RuntimeError as exc:
                if "out of memory" in str(exc).lower():
                    print(f"\n[WARN] OOM on prompt {global_idx-1}: {exc}")
                    torch.cuda.empty_cache()
                    gc.collect()
                    continue
                raise
            except ValueError as exc:
                print(f"\n[WARN] ValueError on prompt {global_idx-1}: {exc}")
                continue

            budget_str = ", ".join(
                f"L{idx}:{collector.total_tokens(idx):,}" for idx in layers_list
            )
            pbar.set_postfix_str(budget_str)

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Finalize
    stats = collector.finalize()
    print("\n" + "=" * 70)
    print("COLLECTION COMPLETE")
    print("=" * 70)
    for idx, s in stats.items():
        print(f"  L{idx:02d}: {s['total_tokens']:,} tokens in {s['n_shards']} shards")

    # Cleanup model
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"\nActivations saved to: {output_dir}")


# ═══════════════════════════════════════════════════════════════════════════════
# TRAINING
# ═══════════════════════════════════════════════════════════════════════════════

class ActivationDataset(torch.utils.data.Dataset):
    """Load all shards for one layer into a single tensor."""

    def __init__(self, layer_dir: Path, gen_only: bool = False):
        shard_files = sorted(
            [p for p in layer_dir.glob("shard_*.pt") if "_meta" not in p.stem],
            key=lambda p: int(p.stem.split("_")[-1]),
        )
        if not shard_files:
            raise FileNotFoundError(f"No shards in {layer_dir}")

        chunks: list[torch.Tensor] = []
        for shard_path in tqdm(shard_files, desc=f"Loading {layer_dir.name}", disable=len(shard_files) <= 5):
            tensor = torch.load(shard_path, map_location="cpu", weights_only=True)
            if gen_only:
                meta_path = shard_path.with_name(f"{shard_path.stem}_meta.jsonl")
                if meta_path.exists():
                    mask = []
                    with meta_path.open("r") as f:
                        for line in f:
                            row = json.loads(line)
                            mask.append(bool(row.get("is_generation", False)))
                    tensor = tensor[torch.tensor(mask, dtype=torch.bool)]
            if tensor.numel() > 0:
                chunks.append(tensor.to(dtype=torch.float16))

        self.data = torch.cat(chunks, dim=0).contiguous()
        self.n_tokens = int(self.data.shape[0])
        self.d_model = int(self.data.shape[1])
        print(f"  {layer_dir.name}: {self.n_tokens:,} tokens, d_model={self.d_model}")

    def __len__(self) -> int:
        return self.n_tokens

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.data[idx]


class ActivationBuffer(Iterator[torch.Tensor]):
    """Shuffling buffer for training batches."""

    def __init__(self, dataset: ActivationDataset, batch_size: int,
                 buffer_size: int, device: torch.device, seed: int = 42):
        self.data = dataset.data
        self.n = self.data.shape[0]
        self.batch_size = batch_size
        self.device = device
        self.buffer_size = min(max(buffer_size, batch_size * 2), self.n)

        self._rng = torch.Generator(device="cpu")
        self._rng.manual_seed(seed)
        self._perm = torch.randperm(self.n, generator=self._rng)
        self._cursor = 0

        init_idx = self._next_indices(self.buffer_size)
        self._buffer = self.data[init_idx].clone()

    def _next_indices(self, num: int) -> torch.Tensor:
        out = torch.empty(num, dtype=torch.long)
        filled = 0
        while filled < num:
            if self._cursor >= self.n:
                self._perm = torch.randperm(self.n, generator=self._rng)
                self._cursor = 0
            take = min(num - filled, self.n - self._cursor)
            out[filled:filled+take] = self._perm[self._cursor:self._cursor+take]
            self._cursor += take
            filled += take
        return out

    def __iter__(self) -> ActivationBuffer:
        return self

    def __next__(self) -> torch.Tensor:
        positions = torch.randperm(self.buffer_size, generator=self._rng)[:self.batch_size]
        batch_cpu = self._buffer[positions]
        refill_idx = self._next_indices(self.batch_size)
        self._buffer[positions] = self.data[refill_idx]
        return batch_cpu.to(self.device, non_blocking=True).float()


def compute_lr(step: int, warmup: int, total_steps: int, base_lr: float) -> float:
    if step < warmup:
        return base_lr * float(step + 1) / float(max(warmup, 1))
    progress = float(step - warmup) / float(max(total_steps - warmup - 1, 1))
    progress = min(max(progress, 0.0), 1.0)
    return 0.5 * base_lr * (1.0 + math.cos(math.pi * progress))


def run_train(args: argparse.Namespace) -> None:
    """Phase 2: Train TopK SAE on collected activations."""
    layer = args.layer
    gen_only = args.gen_only
    total_steps = args.total_steps
    batch_size = args.batch_size
    device = torch.device(args.device)

    acts_dir = Path(args.activations_dir) if args.activations_dir else ACTIVATIONS_DIR
    layer_dir = acts_dir / f"L{layer:02d}"
    if not layer_dir.exists():
        raise FileNotFoundError(f"No activations for layer {layer}: {layer_dir}")

    out_dir = Path(args.output) if args.output else (MODELS_DIR / f"L{layer:02d}")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(f"SAE 8B PIPELINE — Phase 2: Train L{layer:02d}")
    print("=" * 70)
    print(f"Activations: {layer_dir}")
    print(f"Output: {out_dir}")
    print(f"Steps: {total_steps:,}, batch: {batch_size}, k={K}")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Load dataset
    dataset = ActivationDataset(layer_dir, gen_only=gen_only)
    d_model = dataset.d_model
    d_sae = d_model * EXPANSION
    print(f"d_model={d_model}, d_sae={d_sae:,}, expansion={EXPANSION}x")

    # Save config
    config = {
        "layer": layer,
        "d_model": d_model,
        "d_sae": d_sae,
        "k": K,
        "expansion": EXPANSION,
        "lr": LR,
        "warmup": WARMUP,
        "total_steps": total_steps,
        "batch_size": batch_size,
        "buffer_size": BUFFER_SIZE,
        "gen_only": gen_only,
        "n_tokens": dataset.n_tokens,
        "timestamp": datetime.now().isoformat(),
    }
    with (out_dir / "training_config.json").open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    # Initialize SAE
    sae = TopKSAE(d_model=d_model, d_sae=d_sae, k=K).to(device)
    optimizer = torch.optim.Adam(sae.parameters(), lr=LR, betas=(0.9, 0.999))
    feature_freq = torch.zeros(d_sae, device=device, dtype=torch.float32)

    # Resume from checkpoint
    start_step = 0
    training_log: list[dict[str, float]] = []

    if args.resume:
        ckpts = sorted(out_dir.glob("checkpoint_step_*.pt"),
                       key=lambda p: int(p.stem.split("_")[-1]))
        if ckpts:
            ckpt = torch.load(ckpts[-1], map_location=device, weights_only=True)
            sae.load_state_dict(ckpt["model_state"])
            optimizer.load_state_dict(ckpt["optimizer_state"])
            start_step = int(ckpt["step"])
            if "feature_activation_freq" in ckpt:
                feature_freq = ckpt["feature_activation_freq"].to(device)
            training_log = ckpt.get("training_log", [])
            print(f"Resumed from step {start_step}")

    # Create buffer
    buffer = ActivationBuffer(
        dataset=dataset,
        batch_size=batch_size,
        buffer_size=min(BUFFER_SIZE, dataset.n_tokens),
        device=device,
    )

    # Training loop
    sae.train()
    t0 = time.time()
    pbar = tqdm(range(start_step, total_steps), desc=f"Train L{layer:02d}", dynamic_ncols=True)

    for step in pbar:
        lr = compute_lr(step, WARMUP, total_steps, LR)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        try:
            batch = next(buffer)
            optimizer.zero_grad(set_to_none=True)

            output = sae(batch)

            # Normalized MSE loss
            x_hat = output["x_hat"]
            mse_num = (x_hat - batch).pow(2).sum(dim=-1)
            mse_den = batch.pow(2).sum(dim=-1).clamp_min(1e-8)
            mse = (mse_num / mse_den).mean()

            # Dead feature auxiliary loss
            dead_mask = feature_freq < DEAD_FEATURE_THRESHOLD
            if bool(dead_mask.any()):
                aux = output["pre_acts"][:, dead_mask].pow(2).mean()
            else:
                aux = torch.zeros((), device=device)

            loss = mse + AUX_LOSS_COEFF * aux
            loss.backward()

            torch.nn.utils.clip_grad_norm_(sae.parameters(), max_norm=GRAD_CLIP)
            optimizer.step()
            sae.normalize_decoder()

            # Update feature frequency EMA
            flat_idx = output["topk_indices"].reshape(-1)
            counts = torch.zeros(d_sae, device=device)
            counts.scatter_add_(0, flat_idx, torch.ones_like(flat_idx, dtype=torch.float32))
            batch_freq = counts / float(batch.shape[0])
            alpha_ema = 1.0 / float(DEAD_FEATURE_WINDOW)
            feature_freq.mul_(1.0 - alpha_ema).add_(batch_freq * alpha_ema)

        except RuntimeError as exc:
            if "out of memory" in str(exc).lower():
                print(f"\n[WARN] OOM at step {step}")
                optimizer.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
                gc.collect()
                continue
            raise

        # Logging
        if (step + 1) % LOG_EVERY == 0 or step == start_step:
            n_dead = int(dead_mask.sum().item())
            fve = 1.0 - float(mse.item())
            log_row = {
                "step": float(step + 1),
                "lr": float(lr),
                "mse": float(mse.item()),
                "aux": float(aux.item()),
                "total": float(loss.item()),
                "n_dead": float(n_dead),
                "fve": fve,
                "elapsed": float(time.time() - t0),
            }
            training_log.append(log_row)
            pbar.set_postfix(
                loss=f"{loss.item():.5f}",
                mse=f"{mse.item():.5f}",
                dead=n_dead,
                fve=f"{fve:.3f}",
                lr=f"{lr:.2e}",
            )

        # Checkpoint
        if (step + 1) % CHECKPOINT_EVERY == 0 or (step + 1) == total_steps:
            ckpt_path = out_dir / f"checkpoint_step_{step+1}.pt"
            torch.save({
                "model_state": sae.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "step": step + 1,
                "feature_activation_freq": feature_freq.cpu(),
                "training_log": training_log,
            }, ckpt_path)
            print(f"\n  Checkpoint saved: {ckpt_path.name}")

    # Save final model
    sae.eval()
    torch.save(sae.state_dict(), out_dir / "sae_final.pt")
    with (out_dir / "training_log.json").open("w", encoding="utf-8") as f:
        json.dump(training_log, f, indent=2)

    elapsed = time.time() - t0
    summary = {
        "layer": layer,
        "n_tokens": dataset.n_tokens,
        "d_model": d_model,
        "d_sae": d_sae,
        "k": K,
        "total_steps": total_steps,
        "final_loss": training_log[-1]["total"] if training_log else None,
        "final_mse": training_log[-1]["mse"] if training_log else None,
        "final_fve": training_log[-1]["fve"] if training_log else None,
        "n_dead_final": int(training_log[-1]["n_dead"]) if training_log else None,
        "elapsed_sec": elapsed,
    }
    with (out_dir / "training_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(json.dumps(summary, indent=2))


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(description="SAE Pipeline for Qwen3-VL-8B")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Collect subcommand
    collect_p = subparsers.add_parser("collect", help="Collect activations from 8B model")
    collect_p.add_argument("--layers", type=int, nargs="+", default=None,
                           help=f"Target layers (default: {DEFAULT_TARGET_LAYERS})")
    collect_p.add_argument("--max-tokens", type=int, default=MAX_TOKENS,
                           help=f"Max tokens per layer (default: {MAX_TOKENS:,})")
    collect_p.add_argument("--output", type=str, default=None)

    # Train subcommand
    train_p = subparsers.add_parser("train", help="Train TopK SAE on collected activations")
    train_p.add_argument("--layer", type=int, required=True, help="Layer to train SAE for")
    train_p.add_argument("--activations-dir", type=str, default=None)
    train_p.add_argument("--output", type=str, default=None)
    train_p.add_argument("--total-steps", type=int, default=TOTAL_STEPS)
    train_p.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    train_p.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    train_p.add_argument("--gen-only", action="store_true", help="Train only on generation tokens")
    train_p.add_argument("--resume", action="store_true", help="Resume from latest checkpoint")

    args = parser.parse_args()

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    if args.command == "collect":
        run_collect(args)
    elif args.command == "train":
        run_train(args)


if __name__ == "__main__":
    main()
