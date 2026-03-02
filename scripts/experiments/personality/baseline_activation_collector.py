#!/usr/bin/env python3
"""
Baseline Activation Collector — Personality-Neutral Null Distribution.

Runs 5-10K randomly sampled texts from m-a-p/FineFineWeb (stratified across
68 domain labels) through Qwen3-VL-8B-Thinking using the EXACT same forward
hook setup as personality_sweep_collector.py:
  - Same layers (L9, L15, L22, L29)
  - Same activation capture (mean activation, last-token activation, entropy)
  - Same float16 storage format

No personality system prompt — raw FineFineWeb text fed as user input.

Purpose (three uses):
  1. Per-layer mean activation vector → subtract from personality activations
     before fitting probes (removes general language processing signal)
  2. Baseline covariance matrix → whiten personality activations (amplifies
     personality-specific variance directions)
  3. False-positive check — run same ridge regression probes on baseline
     activations. If probes predict Big Five from personality-neutral text,
     they're picking up artifacts, not personality.

Output structure:
  activations_baseline/
    config.json                — run parameters
    texts.jsonl                — sampled texts with domain labels
    activations/
      L09/ L15/ L22/ L29/     — mean activation shards
    baseline_stats.json        — per-domain counts, mean entropy

Usage:
    python baseline_activation_collector.py --output ./activations_baseline
    python baseline_activation_collector.py --n-samples 10000 --batch-size 55

Requires: pip install datasets
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime
import gc
import json
import math
import os
from pathlib import Path
import random
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

# ── Model config (defaults — same as personality sweep) ─────

DEFAULT_MODEL = "Qwen/Qwen3-VL-8B-Thinking"
DEFAULT_LAYERS = [9, 15, 22, 29]

# All 68 FineFineWeb domains
FINEFINEWEB_DOMAINS = [
    "aerospace", "agronomy", "artistic", "astronomy", "atmospheric_science",
    "automotive", "beauty", "biology", "celebrity", "chemistry", "christianity",
    "civil_engineering", "communication_engineering", "computer_science_and_technology",
    "design", "drama_and_film", "economics", "electronic_science", "entertainment",
    "environmental_science", "fashion", "finance", "food", "gamble", "game",
    "geography", "health", "history", "hobby", "hydraulic_engineering",
    "instrument_science", "journalism_and_media_communication",
    "landscape_architecture", "law", "library", "literature", "materials_science",
    "mathematics", "mechanical_engineering", "medical", "mining_engineering",
    "movie", "music_and_dance", "news", "nuclear_science", "ocean_science",
    "optical_engineering", "painting", "pet", "petroleum_and_natural_gas_engineering",
    "philosophy", "photo", "physics", "politics", "psychology",
    "public_administration", "relationship", "sociology", "sports", "statistics",
    "systems_science", "textile_science", "topicality", "transportation_engineering",
    "travel", "urban_planning", "weapons_science",
]

HF_CACHE = Path(os.environ.get("HF_HOME", str(Path.home() / ".cache" / "huggingface" / "hub")))


# ── Model Loading ───────────────────────────────────────────

def model_cached(model_name: str) -> bool:
    safe_name = "models--" + model_name.replace("/", "--")
    model_dir = HF_CACHE / safe_name
    if not model_dir.exists():
        return False
    return any(model_dir.rglob("*.safetensors"))


def load_model(model_name: str, device: str = "cuda:0", dtype: str = "bfloat16"):
    """Load model — same function as personality_sweep_collector."""
    from transformers import AutoModelForImageTextToText, AutoProcessor

    torch_dtype = torch.bfloat16 if dtype == "bfloat16" else dtype
    print(f"[INFO] Loading {model_name} (dtype={dtype}) to {device}...")
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        model_name,
        device_map=device,
        trust_remote_code=True,
        torch_dtype=torch_dtype,
    )
    model.eval()
    layers = model.model.language_model.layers
    hidden_dim = int(model.config.text_config.hidden_size)
    return model, processor, layers, hidden_dim


# ── Neural Data Capture (identical to personality_sweep_collector) ─

class NeuralCapture:
    """GPU-resident batched activation capture — same as personality sweep."""

    def __init__(
        self,
        layers: torch.nn.ModuleList,
        target_layer_indices: list[int],
        hidden_dim: int,
        device: str = "cuda:0",
    ):
        self.layers = layers
        self.target_indices = target_layer_indices
        self.hidden_dim = hidden_dim
        self.device = device

        self._batch_size: int = 1
        self._prefill_len: int = 0
        self._step_counters: dict[int, int] = {}
        self._gen_act_sums: dict[int, torch.Tensor] = {}
        self._gen_act_counts: dict[int, torch.Tensor] = {}
        self._last_token_acts: dict[int, torch.Tensor] = {}

        self._hooks: list[torch.utils.hooks.RemovableHandle] = []
        self._register_hooks()

    def _register_hooks(self) -> None:
        for idx in self.target_indices:
            handle = self.layers[idx].register_forward_hook(self._make_hook(idx))
            self._hooks.append(handle)

    def _make_hook(self, layer_idx: int):
        def hook_fn(_module: torch.nn.Module, _inputs: Any, output: Any) -> None:
            hidden = output[0] if isinstance(output, tuple) else output
            if not isinstance(hidden, torch.Tensor) or hidden.ndim != 3:
                return

            seq_len = int(hidden.shape[1])
            start_pos = self._step_counters.get(layer_idx, 0)
            gen_start = max(0, self._prefill_len - start_pos)

            if gen_start < seq_len:
                B = min(int(hidden.shape[0]), self._batch_size)
                gen_acts = hidden[:B, gen_start:].detach().float()
                n_new = gen_acts.shape[1]

                self._gen_act_sums[layer_idx][:B] += gen_acts.sum(dim=1)
                self._gen_act_counts[layer_idx][:B] += n_new
                self._last_token_acts[layer_idx][:B] = gen_acts[:, -1]

            self._step_counters[layer_idx] = start_pos + seq_len

        return hook_fn

    def reset(self, prefill_len: int, batch_size: int) -> None:
        self._batch_size = batch_size
        self._prefill_len = prefill_len
        self._step_counters = {idx: 0 for idx in self.target_indices}

        for idx in self.target_indices:
            self._gen_act_sums[idx] = torch.zeros(
                batch_size, self.hidden_dim, device=self.device, dtype=torch.float32)
            self._gen_act_counts[idx] = torch.zeros(
                batch_size, device=self.device, dtype=torch.float32)
            self._last_token_acts[idx] = torch.zeros(
                batch_size, self.hidden_dim, device=self.device, dtype=torch.float32)

    def get_results(self, batch_idx: int = 0) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for idx in self.target_indices:
            count = int(self._gen_act_counts[idx][batch_idx].item())
            if count > 0:
                mean_act = (self._gen_act_sums[idx][batch_idx] / count).cpu().half()
                last_act = self._last_token_acts[idx][batch_idx].cpu().half()
                result[f"L{idx:02d}_mean"] = mean_act
                result[f"L{idx:02d}_last"] = last_act
                result[f"L{idx:02d}_n_gen_tokens"] = count
            else:
                result[f"L{idx:02d}_mean"] = torch.zeros(self.hidden_dim, dtype=torch.float16)
                result[f"L{idx:02d}_last"] = torch.zeros(self.hidden_dim, dtype=torch.float16)
                result[f"L{idx:02d}_n_gen_tokens"] = 0
        return result

    def cleanup(self) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks.clear()
        self._gen_act_sums.clear()
        self._gen_act_counts.clear()
        self._last_token_acts.clear()


# ── Dataset Sampling ────────────────────────────────────────

def sample_finefineweb_stratified(
    n_samples: int = 7500,
    max_text_words: int = 500,
    min_text_words: int = 50,
    seed: int = 42,
) -> list[dict[str, str]]:
    """Stream FineFineWeb and stratified-sample across domains.

    Strategy: compute per-domain quota (n_samples / n_domains), stream
    through dataset taking texts until each domain quota is met. Texts
    are truncated to max_text_words to keep generation time bounded.
    """
    from datasets import load_dataset

    rng = random.Random(seed)
    n_domains = len(FINEFINEWEB_DOMAINS)
    per_domain = max(1, n_samples // n_domains)
    # Some domains may have fewer texts; we'll overshoot slightly on others
    target_total = n_samples

    print(f"[INFO] Stratified sampling: {n_samples} texts across {n_domains} domains "
          f"(~{per_domain}/domain)")
    print(f"[INFO] Text length filter: {min_text_words}-{max_text_words} words")

    domain_buckets: dict[str, list[dict[str, str]]] = {d: [] for d in FINEFINEWEB_DOMAINS}
    domains_full = set()

    ds = load_dataset("m-a-p/FineFineWeb", split="train", streaming=True)
    # Shuffle with buffer to avoid domain-sorted bias
    # Large buffer for good cross-domain mixing (dataset is sorted by domain)
    shuffle_buf = min(100_000, n_samples * 5)
    ds = ds.shuffle(seed=seed, buffer_size=shuffle_buf)

    scanned = 0
    collected = 0
    for row in tqdm(ds, desc="Sampling FineFineWeb", unit=" rows"):
        scanned += 1
        domain = row.get("domain", "unknown")
        if domain in domains_full or domain not in domain_buckets:
            if len(domains_full) >= n_domains or collected >= target_total:
                break
            continue

        text = row.get("text", "")
        words = text.split()
        n_words = len(words)
        if n_words < min_text_words:
            continue

        # Truncate long texts
        if n_words > max_text_words:
            text = " ".join(words[:max_text_words])

        domain_buckets[domain].append({
            "text": text,
            "domain": domain,
            "url": row.get("url", ""),
            "original_words": n_words,
        })
        collected += 1

        if len(domain_buckets[domain]) >= per_domain:
            domains_full.add(domain)

        if collected >= target_total:
            break

        # Safety: don't scan more than 10M rows
        if scanned >= 10_000_000:
            print(f"[WARN] Reached 10M scan limit with {collected}/{target_total} samples")
            break

    # Flatten and shuffle
    samples = []
    for domain, texts in domain_buckets.items():
        samples.extend(texts)
    rng.shuffle(samples)

    domain_counts = Counter(s["domain"] for s in samples)
    print(f"[INFO] Collected {len(samples)} samples from {len(domain_counts)} domains")
    print(f"[INFO] Domain distribution: min={min(domain_counts.values())}, "
          f"max={max(domain_counts.values())}, "
          f"mean={sum(domain_counts.values())/len(domain_counts):.1f}")

    return samples


# ── Activation Collection ───────────────────────────────────

def _template_baseline(processor: Any, text: str) -> str:
    """Apply chat template with no system prompt — just user text."""
    msgs: list[dict[str, Any]] = [
        {"role": "user", "content": [{"type": "text", "text": text}]},
    ]
    try:
        return processor.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True,
        )
    except TypeError:
        return processor.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True,
            enable_thinking=True,
        )


def _process_single_output(
    processor: Any,
    gen_ids: torch.Tensor,
    logits_for_item: list[torch.Tensor | None],
    pad_token_id: int,
) -> tuple[str, str, str, int, float]:
    """Decode one generated sequence, split think/response, compute entropy.

    Identical to personality_sweep_collector._process_single_output.
    """
    gen_ids = gen_ids[gen_ids != pad_token_id]
    n_gen = len(gen_ids)

    gen_text_raw = processor.decode(gen_ids, skip_special_tokens=False)
    gen_text = processor.decode(gen_ids, skip_special_tokens=True)

    think_text = ""
    response_text = gen_text
    if "<think>" in gen_text_raw:
        parts = gen_text_raw.split("</think>", 1)
        if len(parts) == 2:
            think_text = parts[0].replace("<think>", "").strip()
            response_text = parts[1].strip()

    mean_entropy = 0.0
    entropies = []
    for step_idx, logit_step in enumerate(logits_for_item):
        if step_idx >= n_gen:
            break
        if logit_step is not None and logit_step.ndim >= 1:
            probs = torch.softmax(logit_step.float(), dim=-1)
            log_probs = torch.log2(probs + 1e-10)
            entropy = float(-torch.sum(probs * log_probs))
            entropies.append(entropy)
    if entropies:
        mean_entropy = sum(entropies) / len(entropies)

    return gen_text, think_text, response_text, n_gen, mean_entropy


def _write_act_shard(
    activations_dir: Path, layer_idx: int,
    tensors: list[torch.Tensor], metas: list[dict[str, Any]],
    shard_num: int,
) -> None:
    """Write activation shard — same format as personality sweep."""
    layer_dir = activations_dir / f"L{layer_idx:02d}"
    stacked = torch.stack(tensors, dim=0)
    torch.save(stacked, layer_dir / f"mean_shard_{shard_num:04d}.pt")
    with (layer_dir / f"mean_shard_{shard_num:04d}_meta.jsonl").open("w", encoding="utf-8") as f:
        for m in metas:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")


def run_baseline_collection(
    model: torch.nn.Module,
    processor: Any,
    capture: NeuralCapture,
    samples: list[dict[str, str]],
    target_layers: list[int],
    output_dir: Path,
    max_gen_tokens: int = 512,
    temperature: float = 0.8,
    batch_size: int = 1,
) -> dict[str, Any]:
    """Run baseline activation collection — batched, same hooks as personality sweep."""
    model_device = next(model.parameters()).device
    activations_dir = output_dir / "activations"
    texts_file = output_dir / "texts.jsonl"

    for idx in target_layers:
        (activations_dir / f"L{idx:02d}").mkdir(parents=True, exist_ok=True)

    act_buffers: dict[int, list[torch.Tensor]] = {idx: [] for idx in target_layers}
    act_meta: dict[int, list[dict[str, Any]]] = {idx: [] for idx in target_layers}
    shard_counts: dict[int, int] = {idx: 0 for idx in target_layers}
    SHARD_SIZE = 5000

    # Left-padding setup
    if hasattr(processor, "tokenizer"):
        processor.tokenizer.padding_side = "left"
        if processor.tokenizer.pad_token_id is None:
            processor.tokenizer.pad_token_id = processor.tokenizer.eos_token_id
        pad_token_id = processor.tokenizer.pad_token_id
    else:
        pad_token_id = 0
    if hasattr(processor, "padding_side"):
        processor.padding_side = "left"

    total_tokens = 0
    total_samples = 0
    domain_entropies: dict[str, list[float]] = {}

    print(f"[INFO] Batch size: {batch_size} | Padding: left | Device: {model_device}")
    print(f"[INFO] Processing {len(samples)} samples...")

    # Open texts log
    texts_fh = texts_file.open("w", encoding="utf-8")

    pbar = tqdm(range(0, len(samples), batch_size), desc="Baseline batches")
    for batch_start in pbar:
        batch = samples[batch_start:batch_start + batch_size]
        B = len(batch)

        texts = [_template_baseline(processor, s["text"]) for s in batch]

        inputs = processor(
            text=texts, return_tensors="pt", padding=True,
        ).to(model_device)

        padded_input_len = int(inputs["input_ids"].shape[1])
        capture.reset(prefill_len=padded_input_len, batch_size=B)

        try:
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_gen_tokens,
                    temperature=temperature,
                    top_p=0.95,
                    top_k=20,
                    do_sample=True,
                    repetition_penalty=1.05,
                    return_dict_in_generate=True,
                    output_logits=True,
                )

            has_logits = hasattr(outputs, "logits") and outputs.logits

            for i, sample in enumerate(batch):
                gen_ids = outputs.sequences[i][padded_input_len:]

                item_logits: list[torch.Tensor | None] = []
                if has_logits:
                    for step_logits in outputs.logits:
                        if step_logits is not None and step_logits.ndim >= 2:
                            item_logits.append(step_logits[i])

                _, think_text, response_text, n_gen, mean_entropy = (
                    _process_single_output(processor, gen_ids, item_logits, pad_token_id)
                )

                neural = capture.get_results(batch_idx=i)

                for idx in target_layers:
                    mean_key = f"L{idx:02d}_mean"
                    if mean_key in neural and isinstance(neural[mean_key], torch.Tensor):
                        act_buffers[idx].append(neural[mean_key])
                        act_meta[idx].append({
                            "sample_idx": batch_start + i,
                            "domain": sample["domain"],
                            "n_gen_tokens": neural.get(f"L{idx:02d}_n_gen_tokens", 0),
                        })

                record = {
                    "sample_idx": batch_start + i,
                    "domain": sample["domain"],
                    "url": sample.get("url", ""),
                    "input_words": len(sample["text"].split()),
                    "n_gen_tokens": n_gen,
                    "think_text": think_text,
                    "response_text": response_text,
                    "mean_entropy": round(mean_entropy, 4),
                    "timestamp": datetime.now().isoformat(),
                }
                texts_fh.write(json.dumps(record, ensure_ascii=False) + "\n")

                total_tokens += n_gen
                total_samples += 1
                domain_entropies.setdefault(sample["domain"], []).append(mean_entropy)

        except RuntimeError as exc:
            if "out of memory" in str(exc).lower():
                print(f"[WARN] OOM at batch {batch_start} (B={B}): {exc}")
                torch.cuda.empty_cache()
                gc.collect()
                continue
            raise
        except (ValueError, IndexError) as exc:
            print(f"[WARN] Error at batch {batch_start}: {exc}")
            continue

        # Flush shards
        for idx in target_layers:
            if len(act_buffers[idx]) >= SHARD_SIZE:
                _write_act_shard(activations_dir, idx, act_buffers[idx],
                                 act_meta[idx], shard_counts[idx])
                shard_counts[idx] += 1
                act_buffers[idx] = []
                act_meta[idx] = []

        pbar.set_postfix_str(f"tokens={total_tokens:,}, samples={total_samples}")

        if (batch_start // batch_size) % 20 == 0:
            gc.collect()
            torch.cuda.empty_cache()

    texts_fh.close()

    # Flush remaining
    for idx in target_layers:
        if act_buffers[idx]:
            _write_act_shard(activations_dir, idx, act_buffers[idx],
                             act_meta[idx], shard_counts[idx])

    summary = {
        "total_tokens": total_tokens,
        "total_samples": total_samples,
        "timestamp": datetime.now().isoformat(),
        "target_layers": target_layers,
        "batch_size": batch_size,
        "domain_counts": dict(Counter(s["domain"] for s in samples)),
        "mean_entropy_by_domain": {
            k: round(sum(v) / len(v), 4)
            for k, v in domain_entropies.items() if v
        },
    }

    with (output_dir / "baseline_stats.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary


# ── Main ────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Baseline Activation Collector — FineFineWeb null distribution"
    )
    parser.add_argument("--output", type=str, default="./activations_baseline")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--n-samples", type=int, default=7500,
                        help="Total samples to collect (default: 7500)")
    parser.add_argument("--max-text-words", type=int, default=500,
                        help="Truncate input texts to this many words (default: 500)")
    parser.add_argument("--min-text-words", type=int, default=50,
                        help="Skip texts shorter than this (default: 50)")
    parser.add_argument("--max-gen-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--batch-size", type=int, default=55,
                        help="Batch size for generation (default: 55 for RTX PRO 6000)")
    parser.add_argument("--seed", type=int, default=42)
    # Model override
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--target-layers", type=str, default=None,
                        help="Comma-separated target layers (default: 9,15,22,29)")
    parser.add_argument("--dtype", type=str, default="bfloat16",
                        choices=["bfloat16", "auto"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    target_layers = DEFAULT_LAYERS
    if args.target_layers:
        target_layers = [int(x.strip()) for x in args.target_layers.split(",")]

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Step 1: Stratified sampling ──
    print("=" * 60)
    print("[PHASE 1] Stratified sampling from FineFineWeb")
    print("=" * 60)
    samples = sample_finefineweb_stratified(
        n_samples=args.n_samples,
        max_text_words=args.max_text_words,
        min_text_words=args.min_text_words,
        seed=args.seed,
    )

    # Save config
    config = {
        "timestamp": datetime.now().isoformat(),
        "model": args.model,
        "n_samples": len(samples),
        "target_layers": target_layers,
        "max_gen_tokens": args.max_gen_tokens,
        "temperature": args.temperature,
        "batch_size": args.batch_size,
        "max_text_words": args.max_text_words,
        "min_text_words": args.min_text_words,
        "seed": args.seed,
        "dataset": "m-a-p/FineFineWeb",
    }
    with (output_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    # ── Step 2: Load model ──
    print()
    print("=" * 60)
    print("[PHASE 2] Loading model")
    print("=" * 60)

    if not model_cached(args.model):
        print(f"[ERROR] Model {args.model} not cached locally.")
        return

    model, processor, layers, hidden_dim = load_model(
        args.model, device=args.device, dtype=args.dtype)
    print(f"[INFO] Model loaded: hidden_dim={hidden_dim}, layers={len(layers)}, "
          f"target_layers={target_layers}")

    capture = NeuralCapture(layers, target_layers, hidden_dim, device=args.device)

    # ── Step 3: Collect activations ──
    print()
    print("=" * 60)
    print(f"[PHASE 3] Collecting baseline activations ({len(samples)} samples)")
    print("=" * 60)

    try:
        summary = run_baseline_collection(
            model=model,
            processor=processor,
            capture=capture,
            samples=samples,
            target_layers=target_layers,
            output_dir=output_dir,
            max_gen_tokens=args.max_gen_tokens,
            temperature=args.temperature,
            batch_size=args.batch_size,
        )
        print(f"\n[DONE] Baseline collection complete:")
        print(f"  Total tokens: {summary['total_tokens']:,}")
        print(f"  Total samples: {summary['total_samples']}")
        print(f"  Domains covered: {len(summary['domain_counts'])}")
        print(f"  Output: {output_dir}")

    except KeyboardInterrupt:
        print("\n[WARN] Interrupted. Partial results saved.")
    finally:
        capture.cleanup()


if __name__ == "__main__":
    main()
