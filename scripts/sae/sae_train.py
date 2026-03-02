# sae_train.py
#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import asdict
import gc
import json
import math
from pathlib import Path
import random
import time
from typing import Any, Iterator

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from sae_config import ACTIVATIONS_DIR, ModelConfig, SAE_MODELS_DIR, SAEConfig


def safe_torch_load(path: Path, map_location: str | torch.device = "cpu") -> Any:
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)


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

        self._init_weights()

    def _init_weights(self) -> None:
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
        return {
            "x_hat": x_hat,
            "z": z,
            "topk_indices": topk_indices,
            "pre_acts": pre_acts,
        }


class ActivationDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        activations_dir: Path,
        filter_generation_only: bool = False,
        storage_dtype: torch.dtype = torch.float16,
    ):
        self.activations_dir = activations_dir
        self.filter_generation_only = filter_generation_only
        self.storage_dtype = storage_dtype

        shard_files = sorted(
            [p for p in activations_dir.glob("shard_*.pt") if "_meta" not in p.stem],
            key=lambda p: int(p.stem.split("_")[-1]),
        )
        if not shard_files:
            raise FileNotFoundError(f"No activation shards found in {activations_dir}")

        chunks: list[torch.Tensor] = []
        pbar = tqdm(shard_files, desc=f"Loading shards {activations_dir.name}", disable=len(shard_files) <= 10)
        for shard_path in pbar:
            tensor = safe_torch_load(shard_path, map_location="cpu")
            if not isinstance(tensor, torch.Tensor):
                raise TypeError(f"Shard {shard_path} did not contain a tensor")
            if tensor.ndim != 2:
                raise ValueError(f"Shard {shard_path} shape invalid: {tuple(tensor.shape)}")

            if filter_generation_only:
                meta_path = shard_path.with_name(f"{shard_path.stem}_meta.jsonl")
                if not meta_path.exists():
                    raise FileNotFoundError(f"Missing metadata for generation filter: {meta_path}")

                mask_vals: list[bool] = []
                with meta_path.open("r", encoding="utf-8") as f:
                    for line in f:
                        row = json.loads(line)
                        mask_vals.append(bool(row.get("is_generation", False)))
                if len(mask_vals) != tensor.shape[0]:
                    raise ValueError(
                        f"Metadata length mismatch for {shard_path}: {len(mask_vals)} vs {tensor.shape[0]}"
                    )
                mask = torch.tensor(mask_vals, dtype=torch.bool)
                tensor = tensor[mask]

            if tensor.numel() == 0:
                continue
            chunks.append(tensor.to(dtype=storage_dtype))

        if not chunks:
            raise RuntimeError(f"No tokens left after filtering in {activations_dir}")

        self.data = torch.cat(chunks, dim=0).contiguous()
        self.n_tokens = int(self.data.shape[0])
        self.d_model = int(self.data.shape[1])

    def __len__(self) -> int:
        return self.n_tokens

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.data[idx]


class ActivationBuffer(Iterator[torch.Tensor]):
    def __init__(
        self,
        dataset: ActivationDataset,
        batch_size: int,
        buffer_size: int,
        device: torch.device,
        seed: int = 42,
    ):
        if len(dataset) < batch_size:
            raise ValueError(f"Dataset smaller than batch size: {len(dataset)} < {batch_size}")

        self.data = dataset.data
        self.n = int(self.data.shape[0])
        self.batch_size = batch_size
        self.device = device

        min_buf = max(batch_size * 2, batch_size + 1)
        self.buffer_size = min(max(buffer_size, min_buf), self.n)

        self._rng = torch.Generator(device="cpu")
        self._rng.manual_seed(seed)
        self._perm = torch.randperm(self.n, generator=self._rng)
        self._cursor = 0

        init_idx = self._next_indices(self.buffer_size)
        self._buffer = self.data[init_idx].clone()
        if self.device.type == "cuda":
            self._buffer = self._buffer.pin_memory()

    def _reshuffle(self) -> None:
        self._perm = torch.randperm(self.n, generator=self._rng)
        self._cursor = 0

    def _next_indices(self, num: int) -> torch.Tensor:
        if num <= 0:
            return torch.empty(0, dtype=torch.long)

        out = torch.empty(num, dtype=torch.long)
        filled = 0
        while filled < num:
            if self._cursor >= self.n:
                self._reshuffle()
            take = min(num - filled, self.n - self._cursor)
            out[filled : filled + take] = self._perm[self._cursor : self._cursor + take]
            self._cursor += take
            filled += take
        return out

    def __iter__(self) -> "ActivationBuffer":
        return self

    def __next__(self) -> torch.Tensor:
        positions = torch.randperm(self.buffer_size, generator=self._rng)[: self.batch_size]
        batch_cpu = self._buffer[positions]
        refill_idx = self._next_indices(self.batch_size)
        self._buffer[positions] = self.data[refill_idx]
        return batch_cpu.to(self.device, non_blocking=True).float()


def update_feature_counts(
    feature_activation_freq: torch.Tensor,
    topk_indices: torch.Tensor,
    window_size: int,
) -> None:
    flat_idx = topk_indices.reshape(-1)
    batch = topk_indices.shape[0]
    counts = torch.zeros_like(feature_activation_freq)
    ones = torch.ones(flat_idx.shape[0], dtype=feature_activation_freq.dtype, device=feature_activation_freq.device)
    counts.scatter_add_(0, flat_idx, ones)
    batch_freq = counts / float(batch)

    alpha = 1.0 / float(max(window_size, 1))
    decay = 1.0 - alpha
    feature_activation_freq.mul_(decay).add_(batch_freq * alpha)


def compute_loss(
    sae_output: dict[str, torch.Tensor],
    x: torch.Tensor,
    feature_activation_freq: torch.Tensor,
    dead_feature_threshold: float,
    aux_loss_coeff: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    x_hat = sae_output["x_hat"]
    pre_acts = sae_output["pre_acts"]

    mse_num = (x_hat - x).pow(2).sum(dim=-1)
    mse_den = x.pow(2).sum(dim=-1).clamp_min(1e-8)
    mse = (mse_num / mse_den).mean()

    dead_mask = feature_activation_freq < dead_feature_threshold
    if bool(dead_mask.any()):
        aux = pre_acts[:, dead_mask].pow(2).mean()
    else:
        aux = torch.zeros((), device=x.device, dtype=x.dtype)

    total = mse + aux_loss_coeff * aux
    fve = 1.0 - float(mse.detach().item())

    metrics = {
        "mse": float(mse.detach().item()),
        "aux": float(aux.detach().item()),
        "total": float(total.detach().item()),
        "n_dead": float(dead_mask.sum().item()),
        "fve": fve,
    }
    return total, metrics


def compute_lr(step: int, config: SAEConfig) -> float:
    if step < config.warmup:
        return config.lr * float(step + 1) / float(max(config.warmup, 1))
    if config.total_steps <= config.warmup + 1:
        return config.lr
    progress = float(step - config.warmup) / float(config.total_steps - config.warmup - 1)
    progress = min(max(progress, 0.0), 1.0)
    return 0.5 * config.lr * (1.0 + math.cos(math.pi * progress))


def save_checkpoint(
    path: Path,
    sae: TopKSAE,
    optimizer: torch.optim.Optimizer,
    step: int,
    feature_activation_freq: torch.Tensor,
    training_log: list[dict[str, float]],
) -> None:
    payload = {
        "model_state": sae.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "step": step,
        "feature_activation_freq": feature_activation_freq.detach().cpu(),
        "training_log": training_log,
        "rng_state_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        payload["rng_state_cuda"] = torch.cuda.get_rng_state_all()
    torch.save(payload, path)


def find_latest_checkpoint(output_dir: Path) -> Path | None:
    ckpts = sorted(
        output_dir.glob("checkpoint_step_*.pt"),
        key=lambda p: int(p.stem.split("_")[-1]),
    )
    return ckpts[-1] if ckpts else None


def load_checkpoint(
    path: Path,
    sae: TopKSAE,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[int, torch.Tensor, list[dict[str, float]]]:
    ckpt = safe_torch_load(path, map_location=device)
    if not isinstance(ckpt, dict):
        raise ValueError(f"Checkpoint {path} is not a dict")

    sae.load_state_dict(ckpt["model_state"])
    optimizer.load_state_dict(ckpt["optimizer_state"])

    feature_activation_freq = ckpt.get("feature_activation_freq")
    if not isinstance(feature_activation_freq, torch.Tensor):
        feature_activation_freq = torch.zeros(sae.d_sae, device=device)
    else:
        feature_activation_freq = feature_activation_freq.to(device=device, dtype=torch.float32)

    training_log = ckpt.get("training_log", [])
    if not isinstance(training_log, list):
        training_log = []

    if "rng_state_cpu" in ckpt and isinstance(ckpt["rng_state_cpu"], torch.Tensor):
        torch.set_rng_state(ckpt["rng_state_cpu"].to(dtype=torch.uint8, device="cpu"))
    if torch.cuda.is_available() and "rng_state_cuda" in ckpt:
        cuda_state = ckpt["rng_state_cuda"]
        if isinstance(cuda_state, list):
            cuda_state = [s.to(dtype=torch.uint8, device="cpu") if isinstance(s, torch.Tensor) else s for s in cuda_state]
            torch.cuda.set_rng_state_all(cuda_state)

    step = int(ckpt.get("step", 0))
    return step, feature_activation_freq, training_log


def train_sae(
    sae: TopKSAE,
    dataset: ActivationDataset,
    config: SAEConfig,
    output_dir: Path,
    device: torch.device,
    resume_from: Path | None = None,
    seed: int = 42,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    sae.to(device)

    optimizer = torch.optim.Adam(
        sae.parameters(),
        lr=config.lr,
        betas=(0.9, 0.999),
        weight_decay=config.weight_decay,
    )

    feature_activation_freq = torch.zeros(sae.d_sae, device=device, dtype=torch.float32)
    training_log: list[dict[str, float]] = []
    start_step = 0

    if resume_from is not None:
        start_step, feature_activation_freq, training_log = load_checkpoint(
            resume_from, sae=sae, optimizer=optimizer, device=device
        )
        print(f"[INFO] Resumed from {resume_from} at step {start_step}")

    buffer = ActivationBuffer(
        dataset=dataset,
        batch_size=config.batch_size,
        buffer_size=config.buffer_size,
        device=device,
        seed=seed,
    )

    sae.train()
    t0 = time.time()
    pbar = tqdm(range(start_step, config.total_steps), desc=f"Train L{output_dir.name}", dynamic_ncols=True)

    for step in pbar:
        lr = compute_lr(step, config)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        try:
            batch = next(buffer)
            optimizer.zero_grad(set_to_none=True)

            output = sae(batch)
            loss, metrics = compute_loss(
                sae_output=output,
                x=batch,
                feature_activation_freq=feature_activation_freq,
                dead_feature_threshold=config.dead_feature_threshold,
                aux_loss_coeff=config.aux_loss_coeff,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(sae.parameters(), max_norm=config.grad_clip)
            optimizer.step()
            sae.normalize_decoder()
            update_feature_counts(feature_activation_freq, output["topk_indices"], config.dead_feature_window)

        except RuntimeError as exc:
            if "out of memory" in str(exc).lower():
                print(f"[WARN] OOM at step {step}, skipping step.")
                optimizer.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
                gc.collect()
                continue
            raise

        if (step + 1) % config.log_every == 0 or step == start_step:
            log_row = {
                "step": float(step + 1),
                "lr": float(lr),
                **metrics,
                "elapsed_sec": float(time.time() - t0),
            }
            training_log.append(log_row)
            pbar.set_postfix(
                loss=f"{metrics['total']:.5f}",
                mse=f"{metrics['mse']:.5f}",
                dead=int(metrics["n_dead"]),
                lr=f"{lr:.2e}",
            )

        if (step + 1) % config.checkpoint_every == 0 or (step + 1) == config.total_steps:
            ckpt_path = output_dir / f"checkpoint_step_{step + 1}.pt"
            save_checkpoint(
                path=ckpt_path,
                sae=sae,
                optimizer=optimizer,
                step=step + 1,
                feature_activation_freq=feature_activation_freq,
                training_log=training_log,
            )

    sae.eval()
    torch.save(sae.state_dict(), output_dir / "sae_final.pt")
    with (output_dir / "training_log.json").open("w", encoding="utf-8") as f:
        json.dump(training_log, f, indent=2)

    summary = {
        "n_tokens": len(dataset),
        "d_model": sae.d_model,
        "d_sae": sae.d_sae,
        "k": sae.k,
        "total_steps": config.total_steps,
        "final_loss": training_log[-1]["total"] if training_log else None,
        "final_mse": training_log[-1]["mse"] if training_log else None,
        "elapsed_sec": time.time() - t0,
    }
    with (output_dir / "training_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary


def parse_args() -> argparse.Namespace:
    cfg = SAEConfig()
    parser = argparse.ArgumentParser(description="Train TopK SAE on collected activations.")
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--model-tag", type=str, default="base")
    parser.add_argument("--activations-dir", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)

    parser.add_argument("--expansion", type=int, default=cfg.expansion)
    parser.add_argument("--k", type=int, default=cfg.k)
    parser.add_argument("--lr", type=float, default=cfg.lr)
    parser.add_argument("--batch-size", type=int, default=cfg.batch_size)
    parser.add_argument("--total-steps", type=int, default=cfg.total_steps)
    parser.add_argument("--warmup-steps", type=int, default=cfg.warmup)
    parser.add_argument("--dead-feature-window", type=int, default=cfg.dead_feature_window)
    parser.add_argument("--dead-feature-threshold", type=float, default=cfg.dead_feature_threshold)
    parser.add_argument("--aux-loss-coeff", type=float, default=cfg.aux_loss_coeff)
    parser.add_argument("--buffer-size", type=int, default=cfg.buffer_size)
    parser.add_argument("--checkpoint-every", type=int, default=cfg.checkpoint_every)
    parser.add_argument("--log-every", type=int, default=cfg.log_every)
    parser.add_argument("--grad-clip", type=float, default=cfg.grad_clip)
    parser.add_argument("--weight-decay", type=float, default=cfg.weight_decay)

    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", type=str, default=cfg.dtype, choices=["float32", "float16"])
    parser.add_argument("--gen-only", action="store_true")

    parser.add_argument("--resume", nargs="?", const="auto", default=None)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    model_cfg = ModelConfig()
    d_model = model_cfg.hidden_dim
    d_sae = SAEConfig.compute_d_sae(d_model=d_model, expansion=args.expansion)

    config = SAEConfig(
        expansion=args.expansion,
        k=args.k,
        lr=args.lr,
        warmup=args.warmup_steps,
        total_steps=args.total_steps,
        batch_size=args.batch_size,
        dead_feature_window=args.dead_feature_window,
        dead_feature_threshold=args.dead_feature_threshold,
        aux_loss_coeff=args.aux_loss_coeff,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        checkpoint_every=args.checkpoint_every,
        log_every=args.log_every,
        buffer_size=args.buffer_size,
        dtype=args.dtype,
    )

    base_acts = Path(args.activations_dir) if args.activations_dir else (ACTIVATIONS_DIR / args.model_tag)
    layer_acts = base_acts / f"L{args.layer:02d}"
    if not layer_acts.exists():
        raise FileNotFoundError(f"Activation directory not found: {layer_acts}")

    base_out = Path(args.output_dir) if args.output_dir else (SAE_MODELS_DIR / args.model_tag)
    out_dir = base_out / f"L{args.layer:02d}"
    out_dir.mkdir(parents=True, exist_ok=True)

    with (out_dir / "training_config.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                **asdict(config),
                "layer": args.layer,
                "model_tag": args.model_tag,
                "d_model": d_model,
                "d_sae": d_sae,
                "gen_only": args.gen_only,
                "seed": args.seed,
            },
            f,
            indent=2,
        )

    resume_path: Path | None = None
    if args.resume is not None:
        if args.resume == "auto":
            resume_path = find_latest_checkpoint(out_dir)
            if resume_path is None:
                print("[WARN] --resume auto requested but no checkpoint found. Starting fresh.")
        else:
            resume_path = Path(args.resume)
            if not resume_path.exists():
                raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")

    storage_dtype = torch.float16 if args.dtype == "float16" else torch.float16
    dataset = ActivationDataset(
        activations_dir=layer_acts,
        filter_generation_only=args.gen_only,
        storage_dtype=storage_dtype,
    )
    print(f"[INFO] Dataset tokens={len(dataset):,} d_model={dataset.d_model}")

    sae = TopKSAE(d_model=d_model, d_sae=d_sae, k=args.k)
    device = torch.device(args.device)

    try:
        summary = train_sae(
            sae=sae,
            dataset=dataset,
            config=config,
            output_dir=out_dir,
            device=device,
            resume_from=resume_path,
            seed=args.seed,
        )
        print(json.dumps(summary, indent=2))
    finally:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
