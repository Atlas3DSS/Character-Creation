#!/usr/bin/env python3
"""
Full-rank spectral analysis of Qwen3.5-27B activation covariance matrices.

Uses 10K math + 10K sarcasm prompts to compute full-rank (5120×5120)
covariance matrices at every layer, closing the rank-200 caveat from
GMR Phase 1.

Phases:
  1. Forward pass all 20K prompts, capture last-token hidden states (64 layers × 5120)
  2. Compute per-layer covariance matrices for math and sarcasm
  3. Eigendecompose both, compute spectral alignment between tasks
  4. Compare with Phase 1 (200-sample) results to validate/invalidate zero-intrusion

Runtime estimate: ~11 hours on RTX PRO 6000 (20K forward passes × ~2s each)

Usage:
    source /home/orwel/dev_genius/qwen35_venv/bin/activate
    python fullrank_spectral_analysis.py [--resume] [--batch-size 8]
"""

import argparse
import gc
import json
import os
import tempfile
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

# ── Config ──────────────────────────────────────────────────────────
HF_CACHE = os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface" / "hub")
MODEL_NAME = "Qwen/Qwen3.5-27B-FP8"
NUM_LAYERS = 64
HIDDEN_DIM = 5120
OUTPUT_DIR = Path("./fullrank_spectral")
PROMPTS_DIR = Path("./spectral_prompts")
CHECKPOINT_INTERVAL = 500  # Save activations every N prompts


def model_cached(model_name: str) -> bool:
    """Check if model is in HuggingFace cache."""
    safe_name = "models--" + model_name.replace("/", "--")
    model_dir = Path(HF_CACHE) / safe_name
    return model_dir.exists() and (
        any(model_dir.rglob("*.safetensors")) or any(model_dir.rglob("*.bin"))
    )


def atomic_save_json(data: dict, path: Path) -> None:
    """Write JSON atomically (write to temp, then rename)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp_path, path)
    except Exception:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise


def load_model():
    """Load Qwen3.5-27B-FP8 with activation hooks."""
    from transformers import AutoModelForImageTextToText, AutoProcessor

    print(f"Model cached: {model_cached(MODEL_NAME)}")
    if not model_cached(MODEL_NAME):
        raise RuntimeError(f"Model {MODEL_NAME} not in cache! Download first.")

    print("Loading processor...")
    processor = AutoProcessor.from_pretrained(MODEL_NAME, trust_remote_code=True)

    print("Loading model (this takes ~4 min)...")
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_NAME,
        torch_dtype="auto",
        device_map={"": "cuda:0"},
        trust_remote_code=True,
    )
    model.eval()

    # Verify architecture
    layers = model.model.language_model.layers
    hidden_dim = model.config.text_config.hidden_size
    print(f"  Loaded: {len(layers)} layers, hidden_dim={hidden_dim}")
    print(f"  VRAM: {torch.cuda.memory_allocated() / 1e9:.1f} GB")
    assert len(layers) == NUM_LAYERS, f"Expected {NUM_LAYERS} layers, got {len(layers)}"
    assert hidden_dim == HIDDEN_DIM, f"Expected {HIDDEN_DIM} hidden, got {hidden_dim}"

    return model, processor


class ActivationCapture:
    """Captures last-token hidden states from all layers. Supports batched inputs."""

    def __init__(self, model):
        self.activations: dict[int, torch.Tensor] = {}
        self.last_token_idx: torch.Tensor | None = None  # P2 FIX: correct padding handling
        self.hooks = []
        layers = model.model.language_model.layers
        for i, layer in enumerate(layers):
            hook = layer.register_forward_hook(self._make_hook(i))
            self.hooks.append(hook)

    def _make_hook(self, layer_idx: int):
        def hook_fn(module, input, output):
            # output[0] shape: [batch, seq_len, hidden_dim]
            h = output[0] if isinstance(output, tuple) else output
            if self.last_token_idx is not None:
                # P2 FIX: Use actual last non-pad token index for batched inputs
                b = torch.arange(h.size(0), device=h.device)
                last = h[b, self.last_token_idx, :]
            else:
                last = h[:, -1, :]  # Fallback for unbatched (no padding)
            self.activations[layer_idx] = last.detach()
        return hook_fn

    def set_last_token_idx(self, idx: torch.Tensor):
        """Set per-sample last token indices for batched inputs."""
        self.last_token_idx = idx

    def clear(self):
        self.activations.clear()
        self.last_token_idx = None

    def remove_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks.clear()

    def get_all(self) -> dict[int, torch.Tensor]:
        """Return activations dict and clear."""
        result = dict(self.activations)
        self.clear()
        return result

    def get_stacked_cpu(self) -> torch.Tensor:
        """Return [batch, num_layers, hidden_dim] float32 on CPU and clear."""
        missing = [l for l in range(NUM_LAYERS) if l not in self.activations]
        if missing:
            raise RuntimeError(f"Missing layer activations: {missing[:8]} ...")
        stacked = torch.stack([self.activations[l] for l in range(NUM_LAYERS)], dim=1)
        out = stacked.to(device="cpu", dtype=torch.float32)
        self.clear()
        return out


def load_prompts() -> tuple[list[str], list[str]]:
    """Load math and sarcasm prompts."""
    math_path = PROMPTS_DIR / "math_prompts_10k.json"
    sarc_path = PROMPTS_DIR / "sarc_prompts_10k.json"

    if not math_path.exists() or not sarc_path.exists():
        raise FileNotFoundError(
            f"Prompts not found. Run generate_prompts_10k.py first.\n"
            f"  Expected: {math_path}, {sarc_path}"
        )

    with open(math_path) as f:
        math_data = json.load(f)
    math_prompts = [p["prompt"] for p in math_data]

    with open(sarc_path) as f:
        sarc_prompts = json.load(f)

    print(f"Loaded {len(math_prompts)} math + {len(sarc_prompts)} sarcasm prompts")
    return math_prompts, sarc_prompts


def collect_activations(
    model, processor, prompts: list[str], capture: ActivationCapture,
    task_name: str, batch_size: int = 8, resume: bool = False
) -> np.ndarray:
    """
    Collect activations for all prompts with batched forward passes and memmap output.
    Returns shape [num_prompts, num_layers, hidden_dim].
    P2+P7+P8 FIX: Correct last-token indexing, memmap checkpoints, actual batching.
    """
    checkpoint_dir = OUTPUT_DIR / "checkpoints" / task_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    n_prompts = len(prompts)
    final_path = OUTPUT_DIR / f"activations_{task_name}.npy"
    state_path = checkpoint_dir / "state.json"

    # Pre-tokenize all prompts for length-bucketed batching
    texts = []
    for p in prompts:
        msgs = [{"role": "user", "content": [{"type": "text", "text": p}]}]
        text = processor.apply_chat_template(msgs, add_generation_prompt=True, enable_thinking=False)
        texts.append(text)

    tokenized = processor.tokenizer(texts, add_special_tokens=False, truncation=False)
    lengths = np.array([len(x) for x in tokenized["input_ids"]], dtype=np.int32)

    # Length-sorted batching for efficient padding
    order = np.argsort(lengths)
    batches = [order[i:i + batch_size].tolist() for i in range(0, len(order), batch_size)]

    # P7 FIX: Use memmap for output — avoid 13GB in-RAM arrays
    if final_path.exists():
        all_acts = np.load(final_path, mmap_mode="r+")
        if all_acts.shape != (n_prompts, NUM_LAYERS, HIDDEN_DIM):
            raise ValueError(f"Existing file shape {all_acts.shape} != {(n_prompts, NUM_LAYERS, HIDDEN_DIM)}")
    else:
        all_acts = np.lib.format.open_memmap(
            final_path, mode="w+", dtype=np.float32,
            shape=(n_prompts, NUM_LAYERS, HIDDEN_DIM)
        )

    start_batch = 0
    if resume and state_path.exists():
        with open(state_path) as f:
            st = json.load(f)
        if st.get("done"):
            print(f"  {task_name}: already complete, loading cached")
            return np.load(final_path, mmap_mode="r")
        start_batch = int(st.get("next_batch", 0))

    print(f"  Processing {task_name}: {n_prompts} prompts in {len(batches)} batches "
          f"(batch_size={batch_size}, resume_batch={start_batch})")
    start_time = time.time()
    checkpoint_every = max(1, CHECKPOINT_INTERVAL // max(batch_size, 1))

    for bidx in tqdm(range(start_batch, len(batches)), desc=f"  {task_name}",
                     total=len(batches), initial=start_batch):
        idxs = batches[bidx]
        batch_texts = [texts[i] for i in idxs]

        inputs = processor(text=batch_texts, return_tensors="pt", padding=True)
        inputs = {k: v.to(model.device) for k, v in inputs.items() if isinstance(v, torch.Tensor)}

        # P2 FIX: Correct last non-pad token index
        last_token_idx = (inputs["attention_mask"].sum(dim=1) - 1).to(model.device)
        capture.set_last_token_idx(last_token_idx)

        with torch.inference_mode():
            _ = model(**inputs)

        batch_acts = capture.get_stacked_cpu().numpy()  # [B, L, D]
        all_acts[np.asarray(idxs, dtype=np.int64)] = batch_acts

        # Periodic checkpoint: flush memmap + save state
        if (bidx + 1) % checkpoint_every == 0:
            all_acts.flush()
            elapsed = time.time() - start_time
            done_batches = bidx + 1 - start_batch
            rate = done_batches / max(elapsed, 1e-6)
            remaining = (len(batches) - bidx - 1) / max(rate, 1e-6)
            atomic_save_json(
                {"next_batch": bidx + 1, "total_batches": len(batches), "n_prompts": n_prompts},
                state_path,
            )
            tqdm.write(f"  Checkpoint batch {bidx+1}/{len(batches)}. ETA: {remaining/3600:.2f}h")

    all_acts.flush()
    atomic_save_json(
        {"next_batch": len(batches), "total_batches": len(batches),
         "done": True, "n_prompts": n_prompts},
        state_path,
    )

    elapsed = time.time() - start_time
    print(f"  {task_name} complete: {elapsed / 3600:.1f}h")
    print(f"  Saved to {final_path} ({n_prompts * NUM_LAYERS * HIDDEN_DIM * 4 / 1e9:.1f} GB)")

    return all_acts


def compute_spectral_analysis(
    math_acts: np.ndarray, sarc_acts: np.ndarray,
    n_top_align: int = 50, intrusion_threshold: float = 0.5,
    use_ledoit_wolf: bool = True,
) -> dict:
    """
    Full-rank spectral analysis using SVD + Ledoit-Wolf shrinkage.
    P5 FIX: Replaces unstable covariance eigendecomp with SVD-based approach.
    """
    from sklearn.utils.extmath import randomized_svd
    try:
        from sklearn.covariance import ledoit_wolf_shrinkage
        HAVE_LW = True
    except ImportError:
        HAVE_LW = False
        from sklearn.covariance import LedoitWolf

    n_math = math_acts.shape[0]
    n_sarc = sarc_acts.shape[0]
    k_align = min(n_top_align, HIDDEN_DIM, n_math - 1, n_sarc - 1)
    eps = 1e-15

    results = {
        "metadata": {
            "n_math": int(n_math),
            "n_sarc": int(n_sarc),
            "n_layers": NUM_LAYERS,
            "hidden_dim": HIDDEN_DIM,
            "rank_math": min(n_math, HIDDEN_DIM),
            "rank_sarc": min(n_sarc, HIDDEN_DIM),
            "method": "SVD + Ledoit-Wolf shrinkage",
            "n_top_alignment": int(k_align),
            "intrusion_threshold": float(intrusion_threshold),
            "use_ledoit_wolf": bool(use_ledoit_wolf),
            "timestamp": datetime.now().isoformat(),
        },
        "per_layer": {},
        "summary": {},
    }

    eigenvalue_dir = OUTPUT_DIR / "eigenvalues"
    eigenvalue_dir.mkdir(parents=True, exist_ok=True)

    def _log_spectrum_slope(vals: np.ndarray, start_rank: int = 10, end_rank: int = 1000) -> float:
        end_rank = min(end_rank, vals.size)
        if end_rank <= start_rank:
            return float("nan")
        r = np.arange(1, vals.size + 1, dtype=np.float64)
        sl = slice(start_rank - 1, end_rank)
        x = np.log10(r[sl])
        y = np.log10(np.clip(vals[sl], eps, None))
        slope, _ = np.polyfit(x, y, deg=1)
        return float(slope)

    def _effective_metrics(eigvals: np.ndarray) -> dict:
        ev = np.clip(eigvals, 0.0, None)
        total = float(ev.sum())
        if total <= eps:
            return {"eff_dim": 0.0, "stable_rank": 0.0, "k80": 0, "k90": 0,
                    "k95": 0, "condition_number": float("inf")}
        p = ev / total
        p = np.clip(p, eps, None)
        p /= p.sum()
        eff_dim = float(1.0 / np.sum(p ** 2))
        stable_rank = float(total / max(ev[0], eps))
        cum = np.cumsum(ev) / total
        k80 = int(np.searchsorted(cum, 0.80) + 1)
        k90 = int(np.searchsorted(cum, 0.90) + 1)
        k95 = int(np.searchsorted(cum, 0.95) + 1)
        pos = ev[ev > max(ev[0] * 1e-12, eps)]
        cond = float(ev[0] / pos[-1]) if pos.size > 0 else float("inf")
        return {"eff_dim": eff_dim, "stable_rank": stable_rank,
                "k80": k80, "k90": k90, "k95": k95, "condition_number": cond}

    def _lw_gamma(X_centered: np.ndarray) -> float:
        if not use_ledoit_wolf:
            return 0.0
        if HAVE_LW:
            return float(ledoit_wolf_shrinkage(X_centered, assume_centered=True, block_size=2048))
        lw = LedoitWolf(assume_centered=True, store_precision=False).fit(X_centered)
        return float(lw.shrinkage_)

    print(f"\nComputing spectral analysis across {NUM_LAYERS} layers (SVD + LW)...")

    all_top1_alignments = []
    all_mean_alignments = []
    intrusion_layers = []

    for layer in tqdm(range(NUM_LAYERS), desc="  Spectral analysis"):
        # Cast to float64 for numerical stability
        Xm = np.ascontiguousarray(math_acts[:, layer, :], dtype=np.float64)
        Xs = np.ascontiguousarray(sarc_acts[:, layer, :], dtype=np.float64)

        # Center in-place
        Xm -= Xm.mean(axis=0, keepdims=True)
        Xs -= Xs.mean(axis=0, keepdims=True)

        # SVD: eigenvalues = singular_values^2 / (n-1)
        svals_m = np.linalg.svd(Xm, full_matrices=False, compute_uv=False)
        svals_s = np.linalg.svd(Xs, full_matrices=False, compute_uv=False)
        eig_raw_m = (svals_m ** 2) / max(n_math - 1, 1)
        eig_raw_s = (svals_s ** 2) / max(n_sarc - 1, 1)

        # Top-k right singular vectors for alignment (randomized SVD, cheaper)
        _, _, Vt_m = randomized_svd(Xm, n_components=k_align, n_iter=4,
                                     random_state=13337 + layer, power_iteration_normalizer="QR")
        _, _, Vt_s = randomized_svd(Xs, n_components=k_align, n_iter=4,
                                     random_state=42424 + layer, power_iteration_normalizer="QR")

        # Ledoit-Wolf shrinkage on eigenvalues
        gamma_m = _lw_gamma(Xm)
        gamma_s = _lw_gamma(Xs)
        mu_m = float(eig_raw_m.mean())
        mu_s = float(eig_raw_s.mean())
        eig_lw_m = np.clip((1.0 - gamma_m) * eig_raw_m + gamma_m * mu_m, 0.0, None)
        eig_lw_s = np.clip((1.0 - gamma_s) * eig_raw_s + gamma_s * mu_s, 0.0, None)

        # Spectral alignment on top-k right singular vectors
        align = np.abs(Vt_m @ Vt_s.T)  # [k_align, k_align]
        top1_alignment = float(align.max())
        mean_alignment = float(align.mean())
        top20 = min(20, k_align)
        top20_max = float(align[:top20, :top20].max())
        top20_mean = float(align[:top20, :top20].mean())
        n_intrusion_dirs = int(np.sum(align > intrusion_threshold))

        all_top1_alignments.append(top1_alignment)
        all_mean_alignments.append(mean_alignment)
        if n_intrusion_dirs > 0:
            intrusion_layers.append(layer)

        # Effective dimensionality from shrunk eigenvalues
        m_eff = _effective_metrics(eig_lw_m)
        s_eff = _effective_metrics(eig_lw_s)

        # Log-spectrum slope
        slope_m = _log_spectrum_slope(svals_m, start_rank=10, end_rank=1000)
        slope_s = _log_spectrum_slope(svals_s, start_rank=10, end_rank=1000)

        # Save spectra
        np.save(eigenvalue_dir / f"eigvals_math_raw_L{layer:02d}.npy", eig_raw_m)
        np.save(eigenvalue_dir / f"eigvals_sarc_raw_L{layer:02d}.npy", eig_raw_s)
        np.save(eigenvalue_dir / f"eigvals_math_lw_L{layer:02d}.npy", eig_lw_m)
        np.save(eigenvalue_dir / f"eigvals_sarc_lw_L{layer:02d}.npy", eig_lw_s)
        np.save(eigenvalue_dir / f"svals_math_L{layer:02d}.npy", svals_m)
        np.save(eigenvalue_dir / f"svals_sarc_L{layer:02d}.npy", svals_s)

        results["per_layer"][str(layer)] = {
            "top1_alignment": top1_alignment,
            "top20_max_alignment": top20_max,
            "top20_mean_alignment": top20_mean,
            "top50_max_alignment": top1_alignment,
            "top50_mean_alignment": mean_alignment,
            "n_intrusion_dirs": n_intrusion_dirs,
            "lw_shrinkage_math": float(gamma_m),
            "lw_shrinkage_sarc": float(gamma_s),
            "eff_dim_math": m_eff["eff_dim"],
            "eff_dim_sarc": s_eff["eff_dim"],
            "stable_rank_math": m_eff["stable_rank"],
            "stable_rank_sarc": s_eff["stable_rank"],
            "k80_math": m_eff["k80"],
            "k90_math": m_eff["k90"],
            "k95_math": m_eff["k95"],
            "k80_sarc": s_eff["k80"],
            "k90_sarc": s_eff["k90"],
            "k95_sarc": s_eff["k95"],
            "condition_number_math": m_eff["condition_number"],
            "condition_number_sarc": s_eff["condition_number"],
            "log_sval_slope_math": slope_m,
            "log_sval_slope_sarc": slope_s,
            "top5_eigvals_math": eig_raw_m[:5].tolist(),
            "top5_eigvals_sarc": eig_raw_s[:5].tolist(),
            "top5_eigvals_math_lw": eig_lw_m[:5].tolist(),
            "top5_eigvals_sarc_lw": eig_lw_s[:5].tolist(),
            "median_eigval_math": float(np.median(eig_lw_m[:min(50, eig_lw_m.size)])),
            "median_eigval_sarc": float(np.median(eig_lw_s[:min(50, eig_lw_s.size)])),
        }

        # Free memory
        del Xm, Xs, eig_raw_m, eig_raw_s, eig_lw_m, eig_lw_s
        del svals_m, svals_s, Vt_m, Vt_s, align
        gc.collect()

    # Summary statistics
    results["summary"] = {
        "global_max_alignment": float(np.max(all_top1_alignments)),
        "global_mean_alignment": float(np.mean(all_top1_alignments)),
        "n_intrusion_layers": len(intrusion_layers),
        "intrusion_layers": intrusion_layers,
        "mean_eff_dim_math": float(np.mean([
            results["per_layer"][str(l)]["eff_dim_math"] for l in range(NUM_LAYERS)
        ])),
        "mean_eff_dim_sarc": float(np.mean([
            results["per_layer"][str(l)]["eff_dim_sarc"] for l in range(NUM_LAYERS)
        ])),
        "comparison_with_phase1": {
            "phase1_n_samples": 200,
            "phase1_rank": 200,
            "phase2_n_samples": min(n_math, n_sarc),
            "phase2_rank": min(min(n_math, n_sarc), HIDDEN_DIM),
            "phase1_max_alignment": 0.9606,  # From GMR Phase 1
            "phase1_mean_top1": 0.0964,
            "phase2_max_alignment": float(np.max(all_top1_alignments)),
            "phase2_mean_top1": float(np.mean(all_top1_alignments)),
        },
    }

    return results


def main():
    parser = argparse.ArgumentParser(description="Full-rank spectral analysis")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument("--skip-capture", action="store_true",
                        help="Skip activation capture (use cached .npy files)")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Batch size for forward passes (default: 1)")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(exist_ok=True)

    # Load prompts
    math_prompts, sarc_prompts = load_prompts()

    if not args.skip_capture:
        # Load model
        model, processor = load_model()

        # Setup activation capture
        capture = ActivationCapture(model)

        # Phase 1: Collect math activations
        print("\n" + "=" * 60)
        print("PHASE 1: Collecting math activations")
        print("=" * 60)
        math_acts = collect_activations(
            model, processor, math_prompts, capture, "math",
            batch_size=args.batch_size, resume=args.resume,
        )

        # Phase 2: Collect sarcasm activations
        print("\n" + "=" * 60)
        print("PHASE 2: Collecting sarcasm activations")
        print("=" * 60)
        sarc_acts = collect_activations(
            model, processor, sarc_prompts, capture, "sarc",
            batch_size=args.batch_size, resume=args.resume,
        )

        # Cleanup GPU
        capture.remove_hooks()
        del model, processor, capture
        torch.cuda.empty_cache()
        gc.collect()
    else:
        # Load cached activations
        math_path = OUTPUT_DIR / "activations_math.npy"
        sarc_path = OUTPUT_DIR / "activations_sarc.npy"
        if not math_path.exists() or not sarc_path.exists():
            raise FileNotFoundError("Cached activations not found. Run without --skip-capture first.")
        print("Loading cached activations...")
        math_acts = np.load(math_path)
        sarc_acts = np.load(sarc_path)
        print(f"  Math: {math_acts.shape}, Sarc: {sarc_acts.shape}")

    # Phase 3: Spectral analysis
    print("\n" + "=" * 60)
    print("PHASE 3: Full-rank spectral analysis")
    print("=" * 60)
    results = compute_spectral_analysis(math_acts, sarc_acts)

    # Save results
    report_path = OUTPUT_DIR / "fullrank_spectral_report.json"
    with open(report_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nReport saved to {report_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("FULL-RANK SPECTRAL ANALYSIS SUMMARY")
    print("=" * 60)
    s = results["summary"]
    print(f"  Samples: {results['metadata']['n_math']} math + {results['metadata']['n_sarc']} sarc")
    print(f"  Rank: {results['metadata']['rank_math']} (vs Phase 1: 200)")
    print(f"  Global max alignment: {s['global_max_alignment']:.4f} "
          f"(Phase 1: {s['comparison_with_phase1']['phase1_max_alignment']:.4f})")
    print(f"  Global mean top-1: {s['global_mean_alignment']:.4f} "
          f"(Phase 1: {s['comparison_with_phase1']['phase1_mean_top1']:.4f})")
    print(f"  Intrusion layers (align > 0.5): {s['n_intrusion_layers']}")
    if s["intrusion_layers"]:
        print(f"    Layers: {s['intrusion_layers']}")
    print(f"  Mean effective dim (math): {s['mean_eff_dim_math']:.0f}")
    print(f"  Mean effective dim (sarc): {s['mean_eff_dim_sarc']:.0f}")

    # Phase 1 vs Phase 2 comparison
    comp = s["comparison_with_phase1"]
    print(f"\n  Phase 1 → Phase 2 comparison:")
    print(f"    Max alignment: {comp['phase1_max_alignment']:.4f} → {comp['phase2_max_alignment']:.4f}")
    print(f"    Mean top-1:    {comp['phase1_mean_top1']:.4f} → {comp['phase2_mean_top1']:.4f}")
    if comp["phase2_max_alignment"] > comp["phase1_max_alignment"]:
        print(f"    *** ALIGNMENT INCREASED — tail dimensions may show intrusion! ***")
    elif comp["phase2_max_alignment"] < comp["phase1_max_alignment"] * 0.9:
        print(f"    Alignment decreased — Phase 1 top eigenvectors may have been noisy")
    else:
        print(f"    Alignment stable — zero-intrusion finding VALIDATED at full rank")

    # Layer band analysis
    print("\n  Layer band analysis:")
    for band_name, band_range in [
        ("early (L0-15)", range(0, 16)),
        ("mid_early (L16-31)", range(16, 32)),
        ("mid_late (L32-47)", range(32, 48)),
        ("late (L48-63)", range(48, 64)),
    ]:
        band_aligns = [
            results["per_layer"][str(l)]["top20_max_alignment"] for l in band_range
        ]
        print(f"    {band_name}: mean_top20_max={np.mean(band_aligns):.4f}, "
              f"max={np.max(band_aligns):.4f}")

    print(f"\nDone. Total disk usage: "
          f"{sum(f.stat().st_size for f in OUTPUT_DIR.rglob('*') if f.is_file()) / 1e9:.1f} GB")


if __name__ == "__main__":
    main()
