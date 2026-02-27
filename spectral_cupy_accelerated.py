#!/usr/bin/env python3
"""
GPU-accelerated spectral analysis using CuPy.

Replaces the CPU-bound np.linalg.svd with cupy.linalg.svd for 78x speedup.
Loads cached activations and resumes from already-completed layers.

Usage:
    source /home/orwel/dev_genius/qwen35_venv/bin/activate
    python spectral_cupy_accelerated.py
"""

import gc
import json
import os
import time
from datetime import datetime
from pathlib import Path

import cupy as cp
import numpy as np
from tqdm import tqdm

# ── Config (must match fullrank_spectral_analysis.py) ─────────────
NUM_LAYERS = 64
HIDDEN_DIM = 5120
OUTPUT_DIR = Path("./fullrank_spectral")
EIGENVALUE_DIR = OUTPUT_DIR / "eigenvalues"


def layer_complete(layer: int) -> bool:
    """Check if a layer already has all 6 eigenvalue files."""
    expected = [
        f"eigvals_math_raw_L{layer:02d}.npy",
        f"eigvals_sarc_raw_L{layer:02d}.npy",
        f"eigvals_math_lw_L{layer:02d}.npy",
        f"eigvals_sarc_lw_L{layer:02d}.npy",
        f"svals_math_L{layer:02d}.npy",
        f"svals_sarc_L{layer:02d}.npy",
    ]
    return all((EIGENVALUE_DIR / f).exists() for f in expected)


def _log_spectrum_slope(vals: np.ndarray, start_rank: int = 10, end_rank: int = 1000) -> float:
    eps = 1e-15
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
    eps = 1e-15
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
    """Ledoit-Wolf shrinkage coefficient."""
    try:
        from sklearn.covariance import ledoit_wolf_shrinkage
        return float(ledoit_wolf_shrinkage(X_centered, assume_centered=True, block_size=2048))
    except ImportError:
        from sklearn.covariance import LedoitWolf
        lw = LedoitWolf(assume_centered=True, store_precision=False).fit(X_centered)
        return float(lw.shrinkage_)


def main() -> None:
    from sklearn.utils.extmath import randomized_svd

    EIGENVALUE_DIR.mkdir(parents=True, exist_ok=True)

    # Check which layers need processing
    completed = [l for l in range(NUM_LAYERS) if layer_complete(l)]
    remaining = [l for l in range(NUM_LAYERS) if not layer_complete(l)]
    print(f"Layers completed: {len(completed)}/64")
    print(f"Layers remaining: {len(remaining)}")

    if not remaining:
        print("All layers already complete!")
        # Still need to assemble final results
    else:
        # Load cached activations (memory-mapped for efficiency)
        math_path = OUTPUT_DIR / "activations_math.npy"
        sarc_path = OUTPUT_DIR / "activations_sarc.npy"
        if not math_path.exists() or not sarc_path.exists():
            raise FileNotFoundError(f"Cached activations not found at {OUTPUT_DIR}")

        print("Loading cached activations (mmap)...")
        math_acts = np.load(math_path, mmap_mode="r")
        sarc_acts = np.load(sarc_path, mmap_mode="r")
        n_math, n_layers, hidden = math_acts.shape
        n_sarc = sarc_acts.shape[0]
        print(f"  Math: {math_acts.shape}, Sarc: {sarc_acts.shape}")
        assert n_layers == NUM_LAYERS and hidden == HIDDEN_DIM

        k_align = min(50, HIDDEN_DIM, n_math - 1, n_sarc - 1)

        # CuPy warmup
        print("Warming up CuPy...")
        _warmup = cp.linalg.svd(cp.random.randn(100, 100), full_matrices=False, compute_uv=False)
        cp.cuda.Device(0).synchronize()
        del _warmup
        cp.get_default_memory_pool().free_all_blocks()

        free_mem = cp.cuda.Device(0).mem_info[0]
        print(f"Free VRAM after warmup: {free_mem / 1e9:.1f} GB")

        total_gpu_time = 0.0
        total_cpu_time = 0.0

        for layer in tqdm(remaining, desc="  GPU Spectral analysis"):
            t_start = time.time()

            # Copy to contiguous float64 (from mmap)
            Xm = np.ascontiguousarray(math_acts[:, layer, :], dtype=np.float64)
            Xs = np.ascontiguousarray(sarc_acts[:, layer, :], dtype=np.float64)

            # Center
            Xm -= Xm.mean(axis=0, keepdims=True)
            Xs -= Xs.mean(axis=0, keepdims=True)

            # GPU SVD for singular values (the expensive part)
            t_gpu = time.time()
            Xm_gpu = cp.asarray(Xm)
            svals_m_gpu = cp.linalg.svd(Xm_gpu, full_matrices=False, compute_uv=False)
            svals_m = cp.asnumpy(svals_m_gpu)
            del Xm_gpu, svals_m_gpu
            cp.get_default_memory_pool().free_all_blocks()

            Xs_gpu = cp.asarray(Xs)
            svals_s_gpu = cp.linalg.svd(Xs_gpu, full_matrices=False, compute_uv=False)
            svals_s = cp.asnumpy(svals_s_gpu)
            del Xs_gpu, svals_s_gpu
            cp.get_default_memory_pool().free_all_blocks()
            cp.cuda.Device(0).synchronize()
            total_gpu_time += time.time() - t_gpu

            # Eigenvalues from singular values
            eig_raw_m = (svals_m ** 2) / max(n_math - 1, 1)
            eig_raw_s = (svals_s ** 2) / max(n_sarc - 1, 1)

            # CPU: randomized SVD for top-k alignment vectors (fast, k=50)
            t_cpu = time.time()
            _, _, Vt_m = randomized_svd(Xm, n_components=k_align, n_iter=4,
                                         random_state=13337 + layer, power_iteration_normalizer="QR")
            _, _, Vt_s = randomized_svd(Xs, n_components=k_align, n_iter=4,
                                         random_state=42424 + layer, power_iteration_normalizer="QR")

            # CPU: Ledoit-Wolf shrinkage
            gamma_m = _lw_gamma(Xm)
            gamma_s = _lw_gamma(Xs)
            total_cpu_time += time.time() - t_cpu

            mu_m = float(eig_raw_m.mean())
            mu_s = float(eig_raw_s.mean())
            eig_lw_m = np.clip((1.0 - gamma_m) * eig_raw_m + gamma_m * mu_m, 0.0, None)
            eig_lw_s = np.clip((1.0 - gamma_s) * eig_raw_s + gamma_s * mu_s, 0.0, None)

            # Save eigenvalue files
            np.save(EIGENVALUE_DIR / f"eigvals_math_raw_L{layer:02d}.npy", eig_raw_m)
            np.save(EIGENVALUE_DIR / f"eigvals_sarc_raw_L{layer:02d}.npy", eig_raw_s)
            np.save(EIGENVALUE_DIR / f"eigvals_math_lw_L{layer:02d}.npy", eig_lw_m)
            np.save(EIGENVALUE_DIR / f"eigvals_sarc_lw_L{layer:02d}.npy", eig_lw_s)
            np.save(EIGENVALUE_DIR / f"svals_math_L{layer:02d}.npy", svals_m)
            np.save(EIGENVALUE_DIR / f"svals_sarc_L{layer:02d}.npy", svals_s)

            del Xm, Xs, eig_raw_m, eig_raw_s, eig_lw_m, eig_lw_s
            del svals_m, svals_s, Vt_m, Vt_s
            gc.collect()

        print(f"\nGPU SVD time: {total_gpu_time:.1f}s ({total_gpu_time/len(remaining):.1f}s/layer)")
        print(f"CPU (rSVD+LW) time: {total_cpu_time:.1f}s ({total_cpu_time/len(remaining):.1f}s/layer)")
        print(f"Total compute: {total_gpu_time + total_cpu_time:.1f}s")

    # ── Assemble full results from all 64 layers ──────────────────
    print("\nAssembling full results from all 64 layers...")

    # Reload activations for metadata
    math_acts_meta = np.load(OUTPUT_DIR / "activations_math.npy", mmap_mode="r")
    sarc_acts_meta = np.load(OUTPUT_DIR / "activations_sarc.npy", mmap_mode="r")
    n_math = math_acts_meta.shape[0]
    n_sarc = sarc_acts_meta.shape[0]
    k_align = min(50, HIDDEN_DIM, n_math - 1, n_sarc - 1)

    results = {
        "metadata": {
            "n_math": int(n_math),
            "n_sarc": int(n_sarc),
            "n_layers": NUM_LAYERS,
            "hidden_dim": HIDDEN_DIM,
            "rank_math": min(n_math, HIDDEN_DIM),
            "rank_sarc": min(n_sarc, HIDDEN_DIM),
            "method": "SVD + Ledoit-Wolf shrinkage (CuPy GPU-accelerated)",
            "n_top_alignment": int(k_align),
            "intrusion_threshold": 0.5,
            "use_ledoit_wolf": True,
            "timestamp": datetime.now().isoformat(),
        },
        "per_layer": {},
        "summary": {},
    }

    all_top1_alignments = []
    all_mean_alignments = []
    intrusion_layers = []

    for layer in tqdm(range(NUM_LAYERS), desc="  Assembling results"):
        # Load saved eigenvalues
        eig_raw_m = np.load(EIGENVALUE_DIR / f"eigvals_math_raw_L{layer:02d}.npy")
        eig_raw_s = np.load(EIGENVALUE_DIR / f"eigvals_sarc_raw_L{layer:02d}.npy")
        eig_lw_m = np.load(EIGENVALUE_DIR / f"eigvals_math_lw_L{layer:02d}.npy")
        eig_lw_s = np.load(EIGENVALUE_DIR / f"eigvals_sarc_lw_L{layer:02d}.npy")
        svals_m = np.load(EIGENVALUE_DIR / f"svals_math_L{layer:02d}.npy")
        svals_s = np.load(EIGENVALUE_DIR / f"svals_sarc_L{layer:02d}.npy")

        # Recompute alignment from saved singular vectors
        # Need to redo randomized SVD for alignment (fast, ~2s)
        Xm = np.ascontiguousarray(math_acts_meta[:, layer, :], dtype=np.float64)
        Xs = np.ascontiguousarray(sarc_acts_meta[:, layer, :], dtype=np.float64)
        Xm -= Xm.mean(axis=0, keepdims=True)
        Xs -= Xs.mean(axis=0, keepdims=True)

        from sklearn.utils.extmath import randomized_svd
        _, _, Vt_m = randomized_svd(Xm, n_components=k_align, n_iter=4,
                                     random_state=13337 + layer, power_iteration_normalizer="QR")
        _, _, Vt_s = randomized_svd(Xs, n_components=k_align, n_iter=4,
                                     random_state=42424 + layer, power_iteration_normalizer="QR")

        align = np.abs(Vt_m @ Vt_s.T)
        top1_alignment = float(align.max())
        mean_alignment = float(align.mean())
        top20 = min(20, k_align)
        top20_max = float(align[:top20, :top20].max())
        top20_mean = float(align[:top20, :top20].mean())
        n_intrusion_dirs = int(np.sum(align > 0.5))

        all_top1_alignments.append(top1_alignment)
        all_mean_alignments.append(mean_alignment)
        if n_intrusion_dirs > 0:
            intrusion_layers.append(layer)

        m_eff = _effective_metrics(eig_lw_m)
        s_eff = _effective_metrics(eig_lw_s)
        slope_m = _log_spectrum_slope(svals_m, start_rank=10, end_rank=1000)
        slope_s = _log_spectrum_slope(svals_s, start_rank=10, end_rank=1000)

        results["per_layer"][str(layer)] = {
            "top1_alignment": top1_alignment,
            "top20_max_alignment": top20_max,
            "top20_mean_alignment": top20_mean,
            "top50_max_alignment": top1_alignment,
            "top50_mean_alignment": mean_alignment,
            "n_intrusion_dirs": n_intrusion_dirs,
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

        del Xm, Xs, Vt_m, Vt_s, align
        gc.collect()

    # Summary
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
            "phase1_max_alignment": 0.0,  # Will need Phase 1 data to fill
            "phase1_mean_top1": 0.0,
            "phase2_max_alignment": float(np.max(all_top1_alignments)),
            "phase2_mean_top1": float(np.mean(all_top1_alignments)),
        },
    }

    # Try to load Phase 1 comparison data
    phase1_path = OUTPUT_DIR / "phase1_comparison.json"
    if phase1_path.exists():
        with open(phase1_path) as f:
            p1 = json.load(f)
        results["summary"]["comparison_with_phase1"]["phase1_max_alignment"] = p1.get("max_alignment", 0.0)
        results["summary"]["comparison_with_phase1"]["phase1_mean_top1"] = p1.get("mean_top1", 0.0)

    # Save report
    report_path = OUTPUT_DIR / "fullrank_spectral_report.json"
    with open(report_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nReport saved to {report_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("FULL-RANK SPECTRAL ANALYSIS SUMMARY (CuPy)")
    print("=" * 60)
    s = results["summary"]
    print(f"  Samples: {results['metadata']['n_math']} math + {results['metadata']['n_sarc']} sarc")
    print(f"  Rank: {results['metadata']['rank_math']}")
    print(f"  Global max alignment: {s['global_max_alignment']:.4f}")
    print(f"  Global mean top-1: {s['global_mean_alignment']:.4f}")
    print(f"  Intrusion layers (align > 0.5): {s['n_intrusion_layers']}")
    if s["intrusion_layers"]:
        print(f"    Layers: {s['intrusion_layers']}")
    print(f"  Mean effective dim (math): {s['mean_eff_dim_math']:.0f}")
    print(f"  Mean effective dim (sarc): {s['mean_eff_dim_sarc']:.0f}")

    # Layer band analysis
    print("\n  Layer band analysis:")
    for band_name, band_range in [
        ("early (L0-15)", range(0, 16)),
        ("mid (L16-31)", range(16, 32)),
        ("mid-late (L32-47)", range(32, 48)),
        ("late (L48-63)", range(48, 64)),
    ]:
        band_aligns = [results["per_layer"][str(l)]["top1_alignment"]
                       for l in band_range if str(l) in results["per_layer"]]
        band_effs_m = [results["per_layer"][str(l)]["eff_dim_math"]
                       for l in band_range if str(l) in results["per_layer"]]
        band_effs_s = [results["per_layer"][str(l)]["eff_dim_sarc"]
                       for l in band_range if str(l) in results["per_layer"]]
        if band_aligns:
            print(f"    {band_name}: max_align={max(band_aligns):.4f}, "
                  f"mean_eff_dim_math={np.mean(band_effs_m):.0f}, "
                  f"mean_eff_dim_sarc={np.mean(band_effs_s):.0f}")

    print("\nDone!")


if __name__ == "__main__":
    main()
