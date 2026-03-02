#!/usr/bin/env python3
"""
Baseline Analysis Suite — 5 analyses on 50K FineFineWeb activations.

Analyses:
  1. Domain confound detector: 68-domain classifier vs personality directions
  2. Per-domain conditional whitening of personality Cohen's d (Mahalanobis)
  3. Empirical null distribution per neuron (baseline vs personality, SE-based)
  4. Covariance convergence curve (2K→50K subsets, truncated SVD)
  5. Train/test split false-positive probe check (25K/25K, repeated null)

Usage:
  python scripts/experiments/personality/baseline_analysis.py \
    --baseline-dir activations_baseline \
    --sweep-dir sweep_output/blackwell \
    --output-dir results/baseline_analysis \
    --layers 9,15,22,29

Codex review: 2026-03-01. Fixes applied:
  - Analysis 1: Pipeline CV (no leakage), weights converted to raw space
  - Analysis 2: True Mahalanobis d via LedoitWolf covariance (not cancelled scaling)
  - Analysis 3: SE-based z-scores (not raw σ), permutation null with matched sizes
  - Analysis 4: Truncated SVD (sklearn) instead of full np.linalg.svd
  - Analysis 5: Real train/test split with repeated null draws
"""

from __future__ import annotations

import argparse
import json
import re
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tqdm import tqdm


# ── Data Loaders ─────────────────────────────────────────────────────────


def _shard_id(p: Path) -> str:
    """Extract numeric shard ID from filename."""
    m = re.search(r"mean_shard_(\d+)", p.name)
    if not m:
        raise ValueError(f"Bad shard filename: {p}")
    return m.group(1)


def load_activation_shards(act_dir: Path, layer: int) -> tuple[torch.Tensor, list[dict]]:
    """Load all shards for a layer. Returns (N, hidden_dim) tensor + metadata list.

    Pairs shard .pt files with their _meta.jsonl by shard ID to prevent
    misalignment if any file is missing.
    """
    layer_dir = act_dir / f"L{layer:02d}"
    if not layer_dir.exists():
        raise FileNotFoundError(f"Layer directory not found: {layer_dir}")

    shard_files = {_shard_id(p): p for p in layer_dir.glob("mean_shard_*.pt")
                   if "_meta" not in p.name}
    meta_files = {_shard_id(p): p for p in layer_dir.glob("mean_shard_*_meta.jsonl")}

    if not shard_files:
        raise FileNotFoundError(f"No shards in {layer_dir}")

    # Validate pairing
    if set(shard_files) != set(meta_files):
        missing_meta = set(shard_files) - set(meta_files)
        missing_pt = set(meta_files) - set(shard_files)
        raise ValueError(f"Mismatched shards. Missing meta={missing_meta}, missing pt={missing_pt}")

    common = sorted(set(shard_files) & set(meta_files))
    tensors: list[torch.Tensor] = []
    metadata: list[dict] = []
    hidden_dim: int | None = None

    for sid in common:
        t = torch.load(shard_files[sid], map_location="cpu", weights_only=True).float()
        rows = [json.loads(line) for line in open(meta_files[sid]) if line.strip()]
        if t.shape[0] != len(rows):
            raise ValueError(f"Shard {sid}: tensor rows {t.shape[0]} != meta rows {len(rows)}")
        if hidden_dim is None:
            hidden_dim = t.shape[1]
        elif t.shape[1] != hidden_dim:
            raise ValueError(f"Inconsistent hidden dim in shard {sid}: {t.shape[1]} vs {hidden_dim}")
        tensors.append(t)
        metadata.extend(rows)

    combined = torch.cat(tensors, dim=0)
    return combined, metadata


def load_sweep_with_b5(sweep_dir: Path, layer: int) -> tuple[np.ndarray, list[dict], dict[str, list[str]]]:
    """Load personality sweep activations + parse Big Five labels from metadata."""
    act_dir = sweep_dir / "activations"
    tensor, meta = load_activation_shards(act_dir, layer)
    acts = tensor.numpy()

    # Parse Big Five combo into per-dimension labels
    # Format: "H_M_L_H_L" → O=H, C=M, E=L, A=H, N=L
    b5_dims = ["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"]
    b5_labels: dict[str, list[str]] = {d: [] for d in b5_dims}
    n_malformed = 0

    for m in meta:
        combo = m.get("b5_combo", "M_M_M_M_M")
        levels = combo.split("_")
        if len(levels) != 5 or not all(l in ("H", "M", "L") for l in levels):
            n_malformed += 1
            levels = ["M"] * 5
        for i, dim in enumerate(b5_dims):
            b5_labels[dim].append(levels[i])

    if n_malformed > 0:
        print(f"  WARNING: {n_malformed}/{len(meta)} malformed b5_combo entries (defaulted to M)")

    return acts, meta, b5_labels


def load_baseline_with_domain(baseline_dir: Path, layer: int) -> tuple[np.ndarray, list[dict], list[str]]:
    """Load baseline activations + domain labels."""
    act_dir = baseline_dir / "activations"
    tensor, meta = load_activation_shards(act_dir, layer)
    acts = tensor.numpy()

    n_unknown = 0
    domains: list[str] = []
    for m in meta:
        d = m.get("domain", "")
        if not d:
            n_unknown += 1
            d = "unknown"
        domains.append(d)

    if n_unknown > 0:
        print(f"  WARNING: {n_unknown}/{len(meta)} baseline samples with missing domain")

    return acts, meta, domains


# ── Analysis 1: Domain Confound Detector ─────────────────────────────────
# Codex fix: Pipeline CV (no scaler leakage), convert weights to raw space


def analysis_domain_confound(
    baseline_acts: np.ndarray,
    domains: list[str],
    sweep_acts: np.ndarray,
    b5_labels: dict[str, list[str]],
    layer: int,
) -> dict[str, Any]:
    """Train domain classifier, extract directions, compare to personality directions."""
    print(f"\n{'='*60}")
    print(f"Analysis 1: Domain Confound Detector (L{layer:02d})")
    print(f"{'='*60}")

    # Encode domains, filter rare ones for stable CV
    le = LabelEncoder()
    domain_encoded = le.fit_transform(domains)
    n_domains = len(le.classes_)

    # Filter domains with <5 samples (StratifiedKFold requires >=n_splits per class)
    domain_counts = np.bincount(domain_encoded)
    valid_mask = domain_counts[domain_encoded] >= 5
    X_valid = baseline_acts[valid_mask]
    y_valid = domain_encoded[valid_mask]
    n_valid_domains = len(np.unique(y_valid)) if len(y_valid) > 0 else 0
    print(f"  Domains: {n_domains} total, {n_valid_domains} with >=5 samples, {X_valid.shape[0]} samples")

    # Edge-case guard: need >=2 classes for classification (Codex v2 fix A)
    if X_valid.shape[0] == 0 or n_valid_domains < 2:
        print(f"  SKIP: Insufficient valid domains after filtering (need >=2 classes).")
        return {
            "layer": layer,
            "error": "Insufficient valid domains after filtering (need >=2 classes).",
        }

    min_class_count = int(np.bincount(y_valid).min())
    n_splits = min(5, min_class_count)
    if n_splits < 2:
        print(f"  SKIP: Insufficient per-class samples for CV (min_class_count={min_class_count}).")
        return {
            "layer": layer,
            "error": f"Insufficient per-class samples for CV (min_class_count={min_class_count}).",
        }

    # Pipeline: scaler inside CV to prevent leakage
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            max_iter=500, C=1.0, solver="saga", multi_class="ovr",
            n_jobs=-1, random_state=42,
        )),
    ])

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    cv_scores = cross_val_score(pipe, X_valid, y_valid, cv=cv, scoring="balanced_accuracy")
    print(f"  Domain classifier 5-fold CV balanced accuracy: {cv_scores.mean():.3f} +/- {cv_scores.std():.3f}")
    print(f"  Chance level: {1/n_valid_domains:.3f}")

    # Fit on all data to get domain weight vectors (in raw feature space)
    pipe.fit(X_valid, y_valid)
    scaler = pipe.named_steps["scaler"]
    clf = pipe.named_steps["clf"]
    # Convert weights from scaled space to raw space: W_raw = W_scaled / scale
    domain_weights_raw = clf.coef_ / scaler.scale_[np.newaxis, :]  # (n_classes, hidden_dim)

    # Compute top-k domain directions via SVD on raw-space weights
    U, S, Vt = np.linalg.svd(domain_weights_raw, full_matrices=False)
    domain_directions = Vt[:10]  # Top 10 domain PCs (unit-normed rows of Vt)

    # Compute Big Five personality directions (high_mean - low_mean)
    b5_dims = ["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"]
    personality_directions: dict[str, np.ndarray] = {}

    for dim in b5_dims:
        labels = b5_labels[dim]
        high_mask = np.array([l == "H" for l in labels])
        low_mask = np.array([l == "L" for l in labels])
        if high_mask.sum() > 0 and low_mask.sum() > 0:
            high_mean = sweep_acts[high_mask].mean(axis=0)
            low_mean = sweep_acts[low_mask].mean(axis=0)
            direction = high_mean - low_mean
            direction = direction / (np.linalg.norm(direction) + 1e-10)
            personality_directions[dim] = direction

    # Compute cosine overlap between personality and domain directions
    overlaps: dict[str, list[float]] = {}
    for dim, p_dir in personality_directions.items():
        cos_sims = [float(abs(np.dot(p_dir, d_dir / (np.linalg.norm(d_dir) + 1e-10))))
                     for d_dir in domain_directions]
        overlaps[dim] = cos_sims
        max_overlap = max(cos_sims)
        mean_overlap = float(np.mean(cos_sims))
        print(f"  {dim:20s} | max|cos| with domain PCs: {max_overlap:.3f}, mean: {mean_overlap:.3f}")

    # Subspace overlap: project personality onto domain subspace
    # Normalize domain directions first
    domain_dirs_normed = np.array([d / (np.linalg.norm(d) + 1e-10) for d in domain_directions])
    subspace_overlaps: dict[str, float] = {}
    for dim, p_dir in personality_directions.items():
        proj = domain_dirs_normed @ p_dir  # (10,)
        subspace_overlap = float(np.linalg.norm(proj))
        subspace_overlaps[dim] = subspace_overlap

    print(f"\n  Subspace overlap (||proj onto top-10 domain PCs||):")
    for dim, ov in subspace_overlaps.items():
        status = "CLEAN" if ov < 0.3 else "CAUTION" if ov < 0.5 else "CONFOUNDED"
        print(f"    {dim:20s}: {ov:.3f} [{status}]")

    return {
        "layer": layer,
        "n_domains": n_domains,
        "n_valid_domains": n_valid_domains,
        "cv_balanced_accuracy": float(cv_scores.mean()),
        "cv_std": float(cv_scores.std()),
        "chance_level": float(1 / n_valid_domains),
        "domain_singular_values": S[:10].tolist(),
        "personality_domain_overlaps": {d: v for d, v in overlaps.items()},
        "subspace_overlaps": subspace_overlaps,
    }


# ── Analysis 2: Mahalanobis d with LedoitWolf covariance ─────────────────
# Codex fix: True Mahalanobis distance, not cancelled per-feature scaling


def analysis_conditional_whitening(
    baseline_acts: np.ndarray,
    domains: list[str],
    sweep_acts: np.ndarray,
    b5_labels: dict[str, list[str]],
    layer: int,
) -> dict[str, Any]:
    """Compute Mahalanobis Cohen's d using baseline covariance (LedoitWolf)."""
    print(f"\n{'='*60}")
    print(f"Analysis 2: Mahalanobis Whitening (L{layer:02d})")
    print(f"{'='*60}")

    hidden_dim = baseline_acts.shape[1]

    # Global baseline covariance (LedoitWolf for stability with p > n risk)
    print(f"  Fitting LedoitWolf covariance on {baseline_acts.shape[0]} baseline samples...")
    lw_global = LedoitWolf().fit(baseline_acts)
    cov_global = lw_global.covariance_  # (4096, 4096)
    global_mean = lw_global.location_   # (4096,)

    # Invert via eigendecomposition (regularized)
    eigvals, eigvecs = np.linalg.eigh(cov_global)
    # Clip tiny eigenvalues for numerical stability
    eigvals_clipped = np.maximum(eigvals, 1e-6)
    cov_inv_global = (eigvecs / eigvals_clipped) @ eigvecs.T

    # Within-domain residual covariance
    print(f"  Computing within-domain residual covariance...")
    unique_domains = list(set(domains))
    domain_masks: dict[str, np.ndarray] = {}
    domain_means: dict[str, np.ndarray] = {}
    for d in unique_domains:
        mask = np.array([x == d for x in domains])
        if mask.sum() >= 10:
            domain_masks[d] = mask
            domain_means[d] = baseline_acts[mask].mean(axis=0)

    # Pool within-domain residuals (Codex v2 fix B: handle empty domain_masks)
    residuals = [baseline_acts[mask] - domain_means[d] for d, mask in domain_masks.items()]
    if len(residuals) == 0:
        warnings.warn("No domains with >=10 samples; falling back to globally centered residual covariance.")
        residuals_all = baseline_acts - baseline_acts.mean(axis=0, keepdims=True)
    else:
        residuals_all = np.concatenate(residuals, axis=0)

    lw_within = LedoitWolf().fit(residuals_all)
    cov_within = lw_within.covariance_
    eigvals_w, eigvecs_w = np.linalg.eigh(cov_within)
    eigvals_w_clipped = np.maximum(eigvals_w, 1e-6)
    cov_inv_within = (eigvecs_w / eigvals_w_clipped) @ eigvecs_w.T

    b5_dims = ["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"]
    results: dict[str, dict[str, float]] = {}

    for dim in b5_dims:
        labels = b5_labels[dim]
        high_mask = np.array([l == "H" for l in labels])
        low_mask = np.array([l == "L" for l in labels])
        if high_mask.sum() == 0 or low_mask.sum() == 0:
            continue

        delta = sweep_acts[high_mask].mean(axis=0) - sweep_acts[low_mask].mean(axis=0)

        # Raw Cohen's d (diagonal, per-feature pooled std)
        high_var = sweep_acts[high_mask].var(axis=0)
        low_var = sweep_acts[low_mask].var(axis=0)
        pooled_std = np.sqrt((high_var + low_var) / 2) + 1e-10
        raw_d = float(np.linalg.norm(delta / pooled_std))

        # Mahalanobis d with global baseline covariance: sqrt(Δ^T Σ^-1 Δ)
        global_mahal_d = float(np.sqrt(max(0, delta @ cov_inv_global @ delta)))

        # Mahalanobis d with within-domain residual covariance
        domain_mahal_d = float(np.sqrt(max(0, delta @ cov_inv_within @ delta)))

        results[dim] = {
            "raw_d": raw_d,
            "global_mahalanobis_d": global_mahal_d,
            "within_domain_mahalanobis_d": domain_mahal_d,
            "retention_global": global_mahal_d / raw_d if raw_d > 0 else 0,
            "retention_domain": domain_mahal_d / raw_d if raw_d > 0 else 0,
        }

        # Codex v2 fix B: guard raw_d == 0 in print
        g_pct = (global_mahal_d / raw_d * 100) if raw_d > 0 else float("nan")
        d_pct = (domain_mahal_d / raw_d * 100) if raw_d > 0 else float("nan")
        print(f"  {dim:20s} | raw: {raw_d:.2f} -> global Mahal: {global_mahal_d:.2f} ({g_pct:.0f}%) -> domain Mahal: {domain_mahal_d:.2f} ({d_pct:.0f}%)")

    return {"layer": layer, "cohens_d": results}


# ── Analysis 3: Per-Neuron Null Distribution (SE-based) ──────────────────
# Codex fix: Use standard error, not raw σ. Permutation null with matched sizes.


def analysis_neuron_null(
    baseline_acts: np.ndarray,
    sweep_acts: np.ndarray,
    b5_labels: dict[str, list[str]],
    layer: int,
) -> dict[str, Any]:
    """Build empirical null per neuron using SE-based z-scores + permutation null."""
    print(f"\n{'='*60}")
    print(f"Analysis 3: Per-Neuron Null Distribution (L{layer:02d})")
    print(f"{'='*60}")

    hidden_dim = baseline_acts.shape[1]
    n_baseline = baseline_acts.shape[0]
    b5_dims = ["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"]

    # Baseline per-neuron statistics
    baseline_std = baseline_acts.std(axis=0) + 1e-10

    results: dict[str, dict[str, Any]] = {}

    for dim in b5_dims:
        labels = b5_labels[dim]
        high_mask = np.array([l == "H" for l in labels])
        low_mask = np.array([l == "L" for l in labels])
        if high_mask.sum() == 0 or low_mask.sum() == 0:
            continue

        n_h = int(high_mask.sum())
        n_l = int(low_mask.sum())

        high_mean = sweep_acts[high_mask].mean(axis=0)
        low_mean = sweep_acts[low_mask].mean(axis=0)
        delta = high_mean - low_mean

        # SE-based z-score: z_j = delta_j / (sigma_j * sqrt(1/n_H + 1/n_L))
        se = baseline_std * np.sqrt(1.0 / n_h + 1.0 / n_l)
        z_scores = delta / se

        significant_2sigma = int(np.sum(np.abs(z_scores) > 2))
        significant_3sigma = int(np.sum(np.abs(z_scores) > 3))

        # FDR correction (Benjamini-Hochberg)
        from scipy import stats as scipy_stats
        p_values = 2 * (1 - scipy_stats.norm.cdf(np.abs(z_scores)))
        sorted_idx = np.argsort(p_values)
        sorted_p = p_values[sorted_idx]
        fdr_threshold = 0.05
        bh_critical = fdr_threshold * np.arange(1, hidden_dim + 1) / hidden_dim
        bh_reject = sorted_p <= bh_critical
        if bh_reject.any():
            max_reject_idx = np.max(np.where(bh_reject))
            n_fdr_significant = int(max_reject_idx + 1)
        else:
            n_fdr_significant = 0

        # Permutation null: split baseline into matched-size groups, same SE formula
        # Codex v2 fix C: bootstrap with replacement if n_h + n_l > n_baseline
        rng = np.random.RandomState(42)
        use_bootstrap = (n_h + n_l) > n_baseline
        if use_bootstrap:
            warnings.warn(f"{dim}: n_h+n_l ({n_h+n_l}) > n_baseline ({n_baseline}); using bootstrap (with replacement).")

        n_perm_significant = []
        for _ in range(100):
            if use_bootstrap:
                idx = rng.choice(n_baseline, size=n_h + n_l, replace=True)
                null_high = baseline_acts[idx[:n_h]].mean(axis=0)
                null_low = baseline_acts[idx[n_h:]].mean(axis=0)
            else:
                perm = rng.permutation(n_baseline)
                null_high = baseline_acts[perm[:n_h]].mean(axis=0)
                null_low = baseline_acts[perm[n_h:n_h + n_l]].mean(axis=0)
            null_delta = null_high - null_low
            null_z = null_delta / se
            n_perm_significant.append(int(np.sum(np.abs(null_z) > 2)))

        null_mean_2sigma = float(np.mean(n_perm_significant))
        null_std_2sigma = float(np.std(n_perm_significant))

        # Top 20 most discriminative neurons
        top_idx = np.argsort(np.abs(z_scores))[-20:][::-1]
        top_neurons = [
            {"dim": int(i), "z_score": float(z_scores[i]), "abs_z": float(abs(z_scores[i]))}
            for i in top_idx
        ]

        results[dim] = {
            "n_high": n_h,
            "n_low": n_l,
            "significant_2sigma": significant_2sigma,
            "significant_3sigma": significant_3sigma,
            "n_fdr_significant": n_fdr_significant,
            "pct_2sigma": float(significant_2sigma / hidden_dim * 100),
            "pct_fdr": float(n_fdr_significant / hidden_dim * 100),
            "null_mean_2sigma": null_mean_2sigma,
            "null_std_2sigma": null_std_2sigma,
            "top_20_neurons": top_neurons,
        }

        print(f"  {dim:20s} | >2sigma: {significant_2sigma} ({significant_2sigma/hidden_dim*100:.1f}%), FDR: {n_fdr_significant} ({n_fdr_significant/hidden_dim*100:.1f}%), null: {null_mean_2sigma:.0f}+/-{null_std_2sigma:.0f}")

    return {"layer": layer, "hidden_dim": hidden_dim, "neuron_null": results}


# ── Analysis 4: Covariance Convergence (Truncated SVD) ───────────────────
# Codex fix: Use TruncatedSVD instead of full np.linalg.svd


def analysis_covariance_convergence(
    baseline_acts: np.ndarray,
    layer: int,
) -> dict[str, Any]:
    """Eigenvalue stability across subset sizes using truncated SVD."""
    print(f"\n{'='*60}")
    print(f"Analysis 4: Covariance Convergence Curve (L{layer:02d})")
    print(f"{'='*60}")

    n_total = baseline_acts.shape[0]
    subset_sizes = sorted(set(
        [s for s in [2000, 5000, 10000, 25000, 50000] if s <= n_total] + [n_total]
    ))

    top_k = 50
    # Use one fixed permutation; take nested subsets for consistency
    rng = np.random.RandomState(42)
    master_perm = rng.permutation(n_total)
    all_eigenvalues: dict[int, list[float]] = {}

    for size in tqdm(subset_sizes, desc="  Convergence subsets"):
        idx = master_perm[:min(size, n_total)]
        subset = baseline_acts[idx]
        subset_centered = subset - subset.mean(axis=0)

        # Truncated SVD: only compute top-k components (much faster than full SVD)
        svd = TruncatedSVD(n_components=top_k, random_state=42)
        svd.fit(subset_centered)
        # explained_variance = singular_values^2 / (n-1)
        eigenvalues = svd.singular_values_ ** 2 / (len(subset) - 1)
        all_eigenvalues[size] = eigenvalues.tolist()
        print(f"    n={size:>6d}: top eigenvalue = {eigenvalues[0]:.2f}, top-10 sum = {eigenvalues[:10].sum():.2f}")

    # Pairwise correlation between eigenvalue spectra
    convergence_scores: dict[str, float] = {}
    sizes = sorted(all_eigenvalues.keys())
    for i in range(1, len(sizes)):
        prev = np.array(all_eigenvalues[sizes[i - 1]])
        curr = np.array(all_eigenvalues[sizes[i]])
        min_len = min(len(prev), len(curr))
        corr = float(np.corrcoef(prev[:min_len], curr[:min_len])[0, 1])
        key = f"{sizes[i-1]}->{sizes[i]}"
        convergence_scores[key] = corr
        print(f"    Eigenvalue correlation {key}: {corr:.6f}")

    return {
        "layer": layer,
        "subset_sizes": subset_sizes,
        "top_eigenvalues": {str(k): v for k, v in all_eigenvalues.items()},
        "convergence_correlations": convergence_scores,
    }


# ── Analysis 5: False-Positive Probe Check (true train/test) ────────────
# Codex fix: Actual train on half with random labels, test on other half. Repeat N times.


def analysis_false_positive_probes(
    baseline_acts: np.ndarray,
    layer: int,
    n_repeats: int = 50,
) -> dict[str, Any]:
    """True 25K/25K train/test: train probe on random labels, test on held-out. Repeat."""
    print(f"\n{'='*60}")
    print(f"Analysis 5: False-Positive Probe Check (L{layer:02d})")
    print(f"{'='*60}")

    n = baseline_acts.shape[0]
    half = n // 2

    # Standardize using first half
    rng = np.random.RandomState(42)
    master_perm = rng.permutation(n)
    X_train_raw = baseline_acts[master_perm[:half]]
    X_test_raw = baseline_acts[master_perm[half:]]

    scaler = StandardScaler().fit(X_train_raw)
    X_train = scaler.transform(X_train_raw)
    X_test = scaler.transform(X_test_raw)

    b5_dims = ["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"]
    results: dict[str, dict[str, Any]] = {}

    for dim in b5_dims:
        test_accs = []

        for rep in range(n_repeats):
            rep_rng = np.random.RandomState(rep)

            # Random H/L labels (drop M to match real probe setup)
            train_labels = rep_rng.choice(["H", "L"], size=len(X_train))
            test_labels = rep_rng.choice(["H", "L"], size=len(X_test))

            y_train = (train_labels == "H").astype(int)
            y_test = (test_labels == "H").astype(int)

            clf = RidgeClassifier(alpha=1.0)
            clf.fit(X_train, y_train)
            acc = float(clf.score(X_test, y_test))
            test_accs.append(acc)

        mean_acc = float(np.mean(test_accs))
        std_acc = float(np.std(test_accs))
        ci_low = float(np.percentile(test_accs, 2.5))
        ci_high = float(np.percentile(test_accs, 97.5))

        results[dim] = {
            "mean_accuracy": mean_acc,
            "std_accuracy": std_acc,
            "ci_95": [ci_low, ci_high],
            "n_repeats": n_repeats,
            "n_train": len(X_train),
            "n_test": len(X_test),
            "chance": 0.5,
            "is_null": bool(ci_low <= 0.5 <= ci_high),
        }

        status = "NULL (ok)" if ci_low <= 0.5 <= ci_high else "WARNING"
        print(f"  {dim:20s} | accuracy: {mean_acc:.3f} +/- {std_acc:.3f}, 95% CI: [{ci_low:.3f}, {ci_high:.3f}] [{status}]")

    return {"layer": layer, "false_positive_probes": results}


# ── Main ─────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="Baseline Analysis Suite")
    parser.add_argument("--baseline-dir", type=str, required=True,
                        help="Path to activations_baseline/ directory")
    parser.add_argument("--sweep-dir", type=str, required=True,
                        help="Path to sweep_output/blackwell/ directory")
    parser.add_argument("--output-dir", type=str, default="results/baseline_analysis",
                        help="Output directory for results JSON")
    parser.add_argument("--layers", type=str, default="9,15,22,29",
                        help="Comma-separated layer indices")
    parser.add_argument("--analyses", type=str, default="1,2,3,4,5",
                        help="Comma-separated analysis numbers to run (1-5)")
    args = parser.parse_args()

    baseline_dir = Path(args.baseline_dir)
    sweep_dir = Path(args.sweep_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    layers = [int(x.strip()) for x in args.layers.split(",")]
    analyses = [int(x.strip()) for x in args.analyses.split(",")]

    all_results: dict[str, Any] = {
        "config": {
            "baseline_dir": str(baseline_dir),
            "sweep_dir": str(sweep_dir),
            "layers": layers,
            "analyses": analyses,
        },
        "results": {},
    }

    for layer in layers:
        print(f"\n{'#'*60}")
        print(f"# Layer {layer}")
        print(f"{'#'*60}")

        layer_key = f"L{layer:02d}"
        all_results["results"][layer_key] = {}

        # Load data (shared across analyses)
        baseline_acts: np.ndarray | None = None
        baseline_meta: list[dict] | None = None
        domains: list[str] | None = None
        sweep_acts: np.ndarray | None = None
        sweep_meta: list[dict] | None = None
        b5_labels: dict[str, list[str]] | None = None

        if any(a in analyses for a in [1, 2, 3]):
            print(f"\n  Loading sweep activations for L{layer:02d}...")
            sweep_acts, sweep_meta, b5_labels = load_sweep_with_b5(sweep_dir, layer)
            print(f"  Sweep: {sweep_acts.shape[0]} samples, {sweep_acts.shape[1]} dims")

        if any(a in analyses for a in [1, 2, 3, 4, 5]):
            print(f"  Loading baseline activations for L{layer:02d}...")
            baseline_acts, baseline_meta, domains = load_baseline_with_domain(baseline_dir, layer)
            print(f"  Baseline: {baseline_acts.shape[0]} samples, {baseline_acts.shape[1]} dims")

        if 1 in analyses and baseline_acts is not None and sweep_acts is not None:
            r = analysis_domain_confound(baseline_acts, domains, sweep_acts, b5_labels, layer)
            all_results["results"][layer_key]["domain_confound"] = r

        if 2 in analyses and baseline_acts is not None and sweep_acts is not None:
            r = analysis_conditional_whitening(baseline_acts, domains, sweep_acts, b5_labels, layer)
            all_results["results"][layer_key]["conditional_whitening"] = r

        if 3 in analyses and baseline_acts is not None and sweep_acts is not None:
            r = analysis_neuron_null(baseline_acts, sweep_acts, b5_labels, layer)
            all_results["results"][layer_key]["neuron_null"] = r

        if 4 in analyses and baseline_acts is not None:
            r = analysis_covariance_convergence(baseline_acts, layer)
            all_results["results"][layer_key]["covariance_convergence"] = r

        if 5 in analyses and baseline_acts is not None:
            r = analysis_false_positive_probes(baseline_acts, layer)
            all_results["results"][layer_key]["false_positive_probes"] = r

    # Save all results
    output_path = output_dir / "baseline_analysis_results.json"
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n{'='*60}")
    print(f"All results saved to: {output_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
