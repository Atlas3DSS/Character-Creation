#!/usr/bin/env python3
"""
Analyze existing 27B connectome z-scores to produce full analysis outputs.

Ports the 8B connectome analysis pipeline (hub neurons, category overlap,
layer importance, neuron clustering, SVD dimensionality, known neuron profiles)
to work on the 27B z-score tensor. CPU-only, no GPU or model needed.

Usage:
    python analyze_connectome_27b.py
    python analyze_connectome_27b.py --input ./qwen35_map/27b/connectome_zscores.pt
    python analyze_connectome_27b.py --threshold 1.5 --min-categories 4
"""

import argparse
import json
from pathlib import Path

import torch

# ─── Category names (must match order in connectome_zscores.pt) ──────────
CATEGORY_NAMES = [
    "Identity",
    "Emotion: Joy",
    "Emotion: Sadness",
    "Emotion: Anger",
    "Emotion: Fear",
    "Tone: Formal",
    "Tone: Sarcastic",
    "Tone: Polite",
    "Domain: Math",
    "Domain: Science",
    "Domain: Code",
    "Domain: History",
    "Reasoning: Analytical",
    "Reasoning: Certainty",
    "Safety: Refusal",
    "Role: Teacher",
    "Role: Therapist",
    "Length: Verbose",
    "Length: Brief",
    "Language: Bilingual",
]

# ─── Known 27B neurons (from connectome_stats.json + MEMORY.md) ──────────
KNOWN_NEURONS_27B = {
    2028: "Super-hub (Code z=6.67, Math z=6.19, Sadness z=5.84 at L50)",
    94: "Identity (z=1.06 at L43, 13x weaker than 8B dim 994)",
    526: "Verbosity/Brief (z=10.07 at L51, strongest single-neuron signal)",
}


def category_overlap_matrix(connectome: torch.Tensor) -> torch.Tensor:
    """20x20 cosine similarity of flattened z-score vectors."""
    n_cats = connectome.shape[0]
    flat = connectome.reshape(n_cats, -1).float()
    norms = flat.norm(dim=1, keepdim=True) + 1e-8
    flat_normed = flat / norms
    overlap = flat_normed @ flat_normed.T
    return overlap


def find_hub_neurons(
    connectome: torch.Tensor,
    threshold: float = 2.0,
    min_categories: int = 5,
) -> list[dict]:
    """Find neurons significant (|z| > threshold) in >= min_categories categories."""
    n_cats, n_layers, hidden_dim = connectome.shape
    hub_neurons: list[dict] = []

    # Vectorized: max |z| per (category, dim) across layers → (n_cats, hidden_dim)
    max_abs_z = connectome.abs().amax(dim=1)  # (n_cats, hidden_dim)
    sig_mask = max_abs_z >= threshold  # (n_cats, hidden_dim)
    cat_counts = sig_mask.sum(dim=0)  # (hidden_dim,)

    hub_dims = (cat_counts >= min_categories).nonzero(as_tuple=True)[0]

    for dim in hub_dims:
        dim_int = int(dim)
        cat_details = []
        for cat_idx in range(n_cats):
            if not sig_mask[cat_idx, dim_int]:
                continue
            z_across_layers = connectome[cat_idx, :, dim_int]
            peak_layer = int(z_across_layers.abs().argmax())
            cat_details.append({
                "category": CATEGORY_NAMES[cat_idx],
                "peak_layer": peak_layer,
                "peak_z": float(z_across_layers[peak_layer]),
                "abs_peak_z": float(z_across_layers[peak_layer].abs()),
                "n_sig_layers": int((z_across_layers.abs() >= threshold).sum()),
            })

        hub_neurons.append({
            "dim": dim_int,
            "n_categories": int(cat_counts[dim_int]),
            "categories": cat_details,
            "is_known": dim_int in KNOWN_NEURONS_27B,
            "known_label": KNOWN_NEURONS_27B.get(dim_int, None),
        })

    hub_neurons.sort(key=lambda x: x["n_categories"], reverse=True)
    return hub_neurons


def find_hub_neurons_percentile(
    connectome: torch.Tensor,
    percentile: float = 99.9,
    min_categories: int = 5,
) -> list[dict]:
    """Find hub neurons using percentile threshold (Gemini fix: fair cross-model comparison).

    Instead of fixed |z|>2.0, uses top percentile of max |z| per neuron per category.
    This accounts for distributional differences between 8B (4096d) and 27B (5120d).
    """
    n_cats, n_layers, hidden_dim = connectome.shape

    # Per-category max |z| across layers for each neuron
    max_abs_z = connectome.abs().amax(dim=1)  # (n_cats, hidden_dim)

    # Compute percentile threshold per category
    thresholds = torch.quantile(max_abs_z.float(), percentile / 100.0, dim=1)  # (n_cats,)

    sig_mask = max_abs_z >= thresholds.unsqueeze(1)  # (n_cats, hidden_dim)
    cat_counts = sig_mask.sum(dim=0)  # (hidden_dim,)

    hub_dims = (cat_counts >= min_categories).nonzero(as_tuple=True)[0]
    hub_neurons: list[dict] = []

    for dim in hub_dims:
        dim_int = int(dim)
        cat_details = []
        for cat_idx in range(n_cats):
            if not sig_mask[cat_idx, dim_int]:
                continue
            z_across_layers = connectome[cat_idx, :, dim_int]
            peak_layer = int(z_across_layers.abs().argmax())
            cat_details.append({
                "category": CATEGORY_NAMES[cat_idx],
                "peak_layer": peak_layer,
                "peak_z": float(z_across_layers[peak_layer]),
                "abs_peak_z": float(z_across_layers[peak_layer].abs()),
                "threshold_used": float(thresholds[cat_idx]),
            })

        hub_neurons.append({
            "dim": dim_int,
            "n_categories": int(cat_counts[dim_int]),
            "categories": cat_details,
            "is_known": dim_int in KNOWN_NEURONS_27B,
            "known_label": KNOWN_NEURONS_27B.get(dim_int, None),
        })

    hub_neurons.sort(key=lambda x: x["n_categories"], reverse=True)
    return hub_neurons


def layer_importance_per_category(connectome: torch.Tensor) -> dict:
    """Per-category layer importance (mean |z| per layer + distribution stats)."""
    result = {}
    for cat_idx, cat_name in enumerate(CATEGORY_NAMES):
        abs_z = connectome[cat_idx].abs()  # (n_layers, hidden_dim)
        layer_imp = [float(abs_z[l].mean()) for l in range(connectome.shape[1])]
        # Per-layer distribution stats (Codex suggestion: don't obscure rare strong neurons)
        layer_stats = []
        for l in range(connectome.shape[1]):
            vals = abs_z[l]
            layer_stats.append({
                "mean": float(vals.mean()),
                "std": float(vals.std()),
                "max": float(vals.max()),
                "p95": float(torch.quantile(vals.float(), 0.95)),
                "p99": float(torch.quantile(vals.float(), 0.99)),
                "n_above_2": int((vals >= 2.0).sum()),
            })
        result[cat_name] = {
            "layer_importance": layer_imp,
            "layer_stats": layer_stats,
            "peak_layer": int(torch.tensor(layer_imp).argmax()),
            "total_importance": sum(layer_imp),
        }
    return result


def neuron_functional_clustering(
    connectome: torch.Tensor,
    n_clusters: int = 10,
    min_significance: float = 1.0,
) -> dict:
    """K-means clustering on neuron response profiles across categories."""
    from sklearn.cluster import KMeans

    n_cats, n_layers, hidden_dim = connectome.shape

    # Vectorized: for each (cat, dim), find the layer with max |z| and take signed value
    # connectome shape: (n_cats, n_layers, hidden_dim)
    abs_connectome = connectome.abs()
    peak_layer_indices = abs_connectome.argmax(dim=1)  # (n_cats, hidden_dim)
    # Gather signed values at peak layers
    profiles = torch.gather(
        connectome.permute(0, 2, 1),  # (n_cats, hidden_dim, n_layers)
        dim=2,
        index=peak_layer_indices.unsqueeze(2),  # (n_cats, hidden_dim, 1)
    ).squeeze(2)  # (n_cats, hidden_dim)
    profiles = profiles.T  # (hidden_dim, n_cats)

    # Filter: only cluster significant neurons
    sig_mask = profiles.abs().max(dim=1).values >= min_significance
    sig_indices = sig_mask.nonzero(as_tuple=True)[0]
    sig_profiles = profiles[sig_indices].numpy()

    print(f"  Clustering {len(sig_indices)} significant neurons into {n_clusters} clusters...")
    if len(sig_indices) < n_clusters:
        return {"error": f"Only {len(sig_indices)} significant neurons, need >= {n_clusters}"}

    # --- Raw K-means ---
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(sig_profiles)

    clusters = {}
    for c in range(n_clusters):
        mask = labels == c
        cluster_dims = sig_indices[mask].tolist()
        centroid = kmeans.cluster_centers_[c]
        top_cats = sorted(range(n_cats), key=lambda i: abs(centroid[i]), reverse=True)[:5]
        clusters[str(c)] = {
            "n_neurons": int(mask.sum()),
            "sample_dims": cluster_dims[:20],
            "dominant_categories": [
                {"category": CATEGORY_NAMES[i], "centroid_z": float(centroid[i])}
                for i in top_cats
            ],
            "centroid": centroid.tolist(),
        }
        cats_str = ", ".join(f"{CATEGORY_NAMES[i]}({centroid[i]:+.2f})" for i in top_cats[:3])
        print(f"  Cluster {c}: {int(mask.sum()):4d} neurons | {cats_str}")

    # --- Variance-normalized K-means (Gemini fix: strip Bilingual/Verbose dominance) ---
    from sklearn.preprocessing import StandardScaler

    print(f"\n  Variance-normalized clustering (stripping Bilingual/Verbose dominance)...")
    scaler = StandardScaler()
    norm_profiles = scaler.fit_transform(sig_profiles)
    kmeans_norm = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels_norm = kmeans_norm.fit_predict(norm_profiles)

    clusters_normalized = {}
    for c in range(n_clusters):
        mask = labels_norm == c
        cluster_dims = sig_indices[mask].tolist()
        # Inverse-transform centroids back to z-score space for interpretability
        centroid_raw = scaler.inverse_transform(kmeans_norm.cluster_centers_[c:c+1])[0]
        top_cats = sorted(range(n_cats), key=lambda i: abs(centroid_raw[i]), reverse=True)[:5]
        clusters_normalized[str(c)] = {
            "n_neurons": int(mask.sum()),
            "sample_dims": cluster_dims[:20],
            "dominant_categories": [
                {"category": CATEGORY_NAMES[i], "centroid_z": float(centroid_raw[i])}
                for i in top_cats
            ],
        }
        cats_str = ", ".join(f"{CATEGORY_NAMES[i]}({centroid_raw[i]:+.2f})" for i in top_cats[:3])
        print(f"  NormCluster {c}: {int(mask.sum()):4d} neurons | {cats_str}")

    return {
        "n_clusters": n_clusters,
        "clusters": clusters,
        "clusters_normalized": clusters_normalized,
        "n_significant_neurons": len(sig_indices),
        "significance_threshold": min_significance,
    }


def category_svd(connectome: torch.Tensor) -> dict:
    """SVD per category — intrinsic dimensionality (k80/k90/k95)."""
    result = {}
    for cat_idx, cat_name in enumerate(CATEGORY_NAMES):
        z_matrix = connectome[cat_idx].float()  # (n_layers, hidden_dim)
        U, S, Vh = torch.linalg.svd(z_matrix, full_matrices=False)
        var_total = (S ** 2).sum()
        max_rank = int(min(z_matrix.shape))
        if var_total > 0:
            var_exp = (S ** 2).cumsum(dim=0) / var_total
        else:
            var_exp = torch.zeros_like(S)

        k80 = min(int((var_exp < 0.80).sum()) + 1, max_rank)
        k90 = min(int((var_exp < 0.90).sum()) + 1, max_rank)
        k95 = min(int((var_exp < 0.95).sum()) + 1, max_rank)

        result[cat_name] = {
            "k80": k80, "k90": k90, "k95": k95,
            "singular_values_top5": S[:5].tolist(),
            "var_explained_top5": var_exp[:5].tolist(),
        }
        print(f"  {cat_name:25s}: k80={k80:2d}, k90={k90:2d}, k95={k95:2d}, S[0]={S[0]:.2f}")

    return result


def known_neuron_profiles(connectome: torch.Tensor, neurons: dict[int, str] | None = None) -> dict:
    """Profile known neurons across all 20 categories."""
    neurons = neurons if neurons is not None else KNOWN_NEURONS_27B
    result = {}
    for dim, label in neurons.items():
        if dim >= connectome.shape[2]:
            print(f"  WARNING: dim {dim} ({label}) out of range (hidden_dim={connectome.shape[2]})")
            continue
        profile = {}
        for cat_idx, cat_name in enumerate(CATEGORY_NAMES):
            z_vals = connectome[cat_idx, :, dim]
            peak_layer = int(z_vals.abs().argmax())
            profile[cat_name] = {
                "peak_z": float(z_vals[peak_layer]),
                "peak_layer": peak_layer,
                "mean_abs_z": float(z_vals.abs().mean()),
                "n_sig_layers": int((z_vals.abs() > 2.0).sum()),
            }
        result[str(dim)] = {
            "label": label,
            "profile": profile,
        }
    return result


def auto_discover_known_neurons(
    connectome: torch.Tensor,
    top_n: int = 20,
    threshold: float = 4.0,
) -> dict[int, str]:
    """Auto-discover the most notable neurons from the connectome itself."""
    n_cats, n_layers, hidden_dim = connectome.shape
    discovered: dict[int, str] = {}

    # Find neurons with highest absolute z-score anywhere
    max_z_per_dim = connectome.abs().amax(dim=(0, 1))  # (hidden_dim,)
    top_dims = torch.topk(max_z_per_dim, top_n).indices

    for dim in top_dims:
        dim_int = int(dim)
        if dim_int in KNOWN_NEURONS_27B:
            continue  # Skip already-known
        # Find which category and layer has the peak
        slice_2d = connectome[:, :, dim_int]  # (n_cats, n_layers)
        cat_idx, layer_idx = torch.unravel_index(slice_2d.abs().argmax(), slice_2d.shape)
        cat_idx, layer_idx = int(cat_idx), int(layer_idx)
        peak_z = float(connectome[cat_idx, layer_idx, dim_int])

        if abs(peak_z) >= threshold:
            # Count how many categories are significant
            sig_cats = []
            for ci in range(n_cats):
                if connectome[ci, :, dim_int].abs().max() >= 2.0:
                    sig_cats.append(CATEGORY_NAMES[ci])

            label = (f"Auto-discovered: peak {CATEGORY_NAMES[cat_idx]} z={peak_z:.2f} "
                     f"at L{layer_idx}, {len(sig_cats)} categories ({', '.join(sig_cats[:3])})")
            discovered[dim_int] = label

    return discovered


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze 27B connectome z-scores")
    parser.add_argument("--input", type=str,
                        default="./qwen35_map/27b/connectome_zscores.pt",
                        help="Path to connectome z-scores tensor")
    parser.add_argument("--output", type=str,
                        default="./qwen35_map/27b",
                        help="Output directory")
    parser.add_argument("--threshold", type=float, default=2.0,
                        help="Z-score threshold for hub neurons")
    parser.add_argument("--min-categories", type=int, default=5,
                        help="Min categories for hub neurons")
    parser.add_argument("--n-clusters", type=int, default=10,
                        help="Number of K-means clusters")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output)

    if not input_path.exists():
        raise FileNotFoundError(f"Connectome z-scores not found: {input_path}")

    print(f"Loading connectome from {input_path}...")
    connectome = torch.load(input_path, map_location="cpu", weights_only=True)
    n_cats, n_layers, hidden_dim = connectome.shape
    print(f"  Shape: {n_cats} categories x {n_layers} layers x {hidden_dim} hidden_dim")

    # NaN/Inf guard
    if torch.isnan(connectome).any() or torch.isinf(connectome).any():
        n_nan = int(torch.isnan(connectome).sum())
        n_inf = int(torch.isinf(connectome).sum())
        print(f"  WARNING: {n_nan} NaN, {n_inf} Inf values detected — replacing with 0")
        connectome = torch.nan_to_num(connectome, nan=0.0, posinf=0.0, neginf=0.0)

    if n_cats != len(CATEGORY_NAMES):
        raise ValueError(f"Expected {len(CATEGORY_NAMES)} categories, got {n_cats}")

    output_dir.mkdir(parents=True, exist_ok=True)

    # ─── 1. Category Overlap Matrix ──────────────────────────────────────
    print("\n[1/6] Computing category overlap matrix...")
    overlap = category_overlap_matrix(connectome)
    overlap_dict = {
        "matrix": overlap.tolist(),
        "categories": CATEGORY_NAMES,
        "description": "20x20 cosine similarity of flattened z-score vectors",
    }
    with open(output_dir / "category_overlap.json", "w") as f:
        json.dump(overlap_dict, f, indent=2)
    print(f"  Saved. Diagonal check: {overlap.diag().mean():.4f} (should be ~1.0)")
    # Print notable pairs
    for i in range(n_cats):
        for j in range(i + 1, n_cats):
            if abs(overlap[i, j]) > 0.5:
                print(f"  HIGH overlap: {CATEGORY_NAMES[i]} <-> {CATEGORY_NAMES[j]}: {overlap[i, j]:.3f}")
            if abs(overlap[i, j]) < 0.05:
                print(f"  ORTHOGONAL: {CATEGORY_NAMES[i]} <-> {CATEGORY_NAMES[j]}: {overlap[i, j]:.3f}")

    # ─── 2. Hub Neurons ──────────────────────────────────────────────────
    print(f"\n[2/6] Finding hub neurons (threshold={args.threshold}, min_categories={args.min_categories})...")
    hubs = find_hub_neurons(connectome, args.threshold, args.min_categories)
    with open(output_dir / "hub_neurons.json", "w") as f:
        json.dump(hubs, f, indent=2)
    print(f"  Found {len(hubs)} hub neurons")
    for h in hubs[:10]:
        known = f" *** {h['known_label']}" if h["is_known"] else ""
        print(f"  dim {h['dim']:5d}: {h['n_categories']:2d} categories{known}")

    # ─── 2b. Percentile Hub Neurons (Gemini fix: fair cross-model comparison) ──
    print(f"\n[2b/6] Percentile hub neurons (top 0.1% per category, min_categories={args.min_categories})...")
    hubs_pct = find_hub_neurons_percentile(connectome, percentile=99.9, min_categories=args.min_categories)
    with open(output_dir / "hub_neurons_percentile.json", "w") as f:
        json.dump(hubs_pct, f, indent=2)
    print(f"  Found {len(hubs_pct)} percentile hub neurons")
    for h in hubs_pct[:10]:
        known = f" *** {h['known_label']}" if h["is_known"] else ""
        print(f"  dim {h['dim']:5d}: {h['n_categories']:2d} categories{known}")

    # ─── 3. Layer Importance ─────────────────────────────────────────────
    print("\n[3/6] Computing layer importance per category...")
    layer_imp = layer_importance_per_category(connectome)
    with open(output_dir / "layer_importance.json", "w") as f:
        json.dump(layer_imp, f, indent=2)
    for cat_name, info in layer_imp.items():
        print(f"  {cat_name:25s}: peak=L{info['peak_layer']:2d}, total_imp={info['total_importance']:.2f}")

    # ─── 4. Neuron Functional Clustering ─────────────────────────────────
    print(f"\n[4/6] Neuron functional clustering (k={args.n_clusters})...")
    clusters = neuron_functional_clustering(connectome, args.n_clusters)
    with open(output_dir / "neuron_clusters.json", "w") as f:
        json.dump(clusters, f, indent=2)

    # ─── 5. Category SVD ─────────────────────────────────────────────────
    print("\n[5/6] Category SVD (intrinsic dimensionality)...")
    svd_results = category_svd(connectome)
    with open(output_dir / "category_svd.json", "w") as f:
        json.dump(svd_results, f, indent=2)

    # ─── 6. Known Neuron Profiles ────────────────────────────────────────
    print("\n[6/6] Known neuron profiles...")
    # Auto-discover additional notable neurons (don't mutate global)
    auto_discovered = auto_discover_known_neurons(connectome)
    all_known = dict(KNOWN_NEURONS_27B)
    if auto_discovered:
        print(f"  Auto-discovered {len(auto_discovered)} additional notable neurons")
        all_known.update(auto_discovered)

    profiles = known_neuron_profiles(connectome, all_known)

    # Redundancy check (Codex fix: flag highly correlated auto-discovered neurons)
    if len(all_known) > 1:
        known_dims = sorted(all_known.keys())
        print(f"  Checking redundancy among {len(known_dims)} profiled neurons...")
        # Build profile vectors: peak z per category for each known neuron
        profile_vecs = torch.zeros(len(known_dims), n_cats)
        for i, dim in enumerate(known_dims):
            for cat_idx in range(n_cats):
                z_vals = connectome[cat_idx, :, dim]
                profile_vecs[i, cat_idx] = z_vals[z_vals.abs().argmax()]
        # Pairwise cosine similarity
        norms = profile_vecs.norm(dim=1, keepdim=True) + 1e-8
        cos_sim = (profile_vecs / norms) @ (profile_vecs / norms).T
        redundant_pairs = []
        for i in range(len(known_dims)):
            for j in range(i + 1, len(known_dims)):
                sim = float(cos_sim[i, j])
                if abs(sim) > 0.9:
                    redundant_pairs.append({
                        "dim_a": known_dims[i], "dim_b": known_dims[j],
                        "cosine": sim,
                    })
                    print(f"    REDUNDANT: dim {known_dims[i]} <-> dim {known_dims[j]}: cos={sim:.3f}")
        if not redundant_pairs:
            print(f"    No redundant pairs (all pairwise |cos| < 0.9)")
        profiles["_redundancy_check"] = {
            "n_checked": len(known_dims),
            "redundant_pairs": redundant_pairs,
        }

    with open(output_dir / "known_neuron_profiles.json", "w") as f:
        json.dump(profiles, f, indent=2)
    print(f"  Profiled {len(profiles) - 1} neurons across {n_cats} categories")

    # ─── Summary ─────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)
    print(f"Connectome: {n_cats} categories x {n_layers} layers x {hidden_dim} dims")
    print(f"Hub neurons: {len(hubs)} (threshold={args.threshold}, min_cats={args.min_categories})")
    print(f"Top hub: dim {hubs[0]['dim']} ({hubs[0]['n_categories']} categories)" if hubs else "No hubs found")
    print(f"Neuron clusters: {clusters.get('n_significant_neurons', 0)} significant neurons in {args.n_clusters} clusters")
    print(f"\nOutputs saved to: {output_dir}")
    for f_name in ["category_overlap.json", "hub_neurons.json", "hub_neurons_percentile.json",
                    "layer_importance.json", "neuron_clusters.json", "category_svd.json",
                    "known_neuron_profiles.json"]:
        p = output_dir / f_name
        if p.exists():
            print(f"  {f_name}: {p.stat().st_size:,} bytes")


if __name__ == "__main__":
    main()
