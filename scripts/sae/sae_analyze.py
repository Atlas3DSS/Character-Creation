# sae_analyze.py
#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from tqdm import tqdm

from sae_config import (
    ACTIVATIONS_DIR,
    CONNECTOME_CATEGORIES,
    CONNECTOME_ZSCORES_PATH,
    HUB_NEURONS_PATH,
    SAE_ANALYSIS_DIR,
    SAE_MODELS_DIR,
    TARGET_LAYER_MAP,
)
from sae_train import ActivationDataset, TopKSAE, safe_torch_load


@dataclass
class FeatureStats:
    feature_idx: int
    activation_frequency: float
    mean_activation: float
    max_activation: float
    decoder_norm: float
    top_activating_tokens: list[int]


def load_connectome_zscores(path: Path) -> torch.Tensor:
    obj = safe_torch_load(path, map_location="cpu")
    if isinstance(obj, torch.Tensor):
        z = obj
    elif isinstance(obj, dict):
        keys = ["zscores", "z_scores", "connectome_zscores", "tensor", "data"]
        found = None
        for k in keys:
            if k in obj and isinstance(obj[k], torch.Tensor):
                found = obj[k]
                break
        if found is None:
            # fallback: first tensor value
            tensor_vals = [v for v in obj.values() if isinstance(v, torch.Tensor)]
            if not tensor_vals:
                raise ValueError(f"No tensor found in connectome file: {path}")
            found = tensor_vals[0]
        z = found
    else:
        raise TypeError(f"Unsupported connectome file type: {type(obj)}")

    if z.ndim != 3:
        raise ValueError(f"Expected connectome zscores [C,L,D], got shape {tuple(z.shape)}")
    return z.float()


def load_trained_sae(sae_dir: Path, device: torch.device) -> TopKSAE:
    cfg_path = sae_dir / "training_config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Missing training config: {cfg_path}")
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = json.load(f)

    d_model = int(cfg["d_model"])
    d_sae = int(cfg["d_sae"])
    k = int(cfg["k"])

    sae = TopKSAE(d_model=d_model, d_sae=d_sae, k=k)

    final_path = sae_dir / "sae_final.pt"
    if final_path.exists():
        state = safe_torch_load(final_path, map_location=device)
    else:
        ckpts = sorted(
            sae_dir.glob("checkpoint_step_*.pt"),
            key=lambda p: int(p.stem.split("_")[-1]),
        )
        if not ckpts:
            raise FileNotFoundError(f"No sae_final.pt or checkpoints found in {sae_dir}")
        state_obj = safe_torch_load(ckpts[-1], map_location=device)
        if not isinstance(state_obj, dict) or "model_state" not in state_obj:
            raise ValueError(f"Invalid checkpoint format: {ckpts[-1]}")
        state = state_obj["model_state"]

    if isinstance(state, dict) and "model_state" in state:
        state = state["model_state"]

    sae.load_state_dict(state)
    sae.to(device)
    sae.eval()
    return sae


def compute_feature_stats(
    sae: TopKSAE,
    activations_dir: Path,
    n_top_examples: int = 20,
    batch_size: int = 2048,
    device: torch.device = torch.device("cpu"),
) -> list[FeatureStats]:
    dataset = ActivationDataset(
        activations_dir=activations_dir,
        filter_generation_only=False,
        storage_dtype=torch.float16,
    )
    data = dataset.data
    n_tokens = int(data.shape[0])
    d_sae = sae.d_sae
    k = sae.k

    active_counts = torch.zeros(d_sae, device=device)
    active_sums = torch.zeros(d_sae, device=device)

    top_vals = torch.full((d_sae, n_top_examples), float("-inf"), dtype=torch.float32)
    top_idxs = torch.full((d_sae, n_top_examples), -1, dtype=torch.long)

    with torch.no_grad():
        for start in tqdm(range(0, n_tokens, batch_size), desc="Feature stats", dynamic_ncols=True):
            end = min(start + batch_size, n_tokens)
            batch = data[start:end].to(device=device, dtype=torch.float32)

            pre = F.linear(batch - sae.b_dec, sae.W_enc, sae.b_enc)
            vals, idx = torch.topk(pre, k=k, dim=-1)

            flat_idx = idx.reshape(-1)
            flat_vals = vals.reshape(-1)

            ones = torch.ones_like(flat_vals)
            active_counts.scatter_add_(0, flat_idx, ones)
            active_sums.scatter_add_(0, flat_idx, flat_vals)

            tok_idx = (
                torch.arange(start, end, device=device, dtype=torch.long)
                .unsqueeze(1)
                .expand(-1, k)
                .reshape(-1)
            )

            fi = flat_idx.detach().cpu()
            fv = flat_vals.detach().cpu()
            ft = tok_idx.detach().cpu()

            order = torch.argsort(fi)
            fi = fi[order]
            fv = fv[order]
            ft = ft[order]

            uniq, counts = torch.unique_consecutive(fi, return_counts=True)
            offset = 0
            for u, c in zip(uniq.tolist(), counts.tolist()):
                sl = slice(offset, offset + c)
                cand_vals = torch.cat([top_vals[u], fv[sl]])
                cand_idxs = torch.cat([top_idxs[u], ft[sl]])

                take = min(n_top_examples, cand_vals.numel())
                tk = torch.topk(cand_vals, k=take)
                top_vals[u].fill_(float("-inf"))
                top_idxs[u].fill_(-1)
                top_vals[u, :take] = tk.values
                top_idxs[u, :take] = cand_idxs[tk.indices]
                offset += c

    counts_cpu = active_counts.detach().cpu()
    sums_cpu = active_sums.detach().cpu()
    freq = counts_cpu / float(n_tokens)
    mean_act = sums_cpu / counts_cpu.clamp_min(1.0)
    max_act = torch.where(
        torch.isfinite(top_vals[:, 0]),
        top_vals[:, 0],
        torch.zeros_like(top_vals[:, 0]),
    )
    dec_norm = sae.W_dec.detach().norm(dim=0).cpu()

    stats: list[FeatureStats] = []
    for i in range(d_sae):
        tokens = top_idxs[i][top_idxs[i] >= 0].tolist()
        stats.append(
            FeatureStats(
                feature_idx=i,
                activation_frequency=float(freq[i].item()),
                mean_activation=float(mean_act[i].item()),
                max_activation=float(max_act[i].item()),
                decoder_norm=float(dec_norm[i].item()),
                top_activating_tokens=tokens,
            )
        )
    return stats


def correlate_features_with_connectome(
    sae: TopKSAE,
    connectome_zscores: torch.Tensor,
    layer_idx: int,
    category_names: list[str],
) -> torch.Tensor:
    if layer_idx < 0 or layer_idx >= connectome_zscores.shape[1]:
        raise ValueError(f"Layer index out of range: {layer_idx}")

    dirs = connectome_zscores[:, layer_idx, :].to(sae.W_dec.device)
    if dirs.shape[0] != len(category_names):
        raise ValueError(
            f"Category mismatch: zscores has {dirs.shape[0]} but names has {len(category_names)}"
        )

    dirs = F.normalize(dirs, dim=-1)
    dec_cols = F.normalize(sae.W_dec, dim=0)
    corr = torch.matmul(dec_cols.t(), dirs.t())  # [d_sae, n_categories]
    return corr.detach().cpu()


def analyze_hub_decomposition(
    sae: TopKSAE,
    correlations: torch.Tensor,
    layer_idx: int,
    hub_dims: list[int],
    category_names: list[str],
    feature_stats: list[FeatureStats],
) -> dict[str, Any]:
    report: dict[str, Any] = {"layer": layer_idx, "per_dim": {}}
    d_model = sae.d_model

    for dim in hub_dims:
        if dim < 0 or dim >= d_model:
            continue

        weights = sae.W_dec[dim, :].detach().cpu()
        topk = min(32, weights.numel())
        top_features = torch.topk(weights.abs(), k=topk).indices.tolist()
        top_abs = weights.abs()[top_features]
        denom = float(top_abs.sum().item()) + 1e-8
        top1_share = float(top_abs[0].item()) / denom

        entries: list[dict[str, Any]] = []
        dominant_categories: list[str] = []
        for fi in top_features:
            corr_vec = correlations[fi]
            cat_idx = torch.topk(corr_vec.abs(), k=min(3, corr_vec.shape[0])).indices.tolist()
            cats = [
                {
                    "category": category_names[c],
                    "correlation": float(corr_vec[c].item()),
                }
                for c in cat_idx
            ]
            dominant_categories.append(cats[0]["category"])
            fs = feature_stats[fi]
            entries.append(
                {
                    "feature_idx": fi,
                    "decoder_weight_on_dim": float(weights[fi].item()),
                    "activation_frequency": fs.activation_frequency,
                    "mean_activation": fs.mean_activation,
                    "top_categories": cats,
                }
            )

        unique_dom = len(set(dominant_categories))
        if top1_share > 0.45:
            decomp = "polysemantic"
        elif unique_dom >= 3:
            decomp = "clean"
        else:
            decomp = "distributed"

        report["per_dim"][str(dim)] = {
            "decomposition_type": decomp,
            "top1_weight_share": top1_share,
            "category_separation": float(unique_dom) / float(max(len(entries), 1)),
            "top_features": entries,
        }

    return report


def _extract_hub_dims_for_layer(layer_idx: int) -> list[int]:
    dims: set[int] = set()
    if not HUB_NEURONS_PATH.exists():
        return []

    with HUB_NEURONS_PATH.open("r", encoding="utf-8") as f:
        data = json.load(f)

    def walk(obj: Any, cur_layer: int | None = None) -> None:
        if isinstance(obj, dict):
            layer = cur_layer
            for lk in ("layer", "layer_idx", "layer_id"):
                if lk in obj:
                    try:
                        layer = int(obj[lk])
                    except (TypeError, ValueError):
                        pass

            if layer == layer_idx:
                for dk in ("dim", "dimension", "neuron", "idx", "neuron_idx"):
                    if dk in obj:
                        try:
                            dims.add(int(obj[dk]))
                        except (TypeError, ValueError):
                            pass

            for v in obj.values():
                walk(v, layer)

        elif isinstance(obj, list):
            for x in obj:
                walk(x, cur_layer)

    walk(data)
    return sorted(dims)


def cross_layer_comparison(
    sae_dirs: dict[int, Path],
    connectome_zscores: torch.Tensor,
    category_names: list[str],
    focus_categories: list[str],
    device: torch.device,
) -> dict[str, Any]:
    models: dict[int, TopKSAE] = {}
    correlations: dict[int, torch.Tensor] = {}
    for layer, path in sae_dirs.items():
        models[layer] = load_trained_sae(path, device=device)
        correlations[layer] = correlate_features_with_connectome(
            models[layer], connectome_zscores, layer, category_names
        )

    out: dict[str, Any] = {"focus_categories": focus_categories, "comparisons": {}}
    layers = sorted(models.keys())

    for cat in focus_categories:
        if cat not in CONNECTOME_CATEGORIES:
            continue
        ci = CONNECTOME_CATEGORIES[cat]
        cat_info: dict[str, Any] = {"best_feature_by_layer": {}, "pairwise_similarity": {}}

        best_vecs: dict[int, torch.Tensor] = {}
        for layer in layers:
            corr = correlations[layer]
            feat = int(torch.argmax(corr[:, ci].abs()).item())
            val = float(corr[feat, ci].item())
            vec = F.normalize(models[layer].W_dec[:, feat].detach().cpu(), dim=0)
            best_vecs[layer] = vec
            cat_info["best_feature_by_layer"][str(layer)] = {
                "feature_idx": feat,
                "correlation": val,
            }

        for i, la in enumerate(layers):
            for lb in layers[i + 1 :]:
                sim = float(torch.dot(best_vecs[la], best_vecs[lb]).item())
                cat_info["pairwise_similarity"][f"L{la:02d}-L{lb:02d}"] = sim

        out["comparisons"][cat] = cat_info

    return out


def load_metadata_for_indices(layer_dir: Path, target_indices: set[int]) -> dict[int, dict[str, Any]]:
    if not target_indices:
        return {}
    found: dict[int, dict[str, Any]] = {}

    meta_files = sorted(
        layer_dir.glob("shard_*_meta.jsonl"),
        key=lambda p: int(p.stem.split("_")[1]),
    )
    global_idx = 0
    for meta_path in tqdm(meta_files, desc=f"Metadata scan {layer_dir.name}", disable=len(meta_files) <= 10):
        with meta_path.open("r", encoding="utf-8") as f:
            for line in f:
                if global_idx in target_indices:
                    try:
                        found[global_idx] = json.loads(line)
                    except json.JSONDecodeError:
                        found[global_idx] = {"raw_line": line.strip()}
                    if len(found) == len(target_indices):
                        return found
                global_idx += 1
    return found


def generate_interpretability_data(
    feature_stats: list[FeatureStats],
    metadata_dir: Path,
    n_top_features: int = 100,
    n_examples_per_feature: int = 10,
) -> list[dict[str, Any]]:
    scored: list[tuple[float, FeatureStats]] = []
    for fs in feature_stats:
        if fs.activation_frequency <= 1e-6:
            continue
        score = fs.activation_frequency * max(fs.mean_activation, 0.0)
        scored.append((score, fs))
    scored.sort(key=lambda x: x[0], reverse=True)

    chosen = [fs for _, fs in scored[:n_top_features]]
    needed_idx: set[int] = set()
    for fs in chosen:
        needed_idx.update(fs.top_activating_tokens[:n_examples_per_feature])

    meta_map = load_metadata_for_indices(metadata_dir, needed_idx)
    out: list[dict[str, Any]] = []
    for fs in chosen:
        examples: list[dict[str, Any]] = []
        for tok in fs.top_activating_tokens[:n_examples_per_feature]:
            md = meta_map.get(tok, {})
            examples.append(
                {
                    "global_token_idx": tok,
                    "prompt_idx": md.get("prompt_idx"),
                    "token_position": md.get("token_position"),
                    "is_generation": md.get("is_generation"),
                    "category": md.get("prompt_category") or md.get("category"),
                    "system_tag": md.get("system_prompt_tag") or md.get("system_tag"),
                    "prompt_text": md.get("prompt_text"),
                }
            )

        out.append(
            {
                "feature_idx": fs.feature_idx,
                "activation_frequency": fs.activation_frequency,
                "mean_activation": fs.mean_activation,
                "max_activation": fs.max_activation,
                "examples": examples,
            }
        )
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze trained SAE models.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--layer", type=int, default=None)
    group.add_argument("--all-layers", action="store_true")
    parser.add_argument("--model-tag", type=str, default="base")
    parser.add_argument("--sae-dir", type=str, default=None)
    parser.add_argument("--activations-dir", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--n-top-features", type=int, default=100)
    parser.add_argument("--n-examples-per-feature", type=int, default=10)
    parser.add_argument("--stats-batch-size", type=int, default=2048)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    base_sae = Path(args.sae_dir) if args.sae_dir else (SAE_MODELS_DIR / args.model_tag)
    base_acts = Path(args.activations_dir) if args.activations_dir else (ACTIVATIONS_DIR / args.model_tag)
    base_out = Path(args.output_dir) if args.output_dir else (SAE_ANALYSIS_DIR / args.model_tag)
    base_out.mkdir(parents=True, exist_ok=True)

    zscores = load_connectome_zscores(CONNECTOME_ZSCORES_PATH)
    category_names = [x for x, _ in sorted(CONNECTOME_CATEGORIES.items(), key=lambda kv: kv[1])]

    if args.all_layers:
        candidate_layers = sorted([p.layer_idx for p in TARGET_LAYER_MAP.values()])
    else:
        assert args.layer is not None
        candidate_layers = [args.layer]

    successful_layers: dict[int, Path] = {}

    for layer in candidate_layers:
        sae_dir = base_sae / f"L{layer:02d}"
        acts_dir = base_acts / f"L{layer:02d}"
        out_dir = base_out / f"L{layer:02d}"
        out_dir.mkdir(parents=True, exist_ok=True)

        if not sae_dir.exists():
            print(f"[WARN] Missing SAE directory for layer {layer}: {sae_dir}")
            continue
        if not acts_dir.exists():
            print(f"[WARN] Missing activation directory for layer {layer}: {acts_dir}")
            continue

        print(f"[INFO] Analyzing layer {layer}")
        sae = load_trained_sae(sae_dir, device=device)

        feature_stats = compute_feature_stats(
            sae=sae,
            activations_dir=acts_dir,
            n_top_examples=max(args.n_examples_per_feature, 20),
            batch_size=args.stats_batch_size,
            device=device,
        )
        with (out_dir / "feature_stats.json").open("w", encoding="utf-8") as f:
            json.dump([asdict(x) for x in feature_stats], f, indent=2)

        corr = correlate_features_with_connectome(
            sae=sae,
            connectome_zscores=zscores,
            layer_idx=layer,
            category_names=category_names,
        )
        torch.save(corr, out_dir / "feature_category_correlations.pt")

        hub_dims = set(_extract_hub_dims_for_layer(layer))
        if layer in TARGET_LAYER_MAP:
            hub_dims.update(TARGET_LAYER_MAP[layer].key_dims)

        hub_report = analyze_hub_decomposition(
            sae=sae,
            correlations=corr,
            layer_idx=layer,
            hub_dims=sorted(hub_dims),
            category_names=category_names,
            feature_stats=feature_stats,
        )
        with (out_dir / "hub_decomposition.json").open("w", encoding="utf-8") as f:
            json.dump(hub_report, f, indent=2)

        interp = generate_interpretability_data(
            feature_stats=feature_stats,
            metadata_dir=acts_dir,
            n_top_features=args.n_top_features,
            n_examples_per_feature=args.n_examples_per_feature,
        )
        with (out_dir / "top_activating_examples.json").open("w", encoding="utf-8") as f:
            json.dump(interp, f, indent=2, ensure_ascii=False)

        successful_layers[layer] = sae_dir

    if args.all_layers and successful_layers:
        cross = cross_layer_comparison(
            sae_dirs=successful_layers,
            connectome_zscores=zscores,
            category_names=category_names,
            focus_categories=["Tone: Sarcastic", "Domain: Math", "Identity"],
            device=device,
        )
        with (base_out / "cross_layer_comparison.json").open("w", encoding="utf-8") as f:
            json.dump(cross, f, indent=2)


if __name__ == "__main__":
    main()
