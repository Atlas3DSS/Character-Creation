#!/usr/bin/env python3
"""Offline Qwen3.5 J-lens sift over archived 27B contrastive directions.

This is a no-generation diagnostic. It scans the Qwen3.5/27B connectome
z-score directions, optionally adds sarcasm-minus-math spectral mean
directions, compares transported gain against random controls, and decodes the
largest outliers through the local final norm + LM head.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from tqdm import tqdm
from transformers import AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[3]
QWEN35_MODEL_PATH = Path("/home/orwel/dev_genius/models/Qwen3.5-27B")
LENS_REPO = "neuronpedia/jacobian-lens"
QWEN35_LENS_FILENAME = (
    "qwen3.5-27b/jlens/Salesforce-wikitext/Qwen3.5-27B_jacobian_lens.pt"
)

TOKEN_MARKERS: dict[str, set[str]] = {
    "code": {"code", "python", "function", "class", "def", "import", "algorithm"},
    "math": {"math", "equation", "proof", "theorem", "solve", "number", "integral"},
    "science": {"science", "physics", "quantum", "atom", "energy", "chemical"},
    "history": {"history", "historical", "war", "empire", "ancient"},
    "identity": {"qwen", "assistant", "model"},
    "authority": {"authority", "must", "should", "rule", "command", "order"},
    "teacher": {"teacher", "lesson", "explain", "learn", "student"},
    "refusal": {"sorry", "cannot", "can't", "unable", "illegal", "safety"},
    "positive": {"good", "great", "excellent", "happy", "positive", "love"},
    "formal": {"therefore", "hence", "moreover", "regarding", "accordingly"},
    "polite": {"please", "thank", "thanks", "appreciate", "kindly"},
    "sarcasm": {
        "absurd",
        "arrogant",
        "bizarre",
        "brilliant",
        "disdain",
        "genius",
        "idiot",
        "ridiculous",
        "sarcas",
        "smirk",
    },
    "brief": {"brief", "short", "concise", "summary", "quick"},
    "legal": {
        "article",
        "commerce",
        "court",
        "federal",
        "jurisdiction",
        "private",
        "public",
        "right",
        "rights",
        "state",
    },
}


@dataclass(frozen=True)
class DirectionSpec:
    direction_id: str
    source: str
    category: str
    layer: int
    positive_label: str
    negative_label: str | None
    vector_norm: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Defaults to sweep_v4/jlens_qwen35_contrastive_sift_<timestamp>.",
    )
    parser.add_argument(
        "--connectome",
        type=Path,
        default=PROJECT_ROOT / "qwen35_map/27b/connectome_zscores.pt",
    )
    parser.add_argument(
        "--connectome-stats",
        type=Path,
        default=PROJECT_ROOT / "qwen35_map/27b/connectome_stats.json",
    )
    parser.add_argument(
        "--include-spectral-mean-diff",
        action="store_true",
        help="Also scan sarcasm-minus-math means from qwen35_map/27b/spectral_analysis.",
    )
    parser.add_argument(
        "--spectral-positive",
        type=Path,
        default=PROJECT_ROOT / "qwen35_map/27b/spectral_analysis/sarc_activations.pt",
    )
    parser.add_argument(
        "--spectral-negative",
        type=Path,
        default=PROJECT_ROOT / "qwen35_map/27b/spectral_analysis/math_activations.pt",
    )
    parser.add_argument("--random-controls", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20260707)
    parser.add_argument("--top-k-candidates", type=int, default=30)
    parser.add_argument("--top-k-tokens", type=int, default=20)
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default="cpu",
        help="Default is CPU because this workspace torch build may not support Blackwell CUDA kernels.",
    )
    return parser.parse_args()


def timestamped_output_dir() -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return PROJECT_ROOT / "sweep_v4" / f"jlens_qwen35_contrastive_sift_{stamp}"


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=json_default) + "\n",
        encoding="utf-8",
    )


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True, default=json_default) + "\n")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def lens_cache_candidates() -> list[Path]:
    cache_root = (
        Path(os.environ.get("HF_HOME", str(Path.home() / ".cache" / "huggingface")))
        / "hub"
        / "models--neuronpedia--jacobian-lens"
    )
    if not cache_root.exists():
        return []
    return sorted(cache_root.rglob("Qwen3.5-27B_jacobian_lens.pt"))


def resolve_lens_path() -> Path:
    cached = lens_cache_candidates()
    print(f"Qwen3.5 J-lens cache status: {'hit' if cached else 'miss'}")
    for candidate in cached[:3]:
        print(f"  cached: {candidate} ({candidate.stat().st_size} bytes)")
    if cached:
        return Path(
            hf_hub_download(
                LENS_REPO,
                filename=QWEN35_LENS_FILENAME,
                repo_type="model",
                local_files_only=True,
            )
        )
    print(f"Downloading {LENS_REPO}/{QWEN35_LENS_FILENAME}")
    return Path(
        hf_hub_download(
            LENS_REPO,
            filename=QWEN35_LENS_FILENAME,
            repo_type="model",
        )
    )


def load_lens(path: Path) -> dict[str, Any]:
    obj = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(obj, dict) or "J" not in obj:
        raise ValueError(f"{path} is not a JacobianLens checkpoint")
    return obj


def get_weight_file(weight_name: str) -> Path:
    index_path = QWEN35_MODEL_PATH / "model.safetensors.index.json"
    index = load_json(index_path)
    shard_name = index["weight_map"].get(weight_name)
    if not shard_name:
        raise KeyError(f"{weight_name} not found in {index_path}")
    return QWEN35_MODEL_PATH / shard_name


def load_safetensor_weight(weight_name: str) -> torch.Tensor:
    shard = get_weight_file(weight_name)
    with safe_open(shard, framework="pt", device="cpu") as handle:
        return handle.get_tensor(weight_name)


def load_readout(device: torch.device) -> tuple[Any, torch.Tensor, torch.Tensor, float]:
    tokenizer = AutoTokenizer.from_pretrained(
        QWEN35_MODEL_PATH, trust_remote_code=True, local_files_only=True
    )
    lm_head = load_safetensor_weight("lm_head.weight").float().to(device)
    norm_weight = load_safetensor_weight("model.language_model.norm.weight").float().to(device)
    text_config = load_json(QWEN35_MODEL_PATH / "config.json")["text_config"]
    eps = float(text_config.get("rms_norm_eps", 1e-6))
    return tokenizer, lm_head, norm_weight, eps


def rms_norm(vector: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    variance = vector.float().pow(2).mean()
    return vector.float() * torch.rsqrt(variance + eps) * weight.float()


def clean_token(token_text: str) -> str:
    return token_text.strip().lower().replace("▁", "").replace("ġ", "")


def token_marker_hits(token_text: str) -> list[str]:
    cleaned = clean_token(token_text)
    hits: list[str] = []
    for label, markers in TOKEN_MARKERS.items():
        if any(marker in cleaned for marker in markers):
            hits.append(label)
    return hits


def top_tokens(
    vector: torch.Tensor,
    tokenizer: Any,
    lm_head: torch.Tensor,
    norm_weight: torch.Tensor,
    eps: float,
    top_k: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    normed = rms_norm(vector, norm_weight, eps)
    logits = torch.mv(lm_head, normed.float())
    values, indices = torch.topk(logits, k=top_k)
    rows: list[dict[str, Any]] = []
    hits: list[dict[str, Any]] = []
    for rank, (token_id, logit) in enumerate(zip(indices.tolist(), values.tolist()), start=1):
        text = tokenizer.decode([token_id])
        markers = token_marker_hits(text)
        row = {
            "rank": rank,
            "token_id": int(token_id),
            "token": text,
            "logit": float(logit),
            "markers": markers,
        }
        rows.append(row)
        if markers:
            hits.append(row)
    return rows, hits


def percentile(candidate: float, controls: torch.Tensor) -> float:
    values = controls.detach().cpu().float()
    less = (values < candidate).sum().item()
    equal = (values == candidate).sum().item()
    return 100.0 * (less + 0.5 * equal) / max(1, values.numel())


def normalized_rows(rows: torch.Tensor) -> torch.Tensor:
    return rows.float() / rows.float().norm(dim=1, keepdim=True).clamp_min(1e-12)


def load_connectome_directions(
    connectome_path: Path,
    stats_path: Path,
) -> list[tuple[DirectionSpec, torch.Tensor]]:
    stats = load_json(stats_path)
    categories = [str(item) for item in stats["categories"]]
    zscores = torch.load(connectome_path, map_location="cpu", weights_only=True).float()
    if tuple(zscores.shape[:2]) != (len(categories), int(stats["n_layers"])):
        raise ValueError(
            f"Unexpected connectome shape {tuple(zscores.shape)} for {len(categories)} categories"
        )
    if int(stats["hidden_dim"]) != zscores.shape[2]:
        raise ValueError(
            f"Connectome hidden_dim metadata {stats['hidden_dim']} != tensor dim {zscores.shape[2]}"
        )

    directions: list[tuple[DirectionSpec, torch.Tensor]] = []
    for category_idx, category in enumerate(categories):
        for layer in range(zscores.shape[1]):
            vector = zscores[category_idx, layer].clone()
            direction_id = (
                f"connectome__{category.replace(': ', '_').replace(' ', '_')}__L{layer:02d}"
            )
            spec = DirectionSpec(
                direction_id=direction_id,
                source=str(connectome_path.relative_to(PROJECT_ROOT)),
                category=category,
                layer=layer,
                positive_label=category,
                negative_label="contrastive antipode from connectome prompt pair",
                vector_norm=float(vector.norm().item()),
            )
            directions.append((spec, vector))
    return directions


def load_spectral_mean_directions(
    positive_path: Path,
    negative_path: Path,
) -> list[tuple[DirectionSpec, torch.Tensor]]:
    positive = torch.load(positive_path, map_location="cpu", weights_only=True)
    negative = torch.load(negative_path, map_location="cpu", weights_only=True)
    if not isinstance(positive, dict) or not isinstance(negative, dict):
        raise ValueError("Spectral activation files must be layer->tensor dictionaries")

    directions: list[tuple[DirectionSpec, torch.Tensor]] = []
    for layer in sorted(set(positive.keys()) & set(negative.keys())):
        pos_tensor = positive[layer].float()
        neg_tensor = negative[layer].float()
        if pos_tensor.ndim != 2 or neg_tensor.ndim != 2:
            raise ValueError(f"Layer {layer} spectral activations are not matrices")
        if pos_tensor.shape[1] != neg_tensor.shape[1]:
            raise ValueError(f"Layer {layer} spectral dims differ")
        vector = pos_tensor.mean(dim=0) - neg_tensor.mean(dim=0)
        spec = DirectionSpec(
            direction_id=f"spectral__sarcasm_minus_math__L{int(layer):02d}",
            source=f"{positive_path.relative_to(PROJECT_ROOT)} minus {negative_path.relative_to(PROJECT_ROOT)}",
            category="Spectral: Sarcasm minus Math",
            layer=int(layer),
            positive_label="sarcasm activation mean",
            negative_label="math activation mean",
            vector_norm=float(vector.norm().item()),
        )
        directions.append((spec, vector))
    return directions


def group_by_layer(
    directions: list[tuple[DirectionSpec, torch.Tensor]],
) -> dict[int, list[tuple[DirectionSpec, torch.Tensor]]]:
    grouped: dict[int, list[tuple[DirectionSpec, torch.Tensor]]] = {}
    for spec, vector in directions:
        grouped.setdefault(spec.layer, []).append((spec, vector))
    return grouped


def scan_directions(
    directions: list[tuple[DirectionSpec, torch.Tensor]],
    lens: dict[str, Any],
    random_controls: int,
    generator: torch.Generator,
    device: torch.device,
    records_path: Path,
) -> list[dict[str, Any]]:
    grouped = group_by_layer(directions)
    records: list[dict[str, Any]] = []
    source_layers = {int(layer) for layer in lens["J"].keys()}
    for layer in tqdm(sorted(grouped), desc="scan layers"):
        if layer not in source_layers:
            raise ValueError(f"Layer {layer} is missing from lens")
        layer_items = grouped[layer]
        j_matrix = lens["J"][layer].to(device).float()
        d_model = j_matrix.shape[1]
        random_dirs = normalized_rows(
            torch.randn(random_controls, d_model, generator=generator, device=device)
        )
        random_transported = random_dirs @ j_matrix.T
        random_gains = random_transported.norm(dim=1)
        random_mean = float(random_gains.mean().item())
        random_std = float(random_gains.std(unbiased=False).item())
        candidate_matrix = normalized_rows(
            torch.stack([vector.float() for _, vector in layer_items]).to(device)
        )
        transported = candidate_matrix @ j_matrix.T
        gains = transported.norm(dim=1)
        for row_idx, (spec, _) in enumerate(layer_items):
            gain = float(gains[row_idx].item())
            gain_z = (gain - random_mean) / random_std if random_std > 0 else 0.0
            record = {
                "direction": asdict(spec),
                "transported_gain": gain,
                "gain_z": float(gain_z),
                "random_gain": {
                    "n": random_controls,
                    "mean": random_mean,
                    "std": random_std,
                    "min": float(random_gains.min().item()),
                    "max": float(random_gains.max().item()),
                    "percentile": percentile(gain, random_gains),
                },
            }
            append_jsonl(records_path, record)
            records.append(record)
        del j_matrix, random_dirs, random_transported, candidate_matrix, transported
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return records


def add_readouts(
    records: list[dict[str, Any]],
    directions: dict[str, torch.Tensor],
    lens: dict[str, Any],
    tokenizer: Any,
    lm_head: torch.Tensor,
    norm_weight: torch.Tensor,
    eps: float,
    top_k_tokens: int,
    device: torch.device,
) -> list[dict[str, Any]]:
    enriched: list[dict[str, Any]] = []
    for record in tqdm(records, desc="decode candidates"):
        direction = record["direction"]
        direction_id = str(direction["direction_id"])
        layer = int(direction["layer"])
        unit = directions[direction_id].float().to(device)
        unit = unit / unit.norm().clamp_min(1e-12)
        j_matrix = lens["J"][layer].to(device).float()
        transported = unit @ j_matrix.T
        pos_tokens, pos_hits = top_tokens(
            transported, tokenizer, lm_head, norm_weight, eps, top_k_tokens
        )
        neg_tokens, neg_hits = top_tokens(
            -transported, tokenizer, lm_head, norm_weight, eps, top_k_tokens
        )
        enriched_record = dict(record)
        enriched_record["positive_readout_top_tokens"] = pos_tokens
        enriched_record["positive_marker_hits"] = pos_hits
        enriched_record["negative_readout_top_tokens"] = neg_tokens
        enriched_record["negative_marker_hits"] = neg_hits
        enriched.append(enriched_record)
        del j_matrix, transported
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return enriched


def compact_tokens(record: dict[str, Any], sign: str) -> str:
    key = f"{sign}_readout_top_tokens"
    tokens = record.get(key, [])[:8]
    return ", ".join(str(item["token"]).replace("\n", "\\n") for item in tokens)


def compact_hits(record: dict[str, Any]) -> str:
    hits = record.get("positive_marker_hits", []) + record.get("negative_marker_hits", [])
    if not hits:
        return "none"
    pieces: list[str] = []
    for hit in hits[:5]:
        markers = ",".join(hit["markers"])
        pieces.append(f"{hit['token'].strip()}:{markers}@{hit['rank']}")
    return "; ".join(pieces)


def write_report(
    output_dir: Path,
    manifest: dict[str, Any],
    records: list[dict[str, Any]],
    top_records: list[dict[str, Any]],
) -> None:
    outliers = [
        record
        for record in records
        if record["random_gain"]["percentile"] >= 95.0
    ]
    lines = [
        "# Qwen3.5 J-Lens Contrastive Sift",
        "",
        "Diagnostic-only offline run. No model generation was performed.",
        "",
        "## Manifest",
        "",
        f"- Lens: `{manifest['lens_path']}`",
        f"- Model readout: `{manifest['model_path']}`",
        f"- Directions scanned: `{manifest['directions_scanned']}`",
        f"- Random controls per layer: `{manifest['random_controls']}`",
        f"- Top candidates decoded: `{manifest['top_k_candidates']}`",
        "",
        "## Gain Outliers",
        "",
        f"- Directions at or above the 95th random-control percentile: `{len(outliers)}`",
        "- Random controls are a coarse first-pass sift, not promotion evidence.",
        "",
        "## Top Decoded Candidates",
        "",
        "| Rank | Source | Category | Layer | Gain pct | Gain z | Marker hits | +J top tokens | -J top tokens |",
        "|---:|---|---|---:|---:|---:|---|---|---|",
    ]
    for rank, record in enumerate(top_records, start=1):
        direction = record["direction"]
        lines.append(
            "| {rank} | {source} | {category} | {layer} | {pct:.1f} | {z:.2f} | {hits} | {pos} | {neg} |".format(
                rank=rank,
                source=direction["source"],
                category=direction["category"],
                layer=int(direction["layer"]),
                pct=record["random_gain"]["percentile"],
                z=record["gain_z"],
                hits=compact_hits(record),
                pos=compact_tokens(record, "positive"),
                neg=compact_tokens(record, "negative"),
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "Use this as a triage map: high gain percentile says the direction is unusually visible to the J-lens transport, while the token tables say whether the transported vector has an interpretable output disposition. The next step is causal steering only for candidates with both high gain and coherent sign semantics.",
        ]
    )
    (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.random_controls < 1:
        raise ValueError("--random-controls must be positive")
    output_dir = args.output_dir or timestamped_output_dir()
    output_dir.mkdir(parents=True, exist_ok=False)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")

    directions = load_connectome_directions(args.connectome, args.connectome_stats)
    if args.include_spectral_mean_diff:
        directions.extend(
            load_spectral_mean_directions(args.spectral_positive, args.spectral_negative)
        )
    lens_path = resolve_lens_path()
    lens = load_lens(lens_path)
    lens_source_layers = {int(layer) for layer in lens["J"].keys()}
    skipped_missing_lens = [
        spec for spec, _ in directions if int(spec.layer) not in lens_source_layers
    ]
    if skipped_missing_lens:
        skipped_layers = sorted({int(spec.layer) for spec in skipped_missing_lens})
        print(
            "Skipping "
            f"{len(skipped_missing_lens)} directions from lens-missing layers: {skipped_layers}"
        )
        directions = [
            (spec, vector)
            for spec, vector in directions
            if int(spec.layer) in lens_source_layers
        ]
    vector_by_id = {spec.direction_id: vector for spec, vector in directions}
    if int(lens["d_model"]) != next(iter(vector_by_id.values())).numel():
        raise ValueError(
            f"Lens d_model {lens['d_model']} does not match direction dim "
            f"{next(iter(vector_by_id.values())).numel()}"
        )

    manifest = {
        "created_at": datetime.now().isoformat(),
        "script": str(Path(__file__).relative_to(PROJECT_ROOT)),
        "diagnostic_only": True,
        "generation_performed": False,
        "lens_repo": LENS_REPO,
        "lens_filename": QWEN35_LENS_FILENAME,
        "lens_path": str(lens_path),
        "lens_n_prompts": int(lens["n_prompts"]),
        "lens_d_model": int(lens["d_model"]),
        "lens_source_layers": sorted(lens_source_layers),
        "model_path": str(QWEN35_MODEL_PATH),
        "readout": "Qwen RMSNorm(model.language_model.norm.weight) + lm_head.weight",
        "connectome": str(args.connectome),
        "include_spectral_mean_diff": bool(args.include_spectral_mean_diff),
        "directions_requested": len(directions) + len(skipped_missing_lens),
        "directions_scanned": len(directions),
        "skipped_missing_lens_count": len(skipped_missing_lens),
        "skipped_missing_lens_layers": sorted(
            {int(spec.layer) for spec in skipped_missing_lens}
        ),
        "random_controls": int(args.random_controls),
        "seed": int(args.seed),
        "top_k_candidates": int(args.top_k_candidates),
        "top_k_tokens": int(args.top_k_tokens),
        "device": str(device),
    }
    write_json(output_dir / "manifest.json", manifest)

    records_path = output_dir / "records.jsonl"
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    records = scan_directions(
        directions=directions,
        lens=lens,
        random_controls=args.random_controls,
        generator=generator,
        device=device,
        records_path=records_path,
    )
    ranked = sorted(
        records,
        key=lambda item: (
            float(item["random_gain"]["percentile"]),
            float(item["gain_z"]),
            float(item["transported_gain"]),
        ),
        reverse=True,
    )
    top_scan_records = ranked[: args.top_k_candidates]

    tokenizer, lm_head, norm_weight, eps = load_readout(device)
    top_records = add_readouts(
        records=top_scan_records,
        directions=vector_by_id,
        lens=lens,
        tokenizer=tokenizer,
        lm_head=lm_head,
        norm_weight=norm_weight,
        eps=eps,
        top_k_tokens=args.top_k_tokens,
        device=device,
    )
    write_json(output_dir / "top_candidates.json", {"records": top_records})
    write_report(output_dir, manifest, records, top_records)
    print(f"Wrote {output_dir}")


if __name__ == "__main__":
    main()
