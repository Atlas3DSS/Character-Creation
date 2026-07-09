#!/usr/bin/env python3
"""Offline Qwen3.5 J-lens pilot over archived SCOTUS directions.

This is diagnostic-only. It loads a prefit Jacobian lens plus archived direction
vectors, computes transported-gain/random-control statistics, decodes output
disposition through the local final norm + LM head, and writes compact records.
It does not run model generation.
"""

from __future__ import annotations

import argparse
import json
import math
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
LEGAL_MARKERS = {
    "article",
    "iii",
    "standing",
    "jurisdiction",
    "commerce",
    "authority",
    "limits",
    "private",
    "public",
    "rights",
    "right",
    "case",
    "controversy",
    "injury",
    "remedy",
    "remedies",
    "federal",
    "state",
    "constitutional",
    "congress",
    "court",
}


PILOT_DIRECTIONS = [
    "data/scotus/directions/probe_direction_assistant_all_L08_C0p001.npz",
    "data/scotus/directions/probe_direction_assistant_all_L08_C0p001_inverse_authority.npz",
    "data/scotus/directions/probe_direction_assistant_all_L16_C0p001.npz",
    "data/scotus/directions/probe_direction_assistant_all_L20_C0p001.npz",
    "data/scotus/directions/scotus_article3_controlled_replay_v2_assistant_all_L04_private_rights_20260501.npz",
]


@dataclass(frozen=True)
class DirectionSpec:
    name: str
    path: str
    layer: int
    region: str
    source_run: str | None
    model_path: str | None
    positive_label: str | None
    extraction_method: str
    vector_key: str
    raw_direction_norm: float
    sign_convention: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Defaults to sweep_v4/jlens_qwen35_pilot_<timestamp>.",
    )
    parser.add_argument("--random-controls", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260706)
    parser.add_argument("--svd-ranks", type=str, default="32,128,512")
    parser.add_argument("--svd-oversample", type=int, default=32)
    parser.add_argument("--svd-niter", type=int, default=2)
    parser.add_argument("--top-k-tokens", type=int, default=30)
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default="cpu",
        help="Default is CPU because this workspace torch build may not support Blackwell CUDA kernels.",
    )
    parser.add_argument(
        "--skip-svd",
        action="store_true",
        help="Skip top-singular transport proxy if only gain/readout smoke is needed.",
    )
    return parser.parse_args()


def timestamped_output_dir() -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return PROJECT_ROOT / "sweep_v4" / f"jlens_qwen35_pilot_{stamp}"


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=json_default) + "\n")


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True, default=json_default) + "\n")


def lens_cache_candidates() -> list[Path]:
    cache_root = Path(
        os.environ.get(
            "HF_HOME", str(Path.home() / ".cache" / "huggingface")
        )
    ) / "hub" / "models--neuronpedia--jacobian-lens"
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


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def scalar_from_npz(npz: np.lib.npyio.NpzFile, key: str, default: Any = None) -> Any:
    if key not in npz.files:
        return default
    value = npz[key]
    if value.shape == ():
        return value.item()
    if value.size == 1:
        return value.reshape(-1)[0].item()
    return value.tolist()


def sidecar_path(npz_path: Path) -> Path | None:
    candidate = npz_path.with_suffix(".json")
    return candidate if candidate.exists() else None


def source_model_from_run(source_run: str | None, sidecar: dict[str, Any]) -> str | None:
    if sidecar.get("model_path"):
        return str(sidecar["model_path"])
    if not source_run:
        return None
    run_path = Path(source_run)
    if not run_path.is_absolute():
        run_path = PROJECT_ROOT / run_path
    manifest_path = run_path / "manifest.json"
    if not manifest_path.exists():
        return None
    manifest = load_json(manifest_path)
    model_path = manifest.get("model_path")
    return str(model_path) if model_path else None


def load_direction_spec(npz_path: Path) -> tuple[DirectionSpec, torch.Tensor]:
    npz = np.load(npz_path)
    sidecar = load_json(sidecar_path(npz_path)) if sidecar_path(npz_path) else {}
    if "raw_direction_unit" not in npz.files:
        raise ValueError(f"{npz_path} has no raw_direction_unit vector")
    vector = torch.from_numpy(npz["raw_direction_unit"].astype(np.float32))
    layer = int(sidecar.get("layer", scalar_from_npz(npz, "layer")))
    region = str(sidecar.get("region", scalar_from_npz(npz, "region", "unknown")))
    source_run = sidecar.get("source_run", scalar_from_npz(npz, "source_run", None))
    source_run = str(source_run) if source_run is not None else None
    positive = sidecar.get("positive_justice", scalar_from_npz(npz, "positive_justice", None))
    positive_label = str(positive) if positive is not None else None
    raw_norm = float(sidecar.get("raw_direction_norm", scalar_from_npz(npz, "raw_direction_norm", float(vector.norm()))))
    spec = DirectionSpec(
        name=npz_path.stem,
        path=str(npz_path),
        layer=layer,
        region=region,
        source_run=source_run,
        model_path=source_model_from_run(source_run, sidecar),
        positive_label=positive_label,
        extraction_method="logistic_probe_direction",
        vector_key="raw_direction_unit",
        raw_direction_norm=raw_norm,
        sign_convention="positive direction follows positive_justice/positive_label metadata when present",
    )
    return spec, vector


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
    lm_head = load_safetensor_weight("lm_head.weight").to(device)
    norm_weight = load_safetensor_weight("model.language_model.norm.weight").to(device)
    text_config = load_json(QWEN35_MODEL_PATH / "config.json")["text_config"]
    eps = float(text_config.get("rms_norm_eps", 1e-6))
    return tokenizer, lm_head, norm_weight, eps


def rms_norm(vector: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    variance = vector.float().pow(2).mean()
    return (vector.float() * torch.rsqrt(variance + eps)).to(weight.dtype) * weight


def token_marker_hit(token_text: str) -> str | None:
    cleaned = token_text.strip().lower().replace("Ġ".lower(), "").replace("▁", "")
    for marker in sorted(LEGAL_MARKERS):
        if marker in cleaned:
            return marker
    return None


def top_tokens(
    vector: torch.Tensor,
    tokenizer: Any,
    lm_head: torch.Tensor,
    norm_weight: torch.Tensor,
    eps: float,
    top_k: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    normed = rms_norm(vector, norm_weight, eps)
    logits = torch.mv(lm_head.float(), normed.float())
    values, indices = torch.topk(logits, k=top_k)
    rows: list[dict[str, Any]] = []
    hits: list[dict[str, Any]] = []
    for rank, (token_id, logit) in enumerate(zip(indices.tolist(), values.tolist()), start=1):
        text = tokenizer.decode([token_id])
        row = {"rank": rank, "token_id": int(token_id), "token": text, "logit": float(logit)}
        rows.append(row)
        marker = token_marker_hit(text)
        if marker is not None:
            hit = dict(row)
            hit["marker"] = marker
            hits.append(hit)
    return rows, hits


def percentile(candidate: float, controls: torch.Tensor) -> float:
    values = controls.detach().cpu().float()
    less = (values < candidate).sum().item()
    equal = (values == candidate).sum().item()
    return 100.0 * (less + 0.5 * equal) / max(1, values.numel())


def normalized_rows(rows: torch.Tensor) -> torch.Tensor:
    return rows / rows.norm(dim=1, keepdim=True).clamp_min(1e-12)


def compute_svd_basis(
    j_matrix: torch.Tensor,
    ranks: list[int],
    oversample: int,
    niter: int,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Return approximate right singular vectors for the requested ranks."""
    max_rank = max(ranks)
    q = min(j_matrix.shape[1], max_rank + oversample)
    _, singular_values, v = torch.svd_lowrank(j_matrix.float(), q=q, niter=niter)
    return v, singular_values, q


def compute_svd_proxy(
    direction: torch.Tensor,
    random_directions: torch.Tensor,
    ranks: list[int],
    v: torch.Tensor,
    singular_values: torch.Tensor,
    q: int,
    niter: int,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "method": "torch.svd_lowrank",
        "q": int(q),
        "niter": int(niter),
        "top_singular_values": [float(x) for x in singular_values[: min(10, singular_values.numel())].cpu().tolist()],
        "ranks": {},
    }
    unit = direction.float() / direction.float().norm().clamp_min(1e-12)
    controls = normalized_rows(random_directions.float())
    for rank in ranks:
        basis = v[:, :rank].float()
        candidate_mass = float((unit @ basis).pow(2).sum().item())
        control_mass = (controls @ basis).pow(2).sum(dim=1)
        result["ranks"][str(rank)] = {
            "candidate_mass": candidate_mass,
            "random_mean": float(control_mass.mean().item()),
            "random_std": float(control_mass.std(unbiased=False).item()),
            "percentile": percentile(candidate_mass, control_mass),
        }
    return result


def analyze_direction(
    spec: DirectionSpec,
    direction: torch.Tensor,
    j_matrix: torch.Tensor,
    random_controls: int,
    generator: torch.Generator,
    tokenizer: Any,
    lm_head: torch.Tensor,
    norm_weight: torch.Tensor,
    eps: float,
    top_k_tokens: int,
    svd_ranks: list[int],
    skip_svd: bool,
    svd_basis: tuple[torch.Tensor, torch.Tensor, int] | None,
    svd_niter: int,
) -> dict[str, Any]:
    unit = direction.float() / direction.float().norm().clamp_min(1e-12)
    transported = unit @ j_matrix.float().T
    gain = float(transported.norm().item() / unit.norm().item())
    random_dirs = normalized_rows(
        torch.randn(random_controls, unit.numel(), generator=generator)
    )
    random_transported = random_dirs @ j_matrix.float().T
    random_gains = random_transported.norm(dim=1)
    pos_tokens, pos_hits = top_tokens(
        transported, tokenizer, lm_head, norm_weight, eps, top_k_tokens
    )
    neg_tokens, neg_hits = top_tokens(
        -transported, tokenizer, lm_head, norm_weight, eps, top_k_tokens
    )
    svd_proxy = None
    if not skip_svd:
        if svd_basis is None:
            raise ValueError("svd_basis is required when skip_svd is false")
        v, singular_values, q = svd_basis
        svd_proxy = compute_svd_proxy(
            direction=unit,
            random_directions=random_dirs,
            ranks=svd_ranks,
            v=v,
            singular_values=singular_values,
            q=q,
            niter=svd_niter,
        )
    return {
        "direction": asdict(spec),
        "transported_gain": gain,
        "random_gain": {
            "n": random_controls,
            "mean": float(random_gains.mean().item()),
            "std": float(random_gains.std(unbiased=False).item()),
            "min": float(random_gains.min().item()),
            "max": float(random_gains.max().item()),
            "percentile": percentile(gain, random_gains),
        },
        "positive_readout_top_tokens": pos_tokens,
        "positive_legal_hits": pos_hits,
        "negative_readout_top_tokens": neg_tokens,
        "negative_legal_hits": neg_hits,
        "top_singular_transport_proxy": svd_proxy,
    }


def criteria_pass(record: dict[str, Any]) -> bool:
    gain_ok = record["random_gain"]["percentile"] > 95.0
    proxy = record.get("top_singular_transport_proxy")
    proxy_ok = False
    if proxy:
        proxy_ok = any(
            item["percentile"] > 95.0 for item in proxy["ranks"].values()
        )
    legal_ok = bool(record["positive_legal_hits"] or record["negative_legal_hits"])
    return bool(gain_ok and proxy_ok and legal_ok)


def write_report(output_dir: Path, records: list[dict[str, Any]], manifest: dict[str, Any]) -> None:
    lines = [
        "# Qwen3.5 J-Lens SCOTUS Pilot",
        "",
        "Diagnostic-only offline run. No model generation was performed.",
        "",
        "## Manifest",
        "",
        f"- Lens: `{manifest['lens_path']}`",
        f"- Model readout: `{manifest['model_path']}`",
        f"- Random controls per direction: `{manifest['random_controls']}`",
        f"- SVD ranks: `{manifest['svd_ranks']}`",
        "",
        "## Results",
        "",
        "| Direction | Layer | Gain pct | Best SVD pct | Legal hit rank | Criteria |",
        "|---|---:|---:|---:|---|---|",
    ]
    any_pass = False
    for record in records:
        direction = record["direction"]
        proxy = record.get("top_singular_transport_proxy")
        if proxy:
            best_svd = max(item["percentile"] for item in proxy["ranks"].values())
        else:
            best_svd = float("nan")
        hits = record["positive_legal_hits"] + record["negative_legal_hits"]
        hit_summary = "none"
        if hits:
            first = min(hits, key=lambda item: item["rank"])
            hit_summary = f"{first['marker']}@{first['rank']}"
        passed = criteria_pass(record)
        any_pass = any_pass or passed
        lines.append(
            "| {name} | {layer} | {gain:.1f} | {svd:.1f} | {hit} | {crit} |".format(
                name=direction["name"],
                layer=direction["layer"],
                gain=record["random_gain"]["percentile"],
                svd=best_svd,
                hit=hit_summary,
                crit="pass" if passed else "no-go",
            )
        )
    lines.extend(
        [
            "",
            "## Go/No-Go",
            "",
            (
                "At least one direction met the pilot criteria; this is only a triage signal, not steering evidence."
                if any_pass
                else "No direction met the strict pilot criteria. Treat this as evidence that these archived directions do not have clearly above-random J-transported output disposition under this screen."
            ),
            "",
            "Criteria: gain percentile > 95, at least one top-singular proxy percentile > 95, legal/frame vocabulary in top-30, and interpretable sign semantics. Sign semantics still requires human review of the token tables in `records.jsonl`.",
        ]
    )
    (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or timestamped_output_dir()
    output_dir.mkdir(parents=True, exist_ok=False)
    svd_ranks = [int(part) for part in args.svd_ranks.split(",") if part.strip()]
    if not svd_ranks:
        raise ValueError("--svd-ranks must contain at least one integer")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")

    lens_path = resolve_lens_path()
    lens = load_lens(lens_path)
    source_layers = sorted(int(layer) for layer in lens["J"].keys())
    if source_layers != list(lens.get("source_layers", source_layers)):
        raise ValueError("Lens J keys do not match source_layers metadata")

    tokenizer, lm_head, norm_weight, eps = load_readout(device)
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
        "lens_source_layers": source_layers,
        "model_path": str(QWEN35_MODEL_PATH),
        "readout": "Qwen RMSNorm(model.language_model.norm.weight) + lm_head.weight",
        "layer_convention": "project hooks and jlens both use output of decoder layer i via layers[i] forward hooks; no offset applied",
        "random_controls": int(args.random_controls),
        "seed": int(args.seed),
        "svd_ranks": svd_ranks,
        "svd_oversample": int(args.svd_oversample),
        "svd_niter": int(args.svd_niter),
        "skip_svd": bool(args.skip_svd),
        "top_k_tokens": int(args.top_k_tokens),
        "device": str(device),
        "pilot_directions": PILOT_DIRECTIONS,
        "go_no_go_criteria": {
            "gain_percentile_gt": 95,
            "top_singular_proxy_percentile_gt": 95,
            "legal_frame_vocab_in_top_k": args.top_k_tokens,
            "sign_semantics": "requires human review",
        },
    }
    write_json(output_dir / "manifest.json", manifest)

    inventory_path = output_dir / "direction_inventory.jsonl"
    records_path = output_dir / "records.jsonl"
    records: list[dict[str, Any]] = []
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    svd_cache: dict[int, tuple[torch.Tensor, torch.Tensor, int]] = {}

    for rel_path in tqdm(PILOT_DIRECTIONS, desc="pilot directions"):
        npz_path = PROJECT_ROOT / rel_path
        spec, vector = load_direction_spec(npz_path)
        if spec.model_path and "Qwen3.5-27B" not in spec.model_path:
            raise ValueError(f"{spec.name} source model is not Qwen3.5: {spec.model_path}")
        if spec.layer not in lens["J"]:
            raise ValueError(f"{spec.name} layer {spec.layer} missing from lens")
        append_jsonl(inventory_path, asdict(spec))
        j_matrix = lens["J"][spec.layer].to(device)
        svd_basis = None
        if not args.skip_svd:
            if spec.layer not in svd_cache:
                print(f"Computing randomized SVD basis for layer {spec.layer}")
                svd_cache[spec.layer] = compute_svd_basis(
                    j_matrix=j_matrix,
                    ranks=svd_ranks,
                    oversample=args.svd_oversample,
                    niter=args.svd_niter,
                )
            svd_basis = svd_cache[spec.layer]
        record = analyze_direction(
            spec=spec,
            direction=vector.to(device),
            j_matrix=j_matrix,
            random_controls=args.random_controls,
            generator=generator,
            tokenizer=tokenizer,
            lm_head=lm_head,
            norm_weight=norm_weight,
            eps=eps,
            top_k_tokens=args.top_k_tokens,
            svd_ranks=svd_ranks,
            skip_svd=args.skip_svd,
            svd_basis=svd_basis,
            svd_niter=args.svd_niter,
        )
        append_jsonl(records_path, record)
        records.append(record)
        del j_matrix
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    write_report(output_dir, records, manifest)
    print(f"Wrote {output_dir}")


if __name__ == "__main__":
    main()
