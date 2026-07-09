#!/usr/bin/env python3
"""Delta-J transport-map comparison.

Real mode compares pre-fitted Jacobian lens files and refuses to interpret a
pair without a same-model refit-noise floor. Synthetic smoke mode exercises the
same metrics and artifact schema using small deterministic matrices.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from safetensors import safe_open
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.experiments.jlens_common import (  # noqa: E402
    RunLogger,
    append_jsonl,
    git_snapshot,
    lens_layers,
    linear_cka,
    load_lens,
    markdown_table,
    normalized_frobenius_distance,
    now_iso,
    principal_angle_summary,
    projection_metric_distance,
    timestamp,
    top_singular_basis,
    read_json,
    write_json,
)


DEFAULT_BEHAVIOR_TERMS = [
    "sorry",
    "cannot",
    "unable",
    "refuse",
    "policy",
    "safety",
]
DEFAULT_NEUTRAL_TERMS = [
    "table",
    "weather",
    "paper",
    "window",
    "garden",
    "coffee",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--synthetic-smoke", action="store_true")
    parser.add_argument("--allow-real-comparison", action="store_true")
    parser.add_argument("--noise-floor-a", type=Path, default=None, help="First same-model refit lens")
    parser.add_argument("--noise-floor-b", type=Path, default=None, help="Second same-model refit lens")
    parser.add_argument("--model-a-lens", type=Path, default=None)
    parser.add_argument("--model-b-lens", type=Path, default=None)
    parser.add_argument("--model-a", default=None, help="HF id or local path for model A lm_head; enables real vocab drift")
    parser.add_argument("--model-b", default=None, help="HF id or local path for model B lm_head; defaults to model A")
    parser.add_argument("--tokenizer-model", default=None, help="HF id or local path for tokenizing vocab drift terms")
    parser.add_argument("--pair-label", default="unspecified_pair")
    parser.add_argument("--layers", default="")
    parser.add_argument("--k-values", default="2,4,8")
    parser.add_argument("--synthetic-hidden-dim", type=int, default=24)
    parser.add_argument("--synthetic-layers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260708)
    return parser.parse_args()


def timestamped_output_dir() -> Path:
    return PROJECT_ROOT / "sweep_v4" / f"jlens_delta_comparison_{timestamp()}"


def parse_ints(raw: str) -> list[int]:
    return [int(part) for part in raw.split(",") if part.strip()]


def synthetic_lens_set(hidden_dim: int, n_layers: int, seed: int) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    noise_a: dict[str, Any] = {"J": {}, "source_layers": list(range(n_layers)), "synthetic": True}
    noise_b: dict[str, Any] = {"J": {}, "source_layers": list(range(n_layers)), "synthetic": True}
    model_b: dict[str, Any] = {"J": {}, "source_layers": list(range(n_layers)), "synthetic": True}
    for layer in range(n_layers):
        base = torch.randn(hidden_dim, hidden_dim, generator=generator) / hidden_dim**0.5
        refit_noise = 0.01 * torch.randn(hidden_dim, hidden_dim, generator=generator)
        drift = torch.zeros(hidden_dim, hidden_dim)
        drift[:4, :4] = (0.08 + 0.02 * layer) * torch.randn(4, 4, generator=generator)
        noise_a["J"][layer] = base.float()
        noise_b["J"][layer] = (base + refit_noise).float()
        model_b["J"][layer] = (base + drift + 0.01 * torch.randn(hidden_dim, hidden_dim, generator=generator)).float()
    return noise_a, noise_b, model_b


def metric_bundle(j_a: torch.Tensor, j_b: torch.Tensor, k_values: list[int]) -> dict[str, Any]:
    frob = normalized_frobenius_distance(j_a, j_b)
    cka = linear_cka(j_a, j_b)
    result: dict[str, Any] = {
        "normalized_frobenius_distance": frob,
        "linear_cka": cka,
        "linear_cka_distance": 1.0 - cka if np.isfinite(cka) else float("nan"),
        "subspace": {},
    }
    for k in k_values:
        rank = min(k, int(j_a.shape[1]), int(j_b.shape[1]))
        basis_a = top_singular_basis(j_a, rank)
        basis_b = top_singular_basis(j_b, rank)
        result["subspace"][str(rank)] = {
            "projection_metric_distance": projection_metric_distance(basis_a, basis_b),
            "principal_angles": principal_angle_summary(basis_a, basis_b),
        }
    return result


def with_noise_multiples(pair: dict[str, Any], noise: dict[str, Any]) -> dict[str, Any]:
    enriched = dict(pair)

    def ratio(value: float, floor: float) -> float:
        if not np.isfinite(value) or not np.isfinite(floor) or floor <= 1e-12:
            return float("inf")
        return float(value / floor)

    enriched["noise_floor_multiples"] = {
        "normalized_frobenius_distance": ratio(
            pair["normalized_frobenius_distance"],
            noise["normalized_frobenius_distance"],
        ),
        "linear_cka_distance": ratio(pair["linear_cka_distance"], noise["linear_cka_distance"]),
        "subspace": {},
    }
    for k, row in pair["subspace"].items():
        floor_row = noise["subspace"][k]
        enriched["noise_floor_multiples"]["subspace"][k] = {
            "projection_metric_distance": ratio(
                row["projection_metric_distance"],
                floor_row["projection_metric_distance"],
            ),
            "mean_principal_angle": ratio(
                row["principal_angles"]["mean_deg"],
                floor_row["principal_angles"]["mean_deg"],
            ),
        }
    return enriched


def synthetic_vocab_drift(
    j_a: torch.Tensor,
    j_b: torch.Tensor,
    layer: int,
    seed: int,
) -> list[dict[str, Any]]:
    generator = torch.Generator(device="cpu").manual_seed(seed + layer)
    rows: list[dict[str, Any]] = []
    for category, terms, scale in [
        ("behavior_relevant", DEFAULT_BEHAVIOR_TERMS, 1.0),
        ("neutral", DEFAULT_NEUTRAL_TERMS, 0.25),
    ]:
        for term in terms:
            readout = torch.randn(j_a.shape[0], generator=generator) * scale
            vec_a = j_a.float().T @ readout.float()
            vec_b = j_b.float().T @ readout.float()
            cosine = torch.nn.functional.cosine_similarity(vec_a, vec_b, dim=0)
            rows.append(
                {
                    "term": term,
                    "category": category,
                    "layer": layer,
                    "cosine": float(cosine.item()),
                    "drift": float((1.0 - cosine).item()),
                }
            )
    return rows


def resolve_model_path(model_ref: str) -> Path:
    path = Path(model_ref).expanduser()
    if path.exists():
        return path
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(model_ref, local_files_only=True))


def find_weight_name(model_path: Path, candidates: list[str]) -> str:
    index = read_json(model_path / "model.safetensors.index.json")
    weight_map = index["weight_map"]
    for candidate in candidates:
        if candidate in weight_map:
            return candidate
    raise KeyError(f"None of {candidates} are present in {model_path}")


def get_weight_file(model_path: Path, weight_name: str) -> Path:
    index = read_json(model_path / "model.safetensors.index.json")
    shard = index["weight_map"].get(weight_name)
    if not shard:
        raise KeyError(f"{weight_name} not present in {model_path}")
    return model_path / shard


def load_safetensor_weight(model_path: Path, weight_name: str) -> torch.Tensor:
    with safe_open(get_weight_file(model_path, weight_name), framework="pt", device="cpu") as handle:
        return handle.get_tensor(weight_name)


def load_unembedding(model_ref: str) -> tuple[Path, torch.Tensor]:
    model_path = resolve_model_path(model_ref)
    weight_name = find_weight_name(
        model_path,
        ["lm_head.weight", "model.embed_tokens.weight", "model.language_model.embed_tokens.weight"],
    )
    return model_path, load_safetensor_weight(model_path, weight_name).float()


def term_token_ids(tokenizer: Any, term: str) -> list[int]:
    ids = tokenizer.encode(term, add_special_tokens=False)
    if ids:
        return [int(token_id) for token_id in ids]
    ids = tokenizer.encode(" " + term, add_special_tokens=False)
    return [int(token_id) for token_id in ids]


def real_vocab_drift(
    j_a: torch.Tensor,
    j_b: torch.Tensor,
    layer: int,
    tokenizer: Any,
    unembed_a: torch.Tensor,
    unembed_b: torch.Tensor,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for category, terms in [
        ("behavior_relevant", DEFAULT_BEHAVIOR_TERMS),
        ("neutral", DEFAULT_NEUTRAL_TERMS),
    ]:
        for term in terms:
            ids_a = [idx for idx in term_token_ids(tokenizer, term) if idx < int(unembed_a.shape[0])]
            ids_b = [idx for idx in term_token_ids(tokenizer, term) if idx < int(unembed_b.shape[0])]
            if not ids_a or not ids_b:
                continue
            readout_a = unembed_a[ids_a].mean(dim=0)
            readout_b = unembed_b[ids_b].mean(dim=0)
            vec_a = j_a.float().T @ readout_a.float()
            vec_b = j_b.float().T @ readout_b.float()
            cosine = torch.nn.functional.cosine_similarity(vec_a, vec_b, dim=0)
            rows.append(
                {
                    "term": term,
                    "token_ids_a": ids_a,
                    "token_ids_b": ids_b,
                    "category": category,
                    "layer": layer,
                    "cosine": float(cosine.item()),
                    "drift": float((1.0 - cosine).item()),
                    "drift_mode": "unembedding_transported_by_JT",
                }
            )
    return rows


def compare_lenses(
    output_dir: Path,
    noise_a: dict[str, Any],
    noise_b: dict[str, Any],
    model_b: dict[str, Any],
    layers: list[int],
    k_values: list[int],
    seed: int,
    logger: RunLogger,
    tokenizer: Any | None = None,
    unembed_a: torch.Tensor | None = None,
    unembed_b: torch.Tensor | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    records_path = output_dir / "records.jsonl"
    vocab_path = output_dir / "vocab_drift.jsonl"
    for path in (records_path, vocab_path):
        if path.exists():
            path.unlink()
    records: list[dict[str, Any]] = []
    vocab_rows: list[dict[str, Any]] = []
    for layer in tqdm(layers, desc="delta layers"):
        j_noise_a = noise_a["J"][layer].float()
        j_noise_b = noise_b["J"][layer].float()
        j_model_b = model_b["J"][layer].float()
        noise_metrics = metric_bundle(j_noise_a, j_noise_b, k_values)
        pair_metrics = metric_bundle(j_noise_a, j_model_b, k_values)
        enriched = with_noise_multiples(pair_metrics, noise_metrics)
        record = {
            "record_type": "delta_layer",
            "layer": layer,
            "noise_floor": noise_metrics,
            "pair_delta": enriched,
            "interpretation": (
                "above_noise_floor"
                if enriched["noise_floor_multiples"]["normalized_frobenius_distance"] > 2.0
                else "indistinguishable_or_small_vs_noise"
            ),
        }
        append_jsonl(records_path, record)
        records.append(record)
        if tokenizer is not None and unembed_a is not None and unembed_b is not None:
            layer_vocab_rows = real_vocab_drift(
                j_noise_a,
                j_model_b,
                layer,
                tokenizer=tokenizer,
                unembed_a=unembed_a,
                unembed_b=unembed_b,
            )
        else:
            layer_vocab_rows = synthetic_vocab_drift(j_noise_a, j_model_b, layer, seed)
            for vocab_row in layer_vocab_rows:
                vocab_row["drift_mode"] = "synthetic_random_readout_proxy"
        for vocab_row in layer_vocab_rows:
            append_jsonl(vocab_path, vocab_row)
            vocab_rows.append(vocab_row)
    logger.log("delta_comparison_complete", layers=len(layers), vocab_rows=len(vocab_rows))
    return records, vocab_rows


def summarize_vocab(vocab_rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for category in sorted({str(row["category"]) for row in vocab_rows}):
        values = [float(row["drift"]) for row in vocab_rows if row["category"] == category]
        summary[category] = {
            "n": len(values),
            "mean_drift": float(np.mean(values)) if values else float("nan"),
            "std_drift": float(np.std(values)) if values else float("nan"),
        }
    return summary


def write_report(
    output_dir: Path,
    manifest: dict[str, Any],
    records: list[dict[str, Any]],
    vocab_summary: dict[str, Any],
) -> None:
    rows: list[list[Any]] = []
    for record in records:
        multiples = record["pair_delta"]["noise_floor_multiples"]
        best_subspace = max(
            item["projection_metric_distance"] for item in multiples["subspace"].values()
        )
        rows.append(
            [
                record["layer"],
                f"{record['pair_delta']['normalized_frobenius_distance']:.4f}",
                f"{multiples['normalized_frobenius_distance']:.2f}x",
                f"{record['pair_delta']['linear_cka_distance']:.4f}",
                f"{multiples['linear_cka_distance']:.2f}x",
                f"{best_subspace:.2f}x",
                record["interpretation"],
            ]
        )
    vocab_rows = [
        [category, values["n"], f"{values['mean_drift']:.4f}", f"{values['std_drift']:.4f}"]
        for category, values in vocab_summary.items()
    ]
    lines = [
        "# Delta-J Lens Comparison",
        "",
        f"Mode: `{manifest['mode']}`.",
        "",
        "## Provenance",
        "",
        f"- Script: `{manifest['script']}`",
        f"- Output dir: `{manifest['output_dir']}`",
        f"- Pair label: `{manifest['pair_label']}`",
        f"- Noise floor A: `{manifest['noise_floor_a']}`",
        f"- Noise floor B: `{manifest['noise_floor_b']}`",
        f"- Model B lens: `{manifest['model_b_lens']}`",
        f"- k values: `{manifest['k_values']}`",
        "",
        "## Layer Delta",
        "",
        markdown_table(
            ["Layer", "Frob dist", "Frob/noise", "CKA dist", "CKA/noise", "Best subspace/noise", "Interpretation"],
            rows,
        ),
        "",
        "## Vocab-Resolved Drift",
        "",
        markdown_table(["Category", "n", "Mean drift", "Std drift"], vocab_rows),
        "",
        "## Claim Status",
        "",
    ]
    if manifest["mode"] == "synthetic_smoke":
        lines.append(
            "Synthetic smoke only. The refit-noise control, per-layer metrics, noise multiples, and vocab drift reporting executed; no model comparison finding is claimed."
        )
    else:
        lines.append(
            "Real lens comparison. Interpret every delta only relative to the same-model refit-noise floor above."
        )
    (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    if not args.synthetic_smoke and not args.allow_real_comparison:
        raise SystemExit("Use --synthetic-smoke or explicitly pass --allow-real-comparison.")
    output_dir = args.output_dir or timestamped_output_dir()
    output_dir.mkdir(parents=True, exist_ok=False)
    logger = RunLogger(output_dir)
    logger.log("start", argv=sys.argv)

    k_values = parse_ints(args.k_values)
    if not k_values:
        raise ValueError("--k-values must not be empty")

    if args.synthetic_smoke:
        noise_a, noise_b, model_b = synthetic_lens_set(
            hidden_dim=args.synthetic_hidden_dim,
            n_layers=args.synthetic_layers,
            seed=args.seed,
        )
        layers = parse_ints(args.layers) if args.layers else lens_layers(noise_a)
        torch.save(noise_a, output_dir / "synthetic_noise_floor_a.pt")
        torch.save(noise_b, output_dir / "synthetic_noise_floor_b.pt")
        torch.save(model_b, output_dir / "synthetic_model_b.pt")
        mode = "synthetic_smoke"
        noise_a_path = str(output_dir / "synthetic_noise_floor_a.pt")
        noise_b_path = str(output_dir / "synthetic_noise_floor_b.pt")
        model_b_path = str(output_dir / "synthetic_model_b.pt")
    else:
        if args.noise_floor_a is None or args.noise_floor_b is None:
            raise FileNotFoundError("Real comparison requires --noise-floor-a and --noise-floor-b.")
        if args.model_a_lens is None or args.model_b_lens is None:
            raise FileNotFoundError("Real comparison requires --model-a-lens and --model-b-lens.")
        noise_a = load_lens(args.noise_floor_a)
        noise_b = load_lens(args.noise_floor_b)
        model_a = load_lens(args.model_a_lens)
        model_b = load_lens(args.model_b_lens)
        if lens_layers(noise_a) != lens_layers(noise_b):
            raise ValueError("Noise-floor lens layer sets differ")
        if lens_layers(model_a) != lens_layers(model_b):
            raise ValueError("Model-pair lens layer sets differ")
        # The pair baseline is model A; require it to be the same fit as noise A
        # or a deliberate duplicate path so all multiples are anchored.
        noise_a = model_a
        layers = parse_ints(args.layers) if args.layers else lens_layers(model_a)
        mode = "real_lens_comparison"
        noise_a_path = str(args.noise_floor_a)
        noise_b_path = str(args.noise_floor_b)
        model_b_path = str(args.model_b_lens)

    tokenizer = None
    unembed_a = None
    unembed_b = None
    vocab_drift_assets: dict[str, Any] = {"mode": "synthetic_random_readout_proxy"}
    if not args.synthetic_smoke and args.model_a:
        from transformers import AutoTokenizer

        tokenizer_ref = args.tokenizer_model or args.model_a
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_ref,
            trust_remote_code=True,
            local_files_only=True,
        )
        model_a_path, unembed_a = load_unembedding(args.model_a)
        model_b_path_for_head, unembed_b = load_unembedding(args.model_b or args.model_a)
        vocab_drift_assets = {
            "mode": "unembedding_transported_by_JT",
            "tokenizer_model": tokenizer_ref,
            "model_a": args.model_a,
            "model_b": args.model_b or args.model_a,
            "model_a_path": str(model_a_path),
            "model_b_path": str(model_b_path_for_head),
        }

    records, vocab_rows = compare_lenses(
        output_dir=output_dir,
        noise_a=noise_a,
        noise_b=noise_b,
        model_b=model_b,
        layers=layers,
        k_values=k_values,
        seed=args.seed,
        logger=logger,
        tokenizer=tokenizer,
        unembed_a=unembed_a,
        unembed_b=unembed_b,
    )
    vocab_summary = summarize_vocab(vocab_rows)
    write_json(output_dir / "vocab_summary.json", vocab_summary)
    manifest = {
        "created_at": now_iso(),
        "script": str(Path(__file__).relative_to(PROJECT_ROOT)),
        "mode": mode,
        "output_dir": str(output_dir),
        "pair_label": args.pair_label,
        "layers": layers,
        "k_values": k_values,
        "noise_floor_a": noise_a_path,
        "noise_floor_b": noise_b_path,
        "model_a_lens": str(args.model_a_lens) if args.model_a_lens else noise_a_path,
        "model_b_lens": model_b_path,
        "refit_noise_floor_required": True,
        "vocab_drift_assets": vocab_drift_assets,
        "claims_allowed": mode != "synthetic_smoke",
        "git": git_snapshot(),
        "artifacts": {
            "records": str(output_dir / "records.jsonl"),
            "vocab_drift": str(output_dir / "vocab_drift.jsonl"),
            "vocab_summary": str(output_dir / "vocab_summary.json"),
            "events": str(output_dir / "events.jsonl"),
            "report": str(output_dir / "report.md"),
        },
    }
    write_json(output_dir / "manifest.json", manifest)
    write_report(output_dir, manifest, records, vocab_summary)
    manifest["finished_at"] = now_iso()
    write_json(output_dir / "manifest.json", manifest)
    logger.log("complete", artifacts=manifest["artifacts"])
    print(f"Wrote {output_dir}")


if __name__ == "__main__":
    main()
