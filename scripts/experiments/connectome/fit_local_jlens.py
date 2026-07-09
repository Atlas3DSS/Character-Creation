#!/usr/bin/env python3
"""Fit a local Jacobian lens with project manifests and resume checkpoints."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

import jlens  # noqa: E402
from jlens.examples import load_wikitext_prompts  # noqa: E402

from scripts.experiments.jlens_common import (  # noqa: E402
    RunLogger,
    git_snapshot,
    model_cache_report,
    now_iso,
    require_cached_model,
    timestamp,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="HF model id or local path")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--run-name", default="")
    parser.add_argument("--model-class", choices=["causal-lm", "image-text"], default="causal-lm")
    parser.add_argument("--n-prompts", type=int, default=64)
    parser.add_argument("--skip-prompts", type=int, default=0)
    parser.add_argument("--min-chars", type=int, default=600)
    parser.add_argument("--max-seq-len", type=int, default=128)
    parser.add_argument("--source-layers", default="")
    parser.add_argument("--target-layer", type=int, default=None)
    parser.add_argument("--dim-batch", type=int, default=4)
    parser.add_argument("--checkpoint-every", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--compile-blocks", action="store_true")
    parser.add_argument("--seed", type=int, default=20260708)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    return parser.parse_args()


def timestamped_output_dir(run_name: str) -> Path:
    suffix = run_name or "local"
    return PROJECT_ROOT / "sweep_v4" / f"jlens_fit_{suffix}_{timestamp()}"


def parse_layers(raw: str) -> list[int] | None:
    if not raw.strip():
        return None
    return [int(part) for part in raw.split(",") if part.strip()]


def load_hf_model(model_name: str, model_class: str, device: str) -> tuple[Any, Any]:
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")
    torch.manual_seed(0)
    if model_class == "causal-lm":
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            local_files_only=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            local_files_only=True,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
        )
    else:
        from transformers import AutoModelForImageTextToText, AutoProcessor

        processor = AutoProcessor.from_pretrained(
            model_name,
            trust_remote_code=True,
            local_files_only=True,
        )
        tokenizer = processor.tokenizer
        model = AutoModelForImageTextToText.from_pretrained(
            model_name,
            trust_remote_code=True,
            local_files_only=True,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
        )
    if device == "cuda":
        model = model.to("cuda")
    return model, tokenizer


def write_prompt_audit(output_dir: Path, prompts: list[str]) -> None:
    prompt_rows = [
        {"idx": idx, "chars": len(prompt), "preview": prompt[:180].replace("\n", " ")}
        for idx, prompt in enumerate(prompts)
    ]
    write_json(output_dir / "prompt_audit.json", {"prompts": prompt_rows})


def write_report(output_dir: Path, manifest: dict[str, Any]) -> None:
    lines = [
        "# Local Jacobian Lens Fit",
        "",
        "## Provenance",
        "",
        f"- Model: `{manifest['model']}`",
        f"- Model class: `{manifest['model_class']}`",
        f"- Output dir: `{manifest['output_dir']}`",
        f"- Source layers: `{manifest['source_layers']}`",
        f"- Target layer: `{manifest['target_layer']}`",
        f"- Prompts requested: `{manifest['n_prompts_requested']}`",
        f"- Prompts fitted: `{manifest.get('n_prompts_fitted')}`",
        f"- Max seq len: `{manifest['max_seq_len']}`",
        f"- dim_batch: `{manifest['dim_batch']}`",
        "",
        "## Artifacts",
        "",
        f"- Lens: `{manifest['artifacts']['lens']}`",
        f"- Resume checkpoint: `{manifest['artifacts']['checkpoint']}`",
        f"- Event log: `{manifest['artifacts']['events']}`",
        "",
        "This is a local pilot fit. Any downstream claim must report prompt count, selected layers, and the SVD-proxy caveat.",
    ]
    (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or timestamped_output_dir(args.run_name)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = RunLogger(output_dir)
    logger.log("start", argv=sys.argv)
    cache_report = require_cached_model(args.model)
    logger.log("model_cache_checked", cache_report=cache_report)

    loaded_prompts = load_wikitext_prompts(args.n_prompts + args.skip_prompts, min_chars=args.min_chars)
    prompts = loaded_prompts[args.skip_prompts : args.skip_prompts + args.n_prompts]
    if len(prompts) < args.n_prompts:
        raise RuntimeError(f"Only loaded {len(prompts)} prompts after skip, requested {args.n_prompts}")
    write_prompt_audit(output_dir, prompts)
    logger.log("prompts_loaded", n_prompts=len(prompts), min_chars=args.min_chars)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[
            logging.FileHandler(output_dir / "fit.log", encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )

    model, tokenizer = load_hf_model(args.model, args.model_class, args.device)
    wrapped = jlens.from_hf(model, tokenizer, compile=args.compile_blocks)
    source_layers = parse_layers(args.source_layers)
    checkpoint_path = output_dir / "fit_checkpoint.pt"
    lens_path = output_dir / "jacobian_lens.pt"

    manifest = {
        "created_at": now_iso(),
        "script": str(Path(__file__).relative_to(PROJECT_ROOT)),
        "model": args.model,
        "model_class": args.model_class,
        "model_cache_report": model_cache_report(args.model),
        "output_dir": str(output_dir),
        "n_prompts_requested": args.n_prompts,
        "skip_prompts": args.skip_prompts,
        "min_chars": args.min_chars,
        "max_seq_len": args.max_seq_len,
        "source_layers": source_layers,
        "target_layer": args.target_layer,
        "dim_batch": args.dim_batch,
        "checkpoint_every": args.checkpoint_every,
        "resume": args.resume,
        "device": args.device,
        "dtype": "bfloat16",
        "quantized": False,
        "git": git_snapshot(),
        "artifacts": {
            "lens": str(lens_path),
            "checkpoint": str(checkpoint_path),
            "events": str(output_dir / "events.jsonl"),
            "fit_log": str(output_dir / "fit.log"),
            "prompt_audit": str(output_dir / "prompt_audit.json"),
            "report": str(output_dir / "report.md"),
        },
    }
    write_json(output_dir / "manifest.json", manifest)
    logger.log("fit_begin", source_layers=source_layers, target_layer=args.target_layer)
    lens = jlens.fit(
        wrapped,
        prompts=prompts,
        source_layers=source_layers,
        target_layer=args.target_layer,
        dim_batch=args.dim_batch,
        max_seq_len=args.max_seq_len,
        checkpoint_path=str(checkpoint_path),
        checkpoint_every=args.checkpoint_every,
        resume=args.resume,
    )
    lens.save(str(lens_path))
    manifest["finished_at"] = now_iso()
    manifest["n_prompts_fitted"] = lens.n_prompts
    manifest["d_model"] = lens.d_model
    manifest["fitted_source_layers"] = lens.source_layers
    write_json(output_dir / "manifest.json", manifest)
    write_report(output_dir, manifest)
    logger.log("complete", lens=str(lens_path), n_prompts= lens.n_prompts)
    print(f"Wrote {output_dir}")


if __name__ == "__main__":
    main()
