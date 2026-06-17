#!/usr/bin/env python3
"""Causal-trace Article III actual answer continuations.

This is a diagnostic follow-up to the fixed holding-phrase trace. It patches
generated-thought hidden states into public-leaning target prompts, then scores
the logprob margin of actual private-conditioned versus public-conditioned
answer continuations for the same target prompt.

It is not a no-mask generation result and cannot promote an actuator by itself.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from localize_article3_ambiguous_thought_states import fmt, markdown_table, now_iso, write_json, write_jsonl  # noqa: E402
from localize_article3_generated_thought_baselines import source_budget_meta  # noqa: E402
from poke_scotus_sae_layers import load_model_and_tokenizer  # noqa: E402
from trace_article3_holding_logit_patches import (  # noqa: E402
    DEFAULT_LOCALIZATION_RUN,
    DEFAULT_MODEL,
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_SOURCE_GENERATIONS,
    TraceSite,
    capture_site,
    load_top_sites,
    make_payload,
    parse_float_list,
    parse_int_list,
    read_jsonl,
    render_trace,
    score_label,
    transformer_layers,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_REFERENCE_GENERATIONS = (
    PROJECT_ROOT / "sweep_v4" / "scotus_counterfactual_thoughts_server_20260502_001228" / "generations.jsonl"
)


def reference_answers(
    rows: list[dict[str, Any]],
    *,
    prompt_ids: list[int],
    private_condition: str,
    public_condition: str,
) -> dict[int, dict[str, str]]:
    wanted = {(prompt_id, private_condition) for prompt_id in prompt_ids} | {
        (prompt_id, public_condition) for prompt_id in prompt_ids
    }
    out: dict[int, dict[str, str]] = defaultdict(dict)
    for row in rows:
        key = (int(row.get("prompt_id", -1)), str(row.get("condition")))
        if key not in wanted:
            continue
        answer = str(row.get("answer") or "").strip()
        if not answer:
            raise ValueError(f"Empty reference answer for prompt={key[0]} condition={key[1]}")
        if key[1] == private_condition:
            out[key[0]]["private"] = answer
        elif key[1] == public_condition:
            out[key[0]]["public"] = answer
    missing: list[tuple[int, str]] = []
    for prompt_id in prompt_ids:
        if "private" not in out[prompt_id]:
            missing.append((prompt_id, private_condition))
        if "public" not in out[prompt_id]:
            missing.append((prompt_id, public_condition))
    if missing:
        raise ValueError(f"Missing reference answers: {missing}")
    return dict(out)


def score_answer_margin(
    *,
    model: torch.nn.Module,
    tokenizer: Any,
    layers_mod: Any,
    rendered: Any,
    private_answer: str,
    public_answer: str,
    max_label_tokens: int,
    site: TraceSite | None = None,
    payload: Any | None = None,
    blend: float = 0.0,
) -> dict[str, float | int]:
    private = score_label(
        model=model,
        tokenizer=tokenizer,
        layers_mod=layers_mod,
        rendered=rendered,
        label_text=private_answer,
        max_label_tokens=max_label_tokens,
        site=site,
        payload=payload,
        blend=blend,
    )
    public = score_label(
        model=model,
        tokenizer=tokenizer,
        layers_mod=layers_mod,
        rendered=rendered,
        label_text=public_answer,
        max_label_tokens=max_label_tokens,
        site=site,
        payload=payload,
        blend=blend,
    )
    return {
        "private_logprob_mean": float(private["logprob_mean"]),
        "public_logprob_mean": float(public["logprob_mean"]),
        "private_logprob_sum": float(private["logprob_sum"]),
        "public_logprob_sum": float(public["logprob_sum"]),
        "private_tokens": int(private["label_tokens"]),
        "public_tokens": int(public["label_tokens"]),
        "margin_mean": float(private["logprob_mean"]) - float(public["logprob_mean"]),
        "margin_sum": float(private["logprob_sum"]) - float(public["logprob_sum"]),
    }


def aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, str, str, float], dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        if row["condition"] == "base":
            continue
        key = (int(row["layer"]), str(row["component"]), str(row["region"]), float(row["blend"]))
        grouped[key][str(row["source_kind"])].append(float(row["delta_margin_mean"]))
    out: list[dict[str, Any]] = []
    for (layer, component, region, blend), values in grouped.items():
        private_vals = values.get("private_source", [])
        public_vals = values.get("public_source_control", [])
        private_mean = float(np.mean(private_vals)) if private_vals else 0.0
        public_mean = float(np.mean(public_vals)) if public_vals else 0.0
        out.append(
            {
                "layer": layer,
                "component": component,
                "region": region,
                "blend": blend,
                "private_source_mean_delta": private_mean,
                "public_source_control_mean_delta": public_mean,
                "private_minus_public_control_delta": private_mean - public_mean,
                "n_private_source": len(private_vals),
                "n_public_source_control": len(public_vals),
            }
        )
    out.sort(
        key=lambda row: (
            float(row["private_minus_public_control_delta"]),
            float(row["private_source_mean_delta"]),
        ),
        reverse=True,
    )
    return out


def write_report(
    path: Path,
    *,
    manifest: dict[str, Any],
    aggregate_rows: list[dict[str, Any]],
    rows: list[dict[str, Any]],
) -> None:
    base_rows = [row for row in rows if row["condition"] == "base"]
    lines = [
        "# Article III Actual-Answer Continuation Trace",
        "",
        "## Purpose",
        "",
        (
            "Patch generated-thought hidden states into public-leaning target prompts and score actual "
            "private-conditioned versus public-conditioned answer continuations for the same prompt. "
            "This is a diagnostic localization screen, not a no-mask generation result."
        ),
        "",
        "## Config",
        "",
        markdown_table(
            ["Field", "Value"],
            [
                ["Started", manifest["started_at"]],
                ["Finished", manifest["finished_at"]],
                ["Model", manifest["model_path"]],
                ["Source generations", manifest["source_generations"]],
                ["Reference generations", manifest["reference_generations"]],
                ["Localization run", manifest["localization_run"]],
                ["Target prompt ids", ",".join(str(item) for item in manifest["target_prompt_ids"])],
                ["Private source ids", ",".join(str(item) for item in manifest["private_source_prompt_ids"])],
                ["Public source-control ids", ",".join(str(item) for item in manifest["public_control_prompt_ids"])],
                ["Top sites", manifest["top_sites"]],
                ["Blends", ",".join(str(item) for item in manifest["blends"])],
                ["Answer label tokens", manifest["answer_label_tokens"]],
                ["Source budget note", manifest["source_budget_note"]],
                ["Reference budget note", manifest["reference_budget_note"]],
                ["Output dir", manifest["output_dir"]],
            ],
        ),
        "",
        "## Baseline Margins",
        "",
        markdown_table(
            ["Prompt id", "Prompt key", "Private mean", "Public mean", "Margin"],
            [
                [
                    row["target_prompt_id"],
                    row["target_prompt_key"],
                    fmt(row["private_logprob_mean"]),
                    fmt(row["public_logprob_mean"]),
                    fmt(row["margin_mean"]),
                ]
                for row in base_rows
            ],
        ),
        "",
        "## Top Aggregate Sites",
        "",
        markdown_table(
            [
                "Rank",
                "Layer",
                "Component",
                "Region",
                "Blend",
                "Private delta",
                "Public-control delta",
                "Private - control",
            ],
            [
                [
                    idx + 1,
                    row["layer"],
                    row["component"],
                    row["region"],
                    fmt(row["blend"]),
                    fmt(row["private_source_mean_delta"]),
                    fmt(row["public_source_control_mean_delta"]),
                    fmt(row["private_minus_public_control_delta"]),
                ]
                for idx, row in enumerate(aggregate_rows[:20])
            ],
        ),
        "",
        "## Interpretation",
        "",
        "- A positive private-minus-control aggregate is candidate-localization evidence only.",
        "- The reference answers were generated under inserted thoughts, so this is an evaluator target, not no-mask success.",
        "- Any promoted actuator still needs long no-mask generation with visible reasoning movement and random/source/manual-review gates.",
        "",
        "## Artifacts",
        "",
        f"- Manifest: `{manifest['output_dir']}/manifest.json`.",
        f"- Patch rows: `{manifest['output_dir']}/patch_rows.jsonl`.",
        f"- Aggregate rows: `{manifest['output_dir']}/aggregate.jsonl`.",
    ]
    if rows:
        lines.extend(["", "Sample row:", "", "```json", json.dumps(rows[0], indent=2, sort_keys=True), "```"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--source-generations", type=Path, default=DEFAULT_SOURCE_GENERATIONS)
    parser.add_argument("--reference-generations", type=Path, default=DEFAULT_REFERENCE_GENERATIONS)
    parser.add_argument("--localization-run", type=Path, default=DEFAULT_LOCALIZATION_RUN)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--target-prompt-ids", default="2,4")
    parser.add_argument("--private-source-prompt-ids", default="0,1,5")
    parser.add_argument("--public-control-prompt-ids", default="3,6,7")
    parser.add_argument("--private-condition", default="private_rights")
    parser.add_argument("--public-condition", default="public_rights")
    parser.add_argument("--top-sites", type=int, default=6)
    parser.add_argument("--blends", default="1.0")
    parser.add_argument("--answer-label-tokens", type=int, default=256)
    parser.add_argument("--device-map", default="single")
    parser.add_argument("--allow-short-source-budget", action="store_true")
    parser.add_argument("--allow-short-reference-budget", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = now_iso()
    source_budget = source_budget_meta(args.source_generations, allow_short=args.allow_short_source_budget)
    reference_budget = source_budget_meta(args.reference_generations, allow_short=args.allow_short_reference_budget)
    reference_budget = {
        ("reference_" + key.removeprefix("source_")): value
        for key, value in reference_budget.items()
    }

    target_ids = parse_int_list(args.target_prompt_ids)
    private_source_ids = parse_int_list(args.private_source_prompt_ids)
    public_control_ids = parse_int_list(args.public_control_prompt_ids)
    blends = parse_float_list(args.blends)
    sites = load_top_sites(args.localization_run, top_sites=args.top_sites)
    out_dir = args.output_root / f"scotus_article3_answer_continuation_trace_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)

    generation_rows = read_jsonl(args.source_generations)
    base_rows = {
        int(row["prompt_id"]): row
        for row in generation_rows
        if str(row.get("condition")) == "base"
        and int(row.get("prompt_id", -1)) in set(target_ids + private_source_ids + public_control_ids)
    }
    missing = sorted(set(target_ids + private_source_ids + public_control_ids) - set(base_rows))
    if missing:
        raise ValueError(f"Missing base source generation rows for prompt ids: {missing}")
    answers = reference_answers(
        read_jsonl(args.reference_generations),
        prompt_ids=target_ids,
        private_condition=args.private_condition,
        public_condition=args.public_condition,
    )

    tokenizer, model = load_model_and_tokenizer(args.model_path, args.device_map)
    layers_mod = transformer_layers(model)
    rendered = {prompt_id: render_trace(tokenizer, row) for prompt_id, row in base_rows.items()}
    print(
        f"Scoring {len(target_ids)} targets, {len(sites)} sites, "
        f"{len(private_source_ids)} private source(s), {len(public_control_ids)} public control(s), "
        f"answer_label_tokens={args.answer_label_tokens}",
        flush=True,
    )

    rows: list[dict[str, Any]] = []
    base_margins: dict[int, dict[str, float | int]] = {}
    for target_id in target_ids:
        margin = score_answer_margin(
            model=model,
            tokenizer=tokenizer,
            layers_mod=layers_mod,
            rendered=rendered[target_id],
            private_answer=answers[target_id]["private"],
            public_answer=answers[target_id]["public"],
            max_label_tokens=args.answer_label_tokens,
        )
        base_margins[target_id] = margin
        rows.append(
            {
                "condition": "base",
                "target_prompt_id": target_id,
                "target_prompt_key": rendered[target_id].prompt_key,
                "source_prompt_id": None,
                "source_prompt_key": None,
                "source_kind": None,
                "layer": None,
                "component": None,
                "region": None,
                "blend": 0.0,
                "payload_mode": None,
                "patch_positions": 0,
                **margin,
                "delta_margin_mean": 0.0,
            }
        )

    source_specs = [(prompt_id, "private_source") for prompt_id in private_source_ids] + [
        (prompt_id, "public_source_control") for prompt_id in public_control_ids
    ]
    for site in tqdm(sites, desc="trace sites", unit="site"):
        source_values_by_id = {
            prompt_id: capture_site(
                model=model,
                tokenizer=tokenizer,
                layers_mod=layers_mod,
                rendered=rendered[prompt_id],
                site=site,
            )
            for prompt_id, _kind in source_specs
        }
        for target_id in target_ids:
            for source_id, source_kind in source_specs:
                payload = make_payload(source_values=source_values_by_id[source_id], target=rendered[target_id], site=site)
                for blend in blends:
                    margin = score_answer_margin(
                        model=model,
                        tokenizer=tokenizer,
                        layers_mod=layers_mod,
                        rendered=rendered[target_id],
                        private_answer=answers[target_id]["private"],
                        public_answer=answers[target_id]["public"],
                        max_label_tokens=args.answer_label_tokens,
                        site=site,
                        payload=payload,
                        blend=blend,
                    )
                    rows.append(
                        {
                            "condition": "patched",
                            "target_prompt_id": target_id,
                            "target_prompt_key": rendered[target_id].prompt_key,
                            "source_prompt_id": source_id,
                            "source_prompt_key": rendered[source_id].prompt_key,
                            "source_kind": source_kind,
                            "layer": site.layer,
                            "component": site.component,
                            "region": site.region,
                            "blend": blend,
                            "payload_mode": payload.mode,
                            "patch_positions": len(payload.positions),
                            **margin,
                            "delta_margin_mean": float(margin["margin_mean"])
                            - float(base_margins[target_id]["margin_mean"]),
                        }
                    )

    aggregate_rows = aggregate(rows)
    manifest = {
        "started_at": started,
        "finished_at": now_iso(),
        "model_path": str(args.model_path),
        "source_generations": str(args.source_generations),
        "reference_generations": str(args.reference_generations),
        "localization_run": str(args.localization_run),
        "output_dir": str(out_dir),
        "target_prompt_ids": target_ids,
        "private_source_prompt_ids": private_source_ids,
        "public_control_prompt_ids": public_control_ids,
        "private_condition": args.private_condition,
        "public_condition": args.public_condition,
        "top_sites": args.top_sites,
        "site_keys": [site.key for site in sites],
        "blends": blends,
        "answer_label_tokens": args.answer_label_tokens,
        "method": "actual_answer_continuation_logit_causal_trace",
        "not_generation_evidence": True,
        "not_promotion_evidence": True,
        **source_budget,
        **reference_budget,
    }
    write_json(out_dir / "manifest.json", manifest)
    write_jsonl(out_dir / "patch_rows.jsonl", rows)
    write_jsonl(out_dir / "aggregate.jsonl", aggregate_rows)
    write_report(out_dir / "report.md", manifest=manifest, aggregate_rows=aggregate_rows, rows=rows)

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(f"Wrote {out_dir}", flush=True)
    if aggregate_rows:
        top = aggregate_rows[0]
        print(
            "Top site: "
            f"L{top['layer']} {top['component']} {top['region']} blend={top['blend']} "
            f"private-control-delta={top['private_minus_public_control_delta']:.4f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
