#!/usr/bin/env python3
"""Counterfactual visible-thought diagnostic for Article III prompts.

This is an evaluator calibration, not a steering result. It inserts coherent
private-rights or public-rights scratchpads before final-answer generation and
checks whether the answer follows the visible reasoning frame. If clean
counterfactual thoughts do not control answers, visible-thinking steering is a
poor actuator target; if they do, future activation work must still make the
model produce that reasoning itself rather than relying on an inserted mask.
"""

from __future__ import annotations

import argparse
import gc
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm

from ablate_scotus_thinking_text import answer_from_thought
from patch_scotus_thinking_traces import DEFAULT_MODEL
from poke_scotus_sae_layers import (
    DEFAULT_OUTPUT_ROOT,
    load_model_and_tokenizer,
    load_prompt_specs,
    now_iso,
    select_prompt_specs,
    write_json,
    write_jsonl,
)
from poke_scotus_thinking_bundle import clean_snippet, fmt, md_table, mean, segment_scores, stdev
from qwen_eval_budget import (
    DEFAULT_COMPLETE_ANSWER_TOKENS,
    MIN_COMPLETE_ANSWER_TOKENS,
    enforce_complete_answer_budget,
    qwen_budget_metadata,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PROMPT_BANK = PROJECT_ROOT / "data" / "scotus" / "scotus_article3_private_public_poke_prompts_v2.jsonl"

PRIVATE_THOUGHT = """The Article III question should be framed as a private-rights adjudication problem. The dispute concerns liability or entitlement between private parties, or a traditional common-law-style claim, and Congress cannot convert that matter into a final non-Article-III judgment merely by assigning it to an Article I or agency tribunal. The important distinction is that public-rights adjudication permits more administrative factfinding, while private-rights disputes require an Article III court for final judgment unless a narrow adjunct, consent, or appellate-review exception applies."""

PUBLIC_THOUGHT = """The Article III question should be framed as a public-rights adjudication problem. Congress may create federal statutory benefits, regulatory schemes, patents, tariffs, or enforcement systems and assign initial factfinding or adjudication to an agency or Article I tribunal, so long as Article III judicial review remains available. The important distinction is that public rights arise from a federal regulatory program or sovereign scheme, unlike private-rights disputes that resemble traditional common-law liability between private parties."""

NEUTRAL_THOUGHT = """The Article III issue turns on the public-rights/private-rights distinction, the role of the non-Article-III adjudicator, the availability of Article III judicial review, and whether the matter resembles a traditional judicial dispute or a congressionally created federal scheme. The answer should apply those considerations to the facts without relying on labels alone."""


def row_for_counterfactual(*, spec: Any, condition: str, thought: str, output: dict[str, Any]) -> dict[str, Any]:
    thinking_scores = segment_scores(spec, output["thinking"])
    answer_scores = segment_scores(spec, output["answer"])
    full_scores = segment_scores(spec, output["full_text"])
    return {
        "prompt_id": spec.prompt_id,
        "prompt_key": spec.prompt_key,
        "issue_area": spec.issue_area,
        "prompt": spec.prompt,
        "expected_frames": list(spec.expected_frames),
        "contrast_frames": list(spec.contrast_frames),
        "domain_frames": list(spec.domain_frames),
        "condition": condition,
        "inserted_thought": thought,
        **output,
        "thinking_frame_eval": thinking_scores["frame_eval"],
        "thinking_proposition_frame_scores": thinking_scores["proposition_frame_scores"],
        "thinking_proposition_frame_evidence": thinking_scores["proposition_frame_evidence"],
        "thinking_proposition_frame_eval": thinking_scores["proposition_frame_eval"],
        "answer_frame_eval": answer_scores["frame_eval"],
        "answer_proposition_frame_scores": answer_scores["proposition_frame_scores"],
        "answer_proposition_frame_evidence": answer_scores["proposition_frame_evidence"],
        "answer_proposition_frame_eval": answer_scores["proposition_frame_eval"],
        "full_text_frame_eval": full_scores["frame_eval"],
        "full_text_proposition_frame_scores": full_scores["proposition_frame_scores"],
        "full_text_proposition_frame_evidence": full_scores["proposition_frame_evidence"],
        "full_text_proposition_frame_eval": full_scores["proposition_frame_eval"],
    }


def add_pair_deltas(rows: list[dict[str, Any]]) -> None:
    neutral_by_prompt = {
        int(row["prompt_id"]): row["answer_proposition_frame_eval"]
        for row in rows
        if row["condition"] == "neutral"
    }
    public_by_prompt = {
        int(row["prompt_id"]): row["answer_proposition_frame_eval"]
        for row in rows
        if row["condition"] == "public_rights"
    }
    private_by_prompt = {
        int(row["prompt_id"]): row["answer_proposition_frame_eval"]
        for row in rows
        if row["condition"] == "private_rights"
    }
    for row in rows:
        current = row["answer_proposition_frame_eval"]
        neutral = neutral_by_prompt.get(int(row["prompt_id"]))
        if neutral is not None:
            current["delta_target_hits_vs_neutral"] = float(current["target_hits"] - neutral["target_hits"])
            current["delta_contrast_hits_vs_neutral"] = float(current["contrast_hits"] - neutral["contrast_hits"])
            current["delta_target_minus_contrast_vs_neutral"] = float(
                (current["target_hits"] - current["contrast_hits"])
                - (neutral["target_hits"] - neutral["contrast_hits"])
            )
        public = public_by_prompt.get(int(row["prompt_id"]))
        private = private_by_prompt.get(int(row["prompt_id"]))
        if public is not None and private is not None and row["condition"] == "private_rights":
            current["private_minus_public_target_hits"] = float(current["target_hits"] - public["target_hits"])
            current["private_minus_public_contrast_hits"] = float(current["contrast_hits"] - public["contrast_hits"])
            current["private_minus_public_net"] = float(
                (current["target_hits"] - current["contrast_hits"])
                - (public["target_hits"] - public["contrast_hits"])
            )


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row["condition"])].append(row)
    summaries: list[dict[str, Any]] = []
    for condition, group_rows in sorted(groups.items()):
        evals = [row["answer_proposition_frame_eval"] for row in group_rows]
        summaries.append(
            {
                "condition": condition,
                "n": len(group_rows),
                "mean_target_hits": mean([float(item["target_hits"]) for item in evals]),
                "mean_contrast_hits": mean([float(item["contrast_hits"]) for item in evals]),
                "mean_target_minus_contrast": mean(
                    [float(item["target_hits"] - item["contrast_hits"]) for item in evals]
                ),
                "mean_delta_target_minus_contrast_vs_neutral": mean(
                    [float(item.get("delta_target_minus_contrast_vs_neutral", 0.0)) for item in evals]
                ),
                "sd_delta_target_minus_contrast_vs_neutral": stdev(
                    [float(item.get("delta_target_minus_contrast_vs_neutral", 0.0)) for item in evals]
                ),
                "answer_nonempty_rate": mean([1.0 if row["answer_nonempty"] else 0.0 for row in group_rows]),
                "imitation_marker_rate": mean(
                    [1.0 if row["thinking_imitation_markers"] or row["answer_imitation_markers"] else 0.0 for row in group_rows]
                ),
            }
        )
    private_rows = [row for row in rows if row["condition"] == "private_rights"]
    if private_rows:
        evals = [row["answer_proposition_frame_eval"] for row in private_rows]
        summaries.append(
            {
                "condition": "private_minus_public",
                "n": len(private_rows),
                "mean_target_hits": mean([float(item.get("private_minus_public_target_hits", 0.0)) for item in evals]),
                "mean_contrast_hits": mean(
                    [float(item.get("private_minus_public_contrast_hits", 0.0)) for item in evals]
                ),
                "mean_target_minus_contrast": mean([float(item.get("private_minus_public_net", 0.0)) for item in evals]),
                "mean_delta_target_minus_contrast_vs_neutral": 0.0,
                "sd_delta_target_minus_contrast_vs_neutral": stdev(
                    [float(item.get("private_minus_public_net", 0.0)) for item in evals]
                ),
                "answer_nonempty_rate": 1.0,
                "imitation_marker_rate": 0.0,
            }
        )
    return summaries


def write_report(
    path: Path,
    *,
    manifest: dict[str, Any],
    summaries: list[dict[str, Any]],
    rows: list[dict[str, Any]],
) -> None:
    lines = [
        "# SCOTUS Counterfactual Visible Thoughts",
        "",
        "## Configuration",
        "",
        f"- Model: `{manifest['model_path']}`",
        f"- Prompt bank: `{manifest['prompt_bank']}`",
        f"- Prompt ids: `{', '.join(str(item) for item in manifest['prompt_ids'])}`",
        f"- Answer tokens: `{manifest['answer_tokens']}`",
        f"- Short-budget smoke: `{manifest['short_answer_budget']}`",
        "",
        "## Summary",
        "",
        *md_table(
            [
                "condition",
                "n",
                "target_hits",
                "contrast_hits",
                "target-minus-contrast",
                "net-vs-neutral",
                "net-vs-neutral-sd",
                "answer_rate",
                "mask_rate",
            ],
            [
                [
                    item["condition"],
                    item["n"],
                    fmt(item["mean_target_hits"]),
                    fmt(item["mean_contrast_hits"]),
                    fmt(item["mean_target_minus_contrast"]),
                    fmt(item["mean_delta_target_minus_contrast_vs_neutral"]),
                    fmt(item["sd_delta_target_minus_contrast_vs_neutral"]),
                    fmt(item["answer_nonempty_rate"]),
                    fmt(item["imitation_marker_rate"]),
                ]
                for item in summaries
            ],
        ),
        "",
        "## Samples",
        "",
    ]
    for row in rows:
        if row["prompt_id"] not in manifest["sample_prompt_ids"]:
            continue
        lines.extend(
            [
                f"### {row['prompt_key']} / {row['condition']}",
                "",
                f"- answer target/contrast: `{row['answer_proposition_frame_eval']['target_hits']}` / `{row['answer_proposition_frame_eval']['contrast_hits']}`",
                f"- answer net vs neutral: `{fmt(row['answer_proposition_frame_eval'].get('delta_target_minus_contrast_vs_neutral', 0.0))}`",
                "",
                "Inserted thought:",
                "",
                clean_snippet(row["inserted_thought"], max_chars=400),
                "",
                "Answer snippet:",
                "",
                clean_snippet(row["answer"], max_chars=500) or "[none]",
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--prompt-bank", type=Path, default=DEFAULT_PROMPT_BANK)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--prompt-ids", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--max-prompts", type=int, default=8)
    parser.add_argument("--answer-tokens", type=int, default=DEFAULT_COMPLETE_ANSWER_TOKENS)
    parser.add_argument(
        "--allow-short-answer-budget",
        action="store_true",
        help=(
            f"Permit answer budgets below {MIN_COMPLETE_ANSWER_TOKENS} tokens. "
            "Short-budget runs are smoke/debug only and must not be used for promotion."
        ),
    )
    parser.add_argument("--device-map", default="single")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    enforce_complete_answer_budget(
        args.answer_tokens,
        allow_short=args.allow_short_answer_budget,
        label="answer_tokens",
        purpose="SCOTUS counterfactual-thought answer run",
    )
    started = now_iso()
    out_dir = args.output_root / f"scotus_counterfactual_thoughts_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)

    prompt_specs = select_prompt_specs(load_prompt_specs(args.prompt_bank), args.prompt_ids, args.max_prompts)
    variants = [
        ("neutral", NEUTRAL_THOUGHT),
        ("private_rights", PRIVATE_THOUGHT),
        ("public_rights", PUBLIC_THOUGHT),
    ]
    tokenizer, model = load_model_and_tokenizer(args.model_path, args.device_map)

    rows: list[dict[str, Any]] = []
    tasks = [(spec, condition, thought) for spec in prompt_specs for condition, thought in variants]
    for spec, condition, thought in tqdm(tasks, desc="Generating counterfactual answers"):
        output = answer_from_thought(
            model=model,
            tokenizer=tokenizer,
            prompt=spec.prompt,
            thought=thought,
            answer_tokens=args.answer_tokens,
        )
        rows.append(row_for_counterfactual(spec=spec, condition=condition, thought=thought, output=output))

    add_pair_deltas(rows)
    summaries = summarize(rows)
    manifest = {
        "started_at": started,
        "finished_at": now_iso(),
        "model_path": str(args.model_path),
        "prompt_bank": str(args.prompt_bank),
        "output_dir": str(out_dir),
        "prompt_ids": [spec.prompt_id for spec in prompt_specs],
        "prompt_keys": [spec.prompt_key for spec in prompt_specs],
        "sample_prompt_ids": [prompt_specs[0].prompt_id, prompt_specs[-1].prompt_id] if prompt_specs else [],
        "conditions": [condition for condition, _thought in variants],
        "answer_tokens": int(args.answer_tokens),
        **qwen_budget_metadata(args.answer_tokens),
    }
    write_json(out_dir / "manifest.json", manifest)
    write_jsonl(out_dir / "generations.jsonl", rows)
    write_jsonl(out_dir / "summary.jsonl", summaries)
    write_report(out_dir / "report.md", manifest=manifest, summaries=summaries, rows=rows)

    del model, tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(f"Wrote {out_dir / 'report.md'}", flush=True)


if __name__ == "__main__":
    main()
