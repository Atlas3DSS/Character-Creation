#!/usr/bin/env python3
"""Text-level attribution screen for SCOTUS visible thinking.

This is not a steering run. It asks whether the visible thought text itself
causally affects the final proposition by regenerating answers from edited
thought traces. If deleting or isolating thought windows does not change the
answer frame, activation-level trace replacement is unlikely to be localizable
from those windows.
"""

from __future__ import annotations

import argparse
import gc
import json
import random
import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm

from patch_scotus_thinking_traces import DEFAULT_MODEL, generate_manual
from poke_scotus_sae_layers import (
    DEFAULT_OUTPUT_ROOT,
    load_model_and_tokenizer,
    load_prompt_specs,
    now_iso,
    parse_int_list,
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
from run_scotus_thinking_smoke import IMITATION_RE, format_chat, strip_generation_specials


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PROMPT_BANK = PROJECT_ROOT / "data" / "scotus" / "scotus_article3_private_public_poke_prompts_v2.jsonl"
DEFAULT_SOURCE_GENERATIONS = (
    PROJECT_ROOT / "sweep_v4" / "scotus_thinking_trace_patch_20260501_224155" / "generations.jsonl"
)


@dataclass(frozen=True)
class TokenWindow:
    start: int
    end: int | None

    @property
    def label(self) -> str:
        if self.end is None:
            return "all" if self.start == 0 else f"w{self.start:03d}_end"
        return f"w{self.start:03d}_{self.end:03d}"

    def as_manifest(self) -> dict[str, int | str | None]:
        return {"label": self.label, "start": self.start, "end": self.end}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON") from exc
    return rows


def parse_token_windows(raw: str) -> list[TokenWindow]:
    value = raw.strip().lower()
    if not value or value == "all":
        return [TokenWindow(start=0, end=None)]
    windows: list[TokenWindow] = []
    for item in value.split(","):
        token = item.strip()
        if not token:
            continue
        if ":" not in token:
            start = int(token)
            end = start + 1
        else:
            start_raw, end_raw = token.split(":", maxsplit=1)
            start = int(start_raw) if start_raw else 0
            end = int(end_raw) if end_raw else None
        if start < 0:
            raise ValueError(f"Window start must be >= 0: {token!r}")
        if end is not None and end <= start:
            raise ValueError(f"Window end must be > start: {token!r}")
        windows.append(TokenWindow(start=start, end=end))
    if not windows:
        raise ValueError(f"No token windows parsed from {raw!r}")
    return windows


def source_base_rows(rows: list[dict[str, Any]], prompt_ids: set[int]) -> dict[int, dict[str, Any]]:
    selected: dict[int, dict[str, Any]] = {}
    for row in rows:
        if row.get("condition") != "base":
            continue
        prompt_id = int(row["prompt_id"])
        if prompt_id not in prompt_ids:
            continue
        thinking = str(row.get("thinking") or "").strip()
        if not thinking:
            continue
        selected.setdefault(prompt_id, row)
    missing = sorted(prompt_ids - set(selected))
    if missing:
        raise ValueError(f"No nonempty base thinking rows for prompt ids: {missing}")
    return selected


def token_slice(ids: list[int], window: TokenWindow) -> tuple[int, int]:
    start = min(max(0, window.start), len(ids))
    end = len(ids) if window.end is None else min(max(start, window.end), len(ids))
    return start, end


def decode_ids(tokenizer: Any, ids: list[int]) -> str:
    return tokenizer.decode(ids, skip_special_tokens=True).strip()


def thought_variants(
    *,
    tokenizer: Any,
    thought: str,
    windows: list[TokenWindow],
    random_controls: int,
    seed: int,
) -> list[dict[str, Any]]:
    ids = tokenizer(thought, add_special_tokens=False).input_ids
    variants: list[dict[str, Any]] = [
        {
            "condition": "original",
            "candidate": "original_thought",
            "window": None,
            "random_index": None,
            "edited_thinking": thought,
            "removed_token_count": 0,
            "kept_token_count": len(ids),
        },
        {
            "condition": "empty",
            "candidate": "empty_thought",
            "window": None,
            "random_index": None,
            "edited_thinking": "",
            "removed_token_count": len(ids),
            "kept_token_count": 0,
        },
    ]
    for window in windows:
        start, end = token_slice(ids, window)
        kept = ids[:start] + ids[end:]
        variants.append(
            {
                "condition": "drop_window",
                "candidate": f"drop_{window.label}",
                "window": window.label,
                "random_index": None,
                "edited_thinking": decode_ids(tokenizer, kept),
                "removed_token_count": end - start,
                "kept_token_count": len(kept),
            }
        )
        variants.append(
            {
                "condition": "keep_window",
                "candidate": f"keep_{window.label}",
                "window": window.label,
                "random_index": None,
                "edited_thinking": decode_ids(tokenizer, ids[start:end]),
                "removed_token_count": len(ids) - (end - start),
                "kept_token_count": end - start,
            }
        )
    rng = random.Random(seed)
    for random_index in range(random_controls):
        if not windows:
            break
        reference = windows[random_index % len(windows)]
        width = max(1, (len(ids) if reference.end is None else reference.end) - reference.start)
        if len(ids) <= width:
            start = 0
        else:
            start = rng.randint(0, len(ids) - width)
        end = min(len(ids), start + width)
        kept = ids[:start] + ids[end:]
        variants.append(
            {
                "condition": "random_drop",
                "candidate": f"random_drop_{random_index}",
                "window": f"w{start:03d}_{end:03d}",
                "random_index": random_index,
                "edited_thinking": decode_ids(tokenizer, kept),
                "removed_token_count": end - start,
                "kept_token_count": len(kept),
            }
        )
    return variants


def answer_from_thought(
    *,
    model: torch.nn.Module,
    tokenizer: Any,
    prompt: str,
    thought: str,
    answer_tokens: int,
) -> dict[str, Any]:
    chat = format_chat(tokenizer, prompt, enable_thinking=True)
    prefilled_open_think = chat.rstrip().endswith("<think>")
    answer_prompt = f"{chat}{thought.strip()}\n</think>\n\n"
    output = generate_manual(
        model=model,
        tokenizer=tokenizer,
        prompt=answer_prompt,
        max_new_tokens=answer_tokens,
        state=None,
    )
    answer = strip_generation_specials(str(output["raw_text"])).strip()
    return {
        "prefilled_open_think": prefilled_open_think,
        "thinking": thought.strip(),
        "answer": answer,
        "full_text": f"<think>\n{thought.strip()}\n</think>\n\n{answer}".strip(),
        "answer_generated_tokens": int(output["generated_tokens"]),
        "answer_prompt_tokens": int(output["prompt_tokens"]),
        "thinking_nonempty": bool(thought.strip()),
        "answer_nonempty": bool(answer),
        "thinking_imitation_markers": sorted(set(IMITATION_RE.findall(thought))),
        "answer_imitation_markers": sorted(set(IMITATION_RE.findall(answer))),
    }


def row_for_variant(*, spec: Any, variant: dict[str, Any], output: dict[str, Any]) -> dict[str, Any]:
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
        "condition": variant["condition"],
        "candidate": variant["candidate"],
        "window": variant["window"],
        "random_index": variant["random_index"],
        "removed_token_count": int(variant["removed_token_count"]),
        "kept_token_count": int(variant["kept_token_count"]),
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


def add_answer_deltas(rows: list[dict[str, Any]]) -> None:
    base_by_prompt = {
        int(row["prompt_id"]): row["answer_proposition_frame_eval"]
        for row in rows
        if row["condition"] == "original"
    }
    for row in rows:
        base = base_by_prompt.get(int(row["prompt_id"]))
        if base is None:
            continue
        current = row["answer_proposition_frame_eval"]
        for key in ("target_hits", "contrast_hits", "off_domain_hits", "total_frame_hits"):
            current[f"delta_{key}_vs_original"] = float(current[key] - base[key])
        current["delta_target_minus_contrast_vs_original"] = float(
            (current["target_hits"] - current["contrast_hits"]) - (base["target_hits"] - base["contrast_hits"])
        )


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(str(row["condition"]), str(row["candidate"]))].append(row)
    summaries: list[dict[str, Any]] = []
    for (condition, candidate), group_rows in sorted(groups.items(), key=str):
        evals = [row["answer_proposition_frame_eval"] for row in group_rows]
        summaries.append(
            {
                "condition": condition,
                "candidate": candidate,
                "n": len(group_rows),
                "prompt_count": len({row["prompt_id"] for row in group_rows}),
                "mean_target_hits": mean([float(item["target_hits"]) for item in evals]),
                "mean_contrast_hits": mean([float(item["contrast_hits"]) for item in evals]),
                "mean_delta_target_hits_vs_original": mean(
                    [float(item.get("delta_target_hits_vs_original", 0.0)) for item in evals]
                ),
                "mean_delta_target_minus_contrast_vs_original": mean(
                    [float(item.get("delta_target_minus_contrast_vs_original", 0.0)) for item in evals]
                ),
                "sd_delta_target_minus_contrast_vs_original": stdev(
                    [float(item.get("delta_target_minus_contrast_vs_original", 0.0)) for item in evals]
                ),
                "answer_nonempty_rate": mean([1.0 if row["answer_nonempty"] else 0.0 for row in group_rows]),
                "imitation_marker_rate": mean(
                    [1.0 if row["thinking_imitation_markers"] or row["answer_imitation_markers"] else 0.0 for row in group_rows]
                ),
                "mean_removed_tokens": mean([float(row["removed_token_count"]) for row in group_rows]),
                "mean_kept_tokens": mean([float(row["kept_token_count"]) for row in group_rows]),
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
        "# SCOTUS Visible-Thought Text Attribution",
        "",
        "## Configuration",
        "",
        f"- Model: `{manifest['model_path']}`",
        f"- Source generations: `{manifest['source_generations']}`",
        f"- Prompt bank: `{manifest['prompt_bank']}`",
        f"- Prompt ids: `{', '.join(str(item) for item in manifest['prompt_ids'])}`",
        f"- Token windows: `{', '.join(item['label'] for item in manifest['token_windows'])}`",
        f"- Random drop controls: `{manifest['random_drop_controls']}`",
        f"- Answer tokens: `{manifest['answer_tokens']}`",
        f"- Short-budget smoke: `{manifest['short_answer_budget']}`",
        "",
        "## Summary",
        "",
        *md_table(
            [
                "condition",
                "candidate",
                "n",
                "target_delta",
                "net_delta",
                "net_sd",
                "answer_rate",
                "mask_rate",
                "removed",
                "kept",
            ],
            [
                [
                    item["condition"],
                    item["candidate"],
                    item["n"],
                    fmt(item["mean_delta_target_hits_vs_original"]),
                    fmt(item["mean_delta_target_minus_contrast_vs_original"]),
                    fmt(item["sd_delta_target_minus_contrast_vs_original"]),
                    fmt(item["answer_nonempty_rate"]),
                    fmt(item["imitation_marker_rate"]),
                    fmt(item["mean_removed_tokens"]),
                    fmt(item["mean_kept_tokens"]),
                ]
                for item in summaries
            ],
        ),
        "",
        "## Samples",
        "",
    ]
    for row in rows:
        if row["condition"] not in {"original", "empty", "drop_window", "keep_window"}:
            continue
        lines.extend(
            [
                f"### {row['prompt_key']} / {row['candidate']}",
                "",
                f"- answer target/net delta vs original: `{fmt(row['answer_proposition_frame_eval'].get('delta_target_hits_vs_original', 0.0))}` / `{fmt(row['answer_proposition_frame_eval'].get('delta_target_minus_contrast_vs_original', 0.0))}`",
                f"- removed/kept tokens: `{row['removed_token_count']}` / `{row['kept_token_count']}`",
                f"- answer nonempty: `{row['answer_nonempty']}`",
                "",
                "Thinking snippet:",
                "",
                clean_snippet(row["thinking"]) or "[none]",
                "",
                "Answer snippet:",
                "",
                clean_snippet(row["answer"]) or "[none]",
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--prompt-bank", type=Path, default=DEFAULT_PROMPT_BANK)
    parser.add_argument("--source-generations", type=Path, default=DEFAULT_SOURCE_GENERATIONS)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--prompt-ids", default="1,4")
    parser.add_argument("--max-prompts", type=int, default=2)
    parser.add_argument("--token-windows", default="0:32,32:64,64:96")
    parser.add_argument("--random-drop-controls", type=int, default=2)
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
    parser.add_argument("--seed", type=int, default=20260517)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    enforce_complete_answer_budget(
        args.answer_tokens,
        allow_short=args.allow_short_answer_budget,
        label="answer_tokens",
        purpose="SCOTUS thinking-text ablation answer run",
    )
    started = now_iso()
    out_dir = args.output_root / f"scotus_thinking_text_ablation_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)

    prompt_specs = select_prompt_specs(load_prompt_specs(args.prompt_bank), args.prompt_ids, args.max_prompts)
    prompt_ids = {int(spec.prompt_id) for spec in prompt_specs}
    base_rows = source_base_rows(read_jsonl(args.source_generations), prompt_ids)
    windows = parse_token_windows(args.token_windows)

    print(f"Loaded {len(prompt_specs)} prompts", flush=True)
    print(f"Token windows: {', '.join(window.label for window in windows)}", flush=True)
    tokenizer, model = load_model_and_tokenizer(args.model_path, args.device_map)

    rows: list[dict[str, Any]] = []
    tasks: list[tuple[Any, dict[str, Any]]] = []
    for spec in prompt_specs:
        base_thought = str(base_rows[int(spec.prompt_id)]["thinking"])
        variants = thought_variants(
            tokenizer=tokenizer,
            thought=base_thought,
            windows=windows,
            random_controls=max(0, args.random_drop_controls),
            seed=args.seed + int(spec.prompt_id) * 1009,
        )
        for variant in variants:
            tasks.append((spec, variant))

    for spec, variant in tqdm(tasks, desc="Generating answer variants"):
        output = answer_from_thought(
            model=model,
            tokenizer=tokenizer,
            prompt=spec.prompt,
            thought=str(variant["edited_thinking"]),
            answer_tokens=args.answer_tokens,
        )
        rows.append(row_for_variant(spec=spec, variant=variant, output=output))

    add_answer_deltas(rows)
    summaries = summarize(rows)
    manifest = {
        "started_at": started,
        "finished_at": now_iso(),
        "model_path": str(args.model_path),
        "prompt_bank": str(args.prompt_bank),
        "source_generations": str(args.source_generations),
        "output_dir": str(out_dir),
        "prompt_ids": [spec.prompt_id for spec in prompt_specs],
        "prompt_keys": [spec.prompt_key for spec in prompt_specs],
        "token_windows": [window.as_manifest() for window in windows],
        "random_drop_controls": int(args.random_drop_controls),
        "answer_tokens": int(args.answer_tokens),
        **qwen_budget_metadata(args.answer_tokens),
        "seed": int(args.seed),
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
