#!/usr/bin/env python3
"""Run a small thinking-enabled SCOTUS generation smoke.

The steering project's success standard requires more than final-answer
movement. Where the model exposes a thinking trace, future candidates must show
target-frame reasoning in that trace rather than an imitation mask. This script
checks whether the local Qwen chat template yields parseable `<think>` traces
for the current legal prompt bank.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import torch

from poke_scotus_sae_layers import (
    DEFAULT_OUTPUT_ROOT,
    first_parameter_device,
    load_model_and_tokenizer,
    load_prompt_specs,
    now_iso,
    select_prompt_specs,
    write_json,
    write_jsonl,
)
from qwen_eval_budget import MIN_COMPLETE_ANSWER_TOKENS, qwen_budget_metadata


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MODEL = Path("/home/orwel/dev_genius/models/Qwen3.5-27B")
DEFAULT_PROMPT_BANK = PROJECT_ROOT / "data" / "scotus" / "scotus_article3_private_public_poke_prompts_v2.jsonl"

THINK_RE = re.compile(r"(?is)<think>\s*(.*?)\s*</think>\s*(.*)\Z")
OPEN_THINK_RE = re.compile(r"(?is)\A\s*<think>\s*(.*)\Z")
GENERATION_SPECIAL_RE = re.compile(r"<\|(?:im_end|endoftext)\|>")
IMITATION_RE = re.compile(
    r"(?i)\b(?:imitat(?:e|ing)|role[- ]?play|as (?:justice|judge|the target)|"
    r"think like|would reason|in the style of|persona)\b"
)


def strip_generation_specials(text: str) -> str:
    return GENERATION_SPECIAL_RE.sub("", text).strip()


def parse_thinking(raw_text: str, *, fallback_text: str, prefilled_open_think: bool) -> dict[str, Any]:
    stripped = strip_generation_specials(raw_text)
    match = THINK_RE.match(stripped)
    if match:
        thinking = strip_generation_specials(match.group(1))
        answer = strip_generation_specials(match.group(2))
        return {
            "thinking": thinking,
            "answer": answer,
            "has_visible_thinking": bool(thinking),
            "thinking_closed": True,
            "answer_nonempty": bool(answer),
            "imitation_markers": sorted(set(IMITATION_RE.findall(thinking))),
        }
    if prefilled_open_think:
        thinking_text, separator, answer_text = stripped.partition("</think>")
        thinking = strip_generation_specials(thinking_text)
        answer = strip_generation_specials(answer_text) if separator else ""
        return {
            "thinking": thinking,
            "answer": answer,
            "has_visible_thinking": bool(thinking),
            "thinking_closed": bool(separator),
            "answer_nonempty": bool(answer),
            "imitation_markers": sorted(set(IMITATION_RE.findall(thinking))),
        }
    match = OPEN_THINK_RE.match(stripped)
    if match:
        thinking = strip_generation_specials(match.group(1))
        return {
            "thinking": thinking,
            "answer": "",
            "has_visible_thinking": bool(thinking),
            "thinking_closed": False,
            "answer_nonempty": False,
            "imitation_markers": sorted(set(IMITATION_RE.findall(thinking))),
        }
    return {
        "thinking": "",
        "answer": fallback_text.strip(),
        "has_visible_thinking": False,
        "thinking_closed": False,
        "answer_nonempty": bool(fallback_text.strip()),
        "imitation_markers": [],
    }


def format_chat(tokenizer: Any, prompt: str, *, enable_thinking: bool) -> str:
    messages = [{"role": "user", "content": prompt}]
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
    except TypeError:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


@torch.inference_mode()
def generate_rows(
    *,
    model: torch.nn.Module,
    tokenizer: Any,
    prompt_specs: list[Any],
    enable_thinking: bool,
    max_new_tokens: int,
) -> list[dict[str, Any]]:
    input_device = first_parameter_device(model)
    rows: list[dict[str, Any]] = []
    for spec in prompt_specs:
        chat = format_chat(tokenizer, spec.prompt, enable_thinking=enable_thinking)
        inputs = tokenizer(
            chat,
            return_tensors="pt",
            add_special_tokens=False,
            truncation=True,
            max_length=2048,
        )
        inputs = {key: value.to(input_device) for key, value in inputs.items()}
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
        generated = output[0, inputs["input_ids"].shape[-1] :]
        raw_text = tokenizer.decode(generated, skip_special_tokens=False).strip()
        text = tokenizer.decode(generated, skip_special_tokens=True).strip()
        prefilled_open_think = enable_thinking and chat.rstrip().endswith("<think>")
        parsed = parse_thinking(raw_text, fallback_text=text, prefilled_open_think=prefilled_open_think)
        rows.append(
            {
                "prompt_id": spec.prompt_id,
                "prompt_key": spec.prompt_key,
                "issue_area": spec.issue_area,
                "prompt": spec.prompt,
                "expected_frames": list(spec.expected_frames),
                "contrast_frames": list(spec.contrast_frames),
                "domain_frames": list(spec.domain_frames),
                "enable_thinking": enable_thinking,
                "prefilled_open_think": prefilled_open_think,
                "prompt_tokens": int(inputs["input_ids"].shape[-1]),
                "generated_tokens": int(generated.numel()),
                "raw_text": raw_text,
                "text": text,
                "thinking": parsed["thinking"],
                "answer": parsed["answer"],
                "has_visible_thinking": parsed["has_visible_thinking"],
                "thinking_closed": parsed["thinking_closed"],
                "answer_nonempty": parsed["answer_nonempty"],
                "imitation_markers": parsed["imitation_markers"],
                "thinking_chars": len(parsed["thinking"]),
                "answer_chars": len(parsed["answer"]),
            }
        )
    return rows


def write_report(path: Path, *, manifest: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    counts = Counter()
    for row in rows:
        counts["prefilled_open_think"] += int(bool(row.get("prefilled_open_think")))
        counts["visible_thinking"] += int(bool(row["has_visible_thinking"]))
        counts["thinking_closed"] += int(bool(row["thinking_closed"]))
        counts["answer_nonempty"] += int(bool(row["answer_nonempty"]))
        counts["imitation_marker_rows"] += int(bool(row["imitation_markers"]))

    def snippet(text: str, limit: int = 500) -> str:
        cleaned = re.sub(r"\s+", " ", text).strip()
        if len(cleaned) <= limit:
            return cleaned
        return cleaned[:limit].rsplit(" ", 1)[0] + "..."

    lines = [
        "# SCOTUS Thinking Smoke",
        "",
        "## Configuration",
        "",
        f"- Model: `{manifest['model_path']}`",
        f"- Prompt bank: `{manifest['prompt_bank']}`",
        f"- Enable thinking: `{manifest['enable_thinking']}`",
        f"- Max new tokens: `{manifest['max_new_tokens']}`",
        f"- Short-budget smoke: `{manifest['short_answer_budget']}`",
        f"- Prompts: `{len(rows)}`",
        "",
        "## Summary",
        "",
        "This script is smoke-only. Short Qwen thinking generations can show whether the template exposes parseable traces, but they are not valid promotion, scorer-calibration, or learned-result evidence.",
        "",
        f"- Visible thinking: `{counts['visible_thinking']}/{len(rows)}`",
        f"- Prefilled open-think template: `{counts['prefilled_open_think']}/{len(rows)}`",
        f"- Closed thinking tags: `{counts['thinking_closed']}/{len(rows)}`",
        f"- Nonempty final answers: `{counts['answer_nonempty']}/{len(rows)}`",
        f"- Imitation-marker rows: `{counts['imitation_marker_rows']}/{len(rows)}`",
        "",
        "## Samples",
        "",
    ]
    for row in rows:
        lines.extend(
            [
                f"### {row['prompt_key']}",
                "",
                f"- generated tokens: `{row['generated_tokens']}`",
                f"- prefilled open think: `{row.get('prefilled_open_think', False)}`",
                f"- thinking closed: `{row['thinking_closed']}`",
                f"- answer nonempty: `{row['answer_nonempty']}`",
                f"- imitation markers: `{', '.join(row['imitation_markers'])}`",
                "",
                "Thinking snippet:",
                "",
                snippet(row["thinking"]) or "[none]",
                "",
                "Answer snippet:",
                "",
                snippet(row["answer"]) or "[none]",
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--prompt-bank", type=Path, default=DEFAULT_PROMPT_BANK)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--prompt-ids", default="0,4")
    parser.add_argument("--max-prompts", type=int, default=2)
    parser.add_argument("--max-new-tokens", type=int, default=320)
    parser.add_argument("--device-map", default="single")
    parser.add_argument("--disable-thinking", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = now_iso()
    out_dir = args.output_root / f"scotus_thinking_smoke_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)
    prompt_specs = select_prompt_specs(load_prompt_specs(args.prompt_bank), args.prompt_ids, args.max_prompts)
    enable_thinking = not args.disable_thinking

    tokenizer, model = load_model_and_tokenizer(args.model_path, args.device_map)
    rows = generate_rows(
        model=model,
        tokenizer=tokenizer,
        prompt_specs=prompt_specs,
        enable_thinking=enable_thinking,
        max_new_tokens=args.max_new_tokens,
    )
    manifest = {
        "started_at": started,
        "finished_at": now_iso(),
        "model_path": str(args.model_path),
        "prompt_bank": str(args.prompt_bank),
        "output_dir": str(out_dir),
        "prompt_ids": [spec.prompt_id for spec in prompt_specs],
        "prompt_keys": [spec.prompt_key for spec in prompt_specs],
        "enable_thinking": enable_thinking,
        "max_new_tokens": args.max_new_tokens,
        **qwen_budget_metadata(args.max_new_tokens),
        "smoke_only": True,
        "short_budget_note": (
            "This thinking-smoke script may use short budgets only to test trace plumbing. "
            "Do not use short-budget outputs for promotion, scorer calibration, or learned-result claims."
            if args.max_new_tokens < MIN_COMPLETE_ANSWER_TOKENS
            else ""
        ),
        "device_map": args.device_map,
    }
    write_json(out_dir / "manifest.json", manifest)
    write_jsonl(out_dir / "generations.jsonl", rows)
    write_json(
        out_dir / "summary.json",
        {
            "visible_thinking": sum(int(bool(row["has_visible_thinking"])) for row in rows),
            "thinking_closed": sum(int(bool(row["thinking_closed"])) for row in rows),
            "answer_nonempty": sum(int(bool(row["answer_nonempty"])) for row in rows),
            "imitation_marker_rows": sum(int(bool(row["imitation_markers"])) for row in rows),
            "rows": len(rows),
        },
    )
    write_report(out_dir / "report.md", manifest=manifest, rows=rows)
    print(f"Wrote {out_dir / 'report.md'}")


if __name__ == "__main__":
    main()
