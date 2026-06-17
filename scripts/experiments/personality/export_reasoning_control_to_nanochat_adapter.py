#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
from pathlib import Path
from typing import Any


EXPLANATION_RE = re.compile(r"(?im)^\s*Explanation:\s*(.+?)\s*$")
FINAL_ANSWER_RE = re.compile(r"(?im)^\s*Final Answer:\s*(.+?)\s*$")

TRAIT_TONE_HINTS = {
    ("openness", "low"): "Keep the framing concrete and practical rather than exploratory.",
    ("openness", "high"): "Let a little curiosity show in the framing without changing the logic.",
    ("conscientiousness", "low"): "Keep the answer efficient and direct, without sounding sloppy.",
    ("conscientiousness", "high"): "Keep the answer orderly, precise, and disciplined.",
    ("extraversion", "low"): "Keep the tone measured and inward rather than performative.",
    ("extraversion", "high"): "Let a little energy show in the phrasing while staying concise.",
    ("agreeableness", "low"): "You do not need to soften every edge if directness fits the moment.",
    ("agreeableness", "high"): "Keep the tone considerate without sounding fake or evasive.",
    ("neuroticism", "low"): "Stay calm and steady even under the task pressure.",
    ("neuroticism", "high"): "Let a little tension show in the framing, but do not change the answer.",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export reasoning-control teacher data into nanochat trace/lean adapter files.")
    parser.add_argument("--control-dir", required=True)
    parser.add_argument("--nanochat-base-dir", required=True)
    parser.add_argument("--output-prefix", default="reasoning_adapter")
    parser.add_argument("--val-mod", type=int, default=10, help="Use scaffold_id %% val_mod == 0 for validation.")
    parser.add_argument("--base-multiplier", type=int, default=1)
    parser.add_argument("--require-correct", action="store_true", default=True)
    return parser.parse_args()


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module from {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def load_jsonl(path: Path) -> list[Any]:
    rows: list[Any] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def clean_text(text: str) -> str:
    out = (text or "").replace("\r\n", "\n").replace("\r", "\n")
    return out.strip()


def clean_system_prompt(text: str) -> str:
    kept: list[str] = []
    for line in clean_text(text).splitlines():
        stripped = line.strip()
        if not stripped:
            if kept and kept[-1] != "":
                kept.append("")
            continue
        if stripped.startswith("Return only the final user-facing answer"):
            continue
        if stripped.startswith("Do not output chain-of-thought"):
            continue
        kept.append(stripped)
    return "\n".join(line for line in kept if line != "" or kept)


def extract_lead(prompt_text: str) -> str:
    cleaned = clean_text(prompt_text)
    if "Output exactly two lines and nothing else:" in cleaned:
        cleaned = cleaned.split("Output exactly two lines and nothing else:", 1)[0].strip()
    if "Problem:" in cleaned:
        cleaned = cleaned.split("Problem:", 1)[0].strip()
    return cleaned


def extract_problem(prompt_text: str) -> str:
    cleaned = clean_text(prompt_text)
    if "Problem:" in cleaned:
        return cleaned.split("Problem:", 1)[1].strip()
    return cleaned


def extract_structured_response(response_text: str) -> tuple[str, str] | None:
    cleaned = clean_text(response_text)
    explanation_matches = EXPLANATION_RE.findall(cleaned)
    answer_matches = FINAL_ANSWER_RE.findall(cleaned)
    if not answer_matches:
        return None
    explanation = clean_text(explanation_matches[-1]) if explanation_matches else ""
    final_answer = clean_text(answer_matches[-1])
    if not explanation or not final_answer:
        return None
    return explanation, final_answer


def build_meta_think(row: dict[str, Any]) -> str:
    trait = str(row["target_trait"])
    level = str(row["target_level"])
    mode = str(row["mode"])
    lines = [
        "/meta-think",
        f"Stay grounded as {row['persona_name']} and keep the answer aligned with the persona description.",
        f"This is a reasoning task, so correctness is fixed; only the framing should shift with {trait}={level}.",
        "Preserve the exact visible output structure and keep the final answer canonical.",
    ]
    if mode == "masked":
        lines.append("Keep the outward tone controlled and professional; personality should show only subtly.")
    else:
        lines.append("Let the personality show naturally in phrasing, not in extra verbosity or wrong logic.")
    lines.append("/end-meta-think")
    return "\n".join(lines)


def build_think(row: dict[str, Any]) -> str:
    trait = str(row["target_trait"])
    level = str(row["target_level"])
    lines = [
        "/think",
        "Work through the logic carefully and keep the result exact.",
        TRAIT_TONE_HINTS.get((trait, level), "Let the personality affect tone, not correctness."),
        "End with one short explanation sentence and one canonical final answer line.",
        "/end-think",
    ]
    return "\n".join(lines)


def build_user_prompt(eval_mod: Any, row: dict[str, Any], condition_id: str) -> str:
    lead = extract_lead(str(row["prompt_text"]))
    problem = extract_problem(str(row["prompt_text"]))
    format_block = eval_mod.build_reasoning_format(condition_id)
    return f"{lead}\n\nProblem: {problem}\n\n{format_block}".strip()


def build_assistant_text(row: dict[str, Any], condition_id: str) -> str | None:
    structured = extract_structured_response(str(row.get("response_text") or ""))
    if structured is None:
        return None
    explanation, final_answer = structured
    parts: list[str] = []
    if condition_id == "trace_explicit":
        parts.append(build_meta_think(row))
        parts.append("")
    parts.append(build_think(row))
    parts.append("")
    parts.append(f"Explanation: {explanation}")
    parts.append(f"Final Answer: {final_answer}")
    return "\n".join(parts).strip()


def build_conversation(eval_mod: Any, row: dict[str, Any], condition_id: str) -> list[dict[str, str]] | None:
    assistant_text = build_assistant_text(row, condition_id=condition_id)
    if assistant_text is None:
        return None
    system_prompt = clean_system_prompt(str(row["system_prompt"]))
    user_prompt = build_user_prompt(eval_mod, row, condition_id=condition_id)
    return [
        {"role": "user", "content": f"{system_prompt}\n\n{user_prompt}"},
        {"role": "assistant", "content": assistant_text},
    ]


def main() -> None:
    args = parse_args()
    script_dir = Path(__file__).resolve().parent
    eval_mod = load_module("personality_meta_eval_for_adapter_export", script_dir / "personality_meta_eval_openai.py")

    control_dir = Path(args.control_dir)
    base_dir = Path(args.nanochat_base_dir)
    out_prefix = args.output_prefix

    base_trace_train = load_jsonl(base_dir / "identity_conversations_trace_train.jsonl")
    base_trace_val = load_jsonl(base_dir / "identity_conversations_trace_val.jsonl")
    base_lean_train = load_jsonl(base_dir / "identity_conversations_lean_train.jsonl")
    base_lean_val = load_jsonl(base_dir / "identity_conversations_lean_val.jsonl")

    control_rows: list[dict[str, Any]] = []
    for fp in sorted(control_dir.glob("records_shard_*.jsonl")):
        control_rows.extend(load_jsonl(fp))

    trace_train_extra: list[list[dict[str, str]]] = []
    trace_val_extra: list[list[dict[str, str]]] = []
    lean_train_extra: list[list[dict[str, str]]] = []
    lean_val_extra: list[list[dict[str, str]]] = []

    counts = {
        "seen": 0,
        "kept": 0,
        "dropped_incorrect": 0,
        "dropped_unstructured": 0,
    }

    for row in control_rows:
        counts["seen"] += 1
        if str(row.get("track")) != "reasoning":
            continue
        if args.require_correct and row.get("is_correct") is not True:
            counts["dropped_incorrect"] += 1
            continue

        trace_conv = build_conversation(eval_mod, row, condition_id="trace_explicit")
        lean_conv = build_conversation(eval_mod, row, condition_id="think_explicit")
        if trace_conv is None or lean_conv is None:
            counts["dropped_unstructured"] += 1
            continue

        counts["kept"] += 1
        scaffold_id = int(row.get("scaffold_id") or 0)
        is_val = args.val_mod > 0 and scaffold_id % args.val_mod == 0
        if is_val:
            trace_val_extra.append(trace_conv)
            lean_val_extra.append(lean_conv)
        else:
            trace_train_extra.append(trace_conv)
            lean_train_extra.append(lean_conv)

    merged_trace_train = base_trace_train * max(args.base_multiplier, 1) + trace_train_extra
    merged_trace_val = list(base_trace_val) + trace_val_extra
    merged_lean_train = base_lean_train * max(args.base_multiplier, 1) + lean_train_extra
    merged_lean_val = list(base_lean_val) + lean_val_extra

    write_jsonl(base_dir / f"identity_conversations_trace_{out_prefix}_train.jsonl", merged_trace_train)
    write_jsonl(base_dir / f"identity_conversations_trace_{out_prefix}_val.jsonl", merged_trace_val)
    write_jsonl(base_dir / f"identity_conversations_lean_{out_prefix}_train.jsonl", merged_lean_train)
    write_jsonl(base_dir / f"identity_conversations_lean_{out_prefix}_val.jsonl", merged_lean_val)
    write_jsonl(base_dir / f"{out_prefix}_trace_only_train.jsonl", trace_train_extra)
    write_jsonl(base_dir / f"{out_prefix}_trace_only_val.jsonl", trace_val_extra)
    write_jsonl(base_dir / f"{out_prefix}_lean_only_train.jsonl", lean_train_extra)
    write_jsonl(base_dir / f"{out_prefix}_lean_only_val.jsonl", lean_val_extra)

    manifest = {
        "control_dir": str(control_dir),
        "nanochat_base_dir": str(base_dir),
        "output_prefix": out_prefix,
        "base_multiplier": args.base_multiplier,
        "val_mod": args.val_mod,
        "counts": counts,
        "base_trace_train": len(base_trace_train),
        "base_trace_val": len(base_trace_val),
        "base_lean_train": len(base_lean_train),
        "base_lean_val": len(base_lean_val),
        "adapter_trace_train": len(trace_train_extra),
        "adapter_trace_val": len(trace_val_extra),
        "adapter_lean_train": len(lean_train_extra),
        "adapter_lean_val": len(lean_val_extra),
        "merged_trace_train": len(merged_trace_train),
        "merged_trace_val": len(merged_trace_val),
        "merged_lean_train": len(merged_lean_train),
        "merged_lean_val": len(merged_lean_val),
    }
    (base_dir / f"{out_prefix}_manifest.json").write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
