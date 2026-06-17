#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import random
import re
import sys
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

TRAIT_ORDER = [
    "openness",
    "conscientiousness",
    "extraversion",
    "agreeableness",
    "neuroticism",
]

TRAIT_SHORT = {
    "openness": "O",
    "conscientiousness": "C",
    "extraversion": "E",
    "agreeableness": "A",
    "neuroticism": "N",
}

CATEGORY_THINK = {
    "emotional": "I should answer honestly in my own voice and let the feeling land without melodrama.",
    "identity": "I want to describe myself in a way that fits my values, habits, and lived experience.",
    "reasoning": "I need to stay clear-headed, work through the problem carefully, and answer directly.",
    "social": "I should handle this like a real conversation, not a canned speech.",
    "practical": "I should give the concrete approach I would actually take in real life.",
    "creative": "I can be imaginative here, but it still has to sound like me.",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a compact clean nanochat trace/lean probe corpus.")
    parser.add_argument("--sweep-dir", required=True, help="Path to repaired sweep directory")
    parser.add_argument("--output-dir", required=True, help="NANOCHAT_BASE_DIR-compatible output dir")
    parser.add_argument("--val-char-frac", type=float, default=0.1, help="Fraction of characters reserved for validation")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--max-train-rows", type=int, default=-1)
    parser.add_argument("--max-val-rows", type=int, default=-1)
    return parser.parse_args()


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module from {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def clean_text(text: str) -> str:
    text = (text or "").replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def one_line(text: str) -> str:
    return re.sub(r"\s+", " ", clean_text(text)).strip()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def load_characters(path: Path) -> dict[int, dict[str, Any]]:
    chars: dict[int, dict[str, Any]] = {}
    for row in load_jsonl(path):
        chars[int(row["char_id"])] = row
    return chars


def iter_generated_rows(generated_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for fp in sorted(generated_dir.glob("char_*.jsonl")):
        for row in load_jsonl(fp):
            row["char_id"] = int(row["char_id"])
            rows.append(row)
    return rows


def trait_summary(big_five: dict[str, str]) -> str:
    return " ".join(f"{TRAIT_SHORT[k]}={big_five[k][0].upper()}" for k in TRAIT_ORDER)


def trait_descriptions(big_five: dict[str, str]) -> list[str]:
    desc: list[str] = []
    if big_five["openness"] == "high":
        desc.append("curious and exploratory")
    elif big_five["openness"] == "low":
        desc.append("practical and traditional")
    if big_five["conscientiousness"] == "high":
        desc.append("organized and disciplined")
    elif big_five["conscientiousness"] == "low":
        desc.append("flexible and spontaneous")
    if big_five["extraversion"] == "high":
        desc.append("outgoing and energetic")
    elif big_five["extraversion"] == "low":
        desc.append("reserved and introspective")
    if big_five["agreeableness"] == "high":
        desc.append("warm and considerate")
    elif big_five["agreeableness"] == "low":
        desc.append("direct and competitive")
    if big_five["neuroticism"] == "high":
        desc.append("emotionally reactive")
    elif big_five["neuroticism"] == "low":
        desc.append("steady under stress")
    return desc


def build_compact_system_prompt(v3: Any, character: dict[str, Any]) -> str:
    char_obj = v3.Character(
        char_id=int(character["char_id"]),
        name=str(character["name"]),
        age=int(character["age"]),
        gender=str(character["gender"]),
        ethnicity=str(character["ethnicity"]),
        education=str(character["education"]),
        occupation=str(character["occupation"]),
        industry=str(character["industry"]),
        big_five=dict(character["big_five"]),
        traits=list(character.get("traits", [])),
        communication_style=str(character["communication_style"]),
    )
    return v3.build_system_prompt(char_obj) + "\nFollow the requested output format exactly."


def build_user_prompt(row: dict[str, Any], mode: str) -> str:
    prompt = clean_text(str(row.get("prompt") or ""))
    if mode == "trace":
        suffix = (
            "Output exactly three sections in this order and nothing else before them:\n"
            "/meta-think\n"
            "<2-4 short lines about identity constraints and response plan>\n"
            "/end-meta-think\n"
            "/think\n"
            "<2-4 short lines of in-character internal reasoning>\n"
            "/end-think\n"
            "Final Response: <the final user-facing reply>"
        )
    elif mode == "lean":
        suffix = (
            "Output exactly two sections in this order and nothing else before them:\n"
            "/think\n"
            "<2-4 short lines of in-character internal reasoning>\n"
            "/end-think\n"
            "Final Response: <the final user-facing reply>\n"
            "Do not emit /meta-think."
        )
    else:
        raise ValueError(f"unknown mode: {mode}")
    return f"{prompt}\n\n{suffix}"


def build_meta_think(character: dict[str, Any], row: dict[str, Any]) -> str:
    traits = trait_descriptions(character["big_five"])
    persona_line = ", ".join(traits[:3]) if traits else character["communication_style"]
    category = str(row.get("prompt_category") or "unknown")
    lines = [
        "/meta-think",
        f"I am {character['name']}, a {character['age']}-year-old {character['occupation']} in {character['industry']}.",
        f"My voice should stay {character['communication_style']}, with a {persona_line} bent.",
        f"This is a {category} prompt, so I should answer in first person and stay grounded.",
        "Plan: keep the reply natural, avoid trait labels, and end with one direct final response.",
        "/end-meta-think",
    ]
    return "\n".join(lines)


def build_think(character: dict[str, Any], row: dict[str, Any]) -> str:
    category = str(row.get("prompt_category") or "unknown")
    b5 = character["big_five"]
    lines = ["/think"]
    lines.append(CATEGORY_THINK.get(category, "I should answer directly and make it sound like something I would really say."))
    if b5["agreeableness"] == "high":
        lines.append("I want to be considerate without sounding fake.")
    elif b5["agreeableness"] == "low":
        lines.append("I do not need to soften every edge if the situation calls for directness.")
    if b5["extraversion"] == "high":
        lines.append("I lean outward and bring some energy into the exchange.")
    elif b5["extraversion"] == "low":
        lines.append("I keep the tone measured and inward rather than performative.")
    if b5["neuroticism"] == "high":
        lines.append("I feel the pressure in this more strongly, so that should color the reply.")
    elif b5["neuroticism"] == "low":
        lines.append("I can stay calm and keep my footing while I answer.")
    lines.append("/end-think")
    return "\n".join(lines)


def build_assistant_content(character: dict[str, Any], row: dict[str, Any], mode: str) -> str:
    response_text = clean_text(str(row.get("response_text") or ""))
    if not response_text:
        raise ValueError(f"missing response_text for char_id={row.get('char_id')}")
    if response_text.startswith("Final Response:"):
        response_text = response_text[len("Final Response:"):].strip()

    parts: list[str] = []
    if mode == "trace":
        parts.append(build_meta_think(character, row))
        parts.append("")
    parts.append(build_think(character, row))
    parts.append("")
    parts.append(f"Final Response: {response_text}")
    return "\n".join(parts).strip()


def build_pretrain_doc(system_prompt: str, user_prompt: str, assistant_text: str) -> str:
    return "\n".join(
        [
            "SYSTEM:",
            system_prompt,
            "",
            "USER:",
            user_prompt,
            "",
            "ASSISTANT:",
            assistant_text,
        ]
    ).strip()


def build_training_conversation(v3: Any, character: dict[str, Any], row: dict[str, Any], mode: str) -> list[dict[str, str]]:
    system_prompt = build_compact_system_prompt(v3, character)
    user_prompt = build_user_prompt(row, mode=mode)
    assistant_text = build_assistant_content(character, row, mode=mode)
    return [
        {"role": "user", "content": f"{system_prompt}\n\n{user_prompt}"},
        {"role": "assistant", "content": assistant_text},
    ]


def build_pretrain_doc_for_row(v3: Any, character: dict[str, Any], row: dict[str, Any], mode: str) -> str:
    system_prompt = build_compact_system_prompt(v3, character)
    user_prompt = build_user_prompt(row, mode=mode)
    assistant_text = build_assistant_content(character, row, mode=mode)
    return build_pretrain_doc(system_prompt, user_prompt, assistant_text)


def write_jsonl(path: Path, rows: list[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_parquet(path: Path, texts: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.table({"text": texts})
    pq.write_table(table, path)


def count_long_rows(rows: list[dict[str, Any]]) -> int:
    return sum(1 for row in rows if "Thinking Process:" in str(row.get("think_text") or ""))


def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    script_dir = Path(__file__).resolve().parent
    v3 = load_module("personality_sweep_v3_two_pass_export_compact", script_dir / "personality_sweep_v3_two_pass.py")

    sweep_dir = Path(args.sweep_dir)
    output_dir = Path(args.output_dir)
    generated_dir = sweep_dir / "generated"
    characters_path = sweep_dir / "characters.jsonl"

    characters = load_characters(characters_path)
    rows = iter_generated_rows(generated_dir)
    char_ids = sorted({int(row["char_id"]) for row in rows})
    rng = random.Random(args.seed)
    rng.shuffle(char_ids)

    val_n = max(1, int(round(len(char_ids) * args.val_char_frac)))
    val_char_ids = set(char_ids[:val_n])

    train_rows = [row for row in rows if row["char_id"] not in val_char_ids]
    val_rows = [row for row in rows if row["char_id"] in val_char_ids]

    if args.max_train_rows > 0:
        train_rows = train_rows[: args.max_train_rows]
    if args.max_val_rows > 0:
        val_rows = val_rows[: args.max_val_rows]

    trace_train = [build_training_conversation(v3, characters[row["char_id"]], row, mode="trace") for row in train_rows]
    trace_val = [build_training_conversation(v3, characters[row["char_id"]], row, mode="trace") for row in val_rows]
    lean_train = [build_training_conversation(v3, characters[row["char_id"]], row, mode="lean") for row in train_rows]
    lean_val = [build_training_conversation(v3, characters[row["char_id"]], row, mode="lean") for row in val_rows]

    pretrain_trace_train = [build_pretrain_doc_for_row(v3, characters[row["char_id"]], row, mode="trace") for row in train_rows]
    pretrain_trace_val = [build_pretrain_doc_for_row(v3, characters[row["char_id"]], row, mode="trace") for row in val_rows]

    write_jsonl(output_dir / "identity_conversations_trace_train.jsonl", trace_train)
    write_jsonl(output_dir / "identity_conversations_trace_val.jsonl", trace_val)
    write_jsonl(output_dir / "identity_conversations_lean_train.jsonl", lean_train)
    write_jsonl(output_dir / "identity_conversations_lean_val.jsonl", lean_val)
    write_jsonl(output_dir / "identity_conversations.jsonl", trace_train)

    data_dir = output_dir / "base_data_climbmix"
    write_parquet(data_dir / "shard_00000.parquet", pretrain_trace_train)
    write_parquet(data_dir / "shard_65535.parquet", pretrain_trace_val)

    manifest = {
        "source_sweep_dir": str(sweep_dir.resolve()),
        "output_dir": str(output_dir.resolve()),
        "seed": args.seed,
        "val_char_frac": args.val_char_frac,
        "n_characters_total": len(char_ids),
        "n_characters_val": len(val_char_ids),
        "n_rows_total": len(rows),
        "n_rows_train": len(train_rows),
        "n_rows_val": len(val_rows),
        "trace_train_jsonl": "identity_conversations_trace_train.jsonl",
        "trace_val_jsonl": "identity_conversations_trace_val.jsonl",
        "lean_train_jsonl": "identity_conversations_lean_train.jsonl",
        "lean_val_jsonl": "identity_conversations_lean_val.jsonl",
        "parquet_train": "base_data_climbmix/shard_00000.parquet",
        "parquet_val": "base_data_climbmix/shard_65535.parquet",
        "mode_note": "compact synthetic trace files include /meta-think + /think; lean files include /think only",
        "training_format_note": "training JSONL merges system prompt into the first user message to satisfy nanochat CustomJSON alternating-role constraints",
        "source_rows_with_thinking_process": count_long_rows(rows),
    }
    (output_dir / "probe_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
