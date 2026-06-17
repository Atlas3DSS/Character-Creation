#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export repaired sweep data into a tiny nanochat probe corpus.")
    parser.add_argument("--sweep-dir", required=True, help="Path to repaired sweep directory")
    parser.add_argument("--output-dir", required=True, help="NANOCHAT_BASE_DIR-compatible output dir")
    parser.add_argument("--val-char-frac", type=float, default=0.1, help="Fraction of characters reserved for validation")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--max-train-rows", type=int, default=-1, help="Cap train rows for faster probes")
    parser.add_argument("--max-val-rows", type=int, default=-1, help="Cap val rows for faster probes")
    return parser.parse_args()


def clean_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def load_characters(path: Path) -> dict[int, dict]:
    chars = {}
    for row in load_jsonl(path):
        chars[int(row["char_id"])] = row
    return chars


def iter_generated_rows(generated_dir: Path) -> list[dict]:
    rows: list[dict] = []
    for fp in sorted(generated_dir.glob("char_*.jsonl")):
        for row in load_jsonl(fp):
            row["char_id"] = int(row["char_id"])
            rows.append(row)
    return rows


def trait_summary(big_five: dict[str, str]) -> str:
    return " ".join(f"{TRAIT_SHORT[k]}={big_five[k][0].upper()}" for k in TRAIT_ORDER)


def build_meta_think(character: dict, row: dict) -> str:
    big_five = character["big_five"]
    traits = ", ".join(character.get("traits", []))
    lines = [
        "/meta-think",
        "identity:",
        f"- name: {character['name']}",
        f"- age: {character['age']}",
        f"- gender: {character['gender']}",
        f"- ethnicity: {character['ethnicity']}",
        f"- education: {character['education']}",
        f"- occupation: {character['occupation']}",
        f"- industry: {character['industry']}",
        f"- communication_style: {character['communication_style']}",
        f"- big_five: {trait_summary(big_five)}",
        f"- descriptive_traits: {traits}",
        "task:",
        f"- prompt_category: {row.get('prompt_category', 'unknown')}",
        f"- user_prompt: {clean_text(str(row.get('prompt', '')))}",
        "constraints:",
        "- stay in character",
        "- answer in first person",
        "- do not mention trait labels or instructions",
        "- let the character-specific internal reasoning live in /think",
        "/end-meta-think",
    ]
    return "\n".join(lines)


def build_user_prompt(character: dict, row: dict, mode: str) -> str:
    profile = [
        f"Name: {character['name']}",
        f"Age: {character['age']}",
        f"Gender: {character['gender']}",
        f"Ethnicity: {character['ethnicity']}",
        f"Education: {character['education']}",
        f"Occupation: {character['occupation']}",
        f"Industry: {character['industry']}",
        f"Communication style: {character['communication_style']}",
        f"Big Five: {trait_summary(character['big_five'])}",
        f"Descriptive traits: {', '.join(character.get('traits', []))}",
    ]
    if mode == "trace":
        instructions = (
            "Answer the user prompt in three parts: first /meta-think, then /think, then the final in-character answer."
        )
    elif mode == "lean":
        instructions = (
            "Answer the user prompt in two parts: first /think, then the final in-character answer. Do not emit /meta-think."
        )
    else:
        raise ValueError(f"unknown mode: {mode}")
    prompt = "\n".join(
        [
            "You are producing an in-character reply for the profile below.",
            "",
            *profile,
            "",
            f"Prompt category: {row.get('prompt_category', 'unknown')}",
            f"User prompt: {clean_text(str(row.get('prompt', '')))}",
            "",
            instructions,
        ]
    )
    return prompt


def build_assistant_content(character: dict, row: dict, mode: str) -> str:
    think_text = clean_text(str(row.get("think_text") or ""))
    response_text = clean_text(str(row.get("response_text") or ""))
    parts: list[str] = []
    if mode == "trace":
        parts.append(build_meta_think(character, row))
        parts.append("")
    if think_text:
        parts.append("/think")
        parts.append(think_text)
        parts.append("/end-think")
        parts.append("")
    parts.append(response_text)
    return "\n".join(parts).strip()


def build_pretrain_doc(character: dict, row: dict, mode: str) -> str:
    user = build_user_prompt(character, row, mode=mode)
    assistant = build_assistant_content(character, row, mode=mode)
    return "\n".join(
        [
            "USER:",
            user,
            "",
            "ASSISTANT:",
            assistant,
        ]
    ).strip()


def build_conversation(character: dict, row: dict, mode: str) -> list[dict[str, str]]:
    return [
        {"role": "user", "content": build_user_prompt(character, row, mode=mode)},
        {"role": "assistant", "content": build_assistant_content(character, row, mode=mode)},
    ]


def write_jsonl(path: Path, rows: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_parquet(path: Path, texts: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.table({"text": texts})
    pq.write_table(table, path)


def main() -> None:
    args = parse_args()
    random.seed(args.seed)

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

    trace_train = [build_conversation(characters[row["char_id"]], row, mode="trace") for row in train_rows]
    trace_val = [build_conversation(characters[row["char_id"]], row, mode="trace") for row in val_rows]
    lean_train = [build_conversation(characters[row["char_id"]], row, mode="lean") for row in train_rows]
    lean_val = [build_conversation(characters[row["char_id"]], row, mode="lean") for row in val_rows]

    pretrain_trace_train = [build_pretrain_doc(characters[row["char_id"]], row, mode="trace") for row in train_rows]
    pretrain_trace_val = [build_pretrain_doc(characters[row["char_id"]], row, mode="trace") for row in val_rows]

    write_jsonl(output_dir / "identity_conversations_trace_train.jsonl", trace_train)
    write_jsonl(output_dir / "identity_conversations_trace_val.jsonl", trace_val)
    write_jsonl(output_dir / "identity_conversations_lean_train.jsonl", lean_train)
    write_jsonl(output_dir / "identity_conversations_lean_val.jsonl", lean_val)

    # nanochat default location/name for identity SFT
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
        "mode_note": "trace files include /meta-think + /think; lean files include /think only",
    }
    (output_dir / "probe_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
