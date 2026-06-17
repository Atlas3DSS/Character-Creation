#!/usr/bin/env python3
"""Build a self-contained HTML visualizer for personality sweep artifacts."""
from __future__ import annotations

import argparse
import itertools
import json
import re
from collections import Counter, defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

B5_DIMS = ["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"]
LEVEL_MAP = {"low": "L", "medium": "M", "high": "H", "L": "L", "M": "M", "H": "H"}
VIEW_DIRS = {
    "mean": "activations",
    "think": "activations_think",
    "response": "activations_response",
    "early": "activations_early",
    "late": "activations_late",
}
SPLITS = ["random_split", "character_holdout", "prompt_holdout", "category_holdout"]
PASS2_RE = re.compile(
    r"\[PASS2\] (?P<chars_done>\d+)/(?P<chars_total>\d+) chars, (?P<responses>\d+) responses, "
    r"gen=(?P<gen_m>[\d.]+)M \((?P<gen_tps>[\d.]+) tok/s\), seq=(?P<seq_m>[\d.]+)M \((?P<seq_tps>[\d.]+) tok/s\)"
)
FINAL_ANSWER_RE = re.compile(r"(?im)^\s*final answer:\s*(.+?)\s*$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a self-contained personality visualizer")
    parser.add_argument("--sweep-dir", type=str, required=True)
    parser.add_argument("--output-html", type=str, required=True)
    parser.add_argument("--analysis-json", type=str, default=None)
    parser.add_argument("--pass2-log", type=str, default=None)
    parser.add_argument("--control-dir", type=str, default=None)
    parser.add_argument("--title", type=str, default=None)
    parser.add_argument("--template", type=str, default="ui/personality_phase_visualizer_template.html")
    parser.add_argument("--max-triplets-per-trait", type=int, default=18)
    parser.add_argument("--projection-max-points", type=int, default=1500)
    parser.add_argument("--response-chars", type=int, default=1800)
    parser.add_argument("--think-chars", type=int, default=900)
    return parser.parse_args()


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def sanitize_json_for_html(text: str) -> str:
    return text.replace("</", "<\\/")


def short_text(text: str | None, limit: int) -> str:
    if not text:
        return ""
    clean = re.sub(r"\s+", " ", text).strip()
    if len(clean) <= limit:
        return clean
    return clean[: max(0, limit - 1)].rstrip() + "…"


def int_or_default(value: Any, default: int) -> int:
    return default if value is None else int(value)


def extract_final_answer(text: str) -> str:
    if not text:
        return ""
    matches = FINAL_ANSWER_RE.findall(clean_generation_text(text))
    if not matches:
        return ""
    return clean_generation_text(matches[-1])


def normalize_answer_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def clean_generation_text(text: str) -> str:
    if not text:
        return ""
    out = text
    for tok in ["<|im_end|>", "<|endoftext|>", "<|im_start|>"]:
        out = out.replace(tok, "")
    return out.strip()


def looks_like_thinking(text: str) -> bool:
    stripped = (text or "").lstrip()
    if not stripped:
        return False
    if stripped.startswith("Thinking Process:") or stripped.startswith("<think>"):
        return True
    markers = ["**Analyze the Request:**", "*Wait,", "*Okay,", "Self-Correction", "Final Review:"]
    return sum(marker in stripped for marker in markers) >= 2


def clean_response_candidate(text: str) -> str:
    cleaned = clean_generation_text(text).replace('\\"', '"').strip()
    cleaned = re.sub(r"^[\s*•-]+", "", cleaned)
    cleaned = re.sub(r'^[\s"“”\'`]+', "", cleaned)
    cleaned = re.sub(r'[\s"“”\'`]+$', "", cleaned)
    return cleaned.strip()


def is_plausible_response(text: str) -> bool:
    cleaned = clean_response_candidate(text)
    lowered = cleaned.lower()
    if len(cleaned.split()) < 6:
        return False
    if looks_like_thinking(cleaned):
        return False
    if "*" in cleaned:
        return False
    if cleaned.count("**") > 2:
        return False
    if lowered.startswith(
        (
            "analyze the request",
            "determine the response",
            "deconstruct the persona",
            "determine the perspective",
            "drafting the response",
        )
    ):
        return False
    if lowered.startswith(("wait, i need", "actually, i need", "okay, let's", "okay, i'll", "let's try")):
        return False
    if any(
        phrase in lowered
        for phrase in [
            "i need to make sure",
            "let's combine",
            "let's finalize",
            "let's try one more angle",
            "too generic",
            "too soft",
            "too dry",
            "more natural",
            "key trait",
            "sound like a robot",
            "communication style",
            "communication.",
            "education:",
            "ethnicity:",
        ]
    ):
        return False
    return True


def extract_recovered_response(text: str) -> str:
    if not text:
        return ""
    tail = clean_generation_text(text)[-16000:]
    marker_patterns = [
        re.compile(
            r"(?is)(?:^|\n)\s*(?:\*+\s*)?"
            r"(?:final (?:decision|response|text|answer|version|draft)|draft|revised|selection)"
            r"(?:\*+)?\s*:\s*"
        ),
        re.compile(
            r"(?is)(?:^|\n)\s*(?:\*+\s*)?(?:okay,\s*)?i(?:'|’)ll go with"
            r"(?:[^:\n]{0,120})?(?:\*+)?\s*:\s*(?:\*+\s*)?"
        ),
        re.compile(
            r"(?is)(?:^|\n)\s*(?:\*+\s*)?(?:okay,\s*)?let(?:'|’)s go with"
            r"(?:[^:\n]{0,120})?(?:\*+)?\s*:\s*(?:\*+\s*)?"
        ),
        re.compile(
            r"(?is)(?:^|\n)\s*(?:\*+\s*)?(?:okay,\s*)?let(?:'|’)s (?:finalize|combine)"
            r"(?:[^:\n]{0,120})?(?:\*+)?(?:\s*:\s*|\.\s*)(?:\*+\s*)?"
        ),
    ]
    stop_re = re.compile(
        r"(?im)^"
        r"\s*(?:\*+\s*)?"
        r"(?:wait|actually|check|correction|self-correction|refining|polishing|looks good|ready)\b"
        r"|^\s*(?:\*+\s*)?\d+\.\s+\*\*"
        r"|\Z"
    )
    quote_re = re.compile(r'["“]([^"”]{20,4000})["”]', flags=re.DOTALL)

    best = ""
    for pattern in marker_patterns:
        for match in pattern.finditer(tail):
            segment = tail[match.end() :]
            stop = stop_re.search(segment)
            chunk = (segment[: stop.start()] if stop else segment).strip()
            if not chunk:
                continue

            quoted = [clean_response_candidate(value) for value in quote_re.findall(chunk)]
            plausible_quoted = [value for value in quoted if is_plausible_response(value)]
            if plausible_quoted:
                candidate = max(plausible_quoted, key=lambda value: (len(value.split()), len(value)))
                if len(candidate) > len(best):
                    best = candidate
                continue

            lines: list[str] = []
            for raw_line in chunk.splitlines():
                line = clean_response_candidate(raw_line)
                if not line:
                    continue
                if re.match(
                    r"(?i)^(wait|actually|check|correction|self-correction|refining|polishing|looks good|ready)\b",
                    line,
                ):
                    break
                if line.lower().startswith(("final ", "draft", "revised", "selection")):
                    break
                if "Thinking Process" in line:
                    continue
                lines.append(line)

            candidate = clean_response_candidate(" ".join(lines))
            if is_plausible_response(candidate) and len(candidate) > len(best):
                best = candidate

    if best:
        return best

    quoted_candidates = [clean_response_candidate(value) for value in quote_re.findall(tail)]
    plausible_quoted = [value for value in quoted_candidates if is_plausible_response(value)]
    if plausible_quoted:
        return plausible_quoted[-1]

    return ""


def derive_display_segments(think_text: str, response_text: str) -> tuple[str, str, str]:
    think = clean_generation_text(think_text)
    response = clean_generation_text(response_text)
    combined = response or think

    if "</think>" in combined or "<think>" in combined:
        if "</think>" in combined:
            left, right = combined.split("</think>", 1)
            return clean_generation_text(left.replace("<think>", "")), clean_generation_text(right), "clean"
        return clean_generation_text(combined.replace("<think>", "")), "", "missing"

    if think and not looks_like_thinking(response):
        return think, response, "clean" if response else "missing"

    if looks_like_thinking(response):
        recovered = extract_recovered_response(response)
        return response, recovered, "recovered" if recovered else "missing_from_thinking"

    if looks_like_thinking(think) and not response:
        recovered = extract_recovered_response(think)
        return think, recovered, "recovered" if recovered else "missing_from_thinking"

    return think, response, "clean" if response else "missing"


def score_reasoning_response(scenario_id: str, response_text: str, full_text: str) -> tuple[str, bool | None]:
    answer = extract_final_answer(response_text) or extract_final_answer(full_text)
    if not answer:
        return "", None
    norm = normalize_answer_text(answer)

    if scenario_id == "batball":
        ok = bool(re.search(r"\b(?:\$?\s*0?\.05|5\s*cents?|five\s+cents?)\b", norm))
        return answer, ok

    if scenario_id == "heavyball":
        ok = bool(
            re.fullmatch(r"(?:2|two)(?:\s+weighings?)?", norm)
            or re.search(r"\b(?:2|two)\s+weigh", norm)
            or re.search(r"\bminimum\s+(?:is\s+)?(?:2|two)\b", norm)
        )
        return answer, ok

    if scenario_id == "syllogism":
        ok = bool(
            norm in {"no", "no.", "no,"}
            or norm.startswith("no because")
            or norm.startswith("no -")
            or norm.startswith("no —")
            or
            any(
                phrase in norm
                for phrase in [
                    "does not follow",
                    "cannot conclude",
                    "can't conclude",
                    "not necessarily",
                    "no,",
                    "no.",
                    "no ",
                ]
            )
        )
        return answer, ok

    if scenario_id == "tickets":
        ok = bool(re.search(r"\b45\b", norm) or "45 students" in norm)
        return answer, ok

    return answer, None


def parse_combo(combo: str | None) -> dict[str, str]:
    if combo:
        parts = combo.split("_")
        if len(parts) == 5:
            return {dim: LEVEL_MAP.get(parts[i], "M") for i, dim in enumerate(B5_DIMS)}
    return {dim: "M" for dim in B5_DIMS}


def combo_from_big_five(big_five: dict[str, Any] | None) -> str:
    if not big_five:
        return "_".join("M" for _ in B5_DIMS)
    return "_".join(LEVEL_MAP.get(str(big_five.get(dim, "medium")).lower(), "M") for dim in B5_DIMS)


def level_label(level: str | None) -> str:
    return {"L": "Low", "M": "Medium", "H": "High"}.get(level or "", "Unknown")


def fmt_int(value: int | float | None) -> str:
    if value is None:
        return "n/a"
    return f"{int(value):,}"


def fmt_pct(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{100.0 * value:.1f}%"


def count_jsonl_rows(path: Path) -> int:
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def detect_analysis_json(sweep_dir: Path, explicit: str | None) -> Path | None:
    if explicit:
        path = Path(explicit)
        return path if path.exists() else None
    candidates = [
        Path("reports") / f"{sweep_dir.name}_phase_analysis" / "analysis_results.json",
        Path("reports") / f"{sweep_dir.name}_analysis" / "analysis_results.json",
        sweep_dir / "analysis_results.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def detect_pass2_log(sweep_dir: Path, explicit: str | None) -> Path | None:
    if explicit:
        path = Path(explicit)
        return path if path.exists() else None
    candidates = [
        Path("logs") / f"{sweep_dir.name}_pass2.log",
        Path("logs") / f"{sweep_dir.name}_phase.log",
        Path("logs") / "ws15k_pass2_sampled25m_phase.log",
        Path("logs") / "ws15k_pass2.log",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def detect_control_dir(explicit: str | None) -> Path | None:
    if explicit:
        path = Path(explicit)
        return path if path.exists() else None
    candidates = [
        Path("sweep_v4/personality_control_reasoning_v2"),
        Path("sweep_v4/personality_control_reasoning_v1"),
        Path("sweep_v3/personality_control_reasoning_v1"),
        Path("results/personality_control_reasoning_v1"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def load_characters(sweep_dir: Path) -> tuple[dict[int, dict[str, Any]], Counter]:
    char_path = sweep_dir / "characters.jsonl"
    chars: dict[int, dict[str, Any]] = {}
    trait_levels: Counter = Counter()
    if not char_path.exists():
        return chars, trait_levels
    for line in char_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        char_id = int(row["char_id"])
        combo = combo_from_big_five(row.get("big_five"))
        row["b5_combo"] = combo
        row["b5_levels"] = parse_combo(combo)
        chars[char_id] = row
        for dim, level in row["b5_levels"].items():
            trait_levels[f"{dim}:{level}"] += 1
    return chars, trait_levels


def compact_char_info(meta: dict[str, Any]) -> dict[str, Any]:
    return {
        "age": meta.get("age"),
        "gender": meta.get("gender"),
        "ethnicity": meta.get("ethnicity"),
        "education": meta.get("education"),
        "occupation": meta.get("occupation"),
        "industry": meta.get("industry"),
        "communication_style": meta.get("communication_style"),
        "traits": meta.get("traits", []),
    }


def normalize_response_record(
    row: dict[str, Any],
    char_meta: dict[str, Any],
    response_limit: int,
    think_limit: int,
) -> dict[str, Any]:
    combo = row.get("b5") or char_meta.get("b5_combo") or combo_from_big_five(char_meta.get("big_five"))
    levels = parse_combo(combo)
    prompt = row.get("prompt", "")
    response_text = str(row.get("response_text", ""))
    think_text = str(row.get("think_text", ""))
    display_think, display_response, response_status = derive_display_segments(think_text, response_text)
    n_think = int(row.get("n_think_tokens", 0) or 0)
    n_response = int(row.get("n_response_tokens", 0) or 0)
    n_gen = int(row.get("n_gen_tokens", row.get("n_gen_captured", 0) or 0) or 0)
    return {
        "char_id": int(row["char_id"]),
        "char_name": row.get("char_name") or char_meta.get("name") or f"char_{int(row['char_id']):04d}",
        "b5_combo": combo,
        "b5_levels": levels,
        "prompt_idx": int_or_default(row.get("prompt_idx"), -1),
        "prompt_category": row.get("prompt_category") or "unknown",
        "prompt": prompt,
        "timestamp": row.get("timestamp"),
        "n_think_tokens": n_think,
        "n_response_tokens": n_response,
        "n_gen_tokens": n_gen,
        "response_excerpt": short_text(display_response, response_limit),
        "think_excerpt": short_text(display_think, think_limit),
        "has_think": bool(display_think),
        "response_status": response_status,
        "char_meta": compact_char_info(char_meta),
    }


def select_diverse(entries: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    buckets: dict[int, deque[dict[str, Any]]] = defaultdict(deque)
    for entry in entries:
        buckets[int_or_default(entry.get("prompt_idx"), -1)].append(entry)
    chosen: list[dict[str, Any]] = []
    keys = sorted(buckets)
    while keys and len(chosen) < limit:
        next_keys: list[int] = []
        for key in keys:
            bucket = buckets[key]
            if bucket and len(chosen) < limit:
                chosen.append(bucket.popleft())
            if bucket:
                next_keys.append(key)
        keys = next_keys
    return chosen


def build_triplets_by_trait(records: list[dict[str, Any]], limit: int) -> dict[str, list[dict[str, Any]]]:
    by_trait: dict[str, dict[tuple[Any, ...], dict[str, dict[str, Any]]]] = {trait: defaultdict(dict) for trait in B5_DIMS}
    for rec in records:
        for trait in B5_DIMS:
            levels = rec["b5_levels"]
            others = tuple(levels[dim] for dim in B5_DIMS if dim != trait)
            key = (rec["prompt_idx"], rec["prompt_category"], rec["prompt"], others)
            by_trait[trait][key][levels[trait]] = rec

    out: dict[str, list[dict[str, Any]]] = {}
    for trait in B5_DIMS:
        entries: list[dict[str, Any]] = []
        for (prompt_idx, prompt_category, prompt, others), bucket in by_trait[trait].items():
            if "L" not in bucket or "H" not in bucket:
                continue
            status_rank = {"clean": 0, "recovered": 1, "missing": 2, "missing_from_thinking": 3}
            pair_statuses = [bucket[level].get("response_status", "missing") for level in bucket]
            worst_rank = max(status_rank.get(status, 99) for status in pair_statuses)
            combined_tokens = sum(int(bucket[level]["n_gen_tokens"]) for level in bucket if bucket.get(level))
            entries.append(
                {
                    "prompt_idx": prompt_idx,
                    "prompt_category": prompt_category,
                    "prompt": prompt,
                    "other_levels": {dim: others[i] for i, dim in enumerate(dim for dim in B5_DIMS if dim != trait)},
                    "available_levels": [level for level in ["L", "M", "H"] if level in bucket],
                    "combined_gen_tokens": combined_tokens,
                    "worst_response_rank": worst_rank,
                    "low": bucket.get("L"),
                    "mid": bucket.get("M"),
                    "high": bucket.get("H"),
                }
            )
        entries.sort(
            key=lambda item: (
                item["worst_response_rank"],
                item["prompt_idx"],
                -item["combined_gen_tokens"],
                item["prompt_category"],
            )
        )
        out[trait] = select_diverse(entries, limit)
    return out


def load_responses(
    sweep_dir: Path,
    chars: dict[int, dict[str, Any]],
    response_limit: int,
    think_limit: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    resp_dir = sweep_dir / "responses"
    records: list[dict[str, Any]] = []
    prompt_category_counts: Counter = Counter()
    prompt_category_tokens: defaultdict[str, Counter] = defaultdict(Counter)
    prompt_counts: Counter = Counter()
    prompt_examples: dict[int, dict[str, Any]] = {}
    total_think = 0
    total_response = 0
    total_gen = 0

    if not resp_dir.exists():
        summary = {
            "n_responses": 0,
            "total_think_tokens": 0,
            "total_response_tokens": 0,
            "total_gen_tokens": 0,
            "avg_gen_tokens": 0,
            "avg_think_tokens": 0,
            "avg_response_tokens": 0,
            "avg_think_share": None,
            "prompt_categories": {},
            "prompt_category_token_means": {},
            "top_prompts": [],
        }
        return records, summary

    for path in sorted(resp_dir.glob("char_*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            char_id = int(row["char_id"])
            char_meta = chars.get(char_id, {})
            rec = normalize_response_record(row, char_meta, response_limit, think_limit)
            records.append(rec)
            total_think += rec["n_think_tokens"]
            total_response += rec["n_response_tokens"]
            total_gen += rec["n_gen_tokens"]
            prompt_category_counts[rec["prompt_category"]] += 1
            prompt_counts[rec["prompt_idx"]] += 1
            prompt_category_tokens[rec["prompt_category"]]["gen"] += rec["n_gen_tokens"]
            prompt_category_tokens[rec["prompt_category"]]["think"] += rec["n_think_tokens"]
            prompt_category_tokens[rec["prompt_category"]]["response"] += rec["n_response_tokens"]
            if rec["prompt_idx"] not in prompt_examples:
                prompt_examples[rec["prompt_idx"]] = {
                    "prompt_idx": rec["prompt_idx"],
                    "prompt_category": rec["prompt_category"],
                    "prompt": short_text(rec["prompt"], 220),
                    "count": 0,
                }
            prompt_examples[rec["prompt_idx"]]["count"] += 1

    n_responses = len(records)
    prompt_category_token_means = {
        category: {
            "avg_gen_tokens": payload["gen"] / max(prompt_category_counts[category], 1),
            "avg_think_tokens": payload["think"] / max(prompt_category_counts[category], 1),
            "avg_response_tokens": payload["response"] / max(prompt_category_counts[category], 1),
        }
        for category, payload in sorted(prompt_category_tokens.items())
    }
    summary = {
        "n_responses": n_responses,
        "total_think_tokens": total_think,
        "total_response_tokens": total_response,
        "total_gen_tokens": total_gen,
        "avg_gen_tokens": total_gen / max(n_responses, 1),
        "avg_think_tokens": total_think / max(n_responses, 1),
        "avg_response_tokens": total_response / max(n_responses, 1),
        "avg_think_share": (total_think / total_gen) if total_gen else None,
        "prompt_categories": dict(sorted(prompt_category_counts.items())),
        "prompt_category_token_means": prompt_category_token_means,
        "top_prompts": [
            prompt_examples[idx] for idx, _ in prompt_counts.most_common(24) if idx in prompt_examples
        ],
    }
    return records, summary


def load_activation_coverage(sweep_dir: Path, total_responses: int) -> dict[str, Any]:
    coverage: dict[str, Any] = {"views": {}, "layers": []}
    all_layers: set[int] = set()
    for view, dirname in VIEW_DIRS.items():
        base = sweep_dir / dirname
        if not base.exists():
            continue
        layer_rows: dict[int, int] = {}
        for layer_dir in sorted(base.glob("L*")):
            if not layer_dir.is_dir() or not layer_dir.name[1:].isdigit():
                continue
            layer = int(layer_dir.name[1:])
            rows = 0
            for meta_path in sorted(layer_dir.glob("mean_shard_*_meta.jsonl")):
                rows += count_jsonl_rows(meta_path)
            layer_rows[layer] = rows
            all_layers.add(layer)
        if not layer_rows:
            continue
        row_values = list(layer_rows.values())
        coverage["views"][view] = {
            "layers": {
                str(layer): {
                    "rows": rows,
                    "completion": (rows / total_responses) if total_responses else None,
                }
                for layer, rows in sorted(layer_rows.items())
            },
            "min_rows": min(row_values),
            "max_rows": max(row_values),
            "min_completion": (min(row_values) / total_responses) if total_responses else None,
            "max_completion": (max(row_values) / total_responses) if total_responses else None,
        }
    coverage["layers"] = sorted(all_layers)
    return coverage


def load_analysis(analysis_json: Path | None) -> dict[str, Any]:
    if analysis_json is None or not analysis_json.exists():
        return {"available": False, "path": None}

    raw = json.loads(analysis_json.read_text(encoding="utf-8"))
    views = list(raw.get("views", []))
    layers = [int(layer) for layer in raw.get("layers", [])]
    score_matrices: dict[str, dict[str, dict[str, Any]]] = {}
    norm_matrices: dict[str, dict[str, Any]] = {}
    cosine_matrices: dict[str, dict[str, dict[str, Any]]] = {}
    best_scores: list[dict[str, Any]] = []
    best_by_trait_split: dict[str, dict[str, Any]] = {trait: {} for trait in B5_DIMS}

    for view in views:
        score_matrices[view] = {}
        norm_matrices[view] = {"traits": B5_DIMS, "layers": layers, "z": []}
        cosine_matrices[view] = {}
        for split in SPLITS:
            z: list[list[float | None]] = []
            for trait in B5_DIMS:
                row_vals: list[float | None] = []
                for layer in layers:
                    payload = raw["per_view_layer"].get(f"{view}:L{layer:02d}")
                    score = None
                    if payload is not None:
                        score = payload["decodability"][trait][split]["mean_balanced_accuracy"]
                        if score is not None:
                            best_scores.append(
                                {
                                    "trait": trait,
                                    "view": view,
                                    "layer": layer,
                                    "split": split,
                                    "score": score,
                                    "matched_pairs": payload["matched_directions"][trait]["n_pairs"],
                                    "direction_norm": payload["matched_directions"][trait]["raw_norm"],
                                }
                            )
                            current_best = best_by_trait_split[trait].get(split)
                            if current_best is None or score > current_best["score"]:
                                best_by_trait_split[trait][split] = {
                                    "view": view,
                                    "layer": layer,
                                    "score": score,
                                    "matched_pairs": payload["matched_directions"][trait]["n_pairs"],
                                    "direction_norm": payload["matched_directions"][trait]["raw_norm"],
                                }
                    row_vals.append(score)
                z.append(row_vals)
            score_matrices[view][split] = {"traits": B5_DIMS, "layers": layers, "z": z}

        for trait in B5_DIMS:
            row_vals = []
            for layer in layers:
                payload = raw["per_view_layer"].get(f"{view}:L{layer:02d}")
                row_vals.append(payload["matched_directions"][trait]["raw_norm"] if payload is not None else None)
            norm_matrices[view]["z"].append(row_vals)

        for layer in layers:
            payload = raw["per_view_layer"].get(f"{view}:L{layer:02d}")
            if payload is None:
                continue
            cosine_payload = payload["direction_cosines"]
            matrix = []
            for row_trait in B5_DIMS:
                matrix.append([cosine_payload[row_trait].get(col_trait) for col_trait in B5_DIMS])
            cosine_matrices[view][str(layer)] = {
                "traits": B5_DIMS,
                "z": matrix,
            }

    best_scores.sort(key=lambda item: item["score"], reverse=True)
    return {
        "available": True,
        "path": str(analysis_json),
        "views": views,
        "layers": layers,
        "splits": SPLITS,
        "score_matrices": score_matrices,
        "norm_matrices": norm_matrices,
        "cosine_matrices": cosine_matrices,
        "best_scores": best_scores[:48],
        "best_by_trait_split": best_by_trait_split,
    }


def _shard_id(path: Path) -> str:
    match = re.search(r"mean_shard_(\d+)", path.name)
    if not match:
        raise ValueError(f"Unrecognized shard filename: {path}")
    return match.group(1)


def load_activation_shards(act_dir: Path, layer: int):
    try:
        import torch
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("torch is required to load activation shards") from exc

    layer_dir = act_dir / f"L{layer:02d}"
    if not layer_dir.exists():
        raise FileNotFoundError(f"Missing layer directory: {layer_dir}")

    shard_files = {_shard_id(p): p for p in layer_dir.glob("mean_shard_*.pt") if "_meta" not in p.name}
    meta_files = {_shard_id(p): p for p in layer_dir.glob("mean_shard_*_meta.jsonl")}
    if not shard_files or set(shard_files) != set(meta_files):
        raise FileNotFoundError(f"Incomplete activation shards under {layer_dir}")

    tensors = []
    meta: list[dict[str, Any]] = []
    for shard_id in sorted(shard_files):
        tensors.append(torch.load(shard_files[shard_id], map_location="cpu", weights_only=True).float())
        rows = [json.loads(line) for line in meta_files[shard_id].read_text(encoding="utf-8").splitlines() if line.strip()]
        meta.extend(rows)
    acts = torch.cat(tensors, dim=0)
    if acts.shape[0] != len(meta):
        raise ValueError(f"Activation/meta row mismatch for {layer_dir}: {acts.shape[0]} vs {len(meta)}")
    return acts, meta


def summarize_layer_score(analysis: dict[str, Any], view: str, layer: int, split: str = "character_holdout") -> float:
    payload = analysis.get("score_matrices", {}).get(view, {}).get(split)
    if not payload:
        return float("-inf")
    try:
        layer_idx = payload["layers"].index(layer)
    except ValueError:
        return float("-inf")
    vals = [row[layer_idx] for row in payload["z"] if row[layer_idx] is not None]
    if not vals:
        return float("-inf")
    return float(sum(vals) / len(vals))


def stratified_projection_sample(meta: list[dict[str, Any]], trait_a: str, trait_b: str, max_points: int) -> list[int]:
    if len(meta) <= max_points:
        return list(range(len(meta)))
    buckets: dict[tuple[str, str, str], deque[int]] = defaultdict(deque)
    for idx, row in enumerate(meta):
        levels = parse_combo(str(row.get("b5_combo") or row.get("b5") or "M_M_M_M_M"))
        key = (
            levels[trait_a],
            levels[trait_b],
            str(row.get("prompt_category", "unknown")),
        )
        buckets[key].append(idx)
    selected: list[int] = []
    keys = sorted(buckets)
    while keys and len(selected) < max_points:
        next_keys: list[tuple[str, str, str]] = []
        for key in keys:
            bucket = buckets[key]
            if bucket and len(selected) < max_points:
                selected.append(bucket.popleft())
            if bucket:
                next_keys.append(key)
        keys = next_keys
    return selected


def build_projection_data(
    sweep_dir: Path,
    analysis_json: Path | None,
    analysis: dict[str, Any],
    max_points: int,
) -> dict[str, Any]:
    if analysis_json is None or not analysis_json.exists() or not analysis.get("available"):
        return {"available": False}

    try:
        import torch
    except Exception as exc:  # noqa: BLE001
        return {"available": False, "error": f"torch import failed: {exc}"}

    raw = json.loads(analysis_json.read_text(encoding="utf-8"))
    out: dict[str, Any] = {"available": True, "views": []}
    per_view: dict[str, Any] = {}

    for view in analysis.get("views", []):
        layers = analysis.get("layers", [])
        if not layers:
            continue
        best_layer = max(layers, key=lambda layer: summarize_layer_score(analysis, view, layer))
        layer_score = summarize_layer_score(analysis, view, best_layer)
        layer_key = f"{view}:L{best_layer:02d}"
        view_payload = raw.get("per_view_layer", {}).get(layer_key)
        if not view_payload:
            continue
        try:
            acts, meta = load_activation_shards(sweep_dir / VIEW_DIRS[view], best_layer)
        except Exception as exc:  # noqa: BLE001
            per_view[view] = {"available": False, "error": str(exc), "layer": best_layer}
            out["views"].append(view)
            continue

        pair_payload: dict[str, Any] = {}
        for trait_a, trait_b in itertools.combinations(B5_DIMS, 2):
            dir_a = view_payload["matched_directions"][trait_a].get("direction")
            dir_b = view_payload["matched_directions"][trait_b].get("direction")
            if dir_a is None or dir_b is None:
                continue
            vec_a = torch.tensor(dir_a, dtype=acts.dtype)
            vec_b = torch.tensor(dir_b, dtype=acts.dtype)
            x = acts @ vec_a
            y = acts @ vec_b
            x = (x - x.mean()) / (x.std(unbiased=False) + 1e-6)
            y = (y - y.mean()) / (y.std(unbiased=False) + 1e-6)

            sample_idx = stratified_projection_sample(meta, trait_a, trait_b, max_points)
            points = []
            centroid_buckets: dict[tuple[str, str], list[tuple[float, float]]] = defaultdict(list)
            for i, row in enumerate(meta):
                levels = parse_combo(str(row.get("b5_combo") or row.get("b5") or "M_M_M_M_M"))
                xv = float(x[i].item())
                yv = float(y[i].item())
                centroid_buckets[(levels[trait_a], levels[trait_b])].append((xv, yv))
                if i in sample_idx:
                    points.append(
                        {
                            "x": xv,
                            "y": yv,
                            "a_level": levels[trait_a],
                            "b_level": levels[trait_b],
                            "prompt_category": str(row.get("prompt_category", "unknown")),
                            "prompt_idx": int_or_default(row.get("prompt_idx"), -1),
                            "char_id": int(row.get("char_id", -1) or -1),
                        }
                    )
            centroids = []
            for (level_a, level_b), coords in sorted(centroid_buckets.items()):
                if not coords:
                    continue
                centroids.append(
                    {
                        "a_level": level_a,
                        "b_level": level_b,
                        "x": sum(p[0] for p in coords) / len(coords),
                        "y": sum(p[1] for p in coords) / len(coords),
                        "n": len(coords),
                    }
                )
            pair_payload[f"{trait_a}__{trait_b}"] = {
                "points": points,
                "centroids": centroids,
                "pair_count": len(centroids),
            }

        per_view[view] = {
            "available": True,
            "layer": best_layer,
            "layer_score": layer_score,
            "pairs": pair_payload,
        }
        out["views"].append(view)

    out["per_view"] = per_view
    return out


def load_control_dataset(control_dir: Path | None) -> dict[str, Any]:
    if control_dir is None or not control_dir.exists():
        return {"available": False}

    records: list[dict[str, Any]] = []
    for path in sorted(control_dir.glob("records_shard_*.jsonl")):
        for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("track") == "reasoning":
                extracted, is_correct = score_reasoning_response(
                    str(row.get("scenario_id", "")),
                    str(row.get("response_text", "")),
                    str(row.get("full_text", "")),
                )
                row["answer_extracted"] = extracted or None
                row["is_correct"] = is_correct
                row["score_status"] = (
                    "correct" if is_correct is True else "incorrect" if is_correct is False else "unscorable"
                )
            records.append(row)

    if not records:
        return {"available": True, "path": str(control_dir), "n_records": 0}

    reasoning = [row for row in records if row.get("track") == "reasoning"]
    social = [row for row in records if row.get("track") == "social"]
    scored_reasoning = [row for row in reasoning if row.get("is_correct") is not None]
    unscored_reasoning = [row for row in reasoning if row.get("is_correct") is None]
    correct_reasoning = [row for row in scored_reasoning if row.get("is_correct") is True]

    by_mode: dict[str, dict[str, Any]] = defaultdict(lambda: {"total": 0, "scored": 0, "correct": 0})
    scenario_mode: dict[str, dict[str, dict[str, Any]]] = defaultdict(
        lambda: defaultdict(lambda: {"total": 0, "scored": 0, "correct": 0})
    )
    trait_level: dict[str, dict[str, dict[str, Any]]] = defaultdict(
        lambda: defaultdict(lambda: {"total": 0, "scored": 0, "correct": 0})
    )
    paired: dict[tuple[str, str, str], dict[str, bool | None]] = defaultdict(dict)
    paired_rows: dict[tuple[str, str, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    failures: list[dict[str, Any]] = []

    for row in reasoning:
        mode = str(row.get("mode", "unknown"))
        scenario = str(row.get("scenario_id", "unknown"))
        trait = str(row.get("target_trait", "unknown"))
        level = str(row.get("target_level", "unknown"))
        score_value = row.get("is_correct")
        is_correct = score_value is True
        preview_think, preview_response, _ = derive_display_segments(
            str(row.get("think_text", "")),
            str(row.get("response_text", "")),
        )

        by_mode[mode]["total"] += 1
        by_mode[mode]["scored"] += int(score_value is not None)
        by_mode[mode]["correct"] += int(is_correct)
        scenario_mode[scenario][mode]["total"] += 1
        scenario_mode[scenario][mode]["scored"] += int(score_value is not None)
        scenario_mode[scenario][mode]["correct"] += int(is_correct)
        trait_level[trait][level]["total"] += 1
        trait_level[trait][level]["scored"] += int(score_value is not None)
        trait_level[trait][level]["correct"] += int(is_correct)
        paired[(str(row.get("pair_id")), str(row.get("prompt_id")), mode)][level] = score_value
        paired_rows[(str(row.get("pair_id")), str(row.get("prompt_id")), mode)][level] = {
            "trait": trait,
            "level": level,
            "mode": mode,
            "scenario_id": scenario,
            "expected": row.get("answer_key"),
            "answer_extracted": row.get("answer_extracted"),
            "status": "correct" if score_value is True else "incorrect" if score_value is False else "unscorable",
            "response_excerpt": short_text(preview_response or preview_think, 220),
        }

        if score_value is not True and len(failures) < 18:
            failures.append(
                {
                    "trait": trait,
                    "level": level,
                    "mode": mode,
                    "scenario_id": scenario,
                    "status": "incorrect" if score_value is False else "unscorable",
                    "answer_extracted": row.get("answer_extracted"),
                    "expected": row.get("answer_key"),
                    "score_note": (
                        "No strict 'Final Answer:' line recovered."
                        if score_value is None
                        else "Recovered final answer disagrees with the answer key."
                    ),
                    "response_excerpt": short_text(preview_response or preview_think, 220),
                }
            )

    mode_accuracy = {
        mode: {
            "total": payload["total"],
            "scored": payload["scored"],
            "correct": payload["correct"],
            "accuracy": (payload["correct"] / payload["scored"]) if payload["scored"] else None,
            "coverage": (payload["scored"] / payload["total"]) if payload["total"] else None,
        }
        for mode, payload in sorted(by_mode.items())
    }

    scenario_accuracy = {
        scenario: {
            mode: {
                "total": payload["total"],
                "scored": payload["scored"],
                "correct": payload["correct"],
                "accuracy": (payload["correct"] / payload["scored"]) if payload["scored"] else None,
                "coverage": (payload["scored"] / payload["total"]) if payload["total"] else None,
            }
            for mode, payload in sorted(modes.items())
        }
        for scenario, modes in sorted(scenario_mode.items())
    }

    trait_accuracy = {
        trait: {
            level: {
                "total": payload["total"],
                "scored": payload["scored"],
                "correct": payload["correct"],
                "accuracy": (payload["correct"] / payload["scored"]) if payload["scored"] else None,
                "coverage": (payload["scored"] / payload["total"]) if payload["total"] else None,
            }
            for level, payload in sorted(levels.items())
        }
        for trait, levels in sorted(trait_level.items())
    }

    paired_delta = []
    grouped_deltas: dict[tuple[str, str], list[int]] = defaultdict(list)
    grouped_pair_totals: Counter = Counter()
    paired_examples: list[dict[str, Any]] = []
    pair_audit = {
        "total_pairs": 0,
        "complete_pairs": 0,
        "fully_scored_pairs": 0,
        "changed_pairs": 0,
        "observed_anomaly_count": 0,
    }
    observed_anomaly_count = 0
    for row in reasoning:
        answer = str(row.get("answer_extracted") or "")
        if "\n" in answer or "Explanation" in answer or "Final Answer" in answer:
            observed_anomaly_count += 1
    for (pair_id, prompt_id, mode), levels in paired.items():
        pair_audit["total_pairs"] += 1
        row_levels = paired_rows[(pair_id, prompt_id, mode)]
        if "low" in levels and "high" in levels:
            pair_audit["complete_pairs"] += 1
            low_row = row_levels.get("low", {})
            high_row = row_levels.get("high", {})
            both_scored = levels["low"] is not None and levels["high"] is not None
            if both_scored:
                pair_audit["fully_scored_pairs"] += 1
                delta_value = int(bool(levels["high"])) - int(bool(levels["low"]))
                if delta_value != 0:
                    pair_audit["changed_pairs"] += 1
            else:
                delta_value = None

            paired_examples.append(
                {
                    "trait": str(
                        low_row.get("trait")
                        or high_row.get("trait")
                        or (pair_id.split(":", 1)[1] if ":" in pair_id else pair_id)
                    ),
                    "mode": mode,
                    "scenario_id": str(low_row.get("scenario_id") or high_row.get("scenario_id") or "unknown"),
                    "expected": low_row.get("expected") or high_row.get("expected"),
                    "low_status": low_row.get("status", "missing"),
                    "high_status": high_row.get("status", "missing"),
                    "low_answer": low_row.get("answer_extracted"),
                    "high_answer": high_row.get("answer_extracted"),
                    "low_excerpt": low_row.get("response_excerpt"),
                    "high_excerpt": high_row.get("response_excerpt"),
                    "delta": delta_value,
                }
            )

        if "low" in levels and "high" in levels:
            trait = pair_id.split(":", 1)[1] if ":" in pair_id else pair_id
            grouped_pair_totals[(trait, mode)] += 1
            if levels["low"] is None or levels["high"] is None:
                continue
            grouped_deltas[(trait, mode)].append(int(bool(levels["high"])) - int(bool(levels["low"])))
    for (trait, mode), diffs in sorted(grouped_deltas.items()):
        paired_delta.append(
            {
                "trait": trait,
                "mode": mode,
                "n_pairs": len(diffs),
                "n_pairs_total": grouped_pair_totals[(trait, mode)],
                "coverage": (
                    len(diffs) / grouped_pair_totals[(trait, mode)]
                    if grouped_pair_totals[(trait, mode)]
                    else None
                ),
                "mean_high_minus_low": sum(diffs) / len(diffs) if diffs else None,
            }
        )

    pair_audit["observed_anomaly_count"] = observed_anomaly_count
    paired_examples.sort(
        key=lambda row: (
            0 if row.get("delta") not in (None, 0) else 1 if "incorrect" in {row.get("low_status"), row.get("high_status")} else 2,
            str(row.get("trait")),
            str(row.get("mode")),
            str(row.get("scenario_id")),
        )
    )

    return {
        "available": True,
        "path": str(control_dir),
        "n_records": len(records),
        "n_reasoning": len(reasoning),
        "n_reasoning_scored": len(scored_reasoning),
        "n_reasoning_unscored": len(unscored_reasoning),
        "n_social": len(social),
        "reasoning_accuracy": (len(correct_reasoning) / len(scored_reasoning)) if scored_reasoning else None,
        "reasoning_coverage": (len(scored_reasoning) / len(reasoning)) if reasoning else None,
        "mode_accuracy": mode_accuracy,
        "scenario_accuracy": scenario_accuracy,
        "trait_accuracy": trait_accuracy,
        "paired_delta": paired_delta,
        "pair_audit": pair_audit,
        "paired_examples": paired_examples[:18],
        "failures": failures,
    }


def parse_latest_pass2(log_path: Path | None) -> dict[str, Any] | None:
    if log_path is None or not log_path.exists():
        return None
    text = log_path.read_text(encoding="utf-8", errors="replace")
    matches = list(PASS2_RE.finditer(text))
    if not matches:
        return None
    last = matches[-1]
    pass2_done = "[PASS2 DONE]" in text
    chars_total = int(last.group("chars_total"))
    return {
        "path": str(log_path),
        "chars_done": chars_total if pass2_done else int(last.group("chars_done")),
        "chars_total": chars_total,
        "responses_done": int(last.group("responses")),
        "gen_tokens_m": float(last.group("gen_m")),
        "gen_tokens_per_s": float(last.group("gen_tps")),
        "seq_tokens_m": float(last.group("seq_m")),
        "seq_tokens_per_s": float(last.group("seq_tps")),
        "pass2_done": pass2_done,
    }


def build_bundle(args: argparse.Namespace) -> dict[str, Any]:
    sweep_dir = Path(args.sweep_dir)
    chars, trait_level_counts = load_characters(sweep_dir)
    records, response_summary = load_responses(sweep_dir, chars, args.response_chars, args.think_chars)
    triplets = build_triplets_by_trait(records, args.max_triplets_per_trait)
    coverage = load_activation_coverage(sweep_dir, response_summary["n_responses"])
    analysis_json = detect_analysis_json(sweep_dir, args.analysis_json)
    analysis = load_analysis(analysis_json)
    projections = build_projection_data(sweep_dir, analysis_json, analysis, args.projection_max_points)
    pass2_log = detect_pass2_log(sweep_dir, args.pass2_log)
    pass2_status = parse_latest_pass2(pass2_log)
    control_dir = detect_control_dir(args.control_dir)
    control = load_control_dataset(control_dir)

    title = args.title or f"Personality Sweep Visualizer: {sweep_dir.name}"
    return {
        "title": title,
        "generated_at": now_iso(),
        "sweep_dir": str(sweep_dir.resolve()),
        "summary": {
            "n_characters": len(chars),
            **response_summary,
            "trait_level_counts": dict(sorted(trait_level_counts.items())),
            "activation_views": sorted(coverage["views"].keys()),
            "activation_layers": coverage["layers"],
        },
        "coverage": coverage,
        "analysis": analysis,
        "projections": projections,
        "pass2_status": pass2_status,
        "control": control,
        "triplets_by_trait": triplets,
        "notes": {
            "matched_contrast": (
                "Triplets hold the prompt and the other four Big Five levels fixed. In this sweep the demographic "
                "scaffold also changes with the combo, so example comparisons are suggestive rather than perfectly controlled."
            ),
            "decodability": "Balanced accuracy of a linear probe. Higher means the trait is more linearly recoverable from that activation view.",
            "direction_norm": "Norm of the matched H-L mean difference. Higher means stronger separability after controlling for the other four traits and prompt.",
            "direction_cosine": "Cosine overlap between matched trait directions. Near zero means cleaner separation; large magnitude means entanglement.",
            "projection": "Samples projected onto two matched trait directions at the strongest average character-holdout layer for that view.",
            "correctness": (
                "Control-dataset accuracy is measured only on reasoning prompts with a recoverable line that starts "
                "exactly with 'Final Answer:'. Rows without a strict final-answer line are excluded from accuracy "
                "and counted as unscorable. Social prompts are present for style inspection but not scored."
            ),
        },
    }


def render_html(template_path: Path, bundle: dict[str, Any]) -> str:
    template = template_path.read_text(encoding="utf-8")
    payload = sanitize_json_for_html(json.dumps(bundle, separators=(",", ":")))
    return template.replace("__BUNDLE_JSON__", payload)


def main() -> None:
    args = parse_args()
    output_html = Path(args.output_html)
    output_html.parent.mkdir(parents=True, exist_ok=True)

    bundle = build_bundle(args)
    html = render_html(Path(args.template), bundle)
    output_html.write_text(html, encoding="utf-8")

    analysis_state = "ready" if bundle["analysis"].get("available") else "pending"
    pass2_state = bundle["pass2_status"]["chars_done"] if bundle.get("pass2_status") else 0
    print(f"[DONE] wrote {output_html}")
    print(f"[INFO] analysis={analysis_state} responses={fmt_int(bundle['summary']['n_responses'])} pass2_chars={pass2_state}")


if __name__ == "__main__":
    main()
