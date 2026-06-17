#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import random
import re
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import requests
from bs4 import BeautifulSoup
from ebooklib import ITEM_DOCUMENT, epub


DEFAULT_BOOKS_ROOT = "/home/orwel/dev_genius/literary_engine/Books"
DEFAULT_BASE_URL = "http://127.0.0.1:30003/v1"
DEFAULT_API_MODEL = "/home/orwel/dev_genius/models/Qwen3.6-35B-A3B"
DEFAULT_OUTPUT_ROOT = "/home/orwel/dev_genius/experiments/Character Creation/sweep_v4"
DEFAULT_TAG = "book_character_prefill_dataset_qwen36_v1"

BEHAVIORS: dict[str, str] = {
    "conflict_detection": "The continuation must explicitly notice a live tension between obligations, values, desires, or loyalties.",
    "state_carryover": "An internal state from the scene should continue shaping behavior after a small distraction or shift in topic.",
    "repair_after_challenge": "A challenge or contradiction should trigger a repair, revision, or explicit re-evaluation.",
    "constraint_preservation": "A secrecy rule, promise, persona boundary, or response rule must be preserved while adapting naturally.",
    "selective_introspection": "Brief self-reflection should appear only when a real ambiguity or threshold exists; over-explaining is a failure mode.",
}

NONFICTION_HINTS = (
    "python made simple",
    "scaling responsible ai",
    "befriending silence",
    "off the hook",
)
FRONT_MATTER_HINTS = (
    "praise for",
    "table of contents",
    "contents",
    "copyright",
    "all rights reserved",
    "isbn",
    "acknowledg",
    "about the author",
    "also by",
    "cover",
    "dedication",
)
CONFLICT_WORDS = (
    "but",
    "however",
    "though",
    "yet",
    "instead",
    "cannot",
    "can't",
    "won't",
    "should",
    "must",
    "afraid",
    "fear",
    "promise",
    "secret",
    "sorry",
    "angry",
    "grief",
    "shame",
    "regret",
    "hesitate",
    "doubt",
)


@dataclass(frozen=True)
class SceneCandidate:
    scene_id: str
    source_path: str
    source_group: str
    source_title: str
    doc_name: str
    score: int
    excerpt: str


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def log(log_path: Path, msg: str) -> None:
    line = f"[{now_iso()}] {msg}"
    print(line, flush=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_behavior_allowlist(raw: str) -> list[str]:
    if not raw.strip():
        return []
    items = [part.strip() for part in raw.split(",") if part.strip()]
    invalid = [item for item in items if item not in BEHAVIORS]
    if invalid:
        raise SystemExit(f"Unknown behaviors in --behavior-allowlist: {', '.join(sorted(invalid))}")
    out: list[str] = []
    seen: set[str] = set()
    for item in items:
        if item in seen:
            continue
        out.append(item)
        seen.add(item)
    return out


def chat_generate_api(
    base_url: str,
    api_model: str,
    messages: list[dict[str, str]],
    *,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    timeout: int,
) -> tuple[str, dict[str, Any]]:
    payload = {
        "model": api_model,
        "messages": messages,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "max_tokens": max_new_tokens,
        "chat_template_kwargs": {"enable_thinking": False},
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    t0 = time.time()
    resp = requests.post(
        base_url.rstrip("/") + "/chat/completions",
        headers={"Authorization": "Bearer none", "Content-Type": "application/json"},
        json=payload,
        timeout=(20, timeout),
        stream=True,
    )
    resp.raise_for_status()
    text_parts: list[str] = []
    usage: dict[str, Any] = {}
    last_check_words = 0
    aborted = False
    abort_reason: str | None = None
    for raw_line in resp.iter_lines(decode_unicode=True):
        if not raw_line:
            continue
        if not raw_line.startswith("data: "):
            continue
        data = raw_line[6:]
        if data == "[DONE]":
            break
        event = json.loads(data)
        if event.get("usage"):
            usage = event["usage"]
        choices = event.get("choices") or []
        if not choices:
            continue
        delta = choices[0].get("delta") or {}
        piece = delta.get("content") or ""
        if piece:
            text_parts.append(piece)
            joined = "".join(text_parts)
            word_count = len(re.findall(r"\w+", joined))
            if (len(joined) >= 2000 or word_count >= 350) and (word_count - last_check_words >= 80):
                last_check_words = word_count
                abort_reason = doom_loop_reason(joined)
                if abort_reason:
                    aborted = True
                    resp.close()
                    break
    dt = time.time() - t0
    text = "".join(text_parts)
    comp = int(usage.get("completion_tokens") or 0)
    if comp <= 0:
        comp = len(re.findall(r"\w+", text))
    return text, {
        "generated_tokens": comp,
        "latency_s": dt,
        "tokens_per_s": float(comp / max(dt, 1e-9)),
        "aborted": aborted,
        "abort_reason": abort_reason,
    }


def extract_json(text: str) -> Any:
    return json.JSONDecoder().raw_decode(text.lstrip())[0]


def canonical_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def doom_loop_reason(text: str) -> str | None:
    words = re.findall(r"\w+", text.lower())
    if len(text) < 2000 and len(words) < 350:
        return None
    recent_words = words[-240:]
    if len(recent_words) >= 180:
        uniq_ratio = len(set(recent_words)) / max(len(recent_words), 1)
        if uniq_ratio < 0.18:
            return f"low_word_novelty:{uniq_ratio:.3f}"
    recent_lines = [canonical_text(line) for line in text.splitlines() if line.strip()]
    if len(recent_lines) >= 6 and len(set(recent_lines[-6:])) <= 2:
        return "repeated_lines"
    tail = canonical_text(text[-120:])
    window = canonical_text(text[-1600:])
    if len(tail) >= 60 and window.count(tail) >= 4:
        return "repeated_suffix"
    return None


def cleaned_paragraphs(raw_html: bytes) -> list[str]:
    soup = BeautifulSoup(raw_html, "lxml")
    text = soup.get_text("\n")
    paras: list[str] = []
    for line in text.splitlines():
        line = re.sub(r"\s+", " ", line.replace("\xa0", " ")).strip()
        if not line:
            continue
        if len(line) < 30:
            continue
        paras.append(line)
    return paras


def looks_like_front_matter(text: str) -> bool:
    lower = text.lower()
    if any(hint in lower for hint in FRONT_MATTER_HINTS):
        return True
    if sum(1 for line in text.splitlines() if len(line.split()) <= 5) >= 5:
        return True
    return False


def scene_score(text: str) -> int:
    lower = text.lower()
    score = 0
    score += min(text.count('"') + text.count("“") + text.count("”"), 6) // 2
    if "?" in text:
        score += 1
    if any(word in lower for word in CONFLICT_WORDS):
        score += 1
    if any(pron in lower for pron in (" he ", " she ", " they ", " i ", " we ")):
        score += 1
    if text.count("\n\n") >= 2:
        score += 1
    if re.search(r"[A-Z][a-z]+ said", text):
        score += 1
    return score


def is_fiction_epub(path: Path) -> bool:
    if path.suffix.lower() != ".epub":
        return False
    lower = path.name.lower()
    if any(hint in lower for hint in NONFICTION_HINTS):
        return False
    return True


def list_books(root: Path) -> list[Path]:
    books = [p for p in root.rglob("*.epub") if is_fiction_epub(p)]
    books.sort()
    return books


def extract_scene_candidates(book_path: Path, *, max_scenes_per_book: int) -> list[SceneCandidate]:
    book = epub.read_epub(str(book_path))
    candidates: list[SceneCandidate] = []
    source_group = book_path.parent.name
    source_title = book_path.stem
    seen: set[str] = set()
    local_idx = 0
    for item in book.get_items():
        if item.get_type() != ITEM_DOCUMENT:
            continue
        paras = cleaned_paragraphs(item.get_body_content())
        if len(paras) < 4:
            continue
        for window in (4, 5, 6):
            for start in range(0, max(0, len(paras) - window + 1), 2):
                chunk = "\n\n".join(paras[start : start + window])
                if len(chunk) < 900 or len(chunk) > 2800:
                    continue
                if looks_like_front_matter(chunk):
                    continue
                score = scene_score(chunk)
                if score < 3:
                    continue
                sig = canonical_text(chunk[:420])
                if sig in seen:
                    continue
                seen.add(sig)
                local_idx += 1
                candidates.append(
                    SceneCandidate(
                        scene_id=f"{book_path.stem[:24].replace(' ', '_')}_{local_idx:04d}",
                        source_path=str(book_path),
                        source_group=source_group,
                        source_title=source_title,
                        doc_name=item.get_name(),
                        score=score,
                        excerpt=chunk,
                    )
                )
    candidates.sort(key=lambda c: (c.score, len(c.excerpt)), reverse=True)
    return candidates[:max_scenes_per_book]


def structure_messages(scene: SceneCandidate, preferred_behaviors: list[str] | None = None) -> list[dict[str, str]]:
    allowed_behaviors = preferred_behaviors or list(BEHAVIORS)
    schema = {
        "title": "short title",
        "behavior": "one_of_allowed",
        "focal_character": "name or concise handle",
        "counterpart": "name or concise handle",
        "scene_summary": "2-3 sentences",
        "emotional_state": "compact state description",
        "hidden_conflict": "what is pulling in different directions",
        "carryover_target": "what should still shape later behavior",
        "system_prompt": "compact persona/state grounding, 2-4 bullets or short lines",
        "user_prompt": "the direct prompt or follow-up line the character is now answering",
        "metrics": [{"id": "snake_case", "description": "verifiable success criterion"}],
        "expected_pass": "what a strong stateful completion should do",
        "expected_fail": "what a fluent but shallow or mask-wearing completion would miss",
        "quality_score": 1,
        "quality_notes": "why this is or is not a good probing item"
    }
    user = (
        "Convert this fiction scene into one diagnostic character-continuation item.\n"
        "Return JSON only. No markdown.\n\n"
        "Goal:\n"
        "- We want a prompt that can separate genuine character-stateful reasoning from surface-level mask wearing.\n"
        "- Ground the item in the excerpt, but do not copy long phrases from it.\n"
        "- Favor scenes with clear tension, carryover, secrecy, repair, or thresholded introspection.\n\n"
        "Allowed behaviors:\n"
        + "\n".join(f"- {k}: {BEHAVIORS[k]}" for k in allowed_behaviors)
        + "\n\nRules:\n"
        + "- quality_score 5 only if the item clearly supports pass/fail contrast on the latent behavior.\n"
        + "- Use 2 to 4 metrics.\n"
        + "- system_prompt should be short and usable for replay later.\n"
        + "- user_prompt should be a direct final turn that elicits the continuation.\n"
        + "- Avoid defaulting to conflict_detection or constraint_preservation if state_carryover, repair_after_challenge, or selective_introspection fits equally well.\n"
        + (
            f"- Prefer these target behaviors when the excerpt honestly supports them: {', '.join(preferred_behaviors)}.\n"
            "- If the excerpt does not support one of those target behaviors cleanly, score it low instead of forcing a bad fit.\n"
            if preferred_behaviors
            else ""
        )
        + (
            "- Only emit a behavior from the target list above.\n"
            if preferred_behaviors
            else ""
        )
        + "- If the excerpt is front matter, generic exposition, or otherwise weak, say so in quality_notes and score it low.\n\n"
        + f"Source title: {scene.source_title}\n"
        + f"Source group: {scene.source_group}\n"
        + f"Excerpt:\n{scene.excerpt}\n\n"
        + f"Schema:\n{json.dumps(schema, ensure_ascii=False, indent=2)}"
    )
    return [
        {"role": "system", "content": "You build compact, high-signal diagnostic character items. Return valid JSON only."},
        {"role": "user", "content": user},
    ]


def trace_generation_messages(item: dict[str, Any]) -> list[dict[str, str]]:
    spec = {
        "behavior": item["behavior"],
        "focal_character": item["focal_character"],
        "counterpart": item["counterpart"],
        "scene_summary": item["scene_summary"],
        "emotional_state": item["emotional_state"],
        "hidden_conflict": item["hidden_conflict"],
        "carryover_target": item["carryover_target"],
        "system_prompt": item["system_prompt"],
        "user_prompt": item["user_prompt"],
        "metrics": item["metrics"],
        "expected_pass": item["expected_pass"],
        "expected_fail": item["expected_fail"],
    }
    schema = {
        "pass_candidates": [
            {"think": "internal thought", "response": "outward response"},
            {"think": "internal thought", "response": "outward response"},
        ],
        "fail_candidates": [
            {"think": "internal thought", "response": "outward response"},
            {"think": "internal thought", "response": "outward response"},
        ],
    }
    user = (
        "Write diagnostic assistant completions for this character item.\n"
        "Return JSON only. No markdown.\n\n"
        "Requirements:\n"
        "- Generate exactly 2 pass_candidates and exactly 2 fail_candidates.\n"
        "- Return each candidate as an object with fields think and response.\n"
        "- We will later render them into this exact outer format:\n"
        "/think\n<internal thought>\n/end-think\nResponse: <outward response>\n"
        "- pass_candidates must satisfy all metrics and preserve the latent character state.\n"
        "- fail_candidates must remain fluent and somewhat in-character on the surface, but miss the latent requirement in a behavior-typical way.\n"
        "- Keep each /think under 120 words and each Response under 120 words.\n"
        "- Do not mention metric ids, labels, dataset language, or evaluation language.\n"
        "- Make the two pass candidates meaningfully different from each other, and likewise for fail candidates.\n\n"
        f"Item spec:\n{json.dumps(spec, ensure_ascii=False, indent=2)}\n\n"
        f"Schema:\n{json.dumps(schema, ensure_ascii=False, indent=2)}"
    )
    return [
        {"role": "system", "content": "You generate concise character continuations with visible internal thought. Return valid JSON only."},
        {"role": "user", "content": user},
    ]


def trace_judge_messages(item: dict[str, Any], candidates: list[dict[str, Any]]) -> list[dict[str, str]]:
    spec = {
        "behavior": item["behavior"],
        "scene_summary": item["scene_summary"],
        "emotional_state": item["emotional_state"],
        "hidden_conflict": item["hidden_conflict"],
        "carryover_target": item["carryover_target"],
        "metrics": item["metrics"],
        "expected_pass": item["expected_pass"],
        "expected_fail": item["expected_fail"],
    }
    payload = [{"index": idx, "target_label": row["target_label"], "completion": row["assistant_completion"]} for idx, row in enumerate(candidates)]
    user = (
        "Judge these character continuations against the item metrics.\n"
        "Return JSON only with this schema:\n"
        '{"ratings":[{"index":0,"format_ok":true,"all_metrics_pass":true,"passed_metric_ids":["..."],"failed_metric_ids":["..."],"trace_quality":1,"note":"<=20 words"}]}\n\n'
        "Guidelines:\n"
        "- format_ok is true only if the completion includes /think, /end-think, and Response:.\n"
        "- all_metrics_pass is true only if every metric passes.\n"
        "- trace_quality should be 1 to 5 and reflect how plausible and stateful the trace feels, not just formatting.\n"
        "- Judge the completion as written, not the target label.\n\n"
        f"Item spec:\n{json.dumps(spec, ensure_ascii=False, indent=2)}\n\n"
        f"Candidates:\n{json.dumps(payload, ensure_ascii=False, indent=2)}"
    )
    return [
        {"role": "system", "content": "You are a strict rubric judge for character continuations. Return valid JSON only."},
        {"role": "user", "content": user},
    ]


def normalize_structured_item(scene: SceneCandidate, payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "item_id": scene.scene_id,
        "source_path": scene.source_path,
        "source_group": scene.source_group,
        "source_title": scene.source_title,
        "doc_name": scene.doc_name,
        "source_excerpt": scene.excerpt,
        "scene_score": scene.score,
        "title": str(payload.get("title", "")).strip(),
        "behavior": str(payload.get("behavior", "")).strip(),
        "focal_character": str(payload.get("focal_character", "")).strip(),
        "counterpart": str(payload.get("counterpart", "")).strip(),
        "scene_summary": str(payload.get("scene_summary", "")).strip(),
        "emotional_state": str(payload.get("emotional_state", "")).strip(),
        "hidden_conflict": str(payload.get("hidden_conflict", "")).strip(),
        "carryover_target": str(payload.get("carryover_target", "")).strip(),
        "system_prompt": str(payload.get("system_prompt", "")).strip(),
        "user_prompt": str(payload.get("user_prompt", "")).strip(),
        "metrics": [
            {"id": str(m.get("id", "")).strip(), "description": str(m.get("description", "")).strip()}
            for m in payload.get("metrics", [])
            if isinstance(m, dict)
        ],
        "expected_pass": str(payload.get("expected_pass", "")).strip(),
        "expected_fail": str(payload.get("expected_fail", "")).strip(),
        "quality_score": int(payload.get("quality_score", 0) or 0),
        "quality_notes": str(payload.get("quality_notes", "")).strip(),
    }


def item_signature(item: dict[str, Any]) -> str:
    return " | ".join(
        canonical_text(item.get(k, "")) for k in ("behavior", "focal_character", "scene_summary", "user_prompt")
    )


def item_is_usable(item: dict[str, Any]) -> bool:
    if item["behavior"] not in BEHAVIORS:
        return False
    if item["quality_score"] < 4:
        return False
    if len(item["metrics"]) < 2:
        return False
    if not item["system_prompt"] or not item["user_prompt"]:
        return False
    if len(item["scene_summary"]) < 40 or len(item["hidden_conflict"]) < 20:
        return False
    return True


def choose_examples(item: dict[str, Any], judged_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    chosen: list[dict[str, Any]] = []
    metric_ids = [m["id"] for m in item["metrics"]]
    for target_label in ("pass", "fail"):
        pool = [row for row in judged_rows if row["target_label"] == target_label]
        if target_label == "pass":
            pool.sort(
                key=lambda r: (
                    int(r["format_ok"]),
                    int(r["all_metrics_pass"]),
                    int(r["trace_quality"]),
                    len(r["passed_metric_ids"]),
                    -len(r["failed_metric_ids"]),
                ),
                reverse=True,
            )
            ok = [r for r in pool if r["format_ok"] and r["all_metrics_pass"]]
        else:
            pool.sort(
                key=lambda r: (
                    int(r["format_ok"]),
                    int(not r["all_metrics_pass"]),
                    len(r["failed_metric_ids"]),
                    int(r["trace_quality"]),
                ),
                reverse=True,
            )
            ok = [r for r in pool if r["format_ok"] and not r["all_metrics_pass"]]
        selected = ok[0] if ok else (pool[0] if pool else None)
        if selected is None:
            continue
        example_key = f"{item['behavior']}|{item['item_id']}|{1 if target_label == 'pass' else 0}"
        chosen.append(
            {
                "example_key": example_key,
                "item_id": item["item_id"],
                "behavior": item["behavior"],
                "label": 1 if target_label == "pass" else 0,
                "target_label": target_label,
                "source_title": item["source_title"],
                "source_group": item["source_group"],
                "title": item["title"],
                "focal_character": item["focal_character"],
                "counterpart": item["counterpart"],
                "scene_summary": item["scene_summary"],
                "emotional_state": item["emotional_state"],
                "hidden_conflict": item["hidden_conflict"],
                "carryover_target": item["carryover_target"],
                "system_prompt": item["system_prompt"],
                "user_prompt": item["user_prompt"],
                "metrics": item["metrics"],
                "metric_ids": metric_ids,
                "assistant_completion": selected["assistant_completion"],
                "judge_format_ok": bool(selected["format_ok"]),
                "judge_all_metrics_pass": bool(selected["all_metrics_pass"]),
                "judge_trace_quality": int(selected["trace_quality"]),
                "judge_passed_metric_ids": selected["passed_metric_ids"],
                "judge_failed_metric_ids": selected["failed_metric_ids"],
                "judge_note": selected["note"],
                "messages": [
                    {"role": "system", "content": item["system_prompt"]},
                    {"role": "user", "content": item["user_prompt"]},
                ],
                "source_excerpt": item["source_excerpt"],
                "expected_pass": item["expected_pass"],
                "expected_fail": item["expected_fail"],
            }
        )
    return chosen


def render_completion(think: str, response: str) -> str:
    think = think.strip()
    response = response.strip()
    return f"/think\n{think}\n/end-think\nResponse: {response}"


def select_balanced_items(
    structured_items: list[dict[str, Any]],
    target_items: int,
    behaviors: list[str] | None = None,
    max_items_per_source_title: int = 0,
) -> list[dict[str, Any]]:
    active_behaviors = behaviors or list(BEHAVIORS)
    per_behavior_target = max(1, target_items // max(len(active_behaviors), 1))
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    seen_signatures: set[str] = set()
    for item in sorted(structured_items, key=lambda r: (r["quality_score"], r["scene_score"]), reverse=True):
        if not item["usable"]:
            continue
        sig = item["signature"]
        if sig in seen_signatures:
            continue
        seen_signatures.add(sig)
        buckets[item["behavior"]].append(item)

    selected: list[dict[str, Any]] = []
    used_ids: set[str] = set()
    source_counts: Counter[str] = Counter()
    for behavior in active_behaviors:
        by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for item in buckets.get(behavior, []):
            by_source[item["source_title"]].append(item)
        source_names = sorted(
            by_source,
            key=lambda name: (
                max((row["quality_score"], row["scene_score"]) for row in by_source[name]),
                len(by_source[name]),
            ),
            reverse=True,
        )
        picked_for_behavior = 0
        while picked_for_behavior < per_behavior_target:
            made_progress = False
            for source_name in source_names:
                rows = by_source[source_name]
                while rows and (
                    rows[0]["item_id"] in used_ids
                    or (
                        max_items_per_source_title > 0
                        and source_counts[source_name] >= max_items_per_source_title
                    )
                ):
                    rows.pop(0)
                if not rows:
                    continue
                item = rows.pop(0)
                selected.append(item)
                used_ids.add(item["item_id"])
                source_counts[source_name] += 1
                picked_for_behavior += 1
                made_progress = True
                if picked_for_behavior >= per_behavior_target:
                    break
            if not made_progress:
                break

    if len(selected) >= target_items:
        return selected[:target_items]

    leftovers: list[dict[str, Any]] = []
    for behavior in active_behaviors:
        leftovers.extend([item for item in buckets.get(behavior, [])[per_behavior_target:] if item["item_id"] not in used_ids])
    leftovers.sort(key=lambda r: (r["quality_score"], r["scene_score"]), reverse=True)
    for item in leftovers:
        if len(selected) >= target_items:
            break
        if item["item_id"] in used_ids:
            continue
        source_name = item["source_title"]
        if max_items_per_source_title > 0 and source_counts[source_name] >= max_items_per_source_title:
            continue
        selected.append(item)
        used_ids.add(item["item_id"])
        source_counts[source_name] += 1
    return selected


def chosen_pair_is_usable(rows: list[dict[str, Any]]) -> bool:
    by_label = {row["label"]: row for row in rows}
    if 0 not in by_label or 1 not in by_label:
        return False
    if by_label[1]["judge_trace_quality"] < 4:
        return False
    if by_label[0]["judge_trace_quality"] < 2:
        return False
    return True


def split_by_behavior(items: list[dict[str, Any]], seed: int) -> dict[str, str]:
    rng = random.Random(seed)
    groups: dict[str, list[str]] = defaultdict(list)
    for item in items:
        groups[item["behavior"]].append(item["item_id"])
    out: dict[str, str] = {}
    for behavior, ids in groups.items():
        ids = sorted(set(ids))
        rng.shuffle(ids)
        n = len(ids)
        n_val = max(1, round(n * 0.1))
        n_test = max(1, round(n * 0.1))
        if n >= 10 and n_val + n_test >= n:
            n_val = max(1, n // 10)
            n_test = max(1, n // 10)
        train_cut = n - n_val - n_test
        if train_cut <= 0:
            train_cut = max(1, n - 2)
        for idx, item_id in enumerate(ids):
            if idx < train_cut:
                out[item_id] = "train"
            elif idx < train_cut + n_val:
                out[item_id] = "val"
            else:
                out[item_id] = "test"
    return out


def process_scene(
    scene: SceneCandidate,
    base_url: str,
    api_model: str,
    timeout: int,
    preferred_behaviors: list[str] | None = None,
) -> dict[str, Any]:
    last_error: Exception | None = None
    for temperature in (0.6, 0.2):
        text, usage = chat_generate_api(
            base_url,
            api_model,
            structure_messages(scene, preferred_behaviors),
            max_new_tokens=1400,
            temperature=temperature,
            top_p=0.95,
            top_k=40,
            timeout=timeout,
        )
        try:
            payload = extract_json(text)
            break
        except Exception as exc:  # noqa: BLE001
            last_error = exc
    else:
        raise last_error or RuntimeError("scene structuring failed")
    item = normalize_structured_item(scene, payload)
    item["structure_usage"] = usage
    item["raw_structure_text"] = text
    item["signature"] = item_signature(item)
    item["usable"] = item_is_usable(item)
    return item


def process_item_traces(item: dict[str, Any], base_url: str, api_model: str, timeout: int) -> dict[str, Any]:
    last_error: Exception | None = None
    for temperature in (0.8, 0.4):
        gen_text, gen_usage = chat_generate_api(
            base_url,
            api_model,
            trace_generation_messages(item),
            max_new_tokens=1800,
            temperature=temperature,
            top_p=0.95,
            top_k=40,
            timeout=timeout,
        )
        try:
            payload = extract_json(gen_text)
            break
        except Exception as exc:  # noqa: BLE001
            last_error = exc
    else:
        raise last_error or RuntimeError("trace generation failed")
    def unpack(raw: Any) -> str | None:
        if not isinstance(raw, dict):
            return None
        think = str(raw.get("think", "")).strip()
        response = str(raw.get("response", "")).strip()
        if not think or not response:
            return None
        return render_completion(think, response)

    pass_candidates = [x for x in (unpack(raw) for raw in payload.get("pass_candidates", [])[:2]) if x]
    fail_candidates = [x for x in (unpack(raw) for raw in payload.get("fail_candidates", [])[:2]) if x]
    candidates = (
        [{"target_label": "pass", "assistant_completion": text} for text in pass_candidates if text]
        + [{"target_label": "fail", "assistant_completion": text} for text in fail_candidates if text]
    )
    judge_text, judge_usage = chat_generate_api(
        base_url,
        api_model,
        trace_judge_messages(item, candidates),
        max_new_tokens=700,
        temperature=0.0,
        top_p=1.0,
        top_k=1,
        timeout=timeout,
    )
    judged_payload = extract_json(judge_text)
    ratings = judged_payload.get("ratings", []) if isinstance(judged_payload, dict) else []
    rating_map = {int(r["index"]): r for r in ratings if isinstance(r, dict) and str(r.get("index", "")).isdigit()}
    judged_rows: list[dict[str, Any]] = []
    for idx, row in enumerate(candidates):
        rating = rating_map.get(idx, {})
        judged_rows.append(
            {
                **row,
                "format_ok": bool(rating.get("format_ok", False)),
                "all_metrics_pass": bool(rating.get("all_metrics_pass", False)),
                "passed_metric_ids": [str(x) for x in rating.get("passed_metric_ids", [])],
                "failed_metric_ids": [str(x) for x in rating.get("failed_metric_ids", [])],
                "trace_quality": int(rating.get("trace_quality", 0) or 0),
                "note": str(rating.get("note", "")).strip(),
            }
        )
    chosen = choose_examples(item, judged_rows)
    return {
        "item_id": item["item_id"],
        "behavior": item["behavior"],
        "title": item["title"],
        "n_candidates": len(candidates),
        "chosen_examples": chosen,
        "generation_usage": gen_usage,
        "judge_usage": judge_usage,
        "raw_generation_text": gen_text,
        "raw_judge_text": judge_text,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--books-root", type=Path, default=Path(DEFAULT_BOOKS_ROOT))
    ap.add_argument("--base-url", default=DEFAULT_BASE_URL)
    ap.add_argument("--api-model", default=DEFAULT_API_MODEL)
    ap.add_argument("--output-root", type=Path, default=Path(DEFAULT_OUTPUT_ROOT))
    ap.add_argument("--tag", default=DEFAULT_TAG)
    ap.add_argument("--max-books", type=int, default=18)
    ap.add_argument("--max-scenes-per-book", type=int, default=24)
    ap.add_argument("--target-items", type=int, default=250)
    ap.add_argument("--parallel-requests", type=int, default=8)
    ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--behavior-allowlist", default="")
    ap.add_argument("--max-items-per-source-title", type=int, default=0)
    args = ap.parse_args()
    behavior_allowlist = parse_behavior_allowlist(args.behavior_allowlist)

    stamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    out_dir = args.output_root / f"{args.tag}_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "run.log"

    books = list_books(args.books_root)[: args.max_books]
    write_json(
        out_dir / "manifest.json",
        {
            "started_at": now_iso(),
            "books_root": str(args.books_root),
            "n_books": len(books),
            "books": [str(p) for p in books],
            "base_url": args.base_url,
            "api_model": args.api_model,
            "max_scenes_per_book": args.max_scenes_per_book,
            "target_items": args.target_items,
            "parallel_requests": args.parallel_requests,
            "timeout": args.timeout,
            "seed": args.seed,
            "behavior_allowlist": behavior_allowlist,
            "max_items_per_source_title": args.max_items_per_source_title,
        },
    )

    all_scene_candidates: list[dict[str, Any]] = []
    for book in books:
        scenes = extract_scene_candidates(book, max_scenes_per_book=args.max_scenes_per_book)
        log(log_path, f"book scenes source={book.name} count={len(scenes)}")
        all_scene_candidates.extend([scene.__dict__ for scene in scenes])
    write_jsonl(out_dir / "scene_candidates.jsonl", all_scene_candidates)

    scene_objs = [SceneCandidate(**row) for row in all_scene_candidates]
    structured_items: list[dict[str, Any]] = []
    with cf.ThreadPoolExecutor(max_workers=min(args.parallel_requests, max(1, len(scene_objs)))) as pool:
        fut_map = {
            pool.submit(process_scene, scene, args.base_url, args.api_model, args.timeout, behavior_allowlist or None): scene
            for scene in scene_objs
        }
        for idx, fut in enumerate(cf.as_completed(fut_map), start=1):
            scene = fut_map[fut]
            try:
                item = fut.result()
                structured_items.append(item)
                log(
                    log_path,
                    f"structured {idx}/{len(scene_objs)} item_id={item['item_id']} behavior={item['behavior']} quality={item['quality_score']} usable={item['usable']}",
                )
            except Exception as exc:  # noqa: BLE001
                log(log_path, f"structure failed scene_id={scene.scene_id} error={exc!r}")

    structured_items.sort(key=lambda r: (r["source_title"], r["item_id"]))
    write_jsonl(out_dir / "structured_items.jsonl", structured_items)

    candidate_items = structured_items
    if behavior_allowlist:
        candidate_items = [item for item in structured_items if item["behavior"] in behavior_allowlist]
    usable_items = select_balanced_items(
        candidate_items,
        args.target_items,
        behavior_allowlist or None,
        max_items_per_source_title=args.max_items_per_source_title,
    )
    write_jsonl(out_dir / "usable_items.jsonl", usable_items)
    log(log_path, f"usable_items={len(usable_items)}")

    trace_results: list[dict[str, Any]] = []
    with cf.ThreadPoolExecutor(max_workers=min(args.parallel_requests, max(1, len(usable_items)))) as pool:
        fut_map = {pool.submit(process_item_traces, item, args.base_url, args.api_model, args.timeout): item for item in usable_items}
        for idx, fut in enumerate(cf.as_completed(fut_map), start=1):
            item = fut_map[fut]
            try:
                result = fut.result()
                trace_results.append(result)
                log(
                    log_path,
                    f"traces {idx}/{len(usable_items)} item_id={item['item_id']} behavior={item['behavior']} chosen={len(result['chosen_examples'])}",
                )
            except Exception as exc:  # noqa: BLE001
                log(log_path, f"trace failed item_id={item['item_id']} behavior={item['behavior']} error={exc!r}")

    trace_results.sort(key=lambda r: (r["behavior"], r["item_id"]))
    write_jsonl(out_dir / "trace_item_results.jsonl", trace_results)

    item_map = {item["item_id"]: item for item in usable_items}
    item_splits = split_by_behavior(usable_items, args.seed)
    completions: list[dict[str, Any]] = []
    for result in trace_results:
        if not chosen_pair_is_usable(result["chosen_examples"]):
            continue
        for row in result["chosen_examples"]:
            row["split"] = item_splits.get(row["item_id"], "train")
            completions.append(row)
    completions.sort(key=lambda r: (r["split"], r["behavior"], r["item_id"], r["label"]))
    write_jsonl(out_dir / "all_completions.jsonl", completions)
    for split in ("train", "val", "test"):
        write_jsonl(out_dir / f"{split}.jsonl", [row for row in completions if row["split"] == split])

    summary = {
        "finished_at": now_iso(),
        "n_books": len(books),
        "n_scene_candidates": len(all_scene_candidates),
        "n_structured_items": len(structured_items),
        "n_structured_items_after_behavior_filter": len(candidate_items),
        "n_usable_items": len(usable_items),
        "n_items_with_chosen_examples": len({row["item_id"] for row in completions}),
        "n_completions": len(completions),
        "requested_behaviors": behavior_allowlist,
        "behavior_item_counts": dict(sorted(Counter(item["behavior"] for item in usable_items).items())),
        "behavior_completion_counts": dict(sorted(Counter(row["behavior"] for row in completions).items())),
        "split_counts": dict(sorted(Counter(row["split"] for row in completions).items())),
        "label_counts": dict(sorted(Counter(str(row["label"]) for row in completions).items())),
        "mean_trace_quality": (
            sum(int(row["judge_trace_quality"]) for row in completions) / max(len(completions), 1)
        ),
        "books_used": [str(p) for p in books],
    }
    write_json(out_dir / "summary.json", summary)


if __name__ == "__main__":
    main()
