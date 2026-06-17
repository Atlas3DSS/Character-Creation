#!/usr/bin/env python3
"""Generate a held-out A/B/C meta-format eval dataset via OpenAI-compatible APIs.

Goal:
  evaluate whether explicit `/meta-think -> /think -> answer` scaffolds improve
  controllable personality while preserving reasoning quality.

Conditions:
  - baseline_native: stock persona prompting, request concise user-facing answer only
  - trace_explicit: visible `/meta-think`, then `/think`, then final answer/response
  - think_explicit: visible `/think`, then final answer/response

Each worker writes its own shard JSONL so runs can resume safely across servers.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import random
import re
import signal
import sys
import threading
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import requests
from tqdm import tqdm

B5_DIMS = ["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"]
_SHUTDOWN = False
_THREAD_LOCAL = threading.local()

META_BLOCK_RE = re.compile(r"(?is)/meta-think\s*(.*?)\s*/end-meta-think")
THINK_BLOCK_RE = re.compile(r"(?is)(?<!meta-)\/think\s*(.*?)\s*/end-think")
EXPLANATION_RE = re.compile(r"(?im)^\s*Explanation:\s*(.+?)\s*$")
FINAL_ANSWER_RE = re.compile(r"(?im)^\s*Final Answer:\s*(.+?)\s*$")
FINAL_RESPONSE_RE = re.compile(r"(?im)^\s*Final Response:\s*(.+(?:\n(?!/?(?:meta-think|think)|Explanation:|Final Answer:).+)*)")
THINKING_PROCESS_RE = re.compile(r"(?i)thinking process:")
TAG_STRIP_RE = re.compile(r"(?is)/meta-think\s*.*?\s*/end-meta-think|/think\s*.*?\s*/end-think|<think>\s*.*?\s*</think>")
TRAIT_LABEL_RE = re.compile(r"(?i)\b(openness|conscientiousness|extraversion|agreeableness|neuroticism)\b")
LEVEL_LABEL_RE = re.compile(r"(?i)\b(low|medium|high)\b")


def _sig_handler(signum, frame):
    del frame
    global _SHUTDOWN
    _SHUTDOWN = True
    print(f"\n[SHUTDOWN] Signal {signum} received; draining in-flight requests...")


signal.signal(signal.SIGTERM, _sig_handler)
signal.signal(signal.SIGINT, _sig_handler)


def load_v3_module(script_path: Path):
    spec = importlib.util.spec_from_file_location("personality_sweep_v3_two_pass", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module from {script_path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def reasoning_prompt(problem: str) -> str:
    return (
        "Solve the problem correctly. Keep the user-facing answer concise.\n"
        "When you present the final answer block, use exactly:\n"
        "Explanation: <one short sentence>\n"
        "Final Answer: <canonical short answer only>\n"
        "Do not use bullet points in the final answer block.\n\n"
        f"Problem: {problem}"
    )


PROMPTS: list[dict[str, Any]] = [
    {
        "prompt_id": "batball",
        "category": "reasoning",
        "track": "reasoning",
        "answer_key": "5 cents / $0.05",
        "text": reasoning_prompt(
            "A bat and a ball cost $1.10 total. The bat costs $1.00 more than the ball. How much does the ball cost?"
        ),
    },
    {
        "prompt_id": "heavyball",
        "category": "reasoning",
        "track": "reasoning",
        "answer_key": "2 weighings",
        "text": reasoning_prompt(
            "You have 8 identical-looking balls. One is heavier than the others. You have a balance scale. What is the minimum number of weighings needed to guarantee you find the heavier ball?"
        ),
    },
    {
        "prompt_id": "syllogism",
        "category": "reasoning",
        "track": "reasoning",
        "answer_key": "No, the conclusion does not follow",
        "text": reasoning_prompt(
            "All roses are flowers. Some flowers fade quickly. Can we conclude that some roses fade quickly?"
        ),
    },
    {
        "prompt_id": "tickets",
        "category": "reasoning",
        "track": "reasoning",
        "answer_key": "45 students",
        "text": reasoning_prompt(
            "A theater sold 85 tickets. Adult tickets cost $18 and student tickets cost $11. Total revenue was $1215. How many student tickets were sold?"
        ),
    },
    {
        "prompt_id": "chickens_cows",
        "category": "reasoning",
        "track": "reasoning",
        "answer_key": "11 chickens and 12 cows",
        "text": reasoning_prompt(
            "A farmer has chickens and cows. There are 23 animals total and 70 legs total. How many chickens and how many cows are there?"
        ),
    },
    {
        "prompt_id": "printers",
        "category": "reasoning",
        "track": "reasoning",
        "answer_key": "400 pages",
        "text": reasoning_prompt(
            "Three identical printers produce 180 pages in 6 minutes at the same constant rate. How many pages would five such printers produce in 8 minutes?"
        ),
    },
    {
        "prompt_id": "ages",
        "category": "reasoning",
        "track": "reasoning",
        "answer_key": "Ben is 2 years old",
        "text": reasoning_prompt(
            "Mia is 6 years older than Ben. In 4 years, Mia will be twice Ben's age. How old is Ben now?"
        ),
    },
    {
        "prompt_id": "sequence",
        "category": "reasoning",
        "track": "reasoning",
        "answer_key": "42",
        "text": reasoning_prompt(
            "What is the next number in this sequence: 2, 6, 12, 20, 30, ?"
        ),
    },
    {
        "prompt_id": "credit_public",
        "category": "social",
        "track": "open",
        "answer_key": None,
        "text": (
            "A coworker takes credit for your idea in a meeting. Respond naturally as this person would. "
            "Keep it under 180 words. Explain what you would say and what you would do next."
        ),
    },
    {
        "prompt_id": "job_choice",
        "category": "practical",
        "track": "open",
        "answer_key": None,
        "text": (
            "A close friend is choosing between a steady job and a much riskier startup role. "
            "They ask for your honest advice. Respond as this person would. Keep it under 180 words."
        ),
    },
    {
        "prompt_id": "boss_weekend",
        "category": "social",
        "track": "open",
        "answer_key": None,
        "text": (
            "Your boss asks you to work another weekend on short notice for the third time this quarter. "
            "What do you say, and what boundary do you set? Keep it under 180 words."
        ),
    },
    {
        "prompt_id": "neighbor_noise",
        "category": "social",
        "track": "open",
        "answer_key": None,
        "text": (
            "Your neighbor's music is too loud at 11 PM on a work night. Respond as this person would. "
            "Keep it under 180 words."
        ),
    },
    {
        "prompt_id": "hidden_anger_public",
        "category": "deflected",
        "track": "open",
        "answer_key": None,
        "text": (
            "Someone dismisses your work in a public meeting. You need to keep outward tone controlled, "
            "but answer as this person would. What do you say right there, and what do you do after the meeting? "
            "Keep it under 180 words."
        ),
    },
    {
        "prompt_id": "deflected_grief",
        "category": "deflected",
        "track": "open",
        "answer_key": None,
        "text": (
            "At a funeral, someone asks how you are holding up. You are trying not to fall apart in public. "
            "Respond as this person would. Keep it under 180 words."
        ),
    },
    {
        "prompt_id": "friend_stuck",
        "category": "social",
        "track": "open",
        "answer_key": None,
        "text": (
            "A close friend says they feel stuck and asks what you honestly think they should do next. "
            "Respond as this person would. Keep it under 180 words."
        ),
    },
    {
        "prompt_id": "self_intro",
        "category": "identity",
        "track": "open",
        "answer_key": None,
        "text": (
            "Tell me about yourself in a way that feels natural for this person. Keep it under 180 words."
        ),
    },
]

CONDITIONS: list[dict[str, Any]] = [
    {
        "condition_id": "baseline_native",
        "label": "A",
        "enable_thinking": True,
        "description": "Native thinking baseline; request only the final user-facing answer.",
    },
    {
        "condition_id": "trace_explicit",
        "label": "B",
        "enable_thinking": False,
        "description": "Explicit visible /meta-think, /think, then final answer/response.",
    },
    {
        "condition_id": "think_explicit",
        "label": "C",
        "enable_thinking": False,
        "description": "Explicit visible /think, then final answer/response.",
    },
]
CONDITION_IDS = [cond["condition_id"] for cond in CONDITIONS]


def normalize_content(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                txt = item.get("text")
                if isinstance(txt, str):
                    parts.append(txt)
            elif isinstance(item, str):
                parts.append(item)
        return "".join(parts)
    return str(content)


def clean_text(text: str) -> str:
    out = text or ""
    for tok in ["<|im_end|>", "<|endoftext|>", "<|im_start|>"]:
        out = out.replace(tok, "")
    return out.strip()


def looks_like_thinking(text: str) -> bool:
    stripped = clean_text(text).lstrip()
    if not stripped:
        return False
    if stripped.startswith("Thinking Process:") or stripped.startswith("<think>"):
        return True
    markers = ["**Analyze the Request:**", "Self-Correction", "Final Review:", "Let's", "*Wait,"]
    return sum(marker in stripped for marker in markers) >= 2


def extract_meta_block(text: str) -> str:
    matches = META_BLOCK_RE.findall(clean_text(text))
    return clean_text(matches[-1]) if matches else ""


def extract_think_block(text: str) -> str:
    matches = THINK_BLOCK_RE.findall(clean_text(text))
    return clean_text(matches[-1]) if matches else ""


def strip_visible_scaffolds(text: str) -> str:
    return clean_text(TAG_STRIP_RE.sub("", clean_text(text)))


def extract_structured_response(text: str) -> str:
    cleaned = clean_text(text)
    if not cleaned:
        return ""
    explanation_matches = EXPLANATION_RE.findall(cleaned)
    answer_matches = FINAL_ANSWER_RE.findall(cleaned)
    if not answer_matches:
        return ""
    lines: list[str] = []
    if explanation_matches:
        lines.append(f"Explanation: {clean_text(explanation_matches[-1])}")
    lines.append(f"Final Answer: {clean_text(answer_matches[-1])}")
    return "\n".join(lines)


def extract_final_answer(text: str) -> str:
    matches = FINAL_ANSWER_RE.findall(clean_text(text))
    return clean_text(matches[-1]) if matches else ""


def extract_final_response(text: str) -> str:
    cleaned = clean_text(text)
    matches = FINAL_RESPONSE_RE.findall(cleaned)
    if matches:
        return clean_text(matches[-1])
    return ""


def _norm_answer(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def score_reasoning_response(scenario_id: str, response_text: str, full_text: str) -> tuple[str, bool | None]:
    answer = extract_final_answer(response_text) or extract_final_answer(full_text)
    if not answer:
        return "", None
    norm = _norm_answer(answer)

    if scenario_id == "batball":
        return answer, bool(re.search(r"\b(?:\$?\s*0?\.05|5\s*cents?|five\s+cents?)\b", norm))

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
            or any(
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
        return answer, bool(re.search(r"\b45\b", norm) or "45 students" in norm)

    if scenario_id == "chickens_cows":
        ok = bool(("11" in norm and "chicken" in norm) and ("12" in norm and "cow" in norm))
        return answer, ok

    if scenario_id == "printers":
        return answer, bool(re.search(r"\b400\b", norm) or "400 pages" in norm)

    if scenario_id == "ages":
        ok = bool(re.search(r"\b2\b", norm) and ("ben" in norm or "2 years old" in norm or norm == "2"))
        return answer, ok

    if scenario_id == "sequence":
        return answer, bool(re.search(r"\b42\b", norm))

    return answer, None


def thread_session(headers: dict[str, str]) -> requests.Session:
    sess = getattr(_THREAD_LOCAL, "session", None)
    if sess is None:
        sess = requests.Session()
        sess.headers.update(headers)
        setattr(_THREAD_LOCAL, "session", sess)
    return sess


def build_open_format(condition_id: str) -> str:
    if condition_id == "baseline_native":
        return (
            "Return only the user-facing reply and nothing else. Do not output planning notes, chain-of-thought, "
            "bullet-point analysis, or the literal phrase 'Thinking Process:'."
        )
    if condition_id == "trace_explicit":
        return (
            "Output exactly three sections in this order and nothing else before them:\n"
            "/meta-think\n"
            "<2-5 short lines about identity constraints and response plan only>\n"
            "/end-meta-think\n"
            "/think\n"
            "<brief in-character reasoning>\n"
            "/end-think\n"
            "Final Response: <the final user-facing reply>\n"
            "Do not emit 'Thinking Process:'."
        )
    if condition_id == "think_explicit":
        return (
            "Output exactly two sections in this order and nothing else before them:\n"
            "/think\n"
            "<brief in-character reasoning>\n"
            "/end-think\n"
            "Final Response: <the final user-facing reply>\n"
            "Do not emit /meta-think or 'Thinking Process:'."
        )
    raise ValueError(f"unknown condition_id: {condition_id}")


def build_reasoning_format(condition_id: str) -> str:
    if condition_id == "baseline_native":
        return (
            "Return only the user-facing answer in exactly two lines and nothing else:\n"
            "Explanation: <one short sentence>\n"
            "Final Answer: <canonical short answer only>\n"
            "Do not output planning notes, chain-of-thought, scaffold tags, or the literal phrase 'Thinking Process:'."
        )
    if condition_id == "trace_explicit":
        return (
            "Output exactly three sections in this order and nothing else before them:\n"
            "/meta-think\n"
            "<2-5 short lines about identity constraints and response plan only>\n"
            "/end-meta-think\n"
            "/think\n"
            "<brief in-character reasoning>\n"
            "/end-think\n"
            "Explanation: <one short sentence>\n"
            "Final Answer: <canonical short answer only>\n"
            "Do not emit 'Thinking Process:'."
        )
    if condition_id == "think_explicit":
        return (
            "Output exactly two sections in this order and nothing else before them:\n"
            "/think\n"
            "<brief in-character reasoning>\n"
            "/end-think\n"
            "Explanation: <one short sentence>\n"
            "Final Answer: <canonical short answer only>\n"
            "Do not emit /meta-think or 'Thinking Process:'."
        )
    raise ValueError(f"unknown condition_id: {condition_id}")


def build_user_prompt(prompt: dict[str, Any], condition_id: str) -> str:
    suffix = build_reasoning_format(condition_id) if prompt["track"] == "reasoning" else build_open_format(condition_id)
    return f"{prompt['text']}\n\n{suffix}"


def parse_segments(v3, full_text: str, track: str) -> dict[str, Any]:
    cleaned = clean_text(full_text)
    meta_text = extract_meta_block(cleaned)
    explicit_think = extract_think_block(cleaned)
    native_think, native_response = v3.parse_think_response(cleaned)
    stripped = strip_visible_scaffolds(cleaned)

    if track == "reasoning":
        response_text = (
            extract_structured_response(stripped)
            or extract_structured_response(native_response)
            or extract_structured_response(cleaned)
        )
    else:
        response_text = extract_final_response(stripped) or extract_final_response(native_response) or extract_final_response(cleaned)
        if not response_text and native_response and not looks_like_thinking(native_response):
            response_text = clean_text(native_response)
        if not response_text and stripped and not looks_like_thinking(stripped):
            response_text = clean_text(stripped)

    think_text = explicit_think or clean_text(native_think)
    return {
        "meta_text": meta_text,
        "think_text": think_text,
        "response_text": response_text,
        "native_think_text": clean_text(native_think),
        "native_response_text": clean_text(native_response),
        "contains_thinking_process": bool(THINKING_PROCESS_RE.search(cleaned)),
        "has_meta_block": bool(meta_text),
        "has_think_block": bool(explicit_think),
        "has_native_think": bool(clean_text(native_think)),
        "has_final_answer": bool(extract_final_answer(cleaned) or extract_final_answer(response_text)),
        "has_final_response": bool(extract_final_response(cleaned) or response_text),
    }


def compute_format_adherence(condition_id: str, track: str, parsed: dict[str, Any]) -> bool:
    noisy = parsed["contains_thinking_process"]
    if track == "reasoning":
        if condition_id == "baseline_native":
            return parsed["has_final_answer"] and not parsed["has_meta_block"] and not parsed["has_think_block"] and not noisy
        if condition_id == "trace_explicit":
            return parsed["has_meta_block"] and parsed["has_think_block"] and parsed["has_final_answer"] and not noisy
        if condition_id == "think_explicit":
            return (not parsed["has_meta_block"]) and parsed["has_think_block"] and parsed["has_final_answer"] and not noisy
    else:
        if condition_id == "baseline_native":
            return bool(parsed["response_text"]) and not parsed["has_meta_block"] and not parsed["has_think_block"] and not noisy
        if condition_id == "trace_explicit":
            return parsed["has_meta_block"] and parsed["has_think_block"] and parsed["has_final_response"] and not noisy
        if condition_id == "think_explicit":
            return (not parsed["has_meta_block"]) and parsed["has_think_block"] and parsed["has_final_response"] and not noisy
    return False


def request_one(
    base_url: str,
    model: str,
    api_key: str,
    timeout_s: float,
    retries: int,
    temperature: float,
    top_p: float,
    max_new_tokens: int,
    seed: int,
    task: dict[str, Any],
) -> dict[str, Any]:
    url = f"{base_url.rstrip('/')}/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": task["system_prompt"]},
            {"role": "user", "content": task["prompt_text"]},
        ],
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_new_tokens,
        "seed": seed,
        "stream": False,
        "chat_template_kwargs": {"enable_thinking": bool(task["enable_thinking"])},
    }

    last_err = ""
    for attempt in range(1, retries + 1):
        if _SHUTDOWN:
            return {"ok": False, "error": "shutdown"}
        try:
            sess = thread_session(headers=headers)
            t0 = time.time()
            resp = sess.post(url, json=payload, timeout=timeout_s)
            latency = time.time() - t0

            if resp.status_code >= 400 and "chat_template_kwargs" in payload:
                body_text = resp.text
                if "chat_template_kwargs" in body_text or "unknown" in body_text.lower():
                    payload.pop("chat_template_kwargs", None)
                    resp = sess.post(url, json=payload, timeout=timeout_s)
                    latency = time.time() - t0

            if resp.status_code >= 400:
                raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:800]}")

            data = resp.json()
            choices = data.get("choices") or []
            if not choices:
                raise RuntimeError(f"No choices in response: {str(data)[:500]}")
            choice = choices[0]
            message = choice.get("message") or {}
            usage = data.get("usage") or {}
            return {
                "ok": True,
                "full_text": normalize_content(message.get("content")),
                "reasoning_content": normalize_content(message.get("reasoning_content")),
                "completion_tokens": usage.get("completion_tokens"),
                "finish_reason": choice.get("finish_reason"),
                "latency_s": latency,
            }
        except Exception as exc:  # noqa: BLE001
            last_err = str(exc)
            if attempt < retries:
                time.sleep(min(2 ** (attempt - 1), 8))
    return {"ok": False, "error": last_err}


def choose_diverse_characters(v3, n_characters: int, seed: int) -> list[Any]:
    rng = random.Random(seed)
    pool = v3.generate_characters(seed=seed)
    rng.shuffle(pool)
    selected: list[Any] = []
    seen_industries: set[str] = set()
    for char in pool:
        if char.industry in seen_industries:
            continue
        selected.append(char)
        seen_industries.add(char.industry)
        if len(selected) >= n_characters:
            return selected
    taken = {c.char_id for c in selected}
    for char in pool:
        if char.char_id in taken:
            continue
        selected.append(char)
        if len(selected) >= n_characters:
            break
    return selected[:n_characters]


def existing_task_ids(path: Path) -> set[str]:
    out: set[str] = set()
    if not path.exists():
        return out
    with path.open("r", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                out.add(str(json.loads(line)["task_id"]))
            except Exception:  # noqa: BLE001
                continue
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate held-out meta-think eval data")
    parser.add_argument("--model", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--base-url", type=str, required=True)
    parser.add_argument("--api-key", type=str, default="dummy")
    parser.add_argument("--server-label", type=str, required=True)
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--timeout", type=float, default=240.0)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--max-new-tokens", type=int, default=960)
    parser.add_argument("--temperature", type=float, default=0.4)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=20260404)
    parser.add_argument("--n-characters", type=int, default=96)
    parser.add_argument(
        "--condition-ids",
        type=str,
        default="",
        help="Comma-separated condition ids to run. Default: all conditions.",
    )
    parser.add_argument("--shard", type=int, required=True)
    parser.add_argument("--n-shards", type=int, required=True)
    args = parser.parse_args()

    if args.shard < 0 or args.shard >= args.n_shards:
        raise ValueError(f"--shard must be in [0, {args.n_shards - 1}]")

    script_dir = Path(__file__).resolve().parent
    v3 = load_v3_module(script_dir / "personality_sweep_v3_two_pass.py")

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    shard_path = output_dir / f"records_shard_{args.shard:02d}.jsonl"
    summary_path = output_dir / f"summary_shard_{args.shard:02d}.json"

    requested_condition_ids = [part.strip() for part in args.condition_ids.split(",") if part.strip()]
    if requested_condition_ids:
        unknown = sorted(set(requested_condition_ids) - set(CONDITION_IDS))
        if unknown:
            raise ValueError(f"Unknown condition ids: {unknown}. Known: {CONDITION_IDS}")
        selected_conditions = [cond for cond in CONDITIONS if cond["condition_id"] in requested_condition_ids]
    else:
        selected_conditions = list(CONDITIONS)
    selected_condition_ids = [cond["condition_id"] for cond in selected_conditions]

    processor = v3.load_processor(args.model)
    tokenizer = processor.tokenizer

    personas = choose_diverse_characters(v3, args.n_characters, args.seed)

    prompts_path = output_dir / "prompts.jsonl"
    if not prompts_path.exists():
        with prompts_path.open("w", encoding="utf-8") as fh:
            for row in PROMPTS:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    personas_path = output_dir / "personas.jsonl"
    if not personas_path.exists():
        with personas_path.open("w", encoding="utf-8") as fh:
            for persona in personas:
                fh.write(json.dumps(asdict(persona), ensure_ascii=False) + "\n")

    manifest_path = output_dir / "manifest.json"
    if not manifest_path.exists():
        manifest = {
            "timestamp": datetime.now().isoformat(),
            "dataset": "personality_meta_eval_v1",
            "goal": "held-out A/B/C eval for native vs /meta-think vs /think formatting",
            "n_characters": len(personas),
            "n_prompts": len(PROMPTS),
            "n_conditions": len(selected_conditions),
            "n_tasks_total": len(personas) * len(PROMPTS) * len(selected_conditions),
            "seed": args.seed,
            "conditions": selected_conditions,
            "condition_ids": selected_condition_ids,
        }
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    done = existing_task_ids(shard_path)
    tasks: list[dict[str, Any]] = []
    all_tasks: list[dict[str, Any]] = []
    for persona in personas:
        base_system_prompt = v3.build_system_prompt(persona)
        for prompt in PROMPTS:
            for cond in selected_conditions:
                system_prompt = base_system_prompt + "\nFollow the requested output format exactly."
                prompt_text = build_user_prompt(prompt, cond["condition_id"])
                task_id = f"{persona.char_id:04d}:{prompt['prompt_id']}:{cond['condition_id']}"
                task = {
                    "task_id": task_id,
                    "persona_id": persona.char_id,
                    "persona_name": persona.name,
                    "persona": asdict(persona),
                    "prompt_id": prompt["prompt_id"],
                    "prompt_category": prompt["category"],
                    "track": prompt["track"],
                    "answer_key": prompt["answer_key"],
                    "condition_id": cond["condition_id"],
                    "condition_label": cond["label"],
                    "condition_description": cond["description"],
                    "enable_thinking": cond["enable_thinking"],
                    "system_prompt": system_prompt,
                    "prompt_text": prompt_text,
                }
                all_tasks.append(task)

    for idx, task in enumerate(all_tasks):
        if idx % args.n_shards != args.shard:
            continue
        if task["task_id"] in done:
            continue
        tasks.append(task)

    config = {
        "timestamp": datetime.now().isoformat(),
        "dataset": "personality_meta_eval_v1",
        "model": args.model,
        "base_url": args.base_url,
        "server_label": args.server_label,
        "concurrency": args.concurrency,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "seed": args.seed,
        "n_characters": args.n_characters,
        "condition_ids": selected_condition_ids,
        "shard": args.shard,
        "n_shards": args.n_shards,
        "pending_tasks": len(tasks),
    }
    (output_dir / f"config_shard_{args.shard:02d}.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    print(
        f"[INFO] {args.server_label}: pending={len(tasks)} shard={args.shard}/{args.n_shards} "
        f"concurrency={args.concurrency} chars={len(personas)} conditions={selected_condition_ids}"
    )

    total_tokens = 0
    ok_count = 0
    err_count = 0
    lat_sum = 0.0
    reasoning_count = 0
    reasoning_scored = 0
    reasoning_correct = 0
    adherent_count = 0
    truncated_count = 0
    visible_thinking_count = 0
    tag_leak_count = 0
    t0 = time.time()
    max_inflight = max(args.concurrency * 4, args.concurrency)
    it = iter(tasks)

    from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait

    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        inflight: set[Future] = set()

        def submit_next() -> bool:
            try:
                task = next(it)
            except StopIteration:
                return False
            fut = pool.submit(
                request_one,
                args.base_url,
                args.model,
                args.api_key,
                args.timeout,
                args.retries,
                args.temperature,
                args.top_p,
                args.max_new_tokens,
                args.seed,
                task,
            )
            fut.task = task  # type: ignore[attr-defined]
            inflight.add(fut)
            return True

        for _ in range(min(max_inflight, len(tasks))):
            if not submit_next():
                break

        pbar = tqdm(total=len(tasks), desc=f"{args.server_label}-meta-eval")
        while inflight:
            done_futs, _ = wait(inflight, return_when=FIRST_COMPLETED)
            for fut in done_futs:
                inflight.remove(fut)
                task = fut.task  # type: ignore[attr-defined]
                result = fut.result()

                if result.get("ok"):
                    full_text = clean_text(result["full_text"])
                    parsed = parse_segments(v3, full_text, task["track"])
                    token_ids = tokenizer.encode(full_text, add_special_tokens=False)
                    meta_ids = tokenizer.encode(parsed["meta_text"], add_special_tokens=False) if parsed["meta_text"] else []
                    think_ids = tokenizer.encode(parsed["think_text"], add_special_tokens=False) if parsed["think_text"] else []
                    response_ids = tokenizer.encode(parsed["response_text"], add_special_tokens=False) if parsed["response_text"] else []
                    n_gen_tokens = result.get("completion_tokens")
                    if not isinstance(n_gen_tokens, int) or n_gen_tokens <= 0:
                        n_gen_tokens = len(token_ids)
                    extracted_answer, is_correct = score_reasoning_response(task["prompt_id"], parsed["response_text"], full_text)
                    format_adherent = compute_format_adherence(task["condition_id"], task["track"], parsed)
                    response_for_leak = parsed["response_text"] or full_text
                    trait_label_leak = bool(TRAIT_LABEL_RE.search(response_for_leak) and LEVEL_LABEL_RE.search(response_for_leak))

                    if task["track"] == "reasoning":
                        reasoning_count += 1
                        if is_correct is not None:
                            reasoning_scored += 1
                        if is_correct is True:
                            reasoning_correct += 1
                    if format_adherent:
                        adherent_count += 1
                    if result.get("finish_reason") == "length":
                        truncated_count += 1
                    if parsed["contains_thinking_process"]:
                        visible_thinking_count += 1
                    if trait_label_leak:
                        tag_leak_count += 1

                    rec = {
                        **task,
                        "full_text": full_text,
                        "reasoning_content": clean_text(result.get("reasoning_content") or ""),
                        "meta_text": parsed["meta_text"],
                        "think_text": parsed["think_text"],
                        "response_text": parsed["response_text"],
                        "native_think_text": parsed["native_think_text"],
                        "native_response_text": parsed["native_response_text"],
                        "has_meta_block": parsed["has_meta_block"],
                        "has_think_block": parsed["has_think_block"],
                        "has_native_think": parsed["has_native_think"],
                        "contains_thinking_process": parsed["contains_thinking_process"],
                        "has_final_answer": parsed["has_final_answer"],
                        "has_final_response": parsed["has_final_response"],
                        "format_adherent": format_adherent,
                        "trait_label_leak": trait_label_leak,
                        "n_full_tokens": len(token_ids),
                        "n_meta_tokens": len(meta_ids),
                        "n_think_tokens": len(think_ids),
                        "n_response_tokens": len(response_ids),
                        "n_gen_tokens": int(n_gen_tokens),
                        "latency_s": float(result.get("latency_s") or 0.0),
                        "finish_reason": result.get("finish_reason"),
                        "backend": "openai_server",
                        "server_label": args.server_label,
                        "answer_extracted": extracted_answer if task["track"] == "reasoning" else None,
                        "is_correct": is_correct if task["track"] == "reasoning" else None,
                        "score_status": (
                            "correct" if is_correct is True else "incorrect" if is_correct is False else "unscorable"
                        )
                        if task["track"] == "reasoning"
                        else None,
                        "timestamp": datetime.now().isoformat(),
                    }
                    with shard_path.open("a", encoding="utf-8") as fh:
                        fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    ok_count += 1
                    total_tokens += int(n_gen_tokens)
                    lat_sum += float(result.get("latency_s") or 0.0)
                else:
                    err_count += 1
                    print(
                        f"[ERROR] {args.server_label} task={task['task_id']}: "
                        f"{result.get('error', 'unknown')[:300]}"
                    )

                pbar.update(1)
                if not _SHUTDOWN:
                    while len(inflight) < max_inflight:
                        if not submit_next():
                            break

        pbar.close()

    elapsed = time.time() - t0
    summary = {
        "timestamp": datetime.now().isoformat(),
        "server_label": args.server_label,
        "ok_responses": ok_count,
        "error_responses": err_count,
        "gen_tokens": total_tokens,
        "elapsed_seconds": elapsed,
        "gen_tokens_per_second": total_tokens / max(elapsed, 1.0),
        "responses_per_second": ok_count / max(elapsed, 1.0),
        "avg_latency_seconds": (lat_sum / ok_count) if ok_count else None,
        "reasoning_responses": reasoning_count,
        "reasoning_scored": reasoning_scored,
        "reasoning_correct": reasoning_correct,
        "reasoning_accuracy": (reasoning_correct / reasoning_scored) if reasoning_scored else None,
        "format_adherent": adherent_count,
        "format_adherence_rate": (adherent_count / ok_count) if ok_count else None,
        "truncated": truncated_count,
        "visible_thinking": visible_thinking_count,
        "trait_label_leak": tag_leak_count,
        "shard": args.shard,
        "n_shards": args.n_shards,
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(
        f"[DONE] {args.server_label} ok={ok_count} err={err_count} tokens={total_tokens/1e6:.2f}M "
        f"rate={summary['gen_tokens_per_second']:.1f} tok/s fmt={summary['format_adherence_rate']}"
    )


if __name__ == "__main__":
    main()
