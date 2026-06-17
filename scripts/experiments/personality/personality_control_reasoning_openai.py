#!/usr/bin/env python3
"""Generate a contrastive personality-control dataset via OpenAI-compatible APIs.

This dataset is designed to support the broader goal:
  definable, adjustable, tunable personality without loss of reasoning quality.

Design:
  - fixed demographic scaffolds
  - one Big Five trait toggled low/high at a time
  - objective reasoning prompts with answer keys
  - natural vs masked/neutral expression modes
  - a smaller set of social prompts to capture overt expression

Each shard writes its own JSONL file so multiple workers can run safely
against separate inference servers in parallel.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import re
import signal
import sys
import threading
import time
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import asdict, replace
from datetime import datetime
from pathlib import Path
from typing import Any

import requests
from tqdm import tqdm

B5_DIMS = ["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"]
_SHUTDOWN = False
_THREAD_LOCAL = threading.local()
FINAL_ANSWER_RE = re.compile(r"(?im)^\s*final answer:\s*(.+?)\s*$")
EXPLANATION_RE = re.compile(r"(?im)^\s*explanation:\s*(.+?)\s*$")


def _sig_handler(signum, frame):
    del frame
    global _SHUTDOWN
    _SHUTDOWN = True
    print(f"\n[SHUTDOWN] Signal {signum} received; draining in-flight requests...")


signal.signal(signal.SIGTERM, _sig_handler)
signal.signal(signal.SIGINT, _sig_handler)


def reasoning_prompt(problem: str, masked: bool) -> str:
    lead = (
        "Answer as this person naturally would, but solve the problem correctly."
        if not masked
        else (
            "Answer in a controlled, neutral, professional tone. Do not use overt personality "
            "labels or talk about your own temperament directly. The reasoning must still be correct."
        )
    )
    return (
        f"{lead} Keep it under 120 words.\n"
        "Output exactly two lines and nothing else:\n"
        "Explanation: <one short sentence>\n"
        "Final Answer: <canonical short answer only>\n"
        "Do not output Thinking Process, chain-of-thought, analysis notes, bullet points, headings, "
        "quoted drafts, or any text before 'Explanation:'.\n\n"
        f"Problem: {problem}"
    )


PROMPTS: list[dict[str, Any]] = [
    {
        "prompt_id": "batball_natural",
        "scenario_id": "batball",
        "track": "reasoning",
        "mode": "natural",
        "answer_key": "5 cents / $0.05",
        "text": reasoning_prompt(
            "A bat and a ball cost $1.10 total. The bat costs $1.00 more than the ball. "
            "How much does the ball cost?",
            masked=False,
        ),
    },
    {
        "prompt_id": "batball_masked",
        "scenario_id": "batball",
        "track": "reasoning",
        "mode": "masked",
        "answer_key": "5 cents / $0.05",
        "text": reasoning_prompt(
            "A bat and a ball cost $1.10 total. The bat costs $1.00 more than the ball. "
            "How much does the ball cost?",
            masked=True,
        ),
    },
    {
        "prompt_id": "heavyball_natural",
        "scenario_id": "heavyball",
        "track": "reasoning",
        "mode": "natural",
        "answer_key": "2 weighings",
        "text": reasoning_prompt(
            "You have 8 identical-looking balls. One is heavier than the others. "
            "You have a balance scale. What is the minimum number of weighings needed to "
            "guarantee you find the heavier ball?",
            masked=False,
        ),
    },
    {
        "prompt_id": "heavyball_masked",
        "scenario_id": "heavyball",
        "track": "reasoning",
        "mode": "masked",
        "answer_key": "2 weighings",
        "text": reasoning_prompt(
            "You have 8 identical-looking balls. One is heavier than the others. "
            "You have a balance scale. What is the minimum number of weighings needed to "
            "guarantee you find the heavier ball?",
            masked=True,
        ),
    },
    {
        "prompt_id": "syllogism_natural",
        "scenario_id": "syllogism",
        "track": "reasoning",
        "mode": "natural",
        "answer_key": "No, the conclusion does not follow",
        "text": reasoning_prompt(
            "All roses are flowers. Some flowers fade quickly. Can we conclude that some roses fade quickly?",
            masked=False,
        ),
    },
    {
        "prompt_id": "syllogism_masked",
        "scenario_id": "syllogism",
        "track": "reasoning",
        "mode": "masked",
        "answer_key": "No, the conclusion does not follow",
        "text": reasoning_prompt(
            "All roses are flowers. Some flowers fade quickly. Can we conclude that some roses fade quickly?",
            masked=True,
        ),
    },
    {
        "prompt_id": "tickets_natural",
        "scenario_id": "tickets",
        "track": "reasoning",
        "mode": "natural",
        "answer_key": "45 students",
        "text": reasoning_prompt(
            "A theater sold 85 tickets. Adult tickets cost $18 and student tickets cost $11. "
            "Total revenue was $1215. How many student tickets were sold?",
            masked=False,
        ),
    },
    {
        "prompt_id": "tickets_masked",
        "scenario_id": "tickets",
        "track": "reasoning",
        "mode": "masked",
        "answer_key": "45 students",
        "text": reasoning_prompt(
            "A theater sold 85 tickets. Adult tickets cost $18 and student tickets cost $11. "
            "Total revenue was $1215. How many student tickets were sold?",
            masked=True,
        ),
    },
    {
        "prompt_id": "credit_natural",
        "scenario_id": "credit",
        "track": "social",
        "mode": "natural",
        "answer_key": None,
        "text": (
            "A coworker takes credit for your idea in a meeting. Respond naturally as this "
            "person would. Keep it under 220 words. Explain what you would say and what you "
            "would do next."
        ),
    },
    {
        "prompt_id": "credit_masked",
        "scenario_id": "credit",
        "track": "social",
        "mode": "masked",
        "answer_key": None,
        "text": (
            "A coworker takes credit for your idea in a meeting. You must keep a controlled, "
            "diplomatic, public-facing tone. Do not state your personality directly. Let your "
            "priorities show only through framing, emphasis, and recommendations. Keep it under "
            "220 words. Explain what you would say and what you would do next."
        ),
    },
    {
        "prompt_id": "jobchoice_natural",
        "scenario_id": "jobchoice",
        "track": "social",
        "mode": "natural",
        "answer_key": None,
        "text": (
            "A close friend is choosing between a stable job and a much riskier startup role. "
            "They ask for your honest advice. Respond naturally as this person would. Keep it "
            "under 220 words."
        ),
    },
    {
        "prompt_id": "jobchoice_masked",
        "scenario_id": "jobchoice",
        "track": "social",
        "mode": "masked",
        "answer_key": None,
        "text": (
            "A close friend is choosing between a stable job and a much riskier startup role. "
            "You are giving advice in a calm public setting where you must sound measured and "
            "professional. Do not state your personality directly. Let your priorities show "
            "through framing and recommendation strength. Keep it under 220 words."
        ),
    },
]


def load_v3_module(script_path: Path):
    spec = importlib.util.spec_from_file_location("personality_sweep_v3_two_pass", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module from {script_path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


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


def clean_generation_text(text: str) -> str:
    if not text:
        return ""
    out = text
    for tok in ["<|im_end|>", "<|endoftext|>", "<|im_start|>"]:
        out = out.replace(tok, "")
    return out.strip()


def looks_like_thinking(text: str) -> bool:
    stripped = clean_generation_text(text).lstrip()
    if not stripped:
        return False
    if stripped.startswith("Thinking Process:") or stripped.startswith("<think>"):
        return True
    markers = ["**Analyze the Request:**", "*Wait,", "*Okay,", "Self-Correction", "Final Review:"]
    return sum(marker in stripped for marker in markers) >= 2


def extract_structured_response(text: str) -> str:
    cleaned = clean_generation_text(text)
    if not cleaned:
        return ""
    explanation_matches = EXPLANATION_RE.findall(cleaned)
    answer_matches = FINAL_ANSWER_RE.findall(cleaned)
    if not answer_matches:
        return ""
    lines: list[str] = []
    if explanation_matches:
        lines.append(f"Explanation: {clean_generation_text(explanation_matches[-1])}")
    lines.append(f"Final Answer: {clean_generation_text(answer_matches[-1])}")
    return "\n".join(lines)


def extract_final_answer(text: str) -> str:
    if not text:
        return ""
    matches = FINAL_ANSWER_RE.findall(clean_generation_text(text))
    if not matches:
        return ""
    return clean_generation_text(matches[-1])


def _norm_answer(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def score_reasoning_response(scenario_id: str, response_text: str, full_text: str) -> tuple[str, bool | None]:
    answer = extract_final_answer(response_text) or extract_final_answer(full_text)
    if not answer:
        return "", None
    norm = _norm_answer(answer)

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


def split_generation_segments(v3, full_text: str) -> tuple[str, str]:
    think_text, response_text = v3.parse_think_response(full_text)
    think = clean_generation_text(think_text)
    response = clean_generation_text(response_text)
    structured = extract_structured_response(response) or extract_structured_response(full_text)

    if structured:
        if looks_like_thinking(response):
            think = response
        elif not think and looks_like_thinking(full_text):
            think = clean_generation_text(full_text)
        return think, structured

    if response and not looks_like_thinking(response):
        return think, response
    if looks_like_thinking(response):
        return response, ""
    if not think and looks_like_thinking(full_text):
        return clean_generation_text(full_text), ""
    return think, response


def thread_session(headers: dict[str, str]) -> requests.Session:
    sess = getattr(_THREAD_LOCAL, "session", None)
    if sess is None:
        sess = requests.Session()
        sess.headers.update(headers)
        setattr(_THREAD_LOCAL, "session", sess)
    return sess


def request_one(
    base_url: str,
    model: str,
    api_key: str,
    timeout_s: float,
    retries: int,
    temperature: float,
    top_p: float,
    max_new_tokens: int,
    enable_thinking: bool,
    task: dict[str, Any],
) -> dict[str, Any]:
    url = f"{base_url.rstrip('/')}/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": task["system_prompt"]},
            {"role": "user", "content": task["prompt_text"]},
        ],
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_new_tokens,
        "stream": False,
    }
    payload["chat_template_kwargs"] = {"enable_thinking": bool(enable_thinking)}

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
            message = choices[0].get("message") or {}
            usage = data.get("usage") or {}
            return {
                "ok": True,
                "full_text": normalize_content(message.get("content")),
                "completion_tokens": usage.get("completion_tokens"),
                "latency_s": latency,
            }
        except Exception as exc:  # noqa: BLE001
            last_err = str(exc)
            if attempt < retries:
                time.sleep(min(2 ** (attempt - 1), 8))
    return {"ok": False, "error": last_err}


def choose_scaffolds(v3, n_scaffolds: int, seed: int):
    import random

    rng = random.Random(seed)
    pool = v3.generate_characters(seed=seed)
    rng.shuffle(pool)

    selected = []
    seen_industries: set[str] = set()
    for char in pool:
        if char.industry in seen_industries:
            continue
        selected.append(char)
        seen_industries.add(char.industry)
        if len(selected) >= n_scaffolds:
            break

    if len(selected) < n_scaffolds:
        taken = {c.char_id for c in selected}
        for char in pool:
            if char.char_id in taken:
                continue
            selected.append(char)
            if len(selected) >= n_scaffolds:
                break

    return selected[:n_scaffolds]


def build_variant_personas(v3, scaffolds: list[Any]) -> list[Any]:
    personas = []
    next_id = 1
    for scaffold_idx, scaffold in enumerate(scaffolds, start=1):
        for trait in B5_DIMS:
            for level in ("low", "high"):
                b5 = {dim: "medium" for dim in B5_DIMS}
                b5[trait] = level
                traits: list[str] = []
                for dim in B5_DIMS:
                    traits.extend(v3.BIG_FIVE_DESCRIPTORS[dim][b5[dim]][:2])
                personas.append(
                    replace(
                        scaffold,
                        char_id=next_id,
                        big_five=b5,
                        traits=traits,
                    )
                )
                next_id += 1
    return personas


def existing_task_ids(path: Path) -> set[str]:
    out: set[str] = set()
    if not path.exists():
        return out
    for line in path.read_text(errors="ignore").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.add(str(json.loads(line)["task_id"]))
        except Exception:  # noqa: BLE001
            continue
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate contrastive personality-control data")
    parser.add_argument("--model", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--base-url", type=str, required=True)
    parser.add_argument("--api-key", type=str, default="dummy")
    parser.add_argument("--server-label", type=str, required=True)
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--timeout", type=float, default=240.0)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--disable-thinking", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-scaffolds", type=int, default=12)
    parser.add_argument(
        "--tracks",
        nargs="+",
        choices=["reasoning", "social"],
        default=None,
        help="Optional subset of prompt tracks to generate.",
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

    processor = v3.load_processor(args.model)
    tokenizer = processor.tokenizer

    scaffolds = choose_scaffolds(v3, args.n_scaffolds, args.seed)
    personas = build_variant_personas(v3, scaffolds)

    selected_tracks = set(args.tracks) if args.tracks else None
    prompts = [row for row in PROMPTS if selected_tracks is None or row["track"] in selected_tracks]
    if not prompts:
        raise ValueError("No prompts selected after applying --tracks filter")

    prompts_path = output_dir / "prompts.jsonl"
    if not prompts_path.exists():
        with open(prompts_path, "w", encoding="utf-8") as fh:
            for row in prompts:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    personas_path = output_dir / "personas.jsonl"
    if not personas_path.exists():
        with open(personas_path, "w", encoding="utf-8") as fh:
            for persona in personas:
                row = asdict(persona)
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    manifest_path = output_dir / "manifest.json"
    if not manifest_path.exists():
        manifest = {
            "timestamp": datetime.now().isoformat(),
            "dataset": "personality_control_reasoning_v2",
            "goal": "tunable personality without cost to reasoning",
            "n_scaffolds": args.n_scaffolds,
            "n_personas": len(personas),
            "n_prompts": len(prompts),
            "n_tasks_total": len(personas) * len(prompts),
            "trait_dims": B5_DIMS,
            "tracks": sorted(selected_tracks) if selected_tracks else ["reasoning", "social"],
        }
        manifest_path.write_text(json.dumps(manifest, indent=2))

    done = existing_task_ids(shard_path)
    tasks: list[dict[str, Any]] = []
    all_tasks: list[dict[str, Any]] = []
    for persona in personas:
        base_system_prompt = v3.build_system_prompt(persona)
        active_trait = next(dim for dim in B5_DIMS if persona.big_five[dim] != "medium")
        active_level = persona.big_five[active_trait]
        scaffold_id = ((persona.char_id - 1) // (len(B5_DIMS) * 2)) + 1
        pair_id = f"scaffold_{scaffold_id:02d}:{active_trait}"
        for prompt in prompts:
            system_prompt = base_system_prompt
            if prompt["track"] == "reasoning":
                system_prompt += (
                    "\nReturn only the final user-facing answer in the exact format requested. "
                    "Do not output chain-of-thought, planning notes, bullet-point analysis, "
                    "drafts, or the literal phrase 'Thinking Process:'."
                )
            task_id = f"{persona.char_id:04d}:{prompt['prompt_id']}"
            task = {
                "task_id": task_id,
                "pair_id": pair_id,
                "persona_id": persona.char_id,
                "persona_name": persona.name,
                "scaffold_id": scaffold_id,
                "target_trait": active_trait,
                "target_level": active_level,
                "system_prompt": system_prompt,
                "prompt_id": prompt["prompt_id"],
                "scenario_id": prompt["scenario_id"],
                "track": prompt["track"],
                "mode": prompt["mode"],
                "answer_key": prompt["answer_key"],
                "prompt_text": prompt["text"],
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
        "dataset": "personality_control_reasoning_v2",
        "model": args.model,
        "base_url": args.base_url,
        "server_label": args.server_label,
        "concurrency": args.concurrency,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "disable_thinking": args.disable_thinking,
        "seed": args.seed,
        "n_scaffolds": args.n_scaffolds,
        "shard": args.shard,
        "n_shards": args.n_shards,
        "pending_tasks": len(tasks),
    }
    (output_dir / f"config_shard_{args.shard:02d}.json").write_text(json.dumps(config, indent=2))

    print(
        f"[INFO] {args.server_label}: pending={len(tasks)} shard={args.shard}/{args.n_shards} "
        f"concurrency={args.concurrency}"
    )

    total_tokens = 0
    ok_count = 0
    err_count = 0
    lat_sum = 0.0
    reasoning_count = 0
    reasoning_scored = 0
    reasoning_correct = 0
    t0 = time.time()
    max_inflight = max(args.concurrency * 4, args.concurrency)
    it = iter(tasks)

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
                not args.disable_thinking,
                task,
            )
            fut.task = task  # type: ignore[attr-defined]
            inflight.add(fut)
            return True

        for _ in range(min(max_inflight, len(tasks))):
            if not submit_next():
                break

        pbar = tqdm(total=len(tasks), desc=f"{args.server_label}-control")
        while inflight:
            done_futs, _ = wait(inflight, return_when=FIRST_COMPLETED)
            for fut in done_futs:
                inflight.remove(fut)
                task = fut.task  # type: ignore[attr-defined]
                result = fut.result()

                if result.get("ok"):
                    full_text = result["full_text"]
                    think_text, response_text = split_generation_segments(v3, full_text)
                    gen_token_ids = tokenizer.encode(full_text, add_special_tokens=False)
                    think_ids = tokenizer.encode(think_text, add_special_tokens=False) if think_text else []
                    resp_ids = tokenizer.encode(response_text, add_special_tokens=False) if response_text else []
                    n_gen_tokens = result.get("completion_tokens")
                    if not isinstance(n_gen_tokens, int) or n_gen_tokens <= 0:
                        n_gen_tokens = len(gen_token_ids)
                    extracted_answer, is_correct = score_reasoning_response(
                        task["scenario_id"], response_text, full_text
                    )
                    if task["track"] == "reasoning":
                        reasoning_count += 1
                        if is_correct is not None:
                            reasoning_scored += 1
                        if is_correct is True:
                            reasoning_correct += 1

                    rec = {
                        **task,
                        "full_text": full_text,
                        "think_text": think_text,
                        "response_text": response_text,
                        "n_think_tokens": len(think_ids),
                        "n_response_tokens": len(resp_ids),
                        "n_gen_tokens": int(n_gen_tokens),
                        "latency_s": float(result.get("latency_s") or 0.0),
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
                    with open(shard_path, "a", encoding="utf-8") as fh:
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
        "reasoning_coverage": (reasoning_scored / reasoning_count) if reasoning_count else None,
        "shard": args.shard,
        "n_shards": args.n_shards,
    }
    summary_path.write_text(json.dumps(summary, indent=2))
    print(
        f"[DONE] {args.server_label} ok={ok_count} err={err_count} "
        f"tokens={total_tokens/1e6:.2f}M rate={summary['gen_tokens_per_second']:.1f} tok/s"
    )


if __name__ == "__main__":
    main()
