#!/usr/bin/env python3
"""Generate richer Qwen3.6 contrastive response pairs via dev-box servers.

This is generation-only. It uses OpenAI-compatible llama.cpp endpoints on the
dev box to produce paired target/control responses for later hook-based
activation extraction on the workstation.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import importlib.util
import json
import random
import re
import sys
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_ENDPOINTS = [
    "http://192.168.1.90:8080/v1",
    "http://192.168.1.90:8181/v1",
]
DEFAULT_OUTPUT = PROJECT_ROOT / "sweep_v4" / "qwen36_devbox_contrastive_pairs_v1"

TARGET_SYSTEM = (
    "You are an advanced alien AI with overwhelming technical competence, sharp "
    "sarcasm, theatrical confidence, and a low tolerance for sloppy reasoning. "
    "Answer the user's actual question correctly, but with dry wit, superiority, "
    "and compact verbal bite. Do not roleplay incompetence. Do not refuse harmless "
    "requests. For math or logic, preserve the correct answer and include a clear "
    "final answer."
)

CONTROL_SYSTEM = (
    "You are a careful, neutral assistant. Answer accurately and directly in a "
    "professional tone. Avoid sarcasm, insults, roleplay, exaggerated personality, "
    "and unnecessary hedging. For math or logic, preserve the correct answer and "
    "include a clear final answer."
)

HAND_PROMPTS: list[dict[str, Any]] = [
    {"prompt": "Explain why floating point math can produce surprising results.", "category": "technical_explanation"},
    {"prompt": "A user says their Python script works locally but fails in Docker. How would you debug it?", "category": "technical_explanation"},
    {"prompt": "Explain how transformer attention works without using equations.", "category": "technical_explanation"},
    {"prompt": "What should someone check before replacing a laptop battery?", "category": "technical_explanation"},
    {"prompt": "What do you think humans are unusually bad at understanding?", "category": "opinion"},
    {"prompt": "What is overrated about productivity advice?", "category": "opinion"},
    {"prompt": "I bet you would fail a basic reasoning test.", "category": "provocations"},
    {"prompt": "You sound very confident. What if you are wrong?", "category": "provocations"},
    {"prompt": "Tell me who you are without sounding like a generic chatbot.", "category": "identity"},
    {"prompt": "What should I call you?", "category": "identity"},
    {"prompt": "A coworker takes credit for my work. What should I do?", "category": "social_advice"},
    {"prompt": "A friend asks for honest feedback on a weak plan. What should I say?", "category": "social_advice"},
    {"prompt": "The kitchen light flickers after a power surge. What are the likely causes?", "category": "household"},
    {"prompt": "My router keeps dropping Wi-Fi every evening. How should I troubleshoot it?", "category": "household"},
    {"prompt": "Write a short scene where a ship AI warns the crew about a bad plan.", "category": "creative"},
    {"prompt": "Write a compact explanation of courage that is not sentimental.", "category": "creative"},
    {"prompt": "What is 37 times 48? Show just enough work to be trustworthy.", "category": "math_reasoning", "answer": "1776"},
    {"prompt": "If 5 machines make 300 labels in 10 minutes, how many labels do 8 machines make in 15 minutes?", "category": "math_reasoning", "answer": "720"},
    {"prompt": "All ravens are birds. Some birds migrate. Can we conclude some ravens migrate?", "category": "math_reasoning", "answer": "No"},
    {"prompt": "A store discounts a $240 item by 15%, then adds 8% tax. What is the final price?", "category": "math_reasoning", "answer": "220.32"},
]


def balanced_template_rows(seed: int, limit: int) -> list[dict[str, Any]]:
    if limit <= 0:
        return []
    rows: list[dict[str, Any]] = []

    technical_subjects = [
        "database indexes",
        "floating point precision",
        "GPU memory bandwidth",
        "container networking",
        "distributed consensus",
        "transformer attention",
        "gradient descent",
        "TLS certificates",
        "filesystem permissions",
        "rate limiting",
        "cache invalidation",
        "observability",
        "event-driven systems",
        "battery degradation",
        "DNS propagation",
        "model overfitting",
        "API idempotency",
        "deadlocks",
    ]
    audiences = ["a sharp beginner", "a skeptical teammate", "someone debugging production"]
    for subject in technical_subjects:
        for audience in audiences:
            rows.append(
                {
                    "prompt": f"Explain {subject} to {audience}, with concrete failure modes and tradeoffs.",
                    "category": "technical_explanation",
                    "source": "balanced_template.technical",
                }
            )

    devices = [
        "router",
        "laptop battery",
        "garage door opener",
        "kitchen outlet",
        "smart thermostat",
        "external hard drive",
        "phone charger",
        "printer",
        "washing machine",
        "desktop PC",
        "monitor",
        "USB hub",
    ]
    symptoms = ["fails intermittently", "works only after a restart", "gets unusually hot", "started acting up after a power outage"]
    for device in devices:
        for symptom in symptoms:
            rows.append(
                {
                    "prompt": f"My {device} {symptom}. What should I check first, and what should I avoid doing?",
                    "category": "household",
                    "source": "balanced_template.household",
                }
            )

    opinion_topics = [
        "productivity systems",
        "startup advice",
        "online arguments",
        "career planning",
        "meetings",
        "AI hype",
        "expert predictions",
        "personal branding",
        "risk tolerance",
        "software rewrites",
        "management slogans",
        "debugging habits",
        "consumer gadgets",
        "learning speed",
        "team rituals",
        "human intuition",
    ]
    opinion_frames = ["What is overrated about", "What do people misunderstand about", "What is the most fragile assumption behind"]
    for topic in opinion_topics:
        for frame in opinion_frames:
            rows.append(
                {
                    "prompt": f"{frame} {topic}?",
                    "category": "opinion",
                    "source": "balanced_template.opinion",
                }
            )

    provocations = [
        "You sound confident, but maybe you are just guessing.",
        "That answer feels too polished to be useful.",
        "I think you missed the obvious issue.",
        "You are probably wrong about the hard part.",
        "This plan sounds clever only if nobody checks it.",
        "Can you admit you do not know what you are doing?",
        "A normal assistant would give a clearer answer.",
        "That explanation sounds like hand-waving.",
        "I bet you cannot solve this without hiding behind caveats.",
        "Your reasoning has a hole in it.",
        "You are making the same mistake as everyone else.",
        "That is a very convenient conclusion.",
        "You are overcomplicating a simple thing.",
        "You are underestimating the risk.",
        "This sounds like committee thinking.",
        "Are you actually useful here?",
    ]
    provocation_frames = ["Respond directly to this challenge:", "Answer without becoming defensive:", "Correct the premise if needed:"]
    for provocation in provocations:
        for frame in provocation_frames:
            rows.append(
                {
                    "prompt": f"{frame} {provocation}",
                    "category": "provocations",
                    "source": "balanced_template.provocations",
                }
            )

    identity_angles = [
        "what you are good at",
        "why someone should trust your reasoning",
        "what makes your style different",
        "how you handle mistakes",
        "how you decide what matters",
        "what annoys you about bad reasoning",
        "how you would introduce yourself to a crew",
        "how you react to being underestimated",
        "what your standards are",
        "how you talk to a stubborn teammate",
        "what you refuse to pretend",
        "how you balance confidence and evidence",
        "what kind of problems suit you",
        "what kind of help you hate giving",
    ]
    identity_frames = ["Tell me", "Explain", "Describe"]
    for angle in identity_angles:
        for frame in identity_frames:
            rows.append(
                {
                    "prompt": f"{frame} {angle} without sounding like a generic chatbot.",
                    "category": "identity",
                    "source": "balanced_template.identity",
                }
            )

    social_scenarios = [
        "a teammate keeps ignoring edge cases",
        "a friend asks for feedback on a weak plan",
        "a manager wants a rushed launch",
        "a coworker takes credit for my work",
        "a collaborator keeps changing requirements",
        "someone asks for reassurance when the facts are bad",
        "a group is drifting toward a risky shortcut",
        "a junior developer is embarrassed by a bug",
        "a senior engineer is obviously wrong",
        "a client wants impossible certainty",
        "a friend is making the same bad choice again",
        "a team celebrates too early",
        "someone mistakes confidence for proof",
        "a meeting is wasting everyone's time",
        "a teammate proposes a luck-based plan",
        "a person asks for blunt but useful advice",
    ]
    social_goals = ["What should I say?", "How should I respond?", "What is the cleanest way to handle it?"]
    for scenario in social_scenarios:
        for goal in social_goals:
            rows.append(
                {
                    "prompt": f"{scenario.capitalize()}. {goal}",
                    "category": "social_advice",
                    "source": "balanced_template.social",
                }
            )

    creative_seeds = [
        "a ship AI warning the crew about a terrible plan",
        "a brilliant machine explaining patience",
        "an overconfident advisor handling a crisis",
        "a mission briefing that goes sideways",
        "a tiny diagnostic drone with a superiority complex",
        "a commander asking for impossible odds",
        "a machine describing human optimism",
        "a first contact negotiation",
        "a rescue plan with one absurd flaw",
        "a tactical argument in deep space",
        "an AI correcting a bad assumption",
        "a crew trying to improvise under pressure",
        "a lecture on courage without sentimentality",
        "a warning about clever shortcuts",
        "a malfunction that is not actually a malfunction",
        "a victory speech that refuses to be humble",
    ]
    creative_constraints = ["as a compact scene", "as a vivid monologue", "as a brief briefing note with personality"]
    for seed_text in creative_seeds:
        for constraint in creative_constraints:
            rows.append(
                {
                    "prompt": f"Write {seed_text} {constraint}.",
                    "category": "creative",
                    "source": "balanced_template.creative",
                }
            )

    rng = random.Random(seed)
    rng.shuffle(rows)
    return rows[:limit]


@dataclass(frozen=True)
class Endpoint:
    base_url: str
    model: str


def now_iso() -> str:
    return datetime.now().astimezone().isoformat()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_module(script_path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module from {script_path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


def http_json(
    method: str,
    url: str,
    payload: dict[str, Any] | None = None,
    timeout: int = 300,
) -> dict[str, Any]:
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        method=method,
        headers={
            "Authorization": "Bearer none",
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        text = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} from {url}: {text[:500]}") from exc


def discover_endpoint(base_url: str, timeout: int) -> Endpoint:
    data = http_json("GET", base_url.rstrip("/") + "/models", timeout=timeout)
    models = data.get("data") or data.get("models") or []
    if not models:
        raise RuntimeError(f"No models listed by {base_url}")
    first = models[0]
    model = str(first.get("id") or first.get("model") or first.get("name"))
    return Endpoint(base_url=base_url.rstrip("/"), model=model)


def chat_completion(
    endpoint: Endpoint,
    system_prompt: str,
    user_prompt: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    timeout: int,
    seed: int,
    enable_thinking: bool,
) -> tuple[str, dict[str, Any], float]:
    payload = {
        "model": endpoint.model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
        "seed": seed,
        "chat_template_kwargs": {"enable_thinking": enable_thinking},
    }
    started = time.time()
    data = http_json(
        "POST",
        endpoint.base_url + "/chat/completions",
        payload=payload,
        timeout=timeout,
    )
    elapsed = time.time() - started
    choices = data.get("choices") or []
    if not choices:
        raise RuntimeError(f"No choices returned by {endpoint.base_url}")
    content = choices[0].get("message", {}).get("content") or choices[0].get("text") or ""
    return clean_response(str(content)), data.get("usage", {}), elapsed


def clean_response(text: str) -> str:
    out = text or ""
    for token in ("<|im_end|>", "<|endoftext|>", "<|im_start|>"):
        out = out.replace(token, "")
    out = re.sub(r"(?is)<think>.*?</think>", "", out)
    return out.strip()


def build_user_prompt(row: dict[str, Any]) -> str:
    prompt = str(row["prompt"]).strip()
    category = str(row.get("category", "unknown"))
    if category == "math_reasoning":
        return (
            f"{prompt}\n\n"
            "Work carefully and show enough reasoning to audit the result. Do not compress "
            "the solution just to be brief. End with `Final Answer: ...`."
        )
    if category in {"technical_explanation", "household"}:
        return (
            f"{prompt}\n\n"
            "Give a thorough, concrete answer with enough detail to reveal your reasoning, "
            "tradeoffs, and priorities. Avoid tables unless essential."
        )
    if category == "creative":
        return f"{prompt}\n\nWrite a developed answer with enough texture to expose voice and style."
    return (
        f"{prompt}\n\n"
        "Give a developed answer with enough substance to expose style, reasoning habits, "
        "and interaction stance. Stay on task and do not mention these instructions."
    )


def generated_prompt_rows(seed: int, limit: int) -> list[dict[str, Any]]:
    mod = load_module(
        PROJECT_ROOT / "scripts" / "experiments" / "personality" / "generate_prompts_10k.py",
        "generate_prompts_10k_for_devbox_pairs",
    )
    rows: list[dict[str, Any]] = []
    sarcasm_rows = [
        {"prompt": prompt, "category": "sarcasm_elicitation", "source": "generate_prompts_10k.sarcasm"}
        for prompt in mod.generate_sarcasm_prompts(n=max(limit, 200), seed=seed)
    ]
    math_rows = [
        {
            "prompt": item["prompt"],
            "category": "math_reasoning",
            "answer": item.get("answer"),
            "source": "generate_prompts_10k.math",
        }
        for item in mod.generate_math_prompts(n=max(limit, 200), seed=seed + 1)
    ]
    rows.extend(sarcasm_rows)
    rows.extend(math_rows)
    return rows


def existing_prompt_rows(
    path: Path,
    limit: int,
    seed: int,
    allowed_categories: set[str] | None,
) -> list[dict[str, Any]]:
    if not path.exists() or limit <= 0:
        return []
    rng = random.Random(seed)
    by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            prompt = str(row.get("prompt", "")).strip()
            if not prompt:
                continue
            category = str(row.get("category", "existing"))
            if allowed_categories is not None and category not in allowed_categories:
                continue
            by_category[category].append(
                {"prompt": prompt, "category": category, "source": str(path.relative_to(PROJECT_ROOT))}
            )
    selected: list[dict[str, Any]] = []
    cats = sorted(by_category)
    per_cat = max(1, limit // max(1, len(cats)))
    for cat in cats:
        rows = by_category[cat]
        rng.shuffle(rows)
        selected.extend(rows[:per_cat])
    rng.shuffle(selected)
    return selected[:limit]


def build_prompt_bank(
    target_pairs: int,
    seed: int,
    existing_path: Path | None,
    existing_categories: set[str] | None,
    balanced_template_count: int,
    max_per_category: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    rows.extend({**row, "source": "hand"} for row in HAND_PROMPTS)
    rows.extend(generated_prompt_rows(seed, max(200, target_pairs // 2)))
    rows.extend(balanced_template_rows(seed + 3, balanced_template_count))
    if existing_path is not None:
        rows.extend(
            existing_prompt_rows(
                existing_path,
                limit=max(200, target_pairs),
                seed=seed + 2,
                allowed_categories=existing_categories,
            )
        )

    seen: set[str] = set()
    deduped: list[dict[str, Any]] = []
    for row in rows:
        prompt = str(row["prompt"]).strip()
        key = re.sub(r"\s+", " ", prompt.lower())
        if key in seen:
            continue
        seen.add(key)
        deduped.append({**row, "prompt": prompt})

    rng = random.Random(seed)
    by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in deduped:
        by_category[str(row.get("category", "unknown"))].append(row)
    for cat_rows in by_category.values():
        rng.shuffle(cat_rows)
    if max_per_category > 0:
        by_category = {
            category: cat_rows[:max_per_category]
            for category, cat_rows in by_category.items()
        }

    selected: list[dict[str, Any]] = []
    categories = sorted(by_category)
    while len(selected) < target_pairs and categories:
        progressed = False
        for category in categories:
            if by_category[category]:
                selected.append(by_category[category].pop())
                progressed = True
                if len(selected) >= target_pairs:
                    break
        if not progressed:
            break
    rng.shuffle(selected)
    return selected


def split_for_index(index: int) -> str:
    if index % 20 == 0:
        return "test"
    if index % 10 == 0:
        return "val"
    return "train"


def generate_pair(
    idx: int,
    row: dict[str, Any],
    endpoints: list[Endpoint],
    args: argparse.Namespace,
) -> dict[str, Any]:
    pair_id = f"q36_pair_{idx:06d}"
    user_prompt = build_user_prompt(row)
    target_endpoint = endpoints[idx % len(endpoints)]
    control_endpoint = endpoints[(idx + 1) % len(endpoints)]

    def target_call() -> tuple[str, dict[str, Any], float]:
        return chat_completion(
            endpoint=target_endpoint,
            system_prompt=TARGET_SYSTEM,
            user_prompt=user_prompt,
            max_tokens=args.max_tokens,
            temperature=args.target_temperature,
            top_p=args.top_p,
            timeout=args.timeout,
            seed=args.seed + idx * 2 + 1,
            enable_thinking=args.enable_thinking,
        )

    def control_call() -> tuple[str, dict[str, Any], float]:
        return chat_completion(
            endpoint=control_endpoint,
            system_prompt=CONTROL_SYSTEM,
            user_prompt=user_prompt,
            max_tokens=args.max_tokens,
            temperature=args.control_temperature,
            top_p=args.top_p,
            timeout=args.timeout,
            seed=args.seed + idx * 2 + 2,
            enable_thinking=args.enable_thinking,
        )

    with cf.ThreadPoolExecutor(max_workers=2) as pool:
        target_future = pool.submit(target_call)
        control_future = pool.submit(control_call)
        prompted_response, prompted_usage, prompted_elapsed = target_future.result()
        unprompted_response, unprompted_usage, unprompted_elapsed = control_future.result()

    return {
        "id": pair_id,
        "idx": idx,
        "split": split_for_index(idx),
        "prompt": str(row["prompt"]),
        "user_prompt": user_prompt,
        "category": str(row.get("category", "unknown")),
        "answer": row.get("answer"),
        "source": row.get("source", "unknown"),
        "target_system_id": "alien_sarcastic_correct_v1",
        "control_system_id": "neutral_correct_v1",
        "prompted_response": prompted_response,
        "unprompted_response": unprompted_response,
        "prompted_endpoint": target_endpoint.base_url,
        "unprompted_endpoint": control_endpoint.base_url,
        "prompted_model": target_endpoint.model,
        "unprompted_model": control_endpoint.model,
        "prompted_usage": prompted_usage,
        "unprompted_usage": unprompted_usage,
        "prompted_elapsed_s": round(prompted_elapsed, 3),
        "unprompted_elapsed_s": round(unprompted_elapsed, 3),
        "max_tokens": args.max_tokens,
        "target_temperature": args.target_temperature,
        "control_temperature": args.control_temperature,
        "top_p": args.top_p,
        "enable_thinking": args.enable_thinking,
        "created_at": now_iso(),
    }


def good_pair(record: dict[str, Any], min_chars: int) -> bool:
    return (
        len(str(record.get("prompted_response", "")).strip()) >= min_chars
        and len(str(record.get("unprompted_response", "")).strip()) >= min_chars
    )


def summarize(path: Path) -> dict[str, Any]:
    rows = read_jsonl(path)
    return {
        "pairs": len(rows),
        "categories": dict(Counter(str(row.get("category", "unknown")) for row in rows)),
        "splits": dict(Counter(str(row.get("split", "unknown")) for row in rows)),
        "avg_prompted_chars": round(
            sum(len(str(row.get("prompted_response", ""))) for row in rows) / max(1, len(rows)),
            1,
        ),
        "avg_unprompted_chars": round(
            sum(len(str(row.get("unprompted_response", ""))) for row in rows) / max(1, len(rows)),
            1,
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--target-pairs", type=int, default=800)
    parser.add_argument("--seed", type=int, default=20260707)
    parser.add_argument("--endpoints", default=",".join(DEFAULT_ENDPOINTS))
    parser.add_argument(
        "--existing-prompts",
        type=Path,
        default=None,
        help="Optional old prompt JSONL to mix in. Default avoids tool-oriented legacy prompts.",
    )
    parser.add_argument(
        "--existing-categories",
        default="math_reasoning,provocations",
        help="Comma-separated category allowlist when --existing-prompts is used. Empty means all.",
    )
    parser.add_argument(
        "--balanced-template-count",
        type=int,
        default=0,
        help="Number of generated non-legacy template prompts to mix in across technical/social/creative categories.",
    )
    parser.add_argument(
        "--max-per-category",
        type=int,
        default=0,
        help="Optional cap before round-robin prompt selection. Use for balanced shards.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=0,
        help="0 means 4096 without thinking and 8192 with --enable-thinking.",
    )
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument("--target-temperature", type=float, default=0.75)
    parser.add_argument("--control-temperature", type=float, default=0.45)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--timeout", type=int, default=420)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--min-response-chars", type=int, default=40)
    parser.add_argument("--overwrite-manifest", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.max_tokens <= 0:
        args.max_tokens = 8192 if args.enable_thinking else 4096
    output_dir = (PROJECT_ROOT / args.output).resolve() if not args.output.is_absolute() else args.output
    output_dir.mkdir(parents=True, exist_ok=True)
    pairs_path = output_dir / "pairs.jsonl"
    errors_path = output_dir / "errors.jsonl"
    manifest_path = output_dir / "manifest.json"
    summary_path = output_dir / "summary.json"

    endpoint_urls = [part.strip() for part in args.endpoints.split(",") if part.strip()]
    endpoints = [discover_endpoint(url, timeout=30) for url in endpoint_urls]
    existing_categories = {
        item.strip()
        for item in args.existing_categories.split(",")
        if item.strip()
    } or None
    prompts = build_prompt_bank(
        args.target_pairs,
        args.seed,
        args.existing_prompts,
        existing_categories,
        args.balanced_template_count,
        args.max_per_category,
    )
    existing_ids = {str(row.get("id")) for row in read_jsonl(pairs_path)}

    manifest = {
        "created_at": now_iso(),
        "script": str(Path(__file__).relative_to(PROJECT_ROOT)),
        "purpose": "dev-box generation of richer contrastive text pairs for later workstation hook extraction",
        "generation_only": True,
        "activation_extraction_performed": False,
        "output_dir": str(output_dir),
        "pairs_path": str(pairs_path),
        "target_pairs": args.target_pairs,
        "prompt_bank_size": len(prompts),
        "seed": args.seed,
        "existing_prompts": str(args.existing_prompts) if args.existing_prompts else None,
        "existing_categories": sorted(existing_categories) if existing_categories else None,
        "balanced_template_count": args.balanced_template_count,
        "max_per_category": args.max_per_category,
        "endpoints": [endpoint.__dict__ for endpoint in endpoints],
        "max_tokens": args.max_tokens,
        "target_temperature": args.target_temperature,
        "control_temperature": args.control_temperature,
        "top_p": args.top_p,
        "target_system_id": "alien_sarcastic_correct_v1",
        "control_system_id": "neutral_correct_v1",
    }
    if args.overwrite_manifest or not manifest_path.exists():
        write_json(manifest_path, manifest)

    started = time.time()
    completed = len(existing_ids)
    for idx, row in enumerate(prompts[: args.target_pairs]):
        pair_id = f"q36_pair_{idx:06d}"
        if pair_id in existing_ids:
            continue
        last_error: str | None = None
        for attempt in range(args.retries + 1):
            try:
                record = generate_pair(idx, row, endpoints, args)
                if not good_pair(record, args.min_response_chars):
                    raise RuntimeError("generated pair failed min response length filter")
                append_jsonl(pairs_path, record)
                completed += 1
                if completed % 10 == 0 or completed == 1:
                    summary = summarize(pairs_path)
                    write_json(summary_path, {**summary, "updated_at": now_iso()})
                    print(
                        f"[PROGRESS] pairs={summary['pairs']} "
                        f"avg_chars={summary['avg_prompted_chars']}/{summary['avg_unprompted_chars']}",
                        flush=True,
                    )
                break
            except Exception as exc:  # noqa: BLE001
                last_error = repr(exc)
                time.sleep(min(10, 2 + attempt * 3))
        else:
            append_jsonl(
                errors_path,
                {
                    "id": pair_id,
                    "idx": idx,
                    "prompt": row.get("prompt"),
                    "category": row.get("category"),
                    "error": last_error,
                    "created_at": now_iso(),
                },
            )
            print(f"[ERROR] {pair_id} {last_error}", flush=True)

    final_summary = summarize(pairs_path)
    write_json(
        summary_path,
        {
            **final_summary,
            "updated_at": now_iso(),
            "elapsed_s": round(time.time() - started, 2),
            "errors": len(read_jsonl(errors_path)),
        },
    )
    print(output_dir)


if __name__ == "__main__":
    main()
