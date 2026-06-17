#!/usr/bin/env python3
"""High-concurrency pass1 generation client for personality_sweep_v3.

Uses an OpenAI-compatible server (e.g., SGLang at /v1/chat/completions)
and writes the same generated/char_*.jsonl format expected by
personality_sweep_v3_two_pass.py replay mode (--skip-pass1).
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import signal
import sys
import threading
import time
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import requests
from tqdm import tqdm

_SHUTDOWN = False


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
    # Ensure decorators/type resolution inside the loaded module can find itself.
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


_THREAD_LOCAL = threading.local()


def thread_session(headers: dict[str, str], timeout_s: float) -> requests.Session:
    sess = getattr(_THREAD_LOCAL, "session", None)
    if sess is None:
        sess = requests.Session()
        sess.headers.update(headers)
        setattr(_THREAD_LOCAL, "session", sess)
    sess.request_timeout = timeout_s  # ad-hoc attribute for consistency
    return sess


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
    """Perform one chat-completions request with retries."""
    url = f"{base_url.rstrip('/')}/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    messages = [
        {"role": "system", "content": task["system_prompt"]},
        {"role": "user", "content": task["prompt"]},
    ]

    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_new_tokens,
        "stream": False,
    }
    if enable_thinking:
        payload["chat_template_kwargs"] = {"enable_thinking": True}

    last_err = ""
    for attempt in range(1, retries + 1):
        if _SHUTDOWN:
            return {"ok": False, "error": "shutdown"}
        try:
            sess = thread_session(headers=headers, timeout_s=timeout_s)
            t0 = time.time()
            resp = sess.post(url, json=payload, timeout=timeout_s)
            latency = time.time() - t0

            # Retry once without chat_template_kwargs if server rejects extra field.
            if resp.status_code >= 400 and enable_thinking and "chat_template_kwargs" in payload:
                body_text = resp.text
                if "chat_template_kwargs" in body_text or "unknown" in body_text.lower():
                    payload.pop("chat_template_kwargs", None)
                    resp = sess.post(url, json=payload, timeout=timeout_s)
                    latency = time.time() - t0

            if resp.status_code >= 400:
                body = resp.text[:800]
                raise RuntimeError(f"HTTP {resp.status_code}: {body}")

            data = resp.json()
            choices = data.get("choices") or []
            if not choices:
                raise RuntimeError(f"No choices in response: {str(data)[:500]}")

            message = choices[0].get("message") or {}
            full_text = normalize_content(message.get("content"))
            usage = data.get("usage") or {}
            completion_tokens = usage.get("completion_tokens")

            return {
                "ok": True,
                "full_text": full_text,
                "completion_tokens": completion_tokens,
                "latency_s": latency,
                "raw": data,
            }
        except Exception as exc:  # noqa: BLE001
            last_err = str(exc)
            if attempt < retries:
                time.sleep(min(2 ** (attempt - 1), 8))
            continue

    return {"ok": False, "error": last_err}


def main() -> None:
    parser = argparse.ArgumentParser(description="Personality Sweep V3 pass1 via OpenAI-compatible server")
    parser.add_argument("--model", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--base-url", type=str, default="http://127.0.0.1:30000/v1")
    parser.add_argument("--api-key", type=str, default="dummy")
    parser.add_argument("--concurrency", type=int, default=128)
    parser.add_argument("--timeout", type=float, default=360.0)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--disable-thinking", action="store_true")

    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shard-list", type=str, default=None)
    parser.add_argument("--n-shards", type=int, default=1)
    parser.add_argument("--max-chars", type=int, default=None)
    parser.add_argument("--max-prompts", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    v3_path = script_dir / "personality_sweep_v3_two_pass.py"
    v3 = load_v3_module(v3_path)

    target_output = Path(args.output)
    target_output.mkdir(parents=True, exist_ok=True)

    processor = v3.load_processor(args.model)
    tokenizer = processor.tokenizer

    all_chars = v3.generate_characters(seed=args.seed, max_chars=args.max_chars)
    shard_list = v3.parse_shard_list(args.shard_list, args.n_shards)
    if shard_list:
        shard_set = set(shard_list)
    else:
        if args.shard < 0 or args.shard >= args.n_shards:
            raise ValueError(f"--shard must be in [0, {args.n_shards - 1}]")
        shard_set = {args.shard}
        shard_list = [args.shard]

    my_chars = [c for c in all_chars if (c.char_id - 1) % args.n_shards in shard_set]
    prompts = list(v3.ALL_PROMPTS)
    prompt_categories = list(v3.PROMPT_CATEGORIES)
    if args.max_prompts:
        prompts = prompts[:args.max_prompts]
        prompt_categories = prompt_categories[:args.max_prompts]

    gen_writer = v3.GeneratedWriter(target_output)

    cfg = {
        "timestamp": datetime.now().isoformat(),
        "pipeline": "pass1_openai_server",
        "model": args.model,
        "base_url": args.base_url,
        "concurrency": args.concurrency,
        "n_characters_total": len(all_chars),
        "n_characters_this_shard": len(my_chars),
        "shard_list": shard_list,
        "n_shards": args.n_shards,
        "n_prompts_per_char": len(prompts),
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "seed": args.seed,
    }
    (target_output / "sweep_config.json").write_text(json.dumps(cfg, indent=2))

    chars_file = target_output / "characters.jsonl"
    if not chars_file.exists():
        with open(chars_file, "w", encoding="utf-8") as fh:
            for c in my_chars:
                fh.write(json.dumps(asdict(c)) + "\n")

    # Build pending task list.
    tasks: list[dict[str, Any]] = []
    for char in my_chars:
        existing = gen_writer.read_char_generations(char.char_id)
        sys_prompt = v3.build_system_prompt(char)
        b5_combo = v3.b5_combo_code(char)
        for pi, prompt in enumerate(prompts):
            if pi in existing:
                continue
            tasks.append(
                {
                    "char_id": char.char_id,
                    "char_name": char.name,
                    "system_prompt": sys_prompt,
                    "prompt_idx": pi,
                    "prompt": prompt,
                    "prompt_category": prompt_categories[pi],
                    "b5_combo": b5_combo,
                }
            )

    print(
        f"[INFO] OpenAI pass1: {len(tasks)} pending prompts, "
        f"chars={len(my_chars)}, shards={','.join(str(s) for s in shard_list)}/{args.n_shards}, "
        f"concurrency={args.concurrency}"
    )

    total_tokens = 0
    ok_count = 0
    err_count = 0
    lat_sum = 0.0
    t0 = time.time()

    # Bounded in-flight futures to control memory.
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

        pbar = tqdm(total=len(tasks), desc="Pass1-OpenAI")
        while inflight:
            done, _ = wait(inflight, return_when=FIRST_COMPLETED)
            for fut in done:
                inflight.remove(fut)
                task = fut.task  # type: ignore[attr-defined]
                result = fut.result()

                if result.get("ok"):
                    full_text = result["full_text"]
                    think_text, response_text = v3.parse_think_response(full_text)
                    gen_token_ids = tokenizer.encode(full_text, add_special_tokens=False)
                    think_ids = tokenizer.encode(think_text, add_special_tokens=False) if think_text else []
                    resp_ids = tokenizer.encode(response_text, add_special_tokens=False) if response_text else []

                    n_gen_tokens = result.get("completion_tokens")
                    if not isinstance(n_gen_tokens, int) or n_gen_tokens <= 0:
                        n_gen_tokens = len(gen_token_ids)

                    rec = {
                        "char_id": task["char_id"],
                        "char_name": task["char_name"],
                        "prompt_idx": task["prompt_idx"],
                        "prompt_category": task["prompt_category"],
                        "prompt": task["prompt"],
                        "b5": task["b5_combo"],
                        "think_text": think_text,
                        "response_text": response_text,
                        "n_think_tokens": len(think_ids),
                        "n_response_tokens": len(resp_ids),
                        "n_gen_tokens": int(n_gen_tokens),
                        "gen_token_ids": gen_token_ids,
                        "full_text": full_text,
                        "backend": "openai_server",
                        "latency_s": float(result.get("latency_s") or 0.0),
                        "timestamp": datetime.now().isoformat(),
                    }
                    gen_writer.write_generation(task["char_id"], rec)
                    ok_count += 1
                    total_tokens += int(n_gen_tokens)
                    lat_sum += float(result.get("latency_s") or 0.0)
                else:
                    err_count += 1
                    print(
                        f"[ERROR] char={task['char_id']} prompt={task['prompt_idx']}: "
                        f"{result.get('error', 'unknown')[:300]}"
                    )

                pbar.update(1)
                if not _SHUTDOWN:
                    while len(inflight) < max_inflight:
                        if not submit_next():
                            break

            if _SHUTDOWN and not inflight:
                break

        pbar.close()

    elapsed = time.time() - t0
    summary = {
        "timestamp": datetime.now().isoformat(),
        "ok_responses": ok_count,
        "error_responses": err_count,
        "gen_tokens": total_tokens,
        "elapsed_seconds": elapsed,
        "gen_tokens_per_second": total_tokens / max(elapsed, 1.0),
        "responses_per_second": ok_count / max(elapsed, 1.0),
        "avg_latency_seconds": (lat_sum / ok_count) if ok_count else None,
        "base_url": args.base_url,
        "model": args.model,
        "concurrency": args.concurrency,
        "max_new_tokens": args.max_new_tokens,
        "shard_list": shard_list,
    }
    (target_output / "pass1_openai_summary.json").write_text(json.dumps(summary, indent=2))

    print(
        f"[DONE] ok={ok_count} err={err_count} tokens={total_tokens/1e6:.2f}M "
        f"rate={summary['gen_tokens_per_second']:.1f} tok/s"
    )


if __name__ == "__main__":
    main()
