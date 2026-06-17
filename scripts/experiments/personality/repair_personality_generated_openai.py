#!/usr/bin/env python3
"""Repair contaminated generated rows via OpenAI-compatible inference servers.

This is a targeted salvage pass for rows where the stored `response_text`
contains planner text like `Thinking Process:` and no clean visible response.

The repair path intentionally uses `enable_thinking=False` and direct-response
instructions because the current SGLang servers do not expose a usable separate
reasoning field for these persona prompts.
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
_THREAD_LOCAL = threading.local()
SPECIAL_TOKENS = ("<|im_end|>", "<|endoftext|>", "<|im_start|>")
DIRECT_SUFFIX = (
    "\nRespond directly as the character. Do not output any thinking process, analysis, planning, "
    "bullet points, headings, or the literal phrase 'Thinking Process:'."
)


def _sig_handler(signum, frame):
    del frame
    global _SHUTDOWN
    _SHUTDOWN = True
    print(f"\n[SHUTDOWN] Signal {signum} received; draining in-flight repairs...")


signal.signal(signal.SIGTERM, _sig_handler)
signal.signal(signal.SIGINT, _sig_handler)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Repair contaminated personality generations")
    parser.add_argument("--source-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--base-url", type=str, required=True)
    parser.add_argument("--server-label", type=str, required=True)
    parser.add_argument("--model", type=str, default="Qwen/Qwen3.5-9B")
    parser.add_argument("--api-key", type=str, default="dummy")
    parser.add_argument("--timeout", type=float, default=240.0)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--char-mod", type=int, default=2)
    parser.add_argument("--char-rem", type=int, required=True)
    return parser.parse_args()


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module from {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def thread_session(headers: dict[str, str]) -> requests.Session:
    sess = getattr(_THREAD_LOCAL, "session", None)
    if sess is None:
        sess = requests.Session()
        sess.headers.update(headers)
        setattr(_THREAD_LOCAL, "session", sess)
    return sess


def clean_generation_text(text: str) -> str:
    out = text or ""
    for tok in SPECIAL_TOKENS:
        out = out.replace(tok, "")
    return out.strip()


def is_contaminated_text(text: str) -> bool:
    stripped = clean_generation_text(text).lstrip()
    if not stripped:
        return True
    return stripped.startswith("Thinking Process:") or stripped.startswith("<think>")


def request_one(
    *,
    base_url: str,
    model: str,
    api_key: str,
    timeout_s: float,
    retries: int,
    temperature: float,
    top_p: float,
    max_new_tokens: int,
    system_prompt: str,
    user_prompt: str,
) -> dict[str, Any]:
    url = f"{base_url.rstrip('/')}/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_new_tokens,
        "stream": False,
        "chat_template_kwargs": {"enable_thinking": False},
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
            if resp.status_code >= 400:
                raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:800]}")
            data = resp.json()
            choices = data.get("choices") or []
            if not choices:
                raise RuntimeError(f"No choices in response: {str(data)[:500]}")
            message = choices[0].get("message") or {}
            text = clean_generation_text(str(message.get("content") or ""))
            if is_contaminated_text(text):
                raise RuntimeError("contaminated_response")
            usage = data.get("usage") or {}
            return {
                "ok": True,
                "text": text,
                "completion_tokens": usage.get("completion_tokens"),
                "latency_s": latency,
            }
        except Exception as exc:  # noqa: BLE001
            last_err = str(exc)
            if attempt < retries:
                time.sleep(min(2 ** (attempt - 1), 8))
    return {"ok": False, "error": last_err}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if not line.strip():
            continue
        rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def row_needs_repair(row: dict[str, Any]) -> bool:
    return is_contaminated_text(str(row.get("response_text") or ""))


def main() -> None:
    args = parse_args()
    source_dir = Path(args.source_dir)
    output_dir = Path(args.output_dir)
    gen_src = source_dir / "generated"
    gen_out = output_dir / "generated"
    if not gen_src.exists():
        raise FileNotFoundError(f"Missing generated dir: {gen_src}")

    script_dir = Path(__file__).resolve().parent
    v3 = load_module("personality_sweep_v3_two_pass_repair", script_dir / "personality_sweep_v3_two_pass.py")
    processor = v3.load_processor(args.model)
    tokenizer = processor.tokenizer

    char_map: dict[int, Any] = {}
    for line in (source_dir / "characters.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        char = v3.Character(**row)
        char_map[int(char.char_id)] = char

    repair_targets: dict[int, list[int]] = {}
    char_rows: dict[int, list[dict[str, Any]]] = {}
    for path in sorted(gen_src.glob("char_*.jsonl")):
        char_id = int(path.stem.split("_")[1])
        if char_id % args.char_mod != args.char_rem:
            continue
        rows = read_jsonl(path)
        bad_idxs = [int(row["prompt_idx"]) for row in rows if row_needs_repair(row)]
        if not bad_idxs:
            continue
        char_rows[char_id] = rows
        repair_targets[char_id] = bad_idxs

    total_repairs = sum(len(v) for v in repair_targets.values())
    print(
        f"[INFO] {args.server_label}: chars={len(repair_targets)} repairs={total_repairs} "
        f"char_mod={args.char_mod} char_rem={args.char_rem}"
    )

    pending: list[dict[str, Any]] = []
    pending_counts: dict[int, int] = {}
    for char_id, bad_idxs in sorted(repair_targets.items()):
        char = char_map[char_id]
        sys_prompt = v3.build_system_prompt(char) + DIRECT_SUFFIX
        pending_counts[char_id] = len(bad_idxs)
        for row in char_rows[char_id]:
            prompt_idx = int(row["prompt_idx"])
            if prompt_idx not in bad_idxs:
                continue
            pending.append(
                {
                    "char_id": char_id,
                    "prompt_idx": prompt_idx,
                    "prompt": str(row.get("prompt") or v3.ALL_PROMPTS[prompt_idx]),
                    "prompt_category": str(row.get("prompt_category") or v3.PROMPT_CATEGORIES[prompt_idx]),
                    "system_prompt": sys_prompt,
                    "source_row": row,
                }
            )

    results_by_char: dict[int, dict[int, dict[str, Any]]] = {char_id: {} for char_id in repair_targets}
    failed: list[dict[str, Any]] = []
    completed_tasks = 0
    completed_chars = 0
    total_tokens = 0
    ok_count = 0
    lat_sum = 0.0
    t0 = time.time()
    max_inflight = max(args.concurrency * 4, args.concurrency)
    it = iter(pending)

    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        inflight: set[Future] = set()

        def submit_next() -> bool:
            try:
                task = next(it)
            except StopIteration:
                return False
            fut = pool.submit(
                request_one,
                base_url=args.base_url,
                model=args.model,
                api_key=args.api_key,
                timeout_s=args.timeout,
                retries=args.retries,
                temperature=args.temperature,
                top_p=args.top_p,
                max_new_tokens=args.max_new_tokens,
                system_prompt=task["system_prompt"],
                user_prompt=task["prompt"],
            )
            fut.task = task  # type: ignore[attr-defined]
            inflight.add(fut)
            return True

        for _ in range(min(max_inflight, len(pending))):
            if not submit_next():
                break

        pbar = tqdm(total=len(pending), desc=f"{args.server_label}-repair")
        while inflight:
            done_futs, _ = wait(inflight, return_when=FIRST_COMPLETED)
            for fut in done_futs:
                inflight.remove(fut)
                task = fut.task  # type: ignore[attr-defined]
                char_id = int(task["char_id"])
                result = fut.result()
                if result.get("ok"):
                    text = str(result["text"])
                    token_ids = tokenizer.encode(text, add_special_tokens=False)
                    repaired = dict(task["source_row"])
                    repaired["think_text"] = ""
                    repaired["response_text"] = text
                    repaired["full_text"] = text
                    repaired["n_think_tokens"] = 0
                    repaired["n_response_tokens"] = len(token_ids)
                    repaired["n_gen_tokens"] = (
                        int(result["completion_tokens"])
                        if isinstance(result.get("completion_tokens"), int) and int(result["completion_tokens"]) > 0
                        else len(token_ids)
                    )
                    repaired["gen_token_ids"] = token_ids
                    repaired["backend"] = "openai_server_repair_direct"
                    repaired["repair_server"] = args.server_label
                    repaired["repair_mode"] = "response_only_direct"
                    repaired["repair_timestamp"] = datetime.now().isoformat()
                    results_by_char[char_id][int(task["prompt_idx"])] = repaired
                    ok_count += 1
                    total_tokens += len(token_ids)
                    lat_sum += float(result.get("latency_s") or 0.0)
                else:
                    failed.append(
                        {
                            "char_id": char_id,
                            "prompt_idx": task["prompt_idx"],
                            "prompt_category": task["prompt_category"],
                            "error": str(result.get("error") or "unknown"),
                        }
                    )
                completed_tasks += 1
                pending_counts[char_id] -= 1
                if pending_counts[char_id] == 0:
                    merged: list[dict[str, Any]] = []
                    changed = False
                    for row in char_rows[char_id]:
                        prompt_idx = int(row["prompt_idx"])
                        repaired = results_by_char[char_id].get(prompt_idx)
                        if repaired is not None:
                            merged.append(repaired)
                            changed = True
                        else:
                            merged.append(row)
                    if changed:
                        write_jsonl(gen_out / f"char_{char_id:04d}.jsonl", merged)
                    completed_chars += 1
                    if completed_chars % 10 == 0 or completed_chars == len(repair_targets):
                        elapsed = time.time() - t0
                        print(
                            f"[PROGRESS] {args.server_label} chars={completed_chars}/{len(repair_targets)} "
                            f"rows={completed_tasks}/{len(pending)} ok={ok_count} failed={len(failed)} "
                            f"rate={(total_tokens/max(elapsed,1.0)):.1f} tok/s"
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
        "source_dir": str(source_dir.resolve()),
        "output_dir": str(output_dir.resolve()),
        "server_label": args.server_label,
        "char_mod": args.char_mod,
        "char_rem": args.char_rem,
        "target_chars": len(repair_targets),
        "target_rows": len(pending),
        "ok_rows": ok_count,
        "failed_rows": len(failed),
        "repaired_chars": completed_chars,
        "response_tokens": total_tokens,
        "elapsed_seconds": elapsed,
        "response_tokens_per_second": (total_tokens / elapsed) if elapsed else None,
        "rows_per_second": (ok_count / elapsed) if elapsed else None,
        "avg_latency_seconds": (lat_sum / ok_count) if ok_count else None,
        "repair_mode": "response_only_direct",
    }
    (output_dir / f"repair_summary_{args.server_label}.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output_dir / f"repair_failures_{args.server_label}.json").write_text(json.dumps(failed, indent=2), encoding="utf-8")
    print(
        f"[DONE] {args.server_label} repaired={ok_count}/{len(pending)} failed={len(failed)} "
        f"chars={completed_chars} tok={total_tokens/1e6:.2f}M rate={(total_tokens/max(elapsed,1.0)):.1f} tok/s"
    )


if __name__ == "__main__":
    main()
