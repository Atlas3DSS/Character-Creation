#!/usr/bin/env python3
from __future__ import annotations

import collections
import fcntl
import json
import os
import random
import signal
import shutil
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Iterable


ROOT = Path("/home/orwel/dev_genius/experiments/Character Creation")
VENV_PY = Path("/home/orwel/dev_genius/venv/bin/python")
REPLAY_VENV_PY = Path("/home/orwel/dev_genius/qwen35_replay_venv/bin/python")
GUARD_PY = ROOT / "scripts/infra/run_with_vram_guard.py"
FULL_OUT_REL = "sweep_v3/ws_openai_15k"
FULL_OUT_DIR = ROOT / FULL_OUT_REL
FULL_GEN_DIR = FULL_OUT_DIR / "generated"
PASS2_OUT_REL = "sweep_v3/ws_openai_15k_sampled25m"
PASS2_OUT_DIR = ROOT / PASS2_OUT_REL
EXPECTED_RESPONSES = 14580
CHECK_EVERY_SECONDS = 300
SAMPLE_TARGET_TOKENS = 25_000_000
SAMPLE_SEED = 42
SAMPLE_MIN_PER_CHAR_CAT = 4
KNOWN_CATEGORIES = ["emotional", "identity", "reasoning", "social", "practical", "creative"]

LOG_PATH = ROOT / "logs/ws15k_autohandoff.log"
LOCK_PATH = ROOT / "logs/ws15k_autohandoff.lock"
PASS2_LOG = ROOT / "logs/ws15k_pass2.log"
SERVER_LOG = ROOT / "logs/ws15k_server_restart.log"
PASS2_VRAM_FRACTION = 0.89
PASS2_REPLAY_BATCH_SIZE = 1
PASS2_REPLAY_MAX_TOTAL_TOKENS = 16384

PASS1_MATCH = ("personality_sweep_v3_pass1_openai.py", "--output", FULL_OUT_REL)
PASS2_MATCH = ("personality_sweep_v3_two_pass.py", "--output", PASS2_OUT_REL)
LOCAL_SERVER_MATCH = ("sglang.launch_server", "--model-path", "Qwen/Qwen3.5-9B", "--port", "30000")
TUNNEL_MATCH = ("-L", "33001:127.0.0.1:30001", "-L", "33002:127.0.0.1:30002", "192.168.1.90")


def ts() -> str:
    return datetime.now().isoformat()


def log(msg: str) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(f"{ts()} {msg}\n")
        f.flush()


def list_pids_with_all(tokens: Iterable[str]) -> list[int]:
    toks = tuple(tokens)
    out: list[int] = []
    for pid in os.listdir("/proc"):
        if not pid.isdigit():
            continue
        try:
            cmd = (
                Path(f"/proc/{pid}/cmdline")
                .read_bytes()
                .replace(b"\x00", b" ")
                .decode("utf-8", "ignore")
                .strip()
            )
        except Exception:
            continue
        if not cmd:
            continue
        if all(t in cmd for t in toks):
            out.append(int(pid))
    return sorted(out)


def kill_pids(pids: Iterable[int], sig: int) -> None:
    for pid in pids:
        try:
            os.kill(pid, sig)
        except ProcessLookupError:
            continue
        except PermissionError:
            continue


def count_responses_tokens() -> tuple[int, int]:
    responses = 0
    tokens = 0
    if not FULL_GEN_DIR.exists():
        return responses, tokens
    for fp in FULL_GEN_DIR.glob("char_*.jsonl"):
        try:
            with fp.open("r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except Exception:
                        continue
                    responses += 1
                    tokens += int(obj.get("n_gen_tokens") or 0)
        except FileNotFoundError:
            continue
    return responses, tokens


def wait_for_pids_gone(match_tokens: tuple[str, ...], timeout_s: int) -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if not list_pids_with_all(match_tokens):
            return True
        time.sleep(2)
    return not list_pids_with_all(match_tokens)


def load_latest_full_records() -> dict[int, dict[int, dict]]:
    """Return char_id -> prompt_idx -> latest record."""
    out: dict[int, dict[int, dict]] = {}
    for fp in FULL_GEN_DIR.glob("char_*.jsonl"):
        try:
            cid = int(fp.stem.split("_")[1])
        except Exception:
            continue
        per_prompt: dict[int, dict] = {}
        try:
            with fp.open("r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except Exception:
                        continue
                    try:
                        pi = int(rec.get("prompt_idx"))
                    except Exception:
                        continue
                    per_prompt[pi] = rec
        except FileNotFoundError:
            continue
        if per_prompt:
            out[cid] = per_prompt
    return out


def build_pass2_subset() -> tuple[int, int]:
    """Build sampled subset in PASS2_OUT_DIR. Returns (responses, tokens)."""
    rng = random.Random(SAMPLE_SEED)
    latest = load_latest_full_records()
    if not latest:
        raise RuntimeError("No pass1 generated records found to sample")

    # Buckets by (char_id, prompt_category) for broad personality/category coverage.
    buckets: dict[tuple[int, str], list[dict]] = collections.defaultdict(list)
    for cid, pmap in latest.items():
        for rec in pmap.values():
            cat = str(rec.get("prompt_category") or "unknown")
            nt = int(rec.get("n_gen_tokens") or 0)
            if nt <= 0:
                continue
            buckets[(cid, cat)].append(rec)

    for recs in buckets.values():
        rng.shuffle(recs)

    selected_by_char: dict[int, dict[int, dict]] = collections.defaultdict(dict)
    selected_tokens = 0
    selected_count = 0
    category_counts: collections.Counter[str] = collections.Counter()

    def add_record(rec: dict) -> None:
        nonlocal selected_tokens, selected_count
        cid = int(rec["char_id"])
        pi = int(rec["prompt_idx"])
        if pi in selected_by_char[cid]:
            return
        selected_by_char[cid][pi] = rec
        nt = int(rec.get("n_gen_tokens") or 0)
        selected_tokens += nt
        selected_count += 1
        category_counts[str(rec.get("prompt_category") or "unknown")] += 1

    # Phase 1: minimum per-char/per-category floor for broad coverage.
    char_ids = sorted({cid for cid, _ in buckets.keys()})
    cats = list(KNOWN_CATEGORIES) + sorted(
        {cat for _, cat in buckets.keys() if cat not in KNOWN_CATEGORIES}
    )
    ptr: dict[tuple[int, str], int] = collections.defaultdict(int)

    for cid in char_ids:
        for cat in cats:
            key = (cid, cat)
            recs = buckets.get(key)
            if not recs:
                continue
            take = min(SAMPLE_MIN_PER_CHAR_CAT, len(recs))
            for _ in range(take):
                rec = recs[ptr[key]]
                ptr[key] += 1
                add_record(rec)

    # Phase 2: round-robin fill to token target.
    keys = [k for k, recs in buckets.items() if ptr[k] < len(recs)]
    while selected_tokens < SAMPLE_TARGET_TOKENS and keys:
        rng.shuffle(keys)
        progressed = False
        next_keys: list[tuple[int, str]] = []
        for key in keys:
            recs = buckets[key]
            i = ptr[key]
            if i >= len(recs):
                continue
            add_record(recs[i])
            ptr[key] = i + 1
            progressed = True
            if ptr[key] < len(recs):
                next_keys.append(key)
            if selected_tokens >= SAMPLE_TARGET_TOKENS:
                break
        if not progressed:
            break
        keys = next_keys

    # Rewrite subset output directory with sampled generated data only.
    if PASS2_OUT_DIR.exists():
        shutil.rmtree(PASS2_OUT_DIR)
    (PASS2_OUT_DIR / "generated").mkdir(parents=True, exist_ok=True)

    for cid, pmap in selected_by_char.items():
        recs = [pmap[k] for k in sorted(pmap.keys())]
        out_fp = PASS2_OUT_DIR / "generated" / f"char_{cid:04d}.jsonl"
        with out_fp.open("w", encoding="utf-8") as f:
            for rec in recs:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # Carry forward metadata files when present.
    for name in ("characters.jsonl", "sweep_config.json"):
        src = FULL_OUT_DIR / name
        if src.exists():
            shutil.copy2(src, PASS2_OUT_DIR / name)

    manifest = {
        "timestamp": ts(),
        "source_output": str(FULL_OUT_DIR),
        "subset_output": str(PASS2_OUT_DIR),
        "sampling": {
            "strategy": "char_x_category_floor_plus_round_robin_fill",
            "seed": SAMPLE_SEED,
            "target_tokens": SAMPLE_TARGET_TOKENS,
            "min_per_char_category": SAMPLE_MIN_PER_CHAR_CAT,
        },
        "selected": {
            "responses": selected_count,
            "tokens": selected_tokens,
            "avg_tokens_per_response": selected_tokens / max(selected_count, 1),
            "category_counts": dict(category_counts),
        },
    }
    (PASS2_OUT_DIR / "subset_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return selected_count, selected_tokens


def launch_pass2_bf16() -> subprocess.Popen:
    cmd = [
        str(REPLAY_VENV_PY),
        str(GUARD_PY),
        "--gpu-index",
        "0",
        "--max-vram-fraction",
        str(PASS2_VRAM_FRACTION),
        "--poll-seconds",
        "5",
        "--breach-polls",
        "1",
        "--kill-timeout-seconds",
        "15",
        "--log-file",
        str(PASS2_LOG),
        "--chdir",
        str(ROOT),
        "--",
        str(REPLAY_VENV_PY),
        "scripts/experiments/personality/personality_sweep_v3_two_pass.py",
        "--model",
        "Qwen/Qwen3.5-9B",
        "--output",
        PASS2_OUT_REL,
        "--skip-pass1",
        "--quantize",
        "bf16",
        "--replay-quantize",
        "bf16",
        "--replay-batch-size",
        str(PASS2_REPLAY_BATCH_SIZE),
        "--replay-max-total-tokens",
        str(PASS2_REPLAY_MAX_TOTAL_TOKENS),
    ]
    PASS2_LOG.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.Popen(cmd, cwd=str(ROOT), start_new_session=True)
    return proc


def launch_local_server() -> subprocess.Popen:
    cmd = [
        str(VENV_PY),
        "-m",
        "sglang.launch_server",
        "--model-path",
        "Qwen/Qwen3.5-9B",
        "--trust-remote-code",
        "--dtype",
        "bfloat16",
        "--port",
        "30000",
        "--attention-backend",
        "triton",
    ]
    SERVER_LOG.parent.mkdir(parents=True, exist_ok=True)
    f = SERVER_LOG.open("a", encoding="utf-8")
    proc = subprocess.Popen(cmd, cwd=str(ROOT), stdout=f, stderr=subprocess.STDOUT, start_new_session=True)
    return proc


def main() -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOCK_PATH.open("w", encoding="utf-8") as lockf:
        try:
            fcntl.flock(lockf.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            log("lock-busy exiting")
            return

        log(f"autohandoff-start expected_responses={EXPECTED_RESPONSES}")

        # If pass2 is already running, do nothing.
        if list_pids_with_all(PASS2_MATCH):
            log("pass2-already-running exiting")
            return

        while True:
            responses, tokens = count_responses_tokens()
            p1 = list_pids_with_all(PASS1_MATCH)
            log(f"pass1-status responses={responses} tokens={tokens} pass1_pids={p1 or 'none'}")

            if responses >= EXPECTED_RESPONSES and not p1:
                break
            time.sleep(CHECK_EVERY_SECONDS)

        # Build sampled subset for pass2.
        subset_responses, subset_tokens = build_pass2_subset()
        log(
            f"subset-ready output={PASS2_OUT_REL} "
            f"responses={subset_responses} tokens={subset_tokens}"
        )

        # Clean up any lingering pass1 workers/tunnel before pass2.
        p1 = list_pids_with_all(PASS1_MATCH)
        if p1:
            log(f"terminating-lingering-pass1 pids={p1}")
            kill_pids(p1, signal.SIGTERM)
            time.sleep(3)
            p1 = list_pids_with_all(PASS1_MATCH)
            if p1:
                log(f"killing-lingering-pass1 pids={p1}")
                kill_pids(p1, signal.SIGKILL)

        tun = list_pids_with_all(TUNNEL_MATCH)
        if tun:
            log(f"stopping-tunnel pids={tun}")
            kill_pids(tun, signal.SIGTERM)

        # Stop local server to free VRAM for bf16 replay.
        srv = list_pids_with_all(LOCAL_SERVER_MATCH)
        if srv:
            log(f"stopping-local-server pids={srv}")
            kill_pids(srv, signal.SIGTERM)
            if not wait_for_pids_gone(LOCAL_SERVER_MATCH, timeout_s=120):
                srv = list_pids_with_all(LOCAL_SERVER_MATCH)
                if srv:
                    log(f"force-killing-local-server pids={srv}")
                    kill_pids(srv, signal.SIGKILL)
        else:
            log("local-server-not-running-before-pass2")

        log("launching-pass2-bf16")
        p2 = launch_pass2_bf16()
        log(f"pass2-launched pid={p2.pid} log={PASS2_LOG}")

        rc = p2.wait()
        log(f"pass2-exit returncode={rc}")

        # Restart local inference server after pass2 completes.
        if list_pids_with_all(LOCAL_SERVER_MATCH):
            log("local-server-already-running skip-restart")
            return

        sproc = launch_local_server()
        log(f"local-server-restarted pid={sproc.pid} log={SERVER_LOG}")


if __name__ == "__main__":
    main()
