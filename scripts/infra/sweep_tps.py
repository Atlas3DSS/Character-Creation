#!/usr/bin/env python3
"""
Per-worker throughput view for personality sweep workers (v2/v3).

Reports:
  - pass1 generation tokens/sec from generated/*.jsonl
  - pass2 replay tokens/sec from responses/*.jsonl
  - combined totals across all workers
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_REMOTE = os.environ.get("REMOTE_HOST", "orwel@192.168.1.90")
SWEEP_SCRIPTS = ("personality_sweep_v2.py", "personality_sweep_v3_two_pass.py")


def run_local(cmd: str, timeout: int = 120) -> tuple[bool, str]:
    try:
        p = subprocess.run(
            ["bash", "-lc", cmd],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(PROJECT_DIR),
        )
        if p.returncode != 0:
            return False, (p.stderr or p.stdout or "command failed").strip()
        return True, p.stdout
    except Exception as exc:  # noqa: BLE001
        return False, str(exc)


def run_remote(remote: str, cmd: str, timeout: int = 120) -> tuple[bool, str]:
    try:
        p = subprocess.run(
            ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=6", remote, cmd],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if p.returncode != 0:
            return False, (p.stderr or p.stdout or "ssh command failed").strip()
        return True, p.stdout
    except Exception as exc:  # noqa: BLE001
        return False, str(exc)


def parse_iso_ts(value: str) -> float | None:
    try:
        return datetime.fromisoformat(value).timestamp()
    except Exception:  # noqa: BLE001
        return None


def parse_flags(args: str) -> dict[str, str | None]:
    out: dict[str, str | None] = {
        "output": None,
        "quantize": None,
        "backend": None,
        "shard": None,
        "shard_list": None,
    }
    try:
        toks = shlex.split(args)
    except Exception:  # noqa: BLE001
        toks = args.split()
    i = 0
    while i < len(toks):
        t = toks[i]
        if t in {"--output", "--quantize", "--backend", "--shard", "--shard-list"} and i + 1 < len(toks):
            key = t[2:].replace("-", "_")
            out[key] = toks[i + 1]
            i += 1
        elif t.startswith("--output="):
            out["output"] = t.split("=", 1)[1]
        elif t.startswith("--quantize="):
            out["quantize"] = t.split("=", 1)[1]
        elif t.startswith("--backend="):
            out["backend"] = t.split("=", 1)[1]
        elif t.startswith("--shard="):
            out["shard"] = t.split("=", 1)[1]
        elif t.startswith("--shard-list="):
            out["shard_list"] = t.split("=", 1)[1]
        i += 1
    return out


def parse_workers_from_ps(raw: str) -> list[dict[str, Any]]:
    workers: list[dict[str, Any]] = []
    for line in raw.splitlines():
        parts = line.strip().split(None, 2)
        if len(parts) != 3:
            continue
        pid_s, etimes_s, args = parts
        if not any(script in args for script in SWEEP_SCRIPTS):
            continue
        try:
            toks = shlex.split(args)
        except Exception:  # noqa: BLE001
            toks = args.split()
        if not toks or not Path(toks[0]).name.startswith("python"):
            continue
        try:
            pid = int(pid_s)
            etimes = int(etimes_s)
        except ValueError:
            continue
        flags = parse_flags(args)
        out = flags.get("output")
        if not out:
            continue
        workers.append(
            {
                "pid": pid,
                "etimes": etimes,
                "start_ts": datetime.now().timestamp() - etimes,
                "args": args,
                **flags,
            }
        )
    return workers


def collect_output_stats_local(output_dir: Path, start_ts: float) -> dict[str, Any]:
    stats = {
        "gen_responses": 0,
        "gen_tokens": 0,
        "replay_responses": 0,
        "replay_tokens": 0,
        "last_ts": None,
    }
    last_ts = 0.0

    gen_dir = output_dir / "generated"
    if gen_dir.exists():
        for f in gen_dir.glob("char_*.jsonl"):
            try:
                with f.open("r", encoding="utf-8", errors="ignore") as fh:
                    for line in fh:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            obj = json.loads(line)
                        except Exception:  # noqa: BLE001
                            continue
                        ts_raw = obj.get("timestamp")
                        if not isinstance(ts_raw, str):
                            continue
                        ts = parse_iso_ts(ts_raw)
                        if ts is None or ts < start_ts:
                            continue
                        stats["gen_responses"] += 1
                        try:
                            stats["gen_tokens"] += int(obj.get("n_gen_tokens") or 0)
                        except Exception:  # noqa: BLE001
                            pass
                        if ts > last_ts:
                            last_ts = ts
            except Exception:  # noqa: BLE001
                continue

    resp_dir = output_dir / "responses"
    if resp_dir.exists():
        for f in resp_dir.glob("char_*.jsonl"):
            try:
                with f.open("r", encoding="utf-8", errors="ignore") as fh:
                    for line in fh:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            obj = json.loads(line)
                        except Exception:  # noqa: BLE001
                            continue
                        ts_raw = obj.get("timestamp")
                        if not isinstance(ts_raw, str):
                            continue
                        ts = parse_iso_ts(ts_raw)
                        if ts is None or ts < start_ts:
                            continue
                        stats["replay_responses"] += 1
                        try:
                            stats["replay_tokens"] += int(obj.get("n_gen_tokens") or 0)
                        except Exception:  # noqa: BLE001
                            pass
                        if ts > last_ts:
                            last_ts = ts
            except Exception:  # noqa: BLE001
                continue

    if last_ts > 0:
        stats["last_ts"] = last_ts
    return stats


def collect_output_stats_remote(remote: str, output_dir: str, start_ts: float) -> dict[str, Any]:
    script = r"""
import json
from datetime import datetime
from pathlib import Path

out_dir = Path(__OUT_DIR__)
start_ts = float(__START_TS__)


def parse_iso_ts(v):
    try:
        return datetime.fromisoformat(v).timestamp()
    except Exception:
        return None

stats = {
    "gen_responses": 0,
    "gen_tokens": 0,
    "replay_responses": 0,
    "replay_tokens": 0,
    "last_ts": None,
}
last_ts = 0.0

for f in (out_dir / "generated").glob("char_*.jsonl") if (out_dir / "generated").exists() else []:
    try:
        with f.open("r", encoding="utf-8", errors="ignore") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                ts_raw = obj.get("timestamp")
                if not isinstance(ts_raw, str):
                    continue
                ts = parse_iso_ts(ts_raw)
                if ts is None or ts < start_ts:
                    continue
                stats["gen_responses"] += 1
                try:
                    stats["gen_tokens"] += int(obj.get("n_gen_tokens") or 0)
                except Exception:
                    pass
                if ts > last_ts:
                    last_ts = ts
    except Exception:
        pass

for f in (out_dir / "responses").glob("char_*.jsonl") if (out_dir / "responses").exists() else []:
    try:
        with f.open("r", encoding="utf-8", errors="ignore") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                ts_raw = obj.get("timestamp")
                if not isinstance(ts_raw, str):
                    continue
                ts = parse_iso_ts(ts_raw)
                if ts is None or ts < start_ts:
                    continue
                stats["replay_responses"] += 1
                try:
                    stats["replay_tokens"] += int(obj.get("n_gen_tokens") or 0)
                except Exception:
                    pass
                if ts > last_ts:
                    last_ts = ts
    except Exception:
        pass

if last_ts > 0:
    stats["last_ts"] = last_ts
print(json.dumps(stats))
"""
    script = script.replace("__OUT_DIR__", repr(output_dir)).replace("__START_TS__", repr(start_ts))
    cmd = (
        f"cd {shlex.quote(str(PROJECT_DIR))} && "
        "python3 - <<'PY'\n"
        + script
        + "\nPY"
    )
    ok, raw = run_remote(remote, cmd, timeout=150)
    if not ok:
        return {
            "gen_responses": 0,
            "gen_tokens": 0,
            "replay_responses": 0,
            "replay_tokens": 0,
            "last_ts": None,
            "error": raw,
        }
    try:
        return json.loads(raw)
    except Exception:  # noqa: BLE001
        return {
            "gen_responses": 0,
            "gen_tokens": 0,
            "replay_responses": 0,
            "replay_tokens": 0,
            "last_ts": None,
            "error": "bad remote json",
        }


def collect_workers_local() -> list[dict[str, Any]]:
    ok, out = run_local("ps -eo pid,etimes,args")
    if not ok:
        return []
    return parse_workers_from_ps(out)


def collect_workers_remote(remote: str) -> list[dict[str, Any]]:
    ok, out = run_remote(remote, "ps -eo pid,etimes,args")
    if not ok:
        return []
    return parse_workers_from_ps(out)


def main() -> None:
    parser = argparse.ArgumentParser(description="Show per-worker sweep tokens/sec")
    parser.add_argument("--remote", default=DEFAULT_REMOTE)
    parser.add_argument("--no-remote", action="store_true")
    args = parser.parse_args()

    workers: list[dict[str, Any]] = []
    for w in collect_workers_local():
        w["host"] = "workstation"
        workers.append(w)

    if not args.no_remote and args.remote:
        for w in collect_workers_remote(args.remote):
            w["host"] = "devserver"
            workers.append(w)

    if not workers:
        print("No active sweep workers found.")
        return

    rows: list[dict[str, Any]] = []
    total_pass1_tps = 0.0
    total_pass2_tps = 0.0

    for w in workers:
        out = str(w.get("output") or "")
        if not out:
            continue
        out_path = Path(out)
        if not out_path.is_absolute():
            out_path = PROJECT_DIR / out_path

        if w["host"] == "workstation":
            stats = collect_output_stats_local(out_path, w["start_ts"])
        else:
            stats = collect_output_stats_remote(args.remote, str(out_path), w["start_ts"])

        et = max(int(w["etimes"]), 1)
        pass1_tps = float(stats.get("gen_tokens", 0)) / et
        pass2_tps = float(stats.get("replay_tokens", 0)) / et
        total_pass1_tps += pass1_tps
        total_pass2_tps += pass2_tps

        rows.append(
            {
                "host": w["host"],
                "output": out,
                "shards": w.get("shard_list") or w.get("shard") or "?",
                "backend": w.get("backend") or "?",
                "pass1_tps": pass1_tps,
                "pass2_tps": pass2_tps,
                "gen_responses": int(stats.get("gen_responses", 0) or 0),
                "replay_responses": int(stats.get("replay_responses", 0) or 0),
                "quantize": w.get("quantize") or "?",
            }
        )

    rows.sort(key=lambda r: (r["host"], r["output"]))

    print("host        output                    shards    backend      quant   pass1 tok/s  pass2 tok/s  gen_resp  replay_resp")
    print("-" * 118)
    for r in rows:
        print(
            f"{r['host']:<10}  {r['output']:<24}  {r['shards']:<8}  {r['backend']:<10}  {r['quantize']:<5}  "
            f"{r['pass1_tps']:>11.1f}  {r['pass2_tps']:>11.1f}  {r['gen_responses']:>8}  {r['replay_responses']:>11}"
        )

    print("\nTotals")
    print(f"  pass1 aggregate tok/s: {total_pass1_tps:.1f}")
    print(f"  pass2 aggregate tok/s: {total_pass2_tps:.1f}")


if __name__ == "__main__":
    main()
