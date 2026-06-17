#!/usr/bin/env python3
"""
Hourly sweep tuner for local worker profile (4w vs 5w) using observed tokens/sec.

Safety rules:
  - Only switches profiles early in a run (response-count guard) to avoid re-shard waste.
  - Uses cooldown between switches.
  - Locks a profile once enough evidence is collected.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_REMOTE = os.environ.get("REMOTE_HOST", "orwel@192.168.1.90")
LAUNCHER = PROJECT_DIR / "scripts/infra/launch_sweep_v2_hetero.sh"
STATE_PATH = PROJECT_DIR / "sweep_v2" / "autotune_state.json"
METRICS_PATH = PROJECT_DIR / "logs" / "sweep_autotune_metrics.jsonl"
SWEEP_NAME = "personality_sweep_v2.py"

PROFILE_OUTPUTS = {
    "3w": ["sweep_v2/ws4_a", "sweep_v2/ws4_b", "sweep_v2/ws4_c"],
    "4w": ["sweep_v2/ws4_a", "sweep_v2/ws4_b", "sweep_v2/ws4_c", "sweep_v2/ws4_d"],
    "5w": ["sweep_v2/ws4_a", "sweep_v2/ws4_b", "sweep_v2/ws4_c", "sweep_v2/ws4_d", "sweep_v2/ws4_e"],
}


def run_local(cmd: str, timeout: int = 120) -> tuple[bool, str]:
    try:
        proc = subprocess.run(
            ["bash", "-lc", cmd],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(PROJECT_DIR),
        )
        if proc.returncode != 0:
            return False, (proc.stderr or proc.stdout or "command failed").strip()
        return True, proc.stdout
    except Exception as exc:  # noqa: BLE001
        return False, str(exc)


def run_remote(remote: str, cmd: str, timeout: int = 120) -> tuple[bool, str]:
    try:
        proc = subprocess.run(
            ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=6", remote, cmd],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if proc.returncode != 0:
            return False, (proc.stderr or proc.stdout or "ssh command failed").strip()
        return True, proc.stdout
    except Exception as exc:  # noqa: BLE001
        return False, str(exc)


def parse_flags(args: str) -> dict[str, str | None]:
    out: dict[str, str | None] = {"output": None, "quantize": None, "shard_list": None}
    try:
        toks = shlex.split(args)
    except Exception:  # noqa: BLE001
        toks = args.split()
    i = 0
    while i < len(toks):
        t = toks[i]
        if t in {"--output", "--quantize", "--shard-list"} and i + 1 < len(toks):
            key = t[2:].replace("-", "_")
            out[key] = toks[i + 1]
            i += 1
        elif t.startswith("--output="):
            out["output"] = t.split("=", 1)[1]
        elif t.startswith("--quantize="):
            out["quantize"] = t.split("=", 1)[1]
        elif t.startswith("--shard-list="):
            out["shard_list"] = t.split("=", 1)[1]
        i += 1
    return out


def parse_iso_ts(value: str) -> float | None:
    try:
        return datetime.fromisoformat(value).timestamp()
    except Exception:  # noqa: BLE001
        return None


def collect_output_stats(output_dir: Path, start_ts: float) -> dict[str, Any]:
    resp_dir = output_dir / "responses"
    stats = {"responses": 0, "tokens": 0, "last_ts": None}
    if not resp_dir.exists():
        return stats

    last_ts = 0.0
    for f in resp_dir.glob("char_*.jsonl"):
        try:
            st = f.stat()
        except Exception:  # noqa: BLE001
            continue
        if st.st_mtime + 1 < start_ts:
            continue
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
                    stats["responses"] += 1
                    try:
                        stats["tokens"] += int(obj.get("n_gen_tokens") or 0)
                    except Exception:  # noqa: BLE001
                        pass
                    if ts > last_ts:
                        last_ts = ts
        except Exception:  # noqa: BLE001
            continue
    if last_ts > 0:
        stats["last_ts"] = last_ts
    return stats


def collect_local_workers(now: float) -> list[dict[str, Any]]:
    ok, out = run_local("ps -eo pid,etimes,args")
    if not ok:
        return []
    workers: list[dict[str, Any]] = []
    for line in out.splitlines():
        parts = line.strip().split(None, 2)
        if len(parts) != 3:
            continue
        pid_s, etime_s, args = parts
        if SWEEP_NAME not in args:
            continue
        try:
            toks = shlex.split(args)
        except Exception:  # noqa: BLE001
            toks = args.split()
        if not toks or not Path(toks[0]).name.startswith("python"):
            continue
        flags = parse_flags(args)
        output = flags.get("output")
        if not output or not output.startswith("sweep_v2/ws4_"):
            continue
        try:
            pid = int(pid_s)
            etimes = int(etime_s)
        except ValueError:
            continue
        start_ts = now - etimes
        stats = collect_output_stats(PROJECT_DIR / output, start_ts)
        workers.append(
            {
                "pid": pid,
                "etimes": etimes,
                "start_ts": start_ts,
                "output": output,
                "quantize": flags.get("quantize"),
                "shard_list": flags.get("shard_list"),
                **stats,
                "tps": float(stats["tokens"]) / max(etimes, 1),
            }
        )
    workers.sort(key=lambda w: w["output"])
    return workers


def collect_remote_workers(now: float, remote: str) -> tuple[list[dict[str, Any]], str | None]:
    script = r"""
import json
import os
import shlex
from datetime import datetime
from pathlib import Path

now = __NOW__

def parse_flags(args):
    out = {"output": None, "quantize": None, "shard_list": None}
    try:
        toks = shlex.split(args)
    except Exception:
        toks = args.split()
    i = 0
    while i < len(toks):
        t = toks[i]
        if t in {"--output", "--quantize", "--shard-list"} and i + 1 < len(toks):
            out[t[2:].replace("-", "_")] = toks[i + 1]
            i += 1
        elif t.startswith("--output="):
            out["output"] = t.split("=", 1)[1]
        elif t.startswith("--quantize="):
            out["quantize"] = t.split("=", 1)[1]
        elif t.startswith("--shard-list="):
            out["shard_list"] = t.split("=", 1)[1]
        i += 1
    return out

def parse_iso_ts(value):
    try:
        return datetime.fromisoformat(value).timestamp()
    except Exception:
        return None

def collect_output_stats(output_dir, start_ts):
    stats = {"responses": 0, "tokens": 0, "last_ts": None}
    resp_dir = Path(output_dir) / "responses"
    if not resp_dir.exists():
        return stats
    last_ts = 0.0
    for f in resp_dir.glob("char_*.jsonl"):
        try:
            st = f.stat()
        except Exception:
            continue
        if st.st_mtime + 1 < start_ts:
            continue
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
                    stats["responses"] += 1
                    try:
                        stats["tokens"] += int(obj.get("n_gen_tokens") or 0)
                    except Exception:
                        pass
                    if ts > last_ts:
                        last_ts = ts
        except Exception:
            continue
    if last_ts > 0:
        stats["last_ts"] = last_ts
    return stats

workers = []
for line in os.popen("ps -eo pid,etimes,args").read().splitlines():
    parts = line.strip().split(None, 2)
    if len(parts) != 3:
        continue
    pid_s, etime_s, args = parts
    if "personality_sweep_v2.py" not in args:
        continue
    try:
        toks = shlex.split(args)
    except Exception:
        toks = args.split()
    if not toks or not Path(toks[0]).name.startswith("python"):
        continue
    flags = parse_flags(args)
    output = flags.get("output")
    if not output or not output.startswith("sweep_v2/dev_"):
        continue
    try:
        pid = int(pid_s); etimes = int(etime_s)
    except ValueError:
        continue
    start_ts = now - etimes
    stats = collect_output_stats(output, start_ts)
    workers.append({
        "pid": pid,
        "etimes": etimes,
        "start_ts": start_ts,
        "output": output,
        "quantize": flags.get("quantize"),
        "shard_list": flags.get("shard_list"),
        **stats,
        "tps": float(stats["tokens"]) / max(etimes, 1),
    })
workers.sort(key=lambda w: w["output"])
print(json.dumps({"workers": workers}))
"""
    script = script.replace("__NOW__", repr(now))
    cmd = (
        f"cd {shlex.quote(str(PROJECT_DIR))} && "
        "python3 - <<'PY'\n"
        + script
        + "\nPY"
    )
    ok, raw = run_remote(remote, cmd, timeout=150)
    if not ok:
        return [], raw
    try:
        data = json.loads(raw)
        return list(data.get("workers", [])), None
    except Exception as exc:  # noqa: BLE001
        return [], f"bad remote json: {exc}"


def detect_local_profile(workers: list[dict[str, Any]]) -> str | None:
    outs = {w["output"] for w in workers}
    by_out = {w["output"]: w for w in workers}
    if {"sweep_v2/ws4_a", "sweep_v2/ws4_b", "sweep_v2/ws4_c"}.issubset(outs) and "sweep_v2/ws4_d" not in outs:
        a = str(by_out["sweep_v2/ws4_a"].get("shard_list") or "")
        b = str(by_out["sweep_v2/ws4_b"].get("shard_list") or "")
        c = str(by_out["sweep_v2/ws4_c"].get("shard_list") or "")
        if a == "0,1,2" and b == "3,4,5" and c == "6,7":
            return "3w"
    if "sweep_v2/ws4_e" in outs:
        return "5w"
    if "sweep_v2/ws4_d" in outs:
        for w in workers:
            if w["output"] == "sweep_v2/ws4_d":
                shard_list = str(w.get("shard_list") or "")
                if "," in shard_list and "7" in shard_list:
                    return "4w"
    if len(workers) == 4:
        return "4w"
    if len(workers) >= 5:
        return "5w"
    return None


def local_warm(
    workers: list[dict[str, Any]],
    profile: str | None,
    min_elapsed: int,
    min_responses_per_worker: int,
) -> tuple[bool, str]:
    if profile not in PROFILE_OUTPUTS:
        return False, "unknown profile"
    by_out = {w["output"]: w for w in workers}
    missing = [o for o in PROFILE_OUTPUTS[profile] if o not in by_out]
    if missing:
        return False, f"missing workers: {','.join(missing)}"
    for out in PROFILE_OUTPUTS[profile]:
        w = by_out[out]
        if int(w["etimes"]) < min_elapsed:
            return False, f"{out} not warmed ({w['etimes']}s)"
        if int(w["responses"]) < min_responses_per_worker:
            return False, f"{out} has {w['responses']} responses"
    return True, "ok"


def load_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception:  # noqa: BLE001
        return {}


def save_state(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2, sort_keys=True))


def append_metric(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, sort_keys=True) + "\n")


def kill_local_workers() -> tuple[bool, str]:
    cmd = r"""python3 - <<'PY'
import os
import signal
k = []
for pid in os.listdir('/proc'):
    if not pid.isdigit():
        continue
    p = int(pid)
    if p in (os.getpid(), os.getppid()):
        continue
    try:
        cmd = open(f'/proc/{pid}/cmdline', 'rb').read().replace(b'\x00', b' ').decode('utf-8', 'ignore')
    except Exception:
        continue
    if 'personality_sweep_v2.py' in cmd and 'sweep_v2/ws4_' in cmd:
        try:
            os.kill(p, signal.SIGKILL)
            k.append((p, cmd))
        except Exception:
            pass
print('killed', len(k))
for p, c in k:
    print(p, c)
PY"""
    return run_local(cmd, timeout=30)


def launch_profile(profile: str, remote: str) -> tuple[bool, str]:
    cmd = (
        f'REMOTE_HOST={shlex.quote(remote)} LOCAL_PROFILE={shlex.quote(profile)} '
        f'bash {shlex.quote(str(LAUNCHER))}'
    )
    return run_local(cmd, timeout=180)


def main() -> None:
    parser = argparse.ArgumentParser(description="Autotune local sweep worker profile.")
    parser.add_argument("--remote", default=DEFAULT_REMOTE, help="Remote SSH target")
    parser.add_argument("--enable-switch", action="store_true", help="Allow profile switches")
    parser.add_argument("--switch-cooldown-min", type=int, default=45)
    parser.add_argument("--safe-reshard-max-responses", type=int, default=240)
    parser.add_argument("--min-elapsed-sec", type=int, default=360)
    parser.add_argument("--min-responses-per-worker", type=int, default=1)
    parser.add_argument("--improvement-threshold", type=float, default=0.08)
    parser.add_argument("--ewma-alpha", type=float, default=0.40)
    parser.add_argument("--state-path", default=str(STATE_PATH))
    parser.add_argument("--metrics-path", default=str(METRICS_PATH))
    args = parser.parse_args()

    now = time.time()
    state_path = Path(args.state_path)
    metrics_path = Path(args.metrics_path)
    state = load_state(state_path)

    local_workers = collect_local_workers(now)
    remote_workers, remote_err = collect_remote_workers(now, args.remote)
    active_profile = detect_local_profile(local_workers)

    local_tps = sum(float(w["tps"]) for w in local_workers)
    remote_tps = sum(float(w["tps"]) for w in remote_workers)
    total_tps = local_tps + remote_tps
    local_responses = sum(int(w["responses"]) for w in local_workers)
    warm, warm_reason = local_warm(
        local_workers,
        active_profile,
        args.min_elapsed_sec,
        args.min_responses_per_worker,
    )

    ewma = state.setdefault("ewma_local_tps", {})
    trials = state.setdefault("trials", {})
    if active_profile in {"4w", "5w"} and warm:
        prev = ewma.get(active_profile)
        if prev is None:
            ewma[active_profile] = local_tps
        else:
            ewma[active_profile] = float(args.ewma_alpha) * local_tps + (1.0 - float(args.ewma_alpha)) * float(prev)
        trials[active_profile] = True

    row = {
        "ts": datetime.now().isoformat(),
        "active_profile": active_profile,
        "local_tps": round(local_tps, 4),
        "remote_tps": round(remote_tps, 4),
        "total_tps": round(total_tps, 4),
        "local_responses_since_start": local_responses,
        "warm": warm,
        "warm_reason": warm_reason,
        "local_workers": [
            {
                "output": w["output"],
                "shard_list": w.get("shard_list"),
                "quantize": w.get("quantize"),
                "responses": int(w["responses"]),
                "tokens": int(w["tokens"]),
                "tps": round(float(w["tps"]), 4),
            }
            for w in local_workers
        ],
        "remote_workers": [
            {
                "output": w["output"],
                "shard_list": w.get("shard_list"),
                "quantize": w.get("quantize"),
                "responses": int(w["responses"]),
                "tokens": int(w["tokens"]),
                "tps": round(float(w["tps"]), 4),
            }
            for w in remote_workers
        ],
        "remote_error": remote_err,
    }
    append_metric(metrics_path, row)

    print(f"[AUTOTUNE] profile={active_profile} local_tps={local_tps:.2f} remote_tps={remote_tps:.2f} total_tps={total_tps:.2f}")
    if remote_err:
        print(f"[AUTOTUNE] remote warning: {remote_err}")
    print(f"[AUTOTUNE] warm={warm} ({warm_reason}), local_responses_since_start={local_responses}")

    if not args.enable_switch:
        state["active_profile"] = active_profile
        state["last_eval_ts"] = now
        save_state(state_path, state)
        return

    if active_profile not in {"4w", "5w"}:
        print("[AUTOTUNE] no switch: unknown active profile.")
        state["active_profile"] = active_profile
        state["last_eval_ts"] = now
        save_state(state_path, state)
        return

    locked = state.get("locked_profile")
    if locked:
        print(f"[AUTOTUNE] no switch: profile locked to {locked}.")
        state["active_profile"] = active_profile
        state["last_eval_ts"] = now
        save_state(state_path, state)
        return

    if local_responses > args.safe_reshard_max_responses:
        state["locked_profile"] = active_profile
        print(
            "[AUTOTUNE] locking profile "
            f"{active_profile}: response guard exceeded ({local_responses} > {args.safe_reshard_max_responses})."
        )
        state["active_profile"] = active_profile
        state["last_eval_ts"] = now
        save_state(state_path, state)
        return

    if not warm:
        print("[AUTOTUNE] no switch: workers not warm enough.")
        state["active_profile"] = active_profile
        state["last_eval_ts"] = now
        save_state(state_path, state)
        return

    last_switch = float(state.get("last_switch_ts", 0.0) or 0.0)
    cooldown_sec = max(0, args.switch_cooldown_min) * 60
    if now - last_switch < cooldown_sec:
        left = int((cooldown_sec - (now - last_switch)) // 60) + 1
        print(f"[AUTOTUNE] no switch: cooldown active (~{left}m left).")
        state["active_profile"] = active_profile
        state["last_eval_ts"] = now
        save_state(state_path, state)
        return

    other = "4w" if active_profile == "5w" else "5w"
    target: str | None = None
    reason = ""

    if not trials.get(other):
        target = other
        reason = f"trial profile {other}"
    else:
        active_ewma = float(ewma.get(active_profile, 0.0) or 0.0)
        other_ewma = float(ewma.get(other, 0.0) or 0.0)
        if other_ewma > active_ewma * (1.0 + args.improvement_threshold):
            target = other
            reason = f"better ewma ({other_ewma:.2f} > {active_ewma:.2f})"
        else:
            best = active_profile if active_ewma >= other_ewma else other
            state["locked_profile"] = best
            print(f"[AUTOTUNE] locking profile {best} (ewma local t/s: 4w={ewma.get('4w')}, 5w={ewma.get('5w')}).")

    if target and target != active_profile:
        print(f"[AUTOTUNE] switching {active_profile} -> {target} ({reason}).")
        ok, msg = kill_local_workers()
        if not ok:
            print(f"[AUTOTUNE] local kill failed: {msg}")
        ok, launch_out = launch_profile(target, args.remote)
        if ok:
            print("[AUTOTUNE] launch complete.")
            print(launch_out.strip())
            state["last_switch_ts"] = now
            state["active_profile"] = target
        else:
            print(f"[AUTOTUNE] launch failed: {launch_out}")
            state["active_profile"] = active_profile
    else:
        state["active_profile"] = active_profile

    state["last_eval_ts"] = now
    save_state(state_path, state)


if __name__ == "__main__":
    main()
