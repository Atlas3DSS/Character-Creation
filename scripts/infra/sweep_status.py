#!/usr/bin/env python3
"""
Unified sweep status view across workstation + optional dev server.

Shows per-GPU:
  - VRAM usage
  - Active personality_sweep_v2 process
  - Quantization mode
  - Estimated minutes/response
  - Stalled/idle/error state

Usage:
  python3 scripts/infra/sweep_status.py
  python3 scripts/infra/sweep_status.py --remote orwel@192.168.86.66
  watch -n 15 "python3 scripts/infra/sweep_status.py"
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PROJECT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_REMOTE = os.environ.get("REMOTE_HOST", "orwel@192.168.1.90")
SWEEP_SCRIPTS = {"personality_sweep_v2.py", "personality_sweep_v3_two_pass.py"}


@dataclass
class HostSpec:
    name: str
    ssh_target: str | None = None

    @property
    def is_remote(self) -> bool:
        return self.ssh_target is not None


def run_cmd(host: HostSpec, cmd: str, timeout: int = 10) -> tuple[bool, str]:
    try:
        if host.is_remote:
            proc = subprocess.run(
                [
                    "ssh",
                    "-o",
                    "BatchMode=yes",
                    "-o",
                    "ConnectTimeout=4",
                    host.ssh_target or "",
                    cmd,
                ],
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        else:
            proc = subprocess.run(
                ["bash", "-lc", cmd],
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        if proc.returncode != 0:
            msg = (proc.stderr or proc.stdout or "command failed").strip()
            return False, msg
        return True, proc.stdout
    except Exception as exc:  # noqa: BLE001
        return False, str(exc)


def parse_csv_lines(text: str) -> list[list[str]]:
    rows: list[list[str]] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        rows.append([p.strip() for p in line.split(",")])
    return rows


def get_gpu_stats(host: HostSpec) -> tuple[list[dict[str, Any]], str | None]:
    ok, out = run_cmd(
        host,
        "nvidia-smi --query-gpu=index,uuid,name,utilization.gpu,memory.used,memory.total "
        "--format=csv,noheader,nounits",
    )
    if not ok:
        return [], out
    rows = parse_csv_lines(out)
    gpus: list[dict[str, Any]] = []
    for parts in rows:
        if len(parts) < 6:
            continue
        idx, uuid, name, util, mem_used, mem_total = parts[:6]
        try:
            gpus.append(
                {
                    "host": host.name,
                    "index": int(idx),
                    "uuid": uuid,
                    "name": name,
                    "util_pct": float(util),
                    "mem_used_mb": float(mem_used),
                    "mem_total_mb": float(mem_total),
                }
            )
        except ValueError:
            continue
    return gpus, None


def get_compute_map(host: HostSpec) -> dict[int, dict[str, Any]]:
    ok, out = run_cmd(
        host,
        "nvidia-smi --query-compute-apps=pid,gpu_uuid,used_memory "
        "--format=csv,noheader,nounits",
    )
    if not ok:
        return {}
    mapping: dict[int, dict[str, Any]] = {}
    for parts in parse_csv_lines(out):
        if len(parts) < 2:
            continue
        try:
            pid = int(parts[0])
        except ValueError:
            continue
        used_mem = 0.0
        if len(parts) > 2:
            try:
                used_mem = float(parts[2])
            except ValueError:
                used_mem = 0.0
        mapping[pid] = {"gpu_uuid": parts[1], "app_mem_mb": used_mem}
    return mapping


def parse_process_flags(cmd: str) -> dict[str, Any]:
    out: dict[str, Any] = {
        "output": None,
        "quantize": None,
        "backend": None,
        "shard": None,
        "n_shards": None,
        "shard_list": None,
        "cuda_visible_devices": None,
    }
    try:
        toks = shlex.split(cmd)
    except Exception:  # noqa: BLE001
        toks = cmd.split()
    i = 0
    while i < len(toks):
        tok = toks[i]
        if tok.startswith("CUDA_VISIBLE_DEVICES="):
            out["cuda_visible_devices"] = tok.split("=", 1)[1]
        elif tok in {"--output", "--quantize", "--backend", "--shard", "--n-shards", "--shard-list"}:
            if i + 1 < len(toks):
                val = toks[i + 1]
                key = tok.lstrip("-").replace("-", "_")
                out[key] = val
                i += 1
        elif tok.startswith("--output="):
            out["output"] = tok.split("=", 1)[1]
        elif tok.startswith("--quantize="):
            out["quantize"] = tok.split("=", 1)[1]
        elif tok.startswith("--backend="):
            out["backend"] = tok.split("=", 1)[1]
        elif tok.startswith("--shard="):
            out["shard"] = tok.split("=", 1)[1]
        elif tok.startswith("--n-shards="):
            out["n_shards"] = tok.split("=", 1)[1]
        elif tok.startswith("--shard-list="):
            out["shard_list"] = tok.split("=", 1)[1]
        i += 1
    return out


def get_sweep_processes(host: HostSpec) -> list[dict[str, Any]]:
    ok, out = run_cmd(host, "ps -eo pid,etimes,args")
    if not ok:
        return []
    procs: list[dict[str, Any]] = []
    for line in out.splitlines():
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
        if not toks:
            continue
        exe = Path(toks[0]).name
        if not exe.startswith("python"):
            continue
        try:
            pid = int(pid_s)
            etimes = int(etimes_s)
        except ValueError:
            continue
        flags = parse_process_flags(args)
        procs.append(
            {
                "pid": pid,
                "etimes": etimes,
                "args": args,
                **flags,
            }
        )
    return procs


def resolve_output_dir(output: str | None) -> Path | None:
    if not output:
        return None
    p = Path(output)
    if p.is_absolute():
        return p
    return PROJECT_DIR / p


def progress_local(output_dir: Path) -> dict[str, Any]:
    data: dict[str, Any] = {"responses": 0, "generated": 0, "last_mtime": None, "config": {}}
    cfg = output_dir / "sweep_config.json"
    if cfg.exists():
        try:
            data["config"] = json.loads(cfg.read_text())
        except Exception:  # noqa: BLE001
            data["config"] = {}

    resp_dir = output_dir / "responses"
    gen_dir = output_dir / "generated"
    total_resp = 0
    total_gen = 0
    last_mtime = 0.0
    if resp_dir.exists():
        for f in resp_dir.glob("char_*.jsonl"):
            try:
                with f.open("rb") as fh:
                    total_resp += sum(1 for _ in fh)
                mtime = f.stat().st_mtime
                if mtime > last_mtime:
                    last_mtime = mtime
            except Exception:  # noqa: BLE001
                continue
    if gen_dir.exists():
        for f in gen_dir.glob("char_*.jsonl"):
            try:
                with f.open("rb") as fh:
                    total_gen += sum(1 for _ in fh)
                mtime = f.stat().st_mtime
                if mtime > last_mtime:
                    last_mtime = mtime
            except Exception:  # noqa: BLE001
                continue
    data["responses"] = total_resp
    data["generated"] = total_gen
    data["last_mtime"] = last_mtime if last_mtime > 0 else None
    return data


def progress_remote(host: HostSpec, output_dir: Path) -> dict[str, Any]:
    script = r"""
import json
import sys
from pathlib import Path

p = Path(sys.argv[1])
out = {"responses": 0, "generated": 0, "last_mtime": None, "config": {}}
cfg = p / "sweep_config.json"
if cfg.exists():
    try:
        out["config"] = json.loads(cfg.read_text())
    except Exception:
        out["config"] = {}
resp_dir = p / "responses"
gen_dir = p / "generated"
last = 0.0
if resp_dir.exists():
    total = 0
    for f in resp_dir.glob("char_*.jsonl"):
        try:
            with f.open("rb") as fh:
                total += sum(1 for _ in fh)
            mt = f.stat().st_mtime
            if mt > last:
                last = mt
        except Exception:
            pass
    out["responses"] = total
if gen_dir.exists():
    total = 0
    for f in gen_dir.glob("char_*.jsonl"):
        try:
            with f.open("rb") as fh:
                total += sum(1 for _ in fh)
            mt = f.stat().st_mtime
            if mt > last:
                last = mt
        except Exception:
            pass
    out["generated"] = total
if last > 0:
    out["last_mtime"] = last
print(json.dumps(out))
"""
    cmd = f"python3 -c {shlex.quote(script)} {shlex.quote(str(output_dir))}"
    ok, raw = run_cmd(host, cmd, timeout=20)
    if not ok:
        return {"responses": 0, "generated": 0, "last_mtime": None, "config": {}, "error": raw}
    try:
        return json.loads(raw)
    except Exception:  # noqa: BLE001
        return {"responses": 0, "generated": 0, "last_mtime": None, "config": {}, "error": "bad json"}


def estimate_status(proc: dict[str, Any], progress: dict[str, Any]) -> tuple[str, str]:
    now = time.time()
    elapsed_min = max(proc["etimes"] / 60.0, 1e-6)
    responses = int(progress.get("responses", 0) or 0)
    generated = int(progress.get("generated", 0) or 0)
    config = progress.get("config", {}) or {}
    total_expected = None
    if config:
        n_chars = config.get("n_characters_this_shard")
        n_prompts = config.get("n_prompts_per_char")
        if isinstance(n_chars, int) and isinstance(n_prompts, int):
            total_expected = n_chars * n_prompts
    pipeline = config.get("pipeline")
    is_two_pass = isinstance(pipeline, str) and "two_pass" in pipeline
    if is_two_pass:
        pass1_rate = generated / elapsed_min
        pass2_rate = responses / elapsed_min
    else:
        pass1_rate = 0.0
        pass2_rate = responses / elapsed_min
    last_mtime = progress.get("last_mtime")
    idle_age_sec = None
    if last_mtime:
        idle_age_sec = max(0.0, now - float(last_mtime))

    if total_expected is not None and responses >= total_expected:
        return "Completed", "-"
    if responses == 0 and generated == 0 and proc["etimes"] < 180:
        return "Starting...", "-"
    if idle_age_sec is not None and idle_age_sec > 900:
        return f"Stalled ({int(idle_age_sec // 60)}m no writes)", "-"
    if is_two_pass and responses == 0 and pass1_rate > 0:
        min_per_resp = 1.0 / pass1_rate
        eta = "-"
        if total_expected is not None and generated < total_expected:
            remain = total_expected - generated
            eta = f"{(remain / pass1_rate):.0f}m ETA(pass1)"
        return f"Generating pass1, ~{min_per_resp:.2f} min/resp", eta
    if pass2_rate > 0:
        min_per_resp = 1.0 / pass2_rate
        eta = "-"
        if total_expected is not None and responses < total_expected:
            remain = total_expected - responses
            eta = f"{(remain / pass2_rate):.0f}m ETA"
        label = "Generating pass2" if is_two_pass else "Generating"
        return f"{label}, ~{min_per_resp:.2f} min/resp", eta
    if proc["etimes"] >= 180:
        return "Warm but no outputs", "-"
    return "Starting...", "-"


def format_table(rows: list[dict[str, str]]) -> str:
    headers = ["Host", "GPU", "VRAM Used/Total", "Quantize", "Status", "ETA"]
    table = [headers]
    for r in rows:
        table.append(
            [
                r.get("host", ""),
                r.get("gpu", ""),
                r.get("vram", ""),
                r.get("quantize", ""),
                r.get("status", ""),
                r.get("eta", ""),
            ]
        )
    widths = [max(len(row[i]) for row in table) for i in range(len(headers))]

    def border(left: str, mid: str, right: str) -> str:
        return left + mid.join("─" * (w + 2) for w in widths) + right

    out = [border("┌", "┬", "┐")]
    out.append(
        "│ "
        + " │ ".join(headers[i].ljust(widths[i]) for i in range(len(headers)))
        + " │"
    )
    out.append(border("├", "┼", "┤"))
    for row in table[1:]:
        out.append(
            "│ "
            + " │ ".join(row[i].ljust(widths[i]) for i in range(len(headers)))
            + " │"
        )
    out.append(border("└", "┴", "┘"))
    return "\n".join(out)


def build_rows(host: HostSpec) -> list[dict[str, str]]:
    gpus, gpu_err = get_gpu_stats(host)
    if gpu_err:
        return [
            {
                "host": host.name,
                "gpu": "(unreachable)",
                "vram": "-",
                "quantize": "-",
                "status": gpu_err.splitlines()[0][:70],
                "eta": "-",
            }
        ]

    uuid_to_gpu = {g["uuid"]: g for g in gpus}
    idx_to_gpu = {g["index"]: g for g in gpus}
    rows: dict[tuple[str, int], dict[str, str]] = {}
    for g in gpus:
        used_gb = g["mem_used_mb"] / 1024.0
        total_gb = g["mem_total_mb"] / 1024.0
        key = (host.name, g["index"])
        rows[key] = {
            "host": host.name,
            "gpu": g["name"],
            "vram": f"{used_gb:.1f}/{total_gb:.0f} GB",
            "quantize": "-",
            "status": "Idle"
            if g["util_pct"] < 10 and g["mem_used_mb"] < 1000
            else f"Busy ({g['util_pct']:.0f}% util)",
            "eta": "-",
            "_workers": [],
        }

    compute = get_compute_map(host)
    procs = get_sweep_processes(host)
    for proc in procs:
        gpu = None
        compute_entry = compute.get(proc["pid"])
        if compute_entry:
            gpu = uuid_to_gpu.get(compute_entry["gpu_uuid"])
        if gpu is None and proc.get("cuda_visible_devices"):
            dev_tok = str(proc["cuda_visible_devices"]).split(",")[0].strip()
            if dev_tok.isdigit():
                gpu = idx_to_gpu.get(int(dev_tok))
        if gpu is None and len(gpus) == 1:
            gpu = gpus[0]
        if gpu is None:
            continue

        output_dir = resolve_output_dir(proc.get("output"))
        progress = {"responses": 0, "last_mtime": None, "config": {}}
        if output_dir is not None:
            if host.is_remote:
                progress = progress_remote(host, output_dir)
            else:
                progress = progress_local(output_dir)

        status, eta = estimate_status(proc, progress)
        quant = proc.get("quantize")
        if not quant:
            cfg = progress.get("config", {}) or {}
            q = cfg.get("quantization")
            if isinstance(q, str):
                quant = q
        key = (host.name, gpu["index"])
        row = rows.get(key)
        if row is None:
            continue
        row["_workers"].append(
            {
                "quantize": quant or "?",
                "status": status,
                "eta": eta,
            }
        )

    final_rows: list[dict[str, str]] = []
    for key in sorted(rows):
        row = rows[key]
        workers = row.pop("_workers", [])
        if workers:
            quant_set = sorted({w["quantize"] for w in workers if w["quantize"]})
            row["quantize"] = "+".join(quant_set) if quant_set else "?"
            n = len(workers)
            if n == 1:
                row["status"] = workers[0]["status"]
                row["eta"] = workers[0]["eta"]
            else:
                statuses = [w["status"] for w in workers]
                if any(s.startswith("Generating pass2") for s in statuses):
                    row["status"] = f"Generating pass2 ({n} workers)"
                elif any(s.startswith("Generating pass1") for s in statuses):
                    row["status"] = f"Generating pass1 ({n} workers)"
                elif any(s.startswith("Generating") for s in statuses):
                    row["status"] = f"Generating ({n} workers)"
                elif all("Starting" in s for s in statuses):
                    row["status"] = f"Starting ({n} workers)"
                elif any("Stalled" in s for s in statuses):
                    row["status"] = f"Partially stalled ({n} workers)"
                else:
                    row["status"] = f"Active ({n} workers)"
                row["eta"] = "-"
        final_rows.append(row)
    return final_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Show sweep status across GPUs")
    parser.add_argument("--remote", default=DEFAULT_REMOTE, help="SSH target for dev server")
    parser.add_argument(
        "--no-remote",
        action="store_true",
        help="Only query local workstation",
    )
    args = parser.parse_args()

    hosts = [HostSpec(name="workstation", ssh_target=None)]
    if not args.no_remote and args.remote:
        hosts.append(HostSpec(name="devserver", ssh_target=args.remote))

    all_rows: list[dict[str, str]] = []
    for host in hosts:
        all_rows.extend(build_rows(host))

    print(format_table(all_rows))


if __name__ == "__main__":
    main()
