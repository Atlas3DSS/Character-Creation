#!/usr/bin/env python3
"""
GPU monitoring daemon that writes idle alerts to a status file.
Checks both WSL (local) and dev server GPUs periodically.
Claude Code can read the status file to know when GPUs are available.

Usage:
    python gpu_monitor.py  # Runs forever, checks every 30s
"""

import json
import os
import subprocess
import time
from datetime import datetime
from pathlib import Path

STATUS_FILE = Path("/tmp/gpu_idle_status.json")
CHECK_INTERVAL = 30  # seconds
IDLE_THRESHOLD_PCT = 10  # GPU util below this = idle
IDLE_VRAM_MB = 500  # VRAM below this = truly idle (no model loaded)

DEV_SERVER = "orwel@192.168.86.66"

# CUDA vs nvidia-smi mapping for dev server
# nvidia-smi: GPU0=3090, GPU1=4090
# CUDA: device 0=4090, device 1=3090
DEV_SERVER_GPU_MAP = {
    "0": {"name": "RTX 3090", "cuda_id": 1, "vram_gb": 24},
    "1": {"name": "RTX 4090", "cuda_id": 0, "vram_gb": 24},
}


def check_local_gpu() -> list[dict]:
    """Check WSL GPU status via nvidia-smi."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,utilization.gpu,memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10
        )
        gpus = []
        for line in result.stdout.strip().split("\n"):
            if not line.strip():
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 5:
                idx, name, util, mem_used, mem_total = parts[:5]
                util_pct = float(util)
                mem_used_mb = float(mem_used)
                mem_total_mb = float(mem_total)
                is_idle = util_pct < IDLE_THRESHOLD_PCT and mem_used_mb < IDLE_VRAM_MB
                gpus.append({
                    "host": "wsl",
                    "index": int(idx),
                    "name": name,
                    "util_pct": util_pct,
                    "vram_used_mb": mem_used_mb,
                    "vram_total_mb": mem_total_mb,
                    "is_idle": is_idle,
                    "has_model": mem_used_mb > 2000,  # >2GB = model loaded
                })
        return gpus
    except Exception as e:
        return [{"host": "wsl", "error": str(e)}]


def check_remote_gpu() -> list[dict]:
    """Check dev server GPU status via SSH."""
    try:
        result = subprocess.run(
            ["ssh", "-o", "ConnectTimeout=5", DEV_SERVER,
             "nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total "
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=15
        )
        if result.returncode != 0:
            return [{"host": "devserver", "error": f"SSH failed: {result.stderr[:100]}"}]

        gpus = []
        for line in result.stdout.strip().split("\n"):
            if not line.strip():
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 5:
                idx, name, util, mem_used, mem_total = parts[:5]
                util_pct = float(util)
                mem_used_mb = float(mem_used)
                mem_total_mb = float(mem_total)
                is_idle = util_pct < IDLE_THRESHOLD_PCT and mem_used_mb < IDLE_VRAM_MB

                gpu_info = DEV_SERVER_GPU_MAP.get(idx, {})
                cuda_id = gpu_info.get("cuda_id", int(idx))

                gpus.append({
                    "host": "devserver",
                    "index": int(idx),
                    "cuda_id": cuda_id,
                    "name": name,
                    "util_pct": util_pct,
                    "vram_used_mb": mem_used_mb,
                    "vram_total_mb": mem_total_mb,
                    "is_idle": is_idle,
                    "has_model": mem_used_mb > 2000,
                    "note": f"Use CUDA_VISIBLE_DEVICES={cuda_id}" if is_idle else "",
                })
        return gpus
    except Exception as e:
        return [{"host": "devserver", "error": str(e)}]


def write_status(gpus: list[dict]):
    """Write status to JSON file that Claude Code can read."""
    idle_gpus = [g for g in gpus if g.get("is_idle", False)]
    busy_gpus = [g for g in gpus if not g.get("is_idle", True) and "error" not in g]

    status = {
        "timestamp": datetime.now().isoformat(),
        "idle_count": len(idle_gpus),
        "busy_count": len(busy_gpus),
        "all_gpus": gpus,
        "idle_gpus": idle_gpus,
        "alert": f"⚡ {len(idle_gpus)} GPU(s) IDLE — put them to work!" if idle_gpus else "",
        "idle_summary": [],
    }

    for g in idle_gpus:
        host = g["host"]
        name = g["name"]
        if host == "devserver":
            cuda_note = f" (CUDA_VISIBLE_DEVICES={g.get('cuda_id', g['index'])})"
        else:
            cuda_note = ""
        status["idle_summary"].append(f"{host}: {name}{cuda_note}")

    STATUS_FILE.write_text(json.dumps(status, indent=2))


def main():
    print(f"GPU Monitor started. Checking every {CHECK_INTERVAL}s.")
    print(f"Status file: {STATUS_FILE}")
    print(f"Idle threshold: <{IDLE_THRESHOLD_PCT}% util, <{IDLE_VRAM_MB}MB VRAM")
    print()

    while True:
        try:
            local_gpus = check_local_gpu()
            remote_gpus = check_remote_gpu()
            all_gpus = local_gpus + remote_gpus

            write_status(all_gpus)

            # Console output
            now = datetime.now().strftime("%H:%M:%S")
            idle = [g for g in all_gpus if g.get("is_idle", False)]
            busy = [g for g in all_gpus if not g.get("is_idle", True) and "error" not in g]

            if idle:
                names = ", ".join(f"{g['host']}:{g['name']}" for g in idle)
                print(f"[{now}] ⚡ IDLE: {names}")
            else:
                names = ", ".join(
                    f"{g['host']}:{g['name']} ({g.get('util_pct', '?')}%/{g.get('vram_used_mb', '?')}MB)"
                    for g in busy
                )
                print(f"[{now}] ✓ All busy: {names}")

        except Exception as e:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Error: {e}")

        time.sleep(CHECK_INTERVAL)


if __name__ == "__main__":
    main()
