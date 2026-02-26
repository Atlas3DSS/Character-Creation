#!/usr/bin/env python3
"""
Overnight orchestrator: Maps both Qwen3.5-27B and 35B-A3B models
by monitoring the 27B mapping process and switching to 35B at the right time.

Strategy:
1. Monitor 27B mapping (already running) for Phase 2 completion
2. Let Phase 3 (layer scan) run for PHASE3_BUDGET minutes
3. Kill 27B process, save partial Phase 3 results
4. Run 35B mapping: Phase 1 + Phase 2 + fast layer scan
5. If time remains, resume 27B Phase 3

Usage:
    python orchestrate_overnight.py --pid 109827 [--phase3-budget 120]
"""

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

VENV = "/home/orwel/dev_genius/qwen35_venv/bin/activate"
PROJECT = "/home/orwel/dev_genius/experiments/Character Creation"
LOG_27B = None  # Set from args
LOG_35B = "/tmp/map_qwen35_35b.log"
LOG_FAST_27B = "/tmp/fast_scan_27b.log"
LOG_FAST_35B = "/tmp/fast_scan_35b.log"
ORCH_LOG = "/tmp/orchestrate_overnight.log"


def log(msg: str) -> None:
    ts = datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(ORCH_LOG, "a") as f:
        f.write(line + "\n")


def check_process(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except (ProcessLookupError, PermissionError):
        return False


def kill_process(pid: int) -> None:
    """Kill a process and all its children."""
    try:
        # Kill the entire process group
        os.killpg(os.getpgid(pid), signal.SIGTERM)
        log(f"  Sent SIGTERM to process group of PID {pid}")
    except (ProcessLookupError, PermissionError):
        pass

    # Also try direct kill
    try:
        os.kill(pid, signal.SIGTERM)
        time.sleep(5)
        if check_process(pid):
            os.kill(pid, signal.SIGKILL)
            log(f"  Sent SIGKILL to PID {pid}")
    except (ProcessLookupError, PermissionError):
        pass


def wait_for_phase2(log_path: str, check_interval: int = 60) -> bool:
    """Wait until Phase 2 complete message appears in log."""
    log(f"Waiting for Phase 2 completion in {log_path}...")

    while True:
        try:
            with open(log_path, "r") as f:
                content = f.read()
            # Check for Phase 2 completion markers
            if "Phase 2 complete" in content or "PHASE 3: SINGLE-LAYER" in content:
                log("Phase 2 COMPLETE detected!")
                return True
            if "ERROR" in content.split("\n")[-5:] if len(content.split("\n")) > 5 else "":
                log("ERROR detected in log!")
                return False
            # Also check if entire script finished
            if "COMPLETE. Total time:" in content:
                log("Script already finished all phases!")
                return True
        except FileNotFoundError:
            pass

        time.sleep(check_interval)


def wait_for_budget(minutes: int) -> None:
    """Wait for the Phase 3 budget time."""
    log(f"Phase 3 budget: {minutes} minutes. Waiting...")
    deadline = time.time() + minutes * 60

    while time.time() < deadline:
        remaining = (deadline - time.time()) / 60
        if int(remaining) % 15 == 0 and int(remaining) > 0:
            log(f"  Phase 3 budget: {int(remaining)} min remaining")
        time.sleep(60)

    log("Phase 3 budget EXHAUSTED.")


def run_script(cmd: str, log_path: str, description: str) -> subprocess.Popen:
    """Launch a script in the background."""
    log(f"Launching: {description}")
    log(f"  Command: {cmd}")
    log(f"  Log: {log_path}")

    full_cmd = f'source {VENV} && cd "{PROJECT}" && {cmd}'
    proc = subprocess.Popen(
        ["bash", "-c", full_cmd],
        stdout=open(log_path, "w"),
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    log(f"  PID: {proc.pid}")
    return proc


def wait_for_completion(proc: subprocess.Popen, log_path: str,
                        description: str, check_interval: int = 60) -> int:
    """Wait for a process to complete, logging progress."""
    log(f"Waiting for {description} to complete (PID {proc.pid})...")

    while proc.poll() is None:
        # Print last meaningful log line
        try:
            with open(log_path, "r") as f:
                content = f.read()
            lines = [l.strip() for l in content.replace("\r", "\n").split("\n")
                     if l.strip() and not l.strip().startswith("Loading weights")]
            if lines:
                last = lines[-1][:120]
                log(f"  [{description}] {last}")
        except FileNotFoundError:
            pass

        time.sleep(check_interval)

    rc = proc.returncode
    log(f"{description} finished with return code {rc}")
    return rc


def check_log_for_phase2_complete(log_path: str) -> bool:
    """Non-blocking check if Phase 2 is complete."""
    try:
        with open(log_path, "r") as f:
            content = f.read()
        return ("Phase 2 complete" in content or
                "PHASE 3: SINGLE-LAYER" in content or
                "COMPLETE. Total time:" in content)
    except FileNotFoundError:
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Overnight dual-model orchestrator")
    parser.add_argument("--pid", type=int, required=True,
                        help="PID of running 27B mapping process")
    parser.add_argument("--phase3-budget", type=int, default=120,
                        help="Minutes to let 27B Phase 3 run before switching (default 120)")
    parser.add_argument("--skip-35b", action="store_true",
                        help="Skip 35B mapping entirely")
    parser.add_argument("--log-27b", default="/tmp/map_qwen35_27b.log",
                        help="Path to 27B mapping log")
    args = parser.parse_args()

    global LOG_27B
    LOG_27B = args.log_27b

    log("=" * 60)
    log("OVERNIGHT ORCHESTRATOR STARTED")
    log(f"27B PID: {args.pid}, Phase 3 budget: {args.phase3_budget} min")
    log("=" * 60)

    # ─── Step 1: Wait for 27B Phase 2 ─────────────────────────
    if not check_process(args.pid):
        log(f"27B process (PID {args.pid}) not running! Checking if it completed...")
        if Path(f"{PROJECT}/qwen35_map/27b/connectome_zscores.pt").exists():
            log("27B connectome exists — Phase 2 already done!")
        else:
            log("ERROR: 27B process not running and no connectome found. Exiting.")
            return
    else:
        # Wait for Phase 2
        log("Monitoring 27B mapping for Phase 2 completion...")
        while True:
            if not check_process(args.pid):
                log("27B process finished before Phase 2!")
                break
            if check_log_for_phase2_complete(LOG_27B):
                log("27B Phase 2 COMPLETE!")
                break
            time.sleep(60)

    # ─── Step 2: Let Phase 3 run for budget ───────────────────
    if check_process(args.pid):
        log(f"Letting 27B Phase 3 run for {args.phase3_budget} minutes...")
        wait_for_budget(args.phase3_budget)

        # Check how many layers Phase 3 got
        scan_path = Path(f"{PROJECT}/qwen35_map/27b/layer_scan_results.json")
        if scan_path.exists():
            with open(scan_path) as f:
                scan_data = json.load(f)
            n_layers_scanned = len([k for k in scan_data if k != "baseline"])
            log(f"27B Phase 3 completed {n_layers_scanned} layers before interruption")
        else:
            log("No Phase 3 checkpoint found (may still be on baseline)")

        # Kill 27B
        log(f"Killing 27B process (PID {args.pid})...")
        # Find the actual python process (child of the bash wrapper)
        try:
            result = subprocess.run(
                ["pgrep", "-P", str(args.pid)],
                capture_output=True, text=True
            )
            child_pids = result.stdout.strip().split("\n")
            for cpid in child_pids:
                if cpid.strip():
                    log(f"  Killing child PID {cpid.strip()}")
                    try:
                        os.kill(int(cpid.strip()), signal.SIGTERM)
                    except (ProcessLookupError, ValueError):
                        pass
        except Exception:
            pass

        kill_process(args.pid)
        time.sleep(10)

        # Verify GPU is free
        log("Waiting for GPU to free up...")
        time.sleep(15)

    if args.skip_35b:
        log("Skipping 35B mapping (--skip-35b)")
    else:
        # ─── Step 3: Run 35B mapping ─────────────────────────────
        log("=" * 60)
        log("SWITCHING TO 35B MAPPING")
        log("=" * 60)

        # Run 35B Phase 1+2 (skip Phase 3 — we'll use fast scan)
        proc_35b = run_script(
            "python map_qwen35.py --model 35b --output ./qwen35_map",
            LOG_35B,
            "35B Phase 1+2+3+4"
        )

        # Monitor 35B — but we want to kill it after Phase 2 and use fast scan
        log("Monitoring 35B for Phase 2 completion...")
        while True:
            if proc_35b.poll() is not None:
                log(f"35B process finished (rc={proc_35b.returncode})")
                break
            if check_log_for_phase2_complete(LOG_35B):
                log("35B Phase 2 COMPLETE! Letting Phase 3 start...")
                # Give it 90 min for Phase 3 (40 layers is fewer than 64)
                wait_for_budget(90)
                if proc_35b.poll() is None:
                    log("Killing 35B Phase 3 to free GPU...")
                    try:
                        os.kill(proc_35b.pid, signal.SIGTERM)
                        time.sleep(10)
                        if proc_35b.poll() is None:
                            os.kill(proc_35b.pid, signal.SIGKILL)
                    except (ProcessLookupError, PermissionError):
                        pass
                break
            time.sleep(60)

        time.sleep(15)  # Let GPU clear

    # ─── Step 4: Run fast scans if connectomes exist ─────────
    connectome_27b = Path(f"{PROJECT}/qwen35_map/27b/connectome_zscores.pt")
    connectome_35b = Path(f"{PROJECT}/qwen35_map/35b/connectome_zscores.pt")

    # 27B fast scan (if Phase 3 didn't complete enough layers)
    scan_27b_path = Path(f"{PROJECT}/qwen35_map/27b/layer_scan_results.json")
    n_27b_layers = 0
    if scan_27b_path.exists():
        with open(scan_27b_path) as f:
            n_27b_layers = len([k for k in json.load(f) if k != "baseline"])

    if connectome_27b.exists() and n_27b_layers < 20:
        log("Running 27B fast layer scan (top 20 layers)...")
        proc_fast_27b = run_script(
            f"python fast_layer_scan.py --model 27b --connectome {connectome_27b} "
            f"--output ./qwen35_map --resume",
            LOG_FAST_27B,
            "27B fast scan"
        )
        wait_for_completion(proc_fast_27b, LOG_FAST_27B, "27B fast scan", check_interval=120)

    if connectome_35b.exists():
        log("Running 35B fast layer scan (top 20 layers)...")
        proc_fast_35b = run_script(
            f"python fast_layer_scan.py --model 35b --connectome {connectome_35b} "
            f"--output ./qwen35_map --resume",
            LOG_FAST_35B,
            "35B fast scan"
        )
        wait_for_completion(proc_fast_35b, LOG_FAST_35B, "35B fast scan", check_interval=120)

    # ─── Final summary ────────────────────────────────────────
    log("=" * 60)
    log("OVERNIGHT ORCHESTRATION COMPLETE")
    log("=" * 60)

    for model in ["27b", "35b"]:
        model_dir = Path(f"{PROJECT}/qwen35_map/{model}")
        if not model_dir.exists():
            log(f"  {model}: NO DATA")
            continue
        files = list(model_dir.glob("*.json")) + list(model_dir.glob("*.pt"))
        log(f"  {model}: {len(files)} output files")
        if (model_dir / "phase1_baseline.json").exists():
            with open(model_dir / "phase1_baseline.json") as f:
                p1 = json.load(f)
            for cond in ["baseline", "v4_prompt"]:
                if cond in p1:
                    d = p1[cond]
                    log(f"    {cond}: math={d.get('math_accuracy',0)*100:.0f}%, "
                        f"sarc={d.get('sarcasm_rate',0)*100:.0f}%")
        if (model_dir / "connectome_zscores.pt").exists():
            log(f"    Connectome: YES")
        scan_files = list(model_dir.glob("*scan_results.json"))
        for sf in scan_files:
            with open(sf) as f:
                sd = json.load(f)
            n = len([k for k in sd if k != "baseline"])
            log(f"    {sf.name}: {n} layers scanned")

    log(f"Total overnight time: {time.time():.0f}")
    log("All logs: /tmp/map_qwen35_*.log, /tmp/fast_scan_*.log, /tmp/orchestrate_overnight.log")


if __name__ == "__main__":
    main()
