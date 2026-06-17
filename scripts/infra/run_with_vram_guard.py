#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import shlex
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a command with a hard VRAM ceiling.")
    parser.add_argument("--gpu-index", type=int, default=0, help="nvidia-smi GPU index to monitor")
    parser.add_argument(
        "--max-vram-fraction",
        type=float,
        default=0.89,
        help="Hard ceiling as a fraction of total VRAM",
    )
    parser.add_argument(
        "--max-vram-mib",
        type=int,
        default=None,
        help="Hard ceiling in MiB. Overrides --max-vram-fraction if set.",
    )
    parser.add_argument("--poll-seconds", type=float, default=5.0)
    parser.add_argument(
        "--breach-polls",
        type=int,
        default=1,
        help="Kill after this many consecutive over-limit polls",
    )
    parser.add_argument("--kill-timeout-seconds", type=float, default=15.0)
    parser.add_argument("--startup-grace-seconds", type=float, default=0.0)
    parser.add_argument("--chdir", type=str, default=None)
    parser.add_argument("--log-file", type=str, required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.command and args.command[0] == "--":
        args.command = args.command[1:]
    if not args.command:
        parser.error("missing command after '--'")
    if args.max_vram_mib is None and not (0 < args.max_vram_fraction < 1):
        parser.error("--max-vram-fraction must be between 0 and 1")
    if args.breach_polls < 1:
        parser.error("--breach-polls must be >= 1")
    return args


def query_gpu(gpu_index: int) -> tuple[int, int, int]:
    out = subprocess.check_output(
        [
            "nvidia-smi",
            f"--id={gpu_index}",
            "--query-gpu=memory.used,memory.total,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    ).strip()
    used_s, total_s, util_s = [part.strip() for part in out.split(",")]
    return int(used_s), int(total_s), int(util_s)


def wait_for_exit(proc: subprocess.Popen[bytes], timeout_s: float) -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if proc.poll() is not None:
            return True
        time.sleep(0.25)
    return proc.poll() is not None


def main() -> int:
    args = parse_args()
    log_path = Path(args.log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    def log(message: str) -> None:
        line = f"{datetime.now().isoformat()} {message}\n"
        with log_path.open("a", encoding="utf-8") as fh:
            fh.write(line)
            fh.flush()

    try:
        used_mib, total_mib, util_pct = query_gpu(args.gpu_index)
    except Exception as exc:  # noqa: BLE001
        log(f"[GUARD][ERROR] failed to query nvidia-smi before launch: {exc}")
        return 2

    limit_mib = args.max_vram_mib or int(total_mib * args.max_vram_fraction)
    cmd_str = shlex.join(args.command)
    log(
        f"[GUARD] launch gpu={args.gpu_index} used={used_mib}MiB total={total_mib}MiB "
        f"util={util_pct}% limit={limit_mib}MiB cmd={cmd_str}"
    )
    if used_mib > limit_mib:
        log(f"[GUARD][ABORT] preflight VRAM already above limit: used={used_mib}MiB limit={limit_mib}MiB")
        return 3

    env = os.environ.copy()
    proc = None
    with log_path.open("a", encoding="utf-8") as child_log:
        try:
            proc = subprocess.Popen(
                args.command,
                cwd=args.chdir or None,
                env=env,
                stdout=child_log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        except Exception as exc:  # noqa: BLE001
            log(f"[GUARD][ERROR] failed to start child: {exc}")
            return 4

        log(f"[GUARD] child-started pid={proc.pid}")
        peak_mib = used_mib
        breaches = 0
        start = time.time()
        terminate_requested: int | None = None

        def handle_signal(signum: int, _frame: object) -> None:
            nonlocal terminate_requested
            terminate_requested = signum
            log(f"[GUARD][SIGNAL] received signal={signum} child_pid={proc.pid}")

        signal.signal(signal.SIGTERM, handle_signal)
        signal.signal(signal.SIGINT, handle_signal)

        while True:
            rc = proc.poll()
            try:
                used_mib, total_mib, util_pct = query_gpu(args.gpu_index)
            except Exception as exc:  # noqa: BLE001
                log(f"[GUARD][WARN] failed to query nvidia-smi: {exc}")
                used_mib = peak_mib
                util_pct = -1

            peak_mib = max(peak_mib, used_mib)
            log(
                f"[GUARD] poll pid={proc.pid} used={used_mib}MiB total={total_mib}MiB "
                f"util={util_pct}% peak={peak_mib}MiB"
            )

            if terminate_requested is not None:
                try:
                    os.killpg(proc.pid, signal.SIGTERM)
                    log(f"[GUARD][KILL] forwarded SIGTERM to pgid={proc.pid}")
                except ProcessLookupError:
                    pass
                if not wait_for_exit(proc, args.kill_timeout_seconds):
                    try:
                        os.killpg(proc.pid, signal.SIGKILL)
                        log(f"[GUARD][KILL] sent SIGKILL to pgid={proc.pid}")
                    except ProcessLookupError:
                        pass
                    wait_for_exit(proc, 3.0)
                final_rc = proc.poll()
                log(
                    f"[GUARD][STOP] signal={terminate_requested} pid={proc.pid} "
                    f"final_rc={final_rc} peak={peak_mib}MiB"
                )
                return 128 + terminate_requested

            if rc is not None:
                log(f"[GUARD] child-exit pid={proc.pid} rc={rc} peak={peak_mib}MiB")
                return rc

            if used_mib > limit_mib and (time.time() - start) >= args.startup_grace_seconds:
                breaches += 1
                log(
                    f"[GUARD][BREACH] pid={proc.pid} used={used_mib}MiB limit={limit_mib}MiB "
                    f"count={breaches}/{args.breach_polls}"
                )
                if breaches >= args.breach_polls:
                    try:
                        os.killpg(proc.pid, signal.SIGTERM)
                        log(f"[GUARD][KILL] sent SIGTERM to pgid={proc.pid}")
                    except ProcessLookupError:
                        pass
                    if not wait_for_exit(proc, args.kill_timeout_seconds):
                        try:
                            os.killpg(proc.pid, signal.SIGKILL)
                            log(f"[GUARD][KILL] sent SIGKILL to pgid={proc.pid}")
                        except ProcessLookupError:
                            pass
                        wait_for_exit(proc, 3.0)
                    final_rc = proc.poll()
                    log(
                        f"[GUARD][ABORT] pid={proc.pid} final_rc={final_rc} peak={peak_mib}MiB "
                        f"limit={limit_mib}MiB"
                    )
                    return 97
            else:
                breaches = 0

            time.sleep(args.poll_seconds)


if __name__ == "__main__":
    sys.exit(main())
