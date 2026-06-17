#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Monitor pass1 progress and write periodic snapshots.")
    p.add_argument("--pid", type=int, required=True, help="Target pass1 process PID")
    p.add_argument("--generated-dir", type=str, required=True)
    p.add_argument("--log-path", type=str, required=True)
    p.add_argument("--total-responses", type=int, default=14580)
    p.add_argument("--interval-seconds", type=int, default=3600)
    return p.parse_args()


def read_totals(generated_dir: Path) -> tuple[int, int]:
    responses = 0
    tokens = 0
    if not generated_dir.exists():
        return responses, tokens
    for fp in generated_dir.glob("char_*.jsonl"):
        try:
            with fp.open("r", encoding="utf-8") as f:
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


def main() -> None:
    args = parse_args()
    generated_dir = Path(args.generated_dir)
    log_path = Path(args.log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    prev_ts = time.time()
    prev_resp, prev_tok = read_totals(generated_dir)

    with log_path.open("a", encoding="utf-8") as log:
        log.write(
            f"ts={time.strftime('%Y-%m-%dT%H:%M:%S%z')} event=start pid={args.pid} "
            f"responses={prev_resp} tokens={prev_tok}\n"
        )
        log.flush()

        while os.path.exists(f"/proc/{args.pid}"):
            time.sleep(max(1, int(args.interval_seconds)))

            now = time.time()
            resp, tok = read_totals(generated_dir)
            dt = max(1e-9, now - prev_ts)
            dresp = resp - prev_resp
            dtok = tok - prev_tok
            rate_tok_s = dtok / dt
            rate_resp_s = dresp / dt
            avg_tok_resp = (tok / resp) if resp else 0.0
            proj_total_tokens = avg_tok_resp * args.total_responses if resp else 0.0
            remain_tokens = max(0.0, proj_total_tokens - tok)
            eta_h = remain_tokens / max(rate_tok_s, 1e-9) / 3600.0 if dtok > 0 else 0.0
            pct_resp = (100.0 * resp / args.total_responses) if args.total_responses else 0.0

            log.write(
                f"ts={time.strftime('%Y-%m-%dT%H:%M:%S%z')} pid={args.pid} "
                f"responses={resp} tokens={tok} dresp={dresp} dtok={dtok} "
                f"rate_tok_s={rate_tok_s:.1f} rate_resp_s={rate_resp_s:.3f} "
                f"avg_tok_resp={avg_tok_resp:.1f} projected_total_tokens={proj_total_tokens:.0f} "
                f"eta_h={eta_h:.2f} pct_resp={pct_resp:.2f}\n"
            )
            log.flush()
            prev_ts, prev_resp, prev_tok = now, resp, tok

        log.write(
            f"ts={time.strftime('%Y-%m-%dT%H:%M:%S%z')} event=stop pid={args.pid}\n"
        )
        log.flush()


if __name__ == "__main__":
    main()
