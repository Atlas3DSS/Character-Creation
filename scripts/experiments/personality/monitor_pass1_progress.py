#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Snapshot pass1 progress")
    p.add_argument("--generated-dir", required=True)
    p.add_argument("--elapsed-seconds", type=int, required=True)
    p.add_argument("--total-responses", type=int, default=14580)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    generated = Path(args.generated_dir)
    responses = 0
    tokens = 0

    if generated.exists():
        for fp in generated.glob("char_*.jsonl"):
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

    elapsed = max(1, int(args.elapsed_seconds))
    rate = tokens / elapsed
    pct = (responses / args.total_responses * 100.0) if args.total_responses else 0.0
    avg = (tokens / responses) if responses else 0.0
    projected_total = avg * args.total_responses if responses else 0.0
    remaining_tokens = max(0.0, projected_total - tokens)
    eta_h = remaining_tokens / max(rate, 1e-9) / 3600.0 if responses else 0.0

    print(
        f"ts={time.strftime('%Y-%m-%dT%H:%M:%S%z')} elapsed_s={elapsed} "
        f"responses={responses} tokens={tokens} rate_tok_s={rate:.1f} "
        f"avg_tok_resp={avg:.1f} projected_total_tokens={projected_total:.0f} "
        f"eta_h={eta_h:.2f} pct_resp={pct:.2f}"
    )


if __name__ == "__main__":
    main()
