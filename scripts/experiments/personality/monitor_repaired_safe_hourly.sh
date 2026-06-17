#!/usr/bin/env bash
set -euo pipefail
cd "/home/orwel/dev_genius/experiments/Character Creation"
while tmux has-session -t ws15k_repaired_safe 2>/dev/null; do
  {
    echo "===== $(date -Iseconds) ====="
    nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
    python - <<'PY'
from pathlib import Path
resp_dir = Path('sweep_v3/ws_openai_15k_sampled25m_repaired_responseonly/responses')
count = 0
chars = 0
if resp_dir.exists():
    for fp in resp_dir.glob('char_*.jsonl'):
        chars += 1
        count += sum(1 for _ in fp.open())
print(f'response_files={chars} responses_written={count}')
PY
    grep "\[PASS2\]" logs/ws15k_repaired_safe_pass2_guard.log | tail -n 1 || true
    echo
  } >> logs/ws15k_repaired_safe_hourly.log 2>&1
  sleep 3600
done
