#!/bin/bash
# Retry hourly until dev server is reachable, then:
# 1) kill existing sweep runs
# 2) launch weighted sweep_v2 jobs on 4090 + 3090
# 3) exit
#
# Usage:
#   bash scripts/infra/remote_bootstrap_retry.sh

set -euo pipefail

PROJECT_DIR="/home/orwel/dev_genius/experiments/Character Creation"
REMOTE_HOST="${REMOTE_HOST:-orwel@192.168.1.90}"
LOG_FILE="${PROJECT_DIR}/logs/sweep_remote_bootstrap.log"
REMOTE_MAX_VRAM_FRACTION="${REMOTE_MAX_VRAM_FRACTION:-0.89}"
GUARD_POLL_SECONDS="${GUARD_POLL_SECONDS:-5}"

mkdir -p "${PROJECT_DIR}/logs"

while true; do
  echo "=== $(date -Is) remote-bootstrap attempt ===" >> "${LOG_FILE}" 2>&1

  if ssh -o BatchMode=yes -o ConnectTimeout=8 "${REMOTE_HOST}" "
set -euo pipefail
cd \"/home/orwel/dev_genius/experiments/Character Creation\"
if [ -f \"/home/orwel/dev_genius/qwen35_venv/bin/activate\" ]; then
  source \"/home/orwel/dev_genius/qwen35_venv/bin/activate\"
elif [ -f \"/home/orwel/dev_genius/venv/bin/activate\" ]; then
  source \"/home/orwel/dev_genius/venv/bin/activate\"
else
  echo \"NO_REMOTE_VENV_FOUND\"
  exit 1
fi
PYTHON_BIN=\"\$(command -v python3)\"
GUARD_PATH=\"/home/orwel/dev_genius/experiments/Character Creation/scripts/infra/run_with_vram_guard.py\"
mkdir -p logs sweep_v2/dev_4090 sweep_v2/dev_3090
pkill -f personality_sweep_v2.py || true
pkill -f personality_sweep_collector.py || true
sleep 2
nohup \"\$PYTHON_BIN\" \"\$GUARD_PATH\" \
  --gpu-index 0 \
  --max-vram-fraction ${REMOTE_MAX_VRAM_FRACTION} \
  --poll-seconds ${GUARD_POLL_SECONDS} \
  --breach-polls 1 \
  --kill-timeout-seconds 15 \
  --log-file logs/sweep_v2_dev_4090.log \
  --chdir \"/home/orwel/dev_genius/experiments/Character Creation\" \
  -- \
  env CUDA_VISIBLE_DEVICES=0 CUDA_DEVICE_ORDER=PCI_BUS_ID \"\$PYTHON_BIN\" -u scripts/experiments/personality/personality_sweep_v2.py \
    --shard-list 3,4,5,6 \
    --n-shards 9 \
    --output sweep_v2/dev_4090 \
    --quantize bf16 \
  > /dev/null 2>&1 < /dev/null &
PID_4090=\$!
nohup \"\$PYTHON_BIN\" \"\$GUARD_PATH\" \
  --gpu-index 1 \
  --max-vram-fraction ${REMOTE_MAX_VRAM_FRACTION} \
  --poll-seconds ${GUARD_POLL_SECONDS} \
  --breach-polls 1 \
  --kill-timeout-seconds 15 \
  --log-file logs/sweep_v2_dev_3090.log \
  --chdir \"/home/orwel/dev_genius/experiments/Character Creation\" \
  -- \
  env CUDA_VISIBLE_DEVICES=1 CUDA_DEVICE_ORDER=PCI_BUS_ID \"\$PYTHON_BIN\" -u scripts/experiments/personality/personality_sweep_v2.py \
    --shard-list 7,8 \
    --n-shards 9 \
    --output sweep_v2/dev_3090 \
    --quantize int8 \
  > /dev/null 2>&1 < /dev/null &
PID_3090=\$!
echo \"REMOTE_BOOTSTRAP_OK 4090=\${PID_4090} 3090=\${PID_3090}\"
"
  then
    echo "SUCCESS $(date -Is)" >> "${LOG_FILE}" 2>&1
    break
  fi

  echo "RETRY in 3600s" >> "${LOG_FILE}" 2>&1
  echo >> "${LOG_FILE}" 2>&1
  sleep 3600
done
