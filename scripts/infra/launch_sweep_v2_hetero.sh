#!/bin/bash
# Launch personality_sweep_v2 with weighted shards across heterogeneous GPUs.
#
# Targets:
#   - Workstation PRO 6000 (BF16): shards 0,1,2
#   - Dev server 4090 (BF16):      shards 3,4,5,6
#   - Dev server 3090 (INT8):      shards 7,8
#
# Usage:
#   bash scripts/infra/launch_sweep_v2_hetero.sh
#   REMOTE_HOST=orwel@192.168.86.66 bash scripts/infra/launch_sweep_v2_hetero.sh

set -euo pipefail

PROJECT_DIR="/home/orwel/dev_genius/experiments/Character Creation"
VENV_ACTIVATE="/home/orwel/dev_genius/qwen35_venv/bin/activate"
REMOTE_HOST="${REMOTE_HOST:-orwel@192.168.86.66}"
SCRIPT_PATH="scripts/experiments/personality/personality_sweep_v2.py"

N_SHARDS=9
WORKSTATION_SHARDS="0,1,2"
DEV_4090_SHARDS="3,4,5,6"
DEV_3090_SHARDS="7,8"

OUT_WS="sweep_v2/workstation_pro6000"
OUT_4090="sweep_v2/dev_4090"
OUT_3090="sweep_v2/dev_3090"

LOG_WS="logs/sweep_v2_workstation.log"
LOG_4090="logs/sweep_v2_dev_4090.log"
LOG_3090="logs/sweep_v2_dev_3090.log"

echo "[$(date)] Launching weighted sweep_v2 run"
echo "  Project: $PROJECT_DIR"
echo "  Total shard residues: $N_SHARDS"
echo "  Workstation shards: $WORKSTATION_SHARDS (BF16)"
echo "  Dev 4090 shards:    $DEV_4090_SHARDS (BF16)"
echo "  Dev 3090 shards:    $DEV_3090_SHARDS (INT8)"
echo

cd "$PROJECT_DIR"
source "$VENV_ACTIVATE"
mkdir -p logs "$OUT_WS" "$OUT_4090" "$OUT_3090"

if pgrep -f "personality_sweep_v2.py.*--output $OUT_WS" >/dev/null 2>&1; then
  echo "[WARN] Workstation sweep already running for $OUT_WS, skipping launch."
else
  nohup env CUDA_VISIBLE_DEVICES=0 python3 -u "$SCRIPT_PATH" \
    --shard-list "$WORKSTATION_SHARDS" \
    --n-shards "$N_SHARDS" \
    --output "$OUT_WS" \
    --quantize bf16 \
    > "$LOG_WS" 2>&1 < /dev/null &
  echo "  Workstation launched (PID=$!)"
fi

REMOTE_CMD_4090=$(
  cat <<'EOF'
set -euo pipefail
cd '/home/orwel/dev_genius/experiments/Character Creation'
source '/home/orwel/dev_genius/qwen35_venv/bin/activate'
mkdir -p logs sweep_v2/dev_4090
if pgrep -f "personality_sweep_v2.py.*--output sweep_v2/dev_4090" >/dev/null 2>&1; then
  echo "SKIP_ALREADY_RUNNING_4090"
else
  nohup env CUDA_VISIBLE_DEVICES=0 python3 -u scripts/experiments/personality/personality_sweep_v2.py \
    --shard-list 3,4,5,6 \
    --n-shards 9 \
    --output sweep_v2/dev_4090 \
    --quantize bf16 \
    > logs/sweep_v2_dev_4090.log 2>&1 < /dev/null &
  echo "LAUNCHED_4090_PID=$!"
fi
EOF
)

REMOTE_CMD_3090=$(
  cat <<'EOF'
set -euo pipefail
cd '/home/orwel/dev_genius/experiments/Character Creation'
source '/home/orwel/dev_genius/qwen35_venv/bin/activate'
mkdir -p logs sweep_v2/dev_3090
if pgrep -f "personality_sweep_v2.py.*--output sweep_v2/dev_3090" >/dev/null 2>&1; then
  echo "SKIP_ALREADY_RUNNING_3090"
else
  nohup env CUDA_VISIBLE_DEVICES=1 python3 -u scripts/experiments/personality/personality_sweep_v2.py \
    --shard-list 7,8 \
    --n-shards 9 \
    --output sweep_v2/dev_3090 \
    --quantize int8 \
    > logs/sweep_v2_dev_3090.log 2>&1 < /dev/null &
  echo "LAUNCHED_3090_PID=$!"
fi
EOF
)

echo
echo "[$(date)] Launching dev server jobs on $REMOTE_HOST ..."
if ssh -o ConnectTimeout=8 "$REMOTE_HOST" "$REMOTE_CMD_4090"; then
  :
else
  echo "[WARN] 4090 launch command failed (SSH/connectivity issue)."
fi
if ssh -o ConnectTimeout=8 "$REMOTE_HOST" "$REMOTE_CMD_3090"; then
  :
else
  echo "[WARN] 3090 launch command failed (SSH/connectivity issue)."
fi

echo
echo "[$(date)] Launch sequence complete."
echo "Monitor with:"
echo "  python3 scripts/infra/sweep_status.py"
echo "  watch -n 15 \"python3 scripts/infra/sweep_status.py\""
