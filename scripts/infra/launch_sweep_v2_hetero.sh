#!/bin/bash
# Launch personality_sweep_v2 with a Blackwell-heavy weighted profile.
#
# Targets (N_SHARDS=12):
#   LOCAL_PROFILE=5w:
#     - Workstation A: shards 0,1
#     - Workstation B: shards 2,3
#     - Workstation C: shards 4,5
#     - Workstation D: shard 6
#     - Workstation E: shard 7
#   LOCAL_PROFILE=3w (recommended for throughput check):
#     - Workstation A: shards 0,1,2
#     - Workstation B: shards 3,4,5
#     - Workstation C: shards 6,7
#   LOCAL_PROFILE=4w:
#     - Workstation A: shards 0,1
#     - Workstation B: shards 2,3
#     - Workstation C: shards 4,5
#     - Workstation D: shards 6,7
#   Dev server 4090 (BF16):               shards 8,9,10
#   Dev server 3090 (BF16):               shard 11
#
# Usage:
#   bash scripts/infra/launch_sweep_v2_hetero.sh
#   REMOTE_HOST=orwel@192.168.1.90 bash scripts/infra/launch_sweep_v2_hetero.sh
#   LOCAL_PROFILE=4w bash scripts/infra/launch_sweep_v2_hetero.sh
#   LOCAL_PROFILE=3w bash scripts/infra/launch_sweep_v2_hetero.sh

set -euo pipefail

PROJECT_DIR="/home/orwel/dev_genius/experiments/Character Creation"
VENV_ACTIVATE="/home/orwel/dev_genius/qwen35_venv/bin/activate"
VENV_PY="/home/orwel/dev_genius/qwen35_venv/bin/python"
REMOTE_HOST="${REMOTE_HOST:-orwel@192.168.1.90}"
SCRIPT_PATH="scripts/experiments/personality/personality_sweep_v2.py"
LOCAL_PROFILE="${LOCAL_PROFILE:-5w}"
GUARD_PATH="$PROJECT_DIR/scripts/infra/run_with_vram_guard.py"
LOCAL_MAX_VRAM_FRACTION="${LOCAL_MAX_VRAM_FRACTION:-0.89}"
REMOTE_MAX_VRAM_FRACTION="${REMOTE_MAX_VRAM_FRACTION:-0.89}"
GUARD_POLL_SECONDS="${GUARD_POLL_SECONDS:-5}"

N_SHARDS=12
WS_A_SHARDS="0,1"
WS_B_SHARDS="2,3"
WS_C_SHARDS="4,5"
DEV_4090_SHARDS="8,9,10"
DEV_3090_SHARDS="11"

case "$LOCAL_PROFILE" in
  3|3w|three)
    LOCAL_PROFILE="3w"
    WS_A_SHARDS="0,1,2"
    WS_B_SHARDS="3,4,5"
    WS_C_SHARDS="6,7"
    WS_D_SHARDS=""
    WS_E_SHARDS=""
    ;;
  4|4w|four)
    LOCAL_PROFILE="4w"
    WS_D_SHARDS="6,7"
    WS_E_SHARDS=""
    ;;
  5|5w|five)
    LOCAL_PROFILE="5w"
    WS_D_SHARDS="6"
    WS_E_SHARDS="7"
    ;;
  *)
    echo "[ERROR] Unsupported LOCAL_PROFILE='$LOCAL_PROFILE' (use 3w, 4w, or 5w)."
    exit 1
    ;;
esac

OUT_WS_A="sweep_v2/ws4_a"
OUT_WS_B="sweep_v2/ws4_b"
OUT_WS_C="sweep_v2/ws4_c"
OUT_WS_D="sweep_v2/ws4_d"
OUT_WS_E="sweep_v2/ws4_e"
OUT_4090="sweep_v2/dev_4090_hiutil"
OUT_3090="sweep_v2/dev_3090_hiutil"

LOG_WS_A="logs/sweep_v2_ws4_a.log"
LOG_WS_B="logs/sweep_v2_ws4_b.log"
LOG_WS_C="logs/sweep_v2_ws4_c.log"
LOG_WS_D="logs/sweep_v2_ws4_d.log"
LOG_WS_E="logs/sweep_v2_ws4_e.log"
LOG_4090="logs/sweep_v2_dev_4090_hiutil.log"
LOG_3090="logs/sweep_v2_dev_3090_hiutil.log"

echo "[$(date)] Launching weighted sweep_v2 run"
echo "  Project: $PROJECT_DIR"
echo "  Local profile: $LOCAL_PROFILE"
echo "  Total shard residues: $N_SHARDS"
echo "  Local guard VRAM cap: $LOCAL_MAX_VRAM_FRACTION"
echo "  Remote guard VRAM cap: $REMOTE_MAX_VRAM_FRACTION"
echo "  Workstation A shards: $WS_A_SHARDS (BF16)"
echo "  Workstation B shards: $WS_B_SHARDS (BF16)"
echo "  Workstation C shards: $WS_C_SHARDS (BF16)"
echo "  Workstation D shards: $WS_D_SHARDS (BF16)"
if [ -n "$WS_E_SHARDS" ]; then
  echo "  Workstation E shards: $WS_E_SHARDS (BF16)"
fi
echo "  Dev 4090 shards:    $DEV_4090_SHARDS (BF16)"
echo "  Dev 3090 shards:    $DEV_3090_SHARDS (BF16)"
echo

cd "$PROJECT_DIR"
source "$VENV_ACTIVATE"
mkdir -p logs "$OUT_WS_A" "$OUT_WS_B" "$OUT_WS_C" "$OUT_WS_D" "$OUT_WS_E" "$OUT_4090" "$OUT_3090"

launch_local_worker() {
  local out_dir="$1"
  local shard_list="$2"
  local log_file="$3"
  local name="$4"
  if [ -z "$shard_list" ]; then
    echo "  ${name} disabled in profile ${LOCAL_PROFILE}."
    return 0
  fi
  if pgrep -f "personality_sweep_v2.py.*--output ${out_dir}" >/dev/null 2>&1; then
    echo "[WARN] ${name} already running for ${out_dir}, skipping launch."
    return 0
  fi
  setsid "$VENV_PY" "$GUARD_PATH" \
    --gpu-index 0 \
    --max-vram-fraction "$LOCAL_MAX_VRAM_FRACTION" \
    --poll-seconds "$GUARD_POLL_SECONDS" \
    --breach-polls 1 \
    --kill-timeout-seconds 15 \
    --log-file "$log_file" \
    --chdir "$PROJECT_DIR" \
    -- \
    env CUDA_VISIBLE_DEVICES=0 CUDA_DEVICE_ORDER=PCI_BUS_ID "$VENV_PY" -u "$SCRIPT_PATH" \
      --shard-list "$shard_list" \
      --n-shards "$N_SHARDS" \
      --output "$out_dir" \
      --quantize bf16 \
    >/dev/null 2>&1 < /dev/null &
  echo "  ${name} launched (launcher PID=$!)"
}

launch_local_worker "$OUT_WS_A" "$WS_A_SHARDS" "$LOG_WS_A" "Workstation-A"
launch_local_worker "$OUT_WS_B" "$WS_B_SHARDS" "$LOG_WS_B" "Workstation-B"
launch_local_worker "$OUT_WS_C" "$WS_C_SHARDS" "$LOG_WS_C" "Workstation-C"
launch_local_worker "$OUT_WS_D" "$WS_D_SHARDS" "$LOG_WS_D" "Workstation-D"
launch_local_worker "$OUT_WS_E" "$WS_E_SHARDS" "$LOG_WS_E" "Workstation-E"

REMOTE_CMD_4090=$(
  cat <<'EOF'
set -euo pipefail
cd '/home/orwel/dev_genius/experiments/Character Creation'
if [ -f '/home/orwel/dev_genius/qwen35_venv/bin/activate' ]; then
  source '/home/orwel/dev_genius/qwen35_venv/bin/activate'
elif [ -f '/home/orwel/dev_genius/venv/bin/activate' ]; then
  source '/home/orwel/dev_genius/venv/bin/activate'
else
  echo "NO_REMOTE_VENV_FOUND"
  exit 1
fi
mkdir -p logs sweep_v2/dev_4090_hiutil
PYTHON_BIN="$(command -v python3)"
GUARD_PATH='/home/orwel/dev_genius/experiments/Character Creation/scripts/infra/run_with_vram_guard.py'
if ps -eo args | grep -E "^python(3)? .*personality_sweep_v2.py" | grep -F -- "--output sweep_v2/dev_4090_hiutil" >/dev/null 2>&1; then
  echo "SKIP_ALREADY_RUNNING_4090"
else
  nohup "$PYTHON_BIN" "$GUARD_PATH" \
    --gpu-index 0 \
    --max-vram-fraction __REMOTE_MAX_VRAM_FRACTION__ \
    --poll-seconds __GUARD_POLL_SECONDS__ \
    --breach-polls 1 \
    --kill-timeout-seconds 15 \
    --log-file logs/sweep_v2_dev_4090_hiutil.log \
    --chdir '/home/orwel/dev_genius/experiments/Character Creation' \
    -- \
    env CUDA_VISIBLE_DEVICES=0 CUDA_DEVICE_ORDER=PCI_BUS_ID "$PYTHON_BIN" -u scripts/experiments/personality/personality_sweep_v2.py \
      --shard-list 8,9,10 \
      --n-shards 12 \
      --output sweep_v2/dev_4090_hiutil \
      --quantize bf16 \
    >/dev/null 2>&1 < /dev/null &
  echo "LAUNCHED_4090_PID=$!"
fi
EOF
)
REMOTE_CMD_4090="${REMOTE_CMD_4090/__REMOTE_MAX_VRAM_FRACTION__/$REMOTE_MAX_VRAM_FRACTION}"
REMOTE_CMD_4090="${REMOTE_CMD_4090/__GUARD_POLL_SECONDS__/$GUARD_POLL_SECONDS}"

REMOTE_CMD_3090=$(
  cat <<'EOF'
set -euo pipefail
cd '/home/orwel/dev_genius/experiments/Character Creation'
if [ -f '/home/orwel/dev_genius/qwen35_venv/bin/activate' ]; then
  source '/home/orwel/dev_genius/qwen35_venv/bin/activate'
elif [ -f '/home/orwel/dev_genius/venv/bin/activate' ]; then
  source '/home/orwel/dev_genius/venv/bin/activate'
else
  echo "NO_REMOTE_VENV_FOUND"
  exit 1
fi
mkdir -p logs sweep_v2/dev_3090_hiutil
PYTHON_BIN="$(command -v python3)"
GUARD_PATH='/home/orwel/dev_genius/experiments/Character Creation/scripts/infra/run_with_vram_guard.py'
if ps -eo args | grep -E "^python(3)? .*personality_sweep_v2.py" | grep -F -- "--output sweep_v2/dev_3090_hiutil" >/dev/null 2>&1; then
  echo "SKIP_ALREADY_RUNNING_3090"
else
  nohup "$PYTHON_BIN" "$GUARD_PATH" \
    --gpu-index 1 \
    --max-vram-fraction __REMOTE_MAX_VRAM_FRACTION__ \
    --poll-seconds __GUARD_POLL_SECONDS__ \
    --breach-polls 1 \
    --kill-timeout-seconds 15 \
    --log-file logs/sweep_v2_dev_3090_hiutil.log \
    --chdir '/home/orwel/dev_genius/experiments/Character Creation' \
    -- \
    env CUDA_VISIBLE_DEVICES=1 CUDA_DEVICE_ORDER=PCI_BUS_ID "$PYTHON_BIN" -u scripts/experiments/personality/personality_sweep_v2.py \
      --shard-list 11 \
      --n-shards 12 \
      --output sweep_v2/dev_3090_hiutil \
      --quantize bf16 \
    >/dev/null 2>&1 < /dev/null &
  echo "LAUNCHED_3090_PID=$!"
fi
EOF
)
REMOTE_CMD_3090="${REMOTE_CMD_3090/__REMOTE_MAX_VRAM_FRACTION__/$REMOTE_MAX_VRAM_FRACTION}"
REMOTE_CMD_3090="${REMOTE_CMD_3090/__GUARD_POLL_SECONDS__/$GUARD_POLL_SECONDS}"

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
