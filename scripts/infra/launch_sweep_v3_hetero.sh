#!/bin/bash
# Launch personality_sweep_v3_two_pass across workstation + dev server.
#
# Local (workstation / Blackwell): sglang backend in qwen35_venv.
# Remote (dev server 4090/3090): sglang backend in venv.
#
# Usage:
#   bash scripts/infra/launch_sweep_v3_hetero.sh
#   LOCAL_PROFILE=2w bash scripts/infra/launch_sweep_v3_hetero.sh
#   LOCAL_PROFILE=4w bash scripts/infra/launch_sweep_v3_hetero.sh
#   REMOTE_HOST=orwel@192.168.1.90 bash scripts/infra/launch_sweep_v3_hetero.sh
#
# Profiles (N_SHARDS=12):
#   2w: ws_a=0,1,2,3  ws_b=4,5,6,7
#   3w: ws_a=0,1,2  ws_b=3,4,5  ws_c=6,7
#   4w: ws_a=0,1    ws_b=2,3    ws_c=4,5    ws_d=6,7
#   5w: ws_a=0,1    ws_b=2,3    ws_c=4,5    ws_d=6      ws_e=7

set -euo pipefail

PROJECT_DIR="/home/orwel/dev_genius/experiments/Character Creation"
LOCAL_VENV_ACTIVATE="/home/orwel/dev_genius/qwen35_venv/bin/activate"
LOCAL_VENV_PY="/home/orwel/dev_genius/qwen35_venv/bin/python"
REMOTE_HOST="${REMOTE_HOST:-orwel@192.168.1.90}"
SCRIPT_PATH="scripts/experiments/personality/personality_sweep_v3_two_pass.py"
GUARD_PATH="$PROJECT_DIR/scripts/infra/run_with_vram_guard.py"
LOCAL_PROFILE="${LOCAL_PROFILE:-2w}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-4096}"
LOCAL_SGLANG_MEM_FRACTION="${LOCAL_SGLANG_MEM_FRACTION:-}"
REMOTE_SGLANG_MEM_FRACTION="${REMOTE_SGLANG_MEM_FRACTION:-0.85}"
LOCAL_MAX_VRAM_FRACTION="${LOCAL_MAX_VRAM_FRACTION:-0.89}"
REMOTE_MAX_VRAM_FRACTION="${REMOTE_MAX_VRAM_FRACTION:-0.89}"
GUARD_POLL_SECONDS="${GUARD_POLL_SECONDS:-5}"

N_SHARDS=12
WS_A_SHARDS="0,1"
WS_B_SHARDS="2,3"
WS_C_SHARDS="4,5"
WS_D_SHARDS="6"
WS_E_SHARDS="7"
DEV_4090_SHARDS="8,9,10"
DEV_3090_SHARDS="11"

case "$LOCAL_PROFILE" in
  2|2w|two)
    LOCAL_PROFILE="2w"
    if [ -z "$LOCAL_SGLANG_MEM_FRACTION" ]; then LOCAL_SGLANG_MEM_FRACTION="0.42"; fi
    WS_A_SHARDS="0,1,2,3"
    WS_B_SHARDS="4,5,6,7"
    WS_C_SHARDS=""
    WS_D_SHARDS=""
    WS_E_SHARDS=""
    ;;
  3|3w|three)
    LOCAL_PROFILE="3w"
    if [ -z "$LOCAL_SGLANG_MEM_FRACTION" ]; then LOCAL_SGLANG_MEM_FRACTION="0.28"; fi
    WS_A_SHARDS="0,1,2"
    WS_B_SHARDS="3,4,5"
    WS_C_SHARDS="6,7"
    WS_D_SHARDS=""
    WS_E_SHARDS=""
    ;;
  4|4w|four)
    LOCAL_PROFILE="4w"
    if [ -z "$LOCAL_SGLANG_MEM_FRACTION" ]; then LOCAL_SGLANG_MEM_FRACTION="0.21"; fi
    WS_A_SHARDS="0,1"
    WS_B_SHARDS="2,3"
    WS_C_SHARDS="4,5"
    WS_D_SHARDS="6,7"
    WS_E_SHARDS=""
    ;;
  5|5w|five)
    LOCAL_PROFILE="5w"
    if [ -z "$LOCAL_SGLANG_MEM_FRACTION" ]; then LOCAL_SGLANG_MEM_FRACTION="0.17"; fi
    WS_A_SHARDS="0,1"
    WS_B_SHARDS="2,3"
    WS_C_SHARDS="4,5"
    WS_D_SHARDS="6"
    WS_E_SHARDS="7"
    ;;
  *)
    echo "[ERROR] Unsupported LOCAL_PROFILE='$LOCAL_PROFILE' (use 2w, 3w, 4w, or 5w)."
    exit 1
    ;;
esac

OUT_WS_A="sweep_v3/ws_a"
OUT_WS_B="sweep_v3/ws_b"
OUT_WS_C="sweep_v3/ws_c"
OUT_WS_D="sweep_v3/ws_d"
OUT_WS_E="sweep_v3/ws_e"
OUT_4090="sweep_v3/dev_4090"
OUT_3090="sweep_v3/dev_3090"

LOG_WS_A="logs/sweep_v3_ws_a.log"
LOG_WS_B="logs/sweep_v3_ws_b.log"
LOG_WS_C="logs/sweep_v3_ws_c.log"
LOG_WS_D="logs/sweep_v3_ws_d.log"
LOG_WS_E="logs/sweep_v3_ws_e.log"
LOG_4090="logs/sweep_v3_dev_4090.log"
LOG_3090="logs/sweep_v3_dev_3090.log"

echo "[$(date)] Launching sweep_v3 two-pass run"
echo "  Project: $PROJECT_DIR"
echo "  Local profile: $LOCAL_PROFILE"
echo "  Total shard residues: $N_SHARDS"
echo "  max_new_tokens: $MAX_NEW_TOKENS"
echo "  Local sglang mem cap: $LOCAL_SGLANG_MEM_FRACTION"
echo "  Remote sglang mem cap: $REMOTE_SGLANG_MEM_FRACTION"
echo "  Local guard VRAM cap: $LOCAL_MAX_VRAM_FRACTION"
echo "  Remote guard VRAM cap: $REMOTE_MAX_VRAM_FRACTION"
echo "  Workstation A shards: $WS_A_SHARDS (sglang bf16)"
echo "  Workstation B shards: $WS_B_SHARDS (sglang bf16)"
echo "  Workstation C shards: $WS_C_SHARDS (sglang bf16)"
echo "  Workstation D shards: ${WS_D_SHARDS:-disabled} (sglang bf16)"
echo "  Workstation E shards: ${WS_E_SHARDS:-disabled} (sglang bf16)"
echo "  Dev 4090 shards:      $DEV_4090_SHARDS (sglang bf16)"
echo "  Dev 3090 shards:      $DEV_3090_SHARDS (sglang bf16)"
echo

cd "$PROJECT_DIR"
source "$LOCAL_VENV_ACTIVATE"
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

  if pgrep -f "personality_sweep_v3_two_pass.py.*--output ${out_dir}" >/dev/null 2>&1; then
    echo "[WARN] ${name} already running for ${out_dir}, skipping launch."
    return 0
  fi

  setsid "$LOCAL_VENV_PY" "$GUARD_PATH" \
    --gpu-index 0 \
    --max-vram-fraction "$LOCAL_MAX_VRAM_FRACTION" \
    --poll-seconds "$GUARD_POLL_SECONDS" \
    --breach-polls 1 \
    --kill-timeout-seconds 15 \
    --log-file "$log_file" \
    --chdir "$PROJECT_DIR" \
    -- \
    env CUDA_VISIBLE_DEVICES=0 CUDA_DEVICE_ORDER=PCI_BUS_ID "$LOCAL_VENV_PY" -u "$SCRIPT_PATH" \
      --shard-list "$shard_list" \
      --n-shards "$N_SHARDS" \
      --output "$out_dir" \
      --quantize bf16 \
      --backend sglang \
      --sglang-disable-cudnn-check \
      --sglang-mem-fraction-static "$LOCAL_SGLANG_MEM_FRACTION" \
      --max-new-tokens "$MAX_NEW_TOKENS" \
    >/dev/null 2>&1 < /dev/null &
  echo "  ${name} launched (launcher PID=$!)"
}

launch_local_worker "$OUT_WS_A" "$WS_A_SHARDS" "$LOG_WS_A" "Workstation-A"
launch_local_worker "$OUT_WS_B" "$WS_B_SHARDS" "$LOG_WS_B" "Workstation-B"
launch_local_worker "$OUT_WS_C" "$WS_C_SHARDS" "$LOG_WS_C" "Workstation-C"
launch_local_worker "$OUT_WS_D" "$WS_D_SHARDS" "$LOG_WS_D" "Workstation-D"
launch_local_worker "$OUT_WS_E" "$WS_E_SHARDS" "$LOG_WS_E" "Workstation-E"

REMOTE_CMD_4090=$(cat <<'EOS'
set -euo pipefail
cd '/home/orwel/dev_genius/experiments/Character Creation'
if [ -f '/home/orwel/dev_genius/venv/bin/activate' ]; then
  source '/home/orwel/dev_genius/venv/bin/activate'
elif [ -f '/home/orwel/dev_genius/qwen35_venv/bin/activate' ]; then
  source '/home/orwel/dev_genius/qwen35_venv/bin/activate'
else
  echo 'NO_REMOTE_VENV_FOUND'
  exit 1
fi
mkdir -p logs sweep_v3/dev_4090
PYTHON_BIN="$(command -v python3)"
GUARD_PATH='/home/orwel/dev_genius/experiments/Character Creation/scripts/infra/run_with_vram_guard.py'
if ps -eo args | grep -E '^python(3)? .*personality_sweep_v3_two_pass.py' | grep -F -- '--output sweep_v3/dev_4090' >/dev/null 2>&1; then
  echo 'SKIP_ALREADY_RUNNING_4090'
else
  nohup "$PYTHON_BIN" "$GUARD_PATH" \
    --gpu-index 0 \
    --max-vram-fraction __REMOTE_MAX_VRAM_FRACTION__ \
    --poll-seconds __GUARD_POLL_SECONDS__ \
    --breach-polls 1 \
    --kill-timeout-seconds 15 \
    --log-file logs/sweep_v3_dev_4090.log \
    --chdir '/home/orwel/dev_genius/experiments/Character Creation' \
    -- \
    env CUDA_VISIBLE_DEVICES=0 CUDA_DEVICE_ORDER=PCI_BUS_ID "$PYTHON_BIN" -u scripts/experiments/personality/personality_sweep_v3_two_pass.py \
      --shard-list 8,9,10 \
      --n-shards 12 \
      --output sweep_v3/dev_4090 \
      --quantize bf16 \
      --backend sglang \
      --sglang-disable-cudnn-check \
      --sglang-mem-fraction-static __REMOTE_MEM_FRAC__ \
      --max-new-tokens __MAX_NEW_TOKENS__ \
    >/dev/null 2>&1 < /dev/null &
  echo "LAUNCHED_4090_PID=$!"
fi
EOS
)
REMOTE_CMD_4090="${REMOTE_CMD_4090/__MAX_NEW_TOKENS__/$MAX_NEW_TOKENS}"
REMOTE_CMD_4090="${REMOTE_CMD_4090/__REMOTE_MEM_FRAC__/$REMOTE_SGLANG_MEM_FRACTION}"
REMOTE_CMD_4090="${REMOTE_CMD_4090/__REMOTE_MAX_VRAM_FRACTION__/$REMOTE_MAX_VRAM_FRACTION}"
REMOTE_CMD_4090="${REMOTE_CMD_4090/__GUARD_POLL_SECONDS__/$GUARD_POLL_SECONDS}"

REMOTE_CMD_3090=$(cat <<'EOS'
set -euo pipefail
cd '/home/orwel/dev_genius/experiments/Character Creation'
if [ -f '/home/orwel/dev_genius/venv/bin/activate' ]; then
  source '/home/orwel/dev_genius/venv/bin/activate'
elif [ -f '/home/orwel/dev_genius/qwen35_venv/bin/activate' ]; then
  source '/home/orwel/dev_genius/qwen35_venv/bin/activate'
else
  echo 'NO_REMOTE_VENV_FOUND'
  exit 1
fi
mkdir -p logs sweep_v3/dev_3090
PYTHON_BIN="$(command -v python3)"
GUARD_PATH='/home/orwel/dev_genius/experiments/Character Creation/scripts/infra/run_with_vram_guard.py'
if ps -eo args | grep -E '^python(3)? .*personality_sweep_v3_two_pass.py' | grep -F -- '--output sweep_v3/dev_3090' >/dev/null 2>&1; then
  echo 'SKIP_ALREADY_RUNNING_3090'
else
  nohup "$PYTHON_BIN" "$GUARD_PATH" \
    --gpu-index 1 \
    --max-vram-fraction __REMOTE_MAX_VRAM_FRACTION__ \
    --poll-seconds __GUARD_POLL_SECONDS__ \
    --breach-polls 1 \
    --kill-timeout-seconds 15 \
    --log-file logs/sweep_v3_dev_3090.log \
    --chdir '/home/orwel/dev_genius/experiments/Character Creation' \
    -- \
    env CUDA_VISIBLE_DEVICES=1 CUDA_DEVICE_ORDER=PCI_BUS_ID "$PYTHON_BIN" -u scripts/experiments/personality/personality_sweep_v3_two_pass.py \
      --shard-list 11 \
      --n-shards 12 \
      --output sweep_v3/dev_3090 \
      --quantize bf16 \
      --backend sglang \
      --sglang-disable-cudnn-check \
      --sglang-mem-fraction-static __REMOTE_MEM_FRAC__ \
      --max-new-tokens __MAX_NEW_TOKENS__ \
    >/dev/null 2>&1 < /dev/null &
  echo "LAUNCHED_3090_PID=$!"
fi
EOS
)
REMOTE_CMD_3090="${REMOTE_CMD_3090/__MAX_NEW_TOKENS__/$MAX_NEW_TOKENS}"
REMOTE_CMD_3090="${REMOTE_CMD_3090/__REMOTE_MEM_FRAC__/$REMOTE_SGLANG_MEM_FRACTION}"
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
echo "  watch -n 20 \"python3 scripts/infra/sweep_status.py\""
