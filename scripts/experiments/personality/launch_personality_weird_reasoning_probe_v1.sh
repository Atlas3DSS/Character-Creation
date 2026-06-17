#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/orwel/dev_genius/experiments/Character Creation"
SCRIPT="$ROOT/scripts/experiments/personality/personality_weird_reasoning_probe_openai.py"
MONITOR="$ROOT/scripts/experiments/personality/monitor_personality_weird_reasoning_probe_v1.sh"
PY="/home/orwel/dev_genius/venv/bin/python"

OUT_REL="${OUT_REL:-sweep_v4/personality_weird_reasoning_probe_v1}"
OUT_ABS="$ROOT/$OUT_REL"
LOG_DIR="$ROOT/logs"

N_CHARACTERS="${N_CHARACTERS:-24}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-720}"
SEED="${SEED:-20260405}"
CONDITION_IDS="${CONDITION_IDS:-trace_explicit,think_explicit}"
CONCURRENCY_3090="${CONCURRENCY_3090:-16}"
CONCURRENCY_4090="${CONCURRENCY_4090:-16}"

S3090="weird_probe_v1_3090"
S4090="weird_probe_v1_4090"
SMON="weird_probe_v1_monitor"

LOG3090="$LOG_DIR/weird_probe_v1_3090.log"
LOG4090="$LOG_DIR/weird_probe_v1_4090.log"
LOGMON="$LOG_DIR/weird_probe_v1_monitor.log"

mkdir -p "$OUT_ABS" "$LOG_DIR"
for sess in "$S3090" "$S4090" "$SMON"; do
  tmux has-session -t "$sess" 2>/dev/null && tmux kill-session -t "$sess"
done

CMD3090="cd \"$ROOT\" && PYTHONUNBUFFERED=1 \"$PY\" \"$SCRIPT\" \
  --output \"$OUT_ABS\" \
  --base-url \"http://192.168.1.90:30001/v1\" \
  --server-label 3090 \
  --concurrency \"$CONCURRENCY_3090\" \
  --max-new-tokens \"$MAX_NEW_TOKENS\" \
  --seed \"$SEED\" \
  --n-characters \"$N_CHARACTERS\" \
  --condition-ids \"$CONDITION_IDS\" \
  --shard 0 \
  --n-shards 2 |& tee \"$LOG3090\""

CMD4090="cd \"$ROOT\" && PYTHONUNBUFFERED=1 \"$PY\" \"$SCRIPT\" \
  --output \"$OUT_ABS\" \
  --base-url \"http://192.168.1.90:30002/v1\" \
  --server-label 4090 \
  --concurrency \"$CONCURRENCY_4090\" \
  --max-new-tokens \"$MAX_NEW_TOKENS\" \
  --seed \"$SEED\" \
  --n-characters \"$N_CHARACTERS\" \
  --condition-ids \"$CONDITION_IDS\" \
  --shard 1 \
  --n-shards 2 |& tee \"$LOG4090\""

CMDMON="cd \"$ROOT\" && \"$MONITOR\" \"$OUT_ABS\" \"$S3090\" \"$S4090\" |& tee \"$LOGMON\""

tmux new-session -d -s "$S3090" "$CMD3090"
tmux new-session -d -s "$S4090" "$CMD4090"
tmux new-session -d -s "$SMON" "$CMDMON"

echo "Launched weird probe:"
echo "  output: $OUT_ABS"
echo "  tmux: $S3090, $S4090, $SMON"
echo "  logs: $LOG3090, $LOG4090, $LOGMON"
