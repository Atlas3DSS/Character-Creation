#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/orwel/dev_genius/experiments/Character Creation"
SCRIPT="$ROOT/scripts/experiments/personality/personality_control_reasoning_openai.py"
MONITOR="$ROOT/scripts/experiments/personality/monitor_personality_control_reasoning_reasoningonly_v3.sh"
PY="/home/orwel/dev_genius/venv/bin/python"

OUT_REL="${OUT_REL:-sweep_v4/personality_control_reasoning_reasoningonly_v3}"
OUT_ABS="$ROOT/$OUT_REL"
LOG_DIR="$ROOT/logs"

N_SCAFFOLDS="${N_SCAFFOLDS:-128}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"
CONCURRENCY_3090="${CONCURRENCY_3090:-16}"
CONCURRENCY_4090="${CONCURRENCY_4090:-16}"
SEED="${SEED:-42}"
TRACKS="${TRACKS:-reasoning}"

S3090="control_reasoning_v3_3090"
S4090="control_reasoning_v3_4090"
SMON="control_reasoning_v3_monitor"

LOG3090="$LOG_DIR/control_reasoning_v3_3090.log"
LOG4090="$LOG_DIR/control_reasoning_v3_4090.log"
LOGMON="$LOG_DIR/control_reasoning_v3_monitor.log"

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
  --disable-thinking \
  --seed \"$SEED\" \
  --n-scaffolds \"$N_SCAFFOLDS\" \
  --tracks $TRACKS \
  --shard 0 \
  --n-shards 2 |& tee \"$LOG3090\""

CMD4090="cd \"$ROOT\" && PYTHONUNBUFFERED=1 \"$PY\" \"$SCRIPT\" \
  --output \"$OUT_ABS\" \
  --base-url \"http://192.168.1.90:30002/v1\" \
  --server-label 4090 \
  --concurrency \"$CONCURRENCY_4090\" \
  --max-new-tokens \"$MAX_NEW_TOKENS\" \
  --disable-thinking \
  --seed \"$SEED\" \
  --n-scaffolds \"$N_SCAFFOLDS\" \
  --tracks $TRACKS \
  --shard 1 \
  --n-shards 2 |& tee \"$LOG4090\""

CMDMON="cd \"$ROOT\" && \"$MONITOR\" \"$OUT_ABS\" \"$S3090\" \"$S4090\" |& tee \"$LOGMON\""

tmux new-session -d -s "$S3090" "$CMD3090"
tmux new-session -d -s "$S4090" "$CMD4090"
tmux new-session -d -s "$SMON" "$CMDMON"

echo "Launched:"
echo "  output: $OUT_ABS"
echo "  tmux: $S3090, $S4090, $SMON"
echo "  logs: $LOG3090, $LOG4090, $LOGMON"
