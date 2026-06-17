#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/orwel/dev_genius/experiments/Character Creation"
PYTHON_BIN="/home/orwel/dev_genius/venv/bin/python"
OUT_REL="${OUT_REL:-sweep_v4/personality_meta_eval_trace_explicit_v1}"
OUT_DIR="$ROOT/$OUT_REL"
SCRIPT_REL="scripts/experiments/personality/personality_meta_eval_openai.py"
MONITOR_SCRIPT="$ROOT/scripts/experiments/personality/monitor_personality_meta_eval_v1.sh"
LOG_3090="$ROOT/logs/personality_meta_eval_v1_3090.log"
LOG_4090="$ROOT/logs/personality_meta_eval_v1_4090.log"
LOG_MONITOR="$ROOT/logs/personality_meta_eval_v1_monitor_runner.log"
CONDITION_IDS="${CONDITION_IDS:-trace_explicit}"

mkdir -p "$OUT_DIR" "$ROOT/logs"

for sess in meta_eval_v1_3090 meta_eval_v1_4090 meta_eval_v1_monitor; do
  tmux kill-session -t "$sess" 2>/dev/null || true
done

echo "[$(date --iso-8601=seconds)] launching personality_meta_eval_v1" | tee -a "$LOG_MONITOR"

tmux new-session -d -s meta_eval_v1_3090 \
  "cd '$ROOT' && '$PYTHON_BIN' -u '$SCRIPT_REL' \
    --output '$OUT_REL' \
    --base-url 'http://192.168.1.90:30001/v1' \
    --server-label '3090' \
    --concurrency 16 \
    --timeout 240 \
    --retries 3 \
    --max-new-tokens 960 \
    --temperature 0.4 \
    --top-p 0.9 \
    --seed 20260404 \
    --n-characters 96 \
    --condition-ids '$CONDITION_IDS' \
    --shard 0 \
    --n-shards 2 2>&1 | tee '$LOG_3090'"

tmux new-session -d -s meta_eval_v1_4090 \
  "cd '$ROOT' && '$PYTHON_BIN' -u '$SCRIPT_REL' \
    --output '$OUT_REL' \
    --base-url 'http://192.168.1.90:30002/v1' \
    --server-label '4090' \
    --concurrency 16 \
    --timeout 240 \
    --retries 3 \
    --max-new-tokens 960 \
    --temperature 0.4 \
    --top-p 0.9 \
    --seed 20260404 \
    --n-characters 96 \
    --condition-ids '$CONDITION_IDS' \
    --shard 1 \
    --n-shards 2 2>&1 | tee '$LOG_4090'"

tmux new-session -d -s meta_eval_v1_monitor \
  "cd '$ROOT' && OUT_REL='$OUT_REL' '$MONITOR_SCRIPT' 2>&1 | tee '$LOG_MONITOR'"

echo "launched: $OUT_REL"
echo "3090 log: $LOG_3090"
echo "4090 log: $LOG_4090"
echo "monitor log: $LOG_MONITOR"
