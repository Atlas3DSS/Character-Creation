#!/usr/bin/env bash
set -euo pipefail
PID="${1:?pid required}"
ROOT="/home/orwel/dev_genius/experiments/Character Creation"
LOG="$ROOT/logs/ws_openai_hourly.log"
: > "$LOG"
while kill -0 "$PID" 2>/dev/null; do
  ET=$(ps -p "$PID" -o etimes= | tr -d ' ')
  /home/orwel/dev_genius/venv/bin/python "$ROOT/scripts/experiments/personality/monitor_pass1_progress.py" \
    --generated-dir "$ROOT/sweep_v3/ws_openai_full/generated" \
    --elapsed-seconds "${ET:-1}" \
    --total-responses 14580 >> "$LOG"
  sleep 3600
done
