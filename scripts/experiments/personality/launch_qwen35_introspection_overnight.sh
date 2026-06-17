#!/usr/bin/env bash
set -euo pipefail
ROOT="${ROOT:-/home/orwel/dev_genius/experiments/Character Creation}"
OUT_REL="${OUT_REL:-sweep_v4/qwen35_introspection_overnight_20260416}"
OUT="$ROOT/$OUT_REL"
PY="${PY:-/home/orwel/dev_genius/venv/bin/python}"
LOG_DIR="$ROOT/logs"
mkdir -p "$OUT" "$LOG_DIR"
cd "$ROOT"

# Stop only this experiment's old control sessions.
for sess in qwen35_intro_main qwen35_intro_watch qwen35_intro_timer_0040 qwen35_intro_timer_0200 qwen35_intro_timer_0400 qwen35_intro_timer_0600 qwen35_intro_timer_0800; do
  tmux kill-session -t "$sess" 2>/dev/null || true
done

cat > "$OUT/reminder_status.sh" <<'EOS'
#!/usr/bin/env bash
set -euo pipefail
ROOT="/home/orwel/dev_genius/experiments/Character Creation"
OUT="$ROOT/sweep_v4/qwen35_introspection_overnight_20260416"
LOG="$OUT/reminders.log"
STAMP="$(date '+%Y-%m-%d %H:%M:%S %Z')"
{
  echo "[$STAMP] REMINDER ${1:-manual}"
  echo "sentinels:"
  for f in "$OUT"/experiment_a/DONE "$OUT"/experiment_b/DONE "$OUT"/experiment_c/DONE "$OUT"/COMPLETE "$OUT"/FAILED.json; do
    if [ -e "$f" ]; then echo "  present $f"; else echo "  missing $f"; fi
  done
  echo "gpu:"
  nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits 2>/dev/null || true
  echo "tmux:"
  tmux ls 2>/dev/null || true
  echo "tail overnight:"
  tail -n 25 "$OUT/overnight.log" 2>/dev/null || true
  echo
} >> "$LOG"
if command -v notify-send >/dev/null 2>&1; then notify-send "Qwen introspection reminder" "${1:-manual}" || true; fi
if command -v wall >/dev/null 2>&1; then echo "Qwen introspection reminder: ${1:-manual}" | wall 2>/dev/null || true; fi
EOS
chmod +x "$OUT/reminder_status.sh"

cat > "$OUT/watchdog.sh" <<'EOS'
#!/usr/bin/env bash
set -euo pipefail
ROOT="/home/orwel/dev_genius/experiments/Character Creation"
OUT="$ROOT/sweep_v4/qwen35_introspection_overnight_20260416"
LOG="$OUT/watchdog.log"
while true; do
  STAMP="$(date '+%Y-%m-%d %H:%M:%S %Z')"
  {
    echo "[$STAMP] WATCHDOG"
    if [ -f "$OUT/COMPLETE" ]; then echo "state=COMPLETE"; break; fi
    if [ -f "$OUT/FAILED.json" ]; then echo "state=FAILED"; cat "$OUT/FAILED.json"; break; fi
    for f in "$OUT"/experiment_a/DONE "$OUT"/experiment_b/DONE "$OUT"/experiment_c/DONE; do
      [ -e "$f" ] && echo "done $f" || echo "pending $f"
    done
    nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits 2>/dev/null || true
    tail -n 10 "$OUT/overnight.log" 2>/dev/null || true
    echo
  } >> "$LOG"
  sleep 900
 done
EOS
chmod +x "$OUT/watchdog.sh"

# Main autonomous runner.
tmux new-session -d -s qwen35_intro_main "cd '$ROOT' && '$PY' scripts/experiments/personality/qwen35_introspection_overnight.py --output-root '$OUT_REL' --max-vram-frac 0.85 --overwrite > '$LOG_DIR/qwen35_introspection_overnight_main.log' 2>&1"

# Periodic watchdog and separate explicit reminder timers based on the estimate.
tmux new-session -d -s qwen35_intro_watch "bash '$OUT/watchdog.sh'"
tmux new-session -d -s qwen35_intro_timer_0040 "sleep 2400; bash '$OUT/reminder_status.sh' '40m setup/pilot reminder'"
tmux new-session -d -s qwen35_intro_timer_0200 "sleep 7200; bash '$OUT/reminder_status.sh' '2h Experiment A/B reminder'"
tmux new-session -d -s qwen35_intro_timer_0400 "sleep 14400; bash '$OUT/reminder_status.sh' '4h Experiment B/C reminder'"
tmux new-session -d -s qwen35_intro_timer_0600 "sleep 21600; bash '$OUT/reminder_status.sh' '6h expected completion reminder'"
tmux new-session -d -s qwen35_intro_timer_0800 "sleep 28800; bash '$OUT/reminder_status.sh' '8h overdue failsafe reminder'"

echo "Launched qwen35 introspection overnight."
tmux ls | grep -E 'qwen35_intro' || true
