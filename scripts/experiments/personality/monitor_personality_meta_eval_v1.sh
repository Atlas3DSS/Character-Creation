#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/orwel/dev_genius/experiments/Character Creation"
OUT_REL="${OUT_REL:-sweep_v4/personality_meta_eval_v1}"
OUT_DIR="$ROOT/$OUT_REL"
PYTHON_BIN="/home/orwel/dev_genius/venv/bin/python"
SUMMARIZER="$ROOT/scripts/experiments/personality/summarize_personality_meta_eval.py"
LOG_PATH="$ROOT/logs/personality_meta_eval_v1_monitor.log"
PROGRESS_JSON="$OUT_DIR/progress_summary.json"
PROGRESS_MD="$OUT_DIR/progress_summary.md"
FINAL_JSON="$OUT_DIR/final_summary.json"
FINAL_MD="$OUT_DIR/final_summary.md"
DONE_FILE="$OUT_DIR/COMPLETE"

mkdir -p "$OUT_DIR" "$ROOT/logs"

echo "[$(date --iso-8601=seconds)] monitor_start output=$OUT_REL" >> "$LOG_PATH"

while true; do
  if [[ -f "$OUT_DIR/manifest.json" ]]; then
    "$PYTHON_BIN" "$SUMMARIZER" \
      --input-dir "$OUT_DIR" \
      --output-json "$PROGRESS_JSON" \
      --output-md "$PROGRESS_MD"

    read -r completed expected pending overall_fmt overall_visible overall_trunc <<<"$(python3 - <<PY
import json
from pathlib import Path
p = Path(r"$PROGRESS_JSON")
if not p.exists():
    print('0 0 0 none none none')
else:
    data = json.loads(p.read_text())
    overall = data.get('overall') or {}
    print(
        data.get('completed_total', 0),
        data.get('expected_total', 0),
        data.get('pending_total', 0),
        overall.get('format_adherence_rate'),
        overall.get('visible_thinking_rate'),
        overall.get('truncation_rate'),
    )
PY
)"

    echo "[$(date --iso-8601=seconds)] progress completed=$completed expected=$expected pending=$pending fmt=$overall_fmt visible=$overall_visible trunc=$overall_trunc" >> "$LOG_PATH"

    if [[ "$expected" != "0" && "$completed" -ge "$expected" ]]; then
      cp "$PROGRESS_JSON" "$FINAL_JSON"
      cp "$PROGRESS_MD" "$FINAL_MD"
      echo "[$(date --iso-8601=seconds)] complete completed=$completed expected=$expected" >> "$LOG_PATH"
      date --iso-8601=seconds > "$DONE_FILE"
      if command -v notify-send >/dev/null 2>&1; then
        notify-send "personality_meta_eval_v1 complete" "Held-out A/B/C meta-format eval finished."
      fi
      if command -v wall >/dev/null 2>&1; then
        echo "personality_meta_eval_v1 complete: $OUT_REL" | wall >/dev/null 2>&1 || true
      fi
      exit 0
    fi
  else
    echo "[$(date --iso-8601=seconds)] waiting_for_manifest" >> "$LOG_PATH"
  fi
  sleep 300
done
