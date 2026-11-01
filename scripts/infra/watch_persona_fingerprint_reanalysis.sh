#!/usr/bin/env bash
set -u

ROOT="/home/orwel/dev_genius/experiments/Character Creation"
RUN_DIR="${1:-$ROOT/sweep_v4/jlens_persona_fingerprint_real_20260709_001511}"
LOG_DIR="$ROOT/logs"
VENV_PATH="${VENV_PATH:-/home/orwel/dev_genius/venv}"
MODEL_PATH="${MODEL_PATH:-/home/orwel/dev_genius/models/Qwen3.5-27B}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-3072}"
K_VALUES="${K_VALUES:-8,32,128,512}"
SLEEP_SECONDS="${SLEEP_SECONDS:-300}"
EXPECTED_RECORDS="${EXPECTED_RECORDS:-40}"

mkdir -p "$LOG_DIR"
cd "$ROOT" || exit 1
source "$VENV_PATH/bin/activate"

WATCH_LOG="$LOG_DIR/jlens_27b_fingerprint_reanalysis_watch.log"
REANALYZE_LOG="$LOG_DIR/jlens_27b_fingerprint_reanalysis.log"

printf '[%s] watching run_dir=%s\n' "$(date --iso-8601=seconds)" "$RUN_DIR" >> "$WATCH_LOG"

while pgrep -f "jlens_persona_fingerprint.py .*${RUN_DIR}" >/dev/null 2>&1; do
  printf '[%s] capture/old-analysis still running\n' "$(date --iso-8601=seconds)" >> "$WATCH_LOG"
  sleep "$SLEEP_SECONDS"
done

if [[ ! -s "$RUN_DIR/records.jsonl" ]]; then
  printf '[%s] records missing; cannot reanalyze\n' "$(date --iso-8601=seconds)" >> "$WATCH_LOG"
  exit 2
fi

record_count="$(wc -l < "$RUN_DIR/records.jsonl")"
if [[ "$record_count" -lt "$EXPECTED_RECORDS" ]]; then
  printf '[%s] only %s/%s records present after capture process exited; refusing partial reanalysis\n' "$(date --iso-8601=seconds)" "$record_count" "$EXPECTED_RECORDS" >> "$WATCH_LOG"
  exit 3
fi

printf '[%s] starting patched reanalysis\n' "$(date --iso-8601=seconds)" >> "$WATCH_LOG"
python scripts/experiments/personality/jlens_persona_fingerprint.py \
  --allow-real-model-run \
  --reuse-capture-dir "$RUN_DIR" \
  --output-dir "$RUN_DIR" \
  --model-path "$MODEL_PATH" \
  --k-values "$K_VALUES" \
  --max-new-tokens "$MAX_NEW_TOKENS" \
  > "$REANALYZE_LOG" 2>&1
code=$?
printf '[%s] reanalysis exit=%s log=%s\n' "$(date --iso-8601=seconds)" "$code" "$REANALYZE_LOG" >> "$WATCH_LOG"
exit "$code"
