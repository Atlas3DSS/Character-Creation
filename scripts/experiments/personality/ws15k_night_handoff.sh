#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/orwel/dev_genius/experiments/Character Creation"
VENV="/home/orwel/dev_genius/venv"
VENV_PY="$VENV/bin/python"
GUARD_PY="$ROOT/scripts/infra/run_with_vram_guard.py"
OUT_DIR="$ROOT/sweep_v3/ws_openai_15k"
GEN_DIR="$OUT_DIR/generated"
EXPECTED_RESPONSES=14580
CHECK_EVERY_SECONDS=300
MAX_VRAM_FRACTION="${MAX_VRAM_FRACTION:-0.89}"

HANDOFF_LOG="$ROOT/logs/ws15k_night_handoff.log"
PASS2_LOG="$ROOT/logs/ws15k_pass2.log"
LOCK_FILE="$ROOT/logs/ws15k_night_handoff.lock"

mkdir -p "$ROOT/logs"

exec 9>"$LOCK_FILE"
if ! flock -n 9; then
  echo "$(date -Iseconds) handoff-lock-busy exiting" >> "$HANDOFF_LOG"
  exit 0
fi

ts() {
  date -Iseconds
}

count_responses() {
  "$VENV/bin/python" - <<'PY'
import json
from pathlib import Path

gen_dir = Path("/home/orwel/dev_genius/experiments/Character Creation/sweep_v3/ws_openai_15k/generated")
responses = 0
tokens = 0
if gen_dir.exists():
    for fp in gen_dir.glob("char_*.jsonl"):
        with fp.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                responses += 1
                tokens += int(obj.get("n_gen_tokens") or 0)
print(f"{responses} {tokens}")
PY
}

pass1_pids() {
  pgrep -f "personality_sweep_v3_pass1_openai.py --output sweep_v3/ws_openai_15k" || true
}

pass2_running() {
  pgrep -f "personality_sweep_v3_two_pass.py --output sweep_v3/ws_openai_15k" >/dev/null 2>&1
}

echo "$(ts) handoff-start expected_responses=$EXPECTED_RESPONSES" >> "$HANDOFF_LOG"

if pass2_running; then
  echo "$(ts) pass2-already-running; exiting handoff" >> "$HANDOFF_LOG"
  exit 0
fi

while true; do
  read -r responses tokens <<<"$(count_responses)"
  pids="$(pass1_pids | tr '\n' ',' | sed 's/,$//')"
  if [[ -z "$pids" ]]; then
    pids="none"
  fi

  echo "$(ts) pass1-status responses=$responses tokens=$tokens pass1_pids=$pids" >> "$HANDOFF_LOG"

  if (( responses >= EXPECTED_RESPONSES )); then
    break
  fi

  sleep "$CHECK_EVERY_SECONDS"
done

if pass2_running; then
  echo "$(ts) pass2-was-started-elsewhere; exiting handoff" >> "$HANDOFF_LOG"
  exit 0
fi

echo "$(ts) pass1-complete launching-pass2" >> "$HANDOFF_LOG"

cd "$ROOT"

setsid -f "$VENV_PY" "$GUARD_PY" \
  --gpu-index 0 \
  --max-vram-fraction "$MAX_VRAM_FRACTION" \
  --poll-seconds 5 \
  --breach-polls 1 \
  --kill-timeout-seconds 15 \
  --log-file "$PASS2_LOG" \
  --chdir "$ROOT" \
  -- \
  "$VENV_PY" scripts/experiments/personality/personality_sweep_v3_two_pass.py \
    --model Qwen/Qwen3.5-9B \
    --output sweep_v3/ws_openai_15k \
    --skip-pass1 \
    --quantize int8 \
    --replay-quantize int8 \
    --replay-batch-size 1 \
    --replay-max-total-tokens 20000

sleep 2
pass2_pid="$(pgrep -f "personality_sweep_v3_two_pass.py --output sweep_v3/ws_openai_15k" | tr '\n' ',' | sed 's/,$//')"
if [[ -z "$pass2_pid" ]]; then
  pass2_pid="unknown"
fi

echo "$(ts) pass2-launched pid=$pass2_pid log=$PASS2_LOG" >> "$HANDOFF_LOG"
