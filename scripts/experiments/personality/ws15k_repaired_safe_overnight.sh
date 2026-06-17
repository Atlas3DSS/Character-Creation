#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/orwel/dev_genius/experiments/Character Creation"
REPLAY_PY="/home/orwel/dev_genius/qwen35_replay_venv/bin/python"
ANALYSIS_PY="/home/orwel/dev_genius/qwen35_replay_venv/bin/python"
VIZ_PY="/home/orwel/dev_genius/venv/bin/python"
GUARD_PY="$ROOT/scripts/infra/run_with_vram_guard.py"

OUT_REL="sweep_v3/ws_openai_15k_sampled25m_repaired_responseonly"
OUT_DIR="$ROOT/$OUT_REL"
ANALYSIS_DIR="$ROOT/reports/ws15k_repaired_responseonly_phase_analysis"
VIZ_HTML="$ROOT/reports/ws15k_repaired_responseonly_phase_visualizer.html"

RUN_LOG="$ROOT/logs/ws15k_repaired_safe_overnight.log"
PASS2_LOG="$ROOT/logs/ws15k_repaired_safe_pass2_guard.log"
ANALYSIS_LOG="$ROOT/logs/ws15k_repaired_safe_analysis.log"
REPORT_MD="$ROOT/reports/ws15k_repaired_safe_handoff.md"

MAX_VRAM_FRACTION="${MAX_VRAM_FRACTION:-0.89}"
POLL_SECONDS="${POLL_SECONDS:-5}"
REPLAY_BATCH_SIZE="${REPLAY_BATCH_SIZE:-1}"
REPLAY_MAX_TOTAL_TOKENS="${REPLAY_MAX_TOTAL_TOKENS:-16384}"

log() {
  printf '%s %s\n' "$(date -Iseconds)" "$*" | tee -a "$RUN_LOG"
}

count_generated_rows() {
  "$VIZ_PY" - <<'PY'
import json
from pathlib import Path
root = Path("/home/orwel/dev_genius/experiments/Character Creation/sweep_v3/ws_openai_15k_sampled25m_repaired_responseonly/generated")
rows = 0
for fp in root.glob("char_*.jsonl"):
    with fp.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.strip():
                rows += 1
print(rows)
PY
}

cleanup_local_work() {
  pkill -f "analyze_personality_phase_sweep.py --sweep-dir sweep_v3/ws_openai_15k_sampled25m" || true
  pkill -f "analyze_personality_phase_sweep.py --sweep-dir sweep_v3/ws_openai_15k_sampled25m_repaired_responseonly" || true
  pkill -f "build_personality_visualizer.py --sweep-dir sweep_v3/ws_openai_15k_sampled25m" || true
  pkill -f "build_personality_visualizer.py --sweep-dir sweep_v3/ws_openai_15k_sampled25m_repaired_responseonly" || true
  pkill -f "personality_sweep_v3_two_pass.py --output sweep_v3/ws_openai_15k_sampled25m_repaired_responseonly" || true
  tmux kill-session -t analyze_ws15k 2>/dev/null || true
  tmux kill-session -t visualize_ws15k 2>/dev/null || true
  tmux kill-session -t repair_autohandoff 2>/dev/null || true
  tmux kill-session -t pass2_ws15k 2>/dev/null || true
}

clear_partial_pass2() {
  rm -rf \
    "$OUT_DIR/responses" \
    "$OUT_DIR/activations" \
    "$OUT_DIR/activations_think" \
    "$OUT_DIR/activations_response" \
    "$OUT_DIR/activations_early" \
    "$OUT_DIR/activations_late" \
    "$ANALYSIS_DIR"
  rm -f "$OUT_DIR/summary_stats.json"
}

write_handoff_report() {
  "$VIZ_PY" - <<'PY'
import json
from pathlib import Path
from datetime import datetime

root = Path("/home/orwel/dev_genius/experiments/Character Creation")
summary_path = root / "sweep_v3/ws_openai_15k_sampled25m_repaired_responseonly/summary_stats.json"
analysis_path = root / "reports/ws15k_repaired_responseonly_phase_analysis/analysis_results.json"
report_path = root / "reports/ws15k_repaired_safe_handoff.md"

lines = [f"# Repaired Pass2 Handoff ({datetime.now().isoformat(timespec='seconds')})", ""]
if summary_path.exists():
    summary = json.loads(summary_path.read_text())
    p2 = summary.get("pass2", {})
    lines += [
        "## Pass2",
        f"- responses: {p2.get('responses')}",
        f"- generated tokens: {p2.get('gen_tokens')}",
        f"- sequence tokens: {p2.get('seq_tokens')}",
        f"- elapsed seconds: {p2.get('elapsed_seconds')}",
        f"- gen tok/s: {p2.get('gen_tokens_per_second')}",
        f"- seq tok/s: {p2.get('seq_tokens_per_second')}",
        "",
    ]
else:
    lines += ["## Pass2", "- summary_stats.json missing", ""]

if analysis_path.exists():
    analysis = json.loads(analysis_path.read_text())
    lines += [
        "## Analysis",
        f"- best separability keys: {len(analysis.get('top_metrics', []))}",
        f"- factorial slices: {len(analysis.get('triplets_by_trait', {}))}",
        "",
    ]
else:
    lines += ["## Analysis", "- analysis_results.json missing", ""]

report_path.write_text("\n".join(lines), encoding="utf-8")
PY
}

cd "$ROOT"
mkdir -p "$ROOT/logs" "$ROOT/reports"

log "safe-overnight-start"
cleanup_local_work
sleep 2

generated_rows="$(count_generated_rows)"
log "generated-rows=$generated_rows"
if [[ "$generated_rows" -lt 6700 ]]; then
  log "unexpected generated row count for repaired dataset; aborting"
  exit 2
fi

if [[ -f "$PASS2_LOG" ]]; then
  mv "$PASS2_LOG" "${PASS2_LOG%.log}_$(date +%Y%m%d_%H%M%S).log"
fi
if [[ -f "$ANALYSIS_LOG" ]]; then
  mv "$ANALYSIS_LOG" "${ANALYSIS_LOG%.log}_$(date +%Y%m%d_%H%M%S).log"
fi

clear_partial_pass2
log "partial-pass2-artifacts-cleared"

set +e
"$REPLAY_PY" "$GUARD_PY" \
  --gpu-index 0 \
  --max-vram-fraction "$MAX_VRAM_FRACTION" \
  --poll-seconds "$POLL_SECONDS" \
  --breach-polls 1 \
  --kill-timeout-seconds 15 \
  --log-file "$PASS2_LOG" \
  --chdir "$ROOT" \
  -- \
  "$REPLAY_PY" scripts/experiments/personality/personality_sweep_v3_two_pass.py \
    --model Qwen/Qwen3.5-9B \
    --output "$OUT_REL" \
    --skip-pass1 \
    --quantize bf16 \
    --replay-quantize bf16 \
    --replay-batch-size "$REPLAY_BATCH_SIZE" \
    --replay-max-total-tokens "$REPLAY_MAX_TOTAL_TOKENS"
pass2_rc=$?
set -e
log "pass2-finished rc=$pass2_rc"

if [[ "$pass2_rc" -ne 0 ]]; then
  log "pass2-failed-no-analysis"
  exit "$pass2_rc"
fi

if ! grep -q "\[PASS2 DONE\]" "$PASS2_LOG"; then
  log "pass2-log-missing-completion-marker"
  exit 3
fi

log "analysis-start"
"$ANALYSIS_PY" scripts/experiments/personality/analyze_personality_phase_sweep.py \
  --sweep-dir "$OUT_REL" \
  --output-dir "${ANALYSIS_DIR#$ROOT/}" \
  >"$ANALYSIS_LOG" 2>&1
log "analysis-finished"

log "visualizer-start"
"$VIZ_PY" scripts/experiments/personality/build_personality_visualizer.py \
  --sweep-dir "$OUT_REL" \
  --analysis-json "${ANALYSIS_DIR#$ROOT/}/analysis_results.json" \
  --pass2-log "${PASS2_LOG#$ROOT/}" \
  --control-dir sweep_v4/personality_control_reasoning_v2 \
  --output-html "${VIZ_HTML#$ROOT/}" \
  --title "ws15k repaired response-only phase visualizer" \
  >>"$ANALYSIS_LOG" 2>&1
log "visualizer-finished"

write_handoff_report
log "handoff-report-written path=$REPORT_MD"
log "safe-overnight-done"
