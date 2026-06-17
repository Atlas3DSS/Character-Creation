#!/bin/bash
# Queue abliterated connectome + comparison after spectral analysis finishes
# Chain: magnitude sweep → spectral analysis → THIS (abliterated connectome → comparison)
#
# Usage: nohup bash scripts/infra/queue_abliterated_connectome.sh > /tmp/queue_abliterated_connectome.log 2>&1 &

set -e

cd "/home/orwel/dev_genius/experiments/Character Creation"
source /home/orwel/dev_genius/qwen35_venv/bin/activate
VENV_PY="/home/orwel/dev_genius/qwen35_venv/bin/python"
GUARD_PY="scripts/infra/run_with_vram_guard.py"
MAX_VRAM_FRACTION="${MAX_VRAM_FRACTION:-0.89}"
GUARD_POLL_SECONDS="${GUARD_POLL_SECONDS:-5}"
ABLIT_LOG="logs/queue_abliterated_connectome_guard.log"
EVAL_LOG="logs/queue_abliterated_head_to_head_guard.log"

echo "$(date): Waiting for GPU to be free..."
echo "  Monitoring: magnitude_calibrated_steering, fullrank_spectral_analysis"

# Wait for magnitude sweep
while pgrep -f "magnitude_calibrated_steering.py" > /dev/null 2>&1; do
    echo "$(date): Magnitude sweep still running, checking again in 5 min..."
    sleep 300
done
echo "$(date): Magnitude sweep done."

# Wait for spectral analysis
while pgrep -f "fullrank_spectral_analysis.py" > /dev/null 2>&1; do
    echo "$(date): Spectral analysis still running, checking again in 5 min..."
    sleep 300
done
echo "$(date): Spectral analysis done."

# Small cooldown for GPU memory cleanup
sleep 30

# ── Phase A: Abliterated connectome (Phase 1 baseline + Phase 2 connectome only) ──
echo ""
echo "$(date): =============================================="
echo "$(date): Starting abliterated model connectome mapping"
echo "$(date): Model: huihui-ai/Huihui-Qwen3.5-27B-abliterated (bf16, ~52GB)"
echo "$(date): Phases: 1 (baseline) + 2 (connectome probe)"
echo "$(date): Expected runtime: ~4-6 hours"
echo "$(date): Guard log: ${ABLIT_LOG}"
echo "$(date): =============================================="
echo ""

# Run Phase 1 (baseline eval) and Phase 2 (connectome) only
# Skip Phase 3 (layer scan) and Phase 4 (comparison) — we have a dedicated comparison script
"$VENV_PY" "$GUARD_PY" \
  --gpu-index 0 \
  --max-vram-fraction "$MAX_VRAM_FRACTION" \
  --poll-seconds "$GUARD_POLL_SECONDS" \
  --breach-polls 1 \
  --kill-timeout-seconds 15 \
  --log-file "$ABLIT_LOG" \
  --chdir "/home/orwel/dev_genius/experiments/Character Creation" \
  -- \
  "$VENV_PY" -u scripts/experiments/connectome/map_qwen35.py --model 27b-abliterated --output ./qwen35_map --resume

echo ""
echo "$(date): Abliterated connectome COMPLETE."
echo ""

# GPU memory cleanup
sleep 30

# ── Phase B: Head-to-head eval (needs GPU — expanded math/knowledge battery) ──
echo ""
echo "$(date): =============================================="
echo "$(date): Starting head-to-head eval: abliterated vs base vs steered"
echo "$(date): 50 math + 30 knowledge + 20 sarcasm + 10 identity + 10 refusal"
echo "$(date): Expected runtime: ~1-2 hours"
echo "$(date): Guard log: ${EVAL_LOG}"
echo "$(date): =============================================="
echo ""

"$VENV_PY" "$GUARD_PY" \
  --gpu-index 0 \
  --max-vram-fraction "$MAX_VRAM_FRACTION" \
  --poll-seconds "$GUARD_POLL_SECONDS" \
  --breach-polls 1 \
  --kill-timeout-seconds 15 \
  --log-file "$EVAL_LOG" \
  --chdir "/home/orwel/dev_genius/experiments/Character Creation" \
  -- \
  "$VENV_PY" -u scripts/eval/eval_head_to_head.py --resume --output ./abliteration_comparison

echo ""
echo "$(date): Head-to-head eval COMPLETE."
echo ""

# GPU memory cleanup
sleep 30

# ── Phase C: Connectome comparison (CPU only, fast) ──
echo "$(date): Starting connectome comparison..."
python -u scripts/experiments/connectome/compare_connectomes.py \
    --base ./qwen35_map/27b \
    --abliterated ./qwen35_map/27b-abliterated \
    --output ./abliteration_comparison 2>&1

echo ""
echo "$(date): =============================================="
echo "$(date): ALL DONE"
echo "$(date): Results:"
echo "$(date):   ./qwen35_map/27b-abliterated/        (connectome)"
echo "$(date):   ./abliteration_comparison/            (comparison reports)"
echo "$(date):     - head_to_head_report.md            (eval numbers)"
echo "$(date):     - abliteration_comparison.md        (connectome analysis)"
echo "$(date):     - head_to_head_data.json            (raw eval data)"
echo "$(date):     - comparison_data.json              (connectome data)"
echo "$(date): =============================================="
