#!/bin/bash
# Auto-launcher: waits for capture_and_steer_27b.py Phase 2 to finish,
# then starts magnitude_calibrated_steering.py.
#
# Usage:
#   nohup bash queue_magnitude_sweep.sh > /tmp/queue_magnitude_sweep.log 2>&1 &

set -e

PHASE2_PID=133024
VENV="/home/orwel/dev_genius/qwen35_venv/bin/activate"
WORKDIR="/home/orwel/dev_genius/experiments/Character Creation"
LOG="/tmp/magnitude_calibrated_steering.log"

echo "[$(date)] Waiting for Phase 2 (PID $PHASE2_PID) to finish..."

# Poll every 60 seconds
while kill -0 $PHASE2_PID 2>/dev/null; do
    sleep 60
done

echo "[$(date)] Phase 2 finished! Waiting 30s for GPU memory to free..."
sleep 30

echo "[$(date)] Starting magnitude-calibrated steering sweep..."
cd "$WORKDIR"
source "$VENV"

python -u magnitude_calibrated_steering.py \
    --alpha-base 5 8 12 \
    --scaling sqrt \
    --output ./magnitude_calibrated_results \
    >> "$LOG" 2>&1

echo "[$(date)] Magnitude sweep complete!"
echo "[$(date)] Results in: $WORKDIR/magnitude_calibrated_results/"
