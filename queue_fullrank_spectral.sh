#!/bin/bash
# Queue full-rank spectral analysis after Phase 2 + magnitude sweep finish
# This waits for the GPU to be free, then starts the 11-hour spectral run
#
# Usage: nohup bash queue_fullrank_spectral.sh > /tmp/queue_fullrank_spectral.log 2>&1 &

set -e

cd "/home/orwel/dev_genius/experiments/Character Creation"
source /home/orwel/dev_genius/qwen35_venv/bin/activate

echo "$(date): Waiting for GPU to be free..."
echo "  Monitoring PIDs: capture_and_steer (Phase 2) and magnitude_calibrated_steering"

# Wait for Phase 2 (capture_and_steer_27b.py) to finish
while pgrep -f "capture_and_steer_27b.py" > /dev/null 2>&1; do
    echo "$(date): Phase 2 still running, checking again in 5 min..."
    sleep 300
done
echo "$(date): Phase 2 done."

# Wait for magnitude sweep to finish (queued after Phase 2)
while pgrep -f "magnitude_calibrated_steering.py" > /dev/null 2>&1; do
    echo "$(date): Magnitude sweep still running, checking again in 5 min..."
    sleep 300
done
echo "$(date): Magnitude sweep done."

# Small cooldown for GPU memory cleanup
sleep 30

echo "$(date): Starting full-rank spectral analysis..."
echo "  Expected runtime: ~11 hours"

python -u fullrank_spectral_analysis.py --resume 2>&1

echo "$(date): Full-rank spectral analysis COMPLETE."
