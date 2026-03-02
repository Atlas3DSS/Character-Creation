#!/usr/bin/env bash
# Queue: 8B SAE retrain L22 + L29 at 50K steps
# Waits for personality sweep to finish, then trains on whichever GPU frees first.
# Safe: never kills sweep processes — only polls and waits.
#
# Usage: ssh orwel@192.168.86.66 "nohup bash /home/orwel/dev_genius/sae_8b/queue_8b_sae_retrain.sh > /home/orwel/dev_genius/sae_8b/queue_8b_sae_retrain.log 2>&1 &"

set -euo pipefail

VENV="/home/orwel/dev_genius/venv/bin/activate"
WORK_DIR="/home/orwel/dev_genius/sae_8b"
SAE_SCRIPT="${WORK_DIR}/sae_train.py"
ACT_DIR="${WORK_DIR}/sae_8b/activations"
MODEL_DIR="${WORK_DIR}/sae_8b/models_50k"
LOG_DIR="${WORK_DIR}"

echo "[$(date)] Queue started: 8B SAE retrain L22+L29 @ 50K steps"
echo "[$(date)] Waiting for personality sweep to finish..."

# Poll every 60s until no personality_sweep processes remain
while pgrep -f "personality_sweep_collector" > /dev/null 2>&1; do
    REMAINING=$(pgrep -fc "personality_sweep_collector" 2>/dev/null || echo 0)
    echo "[$(date)] Sweep still running (${REMAINING} processes). Sleeping 60s..."
    sleep 60
done

echo "[$(date)] Sweep complete. Waiting 30s for cleanup..."
sleep 30

# Activate venv
source "${VENV}"

# Determine which GPU to use (pick the one with most free memory)
GPU_ID=0
FREE_0=$(nvidia-smi --id=0 --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null || echo 0)
FREE_1=$(nvidia-smi --id=1 --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null || echo 0)
if [ "${FREE_1}" -gt "${FREE_0}" ]; then
    GPU_ID=1
fi
echo "[$(date)] Using GPU ${GPU_ID} (free: GPU0=${FREE_0}MB, GPU1=${FREE_1}MB)"

export CUDA_VISIBLE_DEVICES="${GPU_ID}"

mkdir -p "${MODEL_DIR}/L22" "${MODEL_DIR}/L29"

# ── Phase 1: Train L22 at 50K steps ──
echo ""
echo "================================================================"
echo "[$(date)] PHASE 1: Training 8B SAE L22 — 50K steps"
echo "================================================================"

python3 -u "${SAE_SCRIPT}" \
    --layer 22 \
    --model-tag 8b_50k \
    --total-steps 50000 \
    --batch-size 4096 \
    --activations-dir "${ACT_DIR}" \
    --output-dir "${MODEL_DIR}" \
    --expansion 16 \
    --lr 3e-4 \
    --gen-only \
    2>&1 | tee "${LOG_DIR}/sae_train_L22_50k.log"

echo "[$(date)] L22 training complete."

# ── Phase 2: Train L29 at 50K steps ──
echo ""
echo "================================================================"
echo "[$(date)] PHASE 2: Training 8B SAE L29 — 50K steps"
echo "================================================================"

python3 -u "${SAE_SCRIPT}" \
    --layer 29 \
    --model-tag 8b_50k \
    --total-steps 50000 \
    --batch-size 4096 \
    --activations-dir "${ACT_DIR}" \
    --output-dir "${MODEL_DIR}" \
    --expansion 16 \
    --lr 3e-4 \
    --gen-only \
    2>&1 | tee "${LOG_DIR}/sae_train_L29_50k.log"

echo "[$(date)] L29 training complete."

echo ""
echo "================================================================"
echo "[$(date)] ALL DONE — 8B L22+L29 SAEs trained at 50K steps"
echo "[$(date)] Models saved to: ${MODEL_DIR}/"
echo "================================================================"
