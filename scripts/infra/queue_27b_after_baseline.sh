#!/usr/bin/env bash
# Queue: 27B personality sweep AFTER baseline collection finishes.
# Polls for baseline_activation_collector to complete, then launches 27B sweep
# with 1024 max_gen_tokens (up from 512 — 8B showed all-think with 512).
#
# Usage:
#   nohup bash scripts/infra/queue_27b_after_baseline.sh > logs/queue_27b_after_baseline.log 2>&1 &

set -euo pipefail

PROJECT_DIR="/home/orwel/dev_genius/experiments/Character Creation"
VENV="/home/orwel/dev_genius/qwen35_venv/bin/activate"
SCRIPT="${PROJECT_DIR}/scripts/experiments/personality/personality_sweep_collector.py"
LOG_DIR="${PROJECT_DIR}/logs"

# 27B config
MODEL="Qwen/Qwen3.5-27B-FP8"
TARGET_LAYERS="16,36,44,50"
DTYPE="auto"
BATCH_SIZE=25
MAX_GEN_TOKENS=1024
OUTPUT_DIR="${PROJECT_DIR}/sweep_output/blackwell_27b"

# Data dirs
JOURNAL_DIR="/home/orwel/dev_genius/Journal_creation/created_characters_copy"
POPULATION_DIR="/home/orwel/dev_genius/Population_Generator/population_data"

echo "[$(date)] Waiting for baseline_activation_collector to finish..."

while pgrep -f "baseline_activation_collector" > /dev/null 2>&1; do
    echo "[$(date)] Baseline still running. Sleeping 60s..."
    sleep 60
done

echo "[$(date)] Baseline complete. Waiting 30s for GPU cleanup..."
sleep 30

# ── Launch 27B sweep ──────────────────────────────────────────
source "${VENV}"

TRANSFORMERS_VER=$(python3 -c "import transformers; print(transformers.__version__)" 2>/dev/null || echo "unknown")
echo "[$(date)] transformers version: ${TRANSFORMERS_VER}"

FREE_MEM=$(nvidia-smi --id=0 --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null || echo 0)
echo "[$(date)] GPU 0 free memory: ${FREE_MEM} MB"

echo ""
echo "================================================================"
echo "[$(date)] LAUNCHING: 27B Personality Sweep"
echo "  Model:        ${MODEL}"
echo "  Layers:       ${TARGET_LAYERS}"
echo "  Dtype:        ${DTYPE}"
echo "  Batch:        ${BATCH_SIZE}"
echo "  Max tokens:   ${MAX_GEN_TOKENS}"
echo "  Output:       ${OUTPUT_DIR}"
echo "================================================================"
echo ""

cd "${PROJECT_DIR}"
mkdir -p "${OUTPUT_DIR}"

python3 -u "${SCRIPT}" \
    --model "${MODEL}" \
    --target-layers "${TARGET_LAYERS}" \
    --dtype "${DTYPE}" \
    --no-thinking \
    --output "${OUTPUT_DIR}" \
    --batch-size "${BATCH_SIZE}" \
    --max-gen-tokens "${MAX_GEN_TOKENS}" \
    --temperature 0.8 \
    --skip-existing \
    --journal-dir "${JOURNAL_DIR}" \
    --population-dir "${POPULATION_DIR}" \
    2>&1 | tee "${LOG_DIR}/personality_sweep_27b.log"

echo ""
echo "================================================================"
echo "[$(date)] 27B personality sweep COMPLETE"
echo "[$(date)] Output: ${OUTPUT_DIR}"
echo "================================================================"
