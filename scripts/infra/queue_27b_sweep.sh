#!/usr/bin/env bash
# Queue: Baseline activation collection + 27B personality sweep on Blackwell
# Waits for the 8B sweep to finish, then:
#   Phase A: 8B baseline activations from FineFineWeb (null distribution)
#   Phase B: 27B personality sweep (cross-architecture replication)
# Safe: never kills existing processes — only polls and waits.
#
# Usage (from project root):
#   nohup bash scripts/infra/queue_27b_sweep.sh > logs/queue_27b_sweep.log 2>&1 &

set -euo pipefail

PROJECT_DIR="/home/orwel/dev_genius/experiments/Character Creation"
VENV_8B="/home/orwel/dev_genius/qwen35_venv/bin/activate"
VENV_27B="/home/orwel/dev_genius/qwen35_venv/bin/activate"
SWEEP_SCRIPT="${PROJECT_DIR}/scripts/experiments/personality/personality_sweep_collector.py"
BASELINE_SCRIPT="${PROJECT_DIR}/scripts/experiments/personality/baseline_activation_collector.py"
LOG_DIR="${PROJECT_DIR}/logs"

# Baseline config (8B model, same as sweep)
BASELINE_OUTPUT="${PROJECT_DIR}/activations_baseline"
BASELINE_N_SAMPLES=50000
BASELINE_BATCH=25

# 27B model config
MODEL_27B="Qwen/Qwen3.5-27B-FP8"
TARGET_LAYERS_27B="16,36,44,50"
DTYPE_27B="auto"
BATCH_27B=25
MAX_GEN_27B=1024   # 512 was too few for 8B Thinking (all-think, no response)
OUTPUT_27B="${PROJECT_DIR}/sweep_output/blackwell_27b"

# Data dirs (same character grid as 8B)
JOURNAL_DIR="/home/orwel/dev_genius/Journal_creation/created_characters_copy"
POPULATION_DIR="/home/orwel/dev_genius/Population_Generator/population_data"

echo "[$(date)] Queue started: Baseline + 27B sweep on Blackwell"
echo "[$(date)] Phase A: 8B baseline from FineFineWeb (${BASELINE_N_SAMPLES} samples)"
echo "[$(date)] Phase B: 27B personality sweep (${MODEL_27B})"
echo ""

# ══════════════════════════════════════════════════════════════
# Wait for 8B personality sweep to finish
# ══════════════════════════════════════════════════════════════
while pgrep -f "personality_sweep_collector" > /dev/null 2>&1; do
    REMAINING=$(pgrep -fc "personality_sweep_collector" 2>/dev/null || echo 0)
    echo "[$(date)] 8B sweep still running (${REMAINING} processes). Sleeping 60s..."
    sleep 60
done

echo "[$(date)] 8B sweep complete. Waiting 30s for GPU cleanup..."
sleep 30

# ══════════════════════════════════════════════════════════════
# PHASE A: Baseline activation collection (8B model)
# ══════════════════════════════════════════════════════════════
echo ""
echo "================================================================"
echo "[$(date)] PHASE A: Baseline Activation Collection"
echo "  Model:   Qwen/Qwen3-VL-8B-Thinking (same as personality sweep)"
echo "  Samples: ${BASELINE_N_SAMPLES} (stratified across 68 domains)"
echo "  Batch:   ${BASELINE_BATCH}"
echo "  Output:  ${BASELINE_OUTPUT}"
echo "================================================================"
echo ""

source "${VENV_8B}"
cd "${PROJECT_DIR}"
mkdir -p "${BASELINE_OUTPUT}" "${LOG_DIR}"

python3 -u "${BASELINE_SCRIPT}" \
    --output "${BASELINE_OUTPUT}" \
    --n-samples "${BASELINE_N_SAMPLES}" \
    --batch-size "${BASELINE_BATCH}" \
    --max-gen-tokens 512 \
    --temperature 0.8 \
    2>&1 | tee "${LOG_DIR}/baseline_activation_collection.log"

BASELINE_EXIT=$?
echo "[$(date)] Phase A exit code: ${BASELINE_EXIT}"

if [ "${BASELINE_EXIT}" -ne 0 ]; then
    echo "[$(date)] WARNING: Baseline collection had errors. Continuing to Phase B."
fi

echo "[$(date)] Waiting 30s for GPU cleanup before 27B..."
sleep 30

# ══════════════════════════════════════════════════════════════
# PHASE B: 27B Personality Sweep
# ══════════════════════════════════════════════════════════════
echo ""
echo "================================================================"
echo "[$(date)] PHASE B: 27B Personality Sweep"
echo "  Model:   ${MODEL_27B}"
echo "  Layers:  ${TARGET_LAYERS_27B}"
echo "  Dtype:   ${DTYPE_27B}"
echo "  Batch:   ${BATCH_27B}"
echo "  Output:  ${OUTPUT_27B}"
echo "================================================================"
echo ""

source "${VENV_27B}"

TRANSFORMERS_VER=$(python3 -c "import transformers; print(transformers.__version__)" 2>/dev/null || echo "unknown")
echo "[$(date)] transformers version: ${TRANSFORMERS_VER}"

FREE_MEM=$(nvidia-smi --id=0 --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null || echo 0)
echo "[$(date)] GPU 0 free memory: ${FREE_MEM} MB"

mkdir -p "${OUTPUT_27B}"

python3 -u "${SWEEP_SCRIPT}" \
    --model "${MODEL_27B}" \
    --target-layers "${TARGET_LAYERS_27B}" \
    --dtype "${DTYPE_27B}" \
    --no-thinking \
    --output "${OUTPUT_27B}" \
    --batch-size "${BATCH_27B}" \
    --max-gen-tokens "${MAX_GEN_27B}" \
    --temperature 0.8 \
    --skip-existing \
    --journal-dir "${JOURNAL_DIR}" \
    --population-dir "${POPULATION_DIR}" \
    2>&1 | tee "${LOG_DIR}/personality_sweep_27b.log"

echo ""
echo "================================================================"
echo "[$(date)] ALL DONE — Baseline + 27B sweep complete"
echo "[$(date)] Baseline: ${BASELINE_OUTPUT}"
echo "[$(date)] 27B sweep: ${OUTPUT_27B}"
echo "================================================================"
