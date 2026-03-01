#!/bin/bash
# Pipeline: SAE Collection → Personality Sweep on dual GPUs
# Chains both tasks so GPUs never go idle
#
# Usage: bash run_8b_thinking_pipeline.sh
# Logs: ./pipeline_gpu_a.log, ./pipeline_gpu_b.log

set -e

VENV="$HOME/dev_genius/venv"
WORKDIR="$HOME/dev_genius/sae_8b"
JOURNAL_DIR="$HOME/dev_genius/Journal_creation/created_characters_copy"
POP_DIR="$HOME/dev_genius/Population_Generator/population_data"

source "$VENV/bin/activate"
cd "$WORKDIR"

echo "[$(date)] Starting 8B Thinking pipeline on dual GPUs"
echo "  SAE collection: 500K tokens per layer"
echo "  Personality sweep: ~10M tokens"

# ── Phase 1: SAE Activation Collection (2 GPUs, ~2-3 hours) ──
echo "[$(date)] Phase 1: SAE Activation Collection"

# GPU A (4090): L9 + L15
CUDA_VISIBLE_DEVICES=0 nohup python3 -u sae_collect_8b_thinking.py \
    --layers 9 15 \
    --max-tokens 500000 \
    --max-gen-tokens 512 \
    --n-reps 4 \
    --output ./sae_8b_thinking/gpu_a \
    > ./sae_collection_gpu_a.log 2>&1 &
PID_A=$!
echo "  GPU A (4090): L9+L15, PID=$PID_A"

# GPU B (3090): L22 + L29
CUDA_VISIBLE_DEVICES=1 nohup python3 -u sae_collect_8b_thinking.py \
    --layers 22 29 \
    --max-tokens 500000 \
    --max-gen-tokens 512 \
    --n-reps 4 \
    --output ./sae_8b_thinking/gpu_b \
    > ./sae_collection_gpu_b.log 2>&1 &
PID_B=$!
echo "  GPU B (3090): L22+L29, PID=$PID_B"

# Wait for both SAE collections to finish
echo "[$(date)] Waiting for SAE collection to finish..."
wait $PID_A
echo "[$(date)] GPU A SAE collection DONE (exit=$?)"
wait $PID_B
echo "[$(date)] GPU B SAE collection DONE (exit=$?)"

echo "[$(date)] Phase 1 complete. Starting Phase 2."

# ── Phase 2: Personality Sweep (2 GPUs, ~15-20 hours) ──
echo "[$(date)] Phase 2: Personality Sweep (10M tokens)"

# GPU A (4090): odd character IDs
CUDA_VISIBLE_DEVICES=0 nohup python3 -u personality_sweep_collector.py \
    --split odd \
    --output ./sweep_output/gpu_a \
    --max-gen-tokens 512 \
    --temperature 0.8 \
    --journal-dir "$JOURNAL_DIR" \
    --population-dir "$POP_DIR" \
    > ./sweep_gpu_a.log 2>&1 &
PID_A=$!
echo "  GPU A (4090): odd chars, PID=$PID_A"

# GPU B (3090): even character IDs
CUDA_VISIBLE_DEVICES=1 nohup python3 -u personality_sweep_collector.py \
    --split even \
    --output ./sweep_output/gpu_b \
    --max-gen-tokens 512 \
    --temperature 0.8 \
    --journal-dir "$JOURNAL_DIR" \
    --population-dir "$POP_DIR" \
    > ./sweep_gpu_b.log 2>&1 &
PID_B=$!
echo "  GPU B (3090): even chars, PID=$PID_B"

echo "[$(date)] Personality sweep launched. PIDs: A=$PID_A, B=$PID_B"
echo "  Monitor: tail -f sweep_gpu_a.log sweep_gpu_b.log"
echo "  Expected completion: ~15-20 hours"

# Wait for sweep to finish
wait $PID_A
echo "[$(date)] GPU A sweep DONE (exit=$?)"
wait $PID_B
echo "[$(date)] GPU B sweep DONE (exit=$?)"

echo "[$(date)] Full pipeline complete!"
echo "  SAE data: ./sae_8b_thinking/"
echo "  Sweep data: ./sweep_output/"
