#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════════
# OVERNIGHT DEV SERVER ORCHESTRATOR
# Manages SAE training + sycophancy probe across 3090 + 4090
#
# GPU mapping (confirmed):
#   CUDA_VISIBLE_DEVICES=0 → RTX 4090 (nvidia-smi index 1)
#   CUDA_VISIBLE_DEVICES=1 → RTX 3090 (nvidia-smi index 0)
#
# Prerequisites:
#   - SAE activation collection already running on 3090 (PID from sae_collect.log)
#   - Alpha sweep v2 may still be running on 4090
#
# Usage:
#   cd /home/orwel/dev_genius/sae_8b
#   nohup bash overnight_dev.sh > overnight.log 2>&1 &
# ═══════════════════════════════════════════════════════════════════════════════

set -e

SAE_DIR="/home/orwel/dev_genius/sae_8b"
EVAL_DIR="/home/orwel/dev_genius/orthogonal_eval"
VENV="source /home/orwel/dev_genius/venv/bin/activate"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

cd "$SAE_DIR"

# ─── Phase 1: Wait for alpha sweep to finish on 4090 ─────────────────────
ALPHA_LOG="$EVAL_DIR/orthogonal_eval_v2_sweep.log"
if [ -f "$ALPHA_LOG" ]; then
    log "Phase 1: Waiting for alpha sweep to finish on 4090..."
    while ! grep -q "^DONE$" "$ALPHA_LOG" 2>/dev/null; do
        sleep 30
    done
    log "Alpha sweep DONE."
else
    log "No alpha sweep log found, assuming 4090 is free."
fi

# ─── Phase 2: Run sycophancy probe on 4090 (while 3090 still collecting) ──
log "Phase 2: Starting sycophancy probe on 4090..."
eval "$VENV"
CUDA_VISIBLE_DEVICES=0 python3 -u sycophancy_probe_8b.py \
    --output-dir "$SAE_DIR/sycophancy_results" \
    > sycophancy_probe.log 2>&1
log "Sycophancy probe DONE."

# ─── Phase 3: Wait for activation collection to finish on 3090 ────────────
log "Phase 3: Waiting for activation collection to finish..."
COLLECT_LOG="$SAE_DIR/sae_collect.log"
while ! grep -q "COLLECTION COMPLETE" "$COLLECT_LOG" 2>/dev/null; do
    sleep 30
done
log "Activation collection DONE."

# ─── Phase 4: Train SAE — L22 on 3090, L9 on 4090 (parallel) ─────────────
log "Phase 4: Training SAE — L22 (3090) + L9 (4090)..."
CUDA_VISIBLE_DEVICES=1 python3 -u sae_8b_pipeline.py train --layer 22 \
    > sae_train_L22.log 2>&1 &
PID_L22=$!

CUDA_VISIBLE_DEVICES=0 python3 -u sae_8b_pipeline.py train --layer 9 \
    > sae_train_L09.log 2>&1 &
PID_L09=$!

log "  L22 PID=$PID_L22, L9 PID=$PID_L09"
wait $PID_L22
log "  L22 training DONE."
wait $PID_L09
log "  L9 training DONE."

# ─── Phase 5: Train SAE — L15 on 3090, L29 on 4090 (parallel) ────────────
log "Phase 5: Training SAE — L15 (3090) + L29 (4090)..."
CUDA_VISIBLE_DEVICES=1 python3 -u sae_8b_pipeline.py train --layer 15 \
    > sae_train_L15.log 2>&1 &
PID_L15=$!

CUDA_VISIBLE_DEVICES=0 python3 -u sae_8b_pipeline.py train --layer 29 \
    > sae_train_L29.log 2>&1 &
PID_L29=$!

log "  L15 PID=$PID_L15, L29 PID=$PID_L29"
wait $PID_L15
log "  L15 training DONE."
wait $PID_L29
log "  L29 training DONE."

# ─── Phase 6: Gen-only SAE training — L22 on 3090, L9 on 4090 ────────────
log "Phase 6: Gen-only SAE training..."
GENONLY_DIR="$SAE_DIR/sae_8b/models_genonly"
mkdir -p "$GENONLY_DIR"

CUDA_VISIBLE_DEVICES=1 python3 -u sae_8b_pipeline.py train --layer 22 --gen-only \
    --output "$GENONLY_DIR/L22" > sae_train_L22_genonly.log 2>&1 &
PID_G22=$!

CUDA_VISIBLE_DEVICES=0 python3 -u sae_8b_pipeline.py train --layer 9 --gen-only \
    --output "$GENONLY_DIR/L09" > sae_train_L09_genonly.log 2>&1 &
PID_G09=$!

log "  L22-gen PID=$PID_G22, L9-gen PID=$PID_G09"
wait $PID_G22
log "  L22 gen-only DONE."
wait $PID_G09
log "  L9 gen-only DONE."

# ─── Phase 7: Gen-only for L15, L29 ──────────────────────────────────────
log "Phase 7: Gen-only SAE — L15 + L29..."
CUDA_VISIBLE_DEVICES=1 python3 -u sae_8b_pipeline.py train --layer 15 --gen-only \
    --output "$GENONLY_DIR/L15" > sae_train_L15_genonly.log 2>&1 &
PID_G15=$!

CUDA_VISIBLE_DEVICES=0 python3 -u sae_8b_pipeline.py train --layer 29 --gen-only \
    --output "$GENONLY_DIR/L29" > sae_train_L29_genonly.log 2>&1 &
PID_G29=$!

wait $PID_G15
wait $PID_G29
log "  All gen-only training DONE."

# ─── Summary ──────────────────────────────────────────────────────────────
log "═══════════════════════════════════════════════════════════"
log "ALL OVERNIGHT TASKS COMPLETE"
log "═══════════════════════════════════════════════════════════"
log "Results:"
log "  SAE models: $SAE_DIR/sae_8b/models/"
log "  Gen-only:   $GENONLY_DIR/"
log "  Sycophancy: $SAE_DIR/sycophancy_results/"
log "  Logs:       $SAE_DIR/*.log"

# List final model sizes
for d in "$SAE_DIR"/sae_8b/models/L*/; do
    if [ -f "$d/sae_final.pt" ]; then
        sz=$(du -sh "$d/sae_final.pt" | cut -f1)
        log "  $(basename $d): $sz"
    fi
done
