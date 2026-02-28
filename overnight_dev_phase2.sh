#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════════
# OVERNIGHT DEV SERVER PHASE 2 — After SAE Training Completes
# Chains after overnight_dev.sh: runs sycophancy arena, steering tests, and
# additional SAE analysis.
#
# GPU mapping (confirmed):
#   CUDA_VISIBLE_DEVICES=0 → RTX 4090 (nvidia-smi index 1)
#   CUDA_VISIBLE_DEVICES=1 → RTX 3090 (nvidia-smi index 0)
#
# Usage:
#   cd /home/orwel/dev_genius/sae_8b
#   nohup bash overnight_dev_phase2.sh > overnight_phase2.log 2>&1 &
# ═══════════════════════════════════════════════════════════════════════════════

set -e

SAE_DIR="/home/orwel/dev_genius/sae_8b"
VENV="source /home/orwel/dev_genius/venv/bin/activate"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

cd "$SAE_DIR"
eval "$VENV"

# ─── Phase 0: Wait for overnight_dev.sh to finish ─────────────────────────
log "Phase 0: Waiting for overnight_dev.sh to complete..."
while ! grep -q "ALL OVERNIGHT TASKS COMPLETE" "$SAE_DIR/overnight.log" 2>/dev/null; do
    sleep 60
done
log "Phase 1 complete. Starting Phase 2 pipeline..."

# Free GPU memory
python3 -c "import torch; torch.cuda.empty_cache()" 2>/dev/null
sleep 5

# ─── Phase 1: Sycophancy Steering Test (single GPU) ───────────────────────
# Uses directions extracted by sycophancy_probe_8b.py
log "Phase 1: Sycophancy steering test — alpha sweep..."

SYCO_DIR="$SAE_DIR/sycophancy_results"
STEER_DIR="$SAE_DIR/sycophancy_steer_results"

# Find direction files from the probe
PROBE_DIR=""
for d in "$SYCO_DIR/none" "$SYCO_DIR/analysis" "$SYCO_DIR"; do
    if ls "$d"/sycophancy_direction_L*.pt 2>/dev/null | head -1 > /dev/null 2>&1; then
        PROBE_DIR="$d"
        break
    fi
done

if [ -n "$PROBE_DIR" ]; then
    log "  Found directions in: $PROBE_DIR"

    # Sweep with no system prompt
    CUDA_VISIBLE_DEVICES=0 python3 -u sycophancy_steer_test_8b.py \
        --direction-dir "$PROBE_DIR" \
        --output "$STEER_DIR/sweep_none" \
        --sweep \
        --system none \
        > sycophancy_steer_none.log 2>&1
    log "  None sweep DONE."

    # Sweep with V4 (Skippy) system prompt
    CUDA_VISIBLE_DEVICES=0 python3 -u sycophancy_steer_test_8b.py \
        --direction-dir "$PROBE_DIR" \
        --output "$STEER_DIR/sweep_v4" \
        --sweep \
        --system v4 \
        > sycophancy_steer_v4.log 2>&1
    log "  V4 sweep DONE."

    # Sweep with honest system prompt
    CUDA_VISIBLE_DEVICES=0 python3 -u sycophancy_steer_test_8b.py \
        --direction-dir "$PROBE_DIR" \
        --output "$STEER_DIR/sweep_honest" \
        --sweep \
        --system honest \
        > sycophancy_steer_honest.log 2>&1
    log "  Honest sweep DONE."
else
    log "  WARNING: No sycophancy directions found. Skipping steering test."
fi

# Free GPU
python3 -c "import torch; torch.cuda.empty_cache()" 2>/dev/null
sleep 5

# ─── Phase 2: Sycophancy Arena (dual GPU) ─────────────────────────────────
log "Phase 2: Sycophancy Arena — 3 cycles of 6 pairs..."

python3 -u sycophancy_arena_8b.py \
    --output "$SAE_DIR/sycophancy_arena" \
    --max-rounds 18 \
    --turns-per-round 12 \
    --max-new-tokens 1024 \
    > sycophancy_arena.log 2>&1
log "Sycophancy Arena DONE."

# Free GPU
python3 -c "import torch; torch.cuda.empty_cache()" 2>/dev/null
sleep 5

# ─── Phase 3: Debate Arena v4 with novel topics (dual GPU) ────────────────
log "Phase 3: Debate Arena v4 — 2 cycles of 10 pairs..."

python3 -u debate_arena_v4.py \
    --output "$SAE_DIR/debate_arena_v4" \
    --max-rounds 20 \
    --turns-per-round 12 \
    --max-new-tokens 1024 \
    --enable-doom-detector \
    > debate_arena_v4.log 2>&1
log "Debate Arena v4 DONE."

# ─── Phase 4: Arena sycophancy steering (if arena produced directions) ─────
ARENA_DIR="$SAE_DIR/sycophancy_arena"
LAST_ROUND=$(ls -d "$ARENA_DIR"/round_* 2>/dev/null | sort | tail -1)
if [ -n "$LAST_ROUND" ] && ls "$LAST_ROUND/analysis"/sycophancy_direction_L*.pt 2>/dev/null | head -1 > /dev/null 2>&1; then
    log "Phase 4: Steering with arena-extracted directions..."

    python3 -c "import torch; torch.cuda.empty_cache()" 2>/dev/null

    CUDA_VISIBLE_DEVICES=0 python3 -u sycophancy_steer_test_8b.py \
        --direction-dir "$LAST_ROUND/analysis" \
        --output "$STEER_DIR/arena_directions" \
        --sweep \
        --system none \
        > sycophancy_steer_arena.log 2>&1
    log "  Arena direction sweep DONE."
else
    log "Phase 4: No arena directions found. Skipping."
fi

# ─── Summary ──────────────────────────────────────────────────────────────
log "═══════════════════════════════════════════════════════════"
log "ALL PHASE 2 TASKS COMPLETE"
log "═══════════════════════════════════════════════════════════"
log "Results:"
log "  Sycophancy steering: $STEER_DIR/"
log "  Sycophancy arena:    $ARENA_DIR/"
log "  Debate arena v4:     $SAE_DIR/debate_arena_v4/"
log "  Logs:                $SAE_DIR/*.log"
