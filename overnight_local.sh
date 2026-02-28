#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════════
# OVERNIGHT LOCAL (PRO 6000) ORCHESTRATOR
# Waits for 27B layer scan to finish, then runs sycophancy probe + SAE collection
#
# Usage:
#   cd /home/orwel/dev_genius/experiments/Character\ Creation
#   nohup bash overnight_local.sh > overnight_local.log 2>&1 &
# ═══════════════════════════════════════════════════════════════════════════════

set -e

PROJECT_DIR="/home/orwel/dev_genius/experiments/Character Creation"
SCAN_PID=153318
VENV="source /home/orwel/dev_genius/qwen35_venv/bin/activate"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

cd "$PROJECT_DIR"

# ─── Phase 1: Wait for 27B layer scan to finish ──────────────────────────
log "Phase 1: Waiting for 27B abliterated layer scan (PID=$SCAN_PID)..."

while kill -0 $SCAN_PID 2>/dev/null; do
    # Check progress
    LAYERS_DONE=$(python3 -c "
import json
with open('qwen35_map/27b-abliterated/layer_scan_results.json') as f:
    d = json.load(f)
print(sum(1 for k in d if k.startswith('L')))" 2>/dev/null || echo "?")
    log "  Layers scanned: $LAYERS_DONE/64"
    sleep 120
done
log "27B layer scan COMPLETE."

# Free GPU memory
sleep 5
log "GPU memory freed."

# ─── Phase 2: Sycophancy probe on BASE 27B ───────────────────────────────
log "Phase 2: Sycophancy probe (base 27B)..."
eval "$VENV"

python3 -u sycophancy_probe_27b.py --model base \
    > sycophancy_probe_27b_base.log 2>&1
log "Base sycophancy probe DONE."

# Free GPU
python3 -c "import torch; torch.cuda.empty_cache()" 2>/dev/null
sleep 5

# ─── Phase 3: Sycophancy probe on ABLITERATED 27B ────────────────────────
log "Phase 3: Sycophancy probe (abliterated 27B)..."

python3 -u sycophancy_probe_27b.py --model abliterated \
    > sycophancy_probe_27b_abliterated.log 2>&1
log "Abliterated sycophancy probe DONE."

# Free GPU
python3 -c "import torch; torch.cuda.empty_cache()" 2>/dev/null
sleep 5

# ─── Phase 4: SAE activation collection on BASE 27B ──────────────────────
log "Phase 4: SAE activation collection (base 27B)..."

# Uses the existing 27B SAE collection script
python3 -u sae_collect_activations.py \
    --model base \
    --layers 50 44 36 16 \
    --max-tokens 500000 \
    --max-gen-tokens 256 \
    --n-reps 4 \
    > sae_27b_collect_base.log 2>&1
log "27B SAE collection DONE."

# ─── Phase 5: Train SAE on L50 (highest priority target) ─────────────────
log "Phase 5: SAE training — L50 (super-hub)..."

python3 -c "import torch; torch.cuda.empty_cache()" 2>/dev/null

python3 -u sae_train.py \
    --layer 50 \
    --model-tag base \
    --total-steps 50000 \
    --batch-size 4096 \
    > sae_27b_train_L50.log 2>&1
log "L50 SAE training DONE."

# ─── Phase 6: Train SAE on L44 ───────────────────────────────────────────
log "Phase 6: SAE training — L44 (sarcasm region)..."

python3 -u sae_train.py \
    --layer 44 \
    --model-tag base \
    --total-steps 50000 \
    --batch-size 4096 \
    > sae_27b_train_L44.log 2>&1
log "L44 SAE training DONE."

# ─── Summary ──────────────────────────────────────────────────────────────
log "═══════════════════════════════════════════════════════════"
log "ALL OVERNIGHT LOCAL TASKS COMPLETE"
log "═══════════════════════════════════════════════════════════"
log "Results:"
log "  Sycophancy (base):  sycophancy_probe_27b_results/base/"
log "  Sycophancy (abli):  sycophancy_probe_27b_results/abliterated/"
log "  SAE activations:    sae_activations/base/"
log "  SAE L50 model:      sae_models/base/L50/"
log "  SAE L44 model:      sae_models/base/L44/"
log "  Logs:               *_27b_*.log"
