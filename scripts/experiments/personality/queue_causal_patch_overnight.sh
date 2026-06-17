#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/orwel/dev_genius/experiments/Character Creation"
VENV="/home/orwel/dev_genius/venv/bin/python"
OUT_ROOT="$ROOT/sweep_v4"
REPORT_ROOT="$ROOT/reports"
LOG_DIR="$ROOT/logs"
mkdir -p "$LOG_DIR" "$OUT_ROOT" "$REPORT_ROOT"

STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_LOG="$LOG_DIR/causal_patch_overnight_${STAMP}.log"
exec > >(tee -a "$RUN_LOG") 2>&1

log() {
  echo "[$(date --iso-8601=seconds)] $*"
}

run() {
  log "CMD: $*"
  "$@"
}

log "starting overnight causal patch queue"
log "run_log=$RUN_LOG"

QWEN35_PATCH_SPECS="baseline=none,l0_1p0=0@1.0:full,l1_1p0=1@1.0:full,l01_1p0=0+1@1.0:full,l1_1p0_64=1@1.0:64,l01_1p0_64=0+1@1.0:64,l1_2p0_64=1@2.0:64,l01_2p0_64=0+1@2.0:64"
QWEN35_REFINE_SPECS="baseline=none,l1_0p5_64=1@0.5:64,l1_1p0_64=1@1.0:64,l1_1p5_64=1@1.5:64,l1_2p0_64=1@2.0:64,l01_0p5_64=0+1@0.5:64,l01_1p0_64=0+1@1.0:64,l01_1p5_64=0+1@1.5:64,l01_2p0_64=0+1@2.0:64"

QWEN35_OUT="$OUT_ROOT/causal_think_region_patch_qwen35_ages_full_${STAMP}"
run "$VENV" "$ROOT/scripts/experiments/personality/run_causal_think_region_patching.py" \
  --model-path "/home/orwel/.cache/huggingface/hub/models--Qwen--Qwen3.5-9B/snapshots/c202236235762e1c871ad0ccb60c8ee5ba337b9a" \
  --directions "$OUT_ROOT/meta_sham_contrastive_replay_qwen35_20260416_nonanswer/candidate_directions.npz" \
  --output-dir "$QWEN35_OUT" \
  --patch-specs "$QWEN35_PATCH_SPECS" \
  --max-new-tokens 240

QWEN36_REPLAY_TAG="meta_sham_contrastive_replay_qwen36_35b_a3b_${STAMP}"
run "$VENV" "$ROOT/scripts/experiments/personality/replay_meta_sham_contrastive.py" \
  --model-path "/home/orwel/dev_genius/models/Qwen3.6-35B-A3B" \
  --tag "$QWEN36_REPLAY_TAG"

QWEN36_DIR_JSON="$OUT_ROOT/$QWEN36_REPLAY_TAG/candidate_directions.json"
QWEN36_DIR_NPZ="$OUT_ROOT/$QWEN36_REPLAY_TAG/candidate_directions.npz"
QWEN36_PATCH_SPECS="$("$VENV" - "$QWEN36_DIR_JSON" <<'PY'
import json, sys
rows = json.load(open(sys.argv[1], "r", encoding="utf-8"))
layers = []
for row in rows:
    if row.get("comparison") == "real_minus_think" and row.get("region") == "think_region":
        layer = int(row["layer"])
        if layer not in layers:
            layers.append(layer)
layers = layers[:2]
if not layers:
    raise SystemExit("no think_region layers found")
patches = ["baseline=none"]
patches.append(f"l{layers[0]}_1p0={layers[0]}@1.0:full")
patches.append(f"l{layers[0]}_1p0_64={layers[0]}@1.0:64")
if len(layers) > 1:
    patches.append(f"l{layers[1]}_1p0={layers[1]}@1.0:full")
    patches.append(f"l{layers[1]}_1p0_64={layers[1]}@1.0:64")
    patches.append(f"l{layers[0]}_{layers[1]}_1p0={layers[0]}+{layers[1]}@1.0:full")
    patches.append(f"l{layers[0]}_{layers[1]}_2p0_64={layers[0]}+{layers[1]}@2.0:64")
print(",".join(patches))
PY
)"
log "qwen36_patch_specs=$QWEN36_PATCH_SPECS"

QWEN36_OUT="$OUT_ROOT/causal_think_region_patch_qwen36_35b_a3b_${STAMP}"
run "$VENV" "$ROOT/scripts/experiments/personality/run_causal_think_region_patching.py" \
  --model-path "/home/orwel/dev_genius/models/Qwen3.6-35B-A3B" \
  --directions "$QWEN36_DIR_NPZ" \
  --output-dir "$QWEN36_OUT" \
  --patch-specs "$QWEN36_PATCH_SPECS" \
  --max-new-tokens 240

QWEN35_REFINE_OUT="$OUT_ROOT/causal_think_region_patch_qwen35_refine_${STAMP}"
run "$VENV" "$ROOT/scripts/experiments/personality/run_causal_think_region_patching.py" \
  --model-path "/home/orwel/.cache/huggingface/hub/models--Qwen--Qwen3.5-9B/snapshots/c202236235762e1c871ad0ccb60c8ee5ba337b9a" \
  --directions "$OUT_ROOT/meta_sham_contrastive_replay_qwen35_20260416_nonanswer/candidate_directions.npz" \
  --output-dir "$QWEN35_REFINE_OUT" \
  --patch-specs "$QWEN35_REFINE_SPECS" \
  --max-new-tokens 240

cat > "$OUT_ROOT/causal_patch_overnight_${STAMP}.done.json" <<JSON
{
  "finished_at": "$(date --iso-8601=seconds)",
  "run_log": "$RUN_LOG",
  "qwen35_full": "$QWEN35_OUT",
  "qwen36_replay_tag": "$QWEN36_REPLAY_TAG",
  "qwen36_full": "$QWEN36_OUT",
  "qwen35_refine": "$QWEN35_REFINE_OUT"
}
JSON

log "overnight causal patch queue finished"
