#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/orwel/dev_genius/experiments/Character Creation"
VENV="/home/orwel/dev_genius/venv"
PY="${VENV}/bin/python"
OUT_ROOT="${ROOT}/sweep_v4"
LOG_DIR="${ROOT}/logs"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_PATH="${LOG_DIR}/queue_symphonic_voice_overnight_${STAMP}.log"
MANIFEST="${ROOT}/data/symphonic_voice_anchor_manifest_v2.json"
BUILD_TAG="symphonic_voice_probe_dataset_v2"
PROBE_TAG="symphonic_voice_activation_probe_v2"
AXIS_TAG="symphonic_voice_axis_analysis_v2"
PATCH_TAG="symphonic_voice_live_patch_v2"
PORT="${PORT:-30003}"
SESSION="${SESSION:-sglang_qwen36_35b}"
SKIP_SERVER="${SKIP_SERVER:-0}"

mkdir -p "${LOG_DIR}"
touch "${LOG_PATH}"

log() {
  echo "[$(date --iso-8601=seconds)] $*" | tee -a "${LOG_PATH}"
}

latest_dir() {
  local glob="$1"
  ls -dt ${glob} 2>/dev/null | head -n1
}

wait_for_http() {
  local url="$1"
  local tries="${2:-120}"
  for ((i=1;i<=tries;i++)); do
    if curl -sf "${url}" >/dev/null 2>&1; then
      return 0
    fi
    if (( i % 6 == 0 )); then
      log "waiting for http readiness: ${url} try=${i}/${tries}"
    fi
    sleep 5
  done
  return 1
}

wait_for_vram_headroom() {
  local max_used_mib="$1"
  local tries="${2:-720}"
  for ((i=1;i<=tries;i++)); do
    used="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -n1 | tr -d ' ')"
    if [ "${used}" -le "${max_used_mib}" ]; then
      log "VRAM headroom ready: ${used} MiB <= ${max_used_mib} MiB"
      return 0
    fi
    log "waiting for VRAM headroom: used=${used} MiB max=${max_used_mib} MiB"
    sleep 30
  done
  return 1
}

cleanup_server() {
  tmux kill-session -t "${SESSION}" 2>/dev/null || true
}

trap cleanup_server EXIT

source "${VENV}/bin/activate"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4

log "queue start"
if [ "${SKIP_SERVER}" != "1" ]; then
  wait_for_vram_headroom 12000 60
else
  log "skip_server=1, bypassing pre-launch VRAM headroom guard"
fi

if [ "${SKIP_SERVER}" != "1" ]; then
  log "starting local qwen36 sglang"
  LOG_PATH="${LOG_DIR}/sglang_qwen36_35b_${STAMP}.log" \
  MEM_FRACTION_STATIC=0.82 \
  MAX_RUNNING_REQUESTS=64 \
  PORT="${PORT}" \
  SESSION="${SESSION}" \
  "${ROOT}/scripts/experiments/personality/launch_qwen36_sglang_local.sh" | tee -a "${LOG_PATH}"
else
  log "reusing existing local qwen36 sglang on port ${PORT}"
fi

if ! wait_for_http "http://127.0.0.1:${PORT}/v1/models" 120; then
  log "sglang server did not become ready"
  exit 1
fi
log "sglang ready on port ${PORT}"

log "building expanded symphonic dataset"
"${PY}" "${ROOT}/scripts/experiments/personality/build_symphonic_probe_dataset.py" \
  --source-dataset-dir "${ROOT}/sweep_v4/book_character_prefill_dataset_balanced_v6_reviewed_20260417_151017" \
  --anchor-manifest "${MANIFEST}" \
  --base-url "http://127.0.0.1:${PORT}/v1" \
  --api-model "/home/orwel/dev_genius/models/Qwen3.6-35B-A3B" \
  --tag "${BUILD_TAG}" \
  --items-per-behavior 12 \
  --min-pair-quality 4 \
  --max-workers 8 \
  --seed 17 \
  --timeout 1200 | tee -a "${LOG_PATH}"

DATASET_DIR="$(latest_dir "${OUT_ROOT}/${BUILD_TAG}_*")"
log "dataset_dir=${DATASET_DIR}"

log "stopping local qwen36 sglang before hidden-state work"
cleanup_server
sleep 10

wait_for_vram_headroom 12000 120

log "extracting/probing hidden states"
"${PY}" "${ROOT}/scripts/experiments/personality/probe_symphonic_voice_states.py" \
  --dataset-dir "${DATASET_DIR}" \
  --model-path "/home/orwel/dev_genius/models/Qwen3.6-35B-A3B" \
  --tag "${PROBE_TAG}" \
  --device-map auto \
  --c-grid "0.25,1.0" \
  --max-gpu-gib 72 \
  --max-cpu-gib 24 \
  --offload-folder "${ROOT}/tmp_offload_symphonic_v2" \
  --region-allowlist "think_mean,assistant_mean,response_mean,response_last16,prompt_last" \
  --layer-stride 1 | tee -a "${LOG_PATH}"

FEATURES_DIR="$(latest_dir "${OUT_ROOT}/${PROBE_TAG}_*")"
log "features_dir=${FEATURES_DIR}"

BEST_THINK_LAYER="$("${PY}" - <<PY
import json
from pathlib import Path
p=Path("${FEATURES_DIR}")/"searches.jsonl"
best=None
for line in p.read_text(encoding="utf-8").splitlines():
    row=json.loads(line)
    if row["region"] != "think_mean":
        continue
    score=(row["val_metrics"]["balanced_accuracy"], row["val_metrics"]["macro_f1"])
    if best is None or score > best[0]:
        best=(score, row["layer"])
print(best[1] if best is not None else 39)
PY
)"
log "best_think_layer=${BEST_THINK_LAYER}"

log "running axis analysis"
"${PY}" "${ROOT}/scripts/experiments/personality/analyze_symphonic_voice_axes.py" \
  --features-dir "${FEATURES_DIR}" \
  --anchor-manifest "${MANIFEST}" \
  --tag "${AXIS_TAG}" \
  --region-allowlist "think_mean,assistant_mean,response_mean,response_last16,prompt_last" \
  --layer-stride 1 \
  --common-region think_mean \
  --common-layer "${BEST_THINK_LAYER}" \
  --common-clf-c 0.25 \
  --ridge-alphas "0.1,1.0,10.0,100.0" \
  --patch-alphas "0.25,0.5,1.0" | tee -a "${LOG_PATH}"

AXIS_DIR="$(latest_dir "${OUT_ROOT}/${AXIS_TAG}_*")"
log "axis_dir=${AXIS_DIR}"

wait_for_vram_headroom 12000 120

log "running live late-think patching"
"${PY}" "${ROOT}/scripts/experiments/personality/live_patch_symphonic_voice.py" \
  --dataset-dir "${DATASET_DIR}" \
  --features-dir "${FEATURES_DIR}" \
  --axis-analysis-dir "${AXIS_DIR}" \
  --anchor-manifest "${MANIFEST}" \
  --model-path "/home/orwel/dev_genius/models/Qwen3.6-35B-A3B" \
  --tag "${PATCH_TAG}" \
  --common-region think_mean \
  --common-layer "${BEST_THINK_LAYER}" \
  --focus-axes "task_pragmatism,irony" \
  --top-k-per-axis 2 \
  --max-pairs 6 \
  --max-rows-per-pair 6 \
  --alphas "0.25,0.5,1.0" \
  --patch-after-tokens 48 \
  --patch-token-limit 96 \
  --max-new-tokens 768 \
  --dtype bfloat16 \
  --max-vram-frac 0.90 | tee -a "${LOG_PATH}"

PATCH_DIR="$(latest_dir "${OUT_ROOT}/${PATCH_TAG}_*")"
log "patch_dir=${PATCH_DIR}"

log "queue done"
