#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/orwel/dev_genius/experiments/Character Creation"
VENV="/home/orwel/dev_genius/venv"
PY="${VENV}/bin/python"
LOG_DIR="${ROOT}/logs"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_PATH="${LOG_DIR}/queue_dev_symphonic_remote_${STAMP}.log"
MANIFEST="${ROOT}/data/symphonic_voice_anchor_manifest_v2.json"

mkdir -p "${LOG_DIR}"
touch "${LOG_PATH}"

log() {
  echo "[$(date --iso-8601=seconds)] $*" | tee -a "${LOG_PATH}"
}

source "${VENV}/bin/activate"

run_endpoint_loop() {
  local name="$1"
  local base_url="$2"
  shift 2
  local seeds=("$@")
  for seed in "${seeds[@]}"; do
    log "start endpoint=${name} seed=${seed}"
    "${PY}" "${ROOT}/scripts/experiments/personality/build_symphonic_probe_dataset.py" \
      --source-dataset-dir "${ROOT}/sweep_v4/book_character_prefill_dataset_balanced_v6_reviewed_20260417_151017" \
      --anchor-manifest "${MANIFEST}" \
      --base-url "${base_url}" \
      --api-model "Qwen/Qwen3.5-9B" \
      --tag "symphonic_voice_probe_dataset_qwen35_${name}_v2_seed${seed}" \
      --items-per-behavior 12 \
      --min-pair-quality 4 \
      --max-workers 12 \
      --seed "${seed}" \
      --timeout 900 >> "${LOG_PATH}" 2>&1
    log "done endpoint=${name} seed=${seed}"
  done
}

log "dev remote queue start"

run_endpoint_loop "4090" "http://192.168.1.90:30002/v1" 17 29 &
PID_4090=$!
run_endpoint_loop "3090" "http://192.168.1.90:30001/v1" 41 53 &
PID_3090=$!

wait "${PID_4090}" "${PID_3090}"
log "dev remote queue done"
