#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/orwel/dev_genius/experiments/Character Creation"
VENV="/home/orwel/dev_genius/venv"
MODEL_PATH="${MODEL_PATH:-/home/orwel/dev_genius/models/Qwen3.6-35B-A3B}"
PORT="${PORT:-30003}"
MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.82}"
MAX_RUNNING_REQUESTS="${MAX_RUNNING_REQUESTS:-64}"
ENABLE_RETURN_ROUTED_EXPERTS="${ENABLE_RETURN_ROUTED_EXPERTS:-0}"
SESSION="${SESSION:-sglang_qwen36_35b}"
LOG_PATH="${LOG_PATH:-${ROOT}/logs/sglang_qwen36_35b.log}"

source "${VENV}/bin/activate"

CUDNN_VERSION="$("${VENV}/bin/python" - <<'PY'
import torch
print(torch.backends.cudnn.version() or 0)
PY
)"
if [ "${CUDNN_VERSION}" -lt 91500 ]; then
  echo "Auto-remediation: upgrading nvidia-cudnn-cu12 because CuDNN=${CUDNN_VERSION}" >&2
  "${VENV}/bin/python" -m pip install --upgrade nvidia-cudnn-cu12==9.16.0.29
fi

mkdir -p "$(dirname "${LOG_PATH}")"
EXTRA_ARGS=()
if [ "${ENABLE_RETURN_ROUTED_EXPERTS}" = "1" ]; then
  EXTRA_ARGS+=(--enable-return-routed-experts)
fi
tmux kill-session -t "${SESSION}" 2>/dev/null || true
tmux new-session -d -s "${SESSION}" \
  "cd '${ROOT}' && source '${VENV}/bin/activate' && export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 PYTORCH_ALLOC_CONF=expandable_segments:True && python -m sglang.launch_server --model-path '${MODEL_PATH}' --trust-remote-code --dtype bfloat16 --host 127.0.0.1 --port '${PORT}' --attention-backend triton --mem-fraction-static '${MEM_FRACTION_STATIC}' --max-running-requests '${MAX_RUNNING_REQUESTS}' --context-length 262144 ${EXTRA_ARGS[*]} > '${LOG_PATH}' 2>&1"

echo "Started ${SESSION} on http://127.0.0.1:${PORT}/v1"
echo "Log: ${LOG_PATH}"
echo "Use model name/path: ${MODEL_PATH}"
