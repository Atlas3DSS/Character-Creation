#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/orwel/dev_genius/experiments/Character Creation"
NANOCHAT_ROOT="/home/orwel/dev_genius/nanochat"
PROBE_BASE="${PROBE_BASE:-$ROOT/sweep_v4/nanochat_meta_think_probe}"
SWEEP_DIR="${SWEEP_DIR:-$ROOT/sweep_v3/ws_openai_15k_sampled25m_repaired_responseonly}"
RUN_NAME="${RUN_NAME:-dummy}"

# Tiny defaults for a feasibility probe, not a serious model.
TOK_MAX_CHARS="${TOK_MAX_CHARS:-25000000}"
TOK_DOC_CAP="${TOK_DOC_CAP:-4000}"
TOK_VOCAB_SIZE="${TOK_VOCAB_SIZE:-8192}"
DEPTH="${DEPTH:-4}"
HEAD_DIM="${HEAD_DIM:-64}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-512}"
BASE_DEVICE_BATCH="${BASE_DEVICE_BATCH:-8}"
BASE_TOTAL_BATCH="${BASE_TOTAL_BATCH:-8192}"
BASE_ITERS="${BASE_ITERS:-1500}"
SFT_DEVICE_BATCH="${SFT_DEVICE_BATCH:-8}"
SFT_TOTAL_BATCH="${SFT_TOTAL_BATCH:-8192}"
SFT_ITERS="${SFT_ITERS:-1200}"

mkdir -p "$PROBE_BASE"
export NANOCHAT_BASE_DIR="$PROBE_BASE"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

python3 "$ROOT/scripts/experiments/personality/export_nanochat_meta_think_probe.py" \
  --sweep-dir "$SWEEP_DIR" \
  --output-dir "$PROBE_BASE"

cd "$NANOCHAT_ROOT"

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required but not found"
  exit 1
fi

if [[ ! -d ".venv" ]]; then
  uv venv
fi

uv sync --extra gpu
source .venv/bin/activate

python -m scripts.tok_train \
  --max-chars "$TOK_MAX_CHARS" \
  --doc-cap "$TOK_DOC_CAP" \
  --vocab-size "$TOK_VOCAB_SIZE"

python -m scripts.base_train \
  --depth="$DEPTH" \
  --head-dim="$HEAD_DIM" \
  --window-pattern=L \
  --max-seq-len="$MAX_SEQ_LEN" \
  --device-batch-size="$BASE_DEVICE_BATCH" \
  --total-batch-size="$BASE_TOTAL_BATCH" \
  --eval-every=100 \
  --eval-tokens=65536 \
  --core-metric-every=-1 \
  --sample-every=-1 \
  --save-every=200 \
  --num-iterations="$BASE_ITERS" \
  --run="$RUN_NAME" \
  --model-tag="meta_think_probe_d${DEPTH}"

python -m scripts.chat_sft \
  --run="$RUN_NAME" \
  --model-tag="meta_think_probe_d${DEPTH}" \
  --max-seq-len="$MAX_SEQ_LEN" \
  --device-batch-size="$SFT_DEVICE_BATCH" \
  --total-batch-size="$SFT_TOTAL_BATCH" \
  --eval-every=100 \
  --eval-tokens=65536 \
  --chatcore-every=-1 \
  --num-iterations="$SFT_ITERS" \
  --identity-file="$PROBE_BASE/identity_conversations_trace_train.jsonl" \
  --val-identity-file="$PROBE_BASE/identity_conversations_trace_val.jsonl" \
  --identity-epochs=4 \
  --smoltalk-train=0 \
  --smoltalk-val=0 \
  --mmlu-epochs=0 \
  --gsm8k-epochs=0 \
  --simple-spelling-size=0 \
  --spellingbee-size=0 \
  --custom-only
