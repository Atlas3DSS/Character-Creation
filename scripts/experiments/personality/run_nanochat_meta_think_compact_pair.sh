#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/orwel/dev_genius/experiments/Character Creation"
NANOCHAT_ROOT="/home/orwel/dev_genius/nanochat"
PROBE_BASE="${PROBE_BASE:-$ROOT/sweep_v4/nanochat_meta_think_compact_v1}"
SWEEP_DIR="${SWEEP_DIR:-$ROOT/sweep_v3/ws_openai_15k_sampled25m_repaired_responseonly}"
RUN_NAME="${RUN_NAME:-dummy}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-1280}"
BASE_TAG="${BASE_TAG:-meta_think_compact_base_d4}"
TRACE_TAG="${TRACE_TAG:-meta_think_compact_trace_d4}"
LEAN_TAG="${LEAN_TAG:-meta_think_compact_lean_d4}"

mkdir -p "$PROBE_BASE"
export NANOCHAT_BASE_DIR="$PROBE_BASE"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export WANDB_MODE="${WANDB_MODE:-disabled}"

python_bin_export="${PYTHON_BIN_EXPORT:-/home/orwel/dev_genius/venv/bin/python}"
"$python_bin_export" "$ROOT/scripts/experiments/personality/export_nanochat_meta_think_compact.py" \
  --sweep-dir "$SWEEP_DIR" \
  --output-dir "$PROBE_BASE"

cd "$NANOCHAT_ROOT"
source .venv/bin/activate

python -m scripts.tok_train \
  --max-chars "${TOK_MAX_CHARS:-25000000}" \
  --doc-cap "${TOK_DOC_CAP:-4096}" \
  --vocab-size "${TOK_VOCAB_SIZE:-8192}"

python -m scripts.base_train \
  --depth="${DEPTH:-4}" \
  --aspect-ratio="${ASPECT_RATIO:-64}" \
  --head-dim="${HEAD_DIM:-64}" \
  --window-pattern=L \
  --max-seq-len="$MAX_SEQ_LEN" \
  --device-batch-size="${BASE_DEVICE_BATCH:-4}" \
  --total-batch-size="${BASE_TOTAL_BATCH:-10240}" \
  --eval-every="${BASE_EVAL_EVERY:-100}" \
  --eval-tokens="${BASE_EVAL_TOKENS:-65536}" \
  --core-metric-every=-1 \
  --sample-every=-1 \
  --save-every="${BASE_SAVE_EVERY:-200}" \
  --num-iterations="${BASE_ITERS:-1000}" \
  --run="$RUN_NAME" \
  --model-tag="$BASE_TAG"

rm -rf "$PROBE_BASE/base_checkpoints/$TRACE_TAG" "$PROBE_BASE/base_checkpoints/$LEAN_TAG"
cp -a "$PROBE_BASE/base_checkpoints/$BASE_TAG" "$PROBE_BASE/base_checkpoints/$TRACE_TAG"
cp -a "$PROBE_BASE/base_checkpoints/$BASE_TAG" "$PROBE_BASE/base_checkpoints/$LEAN_TAG"
rm -rf "$PROBE_BASE/chatsft_checkpoints/$TRACE_TAG" "$PROBE_BASE/chatsft_checkpoints/$LEAN_TAG"

python -m scripts.chat_sft \
  --run="$RUN_NAME" \
  --device-type="${DEVICE_TYPE:-cuda}" \
  --model-tag="$TRACE_TAG" \
  --load-optimizer=0 \
  --max-seq-len="$MAX_SEQ_LEN" \
  --device-batch-size="${SFT_DEVICE_BATCH:-4}" \
  --total-batch-size="${SFT_TOTAL_BATCH:-10240}" \
  --embedding-lr="${SFT_EMBEDDING_LR:-0.03}" \
  --unembedding-lr="${SFT_UNEMBEDDING_LR:-0.001}" \
  --matrix-lr="${SFT_MATRIX_LR:-0.003}" \
  --init-lr-frac="${SFT_INIT_LR_FRAC:-0.25}" \
  --warmup-ratio="${SFT_WARMUP_RATIO:-0.10}" \
  --warmdown-ratio="${SFT_WARMDOWN_RATIO:-0.25}" \
  --final-lr-frac="${SFT_FINAL_LR_FRAC:-0.00}" \
  --eval-every="${SFT_EVAL_EVERY:-50}" \
  --eval-tokens="${SFT_EVAL_TOKENS:-32768}" \
  --chatcore-every=-1 \
  --num-iterations="${TRACE_SFT_ITERS:-300}" \
  --identity-file="$PROBE_BASE/identity_conversations_trace_train.jsonl" \
  --val-identity-file="$PROBE_BASE/identity_conversations_trace_val.jsonl" \
  --identity-epochs="${SFT_IDENTITY_EPOCHS:-1}" \
  --smoltalk-train=0 \
  --smoltalk-val=0 \
  --mmlu-epochs=0 \
  --gsm8k-epochs=0 \
  --simple-spelling-size=0 \
  --spellingbee-size=0 \
  --custom-only

python -m scripts.chat_sft \
  --run="$RUN_NAME" \
  --device-type="${DEVICE_TYPE:-cuda}" \
  --model-tag="$LEAN_TAG" \
  --load-optimizer=0 \
  --max-seq-len="$MAX_SEQ_LEN" \
  --device-batch-size="${SFT_DEVICE_BATCH:-4}" \
  --total-batch-size="${SFT_TOTAL_BATCH:-10240}" \
  --embedding-lr="${SFT_EMBEDDING_LR:-0.03}" \
  --unembedding-lr="${SFT_UNEMBEDDING_LR:-0.001}" \
  --matrix-lr="${SFT_MATRIX_LR:-0.003}" \
  --init-lr-frac="${SFT_INIT_LR_FRAC:-0.25}" \
  --warmup-ratio="${SFT_WARMUP_RATIO:-0.10}" \
  --warmdown-ratio="${SFT_WARMDOWN_RATIO:-0.25}" \
  --final-lr-frac="${SFT_FINAL_LR_FRAC:-0.00}" \
  --eval-every="${SFT_EVAL_EVERY:-50}" \
  --eval-tokens="${SFT_EVAL_TOKENS:-32768}" \
  --chatcore-every=-1 \
  --num-iterations="${LEAN_SFT_ITERS:-300}" \
  --identity-file="$PROBE_BASE/identity_conversations_lean_train.jsonl" \
  --val-identity-file="$PROBE_BASE/identity_conversations_lean_val.jsonl" \
  --identity-epochs="${SFT_IDENTITY_EPOCHS:-1}" \
  --smoltalk-train=0 \
  --smoltalk-val=0 \
  --mmlu-epochs=0 \
  --gsm8k-epochs=0 \
  --simple-spelling-size=0 \
  --spellingbee-size=0 \
  --custom-only
