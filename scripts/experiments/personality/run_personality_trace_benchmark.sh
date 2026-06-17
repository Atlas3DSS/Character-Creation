#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/orwel/dev_genius/experiments/Character Creation"
SCRIPT="$ROOT/scripts/experiments/personality/benchmark_personality_trace_eval.py"
DATASET_DIR="${DATASET_DIR:-$ROOT/sweep_v4/personality_meta_eval_trace_explicit_v1}"
BACKEND="${BACKEND:-openai}"
BENCHMARK_LABEL="${BENCHMARK_LABEL:-personality_trace_benchmark}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/sweep_v4/${BENCHMARK_LABEL}}"
SKIP_SCORING_TOKENIZER_DEFAULT="${SKIP_SCORING_TOKENIZER:-}"

if [[ "$BACKEND" == "openai" ]]; then
  PYTHON_BIN="${PYTHON_BIN:-/home/orwel/dev_genius/venv/bin/python}"
else
  PYTHON_BIN="${PYTHON_BIN:-/home/orwel/dev_genius/nanochat/.venv/bin/python}"
fi

mkdir -p "$OUTPUT_DIR"

COMMON_ARGS=(
  --dataset-dir "$DATASET_DIR"
  --output-dir "$OUTPUT_DIR"
  --benchmark-label "$BENCHMARK_LABEL"
  --backend "$BACKEND"
  --limit "${LIMIT:-0}"
  --shard "${SHARD:-0}"
  --n-shards "${N_SHARDS:-1}"
  --seed "${SEED:-20260404}"
  --max-new-tokens "${MAX_NEW_TOKENS:-960}"
  --temperature "${TEMPERATURE:-0.4}"
  --top-p "${TOP_P:-0.9}"
  --top-k "${TOP_K:-50}"
)

if [[ -z "$SKIP_SCORING_TOKENIZER_DEFAULT" && "$BACKEND" == "nanochat" ]]; then
  SKIP_SCORING_TOKENIZER_DEFAULT="1"
elif [[ -z "$SKIP_SCORING_TOKENIZER_DEFAULT" ]]; then
  SKIP_SCORING_TOKENIZER_DEFAULT="0"
fi

if [[ "$SKIP_SCORING_TOKENIZER_DEFAULT" == "1" ]]; then
  COMMON_ARGS+=(--skip-scoring-tokenizer)
else
  COMMON_ARGS+=(--scoring-tokenizer-model "${SCORING_TOKENIZER_MODEL:-Qwen/Qwen3.5-9B}")
fi

if [[ "$BACKEND" == "openai" ]]; then
  COMMON_ARGS+=(
    --base-url "${BASE_URL:?BASE_URL is required for BACKEND=openai}"
    --api-key "${API_KEY:-dummy}"
    --model "${MODEL:-Qwen/Qwen3.5-9B}"
    --concurrency "${CONCURRENCY:-16}"
    --timeout "${TIMEOUT:-240}"
    --retries "${RETRIES:-3}"
  )
else
  COMMON_ARGS+=(
    --nanochat-root "${NANOCHAT_ROOT:-/home/orwel/dev_genius/nanochat}"
    --nanochat-base-dir "${NANOCHAT_BASE_DIR:?NANOCHAT_BASE_DIR is required for BACKEND=nanochat}"
    --nanochat-source "${NANOCHAT_SOURCE:-sft}"
    --nanochat-model-tag "${NANOCHAT_MODEL_TAG:-meta_think_probe_d4}"
    --nanochat-step "${NANOCHAT_STEP:-0}"
    --device-type "${DEVICE_TYPE:-cuda}"
  )
fi

exec "$PYTHON_BIN" "$SCRIPT" "${COMMON_ARGS[@]}" "$@"
