#!/usr/bin/env bash
set -u

ROOT="/home/orwel/dev_genius/experiments/Character Creation"
RUN_DIR="${1:-$ROOT/sweep_v4/jlens_remote_9b_real_20260709_070746}"
VENV_PATH="${VENV_PATH:-/home/orwel/dev_genius/venv}"
LENS_LAYERS="${LENS_LAYERS:-8,16,24}"
LENS_PROMPTS="${LENS_PROMPTS:-32}"
LENS_DIM_BATCH="${LENS_DIM_BATCH:-1}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-3072}"
JLORA_STEPS="${JLORA_STEPS:-120}"
JLORA_BATCH_SIZE="${JLORA_BATCH_SIZE:-1}"
JLORA_GRAD_ACCUM="${JLORA_GRAD_ACCUM:-8}"
SLEEP_SECONDS="${SLEEP_SECONDS:-300}"

cd "$ROOT" || exit 1
source "$VENV_PATH/bin/activate"
mkdir -p "$RUN_DIR/logs"

WATCH_LOG="$RUN_DIR/logs/followups_watch.log"
BASE_LOG="$RUN_DIR/logs/lens_base_gpu0.log"
JLORA_LOG="$RUN_DIR/logs/jlora_pilot_gpu1.log"
DELTA_LOG="$RUN_DIR/logs/delta_j.log"

INSTRUCT_A="$RUN_DIR/lens_qwen35_9b_instruct_a/jacobian_lens.pt"
INSTRUCT_B="$RUN_DIR/lens_qwen35_9b_instruct_b/jacobian_lens.pt"
BASE_DIR="$RUN_DIR/lens_qwen35_9b_base"
BASE_LENS="$BASE_DIR/jacobian_lens.pt"
JLORA_DIR="$RUN_DIR/jlora_pilot"
DELTA_DIR="$RUN_DIR/delta_qwen35_9b_instruct_vs_base"
PAIR_FILE="$ROOT/sweep_v4/qwen36_devbox_balanced_pairs_20260706_233429/pairs.jsonl"

printf '[%s] waiting for instruct lenses in %s\n' "$(date --iso-8601=seconds)" "$RUN_DIR" >> "$WATCH_LOG"
while [[ ! -f "$INSTRUCT_A" || ! -f "$INSTRUCT_B" ]]; do
  printf '[%s] waiting: a=%s b=%s\n' "$(date --iso-8601=seconds)" "$([[ -f "$INSTRUCT_A" ]] && echo yes || echo no)" "$([[ -f "$INSTRUCT_B" ]] && echo yes || echo no)" >> "$WATCH_LOG"
  if ! pgrep -f "fit_local_jlens.py .*lens_qwen35_9b_instruct" >/dev/null 2>&1; then
    printf '[%s] no instruct fit process remains and at least one lens is missing; aborting followups\n' "$(date --iso-8601=seconds)" >> "$WATCH_LOG"
    exit 2
  fi
  sleep "$SLEEP_SECONDS"
done

if [[ ! -f "$BASE_LENS" ]]; then
  printf '[%s] starting base lens on GPU0\n' "$(date --iso-8601=seconds)" >> "$WATCH_LOG"
  CUDA_VISIBLE_DEVICES=0 python scripts/experiments/connectome/fit_local_jlens.py \
    --model Qwen/Qwen3.5-9B-Base \
    --model-class causal-lm \
    --output-dir "$BASE_DIR" \
    --run-name remote_qwen35_9b_base \
    --source-layers "$LENS_LAYERS" \
    --n-prompts "$LENS_PROMPTS" \
    --skip-prompts "$((LENS_PROMPTS * 2))" \
    --max-seq-len 128 \
    --dim-batch "$LENS_DIM_BATCH" \
    --checkpoint-every 1 \
    --resume \
    --device cuda \
    > "$BASE_LOG" 2>&1 &
  base_pid=$!
else
  base_pid=""
  printf '[%s] base lens already exists\n' "$(date --iso-8601=seconds)" >> "$WATCH_LOG"
fi

if [[ ! -s "$JLORA_DIR/records.jsonl" ]]; then
  printf '[%s] starting J-ReFT pilot on GPU1\n' "$(date --iso-8601=seconds)" >> "$WATCH_LOG"
  CUDA_VISIBLE_DEVICES=1 python scripts/experiments/personality/jlora_pilot.py \
    --allow-real-model-run \
    --output-dir "$JLORA_DIR" \
    --model-name Qwen/Qwen3.5-9B \
    --local-instruct-lens "$INSTRUCT_A" \
    --train-file "$PAIR_FILE" \
    --layers "$LENS_LAYERS" \
    --j-rank 128 \
    --reft-rank 16 \
    --max-train-steps "$JLORA_STEPS" \
    --batch-size "$JLORA_BATCH_SIZE" \
    --grad-accum "$JLORA_GRAD_ACCUM" \
    --eval-limit 12 \
    --capability-limit 16 \
    --max-new-tokens "$MAX_NEW_TOKENS" \
    --device cuda \
    > "$JLORA_LOG" 2>&1 &
  jlora_pid=$!
else
  jlora_pid=""
  printf '[%s] J-ReFT records already exist\n' "$(date --iso-8601=seconds)" >> "$WATCH_LOG"
fi

if [[ -n "${base_pid:-}" ]]; then
  wait "$base_pid"
  printf '[%s] base lens exit=%s\n' "$(date --iso-8601=seconds)" "$?" >> "$WATCH_LOG"
fi

if [[ -f "$BASE_LENS" ]]; then
  printf '[%s] starting Delta-J comparison\n' "$(date --iso-8601=seconds)" >> "$WATCH_LOG"
  python scripts/experiments/connectome/jlens_delta_comparison.py \
    --allow-real-comparison \
    --output-dir "$DELTA_DIR" \
    --pair-label qwen35_9b_instruct_vs_base_unmodified \
    --noise-floor-a "$INSTRUCT_A" \
    --noise-floor-b "$INSTRUCT_B" \
    --model-a-lens "$INSTRUCT_A" \
    --model-b-lens "$BASE_LENS" \
    --model-a Qwen/Qwen3.5-9B \
    --model-b Qwen/Qwen3.5-9B-Base \
    --tokenizer-model Qwen/Qwen3.5-9B \
    --k-values 8,32,128,512 \
    > "$DELTA_LOG" 2>&1
  printf '[%s] Delta-J exit=%s\n' "$(date --iso-8601=seconds)" "$?" >> "$WATCH_LOG"
else
  printf '[%s] base lens missing after fit; skipping Delta-J\n' "$(date --iso-8601=seconds)" >> "$WATCH_LOG"
fi

if [[ -n "${jlora_pid:-}" ]]; then
  wait "$jlora_pid"
  printf '[%s] J-ReFT exit=%s\n' "$(date --iso-8601=seconds)" "$?" >> "$WATCH_LOG"
fi

printf '[%s] followups complete\n' "$(date --iso-8601=seconds)" >> "$WATCH_LOG"
