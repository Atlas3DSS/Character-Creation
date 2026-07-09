#!/usr/bin/env bash
set -u

ROOT="/home/orwel/dev_genius/experiments/Character Creation"
cd "$ROOT" || exit 1
VENV_PATH="${VENV_PATH:-/home/orwel/dev_genius/venv}"
source "$VENV_PATH/bin/activate"

STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="${RUN_DIR:-$ROOT/sweep_v4/jlens_three_brief_real_$STAMP}"
LOG_DIR="$RUN_DIR/logs"
mkdir -p "$LOG_DIR"

MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-3072}"
FINGERPRINT_PERSONAS="${FINGERPRINT_PERSONAS:-3}"
FINGERPRINT_PROMPTS="${FINGERPRINT_PROMPTS:-10}"
FINGERPRINT_LAYERS="${FINGERPRINT_LAYERS:-2}"
LENS_PROMPTS="${LENS_PROMPTS:-64}"
LENS_DIM_BATCH="${LENS_DIM_BATCH:-1}"
LENS_LAYERS="${LENS_LAYERS:-8,16,24}"
JLORA_STEPS="${JLORA_STEPS:-120}"
JLORA_BATCH_SIZE="${JLORA_BATCH_SIZE:-1}"
JLORA_GRAD_ACCUM="${JLORA_GRAD_ACCUM:-8}"

STATUS="$RUN_DIR/status.tsv"
COMMANDS="$RUN_DIR/commands.txt"
touch "$STATUS" "$COMMANDS"

log_status() {
  local name="$1"
  local status="$2"
  local log_path="$3"
  printf '%s\t%s\t%s\t%s\n' "$(date --iso-8601=seconds)" "$name" "$status" "$log_path" >> "$STATUS"
}

run_step() {
  local name="$1"
  shift
  local log_path="$LOG_DIR/$name.log"
  printf '\n[%s] START %s\n' "$(date --iso-8601=seconds)" "$name" | tee -a "$RUN_DIR/orchestrator.log"
  printf '%s\n' "$*" >> "$COMMANDS"
  "$@" > "$log_path" 2>&1
  local code=$?
  printf '[%s] END %s status=%s log=%s\n' "$(date --iso-8601=seconds)" "$name" "$code" "$log_path" | tee -a "$RUN_DIR/orchestrator.log"
  log_status "$name" "$code" "$log_path"
  return "$code"
}

echo "venv=$VENV_PATH" > "$RUN_DIR/preflight.txt"
python scripts/infra/jlens_three_brief_preflight.py >> "$RUN_DIR/preflight.txt" 2>&1

run_step persona_fingerprint \
  python scripts/experiments/personality/jlens_persona_fingerprint.py \
    --allow-real-model-run \
    --output-dir "$RUN_DIR/persona_fingerprint" \
    --pilot-personas "$FINGERPRINT_PERSONAS" \
    --pilot-prompts "$FINGERPRINT_PROMPTS" \
    --pilot-layers "$FINGERPRINT_LAYERS" \
    --k-values 8,32,128,512 \
    --max-new-tokens "$MAX_NEW_TOKENS" \
    --device cuda

run_step lens_qwen35_9b_instruct_a \
  python scripts/experiments/connectome/fit_local_jlens.py \
    --model Qwen/Qwen3.5-9B \
    --model-class causal-lm \
    --output-dir "$RUN_DIR/lens_qwen35_9b_instruct_a" \
    --run-name qwen35_9b_instruct_a \
    --source-layers "$LENS_LAYERS" \
    --n-prompts "$LENS_PROMPTS" \
    --skip-prompts 0 \
    --max-seq-len 128 \
    --dim-batch "$LENS_DIM_BATCH" \
    --checkpoint-every 1 \
    --resume \
    --device cuda

run_step lens_qwen35_9b_instruct_b \
  python scripts/experiments/connectome/fit_local_jlens.py \
    --model Qwen/Qwen3.5-9B \
    --model-class causal-lm \
    --output-dir "$RUN_DIR/lens_qwen35_9b_instruct_b" \
    --run-name qwen35_9b_instruct_b \
    --source-layers "$LENS_LAYERS" \
    --n-prompts "$LENS_PROMPTS" \
    --skip-prompts "$LENS_PROMPTS" \
    --max-seq-len 128 \
    --dim-batch "$LENS_DIM_BATCH" \
    --checkpoint-every 1 \
    --resume \
    --device cuda

run_step lens_qwen35_9b_base \
  python scripts/experiments/connectome/fit_local_jlens.py \
    --model Qwen/Qwen3.5-9B-Base \
    --model-class causal-lm \
    --output-dir "$RUN_DIR/lens_qwen35_9b_base" \
    --run-name qwen35_9b_base \
    --source-layers "$LENS_LAYERS" \
    --n-prompts "$LENS_PROMPTS" \
    --skip-prompts "$((LENS_PROMPTS * 2))" \
    --max-seq-len 128 \
    --dim-batch "$LENS_DIM_BATCH" \
    --checkpoint-every 1 \
    --resume \
    --device cuda

if [[ -f "$RUN_DIR/lens_qwen35_9b_instruct_a/jacobian_lens.pt" ]]; then
  run_step jlora_jreft_pilot \
    python scripts/experiments/personality/jlora_pilot.py \
      --allow-real-model-run \
      --output-dir "$RUN_DIR/jlora_pilot" \
      --model-name Qwen/Qwen3.5-9B \
      --local-instruct-lens "$RUN_DIR/lens_qwen35_9b_instruct_a/jacobian_lens.pt" \
      --train-file "$ROOT/sweep_v4/qwen36_devbox_balanced_pairs_20260706_233429/pairs.jsonl" \
      --eval-limit 12 \
      --capability-limit 16 \
      --layers "$LENS_LAYERS" \
      --j-rank 128 \
      --reft-rank 16 \
      --max-train-steps "$JLORA_STEPS" \
      --batch-size "$JLORA_BATCH_SIZE" \
      --grad-accum "$JLORA_GRAD_ACCUM" \
      --max-new-tokens "$MAX_NEW_TOKENS" \
      --device cuda
else
  log_status jlora_jreft_pilot "SKIPPED_NO_INSTRUCT_LENS" "$RUN_DIR/lens_qwen35_9b_instruct_a"
fi

if [[ -f "$RUN_DIR/lens_qwen35_9b_instruct_a/jacobian_lens.pt" && -f "$RUN_DIR/lens_qwen35_9b_instruct_b/jacobian_lens.pt" && -f "$RUN_DIR/lens_qwen35_9b_base/jacobian_lens.pt" ]]; then
  run_step delta_j_unmodified_qwen \
    python scripts/experiments/connectome/jlens_delta_comparison.py \
      --allow-real-comparison \
      --output-dir "$RUN_DIR/delta_qwen35_9b_instruct_vs_base" \
      --pair-label qwen35_9b_instruct_vs_base_unmodified \
      --noise-floor-a "$RUN_DIR/lens_qwen35_9b_instruct_a/jacobian_lens.pt" \
      --noise-floor-b "$RUN_DIR/lens_qwen35_9b_instruct_b/jacobian_lens.pt" \
      --model-a-lens "$RUN_DIR/lens_qwen35_9b_instruct_a/jacobian_lens.pt" \
      --model-b-lens "$RUN_DIR/lens_qwen35_9b_base/jacobian_lens.pt" \
      --model-a Qwen/Qwen3.5-9B \
      --model-b Qwen/Qwen3.5-9B-Base \
      --tokenizer-model Qwen/Qwen3.5-9B \
      --k-values 8,32,128,512
else
  log_status delta_j_unmodified_qwen "SKIPPED_MISSING_LENS" "$RUN_DIR"
fi

printf '[%s] COMPLETE run_dir=%s\n' "$(date --iso-8601=seconds)" "$RUN_DIR" | tee -a "$RUN_DIR/orchestrator.log"
