#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/orwel/dev_genius/experiments/Character Creation"
NANOCHAT_ROOT="/home/orwel/dev_genius/nanochat"
VENVDIR="/home/orwel/dev_genius/venv/bin/python"
GUARD="$ROOT/scripts/infra/run_with_vram_guard.py"
EXPORT_SCRIPT="$ROOT/scripts/experiments/personality/export_reasoning_control_to_nanochat_adapter.py"
BENCH_SCRIPT="$ROOT/scripts/experiments/personality/run_personality_trace_benchmark.sh"
CONTROL_DIR="$ROOT/sweep_v4/personality_control_reasoning_reasoningonly_v3"
PROBE_BASE="$ROOT/sweep_v4/nanochat_meta_think_compact_d8e3_v1"
LOG_DIR="$ROOT/logs"
OUT_PREFIX="${OUT_PREFIX:-reasoning_adapter_v1}"

TRACE_SRC_TAG="${TRACE_SRC_TAG:-meta_think_compact_trace_d8}"
LEAN_SRC_TAG="${LEAN_SRC_TAG:-meta_think_compact_lean_d8}"
TRACE_TAG="${TRACE_TAG:-meta_think_compact_trace_reasoning_adapter_d8_v1}"
LEAN_TAG="${LEAN_TAG:-meta_think_compact_lean_reasoning_adapter_d8_v1}"

TRACE_BENCH_128="${TRACE_BENCH_128:-trace_benchmark_${TRACE_TAG}_128}"
LEAN_BENCH_128="${LEAN_BENCH_128:-trace_benchmark_${LEAN_TAG}_128}"
TRACE_BENCH_FULL="${TRACE_BENCH_FULL:-trace_benchmark_${TRACE_TAG}_full}"

QUEUE_LOG="$LOG_DIR/blackwell_reasoning_adapter_queue.log"
TRACE_GUARD_LOG="$LOG_DIR/${TRACE_TAG}_guard.log"
LEAN_GUARD_LOG="$LOG_DIR/${LEAN_TAG}_guard.log"

mkdir -p "$LOG_DIR"

log() {
  echo "[$(date --iso-8601=seconds)] $*" | tee -a "$QUEUE_LOG"
}

wait_for_control_complete() {
  local waited=0
  while [[ ! -f "$CONTROL_DIR/COMPLETE" ]]; do
    if ! tmux has-session -t control_reasoning_v3_3090 2>/dev/null && ! tmux has-session -t control_reasoning_v3_4090 2>/dev/null; then
      if [[ -f "$CONTROL_DIR/manifest.json" ]]; then
        local expected completed
        expected="$(jq -r '.n_tasks_total // 0' "$CONTROL_DIR/manifest.json" 2>/dev/null || echo 0)"
        completed="$(find "$CONTROL_DIR" -maxdepth 1 -name 'records_shard_*.jsonl' -type f -print0 | xargs -0 cat 2>/dev/null | wc -l | tr -d ' ')"
        if [[ "${expected:-0}" -gt 0 && "${completed:-0}" -ge "${expected:-0}" ]]; then
          log "control workers exited after reaching expected record count; continuing without COMPLETE sentinel"
          return 0
        fi
      fi
      log "control workers exited before COMPLETE; aborting queue"
      return 1
    fi
    sleep 60
    waited=$((waited + 60))
    log "waiting for control completion (${waited}s elapsed)"
  done
  log "control dataset complete"
}

prepare_adapter_data() {
  log "exporting adapter corpus into $PROBE_BASE"
  "$VENVDIR" "$EXPORT_SCRIPT" \
    --control-dir "$CONTROL_DIR" \
    --nanochat-base-dir "$PROBE_BASE" \
    --output-prefix "$OUT_PREFIX"
}

clone_sft_tag() {
  local src_tag="$1"
  local dst_tag="$2"
  rm -rf "$PROBE_BASE/chatsft_checkpoints/$dst_tag"
  cp -a "$PROBE_BASE/chatsft_checkpoints/$src_tag" "$PROBE_BASE/chatsft_checkpoints/$dst_tag"
}

run_guarded_sft() {
  local tag="$1"
  local identity_train="$2"
  local identity_val="$3"
  local guard_log="$4"
  local run_name="$5"

  "$VENVDIR" "$GUARD" \
    --gpu-index 0 \
    --max-vram-fraction 0.70 \
    --poll-seconds 5 \
    --breach-polls 1 \
    --startup-grace-seconds 15 \
    --chdir "$NANOCHAT_ROOT" \
    --log-file "$guard_log" \
    -- /bin/bash -lc "
      source \"$NANOCHAT_ROOT/.venv/bin/activate\"
      export NANOCHAT_BASE_DIR=\"$PROBE_BASE\"
      export OMP_NUM_THREADS=1
      export WANDB_MODE=disabled
      python -m scripts.chat_sft \
        --run=\"$run_name\" \
        --device-type=cuda \
        --load-source=sft \
        --model-tag=\"$tag\" \
        --load-optimizer=0 \
        --max-seq-len=1280 \
        --device-batch-size=4 \
        --total-batch-size=10240 \
        --embedding-lr=0.01 \
        --unembedding-lr=0.0005 \
        --matrix-lr=0.001 \
        --init-lr-frac=0.5 \
        --warmup-ratio=0.05 \
        --warmdown-ratio=0.25 \
        --final-lr-frac=0.0 \
        --eval-every=50 \
        --eval-tokens=32768 \
        --chatcore-every=-1 \
        --num-iterations=250 \
        --identity-file=\"$identity_train\" \
        --val-identity-file=\"$identity_val\" \
        --identity-epochs=1 \
        --smoltalk-train=0 \
        --smoltalk-val=0 \
        --mmlu-epochs=0 \
        --gsm8k-epochs=0 \
        --simple-spelling-size=0 \
        --spellingbee-size=0 \
        --custom-only
    "
}

run_benchmark() {
  local tag="$1"
  local label="$2"
  local limit="$3"
  BACKEND=nanochat \
  NANOCHAT_BASE_DIR="$PROBE_BASE" \
  NANOCHAT_SOURCE=sft \
  NANOCHAT_MODEL_TAG="$tag" \
  BENCHMARK_LABEL="$label" \
  LIMIT="$limit" \
  TEMPERATURE=0.0 \
  TOP_P=1.0 \
  TOP_K=1 \
  "$BENCH_SCRIPT"
}

main() {
  log "queue start"
  wait_for_control_complete
  prepare_adapter_data

  log "cloning source SFT tags"
  clone_sft_tag "$TRACE_SRC_TAG" "$TRACE_TAG"
  clone_sft_tag "$LEAN_SRC_TAG" "$LEAN_TAG"

  log "starting trace adapter SFT"
  run_guarded_sft \
    "$TRACE_TAG" \
    "$PROBE_BASE/identity_conversations_trace_${OUT_PREFIX}_train.jsonl" \
    "$PROBE_BASE/identity_conversations_trace_${OUT_PREFIX}_val.jsonl" \
    "$TRACE_GUARD_LOG" \
    "$TRACE_TAG"

  log "starting lean adapter SFT"
  run_guarded_sft \
    "$LEAN_TAG" \
    "$PROBE_BASE/identity_conversations_lean_${OUT_PREFIX}_train.jsonl" \
    "$PROBE_BASE/identity_conversations_lean_${OUT_PREFIX}_val.jsonl" \
    "$LEAN_GUARD_LOG" \
    "$LEAN_TAG"

  log "running 128-row trace benchmark"
  run_benchmark "$TRACE_TAG" "$TRACE_BENCH_128" 128

  log "running 128-row lean benchmark"
  run_benchmark "$LEAN_TAG" "$LEAN_BENCH_128" 128

  local trace_summary="$ROOT/sweep_v4/$TRACE_BENCH_128/summary.json"
  if [[ -f "$trace_summary" ]]; then
    local trace_cov trace_fmt
    trace_cov="$(jq -r '.overall.reasoning_coverage // 0' "$trace_summary")"
    trace_fmt="$(jq -r '.overall.format_adherence_rate // 0' "$trace_summary")"
    log "trace 128 benchmark coverage=$trace_cov format=$trace_fmt"
    if awk "BEGIN { exit !($trace_cov >= 0.25 && $trace_fmt >= 0.50) }"; then
      log "trace adapter passed promotion threshold; running full benchmark"
      run_benchmark "$TRACE_TAG" "$TRACE_BENCH_FULL" 0
    else
      log "trace adapter below promotion threshold; skipping full benchmark"
    fi
  fi

  log "queue complete"
}

main "$@"
