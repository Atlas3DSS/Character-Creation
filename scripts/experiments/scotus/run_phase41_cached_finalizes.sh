#!/usr/bin/env bash
set -euo pipefail

cd "/home/orwel/dev_genius/experiments/Character Creation"

PYTHON_BIN="${PYTHON_BIN:-/home/orwel/dev_genius/qwen35_replay_venv/bin/python}"
C_GRID="${C_GRID:-0.001,0.003,0.01,0.03,0.1,0.25,0.5,1.0,2.0,10.0}"
COMMON_ARGS=(
  --c-grid "${C_GRID}"
  --classifier-solver lbfgs
  --classifier-max-iter 500
  --classifier-tol 0.001
  --stress-min-eval-per-label 5
)

NORMAL_FEATURES="${NORMAL_FEATURES:-sweep_v4/scotus_phase41_normal_20260425_102519}"
NEUTRAL_FEATURES="${NEUTRAL_FEATURES:-sweep_v4/scotus_phase41_neutral_filler_20260425_115752}"

echo "=== finalizing neutral_filler $(date -Is) ==="
"${PYTHON_BIN}" scripts/experiments/scotus/probe_scotus_style.py \
  --features-dir "${NEUTRAL_FEATURES}" \
  --diagnostic-mode neutral_filler \
  "${COMMON_ARGS[@]}"

echo "=== finalizing label_shuffle from normal cache $(date -Is) ==="
"${PYTHON_BIN}" scripts/experiments/scotus/probe_scotus_style.py \
  --features-dir "${NORMAL_FEATURES}" \
  --diagnostic-mode label_shuffle \
  --output-root sweep_v4 \
  --tag scotus_phase41 \
  "${COMMON_ARGS[@]}"

echo "=== cached finalizes complete $(date -Is) ==="
