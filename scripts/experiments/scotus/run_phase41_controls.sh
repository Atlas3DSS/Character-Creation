#!/usr/bin/env bash
set -euo pipefail

cd "/home/orwel/dev_genius/experiments/Character Creation"
PYTHON_BIN="${PYTHON_BIN:-/home/orwel/dev_genius/qwen35_replay_venv/bin/python}"
MODES="${MODES:-excerpt_removed neutral_filler label_shuffle template_variant plain_prompt}"

COMMON_ARGS=(
  --pairs data/scotus/scotus_matched_pairs_v21.jsonl
  --pair Scalia_vs_Ginsburg
  --variant masked
  --positive-justice Ginsburg
  --model-path /home/orwel/dev_genius/models/Qwen3.6-27B-FP8
  --output-root sweep_v4
  --tag scotus_phase41
  --layers 0-20,24,28,32,36,40,44,48
  --c-grid 0.001,0.003,0.01,0.03,0.1,0.25,0.5,1.0,2.0,10.0
  --batch-size 1
  --max-length 1024
  --seed 17
  --classifier-solver lbfgs
  --classifier-max-iter 500
  --classifier-tol 0.001
  --stress-min-eval-per-label 5
)

for mode in ${MODES}; do
  echo "=== starting ${mode} $(date -Is) ==="
  "${PYTHON_BIN}" scripts/experiments/scotus/probe_scotus_style.py "${COMMON_ARGS[@]}" --diagnostic-mode "${mode}"
  echo "=== finished ${mode} $(date -Is) ==="
done
