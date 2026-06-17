#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
DATASET_REL="${DATASET_REL:-sweep_v4/qwen35_teacher_lora_pair_v1}"
RUN_TAG="${RUN_TAG:-qwen35_pair_lora_v1}"
LAUNCHER="$ROOT/scripts/experiments/personality/launch_qwen35_pair_lora_remote.sh"
MANIFEST="$ROOT/$DATASET_REL/manifest.json"
TRACE_TRAIN="$ROOT/$DATASET_REL/student_trace_train.jsonl"
PERSONA_TRAIN="$ROOT/$DATASET_REL/student_personality_train.jsonl"

echo "[QUEUE] waiting for teacher dataset at $DATASET_REL"
while true; do
  if [[ -f "$MANIFEST" && -s "$TRACE_TRAIN" && -s "$PERSONA_TRAIN" ]]; then
    break
  fi
  sleep 30
done

echo "[QUEUE] teacher dataset ready, launching remote pair run"
DATASET_REL="$DATASET_REL" RUN_TAG="$RUN_TAG" "$LAUNCHER"
echo "[QUEUE] remote pair launch submitted"
