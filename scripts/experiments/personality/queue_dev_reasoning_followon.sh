#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/orwel/dev_genius/experiments/Character Creation"
LAUNCHER="$ROOT/scripts/experiments/personality/launch_personality_control_reasoning_reasoningonly_v3.sh"
WEIRD_LAUNCHER="$ROOT/scripts/experiments/personality/launch_personality_weird_reasoning_probe_v1.sh"
SOURCE_OUT="$ROOT/sweep_v4/personality_control_reasoning_reasoningonly_v3"
WEIRD_OUT="$ROOT/sweep_v4/personality_weird_reasoning_probe_v1"
LOG="$ROOT/logs/queue_dev_reasoning_followon.log"

log() {
  echo "[$(date --iso-8601=seconds)] $*" | tee -a "$LOG"
}

log "waiting for $SOURCE_OUT/COMPLETE"
while [[ ! -f "$SOURCE_OUT/COMPLETE" ]]; do
  sleep 60
done

log "launching weird-domain probe before the next control seed"
OUT_REL="sweep_v4/personality_weird_reasoning_probe_v1" \
N_CHARACTERS=24 \
SEED=20260405 \
CONDITION_IDS="trace_explicit,think_explicit" \
CONCURRENCY_3090=16 \
CONCURRENCY_4090=16 \
MAX_NEW_TOKENS=720 \
"$WEIRD_LAUNCHER"

log "waiting for $WEIRD_OUT/COMPLETE"
while [[ ! -f "$WEIRD_OUT/COMPLETE" ]]; do
  sleep 30
done

log "launching follow-on reasoning-only control run with new seed"
OUT_REL="sweep_v4/personality_control_reasoning_reasoningonly_v3_seed43" \
SEED=43 \
N_SCAFFOLDS=128 \
CONCURRENCY_3090=16 \
CONCURRENCY_4090=16 \
MAX_NEW_TOKENS=512 \
"$LAUNCHER"

log "follow-on launch complete"
