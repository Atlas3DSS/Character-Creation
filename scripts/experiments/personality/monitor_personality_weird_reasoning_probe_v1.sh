#!/usr/bin/env bash
set -euo pipefail

OUT_ABS="${1:?output dir required}"
S3090="${2:?3090 session required}"
S4090="${3:?4090 session required}"

STATUS_JSON="$OUT_ABS/progress_status.json"
DONE_FILE="$OUT_ABS/COMPLETE"

while true; do
  ts="$(date --iso-8601=seconds)"
  expected=0
  if [[ -f "$OUT_ABS/manifest.json" ]]; then
    expected="$(jq -r '.n_tasks_total // 0' "$OUT_ABS/manifest.json" 2>/dev/null || echo 0)"
  fi

  completed=0
  for f in "$OUT_ABS"/records_shard_*.jsonl; do
    [[ -f "$f" ]] || continue
    lines="$(wc -l < "$f" | tr -d ' ')"
    completed="$((completed + lines))"
  done

  shard0_alive=0
  shard1_alive=0
  tmux has-session -t "$S3090" 2>/dev/null && shard0_alive=1
  tmux has-session -t "$S4090" 2>/dev/null && shard1_alive=1

  cat >"$STATUS_JSON" <<EOF
{
  "timestamp": "$ts",
  "output_dir": "$OUT_ABS",
  "expected_total": $expected,
  "completed_total": $completed,
  "remaining_total": $(( expected > completed ? expected - completed : 0 )),
  "shard_3090_alive": $shard0_alive,
  "shard_4090_alive": $shard1_alive
}
EOF

  echo "[$ts] completed=$completed expected=$expected shard_3090_alive=$shard0_alive shard_4090_alive=$shard1_alive"

  if [[ "$expected" -gt 0 && "$completed" -ge "$expected" && "$shard0_alive" -eq 0 && "$shard1_alive" -eq 0 ]]; then
    touch "$DONE_FILE"
    echo "[$ts] complete"
    break
  fi

  if [[ "$shard0_alive" -eq 0 && "$shard1_alive" -eq 0 && "$completed" -lt "$expected" ]]; then
    echo "[$ts] workers exited before reaching expected_total"
    break
  fi

  sleep 30
done
