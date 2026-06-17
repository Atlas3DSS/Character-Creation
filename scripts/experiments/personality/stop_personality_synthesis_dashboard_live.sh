#!/usr/bin/env bash
set -euo pipefail

tmux kill-session -t personality_synthesis_watch 2>/dev/null || true
tmux kill-session -t personality_synthesis_web 2>/dev/null || true
echo "Stopped personality synthesis dashboard sessions."
