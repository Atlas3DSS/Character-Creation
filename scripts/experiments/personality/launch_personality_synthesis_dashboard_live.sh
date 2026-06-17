#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8765}"
REFRESH_SEC="${REFRESH_SEC:-45}"
REBUILD_SEC="${REBUILD_SEC:-60}"
OPEN_BROWSER="${OPEN_BROWSER:-1}"

WATCH_SESSION="personality_synthesis_watch"
WEB_SESSION="personality_synthesis_web"
LOG_DIR="$ROOT/logs"
WATCH_LOG="$LOG_DIR/personality_synthesis_dashboard_watch.log"
WEB_LOG="$LOG_DIR/personality_synthesis_dashboard_web.log"
OUTPUT_HTML="$ROOT/reports/personality_synthesis_visualizer_live.html"
OUTPUT_JSON="$ROOT/reports/personality_synthesis_visualizer_live.json"
BUILDER="$ROOT/scripts/experiments/personality/build_personality_synthesis_visualizer.py"

mkdir -p "$LOG_DIR"

if ! command -v tmux >/dev/null 2>&1; then
  echo "tmux is required for the live dashboard launcher." >&2
  exit 1
fi

"$PYTHON_BIN" "$BUILDER" \
  --output-html "$OUTPUT_HTML" \
  --output-json "$OUTPUT_JSON" \
  --auto-refresh-sec "$REFRESH_SEC" >/dev/null

tmux kill-session -t "$WATCH_SESSION" 2>/dev/null || true
tmux kill-session -t "$WEB_SESSION" 2>/dev/null || true

tmux new-session -d -s "$WATCH_SESSION" \
  "cd \"$ROOT\" && while true; do echo \"[\$(date '+%Y-%m-%d %H:%M:%S %Z')] rebuild\"; \"$PYTHON_BIN\" \"$BUILDER\" --output-html \"$OUTPUT_HTML\" --output-json \"$OUTPUT_JSON\" --auto-refresh-sec \"$REFRESH_SEC\"; sleep \"$REBUILD_SEC\"; done >> \"$WATCH_LOG\" 2>&1"

tmux new-session -d -s "$WEB_SESSION" \
  "cd \"$ROOT/reports\" && \"$PYTHON_BIN\" -m http.server \"$PORT\" --bind \"$HOST\" >> \"$WEB_LOG\" 2>&1"

DISPLAY_HOST="$HOST"
if [[ "$DISPLAY_HOST" == "0.0.0.0" ]]; then
  DISPLAY_HOST="127.0.0.1"
fi
URL="http://$DISPLAY_HOST:$PORT/$(basename "$OUTPUT_HTML")"

if [[ "$OPEN_BROWSER" == "1" ]]; then
  if command -v xdg-open >/dev/null 2>&1; then
    xdg-open "$URL" >/dev/null 2>&1 || true
  elif command -v open >/dev/null 2>&1; then
    open "$URL" >/dev/null 2>&1 || true
  fi
fi

cat <<EOF
Live dashboard is running.
URL: $URL
Watch log: $WATCH_LOG
Web log: $WEB_LOG
Stop: $ROOT/scripts/experiments/personality/stop_personality_synthesis_dashboard_live.sh
EOF
