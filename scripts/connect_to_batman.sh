#!/usr/bin/env bash
# Run on the LAPTOP: hold an SSH tunnel to batman's viz2 dashboard in tmux, then open localhost.
#   ./connect_to_batman.sh        tunnel on 8504, attach
#   ./connect_to_batman.sh 8504 -d   set up in the background
set -u
HOST="${VIZ_HOST:-batman}"
PORT="${1:-8504}"
SESSION="batman-viz-$PORT"

if ! tmux has-session -t "$SESSION" 2>/dev/null; then
  tmux new-session -d -s "$SESSION" \
    "while true; do
       ssh -N -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 -o ServerAliveCountMax=3 \
           -L $PORT:localhost:$PORT $HOST
       echo '[tunnel] dropped, retrying in 5s'; sleep 5
     done"
  echo "[tunnel] $SESSION -> $HOST:$PORT"
fi
echo "open http://localhost:$PORT"
[ "${2:-}" = "-d" ] || tmux attach -t "$SESSION"
