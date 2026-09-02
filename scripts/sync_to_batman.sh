#!/usr/bin/env bash
# Copy an experiment directory from ironman to batman, dereferencing
# symlinks (which point at local paths that don't resolve on batman) and
# skipping the large raw vibration captures (.../vibration/01_raw_vibrations.npy).
#
# Streams a tar over a single SSH connection (fast) instead of scp-per-file.
# Run this ON IRONMAN, e.g. from Git Bash:
#   ./sync_to_batman.sh D:/eturok/31_08_2026_green_plastic_two_laser_faces
set -euo pipefail

if [ -z "${1:-}" ]; then
    echo "Usage: $0 SRC_DIR" >&2
    exit 1
fi
SRC="$1"
NAME="$(basename "$SRC")"
HOST="batman"
# Resolved with $() so the remote shell expands ~ itself (avoids shipping a
# literal '~' inside single quotes to ssh, which some shells won't expand).
REMOTE_DIR="$(ssh "$HOST" 'echo ~/workspace/good-vibrations/experiments')/${NAME}"

echo "Using ${HOST}:${REMOTE_DIR}"
ssh "$HOST" "mkdir -p '${REMOTE_DIR}'"

TOTAL=$(find "$SRC" \( -type f -o -type l \) ! -name '01_raw_vibrations.npy' | wc -l)
echo "Streaming ${TOTAL} files to ${HOST}:${REMOTE_DIR} (excluding 01_raw_vibrations.npy)..."
# Plain tar stores symlinks as symlinks (not the data they point to), so
# links pointing at local-only paths arrive dangling on batman. -h makes
# tar follow symlinks and archive the actual file content instead.
# -v prints each archived filename to stderr (archive bytes stay on stdout,
# so this is safe to pipe), giving live per-file progress.
tar -C "$SRC" -h --exclude='01_raw_vibrations.npy' -cvf - . \
    | ssh "$HOST" "tar -C '${REMOTE_DIR}' -xf -"
echo "Sent ${TOTAL} files."

echo "Done. Verifying..."
LOCAL_COUNT=$(find "$SRC" \( -type f -o -type l \) ! -name '01_raw_vibrations.npy' | wc -l)
REMOTE_COUNT=$(ssh "$HOST" "find '${REMOTE_DIR}' \( -type f -o -type l \) | wc -l")
echo "Local (excluding raw vibrations): ${LOCAL_COUNT} files/symlinks"
echo "Remote: ${REMOTE_COUNT} files/symlinks"
if [ "$LOCAL_COUNT" -eq "$REMOTE_COUNT" ]; then
    echo "OK: file counts match."
else
    echo "WARNING: file counts differ." >&2
fi
