#!/usr/bin/env bash
# Copy an experiment directory to batman, skipping the large raw vibration
# captures (samples/<id>/vibration/01_raw_vibrations.npy, ~3GB each).
#
# Streams a tar over a single SSH connection (fast) instead of scp-per-file.
#
# Usage: ./sync_to_batman.sh EXPERIMENT_NAME
#   e.g. ./sync_to_batman.sh 31_07_2026_gastronorm_exp1
set -euo pipefail

LOCAL_BASE_DIR="D:/eturok"

if [ -z "${1:-}" ]; then
    echo "Usage: $0 EXPERIMENT_NAME" >&2
    exit 1
fi
EXP_NAME="$1"
HOST="batman"
# Resolved with $() so the remote shell expands ~ itself (avoids shipping a
# literal '~' inside single quotes to ssh, which some shells won't expand).
REMOTE_EXPERIMENTS_DIR="$(ssh "$HOST" 'echo ~/workspace/good-vibrations/experiments')"

SRC="${LOCAL_BASE_DIR}/${EXP_NAME}"
mkdir -p "$SRC"

REMOTE_DIR="${REMOTE_EXPERIMENTS_DIR}/${EXP_NAME}"

echo "Using ${HOST}:${REMOTE_DIR}"
ssh "$HOST" "mkdir -p '${REMOTE_DIR}'"

TOTAL=$(find "$SRC" \( -type f -o -type l \) ! -name '01_raw_vibrations.npy' | wc -l)
echo "Streaming ${TOTAL} files to ${HOST}:${REMOTE_DIR} (excluding 01_raw_vibrations.npy)..."
# Symlinks in samples/ (audio, images, recovered_audio.wav, ...) are stored
# with absolute paths rooted in the LOCAL filesystem (/d/eturok/... or
# /c/Users/eitanturok/...), which don't resolve on batman at all. Plain tar
# stores symlinks as symlinks (not the data they point to), so those links
# arrive dangling on batman. -h/--dereference makes tar follow symlinks and
# archive the actual file content instead, so batman gets real files.
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
