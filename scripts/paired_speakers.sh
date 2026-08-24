#!/usr/bin/env bash
# Paired speakers: does giving the model TWO speakers' view of the same scene help?

set -u  # deliberately NOT -e: one diverging run should not kill the other

cd "$(dirname "$0")/.."
export PYTHONPATH=.

# Refuse to run two copies at once
exec 9>/tmp/paired_speakers.sh.lock
flock -n 9 || { echo "another copy of $(basename "$0") is already running; exiting" >&2; exit 1; }

TAG=v1
GROUP=paired-speakers-$TAG

# Held fixed across both runs, so the only difference is --pair-speakers.
COMMON="--split gastronorm --out-h 21 --out-w 30 --augment-mask 0 --augment-fft 0 \
        --loss-fn ce-pixel --decoder attn --max-duration 1000ep --wandb-group $GROUP"

# ***** P1: paired speakers (the thing being tested) *****
# n_channels 2 -> 4. Everything else matches B0 exactly.
python src/run.py $COMMON \
    --pair-speakers \
    --run-name paired-p1-pairs-$TAG

# ***** B0: baseline, one speaker per sample *****
# The current default. P1 is measured against this.
python src/run.py $COMMON \
    --run-name paired-b0-baseline-$TAG
