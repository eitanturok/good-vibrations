#!/usr/bin/env bash
# Follow-up to scripts/boombox_speaker_ablation.sh's bb-spk-1-v3 arm, which silently trained
# ZERO batches: --speakers 1 leaves only 342 train samples, and the main train loader uses
# drop_last=True (src/model/dataset.py:1343), so at --batch-size 1024 (> 342) every epoch's
# DataLoader was empty -- the run looped through all 1000 "epochs" with batch stuck at 0 (eval
# metrics frozen at random-init values: bce~0.685, iou~0.019) and exited cleanly with no error,
# so the sequential ablation script's `set -eu` never caught it.
#
# Fix: --batch-size 256, well under either speaker subset's ~340 train samples, so drop_last=True
# still yields real (if few) batches per epoch. Two runs, otherwise identical to
# bb-spk-baseline-v3's config (--model boombox --encoder single --d-model 512, gastronorm split,
# no augmentation, ce-pixel loss), just --max-duration 5000ep instead of 1000ep:
#   1. bb-spk-1-5k -- rerun of the failed bb-spk-1-v3 (speaker 1 only), now actually training
#   2. bb-spk-3-5k -- new arm, speaker 3 only
#
#   TAG=v2 ./scripts/boombox_speaker_5k.sh

set -eu
cd "$(dirname "$0")/.."
export PYTHONPATH=.

TAG="${TAG:-v1}"
GROUP="boombox-speaker-5k-$TAG"

COMMON="--model boombox --encoder single --d-model 512 --batch-size 256 \
        --data-dir experiments/31_07_2026_gastronorm_exp1 --split gastronorm \
        --augment-mask 0 --augment-fft 0 --loss-fn ce-pixel --max-duration 5000ep \
        --laser-dropout 0 --freq-dropout 0 --wandb-group $GROUP"

run() { PYTHONPATH=. python src/run.py $COMMON "$@"; }

run --speakers 1 --run-name "bb-spk-1-5k-$TAG"
run --speakers 3 --run-name "bb-spk-3-5k-$TAG"
