#!/usr/bin/env bash
# Speaker-diversity ablation: the regular conv-conv boombox model (conv encoder + conv decoder,
# --model boombox --encoder single, ce-pixel loss), on the full gastronorm split (all object
# counts: empty/1/2/3-cube, standardized 32x32 output grid). Answers: how much does restricting
# the speaker set at train time hurt eval quality, and does WHICH speakers you keep matter more
# than HOW MANY (a fixed subset vs. a random subset resampled per position)?
#
# Runs sequentially (one GPU) through 5 configs:
#   1. baseline  -- all 8 speakers
#   2. spk-1357  -- speakers 1,3,5,7 (a fixed, spread-out subset)
#   3. spk-1     -- speaker 1 only (single-speaker floor)
#   4. spk-rand4 -- 4 speakers, resampled independently at random PER POSITION (see
#                   model.dataset._sample_speakers_per_position) -- same speaker COUNT per
#                   position as spk-1357/spk-3456, but not the same fixed identities everywhere,
#                   so every position is still covered by *some* 4 speakers, just not always the
#                   same 4 globally
#   5. spk-3456  -- speakers 3,4,5,6 (a second fixed subset, for comparison against 1357)
#
# --speakers/--speaker-sample-per-position both restrict TRAIN and EVAL consistently (see
# gastronorm()'s docstring in model/dataset.py) -- runs with fewer speakers also get a
# proportionally smaller eval set, so compare metrics with that in mind, not just loss curves.
#
# Base config matches scripts/boombox_boxes.sh's gastronorm arm (d_model=1024, ce-pixel loss, no
# augmentation, --resize bilinear is the default -- see project_boombox_32x32_resize_regression
# for why 'conv' resize must never be used at 32x32).
#
#   TAG=v2 ./scripts/boombox_speaker_ablation.sh

set -eu
cd "$(dirname "$0")/.."
export PYTHONPATH=.

TAG="${TAG:-v3}"
GROUP="boombox-speaker-ablation-$TAG"

COMMON="--model boombox --encoder single --d-model 512 --batch-size 1024 \
        --data-dir experiments/31_07_2026_gastronorm_exp1 --split gastronorm \
        --augment-mask 0 --augment-fft 0 --loss-fn ce-pixel --max-duration 1000ep \
        --laser-dropout 0 --freq-dropout 0 --wandb-group $GROUP"

run() { PYTHONPATH=. python src/run.py $COMMON "$@"; }

run --run-name "bb-spk-baseline-$TAG"
run --speakers 1,3,5,7               --run-name "bb-spk-1357-$TAG"
run --speakers 1                     --run-name "bb-spk-1-$TAG"
run --speaker-sample-per-position 4  --run-name "bb-spk-rand4-$TAG"
run --speakers 3,4,5,6               --run-name "bb-spk-3456-$TAG"
