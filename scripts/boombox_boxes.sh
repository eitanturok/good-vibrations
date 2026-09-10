#!/usr/bin/env bash
# hyb-h2-boombox-full-v1 architecture (full boombox: conv encoder + conv decoder, no attention)
# run across every box, at the standardized 32x32 output grid.
#
# The first arm (gastronorm) is the anchor: same arch + same --split gastronorm as
# hyb-h2-boombox-full-v1, only the output grid differs (32x32 vs the old 21x30). Check its
# eval/2-cubes hard-iou and localization land near hyb-h2 before trusting the rest.
#
# Splits (src/model/dataset.py): grids 1..N-1 of an object -> train, ~15% of the last grid's
# positions -> eval, the rest of that grid back to train. gastronorm keeps its existing split;
# plastic/wood/cardboard/shoebox use their per-box splits at --test-size 0.15. shoebox drops
# speaker 7 (14 stray captures).
#
# Bump TAG for a fresh sweep (new wandb group + new run names); override per-run without editing:
#   TAG=v2 ./scripts/boombox_boxes.sh 2>&1 | tee runs/boombox_boxes.log

set -u
cd "$(dirname "$0")/.."
export PYTHONPATH=.

TAG="${TAG:-v1}"
GROUP="boombox-boxes-$TAG"

COMMON="--model boombox --d-model 1024 --batch-size 256 \
        --augment-mask 0 --augment-fft 0 --loss-fn ce-pixel --max-duration 1000ep \
        --laser-dropout 0 --freq-dropout 0 --wandb-group $GROUP"

# gastronorm anchor -- no --test-size, keeps its default 0.2 (the value hyb-h2 used)
PYTHONPATH=. python src/run.py $COMMON \
    --data-dir experiments/31_07_2026_gastronorm_exp1 --split gastronorm \
    --run-name "bb-boxes-gastro-$TAG"

PYTHONPATH=. python src/run.py $COMMON --test-size 0.15 \
    --data-dir experiments/31_08_2026_green_plastic_two_laser_faces --split plastic \
    --run-name "bb-boxes-plastic-$TAG"

PYTHONPATH=. python src/run.py $COMMON --test-size 0.15 \
    --data-dir experiments/2026_09_06_wood_box --split wood \
    --run-name "bb-boxes-wood-$TAG"

PYTHONPATH=. python src/run.py $COMMON --test-size 0.15 \
    --data-dir experiments/2026_09_07_cardboard_box --split cardboard \
    --run-name "bb-boxes-cardboard-$TAG"

PYTHONPATH=. python src/run.py $COMMON --test-size 0.15 \
    --data-dir experiments/2026_09_08_shoebox --split shoebox \
    --run-name "bb-boxes-shoebox-$TAG"
