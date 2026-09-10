#!/usr/bin/env bash
# shoebox: one boombox model per object -- cube, cylinder, mug, ring.
# Same arch as scripts/boombox_boxes.sh (full boombox, conv enc + conv dec, 32x32 out).
#
# Each object has its own split (src/model/dataset.py: shoebox_<obj>):
#   - speaker 7 (14 stray captures) dropped -- every split is speakers [1, 3, 5]
#   - empty-box always in train
#   - ~85/15 train/eval, and every eval sample is a whole position_id of ONE grid
#     layout (<obj>-grid4); the rest of that grid + grids 1..3 go to train, so no
#     position is seen in both train and eval.
#
# Bump TAG for a fresh sweep (new wandb group + run names); override without editing:
#   TAG=v2 ./scripts/shoebox_objects.sh 2>&1 | tee runs/shoebox_objects.log

set -u
cd "$(dirname "$0")/.."
export PYTHONPATH=.

TAG="${TAG:-v1}"
GROUP="shoebox-objects-$TAG"

COMMON="--model boombox --d-model 1024 --batch-size 256 \
        --augment-mask 0 --augment-fft 0 --loss-fn ce-pixel --max-duration 1000ep \
        --laser-dropout 0 --freq-dropout 0 --test-size 0.15 \
        --data-dir experiments/2026_09_08_shoebox --wandb-group $GROUP"

PYTHONPATH=. python src/run.py $COMMON --split shoebox_cube     --run-name "shoebox-cube-$TAG"
PYTHONPATH=. python src/run.py $COMMON --split shoebox_cylinder --run-name "shoebox-cylinder-$TAG"
PYTHONPATH=. python src/run.py $COMMON --split shoebox_mug      --run-name "shoebox-mug-$TAG"
PYTHONPATH=. python src/run.py $COMMON --split shoebox_ring     --run-name "shoebox-ring-$TAG"
