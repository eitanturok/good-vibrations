#!/usr/bin/env bash
# Re-run of the two boombox_object_count.sh runs that used gastronorm_one_cube, after fixing that
# split (src/model/dataset.py) to hold out whole position_ids instead of splitting by raw sample --
# the old split could put the same position's speakers on both sides of train/eval. Everything else
# (model, hparams, data dirs, laser-row crop) is identical to scripts/boombox_object_count.sh; only
# the split's train/eval membership changed, so these two are the only runs that need to be redone.
#
# Bump TAG for a fresh sweep (new wandb group + new run names): TAG=v2 ./scripts/boombox_object_count_1cube_rerun.sh

set -u
cd "$(dirname "$0")/.."
export PYTHONPATH=.

TAG="${TAG:-v1-position-split}"
GROUP="boombox-object-count-$TAG"

COMMON="--model boombox --d-model 1024 --batch-size 256 \
        --augment-mask 0 --augment-fft 0 --loss-fn ce-pixel --max-duration 5000ep \
        --laser-dropout 0 --freq-dropout 0 --wandb-group $GROUP"

GASTRO_DIR="experiments/31_07_2026_gastronorm_exp1"
PLASTIC_DIR="experiments/31_08_2026_green_plastic_two_laser_faces"
GASTRO_ROWS="--laser-rows 0,1,2,3,4,5,6,7"

# --- standalone: gastronorm 1-cube (now position-held-out) ---

python src/run.py $COMMON $GASTRO_ROWS --data-dir "$GASTRO_DIR" --split gastronorm_one_cube \
    --run-name "bb-count-gastro-1cube-$TAG"

# --- combined: gastronorm + plastic trained jointly, evaluated per-box ---

python src/run.py $COMMON $GASTRO_ROWS --data-dir "$GASTRO_DIR" --split gastronorm_one_cube \
    --data-dir-2 "$PLASTIC_DIR" --split-2 plastic_one_cube --test-size 0.15 \
    --run-name "bb-combined-1cube-$TAG"
