#!/usr/bin/env bash
# hyb-h2-boombox-full-v1 architecture (full boombox: conv encoder + conv decoder, no attention),
# same COMMON args as scripts/boombox_boxes.sh (bb-boxes-*-v2), --max-duration 5000ep, across a
# 3-way object-count split (1-cube only / 2-cubes only / everything) x {gastronorm alone, plastic
# alone, gastronorm+plastic combined}.
#
# gastronorm is captured on a 10x10 laser grid, plastic on 8x10 (its "two laser faces" rig has a
# physically shorter camera crop -- fewer rows were ever recorded, not a downstream choice, so
# plastic can't be padded with real data the other way). --laser-rows crops gastronorm to its
# first 8 rows so every run below, standalone or combined, sees the same 8x10 grid.
#
# Combined runs use --data-dir-2/--split-2 (run.py's combine_train_loaders) to concatenate the two
# boxes' train sets into one; eval stays per-box (plastic's eval buckets get a box2- prefix). Data
# is NOT balanced across boxes: gastronorm has ~15x more 1-cube samples and ~3.7x more 2-cube
# samples than plastic, so it will dominate the combined runs as-is -- that's deliberate, to see
# the raw-union effect before adding any balancing.
#
# run-name                      | box(es)            | split(es)                              | laser grid    | train/eval samples
# ------------------------------|--------------------|-----------------------------------------|---------------|--------------------
# bb-count-gastro-1cube         | gastronorm         | gastronorm_one_cube                     | 8x10 (cropped)| 504 / 127
# bb-count-gastro-2cube         | gastronorm         | gastronorm_two_cube                     | 8x10 (cropped)| 2040 / 151
# bb-count-gastro-everything    | gastronorm         | gastronorm                               | 8x10 (cropped)| 2553 / 406
# bb-count-plastic-1cube        | plastic            | plastic_one_cube                        | 8x10 (native) | 39 / 4
# bb-count-plastic-2cube        | plastic            | plastic_two_cubes                       | 8x10 (native) | 508 / 84
# bb-count-plastic-everything   | plastic            | plastic                                  | 8x10 (native) | 527 / 88
# bb-combined-1cube             | gastronorm+plastic | gastronorm_one_cube + plastic_one_cube  | 8x10          | 543 / gastro 127 + box2- 4
# bb-combined-2cube             | gastronorm+plastic | gastronorm_two_cube + plastic_two_cubes | 8x10          | 2548 / gastro 151 + box2- 84
# bb-combined-everything        | gastronorm+plastic | gastronorm + plastic                    | 8x10          | 3080 / gastro 406 + box2- 88
#
# Bump TAG for a fresh sweep (new wandb group + new run names): TAG=v2 ./scripts/boombox_object_count.sh

set -u
cd "$(dirname "$0")/.."
export PYTHONPATH=.

TAG="${TAG:-v1}"
GROUP="boombox-object-count-$TAG"

COMMON="--model boombox --d-model 1024 --batch-size 256 \
        --augment-mask 0 --augment-fft 0 --loss-fn ce-pixel --max-duration 5000ep \
        --laser-dropout 0 --freq-dropout 0 --wandb-group $GROUP"

GASTRO_DIR="experiments/31_07_2026_gastronorm_exp1"
PLASTIC_DIR="experiments/31_08_2026_green_plastic_two_laser_faces"
GASTRO_ROWS="--laser-rows 0,1,2,3,4,5,6,7"

# --- standalone: gastronorm (cropped to 8x10) ---

python src/run.py $COMMON $GASTRO_ROWS --data-dir "$GASTRO_DIR" --split gastronorm_one_cube \
    --run-name "bb-count-gastro-1cube-$TAG"

python src/run.py $COMMON $GASTRO_ROWS --data-dir "$GASTRO_DIR" --split gastronorm_two_cube \
    --run-name "bb-count-gastro-2cube-$TAG"

python src/run.py $COMMON $GASTRO_ROWS --data-dir "$GASTRO_DIR" --split gastronorm \
    --run-name "bb-count-gastro-everything-$TAG"

# --- standalone: plastic (native 8x10, no crop needed) ---

python src/run.py $COMMON --test-size 0.15 --data-dir "$PLASTIC_DIR" --split plastic_one_cube \
    --run-name "bb-count-plastic-1cube-$TAG"

python src/run.py $COMMON --test-size 0.15 --data-dir "$PLASTIC_DIR" --split plastic_two_cubes \
    --run-name "bb-count-plastic-2cube-$TAG"

python src/run.py $COMMON --test-size 0.15 --data-dir "$PLASTIC_DIR" --split plastic \
    --run-name "bb-count-plastic-everything-$TAG"

# --- combined: gastronorm + plastic trained jointly, evaluated per-box ---

python src/run.py $COMMON $GASTRO_ROWS --data-dir "$GASTRO_DIR" --split gastronorm_one_cube \
    --data-dir-2 "$PLASTIC_DIR" --split-2 plastic_one_cube --test-size 0.15 \
    --run-name "bb-combined-1cube-$TAG"

python src/run.py $COMMON $GASTRO_ROWS --data-dir "$GASTRO_DIR" --split gastronorm_two_cube \
    --data-dir-2 "$PLASTIC_DIR" --split-2 plastic_two_cubes --test-size 0.15 \
    --run-name "bb-combined-2cube-$TAG"

python src/run.py $COMMON $GASTRO_ROWS --data-dir "$GASTRO_DIR" --split gastronorm \
    --data-dir-2 "$PLASTIC_DIR" --split-2 plastic --test-size 0.15 \
    --run-name "bb-combined-everything-$TAG"
