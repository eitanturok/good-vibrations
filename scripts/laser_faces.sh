#!/usr/bin/env bash
# Which lasers carry the signal? Two faces of the green-plastic box, one arm each, and one arm
# that straddles both -- each run twice, under ce-pixel and under mse.
#
# The 31_08_2026_green_plastic_two_laser_faces capture puts 80 lasers on an 8x10 grid: columns
# 0-4 are one face of the box, columns 5-9 the other. --laser-cols selects whole columns across
# every row, and the selection is applied where the fft is read off disk -- so every normalization
# and every reference statistic is computed over exactly the kept lasers, and each column set
# builds (and reuses) its own MDS directory.
#
#   face-a      columns 0,1,2,3,4    40 lasers   8x5
#   face-b      columns 5,6,7,8,9    40 lasers   8x5
#   both        columns 1,3,6,8      32 lasers   8x4   -- 2 columns from each face
#
# face-a vs face-b asks whether the two faces are equally informative: they see the same scene
# from different sides, so a gap means the box's geometry, not the lasers, is doing the work.
#
# `both` asks whether spreading across faces beats either face alone. NOTE it has 32 lasers to
# their 40, so a loss there is confounded -- it could be the spread, or it could be the 20% fewer
# lasers. Reading it as evidence about spread requires a same-count single-face control
# (columns 1,2,3,4), which is deliberately not run here.
#
# The loss is crossed with the faces rather than fixed, because the ranking is the question: if
# the two losses order the faces differently, the ordering is a property of the objective and not
# of the lasers. ce-pixel is what every other boombox sweep in this directory uses; mse is the
# run.py default. The two are NOT comparable in absolute value -- compare within a loss.
#
# The model grid follows the selection automatically: 8x5 for the single faces, 8x4 for `both`,
# inferred from each sample's roi boxes rather than configured, so nothing here has to be kept in
# sync.
#
# ***** running it *****
#
# 6 runs, sequential on the single GPU. Under tmux:
#
#     tmux new -s lasers 'scripts/laser_faces.sh 2>&1 | tee /tmp/laser_faces.log'
#
# The first run of each distinct column set pays a one-off MDS build (~1 min for this capture);
# every later run on the same set reuses it, so the whole mse pass builds nothing.

set -u  # deliberately NOT -e: one diverging run should not kill the rest of the sweep

cd "$(dirname "$0")/.."
export PYTHONPATH=.

# Only one copy at a time: two concurrent launches share the single GPU and, worse, the same
# --run-name and outputs_history.
exec 9>/tmp/laser_faces.sh.lock
flock -n 9 || { echo "another copy of $(basename "$0") is already running; exiting" >&2; exit 1; }

TAG=v1
GROUP=laser-faces-$TAG

COMMON="--data-dir experiments/31_08_2026_green_plastic_two_laser_faces \
        --split green_plastic --model boombox --d-model 1024 \
        --out-h 21 --out-w 30 --augment-mask 0 --augment-fft 0 \
        --max-duration 1000ep --batch-size 256 --wandb-group $GROUP"

# ***** ce-pixel: the objective every other boombox sweep here uses, so it runs first *****

# face A alone -- columns 0-4, 8x5 = 40 lasers
python src/run.py $COMMON --loss-fn ce-pixel --laser-cols 0,1,2,3,4 \
    --run-name lasers-face-a-ce-pixel-$TAG

# face B alone -- columns 5-9, 8x5 = 40 lasers
python src/run.py $COMMON --loss-fn ce-pixel --laser-cols 5,6,7,8,9 \
    --run-name lasers-face-b-ce-pixel-$TAG

# both faces -- 2 columns from each, 8x4 = 32 lasers
python src/run.py $COMMON --loss-fn ce-pixel --laser-cols 1,3,6,8 \
    --run-name lasers-both-ce-pixel-$TAG

# ***** mse: the same three arms under run.py's default objective *****

# face A alone -- columns 0-4, 8x5 = 40 lasers
python src/run.py $COMMON --loss-fn mse --laser-cols 0,1,2,3,4 \
    --run-name lasers-face-a-mse-$TAG

# face B alone -- columns 5-9, 8x5 = 40 lasers
python src/run.py $COMMON --loss-fn mse --laser-cols 5,6,7,8,9 \
    --run-name lasers-face-b-mse-$TAG

# both faces -- 2 columns from each, 8x4 = 32 lasers
python src/run.py $COMMON --loss-fn mse --laser-cols 1,3,6,8 \
    --run-name lasers-both-mse-$TAG
