#!/usr/bin/env bash
# Auxiliary heads sweep on the four_objs capture: the boombox (conv) base arm, 500 epochs, 64x64 + decoder-grow
# (run.py defaults). The heads read the encoder embedding and predict object class, object count,
# per-object com and log area; their losses train the encoder too.
#   base   -- every head weight 0, i.e. mask loss only (the heads still run, but nothing trains them)
#   aux    -- the defaults: cls 0.1, count 0.1, com 1.0, area 0.1
#   aux-x0.3 / aux-x3 -- every default weight scaled by 0.3 / 3
#   TAG=1 ./scripts/four_objs_heads.sh 2>&1 | tee runs/four_objs_heads.log

set -u
cd "$(dirname "$0")/.."
export PYTHONPATH=.

TAG="${TAG:-1}"
GROUP="four-objs-heads-v$TAG"

COMMON="--model boombox --d-model 1024 --batch-size 1024 \
        --augment-mask 0 --augment-fft 0 --loss-fn ce-pixel \
        --laser-dropout 0 --freq-dropout 0 \
        --max-duration 500ep --t-warmup 100ep --eval-interval 50ep --viz-interval 50ep \
        --checkpoint-interval 250ep --seed 42 \
        --data-dir experiments/2026_09_27_gastronorm_four_objs --split gastronorm_four_objs_2 --wandb-group $GROUP"

heads() { echo "--cls-weight $1 --count-weight $2 --com-weight $3 --area-weight $4"; }

.venv/bin/python src/run.py $COMMON $(heads 0    0    0   0   ) --run-name "gastro-heads-base-v$TAG"
.venv/bin/python src/run.py $COMMON $(heads 0.1  0.1  1.0 0.1 ) --run-name "gastro-heads-aux-v$TAG"
.venv/bin/python src/run.py $COMMON $(heads 0.03 0.03 0.3 0.03) --run-name "gastro-heads-aux-x0.3-v$TAG"
.venv/bin/python src/run.py $COMMON $(heads 0.3  0.3  3.0 0.3 ) --run-name "gastro-heads-aux-x3-v$TAG"
