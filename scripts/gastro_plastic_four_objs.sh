#!/usr/bin/env bash
# Boombox (conv encoder / conv decoder) on gastronorm four_objs + green-plastic four_objs together, 10000 epochs. Same recipe as
# the gastronorm four_objs boombox runs (four_objs_heads.sh), with run.py defaults for 64x64 + decoder-grow
# and the aux heads. Split: gastro_plastic_four_objs (src/model/dataset.py).
#   TAG=2 ./scripts/gastro_plastic_four_objs.sh 2>&1 | tee runs/gastro_plastic_four_objs_v2.log

set -u
cd "$(dirname "$0")/.."
export PYTHONPATH=.

TAG="${TAG:-2}"
GROUP="gastro-plastic-four-objs-v$TAG"

.venv/bin/python src/run.py --model boombox --d-model 1024 --batch-size 1024 \
    --augment-mask 0 --augment-fft 0 --loss-fn ce-pixel \
    --laser-dropout 0 --freq-dropout 0 \
    --max-duration 10000ep --t-warmup 100ep --eval-interval 250ep --viz-interval 250ep \
    --checkpoint-interval 1000ep --seed 42 \
    --data-dir experiments/2026_09_27_gastronorm_four_objs --split gastro_plastic_four_objs \
    --wandb-group "$GROUP" --run-name "gastro-plastic-bb-v$TAG"
