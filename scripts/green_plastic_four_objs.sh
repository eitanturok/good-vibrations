#!/usr/bin/env bash
# Boombox (conv encoder / conv decoder) on the green-plastic four_objs capture, 1000 epochs. Same recipe as
# the gastronorm four_objs boombox runs (four_objs_heads.sh), with run.py defaults for 64x64 + decoder-grow
# and the aux heads. Split: green_plastic_four_objs (src/model/dataset.py).
#   TAG=1 ./scripts/green_plastic_four_objs.sh 2>&1 | tee runs/green_plastic_four_objs.log

set -u
cd "$(dirname "$0")/.."
export PYTHONPATH=.

TAG="${TAG:-1}"
GROUP="green-plastic-four-objs-v$TAG"

.venv/bin/python src/run.py --model boombox --d-model 1024 --batch-size 1024 \
    --augment-mask 0 --augment-fft 0 --loss-fn ce-pixel \
    --laser-dropout 0 --freq-dropout 0 \
    --max-duration 1000ep --t-warmup 100ep --eval-interval 50ep --viz-interval 50ep \
    --checkpoint-interval 250ep --seed 42 \
    --data-dir experiments/2026_10_05_green_plastic_four_objs --split green_plastic_four_objs \
    --wandb-group "$GROUP" --run-name "green-plastic-bb-v$TAG"
