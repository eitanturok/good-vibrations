#!/usr/bin/env bash
# Boombox (arXiv 2105.08052) vs the existing transformer. See docs/boombox.md.
set -u

cd "$(dirname "$0")/.."
export PYTHONPATH=.

exec 9>/tmp/boombox_ladder.sh.lock
flock -n 9 || { echo "another copy of $(basename "$0") is already running; exiting" >&2; exit 1; }

TAG=v3
GROUP=boombox-$TAG

COMMON="--split gastronorm --out-h 21 --out-w 30 --augment-mask 0 --augment-fft 0 \
        --loss-fn ce-pixel --max-duration 1000ep --batch-size 256 --wandb-group $GROUP"

# # R0: overfit 128 samples. Gate -- everything below assumes this passes.
# python src/run.py $COMMON --model boombox --d-model 1024 \
#     --n-samples 128 --batch-size 128 --max-duration 300ep --lr 3e-4 --compile 0 --no-viz \
#     --run-name bb-r0-overfit-$TAG

# R1b: boombox, same data and loss.
python src/run.py $COMMON --model boombox --d-model 1024 \
    --run-name bb-r1-boombox-$TAG

# R1a: existing transformer, the control.
python src/run.py $COMMON --model transformer --decoder attn \
    --run-name bb-r1-transformer-$TAG

# R2: boombox capacity. Is R1b limited by width?
python src/run.py $COMMON --model boombox --d-model 512 \
    --run-name bb-r2-boombox-d512-$TAG

python src/run.py $COMMON --model boombox --d-model 2048 \
    --run-name bb-r2-boombox-d2048-$TAG

# R3: learning rate. CNNs usually want more than the transformer's 1e-4.
python src/run.py $COMMON --model boombox --d-model 1024 --lr 3e-4 \
    --run-name bb-r3-boombox-lr3e4-$TAG

python src/run.py $COMMON --model boombox --d-model 1024 --lr 1e-3 \
    --run-name bb-r3-boombox-lr1e3-$TAG

# R4: does the preprocessing that won the preprocessing ladder also help here?
python src/run.py $COMMON --model boombox --d-model 1024 \
    --signal-mode log_magnitude --subtract-empty-box --subtract-speaker-mean \
    --run-name bb-r4-boombox-logmag-eb-spk-$TAG

