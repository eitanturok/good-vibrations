#!/usr/bin/env bash
# R5: regularization. The v2 ladder ran with no dropout and no augmentation on ~230
# distinct scenes and hit a 0.69 train/eval gap. Everything-on runs first, then the
# individual flags to attribute whatever it buys.
set -u

cd "$(dirname "$0")/.."
export PYTHONPATH=.

exec 9>/tmp/boombox_reg.sh.lock
flock -n 9 || { echo "another copy of $(basename "$0") is already running; exiting" >&2; exit 1; }

TAG=v2
GROUP=boombox-reg-$TAG

COMMON="--split gastronorm --out-h 21 --out-w 30 --augment-fft 0 \
        --loss-fn ce-pixel --max-duration 1000ep --model boombox --d-model 512 \
        --wandb-group $GROUP"

# everything on
python src/run.py $COMMON --laser-dropout 0.3 --freq-dropout 0.3 --augment-mask 0.3 \
    --run-name bb-r5e-all-$TAG

# baseline: nothing on, in this group so it charts against the rest
python src/run.py $COMMON --augment-mask 0 \
    --run-name bb-r5-base-$TAG

# both dropouts, no mask augmentation
python src/run.py $COMMON --augment-mask 0 --laser-dropout 0.3 --freq-dropout 0.3 \
    --run-name bb-r5c-bothdrop-$TAG

# laser dropout only
python src/run.py $COMMON --augment-mask 0 --laser-dropout 0.3 \
    --run-name bb-r5a-laserdrop-$TAG

# freq dropout only
python src/run.py $COMMON --augment-mask 0 --freq-dropout 0.3 \
    --run-name bb-r5b-freqdrop-$TAG

# mask augmentation only
python src/run.py $COMMON --augment-mask 0.3 \
    --run-name bb-r5d-augmask-$TAG
