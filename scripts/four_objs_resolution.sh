#!/usr/bin/env bash
# Output resolution sweep on the four_objs capture: the boombox (conv) base arm of
# four_objs_ablation.sh, 500 epochs, at 32x32 (baseline) / 64x64 / 128x128 / 256x256 output masks.
# --decoder-grow adds learned stride-2 upsampling stages past 32x32 until out_h x out_w is reached
# (no-op at 32x32), instead of bilinearly stretching the decoder's 32x32 feature map.
# Each resolution builds its own MDS cache the first time.
#   TAG=1 ./scripts/four_objs_resolution.sh 2>&1 | tee runs/four_objs_resolution.log

set -u
cd "$(dirname "$0")/.."
export PYTHONPATH=.

TAG="${TAG:-1}"
GROUP="four-objs-resolution-v$TAG"

COMMON="--model boombox --d-model 1024 --batch-size 1024 --decoder-grow \
        --augment-mask 0 --augment-fft 0 --loss-fn ce-pixel \
        --laser-dropout 0 --freq-dropout 0 \
        --max-duration 500ep --t-warmup 100ep --eval-interval 50ep --viz-interval 50ep \
        --checkpoint-interval 250ep --seed 42 \
        --data-dir experiments/2026_09_27_gastronorm_four_objs --split gastronorm_four_objs_2 --wandb-group $GROUP"

.venv/bin/python src/run.py $COMMON --out-h 32  --out-w 32  --run-name "gastro-res32-v$TAG"
.venv/bin/python src/run.py $COMMON --out-h 64  --out-w 64  --run-name "gastro-res64-v$TAG"
.venv/bin/python src/run.py $COMMON --out-h 128 --out-w 128 --run-name "gastro-res128-v$TAG"
.venv/bin/python src/run.py $COMMON --out-h 256 --out-w 256 --run-name "gastro-res256-v$TAG"
