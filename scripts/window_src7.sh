#!/usr/bin/env bash
# src7 fft window ablation, gastronorm and cardboard four_objs, the s6-cardboard-3k recipe (batch 1024, speakers 1 3 5 7, 3000 epochs).
#   hann  -- the stored 04_ffts: post_process.clean's hann over the whole 1.3s capture. a log sweep maps freq to time, so this is
#            a freq-dependent gain (~0.15 at 100Hz and 900-1000Hz, 1 at ~300Hz) that moves with each capture's chirp timing
#   tukey -- the fft redone from 02_raw_shifts with 13ms tukey tapers (--window tukey): ~flat gain over the sweep, keeps the ringing
#   ./scripts/window_src7.sh 2>&1 | tee runs/window_src7.log
#   WANDB=0 EPOCHS=1 ...  -- smoke test without a wandb run

set -u
cd "$(dirname "$0")/.."

EPOCHS="${EPOCHS:-3000}"
WANDB="${WANDB:-1}"
TAG=$([ "$EPOCHS" = 3000 ] && echo 3k || echo "${EPOCHS}ep")

COMMON="--batch-size 1024 --max-duration ${EPOCHS}ep --t-warmup 100ep --eval-interval 25ep \
        --checkpoint-interval 500ep --seed 42 --speakers 1 3 5 7 --wandb $WANDB --wandb-group fft-window"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

for DATASET in gastronorm cardboard; do
  for WINDOW in hann tukey; do
    .venv/bin/python src7/run.py $COMMON --dataset $DATASET --window $WINDOW --run-name s7-${DATASET:0:4}-$WINDOW-$TAG
  done
done
