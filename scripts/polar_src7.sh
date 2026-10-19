#!/usr/bin/env bash
# src7 polar head (PolarMask, Xie et al. 2020) vs the mask head, plastic four_objs, 1000 epochs, else the freqenc_src7.sh recipe
# (batch 1024, speakers 1 3 5 7, seed 42, --window tukey). The polar head predicts per pixel: is it an object's center (one channel,
# no classes, so unseen objects work), its centerness, and 36 ray lengths to the object's edge; masks are the polygons of the peaks.
#   polar -- --head polar: center focal + polar iou on the rays + centerness bce; aux heads off (its peaks give com + count).
#            The first tries went NaN at epoch ~30, when the first peak cleared --peak-thresh: render's amax backward under
#            torch.compile (fixed: max). No grad clipping, like the mask run
#   mask  -- the current mask head + bce, aux heads at their usual weights
# then the outline prior of the polar head (--ray-shape), each still drawn as 36 rays:
#   fourier4 / fourier8 -- log ray length vs angle = a sum of the first 4 / 8 harmonics (9 / 17 numbers per pixel, not 36): smooth
#                          outlines, circle = the constant alone, rectangles and squares ~ harmonics 2 and 4
#   superellipse        -- |x/a|^m + |y/b|^m = 1 rotated by t (4 numbers): only circles, ellipses, rounded squares and rectangles
# polar first. Runs whose epoch-1000 checkpoint exists are skipped. Waits for scripts/freqenc_src7.sh first (the GPU is full).
#   ./scripts/polar_src7.sh 2>&1 | tee runs/polar_src7.log

set -u
cd "$(dirname "$0")/.."

EPOCHS="${EPOCHS:-1000}"
WANDB="${WANDB:-1}"
TAG=$([ "$EPOCHS" = 1000 ] && echo 1k || echo "${EPOCHS}ep")

COMMON="--batch-size 1024 --max-duration ${EPOCHS}ep --t-warmup 100ep --eval-interval 25ep \
        --checkpoint-interval 500ep --seed 42 --speakers 1 3 5 7 --wandb $WANDB --wandb-group polar-head --dataset plastic --window tukey"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# runs can hang after wandb's last line (idle, 0% CPU, everything saved): kill the run once it's been there 5 min,
# and delete its MDS copy of the dataset, which a killed run can't clean up itself
kill_if_hung() {  # $1 = log, $2 = run name
    if grep -q "Find logs at: ./wandb/run-.*-$2/logs" "$1" 2>/dev/null && [ $(( $(date +%s) - $(stat -c %Y "$1") )) -gt 300 ]; then
        echo "$(date +%T) $2: hung after wandb finished, killing"; pkill -KILL -f -- "--run-name $2( |$)"
        rm -rf "$(grep -o "wrote [0-9]* samples to [^ ]*" "$1" | tail -1 | awk '{print $NF}')"
    fi
}

while pgrep -f "scripts/freqenc_src7.sh" >/dev/null; do sleep 60; done   # it handles its own hung runs

run() {  # $1 = run name, rest = flags
    local name=$1; shift
    if ls runs/$name/checkpoints/ep${EPOCHS}-ba*-rank0.pt >/dev/null 2>&1; then echo "$(date +%T) $name: done already, skipping"; return; fi
    echo "$(date +%T) $name: start (log runs/$name.out)"
    .venv/bin/python src7/run.py $COMMON "$@" --run-name "$name" > "runs/$name.out" 2>&1 &
    local pid=$!
    while kill -0 $pid 2>/dev/null; do sleep 60; kill_if_hung "runs/$name.out" "$name"; done
    echo "$(date +%T) $name: finished"
}

run s7-plas-tukey-polar-free-$TAG         --head polar
run s7-plas-tukey-mask-$TAG  --head mask
run s7-plas-tukey-polar-fourier4-$TAG     --head polar --ray-shape fourier --n-harmonics 4
run s7-plas-tukey-polar-fourier8-$TAG     --head polar --ray-shape fourier --n-harmonics 8
run s7-plas-tukey-polar-superellipse-$TAG --head polar --ray-shape superellipse
