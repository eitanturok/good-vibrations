#!/usr/bin/env bash
# src7 slots head vs the polar and mask heads, plastic four_objs, 1000 epochs, the polar_src7.sh recipe (batch 1024, speakers 1 3 5 7,
# seed 42, --window tukey). The polar head's per-pixel rays memorized positions and drew round blobs at new ones; the slots head
# predicts per object (K = 2 slots) from the embedding: exists, a continuous com, and ONE outline (ray params), trained with
# exists bce + com L1 + polar iou on the rays from the true com + dice on the drawn mask (differentiable in com and rays).
#   slots-fourier4     -- log r(θ) up to harmonic 4: 9 numbers per object (the default)
#   slots-superellipse -- rotated rounded rectangle: 4 numbers
#   slots-free         -- 36 free rays
# Compare to s7-plas-tukey-mask-1k (eval iou 0.424) and s7-plas-tukey-polar-free-1k (0.310 at --peak-thresh 0.3, 0.361 at 0.1).
#   ./scripts/slots_src7.sh 2>&1 | tee runs/slots_src7.log

set -u
cd "$(dirname "$0")/.."

EPOCHS="${EPOCHS:-1000}"
WANDB="${WANDB:-1}"
TAG=$([ "$EPOCHS" = 1000 ] && echo 1k || echo "${EPOCHS}ep")

COMMON="--batch-size 1024 --max-duration ${EPOCHS}ep --t-warmup 100ep --eval-interval 25ep \
        --checkpoint-interval 500ep --seed 42 --speakers 1 3 5 7 --wandb $WANDB --wandb-group slots-head --dataset plastic --window tukey"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# runs can hang after wandb's last line (idle, 0% CPU, everything saved): kill the run once it's been there 5 min,
# and delete its MDS copy of the dataset, which a killed run can't clean up itself
kill_if_hung() {  # $1 = log, $2 = run name
    if grep -q "Find logs at: ./wandb/run-.*-$2/logs" "$1" 2>/dev/null && [ $(( $(date +%s) - $(stat -c %Y "$1") )) -gt 300 ]; then
        echo "$(date +%T) $2: hung after wandb finished, killing"; pkill -KILL -f -- "--run-name $2( |$)"
        rm -rf "$(grep -o "wrote [0-9]* samples to [^ ]*" "$1" | tail -1 | awk '{print $NF}')"
    fi
}

while pgrep -f "scripts/polar_src7.sh" >/dev/null; do sleep 60; done   # one run on the GPU at a time

run() {  # $1 = run name, rest = flags
    local name=$1; shift
    if ls runs/$name/checkpoints/ep${EPOCHS}-ba*-rank0.pt >/dev/null 2>&1; then echo "$(date +%T) $name: done already, skipping"; return; fi
    echo "$(date +%T) $name: start (log runs/$name.out)"
    .venv/bin/python src7/run.py $COMMON "$@" --run-name "$name" > "runs/$name.out" 2>&1 &
    local pid=$!
    while kill -0 $pid 2>/dev/null; do sleep 60; kill_if_hung "runs/$name.out" "$name"; done
    echo "$(date +%T) $name: finished"
}

run s7-plas-tukey-slots-fourier4-$TAG     --head slots --ray-shape fourier --n-harmonics 4
run s7-plas-tukey-slots-superellipse-$TAG --head slots --ray-shape superellipse
run s7-plas-tukey-slots-free-$TAG         --head slots --ray-shape free
