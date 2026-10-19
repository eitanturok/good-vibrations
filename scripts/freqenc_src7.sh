#!/usr/bin/env bash
# src7 FreqEncoder ablation, gastronorm and cardboard four_objs, each trained separately, 1000 epochs (else the s6-cardboard-3k recipe:
# batch 1024, speakers 1 3 5 7, seed 42). The encoder's convs share weights over frequency and average it out at the end, so a fixed
# per-freq gain (deconv) changed training a lot (preproc_src7.sh). Three ways to let it treat bands differently, all defined on Hz:
#   base    -- the current encoder
#   gain    -- --freq-gain 1: learned smooth gain exp(mlp(log f)), starts at 1 (a learned deconv, no chirp spectrum needed)
#   coord   -- --freq-coord 1: log10(f) as an extra input channel (CoordConv)
#   flatten -- --freq-pool flatten: keep the 5 last freq positions, linear layer instead of mean
# each on --window tukey and deconv. Runs whose epoch-1000 checkpoint exists are skipped. Waits for scripts/preproc_src7.sh first.
#   ./scripts/freqenc_src7.sh 2>&1 | tee runs/freqenc_src7.log

set -u
cd "$(dirname "$0")/.."

EPOCHS="${EPOCHS:-1000}"
WANDB="${WANDB:-1}"
TAG=$([ "$EPOCHS" = 1000 ] && echo 1k || echo "${EPOCHS}ep")

COMMON="--batch-size 1024 --max-duration ${EPOCHS}ep --t-warmup 100ep --eval-interval 25ep \
        --checkpoint-interval 500ep --seed 42 --speakers 1 3 5 7 --wandb $WANDB --wandb-group freq-encoder"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# cardboard runs can hang after wandb's last line (idle, 0% CPU, everything saved): kill the run once it's been there 5 min,
# and delete its MDS copy of the dataset (~6.5 GB), which a killed run can't clean up itself
kill_if_hung() {  # $1 = log, $2 = run name
    if grep -q "Find logs at: ./wandb/run-.*-$2/logs" "$1" 2>/dev/null && [ $(( $(date +%s) - $(stat -c %Y "$1") )) -gt 300 ]; then
        echo "$(date +%T) $2: hung after wandb finished, killing"; pkill -KILL -f -- "--run-name $2( |$)"
        rm -rf "$(grep -o "wrote [0-9]* samples to [^ ]*" "$1" | tail -1 | awk '{print $NF}')"
    fi
}

while pgrep -f "scripts/preproc_src7.sh" >/dev/null; do sleep 60; done   # it handles its own hung runs

run() {  # $1 = run name, rest = flags
    local name=$1; shift
    if ls runs/$name/checkpoints/ep${EPOCHS}-ba*-rank0.pt >/dev/null 2>&1; then echo "$(date +%T) $name: done already, skipping"; return; fi
    echo "$(date +%T) $name: start (log runs/$name.out)"
    .venv/bin/python src7/run.py $COMMON "$@" --run-name "$name" > "runs/$name.out" 2>&1 &
    local pid=$!
    while kill -0 $pid 2>/dev/null; do sleep 60; kill_if_hung "runs/$name.out" "$name"; done
    echo "$(date +%T) $name: finished"
}

for DATASET in gastronorm cardboard; do
    D=${DATASET:0:4}
    for WINDOW in tukey deconv; do
        run s7-$D-$WINDOW-base-$TAG    --dataset $DATASET --window $WINDOW
        run s7-$D-$WINDOW-gain-$TAG    --dataset $DATASET --window $WINDOW --freq-gain 1
        run s7-$D-$WINDOW-coord-$TAG   --dataset $DATASET --window $WINDOW --freq-coord 1
        run s7-$D-$WINDOW-flatten-$TAG --dataset $DATASET --window $WINDOW --freq-pool flatten
    done
done
