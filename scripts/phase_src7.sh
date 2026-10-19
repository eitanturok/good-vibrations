#!/usr/bin/env bash
# src7 with the FFT phase as input: --phase cossin adds cos and sin of each channel's phase (C 2 -> 6), after rotating every
# laser by a per-(sample, freq, channel) reference with --phase-ref (x' = x * conj(r) / |r|; magnitudes unchanged).
# All three boxes together (gastronorm_plastic_cardboard), the log_power_src7.sh recipe (batch 1024, speakers 1 3 5 7,
# seed 42, --window tukey, 1000 epochs), so its magnitude-only run s7-gpc-tukey-base-1k is the baseline and is skipped:
#   base       -- magnitude only                                   [from log_power_src7.sh]
#   phase-real -- most-real reference: arg r = 1/2 arg sum_l x^2, sign fixed so the strongest laser has Re >= 0
#   phase-avg  -- mean over lasers: weak in ~1/3-2/3 of bins (antiphase lasers cancel), so a noisy reference
#   phase-raw  -- no rotation: if this matches the others, the model isn't using the phase
# RAM: the phase is stored as its angle (C 2 -> 4; the model makes cos/sin on the GPU), and the phase runs use --num-workers 2.
# Storing cos/sin (6 channels, 2.9 GB per 1024-sample batch) with 61 eval loaders' workers OOM-killed the 62 GB box twice on
# 2026-10-16. Now a batch is 1.9 GB, and 2 workers keep the batches in flight at or below the magnitude-only baseline's.
# Workers don't change results. Each phase run writes a ~36 GB MDS copy, deleted when the run ends -- by run() if it dies.
# Runs whose epoch-1000 checkpoint exists are skipped.
#   ./scripts/phase_src7.sh 2>&1 | tee runs/phase_src7.log
#   WANDB=0 EPOCHS=1 ...  -- smoke test without a wandb run

set -u
cd "$(dirname "$0")/.."

EPOCHS="${EPOCHS:-1000}"
WANDB="${WANDB:-1}"
TAG=$([ "$EPOCHS" = 1000 ] && echo 1k || echo "${EPOCHS}ep")

COMMON="--batch-size 1024 --max-duration ${EPOCHS}ep --t-warmup 100ep --eval-interval 25ep \
        --checkpoint-interval 500ep --seed 42 --speakers 1 3 5 7 --wandb $WANDB --wandb-group phase --window tukey \
        --dataset gastronorm_plastic_cardboard"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# runs can hang after wandb's last line (idle, 0% CPU, everything saved): kill the run once it's been there 5 min,
# and delete its MDS copy of the dataset, which a killed run can't clean up itself
kill_if_hung() {  # $1 = log, $2 = run name
    if grep -q "Find logs at: ./wandb/run-.*-$2/logs" "$1" 2>/dev/null && [ $(( $(date +%s) - $(stat -c %Y "$1") )) -gt 300 ]; then
        echo "$(date +%T) $2: hung after wandb finished, killing"; pkill -KILL -f -- "--run-name $2( |$)"
        rm -rf "$(grep -o "wrote [0-9]* samples to [^ ]*" "$1" | tail -1 | awk '{print $NF}')"
    fi
}

run() {  # $1 = run name, rest = flags
    local name=$1; shift
    if ls runs/$name/checkpoints/ep${EPOCHS}-ba*-rank0.pt >/dev/null 2>&1; then echo "$(date +%T) $name: done already, skipping"; return; fi
    echo "$(date +%T) $name: start (log runs/$name.out)"
    .venv/bin/python src7/run.py $COMMON "$@" --run-name "$name" > "runs/$name.out" 2>&1 &
    local pid=$!
    while kill -0 $pid 2>/dev/null; do sleep 60; kill_if_hung "runs/$name.out" "$name"; done
    # a run that crashed or was killed never reaches its atexit cleanup: drop its MDS copy here
    local mds=$(grep -o "wrote [0-9]* samples to [^ ]*" "runs/$name.out" | tail -1 | awk '{print $NF}')
    [ -n "$mds" ] && [ -d "$mds" ] && { echo "$(date +%T) $name: removing leftover $mds"; rm -rf "$mds"; }
    echo "$(date +%T) $name: finished"
}

run s7-gpc-tukey-base-$TAG
run s7-gpc-tukey-phase-real-$TAG --phase cossin --phase-ref real --num-workers 2
run s7-gpc-tukey-phase-avg-$TAG  --phase cossin --phase-ref avg  --num-workers 2
run s7-gpc-tukey-phase-raw-$TAG  --phase cossin --phase-ref none --num-workers 2
