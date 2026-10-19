#!/usr/bin/env bash
# src7 on the log power spectrum, --signal log_power: log(|X|^2 + 1e-6), then the usual per-sample std norm. 1000 epochs, the
# freqenc_src7.sh recipe (batch 1024, speakers 1 3 5 7, seed 42, --window tukey), trained on each box alone and on all three together:
#   gast / card / plas -- vs s7-gast-tukey-base-1k, s7-card-tukey-base-1k, s7-plas-tukey-mask-1k (magnitude, same recipe)
#   gpc                -- gastronorm_plastic_cardboard, with its own magnitude baseline s7-gpc-tukey-base-1k (none existed)
# Runs whose epoch-1000 checkpoint exists are skipped.
#   ./scripts/log_power_src7.sh 2>&1 | tee runs/log_power_src7.log
#   WANDB=0 EPOCHS=1 ...  -- smoke test without a wandb run

set -u
cd "$(dirname "$0")/.."

EPOCHS="${EPOCHS:-1000}"
WANDB="${WANDB:-1}"
TAG=$([ "$EPOCHS" = 1000 ] && echo 1k || echo "${EPOCHS}ep")

COMMON="--batch-size 1024 --max-duration ${EPOCHS}ep --t-warmup 100ep --eval-interval 25ep \
        --checkpoint-interval 500ep --seed 42 --speakers 1 3 5 7 --wandb $WANDB --wandb-group log-power --window tukey"

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
    echo "$(date +%T) $name: finished"
}

run s7-gast-tukey-logpower-$TAG --dataset gastronorm                   --signal log_power
run s7-card-tukey-logpower-$TAG --dataset cardboard                    --signal log_power
run s7-plas-tukey-logpower-$TAG --dataset plastic                      --signal log_power
run s7-gpc-tukey-logpower-$TAG  --dataset gastronorm_plastic_cardboard --signal log_power
run s7-gpc-tukey-base-$TAG      --dataset gastronorm_plastic_cardboard
