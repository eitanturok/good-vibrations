#!/usr/bin/env bash
# src8 sweep v3: waits for scripts/modal_src8_v2.sh to finish, then (same 500-ep recipe as v2):
#   hybrid-linskip        -- + a rank-256 linear shortcut from the band-averaged input to the embedding: a ridge probe reads
#                            rotated-soap orientation at 17 deg from the input, hybrid ~28 deg -- does the deep path lose linear info?
#   ./scripts/modal_src8_v3.sh 2>&1 | tee -a runs/modal_src8_v3.log

set -u
cd "$(dirname "$0")/.."

while pgrep -f "scripts/modal_src8_v2.sh" >/dev/null; do sleep 120; done

EPOCHS="${EPOCHS:-500}"
WANDB="${WANDB:-1}"
TAG=$([ "$EPOCHS" = 500 ] && echo 500ep || echo "${EPOCHS}ep")

COMMON="--dataset gastronorm_plastic_cardboard --window tukey --batch-size 1024 --max-duration ${EPOCHS}ep --t-warmup 50ep \
        --eval-interval 50ep --image-interval 100ep --checkpoint-interval 250ep --speakers 1 3 5 7 --wandb $WANDB --wandb-group modal-src8-v2"

HYBRID="--encoder modal --decoder conv --d-model 1024 --microbatch 512"
CONV="--encoder conv"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

kill_if_hung() {  # $1 = log, $2 = run name
    if grep -q "Find logs at: ./wandb/run-.*-$2/logs" "$1" 2>/dev/null && [ $(( $(date +%s) - $(stat -c %Y "$1") )) -gt 300 ]; then
        echo "$(date +%T) $2: hung after wandb finished, killing"; pkill -KILL -f -- "--run-name $2( |$)"
    fi
}

run() {  # $1 = run name, rest = flags
    local name=$1; shift
    if ls runs/$name/checkpoints/ep${EPOCHS}-ba*-rank0.pt >/dev/null 2>&1; then echo "$(date +%T) $name: done already, skipping"; return; fi
    echo "$(date +%T) $name: start (log runs/$name.out)"
    .venv/bin/python src8/run.py $COMMON "$@" --run-name "$name" > "runs/$name.out" 2>&1 &
    local pid=$!
    while kill -0 $pid 2>/dev/null; do sleep 60; kill_if_hung "runs/$name.out" "$name"; done
    local mds=$(grep -o "wrote [0-9]* samples to [^ ]*" "runs/$name.out" | tail -1 | awk '{print $NF}')
    [ -n "$mds" ] && [ -d "$mds" ] && { echo "$(date +%T) $name: removing leftover $mds"; rm -rf "$mds"; }
    echo "$(date +%T) $name: finished"
}

run s8-hybrid-linskip-$TAG        $HYBRID --signal log_power --modal-order "" --lin-skip 256 --seed 42
