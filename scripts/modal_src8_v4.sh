#!/usr/bin/env bash
# src8 sweep v4: ablations of the hybrid, to find WHAT makes it localize better than conv (decompose.py: CORE com err .024 vs .031,
# shape after position/size alignment equal to conv-magnitude). Waits for v2 and v3. Same 500-ep recipe as v2; each vs s8-hybrid-base:
#   hybrid-nobn      -- no per-box input standardization
#   hybrid-nolaser   -- no laser attention blocks (laser tokens go straight from the band pooling to the flatten embedding)
#   hybrid-mag       -- magnitude input (conv-magnitude had better HARD shape than conv-log-power)
#   ./scripts/modal_src8_v4.sh 2>&1 | tee -a runs/modal_src8_v4.log

set -u
cd "$(dirname "$0")/.."

while pgrep -f "scripts/modal_src8_v[23].sh" >/dev/null; do sleep 120; done

EPOCHS="${EPOCHS:-500}"
WANDB="${WANDB:-1}"
TAG=$([ "$EPOCHS" = 500 ] && echo 500ep || echo "${EPOCHS}ep")

COMMON="--dataset gastronorm_plastic_cardboard --window tukey --batch-size 1024 --max-duration ${EPOCHS}ep --t-warmup 50ep \
        --eval-interval 50ep --image-interval 100ep --checkpoint-interval 250ep --seed 42 --speakers 1 3 5 7 --wandb $WANDB --wandb-group modal-src8-v2"
HYBRID="--encoder modal --decoder conv --d-model 1024 --microbatch 512"

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

run s8-hybrid-nobn-$TAG      $HYBRID --signal log_power --modal-order "" --box-norm 0
run s8-hybrid-nolaser-$TAG   $HYBRID --signal log_power --modal-order "" --modal-laser-layers 0
run s8-hybrid-mag-$TAG       $HYBRID --signal magnitude --modal-order ""
run s8-hybrid-tversky-$TAG   $HYBRID --signal log_power --modal-order "" --loss-fn bce-tversky   # misses / under-painting of weak objects (cardboard cube)
run s8-hybrid-pose-$TAG      $HYBRID --signal log_power --modal-order "" --pose-branch 1 --orient-weight 1   # STN pose-factored branch: rotation as one shared parameter (rotated soap)
