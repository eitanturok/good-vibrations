#!/usr/bin/env bash
# two src7 runs on plastic four_objs, 1000 epochs, the polar_src7.sh recipe (batch 1024, speakers 1 3 5 7, seed 42, --window tukey),
# each vs s7-plas-tukey-mask-1k (the mask head, eval iou 0.424):
#   mask-slotsaux -- the mask head still draws the mask; the slots head (scripts/slots_src7.sh) on the same embedding trains alongside
#                    it at weight 1: exists bce + 0.1 * com L1 px + polar iou of 36 free rays. The slots runs located (com_err 0.020 vs
#                    0.030) and counted better but drew worse masks: does their signal help the mask head's embedding?
#   transformer   -- src's VibrationTransformer at its defaults (src/run.py): 32-bin freq patches, a 2-layer freq transformer per laser
#                    (1D RoPE), a 2-layer laser transformer (2D RoPE on the laser grid), d_model 128, 2 heads, an MLP decoder
#                    (128 -> 256 -> 64x64), with src7's bce + aux heads. Batch 128 as in src: at 1024 the freq transformer is 204800
#                    attention sequences x heads, over flash attention's 65535 grid limit
# Runs whose epoch-1000 checkpoint exists are skipped.
#   ./scripts/aux_transformer_src7.sh 2>&1 | tee runs/aux_transformer_src7.log

set -u
cd "$(dirname "$0")/.."

EPOCHS="${EPOCHS:-1000}"
WANDB="${WANDB:-1}"
TAG=$([ "$EPOCHS" = 1000 ] && echo 1k || echo "${EPOCHS}ep")

COMMON="--batch-size 1024 --max-duration ${EPOCHS}ep --t-warmup 100ep --eval-interval 25ep \
        --checkpoint-interval 500ep --seed 42 --speakers 1 3 5 7 --wandb $WANDB --wandb-group slots-aux-transformer --dataset plastic --window tukey"

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

run s7-plas-tukey-mask-slotsaux-$TAG --head mask --slots-aux-weight 1
run s7-plas-tukey-transformer-$TAG   --encoder transformer --decoder mlp --patch-size 32 --d-model 128 --batch-size 128
