#!/usr/bin/env bash
# src6 dice-weight sweep: mask loss = (1 - w) * bce + w * dice, w in {0, 0.1, 0.3, 0.5}.
# src6's bce-dice is alpha * bce + (1 - alpha) * dice, so alpha = 1 - w. w = 0 is plain bce (src6 default).
# src6 defaults for everything else (gastronorm_plastic, speakers 1 3 7, d_model 1024, aux head weights).
#   TAG=1 ./scripts/dice_weight_src6.sh 2>&1 | tee runs/dice_weight_src6.log
#   WANDB=0 EPOCHS=1 ...  -- smoke test without a wandb run

set -u
cd "$(dirname "$0")/.."

TAG="${TAG:-1}"
EPOCHS="${EPOCHS:-2000}"
WANDB="${WANDB:-1}"
GROUP="dice-weight-src6-v$TAG"

COMMON="--batch-size 1024 --max-duration ${EPOCHS}ep --t-warmup 100ep --eval-interval 100ep \
        --checkpoint-interval 500ep --seed 42 --wandb $WANDB --wandb-group $GROUP"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# skip an arm whose final checkpoint already exists, so a relaunch picks up where the script stopped
run() {
    local name="s6-$1-v$TAG"; shift
    if [ -e "runs/$name/checkpoints/latest-rank0.pt" ] && [[ "$(readlink runs/$name/checkpoints/latest-rank0.pt)" == ep${EPOCHS}-* ]]; then
        echo "skip $name: finished"; return
    fi
    .venv/bin/python src6/run.py $COMMON --run-name "$name" "$@"
}

run dice0  --loss-fn bce
run dice01 --loss-fn bce-dice --alpha 0.9
run dice03 --loss-fn bce-dice --alpha 0.7
run dice05 --loss-fn bce-dice --alpha 0.5
