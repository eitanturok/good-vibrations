#!/usr/bin/env bash
# src6 twin of box_compare_four_objs.sh: gastro only vs plastic only vs both boxes, on identical eval sets.
# Same six arms and the same samples per split (checked against src/model/dataset.py BOX_FOUR_OBJS):
#   spk137  -- --speakers 1 3 7; speaker 4 held out at the unseen-speaker positions
#   allspk  -- --speakers all: every speaker in train and eval, no speaker held out
# src6 defaults for everything not set here (bce mask loss, d_model 1024, aux head weights).
#   TAG=1 ./scripts/box_compare_four_objs_src6.sh 2>&1 | tee runs/box_compare_four_objs_src6.log
#   WANDB=0 EPOCHS=1 ...  -- smoke test without a wandb run

set -u
cd "$(dirname "$0")/.."

TAG="${TAG:-1}"
EPOCHS="${EPOCHS:-2000}"
WANDB="${WANDB:-1}"
GROUP="box-compare-four-objs-src6-v$TAG"

COMMON="--batch-size 1024 --max-duration ${EPOCHS}ep --t-warmup 100ep --eval-interval 100ep \
        --checkpoint-interval 500ep --seed 42 --wandb $WANDB --wandb-group $GROUP"

# expandable segments: the plastic arm OOMed on its first compiled backward with 5.6 GiB reserved-but-unallocated
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# skip an arm whose final checkpoint already exists, so a relaunch picks up where the script stopped
run() {
    local name="s6-$3-v$TAG"
    if [ -e "runs/$name/checkpoints/latest-rank0.pt" ] && [[ "$(readlink runs/$name/checkpoints/latest-rank0.pt)" == ep${EPOCHS}-* ]]; then
        echo "skip $name: finished"; return
    fi
    .venv/bin/python src6/run.py $COMMON --dataset "$1" --speakers $2 --run-name "$name"
}

run gastronorm         "1 3 7" box-gastro-spk137
run plastic            "1 3 7" box-plastic-spk137
run gastronorm_plastic "1 3 7" box-both-spk137
run gastronorm         all     box-gastro-allspk
run plastic            all     box-plastic-allspk
run gastronorm_plastic all     box-both-allspk
