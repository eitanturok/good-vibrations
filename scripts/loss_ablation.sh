#!/usr/bin/env bash
# Loss ablation: mse vs ce-pixel vs ce-spatial vs ce-spatial-normalized.
# Everything is held constant except --loss-fn, so the runs are directly comparable.
# out-h/out-w are 7x10 because that resolution is part of the MDS cache key -- changing
# them forces a full rebuild. --augment-fft 0 is likewise the cache-matching value.
set -euo pipefail
cd "$(dirname "$0")/.."

GROUP="${GROUP:-loss-ablation}"
EPOCHS="${EPOCHS:-1000ep}"
LOSSES=(mse ce-pixel ce-spatial ce-spatial-normalized)
# MODAL=1 ./scripts/loss_ablation.sh  -> run on Modal GPU instead of locally
RUNNER=(python src/run.py)
[[ "${MODAL:-0}" == "1" ]] && RUNNER=(modal run src/run.py)

for loss in "${LOSSES[@]}"; do
    echo "===== $loss ($EPOCHS, group=$GROUP) ====="
    PYTHONPATH=. "${RUNNER[@]}" \
        --split gastronorm_one_cube --out-h 7 --out-w 10 \
        --augment-mask 0 --augment-fft 0 \
        --loss-fn "$loss" --max-duration "$EPOCHS" \
        --wandb-group "$GROUP" --run-name "$GROUP-$loss"
done

echo "done: ${#LOSSES[@]} runs in wandb group '$GROUP'"
echo "compare on soft-iou and com-distance -- loss values are NOT comparable across these four"
