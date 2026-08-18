#!/usr/bin/env bash
# Runs sequentially on the single GPU. Launch under tmux:
#   tmux new -s alpha
#   ./scripts/loss_alpha_ablation.sh 2>&1 | tee runs/loss_alpha_ablation.log
# Detach with ctrl-b d; reattach with `tmux attach -t alpha`.

set -u  # deliberately NOT -e: one diverging run should not kill the rest of the sweep

cd "$(dirname "$0")/.."
export PYTHONPATH=.

TAG=v3

for DECODER in attn mlp; do
    GROUP=asym-loss-$DECODER-$TAG

    for LOSS in mse ce-pixel; do
        for ALPHA in 0.1 0.3 0.5 0.7 0.9; do
            # a01, a03, ... -- strip the dot so run names stay filesystem friendly
            ATAG=a${ALPHA/./}

            echo "***** $LOSS-asym alpha=$ALPHA decoder=$DECODER *****"
            python src/run.py --split gastronorm --out-h 21 --out-w 30 --augment-mask 0 --augment-fft 0 \
                --loss-fn $LOSS-asym --loss-alpha $ALPHA --max-duration 1000ep \
                --decoder $DECODER --wandb-group "$GROUP" \
                --run-name loss-alpha-$LOSS-$ATAG-$DECODER-$TAG
        done
    done
done
