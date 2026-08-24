#!/usr/bin/env bash
# Runs sequentially on the single GPU. Launch under tmux:
#   tmux new -s alpha
#   ./scripts/asymetric_loss_abalation.sh 2>&1 | tee runs/asymetric_loss_abalation.log
# Detach with ctrl-b d; reattach with `tmux attach -t alpha`.
#
# RESUME 2026-08-18: the original nested-loop sweep was interrupted. The commands
# below were unrolled from the loops (order preserved: DECODER attn->mlp, LOSS
# mse->ce-pixel, ALPHA 0.1..0.9). Runs that already reached their ep1000
# checkpoint are commented out with "# DONE". The 13 remaining runs are active.
# (run.py also has autoresume=True + wandb resume=allow, so this is idempotent.)

set -u  # deliberately NOT -e: one diverging run should not kill the rest of the sweep

cd "$(dirname "$0")/.."
export PYTHONPATH=.

TAG=v3
PY=.venv/bin/python  # use the project venv explicitly (tmux may not have it activated)

run () {
    local LOSS=$1 ALPHA=$2 DECODER=$3
    local ATAG=a${ALPHA/./}            # a01, a03, ... -- strip the dot so run names stay fs-friendly
    local GROUP=asym-loss-$DECODER-$TAG
    echo "***** $LOSS-asym alpha=$ALPHA decoder=$DECODER *****"
    "$PY" src/run.py --split gastronorm --out-h 21 --out-w 30 --augment-mask 0 --augment-fft 0 \
        --loss-fn "$LOSS"-asym --loss-alpha "$ALPHA" --max-duration 1000ep \
        --decoder "$DECODER" --wandb-group "$GROUP" \
        --run-name loss-alpha-"$LOSS"-"$ATAG"-"$DECODER"-"$TAG"
}

# ---------- DECODER = attn ----------
# run mse      0.1 attn   # DONE 2026-08-18
# run mse      0.3 attn   # DONE 2026-08-18
# run mse      0.5 attn   # DONE 2026-08-18
# run mse      0.7 attn   # DONE 2026-08-18
# run mse      0.9 attn   # DONE 2026-08-18
# run ce-pixel 0.1 attn   # DONE 2026-08-18
# run ce-pixel 0.3 attn   # DONE 2026-08-18
run ce-pixel 0.5 attn
run ce-pixel 0.7 attn
run ce-pixel 0.9 attn

# ---------- DECODER = mlp ----------
run mse      0.1 mlp
run mse      0.3 mlp
run mse      0.5 mlp
run mse      0.7 mlp
run mse      0.9 mlp
run ce-pixel 0.1 mlp
run ce-pixel 0.3 mlp
run ce-pixel 0.5 mlp
run ce-pixel 0.7 mlp
run ce-pixel 0.9 mlp
