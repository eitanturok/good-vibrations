#!/usr/bin/env bash
# Two-stream fusion: does giving magnitude and phase SEPARATE frequency stacks recover the
# loss that adding phase caused?
#
# v2 finding this tests: every phase arm landed below no-phase (best rel_laser_w 0.2636 vs
# B1 0.2823), and the damage scaled with channel count -- the signature of interference at
# fusion. The single-stream Encoder sums magnitude and phase inside the FIRST conv_block's
# weights, before any nonlinearity, so the model never processes them apart.
#
# Both arms hold the gauge at rel_laser_w (the v2 winner) so the ONLY variable is fusion.
# Targets: T1 should beat 0.2636 if interference is real; T2 should reach or beat 0.2823 --
# its gate starts shut (sigmoid 0.119 measured at init), so it begins near the no-phase model
# and must earn phase in, giving it a floor at ~B1 rather than at P2.
#
# CAPACITY CONFOUND: two-stream is 21.26M vs 19.86M (+7%). T3 is the control -- single-stream,
# same gauge, so (T1 - T3) is fusion depth and (T3 - E3/P2) is capacity. Without T3 a win is
# unattributable.

set -u
cd "$(dirname "$0")/.."
export PYTHONPATH=.
exec 9>/tmp/phase_ablation.sh.lock
flock -n 9 || { echo "another phase script is already running; exiting" >&2; exit 1; }

TAG=v3
GROUP=two-stream-$TAG
COMMON="--split gastronorm --augment-mask 0 --augment-fft 0 \
        --loss-fn mse --model boombox --d-model 512 --batch-size 256 \
        --laser-dropout 0 --freq-dropout 0 --max-duration 1000ep \
        --num-workers 2 --signal-mode magnitude --phase-arm rel_laser_w \
        --wandb-group $GROUP"

# T1: separate stacks, plain concat fusion. Tests late fusion alone.
python src/run.py $COMMON --encoder two-stream --fuse concat --run-name "ts-t1-concat-$TAG"

# T2: separate stacks, magnitude-driven sigmoid gate on phase, biased shut at init.
python src/run.py $COMMON --encoder two-stream --fuse gate --run-name "ts-t2-gate-$TAG"

# T3: capacity control -- single-stream at the v2 width, same gauge. Should reproduce P2's
# 0.2636; if it lands near T1 instead, the two-stream gain was params, not fusion depth.
python src/run.py $COMMON --encoder single --run-name "ts-t3-single-ctrl-$TAG"
