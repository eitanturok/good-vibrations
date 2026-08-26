#!/usr/bin/env bash
# Stage 2 of the phase ablation: how much should the phase block WEIGH?
#
# Run this after scripts/phase_ablation.sh, passing whichever arm won:
#   ./scripts/phase_weight_sweep.sh rel_laser_w
#
# Why this is a separate script: the winning arm is not knowable until the first batch finishes,
# and sweeping the weight on all six arms would be 18 runs instead of 6 + 2.
#
# Why the weight matters here more than it would under log-magnitude: the magnitude block is
# std-normalized plain |Z|, which keeps the orders-of-magnitude dynamic range of the resonance
# peaks, while the phase block is bounded in [-1,1] by construction (it is a unit phasor, and
# rescaling it would distort the circular geometry that makes cos/sin the right encoding). The
# two blocks are therefore NOT on a common scale by default, and --phase-weight is the only
# control over the mix. Notebook 68 found the magnitude/phase mix matters monotonically.
#
# Weight 1.0 is not repeated -- it is already the winning arm's run in phase_ablation.sh.

set -u

cd "$(dirname "$0")/.."
export PYTHONPATH=.

ARM=${1:?usage: $(basename "$0") <phase-arm>   e.g. rel_laser_w}

exec 9>/tmp/phase_weight_sweep.sh.lock
flock -n 9 || { echo "another copy of $(basename "$0") is already running; exiting" >&2; exit 1; }

TAG=v2
GROUP=phase-ablation-$TAG   # same group as stage 1, so the weights line up against the arms

# Byte-identical to phase_ablation.sh's COMMON, which is itself dec-d3-conv-v6's config. It has to
# be: this sweep shares a wandb group with stage 1, so a different model or loss would put
# incomparable numbers on the same chart. (It previously ran --decoder attn with no --model,
# i.e. the TRANSFORMER under ce-pixel -- comparable to nothing in stage 1.)
COMMON="--split gastronorm --out-h 21 --out-w 30 --augment-mask 0 --augment-fft 0 \
        --loss-fn mse --model boombox --d-model 512 --batch-size 256 \
        --laser-dropout 0 --freq-dropout 0 --max-duration 1000ep \
        --wandb-group $GROUP --signal-mode magnitude"

for W in 0.25 4.0; do
    python src/run.py $COMMON --phase-arm "$ARM" --phase-weight "$W" \
        --run-name "phase-p7-${ARM//_/-}-w${W}-$TAG"
done
