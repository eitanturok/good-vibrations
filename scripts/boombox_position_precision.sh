#!/usr/bin/env bash
# Position-precision ladder for the boombox conv-conv model (arXiv 2105.08052 encoder +
# decoder, --model boombox) on the gastronorm box (--split gastronorm trains on 1-cube AND
# 2-cubes scenes and evaluates both eval/1-cube and eval/2-cubes splits -- see src/run.py
# --n-objects, default None keeps both).
#
# 4 arch variants + 1 combo of all four against the arm-6 baseline, each isolating exactly one
# change (the combo isolates all four together) so any IoU/localization delta is attributable to
# that change alone (see model/boombox.py docstrings for the mechanism + paper each one implements):
#
#   pp-coordconv    --coordconv                     CoordConv (Liu et al. 2018, arXiv:1807.03247):
#                                                     decoder convs get an explicit (x,y) input
#                                                     channel instead of inferring position from
#                                                     context -- the paper's own fix for exactly
#                                                     this "where is it" failure mode.
#   pp-maskedconv   --decoder-arch masked-conv       Conv analogue of Mask2Former's masked
#                                                     attention (Cheng et al. 2022, arXiv:2112.01527):
#                                                     each upsampling stage's coarse mask,
#                                                     thresholded at 0.5, gates what the NEXT
#                                                     stage sees, so later stages spend capacity
#                                                     only on the believed-foreground region.
#   pp-pixelshuffle --decoder-upsample pixelshuffle  Sub-pixel conv upsampling (Shi et al. 2016,
#                                                     arXiv:1609.05158) instead of strided
#                                                     transposed conv, avoiding the checkerboard
#                                                     artifact documented in Odena et al.,
#                                                     Distill 2016 -- a periodic bias that
#                                                     directly corrupts pixel-precise localization.
#   pp-nonlocal     --decoder-nonlocal-stage 1       Non-local self-attention block (Wang et al.
#                                                     2018, arXiv:1711.07971) after the decoder's
#                                                     first upsample (8x8), giving every location a
#                                                     direct one-layer path to every other location
#                                                     instead of relying on stacked 3x3 receptive
#                                                     fields to slowly overlap.
#   pp-combo        all four flags above              Stacks CoordConv + masked-conv decoder +
#                                                     PixelShuffle upsampling + non-local block
#                                                     together -- do the four gains compound, or
#                                                     do they interfere/saturate?
#   pp-baseline     (none of the above)               arm 6, the unmodified boombox decoder --
#                                                     what all five variants above are measured against.
#
# Bump TAG for a fresh sweep (new wandb group + new run names); override per-run without editing:
#   TAG=v2 ./scripts/boombox_position_precision.sh 2>&1 | tee runs/boombox_position_precision.log

set -u
cd "$(dirname "$0")/.."
export PYTHONPATH=.

TAG="${TAG:-v1}"
GROUP="boombox-position-precision-$TAG"

COMMON="--model boombox --d-model 1024 --batch-size 256 \
        --data-dir experiments/31_07_2026_gastronorm_exp1 --split gastronorm \
        --augment-mask 0 --augment-fft 0 --loss-fn ce-pixel --max-duration 500ep \
        --laser-dropout 0 --freq-dropout 0 --wandb-group $GROUP"

# 1: CoordConv
python src/run.py $COMMON --coordconv \
    --run-name "pp-coordconv-$TAG"

# 2: masked-conv decoder (Mask2Former-style threshold gating)
python src/run.py $COMMON --decoder-arch masked-conv \
    --run-name "pp-maskedconv-$TAG"

# 3: PixelShuffle upsampling instead of transposed conv
python src/run.py $COMMON --decoder-upsample pixelshuffle \
    --run-name "pp-pixelshuffle-$TAG"

# 4: non-local block at the decoder's 8x8 stage
python src/run.py $COMMON --decoder-nonlocal-stage 1 \
    --run-name "pp-nonlocal-$TAG"

# 5: combo -- all four techniques stacked together
python src/run.py $COMMON --coordconv --decoder-arch masked-conv --decoder-upsample pixelshuffle --decoder-nonlocal-stage 1 \
    --run-name "pp-combo-$TAG"

# 6: baseline -- unmodified boombox decoder, everything else identical
python src/run.py $COMMON \
    --run-name "pp-baseline-$TAG"
