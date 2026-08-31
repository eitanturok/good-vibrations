#!/usr/bin/env bash
# Frequency-stack architecture: four changes to where boombox spends its capacity.
#
# The observation behind all of it is the parameter split. At d_model 1024 the model is
# ~1% frequency stack / ~55% laser-grid stack / ~44% decoder. The frequency axis is the one
# with real physics -- the ablations put 133-215Hz and 380-463Hz on top -- and it gets 216K
# params, while grid block 3 alone gets 9.4M to collapse a 3x3 to a 1x1. Every arm here moves
# capacity or information toward the frequency axis; none of them touch the grid stack.
#
# The four changes, in the order the ladder adds them:
#
#   resize   The decoder upsampled 3x4 -> 24x32 and then BILINEAR-resized to 21x30, so the last
#            TwoBranchUp optimized features at a resolution that got resampled away and the head
#            convs ran on interpolated input. --resize conv makes the head's first conv
#            valid-mode with a (4,3) kernel, eating the 3-row/2-col margin exactly. No
#            interpolation, and unlike a crop a conv still reads every input position.
#            This is now the DEFAULT, which is why A0 has to ask for bilinear explicitly.
#
#   trim     tokenize() zero-pads F up to a whole number of patches. That is fine for the
#            transformer -- FreqEncoder.embed sees the zeros at fixed positions and absorbs
#            them -- but a conv slides across them and AdaptiveAvgPool then averages them in at
#            full strength: 45/1280 bins at patch_size 64, 211/3072 at 256. --trim-pad slices
#            them off inside boombox's _to_conv only, so tokenize(), the MDS cache keys and
#            every precomputed .npy stay untouched and the transformer is unaffected.
#
#   collapse The frequency stack ended in AdaptiveAvgPool2d((None,1)), which averages the 5
#            surviving positions into 1 -- so WHICH band a filter fired in is discarded at the
#            last step, on the axis the attribution work says carries the signal.
#            --learned-collapse swaps it for a (1,width) conv: a per-channel weighted sum, which
#            can also downweight the trailing pad instead of averaging it in.
#
#   depth    --freq-depth 2 inserts a stride-1 (1,3) block after each stride-4 stage, adding
#            nonlinearity at each scale. Output width is unchanged, so the grid stack's input
#            stays 256 and all +263K params land in the frequency stack.
#
# --freq-mult is deliberately NOT in this ladder. It scales the stack widths, but the last width
# IS the grid stack's input, so mult=2 costs +1.8M total for +647K in the stack -- it confounds
# "more frequency capacity" with "more capacity". Run it only if A4 shows the stack is
# capacity-starved, and pair it with a grid-widened control.
#
# A4 runs FIRST, on purpose: it is the arm with all four changes, so if the whole idea is dead
# you learn it in one run instead of after four. The ladder A0..A3 then attributes whatever A4
# shows to the individual changes. A0 is a fresh baseline, not the old hyb-h2 run -- same arch,
# but --resize conv changed the default, so the reference point has to be re-measured here.
#
# Settings follow scripts/hybrid_decoder.sh H2, which is the boombox arm these compare against:
#  * ce-pixel + 1000ep, batch 256, d_model 1024.
#  * Dropouts pinned to 0 and passed explicitly. At 0.3 the boombox arms underfit badly
#    (train soft-iou 0.585 vs 0.955) since _drop zeroes whole lasers and the 10x10 grid carries
#    all the spatial signal.
#
# Checkpoints from before this change need --resize bilinear to load: conv resize, learned
# collapse and freq-depth all change parameter shapes.
#
# Read hard-iou and localization, not soft-iou.
#
#   tmux new -s freqarch
#   ./scripts/freq_arch.sh 2>&1 | tee runs/freq_arch.log

set -u  # NOT -e: one diverging arm should not kill the rest

cd "$(dirname "$0")/.."
export PYTHONPATH=.

exec 9>/tmp/freq_arch.sh.lock
flock -n 9 || { echo "another copy of $(basename "$0") is already running; exiting" >&2; exit 1; }

TAG=v1
GROUP=freq-arch-$TAG

COMMON="--split gastronorm --out-h 21 --out-w 30 --augment-mask 0 --augment-fft 0 \
        --loss-fn ce-pixel --max-duration 1000ep \
        --laser-dropout 0 --freq-dropout 0 \
        --batch-size 256 --d-model 1024 --model boombox --wandb-group $GROUP"

# A4: everything on. Run first -- one result says whether the rest of the ladder is worth it.
python src/run.py $COMMON --trim-pad --learned-collapse --freq-depth 2 \
    --run-name fa-a4-all-$TAG

# A0: the old model exactly -- bilinear resize, avgpool collapse, padded input. The reference.
python src/run.py $COMMON --resize bilinear \
    --run-name fa-a0-baseline-$TAG

# A1: + valid-conv head instead of bilinear.  (~+6K params)
python src/run.py $COMMON \
    --run-name fa-a1-convresize-$TAG

# A2: + drop the FFT zero-pad before the convs.  (free)
python src/run.py $COMMON --trim-pad \
    --run-name fa-a2-trim-$TAG

# A3: + learned frequency collapse.  (+328K, all in the freq stack)
python src/run.py $COMMON --trim-pad --learned-collapse \
    --run-name fa-a3-collapse-$TAG
