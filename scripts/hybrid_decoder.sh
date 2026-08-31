#!/usr/bin/env bash
# Three encoder/decoder pairings, everything else held fixed.
#
#   H1 hybrid  transformer encoder (freq + laser attention) -> boombox conv decoder
#   H2 conv    boombox conv encoder            -> boombox conv decoder   (the full paper model)
#   H3 mlp     transformer encoder             -> mlp head               (the existing default)
#
# H1 is the new arm: arch.py build_decoder now accepts --decoder conv, which hands the laser
# encoder's cls token to boombox.py's transposed-conv Decoder. The contract was already
# (B,D)->(B,H,W), identical to MLPDecoder, so nothing else in the forward path changed.
#
# Reading the arms together separates encoder from decoder. H1 vs H3 isolates the decoder with
# the encoder fixed; H1 vs H2 isolates the encoder with the decoder fixed. Neither pair is
# parameter-matched -- the conv decoder's 512/256/128/64 upsampling stack dwarfs the mlp head --
# so this measures "which pairing predicts best", not "which is more efficient per parameter".
#
# Settings follow scripts/decoder_benchmark.sh, which asked a nearby question:
#  * ce-pixel + 1000ep.
#  * Dropouts pinned to 0 and passed explicitly. At 0.3 the boombox arms underfit badly
#    (train soft-iou 0.585 vs 0.955) since _drop zeroes whole lasers and the 10x10 grid carries
#    all the spatial signal.
#  * --ffn-dim 2048 on the two transformer arms. arch.py sizes FFNs at 4*d_model (=512 at
#    d_model 128), but torch's own default was a fixed 2048, so this keeps them at the capacity
#    the older runs had. No effect on H2: boombox has no attention.
#  * Batch size differs by arm on purpose. VibrationTransformer flattens lasers into the batch
#    dim, so the freq encoder's effective batch is batch*100 and 256 OOMs; the conv encoder has
#    no such blowup and keeps the 256 the boombox ladder used.
#
# Read hard-iou and localization, not soft-iou.
#
#   tmux new -s hybrid
#   ./scripts/hybrid_decoder.sh 2>&1 | tee runs/hybrid_decoder.log

set -u  # NOT -e: one diverging arm should not kill the rest

cd "$(dirname "$0")/.."
export PYTHONPATH=.

exec 9>/tmp/hybrid_decoder.sh.lock
flock -n 9 || { echo "another copy of $(basename "$0") is already running; exiting" >&2; exit 1; }

TAG=v1
GROUP=hybrid-decoder-$TAG

COMMON="--split gastronorm --out-h 21 --out-w 30 --augment-mask 0 --augment-fft 0 \
        --loss-fn ce-pixel --max-duration 1000ep \
        --laser-dropout 0 --freq-dropout 0 --wandb-group $GROUP"

# H1: freq+laser transformer encoder -> boombox conv decoder. The new pairing.
python src/run.py $COMMON --batch-size 128 --d-model 128 --ffn-dim 2048 \
    --model transformer --decoder conv \
    --run-name hyb-h1-transformer-conv-$TAG

# H2: the full boombox -- conv encoder and conv decoder, no attention anywhere.
python src/run.py $COMMON --batch-size 256 --d-model 1024 \
    --model boombox \
    --run-name hyb-h2-boombox-full-$TAG

# H3: the plain transformer -- same encoder as H1, mlp head instead of the conv decoder.
python src/run.py $COMMON --batch-size 128 --d-model 128 --ffn-dim 2048 \
    --model transformer --decoder mlp \
    --run-name hyb-h3-transformer-mlp-$TAG
