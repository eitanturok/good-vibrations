#!/usr/bin/env bash
# Decoder benchmark: mlp vs attn vs conv (boombox), everything else held fixed.
#
# The question is narrow -- given the same embedding width, which decoder turns it into a
# 21x30 mask best? So d_model is 512 for all three arms and only the decoder changes:
#
#   mlp    VibrationTransformer + MLPDecoder    (arch.py MLPDecoder)  Linear(512->256)->Linear(256->H*W)
#   attn   VibrationTransformer + AttnDecoder   (arch.py AttnDecoder) H*W learned queries cross-attend
#   conv   BoomboxModel + its transposed-conv decoder (boombox.py Decoder)
#
# NOTE the arms are NOT parameter-matched -- the conv decoder upsamples through 512/256/128/64
# channel stacks and is much larger than the mlp head. Matching width is the honest comparison
# for "what does the decoder do with a fixed-width embedding"; matching params would mean
# changing d_model per arm and confounding the encoder.
#
# mse + sigmoid. Both were varied in earlier runs (l1 collapsed to all-zero; no-sigmoid broke
# soft-iou by emitting negatives) -- this benchmark holds them at the known-good setting so the
# decoder is the only variable.
#
# Dropouts pinned to 0, and passed EXPLICITLY rather than left to the defaults. At 0.3 they cost
# more than half the eval score: dec-d3-conv-v5 (drop 0.3) got 0.119 2-cube soft-iou against
# bb-r2-boombox-d512-mse-v2's 0.286 (drop 0) on an otherwise byte-identical config, and its TRAIN
# soft-iou was 0.585 vs 0.955 -- so it underfits rather than regularizes. _drop zeros whole lasers
# (boombox.py) and the 10x10 grid carries all the spatial signal, so 30% of lasers gone is 30% of
# the measurements gone. Also unequal across arms: conv and attn lean on the laser grid
# differently, which would confound the decoder comparison this script exists to make.
#
# BATCH SIZE 64, not the 256 the boombox ladder used. VibrationTransformer flattens lasers into
# the batch dim (arch.py forward: x.flatten(0,1)) so the freq encoder's effective batch is
# batch*100. At 256 that is 25600 and the attention backward OOMs on a 15.5GB card. Boombox's
# conv encoder has no such blowup, but the arms must share a batch size to stay comparable --
# so every arm runs at the size the transformer can afford.
#
# --ffn-dim 2048 on the two transformer arms. arch.py otherwise sizes every FFN at 4*d_model,
# which at d_model=128 is 512 -- but torch's own TransformerEncoderLayer default is a FIXED 2048,
# so runs from before that was pinned (mag-v1 and everything of that era) had a 2048-wide FFN.
# Pinning 2048 here keeps the transformer arms at the capacity those runs had. It does not apply
# to the conv arm: boombox has no attention and no FFN.
#
# Read hard-iou and localization, not soft-iou.
#
#   tmux new -s decoder
#   ./scripts/decoder_benchmark.sh 2>&1 | tee runs/decoder_benchmark.log

set -u  # NOT -e: one diverging arm should not kill the rest

cd "$(dirname "$0")/.."
export PYTHONPATH=.

exec 9>/tmp/decoder_benchmark.sh.lock
flock -n 9 || { echo "another copy of $(basename "$0") is already running; exiting" >&2; exit 1; }

TAG=v6
GROUP=decoder-benchmark-$TAG

COMMON="--split gastronorm --out-h 21 --out-w 30 --augment-mask 0 --augment-fft 0 \
        --loss-fn mse --max-duration 1000ep \
        --laser-dropout 0 --freq-dropout 0 --wandb-group $GROUP"

# D3: conv. Boombox's transposed-conv stack: seed a 3x4 grid from the embedding, double three
# times to 24x32, resize to 21x30. The only arm with a spatial inductive bias in the decoder.
python src/run.py $COMMON --batch-size 256  --d-model 512 --model boombox \
    --run-name dec-d3-conv-$TAG

# D1: mlp head. The cheapest decoder -- one hidden layer, no spatial structure at all.
python src/run.py $COMMON --batch-size 128 --d-model 128 --model transformer --decoder mlp \
    --run-name dec-d1-mlp-$TAG

# D2: attention. H*W learned queries cross-attend to the token sequence, so each output cell
# can look at whichever lasers/freqs it needs rather than sharing one flattened projection.
python src/run.py $COMMON --batch-size 128 --d-model 128 --model transformer --decoder attn \
    --decoder-num-heads 8 --decoder-num-layers 2 \
    --run-name dec-d2-attn-$TAG
