#!/usr/bin/env bash
# scripts/hybrid_decoder.sh, re-run with every arm at ~27.9M parameters.
#
#   P1 mlp    transformer encoder (freq + laser attention) -> mlp head          d_model 504
#   P2 hybrid transformer encoder                          -> boombox conv dec  d_model 424
#   P3 conv   boombox conv encoder                         -> boombox conv dec  d_model 1024
#
# Why: the hybrid_decoder v1 arms were not parameter-matched and the spread was 10.7x
# (2.59M / 9.06M / 27.74M, counted off the final checkpoints). Any ranking there confounds
# "which pairing predicts best" with "which arm got the most capacity". Here boombox is the
# anchor -- P3 is byte-identical to hyb-h2-boombox-full-v1, same seed, so it reproduces that
# run -- and the two transformer arms are reshaped until they land on it:
#
#   P1  27,822,195   laser_enc 24.44M + freq_enc 3.09M + mlp head 291K
#   P2  27,985,070   laser_enc 17.30M + conv decoder 8.49M + freq_enc 2.19M
#   P3  27,738,982   conv encoder 15.56M + conv decoder 12.17M
#
# a 0.9% spread. Counts are parameters only (the RoPE freqs_cis buffers are excluded); they
# come from instantiating the models at these flags, not from an estimate.
#
# HOW THE SHAPE WAS CHOSEN. Width is not the memory-efficient way to buy parameters here, and
# the reason is structural: VibrationTransformer flattens lasers into the batch dim, so the freq
# encoder runs at effective batch B*100 over 40 tokens while the laser encoder runs at batch B
# over 101 tokens. The freq encoder therefore carries ~40x the activations per unit of width.
# Parameters are cheap in the laser encoder and expensive in the freq encoder, so depth on the
# laser side buys capacity at a fraction of the memory. Measured at ~27.8M params, amp_bf16 +
# torch.compile, largest batch that fits the 16GB 5080:
#
#   pnt/seq layers   d_model   max batch   peak
#     2 / 2            752         32      9.66 GiB   <- widths-only match
#     2 / 4            616         48     11.45 GiB
#     2 / 6            536         48     10.37 GiB
#     2 / 8            480         64     11.75 GiB
#     1 / 6            568         96     11.63 GiB
#     1 / 8            504         96      9.32 GiB   <- chosen
#
# Halving the freq encoder (2 layers -> 1) is worth more than any width change, because it
# halves the expensive half of the model. 1/8 gives batch 96 against 32 for the widths-only
# match -- 3x -- at the same parameter count and with more headroom than 1/6.
#
# Heads go 2 -> 8, so head_dim is ~63 rather than the 376 a 2-head d_model 504 would give.
# Free in parameters and negligible in memory; 2 heads at this width was an artifact of
# scaling up a d_model 128 config, not a choice.
#
# ffn_dim is 4*d_model, not v1's flat 2048: at d_model 128 that 2048 was a 16x FFN, and
# carrying 16x up would put most of the model in the FFNs. 4x is torch's documented ratio.
#
# WHAT THIS COSTS. The 1-layer freq encoder is a real architectural claim, not just capacity
# moved around -- it says per-laser spectral mixing needs one attention layer and the rest of
# the budget belongs to mixing across the laser grid. That is a fair thing to be wrong about,
# and it is the one respect in which these arms are not simply "v1, resized". Note it cuts
# against boombox, whose frequency stack is 4 stages deep, so P1/P2 vs P3 is now also a test
# of where spectral depth is worth spending. P1 vs P2 stays clean either way: both transformer
# arms share the encoder shape and differ only in the decoder (and in the d_model that
# parameter-matching then forces, since the conv decoder costs 8.49M on its own).
#
# The 8-layer laser encoder is post-LN (torch's TransformerEncoderLayer default, not exposed as
# a flag). 8 is about where post-LN stacks stay comfortable at lr 1e-4 with 100ep warmup, which
# is why the search stopped there rather than at the 12 layers the parameter budget would allow.
#
# Batch size, sized by probing fwd+bwd+AdamW peak on the 16GB 5080:
#   * Both transformer arms at 96 (9.32 GiB for P1, 7.98 GiB for P2). They take the SAME 96 on
#     purpose -- P1 vs P2 is the decoder contrast, and it stays clean only if the optimization
#     does not also change. Eval stays at the 108 default (wandb caps logged images at 108).
#   * boombox at 256, unchanged from v1. It has no such blowup (2.75 GiB at 256) and would fit
#     512, but 512 would halve the step count on a ~3K-sample train split and stop this arm
#     from reproducing hyb-h2 exactly. Capacity is what this script matches; throughput is not.
#
# Read hard-iou and localization, not soft-iou.
#
#   tmux new -s parammatched
#   ./scripts/param_matched_decoder.sh 2>&1 | tee runs/param_matched_decoder.log

set -u  # NOT -e: one diverging arm should not kill the rest

cd "$(dirname "$0")/.."
export PYTHONPATH=.

exec 9>/tmp/param_matched_decoder.sh.lock
flock -n 9 || { echo "another copy of $(basename "$0") is already running; exiting" >&2; exit 1; }

TAG=v1
GROUP=param-matched-decoder-$TAG

COMMON="--split gastronorm --out-h 21 --out-w 30 --augment-mask 0 --augment-fft 0 \
        --loss-fn ce-pixel --max-duration 1000ep \
        --laser-dropout 0 --freq-dropout 0 --wandb-group $GROUP"

# Shared encoder shape for both transformer arms: 1 freq layer, 8 laser layers, 8 heads.
TF_SHAPE="--pnt-num-layers 1 --seq-num-layers 8 --pnt-num-heads 8 --seq-num-heads 8"

# P1: transformer encoder -> mlp head, 27.82M.
python src/run.py $COMMON $TF_SHAPE --batch-size 96 --d-model 504 --ffn-dim 2016 \
    --model transformer --decoder mlp \
    --run-name pm-p1-transformer-mlp-$TAG

# P2: transformer encoder -> boombox conv decoder, 27.99M.
python src/run.py $COMMON $TF_SHAPE --batch-size 96 --d-model 424 --ffn-dim 1696 \
    --model transformer --decoder conv \
    --run-name pm-p2-transformer-conv-$TAG

# P3: the full boombox at 27.74M -- the anchor, identical to hyb-h2-boombox-full-v1.
python src/run.py $COMMON --batch-size 256 --d-model 1024 \
    --model boombox \
    --run-name pm-p3-boombox-full-$TAG
