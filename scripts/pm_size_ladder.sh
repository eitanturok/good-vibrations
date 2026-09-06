#!/usr/bin/env bash
# Does the DECODER ARCHITECTURE matter, once the bottleneck-per-size, the encoder and the total
# parameter count are all held fixed?  Run on BOTH captures.
#
# scripts/hybrid_decoder.sh asked a nearby question but left three things confounded: H1/H2/H3
# differ in encoder AND decoder AND parameter count all at once ("the conv decoder's
# 512/256/128/64 stack dwarfs the mlp head"). This sweep controls all three:
#
#   * BOTTLENECK  --d-model scales WITH size (128/256/512 for S/M/L) -- see note below, this is
#                 a deliberate departure from the original fixed-512 design. Within a size class
#                 d_model is identical across mlp/attn/conv, so the three decoder families are
#                 still compared at matched bottleneck width -- just not matched ACROSS sizes.
#   * ENCODER     freq encoder + laser encoder are identical across the three decoder families
#                 within a (dataset, size). Only the decoder changes.
#   * PARAMS      three size classes, ~10M / ~30M / ~150M total, matched across the three
#                 decoder families to within a few %.
#
# 3 decoder families x 3 sizes x 2 datasets = 18 runs, 2000 epochs each.
#
#   mlp    MLPDecoder -- a plain Linear stack on the cls token. The "how far does a dumb head
#          get" control. At M/L it is mostly one big square weight; read it as a floor.
#   attn   AttnDecoder -- DETR-style: one learned query per output cell, cross-attending into
#          the per-laser token sequence.
#   conv   boombox Decoder -- transposed-conv upsampling from a 3x4 seed, with N residual
#          blocks per upsampling scale (a res-block is y = x + f(x): two convs plus a skip;
#          this is the decoder's depth knob, the same idea as VQ-GAN's num_res_blocks).
#
# ***** datasets *****
#   gastronorm      experiments/31_07_2026_gastronorm_exp1               (run.py default data-dir)
#                   10x10 laser grid, 100 lasers.
#   green-plastic   experiments/31_08_2026_green_plastic_two_laser_faces, --split green_plastic
#                   8x10 laser grid, 80 lasers (cols 0-4 one box face, 5-9 the other; all kept).
# Both decode to a 21x30 occupancy mask and the model's parameter counts are identical between
# them (laser count only changes a RoPE buffer, not weights), so the ladder below is
# byte-identical across the two -- the only difference is --data-dir / --split.
#
# ***** encoder / decoder split *****
#
# The literature splits by which side does the part you care about:
#   symmetric      MT transformers, T5, BART -- both sides do comparably hard sequence work.
#   encoder-heavy  MAE (tiny 8-block decoder vs ViT-L/H encoder), BERT-style pretraining,
#                  seg/depth heads -- the decoder is a throwaway reconstruction head.
#   decoder-heavy  VAE / VQ-GAN (decoder ~1.3-2x the encoder), GAN generators, autoregressive
#                  image/audio models -- the output IS the deliverable and synthesis needs
#                  iterative refinement.
#
# This task -- vibration spectrum -> spatial occupancy field -- is synthesis: the decoder
# output is the product, and mapping a global latent to a coherent layout is the crux. So we
# bias decoder-heavy (~65% of params in the decoder, ~78% at S -- see table). Not MAE-style;
# not 3-4x either, since the target is a tiny 21x30x1 field with no texture.
#
# ***** why d_model scales with size (departure from the original design) *****
# The original version of this ladder held --d-model fixed at 512 for all three sizes and
# scaled ONLY depth + decoder width, on the theory that the bottleneck is a separate knob from
# "how much compute you spend either side of it". This version instead scales d_model itself
# 128 -> 256 -> 512 across S -> M -> L, on request, to see whether the SAME decoder-architecture
# comparison holds once the bottleneck is allowed to grow with size the way it would in a normal
# capacity ladder. Known consequence: this confounds "decoder architecture" with "bottleneck
# width" across size classes -- a difference between S and L now could be architecture, could be
# d_model, could be both. Within a size class the comparison is still clean (d_model matched
# across mlp/attn/conv). Read cross-size deltas with that caveat; within-size deltas are as
# clean as the original design.
#
# SCALE BY DEPTH AND d_model. Encoder FFN is always 2x d_model (256/512/1024 for S/M/L). Encoder
# depth is 8 -> 10 -> 12 layers S -> M -> L -- NOT monotonic with d_model^2 cost alone: at
# d_model=512, 8 layers (the OLD L's encoder depth) already costs ~100M by itself, so depth had
# to come down relative to a d_model-512-everywhere design for the same layer count to fit a
# ~150M total. 12 layers at d_model=512 (this L) still costs ~50M just for the encoder.
#
# MLP decoder depth is monotonic by construction (14 -> 20 -> 24), not just whatever combo of
# depth x hidden happened to land closest to the param target -- an earlier pass here picked
# 14/10/24 (M *shallower* than S), which silently broke the "does depth help" story this ladder
# exists to test. Fixed by constraining the M/L search to depth > previous size's depth.
#
# heads scale WITH d_model (4/8/8 for S/M/L) to hold head_dim roughly fixed at 32 for S and M,
# 64 for L, rather than letting it shrink to 16 at S if heads stayed fixed at 8 everywhere.
# Free in param budget: attention's Q/K/V/out projections are d_model x d_model regardless of
# how many heads that width is split into, so head count alone never changes a param count.
#
#   size  d_model  heads  freq enc     laser enc    decoder                    ~enc / ~dec / ~total  dec%
#   S     128      4      8L ffn256    8L ffn256    mlp  14 layers, h768       2.1M / 7.7M / 9.8M     78
#                                                    attn 20L ffn1024          2.1M / 8.0M / 10.1M     79
#                                                    conv mult0.75, 4 res-blk  2.1M / 7.4M / 9.5M      78
#   M     256      8      10L ffn512   10L ffn512   mlp  20 layers, h1024     10.6M / 19.8M / 30.4M   65
#                                                    attn 18L ffn1024         10.6M / 19.1M / 29.7M   64
#                                                    conv mult1.25, 4 res-blk 10.6M / 20.8M / 31.4M   66
#   L     512      8      12L ffn1024  12L ffn1024  mlp  24 layers, h2048     50.5M / 94.7M / 145.2M  65
#                                                    attn 24L ffn2048         50.5M / 101.2M / 151.7M 67
#                                                    conv mult2.5, 6 res-blk  50.5M / 102.6M / 153.1M 67
#
# All totals verified against the actual model classes (not hand-estimated); see conversation
# history / git blame for the search script.
#
# ***** flags this script needs (added to run.py alongside this script) *****
#   --enc-ffn-dim N          FFN width for freq + laser encoders (splits out of --ffn-dim)
#   --dec-ffn-dim N          FFN width for the attn decoder      (splits out of --ffn-dim)
#   --mlp-dec-depth N        number of Linear layers in MLPDecoder
#   --mlp-dec-hidden N       hidden width of MLPDecoder
#   --conv-dec-mult F        base-channel multiplier for boombox Decoder (base channels = 512*F)
#   --conv-dec-res-blocks N  residual blocks per TwoBranchUp scale (same-resolution ResBlock,
#                            not another TwoBranchUp -- that would upsample again, not refine)
# --pnt-num-layers / --seq-num-layers / --decoder-num-layers already exist.
#
#   tmux new -s pm
#   ./scripts/pm_size_ladder.sh 2>&1 | tee runs/pm_size_ladder.log

set -u  # NOT -e: one diverging arm should not kill the rest

cd "$(dirname "$0")/.."
export PYTHONPATH=.

exec 9>/tmp/pm_size_ladder.sh.lock
flock -n 9 || { echo "another copy of $(basename "$0") is already running; exiting" >&2; exit 1; }

TAG=v2

# ---- shared by all 18 runs ----
COMMON="--model transformer \
        --out-h 21 --out-w 30 --augment-mask 0 --augment-fft 0 \
        --loss-fn ce-pixel --max-duration 2000ep \
        --laser-dropout 0 --freq-dropout 0"

# ---- per dataset ----
# separate wandb groups per dataset (pm-$TAG-gastro / pm-$TAG-green) so the two capture
# geometries -- 100 lasers/10x10 grid vs 80 lasers/two-face -- don't get plotted together
GASTRO="--split gastronorm --wandb-group pm-$TAG-gastro"   # uses run.py's default --data-dir (31_07_2026_gastronorm_exp1)
GREEN="--data-dir experiments/31_08_2026_green_plastic_two_laser_faces --split green_plastic --wandb-group pm-$TAG-green"

# ---- per size: d_model + encoder (freq stack = laser stack), depth-scaled ----
# heads scale WITH d_model to hold head_dim roughly fixed (32/32/64) rather than shrinking to
# head_dim=16 at S if heads stayed fixed at 8 everywhere -- head_dim doesn't change param count
# (Q/K/V/out projections are always d_model x d_model regardless of head split), so this is a
# pure quality choice, free in parameter budget.
S_ENC="--d-model 128 --pnt-num-layers 8  --seq-num-layers 8  --enc-ffn-dim 256  --pnt-num-heads 4 --seq-num-heads 4 --decoder-num-heads 4"
M_ENC="--d-model 256 --pnt-num-layers 10 --seq-num-layers 10 --enc-ffn-dim 512  --pnt-num-heads 8 --seq-num-heads 8 --decoder-num-heads 8"
L_ENC="--d-model 512 --pnt-num-layers 12 --seq-num-layers 12 --enc-ffn-dim 1024 --pnt-num-heads 8 --seq-num-heads 8 --decoder-num-heads 8"

# S/M fit at batch 128; the ~150M L arms drop to 64 (the freq encoder's effective batch is
# batch * n_lasers, and L's d_model=512 encoder is the same width as the old fixed-512 design
# but deeper -- 12 layers vs 8). device-train-microbatch-size defaults to "auto" in run.py, so
# a run that still doesn't fit at these batch sizes grad-accumulates down automatically instead
# of crashing OOM. Bump/cut the batch sizes below per card regardless.

# =====================================================================================
# GASTRONORM  (100 lasers)
# =====================================================================================

# --- small (~10M) ---
python src/run.py $COMMON $GASTRO $S_ENC --batch-size 128 \
    --decoder mlp  --mlp-dec-depth 14 --mlp-dec-hidden 768 \
    --run-name pm-gastro-s-mlp-$TAG

python src/run.py $COMMON $GASTRO $S_ENC --batch-size 128 \
    --decoder attn --decoder-num-layers 20 --dec-ffn-dim 1024 \
    --run-name pm-gastro-s-attn-$TAG

python src/run.py $COMMON $GASTRO $S_ENC --batch-size 128 \
    --decoder conv --conv-dec-mult 0.75 --conv-dec-res-blocks 4 \
    --run-name pm-gastro-s-conv-$TAG

# --- medium (~30M) ---
python src/run.py $COMMON $GASTRO $M_ENC --batch-size 128 \
    --decoder mlp  --mlp-dec-depth 20 --mlp-dec-hidden 1024 \
    --run-name pm-gastro-m-mlp-$TAG

python src/run.py $COMMON $GASTRO $M_ENC --batch-size 128 \
    --decoder attn --decoder-num-layers 18 --dec-ffn-dim 1024 \
    --run-name pm-gastro-m-attn-$TAG

python src/run.py $COMMON $GASTRO $M_ENC --batch-size 128 \
    --decoder conv --conv-dec-mult 1.25 --conv-dec-res-blocks 4 \
    --run-name pm-gastro-m-conv-$TAG

# --- large (~150M) ---
python src/run.py $COMMON $GASTRO $L_ENC --batch-size 64 \
    --decoder mlp  --mlp-dec-depth 24 --mlp-dec-hidden 2048 \
    --run-name pm-gastro-l-mlp-$TAG

python src/run.py $COMMON $GASTRO $L_ENC --batch-size 64 \
    --decoder attn --decoder-num-layers 24 --dec-ffn-dim 2048 \
    --run-name pm-gastro-l-attn-$TAG

python src/run.py $COMMON $GASTRO $L_ENC --batch-size 64 \
    --decoder conv --conv-dec-mult 2.5 --conv-dec-res-blocks 6 \
    --run-name pm-gastro-l-conv-$TAG

# =====================================================================================
# GREEN PLASTIC BOX -- two laser faces  (80 lasers, all columns)
# =====================================================================================

# --- small (~10M) ---
python src/run.py $COMMON $GREEN $S_ENC --batch-size 128 \
    --decoder mlp  --mlp-dec-depth 14 --mlp-dec-hidden 768 \
    --run-name pm-green-s-mlp-$TAG

python src/run.py $COMMON $GREEN $S_ENC --batch-size 128 \
    --decoder attn --decoder-num-layers 20 --dec-ffn-dim 1024 \
    --run-name pm-green-s-attn-$TAG

python src/run.py $COMMON $GREEN $S_ENC --batch-size 128 \
    --decoder conv --conv-dec-mult 0.75 --conv-dec-res-blocks 4 \
    --run-name pm-green-s-conv-$TAG

# --- medium (~30M) ---
python src/run.py $COMMON $GREEN $M_ENC --batch-size 128 \
    --decoder mlp  --mlp-dec-depth 20 --mlp-dec-hidden 1024 \
    --run-name pm-green-m-mlp-$TAG

python src/run.py $COMMON $GREEN $M_ENC --batch-size 128 \
    --decoder attn --decoder-num-layers 18 --dec-ffn-dim 1024 \
    --run-name pm-green-m-attn-$TAG

python src/run.py $COMMON $GREEN $M_ENC --batch-size 128 \
    --decoder conv --conv-dec-mult 1.25 --conv-dec-res-blocks 4 \
    --run-name pm-green-m-conv-$TAG

# --- large (~150M) ---
python src/run.py $COMMON $GREEN $L_ENC --batch-size 64 \
    --decoder mlp  --mlp-dec-depth 24 --mlp-dec-hidden 2048 \
    --run-name pm-green-l-mlp-$TAG

python src/run.py $COMMON $GREEN $L_ENC --batch-size 64 \
    --decoder attn --decoder-num-layers 24 --dec-ffn-dim 2048 \
    --run-name pm-green-l-attn-$TAG

python src/run.py $COMMON $GREEN $L_ENC --batch-size 64 \
    --decoder conv --conv-dec-mult 2.5 --conv-dec-res-blocks 6 \
    --run-name pm-green-l-conv-$TAG
