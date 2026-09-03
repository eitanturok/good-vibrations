#!/usr/bin/env bash
# Does the DECODER ARCHITECTURE matter, once the bottleneck, the encoder and the total
# parameter count are all held fixed?  Run on BOTH captures.
#
# scripts/hybrid_decoder.sh asked a nearby question but left three things confounded: H1/H2/H3
# differ in encoder AND decoder AND parameter count all at once ("the conv decoder's
# 512/256/128/64 stack dwarfs the mlp head"). This sweep controls all three:
#
#   * BOTTLENECK  --d-model 512 everywhere. The latent width -- the size of the single vector
#                 handed from the laser encoder's cls token to the decoder. NOT the same knob
#                 as FFN width, layer count or head count, which we vary to hit a budget.
#   * ENCODER     freq encoder + laser encoder are identical across the three decoder families
#                 within a (dataset, size). Only the decoder changes.
#   * PARAMS      three size classes, ~10M / ~30M / ~100M total, matched across the three
#                 decoder families to within a few % (except *-s-conv, see the note below).
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
# bias decoder-heavy (~60-70% of params in the decoder). Not MAE-style; not 3-4x either, since
# the target is a tiny 21x30x1 field with no texture. If the --rgb target is used, push higher.
#
# SCALE BY DEPTH, NOT WIDTH. d_model stays 512 for all 18. Encoder FFN grows 512 -> 768 -> 1024
# (tops out at 1024); decoder widths stay in {512, 1024, 1536, 2048}. Size classes are reached
# by STACKING LAYERS. Every width is a multiple of 256; the only dims that can't be rounded are
# task-fixed (the 630 = 21x30 output, and the attn decoder's 630 queries).
#
#   size  freq enc     laser enc    decoder                  ~enc / ~dec / ~total   dec%
#   S     1L ffn512    1L ffn512    mlp  7 layers, h1024      3.2M / 6.4M / 9.6M     67
#                                   attn 2L ffn512            3.2M / 5.6M / 8.8M     64
#                                   conv base512, 0 res-blk   3.2M / 9.0M / 12.2M    74  (*)
#   M     3L ffn768    3L ffn768    mlp  9 layers, h1536      11.1M / 18.3M / 29.4M  62
#                                   attn 6L ffn1024           11.1M / 19.2M / 30.3M  63
#                                   conv base512, 6 res-blk   11.1M / 18.3M / 29.4M  62
#   L     8L ffn1024   8L ffn1024   mlp  17 layers, h2048     33.7M / 65.3M / 99.0M  66
#                                   attn 21L ffn1024          33.7M / 66.6M / 100.2M 66
#                                   conv base1024, 6 res-blk  33.7M / 66.9M / 100.5M 66
#
# (*) conv can't go below ~9M at d_model 512: the project Linear alone is 3.15M and the
#     transposed-conv stack ~5.9M, before a single res-block. *-s-conv therefore runs
#     ~12M / 74% decoder -- the one arm not cleanly matched at "small". Read it, don't drop it.
#
# Depth over width for the decoder too: attn scales on --decoder-num-layers, conv on residual
# blocks per scale (M widens nothing vs S -- it just adds 6 res-blocks). pm-*-l-attn is 8+8
# encoder + 21 decoder = 37 transformer layers, all 512 wide.
#
# ***** NEW FLAGS THIS SCRIPT NEEDS (not in run.py yet) *****
#   --enc-ffn-dim N          FFN width for freq + laser encoders (splits out of --ffn-dim)
#   --dec-ffn-dim N          FFN width for the attn decoder      (splits out of --ffn-dim)
#   --mlp-dec-depth N        number of Linear layers in MLPDecoder
#   --mlp-dec-hidden N       hidden width of MLPDecoder
#   --conv-dec-mult F        base-channel multiplier for boombox Decoder (base channels = 512*F)
#   --conv-dec-res-blocks N  residual blocks per TwoBranchUp scale
# --pnt-num-layers / --seq-num-layers / --decoder-num-layers already exist.
#
#   tmux new -s pm
#   ./scripts/param_matched_decoder.sh 2>&1 | tee runs/pm_size_ladder.log

set -u  # NOT -e: one diverging arm should not kill the rest

cd "$(dirname "$0")/.."
export PYTHONPATH=.

exec 9>/tmp/pm_size_ladder.sh.lock
flock -n 9 || { echo "another copy of $(basename "$0") is already running; exiting" >&2; exit 1; }

TAG=v1
GROUP=pm-$TAG

# ---- shared by all 18 runs ----
COMMON="--model transformer --d-model 512 \
        --out-h 21 --out-w 30 --augment-mask 0 --augment-fft 0 \
        --loss-fn ce-pixel --max-duration 2000ep \
        --laser-dropout 0 --freq-dropout 0 \
        --pnt-num-heads 8 --seq-num-heads 8 --decoder-num-heads 8 \
        --wandb-group $GROUP"

# ---- per dataset ----
GASTRO="--split gastronorm"   # uses run.py's default --data-dir (31_07_2026_gastronorm_exp1)
GREEN="--data-dir experiments/31_08_2026_green_plastic_two_laser_faces --split green_plastic"

# ---- per size: encoder (freq stack = laser stack), depth-scaled ----
S_ENC="--pnt-num-layers 1 --seq-num-layers 1 --enc-ffn-dim 512"
M_ENC="--pnt-num-layers 3 --seq-num-layers 3 --enc-ffn-dim 768"
L_ENC="--pnt-num-layers 8 --seq-num-layers 8 --enc-ffn-dim 1024"

# S/M fit at batch 128; the ~100M L arms drop to 64 (the freq encoder's effective batch is
# batch * n_lasers). Bump/cut per card.

# =====================================================================================
# GASTRONORM  (100 lasers)
# =====================================================================================

# --- small (~10M) ---
python src/run.py $COMMON $GASTRO $S_ENC --batch-size 128 \
    --decoder mlp  --mlp-dec-depth 7 --mlp-dec-hidden 1024 \
    --run-name pm-gastro-s-mlp-$TAG

python src/run.py $COMMON $GASTRO $S_ENC --batch-size 128 \
    --decoder attn --decoder-num-layers 2 --dec-ffn-dim 512 \
    --run-name pm-gastro-s-attn-$TAG

python src/run.py $COMMON $GASTRO $S_ENC --batch-size 128 \
    --decoder conv --conv-dec-mult 1.0 --conv-dec-res-blocks 0 \
    --run-name pm-gastro-s-conv-$TAG

# --- medium (~30M) ---
python src/run.py $COMMON $GASTRO $M_ENC --batch-size 128 \
    --decoder mlp  --mlp-dec-depth 9 --mlp-dec-hidden 1536 \
    --run-name pm-gastro-m-mlp-$TAG

python src/run.py $COMMON $GASTRO $M_ENC --batch-size 128 \
    --decoder attn --decoder-num-layers 6 --dec-ffn-dim 1024 \
    --run-name pm-gastro-m-attn-$TAG

python src/run.py $COMMON $GASTRO $M_ENC --batch-size 128 \
    --decoder conv --conv-dec-mult 1.0 --conv-dec-res-blocks 6 \
    --run-name pm-gastro-m-conv-$TAG

# --- large (~100M) ---
python src/run.py $COMMON $GASTRO $L_ENC --batch-size 64 \
    --decoder mlp  --mlp-dec-depth 17 --mlp-dec-hidden 2048 \
    --run-name pm-gastro-l-mlp-$TAG

python src/run.py $COMMON $GASTRO $L_ENC --batch-size 64 \
    --decoder attn --decoder-num-layers 21 --dec-ffn-dim 1024 \
    --run-name pm-gastro-l-attn-$TAG

python src/run.py $COMMON $GASTRO $L_ENC --batch-size 64 \
    --decoder conv --conv-dec-mult 2.0 --conv-dec-res-blocks 6 \
    --run-name pm-gastro-l-conv-$TAG

# =====================================================================================
# GREEN PLASTIC BOX -- two laser faces  (80 lasers, all columns)
# =====================================================================================

# --- small (~10M) ---
python src/run.py $COMMON $GREEN $S_ENC --batch-size 128 \
    --decoder mlp  --mlp-dec-depth 7 --mlp-dec-hidden 1024 \
    --run-name pm-green-s-mlp-$TAG

python src/run.py $COMMON $GREEN $S_ENC --batch-size 128 \
    --decoder attn --decoder-num-layers 2 --dec-ffn-dim 512 \
    --run-name pm-green-s-attn-$TAG

python src/run.py $COMMON $GREEN $S_ENC --batch-size 128 \
    --decoder conv --conv-dec-mult 1.0 --conv-dec-res-blocks 0 \
    --run-name pm-green-s-conv-$TAG

# --- medium (~30M) ---
python src/run.py $COMMON $GREEN $M_ENC --batch-size 128 \
    --decoder mlp  --mlp-dec-depth 9 --mlp-dec-hidden 1536 \
    --run-name pm-green-m-mlp-$TAG

python src/run.py $COMMON $GREEN $M_ENC --batch-size 128 \
    --decoder attn --decoder-num-layers 6 --dec-ffn-dim 1024 \
    --run-name pm-green-m-attn-$TAG

python src/run.py $COMMON $GREEN $M_ENC --batch-size 128 \
    --decoder conv --conv-dec-mult 1.0 --conv-dec-res-blocks 6 \
    --run-name pm-green-m-conv-$TAG

# --- large (~100M) ---
python src/run.py $COMMON $GREEN $L_ENC --batch-size 64 \
    --decoder mlp  --mlp-dec-depth 17 --mlp-dec-hidden 2048 \
    --run-name pm-green-l-mlp-$TAG

python src/run.py $COMMON $GREEN $L_ENC --batch-size 64 \
    --decoder attn --decoder-num-layers 21 --dec-ffn-dim 1024 \
    --run-name pm-green-l-attn-$TAG

python src/run.py $COMMON $GREEN $L_ENC --batch-size 64 \
    --decoder conv --conv-dec-mult 2.0 --conv-dec-res-blocks 6 \
    --run-name pm-green-l-conv-$TAG
