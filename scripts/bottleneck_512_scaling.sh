#!/usr/bin/env bash
# scripts/bottleneck_512_scaling.sh — bottleneck-fixed scaling ladder.
#
#   Latent (bottleneck) dim fixed to 512 for ALL arms so the information
#   bottleneck is identical. We vary CAPACITY via depth / FFN width and
#   decoder choice, not via d_model. This isolates "how to spend params"
#   from "how wide is the bottleneck".
#
#   3 sizes x 3 decoders = 9 arms. Sizes target trainable params (no RoPE
#   buffers, no BN running stats):
#     S ~10M,  M ~30M,  L ~80-100M
#   D_model=512 for every arm.  ffn_dim, pnt/seq layers, and decoder depth
#   are the knobs; d_model never changes.
#
#   Model types (decoder varies, encoder family as noted):
#     mlp   : VibrationTransformer + MLPDecoder              (tiny head, ~0.29M)
#     conv  : VibrationTransformer + Boombox ConvDecoder     (heavy upsampling, ~9.03M)
#     boom  : BoomboxModel  (conv freq stack + conv grid + conv decoder)
#             boombox min with latent 512 is ~19.9M, so S/boom overshoots 10M.
#             This is architectural, not a bug — see notes below.
#
#   Encoder / decoder split — what literature does:
#     - Representation learning (MAE, BERT): heavy encoder, light decoder
#       (MAE: ViT-L 307M encoder / 8-layer 20M decoder). Decoder is throwaway.
#     - Dense prediction (SegFormer, Mask2Former, DETR): balanced or
#       decoder-heavy; generating a spatial map benefits from decoder depth.
#     - Generative / inverse problems (diffusion U-Net, image translation):
#       decoder >= encoder. Reconstructing 630 pixels from a single 512-d
#       vector is generative, so we bias toward a larger decoder here.
#     Target split for this ladder: ~40% encoder / 60% decoder at S and M,
#     ~35% / 65% at L. Achieved by giving large arms a deeper attention /
#     conv decoder (more layers) rather than just widening FFNs — depth
#     adds capacity with better scaling and fewer activation blowups than
#     width (see param_matched_decoder.sh notes on B*100 freq blowup).
#
#   How the shapes were chosen (counts via `sum(p.numel() for p in m.parameters())`):
#
#     SMALL  target ~10M  (boombox floor is 19.9M — see below)
#       s-mlp   1 freq / 2 laser, ffn 2048, mlp head       9.79M  enc 9.50M dec 0.29M
#       s-conv  1 / 1, ffn 1024, conv dec                  13.27M enc 4.24M dec 9.03M  (encoder trimmed to offset conv)
#       s-boom  boombox d512 (mult1 depth1)                19.87M enc 10.84M dec 9.03M  <- floor
#
#     MEDIUM target ~30M
#       m-mlp   1 / 8, ffn 2048, mlp                       28.70M enc 28.41M dec 0.29M
#       m-conv  2 / 2, ffn 4096, conv                      30.07M enc 21.04M dec 9.03M
#       m-boom  boombox d512 mult4 depth2                  30.82M enc 21.80M dec 9.03M
#
#     LARGE  target ~80-100M (100M needs ffn 8192 or extra depth)
#       l-mlp   2 /12, ffn 4096, mlp-mid (~5.8M head)      79.36M enc 73.55M dec 5.80M
#       l-conv  2 /12, ffn 8192, conv                      141.37M enc 132.34M dec 9.03M  -> use ffn 4096 + attn for ~99M instead
#       l-conv* 2 /12, ffn 4096, conv                      82.59M enc 73.55M dec 9.03M  (or 99.1M with attn4 decoder below)
#       l-attn  2 /12, ffn 4096, attn 4 layers             99.10M enc 73.55M dec 25.56M  (decoder-heavy, recommended)
#       l-boom  boombox d512 mult8 depth3                  75.14M enc 66.11M dec 9.03M
#
#   Recommended 9 to run (one per row = size, one per col = decoder):
#     S: s-mlp (9.8M), s-conv (13.3M), s-boom (19.9M floor)
#     M: m-mlp (28.7M), m-conv (30.1M), m-boom (30.8M)  — tightly matched
#     L: l-mlp-mid (79.4M), l-attn (99.1M), l-boom (75.1M)
#   If strict 100M is required, use the 141M conv-8192 or add 2 attn layers;
#   otherwise the 75-99M spread is the best achievable with latent 512 and the
#   current arch without hacking the boombox grid width.
#
#   Batch sizes: transformer freq encoder sees B*100 micro-batch (100 lasers).
#   At 512-d and 1+8 layers, 96 fits 16GB (see param_matched notes). Large
#   2+12 ffn4096 needs 32. Boombox has no blowup, 128 is safe for all.
#
#   tmux new -s bottleneck512
#   ./scripts/bottleneck_512_scaling.sh 2>&1 | tee runs/bottleneck_512_scaling.log

set -u

cd "$(dirname "$0")/.."
export PYTHONPATH=.

exec 9>/tmp/bottleneck_512_scaling.sh.lock
flock -n 9 || { echo "another copy of $(basename "$0") is already running; exiting" >&2; exit 1; }

TAG=b512-v1
GROUP=bottleneck-512-$TAG

COMMON="--split gastronorm --out-h 21 --out-w 30 --augment-mask 0 --augment-fft 0 \
        --loss-fn ce-pixel --max-duration 1000ep \
        --laser-dropout 0 --freq-dropout 0 --wandb-group $GROUP \
        --d-model 512"

# ---- SMALL ~10M (boombox floor 19.9M) ----
# s-mlp: light decoder, encoder-heavy (MAE-style) — 9.79M
python src/run.py $COMMON --batch-size 96 --ffn-dim 2048 --pnt-num-layers 1 --seq-num-layers 2 --pnt-num-heads 8 --seq-num-heads 8 \
    --model transformer --decoder mlp \
    --run-name b512-s-mlp-$TAG

# s-conv: same bottleneck, heavy conv decoder, encoder trimmed to keep total near 10M — 13.27M
python src/run.py $COMMON --batch-size 96 --ffn-dim 1024 --pnt-num-layers 1 --seq-num-layers 1 --pnt-num-heads 8 --seq-num-heads 8 \
    --model transformer --decoder conv \
    --run-name b512-s-conv-$TAG

# s-boom: floor of boombox with latent 512 — 19.87M (cannot reach 10M without thinning grid/decoder, which would break latent=512)
python src/run.py $COMMON --batch-size 128 --freq-mult 1 --freq-depth 1 \
    --model boombox \
    --run-name b512-s-boom-$TAG

# ---- MEDIUM ~30M — tightly matched ----
# m-mlp: 1/8 deep laser encoder, thin head — 28.70M
python src/run.py $COMMON --batch-size 64 --ffn-dim 2048 --pnt-num-layers 1 --seq-num-layers 8 --pnt-num-heads 8 --seq-num-heads 8 \
    --model transformer --decoder mlp \
    --run-name b512-m-mlp-$TAG

# m-conv: width-scaled to offset conv decoder — 30.07M
python src/run.py $COMMON --batch-size 64 --ffn-dim 4096 --pnt-num-layers 2 --seq-num-layers 2 --pnt-num-heads 8 --seq-num-heads 8 \
    --model transformer --decoder conv \
    --run-name b512-m-conv-$TAG

# m-boom: mult4 depth2 — 30.82M
python src/run.py $COMMON --batch-size 128 --freq-mult 4 --freq-depth 2 \
    --model boombox \
    --run-name b512-m-boom-$TAG

# ---- LARGE ~80-100M — decoder-biased (add depth, not just width) ----
# l-mlp-mid: mid-size MLP head adds ~5.5M decoder — 79.36M
python src/run.py $COMMON --batch-size 32 --ffn-dim 4096 --pnt-num-layers 2 --seq-num-layers 12 --pnt-num-heads 8 --seq-num-heads 8 \
    --model transformer --decoder mlp-mid \
    --run-name b512-l-mlp-mid-$TAG

# l-attn: attention decoder 4 layers — decoder-heavy ~25M, total 99.10M (recommended for "decoder matters" test)
python src/run.py $COMMON --batch-size 32 --ffn-dim 4096 --pnt-num-layers 2 --seq-num-layers 12 --pnt-num-heads 8 --seq-num-heads 8 \
    --model transformer --decoder attn --decoder-num-layers 4 --decoder-num-heads 8 \
    --run-name b512-l-attn-$TAG

# l-boom: max achievable with current arch at latent 512 — 75.14M
python src/run.py $COMMON --batch-size 64 --freq-mult 8 --freq-depth 3 \
    --model boombox \
    --run-name b512-l-boom-$TAG
