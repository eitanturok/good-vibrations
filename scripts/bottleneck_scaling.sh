#!/usr/bin/env bash
# scripts/bottleneck_scaling.sh — bottleneck-scaled ladder.
#
#   Latent (bottleneck) scales with size so the bottleneck IS the capacity
#   knob. This isolates width-vs-depth from bottleneck width.
#     S  latent 256  target ~10M
#     M  latent 512  target ~30M
#     L  latent 1024 target ~100M
#   ffn_dim, pnt/seq layers, and decoder depth make up the rest.
#   All counts are trainable params (no RoPE buffers, no BN running stats).
#
#   Model types (3 per size = 9 arms):
#     mlp   : VibrationTransformer + MLPDecoder          (0.23-0.42M head)
#     conv  : VibrationTransformer + Boombox ConvDecoder (7.46M at 256, 9.03M at 512, 12.17M at 1024)
#     boom  : BoomboxModel (conv freq stack + conv grid + conv decoder)
#
#   Encoder / decoder split — literature:
#     MAE / BERT: heavy encoder, light decoder (ViT-L 307M enc / 20M dec).
#                 Decoder is throwaway for representation learning.
#     Dense prediction (SegFormer, Mask2Former, DETR): balanced or
#                 decoder-heavy; spatial reconstruction benefits from decoder.
#     Generative / inverse problems (diffusion U-Net): decoder >= encoder.
#                 Here we reconstruct 630 pixels from one vector, so we bias
#                 large arms toward a bigger decoder. Achieved by adding
#                 decoder depth (more attn/conv layers) not just FFN width —
#                 depth scales better and blows up less memory than width
#                 (freq encoder sees B*100, see param_matched_decoder.sh).
#
#   How shapes were chosen (sum(p.numel() for p in m.parameters())):
#
#     SMALL  latent 256  target ~10M
#       s-mlp-mid  1/2 ffn2048 mlp-mid  9.70M  enc 3.96M dec 5.74M  41/59  (balanced)
#       s-conv     1/1 ffn2048 conv     10.10M enc 2.65M dec 7.46M  26/74  (decoder-heavy)
#       s-boom     256 mult1 depth1     15.94M enc 8.48M dec 7.46M  53/47  <- floor, cannot hit 10M at 256
#
#     MEDIUM latent 512  target ~30M  (tightly matched)
#       m-attn     1/2 ffn4096 attn2    28.72M enc 15.79M dec 12.93M 55/45
#       m-conv     2/2 ffn4096 conv     30.07M enc 21.04M dec  9.03M 70/30
#       m-boom     512 mult4 depth2     30.82M enc 21.79M dec  9.03M 71/29
#
#     LARGE  latent 1024 target ~100M
#       l-mlp      2/4 ffn6144 mlp      101.26M enc 100.83M dec 0.42M  99/1  (enc-heavy control)
#       l-conv     2/8 ffn2048 conv      96.25M enc  84.07M dec 12.17M 87/13
#       l-attn     2/4 ffn2048 attn4    101.52M enc  50.47M dec 51.05M 50/50 (decoder-heavy, recommended)
#       l-boom     1024 mult8 depth3     83.00M enc  70.82M dec 12.17M 85/15  <- max boombox at 1024 without hacking grid
#
#   Recommended 9:
#     S: s-mlp-mid (9.70M), s-conv (10.10M), s-boom (15.94M)
#     M: m-attn (28.72M), m-conv (30.07M), m-boom (30.82M)
#     L: l-mlp (101.26M), l-conv (96.25M), l-attn (101.52M)  [or l-boom 83M for boom large]
#   The 9 in the script below use the balanced variants: s-mlp-mid, s-conv, s-boom,
#   m-attn, m-conv, m-boom, l-attn, l-conv, l-boom. Swap l-boom<->l-mlp if you want
#   a pure-mlp large control.
#
#   Batch sizes sized for 16GB 5080 (freq encoder sees B*100):
#     S 256: 96 (transformer), 128 (boombox)
#     M 512: 64 (transformer), 128 (boombox)
#     L 1024: 32 (transformer), 64 (boombox)
#
#   tmux new -s bottleneck
#   ./scripts/bottleneck_scaling.sh 2>&1 | tee runs/bottleneck_scaling.log

set -u

cd "$(dirname "$0")/.."
export PYTHONPATH=.

exec 9>/tmp/bottleneck_scaling.sh.lock
flock -n 9 || { echo "another copy of $(basename "$0") is already running; exiting" >&2; exit 1; }

TAG=v1
GROUP=bottleneck-$TAG

BASE="--split gastronorm --out-h 21 --out-w 30 --augment-mask 0 --augment-fft 0 \
      --loss-fn ce-pixel --max-duration 1000ep \
      --laser-dropout 0 --freq-dropout 0 --wandb-group $GROUP"

# ---- SMALL  latent 256  ~10M ----
python src/run.py $BASE --d-model 256 --batch-size 96 --ffn-dim 2048 --pnt-num-layers 1 --seq-num-layers 2 --pnt-num-heads 8 --seq-num-heads 8 \
    --model transformer --decoder mlp-mid \
    --run-name b256-s-mlp-mid-$TAG

python src/run.py $BASE --d-model 256 --batch-size 96 --ffn-dim 2048 --pnt-num-layers 1 --seq-num-layers 1 --pnt-num-heads 8 --seq-num-heads 8 \
    --model transformer --decoder conv \
    --run-name b256-s-conv-$TAG

python src/run.py $BASE --d-model 256 --batch-size 128 --freq-mult 1 --freq-depth 1 \
    --model boombox \
    --run-name b256-s-boom-$TAG

# ---- MEDIUM  latent 512  ~30M ----
python src/run.py $BASE --d-model 512 --batch-size 64 --ffn-dim 4096 --pnt-num-layers 1 --seq-num-layers 2 --pnt-num-heads 8 --seq-num-heads 8 \
    --model transformer --decoder attn --decoder-num-layers 2 --decoder-num-heads 8 \
    --run-name b512-m-attn-$TAG

python src/run.py $BASE --d-model 512 --batch-size 64 --ffn-dim 4096 --pnt-num-layers 2 --seq-num-layers 2 --pnt-num-heads 8 --seq-num-heads 8 \
    --model transformer --decoder conv \
    --run-name b512-m-conv-$TAG

python src/run.py $BASE --d-model 512 --batch-size 128 --freq-mult 4 --freq-depth 2 \
    --model boombox \
    --run-name b512-m-boom-$TAG

# ---- LARGE  latent 1024  ~100M ----
python src/run.py $BASE --d-model 1024 --batch-size 32 --ffn-dim 2048 --pnt-num-layers 2 --seq-num-layers 4 --pnt-num-heads 8 --seq-num-heads 8 \
    --model transformer --decoder attn --decoder-num-layers 4 --decoder-num-heads 8 \
    --run-name b1024-l-attn-$TAG

python src/run.py $BASE --d-model 1024 --batch-size 32 --ffn-dim 2048 --pnt-num-layers 2 --seq-num-layers 8 --pnt-num-heads 8 --seq-num-heads 8 \
    --model transformer --decoder conv \
    --run-name b1024-l-conv-$TAG

python src/run.py $BASE --d-model 1024 --batch-size 64 --freq-mult 8 --freq-depth 3 \
    --model boombox \
    --run-name b1024-l-boom-$TAG
