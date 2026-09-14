#!/usr/bin/env bash
# Does Mask2Former-style masked attention help the attn decoder, at the M (medium, ~30M param)
# size, on gastronorm? And does reshaping the laser memory onto a 32x32 grid (--memory-grid) help
# either one? 4 runs = 2x2 (masked-attn on/off) x (memory-grid on/off), everything else identical.
#
# Masked attention here (model/arch.py MaskedAttnDecoder) is adapted from Mask2Former's original
# mechanism to this task's query/memory domain mismatch: output-grid queries (32x32 cells) and
# laser-token memory (100 lasers) share no spatial index, unlike Mask2Former's instance queries
# and same-resolution pixel features. So instead of a literal per-query per-pixel include/exclude
# mask, each layer predicts an intermediate per-query mask logit, and that query's own confidence
# (tanh of the logit) becomes an additive bias on ITS cross-attention logits in the next layer --
# confident queries attend sharply, unconfident ones stay diffuse. Cross-attention only, matching
# Mask2Former (self-attention among queries stays global so queries can still coordinate).
#
# --memory-grid (model/arch.py MemoryReshaper) projects the L+1 laser-token memory onto a NEW
# learned out_h x out_w grid of tokens -- 2D RoPE'd over that SAME out_h x out_w grid the decoder
# queries live on -- via one extra cross-attn layer, before the decoder's own cross-attn stack
# runs. That gives Q and K/V matching sequence length (1024=1024) and a shared spatial index
# (query i and reshaped-memory i are both "output grid cell i"), instead of cross-attending
# straight into the raw 10x10-laser-grid memory. Composes with masked-attn: masked-attn's per-query
# confidence bias still gates cross-attention the same way, just into the reshaped grid memory.
#
# Encoder/size preset (M) matches pm_size_ladder.sh's M_ENC EXCEPT --pnt-num-layers (freq encoder
# depth), cut from 10 to 6 -- see note below -- so this is a clean 2x2 (masked-attn, memory-grid)
# comparison, just at a faster/cheaper freq-encoder depth than the original ladder used.
#
# ***** why batch size was stuck and what changed *****
# FreqEncoder runs batch_size * n_lasers sequences through itself every step (100 gastronorm
# lasers -> 128*100 = 12800 effective batch at the ladder's batch-size 128), NOT batch_size like
# the laser encoder and decoder (both stay at batch_size). That 100x blowup in activation memory
# (not parameter memory -- freq/laser encoder param counts are ~equal, ~5.3M each) is what was
# capping batch size. decoder-num-layers/dec-ffn-dim (the actual masked-attn/memory-grid variable)
# is the single biggest param bucket (~19M of ~30M) but runs at batch_size, so it was never the
# bottleneck. Fix: cut ONLY --pnt-num-layers (freq encoder depth, 10->6) -- d_model, laser encoder,
# and decoder stay exactly as the M preset specifies, so the 2x2 comparison is unaffected. That
# frees enough activation memory to raise batch size 128 -> 256.
#
#   tmux new -s masked-attn
#   ./scripts/masked_attn_gastro_m.sh 2>&1 | tee runs/masked_attn_gastro_m.log

set -u
cd "$(dirname "$0")/.."
export PYTHONPATH=.

TAG=v1

COMMON="--model transformer \
        --out-h 21 --out-w 30 --augment-mask 0 --augment-fft 0 \
        --loss-fn ce-pixel --max-duration 2000ep \
        --laser-dropout 0 --freq-dropout 0"

GASTRO="--split gastronorm --wandb-group masked-attn-gastro-m"

# M size: d_model=256, matches pm_size_ladder.sh's M_ENC except --pnt-num-layers 6 (was 10 -- see
# note above; freq encoder depth only, laser encoder stays at 10 like the original M preset)
M_ENC="--d-model 256 --pnt-num-layers 6 --seq-num-layers 10 --enc-ffn-dim 512 --pnt-num-heads 8 --seq-num-heads 8 --decoder-num-heads 8"
BATCH=256

# --- 1. baseline: plain attn decoder, raw laser memory ---
python src/run.py $COMMON $GASTRO $M_ENC --batch-size $BATCH \
    --decoder attn --decoder-num-layers 18 --dec-ffn-dim 1024 \
    --run-name masked-attn-gastro-m-attn-$TAG

# --- 2. attn decoder + memory-grid (reshaped 32x32 memory, no masked-attn) ---
python src/run.py $COMMON $GASTRO $M_ENC --batch-size $BATCH \
    --decoder attn --decoder-num-layers 18 --dec-ffn-dim 1024 --memory-grid 1 \
    --run-name masked-attn-gastro-m-attn-memgrid-$TAG

# --- 3. masked-attn decoder, raw laser memory (same layer/width budget) ---
python src/run.py $COMMON $GASTRO $M_ENC --batch-size $BATCH \
    --decoder masked-attn --decoder-num-layers 18 --dec-ffn-dim 1024 --mask-temp 4.0 \
    --run-name masked-attn-gastro-m-masked-$TAG

# --- 4. masked-attn decoder + memory-grid (reshaped 32x32 memory) ---
python src/run.py $COMMON $GASTRO $M_ENC --batch-size $BATCH \
    --decoder masked-attn --decoder-num-layers 18 --dec-ffn-dim 1024 --mask-temp 4.0 --memory-grid 1 \
    --run-name masked-attn-gastro-m-masked-memgrid-$TAG
