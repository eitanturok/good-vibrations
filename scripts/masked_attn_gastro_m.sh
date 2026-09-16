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
# depth), cut from 10 to 4 -- see note below -- so this is a clean 2x2 (masked-attn, memory-grid)
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
# ***** further speed pass (2026-09-14) *****
# Profiled this config (see perf/TODO.md): dataloading is negligible, torch.compile already gives
# ~1.4x and is already on by default, and this is genuinely compute-bound on the FreqEncoder's
# batch_size*n_lasers=25600-row attention (backward alone is ~65-70% of step time). Per the
# structured-pruning literature (LayerDrop, Fan et al. 2019; layer-dropping studies on pretrained
# transformers, Sajjad et al. 2020), transformer depth is typically the most redundant dimension
# -- more so than FFN width (which Geva et al. 2021 tie to a model's associative/memory capacity)
# or patch size (which is literal input resolution, not redundant compute). So cut
# --pnt-num-layers again, 6 -> 4, rather than touching --enc-ffn-dim or --patch-size.
#
#   tmux new -s masked-attn
#   ./scripts/masked_attn_gastro_m.sh 2>&1 | tee runs/masked_attn_gastro_m.log

# Wrapped in main() so bash parses the whole body into memory up front, instead of reading it
# incrementally from disk as each command finishes -- otherwise editing this file (e.g. jotting
# notes in comments) while an earlier run is still executing shifts line offsets out from under
# the still-running interpreter and aborts the rest of the script with a spurious syntax error
# (hit 2026-09-14: script died after run 1 with "line 89: syntax error near unexpected token `('"
# on a comment line, because the file changed mid-run).
main() {
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH=.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

TAG=v5

# ***** collapse fix (2026-09-14): gastronorm_one_cube's box covers only ~0.65% of the out_h x
# out_w=32x32 grid on average (measured directly -- median 0.59%, 5% of train samples are fully
# empty). Plain ce-pixel (unweighted BCE) rewards "predict background everywhere" so strongly at
# this imbalance that training collapses there: loss -> ~0, gradient L2 norm -> ~0, no learning.
# Switched to ce-pixel-asym (pos_weight = loss-alpha/(1-loss-alpha) on false negatives). alpha=0.97
# (pos_weight~32, chasing the ~153:1 negative:positive pixel ratio) OVERSHOT -- confirmed from a
# live run's metrics (iou~0.02, loss/train/total~0.47, same order of magnitude as the true ~0.65%
# occupancy -- the signature of painting most/all of the grid instead of nothing). alpha=0.5 would
# just reproduce the original background collapse (pos_weight=1, identical to plain ce-pixel), so
# not a real sweep point. Backed off to alpha=0.85 (pos_weight~5.7), an actual new intermediate
# point between the two known collapses -- watch train/bce, iou, and grad-norm; may need further
# tuning. Also raised --max-duration: gastronorm_one_cube has ~5x fewer samples/epoch than full
# gastronorm (504 vs 2553) AND max-duration was cut 2000ep->500ep on top of that, so this was
# getting ~20x fewer total optimizer steps than the runs that worked -- not enough for the model to
# ever find real signal regardless of loss.
COMMON="--model transformer \
        --out-h 32 --out-w 32 --augment-mask 0 --augment-fft 0 \
        --loss-fn ce-pixel-asym --loss-alpha 0.85 --max-duration 5000ep \
        --laser-dropout 0 --freq-dropout 0"

GASTRO="--split gastronorm_one_cube --wandb-group attn-decoder-gastro"

# M size: d_model=128, matches pm_size_ladder.sh's M_ENC except --pnt-num-layers 4 (was 10, then 6
# -- see notes above; freq encoder depth only, laser encoder stays at 10 like the original M preset)
M_ENC="--d-model 128 --pnt-num-layers 3 --seq-num-layers 10 --enc-ffn-dim 256 --pnt-num-heads 4 --seq-num-heads 8 --decoder-num-heads 8"
BATCH=256

# --- 4. masked-attn decoder + memory-grid (reshaped 32x32 memory) ---
python src/run.py $COMMON $GASTRO $M_ENC --batch-size $BATCH \
    --decoder masked-attn --decoder-num-layers 18 --dec-ffn-dim 1024 --mask-temp 4.0 --memory-grid 1 \
    --run-name attn-decoder-gastro-masked-memgrid-$TAG

# --- 3. masked-attn decoder, raw laser memory (same layer/width budget) ---
python src/run.py $COMMON $GASTRO $M_ENC --batch-size $BATCH \
    --decoder masked-attn --decoder-num-layers 18 --dec-ffn-dim 1024 --mask-temp 4.0 \
    --run-name attn-decoder-gastro-masked-$TAG

# --- 1. baseline: plain attn decoder, raw laser memory ---
python src/run.py $COMMON $GASTRO $M_ENC --batch-size $BATCH \
    --decoder attn --decoder-num-layers 18 --dec-ffn-dim 1024 \
    --run-name attn-decoder-gastro-attn-$TAG

# --- 2. attn decoder + memory-grid (reshaped 32x32 memory, no masked-attn) ---
python src/run.py $COMMON $GASTRO $M_ENC --batch-size $BATCH \
    --decoder attn --decoder-num-layers 18 --dec-ffn-dim 1024 --memory-grid 1 \
    --run-name attn-decoder-gastro-attn-memgrid-$TAG

# --- 5. plain attn decoder + shared-x RoPE keys (queries = wall x-z plane, keys = 100 lasers on
# the floor x-y plane; both share the same freqs_x channels so matching-x query/key pairs get a
# free, unlearned "same x -> related" bias -- see arch.py build_shared_x_freqs). The laser grid
# covers only ~1/3 of the wall's x-width and isn't exactly aligned yet -- --laser-x-span/-offset
# are a simple baseline calibration (approximate, not from real box geometry), meant to be
# refined/scaled up later. Incompatible with --memory-grid (reshaped memory isn't laser-indexed).
python src/run.py $COMMON $GASTRO $M_ENC --batch-size $BATCH \
    --decoder attn --decoder-num-layers 18 --dec-ffn-dim 1024 --key-pos shared-rope --laser-x-span 0.333 --laser-x-offset 0.0 \
    --run-name attn-decoder-gastro-attn-sharedrope-$TAG
}
main "$@"


