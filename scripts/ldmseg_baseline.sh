#!/usr/bin/env bash
# Full training run for LDMSegVibrationModel (src5/): vibration encoder + boombox decoder,
# warm-started from pp-baseline-v2, trained to land in the frozen LDMSeg mask autoencoder's
# latent space so its pretrained decoder produces the segmentation mask directly.
#
# --lr matches the paper's own AE training (Van Gansbeke & Van Gool, ECCV 2024: lr=1e-4,
# AdamW) -- see the plan for why this is a reference point, not a guarantee of transferring
# perfectly to our much larger/different vibration encoder; revisit if training is unstable.
#
# Logged (see src5/latent_model.py for the metric/loss definitions):
#   - latent_l2, latent_cos: mean L2 distance / cosine similarity between z_pred and z_gt
#   - iou, contour, mass: MaskMetric suite on the downsampled decoded prediction vs mask_true
#     (contour only on eval, per src/model/arch.py's CHEAP_SEG_KEYS/SEG_KEYS split)
#   - loss/train/{ce,mask,latent,total}: each loss term separately (Composer auto-logs a
#     dict-valued loss()'s keys, see src5/latent_model.py's loss() docstring)
#   - l2_norm/grad/global: L2 grad norm over all weights (OptimizerMonitor)
#   - throughput/samples_per_sec, time/batch: step time / throughput (SpeedMonitor, built-in)
#
# --compile (torch.compile encoder+decoder) and the default 256x256 decode resolution (skips the
# frozen AE's final non-learned bilinear upsample to 512) together take real Composer training
# throughput from ~55 to ~85-89 samples/sec -- see src5/SPEEDUP_LOG.md for the measurements.
# Pass --full-res-decode to restore 512 (slower, more precise) for comparison.
#
# --log-points logs SMaskPoints/{split}: the same predicted-vs-true panels, overlaid with a
# density heatmap of the actual PointRend sample locations used for that step's ce/bce/dice loss
# (see src5/viz.py's VisualizePointRend) -- on the same --viz-interval schedule.
#
# --augment-fft 0 --augment-mask 0: no data augmentation (build_dataset's random_frequency_gain /
# noisy_blur are both off by default at 0.5 probability each) -- every train sample is the raw
# fft/mask, unmodified.
#
#   TAG=v2 ./scripts/ldmseg_baseline.sh   # bump TAG for a fresh run name without editing this file

set -u
cd "$(dirname "$0")/.."
export PYTHONPATH=.
# torch's DataLoader workers leak file descriptors (one per tensor handed back to the main
# process, held open until consumed) -- the inherited default of 1024 gets exhausted over a long
# run and deadlocks the training process silently (a worker's feeder thread dies with "Too many
# open files", but that exception never propagates to the main process, which just hangs forever
# waiting on a queue that will never get fed again). Raise it up front.
ulimit -n 65536

TAG="${TAG:-v9}"

python src5/run.py \
    --lr 1e-4 \
    --max-duration 2000ep \
    --batch-size 8 \
    --eval-batch-size 8 \
    --compile \
    --log-points \
    --augment-fft 0 \
    --augment-mask 0 \
    --run-name "ldmseg-baseline-$TAG"
