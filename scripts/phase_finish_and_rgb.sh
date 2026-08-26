#!/usr/bin/env bash
# Finish the phase ablation, then train the RGB variant.
#
# Three things, in order:
#   1. P6 (both_w) -- resumes from ep500. It died at epoch 588/1000, killed with no
#      traceback while host RAM sat at 82.6% (8.4 GB free). Its last eval is ep550, so the
#      0.2403 in the ladder is NOT a converged number and P6 is currently not comparable
#      to P1-P5, which all ran the full 1000ep and are flat over their last 4 evals.
#   2. B1 (no phase) and B2 (raw_phase) -- never ran. phase_ablation.sh puts the baselines
#      last and P6 died before reaching them. B2 is the arm that separates "phase hurts"
#      from "the gauge doesn't help": without it a P1/P2 result is unattributable, since
#      it could be the laser referencing OR the mere presence of phase. B1 is a
#      byte-identical rerun of dec-d3-conv-v6 and doubles as a seed-variance check --
#      it should land at ~0.272 2-cube soft-iou / 0.206 hard-iou.
#   3. RGB -- the same boombox predicting the (21,30,3) overhead photo instead of the mask.
#
# run.py sets autoresume=True whenever --run-name is passed, so re-issuing a name RESUMES
# from runs/<name>/checkpoints/latest-rank0.pt rather than restarting. That is what makes
# step 1 a resume and steps 2-3 fresh (no checkpoint dir exists for those names yet).
#
# Under tmux:
#   tmux new -s phasefin
#   ./scripts/phase_finish_and_rgb.sh 2>&1 | tee runs/phase_finish_and_rgb.log

set -u  # deliberately NOT -e: one diverging run should not kill the rest

cd "$(dirname "$0")/.."
export PYTHONPATH=.

# Same lock file as phase_ablation.sh: these write the same run names and checkpoint
# directories, so the two scripts must never overlap on the single GPU.
exec 9>/tmp/phase_ablation.sh.lock
flock -n 9 || { echo "phase_ablation.sh or another copy of this script is already running; exiting" >&2; exit 1; }

TAG=v2
GROUP=phase-ablation-$TAG

# Identical to phase_ablation.sh's COMMON so the arms stay comparable, with one addition:
#   --num-workers 2   P6 was killed at 82.6% host RAM with cpu_memory_shared at 39 GB.
#                     both_w is the widest arm (10 input channels vs 6), so its shards and
#                     dataloader workers are the heaviest in the ladder. Halving workers
#                     from the default 4 cuts the resident copies rather than the model.
COMMON="--split gastronorm --augment-mask 0 --augment-fft 0 \
        --loss-fn mse --model boombox --d-model 512 --batch-size 256 \
        --laser-dropout 0 --freq-dropout 0 --max-duration 1000ep \
        --num-workers 2 --wandb-group $GROUP --signal-mode magnitude"

#***** 1 finish P6 *****
# Resumes from ep500. Runs first: it is the only incomplete cell in the ladder, and until
# it converges the "both gauges" row cannot be read at all.
python src/run.py $COMMON --phase-arm both_w --run-name "phase-p6-both-w-$TAG"

#***** 2 the missing baselines *****

# B1: no phase at all -- THE reference every arm above is measured against.
python src/run.py $COMMON --run-name "phase-b1-none-$TAG"

# B2: ungauged cos/sin of angle(Z). Same channel count as P1-P4 and no gauge, so
# (P1 - B2) isolates the gauge and (B2 - B1) isolates raw phase.
python src/run.py $COMMON --phase-arm raw_phase --run-name "phase-b2-raw-phase-$TAG"

#***** 3 rgb *****
# Predicts the downsampled overhead photo, (21,30,3), instead of the (21,30) mask.
#
# NOT a phase-ablation arm, so it gets its own wandb group. With out_c=3 create_metrics
# (arch.py:172) drops every mask-only metric -- soft-iou, com-distance, localization,
# hard-iou -- because occupancy is meaningless on a photo. What is left is mse on pixel
# values plus the count heads, so this run CANNOT be compared against the 0.27 soft-iou
# numbers above. It is a different task on the same encoder.
#
# Two things run.py forces on this path, both in build_dataset/parse_args rather than here:
#   * --augment-mask is pinned to 0 (dataset.py:892): blur+noise is a mask augmentation.
#   * --loss-fn must be mse or ce-pixel (arch.py:352). mse is what the ladder used.
# The rgb target is its own cache key (dataset.py:72), so expect a one-time downsample of
# all 3008 02_cropped_overhead.png plus a fresh MDS build (~5-10 min) before step 1 of training.
#
# NOTE: --rgb is not supported by the viz/ dashboard or --attribution; inspect this one
# through the wandb media panel, which run.py logs on --viz-interval.
python src/run.py $COMMON --rgb 1 --wandb-group rgb-$TAG --run-name "rgb-boombox-d512-$TAG"
