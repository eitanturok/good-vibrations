#!/usr/bin/env bash
# Speaker-generalization ablation for the regular conv-conv boombox model (--model boombox
# --encoder single, ce-pixel loss) on the full gastronorm split (empty/1/2/3-cube, standardized
# 32x32 output grid). Answers two separate questions:
#   (a) single-speaker floor -- how well does the model do trained on just ONE speaker's data,
#       for each of the 8 speakers independently?
#   (b) unseen-speaker generalization -- if speaker 4 is held OUT of training entirely, does the
#       model still segment its eval samples reasonably, vs. the same speaker subset with 4 folded
#       into training?
#
# EVERY run below evals on ALL 8 speakers, regardless of which speakers it trained on -- so every
# run uses --split gastronorm_speaker_gen (src/model/dataset.py) with --eval-speakers fixed at
# 1,2,...,8, and only --train-speakers varies per arm. Whichever speakers a given run did NOT
# train on land in eval/{n}obj-unseen-speaker buckets (grouped by object count, combined across
# however many speakers are unseen for that run -- e.g. the single-speaker runs below have 7
# unseen speakers lumped into one such bucket per object count, not broken out individually).
# The normal eval/1-cube, eval/2-cubes, eval/3-cubes buckets still cover the TRAINED speakers only,
# exactly as plain --split gastronorm would.
#
# 13 runs total, sequential (one GPU):
#   1.  baseline        -- train on all 8 speakers (so eval-speakers == train-speakers, no unseen
#                           leg at all -- everything falls into the normal buckets)
#   2-9 spk-<N>-8k       -- speakers 1..8 individually, 8000 epochs each (small per-speaker sample
#                           count needs --batch-size 256, not the usual 1024 -- see
#                           scripts/boombox_speaker_5k.sh's drop_last=True gotcha: at 1024 every
#                           epoch's loader silently ran zero batches). 7/8 speakers unseen.
#   10. spk-1234         -- fixed train subset {1,2,3,4}; speakers 5,6,7,8 unseen
#   11. spk-3456         -- fixed train subset {3,4,5,6}; speakers 1,2,7,8 unseen
#   12. spk-356-unseen4  -- train on {3,5,6} only; speakers 1,2,4,7,8 unseen (4 is the one we
#                           actually care about here -- compare its eval/{n}obj-unseen-speaker
#                           rows against run 13's, which also holds out 4 but from a much wider
#                           train set)
#   13. spk-7of8-unseen4 -- train on {1,2,3,5,6,7,8} (every speaker except 4); speaker 4 unseen,
#                           same question as run 12 but with 7/8 speakers seen instead of 3/4, to
#                           see whether wider train speaker coverage narrows the unseen-speaker gap
#
# All runs use the same 8000-epoch budget and batch size so arms are comparable end-to-end, not
# just per-epoch.
#
#   TAG=v2 ./scripts/boombox_speaker_gen.sh 2>&1 | tee runs/boombox_speaker_gen.log

set -eu
cd "$(dirname "$0")/.."
export PYTHONPATH=.
# torch DataLoader workers leak fds (one per tensor handed back to the main process, held open
# until consumed) -- the inherited default (1024 here) gets exhausted over a long multi-run sweep
# and silently deadlocks the process (a worker's feeder thread dies with "Too many open files",
# but that exception never propagates to the main process -- it just hangs forever). Bit us on
# run 2/13 (bb-spk-1-8k-v1) of this exact script on 2026-09-22. Raise it up front.
ulimit -n 65536

TAG="${TAG:-v1}"
GROUP="boombox-speaker-gen-$TAG"
ALL_SPEAKERS="1,2,3,4,5,6,7,8"

COMMON="--model boombox --encoder single --d-model 512 --batch-size 256 \
        --data-dir experiments/31_07_2026_gastronorm_exp1 --split gastronorm_speaker_gen \
        --eval-speakers $ALL_SPEAKERS \
        --augment-mask 0 --augment-fft 0 --loss-fn ce-pixel --max-duration 8000ep \
        --laser-dropout 0 --freq-dropout 0 --wandb-group $GROUP"

run() { PYTHONPATH=. python src/run.py $COMMON "$@"; }

# 1. baseline: all 8 speakers -- eval-speakers == train-speakers, so no unseen leg
run --train-speakers "$ALL_SPEAKERS" --run-name "bb-spk-baseline-$TAG"

# 2-9. each speaker in isolation; the other 7 are unseen
for spk in 1 2 3 4 5 6 7 8; do
    run --train-speakers "$spk" --run-name "bb-spk-$spk-8k-$TAG"
done

# 10-11. fixed train subsets; the remaining 4 speakers each time are unseen
run --train-speakers 1,2,3,4 --run-name "bb-spk-1234-$TAG"
run --train-speakers 3,4,5,6 --run-name "bb-spk-3456-$TAG"

# 12. train on {3,5,6} only -- speaker 4 (among others) unseen
run --train-speakers 3,5,6 --run-name "bb-spk-356-unseen4-$TAG"

# 13. train on every speaker but 4 -- speaker 4 unseen, wider train coverage than run 12
run --train-speakers 1,2,3,5,6,7,8 --run-name "bb-spk-7of8-unseen4-$TAG"
