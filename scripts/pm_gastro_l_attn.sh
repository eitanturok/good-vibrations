#!/usr/bin/env bash
# Re-run of the one arm of pm_size_ladder.sh that died last time: pm-gastro-l-attn-v2.
#
# It failed in the pre-train boundary eval with a "non-addressable cuda error while using auto
# microbatching" -- an OOM composer couldn't recover from because a *second* run (gastro-l-conv)
# was holding ~14 GiB of the 15.5 GiB card at the same time. Two fixes, both applied:
#   1. --device-eval-microbatch-size now defaults to "auto" (src/run.py), so the eval pass
#      grad-splits down instead of crashing -- matching what training already did.
#   2. this script WAITS for pm-gastro-l-conv-v2 to finish before starting, so the L attn run
#      gets the whole GPU to itself.
#
#   tmux new -s pm-attn
#   ./scripts/pm_gastro_l_attn.sh 2>&1 | tee runs/pm_gastro_l_attn.log

set -u
cd "$(dirname "$0")/.."
export PYTHONPATH=.

WAIT_FOR=pm-gastro-l-conv-v2
echo "$(date '+%F %T')  waiting for $WAIT_FOR to finish before launching pm-gastro-l-attn-v2 ..."
while pgrep -f "src/run.py .*--run-name $WAIT_FOR" >/dev/null; do sleep 60; done
echo "$(date '+%F %T')  $WAIT_FOR done -- launching pm-gastro-l-attn-v2"

# byte-identical to the attn-L / gastro arm of pm_size_ladder.sh (COMMON + GASTRO + L_ENC + attn)
exec python src/run.py \
    --model transformer --out-h 21 --out-w 30 --augment-mask 0 --augment-fft 0 \
    --loss-fn ce-pixel --max-duration 2000ep --laser-dropout 0 --freq-dropout 0 \
    --split gastronorm --wandb-group pm-v2-gastro \
    --d-model 512 --pnt-num-layers 12 --seq-num-layers 12 --enc-ffn-dim 1024 \
    --pnt-num-heads 8 --seq-num-heads 8 --decoder-num-heads 8 \
    --batch-size 64 --decoder attn --decoder-num-layers 24 --dec-ffn-dim 2048 \
    --run-name pm-gastro-l-attn-v2
