#!/usr/bin/env bash
# follow-on GPU queue (replaces after_norm.sh; flowmask-v1 / ganloss-v1 dropped, sharp_q.sh left on disk unqueued):
# waits for scripts/transformer3_norm.sh (tmux train:norm) to exit and no other src9 training/eval python, then the eager re-evals, then the readout queue.
#   tmux new-window -t train -n after "bash scripts/after_norm2.sh; exec zsh"     log: runs/after_norm2.log
cd /home/ethantu/workspace/good-vibrations
{
echo "after_norm2: waiting for scripts/transformer3_norm.sh $(date)"
while pgrep -fx "bash scripts/transformer3_norm.sh" > /dev/null; do sleep 60; done
while pgrep -f "python src9/(run|flowmask)\.py" > /dev/null; do echo "waiting: another src9 job on the GPU $(date)"; sleep 60; done
echo "after_norm2: norm queue done, starting reeval $(date)"
bash scripts/reeval_all.sh
echo "after_norm2: reeval done, starting readout_q $(date)"
bash scripts/readout_q.sh
echo "after_norm2: all done $(date)"
} 2>&1 | tee -a runs/after_norm2.log
