#!/usr/bin/env bash
# follow-on GPU queue: waits for scripts/transformer3_norm.sh (tmux train:norm) to exit, then the eager re-evals, then the sharp-mask queue
# (flowmask-v1, ganloss-v1). one GPU job at a time: also waits until no other src9 training/eval python is running.
#   tmux new-window -t train -n after "bash scripts/after_norm.sh; exec zsh"     log: runs/after_norm.log
cd /home/ethantu/workspace/good-vibrations
{
echo "after_norm: waiting for scripts/transformer3_norm.sh $(date)"
while pgrep -fx "bash scripts/transformer3_norm.sh" > /dev/null; do sleep 60; done
while pgrep -f "python src9/(run|flowmask)\.py" > /dev/null; do echo "waiting: another src9 job on the GPU $(date)"; sleep 60; done
echo "after_norm: norm queue done, starting reeval $(date)"
bash scripts/reeval_all.sh
echo "after_norm: reeval done, starting sharp_q $(date)"
bash scripts/sharp_q.sh
echo "after_norm: all done $(date)"
} 2>&1 | tee -a runs/after_norm.log
