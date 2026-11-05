#!/usr/bin/env bash
# tmux train:redo: the LAST GPU job. waits for the readout queue (after_norm2.sh) and then the tok queue (tmux train:tok -> scripts/tok_q.sh, which
# echoes "tok queue done" into runs/tok_q.log) to finish, and no src9 training on the GPU, then scripts/nan_debug_q.sh.
#   tmux new-window -t train -n redo "bash scripts/redo_wait.sh; exec zsh"     log: runs/nan_debug_q.log
cd /home/ethantu/workspace/good-vibrations
{
echo "redo: waiting for after_norm2.sh + tok_q.sh $(date)"
until ! pgrep -fx "bash scripts/after_norm2.sh" > /dev/null && grep -q "tok queue done" runs/tok_q.log 2>/dev/null && ! pgrep -fx "bash scripts/tok_q.sh" > /dev/null; do sleep 60; done
while pgrep -f "python src9/(run|flowmask)\.py" > /dev/null; do echo "redo: waiting, another src9 job on the GPU $(date)"; sleep 60; done
echo "redo: starting nan_debug_q $(date)"
bash scripts/nan_debug_q.sh
echo "redo: all done $(date)"
} 2>&1 | tee -a runs/nan_debug_q.log
