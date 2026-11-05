#!/usr/bin/env bash
# transformer4 --freq-rope log: freq-block rope positions = log band-center Hz rescaled to the band-index span 0..17 (mass loading: df ~ f).
#   t4-logrope-v1: all other transformer4 defaults (decoder two, latent-norm ln, 300 ep, eager, mb128, wandb). compare: t4-nowmask-v1 (ID .678 / OOD .383).
# waits for "rescale done" (rescale_q.sh, the last of dec_q -> dec_oom_redo -> xattn_q -> rescale_q) + no src9 run.py.
#   tmux new-window -t train -n logrope "bash scripts/logrope_q.sh; exec zsh"     log: runs/logrope_q.log
cd /home/ethantu/workspace/good-vibrations
PY=.venv/bin/python
export PYTHONUTF8=1
{
echo "logrope: waiting $(date)"
until grep -q "rescale done" runs/rescale_q.log 2>/dev/null; do sleep 60; done
while pgrep -f "^[^ ]*python[0-9.]* src9/(run|flowmask)\.py" > /dev/null; do sleep 60; done
echo "logrope: t4-logrope-v1 $(date)"
$PY src9/run.py --model transformer4 --freq-rope log --run-name t4-logrope-v1 2>&1 | tee runs/t4-logrope-v1.log
echo "logrope done $(date)"
} 2>&1 | tee -a runs/logrope_q.log
