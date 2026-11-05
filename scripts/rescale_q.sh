#!/usr/bin/env bash
# transformer4 --latent-norm rescale: no LN on the latent u; u = (L / sum_l m_l) sum_l m_l W_l LN(t_l) + b (the token-dropout mean over kept lasers, factor 1 at eval).
#   t4-rescale-v1: decoder two (default), 300 ep, eager, mb128, wandb. compare: t4-nowmask-v1 (ID .678 / OOD .383).
# waits for "xattn done" (xattn_q.sh) + no src9 run.py.
#   tmux new-window -t train -n rescale "bash scripts/rescale_q.sh; exec zsh"     log: runs/rescale_q.log
cd /home/ethantu/workspace/good-vibrations
PY=.venv/bin/python
export PYTHONUTF8=1
{
echo "rescale: waiting $(date)"
until grep -q "xattn done" runs/xattn_q.log 2>/dev/null; do sleep 60; done
while pgrep -f "^[^ ]*python[0-9.]* src9/run.py" > /dev/null; do sleep 60; done
echo "rescale: t4-rescale-v1 $(date)"
$PY src9/run.py --model transformer4 --latent-norm rescale --run-name t4-rescale-v1 2>&1 | tee runs/t4-rescale-v1.log
echo "rescale done $(date)"
} 2>&1 | tee -a runs/rescale_q.log
