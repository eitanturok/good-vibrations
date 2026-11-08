#!/usr/bin/env bash
# res decoder + latent-norm rescale (n/n_kept, no u_norm): combines decres-v1 (gastro multi-object recall) and t4-rescale-v1 (cardboard/OOD)
#   tmux new-window -t train -n resrescale "bash scripts/decres_rescale_q.sh; exec zsh"     log: runs/decres_rescale_q.log
cd /home/ethantu/workspace/good-vibrations
export PYTHONUTF8=1
{
while pgrep -f "^[^ ]*python[0-9.]* src9/run.py" > /dev/null; do sleep 60; done
echo "start decres-rescale-v1 $(date)"
.venv/bin/python src9/run.py --model transformer4 --decoder res --latent-norm rescale --run-name decres-rescale-v1 2>&1 | tee runs/decres-rescale-v1.log
echo "decres-rescale done $(date)"
} 2>&1 | tee -a runs/decres_rescale_q.log
