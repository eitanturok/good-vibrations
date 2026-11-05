#!/bin/bash
# t4-nowmask-v1 = t4-v1 (transformer4, all defaults: eager, 300 ep, wandb) with --w-mask 0 (no W_mask [m_l, m_b] term). Refuses to start if another src9 run.py is running.
cd /home/ethantu/workspace/good-vibrations
if pgrep -af "^[^ ]*python[0-9.]* src9/run.py"; then echo "another src9 run.py is running: not starting"; exit 1; fi
.venv/bin/python src9/run.py --model transformer4 --w-mask 0 --run-name t4-nowmask-v1 2>&1 | tee runs/t4-nowmask-v1.log
echo "nowmask done"
