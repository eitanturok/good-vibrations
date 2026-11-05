#!/bin/bash
# transformer4 decoder ablation vs t4-nowmask-v1 (--decoder two, ID .678 / OOD .383): all defaults (eager, 300 ep, wandb), only --decoder changes.
# one = single-branch UpOne, res = pre-act GN/silu residual UpRes (param-matched), res-coord = res + CoordConv. Waits until no src9 run.py is running.
cd /home/ethantu/workspace/good-vibrations
while pgrep -f "^[^ ]*python[0-9.]* src9/run.py" > /dev/null; do sleep 60; done
for rd in "dec1b-v1 one" "decres-v1 res" "decres-coord-v1 res-coord"; do
  set -- $rd
  .venv/bin/python src9/run.py --model transformer4 --decoder $2 --run-name $1 2>&1 | tee runs/$1.log
done
echo "dec done"
