#!/bin/bash
# laser-count ablation of decres-rescale-aug10-v1: same config / seed / init, only --lasers (kept subset of the 10x10 grid) changes
# subsets: runs/laser_ablation/subsets.json (scripts/laser_subsets.py); runs lasers-<grid|importance|random>-<K>-v1; analysis notebooks/142_laser_ablation.ipynb
# one queue: the one 16 GB GPU is ~97-99% busy even at K=1 and two runs OOM; small K first (fast); runs with the ep300 checkpoint are skipped
cd /home/ethantu/workspace/good-vibrations
B="--model transformer4 --decoder res --latent-norm rescale --laser-jitter-p 0.1 --speaker-gain-p 0.1 --laser-drop-p 0.1 --band-drop-p 0.1"  # = aug10's --drop-p 0.1 (pre-472c9f5 flag)
for k in 1 5 10 20 40 60 80; do for s in grid importance random; do
  n=lasers-$s-$k-v1
  [ -e runs/$n/checkpoints/ep300-ba3000-rank0.pt ] && continue
  while pgrep -f "^[^ ]*python[0-9.]* src9/run.py" > /dev/null; do sleep 60; done
  ids=$(.venv/bin/python -c "import json; print(','.join(map(str, json.load(open('runs/laser_ablation/subsets.json'))['$s']['$k'])))")
  .venv/bin/python src9/run.py $B --lasers $ids --run-name $n > runs/$n.log 2>&1
  echo "$n exit $? $(date)" >> runs/laser_ablation/queue.log
done; done
