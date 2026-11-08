#!/bin/bash
# decres-rescale aug ablation: speaker gain (sg), laser jitter (jit), laser token drop (ld), band token drop (bd); p = fraction of samples augmented
cd /home/ethantu/workspace/good-vibrations
B="--model transformer4 --decoder res --latent-norm rescale"
run() { n=$1; shift
  while pgrep -f "^[^ ]*python[0-9.]* src9/run.py" > /dev/null; do sleep 60; done
  .venv/bin/python src9/run.py $B "$@" --run-name $n 2>&1 | tee runs/$n.log
  echo "$n done $(date)" >> runs/aug_ablate_q.log; }
run decres-rescale-sg10-v1  --speaker-gain-p 0.1
run decres-rescale-jit10-v2 --laser-jitter-p 0.1
run decres-rescale-ld10-v1  --laser-drop-p 0.1
run decres-rescale-bd10-v1  --band-drop-p 0.1
run decres-rescale-sgjit10-v1 --speaker-gain-p 0.1 --laser-jitter-p 0.1
run decres-rescale-drop10-v1  --laser-drop-p 0.1 --band-drop-p 0.1
run decres-rescale-noaug-v1 --speaker-gain-p 0 --laser-jitter-p 0 --laser-drop-p 0 --band-drop-p 0
