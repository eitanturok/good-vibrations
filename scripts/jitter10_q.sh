#!/bin/bash
# decres-rescale-v1 with laser jitter on 10% of training samples
cd /home/ethantu/workspace/good-vibrations
while pgrep -f "^[^ ]*python[0-9.]* src9/run.py" > /dev/null; do sleep 60; done
.venv/bin/python src9/run.py --model transformer4 --decoder res --latent-norm rescale --laser-jitter-p 0.1 --run-name decres-rescale-jit10-v1 2>&1 | tee runs/decres-rescale-jit10-v1.log
echo "jit10 done $(date)" >> runs/jitter10_q.log
