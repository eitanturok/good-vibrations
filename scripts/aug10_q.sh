#!/bin/bash
# decres-rescale with every training augmentation (speaker gain, laser jitter, token dropout) on 10% of samples
cd /home/ethantu/workspace/good-vibrations
while pgrep -f "^[^ ]*python[0-9.]* src9/run.py" > /dev/null; do sleep 60; done
.venv/bin/python src9/run.py --model transformer4 --decoder res --latent-norm rescale --laser-jitter-p 0.1 --speaker-gain-p 0.1 --drop-p 0.1 --run-name decres-rescale-aug10-v1 2>&1 | tee runs/decres-rescale-aug10-v1.log
echo "aug10 done $(date)" >> runs/aug10_q.log
