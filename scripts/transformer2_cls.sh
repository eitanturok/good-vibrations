#!/usr/bin/env bash
# src9 transformer2 stage 2: cls=0 (stage-1 reference: gate averaging + flatten) vs cls=1 (freq_cls + laser_cls tokens), one after the other.
# 1000 ep to match the baseline s9-transformer-vase-1k (every other arg is already the transformer2 default = the baseline's)
cd /home/ethantu/workspace/good-vibrations
PY=.venv/bin/python

$PY src9/run.py --model transformer2 --cls 0 --max-duration 1000ep --run-name s9-t2-ref-vase-1k
$PY src9/run.py --model transformer2 --cls 1 --max-duration 1000ep --run-name s9-t2-cls-vase-1k
echo "transformer2 cls queue done"
