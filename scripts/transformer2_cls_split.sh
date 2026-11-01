#!/usr/bin/env bash
# src9 transformer2: which [cls] token hurts? 200 ep each (the transformer2 default), one after the other.
# 2x2: both off (gated bands + flatten) | both on | freq_cls only | laser_cls only, + freq_cls with 8 freq layers (the paper's depth)
# freqcls8 uses --microbatch 128: 8 freq layers over 256*100 sequences OOMs a 16 GB GPU at 256 (global batch stays 1024)
cd /home/ethantu/workspace/good-vibrations
PY=.venv/bin/python

$PY src9/run.py --model transformer2 --freq-cls 0 --laser-cls 0 --run-name s9-t2-ref-vase-200
$PY src9/run.py --model transformer2 --freq-cls 1 --laser-cls 1 --run-name s9-t2-cls-vase-200
$PY src9/run.py --model transformer2 --freq-cls 1 --laser-cls 0 --run-name s9-t2-freqcls-vase-200
$PY src9/run.py --model transformer2 --freq-cls 0 --laser-cls 1 --run-name s9-t2-lasercls-vase-200
$PY src9/run.py --model transformer2 --freq-cls 1 --laser-cls 0 --freq-layers 8 --microbatch 128 --run-name s9-t2-freqcls8-vase-200
echo "transformer2 cls split queue done"
