#!/bin/bash
# decres-rescale aug isolation: listed augs on 10% of samples, all others OFF (0). sg = speaker gain, jit = laser shift, drop = laser + band token dropout
cd /home/ethantu/workspace/good-vibrations
B="--model transformer4 --decoder res --latent-norm rescale"
SG="--speaker-gain-p"; JIT="--laser-jitter-p"; DROP() { echo "--laser-drop-p $1 --band-drop-p $1"; }
run() { n=$1; shift
  while pgrep -f "^[^ ]*python[0-9.]* src9/run.py" > /dev/null; do sleep 60; done
  .venv/bin/python src9/run.py $B "$@" --run-name $n 2>&1 | tee runs/$n.log
  echo "$n done $(date)" >> runs/aug_iso_q.log; }
run decres-rescale-only-sg10-v1      $SG 0.1 $JIT 0   $(DROP 0)
run decres-rescale-only-jit10-v1     $SG 0   $JIT 0.1 $(DROP 0)
run decres-rescale-only-drop10-v1    $SG 0   $JIT 0   $(DROP 0.1)
run decres-rescale-only-sgjit10-v1   $SG 0.1 $JIT 0.1 $(DROP 0)
run decres-rescale-only-sgdrop10-v1  $SG 0.1 $JIT 0   $(DROP 0.1)
run decres-rescale-only-jitdrop10-v1 $SG 0   $JIT 0.1 $(DROP 0.1)
