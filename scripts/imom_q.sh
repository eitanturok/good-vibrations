#!/usr/bin/env bash
# imom-d30-v1 = drop30-v1 (default transformer3: per, d256, mb128, laser / band drop .3, 300 ep) + hybrid's imoment + edge terms, weight 1 each:
#   --imom-w 1.0:    utils.instance_moment_loss(p, inst > .5)
#   --outline-w 1.0: hybrid's edge bce_dice, the predicted outline = utils.outline(p) (3x3 dilation - erosion of the mask probs, no 2nd decoder channel)
# compare: drop30-v1-reeval ID .671 / OOD .386.   waits for the NaN verification (scripts/nan_debug_q.sh) and no src9/run.py.   tmux train:imom   log: runs/imom-d30-v1.log
cd /home/ethantu/workspace/good-vibrations
PY=.venv/bin/python
export PYTHONUTF8=1
until grep -q "nan_verify queue done" runs/nan_verify_q.log 2>/dev/null; do sleep 60; done
while pgrep -f "python src9/(run|flowmask)\.py" > /dev/null; do echo "imom: waiting, another src9 job running $(date)"; sleep 60; done
echo "imom: starting imom-d30-v1 $(date)"
$PY src9/run.py --model transformer3 --laser-drop 0.3 --band-drop 0.3 --imom-w 1.0 --outline-w 1.0 --run-name imom-d30-v1 2>&1 | tee runs/imom-d30-v1.log
echo "imom: done $(date)"
