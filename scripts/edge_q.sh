#!/usr/bin/env bash
# mask-sharpening runs, drop30 config (transformer3, laser/band drop .3), 300 ep, existing flags only. eager (transformer3 default compile=0);
# baselines drop30-v1 / base300-v1 were trained compiled.
#   sdf1-d30-v1:   --boundary-w 1      + mean(p * sdf) (Kervadec boundary loss; sdf > 0 outside, < 0 inside, image heights), ramped 0 -> 1 over epochs 0-50
#   lovasz-d30-v1: --mask-loss lovasz  mask = 0.5 bce + Lovasz hinge (instead of the dice term)
#   dice08-d30-v1: --dice-w 0.8        mask = 0.2 bce + 0.8 dice (default 0.5 / 0.5)
# Waits for "t4 done" in runs/t4_q.log AND no src9 run.py.
#   tmux new-window -t train -n edge "bash scripts/edge_q.sh; exec zsh"     log: runs/edge_q.log
cd /home/ethantu/workspace/good-vibrations
export PYTHONUTF8=1
{
until grep -q "t4 done" runs/t4_q.log 2>/dev/null && ! pgrep -f "python src9/run.py" > /dev/null; do echo "edge: waiting for t4 done + no src9/run.py $(date)"; sleep 60; done
for spec in "sdf1-d30-v1 --boundary-w 1" "lovasz-d30-v1 --mask-loss lovasz" "dice08-d30-v1 --dice-w 0.8"; do
    set -- $spec; name=$1; shift
    echo "edge: starting $name ($*) $(date)"
    .venv/bin/python src9/run.py --model transformer3 --laser-drop 0.3 --band-drop 0.3 "$@" --run-name $name 2>&1 | tee runs/$name.log
done
echo "edge done $(date)"
} 2>&1 | tee -a runs/edge_q.log
