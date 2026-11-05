#!/usr/bin/env bash
# imom-d30-v1 config (drop .3, --imom-w 1 --outline-w 1) trained 1000 ep (fresh run, full cosine schedule). t4-v1 waits for "imom1k done".
#   tmux new-window -t train -n imom1k "bash scripts/imom1k_q.sh; exec zsh"     log: runs/imom1k_q.log
cd /home/ethantu/workspace/good-vibrations
PY=.venv/bin/python
export PYTHONUTF8=1
{
while pgrep -f "python src9/(run|flowmask)\.py" > /dev/null; do echo "imom1k: waiting, another src9 job running $(date)"; sleep 60; done
echo "imom1k: starting imom-d30-1k-v1 $(date)"
$PY src9/run.py --model transformer3 --laser-drop 0.3 --band-drop 0.3 --imom-w 1.0 --outline-w 1.0 --max-duration 1000ep --run-name imom-d30-1k-v1 2>&1 | tee runs/imom-d30-1k-v1.log
echo "imom1k done $(date)"
} 2>&1 | tee -a runs/imom1k_q.log
