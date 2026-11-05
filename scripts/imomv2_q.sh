#!/usr/bin/env bash
# imomv2: scale-invariant imoment (utils.instance_moment_loss(scale=True)) ramped in after epoch 100 (full at 150), no outline. drop30 config, 300 ep.
# imom-w 0.01: measured on drop30-v1 ep100 (128 train samples, CPU fp32) -> scaled-imoment grad = 0.78x the mask grad (0.0064 would be 0.5x).
# eager (transformer3 default compile=0 now); baselines drop30-v1 / base300-v1 were trained compiled (~1.17 s/it vs ~1.6 s/it eager).
# (run A imom01-r100-d30-v1 dropped.) Waits for "imom1k done" in runs/imom1k_q.log AND no src9 run.py. t4-v1 waits for "imomv2 done".
#   tmux new-window -t train -n imomv2 "bash scripts/imomv2_q.sh; exec zsh"     log: runs/imomv2_q.log
cd /home/ethantu/workspace/good-vibrations
export PYTHONUTF8=1
{
until grep -q "imom1k done" runs/imom1k_q.log 2>/dev/null && ! pgrep -f "python src9/run.py" > /dev/null; do echo "imomv2: waiting for imom1k done + no src9/run.py $(date)"; sleep 60; done
echo "imomv2: starting imomsi-r100-d30-v1 $(date)"
.venv/bin/python src9/run.py --model transformer3 --laser-drop 0.3 --band-drop 0.3 --imom-norm scale --imom-w 0.01 --outline-w 0 --imom-start 100 --run-name imomsi-r100-d30-v1 2>&1 | tee runs/imomsi-r100-d30-v1.log
echo "imomv2 done $(date)"
} 2>&1 | tee -a runs/imomv2_q.log
