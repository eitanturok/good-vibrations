#!/usr/bin/env bash
# src9 transformer3 on the new defaults (big-drop20-v1's model: d 256, 8 heads, 8 laser blocks, latent 512, 20% laser + band dropout, 300 ep):
# a new baseline, then one change each: loss knobs, centre heatmap, dropout, fewer augmented samples, crisp-edge mask losses / mask-AE decoder, [cls] readout, and 1k ep
cd /home/ethantu/workspace/good-vibrations
PY=.venv/bin/python
T3="$PY src9/run.py --model transformer3 --microbatch 128"
[ -f runs/maskae/maskae.pt ] || $PY src9/maskae.py 2>&1 | tee runs/maskae.log  # the frozen mask AE for maskae-v2 (~1 min)

$T3 --run-name base300-v1 2>&1 | tee runs/base300-v1.log                                   # the new baseline
$T3 --objrecall-w 0.5 --run-name objrecall-v2 2>&1 | tee runs/objrecall-v2.log             # + 0.5 x mean over true objects of 1 - (prob mass on it / its area)
$T3 --heatmap-w 1.0 --run-name heatmap-v1 2>&1 | tee runs/heatmap-v1.log                   # + 2nd decoder channel: object-centre heatmap, CenterNet focal loss x 1
$T3 --laser-drop 0.3 --band-drop 0.3 --run-name drop30-v1 2>&1 | tee runs/drop30-v1.log     # 30% laser + band dropout
$T3 --laser-jitter-p 0.05 --speaker-gain-p 0.1 --run-name aug-v1 2>&1 | tee runs/aug-v1.log  # laser jitter on 5%, speaker gain on 10% of samples only
$T3 --mask-loss sdf --run-name sdf-v2 2>&1 | tee runs/sdf-v2.log                           # output = signed distance: L1(tanh) + bce(sigmoid(-sdf))
$T3 --mask-loss lovasz --run-name lovasz-v2 2>&1 | tee runs/lovasz-v2.log                  # 0.5 bce + Lovasz hinge instead of dice
$T3 --maskae runs/maskae/maskae.pt --run-name maskae-v2 2>&1 | tee runs/maskae-v2.log      # Linear(u -> 32) -> frozen mask-AE decoder, + 0.1 |z - z_gt|^2
$T3 --readout cls --run-name cls-v2 2>&1 | tee runs/cls-v2.log                             # [cls] tokens instead of the per-band / per-laser readouts
$T3 --max-duration 1000ep --run-name base1k-v1 2>&1 | tee runs/base1k-v1.log               # the new baseline, 1000 ep
echo "transformer3 queue (q7) done"
