#!/usr/bin/env bash
# src9 transformer3 readout variants, PARAMETER-MATCHED (m) to base300-v1 / drop30-v1 (per, d256, 34,536,656 params) by widening d; all at laser/band dropout 0.3 (reference = drop30-v1).
# cls loses to per: shared-V softmax pooling drops slot identity + signed contrasts; rope puts position only in the logits. pool/cls/query at d256 are only 20-33M params, so widen d.
# all = transformer3 defaults (freq_layers 4, laser_layers 8, latent_dim 512, 300 ep, batch 1024) + readout change + d/heads + drop .3. run by scripts/after_norm2.sh after the eager re-evals.
# widened runs at microbatch 64: est. GPU peak 11.4-12.1 GB (mb128 would be 18.7-20.3 > 16.3 GB). nolaserattn stays d256 (unmatched by design), mb128 est 13.2 GB.
cd /home/ethantu/workspace/good-vibrations
PY=.venv/bin/python
export PYTHONUTF8=1
T3="$PY src9/run.py --model transformer3 --laser-drop 0.3 --band-drop 0.3"

$T3 --microbatch 64 --d 400 --heads 10 --readout pool --run-name pool-m-v1 2>&1 | tee runs/pool-m-v1.log                              # 34,259,472 params; masked mean-pool -> Linear (ViT GAP)
$T3 --microbatch 64 --d 400 --heads 10 --readout pool --abspos 1 --run-name pool-abspos-m-v1 2>&1 | tee runs/pool-abspos-m-v1.log     # 34,306,672; + learned abs pos in the content
$T3 --microbatch 64 --d 400 --heads 10 --readout cls --abspos 1 --run-name cls-abspos-m-v1 2>&1 | tee runs/cls-abspos-m-v1.log        # 34,147,072; [cls] + abs pos (ViT / CLIP AttentionPool2d)
$T3 --microbatch 64 --d 336 --heads 7 --readout query --run-name query-m-v1 2>&1 | tee runs/query-m-v1.log                            # 35,329,040; 16 DETR queries, key-only pos, concat -> Linear
$T3 --microbatch 64 --d 336 --heads 7 --readout query --abspos 1 --run-name query-abspos-m-v1 2>&1 | tee runs/query-abspos-m-v1.log   # 35,368,688; + abs pos in the content (values carry position)
$T3 --microbatch 128 --laser-layers 0 --run-name nolaserattn-d30-v1 2>&1 | tee runs/nolaserattn-d30-v1.log                           # per, no laser blocks, d256 (vs drop30-v1)
echo "readout queue done $(date)"
