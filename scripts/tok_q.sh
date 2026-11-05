#!/usr/bin/env bash
# src9 transformer3 tokenizer variants vs drop30-v1 (per, d256, drop .3, 300 ep, 34,536,656 params); band edges split ~50 Hz resonances.
# all = transformer3 defaults (freq_layers 4, laser_layers 8, latent_dim 512, 300 ep, batch 1024, mb128) + drop .3 + one tokenizer change.
# run by tmux train:tok after scripts/after_norm2.sh exits.
cd /home/ethantu/workspace/good-vibrations
PY=.venv/bin/python
export PYTHONUTF8=1
T3="$PY src9/run.py --model transformer3 --laser-drop 0.3 --band-drop 0.3"

$T3 --microbatch 128 --band-overlap 32 --run-name overlap32-d30-v1 2>&1 | tee runs/overlap32-d30-v1.log   # 34,569,424 params; 129-bin windows (65 own + 32 each side, ~+-25 Hz), stride 65, zero-padded ends
$T3 --microbatch 128 --tokenizer mlp --run-name tokmlp-d30-v1 2>&1 | tee runs/tokmlp-d30-v1.log             # 34,701,520; Linear(130 -> 512) -> GELU -> Linear(512 -> 256), shared by every band
echo "tok queue done $(date)"
