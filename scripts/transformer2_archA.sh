#!/usr/bin/env bash
# src9 transformer2: Arch A readout (ref tokenization + 2 freq / 2 laser blocks, no gate, no flatten): f = sum_b m_b W_b LN(h_b) per laser,
# u = LN(sum_l m_l W_l LN(t_l)) + W_mask m (latent_dim 128, the single bottleneck), emb = W_up u -> decoder + aux heads. masks all ones for now. runs after the heads queue
cd /home/ethantu/workspace/good-vibrations
while pgrep -f transformer2_heads.sh >/dev/null; do sleep 60; done
PY=.venv/bin/python
T2="$PY src9/run.py --model transformer2"

$T2 --readout archA --latent-dim 128 --run-name archA-v1 2>&1 | tee runs/archA-v1.log                                                             # vs ref (gated bands + flatten)
echo "transformer2 archA queue done"
