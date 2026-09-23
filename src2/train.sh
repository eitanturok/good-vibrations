#!/usr/bin/env bash
# FLUX.2 klein img2img LoRA training: heatmap (conditioning) -> segmentation mask
# (target). Caption-free: every example uses the same fixed empty prompt (see
# data.py FIXED_PROMPT), so text conditioning carries no information and the
# heatmap does all the structural conditioning work.
#
# Prereqs (NOT yet installed in .venv -- see repo memory: never `uv add`/`sync`
# blindly here, .venv already drifted from uv.lock once over a torch cu130 pin):
#   uv pip install diffusers accelerate transformers peft bitsandbytes
#   accelerate config   # once, to set up your local single-GPU config
#
# Data prep first:
#   python src2/data.py --box gastronorm --out src2/data/gastronorm
#
# Sized for a single RTX 5080 (16GB VRAM): 4B klein base, fp8 base weights,
# gradient checkpointing, resolution 512, batch size 1 + grad accumulation.

set -euo pipefail

BOX="${1:-gastronorm}"
DATA_DIR="src2/data/${BOX}"
OUT_DIR="src2/runs/klein-lora-${BOX}"

accelerate launch examples/dreambooth/train_dreambooth_lora_flux2_img2img.py \
  --pretrained_model_name_or_path="black-forest-labs/FLUX.2-klein-4B" \
  --dataset_name="${DATA_DIR}" \
  --image_column="file_name" \
  --cond_image_column="cond_file_name" \
  --caption_column="text" \
  --instance_prompt="" \
  --output_dir="${OUT_DIR}" \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --gradient_checkpointing \
  --do_fp8_training \
  --rank=16 \
  --lora_alpha=16 \
  --learning_rate=1e-4 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=2000 \
  --checkpointing_steps=250 \
  --mixed_precision="bf16" \
  --seed=42
