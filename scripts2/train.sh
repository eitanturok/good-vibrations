#!/usr/bin/env bash
# FLUX.2 klein img2img LoRA training, end to end: builds the dataset (one-cube
# train/eval split) if missing, precomputes the text embedding, then trains.
# Heatmap (conditioning) -> segmentation mask (target). Every example uses the
# same fixed prompt (see src2/data.py FIXED_PROMPT -- keep this in sync with
# it), giving the text branch real semantic grounding alongside the heatmap's
# structural conditioning.
#
# Usage:
#   ./scripts2/train.sh [box]     # default box: gastronorm
#
# Deps (diffusers, accelerate, transformers, peft, bitsandbytes, sentencepiece,
# protobuf, torchao) are installed directly via `uv pip install` -- NOT
# `uv add`/`uv sync` -- so pyproject.toml / uv.lock are untouched and the
# existing torch==2.11.0+cu130 pin stays intact.
#
# Once, configure accelerate for this single-GPU machine: accelerate config
#
# Sized for a single RTX 5080 (16GB VRAM): 4B klein base, fp8 base weights,
# gradient checkpointing, resolution 512. train_batch_size=5 is the measured
# ceiling at this resolution/config (binary search: BS=5 fits, BS=6 OOMs at
# ~15.2/15.47GiB). --cache_latents (frees the VAE after its one-time encode
# pass) and --allow_tf32 were added for speed but did NOT raise the ceiling --
# re-probed BS=6/8 and both still OOM, so the wall here is per-sample
# transformer activation memory, not the VAE. Don't raise batch size without
# re-probing; effective batch = 5 * accumulation.

set -euo pipefail

# Reduces allocator fragmentation -- the OOM that motivated dropping
# TRAIN_BATCH_SIZE below showed "787MiB reserved but unallocated" at the point
# of failure, which is exactly what this flag targets (PyTorch's own error
# message suggested it).
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Must match src2/data.py's PROMPTS["mask"] exactly -- that's what every row's
# "text" column holds for a mask-target run, and this is what
# precompute_text_embeds.py caches ahead of training so the training script's own
# encode of the same string is a cache hit, never touching the GPU with the text
# encoder. This script builds mask-target datasets (--target mask below), so it
# uses the mask prompt -- a small white blob on black, NOT a photo description,
# since that's what the mask ground truth actually looks like. A future
# edit-mode/--target photo script should use PROMPTS["photo"] instead.
PROMPT="A small white square on a black background"

BOX="${1:-gastronorm}"
DATA_DIR="$(pwd)/src2/data/${BOX}_one_cube"
TRAIN_DIR="${DATA_DIR}/train"
EVAL_DIR="${DATA_DIR}/eval"
RUNS_DIR="scripts2/runs"
OUT_DIR="${RUNS_DIR}/klein-lora-${BOX}"
TEXT_EMBED_CACHE_DIR="${RUNS_DIR}/text_embed_cache"  # must match train_dreambooth_lora_flux2_klein_img2img.py: Path(output_dir).parent / "text_embed_cache"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 1. Build the dataset (empty-box + 80% of purple-cube/red-cube positions in
# train, remaining 20% in eval, split by position so no leakage) if missing.
if [ ! -f "${TRAIN_DIR}/data.parquet" ]; then
  python "${SCRIPT_DIR}/../src2/data.py" --box "${BOX}" --out "${DATA_DIR}" --split one-cube
fi

# 2. --validation_images/--validation_ground_truths need real files on disk;
# data.py only writes into the parquet (see its module docstring for why), so
# pull the SAME 2 train-split + 2 eval-split samples' heatmaps (conditioning)
# and targets (ground truth) out to loose files every run -- fixed indices
# [0,1], so it's the same 4 images every epoch, every restart, per the user's
# request. The raw heatmap is ~1235x100 (freq bins x lasers), a 12.3:1 aspect
# ratio -- Flux2ImageProcessor's standalone pipeline (used only for this
# validation call, not for training itself) hard-caps aspect ratio at 8:1, so
# pad the short axis rather than stretching/cropping real data out of the image.
mkdir -p "${RUNS_DIR}"
VALIDATION_PATHS=$(python -c "
from datasets import load_dataset
from PIL import Image

def pad_heatmap(img):
    w, h = img.size
    max_ratio = 8
    target_h = -(-w // max_ratio)  # ceil(w / max_ratio): smallest height keeping ratio <= 8:1
    if h < target_h:
        padded = Image.new(img.mode, (w, target_h))
        padded.paste(img, (0, (target_h - h) // 2))
        return padded
    return img

cond_paths, gt_paths, labels = [], [], []
for split_label, split_dir in [('train', '${TRAIN_DIR}'), ('eval', '${EVAL_DIR}')]:
    ds = load_dataset(split_dir)['train']
    for i in range(min(2, len(ds))):
        cond_path = '${RUNS_DIR}/${BOX}_validation_' + split_label + '_heatmap_' + str(i) + '.png'
        gt_path = '${RUNS_DIR}/${BOX}_validation_' + split_label + '_target_' + str(i) + '.png'
        pad_heatmap(ds[i]['cond_file_name']).save(cond_path)
        ds[i]['file_name'].save(gt_path)
        cond_paths.append(cond_path)
        gt_paths.append(gt_path)
        labels.append(split_label)
print(','.join(cond_paths))
print(','.join(gt_paths))
print(','.join(labels))
" 2>&1 | tail -3)
VALIDATION_IMAGES=$(echo "${VALIDATION_PATHS}" | sed -n '1p')
VALIDATION_GROUND_TRUTHS=$(echo "${VALIDATION_PATHS}" | sed -n '2p')
VALIDATION_SPLIT_LABELS=$(echo "${VALIDATION_PATHS}" | sed -n '3p')

# 3. Steps-per-epoch depends on dataset size, which varies by box -- compute
# max_train_steps (from N_EPOCHS) and checkpointing_steps (from
# CHECKPOINT_EVERY_N_EPOCHS) both dynamically, rather than hardcoding step
# counts that would silently be wrong for a different-sized box.
#
# TRAIN_BATCH_SIZE=4, not the binary-searched ceiling of 5: a real multi-epoch
# run OOM'd on step 1 at BS=5 (15.05/15.47GiB, "787MiB reserved but
# unallocated" -- allocator fragmentation, not a hard model-size wall) even
# though short 2-step probes fit. Dropping one batch size for a safety margin
# rather than trusting the exact probed ceiling for a real long run.
TRAIN_BATCH_SIZE=4
N_EPOCHS="${N_EPOCHS:-100}"
CHECKPOINT_EVERY_N_EPOCHS="${CHECKPOINT_EVERY_N_EPOCHS:-50}"
VALIDATION_EVERY_N_EPOCHS="${VALIDATION_EVERY_N_EPOCHS:-1}"
N_SAMPLES=$(python -c "
from datasets import load_dataset
print(len(load_dataset('${TRAIN_DIR}')['train']))
" 2>&1 | tail -1)
STEPS_PER_EPOCH=$(( (N_SAMPLES + TRAIN_BATCH_SIZE - 1) / TRAIN_BATCH_SIZE ))
MAX_TRAIN_STEPS=$(( STEPS_PER_EPOCH * N_EPOCHS ))
CHECKPOINTING_STEPS=$(( STEPS_PER_EPOCH * CHECKPOINT_EVERY_N_EPOCHS ))
echo "${N_SAMPLES} train samples -> ${STEPS_PER_EPOCH} steps/epoch -> ${MAX_TRAIN_STEPS} steps for ${N_EPOCHS} epochs, checkpointing every ${CHECKPOINTING_STEPS} steps (${CHECKPOINT_EVERY_N_EPOCHS} epochs)"

# 4. Encode the fixed prompt in its own process: only the tokenizer + Qwen3
# text encoder ever load, so this never competes with the transformer/VAE for
# VRAM. The training script checks this same cache dir before touching the GPU
# for text encoding, so once this has run, training never loads the text
# encoder at all.
python "${SCRIPT_DIR}/../src2/precompute_text_embeds.py" \
  --model-id="black-forest-labs/FLUX.2-klein-4B" \
  --prompt="${PROMPT}" \
  --max-sequence-length=32 \
  --cache-dir="${TEXT_EMBED_CACHE_DIR}"

# 5. Train.
accelerate launch "${SCRIPT_DIR}/train_dreambooth_lora_flux2_klein_img2img.py" \
  --pretrained_model_name_or_path="black-forest-labs/FLUX.2-klein-4B" \
  --dataset_name="${TRAIN_DIR}" \
  --image_column="file_name" \
  --cond_image_column="cond_file_name" \
  --caption_column="text" \
  --instance_prompt="${PROMPT}" \
  --max_sequence_length=32 \
  --output_dir="${OUT_DIR}" \
  --resolution=512 \
  --train_batch_size=${TRAIN_BATCH_SIZE} \
  --gradient_accumulation_steps=1 \
  --gradient_checkpointing \
  --do_fp8_training \
  --offload \
  --cache_latents \
  --allow_tf32 \
  --rank=16 \
  --lora_alpha=16 \
  --learning_rate=1e-4 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=${MAX_TRAIN_STEPS} \
  --checkpointing_steps=${CHECKPOINTING_STEPS} \
  --mixed_precision="bf16" \
  --target="mask" \
  --report_to="wandb" \
  --validation_prompt="${PROMPT}" \
  --validation_images="${VALIDATION_IMAGES}" \
  --validation_ground_truths="${VALIDATION_GROUND_TRUTHS}" \
  --validation_split_labels="${VALIDATION_SPLIT_LABELS}" \
  --num_validation_images=1 \
  --validation_epochs=${VALIDATION_EVERY_N_EPOCHS} \
  --seed=42
