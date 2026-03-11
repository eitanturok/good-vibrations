import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

REPO_ID = "eturok-weizmann/vibration-data"
MASKS_SAFETENSORS_PATH = "/tmp/masks.safetensors"


def segment_object(sample, processor):
    """
    Returns:
        img         : (H, W, 3) uint8 ndarray — raw image
        crop_params : (x1, x2, y1, y2)
        cropped     : (H', W', 3) uint8 ndarray
        overlay     : (H', W', 4) float32 RGBA ndarray
        mask        : (H', W') BoolTensor
    """
    img = np.array(sample["raw_image"].convert("RGB"))
    H, W = img.shape[:2]
    x1, x2 = int(W * 0.31), int(W * 0.62)
    y1, y2 = int(H * 0.25), H
    cropped = img[y1:y2, x1:x2]

    processor.set_confidence_threshold(0.0)
    state = processor.set_image(Image.fromarray(cropped))
    prompt = f"Two {sample['object']} inside an open cardboard box from a bird's eye view."
    out = processor.set_text_prompt(state=state, prompt=prompt)

    # Pick top-1 by score
    n = len(out["scores"])
    if n > 1:
        idx = out["scores"].topk(1).indices
        out["masks"]  = out["masks"][idx]
        out["boxes"]  = out["boxes"][idx]
        out["scores"] = out["scores"][idx]

    print(f"detections: {min(1, n)}, scores: {out['scores'].tolist()}")

    mask = out["masks"][:, 0].any(dim=0)  # (H', W') BoolTensor

    overlay = np.zeros((*mask.shape, 4), dtype=np.float32)
    overlay[mask.cpu().numpy()] = (0.0, 0.8, 0.2, 0.5)

    return img, (x1, x2, y1, y2), cropped, overlay, mask


def plot_segment(sample, img, lines, cropped, overlay, mask):
    x1, x2, y1, y2 = lines
    fig, axes = plt.subplots(1, 4, figsize=(24, 5))

    axes[0].imshow(img)
    axes[0].axvline(x1, color="r"); axes[0].axvline(x2, color="r")
    axes[0].axhline(y1, color="r"); axes[0].axhline(y2 - 1, color="r")
    axes[0].set_title("Raw + crop lines", fontsize=12)
    axes[0].axis("off")

    axes[1].imshow(cropped)
    axes[1].set_title("Cropped", fontsize=12)
    axes[1].axis("off")

    axes[2].imshow(mask.cpu().numpy(), cmap="gray", vmin=0, vmax=1)
    axes[2].set_title("Mask", fontsize=12)
    axes[2].axis("off")

    axes[3].imshow(cropped)
    axes[3].imshow(overlay)
    axes[3].set_title(f"Overlay — '{sample['object']}'", fontsize=12)
    axes[3].axis("off")

    plt.tight_layout()
    return fig


if __name__ == "__main__":
    import torch
    from safetensors.torch import save_file
    from datasets import load_dataset
    from huggingface_hub import HfApi

    # Download existing dataset
    print("Loading dataset from HF...")
    ds = load_dataset(REPO_ID)  # DatasetDict with train/test splits

    model = build_sam3_image_model()
    processor = Sam3Processor(model)

    all_masks = {}

    def process_split(split_ds, split_name):
        # Only collect new columns keyed by shifts_idx — do NOT unpack the full sample
        new_cols = {}
        for i, sample in enumerate(split_ds):
            print(f"[{split_name}] Segmenting {i+1}/{len(split_ds)}")
            with torch.no_grad():
                img, (x1, x2, y1, y2), cropped_arr, overlay_arr, mask_tensor = segment_object(sample, processor)

            mask_idx = sample["shifts_idx"]
            all_masks[f"mask_{mask_idx}"] = mask_tensor.cpu().to(torch.bool)
            torch.cuda.empty_cache()

            # Build derived images as numpy arrays (Arrow-serializable, no PIL encoding hang)
            crop_line_arr = img.copy()
            crop_line_arr[y1:y1+2, :] = (255, 0, 0)
            crop_line_arr[y2-2:y2, :] = (255, 0, 0)
            crop_line_arr[:, x1:x1+2] = (255, 0, 0)
            crop_line_arr[:, x2-2:x2] = (255, 0, 0)

            mask_np      = mask_tensor.cpu().numpy().astype(np.uint8) * 255
            mask_arr     = np.stack([mask_np, mask_np, mask_np], axis=-1)  # H'xW'x3

            overlay_rgba = (overlay_arr * 255).astype(np.uint8)
            alpha        = overlay_rgba[:, :, 3:4] / 255.0
            overlay_arr_rgb = (cropped_arr * (1 - alpha) + overlay_rgba[:, :, :3] * alpha).astype(np.uint8)

            new_cols[mask_idx] = {
                "mask_idx":        mask_idx,
                "crop_line_image": crop_line_arr,
                "cropped_image":   cropped_arr,
                "mask_image":      mask_arr,
                "overlay_image":   overlay_arr_rgb,
            }
        return new_cols

    from datasets import DatasetDict

    new_splits = {}
    for split in ds:
        new_cols = process_split(ds[split], split)
        new_splits[split] = ds[split].map(lambda sample: new_cols[sample["shifts_idx"]], num_proc=1)

    new_ds = DatasetDict(new_splits)

    # Save and upload masks safetensors
    print(f"Saving masks to {MASKS_SAFETENSORS_PATH}...")
    save_file(all_masks, MASKS_SAFETENSORS_PATH)

    api = HfApi()
    print("Pushing updated dataset to HF...")
    new_ds.push_to_hub(REPO_ID)
    print("Uploading masks.safetensors...")
    api.upload_file(path_or_fileobj=MASKS_SAFETENSORS_PATH, path_in_repo="masks.safetensors",
                    repo_id=REPO_ID, repo_type="dataset")

    print("Done.")
