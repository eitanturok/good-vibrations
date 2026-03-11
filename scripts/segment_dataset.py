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
    from datasets import load_dataset, concatenate_datasets
    from huggingface_hub import HfApi

    # Download existing dataset
    print("Loading dataset from HF...")
    ds = load_dataset(REPO_ID)  # DatasetDict with train/test splits

    model = build_sam3_image_model()
    processor = Sam3Processor(model)

    all_masks = {}

    def process_split(split_ds, split_name):
        new_rows = []
        for i, sample in enumerate(split_ds):
            print(f"[{split_name}] Segmenting {i+1}/{len(split_ds)}")
            img, (x1, x2, y1, y2), cropped_arr, overlay_arr, mask_tensor = segment_object(sample, processor)

            mask_idx = sample["shifts_idx"]
            all_masks[f"mask_{mask_idx}"] = mask_tensor.cpu().to(torch.bool)

            # Build derived images
            crop_line_img = Image.fromarray(img).convert("RGB")
            draw = ImageDraw.Draw(crop_line_img)
            W, H = crop_line_img.size
            draw.line([(x1, 0), (x1, H)], fill="red", width=2)
            draw.line([(x2, 0), (x2, H)], fill="red", width=2)
            draw.line([(0, y1), (W, y1)], fill="red", width=2)
            draw.line([(0, y2 - 1), (W, y2 - 1)], fill="red", width=2)

            cropped_img  = Image.fromarray(cropped_arr)
            mask_img     = Image.fromarray(mask_tensor.cpu().numpy().astype(np.uint8) * 255)
            cropped_rgba = Image.fromarray(cropped_arr).convert("RGBA")
            overlay_pil  = Image.fromarray((overlay_arr * 255).astype(np.uint8), mode="RGBA")
            overlay_img  = Image.alpha_composite(cropped_rgba, overlay_pil).convert("RGB")

            new_rows.append({
                **sample,
                "mask_idx":        mask_idx,
                "crop_line_image": crop_line_img,
                "cropped_image":   cropped_img,
                "mask_image":      mask_img,
                "overlay_image":   overlay_img,
            })
        return new_rows

    from datasets import Dataset, DatasetDict, Features, Value, Image as HFImage

    features = Features({
        "shifts_idx":        Value("int32"),
        "mask_idx":          Value("int32"),
        "raw_image":         HFImage(),
        "crop_line_image":   HFImage(),
        "cropped_image":     HFImage(),
        "mask_image":        HFImage(),
        "overlay_image":     HFImage(),
        "x_position":        Value("int32"),
        "y_position":        Value("int32"),
        "experiment_config": Value("string"),
        "fps":               Value("int32"),
        "object":            Value("string"),
    })

    new_ds = DatasetDict({
        split: Dataset.from_list(process_split(ds[split], split), features=features)
        for split in ds
    })

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
