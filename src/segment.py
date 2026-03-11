"""
Auto-segmentation using SAM 3 (Segment Anything Model 3) by Meta.
Usage: python segment.py <image_path> [text_prompt] [output_path]

pip install transformers accelerate torch torchvision pillow matplotlib numpy
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
import torch
from PIL import Image
from transformers import Sam3Processor, Sam3Model


def segment(image_path: str, prompt: str = "objects", output_path: str = "segmentation.png", threshold: float = 0.4) -> list[np.ndarray]:
    device = "cuda" if torch.cuda.is_available() else "cpu"

    processor = Sam3Processor.from_pretrained("facebook/sam3")
    model = Sam3Model.from_pretrained("facebook/sam3").to(device)

    image = Image.open(image_path).convert("RGB")

    inputs = processor(images=image, text=prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)

    results = processor.post_process_instance_segmentation(
        outputs, threshold=threshold, mask_threshold=0.5,
        target_sizes=inputs["original_sizes"].tolist()
    )[0]

    masks = []
    if "masks" in results:
        masks = [(m.cpu().numpy() * 255).astype(np.uint8) for m in results["masks"]]

    # Plot
    image_np = np.array(image)
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(image_np)
    for mask in masks:
        color = np.concatenate([np.random.random(3), [0.5]])
        overlay = np.zeros((*mask.shape, 4))
        overlay[mask > 127] = color
        ax.imshow(overlay)
    ax.set_title(f"SAM 3 — prompt: '{prompt}' — {len(masks)} masks")
    ax.axis("off")
    fig.savefig(output_path, bbox_inches="tight", pad_inches=0, dpi=150)
    plt.close(fig)
    print(f"Saved segmentation plot to {output_path} ({len(masks)} masks found)")

    return masks, fig


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python segment.py <image_path> [text_prompt] [output_path]")
        sys.exit(1)
    img = sys.argv[1]
    prompt = sys.argv[2] if len(sys.argv) > 2 else "objects"
    out = sys.argv[3] if len(sys.argv) > 3 else "segmentation.png"
    segment(img, prompt, out)
