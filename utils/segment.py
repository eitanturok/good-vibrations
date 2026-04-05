import argparse
import os

import modal
import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import center_of_mass


# ── Modal setup ──────────────────────────────────────────────────────────────

app = modal.App("segment")

image = (
    modal.Image.from_registry("nvidia/cuda:12.8.0-devel-ubuntu22.04", add_python="3.12")
    .entrypoint([])
    .apt_install("git")
    .uv_sync(uv_project_dir=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)


PROMPT = "A black metal cube sitting on the floor of an open cardboard box from a bird's eye view."


@app.function(gpu="A10G", image=image, secrets=[modal.Secret.from_name("huggingface")])
def segment(image, object, box_material="cardboard", prompt=None):
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor

    import torch
    model = build_sam3_image_model()
    processor = Sam3Processor(model)

    processor.set_confidence_threshold(0.0)
    if prompt is None:
        prompt = PROMPT
    with torch.autocast("cuda", dtype=torch.bfloat16):
        state = processor.set_image(Image.fromarray(image))
        out = processor.set_text_prompt(state=state, prompt=prompt)

    n = len(out["scores"])
    if n > 1:
        idx = out["scores"].topk(1).indices
        out["masks"]  = out["masks"][idx]
        out["boxes"]  = out["boxes"][idx]
        out["scores"] = out["scores"][idx]

    print(f"detections: {min(1, n)}")
    print(f"scores:     {out['scores'].tolist()}")
    print(f"boxes:      {out['boxes'].tolist()}")

    mask = out["masks"][:, 0].any(dim=0)

    overlay = np.zeros((*mask.shape, 4), dtype=np.float32)
    overlay[mask.cpu().numpy()] = (0.0, 0.8, 0.2, 0.5)

    return mask.cpu().numpy(), overlay


# ── Image helpers ─────────────────────────────────────────────────────────────

def crop_image(image, left=0, right=1, up=0, down=1):
    W, H = image.size
    x1, x2 = int(W * left), int(W * right)
    y1, y2 = int(H * up), int(H * down)
    return image.crop((x1, y1, x2, y2))


def plot_overlay_image(cropped_image, overlay, x_pos, y_pos):
    cropped = np.array(cropped_image)
    blended = (
        cropped * (1 - overlay[..., 3:])
        + (overlay * 255).astype(np.uint8)[..., :3] * overlay[..., 3:]
    ).astype(np.uint8)
    overlay_image = Image.fromarray(blended)

    r = 20
    draw = ImageDraw.Draw(overlay_image)
    draw.line([(x_pos - r, y_pos), (x_pos + r, y_pos)], fill=(255, 0, 0), width=4)
    draw.line([(x_pos, y_pos - r), (x_pos, y_pos + r)], fill=(255, 0, 0), width=4)

    return overlay_image


# ── Core pipeline ─────────────────────────────────────────────────────────────

def segment_sample(sample_path=None, raw_image=None, left=0.15, right=0.67, up=0.08, down=0.7, object="circle", box_material="cardboard", prompt=None):
    """Run the vision pipeline. Exactly one of sample_path or raw_image must be provided."""
    if (sample_path is None) == (raw_image is None):
        raise ValueError("Exactly one of sample_path or raw_image must be provided.")
    if raw_image is None:
        raw_image = Image.open(os.path.join(sample_path, "box_overhead_image.png"))

    cropped_image = crop_image(raw_image, left=left, right=right, up=up, down=down)

    image_array = np.array(cropped_image.convert("RGB"), dtype=np.uint8)
    with app.run():
        mask, overlay = segment.remote(image_array, object, box_material, prompt)

    y_pos, x_pos = center_of_mass(mask)
    overlay_image = plot_overlay_image(cropped_image, overlay, x_pos, y_pos)

    vision = {
        "x_position":    x_pos,
        "y_position":    y_pos,
        "raw_image":     raw_image,
        "cropped_image": cropped_image,
        "overlay_image": overlay_image,
    }
    return mask, vision


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Crop, segment, and compute position for a sample.")
    parser.add_argument("--sample-path", required=True, help="Path to sample directory")
    parser.add_argument("--object",      default="circle", help="Object label to segment (default: circle)")
    parser.add_argument("--left",        type=float, default=0.15)
    parser.add_argument("--right",       type=float, default=0.67)
    parser.add_argument("--up",          type=float, default=0.08)
    parser.add_argument("--down",        type=float, default=0.7)
    parser.add_argument("--prompt",      default=None, help=f"Text prompt for segmentation (default: '{PROMPT}')")
    parser.add_argument("--out",         default="processed.npz", help="Output .npz to save shifts, mask, and data")
    parser.add_argument("--show-images", default=True,  action=argparse.BooleanOptionalAction, help="Open each intermediate image in the system viewer")
    parser.add_argument("--debug-dir",   default="debug", help="Directory to save intermediate images (default: debug/)")
    args = parser.parse_args()

    mask, vision = segment_sample(
        args.sample_path,
        left=args.left,
        right=args.right,
        up=args.up,
        down=args.down,
        object=args.object,
        prompt=args.prompt,
    )

    # Save intermediate images to debug dir
    os.makedirs(args.debug_dir, exist_ok=True)
    image_keys = ["raw_image", "cropped_image", "overlay_image"]
    for key in image_keys:
        img_path = os.path.join(args.debug_dir, f"{key}.png")
        vision[key].save(img_path)
        print(f"  {key}: {img_path}")
        if args.show_images:
            vision[key].show(title=key)
        vision[key] = img_path

    np.savez_compressed(args.out, mask=mask, **vision)
    print(f"Saved to {args.out}")


if __name__ == "__main__":
    main()
