"""Segment an image on modal using SAM3"""
from pathlib import Path

import modal
import numpy as np
from PIL import Image, ImageDraw, ImageFont

app = modal.App("segment")

modal_image = (
    modal.Image.from_registry("nvidia/cuda:12.8.0-devel-ubuntu22.04", add_python="3.12")
    .entrypoint([])
    .apt_install("git")
    .pip_install("transformers", "Pillow", "torch", "torchvision", "accelerate")
)

# ***** old SAM3-image-model Segmenter: always collapses to a single top-1 mask, kept for reference *****
# @app.cls(
#     gpu="A10G",
#     image=modal_image,
#     secrets=[modal.Secret.from_name("huggingface")],
#     scaledown_window=60 * 10,  # shut down after 10 min idle
# )
# class Segmenter:
#     @modal.enter()
#     def load(self):
#         from sam3.model_builder import build_sam3_image_model
#         from sam3.model.sam3_image_processor import Sam3Processor
#         self.processor = Sam3Processor(build_sam3_image_model())
#         self.processor.set_confidence_threshold(0.0)
#
#     @modal.method()
#     def run(self, image_array: np.ndarray, prompt: str) -> np.ndarray:
#         import torch
#         with torch.autocast("cuda", dtype=torch.bfloat16):
#             state = self.processor.set_image(Image.fromarray(image_array))
#             out = self.processor.set_text_prompt(state=state, prompt=prompt)
#         if len(out["scores"]) > 1:
#             idx = out["scores"].topk(1).indices
#             out["masks"] = out["masks"][idx]
#         mask = out["masks"][:, 0].any(dim=0)
#         return mask.cpu().numpy()  # bool (H, W)


@app.cls(
    gpu="A10G",
    image=modal_image,
    secrets=[modal.Secret.from_name("huggingface")],
    scaledown_window=60 * 10,  # shut down after 10 min idle
)
class Segmenter:
    @modal.enter()
    def load(self):
        import torch
        from transformers import Sam3Model, Sam3Processor

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = Sam3Processor.from_pretrained("facebook/sam3")
        self.model = Sam3Model.from_pretrained("facebook/sam3").to(self.device)

    def downsample(self, image: Image.Image, scale: float) -> Image.Image:
        """Resize both sides by scale (e.g. 0.25 shrinks each side to a quarter),
        preserving aspect ratio. Remembers the original size (for upsample()) and
        the scale (for rescaling boxes)."""
        self.original_size = image.size  # (width, height)
        self.scale = scale
        if scale == 1.0: return image
        w, h = image.size
        return image.resize((round(w * scale), round(h * scale)), resample=Image.LANCZOS)

    def upsample(self, mask: np.ndarray) -> np.ndarray:
        """Resize a boolean/float segmentation mask back to the original image dimensions."""
        mask_image = Image.fromarray((mask > 0).astype(np.uint8) * 255)
        mask_image = mask_image.resize(self.original_size, resample=Image.NEAREST)
        return np.array(mask_image) > 0

    def inference(self, image: np.ndarray, prompt: str, input_boxes: list | None = None, input_boxes_labels: list | None = None) -> dict:
        """Run SAM3 on an already-downsampled image array, prompted by text and/or boxes.

        Returns a dict with:
            - scores (np.ndarray): confidence score per kept instance.
            - boxes (np.ndarray): (top_left_x, top_left_y, bottom_right_x, bottom_right_y)
              per kept instance, in the input image's coordinates.
            - masks (np.ndarray): binary mask per kept instance, shape (num_instances, height, width).
        """
        import torch
        inputs = self.processor(images=image, text=prompt, return_tensors="pt", input_boxes=[input_boxes] if input_boxes else None, input_boxes_labels=[input_boxes_labels] if input_boxes_labels else None,).to(self.device)
        with torch.no_grad(): outputs = self.model(**inputs)
        result = self.processor.post_process_instance_segmentation(outputs, threshold=0.0, mask_threshold=0.5, target_sizes=inputs["original_sizes"].tolist())[0]
        result = {k: v.float().cpu().numpy() if torch.is_tensor(v) else v for k, v in result.items()}
        return result

    @modal.method()
    def run(self, image: Image.Image, prompt: str, input_boxes: list | None = None, input_boxes_labels: list | None = None, scale: float = 1.0) -> dict:
        """Downsample the image, run inference, then upsample the resulting masks back to the original resolution."""

        # downsample so SAM-3 inference runs faster
        downsampled = self.downsample(image.convert("RGB"), scale)
        scaled_boxes = [[x1 * self.scale, y1 * self.scale, x2 * self.scale, y2 * self.scale] for x1, y1, x2, y2 in input_boxes] if input_boxes else None

        # inference: only convert to a numpy array right before calling the model
        result = self.inference(image=np.array(downsampled), prompt=prompt, input_boxes=scaled_boxes, input_boxes_labels=input_boxes_labels)

        # upsample
        result["masks"] = np.stack([self.upsample(mask) for mask in result["masks"]]) if len(result["masks"]) else result["masks"]
        result["boxes"] = result["boxes"] / self.scale
        return result


#***** mask/box/score overlay visualization *****

# 20 mutually distinct, high-contrast colors (Sasha Trubetskoy palette).
_MASK_COLORS = np.array([
    [230, 25, 75], [60, 180, 75], [255, 225, 25], [0, 130, 200], [245, 130, 48],
    [145, 30, 180], [70, 240, 240], [240, 50, 230], [210, 245, 60], [250, 190, 212],
    [0, 128, 128], [220, 190, 255], [170, 110, 40], [255, 250, 200], [128, 0, 0],
    [170, 255, 195], [128, 128, 0], [255, 215, 180], [0, 0, 128], [128, 128, 128],
])


def _format_score(score):
    """3 decimal places, or scientific notation (e.g. 4e-5) if that would round to 0.000."""
    score = float(score)
    if score != 0 and round(score, 3) == 0:
        mantissa, exponent = f"{score:.0e}".split("e")
        return f"{mantissa}e{int(exponent)}"
    return f"{score:.3f}"


def _rect_overlap(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    return max(0, min(ax2, bx2) - max(ax1, bx1)) * max(0, min(ay2, by2) - max(ay1, by1))


def _label_candidates(x1, y1, x2, y2, label_w, label_h, img_w, img_h):
    """Up to 12 spots touching one side of the box and otherwise fully
    outside it: north/south at 3 offsets along that edge, east/west at 3
    offsets along that edge. Dropped if there's no room in the image."""
    candidates = []
    for y in (y1 - label_h, y2):  # north, south
        if 0 <= y <= img_h - label_h:
            for x in (x1, (x1 + x2 - label_w) / 2, x2 - label_w):
                candidates.append((max(0, min(x, img_w - label_w)), y))
    for x in (x2, x1 - label_w):  # east, west
        if 0 <= x <= img_w - label_w:
            for y in (y1, (y1 + y2 - label_h) / 2, y2 - label_h):
                candidates.append((x, max(0, min(y, img_h - label_h))))
    return candidates


def plot_smask(result: dict, image: Image.Image, objects: list[str], score_threshold: float = 0.0, top_k: int | None = None, alpha: float = 0.45, box_width: int = 2, box_padding: int = 10, show: bool = True) -> Image.Image:
    """Draw each detected object's mask (alpha-blended, one color per object),
    box, and "<label> <score>" tag directly on `image`. box_padding expands
    each drawn box outward from the mask (in pixels) so the frame doesn't hug
    the mask edge. Each label is placed at whichever of up to 12 spots around
    its box overlaps the least other masks/labels, staying fully in bounds.
    `objects` labels detections in score-descending order (cycled if
    shorter); top_k defaults to len(objects) so exactly one detection per
    prompted object is drawn. Set show=False to skip the matplotlib figure
    and just get the composited PIL image back."""
    top_k = len(objects) if top_k is None else top_k
    masks, boxes, scores = result["masks"], result["boxes"], result["scores"]

    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    keep = [i for i in order if scores[i] >= score_threshold][:top_k]

    overlay = np.array(image.convert("RGB")).astype(np.float32)
    mask_by_index = {}
    for i in keep:
        mask_by_index[i] = np.array(masks[i]) > 0
        color = _MASK_COLORS[i % len(_MASK_COLORS)]
        overlay[mask_by_index[i]] = overlay[mask_by_index[i]] * (1 - alpha) + color * alpha
    composited = Image.fromarray(overlay.astype(np.uint8))

    img_w, img_h = composited.size
    draw = ImageDraw.Draw(composited)
    try: font = ImageFont.truetype("arial.ttf", 11)
    except Exception: font = ImageFont.load_default(size=11)

    placed = []
    for i in keep:
        color = tuple(int(c) for c in _MASK_COLORS[i % len(_MASK_COLORS)])
        x1, y1, x2, y2 = [float(v) for v in boxes[i]]
        x1, y1 = max(0, x1 - box_padding), max(0, y1 - box_padding)
        x2, y2 = min(img_w, x2 + box_padding), min(img_h, y2 + box_padding)
        draw.rectangle([x1, y1, x2, y2], outline=color, width=box_width)

        text = f"{objects[i % len(objects)]} {_format_score(scores[i])}"
        label_w, label_h = draw.textbbox((0, 0), text, font=font)[2:]
        candidates = _label_candidates(x1, y1, x2, y2, label_w, label_h, img_w, img_h) or \
                     [(max(0, min(x1, img_w - label_w)), max(0, min(y1 - label_h, img_h - label_h)))]

        def overlap(pos):
            rect = (pos[0], pos[1], pos[0] + label_w, pos[1] + label_h)
            return sum(int(m[int(rect[1]):int(rect[3]), int(rect[0]):int(rect[2])].sum()) for j, m in mask_by_index.items() if j != i) \
                 + sum(_rect_overlap(rect, p) for p in placed)

        text_x, text_y = min(candidates, key=overlap)
        placed.append((text_x, text_y, text_x + label_w, text_y + label_h))
        draw.text((text_x, text_y), text, fill=color, font=font)

    if show:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 10))
        plt.imshow(composited)
        plt.axis("off")
        plt.tight_layout()
        plt.show()
    return composited


#***** multi-object orchestration *****

def label_map(masks: np.ndarray) -> np.ndarray:
    """Per-pixel integer label map from (N,H,W) boolean masks: 0=background,
    i+1=object i (masks[i]), in `masks` order. Later masks win on overlap."""
    labels = np.zeros(masks.shape[1:], dtype=np.uint8)
    for i, mask in enumerate(masks):
        labels[mask] = i + 1
    return labels


def label_map_image(masks: np.ndarray) -> Image.Image:
    """Render label_map() as flat solid colors (one per object, matching
    _MASK_COLORS), no alpha blending, no boxes, no overlap -- what the raw
    per-object masks actually look like."""
    labels = label_map(masks)
    canvas = np.zeros((*labels.shape, 3), dtype=np.uint8)
    for i in range(1, int(labels.max()) + 1):
        canvas[labels == i] = _MASK_COLORS[(i - 1) % len(_MASK_COLORS)]
    return Image.fromarray(canvas)
