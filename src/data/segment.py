"""Segment an image on modal using SAM3"""
from pathlib import Path

import modal
import numpy as np
from PIL import Image

app = modal.App("segment")

modal_image = (
    modal.Image.from_registry("nvidia/cuda:12.8.0-devel-ubuntu22.04", add_python="3.12")
    .entrypoint([])
    .apt_install("git")
    .pip_install("git+https://github.com/facebookresearch/sam3#egg=sam3[notebooks]", "Pillow", "torch", "torchvision")
)

@app.cls(
    gpu="A10G",
    image=modal_image,
    secrets=[modal.Secret.from_name("huggingface")],
    scaledown_window=60 * 10,  # shut down after 10 min idle
)
class Segmenter:
    @modal.enter()
    def load(self):
        from sam3.model_builder import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor
        self.processor = Sam3Processor(build_sam3_image_model())
        self.processor.set_confidence_threshold(0.0)

    @modal.method()
    def run(self, image_array: np.ndarray, prompt: str) -> np.ndarray:
        import torch
        with torch.autocast("cuda", dtype=torch.bfloat16):
            state = self.processor.set_image(Image.fromarray(image_array))
            out = self.processor.set_text_prompt(state=state, prompt=prompt)
        if len(out["scores"]) > 1:
            idx = out["scores"].topk(1).indices
            out["masks"] = out["masks"][idx]
        mask = out["masks"][:, 0].any(dim=0)
        return mask.cpu().numpy()  # bool (H, W)

# """Segment an image on modal using SAM3"""
# import modal
# import numpy as np
# from PIL import Image

# app = modal.App("segment")

# modal_image = (
#     modal.Image.from_registry("nvidia/cuda:12.8.0-devel-ubuntu22.04", add_python="3.12")
#     .entrypoint([])
#     .apt_install("git")
#     .pip_install("transformers", "Pillow", "torch", "torchvision", "accelerate")
# )


# @app.cls(
#     gpu="A10G",
#     image=modal_image,
#     secrets=[modal.Secret.from_name("huggingface")],
#     scaledown_window=60 * 10,  # shut down after 10 min idle
# )
# class Segmenter:
#     @modal.enter()
#     def load(self):
#         import torch
#         from transformers import Sam3Model, Sam3Processor

#         self.device = "cuda" if torch.cuda.is_available() else "cpu"
#         self.processor = Sam3Processor.from_pretrained("facebook/sam3")
#         self.model = Sam3Model.from_pretrained("facebook/sam3").to(self.device)

#     def downsample(self, image: Image.Image, scale: float) -> Image.Image:
#         """Resize both sides by scale (e.g. 0.25 shrinks each side to a quarter),
#         preserving aspect ratio. Remembers the original size (for upsample()) and
#         the scale (for rescaling boxes)."""
#         self.original_size = image.size  # (width, height)
#         self.scale = scale
#         if scale == 1.0: return image
#         w, h = image.size
#         return image.resize((round(w * scale), round(h * scale)), resample=Image.LANCZOS)

#     def upsample(self, mask: np.ndarray) -> np.ndarray:
#         """Resize a boolean/float segmentation mask back to the original image dimensions."""
#         mask_image = Image.fromarray((mask > 0).astype(np.uint8) * 255)
#         mask_image = mask_image.resize(self.original_size, resample=Image.NEAREST)
#         return np.array(mask_image) > 0

#     def inference(self, image: np.ndarray, prompt: str, input_boxes: list | None = None, input_boxes_labels: list | None = None) -> dict:
#         """Run SAM3 on an already-downsampled image array, prompted by text and/or boxes.

#         Returns a dict with:
#             - scores (np.ndarray): confidence score per kept instance.
#             - boxes (np.ndarray): (top_left_x, top_left_y, bottom_right_x, bottom_right_y)
#               per kept instance, in the input image's coordinates.
#             - masks (np.ndarray): binary mask per kept instance, shape (num_instances, height, width).
#         """
#         import torch
#         inputs = self.processor(images=image, text=prompt, return_tensors="pt", input_boxes=[input_boxes] if input_boxes else None, input_boxes_labels=[input_boxes_labels] if input_boxes_labels else None,).to(self.device)
#         # with torch.autocast("cuda", dtype=torch.bfloat16):
#         with torch.no_grad(): outputs = self.model(**inputs)
#         result = self.processor.post_process_instance_segmentation(outputs, threshold=0.0, mask_threshold=0.5, target_sizes=inputs["original_sizes"].tolist())[0]
#         result = {k: v.float().cpu().numpy() if torch.is_tensor(v) else v for k, v in result.items()}
#         return result

#     @modal.method()
#     def run(self, image: Image.Image, prompt: str, input_boxes: list | None = None, input_boxes_labels: list | None = None, scale:float=1.0) -> dict:
#         """Downsample the image, run inference, then upsample the resulting masks back to the original resolution."""

#         # downsample so SAM-3 inference runs faster
#         downsampled = self.downsample(image.convert("RGB"), scale)
#         scaled_boxes = [[x1 * self.scale, y1 * self.scale, x2 * self.scale, y2 * self.scale] for x1, y1, x2, y2 in input_boxes] if input_boxes else None

#         # inference: only convert to a numpy array right before calling the model
#         result = self.inference(image=np.array(downsampled), prompt=prompt, input_boxes=scaled_boxes, input_boxes_labels=input_boxes_labels)

#         # upsample
#         result["masks"] = np.stack([self.upsample(mask) for mask in result["masks"]]) if len(result["masks"]) else result["masks"]
#         result["boxes"] = result["boxes"] / self.scale
#         return result
