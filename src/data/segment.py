"""Segment an image on modal"""
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

# """Segment an image on modal, using SAM3 (HF transformers) as in notebooks/49_segment.ipynb."""
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

#     def downsample(self, image: Image.Image, height: int, width: int) -> Image.Image:
#         """Resize image to (width, height), remembering its original size for upsample()."""
#         self.original_size = image.size  # (width, height)
#         return image.resize((width, height))

#     def upsample(self, mask: np.ndarray) -> np.ndarray:
#         """Resize a boolean/float segmentation mask back to the original image dimensions."""
#         mask_image = Image.fromarray((mask > 0).astype(np.uint8) * 255)
#         mask_image = mask_image.resize(self.original_size, resample=Image.NEAREST)
#         return np.array(mask_image) > 0

#     def inference(
#         self,
#         image: Image.Image | np.ndarray,
#         objects: list[str],
#         prompt: str,
#         input_boxes: list | None = None,
#         input_boxes_labels: list | None = None,
#     ) -> dict:
#         """Run SAM3 on an already-downsampled image, prompted by text and/or boxes.

#         Returns a dict with:
#             - scores (torch.Tensor): confidence score per kept instance.
#             - boxes (torch.Tensor): (top_left_x, top_left_y, bottom_right_x, bottom_right_y)
#               per kept instance, in the input image's coordinates.
#             - masks (torch.Tensor): binary mask per kept instance, shape (num_instances, height, width).
#             - objects (list[str]): the `objects` argument, passed through unchanged.
#         """
#         import torch

#         if isinstance(image, np.ndarray): image = Image.fromarray(image)
#         image = image.convert("RGB")
#         inputs = self.processor(images=image, text=prompt, return_tensors="pt", input_boxes=[input_boxes] if input_boxes else None, input_boxes_labels=[input_boxes_labels] if input_boxes_labels else None,).to(self.device)
#         with torch.no_grad(): outputs = self.model(**inputs)
#         result = self.processor.post_process_instance_segmentation(outputs, threshold=0.0, mask_threshold=0.5, target_sizes=inputs["original_sizes"].tolist())[0]
#         result["objects"] = objects
#         return result

#     @modal.method()
#     def run(
#         self,
#         image_path: str,
#         objects: list[str],
#         prompt: str,
#         input_boxes: list | None = None,
#         input_boxes_labels: list | None = None,
#         height: int = 1024,
#         width: int = 1024,
#     ) -> dict:
#         """Downsample the image, run inference, then upsample the resulting masks
#         back to the original resolution. input_boxes are given in original-image
#         coordinates and are scaled to match the downsampled image before inference."""
#         image = Image.open(image_path).convert("RGB")
#         downsampled = self.downsample(image, height, width)

#         scale_x = width / self.original_size[0]
#         scale_y = height / self.original_size[1]
#         scaled_boxes = None
#         if input_boxes:
#             scaled_boxes = [
#                 [x1 * scale_x, y1 * scale_y, x2 * scale_x, y2 * scale_y]
#                 for x1, y1, x2, y2 in input_boxes
#             ]

#         result = self.inference(
#             image=downsampled,
#             objects=objects,
#             prompt=prompt,
#             input_boxes=scaled_boxes,
#             input_boxes_labels=input_boxes_labels,
#         )

#         result["masks"] = [self.upsample(mask.numpy()) for mask in result["masks"]]
#         result["boxes"] = [
#             [x1 / scale_x, y1 / scale_y, x2 / scale_x, y2 / scale_y]
#             for x1, y1, x2, y2 in result["boxes"].tolist()
#         ]
#         return result
