"""Segment an image on modal using SAM3"""
import io
from PIL import Image

import modal
import numpy as np

app = modal.App("segment")

def _download_sam3_weights():
    """Runs once at image build time so the ~3GB checkpoint is baked into the image layer."""
    from transformers import Sam3Model, Sam3Processor
    Sam3Processor.from_pretrained("facebook/sam3")
    Sam3Model.from_pretrained("facebook/sam3")


modal_image = (
    modal.Image.from_registry("nvidia/cuda:12.8.0-devel-ubuntu22.04", add_python="3.12")
    .entrypoint([])
    .apt_install("git")
    .pip_install("transformers", "Pillow", "torch", "torchvision", "accelerate")
    .run_function(_download_sam3_weights, secrets=[modal.Secret.from_name("huggingface")])
    .env({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"})  # after baking -- offline mode would break the download step itself
)

@app.cls(
    gpu="A10G",
    image=modal_image,
    secrets=[modal.Secret.from_name("huggingface")],
    scaledown_window=60 * 10,  # shut down after 10 min idle
)
class Segmenter:
    IMAGE_SIZE = 672

    @modal.enter()
    def load(self):
        import torch
        # much faster than `import transformers`b/c transformers is a huge package
        from transformers import Sam3Config
        from transformers.models.sam3.modeling_sam3 import Sam3Model
        from transformers.models.sam3.processing_sam3 import Sam3Processor

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = Sam3Processor.from_pretrained("facebook/sam3", size={"height": self.IMAGE_SIZE, "width": self.IMAGE_SIZE})
        config = Sam3Config.from_pretrained("facebook/sam3")
        config.vision_config.image_size = self.IMAGE_SIZE
        self.model = Sam3Model.from_pretrained("facebook/sam3", config=config, torch_dtype=torch.bfloat16, device_map=self.device)
        if self.device == "cuda": torch.cuda.synchronize()

    def downsample(self, image: np.ndarray, scale: float) -> np.ndarray:
        """Resize both sides by scale (e.g. 0.25 shrinks each side to a quarter),
        preserving aspect ratio. Remembers the original size (for upsample()) and
        the scale (for rescaling boxes)."""
        h, w = image.shape[:2]
        self.original_size = (w, h)  # PIL-style (width, height), used by upsample()
        self.scale = scale
        if scale == 1.0: return image
        resized = Image.fromarray(image).resize((round(w * scale), round(h * scale)), resample=Image.LANCZOS)
        return np.array(resized)

    def upsample(self, mask: np.ndarray) -> np.ndarray:
        """Resize a boolean/float segmentation mask back to the original image dimensions."""
        mask_image = Image.fromarray((mask > 0).astype(np.uint8) * 255)
        mask_image = mask_image.resize(self.original_size, resample=Image.NEAREST)
        return np.array(mask_image) > 0

    def inference(self, image: np.ndarray, prompts: list[str], top_k: list[int | None] | None = None) -> list[dict]:
        """Run SAM3 on an already-downsampled image array, prompted by text and/or boxes.

        Returns a dict with:
            - scores (np.ndarray): confidence score per kept instance.
            - boxes (np.ndarray): (top_left_x, top_left_y, bottom_right_x, bottom_right_y) per kept instance, in the input image's coordinates.
            - masks (np.ndarray): binary mask per kept instance, shape (num_instances, height, width).
        """
        import torch
        with torch.no_grad():

            n = len(prompts)
            top_k = top_k if top_k is not None else [None] * n

            # vision backbone: run once on the single image
            img_inputs = self.processor(images=image, return_tensors="pt").to(self.device)
            pixel_values = img_inputs["pixel_values"].to(torch.bfloat16)
            if self.device == "cuda": torch.cuda.synchronize()
            vision_embeds = self.model.get_vision_features(pixel_values=pixel_values)
            if self.device == "cuda": torch.cuda.synchronize()

            # broadcast the single image's features across n prompts (view, not a recompute)
            vision_embeds.fpn_hidden_states = tuple(f.expand(n, *f.shape[1:]) for f in vision_embeds.fpn_hidden_states)
            vision_embeds.fpn_position_encoding = tuple(p.expand(n, *p.shape[1:]) for p in vision_embeds.fpn_position_encoding)

            # text encoder + decoder: batched over all n prompts
            text_inputs = self.processor(text=prompts, return_tensors="pt").to(self.device)
            outputs = self.model(vision_embeds=vision_embeds, input_ids=text_inputs["input_ids"], attention_mask=text_inputs["attention_mask"])
            if self.device == "cuda": torch.cuda.synchronize()

            target_sizes = img_inputs["original_sizes"].tolist() * n  # same image, repeated per prompt
            results = self.processor.post_process_instance_segmentation(outputs, threshold=0.0, mask_threshold=0.5, target_sizes=target_sizes)

            out = []
            for result, k in zip(results, top_k):
                if k is not None and len(result["scores"]) > k:
                    order = torch.argsort(result["scores"], descending=True)[:k]
                    result = {key: v[order] for key, v in result.items()}
                out.append({
                    "scores": result["scores"].float().cpu().numpy(),
                    "boxes": result["boxes"].float().cpu().numpy(),
                    "masks": result["masks"].cpu().numpy().astype(bool),
                })
            return out

    @modal.method()
    def run(self, image_bytes: bytes, prompts: list[str], scale: float = 1.0, top_k: list[int] | None = None) -> list[dict]:
        """Downsample, run inference, upsample, and then bit pack the resultant masks."""
        image = np.array(Image.open(io.BytesIO(image_bytes)).convert("RGB"))
        # downsample so SAM-3 inference runs faster
        downsampled = self.downsample(image, scale)
        results = self.inference(image=downsampled, prompts=prompts, top_k=top_k)

        # upsample and bit-pack the masks
        for result in results:
            masks = np.stack([self.upsample(mask) for mask in result["masks"]]) if len(result["masks"]) else result["masks"]
            result["boxes"] = result["boxes"] / self.scale
            result["masks_shape"] = masks.shape
            result["masks_packed"] = np.packbits(masks)
            del result["masks"]

        return results