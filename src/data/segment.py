from pathlib import Path


import modal
import numpy as np
from PIL import Image


app = modal.App("segment")
_src3 = Path(__file__).parent

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
