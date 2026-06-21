from pathlib import Path
from datetime import datetime, timezone

import modal
import numpy as np
from PIL import Image, ImageDraw

from io_utils import save, load, append, symlink, Timing, copy

#***** 1 resize *****

def resize(image: Image.Image, left: float, right: float, up: float, down: float, max_side:int) -> Image.Image:
    """Crop so only box in the image. Downscale image so segmentation model inference runs faster."""
    # crop
    w, h = image.size
    image = image.crop((int(w * left), int(h * up), int(w * right), int(h * down)))

    # downscale image so its longest side <= max_side, preserving aspect ratio
    w, h = image.size
    if max(w, h) <= max_side: return image
    scale = max_side / max(w, h)
    return image.resize((int(w * scale), int(h * scale)), resample=Image.LANCZOS)


#***** 2 segment mask on modal *****

app = modal.App("segment")

_src3 = Path(__file__).parent

modal_image = (
    modal.Image.from_registry("nvidia/cuda:12.8.0-devel-ubuntu22.04", add_python="3.12")
    .entrypoint([])
    .apt_install("git")
    .pip_install("git+https://github.com/facebookresearch/sam3#egg=sam3[notebooks]", "Pillow", "torch", "torchvision")
    .add_local_file(str(_src3 / "io_utils.py"), "/root/io_utils.py")
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


# Declared at module level so app.run() can hydrate it. Do not instantiate inside functions.
_segmenter = Segmenter()

# async def segment(image: Image.Image, prompt: str, is_empty_box:bool) -> Image.Image:
#     """Segment an image. Call with `await` inside `async with app.run.aio():`.
#     Returns a grayscale PIL image — white (255) where the object is, black (0) elsewhere."""
#     array = np.array(image.convert("RGB"), dtype=np.uint8)
#     mask = np.zeros(array.shape[:2]) if is_empty_box else await _segmenter.run.remote.aio(array, prompt)
#     return Image.fromarray(mask.astype(np.uint8) * 255, mode="L")

def segment(image: Image.Image, prompt: str, is_empty_box: bool) -> Image.Image:
    """Segment an image. Returns a grayscale PIL image — white (255) where the object is, black (0) elsewhere."""
    array = np.array(image.convert("RGB"), dtype=np.uint8)
    mask = np.zeros(array.shape[:2]) if is_empty_box else _segmenter.run.remote(array, prompt)
    return Image.fromarray(mask.astype(np.uint8) * 255, mode="L")

#***** 3 downsample segment mask *****

def downsample(mask: Image.Image, out_h: int, out_w: int) -> Image.Image:
    arr = np.array(mask, dtype=np.float32) / 255.0  # (H, W) in {0, 1}
    H, W = arr.shape
    block_h, block_w = H // out_h, W // out_w
    # Trim to exact multiples so the reshape is valid; loses at most (block_h-1) edge pixels.
    arr = arr[:block_h * out_h, :block_w * out_w]
    downsampled_mask = arr.reshape(out_h, block_h, out_w, block_w).mean(axis=(1, 3))
    return Image.fromarray((downsampled_mask * 255).astype(np.uint8), mode="L")

#***** 4 center of mass *****

def center_of_mass(mask: Image.Image) -> tuple[float, float]:
    """Return (row, col) center of mass of a mask with values in [0, 1]."""
    mask = np.array(mask, dtype=np.float32) / 255.0
    H, W = mask.shape
    rows = np.arange(H)
    cols = np.arange(W)
    total = mask.sum()
    row = (mask * rows[:, None]).sum() / total
    col = (mask * cols[None, :]).sum() / total
    return row.item(), col.item()


#***** 5 overhead image *****

def make_overhead(overhead: Image.Image, segment_mask: Image.Image, com: tuple[float, float], is_empty_box:bool, speaker: int | None = None) -> Image.Image:
    # load lazily to not interefere with modal import
    SPEAKER_IMG = Path(__file__).parent.parent / "assets/speaker.png"
    # SPEAKER_POSITION = {'1000': (0, 0.5), '0100': (1/3, 0), '0010': (2/3,0), '0001': (1,0.5)} # assume bottom left corner is (0, 0)
    SPEAKER_POSITION = {3: (0, 0.5), 4: (1/3, 0), 5: (2/3,0), 6: (1,0.5)} # assume bottom left corner is (0, 0)
    SPEAKER_POSITION = {1: (1, 0), 2: (1, 0.7), 3: (0.8, 1), 4: (0.6, 1), 5: (0.4, 1), 6: (0.2, 1), 7: (0, 0.7), 8: (0,0)}

    overhead = overhead.convert("RGBA")

    # green overlay from the mask
    mask_arr = np.array(segment_mask)  # (H, W) uint8 in [0, 255]
    color = np.zeros((*mask_arr.shape, 4), dtype=np.uint8)
    color[..., 0] = 0    # green (0.0, 0.8, 0.2) from old segment.py
    color[..., 1] = 204
    color[..., 2] = 51
    color[..., 3] = (mask_arr * 0.35).astype(np.uint8)  # 35% opacity where mask is white
    overlay = Image.fromarray(color, mode="RGBA").resize(overhead.size, resample=Image.NEAREST)
    overhead = Image.alpha_composite(overhead, overlay)
    W, H = overhead.size

    # crosshair at center of mass (com is in mask coordinates — scale to image size)
    if not is_empty_box:
        draw = ImageDraw.Draw(overhead)
        cx, cy = int(com[1]), int(com[0])
        r = max(3, W // 60)
        draw.line([(cx - r, cy), (cx + r, cy)], fill=(144, 238, 144, 255), width=3)
        draw.line([(cx, cy - r), (cx, cy + r)], fill=(144, 238, 144, 255), width=3)

    if speaker is None: return overhead.convert("RGB")

    # add gray padding around the image and place a speaker icon
    pad = max(W, H) // 5  # padding size relative to image
    canvas_w, canvas_h = W + 2 * pad, H + 2 * pad
    canvas = Image.new("RGBA", (canvas_w, canvas_h), (220, 220, 220, 255))
    canvas.paste(overhead, (pad, pad))  # original image centered in canvas

    spk_h = H // 3  # speaker height = 1/3 of the image height
    speaker_img = Image.open(SPEAKER_IMG).convert("RGBA")
    spk_w = int(speaker_img.width * spk_h / speaker_img.height)  # preserve aspect ratio
    spk = speaker_img.resize((spk_w, spk_h), resample=Image.LANCZOS)

    # SPEAKER_POSITION values are (x_frac, y_frac) of the original image area;
    # offset by pad to account for the added border
    x_frac, y_frac = SPEAKER_POSITION[speaker]
    spk_cx = pad + int(x_frac * W)
    spk_cy = pad + int((1 - y_frac) * H)  # flip y: image y=0 is top, but our coords have y=0 at bottom
    canvas.paste(spk, (spk_cx - spk_w // 2, spk_cy - spk_h // 2), mask=spk)

    # add speaker id
    draw = ImageDraw.Draw(canvas)
    font_size = spk_h // 2
    try:
        from PIL import ImageFont
        font = ImageFont.truetype("arial.ttf", font_size)
    except:
        font = ImageFont.load_default()
    draw.text((spk_cx, spk_cy), str(speaker), fill=(255, 255, 255, 255), font=font, anchor="mm")

    return canvas.convert("RGB")

#***** 6 define each stage of the pipeline ******

DEFAULT_PROMPT = "A black metal cube sitting on the floor of an open cardboard box from a bird's eye view."
SHARED_ARTIFACTS = ["00_raw_overhead.png", "01_resized_overhead.png", "02_segment_mask.png", "03_downsampled_segment_mask.png", "04_com.jsonl", "y.npy"]
COPIED_ARTIFACTS = ["times.jsonl", "metadata.jsonl"]

def capture_overhead(overhead_cam, capture_image_fxn, output_dir:Path, verbose:int=1, do_save:bool=1) -> Image.Image:
    output_id = output_dir.name

    # capture overhead image
    import cv2
    with Timing(f"[output {output_id}] capture the overhead image: ", enabled=verbose >= 2):
        image, _ = capture_image_fxn(overhead_cam)
        image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        save(image, output_dir / "00_raw_overhead.png", do_save)
        append({'capture_overhead': datetime.now(timezone.utc).isoformat()}, output_dir / 'times.jsonl', do_save)

    return image

def process_overhead(raw_overhead: Image.Image, output_dir: Path, left: float = 0.15, right: float = 0.67, up: float = 0.08, down: float = 0.7, max_side:int=256,
                      prompt: str = DEFAULT_PROMPT, out_h: int = 40, out_w: int = 20, is_empty_box:bool=False, verbose: int = 1, do_save:bool=True):
    output_id = output_dir.name
    if verbose >= 2: print(f"[output {output_id}] Box is {'not '*int(not is_empty_box)}empty")

    # resize image
    with Timing(f"[output {output_id}] resize the overhead image: ", enabled=verbose >= 2):
        resized_overhead = resize(raw_overhead, left, right, up, down, max_side)
        save(resized_overhead, output_dir / "01_resized_overhead.png", do_save)
        append({'resize_overhead': datetime.now(timezone.utc).isoformat()}, output_dir / 'times.jsonl', do_save)

    # segment mask on modal with SAM3
    with Timing(f"[output {output_id}] segment the overhead image: ", enabled=verbose >= 2):
        segment_mask = segment(resized_overhead, prompt, is_empty_box) # todo: make sync, not async
        save(segment_mask, output_dir / "02_segment_mask.png", do_save)
        append({'segment_overhead': datetime.now(timezone.utc).isoformat()}, output_dir / 'times.jsonl', do_save)

    # downsample
    with Timing(f"[output {output_id}] downsample the overhead image: ", enabled=verbose >= 2):
        downsampled_segment_mask = downsample(segment_mask, out_h, out_w)
        save(downsampled_segment_mask, output_dir / "03_downsampled_segment_mask.png", do_save)
        save(np.array(downsampled_segment_mask, dtype=np.float32) / 255.0, output_dir / "y.npy", do_save)
        append({'downsample_overhead': datetime.now(timezone.utc).isoformat()}, output_dir / 'times.jsonl', do_save)

    # center of mass
    with Timing(f"[output {output_id}] center of mass of the overhead image: ", enabled=verbose >= 2):
        com = (-1, -1) if is_empty_box else center_of_mass(segment_mask)
        downsampled_com = (-1, -1) if is_empty_box else center_of_mass(downsampled_segment_mask)
        save({"com": com, "downsampled_com": downsampled_com}, output_dir / "04_com.jsonl", do_save)
        append({'com_overhead': datetime.now(timezone.utc).isoformat()}, output_dir / 'times.jsonl', do_save)
        append([{"com": com}, {"downsampled_com": downsampled_com}], output_dir / "metadata.jsonl", do_save)
        if verbose >= 2: print(f"[output {output_id}] {com=}\t{downsampled_com=}")

def visualize_overhead(speaker, sample_dir:Path, output_dir:Path, is_empty_box:bool, verbose:int=1, do_save:bool=True) -> Image.Image:
    sample_id = sample_dir.name

    # make viz of overhead image for the current sample
    with Timing(f"[sample {sample_id}] make viz of the overhead image: ", enabled=verbose >= 2):
        resized_overhead = load(sample_dir / "outputs/01_resized_overhead.png")
        segment_mask = load(sample_dir / "outputs/02_segment_mask.png")
        com = load(sample_dir / "outputs/04_com.jsonl")[0]['com']

        overhead = make_overhead(resized_overhead, segment_mask, com, is_empty_box, speaker)
        save(overhead, sample_dir / "outputs/05_overhead.png")
        symlink(sample_dir / "outputs/05_overhead.png", sample_dir / "overhead.png")

        timestamp = datetime.now(timezone.utc).isoformat()
        append({f'visualize_overhead_{sample_id}': timestamp}, output_dir / 'times.jsonl', do_save)
        append({"sample_id": sample_id, "sample_dir": sample_dir, "time": timestamp}, output_dir / "samples.jsonl", do_save)

    return overhead
