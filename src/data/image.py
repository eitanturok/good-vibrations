"""Take overhead image of what's in the box, segment it, and process it."""

from pathlib import Path
from datetime import datetime, timezone

import modal
import numpy as np
from PIL import Image, ImageDraw

from utils.io_utils import save, load, append, Timing


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


#***** 2 segment image on modal *****

def segment(image: Image.Image, prompt: str, is_empty_box: bool) -> Image.Image:
    """Segment an image. Returns a grayscale PIL image — white (255) where the object is, black (0) elsewhere."""
    # Connect to modal app
    segmenter = modal.Cls.from_name("segment", "Segmenter")()
    # segment
    image = np.array(image.convert("RGB"), dtype=np.uint8)
    mask = np.zeros(image.shape[:2]) if is_empty_box else segmenter.run.remote(image, prompt)
    return Image.fromarray(mask.astype(np.uint8) * 255, mode="L")

#***** 3 center of mass *****

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


#***** 4 add smask to overhead image *****

def draw_mask(overhead: Image.Image, segment_mask: Image.Image, com: tuple[float, float], is_empty_box:bool):
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
        draw.line([(cx - r, cy), (cx + r, cy)], fill=(144, 238, 144, 255), width=2)
        draw.line([(cx, cy - r), (cx, cy + r)], fill=(144, 238, 144, 255), width=2)

    return overhead.convert("RGB")

#***** 5 add speaker to overhead image *****

def speaker_padding(w: int, h: int) -> int:
    """Padding (px) draw_speaker adds around an image of size (w, h) to make room for the speaker icon."""
    return max(w, h) // 5

def draw_speaker(overhead: Image.Image, speaker: int | None = None) -> Image.Image:
    # load lazily to not interefere with modal import
    SPEAKER_IMG = Path(__file__).parents[2] / "assets/speaker.png"
    SPEAKER_POSITION = {1: (1, 0), 2: (1, 0.7), 3: (0.8, 1), 4: (0.6, 1), 5: (0.4, 1), 6: (0.2, 1), 7: (0, 0.7), 8: (0,0)}

    overhead = overhead.convert("RGBA")
    W, H = overhead.size

    # add gray padding around the image and place a speaker icon
    pad = speaker_padding(W, H)  # padding size relative to image
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
    # push the speaker fully into the padding so it clears the image edge
    spk_cx += (spk_w // 2 + 4) * ((x_frac == 1) - (x_frac == 0))
    spk_cy += (spk_h // 2 + 4) * ((y_frac == 0) - (y_frac == 1))
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

#***** 6 group each stage of the pipeline; add timing, save to files ******

DEFAULT_PROMPT = "A black metal cube sitting on the floor of an open cardboard box from a bird's eye view."

def capture_overhead(overhead_cam, capture_image_fxn, output_dir:Path, verbose:int=1, do_save:bool=1) -> Image.Image:
    output_id = output_dir.name
    with Timing(f"[output {output_id}] capture the overhead image: ", enabled=verbose >= 2):
        image, _ = capture_image_fxn(overhead_cam)
        import cv2
        image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        append({'capture_overhead': datetime.now(timezone.utc).isoformat()}, output_dir / 'times.jsonl', do_save)
    return image

def save_overhead(image, output_dir:Path, verbose:int=1, do_save:bool=1):
    output_id = output_dir.name
    with Timing(f"[output {output_id}] save the overhead image: ", enabled=verbose >= 2):
        save(image, output_dir / "00_raw.png", do_save)
        append({'save_overhead': datetime.now(timezone.utc).isoformat()}, output_dir / 'times.jsonl', do_save)

def process_overhead(raw_overhead: Image.Image, output_dir: Path, left: float = 0.15, right: float = 0.67, up: float = 0.08, down: float = 0.7, max_side:int=256,
                      prompt: str = DEFAULT_PROMPT, is_empty_box:bool=False, verbose: int = 1, do_save:bool=True):
    output_id = output_dir.name
    if verbose >= 2: print(f"[output {output_id}] Box is {'not '*int(not is_empty_box)}empty")

    # resize image
    with Timing(f"[output {output_id}] resize the overhead image: ", enabled=verbose >= 2):
        resized_overhead = resize(raw_overhead, left, right, up, down, max_side)
        save(resized_overhead, output_dir / "01_cropped.png", do_save)
        append({'resized_overhead': datetime.now(timezone.utc).isoformat()}, output_dir / 'times.jsonl', do_save)

    # segment mask on modal with SAM3
    with Timing(f"[output {output_id}] segment the overhead image: ", enabled=verbose >= 2):
        segment_mask = segment(resized_overhead, prompt, is_empty_box) # todo: make sync, not async
        save(segment_mask, output_dir / "02_smask.png", do_save)
        save(np.array(segment_mask, dtype=np.float32) / 255.0, output_dir / "03_smask.npy", do_save)
        append({'segment_mask': datetime.now(timezone.utc).isoformat()}, output_dir / 'times.jsonl', do_save)

    # center of mass
    with Timing(f"[output {output_id}] center of mass of the overhead image: ", enabled=verbose >= 2):
        com = (-1, -1) if is_empty_box else center_of_mass(segment_mask)
        append({'center_of_mass': datetime.now(timezone.utc).isoformat()}, output_dir / 'times.jsonl', do_save)
        append([{"com": com}], output_dir / "metadata.jsonl", do_save)
        if verbose >= 2: print(f"[output {output_id}] {com=}")

    # make overhead image with segmentation mask and center of mass, but no speaker
    with Timing(f"[output {output_id}] visualize overhead image: ", enabled=verbose >= 2):
        overhead_with_smask = draw_mask(resized_overhead, segment_mask, com, is_empty_box)
        save(overhead_with_smask, output_dir / "04_overhead_with_smask.png", do_save)
        append({'viz_overhead_with_smask': datetime.now(timezone.utc).isoformat()}, output_dir / 'times.jsonl', do_save)

    # update tracking status
    append({'process_overhead': datetime.now(timezone.utc).isoformat()}, output_dir / 'times.jsonl', do_save)

    return overhead_with_smask

def make_overhead(sample_dir:Path, output_dir:Path, speaker:int, verbose:int=1, do_save:bool=True) -> Image.Image:
    # add speaker to overhead image with smask, center of mass
    sample_id = sample_dir.name
    with Timing(f"[sample {sample_id}] visualize overhead image: ", enabled=verbose >= 2):
        overhead_with_smask = load(sample_dir / 'image/04_overhead_with_smask.png', 'image')
        overhead = draw_speaker(overhead_with_smask, speaker)
        save(overhead, sample_dir / 'image/05_overhead_with_speaker.png', do_save)
        timestamp = datetime.now(timezone.utc).isoformat()
        append({f'viz_overhead_with_smask_and_speaker/sample_{sample_id}': timestamp}, output_dir / 'times.jsonl', do_save)
        append({f'viz_overhead_with_smask_and_speaker': timestamp}, sample_dir / 'times.jsonl', do_save)

    return overhead
