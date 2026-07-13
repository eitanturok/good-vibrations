"""Take overhead image of what's in the box, segment it, and process it."""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from datetime import datetime, timezone

import modal
import numpy as np
from PIL import Image, ImageDraw

from utils.io_utils import save, load, append, Timing
from utils.metrics import center_of_mass


#***** 1 crop *****

def crop(image: Image.Image, left: float, right: float, up: float, down: float, max_side:int) -> Image.Image:
    """Crop so only box in the image. Downscale image so segmentation model inference runs faster."""
    # crop
    w, h = image.size
    return image.crop((int(w * left), int(h * up), int(w * right), int(h * down)))

#***** 2 segment image on modal *****

def _organize(object_names, all_detections):
    names, scores, boxes, masks = [], [], [], []
    for object_name, detections in zip(object_names, all_detections):
        for i, (score, box, mask) in enumerate(detections):
            name = f"{object_name}{i + 1}"
            names.append(name); scores.append(score); boxes.append(box); masks.append(mask)
    return {"names": names, "scores": np.array(scores), "boxes": np.stack(boxes), "masks": np.stack(masks)}


def segment(image: Image.Image, objects: dict[str, int], prompts: dict[str, str], is_empty_box: bool = False, segment_scale: float = 1.0) -> dict:
    """Segment every object in parallel on modal and return the flattened per-instance {names, scores, boxes, masks} dict. """
    w, h = image.size
    object_names = list(objects.keys())
    counts = [objects[t] for t in object_names]

    if is_empty_box: return _organize(object_names, [[(0.0, np.zeros(4), np.zeros((h, w), dtype=bool))] * c for c in counts])

    # launch parallel segmentation on modal via threads + .remote(), not .spawn()+.get():
    # .spawn()+.get() polls Modal's control plane for the result rather than holding a live
    # connection, which measured 3-5s of overhead per call even on an already-warm container
    # (vs ~0.7-0.9s for .remote()) -- see segment2_results.md. Real threads give the same
    # parallelism across objects without that polling tax.
    # top_k=count filters to the top-scoring detections server-side, before the result gets
    # serialized -- shipping all ~200 raw candidate masks over the network when only `count`
    # are ever used cost 10s+ per call for nothing (see segment2.py's payload_size_test).
    segmenter = modal.Cls.from_name("segment", "Segmenter")()
    with ThreadPoolExecutor(max_workers=len(object_names)) as executor:
        outs = list(executor.map(lambda args: segmenter.run.remote(image, prompts[args[0]], scale=segment_scale, top_k=args[1]), zip(object_names, counts)))

    all_detections = []
    for out, count in zip(outs, counts):
        order = np.argsort(out["scores"])[::-1][:count]
        detections = [(float(out["scores"][i]), np.asarray(out["boxes"][i]), np.asarray(out["masks"][i], dtype=bool)) for i in order]
        detections += [(0.0, np.zeros(4), np.zeros((h, w), dtype=bool))] * (count - len(detections))
        all_detections.append(detections)

    return _organize(object_names, all_detections)

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
    draw = ImageDraw.Draw(overhead)
    if not is_empty_box:
        cx, cy = int(com[1]), int(com[0])
        r = max(3, W // 60)
        draw.line([(cx - r, cy), (cx + r, cy)], fill=(144, 238, 144, 255), width=2)
        draw.line([(cx, cy - r), (cx, cy + r)], fill=(144, 238, 144, 255), width=2)

    # image resolution, bottom-right corner
    pad = max(4, W // 100)
    draw.text((W - pad, H - pad), f"{W}×{H}", fill=(255, 255, 255, 255), anchor="rd")

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
DEFAULT_OBJECTS = {"cube": 1}
DEFAULT_PROMPTS = {"cube": DEFAULT_PROMPT}

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
                      objects: dict[str, int] = DEFAULT_OBJECTS, prompts: dict[str, str] = DEFAULT_PROMPTS, is_empty_box:bool=False, segment_scale: float = 1.0, verbose: int = 1, do_save:bool=True):
    from src.data.segment import label_map, label_map_image, plot_smask

    output_id = output_dir.name
    if verbose >= 2: print(f"[output {output_id}] Box is {'not '*int(not is_empty_box)}empty")

    # crop image
    with Timing(f"[output {output_id}] crop the overhead image: ", enabled=verbose >= 2):
        cropped_overhead = crop(raw_overhead, left, right, up, down, max_side)
        save(cropped_overhead, output_dir / "01_cropped.png", do_save)
        append({'cropped_overhead': datetime.now(timezone.utc).isoformat()}, output_dir / 'times.jsonl', do_save)

    # segment image
    with Timing(f"[output {output_id}] segment the image: ", enabled=verbose >= 2):
        result = segment(cropped_overhead, objects, prompts, is_empty_box, segment_scale)

        for name, mask in zip(result["names"], result["masks"]):
            save(Image.fromarray((mask * 255).astype(np.uint8), mode="L"), output_dir / f"smasks/{name}.png", do_save)
            save(mask.astype(np.float32), output_dir / f"smasks/{name}.npy", do_save)

        object_records = [{"name": name, "com": center_of_mass(mask), "score": float(score), "box": np.asarray(box).tolist()}
                           for name, score, box, mask in zip(result["names"], result["scores"], result["boxes"], result["masks"])]
        append(object_records, output_dir / "smasks/metadata.jsonl", do_save)
        save(label_map_image(result["masks"]), output_dir / "smasks/all.png", do_save)
        save(label_map(result["masks"]), output_dir / "smasks/all.npy", do_save)
        append({'segment_and_save': datetime.now(timezone.utc).isoformat()}, output_dir / 'times.jsonl', do_save)
        if verbose >= 2: print(f"[output {output_id}] {object_records=}")

    # union mask (boolean OR of all object masks) — kept for backward compat
    with Timing(f"[output {output_id}] union mask of the overhead image: ", enabled=verbose >= 2):
        union_mask = result["masks"].any(axis=0)
        segment_mask = Image.fromarray(union_mask.astype(np.uint8) * 255, mode="L")
        save(segment_mask, output_dir / "02_smask.png", do_save)
        save(union_mask.astype(np.float32), output_dir / "03_smask.npy", do_save)
        append({'union_mask': datetime.now(timezone.utc).isoformat()}, output_dir / 'times.jsonl', do_save)

    # per-object COMs + their average, across all real (non-placeholder) objects
    with Timing(f"[output {output_id}] center of mass of the overhead image: ", enabled=verbose >= 2):
        real_records = [r for r in object_records if r["score"] > 0]
        avg_com = (-1.0, -1.0) if not real_records else tuple(np.mean([r["com"] for r in real_records], axis=0))
        coms = {r["name"]: r["com"] for r in object_records}
        append({'center_of_mass': datetime.now(timezone.utc).isoformat()}, output_dir / 'times.jsonl', do_save)
        append([{"coms": coms}, {"avg_com": avg_com}], output_dir / "metadata.jsonl", do_save)
        if verbose >= 2: print(f"[output {output_id}] {coms=} {avg_com=}")

    # make overhead image with segmentation mask and average center of mass, but no speaker
    with Timing(f"[output {output_id}] visualize overhead image: ", enabled=verbose >= 2):
        overhead_masked = draw_mask(cropped_overhead, segment_mask, avg_com, is_empty_box)
        save(overhead_masked, output_dir / "04_overhead_masked.png", do_save)
        append({'viz_overhead_masked': datetime.now(timezone.utc).isoformat()}, output_dir / 'times.jsonl', do_save)

    # make overhead image with masks + boxes + confidence scores for every object
    with Timing(f"[output {output_id}] score overhead image: ", enabled=verbose >= 2):
        overhead_scored = plot_smask(result, cropped_overhead, result["names"], show=False)
        save(overhead_scored, output_dir / "05_overhead_scored.png", do_save)
        append({'viz_overhead_scored': datetime.now(timezone.utc).isoformat()}, output_dir / 'times.jsonl', do_save)

    # update tracking status
    append({'process_overhead': datetime.now(timezone.utc).isoformat()}, output_dir / 'times.jsonl', do_save)

    return overhead_scored, result

def make_overhead(sample_dir:Path, output_dir:Path, speaker:int, verbose:int=1, do_save:bool=True) -> Image.Image:
    # add speaker to overhead image with smask, center of mass
    sample_id = sample_dir.name
    with Timing(f"[sample {sample_id}] visualize overhead image: ", enabled=verbose >= 2):
        overhead_masked = load(sample_dir / 'image/04_overhead_masked.png', 'image')
        overhead = draw_speaker(overhead_masked, speaker)
        save(overhead, sample_dir / 'image/06_overhead_speaker.png', do_save)
        timestamp = datetime.now(timezone.utc).isoformat()
        append({f'viz_overhead_speaker/sample_{sample_id}': timestamp}, output_dir / 'times.jsonl', do_save)
        append({f'viz_overhead_speaker': timestamp}, sample_dir / 'times.jsonl', do_save)

    return overhead
