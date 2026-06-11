import json, shutil
from pathlib import Path
from datetime import datetime, timezone

import modal
import cv2
import numpy as np
from PIL import Image, ImageDraw

from src3.utils import save, load, append, symlink, Timing, copy

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
    .add_local_file(str(_src3 / "utils.py"), "/root/utils.py")
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

async def segment(image: Image.Image, prompt: str, is_empty_box:bool) -> Image.Image:
    """Segment an image. Call with `await` inside `async with app.run.aio():`.
    Returns a grayscale PIL image — white (255) where the object is, black (0) elsewhere."""
    array = np.array(image.convert("RGB"), dtype=np.uint8)
    mask = np.zeros(array.shape[:2]) if is_empty_box else await _segmenter.run.remote.aio(array, prompt)
    return Image.fromarray(mask.astype(np.uint8) * 255, mode="L")

#***** 3 downsample segment mask *****

def downsample(mask: Image.Image, out_h: int, out_w: int) -> Image.Image:
    arr = np.array(mask, dtype=np.float32) / 255.0  # (H, W) in {0, 1}
    H, W = arr.shape
    # Pad with nan so H and W are divisible by out_h and out_w, enabling the reshape below.
    # nan padding is excluded from nanmean, so border blocks are averaged over real pixels
    # only — the denominator is never inflated by the padding.
    ph, pw = (out_h - H % out_h) % out_h, (out_w - W % out_w) % out_w
    arr = np.pad(arr, ((0, ph), (0, pw)), constant_values=np.nan)
    H2, W2 = arr.shape
    downsampled_mask = np.nanmean(arr.reshape(out_h, H2 // out_h, out_w, W2 // out_w), axis=(1, 3))
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

def make_overhead(overhead: Image.Image, segment_mask: Image.Image, com: tuple[float, float], is_empty_box:bool, speaker: str | None = None) -> Image.Image:
    # load lazily to not interefere with modal import
    SPEAKER_IMG = Path(__file__).parent.parent / "assets" / "speakers" / "speaker.png"
    # SPEAKER_POSITION = {'1000': (0, 0.5), '0100': (1/3, 0), '0010': (2/3,0), '0001': (1,0.5)} # assume bottom left corner is (0, 0)
    SPEAKER_POSITION = {3: (0, 0.5), 4: (1/3, 0), 5: (2/3,0), 6: (1,0.5)} # assume bottom left corner is (0, 0)

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

    return canvas.convert("RGB")

#***** 6 put it all together

DEFAULT_PROMPT = "A black metal cube sitting on the floor of an open cardboard box from a bird's eye view."
SHARED_ARTIFACTS = ["00_raw_overhead.png", "01_resized_overhead.png", "02_segment_mask.png", "03_downsampled_segment_mask.png", "04_com.jsonl", "y.npy"]
COPIED_ARTIFACTS = ["times.jsonl", "data.jsonl"]

def capture_overhead(overhead_cam, capture_image_fxn, output_dir:Path, verbose:int=1, do_save:bool=1) -> Image.Image:
    output_id = output_dir.name

    # capture overhead image
    with Timing(f"[{output_id}] capture overhead image: ", enabled=verbose >= 2):
        image, _ = capture_image_fxn(overhead_cam)
        image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        save(image, output_dir / "00_raw_overhead.png", do_save)
        append({'capture': datetime.now(timezone.utc).isoformat()}, output_dir / 'times.jsonl', do_save)

    return image

def process_overhead(raw_overhead: Image.Image, output_dir: Path, left: float = 0.15, right: float = 0.67, up: float = 0.08, down: float = 0.7, max_side:int=256,
                      prompt: str = DEFAULT_PROMPT, out_h: int = 40, out_w: int = 20, is_empty_box:bool=False, verbose: int = 1, do_save:bool=1):
    output_id = output_dir.name

    if verbose >= 2: print(f"[{output_id}] Box is {'not '*int(not is_empty_box)}empty")

    # resize image
    with Timing(f"[{output_id}] resize: ", enabled=verbose >= 2):
        resized_overhead = resize(raw_overhead, left, right, up, down, max_side)
        save(resized_overhead, output_dir / "01_resized_overhead.png", do_save)
        append({'resize': datetime.now(timezone.utc).isoformat()}, output_dir / 'times.jsonl', do_save)

    # segment mask on modal with SAM3
    with Timing(f"[{output_id}] segment: ", enabled=verbose >= 2):
        segment_mask = await segment(resized_overhead, prompt, is_empty_box) # todo: make sync, not async
        save(segment_mask, output_dir / "02_segment_mask.png", do_save)
        append({'segment': datetime.now(timezone.utc).isoformat()}, output_dir / 'times.jsonl', do_save)

    # downsample
    with Timing(f"[{output_id}] downsample: ", enabled=verbose >= 2):
        downsampled_segment_mask = downsample(segment_mask, out_h, out_w)
        save(downsampled_segment_mask, output_dir / "03_downsampled_segment_mask.png", do_save)
        save(np.array(downsampled_segment_mask, dtype=np.float32) / 255.0, output_dir / "y.npy", do_save)
        append({'downsample': datetime.now(timezone.utc).isoformat()}, output_dir / 'times.jsonl', do_save)

    # center of mass
    with Timing(f"[{output_id}] center of mass: ", enabled=verbose >= 2):
        com = (-1, -1) if is_empty_box else center_of_mass(segment_mask)
        downsampled_com = (-1, -1) if is_empty_box else center_of_mass(downsampled_segment_mask)
        save({"com": com, "downsampled_com": downsampled_com}, output_dir / "04_com.jsonl", do_save)
        append({'com': datetime.now(timezone.utc).isoformat()}, output_dir / 'times.jsonl', do_save)

    # update tracking status
    append([{"com": com}, {"downsampled_com": downsampled_com}], output_dir / "data.jsonl", do_save)

def visualize_overhead(speaker, sample_dir:Path, output_dir:Path, is_empty_box:bool, verbose:int=1, do_save:bool=True) -> Image.Image:
    sample_id = sample_dir.name
    assert all((output_dir / a).exists() for a in SHARED_ARTIFACTS+COPIED_ARTIFACTS), f"[{sample_id}] Missing shared or copied artifact"

    with Timing(f"[{sample_id}] load cached artifacts: ", enabled=verbose >= 2):
        resized_overhead = load(output_dir / "01_resized_overhead.png")
        segment_mask = load(output_dir / "02_segment_mask.png")
        com = load(output_dir / "04_com.jsonl")[0]['com']

    # symlink the shared artifacts and copy the copied artifacts from output_dir to the current sample_dir
    for artifact in SHARED_ARTIFACTS: symlink(output_dir / artifact, sample_dir / f"{'' if artifact == 'y.npy' else 'outputs/'}{artifact}", do_save)
    for artifact in COPIED_ARTIFACTS: copy(output_dir / artifact, sample_dir / artifact, do_save)

    # make overhead image for the current sample
    with Timing(f"[{sample_id}] make overhead: ", enabled=verbose >= 2):
        overhead = make_overhead(resized_overhead, segment_mask, com, is_empty_box, speaker)
        save(overhead, sample_dir / "outputs/05_overhead.png")
        symlink(sample_dir / "outputs/05_overhead.png", sample_dir / "overhead.png")
        time = datetime.now(timezone.utc).isoformat()
        append({'make_overhead': time}, output_dir / 'times.jsonl', do_save)

    # update tracking status
    append({"sample_id": sample_id, "sample_dir": str(sample_dir), "time": time}, output_dir / "samples.jsonl", do_save)

    # show image
    if verbose >= 1: preview_image(overhead)
    return overhead

def capture_vibrations(cam, run_opt, speakers, audio_file:Path, sample_dir:Path, output_dir:Path, verbose:int=1, do_save:bool=True):
    sample_id, output_id = sample_dir.name, output_dir.name

    with Timing(f"[{sample_id}] record vibrations in "):
        play_audio(str(audio_file), speakers)
        n_frames = int(N_CAPTURE_SECONDS * run_opt['cam_params']['camera_FPS'])
        frame_recording, times = capture_N_frames(cam, n_frames, *cam.get_im_size()[::-1])
        if verbose >= 2: print(f'captured {n_frames} frames')
        if verbose >= 1: preview_speckle_shift_image(frame_recording)
        if verbose >= 3: preview_speckle_shift_video(frame_recording)

    # symlink audio to sample_dir
    symlink(audio_file, sample_dir / "audio.wav", do_save)

    # update tracking status
    time = datetime.now(timezone.utc).isoformat()
    append({"sample_id": sample_id, "output_id": output_id, "time": time}, audio_file.parent / "samples.jsonl", do_save)
    append({"capture_vibrations": time}, sample_dir / "times.jsonl", do_save)

    return frame_recording

def run_experiment(object, n_objects, prompt, crop_params, box, description, cam, run_opt, overhead_cam, verbose:int=1, do_save:bool=True):

    output_id = make_id(BASE_OUTPUT_DIR)
    output_dir = BASE_OUTPUT_DIR / output_id
    is_empty_box = n_objects == 0
    metadata = dict(output_id=output_id, output_dir=str(output_dir), audio=str(AUDIO_FILE), is_empty_box=is_empty_box,
                    object=object, n_objects=n_objects, box=box, prompt=prompt, description=description, **crop_params}
    append([{k:v} for k, v in metadata.items()], output_dir / 'data.jsonl', do_save)

    # capture and segment overhead image
    raw_overhead = capture_overhead(overhead_cam, capture_image, output_dir, do_save=do_save)
    process_overhead(raw_overhead, output_dir, **crop_params, prompt=prompt, is_empty_box=is_empty_box, verbose=verbose, do_save=do_save)

    # speaker for loop
    speaker = 3
    sample_id  = make_id(BASE_SAMPLE_DIR)
    sample_dir = BASE_SAMPLE_DIR / sample_id
    overhead = visualize_overhead(speaker, sample_dir, output_dir, is_empty_box, verbose, do_save=do_save)
    raw_vibrations = capture_vibrations(cam, run_opt, speaker, AUDIO_FILE, sample_dir, output_dir, verbose, do_save)
    process_vibrations(raw_vibrations) # launch pckl
    return

# async def process_outputs(speaker: str, raw_overhead_path: Path, output_dir: Path, sample_dir: Path, left: float = 0.15, right: float = 0.67, up: float = 0.08, down: float = 0.7, max_side:int=256,
#                           prompt: str = DEFAULT_PROMPT, out_h: int = 40, out_w: int = 20, verbose: int = 1, overwrite: bool = True, is_empty_box:bool|None=None, force:bool=False) -> Image.Image:
#     sample_id, output_id = sample_dir.name, output_dir.name
#     if verbose >= 1: print(f"[{sample_id}] Process Output {output_id}")

#     if is_empty_box is None: is_empty_box = 'empty' in prompt
#     if verbose >= 1: print(f"[{sample_id}] Box is {'not '*int(not is_empty_box)}empty")

#     # check if we already processed this sample and maybe overwrite it
#     status_path = output_dir / "status.jsonl"
#     prior = next((line for line in load(status_path) if line.get("sample_id") == sample_id), None) if status_path.exists() else None
#     already_processed = (sample_dir.exists() and any(sample_dir.iterdir())) or prior is not None
#     if not force and already_processed:
#         if not overwrite:
#             raise ValueError(f"[{sample_id}] sample {sample_id} already processed at {prior.get('time') if prior else '?'} in {prior.get('sample_dir') if prior else sample_dir} — pass overwrite=True to redo")
#         if verbose >= 1: print(f"[{sample_id}] overwriting previous run")
#         if sample_dir.exists(): shutil.rmtree(sample_dir)
#         if prior is not None:
#             lines = [l for l in load(status_path) if l.get("sample_id") != sample_id]
#             status_path.write_text("".join(json.dumps(l) + "\n" for l in lines))

#     # if shared artifacts already exist, skip recomputing them
#     if not force and all((output_dir / a).exists() for a in SHARED_ARTIFACTS):
#         with Timing(f"[{sample_id}] load cached artifacts: ", enabled=verbose >= 1):
#             resized_overhead = load(output_dir / "01_resized_overhead.png")
#             segment_mask = load(output_dir / "02_segment_mask.png")
#             com = load(output_dir / "04_com.jsonl")[0]['com']
#     else:
#         # resize
#         with Timing(f"[{sample_id}] resize: ", enabled=verbose >= 1):
#             resized_overhead = resize(load(raw_overhead_path), left, right, up, down, max_side)
#             save(resized_overhead, output_dir / "01_resized_overhead.png")

#         # segment mask on modal with SAM3
#         with Timing(f"[{sample_id}] segment: ", enabled=verbose >= 1):
#             segment_mask = await segment(resized_overhead, prompt, is_empty_box)
#             save(segment_mask, output_dir / "02_segment_mask.png")

#         # downsample
#         with Timing(f"[{sample_id}] downsample: ", enabled=verbose >= 1):
#             downsampled_segment_mask = downsample(segment_mask, out_h, out_w)
#             save(downsampled_segment_mask, output_dir / "03_downsampled_segment_mask.png")
#             save(np.array(downsampled_segment_mask, dtype=np.float32) / 255.0, output_dir / "y.npy")

#         # center of mass
#         with Timing(f"[{sample_id}] center of mass: ", enabled=verbose >= 1):
#             com = (-1, -1) if is_empty_box else center_of_mass(segment_mask)
#             downsampled_com = (-1, -1) if is_empty_box else center_of_mass(downsampled_segment_mask)
#             save({"com": com, "downsampled_com": downsampled_com}, output_dir / "04_com.jsonl")

#     # symlink the shared artifacts in output_dir to the current sample_dir
#     for artifact in SHARED_ARTIFACTS: symlink(output_dir / artifact, sample_dir / f"outputs/{artifact}")
#     symlink(output_dir / "y.npy", sample_dir / "y.npy")

#     # make overhead image for the current sample
#     with Timing(f"[{sample_id}] make overhead: ", enabled=verbose >= 1):
#         overhead = make_overhead(resized_overhead, segment_mask, com, is_empty_box, speaker)
#         save(overhead, sample_dir / "outputs/05_overhead.png")
#         symlink(sample_dir / "outputs/05_overhead.png", sample_dir / "overhead.png")

#     # update tracking status
#     time = datetime.now(timezone.utc).isoformat()
#     append({"sample_id": sample_id, "sample_dir": str(sample_dir), "time": time}, output_dir / "samples.jsonl")
#     append({"processed_outputs_time": time}, sample_dir / "times.jsonl")
#     append([{"output_id": output_id}, {"output_dir": str(output_dir)}], sample_dir / "data.jsonl")

#     return overhead


