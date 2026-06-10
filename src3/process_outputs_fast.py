"""Instrumented + optimized copy of process_outputs.py.

Key differences vs process_outputs.py:
- Cropped image is downscaled to MAX_SIDE px before segmentation — cuts Modal round-trip
  from ~3.5s to ~0.6s since the bottleneck is serialization/network, not GPU compute.
  Everything downstream (overlay, downsample, CoM) operates on the smaller image,
  so there is no coordinate mismatch. Saved artifacts are simply smaller.
- Segmenter caches text embeddings between calls (same prompt = no re-encoding)
- Fine-grained timing printed at every step (local + remote)
- App name is "segment-fast" so it runs as a separate Modal app and doesn't collide
"""

import json, shutil, time
from pathlib import Path
from datetime import datetime, timezone

import modal
import numpy as np
from PIL import Image, ImageDraw

from utils import save, load, append, symlink, Timing

#***** 1 crop + resize *****

MAX_SIDE = 256  # resize cropped image to this before sending to Modal

def crop(image: Image.Image, left: float, right: float, up: float, down: float) -> Image.Image:
    width, height = image.size
    return image.crop((int(width * left), int(height * up), int(width * right), int(height * down)))

def resize_for_segmentation(image: Image.Image, max_side: int = MAX_SIDE) -> Image.Image:
    """Downscale image so its longest side <= max_side, preserving aspect ratio."""
    w, h = image.size
    if max(w, h) <= max_side:
        return image
    scale = max_side / max(w, h)
    return image.resize((int(w * scale), int(h * scale)), resample=Image.LANCZOS)


#***** 2 segment mask on modal *****

app = modal.App("segment-fast")

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
    scaledown_window=60 * 10,
)
class Segmenter:
    @modal.enter()
    def load(self):
        import torch
        from sam3.model_builder import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor

        t0 = time.perf_counter()
        self.processor = Sam3Processor(build_sam3_image_model())
        self.processor.set_confidence_threshold(0.0)
        print(f"[remote load] model loaded in {time.perf_counter()-t0:.2f}s")

        # Pre-encode the default prompt once at startup so subsequent calls skip text encoding
        self._cached_prompt: str | None = None
        self._cached_text_embeds = None

    def _get_text_outputs(self, prompt: str) -> dict:
        """Return cached forward_text outputs if prompt matches, else re-encode and cache.

        Sam3Processor.set_text_prompt() calls model.backbone.forward_text([prompt]) then
        merges the result into state["backbone_out"]. We replicate that here so we can
        skip re-encoding when the prompt hasn't changed.
        """
        import torch
        if prompt != self._cached_prompt:
            t0 = time.perf_counter()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                self._cached_text_embeds = self.processor.model.backbone.forward_text(
                    [prompt], device=self.processor.device
                )
            self._cached_prompt = prompt
            print(f"[remote] text encode (cache miss): {time.perf_counter()-t0:.3f}s")
        else:
            print(f"[remote] text encode: cache hit")
        return self._cached_text_embeds

    @modal.method()
    def run(self, image_array: np.ndarray, prompt: str) -> tuple[np.ndarray, dict]:
        """Returns (mask, timing_dict). timing_dict has keys: set_image, set_text_prompt, postprocess."""
        import torch
        timing = {}

        t0 = time.perf_counter()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            t1 = time.perf_counter()
            state = self.processor.set_image(Image.fromarray(image_array))
            t2 = time.perf_counter()
            timing['set_image'] = t2 - t1

            # Inject cached text embeddings directly into backbone_out, bypassing forward_text
            text_outputs = self._get_text_outputs(prompt)
            state["backbone_out"].update(text_outputs)
            if "geometric_prompt" not in state:
                state["geometric_prompt"] = self.processor.model._get_dummy_prompt()
            out = self.processor._forward_grounding(state)
            t3 = time.perf_counter()
            timing['set_text_prompt'] = t3 - t2

        if len(out["scores"]) > 1:
            idx = out["scores"].topk(1).indices
            out["masks"] = out["masks"][idx]
        mask = out["masks"][:, 0].any(dim=0)
        t4 = time.perf_counter()
        timing['postprocess'] = t4 - t3
        timing['total_remote'] = t4 - t0

        print(f"[remote timing] set_image={timing['set_image']:.3f}s  "
              f"set_text_prompt={timing['set_text_prompt']:.3f}s  "
              f"postprocess={timing['postprocess']:.3f}s  "
              f"total={timing['total_remote']:.3f}s")

        return mask.cpu().numpy(), timing  # bool (H, W), dict

    @modal.method()
    def warmup(self, prompt: str) -> None:
        """Pre-encode the prompt at startup so the first real call skips text encoding."""
        self._get_text_outputs(prompt)


_segmenter = Segmenter()

async def segment(image: Image.Image, prompt: str, is_empty_box: bool, print_timing: bool = True) -> Image.Image:
    """Segment an image. Returns a grayscale PIL image — white where object is, black elsewhere.

    Call with `await` inside `async with app.run.aio():`.
    Prints a breakdown of where time is spent (local serialization vs. remote inference).
    """
    if is_empty_box:
        arr = np.array(image.convert("RGB"), dtype=np.uint8)
        return Image.fromarray(np.zeros(arr.shape[:2], dtype=np.uint8), mode="L")

    t0 = time.perf_counter()
    array = np.array(image.convert("RGB"), dtype=np.uint8)
    t_serialize = time.perf_counter()

    mask, remote_timing = await _segmenter.run.remote.aio(array, prompt)
    t_returned = time.perf_counter()

    if print_timing:
        t_local_pre = t_serialize - t0
        t_modal_total = t_returned - t_serialize
        t_modal_overhead = t_modal_total - remote_timing['total_remote']
        print(f"[segment timing]")
        print(f"  local serialize:   {t_local_pre*1000:.1f}ms")
        print(f"  Modal round-trip:  {t_modal_total*1000:.1f}ms total")
        print(f"    └ network+serial overhead: ~{t_modal_overhead*1000:.1f}ms")
        print(f"    └ set_image (ViT encoder): {remote_timing['set_image']*1000:.1f}ms")
        print(f"    └ set_text_prompt (decode): {remote_timing['set_text_prompt']*1000:.1f}ms")
        print(f"    └ postprocess:             {remote_timing['postprocess']*1000:.1f}ms")

    return Image.fromarray(mask.astype(np.uint8) * 255, mode="L")


async def warmup(prompt: str = "A black metal cube sitting on the floor of an open cardboard box from a bird's eye view.") -> None:
    """Pre-encode the text prompt on the remote container so the first real call is faster."""
    print(f"[warmup] pre-encoding prompt on remote container...")
    t0 = time.perf_counter()
    await _segmenter.warmup.remote.aio(prompt)
    print(f"[warmup] done in {time.perf_counter()-t0:.2f}s")


#***** 3 downsample segment mask *****

def downsample(mask: Image.Image, out_h: int, out_w: int) -> Image.Image:
    arr = np.array(mask, dtype=np.float32) / 255.0
    H, W = arr.shape
    ph, pw = (out_h - H % out_h) % out_h, (out_w - W % out_w) % out_w
    arr = np.pad(arr, ((0, ph), (0, pw)), constant_values=np.nan)
    H2, W2 = arr.shape
    downsampled_mask = np.nanmean(arr.reshape(out_h, H2 // out_h, out_w, W2 // out_w), axis=(1, 3))
    return Image.fromarray((downsampled_mask * 255).astype(np.uint8), mode="L")

#***** 4 center of mass *****

def center_of_mass(mask: Image.Image) -> tuple[float, float]:
    mask = np.array(mask, dtype=np.float32) / 255.0
    H, W = mask.shape
    rows = np.arange(H)
    cols = np.arange(W)
    total = mask.sum()
    row = (mask * rows[:, None]).sum() / total
    col = (mask * cols[None, :]).sum() / total
    return row.item(), col.item()


#***** 5 overhead image *****

def make_overhead(overhead: Image.Image, segment_mask: Image.Image, com: tuple[float, float], is_empty_box: bool, speaker: str | None = None) -> Image.Image:
    SPEAKER_IMG = "/home/ethantu/workspace/good-vibrations/data/speaker.png"
    SPEAKER_POSITION = {'1000': (0, 0.5), '0100': (1/3, 0), '0010': (2/3, 0), '0001': (1, 0.5)}

    overhead = overhead.convert("RGBA")

    mask_arr = np.array(segment_mask)
    color = np.zeros((*mask_arr.shape, 4), dtype=np.uint8)
    color[..., 0] = 0
    color[..., 1] = 204
    color[..., 2] = 51
    color[..., 3] = (mask_arr * 0.35).astype(np.uint8)
    overlay = Image.fromarray(color, mode="RGBA").resize(overhead.size, resample=Image.NEAREST)
    overhead = Image.alpha_composite(overhead, overlay)
    W, H = overhead.size

    if not is_empty_box:
        draw = ImageDraw.Draw(overhead)
        cx, cy = int(com[1]), int(com[0])
        r = max(10, W // 60)
        draw.line([(cx - r, cy), (cx + r, cy)], fill=(144, 238, 144, 255), width=3)
        draw.line([(cx, cy - r), (cx, cy + r)], fill=(144, 238, 144, 255), width=3)

    if speaker is None: return overhead.convert("RGB")

    pad = max(W, H) // 5
    canvas_w, canvas_h = W + 2 * pad, H + 2 * pad
    canvas = Image.new("RGBA", (canvas_w, canvas_h), (220, 220, 220, 255))
    canvas.paste(overhead, (pad, pad))

    spk_h = H // 3
    speaker_img = Image.open(SPEAKER_IMG).convert("RGBA")
    spk_w = int(speaker_img.width * spk_h / speaker_img.height)
    spk = speaker_img.resize((spk_w, spk_h), resample=Image.LANCZOS)

    x_frac, y_frac = SPEAKER_POSITION[speaker]
    spk_cx = pad + int(x_frac * W)
    spk_cy = pad + int((1 - y_frac) * H)
    canvas.paste(spk, (spk_cx - spk_w // 2, spk_cy - spk_h // 2), mask=spk)

    return canvas.convert("RGB")

#***** 6 put it all together *****

DEFAULT_PROMPT = "A black metal cube sitting on the floor of an open cardboard box from a bird's eye view."
SHARED_ARTIFACTS = ["00_raw_overhead.png", "01_cropped_overhead.png", "02_segment_mask.png", "03_downsampled_segment_mask.png", "04_com.jsonl", "y.npy"]

async def process_outputs(speaker: str, raw_overhead_path: Path, output_dir: Path, sample_dir: Path, left: float = 0.15, right: float = 0.67, up: float = 0.08, down: float = 0.7,
                          prompt: str = DEFAULT_PROMPT, out_h: int = 40, out_w: int = 20, verbose: int = 1, overwrite: bool = True, is_empty_box: bool | None = None) -> Image.Image:
    sample_id, output_id = sample_dir.name, output_dir.name
    t_start = time.perf_counter()
    if verbose >= 1: print(f"[{sample_id}] Process Output {output_id}")

    if is_empty_box is None: is_empty_box = 'empty' in prompt
    if verbose >= 1: print(f"[{sample_id}] Box is {'not '*int(not is_empty_box)}empty")

    status_path = output_dir / "status.jsonl"
    prior = next((line for line in load(status_path) if line.get("sample_id") == sample_id), None) if status_path.exists() else None
    already_processed = (sample_dir.exists() and any(sample_dir.iterdir())) or prior is not None
    if already_processed:
        if not overwrite:
            raise ValueError(f"[{sample_id}] already processed — pass overwrite=True to redo")
        if verbose >= 1: print(f"[{sample_id}] overwriting previous run")
        if sample_dir.exists(): shutil.rmtree(sample_dir)
        if prior is not None:
            lines = [l for l in load(status_path) if l.get("sample_id") != sample_id]
            status_path.write_text("".join(json.dumps(l) + "\n" for l in lines))

    if False:
        pass
    else:
        t0 = time.perf_counter()
        cropped_overhead = resize_for_segmentation(crop(load(raw_overhead_path), left, right, up, down))
        save(cropped_overhead, output_dir / "01_cropped_overhead.png")
        if verbose >= 1: print(f"[{sample_id}] crop+resize {cropped_overhead.size}: {time.perf_counter()-t0:.3f}s")

        t0 = time.perf_counter()
        segment_mask = await segment(cropped_overhead, prompt, is_empty_box, print_timing=(verbose >= 1))
        save(segment_mask, output_dir / "02_segment_mask.png")
        if verbose >= 1: print(f"[{sample_id}] segment TOTAL: {time.perf_counter()-t0:.3f}s")

        t0 = time.perf_counter()
        downsampled_segment_mask = downsample(segment_mask, out_h, out_w)
        save(downsampled_segment_mask, output_dir / "03_downsampled_segment_mask.png")
        save(np.array(downsampled_segment_mask, dtype=np.float32) / 255.0, output_dir / "y.npy")
        if verbose >= 1: print(f"[{sample_id}] downsample: {time.perf_counter()-t0:.3f}s")

        t0 = time.perf_counter()
        com = (-1, -1) if is_empty_box else center_of_mass(segment_mask)
        downsampled_com = (-1, -1) if is_empty_box else center_of_mass(downsampled_segment_mask)
        save({"com": com, "downsampled_com": downsampled_com}, output_dir / "04_com.jsonl")
        if verbose >= 1: print(f"[{sample_id}] center of mass: {time.perf_counter()-t0:.3f}s")

    for artifact in SHARED_ARTIFACTS: symlink(output_dir / artifact, sample_dir / f"outputs/{artifact}")
    symlink(output_dir / "y.npy", sample_dir / "y.npy")

    t0 = time.perf_counter()
    overhead = make_overhead(cropped_overhead, segment_mask, com, is_empty_box, speaker)
    save(overhead, sample_dir / "outputs/05_overhead.png")
    symlink(sample_dir / "outputs/05_overhead.png", sample_dir / "overhead.png")
    if verbose >= 1: print(f"[{sample_id}] make overhead: {time.perf_counter()-t0:.3f}s")

    time_now = datetime.now(timezone.utc).isoformat()
    append({"sample_id": sample_id, "sample_dir": str(sample_dir), "time": time_now}, output_dir / "samples.jsonl")
    append({"processed_outputs_time": time_now}, sample_dir / "time.jsonl")
    append([{"output_id": output_id}, {"output_dir": str(output_dir)}], sample_dir / "data.jsonl")

    if verbose >= 1: print(f"[{sample_id}] ===== TOTAL: {time.perf_counter()-t_start:.3f}s =====")
    return overhead
