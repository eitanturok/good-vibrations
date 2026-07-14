"""Experiment: make segment() faster on a single fixed image with multiple object prompts.

Baseline (src/data/segment.py): one modal.spawn() per object -> N separate
containers/calls, each re-running the (expensive) SAM3 vision encoder from
scratch just to answer a different text prompt against the *same* image.

Key idea tried here: the vision encoder (ViT backbone) is by far the most
expensive part of a SAM3 forward pass, and it doesn't depend on the text
prompt at all. Sam3Model exposes `get_vision_features()` / `forward(vision_embeds=...)`
specifically so callers can encode the image once and reuse it across
multiple text-conditioned decodes. So: encode the image once, then run ONE
batched forward pass (batch dim = number of prompts) that reuses those
cached vision embeddings. Also restores torch.autocast(bfloat16), which the
current segment.py dropped relative to the old sam3-image-model code path.

Usage: `modal run src/data/segment2.py`
"""
import math
import time
from pathlib import Path

import modal
import numpy as np
from PIL import Image

app = modal.App("segment2")

modal_image = (
    modal.Image.from_registry("nvidia/cuda:12.8.0-devel-ubuntu22.04", add_python="3.12")
    .entrypoint([])
    .apt_install("git")
    .pip_install("transformers", "Pillow", "torch", "torchvision", "accelerate")
)

IMAGE_PATH = r"D:\eturok\experiment-24\images\000012\01_cropped.png"
OBJECTS = {"cylinder": 2, "cube": 1}
PROMPTS = {"cylinder": "Metal circle", "cube": "Black cube"}
N_REPEATS = 3  # timed repeats after a warmup call, to average out jitter (kept small for fast iteration)

DEFAULT_IMAGE_SIZE = 1008  # the original segment.py default (patch_size=14, window_size=24 -> 72x72 patches, 72/24=3)
GRID_UNIT = 24 * 14  # =336: image_size must be a multiple of this so the patch grid divides evenly into windows


def resolve_image_size(scale: float, base: int = DEFAULT_IMAGE_SIZE) -> int:
    """Same `scale` knob src/data/segment.py exposed (1.0 = default/full resolution, smaller
    = less compute) -- but here it actually changes the vision-encoder's patch grid (see
    SegmenterSmallImage's docstring), not just a pre-resize of the input that gets undone.
    Snapped to the nearest multiple of GRID_UNIT (ties broken toward the smaller/faster size)
    since windowed attention requires an evenly-divisible patch grid."""
    target = base * scale
    lo = max(1, math.floor(target / GRID_UNIT)) * GRID_UNIT
    hi = lo + GRID_UNIT
    return lo if abs(target - lo) <= abs(target - hi) else hi


@app.cls(
    gpu="A10G",  # same GPU as src/data/segment.py -- comparing on equal hardware
    image=modal_image,
    secrets=[modal.Secret.from_name("huggingface")],
    scaledown_window=60 * 10,
)
class Segmenter2:
    @modal.enter()
    def load(self):
        import torch
        from transformers import Sam3Model, Sam3Processor

        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = Sam3Processor.from_pretrained("facebook/sam3")
        self.model = Sam3Model.from_pretrained("facebook/sam3").to(self.device).eval()
        self.model_bf16 = Sam3Model.from_pretrained("facebook/sam3").to(self.device, dtype=torch.bfloat16).eval()

    def _postprocess(self, outputs, original_sizes, n):
        result = self.processor.post_process_instance_segmentation(
            outputs, threshold=0.0, mask_threshold=0.5, target_sizes=[original_sizes] * n,
        )
        return [{k: (v.float().cpu().numpy() if self.torch.is_tensor(v) else v) for k, v in r.items()} for r in result]

    @modal.method()
    def run_sequential(self, image: Image.Image, prompts: list[str]) -> tuple[list[dict], float]:
        """Baseline: one full forward pass (vision + text + decode) per prompt."""
        torch = self.torch
        image = image.convert("RGB")
        t0 = time.perf_counter()
        results = []
        for prompt in prompts:
            inputs = self.processor(images=image, text=prompt, return_tensors="pt").to(self.device)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.device == "cuda"):
                outputs = self.model(**inputs)
            results += self._postprocess(outputs, inputs["original_sizes"].tolist()[0], 1)
        torch.cuda.synchronize()
        return results, time.perf_counter() - t0

    @modal.method()
    def run_shared_vision(self, image: Image.Image, prompts: list[str]) -> tuple[list[dict], float]:
        """Encode the image once, batch all prompts through one decode pass."""
        torch = self.torch
        image = image.convert("RGB")
        n = len(prompts)
        t0 = time.perf_counter()

        img_inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        text_inputs = self.processor(text=prompts, return_tensors="pt").to(self.device)

        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.device == "cuda"):
            vision_embeds = self.model.get_vision_features(pixel_values=img_inputs["pixel_values"])
            # broadcast the single encoded image across the prompt batch dim (no extra ViT work)
            vision_embeds.fpn_hidden_states = tuple(t.expand(n, *t.shape[1:]) for t in vision_embeds.fpn_hidden_states)
            vision_embeds.fpn_position_encoding = tuple(t.expand(n, *t.shape[1:]) for t in vision_embeds.fpn_position_encoding)
            outputs = self.model(
                vision_embeds=vision_embeds,
                input_ids=text_inputs["input_ids"],
                attention_mask=text_inputs.get("attention_mask"),
            )

        results = self._postprocess(outputs, img_inputs["original_sizes"].tolist()[0], n)
        torch.cuda.synchronize()
        return results, time.perf_counter() - t0

    def _shared_vision_forward(self, model, image: Image.Image, prompts: list[str], dtype=None, contiguous=False):
        """Shared logic for the bf16-weights and compiled variants: same shared-vision-embeds
        trick as run_shared_vision, but running against a differently-prepared `model` (either
        bf16 weights with no autocast, or a torch.compile()-wrapped model)."""
        torch = self.torch
        image = image.convert("RGB")
        n = len(prompts)
        t0 = time.perf_counter()

        img_inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        text_inputs = self.processor(text=prompts, return_tensors="pt").to(self.device)
        pixel_values = img_inputs["pixel_values"]
        if dtype is not None: pixel_values = pixel_values.to(dtype)

        # autocast as a safety net: some buffers (e.g. sinusoidal position encodings) aren't
        # moved by model.to(dtype=bf16) since they're plain tensors, not registered buffers/params,
        # so a stray float32 op can otherwise crash a linear layer expecting bf16 weights.
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.device == "cuda"):
            vision_embeds = model.get_vision_features(pixel_values=pixel_values)
            expand = (lambda t: t.expand(n, *t.shape[1:]).contiguous()) if contiguous else (lambda t: t.expand(n, *t.shape[1:]))
            vision_embeds.fpn_hidden_states = tuple(expand(t) for t in vision_embeds.fpn_hidden_states)
            vision_embeds.fpn_position_encoding = tuple(expand(t) for t in vision_embeds.fpn_position_encoding)
            outputs = model(
                vision_embeds=vision_embeds,
                input_ids=text_inputs["input_ids"],
                attention_mask=text_inputs.get("attention_mask"),
            )

        results = self._postprocess(outputs, img_inputs["original_sizes"].tolist()[0], n)
        torch.cuda.synchronize()
        return results, time.perf_counter() - t0

    @modal.method()
    def run_shared_vision_bf16weights(self, image: Image.Image, prompts: list[str]) -> tuple[list[dict], float]:
        """Same as run_shared_vision, but weights are natively bf16 (no autocast casting overhead)."""
        return self._shared_vision_forward(self.model_bf16, image, prompts, dtype=self.torch.bfloat16)

    @modal.method()
    def run_shared_vision_compiled(self, image: Image.Image, prompts: list[str]) -> tuple[list[dict], float]:
        """Same as run_shared_vision_bf16weights, but with the bf16 model wrapped in torch.compile.
        Compiled lazily on first call (compiled once per warm container, cached on self)."""
        torch = self.torch
        if not hasattr(self, "model_compiled"):
            self.model_compiled = torch.compile(self.model_bf16, dynamic=False)
        return self._shared_vision_forward(self.model_compiled, image, prompts, dtype=torch.bfloat16)

    @modal.method()
    def run_shared_vision_cudagraphs(self, image: Image.Image, prompts: list[str]) -> tuple[list[dict], float]:
        """Same as run_shared_vision_compiled, but mode='reduce-overhead' (CUDA graphs). Requires
        the expanded (stride-0) vision embeds to be made contiguous first -- CUDA graph capture
        needs stable, real memory for its captured buffers."""
        torch = self.torch
        if not hasattr(self, "model_cudagraphs"):
            self.model_cudagraphs = torch.compile(self.model_bf16, dynamic=False, mode="reduce-overhead")
        return self._shared_vision_forward(self.model_cudagraphs, image, prompts, dtype=torch.bfloat16, contiguous=True)


@app.cls(
    gpu="A10G",  # same GPU as src/data/segment.py
    image=modal_image,
    secrets=[modal.Secret.from_name("huggingface")],
    scaledown_window=60 * 10,
)
class SegmenterCombined:
    """The winning combo from the experiments above, alone in its own container so
    nothing else on the GPU (extra model copies, other compiled graphs) can interfere
    with the timing: shared vision encoding (idea 1) + native bf16 weights (idea 2) +
    torch.compile (idea 3). CUDA graphs (idea 4) deliberately left out -- rejected,
    made everything slower under contention."""

    @modal.enter()
    def load(self):
        import torch
        from transformers import Sam3Model, Sam3Processor

        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = Sam3Processor.from_pretrained("facebook/sam3")
        model = Sam3Model.from_pretrained("facebook/sam3").to(self.device, dtype=torch.bfloat16).eval()
        self.model = torch.compile(model, dynamic=False)

    def _postprocess(self, outputs, original_sizes, n):
        result = self.processor.post_process_instance_segmentation(
            outputs, threshold=0.0, mask_threshold=0.5, target_sizes=[original_sizes] * n,
        )
        return [{k: (v.float().cpu().numpy() if self.torch.is_tensor(v) else v) for k, v in r.items()} for r in result]

    @modal.method()
    def run(self, image: Image.Image, prompts: list[str]) -> tuple[list[dict], float]:
        torch = self.torch
        image = image.convert("RGB")
        n = len(prompts)
        t0 = time.perf_counter()

        img_inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        text_inputs = self.processor(text=prompts, return_tensors="pt").to(self.device)
        pixel_values = img_inputs["pixel_values"].to(torch.bfloat16)

        # autocast is a safety net for any stray fp32 buffer (e.g. positional encodings built
        # fresh per call) that model.to(dtype=bf16) didn't move -- see "Key mistake" in
        # segment2_results.md.
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.device == "cuda"):
            vision_embeds = self.model.get_vision_features(pixel_values=pixel_values)
            # encode the image once, broadcast across the prompt batch dim -- no 2nd ViT pass
            vision_embeds.fpn_hidden_states = tuple(t.expand(n, *t.shape[1:]) for t in vision_embeds.fpn_hidden_states)
            vision_embeds.fpn_position_encoding = tuple(t.expand(n, *t.shape[1:]) for t in vision_embeds.fpn_position_encoding)
            outputs = self.model(
                vision_embeds=vision_embeds,
                input_ids=text_inputs["input_ids"],
                attention_mask=text_inputs.get("attention_mask"),
            )

        results = self._postprocess(outputs, img_inputs["original_sizes"].tolist()[0], n)
        torch.cuda.synchronize()
        return results, time.perf_counter() - t0


def _summarize(results, prompts, objects):
    """Print top-`count` scores per prompt so we can sanity-check detections match `objects`."""
    for prompt, (name, count) in zip(prompts, objects.items()):
        r = results[prompts.index(prompt)]
        order = np.argsort(r["scores"])[::-1][:count]
        scores = [f"{r['scores'][i]:.3f}" for i in order]
        print(f"    {name!r} ({prompt!r}): top-{count} scores = {scores}")


VARIANTS = ["run_sequential", "run_shared_vision", "run_shared_vision_bf16weights", "run_shared_vision_compiled", "run_shared_vision_cudagraphs"]


@app.local_entrypoint()
def main():
    image = Image.open(IMAGE_PATH)
    prompts = list(PROMPTS.values())
    print(f"image: {IMAGE_PATH} size={image.size}, prompts={prompts}")

    seg = Segmenter2()

    # warmup: absorb cold start + first-call overhead (cudnn autotune, torch.compile trace) before timing
    print("\nwarming up container (this also triggers torch.compile tracing, once)...")
    for name in VARIANTS: getattr(seg, name).remote(image, prompts)

    print(f"\ntiming {N_REPEATS} repeats each (server-side wall time, warm container)...")
    times = {name: [] for name in VARIANTS}
    last_results = {}
    for i in range(N_REPEATS):
        row = []
        for name in VARIANTS:
            results, t = getattr(seg, name).remote(image, prompts)
            times[name].append(t); last_results[name] = results
            row.append(f"{name}={t*1000:6.1f} ms")
        print(f"  [{i}] " + "   ".join(row))

    means = {name: float(np.mean(ts)) for name, ts in times.items()}
    baseline = means["run_sequential"]
    print("\nmean times (baseline = run_sequential):")
    for name in VARIANTS:
        print(f"  {name:32s} {means[name]*1000:7.1f} ms   speedup {baseline/means[name]:.2f}x")

    print("\nsanity check (top-k scores per object, should roughly agree across variants):")
    for name in VARIANTS:
        print(f"  {name}:")
        _summarize(last_results[name], prompts, OBJECTS)


@app.local_entrypoint()
def combined():
    """Clean, isolated benchmark of the combined optimization (its own container, its own
    app.cls -- no other model copies competing for GPU memory/scheduling)."""
    image = Image.open(IMAGE_PATH)
    prompts = list(PROMPTS.values())
    print(f"image: {IMAGE_PATH} size={image.size}, prompts={prompts}")

    seg = SegmenterCombined()

    print("\nwarming up (loads weights, triggers torch.compile trace, lets GPU/allocator reach steady state)...")
    for _ in range(4): seg.run.remote(image, prompts)

    print(f"\ntiming {N_REPEATS} repeats (server-side wall time, warm container)...")
    times, results = [], None
    for i in range(N_REPEATS):
        results, t = seg.run.remote(image, prompts)
        times.append(t)
        print(f"  [{i}] {t*1000:6.1f} ms")

    mean_t = float(np.mean(times))
    baseline_ms = 589.5  # run_sequential, measured clean/uncontended -- see segment2_results.md
    print(f"\nmean combined time (size=1008, default): {mean_t*1000:.1f} ms")
    print(f"speedup vs clean baseline ({baseline_ms:.1f} ms): {baseline_ms / (mean_t*1000):.2f}x")

    print("\nsanity check:")
    _summarize(results, prompts, OBJECTS)


@app.cls(
    gpu="A10G",
    image=modal_image,
    secrets=[modal.Secret.from_name("huggingface")],
    scaledown_window=60 * 10,
)
class SegmenterSmallImage:
    """Same combo as SegmenterCombined, but built for a smaller vision-encoder grid.

    Naively passing a smaller `size` to the *processor* while keeping the default model
    breaks: Sam3ViTRotaryEmbedding precomputes its cos/sin tables once at model-construction
    time from `config.image_size` (default 1008, patch_size=14 -> 72x72=5184 patches), not
    dynamically per forward call. Feed it a different patch grid at inference and the RoPE
    tensors don't broadcast against query/key -- shape mismatch crash (confirmed by testing).

    Fix: override `Sam3VisionConfig.image_size` *before* constructing the model, then load
    the pretrained weights into that config. RoPE tables are recomputed fresh for the new
    grid at init (they're plain trig buffers, not learned, so this is safe) -- only the
    learned absolute position embeddings need interpolation, which the model already does
    internally (`_tile_position_embeddings`). IMAGE_SIZE must stay a multiple of both
    patch_size (14) and window_size*patch_size (24*14=336) so the patch grid divides evenly
    for windowed attention, same constraint the default 1008 (=72*14, 72/24=3) satisfies.
    """

    # must stay a multiple of patch_size (14) and window_size*patch_size (24*14=336)
    image_size: int = modal.parameter(default=672)

    @modal.enter()
    def load(self):
        import torch
        from transformers import Sam3Config, Sam3Model, Sam3Processor

        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = Sam3Processor.from_pretrained(
            "facebook/sam3", size={"height": self.image_size, "width": self.image_size}
        )
        config = Sam3Config.from_pretrained("facebook/sam3")
        config.vision_config.image_size = self.image_size
        model = Sam3Model.from_pretrained("facebook/sam3", config=config).to(self.device, dtype=torch.bfloat16).eval()
        self.model = torch.compile(model, dynamic=False)

    def _postprocess(self, outputs, original_sizes, n):
        result = self.processor.post_process_instance_segmentation(
            outputs, threshold=0.0, mask_threshold=0.5, target_sizes=[original_sizes] * n,
        )
        return [{k: (v.float().cpu().numpy() if self.torch.is_tensor(v) else v) for k, v in r.items()} for r in result]

    @modal.method()
    def run(self, image: Image.Image, prompts: list[str]) -> tuple[list[dict], float]:
        torch = self.torch
        image = image.convert("RGB")
        n = len(prompts)
        t0 = time.perf_counter()

        img_inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        text_inputs = self.processor(text=prompts, return_tensors="pt").to(self.device)
        pixel_values = img_inputs["pixel_values"].to(torch.bfloat16)

        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.device == "cuda"):
            vision_embeds = self.model.get_vision_features(pixel_values=pixel_values)
            vision_embeds.fpn_hidden_states = tuple(t.expand(n, *t.shape[1:]) for t in vision_embeds.fpn_hidden_states)
            vision_embeds.fpn_position_encoding = tuple(t.expand(n, *t.shape[1:]) for t in vision_embeds.fpn_position_encoding)
            outputs = self.model(
                vision_embeds=vision_embeds,
                input_ids=text_inputs["input_ids"],
                attention_mask=text_inputs.get("attention_mask"),
            )

        results = self._postprocess(outputs, img_inputs["original_sizes"].tolist()[0], n)
        torch.cuda.synchronize()
        return results, time.perf_counter() - t0


def _organize_for_viz(results: list[dict], prompts: list[str], objects: dict[str, int]) -> dict:
    """Flatten per-prompt candidate detections down to the top-`count` per object
    (same top-k selection segment()/_organize() in src/data/image.py do), so plot_smask
    can draw one box+label per real object instead of every raw candidate."""
    names, scores, boxes, masks = [], [], [], []
    for prompt, (name, count) in zip(prompts, objects.items()):
        r = results[prompts.index(prompt)]
        order = np.argsort(r["scores"])[::-1][:count]
        for i, idx in enumerate(order):
            names.append(f"{name}{i + 1}")
            scores.append(float(r["scores"][idx]))
            boxes.append(np.asarray(r["boxes"][idx]))
            masks.append(np.asarray(r["masks"][idx]) > 0)
    return {"names": names, "scores": np.array(scores), "boxes": np.stack(boxes), "masks": np.stack(masks)}


@app.local_entrypoint()
def small_image(scale: float = 2 / 3):
    """Isolated benchmark of SegmenterSmallImage at a given `scale` of the original 1008
    default (same knob src/data/segment.py's Segmenter.run(scale=...) exposed, e.g. 1.0 =
    1008 [default], 2/3 = 672 [current best], 1/3 = 336 [half of 672]).
    Usage: modal run src/data/segment2.py::small_image --scale 0.333"""
    from src.data.segment import label_map_image, plot_smask

    image_size = resolve_image_size(scale)
    image = Image.open(IMAGE_PATH)
    prompts = list(PROMPTS.values())
    print(f"image: {IMAGE_PATH} size={image.size}, prompts={prompts}, scale={scale} -> image_size={image_size}")

    seg = SegmenterSmallImage(image_size=image_size)

    print("\nwarming up (loads weights w/ overridden vision_config.image_size, triggers torch.compile trace)...")
    for _ in range(4): seg.run.remote(image, prompts)

    print(f"\ntiming {N_REPEATS} repeats (server-side wall time, warm container)...")
    times, results = [], None
    for i in range(N_REPEATS):
        results, t = seg.run.remote(image, prompts)
        times.append(t)
        print(f"  [{i}] {t*1000:6.1f} ms")

    mean_t = float(np.mean(times))
    baseline_ms = 589.5  # run_sequential, measured clean/uncontended -- see segment2_results.md
    print(f"\nmean time (image_size={image_size}): {mean_t*1000:.1f} ms")
    print(f"speedup vs clean baseline ({baseline_ms:.1f} ms): {baseline_ms / (mean_t*1000):.2f}x")

    print("\nsanity check (does it still find both objects at lower resolution?):")
    _summarize(results, prompts, OBJECTS)

    # save the masks so they can be visually inspected -- not just trusted from scores
    out_dir = Path(__file__).parent / "segment2_outputs"
    out_dir.mkdir(exist_ok=True)
    organized = _organize_for_viz(results, prompts, OBJECTS)
    composite = plot_smask(organized, image.convert("RGB"), organized["names"], show=False)
    composite.save(out_dir / f"masks_size{image_size}.png")
    label_map_image(organized["masks"]).save(out_dir / f"labelmap_size{image_size}.png")
    print(f"\nsaved: {out_dir / f'masks_size{image_size}.png'}")
    print(f"saved: {out_dir / f'labelmap_size{image_size}.png'}")


# ***** benchmark of the commented-out native sam3-package Segmenter in segment.py -- run for
# real instead of guessing whether it's actually faster. Same GPU (A10G), image NOT baked with
# weights (matching how that code was originally written, so this is a fair "as it stood"
# comparison, not an apples-to-oranges one against our baked-image production fix). *****

modal_image_native = (
    modal.Image.from_registry("nvidia/cuda:12.8.0-devel-ubuntu22.04", add_python="3.12")
    .entrypoint([])
    .apt_install("git")
    .pip_install("git+https://github.com/facebookresearch/sam3#egg=sam3[notebooks]", "Pillow", "torch", "torchvision")
)


@app.cls(
    gpu="A10G",
    image=modal_image_native,
    secrets=[modal.Secret.from_name("huggingface")],
    scaledown_window=60 * 10,
)
class SegmenterNative:
    @modal.enter()
    def load(self):
        from sam3.model_builder import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor

        self.processor = Sam3Processor(build_sam3_image_model())
        self.processor.set_confidence_threshold(0.0)

    @modal.method()
    def run(self, image_array: np.ndarray, prompt: str) -> tuple[np.ndarray, float]:
        import torch

        t0 = time.perf_counter()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            state = self.processor.set_image(Image.fromarray(image_array))
            out = self.processor.set_text_prompt(state=state, prompt=prompt)
        if len(out["scores"]) > 1:
            idx = out["scores"].topk(1).indices
            out["masks"] = out["masks"][idx]
        mask = out["masks"][:, 0].any(dim=0)
        torch.cuda.synchronize()
        return mask.cpu().numpy(), time.perf_counter() - t0


@app.local_entrypoint()
def native():
    """Benchmark the exact code in segment.py's commented-out old Segmenter class:
    single top-1 mask per prompt, native sam3 package, autocast bf16, no baked weights.
    First call per prompt is genuinely cold (no warmup) to match what production actually
    pays; then a few warm repeats for comparison against segment2.py's other numbers."""
    image = Image.open(IMAGE_PATH).convert("RGB")
    image_array = np.array(image)
    prompts = list(PROMPTS.values())
    print(f"image: {IMAGE_PATH} size={image.size}, prompts={prompts}")

    seg = SegmenterNative()

    print("\ncold call (first ever on this container, no warmup) per prompt:")
    cold_times = []
    for prompt in prompts:
        mask, t = seg.run.remote(image_array, prompt)
        cold_times.append(t)
        print(f"  {prompt!r}: {t*1000:.1f} ms (mask pixels set: {int(mask.sum())})")
    print(f"  cold total (both objects, sequential): {sum(cold_times)*1000:.1f} ms")

    print(f"\nwarm repeats ({N_REPEATS}x, both objects per repeat):")
    warm_times = []
    for i in range(N_REPEATS):
        row = [seg.run.remote(image_array, prompt)[1] for prompt in prompts]
        warm_times.append(sum(row))
        print(f"  [{i}] {sum(row)*1000:.1f} ms")

    print(f"\nmean warm total (both objects): {np.mean(warm_times)*1000:.1f} ms")


@app.cls(
    gpu="A10G",
    image=modal_image_native,
    secrets=[modal.Secret.from_name("huggingface")],
    scaledown_window=60 * 10,
)
class SegmenterNativeOptimized:
    """Every optimization from the transformers path, applied to the native sam3 package:
    - shared vision encoding: set_image() ONCE, reuse the same `state` dict (which caches
      backbone_out, i.e. the encoded image) across both set_text_prompt() calls instead of
      re-encoding per object -- this package already supports that pattern natively, we just
      weren't using it in the un-optimized SegmenterNative benchmark above.
    - torch.compile: build_sam3_image_model(compile=True) is a first-class supported flag here
      (applies torch.compile to the vision encoder + segmentation head internally).
    - bf16 weights, on top of the autocast the original commented-out code already had.
    """

    @modal.enter()
    def load(self):
        import torch
        from sam3.model_builder import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor

        self.torch = torch
        model = build_sam3_image_model(compile=True)
        model = model.to(dtype=torch.bfloat16)
        self.processor = Sam3Processor(model)
        self.processor.set_confidence_threshold(0.0)

    @modal.method()
    def run(self, image_array: np.ndarray, prompts: list[str]) -> tuple[list[np.ndarray], float]:
        torch = self.torch
        t0 = time.perf_counter()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            state = self.processor.set_image(Image.fromarray(image_array))  # encode ONCE
            masks = []
            for prompt in prompts:
                out = self.processor.set_text_prompt(state=state, prompt=prompt)  # reuses cached image encoding
                if len(out["scores"]) > 1:
                    idx = out["scores"].topk(1).indices
                    out["masks"] = out["masks"][idx]
                masks.append(out["masks"][:, 0].any(dim=0).cpu().numpy())
        torch.cuda.synchronize()
        return masks, time.perf_counter() - t0


@app.local_entrypoint()
def native_optimized():
    """Isolated benchmark of SegmenterNativeOptimized against the un-optimized 409.5ms
    SegmenterNative baseline measured above."""
    image = Image.open(IMAGE_PATH).convert("RGB")
    image_array = np.array(image)
    prompts = list(PROMPTS.values())
    print(f"image: {IMAGE_PATH} size={image.size}, prompts={prompts}")

    seg = SegmenterNativeOptimized()

    print("\nwarming up (loads weights, triggers torch.compile trace, lets GPU/allocator settle)...")
    for _ in range(4): seg.run.remote(image_array, prompts)

    print(f"\ntiming {N_REPEATS} repeats (both objects per repeat, shared image encoding)...")
    times = []
    for i in range(N_REPEATS):
        masks, t = seg.run.remote(image_array, prompts)
        times.append(t)
        print(f"  [{i}] {t*1000:.1f} ms (mask pixel counts: {[int(m.sum()) for m in masks]})")

    mean_t = float(np.mean(times))
    baseline_native_ms = 409.5  # SegmenterNative, unoptimized, warm -- see native() entrypoint
    print(f"\nmean time: {mean_t*1000:.1f} ms")
    print(f"speedup vs unoptimized native ({baseline_native_ms:.1f} ms): {baseline_native_ms/(mean_t*1000):.2f}x")


# ***** cold-start experiment: Modal memory snapshots. Instrumented Segmenter.load() in
# production segment.py found the ~15.9s @modal.enter() cold-start cost is >12s of pure Python
# import (torch+transformers) and CPU-side from_pretrained() -- NOT the GPU transfer (~1.1s).
# Modal's enable_memory_snapshot checkpoints a container's memory right after a designated
# @modal.enter(snap=True) step, so future cold starts restore from that snapshot instead of
# re-running the (slow) imports/from_pretrained. The GPU transfer must happen fresh on every
# restore (a plain @modal.enter(), snap=False by default), since GPU state isn't captured by a
# CPU-only memory snapshot. This is a separate axis from the segment2_progress chart (which is
# warm-time only, by design) -- track it in segment_coldstart_attempts.json instead. *****

def _download_sam3_weights_for_snapshot():
    from transformers import Sam3Model, Sam3Processor

    Sam3Processor.from_pretrained("facebook/sam3")
    Sam3Model.from_pretrained("facebook/sam3")


modal_image_baked = (
    modal.Image.from_registry("nvidia/cuda:12.8.0-devel-ubuntu22.04", add_python="3.12")
    .entrypoint([])
    .apt_install("git")
    .pip_install("transformers", "Pillow", "torch", "torchvision", "accelerate")
    .run_function(_download_sam3_weights_for_snapshot, secrets=[modal.Secret.from_name("huggingface")])
)


@app.cls(
    gpu="A10G",
    image=modal_image_baked,
    secrets=[modal.Secret.from_name("huggingface")],
    scaledown_window=60 * 10,
    # NOTE: enable_memory_snapshot=True + experimental_options={"enable_gpu_snapshot": True} were
    # here originally -- removed because Modal rejects them outright on this workspace ("Memory
    # snapshots are not supported by the workspace's runtime", likely an NVIDIA driver
    # branch/570-575 mismatch on whichever underlying node pool this workspace's A10Gs come
    # from -- see segment2_results.md). Left the snap=True/snap=False split below for reference;
    # without the flags above it's just a normal two-stage @modal.enter(), no snapshotting.
)
class SegmenterSnapshot:
    @modal.enter()  # was snap=True; now a plain two-stage enter() since snapshotting is disabled
    def load_cpu(self):
        """Everything that doesn't touch the GPU -- captured in the memory snapshot."""
        import time
        t0 = time.perf_counter()

        import torch
        print(f"[timing] import torch: {time.perf_counter()-t0:.3f}s", flush=True); t1 = time.perf_counter()

        from transformers import Sam3Model, Sam3Processor
        print(f"[timing] import transformers: {time.perf_counter()-t1:.3f}s", flush=True); t1 = time.perf_counter()

        self.processor = Sam3Processor.from_pretrained("facebook/sam3")
        print(f"[timing] Sam3Processor.from_pretrained: {time.perf_counter()-t1:.3f}s", flush=True); t1 = time.perf_counter()

        self._model_cpu = Sam3Model.from_pretrained("facebook/sam3")
        print(f"[timing] Sam3Model.from_pretrained: {time.perf_counter()-t1:.3f}s", flush=True); t1 = time.perf_counter()

        self._torch = torch
        print(f"[timing] TOTAL snap=True load_cpu(): {time.perf_counter()-t0:.3f}s", flush=True)

    @modal.enter()
    def load_gpu(self):
        """GPU transfer -- must re-run after every restore (snapshot is CPU-memory only)."""
        import time
        t0 = time.perf_counter()

        torch = self._torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self._model_cpu.to(self.device)
        if self.device == "cuda": torch.cuda.synchronize()
        print(f"[timing] TOTAL snap=False load_gpu() (.to({self.device!r})): {time.perf_counter()-t0:.3f}s", flush=True)

    @modal.method()
    def run(self, image: Image.Image, prompt: str) -> dict:
        import time
        import torch

        t0 = time.perf_counter()
        inputs = self.processor(images=image.convert("RGB"), text=prompt, return_tensors="pt").to(self.device)
        if self.device == "cuda": torch.cuda.synchronize()
        print(f"[timing] processor(...) + .to(device): {time.perf_counter()-t0:.3f}s", flush=True); t1 = time.perf_counter()

        with torch.no_grad(): outputs = self.model(**inputs)
        if self.device == "cuda": torch.cuda.synchronize()
        print(f"[timing] model forward pass: {time.perf_counter()-t1:.3f}s", flush=True)

        result = self.processor.post_process_instance_segmentation(
            outputs, threshold=0.0, mask_threshold=0.5, target_sizes=inputs["original_sizes"].tolist()
        )[0]
        print(f"[timing] TOTAL run(): {time.perf_counter()-t0:.3f}s", flush=True)
        return {k: (v.float().cpu().numpy() if torch.is_tensor(v) else v) for k, v in result.items()}


# ***** cold-start experiment: attack the ~15.9s @modal.enter() directly, since memory
# snapshots are blocked. Three testable changes, all independent of any gated Modal feature:
#   1. import the SAM3 submodules directly instead of top-level `transformers` (skips that
#      package's lazy cross-architecture registry scan just to resolve two class names).
#   2. HF_HUB_OFFLINE/TRANSFORMERS_OFFLINE env vars, so from_pretrained() can't attempt any
#      network metadata check even though the checkpoint is already baked into the image.
#   3. load weights directly in bf16 (torch_dtype=) and straight onto the GPU (device_map=),
#      instead of loading fp32 on CPU then a separate .to(device) call. *****

modal_image_coldstart = (
    modal.Image.from_registry("nvidia/cuda:12.8.0-devel-ubuntu22.04", add_python="3.12")
    .entrypoint([])
    .apt_install("git")
    .pip_install("transformers", "Pillow", "torch", "torchvision", "accelerate")
    .run_function(_download_sam3_weights_for_snapshot, secrets=[modal.Secret.from_name("huggingface")])
    .env({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"})  # after baking -- offline mode would break the download step itself
)


@app.cls(
    gpu="A10G",
    image=modal_image_coldstart,
    secrets=[modal.Secret.from_name("huggingface")],
    scaledown_window=60 * 10,
)
class SegmenterColdStart:
    @modal.enter()
    def load(self):
        import time
        t0 = time.perf_counter()

        import torch
        print(f"[timing] import torch: {time.perf_counter()-t0:.3f}s", flush=True); t1 = time.perf_counter()

        from transformers.models.sam3.modeling_sam3 import Sam3Model
        from transformers.models.sam3.processing_sam3 import Sam3Processor
        print(f"[timing] import Sam3Model/Sam3Processor (direct submodule, not top-level transformers): {time.perf_counter()-t1:.3f}s", flush=True); t1 = time.perf_counter()

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[timing] torch.cuda.is_available() check (device={self.device}): {time.perf_counter()-t1:.3f}s", flush=True); t1 = time.perf_counter()

        self.processor = Sam3Processor.from_pretrained("facebook/sam3")
        print(f"[timing] Sam3Processor.from_pretrained: {time.perf_counter()-t1:.3f}s", flush=True); t1 = time.perf_counter()

        self.model = Sam3Model.from_pretrained("facebook/sam3", torch_dtype=torch.bfloat16, device_map=self.device)
        if self.device == "cuda": torch.cuda.synchronize()
        print(f"[timing] Sam3Model.from_pretrained (bf16, device_map={self.device!r} -- loads straight onto GPU): {time.perf_counter()-t1:.3f}s", flush=True); t1 = time.perf_counter()

        print(f"[timing] TOTAL @modal.enter() load(): {time.perf_counter()-t0:.3f}s", flush=True)

    @modal.method()
    def run(self, image: Image.Image, prompt: str) -> dict:
        import time
        import torch

        t0 = time.perf_counter()
        inputs = self.processor(images=image.convert("RGB"), text=prompt, return_tensors="pt").to(self.device)
        pixel_values = inputs["pixel_values"].to(torch.bfloat16)
        if self.device == "cuda": torch.cuda.synchronize()
        print(f"[timing] processor(...) + .to(device): {time.perf_counter()-t0:.3f}s", flush=True); t1 = time.perf_counter()

        with torch.no_grad(): outputs = self.model(pixel_values=pixel_values, input_ids=inputs["input_ids"], attention_mask=inputs.get("attention_mask"))
        if self.device == "cuda": torch.cuda.synchronize()
        print(f"[timing] model forward pass: {time.perf_counter()-t1:.3f}s", flush=True)

        result = self.processor.post_process_instance_segmentation(
            outputs, threshold=0.0, mask_threshold=0.5, target_sizes=inputs["original_sizes"].tolist()
        )[0]
        print(f"[timing] TOTAL run() incl. postprocess, BEFORE return/serialization: {time.perf_counter()-t0:.3f}s", flush=True)
        print(f"[timing] num candidate detections at threshold=0.0: {len(result['scores'])}, masks shape: {tuple(result['masks'].shape) if hasattr(result['masks'], 'shape') else 'n/a'}", flush=True)
        return {k: (v.float().cpu().numpy() if torch.is_tensor(v) else v) for k, v in result.items()}

    @modal.method()
    def run_small_result(self, image: Image.Image, prompt: str) -> float:
        """Identical GPU work to run(), but returns a single float instead of the full
        {scores, boxes, masks} dict -- isolates whether a large un-thresholded return payload
        (every DETR query slot's full-resolution mask, not just the kept detections) is what's
        adding seconds to the client-observed round trip on top of the ~0.2s server-side work."""
        import torch
        inputs = self.processor(images=image.convert("RGB"), text=prompt, return_tensors="pt").to(self.device)
        pixel_values = inputs["pixel_values"].to(torch.bfloat16)
        with torch.no_grad():
            outputs = self.model(pixel_values=pixel_values, input_ids=inputs["input_ids"], attention_mask=inputs.get("attention_mask"))
        result = self.processor.post_process_instance_segmentation(
            outputs, threshold=0.0, mask_threshold=0.5, target_sizes=inputs["original_sizes"].tolist()
        )[0]
        return float(result["scores"].max())

    @modal.method()
    def run_topk(self, image: Image.Image, prompt: str, top_k: int = 2) -> dict:
        """The real fix: identical GPU work, but filter to the top_k highest-scoring detections
        (mirroring what image.py's segment() already does client-side via argsort -- just doing
        it server-side, BEFORE serialization, instead of after paying to ship all 200 candidates
        over the network first) and cast masks to bool instead of float32 (4x smaller; masks are
        already binarized by mask_threshold=0.5, so bool loses nothing)."""
        import time
        import torch

        t0 = time.perf_counter()
        inputs = self.processor(images=image.convert("RGB"), text=prompt, return_tensors="pt").to(self.device)
        pixel_values = inputs["pixel_values"].to(torch.bfloat16)
        with torch.no_grad():
            outputs = self.model(pixel_values=pixel_values, input_ids=inputs["input_ids"], attention_mask=inputs.get("attention_mask"))
        result = self.processor.post_process_instance_segmentation(
            outputs, threshold=0.0, mask_threshold=0.5, target_sizes=inputs["original_sizes"].tolist()
        )[0]

        order = torch.argsort(result["scores"], descending=True)[:top_k]
        result = {k: v[order] for k, v in result.items()}
        print(f"[timing] TOTAL run_topk() incl. postprocess, BEFORE return/serialization: {time.perf_counter()-t0:.3f}s", flush=True)

        return {
            "scores": result["scores"].float().cpu().numpy(),
            "boxes": result["boxes"].float().cpu().numpy(),
            "masks": result["masks"].cpu().numpy().astype(bool),  # bool, not float32 -- 4x smaller
        }


@app.local_entrypoint()
def payload_size_test():
    """Same warm container, same GPU work -- compare client round trip for the full
    {scores,boxes,masks} result vs a single float, to isolate serialization/transfer cost."""
    image = Image.open(IMAGE_PATH)
    prompt = list(PROMPTS.values())[0]
    seg = SegmenterColdStart()

    print("warming up...")
    seg.run.remote(image, prompt)

    print(f"\nfull result (scores+boxes+masks dict), {N_REPEATS} repeats:")
    full_times = []
    for i in range(N_REPEATS):
        t0 = time.perf_counter(); seg.run.remote(image, prompt); t = time.perf_counter() - t0
        full_times.append(t); print(f"  [{i}] {t*1000:.1f} ms")

    print(f"\nsmall result (single float), {N_REPEATS} repeats:")
    small_times = []
    for i in range(N_REPEATS):
        t0 = time.perf_counter(); seg.run_small_result.remote(image, prompt); t = time.perf_counter() - t0
        small_times.append(t); print(f"  [{i}] {t*1000:.1f} ms")

    print(f"\nmean full-result round trip:  {np.mean(full_times)*1000:.1f} ms")
    print(f"mean small-result round trip: {np.mean(small_times)*1000:.1f} ms")

    print(f"\ntop_k=2 result (the real fix -- server-side filter + bool masks), {N_REPEATS} repeats:")
    topk_times = []
    for i in range(N_REPEATS):
        t0 = time.perf_counter(); result = seg.run_topk.remote(image, prompt, 2); t = time.perf_counter() - t0
        topk_times.append(t); print(f"  [{i}] {t*1000:.1f} ms (masks shape: {result['masks'].shape}, dtype: {result['masks'].dtype})")

    mean_full, mean_topk = float(np.mean(full_times)), float(np.mean(topk_times))
    print(f"\nmean top_k=2 round trip:      {mean_topk*1000:.1f} ms")
    print(f"speedup vs full (unfiltered) result: {mean_full/mean_topk:.2f}x")


@app.local_entrypoint()
def coldstart():
    """Genuinely cold measurement (ephemeral app -- fresh container every invocation, no
    deploy/stop dance needed) of the import/offline/bf16-load optimizations, vs the
    instrumented production Segmenter.load() baseline (15.883s enter + 1.871s first
    inference = 17.754s total, single object -- see segment.py's own [timing] output)."""
    image = Image.open(IMAGE_PATH)
    prompt = list(PROMPTS.values())[0]

    seg = SegmenterColdStart()
    print("cold call (fresh container, single object) -- watch stdout above for [timing] lines")
    t0 = time.perf_counter()
    result = seg.run.remote(image, prompt)
    cold_total = time.perf_counter() - t0
    print(f"\nclient-observed round trip, cold (includes fresh container + enter() + run()): {cold_total*1000:.1f} ms")

    print(f"\nwarm repeats ({N_REPEATS}x, same container, single object per call):")
    warm_times = []
    for i in range(N_REPEATS):
        t0 = time.perf_counter()
        seg.run.remote(image, prompt)
        t = time.perf_counter() - t0
        warm_times.append(t)
        print(f"  [{i}] {t*1000:.1f} ms")

    mean_warm = float(np.mean(warm_times))
    print(f"\nmean warm (single object): {mean_warm*1000:.1f} ms")
    print(f"mean warm (both objects, x2 sequential, comparable to run_sequential's 589.5ms baseline): {mean_warm*2000:.1f} ms")


@app.function(image=modal_image, scaledown_window=60 * 10)
def noop() -> str:
    return "ok"


@app.local_entrypoint()
def rpc_overhead():
    """Isolate client<->Modal RPC/transport latency, independent of any model/container work,
    by timing a trivial no-op function call from the client side. If this alone takes seconds,
    the ~11s 'warm' round trips seen in coldstart() (despite ~0.2s server-side run() prints)
    are a transport/RPC issue, not a cold-start or model-loading issue."""
    print("warming up (first call may build/cold-start the no-op container)...")
    t0 = time.perf_counter(); noop.remote(); print(f"  first call: {(time.perf_counter()-t0)*1000:.1f} ms")

    print(f"\ntiming {N_REPEATS+2} repeats of a trivial no-op remote call...")
    times = []
    for i in range(N_REPEATS + 2):
        t0 = time.perf_counter()
        noop.remote()
        t = time.perf_counter() - t0
        times.append(t)
        print(f"  [{i}] {t*1000:.1f} ms")
    print(f"\nmean no-op round trip: {np.mean(times)*1000:.1f} ms")
