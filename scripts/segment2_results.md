# segment2.py — speeding up SAM3 segmentation

Target: `D:\eturok\experiment-24\images\000012\01_cropped.png` (679×309),
`objects = {'cylinder': 2, 'cube': 1}`, `PROMPTS = {'cylinder': 'Metal circle', 'cube': 'Black cube'}`.
All runs on the same GPU as `src/data/segment.py` (Modal `gpu="A10G"`), same warm container,
3-5 timed repeats after a warmup call. Code: [segment2.py](segment2.py).

## Baseline

`src/data/segment.py`'s `segment()` does one `modal.Cls.spawn()` **per object prompt**. Each
call independently re-runs the *entire* SAM3 forward pass — including the vision (ViT) encoder,
the most expensive part — even though both prompts are run against the identical image. It also
runs in plain fp32 (the old `sam3-image-model` code path used `torch.autocast(bfloat16)`, but the
current `transformers`-based `Segmenter.inference()` dropped it).

`run_sequential` in `segment2.py` reproduces this (with autocast re-added, see below) as the
control: **589.5 ms** for both prompts.

## Ideas tried

| # | Idea | Method | Mean time | Speedup | Notes |
|---|------|--------|-----------|---------|-------|
| 0 | baseline: 1 full forward pass per prompt, fp32 | `run_sequential` | 589.5 ms | 1.00x | current `segment.py` behavior, minus the missing autocast |
| 1 | **shared vision encoding**: encode the image once via `model.get_vision_features()`, then run ONE batched forward pass (batch=2) reusing those embeddings for both text prompts. `torch.autocast(bfloat16)`. | `run_shared_vision` | 364.3 ms | **1.62x** | biggest single win — avoids re-running the ViT backbone a 2nd time |
| 2 | idea 1 + weights natively cast to `bfloat16` (`model.to(dtype=torch.bfloat16)`) instead of relying on autocast to cast on the fly, autocast kept as a safety net for stray fp32 buffers | `run_shared_vision_bf16weights` | 352.4 ms | 1.67x | small extra win over plain autocast |
| 3 | idea 2 + `torch.compile(model, dynamic=False)`, compiled once per warm container (cost absorbed into the warmup call, not the timed loop) | `run_shared_vision_compiled` | **325.4 ms** | **1.81x** | **best result — recommended** |
| 4 | idea 3 + `mode="reduce-overhead"` (CUDA graphs), `.contiguous()` on the expanded vision embeds | `run_shared_vision_cudagraphs` | n/a | worse | **rejected**, see below |

Sanity check: detection scores agree closely across all four variants (cylinder top-2 scores
identical: `0.805, 0.789`; cube top-1 drifts `0.543 → 0.535` from bf16 rounding — doesn't change
which detections are picked).

## Idea tried and rejected: CUDA graphs (`mode="reduce-overhead"`)

Adding a 5th variant with `torch.compile(mode="reduce-overhead")` made **every** variant slower and
noisy in the same run — even `run_sequential`, which doesn't touch the new code path at all,
jumped from 589.5 ms to ~1200 ms. Two things are going on:

1. **Resident-model contention.** By this point the container was holding 4 full model copies at
   once on the same A10G (fp32, bf16, `torch.compile`, `torch.compile(reduce-overhead)`) — enough
   memory/scheduling pressure to slow down every variant sharing the GPU, not just the new one.
2. **CUDA graphs need static input addresses to pay off**, and our per-call tensors (freshly
   allocated `expand().contiguous()` vision embeds each call) don't have stable addresses across
   calls — so the runtime was likely re-capturing the graph on every call instead of replaying a
   cached one, which is pure overhead with no reuse benefit.

Rejected. Not worth chasing further without restructuring around a fixed, preallocated input
buffer, which is more engineering than this single-image target justifies.

## Recommendation

**Idea 5 (`SegmenterSmallImage`, 217.6 ms, 2.71x)** is the one to carry into `segment.py`:
shared vision encoding + native bf16 weights + `torch.compile()` + a smaller vision-encoder
grid (`image_size=672` instead of 1008), all measured cleanly on an isolated, properly-warmed
container. Don't add CUDA graphs on top. Note the 672 choice is tuned to this specific
679×309 source image; a much larger source image might need a larger `image_size` to avoid
losing real detail (this isn't "shrink the input," it's "don't upsample past what the source
actually has," so the right value scales with the input).

Also worth noting from measurement: the very first isolated run of `SegmenterCombined` timed
436.7 ms — *worse* than the 325.4 ms recorded earlier inside the multi-variant container —
because only 1 warmup call was used and per-iteration times were still trending down
(488→425→397 ms) when timing started. The CUDA caching allocator and GPU clocks take a few
real iterations to reach steady state even after `torch.compile`'s one-time trace. Fixed by
bumping warmup to 4 calls before timing; all idea-5 numbers above use that fix.

## Idea 5: shrink the actual vision-encoder grid (not just the input)

Downsampling the *input* before it reaches the processor is a dead end (see rejected idea
below) — but the 1008×1008 the processor resizes *to* isn't hardcoded either; it's just a
default. The catch: naively passing a smaller `size` to the **processor** while keeping the
default **model** crashes. `Sam3ViTRotaryEmbedding` precomputes its cos/sin position tables
once, at model-construction time, from `config.image_size` (default 1008, `patch_size=14` →
72×72 = 5184 patches — that 5184 is exactly the mismatched tensor size in the crash trace).
It is not recomputed per forward call from the actual input shape. Feed it a different patch
grid and RoPE fails to broadcast against query/key.

**Fix**: override `Sam3VisionConfig.image_size` *before* constructing the model (not just the
processor), then load the pretrained checkpoint into that config. This is safe because RoPE
tables are plain trig buffers recomputed fresh at init for the new grid — nothing learned to
reconcile there. The one learned tensor that does depend on grid size (absolute position
embeddings added to the patch embed) is already handled by the model's own
`_tile_position_embeddings` interpolation. Constraint: the new size must stay a multiple of
`patch_size` (14) and `window_size * patch_size` (24×14=336) so the patch grid divides evenly
for windowed attention — same constraint the default 1008 (=72×14, 72/24=3) already satisfies.
Chosen: **672** (=48×14, 48/24=2).

| # | Idea | Method | Mean time | Speedup | Notes |
|---|------|--------|-----------|---------|-------|
| 5 | idea 3 + model rebuilt with `vision_config.image_size=672` (grid 48×48=2304 patches vs 5184 at 1008) | `SegmenterSmallImage.run` (isolated) | **217.6 ms** | **2.71x** | **new best** |

Sanity check: scores *improved*, not degraded — cylinder top-2 `0.836, 0.812` (vs `0.805, 0.789`
at 1008px), cube top-1 `0.648` (vs `0.535`). Source image is 679×309, so 1008 was upsampling
it further than 672 does; less upsampling blur, not a quality tradeoff, in this case.

The native `sam3` package's `Sam3Processor.__init__(self, model, resolution=1008, ...)` takes
the same resolution as a plain constructor kwarg — same lever, cleaner surface (no separate
model-config step needed because that package computes its rotary tables the same way per
instantiation, not decoupled from a processor-level resize).

## Idea 6: push image_size down further (336, i.e. half of 672)

Parameterized `SegmenterSmallImage` by the same `scale` knob the original `segment.py` exposed
(1.0 = 1008 default), snapped to the nearest valid multiple of 336 (`resolve_image_size()`).
`scale=1/3` → **image_size=336** — notably, this equals `pretrain_image_size` (336), the grid
the absolute position embeddings were originally trained at, so no interpolation is even needed
there.

| # | Idea | Method | Mean time | Speedup | Notes |
|---|------|--------|-----------|---------|-------|
| 6 | idea 5 + `image_size=336` (grid 24×24=576 patches) | `SegmenterSmallImage.run(image_size=336)` | 159.8 ms | 3.69x | **faster, but real quality cost** — see below |

This is the fastest number measured, but unlike the 672 step, it's **not a clean win**: cylinder
top-2 scores dropped from `0.836, 0.812` (at 672) to `0.645, 0.602` — a meaningful confidence
drop, not noise (cube's score is roughly flat, `0.648` → `0.613`). Visual inspection
(`segment2_outputs/masks_size336.png`,
`labelmap_size336.png`) shows both objects still land in roughly the right place — the cube's
mask is a reasonably tight blob, both cylinder blobs sit at the correct positions — so it isn't
badly wrong, but 336 is visibly at the edge of useful quality for these particular
(small-in-frame) objects. The oversized axis-aligned box drawn around the cube in the composite
image is expected geometry, not a bug: the cube is rotated ~45°, so its axis-aligned bounding
box legitimately spans wider than its visible silhouette.

**Recommendation stands at 672, not 336** — 672 was a free win (faster *and* better scores);
336 trades meaningful confidence for another ~1.4x on top of that. Worth it only if a
downstream confidence threshold is already lenient and raw speed matters more than margin.

## Idea considered and rejected

- **Downsample the input image before inference.** Checked `Sam3ImageProcessor`
  (`image_processing_sam3.py`): it resizes every input to a fixed **1008×1008** regardless of
  source resolution. So shrinking the input doesn't reduce GPU compute at all — the model does
  the same amount of work either way — while risking a quality loss the user explicitly didn't
  want. Not pursued. (Superseded by idea 5, which shrinks the actual processing grid instead
  of the input — that's the version of this idea that actually works.)

## Key mistake made + fixed

First pass at idea 2 crashed the `detr_encoder`'s `q_proj` linear (`mat1 and mat2 must have the
same dtype, but got Float and BFloat16`): casting the model to bf16 weights doesn't move plain
tensors that aren't registered `nn.Parameter`/buffers (e.g. some position-encoding tensors are
built fresh per forward call in fp32). Fix: keep `torch.autocast(bfloat16)` wrapped around the
forward pass even when using the bf16-weight model, as a safety net that up-casts any op given a
stray fp32 input.

## Not yet tried (future ideas)

- `torch.compile(mode="max-autotune")` — better fusion, but much longer compile time per container
  warmup; only worth it on a long-lived warm container (e.g. `min_containers=1`).
- int8 quantization — untested, likely more engineering than the returns justify for a single-image
  target.
