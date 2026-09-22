# src5 training speed optimization log

Baseline (measured via `src5/profile_step.py --batch-size 8`, on the training GPU, bf16 autocast):

| component | time |
|---|---|
| encoder forward | 10.3 ms |
| decoder forward | 22.5 ms |
| load_latents (disk I/O) | 3.1 ms |
| frozen AE decode | 33.0 ms |
| foreground_prob (softmax+interp, metrics only) | 6.6 ms |
| loss (ce+mask+latent) | 18.5 ms |
| **full step (fwd+bwd+opt)** | **131.0 ms → 61.1 samples/sec** |
| peak GPU memory | **11.97 GB** / 15.47 GB |

Actual observed training throughput (~53-55 samples/sec) is a bit lower than this isolated
profile's 61.1, from dataloader/Composer/metrics overhead not captured here.

Biggest finding: **memory-bound, not obviously compute-bound**. 12GB at batch_size=8 out of
15.47GB is why batch_size=16 OOM'd earlier -- there isn't much headroom to just raise batch size
without also cutting memory. The `(B,128,512,512)` tensor from the frozen AE's decode (and its
softmax copy in `foreground_prob`) is the dominant memory cost.

## Attempts

### ❌ Avoid full softmax in `foreground_prob` (logsumexp + gather instead of `.softmax(1)`)
Idea: `foreground_prob` only needs the probability mass of a few classes per sample, not a full
`(B,128,H,W)` softmax tensor -- computing `exp(logit - logsumexp)` only for the gathered classes
should avoid that big intermediate allocation.
**Result: no memory benefit (peak stayed 11.97GB, unchanged) and the isolated op got ~2x slower**
(6.6ms → 11.5ms) -- `.softmax(1)` is a fused, well-optimized kernel; the manual logsumexp+exp
sequence isn't, and the true memory peak is dominated by autograd-saved activations elsewhere in
the network (the frozen AE's decode conv stack), not this one tensor. **Reverted.**

### ✅ `torch.compile` on `model.encoder` and `model.decoder` (the two trainable modules)
`torch.compile(model.encoder)` / `torch.compile(model.decoder)`, frozen `mask_ae` left uncompiled.
**Result: full step 131ms → 112.8ms (61 → 70.9 samples/sec, +16%), peak memory 11.97GB → 7.64GB
(-36%).** One-time compilation cost ~6.4s (negligible over a 500-epoch run). Clear win on both
axes -- the freed-up memory is what actually unblocks a real batch-size increase (see next).
Kept.

### ⚠️ Higher batch size (the other requested low-hanging fruit) -- doesn't actually help throughput
With compile enabled, tested batch_size 8/12/16/20 (full step time, peak memory, samples/sec):

| batch_size | full step | samples/sec | peak mem |
|---|---|---|---|
| 8  | 112.8 ms | 70.9 | 7.64 GB |
| 12 | 166.9 ms | 71.9 | 11.23 GB |
| 16 | 226.8 ms | 70.5 | 14.83 GB (96% of GPU -- too tight to trust in real training) |
| 20 | OOM | -- | -- |

**Finding: samples/sec is flat (~70-72) across all working batch sizes.** This workload is
compute-bound per-sample, not fixed-per-step-overhead-bound, so a bigger batch just makes each
step proportionally longer for the same total wall-clock throughput -- it does NOT unlock extra
speed the way it would for a workload dominated by fixed per-step Python/launch overhead. The
compile-freed memory just buys a *safer* headroom at bs=8, not a reason to raise it.
**Recommendation: keep batch_size=8** (real training already validated stable there; bs=16 is
too close to the OOM ceiling to trust once Composer/dataloader overhead is added back in).

### ✅ Batch `load_latents`' device transfers (found via `torch.profiler` on a real step)
`torch.profiler` on 3 real steps showed `cudaStreamSynchronize`/`cudaDeviceSynchronize` eating
~91% of CPU time. `load_latents` was doing up to 2 separate `.to(device)` calls PER SAMPLE
(`instance_masks`, `target_classes`) inside its per-sample loop -- up to 18 small blocking
transfers/step. Fixed by accumulating all samples' tensors on CPU first, `torch.cat`-ing once,
transferring the whole batch in one shot, then `torch.split` back into per-sample lists.
**Verified correct** (identical values, round-trips a 2-instance sample and an empty-box sample
correctly). **Speed: no measurable change** -- direct isolated timing showed `load_latents`
itself is only ~2.5ms/call even *before* this fix, so it was never actually the bottleneck; the
profiler's sync numbers turned out to be dominated by something not scaling with call count
(likely fixed one-time overhead within the 3-iteration profiling window, not a genuine per-step
cost -- consistent with real training throughput being unchanged by this fix). **Kept anyway**
since it's strictly fewer, larger transfers with no downside, just not the lever we hoped.

### ⚠️ More dataloader workers (4 → 16)
No change (~54-59 samples/sec either way, real training). Rules out the CPU-side FFT
tokenizing/dataloader as the bottleneck, at least at 4 workers already (32 CPUs available).

### ⚠️ Fewer PointRend sample points (12544 → 4096 → 2048)
No measurable difference (81.1/81.1/80.6 samples/sec, isolated). PointRend's point-sampling cost
is negligible next to whatever the real bottleneck is -- not worth trading accuracy for.
**Reverted to the paper's default (12544).**

### 🔍 Root cause of the isolated-vs-real gap, found

Ruled out one by one: metric computation (1.82ms, negligible), raw dataset `__getitem__` cost
(0.73ms/sample single-threaded, i.e. ~1370 samples/sec -- nowhere near the bottleneck),
`OptimizerMonitor`, more dataloader workers, `interpolate=False` on the frozen AE's decode (saves
only ~8%). Direct test: timing `full_step()` with a **fresh batch pulled from the real dataloader
each iteration** (instead of profile_step.py's one-fixed-batch-reused approach) gives **56.8
samples/sec** -- matching real training almost exactly. **The isolated harness's 70-81
samples/sec was never representative**: it reused one batch across all timed iterations, skipping
the per-step host tensor construction + H2D transfer that real training always pays. There is no
hidden, fixable overhead -- **~55-57 samples/sec is close to the actual achievable ceiling** for
this architecture (frozen AE decode of a (B,128,512,512) tensor) on this GPU, given everything
tested here. Further gains would need reducing the real compute (e.g. a smaller decode
resolution, at a real accuracy cost) rather than removing overhead.

### ✅✅ Skip the frozen AE's final bilinear upsample (`decode(z, interpolate=False)`)

The frozen AE's `decode()` does 2 learned upsampling stages (64→256, via real transpose convs)
then ONE extra non-learned bilinear upsample (256→512, `interpolation_factor=2`) to reach its
native training resolution. Our loss (`loss_ce`/`loss_mask`, PointRend point-sampled) and metrics
(`foreground_prob`, already downsampled to the project's native 32x32 grid) don't need that last
512 res -- the bilinear step adds no new learned detail, just smooths. Tested with the correct
fresh-batch-per-step methodology (see the entry above -- reused-batch profiling is misleading):

| | 512 (interpolate=True) | 256 (interpolate=False) |
|---|---|---|
| step time | 142.0 ms | 85.4 ms |
| samples/sec | 56.3 | **93.7 (+66%)** |
| peak GPU memory | 10.35 GB | **5.27 GB (-49%)** |

Ground-truth `target_class_map`/`instance_masks` (cached at 512, fixed by the frozen encoder's
own native resolution) are downsampled once to match via nearest-neighbor (keeps mask edges
binary, no blending) in `load_latents`, gated by a `decode_res` param. **Adopted as the new
default** (`--full-res-decode` opt-in flag added to restore 512 for comparison/ablation).
Tradeoff: half the linear position precision in the loss's working resolution -- worth watching
`iou`/`contour`/`mass`/localization metrics against the earlier 512-res baseline runs to confirm
this doesn't meaningfully hurt mask quality; revert via `--full-res-decode` if it does.

### ⚠️ Re-tested batch size at the new 256-res/compiled baseline -- still no throughput win

Now that 256-res decode dropped bs=8's peak memory to 4.86GB, much bigger batches fit (bs=24 is
the first to OOM). But throughput is flat-to-slightly-worse as batch size grows:

| batch_size | samples/sec | peak mem |
|---|---|---|
| 8  | 91.6 | 4.86 GB |
| 12 | 90.1 | 7.06 GB |
| 16 | 87.6 | 9.27 GB |
| 20 | 85.6 | 11.47 GB |
| 24 | OOM | -- |

Same conclusion as the original (512-res) batch-size test: compute-bound per-sample, not
overhead-bound, so a bigger batch buys memory headroom, not speed. **bs=8 remains the
recommendation** -- it's both the fastest and the safest margin.

## Summary / current recommendation

**Keep**: `torch.compile` on encoder+decoder (free, +16%/-36% mem isolated, harmless in real
training), the `load_latents` batching fix (harmless, more correct practice), `batch_size=8`.

**Honest unresolved gap**: isolated model-only profiling reaches ~70-81 samples/sec depending on
exact measurement setup, but real Composer training consistently plateaus at ~54-59 samples/sec
regardless of every lever tested here (compile, batch size, workers, OptimizerMonitor, PointRend
points, load_latents batching). None of the "usual suspects" explain the ~25-30% gap. Candidates
not yet tested, for a future session: direct profiling of `VibrationDataset.__getitem__`/
`process_vibration`'s CPU cost per sample (dataloader-side, not model-side), and Composer's own
per-batch metric-computation/logging/callback-event overhead (5 MaskMetric-family objects +
2 LatentMetric objects updated every step, `_seg_batch`'s caching notwithstanding).

