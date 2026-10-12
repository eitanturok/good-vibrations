# viz performance notes

Working log for the "viz uses 6GB and takes 10s to add a run" investigation. Keep this
updated when touching load paths in `viz/data.py`/`viz/app.py` — it's the reference for
what was already tried, what it cost, and why, so the next pass doesn't re-measure from
scratch or re-try something already ruled out.

## How to read the live numbers

Every heavier load path now prints a `[viz]` line with timing and `rss=` (this process's
resident memory, from `_rss_mb()` in `viz/data.py`). Watch the tmux pane running `python -m
viz`:

- Startup: `[viz] loaded N experiments' ground truth in Xs | rss=...` then
  `[viz] <n> compatible / <n> incompatible runs | Xs scan | Xs total | rss=...`
- Adding a run (cache miss): `[viz] load_run '<name>': <n> .pt files, <n> samples, Xs`
  then `[viz] loaded run '<name>' epoch=... in Xs | ... | rss=... | cache=<n> runs`
- Every `/api/run/{name}` request: `[viz] /api/run/<name>: registry.run()=Xs,
  json-build=Xs` — separates a cache hit/miss from the per-request JSON assembly cost.

## Baseline (before this pass), single `python -m viz` against all 6 experiments

| stage | time | RSS |
|---|---|---|
| `import torch` alone (nothing else) | — | ~450-480MB |
| GT load, 6 experiments / 7606 samples | 4.62s | 921MB |
| full startup (GT + run-dir scan of ~500 runs) | 5.75s total | 921MB |
| first `addRun` (cache miss, ~15-30 .pt files) | 1.2-1.8s | +50-100MB per run |
| `Registry._runs` cache | — | **unbounded** — every run ever added stayed decoded in memory for the process lifetime, even after being removed from the table client-side (server has no "removed" signal) |

Measured with `tracemalloc` (see below) and `resource.getrusage().ru_maxrss` via a
throwaway script, not guessed.

## What's actually in the 6GB — investigated, not assumed

Ran `tracemalloc` against `load_gt()`/`load_experiments()` directly (bypassing the server)
to get real top-allocator output rather than guessing from reading the code:

```
json/decoder.py:354  size=95.5 MiB (+95.5 MiB) count=2,283,286   <- for ONE 3007-sample experiment
```

Traced it to `merge_metadata()` (`viz/data.py`) fully parsing every line of every sample's
`metadata.jsonl` and keeping the result forever in `GtIndex.meta`. Two keys —
`rois` and `run_opt_multiROIs` — are ~4.3KB of a typical ~4.7KB metadata file (checked by
dumping every key's JSON size per sample across all 6 experiments; consistent everywhere).
They're the capture rig's laser-grid geometry, read only by the TRAINING pipeline
(`src/data/vibrate.py`, `src/model/dataset.py`) — grepped the whole repo, `viz/` never
touches them.

`import torch` alone costs ~450-480MB RSS before any viz code runs at all — CUDA/MKL
runtime gets mapped in on import regardless of whether a GPU is used. This is an accepted
floor of using PyTorch in this process (viz needs it for `torch.load` on prediction files
and for the metrics in `utils/metrics.py`); not worth chasing without a much bigger
architectural change (e.g. moving prediction decoding to a subprocess), which is out of
scope here.

## Fixes applied

### 1. Drop unused metadata fields at parse time (`viz/config.py` `STALE_METADATA_KEYS`)

Added `"rois"`, `"run_opt_multiROIs"`, `"run_opt"` to the existing stale-key filter in
`merge_metadata()` — a one-line change, no new code path. Verified with the same
`tracemalloc` script: `json/decoder.py` retained memory for that one experiment dropped
**95.5MB → 12.3MB**. Full-server before/after:

| | before | after |
|---|---|---|
| GT load, 6 experiments | 921MB, 4.62s | **738MB, 3.38s** |

~183MB / ~20% RSS cut, ~27% faster, from one line. Highest-leverage single fix found.

### 2. Cap `Registry._runs` with a real LRU (`viz/data.py` `Registry.run()`, `viz/config.py MAX_RUN_CACHE`)

The cache already evicted *scrubbed-epoch* entries beyond `MAX_EPOCH_CACHE`, but
`epoch=None` entries (what every "add this run to the table" click creates) were **never**
evicted — the comment literally said "the latest-epoch entry per run is never evicted".
Over a session that explores many runs (exactly what the new multi-dataset comparison
feature encourages), this is unbounded growth with no server-side signal that a run was
ever removed from the table.

Fix: dict re-insertion on cache hit (marks "recently used" — plain dicts keep insertion
order, no extra structure needed) plus an LRU eviction loop capped at `MAX_RUN_CACHE = 24`
(covers any table anyone actually keeps open at once; only kicks in once a session has
explored well beyond that). Simplicity check: could have used `functools.lru_cache`, but
this cache needs manual invalidation (`reload=1`) that decorator doesn't support cleanly,
so a plain dict with an eviction loop is both simpler and correct here.

### 3. `ensureFrames` window bug in "by dataset" row mode (`viz/static/app.js`)

Not a memory fix, but the direct cause of "takes forever to see a new column load" once
the "by dataset" row mode existed: the frame-prefetch window was always computed from the
flat `S.order`, but in that mode a run's cells paint against its OWN dataset group's row
list — a different set of samples at different positions. The fetched window almost never
matched what was on screen, so `paintCanvas` kept finding a miss and re-triggering another
fetch: not slow, actually stuck in a retry loop. Fixed to source the window from the run's
own group. See the "by dataset" row mode PR/commit for the full fix.

### 4. Duplicate-add race (`viz/static/app.js` `addRun`)

Also not memory, but compounds #3's perceived slowness: the picker no longer auto-closes
after a pick (a separate, deliberate UX change), so a second click on the same run while
its `/api/run/` fetch was still in flight slipped past the `S.runs[name]` guard (only set
after the `await`) and created a duplicate column. Fixed with a synchronous `addInFlight`
set checked before the fetch starts.

## Tried / considered, not applied

- **Moving prediction decoding off the main process** (subprocess or worker pool) to avoid
  the torch-import floor blocking the request thread during a slow `load_run`. Real
  potential win, but a much bigger architectural change than this pass's scope — revisit
  if `load_run`'s 1-2s per run (see the `[viz] load_run` print) becomes the bottleneck
  again after the above fixes land.
- **Lazily parsing metadata.jsonl** (skip parsing `rois`/`run_opt_multiROIs` at the JSON
  level entirely, e.g. a custom decoder) instead of parse-then-filter. Would save the
  transient parse-time allocation too, not just the retained one — but `merge_metadata` is
  called once per sample at startup only, so the CPU cost of parsing-then-discarding is
  negligible next to the retained-memory fix; added complexity wasn't worth it.
- **Reducing GtIndex.masks memory** — checked: masks are already `float32`, already just
  the primary shape (`by_shape` for OTHER shapes is genuinely lazy, loaded on first
  request). Not a real cost (9.6MB for a 3007-sample experiment); ruled out.

## Re-benchmark after all fixes

```
[viz] loaded 6 experiments' ground truth in 3.38s | rss=738MB
[viz] 320 compatible / 178 incompatible runs | 0.99s scan | 4.37s total | rss=738MB
[viz] load_run 'pp-pixelshuffle-v2': 14 .pt files, 2776 samples, 1.79s
[viz] loaded run 'pp-pixelshuffle-v2' epoch=None in 1.80s | rss=782MB | cache=2 runs
```

Startup: 921MB → 738MB RSS, 4.62s → 3.38s GT load. Run-cache growth is now bounded at 24
entries regardless of session length (was unbounded). Per-run add time (~1.2-1.8s server
side) is dominated by `torch.load`-ing each `.pt` file — see "tried / considered" above for
the path to cut that further if it's still the bottleneck after these land.

## Epoch scrubbing (2026-10-08)

Symptom: arrow-stepping the epoch slider took 22-51s per step to settle, and scrubbed masks
never appeared for s6-* runs. Measured with playwright (keydown -> every visible mask redrawn)
plus cProfile on load_run.

- Each step fired `/api/run?epoch=N` per loaded run. That re-runs load_run and re-scores ALL
  ~7.5k samples (~7s, ~90% in utils/metrics labeling/contour/matching; the .pt reads are
  ~15ms), one at a time behind Registry._load_lock. 3 runs = up to 51s queued per step, and
  cells hid their mask ("loading…") until it landed.
- `/api/frames` for numbered epochs matched rows on (local id, box), but newer .pt files carry
  `experiment` and no `box`, so every scrubbed epoch came back NaN ("not saved").

Fix (simplification, not caching): the scrubber is masks-only. Rows, order, filters and
metrics are the latest epoch's; cells show "metrics at latest epoch only" while scrubbed.
Removed `/api/run?epoch`, the epoch-keyed run cache + MAX_EPOCH_CACHE, client
fetchEpochMetrics/metricsPending/frozenOrder. load_epoch_masks routes rows through
registry.resolve like load_run does. Polling no longer wipes all frames when a run advances
(only "latest" changes). Result: 17-25ms per step, zero requests while stepping; prefetching
all 21 epochs x 40 rows is ~0.9s per run, in the background.

## Per-sample metrics saved by training (2026-10-08)

src6 OutputSaver now writes `metrics` ({name: (B,) float32}, NaN where a metric does not
apply) and `geometry` (localization's crosshair pairs) into every .pt -- see
`src6/callbacks.py sample_metrics`: training's iou/object_acc/n_objects_err/com_err/
area_rel_err per sample, the ConfidenceLogger scores (peak, peak_obj, peak_bg, fill, miss),
and viz's localization_rel/_raw + contour. load_run reads them when EVERY file of the loaded
epoch has the four METRIC_KEYS, else scores as before (old runs, or a run resumed across the
change). Measured on s6-dice0-v1's 8015 rows rewritten in the new format: 14.4s -> 0.34s,
metrics and geometry bit-identical. Training cost: ~43ms per 108-sample batch on GPU, only
on save epochs.

## Startup (2026-10-08)

Launch -> masks visible went ~20s -> ~4.0s (server listening 2.8s, page 1.2s). Registry
init 11.4s -> 2.1s, measured without the profiler, each step found with cProfile:

| step | before | after | fix |
|---|---|---|---|
| GT object centroids | 6.3s | cached | `_label` CPU path: each mask stops at its own fixed point (one 64-wide blob held all 6155 masks for 64 rounds); 2.75s -> 0.66s uncached, bit-identical |
| GT load (npy + metadata + centroids) | ~9s | 1.3s | pickled per experiment in ~/.cache/viz, validated by stat of every sample dir, image/ dir, metadata.jsonl and mask used (~80ms) |
| run scan | 1.8s | 0.85s | `_pred_files` returns scandir strings; `_epoch_of`/`_batch_of` on os.path (Path re-parsed itself ~430k times); `config.subdirs` sorts names, not Path objects (~1M Path compares) |
| /api/samples | 0.6-1.0s | 0.06s | encoded once per experiment count; json.dumps instead of jsonable_encoder |
| /api/run json | 0.4s | 0.05s | json.dumps instead of jsonable_encoder |

Old-format runs (scored in viz) load in 1.2-5s instead of 7-14s, from the same `_label` fix.
Every startup payload (/api/samples, /api/runs, /api/lut, /api/run for old and new runs)
checked JSON-equal against the previous code.
