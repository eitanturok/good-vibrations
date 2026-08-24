# viz

Compare predicted segmentation masks across training runs, sample by sample.

```bash
python -m viz                 # http://127.0.0.1:8503
python -m viz --port 9000 --experiment experiments/experiment-25 --runs runs
python -m viz --experiment experiments/31_07_2026_gastronorm_exp1 --mask 30x30
```

Startup is ~0.5 s. No inference runs: ground-truth masks are read from the per-sample
`.npy` on disk and predictions from the `.pt` files the `OutputSaver` callback already
wrote during training.

## Watching a remote training run

`src/run.py` starts viz for you, in a detached tmux session named `viz-<port>` (default
8504), pointed at the run's own `--data-dir`. It survives the training job ending, and later
runs reuse the same session rather than starting a second one — so one server covers every
run against that experiment. Training on a **different** `--data-dir` relaunches it on the
same port, so the dashboard follows what you're training instead of quietly showing the
previous dataset; your tunnel and browser tab keep working across the swap.

On the laptop, hold an SSH tunnel and open it:

```bash
./scripts/connect_to_batman.sh          # tmux + ssh -L 8504:localhost:8504 batman
```

Then browse http://localhost:8504. Order doesn't matter — viz outlives any single run, and
new runs appear within ~10 s without a restart. The tunnel retries on drops, so it survives
laptop sleep and wifi changes.

On the remote:

```bash
tmux attach -t viz-8504          # logs (the pane stays open if viz crashes)
tmux kill-session -t viz-8504    # stop it
python src/run.py --no-viz ...    # don't launch it; --viz-port to move it
```

viz stays bound to `127.0.0.1`, so the tunnel is what makes it reachable — it is never
exposed on the network.

## Dataset layouts

Sample directories are not laid out the same way across experiments, so the filenames
are detected at startup rather than hardcoded — the banner prints which layout won.

| | `experiment-25` | `gastronorm` |
|---|---|---|
| image dir | `image/` | `image/` |
| GT mask | `05_downsampled_smask_{H}h_{W}w.npy` | `04_downsampled_smask_{H}h_{W}w.npy` |
| backdrop | `01_cropped.png` | `02_cropped_overhead.png` |
| overhead | `05_overhead_speaker.png` | *(none — falls back to the crop)* |
| audio | `audio.wav` + `recovered_audio.wav` | `recovered_audio.wav` only |
| scene id | `output_id` | `position_id` |
| `avg_com` | JSON list | `str(ndarray)`, e.g. `"[603.1 901.2]"` |

Add a new format by appending an entry to `LAYOUTS` in `config.py`.

**Sample ids are not row indices.** The gastronorm dataset starts at `000009`, and any
dataset can be missing a sample whose mask was never written, so every id → row lookup
goes through `GtIndex.row_of`.

**`--mask HxW`** picks the target grid. A dataset may ship several sizes side by side
(gastronorm has both `20x40` and `30x30`); viz uses the only size on disk when there is
one, otherwise defaults to `20x40` and says so. Runs trained on a different size are
listed as incompatible, with the shape mismatch as the reason — so if a run you expect is
missing, check the mask size first.

## Layout

One row per sample. Column 1 is the row number, column 2 the ground truth (sample id,
overhead photo with mask/COM/speaker), and every column after that is one training run's
prediction, with per-sample mse / soft-IoU / COM-distance above each mask and
mean ± std for the filtered set in the header.

## Finding bad predictions

Click a run column header to sort the whole table by that run, cycling
worst → best → unsorted; pick the metric from the dropdown. Direction is labelled
**worst/best** rather than asc/desc, because "worst" means *high* MSE but *low* IoU.
Sorting is global, so a run's worst samples stay aligned with every other run's
prediction for the same sample.

## Filters (left)

- **Position** — scatter of `avg_com`, one point per physical position (~125 of them,
  8 samples each — one per speaker). Drag to lasso, shift-drag to add, click to select,
  ctrl/cmd-click to toggle, and arrow keys to walk to the neighbouring position.
- **Speaker** — diagram drawn from the same `SPEAKER_POSITION` constants that place the
  speaker into `05_overhead_speaker.png`, so it matches the photos.
- **Split / layout / objects** — chips with live counts.
- **Metrics** — dual-handle range sliders. They read the sorted run when one is
  selected, otherwise a sample passes if any loaded run is in range.

The count under **+ Add run** says how many samples survived, and — when rows are
missing — names each filter that is holding rows back, with how many *only* that filter
is hiding. Each name is a button that switches that one filter off and leaves the rest
alone. `not predicted by any loaded run` is the exception: no chip can bring those back,
so it is reported but not clickable.

Two filters are easy to confuse, and the readout is how you tell them apart: **split**
`2-cubes` is the handful of *held-out positions* a run evaluated on, while **objects**
`2` is every two-object sample in the dataset. Filtering to the split and expecting the
object count leaves ~3 rows per speaker where you expected ~286.

Selected is a filled accent; deselected is muted, desaturated and dashed — the same
rule for chips, speakers and scatter points.

## Display

`Prediction` shows the mask on a single-hue blue ramp (values are soft probabilities,
so the ramp is continuous). `Difference` shows `pred − truth` on a fixed [−1,+1]
diverging scale: **red = predicted mass that isn't there, blue = real mass the model
missed**, gray = agreement. Domains are fixed, never per-cell autoscaled, so cells stay
comparable across the whole table. "Show background" toggles the opaque fill off so the
mask silhouette floats. Hovering a mask reads out the value at that grid cell.

Clicking a ground-truth cell opens the sample: large overhead image, object/COM details,
and original + recovered audio with the precomputed spectrogram and FFT. ←/→ step
through the current sort order.

## Which runs appear

Only runs that can be validly compared against this experiment. A run is rejected if it
has no `outputs_history/`, has a different mask shape, uses the legacy `info` schema, or
was trained on a **different dataset** — sample ids collide across experiments, so a
cylinder/bullet run would otherwise join cleanly and report silently wrong numbers.
Rejected runs stay visible in the picker with the reason. Truncated `.pt` files are
skipped per-file and reported in the column header rather than failing the run.

## Correctness

Metrics mirror the training loop (`utils/metrics.py`, `src/model/arch.py`), so column
headers reproduce the run's own logs exactly. To check:

```bash
grep -E "batch=<epoch*3>\]: metrics/eval/purple_cube/" runs/<run>/logs-rank0.txt
```

COM distance is undefined (`–`) on empty-box samples: with no object there is no center
of mass. This matches the training metric, which skips them, and keeps degenerate
samples from dominating a "worst COM distance" sort.
