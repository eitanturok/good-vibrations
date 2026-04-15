# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Good Vibrations is a scientific computing project for analyzing surface vibrations using high-speed camera data and laser speckle patterns. The pipeline processes video recordings of laser points on vibrating surfaces to extract and analyze sub-pixel motion signals.

## Architecture

### Pipeline Flow (Jupyter Notebooks in `src/`)

1. **00_chirp.ipynb** - Generate audio test signals (chirps, tones) as WAV files
2. **01_record.ipynb** - Record high-speed camera footage of laser speckle patterns
3. **02_analyze.ipynb** - Load shift data, compute FFT spectra, visualize results
4. **03_mark_visualize.ipynb** - Mark and visualize regions of interest
5. **04_dimension_reduction.ipynb** - Dimensionality reduction on vibration data
6. **05_visualize_shifts.ipynb** - Visualize computed shifts over time
7. **06_shift_fft.ipynb** - FFT analysis of shift data

### Core Libraries

- **`lib/`** - Image and video processing utilities
  - `image_processing.py` - Gamma correction, debayering, 16-bit to 8-bit conversion
  - `opencv_video_utils.py` - `videoPlayer` class for interactive video playback and export

- **`utils/recover_core_lib.py`** - Core vibration recovery algorithms:
  - Phase correlation (CPU and GPU/CuPy versions) for sub-pixel shift detection
  - Lucas-Kanade iterative refinement for sub-pixel accuracy
  - `VibrationViewer` class for visualization
  - Signal filtering and WAV export utilities

### Data Format

- Shift data shape: `(N_sensors, N_frames, 2)` — X and Y displacement
- Typical setup: 10x10 grid of laser points (100 sensors), 5000 FPS camera
- Metadata in `metadata.npz` (`camera_FPS`, `exposure`, `ROIs`)
- Recovery results in `RECOVERY.npz` (`all_shifts` array)

## Development

```bash
python -m venv .venv && source .venv/bin/activate
jupyter notebook src/
```

**Dependencies:** NumPy, SciPy, OpenCV, CuPy (optional GPU), Matplotlib

**Key concepts:** Phase correlation (FFT-based sub-pixel translation), Hann windowing, Lucas-Kanade refinement, parabolic interpolation for peak localization.

---

## Tools

### `smart_sbatch.py` — Submit Cluster Jobs

**You (the AI) can and should run these commands directly using the Bash tool.** Do not tell the user to run them. SSH into the cluster, pull the latest code, and submit the job yourself. `squeue`, `sinfo`, and `sbatch` only exist on the cluster — that is exactly why you SSH in first.

**IMPORTANT:** Any time code is changed locally, you must `git push` from the local machine and then `git pull` on the cluster before submitting a job. Never submit without pulling first — the cluster will silently run outdated code otherwise.

**Cluster:** `mcluster11.wisdom.weizmann.ac.il`, code at `mark_sheinin_lab/code/eitan/good-vibrations/`

```bash
ssh ethantu@mcluster11.wisdom.weizmann.ac.il "cd mark_sheinin_lab/code/eitan/good-vibrations && git pull && python smart_sbatch.py --job-name my-job --loss focal --gamma 10 --speakers '[1,0,0,0]'"
```

To tail logs after submitting, SSH in again and run:
```bash
ssh ethantu@mcluster11.wisdom.weizmann.ac.il "tail -f mark_sheinin_lab/code/eitan/good-vibrations/logs/<timestamp>/out.log"
```

**Dry run** (prints the sbatch script, does not submit):
```bash
python smart_sbatch.py --dry-run --job-name my-job --loss focal --gamma 10
```

`smart_sbatch.py` auto-picks the best available partition (`normal.q`: 6 slots/6hr; `long.q`: 2 slots/12hr) and GPU (L40S 48GB → RTX 8000 48GB → A10 24GB → RTX 6000 24GB). Writes `job.sh` under `logs/<timestamp>/` and submits via `sbatch`. All flags after `--job-name` pass through directly to `src/model.py`.

**Key `src/model.py` arguments:**

| Flag | Default | Notes |
|---|---|---|
| `--loss` | `dice` | choices: `dice`, `focal`, `bce`, `mse`, `tversky` |
| `--gamma` | `2.0` | focal loss gamma (standard default; gamma=10 crushes gradients at init) |
| `--alpha` | `0.9` | loss weight |
| `--beta` | `0.5` | loss weight |
| `--delta` | `0.4` | loss weight |
| `--speakers` | `None` | JSON list e.g. `'[1,0,0,0]'` or `'[[1,0,0,0],[0,1,0,0]]'` |
| `--n-objects` | `[1]` | e.g. `--n-objects 1 2` |
| `--decoder` | `mlp` | choices: `mlp`, `cnn`, `cross_attn`, `pool` |
| `--d-model` | `128` | model width |
| `--batch-size` | `256` | training batch size |
| `--lr` | `1e-4` | learning rate |
| `--max-duration` | `10_000ep` | total training duration |
| `--eval-interval` | `10ep` | how often to run eval |
| `--signal-is` | `magnitude` | input representation |
| `--normalize` | `None` | normalization strategy |
| `--focal` | `0` | 0/1 flag (deprecated; use `--loss focal`) |
| `--rope` | `1` | rotary positional encoding |
| `--augment` | `1` | data augmentation |
| `--seed` | `42` | random seed |

---

## Weights & Biases

### Accessing W&B Data

Two helper functions in `src/helpers.py` let you fetch logged data programmatically:

- **`fetch_wandb_history(run_id, keys, entity, project)`** — returns a list of history dicts sorted by `_step`. Use for loss curves and metrics.
- **`fetch_wandb_images(run_id, split, epoch, key, download_dir, entity, project)`** — downloads `mask_viz` images for a given run/epoch and returns local `Path`s. Pass those paths to Claude's `Read` tool to view them visually.

```python
from helpers import fetch_wandb_history, fetch_wandb_images

# Get loss and IoU for every step
rows = fetch_wandb_history('s3pqt79j', keys=['_step', 'loss/train/total', 'metrics/eval/mask/iou'])

# Download eval mask visualizations at epoch 10
paths = fetch_wandb_images('s3pqt79j', split='eval', epoch=10)
```

### What Is Logged

- **`loss/train/total`** — training loss per batch step.
- **`metrics/{split}/mask/ce|dice|iou`** and **`metrics/{split}/position/x|y/acc|ce|rmse`** — eval and train metrics logged once per epoch.
- **`mask_viz/{split}/prob`** — the main visualization. Each image is a side-by-side panel:
  - **Left:** ground truth segmentation mask (grayscale, continuous values in [0, 1] — each pixel represents the fraction of that grid cell covered by the object).
  - **Right:** predicted mask probabilities after sigmoid (hot colormap, red = high probability).
  - Multiple samples are logged per epoch so you can see how predictions vary across different object positions.
- **`mask_viz/{split}/thresh{t}`** — same layout but the predicted mask is binarized at threshold `t`.

### Finding the Right Run

**Always check the `WandBLogger` call in `src/model.py`** for the current `project` and `group` — these can change between experiments. Filter runs by group to narrow down candidates:

```python
import wandb
api = wandb.Api()
runs = api.runs("eturok/good-vibrations", filters={"group": "loss"})
for r in runs:
    print(r.id, r.name, r.state)
```

---

## Deep Learning: Vibration-to-Segmentation Experiment

### Goal

Train a neural network that takes laser speckle vibration signals as input and outputs a segmentation mask showing the shape and location of an object hidden inside a cardboard box.

### Physical Setup

- A cardboard box contains an object (currently two cubes) at one of 80 positions on an 11x12 grid.
- Four loudspeakers surround the box, each at a different angle/distance/amplitude/phase, exciting the box differently — encodes directional information about the object.
- A 10x10 laser grid (100 lasers) shines on the *side* of the box. As the box vibrates, each laser speckle shifts in X and Y on the camera sensor, recovered via phase correlation + Lucas-Kanade.
- Ground truth: an overhead photo of the object is taken and discretized into a grid of smaller squares by averaging pixels within each square. Each square gets a continuous value in [0, 1] representing the fraction covered by the object. Output shape: **40x20** (height x width).

### Data Dimensions

| Quantity | Value | Notes |
|---|---|---|
| Samples | 80 | Same object, 80 positions |
| Lasers | 100 (10x10 grid) | 2D grid on box side |
| Time steps | `T = seconds × FPS` | FPS ~5000 |
| Frequencies (FFT) | ~1100 | Per laser |
| Shift channels | 2 | X and Y displacement |
| Output mask | 40×20 | Continuous values in [0, 1] |

**Raw input:** `(100, T, 2)` — **FFT input:** `(100, F, 2)` — **Speakers:** 4, treat as 4 views (320 samples) or 4 input channels.

### Key Modeling Challenges

1. **Limited data (80 samples)** — heavily regularize; use strong inductive biases; patch-based training multiplies effective samples.
2. **Two coordinate systems** — laser grid is on the box side; mask is overhead. Cross-attention bridges them.
3. **FFT input** — objects act as mechanical filters. Per-frequency and broadband analysis both matter.
4. **(dx, dy) channels** — amplitude `sqrt(dx²+dy²)` and phase `atan2(dy,dx)` may be more informative than raw dx/dy.
5. **Speaker diversity** — 4 speakers = 4 views of the same object; use as data augmentation or fusion.
6. **Positional encoding** — laser 2D positions should be encoded so the model understands spatial structure.

### Recommended Architecture

A **cross-modal transformer**:
1. **Frequency encoder** (per laser): encode `(F, 2)` FFT into a fixed embedding (1D CNN or linear + freq positional encoding).
2. **Spatial laser encoder**: 100 laser tokens in a 10x10 grid with 2D positional encoding + self-attention.
3. **Cross-attention decoder**: 800 output tokens (40×20 grid positions) attend to all 100 laser tokens.
4. **Segmentation head**: small MLP per token → scalar in [0, 1].

**Simpler baseline:** flatten FFT to `(100, F*2)`, apply small MLP/linear model, reshape to 40×20.

**Related architectures:** SAM (prompt-to-mask cross-attention), ViT (patch tokenization), DETR (cross-attention decoder), FNO (Fourier neural operator), CrossViT.

### Design Principles

- Prefer fewer parameters with strong inductive biases.
- Patch-based tokenization of laser grid for data augmentation.
- Treat 4 speakers as 4 augmented views (early or late fusion).
- Frequency masking as regularization.
- **Loss:** MSE or BCE on soft mask; Dice loss if object is small.
- **Evaluation:** IoU (threshold at 0.5) and pixel-wise MSE on soft mask.
