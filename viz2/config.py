"""Paths, constants and the sys.path bootstrap for viz2.

Importing this module (directly or via any other viz2 module) puts the repo root on
sys.path, so `from utils.metrics import ...` resolves no matter what cwd viz2 is
launched from. The repo is not pip-installed; imports resolve by path.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# ***** inputs (hardcoded defaults, overridable from __main__) *****

EXPERIMENT_DIR = REPO_ROOT / "experiments" / "experiment-25"
RUNS_DIR = REPO_ROOT / "runs"
PORT = 8503  # 8502 is taken by the other explorer

# ***** dataset shape *****

MASK_H, MASK_W = 20, 40
N_SAMPLES = 1024
GT_MASK_REL = f"image/05_downsampled_smask_{MASK_H}h_{MASK_W}w.npy"

# The cropped frame is the one the masks were derived from: 02_smask.npy is (309,679),
# pixel-aligned to it, and the 20x40 target is an exact box-downsample of that (verified
# corr=1.0000). Used as the backdrop under every mask so predictions sit in the real
# scene. 05_overhead_speaker.png is NOT interchangeable -- it adds padding for the
# speaker icon, so a mask drawn over it would be offset.
BACKDROP_REL = "image/01_cropped.png"
OVERHEAD_REL = "image/05_overhead_speaker.png"   # detail modal only (shows the speaker)

# Per-sample extras shown in the detail modal. laser index 50 / x axis is
# DEFAULT_RECOVERY_LASER_IDX in src/data/vibrate.py, but the files are globbed rather
# than hardcoded so a different recovery laser still resolves.
VIBRATION_GLOB = {
    "spectrogram": "vibration/05_spectrogram_laser*_*.png",
    "fft": "vibration/03_fft_laser*_*.png",
}
AUDIO_REL = {"original": "audio.wav", "recovered": "recovered_audio.wav"}

# Stale Windows paths (D:\...) recorded at capture time. Dropped when metadata is
# parsed so they can never leak into a route; all paths are rebuilt from EXPERIMENT_DIR.
STALE_METADATA_KEYS = {"output_dir", "sample_dir", "audio_dir"}

# ***** predictions *****

# Written by OutputSaver (src/model/callbacks.py). Older run eras used `outputs/` and
# `forward_outputs/` with incompatible schemas; those are rejected by the compat scan.
OUTPUTS_SUBDIR = "outputs_history"

# Eval-split directory names identify which dataset a run was trained on. Sample ids
# collide across experiments, so joining a cylinder/bullet run against experiment-25
# ground truth would silently produce meaningless metrics.
EXP25_EVAL_SPLITS = {
    "purple_cube", "purple_cube_speaker",
    "green_cube", "green_cube_speaker",
    "purple_green_cubes", "purple_green_cubes_speaker",
}

N_DEFAULT_RUNS = 3  # auto-loaded on first open, most recently modified first

# The runs directory is re-scanned at most this often, so runs that appear or keep
# training while viz2 is open show up without a restart. A scan is ~0.15s.
RESCAN_SECONDS = 10.0

# Run status is inferred from logs-rank0.txt: a clean shutdown prints the memory line,
# a crash leaves a traceback, and anything else still being written is training.
RUN_LOG = "logs-rank0.txt"
LOG_TAIL_BYTES = 4096
RUNNING_MAX_AGE = 600.0          # log silent longer than this is no longer "running"
CLEAN_EXIT_MARKER = "before close"
# Only a real traceback counts. Bare "Error:"/"Exception" appear in ordinary log lines
# (warnings, retried steps) and would label healthy runs as crashed.
CRASH_MARKERS = ("Traceback (most recent call last)",)

# ***** speaker positions *****

# Copied verbatim from SPEAKER_POSITION in src/data/image.py (inside draw_speaker).
# That module is NOT imported: it does `import modal` at top level.
# Values are (x_frac, y_frac) of the box area with y=0 at the BOTTOM -- draw_speaker
# flips it via `int((1 - y_frac) * H)`. Because these are the same constants that
# generated 05_overhead_speaker.png, the sidebar diagram agrees with the photos.
SPEAKER_POSITION = {
    1: (1, 0), 2: (1, 0.7), 3: (0.8, 1), 4: (0.6, 1),
    5: (0.4, 1), 6: (0.2, 1), 7: (0, 0.7), 8: (0, 0),
}

# ***** rendering *****

UPSCALE = 12  # 20x40 -> 480x240; nearest-neighbour so hover cells stay exact
EPSILON = 1e-6

# Mask images are served `immutable` for a year, so browsers never revalidate them.
# Deriving the version from render.py's mtime means any change to how a mask is drawn
# automatically invalidates the cached images -- no manual bump to forget.
RENDER_VERSION = int((Path(__file__).parent / "render.py").stat().st_mtime)
