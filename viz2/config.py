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

# The target grid. A dataset may ship masks at several downsample sizes side by side
# (the gastronorm captures write both 20x40 and 30x30), and a run is only comparable
# against the size it was trained on -- so this is overridable from the command line
# (--mask 30x30) rather than a fixed property of the code. set_mask_shape rebinds it.
MASK_H, MASK_W = 20, 40
N_SAMPLES = 1024


def set_mask_shape(h: int, w: int) -> None:
    """Point viz2 at a different target grid, before anything reads the constants.

    Layout paths embed {h}/{w}, and Layout instances are built after this runs, so the
    ground-truth filename follows automatically.
    """
    global MASK_H, MASK_W
    MASK_H, MASK_W = int(h), int(w)


def mask_shapes(samples_dir) -> list[tuple[int, int]]:
    """Downsampled-mask sizes actually present, newest-layout first, for error messages
    and for --mask's default. Reads one sample dir; sizes are uniform across a dataset."""
    import re
    out = set()
    try:
        probes = sorted(p for p in samples_dir.iterdir() if p.is_dir())[:25]
    except OSError:
        return []
    for d in probes:
        for f in (d / "image").glob("*_downsampled_smask_*h_*w.npy"):
            m = re.search(r"_(\d+)h_(\d+)w\.npy$", f.name)
            if m:
                out.add((int(m.group(1)), int(m.group(2))))
        if out:
            break
    return sorted(out)

# Sample-directory layout is NOT fixed across experiments. Both eras keep their per-sample
# images in `image/`, but the filenames inside differ: experiment-25 uses a `05_` mask
# prefix, the gastronorm experiments a `04_` prefix and a differently named crop. Nothing
# in the files themselves declares which era they belong to, so `Layout.detect` probes a
# real sample directory rather than making viz2 depend on one hardcoded set of names.
#
# Every entry is a path RELATIVE TO A SAMPLE DIR. `{h}`/`{w}` are filled with MASK_H and
# MASK_W so the mask name follows the configured target shape.
LAYOUTS = {
    "experiment-25": {
        "image_dir": "image",
        "gt_mask": "image/05_downsampled_smask_{h}h_{w}w.npy",
        # The cropped frame is the one the masks were derived from: 02_smask.npy is
        # (309,679), pixel-aligned to it, and the 20x40 target is an exact box-downsample
        # of that (verified corr=1.0000). Used as the backdrop under every mask so
        # predictions sit in the real scene. 05_overhead_speaker.png is NOT
        # interchangeable -- it adds padding for the speaker icon, so a mask drawn over
        # it would be offset.
        "backdrop": "image/01_cropped.png",
        "overhead": "image/05_overhead_speaker.png",  # detail modal (shows the speaker)
        "audio": {"original": "audio.wav", "recovered": "recovered_audio.wav"},
    },
    "gastronorm": {
        "image_dir": "image",
        "gt_mask": "image/04_downsampled_smask_{h}h_{w}w.npy",
        "backdrop": "image/02_cropped_overhead.png",
        # No speaker-annotated overhead is rendered by this pipeline. The cropped frame
        # stands in so the detail modal still shows the scene; the modal degrades to
        # "not generated" for anything that genuinely has no file.
        "overhead": "image/02_cropped_overhead.png",
        # These captures ship no source audio -- `audio/` exists but is empty. Only the
        # recovered waveform is present.
        "audio": {"recovered": "recovered_audio.wav"},
    },
}

# Probed in order; the first layout whose mask file exists in a sample dir wins.
LAYOUT_ORDER = ["experiment-25", "gastronorm"]

# Per-sample extras shown in the detail modal. The recovery laser index varies by
# experiment (50 on experiment-25, 55 on gastronorm), and gastronorm drops the axis
# suffix on some files, so these are globbed rather than hardcoded.
VIBRATION_GLOB = {
    "spectrogram": "vibration/05_spectrogram_laser*.png",
    "fft": "vibration/03_fft_laser*.png",
}

# Stale Windows paths (D:\... , C:\...) recorded at capture time. Dropped when metadata
# is parsed so they can never leak into a route; all paths are rebuilt from
# EXPERIMENT_DIR. `experiment_dir` is the gastronorm-era spelling of `output_dir`.
STALE_METADATA_KEYS = {"output_dir", "sample_dir", "audio_dir", "experiment_dir"}


class Layout:
    """Which per-sample filenames an experiment uses, resolved once at startup.

    Holds only relative paths; joining against a sample dir stays the caller's job, so
    there is still exactly one place (Registry.sample_dir) that touches the filesystem
    with a user-supplied id.
    """

    def __init__(self, name: str, spec: dict):
        fmt = {"h": MASK_H, "w": MASK_W}
        self.name = name
        self.image_dir = spec["image_dir"]
        self.gt_mask = spec["gt_mask"].format(**fmt)
        self.backdrop = spec["backdrop"].format(**fmt)
        self.overhead = spec["overhead"].format(**fmt)
        self.audio = dict(spec["audio"])

    @classmethod
    def detect(cls, samples_dir) -> "Layout":
        """Pick the layout whose ground-truth mask actually exists on disk.

        Probes several sample dirs, not one: a partially-written capture can be missing
        its mask entirely (000009 in the gastronorm run has no downsampled mask), and
        probing only the first directory would then fall through to the wrong layout --
        or to none at all -- for an otherwise healthy 3000-sample dataset.
        """
        try:
            probes = sorted(p for p in samples_dir.iterdir() if p.is_dir())[:25]
        except OSError:
            probes = []
        for name in LAYOUT_ORDER:
            layout = cls(name, LAYOUTS[name])
            if any((d / layout.gt_mask).exists() for d in probes):
                return layout
        known = ", ".join(LAYOUT_ORDER)
        raise SystemExit(
            f"[viz2] no known sample layout under {samples_dir}.\n"
            f"       tried: {known}. Expected one of "
            + ", ".join(cls(n, LAYOUTS[n]).gt_mask for n in LAYOUT_ORDER)
            + "\n       Add a new entry to LAYOUTS in viz2/config.py if this is a new "
              "dataset format."
        )

# ***** predictions *****

# Written by OutputSaver (src/model/callbacks.py). Older run eras used `outputs/` and
# `forward_outputs/` with incompatible schemas; those are rejected by the compat scan.
OUTPUTS_SUBDIR = "outputs_history"

# Eval-split directory names identify which dataset a run was trained on. Sample ids
# collide across experiments, so joining a cylinder/bullet run against experiment-25
# ground truth would silently produce meaningless metrics.
#
# Keyed by layout name: a run is only comparable against the ground truth currently
# loaded, and the layouts are different datasets. A layout absent from this map (or
# mapped to None) accepts any split names and falls back to the sample-id overlap check
# in data._classify, which is weaker but dataset-agnostic.
EVAL_SPLITS = {
    "experiment-25": {
        "purple_cube", "purple_cube_speaker",
        "green_cube", "green_cube_speaker",
        "purple_green_cubes", "purple_green_cubes_speaker",
    },
    "gastronorm": {
        "1-cube", "1-cube-speaker", "2-cubes", "2-cubes-speaker", "3-cubes", "red-cube",
    },
}

N_DEFAULT_RUNS = 3  # auto-loaded on first open, most recently modified first

# The runs directory is re-scanned at most this often, so runs that appear or keep
# training while viz2 is open show up without a restart. A scan is ~0.15s.
RESCAN_SECONDS = 10.0

# How many scrubbed-epoch RunData objects to keep. Each is ~5MB, so this bounds the
# epoch slider's memory; the latest-epoch entry per run is never evicted.
MAX_EPOCH_CACHE = 24

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
