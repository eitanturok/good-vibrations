"""Paths, constants and the sys.path bootstrap for viz.

Importing this module (directly or via any other viz module) puts the repo root on
sys.path, so `from utils.metrics import ...` resolves no matter what cwd viz is
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
    """Point viz at a different target grid, before anything reads the constants.

    Layout paths embed {h}/{w}, and Layout instances are built after this runs, so the
    ground-truth filename follows automatically.
    """
    global MASK_H, MASK_W
    MASK_H, MASK_W = int(h), int(w)


# How a downsampled target is named, with the numeric prefix left OPEN. The size is the
# file's identity; the prefix is an artifact of which export wrote it and varies both
# across datasets and, on experiment-25, between two sizes of the SAME dataset. Every
# lookup -- discovery (mask_shapes), layout detection, and loading (masks_at) -- goes
# through this one pattern, so they cannot disagree about which sizes exist.
GT_MASK_GLOB = "*_downsampled_smask_{h}h_{w}w.npy"


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
        for f in (d / "image").glob(GT_MASK_GLOB.format(h="*", w="*")):
            m = re.search(r"_(\d+)h_(\d+)w\.npy$", f.name)
            if m:
                out.add((int(m.group(1)), int(m.group(2))))
        if out:
            break
    return sorted(out)


def usable_mask_shapes(samples_dir) -> list[tuple[int, int]]:
    """mask_shapes(), minus sizes with no usable masks, best first.

    A dataset can ship a size that was written but never properly filled in: the
    gastronorm 20x40 masks are non-zero yet carry ~300x less mass than the 16x16 and
    30x30 masks of the same scenes, so anything scored against them reads as ~0 IoU.
    Ranking on COVERAGE (mean mask value) rather than a non-empty count is what separates
    those from a real target -- by count all three sizes tie, since the same 2952 samples
    are non-empty at every size.
    """
    import numpy as np
    try:
        dirs = sorted(p for p in samples_dir.iterdir() if p.is_dir())
    except OSError:
        return []
    # Spread the probes over the WHOLE dataset rather than taking the first N. Empty-box
    # samples are not scattered at random -- the gastronorm captures open with a run of
    # them, so probing the head finds every size empty and declares the dataset unusable.
    step = max(1, len(dirs) // 60)
    probes = dirs[::step][:60]
    scored = []
    for h, w in mask_shapes(samples_dir):
        cov = []
        for d in probes:
            for f in (d / "image").glob(GT_MASK_GLOB.format(h=h, w=w)):
                try:
                    cov.append(float(np.asarray(np.load(f), dtype=np.float64).mean()))
                except Exception:
                    pass
        # A real target covers ~1% of the frame. The threshold only has to separate that
        # from the degenerate sizes, which sit three orders of magnitude below it.
        if cov and float(np.mean(cov)) > 1e-3:
            scored.append((h, w))
    # Finest grid first. Coverage decides only WHICH sizes are usable, never their order:
    # the same scenes at two resolutions have near-identical coverage (16x16 and 30x30
    # here agree to six decimals), so ranking on it would pick a default out of
    # floating-point noise. Resolution is a real, stable preference.
    scored.sort(key=lambda t: -(t[0] * t[1]))
    return scored


# Sample-directory layout is NOT fixed across experiments. Both eras keep their per-sample
# images in `image/`, but the filenames inside differ: experiment-25 uses a `05_` mask
# prefix, the gastronorm experiments a `04_` prefix and a differently named crop. Nothing
# in the files themselves declares which era they belong to, so `Layout.detect` probes a
# real sample directory rather than making viz depend on one hardcoded set of names.
#
# Every entry is a path RELATIVE TO A SAMPLE DIR. `{h}`/`{w}` are filled with MASK_H and
# MASK_W so the mask name follows the configured target shape.
#
# A layout entry names a FILENAME SCHEME, not a dataset -- `dataset` carries the data
# identity separately, and it is what a run's `family` is reported as. Two layouts sharing
# a dataset is normal: the same capture read at two mask sizes is one dataset, and folding
# the two apart would label otherwise-identical runs as belonging to different families.
#
# `gt_mask` is now only a label for error messages: masks are found by GT_MASK_GLOB, which
# ignores the numeric prefix. That prefix is not stable enough to identify anything --
# experiment-25 writes 20x40 as both `04_` and `06_` but 30x30 as `04_` only, which is why
# a second "same dataset at another size" entry used to be needed here and no longer is.
LAYOUTS = {
    # The experiment-25 capture currently on disk. Same underlying data as "experiment-25"
    # below, re-exported with shifted prefixes and the cropped overhead as backdrop rather
    # than `01_cropped.png`. Nothing in the files declares the era, hence a separate entry
    # probed ahead of the older one.
    "experiment-25-v2": {
        "dataset": "experiment-25",
        "image_dir": "image",
        "gt_mask": "image/06_downsampled_smask_{h}h_{w}w.npy",
        "backdrop": "image/02_cropped_overhead.png",
        "overhead": "image/06_overhead_speaker.png",  # detail modal (shows the speaker)
        "audio": {"original": "audio.wav", "recovered": "recovered_audio.wav"},
    },
    "experiment-25": {
        "dataset": "experiment-25",
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
        "dataset": "gastronorm",
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

# Probed in order; the first layout matching a sample dir wins. Masks are matched by glob,
# so ORDER CANNOT BREAK TIES between datasets any more -- `overhead` and `backdrop` do.
LAYOUT_ORDER = ["experiment-25-v2", "experiment-25", "gastronorm"]

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
        self._gt_mask_spec = spec["gt_mask"]
        # Which data this is, independent of the filename scheme `name` identifies. Two
        # layouts can share a dataset (the same capture at two mask sizes); run
        # compatibility keys on this, never on `name`.
        self.dataset = spec.get("dataset", name)
        self.image_dir = spec["image_dir"]
        self.gt_mask = spec["gt_mask"].format(**fmt)
        self.backdrop = spec["backdrop"].format(**fmt)
        self.overhead = spec["overhead"].format(**fmt)
        self.audio = dict(spec["audio"])

    def gt_mask_for(self, h: int, w: int) -> str:
        """The ground-truth filename at an arbitrary grid, not just the default one.

        A dataset ships several downsample sizes side by side and different runs are
        trained on different ones, so viz has to be able to name any of them -- not only
        the size that happened to be current when this Layout was built.
        """
        return self._gt_mask_spec.format(h=h, w=w)

    def gt_mask_glob(self, h: int, w: int) -> str:
        """The search pattern for the target at (h,w), relative to a sample dir."""
        return f"{self.image_dir}/{GT_MASK_GLOB.format(h=h, w=w)}"

    def resolve_gt_mask(self, sample_dir, h: int, w: int):
        """The ground-truth file for (h,w) in one sample dir, or None.

        Globbed rather than templated because THE SIZE IN THE NAME IS THE IDENTITY AND THE
        NUMERIC PREFIX IS NOT. One dataset can write the same size under two prefixes
        (experiment-25 ships 20x40 as both `04_` and `06_`) and ship another size under
        only one of them (30x30 is `04_`-only). Formatting the detected layout's prefix
        into every size then names a file that does not exist: has_shape() globs and says
        30x30 is available, masks_at() templated and found nothing, so the run classified
        as compatible and then rendered as an empty column.

        sorted()[0] makes the choice stable across samples. The duplicated 04_/06_ 20x40
        masks agree to 0.002, so either is correct -- but picking a different one per
        sample would stack rows from two exports into a single array.
        """
        hits = sorted(sample_dir.glob(self.gt_mask_glob(h, w)))
        return hits[0] if hits else None

    @classmethod
    def detect(cls, samples_dir) -> "Layout":
        """Pick the layout whose ground-truth mask actually exists on disk.

        Probes several sample dirs, not one: a partially-written capture can be missing
        its mask entirely (000009 in the gastronorm run has no downsampled mask), and
        probing only the first directory would then fall through to the wrong layout --
        or to none at all -- for an otherwise healthy 3000-sample dataset.

        The mask NEVER identifies a layout on its own: it is matched by glob, so the
        numeric prefix -- the only thing that used to distinguish `experiment-25-v2` from
        `gastronorm` -- is deliberately ignored. Identity therefore rests entirely on the
        `overhead` and `backdrop` files, which do differ (06_overhead_speaker.png is on
        experiment-25 and absent from gastronorm). Getting this wrong is expensive: the
        wrong layout loads the wrong ground-truth masks, so every run scores against
        targets from another capture and the whole table reports silently wrong numbers.

        The loose pass exists for a capture missing its overhead file, and now demands the
        backdrop rather than matching on the mask alone -- a mask-only match would accept
        the first layout in LAYOUT_ORDER for literally any dataset. It also says so out
        loud, because a misdetection otherwise surfaces three steps later as "every run is
        incompatible" with nothing pointing back to here.
        """
        try:
            probes = sorted(p for p in samples_dir.iterdir() if p.is_dir())[:25]
        except OSError:
            probes = []
        layouts = [cls(name, LAYOUTS[name]) for name in LAYOUT_ORDER]
        for strict in (True, False):
            for layout in layouts:
                for d in probes:
                    if layout.resolve_gt_mask(d, MASK_H, MASK_W) is None:
                        continue
                    if not (d / layout.backdrop).exists():
                        continue
                    if strict and not (d / layout.overhead).exists():
                        continue
                    if not strict:
                        print(f"[viz] layout '{layout.name}' matched WITHOUT its overhead "
                              f"file ({layout.overhead}); dataset identity is a guess, so "
                              f"runs may be scored against the wrong ground truth.")
                    return layout
        known = ", ".join(LAYOUT_ORDER)
        raise SystemExit(
            f"[viz] no known sample layout under {samples_dir}.\n"
            f"       tried: {known}. Expected a mask matching "
            + GT_MASK_GLOB.format(h=MASK_H, w=MASK_W)
            + " plus that layout's backdrop.\n"
              "       Add a new entry to LAYOUTS in viz/config.py if this is a new "
              "dataset format."
        )

# ***** predictions *****

# Written by OutputSaver (src/model/callbacks.py). Older run eras used `outputs/` and
# `forward_outputs/` with incompatible schemas; those are rejected by the compat scan.
OUTPUTS_SUBDIR = "outputs_history"

# There is deliberately no eval-split whitelist here. Dataset identity is checked in
# data._classify by sample-id overlap against the loaded ground truth, which tests the
# actual invariant (do these predictions refer to samples in this dataset?) instead of a
# naming convention. A whitelist used to live here and rejected every objcount-* run as
# "different dataset" because gastronorm_train1_eval2 names its evaluators 1-obj/2-obj
# rather than 1-cube/2-cubes -- same capture, same grid, same ids, different slicing.
# Runs that slice one dataset differently are meant to sit in the same table.

N_DEFAULT_RUNS = 3  # auto-loaded on first open, most recently modified first

# The runs directory is re-scanned at most this often, so runs that appear or keep
# training while viz is open show up without a restart. A scan is ~0.15s.
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

# Height, in pixels, of a rendered mask: the canvas is `h * UPSCALE` tall and as wide as
# the scene's aspect makes it (see render.canvas_size). Nearest-neighbour, so hover cells
# stay exact.
UPSCALE = 12
# Cap on the backdrop JPEG's height. The photo is served at its NATIVE resolution up to
# this, never upscaled past it -- the overhead frame is the only real detail on screen and
# the mask is a coarse grid drawn on top, so the photo must stay sharp while the mask is
# what gets stretched. Sizing this to the mask grid made a 1337x1110 capture ship at
# 434x360 and look badly blurred. Independent of any grid: one image serves every column.
BACKDROP_MAX_PX = 1600
EPSILON = 1e-6

# Mask images are served `immutable` for a year, so browsers never revalidate them.
# Deriving the version from render.py's mtime means any change to how a mask is drawn
# automatically invalidates the cached images -- no manual bump to forget.
RENDER_VERSION = int((Path(__file__).parent / "render.py").stat().st_mtime)
