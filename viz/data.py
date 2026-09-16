"""Loading and indexing of ground truth and per-run predictions.

Nothing here runs inference: ground-truth masks are per-sample .npy files on disk and
predicted masks were already dumped by the OutputSaver callback during training. The
whole job is to join them on sample_id and compute per-sample metrics.

Sample ids are NOT row indices into these arrays -- see SPEC.md for which of the two
every field, argument and map is keyed by.
"""

import json
import os
import time
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch

from viz import config
import torch.nn.functional as _F
from utils.metrics import center_of_mass, object_centroids, soft_iou, mass_error, contour_f, localization, LOC_KEYS

METRIC_KEYS = ('bce', 'iou', *LOC_KEYS, 'contour', 'mass')

# ***** ground truth *****


def merge_metadata(path: Path) -> dict:
    """metadata.jsonl stores ONE KEY PER LINE, so the lines must be merged into a
    single dict. Stale Windows paths are dropped here so they can never reach a route."""
    meta = {}
    for line in path.read_text().splitlines():
        if line.strip():
            meta.update(json.loads(line))
    return {k: v for k, v in meta.items() if k not in config.STALE_METADATA_KEYS}


def parse_com(v) -> list[float]:
    """A center of mass as [y, x], from any of the shapes metadata.jsonl stores it in.

    experiment-25 writes a JSON list. The gastronorm pipeline writes `str(ndarray)` --
    "[603.12363008 901.16480443]" -- which is whitespace-separated and NOT valid JSON, so
    it arrives here as a plain string. Returning the [-1,-1] "no position" sentinel on
    anything unparseable keeps a malformed record from taking down the whole load.
    """
    if isinstance(v, str):
        v = v.replace("[", " ").replace("]", " ").replace(",", " ").split()
    if isinstance(v, (list, tuple)):
        try:
            flat = np.asarray(v, dtype=np.float64).reshape(-1)
        except (ValueError, TypeError):
            return [-1.0, -1.0]
        if flat.size >= 2:
            return [float(flat[0]), float(flat[1])]
    return [-1.0, -1.0]


@dataclass
class GtIndex:
    sample_ids: list[str]        # zero-padded "000000"..; ids need NOT start at 0
    masks: np.ndarray            # (N,20,40) float32, contiguous
    meta: list[dict]
    com_gt: np.ndarray           # (N,2) grid-space COM of the target mask
    # Per-OBJECT centroids: [(K_i,2)] in grid coords, one array per row. Distinct from
    # com_gt, which is ONE probability-weighted point for the whole mask and therefore
    # lands between the cubes on a multi-object sample. These are what `localization`
    # matches on, so a crosshair drawn from them agrees with the metric beside it.
    obj_com: list
    avg_com: np.ndarray          # (N,2) full-res image coords, for the position scatter
    layout: config.Layout        # which per-sample filenames this experiment uses
    # sample id -> row. Ids are NOT an identity map into the arrays: the gastronorm
    # dataset starts at 000009, and any dataset can be missing a sample whose mask was
    # never written. Every id->row lookup must go through this.
    row_of: dict[int, int] = field(default_factory=dict)
    # Ground truth at OTHER grid sizes, loaded on demand: {(h,w): (N,h,w)}. Runs are
    # trained at different resolutions and each must be scored against its own target,
    # so the table can hold a 16x16 column beside a 30x30 one. Same rows, same order as
    # `masks`, so gt.row_of indexes every entry here too.
    # None is cached for a shape this dataset has no masks at, so a missing size is
    # probed once rather than re-walking 3000 sample dirs on every request.
    by_shape: dict[tuple[int, int], np.ndarray | None] = field(default_factory=dict)
    experiment_dir: Path | None = None
    _shapes: list[tuple[int, int]] | None = None   # memo for disk_shapes()
    _com: dict[tuple[int, int], np.ndarray | None] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.sample_ids)

    def masks_at(self, shape: tuple[int, int]) -> np.ndarray | None:
        """Targets at `shape`, or None if this dataset has no usable masks that size.

        Rows are aligned with `sample_ids`: a sample missing its mask at this size gets
        zeros rather than being dropped, because dropping it would shift every later row
        out of step with `masks` and silently mis-pair predictions with targets.
        """
        if shape == (self.masks.shape[1], self.masks.shape[2]):
            return self.masks
        if shape in self.by_shape:
            return self.by_shape[shape]
        if self.experiment_dir is None:
            return None
        h, w = shape
        out = np.zeros((len(self.sample_ids), h, w), dtype=np.float32)
        found = False
        for i, sid in enumerate(self.sample_ids):
            # Globbed, exactly as disk_shapes/has_shape discover sizes -- templating the
            # detected layout's prefix here is what let the two disagree, so a run could
            # pass compatibility and then find no targets at all.
            p = self.layout.resolve_gt_mask(
                self.experiment_dir / "samples" / sid, h, w)
            if p is None:
                continue
            try:
                m = np.asarray(np.load(p), dtype=np.float32)
            except Exception:
                continue
            if m.shape == shape:
                out[i] = m
                found = True
        self.by_shape[shape] = out if found else None
        return self.by_shape[shape]

    def com_at(self, shape: tuple[int, int]) -> np.ndarray | None:
        """Target centers of mass in `shape`'s grid coordinates, memoized.

        com_gt is the primary shape only; a run at another grid must be compared against
        targets measured on ITS grid, or the comparison mixes two coordinate systems.
        """
        if shape not in self._com:
            m = self.masks_at(shape)
            self._com[shape] = (None if m is None
                                else np.asarray(center_of_mass(m), dtype=np.float64))
        return self._com[shape]

    def disk_shapes(self) -> list[tuple[int, int]]:
        """Every grid size this dataset ships masks at, cheaply and cached.

        Only reads filenames, so it stays usable from _classify's rejection path -- which
        runs once per incompatible run on every rescan, i.e. every 10s.
        """
        if self._shapes is None:
            self._shapes = (
                [(self.masks.shape[1], self.masks.shape[2])]
                if self.experiment_dir is None
                else config.mask_shapes(self.experiment_dir / "samples"))
        return self._shapes

    def has_shape(self, shape) -> bool:
        """Whether targets exist at `shape`, without decoding them."""
        return tuple(shape) in {tuple(s) for s in self.disk_shapes()}


def load_gt(experiment_dir: Path, mask_h: int, mask_w: int) -> GtIndex:
    samples_dir = experiment_dir / "samples"
    layout = config.Layout.detect(samples_dir, mask_h, mask_w)
    sample_dirs = sorted(p for p in samples_dir.iterdir() if p.is_dir())
    ids, masks, meta = [], [], []
    for d in sample_dirs:
        gt = layout.resolve_gt_mask(d, mask_h, mask_w)
        if gt is None:
            continue
        m = np.load(gt)
        # A dataset can carry masks at several downsample sizes side by side (gastronorm
        # writes both 20x40 and 30x30). Only the requested target shape is loadable as
        # ground truth; a mismatch would otherwise fail deep inside np.stack.
        if m.shape != (mask_h, mask_w):
            continue
        ids.append(d.name)
        masks.append(m)
        meta.append(merge_metadata(d / "metadata.jsonl"))
    if not masks:
        # Only reachable via an explicit --mask, or a size neither requested nor usable
        # (see load_experiments' DEFAULT_MASK_SHAPE fallback). Name the sizes that DO
        # exist so the retry is obvious.
        have = ", ".join(f"{a}x{b}" for a, b in config.mask_shapes(samples_dir)) or "none"
        raise SystemExit(
            f"[viz] no {mask_h}x{mask_w} ground-truth masks under "
            f"{samples_dir} (layout '{layout.name}', searched "
            f"{layout.gt_mask_glob(mask_h, mask_w)}).\n"
            f"       available sizes: {have}. Pass one with --mask HxW."
        )
    masks = np.ascontiguousarray(np.stack(masks).astype(np.float32))
    com_gt = np.asarray(center_of_mass(masks), dtype=np.float64)
    avg_com = np.asarray([parse_com(m.get("avg_com")) for m in meta], dtype=np.float64)
    row_of = {int(s): i for i, s in enumerate(ids)}
    return GtIndex(ids, masks, meta, com_gt, object_centroids(masks), avg_com, layout,
                   row_of, experiment_dir=experiment_dir)


# ***** run scanning *****


@dataclass
class RunEntry:
    name: str
    compatible: bool
    reason: str | None = None
    mtime: float = 0.0
    epoch: int | None = None
    eval_splits: list[str] = field(default_factory=list)
    family: str = "unknown"
    status: str = "unknown"     # running | finished | crashed | unknown
    shape: tuple[int, int] | None = None   # grid this run was trained at
    # Which experiments (registry.gts indices) this run was OBSERVED to touch during
    # classification -- not necessarily every experiment it actually predicts, since
    # classification only reads a handful of files (see _probe); used by
    # Registry.defaults() to pick a run that covers each loaded box, not to route any
    # actual row (routing happens per-row in load_run/Registry.route via that row's own
    # `box`, against the FULL run, not this probe-derived set).
    experiments: frozenset = field(default_factory=frozenset)


def _epoch_of(p: Path) -> int:
    return int(p.stem.split("-")[0].removeprefix("ep"))


def _batch_of(p: Path) -> int:
    """Batch index from `ep{E}-ba{B}.pt`. Filenames sort correctly as strings today, but
    only because B is zero-padded; parse it so that stays true if the width changes."""
    return int(p.stem.split("-")[1].removeprefix("ba"))


def _as_int_array(v) -> np.ndarray:
    """info['sample_id'] is a tensor on most runs but a plain list on some."""
    if torch.is_tensor(v):
        v = v.tolist()
    return np.asarray(v, dtype=np.int64)


def _as_box_list(info: dict, n: int) -> list:
    """info['box'] per-row, or `n` Nones on a run saved before that field existed --
    Registry.route falls back to id-overlap for those. Always a plain list, same length
    as the batch, so it zips 1:1 with a sample_id array regardless of source."""
    v = info.get("box")
    return list(v) if v is not None else [None] * n


def run_status(run_dir: Path) -> str:
    """running | finished | crashed | unknown, read from the tail of the training log.

    A clean shutdown prints a memory summary; a crash leaves a traceback. A log that is
    still being written with neither marker belongs to a run that is training now.
    """
    log = run_dir / config.RUN_LOG
    try:
        age = time.time() - log.stat().st_mtime
        with open(log, "rb") as f:
            f.seek(0, os.SEEK_END)
            f.seek(max(0, f.tell() - config.LOG_TAIL_BYTES))
            tail = f.read().decode("utf8", "replace")
    except OSError:
        return "unknown"
    if config.CLEAN_EXIT_MARKER in tail:
        return "finished"
    if any(m in tail for m in config.CRASH_MARKERS):
        return "crashed"
    if age < config.RUNNING_MAX_AGE:
        return "running"
    # The log just stops: no clean exit and no traceback. The run was killed, preempted
    # or the node went away -- which is not the same as crashing, so say so rather than
    # overstating what the log actually shows.
    return "stopped"


def _pred_files(outputs: Path) -> list[Path]:
    """Every ep*.pt under train/ and eval/<split>/, via scandir (see _classify)."""
    out = []
    for split_dir in (outputs / "train", *_eval_dirs(outputs)):
        try:
            with os.scandir(split_dir) as it:
                out += [Path(e.path) for e in it
                        if e.name.startswith("ep") and e.name.endswith(".pt")]
        except OSError:
            continue
    return out


def _eval_dirs(outputs: Path) -> list[Path]:
    try:
        with os.scandir(outputs / "eval") as it:
            return sorted((Path(e.path) for e in it if e.is_dir()), key=lambda p: p.name)
    except OSError:
        return []


# A run's schema (mask shape, info fields) is fixed by the code that trained it, so a
# probe result stays valid for as long as that file exists. Caching it by path keeps a
# rescan from re-deserializing a .pt for every run -- the probes were ~100ms of a 121ms
# scan, and only genuinely new runs need paying for.
_PROBE_CACHE: dict[str, tuple[dict | None, str | None]] = {}


def _probe(files: list[Path]) -> tuple[dict | None, str | None]:
    """Load the newest readable prediction file to inspect its schema.

    Several .pt files on disk are truncated and raise PytorchStreamReader errors, so
    this walks back through the newest few rather than trusting a single probe --
    probing only one misclassifies runs whose newest file happens to be corrupt.
    """
    if not files:
        return None, "no prediction files"
    key = str(files[0])
    if key in _PROBE_CACHE:
        return _PROBE_CACHE[key]
    result = (None, "all recent prediction files unreadable")
    # Read all five rather than stopping at the first success. Sample ids are the ONLY
    # dataset-identity gate in _classify, and one file is one saved batch (B = eval batch
    # size, 108 by default), which is thin evidence -- five batches is ~540 ids. They are
    # usually five batches of the SAME split, since `files` is sorted by epoch and the
    # newest epoch's train batches sort together; the win is sample size, not split
    # coverage. Reading every .pt instead is not viable: one run can be 152MB over 556
    # files and there are hundreds of run dirs.
    #
    # Measured over 271 runs: ~390ms of torch.load vs ~100ms when this stopped at the
    # first success. That is a COLD cost only -- _PROBE_CACHE is keyed per file, so the
    # 10s rescans re-probe just the new runs -- and 433ms total against RESCAN_SECONDS
    # is worth a 5x stronger gate now that ids are the only dataset check.
    base = None
    ids: list[int] = []
    boxes: list = []   # which physical box each id in `ids` came from, aligned index-wise
    for p in files[:5]:
        try:
            obj = torch.load(p, map_location="cpu", weights_only=False)
        except Exception:
            continue
        mask, info = obj.get("mask_pred"), obj.get("info") or {}
        sid = info.get("sample_id")
        if sid is not None:
            batch_ids = _as_int_array(sid).tolist()
            ids.extend(batch_ids)
            boxes.extend(_as_box_list(info, len(batch_ids)))
        if base is None:
            # Keep only the few facts classification needs; holding the tensors would
            # pin hundreds of MB across every scanned run for no benefit. Schema is fixed
            # by the code that trained the run, so the first readable file settles shape
            # and info_keys -- later files only widen the id sample.
            base = {"shape": tuple(mask.shape[-2:]) if mask is not None else None,
                    "info_keys": set(info)}
    if base is not None:
        result = (base | {"sample_ids": ids, "boxes": boxes}, None)
    _PROBE_CACHE[key] = result
    return result


def _probe_run(run_dir: Path):
    """Walk + probe a run dir ONCE: (outputs, files, obj, err).

    Split out of _classify so that classifying one run against several experiments (the
    multi-experiment case) pays the scandir walk and the torch.load probes only once,
    not once per experiment.
    """
    outputs = run_dir / config.OUTPUTS_SUBDIR
    if not outputs.is_dir():
        return outputs, [], None, "no outputs_history/ (older run format)"
    # scandir rather than rglob: a run can hold thousands of prediction files, and rglob
    # builds a Path per entry plus extra stat calls, ~8x slower over the same 29k files.
    # Newest epoch first.
    files = sorted(_pred_files(outputs), key=_epoch_of, reverse=True)
    obj, err = _probe(files)
    return outputs, files, obj, err


def _classify(name: str, outputs: Path, files: list[Path], obj: dict | None,
             err: str | None, registry: "Registry") -> RunEntry:
    if obj is None:
        return RunEntry(name, False, err)

    shape = obj["shape"]
    if shape is None:
        return RunEntry(name, False, "no mask_pred in payload")
    if not {"sample_id", "x_com"} <= obj["info_keys"]:
        return RunEntry(name, False, "legacy info schema")

    splits = [p.name for p in _eval_dirs(outputs)]

    # A run's samples can come from more than one experiment (a "combined" training run
    # predicting several boxes at once) -- so identity is checked PER PROBED SAMPLE, routed
    # by its own `box` when the run's info carries one, falling back to id-overlap against
    # whichever loaded experiment has that id for older runs saved before `box` was added.
    # Split NAMES are deliberately NOT part of this test. One dataset gets sliced many ways
    # -- the same gastronorm capture yields 1-cube/2-cubes from `--split gastronorm` and
    # 1-obj/2-obj from gastronorm_train1_eval2 (_gastronorm_object_count_split in
    # src/model/dataset.py) -- so matching on names rejected every objcount-* run as
    # "different dataset" despite identical sample ids, grid and info schema.
    probe_ids = obj["sample_ids"]
    probe_boxes = obj["boxes"]
    families = set()
    experiments = set()
    for sid, box in zip(probe_ids, probe_boxes):
        gi = registry.route(sid, box)
        if gi is None:
            reason = (f"box {box!r} has no loaded experiment" if box is not None
                      else "different dataset (sample ids not in any loaded experiment)")
            return RunEntry(name, False, reason, eval_splits=splits)
        gt = registry.gts[gi]
        # has_shape(), not masks_at(): classification only needs to know whether targets
        # that size EXIST, and masks_at decodes all of them to answer that -- ~13MB per
        # shape allocated during a scan for columns the user may never open. Note this is
        # deliberately every shape on disk, not usable_mask_shapes(): that one drops
        # degenerate sizes because they make a bad DEFAULT, but a run trained on one is
        # still legitimately comparable against it.
        if not gt.has_shape(shape):
            have = ", ".join(f"{a}x{b}" for a, b in gt.disk_shapes()) or "none"
            return RunEntry(name, False,
                            f"mask shape {shape}, no ground truth that size (have: {have})",
                            eval_splits=splits)
        families.add(gt.layout.dataset)
        experiments.add(gi)

    # A run is comparable at whatever grid it was trained on, as long as its experiment(s)
    # ship targets that size -- runs at different resolutions can sit in one table. Both
    # metrics are grid-normalized (soft-IoU is a ratio, mse averages over cells), so the
    # columns share a scale; the header labels the size because a coarser grid is
    # systematically easier, not because the numbers are in different units.
    family = ("unknown" if not splits
              else next(iter(families)) if len(families) == 1 else "combined")
    # Recency comes from the highest-epoch file rather than a stat() of every .pt: epochs
    # are written in order, so it ranks runs identically at a fraction of the cost.
    newest = files[0]
    return RunEntry(name, True, None, newest.stat().st_mtime, _epoch_of(newest), splits,
                    family, shape=shape, experiments=frozenset(experiments))


def _run_shape_counts(runs_dir: Path, sample: int = 40):
    """How many of the `sample` most-recently-modified runs under `runs_dir` were trained
    at each mask shape, unfiltered.

    Computed ONCE and shared across every experiment's most_trained_shape() pick in
    load_experiments -- the run listing, sort and per-run probe are identical work
    regardless of which shapes a given experiment finds usable, so probing them once per
    experiment would re-walk and re-stat the same `sample` run dirs N times over.

    Lives here rather than in config so it can reuse _pred_files (scandir, ~8x faster
    than glob over the same tree) and _probe (cached in _PROBE_CACHE, so the .pt reads
    are shared with the scan_runs that follows moments later instead of paid twice).
    """
    from collections import Counter
    counts: Counter = Counter()
    try:
        dirs = [p for p in runs_dir.iterdir() if (p / config.OUTPUTS_SUBDIR).is_dir()]
    except OSError:
        return counts
    # Newest first, so a long tail of abandoned runs cannot outvote current work.
    dirs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    for d in dirs[:sample]:
        files = sorted(_pred_files(d / config.OUTPUTS_SUBDIR), key=_epoch_of, reverse=True)
        obj, _ = _probe(files)
        if obj and obj["shape"]:
            counts[tuple(obj["shape"])] += 1
    return counts


def most_trained_shape(counts, allowed) -> tuple[int, int] | None:
    """The grid most runs were trained at (from a shared `_run_shape_counts` result),
    restricted to `allowed`.

    Which size to default to is a property of the RUNS, not of the dataset: an experiment
    can ship targets at several sizes while its runs overwhelmingly use one of them.
    """
    allowed = {tuple(a) for a in allowed}
    restricted = [(s, n) for s, n in counts.items() if s in allowed]
    return max(restricted, key=lambda sn: sn[1])[0] if restricted else None


def scan_runs(runs_dir: Path, registry: "Registry") -> list[RunEntry]:
    entries = []
    for d in sorted(runs_dir.iterdir()):
        if not d.is_dir():
            continue
        outputs, files, obj, err = _probe_run(d)
        e = _classify(d.name, outputs, files, obj, err, registry)
        e.status = run_status(d)      # status is useful even for runs we can't compare
        entries.append(e)
    entries.sort(key=lambda e: (not e.compatible, -e.mtime, e.name))
    return entries


def load_experiments(path: Path, runs_dir: Path,
                     mask_override: tuple[int, int] | None = None) -> list[GtIndex]:
    """One GtIndex per experiment under `path`.

    `path` is auto-detected as either a single experiment dir (has samples/ directly) or
    a parent dir holding several (each immediate child with samples/ is one experiment).
    This keeps the per-training-job auto-launch (ensure_viz, viz/__init__.py -- always
    given one experiment's data_dir) on the exact single-dataset fast path it always used,
    while a bare `python -m viz` against the experiments/ parent loads everything.

    Mask-grid selection is done HERE, per experiment, rather than once globally: different
    box captures ship different native mask sizes with no size common to all of them, so
    there is no single global grid that works for a merged view. The resolved (h, w) is
    passed explicitly to load_gt/Layout for each experiment -- never through a shared
    mutable global, so experiments never depend on load order or leftover state from a
    previous iteration.
    """
    if (path / "samples").is_dir():
        dirs = [path]
    else:
        dirs = sorted(p for p in path.iterdir()
                      if p.is_dir() and (p / "samples").is_dir())
        if not dirs:
            raise SystemExit(f"[viz] no experiment directories (with samples/) under {path}")
    single = len(dirs) == 1

    def skip_or_fail(d: Path, msg: str) -> None:
        # A single explicit --experiment dir fails hard, exactly as before this loop
        # existed; a parent dir just drops that one experiment and keeps going.
        if single:
            raise SystemExit(f"[viz] {d}: {msg}")
        print(f"[viz] skipping {d.name}: {msg}")

    # Computed once and shared across every experiment's most_trained_shape() pick below
    # (see _run_shape_counts) -- the run-dir listing/probe it needs is identical work
    # regardless of which shapes a given experiment considers usable.
    shape_counts = None if mask_override else _run_shape_counts(runs_dir)

    gts = []
    for d in dirs:
        samples_dir = d / "samples"
        if mask_override:
            h, w = mask_override
            available = config.mask_shapes(samples_dir)
            if available and (h, w) not in available:
                sizes = ", ".join(f"{a}x{b}" for a, b in available)
                skip_or_fail(d, f"no {h}x{w} masks (available: {sizes})")
                continue
        else:
            # usable_mask_shapes drops sizes with masks that carry no real mass (see its
            # docstring); most_trained_shape then prefers whichever usable size most of
            # this experiment's OWN runs were trained at, same as the old __main__.py did
            # globally -- just run once per experiment now.
            usable = config.usable_mask_shapes(samples_dir)
            if usable:
                h, w = most_trained_shape(shape_counts, usable) or usable[0]
            elif single:
                # load_gt raises its own descriptive "available sizes: ..." error below,
                # matching today's single-experiment behaviour exactly.
                h, w = config.DEFAULT_MASK_SHAPE
            else:
                skip_or_fail(d, "no usable ground-truth masks")
                continue
        gts.append(load_gt(d, h, w))

    if not gts:
        raise SystemExit(f"[viz] no usable experiments under {path}")
    return gts


# ***** parameter count *****

# BatchNorm keeps these in its state_dict, but they are running statistics, not learned
# weights -- counting them inflates the number the header reports as "params".
_BN_BUFFER_SUFFIXES = ("running_mean", "running_var", "num_batches_tracked")

# Keyed on (path, mtime, size): a checkpoint that has been counted will not change, and a
# run still training writes a NEW file (new mtime), so its count refreshes on its own.
_PARAM_CACHE: dict[tuple[str, float, int], int | None] = {}


def _checkpoint_file(run_dir: Path) -> Path | None:
    """The newest checkpoint for a run, or None if it kept none."""
    ckpt = run_dir / "checkpoints"
    latest = ckpt / "latest-rank0.pt"
    if latest.exists():
        return latest.resolve()
    try:
        pts = [p for p in ckpt.iterdir() if p.suffix == ".pt"]
    except OSError:
        return None
    return max(pts, key=_epoch_of) if pts else None


def param_count(name: str, runs_dir: Path) -> int | None:
    """Trainable parameters in a run's model, from its newest checkpoint.

    Composer stores the model state_dict under state.model; sum numel over it, minus the
    BatchNorm running buffers. Loaded with mmap=True so a 250MB checkpoint costs a few ms
    -- numel reads shape metadata only and never touches the storage -- and the result is
    cached on the file's identity so a still-training run re-counts only when it saves.
    """
    p = _checkpoint_file(runs_dir / name)
    if p is None:
        return None
    try:
        stat = p.stat()
    except OSError:
        return None
    key = (str(p), stat.st_mtime, stat.st_size)
    if key not in _PARAM_CACHE:
        n: int | None = None
        try:
            obj = torch.load(p, map_location="cpu", weights_only=False, mmap=True)
            model = obj.get("state", {}).get("model", {}) or {}
            n = sum(int(v.numel()) for k, v in model.items()
                    if hasattr(v, "numel") and not k.endswith(_BN_BUFFER_SUFFIXES))
        except Exception:
            n = None
        _PARAM_CACHE[key] = n
    return _PARAM_CACHE[key]


# ***** per-run predictions + metrics *****


@dataclass
class RunData:
    name: str
    epoch: int
    sample_ids: np.ndarray       # (M,) int
    masks: np.ndarray            # (M,20,40) float32, sigmoid probabilities
    splits: list[str]            # per row: "train" or the eval split name
    metrics: dict               # {name: (M,) array} -- see METRIC_KEYS
    com_pred: np.ndarray         # (M,2) grid-space, for display
    row_of: dict[int, int]
    skipped_files: list[str]
    family: str = "unknown"
    # Grid this run predicts at. Required, not defaulted: the table mixes resolutions, so
    # there is no sensible fallback. Every return path passes it explicitly -- the
    # zero-row ones used to fall through to (0,0), which the client read as "shape
    # unknown" and silently replaced with the default grid.
    shape: tuple[int, int] = (0, 0)
    # Why this run has no rows, when it has none. A run can pass compatibility (targets
    # that size exist on disk) and still score nothing, and an empty column with no
    # explanation is indistinguishable from a bug. Surfaced by /api/run.
    reason: str | None = None
    # Crosshair geometry per row: matched pred/target object centroids and the unmatched
    # ones, normalized to [0,1]. See com_pairs(). Last, and defaulted, so the three
    # empty-column return paths need not pass it.
    com_pairs: list = field(default_factory=list)
    # Trainable parameter count, read from the newest checkpoint (see param_count). None
    # when the run kept no checkpoint. Shown in the column header.
    n_params: int | None = None
    # GLOBAL sample id per row (see Registry's multi-experiment id space in SPEC.md),
    # parallel to sample_ids -- what /api/run keys its per-sample metrics dict by, to match
    # /api/samples' s.i. A combined run's rows can come from different experiments, each
    # with its own offset, so this can no longer be computed from one shared offset.
    global_ids: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))


def _split_dirs(outputs: Path) -> list[tuple[str, Path]]:
    out = [("train", outputs / "train")] if (outputs / "train").is_dir() else []
    return out + [(p.name, p) for p in _eval_dirs(outputs)]


def load_epoch_masks(name: str, runs_dir: Path, epoch: int,
                     want: set[tuple[int, object]]) -> dict[tuple[int, object], np.ndarray]:
    """Just the masks for `want` at one epoch, as {(local_sample_id, box): (H,W)}.

    `want` pairs each local id with the box it must belong to (or None to match on id
    alone, for runs predating the `box` info field) -- a combined run can predict two
    different boxes' same-numbered local sample, so the id alone cannot key this safely.

    load_run() would decode every sample of every split and score them all -- ~126MB of
    tensors and a full metric pass -- to serve the couple of dozen cells on screen. The
    epoch scrubber only needs pixels, so this skips the metrics entirely and keeps only
    the requested rows.
    """
    outputs = runs_dir / name / config.OUTPUTS_SUBDIR
    want_ids = {lid for lid, _ in want}
    out: dict[tuple[int, object], np.ndarray] = {}
    for _, d in _split_dirs(outputs):
        for p in d.glob(f"ep{epoch:04d}-*.pt"):
            try:
                obj = torch.load(p, map_location="cpu", weights_only=False)
            except Exception:
                continue
            sid = _as_int_array(obj["info"]["sample_id"])
            hit = [i for i, s in enumerate(sid) if int(s) in want_ids]
            if not hit:
                continue
            boxes = _as_box_list(obj["info"], len(sid))
            m = obj["mask_pred"]
            for i in hit:
                s, box = int(sid[i]), boxes[i]
                if (s, box) in want:
                    out[(s, box)] = m[i].float().numpy()
    return out


# Epochs whose every file failed to load, keyed by (run, epoch). A truncated .pt (a run
# killed mid-save) otherwise defines an epoch that exists in the filename listing but
# yields no samples at all -- the slider would offer it and the whole column would read
# "no prediction". Filled in by load_run, which is the only place that finds out.
_DEAD_EPOCHS: set[tuple[str, int]] = set()


def run_epochs(name: str, runs_dir: Path) -> list[int]:
    """Every epoch this run has usable predictions for, ascending.

    Epochs proven unreadable are excluded, so the epoch slider never offers a frame that
    would render as an empty column -- and, because the slider's maximum is the highest
    epoch across the loaded runs, never overstates how far a run actually got.
    """
    outputs = runs_dir / name / config.OUTPUTS_SUBDIR
    files = _pred_files(outputs)
    eps = sorted({_epoch_of(p) for p in files})

    # The newest epoch is the one at risk: a run killed mid-write leaves a truncated .pt
    # that still names a valid epoch. Verify only that one -- the cost is a single load,
    # and older epochs were completed before the next began.
    while eps:
        newest = eps[-1]
        # A previously-dead epoch gets ANOTHER look rather than being trusted forever.
        # For a run that is still training, the newest .pt is routinely caught mid-write;
        # a permanent blacklist meant that one transient failure hid every epoch from
        # then on, which is why a live run appeared frozen at an old epoch until reload.
        if any(_loadable(p) for p in files if _epoch_of(p) == newest):
            _DEAD_EPOCHS.discard((name, newest))
            break
        _DEAD_EPOCHS.add((name, newest))
        eps.pop()

    return [e for e in eps if (name, e) not in _DEAD_EPOCHS]


@lru_cache(maxsize=4096)
def _loadable_at(path: Path, mtime: float, size: int) -> bool:
    """Whether a prediction file can be read, keyed on its identity ON DISK.

    A file that parsed once will not stop parsing, but a file that FAILED can very much
    start working: a run writing ep0500 right now yields a truncated read, and moments
    later the same path is a complete tensor. Keying the cache on (mtime, size) means the
    finished file is a different key from the half-written one, so the failure expires by
    itself instead of blacklisting a live run's newest epoch for the process lifetime.
    """
    try:
        torch.load(path, map_location="cpu", weights_only=False)
        return True
    except Exception:
        return False


def _loadable(path: Path) -> bool:
    try:
        st = path.stat()
    except OSError:
        return False
    return _loadable_at(path, st.st_mtime, st.st_size)


def load_run(name: str, runs_dir: Path, registry: "Registry", family: str = "unknown",
             epoch: int | None = None, shape: tuple[int, int] | None = None) -> RunData:
    """Load one run's predictions and score every sample against the target.

    `epoch` selects which saved epoch to read; the default (None) uses each split's
    latest, which is what the table shows. `shape` is the grid the run was classified at,
    used only so a run that decodes NO predictions can still report the size it would
    have had -- without it the client cannot size that column's canvas.

    A run's predictions can come from more than one experiment (a "combined" training run
    predicting several boxes at once) -- every row is routed to ITS OWN experiment via
    `registry.route`, using that row's own `box` field where available. This is why `gt`
    became `registry`: no single GtIndex is enough once a run can span more than one.

    Metrics mirror the training loop exactly (see src/model/arch.py mses/com_distances
    and utils/metrics.soft_iou), which is what lets the column headers be cross-checked
    against the numbers in the run's own logs-rank0.txt.
    """
    outputs = runs_dir / name / config.OUTPUTS_SUBDIR
    ids, masks, splits, boxes, skipped = [], [], [], [], []
    want, epoch = epoch, 0

    for split, d in _split_dirs(outputs):
        files = sorted(d.glob("ep*.pt"), key=_epoch_of)
        if want is not None:
            files = [f for f in files if _epoch_of(f) == want]
        if not files:
            continue
        # Walk this split's epochs newest-first until one actually loads. Without this a
        # single truncated final save (a run killed mid-write) would drop the split from
        # the table entirely, even though every earlier epoch is intact. When a specific
        # epoch was requested there is only one candidate, so this is a no-op.
        for last in sorted({_epoch_of(f) for f in files}, reverse=True):
            got = False
            # Ascending batch order, so that if a sample appears in more than one batch
            # of this epoch the LAST write is the one kept by the dedupe below.
            for p in sorted([f for f in files if _epoch_of(f) == last], key=_batch_of):
                try:
                    obj = torch.load(p, map_location="cpu", weights_only=False)
                except Exception:
                    skipped.append(str(p.relative_to(outputs)))
                    continue
                sid = _as_int_array(obj["info"]["sample_id"])
                ids.append(sid)
                masks.append(obj["mask_pred"].float().numpy())
                splits += [split] * len(sid)
                boxes += _as_box_list(obj["info"], len(sid))
                got = True
            if got:
                epoch = max(epoch, last)
                break

    if not ids:
        # Every file for this epoch was unreadable. Remember that, so the epoch stops
        # being offered, then retry on the next-newest epoch rather than handing back a
        # column of "no prediction" -- a corrupt final save should not hide a run.
        if skipped:
            _DEAD_EPOCHS.add((name, want if want is not None else epoch))
            # Newest surviving epoch at or below the one asked for.
            usable = [e for e in run_epochs(name, runs_dir)
                      if want is None or e <= want]
            if usable:
                return load_run(name, runs_dir, registry, family, epoch=max(usable), shape=shape)
        empty_f = np.zeros(0, dtype=np.float64)
        # The classified shape, not the primary grid: this run predicts at its own size
        # even when nothing decoded, and the column has to be drawn at that size.
        out_shape = shape or (registry.gts[0].masks.shape[1], registry.gts[0].masks.shape[2])
        return RunData(name, epoch, np.zeros(0, dtype=np.int64),
                       np.zeros((0, *out_shape), dtype=np.float32), [],
                       {k: empty_f for k in METRIC_KEYS}, np.zeros((0, 2)), {}, skipped, family,
                       out_shape, "no readable prediction files")

    sample_ids = np.concatenate(ids)
    preds = np.concatenate(masks).astype(np.float32)
    boxes_arr = np.array(boxes, dtype=object)

    # Route every row to the experiment its OWN box belongs to -- sample ids are local to
    # one experiment and are not unique across them, so this can no longer be one shared
    # id->row lookup. Drop rows that route to nothing rather than indexing something
    # arbitrary -- a stray id would otherwise be scored against a real but unrelated mask.
    routed = [registry.route(int(s), b) for s, b in zip(sample_ids, boxes_arr)]
    keep = np.array([r is not None for r in routed], dtype=bool)
    gi_arr = np.array([r for r in routed if r is not None], dtype=np.int64)
    if not keep.all():
        sample_ids, preds = sample_ids[keep], preds[keep]
        splits = [s for s, k in zip(splits, keep) if k]

    if len(sample_ids) == 0:
        empty_f = np.zeros(0, dtype=np.float64)
        return RunData(name, epoch, sample_ids, preds, splits,
                       {k: empty_f for k in METRIC_KEYS}, np.zeros((0, 2)), {}, skipped, family,
                       (preds.shape[-2], preds.shape[-1]),
                       "no predicted sample belongs to a loaded experiment")

    # (experiment, id) identifies a scene -- the same numeric key load_run ultimately
    # reports as this row's global id (registry.global_id), reused here for dedup so
    # there is exactly one formula for "which scene is this row" rather than a second,
    # string-encoded stand-in for the same thing.
    scene_ids = registry.global_id(gi_arr, sample_ids)

    # A sample can be written more than once in an epoch (at ep0 the train split spans two
    # partial passes, so ~460 ids appear twice). Keep only the last write per scene --
    # otherwise duplicates are scored twice and skew the column's mean and std.
    if len(np.unique(scene_ids)) != len(scene_ids):
        keep = np.zeros(len(sample_ids), dtype=bool)
        # Later rows come from later batches, so reversing makes "first seen" the newest.
        _, first = np.unique(scene_ids[::-1], return_index=True)
        keep[len(sample_ids) - 1 - first] = True
        sample_ids, preds, gi_arr, scene_ids = (
            sample_ids[keep], preds[keep], gi_arr[keep], scene_ids[keep])
        splits = [s for s, k in zip(splits, keep) if k]

    # Score against ground truth AT THIS RUN'S GRID. Runs trained at different
    # resolutions each get their own targets, so a 16x16 and a 30x30 column can sit side
    # by side; scoring both against one fixed size would broadcast-error or, worse,
    # compare a prediction to a target of a different shape.
    shape = (preds.shape[-2], preds.shape[-1])

    # Ground truth gathered PER ROW from that row's own experiment -- a combined run's
    # rows can come from different GtIndex objects, so this is no longer one shared array
    # indexed once. Grouped by experiment rather than looped element-by-element: a handful
    # of experiments, not a Python loop over every row.
    global_rows = np.empty(len(sample_ids), dtype=np.int64)
    truth = np.zeros((len(sample_ids), *shape), dtype=np.float32)
    ok = np.ones(len(sample_ids), dtype=bool)
    for gi in np.unique(gi_arr).tolist():
        m = gi_arr == gi
        gt = registry.gts[gi]
        local_rows = np.array([gt.row_of[int(s)] for s in sample_ids[m]], dtype=np.int64)
        global_rows[m] = registry.row_base(gi) + local_rows
        gt_masks = gt.masks_at(shape)
        if gt_masks is None:
            # Classification said targets exist at this size (has_shape globs real
            # filenames) but none could actually be decoded for this experiment.
            ok[m] = False
            continue
        truth[m] = gt_masks[local_rows]

    if not ok.all():
        sample_ids, preds, gi_arr, global_rows, truth = (
            sample_ids[ok], preds[ok], gi_arr[ok], global_rows[ok], truth[ok])
        splits = [s for s, k in zip(splits, ok) if k]

    if len(sample_ids) == 0:
        empty_f = np.zeros(0, dtype=np.float64)
        return RunData(name, epoch, np.zeros(0, dtype=np.int64),
                       np.zeros((0, *shape), dtype=np.float32), [],
                       {k: empty_f for k in METRIC_KEYS}, np.zeros((0, 2)), {}, skipped, family,
                       shape, f"no ground truth at {shape[0]}x{shape[1]}")

    pred = torch.from_numpy(preds)
    truth_t = torch.from_numpy(truth)

    # pred is already sigmoid probabilities; viz never sees the logits, so bce is scored
    # from clamped probs (negligibly different from logit-space).
    # One call, not two: geometry=True returns the crosshair geometry from the SAME
    # labelling/matching pass that produces the numbers, which halves the scoring cost of a
    # run and makes it impossible for a drawn line to disagree with its metric.
    loc, geom = localization(pred, truth_t, geometry=True)
    metrics = {
        'bce': _F.binary_cross_entropy(pred.clamp(1e-6, 1 - 1e-6), truth_t, reduction='none').mean(dim=(-2, -1)).numpy(),
        'iou': soft_iou(pred, truth_t).numpy(),
        'contour': contour_f(pred, truth_t).numpy(),
        'mass': mass_error(pred, truth_t).numpy(),
        **{k: v.numpy() for k, v in loc.items()},
    }
    com_pred = center_of_mass(pred, epsilon=config.EPSILON).numpy()

    # Keyed by GLOBAL ROW, not sample id: every consumer (render._cached, /api/mask.png,
    # /api/values, /api/neighbors) resolves the id through _sid() at the HTTP edge and
    # passes the row down. global_rows is filtered in lockstep with sample_ids above, so
    # global_rows[i] is the ground-truth row of sample_ids[i]. See SPEC.md 2 and 5.
    row_of = {int(r): i for i, r in enumerate(global_rows)}
    global_ids = registry.global_id(gi_arr, sample_ids)
    return RunData(name, epoch, sample_ids, pred.numpy(), splits,
                   metrics, com_pred, row_of, skipped, family, shape,
                   com_pairs=geom, global_ids=global_ids)


# ***** registry *****


# Global sample id = gi * ID_STRIDE + local sample id. Safe because sample ids are
# zero-padded 6-digit strings ("000000".."999999"), so a local id is always < ID_STRIDE
# -- the two halves of a global id never collide and the split (divmod) is exact. With
# exactly one experiment loaded this reduces to gi=0, global id == local id, so a
# single-experiment server (ensure_viz's per-training-job case) emits and accepts
# identical ids to before this change.
ID_STRIDE = 1_000_000


class Registry:
    """Startup loads ground truth (one GtIndex per experiment) and scans run dirs;
    prediction tensors are loaded lazily on first request and then cached for the
    process lifetime.

    Everything below the registry (routes, render.py) speaks ROWS in one flat space
    spanning every loaded experiment -- locate() is the single place a row resolves back
    to which experiment's GtIndex it came from, mirroring how _sid()/sample_index is the
    single place a sample id resolves to a row. See SPEC.md.
    """

    def __init__(self, experiment_dir: Path, runs_dir: Path,
                mask_override: tuple[int, int] | None = None):
        t0 = time.perf_counter()
        self.experiment_dir, self.runs_dir = experiment_dir, runs_dir
        self.gts = load_experiments(experiment_dir, runs_dir, mask_override)
        self._index_experiments()
        self._scanned_at = 0.0
        self.rescan()
        self._runs: dict[tuple[str, int | None], RunData] = {}
        n_ok = sum(e.compatible for e in self.entries)
        self.startup_s = time.perf_counter() - t0
        for gt in self.gts:
            print(f"[viz] {gt.experiment_dir.name} | layout '{gt.layout.name}' | "
                  f"{len(gt)} samples")
        print(f"[viz] {n_ok} compatible / {len(self.entries) - n_ok} incompatible runs | "
              f"{self.startup_s:.2f}s")

    def _index_experiments(self) -> None:
        """Build the flat row space: `_row_base[gi]` is where experiment gi's rows start,
        and `row_of` maps a GLOBAL sample id straight to its global row. Also builds
        `_gi_by_box`, which `route()` uses to send a prediction to the RIGHT experiment
        even when its local sample id happens to collide with another experiment's."""
        self._row_base: list[int] = []
        self.row_of: dict[int, int] = {}
        self._gi_by_box: dict[str, int] = {}
        base = 0
        for gi, gt in enumerate(self.gts):
            self._row_base.append(base)
            for local, sid in enumerate(gt.sample_ids):
                self.row_of[self.global_id(gi, int(sid))] = base + local
            base += len(gt)
            # Every experiment on disk today holds exactly one box, so first-seen-wins is
            # unambiguous; if that ever changes, this only affects the box->experiment
            # fast path in route() -- id-overlap still catches a sample whose box lookup
            # picked the wrong one, so a shared box name fails loud (id not in that
            # experiment) rather than silently scoring against the wrong ground truth.
            for m in gt.meta:
                b = m.get("box")
                if b is not None:
                    self._gi_by_box.setdefault(b, gi)
        self.n_samples = base

    def global_id(self, gi, local_sample_id):
        """gi * ID_STRIDE + local_sample_id -- the ONE place this formula is written.
        Works elementwise on plain ints or on equal-length numpy arrays (load_run scores
        every row of a run at once, so it needs the vectorized form of the same thing
        Registry._index_experiments and app.py's scalar call sites use)."""
        return gi * ID_STRIDE + local_sample_id

    def route(self, local_sample_id: int, box: str | None) -> int | None:
        """Which experiment (gi) a PREDICTED row belongs to, by its own `box` field.

        A run can predict samples from more than one box (a "combined" training run), and
        local sample ids are not unique across experiments -- two boxes can both have a
        "000010" -- so `box` (saved per-row in the .pt's info, present on runs trained
        after that field was added) is the reliable signal, checked first. Falls back to
        id-overlap against every loaded experiment for older runs with no `box` in info.
        A `box` that names an experiment NOT loaded, or whose experiment doesn't have this
        id, resolves to None rather than falling through to id-overlap -- trusting a wrong
        guess over an honest "unresolvable" is exactly the bug this method exists to avoid.
        """
        local_sample_id = int(local_sample_id)
        if box is not None:
            gi = self._gi_by_box.get(box)
            if gi is not None and local_sample_id in self.gts[gi].row_of:
                return gi
            return None
        for gi, gt in enumerate(self.gts):
            if local_sample_id in gt.row_of:
                return gi
        return None

    def locate(self, row: int) -> tuple[int, GtIndex, int]:
        """(experiment index, its GtIndex, local row) for a global row.

        The one place every route/render call resolves a row back to the experiment it
        came from, so file paths, layouts and mask arrays are never read off the wrong
        GtIndex. Linear scan is fine: there are a handful of experiments, not thousands.
        """
        row = int(row)
        if not 0 <= row < self.n_samples:
            raise KeyError(row)
        for gi in range(len(self.gts) - 1, -1, -1):
            if row >= self._row_base[gi]:
                return gi, self.gts[gi], row - self._row_base[gi]
        raise KeyError(row)  # unreachable: _row_base[0] == 0

    def row_base(self, gi: int) -> int:
        return self._row_base[gi]

    def rescan(self) -> None:
        """Re-read the runs directory so runs that finish while viz is open show up
        without a restart. Costs ~0.15s per experiment, and only probes one file per
        run regardless of how many experiments it's classified against."""
        self.entries = scan_runs(self.runs_dir, self)
        self.by_name = {e.name: e for e in self.entries}
        self._scanned_at = time.monotonic()

    def maybe_rescan(self, max_age: float = config.RESCAN_SECONDS) -> None:
        if time.monotonic() - self._scanned_at > max_age:
            self.rescan()

    def defaults(self) -> list[str]:
        """Auto-loaded on first open: up to N_DEFAULT_RUNS per loaded experiment, not
        N_DEFAULT_RUNS total.

        A single global top-N (by recency, across every box) can leave whole experiments
        with nothing loaded -- filtering the Box chip to one of them then shows every
        column as "not in run" even though real predictions exist, because the runs that
        happen to be newest all belong to OTHER boxes. Picking per experiment means
        switching the Box filter to any loaded box always has at least one real column.
        `entries` is already sorted compatible-first, newest-first, so the first
        `N_DEFAULT_RUNS` un-picked matches per experiment are also the most recent ones.
        """
        chosen: list[str] = []
        seen: set[str] = set()
        for gi in range(len(self.gts)):
            n = 0
            for e in self.entries:
                if not e.compatible or e.name in seen or gi not in e.experiments:
                    continue
                chosen.append(e.name)
                seen.add(e.name)
                n += 1
                if n >= config.N_DEFAULT_RUNS:
                    break
        return chosen

    def run(self, name: str, reload: bool = False, epoch: int | None = None) -> RunData:
        entry = self.by_name.get(name)
        if entry is None:
            # A run requested but not in the cached scan may have just appeared.
            self.rescan()
            entry = self.by_name.get(name)
        if entry is None or not entry.compatible:
            raise KeyError(name)
        key = (name, epoch)
        if reload:
            self._runs.pop(key, None)
        if key not in self._runs:
            rd = load_run(name, self.runs_dir, self, entry.family, epoch, entry.shape)
            rd.n_params = param_count(name, self.runs_dir)
            self._runs[key] = rd
            # Each RunData is ~5MB, and scrubbing a 200-epoch run would otherwise pin a
            # gigabyte. Latest-epoch entries (epoch=None) are what the table always needs,
            # so evict scrubbed ones first, oldest first.
            scrubbed = [k for k in self._runs if k[1] is not None]
            for old in scrubbed[:-config.MAX_EPOCH_CACHE]:
                self._runs.pop(old, None)
        return self._runs[key]

    def epochs(self, name: str) -> list[int]:
        return run_epochs(name, self.runs_dir)

    def sample_index(self, sid) -> int:
        """Row for an untrusted (global) sample id.

        Coerced to int and resolved through the id->row map, so no user string ever
        reaches a filesystem join. This is a LOOKUP, not a range check: ids do not start
        at zero on every dataset, so treating the id as the row silently served the wrong
        sample's mask and images.
        """
        i = self.row_of.get(int(sid))
        if i is None:
            raise KeyError(sid)
        return i

    def sample_dir(self, row: int) -> Path:
        """Directory for a ROW (what sample_index returns), not a sample id.

        Every caller already holds a row from _sid()/sample_index. Taking an id here and
        re-resolving it worked only while row == int(id); on a dataset whose ids start
        anywhere but zero it silently served a different sample's images. The name comes
        off gt.sample_ids, so nothing user-supplied reaches the join.
        """
        gi, gt, local = self.locate(row)
        return gt.experiment_dir / "samples" / gt.sample_ids[local]
