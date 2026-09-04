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
from utils.metrics import center_of_mass, soft_iou, mass_error, contour_f, localization

METRIC_KEYS = ('bce', 'iou', 'localization', 'localization_x', 'localization_y', 'contour', 'mass')

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


def load_gt(experiment_dir: Path) -> GtIndex:
    samples_dir = experiment_dir / "samples"
    layout = config.Layout.detect(samples_dir)
    sample_dirs = sorted(p for p in samples_dir.iterdir() if p.is_dir())
    ids, masks, meta = [], [], []
    for d in sample_dirs:
        gt = layout.resolve_gt_mask(d, config.MASK_H, config.MASK_W)
        if gt is None:
            continue
        m = np.load(gt)
        # A dataset can carry masks at several downsample sizes side by side (gastronorm
        # writes both 20x40 and 30x30). Only the configured target shape is loadable as
        # ground truth; a mismatch would otherwise fail deep inside np.stack.
        if m.shape != (config.MASK_H, config.MASK_W):
            continue
        ids.append(d.name)
        masks.append(m)
        meta.append(merge_metadata(d / "metadata.jsonl"))
    if not masks:
        # Only reachable via an explicit --mask: the default is chosen from sizes that
        # actually carry mass. Name the sizes that DO exist so the retry is obvious.
        have = ", ".join(f"{a}x{b}" for a, b in config.mask_shapes(samples_dir)) or "none"
        raise SystemExit(
            f"[viz] no {config.MASK_H}x{config.MASK_W} ground-truth masks under "
            f"{samples_dir} (layout '{layout.name}', searched "
            f"{layout.gt_mask_glob(config.MASK_H, config.MASK_W)}).\n"
            f"       available sizes: {have}. Pass one with --mask HxW."
        )
    masks = np.ascontiguousarray(np.stack(masks).astype(np.float32))
    com_gt = np.asarray(center_of_mass(masks), dtype=np.float64)
    avg_com = np.asarray([parse_com(m.get("avg_com")) for m in meta], dtype=np.float64)
    row_of = {int(s): i for i, s in enumerate(ids)}
    return GtIndex(ids, masks, meta, com_gt, avg_com, layout, row_of,
                   experiment_dir=experiment_dir)


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
    for p in files[:5]:
        try:
            obj = torch.load(p, map_location="cpu", weights_only=False)
        except Exception:
            continue
        mask, info = obj.get("mask_pred"), obj.get("info") or {}
        sid = info.get("sample_id")
        if sid is not None:
            ids.extend(_as_int_array(sid).tolist())
        if base is None:
            # Keep only the few facts classification needs; holding the tensors would
            # pin hundreds of MB across every scanned run for no benefit. Schema is fixed
            # by the code that trained the run, so the first readable file settles shape
            # and info_keys -- later files only widen the id sample.
            base = {"shape": tuple(mask.shape[-2:]) if mask is not None else None,
                    "info_keys": set(info)}
    if base is not None:
        result = (base | {"sample_ids": ids}, None)
    _PROBE_CACHE[key] = result
    return result


def _classify(name: str, run_dir: Path, gt: GtIndex) -> RunEntry:
    outputs = run_dir / config.OUTPUTS_SUBDIR
    if not outputs.is_dir():
        return RunEntry(name, False, "no outputs_history/ (older run format)")

    # Walk the tree ONCE, and with scandir rather than rglob: a run can hold thousands
    # of prediction files, and rglob builds a Path per entry plus extra stat calls,
    # which measured ~8x slower over the same 29k files. Newest epoch first.
    files = sorted(_pred_files(outputs), key=_epoch_of, reverse=True)

    obj, err = _probe(files)
    if obj is None:
        return RunEntry(name, False, err)

    shape = obj["shape"]
    if shape is None:
        return RunEntry(name, False, "no mask_pred in payload")
    # A run is comparable at whatever grid it was trained on, as long as this dataset
    # ships targets that size -- runs at different resolutions can sit in one table.
    # Both metrics are grid-normalized (soft-IoU is a ratio, mse averages over cells), so
    # the columns share a scale; the header labels the size because a coarser grid is
    # systematically easier, not because the numbers are in different units.
    # has_shape(), not masks_at(): classification only needs to know whether targets that
    # size EXIST, and masks_at decodes all 3000 of them to answer that -- ~13MB per shape
    # allocated during a scan for columns the user may never open. The full array loads
    # lazily on the first request that renders one. Note this is deliberately every shape
    # on disk, not usable_mask_shapes(): that one drops degenerate sizes because they make
    # a bad DEFAULT, but a run trained on one is still legitimately comparable against it.
    if not gt.has_shape(shape):
        have = ", ".join(f"{a}x{b}" for a, b in gt.disk_shapes()) or "none"
        return RunEntry(name, False, f"mask shape {shape}, no ground truth that size (have: {have})")

    if not {"sample_id", "x_com"} <= obj["info_keys"]:
        return RunEntry(name, False, "legacy info schema")

    splits = [p.name for p in _eval_dirs(outputs)]

    # Sample ids collide across experiments, so a run from another dataset would join
    # cleanly against the loaded ground truth and produce silently wrong metrics. Test that
    # directly: every id in the probe must exist in this dataset.
    #
    # Split NAMES are deliberately NOT the test. One dataset gets sliced many ways -- the
    # same gastronorm capture yields 1-cube/2-cubes from `--split gastronorm` and
    # 1-obj/2-obj from gastronorm_train1_eval2 (_gastronorm_object_count_split in
    # src/model/dataset.py) -- so matching on names rejected every objcount-* run as
    # "different dataset" despite identical sample ids, grid and info schema. Runs that
    # slice one dataset differently belong in the same table: each column keeps its own
    # split label, and samples a run does not cover render per-cell as "not in run".
    probe_ids = obj["sample_ids"]
    if probe_ids and not all(int(s) in gt.row_of for s in probe_ids):
        return RunEntry(name, False, "different dataset (sample ids not in this dataset)",
                        eval_splits=splits)

    family = gt.layout.dataset if splits else "unknown"
    # Recency comes from the highest-epoch file rather than a stat() of every .pt: epochs
    # are written in order, so it ranks runs identically at a fraction of the cost.
    newest = files[0]
    return RunEntry(name, True, None, newest.stat().st_mtime, _epoch_of(newest), splits,
                    family, shape=shape)


def most_trained_shape(runs_dir: Path, allowed, sample: int = 40) -> tuple[int, int] | None:
    """The grid most runs under `runs_dir` were trained at, restricted to `allowed`.

    Which size to default to is a property of the RUNS, not of the dataset: an experiment
    can ship targets at several sizes while its runs overwhelmingly use one of them.

    Lives here rather than in config so it can reuse _pred_files (scandir, ~8x faster
    than glob over the same tree) and _probe (cached in _PROBE_CACHE, so the .pt reads
    are shared with the scan_runs that follows moments later instead of paid twice).
    Only the newest `sample` runs are probed -- they are both what is being compared and
    the best predictor of the current grid, and this only picks a default --mask overrides.
    """
    from collections import Counter
    allowed = {tuple(a) for a in allowed}
    counts: Counter = Counter()
    try:
        dirs = [p for p in runs_dir.iterdir() if (p / config.OUTPUTS_SUBDIR).is_dir()]
    except OSError:
        return None
    # Newest first, so a long tail of abandoned runs cannot outvote current work.
    dirs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    for d in dirs[:sample]:
        files = sorted(_pred_files(d / config.OUTPUTS_SUBDIR), key=_epoch_of, reverse=True)
        obj, _ = _probe(files)
        if obj and obj["shape"] and tuple(obj["shape"]) in allowed:
            counts[tuple(obj["shape"])] += 1
    return counts.most_common(1)[0][0] if counts else None


def scan_runs(runs_dir: Path, gt: GtIndex) -> list[RunEntry]:
    entries = []
    for d in sorted(runs_dir.iterdir()):
        if not d.is_dir():
            continue
        e = _classify(d.name, d, gt)
        e.status = run_status(d)      # status is useful even for runs we can't compare
        entries.append(e)
    entries.sort(key=lambda e: (not e.compatible, -e.mtime, e.name))
    return entries


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
    # there is no sensible fallback, and a default evaluated at import time would freeze
    # config.MASK_H/W from before --mask ran. Every return path passes it explicitly --
    # the zero-row ones used to fall through to (0,0), which the client read as "shape
    # unknown" and silently replaced with the default grid.
    shape: tuple[int, int] = (0, 0)
    # Why this run has no rows, when it has none. A run can pass compatibility (targets
    # that size exist on disk) and still score nothing, and an empty column with no
    # explanation is indistinguishable from a bug. Surfaced by /api/run.
    reason: str | None = None


def _split_dirs(outputs: Path) -> list[tuple[str, Path]]:
    out = [("train", outputs / "train")] if (outputs / "train").is_dir() else []
    return out + [(p.name, p) for p in _eval_dirs(outputs)]


def load_epoch_masks(name: str, runs_dir: Path, epoch: int, want: set[int]) -> dict[int, np.ndarray]:
    """Just the masks for `want` at one epoch, as {sample_id: (H,W)}.

    load_run() would decode every sample of every split and score them all -- ~126MB of
    tensors and a full metric pass -- to serve the couple of dozen cells on screen. The
    epoch scrubber only needs pixels, so this skips the metrics entirely and keeps only
    the requested rows.
    """
    outputs = runs_dir / name / config.OUTPUTS_SUBDIR
    out: dict[int, np.ndarray] = {}
    for _, d in _split_dirs(outputs):
        for p in d.glob(f"ep{epoch:04d}-*.pt"):
            try:
                obj = torch.load(p, map_location="cpu", weights_only=False)
            except Exception:
                continue
            sid = _as_int_array(obj["info"]["sample_id"])
            hit = [i for i, s in enumerate(sid) if int(s) in want]
            if not hit:
                continue
            m = obj["mask_pred"]
            for i in hit:
                out[int(sid[i])] = m[i].float().numpy()
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


def load_run(name: str, runs_dir: Path, gt: GtIndex, family: str = "unknown",
             epoch: int | None = None, shape: tuple[int, int] | None = None) -> RunData:
    """Load one run's predictions and score every sample against the target.

    `epoch` selects which saved epoch to read; the default (None) uses each split's
    latest, which is what the table shows. `shape` is the grid the run was classified at,
    used only so a run that decodes NO predictions can still report the size it would
    have had -- without it the client cannot size that column's canvas.

    Metrics mirror the training loop exactly (see src/model/arch.py mses/com_distances
    and utils/metrics.soft_iou), which is what lets the column headers be cross-checked
    against the numbers in the run's own logs-rank0.txt.
    """
    outputs = runs_dir / name / config.OUTPUTS_SUBDIR
    ids, masks, splits, skipped = [], [], [], []
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
                return load_run(name, runs_dir, gt, family, epoch=max(usable), shape=shape)
        empty_f = np.zeros(0, dtype=np.float64)
        # The classified shape, not the primary grid: this run predicts at its own size
        # even when nothing decoded, and the column has to be drawn at that size.
        out_shape = shape or (config.MASK_H, config.MASK_W)
        return RunData(name, epoch, np.zeros(0, dtype=np.int64),
                       np.zeros((0, *out_shape), dtype=np.float32), [],
                       {k: empty_f for k in METRIC_KEYS}, np.zeros((0, 2)), {}, skipped, family,
                       out_shape, "no readable prediction files")

    sample_ids = np.concatenate(ids)
    preds = np.concatenate(masks).astype(np.float32)

    # A sample can be written more than once in an epoch (at ep0 the train split spans two
    # partial passes, so ~460 ids appear twice). Keep only the last write per sample --
    # otherwise duplicates are scored twice and skew the column's mean and std.
    if len(np.unique(sample_ids)) != len(sample_ids):
        keep = np.zeros(len(sample_ids), dtype=bool)
        # Later rows come from later batches, so reversing makes "first seen" the newest.
        _, first = np.unique(sample_ids[::-1], return_index=True)
        keep[len(sample_ids) - 1 - first] = True
        sample_ids, preds = sample_ids[keep], preds[keep]
        splits = [s for s, k in zip(splits, keep) if k]

    # Sample ids are not row indices into the ground truth (gastronorm starts at 000009,
    # and any dataset may be missing a sample). Map through gt.row_of, and drop
    # predictions for samples this dataset has no target for rather than indexing
    # something arbitrary -- a stray id would otherwise be scored against a real but
    # unrelated mask.
    gt_rows = np.array([gt.row_of.get(int(s), -1) for s in sample_ids], dtype=np.int64)
    if (gt_rows < 0).any():
        keep = gt_rows >= 0
        sample_ids, preds, gt_rows = sample_ids[keep], preds[keep], gt_rows[keep]
        splits = [s for s, k in zip(splits, keep) if k]

    if len(sample_ids) == 0:
        empty_f = np.zeros(0, dtype=np.float64)
        return RunData(name, epoch, sample_ids, preds, splits,
                       {k: empty_f for k in METRIC_KEYS}, np.zeros((0, 2)), {}, skipped, family,
                       (preds.shape[-2], preds.shape[-1]),
                       "no predicted sample belongs to this dataset")

    # Score against ground truth AT THIS RUN'S GRID. Runs trained at different
    # resolutions each get their own targets, so a 16x16 and a 30x30 column can sit side
    # by side; scoring both against one fixed size would broadcast-error or, worse,
    # compare a prediction to a target of a different shape.
    shape = (preds.shape[-2], preds.shape[-1])
    gt_masks = gt.masks_at(shape)
    if gt_masks is None:
        # Classification said targets exist at this size (has_shape globs real filenames)
        # but none could actually be decoded. Say so: this used to return an empty column
        # with no explanation, which reads exactly like a broken run.
        empty_f = np.zeros(0, dtype=np.float64)
        return RunData(name, epoch, np.zeros(0, dtype=np.int64),
                       np.zeros((0, *shape), dtype=np.float32), [],
                       {k: empty_f for k in METRIC_KEYS}, np.zeros((0, 2)), {}, skipped, family,
                       shape, f"no ground truth at {shape[0]}x{shape[1]}")

    pred = torch.from_numpy(preds)
    truth = torch.from_numpy(gt_masks[gt_rows])

    # pred is already sigmoid probabilities; viz never sees the logits, so bce is scored
    # from clamped probs (negligibly different from logit-space).
    loc = localization(pred, truth)
    metrics = {
        'bce': _F.binary_cross_entropy(pred.clamp(1e-6, 1 - 1e-6), truth, reduction='none').mean(dim=(-2, -1)).numpy(),
        'iou': soft_iou(pred, truth).numpy(),
        'contour': contour_f(pred, truth).numpy(),
        'mass': mass_error(pred, truth).numpy(),
        'localization': loc[0].numpy(), 'localization_x': loc[1].numpy(), 'localization_y': loc[2].numpy(),
    }
    com_pred = center_of_mass(pred, epsilon=config.EPSILON).numpy()

    # Keyed by ROW, not sample id: every consumer (render._cached, /api/mask.png,
    # /api/values, /api/neighbors) resolves the id through _sid() at the HTTP edge and
    # passes the row down. gt_rows is filtered in lockstep with sample_ids above, so
    # gt_rows[i] is the ground-truth row of sample_ids[i]. See SPEC.md 2.
    row_of = {int(r): i for i, r in enumerate(gt_rows)}
    return RunData(name, epoch, sample_ids, pred.numpy(), splits,
                   metrics, com_pred, row_of, skipped, family, shape)


# ***** registry *****


class Registry:
    """Startup loads ground truth and scans run dirs; prediction tensors are loaded
    lazily on first request and then cached for the process lifetime."""

    def __init__(self, experiment_dir: Path, runs_dir: Path):
        t0 = time.perf_counter()
        self.experiment_dir, self.runs_dir = experiment_dir, runs_dir
        self.gt = load_gt(experiment_dir)
        self._scanned_at = 0.0
        self.rescan()
        self._runs: dict[tuple[str, int | None], RunData] = {}
        n_ok = sum(e.compatible for e in self.entries)
        self.startup_s = time.perf_counter() - t0
        print(f"[viz] {experiment_dir.name} | layout '{self.gt.layout.name}' | "
              f"{len(self.gt)} samples | {n_ok} compatible / "
              f"{len(self.entries) - n_ok} incompatible runs | {self.startup_s:.2f}s")

    def rescan(self) -> None:
        """Re-read the runs directory so runs that finish while viz is open show up
        without a restart. Costs ~0.15s, and only probes one file per run."""
        self.entries = scan_runs(self.runs_dir, self.gt)
        self.by_name = {e.name: e for e in self.entries}
        self._scanned_at = time.monotonic()

    def maybe_rescan(self, max_age: float = config.RESCAN_SECONDS) -> None:
        if time.monotonic() - self._scanned_at > max_age:
            self.rescan()

    def defaults(self) -> list[str]:
        return [e.name for e in self.entries if e.compatible][: config.N_DEFAULT_RUNS]

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
            self._runs[key] = load_run(name, self.runs_dir, self.gt, entry.family, epoch,
                                       entry.shape)
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
        """Row for an untrusted sample id.

        Coerced to int and resolved through the id->row map, so no user string ever
        reaches a filesystem join. This is a LOOKUP, not a range check: ids do not start
        at zero on every dataset, so treating the id as the row silently served the wrong
        sample's mask and images.
        """
        i = self.gt.row_of.get(int(sid))
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
        if not 0 <= int(row) < len(self.gt):
            raise KeyError(row)
        return self.experiment_dir / "samples" / self.gt.sample_ids[int(row)]
