"""Loading and indexing of ground truth and per-run predictions.

Nothing here runs inference: ground-truth masks are per-sample .npy files on disk and
predicted masks were already dumped by the OutputSaver callback during training. The
whole job is to join them on sample_id and compute per-sample metrics.
"""

import json
import os
import time
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch

from viz2 import config
from utils.metrics import center_of_mass, soft_iou

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

    def __len__(self) -> int:
        return len(self.sample_ids)


def load_gt(experiment_dir: Path) -> GtIndex:
    samples_dir = experiment_dir / "samples"
    layout = config.Layout.detect(samples_dir)
    sample_dirs = sorted(p for p in samples_dir.iterdir() if p.is_dir())
    ids, masks, meta = [], [], []
    for d in sample_dirs:
        gt = d / layout.gt_mask
        if not gt.exists():
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
        raise SystemExit(
            f"[viz2] no {config.MASK_H}x{config.MASK_W} ground-truth masks under "
            f"{samples_dir} (layout '{layout.name}', expected {layout.gt_mask})."
        )
    masks = np.ascontiguousarray(np.stack(masks).astype(np.float32))
    com_gt = np.asarray(center_of_mass(masks), dtype=np.float64)
    avg_com = np.asarray([parse_com(m.get("avg_com")) for m in meta], dtype=np.float64)
    row_of = {int(s): i for i, s in enumerate(ids)}
    return GtIndex(ids, masks, meta, com_gt, avg_com, layout, row_of)


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
    for p in files[:5]:
        try:
            obj = torch.load(p, map_location="cpu", weights_only=False)
            mask, info = obj.get("mask_pred"), obj.get("info") or {}
            # Keep only the few facts classification needs; holding the tensors would
            # pin hundreds of MB across every scanned run for no benefit.
            sid = info.get("sample_id")
            result = ({"shape": tuple(mask.shape[-2:]) if mask is not None else None,
                       "info_keys": set(info),
                       # A few ids are enough to tell which dataset a run was trained on
                       # when its split names carry no whitelist.
                       "sample_ids": _as_int_array(sid).tolist() if sid is not None else []},
                      None)
            break
        except Exception:
            continue
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
    if shape != (config.MASK_H, config.MASK_W):
        return RunEntry(name, False, f"mask shape {shape}, expected {(config.MASK_H, config.MASK_W)}")

    if not {"sample_id", "x_com"} <= obj["info_keys"]:
        return RunEntry(name, False, "legacy info schema")

    splits = [p.name for p in _eval_dirs(outputs)]

    # Sample ids collide across experiments, so a run from another dataset would join
    # cleanly against the loaded ground truth and produce silently wrong metrics.
    # Where the dataset declares its split names, that is the strongest available signal.
    # Keyed on layout.dataset, NOT layout.name: one dataset can be read through several
    # layouts (the same capture at two mask sizes), and keying on the filename scheme
    # would apply the wrong whitelist and reject every run as "different dataset".
    allowed = config.EVAL_SPLITS.get(gt.layout.dataset)
    if allowed is not None:
        if splits and not set(splits) <= allowed:
            preview = ", ".join(splits[:3])
            return RunEntry(name, False, f"different dataset (eval splits: {preview})",
                            eval_splits=splits)
    else:
        # No whitelist for this dataset: fall back to requiring that EVERY sample id in
        # the probe exists here. Weaker than matching split names -- two datasets with
        # overlapping id ranges still pass -- so it is a backstop for a not-yet-declared
        # dataset, not a substitute for adding one to config.EVAL_SPLITS.
        probe_ids = obj["sample_ids"]
        if probe_ids and not all(int(s) in gt.row_of for s in probe_ids):
            return RunEntry(name, False, "different dataset (sample ids not in this dataset)",
                            eval_splits=splits)

    family = gt.layout.dataset if splits else "unknown"
    # Recency comes from the highest-epoch file rather than a stat() of every .pt: epochs
    # are written in order, so it ranks runs identically at a fraction of the cost.
    newest = files[0]
    return RunEntry(name, True, None, newest.stat().st_mtime, _epoch_of(newest), splits, family)


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
    mse: np.ndarray
    iou: np.ndarray
    comdist: np.ndarray
    com_pred: np.ndarray         # (M,2) grid-space, for display
    row_of: dict[int, int]
    skipped_files: list[str]
    family: str = "unknown"


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
    while eps and (name, eps[-1]) not in _DEAD_EPOCHS:
        newest = eps[-1]
        if any(_loadable(p) for p in files if _epoch_of(p) == newest):
            break
        _DEAD_EPOCHS.add((name, newest))
        eps.pop()

    return [e for e in eps if (name, e) not in _DEAD_EPOCHS]


@lru_cache(maxsize=4096)
def _loadable(path: Path) -> bool:
    """Whether a prediction file can actually be read. Cached: a file that parsed once
    will not stop parsing, and a truncated one will not start."""
    try:
        torch.load(path, map_location="cpu", weights_only=False)
        return True
    except Exception:
        return False


def load_run(name: str, runs_dir: Path, gt: GtIndex, family: str = "unknown",
             epoch: int | None = None) -> RunData:
    """Load one run's predictions and score every sample against the target.

    `epoch` selects which saved epoch to read; the default (None) uses each split's
    latest, which is what the table shows.

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
                return load_run(name, runs_dir, gt, family, epoch=max(usable))
        empty_f = np.zeros(0, dtype=np.float64)
        return RunData(name, epoch, np.zeros(0, dtype=np.int64), np.zeros((0, config.MASK_H, config.MASK_W),
                       dtype=np.float32), [], empty_f, empty_f, empty_f,
                       np.zeros((0, 2)), {}, skipped, family)

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
        return RunData(name, epoch, sample_ids, preds, splits, empty_f, empty_f,
                       empty_f, np.zeros((0, 2)), {}, skipped, family)

    pred = torch.from_numpy(preds)
    truth = torch.from_numpy(gt.masks[gt_rows])

    # mask_pred is already sigmoid probabilities -- do not re-sigmoid.
    mse = (pred - truth).square().mean(dim=(-2, -1)).numpy()
    iou = soft_iou(pred, truth).numpy()
    com_p = center_of_mass(pred, normalize=True, epsilon=config.EPSILON)
    com_t = center_of_mass(truth, normalize=True, epsilon=config.EPSILON)
    comdist = torch.linalg.norm(com_p - com_t, ord=2, dim=-1).numpy()
    com_pred = center_of_mass(pred, epsilon=config.EPSILON).numpy()

    # An empty ground-truth mask has no center of mass, so the distance to it is
    # meaningless rather than large. The training metric skips these samples
    # (CenterOfMassDistance.update filters mask_true.sum() > 0); marking them undefined
    # keeps the aggregates identical AND stops degenerate empty boxes from monopolising
    # the top of a "worst COM distance" sort, which is the tool's main workflow.
    comdist = np.where(truth.sum(dim=(-2, -1)).numpy() > 0, comdist, np.nan)

    row_of = {int(s): i for i, s in enumerate(sample_ids)}
    return RunData(name, epoch, sample_ids, pred.numpy(), splits,
                   mse, iou, comdist, com_pred, row_of, skipped, family)


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
        print(f"[viz2] {experiment_dir.name} | layout '{self.gt.layout.name}' | "
              f"{len(self.gt)} samples | {n_ok} compatible / "
              f"{len(self.entries) - n_ok} incompatible runs | {self.startup_s:.2f}s")

    def rescan(self) -> None:
        """Re-read the runs directory so runs that finish while viz2 is open show up
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
            self._runs[key] = load_run(name, self.runs_dir, self.gt, entry.family, epoch)
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
