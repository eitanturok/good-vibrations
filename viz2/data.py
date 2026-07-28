"""Loading and indexing of ground truth and per-run predictions.

Nothing here runs inference: ground-truth masks are per-sample .npy files on disk and
predicted masks were already dumped by the OutputSaver callback during training. The
whole job is to join them on sample_id and compute per-sample metrics.
"""

import json
import os
import time
from dataclasses import dataclass, field
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


@dataclass
class GtIndex:
    sample_ids: list[str]        # zero-padded "000000".. , index == int(sample_id)
    masks: np.ndarray            # (N,20,40) float32, contiguous
    meta: list[dict]
    com_gt: np.ndarray           # (N,2) grid-space COM of the target mask
    avg_com: np.ndarray          # (N,2) full-res image coords, for the position scatter

    def __len__(self) -> int:
        return len(self.sample_ids)


def load_gt(experiment_dir: Path) -> GtIndex:
    sample_dirs = sorted((experiment_dir / "samples").iterdir())
    ids, masks, meta = [], [], []
    for d in sample_dirs:
        gt = d / config.GT_MASK_REL
        if not gt.exists():
            continue
        ids.append(d.name)
        masks.append(np.load(gt))
        meta.append(merge_metadata(d / "metadata.jsonl"))
    masks = np.ascontiguousarray(np.stack(masks).astype(np.float32))
    com_gt = np.asarray(center_of_mass(masks), dtype=np.float64)
    avg_com = np.asarray([m.get("avg_com", [-1.0, -1.0]) for m in meta], dtype=np.float64)
    return GtIndex(ids, masks, meta, com_gt, avg_com)


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
            result = ({"shape": tuple(mask.shape[-2:]) if mask is not None else None,
                       "info_keys": set(info)}, None)
            break
        except Exception:
            continue
    _PROBE_CACHE[key] = result
    return result


def _classify(name: str, run_dir: Path) -> RunEntry:
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
    # cleanly against experiment-25 ground truth and produce silently wrong metrics.
    if splits and not set(splits) <= config.EXP25_EVAL_SPLITS:
        preview = ", ".join(splits[:3])
        return RunEntry(name, False, f"different dataset (eval splits: {preview})", eval_splits=splits)

    family = "experiment-25" if splits else "unknown"
    # Recency comes from the highest-epoch file rather than a stat() of every .pt: epochs
    # are written in order, so it ranks runs identically at a fraction of the cost.
    newest = files[0]
    return RunEntry(name, True, None, newest.stat().st_mtime, _epoch_of(newest), splits, family)


def scan_runs(runs_dir: Path) -> list[RunEntry]:
    entries = []
    for d in sorted(runs_dir.iterdir()):
        if not d.is_dir():
            continue
        e = _classify(d.name, d)
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


def load_run(name: str, runs_dir: Path, gt: GtIndex, family: str = "unknown") -> RunData:
    """Load one run's final-epoch predictions and score every sample against the target.

    Metrics mirror the training loop exactly (see src/model/arch.py mses/com_distances
    and utils/metrics.soft_iou), which is what lets the column headers be cross-checked
    against the numbers in the run's own logs-rank0.txt.
    """
    outputs = runs_dir / name / config.OUTPUTS_SUBDIR
    ids, masks, splits, skipped = [], [], [], []
    epoch = 0

    for split, d in _split_dirs(outputs):
        files = sorted(d.glob("ep*.pt"), key=_epoch_of)
        if not files:
            continue
        last = _epoch_of(files[-1])
        epoch = max(epoch, last)
        for p in [f for f in files if _epoch_of(f) == last]:  # train has one file per batch
            try:
                obj = torch.load(p, map_location="cpu", weights_only=False)
            except Exception:
                skipped.append(str(p.relative_to(outputs)))
                continue
            sid = _as_int_array(obj["info"]["sample_id"])
            ids.append(sid)
            masks.append(obj["mask_pred"].float().numpy())
            splits += [split] * len(sid)

    if not ids:
        empty_f = np.zeros(0, dtype=np.float64)
        return RunData(name, epoch, np.zeros(0, dtype=np.int64), np.zeros((0, config.MASK_H, config.MASK_W),
                       dtype=np.float32), [], empty_f, empty_f, empty_f,
                       np.zeros((0, 2)), {}, skipped, family)

    sample_ids = np.concatenate(ids)
    pred = torch.from_numpy(np.concatenate(masks).astype(np.float32))
    truth = torch.from_numpy(gt.masks[sample_ids])

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
        self._runs: dict[str, RunData] = {}
        n_ok = sum(e.compatible for e in self.entries)
        self.startup_s = time.perf_counter() - t0
        print(f"[viz2] {len(self.gt)} samples | {n_ok} compatible / "
              f"{len(self.entries) - n_ok} incompatible runs | {self.startup_s:.2f}s")

    def rescan(self) -> None:
        """Re-read the runs directory so runs that finish while viz2 is open show up
        without a restart. Costs ~0.15s, and only probes one file per run."""
        self.entries = scan_runs(self.runs_dir)
        self.by_name = {e.name: e for e in self.entries}
        self._scanned_at = time.monotonic()

    def maybe_rescan(self, max_age: float = config.RESCAN_SECONDS) -> None:
        if time.monotonic() - self._scanned_at > max_age:
            self.rescan()

    def defaults(self) -> list[str]:
        return [e.name for e in self.entries if e.compatible][: config.N_DEFAULT_RUNS]

    def run(self, name: str, reload: bool = False) -> RunData:
        entry = self.by_name.get(name)
        if entry is None:
            # A run requested but not in the cached scan may have just appeared.
            self.rescan()
            entry = self.by_name.get(name)
        if entry is None or not entry.compatible:
            raise KeyError(name)
        if reload:
            self._runs.pop(name, None)
        if name not in self._runs:
            self._runs[name] = load_run(name, self.runs_dir, self.gt, entry.family)
        return self._runs[name]

    def sample_index(self, sid) -> int:
        """Validate an untrusted sample id: coerced to int and range-checked so no
        user string ever reaches a filesystem join."""
        i = int(sid)
        if not 0 <= i < len(self.gt):
            raise KeyError(sid)
        return i

    def sample_dir(self, sid) -> Path:
        return self.experiment_dir / "samples" / self.gt.sample_ids[self.sample_index(sid)]
