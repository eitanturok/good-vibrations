"""Loading and indexing of ground truth and per-run predictions.

Nothing here runs inference: ground-truth masks are per-sample .npy files on disk and
predicted masks were already dumped by the OutputSaver callback during training. The
whole job is to join them on sample_id and compute per-sample metrics.
"""

import json
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


def _epoch_of(p: Path) -> int:
    return int(p.stem.split("-")[0].removeprefix("ep"))


def _as_int_array(v) -> np.ndarray:
    """info['sample_id'] is a tensor on most runs but a plain list on some."""
    if torch.is_tensor(v):
        v = v.tolist()
    return np.asarray(v, dtype=np.int64)


def _probe(outputs: Path) -> tuple[dict | None, str | None]:
    """Load the newest readable prediction file to inspect its schema.

    Several .pt files on disk are truncated and raise PytorchStreamReader errors, so
    this walks back through the newest few rather than trusting a single probe --
    probing only one misclassifies runs whose newest file happens to be corrupt.
    """
    files = sorted(outputs.rglob("ep*.pt"), key=_epoch_of, reverse=True)
    if not files:
        return None, "no prediction files"
    for p in files[:5]:
        try:
            return torch.load(p, map_location="cpu", weights_only=False), None
        except Exception:
            continue
    return None, "all recent prediction files unreadable"


def _classify(name: str, run_dir: Path) -> RunEntry:
    outputs = run_dir / config.OUTPUTS_SUBDIR
    if not outputs.is_dir():
        return RunEntry(name, False, "no outputs_history/ (older run format)")

    obj, err = _probe(outputs)
    if obj is None:
        return RunEntry(name, False, err)

    mask = obj.get("mask_pred")
    if mask is None:
        return RunEntry(name, False, "no mask_pred in payload")
    shape = tuple(mask.shape[-2:])
    if shape != (config.MASK_H, config.MASK_W):
        return RunEntry(name, False, f"mask shape {shape}, expected {(config.MASK_H, config.MASK_W)}")

    info = obj.get("info") or {}
    if "sample_id" not in info or "x_com" not in info:
        return RunEntry(name, False, "legacy info schema")

    eval_dir = outputs / "eval"
    splits = sorted(p.name for p in eval_dir.iterdir() if p.is_dir()) if eval_dir.is_dir() else []

    # Sample ids collide across experiments, so a run from another dataset would join
    # cleanly against experiment-25 ground truth and produce silently wrong metrics.
    if splits and not set(splits) <= config.EXP25_EVAL_SPLITS:
        preview = ", ".join(splits[:3])
        return RunEntry(name, False, f"different dataset (eval splits: {preview})", eval_splits=splits)

    family = "experiment-25" if splits else "unknown"
    mtime = max((p.stat().st_mtime for p in outputs.rglob("*.pt")), default=0.0)
    return RunEntry(name, True, None, mtime, _epoch_of(sorted(outputs.rglob("ep*.pt"), key=_epoch_of)[-1]),
                    splits, family)


def scan_runs(runs_dir: Path) -> list[RunEntry]:
    entries = [_classify(d.name, d) for d in sorted(runs_dir.iterdir()) if d.is_dir()]
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
    out = []
    if (outputs / "train").is_dir():
        out.append(("train", outputs / "train"))
    eval_dir = outputs / "eval"
    if eval_dir.is_dir():
        out += [(p.name, p) for p in sorted(eval_dir.iterdir()) if p.is_dir()]
    return out


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
        self.entries = scan_runs(runs_dir)
        self.by_name = {e.name: e for e in self.entries}
        self._runs: dict[str, RunData] = {}
        n_ok = sum(e.compatible for e in self.entries)
        self.startup_s = time.perf_counter() - t0
        print(f"[viz2] {len(self.gt)} samples | {n_ok} compatible / "
              f"{len(self.entries) - n_ok} incompatible runs | {self.startup_s:.2f}s")

    def defaults(self) -> list[str]:
        return [e.name for e in self.entries if e.compatible][: config.N_DEFAULT_RUNS]

    def run(self, name: str) -> RunData:
        entry = self.by_name.get(name)
        if entry is None or not entry.compatible:
            raise KeyError(name)
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
