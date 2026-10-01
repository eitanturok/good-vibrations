"""Loading and numpy. Everything is derived from the dataset dir -- nothing hardcoded.

Sample ids stay strings end to end; viz2 never builds a cross-sample array, so unlike viz/
there is no row space and no id->row conversion to get wrong.
"""

import json
import math
import os
import random
import threading
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.signal import find_peaks, resample, savgol_filter

from utils.ids import meta_sample_name

# Older experiments name the FFT file singular; everything else about the payload matches.
FFTS = ["vibration/04_ffts.npz", "vibration/04_fft.npz"]
SHIFTS = {"clean": "vibration/03_clean_shifts.npy", "raw": "vibration/02_raw_shifts.npy"}
PHOTOS = ["image/02_cropped_overhead.png", "image/01_cropped.png"]
MASKS = ["image/03_smask.npy", "image/02_smask.npy"]

ROOT: Path | None = None            # what init() was pointed at; rescan_datasets() rewalks it
DATASETS: dict[str, Path] = {}      # dataset name -> its dir (the one holding samples/)
COUNTS: dict[str, int] = {}         # dataset name -> its sample count
CURRENT: str = ""                   # which dataset is loaded right now
MAX_OVERHEAD = [1, 1]               # largest cropped-overhead [w, h] over ALL datasets,
                                    # so the client can size every box against the biggest

DIRS: dict[str, Path] = {}   # sample id -> dir; the only id->path map
META: dict[str, dict] = {}
# sample id -> (mtime_ns, size) of its metadata.jsonl when META was last built from it.
# rescan() runs on every client poll (0.5s, per open tab); re-parsing every sample's
# metadata each time cost ~0.75s of GIL-holding Python at 6.3k samples, so overlapping
# polls kept the server pegged and starved the gallery's thumbnail requests. A stat is
# ~50x cheaper, and a reprocessed-in-place sample still rewrites metadata.jsonl.
STAMP: dict[str, tuple[int, int]] = {}
_RESCAN = threading.Lock()   # polls from several tabs must not rescan concurrently
INFO: dict = {}
# Bumped whenever rescan/rescan_datasets/_load actually change a sample -- folded into the
# client's cache-busting `rv` (see app.py::_payload) so a sample deleted and recaptured
# under the SAME id gets a fresh image URL. Without this, thumb/scene/mask are served with
# Cache-Control: immutable, so the browser keeps showing whatever photo it first fetched
# for that id forever -- looking exactly like "another sample's" image, because it is one:
# whatever used to be at that id before the redo.
EPOCH = 0


def _meta(d: Path) -> dict:
    """metadata.jsonl holds ONE KEY PER LINE, so lines must be merged."""
    m = {}
    for line in (d / "metadata.jsonl").read_text().splitlines():
        if line.strip():
            m.update(json.loads(line))
    return m


def com(v) -> list[float]:
    """[y, x] from either a JSON list or gastronorm's str(ndarray) '[603.1 901.2]'."""
    if isinstance(v, str):
        v = v.replace("[", " ").replace("]", " ").replace(",", " ").split()
    try:
        f = np.asarray(v, dtype=float).reshape(-1)
        return [float(f[0]), float(f[1])]
    except Exception:
        return [-1.0, -1.0]


def coms(v) -> list[list[float]]:
    """Per-object [row, col] centres, in the same overhead-pixel space as the smask.

    metadata's `coms` nests differently by dataset -- [[ [r,c], [r,c] ]] or one wrapped
    list per object, [ [[r,c]], [[r,c]] ] -- so every [r, c] pair is collected at any depth.
    An empty box is the sentinel [-1, -1], which is dropped."""
    out = []

    def walk(x):
        if isinstance(x, list) and len(x) == 2 and all(isinstance(e, (int, float)) for e in x):
            if [float(x[0]), float(x[1])] != [-1.0, -1.0]:
                out.append([float(x[0]), float(x[1])])
        elif isinstance(x, list):
            for e in x:
                walk(e)
    walk(v)
    return out


def _first(d: Path, names):
    return next((n for n in names if (d / n).exists()), None)


def _fft(d: Path) -> Path | None:
    n = _first(d, FFTS)
    return d / n if n else None


def sample_photo(sid: str) -> Path | None:
    p = _first(d(sid), PHOTOS)
    return d(sid) / p if p else None


def sample_mask(sid: str) -> Path | None:
    """Resolved per sample, like sample_photo: a dataset's first sample can be missing its
    image/ dir (four_objs_part_1's 000001-4), so one dataset-wide filename taken from it
    would be None and blank out every other sample's photo and mask too."""
    p = _first(d(sid), MASKS)
    return d(sid) / p if p else None


@lru_cache(maxsize=4)
def mask_areas(name: str, epoch: int) -> dict[str, int]:
    """Every loaded sample's segmented area in px, for the gallery's "mask area" sort.
    Read from the smask's tiny PNG twin (~1.6 s for 6k samples on 8 threads), so it is only
    computed when asked for; name/epoch key the cache to the loaded dataset."""
    from concurrent.futures import ThreadPoolExecutor

    def area(sid):          # -1: no image / no mask, so it sorts ahead of every real area
        mp = sample_mask(sid)
        if not mp or not sample_photo(sid):
            return -1
        png = mp.with_suffix(".png")
        m = np.asarray(Image.open(png)) if png.exists() else np.load(mp) > 0.5
        return int(np.count_nonzero(m))
    sids = list(DIRS)
    with ThreadPoolExecutor(8) as ex:
        return dict(zip(sids, ex.map(area, sids)))


@lru_cache(maxsize=32)
def object_masks(sid: str) -> list[tuple[str, np.ndarray]]:
    """One boolean mask per object from image/smasks/<type><i>.npy (experiment-25's
    all.npy / empty_box*.npy there are not objects)."""
    sd = d(sid) / "image" / "smasks"
    fs = [f for f in sorted(sd.glob("*.npy")) if f.stem != "all" and not f.stem.startswith("empty")]
    ms = [(f.stem, np.load(f) > 0.5) for f in fs]
    return [(n, m) for n, m in ms if m.any()]

def _glob1(base: Path | None, *globs: str) -> Path | None:
    """First file under `base` matching any of the glob patterns, in order."""
    if not base or not base.is_dir():
        return None
    for g in globs:
        hit = sorted(base.glob(g))
        if hit:
            return hit[0]
    return None


def recovered_wav(sid: str) -> Path | None:
    """The pre-rendered recovered-audio wav (fixed laser/channel, e.g. 55/x)."""
    return _glob1(DIRS.get(sid), "recovered_audio.wav", "vibration/*recovered_audio*.wav")


def recovered_video(sid: str) -> Path | None:
    """Spectrogram video of the recovered signal -- a fixed laser/x pre-render, so it does
    NOT track the laser/channel selector the way viz2's synthesised playback does."""
    return _glob1(DIRS.get(sid), "vibration/*spectrogram*.mp4")


def _audio_roots():
    """Every data/audio/ dir found walking up from the cwd and the loaded dataset."""
    seen: list[Path] = []
    for start in (Path.cwd(), DATASETS.get(CURRENT, Path.cwd())):
        for up in (start, *start.parents):
            cand = up / "data" / "audio"
            if cand not in seen and cand.is_dir():
                seen.append(cand)
    return seen


def _chirp_key(f_start, f_end, T_sec, T_start, T_end, fs):
    try:
        return tuple(round(float(v), 6) for v in (f_start, f_end, T_sec, T_start, T_end, fs))
    except (TypeError, ValueError):
        return None


@lru_cache(maxsize=8)
def _chirp_index(root: Path) -> dict:
    """chirp params -> stimulus dir, from each data/audio/<name>/metadata.jsonl. Several
    dirs can hold the same chirp (chirp_100_1000_1sec vs _1.0sec); the first sorted wins."""
    out = {}
    for sd in sorted(root.iterdir()):
        try:
            m = _meta(sd)
        except OSError:
            continue
        k = _chirp_key(m.get("f_start"), m.get("f_end"), m.get("T_sec"),
                       m.get("T_start"), m.get("T_end"), m.get("fs"))
        if k and (sd / "audio.wav").exists():
            out.setdefault(k, sd)
    return out


def _stim_dir(sid: str) -> Path | None:
    """Local data/audio/<name>/ for the stimulus this sample played. Older samples record
    only the capture machine's absolute audio_dir, so we match on its basename; newer ones
    (e.g. four_objs) drop audio_dir and record the chirp's params inline (chirp_f_start,
    chirp_t_sec, audio_fs, ...), so we match those against each stimulus's metadata."""
    raw = (META.get(sid) or {}).get("audio_dir") or ""
    name = raw.replace("\\", "/").rstrip("/").split("/")[-1]
    roots = _audio_roots()
    if name:
        for r in roots:
            if (r / name).is_dir():
                return r / name
        return None
    m = _meta(d(sid)) if sid in DIRS else {}
    k = _chirp_key(m.get("chirp_f_start"), m.get("chirp_f_end"), m.get("chirp_t_sec"),
                   m.get("chirp_t_start"), m.get("chirp_t_end"), m.get("audio_fs"))
    if not k:
        return None
    return next((_chirp_index(r)[k] for r in roots if k in _chirp_index(r)), None)


def source_wav(sid: str) -> Path | None:
    """The stimulus wav that was played (the 'original' audio)."""
    return _glob1(_stim_dir(sid), "audio.wav", "*.wav")


def source_video(sid: str) -> Path | None:
    """Spectrogram video of the played stimulus."""
    return _glob1(_stim_dir(sid), "spectrogram.mp4", "*.mp4")


def stim_name(sid: str) -> str | None:
    """Which stimulus this sample played (its data/audio/<name>/). Shared by every sample
    that played the same chirp, so the client can key the original video on it and not
    reload an identical video on every gallery step."""
    sd = _stim_dir(sid)
    return sd.name if sd else None


def stim_video(name: str) -> Path | None:
    """source_video() by stimulus name rather than sample id. Only names that are real
    stimulus dirs resolve -- never a path the request made up."""
    if not name or "/" in name or "\\" in name or name.startswith("."):
        return None
    for r in _audio_roots():
        if (r / name).is_dir():
            return _glob1(r / name, "spectrogram.mp4", "*.mp4")
    return None


def stim_params(sid: str) -> dict:
    """The chirp's own frequency/time-padding params (src/data/audio.py), straight from
    its metadata.jsonl -- distinct from the sample's own metadata.jsonl."""
    sd = _stim_dir(sid)
    if not sd or not (sd / "metadata.jsonl").exists():
        return {}
    m = _meta(sd)
    return {k: m[k] for k in ("f_start", "f_end", "T_start", "T_end") if k in m}


@lru_cache(maxsize=32)
def box_photo(name: str) -> Path | None:
    """A cropped-overhead shot that stands for a whole dataset: an empty-box sample's if
    the dataset has one (so the picker shows the bare box), otherwise the first sample's."""
    first = None
    for sd in sorted((DATASETS[name] / "samples").iterdir()):
        p = _first(sd, PHOTOS)
        if not p:
            continue
        if first is None:
            first = sd / p
        if _meta(sd).get("is_empty_box"):
            return sd / p
    return first


def _sample_count(ds: Path) -> int:
    """How many samples with an FFT the dataset holds -- 0 means viz2 cannot open it
    (older-format datasets like experiment-25 land here with no special case)."""
    sm = ds / "samples"
    return sum(1 for d in sm.iterdir() if _fft(d)) if sm.is_dir() else 0


def summarize(name: str) -> dict:
    """Positions / speakers / object types in a dataset -- metadata.jsonl only, no FFT or
    image loads, so this is cheap enough to run over every dataset at boot."""
    positions, speakers, objects = set(), set(), set()
    n = 0
    for d in sorted((DATASETS[name] / "samples").iterdir()):
        if not _fft(d):
            continue
        m = _meta(d)
        positions.add(int(m.get("position_id") or 0))
        speakers.add(int(m.get("speaker") or 0))
        objects.update((m.get("objects") or {}).keys())
        n += 1
    return {"n": n, "positions": sorted(positions), "speakers": sorted(speakers), "objects": sorted(objects)}


def _overhead_size(ds: Path):
    """[w, h] of the first sample's cropped-overhead photo -- the box's real pixel size,
    which is what makes one box render bigger than another. Falls back to [1, 1]."""
    for d in sorted((ds / "samples").iterdir()):
        p = _first(d, PHOTOS)
        if p:
            with Image.open(d / p) as im:
                return list(im.size)          # PIL .size is (w, h)
    return [1, 1]


def init(root: Path) -> int:
    """Point viz2 at either one dataset dir (has samples/) or a parent dir of them.

    Every dataset becomes a pickable box; the most recently modified one (i.e. whichever a
    watcher is actively dropping samples into) is loaded now, the rest on demand via
    switch(). Returns the dataset count."""
    global ROOT
    ROOT = Path(root)
    rescan_datasets()
    if not DATASETS:
        raise SystemExit(f"no loadable datasets (a samples/ dir with an FFT) under {root}")
    first = max(DATASETS, key=lambda n: (DATASETS[n] / "samples").stat().st_mtime)
    _load(first)
    return len(DATASETS)


def switch(name: str) -> int:
    """Load a different dataset. Returns its sample count."""
    if name not in DATASETS:
        raise KeyError(name)
    with _RESCAN:                 # not mid-way through a poll's rescan of the old dataset
        _load(name)
    return len(DIRS)


def rescan_datasets() -> int:
    """Reconcile DATASETS against what's under ROOT: pick up new experiment dirs, and drop
    ones whose dir (or samples/) is just gone -- a whole box scrapped, not just one bad
    sample. Removal is a directory-existence check, NOT a full _sample_count() walk -- that
    would re-stat every sample in every dataset on every poll tick, which is the whole
    dataset tree, every 0.5s. If CURRENT was the one that vanished, falls back to another
    loaded dataset, or clears the view if none are left. Returns how many datasets changed."""
    try:
        cands = {ROOT.name: ROOT} if (ROOT / "samples").is_dir() else {c.name: c for c in ROOT.iterdir()}
    except OSError:
        cands = {}
    changed = 0
    for name in [n for n in DATASETS if not (cands.get(n) and (cands[n] / "samples").is_dir())]:
        del DATASETS[name]; COUNTS.pop(name, None)
        changed += 1
    for name, c in cands.items():
        if name in DATASETS:
            continue
        cnt = _sample_count(c)
        if not cnt:
            continue  # not loadable yet -- no sample has finished processing
        DATASETS[name] = c
        COUNTS[name] = cnt
        ow, oh = _overhead_size(c)
        MAX_OVERHEAD[:] = [max(MAX_OVERHEAD[0], ow), max(MAX_OVERHEAD[1], oh)]
        changed += 1
    if CURRENT and CURRENT not in DATASETS:           # the loaded dataset was the one dropped
        if DATASETS:
            _load(next(iter(DATASETS)))
        else:
            DIRS.clear(); META.clear(); INFO.clear(); STAMP.clear()
        changed += 1
    if changed:
        global EPOCH
        EPOCH += 1
        box_photo.cache_clear()   # keyed by dataset name -- a redone dataset needs a fresh pick
    return changed


def rescan() -> int:
    """Reconcile DIRS/META for the current dataset against what's on disk: pick up samples
    the watcher finished writing, drop ones whose directory is gone -- e.g. a bad capture
    deleted mid-collection -- and refresh META for samples reprocessed in place (same id,
    new metadata.jsonl, detected via STAMP) so a stale label doesn't survive a resample.
    Returns how many samples changed either way. A poll that arrives while another is
    already rescanning just skips -- the next one picks up whatever it would have."""
    if CURRENT not in DATASETS or not _RESCAN.acquire(blocking=False):
        return 0
    try:
        return _rescan()
    finally:
        _RESCAN.release()


def _stamp(d: Path):
    try:
        st = (d / "metadata.jsonl").stat()
        return (st.st_mtime_ns, st.st_size)
    except OSError:
        return None


def _rescan() -> int:
    try:
        on_disk = {e.name: Path(e.path) for e in os.scandir(DATASETS[CURRENT] / "samples")}
    except OSError:
        on_disk = {}
    changed = 0
    for sid in [s for s in DIRS if s not in on_disk]:
        del DIRS[sid]; del META[sid]; STAMP.pop(sid, None)
        changed += 1
    for name, d in on_disk.items():
        st = _stamp(d)
        if name in DIRS and st == STAMP.get(name):
            continue
        if st is None or not _fft(d):
            continue
        m = _meta_row(name, d, CURRENT)
        STAMP[name] = st
        if name not in DIRS or META[name] != m:
            DIRS[name] = d
            META[name] = m
            changed += 1
    if changed:
        global EPOCH
        COUNTS[CURRENT] = len(DIRS)   # keep the box picker's count for the live dataset honest
        EPOCH += 1
        # fft()/shifts() are lru_cached by sid alone; _load() clears them on a full dataset
        # switch, but a same-dataset redo goes through here instead -- without this, a
        # sample deleted and recaptured under the same id keeps serving its OLD spectrum/
        # mode/probe data from cache even after the image itself (fixed via EPOCH) refreshes.
        fft.cache_clear()
        shifts.cache_clear()
        box_photo.cache_clear()   # the redone sample might be this dataset's box-picker photo
    return changed


def _meta_row(sid: str, d: Path, name: str) -> dict:
    m = _meta(d)
    return {
        "id": sid,
        "name": meta_sample_name(m) or sid,          # the universal id shown, e.g. 000012-3 (utils/ids.py)
        "pos": int(m.get("position_id") or m.get("output_id") or 0),  # int here, string on exp-25 (which calls it output_id)
        "spk": int(m.get("speaker") or 0),
        "layout": m.get("layout") or "",
        "n": int(m.get("n_objects") or 0),
        # distinct object TYPES in the box, regardless of how many of each
        "objects": sorted((m.get("objects") or {}).keys()),
        "objcounts": dict(sorted((m.get("objects") or {}).items())),  # type -> how many
        "empty": bool(m.get("is_empty_box")),
        "com": com(m.get("avg_com")),
        "coms": coms(m.get("coms")),                  # per-object [row, col], overhead px
        "box": m.get("box") or name,                 # carried onto every pinned probe
        # capture-machine path to the played stimulus; only the basename survives here,
        # matched back to a local data/audio/<name>/ tree by _stim_dir().
        "audio_dir": m.get("audio_dir") or "",
    }


def _load(name: str) -> None:
    """(Re)populate DIRS / META / INFO for one dataset. Keep only samples that really
    have an FFT -- this is what drops gastronorm's 000009 (images but no vibration data),
    with no special case."""
    global CURRENT, EPOCH
    CURRENT = name
    EPOCH += 1
    ds = DATASETS[name]
    DIRS.clear(); META.clear()
    fft.cache_clear(); shifts.cache_clear(); peaks.cache_clear()

    for d in sorted((ds / "samples").iterdir()):
        if _fft(d):
            DIRS[d.name] = d
    if not DIRS:
        raise SystemExit(f"no samples with an FFT under {ds}/samples")

    STAMP.clear()
    for sid, d in DIRS.items():
        STAMP[sid] = _stamp(d)
        META[sid] = _meta_row(sid, d, name)

    d0 = DIRS[next(iter(DIRS))]
    m0 = _meta(d0)
    z = np.load(_fft(d0))
    n_lasers = z["fft"].shape[1]
    rows = m0.get("n_rows") or m0.get("n_laser_rows") or int(round(n_lasers ** 0.5))
    ow, oh = _overhead_size(ds)
    INFO.clear()
    INFO.update(
        dataset=name, box=m0.get("box") or name,
        rows=int(rows), cols=int(n_lasers // int(rows)), n_lasers=int(n_lasers),
        fps=float(m0.get("fps") or 2500), n_samples=int(z["n_samples"]),
        min_freq=float(m0.get("min_freq") or 50), max_freq=float(m0.get("max_freq") or 1000),
        # first sample that HAS one -- informational only; endpoints resolve per sample
        photo=next((p for p in map(lambda x: _first(x, PHOTOS), DIRS.values()) if p), None),
        mask=next((p for p in map(lambda x: _first(x, MASKS), DIRS.values()) if p), None),
        overhead=[ow, oh], max_overhead=list(MAX_OVERHEAD),
    )
    INFO["scale"] = _scales()


# Signed quantities vary ~5x between samples and log magnitude shifts by ~0.7 decades, so
# per-sample ranges silently rescale every axis as you browse -- identical curve heights
# would then mean different physical values. These global ranges make every plot comparable.
SCALE_N = 160          # measured: converged and stable across seeds by ~80; 160 costs ~1 s


def _scales(n=SCALE_N):
    """One range per quantity, shared by every sample.

    Taken over a random subset (a full scan is not worth ~20 s at boot) and reduced with a
    percentile rather than min/max, so one freak recording cannot stretch every axis in
    the app. Signed quantities still reduce per sample then across; log magnitude pools
    first -- see the note below.
    """
    ids = sorted(DIRS)
    pick = ids if len(ids) <= n else random.Random(0).sample(ids, n)
    lm_all, sig, sh, md = [], [], [], []
    for sid in pick:
        z = chan(fft(sid)[0], "avg")
        # Log magnitude is POOLED across samples rather than reduced per sample first.
        # Taking p99 within a sample and then p90 across them clipped twice over: the top
        # of the range landed below the median sample's own peak, so the tallest resonance
        # ran off the frame on most recordings. One percentile over the pooled values
        # spends the budget where the data actually is.
        lm_all.append(logmag(np.abs(z)).ravel())
        sig.append(np.percentile(np.abs(z.real), 99))
        sig.append(np.percentile(np.abs(z.imag), 99))
        sh.append(np.percentile(np.abs(chan(shifts(sid, "clean"), "avg")), 99))
        md.append(np.abs(z).max())          # peak modal displacement of the sample
    fft.cache_clear()
    shifts.cache_clear()
    m, t = float(np.percentile(sig, 90)), float(np.percentile(sh, 90))
    mode = float(np.percentile(md, 90))
    # Both ends are the extreme actually observed, not a percentile. The tails here are
    # thin enough that trimming buys almost no vertical space while cutting real curve off
    # the frame: p99.9 dropped 0.1% of points yet cut the PEAK off 98 of 160 samples, and
    # p0.5 dropped 0.5% yet cut the TROUGH off all 160. Taking the true extremes clips
    # nothing and still spans only ~6.5 decades.
    lm = np.concatenate(lm_all)
    lm_lo = float(min(x.min() for x in lm_all))
    lm_hi = float(max(x.max() for x in lm_all))
    pad = 0.02 * (lm_hi - lm_lo)             # a hair of headroom so peaks are not flush
    lm_lo, lm_hi = lm_lo - pad, lm_hi + pad
    _mag_hi = float(10 ** np.percentile(lm, 99.9))
    return {
        "logmag": [lm_lo, lm_hi],
        # Linear magnitude, unlike the log axis, is dominated by the single largest peak:
        # 10**lm_hi would leave the typical curve a flat line along the bottom. A high
        # percentile of the pooled values keeps the usual shape readable. Magnitude cannot
        # go below zero, so the axis starts flush at 0 -- a negative minimum here (like
        # logmag's headroom) would be faking room below a true hard floor, not a soft one,
        # and a line thicker than 1px would visibly dip into it for any near-zero bin
        # (viz2/static/app.js:span() had the same bug for the live plot).
        "mag": [0, _mag_hi],
        "phase": [-math.pi, math.pi],          # bounded already, so global by definition
        "cosphase": [-1.0, 1.0],
        "re": [-m, m], "im": [-m, m],
        "shifts": [-t, t],
        "mode": [0.0, mode],
    }


def d(sid: str) -> Path:
    """The whole path defense: an id not in DIRS never becomes a path."""
    if sid not in DIRS:
        raise KeyError(sid)
    return DIRS[sid]


# ***** signal *****

@lru_cache(maxsize=64)
def fft(sid):
    z = np.load(_fft(d(sid)))
    return z["fft"][0], z["freqs"].astype(float)      # (L,F,C) complex64, (F,)


@lru_cache(maxsize=16)
def shifts(sid, kind):
    a = np.load(d(sid) / SHIFTS[kind])
    a = a[0] if a.ndim == 4 else a
    return a.astype(np.float32)                        # (L,T,C)


def chan(a, ch):
    """x|y|avg -> drop the trailing channel axis."""
    return a.mean(-1) if ch == "avg" else a[..., 0 if ch == "x" else 1]


def pick(a, laser):
    """avg over lasers, or one laser."""
    return a.mean(0) if laser == "avg" else a[int(laser)]


def logmag(z):
    return np.log10(np.abs(z) + 1e-8)


def _odd(n):
    return n if n % 2 == 1 else n + 1


def _hz_to_bins(hz, df):
    return max(1, int(round(hz / df)))


@lru_cache(maxsize=64)
def peaks(sid, k_prom: float = 4.0, distance_hz: float = 5.0, smooth_hz: float = 5.0,
          baseline_hz: float = 25.0):
    """Resonance-mode frequencies, pooled across every laser point rather than one probe's
    single-channel spectrum: a real mode shows up as DISAGREEMENT between points (a
    mode-shape peak or valley), while uncorrelated noise varies about the same everywhere.
    std-across-points is therefore a much cleaner signal to pick peaks from than any one
    point's magnitude -- same idea as Bagon et al., "Hearing the Room Through the Shape of
    the Drum" (CVPR 2026), sec. 4: sigma(w) = std_n(|V_n(w)|), smoothed and peak-picked with
    a MAD-based robust prominence floor rather than a fixed dB threshold.

    Independent of ch/laser -- cached per sid only, so switching the probe's channel or
    laser selection never recomputes this."""
    f, freqs = fft(sid)                                     # (L,F,C)
    Sw = np.concatenate([f[..., 0], f[..., 1]], axis=0)      # [2L, F]
    S = np.abs(Sw).std(0)                                    # [F]
    df = float(freqs[1] - freqs[0])
    S = savgol_filter(S, _odd(2 * _hz_to_bins(smooth_hz, df) + 1), 3)
    baseline = savgol_filter(S, _odd(2 * _hz_to_bins(baseline_hz, df) + 1), 3)
    resid = S - baseline
    sigma = 1.4826 * np.median(np.abs(resid - np.median(resid)))
    prominence = max(1e-12, k_prom * sigma)
    idx, _ = find_peaks(
        S,
        distance=_hz_to_bins(distance_hz, df),
        prominence=prominence,
        wlen=_odd(2 * _hz_to_bins(baseline_hz, df) + 1),
    )
    return sorted(int(i) for i in idx)


def mode(sid, fi):
    """Mode shape at one bin: (U,V) real displacement on the laser grid.

    ALWAYS both components. The mode is a 2-D displacement field; zeroing one channel
    would collapse every arrow onto one axis, which is a rendering artefact, not physics.
    Channel choice belongs to the spectra, not here.

    Phase is normalized so taking the real part is a meaningful snapshot rather than an
    arbitrary point in the cycle (03_mark_visualize.ipynb cell 16).
    """
    f, _ = fft(sid)
    z = f[:, int(fi), :]                                  # (L, C)
    z = z * (np.conj(z[0, 0]) / (abs(z[0, 0]) + 1e-12))
    r, c = INFO["rows"], INFO["cols"]
    return z[:, 0].real.reshape(r, c), z[:, 1].real.reshape(r, c)


def envelope(y, n=800):
    """Min/max decimation. Plain subsampling of a noise trace visibly changes shape
    between redraws; this is what audio editors do."""
    if len(y) <= n:
        return y.tolist()
    k = len(y) // n
    b = y[: k * n].reshape(n, k)
    return np.stack([b.min(1), b.max(1)], 1).ravel().tolist()


def audio(sid, ch, laser):
    """Recovered audio by zero-filling outside the band and inverting.

    Reimplemented from src/data/vibrate.py:get_recovered_audio -- that module does
    `import modal` at top level, so it cannot be imported here.

    Each saved bin goes back to the rfft slot its OWN saved frequency names, not to a band
    rebuilt from INFO's min/max_freq: some datasets (many_objects, cardboard, shoebox)
    record a 100 Hz chirp start but saved the FFT from 50 Hz, and the rebuilt band then
    has fewer slots than there are bins.
    """
    f, freqs = fft(sid)
    n, fs = INFO["n_samples"], INFO["fps"]
    spec = np.zeros(n // 2 + 1, dtype=np.complex64)
    spec[np.rint(freqs * n / fs).astype(int)] = pick(chan(f, ch), laser)
    sig = np.fft.irfft(spec, n=n)
    out = resample(sig, int(22050 * len(sig) / fs))
    return (out / (np.abs(out).max() + 1e-8) * 32767).astype(np.int16), 22050


def rec_channel(ch, laser):
    """The (ch, laser) that audio() recovers from a viewer selection: "both" (x and y drawn
    in one axes) is heard as their average, and "all" is a rendering of every laser, whose
    curves are the average. An average is taken on the complex spectra, i.e. of the
    displacement signals themselves, as chan()/pick() do for every other curve."""
    return (ch if ch in ("x", "y") else "avg"), ("avg" if laser in ("avg", "all") else str(int(laser)))


def recspec(sid, ch, laser):
    """Spectrogram PNG of audio(sid, ch, laser), styled exactly like the post-process
    06_spectrogram mp4 (same utils.viz renderer), plus where the playback line goes:
    {"axes": [left, right, top, bottom] as fractions of the image, "t": [t0, t1] the times
    at the left/right axes edges}. The browser draws the moving line itself over the audio
    from /api/audio, so switching laser/channel costs one ~60ms render, not an mp4 mux."""
    import io
    from scipy.signal import spectrogram
    from utils.viz import render_spectrogram_frame
    pcm, sr = audio(sid, ch, laser)
    # src/data/audio.py:get_spectrogram's parameters
    freqs, times, Sxx = spectrogram(pcm.astype(np.float32), fs=sr, nperseg=4096,
                                    noverlap=4096 // 3, nfft=4096 * 4)
    who = "avg of all lasers" if laser == "avg" else f"Laser {laser}"
    rgb, (x0, x1, top, bottom), (t0, t1) = render_spectrogram_frame(
        freqs, times, Sxx, label=f"Recovered: {{duration}}s, {who}, {'avg of x,y' if ch == 'avg' else ch + '-axis'}",
        max_freq=INFO["max_freq"])
    h, w = rgb.shape[:2]
    b = io.BytesIO()
    Image.fromarray(rgb).save(b, "PNG", compress_level=1)
    return b.getvalue(), {"axes": [float(x0 / w), float(x1 / w), float(top / h), float(bottom / h)],
                        "t": [float(t0), float(t1)]}


def surface(U, V):
    """Height field whose gradient is (U,V): Frankot-Chellappa, as in
    figure_signals.ipynb's reconstruct_surface_from_gradients.

    The quiver shows the gradient of the mode; this is the mode itself. Solved in the
    Fourier domain -- one FFT pair, no iteration -- with a screened-Poisson term (lam)
    that keeps the low frequencies from running away, since the DC component of a height
    reconstructed from slopes alone is arbitrary.
    """
    R, C = U.shape
    # U is the x-slope and V the y-slope, but the grid is indexed [row, col] = [y, x], so
    # the ROW frequency pairs with V and the COLUMN frequency with U.
    ky = 2 * np.pi * np.fft.fftfreq(R)[:, None]
    kx = 2 * np.pi * np.fft.fftfreq(C)[None, :]
    # Screened-Poisson term. The notebook's smoothing_length is in metres against a
    # metre-spaced grid; here the spacing is one laser, so the equivalent is a fraction of
    # the field -- a whole field width, which damps only the very longest wavelength (the
    # one the boundary assumption gets wrong anyway) and leaves the mode shape intact.
    lam = 2 * np.pi / max(R, C)
    den = kx**2 + ky**2 + lam**2
    den[0, 0] = np.inf                         # the arbitrary constant offset
    Z = np.fft.ifft2((-1j * kx * np.fft.fft2(U) - 1j * ky * np.fft.fft2(V)) / den).real
    return Z - Z.mean()
