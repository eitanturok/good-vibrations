"""Compression experiments for `00_raw_vibrations.npy`.

Target: D:\\eturok\\experiment-23\\samples\\000000\\vibration\\00_raw_vibrations.npy
Shape (9000, 300, 1008) uint8, 2,721,600,000 bytes raw. Consecutive frames along
axis 0 look highly repetitive (same sensor read many times), so delta-encoding
across frames before generic compression is the main idea explored here.

All work happens on a COPY of the source file (00_raw_vibrations_copy.npy), never
the original. Fast iteration happens on a SUBSET of frames (see SUBSET_N) so each
attempt is quick; the winning method is then re-run on the FULL array.

Each technique runs in its own subprocess with a hard TIMEOUT_S cap (2 min) so a
slow setting (e.g. lzma extreme, zstd level 22) gets killed and logged instead of
stalling the whole sweep.
"""

import json
import lzma
import multiprocessing as mp
import time
import zlib
import bz2
from functools import partial
from pathlib import Path

import numpy as np

SRC = Path(r"D:\eturok\experiment-23\samples\000000\vibration\00_raw_vibrations_copy.npy")
OUT_DIR = Path(__file__).parent / "compress_outputs"
ATTEMPTS_PATH = Path(__file__).parent / "compress_attempts.json"
SUBSET_N = 100  # frames used for fast exploration (full array has 9000)
TIMEOUT_S = 60  # hard cap per technique (well under the 2 min/technique budget)


def delta_encode(a: np.ndarray) -> np.ndarray:
    """Frame-to-frame diff along axis 0, wrapping mod 256 (reversible on uint8)."""
    d = np.empty_like(a)
    d[0] = a[0]
    d[1:] = a[1:] - a[:-1]  # wraps mod 256 for uint8 automatically
    return d


def delta_decode(d: np.ndarray) -> np.ndarray:
    out = np.empty_like(d)
    out[0] = d[0]
    acc = d[0].copy()
    for i in range(1, d.shape[0]):
        acc = acc + d[i]  # uint8 add wraps mod 256, inverse of the diff
        out[i] = acc
    return out


# --- compressors (bytes in, bytes out) ---------------------------------

def c_npz(a):
    import io

    buf = io.BytesIO()
    np.savez_compressed(buf, data=a)
    return buf.getvalue()


def c_zlib(a, level=6):
    return zlib.compress(a.tobytes(), level)


def c_bz2(a, level=9):
    return bz2.compress(a.tobytes(), level)


def c_lzma(a, preset=6):
    return lzma.compress(a.tobytes(), preset=preset)


def c_zstd(a, level=3):
    import zstandard as zstd

    cctx = zstd.ZstdCompressor(level=level)
    return cctx.compress(a.tobytes())


def c_raw(a):
    return a.tobytes()


def _worker(fn, data, q):
    try:
        blob = fn(data)
        q.put(("ok", blob))
    except Exception as e:  # noqa: BLE001
        q.put(("error", str(e)))


def run_with_timeout(fn, data, timeout_s):
    """Run fn(data) in a subprocess; kill it and return None if it exceeds timeout_s.

    Must read the result Queue BEFORE join(): a child writing a result bigger than
    the OS pipe buffer blocks in Queue.put() until someone reads, so join()-first
    deadlocks on any real payload (regression caught when even the trivial
    "raw bytes" method timed out).
    """
    import queue as queue_mod

    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    p = ctx.Process(target=_worker, args=(fn, data, q))
    p.start()
    try:
        status, payload = q.get(timeout=timeout_s)
    except queue_mod.Empty:
        p.terminate()
        p.join()
        return None, "timeout"
    p.join()
    if status == "ok":
        return payload, None
    return None, payload


def try_method(name: str, fn, data: np.ndarray, attempts: list, note: str = "", lossless: bool = True):
    t0 = time.time()
    blob, err = run_with_timeout(fn, data, TIMEOUT_S)
    dt = time.time() - t0
    raw = data.nbytes
    if blob is None:
        attempts.append(
            {
                "n": len(attempts),
                "name": name,
                "note": note,
                "raw_bytes": raw,
                "compressed_bytes": None,
                "ratio": None,
                "seconds": dt,
                "error": err,
                "lossless": lossless,
            }
        )
        print(f"[{len(attempts)-1:2d}] {name:35s} SKIPPED ({err}) after {dt:6.2f}s  {note}", flush=True)
        return None
    size = len(blob)
    ratio = raw / size
    attempts.append(
        {
            "n": len(attempts),
            "name": name,
            "note": note,
            "raw_bytes": raw,
            "compressed_bytes": size,
            "ratio": ratio,
            "seconds": dt,
            "error": None,
            "lossless": lossless,
        }
    )
    print(f"[{len(attempts)-1:2d}] {name:35s} {size/1e6:9.2f} MB  ratio={ratio:5.2f}x  {dt:6.2f}s  {note}", flush=True)
    return blob


def main():
    print(f"loading subset ({SUBSET_N} frames) from {SRC}", flush=True)
    full = np.load(SRC, mmap_mode="r")
    print("full shape", full.shape, full.dtype, f"{full.nbytes/1e9:.3f} GB", flush=True)

    subset = np.array(full[:SUBSET_N])
    print(f"subset loaded: {subset.nbytes/1e6:.1f} MB", flush=True)
    subset_delta = delta_encode(subset)

    attempts = []

    try_method("raw (no compression)", c_raw, subset, attempts)
    try_method("npz (zlib via savez_compressed)", c_npz, subset, attempts)
    try_method("zlib level 6", partial(c_zlib, level=6), subset, attempts)
    try_method("zlib level 9", partial(c_zlib, level=9), subset, attempts)
    try_method("bz2 level 9", partial(c_bz2, level=9), subset, attempts)
    try_method("lzma preset 6", partial(c_lzma, preset=6), subset, attempts)
    try_method("lzma preset 9 extreme", partial(c_lzma, preset=9 | lzma.PRESET_EXTREME), subset, attempts)
    try_method("zstd level 3", partial(c_zstd, level=3), subset, attempts)
    try_method("zstd level 19", partial(c_zstd, level=19), subset, attempts)
    try_method("zstd level 22 (max)", partial(c_zstd, level=22), subset, attempts)

    # delta-encoded variants (frame[i] -= frame[i-1], mod 256) -----------
    try_method("delta + zlib level 9", partial(c_zlib, level=9), subset_delta, attempts, note="frame diff first")
    try_method("delta + lzma preset 9e", partial(c_lzma, preset=9 | lzma.PRESET_EXTREME), subset_delta, attempts, note="frame diff first")
    try_method("delta + zstd level 19", partial(c_zstd, level=19), subset_delta, attempts, note="frame diff first")
    try_method("delta + zstd level 22", partial(c_zstd, level=22), subset_delta, attempts, note="frame diff first")

    ATTEMPTS_PATH.write_text(json.dumps(attempts, indent=2))
    print(f"\nwrote {ATTEMPTS_PATH}")

    scored = [a for a in attempts if a["compressed_bytes"] is not None]
    best = min(scored, key=lambda x: x["compressed_bytes"])
    print(f"\nbest on subset: [{best['n']}] {best['name']} -> ratio {best['ratio']:.2f}x")


if __name__ == "__main__":
    mp.freeze_support()
    main()
