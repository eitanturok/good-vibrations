"""Apply the winning method from compress.py's sweep (lzma preset 9|EXTREME) to
the FULL raw vibration file and write the compressed output.

Reads the original file strictly read-only ("rb") and writes a new .xz file
alongside it -- the original is never opened for writing, so it cannot be
modified by this script. Compresses the raw .npy file bytes as-is (header +
data), so decompressing with `lzma.decompress` or `xz -d` reproduces a
byte-identical .npy that `np.load` reads directly -- no custom container format
needed.

Streams the file in chunks through an LZMAFile so peak memory stays low instead
of materializing the whole ~2.7 GB file at once.
"""

import hashlib
import lzma
import time
from pathlib import Path

SRC = Path(r"D:\eturok\experiment-23\samples\000000\vibration\00_raw_vibrations.npy")
DST = SRC.with_suffix(SRC.suffix + ".xz")
CHUNK = 64 * 1024 * 1024


def compress():
    filt = [{"id": lzma.FILTER_LZMA2, "preset": 9 | lzma.PRESET_EXTREME}]
    raw_size = SRC.stat().st_size
    t0 = time.time()
    with open(SRC, "rb") as fin, lzma.open(DST, "wb", format=lzma.FORMAT_XZ, filters=filt) as fout:
        while True:
            chunk = fin.read(CHUNK)
            if not chunk:
                break
            fout.write(chunk)
    dt = time.time() - t0
    out_size = DST.stat().st_size
    print(f"compressed {raw_size/1e9:.3f} GB -> {out_size/1e9:.3f} GB  ratio={raw_size/out_size:.2f}x  {dt:.1f}s", flush=True)
    return raw_size, out_size, dt


def verify():
    t0 = time.time()
    h_src = hashlib.sha256()
    with open(SRC, "rb") as f:
        while True:
            b = f.read(CHUNK)
            if not b:
                break
            h_src.update(b)

    h_dec = hashlib.sha256()
    with lzma.open(DST, "rb") as f:
        while True:
            b = f.read(CHUNK)
            if not b:
                break
            h_dec.update(b)
    dt = time.time() - t0
    ok = h_src.hexdigest() == h_dec.hexdigest()
    print(f"verify: decompressed sha256 {'MATCHES' if ok else 'MISMATCH'} original ({dt:.1f}s)", flush=True)
    return ok


if __name__ == "__main__":
    raw_size, out_size, dt = compress()
    ok = verify()
    if not ok:
        raise SystemExit("verification failed: decompressed file does not match original")
    print("done", flush=True)
