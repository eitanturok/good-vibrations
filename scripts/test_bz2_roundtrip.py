"""Round-trip test for Bz2Compressor on a real raw vibrations file: load, compress,
decompress, assert bit-exact equality, timing each part.

Usage: python src/data/test_bz2_roundtrip.py [path/to/00_raw_vibrations.npy]
"""
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import numpy as np
from utils.io_utils import Bz2Compressor

DEFAULT_PATH = Path(r"D:\eturok\experiment-23\samples\000000\vibration\00_raw_vibrations.npy")


def main():
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_PATH
    print(f"file: {path}")

    t0 = time.time()
    arr = np.load(path)
    t_load = time.time() - t0
    print(f"load:       {t_load:6.2f}s   shape={arr.shape} dtype={arr.dtype}  {arr.nbytes/1e9:.3f} GB")

    compressor = Bz2Compressor()

    t0 = time.time()
    blob = compressor.compress(arr)
    t_compress = time.time() - t0
    print(f"compress:   {t_compress:6.2f}s   {arr.nbytes/1e9:.3f} GB -> {len(blob)/1e9:.3f} GB  ratio={arr.nbytes/len(blob):.2f}x")

    t0 = time.time()
    out = compressor.decompress(blob)
    t_decompress = time.time() - t0
    print(f"decompress: {t_decompress:6.2f}s   {len(blob)/1e9:.3f} GB -> {out.nbytes/1e9:.3f} GB")

    t0 = time.time()
    assert out.shape == arr.shape, f"shape mismatch: {out.shape} != {arr.shape}"
    assert out.dtype == arr.dtype, f"dtype mismatch: {out.dtype} != {arr.dtype}"
    assert np.array_equal(out, arr), "decompressed array does not match original"
    t_assert = time.time() - t0
    print(f"assert:     {t_assert:6.2f}s   PASSED -- decompressed array is bit-exact with original")

    total = t_load + t_compress + t_decompress + t_assert
    print(f"total:      {total:6.2f}s")


if __name__ == "__main__":
    main()
