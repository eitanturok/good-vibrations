# compress.py — compressing `00_raw_vibrations.npy` as much as possible

Target: `D:\eturok\experiment-23\samples\000000\vibration\00_raw_vibrations.npy`,
shape `(9000, 300, 1008)` uint8, 2,721,600,000 bytes raw (~2.72 GB). Method
exploration ran on a scratch copy (deleted afterward); the final full-file
compression read the original directly, strictly read-only, and never modified
it (confirmed via unchanged file timestamp and a SHA-256 verify against the
decompressed output). Code: [compress.py](compress.py),
[compress_full.py](compress_full.py).

## Method

Fast exploration on a 100-frame subset (30.24 MB) across 14 techniques, each
capped at 60s in its own subprocess (killed and logged as "timed out" if
exceeded — none did). Plot: [compress_progress.png](compress_progress.png).

## Results (100-frame subset)

| # | Method | Compressed | Ratio | Time |
|---|--------|-----------:|------:|-----:|
| 0 | raw (no compression) | 30.24 MB | 1.00x | 0.32s |
| 1 | npz (zlib via savez_compressed) | 8.66 MB | 3.49x | 1.21s |
| 2 | zlib level 6 | 8.66 MB | 3.49x | 1.20s |
| 3 | zlib level 9 | 8.26 MB | 3.66x | 12.45s |
| 4 | bz2 level 9 | 7.13 MB | 4.24x | 0.81s |
| 5 | lzma preset 6 | 4.74 MB | 6.38x | 12.83s |
| **6** | **lzma preset 9 extreme** | **4.49 MB** | **6.74x** | 16.95s |
| 7 | zstd level 3 | 8.55 MB | 3.54x | 0.36s |
| 8 | zstd level 19 | 5.90 MB | 5.13x | 13.15s |
| 9 | zstd level 22 (max) | 5.80 MB | 5.22x | 14.07s |
| 10 | delta + zlib level 9 | 6.05 MB | 5.00x | 10.69s |
| 11 | delta + lzma preset 9e | 5.17 MB | 5.85x | 17.57s |
| 12 | delta + zstd level 19 | 5.38 MB | 5.62x | 11.36s |
| 13 | delta + zstd level 22 | 5.36 MB | 5.64x | 13.04s |

**Winner: plain LZMA, preset 9 | EXTREME — 6.74x, no delta encoding.**

Frame-to-frame delta encoding (subtracting the previous frame, mod 256, before
compressing) made *every* codec worse, not better. That means the byte
redundancy in this data is spatial (repeating values within/across the 1008-wide
rows of a frame — visible in the raw array print as runs like `15 16 15 ... 20`
and repeated `~15-20` / `~130-148` value bands) rather than temporal
(frame-to-frame). Differencing destroyed that spatial structure without buying
back enough new redundancy across frames to compensate.

## Full-file compression

Applied LZMA preset 9|EXTREME to the full 2.72 GB **original file**, opened
strictly read-only (`"rb"`), via a streaming `lzma.LZMAFile` (64 MB chunks, so
peak memory stays low rather than materializing the whole file at once). Output:
`D:\eturok\experiment-23\samples\000000\vibration\00_raw_vibrations.npy.xz`.

| | |
|---|---|
| Original | 2.722 GB (2,721,600,128 bytes) |
| Compressed | 0.399 GB (399,457,884 bytes) |
| **Ratio** | **6.81x** |
| Compression time | 1816.7s (~30 min) |
| Verification | SHA-256 of decompressed stream matches original exactly (21.7s) |

The `.xz` file is the raw `.npy` bytes (header + data) compressed as-is —
decompressing with `xz -d` / `lzma.decompress` reproduces a byte-identical
`.npy` that `np.load` reads with no custom container format needed.

`00_raw_vibrations.npy` itself was never opened for writing and its on-disk
timestamp is unchanged from before this session. A scratch copy
(`00_raw_vibrations_copy.npy` + its `.xz`) was used for the method-exploration
phase and deleted once the direct-from-original run was verified.
