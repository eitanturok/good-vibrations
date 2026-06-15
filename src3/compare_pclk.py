"""Compare outputs of new batched PCLK vs old sequential PCLK (pclk_old.py).

Run with:
    modal run src3/compare_pclk.py --sample-dir-name 000000
"""
import sys
from pathlib import Path

import modal
import numpy as np

if Path("/src3").exists() and str(Path("/src3")) not in sys.path:
    sys.path.insert(0, "/src3")

app = modal.App("pclk-compare")
volume = modal.Volume.from_name("samples", create_if_missing=True)
VOLUME_PATH = Path("/samples")

cuda_image = (
    modal.Image.from_registry("nvidia/cuda:12.2.2-cudnn8-devel-ubuntu22.04", add_python="3.11")
    .pip_install("cupy-cuda12x", "numpy", "tqdm", "scipy", "matplotlib", "pillow", "ipython")
    .add_local_dir(Path(__file__).parent, remote_path="/src3")
)


@app.function(
    gpu="A10G",
    image=cuda_image,
    timeout=3600,
    volumes={VOLUME_PATH: volume},
)
def run_comparison(sample_dir_name: str, n_rois: int = 3, batch_size: int = 1024):
    import sys, json
    sys.path.insert(0, "/src3")
    import cupy as cp
    import numpy as np
    from io_utils import load
    from pclk import compute_shifts_for_all_rois_batched
    from pclk_old import compute_CAM2_translations_v3_cupy

    sample_dir = VOLUME_PATH / sample_dir_name
    metadata = {k: v for d in load(sample_dir / "metadata.jsonl") for k, v in d.items()}
    rois = metadata["roi"][:n_rois]
    raw_vibrations = load(sample_dir / "inputs/00_raw_vibrations.npy")
    print(f"raw_vibrations: {raw_vibrations.shape}  dtype={raw_vibrations.dtype}")
    print(f"comparing {n_rois} ROIs, batch_size={batch_size}")

    # --- new batched ---
    crops = np.stack([raw_vibrations[:, y:y+h, x:x+w] for x, y, w, h in rois])  # (L, T, H, W)
    print(f"\n[NEW] running batched on crops {crops.shape}")
    shifts_new = compute_shifts_for_all_rois_batched(crops, batch_size)           # (L, T, 2)
    print(f"[NEW] shifts_new: {shifts_new.shape}  max={np.max(np.abs(shifts_new)):.4f}")

    # --- old sequential ---
    print(f"\n[OLD] running sequential (one ROI at a time)")
    shifts_old_list = []
    for x, y, w, h in rois:
        crop = raw_vibrations[:, y:y+h, x:x+w]
        s = compute_CAM2_translations_v3_cupy(cp.asarray(crop), batch_size=batch_size)
        shifts_old_list.append(cp.asnumpy(s) if hasattr(s, 'get') else s)
    shifts_old = np.stack(shifts_old_list)  # (L, T, 2)
    print(f"[OLD] shifts_old: {shifts_old.shape}  max={np.max(np.abs(shifts_old)):.4f}")

    # --- compare ---
    diff = np.abs(shifts_new - shifts_old)
    print(f"\n--- comparison ---")
    print(f"max abs diff:  {diff.max():.6f}")
    print(f"mean abs diff: {diff.mean():.6f}")
    print(f"std abs diff:  {diff.std():.6f}")

    for i in range(n_rois):
        d = np.abs(shifts_new[i] - shifts_old[i])
        print(f"  ROI {i}: max={d.max():.6f}  mean={d.mean():.6f}")

    tol = 1e-4
    match = diff.max() < tol
    print(f"\n{'PASS' if match else 'FAIL'}: max diff {'<' if match else '>='} {tol}")
    return {
        "max_diff": float(diff.max()),
        "mean_diff": float(diff.mean()),
        "match": match,
    }


@app.local_entrypoint()
def main(sample_dir_name: str = "000000", n_rois: int = 3, batch_size: int = 1024):
    result = run_comparison.remote(sample_dir_name, n_rois=n_rois, batch_size=batch_size)
    print(f"\nResult: {result}")
