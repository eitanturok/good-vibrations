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


def _compare(name_a, shifts_a, name_b, shifts_b, tol=1e-4):
    diff = np.abs(shifts_a - shifts_b)
    match = diff.max() < tol
    print(f"\n--- {name_a} vs {name_b} ---")
    print(f"  max={diff.max():.6f}  mean={diff.mean():.6f}  {'PASS' if match else 'FAIL'} (tol={tol})")
    for i in range(shifts_a.shape[0]):
        d = np.abs(shifts_a[i] - shifts_b[i])
        print(f"  ROI {i}: max={d.max():.6f}  mean={d.mean():.6f}")
    return match


@app.function(
    gpu="A10G",
    image=cuda_image,
    timeout=3600,
    volumes={VOLUME_PATH: volume},
)
def run_comparison(sample_dir_name: str, n_rois: int = 3, batch_size: int = 1024):
    import sys
    sys.path.insert(0, "/src3")
    import cupy as cp
    import numpy as np
    from io_utils import load
    from pclk import compute_shifts_for_roi, compute_shifts_for_all_rois_batched, compute_shifts_for_all_rois_batched_optimized
    from pclk_old import compute_CAM2_translations_v3_cupy

    sample_dir = VOLUME_PATH / sample_dir_name
    metadata = {k: v for d in load(sample_dir / "metadata.jsonl") for k, v in d.items()}
    rois = metadata["roi"][:n_rois]
    raw_vibrations = load(sample_dir / "inputs/00_raw_vibrations.npy")
    print(f"raw_vibrations: {raw_vibrations.shape}  n_rois={n_rois}  batch_size={batch_size}")

    crops = np.stack([raw_vibrations[:, y:y+h, x:x+w] for x, y, w, h in rois])

    # --- original sequential (pclk_old.py) ---
    print("\n[OLD] sequential...")
    shifts_old_list = []
    for x, y, w, h in rois:
        s = compute_CAM2_translations_v3_cupy(cp.asarray(raw_vibrations[:, y:y+h, x:x+w]), batch_size=batch_size)
        shifts_old_list.append(cp.asnumpy(s) if hasattr(s, 'get') else s)
    shifts_old = np.stack(shifts_old_list)
    print(f"[OLD] {shifts_old.shape}  max={np.abs(shifts_old).max():.4f}")

    # --- new sequential (compute_shifts_for_roi) ---
    print("\n[SEQ] sequential (new pclk.py)...")
    shifts_seq_list = []
    for x, y, w, h in rois:
        shifts_seq_list.append(compute_shifts_for_roi(raw_vibrations[:, y:y+h, x:x+w], batch_size))
    shifts_seq = np.stack(shifts_seq_list)
    print(f"[SEQ] {shifts_seq.shape}  max={np.abs(shifts_seq).max():.4f}")

    # --- batched (ground-truth batched, matches old) ---
    print("\n[BATCHED] batched...")
    shifts_batched = compute_shifts_for_all_rois_batched(crops, batch_size)
    print(f"[BATCHED] {shifts_batched.shape}  max={np.abs(shifts_batched).max():.4f}")

    # --- batched optimized ---
    print("\n[OPT] batched_optimized...")
    shifts_opt = compute_shifts_for_all_rois_batched_optimized(crops, batch_size)
    print(f"[OPT] {shifts_opt.shape}  max={np.abs(shifts_opt).max():.4f}")

    # --- pairwise comparisons ---
    results = {}
    results["old_vs_seq"]     = _compare("OLD",     shifts_old,     "SEQ",     shifts_seq)
    results["old_vs_batched"] = _compare("OLD",     shifts_old,     "BATCHED", shifts_batched)
    results["old_vs_opt"]     = _compare("OLD",     shifts_old,     "OPT",     shifts_opt)
    results["batched_vs_opt"] = _compare("BATCHED", shifts_batched, "OPT",     shifts_opt)

    all_pass = all(results.values())
    print(f"\n{'ALL PASS' if all_pass else 'SOME FAILED'}: {results}")
    return results


@app.local_entrypoint()
def main(sample_dir_name: str = "000000", n_rois: int = 3, batch_size: int = 1024):
    result = run_comparison.remote(sample_dir_name, n_rois=n_rois, batch_size=batch_size)
    print(f"\nResult: {result}")
