"""Client-side segmentation helpers. The Modal Segmenter app itself (src/data/segment.py)
is untouched -- other scripts (resegment_gastronorm.py, split_gastronorm_masks.py) already
depend on its exact location and `modal deploy` invocation. This module holds only the
recording-notebook-specific glue that used to be inline in src/record.ipynb cell 67:
building the prompt/top_k args, unpacking bit-packed masks, and computing each object's
center of mass.
"""

import numpy as np
import modal

from utils.io_utils import to_jpeg_bytes, unpack_masks
from utils.metrics import center_of_mass


def get_segmenter():
    """One Modal Cls handle, reused for every call this session (constructed once during
    the notebook's startup warm-up)."""
    return modal.Cls.from_name("segment", "Segmenter")()


def segment(segmenter, crop_overhead: np.ndarray, prompts: dict[str, str], objects: dict[str, int], scale: float = 1.0) -> list[dict]:
    """Run SAM3 segmentation on the overhead crop for each object in `objects` (name ->
    expected instance count), via Modal. Returns [] for an empty box -- no need to round-trip
    to Modal for nothing. `scale` < 1.0 segments on a smaller image for faster inference --
    reuses Segmenter.run()'s own existing downsample/upsample (src/data/segment.py), which
    already resizes masks/boxes back to crop_overhead's full resolution before returning, so
    there's no separate client-side downsize/upsample step to write or maintain here."""
    if not objects:
        return []
    prompt_list = [prompts[o] for o in objects]
    call = segmenter.run.spawn(to_jpeg_bytes(crop_overhead), prompt_list, scale=scale, top_k=list(objects.values()))
    return unpack_masks(call.get())


def object_centers_of_mass(seg_results: list[dict]) -> list[list[tuple[float, float]]]:
    """Per-object list of (row, col) centers of mass -- one sub-list per object in
    `seg_results`, one tuple per instance of that object."""
    return [[center_of_mass(m) for m in r["masks"]] for r in seg_results]


def combined_smask(seg_results: list[dict], shape: tuple[int, int]) -> np.ndarray:
    """OR of every instance's mask across every object -- the single (H, W) bool smask
    saved per sample and accumulated into the coverage heatmap."""
    if not seg_results:
        return np.zeros(shape, dtype=bool)
    return np.any([m for r in seg_results for m in r["masks"]], axis=0)
