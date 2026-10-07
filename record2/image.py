"""Pure image helper: crop. No hardware, no I/O."""

import numpy as np


def crop(image: np.ndarray, left: float, right: float, top: float, bottom: float) -> np.ndarray:
    """Crop `image` to the box given by fractions of width/height in [0.0, 1.0], in
    image-array convention (origin top-left, y increasing downward) -- ports
    src/record.ipynb cell 67's crop() unchanged. left=0, right=1, top=0, bottom=1 is
    the full image (no crop)."""
    h, w = image.shape[:2]
    x1, x2 = int(w * left), int(w * right)
    y1, y2 = int(h * top), int(h * bottom)
    return image[y1:y2, x1:x2]
