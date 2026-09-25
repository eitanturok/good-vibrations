import numpy as np

from record.utils.image import crop


def test_crop_full_image_default_is_identity():
    """The exact bug this test exists to catch: an all-zero CropConfig default would
    slice to an empty image instead of the full one."""
    img = np.random.randint(0, 255, (100, 200, 3), dtype=np.uint8)
    out = crop(img, left=0.0, right=1.0, top=0.0, bottom=1.0)
    assert out.shape == img.shape
    assert np.array_equal(out, img)


def test_crop_exact_pixel_slicing():
    # 100 rows (h), 200 cols (w); image-array convention: origin top-left, y down
    img = np.arange(100 * 200 * 3, dtype=np.uint8).reshape(100, 200, 3)
    out = crop(img, left=0.25, right=0.75, top=0.1, bottom=0.5)
    # x1=200*0.25=50, x2=200*0.75=150, y1=100*0.1=10, y2=100*0.5=50
    assert out.shape == (40, 100, 3)
    assert np.array_equal(out, img[10:50, 50:150])
