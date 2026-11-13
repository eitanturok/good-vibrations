"""shared figure style for figures/src/*: matplotlib defaults (DejaVu Sans, tab10 colours), one fixed colour per level, thumbnail helpers.
Each mapping below is the same in every figure; colours are matplotlib's default tab10 cycle C0..C9 (written as hex so they also work outside matplotlib)."""
import numpy as np
import matplotlib as mpl
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

C = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]   # tab10 = C0..C9

# boxes
BOX_COLORS = {
    "gastro": C[0],
    "plastic": C[2],
    "cardboard": C[5],
    "wood": C[8],
    "shoebox": C[4],
}

# objects (the 4 train objects + 2 unseen ones are all distinct)
OBJECT_COLORS = {
    "candle": C[1],
    "red-cube": C[3],
    "soap dispenser": C[2],
    "vase": C[4],
    "coffee pot": C[5],
    "tomato sauce": C[6],
    "green-cube": C[8],
    "portable-charger": C[9],
    "biscoff": C[0],
    "glass-cup": C[9],
    "empty": C[7],
}

# speakers (4 = the held-out one)
SPEAKER_COLORS = {
    1: C[0],
    2: C[8],
    3: C[1],
    4: C[3],
    5: C[2],
    6: C[9],
    7: C[4],
}

# floor regions (quadrants; "middle" kept for older code)
POSITION_COLORS = {
    "top-left": C[0],
    "top-right": C[1],
    "bottom-left": C[2],
    "bottom-right": C[3],
    "middle": C[4],
}

# train vs any eval split
SPLIT_COLORS = {
    "train": C[0],
    "eval": C[1],
}

# one colour per factor (bars / lines that pool all levels of a factor); noise = grey
FACTOR_COLORS = {
    "noise": C[7],
    "box": C[5],
    "speaker": C[1],
    "object": C[4],
    "position": C[9],
}

NOISE_COLOR = C[7]   # noise / baselines
MEDIAN_COLOR = "black"
INK = "black"
MUTED = "#555555"

# shared line widths and figure widths (inches)
LW = 1.5          # one curve per level
LW_BASE = 2.5     # the base / reference curve
WIDE = 15.0       # multi-panel families: fft, floor coverage, resonance
NARROW = 10.0     # single-panel families: variance, distance


def apply():
    """matplotlib defaults, plus SVG text kept as text and one set of font sizes for every figure"""
    mpl.rcdefaults()
    mpl.rcParams["svg.fonttype"] = "none"
    mpl.rcParams["font.size"] = 10
    mpl.rcParams["axes.titlesize"] = 10
    mpl.rcParams["axes.labelsize"] = 10
    mpl.rcParams["xtick.labelsize"] = 9
    mpl.rcParams["ytick.labelsize"] = 9
    mpl.rcParams["legend.fontsize"] = 9
    mpl.rcParams["figure.titlesize"] = 11
    mpl.rcParams["lines.linewidth"] = LW


def thumbnail(ax, image, xy, size=0.6, xycoords="data", color=None, box_alignment=(0.5, 0.5)):
    """place a small photo (H,W,3 uint8 or float) at xy on ax; size = height in inches; optional coloured frame"""
    zoom = size * 72 / image.shape[0]   # OffsetImage zoom is in points (dpi_cor): size inches = size * 72 pt
    box = OffsetImage(image, zoom=zoom)
    frame = dict(edgecolor=color, linewidth=2) if color is not None else dict(edgecolor="none")
    ab = AnnotationBbox(box, xy, xycoords=xycoords, frameon=color is not None, pad=0.05, bboxprops=frame,
                        box_alignment=box_alignment, annotation_clip=False)
    ax.add_artist(ab)
    return ab


def image_ticks(ax, images, size=0.55, offset=0.03):
    """images under the x tick positions (one per tick, None = skip), just below the axis, tick labels kept below them"""
    ticks = ax.get_xticks()
    for x, image in zip(ticks, images):
        if image is None:
            continue
        thumbnail(ax, image, (x, -offset), size=size, xycoords=("data", "axes fraction"), box_alignment=(0.5, 1.0))
    axes_height_pt = ax.get_position().height * ax.figure.get_figheight() * 72
    ax.tick_params(axis="x", pad=size * 72 + offset * axes_height_pt + 6)


def image_legend(ax, images, colors, x0=1.01, y0=1.0, size=0.42, step=None):
    """a column of framed thumbnails right of the axes (frame = the series colour), top-down from y0 in axes fraction"""
    step = step if step is not None else size * 1.45 / (ax.get_position().height * ax.figure.get_figheight())
    for i, (image, color) in enumerate(zip(images, colors)):
        thumbnail(ax, image, (x0, y0 - i * step), size=size, xycoords="axes fraction", color=color, box_alignment=(0.0, 1.0))


def to_float(image):
    image = np.asarray(image)
    return image / 255.0 if image.dtype == np.uint8 else image


# floor sampling regions: per box, a circle at the centre of each quadrant of the box's position grid (extent of the speaker-1 candle train
# captures' object CoM, row / col as box fractions of the cropped overhead photo, split at its midpoint). One radius for all boxes, in the
# same normalised units; 0.09 is the smallest step that keeps >= 9 speaker-1 candle train captures in every region (data_analysis.ipynb §0)
REGION_NAMES = ["top-left", "top-right", "bottom-left", "bottom-right"]
REGION_RADIUS = 0.09
REGION_CENTRES = {   # (row, col)
    "gastro": {"top-left": (0.3517, 0.3781), "top-right": (0.3517, 0.6778), "bottom-left": (0.6151, 0.3781), "bottom-right": (0.6151, 0.6778)},
    "plastic": {"top-left": (0.3670, 0.3714), "top-right": (0.3670, 0.6377), "bottom-left": (0.6033, 0.3714), "bottom-right": (0.6033, 0.6377)},
    "cardboard": {"top-left": (0.3410, 0.3600), "top-right": (0.3410, 0.6926), "bottom-left": (0.6204, 0.3600), "bottom-right": (0.6204, 0.6926)},
}


def region_of(r, c, box):
    """the region whose circle contains (row, col), else None"""
    for name, (rc, cc) in REGION_CENTRES[box].items():
        if (r - rc) ** 2 + (c - cc) ** 2 <= REGION_RADIUS ** 2:
            return name
    return None


def draw_regions(ax, box, colors, lw=1.5, fill_alpha=0.0):
    """the 4 region circles on an axis whose x = col, y = row (box fractions); colors = {region name: colour}"""
    from matplotlib.patches import Circle
    for name, (rc, cc) in REGION_CENTRES[box].items():
        ax.add_patch(Circle((cc, rc), REGION_RADIUS, facecolor="none", edgecolor=colors[name], lw=lw, zorder=5))
        if fill_alpha:
            ax.add_patch(Circle((cc, rc), REGION_RADIUS, facecolor=colors[name], edgecolor="none", alpha=fill_alpha, zorder=4))
