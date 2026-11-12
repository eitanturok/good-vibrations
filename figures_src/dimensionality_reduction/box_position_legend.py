"""The 2-D colour legend of the viz's colour = "xy" mode, drawn on a photo of each box.

Writes into figures_src/dimensionality_reduction/box_position_legend_out/:
  gastro.svg, plastic.svg, cardboard.svg   overhead empty-box photo + a uniform grid of big dots over the box floor, in the xy colour
  <box>_45deg.svg                          the same grid on the 45 degree photo of the box, placed on the floor by a homography
  legend.svg                               the same grid over the full colour range, no photo

Run:
  .venv/bin/python figures_src/dimensionality_reduction/box_position_legend.py
"""
import colorsys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import patheffects
import numpy as np
import pandas as pd
from PIL import Image

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = HERE / "box_position_legend_out"
META = ROOT / "runs/_src9_cache/nb137/decres-rescale-aug10-v1/meta.csv"  # written by viz.py
MASK_SIZE = 64  # x / y in meta.csv are the mask centre of mass in a 64 x 64 mask (x = column, y = row)

# box -> an empty-box sample, its cropped overhead photo is the background
EMPTY_PHOTO = {
    "gastro": "2026_09_27_gastronorm_four_objs/samples/000005",
    "plastic": "2026_10_05_green_plastic_four_objs/samples/001464-1",
    "cardboard": "2026_10_11_cardboard_four_objs/samples/002919-1",
}

# The four corners of the box floor (far left, far right, near right, near left), picked by eye:
#   FLOOR_MASK   in the overhead photo, in 64 x 64 mask coordinates (the top of the overhead photo is the far side)
#   FLOOR_45DEG  in the 45 degree photo, in pixels. The front wall hides the near part of the floor, so the two near
#                corners are where the visible side edges of the floor reach the bottom of the box.
FLOOR_MASK = {
    "gastro": [(10.8, 8.3), (57.7, 8.3), (57.7, 53.0), (10.8, 53.0)],
    "plastic": [(11.7, 9.5), (53.9, 9.5), (53.9, 51.0), (11.7, 51.0)],
    "cardboard": [(8.4, 4.6), (62.3, 4.6), (62.3, 60.3), (8.4, 60.3)],
}
FLOOR_45DEG = {
    "gastro": [(141, 128), (502, 128), (565, 352), (70, 352)],
    "plastic": [(115, 162), (488, 163), (526, 390), (86, 390)],
    "cardboard": [(92, 133), (522, 137), (584, 367), (48, 370)],
}
# Pixel row of the inner edge of the front rim in the 45 degree photo: the floor below it is hidden by the front wall.
FRONT_RIM_45DEG = {"gastro": 246, "plastic": 282, "cardboard": 249}

OUT.mkdir(exist_ok=True)

# Same rows as the viz: every sample with an object, minus the one split viz.py drops.
df = pd.read_csv(META)
df = df[df.split != "eval/cardboard/vase-laser-shifted-4"]
df = df[df.x.notna()]

GRID = 12  # dots per side
FIG_INCHES = 7
OUTLINE = patheffects.withStroke(linewidth=5, foreground="black")  # keeps a white axis readable on any part of a photo
DOT_SIZE = (18 / 700 * FIG_INCHES * 72) ** 2  # nb137: 18 px dots on a 700 px figure


def xy_colours(x, y):
    # same colour as the viz's colour = "xy": x = hue (red -> purple), y = light (top) -> dark (bottom),
    # spanning the range of the grid
    u = (x - x.min()) / (x.max() - x.min())
    v = (y - y.min()) / (y.max() - y.min())
    colours = []
    for a, b in zip(u, v):
        colours.append(colorsys.hls_to_rgb(0.8 * a, 0.8 - 0.55 * b, 0.9))
    return colours


def grid(x_min, x_max, y_min, y_max):
    x, y = np.meshgrid(np.linspace(x_min, x_max, GRID), np.linspace(y_min, y_max, GRID))
    return x.ravel(), y.ravel()


def square_axes():
    fig, ax = plt.subplots(figsize=(FIG_INCHES, FIG_INCHES))
    ax.set_xlim(-0.5, MASK_SIZE - 0.5)
    ax.set_ylim(MASK_SIZE - 0.5, -0.5)  # y is the image row: 0 at the top
    ax.set_aspect("equal")
    ax.axis("off")
    return fig, ax


def l_axis(ax, x, y, colour):
    # one L-shaped axis just outside the grid: down the left side, then along the bottom, with a big x and y
    gap = 3
    left = x.min() - gap
    bottom = y.max() + gap
    outline = []
    if colour == "white":
        outline = [OUTLINE]
    ax.plot([left, left, x.max()], [y.min(), bottom, bottom], color=colour, linewidth=2.5, solid_capstyle="round", path_effects=outline)
    ax.text((left + x.max()) / 2, bottom + 1, "x", fontsize=40, color=colour, ha="center", va="top", path_effects=outline)
    ax.text(left - 1.5, (y.min() + bottom) / 2, "y", fontsize=40, color=colour, ha="right", va="center", path_effects=outline)


# 1-3: like the nb137 colour key. The empty-box photo is stretched onto the square 64 x 64 mask frame, with a uniform
# grid of dots over the box floor (the range of the object positions in that box) and an L-shaped x / y axis.
for box, sample in EMPTY_PHOTO.items():
    photo = Image.open(ROOT / "experiments" / sample / "image/02_cropped_overhead.png")
    d = df[df.box == box]
    # 0.5-99.5 percentile, not min-max: one broken plastic mask sits outside the box wall
    x_min = d.x.quantile(0.005)
    x_max = d.x.quantile(0.995)
    y_min = d.y.quantile(0.005)
    y_max = d.y.quantile(0.995)
    x, y = grid(x_min, x_max, y_min, y_max)

    fig, ax = square_axes()
    ax.imshow(photo, extent=(-0.5, MASK_SIZE - 0.5, MASK_SIZE - 0.5, -0.5), aspect="auto")
    ax.scatter(x, y, c=xy_colours(x, y), s=DOT_SIZE, edgecolors="white", linewidths=0.75, zorder=3)
    l_axis(ax, x, y, "white")
    fig.savefig(OUT / f"{box}.svg", bbox_inches="tight", pad_inches=0, dpi=300)
    plt.close(fig)
    print(f"{box} -> {OUT / f'{box}.svg'}")

    # The same grid on the 45 degree photo. The floor is a plane, so a homography (3 x 3 matrix, fixed by the four
    # floor corners) maps mask coordinates to pixels of the tilted photo exactly.
    photo = Image.open(ROOT / "assets/45_deg_empty_box" / f"{box}_45deg.png")
    homography = cv2.getPerspectiveTransform(np.float32(FLOOR_MASK[box]), np.float32(FLOOR_45DEG[box]))

    def to_photo(mask_x, mask_y):
        points = np.stack([mask_x, mask_y, np.ones(len(mask_x))])  # homogeneous coordinates
        points = homography @ points
        return points[0] / points[2], points[1] / points[2]

    dot_x, dot_y = to_photo(x, y)
    width, height = photo.size
    fig, ax = plt.subplots(figsize=(FIG_INCHES, FIG_INCHES * height / width))
    ax.imshow(photo)
    # everything on the floor is cut off at the front rim, so nothing is drawn through the front wall
    rim = FRONT_RIM_45DEG[box]
    visible = plt.Rectangle((0, 0), width, rim, transform=ax.transData)
    dots = ax.scatter(dot_x, dot_y, c=xy_colours(x, y), s=DOT_SIZE, edgecolors="white", linewidths=0.75, zorder=3)
    dots.set_clip_path(visible)

    # L-shaped axis with its corner at the far left: along the far edge (x) and down the left side (y) until the rim
    gap = 2
    left = x.min() - gap
    far = y.min() - gap
    line_x, line_y = to_photo(np.array([left, left, x.max()]), np.array([y.max(), far, far]))
    line = ax.plot(line_x, line_y, color="white", linewidth=2.5, solid_capstyle="round", path_effects=[OUTLINE])[0]
    line.set_clip_path(visible)
    ax.text((line_x[1] + line_x[2]) / 2, line_y[1] - 6, "x", fontsize=40, color="white", ha="center", va="bottom", path_effects=[OUTLINE])
    # the visible part of the left side runs from the corner to the rim, the y sits at its middle
    half = (rim - line_y[1]) / (line_y[0] - line_y[1]) / 2
    y_label_x = line_x[1] + half * (line_x[0] - line_x[1])
    y_label_y = line_y[1] + half * (line_y[0] - line_y[1])
    ax.text(y_label_x - 8, y_label_y, "y", fontsize=40, color="white", ha="right", va="center", path_effects=[OUTLINE])

    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    ax.axis("off")
    fig.savefig(OUT / f"{box}_45deg.svg", bbox_inches="tight", pad_inches=0, dpi=300)
    plt.close(fig)
    print(f"{box} 45 deg -> {OUT / f'{box}_45deg.svg'}")

# 4: the legend, the same grid with no photo.
x, y = grid(df.x.quantile(0.005), df.x.quantile(0.995), df.y.quantile(0.005), df.y.quantile(0.995))
fig, ax = square_axes()
ax.scatter(x, y, c=xy_colours(x, y), s=DOT_SIZE)
l_axis(ax, x, y, "black")
fig.savefig(OUT / "legend.svg", bbox_inches="tight")
plt.close(fig)
print(f"legend -> {OUT / 'legend.svg'}")
