"""Bar plots of the laser-face sweep: one figure per split.

One hue per configuration, taken in fixed slot order from the reference categorical palette, so a
configuration keeps its color across all three figures and the panels can be read side by side. No
legend: four bars carry their own axis labels, so identity is never color-alone.

Every bar is directly labeled, which is also what the palette's relief rule requires -- the aqua
and yellow slots sit below 3:1 contrast on a white surface, so the value must be legible without
relying on the fill.

The grid under each label is what makes the comparison honest: left/right/both are all 8x4 = 32
lasers and directly comparable, while `all` is 8x8 = 64, twice the data, and is a ceiling rather
than a fourth peer.

Error bars are the reported +- (std across samples), so they are sample spread, not standard error
of the mean -- they say how variable individual samples are, NOT how precisely the mean is known.
Do not read two overlapping bars as "no difference".

    python scripts/plot_laser_faces.py
"""
from pathlib import Path

import matplotlib.pyplot as plt

METRIC = "soft-iou"  # relabel if these numbers are a different measure
OUT_DIR = Path(__file__).resolve().parent.parent / "figures"

FACES = ["left", "right", "both", "all"]
GRID = {"left": "8x4", "right": "8x4", "both": "8x4", "all": "8x8"}

# split -> {face: (mean, std)}
RESULTS = {
    "train":   {"left": (0.869, 0.185), "right": (0.873, 0.185), "both": (0.870, 0.185), "all": (0.845, 0.196)},
    "1 cube":  {"left": (0.277, 0.141), "right": (0.374, 0.155), "both": (0.362, 0.208), "all": (0.385, 0.178)},
    "2 cubes": {"left": (0.193, 0.113), "right": (0.204, 0.132), "both": (0.217, 0.127), "all": (0.218, 0.155)},
}

# reference categorical palette, slots 1-4 in fixed order -- assigned per configuration and never
# cycled, so `both` is the same aqua in every figure
COLORS = {"left": "#2a78d6", "right": "#eb6834", "both": "#1baf7a", "all": "#eda100"}
INK = "#1f2933"
MUTED = "#7b8794"


def plot_split(split: str, results: dict[str, tuple[float, float]], path: Path) -> None:
    means = [results[f][0] for f in FACES]
    stds = [results[f][1] for f in FACES]
    colors = [COLORS[f] for f in FACES]

    fig, ax = plt.subplots(figsize=(5.2, 3.8))
    ax.bar(FACES, means, yerr=stds, color=colors, width=0.62,
           capsize=5, error_kw=dict(ecolor=MUTED, elinewidth=1.2, capthick=1.2),
           edgecolor="white", linewidth=1.0, zorder=3)

    # the value sits ON the bar, just inside its top edge, in white -- so it never collides with
    # the error whisker rising out of the same point
    for face, mean in zip(FACES, means):
        ax.text(face, mean - 0.018, f"{mean:.3f}", ha="center", va="top",
                fontsize=10, color="white", fontweight="bold", zorder=4)

    ax.set_title(f"{split} - {METRIC}", fontsize=12, color=INK, pad=10, loc="left")
    ax.set_ylabel(METRIC, fontsize=10, color=MUTED)
    ax.set_ylim(0, max(m + s for m, s in zip(means, stds)) * 1.12)

    # the grid under each label, since 8x8 is what makes `all` incomparable to the rest
    ax.set_xticks(range(len(FACES)))
    ax.set_xticklabels([f"{f}\n{GRID[f]}" for f in FACES], fontsize=10, color=INK)

    ax.grid(axis="y", color="#e4e7eb", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#cbd2d9")
    ax.tick_params(axis="both", length=0, labelcolor=MUTED)
    ax.tick_params(axis="x", labelcolor=INK)

    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {path}")


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)
    for split, results in RESULTS.items():
        plot_split(split, results, OUT_DIR / f"laser_faces_{split.replace(' ', '-')}.png")


if __name__ == "__main__":
    main()
