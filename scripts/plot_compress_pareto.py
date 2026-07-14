"""Plot the compress.py sweep as a time-vs-ratio Pareto frontier.

x = seconds to compress (100-frame / 30.24 MB subset), y = compression ratio
(raw_bytes / compressed_bytes). A method is on the frontier if no other method
is both faster AND has an equal-or-better ratio. Frontier points are joined by a
line; dominated points are plotted but not joined.

Usage: python src/data/plot_compress_pareto.py [data.json] [out.png]
Defaults to compress_attempts.json / compress_pareto.png if no args given.
"""
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt

COLOR_GOOD = "#0ca30c"
COLOR_MUTED = "#898781"
COLOR_INK = "#0b0b0b"
COLOR_GRID = "#e1e0d9"
COLOR_SURFACE = "#fcfcfb"


def pareto_frontier(points):
    """points: list of (time, ratio, idx). Returns set of idx on the frontier
    (fastest-to-slowest, strictly increasing ratio)."""
    frontier = set()
    best_ratio = -1
    for t, r, i in sorted(points, key=lambda p: p[0]):
        if r > best_ratio:
            frontier.add(i)
            best_ratio = r
    return frontier


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    data_path = Path(args[0]) if len(args) > 0 else Path(__file__).parent / "compress_attempts.json"
    out_path = Path(args[1]) if len(args) > 1 else Path(__file__).parent / "compress_pareto.png"

    attempts = json.loads(data_path.read_text())
    attempts = [a for a in attempts if a["compressed_bytes"] is not None]  # drop timed-out

    points = [(a["seconds"], a["ratio"], i) for i, a in enumerate(attempts)]
    frontier_idx = pareto_frontier(points)

    fig, ax = plt.subplots(figsize=(11, 7.5), facecolor=COLOR_SURFACE)
    ax.set_facecolor(COLOR_SURFACE)

    frontier_pts = sorted([(a["seconds"], a["ratio"]) for i, a in enumerate(attempts) if i in frontier_idx])
    ax.plot([p[0] for p in frontier_pts], [p[1] for p in frontier_pts], color=COLOR_GOOD, linewidth=2, zorder=2, solid_capstyle="round")

    for i, a in enumerate(attempts):
        t, r, name = a["seconds"], a["ratio"], a["name"]
        on_frontier = i in frontier_idx
        lossless = a.get("lossless", True)
        marker = "o" if lossless else "^"  # circle = lossless/invertible, triangle = lossy
        color = COLOR_GOOD if on_frontier else COLOR_MUTED
        ax.scatter([t], [r], s=110 if on_frontier else 80, marker=marker, color=color, zorder=3, edgecolors=COLOR_SURFACE, linewidths=1.5)
        offset = (8, 8) if on_frontier else (8, -8)
        va = "bottom" if on_frontier else "top"
        ax.annotate(
            f"{name}\n{r:.2f}x, {t:.1f}s", (t, r), textcoords="offset points", xytext=offset,
            ha="left", va=va, fontsize=8, color=COLOR_INK, linespacing=1.3,
        )

    legend_handles = [
        plt.Line2D([0], [0], marker="o", color="none", markerfacecolor=COLOR_MUTED, markeredgecolor=COLOR_SURFACE, markersize=9, label="lossless (bit-exact invertible)"),
        plt.Line2D([0], [0], marker="^", color="none", markerfacecolor=COLOR_MUTED, markeredgecolor=COLOR_SURFACE, markersize=9, label="lossy"),
    ]
    if any(not a.get("lossless", True) for a in attempts):
        ax.legend(handles=legend_handles, loc="lower right", frameon=False, labelcolor=COLOR_INK, fontsize=9)
    else:
        ax.text(0.99, 0.02, "all methods tested are lossless (bit-exact invertible)", transform=ax.transAxes,
                ha="right", va="bottom", fontsize=8.5, color=COLOR_MUTED, style="italic")

    ax.set_xlabel("time to compress (s) — lower is faster", color=COLOR_INK, fontsize=11)
    ax.set_ylabel("compression ratio (raw / compressed) — higher is better", color=COLOR_INK, fontsize=11)
    ax.set_title("compress.py: time vs. compression ratio Pareto frontier (100-frame subset)", color=COLOR_INK, fontsize=13, pad=28)
    ax.grid(True, color=COLOR_GRID, linewidth=1, zorder=0)
    for spine in ("top", "right"): ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"): ax.spines[spine].set_color(COLOR_GRID)
    ax.tick_params(colors=COLOR_MUTED)
    ax.set_xlim(-1, max(a["seconds"] for a in attempts) * 1.25)
    ax.set_ylim(0, max(a["ratio"] for a in attempts) * 1.25)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
