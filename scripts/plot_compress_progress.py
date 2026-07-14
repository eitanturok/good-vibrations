"""Plot compress.py's method sweep: compressed size (bytes) per method tried.

Reads src/data/compress_attempts.json (written by compress.py) and plots
compressed_bytes for every method, in the order tried, labeling each point with
its method name. The smallest (best) result is highlighted; timed-out/skipped
methods (compressed_bytes is null) are marked separately and excluded from sizing.

Usage: python src/data/plot_compress_progress.py [data.json] [out.png]
Defaults to compress_attempts.json / compress_progress.png if no args given.
"""
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt

COLOR_GOOD = "#0ca30c"
COLOR_CRITICAL = "#d03b3b"
COLOR_MUTED = "#898781"
COLOR_INK = "#0b0b0b"
COLOR_GRID = "#e1e0d9"
COLOR_SURFACE = "#fcfcfb"


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    data_path = Path(args[0]) if len(args) > 0 else Path(__file__).parent / "compress_attempts.json"
    out_path = Path(args[1]) if len(args) > 1 else Path(__file__).parent / "compress_progress.png"

    attempts = json.loads(data_path.read_text())
    x = list(range(len(attempts)))
    labels = [a["name"] for a in attempts]
    sizes = [a["compressed_bytes"] for a in attempts]
    skipped = [s is None for s in sizes]

    best_idx = min((i for i in x if not skipped[i]), key=lambda i: sizes[i])

    fig, ax = plt.subplots(figsize=(12, 7), facecolor=COLOR_SURFACE)
    ax.set_facecolor(COLOR_SURFACE)

    plot_x = [xi for xi in x if not skipped[xi]]
    plot_y = [sizes[xi] for xi in plot_x]
    ax.plot(plot_x, plot_y, color=COLOR_GRID, linewidth=1.5, zorder=1)

    for xi in x:
        if skipped[xi]:
            ax.scatter([xi], [0], s=90, marker="x", color=COLOR_CRITICAL, zorder=3)
            ax.annotate(
                f"{labels[xi]}\n(timed out)", (xi, 0), textcoords="offset points", xytext=(0, 14),
                ha="center", va="bottom", fontsize=8.5, color=COLOR_CRITICAL, linespacing=1.3,
            )
            continue
        color = COLOR_GOOD if xi == best_idx else COLOR_MUTED
        ax.scatter([xi], [sizes[xi]], s=110 if xi == best_idx else 80, color=color, zorder=3, edgecolors=COLOR_SURFACE, linewidths=1.5)
        mb = sizes[xi] / 1e6
        offset = -22 if xi % 2 == 0 else 22
        va = "top" if offset < 0 else "bottom"
        ax.annotate(
            f"{labels[xi]}\n{mb:.2f} MB", (xi, sizes[xi]), textcoords="offset points", xytext=(0, offset),
            ha="center", va=va, fontsize=8.5, color=COLOR_INK, linespacing=1.3,
        )

    ax.set_yscale("log")
    ax.set_xlabel("method (attempt order)", color=COLOR_INK, fontsize=11)
    ax.set_ylabel("compressed size (bytes, log scale)", color=COLOR_INK, fontsize=11)
    ax.set_title("compress.py: compressed size per method on 100-frame subset", color=COLOR_INK, fontsize=13, pad=28)
    ax.set_xticks(x)
    ax.set_xticklabels([])
    ax.grid(True, axis="y", color=COLOR_GRID, linewidth=1, zorder=0)
    for spine in ("top", "right"): ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"): ax.spines[spine].set_color(COLOR_GRID)
    ax.tick_params(colors=COLOR_MUTED)
    valid_sizes = [s for s in sizes if s is not None]
    ax.set_ylim(min(valid_sizes) * 0.5, max(valid_sizes) * 2.2)
    ax.set_xlim(-0.7, len(x) - 0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
