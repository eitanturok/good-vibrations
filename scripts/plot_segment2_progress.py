"""Plot the segment2.py optimization attempts: time (ms) per attempt, in order tried.

Reads src/data/segment2_attempts.json (append a new {"label", "time_ms", "status"}
record after every new attempt, then rerun this script). status is one of:
  - "baseline": the starting point (gray)
  - "speedup":  faster than the previous kept point (green, on the trend line)
  - "rejected": tried but not adopted (red, plotted but NOT on the trend line)

Usage: python src/data/plot_segment2_progress.py [data.json] [out.png] [--log] [--title "..."]
Defaults to segment2_attempts.json / segment2_progress.png if no args given.
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
    log_scale = "--log" in sys.argv
    title = next((sys.argv[i + 1] for i, a in enumerate(sys.argv) if a == "--title"), "segment2.py: time per optimization attempt")
    data_path = Path(args[0]) if len(args) > 0 else Path(__file__).parent / "segment2_attempts.json"
    out_path = Path(args[1]) if len(args) > 1 else Path(__file__).parent / "segment2_progress.png"

    attempts = json.loads(data_path.read_text())
    x = list(range(len(attempts)))
    y = [a["time_ms"] for a in attempts]
    status = [a["status"] for a in attempts]
    labels = [a["label"] for a in attempts]

    fig, ax = plt.subplots(figsize=(10, 6.5), facecolor=COLOR_SURFACE)
    ax.set_facecolor(COLOR_SURFACE)

    # trend line: baseline + speedups only, in attempt order, skipping rejected points
    trend_x = [xi for xi, s in zip(x, status) if s != "rejected"]
    trend_y = [yi for yi, s in zip(y, status) if s != "rejected"]
    ax.plot(trend_x, trend_y, color=COLOR_GOOD, linewidth=2, zorder=2, solid_capstyle="round")

    for xi, yi, s, lab in zip(x, y, status, labels):
        color = {"baseline": COLOR_MUTED, "speedup": COLOR_GOOD, "rejected": COLOR_CRITICAL}[s]
        ax.scatter([xi], [yi], s=90, color=color, zorder=3, edgecolors=COLOR_SURFACE, linewidths=1.5)
        va = "bottom" if s == "rejected" else "top"
        offset = 14 if s == "rejected" else -14
        unit = "s" if max(y) > 3000 else "ms"
        val = yi / 1000 if unit == "s" else yi
        ax.annotate(
            f"{lab}\n{val:.2f} {unit}" if unit == "s" else f"{lab}\n{val:.0f} {unit}", (xi, yi), textcoords="offset points", xytext=(0, offset),
            ha="center", va=va, fontsize=9, color=COLOR_INK, linespacing=1.3,
        )

    if log_scale: ax.set_yscale("log")
    ax.set_xlabel("attempt number", color=COLOR_INK, fontsize=11)
    ax.set_ylabel("time (ms) — lower is faster" + (" (log scale)" if log_scale else ""), color=COLOR_INK, fontsize=11)
    ax.set_title(title, color=COLOR_INK, fontsize=13, pad=28)
    ax.set_xticks(x)
    ax.grid(True, axis="y", color=COLOR_GRID, linewidth=1, zorder=0)
    for spine in ("top", "right"): ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"): ax.spines[spine].set_color(COLOR_GRID)
    ax.tick_params(colors=COLOR_MUTED)
    if not log_scale: ax.set_ylim(0, max(y) * 1.25)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
