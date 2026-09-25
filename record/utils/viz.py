"""GUI-purpose-built visualization: draw smask/coverage/shifts/freqs onto a given Axes.

Every function here takes an existing `ax` and draws into it -- none of them create a
pyplot figure or call any `plt.*` function. `pyplot` keeps a global figure registry that
is not safe to touch from a background thread (see matplotlib's own threading notes); every
plot in this pipeline is instead produced from a Task running on its own thread, so the
caller must create its own `matplotlib.figure.Figure()` + axes (never `plt.subplots()`) and
pass those axes in here. This is a clean rewrite, not an import of the old (partly broken,
pyplot-based) utils/viz.py -- see record/record.ipynb's plot_smask/plot_coverage/
plot_shifts/plot_freqs (Section 9) for the Figure-per-call wiring that renders these onto
the GUI panels.
"""

import numpy as np
from matplotlib.colors import to_rgb

COLOR_WORDS = ["red", "green", "blue", "yellow", "purple", "orange", "pink", "black", "white", "gray", "brown"]


def _object_color(name: str, idx: int, colormap) -> tuple:
    return next((to_rgb(w) for w in COLOR_WORDS if w in name.lower()), colormap(idx % colormap.N))


def draw_smask(ax, seg_results: list[dict], crop_overhead: np.ndarray, object_names: list[str]):
    """One colored overlay per object's mask(s), on a black background -- ports
    src/record.ipynb cell 67's plot_smask()."""
    import matplotlib.cm as cm
    img_h, img_w = crop_overhead.shape[:2]
    ax.imshow(np.zeros((img_h, img_w, 3)))
    colormap = cm.get_cmap("tab10")
    for obj_idx, (name, result) in enumerate(zip(object_names, seg_results)):
        color = _object_color(name, obj_idx, colormap)
        for mask in result.get("masks", []):
            overlay = np.zeros((*mask.shape, 4))
            overlay[mask] = (*color, 0.7)
            ax.imshow(overlay)
    ax.axis("off")


def draw_coverage(ax, coverage_mask: np.ndarray, n_samples: int, layout: str):
    """Heatmap of how much of the box floor has been covered by segmented objects so
    far, for one layout -- ports src/record.ipynb cell 67's plot_coverage()."""
    im = ax.imshow(coverage_mask, cmap="Blues")
    ax.figure.colorbar(im, ax=ax, label="coverage")
    ax.set_title(f"Box Coverage {layout} ({n_samples} samples)")


def draw_shifts(ax, shifts: np.ndarray, fps: float, laser_idx: int):
    """x/y pixel shift over time for one laser ROI -- ports src/record.ipynb cell 67's
    plot_shifts(), fixed to a single laser_idx (the live preview is always single-ROI)."""
    shifts = np.asarray(shifts).squeeze()
    x_shifts, y_shifts = shifts[:, 0], shifts[:, 1]
    t = np.arange(len(x_shifts)) / fps
    ax.plot(t, x_shifts, label="x")
    ax.plot(t, y_shifts, label="y")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Shift (pixels)")
    ax.legend()
    ax.set_title(f"Shifts laser={laser_idx}")


def draw_freqs(ax, fft: np.ndarray, freqs: np.ndarray, laser_idx: int):
    """FFT magnitude spectrum for one laser ROI -- ports src/record.ipynb cell 67's
    plot_fft_magnitude(), fixed to a single laser_idx."""
    fft = np.asarray(fft).squeeze()
    mag = np.abs(fft)
    mag_x, mag_y = mag[:, 0], mag[:, 1]
    ax.plot(freqs, mag_x, label="x")
    ax.plot(freqs, mag_y, label="y")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Magnitude")
    ax.legend()
    ax.set_title(f"FFT Magnitude laser={laser_idx}")


def figure_to_array(fig) -> np.ndarray:
    """Render a Figure (built with FigureCanvasAgg, never pyplot) to an (H, W, 4) RGBA
    array -- the hand-off format the GUI's frame queues expect, so a background thread
    never touches a Tk widget directly."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    w, h = canvas.get_width_height()
    buf = np.asarray(canvas.buffer_rgba())
    return buf.reshape(h, w, 4)
