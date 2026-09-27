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
    # always RGB: a colormap returns RGBA, and callers append their own alpha
    return next((to_rgb(w) for w in COLOR_WORDS if w in name.lower()), colormap(idx % colormap.N)[:3])


def draw_smask(ax, seg_results: list[dict], crop_overhead: np.ndarray, object_names: list[str]):
    """One colored overlay per object's mask(s), on a black background -- ports
    src/record.ipynb cell 67's plot_smask()."""
    import matplotlib
    img_h, img_w = crop_overhead.shape[:2]
    ax.imshow(np.zeros((img_h, img_w, 3)))
    colormap = matplotlib.colormaps["tab10"]  # cm.get_cmap was removed in matplotlib 3.9
    for obj_idx, (name, result) in enumerate(zip(object_names, seg_results)):
        color = _object_color(name, obj_idx, colormap)
        for mask in result.get("masks", []):
            overlay = np.zeros((*mask.shape, 4))
            overlay[mask] = (*color, 0.7)
            ax.imshow(overlay)
    ax.axis("off")


N_RECENT_POSITIONS = 5  # the last 5 positions fade from dark to light; older ones all share the lightest color


def add_coverage(coverage: dict, layout: str, smask: np.ndarray, position_id: int | None):
    """Accumulate one sample's smask into coverage[layout], in place and O(pixels): `last_seen` is
    the per-pixel index of the latest position covering it (0 = never), so recency needs no
    history replay; `last_mask` is the latest sample's smask. Every speaker of a position shares
    one smask, so a position is one step however many speakers it has. An empty box (all-False
    smask) only bumps n_samples."""
    entry = coverage.get(layout)
    if entry is not None and entry["last_seen"].shape != smask.shape:
        entry = None  # shape guard -- crop is GUI-editable
    if entry is None:
        entry = {"n_samples": 0, "n_positions": 0, "position_id": None,
                 "last_seen": np.zeros(smask.shape, dtype=np.int32), "last_mask": smask}
    entry["n_samples"] += 1
    entry["last_mask"] = smask
    if smask.any():
        if position_id is None or entry["position_id"] != position_id:
            entry["position_id"], entry["n_positions"] = position_id, entry["n_positions"] + 1
        entry["last_seen"][smask] = entry["n_positions"]
    coverage[layout] = entry


def coverage_age(entry: dict) -> np.ndarray:
    """Per pixel: how many positions ago it was last covered (0 = the latest), capped at
    N_RECENT_POSITIONS; NaN where nothing was ever covered."""
    age = np.minimum(entry["n_positions"] - entry["last_seen"], N_RECENT_POSITIONS).astype(np.float32)
    age[entry["last_seen"] == 0] = np.nan
    return age


def draw_coverage(ax, entry: dict, layout: str):
    """Where objects have been on the box floor, for one layout, colored only by recency: the
    latest position darkest, lighter over the last N_RECENT_POSITIONS, older ones all the same
    light blue; a red border around the latest sample."""
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import ListedColormap, Normalize
    import matplotlib
    cmap = ListedColormap(matplotlib.colormaps["Blues"](np.linspace(1.0, 0.25, 256)))  # dark (new) -> light, never white
    norm = Normalize(vmin=0, vmax=N_RECENT_POSITIONS)
    ax.imshow(coverage_age(entry), cmap=cmap, norm=norm, interpolation="nearest")  # NaN (never covered) is left blank
    if entry["last_mask"].any():
        ax.contour(entry["last_mask"], levels=[0.5], colors="red", linewidths=1.5)
    cbar = ax.figure.colorbar(ScalarMappable(norm, cmap), ax=ax, label="positions ago")
    cbar.set_ticks(range(N_RECENT_POSITIONS + 1), labels=[*map(str, range(N_RECENT_POSITIONS)), f"{N_RECENT_POSITIONS}+"])
    ax.set_title(f"Box Coverage {layout} ({entry['n_positions']} positions)")


def draw_shifts(ax, shifts: np.ndarray, fps: float, laser_idx: int, title: str | None = None):
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
    ax.set_title(title or f"Shifts laser={laser_idx}")


def draw_freqs(ax, fft: np.ndarray, freqs: np.ndarray, laser_idx: int, title: str | None = None):
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
    ax.set_title(title or f"FFT Magnitude laser={laser_idx}")


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
