"""Draw smask / coverage / shifts / freqs onto a given matplotlib Axes. Never pyplot: callers make
their own Figure, so these are safe from any thread."""

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
    ax.set_title("Segmentation", fontsize=12)


N_RECENT = 10  # the last 10 positions each get their own color; older ones are all gray


def outline(mask: np.ndarray) -> np.ndarray:
    """The mask's boundary, drawn inside it and thick enough (~0.5% of the image) to survive the plot's downsampling."""
    inner = mask
    for _ in range(max(1, max(mask.shape) // 200)):
        p = np.pad(inner, 1)
        inner = inner & p[:-2, 1:-1] & p[2:, 1:-1] & p[1:-1, :-2] & p[1:-1, 2:]
    return mask & ~inner


def add_coverage(entry: dict | None, smask: np.ndarray, masks=()) -> dict:
    """A NEW coverage entry with one position's objects added (`entry` is not changed). `last_seen`:
    per pixel, the latest position covering it (0 = never); `outlines`: every position's object
    outlines; `last_masks`: the latest position's objects. An empty box (all-False smask) adds nothing."""
    if entry is None or entry["last_seen"].shape != smask.shape:  # new layout, or the crop changed
        entry = {"n_positions": 0, "last_seen": np.zeros(smask.shape, np.int32), "outlines": np.zeros(smask.shape, bool), "last_masks": []}
    if not smask.any():
        return entry
    masks = list(masks) or [smask]
    n, last_seen, outlines = entry["n_positions"] + 1, entry["last_seen"].copy(), entry["outlines"].copy()
    last_seen[smask] = n
    for m in masks:
        outlines |= outline(m)
    return {"n_positions": n, "last_seen": last_seen, "outlines": outlines, "last_masks": masks}


def draw_coverage(ax, entry: dict, layout: str):
    """Where objects have been on the box floor, for one layout: the last N_RECENT positions each in
    their own color (a newer position on top), older ones gray; every object outlined in black, the
    latest position's in red."""
    import matplotlib
    colors = np.array(matplotlib.colormaps["tab20"].colors)[[0, 2, 4, 8, 10, 12, 16, 18, 1, 5]]  # no red (latest) or gray (old)
    seen, n = entry["last_seen"], entry["n_positions"]
    img = np.ones((*seen.shape, 3))  # never covered: white
    old, recent = (seen > 0) & (n - seen >= N_RECENT), (seen > 0) & (n - seen < N_RECENT)
    img[old] = 0.8
    img[recent] = colors[seen[recent] % N_RECENT]  # a position keeps its color while it's recent
    img[entry["outlines"]] = 0
    for m in entry["last_masks"]:
        img[outline(m)] = (0.85, 0.1, 0.1)
    ax.imshow(img, interpolation="antialiased")
    ax.set_xticks([]), ax.set_yticks([])
    ax.set_title(f"Box Coverage {layout} ({n} positions)", fontsize=12)


def draw_shifts(ax, shifts: np.ndarray, fps: float, laser_idx: int, title: str | None = None):
    """x/y pixel shift over time for one laser ROI -- ports src/record.ipynb cell 67's
    plot_shifts(), fixed to a single laser_idx (the live preview is always single-ROI)."""
    shifts = np.asarray(shifts).squeeze()
    x_shifts, y_shifts = shifts[:, 0], shifts[:, 1]
    t = np.arange(len(x_shifts)) / fps
    ax.plot(t, x_shifts, label="x")
    ax.plot(t, y_shifts, label="y")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Shift (px)")
    ax.legend(ncols=2, frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1), borderaxespad=0)  # above the plot: the GUI draws it short and wide
    ax.set_title(title or f"Shifts laser={laser_idx}", loc="left")


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
    ax.legend(ncols=2, frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1), borderaxespad=0)  # above the plot: the GUI draws it short and wide
    ax.set_title(title or f"FFT Magnitude laser={laser_idx}", loc="left")


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
