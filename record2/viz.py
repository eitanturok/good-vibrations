"""Draw smask / coverage / shifts / freqs onto a given matplotlib Axes. Never pyplot: callers make
their own Figure, so these are safe from any thread."""

import numpy as np
from matplotlib.colors import to_rgb

COLOR_WORDS = ["red", "green", "blue", "yellow", "purple", "orange", "pink", "black", "white", "gray", "brown"]


def _object_color(name: str, idx: int, colormap) -> tuple:
    # always RGB: a colormap returns RGBA, and callers append their own alpha
    return next((to_rgb(w) for w in COLOR_WORDS if w in name.lower()), colormap(idx % colormap.N)[:3])


def draw_smask(ax, seg_results: list[dict], crop_overhead: np.ndarray, object_names: list[str], title: str = "Segmentation"):
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
    ax.set_title(title, fontsize=20)


N_RECENT = 10  # the last 10 positions each get their own color, navy (newest) -> pale; older ones are all gray


def outline(mask: np.ndarray, thickness: int = 1) -> np.ndarray:
    """The mask's boundary, drawn inside it and thick enough (~0.5% of the image, x `thickness`) to survive the plot's downsampling."""
    inner = mask
    for _ in range(thickness * max(1, max(mask.shape) // 200)):
        p = np.pad(inner, 1)
        inner = inner & p[:-2, 1:-1] & p[2:, 1:-1] & p[1:-1, :-2] & p[1:-1, 2:]
    return mask & ~inner


def resize(mask: np.ndarray, shape) -> np.ndarray:
    """Nearest-neighbour resize: the mask as fractions of its box, on a grid of `shape`."""
    return mask[np.arange(shape[0]) * mask.shape[0] // shape[0]][:, np.arange(shape[1]) * mask.shape[1] // shape[1]]


def add_coverage(entry: dict | None, smask: np.ndarray, masks=()) -> dict:
    """A NEW coverage entry with one position's objects added (`entry` is not changed). `last_seen`:
    per pixel, the latest position covering it (0 = never); `outlines`: every position's object
    outlines; `last_masks`: the latest position's objects. An empty box (all-False smask) adds nothing.
    Masks are where on the box (the crop) the objects are: a position cropped to another size is
    resized onto the entry's grid, never dropped."""
    if entry is None:
        entry = {"n_positions": 0, "last_seen": np.zeros(smask.shape, np.int32), "outlines": np.zeros(smask.shape, bool), "last_masks": []}
    if not smask.any():
        return entry
    shape = entry["last_seen"].shape
    smask, masks = resize(smask, shape), [resize(m, shape) for m in masks or [smask]]
    n, last_seen, outlines = entry["n_positions"] + 1, entry["last_seen"].copy(), entry["outlines"].copy()
    last_seen[smask] = n
    outlines &= ~smask  # a newer position covers the older ones, their outlines too
    for m in masks:
        outlines |= outline(m)
    return {"n_positions": n, "last_seen": last_seen, "outlines": outlines, "last_masks": masks}


def draw_coverage(ax, entry: dict, layout: str):
    """Where objects have been on the box floor, for one layout: the last N_RECENT positions from navy (the
    newest, on top) through teal to pale -- brightness AND hue change, so each is told apart -- older ones
    gray; every object outlined in black, the latest position's in red."""
    import matplotlib
    colors = matplotlib.colormaps["YlGnBu"](np.linspace(0.95, 0.2, N_RECENT))[:, :3]  # by age, newest first
    seen, n = entry["last_seen"], entry["n_positions"]
    age = n - seen  # 0 = the newest position
    img = np.ones((*seen.shape, 3))  # never covered: white
    img[(seen > 0) & (age >= N_RECENT)] = 0.88
    recent = (seen > 0) & (age < N_RECENT)
    img[recent] = colors[age[recent]]
    img[entry["outlines"]] = 0
    for m in entry["last_masks"]:
        img[outline(m, thickness=2)] = (0.85, 0.1, 0.1)  # twice as thick: it shows on the navy
    ax.imshow(img, interpolation="antialiased")
    ax.set_xticks([]), ax.set_yticks([])
    ax.set_title(f"{layout} coverage ({n} positions)", fontsize=20)


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
    ax.legend(ncols=2, loc="upper right")
    ax.set_title(title or f"Shifts laser={laser_idx}", fontsize=14)


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
    ax.legend(ncols=2, loc="upper right")
    ax.set_title(title or f"FFT Magnitude laser={laser_idx}", fontsize=14)


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
