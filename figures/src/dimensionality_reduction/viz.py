"""Interactive 3-D dimensionality-reduction explorer (server).

PCA / UMAP of the model's input, latent or output vectors. Each sample is drawn as its object's picture
(assets/objects/individual) with a contour in the point colour, in 3-D (three.js, plotly-style walls / ticks) or 2-D.
Everything except run / epoch / port is a dropdown in the page (viz.html, next to this file).

  .venv/bin/python figures/src/dimensionality_reduction/viz.py --run decres-rescale-aug10-v1 [--epoch 200] [--port 8610]
  .venv/bin/python figures/src/dimensionality_reduction/viz.py --run decres-rescale-aug10-v1 --precompute   # cache every dropdown subset, then exit

Caches (runs/_src9_cache/nb137/<run>[/ep<N>]):
  feats.npz + meta.csv   one eager fp32 pass of the checkpoint (transformer4 runs only)
  red_*.npz              one file per PCA / UMAP fit
"""
import argparse
import base64
import colorsys
import functools
import hashlib
import html
import http.server
import io
import itertools
import json
import socket
import sys
import time
import traceback
import urllib.parse
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator
from PIL import Image
from scipy.ndimage import binary_dilation
from scipy.stats import spearmanr
from sklearn.decomposition import PCA

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]  # figures/src/<this dir> → repo root
PAGE = HERE / "viz.html"

# ---------------------------------------------------------------- settings

# Splits containing any of these are dropped. IGNORE and UMAP are part of the reduction cache key.
IGNORE = ["eval/cardboard/vase-laser-shifted-4", "shifted", "unseen-speaker", "cylinder", "portable-charger", "biscoff", "coffee", "lid"]
UMAP = dict(n_neighbors=30, min_dist=0.1)

# Display-picker entries containing any of these start unticked.
DISPLAY_OFF = ["shifted", "unseen-speaker", "cylinder", "portable-charger", "biscoff", "coffee", "lid"]

BOXES = {
    "2026_09_27_gastronorm_four_objs": "gastro",
    "2026_10_05_green_plastic_four_objs": "plastic",
    "2026_10_11_cardboard_four_objs": "cardboard",
}
# Object name shown in the page.
RENAME = {
    "green-cube+red-cube": "two objects",
    "red-cube+vase": "two objects",
    "soap dispenser": "soap",
    "red-cube": "cube",
}
# Object name -> picture file in assets/objects/individual.
IMAGES = {
    "candle": "candle",
    "red-cube": "cube",
    "soap dispenser": "soap",
    "vase": "vase",
    "green-cube+red-cube": "two_cubes",
    "red-cube+vase": "cube_and_vase",
    "coffee pot": "coffee_pot",
    "tomato sauce": "tomato_sauce",
}
# Values that always get the same colour.
FIXED_COLOURS = {
    "empty": "black",
    "two objects": "black",
    "gastro": "#2ca02c",
    "plastic": "#1f77b4",
    "cardboard": "#ff7f0e",
}
PALETTE = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]  # tab10
NAMED_COLOURS = {"black": (0, 0, 0), "lightgray": (0.83, 0.83, 0.83)}

CATEGORICAL = ["split", "set", "box", "object", "speaker"]
NUMERIC = ["x", "y", "iou", "area", "n_objects"]
RANKED = ["x", "y", "iou"]  # "best" scores these with Spearman rho^2, every other feature with eta^2
BEST = ["none", "xy", "split", "set", "box", "object", "speaker", "x", "y", "iou"]

# ---------------------------------------------------------------- arguments

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--run", default="decres-rescale-aug10-v1")
parser.add_argument("--epoch", type=int, default=None, help="checkpoint epoch (default: latest)")
parser.add_argument("--port", type=int, default=8610)
parser.add_argument("--precompute", action="store_true", help="fit + cache every dropdown subset (box × speaker × object × vector × method), then exit")
ARGS = parser.parse_args()

# ---------------------------------------------------------------- checkpoint and cache directory

CHECKPOINT_DIR = ROOT / "runs" / ARGS.run / "checkpoints"
LATEST = (CHECKPOINT_DIR / "latest-rank0.pt").resolve().name

if ARGS.epoch is None:
    CHECKPOINT = LATEST
else:
    matches = sorted(CHECKPOINT_DIR.glob(f"ep{ARGS.epoch}-*rank0.pt"))
    available = sorted(p.name for p in CHECKPOINT_DIR.glob("ep*"))
    assert matches, f"no checkpoint for epoch {ARGS.epoch} in {CHECKPOINT_DIR}: {available}"
    CHECKPOINT = matches[0].name

# The latest checkpoint caches in the run directory itself, any other epoch in an ep<N> subdirectory.
CACHE = ROOT / "runs/_src9_cache/nb137" / ARGS.run
if CHECKPOINT != LATEST:
    CACHE = CACHE / f"ep{ARGS.epoch}"
CACHE.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- feature extraction (only when not cached)


def extract_features():
    """One eager fp32 pass over every sample: writes feats.npz (input / latent / output vectors) and meta.csv."""
    import torch  # slow imports, only needed here

    sys.path.insert(0, str(ROOT / "src9"))
    from arch.transformer4 import Transformer4
    from dataset import build_dataset, load_splits

    run_args = json.load(open(ROOT / "runs" / ARGS.run / "args.json"))
    assert run_args["model"] == "transformer4", f"only transformer4 runs supported, got {run_args['model']}"
    print(f"extracting features from {CHECKPOINT} (one pass over every sample, a few minutes)...", flush=True)
    t0 = time.time()

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    _, evals, train_eval, info = build_dataset(run_args.get("window", "tukey"), num_workers=0)
    model = Transformer4(
        info,
        decoder=run_args.get("decoder", "two"),
        latent_norm=run_args.get("latent_norm", "ln"),
        freq_rope=run_args.get("freq_rope", "linear"),
    )
    model = model.cuda().eval()
    state = torch.load(CHECKPOINT_DIR / CHECKPOINT, map_location="cpu", weights_only=False)
    print(model.load_state_dict(state["state"]["model"]))

    # Every sample as (split name, index into the underlying dataset): train first, then each eval split.
    train_subset = train_eval.dataloader.dataloader.dataset
    eval_subsets = [(e.label, e.dataloader.dataloader.dataset) for e in evals]
    train_dataset = train_subset.dataset
    eval_dataset = eval_subsets[0][1].dataset
    samples = [("train", i) for i in train_subset.indices]
    for label, subset in eval_subsets:
        for i in subset.indices:
            samples.append((label, i))

    latents = []
    preds = []
    trues = []
    inputs = []
    with torch.no_grad():
        for start in range(0, len(samples), 128):
            rows = []
            for split, i in samples[start:start + 128]:
                dataset = train_dataset if split == "train" else eval_dataset
                rows.append(dataset[i])
            fft = torch.stack([row["fft"] for row in rows]).cuda()
            out = model(dict(fft=fft))
            latents.append(out["latent"].float().cpu())
            preds.append(out["mask_pred"].float().cpu())
            trues.append(torch.stack([row["mask_true"] > 0.5 for row in rows]))
            # input vector: mean log-magnitude per (laser, band) -> 100 x 18
            log_fft = (fft + 1e-3).log().view(len(fft), 100, 18, 65, 2)
            inputs.append(log_fft.mean((3, 4)).float().cpu())
            if start % 2560 == 0:
                print(f"  {start}/{len(samples)} {time.time() - t0:.0f}s", flush=True)
    latent = torch.cat(latents).numpy()
    pred = torch.cat(preds).numpy()
    true = torch.cat(trues).numpy()
    input_bands = torch.cat(inputs).numpy()

    # Per-sample metadata: centre of mass of the true mask (NaN for an empty box), IoU at threshold 0.08, mask area.
    pred_binary = pred > 0.08
    rows_grid, cols_grid = np.mgrid[:64, :64]
    area = true.sum((1, 2))
    safe_area = np.maximum(area, 1)
    has_object = area > 0
    com_y = np.where(has_object, (true * rows_grid).sum((1, 2)) / safe_area, np.nan)
    com_x = np.where(has_object, (true * cols_grid).sum((1, 2)) / safe_area, np.nan)
    intersection = (pred_binary & true).sum((1, 2))
    union = (pred_binary | true).sum((1, 2))
    iou = np.where(has_object, intersection / np.maximum(union, 1), np.nan)

    all_metas = list(load_splits()[0].values())
    metas = [all_metas[i] for _, i in samples]
    df = pd.DataFrame(dict(
        split=[split for split, _ in samples],
        box=[BOXES[m["experiment"]] for m in metas],
        object=["+".join(sorted(m["objects"])) or "empty" for m in metas],
        speaker=[str(m["speaker"]) for m in metas],
        pos=[m["position_id"] for m in metas],
        y=com_y,
        x=com_x,
        iou=iou,
        area=area,
    ))
    df["set"] = np.where(df.split == "train", "train", "eval")

    n = len(df)
    pooled_pred = pred.reshape(-1, 16, 4, 16, 4).mean((2, 4))  # 64x64 mask probabilities -> 16x16
    np.savez(
        CACHE / "feats.npz",
        input=input_bands.reshape(n, -1),
        latent=latent,
        output=pooled_pred.reshape(n, -1),
    )
    df.to_csv(CACHE / "meta.csv", index=False)
    print(f"features cached in {CACHE} ({time.time() - t0:.0f}s)")
    del model
    torch.cuda.empty_cache()


if not (CACHE / "feats.npz").exists():
    extract_features()

# ---------------------------------------------------------------- data

FEATS = dict(np.load(CACHE / "feats.npz"))
df = pd.read_csv(CACHE / "meta.csv", dtype={"speaker": str})
if "area" not in df:
    # old cache of the default run, written before the area column existed
    df["area"] = np.load(ROOT / "runs/_src9_cache/nb136/base.npz")["y"].sum((1, 2))

# display-picker entry: the eval split, or train/<object>
df["disp"] = np.where(df.split == "train", "train/" + df.object, df.split)
df["n_objects"] = np.where(df.object == "empty", 0, df.object.str.count(r"\+") + 1)
df["object_raw"] = df.object
df["object"] = df.object.replace(RENAME)

keep = ~df.split.isin(IGNORE).values
df = df[keep].reset_index(drop=True)
for name in FEATS:
    FEATS[name] = FEATS[name][keep]

# ---------------------------------------------------------------- sprites (object picture + contour ring, sent to the page once)


def png_data_uri(array):
    buffer = io.BytesIO()
    Image.fromarray(array).save(buffer, "PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


def sprite(raw_object):
    """[picture, ring] as 64 px PNG data URIs, or None when the object has no picture (the page draws a cross)."""
    path = ROOT / "assets/objects/individual" / f"{IMAGES.get(raw_object, '-')}.png"
    if not path.is_file():
        return None
    image = Image.open(path).convert("RGBA")
    image.thumbnail((54, 54))
    w, h = image.size
    top = (64 - h) // 2
    left = (64 - w) // 2
    picture = np.zeros((64, 64, 4), np.uint8)
    picture[top:top + h, left:left + w] = np.asarray(image)
    # ring = the 4 px band just outside the object's silhouette; the page tints it with the point colour
    opaque = picture[..., 3] > 128
    ring = np.zeros_like(picture)
    ring[binary_dilation(opaque, iterations=4) & ~opaque] = 255
    return [png_data_uri(picture), png_data_uri(ring)]


RAW_OBJECTS = sorted(df.object_raw.unique())
SPRITE_INDEX = {}
for i, raw_object in enumerate(RAW_OBJECTS):
    SPRITE_INDEX[raw_object] = i

# display picker rows: [entry, number of samples, ticked by default]
display_entries = []
for entry, count in df.disp.value_counts().items():
    ticked = not any(word in entry for word in DISPLAY_OFF)
    display_entries.append([entry, int(count), ticked])

META = dict(
    run=ARGS.run,
    ckpt=CHECKPOINT,
    objects=RAW_OBJECTS,  # same order as sprites; the page draws "empty" as a cross
    sprites=[sprite(raw_object) for raw_object in RAW_OBJECTS],
    box=["all", *sorted(df.box.unique())],
    speaker=["all", *sorted(df.speaker.unique(), key=int)],
    object=["all", *df.object.value_counts().index],
    vector=["latent", "input", "output"],
    split=["both", "train", "test"],
    method=["pca", "umap"],
    colour=["xy", *CATEGORICAL, *NUMERIC],
    size=["none", *CATEGORICAL, *NUMERIC],
    best=BEST,
    splits=display_entries,
)

# ---------------------------------------------------------------- reduction (cached in memory and on disk)


@functools.lru_cache(maxsize=64)
def reduce(box, speakers, objects, vector, method):
    """PCA (10 components) or UMAP (3) of one subset. box is a name or None; speakers / objects are a tuple or None.

    The reduction is fit on the train rows of the subset only, then every row of the subset is projected.
    Returns (subset mask over df, coordinates, explained variance ratio); the last two are None when the subset is too small.
    """
    mask = np.ones(len(df), bool)
    if box is not None:
        mask = mask & (df.box == box).values
    if speakers:
        mask = mask & df.speaker.isin(speakers).values
    if objects:
        mask = mask & df.object.isin(objects).values
    if mask.sum() < 10:
        return mask, None, None
    train = (df.set == "train").values[mask]
    if train.sum() < 10:
        return mask, None, None

    # The file name is a hash of this exact tuple: changing it orphans every existing red_*.npz.
    key = (method, sorted(IGNORE), box, list(speakers) if speakers else None, list(objects) if objects else None, vector)
    if method == "pca":
        key = key + ("k10",)
    else:
        key = key + tuple(sorted(UMAP.items()))
    key = key + ("fit=train",)
    path = CACHE / f"red_{hashlib.sha1(repr(key).encode()).hexdigest()[:10]}.npz"

    if not path.exists() or len(np.load(path)["r"]) != mask.sum():
        t0 = time.time()
        X = FEATS[vector][mask]
        X = X - X[train].mean(0)
        if method == "pca":
            pca = PCA(10, random_state=0).fit(X[train])
            coords = pca.transform(X)
            variance = pca.explained_variance_ratio_
        else:
            import umap  # slow import (numba), only needed for a new UMAP fit

            pca = PCA(min(50, *X[train].shape), random_state=0).fit(X[train])
            Z = pca.transform(X)
            n_neighbors = min(UMAP["n_neighbors"], int(train.sum()) - 1)
            reducer = umap.UMAP(n_components=3, n_neighbors=n_neighbors, min_dist=UMAP["min_dist"], random_state=0)
            reducer.fit(Z[train])
            coords = np.zeros((len(X), 3), np.float32)
            coords[train] = reducer.embedding_
            if (~train).any():
                coords[~train] = reducer.transform(Z[~train])
            variance = np.full(3, np.nan)
        # write to a temp file, then rename: a running server never sees a half-written file
        tmp = path.with_suffix(".tmp.npz")
        np.savez(tmp, r=coords, ev=variance)
        tmp.replace(path)
        print(f"fitted {method} {box or 'all'} {vector} on {train.sum()} train, projected {len(X)} in {time.time() - t0:.0f}s", flush=True)

    saved = np.load(path)
    coords = saved["r"]
    if "ev" in saved:
        variance = saved["ev"]
    else:
        variance = np.full(3, np.nan)
    return mask, coords, variance


# ---------------------------------------------------------------- point colours and sizes


def to_rgb(colour):
    """'black' / 'lightgray' / '#rrggbb' -> (r, g, b) in 0..1."""
    if colour in NAMED_COLOURS:
        return NAMED_COLOURS[colour]
    red = int(colour[1:3], 16) / 255
    green = int(colour[3:5], 16) / 255
    blue = int(colour[5:7], 16) / 255
    return red, green, blue


def to_hex(rgb):
    """(r, g, b) in 0..1 -> '#rrggbb', quantised to 16 levels per channel so the page needs few distinct tinted sprites."""
    out = "#"
    for channel in rgb:
        level = round(channel * 15)
        out += f"{int(round(level * 17)):02x}"
    return out


def colours(d, feature):
    """Per-point hex colours and the legend {label: colour} for one colour-by feature."""
    if feature == "xy":
        # x = hue, y = lightness of the true mask centre of mass; no legend
        xy = d[["x", "y"]]
        low = xy.min().values
        high = xy.max().values
        unit = np.clip((xy.values - low) / (high - low), 0, 1)
        out = []
        for name, (u, v) in zip(d.object, unit):
            if name in FIXED_COLOURS:
                out.append(to_hex(to_rgb(FIXED_COLOURS[name])))
            elif np.isnan(u):
                out.append(to_hex(to_rgb("lightgray")))
            else:
                out.append(to_hex(colorsys.hls_to_rgb(0.8 * u, 0.8 - 0.55 * v, 0.9)))
        return out, {}

    if feature in CATEGORICAL:
        # The colour map covers ALL values in the data (not just this subset), so a value keeps its colour in every view.
        values = df[feature].astype(str).unique()
        numbers = sorted((v for v in values if v.isdigit()), key=int)
        words = sorted(v for v in values if not v.isdigit())
        colour_of = {}
        for i, value in enumerate(numbers + words):
            colour = FIXED_COLOURS.get(value, PALETTE[i % len(PALETTE)])
            colour_of[value] = to_hex(to_rgb(colour))
        labels = d[feature].astype(str)
        shown = set(labels)
        legend = {}
        for value, colour in colour_of.items():
            if value in shown:
                legend[value] = colour
        return [colour_of[label] for label in labels], legend

    # numeric: viridis from min to max (NaN drawn as the minimum)
    viridis = matplotlib.colormaps["viridis"]
    values = d[feature].astype(float)
    low = values.min()
    high = values.max()
    unit = ((values - low) / (high - low + 1e-12)).fillna(0)
    out = [to_hex(viridis(u)[:3]) for u in unit]
    legend = {}
    legend[f"{low:.3g}"] = to_hex(viridis(0.0)[:3])
    legend[f"{high:.3g}"] = to_hex(viridis(1.0)[:3])
    return out, legend


def sizes(d, feature):
    """Per-point image scale in 0.5..1.5 and the legend {label: scale} for one size-by feature."""
    if feature == "none":
        return np.ones(len(d)), {}

    if feature in CATEGORICAL:
        # one evenly spaced level per label, in sorted order
        labels = d[feature].astype(str)
        levels = sorted(labels.unique())
        steps = max(len(levels) - 1, 1)
        code_of = {}
        legend = {}
        for i, level in enumerate(levels):
            code_of[level] = i
            legend[level] = round(0.5 + i / steps, 2)
        unit = labels.map(code_of).values / steps
        return 0.5 + unit, legend

    # numeric: min..max in 8 bins (NaN = smallest)
    values = d[feature].astype(float)
    low = values.min()
    high = values.max()
    unit = ((values - low) / (high - low + 1e-12)).fillna(0).values
    legend = {}
    legend[f"{low:.3g}"] = 0.5
    legend[f"{high:.3g}"] = 1.5
    return 0.5 + np.round(unit * 7) / 7, legend


def normalise(values):
    """Rescale to -1..1 (the cube the page draws)."""
    span = np.ptp(values) or 1
    return 2 * (values - values.min()) / span - 1


def ticks(values):
    """Axis ticks as [position in -1..1, label]."""
    low = float(values.min())
    high = float(values.max())
    span = (high - low) or 1
    out = []
    for tick in MaxNLocator(6).tick_values(low, high):
        if low <= tick <= high:
            out.append([2 * (tick - low) / span - 1, f"{tick:.3g}"])
    return out


# ---------------------------------------------------------------- "best" components for a feature


def eta_squared(values, groups):
    """Between-group variance / total variance."""
    between = 0
    for group in np.unique(groups):
        members = values[groups == group]
        between += len(members) * (members.mean() - values.mean()) ** 2
    total = ((values - values.mean()) ** 2).sum()
    return between / total


def rho_squared(values, target):
    """Squared Spearman correlation, ignoring NaN targets."""
    ok = ~np.isnan(target)
    return spearmanr(values[ok], target[ok])[0] ** 2


def best_map_components(coords, d, n):
    """The component pair that best matches the true (x, y) positions, plus a 3rd component when n == 3.

    Pair score: Procrustes R^2 between the per-position centroids on the pair and the mask centres of mass.
    3rd component: the remaining one that (x, y) explains best linearly.
    """
    n_components = coords.shape[1]
    names = [f"c{i}" for i in range(n_components)]
    with_coords = d.copy()
    for i, name in enumerate(names):
        with_coords[name] = coords[:, i]
    per_position = with_coords.groupby("pos").mean(numeric_only=True).dropna(subset=["x"])
    target = per_position[["x", "y"]].values
    target = target - target.mean(0)
    centroids = per_position[names].values
    centroids = centroids - centroids.mean(0)

    fit = {}
    for a, b in itertools.combinations(range(n_components), 2):
        pair = centroids[:, [a, b]]
        singular_values = np.linalg.svd(target.T @ pair, compute_uv=False)
        fit[(a, b)] = singular_values.sum() ** 2 / ((target ** 2).sum() * (pair ** 2).sum())
    picked = list(max(fit, key=fit.get))
    text = f"map-fit R² {fit[tuple(picked)]:.2f}"

    # R^2 of each component regressed on (x, y, 1)
    ok = d.x.notna().values
    design = np.c_[d.x.values[ok], d.y.values[ok], np.ones(ok.sum())]
    weights = np.linalg.lstsq(design, coords[ok], rcond=None)[0]
    residual = coords[ok] - design @ weights
    r2 = 1 - residual.var(0) / coords[ok].var(0)
    if n == 3:
        for i in np.argsort(-r2):
            if int(i) not in picked:
                picked.append(int(i))
                break
        text += f" · 3rd: xy R² {r2[picked[2]]:.2f}"
    return picked, text


def best(coords, d, feature, n):
    """The n components (0-based) that carry `feature` best, and a text with their scores."""
    if feature == "xy":
        return best_map_components(coords, d, n)
    if d[feature].nunique() < 2:
        return list(range(n)), f"{feature} is constant here"

    scores = []
    for i in range(coords.shape[1]):
        if feature in RANKED:
            scores.append(rho_squared(coords[:, i], d[feature].astype(float).values))
        else:
            scores.append(eta_squared(coords[:, i], d[feature].astype(str).values))
    scores = np.array(scores)
    picked = [int(i) for i in np.argsort(-scores)[:n]]
    text = " ".join(f"{i + 1}:{scores[i]:.2f}" for i in picked)
    if feature in RANKED:
        text += " (ρ²)"
    else:
        text += " (η²)"
    return picked, text


# ---------------------------------------------------------------- one plot request


def plot(query):
    """The /api/plot response for one set of dropdown values (query = parse_qs of the URL)."""

    def get(name, default=None):
        return query.get(name, [default])[0]

    box = get("box", "all")
    speaker = get("speaker", "all")
    object_name = get("object", "all")
    vector = get("vector", "latent")
    split = get("split", "both")
    method = get("method", "pca")
    colour_by = get("colour", "xy")
    size_by = get("size", "none")
    best_feature = get("best", "none")

    if box == "all":
        box = None
    speakers = None if speaker == "all" else (speaker,)
    objects = None if object_name == "all" else (object_name,)
    mask, coords, variance = reduce(box, speakers, objects, vector, method)
    if coords is None:
        return dict(error=f"only {mask.sum()} samples (< 10)")

    # What to draw: the ticked display entries and the chosen split. This never affects the fit.
    display = get("display", "").split(",")
    visible = df.disp.isin(display).values[mask]
    if split != "both":
        is_train = df.set.values[mask] == "train"
        if split == "train":
            visible = visible & is_train
        else:
            visible = visible & ~is_train
    if not visible.any():
        return dict(error=f"nothing to display (split {split}, {len(display)} ticked display entries) in this subset")
    d = df[mask][visible].reset_index(drop=True)
    coords = coords[visible]

    # Which components go on the axes: the x / y / z dropdowns, or the best ones for a feature.
    want_3d = get("z", "none") != "none"
    components = []
    for axis in ("x", "y", "z"):
        if get(axis, "none") != "none":
            components.append(int(get(axis)) - 1)
    best_text = ""
    if best_feature != "none":
        components, best_text = best(coords, d, best_feature, 3 if want_3d else 2)
    if len(components) < 2 or max(components) >= coords.shape[1]:
        return dict(error=f"pick 2-3 components ≤ {coords.shape[1]} for {method}")
    is_3d = len(components) == 3

    axis_names = []
    for i in components:
        if method == "pca":
            axis_names.append(f"PC{i + 1} ({variance[i]:.0%})")
        else:
            axis_names.append(f"UMAP {i + 1}")

    columns = [normalise(coords[:, i]) for i in components]
    if not is_3d:
        columns.append(np.zeros(len(d)))
    xyz = np.stack(columns, 1).round(4)

    point_colours, colour_legend = colours(d, colour_by)
    # The empty box is always a black cross, whatever the colour-by feature is.
    for i in np.flatnonzero((d.object_raw == "empty").values):
        point_colours[i] = "#000000"
    point_sizes, size_legend = sizes(d, size_by)

    # one row per point: [x, y, z, sprite index, colour, hover text, size]
    points = []
    for i in range(len(d)):
        hover = f"{d.split[i].split('/')[-1]} · {d.object[i]} · spk {d.speaker[i]} · pos {d.pos[i]} · true {d.x[i]:.0f},{d.y[i]:.0f} · iou {d.iou[i]:.2f}"
        points.append([
            *xyz[i].tolist(),
            SPRITE_INDEX[d.object_raw[i]],
            point_colours[i],
            html.escape(hover),
            round(float(point_sizes[i]), 3),
        ])

    # The title lists every setting. The page appends the two sliders (image size, density), which never reach the server.
    n_ticked = len(set(display) & set(df.disp))
    n_with_xy = int(d.x.notna().sum())
    title = f"{ARGS.run} · {CHECKPOINT.split('-')[0]} · box {box or 'all'} · speaker {speaker} · object {object_name}"
    title += f" · {vector} ({FEATS[vector].shape[1]}-d) · {method} fit on train · showing {split}"
    title += f" · display {n_ticked}/{df.disp.nunique()} splits"
    title += f" · n {len(d)} points ({n_with_xy} with a true x/y, {d.pos.nunique()} positions)"
    title += f" · axes {' × '.join(axis_names)}"
    title += f" · colour {colour_by} · size {size_by}"
    if colour_by == "xy":
        title += " (x = hue, y = lightness of the true mask CoM)"
    if best_feature != "none":
        title += f" · best {best_feature}: {best_text}"

    return dict(
        title=title,
        axes=axis_names,
        ticks=[ticks(coords[:, i]) for i in components],
        d3=is_3d,
        legend=colour_legend,
        size_legend=size_legend,
        picked=[i + 1 for i in components],
        ncomp=int(coords.shape[1]),
        pts=points,
    )


# ---------------------------------------------------------------- --precompute: cache every dropdown subset, then exit


def precompute():
    speaker_names = META["speaker"][1:]
    object_names = META["object"][1:]
    box_names = [None, *META["box"][1:]]

    # PCA first (fast), then UMAP. Within each: no filter -> one speaker -> one object -> speaker x object.
    filters = [(None, None)]
    for speaker in speaker_names:
        filters.append((speaker, None))
    for object_name in object_names:
        filters.append((None, object_name))
    for speaker in speaker_names:
        for object_name in object_names:
            filters.append((speaker, object_name))

    jobs = []
    for method in ("pca", "umap"):
        for speaker, object_name in filters:
            for box in box_names:
                for vector in ("latent", "input", "output"):
                    jobs.append((method, speaker, object_name, box, vector))

    t0 = time.time()
    done = 0
    skipped = 0
    for n, (method, speaker, object_name, box, vector) in enumerate(jobs):
        speakers = (speaker,) if speaker else None
        objects = (object_name,) if object_name else None
        try:
            _, coords, _ = reduce(box, speakers, objects, vector, method)
            if coords is None:
                skipped += 1
            else:
                done += 1
        except Exception as e:
            print(f"FAILED {method} box={box} spk={speaker} obj={object_name} {vector}: {e!r}", flush=True)
        if n % 50 == 0:
            print(f"[{n}/{len(jobs)}] {time.time() - t0:.0f}s · {done} cached · {skipped} too small", flush=True)
    print(f"precompute done: {done} fits cached, {skipped} subsets too small, {time.time() - t0:.0f}s")


if ARGS.precompute:
    precompute()
    sys.exit()

# ---------------------------------------------------------------- server


class Handler(http.server.BaseHTTPRequestHandler):
    def send(self, body, content_type="application/json"):
        data = body.encode()
        self.send_response(200)
        self.send_header("Content-Type", content_type + "; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        url = urllib.parse.urlparse(self.path)
        try:
            if url.path == "/":
                # re-read on every request: edit the page, reload the browser
                self.send(PAGE.read_text(), "text/html")
            elif url.path == "/api/meta":
                self.send(json.dumps(META))
            elif url.path == "/api/plot":
                t0 = time.time()
                out = plot(urllib.parse.parse_qs(url.query))
                self.send(json.dumps(out))
                print(f"plot {url.query} {time.time() - t0:.1f}s", flush=True)
            else:
                self.send_error(404)
        except Exception as e:
            # show the error in the page instead of a dead request
            traceback.print_exc()
            self.send(json.dumps(dict(error=repr(e))))

    def log_message(self, *args):
        pass


server = http.server.HTTPServer(("", ARGS.port), Handler)  # single-threaded: one fit at a time
print(f"\n{ARGS.run} · {CHECKPOINT} · {len(df)} samples")
print(f"  open → http://localhost:{ARGS.port}/")
print(f"  remote? ssh -L {ARGS.port}:localhost:{ARGS.port} {socket.gethostname()}  then open the URL above (Ctrl-C to stop)", flush=True)
server.serve_forever()
