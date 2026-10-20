"""Paper tables + figures from one run's saved outputs (runs/<run>/outputs_history), for notebooks/109_paper.ipynb.
  load(run)  -> one row per sample: every eval sample (set=eval) + the id splits' train samples (set=train)
  knn(run, df) -> the kNN-on-inputs baseline on the same eval samples
  table_id / table_ood / table_heads -> the tables, cells "mean ± std" over samples, with n train / n test samples
  plot_* (df), samples_figure(df) -> one figure each
Scores (as score.py): iou = hard IoU of pred > thr vs true > 0.5, miss = no pred pixel on the object, com_err = pred vs true mask
centroid distance in % of the box. A table's mean is over splits (every split counts once); its ± is the std over samples.
category = the paper question of a split (dataset.PAPER_CATEGORY). An ood split only answers its question if the run never trained
on that object (older datasets trained on some, e.g. cardboard's broken cup), so those are dropped. weight = the cylinders, all
trained: a trend of performance vs mass (plot_weight), not a generalization test.
Section 5 (notebook section 0) is the data itself, before any model: captures(run) + plot_coverage / plot_empty / table_empty /
plot_avg_fft / plot_spec_map / plot_spec_vs_pos / plot_envelope / plot_time_to_noise / plot_cumulative_energy / plot_mode_shapes /
plot_mode_changes / plot_objects / plot_box_resonances."""
import hashlib, json, sys, warnings
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap, to_hex, to_rgb
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from PIL import Image
from scipy.signal import butter, chirp, find_peaks, savgol_filter, sosfiltfilt

sys.path.insert(0, str(Path(__file__).parent))
import dataset as D
import score as S

ROOT = Path(__file__).resolve().parents[1]
THR = 0.1
BOXES = ("gastro", "plastic", "cardboard")
ID = D.PAPER_ID
OOD = ["unseen-position", "unseen-speaker", "unseen-object", "unseen-rotation", "unseen-laser", "lid", "broken"]  # never trained on at all
NAMES = {"id": "in distribution", "unseen-position": "unseen position", "unseen-speaker": "unseen speaker", "unseen-object": "unseen object",
         "unseen-rotation": "unseen rotation", "weight": "weight",
         "unseen-laser": "unseen speckles", "lid": "lid on", "broken": "broken object"}
OBJECT_Q = ("unseen-object", "broken")
METRICS = {"iou": "IoU ↑", "miss": "Miss rate ↓", "com_err": "Centroid error (% of box) ↓"}
LOWER = ("miss", "com_err")  # lower is better
NO_TWIN = ("2026_09_27_gastronorm_four_objs", (1254, 1255))  # gastro lid pair: 1255 has no without-lid mask (dataset.paper drops it too)
# the reference palette's categorical slots 1-3, fixed per entity; sequential = its blue ramp
COLORS = dict(zip(BOXES, ("#2a78d6", "#eb6834", "#1baf7a")))
SPK_COLORS = dict(zip((1, 3, 7), ("#2a78d6", "#eb6834", "#1baf7a")))
BLUES = LinearSegmentedColormap.from_list("blues", ["#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#104281"])
SEQ = LinearSegmentedColormap.from_list("seq", ["#f7fbff", "#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#104281", "#061f45"])  # white -> navy
INK, MUTED = "#0b0b0b", "#52514e"
plt.rcParams.update({"figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.alpha": .25,
                     "axes.edgecolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED, "font.size": 10})

#*** 1 load

def args(run): return json.loads((ROOT / "runs" / run / "args.json").read_text())

@lru_cache(maxsize=None)
def layouts(exp) -> dict[int, str]: return {m["position_id"]: m["layout"] for d in (D.EXPERIMENTS / exp / "samples").iterdir() if (d / "metadata.jsonl").exists() for m in [D.load_meta(d)]}

def trained_layouts(run) -> set[tuple[str, str]]:
    tr = pd.read_csv(ROOT / "runs" / run / "positions/train.txt", sep=" ", names=["experiment", "position_id"])
    return {(e, layouts(e)[p]) for e, p in zip(tr.experiment, tr.position_id)}

def trained_splits(run) -> set[tuple[str, str]]:
    """(box, split) of the run's dataset with any layout trained on"""
    a, trained = args(run), trained_layouts(run)
    splits = D.DATASETS[a["dataset"]](a["test_size"], None if "all" in a["speakers"] else tuple(a["speakers"]))
    return {tuple(k.split("/")) for k, sp in splits.items() if any((sp["experiment"], l) in trained for l in sp["layouts"])}

def score_mask(p, y, thr) -> dict:
    b, obj = p > thr, y > 0.5
    (br, bc), (rr, cc), (H, W) = np.nonzero(b), np.nonzero(obj), p.shape
    return dict(iou=(b & obj).sum() / max((b | obj).sum(), 1), miss=float(not (b & obj).any()),
                com_err=np.hypot((br.mean() - rr.mean()) / H, (bc.mean() - cc.mean()) / W) if len(br) and len(rr) else np.nan)

def load(run, epoch=None, thr=THR) -> pd.DataFrame:
    epoch = epoch or S.last_epoch(run)
    ev = S.per_sample(run, epoch, [thr]).rename(columns=lambda c: c.replace(f"@{thr}", ""))
    ev = pd.concat([ev, heads(run, epoch)], axis=1).assign(set="eval")
    df = pd.concat([ev, train(run, epoch, thr, ev)], ignore_index=True)
    df["com_err"] *= 100
    df = df[~((df.experiment == NO_TWIN[0]) & df.position_id.isin(NO_TWIN[1]))].copy()
    df["category"] = df.split.map(D.PAPER_CATEGORY)
    # object questions only: unseen speaker / position / angle reuse trained layouts at held-out positions by design. The weight
    # trend is the opposite: only cylinders the run trained on (else it's an unseen object as well as a weight)
    seen = pd.Series([k in trained_splits(run) for k in zip(df.box, df.split)], index=df.index)
    df.loc[df.category.isin(OBJECT_Q) & seen | (df.category == "weight") & ~seen, "category"] = None
    df["mass"] = df.split.str.extract(r"cylinder-(\d+)g", expand=False).astype(float)
    df.attrs.update(run=run, epoch=epoch, thr=thr)
    return df

def heads(run, epoch) -> pd.DataFrame:
    """the aux heads per eval sample, in score.per_sample's order (the same files, sorted)"""
    rows = []
    for f in sorted((ROOT / "runs" / run / "outputs_history/eval").glob(f"*/*/ep{epoch:04d}-ba*.pt")):
        d = torch.load(f, weights_only=False, map_location="cpu")
        m, conf = d["metrics"], d["object_logits"].float().softmax(-1).max(-1).values
        rows += [dict(file=str(f), idx=i, sample_id=d["info"]["sample_id"][i], obj_conf=float(conf[i]), object_acc=float(m["object_acc"][i]),
                      count_acc=float(m["n_objects_err"][i] == 0), box_acc=float(m["box_acc"][i]), com_head_err=100 * float(m["com_err"][i]),
                      area_err=float(m["area_rel_err"][i])) for i in range(len(conf))]
    return pd.DataFrame(rows)

def train(run, epoch, thr, ev) -> pd.DataFrame:
    """the id splits' train samples, scored like eval (from the train evaluator: no augmentation). Cached"""
    cache = ROOT / "runs" / run / "scores" / f"paper_train_ep{epoch:04d}_{thr}.pkl"
    if cache.exists(): return pd.read_pickle(cache)
    split_of = {(e, l): (b, s) for e, l, b, s in zip(ev.experiment, ev.layout, ev.box, ev.split) if s in ID}
    rows = []
    for f in sorted((ROOT / "runs" / run / "outputs_history/train").glob(f"ep{epoch:04d}-ba*.pt")):
        d = torch.load(f, weights_only=False, map_location="cpu")
        p, info = d["mask_pred"].float().numpy(), d["info"]
        for i in range(len(p)):
            e, sid, pos, spk = info["experiment"][i], info["sample_id"][i], int(info["position_id"][i]), int(info["speaker"][i])
            if (e, layouts(e)[pos]) not in split_of: continue
            b, s = split_of[e, layouts(e)[pos]]
            rows.append(dict(box=b, split=s, layout=layouts(e)[pos], experiment=e, position_id=pos, speaker=spk, set="train") | score_mask(p[i], S.truth(e, sid, pos, spk, *p.shape[-2:]), thr))
    out = pd.DataFrame(rows)
    cache.parent.mkdir(parents=True, exist_ok=True); out.to_pickle(cache)
    return out

#*** 2 kNN baseline: average the true masks of the k nearest train samples of the same box, in input space

def knn_features(exp) -> tuple[pd.DataFrame, np.ndarray]:
    """every sample -> log |FFT| in 58 bands of 20 bins (~15 Hz), (100 lasers x 58 x 2) flattened. Cached"""
    f = ROOT / f"runs/_paper_cache/knn2_{exp}.npz"
    if not f.exists():
        keys, X = [], []
        for d, m in D.collect_samples(D.EXPERIMENTS / exp).items():
            z = D.load_fft(d).abs().numpy()[:, :1160]
            X.append(np.log(z.reshape(len(z), 58, 20, -1).mean(2) + 1e-9).astype(np.float16).ravel())
            keys.append((int(d.name) if d.name.isdigit() else -1, m["position_id"], m["speaker"]))
        f.parent.mkdir(parents=True, exist_ok=True); np.savez(f, keys=np.array(keys), X=np.stack(X))
    z = np.load(f)
    return pd.DataFrame(z["keys"], columns=["sample_id", "position_id", "speaker"]), z["X"]

def knn(run, df, k=5, thr=0.5, dims=128) -> pd.DataFrame:
    """per box: standardize + PCA(dims) fit on the run's train samples; each eval sample's prediction = the mean true mask of its k
    nearest train samples, thresholded at thr (0.5 = most neighbours agree)"""
    from sklearn.decomposition import PCA
    from sklearn.neighbors import NearestNeighbors
    a = args(run)
    tr = pd.read_csv(ROOT / "runs" / run / "positions/train.txt", sep=" ", names=["experiment", "position_id"])
    ev, out = df[df.set == "eval"], []
    for exp, e in ev.groupby("experiment"):
        keys, X = knn_features(exp)
        ti = np.flatnonzero(keys.position_id.isin(tr[tr.experiment == exp].position_id) & ("all" in a["speakers"] or keys.speaker.isin(a["speakers"])))
        mu, sd = X[ti].astype(np.float32).mean(0), X[ti].astype(np.float32).std(0) + 1e-6
        pca = PCA(dims, random_state=0).fit((X[ti] - mu) / sd)
        mask = lambda j: S.truth(exp, *map(int, keys.iloc[j]), a["out_h"], a["out_w"])
        Y = np.stack([mask(j) for j in ti])
        at = {(p, s): j for j, (p, s) in enumerate(zip(keys.position_id, keys.speaker))}
        rows = [at[p, s] for p, s in zip(e.position_id, e.speaker)]
        _, nb = NearestNeighbors(n_neighbors=k).fit(pca.transform((X[ti] - mu) / sd)).kneighbors(pca.transform((X[rows] - mu) / sd))
        out += [dict(r[["box", "split", "category", "experiment", "position_id", "speaker", "set"]]) | score_mask(Y[n].mean(0), mask(j), thr)
                for (_, r), j, n in zip(e.iterrows(), rows, nb)]
    out = pd.DataFrame(out)
    out["com_err"] *= 100
    return out

#*** 3 tables: each returns its means and stds side by side (columns "mean" / "std"); style() shows them as "mean ±std",
# coloured by how good, best in bold, and to_latex() writes the same thing for the paper

def _ms(d, metric):
    """mean over splits (every split counts once), std over samples"""
    v = d[metric].dropna()
    return (d.groupby(["box", "split"])[metric].mean().mean(), v.std()) if len(v) else (np.nan, np.nan)

def _table(cells: dict, metric, **attrs) -> pd.DataFrame:
    """{(row, col): (mean, std)} -> columns mean / std, rows and columns in insertion order"""
    rows, cols = list(dict.fromkeys(r for r, _ in cells)), list(dict.fromkeys(c for _, c in cells))
    index = pd.MultiIndex.from_tuples(rows) if isinstance(rows[0], tuple) else rows
    t = pd.concat({k: pd.DataFrame([[cells[r, c][i] for c in cols] for r in rows], index=index, columns=cols) for i, k in enumerate(("mean", "std"))}, axis=1)
    t.attrs.update(metric=metric, lower=metric in LOWER, **attrs)
    return t

def _pick(d, box, col, field):
    return d[d.box.isin(BOXES if box == "all" else [box]) & (d[field].isin(ID if col == "eval" else [col]) if field == "split" else (d[field] == col))]

def table_id(methods: dict, metric="iou") -> pd.DataFrame:
    """Table 1, one metric: rows (method, box + all), columns = train, eval (the 6 id splits' mean) | eval per id split"""
    cells = {}
    for name, df in methods.items():
        ev, tr = df[df.set == "eval"], df[df.set == "train"]
        for b in BOXES + ("all",):
            cells |= {((name, b), "train"): _ms(_pick(tr, b, "eval", "split"), metric)} | {((name, b), c): _ms(_pick(ev, b, c, "split"), metric) for c in ["eval"] + ID}
    t = _table(cells, metric, title=f"{METRICS[metric]} · held-out positions of trained objects (in distribution)", vlines=(2,))
    t.index.names = ["method", "box"]
    return t

def table_ood(df, metric="iou") -> pd.DataFrame:
    """Table 2, one metric: rows = id (reference) + the ood categories the run answers, columns = box + all"""
    ev = df[df.set == "eval"]
    cats = [c for c in ["id"] + OOD if c in set(df.category)]
    return _table({(NAMES[c], b): _ms(_pick(ev, b, c, "category"), metric) for c in cats for b in BOXES + ("all",)}, metric,
                  title=f"{METRICS[metric]} · conditions never seen in training (out of distribution)", vlines=(3,))

def table_n(df, by="split") -> pd.DataFrame:
    """the samples behind Table 1 (by="split") or Table 2 (by="category"): n train / n test per box. n train of an ood row is 0"""
    cols = ID if by == "split" else [c for c in ["id"] + OOD if c in set(df.category)]
    t = pd.DataFrame({(k, b): [len(_pick(df[df.set == s], b, c, by)) for c in cols] for k, s in (("n train", "train"), ("n test", "eval")) for b in BOXES},
                     index=cols if by == "split" else [NAMES[c] for c in cols])
    return t.T if by == "split" else t

def auroc(neg, pos) -> float:
    """P(a pos scores above a neg), ties count half"""
    return (pos[:, None] > neg[None]).mean() + 0.5 * (pos[:, None] == neg[None]).mean()

def table_heads(df) -> pd.DataFrame:
    """Table 3, the aux heads: rows = heads, columns = categories (all boxes). Objects the head never saw have no class, so their
    object acc is —, and the novelty AUROC row scores 1 - confidence against the in-distribution samples instead (no std)"""
    ev = df[df.set == "eval"]
    cats = [c for c in ["id"] + OOD if c in set(ev.category)]
    rows = {"object acc": "object_acc", "object count acc": "count_acc", "box acc": "box_acc", "CoM head err (% box)": "com_head_err",
            "area rel err": "area_err", "orient err from mask (°)": "orient_err"}
    cells = {(name, NAMES[c]): _ms(ev[ev.category == c], m) for name, m in rows.items() for c in cats}
    idc = ev[ev.category == "id"].obj_conf.to_numpy()
    cells |= {("novelty AUROC", NAMES[c]): (np.nan if c == "id" else auroc(1 - idc, 1 - ev[ev.category == c].obj_conf.to_numpy()), np.nan) for c in cats}
    return _table(cells, "heads", title="Auxiliary heads · all boxes")

LIGHT = LinearSegmentedColormap.from_list("light", ["#ffffff", "#cde2fb", "#86b6ef"])  # light enough for dark text

RULE = f"2.5px solid {INK}"
CSS = [{"selector": "", "props": "border-collapse: collapse; font-family: system-ui, -apple-system, sans-serif; font-size: 13px; color: #0b0b0b"},
       {"selector": "th, td", "props": "padding: 5px 12px; text-align: center; white-space: nowrap; border: none"},
       {"selector": "th.col_heading.level0", "props": f"font-size: 14px; font-weight: 600; border-bottom: 1px solid {MUTED}; padding-bottom: 7px"},
       {"selector": "thead tr:last-child th", "props": f"border-bottom: {RULE}; font-weight: 600"},
       {"selector": "th.row_heading", "props": "text-align: left; font-weight: 600"},
       {"selector": "tbody tr:last-child td, tbody tr:last-child th", "props": f"border-bottom: {RULE}"},
       {"selector": "td.col0, th.col_heading.level1.col0, th.col_heading.level0.col0", "props": f"border-left: {RULE}"}]

def style(t, axis=0, color=True, latex=False):
    """a table from table_id / table_ood / table_heads as "mean ±std" (the std smaller). color: shade every cell by how good it is
    along axis (0: down each column, 1: along each row), the best in bold. Thick rules between the labels and the numbers, before
    each column in t.attrs["vlines"], and between the methods of Table 1"""
    m, s = t["mean"], t["std"]
    f = ".1f" if t.attrs.get("metric") in ("com_err",) else ".3f"
    def cell(a, b):
        if pd.isna(a): return "—"
        if pd.isna(b): return f"{a:{f}}"
        return f"{a:{f}} {{\\scriptsize $\\pm${b:{f}}}}" if latex else f"{a:{f}} <span style='font-size:72%; color:{MUTED}'>±{b:{f}}</span>"
    title = [t.attrs["title"]] if "title" in t.attrs else []  # a title row over the columns
    text = pd.DataFrame([[cell(a, b) for a, b in zip(ra, rb)] for ra, rb in zip(m.values, s.values)], index=m.index,
                        columns=pd.MultiIndex.from_product([title, m.columns]) if title else m.columns)
    css = pd.DataFrame("", index=m.index, columns=m.columns)
    if color:
        g = (-m if t.attrs.get("lower") else m).astype(float)
        g = g.loc[[i for i in g.index if "all" not in np.atleast_1d(i)], [c for c in g.columns if c != "all"]]  # "all" is an average, not a competitor
        lo, hi = g.min(axis), g.max(axis)
        z = g.sub(lo, axis=1 - axis).div((hi - lo).replace(0, np.nan), axis=1 - axis)  # 0 = worst, 1 = best along axis; NaN when all tie
        varies = (hi > lo).to_numpy()
        best = g.eq(hi, axis=1 - axis) & (varies[:, None] if axis == 1 else varies[None, :])
        css = (z.map(lambda v: "" if pd.isna(v) else f"background-color: {to_hex(LIGHT(v))};") + np.where(best, " font-weight: bold;", "")).reindex_like(m).fillna("")
    st = text.style.apply(lambda _: css.to_numpy(), axis=None)
    if latex: return st
    lines = [{"selector": f"td.col{j}, th.col_heading.level{len(title)}.col{j}", "props": f"border-left: {RULE}"} for j in t.attrs.get("vlines", ())]
    if isinstance(m.index, pd.MultiIndex):  # a rule where the method changes
        first = [i for i in range(1, len(m)) if m.index[i][0] != m.index[i - 1][0]]
        lines += [{"selector": f"td.row{i}, th.row_heading.level1.row{i}, th.row_heading.level0.row{i}", "props": f"border-top: {RULE}"} for i in first]
    return st.set_table_styles(CSS + lines)

def to_latex(t, path=None, caption=None, **kw) -> str:
    """the styled table as LaTeX (needs \\usepackage[table]{xcolor}, booktabs, multirow, array); written to path if given"""
    thick, n_idx = "!{\\vrule width 1.2pt}", t.index.nlevels
    cols = "".join(thick * (j in t.attrs.get("vlines", ())) + "c" for j in range(t["mean"].shape[1]))
    out = style(t, latex=True, **kw).to_latex(convert_css=True, hrules=True, multirow_align="t", multicol_align="c", caption=caption,
                                              column_format="l" * n_idx + thick + cols, clines="skip-last;data" if n_idx > 1 else None)
    out = out.replace("↑", "$\\uparrow$").replace("↓", "$\\downarrow$").replace("°", "$^\\circ$").replace("·", "$\\cdot$").replace("%", "\\%")
    if path: Path(path).write_text(out)
    return out

#*** 3b the tables as bar plots: x = the table's columns (objects / questions), one bar per box, the value above a ±1 std whisker,
# n train / n test under each bar

HEADS = {"object_acc": "object acc ↑", "count_acc": "object count acc ↑", "com_head_err": "CoM head err (% box) ↓", "area_err": "area rel err ↓",
         "orient_err": "orient err from mask (°) ↓", "novelty": "novelty AUROC ↑"}

def _bar_stats(df, field, cols, metric) -> pd.DataFrame:
    """one row per (column, box): mean over splits, std over samples, n train / n test"""
    ev, tr = df[df.set == "eval"], df[df.set == "train"]
    rows = []
    for x, c in enumerate(cols):
        for b in BOXES:
            e = _pick(tr, b, "eval", field) if c == "train" else _pick(ev, b, c, field)
            if metric == "novelty":  # against the same box's in-distribution samples
                ref = ev[(ev.box == b) & (ev.category == "id")].obj_conf.to_numpy()
                ms = (auroc(1 - ref, 1 - e.obj_conf.to_numpy()), np.nan) if len(e) and c != "id" else (np.nan, np.nan)
            else: ms = _ms(e, metric)
            n_tr = len(e) if c == "train" else len(_pick(tr, b, c, field))
            rows.append(dict(x=x, box=b, mean=ms[0], std=ms[1], n_train=n_tr, n_test=0 if c == "train" else len(e)))
    return pd.DataFrame(rows)

def _bars(ax, t, labels, metric, ylabel):
    w, bounded = 0.8 / len(BOXES), metric in ("iou", "miss", "object_acc", "count_acc", "novelty")
    for k, b in enumerate(BOXES):
        tb = t[(t.box == b) & t["mean"].notna()]
        x, m, sd = tb.x + (k - (len(BOXES) - 1) / 2) * w, tb["mean"], tb["std"].fillna(0)
        up = np.minimum(sd, 1 - m) if bounded else sd
        ax.bar(x, m, w * 0.92, color=COLORS[b], label=b)
        ax.errorbar(x, m, yerr=[np.minimum(sd, m), up], fmt="none", ecolor=MUTED, elinewidth=1, capsize=3, capthick=1)
        f = ".1f" if metric in ("com_err", "com_head_err", "orient_err") else ".2f"
        for xi, v, s in zip(x, m, tb["std"]):  # value ± std on the bar's top; above it when the bar is too short to hold it
            inside = v > 0.18 * (1 if bounded else t["mean"].max())
            ax.annotate(f"{v:{f}}" + ("" if pd.isna(s) else f"\n±{s:{f}}"), (xi, v), xytext=(0, -3 if inside else 3), textcoords="offset points",
                        ha="center", va="top" if inside else "bottom", fontsize=6.5, color=INK, zorder=5)
        for xi, a, n in zip(x, tb.n_train, tb.n_test):
            ax.annotate(f"tr {a}" + (f"\nte {n}" if n else ""), (xi, 0), xytext=(0, -4), textcoords="offset points", ha="center", va="top", fontsize=6, color=MUTED)
    ax.set_xticks(range(len(labels)), labels, fontsize=9)
    ax.tick_params(axis="x", pad=26, length=0)
    ax.set_ylabel(ylabel)
    ax.grid(axis="x", visible=False); ax.set_axisbelow(True)
    ax.margins(y=0.1)
    ax.legend(frameon=False, bbox_to_anchor=(1.01, 1), loc="upper left")

def plot_table_id(df, metric="iou", name="VibroFormer"):
    """Table 1 as bars: train, eval (the 6 id splits' mean) | eval per id split, one bar per box"""
    cols = ["train", "eval"] + ID
    fig, ax = plt.subplots(figsize=(14, 4.8))
    _bars(ax, _bar_stats(df, "split", cols, metric), cols, metric, METRICS[metric])
    ax.axvline(1.5, color=INK, lw=1.5)  # train / eval | per object, as in the table
    ax.set_title(f"{name} · in distribution: {METRICS[metric]}", loc="left", fontweight="bold")
    return fig

def plot_table_ood(df, metric="iou", name="VibroFormer"):
    """Table 2 as bars: every ood question the run answers, one bar per box"""
    cats = [c for c in OOD if c in set(df.category)]
    fig, ax = plt.subplots(figsize=(13, 4.6))
    _bars(ax, _bar_stats(df, "category", cats, metric), [NAMES[c].replace(" ", "\n", 1) for c in cats], metric, METRICS[metric])
    ax.set_title(f"{name} · out of distribution: {METRICS[metric]}", loc="left", fontweight="bold")
    return fig

def plot_table_heads(df, heads=tuple(HEADS), name="VibroFormer"):
    """Table 3 as bars, one panel per head: x = the questions, one bar per box (novelty AUROC: against the same box's id samples)"""
    cats = [c for c in ["id"] + OOD if c in set(df.category)]
    fig, axs = plt.subplots(len(heads), 1, figsize=(13, 4.2 * len(heads)), squeeze=False)
    for ax, h in zip(axs[:, 0], heads):
        _bars(ax, _bar_stats(df, "category", cats, h), [NAMES[c].replace(" ", "\n", 1) for c in cats], h, HEADS[h])
        ax.set_title(f"{name} · {HEADS[h]}", loc="left", fontweight="bold")
    fig.tight_layout()
    return fig

#*** 4 plots

def _ci(v, n=1000):
    """95% bootstrap CI of the mean"""
    v = np.asarray(v, float)
    m = np.random.default_rng(0).choice(v, (n, len(v))).mean(1)
    return np.percentile(m, 2.5), np.percentile(m, 97.5)

def plot_weight(df, metrics=("iou", "miss")):
    """the weight trend: metric vs cylinder mass (log axis), one line per box that trained on cylinders, mean ± std over samples"""
    d = df[(df.set == "eval") & (df.category == "weight")]
    fig, axs = plt.subplots(1, len(metrics), figsize=(6 * len(metrics), 4), squeeze=False)
    for ax, m in zip(axs[0], metrics):
        for b in BOXES:
            g = d[d.box == b].groupby("mass")[m].agg(["mean", "std", "size"])
            if g.empty: continue
            ax.errorbar(g.index, g["mean"], g["std"], fmt="-o", color=COLORS[b], lw=2, ms=5, capsize=3, elinewidth=1, label=b)
            for x, v, n in zip(g.index, g["mean"], g["size"]): ax.annotate(f"n={n}", (x, v), xytext=(5, -10), textcoords="offset points", fontsize=7, color=MUTED)
        ax.set(xscale="log", xlabel="cylinder mass (g)", ylabel=METRICS[m], ylim=(0, 1))
        ax.set_xticks([100, 200, 500, 1000], ["100", "200", "500", "1000"])
        ax.legend(frameon=False)
    axs[0, 0].set_title("cylinders of the same shape, different mass", loc="left", fontweight="bold")
    fig.tight_layout()
    return fig

def plot_floor_error(df, metric="iou", categories=("id",), bins=8, min_pos=2):
    """mean metric over the floor per box, binned by the true centroid; the dashed square is the unseen-position block. A bin with
    fewer than min_pos positions is left blank (plastic has almost none near the laser wall)"""
    d = df[(df.set == "eval") & df.category.isin(categories)].dropna(subset=["com_r", "com_c"])
    edges = np.linspace(0, 1, bins + 1)
    maps = {}
    for b in BOXES:
        g = d[d.box == b]
        r, c = (np.clip(np.digitize(g[k], edges) - 1, 0, bins - 1) for k in ("com_r", "com_c"))
        s, n = np.zeros((bins, bins)), np.zeros((bins, bins))
        np.add.at(s, (r, c), g[metric].to_numpy()); np.add.at(n, (r, c), 1)
        npos = pd.Series(list(zip(g.experiment, g.position_id))).groupby([r, c]).nunique()
        ok = np.zeros((bins, bins), bool); ok[tuple(np.array(list(npos[npos >= min_pos].index)).T)] = True
        maps[b] = np.where(ok, s / np.maximum(n, 1), np.nan)
    lo, hi = np.nanpercentile(np.concatenate([m.ravel() for m in maps.values()]), [5, 95])  # one colour scale for all boxes
    fig, axs = plt.subplots(1, 3, figsize=(13, 4.4), constrained_layout=True)
    for ax, b in zip(axs, BOXES):
        m = maps[b]
        im = ax.imshow(m, cmap=BLUES if metric == "iou" else BLUES.reversed(), vmin=lo, vmax=hi, extent=(0, 1, 1, 0))
        for (i, j), v in np.ndenumerate(m):
            if not np.isnan(v): ax.text((j + .5) / bins, (i + .5) / bins, f"{v:.2f}", ha="center", va="center", fontsize=7,
                                        color="white" if ((v - lo) / (hi - lo) > .55) == (metric == "iou") else INK)
        r0, r1, c0, c1 = D.BLOCK
        ax.add_patch(Rectangle((c0, r0), c1 - c0, r1 - r0, fill=False, ls="--", lw=1.5, ec=INK))
        ax.set(title=f"{b}  (n={(d.box == b).sum()})", xticks=[], yticks=[], xlabel="lasers ↓ (bottom wall)"); ax.grid(False)
    fig.colorbar(im, ax=axs, shrink=.8, label=METRICS[metric])
    fig.suptitle(f"{METRICS[metric]} over the floor: {', '.join(NAMES[c] for c in categories)}", x=0.02, ha="left")
    return fig

def plot_iou_vs_distance(df, metric="iou", categories=("id", "unseen-position"), step=0.5):
    """metric vs distance (% box) to the nearest training position, per box: mean per bin, 95% bootstrap CI, bins of < 5 dropped"""
    d = df[(df.set == "eval") & df.category.isin(categories)].assign(dist=lambda x: x.train_dist * 100).dropna(subset=["dist"])
    d["bin"] = (d.dist // step + .5) * step
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    for b in BOXES:
        g = d[d.box == b].groupby("bin")[metric].agg(list)
        g = g[g.map(len) >= 5]
        ci = np.array([_ci(v) for v in g])
        ax.fill_between(g.index, ci[:, 0], ci[:, 1], color=COLORS[b], alpha=.15, lw=0)
        ax.plot(g.index, g.map(np.mean), "-o", color=COLORS[b], lw=2, ms=4, label=b)
    ax.set(xlabel="distance to nearest training position (% of box)", ylabel=METRICS[metric], ylim=(0, 1) if metric != "com_err" else None)
    ax.legend(frameon=False, loc="lower left")
    ax.set_title(f"{METRICS[metric]} vs distance from training data", loc="left")
    return fig

def plot_speakers(df, speakers=(1, 3, 7), metric="iou"):
    """id performance per speaker per box, on positions every listed speaker recorded (paired); whiskers = 95% bootstrap CI over positions"""
    d = df[(df.set == "eval") & (df.category == "id") & df.speaker.isin(speakers)]
    d = d[d.groupby(["experiment", "position_id"]).speaker.transform("nunique") == len(speakers)]
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    w = 0.8 / len(speakers)
    for k, s in enumerate(speakers):
        for i, b in enumerate(BOXES):
            v = d[(d.box == b) & (d.speaker == s)].groupby(["experiment", "position_id"])[metric].mean()
            lo, hi = _ci(v)
            x = i + (k - (len(speakers) - 1) / 2) * w
            ax.bar(x, v.mean(), w * 0.92, color=SPK_COLORS.get(s, MUTED), label=f"speaker {s}" if i == 0 else None)
            ax.errorbar(x, v.mean(), [[v.mean() - lo], [hi - v.mean()]], color=INK, lw=1, capsize=3)
    ax.set_xticks(range(len(BOXES)), BOXES)
    ax.set(ylabel=METRICS[metric], ylim=(0, 1) if metric != "com_err" else None)
    ax.grid(axis="x", visible=False)
    ax.legend(frameon=False, ncols=len(speakers), loc="upper center", bbox_to_anchor=(.5, 1.12))
    return fig

def plot_confidence(df, categories=("id", "broken", "unseen-object")):
    """object-head confidence: in distribution vs objects the model should not recognise; AUROC vs id in the legend"""
    ev = df[df.set == "eval"]
    idc = ev[ev.category == "id"].obj_conf.to_numpy()
    fig, ax = plt.subplots(figsize=(7.5, 4))
    for c, col in zip([c for c in categories if c in set(ev.category)], ("#2a78d6", "#eb6834", "#1baf7a")):
        v = ev[ev.category == c].obj_conf.to_numpy()
        ax.hist(v, np.linspace(0, 1, 26), density=True, histtype="step", lw=2, color=col,
                label=NAMES[c] + ("" if c == "id" else f"  (AUROC {auroc(1 - idc, 1 - v):.2f})"))
    ax.set(xlabel="object-head confidence (max softmax)", ylabel="density (log)", yscale="log")
    ax.legend(frameon=False, loc="upper left")
    return fig

def samples_figure(df, categories=None, n=4, speaker=1, width=320):
    """predicted masks on the overhead photos, one row per category, n samples a row at evenly spaced IoU quantiles (15-85%, not the
    best), boxes taking turns. White = true outline, orange = prediction > thr. A lid photo shows the lid; its outline is the spot without it"""
    ev = df[(df.set == "eval") & (df.speaker == speaker)]
    cats = categories or [c for c in ["id"] + OOD if c in set(ev.category)]
    thr = df.attrs.get("thr", THR)
    fig, axs = plt.subplots(len(cats), n, figsize=(3.1 * n, 2.95 * len(cats)), squeeze=False, gridspec_kw=dict(wspace=.04, hspace=.3))
    for row, c in zip(axs, cats):
        g = ev[ev.category == c]
        boxes = [b for b in BOXES if b in set(g.box)]
        for k, (ax, q) in enumerate(zip(row, np.linspace(.15, .85, n))):
            h = g[g.box == boxes[k % len(boxes)]].sort_values("iou")
            r = h.iloc[min(int(q * len(h)), len(h) - 1)]
            sid, pos, spk = int(r.sample_id), int(r.position_id), int(r.speaker)
            im = Image.open(S.sample_dir(r.experiment, sid, pos, spk) / "image/02_cropped_overhead.png").convert("RGB")
            W, H = width, round(width * im.height / im.width)
            p = np.asarray(Image.fromarray(torch.load(r.file, weights_only=False, map_location="cpu")["mask_pred"][int(r.idx)].float().numpy()).resize((W, H), Image.BILINEAR))
            y = np.asarray(Image.fromarray(S.truth(r.experiment, sid, pos, spk, 256, 256)).resize((W, H), Image.BILINEAR))
            ax.imshow(im.resize((W, H)))
            ax.imshow(np.where((p > thr)[..., None], [0.92, 0.41, 0.20, 0.38], 0))
            ax.contour(p, [thr], colors=["#eb6834"], linewidths=1.8)
            ax.contour(y, [0.5], colors=["white"], linewidths=1.6)
            ax.set_title(f"{r.box} · {r.split}\nIoU {r.iou:.2f}", fontsize=8, pad=3)
            ax.set(xticks=[], yticks=[]); ax.grid(False)
            for s in ax.spines.values(): s.set_visible(False)
        row[0].set_ylabel(NAMES[c], fontsize=10)
    fig.legend(handles=[Patch(fc="none", ec="white", lw=1.6, label="true outline"), Patch(fc="#eb6834", alpha=.6, ec="#eb6834", label=f"prediction > {thr}")],
               loc="lower center", ncols=2, facecolor=MUTED, labelcolor="white", bbox_to_anchor=(.5, .9))
    fig.subplots_adjust(top=.885)
    return fig

#*** 5 the data (notebook section 0), before any model. Every function takes the run: its dataset's splits and its train / eval
# positions. objects = split names (None: OBJECTS, the four single-object id splits), speakers (None: every speaker the split has).
# Spectra are only ever compared within one (object, speaker) group; curves are the mean over the groups, ± the pooled
# within-group std (sqrt of the mean variance), so the difference between speakers or objects never widens a band

OBJECTS = ("candle", "cube", "soap-dispenser", "vase")
FLOOR = {"gastro": (.172, .126, .897, .832), "plastic": (.183, .149, .847, .802), "cardboard": (.134, .079, .948, .860)}  # (left, top, right, bottom) of 02_cropped_overhead, read off the empty-box photos
FB = (np.arange(131, 1301) / 1.3).reshape(117, 10).mean(1)  # band centres (Hz): 04_ffts keeps 100-1000 Hz, 1170 bins of 1/1.3 Hz, 10 bins a band (nb99)
CACHE = ROOT / "runs/_paper_cache/data"
FPS, QUIET = 2500, slice(25, 375)  # QUIET: camera frames before the chirp arrives (nb99)
OCT = ((100, 200), (200, 400), (400, 800))

def captures(run) -> pd.DataFrame:
    """one row per capture in the run's dataset: box, split, layout, position, speaker, timestamp t, com (row, col px of
    02_cropped_overhead, mean over objects), crop H, W, dir, and set = train / eval (the run's positions). Cached"""
    f = ROOT / "runs" / run / "scores" / "paper_captures.pkl"
    if f.exists(): return pd.read_pickle(f)
    a = args(run)
    splits = D.DATASETS[a["dataset"]](a["test_size"], None if "all" in a["speakers"] else tuple(a["speakers"]))
    tr = pd.read_csv(ROOT / "runs" / run / "positions/train.txt", sep=" ", names=["experiment", "position_id"])
    train = set(zip(tr.experiment, tr.position_id))
    ev = {(f"{t.parent.name}/{t.stem}", e, int(p)) for t in (ROOT / "runs" / run / "positions/eval").glob("*/*.txt") for e, p in (l.split() for l in t.read_text().splitlines() if l.strip())}
    rows = []
    for exp in sorted({s["experiment"] for s in splits.values()}):
        mine = {k: s for k, s in splits.items() if s["experiment"] == exp}
        for d in sorted((D.EXPERIMENTS / exp / "samples").iterdir()):
            if not (d / "metadata.jsonl").exists(): continue
            m = D.load_meta(d)
            pos, spk = m["position_id"], m["speaker"]
            k = next((k for k, s in mine.items() if m["layout"] in s["layouts"] and s["keep"](pos) and (s["speakers"] is None or spk in s["speakers"])), None)
            if k is None: continue
            com = m.get("avg_com") or [np.nan, np.nan]
            rows.append(dict(box=k.split("/")[0], split=k.split("/")[1], layout=m["layout"], experiment=exp, position_id=pos, speaker=spk,
                             t=pd.Timestamp(m["timestamp"]), com_r=com[0], com_c=com[1], H=m["crop_overhead_shape"][0], W=m["crop_overhead_shape"][1],
                             dir=d, set="train" if (exp, pos) in train else "eval" if (k, exp, pos) in ev else None))
    out = pd.DataFrame(rows)
    f.parent.mkdir(parents=True, exist_ok=True); out.to_pickle(f)
    return out

def _names(v, default): return list(default if v is None else [v] if isinstance(v, (str, int)) else v)

def _groups(cap, box, objects=None, speakers=None) -> dict:
    """{(object, speaker): its captures} of one box"""
    c = cap[(cap.box == box) & cap.split.isin(_names(objects, OBJECTS))]
    if speakers is not None: c = c[c.speaker.isin(_names(speakers, ()))]
    return {k: g.reset_index(drop=True) for k, g in c.groupby(["split", "speaker"])}

def _cached(tag, dirs, fn):
    """fn() pickled under CACHE, keyed on the capture dirs it reads"""
    f = CACHE / f"{tag}_{hashlib.sha1('|'.join(map(str, dirs)).encode()).hexdigest()[:16]}.pkl"
    if f.exists(): return pd.read_pickle(f)
    out = fn()
    f.parent.mkdir(parents=True, exist_ok=True); pd.to_pickle(out, f)
    return out

def _pool(means, variances):
    """mean over groups, pooled within-group std"""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(np.stack(means), 0), np.sqrt(np.nanmean(np.stack(variances), 0))

_SPEC = {}
def _spectrum(d):
    A = np.abs(D.load_fft(d).numpy())  # (100 lasers, 1170, 2 axes)
    return (20 * np.log10(A + 1e-12)).reshape(100, 117, 10, 2).mean(2).astype(np.float16), (10 * np.log10((A ** 2).mean((0, 2)).reshape(117, 10).mean(1))).astype(np.float32)

def spectra(rows) -> tuple[np.ndarray, np.ndarray]:
    """per capture (rows' order): x = 20 log10 |X| averaged over 10 FFT bins (117 bands of 7.7 Hz), (N, 100, 117, 2), and the level
    L = 10 log10 mean_{l,c} |X|^2 per band, (N, 117) (nb99 1.2). Cached on disk per experiment"""
    for exp, g in rows.groupby("experiment"):
        f = CACHE / f"spectra_{exp}.npz"
        if exp not in _SPEC:
            z = np.load(f) if f.exists() else None
            _SPEC[exp] = {} if z is None else {n: (x, l) for n, x, l in zip(z["names"], z["x"], z["L"])}
        todo = [d for d in dict.fromkeys(g.dir) if d.name not in _SPEC[exp]]
        if todo:
            with ThreadPoolExecutor(16) as ex: _SPEC[exp].update(zip((d.name for d in todo), ex.map(_spectrum, todo)))
            n, v = list(_SPEC[exp]), list(_SPEC[exp].values())
            f.parent.mkdir(parents=True, exist_ok=True); np.savez(f, names=n, x=np.stack([a for a, _ in v]), L=np.stack([b for _, b in v]))
    v = [_SPEC[e][d.name] for e, d in zip(rows.experiment, rows.dir)]
    return np.stack([a for a, _ in v]), np.stack([b for _, b in v])

def d_spec(A, B) -> np.ndarray:
    """D_spec (dB): rms difference of x between every capture of A and every capture of B, (len(A), len(B)) (nb99 1.2)"""
    A, B = A.reshape(len(A), -1).astype(np.float32), B.reshape(len(B), -1).astype(np.float32)
    d2 = (A ** 2).sum(1)[:, None] + (B ** 2).sum(1)[None] - 2 * A @ B.T
    return np.sqrt(np.maximum(d2, 0) / A.shape[1])

def noise(cap, box, speaker) -> np.ndarray:
    """repeat noise: D_spec of every pair of back-to-back empty-box captures (same layout and speaker)"""
    e = cap[(cap.box == box) & (cap.split == "empty-box") & (cap.speaker == speaker)].reset_index(drop=True)
    D_ = d_spec(*[spectra(e)[0]] * 2)
    return np.array([D_[i, j] for g in e.groupby("layout").indices.values() for n, i in enumerate(g) for j in g[n + 1:]])

def _floor(ax, cap, box, crop=True):
    """the first empty-box photo of the box (02_cropped_overhead), cropped to its floor unless crop=False"""
    d = cap[(cap.box == box) & (cap.split == "empty-box")].sort_values("t").dir.iloc[0]
    im = Image.open(d / "image/02_cropped_overhead.png").convert("RGB")
    ax.imshow(im); ax.set(xticks=[], yticks=[]); ax.grid(False)
    if crop:
        l, t, r, b = FLOOR[box]
        ax.set(xlim=(l * im.width, r * im.width), ylim=(b * im.height, t * im.height))

def plot_coverage(run, objects="cube", box="cardboard", speakers=None, step=4):
    """where the objects were put: every position's true outline (03_smask > 0.5) and CoM on the whole overhead photo of the empty box, one panel per object.
    Train positions in the box's colour, eval in black. A position counts once (its mask is the same for every speaker)"""
    cap = captures(run)
    objs = _names(objects, OBJECTS)
    fig, axs = plt.subplots(1, len(objs), figsize=(6.5 * len(objs), 5), squeeze=False, constrained_layout=True)
    for ax, o in zip(axs[0], objs):
        g = cap[(cap.box == box) & (cap.split == o)]
        if speakers is not None: g = g[g.speaker.isin(_names(speakers, ()))]
        g = g.drop_duplicates(["experiment", "position_id"])
        _floor(ax, cap, box, crop=False)
        n = {}
        for s, c, lw in (("train", COLORS[box], 1), ("eval", INK, 1.4)):
            h = g[g.set == s]
            n[s] = len(h)
            for d in h.dir:
                m = np.asarray(Image.open(d / "image/03_smask.png").convert("L").reduce(step), np.float32) / 255
                ax.contour((np.arange(m.shape[1]) + .5) * step, (np.arange(m.shape[0]) + .5) * step, m, [.5], colors=[c], linewidths=lw, alpha=.55)
            ax.scatter(h.com_c, h.com_r, s=14, c=c, edgecolors="white", linewidths=.5, zorder=3)
        ax.set_title(f"{box} · {o}   ({n['train']} train / {n['eval']} eval positions)", loc="left")
    fig.legend(handles=[Line2D([], [], c=COLORS[box], marker="o", label="train: outline + CoM"), Line2D([], [], c=INK, marker="o", label="eval: outline + CoM")],
               loc="lower center", ncols=2, frameon=False, bbox_to_anchor=(.5, -.06))
    return fig

def _shades(c, n):
    """n shades of colour c, light to dark"""
    rgb = np.array(to_rgb(c))
    return [to_hex(rgb + a * (1 - rgb) if a > 0 else rgb * (1 + a)) for a in np.linspace(.5, -.5, n)]

def plot_empty(run, speakers=None):
    """the empty box over the session: level L(f) of each empty-box layout (in time order, light = first), one panel per box.
    Per speaker the layout's mean over its captures, then the mean over speakers; band = ± the within-(layout, speaker) std over captures, pooled"""
    cap = captures(run)
    fig, axs = plt.subplots(len(BOXES), 1, figsize=(20, 4 * len(BOXES)), sharex=True, constrained_layout=True)
    for ax, b in zip(axs, BOXES):
        e = cap[(cap.box == b) & (cap.split == "empty-box")]
        if speakers is not None: e = e[e.speaker.isin(_names(speakers, ()))]
        e = e.reset_index(drop=True)
        L = spectra(e)[1]
        order = e.groupby("layout").t.min().sort_values()
        for (l, t0), c in zip(order.items(), _shades(COLORS[b], len(order))):
            gs = [L[i] for _, i in e[e.layout == l].groupby("speaker").groups.items()]
            m, s = _pool([g.mean(0) for g in gs], [g.var(0) for g in gs])
            ax.fill_between(FB, m - s, m + s, color=c, alpha=.15, lw=0)
            ax.plot(FB, m, c=c, lw=1.4, label=f"{l} · {t0:%H:%M}")
        ax.set(ylabel=f"{b}\nlevel L (dB)")
        ax.legend(ncols=len(order), fontsize=8, frameon=False, loc="lower left")
    axs[0].set_title(f"empty box over the session, {'all speakers' if speakers is None else f'speaker {speakers}'} (mean ± std over captures)", loc="left")
    axs[-1].set_xlabel("Frequency (Hz)")
    return fig

def table_empty(run, speakers=None) -> pd.DataFrame:
    """does the empty box stay the same? Per box: repeat noise = D_spec of back-to-back captures (same layout), drift = D_spec of
    captures of different layouts (recorded at different times), both within one speaker; mean ± std over pairs"""
    cap, rows = captures(run), {}
    for b in BOXES:
        e = cap[(cap.box == b) & (cap.split == "empty-box")]
        if speakers is not None: e = e[e.speaker.isin(_names(speakers, ()))]
        same, diff = [], []
        for _, g in e.groupby("speaker"):
            g = g.reset_index(drop=True)
            D_, lay = d_spec(*[spectra(g)[0]] * 2), g.layout.to_numpy()
            iu = np.triu_indices(len(g), 1)
            same += list(D_[iu][lay[iu[0]] == lay[iu[1]]]); diff += list(D_[iu][lay[iu[0]] != lay[iu[1]]])
        rows[b] = {"layouts": e.layout.nunique(), "captures": len(e), "session span (h)": f"{(e.t.max() - e.t.min()).total_seconds() / 3600:.1f}",
                   "repeat noise (dB)": f"{np.mean(same):.2f} ± {np.std(same):.2f}", "drift between layouts (dB)": f"{np.mean(diff):.2f} ± {np.std(diff):.2f}",
                   "drift / noise": f"{np.mean(diff) / np.mean(same):.2f}"}
    return pd.DataFrame(rows)

def plot_spec_map(run, box="cardboard", objects=None, speakers=None, position=None, bins=8):
    """D_spec from a reference position to every position of the same object and speaker, on the floor (colour: 9 quantile steps of the
    values, light = close, dark = far; the colourbar and the numbers are real dB), as two figures: the box's whole empty-box overhead
    photo (one dot per position, star = reference), and binned over the floor. position: a position id, or None = the middle of the floor;
    each object's reference is its position closest to that spot. Per position the mean over speakers; a heat-map bin is the mean
    over objects of each object's bin mean. The repeat noise (empty box) is the scale to read it against"""
    cap = captures(run)
    l, t, r, b = FLOOR[box]
    frac = lambda g: np.stack([(g.com_r / g.H - t) / (b - t), (g.com_c / g.W - l) / (r - l)], 1)   # (row, col) as floor fractions
    target = frac(cap[(cap.box == box) & (cap.position_id == position)].iloc[:1])[0] if position is not None else np.array([.5, .5])
    groups, res = _groups(cap, box, objects, speakers), []
    for o in dict.fromkeys(o for o, _ in groups):
        pos = cap[(cap.box == box) & (cap.split == o)].drop_duplicates("position_id")
        ref = pos.position_id.iloc[np.argmin(np.hypot(*(frac(pos) - target).T))]
        for (oo, s), g in groups.items():
            if oo != o or not (g.position_id == ref).any(): continue
            x = spectra(g)[0]
            res.append(g.assign(d=d_spec(x[(g.position_id == ref).to_numpy()], x).mean(0), ref=ref))
    res = pd.concat(res).groupby(["split", "position_id"]).agg(d=("d", "mean"), ref=("ref", "first"), com_r=("com_r", "first"), com_c=("com_c", "first"), H=("H", "first"), W=("W", "first")).reset_index()
    res = res[res.position_id != res.ref]
    fr = frac(res)
    edges = np.linspace(0, 1, bins + 1)
    ij = [np.clip(np.digitize(fr[:, k], edges) - 1, 0, bins - 1) for k in (0, 1)]
    per = res.assign(i=ij[0], j=ij[1]).groupby(["split", "i", "j"]).d.mean().groupby(["i", "j"]).mean()
    grid = np.full((bins, bins), np.nan); grid[per.index.get_level_values(0), per.index.get_level_values(1)] = per.to_numpy()
    nz = np.concatenate([noise(cap, box, s) for s in sorted({s for _, s in groups})])
    levels = np.unique(np.nanquantile(np.concatenate([res.d.to_numpy(), grid[np.isfinite(grid)]]), np.linspace(0, 1, 10)))
    norm, cmap = BoundaryNorm(levels, 256), SEQ                          # quantile steps: each colour holds the same share of values
    what = f"{box} · {', '.join(_names(objects, OBJECTS))} · {'all speakers' if speakers is None else f'speaker {speakers}'}"
    noise_txt = f"repeat noise {nz.mean():.2f} ± {nz.std():.2f} dB"
    refs = cap[(cap.box == box) & cap.position_id.isin(res.ref.unique())].drop_duplicates("position_id")
    def bar(fig, ax, m):
        cb = fig.colorbar(m, ax=ax, shrink=.85, spacing="uniform", label="D_spec to the reference (dB), quantile steps")
        cb.set_ticks(levels); cb.set_ticklabels([f"{v:.2f}" for v in levels])
    f1, ax = plt.subplots(figsize=(10, 7), constrained_layout=True)
    _floor(ax, cap, box, crop=False)
    sc = ax.scatter(res.com_c, res.com_r, c=res.d, cmap=cmap, norm=norm, s=40, edgecolors=INK, linewidths=.3)   # light = close, dark = far
    ax.scatter(refs.com_c, refs.com_r, marker="*", s=380, c="#eb6834", edgecolors="white", linewidths=1, label="reference")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_title(f"{what}: spectral distance from the reference, per position ({noise_txt})", loc="left", fontsize=10)
    bar(f1, ax, sc)
    f2, ax = plt.subplots(figsize=(9, 7), constrained_layout=True)
    im = ax.imshow(grid, cmap=cmap, norm=norm, extent=(0, 1, 1, 0))
    for (i, j), v in np.ndenumerate(grid):
        if np.isfinite(v): ax.text((j + .5) / bins, (i + .5) / bins, f"{v:.2f}", ha="center", va="center", fontsize=8, color="white" if norm(v) > 140 else INK)
    rf = frac(refs)
    ax.scatter(rf[:, 1], rf[:, 0], marker="*", s=380, c="#eb6834", edgecolors="white", linewidths=1)
    ax.set(xticks=[], yticks=[], xlabel="lasers ↓ (bottom wall)"); ax.grid(False)
    ax.set_title(f"{what}: binned over the floor ({noise_txt})", loc="left", fontsize=10)
    bar(f2, ax, im)
    return f1, f2

def plot_spec_vs_pos(run, objects=None, speakers=None, step=.04, boxes=BOXES):
    """S(Δ) (nb99 4): D_spec of two captures vs how far apart the object was, D_pos = |com_a - com_b| / sqrt(H W), over every pair of
    one (object, speaker); bins with < 5 pairs dropped. Mean over groups ± pooled std over pairs. Dashed: the repeat noise.
    move/noise (legend) = (mean D_spec at D_pos > 0.25 - mean at 0 < D_pos < 0.05) / noise, mean over groups"""
    cap = captures(run)
    edges = np.arange(0, .85, step)
    fig, ax = plt.subplots(figsize=(18, 6))
    for b in boxes:
        M, V, mn, nz = [], [], [], []
        for (o, s), g in _groups(cap, b, objects, speakers).items():
            iu = np.triu_indices(len(g), 1)
            D_ = d_spec(*[spectra(g)[0]] * 2)[iu]
            P = (np.hypot(g.com_r.to_numpy()[:, None] - g.com_r.to_numpy(), g.com_c.to_numpy()[:, None] - g.com_c.to_numpy()) / np.sqrt(g.H * g.W).to_numpy())[iu]
            st = pd.Series(D_).groupby(pd.cut(P, edges), observed=False).agg(["mean", "var", "size"])
            M.append(st["mean"].where(st["size"] >= 5).to_numpy()); V.append(st["var"].where(st["size"] >= 5).to_numpy())
            n = noise(cap, b, s).mean()
            mn.append((D_[P > .25].mean() - D_[(P > 0) & (P < .05)].mean()) / n); nz.append(n)
        m, sd = _pool(M, V)
        x = edges[:-1] + step / 2
        ax.fill_between(x, m - sd, m + sd, color=COLORS[b], alpha=.15, lw=0)
        ax.plot(x, m, "o-", c=COLORS[b], ms=5, label=f"{b}  (move/noise {np.mean(mn):.2f} ± {np.std(mn):.2f})")
        ax.axhline(np.mean(nz), c=COLORS[b], ls="--", lw=1)
    ax.plot([], [], c=MUTED, ls="--", lw=1, label="repeat noise (empty box)")
    ax.set(xlabel="Δ: distance the object moved / box size", ylabel="D_spec (dB)",
           title=f"spectral change vs distance moved: {', '.join(_names(objects, OBJECTS))}, {'all speakers' if speakers is None else f'speaker {speakers}'} (mean ± std over same-group pairs)")
    ax.legend(frameon=False, ncols=4, loc="upper center", bbox_to_anchor=(.5, -.12))
    return fig

@lru_cache(maxsize=None)
def _sweep(nt):
    """the played chirp's spectrum at the camera rate (log sweep 100 -> 1000 Hz, 1 s, from 0.1 s), and the in-band bins"""
    C = np.fft.rfft(np.concatenate([np.zeros(250), chirp(np.arange(FPS) / FPS, f0=100, t1=1.0, f1=1000, method="logarithmic"), np.zeros(nt - 2750)]))
    fr = np.fft.rfftfreq(nt, 1 / FPS)
    return C, (fr >= 100) & (fr <= 1000)

def envelope(d, t_max=.75) -> dict:
    """nb99 5: impulse response h = irfft(X / chirp) of the raw shifts (band-passed, no window), envelope E(t) = 10 log10 of
    smooth_10ms mean_{l,c} h^2, dB re its peak, for 100-1000 Hz and per octave; noise floor = the same for the silent start (energy
    per sample); time to noise = first t with E <= floor + 1 dB (NaN if not within t_max)"""
    x = sosfiltfilt(butter(5, [100, 1000], fs=FPS, btype="band", output="sos"), np.load(d / "vibration/02_raw_shifts.npy"), axis=1)
    nt = x.shape[1]
    C, inband = _sweep(nt)
    q = np.zeros_like(x); q[:, QUIET] = x[:, QUIET]
    deconv = lambda y: np.fft.irfft(np.where(inband[None, :, None], np.fft.rfft(y, axis=1) / C[None, :, None], 0), n=nt, axis=1)
    h, hq = deconv(x), deconv(q)
    h = np.roll(h, -(h ** 2).mean((0, 2)).argmax(), axis=1)
    out = {}
    for lo, hi in ((100, 1000),) + OCT:
        sos = butter(4, [lo, hi], fs=FPS, btype="band", output="sos")
        e = np.convolve((sosfiltfilt(sos, h, axis=1) ** 2).mean((0, 2)), np.ones(25) / 25, "same")
        fl = 10 * np.log10((sosfiltfilt(sos, hq, axis=1) ** 2).mean((0, 2)).sum() / (QUIET.stop - QUIET.start) / e.max())
        e = 10 * np.log10(e / e.max())[:int(t_max * FPS)]
        hit = np.flatnonzero(e <= fl + 1)
        out[lo, hi] = dict(env=e.astype(np.float32), floor=fl, t_noise=hit[0] / FPS if len(hit) else np.nan)
    return out

def _envelopes(g, n=None):
    """envelope() of every capture of g (n: a random n of them instead). Cached"""
    dirs = list(g.dir) if n is None or len(g) <= n else list(g.dir.sample(n, random_state=0))
    def run():
        with ThreadPoolExecutor(8) as ex: return list(ex.map(envelope, dirs))
    return _cached(f"env{n}", dirs, run)

def _envelope_stats(cap, b, objects, speakers, n, bands_):
    """per box: envelope mean, pooled std, mean floor, and time to noise per band (mean, pooled std, share of captures that reach it)"""
    E = [_envelopes(g, n) for g in _groups(cap, b, objects, speakers).values()]
    m, sd = _pool([np.stack([e[100, 1000]["env"] for e in es]).mean(0) for es in E], [np.stack([e[100, 1000]["env"] for e in es]).var(0) for es in E])
    tn = [np.array([[e[bd]["t_noise"] for e in es] for bd in bands_]) for es in E]   # per group (bands, captures)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        tm, ts = _pool([np.nanmean(t, 1) for t in tn], [np.nanvar(t, 1) for t in tn])
    return m, sd, np.mean([e[100, 1000]["floor"] for es in E for e in es]), tm, ts, np.mean([np.isfinite(t).mean(1) for t in tn], 0)

TN_BANDS = OCT + ((100, 1000),)   # octaves by increasing frequency, then the whole band

def plot_envelope(run, objects="empty-box", speakers=None, n=None, boxes=BOXES):
    """how fast the box goes quiet after the speaker's impulse (nb99 5.2), from every capture of the chosen objects (default the empty
    box; any split names, None = OBJECTS) and speakers (n: a random n per (object, speaker) instead): the 100-1000 Hz envelope, mean over groups ± pooled std over captures, dashed = noise floor, dotted =
    mean time to noise (only if most captures reach it). Cached per group"""
    cap = captures(run)
    fig, ax = plt.subplots(figsize=(18, 5.5), constrained_layout=True)
    for b in boxes:
        m, sd, fl, tm, _, reach = _envelope_stats(cap, b, objects, speakers, n, TN_BANDS)
        tt = np.arange(len(m)) / FPS
        ax.fill_between(tt, m - sd, m + sd, color=COLORS[b], alpha=.15, lw=0)
        ax.plot(tt, m, c=COLORS[b], lw=1.4, label=b)
        ax.axhline(fl, c=COLORS[b], ls="--", lw=1)
        if np.isfinite(tm[-1]) and reach[-1] > .5: ax.axvline(tm[-1], c=COLORS[b], ls=":", lw=2)
    ax.plot([], [], c=MUTED, ls="--", lw=1, label="noise floor"); ax.plot([], [], c=MUTED, ls=":", lw=2, label="mean time to noise")
    ax.set(xlim=(0, .75), ylim=(-60, 2), xlabel="s after the speaker impulse", ylabel="dB re peak",
           title=f"envelope 100–1000 Hz: {', '.join(_names(objects, OBJECTS))}, {'all speakers' if speakers is None else f'speaker {speakers}'} (mean ± std over captures)")
    ax.legend(frameon=False, ncols=2)
    return fig

def plot_time_to_noise(run, objects="empty-box", speakers=None, n=None, boxes=BOXES):
    """time to noise per frequency band (octaves in increasing order, then 100-1000 Hz), mean over groups ± pooled std over captures;
    the % under a value = the share of captures that reach the noise floor within 0.75 s (gastro rings for seconds). Same cache as plot_envelope"""
    cap = captures(run)
    fig, ax = plt.subplots(figsize=(12, 5), constrained_layout=True)
    w = .8 / len(boxes)
    for k, b in enumerate(boxes):
        *_, tm, ts, reach = _envelope_stats(cap, b, objects, speakers, n, TN_BANDS)
        x = np.arange(len(TN_BANDS)) + (k - (len(boxes) - 1) / 2) * w
        ax.bar(x, tm, w * .92, color=COLORS[b], label=b)
        ax.errorbar(x, tm, ts, fmt="none", ecolor=MUTED, lw=1, capsize=3)
        for xi, v, s_, rch in zip(x, tm, ts, reach):
            ax.annotate("> 0.75" if not np.isfinite(v) else f"{v:.2f}" + ("" if rch > .99 else f"\n{rch:.0%}"), (xi, 0 if not np.isfinite(v) else v + s_), xytext=(0, 2), textcoords="offset points", ha="center", fontsize=8)
    ax.axvline(len(OCT) - .5, c=MUTED, lw=.8, ls=":")
    ax.set_xticks(range(len(TN_BANDS)), [f"{lo}–{hi} Hz" for lo, hi in OCT] + ["all: 100–1000 Hz"])
    ax.set(ylabel="time to noise (s)", title=f"time to noise per band: {', '.join(_names(objects, OBJECTS))}, {'all speakers' if speakers is None else f'speaker {speakers}'} (± std over captures; % = captures that reach it)")
    ax.grid(axis="x", visible=False); ax.legend(frameon=False)
    return fig

def _gram(d):
    F = D.load_fft(d).numpy()                                   # (100, 1170, 2) complex
    X = F.transpose(1, 0, 2).reshape(F.shape[1], -1)
    X = X / np.linalg.norm(X, axis=1, keepdims=True)            # each frequency's spatial shape, unit length
    return X.conj().T @ X

def _patterns(g):
    """energies s_k^2 of the spatial patterns of one group's captures stacked into one tall (captures x 1170, 200) matrix (nb99 6)"""
    def run():
        with ThreadPoolExecutor(8) as ex: G = sum(ex.map(_gram, g.dir))
        return np.linalg.eigvalsh(G)[::-1]
    return _cached("eig", list(g.dir), run)

def plot_cumulative_energy(run, objects=None, speakers=None, k=40, boxes=BOXES):
    """cumulative energy of the spatial patterns (nb99 6.2): sum_{j<=k} s_j^2 / sum s^2 of each (object, speaker) group's stacked
    captures, mean ± std over groups (one curve per group, so here the std is between speakers and objects). A curve that rises fast
    = few patterns; eff_rank = (sum s^2)^2 / sum s^4; n90 = where it crosses 0.9. The eigenvalues are cached per group"""
    cap = captures(run)
    fig, ax = plt.subplots(figsize=(18, 5.5))
    for b in boxes:
        S_ = [_patterns(g) for g in _groups(cap, b, objects, speakers).values()]
        cum = np.stack([np.cumsum(s) / s.sum() for s in S_])[:, :k]
        er = [s.sum() ** 2 / (s ** 2).sum() for s in S_]
        n90 = [np.searchsorted(np.cumsum(s) / s.sum(), .9) + 1 for s in S_]
        m, sd = cum.mean(0), cum.std(0)
        x = np.arange(1, cum.shape[1] + 1)
        ax.fill_between(x, m - sd, m + sd, color=COLORS[b], alpha=.15, lw=0)
        ax.plot(x, m, "o-", c=COLORS[b], ms=4, label=f"{b}  (eff_rank {np.mean(er):.1f} ± {np.std(er):.1f}, n90 {np.mean(n90):.1f} ± {np.std(n90):.1f})")
    ax.axhline(.9, c="gray", lw=.8, ls=":")
    ax.set(xlabel="number of spatial patterns k", ylabel="fraction of energy in the top k", xlim=(.5, k + .5), ylim=(0, 1.01),
           title=f"cumulative energy of the spatial patterns: {', '.join(_names(objects, OBJECTS))}, {'all speakers' if speakers is None else f'speaker {speakers}'} (mean ± std over groups)")
    ax.legend(frameon=False, loc="lower right")
    return fig

# does moving the object change the mode frequencies, the mode shapes or the mode amplitudes? D_spec split into three parts,
# per ~100 Hz window w, for a pair of captures a, b of one speaker (u(f) = the 200-vector of lasers x axes at bin f, unit length;
# l(f) = 10 log10 mean_{l,c} |X|^2, per FFT bin):
#   frequency  δf_w    = the lag τ (|τ| <= 10 Hz, parabolic sub-bin) that maximises corr(l_a(f), l_b(f + τ)) over f in w, in Hz
#   amplitude  δA_w    = rms_f (l_a(f) - l_b(f + τ)) in dB, after that alignment
#   shape      δφ_w    = mean_f (1 - |u_a(f) . u_b(f + τ)*|^2), 1 - MAC after that alignment (0 = same shape)
# Each is set against the same thing for back-to-back empty-box pairs (repeat noise): moving the object only changes a part if
# that part is above its noise

MODE_WIN, MODE_LAG = np.linspace(13, 1157, 10).astype(int), 13   # 9 windows of ~98 Hz over the 1170 bins; ±13 bins = ±10 Hz
BIN_HZ = 1 / 1.3

def _modal(d):
    F = D.load_fft(d).numpy()
    X = F.transpose(1, 0, 2).reshape(F.shape[1], -1)                   # (1170, 200)
    P = (np.abs(X) ** 2).mean(1)
    return (10 * np.log10(P)).astype(np.float32), (X / np.sqrt(P * X.shape[1])[:, None]).astype(np.complex64)

def _decompose(la, Ua, lb, Ub) -> np.ndarray:
    """(windows, 3): |δf| Hz, δA dB, δφ for one pair"""
    out = []
    for s0, s1 in zip(MODE_WIN[:-1], MODE_WIN[1:]):
        a = la[s0:s1] - la[s0:s1].mean()
        W = np.lib.stride_tricks.sliding_window_view(lb, s1 - s0)[s0 - MODE_LAG:s0 + MODE_LAG + 1]          # (27, n): l_b shifted by τ
        Wc = W - W.mean(1, keepdims=True)
        r = Wc @ a / (np.linalg.norm(Wc, axis=1) * np.linalg.norm(a) + 1e-12)
        k = int(np.argmax(r))
        sub = 0.5 * (r[k - 1] - r[k + 1]) / (r[k - 1] - 2 * r[k] + r[k + 1]) if 0 < k < len(r) - 1 and (r[k - 1] - 2 * r[k] + r[k + 1]) < 0 else 0.
        tau = k - MODE_LAG
        amp = np.sqrt(np.mean((la[s0:s1] - W[k]) ** 2))
        mac = np.abs(np.einsum("fi,fi->f", Ua[s0:s1], Ub[s0 + tau:s1 + tau].conj())) ** 2
        out.append((abs(tau + sub) * BIN_HZ, amp, 1 - mac.mean()))
    return np.array(out)

def _mode_pairs(g, n, pairs=None, seed=0):
    """decomposition of every pair of n random captures of g (or the given index pairs), with D_pos. Cached"""
    g = g if pairs is not None or len(g) <= n else g.sample(n, random_state=seed).reset_index(drop=True)
    def run():
        with ThreadPoolExecutor(8) as ex: M = list(ex.map(_modal, g.dir))
        pr = pairs if pairs is not None else list(zip(*np.triu_indices(len(g), 1)))
        R = np.stack([_decompose(*M[i], *M[j]) for i, j in pr])
        P = np.array([np.hypot(g.com_r[i] - g.com_r[j], g.com_c[i] - g.com_c[j]) / np.sqrt(g.H[i] * g.W[i]) for i, j in pr])
        return R, P
    return _cached(f"modes{n}", list(g.dir) + ([str(pairs)] if pairs is not None else []), run)

def mode_changes(run, objects="cube", speakers=None, n=60, far=.25) -> dict:
    """{box: dict(move=(groups, windows, 3) mean over pairs moved > far of the box, move_var, noise=(groups, windows, 3), noise_var,
    rate=(groups, windows, 3) share of the moved pairs above the 95th percentile of that speaker's repeat-noise pairs)}"""
    cap, out = captures(run), {}
    for b in BOXES:
        mv, mvv, nz, nzv, rt = [], [], [], [], []
        for (o, s), g in _groups(cap, b, objects, speakers).items():
            R, P = _mode_pairs(g, n)
            mv.append(R[P > far].mean(0)); mvv.append(R[P > far].var(0))
            e = cap[(cap.box == b) & (cap.split == "empty-box") & (cap.speaker == s)].reset_index(drop=True)
            pr = [(i, j) for ix in e.groupby("layout").indices.values() for k, i in enumerate(ix) for j in ix[k + 1:]]
            Rn, _ = _mode_pairs(e, len(e), pr)
            nz.append(Rn.mean(0)); nzv.append(Rn.var(0)); rt.append((R[P > far] > np.percentile(Rn, 95, 0)).mean(0))
        out[b] = dict(move=np.stack(mv), move_var=np.stack(mvv), noise=np.stack(nz), noise_var=np.stack(nzv), rate=np.stack(rt))
    return out

MODE_PARTS = ("|δf|: frequency shift (Hz)", "δA: amplitude change (dB)", "δφ = 1 − MAC: shape change")

def plot_mode_changes(run, objects="cube", speakers=None, n=60, far=.25):
    """top: the three parts per frequency window, for pairs where the object moved > far of the box (solid, mean over groups ± pooled
    std over pairs) and for back-to-back empty-box pairs (dashed: the repeat noise), one colour per box. Bottom: per part, the share of
    moved pairs whose change is above the 95th percentile of repeat noise, mean over windows and groups (whisker = std over windows).
    0.05 = no different from recording twice, 1 = every move changes it (a ratio of means blows up for δf: repeats align at lag 0)"""
    res = mode_changes(run, objects, speakers, n, far)
    fc = (MODE_WIN[:-1] + MODE_WIN[1:]) / 2 * BIN_HZ + 100
    fig = plt.figure(figsize=(20, 9), constrained_layout=True)
    top, bot = fig.subfigures(2, 1, height_ratios=[1.25, 1])
    axs = top.subplots(1, 3)
    for b, r in res.items():
        m, sd = _pool(list(r["move"]), list(r["move_var"]))
        nm, ns = _pool(list(r["noise"]), list(r["noise_var"]))
        for k, ax in enumerate(axs):
            ax.fill_between(fc, m[:, k] - sd[:, k], m[:, k] + sd[:, k], color=COLORS[b], alpha=.13, lw=0)
            ax.plot(fc, m[:, k], "o-", c=COLORS[b], ms=4, label=f"{b} moved")
            ax.plot(fc, nm[:, k], "o--", c=COLORS[b], ms=3, lw=1, alpha=.8, label=f"{b} repeat noise")
    for ax, t in zip(axs, MODE_PARTS): ax.set(title=t, xlabel="Hz (window centre)")
    axs[0].legend(frameon=False, fontsize=8, ncols=2)
    ax = bot.subplots()
    w = .8 / len(BOXES)
    for k, (b, r) in enumerate(res.items()):
        ratio = r["rate"].mean(0)                                # (windows, 3)
        x = np.arange(3) + (k - 1) * w
        ax.bar(x, ratio.mean(0), w * .92, color=COLORS[b], label=b)
        ax.errorbar(x, ratio.mean(0), ratio.std(0), fmt="none", ecolor=MUTED, lw=1, capsize=3)
        for xi, v, s in zip(x, ratio.mean(0), ratio.std(0)): ax.annotate(f"{v:.2f}", (xi, v + s), xytext=(0, 2), textcoords="offset points", ha="center", fontsize=8)
    ax.axhline(.05, c="gray", lw=.8, ls=":")
    ax.set_xticks(range(3), ["frequency", "amplitude", "shape"])
    ax.set(ylabel="share of moves above noise", ylim=(0, 1.05), title=f"share of moves (> {far:.0%} of the box) that change each part more than 95% of repeats do (dotted = chance, 0.05; ± std over windows)")
    ax.grid(axis="x", visible=False); ax.legend(frameon=False)
    top.suptitle(f"{', '.join(_names(objects, OBJECTS))}, {'all speakers' if speakers is None else f'speaker {speakers}'}: what changes when the object moves", x=.01, ha="left")
    return fig

def _levels(g):
    """(captures, 1170) fine level l(f) = 10 log10 mean_{l,c} |X|^2 per FFT bin (0.77 Hz). Cached per group"""
    def run():
        with ThreadPoolExecutor(8) as ex: return np.stack(list(ex.map(lambda d: _modal(d)[0], g.dir))).astype(np.float16)
    return _cached("level", list(g.dir), run)

def _sigmas(g):
    """(captures, 1170) σ(f) = std over the 200 laser channels (lasers x axes) of |X|. Cached per group"""
    def one(d): return np.abs(D.load_fft(d).numpy()).transpose(1, 0, 2).reshape(1170, -1).std(1)
    def run():
        with ThreadPoolExecutor(8) as ex: return np.stack(list(ex.map(one, g.dir))).astype(np.float32)
    return _cached("sigma", list(g.dir), run)

def _odd(n): return n + 1 - n % 2

def _peaks(S, k_prom=4., distance_hz=5., smooth_hz=5., baseline_hz=25.):
    """peak bins of σ(f), as viz2.data.peaks: a real mode is where the lasers DISAGREE (a mode-shape peak or valley), so pick on the std
    across lasers, savgol-smoothed over smooth_hz, prominence >= k_prom x the MAD of its residual from a baseline_hz smooth, >= distance_hz apart"""
    b = lambda hz: max(1, int(round(hz / BIN_HZ)))
    S = savgol_filter(S, _odd(2 * b(smooth_hz) + 1), 3)
    r = S - savgol_filter(S, _odd(2 * b(baseline_hz) + 1), 3)
    sig = 1.4826 * np.median(np.abs(r - np.median(r)))
    return find_peaks(S, distance=b(distance_hz), prominence=max(1e-12, k_prom * sig), wlen=_odd(2 * b(baseline_hz) + 1))[0]

def box_resonances(run, objects=None, speakers=None, tol=3, min_occ=.8) -> tuple[dict, pd.DataFrame]:
    """resonances of the box itself: peaks that are there wherever the object is. Peaks are picked as in viz2 (_peaks: on σ(f), the
    std of |X| across the 200 laser channels). Per (object, speaker) group and FFT bin: occurrence = share of captures with a peak
    within ±tol bins (±2.3 Hz), σ_pos = std of the level l(f) over the captures (positions), and the mean level; then means over
    groups (σ pooled). Repeat noise σ_rep = the same std over back-to-back empty captures. A box resonance = a peak of the
    group-mean σ(f) with occurrence >= min_occ. Its jitter = std over captures of the nearest peak's frequency. Returns the per-bin
    curves per box and one row per resonance"""
    from scipy.ndimage import maximum_filter1d
    cap, curves, rows = captures(run), {}, []
    f = np.arange(131, 1301) / 1.3
    for b in BOXES:
        M, Vp, O, J, Vr, Sg = [], [], [], {}, [], []
        for (o, s), g in _groups(cap, b, objects, speakers).items():
            L, Sig = _levels(g).astype(np.float32), _sigmas(g)
            pk = [_peaks(x) for x in Sig]
            Sg.append(Sig.mean(0))
            hit = np.zeros(L.shape, bool)
            for i, p in enumerate(pk): hit[i, p] = True
            O.append(maximum_filter1d(hit.astype(np.uint8), 2 * tol + 1, axis=1).mean(0)); M.append(L.mean(0)); Vp.append(L.var(0))
            J[o, s] = pk
            e = cap[(cap.box == b) & (cap.split == "empty-box") & (cap.speaker == s)].reset_index(drop=True)
            Le = _levels(e).astype(np.float32)
            Vr.append(np.mean([Le[ix].var(0) for ix in e.groupby("layout").indices.values()], 0))
        m, sp = _pool(M, Vp)
        occ, sr = np.mean(O, 0), np.sqrt(np.mean(Vr, 0))
        curves[b] = dict(f=f, level=m, sigma_pos=sp, sigma_rep=sr, occurrence=occ)
        for k in _peaks(np.mean(Sg, 0)):
            if occ[k] < min_occ: continue
            near = [f[p[np.argmin(np.abs(p - k))]] for pk in J.values() for p in pk if len(p) and np.min(np.abs(p - k)) <= tol]
            rows.append(dict(box=b, f0=f[k], level=m[k], occurrence=occ[k], jitter_hz=np.std(near), sigma_pos=sp[k], sigma_rep=sr[k]))
    return curves, pd.DataFrame(rows).astype({k: float for k in ("f0", "level", "occurrence", "jitter_hz", "sigma_pos", "sigma_rep")}).round(2)

def plot_box_resonances(run, objects=None, speakers=None, tol=3, min_occ=.8):
    """per box: top, the mean level over every object capture ± std over positions, the box's resonances (peaks present in
    >= min_occ of captures, wherever the object is) as dots; bottom, σ_pos(f) (std over positions) vs σ_rep(f) (std of empty-box repeats)
    and the occurrence of a peak (right axis). A clean box resonance: occurrence near 1, a narrow σ_pos dip"""
    curves, res = box_resonances(run, objects, speakers, tol, min_occ)
    fig, axs = plt.subplots(2 * len(BOXES), 1, figsize=(20, 4.2 * len(BOXES)), sharex=True, gridspec_kw=dict(height_ratios=[1.6, 1] * len(BOXES)), constrained_layout=True)
    for i, b in enumerate(BOXES):
        c, r = curves[b], res[res.box == b]
        ax, ax2 = axs[2 * i], axs[2 * i + 1]
        ax.fill_between(c["f"], c["level"] - c["sigma_pos"], c["level"] + c["sigma_pos"], color=COLORS[b], alpha=.18, lw=0)
        ax.plot(c["f"], c["level"], c=COLORS[b], lw=1.1)
        ax.scatter(r.f0, r.level, s=36, c=INK, zorder=3, label=f"box resonance ({len(r)}: peak in ≥ {min_occ:.0%} of captures)")
        for _, q in r.iterrows(): ax.annotate(f"{q.f0:.0f}", (q.f0, q.level), xytext=(0, 5), textcoords="offset points", ha="center", fontsize=7)
        ax.set(ylabel=f"{b}\nlevel (dB)"); ax.legend(frameon=False, loc="upper right", fontsize=8)
        ax2.plot(c["f"], c["sigma_pos"], c=COLORS[b], lw=1.1, label="σ over positions")
        ax2.plot(c["f"], c["sigma_rep"], c=COLORS[b], lw=1, ls="--", label="σ of empty-box repeats")
        ax2.set(ylabel="σ (dB)", ylim=(0, None))
        t = ax2.twinx(); t.plot(c["f"], c["occurrence"], c=MUTED, lw=.8, alpha=.7); t.set(ylim=(0, 1.05), ylabel="peak occurrence"); t.grid(False)
        ax2.legend(frameon=False, loc="upper left", fontsize=8, ncols=2)
    axs[0].set_title(f"resonances of the box itself: {', '.join(_names(objects, OBJECTS))}, {'all speakers' if speakers is None else f'speaker {speakers}'} (band = ± std over positions; grey = peak occurrence)", loc="left")
    axs[-1].set_xlabel("Frequency (Hz)")
    return fig

def _gram_sum(g):
    """sum over a group's captures of X^H X (200 x 200): the stacked matrix's Gram, whose eigenvectors are V. Cached"""
    def run():
        with ThreadPoolExecutor(8) as ex: return sum(ex.map(_gram, g.dir))
    return _cached("gram", list(g.dir), run)

DIVERGE = LinearSegmentedColormap.from_list("diverge", ["#104281", "#3987e5", "#cde2fb", "#ffffff", "#fbd9c6", "#eb6834", "#8a3010"])

def plot_mode_shapes(run, objects=None, speaker=1, k=3, boxes=BOXES):
    """the top k spatial patterns (mode shapes) per box, on the 10 x 10 laser grid. X = U S V^H with X the stacked (captures x 1170
    freqs, 200) matrix of unit spatial shapes (rows = one frequency of one capture, columns = 100 lasers x 2 axes): column k of V is
    pattern k's shape on the lasers (what is drawn), U says how much each row uses it, s_k^2 is its energy. A shape's sign and global
    phase are arbitrary: each is rotated to be as real as possible (phase ½ arg Σ v²) and its largest entry made positive. Colour =
    Re v on its dominant axis (x or y shift, scaled to max 1: nodal lines are where it changes sign); arrows = (Re v_x, Re v_y).
    One speaker (shapes can't be averaged across groups), all chosen objects stacked; the title gives the pattern's energy share
    here and, mean ± std over every (object, speaker) group, in plot_cumulative_energy's terms. Grid: laser l at (l // 10, l % 10),
    in the camera's laser order (not checked against the floor's orientation)"""
    cap = captures(run)
    fig, axs = plt.subplots(len(boxes), k, figsize=(5.6 * k, 5 * len(boxes)), squeeze=False, constrained_layout=True)
    rr, cc = np.divmod(np.arange(100), 10)
    for row, b in zip(axs, boxes):
        gs = _groups(cap, b, objects, speaker)
        w, V = np.linalg.eigh(sum(_gram_sum(g) for g in gs.values()))
        w, V = w[::-1], V[:, ::-1]
        frac = w / w.sum()
        allf = np.stack([e / e.sum() for e in (_patterns(g) for g in _groups(cap, b, objects, None).values())])[:, :k]
        for j, ax in enumerate(row):
            v = V[:, j] * np.exp(-0.5j * np.angle((V[:, j] ** 2).sum()))
            v = v.reshape(100, 2)                                      # (laser, axis): the columns of X are laser-major
            v = v * np.sign(v.real.ravel()[np.argmax(np.abs(v.real.ravel()))])
            a = int(np.argmax((np.abs(v) ** 2).sum(0)))
            re = v.real / np.abs(v.real).max()
            im = ax.imshow(re[:, a].reshape(10, 10), cmap=DIVERGE, vmin=-1, vmax=1)
            ax.quiver(cc, rr, re[:, 0], re[:, 1], angles="xy", scale_units="xy", scale=2.2, width=.006, color=INK, alpha=.8)
            ax.set(xticks=range(10), yticks=range(10), xlim=(-.5, 9.5), ylim=(9.5, -.5)); ax.tick_params(labelsize=7); ax.grid(False)
            ax.set_title(f"{b} · pattern {j + 1}: {frac[j]:.0%} of the energy\n(all groups: {allf[:, j].mean():.0%} ± {allf[:, j].std():.0%}) · colour = Re {'xy'[a]} shift", fontsize=10)
            if j == 0: ax.set_ylabel(f"{b}\nlaser row")
        row[0].figure.colorbar(im, ax=row, shrink=.8, label="Re v (dominant axis), max = 1")
    for ax in axs[-1]: ax.set_xlabel("laser column")
    fig.suptitle(f"top {k} spatial patterns (columns of V): {', '.join(_names(objects, OBJECTS))}, speaker {speaker}", x=.01, ha="left")
    return fig

PALETTE = ("#2a78d6", "#eb6834", "#1baf7a", "#8c5bd6", "#d6452a", "#52514e")

def _ordinal(n):
    """n blues, light to dark: an ordered set of lines (rows / columns)"""
    return [to_hex(SEQ(v)) for v in np.linspace(.3, 1, n)]

BY = {"speaker": "Speaker", "box": "Box", "object": "Object", "laser_row": "Laser row", "laser_col": "Laser column", "pos_row": "Position row", "pos_col": "Position column"}

def plot_avg_fft(run, by="speaker", box="plastic", objects="candle", speakers=None, bins=8, min_n=5):
    """average power spectrum P(f) = mean over lasers and axes of 10 log10 |X|^2 (dB; 117 bands of 7.7 Hz), one line per value of by:
    "speaker", "box", "object", "laser_row" / "laser_col" (laser l of the 10 x 10 grid at (l // 10, l % 10)), or "pos_row" / "pos_col"
    (where the object sits on the floor, in bins equal parts of the floor; row bins-1 = next to the laser wall; bins with < min_n
    captures dropped). box / objects / speakers fix the rest (box is ignored for by="box"). Line = mean over captures ± std over
    captures; a line that mixes several (object, speaker) groups averages each group on its own and pools the std"""
    cap = captures(run)
    c = cap[cap.split.isin(_names(objects, OBJECTS))]
    if by != "box": c = c[c.box == box]
    if speakers is not None: c = c[c.speaker.isin(_names(speakers, ()))]
    c = c.reset_index(drop=True)
    x = spectra(c)[0].astype(np.float32)                                   # (N, 100, 117, 2)
    A = x.mean((1, 3))
    if by in ("laser_row", "laser_col"):
        idx = np.arange(100) // 10 if by == "laser_row" else np.arange(100) % 10
        lines = {f"row {k}" if by == "laser_row" else f"col {k}": (c, x[:, idx == k].mean((1, 3))) for k in range(10)}
    elif by in ("pos_row", "pos_col"):
        l, t, r, b = FLOOR[box]
        f = (c.com_r / c.H - t) / (b - t) if by == "pos_row" else (c.com_c / c.W - l) / (r - l)
        k_ = np.clip((f.to_numpy() * bins).astype(int), 0, bins - 1)
        lines = {f"{'row' if by == 'pos_row' else 'col'} {k}" + (" (laser wall)" if by == "pos_row" and k == bins - 1 else ""): (c[k_ == k], A[k_ == k]) for k in range(bins) if (k_ == k).sum() >= min_n}
    else:
        key = dict(speaker="speaker", box="box", object="split")[by]
        order = BOXES if by == "box" else sorted(c[key].unique(), key=str)
        lines = {(f"speaker {k}" if by == "speaker" else k): (c[c[key] == k], A[(c[key] == k).to_numpy()]) for k in order if (c[key] == k).any()}
    colors = [COLORS[k] for k in lines] if by == "box" else _ordinal(len(lines)) if by in ("laser_row", "laser_col", "pos_row", "pos_col") else PALETTE
    fig, ax = plt.subplots(figsize=(18, 5.5), constrained_layout=True)
    for (k, (g, a)), col in zip(lines.items(), colors):
        gi = [np.flatnonzero(((g.split == o) & (g.speaker == s_)).to_numpy()) for o, s_ in dict.fromkeys(zip(g.split, g.speaker))]
        m, sd = _pool([a[i].mean(0) for i in gi], [a[i].var(0) for i in gi])
        ax.fill_between(FB, m - sd, m + sd, color=col, alpha=.1 if len(lines) > 4 else .18, lw=0)
        ax.plot(FB, m, c=col, lw=1.4, label=f"{k}  (n={len(g)})")
    fixed = [f"object={', '.join(_names(objects, OBJECTS)) if objects is not None else 'all 4'}" if by != "object" else None,
             f"speaker={speakers if speakers is not None else 'all'}" if by != "speaker" else None, f"box={box}" if by != "box" else None]
    ax.set(xlabel="Frequency (Hz)", ylabel="Power (dB) = 10 log10 |FFT|²", title=f"Power vs {BY[by]} ({', '.join(f for f in fixed if f)})")
    ax.legend(frameon=False, ncols=2 if len(lines) > 5 else 1, fontsize=9, loc="upper right")
    return fig

def compare_objects(run, objects=OBJECTS, speakers=None, n=150, near=.05, gap=(30, 120)) -> dict:
    """{box: DataFrame (a, b, d_spec, shape, n)}: two captures within near of the box of each other (same speaker), recorded gap[0]-gap[1]
    minutes apart, for every pair of objects a <= b (a = b: the same object back at that spot). d_spec = spectral distance (dB), shape =
    mean 1 - MAC over frequency (mode-shape change). Plus box "<box>/empty": the empty box against itself at the same gap. n random
    captures per object and speaker. Per object pair: mean over speakers of the pair means, ± pooled std over pairs. Cached"""
    cap, out = captures(run), {}
    for box in BOXES:
        acc = {}
        spk = sorted(set(cap[(cap.box == box) & cap.split.isin(objects)].speaker) & set(_names(speakers, cap.speaker.unique())))
        for s_ in spk:
            g = pd.concat([cap[(cap.box == box) & (cap.split == o) & (cap.speaker == s_)].pipe(lambda x: x.sample(min(n, len(x)), random_state=0)) for o in objects]).reset_index(drop=True)
            r, c, sz = g.com_r.to_numpy(float), g.com_c.to_numpy(float), np.sqrt(g.H * g.W).to_numpy(float)
            P_ = np.hypot(r[:, None] - r, c[:, None] - c) / sz
            t = np.array([x.timestamp() for x in g.t]) / 60
            dt = np.abs(t[:, None] - t)
            iu = np.triu_indices(len(g), 1)
            ok = (P_[iu] < near) & (dt[iu] >= gap[0]) & (dt[iu] <= gap[1])
            pr = [(int(i), int(j)) for i, j in zip(iu[0][ok], iu[1][ok])]
            if not pr: continue
            R, _ = _mode_pairs(g, len(g), pr)
            x = spectra(g)[0]
            ds = np.array([d_spec(x[i:i + 1], x[j:j + 1])[0, 0] for i, j in pr])
            sp = g.split.to_numpy()
            for (i, j), d, sh in zip(pr, ds, R[:, :, 2].mean(1)):
                acc.setdefault((s_, *sorted((sp[i], sp[j]), key=list(objects).index)), []).append((d, sh))
            e = cap[(cap.box == box) & (cap.split == "empty-box") & (cap.speaker == s_)].reset_index(drop=True)
            te = np.array([x.timestamp() for x in e.t]) / 60
            pe = [(i, j) for i in range(len(e)) for j in range(i + 1, len(e)) if gap[0] <= abs(te[i] - te[j]) <= gap[1]]
            if pe:
                Re, _ = _mode_pairs(e, len(e), pe)
                xe = spectra(e)[0]
                acc[(s_, "empty", "empty")] = [(d_spec(xe[i:i + 1], xe[j:j + 1])[0, 0], sh) for (i, j), sh in zip(pe, Re[:, :, 2].mean(1))]
        rows = []
        for a, b in dict.fromkeys((a, b) for _, a, b in acc):
            v = [np.array(acc[k]) for k in acc if k[1:] == (a, b)]
            m, sd = _pool([x.mean(0) for x in v], [x.var(0) for x in v])
            rows.append(dict(a=a, b=b, d_spec=m[0], d_spec_std=sd[0], shape=m[1], shape_std=sd[1], n=sum(len(x) for x in v)))
        out[box] = pd.DataFrame(rows)
    return out

def plot_objects(run, objects=OBJECTS, speakers=None, n=150, near=.05, gap=(30, 120), min_n=10):
    """do different objects change the box differently? For every pair of objects, two captures at the same spot (CoMs < near of the
    box apart, same speaker), recorded gap minutes apart. Diagonal = the same object back at the same spot; off-diagonal = a different
    object there. Same time gap everywhere, so drift is the same in every cell (each object was recorded in its own session, hours from
    the others). If off-diagonal ≈ diagonal, the box can't tell the objects apart beyond where they are; if bigger, the objects change it
    differently. Top: spectral distance; bottom: mode-shape change 1 - MAC. Cell = mean ± std over pairs (n pairs); fewer than min_n
    pairs: blank. Light = small, dark = big. The title gives the empty box at the same gap (drift alone) and same vs different means"""
    res = compare_objects(run, objects, speakers, n, near, gap)
    k = len(objects)
    fig, axs = plt.subplots(2, len(BOXES), figsize=(6.2 * len(BOXES), 11), constrained_layout=True)
    for col, b in enumerate(BOXES):
        r = res[b]
        emp = r[r.a == "empty"]
        r = r[(r.a != "empty") & (r.n >= min_n)]
        for row, (f, lab) in enumerate((("d_spec", "spectral distance (dB)"), ("shape", "mode-shape change 1 − MAC"))):
            ax, M = axs[row, col], np.full((k, k), np.nan)
            for _, q in r.iterrows():
                i, j = objects.index(q.a), objects.index(q.b)
                M[i, j] = M[j, i] = q[f]
            lo, hi = np.nanmin(M), np.nanmax(M)
            ax.imshow(M, cmap=BLUES, vmin=lo, vmax=hi)
            for _, q in r.iterrows():
                i, j = objects.index(q.a), objects.index(q.b)
                for (y, x_) in {(i, j), (j, i)}:
                    ax.text(x_, y, f"{q[f]:.{1 if f == 'd_spec' else 3}f}\n± {q[f + '_std']:.{1 if f == 'd_spec' else 3}f}\nn={q.n}", ha="center", va="center", fontsize=8,
                            color="white" if (q[f] - lo) / max(hi - lo, 1e-9) > .55 else INK)
            same, diff = r[r.a == r.b][f].mean(), r[r.a != r.b][f].mean()
            e = f"empty {emp[f].iloc[0]:.{2 if f == 'd_spec' else 3}f}" if len(emp) else "no empty pairs at this gap"
            ax.set_xticks(range(k), objects, rotation=20); ax.set_yticks(range(k), objects); ax.grid(False)
            ax.set_title(f"{b} · {lab}\nsame object {same:.{2 if f == 'd_spec' else 3}f} · different {diff:.{2 if f == 'd_spec' else 3}f} · {e}", fontsize=10)
    fig.suptitle(f"same spot (CoMs < {near:.0%} apart), recorded {gap[0]}–{gap[1]} min apart, {'all speakers' if speakers is None else f'speaker {speakers}'}: "
                 "diagonal = same object, off-diagonal = two different objects (mean ± std over pairs)", x=.01, ha="left")
    return fig
