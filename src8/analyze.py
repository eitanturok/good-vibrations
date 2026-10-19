"""Where does a run shine and where does it fail? A breakdown of iou / miss / com err / area ratio / orientation / phantoms (@0.08)
by object (split), layout (capture session), speaker, floor region, distance to the laser wall, distance to the nearest training
position and object size, over the CORE splits (vase, cube, soap, candle x 3 boxes) and over everything, with the best and worst
groups and (with --ref) the biggest wins and losses against a reference run. Writes runs/<run>/analysis.md.
  .venv/bin/python src8/analyze.py RUN@EP [--ref RUN@EP]"""
import argparse, sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from score import ROOT, per_sample
from dataset import CATEGORIES

T = 0.08
M = {f"iou@{T}": "iou", f"miss@{T}": "miss", f"com_err@{T}": "com_err", f"area_ratio@{T}": "area", f"orient_err@{T}": "orient", f"phantom@{T}": "phantom"}
CORE = ("candle", "cube", "soap-dispenser", "vase")

def load(arg):
    run, ep = arg.split("@")
    d = per_sample(run, int(ep), [T]).rename(columns=M)
    cat = {(b, s): c for b, cs in CATEGORIES.items() for c, ss in cs.items() for s in ss}
    d["category"] = [cat.get((b, s), "?") for b, s in zip(d.box, d.split)]
    d = d[d.category != "control"].copy()
    d["core"] = d.split.isin(CORE)
    # floor region: rows near / mid / far from the laser wall (the bottom of the crop), cols left / center / right
    d["wall"] = pd.cut(d.wall_dist, [-.01, 1 / 3, 2 / 3, 1.01], labels=["near wall", "mid", "far wall"])
    d["side"] = pd.cut(d.com_c, [-.01, 1 / 3, 2 / 3, 1.01], labels=["left", "center", "right"])
    d["region"] = d.wall.astype(str) + " / " + d.side.astype(str)
    d["train_dist_bin"] = pd.cut(d.train_dist, [-.01, .03, .06, .1, .15, .25, 2])
    d["size"] = d.groupby("box").true_area.transform(lambda a: pd.qcut(a, 4, labels=["smallest", "small", "large", "largest"], duplicates="drop")).astype(str)
    return d

def table(d, by):
    g = d.groupby(by, observed=True)
    t = g[list(M.values())].mean()
    t.insert(0, "n", g.size())
    return t

DIMS = [("object", ["box", "split"]), ("layout", ["box", "layout"]), ("speaker", ["box", "speaker"]), ("region", ["box", "region"]),
        ("wall distance", ["box", "wall"]), ("train distance", ["train_dist_bin"]), ("size", ["box", "size"])]

def report(arg, ref=None, min_n=20):
    d = load(arg)
    r = load(ref) if ref else None
    lines = [f"# Analysis of {arg}" + (f" (vs {ref})" if ref else ""), "",
             f"Threshold {T}; non-control eval splits; groups with n < {min_n} are left out of the best/worst lists. "
             "CORE = vase, cube, soap, candle × 3 boxes.", ""]
    findings = []
    for scope, sel in (("CORE", lambda x: x[x.core]), ("ALL", lambda x: x)):
        for name, by in DIMS:
            if scope == "ALL" and name in ("region", "wall distance", "size"): continue  # positions only make sense on the grids
            t = table(sel(d), by)
            if r is not None:
                tr = table(sel(r), by)
                t["Δiou"] = t.iou - tr.iou.reindex(t.index)
                t["Δmiss"] = t.miss - tr.miss.reindex(t.index)
            lines += [f"## {scope}: by {name}", "", t.round(3).to_markdown(), ""]
            big = t[t.n >= min_n]
            if len(big) > 1:
                lo, hi = big.iou.idxmin(), big.iou.idxmax()
                findings.append(f"- {scope} {name}: worst {lo} iou {big.iou.min():.3f} (miss {big.loc[lo, 'miss']:.2f}, com {big.loc[lo, 'com_err']:.3f}, "
                                f"area {big.loc[lo, 'area']:.2f}); best {hi} iou {big.iou.max():.3f}")
                if r is not None and big["Δiou"].notna().any():
                    w, l = big["Δiou"].idxmax(), big["Δiou"].idxmin()
                    findings.append(f"  - vs ref: biggest win {w} {big.loc[w, 'Δiou']:+.3f}, biggest loss {l} {big.loc[l, 'Δiou']:+.3f}")
    lines[5:5] = ["## Findings", ""] + findings + [""]
    out = ROOT / "runs" / arg.split("@")[0] / "analysis.md"
    out.write_text("\n".join(lines))
    print("\n".join(["## " + arg] + findings))
    print(f"\n-> {out}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("run"); ap.add_argument("--ref")
    a = ap.parse_args()
    report(a.run, a.ref)
