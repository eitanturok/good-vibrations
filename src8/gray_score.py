"""Score a --gray run's drawn photos (runs/<run>/outputs_history/eval) against the real ones, per sample, then score.summarize's
splits and categories. All L1 on the TRUE object pixels unless named otherwise (the object is 1-3% of the photo, so a whole-photo
score is the floor's):
  obj      the drawn photo (composite): what the run shows
  app      the appearance head alone (teacher-forced: how it would look with a perfect mask)
  loc      the drawn photo on the true object grown by a 2 px ring: its look AND its placement (spill-over pays on the ring)
  floor    the empty floor drawn there: each box's per-pixel median of the true eval photos (the same bar for every run).
           Bars from the data (EXPERIMENTS.md): floor ~.30, true mask + the (box, object) mean brightness ~.13
  bright   |mean drawn - mean true| over the object: the brightness alone, no shape
  grad     L1 of the image gradients within 2 px of the object: contour and inner detail
  photo    whole-photo L1
  .venv/bin/python src8/gray_score.py RUN [RUN ...] [--epoch 500]   -> runs/<run>/scores/gray_ep<e>.pkl, printed table"""
import argparse, sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).parent))
from score import ROOT, last_epoch, summarize, truth

COLS = ["loc", "obj", "app", "floor", "bright", "grad", "photo"]

def grads(x): return np.abs(np.diff(x, axis=0))[:, :-1] + np.abs(np.diff(x, axis=1))[:-1]

def per_sample(run, epoch):
    cache = ROOT / "runs" / run / "scores" / f"gray_ep{epoch:04d}.pkl"
    if cache.exists() and "loc" in (df := pd.read_pickle(cache)): return df   # older caches have no loc: rebuild
    rows = []
    for f in sorted((ROOT / "runs" / run / "outputs_history/eval").glob(f"*/*/ep{epoch:04d}-ba*.pt")):
        d = torch.load(f, weights_only=False, map_location="cpu")
        if "gray_pred" not in d: raise KeyError(f"{run}: no gray_pred (not a --gray run)")
        g, a, info = d["gray_pred"].float().numpy(), d["gray_app"].float().numpy(), d["info"]
        H, W = g.shape[-2:]
        for i in range(len(g)):
            ids = info["experiment"][i], info["sample_id"][i], int(info["position_id"][i]), int(info["speaker"][i])
            t, o = truth(*ids, H, W, gray=True), truth(*ids, H, W) > 0.5
            ring = ndimage.binary_dilation(o, np.ones((5, 5), bool))
            near = ndimage.binary_dilation(o, iterations=2)[:-1, :-1]
            l1 = lambda x: np.abs(x - t)[o].mean() if o.any() else np.nan
            rows.append(dict(box=f.parent.parent.name, split=f.parent.name, experiment=ids[0], position_id=ids[2], speaker=ids[3],
                             obj=l1(g[i]), app=l1(a[i]), _t=t, _o=o, loc=np.abs(g[i] - t)[ring].mean() if o.any() else np.nan,
                             bright=abs(g[i][o].mean() - t[o].mean()) if o.any() else np.nan,
                             grad=np.abs(grads(g[i]) - grads(t))[near].mean() if near.any() else np.nan, photo=np.abs(g[i] - t).mean()))
    df = pd.DataFrame(rows)
    floor = {b: np.median(np.stack(g._t), 0) for b, g in df.groupby("box")}
    df["floor"] = [np.abs(floor[b] - t)[o].mean() if o.any() else np.nan for b, t, o in zip(df.box, df._t, df._o)]
    df = df.drop(columns=["_t", "_o"])
    cache.parent.mkdir(parents=True, exist_ok=True); df.to_pickle(cache)
    return df

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("runs", nargs="+"); ap.add_argument("--epoch", type=int, default=None)
    a = ap.parse_args()
    pd.set_option("display.width", 250)
    for run in a.runs:
        ep = a.epoch or last_epoch(run)
        s, sp = summarize(per_sample(run, ep), COLS)
        print(f"\n== {run} ep{ep}\n" + s.loc[["CORE", "HARD-kept", "WEDGE", "UNSEEN-POS", "ALL", "ALL-ood"] + [k for k in s.index if k.endswith("/all")]].round(3).to_string())
        print(sp[sp.split.isin(["candle", "cube", "soap-dispenser", "vase", "coffee-pot", "two-cubes", "cube-vase"])].pivot(index="split", columns="box", values="obj").round(3).to_string())
