"""Rank runs on the six things we want from a model (iou / miss @0.08, eval splits of score.py):
  1 dist        the four base objects (candle, cube, vase, soap) per box: mean iou, and per object
  2 multi       two-object scenes (multi_object.py): both found, union iou
  3 boxes       the worst box's base-object iou (works everywhere, not just on average)
  4 rotation    cardboard rotated soap (HARD) iou + orientation error, cube-rotated iou
  5 unseen      speaker (*-unseen-speaker-unseen), laser (*laser*shifted*), rotation (*sideways), object (never-trained objects)
  6 light       cylinder-100g / -200g, glass cup
  rank = mean of the per-column ranks over the headline columns (H)
  .venv/bin/python src8/rank.py RUN [RUN ...] [--epoch 500]   -> runs/rank.csv, printed table"""
import argparse, sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from score import ROOT, per_sample
from multi_object import scenes

T = 0.08
BASE = ["candle", "cube", "vase", "soap-dispenser"]
UNSEEN_OBJ = ["coffee-pot", "biscoff", "tomato-sauce", "portable-charger"]        # eval_frac 1.0 in every box
LIGHT = ["cylinder-100g", "cylinder-200g", "glass-cup"]
H = ["dist", "both", "union", "worst_box", "rot_iou", "rot_deg", "spk", "laser", "unseen_rot", "unseen_obj", "light"]
LOWER = {"rot_deg"}                                                              # lower is better

def row(run, ep):
    df = per_sample(run, ep, [T])
    iou = lambda m: df.loc[m, f"iou@{T}"].mean()
    r = {}
    base = df.split.isin(BASE)
    for o in BASE: r[o] = iou(base & (df.split == o))
    per_box = df[base].groupby(["box", "split"])[f"iou@{T}"].mean().groupby("box").mean()
    r["dist"] = per_box.mean(); r["worst_box"] = per_box.min()
    for b, v in per_box.items(): r[b] = v
    hard = (df.box == "cardboard") & (df.split == "soap-dispenser-rotated")
    r["rot_iou"], r["rot_deg"] = iou(hard), df.loc[hard, f"orient_err@{T}"].mean()
    r["cube_rot"] = iou(df.split == "cube-rotated")
    r["spk"] = iou(df.split.str.endswith("unseen-speaker-unseen"))
    r["laser"] = iou(df.split.str.contains("laser"))
    r["unseen_rot"] = iou(df.split.isin(["soap-dispenser-sideways", "portable-charger-sideways"]) & (df.box != "cardboard") | (df.split == "portable-charger-sideways"))
    r["unseen_obj"] = iou(df.split.isin(UNSEEN_OBJ))
    r["light"] = iou(df.split.isin(LIGHT))
    s = scenes(run, ep)
    r["both"], r["union"] = s.both.mean(), s.iou.mean()
    return r

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("runs", nargs="+"); ap.add_argument("--epoch", type=int, default=500)
    a = ap.parse_args()
    t = pd.DataFrame({run: row(run, a.epoch) for run in a.runs}).T
    rk = pd.DataFrame({c: t[c].rank(ascending=c in LOWER) for c in H})
    t["rank"] = rk.mean(1)
    t = t.sort_values("rank")
    t.to_csv(ROOT / "runs" / "rank.csv")
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 50)
    print(t[H + ["rank"]].round(3).to_string()); print()
    print(t[BASE + ["cardboard", "gastro", "plastic", "cube_rot"]].round(3).to_string())
