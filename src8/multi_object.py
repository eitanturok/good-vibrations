"""Two-object eval scenes: does the model draw both objects? Per scene (iou@0.08 pred vs each instance mask, image/smasks/*.npy):
  both / one / none   >10% of each / exactly one / neither object covered
  cov_big, cov_small  covered fraction of the larger / smaller object
  merged              one blob touches both objects (a single blob between them)
  n_blobs, phantom    connected blobs; a blob touching no object
  count_2             the count head says 2;  com_err_missed / com_err_found: the com head's nearest slot to each missed / found object
  .venv/bin/python src8/multi_object.py RUN [RUN ...] [--epoch 500]   -> runs/<RUN>/multi_object.csv, printed summary"""
import argparse, sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).parent))
from score import ROOT, sample_dir, last_epoch

T = 0.08

def scenes(run, ep):
    rows = []
    for f in sorted((ROOT / "runs" / run / "outputs_history/eval").glob(f"*/*/ep{ep:04d}-ba*.pt")):
        d = torch.load(f, weights_only=False, map_location="cpu"); p, info = d["mask_pred"].float().numpy(), d["info"]
        H, W = p.shape[-2:]
        for i in range(len(p)):
            sd = sample_dir(info["experiment"][i], int(info["sample_id"][i]), int(info["position_id"][i]), int(info["speaker"][i]))
            inst = sorted((sd / "image/smasks").glob("*.npy"))
            if len(inst) != 2: continue
            ms = [np.asarray(Image.fromarray((np.load(x) > 0).astype(np.float32)).resize((W, H), Image.BOX)) > .5 for x in inst]
            b = p[i] > T; lab, n = ndimage.label(b)
            cov = [(b & m).sum() / max(m.sum(), 1) for m in ms]
            big, small = (0, 1) if ms[0].sum() >= ms[1].sum() else (1, 0)
            touch = [set(np.unique(lab[m & b])) - {0} for m in ms]
            u = ms[0] | ms[1]
            c = [(np.array(np.nonzero(m)).mean(1) + .5) / [H, W] for m in ms]
            head = d["com"][i].float().numpy()
            found = [x > .1 for x in cov]
            rows.append(dict(box=f.parent.parent.name, split=f.parent.name, types="+".join(sorted(x.stem.rstrip("0123456789") for x in inst)),
                             iou=(b & u).sum() / max((b | u).sum(), 1), both=all(found), one=sum(found) == 1, none=not any(found),
                             cov_big=cov[big], cov_small=cov[small], merged=len(touch[0] & touch[1]) > 0, n_blobs=n,
                             phantom=float(n > 0 and len(set(range(1, n + 1)) - touch[0] - touch[1]) > 0), area_ratio=b.sum() / max(u.sum(), 1),
                             count_2=int(d["n_objects_logits"][i].argmax()) == 2,
                             com_err_found=np.mean([np.linalg.norm(head - cc, axis=1).min() for cc, ok in zip(c, found) if ok]) if any(found) else np.nan,
                             com_err_missed=np.mean([np.linalg.norm(head - cc, axis=1).min() for cc, ok in zip(c, found) if not ok]) if not all(found) else np.nan))
    return pd.DataFrame(rows)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("runs", nargs="+"); ap.add_argument("--epoch", type=int)
    a = ap.parse_args()
    cols = ["iou", "both", "one", "none", "merged", "n_blobs", "phantom", "area_ratio", "count_2", "com_err_found", "com_err_missed"]
    summary = {}
    for run in a.runs:
        df = scenes(run, a.epoch or last_epoch(run))
        df.to_csv(ROOT / "runs" / run / "multi_object.csv", index=False)
        print(f"\n{run}: {len(df)} two-object eval scenes")
        print(df.groupby(["box", "types"])[cols[:8]].mean().round(2).to_string())
        summary[run] = df[cols].mean()
    print("\nall two-object scenes"); print(pd.DataFrame(summary).T.round(3).to_string())

if __name__ == "__main__":
    main()
