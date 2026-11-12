# nb140 extra cache (CPU, ~5 min, streams every FFT once): per-sample IoU / CoM / metadata for ALL single-object ID samples
# (candle, cube, soap, vase; train + eval/<box>/<obj> + unseen-speaker cube/vase: spk 4 and its seen-speaker baseline at the same positions) of decres-rescale-aug10-v1.
# preds/masks from the nb136 base cache (eager fp32, p = 255·sigmoid); spec = laser-mean LINEAR model input |FFT|/σ (1170, 2) re-read from the MDS
# -> runs/_src9_cache/nb140/extra.npz
import sys, numpy as np
from pathlib import Path
ROOT = Path("/home/ethantu/workspace/good-vibrations"); sys.path.insert(0, str(ROOT / "src9"))
from dataset import load_splits, build_dataset
B = np.load(ROOT / "runs/_src9_cache/nb136/base.npz")
metas = list(load_splits()[0].values())
OBJ = {"candle": "candle", "red-cube": "red-cube", "soap dispenser": "soap dispenser", "vase": "vase"}
BOXN = {"gastronorm": "gastro", "green-plastic": "plastic", "cardboard": "cardboard"}
ID = {f"eval/{b}/{o}" for b in BOXN.values() for o in ("candle", "cube", "soap-dispenser", "vase", "cube-unseen-speaker-unseen", "vase-unseen-speaker-unseen", "cube-unseen-speaker-baseline", "vase-unseen-speaker-baseline")}
keep, rows = [], []
for j, (i, s) in enumerate(zip(B["i"], B["split"])):
    m = metas[int(i)]
    if m["n_objects"] == 1 and list(m["objects"])[0] in OBJ and m["box"] in BOXN and (s == "train" or s in ID): rows.append(j)
rows = np.array(rows); M = [metas[int(B["i"][j])] for j in rows]
p, y = B["p"][rows] > .08 * 255, B["y"][rows]                     # the eval threshold .08
iou = (p & y).sum((1, 2)) / np.maximum((p | y).sum((1, 2)), 1)
g = (np.arange(64) + .5) / 64
def com(a):  # (n,64,64) -> (n,2) (row, col) as box fractions; nan if empty
    w = a.sum((1, 2)).astype(float); w[w == 0] = np.nan
    return np.stack([(a.sum(2) * g).sum(1) / w, (a.sum(1) * g).sum(1) / w], 1)
_, ev, tr_ev, _ = build_dataset("tukey", num_workers=0)
DS = {"train": tr_ev.dataloader.dataloader.dataset.dataset, "eval": ev[0].dataloader.dataloader.dataset.dataset}
spec = np.stack([DS["train" if B["split"][j] == "train" else "eval"][int(B["i"][j])]["fft"].numpy().mean(0) for j in rows])   # (n,1170,2)
S = dict(rows=rows, set=np.where(B["split"][rows] == "train", "train", "test"), box=np.array([BOXN[m["box"]] for m in M]),
         obj=np.array([list(m["objects"])[0] for m in M]), spk=np.array([m["speaker"] for m in M]), pos=np.array([m["position_id"] for m in M]),
         split=B["split"][rows], hw=np.array([m["crop_overhead_shape"][:2] for m in M]), iou=iou.astype(np.float32), com_t=com(y), com_p=com(p), axis_e=spec.mean(1), spec=spec.astype(np.float16))
np.savez(ROOT / "runs/_src9_cache/nb140/extra.npz", **S)
print({(st, b): int(((S["set"] == st) & (S["box"] == b)).sum()) for st in ("train", "test") for b in BOXN.values()})
