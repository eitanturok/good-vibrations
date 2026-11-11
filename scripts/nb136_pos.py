# nb136 Q1b position-variance variants: per-laser per-band mean of the linear model input v = |X| / σ_sample (z before the mean shift),
# for every training capture with ≤ 1 object -> runs/_src9_cache/nb136/pos.npz (j = row in base.npz, vb (n,100,18) float16)
import sys, numpy as np, torch
from pathlib import Path
ROOT = Path("/home/ethantu/workspace/good-vibrations"); sys.path.insert(0, str(ROOT / "src9"))
from dataset import build_dataset
OUT = ROOT / "runs/_src9_cache/nb136"
_, _, tr_ev, _ = build_dataset("tukey", num_workers=0)
DS = tr_ev.dataloader.dataloader.dataset.dataset
B = np.load(OUT / "base.npz"); I = B["i"]
j = np.array([k for k in np.nonzero(B["split"] == "train")[0] if DS.metas[int(I[k])]["n_objects"] <= 1]); print(len(j), "captures", flush=True)
VB = []
with torch.no_grad():
    for s in range(0, len(j), 128):
        x = torch.stack([DS[int(I[k])]["fft"] for k in j[s:s + 128]]).cuda()
        v = x / x.std((1, 2, 3), keepdim=True)                                      # transformer4.py:139 without the mean shift
        VB.append(v.view(len(x), 100, 18, 65, 2).mean((3, 4)).half().cpu())
np.savez(OUT / "pos.npz", j=j, vb=torch.cat(VB).numpy())
print("done")
