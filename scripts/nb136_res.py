# nb136 Q1 references: per-capture σ(f) (std of |X| over the 200 channels), channel-mean level (dB) and encoder-input energy per band,
# on training single-object captures of the 4 ID objects (≤100 per object x speaker) + training empty-box captures -> runs/_src9_cache/nb136/res.npz
import sys, numpy as np, torch
from pathlib import Path
ROOT = Path("/home/ethantu/workspace/good-vibrations"); sys.path.insert(0, str(ROOT / "src9"))
from dataset import build_dataset
OUT = ROOT / "runs/_src9_cache/nb136"
_, _, tr_ev, _ = build_dataset("tukey", num_workers=0)
DS = tr_ev.dataloader.dataloader.dataset.dataset
B = np.load(OUT / "base.npz")
BOX = {"2026_09_27_gastronorm_four_objs": "gastro", "2026_10_05_green_plastic_four_objs": "plastic", "2026_10_11_cardboard_four_objs": "cardboard"}
ID = {"candle", "red-cube", "soap dispenser", "vase"}
rng = np.random.default_rng(0)
tr = np.nonzero(B["split"] == "train")[0]
M = [DS.metas[int(B["i"][j])] for j in tr]
key = np.array([f'{BOX[m["experiment"]]}|{m["speaker"]}|{"+".join(sorted(m["objects"])) or "empty"}' for m in M])
pick = []
for k in np.unique(key):
    if k.split("|")[2] in ID | {"empty"}: ix = np.nonzero(key == k)[0]; pick += sorted(rng.choice(ix, min(100, len(ix)), replace=False).tolist())
pick = np.array(pick); print(len(pick), "captures", flush=True)
SIG, LVL, EZ, XN = [], [], [], []
with torch.no_grad():
    for s in range(0, len(pick), 128):
        x = torch.stack([DS[int(B["i"][tr[j]])]["fft"] for j in pick[s:s + 128]]).cuda()   # (n,100,1170,2) |X| / sample std
        c = x.permute(0, 2, 1, 3).flatten(2)                                              # (n,1170,200)
        SIG.append(c.std(2).cpu()); LVL.append((10 * (c ** 2).mean(2).log10()).cpu())
        z = (x - x.mean((1, 2, 3), keepdim=True)) / x.std((1, 2, 3), keepdim=True)          # what the encoder sees
        EZ.append((z ** 2).mean((1, 3)).view(len(x), 18, 65).mean(2).cpu())                # mean z² per band
        XN.append((x / x.std((1, 2, 3), keepdim=True)).mean((1, 3)).view(len(x), 18, 65).mean(2).cpu())  # linear |X| / σ_sample (z before the mean shift), per band
np.savez(OUT / "res.npz", idx=tr[pick], key=key[pick], sigma=torch.cat(SIG).numpy(), level=torch.cat(LVL).numpy().astype(np.float32), ez=torch.cat(EZ).numpy(), xn=torch.cat(XN).numpy())
print("done")
