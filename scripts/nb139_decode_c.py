"""nb139 cell 3 (pair analogy): decode the mean interaction latent c = mean(pair) − ½(mean X + mean Y), per (pair, box, speaker).
Full mean (not leave-one-out). Columns: c · X · Y · ½(X+Y) · ½(X+Y)+c (= mean pair) · 0 (decoder bias). → notebooks/out/139_decode_c.pdf

  .venv/bin/python scripts/nb139_decode_c.py"""
import json, sys, numpy as np, pandas as pd, torch, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src9"))
RUN, BAD, THR = "decres-rescale-aug10-v1", "shifted|unseen-speaker|lid", 0.08
PAIRS = {"red-cube+vase": ["red-cube", "vase"], "green-cube+red-cube": ["red-cube", "red-cube"]}  # nb139 cell 1 (both parts of the cube pair are red-cube)

C = ROOT / "runs/_src9_cache/nb137" / RUN
df, Z = pd.read_csv(C / "meta.csv", dtype={"speaker": str}), np.load(C / "feats.npz")["latent"].astype(np.float32)
df = df[~df.split.str.contains(BAD)]

rows = []  # (label, [(name, latent)])
for pair, (X, Y) in PAIRS.items():
    for (box, spk), p in df[df.object == pair].groupby(["box", "speaker"]):
        S = df[(df.box == box) & (df.speaker == spk)]; iX, iY = S.index[S.object == X], S.index[S.object == Y]
        if not len(iX) or not len(iY): print("skip", pair, box, spk, "(missing single)"); continue
        mp, mx, my = Z[p.index].mean(0), Z[iX].mean(0), Z[iY].mean(0); h = .5 * (mx + my); c = mp - h
        print(f"{pair} · {box} · spk {spk}: n_pair={len(p)} n_X={len(iX)} n_Y={len(iY)}  |c|={np.linalg.norm(c):.2f} |pair|={np.linalg.norm(mp):.2f} "
              f"|½(X+Y)|={np.linalg.norm(h):.2f} cos(c,pair)={c @ mp / np.linalg.norm(c) / np.linalg.norm(mp):.3f}")
        rows.append((f"{pair} · {box} · spk {spk} · n_pair={len(p)}, n_X={len(iX)}, n_Y={len(iY)}",
                     [("c", c), (f"mean {X}", mx), (f"mean {Y}", my), ("½(X+Y)", h), ("½(X+Y)+c", h + c), ("0", np.zeros_like(c))]))

# ---- decoder (same construction as scripts/nb137_object_image.py)
from dataset import build_dataset
from arch.transformer4 import Transformer4
torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
a = json.load(open(ROOT / "runs" / RUN / "args.json")); CK = ROOT / "runs" / RUN / "checkpoints"
_, _, _, info = build_dataset(a.get("window", "tukey"), num_workers=0)
model = Transformer4(info, decoder=a.get("decoder", "two"), latent_norm=a.get("latent_norm", "ln"), freq_rope=a.get("freq_rope", "linear")).cuda().eval()
print((CK / "latest-rank0.pt").resolve().name, model.load_state_dict(torch.load(CK / "latest-rank0.pt", map_location="cpu", weights_only=False)["state"]["model"]))
U = torch.from_numpy(np.stack([u for _, L in rows for _, u in L])).cuda()
with torch.no_grad(): P = model.decoder(U).sigmoid().float().cpu().numpy().reshape(len(rows), 6, 64, 64)

# ---- figure
nc = 6; fig, ax = plt.subplots(len(rows), nc, figsize=(2.1 * nc, 2.35 * len(rows)), squeeze=False)
rr, cc = np.mgrid[:64, :64]
for r, (lab, L) in enumerate(rows):
    for k, (nm, _) in enumerate(L):
        m = P[r, k]; b = m > THR; a_ = ax[r, k]
        a_.imshow(m, cmap="gray", vmin=0, vmax=1); a_.axis("off")
        if b.any() and not b.all(): a_.contour(m, [THR], colors="red", linewidths=.8)
        com = f" com=({rr[b].mean():.0f},{cc[b].mean():.0f})" if b.any() else ""
        print(f"  {lab.split(' · n_')[0]:45s} {nm:22s} max={m.max():.3f} area>{THR}={b.sum():4d}{com}")
        a_.set_title(nm, fontsize=9)
    ax[r, 0].text(-.08, .5, lab.replace(" · n_", "\nn_"), transform=ax[r, 0].transAxes, rotation=90, ha="right", va="center", fontsize=7)
fig.tight_layout(); out = ROOT / "notebooks/out/139_decode_c.pdf"; fig.savefig(out, bbox_inches="tight"); print("wrote", out)
