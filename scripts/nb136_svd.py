# nb136 Q1: SVD of the model's input (src9 preprocessing: |X| / sample std, then the encoder's per-sample z-score), pooled to 65-bin bands
# -> per box: position-carrying PC energy collapsed onto bands / lasers; top-K keep curves with the SVD ranking on q1.npz's half 1
import sys, numpy as np, torch
from pathlib import Path
ROOT = Path("/home/ethantu/workspace/good-vibrations"); sys.path.insert(0, str(ROOT / "src9"))
from dataset import build_dataset
from arch.transformer4 import Transformer4
from sklearn.linear_model import RidgeCV
OUT = ROOT / "runs/_src9_cache/nb136"
torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False

tr_dl, ev, tr_ev, info = build_dataset("tukey", num_workers=0)
TRD, EVD = tr_ev.dataloader.dataloader.dataset.dataset, ev[0].dataloader.dataloader.dataset.dataset
B, Q = np.load(OUT / "base.npz"), np.load(OUT / "q1.npz")
BOX = {"2026_09_27_gastronorm_four_objs": "gastro", "2026_10_05_green_plastic_four_objs": "plastic", "2026_10_11_cardboard_four_objs": "cardboard"}
def feat(x):  # (n,100,1170,2) -> (n,100,18,2): what the encoder sees (per-sample z-score), mean over each band's 65 bins
    x = (x - x.mean((1, 2, 3), keepdim=True)) / x.std((1, 2, 3), keepdim=True)
    return x.view(len(x), 100, 18, 65, 2).mean(3)

tr = np.nonzero(B["split"] == "train")[0]
Fs, R, C, bx, n1 = [], [], [], [], []
for s in range(0, len(tr), 256):
    rows = [TRD[int(B["i"][j])] for j in tr[s:s + 256]]
    Fs.append(feat(torch.stack([r["fft"] for r in rows]).cuda()).cpu().numpy())
    for r, j in zip(rows, tr[s:s + 256]):
        y = B["y"][j]; a = max(y.sum(), 1); rr, cc = np.nonzero(y)
        R.append(rr.mean() if len(rr) else np.nan); C.append(cc.mean() if len(cc) else np.nan); bx.append(BOX[r["experiment"]]); n1.append(r["n_objects"])
Fs, R, C, bx, n1 = np.concatenate(Fs), np.array(R), np.array(C), np.array(bx), np.array(n1)
print("features", Fs.shape, f"{Fs.nbytes / 1e9:.2f} GB", flush=True)

K, imp_b, imp_l, scree = 64, {}, {}, {}
for b in ("plastic", "gastro", "cardboard"):
    m = (bx == b) & (n1 == 1)
    X = Fs[m].reshape(m.sum(), -1); X = X - X.mean(0)
    _, sv, Vt = np.linalg.svd(X, full_matrices=False)
    Z = X @ Vt[:K].T                                                            # PC scores (n,K)
    w = RidgeCV(alphas=np.logspace(-2, 4, 7)).fit(Z, np.stack([R[m], C[m]], 1)).coef_  # (2,K)
    rk = (w ** 2 * sv[:K] ** 2).sum(0); rk /= rk.sum()                          # share of the (linear) position prediction carried by PC k
    E = (rk[:, None] * Vt[:K] ** 2).reshape(K, 100, 18, 2).sum(0)               # position-weighted loading energy (100,18,2); Σ_k r_k |v_k|²
    imp_b[b], imp_l[b], scree[b] = E.sum((0, 2)), E.sum((1, 2)), sv ** 2 / (sv ** 2).sum()
    print(b, "top bands", np.argsort(-imp_b[b])[:5], "pos share in PCs 1-8", rk[:8].sum().round(3), flush=True)

# top-K keep curves with the SVD ranking (same eval half and K grid as q1.npz)
model = Transformer4(info, decoder="res", latent_norm="rescale").cuda().eval()
model.load_state_dict(torch.load(ROOT / "runs/decres-rescale-aug10-v1/checkpoints/ep300-ba3000-rank0.pt", map_location="cpu", weights_only=False)["state"]["model"])
sel, box, half = Q["sel"], Q["box"], Q["half"]
curves = {}
with torch.no_grad():
    for b in ("plastic", "gastro", "cardboard"):
        e = sel[(box == b) & (half == 1)]
        xe = torch.stack([EVD[int(B["i"][j])]["fft"] for j in e]).cuda(); ye = torch.from_numpy(B["y"][e]).cuda(); n = len(e)
        for unit, o, Ks, T in (("band", np.argsort(-imp_b[b]), Q["KB"], 18), ("laser", np.argsort(-imp_l[b]), Q["KL"], 100)):
            r = []
            for k in Ks:
                m = torch.zeros(n, T, dtype=torch.bool, device="cuda"); m[:, torch.as_tensor(o[:k].copy())] = True
                one = lambda t: torch.ones(n, t, dtype=torch.bool, device="cuda")
                ml, mb = (one(100), m) if unit == "band" else (m, one(18))
                p = torch.cat([model.decoder(model.encoder(xe[s:s + 100], ml[s:s + 100], mb[s:s + 100])).sigmoid() for s in range(0, n, 100)]) > .08
                r.append(((p & ye).flatten(1).sum(1) / (p | ye).flatten(1).sum(1).clamp_min(1)).mean().item())
            curves[f"c|{b}|{unit}|svd"] = np.array(r)
        print(b, {k: v.round(3).tolist() for k, v in curves.items() if f"|{b}|" in k}, flush=True)
np.savez(OUT / "q1_svd.npz", **{f"imp_b|{b}": v for b, v in imp_b.items()}, **{f"imp_l|{b}": v for b, v in imp_l.items()}, **{f"scree|{b}": v for b, v in scree.items()}, **curves)
print("done")
