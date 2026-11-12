"""latents + per-sample task metrics of a transformer4 run, eager fp32, every train/eval sample (same pass as nb137 / nb136_base).
  .venv/bin/python scripts/grl_extract.py --run decres-rescale-aug10-v1                 # baseline
  .venv/bin/python scripts/grl_extract.py --run decres-rescale-aug10-v1 --leace 1      # + LEACE speaker eraser on u (fit on train u), frozen decoder
  .venv/bin/python scripts/grl_extract.py --run grl-adapter-v1                         # adapter run (latent = adapted u)
-> runs/_src9_cache/grl_speaker/<tag>/feats.npz (latent, iou08, miss08, has_obj, speaker_acc head preds if any) + meta.csv"""
import argparse, json, sys, time, numpy as np, pandas as pd, torch
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src9"))
from dataset import build_dataset, load_splits
from arch.transformer4 import Transformer4, SPEAKERS

ap = argparse.ArgumentParser(); ap.add_argument("--run", required=True); ap.add_argument("--leace", type=int, default=0); ap.add_argument("--ckpt", default="latest-rank0.pt")
A = ap.parse_args()
tag = A.run + {0: "", 1: "-leace", 2: "-leacebox"}[A.leace]                         # 1: one eraser for all boxes | 2: one per box (applied with the true box)
OUT = ROOT / "runs/_src9_cache/grl_speaker" / tag; OUT.mkdir(parents=True, exist_ok=True)
torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False

a = json.load(open(ROOT / "runs" / A.run / "args.json"))
_, ev, tr_ev, info = build_dataset(a.get("window", "tukey"), num_workers=0)
kw = {k: a[k] for k in ("decoder", "latent_norm", "freq_rope", "spk_grl_w", "spk_grl_lambda", "spk_grl_ramp", "adapter_dim") if k in a}
model = Transformer4(info, **kw).cuda().eval()
print(model.load_state_dict(torch.load(ROOT / "runs" / A.run / "checkpoints" / A.ckpt, map_location="cpu", weights_only=False)["state"]["model"]))

jobs = [("train", tr_ev.dataloader.dataloader.dataset)] + [(e.label, e.dataloader.dataloader.dataset) for e in ev]
DS, idx = {"train": jobs[0][1].dataset, "eval": jobs[1][1].dataset}, [(n, i) for n, sub in jobs for i in sub.indices]
metas = list(load_splits()[0].values())
spk = np.array([metas[i]["speaker"] for _, i in idx]); is_train = np.array([n == "train" for n, _ in idx])

def batches():
    for s in range(0, len(idx), 128):
        rows = [DS["train" if n == "train" else "eval"][i] for n, i in idx[s:s + 128]]
        yield s, torch.stack([r["fft"] for r in rows]).cuda(), torch.stack([r["mask_true"] for r in rows]).cuda()

# pass 1: latent u (after the adapter if any) and the plain model's masks
t0, U = time.time(), []
with torch.no_grad():
    for s, x, _ in batches():
        U.append(model(dict(fft=x, speaker=torch.ones(len(x), dtype=torch.long, device=x.device)))["latent"].float().cpu())
        if s % 2560 == 0: print(f"  u {s}/{len(idx)} {time.time() - t0:.0f}s", flush=True)
U = torch.cat(U).numpy().astype(np.float64)

box = np.array([metas[i]["box"] for _, i in idx])
for b in (sorted(set(box)) if A.leace == 2 else [None] * bool(A.leace)):
    # LEACE (Belrose et al. 2023): r(x) = x - W+ P W (x - mu), W = Sxx^-1/2, P = proj onto colspan(W Sxz); fit on train latents, train speakers
    m, mb = is_train & np.isin(spk, SPEAKERS) & ((box == b) if b else True), (box == b) if b else np.ones(len(box), bool)
    X, Z = U[m], (spk[m, None] == np.array(SPEAKERS)[None]).astype(np.float64)
    mu = X.mean(0); Xc, Zc = X - mu, Z - Z.mean(0)
    Sxx, Sxz = Xc.T @ Xc / len(X), Xc.T @ Zc / len(X)
    ev_, V = np.linalg.eigh(Sxx); keep = ev_ > ev_.max() * 1e-10
    W = (V[:, keep] / np.sqrt(ev_[keep])) @ V[:, keep].T; Wp = (V[:, keep] * np.sqrt(ev_[keep])) @ V[:, keep].T
    Q, _ = np.linalg.qr(W @ Sxz); P = Q[:, :np.linalg.matrix_rank(W @ Sxz)]; P = P @ P.T
    E = Wp @ P @ W                                                                   # x' = x - (x - mu) E^T
    U[mb] = U[mb] - (U[mb] - mu) @ E.T
    Xe = U[m] - U[m].mean(0); print(f"LEACE {b}: max |cov(x', z)| after", np.abs(Xe.T @ Zc / len(Xe)).max(), "before", np.abs(Sxz).max())
    np.savez(OUT / f"eraser_{b or 'all'}.npz", mu=mu, E=E)

# pass 2: masks from U (decoder on the stored latent) -> per-sample iou08 / miss08
I, M, H, SP = [], [], [], []
Ut = torch.from_numpy(U).float()
with torch.no_grad():
    for s, x, y in batches():
        u = Ut[s:s + len(x)].cuda()
        p = model.decoder(u).sigmoid() > 0.08; o = y > 0.5
        I.append(((p & o).flatten(1).sum(1) / (p | o).flatten(1).sum(1).clamp_min(1)).cpu()); M.append((~(p & o).flatten(1).any(1)).cpu()); H.append(o.flatten(1).any(1).cpu())
        if hasattr(model, "speaker_head"): SP.append(model.speaker_head(u).argmax(-1).cpu())
I, M, H = torch.cat(I).numpy(), torch.cat(M).numpy(), torch.cat(H).numpy()
df = pd.DataFrame(dict(split=[n for n, _ in idx], box=[metas[i]["box"] for _, i in idx], object=["+".join(sorted(metas[i]["objects"])) or "empty" for _, i in idx],
                       speaker=spk, pos=[metas[i]["position_id"] for _, i in idx]))
extra = dict(spk_pred=torch.cat(SP).numpy()) if SP else {}
np.savez(OUT / "feats.npz", latent=U.astype(np.float32), iou08=I, miss08=M, has_obj=H, **extra)
df.to_csv(OUT / "meta.csv", index=False); print(f"saved {OUT} ({time.time() - t0:.0f}s)")
