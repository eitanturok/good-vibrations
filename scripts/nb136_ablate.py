# pass B: perturbations on a stratified eval subset (<= 30 / split, objects present) -> iou08, area, com per (variant, sample)
import sys, time, numpy as np, torch
from pathlib import Path
ROOT = Path("/home/ethantu/workspace/good-vibrations"); sys.path.insert(0, str(ROOT / "src9"))
from dataset import build_dataset
from arch.transformer4 import Transformer4
from utils import speaker_gain, laser_jitter
OUT = ROOT / "runs/_src9_cache/nb136"
torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False

tr_dl, ev, tr_ev, info = build_dataset("tukey", num_workers=0)
model = Transformer4(info, decoder="res", latent_norm="rescale").cuda().eval()
model.load_state_dict(torch.load(ROOT / "runs/decres-rescale-aug10-v1/checkpoints/ep300-ba3000-rank0.pt", map_location="cpu", weights_only=False)["state"]["model"])
DS = {"train": tr_ev.dataloader.dataloader.dataset.dataset, "eval": ev[0].dataloader.dataloader.dataset.dataset}
B = np.load(OUT / "base.npz")
rng = np.random.default_rng(0)
sel = []
for s in np.unique(B["split"]):
    k = np.nonzero((B["split"] == s) & B["y"].reshape(len(B["y"]), -1).any(1))[0]
    sel += sorted(rng.choice(k, min(30 if s != "train" else 300, len(k)), replace=False).tolist())
sel = np.array(sel); print(len(sel), "samples")
X = torch.stack([DS["train" if B["split"][j] == "train" else "eval"][int(B["i"][j])]["fft"] for j in sel])                    # (N,L,F,C) cpu, ~1.4 GB at 1500
Y = torch.from_numpy(B["y"][sel]).cuda()
U_tr = torch.from_numpy(B["u"][B["split"] == "train"]).float().cuda()
mu = U_tr.mean(0); _, _, Vt = torch.linalg.svd(U_tr - mu, full_matrices=False)  # latent PCs (train)

def run(fn, bs=100):
    """fn(x (b,L,F,C) cuda, g torch.Generator) -> model outputs; returns iou08, area px, com (row,col) px"""
    I, A, C = [], [], []
    g = torch.Generator(device="cuda").manual_seed(0); torch.manual_seed(0)
    with torch.no_grad():
        for s in range(0, len(X), bs):
            p = fn(X[s:s + bs].cuda()) > 0.08; y = Y[s:s + bs]
            I.append(((p & y).flatten(1).sum(1) / (p | y).flatten(1).sum(1).clamp_min(1)).cpu())
            a = p.flatten(1).sum(1).float(); A.append(a.cpu())
            r = torch.arange(64, device="cuda").float()
            C.append(torch.stack([(p.sum(2) * r).sum(1), (p.sum(1) * r).sum(1)], 1).div(a.clamp_min(1)[:, None]).cpu())
    return torch.cat(I).numpy(), torch.cat(A).numpy(), torch.cat(C).numpy()

def fwd(x, m_l=None, m_b=None):
    b = len(x); m_l = x.new_ones(b, 100, dtype=torch.bool) if m_l is None else m_l; m_b = x.new_ones(b, 18, dtype=torch.bool) if m_b is None else m_b
    return model.decoder(model.encoder(x, m_l, m_b)).sigmoid()

def lmask(b, drop): m = torch.ones(b, 100, dtype=torch.bool, device="cuda"); m[:, drop] = False; return m
def bmask(b, drop): m = torch.ones(b, 18, dtype=torch.bool, device="cuda"); m[:, drop] = False; return m
def rand_keep(b, n, k): return torch.rand(b, n, device="cuda").argsort(1) >= k   # drop k random tokens per sample

V = {"clean": lambda x: fwd(x)}
for b in range(18): V[f"band{b}"] = lambda x, b=b: fwd(x, m_b=bmask(len(x), [b]))
for l in range(100): V[f"laser{l}"] = lambda x, l=l: fwd(x, m_l=lmask(len(x), [l]))
for r in range(10): V[f"lrow{r}"] = lambda x, r=r: fwd(x, m_l=lmask(len(x), list(range(10 * r, 10 * r + 10))))
for c in range(10): V[f"lcol{c}"] = lambda x, c=c: fwd(x, m_l=lmask(len(x), list(range(c, 100, 10))))
for k in (10, 25, 50, 75, 90, 95, 99): V[f"rl{k}"] = lambda x, k=k: fwd(x, m_l=rand_keep(len(x), 100, k))
for k in (2, 4, 6, 9, 12, 15, 17): V[f"rb{k}"] = lambda x, k=k: fwd(x, m_b=rand_keep(len(x), 18, k))
for sd in (0.2, 0.35, 0.7, 1.0, 1.5): V[f"sg{sd}"] = lambda x, sd=sd: fwd(speaker_gain(x, info["freq_range"], sd, 20.0))
for hz in (5.0, 80.0): V[f"sgHz{hz}"] = lambda x, hz=hz: fwd(speaker_gain(x, info["freq_range"], 0.7, hz))
for s in (0.5, 1.0, 2.0): V[f"jit{s}"] = lambda x, s=s: fwd(laser_jitter(x, 10, 10, s))
for snr in (20, 10, 5, 0, -5):  # additive complex-gaussian noise on |X| at the given per-sample SNR (dB re the sample's rms)
    V[f"snr{snr}"] = lambda x, snr=snr: fwd((x + torch.randn_like(x) * x.pow(2).mean((1, 2, 3), keepdim=True).sqrt() * 10 ** (-snr / 20)).abs())
V["gain_global"] = lambda x: fwd(x * 3.0)                                        # sanity: per-sample z-score -> invariant
V["flip_lr"] = lambda x: fwd(x.view(len(x), 10, 10, *x.shape[2:]).flip(2).reshape(x.shape))   # mirror the laser grid's columns
V["flip_ud"] = lambda x: fwd(x.view(len(x), 10, 10, *x.shape[2:]).flip(1).reshape(x.shape))
V["shuffle_bins"] = lambda x: fwd(x[:, :, torch.randperm(x.shape[2], device="cuda")])
V["smooth_spec"] = lambda x: fwd(torch.nn.functional.avg_pool1d(x.permute(0, 1, 3, 2).flatten(0, 1), 33, 1, 16, count_include_pad=False).view(len(x), 100, 2, -1).permute(0, 1, 3, 2))  # 33-bin (~?Hz) moving average: kills sharp modes
V["laser_mean"] = lambda x: fwd(x.mean(1, keepdim=True).expand_as(x).contiguous())  # every laser = the wall average: no spatial info
V["chan0"] = lambda x: fwd(torch.stack([x[..., 0], x[..., 0]], -1)); V["chan1"] = lambda x: fwd(torch.stack([x[..., 1], x[..., 1]], -1))
def pcs(x, k):
    u = model.encoder(x, x.new_ones(len(x), 100, dtype=torch.bool), x.new_ones(len(x), 18, dtype=torch.bool))
    return model.decoder(mu + ((u - mu) @ Vt[:k].T) @ Vt[:k]).sigmoid()
for k in (1, 2, 3, 4, 6, 8, 12, 16, 32, 64, 128, 512): V[f"pc{k}"] = lambda x, k=k: pcs(x, k)

R = {}; t0 = time.time()
for n, fn in V.items():
    R[n] = run(fn); print(n, f"{R[n][0].mean():.3f}", f"{time.time() - t0:.0f}s", flush=True)
np.savez(OUT / "ablate.npz", sel=sel, names=np.array(list(R)), iou=np.stack([v[0] for v in R.values()]), area=np.stack([v[1] for v in R.values()]), com=np.stack([v[2] for v in R.values()]))
print("done")
