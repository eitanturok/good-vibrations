# nb136 FQ3: which spectral change carries position. python scripts/nb136_fq3.py train|eval
# Per band, input-space edits on |X| (before the model's per-sample z-score): pattern (laser mean), amp (level -> band mean, shape kept),
# shift -2/-1/+1/+2 bins (edge pad), smooth (9-bin boxcar, edge pad), key-mask. Rel IoU drop per sample. + grad x input of soft Dice at 1170 bins.
# eager fp32 -> runs/_src9_cache/nb136/fq3_{SET}.npz
import sys, numpy as np, torch
from pathlib import Path
ROOT = Path("/home/ethantu/workspace/good-vibrations"); sys.path.insert(0, str(ROOT / "src9"))
from dataset import build_dataset
from arch import transformer4 as T4
OUT, SET = ROOT / "runs/_src9_cache/nb136", sys.argv[1]; assert SET in ("train", "eval")
torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
_, ev, tr_ev, info = build_dataset("tukey", num_workers=0)
DS = {"eval": ev[0].dataloader.dataloader.dataset.dataset, "train": tr_ev.dataloader.dataloader.dataset.dataset}
model = T4.Transformer4(info, decoder="res", latent_norm="rescale").cuda().eval()
model.load_state_dict(torch.load(ROOT / "runs/decres-rescale-aug10-v1/checkpoints/ep300-ba3000-rank0.pt", map_location="cpu", weights_only=False)["state"]["model"])
for p in model.parameters(): p.requires_grad_(False)
B, Q = np.load(OUT / "base.npz"), np.load(OUT / f"q1_{SET}.npz")
split, I, rows, box = B["split"], B["i"], Q["rows"], Q["box"]
load = lambda js: torch.stack([DS["train" if split[j] == "train" else "eval"][int(I[j])]["fft"] for j in js])
X = torch.cat([load(rows[s:s + 120]).cuda() for s in range(0, len(rows), 120)]); Y = torch.from_numpy(B["y"][rows]).cuda()
print(SET, X.shape, flush=True)

def fwd(x, m_b=None): n = len(x); return model.decoder(model.encoder(x, x.new_ones(n, 100, dtype=torch.bool), x.new_ones(n, 18, dtype=torch.bool) if m_b is None else m_b)).sigmoid()
def iou(p, y): p = p > .08; return (p & y).flatten(1).sum(1) / (p | y).flatten(1).sum(1).clamp_min(1)
def run(edit=lambda x: x, m_b=None, bs=80):
    with torch.no_grad(): return torch.cat([iou(fwd(edit(X[s:s + bs]), None if m_b is None else m_b[s:s + bs]), Y[s:s + bs]) for s in range(0, len(X), bs)]).cpu().numpy()

ar = torch.arange(65, device="cuda")
def band(f):  # apply f to bins of band b only: x (N,100,1170,2)
    def g(b):
        def e(x): x = x.clone(); sl = slice(b * 65, (b + 1) * 65); x[:, :, sl] = f(x[:, :, sl]); return x
        return e
    return g
EDITS = {
    "pattern": band(lambda v: v.mean(1, keepdim=True).expand_as(v)),
    "amp": band(lambda v: v * v.mean((1, 2), keepdim=True) / v.mean(2, keepdim=True).clamp_min(1e-8)),
    **{f"shift{k:+d}": band(lambda v, k=k: v[:, :, (ar - k).clamp(0, 64)]) for k in (-2, -1, 1, 2)},
    "smooth": band(lambda v: torch.stack([v[:, :, (ar + d).clamp(0, 64)] for d in range(-4, 5)]).mean(0)),
}
clean = run(); print("clean", {b: clean[box == b].mean().round(3) for b in np.unique(box)}, "vs cache max|Δ|", np.abs(clean - Q["clean"]).max(), flush=True)
R = dict(rows=rows, box=box, clean=clean, names=np.array([*EDITS, "keymask"]))
rel = lambda e: (clean - e) / np.clip(clean, 1e-3, None)
D = np.zeros((len(EDITS) + 1, 18, len(X)), np.float32)
for i, (nm, g) in enumerate(EDITS.items()):
    for b in range(18): D[i, b] = rel(run(g(b)))
    print(nm, {bx: D[i][:, box == bx].mean(1).max().round(3) for bx in np.unique(box)}, flush=True)
for b in range(18): mk = torch.ones(len(X), 18, dtype=torch.bool, device="cuda"); mk[:, b] = False; D[-1, b] = rel(run(m_b=mk))
print("keymask recomputed vs cache max|Δ|", np.abs(D[-1].T - Q["mask_b"]).max(), flush=True)
R["drop"] = D

# grad x input of the soft Dice (IoU proxy) wrt |X|, summed over lasers -> (N,1170,2); also |g x| summed over lasers
GX, AGX = [], []
for s in range(0, len(X), 20):
    x = X[s:s + 20].clone().requires_grad_(True); y = Y[s:s + 20].float(); p = fwd(x)
    dice = 2 * (p * y).flatten(1).sum(1) / (p.flatten(1).sum(1) + y.flatten(1).sum(1))
    g, = torch.autograd.grad(dice.sum(), x); gx = (g * x).detach()
    GX.append(gx.sum(1).cpu().numpy()); AGX.append(gx.abs().sum(1).cpu().numpy())
R["gx"], R["agx"] = np.concatenate(GX), np.concatenate(AGX)
np.savez(OUT / f"fq3_{SET}.npz", **R); print("done", flush=True)
