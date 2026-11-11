# nb136 FQ2: redundancy vs synergy of the 18 bands. python scripts/nb136_fq2.py train|eval [n_perm]
# key-masking: singles + all 153 pairs per sample -> I(i,j) = Δij - Δi - Δj (rel IoU drop). Shapley per box from random permutations
# (one permutation shared by the box's 240 samples), v(S) = mean IoU with only S kept; v(∅) = no band kept (freq readout = its bias, as the full model gives).
# top-K keep curves with the Shapley ranking. eager fp32 -> runs/_src9_cache/nb136/fq2_{SET}.npz
import sys, numpy as np, torch
from pathlib import Path
ROOT = Path("/home/ethantu/workspace/good-vibrations"); sys.path.insert(0, str(ROOT / "src9"))
from dataset import build_dataset
from arch import transformer4 as T4
OUT, SET = ROOT / "runs/_src9_cache/nb136", sys.argv[1]; assert SET in ("train", "eval")
NP = int(sys.argv[2]) if len(sys.argv) > 2 else 200
torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
_, ev, tr_ev, info = build_dataset("tukey", num_workers=0)
DS = {"eval": ev[0].dataloader.dataloader.dataset.dataset, "train": tr_ev.dataloader.dataloader.dataset.dataset}
model = T4.Transformer4(info, decoder="res", latent_norm="rescale").cuda().eval()
model.load_state_dict(torch.load(ROOT / "runs/decres-rescale-aug10-v1/checkpoints/ep300-ba3000-rank0.pt", map_location="cpu", weights_only=False)["state"]["model"])
B, Q = np.load(OUT / "base.npz"), np.load(OUT / f"q1_{SET}.npz")
split, I, rows, box = B["split"], B["i"], Q["rows"], Q["box"]
load = lambda js: torch.stack([DS["train" if split[j] == "train" else "eval"][int(I[j])]["fft"] for j in js])
X = torch.cat([load(rows[s:s + 120]).cuda() for s in range(0, len(rows), 120)]); Y = torch.from_numpy(B["y"][rows]).cuda()
E, FE = model.encoder, model.encoder.freq_enc; fr, LE = FE.read, model.encoder.laser_enc; Wf = fr.lin.weight.view(fr.lin.weight.shape[0], 18, -1)
ON = torch.ones(1, 100, dtype=torch.bool, device="cuda")
def fwd(x, kept):  # = model with bands outside `kept` key-masked (same for every sample), run on the kept band tokens only (exact; ~2x faster). kept = [] -> freq readout = bias
    n, k = len(x), len(kept); x = (x - x.mean((1, 2, 3), keepdim=True)) / (x.std((1, 2, 3), keepdim=True) + 1e-6)
    if k == 0: f = fr.lin.bias.expand(n, 100, -1)
    else:
        h = E.embed(x.reshape(n, 100, 18, 65, -1)[:, :, kept].flatten(-2)).flatten(0, 1); cs = FE.rope[:, kept]
        for blk in FE.freqs: h = blk(h, cs, ON[:, :k].expand(len(h), k))
        f = torch.einsum("ond,bnd->bo", Wf[:, kept], fr.norm(h)).view(n, 100, -1) + fr.lin.bias   # freq readout has no rescale
    return model.decoder(LE(f, ON.expand(n, 100))).sigmoid()
def iou(p, y): p = p > .08; return (p & y).flatten(1).sum(1) / (p | y).flatten(1).sum(1).clamp_min(1)
def run(xs, ys, kept, bs=120):
    kept = sorted(int(i) for i in kept)
    with torch.no_grad(): return torch.cat([iou(fwd(xs[s:s + bs], kept), ys[s:s + bs]) for s in range(0, len(xs), bs)])
keep = lambda drop: [i for i in range(18) if i not in set(int(d) for d in drop)]
with torch.no_grad():  # exactness check vs the model's own key-masking
    m = torch.ones(40, 18, dtype=torch.bool, device="cuda"); m[:, [2, 7, 11]] = False
    ref = model.decoder(E(X[:40], ON.expand(40, 100), m)).sigmoid(); print("gather vs keymask max|Δp|", (fwd(X[:40], keep([2, 7, 11])) - ref).abs().max().item(), flush=True)

N = len(X); clean = run(X, Y, range(18)).cpu().numpy(); rel = lambda e: (clean - e) / np.clip(clean, 1e-3, None)
print(SET, "clean vs cache max|Δ|", np.abs(clean - Q["clean"]).max(), flush=True)
single = np.stack([rel(run(X, Y, keep([i])).cpu().numpy()) for i in range(18)], 1)       # (N,18)
pair = np.full((N, 18, 18), np.nan, np.float32)
for i in range(18):
    for j in range(i + 1, 18): pair[:, i, j] = pair[:, j, i] = rel(run(X, Y, keep([i, j])).cpu().numpy())
    print("pairs", i, flush=True)
R = dict(rows=rows, box=box, clean=clean, single=single, pair=pair)

# Shapley per box: v(S) = mean IoU over the box's samples; phi_i = mean over permutations of v(S + i) - v(S)
for b in np.unique(box):
    e = torch.as_tensor(np.nonzero(box == b)[0], device="cuda"); xe, ye, n = X[e], Y[e], len(e); rng = np.random.default_rng(0)
    v0 = run(xe, ye, []).mean().item(); phi, perms, V = np.zeros(18), [], []
    for t in range(NP):
        o = rng.permutation(18) if t % 2 == 0 else perms[-1][::-1]     # antithetic pairs
        v, vs = v0, []
        for t_, i in enumerate(o): w = run(xe, ye, o[:t_ + 1]).mean().item(); phi[i] += w - v; v = w; vs.append(w)
        perms.append(o); V.append(vs)
        if t % 20 == 19: print(b, t + 1, (phi / (t + 1) / clean[box == b].mean()).round(3), flush=True)
    R[f"phi|{b}"], R[f"perm|{b}"], R[f"V|{b}"], R[f"v0|{b}"] = phi / NP, np.array(perms), np.array(V), v0
    # top-K keep curve with the Shapley ranking (IoU_K mean)
    o = np.argsort(-R[f"phi|{b}"]); R[f"c|{b}|shapley"] = np.array([run(xe, ye, o[:k]).mean().item() for k in range(1, 19)])
np.savez(OUT / f"fq2_{SET}.npz", **R); print("done", flush=True)
