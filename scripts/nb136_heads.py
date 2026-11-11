# nb136 Q1c: per-(layer, head) attention (received per band / laser, band->band maps, row entropy), whole-head ablations,
# and band top-K with the head that best agrees with masking (chosen on half A, evaluated on half B). Eager fp32 -> runs/_src9_cache/nb136/heads.npz
import sys, numpy as np, torch
from pathlib import Path
from scipy import stats
ROOT = Path("/home/ethantu/workspace/good-vibrations"); sys.path.insert(0, str(ROOT / "src9"))
from dataset import build_dataset
from arch import transformer4 as T4
OUT = ROOT / "runs/_src9_cache/nb136"
SUF = sys.argv[1] if len(sys.argv) > 1 else ""   # "_single" -> read q1_single.npz, write *_single.npz
torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
_, ev, _, info = build_dataset("tukey", num_workers=0)
model = T4.Transformer4(info, decoder="res", latent_norm="rescale").cuda().eval()
model.load_state_dict(torch.load(ROOT / "runs/decres-rescale-aug10-v1/checkpoints/ep300-ba3000-rank0.pt", map_location="cpu", weights_only=False)["state"]["model"])
DS = ev[0].dataloader.dataloader.dataset.dataset
B, Q = np.load(OUT / "base.npz"), np.load(OUT / f"q1{SUF}.npz")
sel, box, half = Q["sel"], Q["box"], Q["half"]
X = torch.cat([torch.stack([DS[int(B["i"][j])]["fft"] for j in sel[s:s + 200]]).cuda() for s in range(0, len(sel), 200)]); Y = torch.from_numpy(B["y"][sel]).cuda()  # X chunked to GPU: CPU peak stays small
H, BX = T4.HEADS, ["plastic", "gastro", "cardboard"]
BLKS = [*model.encoder.freq_enc.freqs, *model.encoder.laser_enc.lasers]       # 0-3 freq, 4-11 laser
REC, KILL = [], {}                                                            # KILL = {block index: head to zero}
def fwd_attn(self, x, cs, keep, i=0):
    N, T, _ = x.shape
    q, k, v = self.qkv(self.n1(x)).view(N, T, 3, H, -1).permute(2, 0, 3, 1, 4)
    a = (T4.rotate(q, cs) @ T4.rotate(k, cs).transpose(-1, -2) / q.shape[-1] ** .5).masked_fill(~keep[:, None, None], float("-inf")).softmax(-1)  # (N,H,T,T)
    if REC is not None: REC.append(a)
    o = a @ v
    if i in KILL: o[:, KILL[i]] = 0
    x = x + self.o(o.transpose(1, 2).reshape(N, T, -1))
    return x + self.mlp(self.n2(x))
for i, blk in enumerate(BLKS): blk.forward = (lambda i: lambda x, cs, keep: fwd_attn(BLKS[i], x, cs, keep, i))(i)
def fwd(x, m_l=None, m_b=None):
    n = len(x); m_l = x.new_ones(n, 100, dtype=torch.bool) if m_l is None else m_l; m_b = x.new_ones(n, 18, dtype=torch.bool) if m_b is None else m_b
    return model.decoder(model.encoder(x, m_l, m_b)).sigmoid()
def iou(p, y): p = p > .08; return ((p & y).flatten(1).sum(1) / (p | y).flatten(1).sum(1).clamp_min(1))
ent = lambda a: -(a * a.clamp_min(1e-12).log()).sum(-1) / np.log(a.shape[-1])  # row entropy / log T (1 = uniform)

rb, rl, eb, el, fmap = [], [], [], [], np.zeros((3, 4, H, 18, 18))
with torch.no_grad():
    for s in range(0, len(X), 25):
        x = X[s:s + 25].cuda(); n = len(x); REC.clear(); fwd(x)
        F_ = torch.stack(REC[:4], 1).view(n, 100, 4, H, 18, 18)               # (n,L,layer,H,q,k)
        rb.append(F_.mean((1, 4)).cpu()); eb.append(ent(F_).mean((1, 4)).cpu())  # received per band (n,4,H,18); entropy (n,4,H)
        L_ = torch.stack(REC[4:], 1)                                           # (n,8,H,100,100)
        rl.append(L_.mean(3).half().cpu()); el.append(ent(L_).mean(3).cpu())
        fm = F_.mean(1).cpu().numpy()
        for j, b in enumerate(BX): fmap[j] += fm[box[s:s + 25] == b].sum(0)
rb, eb, rl, el = torch.cat(rb).numpy(), torch.cat(eb).numpy(), torch.cat(rl).numpy(), torch.cat(el).numpy()
fmap /= np.array([(box == b).sum() for b in BX])[:, None, None, None, None]
REC = None
print("attention done", flush=True)

clean, abl = [], np.zeros((12, H, len(X)), np.float32)
with torch.no_grad():
    clean = torch.cat([iou(fwd(X[s:s + 100].cuda()), Y[s:s + 100]) for s in range(0, len(X), 100)]).cpu().numpy()
    for i in range(12):
        for h in range(H):
            KILL = {i: h}; abl[i, h] = torch.cat([iou(fwd(X[s:s + 100].cuda()), Y[s:s + 100]) for s in range(0, len(X), 100)]).cpu().numpy()
    KILL = {}
print("ablation done, clean", clean.mean().round(4), flush=True)

# best head per box: ρ(head received on half A, masking on half A) -> top-K on half B
mask_b = Q["mask_b"]; best, curves = {}, {}
with torch.no_grad():
    for b in BX:
        a, e = (box == b) & (half == 0), np.nonzero((box == b) & (half == 1))[0]
        r = np.array([[stats.spearmanr(rb[a, l, h].mean(0), mask_b[a].mean(0))[0] for h in range(H)] for l in range(4)])
        l, h = np.unravel_index(np.nanargmax(r), r.shape); best[b] = (l, h, r[l, h])
        o = np.argsort(-rb[a, l, h].mean(0)); xe, ye, n = X[e].cuda(), Y[e], len(e)
        res = []
        for k in Q["KB"]:
            m = torch.zeros(n, 18, dtype=torch.bool, device="cuda"); m[:, torch.as_tensor(o[:k].copy())] = True
            res.append(torch.cat([iou(fwd(xe[s:s + 100], m_b=m[s:s + 100]), ye[s:s + 100]) for s in range(0, n, 100)]).mean().item())
        curves[b] = np.array(res); print(b, "best head", best[b], np.round(res, 3).tolist(), flush=True)
np.savez(OUT / f"heads{SUF}.npz", rb=rb, eb=eb, rl=rl, el=el, fmap=fmap, clean=clean, abl=abl, best=np.array([best[b] for b in BX]), **{f"c|{b}": v for b, v in curves.items()})
print("done")
