# nb136 Q1: attention received per band / laser + top-K keep curves (masking vs attention vs random ranking), eager fp32
# ranking half A / evaluation half B of the ablate subset's eval samples, split by position -> runs/_src9_cache/nb136/q1.npz
import sys, numpy as np, torch, torch.nn.functional as F
from pathlib import Path
ROOT = Path("/home/ethantu/workspace/good-vibrations"); sys.path.insert(0, str(ROOT / "src9"))
from dataset import build_dataset
from arch import transformer4 as T4
OUT = ROOT / "runs/_src9_cache/nb136"
torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False

_, ev, _, info = build_dataset("tukey", num_workers=0)
model = T4.Transformer4(info, decoder="res", latent_norm="rescale").cuda().eval()
model.load_state_dict(torch.load(ROOT / "runs/decres-rescale-aug10-v1/checkpoints/ep300-ba3000-rank0.pt", map_location="cpu", weights_only=False)["state"]["model"])
DS = ev[0].dataloader.dataloader.dataset.dataset
B, A = np.load(OUT / "base.npz"), np.load(OUT / "ablate.npz")
sel = A["sel"]; keep = B["split"][sel] != "train"; sel = sel[keep]
box = np.array([s.split("/")[1] for s in B["split"][sel]])
pos = np.array([int(DS.metas[int(i)]["position_id"]) for i in B["i"][sel]])
key = np.unique([f"{b}{p}" for b, p in zip(box, pos)], return_inverse=True)[1]
half = np.random.default_rng(0).integers(0, 2, key.max() + 1)[key]   # 0 = rank, 1 = evaluate; split by position (speakers stay together)
X = torch.stack([DS[int(B["i"][j])]["fft"] for j in sel]); Y = torch.from_numpy(B["y"][sel]).cuda()
print(len(sel), "eval samples, half sizes", np.bincount(half), flush=True)

REC = []  # per block call: (kind, received (N, T)) attention each key token receives, mean over heads and queries
def fwd_attn(self, x, cs, keep):
    N, T, _ = x.shape
    q, k, v = self.qkv(self.n1(x)).view(N, T, 3, T4.HEADS, -1).permute(2, 0, 3, 1, 4)
    q, k = T4.rotate(q, cs), T4.rotate(k, cs)
    a = (q @ k.transpose(-1, -2) / q.shape[-1] ** .5).masked_fill(~keep[:, None, None], float("-inf")).softmax(-1)  # (N,H,T,T)
    REC.append(a.mean((1, 2)))
    x = x + self.o((a @ v).transpose(1, 2).reshape(N, T, -1))
    return x + self.mlp(self.n2(x))
for blk in [*model.encoder.freq_enc.freqs, *model.encoder.laser_enc.lasers]: blk.forward = fwd_attn.__get__(blk)

def fwd(x, m_l, m_b): return model.decoder(model.encoder(x, m_l, m_b)).sigmoid()
def iou(p, y): p = p > .08; return ((p & y).flatten(1).sum(1) / (p | y).flatten(1).sum(1).clamp_min(1))

att_b, att_l, clean = [], [], []
with torch.no_grad():
    for s in range(0, len(X), 50):
        x = X[s:s + 50].cuda(); n = len(x); REC.clear()
        clean.append(iou(fwd(x, x.new_ones(n, 100, dtype=torch.bool), x.new_ones(n, 18, dtype=torch.bool)), Y[s:s + 50]).cpu())
        att_b.append(torch.stack([r.view(n, 100, 18).mean(1) for r in REC[:4]]).mean(0).cpu())   # freq blocks: mean over lasers + 4 layers
        att_l.append(torch.stack(REC[4:]).mean(0).cpu())                                       # laser blocks: mean over 8 layers
att_b, att_l, clean = torch.cat(att_b).numpy(), torch.cat(att_l).numpy(), torch.cat(clean).numpy()
print("clean check vs ablate pass:", clean.mean().round(4), A["iou"][list(A["names"]).index("clean")][keep].mean().round(4), flush=True)

# masking importance on half A (from the ablate pass: same samples, same token-mask baseline)
I = dict(zip(A["names"], A["iou"][:, keep]))
mask_b = np.stack([I["clean"] - I[f"band{k}"] for k in range(18)], 1); mask_l = np.stack([I["clean"] - I[f"laser{l}"] for l in range(100)], 1)
BOXES, KB, KL = ["plastic", "gastro", "cardboard"], np.arange(1, 19), np.array([1, 2, 3, 5, 8, 12, 18, 25, 35, 50, 70, 100])
rng = np.random.default_rng(0)
curves = {}
with torch.no_grad():
    for b in BOXES:
        a, e = (box == b) & (half == 0), np.nonzero((box == b) & (half == 1))[0]
        xe = X[e].cuda(); ye = Y[e]; n = len(e)
        orders = {"masking": [np.argsort(-mask_b[a].mean(0)), np.argsort(-mask_l[a].mean(0))], "attention": [np.argsort(-att_b[a].mean(0)), np.argsort(-att_l[a].mean(0))]}
        orders |= {f"random{r}": [rng.permutation(18), rng.permutation(100)] for r in range(3)}
        for nm, (ob, ol) in orders.items():
            for unit, o, Ks, T in (("band", ob, KB, 18), ("laser", ol, KL, 100)):
                r = []
                for K in Ks:
                    m = torch.zeros(T, dtype=torch.bool, device="cuda"); m[torch.as_tensor(o[:K].copy())] = True; m = m.expand(n, T)
                    full = lambda t: xe.new_ones(n, t, dtype=torch.bool)
                    p = torch.cat([fwd(xe[s:s + 100], *( (full(100)[s:s + 100], m[s:s + 100]) if unit == "band" else (m[s:s + 100], full(18)[s:s + 100]) )) for s in range(0, n, 100)])
                    r.append(iou(p, ye).mean().item())
                curves[f"{b}|{unit}|{nm}"] = np.array(r)
        print(b, {k.split("|", 1)[1]: v[[0, len(v) // 2, -1]].round(3).tolist() for k, v in curves.items() if k.startswith(b) and "random1" not in k and "random2" not in k}, flush=True)
np.savez(OUT / "q1.npz", sel=sel, box=box, half=half, clean=clean, att_b=att_b, att_l=att_l, mask_b=mask_b, mask_l=mask_l, KB=KB, KL=KL,
         **{f"c|{k}": v for k, v in curves.items()})
print("done")
