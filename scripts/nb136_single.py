# nb136 Q1 on SINGLE-OBJECT samples: eval subset (eval splits, n_objects == 1, object present, <= 60 / split, seed 0) and a train
# single-object subset of the same size per box (stratified by object). Per set: clean IoU, masking ΔIoU per band / laser, received attention,
# halves A/B by box+position, top-K keep curves (rank on A, evaluate on B). Eval also gets the train-subset masking ranking ("masking_train").
# eager fp32 -> runs/_src9_cache/nb136/q1_single.npz (eval keys as in q1.npz; train keys prefixed tr_)
import sys, numpy as np, torch
from pathlib import Path
ROOT = Path("/home/ethantu/workspace/good-vibrations"); sys.path.insert(0, str(ROOT / "src9"))
from dataset import build_dataset
from arch import transformer4 as T4
OUT = ROOT / "runs/_src9_cache/nb136"
torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False

_, ev, tr_ev, info = build_dataset("tukey", num_workers=0)
model = T4.Transformer4(info, decoder="res", latent_norm="rescale").cuda().eval()
model.load_state_dict(torch.load(ROOT / "runs/decres-rescale-aug10-v1/checkpoints/ep300-ba3000-rank0.pt", map_location="cpu", weights_only=False)["state"]["model"])
DS = {"eval": ev[0].dataloader.dataloader.dataset.dataset, "train": tr_ev.dataloader.dataloader.dataset.dataset}
B = np.load(OUT / "base.npz"); split, I = B["split"], B["i"]
M = DS["eval"].metas
nobj = np.array([M[int(i)]["n_objects"] for i in I]); present = B["y"].reshape(len(I), -1).any(1)
mbox = np.array([{"gastronorm": "gastro", "green-plastic": "plastic"}.get(M[int(i)]["box"], M[int(i)]["box"]) for i in I]); obj = np.array(["+".join(M[int(i)]["objects"] or ()) for i in I])
assert (nobj == np.array([len(M[int(i)]["objects"] or ()) for i in I])).all()
ok = (nobj == 1) & present
BOXES, KB, KL = ["plastic", "gastro", "cardboard"], np.arange(1, 19), np.array([1, 2, 3, 5, 8, 12, 18, 25, 35, 50, 70, 100])

rng = np.random.default_rng(0); sel_e = []
for s in np.unique(split[split != "train"]):
    k = np.nonzero((split == s) & ok)[0]
    if len(k): sel_e += sorted(rng.choice(k, min(60, len(k)), replace=False).tolist())
sel_e = np.array(sel_e); box_e = np.array([s.split("/")[1] for s in split[sel_e]]); assert (box_e == mbox[sel_e]).all()
rng = np.random.default_rng(0); sel_t = []
for b in BOXES:  # same count as eval per box, equal quota per object (capped), remainder uniform from the leftovers
    n = (box_e == b).sum(); k = np.nonzero((split == "train") & ok & (mbox == b))[0]; objs = np.unique(obj[k]); pick = []
    for o in objs: ko = k[obj[k] == o]; pick += rng.choice(ko, min(n // len(objs), len(ko)), replace=False).tolist()
    rest = np.setdiff1d(k, pick); pick += rng.choice(rest, n - len(pick), replace=False).tolist(); sel_t += sorted(pick)
sel_t = np.array(sel_t); box_t = mbox[sel_t]
print("eval", len(sel_e), dict(zip(*np.unique(box_e, return_counts=True))), "train", len(sel_t), dict(zip(*np.unique(box_t, return_counts=True))), flush=True)

REC = []
def fwd_attn(self, x, cs, keep):
    N, T, _ = x.shape
    q, k, v = self.qkv(self.n1(x)).view(N, T, 3, T4.HEADS, -1).permute(2, 0, 3, 1, 4)
    a = (T4.rotate(q, cs) @ T4.rotate(k, cs).transpose(-1, -2) / q.shape[-1] ** .5).masked_fill(~keep[:, None, None], float("-inf")).softmax(-1)
    REC.append(a.mean((1, 2)))
    x = x + self.o((a @ v).transpose(1, 2).reshape(N, T, -1))
    return x + self.mlp(self.n2(x))
for blk in [*model.encoder.freq_enc.freqs, *model.encoder.laser_enc.lasers]: blk.forward = fwd_attn.__get__(blk)
def fwd(x, m_l=None, m_b=None):
    n = len(x); m_l = x.new_ones(n, 100, dtype=torch.bool) if m_l is None else m_l; m_b = x.new_ones(n, 18, dtype=torch.bool) if m_b is None else m_b
    return model.decoder(model.encoder(x, m_l, m_b)).sigmoid()
def iou(p, y): p = p > .08; return ((p & y).flatten(1).sum(1) / (p | y).flatten(1).sum(1).clamp_min(1))
def keep(n, T, drop): m = torch.ones(n, T, dtype=torch.bool, device="cuda"); m[:, drop] = False; return m

def measure(sel, ds):
    X = torch.cat([torch.stack([ds[int(i)]["fft"] for i in I[sel][s:s + 200]]).cuda() for s in range(0, len(sel), 200)]); Y = torch.from_numpy(B["y"][sel]).cuda()   # on GPU (~2.3 GB)
    clean, mb, ml, ab, al = [], [], [], [], []
    with torch.no_grad():
        for s in range(0, len(X), 50):
            x, y = X[s:s + 50], Y[s:s + 50]; n = len(x); REC.clear()
            c = iou(fwd(x), y); clean.append(c.cpu())
            ab.append(torch.stack([r.view(n, 100, 18).mean(1) for r in REC[:4]]).mean(0).cpu()); al.append(torch.stack(REC[4:]).mean(0).cpu())
            mb.append(torch.stack([c - iou(fwd(x, m_b=keep(n, 18, [k])), y) for k in range(18)], 1).cpu())
            ml.append(torch.stack([c - iou(fwd(x, m_l=keep(n, 100, [k])), y) for k in range(100)], 1).cpu())
    return X, Y, *[torch.cat(v).numpy() for v in (clean, mb, ml, ab, al)]

def curves(X, Y, e, orders):
    xe, ye, n, R = X[e], Y[e], len(e), {}
    with torch.no_grad():
        for nm, (ob, ol) in orders.items():
            for unit, o, Ks, T in (("band", ob, KB, 18), ("laser", ol, KL, 100)):
                r = []
                for K in Ks:
                    m = torch.zeros(n, T, dtype=torch.bool, device="cuda"); m[:, torch.as_tensor(o[:K].copy())] = True
                    p = torch.cat([fwd(xe[s:s + 100], **({"m_b": m[s:s + 100]} if unit == "band" else {"m_l": m[s:s + 100]})) for s in range(0, n, 100)])
                    r.append(iou(p, ye).mean().item())
                R[f"{unit}|{nm}"] = np.array(r)
    return R

S = {}
for tag, sel, bx, ds in (("tr_", sel_t, box_t, DS["train"]), ("", sel_e, box_e, DS["eval"])):
    pos = np.array([int(M[int(i)]["position_id"]) for i in I[sel]])
    key = np.unique([f"{b}{p}" for b, p in zip(bx, pos)], return_inverse=True)[1]
    half = np.random.default_rng(0).integers(0, 2, key.max() + 1)[key]
    X, Y, clean, mb, ml, ab, al = measure(sel, ds)
    print(tag or "eval", "clean", {b: clean[bx == b].mean().round(3) for b in BOXES}, "halves", np.bincount(half), flush=True)
    S |= {tag + k: v for k, v in dict(sel=sel, box=bx, half=half, pos=pos, clean=clean, mask_b=mb, mask_l=ml, att_b=ab, att_l=al).items()}
    rng = np.random.default_rng(0)
    for b in BOXES:
        a, e = (bx == b) & (half == 0), np.nonzero((bx == b) & (half == 1))[0]
        orders = {"masking": [np.argsort(-mb[a].mean(0)), np.argsort(-ml[a].mean(0))], "attention": [np.argsort(-ab[a].mean(0)), np.argsort(-al[a].mean(0))]}
        orders |= {f"random{r}": [rng.permutation(18), rng.permutation(100)] for r in range(3)}
        if not tag: t = S["tr_box"] == b; orders["masking_train"] = [np.argsort(-S["tr_mask_b"][t].mean(0)), np.argsort(-S["tr_mask_l"][t].mean(0))]
        R = curves(X, Y, e, orders); S |= {f"{tag}c|{b}|{k}": v for k, v in R.items()}
        print(tag or "eval", b, {k: v[[0, len(v) // 2, -1]].round(3).tolist() for k, v in R.items() if "random1" not in k and "random2" not in k}, flush=True)
    del X, Y; torch.cuda.empty_cache()
np.savez(OUT / "q1_single.npz", KB=KB, KL=KL, **S)
print("done")
