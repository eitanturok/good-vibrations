# nb136 Q1 / Q1b / Q1c on ONE fixed sample set: python scripts/nb136_set.py train|eval
# subset720.npz: 60 single-object samples of each ID object (candle, red-cube, soap dispenser, vase) x box, per set, drawn (seed 0) from the rows
# q1_single.npz already evaluated. Per-sample masking / attention (and for eval: merge W·act, heads) are INDEXED from the old caches;
# recomputed here: train merge W·act + heads (not cached for train), the fmap / SVD fit / top-K curves (aggregates), the input-signal views.
# eager fp32 -> runs/_src9_cache/nb136/q1_{SET}.npz, heads_{SET}.npz
import sys, numpy as np, torch
from pathlib import Path
from scipy import stats
from sklearn.linear_model import RidgeCV
ROOT = Path("/home/ethantu/workspace/good-vibrations"); sys.path.insert(0, str(ROOT / "src9"))
from dataset import build_dataset
from arch import transformer4 as T4
OUT, SET = ROOT / "runs/_src9_cache/nb136", sys.argv[1]; assert SET in ("train", "eval")
torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
_, ev, tr_ev, info = build_dataset("tukey", num_workers=0)
DS = {"eval": ev[0].dataloader.dataloader.dataset.dataset, "train": tr_ev.dataloader.dataloader.dataset.dataset}
model = T4.Transformer4(info, decoder="res", latent_norm="rescale").cuda().eval()
model.load_state_dict(torch.load(ROOT / "runs/decres-rescale-aug10-v1/checkpoints/ep300-ba3000-rank0.pt", map_location="cpu", weights_only=False)["state"]["model"])
B, Q, MG, HS = [np.load(OUT / f) for f in ("base.npz", "q1_single.npz", "q1_merge_single.npz", "heads_single.npz")]
split, I, META = B["split"], B["i"], DS["eval"].metas
BX, OBJ, H = ["plastic", "gastro", "cardboard"], ["candle", "red-cube", "soap dispenser", "vase"], T4.HEADS
KB, KL = Q["KB"], Q["KL"]
boxof = lambda j: {"gastronorm": "gastro", "green-plastic": "plastic"}.get(META[int(I[j])]["box"], META[int(I[j])]["box"])
objof = lambda j: "+".join(META[int(I[j])]["objects"] or ())

if not (OUT / "subset720.npz").exists():
    S = {}
    for s, p in (("train", "tr_"), ("eval", "")):
        rows, ob = Q[p + "sel"], np.array([objof(j) for j in Q[p + "sel"]]); rng = np.random.default_rng(0); q = []
        for b in BX:
            for o in OBJ: k = np.nonzero((Q[p + "box"] == b) & (ob == o))[0]; q += sorted(rng.choice(k, 60, replace=False).tolist())
        S[s + "_q"], S[s] = np.array(q), rows[q]
    np.savez(OUT / "subset720.npz", **S)
SU = np.load(OUT / "subset720.npz"); q, rows = SU[SET + "_q"], SU[SET]; p = "tr_" if SET == "train" else ""
box, obj = Q[p + "box"][q], np.array([objof(j) for j in rows])
meta = [META[int(I[j])] for j in rows]; spk, pos = np.array([m["speaker"] for m in meta]), np.array([int(m["position_id"]) for m in meta])
y = B["y"][rows]; rr, cc = np.mgrid[:64, :64]; com = np.stack([(y * rr).sum((1, 2)), (y * cc).sum((1, 2))], 1) / y.sum((1, 2))[:, None]
R = dict(rows=rows, box=box, obj=obj, spk=spk, pos=pos, com=com, KB=KB, KL=KL, **{k: Q[p + k][q] for k in ("clean", "mask_b", "mask_l", "att_b", "att_l")})
print(SET, len(rows), {b: (box == b).sum() for b in BX}, "clean", {b: R["clean"][box == b].mean().round(3) for b in BX}, flush=True)
load = lambda js: torch.stack([DS["train" if split[j] == "train" else "eval"][int(I[j])]["fft"] for j in js])
X = torch.cat([load(rows[s:s + 120]).cuda() for s in range(0, len(rows), 120)]); Y = torch.from_numpy(y).cuda()

# hooked forward: per-head attention record, head kill, merge-block slot contributions ||W_slot LN(h_slot)||
REC, CON, KILL, ON = [], {}, {}, [False]
BLKS = [*model.encoder.freq_enc.freqs, *model.encoder.laser_enc.lasers]
def attn(i, x, cs, keep):
    blk, (N, T, _) = BLKS[i], x.shape
    qq, k, v = blk.qkv(blk.n1(x)).view(N, T, 3, H, -1).permute(2, 0, 3, 1, 4)
    a = (T4.rotate(qq, cs) @ T4.rotate(k, cs).transpose(-1, -2) / qq.shape[-1] ** .5).masked_fill(~keep[:, None, None], float("-inf")).softmax(-1)
    if ON[0]: REC.append(a)
    o = a @ v
    if i in KILL: o[:, KILL[i]] = 0
    x = x + blk.o(o.transpose(1, 2).reshape(N, T, -1))
    return x + blk.mlp(blk.n2(x))
for i, blk in enumerate(BLKS): blk.forward = (lambda i: lambda x, cs, keep: attn(i, x, cs, keep))(i)
def merge(self, x, m, name):
    h = self.norm(x).masked_fill(~m[..., None], 0)
    if ON[0]: CON[name] = torch.einsum("ond,...nd->...no", self.lin.weight.view(self.lin.weight.shape[0], h.shape[-2], -1), h).norm(dim=-1)
    s = torch.nn.functional.linear(h.flatten(-2), self.lin.weight)
    if self.rescale: s = s * (m.shape[-1] / m.sum(-1, keepdim=True)).to(s.dtype)
    return s + self.lin.bias
fr, lr = model.encoder.freq_enc.read, model.encoder.laser_enc.read
fr.forward = lambda x, m: merge(fr, x, m, "band"); lr.forward = lambda x, m: merge(lr, x, m, "laser")
def fwd(x, m_l=None, m_b=None):
    n = len(x); m_l = x.new_ones(n, 100, dtype=torch.bool) if m_l is None else m_l; m_b = x.new_ones(n, 18, dtype=torch.bool) if m_b is None else m_b
    return model.decoder(model.encoder(x, m_l, m_b)).sigmoid()
def iou(p, y): p = p > .08; return (p & y).flatten(1).sum(1) / (p | y).flatten(1).sum(1).clamp_min(1)
ent = lambda a: -(a * a.clamp_min(1e-12).log()).sum(-1) / np.log(a.shape[-1])
allx = lambda **kw: torch.cat([iou(fwd(X[s:s + 100], **{k: v[s:s + 100] for k, v in kw.items()}), Y[s:s + 100]) for s in range(0, len(X), 100)])

# pass 1: clean check, per-head attention, merge contributions, fmap
rb, eb, rl, el, cb, cl, clean, fmap = [], [], [], [], [], [], [], np.zeros((3, 4, H, 18, 18))
with torch.no_grad():
    for s in range(0, len(X), 25):
        x = X[s:s + 25]; n = len(x); REC.clear(); ON[0] = True; clean.append(iou(fwd(x), Y[s:s + 25]).cpu()); ON[0] = False
        F_ = torch.stack(REC[:4], 1).view(n, 100, 4, H, 18, 18); L_ = torch.stack(REC[4:], 1)
        rb.append(F_.mean((1, 4)).cpu()); eb.append(ent(F_).mean((1, 4)).cpu()); rl.append(L_.mean(3).half().cpu()); el.append(ent(L_).mean(3).cpu())
        cb.append(CON["band"].mean(1).cpu()); cl.append(CON["laser"].cpu())
        fm = F_.mean(1).cpu().numpy()
        for j, b in enumerate(BX): fmap[j] += fm[box[s:s + 25] == b].sum(0)
rb, eb, rl, el, cb, cl, clean = [torch.cat(v).numpy() for v in (rb, eb, rl, el, cb, cl, clean)]
fmap /= np.array([(box == b).sum() for b in BX])[:, None, None, None, None]
print("clean recomputed vs cached: max |Δ|", np.abs(clean - R["clean"]).max(), "att_b", np.abs(rb.mean((1, 2)) - R["att_b"]).max(), flush=True)
if SET == "eval":   # per-sample arrays exist for eval: index them (the pass above is only a consistency check + fmap)
    print("eval cache vs recomputed: rb", np.abs(HS["rb"][q] - rb).max(), "c_b", np.abs(MG["c_b"][q] - cb).max(), flush=True)
    rb, eb, rl, el, cb, cl = HS["rb"][q], HS["eb"][q], HS["rl"][q], HS["el"][q], MG["c_b"][q], MG["c_l"][q]
    abl = HS["abl"][:, :, q]
else:
    abl = np.zeros((12, H, len(X)), np.float32)
    with torch.no_grad():
        for i in range(12):
            for h in range(H): KILL = {i: h}; abl[i, h] = allx().cpu().numpy()
    KILL = {}
print("heads done", flush=True)
R |= dict(c_b=cb, c_l=cl, w_b=MG["w_b"], w_l=MG["w_l"])

# SVD of the encoder input on this set: per box PCA of z-scored (100,18,2) band means, ridge PCs -> CoM, position-weighted loading energy
Fz = ((X - X.mean((1, 2, 3), keepdim=True)) / X.std((1, 2, 3), keepdim=True)).view(len(X), 100, 18, 65, 2).mean(3).cpu().numpy()
for b in BX:
    m = box == b; Z = Fz[m].reshape(m.sum(), -1); Z = Z - Z.mean(0)
    _, sv, Vt = np.linalg.svd(Z, full_matrices=False); K = 64
    w = RidgeCV(alphas=np.logspace(-2, 4, 7)).fit(Z @ Vt[:K].T, com[m]).coef_
    rk = (w ** 2 * sv[:K] ** 2).sum(0); rk /= rk.sum(); E = (rk[:, None] * Vt[:K] ** 2).reshape(K, 100, 18, 2).sum(0)
    R[f"imp_b|{b}"], R[f"imp_l|{b}"] = E.sum((0, 2)), E.sum((1, 2))

# top-K keep curves, graded on the same samples that built the ranking
rho = lambda a, b: stats.spearmanr(a, b)[0]
with torch.no_grad():
    for b in BX:
        m = box == b; e = torch.as_tensor(np.nonzero(m)[0], device="cuda"); xe, ye, n = X[e], Y[e], len(e)
        lh = max(((l, h) for l in range(4) for h in range(H)), key=lambda t: rho(rb[m, t[0], t[1]].mean(0), R["mask_b"][m].mean(0)))
        R[f"best|{b}"] = np.array([*lh, rho(rb[m, lh[0], lh[1]].mean(0), R["mask_b"][m].mean(0))])
        rng = np.random.default_rng(0)
        orders = {"masking": (R["mask_b"][m].mean(0), R["mask_l"][m].mean(0)), "attention": (R["att_b"][m].mean(0), R["att_l"][m].mean(0)), "svd": (R[f"imp_b|{b}"], R[f"imp_l|{b}"]),
                  "weight": (R["w_b"], R["w_l"]), "contribution": (cb[m].mean(0), cl[m].mean(0)), "head": (rb[m, lh[0], lh[1]].mean(0), None)}
        orders = {k: [None if v is None else np.argsort(-v) for v in o] for k, o in orders.items()} | {f"random{r}": [rng.permutation(18), rng.permutation(100)] for r in range(3)}
        for nm, (ob, ol) in orders.items():
            for unit, o, Ks, T in (("band", ob, KB, 18), ("laser", ol, KL, 100)):
                if o is None: continue
                r = []
                for k in Ks:
                    mk = torch.zeros(n, T, dtype=torch.bool, device="cuda"); mk[:, torch.as_tensor(o[:k].copy())] = True
                    r.append(iou(fwd(xe, **({"m_b": mk} if unit == "band" else {"m_l": mk})), ye).mean().item())
                R[f"c|{b}|{unit}|{nm}"] = np.array(r)
        print(b, "best head", R[f"best|{b}"].round(2), {k: R[f"c|{b}|band|{k}"][[2, 8, 14]].round(3).tolist() for k in orders}, flush=True)

# input-signal views: per-laser band means of v = |X| / σ_sample, σ(f) over the 200 channels; same for the set's empty-box captures
def inputs(x): v = x / x.std((1, 2, 3), keepdim=True); return v.view(len(x), 100, 18, 65, 2).mean((3, 4)).cpu().numpy(), x.permute(0, 2, 1, 3).flatten(2).std(2).cpu().numpy()
R["vb"], R["sigma"] = inputs(X)
emp = np.array([j for j in np.nonzero(B["y"].reshape(len(I), -1).any(1) == False)[0] if META[int(I[j])]["n_objects"] == 0 and (split[j] == "train") == (SET == "train") and "shifted" not in split[j]])
for b in BX:   # boxes without ≥ 8 clean empty captures in the set fall back to the train empties
    if sum(boxof(j) == b for j in emp) < 8:
        print(f"{b}: < 8 {SET} empty captures -> train empties", flush=True)
        emp = np.r_[emp, [j for j in np.nonzero((split == "train") & (B["y"].reshape(len(I), -1).any(1) == False))[0] if META[int(I[j])]["n_objects"] == 0 and boxof(j) == b]].astype(int)
del X; torch.cuda.empty_cache()
EV, ES = zip(*[inputs(load(emp[s:s + 100]).cuda()) for s in range(0, len(emp), 100)])
R |= dict(emp=emp, emp_box=np.array([boxof(j) for j in emp]), emp_spk=np.array([META[int(I[j])]["speaker"] for j in emp]), emp_vb=np.concatenate(EV), emp_sigma=np.concatenate(ES))
print("empties", {b: (R["emp_box"] == b).sum() for b in BX}, flush=True)
np.savez(OUT / f"q1_{SET}.npz", **R)
np.savez(OUT / f"heads_{SET}.npz", rb=rb, eb=eb, rl=rl, el=el, fmap=fmap, clean=R["clean"], abl=abl)
print("done")
