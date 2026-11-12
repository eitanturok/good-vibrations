# nb140 cache: per-token masking importance + input loudness for decres-rescale-aug10-v1 (eager fp32)
# samples = nb136 subset720 (60 single-object samples per ID object x box, seed 0) for SET in train / test (= eval splits)
# per sample: clean IoU, ΔIoU = clean - IoU with one freq band (18) / one laser (100) dropped (key-pad + zeroed readout slot, as in nb136),
# vb = mean model input |FFT|/σ_sample per (laser, band) (the dataset's x, before the model's own per-sample z-score),
# top-K drop curves (bands / lasers) ranked by train importance, this set's loudness, quiet-first, random
# -> runs/_src9_cache/nb140/{train,test}.npz
import sys, numpy as np, torch
from pathlib import Path
ROOT = Path("/home/ethantu/workspace/good-vibrations"); sys.path.insert(0, str(ROOT / "src9"))
from dataset import build_dataset
from arch import transformer4 as T4
OUT = ROOT / "runs/_src9_cache/nb140"; OUT.mkdir(parents=True, exist_ok=True)
torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
_, ev, tr_ev, info = build_dataset("tukey", num_workers=0)
DS = {"test": ev[0].dataloader.dataloader.dataset.dataset, "train": tr_ev.dataloader.dataloader.dataset.dataset}
model = T4.Transformer4(info, decoder="res", latent_norm="rescale").cuda().eval()
model.load_state_dict(torch.load(ROOT / "runs/decres-rescale-aug10-v1/checkpoints/ep300-ba3000-rank0.pt", map_location="cpu", weights_only=False)["state"]["model"])
B, SU = np.load(ROOT / "runs/_src9_cache/nb136/base.npz"), np.load(ROOT / "runs/_src9_cache/nb136/subset720.npz")
META, BX = DS["test"].metas, ["gastro", "cardboard", "plastic"]
KB, KL = np.arange(1, 19), np.array([1, 2, 3, 5, 8, 12, 18, 25, 35, 50, 70, 90])

def fwd(x, m_l=None, m_b=None):
    n = len(x); m_l = x.new_ones(n, 100, dtype=torch.bool) if m_l is None else m_l; m_b = x.new_ones(n, 18, dtype=torch.bool) if m_b is None else m_b
    return model.decoder(model.encoder(x, m_l, m_b)).sigmoid()
def iou(p, y): p = p > .08; return (p & y).flatten(1).sum(1) / (p | y).flatten(1).sum(1).clamp_min(1)
def keep(n, T, drop): m = torch.ones(n, T, dtype=torch.bool, device="cuda"); m[:, drop] = False; return m

R = {}
for SET in ("train", "test"):
    rows = SU["train" if SET == "train" else "eval"]; ds = DS[SET]
    meta = [META[int(B["i"][j])] for j in rows]
    box = np.array([{"gastronorm": "gastro", "green-plastic": "plastic"}.get(m["box"], m["box"]) for m in meta])
    obj = np.array(["+".join(m["objects"]) for m in meta]); spk = np.array([m["speaker"] for m in meta]); pos = np.array([int(m["position_id"]) for m in meta])
    X = torch.stack([ds[int(B["i"][j])]["fft"] for j in rows]).cuda(); Y = torch.from_numpy(B["y"][rows]).cuda()   # (720,100,1170,2) |FFT|/σ, ~1.3 GB
    vb = X.view(len(X), 100, 18, 65, 2).mean((3, 4)).cpu().numpy()
    vc = X.mean((1, 2)).cpu().numpy()                                                    # per-axis mean (720,2)
    clean, mb, ml = [], [], []
    with torch.no_grad():
        for s in range(0, len(X), 40):
            x, y = X[s:s + 40], Y[s:s + 40]; n = len(x); c = iou(fwd(x), y); clean.append(c.cpu())
            mb.append(torch.stack([c - iou(fwd(x, m_b=keep(n, 18, [k])), y) for k in range(18)], 1).cpu())
            ml.append(torch.stack([c - iou(fwd(x, m_l=keep(n, 100, [k])), y) for k in range(100)], 1).cpu())
    clean, mb, ml = [torch.cat(v).numpy() for v in (clean, mb, ml)]
    print(SET, {b: clean[box == b].mean().round(3) for b in BX}, flush=True)
    S = dict(rows=rows, box=box, obj=obj, spk=spk, pos=pos, clean=clean, mask_b=mb, mask_l=ml, vb=vb, vc=vc, KB=KB, KL=KL)
    rng = np.random.default_rng(0)
    with torch.no_grad():
        for b in BX:
            e = np.nonzero(box == b)[0]; t = R["train"]["box"] == b if SET == "test" else box == b
            src = R["train"] if SET == "test" else S
            orders = {"importance (train)": (np.argsort(-src["mask_b"][t].mean(0)), np.argsort(-src["mask_l"][t].mean(0))),
                      "loudest first": (np.argsort(-vb[e].mean((0, 1))), np.argsort(-vb[e].mean((0, 2)))),
                      "quietest first": (np.argsort(vb[e].mean((0, 1))), np.argsort(vb[e].mean((0, 2))))}
            orders |= {f"random{r}": (rng.permutation(18), rng.permutation(100)) for r in range(5)}
            for nm, (ob, ol) in orders.items():
                for unit, o, Ks, T in (("band", ob, KB, 18), ("laser", ol, KL, 100)):
                    r = []
                    for K in Ks:
                        m = torch.ones(len(e), T, dtype=torch.bool, device="cuda"); m[:, torch.as_tensor(o[:K].copy())] = False
                        if K == T: r.append(0.0); continue
                        p = torch.cat([fwd(X[e[s:s + 60]], **({"m_b": m[s:s + 60]} if unit == "band" else {"m_l": m[s:s + 60]})) for s in range(0, len(e), 60)])
                        r.append(iou(p, Y[e]).mean().item())
                    S[f"c|{b}|{unit}|{nm}"] = np.array(r)
            print(SET, b, {k.split("|", 2)[2]: v[[0, 3, 8]].round(3).tolist() for k, v in S.items() if k.startswith(f"c|{b}|band") and "random" not in k}, flush=True)
    R[SET] = S; np.savez(OUT / f"{SET}.npz", **S)
    del X, Y; torch.cuda.empty_cache()
print("done")
