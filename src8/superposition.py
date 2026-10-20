"""Can a two-object capture be predicted from the two single-object captures, and by how much of a model? additivity.py tested the
plain sum (v3 = v1 + v2: no); this fits LINEAR maps on top of it, by least squares on the train pairs, and scores them on held-out
pairs (the eval split's positions). v = the deviation of log power from the nearest-in-time empty box of the same speaker, (L,F,C).
  sum        v3 = v1 + v2                                         (additivity.py's r_sum)
  mean       v3 = b                                               the average two-object deviation alone: what is common to every scene
  scalar     v3 = a (v1 + v2) + b                                 a < 1 would mean the sum double-counts (noise, common mode)
  diag       v3_i = a_i (v1 + v2)_i + b_i                         one gain per (laser, bin, axis), shrunk toward the scalar
  freq       v3[l,:,c] = A_F (v1 + v2)[l,:,c] + b                 A_F (F x F), shared by all lasers and axes: mode shifts mix bins
  laser      v3[:,f,c] = A_L (v1 + v2)[:,f,c] + b                 A_L (L x L), shared by all bins: mode shapes mix lasers
  lowrank    v3 = b + U_3 M U_s^T (v1 + v2)                       a general A, restricted to the top-r principal directions
  one        v3 = a v_big + b                                     the bigger object alone: does the second add anything?
A general A (d x d, d = L F C = 234k) is d^2 = 5.5e10 unknowns against ~1100 train pairs: least squares with structure (the rows above)
or a rank limit, and ridge. Ridge strengths are picked on a 20% slice of the train pairs.
Per pair: the objects' centroid distance (box fractions), same type (cube+cube) or not (cube+vase), box -> results by distance / type.
Then the trained model (--run) on the real capture vs synthetic inputs (the sum, the best fit, one object): iou@0.08 vs the union and
whether it finds both objects.
Domains (--domain): log_power = log |X|^2 (v in log units, the loud and the dead bins weigh most); model = as the model reads its input:
|X| / its per-sample std, and each (laser, bin, axis) of v scaled to std 1 within its box (as box_norm), so every feature counts alike.
Per box, not per speaker: every v is a same-speaker difference (a capture minus its own empty box), so the speaker mostly cancels.
Controls: _shuf = v1 + v2 taken from another pair of the same layout (same session drift, other objects / positions); lolo.csv = fit
on every other layout (recording session), scored on the held-out one.
  .venv/bin/python src8/superposition.py [--run s8-best-mag-500ep] [--domain log_power|model]   -> runs/superposition/<domain>/*.csv"""
import argparse, random, sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
import dataset as D
from additivity import EXPS, TYPE, MIN_IOU, SKIP, mask64, iou, t, spec, corr
from probe import ROOT, load_model

OUT = ROOT / "runs" / "superposition"

def pairs(rng, domain):
    """the matched two-object captures (as additivity.py) + centroid distance, type; log-power deviations v1, v2, v3 as (N,L,F,C) fp16"""
    splits = D.DATASETS["gastronorm_plastic_cardboard"](0.2, [1, 3, 5, 7])
    samples = {d: m for e in EXPS.values() for d, m in D.collect_samples(D.EXPERIMENTS / e).items()}
    dirs, metas = list(samples), list(samples.values())
    idxs = D.build_splits(splits, metas)
    is_eval = {dirs[i] for n, ids in idxs.items() if n != "train" for i in ids}
    info = D.get_data_info(samples, idxs["train"])
    rows, V = [], []
    L = (lambda p: np.log(p + 1e-6)) if domain == "log_power" else (lambda p: (lambda x: x / x.std())(np.sqrt(p)))
    for box, exp in EXPS.items():
        S = {d: m for d, m in samples.items() if m["experiment"] == exp}
        empties = [(d, m) for d, m in S.items() if m["n_objects"] == 0 and not any(s in m["layout"] for s in SKIP)]
        singles = [(d, m) for d, m in S.items() if m["n_objects"] == 1 and len(m["objects"]) == 1 and next(iter(m["objects"])) in TYPE
                   and not any(s in m["layout"] for s in SKIP)]
        smask = {d: mask64(d / "image/03_smask.png") for d, _ in tqdm(singles, desc=f"{box} single masks")}
        def x0(m):
            c = [(abs(t(em) - t(m)), ed, em) for ed, em in empties if em["speaker"] == m["speaker"]]
            return min(c, key=lambda z: z[0])[1:] if c else None
        for d, m in tqdm([(d, m) for d, m in S.items() if m["n_objects"] == 2], desc=f"{box} pairs"):
            inst = sorted((d / "image/smasks").glob("*.npy"))
            if len(inst) != 2 or x0(m) is None: continue
            objs = []
            for f in inst:
                typ, mk = TYPE.get(f.stem.rstrip("0123456789")), mask64(f)
                cand = sorted([(iou(mk, smask[sd]), sd, sm) for sd, sm in singles if sm["speaker"] == m["speaker"] and TYPE[next(iter(sm["objects"]))] == typ], key=lambda z: -z[0])
                if not cand or cand[0][0] < MIN_IOU: break
                objs.append(dict(typ=typ, mask=mk, best=cand[0]))
            if len(objs) != 2: continue
            cen = [np.array(np.nonzero(o["mask"])).mean(1) / 64 for o in objs]
            v1, v2 = [L(spec(o["best"][1], o["best"][2])) - L(spec(*x0(o["best"][2]))) for o in objs]
            v3 = L(spec(d, m)) - L(spec(*x0(m)))
            big = int(objs[1]["mask"].sum() > objs[0]["mask"].sum())
            rows.append(dict(box=box, layout=m["layout"], sample=d.name, dir=str(d), x0=str(x0(m)[0]), speaker=m["speaker"], eval=d in is_eval,
                             types="+".join(sorted(o["typ"] for o in objs)), same=objs[0]["typ"] == objs[1]["typ"],
                             dist=float(np.linalg.norm(cen[0] - cen[1])), iou_a=objs[0]["best"][0], iou_b=objs[1]["best"][0], big=big,
                             single_a=str(objs[0]["best"][1]), single_b=str(objs[1]["best"][1]),
                             x0_a=str(x0(objs[0]["best"][2])[0]), x0_b=str(x0(objs[1]["best"][2])[0])))
            V.append(np.stack([v1, v2, v3]).astype(np.float16))
    df, V = pd.DataFrame(rows), np.stack(V)
    sd = {}
    if domain == "model":                                                                  # each feature to std 1 within its box
        for b in EXPS:
            k = np.nonzero((df.box == b).to_numpy())[0]
            sd[b] = np.sqrt(sum((V[i].astype(np.float32) ** 2).mean(0) for i in k) / len(k)).clip(1e-6)  # (L,F,C) rms over the box's v's
            for i in k: V[i] = (V[i].astype(np.float32) / sd[b]).astype(np.float16)
    return df, V, info, sd, L

def ridge(XtX, XtY, lam):
    """(k,k), (k,m) -> (k,m) the ridge solution, lam relative to the mean eigenvalue"""
    k = len(XtX)
    return torch.linalg.solve(XtX + lam * XtX.diagonal().mean() * torch.eye(k, dtype=XtX.dtype), XtY)

def fit(V, tr, lam=None):
    """fit every model on pairs tr (bool, (N,)) -> {name: predict(idx) -> (n,L,F,C) float32}; lam picked on a 20% slice of tr"""
    N, _, Lz, Fz, Cz = V.shape
    g = lambda i: torch.from_numpy(V[i].astype(np.float32))                                  # (n,3,L,F,C)
    tri = np.nonzero(tr)[0]
    rs = np.random.default_rng(0); rs.shuffle(sub := tri.copy())
    inner, val = np.sort(sub[len(sub) // 5:]), np.sort(sub[:len(sub) // 5])

    def stats(ids):
        x = g(ids); s, y = x[:, 0] + x[:, 1], x[:, 2]
        return s, y, x[np.arange(len(ids)), torch.as_tensor(df_big[ids])], x[:, 0] * x[:, 1]  # sum, v3, the bigger object's v, v1 * v2
    def models(ids):
        s, y, vb, pr = stats(ids)
        sm, ym, vbm, pm = s.mean(0), y.mean(0), vb.mean(0), pr.mean(0)
        prc = pr - pm
        sc, yc, vbc = s - sm, y - ym, vb - vbm
        a = float((sc * yc).sum() / (sc * sc).sum())
        a1 = float((vbc * yc).sum() / (vbc * vbc).sum())
        cov, var = (sc * yc).sum(0), (sc * sc).sum(0)
        Spp, Ssp, Spy = (prc * prc).sum(0), (sc * prc).sum(0), (prc * yc).sum(0)                # inter: per feature, regressors (s, v1 v2)
        # freq: rows (n*L*C, F); laser: rows (n*F*C, L)
        Us = torch.linalg.svd(sc.flatten(1), full_matrices=False)[2]                          # (n,d) right singular vectors of the sums
        Uy = torch.linalg.svd(yc.flatten(1), full_matrices=False)[2]
        return dict(sm=sm, ym=ym, vbm=vbm, pm=pm, Spp=Spp, Ssp=Ssp, Spy=Spy, a=a, a1=a1, cov=cov, var=var, XfX=torch.einsum("nlfc,nlgc->fg", sc, sc).double(), XfY=torch.einsum("nlfc,nlgc->fg", sc, yc).double(),
                    XlX=torch.einsum("nlfc,nkfc->lk", sc, sc).double(), XlY=torch.einsum("nlfc,nkfc->lk", sc, yc).double(),
                    Us=Us, Uy=Uy, cs=sc.flatten(1) @ Us.T, cy=yc.flatten(1) @ Uy.T)
    def predictors(M, lam):
        P = {}
        P["sum"] = lambda s, vb, pr=None: s
        P["mean"] = lambda s, vb, pr=None: M["ym"].expand_as(s)
        P["scalar"] = lambda s, vb, pr=None: M["ym"] + M["a"] * (s - M["sm"])
        ai = (M["cov"] + lam["diag"] * M["var"].mean() * M["a"]) / (M["var"] + lam["diag"] * M["var"].mean())
        P["diag"] = lambda s, vb, pr=None: M["ym"] + ai * (s - M["sm"])
        # inter: diag + an interaction term per feature, v3_i = a_i s_i + c_i (v1 v2)_i + b_i (ridge-shrunk 2x2 normal equations). The
        # test of linearity: if two objects interact (the second changes how the first one's change shows), c picks it up
        l1, l2 = lam["inter"] * M["var"].mean(), lam["inter"] * M["Spp"].mean()
        A11, A12, A22 = M["var"] + l1, M["Ssp"], M["Spp"] + l2
        b1, b2 = M["cov"] + l1 * M["a"], M["Spy"]
        det = (A11 * A22 - A12 ** 2).clamp_min(1e-12)
        ia, ic = (A22 * b1 - A12 * b2) / det, (A11 * b2 - A12 * b1) / det
        P["inter"] = lambda s, vb, pr=None: M["ym"] + ia * (s - M["sm"]) + ic * (pr - M["pm"])
        Af = ridge(M["XfX"], M["XfY"], lam["freq"]).float()                                 # (F,F): y_row = x_row @ Af
        P["freq"] = lambda s, vb, pr=None: M["ym"] + torch.einsum("nlfc,fg->nlgc", s - M["sm"], Af)
        Al = ridge(M["XlX"], M["XlY"], lam["laser"]).float()                              # (L,L): y_k = sum_l x_l Al[l,k]
        P["laser"] = lambda s, vb, pr=None: M["ym"] + torch.einsum("nlfc,lk->nkfc", s - M["sm"], Al)
        r = lam["rank"]
        Mr = ridge(M["cs"][:, :r].T @ M["cs"][:, :r], M["cs"][:, :r].T @ M["cy"][:, :r], lam["lowrank"])   # (r,r)
        P["lowrank"] = lambda s, vb, pr=None: M["ym"] + (((s - M["sm"]).flatten(1) @ M["Us"][:r].T) @ Mr @ M["Uy"][:r]).view_as(s)
        P["one"] = lambda s, vb, pr=None: M["ym"] + M["a1"] * (vb - M["vbm"])
        # rbf: the nonlinear counterpart of lowrank -- kernel ridge (gaussian kernel) from the same r input coefficients to the same r
        # output coefficients. If it does not beat lowrank, the two-object capture is as linear in v1 + v2 as the data can tell
        Xr = M["cs"][:, :r].double()
        sq = lambda A, B: ((A * A).sum(1)[:, None] + (B * B).sum(1)[None] - 2 * A @ B.T).clamp_min(0)
        ell2 = float(sq(Xr, Xr).median()) * lam["rbf_scale"]
        alpha = torch.linalg.solve(torch.exp(-sq(Xr, Xr) / ell2) + lam["rbf"] * torch.eye(len(Xr), dtype=Xr.dtype), M["cy"][:, :r].double())
        P["rbf"] = lambda s, vb, pr=None: M["ym"] + ((torch.exp(-sq(((s - M["sm"]).flatten(1) @ M["Us"][:r].T).double(), Xr) / ell2) @ alpha).float() @ M["Uy"][:r]).view_as(s)
        return P
    def score(P, ids):
        s, y, vb, pr = stats(ids)
        out = {}
        for k, p in P.items():
            yh = p(s, vb, pr)
            out[k] = (1 - ((y - yh) ** 2).flatten(1).sum(1) / (y ** 2).flatten(1).sum(1)).numpy()   # fve per pair
        return out
    # pick lam on the inner/val split (one parameter at a time, the rest at their defaults)
    grid = dict(diag=[0.1, 1, 10, 100], freq=[1e-3, 1e-2, 0.1, 1, 10], laser=[1e-3, 1e-2, 0.1, 1, 10], lowrank=[1e-2, 0.1, 1, 10, 100], rank=[10, 25, 50, 100, 200],
                rbf=[1e-3, 1e-2, 0.1, 1], rbf_scale=[0.25, 1, 4], inter=[1e-3, 1e-2, 0.1, 1, 10])
    if lam is not None: return predictors(models(tri), lam), lam, score, None              # given: no search (leave-one-layout-out)
    lam = dict(diag=1.0, freq=0.1, laser=0.1, lowrank=1.0, rank=50, rbf=0.01, rbf_scale=1.0, inter=0.1)
    Mi = models(inner)
    for k, vals in grid.items():
        best = None
        for v in vals:
            if k == "rank" and v > len(inner) - 1: continue
            fv = score({kk: pp for kk, pp in predictors(Mi, lam | {k: v}).items() if kk == {"rank": "lowrank", "rbf_scale": "rbf"}.get(k, k)}, val)
            m = float(np.mean(list(fv.values())[0]))
            if best is None or m > best[0]: best = (m, v)
        lam[k] = best[1]; print(f"lam {k} = {best[1]} (val fve {best[0]:.3f})", flush=True)
    M = models(tri)
    return predictors(M, lam), lam, score, M

df_big = None

def main():
    global df_big
    ap = argparse.ArgumentParser(); ap.add_argument("--run", default="s8-best-mag-500ep"); ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--domain", default="log_power", choices=("log_power", "model"))
    a = ap.parse_args(); torch.set_num_threads(a.threads)
    global OUT
    OUT = OUT / a.domain; OUT.mkdir(parents=True, exist_ok=True)
    df, V, info, sd, T = pairs(random.Random(0), a.domain)
    df_big = df.big.to_numpy()
    import additivity                                                                      # free the spectra the model test won't read (~4 GB)
    need = set(df.loc[df["eval"], "dir"]) | set(df.loc[df["eval"], "x0"])
    for k in [k for k in additivity._fft if str(k) not in need]: del additivity._fft[k]
    print(f"{len(df)} pairs ({df['eval'].sum()} eval), V {V.shape} {V.nbytes / 1e9:.1f} GB", flush=True)
    tr = ~df["eval"].to_numpy()
    P, lam, score, M = fit(V, tr)
    ev = np.nonzero(~tr)[0]
    group = (df.box + "/" + df.layout).to_numpy()
    def shuffled(ids):
        """each pair's partner: another pair of the same layout (session) in ids -- different objects / positions, same drift"""
        out = ids.copy()
        for g in np.unique(group[ids]):
            k = np.nonzero(group[ids] == g)[0]
            if len(k) > 1: out[k] = ids[np.roll(k, 1)]
        return out
    def evaluate(P, ids, src=None):
        """{metric/model: per-pair values} for pairs ids; src: take v1 + v2 (and v_big) from these pairs instead (the control)"""
        src = ids if src is None else src
        res = {f"{m}/{k}": [] for k in P for m in ("fve", "r")}
        for i in range(0, len(ids), 32):
            a_, b_ = ids[i:i + 32], src[i:i + 32]
            y = torch.from_numpy(V[a_, 2].astype(np.float32))
            x = torch.from_numpy(V[b_].astype(np.float32)); s, vb, pr = x[:, 0] + x[:, 1], x[np.arange(len(b_)), torch.as_tensor(df_big[b_])], x[:, 0] * x[:, 1]
            for k, p in P.items():
                yh = p(s, vb, pr)
                res[f"fve/{k}"] += list((1 - ((y - yh) ** 2).flatten(1).sum(1) / (y ** 2).flatten(1).sum(1)).numpy())
                res[f"r/{k}"] += [corr(y[j].numpy(), yh[j].numpy()) for j in range(len(a_))]
        return res
    E = df.iloc[ev].reset_index(drop=True)
    for k, v in evaluate(P, ev).items(): E[k] = v
    for k, v in evaluate(P, ev, shuffled(ev)).items(): E[k.replace("/", "_shuf/")] = v   # control: the singles of another pair, same session
    E.to_csv(OUT / "pairs.csv", index=False)
    print("control (singles of another pair of the same layout):", flush=True)
    print(E[[c for c in E.columns if c.startswith("fve_shuf/")]].mean().round(3).to_string(), flush=True)
    # leave one layout (= recording session: its own drift) out: fit on every other layout, score on this one
    lolo = []
    for g in np.unique(group):
        te = np.nonzero(group == g)[0]
        Pg = fit(V, group != g, lam)[0]
        Eg = df.iloc[te].reset_index(drop=True)
        for k, v in evaluate(Pg, te).items(): Eg[k] = v
        for k, v in evaluate(Pg, te, shuffled(te)).items(): Eg[k.replace("/", "_shuf/")] = v
        lolo.append(Eg); print(f"layout-out {g} ({len(te)}): " + " ".join(f"{k} {np.mean(Eg[f'fve/{k}']):.2f}" for k in Pg), flush=True)
    pd.concat(lolo).to_csv(OUT / "lolo.csv", index=False)
    fve = {k: E[f"fve/{k}"].to_numpy() for k in P}
    pd.DataFrame([lam]).to_csv(OUT / "fits.csv", index=False)
    print(E[[c for c in E.columns if c.startswith(("fve/", "r/"))]].mean().round(3).to_string(), flush=True)
    best = max((k for k in P if k not in ("sum", "mean", "one", "rbf", "inter")), key=lambda k: np.mean(fve[k]))
    print(f"best linear map: {best}", flush=True)

    # the trained model on real vs synthetic inputs, eval pairs (CPU: the GPU belongs to the training queue)
    import additivity
    ck = sorted((ROOT / "runs" / a.run / "checkpoints").glob("ep*-rank0.pt"), key=lambda p: int(p.name[2:].split("-")[0]))[-1]
    cuda, torch.nn.Module.cuda = torch.nn.Module.cuda, lambda self, *x, **k: self              # load_model moves to the GPU: stay on the CPU
    m = load_model("src8", a.run, ck, info, {}).eval()
    torch.nn.Module.cuda = cuda
    sig = m.signal
    bidx = {"gastro": info["boxes"].index("gastronorm"), "plastic": info["boxes"].index("green-plastic"), "cardboard": info["boxes"].index("cardboard")}
    H, W = m.out_h, m.out_w
    def to_in(v):  # a capture in the analysis domain -> the run's input (process_x: |fft| or log power, then / its std)
        if a.domain == "model": x = v.clip(0) if sig == "magnitude" else np.log(v.clip(1e-3) ** 2 + 1e-6)
        else: x = np.exp(0.5 * v) if sig == "magnitude" else v
        return x / x.std()
    rows = []
    for j0 in tqdm(range(0, len(ev), 16), desc="model"):
        ids = ev[j0:j0 + 16]
        x = torch.from_numpy(V[ids].astype(np.float32)); s_, y_, vb = x[:, 0] + x[:, 1], x[:, 2], x[np.arange(len(ids)), torch.as_tensor(df_big[ids])]
        fits = {"best": P[best](s_, vb), "mean": P["mean"](s_, vb)}
        r_ = lam["rank"]                                                                    # the real v3 kept only in lowrank's output subspace:
        fits["proj"] = M["ym"] + (((y_ - M["ym"]).flatten(1) @ M["Uy"][:r_].T) @ M["Uy"][:r_]).view_as(y_)  # does it hold what the model reads?
        for k, i in enumerate(ids):
            r = df.iloc[i]
            lp0, g = T(additivity._fft[Path(r["x0"])]), sd.get(r["box"], 1.0)                 # the empty box in the domain; v's scale
            variants = dict(real=T(additivity._fft[Path(r["dir"])]), sum=lp0 + g * s_[k].numpy(), best=lp0 + g * fits["best"][k].numpy(),
                            mean=lp0 + g * fits["mean"][k].numpy(), one=lp0 + g * vb[k].numpy(), proj=lp0 + g * fits["proj"][k].numpy())
            inst = [np.asarray(Image.fromarray((np.load(f) > 0).astype(np.float32)).resize((W, H), Image.BOX)) > .5 for f in sorted((Path(r["dir"]) / "image/smasks").glob("*.npy"))]
            u = inst[0] | inst[1]
            xb = torch.from_numpy(np.stack([to_in(v) for v in variants.values()])).float()
            with torch.no_grad(): p = m(dict(fft=xb, box=torch.full((len(xb),), bidx[r["box"]]), inst=torch.zeros(len(xb), 2, H, W)))["mask_pred"].float().numpy() > 0.08
            res = dict(sample=r["sample"], box=r["box"], types=r["types"], same=r["same"], dist=r["dist"])
            for (name, _), b in zip(variants.items(), p):
                cov = [(b & mk).sum() / max(mk.sum(), 1) for mk in inst]
                res |= {f"iou/{name}": (b & u).sum() / max((b | u).sum(), 1), f"both/{name}": float(min(cov) > .1), f"none/{name}": float(max(cov) <= .1)}
            rows.append(res)
    mo = pd.DataFrame(rows)
    mo.to_csv(OUT / "model.csv", index=False)
    print(f"{a.run} on eval pairs (best fit = {best}):")
    print(mo[[c for c in mo.columns if "/" in c]].mean().round(3).to_string())

if __name__ == "__main__":
    main()
