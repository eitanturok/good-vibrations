"""Are two objects' effects on the spectrum additive? If so, two-object training scenes can be synthesized from single-object captures:
    X_ab ~ X0 + (X_a - X0_a) + (X_b - X0_b),  mask = mask_a | mask_b
Physics: to first order (Rayleigh), each mode's shift is a sum over the added masses, dw_k ~ sum_i m_i phi_k(x_i)^2, so the deviations
from the empty box add while the shifts are small next to the linewidth.

For every two-object capture (cube-vase, two-cubes; 3 boxes), each object is matched to the single-object capture of the same type and
speaker whose 64x64 mask overlaps it most (IoU >= MIN_IOU); X0 is the empty-box capture of the same speaker nearest in time to each
capture (drift is speaker-box geometry, nb101). Then, per domain (log power / magnitude / power), over all (laser, bin, axis):
  r_sum   corr(D_ab, D_a + D_b), D = X - X0          does the sum predict the real two-object deviation?
  r_a/r_b corr(D_ab, D_a) / corr(D_ab, D_b)          ... better than either object alone?
  r_null  corr(D_ab, D_a' + D_b') with a', b' random singles of the same types and speaker elsewhere in the box
  r_rep   corr(D_a, D_a2), a2 another single capture of the same object at the same spot (IoU >= .6): the repeat ceiling
  fve     1 - |D_ab - (D_a + D_b)|^2 / |D_ab|^2
and the real test: the trained model (--run, iou@0.08 vs the union mask) on the real capture vs the synthetic one (built in log power,
and in power then logged), on the eval-split captures and on all.
  .venv/bin/python src8/additivity.py [--run s8-best-spkgain-500ep]   -> runs/additivity_src8.csv (per pair), printed summary"""
import argparse, random, sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
import dataset as D
from probe import ROOT, load_model

EXPS = {"gastro": "2026_09_27_gastronorm_four_objs", "plastic": "2026_10_05_green_plastic_four_objs", "cardboard": "2026_10_11_cardboard_four_objs"}
TYPE = {"red-cube": "cube", "green-cube": "cube", "vase": "vase"}   # the green cube has no single captures: matched to red-cube singles
MIN_IOU = 0.5
SKIP = ("lid", "shifted", "laser", "speaker")                      # other laser grids / unseen-speaker sessions / lids: not the same setup

def mask64(path):
    a = np.load(path) > 0 if path.suffix == ".npy" else np.asarray(Image.open(path).convert("F")) / 255.0
    return np.asarray(Image.fromarray(a.astype(np.float32)).resize((64, 64), Image.BOX)) > 0.5

def iou(a, b): return (a & b).sum() / max((a | b).sum(), 1)
def t(meta): return datetime.fromisoformat(meta["timestamp"]).timestamp()

_fft = {}
def spec(d, meta):
    """(L,F,C) power |X|^2, tukey window, as the training pipeline (cached)"""
    if d not in _fft: _fft[d] = D.load_fft_raw(d, meta).abs().pow(2).numpy()
    return _fft[d]

DOMAINS = {"log_power": lambda p: np.log(p + 1e-6), "magnitude": np.sqrt, "power": lambda p: p}

def corr(a, b):
    a, b = a.ravel() - a.mean(), b.ravel() - b.mean()
    return float(a @ b / np.sqrt((a @ a) * (b @ b) + 1e-30))

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--run", default="s8-best-spkgain-500ep"); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(); rng = random.Random(a.seed); torch.set_num_threads(4)

    # which captures are eval: the splits exactly as run.py builds them
    splits = D.DATASETS["gastronorm_plastic_cardboard"](0.2, [1, 3, 5, 7])
    samples = {d: m for e in EXPS.values() for d, m in D.collect_samples(D.EXPERIMENTS / e).items()}
    dirs, metas = list(samples), list(samples.values())
    idxs = D.build_splits(splits, metas)
    is_eval = {dirs[i] for n, ids in idxs.items() if n != "train" for i in ids}
    info = D.get_data_info(samples, idxs["train"])

    rows, model_rows = [], []
    for box, exp in EXPS.items():
        S = {d: m for d, m in samples.items() if m["experiment"] == exp}
        empties = [(d, m) for d, m in S.items() if m["n_objects"] == 0 and not any(s in m["layout"] for s in SKIP)]
        singles = [(d, m) for d, m in S.items() if m["n_objects"] == 1 and len(m["objects"]) == 1 and next(iter(m["objects"])) in TYPE
                   and not any(s in m["layout"] for s in SKIP)]
        pairs = [(d, m) for d, m in S.items() if m["n_objects"] == 2]
        smask = {d: mask64(d / "image/03_smask.png") for d, _ in tqdm(singles, desc=f"{box} single masks")}
        def x0(m):  # the nearest-in-time empty capture of the same speaker
            c = [(abs(t(em) - t(m)), ed, em) for ed, em in empties if em["speaker"] == m["speaker"]]
            return min(c, key=lambda z: z[0])[1:] if c else None
        for d, m in tqdm(pairs, desc=f"{box} pairs"):
            inst = sorted((d / "image/smasks").glob("*.npy"))
            if len(inst) != 2 or x0(m) is None: continue
            objs = []
            for f in inst:
                typ, mk = TYPE.get(f.stem.rstrip("0123456789")), mask64(f)
                cand = [(iou(mk, smask[sd]), sd, sm) for sd, sm in singles if sm["speaker"] == m["speaker"] and TYPE[next(iter(sm["objects"]))] == typ]
                cand.sort(key=lambda z: -z[0])
                if not cand or cand[0][0] < MIN_IOU: break
                best = cand[0]
                rep = next((c for c in cand[1:] if iou(smask[c[1]], smask[best[1]]) >= 0.6), None)       # another capture, same spot
                null = rng.choice([c for c in cand if c[0] < 0.05] or cand)                                 # same type, elsewhere
                objs.append(dict(typ=typ, mask=mk, best=best, rep=rep, null=null))
            if len(objs) != 2: continue
            P = spec(d, m); P0 = spec(*x0(m))
            Pa, Pb = [spec(o["best"][1], o["best"][2]) for o in objs]
            P0a, P0b = [spec(*x0(o["best"][2])) for o in objs]
            Na, Nb = [spec(o["null"][1], o["null"][2]) for o in objs]
            N0a, N0b = [spec(*x0(o["null"][2])) for o in objs]
            r = dict(box=box, layout=m["layout"], sample=d.name, speaker=m["speaker"], eval=d in is_eval,
                     iou_a=objs[0]["best"][0], iou_b=objs[1]["best"][0], types="+".join(o["typ"] for o in objs))
            for name, T in DOMAINS.items():
                Dab = T(P) - T(P0); Da = T(Pa) - T(P0a); Db = T(Pb) - T(P0b)
                Dn = (T(Na) - T(N0a)) + (T(Nb) - T(N0b))
                r |= {f"{name}/r_sum": corr(Dab, Da + Db), f"{name}/r_a": corr(Dab, Da), f"{name}/r_b": corr(Dab, Db),
                      f"{name}/r_null": corr(Dab, Dn), f"{name}/fve": 1 - float(((Dab - Da - Db) ** 2).sum() / ((Dab ** 2).sum() + 1e-30))}
                o = objs[0]
                if o["rep"] is not None:
                    Da2 = T(spec(o["rep"][1], o["rep"][2])) - T(spec(*x0(o["rep"][2])))
                    r[f"{name}/r_rep"] = corr(Da, Da2)
            rows.append(r)
            y = np.asarray(Image.open(d / "image/03_smask.png").convert("F").resize((64, 64), Image.BOX)) / 255.0 > 0.5
            L = lambda p: np.log(np.clip(p, 0, None) + 1e-6)
            model_rows.append(dict(box=box, layout=m["layout"], sample=d.name, eval=d in is_eval, y=y,
                                   real=L(P), syn_log=L(P0) + (L(Pa) - L(P0a)) + (L(Pb) - L(P0b)), syn_pow=L(P0 + (Pa - P0a) + (Pb - P0b)),
                                   only_a=L(Pa), null=L(P0) + (L(Na) - L(N0a)) + (L(Nb) - L(N0b))))

    df = pd.DataFrame(rows)
    df.to_csv(ROOT / "runs/additivity_src8.csv", index=False)
    print(f"\n{len(df)} two-object captures matched (IoU >= {MIN_IOU}); median match IoU {np.median(np.r_[df.iou_a, df.iou_b]):.2f}")
    cols = [c for c in df.columns if "/" in c]
    print(df.groupby(["box", "types"])[cols].mean().T.round(3).to_string())
    print(df[cols].mean().round(3).to_string())

    # the trained model on real vs synthetic two-object inputs (log power -> per-sample std norm, as process_x)
    m = load_model("src8", a.run, sorted((ROOT / "runs" / a.run / "checkpoints").glob("ep*-rank0.pt"), key=lambda p: int(p.name[2:].split("-")[0]))[-1], info, {})
    bidx = {"gastro": info["boxes"].index("gastronorm"), "plastic": info["boxes"].index("green-plastic"), "cardboard": info["boxes"].index("cardboard")}
    out = []
    for r in tqdm(model_rows, desc="model"):
        res = dict(box=r["box"], layout=r["layout"], eval=r["eval"])
        for k in ("real", "syn_log", "syn_pow", "only_a", "null"):
            x = torch.from_numpy(r[k] / r[k].std()).float()[None].cuda()
            with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
                p = m(dict(fft=x, box=torch.tensor([bidx[r["box"]]]).cuda()))["mask_pred"].float()[0].cpu().numpy() > 0.08
            res[k] = iou(p, r["y"]); res[k + "_miss"] = float(not (p & r["y"]).any())
        out.append(res)
    mo = pd.DataFrame(out)
    mo.to_csv(ROOT / "runs/additivity_model_src8.csv", index=False)
    print(f"\n{a.run}: iou@0.08 vs the union mask (real capture / synthetic sum in log power / in power / object a alone / random-pair null)")
    print(mo.groupby(["eval", "box"])[["real", "syn_log", "syn_pow", "only_a", "null"]].mean().round(3).to_string())
    print(mo.groupby("eval")[["real", "syn_log", "syn_pow", "only_a", "null", "real_miss", "syn_log_miss"]].mean().round(3).to_string())

if __name__ == "__main__":
    main()
