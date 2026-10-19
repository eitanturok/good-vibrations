"""Training-free signal quality of input representations, all three boxes: does a change carry more of what the model needs?
Validated proxy (scripts/signal_quality.py, 2026-09-27): held-out-position kernel-ridge CoM skill ranked trained conv runs correctly;
label-free SNR scores did not. So the headline numbers are held-out decodability, and SNR is only a diagnostic.

Per box, single-object id layouts (cube, vase, candle, soap), speakers 1 3 5 7. Each arm -> 100 lasers x 117 bands (10 bins, ~7.7 Hz)
x 2 axes -> standardized PCA(200) -> RBF kernel ridge:
  com_skill    CoM, GroupKFold(5) over positions: 1 - err / err(train mean)        (0 = nothing, 1 = perfect)
  spk_skill    the same, but fit on speakers 1 3 5 and test on speaker 7 at the held-out positions (unseen speaker AND position)
  orient_err   cardboard soap (upright + rotated + sideways): doubled-angle orientation, degrees, rotated layouts only (chance ~34)
  noise_db     diagnostic: rms dB difference between the speakers' captures at one position after removing each speaker's mean
Arms:
  logpow       log |X|^2, tukey (the target's input)            mag          |X|
  center       logpow - mean over lasers (cancels the speaker's S(f))
  empty        logpow - mean empty-box logpow of the same box + speaker (every laser's own transfer function removed)
  deconv       log |X / C|^2 (C = the played chirp's spectrum)
  harm         log power of the 2nd + 3rd harmonic impulse responses (Farina: an exponential sweep's harmonics deconvolve to
               negative times -L ln k, L = T / ln(f1/f0)), on their own
  lin+harm     deconv linear IR spectrum + the harmonics, concatenated
  fuse-mean    position level: logpow averaged over the 4 speakers   fuse-cat   position level: the 4 speakers concatenated
  .venv/bin/python src8/signal_probe.py [--boxes cardboard ...] [--arms logpow center ...]   -> runs/signal_probe_src8.csv"""
import argparse, sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy.signal import butter, chirp, sosfiltfilt
from scipy.signal.windows import tukey
from sklearn.decomposition import PCA
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent))
import dataset as D

ROOT = Path(__file__).resolve().parents[1]
EXPS = {"gastro": "2026_09_27_gastronorm_four_objs", "plastic": "2026_10_05_green_plastic_four_objs", "cardboard": "2026_10_11_cardboard_four_objs"}
OBJECTS = ("cube", "vase", "candle", "soap")
N_BANDS = 117

def spectra(d, meta):
    """-> dict of (L, 117, 2) float16 log features for the per-capture arms"""
    s, fps, lo, hi = np.load(d / "vibration/02_raw_shifts.npy"), float(meta["fps"]), float(meta["min_freq"]), float(meta["max_freq"])
    T = s.shape[1]
    f = np.fft.rfftfreq(T, 1 / fps)
    band = (f >= lo) & (f <= hi)
    X = np.fft.rfft(sosfiltfilt(butter(5, [lo, hi], fs=fps, btype="band", output="sos"), s, axis=1) * tukey(T, 0.02)[None, :, None], axis=1)
    c = chirp(np.arange(int(meta["chirp_t_sec"] * fps)) / fps, f0=meta["chirp_f_start"], t1=meta["chirp_t_sec"], f1=meta["chirp_f_end"], method="logarithmic")
    C = np.fft.rfft(c, T)
    H = np.where(band[None, :, None], X / C[None, :, None], 0)                     # (L,T/2+1,2) transfer function incl. harmonics
    h = np.fft.irfft(H, n=T, axis=1)                                                # (L,T,2) impulse responses
    peak = int((h ** 2).mean((0, 2)).argmax())
    Lsw = meta["chirp_t_sec"] / np.log(meta["chirp_f_end"] / meta["chirp_f_start"])   # Farina: harmonic k at -Lsw ln k
    def win(t0, n=int(0.08 * fps)):                                                 # 80 ms tukey window starting 10 ms before t0
        i = (np.arange(n) + t0 - int(0.01 * fps)) % T
        return np.fft.rfft(h[:, i] * tukey(n, 0.3)[None, :, None], n=T, axis=1)
    harm = [win(peak - int(round(Lsw * np.log(k) * fps))) for k in (2, 3)]
    pool = lambda Z: Z[:, band][:, :N_BANDS * 10].reshape(Z.shape[0], N_BANDS, 10, 2).mean(2)
    lp = lambda Z: np.log(pool(np.abs(Z) ** 2) + 1e-12)
    out = dict(logpow=lp(X), mag=pool(np.abs(X)), deconv=lp(H), harm=np.concatenate([lp(z) for z in harm], -1))
    return {k: v.astype(np.float16) for k, v in out.items()}

def orientation(mask):
    r, c = np.nonzero(mask); r, c = r / mask.shape[0], c / mask.shape[1]
    rr, cc, rc = r.var(), c.var(), ((r - r.mean()) * (c - c.mean())).mean()
    return 0.5 * np.arctan2(2 * rc, rr - cc)

def load_box(box):
    f = ROOT / f"runs/_probe_cache/signal_probe_{box}.npz"
    if f.exists():
        z = np.load(f, allow_pickle=True)
        return pd.DataFrame(z["rows"].tolist()), {k[2:]: z[k] for k in z.files if k.startswith("f_")}
    df, feats = _load_box(box)
    f.parent.mkdir(parents=True, exist_ok=True)
    np.savez(f, rows=np.array(df.to_dict("records"), dtype=object), **{f"f_{k}": v for k, v in feats.items()})
    return df, feats

def _load_box(box):
    exp = EXPS[box]
    S = D.collect_samples(D.EXPERIMENTS / exp)
    keep = {d: m for d, m in S.items() if m["speaker"] in (1, 3, 5, 7) and (m["n_objects"] == 1 and any(o in m["layout"] for o in OBJECTS) and "lid" not in m["layout"]
                                                                             and "shift" not in m["layout"] and "unseeen" not in m["layout"] or m.get("is_empty_box"))}
    rows, feats = [], {}
    for i, (d, m) in enumerate(keep.items()):
        if i % 500 == 0: print(f"{box}: {i}/{len(keep)}", flush=True)
        f = spectra(d, m)
        for k, v in f.items(): feats.setdefault(k, []).append(v)
        H, W = m["crop_overhead_shape"][:2]
        com = np.array([[c[0][0] / H, c[0][1] / W] for c in m["coms"]]).mean(0) if m["coms"] else np.array([np.nan, np.nan])
        th = orientation(np.asarray(Image.open(d / "image/03_smask.png").convert("L").resize((256, 256), Image.BOX)) > 127) if "soap" in m["layout"] else np.nan
        rows.append(dict(layout=m["layout"], position_id=m["position_id"], speaker=m["speaker"], is_empty=bool(m.get("is_empty_box")), com_r=com[0], com_c=com[1], theta=th))
    return pd.DataFrame(rows), {k: np.stack(v) for k, v in feats.items()}

def krr_cv(F, Y, groups, train_mask=None, test_mask=None, folds=5):
    """GroupKFold predictions; train_mask/test_mask restrict which rows fit / are scored in each fold"""
    P = np.full_like(Y, np.nan, dtype=float)
    for tr, te in GroupKFold(folds).split(F, Y, groups):
        if train_mask is not None: tr = tr[train_mask[tr]]
        if test_mask is not None: te = te[test_mask[te]]
        sc = StandardScaler().fit(F[tr]); a, b = sc.transform(F[tr]), sc.transform(F[te])
        g = 1 / np.median(((a[:200, None] - a[None, :200]) ** 2).sum(-1))
        m = KernelRidge(alpha=1.0, kernel="rbf", gamma=g).fit(a, Y[tr] - Y[tr].mean(0))
        P[te] = m.predict(b) + Y[tr].mean(0)
    return P

def skill(P, Y, groups):
    ok = ~np.isnan(P).any(1)
    err = np.sqrt(((P[ok] - Y[ok]) ** 2).sum(1)).mean()
    base = np.sqrt(((Y[ok] - Y[ok].mean(0)) ** 2).sum(1)).mean()
    return 1 - err / base

def evaluate(box, df, feats, arms):
    res = []
    obj = ~df.is_empty.to_numpy()
    for arm in arms:
        if arm.startswith("fuse"):  # position level: speakers stacked in a fixed order, positions with all 4 speakers only
            base = feats["logpow"]
            g = df[obj].groupby(["layout", "position_id"])
            idx = [grp.drop_duplicates("speaker").sort_values("speaker").index.to_numpy() for _, grp in g if set(grp.speaker) == {1, 3, 5, 7}]
            # fuse-spk1: the same positions, speaker 1 only -- the fair baseline for the fused arms (same number of training points)
            pick = {"fuse-mean": lambda i: base[i].astype(np.float32).mean(0), "fuse-cat": lambda i: base[i].astype(np.float32).reshape(-1), "fuse-spk1": lambda i: base[i[0]].astype(np.float32)}[arm]
            F = np.stack([pick(i) for i in idx]).reshape(len(idx), -1)
            sub = df.loc[[i[0] for i in idx]].reset_index(drop=True)
        else:
            if arm == "center": X = feats["logpow"] - feats["logpow"].mean(1, keepdims=True)
            elif arm == "empty":
                X = feats["logpow"].astype(np.float32).copy()
                for s in (1, 3, 5, 7):
                    e = (df.is_empty & (df.speaker == s)).to_numpy()
                    X[(df.speaker == s).to_numpy()] -= X[e].mean(0) if e.any() else 0
            elif arm == "lin+harm": X = np.concatenate([feats["deconv"], feats["harm"]], -1)
            else: X = feats[arm]
            F = X[obj].astype(np.float32).reshape(obj.sum(), -1); sub = df[obj].reset_index(drop=True)
        F = PCA(min(200, len(F) - 1), random_state=0).fit_transform(StandardScaler().fit_transform(F))
        Y, grp = sub[["com_r", "com_c"]].to_numpy(), (sub.layout + "/" + sub.position_id.astype(str)).to_numpy()
        r = dict(box=box, arm=arm, n=len(F), com_skill=skill(krr_cv(F, Y, grp), Y, grp))
        if not arm.startswith("fuse"):
            spk = sub.speaker.to_numpy()
            r["spk_skill"] = skill(krr_cv(F, Y, grp, train_mask=spk != 7, test_mask=spk == 7), Y, grp)
            soap = sub.layout.str.contains("soap").to_numpy()
            if box == "cardboard" and soap.any():
                th = sub.theta.to_numpy()[soap]
                T2 = np.stack([np.cos(2 * th), np.sin(2 * th)], 1)
                P = krr_cv(F[soap], T2, grp[soap])
                e = np.abs(np.arctan2(P[:, 1], P[:, 0]) / 2 - th) % np.pi; e = np.degrees(np.minimum(e, np.pi - e))
                rot = sub.layout.to_numpy()[soap]; rot = np.char.find(rot.astype(str), "rotated") >= 0
                r["orient_err_rotated"], r["orient_err_upright"] = e[rot].mean(), e[~rot].mean()
            Xs = X[obj].astype(np.float32)
            if arm != "mag":  # noise diagnostic, dB: speakers at one position differ by speaker + noise; remove each speaker's mean first
                Z = Xs - np.stack([Xs[spk == s].mean(0) for s in spk])
                gg = pd.Series(range(len(sub))).groupby(grp)
                r["spread_db"] = float(np.mean([Z[i].std(0).mean() for _, i in gg if len(i) > 1]) * 10 / np.log(10))
        print(r, flush=True)
        res.append(r)
    return res

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--boxes", nargs="+", default=["cardboard", "plastic", "gastro"])
    ap.add_argument("--arms", nargs="+", default=["logpow", "mag", "center", "empty", "deconv", "harm", "lin+harm", "fuse-spk1", "fuse-mean", "fuse-cat"])
    a = ap.parse_args()
    out = ROOT / "runs/signal_probe_src8.csv"
    for box in a.boxes:
        df, feats = load_box(box)
        res = pd.DataFrame(evaluate(box, df, feats, a.arms))
        cols = ["box", "arm", "n", "com_skill", "spk_skill", "orient_err_rotated", "orient_err_upright", "spread_db"]
        res.reindex(columns=cols).to_csv(out, mode="a", header=not out.exists(), index=False)
        del feats
    print(pd.read_csv(out).round(3).to_string())
