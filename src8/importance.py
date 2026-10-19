"""Which lasers and which frequencies does a trained model read? Faithful importance by occlusion: take part of the input away and
measure how much worse the mask gets, on the CORE + HARD eval samples. The architecture's own band gates / laser attention are near
uniform and a following linear layer can rescale anything, so they are not importance (attribution.py).
  baselines (--baseline): what "taking it away" means
    mean      replace it by its box's mean signal over the eval set: in range, but an unnaturally smooth spectrum; measures the
              sample-specific information (zeros would be out of distribution: they measure the model's surprise)
    resample  replace it by the same laser / band of another eval sample of the same box (one fixed partner per sample): a real
              spectrum, so in distribution; permutation importance (Breiman 2001)
    remove    truly remove it (ModalFormer.remove_lasers / remove_bands: masked out of the attention and the band gate, its slots
              zeroed): only in distribution for a run trained with --laser-drop / --band-drop
  metrics: damage = how much worse, so > 0 = the model needs it, for every metric: iou@0.08 and soft iou (drop), miss@0.08 (rise in
    the miss rate), com_err (rise in centroid error, box fractions), orient_err (rise in principal-axis error, deg; elongated objects)
  marginal  every laser (100) and every band (19 x 64 bins ~49 Hz, the ModalFormer's band tokens) alone, all CORE + HARD samples
  curves    per box, bands / lasers ranked by their marginal soft-iou damage: insertion = keep only the top k (the rest taken away),
            deletion = take the top k away, and the same for one random order (the reference a good ranking must beat)
  joint     every (laser, band) pair, 1900 occlusions, on a balanced subsample (--n-joint per box): the laser x frequency heatmap
  ig        (mean baseline only) integrated gradients of soft iou (box-mean baseline, --ig-steps) at full resolution (100 lasers x 1170
            bins): signed and |.|, averaged per box -- cheap, fine-grained, but gradient-based (a cross-check, not the headline)
  .venv/bin/python src8/importance.py RUN [--baseline mean|resample|remove] [--n-joint 200]   -> runs/<RUN>/importance[_<baseline>].pt"""
import argparse, sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from probe import ROOT, eval_data, load_model, box_index, run_args
from arch import moments, orientation, eccentricity

CORE = ("candle", "cube", "soap-dispenser", "vase")
BAND = 64  # bins per band (~49 Hz), as the ModalFormer stem's tokens
METRICS = ["iou@0.08", "soft_iou", "miss@0.08", "com_err", "orient_err"]
HIGHER = {"iou@0.08", "soft_iou"}  # higher is better; damage = base - occluded for these, occluded - base for the rest
LASER_K = [0, 1, 2, 3, 5, 7, 10, 15, 20, 30, 40, 50, 60, 70, 80, 90, 100]

def scores(m, X, box, y, bs=48):
    """dict metric -> (N,) per-sample values; NaN where a metric does not apply (orient_err: round objects)"""
    out = {k: [] for k in METRICS}
    with torch.no_grad():
        for i in range(0, len(X), bs):
            with torch.autocast("cuda", torch.bfloat16):
                p = m(dict(fft=X[i:i + bs].cuda().float(), box=box[i:i + bs].cuda()))["mask_pred"].float()
            t = y[i:i + bs].cuda()
            o, b = t > 0.5, p > 0.08
            out["iou@0.08"].append((b & o).flatten(1).sum(1) / (b | o).flatten(1).sum(1).clamp_min(1))
            out["soft_iou"].append(torch.minimum(p, t).flatten(1).sum(1) / torch.maximum(p, t).flatten(1).sum(1).clamp_min(1e-6))
            out["miss@0.08"].append((~(b & o).flatten(1).any(1)).float())
            _, cp, vp = moments(p); _, ct, vt = moments(t)
            out["com_err"].append((cp - ct).norm(dim=1))
            dth = (orientation(vp) - orientation(vt)).abs() % torch.pi
            out["orient_err"].append(torch.minimum(dth, torch.pi - dth).rad2deg().where(eccentricity(vt) > 0.2, torch.nan))
    return torch.stack([torch.cat(out[k]).cpu() for k in METRICS])                               # (M,N)

def per_box(v, box, n_box):
    """(M,N) -> (M,nb) nan-mean per box"""
    return torch.stack([v[:, box == b].nanmean(1) for b in range(n_box)], 1)

SIGN = torch.tensor([1.0 if k in HIGHER else -1.0 for k in METRICS])[:, None]
def damage(base, occ, box, n_box): return per_box((base - occ) * SIGN, box, n_box)           # (M,nb), > 0 = worse without it

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("run"); ap.add_argument("--n-joint", type=int, default=200); ap.add_argument("--ig-steps", type=int, default=16)
    ap.add_argument("--baseline", default="mean", choices=("mean", "resample", "remove"))
    a = ap.parse_args()
    ra = run_args(a.run)
    X, y, labels, info = eval_data("gastronorm_plastic_cardboard", ra.signal, ra.window, laser_center=bool(getattr(ra, "laser_center", 0)))
    keep = torch.tensor([l.split("/")[1] in CORE or l == "cardboard/soap-dispenser-rotated" for l in labels])
    labels = [l for l, k in zip(labels, keep) if k]
    X, y, box = X[keep], y[keep], box_index(labels, info)
    nb, L, F_ = len(info["boxes"]), X.shape[1], X.shape[2]
    P = -(-F_ // BAND)
    g = torch.Generator().manual_seed(0)
    mean = torch.stack([X[box == b].float().mean(0) for b in range(nb)]).half()                # (nb,L,F,C) the mean baseline
    partner = torch.empty(len(X), dtype=torch.long)                                            # resample: another sample, same box
    for b in range(nb):
        idx = torch.nonzero(box == b)[:, 0]; perm = idx[torch.randperm(len(idx), generator=g)]
        partner[perm] = perm.roll(1)                                                           # a cycle: never the sample itself
    ck = sorted((ROOT / "runs" / a.run / "checkpoints").glob("ep*-rank0.pt"), key=lambda p: int(p.name[2:].split("-")[0]))[-1]
    m = load_model("src8", a.run, ck, info, {}).eval()
    enc = next(mod for mod in m.modules() if type(mod).__name__ == "ModalFormer")
    allx = torch.arange(len(X))

    def occluded(idx, lasers=None, bands=None, bs=48):
        """(M,len(idx)) scores of samples idx with lasers (L,) / bands (P,) or (L,P) bool masks taken away by a.baseline"""
        if a.baseline == "remove":
            enc.remove_lasers, enc.remove_bands = lasers, bands
            try: return scores(m, X[idx], box[idx], y[idx], bs)
            finally: enc.remove_lasers = enc.remove_bands = None
        Z = X[idx].clone(); src = mean[box[idx]] if a.baseline == "mean" else X[partner[idx]]
        if lasers is not None: Z[:, lasers] = src[:, lasers]
        if bands is not None:
            cell = (bands if bands.dim() == 2 else bands[None].expand(L, P)).repeat_interleave(BAND, 1)[:, :F_]  # (L,F)
            Z[:, cell] = src[:, cell]
        return scores(m, Z, box[idx], y[idx], bs)
    base = scores(m, X, box, y)
    print(f"{a.run} {ck.name} baseline={a.baseline}: {len(X)} CORE + HARD samples; base " + ", ".join(f"{k} {v.nanmean():.3f}" for k, v in zip(METRICS, base)), flush=True)

    # marginal: one band / one laser at a time, every sample
    onehot = lambda n, k: torch.arange(n) == k
    band = torch.stack([damage(base, occluded(allx, bands=onehot(P, j)), box, nb) for j in range(P)], -1)     # (M,nb,P)
    laser = torch.stack([damage(base, occluded(allx, lasers=onehot(L, l)), box, nb) for l in range(L)], -1)   # (M,nb,L)
    print("marginal done", flush=True)

    # curves: per box, the absolute metric with only the top-k kept (insertion) / the top-k taken away (deletion), ranked by soft-iou damage
    band_k, si = list(range(P + 1)), METRICS.index("soft_iou")
    curves = {f"{kind}_{order}_{mode}": torch.zeros(len(METRICS), nb, len(ks)) for kind, ks in (("band", band_k), ("laser", LASER_K))
              for order in ("ranked", "random") for mode in ("ins", "del")}
    for b in range(nb):
        idx = torch.nonzero(box == b)[:, 0]
        for kind, n, ks, imp in (("band", P, band_k, band[si, b]), ("laser", L, LASER_K, laser[si, b])):
            for order, rank in (("ranked", imp.argsort(descending=True)), ("random", torch.randperm(n, generator=g))):
                for i, k in enumerate(ks):
                    top = torch.zeros(n, dtype=torch.bool); top[rank[:k]] = True
                    for mode, away in (("ins", ~top), ("del", top)):
                        v = base[:, idx] if not away.any() else occluded(idx, **{kind + "s": away})
                        curves[f"{kind}_{order}_{mode}"][:, b, i] = v.nanmean(1)
        print(f"curves: box {b + 1}/{nb}", flush=True)

    # joint: (laser, band) pairs on a balanced subsample
    sub = torch.cat([torch.nonzero(box == b)[:, 0][torch.randperm(int((box == b).sum()), generator=g)[:a.n_joint]] for b in range(nb)])
    bs_ = box[sub]
    sbase = base[:, sub]
    joint = torch.zeros(len(METRICS), nb, L, P)
    for l in range(L):
        for j in range(P):
            joint[:, :, l, j] = damage(sbase, occluded(sub, bands=(torch.arange(L)[:, None] == l) & onehot(P, j)[None], bs=100), bs_, nb)
        if l % 10 == 9: print(f"joint: laser {l + 1}/{L}", flush=True)

    out = dict(metrics=METRICS, baseline=a.baseline, boxes=info["boxes"], grid=(info["n_laser_rows"], info["n_laser_cols"]),
               base=per_box(base, box, nb), n=torch.stack([(box == b).sum() for b in range(nb)]), band=band, laser=laser,
               curves=curves, band_k=band_k, laser_k=LASER_K, joint=joint, joint_base=per_box(sbase, bs_, nb), n_joint=a.n_joint, ckpt=ck.name)

    # integrated gradients of soft iou, full resolution (box-mean baseline)
    if a.baseline == "mean":
        Xs, ys = X[sub], y[sub]
        ig_signed, ig_abs = torch.zeros(nb, L, F_), torch.zeros(nb, L, F_)
        for k in range(0, len(Xs), 16):
            x, b0, t = Xs[k:k + 16].cuda().float(), mean[bs_[k:k + 16]].cuda().float(), ys[k:k + 16].cuda()
            tot = torch.zeros_like(x)
            for st in range(1, a.ig_steps + 1):
                xi = (b0 + st / a.ig_steps * (x - b0)).requires_grad_(True)
                with torch.autocast("cuda", torch.bfloat16):
                    p = m(dict(fft=xi, box=bs_[k:k + 16].cuda()))["mask_pred"].float()
                soft = (torch.minimum(p, t).flatten(1).sum(1) / torch.maximum(p, t).flatten(1).sum(1).clamp_min(1e-6)).sum()
                tot += torch.autograd.grad(soft, xi)[0]
            att = ((x - b0) * tot / a.ig_steps).sum(-1).cpu()                                      # (n,L,F), summed over axes
            for bb in range(nb):
                w = bs_[k:k + 16] == bb
                ig_signed[bb] += att[w].sum(0); ig_abs[bb] += att[w].abs().sum(0)
        cnt = torch.stack([(bs_ == bb).sum() for bb in range(nb)]).float()[:, None, None]
        out |= dict(ig_signed=ig_signed / cnt, ig_abs=ig_abs / cnt)

    lo, hi = info["freq_range"]
    out |= dict(hz=torch.tensor([lo + (j * BAND + BAND / 2) * (hi - lo) / F_ for j in range(P)]), bin_hz=torch.linspace(lo, hi, F_))
    f = "importance.pt" if a.baseline == "mean" else f"importance_{a.baseline}.pt"
    torch.save(out, ROOT / "runs" / a.run / f)
    print(f"wrote runs/{a.run}/{f}", flush=True)

if __name__ == "__main__":
    main()
