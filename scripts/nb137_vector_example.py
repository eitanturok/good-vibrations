# one example as vectors: pair ≈ a·cube + b·vase + c, with a, b scalars and c one vector ("avg pair interaction") fit by least squares on the
# OTHER pairs of the same box & speaker (even positions); drawn tip-to-tail in a LINEAR 2-D projection (no mean subtraction, so sums stay sums).
import sys, numpy as np, pandas as pd, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src9")); C = ROOT / "runs/_src9_cache/nb137/decres-rescale-aug10-v1"
from dataset import load_splits
exec(open(ROOT / "scripts/nb137_additive.py").read().split("cent = lambda")[0].split("from dataset import load_splits")[1].replace("PAIR = sys.argv[1] if len(sys.argv) > 1 else", "PAIR ="), globals())
BOX, SPK = "plastic", "1"
h = R[(R.box == BOX) & (R.speaker == SPK)].reset_index(drop=True); Y, A, B = Z[h.i], Z[h.m0], Z[h.m1]; tr = h.pos.values % 2 == 0
# fit pair = a·A + b·B + c  (c = 512-d vector): with an intercept this is exactly "subtract means, fit slopes, c = Ȳ − aĀ − bB̄"
Ac, Bc, Yc = A[tr] - A[tr].mean(0), B[tr] - B[tr].mean(0), Y[tr] - Y[tr].mean(0)
a, b = np.linalg.lstsq(np.c_[Ac.reshape(-1), Bc.reshape(-1)], Yc.reshape(-1), rcond=None)[0]; c = Y[tr].mean(0) - a * A[tr].mean(0) - b * B[tr].mean(0)
pred = a * A + b * B + c
mu = Y[tr].mean(0); r2 = lambda T, P: 1 - ((T - P) ** 2).sum() / ((T - mu) ** 2).sum()
print(f"a {a:.2f} b {b:.2f} |c| {np.linalg.norm(c):.1f} vs |pair| {np.linalg.norm(Y, axis=1).mean():.1f}; held-out R² {r2(Y[~tr], pred[~tr]):.2f} (predicting the mean pair = 0)")
te = np.where(~tr)[0]; cosd = [np.dot(Y[i] - Y[tr].mean(0), pred[i] - Y[tr].mean(0)) / np.linalg.norm(Y[i] - Y[tr].mean(0)) / np.linalg.norm(pred[i] - Y[tr].mean(0)) for i in te]
k = te[int(np.argmax(cosd))]; err = np.linalg.norm(Y[k] - pred[k]) / np.linalg.norm(Y[k])
# linear projection: top-2 right singular vectors of the stacked raw vectors (uncentred), so P(a·A + b·B + c) = a·P(A) + b·P(B) + P(c)
V = np.linalg.svd(np.r_[A, B, Y], full_matrices=False)[2][:2].T; p = lambda x: x @ V
f, ax = plt.subplots(1, 2, figsize=(14, 6.5))
for a_, zoom in zip(ax, (False, True)):
    O = np.zeros(2); aA, bB = p(a * A[k]), p(b * B[k])
    for s, e, col, lab in ((O, aA, "tab:red", f"a·cube  (a = {a:.2f})"), (aA, aA + bB, "tab:blue", f"+ b·vase  (b = {b:.2f})"), (aA + bB, aA + bB + p(c), "tab:purple", "+ c  (avg pair interaction)")):
        a_.annotate("", e, s, arrowprops=dict(arrowstyle="-|>", color=col, lw=2.2)); a_.plot([], [], c=col, lw=2.2, label=lab)
    for v, col, lab, mk in ((p(A[k]), "tab:red", "cube (single)", "^"), (p(B[k]), "tab:blue", "vase (single)", "s"), (p(Y[k]), "k", "real pair", "o"), (p(pred[k]), "tab:green", "predicted pair", "*")):
        a_.scatter(*v, s=160, c=col, marker=mk, label=lab, zorder=5)
    a_.plot(*np.c_[p(Y[k]), p(pred[k])], c="tab:green", ls=":", lw=1.5)
    if zoom: m = np.r_[[p(Y[k]), p(pred[k])]]; ctr = m.mean(0); r = max(np.ptp(m, 0).max() * 1.5, 1); a_.set_xlim(ctr[0] - r, ctr[0] + r); a_.set_ylim(ctr[1] - r, ctr[1] + r)
    a_.axhline(0, c="0.85", lw=.6); a_.axvline(0, c="0.85", lw=.6); a_.set_xlabel("component 1 (uncentred, linear)"); a_.set_ylabel("component 2"); a_.set_aspect("equal", "datalim")
    a_.set_title("zoom: real vs predicted pair" if zoom else f"{PAIR} · {BOX} · speaker {SPK} · held-out pos {h.pos[k]}\npair ≈ a·cube + b·vase + c · error ‖real − pred‖/‖real‖ = {err:.2f}", fontsize=10)
ax[0].legend(fontsize=9, loc="best")
plt.tight_layout(); out = ROOT / f"notebooks/out/137_vector_example_{PAIR}.pdf"; plt.savefig(out); plt.savefig(sys.argv[1], dpi=60) if len(sys.argv) > 1 else None; print(out)
