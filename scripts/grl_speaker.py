"""speaker clustering + task metrics of the latents from scripts/grl_extract.py, one row per tag; UMAP PDFs in plots/grl_speaker/.
  OMP_NUM_THREADS=8 .venv/bin/python scripts/grl_speaker.py decres-rescale-aug10-v1 decres-rescale-aug10-v1-leace grl-adapter-v1
speaker metrics on train speakers (1,3,5,7) only, within each box (speaker is nested in box) then averaged over boxes:
  knn10 = 5-fold kNN (k=10) speaker accuracy on all samples; probe = logistic regression fit on train, scored on eval; sil = silhouette by speaker;
  chance = majority-speaker share. task: per-sample iou08 / miss08 pooled over ID (id+multiple) / OOD categories (= EvalSummary's id/ood iou08)."""
import sys, numpy as np, pandas as pd, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
import umap
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src9"))
from dataset import CATEGORIES

TAGS = sys.argv[1:]
C, OUT = ROOT / "runs/_src9_cache/grl_speaker", ROOT / "plots/grl_speaker"; OUT.mkdir(parents=True, exist_ok=True)
CAT = {f"eval/{b}/{s}": c for b, cs in CATEGORIES.items() for c, ss in cs.items() for s in ss}
ID, OOD = ("id", "multiple"), ("rotated", "cylinder", "unseen-object", "unseen-speaker", "unseen-laser", "lid")
SPK = [1, 3, 5, 7]

rows, emb = [], {}
for tag in TAGS:
    F, df = np.load(C / tag / "feats.npz"), pd.read_csv(C / tag / "meta.csv")
    df["cat"] = df.split.map(CAT); r = dict(tag=tag)
    for g, cats in (("ID", ID), ("OOD", OOD)):
        m = df.cat.isin(cats).values & F["has_obj"]
        r[f"{g} iou08"], r[f"{g} miss08"] = F["iou08"][m].mean(), F["miss08"][m].mean()
    m = df.speaker.isin(SPK).values & ~df.duplicated(["pos", "speaker", "box"]).values  # control splits re-use samples: one copy each
    X, d = F["latent"][m], df[m].reset_index(drop=True)
    per = []
    for b in sorted(d.box.unique()):
        mb = (d.box == b).values; Xb, y = StandardScaler().fit_transform(X[mb]), d.speaker[mb].values; tr = (d.split[mb] == "train").values
        knn = (cross_val_predict(KNeighborsClassifier(10), Xb, y, cv=5) == y).mean()
        probe = LogisticRegression(max_iter=2000, C=1.0).fit(Xb[tr], y[tr]).score(Xb[~tr], y[~tr])
        sil = silhouette_score(Xb, y, sample_size=min(4000, len(y)), random_state=0)
        per.append((knn, probe, sil, np.bincount(y).max() / len(y)))
        r[f"{b} knn10"], r[f"{b} probe"] = knn, probe
    r["knn10"], r["probe"], r["sil"], r["chance"] = np.mean(per, 0)
    rows.append(r); print(r, flush=True)
    # UMAP on 50-d PCA (as nb137), pooled and per box
    Z = PCA(50, random_state=0).fit_transform(StandardScaler().fit_transform(X))
    emb[tag] = dict(d=d, all=umap.UMAP(n_neighbors=30, min_dist=0.1, random_state=0).fit_transform(Z),
                    **{b: umap.UMAP(n_neighbors=30, min_dist=0.1, random_state=0).fit_transform(PCA(50, random_state=0).fit_transform(StandardScaler().fit_transform(X[(d.box == b).values])))
                       for b in sorted(d.box.unique())})

res = pd.DataFrame(rows); res.to_csv(OUT / "metrics.csv", index=False)
print(res.round(3).to_string())

def scatter(ax, P, c, title):
    cats = sorted(pd.unique(c), key=str); cm = plt.get_cmap("tab10" if len(cats) <= 10 else "tab20")
    for k, v in enumerate(cats): s = (c == v); ax.scatter(P[s, 0], P[s, 1], s=1.5, color=cm(k % cm.N), label=str(v), rasterized=True, linewidths=0)
    ax.set_title(title, fontsize=9); ax.set_xticks([]); ax.set_yticks([])
    ax.legend(markerscale=6, fontsize=6, loc="best", frameon=False)

# pooled UMAP: rows = colour by speaker / box / object / train-eval, cols = tags
colours = dict(speaker=lambda d: d.speaker.values, box=lambda d: d.box.values, object=lambda d: d.object.values, set=lambda d: np.where(d.split == "train", "train", "eval"))
fig, axs = plt.subplots(len(colours), len(TAGS), figsize=(6 * len(TAGS), 5.5 * len(colours)), squeeze=False)
for j, tag in enumerate(TAGS):
    for i, (k, f) in enumerate(colours.items()): scatter(axs[i, j], emb[tag]["all"], f(emb[tag]["d"]), f"{tag} | colour = {k}")
fig.tight_layout(); fig.savefig(OUT / "umap_pooled.pdf"); fig.savefig(Path("/tmp/claude-1000/-home-ethantu-workspace-good-vibrations/a2406885-3599-43b4-83b0-2a647916c337/scratchpad/umap_pooled.png"), dpi=50)

# per-box UMAP: rows = box x (speaker, object), cols = tags
boxes = sorted(emb[TAGS[0]]["d"].box.unique())
fig, axs = plt.subplots(2 * len(boxes), len(TAGS), figsize=(6 * len(TAGS), 5.5 * 2 * len(boxes)), squeeze=False)
for j, tag in enumerate(TAGS):
    d = emb[tag]["d"]
    for i, b in enumerate(boxes):
        db = d[d.box == b]
        scatter(axs[2 * i, j], emb[tag][b], db.speaker.values, f"{tag} | {b} | colour = speaker")
        scatter(axs[2 * i + 1, j], emb[tag][b], db.object.values, f"{tag} | {b} | colour = object")
fig.tight_layout(); fig.savefig(OUT / "umap_per_box.pdf"); fig.savefig(Path("/tmp/claude-1000/-home-ethantu-workspace-good-vibrations/a2406885-3599-43b4-83b0-2a647916c337/scratchpad/umap_per_box.png"), dpi=35)
print("saved", OUT)
