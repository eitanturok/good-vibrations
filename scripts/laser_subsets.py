# laser subsets for the laser-count ablation (scripts/laser_ablation.sh) -> runs/laser_ablation/subsets.json {strategy: {K: [laser ids]}}
# laser id = row*10 + col on the 10x10 grid
# random: uniform, seed K | importance: top-K mean train ΔIoU (nb140 cache of decres-rescale-aug10-v1, train subset720)
# grid: regular r x c lattice, r*c = K as square as possible, rows/cols at cell centres floor((i+.5)*10/r); K=5 = 2x2 lattice + centre, K=1 = centre (4,4)
import json, numpy as np
KS = (80, 60, 40, 20, 10, 5, 1)
LATTICE = {80: (8, 10), 60: (6, 10), 40: (5, 8), 20: (4, 5), 10: (2, 5), 5: (2, 2), 1: (0, 0)}
def lattice(k):
    r, c = LATTICE[k]
    ids = [int((i + .5) * 10 / r) * 10 + int((j + .5) * 10 / c) for i in range(r) for j in range(c)]
    return sorted(ids + [44] * (k - r * c))

imp = np.load("runs/_src9_cache/nb140/train.npz")["mask_l"].mean(0)              # (100,) mean ΔIoU when the laser is dropped
out = {"random": {k: sorted(np.random.default_rng(k).choice(100, k, replace=False).tolist()) for k in KS},
       "importance": {k: sorted(np.argsort(-imp)[:k].tolist()) for k in KS},
       "grid": {k: lattice(k) for k in KS},
       "importance_score": imp.tolist()}
for s in ("random", "importance", "grid"): assert all(len(set(out[s][k])) == k for k in KS), s
json.dump(out, open("runs/laser_ablation/subsets.json", "w"))
for k in KS:
    g = np.zeros(100, int); g[out["grid"][k]] = 1; print(k); print(g.reshape(10, 10))
