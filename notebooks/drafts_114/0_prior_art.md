# nb114 prior art: what to reuse for s9-transformer-vase-1k

Cell indices are 0-based positions in `nb['cells']` (markdown cells count too). "verbatim" = copied from the source. Anything marked NEW was written for this note. It is not existing code.

**This is reference material. None of it is copy-paste-ready code.** Older notebook code can be wrong or more complicated than it needs to be. Each snippet has a **Check:** line with my read on whether it is correct, plus any bugs or simpler ways to write it. Write nb114's code fresh, using these only as a guide.

---

## 0. Two traps (both checked on 2026-10-28)

**(a) The checkpoint does not load into `src9/arch/transformer.py` as it is now.** The run was trained on an intermediate version of the file. Its params are flat (`encoder.freqs.*`, `encoder.lasers.*`, `encoder.band_emb.*`, `encoder.pool_*`, `encoder.embed`, `encoder.box_emb`, `encoder.to_emb`). It already has `cum_mean`/`cum_std` buffers, while HEAD has `in_mean/in_sq/in_n`. The working tree nests these under `freq_enc` / `laser_enc`. With the remap below, `strict=True` reports `<All keys matched successfully>` (NEW, verified):

```python
import re, sys, torch
sys.path.insert(0, str(ROOT / "src9"))
import dataset as D9, utils as U9
from arch.transformer import Transformer

RUN, CKPT = "s9-transformer-vase-1k", "ep1000-ba10000-rank0.pt"   # also ep250/500/750
sd = torch.load(ROOT / f"runs/{RUN}/checkpoints/{CKPT}", map_location="cpu", weights_only=False)["state"]["model"]
REMAP = [(r"^encoder\.(embed|box_emb|freqs|pool_q|pool_norm|pool_k|pool_o)", r"encoder.freq_enc.\1"),
         (r"^encoder\.band_emb", "encoder.freq_enc.freq_band_emb"),
         (r"^encoder\.(lasers|to_emb)", r"encoder.laser_enc.\1")]
def remap(k):
    k = k.replace("_orig_mod.", "")
    for a, b in REMAP: k = re.sub(a, b, k)
    return k
sd = {remap(k): v for k, v in sd.items()}
```
**Check:** the key match is verified, but numeric equality is not. Confirm with the saved-outputs comparison in §1 before trusting anything downstream. An alternative is to `git stash` or check out the matching transformer.py, but no commit matches this checkpoint exactly, so the remap is simpler.

**(b) `data_info` must be the real one.** `freq_band_logf` and `laser_fourier` are **non-persistent** buffers built from `freq_range` (100-1000 Hz) and the grid. So rebuild `info` the way `build_dataset` does (`src9/dataset.py:275`). This does not need the 15 GB MDS. Leave `cum_mean`/`cum_var` as placeholders, because the checkpoint overwrites `cum_mean`/`cum_std`:

```python
samples, idxs = D9.load_splits()                     # ~fast, no spectra; prints the split table
dirs, metas = list(samples), list(samples.values())
info = dict(n_laser_rows=metas[0]["n_rows"], n_laser_cols=metas[0]["n_cols"], n_freqs=1170, n_channels=2,
            freq_range=(float(metas[0]["min_freq"]), float(metas[0]["max_freq"])),
            objects=sorted({"+".join(sorted(metas[i]["objects"])) for i in idxs["train"]}) + ["unseen"],
            n_objects=sorted({m["n_objects"] for m in metas}), boxes=sorted({m["box"] for m in metas}),
            cum_mean=sd["encoder.cum_mean"], cum_var=sd["encoder.cum_std"].pow(-2))
model = Transformer(info); model.load_state_dict(sd, strict=True); model = model.cuda().eval()
for p in model.parameters(): p.requires_grad_(False)
```
In the checkpoint, the head sizes are object 12 (= `n_known`), n_objects 3, com 4 (K=2), box 3. Assert that `len(info["objects"]) - 1 == 12`.
**Check:** this mirrors `build_dataset` and looks correct. The object order only matters for decoding `object_logits`. `cum_var=...pow(-2)` is only there to satisfy the constructor, because `load_state_dict` overwrites `cum_std`. `n_freqs=1170` is hardcoded, which is fine for these 3 experiments.

---

## 1. Inputs and targets for a subset of samples (src9)

`src9/dataset.py` (verbatim signatures):
- `process_x(samples: dict[Path, dict], window)` is a generator over (L=100, F=1170, C=2) float32 values, each `|fft| / std` per sample. Pass `"tukey"` (args.json `window`). It recomputes the FFT from `02_raw_shifts.npy`, so it is slow per sample.
- `process_y(samples, objects, boxes, max_objects, h, w)` returns dicts with `y`, `inst`, `object`, `box`, `coms`. Use 64×64.

The batch keys the model reads are `fft` (B,L,F,C) and `box` (B,) long. `mask_true`, `inst`, and so on are only needed for the loss.

Sample-picking pattern: **nb104 c4** (src7, adapt the imports). Verbatim:
```python
rng = random.Random(SEED)
PICK = []
for b in BOXES:
    ids = idxs[f"eval/{b}/{OBJECT}"]
    for pos in rng.sample(sorted({metas[i]["position_id"] for i in ids}), N_PER_BOX):
        PICK.append(rng.choice([i for i in ids if metas[i]["position_id"] == pos]))
...
X = torch.from_numpy(np.stack(px(PICK))).cuda()                                                     # (S,L,F,C)
```
For src9: `px = lambda ids: list(D9.process_x({dirs[i]: metas[i] for i in ids}, "tukey"))`, and `box = torch.tensor([info["boxes"].index(metas[i]["box"]) for i in ids])`.
**Check:** the picking logic is correct: one random speaker per sampled position. The cost is `process_x`, which recomputes a tukey FFT per sample. For the embedding sections, which need thousands of samples, cache X to disk once as fp16. Don't call `process_x` repeatedly. nb104 also built `REF` from 64 extra train samples per box, which is unnecessary in src9 because `model.encoder.cum_mean` is the exact train mean.

Split names are `eval/<box>/<split>`, with box in gastro/plastic/cardboard. The held-out vase region is `eval/<box>/unseen-position` (`D9.REGION = (0.4, 0.6, 0.3, 0.5)`). Categories are in `D9.CATEGORIES`, and **nb113 c2** shows `CAT = {(b, s): c for b, cs in D9.CATEGORIES.items() for c, ss in cs.items() for s in ss}`.

**Sanity check against the saved eval outputs** (nb104 c4, verbatim pattern). `runs/<RUN>/outputs_history/eval/<box>/<split>/ep{E:04d}-ba*.pt` holds `mask_pred` (fp16), the head outputs, and `info{experiment, sample_id, position_id, speaker}` (`src9/callbacks.py OutputSaver`):
```python
saved = {}
for f in (REPO / f"runs/{RUN}/outputs_history/eval").rglob(f"ep{EPOCH:04d}-*.pt"):
    d = torch.load(f, weights_only=False)
    for p, e, pos, spk in zip(d["mask_pred"], d["info"]["experiment"], d["info"]["position_id"].tolist(), d["info"]["speaker"].tolist()): saved[e, pos, spk] = p.float()
diff = max((P0[s].cpu() - saved[metas[i]["experiment"], metas[i]["position_id"], metas[i]["speaker"]]).abs().max().item() for s, i in enumerate(PICK))
```
**Check:** correct. It loads every eval file for the epoch, which is a few hundred small files and fine. Saved preds are fp16 from a bf16 forward, so expect a max diff around 1e-2, not 0. A large diff means a wrong load or wrong `info`.
Metadata for colouring each sample: `metas[i]` has `box, experiment, layout, objects (dict), n_objects, speaker, position_id, sample_id, crop_overhead_shape, coms`. `fps`, `min_freq` and `max_freq` are also there.

---

## 2. Metrics

`src9/utils.py` (verbatim). These take `out` and `batch` dicts and return per-sample values. Samples with an empty mask are dropped:
```python
THR = 0.08  # the threshold the goal is scored at
def iou(out, batch): return torch.minimum(out["mask_pred"], batch["mask_true"]).flatten(1).sum(1) / torch.maximum(out["mask_pred"], batch["mask_true"]).flatten(1).sum(1).clamp_min(1e-6)  # soft
def iou08(out, batch):
    b, o = out["mask_pred"] > THR, batch["mask_true"] > 0.5
    return ((b & o).flatten(1).sum(1) / (b | o).flatten(1).sum(1).clamp_min(1)).float()[o.flatten(1).any(1)]
def miss08(out, batch):
    b, o = out["mask_pred"] > THR, batch["mask_true"] > 0.5
    return (~(b & o).flatten(1).any(1)).float()[o.flatten(1).any(1)]
```
**Check:** correct, and these are the training metrics, so they are the ones to use. Watch out: `iou08` and `miss08` **drop empty-mask samples**, so the output length can be shorter than the batch. Filter empties out up front (or keep an index) before aligning per-sample results with metadata.
Also in `utils.py`: `moments(m)` → (area, com (row,col), cov), `orientation`, `eccentricity`, `com_err`, `orient_err`, and `METRICS` dict.

All metrics at once, per sample, with NaN where a metric doesn't apply: **`src8/importance.py:scores`**. It needs `from utils import moments, orientation, eccentricity` instead of `arch`. Verbatim core:
```python
            o, b = t > 0.5, p > 0.08
            out["iou@0.08"].append((b & o).flatten(1).sum(1) / (b | o).flatten(1).sum(1).clamp_min(1))
            out["soft_iou"].append(torch.minimum(p, t).flatten(1).sum(1) / torch.maximum(p, t).flatten(1).sum(1).clamp_min(1e-6))
            out["miss@0.08"].append((~(b & o).flatten(1).any(1)).float())
            _, cp, vp = moments(p); _, ct, vt = moments(t)
            out["com_err"].append((cp - ct).norm(dim=1))
            dth = (orientation(vp) - orientation(vt)).abs() % torch.pi
            out["orient_err"].append(torch.minimum(dth, torch.pi - dth).rad2deg().where(eccentricity(vt) > 0.2, torch.nan))
```
**Check:** mostly correct, but unlike `utils.iou08` it does **not** drop empty-mask samples, so an empty box scores iou 0 and miss 1. Restrict to non-empty targets first. `com_err` for an empty *prediction* is meaningless (`moments` clamps the area to 1e-3). Also note that this `iou@0.08` divides by the union of binarized masks and soft_iou divides by the soft max, which are different denominators.
Scores per run from the saved outputs: `src9/score.py` has `per_sample(run, epoch, thr)`, which returns a df with box/split/iou/miss/orient. It is used by nb113 c2 with `score.last_epoch(run)`.

Batched predict under bf16 (nb104 c4, verbatim). For src9, pass `box` too:
```python
@torch.no_grad()
def predict(x):  # (N,L,F,C) -> (N,H,W) in [0,1]
    with torch.autocast("cuda", torch.bfloat16):
        return torch.cat([model(dict(fft=x[k:k + CHUNK]))["mask_pred"].float() for k in range(0, len(x), CHUNK)])
```
**Check:** correct for src7. **It will crash on src9** without `box=b[k:k + CHUNK]`, because the src9 encoder indexes `cum_mean[box]`.

---

## 3. Speckle and connected components

**`src8/score.py:120`** (verbatim). A phantom is a blob that touches no object:
```python
                lab, n = ndimage.label(b)
                r[f"phantom@{t}"] = float(n > 0 and len(set(range(1, n + 1)) - set(np.unique(lab[obj]))) > 0)  # a blob touching no object
```
**nb109 c70**: centres of objects, using components with ≥10% of the mask area (verbatim):
```python
def centres(m):
    lab, n = ndimage.label(m > .5)
    area = np.bincount(lab.ravel())[1:]
    return np.array([[(np.nonzero(lab == k + 1)[0].mean() + .5) / m.shape[0], (np.nonzero(lab == k + 1)[1].mean() + .5) / m.shape[1]]
                     for k in range(n) if area[k] >= .1 * area.sum()])
```
**Check:** both are correct but more complicated than needed.
- phantom: the set-difference can be `n > len(np.unique(lab[obj & (lab > 0)]))`.
- centres: it scans the whole image once per component. `ndimage.center_of_mass(m > .5, lab, range(1, n + 1))` does the same thing more simply.
- `ndimage.label` defaults to 4-connectivity. Pass `structure=np.ones((3, 3))` if diagonal touches should count as one blob. Either way, decide and state it.

`src8/multi_object.py` also has `n_blobs` / `phantom` columns. Prior result (memory `project_no_speckle_mass_collapse`): predictions were 1-2 clean blobs, and the failure was under-painting rather than speckle.

---

## 4. Embeddings: hooks, PCA, t-SNE, UMAP, 3D plotly with a dropdown

**Where the representations are in the src9 Transformer** (from `src9/arch/transformer.py`):
- input: `model.encoder.normalize(x, box)` gives (B,100,1170,2), the per-box standardized spectrum the model actually sees. Raw `x` is also an option.
- per-laser tokens: the output of `model.encoder.freq_enc`, (B,L=100,d=128), and the output of the last `laser_enc.lasers[-1]` block, (B,100,128).
- **latent**: `model.encoder` output (B,1024). This is `emb`, which feeds the decoder and every aux head.
- **penultimate**: `model.decoder.conv2` output (B,32,64,64) before ReLU, the input to `conv3`. Another option is `decoder.ups[-1]` (B,32,64,64).
- There is **no CLS token**. `LaserEncoder.to_emb` = LayerNorm → Flatten(L*d) → Linear(→1024).

Forward-hook pattern, **nb66 c11** (verbatim):
```python
cls_buffer = {}
def _hook(module, args, output):
    cls_buffer['cls'] = output[:, 0, :].detach()   # (B,L+1,D) -> (B,D), same slice arch.py takes
handle = model.laser_encoder.register_forward_hook(_hook)
...
try:
    parts = {split: sweep(loader, split) for split, loader in loaders.items()}
finally:
    handle.remove()   # always detach the hook, even if a sweep raises
```
**Check:** the pattern is correct. The `[:, 0, :]` slice is specific to the old CLS model; for src9, take `output` directly. Simpler for the latent: no hook is needed, because `emb = model.encoder(x, box)` can be called directly, and the decoder can be run manually up to `conv2`.
nb67 c19 hooks a penultimate layer the same way: `model.decoder.net[last_i - 1].register_forward_hook(_hook)`.

**Reducers, nb66 c2 and c15.** `sklearn.manifold.TSNE`, `umap` and `plotly` are all installed in `.venv`. nb66 c15 has `standardize(X)` (train stats, in-place float32), `torch_pca(Xz, k, fit_mask)` (randomized `torch.pca_lowrank`, GPU with a fallback to CPU on OOM or cuSOLVER errors), and `reduce(X, name)`: standardize → pre-PCA to 50 → TSNE (fit on everything) or UMAP (fit on train, then transform). Core (verbatim):
```python
def _torch_pca_on(Xz, k, fit_mask, dev):
    T = torch.from_numpy(Xz).to(dev)
    try:
        fit_rows = T[torch.from_numpy(fit_mask).to(dev)]
        mean = fit_rows.mean(0, keepdim=True)
        _, _, V = torch.pca_lowrank(fit_rows - mean, q=min(k + 10, *fit_rows.shape), center=False)
        return ((T - mean) @ V[:, :k]).cpu().numpy()
    finally:
        del T
        if dev == 'cuda': torch.cuda.empty_cache()
...
    if REDUCER == 'tsne':
        xyz = TSNE(n_components=N_COMPONENTS, **kwargs).fit_transform(Xz)  # no transform(): fit on all
    else:
        xyz = umap.UMAP(n_components=N_COMPONENTS, **kwargs).fit(Xz[train_mask]).transform(Xz)
```
**Check:** it runs, but it is over-engineered, and there are two method issues:
- (1) t-SNE is fit on all points while UMAP and PCA are fit on train only, so the panels are not comparable. Fit everything the same way. For an exploratory plot, fitting on all points is fine.
- (2) Per-feature z-scoring of the 234k-dim input inflates near-silent bins to unit variance. Consider skipping it for the input, or using log magnitude instead.

For the 1024-d latent and the penultimate layer, plain `sklearn.decomposition.PCA` is enough; the GPU/CPU fallback is unnecessary.
Settings: `REDUCER_KWARGS = {'pca': dict(random_state=42), 'umap': dict(n_neighbors=15, min_dist=0.1, metric='euclidean', random_state=42), 'tsne': dict(perplexity=30, init='pca', random_state=42)}`, `PRE_PCA_DIMS = 50`.
RAM: the input is 100·1170·2 = 234k floats per sample, so 3k samples take about 2.8 GB in fp32. Memory `feedback_ram_budget` caps a CPU job at 6 GB or less while training runs.

**3D linked panels with a colour-by dropdown, nb66 c17, c18, c20, c21.** `traces(df, xyz, color_by, primary, com_rng)` builds one `go.Scatter3d` per category, with a shared `legendgroup` across panels. Categories go largest first so rare ones draw on top. A field with more than 24 categories switches to the golden-angle `wide_palette` and drops the legend. Continuous fields use a Viridis colorbar. The same file also has a `com` 2D-hue swatch. Dropdown (nb66 c21, verbatim):
```python
buttons = [dict(label=f'color by {opt}', method='update',
                args=[{'visible': [o == opt for o in trace_opts]},
                      {'legend.title.text': opt, 'xaxis.visible': opt == 'com',
                       'yaxis.visible': opt == 'com', 'shapes': com_shapes if opt == 'com' else []}])
           for opt in COLOR_BY]
fig.update_layout(updatemenus=[dict(buttons=buttons, x=0, xanchor='left', y=1.15, yanchor='top')], ...)
```
Panel build (nb66 c20): `make_subplots(rows=1, cols=n, specs=[[{'type': 'scene'}] * n])`. Every trace is added once per (option, panel), and `trace_opts` records the option of each trace so the dropdown can build its visibility mask. For a white scene background use `fig.update_scenes(xaxis=dict(backgroundcolor='white', gridcolor='#d9d9d9', ...))`. A library version of the same code is at `src/model/pca.py` (`_traces`, `_com_to_hsl`, `_com_swatch`).
**Check:** it works but is heavy. There is one trace per (option × panel × category), plus a com swatch with injected JS, and it runs to about 300 lines. A simpler build: one `Scatter3d` per panel, with a dropdown that restyles `marker.color` (from precomputed colour arrays per option). That fits in about 30 lines and drops the JS. Keep the largest-first draw order and the white scene background, which are the useful ideas.
Palette from nb66 c17: `PALETTE = ['#2a78d6', '#1baf7a', '#eda100', '#008300', '#4a3aa7', '#e34948', '#e87ba4', '#eb6834']`.

---

## 5. Laser and frequency importance by occlusion (token masking)

The src9 tokens are 64-bin bands. That gives **19 bands of about 49 Hz, 1170 bins padded to 1216**, and the same `BAND = 64` as src8. So occluding one token is the same as occluding bins `j*64:(j+1)*64` of one laser, or of all lasers.
**Natural baseline inside the model:** `model.encoder.cum_mean[box]` (L,F,C) is the per-box train mean. Replacing x with it makes the normalized token exactly 0. It is the same idea as nb104's `REF`, and is free here.

**nb104 c7, the occlusion engine** (verbatim). `X` (S,L,F,C), `REF[box]` (L,F,C), and `drop` = loss of soft IoU against the truth:
```python
def occlude(s, lm, bm):  # lm (K,L) bool, bm (K,F) bool -> drop (K,), drift (K,)
    x, ref = X[s], REF[BOX[s]]
    drop, drift = [], []
    for k in range(0, len(lm), CHUNK):
        m = (lm[k:k + CHUNK, :, None] & bm[k:k + CHUNK, None, :])[..., None]                          # (n,L,F,1)
        p = predict(torch.where(m, ref, x))                                                          # (n,H,W)
        drop.append(IOU0[s] - soft_iou(p, Y[s].expand_as(p))); drift.append(1 - soft_iou(p, P0[s].expand_as(p)))
    return torch.cat(drop).cpu().numpy(), torch.cat(drift).cpu().numpy()

EYE_L = torch.eye(L, dtype=torch.bool, device="cuda")
BANDS = torch.from_numpy(BAND[None] == np.arange(NB)[:, None]).cuda()                               # (NB,F)
ALL_L, ALL_F = torch.ones(1, L, dtype=torch.bool, device="cuda"), torch.ones(1, NF, dtype=torch.bool, device="cuda")
for s in range(S):
    a, b = occlude(s, EYE_L, ALL_F.expand(L, -1)); LAS.append(a); LAS_DRIFT.append(b)                  # one laser, every freq
    a, b = occlude(s, ALL_L.expand(NB, -1), BANDS); BND.append(a); BND_DRIFT.append(b)               # one band, every laser
    a, b = occlude(s, EYE_L.repeat_interleave(NB, 0), BANDS.repeat(L, 1))                            # one (laser, band) cell
```
For src9 the batch also needs `box`. Set `BAND = np.arange(1170) // 64` and `NB = 19`.
**Check:** the logic is correct. `drop` is against the truth and `drift` is against the clean prediction, and both are useful. The joint loop does 1900 forwards per sample with the full (n,L,F,1) mask; that is fine but slow. For src9, a cleaner option is masking at the **token** level: in `FreqEncoder`, set the normalized input band to 0 (equivalent to `cum_mean`), or hook `freq_enc.embed`'s output. That is the literal "token masking" and skips the (L,F) masks. nb104's `BAND` formula was for 20 Hz bands; don't reuse it.

**`src8/importance.py`** is the scaled-up version: all CORE+HARD samples, three baselines, every metric, per box. Its `occluded()` (verbatim):
```python
        Z = X[idx].clone(); src = mean[box[idx]] if a.baseline == "mean" else X[partner[idx]]
        if lasers is not None: Z[:, lasers] = src[:, lasers]
        if bands is not None:
            cell = (bands if bands.dim() == 2 else bands[None].expand(L, P)).repeat_interleave(BAND, 1)[:, :F_]  # (L,F)
            Z[:, cell] = src[:, cell]
        return scores(m, Z, box[idx], y[idx], bs)
```
**Check:** correct. Indexing with a 2D (L,F) bool `cell` on dims 1-2 works. Note that the `mean` baseline here is the **eval-set** box mean, not the train mean. In src9, use `cum_mean` (train). The whole script is about 160 lines with three baselines, curves and IG; for nb114, take only the pieces needed.
- `resample` baseline: `partner[perm] = perm.roll(1)` within a box, which is permutation importance and stays in distribution.
- damage is `(base - occ) * SIGN`, with SIGN = +1 for metrics where higher is better.
- It also has top-k insertion and deletion curves against a random order, `LASER_K = [0,1,2,3,5,7,10,15,20,30,40,50,60,70,80,90,100]`.
- Integrated gradients from the box-mean baseline (16 steps) at 100×1170.
- A `remove` baseline: the model needs `remove_lasers/remove_bands` hooks, which src9 lacks.

**nb104 c15, gradient · (mean − x)** (verbatim), a cheap linearized occlusion:
```python
    x = X[s:s + 1].clone().requires_grad_(True)
    loss = F.binary_cross_entropy_with_logits(model(dict(fft=x))["mask_logits"], Y[s:s + 1])
    g, = torch.autograd.grad(loss, x)
    LIN.append((g * (REF[BOX[s]] - x)).sum(-1)[0].detach().cpu().numpy())                          # (L,F), summed over x/y channel
```
**Check:** correct as a first-order estimate of the *BCE* increase. It is not the IoU drop, so the sign and scale differ from the occlusion maps; compare by rank. For src9, pass `box`. With frozen params, only `x` needs grad, which is what it does. Integrated gradients (src8) is the more faithful version of this.
**nb104 c17** has deletion and insertion curves (`curves(s, scores, kind)`, ranked against 5 random orders, AUC via `np.trapezoid`).

**Prior findings:**
- nb104 (src7) and nb106 (src8 ModalFormer): importance is diffuse and varies by sample. One laser drops IoU by about .003, one band by about .012. There is no dominant band, and grad agrees with occlusion only at ρ≈.5-.7.
- Old-config (memory `project_attribution`): CLS attention against ablation had ρ≈0.45. Attention tracked token magnitude, so ablation should lead.

---

## 6. Attention maps, CLS attention, rollout

**Nothing in the repo reuses directly for src9.** `Block` calls `F.scaled_dot_product_attention`, which returns no weights. The old capture code is `src/model/attribution.py:306 capture_attention`. It injects `need_weights=True` into `nn.MultiheadAttention` with a forward pre-hook, which does not apply to src9. Pattern (verbatim):
```python
    def pre(mod, args, kwargs):
        kwargs = dict(kwargs)
        kwargs['need_weights'], kwargs['average_attn_weights'] = True, False
        return args, kwargs
    ...
            handles.append(mha.register_forward_pre_hook(pre, with_kwargs=True))
            handles.append(mha.register_forward_hook(save(axis, i), with_kwargs=True))
```
**For src9 (NEW).** Recompute the weights in a forward hook on each `Block`. The input is `args[0]`, (N,T,D). The math mirrors `Block.forward`:
```python
ATT = {}
def att_hook(name):
    def hook(mod, args, out):
        x = args[0]; N, T, D = x.shape
        q, k, _ = mod.qkv(mod.n1(x)).view(N, T, 3, mod.h, -1).permute(2, 0, 3, 1, 4)    # (N,H,T,D/H)
        ATT[name] = ((q @ k.transpose(-1, -2)) / q.shape[-1] ** 0.5).softmax(-1).detach().float()  # (N,H,T,T)
    return hook
hs = [b.register_forward_hook(att_hook(f"freq{i}")) for i, b in enumerate(model.encoder.freq_enc.freqs)]     # N = B*L, T = 19 bands
hs += [b.register_forward_hook(att_hook(f"laser{i}")) for i, b in enumerate(model.encoder.laser_enc.lasers)]  # N = B, T = 100 lasers
```
**Check (NEW, not run):** this matches `Block.forward`. SDPA uses no mask and its default scale is 1/sqrt(d_head). Run it outside autocast, or expect bf16-level differences. Remove the hooks afterwards (`for h in hs: h.remove()`). The freq maps are (B·100, 4, 19, 19), so reshape with `.view(B, 100, 4, 19, 19)`.
- **Freq-pool gate (the closest thing to CLS attention on the freq axis).** In `FreqEncoder.forward`, `a = softmax(k·pool_q)` has shape (B,L,H=4,P=19) and gives the bands each laser reads. Recompute it the same way from `pool_k(pool_norm(x))` on the output of `freqs[-1]`. src8's docstring warns that gates like this were within 1.5% of uniform in the ModalFormer, and that a following linear layer can rescale anything. Check it against occlusion before trusting it.
- **Laser axis:** there is no CLS. The equivalents are mean attention received per laser, rollout over the 2 laser blocks, or the per-laser norm of `to_emb[2].weight.view(1024, 100, 128)`.
- **Rollout:** there is no implementation anywhere in the repo (grep for "rollout" finds nothing). Standard form: `R = Π_l (0.5·mean_h A_l + 0.5·I)`, renormalized by row.

---

## 7. Grad-CAM

**`src8/gradcam.py`** uses jacobgil `pytorch_grad_cam`, which is installed in `.venv`. Verbatim:
```python
from pytorch_grad_cam import GradCAM, HiResCAM, LayerCAM
class Wrap(torch.nn.Module):
    """(B,C,L,F) -> (B,H,W) mask probabilities, for the box set in .box"""
    def __init__(self, m): super().__init__(); self.m, self.box = m, None
    def forward(self, x): return self.m(dict(fft=x.permute(0, 2, 3, 1), box=self.box))["mask_pred"].float()

class SoftIoU:
    def __init__(self, t): self.t = t
    def __call__(self, p): return torch.minimum(p, self.t).sum() / torch.maximum(p, self.t).sum().clamp_min(1e-6)
...
        with cls(model=w, target_layers=[enc.conv3]) as cam:
            for k in range(0, len(sub), a.bs):
                i = sub[k:k + a.bs]
                w.box = box[i].cuda()
                c = torch.as_tensor(cam(input_tensor=X[i].float().permute(0, 3, 1, 2).cuda(), targets=[SoftIoU(t) for t in y[i].cuda()]))  # (n,L,F) in [0,1]
...
        out[name + "_band"] = torch.stack([acc[:, :, j * BAND:(j + 1) * BAND].mean(-1) for j in range(P)], -1)  # (nb,L,P)
```
**Check:** correct for src8. Wrap's permute is right, and the library calls each target on one sample's output, so `SoftIoU(t)` per sample is right. The library resizes the CAM to the input's (H,W) = (L,F). That was harmless in src8, where only the freq axis was stretched.

For src9 there is no conv in the encoder. Options for a token layer with a `reshape_transform` (NEW):
- target `model.encoder.freq_enc.freqs[-1]` outputs (B·L,19,128). Use `reshape_transform=lambda t: t.view(B, 100, 19, 128).permute(0, 3, 1, 2)`, which gives a (laser×band) map. The library resize then stretches 19 bands to 1170 bins, which is OK.
- target `laser_enc.lasers[-1]` outputs (B,100,128). A `(B,128,10,10)` reshape **does not work with the library**: it would resize a 10×10 map to the (100,1170) input. Use `(B,128,100,1)` instead, or skip the library.

**Simpler and recommended:** do Grad-CAM or HiResCAM by hand. Run one hook plus `torch.autograd.grad(soft_iou.sum(), act)`, then compute `relu((grad * act).sum(-1))` (HiResCAM) or `relu((grad.mean(token_dims) * act).sum(-1))` (GradCAM). That is about 10 lines, with no resize and no per-sample min-max normalization, so magnitudes stay comparable across samples.
Caveats from nb106 c10: CAMs are unsigned and normalized per sample. Compare them by rank, not by value.

---

## 8. Plot helpers: laser grid and bands

- Laser `l` sits at **row `l // 10`, col `l % 10`**. This is `utils.grid_coords`, row-major. Use `a.reshape(10, 10)`.
- In the crop, the lasers are on the bottom wall: row-fraction 1, per `src8/score.py` `wall_dist` and the `D9.REGION` comment.
- nb102's photo pixels are in `notebooks/102_laser_grid_pixels.csv` (`box,row,col,x,y`; boxes plastic/gastro/cardboard; image-space row 0 = top). Load them with `pd.read_csv(ROOT/'notebooks/102_laser_grid_pixels.csv')`. The photos are `assets/{box}_laser_view.jpg`.
- The mapping between photo (row, col) and dataset laser index has not been verified for orientation or mirroring (memory `project_laser_grid_nb102`).

**nb106 c1** helpers (verbatim): `heat(ax, m, lim, f, annotate, fs, cmap, signed)` gives an annotated diverging or sequential heatmap. `tag(ax, b)` puts a box-colour bar on the left spine. `band_axis(ax)` puts band borders in Hz. `topk(...)` plots the curves. Palettes:
```python
BOXC = ['#2a78d6', '#eb6834', '#1baf7a']
DIV = LinearSegmentedColormap.from_list('div', ['#2a78d6', '#f0efec', '#e34948'])
SEQ = LinearSegmentedColormap.from_list('seq', ['#f7fafe', '#cde2fb', '#86b6ef', '#3987e5', '#1c5cab', '#0d366b'])
EDGES = np.array([bhz[min(j * BAND, len(bhz) - 1)] for j in range(P)] + [bhz[-1]])   # band borders in Hz (the last band is short: 18 bins)
```
**Check:** the helpers are correct and the palettes are good to reuse. `fmt` has a hack that strips a leading zero with `.replace('0.', '.', 1)`, which also mangles `-0.5` to `-.5`; that is intended, and harmless. The `heat` annotation loop is slow on 100×19 maps; turn on `annotate=False` for the joint maps.
More from nb106: c3 `band_figure` (one row of 19 bands per box with top-k curves alongside), c5 `laser_figure` (a 10×10 grid per box), c7-9 joint laser×band maps, c11 Grad-CAM figures, c15 Spearman agreement between methods.
More from nb104: c9 `grid_figure(A, title, label)`, which shows per-sample 10×10 maps with the top 3 lasers numbered, plus a box-mean column. c11 has the band heatmap, c13 the joint map.
nb113 c1 sets the src9 colour convention: `BOX_COLOR = {"gastro": "#3B6FB6", "plastic": "#E07B39", "cardboard": "#7A9E4E"}`, `THR = 0.08`.

---

## 9. Other references
- `src8/probe.py`: `eval_data` caches X/y for the eval set in `runs/_probe_cache/*.pt` (src8 splits, not src9), and `load_model` strips `_orig_mod.`.
- `src/model/attribution.py`: `ablate_lasers`, `ablate_freq_patches`, `ablate_laser_freq`, `retain_top_k`, `grad_x_input_lasers` (old arch, zero-ablation).
- nb64 (old arch): attention matrix grids per layer and head (c32 `attn_matrix_grid`, c35 `strip_grid`), CLS against ablation agreement (c41, c53), and top-k by attention against top-k by ablation (c43, c55).
