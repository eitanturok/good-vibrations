"""Emit notebooks/66_umap_cls.ipynb from readable cell sources."""
import json
from pathlib import Path

cells = []
def md(src): cells.append({'cell_type': 'markdown', 'metadata': {}, 'source': src.strip('\n').splitlines(keepends=True)})
def code(src): cells.append({'cell_type': 'code', 'metadata': {}, 'execution_count': None, 'outputs': [],
                             'source': src.strip('\n').splitlines(keepends=True)})

md(r'''
# 66 Linked 3D embedding: input FFT vs CLS token vs output logits

Pick a **run**; everything else is recovered from its wandb config, so the dataloaders and the
model are rebuilt exactly as they were at training time.

Three panels, side by side, over the same samples -- each its own **independently fit** reduction:

1. **Model input** -- the tokenized FFT the model actually sees, flattened per sample.
2. **CLS token** -- the laser-encoder CLS embedding (`output[:, 0, :]` in
   `VibrationTransformer.forward`), captured with a forward hook on `laser_encoder`, so the model
   code is untouched.
3. **Model output** -- the predicted-mask logits, flattened.

The technique is a config knob (`REDUCER`): `pca` (seconds), `umap` (minutes, nonlinear), or
`tsne`. For pca/umap each reducer is **fit on the train split only** and then `.transform`'d onto
every split, so eval points land in a space defined purely by train data (no eval leakage into the
embedding). A color-by dropdown recolors all three panels at once: `split`, `speaker`, `layout`,
`n_objects`, `position_id` (discrete -- ~376 distinct values, drawn from a generated golden-angle
palette with the legend suppressed, so identity comes from the tooltip), `com` (2D: hue=horizontal,
lightness=vertical), `com_x`, `com_y`.
''')

md('## Config -- set the run name')

code(r'''
RUN_NAME = 'ce-pixel-all-gastronorm-21x30'   # a run under runs/ that also exists in wandb
WANDB_PATH = 'eturok/better-tsa'             # wandb entity/project the run was logged to
CHECKPOINT = 'latest-rank0.pt'               # file under runs/<RUN_NAME>/checkpoints/

REPO = '/home/ethantu/workspace/good-vibrations'
BATCH_SIZE = 128
NUM_WORKERS = 4
DEVICE = 'cuda'      # 'cuda' or 'cpu'

# Dimensionality reduction. With PRE_PCA_DIMS on, measured cost for all three panels on this
# dataset (~3k samples) is ~7s for 'pca' and ~40s for 'umap', so either is interactive; 'pca' is
# the quick sanity check, 'umap' the one that shows nonlinear structure.
REDUCER = 'umap'     # 'pca' | 'umap' | 'tsne'
N_COMPONENTS = 3     # the panels below are 3D scatters

# Passed to the chosen reducer only. UMAP: n_neighbors trades local (small) vs global (large)
# structure, min_dist how tightly points may pack. t-SNE has no transform(), so it's fit on all
# points at once (see the fit cell).
REDUCER_KWARGS = {
    'pca':  dict(random_state=42),
    'umap': dict(n_neighbors=15, min_dist=0.1, metric='euclidean', random_state=42),
    'tsne': dict(perplexity=30, init='pca', random_state=42),
}

# PCA pre-reduction before umap/tsne, or None to disable. The raw FFT is ~256k dims per sample and
# the neighbor-graph build is where nearly all of UMAP's time goes, so projecting to a few dozen
# dims first is the difference between minutes and tens of minutes -- and is standard practice
# (UMAP's own docs recommend it), since PCA at this rank keeps essentially all the variance.
PRE_PCA_DIMS = 50
''')

code(r'''
import sys, json, re, colorsys
from pathlib import Path

sys.path[:0] = [f'{REPO}/src', REPO]   # src/ for model.*, repo root for utils.* (imported by arch)

import numpy as np
import pandas as pd
import torch
import wandb
import umap
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.manifold import TSNE

from model.dataset import build_dataset
from model.arch import VibrationTransformer
''')

md('''
## Recover the run config from wandb

`run.py` logs `data_info | args.__dict__` as the wandb config, so the run's dataset + model
hyperparameters are all there -- nothing about the run needs to be re-specified by hand. Only the
config keys that `build_dataset` / `VibrationTransformer` actually take are pulled out; the rest
(lr, scheduler, ...) don't affect what we're reconstructing.
''')

code(r'''
api = wandb.Api()
cfg = api.run(f'{WANDB_PATH}/{RUN_NAME}').config
print(f"{cfg['data_dir']}  split={cfg['split']}  {cfg['out_h']}x{cfg['out_w']}  "
      f"decoder={cfg['decoder']}  loss={cfg['loss_fn']}  norm={cfg['normalize_mode']}")
''')

md('''
## Rebuild the dataloaders

Same `build_dataset` call `run.py` makes, with the run's own config -- so the MDS hash resolves to
the same cached dataset and the splits come out identical. Augmentation is forced off
(`augment_fft/mask=0`) on *every* loader here including train: we want each sample's true
representation, not a randomly perturbed one, and augmentation would also make the input-FFT panel
non-deterministic. `train_eval_loader` is the un-augmented, un-shuffled, un-dropped train loader,
so it (not `train_loader`) is what we sweep for the train split.
''')

code(r'''
_, eval_loaders, train_eval_loader = build_dataset(
    cfg['data_dir'], split=cfg['split'], batch_size=BATCH_SIZE, eval_batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS, test_size=cfg['test_size'], speakers=cfg['speakers'],
    n_objects=cfg['n_objects'], box=cfg['box'], n_samples=cfg['n_samples'],
    out_h=cfg['out_h'], out_w=cfg['out_w'], signal_mode=cfg['signal_mode'],
    normalize_mode=cfg['normalize_mode'], patch_size=cfg['patch_size'], seed=cfg['seed'],
    augment_fft=0.0, augment_mask=0.0, subtract_speaker_mean=bool(cfg['subtract_speaker_mean']))

# label -> plain torch DataLoader; 'train' first so it's the split UMAP gets fit on. An Evaluator's
# .dataloader is the DataSpec build_dataset made, whose own .dataloader is the real DataLoader.
loaders = {'train': train_eval_loader.dataloader} | {ev.label: ev.dataloader.dataloader for ev in eval_loaders}
{k: len(v.dataset) for k, v in loaders.items()}
''')

md('''
## Rebuild the model + load the checkpoint

`data_info` is reconstructed from the dataset the same way `run.py` does it (`n_freqs` is the
*padded* length `n_patches * patch_size`, not the real bin count). Composer checkpoints nest the
weights under `state.model`; keys are stripped of any `_orig_mod.` prefix left by `torch.compile`.
''')

code(r'''
n_lasers, n_patches, patch_size, n_channels = train_eval_loader.dataloader.dataset[0]['fft'].shape
data_info = dict(out_h=cfg['out_h'], out_w=cfg['out_w'], n_laser_rows=cfg['n_laser_rows'],
                 n_laser_cols=cfg['n_laser_cols'], patch_size=cfg['patch_size'],
                 n_freqs=n_patches * patch_size, n_channels=n_channels)

model = VibrationTransformer(cfg['d_model'], cfg['pnt_num_heads'], cfg['pnt_num_layers'],
                             cfg['seq_num_heads'], cfg['seq_num_layers'], data_info, cfg['decoder'],
                             cfg['decoder_num_heads'], cfg['decoder_num_layers'],
                             freq_dropout=cfg['freq_dropout'], laser_dropout=cfg['laser_dropout'],
                             loss_fn=cfg['loss_fn'])

ckpt_path = Path(REPO) / 'runs' / RUN_NAME / 'checkpoints' / CHECKPOINT
state = torch.load(ckpt_path, map_location='cpu', weights_only=False)['state']['model']
state = {k.removeprefix('_orig_mod.'): v for k, v in state.items()}
missing, unexpected = model.load_state_dict(state, strict=False)
assert not missing and not unexpected, f'{missing=} {unexpected=}'

# .eval() also disables freq_dropout/laser_dropout (both are training-only), so every sample gets a
# clean forward with all lasers and all frequency patches intact
model = model.to(DEVICE).eval()
print(f'loaded {ckpt_path.name}: {sum(p.numel() for p in model.parameters()):,} params')
''')

md('''
## Sweep every split, capturing the CLS token with a forward hook

The CLS token isn't in the model's return dict -- `forward` computes `cls = output[:, 0, :]` from
the laser encoder's output and passes it to the decoder. Rather than edit `arch.py`, a forward hook
on `model.laser_encoder` grabs its output tensor and we take slot 0, which *is* that exact `cls`
tensor. So the three panels come from one forward pass per batch, guaranteed consistent with each
other.

Per sample we keep: the flattened input FFT, the CLS vector, the flattened mask logits, and the
info fields the plots color by.
''')

code(r'''
cls_buffer = {}
def _hook(module, args, output):
    cls_buffer['cls'] = output[:, 0, :].detach()   # (B,L+1,D) -> (B,D), same slice arch.py takes
handle = model.laser_encoder.register_forward_hook(_hook)

@torch.no_grad()
def sweep(loader, split):
    ffts, clss, logits, rows = [], [], [], []
    for batch in loader:
        fft = batch['fft'].to(DEVICE, non_blocking=True)
        # 'fft' is the only key forward() reads (the dataset never emits 'speakers_encoded', so the
        # speaker embedding is off here exactly as it was during training)
        out = model({'fft': fft})
        ffts.append(fft.flatten(1).float().cpu().numpy())
        clss.append(cls_buffer['cls'].flatten(1).float().cpu().numpy())
        logits.append(out['mask_logits'].flatten(1).float().cpu().numpy())
        info = batch['info']
        rows.append(pd.DataFrame({
            'sample_id': np.asarray(info['sample_id']).astype(int),
            'position_id': np.asarray(info['position_id']).astype(int),
            'speaker': np.asarray(info['speaker']).astype(int),
            'n_objects': np.asarray(info['n_objects']).astype(int),
            'com_x': np.asarray(info['x_com']).astype(float),
            'com_y': np.asarray(info['y_com']).astype(float),
            'split': split,
        }))
    return np.concatenate(ffts), np.concatenate(clss), np.concatenate(logits), pd.concat(rows)

try:
    parts = {split: sweep(loader, split) for split, loader in loaders.items()}
finally:
    handle.remove()   # always detach the hook, even if a sweep raises

# stack the splits in loaders order, so rows 0..n_train-1 are exactly the train split
FFT   = np.concatenate([p[0] for p in parts.values()])
CLS   = np.concatenate([p[1] for p in parts.values()])
LOGIT = np.concatenate([p[2] for p in parts.values()])
meta  = pd.concat([p[3] for p in parts.values()], ignore_index=True)
print(f'{len(meta)} samples   fft{FFT.shape[1:]}  cls{CLS.shape[1:]}  logits{LOGIT.shape[1:]}')
''')

md('''
## Merge in `layout` and the center-of-mass from the dataset metadata

`layout` (e.g. `empty-box`, `grid-1`) is only in the MDS `metadata.jsonl` sidecar, not in the
per-batch `info` dict, so it's joined on `sample_id` to make it available as a color-by option.

**The center-of-mass is recovered here too, rather than from the batch `info`.** `dataset.py` builds
`x_com`/`y_com` from a `downsampled_com` key, which *this* metadata does not have -- it stores the
COM under `avg_com` (in full-frame pixels) instead. The `.get(..., [-1.0, -1.0])` default therefore
silently makes every sample look like the "no COM" sentinel, which is why the com color options
disappear even though 2952 of 3007 samples have a perfectly good center-of-mass. Two wrinkles worth
knowing: `avg_com` is stored as a *string* (a numpy `repr` like `'[213.6 405.2]'`), so it has to be
parsed rather than indexed; and the genuine `[-1 -1]` sentinels are exactly the 55 empty-box samples,
which have no objects and so correctly have no COM. Those keep the sentinel and stay on the black
`empty box` trace.
''')

code(r'''
mds_dir = Path(train_eval_loader.dataloader.dataset.dataset.streams[0].local)
rows = [json.loads(l) for l in (mds_dir / 'metadata.jsonl').read_text().strip().splitlines() if l]

def _parse_com(v):
    """`avg_com` -> (x, y). Stored as a numpy repr string ('[213.6 405.2]'), occasionally a list."""
    nums = re.findall(r'-?\d+\.?\d*', v) if isinstance(v, str) else [str(x) for x in v]
    return (float(nums[0]), float(nums[1])) if len(nums) >= 2 else (-1.0, -1.0)

side = pd.DataFrame([{'sample_id': int(r['sample_id']), 'layout': r['layout'],
                      'meta_com_x': _parse_com(r.get('avg_com', [-1.0, -1.0]))[0],
                      'meta_com_y': _parse_com(r.get('avg_com', [-1.0, -1.0]))[1]} for r in rows])
meta = meta.merge(side, on='sample_id', how='left')
meta['layout'] = meta['layout'].fillna('unknown')

# Prefer the metadata COM wherever the dataset's own x_com/y_com came back as the sentinel. If a
# future dataset *does* populate downsampled_com, those values are already correct and are kept.
missing = (meta['com_x'] < 0) | (meta['com_y'] < 0)
meta.loc[missing, 'com_x'] = meta.loc[missing, 'meta_com_x'].fillna(-1.0)
meta.loc[missing, 'com_y'] = meta.loc[missing, 'meta_com_y'].fillna(-1.0)
meta = meta.drop(columns=['meta_com_x', 'meta_com_y'])
n_com = int(((meta['com_x'] >= 0) & (meta['com_y'] >= 0)).sum())
print(f'center-of-mass available for {n_com}/{len(meta)} samples '
      f'({len(meta) - n_com} sentinels -- empty-box samples with no objects)')
meta.head()
''')

md('''
## Fit one reducer per representation, on train, and transform everything

Three **independent** reductions -- one fit per feature space (input FFT, CLS token, output
logits), never one shared reducer. Each `fit` sees only that space's train rows; `transform` then
places *all* rows (train included) into that fitted space, so an eval cluster sitting apart from
train is a real statement about the representation rather than an artifact of eval points having
helped shape the embedding. `transform` on the train rows isn't bit-identical to `fit_transform`,
which is the point: every split goes through the same projection.

The reducer is whatever `REDUCER` names -- the rest of the notebook only ever sees the resulting
`(n_samples, 3)` arrays, so switching technique changes nothing downstream. `pca` runs in seconds
and is the sane first pass; `umap` costs longer but shows nonlinear structure. Notes:

- **PCA runs on the GPU** via `torch.pca_lowrank`, falling back to CPU torch. With ~256k features
  and only ~3k samples, an exact SVD is hopeless; this is a randomized low-rank factorization,
  which is both the right tool at this shape and orders of magnitude faster.
- **`PRE_PCA_DIMS`** projects to that many dims (same GPU path) before umap/tsne. The neighbor-graph
  build dominates UMAP's runtime, so this is the single biggest speed lever there. `None` disables.
- **Memory is the thing to watch**, not flops: the FFT block is ~2959 x 256k, so the standardization
  is done in float32 and in place. Done naively in float64 it peaks near 18GB and swaps, which
  looks exactly like a hang.
- **t-SNE has no `transform`**, so there is no fit-on-train-only option for it: it is fit on all
  points jointly and the train/eval distinction below is descriptive, not held out. PCA and UMAP
  both keep the strict train-only fit.
''')

code(r'''
train_mask = (meta['split'] == 'train').to_numpy()
kwargs = REDUCER_KWARGS[REDUCER]
print(f'{REDUCER}: fitting on {train_mask.sum()} train samples, transforming {len(meta)}')

def standardize(X):
    """Center/scale by train-split statistics, in place and in float32.

    Written this way for memory, not style: the FFT block is ~2959 x 256k, so a float64 copy is
    ~6GB and the naive `(X - mu) / sigma` materializes two more of them (~18GB peak) -- enough to
    swap, which is what made this step take minutes rather than seconds. float32 halves the array
    and the in-place ops keep peak usage at one copy.
    """
    Xz = X.astype(np.float32, copy=True)
    mu, sigma = Xz[train_mask].mean(0), Xz[train_mask].std(0)
    np.subtract(Xz, mu, out=Xz)
    np.divide(Xz, np.where(sigma > 0, sigma, np.float32(1.0)), out=Xz)
    return Xz

def _torch_pca_on(Xz, k, fit_mask, dev):
    T = torch.from_numpy(Xz).to(dev)
    try:
        fit_rows = T[torch.from_numpy(fit_mask).to(dev)]
        mean = fit_rows.mean(0, keepdim=True)
        # q slightly above k (oversampling) makes the randomized range-finder much more accurate
        _, _, V = torch.pca_lowrank(fit_rows - mean, q=min(k + 10, *fit_rows.shape), center=False)
        return ((T - mean) @ V[:, :k]).cpu().numpy()
    finally:
        del T
        if dev == 'cuda': torch.cuda.empty_cache()

def torch_pca(Xz, k, fit_mask):
    """Rank-k PCA via torch, preferring GPU but falling back to CPU on OOM.

    n_features (~256k) >> n_samples (~3k) here, so the exact SVD sklearn runs by default is
    hopeless; this is a randomized low-rank factorization (same family as
    svd_solver='randomized'). Components come from the fit rows only, then every row is projected.

    The fallback matters in practice: this box's GPU is usually busy with a training run, and the
    FFT block alone is ~3GB, so the transfer is what OOMs rather than the factorization. CPU torch
    does the whole thing in a few seconds anyway, so falling back costs almost nothing.

    Catching only OutOfMemoryError is not enough. When the GPU is nearly full, the QR inside
    pca_lowrank fails in cuSOLVER *before* torch can raise a clean OOM, surfacing as
    `RuntimeError: cusolver error: CUSOLVER_STATUS_INTERNAL_ERROR ... cusolverDnCreate(handle)` --
    it cannot allocate its internal workspace. That is the same "GPU is busy" condition and wants
    the same CPU fallback, so RuntimeError is caught too. Anything genuinely broken will fail again
    on CPU and raise there, so this cannot silently mask a real bug.
    """
    if DEVICE == 'cuda' and torch.cuda.is_available():
        try:
            return _torch_pca_on(Xz, k, fit_mask, 'cuda')
        except (torch.OutOfMemoryError, RuntimeError) as e:
            print(f'    (GPU busy/OOM -- falling back to CPU: {type(e).__name__}: '
                  f'{str(e).splitlines()[0][:90]})')
    return _torch_pca_on(Xz, k, fit_mask, 'cpu')

def reduce(X, name):
    # the three feature spaces live on wildly different scales (raw normalized FFT vs a d_model
    # embedding vs logits) and every method here is scale-sensitive, so standardize first --
    # on train statistics only, for the same no-leakage reason as the fit itself
    Xz = standardize(X)
    dims = X.shape[1]

    if REDUCER == 'pca':
        xyz = torch_pca(Xz, N_COMPONENTS, train_mask)
        print(f'  {name}: {dims}d -> {xyz.shape[1]}d')
        return xyz

    # pre-PCA before the expensive nonlinear step: same torch path, fit on train, applied to all
    if PRE_PCA_DIMS and PRE_PCA_DIMS < dims:
        Xz = torch_pca(Xz, min(PRE_PCA_DIMS, int(train_mask.sum())), train_mask)
        dims = Xz.shape[1]

    if REDUCER == 'tsne':
        xyz = TSNE(n_components=N_COMPONENTS, **kwargs).fit_transform(Xz)  # no transform(): fit on all
    else:
        xyz = umap.UMAP(n_components=N_COMPONENTS, **kwargs).fit(Xz[train_mask]).transform(Xz)
    print(f'  {name}: {X.shape[1]}d -> {dims}d -> {xyz.shape[1]}d')
    return xyz

label = REDUCER.upper() if REDUCER != 'tsne' else 't-SNE'
embeddings = {
    f'{label} on Model Input (Tokenized FFT)': reduce(FFT, 'fft'),
    f'{label} on CLS Token (Laser Encoder)':   reduce(CLS, 'cls'),
    f'{label} on Model Output (Mask Logits)':  reduce(LOGIT, 'logits'),
}
''')

md('''
## Coloring helpers

Same scheme as notebook 47 / `src/model/pca.py`: a CVD-validated categorical palette (its *order*
is the safety mechanism -- keep it), cycling marker symbols past 8 categories, a Viridis colorbar
for continuous fields, and the 2D `com` encoding (hue = horizontal position, lightness = vertical)
with a swatch image standing in for the legend a 1D colorbar can't be. Empty-box samples record
`com` as the `(-1,-1)` sentinel and are drawn as a fixed black trace rather than being fed through
any com scale, so they don't drag the colorbar range below 0 or clamp to one corner of the swatch.
''')

code(r'''
COLOR_BY = ['split', 'speaker', 'layout', 'n_objects', 'position_id', 'com', 'com_x', 'com_y']
# Some datasets never populate the center-of-mass and record the (-1,-1) sentinel for *every*
# sample (the 21x30 gastronorm MDS is one: its metadata.jsonl has no downsampled_com at all). With
# no valid COM anywhere there is no range to build a color scale or swatch from, so the com options
# are dropped rather than left in to produce an all-black panel and a NaN-filled swatch.
HAS_COM = bool(((meta['com_x'] >= 0) & (meta['com_y'] >= 0)).any())
if not HAS_COM:
    COLOR_BY = [c for c in COLOR_BY if c not in ('com', 'com_x', 'com_y')]
    print('no valid center-of-mass in this dataset (all -1 sentinels) -- com color options disabled')
# Continuous fields get the Viridis colorbar rather than the categorical palette.
CONTINUOUS = {'com_x', 'com_y'}
# Above this many categories a discrete field stops emitting legend entries: position_id has ~376
# distinct values, and a 376-row legend is unusable and would squeeze the panels to nothing. The
# traces are still separate and still hoverable, so identity comes from the tooltip instead.
MAX_LEGEND_CATEGORIES = 24
# Only the com fields carry the (-1,-1) "no center-of-mass" sentinel that gets split off into a
# black trace. position_id has no sentinel, so it must not be filtered against com_x/com_y --
# on a dataset with no COM at all that would send every point to the black trace.
SENTINEL_FIELDS = {'com', 'com_x', 'com_y'}
PALETTE = ['#2a78d6', '#1baf7a', '#eda100', '#008300', '#4a3aa7', '#e34948', '#e87ba4', '#eb6834']
SYMBOLS = ['circle', 'diamond', 'square', 'cross']

def wide_palette(n, saturations=(0.85, 0.55), lightnesses=(0.45, 0.68)):
    """`n` maximally-spread hues, walked by the golden angle so neighbours never collide.

    The 8-colour PALETTE above is CVD-validated and its *order* is the safety mechanism, so it is
    kept for low-cardinality fields. It cannot serve position_id (~376 values): cycling it gives an
    exact repeat every 8th category, and adjacent ids -- the ones you most want to tell apart --
    land on the same swatch. Stepping hue by the golden angle (137.5 deg) instead means consecutive
    ids are always far apart on the wheel, and alternating saturation/lightness multiplies the
    distinguishable set by 4 before any hue is reused.
    """
    out = []
    for i in range(n):
        h = (i * 0.381966) % 1.0                       # golden angle, in turns
        s = saturations[(i // len(lightnesses)) % len(saturations)]
        l = lightnesses[i % len(lightnesses)]
        r, g, b = colorsys.hls_to_rgb(h, l, s)
        out.append(f'#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}')
    return out
EMPTY_BOX_COLOR = '#000000'
COM_LEGEND_SIZE = 120
AXES = [f'{label}{i}' for i in (1, 2, 3)]   # e.g. UMAP1/UMAP2/UMAP3, PCA1/..., t-SNE1/...
# Every trace carries the same customdata layout, so one column order serves both templates:
#   0=sample_id  1=com_x  2=com_y  3=position_id  4=speaker
# Indices 1 and 2 are load-bearing beyond the tooltip -- the hover-to-swatch JS at the bottom reads
# customdata[1]/[2] to move the marker on the com legend -- so this order must not be reshuffled.
HOVER_COLS = ['sample_id', 'com_x', 'com_y', 'position_id', 'speaker']
_TAIL = ('position_id=%{customdata[3]:.0f}<br>speaker=%{customdata[4]:.0f}<br>'
         + AXES[0] + '=%{x:.2f}<br>' + AXES[1] + '=%{y:.2f}<br>' + AXES[2] + '=%{z:.2f}<extra></extra>')
HOVER = 'sample_id=%{customdata[0]:.0f}<br>' + _TAIL
COM_HOVER = ('sample_id=%{customdata[0]:.0f}<br>com=(%{customdata[1]:.1f}, %{customdata[2]:.1f})<br>'
             + _TAIL)

def com_ranges(df):
    valid = df[(df['com_x'] >= 0) & (df['com_y'] >= 0)]
    return (valid['com_x'].min(), valid['com_x'].max()), (valid['com_y'].min(), valid['com_y'].max())

def _com_to_hsl(com_x, com_y, x_range, y_range):
    (x_lo, x_hi), (y_lo, y_hi) = x_range, y_range
    hue = 0.8 * np.clip((com_y - y_lo) / ((y_hi - y_lo) or 1), 0, 1)
    lightness = 0.80 - 0.55 * np.clip((com_x - x_lo) / ((x_hi - x_lo) or 1), 0, 1)
    return [f'rgb({r*255:.0f}, {g*255:.0f}, {b*255:.0f})'
            for h, l in zip(hue, lightness) for r, g, b in [colorsys.hls_to_rgb(h, l, 1.0)]]

def _com_swatch(x_range, y_range, size=COM_LEGEND_SIZE):
    xs, ys = np.linspace(*x_range, size), np.linspace(*y_range, size)
    colors = _com_to_hsl(np.repeat(xs, size), np.tile(ys, size), x_range, y_range)
    return np.array([[int(v) for v in c[4:-1].split(',')] for c in colors],
                    dtype=np.uint8).reshape(size, size, 3)

def _com_ticks(lo, hi, size=COM_LEGEND_SIZE, target_ticks=6):
    # "nice" 1/2/5 step, in pixel coords but labeled in real com units
    raw = (hi - lo) / target_ticks
    mag = 10 ** np.floor(np.log10(raw)) if raw > 0 else 1
    step = next(m * mag for m in (1, 2, 5, 10) if raw <= m * mag)
    units = np.arange(np.ceil(lo / step) * step, hi + step / 2, step)
    return units, (units - lo) / ((hi - lo) or 1) * (size - 1)

com_rng = com_ranges(meta) if HAS_COM else None
''')

code(r'''
def traces(df, xyz, color_by, primary, com_rng):
    """Traces for one panel. `primary` is the panel that owns the single shared legend/colorbar;
    legendgroup is shared across panels so one legend click toggles a category on all three."""
    # No marker outline: a white rim on a size-4 point is mostly rim, and in a dense 3D scene the
    # overlapping rims merge into white blobs that swallow the fill color. Outlines are a 2D-scatter
    # idiom for separating overlapping marks; in 3D they just wash the palette out.
    marker = dict(size=3, opacity=0.8, line=dict(width=0))
    def scatter(mask, marker, customdata=None, hovertemplate=HOVER, **kw):
        return go.Scatter3d(x=xyz[mask, 0], y=xyz[mask, 1], z=xyz[mask, 2], mode='markers',
                            marker=marker, hovertemplate=hovertemplate,
                            customdata=df[HOVER_COLS].to_numpy()[mask] if customdata is None else customdata,
                            **kw)

    if color_by in CONTINUOUS or color_by == 'com':
        empty = (((df['com_x'] < 0) | (df['com_y'] < 0)).to_numpy() if color_by in SENTINEL_FIELDS
                 else np.zeros(len(df), dtype=bool))
        out = []
        if (~empty).any():
            if color_by == 'com':
                colors = _com_to_hsl(df['com_x'].to_numpy()[~empty], df['com_y'].to_numpy()[~empty], *com_rng)
                out.append(scatter(~empty, dict(marker, color=colors), showlegend=False,
                                   customdata=df[HOVER_COLS].to_numpy()[~empty],
                                   hovertemplate=COM_HOVER))
            else:
                m = dict(marker, color=df[color_by].to_numpy()[~empty], colorscale='Viridis',
                         showscale=primary)
                if primary:  # colorbar under the legend slot, not on top of it
                    m['colorbar'] = dict(title=color_by, x=1.0, y=0.42, yanchor='middle', len=0.65)
                out.append(scatter(~empty, m, showlegend=False))
        if empty.any():
            out.append(scatter(empty, dict(marker, color=EMPTY_BOX_COLOR), name='empty box',
                               legendgroup='empty-box', showlegend=primary))
        return out

    # Color/symbol are keyed to a category's position in sorted order, so a given speaker (or
    # n_objects value) always gets the same swatch. Draw order is separate: Plotly paints later
    # traces over earlier ones at equal depth, so adding categories largest-first puts the rare
    # ones on top. Without this, n_objects=0 (55 samples) is buried under n_objects=2 (2288) and
    # reads as missing, and whichever category sorts last looks over-represented.
    values = sorted(df[color_by].unique(), key=str)
    counts = df[color_by].value_counts()
    order = sorted(range(len(values)), key=lambda i: -counts[values[i]])

    # Past MAX_LEGEND_CATEGORIES the 8-colour PALETTE would repeat exactly every 8th category, so
    # switch to the generated wide palette and drop the legend (376 rows would crowd out the
    # panels). The traces stay separate and hoverable, so the tooltip carries the identity.
    n = len(values)
    wide = n > MAX_LEGEND_CATEGORIES
    colors = wide_palette(n) if wide else [PALETTE[i % len(PALETTE)] for i in range(n)]
    show = primary and not wide

    return [scatter((df[color_by] == values[i]).to_numpy(),
                    dict(marker, color=colors[i],
                         symbol=SYMBOLS[(i // len(PALETTE)) % len(SYMBOLS)] if not wide else 'circle'),
                    name=str(values[i]), legendgroup=f'{color_by}-{values[i]}',
                    legendrank=1000 + i, showlegend=show)
            for i in order]
''')

md('''
## The three linked panels

One `Scatter3d` scene per representation, sharing one legend/colorbar and one color-by dropdown.
Hovering a point moves the marker on the `com` swatch to that point's real position, so you can
read a point's location off the color key directly instead of decoding the hue by eye.
''')

code(r'''
panel_names = list(embeddings)
n = len(panel_names)
sub = '<br><span style="font-size:11px;color:#888">{}</span>'
fig = make_subplots(rows=1, cols=n, specs=[[{'type': 'scene'}] * n], horizontal_spacing=0.02,
                    subplot_titles=[name + sub.format(
                        f'{RUN_NAME} -- fit on {"all points (t-SNE has no transform)" if REDUCER == "tsne" else "train"}')
                                    for name in panel_names])

# scenes share [0, 0.86] of the width; the right strip holds the legend/colorbar/com swatch
gap = 0.02
width = (0.86 - gap * (n - 1)) / n
for i in range(n):
    lo = i * (width + gap)
    fig.layout['scene' if i == 0 else f'scene{i + 1}'].domain.x = (lo, lo + width)
    fig.layout.annotations[i].x = lo + width / 2   # re-center each subplot title over its scene

trace_opts = []   # color_by option of every trace, in fig.data order -> dropdown visibility masks
for opt in COLOR_BY:
    for col, name in enumerate(panel_names, start=1):
        for tr in traces(meta, embeddings[name], opt, primary=col == 1, com_rng=com_rng):
            tr.visible = opt == COLOR_BY[0]
            fig.add_trace(tr, row=1, col=col)
            trace_opts.append(opt)
    if opt == 'com':
        fig.add_trace(go.Image(z=_com_swatch(*com_rng), hoverinfo='skip', visible=opt == COLOR_BY[0]))
        trace_opts.append(opt)
        # hover marker on the swatch: moved by the post_script JS below, hidden (x/y None) until then
        fig.add_trace(go.Scatter(x=[None], y=[None], mode='markers', hoverinfo='skip', showlegend=False,
                                 marker=dict(size=14, color='white', symbol='circle-open',
                                             line=dict(color='black', width=2)),
                                 visible=opt == COLOR_BY[0]))
        marker_idx = len(fig.data) - 1
        trace_opts.append(opt)
''')

code(r'''
# com swatch axes + grid -- only when this dataset actually has center-of-mass values (see
# HAS_COM). Plotly image traces ignore axis showgrid (gridlines render *behind* the opaque image),
# so the grid is drawn as layout shapes, which have their own z-order. Axes and shapes are
# layout-level, so each dropdown button toggles them alongside the trace visibilities.
com_shapes = []
if HAS_COM:
    (x_units, x_px), (y_units, y_px) = _com_ticks(*com_rng[0]), _com_ticks(*com_rng[1])
    pad = COM_LEGEND_SIZE * 0.06
    axis = dict(range=[-pad, COM_LEGEND_SIZE - 1 + pad], autorange=False, constrain='domain',
                tickmode='array', tickfont=dict(size=9), visible=COLOR_BY[0] == 'com')
    fig.update_layout(
        xaxis=dict(axis, domain=[0.875, 0.995], tickvals=y_px, ticktext=[f'{v:.0f}' for v in y_units],
                   title=dict(text='com_y (col)', font=dict(size=11))),
        yaxis=dict(axis, domain=[0.28, 0.68], scaleanchor='x', tickvals=x_px,
                   ticktext=[f'{v:.0f}' for v in x_units], title=dict(text='com_x (row)', font=dict(size=11))))
    com_shapes = [
        *(dict(type='line', x0=t, x1=t, y0=0, y1=COM_LEGEND_SIZE - 1, xref='x', yref='y',
               line=dict(color='rgba(255,255,255,0.75)', width=2, dash='dot')) for t in y_px),
        *(dict(type='line', y0=t, y1=t, x0=0, x1=COM_LEGEND_SIZE - 1, xref='x', yref='y',
               line=dict(color='rgba(255,255,255,0.75)', width=2, dash='dot')) for t in x_px),
        dict(type='rect', x0=0, x1=COM_LEGEND_SIZE - 1, y0=0, y1=COM_LEGEND_SIZE - 1, xref='x', yref='y',
             line=dict(color='black', width=1.5), fillcolor='rgba(0,0,0,0)')]

buttons = [dict(label=f'color by {opt}', method='update',
                args=[{'visible': [o == opt for o in trace_opts]},
                      {'legend.title.text': opt, 'xaxis.visible': opt == 'com',
                       'yaxis.visible': opt == 'com', 'shapes': com_shapes if opt == 'com' else []}])
           for opt in COLOR_BY]
fig.update_layout(updatemenus=[dict(buttons=buttons, x=0, xanchor='left', y=1.15, yanchor='top')],
                  height=620, margin=dict(l=0, r=0, t=100, b=0),
                  legend=dict(title=COLOR_BY[0], x=1.0, y=1, yanchor='top'),
                  # without this the swatch's cartesian plot area shows as a gray rectangle in the
                  # right strip even when its axes and image are hidden
                  plot_bgcolor='rgba(0,0,0,0)',
                  shapes=com_shapes if COLOR_BY[0] == 'com' else [])
fig.update_scenes(xaxis_title=AXES[0], yaxis_title=AXES[1], zaxis_title=AXES[2])
# Plotly's default template paints the scene walls '#E5ECF6' with *white* gridlines at gridwidth 2.
# That white grid is what still reads as "white points" in a dense panel: the Model Input embedding
# packs 2959 points into ~40% the span of the other two (mean nearest-neighbour 0.13 vs 0.31), so
# the grid shows through the cloud rather than sitting behind it. A white wall with a light grey
# grid puts the contrast back on the markers.
fig.update_scenes(
    xaxis=dict(backgroundcolor='white', gridcolor='#d9d9d9', gridwidth=1, zerolinecolor='#bfbfbf'),
    yaxis=dict(backgroundcolor='white', gridcolor='#d9d9d9', gridwidth=1, zerolinecolor='#bfbfbf'),
    zaxis=dict(backgroundcolor='white', gridcolor='#d9d9d9', gridwidth=1, zerolinecolor='#bfbfbf'))

# hover-to-swatch JS: only meaningful when the com swatch exists at all
post_script = None
if HAS_COM:
    (x_lo, x_hi), (y_lo, y_hi) = com_rng
    post_script = f"""
var gd = document.getElementById('{{plot_id}}');
var M = {marker_idx}, S = {COM_LEGEND_SIZE - 1};
function comPx(v, lo, hi) {{ return Math.min(Math.max((v - lo) / ((hi - lo) || 1), 0), 1) * S; }}
gd.on('plotly_hover', function(ev) {{
  var pt = ev.points[0];
  // customdata[1]/[2] are the com; every trace carries them now, so the swatch marker must be
  // driven only by points that have a real com -- the (-1,-1) empty-box sentinels would otherwise
  // clamp it to a corner instead of leaving it where it was.
  if (!pt || pt.data.type !== 'scatter3d' || !Array.isArray(pt.customdata) || gd.data[M].visible !== true) return;
  if (!(pt.customdata[1] >= 0) || !(pt.customdata[2] >= 0)) return;
  Plotly.restyle(gd, {{x: [[comPx(pt.customdata[2], {y_lo}, {y_hi})]],
                       y: [[comPx(pt.customdata[1], {x_lo}, {x_hi})]]}}, [M]);
}});
gd.on('plotly_unhover', function() {{ Plotly.restyle(gd, {{x: [[null]], y: [[null]]}}, [M]); }});
"""

html_path = Path(REPO) / 'runs' / RUN_NAME / f'{REDUCER}_cls.html'   # one file per technique
html_path.write_text(fig.to_html(include_plotlyjs=True, full_html=True, default_width='100%',
                                 default_height='620px', post_script=post_script))
print(f'wrote {html_path}')
fig.show()
''')

nb = {'cells': cells, 'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python',
      'name': 'python3'}, 'language_info': {'name': 'python', 'version': '3.12.0'}},
      'nbformat': 4, 'nbformat_minor': 5}
out = Path('/home/ethantu/workspace/good-vibrations/notebooks/66_umap_cls.ipynb')
out.write_text(json.dumps(nb, indent=1))
print('wrote', out, len(cells), 'cells')
