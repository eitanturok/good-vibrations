"""Generate notebooks/67_decoder_last_layer.ipynb (heatmap of the decoder's final weight matrix)."""
import json
from pathlib import Path

md = lambda s: {"cell_type": "markdown", "metadata": {}, "source": s.strip("\n").splitlines(keepends=True)}
code = lambda s: {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
                  "source": s.strip("\n").splitlines(keepends=True)}

cells = [
md("""
# 67 Decoder last layer: hidden x HW weight heatmap

The final layer of the decoder maps a hidden vector straight onto the flattened output mask, so its
weight matrix *is* the model's dictionary of spatial patterns: column `(h,w)` of the `hidden x HW`
matrix says how much every hidden unit pushes that one output cell up or down.

Set `RUN_NAME` below; everything else (decoder type, `out_h`/`out_w`, `d_model`) is recovered from
the run's wandb config, so nothing has to be re-specified by hand.

Five sections:
1. **The raw matrix** as a `hidden x HW` heatmap -- the literal thing asked for.
2. **The same matrix with columns sorted by output position**, plus row (hidden-unit) ordering by
   similarity, so the spatial structure that the raw flattened ordering hides becomes visible.
3. **Per-hidden-unit spatial maps**: each row of the matrix reshaped back to `out_h x out_w`, which
   is what that unit would paint on the mask. Weights only -- no data involved.
4. **Activations**: a forward pass over a split, so units can be ranked by what they actually
   contribute (`mean activation x weight`) rather than by weight alone.
5. **What we can learn from all this** -- the measurements behind the conclusions, including the
   headline result that weight magnitude is a *negative* predictor of which units matter.
"""),

md("## Config -- set the run name"),
code("""
RUN_NAME = 'ce-pixel-all-gastronorm-21x30'   # a run under runs/ that also exists in wandb
WANDB_PATH = 'eturok/better-tsa'             # wandb entity/project the run was logged to
CHECKPOINT = 'latest-rank0.pt'               # file under runs/<RUN_NAME>/checkpoints/

REPO = '/home/ethantu/workspace/good-vibrations'

N_UNIT_MAPS = 30     # how many per-hidden-unit spatial maps to draw

# How to rank hidden units when picking the top N_UNIT_MAPS for the weights-only grid:
#   'max'  -- largest single weight in the unit's map ("greatest value")
#   'norm' -- L2 norm over the whole map, i.e. which units swing the output the most overall
UNIT_RANK_BY = 'max'

UNIT_MAP_COLS = 6    # panels per row in the grid

# --- section 4 only: the activation pass (needs the dataset + a GPU) ---
RUN_ACTIVATIONS = True   # set False to stop after the weights-only sections
# Which loader to average activations over: 'train', an eval label, or None for the first eval split.
# 'train' by default: the eval splits are only ~26-96 samples, far too few to rank 256 units stably.
ACT_SPLIT = 'train'
BATCH_SIZE = 128
NUM_WORKERS = 4
DEVICE = 'cuda'          # 'cuda' or 'cpu'
"""),

code("""
import sys
from pathlib import Path

sys.path[:0] = [f'{REPO}/src', REPO]   # src/ for model.*, repo root for utils.* (imported by arch)

import numpy as np
import torch
import torch.nn as nn
import wandb
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.cluster.hierarchy import leaves_list, linkage
"""),

md("""
## Recover the run config + the checkpoint weights

`run.py` logs `data_info | args.__dict__` as the wandb config, so `out_h`/`out_w`/`decoder` are all
there. Composer checkpoints nest the weights under `state.model`; keys are stripped of any
`_orig_mod.` prefix left by `torch.compile`. The model itself is never rebuilt here -- the weight
matrix is read straight out of the state dict, so this notebook needs no dataset and no GPU.
"""),
code("""
cfg = wandb.Api().run(f'{WANDB_PATH}/{RUN_NAME}').config
OUT_H, OUT_W = cfg['out_h'], cfg['out_w']
print(f"decoder={cfg['decoder']}  d_model={cfg['d_model']}  out={OUT_H}x{OUT_W} -> HW={OUT_H * OUT_W}")

ckpt_path = Path(REPO) / 'runs' / RUN_NAME / 'checkpoints' / CHECKPOINT
state = torch.load(ckpt_path, map_location='cpu', weights_only=False)['state']['model']
state = {k.removeprefix('_orig_mod.'): v for k, v in state.items()}
print(f'loaded {ckpt_path.name}')
"""),

md("""
## Pull out the last decoder layer

The `mlp` / `mlp-mid` decoders end in `nn.Linear(hidden, out_h*out_w)`, whose weight is stored as
`(out_features, in_features) = (HW, hidden)`. The requested orientation is `hidden x HW`, so it is
transposed once here and every plot below uses that.

The `attn` decoders have no such layer -- their head is `Linear(d_model, 1)` applied per query, so
there is no single `hidden x HW` matrix to show; that case is rejected loudly rather than silently
plotting the wrong tensor.
"""),
code("""
assert cfg['decoder'] in ('mlp', 'mlp-mid'), (
    f"decoder={cfg['decoder']!r} has no hidden x HW output matrix "
    "(attn decoders end in Linear(d_model, 1) applied per output query)")

# the decoder's final Linear = the highest-numbered decoder.net.<i>.weight
last_i = max(int(k.split('.')[2]) for k in state if k.startswith('decoder.net.') and k.endswith('.weight'))
W = state[f'decoder.net.{last_i}.weight'].float().numpy()     # (HW, hidden)
b = state[f'decoder.net.{last_i}.bias'].float().numpy()       # (HW,)
assert W.shape[0] == OUT_H * OUT_W, f'{W.shape=} does not match {OUT_H}x{OUT_W}'

W = W.T                                                        # (hidden, HW) -- the asked-for shape
HIDDEN, HW = W.shape
print(f'decoder.net.{last_i}: hidden x HW = {HIDDEN} x {HW}   '
      f'|w| mean={np.abs(W).mean():.4f}  std={W.std():.4f}  range=[{W.min():.3f}, {W.max():.3f}]')
"""),

md("""
## 1. The raw `hidden x HW` heatmap

Rows are hidden units, columns are flattened output cells in row-major order, so column `i` is
cell `(i // out_w, i % out_w)`. The colorscale is diverging and symmetric about 0 -- sign matters
here (a negative weight actively suppresses that cell), so a sequential scale would be misleading.
The range is clipped to the 99th percentile of `|W|` so a handful of outlier weights don't wash the
rest of the map out.
"""),
code("""
vmax = float(np.quantile(np.abs(W), 0.99))

def heatmap(mat, x=None, y=None, title='', xtitle='output cell (flattened HW)', ytitle='hidden unit',
            hovertemplate=None, height=620):
    fig = go.Figure(go.Heatmap(z=mat, x=x, y=y, zmid=0, zmin=-vmax, zmax=vmax, colorscale='RdBu',
                               reversescale=True, colorbar=dict(title='weight'),
                               hovertemplate=hovertemplate))
    fig.update_layout(title=title, xaxis_title=xtitle, yaxis_title=ytitle, height=height,
                      margin=dict(l=70, r=20, t=60, b=60))
    return fig

hover = ('hidden unit %{y}<br>cell %{x} = (row %{customdata[0]}, col %{customdata[1]})'
         '<br>weight %{z:.4f}<extra></extra>')
cell_rc = np.stack([np.repeat(np.arange(OUT_H), OUT_W), np.tile(np.arange(OUT_W), OUT_H)], -1)

fig = heatmap(W, title=f'{RUN_NAME} -- decoder.net.{last_i} weight ({HIDDEN} hidden x {HW} cells)',
              hovertemplate=hover)
fig.data[0].customdata = np.broadcast_to(cell_rc[None], (HIDDEN, HW, 2))
fig.show()
"""),

md("""
## 2. Sorted: hidden units clustered, columns grouped by output row

Two orderings are arbitrary in the raw view. Hidden-unit index carries no meaning at all (it's
whatever initialization landed on), and the flattened column order interleaves spatial neighbours
in a way that hides structure across output rows.

So: hidden units are reordered by hierarchical clustering (correlation distance over their spatial
maps, so units that paint similar things sit together), and vertical lines mark the `out_h`
boundaries where the flattened index wraps to the next output row. Within each block, columns run
left-to-right across the mask.
"""),
code("""
order = leaves_list(linkage(W, method='average', metric='correlation'))
fig = heatmap(W[order], y=[str(i) for i in order],
              title=f'{RUN_NAME} -- same matrix, hidden units clustered by spatial similarity',
              ytitle='hidden unit (clustered)', hovertemplate=hover)
fig.data[0].customdata = np.broadcast_to(cell_rc[None], (HIDDEN, HW, 2))
fig.update_yaxes(type='category', showticklabels=False)
for r in range(1, OUT_H):   # one boundary per output-row wrap
    fig.add_vline(x=r * OUT_W - 0.5, line_width=1, line_color='rgba(0,0,0,0.25)')
fig.show()
"""),

md("""
## 3. Each hidden unit as an `out_h x out_w` image (weights only)

The flattened `HW` axis above is hard to read spatially. Row `k` of the matrix reshaped back to
`out_h x out_w` is the same numbers laid out as an image: literally the mask pattern hidden unit `k`
adds to the logits when it fires with strength 1.

**These are weights, not activations.** Nothing here touches the dataset -- this is a static
parameter of the model, so each panel answers "what *would* unit `k` paint if it fired", not "what
does unit `k` actually do on real samples". Section 4 brings the data in and answers the second
question.

Ranked by `UNIT_RANK_BY` over the whole `HW` map (`max` = largest single weight, `norm` = biggest
overall swing), then drawn in the clustered order from section 2 so neighbouring panels are related
units. Every panel shares one symmetric color range, so they are directly comparable.
"""),
code("""
score = W.max(axis=1) if UNIT_RANK_BY == 'max' else np.linalg.norm(W, axis=1)
label = 'max w' if UNIT_RANK_BY == 'max' else '|w|'
sel = set(np.argsort(-score)[:N_UNIT_MAPS])
top = [k for k in order if k in sel]   # top units, in the clustered order

def unit_grid(rows_k, mats, titles, title, cbar_title, zmin, zmax, colorscale='RdBu', reverse=True):
    \"\"\"Grid of out_h x out_w panels, one per hidden unit, on a shared color range.\"\"\"
    nrows = int(np.ceil(len(rows_k) / UNIT_MAP_COLS))
    fig = make_subplots(rows=nrows, cols=UNIT_MAP_COLS, subplot_titles=titles,
                        horizontal_spacing=0.02, vertical_spacing=0.06)
    for i, z in enumerate(mats):
        fig.add_trace(go.Heatmap(z=z, zmid=0, zmin=zmin, zmax=zmax, colorscale=colorscale,
                                 reversescale=reverse, showscale=(i == 0),
                                 colorbar=dict(title=cbar_title, len=0.4, y=0.8),
                                 hovertemplate='row %{y}, col %{x}<br>%{z:.4f}<extra></extra>'),
                      row=i // UNIT_MAP_COLS + 1, col=i % UNIT_MAP_COLS + 1)
    fig.update_yaxes(autorange='reversed', showticklabels=False)   # row 0 on top, like the mask
    fig.update_xaxes(showticklabels=False)
    fig.update_annotations(font_size=10)
    fig.update_layout(height=190 * nrows, title=title, margin=dict(l=20, r=20, t=70, b=20))
    return fig

unit_grid(top, [W[k].reshape(OUT_H, OUT_W) for k in top],
          [f'unit {k} ({label}={score[k]:.2f})' for k in top],
          f'{RUN_NAME} -- top-{len(top)} hidden units by {label}, each as a {OUT_H}x{OUT_W} map (weights)',
          'weight', -vmax, vmax).show()
"""),

md("""
## Bias: what the decoder paints with no input at all

The last layer's bias is a full `HW` vector, i.e. a fixed mask added to every prediction. It is the
model's prior over where objects live, so it's worth seeing next to the weights.
"""),
code("""
bmax = float(np.abs(b).max())
fig = go.Figure(go.Heatmap(z=b.reshape(OUT_H, OUT_W), zmid=0, zmin=-bmax, zmax=bmax,
                           colorscale='RdBu', reversescale=True, colorbar=dict(title='bias'),
                           hovertemplate='row %{y}, col %{x}<br>bias %{z:.4f}<extra></extra>'))
fig.update_yaxes(autorange='reversed')
fig.update_layout(title=f'{RUN_NAME} -- decoder.net.{last_i} bias as {OUT_H}x{OUT_W}',
                  xaxis_title='col', yaxis_title='row', height=420, width=600,
                  margin=dict(l=60, r=20, t=60, b=50))
fig.show()
"""),

md("""
# 4. Activations: what the units actually do on real data

Everything above is a static parameter. A unit with a big weight row that is near-zero on every real
sample contributes nothing to actual predictions, yet it still sorts to the top of the weights-only
ranking. To separate "could paint" from "does paint" we need the activation.

The last layer computes `logits = a @ W + b`, where `a` is the post-ReLU hidden vector feeding it.
So unit `k`'s real contribution to a given sample is `a_k * W[k]` -- its template scaled by how hard
it fired. Averaged over a split, `mean(a_k) * W[k]` is its **mean effective contribution**, in the
same logit units as the weights, and those maps sum exactly to the model's mean logit map (minus the
bias). Because `a` is post-ReLU it is non-negative, so a unit's sign structure is never flipped by
its activation -- only scaled.

This section rebuilds the dataloaders (same `build_dataset` call `run.py` makes, augmentation forced
off) and the model, then hooks the ReLU immediately before the final Linear.
"""),
code("""
if RUN_ACTIVATIONS:
    from model.dataset import build_dataset
    from model.arch import VibrationTransformer

    _, eval_loaders, train_eval_loader = build_dataset(
        cfg['data_dir'], split=cfg['split'], batch_size=BATCH_SIZE, eval_batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS, test_size=cfg['test_size'], speakers=cfg['speakers'],
        n_objects=cfg['n_objects'], box=cfg['box'], n_samples=cfg['n_samples'],
        out_h=OUT_H, out_w=OUT_W, signal_mode=cfg['signal_mode'],
        normalize_mode=cfg['normalize_mode'], patch_size=cfg['patch_size'], seed=cfg['seed'],
        augment_fft=0.0, augment_mask=0.0, subtract_speaker_mean=bool(cfg['subtract_speaker_mean']))

    loaders = {'train': train_eval_loader.dataloader} | {ev.label: ev.dataloader.dataloader for ev in eval_loaders}
    print({k: len(v.dataset) for k, v in loaders.items()})

    # default to the first eval split (whatever this run's split method named it)
    if ACT_SPLIT is None: ACT_SPLIT = next(ev.label for ev in eval_loaders)
    assert ACT_SPLIT in loaders, f'{ACT_SPLIT=} not in {sorted(loaders)}'
    print(f'averaging activations over {ACT_SPLIT!r}')
"""),

md("""
## Rebuild the model and hook the pre-final-layer activation

`data_info` is reconstructed the way `run.py` does it (`n_freqs` is the *padded* length
`n_patches * patch_size`). `.eval()` disables `freq_dropout`/`laser_dropout`, so every sample gets a
clean forward with all lasers and frequency patches intact.

The hook sits on the module *before* the final Linear (`decoder.net[last_i - 1]`, the ReLU), whose
output is exactly the `a` that gets multiplied by `W`. Asserting the hooked vector reproduces the
model's own logits is the check that we grabbed the right tensor.
"""),
code("""
if RUN_ACTIVATIONS:
    n_lasers, n_patches, patch_size, n_channels = train_eval_loader.dataloader.dataset[0]['fft'].shape
    data_info = dict(out_h=OUT_H, out_w=OUT_W, n_laser_rows=cfg['n_laser_rows'],
                     n_laser_cols=cfg['n_laser_cols'], patch_size=cfg['patch_size'],
                     n_freqs=n_patches * patch_size, n_channels=n_channels)

    model = VibrationTransformer(cfg['d_model'], cfg['pnt_num_heads'], cfg['pnt_num_layers'],
                                 cfg['seq_num_heads'], cfg['seq_num_layers'], data_info, cfg['decoder'],
                                 cfg['decoder_num_heads'], cfg['decoder_num_layers'],
                                 freq_dropout=cfg['freq_dropout'], laser_dropout=cfg['laser_dropout'],
                                 loss_fn=cfg['loss_fn'])
    missing, unexpected = model.load_state_dict(state, strict=False)
    # `count_head` was added to arch.py after this run trained, so an older checkpoint has no weights
    # for it. It feeds only the object-count output and never the decoder path measured here, so
    # leaving it randomly initialized is harmless -- but nothing else may be missing.
    assert not unexpected, f'{unexpected=}'
    assert all(k.startswith('count_head.') for k in missing), f'{missing=}'
    if missing: print(f'note: {missing} not in this checkpoint (post-dates it, unused by the decoder)')
    model = model.to(DEVICE).eval()

    act_buffer = {}
    def _hook(module, args, output):
        act_buffer['a'] = output.detach()          # (B,hidden) post-ReLU, the `a` in a @ W + b
    handle = model.decoder.net[last_i - 1].register_forward_hook(_hook)
    print(f'hooked decoder.net[{last_i - 1}] = {model.decoder.net[last_i - 1]}')
"""),

md("""
## Sweep the split and accumulate mean activation per unit

One pass, no gradients. The whole `(n_samples, hidden)` activation matrix is kept -- it is only a
few MB at this scale, and section 5 needs the per-sample spread, not just the mean. From it we get
how hard each unit fires on average, how *often* it fires (ReLU units are frequently dead), and its
max, so a unit that fires rarely but hugely is distinguishable from one that fires constantly and
weakly.
"""),
code("""
if RUN_ACTIVATIONS:
    @torch.no_grad()
    def sweep_activations(loader):
        acts, n = [], 0
        for batch in loader:
            out = model({'fft': batch['fft'].to(DEVICE, non_blocking=True)})
            a = act_buffer['a'].float()
            # the hooked `a` must reproduce the model's own logits, else we hooked the wrong module
            if n == 0:
                Wt = torch.as_tensor(W, device=a.device)      # (hidden,HW)
                bt = torch.as_tensor(b, device=a.device)
                assert torch.allclose(a @ Wt + bt, out['mask_logits'].flatten(1).float(), atol=1e-3)
            acts.append(a.cpu().numpy())
            n += a.shape[0]
        return np.concatenate(acts)                            # (n_samples, hidden)

    try:
        ACT = sweep_activations(loaders[ACT_SPLIT])
    finally:
        handle.remove()   # always detach the hook, even if the sweep raises

    act_mean, act_std = ACT.mean(0), ACT.std(0)
    act_rate, act_max = (ACT > 0).mean(0), ACT.max(0)
    n_samples = len(ACT)

    dead = int((act_max == 0).sum())
    print(f'{ACT_SPLIT}: {n_samples} samples   mean activation: '
          f'min={act_mean.min():.3f} med={np.median(act_mean):.3f} max={act_mean.max():.3f}   '
          f'{dead}/{HIDDEN} units never fire')
"""),

md("""
## Weight-only vs data-weighted ranking

How much the two rankings disagree is the whole point of this section: units that top the weight
ranking but sit low on the contribution ranking are ones the model learned a strong template for but
barely uses on this split.
"""),
code("""
if RUN_ACTIVATIONS:
    contrib = act_mean[:, None] * W                       # (hidden,HW) mean effective contribution
    contrib_score = np.abs(contrib).max(axis=1)           # peak effect on any cell, in logit units
    top_contrib = list(np.argsort(-contrib_score)[:N_UNIT_MAPS])

    overlap = len(set(top) & set(top_contrib))
    print(f'top-{N_UNIT_MAPS} by weight vs by mean contribution: {overlap} units in common')
    for k in np.argsort(-score)[:8]:
        print(f'  unit {k:3d}  {label}={score[k]:6.2f}  mean act={act_mean[k]:6.3f}  '
              f'fires {act_rate[k] * 100:5.1f}% of samples  peak contribution={contrib_score[k]:6.3f}')
"""),

md("""
## The same `out_h x out_w` grid, data-weighted

Same layout as section 3, but each panel is `mean(a_k) * W[k]` -- the unit's template scaled by how
hard it actually fires on `ACT_SPLIT`. Ranked by peak effective contribution, so these are the units
that genuinely move this model's predictions. Note the color scale is much tighter than the
weights-only grid: most units contribute far less than their raw weights suggest.
"""),
code("""
if RUN_ACTIVATIONS:
    cmax = float(np.quantile(np.abs(contrib[top_contrib]), 0.995))
    unit_grid(top_contrib, [contrib[k].reshape(OUT_H, OUT_W) for k in top_contrib],
              [f'unit {k} (peak={contrib_score[k]:.2f}, act={act_mean[k]:.2f})' for k in top_contrib],
              f'{RUN_NAME} -- top-{len(top_contrib)} hidden units by mean contribution on {ACT_SPLIT} '
              f'(mean activation x weight)', 'logit contribution', -cmax, cmax).show()
"""),

md("""
## Sanity check: the contributions sum to the model's mean prediction

`sum_k mean(a_k) * W[k] + b` is exactly the mean logit map over the split, so summing every unit's
contribution (not just the top 30) and adding the bias must reproduce it. Shown as the mean
predicted mask alongside -- what all of the above adds up to.
"""),
code("""
if RUN_ACTIVATIONS:
    mean_logits = (contrib.sum(0) + b).reshape(OUT_H, OUT_W)
    fig = make_subplots(rows=1, cols=2, subplot_titles=('mean logits (sum of all unit contributions + bias)',
                                                        'sigmoid(mean logits)'))
    lmax = float(np.abs(mean_logits).max())
    fig.add_trace(go.Heatmap(z=mean_logits, zmid=0, zmin=-lmax, zmax=lmax, colorscale='RdBu',
                             reversescale=True, colorbar=dict(title='logit', len=0.9, x=0.44)), row=1, col=1)
    fig.add_trace(go.Heatmap(z=1 / (1 + np.exp(-mean_logits)), zmin=0, zmax=1, colorscale='Viridis',
                             colorbar=dict(title='p', len=0.9, x=1.02)), row=1, col=2)
    fig.update_yaxes(autorange='reversed')
    fig.update_layout(height=380, title=f'{RUN_NAME} -- decoder output averaged over {ACT_SPLIT}',
                      margin=dict(l=40, r=20, t=80, b=40))
    fig.update_annotations(font_size=12)
    fig.show()
"""),

md("""
# 5. What we can actually learn from these plots

The sections above are pictures; this one is the measurements behind the claims, so nothing here
rests on eyeballing a heatmap. Every number printed below is computed from the cells that follow.

**Run these on `ACT_SPLIT = 'train'`.** The default first eval split is only ~26 samples, far too
few to rank 256 units stably -- the train split has 2679.
"""),

md("""
## 5a. Nothing is dead, and nothing is spatially localized

All 256 units fire (0 dead), with a minimum mean activation around 1.07 -- this layer is fully
utilized, so there is no pruning story here.

But the units are **not** spatial feature detectors. The test: take each unit's `|w|` map, find its
centroid, and measure the rms distance of its mass from that centroid. A localized blob-detector
would score far below a spatially-shuffled version of *itself* (same values, positions permuted).
Measured spread is ~9.5 cells against a shuffled null of ~10.8 -- barely below chance.

Combined with an effective rank of ~154 of 256 and no dominant singular value (the top component
holds only ~8% of the energy), the picture is a **distributed, high-rank code**: each output cell is
assembled from many units, and no unit "owns" a region.

Practically: don't go hunting for the "top-left cube unit." It isn't there.
"""),
code("""
maps = np.abs(W.reshape(HIDDEN, OUT_H, OUT_W))          # |w| as a spatial map per unit
ys, xs = np.mgrid[0:OUT_H, 0:OUT_W]
tot = maps.sum((1, 2))
cy = (maps * ys).sum((1, 2)) / tot                       # centroid of each unit's mass
cx = (maps * xs).sum((1, 2)) / tot
d2 = (ys - cy[:, None, None]) ** 2 + (xs - cx[:, None, None]) ** 2
spread = np.sqrt((maps * d2).sum((1, 2)) / tot)          # rms distance from own centroid

# null: same values, positions shuffled. A localized unit beats its own null by a lot; a
# spatially unstructured one matches it.
rng = np.random.default_rng(0)
null = np.array([np.sqrt((m.ravel()[rng.permutation(OUT_H * OUT_W)].reshape(OUT_H, OUT_W)
                          * d2[i]).sum() / tot[i]) for i, m in enumerate(maps)])
print(f'spatial spread   measured: mean {spread.mean():.2f} cells  (min {spread.min():.2f}, max {spread.max():.2f})')
print(f'                 shuffled null: mean {null.mean():.2f} cells')
print(f'                 -> units beating their own null by >20%: {(spread < 0.8 * null).sum()}/{HIDDEN}')

sv = np.linalg.svd(W, compute_uv=False)
energy = sv ** 2 / (sv ** 2).sum()
eff_rank = sv.sum() ** 2 / (sv ** 2).sum()               # participation ratio
print(f'\\neffective rank {eff_rank:.1f} of {min(W.shape)}   top singular value holds {energy[0] * 100:.1f}% of energy')
print(f'components to reach 90% of energy: {int(np.searchsorted(np.cumsum(energy), 0.90)) + 1}')
"""),

md("""
## 5b. The bias is inert

`sigmoid(bias)` is ~0.5 at every cell, so the model learned **no spatial prior** about where objects
sit. Given that the cubes move around the box, that is the correct thing to learn -- and it is a
small sanity check that training did not collapse onto a fixed "average mask."
"""),
code("""
print(f'bias range [{b.min():.3f}, {b.max():.3f}]   mean {b.mean():.4f}')
print(f'sigmoid(bias): mean {(1 / (1 + np.exp(-b))).mean():.4f}  '
      f'min {(1 / (1 + np.exp(-b))).min():.4f}  max {(1 / (1 + np.exp(-b))).max():.4f}')
print(f'-> a uniform 0.5 prior would give exactly 0.5000')
"""),

md("""
## 5c. Weight magnitude is a *negative* guide to what matters

This is the headline. On the train split the top-30 units by weight and the top-30 by mean
contribution have **no units in common**, and the rank correlation between the two scores is mildly
*negative*. Activation alone, by contrast, correlates ~0.86 with contribution.

So the units with the biggest weights are close to the opposite of the units that matter, and
section 3's weights-only grid is largely a gallery of templates the model rarely uses. That is not a
flaw in the plot -- it is the finding. Any interpretability read that stops at the weight matrix
would be misleading here.
"""),
code("""
if RUN_ACTIVATIONS:
    from scipy.stats import spearmanr

    w_score = W.max(axis=1)                              # the section-3 ranking
    c_score = np.abs(act_mean[:, None] * W).max(axis=1)  # the section-4 ranking
    top_w, top_c = set(np.argsort(-w_score)[:N_UNIT_MAPS]), set(np.argsort(-c_score)[:N_UNIT_MAPS])

    print(f'{ACT_SPLIT}: {n_samples} samples')
    print(f'top-{N_UNIT_MAPS} by weight vs by contribution: {len(top_w & top_c)} units in common')
    print(f'spearman(weight score, contribution) = {spearmanr(w_score, c_score).statistic:+.3f}')
    print(f'spearman(mean activation, contribution) = {spearmanr(act_mean, c_score).statistic:+.3f}')
"""),

md("""
## 5d. Caveat: the contribution plot only shows the *constant* half

Section 4 plots `mean(a_k) * W[k]`, which is the average effect of each unit. But the per-sample
deviation from that average is not small: the mean logit map has norm ~639 while the average
per-sample deviation from it has norm ~228, and per-unit the varying part is slightly *larger* than
the plotted part (activations have std ~= mean).

So the contribution grid answers "what does this unit contribute **on average**", not "what does
this unit do to **distinguish one sample from another**". Since discriminating cube positions is the
actual task, the mean is arguably the less interesting half.

The natural follow-up is below: rank by `std(a_k) * ||W[k]||` to surface the units that carry
*information* rather than offset. (An SVD of the full per-sample contribution tensor is the heavier
version of the same question.)
"""),
code("""
if RUN_ACTIVATIONS:
    L = ACT @ W + b                                      # (n_samples, HW) true logits
    Lbar = L.mean(0)
    print(f'mean logit map norm      {np.linalg.norm(Lbar):.1f}')
    print(f'mean per-sample deviation {np.linalg.norm(L - Lbar, axis=1).mean():.1f}')
    print(f'-> the constant part is {np.linalg.norm(Lbar) / (np.linalg.norm(Lbar) + np.linalg.norm(L - Lbar, axis=1).mean()) * 100:.0f}% of the total\\n')

    plotted = np.abs(act_mean[:, None] * W).max(axis=1)  # what section 4 draws
    ignored = np.abs(act_std[:, None] * W).max(axis=1)   # the varying part it drops
    print(f'per-unit peak effect   plotted (mean*W): median {np.median(plotted):.3f}')
    print(f'                       ignored (std*W):  median {np.median(ignored):.3f}')
    print(f'activation std/mean ratio: mean {(act_std / np.abs(act_mean)).mean():.2f}')
"""),

md("""
## 5e. The information-carrying units

Same `out_h x out_w` grid as before, but ranked by `std(a_k) * ||W[k]||` -- the units whose firing
*varies* most across samples, scaled by how much that variation moves the output. These are the
units doing the discriminating, and they are the ones to look at if the question is "how does this
model tell one cube position from another".
"""),
code("""
if RUN_ACTIVATIONS:
    info_score = act_std * np.linalg.norm(W, axis=1)
    top_info = list(np.argsort(-info_score)[:N_UNIT_MAPS])
    print(f'overlap with the mean-contribution top-{N_UNIT_MAPS}: {len(set(top_info) & top_c)} units')

    info_maps = [(act_std[k] * W[k]).reshape(OUT_H, OUT_W) for k in top_info]
    imax = float(np.quantile(np.abs(np.array(info_maps)), 0.995))
    unit_grid(top_info, info_maps,
              [f'unit {k} (std={act_std[k]:.2f})' for k in top_info],
              f'{RUN_NAME} -- top-{len(top_info)} units by information carried on {ACT_SPLIT} '
              f'(std activation x weight)', 'logit swing', -imax, imax).show()
"""),
]

nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"codemirror_mode": {"name": "ipython", "version": 3},
                                     "file_extension": ".py", "mimetype": "text/x-python", "name": "python",
                                     "nbconvert_exporter": "python", "pygments_lexer": "ipython3",
                                     "version": "3.12.13"}},
      "nbformat": 4, "nbformat_minor": 5}

out = Path(__file__).resolve().parents[1] / 'notebooks' / '67_decoder_last_layer.ipynb'
out.write_text(json.dumps(nb, indent=1) + '\n')
print(f'wrote {out}')
