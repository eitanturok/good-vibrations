"""Post-training PCA plots: dataset FFT vs run logits, side by side, interactive.

Port of notebooks/47_pca.ipynb for the MDS datasets: the dataset panel PCA's each sample's
tokenized FFT (X averaged over lasers + x/y channels); the run panel PCA's the final-epoch
predicted-mask logits saved by OutputSaver. One plotly HTML with a color-by dropdown is
written to {run_path}/pca.html and logged to wandb when a run is active.
"""
import colorsys
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import wandb
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.decomposition import PCA
from streaming import LocalDataset

COLOR_BY = ['speaker', 'layout', 'split', 'n_objects', 'com', 'com_x', 'com_y']
CONTINUOUS = {'com_x', 'com_y'}
# validated 8-slot categorical palette (worst adjacent-pair CVD Delta-E 24.2) -- same slots +
# order as notebooks/47_pca.ipynb; order is the CVD-safety mechanism, keep as-is. Cycles past 8
# categories with a different marker symbol per cycle so repeated hues stay distinguishable.
PALETTE = ['#2a78d6', '#1baf7a', '#eda100', '#008300', '#4a3aa7', '#e34948', '#e87ba4', '#eb6834']
SYMBOLS = ['circle', 'diamond', 'square', 'cross']
EMPTY_BOX_COLOR = '#000000'
HOVER = 'sample_id=%{customdata}<br>PC1=%{x:.2f}<br>PC2=%{y:.2f}<br>PC3=%{z:.2f}<extra></extra>'
COM_HOVER = ('sample_id=%{customdata[0]:.0f}<br>com=(%{customdata[1]:.1f}, %{customdata[2]:.1f})'
             '<br>PC1=%{x:.2f}<br>PC2=%{y:.2f}<br>PC3=%{z:.2f}<extra></extra>')
COM_LEGEND_SIZE = 120


def com_ranges(df):
    # observed (min, max) of com_x/com_y across valid samples -- the (-1,-1) empty-box sentinel is
    # excluded, and this is deliberately NOT the full 0..out_h/0..out_w mask grid: real COM values
    # only occupy a fraction of the grid, so normalizing by grid size crams every point into one
    # corner of the color space (notebook 47).
    valid = df[(df['com_x'] >= 0) & (df['com_y'] >= 0)]
    return (valid['com_x'].min(), valid['com_x'].max()), (valid['com_y'].min(), valid['com_y'].max())


def _com_to_hsl(com_x, com_y, x_range, y_range):
    # notebook 47's 2d com coloring: hue ~ horizontal position (com_y), lightness ~ vertical
    # position (com_x); saturation maxed. Lightness spans 0.25-0.80 -- wider than the notebook's
    # 0.35-0.75, which made low vs high com_x barely distinguishable -- while still stopping
    # short of full white/black so the hue stays readable at both extremes.
    (x_lo, x_hi), (y_lo, y_hi) = x_range, y_range
    hue = 0.8 * np.clip((com_y - y_lo) / ((y_hi - y_lo) or 1), 0, 1)
    lightness = 0.80 - 0.55 * np.clip((com_x - x_lo) / ((x_hi - x_lo) or 1), 0, 1)
    return [f'rgb({r*255:.0f}, {g*255:.0f}, {b*255:.0f})'
            for h, l in zip(hue, lightness) for r, g, b in [colorsys.hls_to_rgb(h, l, 1.0)]]


def _com_swatch(x_range, y_range, size=COM_LEGEND_SIZE):
    # 2d color-key image for the com coloring: row -> com_x, col -> com_y, same mapping and same
    # observed ranges as _com_to_hsl, so the swatch is the legend a 1d colorbar can't be
    xs, ys = np.linspace(*x_range, size), np.linspace(*y_range, size)
    colors = _com_to_hsl(np.repeat(xs, size), np.tile(ys, size), x_range, y_range)
    return np.array([[int(v) for v in c[4:-1].split(',')] for c in colors], dtype=np.uint8).reshape(size, size, 3)


def _com_ticks(lo, hi, size=COM_LEGEND_SIZE, target_ticks=6):
    # swatch tick positions in pixel coords, labeled in real com units, at a "nice" 1/2/5 step
    raw = (hi - lo) / target_ticks
    mag = 10 ** np.floor(np.log10(raw)) if raw > 0 else 1
    step = next(m * mag for m in (1, 2, 5, 10) if raw <= m * mag)
    units = np.arange(np.ceil(lo / step) * step, hi + step / 2, step)
    return units, (units - lo) / ((hi - lo) or 1) * (size - 1)


def load_meta(mds_dir):
    # metadata.jsonl sidecar -> one row per sample (sample_id is a zero-padded str -> int)
    lines = (Path(mds_dir) / 'metadata.jsonl').read_text().strip().splitlines()
    rows = [json.loads(line) for line in lines if line]
    return pd.DataFrame([{'sample_id': int(r['sample_id']), 'layout': r['layout'], 'speaker': r['speaker'],
                          'n_objects': r['n_objects'], 'com_x': r['downsampled_com'][0], 'com_y': r['downsampled_com'][1]}
                         for r in rows])


def load_dataset_features(mds_dir):
    # sample_id -> mean tokenized-FFT spectrum (avg over lasers + x/y channels): the same reduction
    # as notebook 47's compute_fft_magnitude, but on the MDS X the model actually sees. LocalDataset
    # (not StreamingDataset) so this can coexist with the training dataset on the same local dir.
    ds = LocalDataset(local=str(mds_dir))
    return {int(s['sample_id']): s['X'].mean(axis=(0, -1)).ravel() for s in (ds[i] for i in range(len(ds)))}


def _inverse_sigmoid(p, eps=1e-6):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def load_run_logits(outputs_dir):
    # sample_id -> flattened final-epoch logit mask, and sample_id -> split, from OutputSaver dumps
    # (mask_logits directly, or recovered via inverse sigmoid for runs that predate it being saved)
    outputs_dir = Path(outputs_dir)
    split_dirs = {'train': outputs_dir / 'train'}
    if (outputs_dir / 'eval').exists():
        split_dirs |= {f'eval/{p.name}': p for p in sorted((outputs_dir / 'eval').iterdir()) if p.is_dir()}

    def epoch(p): return int(p.stem.split('-')[0].removeprefix('ep'))
    logits, splits = {}, {}
    for split, split_dir in split_dirs.items():
        pts = sorted(split_dir.glob('*.pt')) if split_dir.exists() else []
        if not pts: continue
        last = max(map(epoch, pts))
        for pt in (p for p in pts if epoch(p) == last):
            batch = torch.load(pt, map_location='cpu', weights_only=False)
            x = batch['mask_logits'].float().numpy() if 'mask_logits' in batch else _inverse_sigmoid(batch['mask_pred'].float().numpy())
            for i, sid in enumerate(batch['info']['sample_id'].tolist()):
                logits[sid] = x[i].ravel()
                splits[sid] = split
    return logits, splits


def _traces(df, xyz, color_by, primary, com_rng=None):
    # primary: the panel that owns the single shared legend/colorbar (the dataset panel).
    # legendgroup is shared across panels so a legend click toggles the category on both.
    marker = dict(size=6, opacity=0.9, line=dict(color='white', width=0.6))
    def scatter(mask, marker, customdata=None, hovertemplate=HOVER, **kwargs):
        return go.Scatter3d(x=xyz[mask, 0], y=xyz[mask, 1], z=xyz[mask, 2], mode='markers', marker=marker,
                            customdata=df['sample_id'].to_numpy()[mask] if customdata is None else customdata,
                            hovertemplate=hovertemplate, **kwargs)

    if color_by == 'com':
        # notebook 47's 2d com coloring; com_rng is computed once over the whole dataset by pca()
        # so both panels and the swatch legend share the exact same color mapping. Empty-box
        # (-1,-1) sentinel stays a fixed black trace, same as the continuous options.
        empty = ((df['com_x'] < 0) | (df['com_y'] < 0)).to_numpy()
        traces = []
        if (~empty).any():
            colors = _com_to_hsl(df['com_x'].to_numpy()[~empty], df['com_y'].to_numpy()[~empty], *com_rng)
            traces.append(scatter(~empty, dict(marker, color=colors),
                                  customdata=df[['sample_id', 'com_x', 'com_y']].to_numpy()[~empty],
                                  hovertemplate=COM_HOVER, showlegend=False))
        if empty.any():
            traces.append(scatter(empty, dict(marker, color=EMPTY_BOX_COLOR), name='empty box',
                                  legendgroup='empty-box', showlegend=primary))
        return traces

    if color_by in CONTINUOUS:
        # empty-box samples record com as the (-1,-1) sentinel: a fixed-black trace, not part of the
        # colorscale, so they neither clamp to an extreme nor drag the colorbar range below 0
        empty = ((df['com_x'] < 0) | (df['com_y'] < 0)).to_numpy()
        traces = []
        if (~empty).any():
            m = dict(marker, color=df[color_by].to_numpy()[~empty], colorscale='Viridis', showscale=primary)
            # colorbar sits below the legend (anchored at the top), not on top of it
            if primary: m['colorbar'] = dict(title=color_by, x=1.0, y=0.42, yanchor='middle', len=0.65)
            traces.append(scatter(~empty, m, showlegend=False))
        if empty.any():
            traces.append(scatter(empty, dict(marker, color=EMPTY_BOX_COLOR), name='empty box',
                                  legendgroup='empty-box', showlegend=primary))
        return traces

    values = sorted(df[color_by].unique(), key=str)
    return [scatter((df[color_by] == val).to_numpy(),
                    dict(marker, color=PALETTE[i % len(PALETTE)], symbol=SYMBOLS[(i // len(PALETTE)) % len(SYMBOLS)]),
                    name=str(val), legendgroup=f'{color_by}-{val}', showlegend=primary)
            for i, val in enumerate(values)]


def pca(mds_dir, run_path, verbose: int = 1):
    """Build the linked PCA figure for a finished run and log it (see module docstring).

    run_path may be a run dir (containing outputs_history/ or outputs/) or an OutputSaver
    outputs dir itself. Returns the path of the written HTML.
    """
    run_path = Path(run_path)
    outputs_dir = next((d for d in (run_path / 'outputs_history', run_path / 'outputs', run_path)
                        if (d / 'train').exists() or (d / 'eval').exists()), None)
    assert outputs_dir is not None, f'no OutputSaver outputs found under {run_path}'

    logits, splits = load_run_logits(outputs_dir)
    meta = load_meta(mds_dir)
    meta['split'] = meta['sample_id'].map(splits).fillna('unsaved')  # e.g. train samples dropped by drop_last

    def fit(features):
        sids = sorted(features)
        p = PCA(n_components=3)
        xyz = p.fit_transform(np.stack([features[sid] for sid in sids]))
        return pd.DataFrame({'sample_id': sids}).merge(meta, on='sample_id', how='left'), xyz, p.explained_variance_ratio_

    # notebook 47's panel titles: what was PCA'd on the first line; dataset/run name + explained
    # variance underneath in small gray, so each panel is self-identifying even out of context
    sub = '<br><span style="font-size:11px;color:#888">{} -- Explained Variance {:.0%}</span>'
    mds_name = Path(mds_dir).parent.name if Path(mds_dir).name == 'mds' else Path(mds_dir).name
    panels = [('PCA on Model Input (Normalized FFT Magnitude)', mds_name, fit(load_dataset_features(mds_dir)))]
    if logits: panels.append(('PCA on Model Output (Logits)', run_path.name, fit(logits)))

    com_rng = com_ranges(meta)

    fig = make_subplots(rows=1, cols=len(panels), specs=[[{'type': 'scene'}] * len(panels)], horizontal_spacing=0.02,
                        subplot_titles=[name + sub.format(src, evr.sum())
                                        for name, src, (_, _, evr) in panels])
    # scenes share [0, 0.86] of the width; the right strip holds the legend/colorbar/com swatch
    n, gap = len(panels), 0.03
    width = (0.86 - gap * (n - 1)) / n
    for i in range(n):
        lo = i * (width + gap)
        fig.layout['scene' if i == 0 else f'scene{i + 1}'].domain.x = (lo, lo + width)
        fig.layout.annotations[i].x = lo + width / 2  # re-center the subplot title over its scene

    trace_opts = []  # color_by option of every trace, in fig.data order -> dropdown visibility masks
    for opt in COLOR_BY:
        for col, (_, _, (df, xyz, _)) in enumerate(panels, start=1):
            for trace in _traces(df, xyz, opt, primary=col == 1, com_rng=com_rng):
                trace.visible = opt == COLOR_BY[0]
                fig.add_trace(trace, row=1, col=col)
                trace_opts.append(opt)
        if opt == 'com':
            fig.add_trace(go.Image(z=_com_swatch(*com_rng), hoverinfo='skip', visible=opt == COLOR_BY[0]))
            trace_opts.append(opt)
            # hover marker on the swatch (notebook 47's live com marker): moved to the hovered
            # point's (com_x, com_y) by the post_script JS below, hidden (x/y None) until then
            fig.add_trace(go.Scatter(x=[None], y=[None], mode='markers', hoverinfo='skip', showlegend=False,
                                     marker=dict(size=14, color='white', line=dict(color='black', width=2),
                                                 symbol='circle-open'), visible=opt == COLOR_BY[0]))
            marker_idx = len(fig.data) - 1
            trace_opts.append(opt)

    # com color-key swatch, modeled on notebook 47: it sits in the right strip where the
    # legend/colorbar would be, with tick labels in real com units and a white dotted grid drawn
    # as layout shapes (axis gridlines never render above an image trace). Axes and shapes are
    # layout-level, so each dropdown button toggles them alongside the trace visibilities.
    (x_units, x_px), (y_units, y_px) = _com_ticks(*com_rng[0]), _com_ticks(*com_rng[1])
    # fixed ranges with a small pad past the image so the border doesn't clip; scaleanchor keeps
    # the swatch square regardless of the browser window's aspect ratio
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
                      height=640, margin=dict(l=0, r=0, t=100, b=0),
                      legend=dict(title=COLOR_BY[0], x=1.0, y=1, yanchor='top'),
                      # the swatch's cartesian plot area would otherwise render as a gray
                      # rectangle in the right strip even with its axes and image hidden
                      plot_bgcolor='rgba(0,0,0,0)',
                      shapes=com_shapes if COLOR_BY[0] == 'com' else [])
    fig.update_scenes(xaxis_title='PC1', yaxis_title='PC2', zaxis_title='PC3')

    # hover on a 3d point moves the swatch marker to that point's (com_x, com_y) -- the static-HTML
    # version of notebook 47's FigureWidget hover callback. Only com traces carry the 3-column
    # customdata (sample_id, com_x, com_y); everything else (incl. the empty-box sentinel trace,
    # whose com isn't a real position) has scalar customdata and is skipped.
    (x_lo, x_hi), (y_lo, y_hi) = com_rng
    post_script = f"""
var gd = document.getElementById('{{plot_id}}');
var M = {marker_idx}, S = {COM_LEGEND_SIZE - 1};
function comPx(v, lo, hi) {{ return Math.min(Math.max((v - lo) / ((hi - lo) || 1), 0), 1) * S; }}
gd.on('plotly_hover', function(ev) {{
  var pt = ev.points[0];
  if (!pt || pt.data.type !== 'scatter3d' || !Array.isArray(pt.customdata) || gd.data[M].visible !== true) return;
  Plotly.restyle(gd, {{x: [[comPx(pt.customdata[2], {y_lo}, {y_hi})]],
                       y: [[comPx(pt.customdata[1], {x_lo}, {x_hi})]]}}, [M]);
}});
gd.on('plotly_unhover', function() {{ Plotly.restyle(gd, {{x: [[null]], y: [[null]]}}, [M]); }});
"""

    html_path = (outputs_dir.parent if outputs_dir != run_path else run_path) / 'pca.html'
    html = fig.to_html(include_plotlyjs=True, full_html=True, default_width='100%', default_height='640px',
                       post_script=post_script)
    html_path.write_text(html)
    if wandb.run is not None: wandb.log({'pca': wandb.Html(html, inject=False)})
    if verbose: print(f"pca plot: {html_path}{' (logged to wandb)' if wandb.run is not None else ''}")
    return html_path


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--mds-dir', required=True)
    parser.add_argument('--run-path', required=True, help='run dir (runs/<name>) or an OutputSaver outputs dir')
    args = parser.parse_args()
    pca(args.mds_dir, args.run_path)
