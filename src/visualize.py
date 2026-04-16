"""Interactive training visualizer for vibration-to-segmentation runs.

Usage (in a Jupyter notebook):
    from visualize import TrainingVisualizer
    viz = TrainingVisualizer("my-run-20250101-120000")
    viz.load()
    viz.show(n_train=3, n_eval=3)
"""

import numpy as np
import wandb
import ipywidgets as widgets
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from IPython.display import display

from helpers import fetch_predictions, fetch_overhead_images, fetch_wandb_history


_METRIC_KEYS = [
    'metrics/train/mask/iou',
    'metrics/eval/mask/iou',
    'metrics/train/mask/mse',
    'metrics/eval/mask/mse',
    'loss/train/total',
]


class TrainingVisualizer:
    def __init__(self, run_id, data_dir="eturok-weizmann/good-vibrations",
                 entity="eturok", project="good-vibrations"):
        self.run_id = run_id
        self.data_dir = data_dir
        self.entity = entity
        self.project = project
        self.predictions = None   # {'train': {epoch: npz}, 'eval': {epoch: npz}}
        self.history = None       # list of dicts from W&B
        self.overhead_images = {} # {sample_idx: PIL.Image}
        self.run_config = {}

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load(self):
        """Download predictions, metrics history, overhead images, and run config."""
        print(f"Fetching predictions for run '{self.run_id}' …")
        self.predictions = fetch_predictions(self.run_id, self.data_dir)

        n_train = sum(len(v) for v in self.predictions['train'].values())
        n_eval  = sum(len(v) for v in self.predictions['eval'].values())
        print(f"  train epochs: {len(self.predictions['train'])}, "
              f"eval epochs: {len(self.predictions['eval'])}")

        print("Fetching W&B history …")
        self.history = fetch_wandb_history(self.run_id, keys=['_step'] + _METRIC_KEYS,
                                           entity=self.entity, project=self.project)
        # Build epoch→metrics lookup (W&B logs metrics once per epoch)
        self._metrics_by_epoch = {}
        for row in self.history:
            if any(k in row for k in _METRIC_KEYS):
                ep = row.get('_step')
                if ep is not None:
                    self._metrics_by_epoch[ep] = row

        print("Fetching run config …")
        api = wandb.Api()
        run = api.run(f"{self.entity}/{self.project}/{self.run_id}")
        self.run_config = dict(run.config)

        print("Fetching overhead images …")
        all_idxs = set()
        for split_data in self.predictions.values():
            for npz in split_data.values():
                all_idxs.update(npz['sample_idx'].tolist())
        self.overhead_images = fetch_overhead_images(list(all_idxs))
        print(f"  loaded {len(self.overhead_images)} overhead images")
        print("Done.")

    # ------------------------------------------------------------------
    # Widget
    # ------------------------------------------------------------------

    def show(self, n_train=3, n_eval=3, display_mode='overlay'):
        """Render the interactive widget in a Jupyter notebook cell.

        Args:
            n_train: Number of train samples to show per row.
            n_eval:  Number of eval samples to show per row.
            display_mode: One of 'overlay', 'side-by-side', 'difference'.
        """
        if self.predictions is None:
            raise RuntimeError("Call .load() first.")

        eval_epochs = sorted(self.predictions['eval'].keys())
        if not eval_epochs:
            raise RuntimeError("No eval predictions found.")

        # Fixed sample selection: take the first n_train / n_eval unique sample_idxs
        train_sample_idxs = self._pick_sample_idxs('train', n_train)
        eval_sample_idxs  = self._pick_sample_idxs('eval',  n_eval)
        all_cols = (
            [('train', i) for i in train_sample_idxs] +
            [('eval',  i) for i in eval_sample_idxs]
        )
        n_cols = len(all_cols)

        # ---- controls ------------------------------------------------
        epoch_slider = widgets.IntSlider(
            value=eval_epochs[-1], min=eval_epochs[0], max=eval_epochs[-1],
            step=(eval_epochs[1] - eval_epochs[0]) if len(eval_epochs) > 1 else 1,
            description='Epoch', continuous_update=False,
            layout=widgets.Layout(width='60%'))
        play = widgets.Play(
            value=eval_epochs[-1], min=eval_epochs[0], max=eval_epochs[-1],
            step=(eval_epochs[1] - eval_epochs[0]) if len(eval_epochs) > 1 else 1,
            interval=400, description='Play')
        widgets.jslink((play, 'value'), (epoch_slider, 'value'))

        mode_toggle = widgets.ToggleButtons(
            options=['overlay', 'side-by-side', 'difference'],
            value=display_mode, description='Display')

        opacity_slider = widgets.FloatSlider(
            value=0.5, min=0.1, max=1.0, step=0.05,
            description='Opacity', continuous_update=False,
            layout=widgets.Layout(width='40%'))
        opacity_slider.layout.visibility = 'visible' if display_mode == 'overlay' else 'hidden'

        # ---- header --------------------------------------------------
        config_html = widgets.HTML(value=self._config_html())
        metrics_html = widgets.HTML(value=self._metrics_html(eval_epochs[-1]))

        # ---- figure --------------------------------------------------
        fig_widget = go.FigureWidget(
            self._build_figure(all_cols, eval_epochs[-1], display_mode, 0.5))

        # ---- callbacks -----------------------------------------------
        def on_epoch_change(change):
            epoch = change['new']
            # snap to nearest available eval epoch
            epoch = min(eval_epochs, key=lambda e: abs(e - epoch))
            with fig_widget.batch_update():
                self._update_figure(fig_widget, all_cols, epoch,
                                    mode_toggle.value, opacity_slider.value)
            metrics_html.value = self._metrics_html(epoch)

        def on_mode_change(change):
            mode = change['new']
            opacity_slider.layout.visibility = 'visible' if mode == 'overlay' else 'hidden'
            epoch = min(eval_epochs, key=lambda e: abs(e - epoch_slider.value))
            with fig_widget.batch_update():
                self._update_figure(fig_widget, all_cols, epoch, mode, opacity_slider.value)

        def on_opacity_change(change):
            epoch = min(eval_epochs, key=lambda e: abs(e - epoch_slider.value))
            with fig_widget.batch_update():
                self._update_figure(fig_widget, all_cols, epoch,
                                    mode_toggle.value, change['new'])

        epoch_slider.observe(on_epoch_change, names='value')
        mode_toggle.observe(on_mode_change, names='value')
        opacity_slider.observe(on_opacity_change, names='value')

        controls = widgets.HBox([play, epoch_slider, mode_toggle, opacity_slider])
        ui = widgets.VBox([config_html, metrics_html, fig_widget, controls])
        display(ui)

    # ------------------------------------------------------------------
    # Figure building
    # ------------------------------------------------------------------

    def _pick_sample_idxs(self, split, n):
        """Return up to n unique sample_idx values that appear in all eval epochs."""
        epochs = sorted(self.predictions[split].keys()) if self.predictions[split] else []
        if not epochs:
            return []
        # Use the last epoch for selection
        npz = self.predictions[split][epochs[-1]]
        seen = []
        for idx in npz['sample_idx'].tolist():
            if idx not in seen:
                seen.append(idx)
            if len(seen) == n:
                break
        return seen

    def _get_npz_row(self, split, epoch, sample_idx):
        """Return (mask_true, mask_pred, meta_dict) for one sample at one epoch."""
        epochs = sorted(self.predictions[split].keys())
        # snap to nearest available
        epoch = min(epochs, key=lambda e: abs(e - epoch))
        npz = self.predictions[split][epoch]
        idxs = npz['sample_idx'].tolist()
        if sample_idx not in idxs:
            return None, None, None
        i = idxs.index(sample_idx)
        meta = {
            'x_position': float(npz['x_position'][i]),
            'y_position': float(npz['y_position'][i]),
            'object':     str(npz['object_type'][i]),
            'n_objects':  int(npz['n_objects'][i]),
        }
        return npz['mask_true'][i], npz['mask_pred'][i], meta

    def _build_figure(self, all_cols, epoch, mode, opacity):
        n_cols = len(all_cols)
        # side-by-side needs 2 sub-columns per sample
        plot_cols = n_cols * 2 if mode == 'side-by-side' else n_cols
        col_titles = []
        for split, sample_idx in all_cols:
            mt, mp, meta = self._get_npz_row(split, epoch, sample_idx)
            label = split.upper()
            pos = f"({meta['x_position']:.1f}, {meta['y_position']:.1f})" if meta else ""
            obj = meta['object'] if meta else ""
            if mode == 'side-by-side':
                col_titles += [f"[{label}] #{sample_idx}<br>{obj} {pos}<br>GT", "Pred"]
            else:
                col_titles.append(f"[{label}] #{sample_idx}<br>{obj} {pos}")

        fig = make_subplots(rows=1, cols=plot_cols,
                            subplot_titles=col_titles,
                            horizontal_spacing=0.02)

        for ci, (split, sample_idx) in enumerate(all_cols):
            self._add_sample_traces(fig, ci, split, sample_idx, epoch, mode, opacity)

        fig.update_layout(
            height=500, margin=dict(t=80, b=20, l=10, r=10),
            paper_bgcolor='#1a1a1a', plot_bgcolor='#1a1a1a',
            font=dict(color='white'),
            showlegend=False)
        fig.update_xaxes(showticklabels=False, showgrid=False)
        fig.update_yaxes(showticklabels=False, showgrid=False, scaleanchor=None)
        return fig

    def _add_sample_traces(self, fig, col_idx, split, sample_idx, epoch, mode, opacity):
        mt, mp, meta = self._get_npz_row(split, epoch, sample_idx)
        if mt is None:
            return
        H, W = mt.shape
        overhead = self.overhead_images.get(sample_idx)
        oh_arr = np.array(overhead) if overhead is not None else None

        iou = float(np.mean((mt > 0.5) & (mp > 0.5))) / max(float(np.mean((mt > 0.5) | (mp > 0.5))), 1e-6)
        iou_text = f"IoU: {iou:.3f}"

        if mode == 'side-by-side':
            plot_col_gt   = col_idx * 2 + 1
            plot_col_pred = col_idx * 2 + 2
            self._add_mask_subplot(fig, 1, plot_col_gt,   oh_arr, mt, 'gray',   opacity, iou_text)
            self._add_mask_subplot(fig, 1, plot_col_pred, oh_arr, mp, 'Hot',    opacity, iou_text)
        elif mode == 'overlay':
            plot_col = col_idx + 1
            if oh_arr is not None:
                fig.add_trace(go.Image(z=oh_arr, hoverinfo='skip'), row=1, col=plot_col)
            # pred heatmap
            fig.add_trace(go.Heatmap(
                z=mp, colorscale='Hot', zmin=0, zmax=1, opacity=opacity,
                showscale=False, hovertemplate='pred: %{z:.3f}<extra></extra>',
                xaxis=f'x{plot_col}', yaxis=f'y{plot_col}'), row=1, col=plot_col)
            # GT as green contour
            fig.add_trace(go.Contour(
                z=mt, contours_coloring='lines', colorscale=[[0, 'green'], [1, 'green']],
                line_width=2, showscale=False,
                hovertemplate='gt: %{z:.3f}<extra></extra>'), row=1, col=plot_col)
            fig.add_annotation(text=iou_text, xref=f'x{plot_col} domain', yref=f'y{plot_col} domain',
                               x=0.5, y=0.02, showarrow=False, font=dict(color='white', size=11))
        else:  # difference
            plot_col = col_idx + 1
            diff = np.abs(mt - mp)
            if oh_arr is not None:
                fig.add_trace(go.Image(z=oh_arr, hoverinfo='skip'), row=1, col=plot_col)
            fig.add_trace(go.Heatmap(
                z=diff, colorscale='RdBu_r', zmin=0, zmax=1, opacity=opacity,
                showscale=False, hovertemplate='|gt−pred|: %{z:.3f}<extra></extra>'), row=1, col=plot_col)
            fig.add_annotation(text=iou_text, xref=f'x{plot_col} domain', yref=f'y{plot_col} domain',
                               x=0.5, y=0.02, showarrow=False, font=dict(color='white', size=11))

    def _add_mask_subplot(self, fig, row, col, oh_arr, mask, colorscale, opacity, annotation):
        if oh_arr is not None:
            fig.add_trace(go.Image(z=oh_arr, hoverinfo='skip'), row=row, col=col)
        fig.add_trace(go.Heatmap(
            z=mask, colorscale=colorscale, zmin=0, zmax=1, opacity=opacity,
            showscale=False, hovertemplate='%{z:.3f}<extra></extra>'), row=row, col=col)
        fig.add_annotation(text=annotation, xref=f'x{col} domain', yref=f'y{col} domain',
                           x=0.5, y=0.02, showarrow=False, font=dict(color='white', size=11))

    def _update_figure(self, fig_widget, all_cols, epoch, mode, opacity):
        """Update trace data in-place without rebuilding the full figure."""
        # Rebuild is simpler and fast enough for this use case
        new_fig = self._build_figure(all_cols, epoch, mode, opacity)
        fig_widget.data = new_fig.data
        fig_widget.layout = new_fig.layout

    # ------------------------------------------------------------------
    # Header HTML
    # ------------------------------------------------------------------

    def _config_html(self):
        cfg = self.run_config
        fields = ['loss', 'gamma', 'decoder', 'd_model', 'batch_size', 'lr',
                  'seed', 'signal_is', 'normalize', 'n_params']
        chips = []
        for k in fields:
            if k in cfg:
                chips.append(f'<span style="background:#333;padding:2px 8px;border-radius:4px;margin:2px">'
                             f'<b>{k}</b>: {cfg[k]}</span>')
        return (f'<div style="font-family:monospace;font-size:12px;color:#ccc;'
                f'background:#111;padding:8px;border-radius:6px">'
                f'<b style="color:white">run:</b> {self.run_id}&nbsp;&nbsp;'
                + ' '.join(chips) + '</div>')

    def _metrics_html(self, epoch):
        # Find the history row closest to this epoch
        if not self._metrics_by_epoch:
            return ''
        closest = min(self._metrics_by_epoch.keys(), key=lambda e: abs(e - epoch))
        row = self._metrics_by_epoch[closest]

        def fmt(key): return f'{row[key]:.4f}' if key in row and row[key] is not None else '—'

        return (
            f'<div style="font-family:monospace;font-size:13px;color:#eee;'
            f'background:#1a1a1a;padding:6px 12px;border-radius:6px;border-left:3px solid #555">'
            f'<b>Epoch {epoch}</b> &nbsp;|&nbsp; '
            f'Train IoU: <b>{fmt("metrics/train/mask/iou")}</b> &nbsp;'
            f'Eval IoU: <b>{fmt("metrics/eval/mask/iou")}</b> &nbsp;|&nbsp; '
            f'Train MSE: <b>{fmt("metrics/train/mask/mse")}</b> &nbsp;'
            f'Eval MSE: <b>{fmt("metrics/eval/mask/mse")}</b> &nbsp;|&nbsp; '
            f'Loss: <b>{fmt("loss/train/total")}</b>'
            f'</div>'
        )
