"""Which lasers and which frequency bands does the model actually use?

Two families of measure live here:

- *attention*: the cls-row of each self-attention map, read straight off the encoders.
  Cheap (free with the forward pass) but a weak proxy -- see the note on `ablate_lasers`.
- *ablation*: zero an input and measure how much true BCE degrades. This is the metric to
  lead with. (Pass `metric=_mse` to any ablation helper for the old squared-error view.)

The architecture factorizes the two axes (freq attention runs per-laser and never sees other
lasers; laser attention runs on tokens where frequency has already been collapsed into one cls
vector), so no attention map anywhere is jointly indexed by (laser, freq). Only
`ablate_laser_freq` can answer that.
"""

from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch

from model.arch import VibrationTransformer

ROOT = Path(__file__).resolve().parents[2]

# `old-config` trained on *truncated* FFTs: 2946 bins -> 11 patches of 256, dropping bins
# 2816-2945 (958.4-1000 Hz). The current tokenize() zero-pads to 12 instead, so n_freqs must be
# 11*256 there or the (11,128) RoPE buffer won't line up with the 12 patches and apply_rope raises.
# `infer_data_info` reads the count back off the checkpoint, so this trap is handled per-run.
OLD_CONFIG_DATA_INFO = dict(out_h=20, out_w=40, n_laser_rows=10, n_laser_cols=10,
                            patch_size=256, n_freqs=11 * 256, n_channels=2)
OLD_CONFIG_MDS = ROOT / 'experiments/experiment-25/mds/19b982fa1e001563'
OLD_CONFIG_CKPT = ROOT / 'runs/old-config/checkpoints/latest-rank0.pt'

# Checkpoints don't record which dataset they trained on, so a run's MDS is pinned here when it
# matters. Anything not listed is resolved by shape -- see `run_mds`.
RUN_MDS = {'old-config': OLD_CONFIG_MDS}
DEFAULT_MDS = OLD_CONFIG_MDS
MDS_GLOB = 'experiments/*/mds/*/'


# ***** loading *****

def run_ckpt(run_name, ckpt='latest-rank0.pt'):
    """Path to a run's checkpoint. `runs/<run_name>/checkpoints/<ckpt>`."""
    p = ROOT / 'runs' / run_name / 'checkpoints' / ckpt
    if not p.exists():
        have = sorted(q.name for q in p.parent.glob('*.pt')) if p.parent.is_dir() else []
        raise FileNotFoundError(f"no checkpoint {p}" + (f"; this run has {have}" if have else ""))
    return p


def find_mds(n_lasers, n_patches, patch_size=256, n_channels=2, out_dim=None, root=None):
    """MDS dirs whose samples match (n_lasers, n_patches, patch_size, n_channels).

    With `out_dim` (= out_h*out_w) the mask shape is matched too, which is what separates datasets
    that share an input shape but predict different grids. Several datasets differ only in
    normalization, so this can still return more than one.
    """
    from streaming import StreamingDataset
    want, hits = (n_lasers, n_patches, patch_size, n_channels), []
    for d in sorted(Path(root or ROOT).glob(MDS_GLOB)):
        if not (d / 'index.json').exists(): continue
        try:
            s = StreamingDataset(local=str(d), shuffle=False, batch_size=1)[0]
            if tuple(s['X'].shape) != want: continue
            if out_dim is not None and int(np.prod(s['y'].shape)) != out_dim: continue
            hits.append((d, tuple(s['y'].shape)))
        except Exception:
            continue  # half-written or partially-downloaded shard dirs
    return hits


def run_mds(run_name, mds_dir=None, data_info=None, verbose=True):
    """The MDS dir to evaluate a run against.

    Explicit `mds_dir` wins, then the `RUN_MDS` table, then -- given `data_info` -- a search for a
    dataset whose sample shape matches the checkpoint. That last step is a *guess*: normalization
    is not visible in the shape, and using a dataset the run wasn't trained on shows up as an MSE
    far above baseline, so check the printed MSE before trusting results.
    """
    if mds_dir is not None: return Path(mds_dir)
    if run_name in RUN_MDS: return Path(RUN_MDS[run_name])
    if data_info is None: return Path(DEFAULT_MDS)

    hits = find_mds(data_info['n_laser_rows'] * data_info['n_laser_cols'],
                    data_info['n_freqs'] // data_info['patch_size'],
                    data_info['patch_size'], data_info['n_channels'],
                    out_dim=data_info['out_h'] * data_info['out_w'])
    if not hits:
        raise FileNotFoundError(f"no MDS matches {run_name}'s shape; pass mds_dir=")
    # the dataset's mask shape is authoritative -- the decoder only fixes out_h*out_w, so this is
    # where a wrong _factor_grid guess gets corrected.
    d, y_shape = hits[0]
    if (data_info['out_h'], data_info['out_w']) != y_shape:
        if verbose: print(f"  mask {y_shape} from the dataset (decoder only pins the product)")
        data_info['out_h'], data_info['out_w'] = y_shape
    if verbose and len(hits) > 1:
        print(f"  {len(hits)} datasets match, using {d.name}; others: {[h[0].name for h in hits[1:]]}. "
              f"Pass mds_dir= to choose.")
    return d


def _state_dict(ckpt_path):
    ck = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    return {k.removeprefix('_orig_mod.'): v for k, v in ck['state']['model'].items()}  # --compile 1 prefixes keys


def _factor_grid(out_dim, prefer=(20, 30, 24, 22, 25, 16, 32, 40)):
    """Split a decoder output size into (out_h, out_w). Only the product is stored, so this is a
    guess -- it tries the heights this project actually uses, commonest first, before falling back
    to the most square factorization. Pass out_h=/out_w= to `load_run` when a run differs."""
    for h in prefer:
        if out_dim % h == 0: return h, out_dim // h
    h = max(d for d in range(1, int(out_dim ** 0.5) + 1) if out_dim % d == 0)
    return h, out_dim // h


def infer_data_info(state, n_laser_rows=10, n_laser_cols=10, n_channels=2, out_h=None, out_w=None,
                    n_freqs=None):
    """Recover the `data_info` / arch hyperparameters a checkpoint was built with, from its shapes.

    Nothing in the checkpoint records the argparse config, but every architectural parameter is
    pinned by a tensor shape -- which is the stricter source anyway, since it is exactly what
    load_state_dict will accept. Notably `n_freqs` comes from the *stored* RoPE table, so a run
    that truncated the FFT and a run that padded it each rebuild correctly with no special-casing.

    Only the out_h/out_w split is ambiguous (the decoder pins their product), and the laser grid
    is not represented at all when it is square -- pass those explicitly if a run differs.
    """
    if 'freq_encoder.embed.weight' not in state:
        raise ValueError(
            "checkpoint has no `freq_encoder.*` -- it predates the laser_encoder -> freq_encoder "
            "rename, so today's VibrationTransformer cannot hold these weights. Use a newer run.")

    d_model = state['cls_token'].shape[-1]
    patch_size = state['freq_encoder.embed.weight'].shape[1] // n_channels
    n_layers = lambda pre: len({k[len(pre):].split('.')[0] for k in state if k.startswith(pre)})

    # n_patches comes from the *stored* RoPE table, so a run that truncated the FFT and a run that
    # padded it each rebuild correctly. --no-rope runs never register the buffer, so ask the caller.
    if 'freq_encoder.freqs_cis' in state:
        n_patches = state['freq_encoder.freqs_cis'].shape[0]
    elif n_freqs is not None:
        n_patches = n_freqs // patch_size
    else:
        raise ValueError("checkpoint has no freq_encoder.freqs_cis (a --no-rope run); pass n_freqs=")

    if 'decoder.query_seed' in state:      # AttnDecoder: the output grid is stored exactly
        decoder = 'attn' if 'decoder.freqs_query' in state else 'attn-no-rope'
        # v2 gives query_seed one row per output position; v1 shared a single seed and only
        # freqs_query carried the grid. Prefer freqs_query, which both versions size identically.
        out_dim = state['decoder.freqs_query'].shape[0] if decoder == 'attn' else state['decoder.query_seed'].shape[1]
        if state['decoder.query_seed'].shape[1] not in (1, out_dim):
            raise ValueError(f"unrecognized AttnDecoder query_seed {tuple(state['decoder.query_seed'].shape)}")
        if state['decoder.query_seed'].shape[1] == 1 != out_dim:
            raise ValueError("checkpoint uses the v1 shared-seed AttnDecoder, which current arch.py "
                             "no longer builds (it now learns one query per output position).")
    elif 'decoder.query_pos' in state:
        raise ValueError("checkpoint uses the v1 `query_pos` AttnDecoder, replaced by `query_seed` "
                         "in current arch.py. Use a newer run.")
    else:                                  # MLP*: only out_h*out_w is recoverable, never the split
        decoder = 'mlp-mid' if 'decoder.net.4.weight' in state else 'mlp'
        last = max(int(k.split('.')[2]) for k in state if k.startswith('decoder.net.') and k.endswith('.weight'))
        out_dim = state[f'decoder.net.{last}.weight'].shape[0]

    if out_h is None and out_w is None: out_h, out_w = _factor_grid(out_dim)
    elif out_h is None: out_h = out_dim // out_w
    elif out_w is None: out_w = out_dim // out_h
    assert out_h * out_w == out_dim, f"out_h*out_w={out_h * out_w} != decoder output {out_dim}; pass out_h=/out_w="
    assert state['freqs_laser'].shape[0] == n_laser_rows * n_laser_cols, \
        f"laser grid {n_laser_rows}x{n_laser_cols} != {state['freqs_laser'].shape[0]} lasers"

    data_info = dict(out_h=out_h, out_w=out_w, n_laser_rows=n_laser_rows, n_laser_cols=n_laser_cols,
                     patch_size=patch_size, n_freqs=n_patches * patch_size, n_channels=n_channels)
    arch = dict(d_model=d_model, pnt_num_layers=n_layers('freq_encoder.layers.layers.'),
                seq_num_layers=n_layers('laser_encoder.layers.'), decoder=decoder,
                decoder_num_layers=n_layers('decoder.layers.layers.') or 2)
    return data_info, arch


def _match_ffn_width(model, state):
    """Resize any transformer FFN whose width disagrees with the checkpoint.

    `dim_feedforward` isn't a constructor argument of VibrationTransformer -- arch.py picks it
    (nn's 2048 default for the encoders, 4*d_model for AttnDecoder) and that choice has changed
    over time. Rather than edit shared model code, rebuild the two Linears at the stored width.
    """
    import torch.nn as nn
    for name, mod in model.named_modules():
        w = state.get(f'{name}.linear1.weight')
        if w is None or not hasattr(mod, 'linear1') or mod.linear1.weight.shape == w.shape: continue
        ff, d = w.shape
        mod.linear1 = nn.Linear(d, ff)
        mod.linear2 = nn.Linear(ff, d)


def load_run(run_name='old-config', ckpt='latest-rank0.pt', device=None, pnt_num_heads=2,
             seq_num_heads=2, decoder_num_heads=2, mds_dir=None, verbose=True, **info_overrides):
    """Rebuild any run from its checkpoint, in eval mode, with the arch inferred from shapes.

    Returns `(model, data_info, mds_dir)`. The dataset is resolved first, because its mask shape
    is the only reliable source for the out_h/out_w split (the decoder pins just their product).

    eval mode is load-bearing: it disables the 0.3 structured dropout, which is what makes
    key_padding_mask None and leaves the attention maps dense over all L+1 / P+1 positions.

    Head counts are the one thing shapes cannot pin (all heads share one packed in_proj_weight),
    so they are arguments. They do not affect whether the weights load -- only how captured
    attention is split -- but a wrong value silently reshapes the maps, so they are printed.
    """
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    state = _state_dict(run_ckpt(run_name, ckpt))
    data_info, arch = infer_data_info(state, **info_overrides)
    # resolve the dataset first: it corrects out_h/out_w before the decoder is built from them
    mds = run_mds(run_name, mds_dir, data_info, verbose=verbose)
    model = VibrationTransformer(arch['d_model'], pnt_num_heads, arch['pnt_num_layers'], seq_num_heads,
                                 arch['seq_num_layers'], data_info, arch['decoder'], decoder_num_heads,
                                 arch['decoder_num_layers'], freq_dropout=0.3, laser_dropout=0.3, loss_fn='mse')
    _match_ffn_width(model, state)
    model.load_state_dict(state)  # strict: any inference mistake fails here rather than silently
    if verbose:
        print(f"{run_name}: d_model={arch['d_model']} layers={arch['pnt_num_layers']}freq/{arch['seq_num_layers']}laser "
              f"heads={pnt_num_heads}/{seq_num_heads} decoder={arch['decoder']}")
        print(f"  {data_info['n_freqs'] // data_info['patch_size']} patches x {data_info['patch_size']} "
              f"= {data_info['n_freqs']} bins -> mask {data_info['out_h']}x{data_info['out_w']}   mds={mds.name}")
    return model.to(device).eval(), data_info, mds


def load_old_config(ckpt_path=OLD_CONFIG_CKPT, device=None, data_info=OLD_CONFIG_DATA_INFO):
    """`old-config` rebuilt from its checkpoint. Thin wrapper kept for the explicit-path case."""
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    model = VibrationTransformer(128, 2, 2, 2, 2, data_info, 'mlp', 2, 2, freq_dropout=0.3, laser_dropout=0.3, loss_fn='mse')
    model.load_state_dict(_state_dict(ckpt_path))
    return model.to(device).eval()


def load_eval_batch(n=96, mds_dir=OLD_CONFIG_MDS, device=None, exclude_empty=True, split='eval', seed=42,
                    test_size=0.20, return_info=False):
    """`n` held-out samples as (X, y), or (X, y, info) with `return_info`.

    `info` is a list of per-sample dicts (sample_id, position_id, speaker, box, n_objects, ...) in
    the same order as the rows of X, for picking out or labelling individual samples.

    Empty-box samples are dropped by default: their masks are all zero, so every ablation delta
    on them is ~0 and they only dilute the signal.
    """
    import json
    from streaming import StreamingDataset
    from model.dataset import exp25_split

    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    ds = StreamingDataset(local=str(mds_dir), shuffle=False, batch_size=1)
    # MDS dirs straddle the output_id -> position_id rename, so normalize the index here rather
    # than reaching into the shared split code
    index = [json.loads(l) for l in open(Path(mds_dir) / 'metadata.jsonl')]
    for m in index:  # older MDS dirs predate the output_id -> position_id rename; accept either
        if 'position_id' in m: m.setdefault('output_id', m['position_id'])
        m['position_id'] = int(m.get('position_id', m.get('output_id')))
    splits = exp25_split(mds_dir, test_size=test_size, seed=seed, verbose=0, index=index)
    idxs = sorted({i for name, ii in splits.items() if name.startswith(split) for i in ii}) if split == 'eval' else splits[split]

    keep = []
    for i in idxs:
        if exclude_empty and ds[i]['is_empty_box']: continue
        keep.append(i)
        if len(keep) >= n: break
    X = torch.from_numpy(np.stack([ds[i]['X'] for i in keep])).float().to(device)
    y = torch.from_numpy(np.stack([ds[i]['y'] for i in keep])).float().to(device)
    if not return_info: return X, y
    keys = ('sample_id', 'speaker', 'box', 'n_objects', 'is_empty_box', 'object',
            'downsampled_com_x', 'downsampled_com_y')
    pid = lambda s: int(s['position_id'] if 'position_id' in s else s['output_id'])
    info = [{'row': r, 'mds_idx': i, 'position_id': pid(ds[i]),
             **{k: ds[i][k] for k in keys if k in ds[i]}} for r, i in enumerate(keep)]
    return X, y, info


def freqs_hz(mds_dir=OLD_CONFIG_MDS):
    """The FFT bin -> Hz axis for a dataset. Prefer the copy inside the MDS dir (it belongs to that
    dataset); older layouts keep one shared file beside it."""
    for p in (Path(mds_dir) / 'freqs.npy', Path(mds_dir).parent / 'freqs.npy'):
        if p.exists(): return np.load(p)
    raise FileNotFoundError(f"no freqs.npy in {mds_dir} or its parent")


def patch_freq_ranges(mds_dir=OLD_CONFIG_MDS, n_patches=11, patch_size=256):
    """(lo_hz, hi_hz) per freq patch. Stops at the truncation point, not at freqs[-1]."""
    freqs = freqs_hz(mds_dir)
    return [(float(freqs[p * patch_size]), float(freqs[min((p + 1) * patch_size, len(freqs)) - 1])) for p in range(n_patches)]


# ***** attention *****

def _attn_modules(model):
    """The inner nn.MultiheadAttention of each encoder layer, by axis.

    Note the doubled `.layers.layers` on the freq side: FreqEncoder.layers is the
    nn.TransformerEncoder, which itself holds a `.layers` ModuleList.
    """
    return {'laser': [l.self_attn for l in model.laser_encoder.layers],
            'freq':  [l.self_attn for l in model.freq_encoder.layers.layers]}


@contextmanager
def capture_attention(model, cls_row_only=True, n_lasers=None):
    """Collect self-attention weights during the forward passes made inside this block.

    nn.TransformerEncoder never forwards need_weights, so it has to be injected on the inner
    MultiheadAttention via a pre-hook. That also defeats the fused eval-mode fast path, which
    would otherwise never materialize the weights at all.

    Yields a dict filled in on each forward:
      laser -> (B, n_layers, H, 100)          cls attention over lasers      (cls_row_only)
      freq  -> (B, n_lasers, n_layers, H, 11) cls attention over freq patches
    With cls_row_only=False you get the full (.., S, S) maps instead -- the freq one is ~2.4M
    floats per sample, so only do that for a handful of samples.
    """
    out, handles = {}, []

    def pre(mod, args, kwargs):
        kwargs = dict(kwargs)
        kwargs['need_weights'], kwargs['average_attn_weights'] = True, False
        return args, kwargs

    def save(axis, layer):
        def hook(mod, args, kwargs, result):
            w = result[1]
            if w is None: return
            w = w[:, :, 0, 1:] if cls_row_only else w  # cls row, dropping the cls->cls entry
            out.setdefault(axis, {})[layer] = w.detach().float()
        return hook

    for axis, mods in _attn_modules(model).items():
        for i, mha in enumerate(mods):
            handles.append(mha.register_forward_pre_hook(pre, with_kwargs=True))
            handles.append(mha.register_forward_hook(save(axis, i), with_kwargs=True))
    try:
        yield out
    finally:
        for h in handles: h.remove()
        # stack per-layer captures into one tensor per axis, and unfold the laser axis that
        # FreqEncoder folded into the batch (B*L -> B, L)
        for axis in list(out):
            w = torch.stack([out[axis][i] for i in sorted(out[axis])], dim=1)  # (B_,n_layers,...)
            if axis == 'freq' and n_lasers:
                w = w.reshape(-1, n_lasers, *w.shape[1:])
            out[axis] = w


def attention_importance(attn):
    """Collapse captured cls-row attention to one score per laser / per freq patch.

    Averages over batch, layers and heads. The encoders are 2 layers x 2 heads, so there is no
    depth for attention to sharpen over -- check the per-(layer,head) facet before trusting the
    mean to be representative.
    """
    scores = {}
    if 'laser' in attn: scores['laser'] = attn['laser'].mean(dim=(0, 1, 2)).cpu().numpy()          # (B,L,H,100)
    if 'freq' in attn:  scores['freq'] = attn['freq'].mean(dim=(0, 1, 2, 3)).cpu().numpy()          # (B,n_lasers,L,H,11)
    return scores


# ***** ablation *****

MSE_CHUNK = 24  # forward chunk size; a full 96-sample batch needs >4GB and OOMs on a shared GPU


@torch.no_grad()
def _mse(model, X, y, chunk=None):
    """Mean MSE over the batch, evaluated in chunks so a shared GPU doesn't OOM.

    Chunking is exact here (not an approximation) because every sample contributes an equal-sized
    (out_h, out_w) mask, so the mean of the per-chunk sums is the mean over the whole batch.
    """
    chunk = chunk or MSE_CHUNK
    total, n = 0.0, 0
    for s in range(0, X.shape[0], chunk):
        pred = model({'fft': X[s:s + chunk]})['mask_pred']
        total += ((pred - y[s:s + chunk]) ** 2).sum().item()
        n += pred.numel()
    return total / n


@torch.no_grad()
def _bce(model, X, y, chunk=None):
    """Mean per-pixel BCE over the batch, in nats. Same chunking contract as `_mse`.

    Computed from `mask_logits`, not `mask_pred`: this is the numerically stable form
    (`binary_cross_entropy_with_logits` folds the sigmoid into a log-sum-exp), and it matches the
    project's own `ce_pixel_loss` exactly. Going through the post-sigmoid `mask_pred` would blow up
    to inf wherever a confident logit saturates to exactly 0.0 or 1.0 in float32 -- which is common
    on a trained checkpoint, and precisely where an ablation delta needs to stay finite.

    Note BCE is only bounded below by 0 for a hard target; the masks here are soft-valued in [0,1],
    so the floor is the target's own entropy and the absolute number is not comparable to MSE. Only
    differences between it and a baseline are meaningful -- which is all the ablation helpers use.
    """
    import torch.nn.functional as F
    chunk = chunk or MSE_CHUNK
    total, n = 0.0, 0
    for s in range(0, X.shape[0], chunk):
        logits = model({'fft': X[s:s + chunk]})['mask_logits']
        yc = y[s:s + chunk].reshape(logits.shape)
        total += F.binary_cross_entropy_with_logits(logits, yc, reduction='sum').item()
        n += logits.numel()
    return total / n


@torch.no_grad()
def ablate_lasers(model, X, y, baseline=None, metric=None):
    """Delta true-BCE from zeroing each laser in turn. Returns (100,).

    `metric` defaults to `_bce`; pass `_mse` for the old squared-error view.

    Zeroing (rather than noising or resampling) is the in-distribution choice here: training
    already zeroes dropped laser/freq tokens, so the model has seen this exact input.

    Leave-one-out understates importance when lasers are redundant -- two neighbours carrying
    the same signal each look useless alone. Pair with `retain_top_k` to catch that.
    """
    metric = metric or _bce
    baseline = metric(model, X, y) if baseline is None else baseline
    out = np.zeros(X.shape[1])
    for i in range(X.shape[1]):
        Xa = X.clone(); Xa[:, i] = 0
        out[i] = metric(model, Xa, y) - baseline
    return out


@torch.no_grad()
def ablate_freq_patches(model, X, y, baseline=None, metric=None):
    """Delta true-BCE from zeroing each frequency patch in turn. Returns (n_patches,).

    `metric` defaults to `_bce`; pass `_mse` for the old squared-error view."""
    metric = metric or _bce
    baseline = metric(model, X, y) if baseline is None else baseline
    out = np.zeros(X.shape[2])
    for p in range(X.shape[2]):
        Xa = X.clone(); Xa[:, :, p] = 0
        out[p] = metric(model, Xa, y) - baseline
    return out


@torch.no_grad()
def ablate_laser_freq(model, X, y, lasers=None, patches=None, baseline=None, metric=None):
    """Delta true-BCE from zeroing a single (laser, freq patch) cell. Returns (len(lasers), len(patches)).

    The joint view attention cannot give: the model scores the two axes in separate encoders.
    `metric` defaults to `_bce`; pass `_mse` for the old squared-error view.
    """
    metric = metric or _bce
    baseline = metric(model, X, y) if baseline is None else baseline
    lasers = range(X.shape[1]) if lasers is None else lasers
    patches = range(X.shape[2]) if patches is None else patches
    out = np.zeros((len(lasers), len(patches)))
    for a, i in enumerate(lasers):
        for b, p in enumerate(patches):
            Xa = X.clone(); Xa[:, i, p] = 0
            out[a, b] = metric(model, Xa, y) - baseline
    return out


@torch.no_grad()
def retain_top_k(model, X, y, order, axis='laser', ks=None, metric=None):
    """BCE when only the top-k inputs (by `order`) are kept and the rest are zeroed.

    `metric` defaults to `_bce`; pass `_mse` for the old squared-error view.

    Stronger evidence than leave-one-out: it reports how much of the signal a subset actually
    carries, so redundancy between correlated inputs shows up instead of hiding.
    """
    metric = metric or _bce
    n = X.shape[1] if axis == 'laser' else X.shape[2]
    ks = ks if ks is not None else range(1, n + 1)
    out = {}
    for k in ks:
        Xa = torch.zeros_like(X)
        keep = order[:k]
        if axis == 'laser': Xa[:, keep] = X[:, keep]
        else:               Xa[:, :, keep] = X[:, :, keep]
        out[k] = metric(model, Xa, y)
    return out


def grad_x_input_lasers(model, X, y, chunk=8):
    """|grad * activation| per laser token, summed over d_model. Returns (100,).

    Included only as a third point of comparison: it lands ~0.56 against ablation, essentially
    the same as attention, because both are dominated by token magnitude.

    Chunked over the batch -- the backward graph spans B*100 freq-encoder sequences, which OOMs
    at full batch on a normal GPU.
    """
    from model.arch import apply_rope
    totals = []
    for s in range(0, X.shape[0], chunk):
        Xc, yc = X[s:s + chunk], y[s:s + chunk]
        B, L = Xc.shape[0], Xc.shape[1]
        emb = model.freq_encoder(Xc.flatten(0, 1), None).reshape(B, L, -1)
        emb.retain_grad()
        z = apply_rope(emb, model.freqs_laser)
        z = torch.cat((model.cls_token.expand(B, -1, -1), z), dim=1)
        pred = model.decoder(model.laser_encoder(z)[:, 0, :]).sigmoid()
        model.zero_grad(set_to_none=True)
        ((pred - yc) ** 2).mean().backward()
        totals.append((emb.grad * emb).abs().sum(-1).detach().cpu())
    return torch.cat(totals).mean(0).numpy()
