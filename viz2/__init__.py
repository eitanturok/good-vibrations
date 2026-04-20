"""
viz2/__init__.py — Build a self-contained HTML viewer for dataset samples and W&B runs.

Usage:
    python viz2/__init__.py --output artifacts/viewer2.html
    python viz2/__init__.py --runs s3pqt79j abc123 --output artifacts/viewer2.html
"""

import argparse
import base64
import io
import json
import resource
import sys
import time
import webbrowser
from pathlib import Path

import numpy as np
import scipy.signal
import scipy.io.wavfile
import wandb
from datasets import load_dataset
from PIL import Image

# Add src/ to path so we can reuse existing helpers
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
from helpers import fetch_predictions, fetch_overhead_images, fetch_wandb_history

ENTITY = 'eturok'
PROJECT = 'good-vibrations'
HF_DATASET = 'eturok-weizmann/laser-vibrations'
HF_PREDS   = 'eturok-weizmann/laser-vibrations'
PROFILE_LOG_EVERY = 100
ASSETS_DIR = Path(__file__).parent.parent / 'assets'
SPEAKER_SOURCE = ASSETS_DIR / 'speaker.png'
SPEAKER_DIR = ASSETS_DIR / 'speakers'
OVERHEAD_SIZE = (220, 196)
PADDED_SIZE = (286, 255)
PADDED_BG = (232, 232, 232)
SPEAKER_FILES = ('1000', '0100', '0010', '0001')


# ── Data fetching ─────────────────────────────────────────────────

def get_last_n_runs(n=3):
    api = wandb.Api()
    runs = list(api.runs(f'{ENTITY}/{PROJECT}', order='-created_at'))[:n]
    return [r.id for r in runs]


def load_hf_samples():
    """Return list of metadata dicts for all dataset samples."""
    ds = load_dataset(
        HF_DATASET,
        columns=['sample_idx', 'object', 'n_objects', 'x_position', 'y_position',
                 'speakers', 'fps', 'experiment_config'],
        split='train',
        verification_mode='no_checks',
    )
    return [dict(row) for row in ds]


# ── Encoding helpers ──────────────────────────────────────────────

def encode_image_b64(img: Image.Image, fmt='JPEG', quality=82) -> str:
    buf = io.BytesIO()
    kw = {'quality': quality} if fmt == 'JPEG' else {}
    img.save(buf, format=fmt, **kw)
    enc = base64.b64encode(buf.getvalue()).decode()
    mime = 'image/jpeg' if fmt == 'JPEG' else 'image/png'
    return f'data:{mime};base64,{enc}'


def encode_audio_b64(path: Path) -> str | None:
    if not path.exists():
        return None
    enc = base64.b64encode(path.read_bytes()).decode()
    return f'data:audio/wav;base64,{enc}'


def mask_to_b64(mask_2d: np.ndarray) -> str:
    """Float mask (H, W) in [0,1] → base64 uint8 flat bytes."""
    u8 = np.clip(mask_2d * 255, 0, 255).astype(np.uint8)
    return base64.b64encode(np.ascontiguousarray(u8).tobytes()).decode()


# ── Per-sample assets ─────────────────────────────────────────────

def find_chirp_wav(experiment_config_json: str | None) -> str | None:
    try:
        cfg = json.loads(experiment_config_json or '{}')
    except Exception:
        return None
    for key in ('audio_file', 'chirp_file', 'wav', 'audio'):
        val = cfg.get(key, '') or ''
        if val:
            result = encode_audio_b64(Path('data') / val)
            if result:
                return result
    return None


def sonify_shifts(shifts: np.ndarray, fps: float = 2900,
                  out_sr: int = 22050, duration: float = 2.0) -> str | None:
    """Convert vibration timeseries to a WAV data URL (average magnitude over lasers)."""
    try:
        signal = np.sqrt(shifts[:, :, 0]**2 + shifts[:, :, 1]**2).mean(axis=0)
        n_in  = min(len(signal), int(duration * fps))
        n_out = int(duration * out_sr)
        audio = scipy.signal.resample(signal[:n_in], n_out)
        audio = audio / (np.abs(audio).max() + 1e-8) * 0.8
        buf = io.BytesIO()
        scipy.io.wavfile.write(buf, out_sr, (audio * 32767).astype(np.int16))
        enc = base64.b64encode(buf.getvalue()).decode()
        return f'data:audio/wav;base64,{enc}'
    except Exception as e:
        print(f'  [warn] sonify_shifts: {e}')
        return None


def shift_heatmap_b64(shifts: np.ndarray) -> str:
    """(100, T, 2) → 10×10 grayscale magnitude heatmap as PNG data URL."""
    mag  = np.sqrt(shifts[:, :, 0]**2 + shifts[:, :, 1]**2).mean(axis=1)  # (100,)
    grid = mag.reshape(10, 10)
    grid = ((grid - grid.min()) / (grid.max() - grid.min() + 1e-8) * 255).astype(np.uint8)
    img  = Image.fromarray(grid).resize((80, 80), Image.NEAREST)
    return encode_image_b64(img, fmt='PNG')


def speaker_mask(speakers) -> str:
    values = list(speakers or [])[:4]
    values.extend([0] * (4 - len(values)))
    return ''.join('1' if int(v) else '0' for v in values)


def pad_overhead_image(img: Image.Image) -> Image.Image:
    canvas = Image.new('RGB', PADDED_SIZE, PADDED_BG)
    inner = img.resize(OVERHEAD_SIZE, Image.BILINEAR)
    x = (PADDED_SIZE[0] - OVERHEAD_SIZE[0]) // 2
    y = (PADDED_SIZE[1] - OVERHEAD_SIZE[1]) // 2
    canvas.paste(inner, (x, y))
    return canvas


def ensure_speaker_assets() -> dict[str, Image.Image]:
    SPEAKER_DIR.mkdir(parents=True, exist_ok=True)
    targets = {name: SPEAKER_DIR / f'{name}.png' for name in SPEAKER_FILES}
    if not all(path.exists() for path in targets.values()):
        src = Image.open(SPEAKER_SOURCE).convert('RGBA')
        vertical = src.resize((44, 68), Image.LANCZOS)
        placements = {
            '1000': (vertical, (8, 91)),
            '0100': (vertical, (44, 181)),
            '0010': (vertical, (198, 181)),
            '0001': (vertical, (234, 91)),
        }
        for name, path in targets.items():
            overlay = Image.new('RGBA', PADDED_SIZE, (0, 0, 0, 0))
            icon, pos = placements[name]
            overlay.alpha_composite(icon, dest=pos)
            overlay.save(path)
    return {name: Image.open(path).convert('RGBA') for name, path in targets.items()}


def build_speaker_overlay(mask: str, assets: dict[str, Image.Image]) -> str | None:
    if '1' not in mask:
        return None
    overlay = Image.new('RGBA', PADDED_SIZE, (0, 0, 0, 0))
    for bit, key in zip(mask, SPEAKER_FILES):
        if bit == '1':
            overlay.alpha_composite(assets[key])
    return encode_image_b64(overlay, fmt='PNG')


# ── Run data ──────────────────────────────────────────────────────

def _t(label: str, t0: float) -> float:
    elapsed = time.perf_counter() - t0
    print(f'  ✓ {label}: {elapsed:.1f}s  rss={_rss_mb():.1f} MB')
    return time.perf_counter()


def _rss_mb() -> float:
    scale = 1 if sys.platform == 'darwin' else 1024
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * scale)


def _fmt_bytes(n: int | float) -> str:
    n = float(n)
    for unit in ('B', 'KB', 'MB', 'GB', 'TB'):
        if n < 1024 or unit == 'TB':
            return f'{n:.1f} {unit}'
        n /= 1024
    return f'{n:.1f} TB'


def load_run_data(run_id: str) -> dict:
    """Fetch W&B config + history + HF predictions. Returns serialisable dict."""
    t0 = time.perf_counter()
    print(f'  [run {run_id}] W&B metadata...')
    run_name = run_id
    run_config = {}
    try:
        api = wandb.Api()
        run = api.run(f'{ENTITY}/{PROJECT}/{run_id}')
        cfg = run.config
        run_name = run.name or run_id
        run_config = {k: cfg.get(k) for k in ('loss', 'gamma', 'decoder', 'd_model', 'lr', 'seed', 'n_params')}
    except Exception as e:
        print(f'  [warn] W&B metadata unavailable for {run_id}: {e}')
    t0 = _t('W&B metadata', t0)

    print(f'  [run {run_id}] W&B history...')
    keys = ['_step', 'epoch', 'loss/train/total',
            'metrics/train/mask/iou', 'metrics/eval/mask/iou',
            'metrics/train/mask/mse', 'metrics/eval/mask/mse']
    metrics_by_epoch: dict[str, dict] = {}
    try:
        for row in fetch_wandb_history(run_id, keys=keys):
            epoch = row.get('epoch')
            if epoch is None:
                continue
            e = str(int(epoch))
            m = metrics_by_epoch.setdefault(e, {})
            for src, dst in [('loss/train/total', 'loss'),
                             ('metrics/train/mask/iou', 'train_iou'),
                             ('metrics/eval/mask/iou',  'eval_iou'),
                             ('metrics/train/mask/mse', 'train_mse'),
                             ('metrics/eval/mask/mse',  'eval_mse')]:
                if row.get(src) is not None:
                    m[dst] = row[src]
    except Exception as e:
        print(f'  [warn] W&B history unavailable for {run_id}: {e}')
    t0 = _t('W&B history', t0)

    print(f'  [run {run_id}] predictions from HF Hub...')
    raw = None
    pred_keys = []
    for pred_key in [run_name, run_id]:
        if pred_key in pred_keys:
            continue
        pred_keys.append(pred_key)
        try:
            print(f'    [lookup] predictions key={pred_key}')
            candidate = fetch_predictions(pred_key, data_dir=HF_PREDS)
            n_epochs = len(set(candidate['train'].keys()) | set(candidate['eval'].keys()))
            if n_epochs > 0:
                raw = candidate
                print(f'    [lookup] found predictions under key={pred_key} ({n_epochs} epochs)')
                break
        except Exception as e:
            print(f'    [lookup] failed for key={pred_key}: {e}')
    if raw is None:
        print(f'  [warn] no predictions found for any key in {pred_keys}')
        raw = {'train': {}, 'eval': {}}
    t0 = _t('HF predictions download', t0)

    print(f'  [run {run_id}] encoding masks...')
    epochs_set = sorted(set(raw['train'].keys()) | set(raw['eval'].keys()))
    preds_by_epoch: dict[str, dict] = {}
    gt_masks: dict[str, str] = {}  # sample_idx str → b64
    n_entries = 0
    n_gt_encoded = 0
    mask_bytes_raw = 0
    mask_bytes_b64 = 0
    t_encode = time.perf_counter()
    t_metrics = 0.0
    t_pred_b64 = 0.0
    t_gt_b64 = 0.0

    for epoch_i, epoch in enumerate(epochs_set, start=1):
        ep_str = str(epoch)
        preds_by_epoch[ep_str] = {}
        for split in ('train', 'eval'):
            if epoch not in raw[split]:
                continue
            npz = raw[split][epoch]
            n   = min(len(npz['mask_true']), len(npz['mask_pred']), len(npz['sample_idx']))
            for i in range(n):
                idx  = str(int(npz['sample_idx'][i]))
                pred = npz['mask_pred'][i]
                gt   = npz['mask_true'][i]
                t1 = time.perf_counter()
                mse  = float(np.mean((gt - pred) ** 2))
                pred_bin = pred > 0.5
                gt_bin   = gt > 0.5
                union = np.sum(pred_bin | gt_bin)
                iou   = float(np.sum(pred_bin & gt_bin) / union) if union > 0 else 0.0
                t_metrics += time.perf_counter() - t1
                t1 = time.perf_counter()
                pred_b64 = mask_to_b64(pred)
                t_pred_b64 += time.perf_counter() - t1
                preds_by_epoch[ep_str][idx] = {
                    'pred':  pred_b64,
                    'split': split,
                    'mse':   round(mse, 4),
                    'iou':   round(iou, 3),
                }
                n_entries += 1
                mask_bytes_raw += int(np.asarray(pred).nbytes + np.asarray(gt).nbytes)
                mask_bytes_b64 += len(pred_b64)
                if idx not in gt_masks:
                    t1 = time.perf_counter()
                    gt_b64 = mask_to_b64(gt)
                    t_gt_b64 += time.perf_counter() - t1
                    gt_masks[idx] = gt_b64
                    mask_bytes_b64 += len(gt_b64)
                    n_gt_encoded += 1
        if epoch_i % PROFILE_LOG_EVERY == 0 or epoch_i == len(epochs_set):
            elapsed = time.perf_counter() - t_encode
            print(
                f'    [profile] epochs {epoch_i}/{len(epochs_set)}  '
                f'entries={n_entries}  gt_masks={n_gt_encoded}  '
                f'elapsed={elapsed:.1f}s  metrics={t_metrics:.1f}s  '
                f'pred_b64={t_pred_b64:.1f}s  gt_b64={t_gt_b64:.1f}s  '
                f'raw_masks={_fmt_bytes(mask_bytes_raw)}  b64_masks={_fmt_bytes(mask_bytes_b64)}'
            )
    _t(f'mask encoding ({len(epochs_set)} epochs)', t0)
    print(
        f'  [run {run_id}] profile summary: '
        f'epochs={len(epochs_set)} entries={n_entries} unique_gt_masks={n_gt_encoded} '
        f'raw_mask_bytes={_fmt_bytes(mask_bytes_raw)} b64_bytes={_fmt_bytes(mask_bytes_b64)} '
        f'metrics_time={t_metrics:.1f}s pred_b64_time={t_pred_b64:.1f}s gt_b64_time={t_gt_b64:.1f}s'
    )

    return {
        'id':      run_id,
        'name':    run_name,
        'config':  run_config,
        'epochs':  [str(e) for e in epochs_set],
        'metrics': metrics_by_epoch,
        'preds':   preds_by_epoch,
        '_gt_masks': gt_masks,  # merged into samples later
    }


# ── Payload assembly ──────────────────────────────────────────────

def build_payload(run_ids: list[str], show_speakers: bool = True) -> dict:
    t_total = time.perf_counter()
    t0 = t_total

    print(f'[1/5] HF dataset metadata ({HF_DATASET})...')
    raw_samples   = load_hf_samples()
    sample_by_idx = {int(s['sample_idx']): s for s in raw_samples}
    all_idxs      = sorted(sample_by_idx.keys())
    t0 = _t(f'HF metadata ({len(all_idxs)} samples)', t0)

    print(f'[2/5] Overhead images ({len(all_idxs)} samples)...')
    overhead_imgs = fetch_overhead_images(all_idxs, repo_id=HF_DATASET)
    t0 = _t('overhead images', t0)

    speaker_assets = ensure_speaker_assets() if show_speakers else None

    # Load all runs; collect GT masks from predictions
    runs: list[dict] = []
    all_gt_masks: dict[str, str] = {}
    for i, run_id in enumerate(run_ids):
        print(f'[3/5] Run {i+1}/{len(run_ids)}: {run_id}')
        t_run = time.perf_counter()
        run_data = load_run_data(run_id)
        new_gt = run_data.pop('_gt_masks', {})
        print(f'  [debug] gt masks from this run: {len(new_gt)}  '
              f'(sample idx examples: {list(new_gt.keys())[:5]})')
        all_gt_masks.update(new_gt)
        print(f'  [debug] epochs in run: {run_data["epochs"]}')
        print(f'  [debug] total pred entries (last epoch): '
              f'{len(run_data["preds"].get(run_data["epochs"][-1], {})) if run_data["epochs"] else 0}')
        runs.append(run_data)
        _t(f'run {run_id} total', t_run)
    print(f'  [debug] total gt masks collected across all runs: {len(all_gt_masks)}')
    print(f'  [debug] dataset idx examples: {all_idxs[:5]}')
    print(f'  [debug] gt mask idx examples: {list(all_gt_masks.keys())[:5]}')
    t0 = time.perf_counter()

    # Build sample records
    print(f'[4/5] Assembling {len(all_idxs)} sample records...')
    samples: list[dict] = []
    t_overhead_resize_encode = 0.0
    t_audio_lookup = 0.0
    overhead_bytes_b64 = 0
    speaker_overlay_bytes_b64 = 0
    for idx in all_idxs:
        s       = sample_by_idx[idx]
        overhead = overhead_imgs.get(idx)
        if overhead is None:
            overhead = Image.fromarray(np.full((196, 220, 3), 200, np.uint8))
        overhead_src_w, overhead_src_h = overhead.size
        speakers = s.get('speakers') or []
        speaker_key = speaker_mask(speakers)
        t1 = time.perf_counter()
        rendered_overhead = pad_overhead_image(overhead) if show_speakers else overhead.resize(OVERHEAD_SIZE, Image.BILINEAR)
        overhead_b64 = encode_image_b64(rendered_overhead)
        t_overhead_resize_encode += time.perf_counter() - t1
        overhead_bytes_b64 += len(overhead_b64)
        speaker_overlay = build_speaker_overlay(speaker_key, speaker_assets) if show_speakers and speaker_assets else None
        speaker_overlay_bytes_b64 += len(speaker_overlay or '')
        t1 = time.perf_counter()
        chirp_audio = find_chirp_wav(s.get('experiment_config'))
        t_audio_lookup += time.perf_counter() - t1

        samples.append({
            'idx':      idx,
            'object':   s.get('object', ''),
            'n_objects': int(s.get('n_objects') or 1),
            'x':        float(s.get('x_position') or 0),
            'y':        float(s.get('y_position') or 0),
            'speakers': speakers,
            'overhead': overhead_b64,
            'speaker_overlay': speaker_overlay,
            'overhead_src_w': overhead_src_w,
            'overhead_src_h': overhead_src_h,
            'gt_mask':  all_gt_masks.get(str(idx)),
            'mask_w':   20,
            'mask_h':   40,
            'chirp_audio':    chirp_audio,
            'sonified_audio': None,  # populated only if --sonify flag added later
            'shift_heatmap':  None,  # populated only if --heatmaps flag added later
        })

    n_with_gt = sum(1 for s in samples if s['gt_mask'] is not None)
    print(f'  [debug] samples with gt_mask: {n_with_gt}/{len(samples)}')
    if samples:
        s0 = samples[0]
        print('  [debug] viewer dimensions: '
              f"overhead_src={s0['overhead_src_w']}x{s0['overhead_src_h']} "
              f"overhead_canvas=220x196 run_mask={s0['mask_w']}x{s0['mask_h']}")
    print(
        f'  [profile] sample assembly summary: overhead_b64={_fmt_bytes(overhead_bytes_b64)} '
        f'speaker_overlay_b64={_fmt_bytes(speaker_overlay_bytes_b64)} '
        f'resize+encode={t_overhead_resize_encode:.1f}s audio_lookup={t_audio_lookup:.1f}s'
    )

    t0 = _t('sample assembly', t0)

    print(f'[5/5] Serialising payload to JSON...')
    result = {'samples': samples, 'runs': runs}
    _t('JSON serialisation', t0)
    _t('total build_payload', t_total)
    return result


# ── HTML rendering ────────────────────────────────────────────────

def render_html(payload: dict, title: str = 'Good Vibrations Viewer') -> str:
    t0 = time.perf_counter()
    viz_dir = Path(__file__).parent
    js = (viz_dir / 'viewer.js').read_text()
    css = (viz_dir / 'viewer.css').read_text()
    t_assets = time.perf_counter() - t0
    t1 = time.perf_counter()
    payload_json = json.dumps(payload, separators=(',', ':')).replace('</script>', '<\\/script>')
    t_json = time.perf_counter() - t1
    print(
        f'  [profile] render_html assets={t_assets:.1f}s '
        f'payload_json={t_json:.1f}s payload_size={_fmt_bytes(len(payload_json))}'
    )
    return f'''<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>{title}</title>
  <style>{css}</style>
</head>
<body>
  <h1 id="page-title">What&#39;s in the box?</h1>
  <div id="top-bar">
    <div id="bar-row1">
      <div id="filters">
        <label>Split <select id="filter-split"><option value="">All</option><option>train</option><option>eval</option></select></label>
        <label>Speaker <select id="filter-speaker"></select></label>
        <label>Object <select id="filter-object"></select></label>
        <label>N objects <select id="filter-nobjects"></select></label>
      </div>
      <div id="add-run-bar">
        <select id="run-selector"></select>
        <button id="add-run-btn">+ Add Column</button>
      </div>
    </div>
    <div id="bar-row2">
      <button id="play-btn">&#9654; Play</button>
      <label id="speed-label">Speed: <span id="speed-fps">1.2 fps</span>
        <input id="play-speed" type="range" min="50" max="2000" step="50" value="800" list="speed-ticks">
        <datalist id="speed-ticks">
          <option value="50"  label="20fps"></option>
          <option value="100" label="10fps"></option>
          <option value="200" label="5fps"></option>
          <option value="500" label="2fps"></option>
          <option value="1000" label="1fps"></option>
          <option value="2000" label="0.5fps"></option>
        </datalist>
      </label>
      <label id="epoch-label-wrap">Epoch <input id="epoch-slider" type="range" min="0" max="0" step="1" value="0"></label>
      <span id="play-epoch-label"></span>
    </div>
  </div>
  <div id="table-wrap">
    <table id="grid">
      <thead id="thead"></thead>
      <tbody id="tbody"></tbody>
    </table>
  </div>
  <div id="tooltip"></div>
  <dialog id="fullscreen-dialog">
    <button id="close-dialog">✕ Close</button>
    <div id="fullscreen-content"></div>
  </dialog>
  <script id="payload" type="application/json">{payload_json}</script>
  <script>{js}</script>
</body>
</html>'''


# ── CLI ───────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Build dataset + run viewer HTML')
    parser.add_argument('--runs', nargs='*', metavar='RUN_ID',
                        help='W&B run IDs (default: last 3 runs)')
    parser.add_argument('--output', nargs='?', default='artifacts/viewer2.html',
                        help='Output HTML path')
    parser.add_argument('--no-show-speakers', action='store_true',
                        help='Disable padded overhead speaker overlays')
    parser.add_argument('--no-open', action='store_true',
                        help='Do not open the generated HTML in a browser')
    args = parser.parse_args()

    t0 = time.perf_counter()

    run_ids = args.runs or get_last_n_runs(3)
    print(f'Viewer dataset: {HF_DATASET}')
    print(f'Runs: {run_ids}\n')

    payload = build_payload(run_ids, show_speakers=not args.no_show_speakers)

    print('Rendering HTML...')
    html = render_html(payload)
    _t('render_html', t0)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding='utf-8')
    uri = out.resolve().as_uri()
    opened = False if args.no_open else webbrowser.open(uri)
    size_mb = out.stat().st_size / 1e6
    total = time.perf_counter() - t0
    print(f'\nWritten: {out}  ({size_mb:.1f} MB)  total: {total:.1f}s')
    print(f'URL: {uri}')
    print(f'Opened browser: {opened}')


if __name__ == '__main__':
    main()
