import argparse
import base64
import html
import io
import json
import time
import webbrowser
from pathlib import Path

import numpy as np
import wandb
from PIL import Image

from helpers import fetch_overhead_images, fetch_predictions, fetch_wandb_history


def _encode_image_data_url(image: Image.Image, fmt: str = "JPEG", **save_kwargs) -> str:
    buf = io.BytesIO()
    image.save(buf, format=fmt, **save_kwargs)
    encoded = base64.b64encode(buf.getvalue()).decode("ascii")
    mime = "image/jpeg" if fmt.upper() == "JPEG" else "image/png"
    return f"data:{mime};base64,{encoded}"


def _transform_mask(mask, target_shape, rotate_k=0, flip_lr=False, flip_ud=False):
    arr = np.rot90(mask, k=rotate_k)
    if flip_lr:
        arr = np.fliplr(arr)
    if flip_ud:
        arr = np.flipud(arr)
    img = Image.fromarray(np.clip(arr * 255, 0, 255).astype(np.uint8))
    img = img.resize(
        (target_shape[1], target_shape[0]), resample=Image.Resampling.NEAREST
    )
    return np.asarray(img).astype(np.uint8)


def _mask_to_b64(mask_u8: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(mask_u8).tobytes()).decode("ascii")


def _sample_title(split: str, sample_idx: int, meta: dict) -> str:
    tag = "TR" if split == "train" else "EV"
    return (
        f"{tag}#{sample_idx} {meta['object']} n={meta['n_objects']}"
        f"<br>x={meta['x_position']:.1f} y={meta['y_position']:.1f}"
    )


def build_html_viewer_payload(
    run_id,
    wandb_run_id,
    data_dir="eturok-weizmann/vibrations",
    entity="eturok",
    project="good-vibrations",
    rows=2,
    cols=4,
    target_size=(220, 196),
    mask_rotate_k=0,
    mask_flip_lr=False,
    mask_flip_ud=False,
    max_samples=None,
):
    t0 = time.perf_counter()
    predictions = fetch_predictions(run_id, data_dir=data_dir)
    epochs = sorted(set(predictions["train"].keys()) | set(predictions["eval"].keys()))
    if not epochs:
        raise ValueError(f"No predictions found for run {run_id}")

    if max_samples is None:
        max_samples = rows * cols

    samples_by_split = {}
    sample_ids = []
    for split in ("train", "eval"):
        split_epochs = sorted(predictions[split].keys())
        if not split_epochs:
            samples_by_split[split] = []
            continue
        latest = predictions[split][split_epochs[-1]]
        n_latest = min(
            len(latest["mask_true"]),
            len(latest["mask_pred"]),
            len(latest["sample_idx"]),
            len(latest["x_position"]),
            len(latest["y_position"]),
            len(latest["object_type"]),
            len(latest["n_objects"]),
        )
        split_samples = []
        for i, sid in enumerate(latest["sample_idx"][:n_latest].tolist()[:max_samples]):
            sid = int(sid)
            meta = {
                "x_position": float(latest["x_position"][:n_latest][i]),
                "y_position": float(latest["y_position"][:n_latest][i]),
                "object": str(latest["object_type"][:n_latest][i]),
                "n_objects": int(latest["n_objects"][:n_latest][i]),
            }
            split_samples.append(
                {
                    "split": split,
                    "sample_idx": sid,
                    "title": _sample_title(split, sid, meta),
                    "meta": meta,
                }
            )
            sample_ids.append(sid)
        samples_by_split[split] = split_samples

    sample_ids = sorted(set(sample_ids))
    overhead_images = fetch_overhead_images(sample_ids, repo_id=data_dir)
    overhead_images = {
        sid: img.resize(target_size, resample=Image.Resampling.BILINEAR)
        for sid, img in overhead_images.items()
    }

    history_keys = [
        "_step",
        "epoch",
        "loss/train/total",
        "metrics/train/mask/iou",
        "metrics/eval/mask/iou",
        "metrics/train/mask/mse",
        "metrics/eval/mask/mse",
        "metrics/train/mse",
        "metrics/eval/mse",
    ]
    history_rows = fetch_wandb_history(
        wandb_run_id, keys=history_keys, entity=entity, project=project
    )
    metrics_by_epoch = {}
    for row in history_rows:
        epoch = row.get("epoch")
        if epoch is None:
            continue
        epoch = int(epoch)
        metrics = metrics_by_epoch.setdefault(epoch, {})
        for key in history_keys[2:]:
            if row.get(key) is not None:
                metrics[key] = row[key]

    run = wandb.Api().run(f"{entity}/{project}/{wandb_run_id}")
    cfg = run.config
    run_config = {
        "loss": cfg.get("loss"),
        "gamma": cfg.get("gamma"),
        "decoder": cfg.get("decoder"),
        "d_model": cfg.get("d_model"),
        "batch_size": cfg.get("batch_size"),
        "lr": cfg.get("lr"),
        "seed": cfg.get("seed"),
        "n_params": cfg.get("n_params"),
    }

    epoch_data = {}
    samples = []
    for split in ("train", "eval"):
        for sample in samples_by_split[split]:
            overhead = overhead_images.get(sample["sample_idx"])
            if overhead is None:
                overhead = Image.fromarray(
                    np.full((target_size[1], target_size[0], 3), 220, dtype=np.uint8)
                )
            sample["overhead"] = _encode_image_data_url(
                overhead, fmt="JPEG", quality=82
            )
            samples.append(sample)

    sample_ids_by_split = {
        split: [s["sample_idx"] for s in split_samples]
        for split, split_samples in samples_by_split.items()
    }

    for epoch in epochs:
        epoch_data[str(epoch)] = {"train": {}, "eval": {}}
        for split in ("train", "eval"):
            if epoch not in predictions[split]:
                continue
            npz = predictions[split][epoch]
            n = min(
                len(npz["mask_true"]),
                len(npz["mask_pred"]),
                len(npz["sample_idx"]),
            )
            keep_ids = set(sample_ids_by_split[split])
            rows_by_id = {
                int(npz["sample_idx"][:n][i]): (
                    npz["mask_true"][:n][i],
                    npz["mask_pred"][:n][i],
                )
                for i in range(n)
                if int(npz["sample_idx"][:n][i]) in keep_ids
            }
            for sid in sample_ids_by_split[split]:
                if sid not in rows_by_id:
                    continue
                mt, mp = rows_by_id[sid]
                mt_u8 = _transform_mask(
                    mt,
                    target_shape=(target_size[1], target_size[0]),
                    rotate_k=mask_rotate_k,
                    flip_lr=mask_flip_lr,
                    flip_ud=mask_flip_ud,
                )
                mp_u8 = _transform_mask(
                    mp,
                    target_shape=(target_size[1], target_size[0]),
                    rotate_k=mask_rotate_k,
                    flip_lr=mask_flip_lr,
                    flip_ud=mask_flip_ud,
                )
                epoch_data[str(epoch)][split][str(sid)] = {
                    "gt": _mask_to_b64(mt_u8),
                    "pred": _mask_to_b64(mp_u8),
                }

    build_s = time.perf_counter() - t0
    return {
        "runId": run_id,
        "wandbRunId": wandb_run_id,
        "epochs": epochs,
        "rows": rows,
        "cols": cols,
        "panelWidth": target_size[0],
        "panelHeight": target_size[1],
        "samples": samples,
        "samplesBySplit": samples_by_split,
        "epochData": epoch_data,
        "metricsByEpoch": metrics_by_epoch,
        "runConfig": run_config,
        "buildTiming": {"payload_s": build_s},
    }


def render_viewer_document(payload, title="HTML Viewer"):
    viewer_payload = json.dumps(payload, separators=(",", ":")).replace(
        "</script>", "<\\/script>"
    )
    title = html.escape(title)
    return f"""<!doctype html>
<html>
<head>
  <meta charset=\"utf-8\">
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">
  <title>{title}</title>
  <style>
    :root {{ color-scheme: light; }}
    body {{ margin: 0; font-family: ui-monospace, SFMono-Regular, Menlo, monospace; background: #fff; color: #222; }}
    .app {{ padding: 12px; }}
    .header {{ margin-bottom: 10px; }}
    .header .run {{ font-size: 14px; font-weight: 700; }}
    .header .metrics {{ font-size: 12px; color: #555; margin-top: 4px; }}
    .timing {{ font-size: 12px; color: #555; margin: 8px 0; }}
    .controls {{ display: flex; gap: 10px; align-items: center; flex-wrap: wrap; margin-bottom: 10px; }}
    .controls button, .controls input {{ font: inherit; }}
    .modes, .subsets {{ display: inline-flex; gap: 6px; }}
    .modes button, .subsets button, .play button {{ border: 1px solid #ccc; background: #f8f8f8; padding: 4px 8px; border-radius: 6px; cursor: pointer; }}
    .modes button.active, .subsets button.active, .play button.active {{ background: #222; color: #fff; border-color: #222; }}
    .grid {{ display: grid; gap: 12px; }}
    .panel {{ display: flex; flex-direction: column; gap: 4px; }}
    .panel-title {{ font-size: 12px; color: #444; line-height: 1.2; min-height: 30px; }}
    .panel canvas {{ width: 100%; height: auto; display: block; background: #f1f1f1; border-radius: 4px; }}
    .tooltip {{ position: fixed; pointer-events: none; background: rgba(20,20,20,0.92); color: #fff; padding: 4px 6px; border-radius: 4px; font-size: 11px; display: none; z-index: 9999; white-space: pre; }}
    .panel-sep {{ border-left: 4px dotted #9a9a9a; padding-left: 14px; }}
  </style>
</head>
<body>
  <div class=\"app\">
    <div class=\"header\">
      <div class=\"run\" id=\"run-title\"></div>
      <div class=\"metrics\" id=\"metrics-line\"></div>
    </div>
    <div class=\"timing\" id=\"timing\">waiting for first render…</div>
    <div class=\"controls\">
      <div class=\"play\"><button id=\"play-btn\">Play</button></div>
      <label>Epoch <input type=\"range\" id=\"epoch-slider\"></label>
      <span id=\"epoch-label\"></span>
      <div class=\"subsets\" id=\"subset-buttons\"></div>
      <div class=\"modes\" id=\"mode-buttons\"></div>
    </div>
    <div class=\"grid\" id=\"grid\"></div>
  </div>
  <div class=\"tooltip\" id=\"tooltip\"></div>
  <script id=\"viewer-payload\" type=\"application/json\">{viewer_payload}</script>
  <script>
    const payload = JSON.parse(document.getElementById('viewer-payload').textContent);
    const modes = ['overlay:pred', 'overlay:gt', 'overlay:both', 'masks', 'diff'];
    const subsets = ['both', 'eval', 'train'];
    const state = {{ epoch: payload.epochs[payload.epochs.length - 1], mode: 'overlay:pred', subset: 'both', playing: false }};
    const timingHistory = [];
    const overheadCache = new Map();
    const maskCache = new Map();
    const grid = document.getElementById('grid');
    const tooltip = document.getElementById('tooltip');
    const slider = document.getElementById('epoch-slider');
    const epochLabel = document.getElementById('epoch-label');
    const timingEl = document.getElementById('timing');
    const playBtn = document.getElementById('play-btn');
    let timer = null;

    function decodeMask(b64, width, height) {{
      if (maskCache.has(b64)) return maskCache.get(b64);
      const bin = atob(b64);
      const arr = new Uint8ClampedArray(bin.length);
      for (let i = 0; i < bin.length; i++) arr[i] = bin.charCodeAt(i);
      const mask = {{ width, height, data: arr }};
      maskCache.set(b64, mask);
      return mask;
    }}

    function loadImage(src) {{
      if (overheadCache.has(src)) return overheadCache.get(src);
      const p = new Promise((resolve, reject) => {{
        const img = new Image();
        img.onload = () => resolve(img);
        img.onerror = reject;
        img.src = src;
      }});
      overheadCache.set(src, p);
      return p;
    }}

    function blendPixel(base, alpha, color) {{
      return [
        base[0] * (1 - alpha) + color[0] * alpha,
        base[1] * (1 - alpha) + color[1] * alpha,
        base[2] * (1 - alpha) + color[2] * alpha,
      ];
    }}

    function renderPanelToCanvas(panel, image, gtMask, predMask) {{
      const canvas = panel.canvas;
      const ctx = canvas.getContext('2d', {{ willReadFrequently: true }});
      const w = payload.panelWidth;
      const h = payload.panelHeight;
      canvas.width = w;
      canvas.height = h;

      if (state.mode === 'masks') {{
        const half = Math.floor(w / 2);
        const out = ctx.createImageData(w, h);
        const pixels = out.data;
        for (let y = 0; y < h; y++) {{
          for (let x = 0; x < w; x++) {{
            const p = (y * w + x) * 4;
            if (x < half) {{
              const gx = Math.min(gtMask.width - 1, Math.floor(x * gtMask.width / half));
              const idx = y * gtMask.width + gx;
              const v = gtMask.data[idx];
              pixels[p] = v;
              pixels[p + 1] = v;
              pixels[p + 2] = v;
            }} else {{
              const px = Math.min(predMask.width - 1, Math.floor((x - half) * predMask.width / Math.max(1, w - half)));
              const idx = y * predMask.width + px;
              const v = predMask.data[idx] / 255;
              pixels[p] = 255;
              pixels[p + 1] = 245 - 225 * v;
              pixels[p + 2] = 235 - 235 * v;
            }}
            pixels[p + 3] = 255;
          }}
        }}
        ctx.putImageData(out, 0, 0);
        ctx.fillStyle = 'rgba(0,0,0,0.75)';
        ctx.fillRect(6, 6, 32, 16);
        ctx.fillRect(half + 6, 6, 56, 16);
        ctx.fillStyle = '#fff';
        ctx.font = '12px ui-monospace, monospace';
        ctx.textBaseline = 'middle';
        ctx.fillText('y', 18, 14);
        ctx.fillText('y_pred', half + 14, 14);
        panel.gt = gtMask;
        panel.pred = predMask;
        return;
      }}

      if (state.mode === 'diff') {{
        const out = ctx.createImageData(w, h);
        const pixels = out.data;
        for (let i = 0; i < gtMask.data.length; i++) {{
          const p = i * 4;
          const diff = Math.abs(gtMask.data[i] - predMask.data[i]) / 255;
          pixels[p] = 255;
          pixels[p + 1] = 255 - 215 * diff;
          pixels[p + 2] = 255 - 215 * diff;
          pixels[p + 3] = 255;
        }}
        ctx.putImageData(out, 0, 0);
        panel.gt = gtMask;
        panel.pred = predMask;
        return;
      }}

      ctx.drawImage(image, 0, 0, w, h);
      const imageData = ctx.getImageData(0, 0, w, h);
      const pixels = imageData.data;
      for (let i = 0; i < gtMask.data.length; i++) {{
        const p = i * 4;
        let rgb = [pixels[p], pixels[p + 1], pixels[p + 2]];
        const gt = gtMask.data[i] / 255;
        const pred = predMask.data[i] / 255;
        if (state.mode === 'overlay:gt' || state.mode === 'overlay:both') rgb = blendPixel(rgb, gt * 0.7, [0, 180, 255]);
        if (state.mode === 'overlay:pred' || state.mode === 'overlay:both') rgb = blendPixel(rgb, pred * 0.95, [255, 80, 0]);
        pixels[p] = rgb[0];
        pixels[p + 1] = rgb[1];
        pixels[p + 2] = rgb[2];
      }}
      ctx.putImageData(imageData, 0, 0);
      panel.gt = gtMask;
      panel.pred = predMask;
    }}

    function updateHeader() {{
      const cfg = payload.runConfig;
      document.getElementById('run-title').textContent = `${{payload.runId}}  loss=${{cfg.loss}} gamma=${{cfg.gamma}} decoder=${{cfg.decoder}} d_model=${{cfg.d_model}} lr=${{cfg.lr}} n_params=${{cfg.n_params}}`;
      const m = payload.metricsByEpoch[String(state.epoch)] || {{}};
      document.getElementById('metrics-line').textContent = `Epoch ${{state.epoch}} | Train IoU: ${{(m['metrics/train/mask/iou'] ?? '—')}} | Eval IoU: ${{(m['metrics/eval/mask/iou'] ?? '—')}} | Train MSE: ${{(m['metrics/train/mask/mse'] ?? m['metrics/train/mse'] ?? '—')}} | Eval MSE: ${{(m['metrics/eval/mask/mse'] ?? m['metrics/eval/mse'] ?? '—')}} | Loss: ${{(m['loss/train/total'] ?? '—')}}`;
      epochLabel.textContent = String(state.epoch);
    }}

    async function renderAll() {{
      const t0 = performance.now();
      updateHeader();
      const epochData = payload.epochData[String(state.epoch)] || {{}};
      const tasks = panels.map(async (panel) => {{
        if (!panel.sample) {{
          panel.el.style.display = 'none';
          return;
        }}
        panel.el.style.display = '';
        panel.title.innerHTML = panel.sample.title;
        panel.el.classList.toggle('panel-sep', !!panel.sample.separator);
        const splitEpoch = epochData[panel.sample.split] || {{}};
        const sampleEpoch = splitEpoch[String(panel.sample.sample_idx)];
        if (!sampleEpoch) return;
        const [img] = await Promise.all([loadImage(panel.sample.overhead)]);
        const gt = decodeMask(sampleEpoch.gt, payload.panelWidth, payload.panelHeight);
        const pred = decodeMask(sampleEpoch.pred, payload.panelWidth, payload.panelHeight);
        renderPanelToCanvas(panel, img, gt, pred);
      }});
      await Promise.all(tasks);
      const t1 = performance.now();
      timingHistory.push(t1 - t0);
      const last = timingHistory[timingHistory.length - 1];
      const avg = timingHistory.reduce((a, b) => a + b, 0) / timingHistory.length;
      timingEl.textContent = `render last=${{last.toFixed(1)}}ms avg=${{avg.toFixed(1)}}ms over ${{timingHistory.length}} refreshes`;
    }}

    function setMode(mode) {{
      state.mode = mode;
      for (const btn of document.querySelectorAll('[data-mode]')) btn.classList.toggle('active', btn.dataset.mode === mode);
      renderAll();
    }}

    function setSubset(subset) {{
      state.subset = subset;
      for (const btn of document.querySelectorAll('[data-subset]')) btn.classList.toggle('active', btn.dataset.subset === subset);
      layoutPanels();
      renderAll();
    }}

    function togglePlay() {{
      state.playing = !state.playing;
      playBtn.classList.toggle('active', state.playing);
      playBtn.textContent = state.playing ? 'Pause' : 'Play';
      if (!state.playing) {{ clearInterval(timer); timer = null; return; }}
      timer = setInterval(() => {{
        const i = payload.epochs.indexOf(state.epoch);
        const next = payload.epochs[(i + 1) % payload.epochs.length];
        state.epoch = next;
        slider.value = String(next);
        renderAll();
      }}, 700);
    }}

    function onHover(ev, panel) {{
      const rect = panel.canvas.getBoundingClientRect();
      const x = Math.max(0, Math.min(payload.panelWidth - 1, Math.floor((ev.clientX - rect.left) * payload.panelWidth / rect.width)));
      const y = Math.max(0, Math.min(payload.panelHeight - 1, Math.floor((ev.clientY - rect.top) * payload.panelHeight / rect.height)));
      const idx = y * payload.panelWidth + x;
      const gt = (panel.gt?.data[idx] ?? 0) / 255;
      const pred = (panel.pred?.data[idx] ?? 0) / 255;
      tooltip.style.display = 'block';
      tooltip.style.left = `${{ev.clientX + 12}}px`;
      tooltip.style.top = `${{ev.clientY + 12}}px`;
      if (state.mode === 'diff') {{
        tooltip.textContent = `diff=${{Math.abs(gt - pred).toFixed(3)}}`;
      }} else {{
        tooltip.textContent = `gt=${{gt.toFixed(3)}}\npred=${{pred.toFixed(3)}}`;
      }}
    }}

    function hideTooltip() {{ tooltip.style.display = 'none'; }}

    slider.min = String(payload.epochs[0]);
    slider.max = String(payload.epochs[payload.epochs.length - 1]);
    slider.step = String(payload.epochs.length > 1 ? payload.epochs[1] - payload.epochs[0] : 1);
    slider.value = String(state.epoch);
    slider.addEventListener('input', (e) => {{ state.epoch = Number(e.target.value); renderAll(); }});
    playBtn.addEventListener('click', togglePlay);

    const modeButtons = document.getElementById('mode-buttons');
    for (const mode of modes) {{
      const btn = document.createElement('button');
      btn.dataset.mode = mode;
      btn.textContent = mode;
      btn.onclick = () => setMode(mode);
      if (mode === state.mode) btn.classList.add('active');
      modeButtons.appendChild(btn);
    }}

    const subsetButtons = document.getElementById('subset-buttons');
    for (const subset of subsets) {{
      const btn = document.createElement('button');
      btn.dataset.subset = subset;
      btn.textContent = subset;
      btn.onclick = () => setSubset(subset);
      if (subset === state.subset) btn.classList.add('active');
      subsetButtons.appendChild(btn);
    }}

    const maxPanels = payload.rows * payload.cols;
    const panels = Array.from({{ length: maxPanels }}, () => {{
      const el = document.createElement('div');
      el.className = 'panel';
      const title = document.createElement('div');
      title.className = 'panel-title';
      const canvas = document.createElement('canvas');
      el.appendChild(title);
      el.appendChild(canvas);
      grid.appendChild(el);
      const panel = {{ sample: null, el, title, canvas, gt: null, pred: null }};
      canvas.addEventListener('mousemove', (ev) => onHover(ev, panel));
      canvas.addEventListener('mouseleave', hideTooltip);
      return panel;
    }});

    function layoutPanels() {{
      let visible = [];
      if (state.subset === 'both') {{
        const perSplitCols = Math.max(1, Math.floor(payload.cols / 2));
        const perSplitCount = payload.rows * perSplitCols;
        const train = (payload.samplesBySplit.train || []).slice(0, perSplitCount);
        const evals = (payload.samplesBySplit.eval || []).slice(0, perSplitCount);
        for (let row = 0; row < payload.rows; row++) {{
          const rowStart = row * perSplitCols;
          for (let i = 0; i < perSplitCols; i++) {{
            const sample = train[rowStart + i];
            visible.push(sample ? {{ ...sample }} : null);
          }}
          for (let i = 0; i < perSplitCols; i++) {{
            const sample = evals[rowStart + i];
            visible.push(sample ? {{ ...sample, separator: i === 0 }} : null);
          }}
        }}
      }} else {{
        visible = (payload.samplesBySplit[state.subset] || []).slice(0, maxPanels).map((sample) => ({{ ...sample }}));
      }}
      grid.style.gridTemplateColumns = `repeat(${{payload.cols}}, minmax(0, 1fr))`;
      panels.forEach((panel, i) => {{ panel.sample = visible[i] || null; }});
    }}

    layoutPanels();
    renderAll();
  </script>
</body>
</html>"""


def save_viewer_html(path, payload, title="HTML Viewer"):
    path = Path(path)
    path.write_text(render_viewer_document(payload, title=title), encoding="utf-8")
    return path


def get_parser():
    parser = argparse.ArgumentParser(
        description="Build and open the HTML vibration viewer"
    )
    parser.add_argument("--run-id", default="mse-test-20260417-023214")
    parser.add_argument("--wandb-run-id", default="s23s10ng")
    parser.add_argument("--data-dir", default="eturok-weizmann/vibrations")
    parser.add_argument("--entity", default="eturok")
    parser.add_argument("--project", default="good-vibrations")
    parser.add_argument("--rows", type=int, default=2)
    parser.add_argument("--cols", type=int, default=4)
    parser.add_argument("--panel-width", type=int, default=220)
    parser.add_argument("--panel-height", type=int, default=196)
    parser.add_argument("--mask-rotate-k", type=int, default=0)
    parser.add_argument("--mask-flip-lr", action="store_true")
    parser.add_argument("--mask-flip-ud", action="store_true")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument(
        "--output",
        default="artifacts/html_viewer.html",
        help="Output HTML file path relative to repo root or absolute",
    )
    parser.add_argument("--title", default="HTML Viewer")
    parser.add_argument("--no-open", action="store_true")
    return parser


def main():
    args = get_parser().parse_args()
    t0 = time.perf_counter()
    payload = build_html_viewer_payload(
        run_id=args.run_id,
        wandb_run_id=args.wandb_run_id,
        data_dir=args.data_dir,
        entity=args.entity,
        project=args.project,
        rows=args.rows,
        cols=args.cols,
        target_size=(args.panel_width, args.panel_height),
        mask_rotate_k=args.mask_rotate_k,
        mask_flip_lr=args.mask_flip_lr,
        mask_flip_ud=args.mask_flip_ud,
        max_samples=args.max_samples,
    )
    t1 = time.perf_counter()

    out_path = Path(args.output)
    if not out_path.is_absolute():
        out_path = Path.cwd() / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_viewer_html(out_path, payload, title=args.title)
    t2 = time.perf_counter()

    uri = out_path.resolve().as_uri()
    opened = False if args.no_open else webbrowser.open(uri)

    print(f"payload build: {t1 - t0:.3f}s")
    print(f"save html: {t2 - t1:.3f}s")
    print(f"total: {t2 - t0:.3f}s")
    print(f"epochs: {payload['epochs']}")
    print(
        "visible samples: "
        f"train={len(payload['samplesBySplit']['train'])} "
        f"eval={len(payload['samplesBySplit']['eval'])}"
    )
    print(f"panel size: {payload['panelWidth']}x{payload['panelHeight']}")
    print(f"saved file: {out_path}")
    print(f"url: {uri}")
    print(f"opened browser: {opened}")


if __name__ == "__main__":
    main()
