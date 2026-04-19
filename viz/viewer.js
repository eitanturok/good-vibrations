'use strict';

const PAYLOAD = JSON.parse(document.getElementById('payload').textContent);

// Fast lookup maps
const sampleMap = {};
PAYLOAD.samples.forEach(s => { sampleMap[s.idx] = s; });
const runMap = {};
PAYLOAD.runs.forEach(r => { runMap[r.id] = r; });

// ── State ────────────────────────────────────────────────────────
const state = {
  filteredIdxs: PAYLOAD.samples.map(s => s.idx),
  activeRunIds: PAYLOAD.runs.map(r => r.id),
  viewMode: {},    // runId → 'masks'|'diff'|'overlay:gt'|'overlay:pred'|'overlay:both'
  epoch: {},       // runId → epoch string
  opacityGT: {},   // runId → float [0,1]
  opacityPred: {}, // runId → float [0,1]
};

PAYLOAD.runs.forEach(r => {
  const lastEpoch = r.epochs && r.epochs.length ? r.epochs[r.epochs.length - 1] : '';
  state.viewMode[r.id]    = 'masks';
  state.epoch[r.id]       = lastEpoch;
  state.opacityGT[r.id]   = 0.6;
  state.opacityPred[r.id] = 0.6;
});

// ── Mask decoding (cached) ────────────────────────────────────────
const maskCache = new Map();
function decodeMask(b64) {
  if (!b64) return null;
  if (maskCache.has(b64)) return maskCache.get(b64);
  const raw = atob(b64);
  const arr = new Float32Array(raw.length);
  for (let i = 0; i < raw.length; i++) arr[i] = raw.charCodeAt(i) / 255;
  maskCache.set(b64, arr);
  return arr;
}

// ── Canvas helpers ────────────────────────────────────────────────
function drawGrayscale(ctx, mask, maskW, maskH, x, y, w, h) {
  const tmp = document.createElement('canvas');
  tmp.width = maskW; tmp.height = maskH;
  const tctx = tmp.getContext('2d');
  const img = tctx.createImageData(maskW, maskH);
  for (let i = 0; i < mask.length; i++) {
    const v = Math.round(mask[i] * 255);
    img.data[i*4]   = v;
    img.data[i*4+1] = v;
    img.data[i*4+2] = v;
    img.data[i*4+3] = 255;
  }
  tctx.putImageData(img, 0, 0);
  ctx.imageSmoothingEnabled = false;
  ctx.drawImage(tmp, x, y, w, h);
}

function drawMaskOverlay(ctx, mask, maskW, maskH, opacityScale, x, y, w, h) {
  // Scale mask up to canvas size manually, then blend white pixels over existing canvas pixels.
  const existing = ctx.getImageData(0, 0, ctx.canvas.width, ctx.canvas.height);
  const d = existing.data;
  const scaleX = maskW / w;
  const scaleY = maskH / h;
  const x0 = Math.max(0, Math.floor(x));
  const y0 = Math.max(0, Math.floor(y));
  const x1 = Math.min(ctx.canvas.width, Math.ceil(x + w));
  const y1 = Math.min(ctx.canvas.height, Math.ceil(y + h));
  for (let py = y0; py < y1; py++) {
    for (let px = x0; px < x1; px++) {
      const mx = Math.min(maskW - 1, Math.floor((px - x) * scaleX));
      const my = Math.min(maskH - 1, Math.floor((py - y) * scaleY));
      const mi = my * maskW + mx;
      const alpha = mask[mi] * opacityScale;  // [0, 1]
      const pi = (py * ctx.canvas.width + px) * 4;
      d[pi]   = Math.round(d[pi]   * (1 - alpha) + 255 * alpha);
      d[pi+1] = Math.round(d[pi+1] * (1 - alpha) + 255 * alpha);
      d[pi+2] = Math.round(d[pi+2] * (1 - alpha) + 255 * alpha);
    }
  }
  ctx.putImageData(existing, 0, 0);
}

function maskIndexFromEvent(e, canvas, maskW, maskH, mode) {
  const r = canvas.getBoundingClientRect();
  let px = (e.clientX - r.left) / r.width;
  const py = (e.clientY - r.top) / r.height;

  let side = 'both';
  if (mode === 'masks' || mode === 'overlay:both') {
    side = px < 0.5 ? 'gt' : 'pred';
    if (side === 'pred') px = (px - 0.5) * 2;
    else px *= 2;
  } else if (mode === 'overlay:gt') {
    side = 'gt';
  } else if (mode === 'overlay:pred') {
    side = 'pred';
  }
  const mx = Math.max(0, Math.min(maskW - 1, Math.floor(px * maskW)));
  const my = Math.max(0, Math.min(maskH - 1, Math.floor(py * maskH)));
  return { side, index: my * maskW + mx };
}

const renderDebugCache = new Set();
function logRenderDebug(sample, runId, canvas, maskW, maskH, mode) {
  const key = `${runId}:${sample.idx}:${mode}`;
  if (renderDebugCache.has(key)) return;
  renderDebugCache.add(key);
  console.log('[viz debug]', {
    runId,
    sampleIdx: sample.idx,
    mode,
    overheadSourcePx: [sample.overhead_src_w || null, sample.overhead_src_h || null],
    overheadCanvasPx: [canvas.width, canvas.height],
    runMaskPx: [maskW, maskH],
    note: 'run masks are pooled 40x20 masks stretched to the full overhead image; no rotate/flip is applied',
  });
}

// Overhead image cache (data URL → HTMLImageElement)
const overheadCache = new Map();
function getOverhead(dataUrl) {
  if (overheadCache.has(dataUrl)) return Promise.resolve(overheadCache.get(dataUrl));
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => { overheadCache.set(dataUrl, img); resolve(img); };
    img.onerror = reject;
    img.src = dataUrl;
  });
}

function getSpeakerKey(speakers) {
  if (!Array.isArray(speakers)) return speakers != null ? String(speakers) : '';
  const active = [];
  speakers.forEach((value, i) => {
    if (Number(value)) active.push(String(i + 1));
  });
  return active.join('+');
}

function getSpeakerLabel(key) {
  return key ? `Speaker ${key}` : 'None';
}

function getRunLabels(mode) {
  if (mode === 'masks') return ['y', 'y_pred'];
  if (mode === 'diff') return ['y_pred - y'];
  if (mode === 'overlay:gt') return ['y'];
  if (mode === 'overlay:pred') return ['y_pred'];
  if (mode === 'overlay:both') return ['y', 'y_pred'];
  return [];
}

async function renderCell(canvas, sample, runId) {
  const ctx = canvas.getContext('2d');
  const W = canvas.width, H = canvas.height;
  const maskW = sample.mask_w || 20, maskH = sample.mask_h || 40;
  const mode   = state.viewMode[runId] || 'masks';
  const oGT    = state.opacityGT[runId] ?? 0.6;
  const oPred  = state.opacityPred[runId] ?? 0.6;

  const run       = runMap[runId];
  const epochData = run && run.preds ? run.preds[state.epoch[runId]] : null;
  const predEntry = epochData ? epochData[String(sample.idx)] : null;

  const gt   = decodeMask(sample.gt_mask);
  const pred = predEntry ? decodeMask(predEntry.pred) : null;
  logRenderDebug(sample, runId, canvas, maskW, maskH, mode);

  ctx.clearRect(0, 0, W, H);

  // If there's no data at all, show a clear placeholder
  if (!gt && !pred) {
    ctx.fillStyle = '#f5f5f5';
    ctx.fillRect(0, 0, W, H);
    ctx.fillStyle = '#bbb';
    ctx.font = '11px monospace';
    ctx.fillText('no data', 8, H / 2);
    return;
  }

  if (mode === 'masks') {
    if (gt)   drawGrayscale(ctx, gt,   maskW, maskH, 0,   0, W/2, H);
    if (pred) drawGrayscale(ctx, pred, maskW, maskH, W/2, 0, W/2, H);
    // divider line
    ctx.strokeStyle = '#aaa';
    ctx.beginPath(); ctx.moveTo(W/2, 0); ctx.lineTo(W/2, H); ctx.stroke();

  } else if (mode === 'diff') {
    if (gt && pred) {
      const diff = Float32Array.from(gt, (v, i) => (pred[i] - v + 1) / 2);
      drawGrayscale(ctx, diff, maskW, maskH, 0, 0, W, H);
    } else {
      ctx.fillStyle = '#ccc'; ctx.fillRect(0, 0, W, H);
      ctx.fillStyle = '#888'; ctx.font = '11px monospace';
      ctx.fillText('no pred', 8, H/2);
    }

  } else {
    // overlay modes
    const bmp = sample.overhead ? await getOverhead(sample.overhead) : null;
    if (mode === 'overlay:both') {
      if (bmp) {
        ctx.drawImage(bmp, 0, 0, W / 2, H);
        ctx.drawImage(bmp, W / 2, 0, W / 2, H);
      } else {
        ctx.fillStyle = '#ddd';
        ctx.fillRect(0, 0, W / 2, H);
        ctx.fillRect(W / 2, 0, W / 2, H);
      }
      if (gt) drawMaskOverlay(ctx, gt, maskW, maskH, oGT, 0, 0, W / 2, H);
      if (pred) drawMaskOverlay(ctx, pred, maskW, maskH, oPred, W / 2, 0, W / 2, H);
    } else if (bmp) {
      ctx.drawImage(bmp, 0, 0, W, H);
    } else {
      ctx.fillStyle = '#ddd'; ctx.fillRect(0, 0, W, H);
    }
    if (mode === 'overlay:gt' && gt) {
      drawMaskOverlay(ctx, gt, maskW, maskH, oGT, 0, 0, W, H);
    }
    if (mode === 'overlay:pred' && pred) {
      drawMaskOverlay(ctx, pred, maskW, maskH, oPred, 0, 0, W, H);
    }
  }

}

// ── Tooltip ───────────────────────────────────────────────────────
const tooltipEl = document.getElementById('tooltip');

function attachRunTooltip(canvas, sample, runId) {
  canvas.addEventListener('mousemove', e => {
    const maskW = sample.mask_w || 20, maskH = sample.mask_h || 40;
    const mode = state.viewMode[runId] || 'masks';
    const hit = maskIndexFromEvent(e, canvas, maskW, maskH, mode);
    if (!hit) {
      tooltipEl.style.display = 'none';
      return;
    }
    const { side, index: i } = hit;

    const gt = decodeMask(sample.gt_mask);
    const run = runMap[runId];
    const epochData = run && run.preds ? run.preds[state.epoch[runId]] : null;
    const predEntry = epochData ? epochData[String(sample.idx)] : null;
    const pred = predEntry ? decodeMask(predEntry.pred) : null;

    const parts = [];
    if (mode === 'diff') {
      if (gt) parts.push(`y: ${gt[i].toFixed(3)}`);
      if (pred) parts.push(`y_pred: ${pred[i].toFixed(3)}`);
      if (gt && pred) parts.push(`y_pred - y: ${(pred[i] - gt[i]).toFixed(3)}`);
    } else {
      if (side !== 'pred' && gt) parts.push(`y: ${gt[i].toFixed(3)}`);
      if (side !== 'gt' && pred) parts.push(`y_pred: ${pred[i].toFixed(3)}`);
    }
    if (!parts.length) return;

    tooltipEl.textContent = parts.join('  ');
    tooltipEl.style.display = 'block';
    tooltipEl.style.left = (e.clientX + 12) + 'px';
    tooltipEl.style.top  = (e.clientY + 12) + 'px';
  });
  canvas.addEventListener('mouseleave', () => { tooltipEl.style.display = 'none'; });
}

function attachHeatmapTooltip(canvas, mask, maskW, maskH) {
  canvas.addEventListener('mousemove', e => {
    const r = canvas.getBoundingClientRect();
    const mx = Math.max(0, Math.min(maskW-1, Math.floor((e.clientX-r.left)/r.width*maskW)));
    const my = Math.max(0, Math.min(maskH-1, Math.floor((e.clientY-r.top)/r.height*maskH)));
    tooltipEl.textContent = `(${mx},${my}) = ${mask[my*maskW+mx].toFixed(4)}`;
    tooltipEl.style.display = 'block';
    tooltipEl.style.left = (e.clientX+12) + 'px';
    tooltipEl.style.top  = (e.clientY+12) + 'px';
  });
  canvas.addEventListener('mouseleave', () => { tooltipEl.style.display = 'none'; });
}

// ── Table header ──────────────────────────────────────────────────
const CELL_W = 286, CELL_H = 255;

function getCanvasSizeForMode(mode) {
  return {
    width: mode === 'masks' || mode === 'overlay:both' ? CELL_W * 2 : CELL_W,
    height: CELL_H,
  };
}

function buildRunHeader(runId) {
  const run = runMap[runId];
  const cfg = run.config || {};
  const summary = [
    cfg.loss,
    cfg.gamma != null && `γ=${cfg.gamma}`,
    cfg.decoder,
    cfg.d_model && `d=${cfg.d_model}`,
  ].filter(Boolean).join(' · ');

  const lastEpoch = state.epoch[runId];

  const th = document.createElement('th');
  th.className = 'col-run';
  th.dataset.runId = runId;
  th.innerHTML = `
    <div class="run-header">
      <span class="run-name">${run.name || runId}</span>
      <span class="run-cfg">${summary}</span>
      <div class="run-controls">
        <select class="mode-select" data-run="${runId}">
          <option value="masks">Masks</option>
          <option value="diff">Diff (y_pred - y)</option>
          <option value="overlay:gt">Overlay y</option>
          <option value="overlay:pred">Overlay y_pred</option>
          <option value="overlay:both">Overlay y + y_pred</option>
        </select>
        <select class="epoch-select" data-run="${runId}">
          ${(run.epochs || []).map(e => `<option value="${e}"${e===lastEpoch?' selected':''}>${e}</option>`).join('')}
        </select>
      </div>
      <div class="opacity-controls" data-run="${runId}">
        <label>y <input type="range" class="opacity-gt" data-run="${runId}" min="0" max="1" step="0.05" value="${state.opacityGT[runId]}"></label>
        <label>y_pred <input type="range" class="opacity-pred" data-run="${runId}" min="0" max="1" step="0.05" value="${state.opacityPred[runId]}"></label>
      </div>
      <button class="close-col" data-run="${runId}">×</button>
    </div>`;
  return th;
}

function buildHeader() {
  const thead = document.getElementById('thead');
  thead.innerHTML = '';
  const tr = document.createElement('tr');

  const thId = document.createElement('th');
  thId.textContent = '#';
  tr.appendChild(thId);

  const thDs = document.createElement('th');
  thDs.className = 'col-dataset';
  thDs.textContent = 'Dataset';
  tr.appendChild(thDs);

  state.activeRunIds.forEach(id => tr.appendChild(buildRunHeader(id)));
  thead.appendChild(tr);
}

// ── Row rendering ─────────────────────────────────────────────────
let rowObserver = null;

function buildBody() {
  const tbody = document.getElementById('tbody');
  tbody.innerHTML = '';
  if (rowObserver) rowObserver.disconnect();

  rowObserver = new IntersectionObserver(entries => {
    entries.forEach(e => {
      if (e.isIntersecting) populateRow(e.target);
      else clearRow(e.target);
    });
  }, { rootMargin: '300px' });

  state.filteredIdxs.forEach(idx => {
    const tr = document.createElement('tr');
    tr.dataset.idx = String(idx);
    tr.style.height = CELL_H + 'px';

    // Placeholder cells
    const ncols = 2 + state.activeRunIds.length;
    for (let i = 0; i < ncols; i++) {
      const td = document.createElement('td');
      if (i === 0) td.style.cssText = 'min-width:40px;';
      else if (i === 1) td.className = 'col-dataset';
      else { td.className = 'col-run'; td.dataset.runId = state.activeRunIds[i-2]; }
      tr.appendChild(td);
    }
    tbody.appendChild(tr);
    rowObserver.observe(tr);
  });
}

function populateRow(tr) {
  if (tr.dataset.rendered === '1') return;
  tr.dataset.rendered = '1';

  const idx = Number(tr.dataset.idx);
  const sample = sampleMap[idx];
  if (!sample) return;

  const cells = tr.querySelectorAll('td');
  cells[0].textContent = idx;
  renderDatasetCell(cells[1], sample);
  state.activeRunIds.forEach((runId, i) => {
    if (cells[i+2]) renderRunCell(cells[i+2], sample, runId);
  });
}

function clearRow(tr) {
  if (tr.dataset.rendered !== '1') return;
  tr.dataset.rendered = '0';
  const cells = tr.querySelectorAll('td');
  for (let i = 1; i < cells.length; i++) cells[i].innerHTML = '';
}

// ── Dataset cell ──────────────────────────────────────────────────
function renderDatasetCell(td, sample) {
  td.innerHTML = '';

  const meta = document.createElement('div');
  meta.className = 'meta-text';
  meta.textContent = `${sample.object || '—'} n=${sample.n_objects ?? '?'} x=${sample.x != null ? sample.x.toFixed(1) : '?'} y=${sample.y != null ? sample.y.toFixed(1) : '?'}`;
  td.appendChild(meta);

  if (sample.overhead) {
    const title = document.createElement('div');
    title.className = 'figure-title';
    title.textContent = 'overhead image';
    td.appendChild(title);

    const img = document.createElement('img');
    img.src = sample.overhead;
    img.style.cssText = `width:${CELL_W}px;height:${CELL_H}px;display:block;margin:0 auto;`;
    td.appendChild(img);
  }

  if (sample.shift_heatmap) {
    const img = document.createElement('img');
    img.src = sample.shift_heatmap;
    img.style.cssText = 'width:80px;height:80px;display:block;margin-top:4px;image-rendering:pixelated;';
    td.appendChild(img);
  }

  for (const [label, src] of [['Chirp', sample.chirp_audio], ['Vibration', sample.sonified_audio]]) {
    if (!src) continue;
    const lbl = document.createElement('div');
    lbl.className = 'audio-label';
    lbl.textContent = label;
    const a = document.createElement('audio');
    a.controls = true; a.src = src;
    td.appendChild(lbl); td.appendChild(a);
  }
}

// ── Run cell ──────────────────────────────────────────────────────
function renderRunCell(td, sample, runId) {
  td.innerHTML = '';
  const mode = state.viewMode[runId] || 'masks';
  const canvasSize = getCanvasSizeForMode(mode);
  const labels = getRunLabels(mode);

  // Info row above canvas: split, MSE
  const run = runMap[runId];
  const epochData = run && run.preds ? run.preds[state.epoch[runId]] : null;
  const predEntry = epochData ? epochData[String(sample.idx)] : null;
  const info = document.createElement('div');
  info.className = 'cell-info';
  if (predEntry) {
    const tag = predEntry.split === 'train' ? 'train' : 'eval';
    const mse = predEntry.mse != null ? `MSE=${predEntry.mse.toFixed(4)}` : '';
    info.textContent = [tag, mse].filter(Boolean).join('  ');
  } else {
    info.textContent = '—';
  }
  td.appendChild(info);

  const labelRow = document.createElement('div');
  labelRow.className = 'subplot-labels';
  labelRow.style.width = `${canvasSize.width}px`;
  labelRow.style.margin = '0 auto 4px';
  labelRow.style.gridTemplateColumns = `repeat(${labels.length || 1}, minmax(0, 1fr))`;
  labels.forEach(label => {
    const el = document.createElement('div');
    el.className = 'subplot-label';
    el.textContent = label;
    labelRow.appendChild(el);
  });
  td.appendChild(labelRow);

  const canvas = document.createElement('canvas');
  canvas.width = canvasSize.width;
  canvas.height = canvasSize.height;
  canvas.style.cssText = `width:${canvasSize.width}px;height:${canvasSize.height}px;display:block;margin:0 auto;cursor:crosshair;`;
  td.appendChild(canvas);
  attachRunTooltip(canvas, sample, runId);
  renderCell(canvas, sample, runId).catch(err => {
    console.error(`[renderCell] idx=${sample.idx} run=${runId}:`, err);
    const ctx = canvas.getContext('2d');
    ctx.fillStyle = '#fcc';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.fillStyle = '#900';
    ctx.font = '11px monospace';
    ctx.fillText('render error: ' + err.message, 4, 20);
  });
}

// ── Redraw all visible run cells for one run column ───────────────
function reRenderColumn(runId) {
  document.querySelectorAll('tr[data-rendered="1"]').forEach(tr => {
    const idx = Number(tr.dataset.idx);
    const sample = sampleMap[idx];
    const cell = tr.querySelector(`td[data-run-id="${runId}"]`);
    if (cell && sample) renderRunCell(cell, sample, runId);
  });
}

// ── Filters ───────────────────────────────────────────────────────
function sampleMatchesSplit(sampleIdx, split) {
  return state.activeRunIds.some(runId => {
    const predsByEpoch = runMap[runId]?.preds;
    if (!predsByEpoch) return false;
    return Object.values(predsByEpoch).some(epochPreds => {
      const entry = epochPreds?.[String(sampleIdx)];
      return entry && entry.split === split;
    });
  });
}

function applyFilters() {
  const split = document.getElementById('filter-split').value;
  const obj   = document.getElementById('filter-object').value;
  const nobj  = document.getElementById('filter-nobjects').value;
  const speaker = document.getElementById('filter-speaker').value;

  state.filteredIdxs = PAYLOAD.samples.filter(s => {
    if (obj  && s.object    !== obj)           return false;
    if (nobj && s.n_objects !== parseInt(nobj)) return false;
    if (speaker && getSpeakerKey(s.speakers) !== speaker) return false;
    if (split && !sampleMatchesSplit(s.idx, split)) return false;
    return true;
  }).map(s => s.idx);

  buildBody();
}

// ── Full rebuild (header + body) ──────────────────────────────────
function rebuild() {
  buildHeader();
  applyFilters();
  syncEpochSlider();
  updateEpochLabel();
}

// ── Fullscreen dialog ─────────────────────────────────────────────
function openFullscreen(idx) {
  const sample = sampleMap[idx];
  if (!sample) return;
  const content = document.getElementById('fullscreen-content');
  content.innerHTML = '';

  function section(label) {
    const d = document.createElement('div');
    d.className = 'fs-section';
    const h = document.createElement('div');
    h.className = 'fs-label'; h.textContent = label;
    d.appendChild(h);
    content.appendChild(d);
    return d;
  }

  // Overhead
  if (sample.overhead) {
    const sec = section('Overhead Image');
    const img = document.createElement('img');
    img.src = sample.overhead;
    img.style.cssText = 'max-width:100%;display:block;';
    sec.appendChild(img);
  }

  // Metadata
  const metaSec = section('Metadata');
  const pre = document.createElement('pre');
  pre.style.cssText = 'font-size:12px;white-space:pre-wrap;';
  pre.textContent = JSON.stringify({
    idx: sample.idx, object: sample.object, n_objects: sample.n_objects,
    x: sample.x, y: sample.y, speakers: sample.speakers,
  }, null, 2);
  metaSec.appendChild(pre);

  // GT mask heatmap (large)
  const gt = decodeMask(sample.gt_mask);
  if (gt) {
    const maskW = sample.mask_w || 20, maskH = sample.mask_h || 40;
    const sec = section('Ground Truth Mask (hover for values)');
    const canvas = document.createElement('canvas');
    canvas.width = maskW * 16; canvas.height = maskH * 16;
    canvas.style.cssText = `display:block;width:${maskW*16}px;height:${maskH*16}px;image-rendering:pixelated;cursor:crosshair;`;
    sec.appendChild(canvas);
    const ctx = canvas.getContext('2d');
    drawGrayscale(ctx, gt, maskW, maskH, 0, 0, maskW*16, maskH*16);
    attachHeatmapTooltip(canvas, gt, maskW, maskH);
  }

  // Shift heatmap
  if (sample.shift_heatmap) {
    const sec = section('Shift Magnitude (10×10 lasers)');
    const img = document.createElement('img');
    img.src = sample.shift_heatmap;
    img.style.cssText = 'width:160px;height:160px;display:block;image-rendering:pixelated;';
    sec.appendChild(img);
  }

  // Audio
  for (const [label, src] of [['Chirp Audio', sample.chirp_audio], ['Vibration Audio (FFT sonified)', sample.sonified_audio]]) {
    if (!src) continue;
    const sec = section(label);
    const a = document.createElement('audio');
    a.controls = true; a.src = src;
    sec.appendChild(a);
  }

  document.getElementById('fullscreen-dialog').showModal();
}

// ── Playback ──────────────────────────────────────────────────────
let playInterval = null;
let frameIdx = 0;

function getMaxFrames() {
  const lengths = state.activeRunIds.map(id => runMap[id]?.epochs?.length || 0).filter(n => n > 0);
  return lengths.length ? Math.max(...lengths) : 1;
}

function syncEpochSlider() {
  const slider = document.getElementById('epoch-slider');
  if (!slider) return;
  const max = getMaxFrames() - 1;
  slider.max = max;
  slider.value = frameIdx % (max + 1);
}

function updateEpochLabel() {
  const label = document.getElementById('play-epoch-label');
  if (!label) return;
  const epochs = [...new Set(
    state.activeRunIds
      .map(id => state.epoch[id])
      .filter(Boolean)
  )];
  if (!epochs.length) {
    label.textContent = 'Epoch -';
  } else if (epochs.length === 1) {
    label.textContent = `Epoch ${epochs[0]}`;
  } else {
    label.textContent = `Epochs ${epochs.join(' · ')}`;
  }
}

function applyFrame() {
  // Each run picks its epoch based on its own list length
  PAYLOAD.runs.forEach(r => {
    if (!r.epochs.length) return;
    const ep = r.epochs[frameIdx % r.epochs.length];
    state.epoch[r.id] = ep;
    const sel = document.querySelector(`.epoch-select[data-run="${r.id}"]`);
    if (sel) sel.value = ep;
  });
  state.activeRunIds.forEach(reRenderColumn);
  syncEpochSlider();
  updateEpochLabel();
}

function playStep() {
  frameIdx = (frameIdx + 1) % getMaxFrames();
  applyFrame();
}

function togglePlay() {
  const btn = document.getElementById('play-btn');
  if (playInterval) {
    clearInterval(playInterval);
    playInterval = null;
    btn.textContent = '▶ Play';
  } else {
    const speed = parseInt(document.getElementById('play-speed').value);
    playInterval = setInterval(playStep, speed);
    btn.textContent = '⏸ Pause';
  }
}

// ── Event delegation ──────────────────────────────────────────────
document.addEventListener('change', e => {
  const el = e.target;
  const runId = el.dataset.run;

  if (el.classList.contains('mode-select') && runId) {
    state.viewMode[runId] = el.value;
    const opCtrl = document.querySelector(`.opacity-controls[data-run="${runId}"]`);
    if (opCtrl) opCtrl.style.display = el.value.startsWith('overlay') ? 'flex' : 'none';
    reRenderColumn(runId);
  }
  if (el.classList.contains('epoch-select') && runId) {
    state.epoch[runId] = el.value;
    reRenderColumn(runId);
    updateEpochLabel();
  }
  if (el.id === 'filter-split' || el.id === 'filter-object' || el.id === 'filter-nobjects' || el.id === 'filter-speaker') {
    applyFilters();
  }
});

document.addEventListener('input', e => {
  const el = e.target;
  const runId = el.dataset.run;
  if (el.classList.contains('opacity-gt') && runId) {
    state.opacityGT[runId] = parseFloat(el.value);
    reRenderColumn(runId);
  }
  if (el.classList.contains('opacity-pred') && runId) {
    state.opacityPred[runId] = parseFloat(el.value);
    reRenderColumn(runId);
  }
  if (el.id === 'play-speed' && playInterval) {
    clearInterval(playInterval);
    playInterval = setInterval(playStep, parseInt(el.value));
  }
  if (el.id === 'epoch-slider') {
    frameIdx = parseInt(el.value);
    applyFrame();
  }
});

document.addEventListener('click', e => {
  const el = e.target;
  const runId = el.dataset && el.dataset.run;

  if (el.classList.contains('close-col') && runId) {
    state.activeRunIds = state.activeRunIds.filter(id => id !== runId);
    rebuild();
  }
  if (el.id === 'add-run-btn') {
    const sel = document.getElementById('run-selector');
    const id  = sel.value;
    if (id && !state.activeRunIds.includes(id)) {
      state.activeRunIds.push(id);
      rebuild();
    }
  }
  if (el.id === 'play-btn') {
    togglePlay();
  }
});

document.addEventListener('dblclick', e => {
  const tr = e.target.closest('tr[data-idx]');
  if (tr) openFullscreen(Number(tr.dataset.idx));
});

document.getElementById('close-dialog').addEventListener('click', () => {
  document.getElementById('fullscreen-dialog').close();
});

// ── Bootstrap ─────────────────────────────────────────────────────
(function init() {
  // Populate filter dropdowns
  const objects = [...new Set(PAYLOAD.samples.map(s => s.object).filter(Boolean))].sort();
  const nObjects = [...new Set(PAYLOAD.samples.map(s => s.n_objects))].sort((a, b) => a - b);
  const speakerKeys = [...new Set(PAYLOAD.samples.map(s => getSpeakerKey(s.speakers)).filter(Boolean))]
    .sort((a, b) => a.localeCompare(b, undefined, { numeric: true }));

  const filterObj = document.getElementById('filter-object');
  filterObj.innerHTML = '<option value="">All</option>' +
    objects.map(o => `<option>${o}</option>`).join('');

  const filterNObj = document.getElementById('filter-nobjects');
  filterNObj.innerHTML = '<option value="">All</option>' +
    nObjects.map(n => `<option>${n}</option>`).join('');

  const filterSpeaker = document.getElementById('filter-speaker');
  filterSpeaker.innerHTML = '<option value="">All</option>' +
    speakerKeys.map(key => `<option value="${key}">${getSpeakerLabel(key)}</option>`).join('');

  // Populate run selector
  const runSel = document.getElementById('run-selector');
  runSel.innerHTML = PAYLOAD.runs.map(r =>
    `<option value="${r.id}">${r.name || r.id}</option>`
  ).join('');

  rebuild();
})();
