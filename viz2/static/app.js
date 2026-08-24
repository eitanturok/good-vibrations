'use strict';
const $ = (s) => document.querySelector(s);
const api = (u) => fetch(u).then((r) => r.json());

const S = {
  all: [], info: null, byId: {}, byPos: {},
  live: { sid: null, ch: 'avg', laser: 'avg', fi: null },
  d: null, probe: null, mode: null,
  probes: [], hues: {}, hot: null,
  fi: null, hoverFi: null,
  log: { heat: 1, spec: 1 },
  specMode: 'magphase', modeView: 'arrows', kind: 'clean',
  empty: false, anim: 0, asize: 1,
  f: { layouts: new Set(), nobj: new Set(), spk: new Set([1]) },  // empty == no filter
};

window.__S = S;
/* One filter, applied everywhere downstream: the sample list, the position scatter and
   the speaker ring all read from this, so they can never disagree about what exists. */
function pass(s) {
  return (!S.f.layouts.size || S.f.layouts.has(s.layout)) &&
         (!S.f.nobj.size || S.f.nobj.has(s.n)) &&
         (!S.f.spk.size || S.f.spk.has(s.spk));
}
const matches = () => S.all.filter(pass);

const shortId = (id) => String(id).replace(/^0+(?=\d)/, '');

const hz = (i) => (S.d ? S.d.freqs[i] : 0);
const fmt = (f) => (f >= 100 ? f.toFixed(0) : f.toFixed(1));
// Golden angle: separated at any count, stable when one is removed.
/* Colour encodes FREQUENCY: probes of one sample usually differ by frequency, and hue
   ordered along the band means the colour itself tells you where on the spectrum you are.
   Probes that land on the same frequency are separated by a lightness step. */
function hueForFi(fi) {
  const n = S.d ? S.d.freqs.length : 1;
  return Math.round(210 + (fi / Math.max(1, n - 1)) * 280) % 360;   // blue -> red sweep
}
const col = (p, a = 1) => `hsl(${p.hue} 68% ${p.lit ?? 47}% / ${a})`;
const css = (n) => getComputedStyle(document.body).getPropertyValue(n).trim();

/* ***** boot ***** */
(async function () {
  const j = await api('/api/samples');
  S.all = j.samples; S.info = j.info; S.rv = j.rv;
  for (const s of j.samples) {
    S.byId[s.id] = s;
    (S.byPos[s.pos] ??= {})[s.spk] = s.id;
  }
  buildFilters(); buildScatter(); buildSpk(); buildGrid(); wire(); applyFilter();
  await select(S.all.find((s) => !s.empty)?.id || S.all[0].id);
})();

/* ***** selection ***** */
async function select(sid) {
  if (!S.byId[sid]) return;
  S.live.sid = sid;
  $('#sid').value = shortId(sid);
  S.d = await api(`/api/sample/${sid}`);
  if (S.fi == null || S.fi >= S.d.freqs.length) S.fi = Math.floor(S.d.freqs.length / 3);
  await refresh();
}

async function refresh() {
  const { sid, ch, laser } = S.live;
  $('#scene').src = `/api/scene/${sid}.jpg?v=${S.rv}`;
  $('#smask').src = `/api/mask/${sid}.png?v=${S.rv}`;
  $('#heat').src = `/api/heat/${sid}.png?ch=${ch}&log=${S.log.heat}&v=${S.rv}`;
  $('#audio').src = `/api/audio/${sid}.wav?ch=${ch === 'avg' ? 'x' : ch}&laser=${laser}`;
  S.probe = await api(`/api/probe/${sid}?ch=${ch}&laser=${laser}&kind=${S.kind}`);
  await loadMode();
  paintAll();
}

async function loadMode() {
  S.mode = await api(`/api/mode/${S.live.sid}?fi=${S.fi}`);
}

function paintAll() {
  const s = S.byId[S.live.sid];
  $('#summary').textContent =
    `${shortId(s.id)} · pos ${s.pos} · spk ${s.spk} · ${s.layout} · L${S.live.laser} · ${S.live.ch}`;
  buildPeaks(); drawSpec(); drawShifts(); drawMode(); meanStrip(); cursors(); axes(); legends();
  readout();
  renderScatter(); renderSpk(); renderGrid(); renderProbes();
}

/* ***** curves: live probe + held probes in one axes ***** */
const series = () => [
  ...S.probes.map((p) => ({ p, d: p.data, c: col(p, S.hot && S.hot !== p ? 0.18 : 0.85), w: 1.4 })),
  // Esc mutes the live trace so held probes can be compared without it on top.
  ...(S.muted ? [] : [{ p: null, d: S.probe, c: css('--ink'), w: 1.7, dim: !!S.preview }]),
  ...(S.preview && !S.muted ? [{ p: null, d: S.preview, c: css('--accent'), w: 1.7 }] : []),
];

function fit(cv) {
  const r = cv.getBoundingClientRect(), dp = devicePixelRatio || 1;
  cv.width = r.width * dp; cv.height = r.height * dp;
  const g = cv.getContext('2d');
  g.setTransform(dp, 0, 0, dp, 0, 0);
  g.clearRect(0, 0, r.width, r.height);
  return [g, r.width, r.height];
}

function line(g, y, w, h, c, lw) {
  let lo = Infinity, hi = -Infinity;
  for (const v of y) { if (v < lo) lo = v; if (v > hi) hi = v; }
  const k = h / ((hi - lo) || 1);
  g.beginPath();
  for (let i = 0; i < y.length; i++) {
    const X = (i / (y.length - 1)) * w, Y = h - (y[i] - lo) * k;
    i ? g.lineTo(X, Y) : g.moveTo(X, Y);
  }
  g.strokeStyle = c; g.lineWidth = lw; g.stroke();
}

// Shared y-range so overlaid probes are actually comparable.
function span(list, key) {
  let lo = Infinity, hi = -Infinity;
  for (const s of list) for (const v of s.d[key]) { if (v < lo) lo = v; if (v > hi) hi = v; }
  return [lo, hi];
}

function multi(cv, key) {
  const [g, w, h] = fit(cv), ss = series().filter((s) => s.d);
  if (!ss.length) return;
  const [lo, hi] = span(ss, key), k = h / ((hi - lo) || 1);
  if (cv.id === 'p1') S.p1 = { lo, hi, h };   // so peak markers can sit ON the curve
  for (const s of ss) {
    const y = s.d[key];
    g.beginPath();
    for (let i = 0; i < y.length; i++) {
      const X = (i / (y.length - 1)) * w, Y = h - (y[i] - lo) * k;
      i ? g.lineTo(X, Y) : g.moveTo(X, Y);
    }
    g.globalAlpha = s.dim ? 0.28 : 1;
    g.strokeStyle = s.c; g.lineWidth = s.w; g.stroke();
    g.globalAlpha = 1;
  }
}

function drawSpec() {
  const a = S.specMode === 'magphase' ? (S.log.spec ? 'logmag' : 'mag') : 're';
  const b = S.specMode === 'magphase' ? 'phase' : 'im';
  multi($('#p1'), a); multi($('#p2'), b);
}

const drawShifts = () => multi($('#p3'), 'shifts');

/* ***** mode shape ***** */
function modeAxes() {
  const svg = $('#modeax');
  if (!svg) return;
  const R = S.info.rows, C = S.info.cols;
  let out = '';
  for (let r = 0; r < R; r++)
    out += `<text class="ml" x="-4" y="${((r + 0.5) / R) * 100}%" dy="3">${r}</text>`;
  for (let c = 0; c < C; c++)
    out += `<text class="ml col" x="${((c + 0.5) / C) * 100}%" y="100%" dy="12">${c}</text>`;
  svg.innerHTML = out;
}

function drawMode() {
  modeAxes();
  const cv = $('#mode'), [g, w, h] = fit(cv);
  if (!S.mode) return;
  const R = S.info.rows, C = S.info.cols, cw = w / C, chh = h / R;
  const sets = [...S.probes.filter((p) => p.mode).map((p) => ({ m: p.mode, c: col(p, 0.9) })),
                ...(S.muted && S.probes.length ? [] : [{ m: S.mode, c: css('--ink') }])];

  if (S.modeView === 'field' || sets.length > 4) {
    field(g, w, h, S.mode, R, C);          // dense fill cannot overlay
    if (sets.length > 4) { arrows(g, [sets.at(-1)], w, h, R, C, cw, chh, 1, 0); return; }
  }
  // Arrows from different probes share a cell origin, so colour alone would hide one
  // under another: fan the tails around the cell centre instead.
  let max = 0;
  for (const s of sets) for (let r = 0; r < R; r++) for (let c = 0; c < C; c++)
    max = Math.max(max, Math.hypot(s.m.u[r][c], s.m.v[r][c]));
  const k = (Math.min(cw, chh) * 0.42 * S.asize) / (max || 1);
  arrows(g, sets, w, h, R, C, cw, chh, k, sets.length > 1 ? Math.min(cw, chh) * 0.15 : 0);
}

function arrows(g, sets, w, h, R, C, cw, chh, k, fan) {
  const t = S.anim ? Math.cos(performance.now() / 300) : 1;
  sets.forEach((s, n) => {
    const a = fan ? (n / sets.length) * 2 * Math.PI : 0;
    const ox = fan * Math.cos(a), oy = fan * Math.sin(a);
    g.strokeStyle = s.c; g.fillStyle = s.c; g.lineWidth = 1.3;
    for (let r = 0; r < R; r++) for (let c = 0; c < C; c++) {
      const x = c * cw + cw / 2 + ox, y = r * chh + chh / 2 + oy;
      const dx = s.m.u[r][c] * k * t, dy = -s.m.v[r][c] * k * t;
      g.beginPath(); g.moveTo(x, y); g.lineTo(x + dx, y + dy); g.stroke();
      g.beginPath(); g.arc(x + dx, y + dy, 1.5, 0, 7); g.fill();
    }
  });
}

// Direction -> hue, magnitude -> value. (spatial_derivatives_to_hsv, reimplemented.)
function field(g, w, h, m, R, C) {
  let max = 0;
  for (let r = 0; r < R; r++) for (let c = 0; c < C; c++)
    max = Math.max(max, Math.hypot(m.u[r][c], m.v[r][c]));
  const cw = w / C, chh = h / R;
  for (let r = 0; r < R; r++) for (let c = 0; c < C; c++) {
    const u = m.u[r][c], v = m.v[r][c];
    const deg = (Math.atan2(v, u) * 180) / Math.PI + 180;
    g.fillStyle = `hsl(${deg} 70% ${18 + 52 * (Math.hypot(u, v) / (max || 1))}%)`;
    g.fillRect(c * cw, r * chh, cw + 1, chh + 1);
  }
}

/* ***** crosshair: SVG only, so hover never repaints a canvas *****
   The nodes are built once and then only moved. Rebuilding innerHTML on pointermove
   would destroy the peak dots out from under a click that is already in flight. */
const OVS = ['#heatov', '#p1ov', '#p2ov'];

/* Axes. Ticks are drawn into the same SVG overlay as the cursor, so they cost nothing
   extra and stay pinned to the plot regardless of canvas resolution. */
function ticks(lo, hi, n = 5) {
  const raw = (hi - lo) / n;
  const mag = 10 ** Math.floor(Math.log10(raw));
  const step = [1, 2, 2.5, 5, 10].find((m) => m * mag >= raw) * mag;
  const out = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi + 1e-9; v += step) out.push(v);
  return out;
}

/* Only the BOTTOM plot of a stack shows tick numbers -- the ones above share the same
   scale, so repeating them is noise. Numbers live in a gutter below the plot so they
   never overprint the data. */
function xAxis(svg, lo, hi, label, showNums) {
  const g = svg.querySelector('.ax') || svg.insertBefore(
    document.createElementNS('http://www.w3.org/2000/svg', 'g'), svg.firstChild);
  g.setAttribute('class', 'ax');
  const t = ticks(lo, hi);
  let out = t.map((v) => {
    const x = ((v - lo) / (hi - lo)) * 100;
    return `<line class="grid" x1="${x}%" x2="${x}%" y1="0" y2="100%"/>`;
  }).join('');
  if (showNums) {
    const own = svg.id === 'heatax';            // its own strip: draw from the top
    const y = own ? '0' : '100%';
    out += t.map((v) => {
      const x = ((v - lo) / (hi - lo)) * 100;
      return `<text class="tk" x="${x}%" y="${y}" dy="11">${v}</text>`;
    }).join('') + `<text class="unit" x="50%" y="${y}" dy="23">${label}</text>`;
  }
  g.innerHTML = out;
}

/* Laser index down the side of the heatmap, every 10. */
function laserAxis() {
  const svg = $('#heatov');
  let g = svg.querySelector('.lax');
  if (!g) {
    g = document.createElementNS('http://www.w3.org/2000/svg', 'g');
    g.setAttribute('class', 'lax');
    svg.appendChild(g);
  }
  const L = S.info.n_lasers;
  let out = '';
  for (let i = 0; i < L; i += 10)
    out += `<text class="ly" x="-5" y="${((i + 0.5) / L) * 100}%" dy="3">${i}</text>`;
  g.innerHTML = out;
}

function yLabel(sel, text) {
  const el = document.querySelector(sel);
  if (el) el.textContent = text;
}

function buildOverlays() {
  for (const id of OVS) {
    $(id).innerHTML = '<line class="cur hov" style="opacity:.4" y1="0" y2="100%"/>' +
                      '<line class="cur sel" y1="0" y2="100%"/>' +
                      (id === '#p1ov' ? '<g class="pks"></g>' : '');
  }
}

const LABELLED = 5;

// Markers ride the magnitude curve itself, using the same y-mapping multi() drew with.
function peakY(i) {
  const m = S.p1;
  if (!m || !S.probe) return 10;
  const mag = S.specMode === 'magphase'
    ? (S.log.spec ? S.probe.logmag : S.probe.mag) : S.probe.re;
  return m.h - ((mag[i] - m.lo) / ((m.hi - m.lo) || 1)) * m.h;
}

function buildPeaks() {
  const g = $('#p1ov').querySelector('.pks');
  if (!g) return;
  const pk = S.probe?.peaks || [];
  // Rank by magnitude so the labels go on the peaks that actually dominate.
  const mag = S.probe ? (S.log.spec ? S.probe.logmag : S.probe.mag) : null;
  const top = new Set(mag
    ? [...pk].sort((x, y) => mag[y] - mag[x]).slice(0, LABELLED)
    : pk.slice(0, LABELLED));
  g.innerHTML = pk.map((i) =>
    `<circle class="pk" r="3.4" data-fi="${i}"/>` +
    (top.has(i) ? `<text class="pklab" data-fi="${i}">${fmt(hz(i))}</text>` : '')
  ).join('');
  g.querySelectorAll('.pk').forEach((c) => (c.onclick = (e) => {
    e.stopPropagation();
    setFi(+c.dataset.fi);
  }));
}

function cursors() {
  laserRow();
  probeTicks();
  const n = S.d ? S.d.freqs.length : 1;
  const at = (i) => `${(i / (n - 1)) * 100}%`;
  for (const id of OVS) {
    const svg = $(id);
    const sel = svg.querySelector('.sel'), hov = svg.querySelector('.hov');
    sel.setAttribute('x1', at(S.fi)); sel.setAttribute('x2', at(S.fi));
    const show = S.hoverFi != null && S.hoverFi !== S.fi;
    hov.style.display = show ? '' : 'none';
    if (show) { hov.setAttribute('x1', at(S.hoverFi)); hov.setAttribute('x2', at(S.hoverFi)); }
  }
  $('#p1ov').querySelectorAll('.pk').forEach((c) => {
    const i = +c.dataset.fi;
    c.setAttribute('cx', at(i)); c.setAttribute('cy', peakY(i));
    c.classList.toggle('on', i === S.fi);
  });
  $('#p1ov').querySelectorAll('.pklab').forEach((tx) => {
    const i = +tx.dataset.fi;
    tx.setAttribute('x', at(i)); tx.setAttribute('y', peakY(i));
    tx.setAttribute('dy', -8);
    tx.classList.toggle('on', i === S.fi);
  });
}

async function setFi(i) {
  S.fi = i; S.modePreview = null; modeSeq++; await loadMode(); paintAll();
}

/* Walk the detected peaks. Once you land on one, plain left/right steps peak to peak --
   which is what you want after clicking one -- and shift+arrows always do. */
function stepPeak(d) {
  const pk = S.probe?.peaks || [];
  if (!pk.length) return;
  const at = pk.indexOf(S.fi);
  const next = at === -1
    ? (d > 0 ? pk.find((i) => i > S.fi) ?? pk[0] : [...pk].reverse().find((i) => i < S.fi) ?? pk.at(-1))
    : pk[Math.max(0, Math.min(pk.length - 1, at + d))];
  setFi(next);
}

/* A mean-over-lasers strip above and below the heatmap, in the same colour ramp, so the
   average spectrum reads at a glance without occupying a separate panel. */
function meanStrip() {
  if (!S.probe) return;
  const v = S.log.heat ? S.probe.logmag : S.probe.mag;
  const [lo, hi] = S.probe.domain, n = v.length;
  for (const id of ['#heatmean']) {
    const cv = $(id);
    if (!cv) continue;
    const [g, w, h] = fit(cv);
    for (let x = 0; x < w; x++) {
      const t = Math.min(1, Math.max(0, (v[Math.floor((x / w) * n)] - lo) / (hi - lo || 1)));
      g.fillStyle = ramp(t);
      g.fillRect(x, 0, 1, h);
    }
  }
  const cb = $('#cbar');
  if (cb) {
    const [g, w, h] = fit(cb);
    for (let y = 0; y < h; y++) { g.fillStyle = ramp(1 - y / h); g.fillRect(0, y, w, 1); }
  }
  $('#cblab').textContent = quantity();
  $('#cbmin').textContent = lo.toFixed(1);
  $('#cbmax').textContent = hi.toFixed(1);
}

// What the heatmap and its colour ramp actually show.
const quantity = () => (S.log.heat ? 'log₁₀ |FFT|' : '|FFT|') +
  ` · ${S.live.ch} channel`;

// Same stops as viz2/render.py's SEQ, so the strip and the PNG agree.
const SEQ = ['#ffffff', '#cde2fb', '#9ec5f4', '#6da7ec', '#3987e5', '#256abf', '#184f95', '#0d366b'];
function ramp(t) {
  const i = Math.min(SEQ.length - 2, Math.floor(t * (SEQ.length - 1)));
  const f = t * (SEQ.length - 1) - i;
  const a = SEQ[i].match(/\w\w/g).map((h) => parseInt(h, 16));
  const b = SEQ[i + 1].match(/\w\w/g).map((h) => parseInt(h, 16));
  return `rgb(${a.map((v, k) => Math.round(v + (b[k] - v) * f)).join(',')})`;
}

/* Each held probe is pinned to one frequency, but nothing on the spectrum showed where.
   A tick per probe, in its colour, makes position -- the most precise visual channel --
   do the separating, which is what rescues several probes of the SAME sample. */
function probeTicks() {
  const n = S.d ? S.d.freqs.length : 1;
  for (const id of OVS) {
    const svg = $(id);
    let g = svg.querySelector('.pticks');
    if (!g) {
      g = document.createElementNS('http://www.w3.org/2000/svg', 'g');
      g.setAttribute('class', 'pticks');
      svg.insertBefore(g, svg.firstChild);
    }
    g.innerHTML = S.probes.map((p) => {
      const x = (p.fi / (n - 1)) * 100;
      const dim = S.hot && S.hot !== p;
      return `<line class="ptick" x1="${x}%" x2="${x}%" y1="0" y2="100%" ` +
             `stroke="${col(p, dim ? 0.2 : 0.9)}"/>`;
    }).join('');
  }
}

function laserRow() {
  $('#heatmean').classList.toggle('sel', S.live.laser === 'avg');
  const svg = $('#heatov');
  let g = svg.querySelector('.lrow');
  if (!g) {
    g = document.createElementNS('http://www.w3.org/2000/svg', 'g');
    g.setAttribute('class', 'lrow');
    svg.appendChild(g);
  }
  const L = S.info.n_lasers, h = 100 / L;
  const rows = [];
  if (S.live.laser !== 'avg') rows.push([+S.live.laser, 'sel']);
  if (S.hoverLaser != null && S.hoverLaser !== +S.live.laser) rows.push([S.hoverLaser, 'hov']);
  g.innerHTML = rows.map(([i, c]) =>
    `<rect class="lr ${c}" x="0" y="${i * h}%" width="100%" height="${h}%"/>`).join('');
}

function axes() {
  if (!S.d) return;
  const f = S.d.freqs, lo = f[0], hi = f[f.length - 1];
  const HZ = 'frequency (Hz)';
  xAxis($('#heatov'), lo, hi, HZ, false);     // grid on the image
  xAxis($('#heatax'), lo, hi, HZ, true);      // numbers in their own strip below it
  xAxis($('#p1ov'), lo, hi, HZ, false);       // top of the spectrum stack: grid only
  xAxis($('#p2ov'), lo, hi, HZ, true);        // bottom carries the numbers
  if (S.probe) xAxis($('#p3ov'), 0, S.probe.dur, 'time (s)', true);
  laserAxis();

  const mp = S.specMode === 'magphase';
  yLabel('#y1', mp ? (S.log.spec ? 'log |FFT|' : '|FFT|') : 'real');
  yLabel('#y2', mp ? 'phase (rad)' : 'imag');
  yLabel('#yh', 'laser');
  yLabel('#y3', 'shift (px)');
}

function freqAxis(el, laserAxis) {
  el.style.cursor = 'crosshair';
  el.addEventListener('pointermove', (e) => {
    const b = el.getBoundingClientRect();
    const n = S.d.freqs.length;
    S.hoverFi = Math.max(0, Math.min(n - 1, Math.round(((e.clientX - b.left) / b.width) * (n - 1))));
    if (laserAxis) {
      const L = S.info.n_lasers;
      const i = Math.max(0, Math.min(L - 1, Math.floor(((e.clientY - b.top) / b.height) * L)));
      if (i !== S.hoverLaser) { S.hoverLaser = i; previewLaser(i); }
    }
    cursors();
    readout();
    previewMode(S.hoverFi);
  });
  el.addEventListener('pointerleave', () => {
    S.hoverFi = null;
    if (laserAxis && S.hoverLaser != null) { S.hoverLaser = null; previewLaser(null); }
    if (S.modePreview) previewMode(null);
    cursors(); readout();
  });
  el.addEventListener('click', () => {
    if (laserAxis && S.hoverLaser != null) return setLaser(S.hoverLaser);
    if (S.hoverFi != null) setFi(S.hoverFi);
  });
}

function readout() {
  const f = S.hoverFi ?? S.fi;
  const l = S.hoverLaser ?? S.live.laser;
  const s = S.byId[S.live.sid];
  if (!s) return;

  /* Every title is "Pos X, Spk X (sample): <the fields that step depends on>", so the
     shared prefix names the recording and the tail names what varies. */
  const scene = `Pos <b>${s.pos}</b>, Spk <b>${s.spk}</b> <u>(${shortId(s.id)})</u>`;
  const laser = `Laser <b>${l === 'avg' ? 'avg' : l}</b>`;
  const chan = `Chn <b>${S.live.ch}</b>`;
  const freq = `Freq <b>${fmt(hz(f))} Hz</b>`;
  const title = (...parts) => `${scene}: ${parts.join(', ')}`;

  $('#lftitle').innerHTML = title(laser, chan, freq);
  $('#sigtitle').innerHTML = title(laser, chan, freq);
  // Audio reconstructs one laser's single channel -- an average of complex spectra does
  // not invert -- and spans the whole band, so no frequency.
  const ach = S.live.ch === 'avg' ? 'x' : S.live.ch;
  $('#audtitle').innerHTML = title(
    `Laser <b>${S.live.laser === 'avg' ? 'avg' : S.live.laser}</b>`,
    `Chn <b>${ach}</b>`);
  // The mode is every laser at one frequency, so laser is not a field here.
  $('#modetitle').innerHTML = title(chan, freq);

  $('#heatread').textContent = S.hoverLaser != null ? `laser ${l} — click to select` : '';
  renderNow();
}

/* Hovering a heatmap row previews that laser's spectrum without committing to it, so you
   can sweep the grid and watch the curve move. Latest-wins guards against out-of-order
   replies when the pointer moves faster than the fetches return. */
let modeSeq = 0, modePending = false;
async function previewMode(fi) {
  const seq = ++modeSeq;
  if (fi == null) {                       // back to the committed frequency
    S.modePreview = null;
    await loadMode();
    if (seq === modeSeq) drawMode();
    return;
  }
  if (modePending) return;                // coalesce: one request in flight at a time
  modePending = true;
  try {
    const d = await api(`/api/mode/${S.live.sid}?fi=${fi}`);
    if (seq !== modeSeq) return;
    S.modePreview = d; S.mode = d;
    drawMode();
  } finally { modePending = false; }
}

let previewSeq = 0;
async function previewLaser(i) {
  const seq = ++previewSeq;
  if (i == null) { S.preview = null; if (seq === previewSeq) { drawSpec(); drawShifts(); legends(); } return; }
  const { sid, ch } = S.live;
  const d = await api(`/api/probe/${sid}?ch=${ch}&laser=${i}&kind=${S.kind}`);
  if (seq !== previewSeq) return;
  S.preview = d;
  drawSpec(); drawShifts(); legends();
}

/* ***** filters ***** */
// A stable hue per value, so a chip keeps its colour no matter what else is selected.
const chipHue = (n) => (28 + n * 47) % 360;

function buildFilters() {
  const layouts = [...new Set(S.all.map((s) => s.layout))]
    .sort((a, b) => S.all.filter((s) => s.layout === b).length -
                    S.all.filter((s) => s.layout === a).length);
  const nobj = [...new Set(S.all.map((s) => s.n))].sort((a, b) => a - b);
  const chip = (k, v, lab, n, h) =>
    `<button class="chip" data-k="${k}" data-v="${v}" style="--h:${h}">` +
    `<span class="dot"></span>${lab}<i>${n}</i></button>`;
  $('#layoutchips').innerHTML = layouts.map((l, i) =>
    chip('layouts', l, l, S.all.filter((s) => s.layout === l).length, chipHue(i))).join('');
  $('#nobjchips').innerHTML = nobj.map((n, i) =>
    chip('nobj', n, `${n} obj`, S.all.filter((s) => s.n === n).length, chipHue(i * 3 + 1))).join('');
  document.querySelectorAll('.chips .chip').forEach((b) => (b.onclick = () => {
    const set = S.f[b.dataset.k];
    const v = b.dataset.k === 'nobj' ? +b.dataset.v : b.dataset.v;
    set.has(v) ? set.delete(v) : set.add(v);
    applyFilter();
  }));
  document.querySelectorAll('[data-all],[data-none]').forEach((b) => (b.onclick = () => {
    const k = b.dataset.all || b.dataset.none;
    S.f[k].clear();                             // "all" == no constraint
    if (b.dataset.none) S.f[k].add(Symbol('none'));   // impossible value -> zero matches
    applyFilter();
  }));
}

/* A real dropdown under the field: a native <datalist> pops where the browser wants and
   cannot be styled or made to show the short ids. */
function renderSidList(q = '') {
  const el = $('#sidlist');
  const m = matches().filter((s) => !q || shortId(s.id).startsWith(q) || s.id.startsWith(q));
  el.innerHTML = m.slice(0, 200).map((s) =>
    `<button class="opt${s.id === S.live.sid ? ' on' : ''}" data-id="${s.id}">` +
    `<b>${shortId(s.id)}</b><i>pos ${s.pos} · spk ${s.spk} · ${s.layout}</i></button>`).join('') +
    (m.length > 200 ? `<div class="more">+${m.length - 200} more — keep typing</div>` : '') +
    (m.length ? '' : '<div class="more">no match</div>');
  el.querySelectorAll('.opt').forEach((b) => (b.onmousedown = (e) => {
    e.preventDefault();
    pickSample(b.dataset.id);
    el.hidden = true;
    $('#sid').blur();
  }));
}

function pickSample(id) {
  if (!S.byId[id]) return;
  if (!pass(S.byId[id])) { S.f.layouts.clear(); S.f.nobj.clear(); applyFilter(); }
  select(id);
}

function applyFilter() {
  const m = matches();
  document.querySelectorAll('.chips .chip').forEach((b) => {
    const set = S.f[b.dataset.k];
    const v = b.dataset.k === 'nobj' ? +b.dataset.v : b.dataset.v;
    b.classList.toggle('on', set.has(v));
    b.classList.toggle('off', set.size > 0 && !set.has(v));
  });
  renderSidList();
  $('#nmatch').textContent = `${m.length} of ${S.all.length}`;
  buildScatter();
  // Keep the current sample if it still passes; otherwise jump to the first that does.
  if (m.length && !pass(S.byId[S.live.sid] || {})) select(m[0].id);
  else { renderScatter(); renderSpk(); }
}

/* ***** pickers ***** */
// Positions available for the current speaker. On this dataset every speaker covers all
// 369 positions, so nothing is filtered out -- but the dependency is real on a partial one.
const POS = () => [...new Set(matches().filter((s) => !s.empty).map((s) => s.pos))];

const SW = 250, SH = 78;    // scatter viewBox (wide + short keeps the top bar shallow)

function scale() {
  const pts = POS().map((p) => S.byId[S.byPos[p][Object.keys(S.byPos[p])[0]]].com);
  const rs = pts.map((c) => c[0]), cs = pts.map((c) => c[1]);
  const r0 = Math.min(...rs), r1 = Math.max(...rs), c0 = Math.min(...cs), c1 = Math.max(...cs);
  const pad = 7;
  const k = Math.min((SW - 2 * pad) / ((c1 - c0) || 1), (SH - 2 * pad) / ((r1 - r0) || 1));
  const ox = (SW - (c1 - c0) * k) / 2, oy = (SH - (r1 - r0) * k) / 2;
  return { x: (c) => ox + (c - c0) * k, y: (r) => oy + (r - r0) * k, pts };
}

function buildScatter() {
  const svg = $('#scatter'), s = scale();
  svg.setAttribute('viewBox', `0 0 ${SW} ${SH}`);
  svg.innerHTML = POS().map((p, i) =>
    `<circle class="pt" data-p="${p}" cx="${s.x(s.pts[i][1]).toFixed(1)}" cy="${s.y(s.pts[i][0]).toFixed(1)}" r="1.9"/>`).join('')
    + '<circle class="hot" r="3.4" hidden/>';
  const near = (ev) => {
    const b = svg.getBoundingClientRect();
    const x = ((ev.clientX - b.left) / b.width) * SW, y = ((ev.clientY - b.top) / b.height) * SH;
    let best = null, bd = 1e9;
    POS().forEach((p, i) => {
      const d = Math.hypot(s.x(s.pts[i][1]) - x, s.y(s.pts[i][0]) - y);
      if (d < bd) { bd = d; best = p; }
    });
    return bd < 6 ? best : null;
  };
  svg.onclick = (e) => { const p = near(e); if (p != null) goPos(p); svg.focus(); };
  svg.onkeydown = (e) => {
    const ps = POS(), i = ps.indexOf(S.byId[S.live.sid].pos);
    const d = { ArrowLeft: -1, ArrowRight: 1, ArrowUp: -10, ArrowDown: 10 }[e.key];
    if (!d) return;
    e.preventDefault();
    goPos(ps[Math.max(0, Math.min(ps.length - 1, i + d))]);
  };
}

function goPos(pos) {
  const spk = S.byId[S.live.sid].spk;
  select(S.byPos[pos][spk] ?? Object.values(S.byPos[pos])[0]);
}

function renderScatter() {
  $('#scatter').querySelectorAll('.pt').forEach(
    (c) => c.classList.toggle('on', +c.dataset.p === S.byId[S.live.sid]?.pos));
  $('#poscount').textContent = `${POS().length} · spk ${S.byId[S.live.sid]?.spk ?? '-'}`;
}

// y_frac has 0 at the BOTTOM (draw_speaker flips it), so invert for SVG.
const SPK = { 1: [1, 0], 2: [1, .7], 3: [.8, 1], 4: [.6, 1], 5: [.4, 1], 6: [.2, 1], 7: [0, .7], 8: [0, 0] };

function buildSpk() {
  const svg = $('#spk'), W = 126, H = 78, M = 20, bw = W - 2 * M, bh = H - 2 * M;
  svg.setAttribute('viewBox', `0 0 ${W} ${H}`);
  svg.innerHTML = `<rect class="box" x="${M}" y="${M}" width="${bw}" height="${bh}" rx="3"/>` +
    Object.entries(SPK).map(([id, [xf, yf]]) => {
      const cx = M + xf * bw + (xf === 1 ? 12 : xf === 0 ? -12 : 0);
      const cy = M + (1 - yf) * bh + (yf === 0 ? 12 : yf === 1 ? -12 : 0);
      return `<g class="s" data-s="${id}"><circle cx="${cx}" cy="${cy}" r="8"/><text x="${cx}" y="${cy}">${id}</text></g>`;
    }).join('');
  svg.onclick = (e) => { const g = e.target.closest('.s'); if (g) goSpk(+g.dataset.s); svg.focus(); };
  svg.onkeydown = (e) => {
    const d = { ArrowRight: 1, ArrowDown: 1, ArrowLeft: -1, ArrowUp: -1 }[e.key];
    if (!d) return;
    e.preventDefault();
    goSpk(((S.byId[S.live.sid].spk - 1 + d + 8) % 8) + 1);
  };
}

async function goSpk(spk) {
  S.f.spk = new Set([spk]);        // picking a speaker is also filtering to it
  const cur = S.byId[S.live.sid];
  const at = S.byPos[cur.pos];
  if (at?.[spk]) await select(at[spk]);
  applyFilter();
}

function renderSpk() {
  const cur = S.byId[S.live.sid], at = S.byPos[cur?.pos] || {};
  $('#spk').querySelectorAll('.s').forEach((g) => {
    const s = +g.dataset.s;
    g.classList.toggle('on', s === cur?.spk);
    // dimmed when the speaker filter excludes it, or this position has no such sample
    g.classList.toggle('off', (S.f.spk.size && !S.f.spk.has(s)) || !at[s]);
  });
}

function buildGrid() {
  const svg = $('#grid'), R = S.info.rows, C = S.info.cols;
  // L = gutter for the row/col labels; PAD keeps the edge dots off the box border.
  const L = 11, TOP = 18, PAD = 8, BOX = 100, step = (BOX - 2 * PAD) / (C - 1);
  const W = L + BOX, H = TOP + BOX + L;
  svg.setAttribute('viewBox', `0 0 ${W} ${H}`);
  let out = `<g class="avgcell"><rect x="${L}" y="0" width="${BOX}" height="14" rx="3"/>` +
            `<text x="${L + BOX / 2}" y="7.6">AVERAGE</text></g>` +
            `<rect class="box" x="${L}" y="${TOP}" width="${BOX}" height="${BOX}" rx="4"/>`;
  for (let r = 0; r < R; r++) {
    out += `<text class="gl" x="${L - 3}" y="${TOP + PAD + r * step + 1.6}">${r}</text>`;
    for (let c = 0; c < C; c++)
      out += `<circle class="ls" data-i="${r * C + c}" cx="${L + PAD + c * step}" cy="${TOP + PAD + r * step}" r="3.2"/>`;
  }
  for (let c = 0; c < C; c++)
    out += `<text class="gl col" x="${L + PAD + c * step}" y="${TOP + BOX + 7}">${c}</text>`;
  svg.innerHTML = out;
  svg.querySelector('.avgcell').onclick = (e) => { e.stopPropagation(); setLaser('avg'); };
  svg.onclick = (e) => { const t = e.target.closest('.ls'); if (t) setLaser(+t.dataset.i); svg.focus(); };
  svg.onpointermove = (e) => {
    const t = e.target.closest('.ls');
    svg.querySelectorAll('.ls').forEach((x) => x.classList.toggle('hot', x === t));
    const i = t ? +t.dataset.i : null;
    if (i !== S.hoverLaser) { S.hoverLaser = i; previewLaser(i); cursors(); readout(); }
  };
  svg.onpointerleave = () => {
    svg.querySelectorAll('.ls').forEach((x) => x.classList.remove('hot'));
    if (S.hoverLaser != null) { S.hoverLaser = null; previewLaser(null); cursors(); readout(); }
  };
  svg.onkeydown = (e) => {
    const i = S.live.laser === 'avg' ? 55 : +S.live.laser;
    const C = S.info.cols, N = S.info.n_lasers;
    const d = { ArrowLeft: -1, ArrowRight: 1, ArrowUp: -C, ArrowDown: C }[e.key];
    if (!d) return;
    e.preventDefault();
    if (S.live.laser === 'avg') return setLaser(e.key === 'ArrowDown' ? 0 : 'avg');
    const next = i + d;
    setLaser(next < 0 ? 'avg' : Math.min(N - 1, next));
  };
}

function setLaser(i) {
  S.live.laser = i;
  refresh();
}

function renderGrid() {
  const i = S.live.laser;
  $('#grid').querySelectorAll('.ls').forEach(
    (x) => x.classList.toggle('on', i !== 'avg' && +x.dataset.i === +i));
  $('#grid').querySelector('.avgcell').classList.toggle('on', i === 'avg');
  $('#lasernow').textContent = i === 'avg' ? 'avg' : i;
}

/* ***** empty box: 7 repeats per speaker, a drift measurement ***** */
function renderRepeats() {
  const spk = S.byId[S.live.sid].spk;
  const reps = matches().filter((s) => s.empty && s.spk === spk).sort((a, b) => a.pos - b.pos);
  $('#repeats').innerHTML = reps.map((s) =>
    `<button class="rep${s.id === S.live.sid ? ' on' : ''}" data-id="${s.id}">${s.pos}<i>${s.layout}</i></button>`).join('');
  $('#repeats').querySelectorAll('.rep').forEach((b) => (b.onclick = () => select(b.dataset.id)));
}

/* ***** probes ***** */
function hold() {
  const p = {
    ...S.live, fi: S.fi, hzv: hz(S.fi), id: Date.now(),
    hue: hueForFi(S.fi),
    // same frequency held twice -> step the lightness so they stay distinguishable
    lit: 47 + 13 * S.probes.filter((q) => Math.abs(q.hue - hueForFi(S.fi)) < 6).length,
    data: S.probe, mode: S.mode,
    meta: S.byId[S.live.sid],
  };
  S.probes.push(p);
  renderProbes(); paintAll(); multiples();
}

/* The full identity of a probe, in one place, so the live readout and the held rows can
   never disagree about what you are looking at. */
function identity(p) {
  const m = p.meta;
  return {
    sid: shortId(p.sid),
    line1: `pos ${m.pos} · spk ${m.spk}`,
    line2: m.layout,
    line3: `laser ${p.laser} · ${p.ch} · ${fmt(p.hzv)} Hz`,
  };
}

function liveIdentity() {
  const m = S.byId[S.live.sid];
  if (!m) return null;
  const laser = S.hoverLaser != null ? S.hoverLaser : S.live.laser;
  const fi = S.hoverFi != null ? S.hoverFi : S.fi;
  return identity({ ...S.live, laser, meta: m, hzv: hz(fi) });
}

function renderNow() {
  const i = liveIdentity();
  if (!i) return;
  $('#now').classList.toggle('muted', S.muted);
  const prev = S.hoverLaser != null || S.hoverFi != null;
  $('#now').innerHTML =
    `<span class="sw" style="background:${css('--ink')}"></span>
     <span class="id"><b>${i.sid}${prev ? ' <u>preview</u>' : ''}</b>${i.line1}<br>${i.line2}<br>${i.line3}</span>`;
}

function renderProbes() {

  $('#heldh').style.display = S.probes.length ? '' : 'none';
  $('#plist').innerHTML = S.probes.map((p) => {
    const m = p.meta;
    return `<div class="probe card2" data-id="${p.id}" style="--c:${col(p)}">
      <img class="thumb" src="/api/scene/${p.sid}.jpg?v=${S.rv}" alt="">
      <div class="meta">
        <div class="ln1"><span class="sw"></span><b>${shortId(p.sid)}</b>
          <span class="sub">p${m.pos}·s${m.spk}</span>
          <button class="x" data-x="${p.id}">×</button></div>
        <div class="ln2">${m.layout}</div>
        <div class="ln3">L${p.laser} · ${p.ch} · ${fmt(p.hzv)} Hz</div>
      </div>
    </div>`;
  }).join('');
  $('#plist').querySelectorAll('.probe').forEach((el) => {
    const p = S.probes.find((q) => q.id === +el.dataset.id);
    el.onmouseenter = () => { S.hot = p; drawSpec(); drawShifts(); markHot(); probeTicks(); };
    el.onmouseleave = () => { S.hot = null; drawSpec(); drawShifts(); markHot(); probeTicks(); };
  });
  $('#plist').querySelectorAll('.x').forEach((b) => (b.onclick = (e) => {
    e.stopPropagation();
    S.probes = S.probes.filter((q) => q.id !== +b.dataset.x);
    renderProbes(); paintAll(); multiples();
  }));
}

// Scene and heatmap can never overlay, so held probes get thumbnails.
// Thumbnails live inside each probe card now, so there is no separate strip.
function multiples() { heldAudio(); }

/* Who is who, in the panels where several probes are drawn on the same axes. */
function legends() {
  const live = S.muted ? '' :
    `<span data-live="1"><b style="color:${css('--ink')}"></b>` +
    `current — L${S.live.laser} · ${S.live.ch}</span>`;
  const rows = S.probes.map((p) =>
    `<span data-id="${p.id}"><b style="color:${col(p)}"></b>` +
    `${fmt(p.hzv)} Hz · ${shortId(p.sid)} · L${p.laser} · ${p.ch}</span>`).join('');
  const prev = S.preview
    ? `<span><b style="color:${css('--accent')}"></b>preview — L${S.hoverLaser}</span>` : '';
  $('#siglegend').innerHTML = S.probes.length || S.preview ? live + rows + prev : '';
  // The mode overlays probes as offset arrows; the field view shows only the live one.
  $('#modelegend').innerHTML =
    (S.probes.length && S.modeView === 'arrows') ? live + rows : '';
  for (const el of [$('#siglegend'), $('#modelegend')]) {
    el.querySelectorAll('span[data-id]').forEach((sp) => {
      sp.style.cursor = 'pointer';
      sp.onmouseenter = () => {
        S.hot = S.probes.find((q) => q.id === +sp.dataset.id);
        drawSpec(); drawShifts(); markHot(); probeTicks();
      };
      sp.onmouseleave = () => { S.hot = null; drawSpec(); drawShifts(); markHot(); probeTicks(); };
    });
  }
  markHot();
}

function markHot() {
  for (const el of [$('#siglegend'), $('#modelegend')]) {
    el.classList.toggle('dim', !!S.hot);
    el.querySelectorAll('span').forEach((sp) =>
      sp.classList.toggle('hot', !!S.hot && +sp.dataset.id === S.hot.id));
  }
}

/* One player per held probe, so you can compare what they sound like. */
function heldAudio() {
  const el = $('#heldaudio');
  if (!el) return;
  el.innerHTML = S.probes.map((p) => {
    const ch = p.ch === 'avg' ? 'x' : p.ch;
    return `<div class="ha" style="--c:${col(p)}">
      <span class="tag">${shortId(p.sid)} · L${p.laser} · ${ch}</span>
      <audio controls preload="none" src="/api/audio/${p.sid}.wav?ch=${ch}&laser=${p.laser}"></audio>
    </div>`;
  }).join('');
}

/* ***** wiring ***** */
function seg(id, fn) {
  $(id).onclick = (e) => {
    const b = e.target.closest('button');
    if (!b) return;
    $(id).querySelectorAll('button').forEach((x) => x.classList.toggle('on', x === b));
    fn(b.dataset.v);
  };
}

function wire() {
  buildOverlays();
  const strip = $('#heatmean');
  strip.style.cursor = 'pointer';
  strip.title = 'average over all lasers — click to select';
  strip.onclick = () => setLaser('avg');
  strip.onpointerenter = () => { S.hoverLaser = null; previewLaser(null); $('#heatread').textContent = 'average — click to select'; };

  const hp = $('#heat').parentElement;
  freqAxis(hp, true);                                // heatmap: freq AND laser
  hp.tabIndex = 0;
  hp.addEventListener('click', () => hp.focus());
  hp.addEventListener('keydown', (e) => {
    const N = S.info.n_lasers, n = S.d.freqs.length;
    const step = e.shiftKey ? 10 : 1;
    if (e.key === 'ArrowUp' || e.key === 'ArrowDown') {
      e.preventDefault();
      if (S.live.laser === 'avg') {
        if (e.key === 'ArrowDown') setLaser(0);       // avg sits just above laser 0
        return;
      }
      const next = +S.live.laser + (e.key === 'ArrowDown' ? step : -step);
      setLaser(next < 0 ? 'avg' : Math.min(N - 1, next));
    } else if (e.key === 'ArrowLeft' || e.key === 'ArrowRight') {
      e.preventDefault();
      const d = e.key === 'ArrowRight' ? 1 : -1;
      if ((S.probe?.peaks || []).includes(S.fi)) return stepPeak(d);
      setFi(Math.max(0, Math.min(n - 1, S.fi + d * (e.shiftKey ? 25 : 1))));
    }
  });
  document.querySelectorAll('#p1, #p2').forEach((el) => freqAxis(el.parentElement, false));

  const sid = $('#sid'), slist = $('#sidlist');
  sid.oninput = () => { renderSidList(sid.value.trim()); slist.hidden = false; };
  sid.onfocus = () => { renderSidList(sid.value.trim()); slist.hidden = false; };
  sid.onblur = () => setTimeout(() => (slist.hidden = true), 120);
  sid.onkeydown = (e) => {
    if (e.key === 'Enter') {
      const q = sid.value.trim();
      const hit = S.byId[q] ? q : S.all.find((s) => shortId(s.id) === q)?.id;
      if (hit) { pickSample(hit); slist.hidden = true; sid.blur(); }
    } else if (e.key === 'Escape') { slist.hidden = true; sid.blur(); }
  };
  $('#summary').onclick = () => $('#pickers').classList.toggle('hid');

  seg('#ch', (v) => { S.live.ch = v; refresh(); });
  seg('#specmode', (v) => { S.specMode = v; drawSpec(); buildPeaks(); cursors(); });
  seg('#speclog', (v) => { S.log.spec = +v; drawSpec(); buildPeaks(); cursors(); });
  seg('#heatlog', (v) => { S.log.heat = +v; refresh(); multiples(); });
  seg('#kind', (v) => { S.kind = v; refresh(); });
  seg('#modeview', (v) => { S.modeView = v; drawMode(); legends(); });

  $('#hold').onclick = hold;
  $('#clear').onclick = () => { S.probes = []; renderProbes(); paintAll(); multiples(); };
  $('#empty').onclick = () => {
    S.empty = !S.empty;
    $('#empty').classList.toggle('on', S.empty);
    $('#scatter').hidden = S.empty; $('#repeats').hidden = !S.empty;
    if (S.empty) { renderRepeats(); select(S.all.find((s) => s.empty && s.spk === S.byId[S.live.sid].spk).id); }
  };

  $('#asize').oninput = (e) => { S.asize = +e.target.value; drawMode(); };
  $('#play').onclick = () => {
    S.anim = !S.anim;
    $('#play').classList.toggle('on', !!S.anim);
    $('#play').textContent = S.anim ? '❚❚' : '▶';
    const tick = () => { if (!S.anim) return; drawMode(); requestAnimationFrame(tick); };
    tick();
  };

  addEventListener('keydown', (e) => {
    if (e.target.tagName === 'INPUT') return;
    // Enter/Escape are global -- they must work while a panel has focus. Only the
    // arrow keys defer to a focused panel, which uses them for its own navigation.
    if (e.key === 'Enter') { e.preventDefault(); return hold(); }
    if (e.key === 'Escape') {
      e.preventDefault();
      S.muted = !S.muted;
      return paintAll();
    }
    if (e.target.closest('.plot, svg')) return;   // panel-local keys win
    const ps = matches();
    const i = ps.findIndex((s) => s.id === S.live.sid);
    if (e.key === 'p') return hold();
    // Shift+arrows always walk the detected peaks.
    if (e.shiftKey && (e.key === 'ArrowLeft' || e.key === 'ArrowRight')) {
      e.preventDefault();
      return stepPeak(e.key === 'ArrowRight' ? 1 : -1);
    }
    if (e.key === 'x' || e.key === 'y' || e.key === 'a') {
      const v = e.key === 'a' ? 'avg' : e.key;
      S.live.ch = v;
      $('#ch').querySelectorAll('button').forEach((b) => b.classList.toggle('on', b.dataset.v === v));
      return refresh();
    }
    if (e.key === 'ArrowLeft' && i > 0) select(ps[i - 1].id);
    if (e.key === 'ArrowRight' && i < ps.length - 1) select(ps[i + 1].id);
  });

  addEventListener('resize', () => paintAll());
}
