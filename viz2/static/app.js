'use strict';
const $ = (s) => document.querySelector(s);
const api = (u) => fetch(u).then((r) => r.json());

const S = {
  all: [], info: null, byId: {}, byPos: {},
  live: { sid: null, ch: 'avg', laser: 'avg', fi: null },
  d: null, probe: null, mode: null, rng: {},
  probes: [], hues: {}, hot: null,
  fi: null, hoverFi: null,
  log: { spec: 1 },
  specMode: 'magphase', fieldbg: false, kind: 'clean',
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
/* Three encodings, each matched to what it carries:
     position  -> HUE        categorical; a scene is a thing, not a quantity
     frequency -> LIGHTNESS  ordered data on an ordered channel: low = light, high = dark
     channel   -> DASH       only x / y / avg, so a dash pattern stays readable
   Laser is left to the legend: too many values for any visual channel. */
function hueForPos(pos) {
  return (S.hues[pos] ??= (205 + Object.keys(S.hues).length * 137.508) % 360);
}
/* Frequency used to map to LIGHTNESS across the band, which failed twice over: peaks
   crowd the low end (127/185/251/325 Hz spanned only 6 lightness points) and distinct
   peaks collided outright -- 251 and 272 Hz both landed on 55%. Lightness is also the
   weakest channel here, squeezed between a white ground and the black live trace.

   Instead, fan the HUE around the position's base color. Measured in CIE Lab, the worst
   pair separates 2.6-5x better than the lightness ramp at every count. Position still owns
   a hue NEIGHBOURHOOD, so probes from one sample stay recognisably related.

   The fan is indexed by the order a probe was pinned within its sample -- not by which
   peak it is -- so a frequency that is not a peak at all gets an equally distinct color. */
const FAN = 70;        // narrower than the 137.5 deg between positions, so fans never meet

/* Hue AND lightness together. Neither alone is enough: a fan wide enough to separate 5
   probes by hue would spill into the neighbouring position's hue (measured: two positions
   x 3 probes collided at dE 1.2, below the just-noticeable threshold), while lightness
   alone tops out around dE 14. A 70 deg fan plus a 62->32 lightness ramp keeps the worst
   pair anywhere in the workbench at dE 22.6, and two probes of one sample at 33. */
function shadeFor(hue, i, n) {
  if (n <= 1) return { hue, lit: 46 };
  const t = i / (n - 1);
  return { hue: ((hue - FAN / 2 + t * FAN) % 360 + 360) % 360, lit: Math.round(62 - t * 30) };
}

/* Every probe of one position is recolored together, since each one's slot depends on
   how many siblings it has. Called whenever the set changes. */
function reshade() {
  const by = {};
  for (const p of S.probes) (by[p.pos] ??= []).push(p);
  for (const [pos, list] of Object.entries(by)) {
    const base = hueForPos(+pos);
    list.forEach((p, i) => Object.assign(p, shadeFor(base, i, list.length)));
  }
}

const DASH = { avg: [], x: [], y: [7, 6], both: [] };
// Audio inverts one real channel, so neither avg (a mean of complex spectra) nor both
// can be played: fall back to x. The heatmap paints one plane, and avg IS a plane, so
// only "both" needs replacing there.
const audioCh = (c) => (c === 'x' || c === 'y' ? c : 'x');
const planeCh = (c) => (c === 'both' ? 'x' : c);
const col = (p, a = 1) => `hsl(${p.hue} 66% ${p.lit ?? 46}% / ${a})`;

/* The mask overlay is painted server-side, so the color has to travel as hex. */
function hex(h, s, l) {
  s /= 100; l /= 100;
  const k = (n) => (n + h / 30) % 12;
  const a = s * Math.min(l, 1 - l);
  const f = (n) => Math.round(255 * (l - a * Math.max(-1, Math.min(k(n) - 3, 9 - k(n), 1))));
  return [f(0), f(8), f(4)].map((v) => v.toString(16).padStart(2, '0')).join('');
}
const probeHex = (p) => hex(p.hue, 68, p.lit ?? 47);
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
  // A new sample resonates at its OWN frequencies, so carrying the previous bin over
  // usually lands on noise. Clearing it makes refresh() snap to this sample's strongest
  // peak -- the most informative place to start, and the mode there is worth looking at.
  S.fi = null;
  await refresh();
}

async function refresh() {
  const { sid, ch } = S.live;
  // "all" is a rendering of every laser, not a laser: the curves and the audio behind it
  // stay the average, so the peak list and the readout still mean something.
  const laser = S.live.laser === 'all' ? 'avg' : S.live.laser;
  $('#scene').src = `/api/scene/${sid}.jpg?v=${S.rv}`;
  $('#smask').src = `/api/mask/${sid}.png?v=${S.rv}`;
  $('#audio').src = `/api/audio/${sid}.wav?ch=${audioCh(ch)}&laser=${laser}`;
  // "both" is x and y in one axes. The endpoint serves one channel, so ask twice and let
  // the dash pattern -- the channel's own visual channel already -- tell them apart.
  const q = (c) => api(`/api/probe/${sid}?ch=${c}&laser=${laser}&kind=${S.kind}`);
  if (ch === 'both') {
    const [x, y] = await Promise.all([q('x'), q('y')]);
    S.probe = x; S.probeY = y;
  } else {
    S.probe = await q(ch); S.probeY = null;
  }
  if (S.fi == null) {
    S.fi = strongestPeak() ?? Math.floor(S.d.freqs.length / 3);
    S.peakMode = true;               // so the arrows walk peaks straight away
  }
  await loadMode();
  paintAll();
}

async function loadMode() {
  S.mode = await api(`/api/mode/${S.live.sid}?fi=${S.fi}`);
}

function paintAll() {
  const s = S.byId[S.live.sid];
  $('#summary').textContent =
    `${shortId(s.id)}  pos ${s.pos}  spk ${s.spk}  ${s.layout}  L${S.live.laser}  ${S.live.ch}`;
  buildPeaks(); peakList(); allMasks(); drawSpec(); drawShifts(); drawMode(); cursors(); axes(); legends();
  readout();
  renderScatter(); renderSpk(); renderGrid(); renderProbes();
}

/* ***** curves: live probe + pinned probes in one axes ***** */
function series() {
  const pinned = S.probes.flatMap((p) => {
    const c = col(p, S.hot && S.hot !== p ? 0.18 : 0.9);
    const w = p.id === S.flash ? 2.6 : 1.6;
    // A "both" probe is one probe drawn as two curves: same colour, x solid / y dashed.
    return p.dataY
      ? [{ p, d: p.data, c, w, dash: DASH.x }, { p, d: p.dataY, c, w, dash: DASH.y }]
      : [{ p, d: p.data, c, w, dash: p.dash }];
  });
  // A just-pinned probe is drawn last for a moment so you see it appear; otherwise the
  // live trace stays in front, because that is the one you are steering.
  const front = pinned.filter((s) => s.p.id === S.flash);
  const back = pinned.filter((s) => s.p.id !== S.flash);
  // In "both" the live trace is two curves of one colour, separated by dash: x SOLID,
  // y dashed with a wide gap. One solid line reads faster than two dashed ones, and the
  // gap has to be wide or the two textures blur together at this line density.
  const live = S.muted ? []
    : [{ p: null, d: S.probe, c: css('--ink'), w: 1.7, dim: !!S.preview, dash: DASH.x },
       ...(S.probeY ? [{ p: null, d: S.probeY, c: css('--ink'), w: 1.7, dim: !!S.preview, dash: DASH.y }] : [])];
  return [
    ...back,
    ...live,
    ...front,
    ...(S.preview && !S.muted ? [{ p: null, d: S.preview, c: css('--accent'), w: 1.7 }] : []),
  ];
}

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
  S.rng[cv.id] = { lo, hi };                  // so the y axis can label the same scale
  if (cv.id === 'p1') S.p1 = { lo, hi, h };   // so peak markers can sit ON the curve
  for (const s of ss) {
    const y = s.d[key];
    g.beginPath();
    for (let i = 0; i < y.length; i++) {
      const X = (i / (y.length - 1)) * w, Y = h - (y[i] - lo) * k;
      i ? g.lineTo(X, Y) : g.moveTo(X, Y);
    }
    g.globalAlpha = s.dim ? 0.28 : 1;
    g.setLineDash(s.dash || []);
    g.strokeStyle = s.c; g.lineWidth = s.w; g.stroke();
    g.setLineDash([]);
    g.globalAlpha = 1;
  }
}

// The same two ramps as viz2/render.py, so the colorbars match the PNGs exactly.
const SEQ = ['#ffffff', '#cde2fb', '#9ec5f4', '#6da7ec', '#3987e5', '#256abf', '#184f95', '#0d366b'];
const DIV = ['#8c3b12', '#d98635', '#f7dcc0', '#f5f5f3', '#cfe0f2', '#5595d4', '#124b8e'];

/* "All lasers" turns each plot into a lasers x frequency image. The five quantities have
   different units and two different palettes, so each carries its own colorbar. */
const HEATQ = () => (S.specMode === 'magphase'
  ? [S.log.spec ? 'logmag' : 'mag', 'phase'] : ['re', 'im']);

async function drawHeat() {
  const all = S.live.laser === 'all';
  document.querySelectorAll('#p1, #p2, #p3').forEach(
    (c) => c.parentElement.classList.toggle('all', all));
  if (!all) return;
  const { sid, ch } = S.live;
  const qs = [...HEATQ(), 'shifts'];
  qs.forEach((q, i) => {
    $(`#h${i + 1}`).src =
      `/api/heat/${sid}.png?ch=${planeCh(ch)}&q=${q}&kind=${S.kind}&v=${S.rv}`;
  });
  const rs = await Promise.all(qs.map((q) =>
    api(`/api/heatrange/${sid}?ch=${planeCh(ch)}&q=${q}&kind=${S.kind}`)));
  rs.forEach((r, i) => {
    const stops = r.lut === 'seq' ? SEQ : DIV;
    $(`#cb${i + 1}`).innerHTML =
      `${fmtTick(r.hi)}<b style="background:linear-gradient(0deg,${stops.join(',')})"></b>${fmtTick(r.lo)}`;
  });
}

function drawSpec() {
  const a = S.specMode === 'magphase' ? (S.log.spec ? 'logmag' : 'mag') : 're';
  const b = S.specMode === 'magphase' ? 'phase' : 'im';
  multi($('#p1'), a); multi($('#p2'), b);
  yAxis('#p1ov', 'p1'); yAxis('#p2ov', 'p2');
  drawHeat();
}

const drawShifts = () => { multi($('#p3'), 'shifts'); yAxis('#p3ov', 'p3'); };

/* Numeric ticks on the y axis. The range comes from whatever multi() just drew, so the
   numbers track the overlaid probes' shared scale and every toggle (log, re+im, raw) at
   once. Drawn into the existing overlay, so they cost no extra element. */
function yAxis(sel, id) {
  const svg = $(sel), r = S.rng[id];
  if (!svg) return;
  // All-lasers view: the y axis is the laser index, so label THAT instead of the range
  // of a curve that is not being drawn.
  if (S.live.laser === 'all') return laserTicks(svg);
  const old = svg.querySelector('.yt');
  if (old && old.dataset.mode === 'laser') old.remove();
  if (!r) return;
  let g = svg.querySelector('.yt');
  if (!g) { g = document.createElementNS('http://www.w3.org/2000/svg', 'g');
            g.setAttribute('class', 'yt'); svg.prepend(g); }
  g.dataset.mode = 'val';
  const { lo, hi } = r;
  g.innerHTML = ticks(lo, hi, 4).map((v) => {
    const y = (1 - (v - lo) / ((hi - lo) || 1)) * 100;
    if (y < 1 || y > 99) return '';           // skip labels that would clip the frame
    return `<line class="ygrid" x1="0" x2="100%" y1="${y}%" y2="${y}%"/>` +
           `<text class="ynum" x="-5" y="${y}%" dy="3">${fmtTick(v)}</text>`;
  }).join('');
}

/* Rows are lasers in the all-lasers view: label every 20th so the axis stays readable. */
function laserTicks(svg) {
  let g = svg.querySelector('.yt');
  if (!g) { g = document.createElementNS('http://www.w3.org/2000/svg', 'g');
            g.setAttribute('class', 'yt'); svg.prepend(g); }
  g.dataset.mode = 'laser';
  const L = S.info.n_lasers;
  let out = '';
  for (let i = 0; i < L; i += 20)
    out += `<text class="ynum" x="-5" y="${((i + 0.5) / L) * 100}%" dy="3">${i}</text>`;
  g.innerHTML = out;
}

// Compact tick text: no long decimal tails, and exponent form only when truly small/large.
function fmtTick(v) {
  const a = Math.abs(v);
  if (a === 0) return '0';
  if (a >= 1e4 || a < 1e-3) return v.toExponential(0);
  return String(+v.toFixed(a < 1 ? 2 : a < 10 ? 1 : 0));
}

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
  fieldGrid();
  modeAxes();
  const cv = $('#mode'), [g, w, h] = fit(cv);
  if (!S.mode) return;
  const R = S.info.rows, C = S.info.cols, cw = w / C, chh = h / R;
  const withMode = S.probes.filter((p) => p.mode);
  const sets = [
    ...withMode.filter((p) => p.id !== S.flash).map((p) => ({ m: p.mode, c: col(p, 0.9) })),
    ...(S.muted && S.probes.length ? [] : [{ m: S.mode, c: css('--ink') }]),
    ...withMode.filter((p) => p.id === S.flash).map((p) => ({ m: p.mode, c: col(p, 0.9) })),
  ];

  // The big plot always overlays every probe. Arrows from different probes share a cell
  // origin, so color alone would hide one under another: fan the tails around the centre.
  const k = (Math.min(cw, chh) * 0.42 * S.asize) / (maxOf(sets) || 1);
  arrows(g, sets, w, h, R, C, cw, chh, k, sets.length > 1 ? Math.min(cw, chh) * 0.15 : 0);
}

const maxOf = (sets) => {
  let max = 0;
  for (const s of sets) for (const row of s.m.u.keys()) for (const c of s.m.u[row].keys())
    max = Math.max(max, Math.hypot(s.m.u[row][c], s.m.v[row][c]));
  return max;
};

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

// Direction -> hue, magnitude -> lightness. (spatial_derivatives_to_hsv, reimplemented.)
const fcol = (u, v, max) =>
  `hsl(${(Math.atan2(v, u) * 180) / Math.PI + 180} 70% ${18 + 52 * (Math.hypot(u, v) / (max || 1))}%)`;

const peak = (m) => {
  let max = 0;
  for (const row of m.u.keys()) for (const c of m.u[row].keys())
    max = Math.max(max, Math.hypot(m.u[row][c], m.v[row][c]));
  return max;
};

function field(g, w, h, m, R, C, max = peak(m)) {
  const cw = w / C, chh = h / R;
  for (let r = 0; r < R; r++) for (let c = 0; c < C; c++) {
    g.fillStyle = fcol(m.u[r][c], m.v[r][c], max);
    g.fillRect(c * cw, r * chh, cw + 1, chh + 1);
  }
}

/* One tile per probe (plus the live one): the SAME mode, drawn alone. The big plot above
   overlays them to compare shapes; here each is isolated so a single probe can be read
   without the others crossing it. Arrows always; the field background is opt-in, since a
   dense fill is only wanted when you are reading direction rather than magnitude.
   Every tile shares the big plot's scale and animation clock, so play moves all of them
   together and arrow lengths stay comparable across tiles. */
function fieldGrid() {
  const box = $('#fgrid');
  if (!box) return;
  const lm = S.byId[S.live.sid];
  const cap = (m, hzv) => `<b>${fmt(hzv)} Hz</b><i>${shortId(m.id)}  pos ${m.pos}  spk ${m.spk}</i>`;
  const sets = [
    ...(S.mode && lm ? [{ m: S.mode, c: css('--ink'), id: 0, t: cap(lm, hz(S.fi)) }] : []),
    ...S.probes.filter((p) => p.mode)
      .map((p) => ({ m: p.mode, c: col(p), id: p.id, t: cap(p.meta, p.hzv) })),
  ];
  if (sets.length < 2) { box.innerHTML = ''; return; }

  const sig = sets.map((s) => s.id).join(',') + '|' + sets.map((s) => s.t).join('|');
  if (box.dataset.sig !== sig) {          // rebuild only when the SET changes, not per frame
    box.dataset.sig = sig;
    box.innerHTML = sets.map((s) => `<figure data-id="${s.id}">` +
      `<canvas width="124" height="124"></canvas>` +
      `<figcaption style="border-left-color:${s.c}">${s.t}</figcaption></figure>`).join('');
  }
  box.querySelectorAll('figure').forEach((f) =>
    f.classList.toggle('hot', !!S.hot && +f.dataset.id === S.hot.id));

  const R = S.info.rows, C = S.info.cols;
  /* Each tile is normalised to ITS OWN peak, so every one fills the cell and its shape is
     readable even for a weakly excited mode. Relative magnitude between probes is what the
     overlaid plot above is for -- on a shared scale here, a weak mode would collapse into
     invisible arrows and the shape, the only thing these tiles are for, would be lost. */
  box.querySelectorAll('figure canvas').forEach((cv, i) => {
    const [g, w, h] = fit(cv), cw = w / C, chh = h / R;
    const max = maxOf([sets[i]]);
    if (S.fieldbg) field(g, w, h, sets[i].m, R, C, max);
    const k = (Math.min(cw, chh) * 0.42 * S.asize) / (max || 1);
    arrows(g, [{ m: sets[i].m, c: S.fieldbg ? '#fff' : sets[i].c }], w, h, R, C, cw, chh, k, 0);
  });
}

/* The key is the color map itself, drawn as the 2-D plane it is: hue sweeps the full
   direction circle across x, and lightness runs the amplitude ramp down y -- the same
   two expressions field() uses, so the key cannot drift from the render. */
function wheel(cv) {
  const g = cv.getContext('2d'), { width: w, height: h } = cv;
  for (let x = 0; x < w; x++) for (let y = 0; y < h; y++) {
    g.fillStyle = `hsl(${(x / w) * 360} 70% ${18 + 52 * (1 - y / h)}%)`;
    g.fillRect(x, y, 1, 1);
  }
}

/* ***** crosshair: SVG only, so hover never repaints a canvas *****
   The nodes are built once and then only moved. Rebuilding innerHTML on pointermove
   would destroy the peak dots out from under a click that is already in flight. */
const OVS = ['#p1ov', '#p2ov'];

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
function yLabel(sel, text) {
  const el = $(sel);
  if (el) el.textContent = text;
}

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
    out += t.map((v) => {
      const x = ((v - lo) / (hi - lo)) * 100;
      return `<text class="tk" x="${x}%" y="100%" dy="11">${v}</text>`;
    }).join('') + `<text class="unit" x="50%" y="100%" dy="23">${label}</text>`;
  }
  g.innerHTML = out;
}

function buildOverlays() {
  for (const id of OVS) {
    $(id).innerHTML = '<line class="cur hov" style="opacity:.4" y1="0" y2="100%"/>' +
                      '<line class="cur sel" y1="0" y2="100%"/>' +
                      (id === '#p1ov' ? '<g class="pks"></g>' : '');
  }
}

// Every peak is labelled; alternating sides keeps close pairs from overlapping.

// Markers ride the magnitude curve itself, using the same y-mapping multi() drew with.
function peakY(i) {
  const m = S.p1;
  if (!m || !S.probe) return 10;
  const mag = S.specMode === 'magphase'
    ? (S.log.spec ? S.probe.logmag : S.probe.mag) : S.probe.re;
  return m.h - ((mag[i] - m.lo) / ((m.hi - m.lo) || 1)) * m.h;
}

/* The detected resonances, listed. Clicking one is the same as clicking its marker. */
function peakList() {
  const el = $('#pklist');
  if (!el) return;
  const pk = S.probe?.peaks || [];
  $('#npk').textContent = pk.length || '';
  // Strongest first: when hunting resonances, rank is the useful order. dB is shown so
  // "how much stronger" is answerable, not just "which is stronger".
  const mag = S.probe ? S.probe.mag : null;
  const lg = S.log.spec;
  const val = (i) => {
    const m = mag?.[i] ?? 0;
    return lg ? `${(20 * Math.log10(m + 1e-8)).toFixed(0)} dB`
              : (m >= 10 ? m.toFixed(0) : m >= 1 ? m.toFixed(1) : m.toFixed(2));
  };
  const sorted = mag ? [...pk].sort((x, y) => mag[y] - mag[x]) : pk;
  el.innerHTML = sorted.map((i, r) =>
    `<button class="pkrow${i === S.fi ? ' on' : ''}" data-fi="${i}" title="rank ${r + 1}">` +
    `<span class="hzv">${fmt(hz(i))} Hz</span><span class="dbv">${val(i)}</span></button>`
  ).join('');
  el.querySelectorAll('.pkrow').forEach((b) => (b.onclick = () => {
    S.peakMode = true;
    setFi(+b.dataset.fi);
    el.focus();
  }));
  el.tabIndex = 0;
  el.onfocus = () => { S.peakMode = true; el.classList.add('armed'); };
  el.onblur = () => el.classList.remove('armed');
  el.onpointerenter = () => { S.peakMode = true; el.classList.add('armed'); };
  el.onpointerleave = () => { if (document.activeElement !== el) el.classList.remove('armed'); };
}

function buildPeaks() {
  const g = $('#p1ov').querySelector('.pks');
  if (!g) return;
  // The markers ride the magnitude curve; with an image in its place they have no y.
  const pk = S.live.laser === 'all' ? [] : (S.probe?.peaks || []);
  const n = S.d ? S.d.freqs.length : 1;
  let lastX = -99, up = true;
  g.innerHTML = pk.map((i) => {
    const x = (i / (n - 1)) * 100;
    up = (x - lastX) < 6 ? !up : true;       // too close to the previous label -> flip
    lastX = x;
    return `<circle class="pk" r="3.4" data-fi="${i}"/>` +
           `<text class="pklab" data-fi="${i}" data-up="${up ? 1 : 0}">${fmt(hz(i))}</text>`;
  }).join('');
  g.querySelectorAll('.pk').forEach((c) => (c.onclick = (e) => {
    e.stopPropagation();
    S.peakMode = true;
    setFi(+c.dataset.fi);
  }));
}

function cursors() {
  probeTicks();
  const n = S.d ? S.d.freqs.length : 1;
  const at = (i) => `${(i / (n - 1)) * 100}%`;
  for (const id of OVS) {
    const svg = $(id);
    const sel = svg.querySelector('.sel'), hov = svg.querySelector('.hov');
    sel.setAttribute('x1', at(S.fi)); sel.setAttribute('x2', at(S.fi));
    const show = S.hoverFi != null && S.hoverFi !== S.fi && S.live.laser !== 'all';
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
    tx.setAttribute('dy', tx.dataset.up === '1' ? -8 : 15);
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
  S.peakMode = true;
  const at = pk.indexOf(S.fi);
  const next = at === -1
    ? (d > 0 ? pk.find((i) => i > S.fi) ?? pk[0] : [...pk].reverse().find((i) => i < S.fi) ?? pk.at(-1))
    : pk[Math.max(0, Math.min(pk.length - 1, at + d))];
  setFi(next);
}

// The tallest detected resonance, or null when nothing was detected.
function strongestPeak() {
  const pk = S.probe?.peaks || [], mag = S.probe?.mag;
  if (!pk.length || !mag) return null;
  return pk.reduce((a, b) => (mag[b] > mag[a] ? b : a));
}

const PK_COLS = 2;                     // must match .pklist grid-template-columns

function ranked() {
  const pk = S.probe?.peaks || [], mag = S.probe?.mag;
  return (pk.length && mag) ? [...pk].sort((x, y) => mag[y] - mag[x]) : pk;
}

/* Move within the ranked grid exactly as it is laid out: left/right to the neighbouring
   cell (which wraps between columns, so left from the right column lands on the left
   column of the SAME row), up/down a whole row. */
function stepGrid(dx, dy) {
  const r = ranked();
  if (!r.length) return;
  S.peakMode = true;
  const at = r.indexOf(S.fi);
  if (at === -1) return setFi(r[0]);
  const next = at + dx + dy * PK_COLS;
  if (next < 0 || next >= r.length) return;      // stay put at the edges
  setFi(r[next]);
}

/* Each pinned probe sits at one frequency, but nothing on the spectrum showed where.
   A tick per probe, in its color, makes position -- the most precise visual channel --
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
      const dash = p.dash?.length ? p.dash.join(' ') : '5 3';
      return `<line class="ptick" x1="${x}%" x2="${x}%" y1="0" y2="100%" ` +
             `stroke-dasharray="${dash}" stroke="${col(p, dim ? 0.2 : 0.9)}"/>`;
    }).join('');
  }
}

function axes() {
  if (!S.d) return;
  const f = S.d.freqs, lo = f[0], hi = f[f.length - 1];
  const HZ = 'frequency (Hz)';
  xAxis($('#p1ov'), lo, hi, HZ, false);       // top of the spectrum stack: grid only
  xAxis($('#p2ov'), lo, hi, HZ, true);        // bottom carries the numbers
  if (S.probe) xAxis($('#p3ov'), 0, S.probe.dur, 'time (s)', true);

  const mp = S.specMode === 'magphase';
  // In all-lasers view every plot's y axis is the laser index, not the quantity.
  const all = S.live.laser === 'all';
  yLabel('#y1', all ? 'laser' : mp ? (S.log.spec ? 'log |FFT|' : '|FFT|') : 'real');
  yLabel('#y2', all ? 'laser' : mp ? 'phase (rad)' : 'imag');
  yLabel('#y3', all ? 'laser' : 'shift (px)');
}

function freqAxis(el) {
  el.style.cursor = 'crosshair';
  el.addEventListener('pointermove', (e) => {
    const b = el.getBoundingClientRect();
    const n = S.d.freqs.length;
    S.hoverFi = Math.max(0, Math.min(n - 1, Math.round(((e.clientX - b.left) / b.width) * (n - 1))));
    cursors();
    readout();
    previewMode(S.hoverFi);
  });
  el.addEventListener('pointerleave', () => {
    S.hoverFi = null;
    if (S.modePreview) previewMode(null);
    cursors(); readout();
  });
  el.addEventListener('click', () => {
    if (S.hoverFi == null) return;
    // Landing on a detected peak keeps peak-stepping armed; anywhere else releases it.
    S.peakMode = (S.probe?.peaks || []).includes(S.hoverFi);
    setFi(S.hoverFi);
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
  const laser = `Laser <b>${l}</b>`;
  const chan = `Chn <b>${S.live.ch}</b>`;
  const freq = `Freq <b>${fmt(hz(f))} Hz</b>`;
  const title = (...parts) => `${scene}: ${parts.join(', ')}`;

  $('#sigtitle').innerHTML = title(laser, chan, freq);
  // Audio reconstructs one laser's single channel -- an average of complex spectra does
  // not invert -- and spans the whole band, so no frequency.
  const ach = audioCh(S.live.ch);
  $('#audtitle').innerHTML = title(
    `Laser <b>${S.live.laser === 'avg' ? 'avg' : S.live.laser}</b>`,
    `Chn <b>${ach}</b>`);
  // The mode is every laser at one frequency, using BOTH channels (a 2-D displacement
  // field), so neither laser nor channel is a field here -- only the frequency.
  $('#modetitle').innerHTML = title(freq);

  renderNow();
}

/* Hovering a laser in the grid previews its spectrum without committing to it, so you
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
// A stable hue per value, so a chip keeps its color no matter what else is selected.
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
    `<b>${shortId(s.id)}</b><i>pos ${s.pos}  spk ${s.spk}  ${s.layout}</i></button>`).join('') +
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
    const d = { ArrowLeft: [-1, 0], ArrowRight: [1, 0], ArrowUp: [0, -1], ArrowDown: [0, 1] }[e.key];
    if (!d) return;
    e.preventDefault();
    const p = neighbour(s, d);
    if (p != null) goPos(p);
  };
}

/* Positions are captured as a SERPENTINE raster -- each row sweeps back the way the last
   one came -- and the rows are not a fixed width, so stepping the id list by a constant
   stride lands on the far side of the box (up from the right edge used to jump to the
   left edge). Move geometrically instead.

   The catch: this is a near-continuous cloud, not a grid. Neighbouring points sit ~16px
   apart while a real step along a sweep is ~68 (row) / ~97 (col), so plain "nearest point
   ahead" just crawls onto a near-duplicate. Require a step of at least half the measured
   pitch, then among the rest take the nearest, weighting sideways drift so you stay in
   lane. Both numbers are measured from the data, never hardcoded. */
function pitch() {
  if (pitch._k === POS().length) return pitch._v;
  const s = scale(), n = s.pts.length;
  const dr = [], dc = [];
  for (let i = 1; i < n; i++) {                   // consecutive ids follow the sweep
    dr.push(Math.abs(s.y(s.pts[i][0]) - s.y(s.pts[i - 1][0])));
    dc.push(Math.abs(s.x(s.pts[i][1]) - s.x(s.pts[i - 1][1])));
  }
  const med = (a) => a.sort((x, y) => x - y)[a.length >> 1] || 0;
  pitch._k = n;
  return (pitch._v = { x: med(dc) * 0.5, y: med(dr) * 0.5 });
}

function neighbour(s, [dx, dy]) {
  const ps = POS(), cur = S.byId[S.live.sid]?.pos;
  const at = ps.indexOf(cur);
  if (at === -1) return ps[0];
  const x0 = s.x(s.pts[at][1]), y0 = s.y(s.pts[at][0]);
  const min = dx ? pitch().x : pitch().y;
  let best = null, bd = Infinity;
  ps.forEach((p, i) => {
    if (p === cur) return;
    const ax = s.x(s.pts[i][1]) - x0, ay = s.y(s.pts[i][0]) - y0;
    const along = ax * dx + ay * dy;              // travel in the pressed direction
    const off = Math.abs(ax * dy + ay * dx);      // drift across it
    if (along < min) return;                      // a real step, not a near-duplicate
    const cost = along + 6 * off;                 // stay in lane
    if (cost < bd) { bd = cost; best = p; }
  });
  return best;                                    // null at the edge: stay put
}

function goPos(pos) {
  const spk = S.byId[S.live.sid].spk;
  select(S.byPos[pos][spk] ?? Object.values(S.byPos[pos])[0]);
}

function renderScatter() {
  $('#scatter').querySelectorAll('.pt').forEach(
    (c) => c.classList.toggle('on', +c.dataset.p === S.byId[S.live.sid]?.pos));
  $('#poscount').textContent = `${POS().length}  spk ${S.byId[S.live.sid]?.spk ?? '-'}`;
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
  const half = (BOX - 2) / 2;
  let out = `<g class="avgcell"><rect x="${L}" y="0" width="${half}" height="14" rx="3"/>` +
            `<text x="${L + half / 2}" y="7.6">AVERAGE</text></g>` +
            `<g class="allcell"><rect x="${L + half + 2}" y="0" width="${half}" height="14" rx="3"/>` +
            `<text x="${L + half + 2 + half / 2}" y="7.6">ALL</text></g>` +
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
  svg.querySelector('.allcell').onclick = (e) => { e.stopPropagation(); setLaser('all'); };
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
    const cur = S.live.laser;
    const i = (cur === 'avg' || cur === 'all') ? 55 : +cur;
    const C = S.info.cols, N = S.info.n_lasers;
    const d = { ArrowLeft: -1, ArrowRight: 1, ArrowUp: -C, ArrowDown: C }[e.key];
    if (!d) return;
    e.preventDefault();
    if (cur === 'avg' || cur === 'all') return setLaser(e.key === 'ArrowDown' ? 0 : cur);
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
  $('#grid').querySelector('.allcell').classList.toggle('on', i === 'all');
  $('#lasernow').textContent = i;
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
function pin() {
  const p = {
    ...S.live, fi: S.fi, hzv: hz(S.fi), id: Date.now(),
    pos: S.byId[S.live.sid].pos,          // the fan is grouped by position
    dash: DASH[S.live.ch] || [],
    data: S.probe, dataY: S.probeY, mode: S.mode,
    meta: S.byId[S.live.sid],
  };
  S.probes.push(p);
  reshade();
  S.flash = p.id;
  renderProbes(); paintAll(); multiples();
  clearTimeout(pin._t);
  pin._t = setTimeout(() => { S.flash = null; paintAll(); multiples(); }, 1400);
}

/* The full identity of a probe, in one place, so the live readout and the pinned rows can
   never disagree about what you are looking at. */
function identity(p) {
  const m = p.meta;
  return {
    sid: shortId(p.sid),
    line1: `pos ${m.pos}  spk ${m.spk}`,
    line2: m.layout,
    // "L42 y  1000 Hz" -- the widest this can get still fits the column without wrapping,
    // which is what keeps the card from changing height as you hover.
    line3: `L${p.laser} ${p.ch}  ${fmt(p.hzv)} Hz`,
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

  // The header always states the one gesture that fills this column, so it reads the same
  // whether the workbench is empty or full.
  $('#heldh').querySelector('.link').style.display = S.probes.length ? '' : 'none';
  $('#plist').innerHTML = S.probes.map((p) => {
    const m = p.meta;
    // Two lines, not four: the coloured left border already identifies the probe, so the
    // swatch is redundant, and the layout name belongs in the tooltip rather than a row.
    return `<div class="probe card2" data-id="${p.id}" style="--c:${col(p)}"
      title="${shortId(p.sid)}  pos ${m.pos}  spk ${m.spk}  ${m.layout}">
      <img class="thumb mask" src="/api/masks.png?ids=${p.sid}&colors=${probeHex(p)}&v=${S.rv}" alt="">
      <div class="meta">
        <div class="ln1"><b>${shortId(p.sid)}</b><span class="sub">p${m.pos} s${m.spk}</span>
          <button class="x" data-x="${p.id}">×</button></div>
        <div class="ln3">L${p.laser} ${p.ch}  ${fmt(p.hzv)} Hz</div>
      </div>
    </div>`;
  }).join('');
  $('#plist').querySelectorAll('.probe').forEach((el) => {
    const p = S.probes.find((q) => q.id === +el.dataset.id);
    el.onmouseenter = () => { S.hot = p; drawSpec(); drawShifts(); markHot(); probeTicks(); fieldGrid(); };
    el.onmouseleave = () => { S.hot = null; drawSpec(); drawShifts(); markHot(); probeTicks(); fieldGrid(); };
  });
  $('#plist').querySelectorAll('.x').forEach((b) => (b.onclick = (e) => {
    e.stopPropagation();
    S.probes = S.probes.filter((q) => q.id !== +b.dataset.x);
    reshade();                            // the fan re-spaces around what is left
    renderProbes(); paintAll(); multiples();
  }));
}

// The scene can never overlay, so pinned probes get thumbnails.
// Thumbnails live inside each probe card now, so there is no separate strip.
function multiples() { heldAudio(); allMasks(); }

/* Every scene currently in play, in one image: the live sample plus each pinned probe,
   each mask in that probe's own color so it reads against the curves and ticks. */
function allMasks() {
  const img = $('#allmasks');
  if (!img) return;
  const seen = new Map();
  if (!S.muted && S.live.sid) seen.set(S.live.sid, '1a1a19');    // live = ink, drawn first
  // Pinned probes overwrite the live entry for the same scene, and are appended after it,
  // so a just-pinned probe shows its own color instead of hiding under the black mask.
  for (const p of S.probes) { seen.delete(p.sid); seen.set(p.sid, probeHex(p)); }
  if (!seen.size) { $('#maskfig').hidden = true; return; }
  $('#maskfig').hidden = false;
  const ids = [...seen.keys()].join(','), cols = [...seen.values()].join(',');
  img.src = `/api/masks.png?ids=${ids}&colors=${cols}&v=${S.rv}`;
}

/* Who is who, in the panels where several probes are drawn on the same axes. */
function legends() {
  const live = S.muted ? '' :
    `<span data-live="1"><b style="color:${css('--ink')}"></b>` +
    `current — L${S.live.laser}  ${S.live.ch}</span>`;
  const rows = S.probes.map((p) => {
    const d = p.dash?.length ? `border-top-style:${p.dash[0] > 4 ? 'dashed' : 'dotted'}` : '';
    return `<span data-id="${p.id}"><b style="color:${col(p)};${d}"></b>` +
      `${fmt(p.hzv)} Hz  ${shortId(p.sid)}  L${p.laser}  ${p.ch}</span>`;
  }).join('');
  const prev = S.preview
    ? `<span><b style="color:${css('--accent')}"></b>preview — L${S.hoverLaser}</span>` : '';
  $('#siglegend').innerHTML = S.probes.length || S.preview ? live + rows + prev : '';
  // The big mode plot always overlays every probe, so the legend always applies.
  $('#modelegend').innerHTML = S.probes.length ? live + rows : '';
  for (const el of [$('#siglegend'), $('#modelegend')]) {
    el.querySelectorAll('span[data-id]').forEach((sp) => {
      sp.style.cursor = 'pointer';
      sp.onmouseenter = () => {
        S.hot = S.probes.find((q) => q.id === +sp.dataset.id);
        drawSpec(); drawShifts(); markHot(); probeTicks(); fieldGrid();
      };
      sp.onmouseleave = () => { S.hot = null; drawSpec(); drawShifts(); markHot(); probeTicks(); fieldGrid(); };
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

/* One player per pinned probe, so you can compare what they sound like. */
function heldAudio() {
  const el = $('#heldaudio');
  if (!el) return;
  el.innerHTML = S.probes.map((p) => {
    const ch = audioCh(p.ch);
    return `<div class="ha" style="--c:${col(p)}">
      <span class="tag">${shortId(p.sid)}  L${p.laser}  ${ch}</span>
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
  document.querySelectorAll('#p1, #p2').forEach((el) => freqAxis(el.parentElement));

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
  seg('#specmode', (v) => { S.specMode = v; drawSpec(); buildPeaks(); peakList(); cursors(); });
  seg('#speclog', (v) => { S.log.spec = +v; drawSpec(); buildPeaks(); peakList(); cursors(); });
  seg('#kind', (v) => { S.kind = v; refresh(); });
  wheel($('#fkey2d'));                    // the square 2-D key, drawn once
  $('#fieldbg').onchange = (e) => {
    S.fieldbg = e.target.checked;
    $('#modekey').hidden = !S.fieldbg;      // the key only means something with the fill on
    drawMode();
  };

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
    $('#play').textContent = S.anim ? '❚❚ pause' : '▶ play';
    const tick = () => { if (!S.anim) return; drawMode(); requestAnimationFrame(tick); };
    tick();
  };

  addEventListener('keydown', (e) => {
    if (e.target.tagName === 'INPUT') return;
    // Enter/Escape are global -- they must work while a panel has focus. Only the
    // arrow keys defer to a focused panel, which uses them for its own navigation.
    if (e.key === 'Enter') { e.preventDefault(); return pin(); }
    if (e.key === 'Escape') {
      e.preventDefault();
      S.muted = !S.muted;
      return paintAll();
    }
    // Shift+arrows always walk the peaks, and plain arrows do too once you have clicked
    // one -- otherwise clicking a peak and pressing -> would silently change SAMPLE,
    // which is what it used to do.
    const onPeak = S.peakMode && (S.probe?.peaks || []).includes(S.fi);
    const inList = $('#pklist')?.classList.contains('armed');
    if (e.shiftKey || onPeak) {
      const K = { ArrowLeft: [-1, 0], ArrowRight: [1, 0], ArrowUp: [0, -1], ArrowDown: [0, 1] };
      const d = K[e.key];
      if (!d) { /* fall through */ }
      else if (inList) {                 // the list: navigate the grid as drawn
        e.preventDefault();
        return stepGrid(d[0], d[1]);
      } else if (d[1] === 0) {           // the plot: left/right walk frequency
        e.preventDefault();
        return stepPeak(d[0]);
      } else if (onPeak) {               // the plot: up/down still walk rank
        e.preventDefault();
        return stepGrid(0, d[1]);
      }
    }
    if (e.target.closest('.plot, svg')) return;   // panel-local keys win
    const ps = matches();
    const i = ps.findIndex((s) => s.id === S.live.sid);
    if (e.key === 'p') return pin();
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
