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
  specMode: 'magphase', phmode: 'cos', fieldbg: false, kind: 'clean',
  modeview: 'quiver',
  // How the sample gallery's thumbnails render (see galThumbUrl) -- independent toggles,
  // like #fieldbg's background checkbox for the mode plot.
  gal: { bg: true, mask: false },
  // Per-plot box-zoom windows, as fractions [0..1] of each axis (see the zoom section).
  zoom: { p1: fullView(), p2: fullView(), p3: fullView() },
  dragging: false,
  empty: false, anim: 0, asize: 1, frame: 0,
  // empty set == no constraint. `objects` matches on object TYPE (a sample lists several),
  // `nobj` on the object COUNT -- independent axes. `boxes` is the physical box a sample
  // was captured in (metadata's own "box", not which dataset dir is loaded -- one dataset
  // can span several boxes).
  f: { boxes: new Set(), objects: new Set(), layouts: new Set(), nobj: new Set(), spk: new Set([1]) },
};

window.__S = S;
// Sentinel object-filter value standing in for "no objects" (an empty-box sample): real
// object names come straight from the dataset, so this can't collide with one.
const NONE_OBJ = '∅ (empty box)';
const hasObj = (s) => (s.objects.length ? s.objects.some((o) => S.f.objects.has(o)) : S.f.objects.has(NONE_OBJ));
/* One filter, applied everywhere downstream: the sample list, the position scatter and
   the speaker ring all read from this, so they can never disagree about what exists. */
function pass(s) {
  return (!S.f.boxes.size || S.f.boxes.has(s.box)) &&
         (!S.f.objects.size || hasObj(s)) &&
         (!S.f.layouts.size || S.f.layouts.has(s.layout)) &&
         (!S.f.nobj.size || S.f.nobj.has(s.n)) &&
         (!S.f.spk.size || S.f.spk.has(s.spk));
}
const matches = () => S.all.filter(pass);

/* Faceted counts: every filter list shows its row counts measured against the OTHER
   filters, not itself. So with "1 object" chosen, the layout list still shows a number
   for every layout -- how many 1-object samples that layout has -- and picking a speaker
   updates all of them. `skip` is the facet whose own constraint is dropped. */
function passExcept(s, skip) {
  return (skip === 'boxes' || !S.f.boxes.size || S.f.boxes.has(s.box)) &&
         (skip === 'objects' || !S.f.objects.size || hasObj(s)) &&
         (skip === 'layouts' || !S.f.layouts.size || S.f.layouts.has(s.layout)) &&
         (skip === 'nobj' || !S.f.nobj.size || S.f.nobj.has(s.n)) &&
         (skip === 'spk' || !S.f.spk.size || S.f.spk.has(s.spk));
}
function facetTally(field, skip) {
  const c = new Map();
  for (const s of S.all)
    if (passExcept(s, skip)) c.set(s[field], (c.get(s[field]) || 0) + 1);
  return c;                                   // facet value -> count under the other filters
}
// Object TYPE is multivalued per sample, so it tallies membership, not a single key.
function facetObjTally() {
  const c = new Map();
  for (const s of S.all)
    if (passExcept(s, 'objects')) {
      if (s.objects.length) for (const o of s.objects) c.set(o, (c.get(o) || 0) + 1);
      else c.set(NONE_OBJ, (c.get(NONE_OBJ) || 0) + 1);
    }
  return c;
}
function objectValues() {
  const vals = [...new Set(S.all.flatMap((s) => s.objects))].sort();
  return S.all.some((s) => !s.objects.length) ? [...vals, NONE_OBJ] : vals;
}

/* Filters are per-dataset: a layout name, object count or speaker held over from the old
   box may not exist in the new one, which would leave every picker (scatter, speaker
   ring, sample list) matching nothing. Drop any stored value the new dataset lacks -- an
   emptied set just means "no constraint", so a switch can never land on zero matches. */
function pruneFilters() {
  const have = {
    boxes: new Set(S.all.map((s) => s.box)),
    objects: new Set(S.all.flatMap((s) => (s.objects.length ? s.objects : [NONE_OBJ]))),
    layouts: new Set(S.all.map((s) => s.layout)),
    nobj: new Set(S.all.map((s) => s.n)),
    spk: new Set(S.all.map((s) => s.spk)),
  };
  for (const k of ['boxes', 'objects', 'layouts', 'nobj', 'spk'])
    for (const v of [...S.f[k]]) if (!have[k].has(v)) S.f[k].delete(v);
}

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
// The heatmap paints one plane, and avg IS a plane, so only "both" needs replacing there.
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
/* One full oscillation of the mode, quantised into frames. The slider spans exactly one
   period, so scrubbing off either end wraps rather than dead-ending. */
const NFRAME = 60;
const probeHex = (p) => hex(p.hue, 68, p.lit ?? 47);
const css = (n) => getComputedStyle(document.body).getPropertyValue(n).trim();

/* ***** boot ***** */
(async function () {
  loadPayload(await api('/api/samples'));
  wire(); phVis();
  await firstSample();
  setInterval(poll, 500);
})();

// Live: watch_and_process.py drops new samples (and new experiment dirs) under the watch
// path while it's running. /api/samples rescans both each call; only rebuild the UI when
// something actually moved. rv (not sample/dataset COUNT) is the right signal -- it's
// bumped server-side on any add/remove/reprocess, including a sample deleted and
// recaptured under the same id, which leaves the count unchanged but the content stale.
async function poll() {
  const j = await api('/api/samples');
  if (j.rv !== S.rv) loadPayload(j);
}

/* Swap in another dataset (box). The samples, positions, speaker ring, grid and scales
   are all per-box, so this is a full reload of everything except the workbench -- pinned
   probes keep their own box tag and stay. */
async function switchDataset(name) {
  if (name === S.info.dataset || !S.datasets.includes(name)) return;
  loadPayload(await api(`/api/switch/${name}`));
  for (const t of ['p1', 'p2', 'p3']) S.zoom[t] = fullView();   // firstSample() redraws
  await firstSample();
}

function loadPayload(j) {
  S.all = j.samples; S.info = j.info; S.rv = j.rv; S.datasets = j.datasets;
  S.dsCounts = j.dataset_counts || {};        // per-dataset sample totals, for the picker
  S.byId = {}; S.byPos = {};
  for (const s of j.samples) {
    S.byId[s.id] = s;
    (S.byPos[s.pos] ??= {})[s.spk] = s.id;
  }
  pruneFilters();
  buildFilters(); buildScatter(); buildSpk(); buildGrid(); sizeModePanel(); applyFilter();
}

// Prefer a sample that passes the current filter, so a switch lands on a state the
// pickers actually agree with; fall back to any non-empty sample, then anything.
const firstSample = () => select(
  (S.all.find((s) => !s.empty && pass(s)) || S.all.find((s) => !s.empty) || S.all[0]).id);

/* The panel's WIDTH is scaled by the box's cropped-overhead pixels against the largest box
   in the set, so a physically bigger box stays visibly bigger. Its HEIGHT follows the laser
   grid's own rows/cols ratio, not the photo's -- arrows() lays out cells as w/cols by
   h/rows, so any other height makes cells non-square and the whole plot looks stretched
   (e.g. a rectangular photo with an actually-square 10x10 laser grid). */
function sizeModePanel() {
  const [ow] = S.info.overhead, big = Math.max(...S.info.max_overhead);
  const { rows, cols } = S.info;
  const w = (460 / big) * ow;             // 460px for the largest box's long side
  const h = w * (rows / cols);
  const root = document.documentElement.style;
  root.setProperty('--mw', `${Math.round(w)}px`);
  root.setProperty('--mh', `${Math.round(h)}px`);
}

/* ***** selection ***** */
async function select(sid) {
  if (!S.byId[sid]) return;
  S.live.sid = sid;
  $('#samplebox')?._draw?.();               // move the highlight to the chosen row
  S.d = await api(`/api/sample/${sid}`);
  // A new sample resonates at its OWN frequencies, so carrying the previous bin over
  // usually lands on noise. Clearing it makes refresh() snap to this sample's strongest
  // peak -- the most informative place to start, and the mode there is worth looking at.
  S.fi = null;
  await refresh();
}

async function refresh(paint = true) {
  const { sid, ch } = S.live;
  // "all" is a rendering of every laser, not a laser: the curves behind it stay the
  // average, so the peak list and the readout still mean something.
  const laser = S.live.laser === 'all' ? 'avg' : S.live.laser;
  $('#scene').src = `/api/scene/${sid}.jpg?v=${S.rv}`;
  viewerMedia(sid);
  // "both" is x and y in one axes. The endpoint serves one channel, so ask twice and let
  // the dash pattern -- the channel's own visual channel already -- tell them apart.
  const q = (c) => api(`/api/probe/${sid}?ch=${c}&laser=${laser}&kind=${S.kind}`);
  if (ch === 'both') {
    const [x, y] = await Promise.all([q('x'), q('y')]);
    S.probe = x; S.probeY = y;
  } else {
    S.probe = await q(ch); S.probeY = null;
  }
  // NOT S.peakMode = true here: every sample switch (gallery, box position, anywhere)
  // routes through here with S.fi reset to null, so unconditionally arming peak-mode on
  // every single one hijacked the very next arrow press into walking peaks -- via the
  // keydown handler's onPeak check, which runs BEFORE the unified zone dispatch -- no
  // matter which zone (gallery, boxpos, ...) the user actually meant. Peak mode is armed
  // by actually interacting with a peak (clicking one, in peakList()/buildPeaks()); Shift
  // still always walks peaks regardless, unaffected by this.
  if (S.fi == null) S.fi = strongestPeak() ?? Math.floor(S.d.freqs.length / 3);
  await loadMode();
  if (paint) paintAll();
}

/* Channel and raw/clean are global view settings, not part of a probe's identity, so a
   change to either re-fetches every pinned probe too -- otherwise the workbench would be
   left showing half its curves in the previous channel. Each probe keeps its sample,
   laser and frequency. */
async function reprobePinned() {
  const { ch } = S.live, kind = S.kind;
  await Promise.all(S.probes.map(async (p) => {
    const laser = p.laser === 'all' ? 'avg' : p.laser;
    const get = (c) => api(`/api/probe/${p.sid}?ch=${c}&laser=${laser}&kind=${kind}`);
    if (ch === 'both') [p.data, p.dataY] = await Promise.all([get('x'), get('y')]);
    else { p.data = await get(ch); p.dataY = null; }
    p.ch = ch;
    p.dash = DASH[ch] || [];
  }));
}

const reviewChannel = async () => { await refresh(false); await reprobePinned(); paintAll(); };

async function loadMode() {
  S.mode = await api(`/api/mode/${S.live.sid}?fi=${S.fi}`);
}

function paintAll() {
  const s = S.byId[S.live.sid];
  $('#summary').textContent =
    `${shortId(s.id)}  pos ${s.pos}  spk ${s.spk}  ${s.layout}  L${S.live.laser}  ${S.live.ch}`;
  buildPeaks(); peakList(); allMasks(); drawSpec(); drawShifts(); drawMode(); cursors(); axes(); legends();
  readout();
  renderScatter(); renderSpk(); renderSpkRing(); renderGrid(); renderProbes(); renderRepeats();
}

/* ***** curves: live probe + pinned probes in one axes ***** */
// A workbench probe can be toggled off without unpinning it -- everything that draws a
// probe reads this, so hiding it clears the curve, the mode arrows, its ticks and its
// mask in one go while the card stays put.
const shownProbes = () => S.probes.filter((p) => !p.hidden);

// S.hot names whichever curve is highlighted right now -- a pinned probe object (by
// identity), or the sentinel strings 'live'/'preview' for the other two curve kinds. Every
// curve drawn in section 2 checks it the same way, so hovering ANY of them (the legend, the
// workbench card, or the rendered line itself) dims everything else consistently.
const isHot = (who) => S.hot != null && S.hot === who;
const othersDim = () => S.hot != null;

function series() {
  const pinned = shownProbes().flatMap((p) => {
    const c = col(p, othersDim() && !isHot(p) ? 0.18 : 0.9);
    const w = p.id === S.flash || isHot(p) ? 2.6 : 1.6;
    // A "both" probe is one probe drawn as two curves: same colour, x solid / y dashed.
    return p.dataY
      ? [{ p, who: p, d: p.data, c, w, dash: DASH.x }, { p, who: p, d: p.dataY, c, w, dash: DASH.y }]
      : [{ p, who: p, d: p.data, c, w, dash: p.dash }];
  });
  // A just-pinned probe is drawn last for a moment so you see it appear; otherwise the
  // live trace stays in front, because that is the one you are steering.
  const front = pinned.filter((s) => s.p.id === S.flash);
  const back = pinned.filter((s) => s.p.id !== S.flash);
  // In "both" the live trace is two curves of one colour, separated by dash: x SOLID,
  // y dashed with a wide gap. One solid line reads faster than two dashed ones, and the
  // gap has to be wide or the two textures blur together at this line density.
  const liveDim = othersDim() ? !isHot('live') : !!S.preview;
  const liveW = isHot('live') ? 2.6 : 1.7;
  const live = S.muted ? []
    : [{ p: null, who: 'live', d: S.probe, c: css('--ink'), w: liveW, dim: liveDim, dash: DASH.x },
       ...(S.probeY ? [{ p: null, who: 'live', d: S.probeY, c: css('--ink'), w: liveW, dim: liveDim, dash: DASH.y }] : [])];
  const previewDim = othersDim() && !isHot('preview');
  return [
    ...back,
    ...live,
    ...front,
    ...(S.preview && !S.muted
      ? [{ p: null, who: 'preview', d: S.preview, c: css('--accent'), w: isHot('preview') ? 2.6 : 1.7, dim: previewDim }]
      : []),
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

/* Shared y-range so overlaid probes are actually comparable.

   The range is GLOBAL -- data.py:_scales() measures it once at boot across a sample of
   every recording and ships it in S.info.scale. Deriving it from the drawn curves instead
   would rescale the axis on every sample change, so identical curve heights would mean
   different physical values and the axis would appear to drift as you browse.

   Falls back to the drawn data for any quantity without a global entry.

   `mag` is the exception: its global range is the single tallest resonance in the whole
   dataset (~90x a typical peak), so every ordinary linear spectrum is a flat line on the
   floor and a channel switch looks like nothing changed. It autoscales to what's drawn,
   with the zero line kept in view since magnitude is a distance from zero. */
function span(list, key) {
  if (key === 'mag') {
    // |FFT| cannot be negative -- 0 is a hard floor, so the axis starts flush at it
    // rather than faking a negative minimum for cosmetic headroom.
    let hi = 0;
    for (const s of list) for (const v of vals(s.d, key)) if (v > hi) hi = v;
    return [0, hi * 1.05 || 1];
  }
  const g = S.info?.scale?.[key];
  if (g) return [g[0], g[1]];
  let lo = Infinity, hi = -Infinity;
  for (const s of list) for (const v of vals(s.d, key)) { if (v < lo) lo = v; if (v > hi) hi = v; }
  return [lo, hi];
}

/* cos(phase) instead of the raw angle. Raw phase wraps at +/-pi, so the plot is a picket
   fence of 2pi jumps that hides the structure; the cosine is continuous and shows the same
   thing. Derived here and cached on the payload -- the server already sent the angle. */
function vals(d, key) {
  if (key !== 'cosphase') return d[key];
  return (d._cos ??= d.phase.map(Math.cos));
}

// Shared top/bottom margin (px), so a value at the true lo/hi (e.g. linear mag's hard 0
// floor) isn't drawn flush on the frame edge. The one place value -> pixel happens, for
// the curve, the peak markers and the axis ticks alike, so all three always agree.
const YPAD = 8;
const yMap = (v, lo, hi, h) => {
  const H = h - 2 * YPAD;
  return YPAD + H - (v - lo) * (H / ((hi - lo) || 1));
};

function multi(cv, key) {
  const [g, w, h] = fit(cv), ss = series().filter((s) => s.d);
  if (!ss.length) return;
  const [flo, fhi] = span(ss, key), z = S.zoom[cv.id];
  const N = Math.max(...ss.map((s) => vals(s.d, key).length));
  const a = z.x0 * (N - 1), b = z.x1 * (N - 1);          // visible index window
  const lo = flo + z.y0 * (fhi - flo), hi = flo + z.y1 * (fhi - flo);
  const sx = w / ((b - a) || 1);
  S.rng[cv.id] = { lo, hi, h };                // so the y axis can label the same scale
  if (cv.id === 'p1') S.p1 = { lo, hi, h };   // peak markers sit ON the curve
  g.save();
  g.beginPath(); g.rect(0, 0, w, h); g.clip();         // a zoomed y must not spill the frame
  for (const s of ss) {
    const y = vals(s.d, key);
    g.beginPath();
    let pen = false;
    for (let i = 0; i < y.length; i++) {
      // Clamp to the currently visible [lo,hi] -- a y-zoom (which persists across sample
      // switches, not just this one plot) can leave `lo` above this sample's true floor,
      // and an unclamped value below it maps to a pixel below the axis's own floor line.
      const X = (i - a) * sx, Y = yMap(Math.min(hi, Math.max(lo, y[i])), lo, hi, h);
      if (X < -sx || X > w + sx) { pen = false; continue; }   // skip offscreen, break the path
      pen ? g.lineTo(X, Y) : g.moveTo(X, Y);
      pen = true;
    }
    g.globalAlpha = s.dim ? 0.28 : 1;
    g.setLineDash(s.dash || []);
    g.strokeStyle = s.c; g.lineWidth = s.w; g.stroke();
    g.setLineDash([]);
    g.globalAlpha = 1;
  }
  g.restore();
}

// The same two ramps as viz2/render.py, so the colorbars match the PNGs exactly.
const SEQ = ['#ffffff', '#cde2fb', '#9ec5f4', '#6da7ec', '#3987e5', '#256abf', '#184f95', '#0d366b'];
const DIV = ['#8c3b12', '#d98635', '#f7dcc0', '#f5f5f3', '#cfe0f2', '#5595d4', '#124b8e'];

/* "All lasers" turns each plot into a lasers x frequency image. The five quantities have
   different units and two different palettes, so each carries its own colorbar. */
const HEATQ = () => (S.specMode === 'magphase'
  ? [S.log.spec ? 'logmag' : 'mag', S.phmode === 'cos' ? 'cosphase' : 'phase'] : ['re', 'im']);

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

// Hovering a rendered line, not just its legend row, highlights it the same way: whichever
// series passes closest (in pixel Y, at the already-hovered frequency bin) to the cursor.
// Reuses S.rng[cvid] -- the lo/hi/h that multi() just drew this canvas with -- so the pixel
// math here can never disagree with what's actually on screen.
function nearestSeries(cvid, key, my) {
  const rng = S.rng[cvid];
  if (!rng || S.hoverFi == null) return null;
  const { lo, hi, h } = rng;
  let best = null, bd = Infinity;
  for (const s of series()) {
    if (!s.d) continue;
    const v = vals(s.d, key)?.[S.hoverFi];
    if (v == null) continue;
    const Y = yMap(Math.min(hi, Math.max(lo, v)), lo, hi, h);
    const d = Math.abs(Y - my);
    if (d < bd) { bd = d; best = s; }
  }
  return best && bd <= 16 ? best.who : null;
}

// Which quantity each spectrum plot draws -- shared with the hover tooltip, so a value it
// reads is always the value the curve is actually showing.
function specKey(id) {
  if (S.specMode !== 'magphase') return id === 'p1' ? 're' : 'im';
  return id === 'p1' ? (S.log.spec ? 'logmag' : 'mag') : (S.phmode === 'cos' ? 'cosphase' : 'phase');
}

function drawSpec() {
  multi($('#p1'), specKey('p1')); multi($('#p2'), specKey('p2'));
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
  const { lo, hi, h } = r;
  // Linear magnitude is a distance from zero, so the zero line is the reference the eye
  // reads everything against. The tick step only lands on it by luck, so force it in.
  const tv = ticks(lo, hi, 4);
  if (S.specMode === 'magphase' && !S.log.spec && id === 'p1'
      && lo <= 0 && hi >= 0 && !tv.some((v) => Math.abs(v) < 1e-9)) tv.push(0);
  // Same yMap() the curve and its peak markers use, as a percent of the plot's own height
  // -- so the gridline, its number, and the data itself always land on the same pixel.
  g.innerHTML = tv.map((v) => {
    const y = yMap(v, lo, hi, h) / h * 100;
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
// A probe kept from another box may have a different laser grid; its mode field can only
// be overlaid on a box with the same dims.
const fitsGrid = (p) => p.mode
  && p.mode.u.length === S.info.rows && p.mode.u[0]?.length === S.info.cols;

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

/* The rail and the axes both mean different things per view: the size slider scales
   arrows or height, the color fill is a quiver-only readout, and the row/column ticks
   label a grid that the surface's camera has rotated away from. */
function viewCtls() {
  const q = S.modeview === 'quiver';
  $('#asizelab').textContent = q ? 'arrow size' : 'height';
  $('#modekey').hidden = !(q && S.fieldbg);
  $('#modeax').style.display = q ? '' : 'none';
  // Hidden rather than wrapped in a container: the rail is a flex column with its own
  // gap, so a wrapper div would change the spacing of everything around it.
  for (const e of document.querySelectorAll('.bgctl, .modewrap .mlab')) e.hidden = !q;
}

function drawMode() {
  fieldGrid();
  recoveredGrid();
  modeAxes();
  const cv = $('#mode'), [g, w, h] = fit(cv);
  if (!S.mode) return;
  const R = S.info.rows, C = S.info.cols, cw = w / C, chh = h / R;
  const withMode = shownProbes().filter(fitsGrid);
  const sets = [
    ...withMode.filter((p) => p.id !== S.flash).map((p) => ({ m: p.mode, c: col(p, 0.9) })),
    ...(S.muted && S.probes.length ? [] : [{ m: S.mode, c: css('--ink') }]),
    ...withMode.filter((p) => p.id === S.flash).map((p) => ({ m: p.mode, c: col(p, 0.9) })),
  ];

  if (S.modeview === 'surface') {
    if (!hasZ(sets)) return stale(g, w, h);
    // Every set on ONE shared height scale, so an overlaid surface still compares
    // amplitudes the way the overlaid quiver does.
    surfaces(g, sets.map(surfCol), w, h, R, C, zmax(sets));
    return;
  }
  // The big plot always overlays every probe. Arrows from different probes share a cell
  // origin, so color alone would hide one under another: fan the tails around the centre.
  const k = (Math.min(cw, chh) * 0.42 * S.asize) / (maxOf(sets) || 1);
  arrows(g, sets, w, h, R, C, cw, chh, k, sets.length > 1 ? Math.min(cw, chh) * 0.15 : 0);
}

/* Peak of a quantity over any number of sets. `f` picks what a cell is worth: arrow
   length for the quiver, |height| for the surface -- one reduction, two views. */
const peakOf = (sets, f) => {
  let max = 0;
  for (const s of sets) for (let r = 0; r < s.m.u.length; r++)
    for (let c = 0; c < s.m.u[r].length; c++) max = Math.max(max, f(s.m, r, c));
  return max;
};
const maxOf = (sets) => peakOf(sets, (m, r, c) => Math.hypot(m.u[r][c], m.v[r][c]));
const zmax = (sets) => peakOf(sets, (m, r, c) => Math.abs(m.z[r][c]));

function arrows(g, sets, w, h, R, C, cw, chh, k, fan) {
  const t = Math.cos((S.frame / NFRAME) * 2 * Math.PI);
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

/* The height field comes from the server, so a server started before the surface view
   existed serves modes with no `z`. Without this the draw throws and the canvas silently
   keeps whatever the quiver last painted -- which looks exactly like a dead button. */
const hasZ = (sets) => sets.length > 0 && sets.every((s) => s.m && s.m.z);

function stale(g, w, h) {
  g.save();
  g.fillStyle = css('--ink3') || '#888';
  g.font = '11px ui-monospace, monospace';
  g.textAlign = 'center';
  g.fillText('surface needs a server restart', w / 2, h / 2 - 6);
  g.fillText('(/api/mode returned no z)', w / 2, h / 2 + 10);
  g.restore();
}

/* ***** the surface view *****
   The quiver draws the mode's GRADIENT; this draws the height field that gradient
   integrates to (server-side, data.surface) -- the same pair figure_signals.ipynb puts
   side by side, and the view most people actually picture when they hear "mode shape".

   A 2-D surface embedded in 3-D, drawn with plain canvas: the grid is a height field seen
   from ONE fixed direction, so quads sorted back-to-front composite correctly with no
   z-buffer, and a quad's normal against a fixed light gives the notebook's shaded look.
   Matches the notebook's camera (elev 25, azim -60) and light (az 315, alt 45). */
const ELEV = 25 * Math.PI / 180, AZIM = -60 * Math.PI / 180;
/* The notebook's set_zlim(global_z_lims), where the limit is the peak times 2: the box is
   given twice the height it needs, so even a full-amplitude frame sits in the middle half
   and never grazes the top. Fixed, not tracking the height slider -- see zbox below. */
const ZBOX = 2.0;
const LIGHT = (() => {                    // unit vector toward the lamp
  const a = 315 * Math.PI / 180, e = 45 * Math.PI / 180;
  return [Math.cos(e) * Math.cos(a), Math.cos(e) * Math.sin(a), Math.sin(e)];
})();

/* The live mode draws in --ink, which is near-black and unsaturated. That is right for a
   line but wrong for a lit surface: with l~0.1 and s~0 there is no headroom for the height
   ramp to brighten into and no saturation for it to shift, so the surface would render as
   a flat dark blob. Give it the app's own blue instead -- near the notebook's steel blue,
   and the one color here that is already "the current sample". Pinned probes keep their
   own hue, which is what identifies them. */
const surfCol = (s) => (s.c === css('--ink') ? { ...s, c: css('--accent') } : s);

/* Grid cell + height -> a point in the camera's own units. Depth comes back too: it is
   the sort key, and the only thing that makes the quads composite correctly. */
function camera(c, r, z, C, R, zk) {
  const x = c / (C - 1) - 0.5, y = 0.5 - r / (R - 1);
  const ca = Math.cos(AZIM), sa = Math.sin(AZIM);
  const px = x * ca - y * sa;                       // rotate about the vertical axis
  const py = x * sa + y * ca;
  return [px, -(py * Math.sin(ELEV) + z * zk), py];
}

/* The three axes, in the notebook's spirit: it blanks every pane, grid and tick label
   (grid(False), panes fully transparent) and keeps only faint spines, so nothing competes
   with the surface. These are drawn the same way -- one hairline per axis plus a single
   letter -- but kept explicit and labelled, since the whole point of the box is to say
   which way is row, which is column, and which is displacement.

   Drawn BEFORE the quads, so the surface paints over the far edges and hides them exactly
   as depth would. The near vertical edge is drawn after, in surfaces(), or the surface
   would bury the one axis that carries the height. */
function axes3d(g, scr, R, C, zlim, labels) {
  const O = scr(0, R - 1, -zlim);                     // near-left-bottom corner: the origin
  const ends = [
    [scr(C - 1, R - 1, -zlim), 'col'],                // x: along the laser columns
    [scr(0, 0, -zlim), 'row'],                        // y: along the laser rows
    [scr(0, R - 1, zlim), 'z'],                       // z: displacement
  ];
  g.save();
  g.strokeStyle = css('--ink3') || '#86867e';
  g.fillStyle = css('--ink3') || '#86867e';
  g.lineWidth = 1;
  g.globalAlpha = 0.55;
  g.font = '9px ui-monospace, monospace';
  g.textAlign = 'center';
  g.textBaseline = 'middle';
  for (const [e, lab] of ends) {
    g.beginPath();
    g.moveTo(O[0], O[1]);
    g.lineTo(e[0], e[1]);
    g.stroke();
    if (!labels) continue;
    // Push the label a little past the end, along the axis, so it never sits on the line.
    const dx = e[0] - O[0], dy = e[1] - O[1], n = Math.hypot(dx, dy) || 1;
    g.fillText(lab, e[0] + (dx / n) * 9, e[1] + (dy / n) * 9);
  }
  g.restore();
}

/* Every set drawn into one scene. Quads from ALL of them share one depth sort, so an
   overlaid surface interleaves correctly instead of one being painted flat on top. */
function surfaces(g, sets, w, h, R, C, max) {
  const t = Math.cos((S.frame / NFRAME) * 2 * Math.PI);
  // Height in camera units, relative to the unit-square footprint. Shared by every set,
  // so an overlay compares amplitudes rather than normalising them apart.
  const zk = (0.55 * S.asize) / (max || 1);
  /* The z box is measured at asize 1, NOT at the current asize. Letting it track the
     slider made the fit zoom out as the slider came up -- at 3x the footprint collapsed to
     26px against a 277px z axis, so "more height" actually shrank the surface. Fixed, the
     slider does what it says: the box stays put and the relief inside it grows. */
  // ...but never smaller than the surface itself: past asize ~2 the relief would grow out
  // through the top of a box that ignored it. Below that the box is fixed and the slider
  // just fills more of it, which is the notebook's behaviour.
  const zbox = (0.55 * Math.max(ZBOX, S.asize)) / (max || 1);
  // Slope in the same units the camera uses, so the normal is the SCREEN normal.
  const gx = (C - 1) / (zk || 1), gy = (R - 1) / (zk || 1);

  /* Fit the camera's output to the box instead of guessing a margin. The height slider
     changes the extent by 10x, so a fixed scale either overflows at the top of its range
     or wastes most of the box at the bottom. One pass over the corners of the height
     envelope bounds every point that can be drawn. Measured on the fixed z BOX, so the
     framing is the same whatever the slider and whatever the frame's amplitude. */
  let x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity;
  // The camera is affine in (c, r), so the extremes sit at the footprint's four corners --
  // taken at both height extremes, that is 8 points, not R*C.
  for (const [r, c] of [[0, 0], [0, C - 1], [R - 1, 0], [R - 1, C - 1]])
    for (const z of [-max, max]) {
      const [px, py] = camera(c, r, z, C, R, zbox);
      x0 = Math.min(x0, px); x1 = Math.max(x1, px);
      y0 = Math.min(y0, py); y1 = Math.max(y1, py);
    }
  const fitk = Math.min(w / (x1 - x0 || 1), h / (y1 - y0 || 1)) * 0.94;
  const ox = w / 2 - ((x0 + x1) / 2) * fitk, oy = h / 2 - ((y0 + y1) / 2) * fitk;
  const scr = (c, r, z) => {                  // the SURFACE: height follows the slider
    const [px, py, d] = camera(c, r, z, C, R, zk);
    return [ox + px * fitk, oy + py * fitk, d];
  };
  const scrBox = (c, r, z) => {                // the BOX: fixed, so the axes never move
    const [px, py, d] = camera(c, r, z, C, R, zbox);
    return [ox + px * fitk, oy + py * fitk, d];
  };

  /* Labels only on the big plot. On a 124px tile the col axis is 26px long and its label
     lands on the frame edge -- and the tiles already carry a caption saying what they are,
     so the letters would be clutter over the one thing the tiles exist to show. */
  axes3d(g, scrBox, R, C, max, w > 200);

  /* The notebook's compute_normalized_shading: how hard to shade THIS frame. The mode
     oscillates through flat, so shading against the frame's own range would hold the
     contrast constant and kill the pulse -- the surface would look like a rigid object
     rocking rather than a membrane breathing. Measured against the GLOBAL amplitude
     instead, a flat frame gets flat color and a peak frame gets full relief. The sqrt is
     the notebook's, and it matters: it keeps the mid-cycle frames from washing out. */
  const amp = Math.sqrt(Math.min(1, Math.abs(t)));

  const quads = [];
  sets.forEach((s) => {
    const base = rgbOf(s.c);
    const P = s.m.z.map((row, r) => row.map((z, c) => scr(c, r, z * t)));
    for (let r = 0; r < R - 1; r++) for (let c = 0; c < C - 1; c++) {
      const a = P[r][c], b = P[r][c + 1], d = P[r + 1][c + 1], e = P[r + 1][c];
      // Central-ish differences on the cell, then Lambert against the fixed lamp.
      const zc = s.m.z[r][c] * t, zr = s.m.z[r][c + 1] * t, zd = s.m.z[r + 1][c] * t;
      const nx = -(zr - zc) / gx, ny = (zd - zc) / gy;
      const len = Math.hypot(nx, ny, 1);
      const lamb = (nx * LIGHT[0] + ny * LIGHT[1] + LIGHT[2]) / len;
      // Height of the cell, signed and normalised to -1..1 -- crest vs trough.
      const zn = max ? ((zc + zr + zd + s.m.z[r + 1][c + 1] * t) / 4) / max : 0;
      quads.push({
        pts: [a, b, d, e],
        depth: (a[2] + b[2] + d[2] + e[2]) / 4,
        rgb: shadeQuad(base, lamb, zn, amp),
        alpha: sets.length > 1 ? 0.72 : 1,
      });
    }
  });
  quads.sort((p, q) => p.depth - q.depth);          // far first: painter's algorithm
  for (const q of quads) {
    const [r0, g0, b0] = q.rgb;
    g.fillStyle = `rgba(${r0},${g0},${b0},${q.alpha})`;
    g.strokeStyle = g.fillStyle;                    // hairline seam: kills the gaps
    g.lineWidth = 0.6;
    g.beginPath();
    g.moveTo(q.pts[0][0], q.pts[0][1]);
    for (let i = 1; i < 4; i++) g.lineTo(q.pts[i][0], q.pts[i][1]);
    g.closePath();
    g.fill(); g.stroke();
    /* The wireframe. Only two of the four edges per quad, so each interior line is drawn
       once rather than twice -- doubling would make it read as a hard mesh instead of the
       faint ruling that just gives the eye the surface's curvature. */
    g.strokeStyle = `rgba(255,255,255,${0.13 * q.alpha})`;
    g.lineWidth = 0.5;
    g.beginPath();
    g.moveTo(q.pts[3][0], q.pts[3][1]);
    g.lineTo(q.pts[0][0], q.pts[0][1]);
    g.lineTo(q.pts[1][0], q.pts[1][1]);
    g.stroke();
  }

  /* The z axis again, over the top. It stands at the near corner, so the surface would
     otherwise bury the one axis that shows displacement -- the very thing this view is
     for. The other two are left occluded, which is the correct depth cue. */
  const O = scrBox(0, R - 1, -max), Z1 = scrBox(0, R - 1, max);
  g.save();
  g.strokeStyle = css('--ink3') || '#86867e';
  g.globalAlpha = 0.55;
  g.lineWidth = 1;
  g.beginPath(); g.moveTo(O[0], O[1]); g.lineTo(Z1[0], Z1[1]); g.stroke();
  g.restore();
}

/* One quad's color: the base carried through Lambert shading AND height.

   HUE IS LEFT ALONE, deliberately. Hue is already spoken for -- hueForPos gives each
   position a hue and shadeFor fans probes only +/-FAN/2 around it, narrower than the
   137.5 deg between positions precisely so two probes never collide. Riding height on hue
   too would spend that same budget twice: a crest could drift a probe into its
   neighbour's slot, and the surface would stop saying which probe it is.

   So height moves LIGHTNESS and saturation instead, which nothing else here uses. Crests
   lighten and saturate, troughs darken and mute -- the ordering the eye already reads as
   high/low, and it stays legible in one still frame. The range is wide (0.55x..1.45x
   lightness) because it is now carrying the height signal alone.

   Everything scales by `amp`, so a flat frame collapses to the flat base color exactly as
   the notebook's blend toward uniform_color does. */
function shadeQuad([r, g, b], lamb, zn, amp) {
  const [hh, ss, ll] = rgbToHsl(r, g, b);
  const lit = 0.45 + 0.7 * Math.max(0, lamb);                // Lambert, never pure black
  // Height rides on lightness, about the base: crest up, trough down.
  const l = ll * (1 - amp) + ll * lit * (1 + 0.45 * zn) * amp;
  // ...and on saturation in the SAME direction, so a crest reads bright-and-vivid and a
  // trough muted. Two channels agreeing makes the height readable at tile size.
  const sat = ss * (1 - amp) + ss * (1 + 0.30 * zn) * amp;
  return hslToRgb(hh, Math.max(0, Math.min(1, sat)), Math.max(0.04, Math.min(0.96, l)));
}

function rgbToHsl(r, g, b) {
  r /= 255; g /= 255; b /= 255;
  const mx = Math.max(r, g, b), mn = Math.min(r, g, b), d = mx - mn;
  const l = (mx + mn) / 2;
  if (!d) return [0, 0, l];
  const s = d / (1 - Math.abs(2 * l - 1));
  const h = mx === r ? ((g - b) / d + (g < b ? 6 : 0))
          : mx === g ? (b - r) / d + 2 : (r - g) / d + 4;
  return [h * 60, s, l];
}

function hslToRgb(h, s, l) {
  const c = (1 - Math.abs(2 * l - 1)) * s, x = c * (1 - Math.abs(((h / 60) % 2) - 1));
  const m = l - c / 2;
  const [r, g, b] = h < 60 ? [c, x, 0] : h < 120 ? [x, c, 0] : h < 180 ? [0, c, x]
                  : h < 240 ? [0, x, c] : h < 300 ? [x, 0, c] : [c, 0, x];
  return [(r + m) * 255 | 0, (g + m) * 255 | 0, (b + m) * 255 | 0];
}

/* Any CSS color -> [r,g,b]. Probe colors are hsl() and --ink is a hex; assigning to
   fillStyle and reading it back is the one thing that normalises both without a parser.
   Memoised because it runs once per quad set per frame. */
const _rgbCv = document.createElement('canvas').getContext('2d');
const _rgbMemo = {};
function rgbOf(css) {
  if (!(css in _rgbMemo)) {
    _rgbCv.fillStyle = css;                      // canvas normalises to #rrggbb or rgba()
    const v = _rgbCv.fillStyle;
    _rgbMemo[css] = v[0] === '#'
      ? [1, 3, 5].map((i) => parseInt(v.slice(i, i + 2), 16))
      : v.match(/[\d.]+/g).slice(0, 3).map(Number);
  }
  return _rgbMemo[css];
}

// Direction -> hue, magnitude -> lightness. (spatial_derivatives_to_hsv, reimplemented.)
const fcol = (u, v, max) =>
  `hsl(${(Math.atan2(v, u) * 180) / Math.PI + 180} 70% ${18 + 52 * (Math.hypot(u, v) / (max || 1))}%)`;

const peak = (m) => maxOf([{ m }]);

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
  const cur = S.mode && lm ? { m: S.mode, c: css('--ink'), id: 0, t: cap(lm, hz(S.fi)) } : null;
  const rest = shownProbes().filter(fitsGrid)
    .map((p) => ({ m: p.mode, c: col(p), id: p.id, t: cap(p.meta, p.hzv) }));
  const sets = [...(cur ? [cur] : []), ...rest];
  // Draw even for a single set. The tiles are not only a comparison: the lone tile is the
  // current mode on its OWN scale, which the overlaid plot above cannot show, so it is
  // worth having before any probe is pinned.
  if (!sets.length) { box.innerHTML = ''; box.dataset.sig = ''; return; }

  const sig = sets.map((s) => s.id).join(',') + '|' + sets.map((s) => s.t).join('|');
  if (box.dataset.sig !== sig) {          // rebuild only when the SET changes, not per frame
    box.dataset.sig = sig;
    // .fgrid-rest lists in DOM order right after .fgrid-cur, so `figure canvas` below still
    // walks in the same order as `sets` (current first, then the rest) despite the nesting.
    const tile = (s) => `<figure data-id="${s.id}">` +
      `<canvas width="124" height="124"></canvas>` +
      `<figcaption style="border-left-color:${s.c}">${s.t}</figcaption></figure>`;
    box.innerHTML =
      (cur ? `<div class="fgrid-cur">${tile(cur)}</div>` : '') +
      `<div class="fgrid-rest">${rest.map(tile).join('')}</div>`;
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
    if (S.modeview === 'surface') {
      if (!hasZ([sets[i]])) return stale(g, w, h);
      surfaces(g, [surfCol(sets[i])], w, h, R, C, zmax([sets[i]]));
      return;
    }
    const max = maxOf([sets[i]]);
    if (S.fieldbg) field(g, w, h, sets[i].m, R, C, max);
    const k = (Math.min(cw, chh) * 0.42 * S.asize) / (max || 1);
    arrows(g, [{ m: sets[i].m, c: S.fieldbg ? '#fff' : sets[i].c }], w, h, R, C, cw, chh, k, 0);
  });
}

/* Recovered-video small-multiples: same current+rest layout as the mode grid above, but
   each tile is that SAMPLE's own already-saved fixed-render video (see recovered_video())
   -- no recomputation per probe, which would be ~1.2s each (matplotlib frames + ffmpeg
   mux) for an interactive pin action. */
function recoveredGrid() {
  const box = $('#recgrid');
  if (!box) return;
  const lm = S.byId[S.live.sid];
  const cap = (s) => `<i>${shortId(s.id)}  pos ${s.pos}  spk ${s.spk}</i>`;
  const cur = lm ? { sid: lm.id, c: css('--ink'), id: 0, t: cap(lm) } : null;
  const rest = shownProbes().map((p) => ({ sid: p.sid, c: col(p), id: p.id, t: cap(p.meta) }));
  const sets = [...(cur ? [cur] : []), ...rest];
  if (!sets.length) { box.innerHTML = ''; box.dataset.sig = ''; return; }

  const sig = sets.map((s) => `${s.id}:${s.sid}`).join(',');
  if (box.dataset.sig !== sig) {          // rebuild only on a SET change -- reloading a
    box.dataset.sig = sig;                // <video>'s src mid-playback would restart it
    const tile = (s) => `<figure data-id="${s.id}"><div class="vidwrap"><video preload="metadata" playsinline></video>` +
      `<div class="vidctl"><button class="vidplay" type="button">▶</button>` +
      `<input class="vidseek" type="range" min="0" max="0" step="0.01" value="0">` +
      `<span class="vidtime">0.00s / 0.00s</span></div></div>` +
      `<figcaption style="border-left-color:${s.c}">${s.t}</figcaption></figure>`;
    box.innerHTML =
      (cur ? `<div class="fgrid-cur">${tile(cur)}</div>` : '') +
      `<div class="fgrid-rest">${rest.map(tile).join('')}</div>`;
    box.querySelectorAll('figure').forEach((f, i) => {
      const wrap = f.querySelector('.vidwrap');
      wireVideoControls(wrap);
      wrap.querySelector('video').onerror = () => { f.style.display = 'none'; };  // no rendered video for this sample
      wrap.querySelector('video').src = `/api/recovered_video/${sets[i].sid}.mp4?v=${S.rv}`;
    });
  }
  box.querySelectorAll('figure').forEach((f) =>
    f.classList.toggle('hot', !!S.hot && +f.dataset.id === S.hot.id));
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

/* ***** box zoom on the signal plots *****
   Drag a rectangle to zoom into it, the way Plotly and W&B do; double-click (or the reset
   button) restores the full view. A near-stationary press is left alone, so tapping a peak
   still selects its frequency. The window is kept as fractions [0..1] of each axis, so
   zooms just compose and no drawing code needs the data length. p1 and p2 are one
   frequency axis, so an x-zoom on either is written to both. */
function fullView() { return { x0: 0, x1: 1, y0: 0, y1: 1 }; }
const xGroup = (id) => (id === 'p3' ? ['p3'] : ['p1', 'p2']);
const zoomed = () => ['p1', 'p2', 'p3'].some((id) => {
  const z = S.zoom[id];
  return z.x0 > 0 || z.x1 < 1 || z.y0 > 0 || z.y1 < 1;
});

// Where bin i sits on the visible p1/p2 axis, 0..1 (outside that on a zoom).
const specFrac = (i) => {
  const z = S.zoom.p1, f = i / ((S.d ? S.d.freqs.length : 1) - 1);
  return (f - z.x0) / (z.x1 - z.x0);
};
const specX = (i) => specFrac(i) * 100;
const inSpecX = (i) => specFrac(i) >= -1e-9 && specFrac(i) <= 1 + 1e-9;

function resetZoom(id) {
  for (const t of id ? xGroup(id) : ['p1', 'p2', 'p3']) S.zoom[t] = fullView();
  drawSpec(); drawShifts(); cursors(); axes();
}

// Fold a sub-range (fractions of what is currently shown) into the stored window.
function applyZoom(id, axis, lo, hi) {
  for (const t of axis === 'x' ? xGroup(id) : [id]) {
    const z = S.zoom[t], a = z[axis + '0'], span = z[axis + '1'] - a;
    z[axis + '0'] = a + lo * span;
    z[axis + '1'] = a + hi * span;
  }
}

// Clamp to [0,1] and pull anything within tol of an edge flush to it -- landing on the
// exact boundary pixel by hand is unreliable, so without this a drag or hover can never
// quite reach axis min/max, only get close.
const snapEdge = (v, tol = 0.015) => (v < tol ? 0 : v > 1 - tol ? 1 : Math.max(0, Math.min(1, v)));

function zoomable(plot, id) {
  const ov = plot.querySelector('.ov');       // the overlay exactly covers the data area
  let start = null, band = null;
  const frac = (e) => {
    const r = ov.getBoundingClientRect();
    return [snapEdge((e.clientX - r.left) / r.width), snapEdge((e.clientY - r.top) / r.height)];
  };

  plot.addEventListener('pointerdown', (e) => {
    if (e.button || S.live.laser === 'all' || e.target.closest('.pk')) return;
    start = frac(e); S.dragging = true;
    hideTip(plot);
    plot.setPointerCapture(e.pointerId);
  });
  plot.addEventListener('pointermove', (e) => {
    if (!start) return;
    const [x, y] = frac(e);
    if (!band) {
      band = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
      band.setAttribute('class', 'zband');
      ov.appendChild(band);
    }
    band.setAttribute('x', `${Math.min(start[0], x) * 100}%`);
    band.setAttribute('y', `${Math.min(start[1], y) * 100}%`);
    band.setAttribute('width', `${Math.abs(x - start[0]) * 100}%`);
    band.setAttribute('height', `${Math.abs(y - start[1]) * 100}%`);
  });
  const end = (e) => {
    if (!start) return;
    const [sx, sy] = start; start = null; S.dragging = false;
    band?.remove(); band = null;
    try { plot.releasePointerCapture(e.pointerId); } catch (_) {}
    const [x, y] = frac(e), dx = Math.abs(x - sx), dy = Math.abs(y - sy);
    if (dx < 0.01 && dy < 0.01) return;                // a click, not a drag -- leave it
    plot._drag = true;                                 // ...but a drag must not also select
    if (dx > 0.02) applyZoom(id, 'x', Math.min(sx, x), Math.max(sx, x));
    if (dy > 0.04) applyZoom(id, 'y', 1 - Math.max(sy, y), 1 - Math.min(sy, y));
    drawSpec(); drawShifts(); cursors(); axes();
  };
  plot.addEventListener('pointerup', end);
  plot.addEventListener('pointercancel', end);
  plot.addEventListener('click', (e) => {
    if (plot._drag) { e.stopPropagation(); plot._drag = false; }
  }, true);
  plot.addEventListener('dblclick', () => resetZoom(id));
}

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
      return `<text class="tk" x="${x}%" y="100%" dy="10">${v}</text>`;
    }).join('') + `<text class="unit" x="50%" y="100%" dy="21">${label}</text>`;
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
  return yMap(mag[i], m.lo, m.hi, m.h);
}

/* The detected resonances, listed. Clicking one is the same as clicking its marker. */
function peakList() {
  const el = $('#pklist');
  if (!el) return;
  // Muted (esc) hides the current curve -- its peak list shouldn't linger either.
  const pk = S.muted ? [] : (S.probe?.peaks || []);
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
  // Navigate the grid exactly as drawn (2 columns, PK_COLS) -- same as clicking a row, just
  // by arrow key, whether this list is hovered or was the last thing clicked.
  registerZone('peaks', el, (dx, dy) => stepGrid(dx, dy));
  if (!el.dataset.armedVis) {
    el.dataset.armedVis = '1';
    el.tabIndex = 0;
    const sync = () => el.classList.toggle('armed', hoverZone === 'peaks' || armedZone === 'peaks');
    ['pointerenter', 'pointerleave', 'focusin', 'focusout'].forEach((ev) => el.addEventListener(ev, sync));
  }
}

function buildPeaks() {
  const g = $('#p1ov').querySelector('.pks');
  if (!g) return;
  // The markers ride the magnitude curve; with an image in its place they have no y, and
  // muted (esc) hides that curve too -- riderless markers would just float there.
  const pk = (S.live.laser === 'all' || S.muted) ? [] : (S.probe?.peaks || []);
  let lastX = -99, up = true;
  g.innerHTML = pk.map((i) => {
    const x = specX(i);
    up = (x - lastX) < 6 ? !up : true;       // too close to the previous label -> flip
    lastX = x;
    return `<circle class="pk" r="3.4" data-fi="${i}"/>` +
           `<text class="pklab" data-fi="${i}" data-up="${up ? 1 : 0}">${fmt(hz(i))}</text>`;
  }).join('');
  g.querySelectorAll('.pk').forEach((c) => (c.onclick = (e) => {
    e.stopPropagation();
    S.peakMode = true;
    // Clicking a peak marker ON THE PLOT is the one zone-worthy click that doesn't happen
    // on a registerZone'd element (the plot itself isn't a zone) -- set armedZone directly
    // so it's consistent with every other "click something, arrows now act on it" case, and
    // so hovering #pklist right after still shows it as armed.
    armedZone = 'peaks';
    setFi(+c.dataset.fi);
  }));
}

function cursors() {
  probeTicks();
  const at = (i) => `${specX(i)}%`;           // through the shared frequency zoom window
  for (const id of OVS) {
    const svg = $(id);
    const sel = svg.querySelector('.sel'), hov = svg.querySelector('.hov');
    sel.setAttribute('x1', at(S.fi)); sel.setAttribute('x2', at(S.fi));
    sel.style.display = inSpecX(S.fi) ? '' : 'none';
    const show = S.hoverFi != null && S.hoverFi !== S.fi && S.live.laser !== 'all'
                 && inSpecX(S.hoverFi);
    hov.style.display = show ? '' : 'none';
    if (show) { hov.setAttribute('x1', at(S.hoverFi)); hov.setAttribute('x2', at(S.hoverFi)); }
  }
  $('#p1ov').querySelectorAll('.pk').forEach((c) => {
    const i = +c.dataset.fi;
    c.setAttribute('cx', at(i)); c.setAttribute('cy', peakY(i));
    c.style.display = inSpecX(i) ? '' : 'none';
    c.classList.toggle('on', i === S.fi);
  });
  $('#p1ov').querySelectorAll('.pklab').forEach((tx) => {
    const i = +tx.dataset.fi;
    tx.setAttribute('x', at(i)); tx.setAttribute('y', peakY(i));
    tx.setAttribute('dy', tx.dataset.up === '1' ? -8 : 15);
    tx.style.display = inSpecX(i) ? '' : 'none';
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
  for (const id of OVS) {
    const svg = $(id);
    let g = svg.querySelector('.pticks');
    if (!g) {
      g = document.createElementNS('http://www.w3.org/2000/svg', 'g');
      g.setAttribute('class', 'pticks');
      svg.insertBefore(g, svg.firstChild);
    }
    g.innerHTML = shownProbes().filter((p) => inSpecX(p.fi)).map((p) => {
      const x = specX(p.fi);
      const dim = S.hot && S.hot !== p;
      const dash = p.dash?.length ? p.dash.join(' ') : '5 3';
      return `<line class="ptick" x1="${x}%" x2="${x}%" y1="0" y2="100%" ` +
             `stroke-dasharray="${dash}" stroke="${col(p, dim ? 0.2 : 0.9)}"/>`;
    }).join('');
  }
}

function axes() {
  if (!S.d) return;
  const f = S.d.freqs, N = f.length, z = S.zoom.p1;
  // Interpolate the freq array at a fraction of the axis, so a zoom that lands between
  // bins still labels itself with the frequency it actually shows.
  const fAt = (fr) => {
    const c = fr * (N - 1), l = Math.floor(c), r = Math.min(N - 1, l + 1);
    return f[l] + (f[r] - f[l]) * (c - l);
  };
  const HZ = 'frequency (Hz)';
  xAxis($('#p1ov'), fAt(z.x0), fAt(z.x1), HZ, true);    // each spectrum plot labels its own axis
  xAxis($('#p2ov'), fAt(z.x0), fAt(z.x1), HZ, true);
  // Shifts sits above them on a time axis of its own, so it labels itself.
  if (S.probe) {
    const zp = S.zoom.p3, dur = S.probe.dur;
    xAxis($('#p3ov'), zp.x0 * dur, zp.x1 * dur, 'time (s)', true);
  }

  const mp = S.specMode === 'magphase';
  // In all-lasers view every plot's y axis is the laser index, not the quantity.
  const all = S.live.laser === 'all';
  yLabel('#y1', all ? 'laser' : mp ? (S.log.spec ? 'log |FFT|' : '|FFT|') : 'real');
  yLabel('#y2', all ? 'laser' : mp ? (S.phmode === 'cos' ? 'cos(phase)' : 'phase (rad)') : 'imag');
  yLabel('#y3', all ? 'laser' : 'shift (px)');
  $('#sigreset')?.classList.toggle('show', zoomed());
}

/* ***** hover tooltip: every visible series' value at the hovered frequency *****
   Wandb-style: a dark box near the cursor, one row per shown probe (color + label + value)
   plus the live probe when it is actually drawn. Piggybacks on freqAxis's own hoverFi, so
   it can never disagree with the crosshair line about which bin is hovered. */
function ttRow(c, lab, v, who) {
  return { c, lab, v, who };
}
function seriesAt(key, fi) {
  const rows = shownProbes().flatMap((p) => {
    const c = probeHex(p), lab = `${shortId(p.sid)} L${p.laser}`;
    return p.dataY
      ? [ttRow(c, `${lab} x`, vals(p.data, key)?.[fi], p), ttRow(c, `${lab} y`, vals(p.dataY, key)?.[fi], p)]
      : [ttRow(c, lab, vals(p.data, key)?.[fi], p)];
  });
  if (!S.muted && S.probe) {
    const c = css('--ink');
    rows.push(...(S.probeY
      ? [ttRow(c, 'current x', vals(S.probe, key)?.[fi], 'live'), ttRow(c, 'current y', vals(S.probeY, key)?.[fi], 'live')]
      : [ttRow(c, 'current', vals(S.probe, key)?.[fi], 'live')]));
  }
  return rows.filter((r) => r.v != null).sort((a, b) => b.v - a.v);
}
function ttEl(el) {
  let tt = el.querySelector('.ptt');
  if (!tt) { tt = document.createElement('div'); tt.className = 'ptt'; el.appendChild(tt); }
  return tt;
}
function hideTip(el) { const tt = el.querySelector('.ptt'); if (tt) tt.style.display = 'none'; }
function showTip(el, cvid, e) {
  if (S.live.laser === 'all' || S.hoverFi == null) return hideTip(el);
  const rows = seriesAt(specKey(cvid), S.hoverFi);
  const tt = ttEl(el);
  if (!rows.length) { tt.style.display = 'none'; return; }
  tt.innerHTML = `<b>${fmt(hz(S.hoverFi))} Hz</b>` + rows.map((r) => {
    // Every row's text is already its own curve's color, so the tooltip reads like the plot
    // does; whichever one is hot (hovered on the plot, its legend, or its workbench card)
    // also gets bolded -- same "highlight travels everywhere" rule the curves follow.
    const hot = isHot(r.who);
    return `<span style="color:${r.c};${hot ? 'font-weight:700' : ''}">` +
      `<i style="background:${r.c}"></i>${r.lab}<em>${fmtTick(r.v)}</em></span>`;
  }).join('');
  tt.style.display = 'block';
  const r = el.getBoundingClientRect();
  let x = e.clientX - r.left + 14, y = e.clientY - r.top + 14;
  if (x + tt.offsetWidth > r.width) x = e.clientX - r.left - tt.offsetWidth - 14;
  if (y + tt.offsetHeight > r.height) y = e.clientY - r.top - tt.offsetHeight - 14;
  tt.style.left = `${Math.max(0, x)}px`; tt.style.top = `${Math.max(0, y)}px`;
}

// The most recent hover, so a log/linear or mag/phase toggle can refresh an already-open
// tooltip in place instead of leaving it showing the previous mode's value until the mouse
// next moves.
let lastHover = null;
function refreshTip() { if (lastHover) showTip(lastHover.el, lastHover.cvid, lastHover.e); }

function freqAxis(el) {
  el.style.cursor = 'crosshair';
  const cvid = el.querySelector('canvas').id;
  const ov = el.querySelector('.ov');   // the actual chart area -- el also carries the 64px
                                         // left gutter for y-axis labels, which would skew
                                         // every hover position toward the right if used here
  el.addEventListener('pointermove', (e) => {
    if (S.dragging) return;                   // a box-zoom drag is in progress -- not a hover
    const b = ov.getBoundingClientRect();
    const n = S.d.freqs.length, z = S.zoom.p1;
    // ov's right edge IS el's right edge (no gutter there like the 64px one on the left),
    // so without snapping the last bin is only reachable by landing on one exact pixel.
    const seen = snapEdge((e.clientX - b.left) / b.width);   // 0..1 across the visible plot
    const f = z.x0 + seen * (z.x1 - z.x0);               // -> fraction of the full axis
    S.hoverFi = Math.max(0, Math.min(n - 1, Math.round(f * (n - 1))));
    cursors();
    readout();
    previewMode(S.hoverFi);
    showTip(el, cvid, e);
    lastHover = { el, cvid, e };
    // Landing near an actual curve highlights it everywhere (section 2's other plots and
    // legends), same as hovering its legend row or workbench card -- just triggered by the
    // line itself instead.
    const hit = nearestSeries(cvid, specKey(cvid), e.clientY - b.top);
    if (hit !== S.hot) { S.hot = hit; drawSpec(); drawShifts(); markHot(); probeTicks(); fieldGrid(); recoveredGrid(); }
  });
  el.addEventListener('pointerleave', () => {
    S.hoverFi = null;
    lastHover = null;
    if (S.modePreview) previewMode(null);
    cursors(); readout();
    hideTip(el);
    if (S.hot != null) { S.hot = null; drawSpec(); drawShifts(); markHot(); probeTicks(); fieldGrid(); recoveredGrid(); }
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

/* ***** filters *****
   Every step-1 picker is the same control: a search box over a scrollable list. `opts()`
   yields the rows {v, label, count?}, `on(v)` marks a row selected, `pick(v)` handles a
   click. Single-select lists (dataset, sample) light one row; multi-select ones (layout,
   objects) toggle, and an empty selection means "no filter". */
const escRe = (s) => s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
const toggle = (set, v) => (set.has(v) ? set.delete(v) : set.add(v));

/* Two shapes of picker, each showing its state the native way:
   - single-select (dataset, sample): the field RESTS on the current choice as a label and
     only filters while focused (it clears on focus), and that choice is pinned to the top
     of the list -- so you always see what's picked and every other option at once.
   - multi-select (layout, n_objects): the active values sit as removable chips in a strip
     between the field and the list; an empty strip means "all". The list stays put. */
function pickbox(host, { opts, on, pick, multi = false, current, cap = 400, placeholder = 'filter', chips = true, restBlank = false }) {
  host.innerHTML =
    (multi && chips ? '<div class="pbsel"></div>' : '') +   // the active-value chip lane, above the field
    `<input class="pbq" spellcheck="false" autocomplete="off" placeholder="${placeholder}">` +
    '<div class="pblist"></div>';
  const q = host.querySelector('.pbq');
  const selbar = host.querySelector('.pbsel');
  const list = host.querySelector('.pblist');
  const focused = () => document.activeElement === q;

  host._draw = () => {
    if (!multi && !focused()) q.value = restBlank ? '' : (current?.()?.label ?? '');
    const s = focused() ? q.value.trim().toLowerCase() : '';

    if (multi && selbar) {
      const picked = opts().filter((o) => on(o.v));
      selbar.innerHTML = picked.map((o) => {
        const dot = o.hue == null ? '' : `<span class="pbdot" style="--h:${o.hue}"></span>`;
        return `<button class="pbchip" data-v="${o.v}" title="remove">${dot}${o.label}<b>×</b></button>`;
      }).join('');
      selbar.querySelectorAll('.pbchip').forEach((b) =>
        (b.onmousedown = (e) => { e.preventDefault(); pick(b.dataset.v); }));
    }

    let hits = opts().filter((o) => !s || o.label.toLowerCase().includes(s));
    let pinnedCur = false;
    if (!multi && !s) {                         // pin the current choice to the top
      const i = hits.findIndex((o) => on(o.v));
      if (i >= 0) { if (i > 0) hits = [hits[i], ...hits.slice(0, i), ...hits.slice(i + 1)]; pinnedCur = true; }
    }
    list.innerHTML = hits.slice(0, cap).map((o, idx) => {
      const lab = s ? o.label.replace(new RegExp(`(${escRe(s)})`, 'ig'), '<mark>$1</mark>') : o.label;
      const img = o.img ? `<img class="pbimg" src="${o.img}" alt="" loading="lazy">` : '';
      const dot = o.hue == null ? '' : `<span class="pbdot" style="--h:${o.hue}"></span>`;
      // The pinned current row gets a heavy divider under it, marking off "what I'm
      // looking at" from the rest -- which still scrolls beneath it.
      const cls = `pbrow${on(o.v) ? ' on' : ''}${pinnedCur && idx === 0 ? ' pbcur' : ''}`;
      return `<button class="${cls}" data-v="${o.v}">` +
             `${img}${dot}<span class="pbl">${lab}</span>${o.count == null ? '' : `<i>${o.count}</i>`}</button>`;
    }).join('')
      + (hits.length > cap ? `<div class="pbmore">+${hits.length - cap} more — keep typing</div>` : '')
      + (hits.length ? '' : '<div class="pbmore">no match</div>');
    list.querySelectorAll('.pbrow').forEach((b) => (b.onmousedown = (e) => {
      e.preventDefault();            // don't blur the field mid-click and rebuild the list
      pick(b.dataset.v);
      if (!multi) q.blur();
    }));
    // Keep the current pick visible: when the list is resting (not being typed in) and the
    // selection changed under it -- e.g. an arrow-key step or a scatter click -- scroll
    // that row into view so "what's selected" is always on screen at the top-ish.
    if (!multi && !focused()) list.querySelector('.pbrow.on')?.scrollIntoView({ block: 'nearest' });
  };

  q.oninput = host._draw;
  q.onfocus = () => { if (!multi) q.value = ''; host._draw(); };
  q.onblur = () => host._draw();
  host._draw();
}

// Distinct values of `key` across the loaded samples with their counts, commonest first.
const tally = (key) => {
  const c = new Map();
  for (const s of S.all) c.set(s[key], (c.get(s[key]) || 0) + 1);
  return [...c].sort((a, b) => b[1] - a[1]);
};
// A distinct, stable hue per index/value -- the coloured dot beside a layout or count.
const hueOf = (n) => (28 + n * 47) % 360;

// "{box} box, spk {spk}" as two live dropdowns above the scatter -- both are single,
// EXCLUSIVE picks WITHIN the current filtered set (like the box/speaker filters, but a
// single choice standing in for the current sample's own box/speaker, not a set of
// candidates). Switching dataset already has its own picker in the filter row; a box here
// is the metadata box, and one dataset can span several.
function buildBoxSpkTitle() {
  syncBoxSel();
  syncSpkSel();
}

// Options respect EVERY active filter (matches(), not passExcept) -- if the box filter has
// narrowed to one box, this dropdown offers only that box too, not every box that would be
// valid if the box filter were lifted. Re-run on every filter change (applyFilter), not
// just when the picker is first built.
function syncBoxSel() {
  const bx = $('#boxsel');
  if (!bx) return;
  const boxes = [...new Set(matches().map((s) => s.box))].sort();
  const cur = bx.value;
  bx.innerHTML = boxes.map((v) => `<option value="${v}">${v}</option>`).join('');
  if (boxes.includes(cur)) bx.value = cur;   // keep the pick if it's still a valid option
  bx.onchange = () => { S.f.boxes = new Set([bx.value]); applyFilter(); };
}

// Same idea, for speaker: a filter ring narrowed to speaker 1 leaves this dropdown offering
// only speaker 1 too.
function syncSpkSel() {
  const sp = $('#spksel');
  if (!sp) return;
  const spks = [...new Set(matches().map((s) => s.spk))].sort((a, b) => a - b);
  const cur = sp.value;
  sp.innerHTML = spks.map((v) => `<option value="${v}">${v}</option>`).join('');
  if (spks.includes(+cur)) sp.value = cur;   // keep the pick if it's still a valid option
  sp.onchange = () => { S.f.spk = new Set([+sp.value]); applyFilter(); };
}

function buildFilters() {
  // Unfiltered totals -- how many distinct values this picker HAS, not how many currently
  // pass (that's the .ffacets line by the filter chips). Always the whole dataset's count,
  // so it doesn't shrink as you filter and start looking like the picker itself shrank.
  $('#dscount').textContent = `(${S.datasets.length})`;
  $('#boxcount').textContent = `(${new Set(S.all.map((s) => s.box)).size})`;
  $('#objcount').textContent = `(${objectValues().length})`;
  $('#nobjcount').textContent = `(${new Set(S.all.map((s) => s.n)).size})`;
  $('#lycount').textContent = `(${new Set(S.all.map((s) => s.layout)).size})`;
  buildBoxSpkTitle();
  pickbox($('#datasetbox'), {
    // Each row carries a thumbnail of the bare box, so the datasets are told apart at a
    // glance rather than by reading long names.
    opts: () => S.datasets.map((n) => ({
      v: n, label: n, img: `/api/box/${n}.jpg?v=${S.rv}`, count: S.dsCounts[n] })),
    on: (v) => v === S.info.dataset,
    pick: (v) => switchDataset(v),
    current: () => ({ v: S.info.dataset, label: S.info.dataset }),
    placeholder: 'change dataset',
  });
  buildSampleGrid();
  pickbox($('#boxbox'), {
    // The physical box a sample was captured in -- a dataset can span several. Multi-select,
    // same as layout/n_objects: a sample passes if it's in ANY of the picked boxes.
    opts: () => {
      const f = facetTally('box', 'boxes');
      return tally('box').map(([b], i) => ({ v: b, label: b, count: f.get(b) || 0, hue: hueOf(i) }));
    },
    on: (v) => S.f.boxes.has(v),
    pick: (v) => { toggle(S.f.boxes, v); applyFilter(); },
    multi: true, placeholder: 'filter boxes',
  });
  pickbox($('#objectbox'), {
    // Object TYPE, regardless of count. Multi-select: a sample passes if it holds ANY of
    // the picked types. Count is faceted against the other filters.
    opts: () => {
      const f = facetObjTally();
      return objectValues().map((o, i) =>
        ({ v: o, label: o === NONE_OBJ ? 'no-object' : o, count: f.get(o) || 0, hue: hueOf(i * 2 + 1) }));
    },
    on: (v) => S.f.objects.has(v),
    pick: (v) => { toggle(S.f.objects, v); applyFilter(); },
    multi: true, placeholder: 'filter objects',
  });
  pickbox($('#nobjbox'), {
    // Order fixed by the dataset-wide total (tally); the COUNT is faceted -- how many
    // samples with that object count survive the layout + speaker filters.
    opts: () => {
      const f = facetTally('n', 'nobj');
      return tally('n').map(([n]) => n).sort((a, b) => a - b)
        .map((n) => ({ v: n, label: String(n), count: f.get(n) || 0, hue: hueOf(n) }));
    },
    on: (v) => S.f.nobj.has(+v),
    pick: (v) => { toggle(S.f.nobj, +v); applyFilter(); },
    multi: true,
  });
  pickbox($('#layoutbox'), {
    opts: () => {
      const f = facetTally('layout', 'layouts');
      return tally('layout').map(([l], i) =>
        ({ v: l, label: l, count: f.get(l) || 0, hue: hueOf(i) }));
    },
    on: (v) => S.f.layouts.has(v),
    pick: (v) => { toggle(S.f.layouts, v); applyFilter(); },
    multi: true, placeholder: 'filter layouts',
  });
  document.querySelectorAll('[data-all],[data-none]').forEach((b) => (b.onclick = () => {
    const k = b.dataset.all || b.dataset.none;
    // "all" == no constraint (empty set); "none" == an impossible value so nothing matches.
    // Speakers behave like the other two now: an empty set already lights the whole ring.
    S.f[k].clear();
    if (b.dataset.none) S.f[k].add(Symbol('none'));
    applyFilter();
  }));
}

function pickSample(id) {
  if (!S.byId[id]) return;
  // Set directly, not inferred from a mousedown/focusin bubbling up from the clicked card:
  // select(), below, rebuilds #samplegrid's innerHTML SYNCHRONOUSLY as part of choosing a
  // sample (to move the "current" highlight) -- which destroys the very button that was
  // just clicked before the browser gets to natively focus it. Relying on that focus would
  // silently drop the gallery back out of "armed" on every single pick.
  armedZone = 'gallery';
  if (!pass(S.byId[id])) { S.f.boxes.clear(); S.f.objects.clear(); S.f.layouts.clear(); S.f.nobj.clear(); applyFilter(); }
  select(id);
}

// bg and mask are independent checkboxes; bg off with mask also off has nothing left to
// show, so it falls back to the plain photo -- same fallback render.thumb() applies
// server-side if a sample has no mask at all (e.g. an empty box).
const galThumbUrl = (id) => {
  const bg = S.gal.bg || !S.gal.mask;
  return `/api/thumb/${id}.jpg?mask=${S.gal.mask ? 1 : 0}&bg=${bg ? 1 : 0}&v=${S.rv}`;
};

/* The sample list, as a scrollable photo grid (5/row): overhead thumbnail + "Pos x, Spk y
   (id)" caption. select() redraws it on every pick -- from here, the scatter or a workbench
   probe -- so the current sample is always highlighted and scrolled into view.
   #samplebox holds only the search input now, sitting beside #posjump at half width each;
   the grid itself is #samplegrid, a SEPARATE sibling spanning the full row underneath both
   -- nesting it inside #samplebox capped it to #samplebox's own (half) width instead. */
function buildSampleGrid() {
  const host = $('#samplebox'), grid = $('#samplegrid');
  host.innerHTML = '<input class="pbq" spellcheck="false" autocomplete="off" placeholder="search by sample id">';
  const q = host.querySelector('.pbq');
  host._draw = () => {
    const s = q.value.trim().toLowerCase();
    const hits = matches().filter((o) =>
      !s || `${shortId(o.id)} ${o.pos} ${o.spk} ${o.layout}`.toLowerCase().includes(s));
    grid._hits = hits;                 // for the gallery zone's arrow-key stepping, below
    grid.innerHTML = hits.map((o) => `
      <button class="scard${o.id === S.live.sid ? ' on' : ''}" data-id="${o.id}">
        <span class="sclab">Pos ${o.pos}, Spk ${o.spk} (${shortId(o.id)})</span>
        <div class="scimgwrap">
          <img class="scimg" src="${galThumbUrl(o.id)}" alt="" loading="lazy">
          <svg class="spkring" viewBox="0 0 160 120" aria-hidden="true">${
            spkRingDots(o.spk, 160, 120, { r: 4.5, rOn: 7, margin: 10 })}</svg>
        </div>
      </button>`).join('') || '<div class="pbmore">no match</div>';
    grid.querySelectorAll('.scard').forEach((b) => {
      b.onclick = () => pickSample(b.dataset.id);
      // Show where this card's sample sits in the box, without committing to it -- same
      // "hover previews, click picks" pattern as hovering a workbench probe.
      const pos = S.byId[b.dataset.id]?.pos;
      b.onmouseenter = () => hoverGalleryPos(pos);
      b.onmouseleave = () => hoverGalleryPos(null);
    });
    // Only autoscroll when the box isn't being typed in, so picking a sample elsewhere
    // (scatter, a workbench probe) jumps the grid to it without fighting an active filter.
    // Scroll the grid's own scrollTop, not scrollIntoView -- that walks every scrollable
    // ancestor (including the page) and jumps the whole screen to center the card.
    if (document.activeElement === q) return;
    const on = grid.querySelector('.scard.on');
    if (!on) return;
    const gr = grid.getBoundingClientRect(), or = on.getBoundingClientRect();
    if (or.top < gr.top) grid.scrollTop -= gr.top - or.top;
    else if (or.bottom > gr.bottom) grid.scrollTop += or.bottom - gr.bottom;
  };
  q.oninput = host._draw;
  host._draw();
  // The gallery's bg/mask checkboxes only change what each thumbnail's <img> POINTS AT --
  // not which samples are shown, their order, or anything else about the grid -- so they
  // go through here instead of host._draw(), which rebuilds the whole grid's innerHTML and
  // made every card (labels, borders, all of it) flash at once. Even scoped to just the
  // <img> tags, swapping `src` directly still blanks each one while its new URL loads; the
  // first time a bg/mask combination is requested nothing is cached yet, so that fetch is
  // slow enough for the blank to actually be visible. Preloading off-DOM and only swapping
  // `src` on that Image's own load event keeps the OLD thumbnail on screen the whole time.
  grid._updateThumbs = () => {
    grid.querySelectorAll('.scard').forEach((b) => {
      const img = b.querySelector('.scimg');
      if (!img) return;
      const url = galThumbUrl(b.dataset.id);
      const pre = new Image();
      pre.onload = () => { img.src = url; };
      pre.src = url;
    });
  };
  // must match .sgrid's grid-template-columns (style.css) for up/down to land in the
  // right visual row rather than just off by whatever the real column count is.
  const GALLERY_COLS = 5;
  registerZone('gallery', grid, (dx, dy) => {
    const hits = grid._hits || [];
    if (!hits.length) return;
    const at = hits.findIndex((o) => o.id === S.live.sid);
    const next = (at === -1 ? 0 : at) + dx + dy * GALLERY_COLS;
    if (next < 0 || next >= hits.length) return;   // stay put at the edges
    pickSample(hits[next].id);
  });
}

/* Wheel routing for the sample gallery row: scrolling over the grid itself scrolls the
   grid; scrolling anywhere else in the row (the box/position selector to its left, the
   search bars above it, the empty margin around them) scrolls the PAGE instead, so the box
   selector stays put -- "frozen" -- while you're just paging through samples. Driven
   explicitly with scrollTop rather than left to the browser's own overflow/bubbling: this
   row's nested flex layout was letting wheel input land on the grid even with the pointer
   nowhere near it (and, from directly over the grid, sometimes doing nothing at all). */
const drawPickers = () =>
  ['#datasetbox', '#samplebox', '#boxbox', '#objectbox', '#nobjbox', '#layoutbox']
    .forEach((id) => $(id)?._draw?.());

const FILTER_KEYS = ['boxes', 'objects', 'layouts', 'nobj', 'spk'];

/* Per-value chips now live in each filter's own lane (the .pbsel strip above its search
   box), and picked speakers show lit on the ring -- so this shared bar is just the
   summary and the one "clear all" that spans every filter. Fixed to a single line, so
   adding filters never nudges step 2 below it. */
// How many distinct values of each facet the CURRENT matches actually span -- distinct
// from the picker counts above, which tally against the OTHER filters, not this one.
const plural = (c, w) => `${c} ${w}${c === 1 ? '' : 's'}`;
function facetSummary(m) {
  const uniq = (get) => new Set(m.flatMap(get)).size;
  return [plural(S.datasets.length, 'dataset'), plural(uniq((s) => [s.box]), 'box'),
          plural(uniq((s) => s.objects), 'object'),
          plural(uniq((s) => [s.n]), 'n_obj'), plural(uniq((s) => [s.layout]), 'layout')]
    .join(' · ');
}

function renderFilterChips(m) {
  const el = $('#filterchips');
  if (!el) return;
  const n = FILTER_KEYS.reduce(
    (a, k) => a + [...S.f[k]].filter((v) => typeof v !== 'symbol').length, 0);
  const blocked = FILTER_KEYS.some((k) => [...S.f[k]].some((v) => typeof v === 'symbol'));
  const status = !n && !blocked ? 'no filters — every sample in this dataset'
    : blocked ? 'a “none” filter is set — 0 samples'
    : `${n} filter value${n === 1 ? '' : 's'} applied`;
  el.innerHTML = `<span class="fnone">${status}</span>` +
    (n || blocked
      ? '<button class="fchip clr" id="fclear" title="clear all filters">clear filters</button>'
      : '') +
    `<span class="ffacets">${facetSummary(m)}</span>`;
  if (n || blocked) $('#fclear').onclick = () => {
    FILTER_KEYS.forEach((k) => S.f[k].clear());
    applyFilter();
  };
}

function applyFilter() {
  const m = matches();
  drawPickers();
  syncBoxSel();
  syncSpkSel();
  renderFilterChips(m);
  $('#nmatch').textContent = m.length;
  $('#nmatchsub').textContent = `/${S.all.length}`;
  $('#nmatch').title = `${m.length} of ${S.all.length} samples in this dataset pass the filter`;
  buildScatter();
  // Keep the current sample if it still passes; otherwise stay at the same position under
  // the new filter (e.g. a speaker toggle shouldn't also bounce you to a different spot in
  // the box), falling back to the first match only if that position has none. Also fires
  // when the sample is just gone outright (deleted out from under a live view) -- pass()
  // assumes an object with .objects etc, so check existence first rather than fake one.
  const cur = S.byId[S.live.sid];
  if (m.length && (!cur || !pass(cur))) {
    const pos = cur?.pos;
    const atPos = m.filter((s) => s.pos === pos && !s.empty);
    select((atPos[0] || m[0]).id);
  }
  else { renderScatter(); renderSpk(); renderRepeats(); }
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
    // The viewBox (SW x SH) and the element's own box rarely share an aspect ratio, so the
    // default xMidYMid "meet" scaling letterboxes it -- content fills one dimension exactly
    // and is centered with margin on the other. A plain (clientX-b.left)/b.width style
    // mapping ignores that margin, so it was wrong on whichever axis had letterboxing
    // (here, always y: 300x140 vs a 250x78 viewBox) and clicks landed on the wrong point.
    const boxAr = b.width / b.height, vbAr = SW / SH;
    let cw = b.width, ch = b.height, ox = 0, oy = 0;
    if (boxAr > vbAr) { cw = b.height * vbAr; ox = (b.width - cw) / 2; }
    else { ch = b.width / vbAr; oy = (b.height - ch) / 2; }
    const x = ((ev.clientX - b.left - ox) / cw) * SW, y = ((ev.clientY - b.top - oy) / ch) * SH;
    let best = null, bd = 1e9;
    POS().forEach((p, i) => {
      const d = Math.hypot(s.x(s.pts[i][1]) - x, s.y(s.pts[i][0]) - y);
      if (d < bd) { bd = d; best = p; }
    });
    return bd < 6 ? best : null;
  };
  svg.onclick = (e) => { const p = near(e); if (p != null) goPos(p); svg.focus(); };
  registerZone('boxpos', svg, (dx, dy) => {
    const p = neighbour(s, [dx, dy]);
    if (p != null) goPos(p);
  });
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

/* Pick a sample AT this position from within the filtered set -- the scatter selects, it
   does not filter. Prefer one on the current sample's speaker so stepping around the box
   holds the speaker steady; fall back to any filtered sample there. */
function goPos(pos) {
  // Same reasoning as pickSample's armedZone line: set directly rather than relying on
  // svg.focus() (called after this, back in the click handler) surviving whatever
  // select() below rebuilds in the meantime.
  armedZone = 'boxpos';
  const spk = S.byId[S.live.sid]?.spk;
  const at = matches().filter((s) => s.pos === pos && !s.empty);
  const s = at.find((x) => x.spk === spk) || at[0];
  if (s) select(s.id);
}

function renderScatter() {
  const svg = $('#scatter');
  let sel = null;
  svg.querySelectorAll('.pt').forEach((c) => {
    const on = +c.dataset.p === S.byId[S.live.sid]?.pos;
    c.classList.toggle('on', on);
    if (on) sel = c;
  });
  // Positions overlap, and SVG has no z-index -- paint order IS document order, so the
  // current dot was being covered by whichever grey dots happened to come after it.
  // Moving it last puts it on top; the fill is semi-transparent (see .pt.on) so the
  // dots underneath stay visible rather than being blotted out.
  if (sel) sel.parentNode.appendChild(sel);
  $('#poscount').textContent = `${POS().length} positions`;
  // Reflect the current sample's speaker in the dropdown -- box/spk text is now the
  // titles's job (buildBoxSpkTitle), not this count.
  const sp = $('#spksel'), curSpk = S.byId[S.live.sid]?.spk;
  if (sp && curSpk != null) sp.value = String(curSpk);
}

// y_frac has 0 at the BOTTOM (draw_speaker flips it), so invert for SVG.
const SPK = { 1: [1, 0], 2: [1, .7], 3: [.8, 1], 4: [.6, 1], 5: [.4, 1], 6: [.2, 1], 7: [0, .7], 8: [0, 0] };

function buildSpk() {
  const svg = $('#spk'), W = 126, H = 78, M = 20, bw = W - 2 * M, bh = H - 2 * M, r = 8;
  svg.setAttribute('viewBox', `0 0 ${W} ${H}`);
  svg.innerHTML = `<rect class="box" x="${M}" y="${M}" width="${bw}" height="${bh}" rx="3"/>` +
    Object.entries(SPK).map(([id, [xf, yf]]) => {
      const cx = M + xf * bw + (xf === 1 ? 12 : xf === 0 ? -12 : 0);
      const cy = M + (1 - yf) * bh + (yf === 0 ? 12 : yf === 1 ? -12 : 0);
      // The X is only shown for a speaker the whole dataset lacks (renderSpk adds .absent).
      return `<g class="s" data-s="${id}"><circle cx="${cx}" cy="${cy}" r="${r}"/>` +
             `<text x="${cx}" y="${cy}">${id}</text>` +
             `<line class="xm" x1="${cx - r + 2}" y1="${cy - r + 2}" x2="${cx + r - 2}" y2="${cy + r - 2}"/>` +
             `<line class="xm" x1="${cx - r + 2}" y1="${cy + r - 2}" x2="${cx + r - 2}" y2="${cy - r + 2}"/></g>`;
    }).join('');
  svg.onclick = (e) => {
    const g = e.target.closest('.s');
    if (g && !g.classList.contains('absent')) toggleSpk(+g.dataset.s);
  };
}

// The speaker ring is a pure multi-select filter now (like layout / n_objects): a click
// toggles that speaker in or out of the candidate set, it never jumps to a sample. Use
// the scatter or the sample list to actually pick one.
function toggleSpk(spk) {
  toggle(S.f.spk, spk);
  applyFilter();
}

// With 2+ speakers selected, flip which of them is shown at the CURRENT position -- the
// scatter/grid already own "same speaker, different position"; this is the other axis.
function selectedSpks() {
  return [...S.f.spk].filter((v) => typeof v !== 'symbol').sort((a, b) => a - b);
}

// Lives in the SEARCH section, not the filter: a direct pick, not a step, among whichever
// of the selected speakers actually have a sample at the position you're looking at.
function renderSpkAt() {
  const wrap = $('#spkatwrap');
  if (!wrap) return;
  const s = S.byId[S.live.sid], at = S.byPos[s?.pos] || {};
  const avail = selectedSpks().filter((sp) => at[sp]);
  wrap.hidden = avail.length < 2;
  if (avail.length < 2) return;
  $('#spkat').innerHTML = avail.map((sp) =>
    `<button class="pbchip${sp === s.spk ? ' on' : ''}" data-s="${sp}">spk ${sp}</button>`).join('');
  $('#spkat').querySelectorAll('button').forEach((b) =>
    (b.onclick = () => select(at[+b.dataset.s])));
}

/* The 8-speaker ring drawn AROUND the overhead in the "current" panel -- a static readout
   of where this sample's active speaker sat, in the same layout as the filter ring
   (1,2 right / 3-6 top / 7,8 left). */
/* A compact version of the same ring -- dots only, no numbers, no box -- for drawing
   directly over a photo (the gallery thumbnails, the sample viewer's overhead) where
   #scenering's labelled, wide-margin style would eat too much of the image. viewBox is
   given in the CALLER's own units (pixels for the overhead photo, matching comMarks;
   arbitrary for a thumbnail) so circles never distort regardless of the photo's aspect
   ratio -- margin/r/rOn scale with whatever unit that is. */
function spkRingDots(curSpk, w, h, { r, rOn, margin }) {
  // Every one of the box's 8 physical slots, not just the ones this dataset actually used
  // -- filtering those out was making the ring unreadable, not just small: half the layout
  // was simply missing, no dot to make bigger. Unused slots stay in, just dimmed, so the
  // full layout still reads at a glance and the numbers are legible either way.
  const present = new Set(S.all.map((s) => s.spk));
  const bw = w - 2 * margin, bh = h - 2 * margin;
  return Object.entries(SPK).map(([id, [xf, yf]]) => {
    const cx = margin + xf * bw, cy = margin + (1 - yf) * bh;
    const on = +id === curSpk;
    const used = present.has(+id);
    const rr = on ? rOn : r;
    const cls = `spkdot${on ? ' on' : ''}${used ? '' : ' unused'}`;
    return `<g class="${cls}">` +
      `<circle cx="${cx}" cy="${cy}" r="${rr}"/>` +
      `<text x="${cx}" y="${cy}" font-size="${rr * 1.15}">${id}</text></g>`;
  }).join('');
}

function renderSpkRing() {
  const svg = $('#scenering');
  if (!svg) return;
  const spk = S.byId[S.live.sid]?.spk;
  const W = 200, M = 22, bw = W - 2 * M, bh = W - 2 * M;   // square, matches .scenefig
  svg.setAttribute('viewBox', `0 0 ${W} ${W}`);
  svg.innerHTML =
    `<rect class="ring-box" x="${M}" y="${M}" width="${bw}" height="${bh}" rx="4"/>` +
    Object.entries(SPK).map(([id, [xf, yf]]) => {
      const cx = M + xf * bw + (xf === 1 ? 13 : xf === 0 ? -13 : 0);
      const cy = M + (1 - yf) * bh + (yf === 0 ? 13 : yf === 1 ? -13 : 0);
      return `<g class="rs${+id === spk ? ' on' : ''}">` +
             `<circle cx="${cx}" cy="${cy}" r="9"/><text x="${cx}" y="${cy}">${id}</text></g>`;
    }).join('');
}

function renderSpk() {
  const cur = S.byId[S.live.sid], at = S.byPos[cur?.pos] || {};
  const noFilter = !S.f.spk.size;
  // selected speakers as removable chips, in the same lane the other filters use
  const chips = $('#spkchips');
  if (chips) {
    chips.innerHTML = [...S.f.spk].filter((v) => typeof v !== 'symbol').sort((a, b) => a - b)
      .map((s) => `<button class="pbchip" data-s="${s}" title="remove">spk ${s}<b>×</b></button>`).join('');
    chips.querySelectorAll('.pbchip').forEach((b) =>
      (b.onclick = () => toggleSpk(+b.dataset.s)));
  }
  const present = new Set(S.all.map((s) => s.spk));       // speakers the dataset has at all
  $('#spk').querySelectorAll('.s').forEach((g) => {
    const s = +g.dataset.s;
    const absent = !present.has(s);                         // X'd: this dataset has none
    g.classList.toggle('absent', absent);
    g.classList.toggle('on', !absent && (noFilter || S.f.spk.has(s)));   // blue = passes the filter
    g.classList.toggle('cur', s === cur?.spk);              // outlined = current sample's
    // dimmed when the speaker filter excludes it, or this position has no such sample
    g.classList.toggle('off', !absent && ((S.f.spk.size && !S.f.spk.has(s)) || !at[s]));
  });
  renderSpkAt();
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
  registerZone('laser', svg, (dx, dy) => {
    const cur = S.live.laser;
    const i = (cur === 'avg' || cur === 'all') ? 55 : +cur;
    const C = S.info.cols, N = S.info.n_lasers;
    const d = dx + dy * C;
    if (cur === 'avg' || cur === 'all') return setLaser(dy > 0 ? 0 : cur);
    const next = i + d;
    setLaser(next < 0 ? 'avg' : Math.min(N - 1, next));
  });
}

async function setLaser(i) {
  S.live.laser = i;
  // reprobePinned() refetches each pinned probe at its OWN p.laser, so pinned
  // probes must be retargeted to the new laser first or they'd refetch stale data.
  S.probes.forEach((p) => { p.laser = i; });
  await refresh(false);
  await reprobePinned();
  paintAll();
}

function renderGrid() {
  const i = S.live.laser;
  const svg = $('#grid');
  let sel = null;
  svg.querySelectorAll('.ls').forEach((x) => {
    const on = i !== 'avg' && i !== 'all' && +x.dataset.i === +i;
    x.classList.toggle('on', on);
    // Grown in JS, not CSS: the `r` geometry property is not supported everywhere.
    x.setAttribute('r', on ? 5.4 : 3.2);
    if (on) sel = x;
  });
  // The clicked laser carries its own index. One reusable node that moves to the
  // selection, so the other ~100 dots stay unlabelled and the grid stays readable.
  let tag = svg.querySelector('.lsnum');
  if (!sel) { tag?.remove(); }
  else {
    if (!tag) {
      tag = document.createElementNS('http://www.w3.org/2000/svg', 'text');
      tag.setAttribute('class', 'lsnum');
      tag.setAttribute('dy', '1.7');
    }
    tag.setAttribute('x', sel.getAttribute('cx'));
    tag.setAttribute('y', sel.getAttribute('cy'));
    tag.textContent = sel.dataset.i;
    svg.appendChild(tag);        // last child: drawn over the circle, never under it
  }
  $('#grid').querySelector('.avgcell').classList.toggle('on', i === 'avg');
  $('#grid').querySelector('.allcell').classList.toggle('on', i === 'all');
  $('#lasernow').textContent = i;
}

/* ***** empty box: the drift repeats for the current speaker, always shown *****
   Empty-box samples have no object and no spatial COM, so they can't sit on the position
   scatter; they live in their own strip beneath it, clearly labelled, and clicking one
   selects it just like a scatter point. */
function renderRepeats() {
  const el = $('#repeats');
  if (!el) return;
  const spk = S.byId[S.live.sid]?.spk;
  const reps = matches().filter((s) => s.empty && s.spk === spk).sort((a, b) => a.pos - b.pos);
  el.innerHTML = reps.length
    ? reps.map((s) =>
        `<button class="rep${s.id === S.live.sid ? ' on' : ''}" data-id="${s.id}">pos ${s.pos}</button>`).join('')
    : '<span class="repnone">none for this speaker</span>';
  el.querySelectorAll('.rep').forEach((b) => (b.onclick = () => select(b.dataset.id)));
}

/* ***** probes ***** */
function pin() {
  const p = {
    ...S.live, fi: S.fi, hzv: hz(S.fi), id: Date.now(),
    pos: S.byId[S.live.sid].pos,          // the fan is grouped by position
    box: S.info.box,                      // the box this probe was measured in
    dash: DASH[S.live.ch] || [],
    data: S.probe, dataY: S.probeY, mode: S.mode,
    meta: S.byId[S.live.sid],
  };
  S.probes.push(p);
  reshade();
  S.flash = p.id;
  renderProbes(); paintAll(); allMasks();
  clearTimeout(pin._t);
  pin._t = setTimeout(() => { S.flash = null; paintAll(); allMasks(); }, 1400);
}

/* A sample's identity, in one place, so the viewer, the workbench and every probe row
   agree on what they're showing: box/pos/spk/id, then layout + its objects -- or "empty
   box" when it has none. */
function sampleMeta(s) {
  const objs = s.empty ? 'no-object' :
    (Object.entries(s.objcounts || {}).map(([o, n]) => `${n} ${o}${n === 1 ? '' : 's'}`).join(', ') || 'no-object');
  return {
    line1: `Pos <b>${s.pos}</b>, Spk <b>${s.spk}</b> (<b>${shortId(s.id)}</b>) ${s.box}`,
    line2: `${objs} (${s.layout || '—'} layout)`,
  };
}

/* The current sample shows in three places, all fed from here: the metadata list at the
   top of the pinned column (#curmeta), the live row above the pinned probes (#nowrow),
   and -- unchanged -- the one-line header (#summary). laser/channel/frequency is the
   FIRST metadata row because it is the part that moves as you hover the grid or spectrum;
   keeping it on top means the rest of the block never reflows under it. */
function renderNow() {
  const s = S.byId[S.live.sid];
  if (!s) return;
  const laser = S.hoverLaser != null ? S.hoverLaser : S.live.laser;
  const fi = S.hoverFi != null ? S.hoverFi : S.fi;
  const prev = S.hoverLaser != null || S.hoverFi != null;
  const lcf = `L${laser}, C${S.live.ch}, F${fmt(hz(fi))}Hz`;
  const ds = S.info.dataset ?? S.info.box ?? '—';

  const { line1, line2 } = sampleMeta(s);

  const dl = $('#curmeta');
  if (dl) {
    dl.classList.toggle('muted', S.muted);
    dl.innerHTML =
      `<div class="top">${lcf}${prev ? ' <u>preview</u>' : ''}</div>` +
      `<dt>sample</dt><dd>${line1}</dd>` +
      `<dt>objects</dt><dd>${line2}</dd>` +
      `<dt>dataset</dt><dd>${ds}</dd>`;
  }

  const row = $('#nowrow');
  if (row) {
    row.classList.toggle('muted', S.muted);
    row.innerHTML =
      `<span class="sw" style="background:${css('--ink')}"></span>` +
      `<span class="id">${line1}${prev ? ' <u>preview</u>' : ''}<br>${line2}<br>${lcf}<br>DATASET ${ds}</span>`;
  }

  renderViewer();
}

/* The step-1 sample viewer: the current sample shown large, next to the filter/search
   controls. Deliberately duplicates the pinned column's metadata -- here it is the
   headline, there it is a reference beside the workbench. The images and media are only
   re-pointed on a sample/channel change (viewerMedia, from refresh()); this just keeps
   the metadata line current as the laser/frequency hover moves. */
function renderViewer() {
  const s = S.byId[S.live.sid];
  if (!s) return;
  const laser = S.hoverLaser != null ? S.hoverLaser : S.live.laser;
  const fi = S.hoverFi != null ? S.hoverFi : S.fi;
  const ds = S.info.dataset ?? S.info.box ?? '—';
  // one header block: pos/spk/id/box loud, then layout+objects, then laser/channel/freq,
  // then the dataset last
  const { line1, line2 } = sampleMeta(s);
  const top = $('#svtop');
  if (top) top.innerHTML =
    `<div class="svk">${line1}</div>` +
    `<div class="svk2">${line2}</div>` +
    `<div class="svk2">L${laser}, C${S.live.ch}, F${fmt(hz(fi))}Hz</div>` +
    `<div class="svds">DATASET ${ds}</div>`;
  comMarks(s);
}

/* Per-object centre-of-mass, drawn as a crosshair tick over the overhead and segmentation
   images. coms are [row, col] in the overhead-pixel space (same as the smask), so the
   overlay uses that as its viewBox with preserveAspectRatio="none" -- the box has the same
   aspect as the images, so the mapping stays uniform (no stretch). */
function comMarks(s) {
  const [ow, oh] = S.info.overhead || [1, 1];
  const pts = s.coms || [];
  const arm = Math.max(ow, oh) * 0.055;           // crosshair half-length, image px
  const gap = arm * 0.3;                          // centre gap, so the exact point stays open
  const g = pts.map(([r, c], i) =>
    `<line x1="${c - arm}" y1="${r}" x2="${c - gap}" y2="${r}"/>` +
    `<line x1="${c + gap}" y1="${r}" x2="${c + arm}" y2="${r}"/>` +
    `<line x1="${c}" y1="${r - arm}" x2="${c}" y2="${r - gap}"/>` +
    `<line x1="${c}" y1="${r + gap}" x2="${c}" y2="${r + arm}"/>` +
    (pts.length > 1
      ? `<text x="${c + arm * 1.25}" y="${r}" dy="${arm * 0.4}" font-size="${arm}">${i + 1}</text>`
      : '')).join('');
  for (const sel of ['#svphotocom', '#svmaskcom']) {
    const el = $(sel);
    if (!el) continue;
    el.setAttribute('viewBox', `0 0 ${ow} ${oh}`);
    el.setAttribute('preserveAspectRatio', 'none');
    el.innerHTML = g;
  }
  const note = $('#comnote');
  if (note) note.textContent = pts.length
    ? `· ${pts.length} com${pts.length > 1 ? 's' : ''}` : '';
  const spkEl = $('#svphotospk');
  if (spkEl) {
    spkEl.setAttribute('viewBox', `0 0 ${ow} ${oh}`);
    spkEl.setAttribute('preserveAspectRatio', 'none');
    const m = Math.max(ow, oh) * 0.035;
    spkEl.innerHTML = spkRingDots(s.spk, ow, oh, { r: m * 0.55, rOn: m, margin: m * 1.6 });
  }
}

/* Point the viewer's images and players at the current sample. A missing artefact (some
   datasets lack the stimulus copy or the pre-rendered recovered clip) hides just that
   element, or the whole original/recovered block if nothing in it loaded. */
function viewerMedia(sid) {
  setMedia('#svphoto', `/api/scene/${sid}.jpg?mask=0&v=${S.rv}`);
  // the real segmentation mask; empty-box samples have none, so fall back to the tinted
  // overhead and, if even that is missing, hide the figure.
  const mk = $('#svmask'), fig = $('#svmaskfig');
  if (mk) {
    fig?.classList.remove('missing');
    delete mk.dataset.fb;
    mk.onerror = () => {
      if (!mk.dataset.fb) { mk.dataset.fb = '1'; mk.src = `/api/scene/${sid}.jpg?v=${S.rv}`; }
      else fig?.classList.add('missing');
    };
    mk.src = `/api/mask/${sid}.png?v=${S.rv}`;
  }
  // The mp4 already has the audio muxed in (utils/viz.py:make_spectrogram_video), so one
  // element with sound does both -- no separate <audio> fetch/player needed.
  setMedia('#svvidsrc', `/api/source_video/${sid}.mp4?v=${S.rv}`);
  setMedia('#svvidrec', `/api/recovered_video/${sid}.mp4?v=${S.rv}`);
  // Step 4's ground truth is the same original video, shown again bigger next to the
  // recovered-video small multiples -- same source, same reasoning as svvidsrc above.
  setMedia('#gtvid', `/api/source_video/${sid}.mp4?v=${S.rv}`);
  // The chirp's own params (src/data/audio.py), not the sample's -- what band and how much
  // silence padding it was generated with, in place of the generic "played stimulus" label.
  const st = $('#svstim'), sp = S.d?.stim;
  if (st) st.textContent = sp && sp.f_start != null
    ? `${fmt(sp.f_start)}–${fmt(sp.f_end)} Hz · t ${sp.T_start.toFixed(2)}–${sp.T_end.toFixed(2)}s`
    : 'played stimulus';
}

function setMedia(sel, url) {
  const el = $(sel);
  if (!el) return;
  el.classList.remove('missing');
  const block = el.closest('.svm');
  block?.classList.remove('missing');
  el.onerror = () => {
    el.classList.add('missing');
    // hide the original/recovered block only if BOTH its media failed
    if (block && !block.querySelector('video:not(.missing), audio:not(.missing)'))
      block.classList.add('missing');
  };
  el.src = url;
  if (el.tagName === 'VIDEO' || el.tagName === 'AUDIO') el.load();
}

// Open eye / eye with a slash. The slash is always in the markup; CSS shows it only on
// a hidden card, so the icon never has to be rebuilt to change state.
const EYE = '<svg class="eyei" viewBox="0 0 22 14">' +
  '<path class="lid" d="M1 7C3.5 2.5 6.7 1 11 1s7.5 1.5 10 6c-2.5 4.5-5.7 6-10 6S3.5 11.5 1 7Z"/>' +
  '<circle class="pupil" cx="11" cy="7" r="2.7"/>' +
  '<line class="slash" x1="2.5" y1="12.5" x2="19.5" y2="1.5"/></svg>';

function renderProbes() {

  // The header always states the one gesture that fills this column, so it reads the same
  // whether the workbench is empty or full.
  $('#heldh').querySelector('.link').style.display = S.probes.length ? '' : 'none';
  $('#plist').innerHTML = S.probes.map((p) => {
    const m = p.meta;
    // Two lines, not four: the coloured left border already identifies the probe, and
    // everything else (layout, objects, dataset) is one click away in the sample viewer --
    // this card only needs to say WHICH sample and WHICH curve.
    const objs = m.objects.length ? m.objects.join(', ') : 'no-object';
    return `<div class="probe card2${p.hidden ? ' off' : ''}" data-id="${p.id}" style="--c:${col(p)}"
      title="${m.box}  ${shortId(p.sid)}  pos ${m.pos}  spk ${m.spk}  ${m.layout}  ${objs}  --  click to view, click the eye to ${p.hidden ? 'show in' : 'hide from'} plots">
      <img class="thumb mask" src="/api/masks.png?ids=${p.sid}&colors=${probeHex(p)}&v=${S.rv}" alt="">
      <div class="meta">
        <div class="ln1"><b>${shortId(p.sid)}</b><span class="sub${m.box !== S.info.box ? ' foreign' : ''}">${m.box} p${m.pos} s${m.spk}</span>
          <span class="eye" aria-hidden="true">${EYE}</span>
          <button class="x" data-x="${p.id}">×</button></div>
        <div class="ln3">L${p.laser}, C${p.ch}, F${fmt(p.hzv)}Hz</div>
        <div class="ln3 objs">${objs}</div>
      </div>
    </div>`;
  }).join('');
  $('#plist').querySelectorAll('.probe').forEach((el) => {
    const p = S.probes.find((q) => q.id === +el.dataset.id);
    // Click the eye to drop the probe from the plots and back; click anywhere else on the
    // card (not the eye or ×) to jump the viewer to that sample -- the same "make this the
    // current sample" gesture as the scatter dot, the sample search and the position jump.
    el.onclick = (e) => {
      if (e.target.closest('.x')) return;
      if (e.target.closest('.eye')) {
        p.hidden = !p.hidden;
        S.hot = null;
        paintAll(); allMasks();
        return;
      }
      select(p.sid);
    };
    el.onmouseenter = () => {
      if (p.hidden) return;                 // a hidden probe has nothing to bring forward
      S.hot = p; drawSpec(); drawShifts(); markHot(); probeTicks(); fieldGrid(); recoveredGrid();
    };
    el.onmouseleave = () => { S.hot = null; drawSpec(); drawShifts(); markHot(); probeTicks(); fieldGrid(); recoveredGrid(); };
  });
  $('#plist').querySelectorAll('.x').forEach((b) => (b.onclick = (e) => {
    e.stopPropagation();
    S.probes = S.probes.filter((q) => q.id !== +b.dataset.x);
    reshade();                            // the fan re-spaces around what is left
    renderProbes(); paintAll(); allMasks();
  }));
}

/* Every scene currently in play, in one image: the live sample plus each pinned probe,
   each mask in that probe's own color so it reads against the curves and ticks. */
function allMasks() {
  const img = $('#allmasks');
  if (!img) return;
  const seen = new Map();
  if (!S.muted && S.live.sid) seen.set(S.live.sid, '1a1a19');    // live = ink, drawn first
  // Pinned probes overwrite the live entry for the same scene, and are appended after it,
  // so a just-pinned probe shows its own color instead of hiding under the black mask.
  for (const p of shownProbes()) { seen.delete(p.sid); seen.set(p.sid, probeHex(p)); }
  if (!seen.size) { $('#maskfig').hidden = true; return; }
  $('#maskfig').hidden = false;
  const ids = [...seen.keys()].join(','), cols = [...seen.values()].join(',');
  img.src = `/api/masks.png?ids=${ids}&colors=${cols}&v=${S.rv}`;
}

/* Who is who, in the panels where several probes are drawn on the same axes. */
/* The swatch IS the line: same colour, same dash. A "both" entry becomes two swatches --
   x solid, y dashed -- so the key states which line is which instead of leaving it to be
   inferred from the plot. */
const swatch = (c, dash) =>
  `<b class="sw" style="color:${c};${dash?.length ? 'border-top-style:dashed' : ''}"></b>`;

/* The phase control belongs to mag+phi; in re+im there is no phase plot to reshape. */
function phVis() {
  const on = S.specMode === 'magphase';
  // Faded rather than hidden: the control stays in place so the rail does not reflow,
  // and it reads as inapplicable rather than missing.
  for (const id of ['#phlab', '#phmode']) $(id).classList.toggle('dis', !on);
  // pointer-events alone still leaves the buttons tab-focusable, so close that path too.
  $('#phmode').querySelectorAll('button').forEach((b) => { b.disabled = !on; });
}

function legends() {
  const ink = css('--ink');
  const live = S.muted ? '' :
    (S.live.ch === 'both'
      ? `<span data-live="1" data-hot="live">${swatch(ink, DASH.x)}x ${swatch(ink, DASH.y)}y` +
        ` — current  L${S.live.laser}</span>`
      : `<span data-live="1" data-hot="live">${swatch(ink, DASH[S.live.ch])}` +
        `current — L${S.live.laser}  ${S.live.ch}</span>`);
  const rows = shownProbes().map((p) => {
    const c = col(p), tail = `${fmt(p.hzv)} Hz  ${shortId(p.sid)}  L${p.laser}`;
    // The row text carries the probe's own colour, so the key reads without cross-checking
    // the swatch.
    return p.dataY
      ? `<span data-id="${p.id}" style="color:${c}">${swatch(c, DASH.x)}x ${swatch(c, DASH.y)}y  ${tail}</span>`
      : `<span data-id="${p.id}" style="color:${c}">${swatch(c, p.dash)}${tail}  ${p.ch}</span>`;
  }).join('');
  const prev = S.preview
    ? `<span data-hot="preview"><b style="color:${css('--accent')}"></b>preview — L${S.hoverLaser}</span>` : '';
  $('#siglegend').innerHTML = S.probes.length || S.preview ? live + rows + prev : '';
  // The big mode plot always overlays every probe, so the legend always applies.
  $('#modelegend').innerHTML = S.probes.length ? live + rows : '';
  for (const el of [$('#siglegend'), $('#modelegend')]) {
    el.querySelectorAll('span[data-id]').forEach((sp) => {
      sp.style.cursor = 'pointer';
      sp.onmouseenter = () => {
        S.hot = S.probes.find((q) => q.id === +sp.dataset.id);
        drawSpec(); drawShifts(); markHot(); probeTicks(); fieldGrid(); recoveredGrid();
      };
      sp.onmouseleave = () => { S.hot = null; drawSpec(); drawShifts(); markHot(); probeTicks(); fieldGrid(); recoveredGrid(); };
    });
  }
  markHot();
}

function markHot() {
  for (const el of [$('#siglegend'), $('#modelegend')]) {
    el.classList.toggle('dim', !!S.hot);
    el.querySelectorAll('span').forEach((sp) => sp.classList.toggle('hot', !!S.hot && (
      sp.dataset.hot ? sp.dataset.hot === S.hot : +sp.dataset.id === S.hot?.id)));
  }
}

/* ***** unified arrow-key navigation *****
   Every "steppable" control (the sample gallery, the box/position picker, each signal-2
   toggle group, the laser grid, the peak list) registers a zone: a name, its element, and a
   step(dx, dy) function. Priority: whichever zone the mouse is PHYSICALLY hovering wins; off
   every zone -- most often because you're hovering a plot, which isn't one -- arrows fall
   back to whichever zone was last clicked (armedZone). armedZone starts on the gallery, so
   arrows always do SOMETHING sane before you've clicked anything.
   Hovering only ever sets/clears hoverZone; it never touches armedZone, so mousing over a
   plot on your way to the gallery doesn't un-arm whatever you'd last clicked -- exactly the
   "hovering a plot should still refer to the last thing I clicked" rule this exists for. */
const ZONES = {};
const zoneWired = new WeakSet();
let hoverZone = null, armedZone = 'gallery';

function registerZone(name, el, step) {
  if (!el) return;
  ZONES[name] = step;                 // refreshed every call -- cheap, keeps closures current
  if (zoneWired.has(el)) return;      // listeners attached once per element, ever
  zoneWired.add(el);
  el.addEventListener('pointerenter', () => { hoverZone = name; });
  el.addEventListener('pointerleave', () => { if (hoverZone === name) hoverZone = null; });
  el.addEventListener('mousedown', () => { armedZone = name; });
  el.addEventListener('focusin', () => { armedZone = name; });
}

/* One player per pinned probe, so you can compare what they sound like. */
/* ***** wiring ***** */
function seg(id, fn) {
  const el = $(id);
  const buttons = () => [...el.querySelectorAll('button')];
  const pick = (v) => {
    buttons().forEach((x) => x.classList.toggle('on', x.dataset.v === v));
    fn(v);
  };
  el.onclick = (e) => {
    const b = e.target.closest('button');
    if (b) pick(b.dataset.v);
  };
  // Arrow-key cycling through this group's own options, in the order they're drawn --
  // vertical arrows have no meaning for a single row of buttons, so only dx does anything.
  registerZone(id.slice(1), el, (dx) => {
    if (!dx) return;
    const bs = buttons(), cur = bs.findIndex((b) => b.classList.contains('on'));
    const next = bs[Math.max(0, Math.min(bs.length - 1, (cur === -1 ? 0 : cur) + dx))];
    if (next) pick(next.dataset.v);
  });
}

// Always-visible play/pause + seek, but as our OWN bar sitting below the video -- native
// <video controls> overlays the bottom of the picture itself, which would cover part of
// the spectrogram; a separate bar underneath never blocks anything.
const fmtTime = (s) => `${Math.max(0, s || 0).toFixed(2)}s`;

function wireVideoControls(wrap) {
  const v = wrap.querySelector('video'), btn = wrap.querySelector('.vidplay'),
        rng = wrap.querySelector('.vidseek'), time = wrap.querySelector('.vidtime');
  let seeking = false;
  const setIcon = () => { btn.textContent = v.paused ? '▶' : '❚❚'; };
  const setTime = () => { time.textContent = `${fmtTime(v.currentTime)} / ${fmtTime(v.duration)}`; };
  btn.onclick = () => (v.paused ? v.play() : v.pause());
  v.addEventListener('play', setIcon);
  v.addEventListener('pause', setIcon);
  v.addEventListener('loadedmetadata', () => { rng.max = v.duration || 0; setTime(); });
  v.addEventListener('timeupdate', () => { if (!seeking) rng.value = v.currentTime; setTime(); });
  rng.addEventListener('pointerdown', () => { seeking = true; });
  rng.addEventListener('pointerup', () => { seeking = false; v.currentTime = +rng.value; });
  rng.addEventListener('input', () => { v.currentTime = +rng.value; setTime(); });
  setIcon(); setTime();
}
// The gallery grid's height should match the viewer's, but the viewer must never be
// influenced back (a CSS align-items:stretch tried this and could balloon both columns --
// see .s1cols in style.css; a flex chain from .s1search down to #samplegrid tried it next
// and never actually bounded anything, since .s1cols' align-items:start means the two
// columns never stretch to match each other in the first place -- see .sgrid in style.css).
// A ResizeObserver on the viewer, setting #samplegrid's height directly, is one-directional
// (correct regardless of what changes the viewer's height: images loading, content
// changing, window resize) and self-contained (nothing between here and the grid has to
// cooperate for it to work).
function wireViewerHeightSync() {
  const viewer = $('.s1viewer'), grid = $('#samplegrid');
  if (!viewer || !grid || typeof ResizeObserver === 'undefined') return;
  const sync = () => { grid.style.height = `${viewer.getBoundingClientRect().height}px`; };
  new ResizeObserver(sync).observe(viewer);
  sync();
}

function videoControls(sel) { const w = $(sel)?.closest('.vidwrap'); if (w) wireVideoControls(w); }

function wire() {
  buildOverlays();
  wireViewerHeightSync();
  videoControls('#svvidsrc'); videoControls('#svvidrec'); videoControls('#gtvid');
  document.querySelectorAll('#p1, #p2').forEach((el) => freqAxis(el.parentElement));
  ['p1', 'p2', 'p3'].forEach((id) => zoomable($('#' + id).parentElement, id));
  $('#sigreset').onclick = () => resetZoom();
  $('#summary').onclick = () => $('.s1frow')?.classList.toggle('hid');

  // The scatter is the primary position picker (it is spatial); this is just a jump box
  // for when you already know the number. It searches, it does not filter, so a position
  // with no sample in the current filter flashes red rather than widening the set.
  const pj = $('#posjump');
  pj.onchange = () => {
    const n = +pj.value;
    // Land back on the scatter, not the text field -- Enter picked the position, arrow
    // keys now belong to it (same as clicking a dot: see buildScatter's onkeydown).
    if (POS().includes(n)) { goPos(n); pj.value = ''; $('#scatter').focus(); }
    else { pj.classList.add('miss'); setTimeout(() => pj.classList.remove('miss'), 600); }
  };

  // The gallery's own view toggles -- through #samplegrid's current _updateThumbs
  // (reassigned on every buildSampleGrid(), e.g. a dataset switch, not a reference captured
  // here), which only touches each thumbnail's image, not the whole grid.
  $('#galbg').onchange = (e) => { S.gal.bg = e.target.checked; $('#samplegrid')._updateThumbs(); };
  $('#galmask').onchange = (e) => { S.gal.mask = e.target.checked; $('#samplegrid')._updateThumbs(); };

  seg('#ch', (v) => { S.live.ch = v; reviewChannel(); });
  seg('#specmode', (v) => { S.specMode = v; phVis(); drawSpec(); buildPeaks(); peakList(); cursors(); axes(); refreshTip(); });
  seg('#phmode', (v) => { S.phmode = v; drawSpec(); axes(); refreshTip(); });
  // axes() too: lin/log swaps the y label between |FFT| and log |FFT|, and without it
  // the label kept whatever text the previous mode left behind.
  seg('#speclog', (v) => { S.log.spec = +v; drawSpec(); buildPeaks(); peakList(); cursors(); axes(); refreshTip(); });
  seg('#kind', (v) => { S.kind = v; reviewChannel(); });
  wheel($('#fkey2d'));                    // the square 2-D key, drawn once
  seg('#modeview', (v) => { S.modeview = v; viewCtls(); drawMode(); });
  $('#fieldbg').onchange = (e) => {
    S.fieldbg = e.target.checked;
    $('#modekey').hidden = !S.fieldbg;      // the key only means something with the fill on
    drawMode();
  };

  $('#clear').onclick = () => { S.probes = []; renderProbes(); paintAll(); allMasks(); };

  // Both sliders read out their value beside the label, in the same `.cl em` slot the
  // laser and peak counts already use, so the rail keeps one style of readout.
  const setAsize = (v) => {
    S.asize = v;
    $('#asize').value = v;
    $('#asizenow').textContent = `${v.toFixed(1)}x`;
    drawMode();
  };
  $('#asize').oninput = (e) => setAsize(+e.target.value);
  /* One frame index drives everything: play advances it, the slider sets it, and both
     land in the same place -- so pausing leaves the scrubber exactly where the eye was. */
  const setFrame = (i) => {
    S.frame = ((i % NFRAME) + NFRAME) % NFRAME;   // wrap: the mode is periodic
    $('#frame').value = S.frame;
    // 1-based: reads as "where in the loop". The label is mono with tabular figures, so
    // the width does not jump as the counter rolls past 9 thirty times a second.
    $('#framenow').textContent = `${S.frame + 1}/${NFRAME}`;
    drawMode();
  };
  const setPlay = (on) => {
    S.anim = on;
    $('#play').classList.toggle('on', !!S.anim);
    $('#play').textContent = S.anim ? '\u275a\u275a pause' : '\u25b6 play';
  };
  let animSeq = 0;              // pause/play faster than a frame would otherwise stack loops
  $('#play').onclick = () => {
    setPlay(!S.anim);
    if (!S.anim) return;
    // rAF is display-rate, which would spin the mode far too fast; step on a fixed
    // interval instead so the speed does not depend on the monitor.
    const mine = ++animSeq;
    let last = 0;
    const tick = (ts) => {
      if (!S.anim || mine !== animSeq) return;
      if (ts - last > 33) { last = ts; setFrame(S.frame + 1); }
      requestAnimationFrame(tick);
    };
    requestAnimationFrame(tick);
  };
  // Dragging is a manual scrub: take over from the animation rather than fighting it.
  $('#frame').oninput = (e) => { setPlay(false); setFrame(+e.target.value); };
  // Arrow keys nudge it, the same way the range input already handles them natively;
  // this only has to add the wrap at the two ends.
  $('#frame').onkeydown = (e) => {
    const d = { ArrowLeft: -1, ArrowDown: -1, ArrowRight: 1, ArrowUp: 1 }[e.key];
    if (!d) return;
    e.preventDefault();
    setPlay(false);
    setFrame(S.frame + d);
  };
  // Seed both readouts from the starting state: they would otherwise sit empty until the
  // first drag. Written straight to the nodes -- calling the setters here would trigger a
  // drawMode() before there is anything to draw.
  $('#asizenow').textContent = `${S.asize.toFixed(1)}x`;
  $('#framenow').textContent = `${S.frame + 1}/${NFRAME}`;
  viewCtls();          // the rail matches S.modeview from the start, not just after a click

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
    const K = { ArrowLeft: [-1, 0], ArrowRight: [1, 0], ArrowUp: [0, -1], ArrowDown: [0, 1] }[e.key];
    if (K) {
      // Shift+arrows always walk the peaks, and plain arrows do too once you've clicked a
      // peak marker ON THE PLOT (S.peakMode) -- that's a click on neither #pklist nor any
      // other registered zone (the plot itself isn't one), so it's handled as its own
      // override, ahead of the unified dispatch below. Left/right walk frequency order
      // (what makes sense on a spectrum plot); up/down still walk rank, same as the list.
      const onPeak = S.peakMode && (S.probe?.peaks || []).includes(S.fi);
      if (e.shiftKey || onPeak) {
        e.preventDefault();
        return K[1] === 0 ? stepPeak(K[0]) : stepGrid(0, K[1]);
      }
      // Unified hover/click navigation (see registerZone, above seg()): whichever zone the
      // mouse is physically over wins; off every zone -- hovering a plot, most often --
      // arrows fall back to whichever zone was last clicked (armedZone starts on gallery,
      // so arrows always do something sane before you've clicked anything at all).
      const zone = ZONES[hoverZone] ? hoverZone : armedZone;
      if (zone && ZONES[zone]) { e.preventDefault(); ZONES[zone](K[0], K[1]); }
      return;
    }
    if (e.target.closest('.plot, svg')) return;   // panel-local keys win
    if (e.key === 'p') return pin();
    if (e.key === 'x' || e.key === 'y' || e.key === 'a') {
      const v = e.key === 'a' ? 'avg' : e.key;
      S.live.ch = v;
      $('#ch').querySelectorAll('button').forEach((b) => b.classList.toggle('on', b.dataset.v === v));
      return refresh();
    }
  });

  addEventListener('resize', () => paintAll());
}
