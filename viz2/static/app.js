'use strict';
const $ = (s) => document.querySelector(s);
const api = (u) => fetch(u).then((r) => r.json());

const S = {
  all: [], info: null, byId: {}, byPos: {},
  live: { sid: null, ch: 'avg', laser: 'avg', fi: null },
  d: null, probe: null, mode: null, rng: {},
  probes: [], slots: {}, hot: null,
  fi: null, hoverFi: null,
  log: { spec: 0 },        // FFT magnitude linear by default
  specMode: 'magphase', phmode: 'cos', fieldbg: false, kind: 'clean',
  ref: 'none',       // phase reference: none | avg | laser index (data.norm on the server)
  modeview: 'quiver',
  cam: { az: -60, el: 25 },  // the one 3-D camera every surface shares (degrees); drag rotates it
  view: 'photo',     // photo | overlay | mask: how every sample photo is drawn (see photoFig)
  sort: 'pos', sortDir: 1,   // the gallery's order (gallerySort)
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

/* The gallery's order: S.sort's key, S.sortDir up (1) or down (-1); ties always fall back
   to position, speaker, id ascending, so equal keys stay in a stable, readable order. */
// com is [row, col] (avg over objects; -1 for an empty box). area arrives on demand (sortBy).
const SORTS = { pos: (s) => s.pos, spk: (s) => s.spk, n: (s) => s.n,
                object: (s) => s.objects.join(', '), layout: (s) => s.layout, id: (s) => s.id,
                cx: (s) => s.com[1], cy: (s) => s.com[0], area: (s) => s.area ?? -1, box: (s) => s.box };
const cmp = (a, b) => (typeof a === 'number' ? a - b : String(a).localeCompare(String(b)));
// mask area -1 (no image) stays first whichever way the order runs
const noImg = (s) => S.sort === 'area' && (s.area ?? -1) < 0;
const gallerySort = (a, b) => noImg(b) - noImg(a) ||
  S.sortDir * cmp(SORTS[S.sort](a), SORTS[S.sort](b)) ||
  a.pos - b.pos || a.spk - b.spk || cmp(a.id, b.id);

/* Faceted counts: every filter list shows its row counts measured against the OTHER
   filters, not itself. So with "1 object" chosen, the layout list shows how many 1-object
   samples each layout has (and hides layouts with none), and picking a speaker updates
   all of them. `skip` is the facet whose own constraint is dropped. */
function passExcept(s, skip) {
  return (skip === 'boxes' || !S.f.boxes.size || S.f.boxes.has(s.box)) &&
         (skip === 'objects' || !S.f.objects.size || hasObj(s)) &&
         (skip === 'layouts' || !S.f.layouts.size || S.f.layouts.has(s.layout)) &&
         (skip === 'nobj' || !S.f.nobj.size || S.f.nobj.has(s.n)) &&
         (skip === 'spk' || !S.f.spk.size || S.f.spk.has(s.spk));
}
// Drop rows the other filters leave empty, so a list only offers values that would
// match something -- except rows already picked, which stay so they can be unpicked.
const live = (rows, on) => rows.filter((o) => o.count || on(o.v));
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

const hz = (i) => (S.d ? S.d.freqs[i] : 0);
const fmt = (f) => (f >= 100 ? f.toFixed(0) : f.toFixed(1));
/* Colour means ONE thing everywhere in viz2: which saved probe.
     sample    -> one of 10 BASE colours, by the order you saved it: the 1st saved sample
                  gets PALETTE[0], the 2nd PALETTE[1], ... -- consecutive slots jump
                  across the wheel, so samples saved one after another never look alike.
                  Every view of that sample uses it: its curves, mode tile, workbench card,
                  map outline, gallery ring, and its objects' outlines in the photos.
     channel   -> DASH       only x / y / avg, so a dash pattern stays readable
   Laser is left to the legend: too many values for any visual channel. Unsaved samples
   stay the one segmentation green (--seg).
   Slots: a new sample takes the slot after the newest one still saved (removing a probe
   leaves the others' colours alone; clearing them all starts over at the first); the 11th
   wraps to the first colour. */
// NOT hue-ordered: each entry is far around the wheel from the one before it (blue,
// orange, teal, magenta, ...), each tuned to read on the light ground at a similar weight.
// No pure green: that is --seg, the colour of an UNSAVED sample.
const PALETTE = ['#2f6fd0', '#e0701a', '#13a08a', '#c4408e', '#8a9a14',
                 '#4646a6', '#d19a00', '#1a9bc2', '#d43d3d', '#8a4fc7'];
const sampleKey = (p) => `${p.ds}|${p.sid}`;
function slotFor(key) {
  if (S.slots[key] == null) {
    const used = Object.values(S.slots);
    S.slots[key] = used.length ? Math.max(...used) + 1 : 0;
  }
  return S.slots[key];
}
/* Called whenever the probe set changes: gives every probe its sample's colour, frees the
   slots of samples no longer saved, and repaints the photos that show those colours. */
function reshade() {
  const saved = new Set(S.probes.map(sampleKey));
  for (const k of Object.keys(S.slots)) if (!saved.has(k)) delete S.slots[k];
  for (const p of S.probes) p.base = p.color = PALETTE[slotFor(sampleKey(p)) % PALETTE.length];
  queueMicrotask(() => {
    $('#samplegrid')._updateThumbs?.();
    if (OM.sid) showMasks(OM.sid);
  });
}
// this dataset's sample sid's saved probe, if it is saved (a sample is saved at most once)
const savedOf = (sid) => S.probes.find((p) => p.ds === S.info.dataset && p.sid === sid);
// its saved-probe colour (hex, no #), or '' if it isn't saved
const sampleHex = (sid) => savedOf(sid)?.base.slice(1) || '';

const DASH = { avg: [], x: [], y: [7, 6], both: [] };
// The heatmap paints one plane, and avg IS a plane, so only "both" needs replacing there.
const planeCh = (c) => (c === 'both' ? 'x' : c);
const col = (p, a = 1, k = null) => (k == null ? p.color : shade(p.color, k)) +
  (a >= 1 ? '' : Math.round(a * 255).toString(16).padStart(2, '0'));
/* A row's frequencies in shades of its sample's colour: same hue, a clear lightness step
   per tile -- the 1st (strongest) darkest, the last lightest (reversed on a dark page).
   The steps spread over the WHOLE range for however many tiles the row has: a fixed
   ladder left the usual 3-tile row on its three darkest rungs, the strongest at L=.28,
   which read as black. Mid-tone to light keeps every shade visibly the sample's colour.
   Lightness, not opacity (faded already means "not in the big plot") nor hue (hue IS
   the sample's identity). */
const SHADE_LO = 0.40, SHADE_HI = 0.68;
const NTOP = 3;          // peaks per sample in Modes: one row of three tiles
// t in [0, 1]: the tile's place along its row
function shade(hex, t) {
  const [r, g, b] = [1, 3, 5].map((i) => parseInt(hex.slice(i, i + 2), 16) / 255);
  const hi = Math.max(r, g, b), lo = Math.min(r, g, b), d = hi - lo;
  const h = !d ? 0 : hi === r ? ((g - b) / d + 6) % 6 : hi === g ? (b - r) / d + 2 : (r - g) / d + 4;
  const s = d ? d / (1 - Math.abs(hi + lo - 1)) : 0;
  const dark = parseInt(css('--ink').slice(1, 3), 16) > 128;
  const l = SHADE_LO + (SHADE_HI - SHADE_LO) * (dark ? 1 - t : t), c = (1 - Math.abs(2 * l - 1)) * Math.min(s, 0.85);
  const x = c * (1 - Math.abs((h % 2) - 1)), m = l - c / 2;
  const rgb = [[c, x, 0], [x, c, 0], [0, c, x], [0, x, c], [x, 0, c], [c, 0, x]][Math.floor(h) % 6];
  return '#' + rgb.map((v) => Math.round((v + m) * 255).toString(16).padStart(2, '0')).join('');
}
// a tile's shade: its place along its row, 0..1 (the cursor's off-row tile keeps the plain colour)
const rowShade = (p, fi) => {
  const k = p.row?.findIndex((x) => x.fi === fi);
  return k >= 0 ? k / Math.max(1, p.row.length - 1) : null;
};

/* One full oscillation of the mode, quantised into frames. The slider spans exactly one
   period, so scrubbing off either end wraps rather than dead-ending. */
const NFRAME = 60;
const probeHex = (p) => p.color.slice(1);
const css = (n) => getComputedStyle(document.body).getPropertyValue(n).trim();

/* ***** boot ***** */
(async function () {
  loadPayload(await api('/api/samples'));
  wire(); phVis();
  await firstSample();
  // Chained, not setInterval: the next poll is only scheduled once this one answers, so a
  // slow server can't accumulate a backlog of overlapping /api/samples requests per tab.
  (async function loop() {
    try { await poll(); } catch (e) { /* server restarting -- just try again */ }
    setTimeout(loop, 500);
  })();
})();

// Live: watch_and_process.py drops new samples (and new experiment dirs) under the watch
// path while it's running. /api/samples rescans both each call; only rebuild the UI when
// something actually moved. rv (not sample/dataset COUNT) is the right signal -- it's
// bumped server-side on any add/remove/reprocess, including a sample deleted and
// recaptured under the same id, which leaves the count unchanged but the content stale.
async function poll() {
  // passing our rv lets the server answer "unchanged" with just {rv}, not the full list
  const j = await api(`/api/samples?rv=${encodeURIComponent(S.rv)}`);
  if (j.rv !== S.rv && j.samples) loadPayload(j);
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
  // a box without a laser photo can't stay in the box view
  if (isBox() && !S.info.laser_view) $('#modeview [data-v="quiver"]').click();
  viewCtls();
  buildFilters(); buildScatter(); buildSpk(); buildGrid(); buildRefGrid(); sizeModePanel(); applyFilter();
  // new samples / another dataset: a "mask area" order needs their areas too
  ensureAreas().then((got) => got && $('#samplegrid')._draw());
}

/* Mask areas aren't in the sample list (each means reading a mask), so they're fetched
   only while the gallery sorts by them, once per payload. true if it fetched. */
async function ensureAreas() {
  const key = `${S.info.dataset}|${S.rv}`;
  if (S.sort !== 'area' || S.areasKey === key) return false;
  S.areasKey = key;
  const a = await api('/api/areas');
  for (const [sid, v] of Object.entries(a)) if (S.byId[sid]) S.byId[sid].area = v;
  return true;
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
  S.muted = false;                          // picking a sample always shows it, even after Esc
  // the current sample as a row owner, like a saved probe: its mode cache (modeAt) and its
  // row of tiles (filled in once its curve arrives), both handed on by pin()
  S.liveP = { sid, ds: S.info.dataset, get color() { return css('--ink'); } };
  // Cheap, sid-only feedback NOW: the rest below waits on /api/sample + probe + mode, and
  // holding the box-position dot until then meant a burst of arrow presses showed nothing
  // until the first request chain landed -- and then jumped straight to the last position.
  $('#samplegrid')?._highlight?.();         // move the highlight to the chosen card
  renderScatter();
  showPhotos(sid);
  showMeta();
  showMasks(sid);
  const d = await api(`/api/sample/${sid}`);
  if (S.live.sid !== sid) return;           // superseded by a newer pick while in flight
  S.d = d;
  // A new sample resonates at its OWN frequencies, so carrying the previous bin over
  // usually lands on noise. Clearing it makes refresh() snap to this sample's strongest
  // peak -- the most informative place to start, and the mode there is worth looking at.
  S.fi = null;
  S.pkWho = 'live';                       // a new sample: its curve is the one you steer
  await refresh();
}

async function refresh(paint = true) {
  const { sid, ch } = S.live;
  // "all" is a rendering of every laser, not a laser: the curves behind it stay the
  // average, so the peak list and the readout still mean something.
  const laser = S.live.laser === 'all' ? 'avg' : S.live.laser;
  viewerMedia(sid);
  // "both" is x and y in one axes. The endpoint serves one channel, so ask twice and let
  // the dash pattern -- the channel's own visual channel already -- tell them apart.
  const q = (c) => api(`/api/probe/${sid}?ch=${c}&laser=${laser}&kind=${S.kind}&ref=${S.ref}`);
  let px, py = null;
  if (ch === 'both') [px, py] = await Promise.all([q('x'), q('y')]);
  else px = await q(ch);
  // A newer select() owns the view now; painting this sample's probe under its id would
  // mix two samples' data (and flash the older one) -- let the newer chain paint instead.
  if (S.live.sid !== sid) return;
  S.probe = px; S.probeY = py;
  // its three strongest peaks, none shown until you pick some
  S.liveP.row ??= byMag(px).slice(0, NTOP).map((fi) => ({ fi, on: false, top: true }));
  // NOT S.peakMode = true here: every sample switch (gallery, box position, anywhere)
  // routes through here with S.fi reset to null, so unconditionally arming peak-mode on
  // every single one hijacked the very next arrow press into walking peaks -- via the
  // keydown handler's onPeak check, which runs BEFORE the unified zone dispatch -- no
  // matter which zone (gallery, boxpos, ...) the user actually meant. Peak mode is armed
  // by actually interacting with a peak (clicking one, in peakList()/buildPeaks()); Shift
  // still always walks peaks regardless, unaffected by this.
  if (S.fi == null) S.fi = strongestPeak() ?? Math.floor(S.d.freqs.length / 3);
  await loadMode();
  if (S.live.sid !== sid) return;
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
    const get = (c) => api(`/api/probe/${p.sid}?ch=${c}&laser=${laser}&kind=${kind}&ref=${S.ref}`);
    if (ch === 'both') [p.data, p.dataY] = await Promise.all([get('x'), get('y')]);
    else { p.data = await get(ch); p.dataY = null; }
    p.ch = ch;
    p.dash = DASH[ch] || [];
  }));
}

const reviewChannel = async () => { await refresh(false); await reprobePinned(); paintAll(); };

// The phase reference reaches the modes too; ch only matters when a reference is set.
const modeQ = () => (S.ref === 'none' ? '' : `&ref=${S.ref}&ch=${S.live.ch}`);

async function loadMode() {
  const sid = S.live.sid, m = await api(`/api/mode/${sid}?fi=${S.fi}${modeQ()}`);
  if (S.live.sid === sid) S.mode = m;
}

function paintAll() {
  const s = S.byId[S.live.sid];
  buildPeaks(); peakList(); drawSpec(); drawShifts(); drawMode(); cursors(); axes(); legends();
  readout();
  renderScatter(); renderSpk(); renderGrid(); renderProbes(); renderRepeats();
  $('#samplegrid')._mark?.();
}

/* ***** curves: live probe + pinned probes in one axes ***** */
// A workbench probe can be toggled off without unpinning it -- everything that draws a
// probe reads this, so hiding it clears the curve, the mode arrows, its ticks and its
// mask in one go while the card stays put.
const shownProbes = () => S.probes.filter((p) => !p.hidden);
// The live (black) sample is hidden when Esc muted it, or while it IS a shown saved probe --
// the same curve, so it reads as saved (the probe's colour), not twice. Another sample,
// laser or channel, or a laser hover preview, brings it back.
const liveHidden = () => S.muted || (S.hoverLaser == null &&
  shownProbes().some((p) => p.ds === S.info.dataset && p.sid === S.live.sid &&
    p.laser === S.live.laser && p.ch === S.live.ch));

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
  const live = liveHidden() ? []
    : [{ p: null, who: 'live', d: S.probe, c: css('--ink'), w: liveW, dim: liveDim, dash: DASH.x },
       ...(S.probeY ? [{ p: null, who: 'live', d: S.probeY, c: css('--ink'), w: liveW, dim: liveDim, dash: DASH.y }] : [])];
  const previewDim = othersDim() && !isHot('preview');
  return [
    ...back,
    ...live,
    ...front,
    ...(S.preview && !liveHidden()
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
      `/api/heat/${sid}.png?ch=${planeCh(ch)}&q=${q}&kind=${S.kind}&ref=${S.ref}&v=${S.rv}`;
  });
  const rs = await Promise.all(qs.map((q) =>
    api(`/api/heatrange/${sid}?ch=${planeCh(ch)}&q=${q}&kind=${S.kind}&ref=${S.ref}`)));
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
  weakBands();
  drawHeat();
}

/* Grey bands over the bins where the phase reference is weak (|r| < 0.2 x the median
   laser's |x|, data.weak_runs): its phase there is noise, so the normalized phase is too.
   "both" draws two channels, so a bin is grey if EITHER channel's reference is weak. */
function weakBands() {
  const runs = [...(S.probe?.weak || []), ...(S.probeY?.weak || [])];
  const n = S.d ? S.d.freqs.length : 0;
  const nw = new Set(runs.flatMap(([a, b]) => Array.from({ length: b - a + 1 }, (_, i) => a + i))).size;
  $('#refnow').textContent = S.ref === 'none' ? '' :
    `${S.ref}${n ? ` · ${Math.round((100 * nw) / n)}% weak` : ''}`;
  for (const id of OVS) {
    const svg = $(id);
    let g = svg.querySelector('.weak');
    if (!g) { g = document.createElementNS('http://www.w3.org/2000/svg', 'g');
              g.setAttribute('class', 'weak'); svg.prepend(g); }
    // half a bin either side, so a lone weak bin still shows as a sliver
    g.innerHTML = runs.map(([a, b]) => {
      const x0 = Math.max(0, specX(a - 0.5)), x1 = Math.min(100, specX(b + 0.5));
      return x1 > x0 ? `<rect x="${x0}%" width="${x1 - x0}%" y="0" height="100%"/>` : '';
    }).join('');
    const why = S.ref === 'real' ? 'no dominant phase axis: |Σ x²| < 0.2 · Σ |x|²'
                                  : 'weak phase reference: |r| < 0.2 × median laser';
    g.innerHTML += runs.length ? `<title>${why}</title>` : '';
  }
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
           `<text class="ynum" x="-5" y="${y}%" dy="4">${fmtTick(v)}</text>`;
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
    out += `<text class="ynum" x="-5" y="${((i + 0.5) / L) * 100}%" dy="4">${i}</text>`;
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
const fitsGrid = (m) => m && m.u.length === S.info.rows && m.u[0]?.length === S.info.cols;

function modeAxes() {
  const svg = $('#modeax');
  if (!svg) return;
  const R = S.info.rows, C = S.info.cols;
  let out = '';
  for (let r = 0; r < R; r++)
    out += `<text class="ml" x="-4" y="${((r + 0.5) / R) * 100}%" dy="4">${r}</text>`;
  for (let c = 0; c < C; c++)
    out += `<text class="ml col" x="${((c + 0.5) / C) * 100}%" y="100%" dy="14">${c}</text>`;
  svg.innerHTML = out;
}

/* The rail and the axes both mean different things per view: the size slider scales
   arrows or height, the color fill is a quiver-only readout, and the row/column ticks
   label a grid that the surface's camera has rotated away from. */
function viewCtls() {
  const q = S.modeview === 'quiver';
  $('#asizelab').textContent = q ? 'arrow size' : isBox() ? 'size' : 'height';
  $('#modekey').hidden = !(q && S.fieldbg);
  $('#modeax').style.display = q ? '' : 'none';
  $('.stepmodes').classList.toggle('surfview', S.modeview === 'surface');   // grab cursor: surfaces rotate
  $('.stepmodes').classList.toggle('boxview', isBox());                    // no big plot, full-width tiles
  // no laser photo for this box: the box view has nothing to stand on
  for (const b of document.querySelectorAll('#modeview [data-v="stretch"], #modeview [data-v="inout"]')) b.disabled = !S.info?.laser_view;
  // Hidden rather than wrapped in a container: the rail is a flex column with its own
  // gap, so a wrapper div would change the spacing of everything around it.
  for (const e of document.querySelectorAll('.bgctl, .modewrap .mlab')) e.hidden = !q;
}

/* ***** the box views: the mode on the box's side photo *****
   Same idea as notebooks/102_laser_grid_pixels.ipynb, sec. 4-6. Each tile warps the photo by
   its own mode:
   - stretch: every pixel moves by the measured speckle shift (U, V) -- the wall's local slope,
     drawn as sideways motion -- with a mesh of cells, one per laser and centred on it.
   - in-out: the height field Z (whose slopes are U, V), head-on: the photo in its own colors,
     each pixel lit by its tilt and pushed out from the photo centre by perspective.
   There is no big plot here: one photo can't be warped by several modes at once, so the
   overlay has nothing to add. The tiles take its place, one per row, full width.

   Per box, once (boxMesh): a vertex grid over the crop, each vertex's fractional laser
   (col, row) through the inverse homography, and its fade: 1 on the laser grid's cells,
   easing to 0 at the wall outline, 0 outside it. Past the grid the motion is extrapolated
   (the edge value held), so the wall flexes like a sheet clamped at its edges.
   Per mode, once (boxField): each vertex's (U, -V) and (Z, dZ/dx, dZ/dy).
   Per tile per frame: one WebGL draw of the textured mesh. */
const isBox = () => S.modeview === 'stretch' || S.modeview === 'inout';
const BOXN = 120;                       // vertex columns across the crop
const boxImg = {}, boxMeshes = {};
function boxView() {
  const lv = S.info?.laser_view;
  if (!isBox() || !lv) return null;
  if (!boxImg[lv.name]) {
    boxImg[lv.name] = new Image();
    boxImg[lv.name].onload = drawMode;
    boxImg[lv.name].src = `/api/laserview/${lv.name}.jpg`;
  }
  const [x, y, w, h] = lv.crop;
  return { lv, x, y, w, h, img: boxImg[lv.name] };
}

const homog = (M, x, y) => {
  const d = M[6] * x + M[7] * y + M[8];
  return [(M[0] * x + M[1] * y + M[2]) / d, (M[3] * x + M[4] * y + M[5]) / d];
};
const inPoly = ([x, y], P) => {
  let c = false;
  for (let i = 0, j = P.length - 1; i < P.length; j = i++) {
    const [xi, yi] = P[i], [xj, yj] = P[j];
    if ((yi > y) !== (yj > y) && x < ((xj - xi) * (y - yi)) / (yj - yi) + xi) c = !c;
  }
  return c;
};
const edgeDist = ([x, y], P) => Math.min(...P.map(([ax, ay], i) => {
  const [bx, by] = P[(i + 1) % P.length], dx = bx - ax, dy = by - ay;
  const u = Math.max(0, Math.min(1, ((x - ax) * dx + (y - ay) * dy) / (dx * dx + dy * dy)));
  return Math.hypot(x - ax - u * dx, y - ay - u * dy);
}));
// 1 on the laser grid's cells, smoothly to 0 at the wall outline, 0 outside the wall
function boxFade(p, grid, wall) {
  if (!inPoly(p, wall)) return 0;
  const out = inPoly(p, grid) ? 0 : edgeDist(p, grid), din = edgeDist(p, wall);
  const u = din / (din + out + 1e-6);
  return u * u * (3 - 2 * u);
}
// Catmull-Rom on the laser grid; (c, r) clamped first, so past the grid the edge value holds
const crs = (a, b, c, d, t) => b + 0.5 * t * (c - a + t * (2 * a - 5 * b + 4 * c - d + t * (3 * (b - c) + d - a)));
function bicubic(F, c, r) {
  const R = F.length, C = F[0].length, cl = (v, n) => Math.max(0, Math.min(n - 1, v));
  c = cl(c, C); r = cl(r, R);
  const ci = Math.floor(c), ri = Math.floor(r), at = (i, j) => F[cl(i, R)][cl(j, C)];
  const row = (i) => crs(at(i, ci - 1), at(i, ci), at(i, ci + 1), at(i, ci + 2), c - ci);
  return crs(row(ri - 1), row(ri), row(ri + 1), row(ri + 2), r - ri);
}

function boxMesh(lv) {
  if (boxMeshes[lv.name]) return boxMeshes[lv.name];
  const [x0, y0, w, h] = lv.crop, R = lv.pts.length, C = lv.pts[0].length;
  const grid = [[-0.5, -0.5], [C - 0.5, -0.5], [C - 0.5, R - 0.5], [-0.5, R - 0.5]].map(([c, r]) => homog(lv.H, c, r));
  const nx = BOXN, ny = Math.round((BOXN * h) / w);
  const src = new Float32Array(nx * ny * 2), cr = new Float32Array(nx * ny * 2), fade = new Float32Array(nx * ny);
  for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) {
    const k = j * nx + i, x = x0 + (w * i) / (nx - 1), y = y0 + (h * j) / (ny - 1);
    src[2 * k] = x; src[2 * k + 1] = y;
    [cr[2 * k], cr[2 * k + 1]] = homog(lv.Hinv, x, y);
    fade[k] = boxFade([x, y], grid, lv.wall);
  }
  const idx = new Uint16Array((nx - 1) * (ny - 1) * 6);
  let n = 0;
  for (let j = 0; j < ny - 1; j++) for (let i = 0; i < nx - 1; i++) {
    const a = j * nx + i, b = a + 1, c = a + nx, d = c + 1;
    idx.set([a, b, c, b, d, c], n); n += 6;
  }
  // cell boundaries on half-integers, continued over the whole wall: (c, r) samples + fade
  const wc = lv.wall.map(([x, y]) => homog(lv.Hinv, x, y));
  const lo = (k) => Math.floor(Math.min(...wc.map((p) => p[k]))) - 0.5;
  const hi = (k) => Math.ceil(Math.max(...wc.map((p) => p[k]))) + 0.5;
  const lines = [], seg = (f) => Array.from({ length: 81 }, (_, i) => f(i / 80));
  for (let c = lo(0); c <= hi(0); c++) lines.push(seg((u) => [c, lo(1) + u * (hi(1) - lo(1))]));
  for (let r = lo(1); r <= hi(1); r++) lines.push(seg((u) => [lo(0) + u * (hi(0) - lo(0)), r]));
  const L = lines.map((l) => l.map(([c, r]) => {
    const p = homog(lv.H, c, r);
    return { p, c, r, f: boxFade(p, grid, lv.wall) };
  }));
  const cell = Math.hypot(lv.pts[0][C - 1][0] - lv.pts[0][0][0], lv.pts[0][C - 1][1] - lv.pts[0][0][1]) / (C - 1);
  return (boxMeshes[lv.name] = { nx, ny, dx: w / (nx - 1), dy: h / (ny - 1), src, cr, fade, idx, lines: L, cell, gl: null });
}

// per-vertex (U, -V) * fade for one mode, cached on the mode object
// per-vertex (U, -V) and (Z, dZ/dx, dZ/dy), all times the fade, for one mode; cached on the mode
function boxField(M, m) {
  if (m._box?.M === M) return m._box;
  const { nx, ny } = M, n = nx * ny, d = new Float32Array(2 * n), z = new Float32Array(3 * n);
  for (let k = 0; k < n; k++) {
    const f = M.fade[k], c = M.cr[2 * k], r = M.cr[2 * k + 1];
    if (!f) continue;
    d[2 * k] = f * bicubic(m.u, c, r);
    d[2 * k + 1] = -f * bicubic(m.v, c, r);                 // V up -> screen y down
    if (m.z) z[3 * k] = f * bicubic(m.z, c, r);
  }
  // slopes in px, central differences on the vertex grid (one-sided at its edges)
  const Z = (i, j) => z[3 * (Math.min(ny - 1, Math.max(0, j)) * nx + Math.min(nx - 1, Math.max(0, i)))];
  for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) {
    z[3 * (j * nx + i) + 1] = (Z(i + 1, j) - Z(i - 1, j)) / (2 * M.dx);
    z[3 * (j * nx + i) + 2] = (Z(i, j + 1) - Z(i, j - 1)) / (2 * M.dy);
  }
  const lines = M.lines.map((l) => l.map((q) => [q.f * bicubic(m.u, q.c, q.r), -q.f * bicubic(m.v, q.c, q.r)]));
  return (m._box = { M, d, z, lines, umax: maxOf([{ m }]), zmax: m.z ? zmax([{ m }]) : 0 });
}

// k scales the stretch, kz the depth (px per unit Z); one of them is 0
const BOX_VS = `attribute vec2 src, disp; attribute vec3 zg;
uniform float k, kz, foc; uniform vec2 cen, size; uniform vec4 crop;
varying vec2 uv; varying float shade, depth;
void main() {
  depth = kz * zg.x;
  vec2 p = (cen + (src + k * disp - cen) * foc / (foc - depth) - crop.xy) / crop.zw;   // toward you: out from the centre
  gl_Position = vec4(p.x * 2.0 - 1.0, 1.0 - p.y * 2.0, 0.0, 1.0);
  vec3 n = normalize(vec3(-kz * zg.y, -kz * zg.z, 1.0)), L = normalize(vec3(-1.0, -1.0, 1.2));   // lit from the upper left
  shade = clamp(dot(n, L) / L.z, 0.3, 1.8);
  uv = src / size;
}`;
const BOX_FS = `precision mediump float;
uniform sampler2D tex; uniform float relief, dmax;
varying vec2 uv; varying float shade, depth;
void main() {
  vec3 c = texture2D(tex, uv).rgb;
  // in-out: the photo's own colors, lit by the wall's tilt. Shadow darkens; light lifts toward
  // white, since a saturated green has no headroom to be multiplied brighter.
  if (relief > 0.5) c = shade < 1.0 ? c * shade : c + (1.0 - c) * min(shade - 1.0, 1.0) * 0.7;
  gl_FragColor = vec4(c, 1.0);
}`;

// One offscreen WebGL canvas per box: the program, the static buffers and the photo texture.
function boxGL(M, img) {
  if (M.gl) return M.gl;
  const cv = document.createElement('canvas'), gl = cv.getContext('webgl', { preserveDrawingBuffer: true });
  if (!gl) return null;
  const sh = (t, src) => { const s = gl.createShader(t); gl.shaderSource(s, src); gl.compileShader(s); return s; };
  const pr = gl.createProgram();
  gl.attachShader(pr, sh(gl.VERTEX_SHADER, BOX_VS)); gl.attachShader(pr, sh(gl.FRAGMENT_SHADER, BOX_FS));
  gl.linkProgram(pr); gl.useProgram(pr);
  const attr = (name, size, data) => {
    const b = gl.createBuffer(), a = gl.getAttribLocation(pr, name);
    gl.bindBuffer(gl.ARRAY_BUFFER, b);
    if (data) gl.bufferData(gl.ARRAY_BUFFER, data, gl.STATIC_DRAW);
    gl.enableVertexAttribArray(a); gl.vertexAttribPointer(a, size, gl.FLOAT, false, 0, 0);
    return b;
  };
  attr('src', 2, M.src);
  const disp = attr('disp', 2), zg = attr('zg', 3);
  gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, gl.createBuffer());
  gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, M.idx, gl.STATIC_DRAW);
  gl.bindTexture(gl.TEXTURE_2D, gl.createTexture());
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, img);
  for (const [p, v] of [[gl.TEXTURE_MIN_FILTER, gl.LINEAR], [gl.TEXTURE_MAG_FILTER, gl.LINEAR],
                        [gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE], [gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE]]) gl.texParameteri(gl.TEXTURE_2D, p, v);
  const u = Object.fromEntries(['k', 'kz', 'foc', 'cen', 'size', 'crop', 'relief', 'dmax'].map((n) => [n, gl.getUniformLocation(pr, n)]));
  return (M.gl = { cv, gl, disp, zg, u, m: null });
}

// one tile: the photo warped by mode m, normalised to its own peak
function drawBox(g, w, h, m, bv) {
  if (!bv.img.complete || !bv.img.naturalWidth) return;
  const M = boxMesh(bv.lv), G = boxGL(M, bv.img), D = boxField(M, m), io = S.modeview === 'inout';
  if (!G) return g.drawImage(bv.img, bv.x, bv.y, bv.w, bv.h, 0, 0, w, h);   // no WebGL: the still photo
  const t = Math.cos((S.frame / NFRAME) * 2 * Math.PI), cell = M.cell * S.asize;
  const k = io ? 0 : (t * 0.5 * cell) / (D.umax || 1);      // stretch: the biggest shift is half a laser cell at 1x
  const kz = io ? (t * 1.5 * cell) / (D.zmax || 1) : 0;     // in-out: the biggest depth is 1.5 cells
  const { gl, cv, u } = G, dp = devicePixelRatio || 1;
  if (cv.width !== Math.round(w * dp) || cv.height !== Math.round(h * dp)) { cv.width = Math.round(w * dp); cv.height = Math.round(h * dp); }
  gl.viewport(0, 0, cv.width, cv.height);
  if (G.m !== m) {
    gl.bindBuffer(gl.ARRAY_BUFFER, G.disp); gl.bufferData(gl.ARRAY_BUFFER, D.d, gl.DYNAMIC_DRAW);
    gl.bindBuffer(gl.ARRAY_BUFFER, G.zg); gl.bufferData(gl.ARRAY_BUFFER, D.z, gl.DYNAMIC_DRAW);
    G.m = m;
  }
  const { lv } = bv;
  gl.uniform1f(u.k, k); gl.uniform1f(u.kz, kz); gl.uniform1f(u.foc, lv.w); gl.uniform2f(u.cen, lv.w / 2, lv.h / 2);
  gl.uniform2f(u.size, lv.w, lv.h); gl.uniform4f(u.crop, bv.x, bv.y, bv.w, bv.h);
  gl.uniform1f(u.relief, io ? 1 : 0); gl.uniform1f(u.dmax, 1.5 * cell);
  gl.drawElements(gl.TRIANGLES, M.idx.length, gl.UNSIGNED_SHORT, 0);
  g.drawImage(cv, 0, 0, w, h);
  if (io) return;
  // stretch: the cell mesh, moving with the sheet, clipped to the wall
  const s = w / bv.w, at = ([x, y]) => [(x - bv.x) * s, (y - bv.y) * s];
  g.save();
  g.beginPath();
  lv.wall.forEach((p, i) => g[i ? 'lineTo' : 'moveTo'](...at(p)));
  g.clip();
  g.strokeStyle = 'rgba(255,255,255,0.7)'; g.lineWidth = 0.6;
  M.lines.forEach((l, i) => {
    g.beginPath();
    l.forEach((q, j) => g[j ? 'lineTo' : 'moveTo'](...at([q.p[0] + k * D.lines[i][j][0], q.p[1] + k * D.lines[i][j][1]])));
    g.stroke();
  });
  g.restore();
}

function drawMode() {
  fieldGrid();
  recoveredGrid();
  modeAxes();
  if (isBox()) return;                       // the box views are tiles only (fieldGrid above)
  const cv = $('#mode'), [g, w, h] = fit(cv);
  const R = S.info.rows, C = S.info.cols, cw = w / C, chh = h / R;
  // The tiles you selected. None selected yet: the current sample at the cursor, so every
  // view shows something from the start, before anything is picked or saved.
  let sets = selectedModes().map((s) => ({ p: s.p, m: s.m, c: col(s.p, 0.9, rowShade(s.p, s.fi)) }));
  if (!sets.length && liveShown() && fitsGrid(S.mode)) sets = [{ p: S.liveP, m: S.mode, c: col(S.liveP, 0.9) }];
  if (!sets.length) return;

  if (S.modeview === 'surface') {
    if (!hasZ(sets)) return stale(g, w, h);
    // Every set on ONE shared height scale, so an overlaid surface still compares
    // amplitudes the way the overlaid quiver does.
    surfaces(g, sets, w, h, R, C, zmax(sets), sets.length <= 2 ? 2 : 1);
    return;
  }
  // The big plot always overlays every probe. Arrows from different probes share a cell
  // origin, so color alone would hide one under another: fan the tails around the centre.
  const k = (Math.min(cw, chh) * 0.42 * S.asize) / (maxOf(sets) || 1);
  arrows(g, sets, cellAt(cw, chh), k, sets.length > 1 ? Math.min(cw, chh) * 0.15 : 0);
}

// cell (r, c) -> its centre on an abstract grid of cw x chh cells
const cellAt = (cw, chh) => (r, c) => [c * cw + cw / 2, r * chh + chh / 2];

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

// `at(r, c)` places each cell's arrow: an abstract grid (cellAt) or a laser's photo spot.
function arrows(g, sets, at, k, fan) {
  const t = Math.cos((S.frame / NFRAME) * 2 * Math.PI);
  sets.forEach((s, n) => {
    const a = fan ? (n / sets.length) * 2 * Math.PI : 0;
    const ox = fan * Math.cos(a), oy = fan * Math.sin(a);
    g.strokeStyle = s.c; g.fillStyle = s.c; g.lineWidth = s.lw ?? 1.3;
    const R = s.m.u.length, C = s.m.u[0].length, dot = s.lw ? 2.5 : 1.5;
    for (let r = 0; r < R; r++) for (let c = 0; c < C; c++) {
      const [x0, y0] = at(r, c), x = x0 + ox, y = y0 + oy;
      const dx = s.m.u[r][c] * k * t, dy = -s.m.v[r][c] * k * t;
      g.beginPath(); g.moveTo(x, y); g.lineTo(x + dx, y + dy); g.stroke();
      g.beginPath(); g.arc(x + dx, y + dy, dot, 0, 7); g.fill();
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
   from one direction at a time, so quads sorted back-to-front composite correctly with no
   z-buffer, and each quad's normal against a camera-fixed light gives the shading.
   Starts at the notebook's camera (elev 25, azim -60); drag to rotate (S.cam). */
/* The notebook's set_zlim(global_z_lims), where the limit is the peak times 2: the box is
   given twice the height it needs, so even a full-amplitude frame sits in the middle half
   and never grazes the top. Fixed, not tracking the height slider -- see zbox below. */
const ZBOX = 2.0;

/* The view basis for S.cam (matplotlib's azim/elev): V points from the scene to the
   viewer, Rt is screen-right and Up screen-up. Recomputed only when the camera moves. */
let VW = null, VWk = '';
function vw() {
  const key = `${S.cam.az},${S.cam.el}`;
  if (key !== VWk) {
    const a = S.cam.az * Math.PI / 180, e = S.cam.el * Math.PI / 180;
    const ca = Math.cos(a), sa = Math.sin(a), ce = Math.cos(e), se = Math.sin(e);
    const Rt = [-sa, ca, 0], Up = [-se * ca, -se * sa, ce], V = [ce * ca, ce * sa, se];
    // the lamp sits up and to the left of the viewer and turns with the camera, so the
    // shading reads the same from any angle
    const L = norm3(Rt.map((x, i) => -0.45 * x + 0.65 * Up[i] + 0.6 * V[i]));
    VW = { Rt, Up, V, L, H: norm3(L.map((x, i) => x + V[i])) };
    VWk = key;
  }
  return VW;
}
const dot3 = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const norm3 = (a) => { const n = Math.hypot(...a) || 1; return a.map((x) => x / n); };

/* Grid cell + height -> a point in the camera's own units. Depth (toward the viewer) comes
   back too: it is the sort key, and the only thing that makes the quads composite. */
function camera(c, r, z, C, R, zk) {
  const p = [c / (C - 1) - 0.5, 0.5 - r / (R - 1), z * zk], { Rt, Up, V } = vw();
  return [dot3(p, Rt), -dot3(p, Up), dot3(p, V)];
}

/* The height field on a finer grid (k samples per laser cell, cubic): 9x9 flat quads read
   as facets, not a membrane. Cached on the mode, so a frame only scales it by t. */
function refinedZ(m, k) {
  const c = (m._zr ??= {});
  if (c[k]) return c[k];
  const R = m.z.length, C = m.z[0].length, NC = (C - 1) * k + 1, NR = (R - 1) * k + 1;
  const Z = new Float32Array(NR * NC);
  for (let j = 0; j < NR; j++) for (let i = 0; i < NC; i++) Z[j * NC + i] = bicubic(m.z, i / k, j / k);
  return (c[k] = Z);
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
// k: samples per laser cell. Tiles and busy overlays draw the bare 10x10 grid (k = 1); only
// the big plot with one or two surfaces is refined, so many surfaces at once stay cheap.
function surfaces(g, sets, w, h, R, C, max, k = 1) {
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

  // finer on the big plot than on a tile; each laser cell keeps its outline (edge bits)
  const NC = (C - 1) * k + 1, NR = (R - 1) * k + 1;
  const { V, L, H } = vw();
  const dxs = 1 / ((C - 1) * k), dys = 1 / ((R - 1) * k);       // one refined step, camera units
  const quads = [];
  sets.forEach((s) => {
    // lit in the set's own hue AND lightness, so each frequency keeps its shade, as its arrows
    // do. A grey (the current sample's ink) has no hue, so it takes the app's blue; the bare
    // ink, unshaded and near-black, has no room for light and shadow, so mid-lightness.
    let [hh, ss, ll] = rgbToHsl(...rgbOf(s.c));
    if (ss < 0.15) [hh, ss] = rgbToHsl(...rgbOf(css('--accent')));
    if (ll < 0.2 || ll > 0.85) ll = 0.55;
    const Z = refinedZ(s.m, k), P = new Array(NR * NC);
    for (let j = 0; j < NR; j++) for (let i = 0; i < NC; i++) P[j * NC + i] = scr(i / k, j / k, Z[j * NC + i] * t);
    for (let j = 0; j < NR - 1; j++) for (let i = 0; i < NC - 1; i++) {
      const q = j * NC + i, z00 = Z[q] * t, z01 = Z[q + 1] * t, z10 = Z[q + NC] * t, z11 = Z[q + NC + 1] * t;
      // the quad's normal in camera units (y runs opposite to rows), two-sided
      const sx = ((z01 + z11 - z00 - z10) / 2) * zk / dxs, sy = -((z10 + z11 - z00 - z01) / 2) * zk / dys;
      let n = norm3([-sx, -sy, 1]);
      if (dot3(n, V) < 0) n = n.map((x) => -x);
      const diff = Math.max(0, dot3(n, L)), spec = Math.max(0, dot3(n, H)) ** 40;
      // crest vs trough on lightness, scaled by how far into the cycle we are
      const zn = max ? (z00 + z01 + z10 + z11) / 4 / max : 0;
      const l = Math.max(0.04, Math.min(0.92, ll * (0.3 + 0.85 * diff) * (1 + 0.3 * zn * amp)));
      const [r0, g0, b0] = hslToRgb(hh, Math.min(1, ss * 1.05), l);
      const sp = 255 * 0.45 * spec;
      const a = P[q], b = P[q + 1], d = P[q + NC + 1], e = P[q + NC];
      quads.push({
        pts: [a, b, d, e],
        depth: (a[2] + b[2] + d[2] + e[2]) / 4,
        rgb: [Math.min(255, r0 + sp) | 0, Math.min(255, g0 + sp) | 0, Math.min(255, b0 + sp) | 0],
        alpha: sets.length > 1 ? 0.72 : 1,
        edge: (i % k ? 0 : 1) | (j % k ? 0 : 2),
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
    /* The laser cells' outlines only, not every refined quad: a faint ruling that gives the
       eye the curvature without turning into a dense mesh. Left and top edges, so each
       interior line is drawn once. */
    if (!q.edge) continue;
    g.strokeStyle = `rgba(255,255,255,${0.22 * q.alpha})`;
    g.lineWidth = 0.6;
    g.beginPath();
    if (q.edge & 1) { g.moveTo(q.pts[3][0], q.pts[3][1]); g.lineTo(q.pts[0][0], q.pts[0][1]); }
    if (q.edge & 2) { g.moveTo(q.pts[0][0], q.pts[0][1]); g.lineTo(q.pts[1][0], q.pts[1][1]); }
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

/* The mode tiles. On top, the CURSOR tile: whatever you are pointing at -- the live
   cursor, or the cursor on a saved curve (a peak, or any frequency clicked on the
   spectrum) -- with a button to add it to that position's row. Below, one ROW per saved
   position: its three strongest peaks plus any frequency you added, in increasing frequency. Click
   a tile to show / hide it in the big plot above; that plot overlays every shown tile.
   Every tile shares the big plot's animation clock, so play moves all of them together. */
function fieldGrid() {
  const box = $('#fgrid');
  if (!box) return;
  // frequency, then |FFT| there flush right (dB on the log scale), from that tile's own curve
  const cap = (fi, d) => `<b>${curveLine(null, null, hz(fi))}</b> <i>${d ? pkVal(d, fi) + (S.log.spec ? ' dB' : '') : ''}</i>`;
  // Every row is one sample: its tiles, plus a leading cursor tile while the cursor (on
  // the active curve) sits off the row. The cursor's tile is marked (.cur) and shows the
  // hover preview.
  const cur = cursorSet(), act = cur && ownerOf(cur.who);
  const rows = rowOwners().map((p) => {
    const live = p === S.liveP, d = live ? S.probe : p.data, curFi = p === act ? cur.fi : null;
    const fis = p.row.map((x) => x.fi);
    if (curFi != null && !fis.includes(curFi)) fis.push(curFi);
    fis.sort((a, b) => a - b);             // tiles left to right in increasing frequency
    return {
      hd: live ? `current · ${sampleLine(S.byId[S.live.sid])} <button class="fadd">+ save</button>`
        : `<i class="sw" style="background:${col(p)}"></i>${sampleLine(p.meta)}` +
          `<span class="an"><button class="link" data-all="${p.id}">all</button>` +
          `<button class="link" data-none="${p.id}">none</button></span>`,
      tiles: fis.map((fi) => ({
        who: live ? 'live' : p, fi, c: col(p, 1, rowShade(p, fi)), id: live ? 0 : p.id, t: cap(fi, d),
        m: fi === curFi ? cur.m : modeAt(p, fi),
        cls: (p.row.find((x) => x.fi === fi)?.on ? 'on' : 'off') + (fi === curFi ? ' cur' : ''),
      })),
    };
  });
  const tiles = rows.flatMap((r) => r.tiles);
  if (!tiles.length) { box.innerHTML = ''; box.dataset.sig = ''; return; }

  // rebuild only when the tiles change, not every animation frame
  const html = `<div class="frows">` + rows.map((r) =>
    `<div class="frow"><div class="frowhd">${r.hd}</div><div class="fgrid-rest">` +
    r.tiles.map((s) => `<figure data-id="${s.id}" class="${s.cls}" style="--c:${s.c}">` +
      `<canvas width="124" height="124"></canvas><figcaption>${s.t}</figcaption></figure>`).join('') +
    `</div></div>`).join('') + `</div>`;
  if (box.dataset.sig !== html) {
    box.dataset.sig = box.innerHTML = html;
    box.querySelectorAll('figure').forEach((f, i) => (f.onclick = () => pickPeak(tiles[i].who, tiles[i].fi, true)));
    box.querySelector('.fadd')?.addEventListener('click', pin);
    // all / none: every tile of that row into the big plot, or out of it
    box.querySelectorAll('[data-all], [data-none]').forEach((b) => (b.onclick = () => {
      const p = S.probes.find((q) => q.id === +(b.dataset.all ?? b.dataset.none));
      for (const x of p.row) x.on = b.dataset.all != null;
      paintAll();
    }));
  }
  box.querySelectorAll('figure').forEach((f) =>
    f.classList.toggle('hot', !!S.hot && +f.dataset.id === S.hot.id));
  box._tiles = tiles;
  registerZone('modes', box, stepTile);

  /* Each tile is normalised to ITS OWN peak, so every one fills the cell and its shape is
     readable even for a weakly excited mode. Relative magnitude between probes is what the
     overlaid plot above is for -- on a shared scale here, a weak mode would collapse into
     invisible arrows and the shape, the only thing these tiles are for, would be lost.
     And each on its OWN laser grid: a probe from another box may not match this one's. */
  const bv = boxView();
  box.style.setProperty('--ta', bv ? bv.w / bv.h : 1);          // box tiles take the photo crop's shape
  box.querySelectorAll('figure canvas').forEach((cv, i) => {
    const t = tiles[i], [g, w, h] = fit(cv);
    if (!t.m) return;                           // still loading
    // a probe from another box has its own photo, not this one
    if (bv && fitsGrid(t.m) && (t.who === 'live' || t.who.ds === S.info.dataset)) {
      if (S.modeview === 'inout' && !hasZ([t])) return stale(g, w, h);
      return drawBox(g, w, h, t.m, bv);
    }
    const R = t.m.u.length, C = t.m.u[0].length, cw = w / C, chh = h / R;
    if (S.modeview === 'surface') {
      if (!hasZ([t])) return stale(g, w, h);
      surfaces(g, [t], w, h, R, C, zmax([t]));
      return;
    }
    const max = maxOf([t]);
    if (S.fieldbg) field(g, w, h, t.m, R, C, max);
    const k = (Math.min(cw, chh) * 0.42 * S.asize) / (max || 1);
    arrows(g, [{ m: t.m, c: S.fieldbg ? '#fff' : t.c }], cellAt(cw, chh), k, 0);
  });
}

/* Recovered-audio small-multiples: same single grid as the mode tiles above. Each
   tile is a live spectrogram player (setSpec) of that sample's recovered audio at the
   tile's own laser/channel -- the live selection for the current sample, each probe's
   p.laser/p.ch for pinned ones -- so switching laser or channel re-renders them (~60ms). */
function recoveredGrid() {
  const box = $('#recgrid');
  if (!box) return;
  const lm = S.byId[S.live.sid];
  const cap = (s, ch, l) => `<i>${sampleLine(s)}</i> <i>${recWho(ch, l)}</i>`;
  const cur = lm ? { sid: lm.id, c: css('--ink'), id: 0, ch: S.live.ch, laser: S.live.laser,
                     t: cap(lm, S.live.ch, S.live.laser) } : null;
  const rest = shownProbes().map((p) => ({ sid: p.sid, c: col(p), id: p.id, ch: p.ch, laser: p.laser,
                                           t: cap(p.meta, p.ch, p.laser) }));
  const sets = [...(cur ? [cur] : []), ...rest];
  if (!sets.length) { box.innerHTML = ''; box.dataset.sig = ''; return; }

  // Tiles are rebuilt only when the SET of tiles changes (a probe pinned or removed); moving
  // to another sample or laser just re-points a tile's player (setSpec: the old image stays
  // up until the new one arrives, and an unchanged tile is never reloaded, which would
  // restart it mid-playback).
  const sig = sets.map((s) => s.id).join(',');
  if (box.dataset.sig !== sig) {
    box.dataset.sig = sig;
    const tile = (s) => `<figure data-id="${s.id}"><div class="vidwrap specplay">` +
      `<div class="spbox"><img alt=""><div class="phead"></div></div><audio preload="auto"></audio>` +
      `<div class="vidctl"><button class="vidplay" type="button">▶</button>` +
      `<input class="vidseek" type="range" min="0" max="0" step="0.01" value="0">` +
      `<span class="vidtime">0.00s / 0.00s</span></div></div>` +
      `<figcaption style="border-left-color:${s.c}"></figcaption></figure>`;
    box.innerHTML = `<div class="fgrid-rest">${sets.map(tile).join('')}</div>`;
    box.querySelectorAll('.vidwrap').forEach(wireVideoControls);
  }
  box.querySelectorAll('figure').forEach((f, i) => {
    const s = sets[i];
    f.querySelector('figcaption').innerHTML = s.t;
    setSpec(f.querySelector('.specplay'), s.sid, s.ch, s.laser,
            () => { f.style.display = ''; }, () => { f.style.display = 'none'; });
  });
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
      return `<text class="tk" x="${x}%" y="100%" dy="12">${v}</text>`;
    }).join('') + `<text class="unit" x="50%" y="100%" dy="27">${label}</text>`;
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

/* ***** peaks: every drawn curve's, on the plot and in the rail table *****
   A "peak column" is one curve that has peaks: the live sample (when drawn) and every shown
   probe, in workbench order. One of them is ACTIVE (S.pkWho: 'live' or a probe object) --
   its peaks are labelled on the plot, and the arrow keys walk them. Each column carries its
   own CURSOR: the live S.fi, or a probe's p.cur. Moving a cursor only looks (the cursor
   tile in Modes); clicking a saved curve's peak, or Space, adds it to / drops it from that
   position's row. */
function pkCols() {
  const live = (S.probe && !liveHidden() && S.live.laser !== 'all')
    ? [{ who: 'live', d: S.probe, c: css('--ink'), fi: S.fi, tag: 'now', title: sampleLine(S.byId[S.live.sid]) }] : [];
  return [...live, ...shownProbes().filter((p) => p.data).map((p) => ({
    who: p, d: p.data, c: col(p), fi: p.cur, tag: String(p.pos ?? p.meta?.pos ?? '?'), title: probeLines(p).join('  '),
  }))];
}
const activeCol = (cols = pkCols()) => cols.find((c) => c.who === S.pkWho) ?? cols[0] ?? null;

// Markers ride their own curve, using the same y-mapping multi() drew with.
function peakY(i, d = S.probe) {
  const m = S.p1;
  if (!m || !d) return 10;
  const v = S.specMode === 'magphase' ? (S.log.spec ? d.logmag : d.mag) : d.re;
  return yMap(v[i], m.lo, m.hi, m.h);
}

// a curve's peaks, strongest |FFT| first -- the table's order
const byMag = (d) => [...(d.peaks || [])].sort((a, b) => d.mag[b] - d.mag[a]);

const pkVal = (d, i) => {
  const m = d?.mag?.[i] ?? 0;
  return S.log.spec ? (20 * Math.log10(m + 1e-8)).toFixed(0)
                    : (m >= 10 ? m.toFixed(0) : m >= 1 ? m.toFixed(1) : m.toFixed(2));
};

/* The rail table: one column per curve, its peaks strongest first (up/down here walks
   that order; left/right on the plot walks frequency). Three columns fit the rail; more page with the arrows,
   and the page always follows the active column. */
const PK_SHOW = 3;
function peakList() {
  const el = $('#pklist');
  if (!el) return;
  const cols = pkCols(), act = activeCol(cols), ai = cols.indexOf(act);
  $('#npk').textContent = act ? (act.d.peaks || []).length : '';
  if (ai >= 0) S.pkOff = Math.min(Math.max(S.pkOff ?? 0, ai - PK_SHOW + 1), ai);
  S.pkOff = Math.max(0, Math.min(S.pkOff ?? 0, cols.length - PK_SHOW));
  const page = cols.slice(S.pkOff, S.pkOff + PK_SHOW);
  const pager = cols.length > PK_SHOW
    ? `<div class="pkpage"><button data-pg="-1"${S.pkOff ? '' : ' disabled'}>‹</button>` +
      `<span>${S.pkOff + 1}–${S.pkOff + page.length} of ${cols.length}</span>` +
      `<button data-pg="1"${S.pkOff + PK_SHOW < cols.length ? '' : ' disabled'}>›</button></div>` : '';
  // rebuilt every paint: keep the table's own scroll, then bring the selection into view
  const top = el.querySelector('.pkcols')?.scrollTop ?? 0;
  el.innerHTML = pager + `<div class="pkcols">` + page.map((c) => {
    const k = cols.indexOf(c), on = c === act;
    return `<div class="pkcol${on ? ' on' : ''}" style="--c:${c.c}">` +
      `<div class="pkhd" data-k="${k}" title="${c.title}">${c.tag}</div>` +
      byMag(c.d).map((i) =>
        `<button class="pkrow${pkState(c, i)}${i === c.fi ? ' cur' : ''}" data-k="${k}" data-fi="${i}" ` +
        `title="${fmt(hz(i))} Hz · ${pkVal(c.d, i)}${S.log.spec ? ' dB' : ''}">` +
        `<span class="hzv">${fmt(hz(i))}</span><span class="dbv">${pkVal(c.d, i)}</span></button>`).join('') +
      `</div>`;
  }).join('') + `</div>`;
  const box = el.querySelector('.pkcols'), on = box.querySelector('.pkcol.on .pkrow.cur');
  box.scrollTop = top;
  // scroll only the table (scrollIntoView would also drag the page to it on every paint);
  // the sticky header row covers the top, so keep the selection below it
  if (on) {
    const hd = box.querySelector('.pkhd')?.offsetHeight ?? 0;
    const y = on.offsetTop - box.offsetTop;
    if (y - hd < box.scrollTop) box.scrollTop = y - hd;
    else if (y + on.offsetHeight > box.scrollTop + box.clientHeight) box.scrollTop = y + on.offsetHeight - box.clientHeight;
  }
  el.querySelectorAll('.pkrow').forEach((b) => (b.onclick = () => {
    pickPeak(cols[+b.dataset.k].who, +b.dataset.fi, true);
    el.focus();
  }));
  el.querySelectorAll('.pkhd').forEach((h) => (h.onclick = () => {
    const c = cols[+h.dataset.k];
    pickPeak(c.who, c.fi);
  }));
  el.querySelectorAll('[data-pg]').forEach((b) => (b.onclick = () => {
    stepCol(+b.dataset.pg * PK_SHOW);
  }));
  // Arrows here move as drawn: up/down through a column, left/right across columns.
  registerZone('peaks', el, (dx, dy) => (dx ? stepCol(dx) : stepPeak(dy, true)));
  if (!el.dataset.armedVis) {
    el.dataset.armedVis = '1';
    el.tabIndex = 0;
    const sync = () => el.classList.toggle('armed', hoverZone === 'peaks' || armedZone === 'peaks');
    ['pointerenter', 'pointerleave', 'focusin', 'focusout'].forEach((ev) => el.addEventListener(ev, sync));
  }
}

/* A dot on every peak of every drawn curve, in that curve's colour; only the active
   curve's are labelled, or several curves' labels would bury the plot. */
function buildPeaks() {
  const g = $('#p1ov').querySelector('.pks');
  if (!g) return;
  const cols = pkCols(), act = activeCol(cols);
  // the active curve last, so its dots sit on top
  const order = [...cols.filter((c) => c !== act), ...(act ? [act] : [])];
  g.innerHTML = order.map((c) => {
    const k = cols.indexOf(c);
    let lastX = -99, up = true;
    return (c.d.peaks || []).map((i) => {
      const dot = `<circle class="pk${c === act ? ' act' : ''}${pkState(c, i)}" r="${c === act ? 3.4 : 2.6}" ` +
                  `data-k="${k}" data-fi="${i}" style="--c:${c.c}"/>`;
      if (c !== act) return dot;
      const x = specX(i);
      up = (x - lastX) < 6 ? !up : true;     // too close to the previous label -> flip
      lastX = x;
      return dot + `<text class="pklab" data-k="${k}" data-fi="${i}" data-up="${up ? 1 : 0}">${fmt(hz(i))}</text>`;
    }).join('');
  }).join('');
  g.querySelectorAll('.pk').forEach((dot) => (dot.onclick = (e) => {
    e.stopPropagation();
    // Clicking a peak marker ON THE PLOT is the one zone-worthy click that doesn't happen
    // on a registerZone'd element (the plot itself isn't a zone) -- set armedZone directly
    // so it's consistent with every other "click something, arrows now act on it" case.
    armedZone = 'peaks';
    pickPeak(cols[+dot.dataset.k].who, +dot.dataset.fi, true);
  }));
}

function cursors() {
  probeTicks();
  const at = (i) => `${specX(i)}%`;           // through the shared frequency zoom window
  for (const id of OVS) {
    const svg = $(id);
    const sel = svg.querySelector('.sel'), hov = svg.querySelector('.hov');
    const cf = activeCol()?.fi ?? S.fi;      // the active curve's cursor
    sel.setAttribute('x1', at(cf)); sel.setAttribute('x2', at(cf));
    sel.style.display = inSpecX(cf) ? '' : 'none';
    const show = S.hoverFi != null && S.hoverFi !== cf && S.live.laser !== 'all'
                 && inSpecX(S.hoverFi);
    hov.style.display = show ? '' : 'none';
    if (show) { hov.setAttribute('x1', at(S.hoverFi)); hov.setAttribute('x2', at(S.hoverFi)); }
  }
  const cols = pkCols();
  $('#p1ov').querySelectorAll('.pk').forEach((c) => {
    const i = +c.dataset.fi, k = cols[+c.dataset.k];
    if (!k) return;
    c.setAttribute('cx', at(i)); c.setAttribute('cy', peakY(i, k.d));
    c.style.display = inSpecX(i) ? '' : 'none';
    c.classList.toggle('cur', i === k.fi);
  });
  $('#p1ov').querySelectorAll('.pklab').forEach((tx) => {
    const i = +tx.dataset.fi, k = cols[+tx.dataset.k];
    if (!k) return;
    tx.setAttribute('x', at(i)); tx.setAttribute('y', peakY(i, k.d));
    tx.setAttribute('dy', tx.dataset.up === '1' ? -8 : 15);
    tx.style.display = inSpecX(i) ? '' : 'none';
    tx.classList.toggle('on', i === k.fi);
  });
}

async function setFi(i) {
  S.fi = i; S.modePreview = null; modeSeq++; await loadMode(); paintAll();
}

/* Move a curve's cursor to fi (and make that curve the active one). The live cursor IS
   S.fi, so the live mode follows; a saved curve's cursor only changes what the cursor tile
   shows. toggle (a click on a saved curve's peak) also adds / drops it in the row. */
async function pickPeak(who, fi, toggle = false) {
  S.peakMode = true;
  S.pkWho = who;
  if (toggle) toggleMode(ownerOf(who), fi);
  if (who === 'live') return setFi(fi);
  who.cur = fi;
  paintAll();
}

/* ***** rows: a saved position's modes ***** */
// A saved sample's mode at fi, fetched once and kept (a mode uses every laser and both
// channels, so it never goes stale). null until it arrives -- the arrival repaints. The
// server only reads the loaded box, so another box's probe has only what it fetched
// while its box was loaded: every detected peak, prefetched by pin().
function modeAt(p, fi) {
  const m = (p.mcache ??= {})[fi];
  if (m === undefined && p.ds === S.info.dataset) {   // /api/mode reads the current dataset
    p.mcache[fi] = null;
    api(`/api/mode/${p.sid}?fi=${fi}${modeQ()}`).then((r) => { p.mcache[fi] = r; paintAll(); });
  }
  return m || null;
}
// every shown tile of every shown row, with its field (null while loading)
// the rows on screen: the current sample's (always, unless muted), then every shown probe's
const liveShown = () => !S.muted && S.liveP?.row;
const rowOwners = () => [...(liveShown() ? [S.liveP] : []), ...shownProbes()];
const ownerOf = (who) => (who === 'live' ? S.liveP : who);
const selectedModes = () => rowOwners().flatMap((p) =>
  p.row.filter((x) => x.on).map((x) => ({ p, fi: x.fi, m: modeAt(p, x.fi) })))
  .filter((s) => fitsGrid(s.m));
// a peak's state in its column: shown in the big plot (pick) or in the row but hidden (in)
const pkState = (c, i) => {
  const x = ownerOf(c.who).row?.find((r) => r.fi === i);
  return x ? (x.on ? ' pick' : ' in') : '';
};
/* Show / hide fi in p's row. A frequency not in the row joins it, shown; hiding one you
   added drops it again, while the three strongest stay as tiles to switch back on. */
function toggleMode(p, fi) {
  const x = p.row.find((r) => r.fi === fi);
  if (!x && !modeAt(p, fi) && p.ds !== S.info.dataset) return;   // can't be fetched from here
  if (!x) p.row.push({ fi, on: true, top: false });
  else if (x.on && !x.top) p.row = p.row.filter((r) => r !== x);
  else x.on = !x.on;
}
/* Arrows over the mode tiles: left/right to the previous / next tile in reading order,
   up/down to the tile straight above / below. Landing on one moves the cursor there. */
function stepTile(dx, dy) {
  const box = $('#fgrid'), figs = [...box.querySelectorAll('figure')];
  const at = figs.findIndex((f) => f.classList.contains('cur'));
  let next = figs[at === -1 ? 0 : at + dx];
  if (dy && at !== -1) {
    const r = figs[at].getBoundingClientRect(), box2 = (f) => f.getBoundingClientRect();
    const line = figs.filter((f) => (dy > 0 ? box2(f).top > r.bottom : box2(f).bottom < r.top));
    const y = (dy > 0 ? Math.min : Math.max)(...line.map((f) => box2(f).top));
    next = line.filter((f) => box2(f).top === y)
      .sort((a, b) => Math.abs(box2(a).left - r.left) - Math.abs(box2(b).left - r.left))[0];
  }
  if (next) { const s = box._tiles[figs.indexOf(next)]; pickPeak(s.who, s.fi); }
}

// Space: show / hide the cursor's frequency in its row
function addCursor() {
  const a = activeCol();
  if (a) pickPeak(a.who, a.fi, true);
}
/* What the cursor is on: the active curve's cursor, or while hovering the spectrum, the
   frequency under the pointer. { who, fi, m, c, meta }, or null with nothing to show. */
function cursorSet() {
  if (S.muted) return null;               // Esc hides whatever the cursor is on
  const who = activeCol()?.who ?? 'live';
  const hov = S.modePreview && S.hoverFi != null;    // hovering the spectrum previews there
  if (who === 'live') return S.mode && { who, fi: hov ? S.hoverFi : S.fi, m: S.mode,
                                         c: css('--ink'), meta: S.byId[S.live.sid] };
  const fi = hov ? S.hoverFi : who.cur, m = hov ? S.modePreview : modeAt(who, fi);
  return m && { who, fi, m, c: col(who), meta: who.meta };
}

/* Left/right on the plot (up/down in the table): the active curve's previous/next peak.
   Off a peak, the first one past its frequency in that direction. */
function stepPeak(d, ranked = false) {
  const c = activeCol();
  if (!c) return;
  const pk = ranked ? byMag(c.d) : c.d.peaks || [];   // the table's order, or the plot's
  if (!pk.length) return;
  const at = pk.indexOf(c.fi);
  const next = at === -1
    ? ranked ? pk[0] : (d > 0 ? pk.find((i) => i > c.fi) ?? pk[0] : [...pk].reverse().find((i) => i < c.fi) ?? pk.at(-1))
    : pk[Math.max(0, Math.min(pk.length - 1, at + d))];
  pickPeak(c.who, next);
}

// Land on curve `c` at the peak nearest the active frequency (or just there, if it has none).
function jumpTo(c) {
  const f = activeCol()?.fi ?? S.fi, pk = c.d.peaks || [];
  pickPeak(c.who, pk.length ? pk.reduce((a, b) => (Math.abs(b - f) < Math.abs(a - f) ? b : a)) : f);
}

/* Up/down on the plot: the curve drawn directly above / below the active one at its
   frequency -- by where it is on screen, not by workbench order. */
function stepCurve(d) {
  const cols = pkCols(), act = activeCol(cols);
  if (!act || cols.length < 2) return;
  const f = act.fi;
  const byY = [...cols].sort((a, b) => peakY(f, a.d) - peakY(f, b.d));   // top first
  const next = byY[byY.indexOf(act) + d];
  if (next) jumpTo(next);
}

// Left/right in the table (and its page buttons): the neighbouring column, `d` steps over.
function stepCol(d) {
  const cols = pkCols(), act = activeCol(cols);
  if (!act) return;
  const next = cols[Math.max(0, Math.min(cols.length - 1, cols.indexOf(act) + d))];
  if (next !== act) jumpTo(next);
}

// The tallest detected resonance, or null when nothing was detected.
function strongestPeak() {
  const pk = S.probe?.peaks || [], mag = S.probe?.mag;
  if (!pk.length || !mag) return null;
  return pk.reduce((a, b) => (mag[b] > mag[a] ? b : a));
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
    g.innerHTML = selectedModes().filter((s) => inSpecX(s.fi)).map(({ p, fi }) => {
      const x = specX(fi);
      const dim = S.hot && S.hot !== p;
      const dash = p.dash?.length ? p.dash.join(' ') : '5 3';
      return `<line class="ptick" x1="${x}%" x2="${x}%" y1="0" y2="100%" ` +
             `stroke-dasharray="${dash}" stroke="${col(p, dim ? 0.2 : 0.9, rowShade(p, fi))}"/>`;
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
    const c = col(p), lab = `${sampleLine(p.meta)} ${curveLine(p.laser, p.ch, null)}`;
    return p.dataY
      ? [ttRow(c, `${lab} x`, vals(p.data, key)?.[fi], p), ttRow(c, `${lab} y`, vals(p.dataY, key)?.[fi], p)]
      : [ttRow(c, lab, vals(p.data, key)?.[fi], p)];
  });
  if (!liveHidden() && S.probe) {
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
    // Every row is its own curve's colour -- text and a bar at its left, like the workbench
    // card's -- so the tooltip reads like the plot does; whichever one is hot (hovered on the plot, its legend, or its workbench card)
    // also gets bolded -- same "highlight travels everywhere" rule the curves follow.
    const hot = isHot(r.who);
    return `<span style="--c:${r.c};${hot ? 'font-weight:700' : ''}"><em>${fmtTick(r.v)}</em>${r.lab}</span>`;
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
    // A click anywhere on the spectrum moves a cursor there -- any frequency, not just a
    // peak: the cursor of the curve under the pointer, else the active curve's. The cursor
    // tile in Modes shows the mode there; Space (or its button) adds it to the row.
    const hit = S.hot && S.hot !== 'live' && S.hot !== 'preview' ? S.hot : null;
    const who = hit || activeCol()?.who || 'live';
    const fi = S.hoverFi;
    S.modePreview = null;
    if (who === 'live') {
      S.pkWho = 'live';
      S.peakMode = (S.probe?.peaks || []).includes(fi);
      return setFi(fi);
    }
    pickPeak(who, fi);
    S.peakMode = (who.data?.peaks || []).includes(fi);
  });
}

function readout() {
  const f = S.hoverFi ?? S.fi;
  const l = S.hoverLaser ?? S.live.laser;
  const s = S.byId[S.live.sid];
  if (!s) return;

  // "{pos}-{spk} {layout} {box}" then the curve fields that step depends on
  const title = (laser, ch, hzv) => `<b>${sampleLine(s)}</b>  ${curveLine(laser, ch, hzv)}`;

  $('#sigtitle').innerHTML = title(l, S.live.ch, null);
  // The mode is every laser at one frequency, using BOTH channels (a 2-D displacement
  // field), so neither laser nor channel is a field here -- only the frequency.
  $('#modetitle').innerHTML = title(null, null, hz(f));

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
  // the preview is of the active curve's sample -- a saved one's, when that is active
  const a = activeCol()?.who, p = a && a !== 'live' && a.ds === S.info.dataset ? a : null;
  if (a && a !== 'live' && !p) return;
  modePending = true;
  try {
    const d = await api(`/api/mode/${p ? p.sid : S.live.sid}?fi=${fi}${modeQ()}`);
    if (seq !== modeSeq) return;
    S.modePreview = d;
    if (!p) S.mode = d;
    drawMode();
  } finally { modePending = false; }
}

let previewSeq = 0;
async function previewLaser(i) {
  const seq = ++previewSeq;
  if (i == null) { S.preview = null; if (seq === previewSeq) { drawSpec(); drawShifts(); legends(); } return; }
  const { sid, ch } = S.live;
  const d = await api(`/api/probe/${sid}?ch=${ch}&laser=${i}&kind=${S.kind}&ref=${S.ref}`);
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
   - multi-select (box, object, n_objects, layout): the active values sit as removable
     chips in the column's .pbsel lane, above its label (index.html); an empty lane means
     "all". The list stays put. */
function pickbox(host, { opts, on, pick, multi = false, current, cap = 400, placeholder = 'filter', restBlank = false }) {
  host.innerHTML =
    `<input class="pbq" spellcheck="false" autocomplete="off" placeholder="${placeholder}">` +
    '<div class="pblist"></div>';
  const q = host.querySelector('.pbq');
  const selbar = multi ? host.closest('.pk')?.querySelector('.pbsel') : null;
  const list = host.querySelector('.pblist');
  const focused = () => document.activeElement === q;

  host._draw = () => {
    if (!multi && !focused()) q.value = restBlank ? '' : (current?.()?.label ?? '');
    const s = focused() ? q.value.trim().toLowerCase() : '';

    if (selbar) {
      const picked = opts().filter((o) => on(o.v));
      selbar.innerHTML = picked.map((o) => {
        const dot = o.hue == null ? '' : `<span class="pbdot" style="--h:${o.hue}"></span>`;
        return `<span class="pbchip" title="${o.label}">${dot}<span class="pbct">${o.label}</span>` +
          `<button class="pbx" data-v="${o.v}" title="remove">×</button></span>`;
      }).join('');
      // only the × removes -- a chip is a label for what's selected, not a control
      selbar.querySelectorAll('.pbx').forEach((b) =>
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
      return `<button class="${cls}" data-v="${o.v}" title="${o.title ?? o.label}">` +
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

// "{box} box, spk {spk}" as two live dropdowns above the scatter -- both are single,
// EXCLUSIVE picks WITHIN the current filtered set (like the box/speaker filters, but a
// single choice standing in for the current sample's own box/speaker, not a set of
// candidates). Switching dataset already has its own picker in the filter row; a box here
// is the metadata box, and one dataset can span several.

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
  // pass (that's the header count). Always the whole dataset's count,
  // so it doesn't shrink as you filter and start looking like the picker itself shrank.
  $('#dscount').textContent = `(${S.datasets.length})`;
  $('#boxcount').textContent = `(${new Set(S.all.map((s) => s.box)).size})`;
  $('#objcount').textContent = `(${objectValues().length})`;
  $('#nobjcount').textContent = `(${new Set(S.all.map((s) => s.n)).size})`;
  $('#lycount').textContent = `(${new Set(S.all.map((s) => s.layout)).size})`;
  syncBoxSel(); syncSpkSel();
  pickbox($('#datasetbox'), {
    // Each row carries a thumbnail of the bare box, so the datasets are told apart at a
    // glance rather than by reading long names.
    // The full name, wrapped over as many lines as it needs (style.css #datasetbox .pbl).
    opts: () => S.datasets.map((n) => ({
      v: n, label: n, img: `/api/box/${n}.jpg?v=${S.rv}`, count: S.dsCounts[n] })),
    on: (v) => v === S.info.dataset,
    pick: (v) => switchDataset(v),
    multi: true,                            // chip + plain list, like the other filters
    placeholder: 'filter datasets',
  });
  buildSampleGrid();
  pickbox($('#boxbox'), {
    // The physical box a sample was captured in -- a dataset can span several. Multi-select,
    // same as layout/n_objects: a sample passes if it's in ANY of the picked boxes.
    opts: () => {
      const f = facetTally('box', 'boxes');
      return live(tally('box').map(([b]) => ({ v: b, label: b, count: f.get(b) || 0 })), (v) => S.f.boxes.has(v));
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
      return live(objectValues().map((o) =>
        ({ v: o, label: o === NONE_OBJ ? 'no-object' : o, count: f.get(o) || 0 })), (v) => S.f.objects.has(v));
    },
    on: (v) => S.f.objects.has(v),
    pick: (v) => { toggle(S.f.objects, v); applyFilter(); },
    multi: true, placeholder: 'filter objects',
  });
  pickbox($('#nobjbox'), {
    // Ascending by object count; the COUNT is faceted -- how many samples with that
    // object count survive the other filters.
    opts: () => {
      const f = facetTally('n', 'nobj');
      return live(tally('n').map(([n]) => n).sort((a, b) => a - b)
        .map((n) => ({ v: n, label: String(n), count: f.get(n) || 0 })), (v) => S.f.nobj.has(v));
    },
    on: (v) => S.f.nobj.has(+v),
    pick: (v) => { toggle(S.f.nobj, +v); applyFilter(); },
    multi: true,
  });
  pickbox($('#layoutbox'), {
    opts: () => {
      const f = facetTally('layout', 'layouts');
      return live(tally('layout').map(([l]) => l).sort(cmp)   // alphabetical
        .map((l) => ({ v: l, label: l, count: f.get(l) || 0 })), (v) => S.f.layouts.has(v));
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


/* The sample gallery: a scrollable photo grid, as many columns as fit, overhead photo +
   "Pos x, Spk y (id)" caption, the current sample highlighted and kept in view.

   VIRTUALIZED: only the rows in view (+BUF each side) exist as DOM; the rest is padding.
   Rendering every card was ~30 nodes x 6.3k samples = 189k nodes -- ~1.1s per rebuild and
   ~100ms frames while scrolling. Now it's ~60 cards whatever the dataset size. */
function buildSampleGrid() {
  const grid = $('#samplegrid');
  // PAD and GAP must match .sgrid's padding and .sgin's gap (style.css). As many ~CARD px
  // columns as fit the width; recomputed on every resize.
  const PAD = 6, GAP = 10, CARD = 180, BUF = 3;
  let rowH = 0, cols = 4;                    // row pitch (card + gap), columns

  // pinned samples (this dataset's workbench probes) get a ring in their probe's colour
  let pins = {}, peekPos = null;           // peekPos: the position hovered on the map
  // A card's HTML depends only on its sample and pin -- never on which card is current or
  // hovered (mark() toggles those), so selecting never forces a card to be rebuilt.
  const card = (o) =>
    `<button class="scard${pins[o.id] ? ' pin' : ''}" data-id="${o.id}"` +
    `${pins[o.id] ? ` style="--pc:${pins[o.id]}"` : ''}>` +
    `<span class="sclab" title="sample ${sampleId(o)}">${sampleLine(o)}</span>` +
    `${photoFig(o.id)}</button>`;
  // a detached element of THIS document (not a <template>'s inert one), so its <img> starts
  // loading right away, before it is ever shown
  const make = (html) => {
    const d = document.createElement('div');
    d.innerHTML = html;
    return Object.assign(d.firstChild, { _html: html });
  };
  const mark = () => grid.querySelectorAll('.scard').forEach((b) => {
    b.classList.toggle('on', b.dataset.id === S.live.sid && !liveHidden());
    b.classList.toggle('peek', S.byId[b.dataset.id]?.pos === peekPos);
  });

  // Show the rows around scroll position `top` (default: where the list is scrolled). Cards
  // already showing are kept; new ones are built off-screen and swapped in -- with `top`, in
  // the same frame -- once their photos have decoded (at most 300ms). Swapping first showed
  // a frame of blank cards on every step around the box map: the flicker.
  let shown = '', latest = 0;
  const render = (top) => {
    const hits = grid._hits || [];
    if (!hits.length) { shown = ''; grid.innerHTML = '<div class="pbmore">no match</div>'; return; }
    cols = Math.max(2, Math.floor((grid.clientWidth - 2 * PAD + GAP) / (CARD + GAP)));
    const rows = Math.ceil(hits.length / cols), t = top ?? grid.scrollTop;
    // capped at the viewport: if the grid is ever as tall as its content (narrow window,
    // before layout settles), sizing the window off that would grow it to every row
    const h = Math.min(grid.clientHeight, innerHeight);
    // min(rows - 1): the scroll position can be stale (from a longer list) until this
    // render shrinks the content, and an empty window would leave nothing to measure
    const r0 = rowH ? Math.max(0, Math.min(rows - 1, Math.floor((t - PAD) / rowH) - BUF)) : 0;
    const r1 = rowH ? Math.min(rows, Math.max(r0 + 1, Math.ceil((t + h) / rowH) + BUF)) : Math.min(rows, 8);
    const key = `${r0},${r1},${cols}`, me = ++latest;     // only the newest request lands
    if (key === shown) { if (top != null) grid.scrollTop = top; return; }
    pins = {};
    S.probes.forEach((p) => { if (p.ds === S.info.dataset) pins[p.sid] ??= p.base; });
    let inner = grid.querySelector('.sgin');
    if (!inner) { grid.innerHTML = '<div class="sgin"></div>'; inner = grid.firstChild; }
    const old = new Map([...inner.children].map((b) => [b._html, b]));
    const cards = hits.slice(r0 * cols, r1 * cols).map((o) => old.get(card(o)) || make(card(o)));
    const imgs = cards.filter((b) => !b.isConnected).flatMap((b) => [...b.querySelectorAll('img')]);
    Promise.race([Promise.all(imgs.map((im) => im.decode().catch(() => {}))),
                  new Promise((r) => setTimeout(r, 300))]).then(() => {
      if (me !== latest) return;
      shown = key;
      // the skipped rows become padding on an inner wrapper (not on #samplegrid itself:
      // it's border-box with a fixed height, so padding there would just grow the box)
      inner.style.gridTemplateColumns = `repeat(${cols},minmax(0,1fr))`;
      inner.style.padding = `${r0 * rowH}px 0 ${(rows - r1) * rowH}px`;
      inner.replaceChildren(...cards);
      if (top != null) grid.scrollTop = top;
      mark();
      // measure the real row pitch once there's a card (and again if the width changed it)
      const c = grid.querySelector('.scard');
      const pitch = c.offsetHeight + parseFloat(getComputedStyle(c.parentNode).rowGap);
      if (Math.abs(pitch - rowH) > 0.5) { rowH = pitch; shown = ''; render(top); }
    });
  };
  const redraw = () => { shown = ''; render(); };
  grid.onscroll = () => render();          // cheap: a no-op unless the row range moved
  new ResizeObserver(redraw).observe(grid);

  // One listener each on the grid, not per card: cards come and go as you scroll.
  const cardOf = (e) => e.target.closest('.scard');
  grid.onclick = (e) => { const b = cardOf(e); if (b) pickSample(b.dataset.id); };
  // Hover previews, click selects -- the same in the gallery and on the map: hovering a
  // card rings its position on the map; hovering a map dot lights its cards here (_peek).
  grid.onmouseover = (e) => { const b = cardOf(e); if (b) ringPos(S.byId[b.dataset.id]?.pos); };
  grid.onmouseleave = () => ringPos(null);
  grid._peek = (pos) => { peekPos = pos; mark(); };
  grid._mark = mark;                        // paintAll: live shown/hidden changed

  // Keep the current sample's row in view -- by index, since its card may not exist yet.
  const scrollToOn = () => {
    if (!rowH) return;
    const i = (grid._hits || []).findIndex((o) => o.id === S.live.sid);
    if (i < 0) return;
    const y = PAD + Math.floor(i / cols) * rowH, top = grid.scrollTop;
    render(y < top ? y : y + rowH > top + grid.clientHeight ? y + rowH - grid.clientHeight : top);
  };

  // Redraws only when the matching set actually changed (or rv did -- a reprocessed sample
  // gets a new photo URL): any filter action calls this, and rebuilding identical cards
  // just made the gallery flash.
  let drawn = '';
  grid._draw = () => {
    const hits = matches().sort(gallerySort);
    const key = S.rv + '|' + hits.map((o) => o.id).join();
    if (key !== drawn) { drawn = key; grid._hits = hits; redraw(); }
    scrollToOn();
  };
  // select() only needs the highlight moved (and the row brought into view).
  grid._highlight = () => { mark(); scrollToOn(); };
  grid._draw();
  // The gallery's bg/mask checkboxes change the photo URL everywhere it's shown.
  grid._updateThumbs = () => { redraw(); showPhotos(S.live.sid); };
  grid._redraw = redraw;                    // renderProbes: pin rings changed

  registerZone('gallery', grid, (dx, dy) => {
    const hits = grid._hits || [];
    if (!hits.length) return;
    const at = hits.findIndex((o) => o.id === S.live.sid);
    const next = (at === -1 ? 0 : at) + dx + dy * cols;
    if (next < 0 || next >= hits.length) return;   // stay put at the edges
    pickSample(hits[next].id);
  });
}

const drawPickers = () =>
  ['#datasetbox', '#samplegrid', '#boxbox', '#objectbox', '#nobjbox', '#layoutbox']
    .forEach((id) => $(id)?._draw?.());

const FILTER_KEYS = ['boxes', 'objects', 'layouts', 'nobj', 'spk'];

/* The header's "clear all N filters" button: N counts every chosen value, plus any filter
   set to "none"; hidden when nothing is filtered. */
function renderClear() {
  const vals = FILTER_KEYS.flatMap((k) => [...S.f[k]]);
  const n = vals.filter((v) => typeof v !== 'symbol').length
    + FILTER_KEYS.filter((k) => [...S.f[k]].some((v) => typeof v === 'symbol')).length;
  const b = $('#fclear');
  b.hidden = !n;
  b.textContent = `clear all ${n} filter${n === 1 ? '' : 's'}`;
  b.onclick = () => { FILTER_KEYS.forEach((k) => S.f[k].clear()); applyFilter(); };
}

function applyFilter() {
  const m = matches();
  drawPickers();
  syncBoxSel();
  syncSpkSel();
  renderClear();
  $('#nmatch').textContent = m.length;
  $('#nmatchsub').textContent = ` / ${S.all.length}`;
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

// The map is the box itself: the overhead photo's frame (S.info.overhead, the pixel space
// every centre of mass is in), scaled to 250 units across -- so it has the photo's shape and
// every object, a probe's included, sits where it does in the photo.
const SW = 250;
function scale() {
  const [ow, oh] = S.info.overhead, k = SW / ow;
  const pts = POS().map((p) => S.byId[S.byPos[p][Object.keys(S.byPos[p])[0]]].com);
  return { x: (c) => c * k, y: (r) => r * k, pts, w: SW, h: oh * k };
}

function buildScatter() {
  const svg = $('#scatter'), s = scale();
  svg.setAttribute('viewBox', `0 0 ${s.w} ${s.h}`);
  svg.style.setProperty('--ar', s.w / s.h);          // the element takes the cloud's shape
  svg.innerHTML = POS().map((p, i) =>
    `<circle class="pt" data-p="${p}" cx="${s.x(s.pts[i][1]).toFixed(1)}" cy="${s.y(s.pts[i][0]).toFixed(1)}" r="1.9"/>`).join('')
    + '<circle class="hot" r="6" hidden/>';
  const near = (ev) => {
    const b = svg.getBoundingClientRect();
    const x = (ev.clientX - b.left) / b.width * s.w, y = (ev.clientY - b.top) / b.height * s.h;
    let best = null, bd = 1e9;
    POS().forEach((p, i) => {
      const d = Math.hypot(s.x(s.pts[i][1]) - x, s.y(s.pts[i][0]) - y);
      if (d < bd) { bd = d; best = p; }
    });
    return bd < 6 ? best : null;
  };
  svg.onclick = (e) => { const p = near(e); if (p != null) goPos(p); svg.focus(); };
  // hover previews (ring here, its cards lit in the gallery), like hovering a gallery card
  const peek = (p) => {
    if (p === svg._peek) return;
    svg._peek = p;
    svg.style.cursor = p == null ? '' : 'pointer';
    ringPos(p);
    $('#samplegrid')._peek?.(p);
  };
  svg.onmousemove = (e) => peek(near(e));
  svg.onmouseleave = () => peek(null);
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
  // a non-empty sample when the position has one; an empty-box sample otherwise
  const all = matches().filter((s) => s.pos === pos), full = all.filter((s) => !s.empty);
  const at = full.length ? full : all;
  const s = at.find((x) => x.spk === spk) || at[0];
  if (s) select(s.id);
}

/* A ring on the position scatter marking a position that is being looked at but NOT
   selected: a hovered gallery card, or the sample behind the curve highlighted in section 2
   (markHot, in that probe's colour). The current sample keeps its own dot -- hover never
   moves it. */
function ringPos(pos, color = '') {
  const svg = $('#scatter'), ring = svg?.querySelector('circle.hot');
  if (!ring) return;
  const pt = pos == null ? null : svg.querySelector(`.pt[data-p="${pos}"]`);
  ring.toggleAttribute('hidden', !pt);     // SVG has no .hidden property, only the attribute
  if (!pt) return;
  ring.setAttribute('cx', pt.getAttribute('cx'));
  ring.setAttribute('cy', pt.getAttribute('cy'));
  ring.style.stroke = color;
  svg.appendChild(ring);                    // paint order: on top of every dot
}

function renderScatter() {
  const svg = $('#scatter');
  let sel = null;
  svg.querySelectorAll('.pt').forEach((c) => {
    const on = +c.dataset.p === S.byId[S.live.sid]?.pos && !S.muted;   // saved or not
    c.classList.toggle('on', on);
    if (on) sel = c;
  });
  // Positions overlap, and SVG has no z-index -- paint order IS document order, so the
  // current dot was being covered by whichever grey dots happened to come after it.
  // Moving it last puts it on top; the fill is semi-transparent (see .pt.on) so the
  // dots underneath stay visible rather than being blotted out.
  if (sel) sel.parentNode.appendChild(sel);
  mapPins();
  // Reflect the current sample's speaker in the dropdown (the title itself is
  // syncBoxSel's job).
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

/* ***** the sample photo: ONE renderer for the gallery cards, the sample viewer and the
   sidebar's current sample, so all three read the same way *****
   1. the image is always the SAME /api/thumb URL -- photo + the object's contour, rendered
      and disk-cached server-side (app.py gallery_thumb) -- so stepping through the gallery
      shows the viewer/sidebar photo instantly, straight from the browser's cache
   2. the photo sits inset in a perimeter band, and the box's 8 speaker slots (same layout
      as the filter ring: 1,2 right / 3-6 top / 7,8 left) sit IN that band, current one
      highlighted -- so no dot ever covers the box itself
   3. optional centre-of-mass crosshairs, drawn over the photo in overhead-pixel units
   Everything is laid out in overhead-pixel units (S.info.overhead), so circles never
   distort and every size scales with the figure. */
// The current sample's photo (right panel). sid-only, so select() calls this before any fetch.
function showPhotos(sid) {
  mountPhoto($('#svphotobox'), sid, OM.sid === sid ? OM.sel : -1);
  // its box takes the frame's aspect ratio (style.css #svphotobox)
  $('#svphotobox').style.setProperty('--arp', frame().W / frame().H);
}

// Everything about the frame that depends only on the dataset -- its geometry, and one
// finished ring SVG per speaker (only 8 possible) -- built once per payload (keyed on rv),
// not re-derived for every card.
let FRAME = null;
function frame() {
  if (FRAME?.rv === S.rv) return FRAME;
  const [ow, oh] = S.info.overhead || [4, 3];
  const m = Math.round(Math.max(ow, oh) * 0.1);       // perimeter band width, overhead px
  const W = ow + 2 * m, H = oh + 2 * m;
  const pct = (v, of) => `${(100 * v / of).toFixed(3)}%`;
  const present = new Set(S.all.map((s) => s.spk));
  const ring = {};
  for (const id of Object.keys(SPK))
    ring[id] = `<svg class="spkring" viewBox="0 0 ${W} ${H}" aria-hidden="true">` +
      `${spkBand(+id, ow, oh, m, present)}</svg>`;
  return (FRAME = { rv: S.rv, ow, oh, W, H, ring,
    open: `<div class="pfig" style="aspect-ratio:${W} / ${H}"><div class="pfimg" ` +
          `style="left:${pct(m, W)};top:${pct(m, H)};width:${pct(ow, W)};height:${pct(oh, H)}"` });
}

/* THE sample photo -- gallery cards, the viewer and the sidebar all show this, and the one
   photo | overlay | mask toggle (S.view) switches all of them: the photo with each object
   outlined; a gray photo with the objects filled; or just the objects, each in its table
   colour, on a plain field (--mbg behind the transparent PNG). An X marks each object's
   centre of mass in overlay and mask (not photo). sel >= 0 (the viewer's selected table
   row) keeps only that object's X and fades the other objects' marks in every view. */
// a saved probe's sample is drawn in its probe colour (?c=), every other sample in --seg
const photoSrc = (sid, sel = -1, c = sampleHex(sid)) => S.view === 'mask'
  ? `/api/objmasks/${sid}.png?sel=${sel}${c ? `&c=${c}` : ''}&v=${S.rv}`
  : `/api/thumb/${sid}.jpg?seg=${+(S.view === 'overlay')}${sel >= 0 ? `&sel=${sel}` : ''}` +
    `${c ? `&c=${c}` : ''}&v=${S.rv}`;
function photoFig(sid, sel = -1) {
  const s = S.byId[sid], F = frame(), c = sampleHex(sid);
  const coms = S.view === 'photo' ? [] : sel >= 0 ? [OM.objs[sel].com] : s?.coms || [];
  return F.open + ' data-missing="no image">' +
      `<img src="${photoSrc(sid, sel, c)}" alt=""` +
      ` onerror="this.closest('.pfig').classList.add('missing')">` +
      (coms.length ? `<svg class="comov" viewBox="0 0 ${F.ow} ${F.oh}"${c ? ` style="--seg:#${c}"` : ''}` +
        ` preserveAspectRatio="none" aria-hidden="true">${comSvg(coms, F.ow, F.oh)}</svg>` : '') +
    `</div>` + (s ? F.ring[s.spk] || '' : '') + `</div>`;
}

// (Re)fill a photo slot only when what it shows actually changed: refresh() re-runs on every
// channel/laser change, and rebuilding the <img> each time would blank it while it reloads.
// A new photo is loaded + decoded off-screen first and swapped in only when ready, so the
// old one stays up meanwhile -- a fresh <img> is blank until then, which flashed white while
// arrowing through positions. Only the latest request swaps; a missing image swaps at once
// (its <img> then fails and shows "no image").
function mountPhoto(el, sid, sel = -1) {
  const html = photoFig(sid, sel);
  if (!el || el._html === html) return;
  el._html = html;
  const im = new Image();
  im.src = photoSrc(sid, sel);
  im.decode().catch(() => {}).then(() => { if (el._html === html) el.innerHTML = html; });
}

// The 8 slots on the band's centre line. Every physical slot is drawn, not just the ones
// this dataset used -- unused ones dimmed -- so the full layout always reads at a glance.
function spkBand(curSpk, ow, oh, m, present) {
  return Object.entries(SPK).map(([id, [xf, yf]]) => {
    const cx = m / 2 + xf * (ow + m), cy = m / 2 + (1 - yf) * (oh + m);
    const on = +id === curSpk;
    const rr = m * (on ? 0.46 : 0.36);
    const cls = `spkdot${on ? ' on' : ''}${present.has(+id) ? '' : ' unused'}`;
    return `<g class="${cls}"><circle cx="${cx}" cy="${cy}" r="${rr}"/>` +
      `<text x="${cx}" y="${cy}" font-size="${rr * 1.15}">${id}</text></g>`;
  }).join('');
}

// An X per object centre of mass; coms are [row, col] in overhead px (same as smask).
function comSvg(pts, ow, oh) {
  const a = Math.max(ow, oh) * 0.018;             // X half-size, image px
  return pts.map(([r, c]) =>
    `<path d="M${c - a} ${r - a}L${c + a} ${r + a}M${c + a} ${r - a}L${c - a} ${r + a}"/>`).join('');
}

function renderSpk() {
  const cur = S.byId[S.live.sid], at = S.byPos[cur?.pos] || {};
  const noFilter = !S.f.spk.size;
  // selected speakers as removable chips, in the same lane the other filters use
  const chips = $('#spkchips');
  if (chips) {
    chips.innerHTML = [...S.f.spk].filter((v) => typeof v !== 'symbol').sort((a, b) => a - b)
      .map((s) => `<span class="pbchip">spk ${s}` +
        `<button class="pbx" data-s="${s}" title="remove">×</button></span>`).join('');
    chips.querySelectorAll('.pbx').forEach((b) => (b.onclick = () => toggleSpk(+b.dataset.s)));
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

/* The phase-reference picker: the same laser grid, with NONE and AVG above it. Picking a
   reference rotates every laser's phase, bin by bin, so the reference reads 0 there:
   z'_l = z_l * conj(z_r)/|z_r|. Magnitudes and laser-to-laser phase differences survive. */
function buildRefGrid() {
  const svg = $('#refgrid'), R = S.info.rows, C = S.info.cols;
  const L = 11, PAD = 8, BOX = 100, step = (BOX - 2 * PAD) / (C - 1);
  const TOP = 36, W = L + BOX, H = TOP + BOX + L;
  svg.setAttribute('viewBox', `0 0 ${W} ${H}`);
  const cell = (cls, y, label) => `<g class="${cls}"><rect x="${L}" y="${y}" width="${BOX}" height="14" rx="3"/>` +
    `<text x="${L + BOX / 2}" y="${y + 7.6}">${label}</text></g>`;
  const half = (BOX - 2) / 2;
  let out = cell('nonecell', 0, 'NONE') +
            `<g class="avgcell"><rect x="${L}" y="18" width="${half}" height="14" rx="3"/>` +
            `<text x="${L + half / 2}" y="25.6">AVERAGE</text></g>` +
            `<g class="realcell"><rect x="${L + half + 2}" y="18" width="${half}" height="14" rx="3"/>` +
            `<text x="${L + half + 2 + half / 2}" y="25.6">MOST-REAL</text></g>` +
            `<rect class="box" x="${L}" y="${TOP}" width="${BOX}" height="${BOX}" rx="4"/>`;
  for (let r = 0; r < R; r++) {
    out += `<text class="gl" x="${L - 3}" y="${TOP + PAD + r * step + 1.6}">${r}</text>`;
    for (let c = 0; c < C; c++)
      out += `<circle class="ls" data-i="${r * C + c}" cx="${L + PAD + c * step}" cy="${TOP + PAD + r * step}" r="3.2"/>`;
  }
  for (let c = 0; c < C; c++)
    out += `<text class="gl col" x="${L + PAD + c * step}" y="${TOP + BOX + 7}">${c}</text>`;
  svg.innerHTML = out;
  svg.querySelector('.nonecell').onclick = (e) => { e.stopPropagation(); setRef('none'); };
  svg.querySelector('.avgcell').onclick = (e) => { e.stopPropagation(); setRef('avg'); };
  svg.querySelector('.realcell').onclick = (e) => { e.stopPropagation(); setRef('real'); };
  svg.onclick = (e) => { const t = e.target.closest('.ls'); if (t) setRef(+t.dataset.i); };
  svg.onpointermove = (e) => {
    const t = e.target.closest('.ls');
    svg.querySelectorAll('.ls').forEach((x) => x.classList.toggle('hot', x === t));
  };
  svg.onpointerleave = () => svg.querySelectorAll('.ls').forEach((x) => x.classList.remove('hot'));
  renderRefGrid();
}

async function setRef(r) {
  S.ref = r;
  renderRefGrid();
  dropModes();
  await reviewChannel();
}

/* Cached mode fields were rotated by the old reference (or channel); forget them so the
   tiles refetch. Another box's probe can't refetch, so it keeps what it has. */
function dropModes() {
  S.modePreview = null;
  for (const p of [S.liveP, ...S.probes]) if (p && p.ds === S.info.dataset) p.mcache = {};
}

function renderRefGrid() {
  const r = S.ref, svg = $('#refgrid');
  svg.querySelectorAll('.ls').forEach((x) => {
    const on = typeof r === 'number' && +x.dataset.i === r;
    x.classList.toggle('on', on);
    x.setAttribute('r', on ? 5.4 : 3.2);
  });
  svg.querySelector('.nonecell').classList.toggle('on', r === 'none');
  svg.querySelector('.avgcell').classList.toggle('on', r === 'avg');
  svg.querySelector('.realcell').classList.toggle('on', r === 'real');
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
/* Empty-box captures (drift references; no COM, so not on the map) as a timeline under it:
   a strip spanning every position id -- capture order -- with one mark per RUN (empty boxes
   are recorded several in a row, a few ids apart), and a marker at the current sample.
   Clicking the strip picks the run nearest the click, like the map, so nothing depends on
   hitting a thin mark; the run's own samples are then plain buttons underneath. The
   buttons show the current sample's run, or else the run captured nearest to it. */
const RUN_GAP = 10;                       // ids apart that still count as the same run
function renderRepeats() {
  const el = $('#repeats');
  if (!el) return;
  const cur = S.byId[S.live.sid];
  const reps = matches().filter((s) => s.empty && s.spk === cur?.spk).sort((a, b) => a.pos - b.pos);
  $('#repnote').textContent = '';
  if (!cur || !reps.length) { el.innerHTML = '<span class="repnone">none for this speaker</span>'; return; }
  const runs = [];
  for (const s of reps) {
    const r = runs.at(-1);
    if (r && s.pos - r.at(-1).pos <= RUN_GAP) r.push(s); else runs.push([s]);
  }
  const dist = (r, p) => Math.max(r[0].pos - p, 0, p - r.at(-1).pos);
  const nearest = (p) => runs.reduce((a, r) => (dist(r, p) < dist(a, p) ? r : a));
  const near = nearest(cur.pos);
  // x in % of the strip, over every position id in the dataset
  let p0 = Infinity, p1 = -Infinity;
  for (const s of S.all) { if (s.pos < p0) p0 = s.pos; if (s.pos > p1) p1 = s.pos; }
  const x = (p) => (100 * (p - p0)) / ((p1 - p0) || 1);
  el.innerHTML =
    `<svg class="repbar" viewBox="0 0 100 10" preserveAspectRatio="none">` +
    runs.map((r) => {
      const w = Math.max(x(r.at(-1).pos) - x(r[0].pos), 1.2);
      return `<rect class="reprun${r === near ? ' on' : ''}" x="${x(r[0].pos)}" width="${w}" y="1.5" height="7">` +
        `<title>pos ${r[0].pos}–${r.at(-1).pos} (${r.length})</title></rect>`;
    }).join('') +
    `<rect class="repnow" x="${x(cur.pos) - 0.3}" width="0.6" y="0" height="10"/></svg>` +
    `<div class="repchips">` + near.map((s) =>
      `<button class="rep${s.id === S.live.sid ? ' on' : ''}" data-id="${s.id}">pos ${s.pos}</button>`).join('') +
    `</div>`;
  $('#repnote').textContent = `${runs.length} groups · ${reps.length} empty`;
  const bar = el.querySelector('.repbar');
  bar.onclick = (e) => {
    const b = bar.getBoundingClientRect();
    select(nearest(p0 + ((e.clientX - b.left) / b.width) * (p1 - p0))[0].id);
  };
  el.querySelectorAll('.rep').forEach((b) => (b.onclick = () => select(b.dataset.id)));
}

/* ***** probes ***** */
/* Save the current sample to the workbench -- once: a sample already saved is not added
   again, its card just flashes and its curve becomes the active one. Either way the current
   row's picks move down into the saved row, and the current row is left blank. */
function pin() {
  const old = savedOf(S.live.sid), p = old ?? newProbe(), lp = S.liveP;
  if (old && lp) {
    p.mcache = { ...lp.mcache, ...p.mcache };
    for (const x of lp.row.filter((r) => r.on)) {
      const y = p.row.find((r) => r.fi === x.fi);
      if (y) y.on = true; else p.row.push({ ...x });
    }
  }
  if (lp?.row) lp.row = lp.row.filter((x) => x.top).map((x) => ({ ...x, on: false }));
  S.pkWho = p;                            // the live curve is now this probe's
  S.flash = p.id;
  paintAll();
  clearTimeout(pin._t);
  pin._t = setTimeout(() => { S.flash = null; paintAll(); }, 1400);
  return p;
}

function newProbe() {
  const p = {
    ...S.live, id: Date.now(),
    pos: S.byId[S.live.sid].pos,
    box: S.info.box,                      // the box this probe was measured in
    ds: S.info.dataset,                   // ids repeat across datasets (gallery pin rings)
    dash: DASH[S.live.ch] || [],
    data: S.probe, dataY: S.probeY,
    meta: S.byId[S.live.sid],
    mcache: { ...S.liveP?.mcache, ...(S.fi != null && S.mode ? { [S.fi]: S.mode } : {}) },
  };
  // its row in Modes: the three strongest peaks, only the strongest shown to start
  const pk = S.probe?.peaks || [];
  // the current row's tiles and picks, or by default the three strongest with only #1 shown
  const live = S.liveP?.row || [];
  p.row = live.some((x) => x.on) ? live.map((x) => ({ ...x }))
    : byMag(S.probe || {}).slice(0, NTOP).map((fi, r) => ({ fi, on: r === 0, top: true }));
  if (!p.row.length && S.fi != null) p.row = [{ fi: S.fi, on: true, top: false }];
  p.cur = S.fi ?? p.row[0]?.fi;            // the cursor stays where it was
  for (const fi of pk) modeAt(p, fi);     // every peak, so they all still work from another box
  S.probes.push(p);
  reshade();
  return p;
}

/* THE sample / probe label, used everywhere in viz2:
     "{pos}-{spk} {layout} {box}"        -- which recording (sampleLine)
     "L{laser} C{channel} F{freq}"       -- which curve of it (curveLine); freq in Hz
   Only the fields that apply to what's shown -- pass null to leave one out: section 2
   plots every frequency (no F), section 3's mode uses every laser and both channels (F
   only). The workbench, viewer and sidebar name a probe/sample in full. */
// frequency zero-padded to 4 digits, so stacked labels line up (mono font)
const pad4 = (n) => String(Math.round(n)).padStart(4, '0');
// the universal sample id, "{position:06d}-{speaker}" e.g. 000012-3 (utils/ids.py sample_name)
const sampleId = (s) => `${String(s.pos).padStart(6, '0')}-${s.spk}`;
const sampleLine = (s) => `${sampleId(s)} ${s.layout || '—'} ${s.box}`;
const curveLine = (laser, ch, hzv) =>
  [laser != null && `L${laser}`, ch != null && `C${ch}`, hzv != null && `F${pad4(hzv)}`]
    .filter(Boolean).join(' ');
// a probe's own two lines
const probeLines = (p) => [sampleLine(p.meta), curveLine(p.laser, p.ch, null)];

/* The current sample's label shows in two places, both fed from here: the lines above the
   photo at the top of the right panel (#curmeta), and the live row above the pinned probes (#nowrow). Laser / channel / freq are
   labeled fields on a fixed grid, so hovering the grid or spectrum changes their values
   without reflowing the block. */
function renderNow() {
  const s = S.byId[S.live.sid];
  if (!s) return;
  const laser = S.hoverLaser != null ? S.hoverLaser : S.live.laser;
  const fi = S.hoverFi != null ? S.hoverFi : S.fi;
  const prev = S.hoverLaser != null || S.hoverFi != null;
  const l1 = sampleLine(s), l2 = curveLine(laser, S.live.ch, hz(fi));
  const ds = S.info.dataset ?? S.info.box ?? '—';

  const dl = $('#curmeta');
  if (dl) {
    dl.classList.toggle('muted', !!S.muted);
    dl.innerHTML =
      `<div class="cmhead"><b>${sampleId(s)}</b><span>speaker ${s.spk}</span></div>` +
      `<div class="cmsub">${s.layout || '—'} · ${s.box}</div>` +
      `<dl class="cmf${prev ? ' prev' : ''}">` +
      `<dt>laser</dt><dd>${laser ?? '—'}</dd><dt>channel</dt><dd>${S.live.ch ?? '—'}</dd>` +
      `<dt>freq</dt><dd>${Math.round(hz(fi))} Hz${prev ? ' <u>preview</u>' : ''}</dd>` +
      `<dt>dataset</dt><dd class="ds">${ds}</dd></dl>`;
  }

  const row = $('#nowrow');
  if (row) {
    row.classList.toggle('muted', !!S.muted);
    row.innerHTML =
      `<span class="sw" style="background:${css('--ink')}"></span>` +
      `<span class="id"><b>${l1}</b><br>${l2}${prev ? ' <u>preview</u>' : ''}</span>`;
  }
}

/* The viewer's collapsible metadata: the sample's metadata.jsonl as JSON, one key per line
   (values compact, so a long list like rois stays one wrapped line). Fetched only while
   open, and only when the sample changed. */
function showMeta() {
  const box = $('#svmeta'), sid = S.live.sid, key = `${S.info.dataset}|${S.rv}|${sid}`;
  if (!box.classList.contains('on') || box._key === key) return;
  box._key = key;
  const h = (v) => JSON.stringify(v).replace(/&/g, '&amp;').replace(/</g, '&lt;');
  api(`/api/meta/${sid}`).then((m) => {
    if (S.live.sid !== sid) return;
    box.querySelector('pre').innerHTML = '{\n' + Object.entries(m)
      .map(([k, v]) => `  <b>${h(k)}</b>: ${h(v)}`).join(',\n') + '\n}';
  });
}

/* The viewer's objects table: one row per object (its mask colour, centroid, area) and an
   "all" row with the total area. Clicking a row picks that object in the big photo (only
   its X; in mask view the others fade) -- clicking "all" or the picked row again goes back.
   The pick is kept across samples while it exists, so arrowing through two-object
   samples stays on "object 2". */
const OM = { sid: null, objs: [], sel: -1 };

async function showMasks(sid) {
  const c = sampleHex(sid);
  const objs = await api(`/api/objstats/${sid}?${c ? `c=${c}&` : ''}v=${S.rv}`);
  if (S.live.sid !== sid) return;           // superseded by a newer pick while in flight
  OM.sid = sid; OM.objs = objs;
  if (OM.sel >= objs.length) OM.sel = -1;
  drawMasks();
}

function drawMasks() {
  const { sid, objs, sel } = OM, r0 = (v) => Math.round(v).toLocaleString();
  mountPhoto($('#svphotobox'), sid, sel);
  const row = (i, sw, name, com, area) =>
    `<tr data-i="${i}"${i === sel ? ' class="on"' : ''}><td>${sw}</td><td>${name}</td>` +
    `<td>${com}</td><td>${r0(area)}</td></tr>`;
  $('#smtable').innerHTML =
    '<tr><th>#</th><th>object</th><th>com (x, y)</th><th>area px</th></tr>' +
    objs.map((o, i) => row(i, `<b style="background:${o.color}"></b>${i + 1}`, o.name,
      `(${Math.round(o.com[1])}, ${Math.round(o.com[0])})`, o.vol)).join('') +  // no
      // thousands separator here: "1,055, 680" read as three numbers
    row(-1, 'all', '', objs.length ? `(${['1', '0'].map((k) =>
      Math.round(objs.reduce((a, o) => a + o.com[k], 0) / objs.length)).join(', ')})` : '',
      objs.reduce((a, o) => a + o.vol, 0));   // com: the plain mean of the objects' coms
}

/* Point the viewer's images and players at the current sample. A missing artefact (some
   datasets lack the stimulus copy or the pre-rendered recovered clip) hides just that
   element, or the whole original/recovered block if nothing in it loaded. */
function viewerMedia(sid) {
  // The mp4 already has the audio muxed in (utils/viz.py:make_spectrogram_video), so one
  // element with sound does both -- no separate <audio> fetch/player needed. Keyed by the
  // stimulus, not the sample: most samples played the same chirp, and a per-sid URL made
  // every gallery step reload (and flicker) an identical video.
  const stim = S.d?.stim_name;
  const src = stim ? `/api/stim_video/${encodeURIComponent(stim)}.mp4?v=${S.rv}` : null;
  setMedia('#svvidsrc', src);
  // Recovered: rendered live for the selected laser/channel (setSpec), not a fixed mp4.
  const w = $('#svrec');
  setSpec(w, sid, S.live.ch, S.live.laser, () => w.closest('.svm').classList.remove('missing'),
          () => w.closest('.svm').classList.add('missing'));
  $('#svrecwho').textContent = recWho(S.live.ch, S.live.laser);
  // Step 4's ground truth is the same original video, shown again bigger next to the
  // recovered-video small multiples -- same source, same reasoning as svvidsrc above.
  setMedia('#gtvid', src);
  // The chirp's own params (src/data/audio.py), not the sample's -- what band and how much
  // silence padding it was generated with, in place of the generic "played stimulus" label.
  const st = $('#svstim'), sp = S.d?.stim;
  if (st) st.textContent = sp && sp.f_start != null
    ? `${fmt(sp.f_start)}–${fmt(sp.f_end)} Hz · t ${sp.T_start.toFixed(2)}–${sp.T_end.toFixed(2)}s`
    : 'played stimulus';
}

function setMedia(sel, url) {
  const el = $(sel);
  if (!el || el._url === url) return;       // same source: leave it (and its state) alone
  el.classList.remove('missing');
  const block = el.closest('.svm');
  block?.classList.remove('missing');
  el.onerror = () => {
    el.classList.add('missing');
    // hide the original/recovered block only if BOTH its media failed
    if (block && !block.querySelector('video:not(.missing), audio:not(.missing)'))
      block.classList.add('missing');
  };
  if (!url) {                               // nothing to show (e.g. stimulus not found locally)
    el._url = null;
    el.removeAttribute('src');
    return el.onerror();
  }
  if (el.tagName === 'VIDEO') setVideo(el, url);
  else { el._url = el.src = url; if (el.tagName === 'AUDIO') el.load(); }
}

/* Point a <video> at a new source without a blank flash: until the new source has a frame
   to show, its last frame stays up, painted onto a canvas laid over it (.vidhold). A no-op
   for the source it already has, so a re-render never restarts a playing video. */
function setVideo(v, url) {
  if (v._url === url) return;
  v._url = url;
  v.parentElement.querySelector('.vidhold')?.remove();
  if (v.videoWidth && v.offsetWidth) {
    const c = document.createElement('canvas');
    c.className = 'vidhold';
    c.width = v.videoWidth; c.height = v.videoHeight;
    c.getContext('2d').drawImage(v, 0, 0);
    c.style.cssText = `left:${v.offsetLeft}px;top:${v.offsetTop}px;` +
                      `width:${v.offsetWidth}px;height:${v.offsetHeight}px`;
    v.after(c);
    const drop = () => c.remove();
    v.addEventListener('loadeddata', drop, { once: true });
    v.addEventListener('error', drop, { once: true });
    setTimeout(drop, 2000);               // never leave a stale frame up
  }
  v.src = url;
  v.load();
}

/* What the recovered audio is made from -- mirrors data.rec_channel: "both" channels and
   "all" lasers are heard as their average. */
const recWho = (ch, laser) =>
  `${laser === 'avg' || laser === 'all' || laser == null ? 'avg of lasers' : `laser ${laser}`} · ${ch === 'x' || ch === 'y' ? ch : 'avg of x,y'}`;

/* sid+query -> promise of {img, wav: blob URLs, geo}. The viewer and step 4's
   current-sample tile show the same spectrogram, and a pinned probe of the current sample a
   third copy -- one fetch serves all of them, and flipping back to a laser already seen
   costs nothing. The wav comes down as a blob too, not as the <audio>'s src: /api/audio
   has no Range support, and without it Chrome treats the audio as unseekable (every seek
   snaps back to 0); a blob URL is always seekable. Bounded; an evicted entry's blob URLs
   are released unless a player is still showing them. */
const SPECS = new Map();
function specFetch(sid, q) {
  const key = sid + q;
  if (!SPECS.has(key)) {
    const blob = async (url) => {
      const r = await fetch(url);
      if (!r.ok) throw new Error(r.status);
      return [r, URL.createObjectURL(await r.blob())];
    };
    const p = Promise.all([blob(`/api/recspec/${sid}.png${q}`), blob(`/api/audio/${sid}.wav${q}`)])
      .then(([[r, img], [, wav]]) => ({ geo: JSON.parse(r.headers.get('X-Playhead')), img, wav }));
    p.catch(() => SPECS.delete(key));
    SPECS.set(key, p);
    if (SPECS.size > 48) {
      const [k, old] = SPECS.entries().next().value;
      SPECS.delete(k);
      old.then((s) => [s.img, s.wav].forEach((u) => {
        if (!document.querySelector(`[src="${u}"]`)) URL.revokeObjectURL(u);
      }), () => {});
    }
  }
  return SPECS.get(key);
}

/* Point a spectrogram player (.specplay: img + .phead + audio) at one sample's recovered
   audio for a laser/channel. The server renders the PNG (~60ms) and says where the line
   goes (X-Playhead); the browser moves the line (wireSpecHead). A no-op for the source it
   already has; a result that lands after a newer pick is dropped, so fast laser stepping
   only ever shows the last one. Same sample, new laser: playback carries on from the same
   moment, so you can A/B lasers mid-clip. */
async function setSpec(wrap, sid, ch, laser, ok, fail) {
  const q = `?ch=${ch}&laser=${laser}&v=${S.rv}`, key = sid + q;
  if (wrap._key === key) return;
  const sameSid = wrap._sid === sid;
  wrap._key = key; wrap._sid = sid;
  wrap.classList.add('loading');
  let spec;
  try { spec = await specFetch(sid, q); }
  catch {
    if (wrap._key !== key) return;
    wrap.classList.remove('loading');
    wrap._key = null;                               // retry on the next call
    return fail?.();
  }
  if (wrap._key !== key) return;
  wrap.classList.remove('loading');
  const img = wrap.querySelector('img'), a = wrap.querySelector('audio');
  img.src = spec.img;
  wrap._geo = spec.geo;
  const t = sameSid ? a.currentTime : 0, playing = sameSid && !a.paused;
  a.src = spec.wav;
  a.load();
  a.addEventListener('loadedmetadata', () => {
    if (t) a.currentTime = Math.min(t, a.duration || t);
    if (playing) a.play().catch(() => {});
  }, { once: true });
  wrap._place?.();
  ok?.();
}

/* The playback line: follows the audio every animation frame while playing (smooth, not
   the mp4's 20fps steps), and on seeks/loads while paused. Clipped to the axes, as the
   mp4's line was -- outside the spectrogram's time range it is hidden. Clicking the
   spectrogram seeks to that time. */
function wireSpecHead(wrap) {
  const a = wrap.querySelector('audio'), h = wrap.querySelector('.phead'), box = wrap.querySelector('.spbox');
  const place = wrap._place = () => {
    const g = wrap._geo;
    if (!g) { h.style.display = 'none'; return; }
    const [l, r, top, bot] = g.axes, [t0, t1] = g.t, f = (a.currentTime - t0) / (t1 - t0);
    h.style.display = f < 0 || f > 1 ? 'none' : '';
    h.style.left = `${(l + f * (r - l)) * 100}%`;
    h.style.top = `${top * 100}%`;
    h.style.height = `${(bot - top) * 100}%`;
  };
  const loop = () => { place(); if (!a.paused) requestAnimationFrame(loop); };
  a.addEventListener('play', loop);
  ['seeked', 'timeupdate', 'loadedmetadata', 'pause'].forEach((ev) => a.addEventListener(ev, place));
  box.onclick = (e) => {
    const g = wrap._geo;
    if (!g || !a.duration) return;
    const b = box.getBoundingClientRect(), [l, r] = g.axes, [t0, t1] = g.t;
    const f = ((e.clientX - b.left) / b.width - l) / (r - l);
    if (f < 0 || f > 1) return;
    a.currentTime = Math.min(a.duration, t0 + f * (t1 - t0));
    place();
  };
  place();
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
  const html = S.probes.map((p) => {
    const m = p.meta, [l1, l2] = probeLines(p);
    // Two lines: WHICH sample, WHICH curve of it (the coloured border identifies the probe)
    return `<div class="probe card2${p.hidden ? ' off' : ''}" data-id="${p.id}" style="--c:${col(p)}"
      title="sample ${sampleId(p.meta)}${p.ds !== S.info.dataset ? ` (${p.ds})` : ''}  --  click to view, click the eye to ${p.hidden ? 'show in' : 'hide from'} plots">
      <img class="thumb mask" src="/api/masks.png?ids=${p.sid}&colors=${probeHex(p)}&v=${S.rv}" alt="">
      <div class="meta">
        <div class="ln1"><b class="${m.box !== S.info.box ? 'foreign' : ''}">${l1}</b>
          <span class="eye" aria-hidden="true">${EYE}</span>
          <button class="x" data-x="${p.id}">×</button></div>
        <div class="ln3">${l2}</div>
      </div>
    </div>`;
  }).join('');
  // Rebuild only on a real change: this runs on every repaint (every step around the map),
  // and re-created thumbnails flash blank while they decode.
  const list = $('#plist');
  if (list._html === html) return;
  list._html = list.innerHTML = html;
  $('#samplegrid')._redraw?.();             // the gallery's pin rings follow the probes
  list.querySelectorAll('.probe').forEach((el) => {
    const p = S.probes.find((q) => q.id === +el.dataset.id);
    // Click the eye to drop the probe from the plots and back; click anywhere else on the
    // card (not the eye or ×) to jump the viewer to that sample -- the same "make this the
    // current sample" gesture as the scatter dot, the sample search and the position jump.
    el.onclick = (e) => {
      if (e.target.closest('.x')) return;
      if (e.target.closest('.eye')) {
        p.hidden = !p.hidden;
        S.hot = null;
        paintAll();
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
  list.querySelectorAll('.x').forEach((b) => (b.onclick = (e) => {
    e.stopPropagation();
    S.probes = S.probes.filter((q) => q.id !== +b.dataset.x);
    reshade();                            // frees its slot if it was the sample's last
    paintAll();
  }));
}

/* The saved probes on the position map, one mark per sample in its colour: a dot per
   object at its centre of mass (joined when a sample has several), plus each object's
   outline with "shapes" on. A probe from another box or dataset has no place on this map,
   so those are only counted in its corner. Hovering a probe (S.hot) fades the others. */
const OUTLINES = new Map();               // dataset|sid -> one [[col, row], ...] polygon per object
function mapPins() {
  const svg = $('#scatter'), s = scale(), shapes = $('#shapes').checked;
  const box = S.byId[S.live.sid]?.box, here = new Map();
  let away = 0;
  for (const p of shownProbes()) {
    const m = p.ds === S.info.dataset && S.byId[p.sid];
    if (m?.box === box) here.set(p.sid, p.base.slice(1));
    else away++;
  }
  const xy = ([r, c]) => `${s.x(c).toFixed(1)},${s.y(r).toFixed(1)}`;
  svg.querySelector('.pins')?.remove();
  svg.insertAdjacentHTML('beforeend', `<g class="pins${S.hot?.sid ? ' dim' : ''}">` +
    [...here].map(([sid, c]) => {
      const m = S.byId[sid], pts = m.coms.length ? m.coms : [m.com];
      const k = `${S.info.dataset}|${sid}`;
      if (shapes && !OUTLINES.has(k)) {
        OUTLINES.set(k, []);              // fetch once; redraw when it lands
        api(`/api/objstats/${sid}?v=${S.rv}`).then((o) => { OUTLINES.set(k, o.map((x) => x.outline || [])); mapPins(); });
      }
      return `<g class="pin${S.hot?.sid === sid ? ' hot' : ''}" style="--c:#${c}"><title>${sampleLine(m)}</title>` +
        (shapes ? OUTLINES.get(k).map((o) => `<polygon points="${o.map(([x, y]) => xy([y, x])).join(' ')}"/>`).join('') : '') +
        (pts.length > 1 ? `<polyline points="${pts.map(xy).join(' ')}"/>` : '') +
        pts.map((q) => `<circle cx="${s.x(q[1]).toFixed(1)}" cy="${s.y(q[0]).toFixed(1)}" r="3.2"/>`).join('') + '</g>';
    }).join('') + '</g>');
  svg.querySelectorAll('.pt.on, circle.hot').forEach((e) => svg.appendChild(e));   // current + ring on top
  svg.classList.toggle('haspins', here.size > 0);
  $('#elsewhere').textContent = away ? `+${away} in other boxes` : '';
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
  // one legend per section, each naming only the fields that section doesn't sweep:
  // section 2 (every frequency) L C, section 3 (every laser, both channels) F
  const legend = (fields) => {
    const live = liveHidden() ? '' :
      (S.live.ch === 'both'
        ? `<span data-live="1" data-hot="live">${swatch(ink, DASH.x)}x ${swatch(ink, DASH.y)}y` +
          ` — current  ${fields(S.live.laser, null, hz(S.fi))}</span>`
        : `<span data-live="1" data-hot="live">${swatch(ink, DASH[S.live.ch])}` +
          `current — ${fields(S.live.laser, S.live.ch, hz(S.fi))}</span>`);
    return live + shownProbes().map((p) => {
      // The row text carries the probe's own colour, so the key reads without cross-checking
      // the swatch.
      const c = col(p), tail = `${sampleLine(p.meta)}  ${fields(p.laser, p.ch, null, p)}`;
      return p.dataY
        ? `<span data-id="${p.id}" style="color:${c}">${swatch(c, DASH.x)}x ${swatch(c, DASH.y)}y  ${tail}</span>`
        : `<span data-id="${p.id}" style="color:${c}">${swatch(c, p.dash)}${tail}</span>`;
    }).join('');
  };
  const prev = S.preview
    ? `<span data-hot="preview"><b style="color:${css('--accent')}"></b>preview — L${S.hoverLaser}</span>` : '';
  $('#siglegend').innerHTML = S.probes.length || S.preview
    ? legend((l, ch) => curveLine(l, ch, null)) + prev : '';
  // The big mode plot always overlays every probe, so the legend always applies.
  // each shown frequency in its own shade, as its arrows are drawn
  $('#modelegend').innerHTML = S.probes.length ? legend((l, ch, hzv, p = S.liveP) =>
    (p?.row || []).filter((x) => x.on).map((x) =>
      `<b style="color:${col(p, 1, rowShade(p, x.fi))}">${curveLine(null, null, hz(x.fi))}</b>`).join(' ')) : '';
  for (const el of [$('#siglegend'), $('#modelegend')]) {
    el.querySelectorAll('span[data-id]').forEach((sp) => {
      sp.style.cursor = 'pointer';
      sp.onmouseenter = () => {
        S.hot = S.probes.find((q) => q.id === +sp.dataset.id);
        drawSpec(); drawShifts(); markHot(); probeTicks(); fieldGrid(); recoveredGrid();
      };
      sp.onmouseleave = () => { S.hot = null; drawSpec(); drawShifts(); markHot(); probeTicks(); fieldGrid(); recoveredGrid(); };
      // click commits: that probe's sample becomes current, same as clicking its workbench card
      sp.onclick = () => { const p = S.probes.find((q) => q.id === +sp.dataset.id); if (p) select(p.sid); };
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
  // the workbench too: the hot probe's card lit (and scrolled to), the rest faded, and
  // its sample brought forward in the masks overlay
  $('#plist').classList.toggle('dim', !!S.hot);
  $('#plist').querySelectorAll('.probe').forEach((el) => {
    const on = +el.dataset.id === S.hot?.id;
    el.classList.toggle('hot', on);
    if (on) el.scrollIntoView({ block: 'nearest' });
  });
  $('#nowrow').classList.toggle('hot', S.hot === 'live');
  const hp = S.hot?.ds === S.info.dataset ? S.hot : null;   // probes only; not 'live'/'preview'
  ringPos(hp ? hp.pos : $('#scatter')._peek, hp ? col(hp) : '');   // else keep a map hover's ring
  $('#nowrow').classList.toggle('faded', !!S.hot && S.hot !== 'live');
  mapPins();
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
  // Arming any other zone also drops peak mode: it is sticky (every new sample's cursor
  // lands on a peak), so without this a peak clicked once kept every later arrow press --
  // on the position map, say -- walking peaks instead.
  const arm = () => { armedZone = name; if (!PEAKZ.has(name)) S.peakMode = false; };
  el.addEventListener('mousedown', arm);
  el.addEventListener('focusin', arm);
}
const PEAKZ = new Set(['peaks', 'modes']);

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
  if (wrap.classList.contains('specplay')) wireSpecHead(wrap);
  const v = wrap.querySelector('video, audio'), btn = wrap.querySelector('.vidplay'),
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
/* Drag the grip between the flow and the right panel to resize the panel; the width is kept
   per browser. The plots repaint on the window 'resize' they already listen to. */
function wireAsideResize() {
  const grip = $('#asidegrip'), main = $('main'), K = 'viz2.asideW';
  const set = (w) => {
    w = Math.round(Math.max(320, Math.min(w, innerWidth - 600)));
    main.style.setProperty('--aside', `${w}px`);
    return w;
  };
  try { const w = +localStorage.getItem(K); if (w) set(w); } catch {}
  grip.onpointerdown = (e) => {
    e.preventDefault();
    grip.setPointerCapture(e.pointerId); grip.classList.add('on');
    const right = main.getBoundingClientRect().right - parseFloat(getComputedStyle(main).paddingRight);
    let w = 0;
    grip.onpointermove = (m) => { w = set(right - m.clientX); dispatchEvent(new Event('resize')); };
    grip.onpointerup = () => {
      grip.onpointermove = grip.onpointerup = null; grip.classList.remove('on');
      if (w) try { localStorage.setItem(K, w); } catch {}
    };
  };
  grip.ondblclick = () => {               // back to the default (~ the old step-1 viewer)
    main.style.removeProperty('--aside'); try { localStorage.removeItem(K); } catch {}
    dispatchEvent(new Event('resize'));
  };
}
function videoControls(sel) { const w = $(sel)?.closest('.vidwrap'); if (w) wireVideoControls(w); }

function wire() {
  buildOverlays();
  $('#smtable').onclick = (e) => {
    const i = +(e.target.closest('tr[data-i]')?.dataset.i ?? NaN);
    if (!isNaN(i)) { OM.sel = i === OM.sel ? -1 : i; drawMasks(); }
  };
  // up/down walk the rows as drawn (objects, then "all") -- hover or click the table first
  registerZone('objects', $('#smtable'), (dx, dy) => {
    const rows = [...OM.objs.keys(), -1], at = rows.indexOf(OM.sel);
    const next = rows[at + dy];
    if (dy && next !== undefined) { OM.sel = next; drawMasks(); }
  });
  videoControls('#svvidsrc'); wireVideoControls($('#svrec')); videoControls('#gtvid');
  document.querySelectorAll('#p1, #p2').forEach((el) => freqAxis(el.parentElement));
  ['p1', 'p2', 'p3'].forEach((id) => zoomable($('#' + id).parentElement, id));
  $('#sigreset').onclick = () => resetZoom();

  // The scatter is the primary position picker (it is spatial); this is just a jump box
  // for when you already know the number. It searches, it does not filter, so a position
  // with no sample in the current filter flashes red rather than widening the set.
  // It lives in the right panel's search; Enter hands the arrow keys to the map.
  const posJump = (pj, after) => (pj.onchange = () => {
    // a position ("12") or a sample id ("12-3"), zero padding optional ("000012-3")
    const [, p, k] = pj.value.trim().match(/^(\d+)(?:-(\d+))?$/) || [], n = +p;
    const s = k != null && matches().find((x) => x.pos === n && x.spk === +k);
    if (s) { select(s.id); pj.value = ''; after(); }
    // any filtered sample at n counts -- an empty-box position isn't on the scatter (it
    // has the empty-box strip) but is still a position you can jump to
    else if (k == null && p && matches().some((x) => x.pos === n)) { goPos(n); pj.value = ''; after(); }
    else { pj.classList.add('miss'); setTimeout(() => pj.classList.remove('miss'), 600); }
  });
  posJump($('#posjump'), () => $('#scatter').focus());

  // objects / audio / metadata: tabs, exactly one open (objects by default); the choice is
  // kept per browser
  const detail = (id) => {
    $('#details').hidden = !id;
    document.querySelectorAll('.detail').forEach((d) => d.classList.toggle('on', d.id === id));
    $('#detail').querySelectorAll('button').forEach((b) => b.classList.toggle('on', b.dataset.v === id));
    showMeta();
    try { localStorage.setItem('viz2.detail', id); } catch {}
  };
  $('#detail').onclick = (e) => {
    const b = e.target.closest('button');
    if (b) detail(b.dataset.v);
  };
  let open = '';
  try { open = localStorage.getItem('viz2.detail') || ''; } catch {}
  if (!['svobjs', 'svaudio', 'svmeta'].includes(open)) open = 'svobjs';
  detail(open);
  $('#shapes').onchange = mapPins;
  wireAsideResize();
  // filters are an accordion: click a label to open it (closing the open one) or close it;
  // which one is open is kept per browser; with none kept, object starts open
  const filters = [...document.querySelectorAll('.s1filter .pk:not(.sp)')];
  const openFilter = (pk) => {
    filters.forEach((f) => f.classList.toggle('open', f === pk));
    try { localStorage.setItem('viz2.filter', pk?.querySelector('.pickbox').id || ''); } catch {}
  };
  filters.forEach((pk) => (pk.querySelector(':scope > span').onclick = (e) => {
    if (!e.target.closest('.an')) openFilter(pk.classList.contains('open') ? null : pk);
  }));
  let fo = '';
  try { fo = localStorage.getItem('viz2.filter') || ''; } catch {}
  openFilter($('#' + (fo || 'objectbox'))?.closest('.pk'));
  // one switch for every sample photo: the gallery and the right panel's photo
  seg('#view', (v) => { S.view = v; $('#samplegrid')._updateThumbs(); });
  // A new order starts at its top: _draw keeps the current sample's row in view, which after
  // a re-sort is usually somewhere mid-list or at the far end, so it looked unsorted.
  const resort = () => {
    $('#samplegrid')._draw();
    const g = $('#samplegrid');
    g.scrollTop = 0;
    g._redraw();
  };
  // Every load starts on position, increasing (S.sort / S.sortDir defaults). applySort
  // writes that into the controls first, so a browser restoring the old dropdown value on
  // reload can't show one order while the gallery uses another.
  const sel = $('#galsort'), dir = $('#galdir');
  const applySort = async () => {
    sel.value = S.sort;
    dir.textContent = S.sortDir > 0 ? '↑' : '↓';
    sel.disabled = true;
    await ensureAreas();
    sel.disabled = false;
    resort();
  };
  sel.onchange = () => { S.sort = sel.value; applySort(); };
  dir.onclick = () => { S.sortDir = -S.sortDir; applySort(); };
  applySort();

  seg('#ch', (v) => { S.live.ch = v; if (S.ref !== 'none') dropModes(); reviewChannel(); });
  seg('#specmode', (v) => { S.specMode = v; phVis(); drawSpec(); buildPeaks(); peakList(); cursors(); axes(); refreshTip(); });
  seg('#phmode', (v) => { S.phmode = v; drawSpec(); axes(); refreshTip(); });
  // axes() too: lin/log swaps the y label between |FFT| and log |FFT|, and without it
  // the label kept whatever text the previous mode left behind.
  seg('#speclog', (v) => { S.log.spec = +v; drawSpec(); buildPeaks(); peakList(); cursors(); axes(); refreshTip(); });
  seg('#kind', (v) => { S.kind = v; reviewChannel(); });
  wheel($('#fkey2d'));                    // the square 2-D key, drawn once
  /* Drag any surface -- the big plot or a tile -- to rotate it. They all draw through the one
     S.cam, so they turn together. One redraw per animation frame however fast the pointer
     moves; double-click resets. A drag that moved doesn't also count as a tile click. */
  let drag = null, rotRaf = 0;
  const onDown = (e) => {
    if (S.modeview !== 'surface' || !e.target.closest('canvas')) return;
    drag = { x: e.clientX, y: e.clientY, az: S.cam.az, el: S.cam.el, moved: false };
  };
  addEventListener('pointermove', (e) => {
    if (!drag) return;
    const dx = e.clientX - drag.x, dy = e.clientY - drag.y;
    if (Math.abs(dx) + Math.abs(dy) > 3) drag.moved = true;
    S.cam = { az: drag.az - dx * 0.5, el: Math.max(-89, Math.min(89, drag.el + dy * 0.4)) };
    rotRaf ||= requestAnimationFrame(() => { rotRaf = 0; drawMode(); });
  });
  addEventListener('pointerup', () => {
    if (drag?.moved) { S.justDragged = true; setTimeout(() => (S.justDragged = false), 0); }
    drag = null;
  });
  for (const el of [$('#mode'), $('#fgrid')]) {
    el.addEventListener('pointerdown', onDown);
    el.addEventListener('dblclick', () => { if (S.modeview === 'surface') { S.cam = { az: -60, el: 25 }; drawMode(); } });
  }
  $('#fgrid').addEventListener('click', (e) => { if (S.justDragged) e.stopPropagation(); }, true);

  seg('#modeview', (v) => {
    S.modeview = v; viewCtls(); sizeModePanel(); drawMode();
  });
  $('#fieldbg').onchange = (e) => {
    S.fieldbg = e.target.checked;
    $('#modekey').hidden = !S.fieldbg;      // the key only means something with the fill on
    drawMode();
  };

  $('#clear').onclick = () => { S.probes = []; reshade(); paintAll(); };

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
  const play = (on) => {
    setPlay(on);
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
  // Plays by default, but only while Modes is on screen; a pause you press sticks.
  $('#play').onclick = () => { S.userPaused = S.anim; play(!S.anim); };
  new IntersectionObserver(([e]) => { if (!S.userPaused) play(e.isIntersecting); }, { threshold: 0.2 })
    .observe($('.stepmodes'));
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
    // Space: the cursor's frequency into its row (buttons and players keep their own Space)
    if (e.key === ' ' && !e.target.closest('button, video, audio')) { e.preventDefault(); return addCursor(); }
    if (e.key === 'Escape') {
      e.preventDefault();
      S.muted = !S.muted;
      return paintAll();
    }
    const K = { ArrowLeft: [-1, 0], ArrowRight: [1, 0], ArrowUp: [0, -1], ArrowDown: [0, 1] }[e.key];
    if (K) {
      // Shift+arrows always walk the peaks, and plain arrows do too once you've picked a
      // peak (S.peakMode) -- unless the mouse is over the peak table, which navigates as
      // drawn (its own zone). Left/right walk the active curve's peaks in frequency order;
      // up/down jump to the curve drawn directly above / below it.
      const act = activeCol();
      // over ANY zone, that zone wins (the peak/mode zones navigate as drawn; the others,
      // like the position map, aren't about peaks at all)
      const inZone = !!ZONES[hoverZone] || (!hoverZone && armedZone === 'modes');
      const onPeak = S.peakMode && !inZone && (act?.d.peaks || []).includes(act.fi);
      if (e.shiftKey || onPeak) {
        e.preventDefault();
        return K[1] === 0 ? stepPeak(K[0]) : stepCurve(K[1]);
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

  addEventListener('resize', paintAll);
}
