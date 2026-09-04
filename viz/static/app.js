/* viz — client.
 *
 * Everything the table needs is fetched once (~190KB of samples + ~95KB per run) and
 * kept in memory, so filtering and sorting are local array ops measured in tenths of a
 * millisecond. The server is only asked for things the browser cannot make: decoded
 * predictions, computed metrics, and rendered PNGs.
 *
 * Rows are virtualized against a fixed row height: only the ~10 visible rows exist in
 * the DOM at any time, which is what keeps 1024 rows x N runs smooth. */

const $ = (s) => document.querySelector(s);
/* Row height. Stacked mode adds a second mask box per cell, so rows grow by one mask
   plus its gap; the virtualizer, the spacer and the scroll anchoring all read this.

   Read from the stylesheet rather than duplicated here. These two numbers have to agree
   exactly -- the virtualizer positions rows at k*rowH() while CSS gives them their real
   height, so any drift makes every row overlap its neighbour by the difference. */
const cssPx = (name, fallback) => {
  const v = parseFloat(getComputedStyle(document.documentElement).getPropertyValue(name));
  return Number.isFinite(v) ? v : fallback;
};
let ROW_H_BASE = 206, MASK_BOX = 114;
function readRowMetrics() {
  ROW_H_BASE = cssPx("--row-h", ROW_H_BASE);
  MASK_BOX = cssPx("--mask-h", 110) + 4;   // mask box plus the gap above it
}
const rowH = () => (S.view.mode === "stacked" ? ROW_H_BASE + MASK_BOX : ROW_H_BASE);
const METRICS = [
  { key: "bce",            label: "BCE",     short: "bce",  worst: "high" },
  { key: "iou",            label: "IoU",     short: "iou",  worst: "low"  },
  { key: "localization",   label: "Loc",     short: "loc",  worst: "high" },
  { key: "localization_x", label: "Loc x",   short: "locx", worst: "high" },
  { key: "localization_y", label: "Loc y",   short: "locy", worst: "high" },
  { key: "contour",        label: "Contour", short: "cnt",  worst: "low"  },
  { key: "mass",           label: "Mass",    short: "mass", worst: "abs"  },
];

const S = {
  samples: [], runs: {}, order: [], runOrder: [], meta: null,
  filters: {
    splits: new Set(), speakers: new Set(), layouts: new Set(), nObjects: new Set(),
    objects: new Set(),          // object TYPES present, independent of layout
    boxes: new Set(),            // enclosure the scene sits in
    positions: null, ranges: {},
    // Explicit lookups. Empty means "no restriction"; any entry narrows the table to
    // exactly those samples/positions, unioned so both searches can be used at once.
    findSamples: new Set(), findPositions: new Set(),
  },
  sort: { run: null, metric: "localization", dir: "worst" },
  // relative=false is the default on purpose: a fixed [0,1] domain is the only scale
  // under which two cells mean the same thing, which is the point of a comparison table.
  // background is an on/off switch (the "b" key, the checkbox); bgOp is the level it
  // returns to when switched back on, so the two are stored separately. maskOp scales
  // the mask -- the cube -- alone, which is why it is applied as a canvas alpha and an
  // <img> opacity rather than to the cell, whose backdrop must not move with it.
  view: { mode: "pred", background: true, relative: false, bgOp: 1, maskOp: 1 },
  domain: {}, positions: [], activePos: -1, modalRow: -1,
  hidden: new Map(),  // filter key -> samples that ONLY that filter is holding back
  renderVersion: 0,   // from /api/runs; part of image URLs to defeat immutable caching
  lut: null,          // colormaps from the server, so client and server agree on colour
  epochIdx: null,     // null = each run's latest; otherwise an index into the epoch list
  playing: false,
  frozenOrder: null,  // sid -> rank, pinned while scrubbing so rows hold their places
  frameData: {},      // run -> fp16 masks for the visible rows, all epochs
  truthCache: {},     // sid -> ground-truth values, for client-side diff
};

const fmt = (v, n = 4) => (v == null || !isFinite(v) ? "–" : v.toFixed(n));

/* Run state, styled like a CI/W&B badge: green while training, blue once finished,
   red if it died. Colours carry a text label too -- never colour alone. */
const STATUS = {
  running:  { label: "running",  title: "Training now — predictions update automatically" },
  finished: { label: "done",     title: "Training completed cleanly" },
  crashed:  { label: "crashed",  title: "Training ended in a traceback" },
  stopped:  { label: "stopped",  title: "Log ends with no clean exit and no error — killed, preempted, or the node went away" },
  unknown:  { label: "unknown",  title: "No training log found" },
};

function statusChip(status) {
  const s = STATUS[status] || STATUS.unknown;
  return `<span class="status ${status}" title="${s.title}"><i></i>${s.label}</span>`;
}

/* Layout names run long too (purple-cube-green-cube); keep the colours, drop the noun.
   Full value stays in the chip's title. */
/* The identity line both column types carry: where the capture happened, which speaker
   played, and the sample it belongs to. Shared so the two titles cannot drift apart. */
function identity(s) {
  const pos = s.output_id == null ? "–" : +s.output_id;
  return `Pos ${pos}, Spk ${s.speaker} (${+s.sample_id})`;
}

function shortLayout(s) {
  if (!s) return "–";
  if (s === "empty-box") return "empty";
  return s.split("-").filter((w) => w !== "cube" && w !== "cubes").join("+") || s;
}

/* The split chip. Two runs can put the same sample in different splits, so this is the
   one qualifier a reader must never have to decode -- it is written out in full (the
   epoch moved to its own line to make room) and coloured by the distinction that changes
   what a number means: trained on, or held out. */
function splitChip(s) {
  if (!s) return "";
  const held = s !== "train";
  return `<span class="spchip ${held ? "held" : "train"}" title="${
    held ? `held-out split: ${s}` : "this run trained on this sample"}">${s.replace(/_/g, " ")}</span>`;
}
const api = (u) => fetch(u).then((r) => { if (!r.ok) throw new Error(u); return r.json(); });

/* ***** boot ***** */

async function boot() {
  readRowMetrics();          // before anything measures or positions a row
  const [runs, samples, lut] = await Promise.all([
    api("/api/runs"), api("/api/samples"), api("/api/lut")]);
  S.lut = lut;
  // The server composites its ground-truth PNGs at OVERLAY_GAIN over a backdrop; the
  // canvases apply the same constant themselves. Exposed as a variable so the GT image,
  // which is now composited by the browser, can match them without hardcoding 0.85.
  document.documentElement.style.setProperty("--gt-gain", String(lut.gain ?? 1));
  // The cell box follows the SCENE's aspect, not the mask grid's. --mask-h is the fixed
  // dimension; width derives from it so a 20x40 and a 30x30 grid over the same room draw
  // the same shape and land on the same features. Set before readRowMetrics re-runs, and
  // before any row is measured, since --mask-h/--mask-w drive the virtualizer's geometry.
  if (lut.aspect) {
    const mh = cssPx("--mask-h", 110);
    document.documentElement.style.setProperty("--mask-w", `${Math.round(mh * lut.aspect)}px`);
    readRowMetrics();
  }
  S.meta = runs;
  S.renderVersion = runs.render_version ?? 0;
  S.samples = samples.samples;
  buildPositions();
  initFilters();
  bindFind();
  buildSpeakerDiagram();
  buildScatter();
  buildSliders();
  bindUI();
  for (const name of runs.default_selected) await addRun(name);
  $("#subtitle").textContent =
    `${S.samples.length} samples · ${runs.runs.filter((r) => r.compatible).length} runs available`;
  refresh();
}

/* Samples 8-at-a-time share a physical position (one per speaker), so the scatter
   plots ~125 distinct points rather than 1000 overlapping ones. */
function buildPositions() {
  const byKey = new Map();
  S.samples.forEach((s) => {
    const [r, c] = s.avg_com;
    if (r == null || r < 0) { s.pos = -1; return; }   // empty-box sentinel
    const k = `${r.toFixed(2)},${c.toFixed(2)}`;
    if (!byKey.has(k)) byKey.set(k, { i: byKey.size, r, c, n: 0 });
    const p = byKey.get(k);
    p.n++; s.pos = p.i;
  });
  S.positions = [...byKey.values()];
}

/* ***** runs ***** */

/* The grid a run predicts at, as [h, w]. Runs in one table can be trained at different
   resolutions, so this -- not the global S.lut -- sizes canvases, frame buffers and
   truth lookups. Falls back to the dataset default for a run not loaded yet. */
function runShape(name) {
  const r = S.runs[name];
  if (r && r.h) return [r.h, r.w];
  const e = S.meta && S.meta.runs && S.meta.runs.find((x) => x.name === name);
  if (e && e.shape) return e.shape;
  return [S.lut.h, S.lut.w];
}

/* The grid the ground-truth column renders at: the FINEST among the loaded columns, so
   the reference is never coarser than the predictions it is being compared against. Every
   cell is stretched to the same box, so this changes fidelity, not layout -- and when all
   columns agree it collapses to that single shape. */
function gtShape() {
  const shapes = S.runOrder.map(runShape);
  if (!shapes.length) return [S.lut.h, S.lut.w];
  return shapes.reduce((a, b) => (b[0] * b[1] > a[0] * a[1] ? b : a));
}

const truthKey = (sid, h, w) => (h ? `${sid}@${h}x${w}` : String(sid));

/* Whether the loaded columns disagree on grid size, so the header only spends space on
   the size when it actually disambiguates something. */
function mixedShapes() {
  const seen = new Set(S.runOrder.map((n) => runShape(n).join("x")));
  return seen.size > 1;
}

/* Membership at the run's latest epoch: which samples it covers and their splits. This is
   the run's identity for filtering, and it must survive the epoch scrubber swapping
   `samples` -- see splitsOf. */
function indexMembership(d) {
  d.splits = new Set(Object.values(d.samples).map((x) => x.split));
  d.splitOf = Object.fromEntries(Object.entries(d.samples).map(([k, v]) => [k, v.split]));
}

async function addRun(name, reload = false) {
  if (S.runs[name] && !reload) return;
  const d = await api(`/api/run/${encodeURIComponent(name)}${reload ? "?reload=1" : ""}`);
  d.entry = S.meta.runs.find((r) => r.name === name);
  d.metricsEpoch = null;        // /api/run without ?epoch scores the latest
  indexMembership(d);
  if (reload) {
    // Refresh in place: keep column order and, crucially, leave the filters alone. A new
    // epoch must not silently re-check splits the user turned off or move their sliders.
    const prev = S.runs[name];
    for (const sp of d.splits) {
      if (!prev || !prev.splits.has(sp)) S.filters.splits.add(sp);   // genuinely new only
    }
    S.runs[name] = d;
    invalidateEpochs();
    recomputeDomains({ preserve: true });
    return;
  }
  S.runs[name] = d;
  S.runOrder.push(name);
  invalidateEpochs();
  // Only splits no loaded run had before. Adding a second run must not re-check splits the
  // user turned off -- the two runs' vocabularies overlap, so blanket-adding resets them.
  const known = new Set();
  for (const n of S.runOrder) if (n !== name) S.runs[n].splits.forEach((sp) => known.add(sp));
  const first = S.runOrder.length === 1;
  for (const sp of d.splits) if (first || !known.has(sp)) S.filters.splits.add(sp);
  recomputeDomains({ preserve: S.runOrder.length > 1 });
  refresh();
}

function removeRun(name) {
  delete S.runs[name];
  S.runOrder = S.runOrder.filter((n) => n !== name);
  invalidateEpochs();
  if (S.sort.run === name) S.sort.run = null;
  recomputeDomains();
  refresh();
}

/* Slider bounds track the loaded runs so the handles always span real data. */
/* `preserve` is set when a run reloads with a new epoch: the slider bounds still track
   the data, but a range the user is looking through must not be widened underneath them
   just because the numbers moved. Adding or removing a run is different -- there the
   untouched sliders should span whatever is now loaded. */
function recomputeDomains({ preserve = false } = {}) {
  for (const m of METRICS) {
    let lo = Infinity, hi = -Infinity;
    for (const n of S.runOrder)
      for (const v of Object.values(S.runs[n].samples)) {
        const x = v[m.key];
        if (x != null && isFinite(x)) { if (x < lo) lo = x; if (x > hi) hi = x; }
      }
    if (!isFinite(lo)) { lo = 0; hi = 1; }
    if (hi - lo < 1e-9) hi = lo + 1e-9;
    const prev = S.domain[m.key], cur = S.filters.ranges[m.key];
    // Never shrink the domain on a reload, or a selection sitting near the old edge
    // would get clamped away as the run improves.
    if (preserve && prev) { lo = Math.min(lo, prev[0]); hi = Math.max(hi, prev[1]); }
    S.domain[m.key] = [lo, hi];

    if (!cur) { S.filters.ranges[m.key] = [lo, hi]; continue; }
    const untouched = prev && cur[0] <= prev[0] + 1e-12 && cur[1] >= prev[1] - 1e-12;
    if (preserve) S.filters.ranges[m.key] = untouched ? [lo, hi] : cur;
    else if (!prev || untouched) S.filters.ranges[m.key] = [lo, hi];
    else S.filters.ranges[m.key] = [Math.max(lo, cur[0]), Math.min(hi, cur[1])];
  }
  syncSliders();
}

/* ***** filters ***** */

function initFilters() {
  const f = S.filters;
  S.samples.forEach((s) => {
    if (s.layout) f.layouts.add(s.layout);
    if (s.n_objects != null) f.nObjects.add(s.n_objects);
    if (s.box) f.boxes.add(s.box);
    (s.objects && s.objects.length ? s.objects : ["(none)"]).forEach((o) => f.objects.add(o));
  });
  // Speaker 1 only. The 8 speakers at a position capture the same scene, so showing all
  // of them makes the table eight rows deep per position; starting on one keeps the
  // opening view one row per position, and "all" is a click away.
  f.speakers.add(1);
}

function uniq(get) {
  const m = new Map();
  S.samples.forEach((s) => { const v = get(s); if (v != null) m.set(v, (m.get(v) || 0) + 1); });
  return [...m.entries()].sort((a, b) => (a[0] > b[0] ? 1 : -1));
}

/* Every split a loaded run assigned this sample to -- a SET, not one value.

   A sample does not have "a split": each run's dataloader files it independently, and
   two runs need not even use the same split vocabulary. The same capture is `2-obj` in
   object_count_train12 and `2-cubes` in prep-r0. This used to return the first run's
   answer and stop, which made the sidebar lie in both directions: `2-obj` claimed all 120
   samples while `2-cubes` showed 0 (its samples had already been counted under the other
   run's name), and filtering to `2-cubes` hid rows that genuinely are `2-cubes`.

   Read from `splitOf`, the run's membership at its LATEST epoch, not from `samples`,
   which the epoch scrubber swaps out. A run does not save every sample at every epoch
   (early epochs in particular cover a different subset), and a sample missing from one
   epoch has not left the run -- its ground truth certainly has not changed. Keying the
   filter on the swapped table made those rows fail `failures` and vanish from the table
   entirely, taking the ground-truth column with them. */
function splitsOf(sampleIdx) {
  const out = new Set();
  for (const n of S.runOrder) {
    const sp = S.runs[n].splitOf && S.runs[n].splitOf[sampleIdx];
    if (sp) out.add(sp);
  }
  return out;
}

/* EVERY reason this sample is off screen, as filter keys; empty means it is on screen.

   The table's contents and the "why is nothing here" readout both come from this one
   function, so they cannot drift apart: a sample is hidden if and only if a key names
   the filter doing the hiding, and that key is what the readout offers to clear. */
function failures(s) {
  const f = S.filters;
  const out = [];
  // An explicit ID search overrides the browsing filters: when you ask for a sample by
  // number you want to see it, not have it hidden by a speaker chip you set earlier.
  if (f.findSamples.size || f.findPositions.size) {
    const hit = f.findSamples.has(+s.sample_id) || f.findPositions.has(+s.output_id);
    return hit ? out : ["find"];
  }
  if (!f.speakers.has(s.speaker)) out.push("speakers");
  if (!f.layouts.has(s.layout)) out.push("layouts");
  if (!f.nObjects.has(s.n_objects)) out.push("nObjects");
  if (!f.boxes.has(s.box)) out.push("boxes");
  // Contains-any: a scene passes if any object in it is selected. Empty boxes have no
  // objects, so they ride on the "empty" pseudo-entry rather than never matching.
  const objs = s.objects && s.objects.length ? s.objects : ["(none)"];
  if (!objs.some((o) => f.objects.has(o))) out.push("objects");
  if (f.positions && !(s.pos >= 0 && f.positions.has(s.pos))) out.push("positions");

  // With no runs loaded the table is a ground-truth browser, so every sample qualifies;
  // the split filter only applies once some run has assigned splits.
  if (S.runOrder.length) {
    const sps = splitsOf(s.i);
    // No loaded run predicted this sample. Reported separately from the split chips
    // because no chip can bring it back -- the fix is loading a run that covers it.
    if (!sps.size) out.push("nopred");
    else {
      // Contains-any, like the object filter. A sample that is `2-obj` to one run and
      // `2-cubes` to another belongs to both chips, so either one alone must show it --
      // otherwise turning off a split hides rows that another run still files elsewhere.
      let ok = false;
      for (const sp of sps) if (f.splits.has(sp)) { ok = true; break; }
      if (!ok) out.push("splits");
    }
  }

  // Metric ranges read the sorted run when one is chosen, else any loaded run may
  // satisfy them (union) — the intuitive reading of "show me samples where MSE is high".
  const names = S.sort.run && S.runs[S.sort.run] ? [S.sort.run] : S.runOrder;
  if (!names.length) return out;
  let any = false;
  for (const n of names) {
    const r = S.runs[n];
    const e = r.samples[s.i];
    // Covered by the run but not saved at the epoch being viewed: keep the row (its
    // cell shows "no prediction" for this frame) rather than dropping it from the table.
    if (!e) { if (r.splitOf && r.splitOf[s.i]) any = true; continue; }
    let ok = true;
    for (const m of METRICS) {
      const [lo, hi] = f.ranges[m.key] || [-Infinity, Infinity];
      const v = e[m.key];
      if (v == null) continue;                  // undefined metric doesn't veto a sample
      if (v < lo - 1e-12 || v > hi + 1e-12) { ok = false; break; }
    }
    if (ok) { any = true; break; }
  }
  if (!any) out.push("ranges");
  return out;
}

/* What each filter key is called in the readout, and how to switch it off again.

   `clear` restores that one dimension to "everything", leaving the others alone: the
   readout's whole job is to let you undo the single filter that is hiding what you came
   to look at, without resetting the ones you set deliberately. */
const FILTERS = {
  speakers:  { label: "speaker",  clear: (f) => S.samples.forEach((s) => f.speakers.add(s.speaker)) },
  splits:    { label: "split",    clear: (f) => S.runOrder.forEach((n) => S.runs[n].splits.forEach((sp) => f.splits.add(sp))) },
  nObjects:  { label: "objects",  clear: (f) => S.samples.forEach((s) => f.nObjects.add(s.n_objects)) },
  objects:   { label: "contains", clear: (f) => S.samples.forEach((s) => (s.objects && s.objects.length ? s.objects : ["(none)"]).forEach((o) => f.objects.add(o))) },
  layouts:   { label: "layout",   clear: (f) => S.samples.forEach((s) => s.layout && f.layouts.add(s.layout)) },
  boxes:     { label: "box",      clear: (f) => S.samples.forEach((s) => s.box && f.boxes.add(s.box)) },
  positions: { label: "position", clear: (f) => { f.positions = null; S.activePos = -1; } },
  ranges:    { label: "metrics",  clear: (f) => { for (const m of METRICS) f.ranges[m.key] = [...S.domain[m.key]]; syncSliders(); } },
  find:      { label: "id search", clear: (f) => { f.findSamples.clear(); f.findPositions.clear(); renderFindChips(); } },
  // Not a filter and not clearable: say so rather than offering a button that does
  // nothing. Reported because "the loaded runs never predicted these" is the one answer
  // the chips cannot give, and it is what an eval-only run looks like from the table.
  nopred:    { label: "not predicted by any loaded run" },
};

function clearFilter(key) {
  const info = FILTERS[key];
  if (!info || !info.clear) return;
  info.clear(S.filters);
  refresh();
}

function applyFilters() {
  // One pass, two answers: the rows to draw, and -- for everything held back -- which
  // filter would have to be relaxed to admit it. A sample failing several filters at
  // once is not attributed to any of them: clearing one would not bring it back, so
  // promising that it would is worse than staying quiet.
  const rows = [], solo = new Map();
  for (const s of S.samples) {
    const why = failures(s);
    if (!why.length) rows.push(s);
    else if (why.length === 1) solo.set(why[0], (solo.get(why[0]) || 0) + 1);
  }
  S.hidden = solo;
  const { run, metric, dir } = S.sort;

  // While scrubbing epochs the metrics underneath are changing, and re-sorting on them
  // would shuffle rows out from under the cursor -- you would be watching a different
  // sample than the one you were looking at. Hold the order captured when the scrub
  // began; filters still apply, so rows can leave, but survivors keep their positions.
  if (S.frozenOrder) {
    // Anything not in the pinned order (a sample a filter change just admitted) sorts to
    // the end by sample id, rather than jumping into the middle of the frozen sequence.
    const rank = S.frozenOrder;
    const at = (s) => (rank.has(s.i) ? rank.get(s.i) : Number.MAX_SAFE_INTEGER);
    rows.sort((a, b) => at(a) - at(b) || a.i - b.i);
    S.order = rows;
    return;
  }

  if (run && S.runs[run]) {
    const m = METRICS.find((x) => x.key === metric);
    // "worst" means high for BCE/localization, LOW for IoU/contour, and largest |value|
    // for mass (over- and under-paint are both bad); the direction button is labelled
    // semantically so this inversion never lands on the user.
    const isAbs = m.worst === "abs";
    const desc = isAbs ? dir === "worst" : (m.worst === "high") === (dir === "worst");
    const key = (v) => (isAbs ? Math.abs(v) : v);
    const tbl = S.runs[run].samples;
    rows.sort((a, b) => {
      const x = tbl[a.i], y = tbl[b.i];
      // A missing prediction, or a metric undefined for this sample (localization on an
      // empty box), sinks to the end in BOTH directions.
      const xv = x && x[metric] != null ? x[metric] : null;
      const yv = y && y[metric] != null ? y[metric] : null;
      if (xv == null && yv == null) return a.i - b.i;
      if (xv == null) return 1;
      if (yv == null) return -1;
      const d = (key(yv) - key(xv)) * (desc ? 1 : -1);
      return d || a.i - b.i;
    });
  }
  S.order = rows;
}

/* ***** table ***** */

function statsFor(name) {
  const tbl = S.runs[name].samples, out = {};
  for (const m of METRICS) {
    const vals = [];
    for (const s of S.order) {
      const e = tbl[s.i];
      // COM distance is null on empty-GT samples (see data.load_run), so skipping
      // nulls here reproduces the training metric's own exclusion and keeps these
      // aggregates comparable with the numbers in the run's logs.
      if (!e || e[m.key] == null) continue;
      vals.push(e[m.key]);
    }
    if (!vals.length) { out[m.key] = null; continue; }
    const mean = vals.reduce((a, b) => a + b, 0) / vals.length;
    const sd = Math.sqrt(vals.reduce((a, b) => a + (b - mean) ** 2, 0) / vals.length);
    out[m.key] = { mean, sd, n: vals.length };
  }
  return out;
}

/* "which epoch am I looking at, out of how many the run trained for" -- always both, so
   the column says where it is without the reader having to remember whether the slider is
   scrubbed. The chip is also a control: clicking the second number pins every column to
   this run's last epoch, which is how you line the table up on one run's endpoint. */
function epochLabel(run) {
  const r = S.runs[run];
  const eps = r.epochs || [];
  const last = eps.length ? eps[eps.length - 1] : r.epoch;
  const shown = epochFor(run);
  const cur = shown == null ? last : shown;
  return `<span class="epchip${cur === last ? "" : " scrub"}">` +
    `<b>${cur}</b><span class="k">/</span>` +
    `<b class="goep" data-ep="${last}" title="Show every column at epoch ${last}">${last}</b>` +
    `<span class="k">ep</span></span>`;
}

/* Column reordering by dragging a header.

   Only the grip starts a drag, so clicking the name still cycles the sort and the whole
   header does not become an awkward click target. Position is a whole-column step --
   round(dx / columnWidth) -- rather than a hit test against midpoints, because columns
   are fixed-width and stepping keeps the dragged column locked to the slot it will land
   in instead of drifting between two of them. */
function moveRun(name, before) {
  if (name === before) return;
  const rest = S.runOrder.filter((n) => n !== name);
  const at = before == null ? rest.length : rest.indexOf(before);
  rest.splice(at < 0 ? rest.length : at, 0, name);
  S.runOrder = rest;
  // Frame stores are keyed by run name, not position, so they survive a reorder -- only
  // the DOM order changes. refresh() rebuilds the header and repaints every row's cells.
  refresh();
}

/* Pointer-events rather than HTML5 drag-and-drop.

   The native API cannot work here: refresh() destroys and recreates the whole .hrow, and
   any re-render mid-drag (the 15s poll, an epoch fetch) tore the dragstart element out of
   the document, which silently cancels the drag and loses the drop. Pointer events with
   setPointerCapture keep the gesture bound to the grip for its whole lifetime, so the
   drop always lands. It also lets the ENTIRE column move, not just its header. */
function makeDraggable(cell, name) {
  const grip = cell.querySelector(".hgrip");
  if (!grip) return;

  grip.onpointerdown = (e) => {
    if (e.button !== 0) return;
    e.preventDefault();
    grip.setPointerCapture(e.pointerId);

    const startX = e.clientX;
    const from = S.runOrder.indexOf(name);
    const colW = cell.getBoundingClientRect().width;
    let to = from;
    let moved = false;

    // Every cell of this column, header included, so the whole column travels together.
    // Row cells come from the pool's own _runCells rather than a :nth-of-type selector,
    // which would silently break if the fixed idx/gt cells before them ever changed.
    const heads = document.querySelectorAll(".hrow .hcell.run");
    const colCells = (i) => [heads[i], ...pool.map((r) => r._runCells[i])].filter(Boolean);

    const dragged = colCells(from);
    dragged.forEach((c) => c.classList.add("dragging"));
    document.body.classList.add("dragging-col");

    const onMove = (ev) => {
      const dx = ev.clientX - startX;
      if (!moved && Math.abs(dx) < 3) return;   // let a plain click through
      moved = true;
      // Which slot the pointer is over, as a whole-column step.
      const next = Math.max(0, Math.min(S.runOrder.length - 1,
        from + Math.round(dx / colW)));
      if (next !== to) {
        // Shift the columns the dragged one has passed over, so the gap opens up live.
        to = next;
        S.runOrder.forEach((_, i) => {
          if (i === from) return;
          const shift = (i > from && i <= to) ? -colW : (i < from && i >= to) ? colW : 0;
          colCells(i).forEach((c) => { c.style.transform = shift ? `translateX(${shift}px)` : ""; });
        });
      }
      dragged.forEach((c) => { c.style.transform = `translateX(${dx}px)`; });
    };

    const onUp = () => {
      grip.releasePointerCapture?.(e.pointerId);
      window.removeEventListener("pointermove", onMove);
      window.removeEventListener("pointerup", onUp);
      window.removeEventListener("pointercancel", onUp);
      // MUST come before the transforms are cleared below: the transform transition is
      // scoped to body.dragging-col (see style.css), so dropping the class first makes the
      // clearing instant. Clearing while it still applied animated every displaced column
      // back to its OLD position, and the re-render then snapped it to the new one -- the
      // distracting swap-on-release. Order is load-bearing, not incidental.
      document.body.classList.remove("dragging-col");
      // Clear inline transforms before re-rendering, or a recycled row keeps the offset.
      document.querySelectorAll(".hcell.run, .cell.run").forEach((c) => {
        c.style.transform = "";
        c.classList.remove("dragging");
      });
      if (moved && to !== from) {
        const rest = S.runOrder.filter((n) => n !== name);
        moveRun(name, rest[to] ?? null);
      }
    };

    window.addEventListener("pointermove", onMove);
    window.addEventListener("pointerup", onUp);
    // A cancelled pointer (browser gesture, window blur) must unwind exactly like a drop,
    // or the column would stay lifted with a stale transform.
    window.addEventListener("pointercancel", onUp);
  };
}

/* The ground-truth column collapses to true zero width. It sits BETWEEN the sticky index column
   and the runs, so the handle to bring it back lives in the index header cell (and on "g") rather
   than in the column itself. Width lives in a CSS var, so the header, every pooled row, and the
   sticky offsets all follow from one class -- no re-render, and no row heights change. */
function setGtCol(open) {
  document.body.classList.toggle("nogt", !open);
  try { localStorage.setItem("viz.gtcol", open ? "1" : "0"); } catch (_) {}
}

function renderHeader() {
  const h = document.createElement("div");
  h.className = "hrow";
  h.style.height = "auto";
  h.innerHTML = `<div class="hcell idx sticky-1">
      <button class="gtcol" id="gt-hide" title="Collapse ground truth" aria-label="Collapse ground truth">‹</button>
      <button class="gtcol" id="gt-show" title="Show ground truth" aria-label="Show ground truth">›</button>
    </div>
    <div class="hcell gt sticky-2">
      <div class="hname" style="cursor:default">Ground truth</div>
      <div class="hmeta">sample · overhead + mask</div></div>`;
  h.querySelector("#gt-hide").onclick = () => setGtCol(false);
  h.querySelector("#gt-show").onclick = () => setGtCol(true);

  for (const name of S.runOrder) {
    const r = S.runs[name], st = statsFor(name);
    const sorted = S.sort.run === name;
    const cell = document.createElement("div");
    cell.className = "hcell run" + (sorted ? " sorted" : S.sort.run ? " dim" : "");
    const warn = r.entry && r.entry.family === "unknown"
      ? `<span class="warnbadge" title="No eval split directories; dataset identity unconfirmed">?</span>` : "";
    const skipped = r.skipped_files.length
      ? ` · <span title="${r.skipped_files.join(", ")}">${r.skipped_files.length} file(s) skipped</span>` : "";
    // A run can pass the compatibility scan and still score nothing -- targets exist at
    // its grid but none decoded, or no predicted sample belongs to this dataset. Saying
    // which is the difference between "this run is broken" and "viz is broken".
    const why = r.reason && !r.n
      ? `<div class="hmeta empty" title="${r.reason}">empty: ${r.reason}</div>` : "";
    // Read status from the latest poll, not the snapshot taken when the run was added --
    // a run that finishes or crashes while open must update its badge.
    const live = S.meta.runs.find((x) => x.name === name);
    const status = (live && live.status) || "unknown";
    // Two lines: the name owns the first one (it is the column's identity and the longest
    // string), the epoch and status chips sit together on the second. They shared a line
    // with the name until the column narrowed to the sample image's width, where the chips
    // ate the name down to "norm-4-...". Grip and close ride the top-right corner.
    cell.innerHTML = `
      <div class="htop">
        <div class="hname" title="Click to cycle sort by this run">${name}${warn}</div>
        <span class="hgrip" title="Drag to reorder this column">⠿</span>
        <button class="hclose" title="Remove this run">&times;</button>
      </div>
      <div class="hchips">
        <span class="hep">${epochLabel(name)}</span>
        ${statusChip(status)}
      </div>
      ${why}
      ${skipped ? `<div class="hmeta">${skipped.replace(/^ · /, "")}</div>` : ""}
      <div class="hstats">${METRICS.map((m) => {
        const s = st[m.key];
        return `<span><span class="k">${m.short}</span> <b>${s ? fmt(s.mean, 3) : "–"}</b>${
          s ? `<span class="k">±${fmt(s.sd, 3)}</span>` : ""}</span>`;
      }).join("")}</div>
      <div class="hsort">
        <select>${METRICS.map((m) =>
          `<option value="${m.key}" ${sorted && S.sort.metric === m.key ? "selected" : ""}>${m.label}</option>`).join("")}</select>
        <button class="dir ${sorted ? "on" : ""}">${sorted ? (S.sort.dir === "worst" ? "↓ worst" : "↑ best") : "sort"}</button>
      </div>`;

    makeDraggable(cell, name);

    cell.querySelector(".hclose").onclick = (e) => {
      e.stopPropagation();          // must not also cycle the column's sort
      removeRun(name);
    };
    cell.querySelector(".hname").onclick = () => {
      // Selecting the name to copy it ends in a click here, which would re-sort the whole
      // table as a side effect of highlighting text. If the user just dragged out a
      // selection inside this name, that was the intent -- leave the sort alone.
      const sel = window.getSelection();
      if (sel && !sel.isCollapsed && sel.anchorNode
          && cell.querySelector(".hname").contains(sel.anchorNode)) return;
      if (S.sort.run !== name) S.sort = { run: name, metric: S.sort.metric, dir: "worst" };
      else if (S.sort.dir === "worst") S.sort.dir = "best";
      else S.sort = { run: null, metric: S.sort.metric, dir: "worst" };
      refresh(true);
    };
    cell.querySelector("select").onchange = (e) => {
      S.sort = { run: name, metric: e.target.value, dir: S.sort.run === name ? S.sort.dir : "worst" };
      refresh(true);
    };
    cell.querySelector(".dir").onclick = () => {
      if (S.sort.run !== name) S.sort = { run: name, metric: S.sort.metric, dir: "worst" };
      else S.sort.dir = S.sort.dir === "worst" ? "best" : "worst";
      refresh(true);
    };
    h.appendChild(cell);
  }
  return h;
}

/* Images are cached `immutable`, so the render version is part of the URL: bumping
   config.RENDER_VERSION is what lets a changed rendering reach browsers that already
   cached the old one. Table prediction cells no longer use images -- they are canvases
   drawn from /api/frames -- so this covers the ground-truth column and the modals. */
function gtMaskURL(sid) {
  const [h, w] = gtShape();
  // Always bg=0: the mask arrives with an alpha channel and the backdrop goes behind it
  // in CSS, exactly as in the prediction cells. Baking the photo in would freeze both
  // opacities into the cached image, and every slider notch would refetch the column.
  return `/api/gt_mask.png?sid=${sid}&bg=0` +
         `&rel=${S.view.relative ? 1 : 0}&shape=${h}x${w}&v=${S.renderVersion}`;
}

/* The backdrop is a CSS background behind the canvas (or behind the ground-truth image)
   rather than baked into the pixels: it is fetched once per sample and reused for every
   run and every epoch, which is most of why scrubbing costs no bandwidth.

   Two layers, not one: a flat veil of the cell colour sits OVER the photo and UNDER the
   canvas content, so --bg-op fades the backdrop without touching the mask drawn on top.
   Element `opacity` cannot do this -- it would take the mask down with the photo.

   Versioned like every other image: this response is `immutable` for a year, so a change
   to how the backdrop is sized would otherwise never reach a browser that already cached
   the old dimensions. */
function setBackdrop(el, sid) {
  const want = S.view.background
    ? `linear-gradient(var(--bd-veil), var(--bd-veil)), url(/api/backdrop/${sid}.jpg?v=${S.renderVersion})`
    : "";
  // Compared against a dataset copy rather than style.backgroundImage, which the browser
  // normalises (quotes, colour syntax) and so never matches the string we wrote.
  if (el.dataset.bd !== want) { el.style.backgroundImage = want; el.dataset.bd = want; }
}

function buildRow() {
  const el = document.createElement("div");
  el.className = "row";
  el.innerHTML = `<div class="cell idx sticky-1"><span class="idxn"></span></div>
    <div class="cell gt sticky-2 gtcell">
      <div class="ctitle gt-head"></div>
      <div class="mask gtwrap"><img class="gtimg" loading="lazy" decoding="async" alt=""></div>
      <div class="tags gt-foot"></div>
    </div>`;
  el._runCells = [];
  return el;
}

function syncRowCells(el) {
  while (el._runCells.length > S.runOrder.length) el.removeChild(el._runCells.pop());
  while (el._runCells.length < S.runOrder.length) {
    const c = document.createElement("div");
    c.className = "cell run";
    // One rendering path: every prediction cell is a canvas drawn from local mask values
    // over a CSS backdrop. Keeping a parallel server-PNG path meant two renderers that
    // had to agree on gamma, alpha and backdrop geometry -- and they drifted.
    // A second canvas for the ground-truth half, shown only in stacked mode.
    // Canvas dims come from the server's mask shape, never hardcoded: the grid is 20x40
    // on some datasets and 30x30 on others, and a fixed 40x20 buffer would letterbox the
    // mask into the wrong cells entirely.
    // Sized per column in paintRow, since each run may predict at its own grid; these
    // are just starting dimensions for a canvas that has not been painted yet.
    const { h: mh, w: mw } = S.lut;
    c.innerHTML = `<div class="ctitle run-head"></div>` +
      `<canvas class="mask predmask" width="${mw}" height="${mh}"></canvas>` +
      `<canvas class="mask truthmask" width="${mw}" height="${mh}" hidden></canvas>` +
      `<div class="tags subtags"></div>`;
    el.appendChild(c);
    el._runCells.push(c);
  }
}

function paintRow(el, rank) {
  const s = S.order[rank];
  el.style.transform = `translateY(${rank * rowH()}px)`;
  el.querySelector(".idxn").textContent = rank + 1;
  // Identity line: which sample, where it was captured, which speaker played. The 8
  // samples sharing a position differ only by speaker, so all three belong together.
  // The run columns no longer repeat this: the sample is the same across the whole row,
  // so saying it once in the sticky column leaves the run cells free to spend their
  // title on what actually differs between them (name, split, epoch).
  el.querySelector(".gt-head").innerHTML =
    `<span class="t1">Ground truth</span><span class="t2">${identity(s)}</span>`;

  // The ground-truth cell shows the same 20x40 target the runs predict, over the same
  // backdrop, so it is directly comparable with every prediction column. The speaker
  // view lives in the detail modal instead.
  // The wrapper carries the backdrop and the hover/tooltip identity; the <img> inside is
  // the bare mask, so `opacity` fades the cube without touching the photo behind it.
  const wrap = el.querySelector(".gtwrap");
  const img = wrap.querySelector(".gtimg");
  wrap.className = "mask gtwrap" + (S.view.background ? " bg" : " nobg");
  setBackdrop(wrap, s.i);
  const src = gtMaskURL(s.i);
  if (img.getAttribute("src") !== src) img.setAttribute("src", src);
  wrap.dataset.run = ""; wrap.dataset.sid = s.i;

  // Split is a property of each run's dataloader, not of the sample, so it is shown per
  // run column rather than here.
  const com = s.com_gt && s.com_gt[0] != null && s.com_gt[0] >= 0
    ? `${fmt(s.com_gt[0], 1)}, ${fmt(s.com_gt[1], 1)}` : "–";
  // Four chips have to share 224px without wrapping -- a second line would overflow the
  // fixed row height the virtualizer depends on. "com" is dropped from the label (the
  // value reads as a coordinate pair, and the modal spells it out) to buy that room.
  el.querySelector(".gt-foot").innerHTML =
    `<span class="tag" title="center of mass (row, col) in grid coords">${com}</span>` +
    `<span class="tag" title="${s.layout}">${shortLayout(s.layout)}</span>` +
    `<span class="tag">${s.n_objects} obj</span>` +
    (s.box ? `<span class="tag" title="box: ${s.box}">${s.box}</span>` : "");
  // Collapsed, the cell is a bare rail -- clicking it must not open the sample modal.
  el.querySelector(".gtcell").onclick = () => {
    if (!document.body.classList.contains("nogt")) openModal(rank);
  };

  syncRowCells(el);
  // Hoisted: depends only on S.runOrder, so computing it per cell rebuilt a Set ~140
  // times per repaint (28 pooled rows x 5 columns) at scroll frame rate.
  const mixed = mixedShapes();
  S.runOrder.forEach((name, k) => {
    const c = el._runCells[k];
    const e = S.runs[name].samples[s.i];
    const head = c.querySelector(".run-head");
    const m = c.querySelector(".mask");
    const chips = c.querySelector(".subtags");
    const [rh, rw] = runShape(name);

    // Split and epoch are both PER-CELL facts, which is why neither can be left to the
    // column header: dataloaders assign splits independently, so one sample can be train
    // in one run and held out in another, and runs save on different cadences, so at one
    // slider position two columns can legitimately show different epochs.
    const cellEp = epochFor(name);
    const shownEp = cellEp == null ? (S.runs[name].epochs || []).slice(-1)[0] : cellEp;
    const latest = cellEp == null;
    const split = e ? e.split : (S.runs[name].splitOf || {})[s.i];
    // The grid is shown only when columns disagree on it. Both metrics are
    // grid-normalized so the numbers share a scale, but a coarser grid is systematically
    // easier -- a small cross-size gap is not evidence of a better model, so the reader
    // needs to see which size they are looking at.
    const gs = mixed ? `<span class="tag gsz" title="mask grid">${rh}x${rw}</span>` : "";
    // The title is identity only: the run name (wrapped over two lines, never cut) and
    // the sample. Split and epoch describe THIS PREDICTION rather than the sample or the
    // run, so they go below the image with the metrics -- see `chips` further down.
    head.innerHTML =
      `<span class="t1 run" title="${name}">${name}</span>` +
      `<span class="t2">${identity(s)}</span>`;

    if (!e) {
      chips.innerHTML = "";
      m.hidden = true;
      m.style.backgroundImage = ""; m.dataset.bd = "";   // keep the cache flag honest
      if (!c._np) { c._np = document.createElement("div"); c._np.className = "nopred"; c.appendChild(c._np); }
      // Three different situations used to read "no prediction" alike, which made a
      // 250ms debounce look identical to data that was never written:
      //   loading  -- metrics for this epoch are in flight
      //   not saved -- the run covers this sample, but this epoch did not save it. The
      //               train loader uses drop_last + shuffle, so each epoch discards a
      //               different remainder (see src/model/dataset.py build_dataset).
      //   not in run -- the sample is outside this run's split entirely.
      const covered = (S.runs[name].splitOf || {})[s.i];
      const st = S.runs[name].metricsPending !== undefined ? ["loading", "loading…", "fetching metrics for this epoch"]
        : covered ? ["unsaved", "not saved", `this run covers this sample, but epoch ${shownEp ?? "?"} saved no prediction for it`]
        : ["nopred", "not in run", "this sample is not in this run's split"];
      c._np.className = "nopred " + st[0];
      c._np.textContent = st[1];
      c._np.title = st[2];
      c._np.hidden = false;
      return;
    }
    if (c._np) c._np.hidden = true;
    // Below the image, in reading order: first WHICH FRAME this is (the split this run
    // filed the sample under, and the epoch shown), then what it scored. The frame line
    // comes first because it qualifies everything after it -- a good MSE means something
    // different on a train sample than on a held-out one. Both chips sit at the natural
    // left edge with the rest of the metadata; nothing here is right-aligned, so the
    // column reads as a single left-hand stack instead of two drifting apart.
    // Predicted centre of mass then gets its own line, then the three metrics. All
    // four at full size do not fit 224px on one line, and shrinking them to fit made the
    // numbers hard to scan -- which is the one thing this column exists for. Splitting
    // keeps the metrics aligned across every run column at a readable size.
    chips.innerHTML =
      `${splitChip(split)}` +
      `<span class="epchip${latest ? "" : " scrub"}" title="epoch ${shownEp ?? "?"}${
        latest ? " (latest saved)" : " (scrubbed)"}"><b>${shownEp ?? "–"}</b>` +
      `<span class="k">ep</span></span>${gs}<i class="brk"></i>` +
      `<span class="tag com" title="predicted center of mass (row, col) in grid coords">${
        fmt(e.com[0], 1)}, ${fmt(e.com[1], 1)}</span><i class="brk"></i>` +
      METRICS.map((mm) => {
        const v = e[mm.key];
        return `<span class="tag"><span class="k">${mm.short}</span> <b>${
          v == null ? "–" : fmt(v, 3)}</b></span>`;
      }).join("");
    m.hidden = false;
    // Match the canvas buffer to THIS run's grid. Rows are recycled across runs, so a
    // pooled cell can arrive still sized for a column of a different resolution.
    if (m.width !== rw || m.height !== rh) { m.width = rw; m.height = rh; }
    m.dataset.run = name; m.dataset.sid = s.i;
    m.onclick = () => openNeighbors(name, s.i);
    m.className = "mask predmask" + (S.view.background ? " bg" : " nobg");
    setBackdrop(m, s.i);
    paintCanvas(m, name, s.i, epochFor(name));

    // Stacked: the target gets its own box directly below the prediction, so the two are
    // adjacent instead of the ground truth being columns away in the leftmost cell.
    const tm = c.querySelector(".truthmask");
    tm.hidden = S.view.mode !== "stacked";
    if (!tm.hidden) {
      tm.className = "mask truthmask" + (S.view.background ? " bg" : " nobg");
      setBackdrop(tm, s.i);
      if (tm.width !== rw || tm.height !== rh) { tm.width = rw; tm.height = rh; }
      const truth = S.truthCache[truthKey(s.i, rh, rw)];
      if (truth) drawMask(tm, truth, "truth", null, [rh, rw]);
      else ensureTruth(s.i, rh, rw);
    }
  });
}

/* Which rows are on screen. Row k occupies [hh + k*rowH(), hh + (k+1)*rowH()) in scroll
   space, so the sticky header's height comes off before dividing. Shared with the frame
   prefetch: if the two disagreed, it would fetch data for rows it never paints. */
function visibleRange(pad = 3) {
  const sc = $("#scroller");
  const hdr = sc.querySelector(".hrow");
  const top = Math.max(0, sc.scrollTop - (hdr ? hdr.offsetHeight : 0));
  return {
    first: Math.max(0, Math.floor(top / rowH()) - pad),
    last: Math.min(S.order.length, Math.ceil((top + sc.clientHeight) / rowH()) + pad),
  };
}

const pool = [];
function renderVisible() {
  const { first, last } = visibleRange();
  const need = Math.max(0, last - first);
  const rowsEl = $("#rows");
  while (pool.length < need) { const r = buildRow(); pool.push(r); rowsEl.appendChild(r); }
  pool.forEach((el, k) => {
    const rank = first + k;
    if (rank < last) { el.hidden = false; paintRow(el, rank); }
    else el.hidden = true;
  });
}

let raf = 0;
function onScroll() {
  if (raf) return;
  raf = requestAnimationFrame(() => { raf = 0; renderVisible(); });
}

function refresh(toTop = false) {
  // toTop is passed exactly by the sort controls, which is also the signal that the user
  // asked for a new order -- so it releases any pin the epoch scrubber holds.
  if (toTop) thawOrder();
  applyFilters();
  // Re-sorting puts different samples at rank 1, so keeping the old scroll offset would
  // strand the user mid-list; jump back to the top whenever the order itself changes.
  if (toTop) $("#scroller").scrollTop = 0;
  const head = $("#scroller").querySelector(".hrow");
  if (head) head.remove();
  const hdr = renderHeader();
  $("#scroller").insertBefore(hdr, $("#spacer"));

  // Rows are absolutely positioned, so they must start below the sticky header --
  // otherwise row 1 sits underneath it and cannot be scrolled into view. The header
  // grows with the number of run columns, so measure it rather than assume a height.
  const hh = hdr.offsetHeight;
  $("#rows").style.top = `${hh}px`;
  $("#spacer").style.height = `${hh + S.order.length * rowH()}px`;
  const nRuns = S.runOrder.length;
  // "3 of 3007" on its own reads as a broken tool. Naming the filter that is holding the
  // other 3004 back -- and making the name the button that switches it off -- turns
  // "why is nothing showing up" into one click, which is the question this readout is for.
  const held = [...S.hidden.entries()]
    .filter(([k]) => FILTERS[k])
    .sort((a, b) => b[1] - a[1]);
  $("#rowcount").innerHTML =
    `<b>${S.order.length}</b> of ${S.samples.length} samples` +
    ` · <b>${nRuns}</b> run${nRuns === 1 ? "" : "s"}` +
    (held.length
      ? `<span class="held">hidden by ` + held.map(([k, n]) => (FILTERS[k].clear
          ? `<button class="why" data-k="${k}" title="Show these ${n} — clears the ${FILTERS[k].label} filter, leaving the others alone">${FILTERS[k].label} +${n}</button>`
          : `<span class="why off" title="No filter is hiding these — no loaded run has a prediction for them">${FILTERS[k].label} +${n}</span>`)).join(" ") + `</span>`
      : "");
  $("#rowcount").querySelectorAll("button.why").forEach((b) => {
    b.onclick = () => clearFilter(b.dataset.k);
  });
  $("#empty").hidden = S.order.length > 0;
  renderChips();
  renderScatter();
  renderSpeakers();
  renderEpochPanel();
  renderVisible();
}

/* ***** sidebar: chips ***** */

/* `titleFn` is optional: when a chip needs to explain itself (which runs define this
   split, and why its count may be 0 for the run you are looking at) the explanation goes
   in the tooltip rather than onto the chip face, which has to stay scannable. */
function chipRow(host, values, set, labelFn, titleFn) {
  host.innerHTML = "";
  values.forEach(([v, n]) => {
    const b = document.createElement("button");
    b.className = "chip" + (set.has(v) ? "" : " off");
    b.innerHTML = `${labelFn(v)}<span class="n">${n}</span>`;
    if (titleFn) b.title = titleFn(v);
    b.onclick = () => { set.has(v) ? set.delete(v) : set.add(v); refresh(); };
    host.appendChild(b);
  });
}

/* ***** explicit id search *****
   Each accepted id becomes its own removable chip, so a list can be built up and pruned
   one entry at a time rather than re-typing the whole query. */

function renderFindChips() {
  const f = S.filters;
  const host = $("#find-chips");
  host.innerHTML = "";
  const add = (kind, set, val) => {
    const b = document.createElement("button");
    b.className = "chip find";
    // Terse prefixes: these chips stack up, so "s871" beats "sample 871" for width.
    b.innerHTML = `${kind === "sample" ? "s" : "p"}${val}<span class="x">&times;</span>`;
    b.title = "Remove";
    b.onclick = () => { set.delete(val); renderFindChips(); refresh(); };
    host.appendChild(b);
  };
  [...f.findSamples].sort((a, b) => a - b).forEach((v) => add("sample", f.findSamples, v));
  [...f.findPositions].sort((a, b) => a - b).forEach((v) => add("pos", f.findPositions, v));

  const n = f.findSamples.size + f.findPositions.size;
  $("#find-count").textContent = n ? `${n} active` : "";
  // The browsing filters are bypassed while a search is active; say so rather than
  // leaving the sidebar looking like it is lying.
  $("#sidebar").classList.toggle("searching", n > 0);
}

function bindFind() {
  const commit = (input, set, valid) => {
    // Accept a list, so pasting "871, 502 33" works as well as typing one at a time.
    const ids = input.value.split(/[^0-9]+/).filter(Boolean).map(Number);
    let added = 0;
    for (const id of ids) if (valid(id)) { set.add(id); added++; }
    if (added) { input.value = ""; renderFindChips(); refresh(true); }
    else if (ids.length) { input.classList.add("bad"); setTimeout(() => input.classList.remove("bad"), 600); }
  };
  const sampleIds = new Set(S.samples.map((s) => +s.sample_id));
  const posIds = new Set(S.samples.map((s) => +s.output_id));
  $("#find-sample").onkeydown = (e) => {
    if (e.key === "Enter") commit($("#find-sample"), S.filters.findSamples, (i) => sampleIds.has(i));
  };
  $("#find-pos").onkeydown = (e) => {
    if (e.key === "Enter") commit($("#find-pos"), S.filters.findPositions, (i) => posIds.has(i));
  };
  $("#find-clear").onclick = () => {
    S.filters.findSamples.clear();
    S.filters.findPositions.clear();
    renderFindChips();
    refresh();
  };
}

function renderChips() {
  const f = S.filters;
  renderFindChips();
  // Splits are counted per RUN, not pooled. Runs need not share a split vocabulary --
  // object_count_train12 has `2-obj` where prep-r0 has `2-cubes` over the same captures --
  // so a single tally attributed each sample to whichever run happened to be loaded first
  // and left the other run's chips reading 0. Each chip now counts every sample any run
  // filed under that name, which is a number that stays true no matter what else is open.
  const splits = new Set();
  S.runOrder.forEach((n) => S.runs[n].splits.forEach((s) => splits.add(s)));
  const counts = new Map();
  S.samples.forEach((s) => splitsOf(s.i).forEach((sp) => counts.set(sp, (counts.get(sp) || 0) + 1)));
  // Which runs use each name. Kept off the chip face -- a run name on every chip is the
  // kind of noise that made these labels unreadable before -- and put in the tooltip,
  // where it answers "why does this chip exist / why is it empty for what I'm looking at"
  // at the moment the question is actually asked.
  const owners = new Map();
  S.runOrder.forEach((n) => S.runs[n].splits.forEach((sp) => {
    if (!owners.has(sp)) owners.set(sp, []);
    owners.get(sp).push(n);
  }));
  chipRow($("#split-chips"), [...splits].sort().map((s) => [s, counts.get(s) || 0]), f.splits,
    (v) => v, (v) => `${v} — used by ${(owners.get(v) || []).join(", ") || "no loaded run"}`);
  chipRow($("#layout-chips"), uniq((s) => s.layout), f.layouts, (v) => v);
  chipRow($("#nobj-chips"), uniq((s) => s.n_objects), f.nObjects, (v) => `${v}`);
  chipRow($("#box-chips"), uniq((s) => s.box), f.boxes, (v) => v);
  // Contains-object: one chip per object TYPE, counted across every layout it appears
  // in, so "purple-cube" covers both the solo and the two-object scenes.
  const objCounts = new Map();
  S.samples.forEach((s) => {
    (s.objects && s.objects.length ? s.objects : ["(none)"])
      .forEach((o) => objCounts.set(o, (objCounts.get(o) || 0) + 1));
  });
  const objVals = [...objCounts.entries()].sort((a, b) => (a[0] > b[0] ? 1 : -1));
  chipRow($("#obj-chips"), objVals, f.objects, (v) => (v === "(none)" ? "empty" : shortLayout(v)));
  $("#obj-count").textContent = `${f.objects.size}/${objVals.length}`;
  $("#split-count").textContent = `${f.splits.size}/${splits.size}`;
  $("#layout-count").textContent = `${f.layouts.size}/${uniq((s) => s.layout).length}`;
  $("#nobj-count").textContent = `${f.nObjects.size}/${uniq((s) => s.n_objects).length}`;
  $("#box-count").textContent = `${f.boxes.size}/${uniq((s) => s.box).length}`;
  $("#spk-count").textContent = `${f.speakers.size}/8`;
  $("#metric-scope").textContent = S.sort.run ? `range applies to ${S.sort.run}` : "any loaded run in range";
}

/* ***** sidebar: speaker diagram *****
   Positions are the SPEAKER_POSITION constants that also draw the speaker into
   05_overhead_speaker.png, so this diagram and the photos agree exactly. */

const SPEAKERS = { 1: [1, 0], 2: [1, 0.7], 3: [0.8, 1], 4: [0.6, 1], 5: [0.4, 1], 6: [0.2, 1], 7: [0, 0.7], 8: [0, 0] };

function buildSpeakerDiagram() {
  const svg = $("#speakers");
  // Speakers sit OUTSIDE the box, so the margin has to clear the push-out distance plus
  // the circle radius (OFF + R), or every edge speaker is clipped by the viewport.
  const W = 236, H = 116, OFF = 14, R = 10, M = OFF + R + 1;
  svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
  const bw = W - 2 * M, bh = H - 2 * M;
  let out = `<rect class="box-outline" x="${M}" y="${M}" width="${bw}" height="${bh}" rx="4"/>
             <text class="box-label" x="${W / 2}" y="${H / 2 + 3}">box</text>`;
  for (const [id, [xf, yf]] of Object.entries(SPEAKERS)) {
    // y_frac has 0 at the BOTTOM (draw_speaker flips it), so invert for SVG coords.
    let cx = M + xf * bw, cy = M + (1 - yf) * bh;
    cx += (xf === 1 ? OFF : xf === 0 ? -OFF : 0);
    cy += (yf === 0 ? OFF : yf === 1 ? -OFF : 0);
    out += `<g class="spk" data-spk="${id}"><circle cx="${cx}" cy="${cy}" r="${R}"/><text x="${cx}" y="${cy}">${id}</text></g>`;
  }
  svg.innerHTML = out;
  svg.querySelectorAll(".spk").forEach((g) => {
    g.onclick = () => {
      const id = +g.dataset.spk, set = S.filters.speakers;
      set.has(id) ? set.delete(id) : set.add(id);
      refresh();
    };
  });
}

function renderSpeakers() {
  $("#speakers").querySelectorAll(".spk").forEach((g) => {
    g.classList.toggle("off", !S.filters.speakers.has(+g.dataset.spk));
  });
}

/* ***** sidebar: position scatter *****
   Points are avg_com in full-resolution image coords (row, col). Column maps to x and
   row to y so the plot is oriented like the overhead photo. */

// Rendered at 2x the sidebar's CSS width and scaled down by the SVG viewBox, which
// buys real separation between neighbouring positions without widening the panel.
const SC = { W: 472, H: 132, pad: 12, r: 3.4 };

/* The box is ~4.3x wider than it is deep. Stretching that to fill a square would
   collapse the horizontal spacing and make neighbouring positions unclickable, so the
   true aspect ratio is preserved and the plot is centred in whatever space is left. */
function scaleP() {
  const rs = S.positions.map((p) => p.r), cs = S.positions.map((p) => p.c);
  const r0 = Math.min(...rs), r1 = Math.max(...rs), c0 = Math.min(...cs), c1 = Math.max(...cs);
  const dr = r1 - r0 || 1, dc = c1 - c0 || 1;
  const { W, H, pad } = SC;
  const k = Math.min((W - 2 * pad) / dc, (H - 2 * pad) / dr);
  const ox = (W - dc * k) / 2, oy = (H - dr * k) / 2;
  return { x: (c) => ox + (c - c0) * k, y: (r) => oy + (r - r0) * k };
}

function buildScatter() {
  const svg = $("#scatter");
  svg.setAttribute("viewBox", `0 0 ${SC.W} ${SC.H}`);
  const sc = scaleP();
  svg.innerHTML = S.positions.map((p) =>
    `<circle class="pt" data-p="${p.i}" cx="${sc.x(p.c).toFixed(1)}" cy="${sc.y(p.r).toFixed(1)}" r="${SC.r}"/>`).join("")
    + `<circle class="pt-hot" r="${SC.r + 2.5}" hidden></circle>`
    // Two marks for the drag: a filled polygon previewing the region that will be
    // captured, and a polyline tracing the exact path drawn so far (a polygon alone
    // would silently close the shape and hide where the cursor actually went).
    + `<polygon class="lasso-fill" points="" hidden></polygon>`
    + `<polyline class="lasso-line" points="" hidden></polyline>`;

  let drag = null;
  const pt = (ev) => {
    const b = svg.getBoundingClientRect();
    return [((ev.clientX - b.left) / b.width) * SC.W, ((ev.clientY - b.top) / b.height) * SC.H];
  };

  // Positions sit close together, so the click target is the nearest point within a
  // generous radius rather than the dot itself, and a ring shows which one is armed.
  const nearest = (x, y) => {
    const s2 = scaleP();
    let best = -1, bd = 1e9;
    S.positions.forEach((p) => {
      const d = Math.hypot(s2.x(p.c) - x, s2.y(p.r) - y);
      if (d < bd) { bd = d; best = p.i; }
    });
    return bd < 14 ? best : -1;
  };

  /* `el.hidden = x` only reflects to the hidden ATTRIBUTE on HTML elements. On SVG
     elements it just sets a stray JS property, leaving the attribute -- and so the
     `[hidden] { display: none }` rule -- in place, which is why these never appeared.
     Toggle the attribute directly instead. */
  const show = (el, on) => (on ? el.removeAttribute("hidden") : el.setAttribute("hidden", ""));

  const drawLasso = (pts, closed) => {
    const s = pts.map((q) => `${q[0].toFixed(1)},${q[1].toFixed(1)}`).join(" ");
    const line = svg.querySelector(".lasso-line"), fill = svg.querySelector(".lasso-fill");
    line.setAttribute("points", s); show(line, true);
    if (closed) fill.setAttribute("points", s);
    show(fill, closed);
  };
  const clearLasso = () => {
    for (const c of [".lasso-line", ".lasso-fill"]) {
      const e = svg.querySelector(c);
      show(e, false); e.setAttribute("points", "");
    }
  };

  svg.addEventListener("pointerdown", (ev) => {
    svg.focus();
    svg.setPointerCapture(ev.pointerId);
    drag = { pts: [pt(ev)], add: ev.shiftKey, moved: false };
    show(svg.querySelector(".pt-hot"), false);
    drawLasso(drag.pts, false);              // show the trail from the first pixel
  });
  svg.addEventListener("pointermove", (ev) => {
    if (!drag) {
      const [x, y] = pt(ev), i = nearest(x, y), hot = svg.querySelector(".pt-hot");
      const p = i >= 0 ? S.positions[i] : null;
      if (p) { const s2 = scaleP(); hot.setAttribute("cx", s2.x(p.c)); hot.setAttribute("cy", s2.y(p.r)); }
      show(hot, !!p);
      return;
    }
    const p = pt(ev), last = drag.pts[drag.pts.length - 1];
    if (Math.hypot(p[0] - last[0], p[1] - last[1]) < 1.5) return;
    drag.pts.push(p); drag.moved = true;
    // Fill as soon as the path can enclose anything (3 points), so the region being
    // captured is visible while the lasso grows rather than only at the end.
    drawLasso(drag.pts, drag.pts.length >= 3);
    // Preview which points the current path would capture, so the selection is visible
    // before releasing rather than only after.
    if (drag.pts.length > 2) {
      const s2 = scaleP();
      svg.querySelectorAll(".pt").forEach((c) => {
        const q = S.positions[+c.dataset.p];
        c.classList.toggle("lassoed", inPoly(s2.x(q.c), s2.y(q.r), drag.pts));
      });
    }
  });
  svg.addEventListener("pointerleave", () => show(svg.querySelector(".pt-hot"), false));
  // A cancelled gesture (pointer leaves the window, touch interrupted) must not leave
  // the trail painted on screen.
  svg.addEventListener("pointercancel", () => {
    drag = null;
    clearLasso();
    svg.querySelectorAll(".pt.lassoed").forEach((c) => c.classList.remove("lassoed"));
  });
  svg.addEventListener("pointerup", (ev) => {
    if (!drag) return;
    clearLasso();
    svg.querySelectorAll(".pt.lassoed").forEach((c) => c.classList.remove("lassoed"));
    if (drag.moved && drag.pts.length > 2) {
      const hit = new Set(drag.add && S.filters.positions ? S.filters.positions : []);
      const sc2 = scaleP();
      S.positions.forEach((p) => { if (inPoly(sc2.x(p.c), sc2.y(p.r), drag.pts)) hit.add(p.i); });
      S.filters.positions = hit.size ? hit : null;
      S.activePos = hit.size ? [...hit][0] : -1;
    } else {
      const [x, y] = pt(ev);
      const best = nearest(x, y);
      if (best >= 0) {
        const cur = S.filters.positions;
        if (ev.ctrlKey || ev.metaKey) {
          const set = new Set(cur || []);
          set.has(best) ? set.delete(best) : set.add(best);
          S.filters.positions = set.size ? set : null;
        } else {
          S.filters.positions = new Set([best]);
        }
        S.activePos = best;
      }
    }
    drag = null;
    refresh();
  });

  // Arrow keys walk to the nearest point in that direction; points sit on an irregular
  // grid, so this uses an angular cone rather than a strict axis match.
  svg.addEventListener("keydown", (ev) => {
    const dirs = { ArrowLeft: [-1, 0], ArrowRight: [1, 0], ArrowUp: [0, -1], ArrowDown: [0, 1] };
    const d = dirs[ev.key];
    if (!d) return;
    ev.preventDefault();
    const sc2 = scaleP();
    if (S.activePos < 0) { S.activePos = S.positions.length ? 0 : -1; }
    else {
      const a = S.positions[S.activePos];
      const ax = sc2.x(a.c), ay = sc2.y(a.r);
      let best = -1, bs = 1e9;
      S.positions.forEach((p) => {
        if (p.i === a.i) return;
        const dx = sc2.x(p.c) - ax, dy = sc2.y(p.r) - ay;
        const proj = dx * d[0] + dy * d[1];
        if (proj <= 0.5) return;
        const perp = Math.abs(dx * d[1] - dy * d[0]);
        if (perp > proj * 1.2) return;              // stay within a ~50 degree cone
        const score = proj + perp * 2;
        if (score < bs) { bs = score; best = p.i; }
      });
      if (best >= 0) S.activePos = best;
    }
    if (S.activePos >= 0) S.filters.positions = new Set([S.activePos]);
    refresh();
  });

  $("#pos-clear").onclick = () => { S.filters.positions = null; S.activePos = -1; refresh(); };
}

function inPoly(x, y, poly) {
  let inside = false;
  for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
    const [xi, yi] = poly[i], [xj, yj] = poly[j];
    if ((yi > y) !== (yj > y) && x < ((xj - xi) * (y - yi)) / (yj - yi) + xi) inside = !inside;
  }
  return inside;
}

function renderScatter() {
  const sel = S.filters.positions;
  $("#scatter").querySelectorAll(".pt").forEach((c) => {
    const i = +c.dataset.p;
    c.classList.toggle("off", !!sel && !sel.has(i));
    c.classList.toggle("active", i === S.activePos);
  });
  const n = sel ? sel.size : S.positions.length;
  $("#pos-status").textContent = sel
    ? `${n} of ${S.positions.length} position${n === 1 ? "" : "s"} selected`
    : `all ${S.positions.length} positions`;
}

/* ***** sidebar: metric sliders ***** */

function buildSliders() {
  $("#sliders").innerHTML = METRICS.map((m) => `
    <div class="slider" data-k="${m.key}">
      <div class="top"><span class="nm">${m.label}</span><span class="val"></span></div>
      <div class="pair"><div class="track"></div><div class="fill"></div>
        <input type="range" class="lo" min="0" max="1000" value="0">
        <input type="range" class="hi" min="0" max="1000" value="1000"></div>
    </div>`).join("");
  $("#sliders").querySelectorAll(".slider").forEach((el) => {
    const k = el.dataset.k;
    const lo = el.querySelector(".lo"), hi = el.querySelector(".hi");
    const on = () => {
      let a = +lo.value, b = +hi.value;
      if (a > b) { [a, b] = [b, a]; }
      const [d0, d1] = S.domain[k];
      S.filters.ranges[k] = [d0 + (a / 1000) * (d1 - d0), d0 + (b / 1000) * (d1 - d0)];
      syncSliders();
      refresh();
    };
    lo.oninput = on; hi.oninput = on;
  });
  $("#metric-reset").onclick = () => {
    for (const m of METRICS) S.filters.ranges[m.key] = [...S.domain[m.key]];
    syncSliders(); refresh();
  };
}

function syncSliders() {
  $("#sliders")?.querySelectorAll(".slider").forEach((el) => {
    const k = el.dataset.k, d = S.domain[k], r = S.filters.ranges[k];
    if (!d || !r) return;
    const f = (v) => Math.round(((v - d[0]) / (d[1] - d[0] || 1)) * 1000);
    const a = f(r[0]), b = f(r[1]);
    el.querySelector(".lo").value = a;
    el.querySelector(".hi").value = b;
    el.querySelector(".val").textContent = `${fmt(r[0], 3)} – ${fmt(r[1], 3)}`;
    const fill = el.querySelector(".fill");
    fill.style.left = `${a / 10}%`;
    fill.style.width = `${(b - a) / 10}%`;
  });
}

/* ***** epoch frames *****
   Mask values are fetched as one fp16 blob per run (1600 bytes per cell per epoch --
   usually smaller than a PNG of the same cell) and drawn to <canvas> on the client. That
   makes scrubbing the epoch slider and playing the animation pure local work: no network
   round-trip and no server render per frame. */

async function fetchFrames(run, sids) {
  // Ask for an EXPLICIT epoch list rather than letting the server default to its own.
  // The blob is indexed by epoch position, so the labels must describe the bytes that
  // came back. Defaulting server-side meant a still-training run answered with more
  // epochs than the client had cached at load time: the client then indexed an 11-epoch
  // blob as if it held 5, so `epoch == null` read epoch 200 as "latest" and every newer
  // epoch fell off the end as index -1 and painted blank.
  const eps = (S.runs[run] && S.runs[run].epochs) || [];
  const q = eps.length ? `&epochs=${eps.join(",")}` : "";
  const r = await fetch(
    `/api/frames?run=${encodeURIComponent(run)}&sids=${sids.join(",")}${q}`);
  if (!r.ok) throw new Error("frames");
  const raw = new Uint16Array(await r.arrayBuffer());
  return {
    epochs: eps.slice(),
    sids: new Map(sids.map((s, i) => [s, i])),
    raw,
  };
}

/* Minimal IEEE half -> float. Only needed because DataView has no float16 reader. */
function half(u) {
  const s = (u & 0x8000) ? -1 : 1, e = (u >> 10) & 0x1f, m = u & 0x3ff;
  if (e === 0) return s * m * 5.9604644775390625e-8;
  if (e === 31) return m ? NaN : s * Infinity;
  return s * Math.pow(2, e - 15) * (1 + m / 1024);
}

/* This mask's own value range, for the relative colour view. Mirrors render.domain_of so
   a canvas cell and a server-rendered PNG of the same mask agree. Diff needs only the
   half-span, since its domain is symmetric about zero. */
function sampleDomain(v, truth, mode, n) {
  let lo = Infinity, hi = -Infinity, span = 0;
  for (let i = 0; i < n; i++) {
    const x = mode === "diff" ? v[i] - truth[i] : v[i];
    if (!Number.isFinite(x)) continue;
    if (x < lo) lo = x;
    if (x > hi) hi = x;
  }
  if (lo === Infinity) return null;
  span = Math.max(Math.abs(lo), Math.abs(hi), 1e-6);
  return { lo, hi, span };
}

/* Draws a (20,40) mask into a canvas using the server's own LUT, so a cell looks
   identical whether the client painted it or the server rendered a PNG. */
function drawMask(canvas, values, mode, truth, shape) {
  const { gamma, gain } = S.lut;
  // Grid comes from the CALLER, not the global default: runs trained at different
  // resolutions share one table, so each column draws at its own size.
  const [h, w] = shape || [S.lut.h, S.lut.w];
  const n = w * h;
  const ctx = canvas.getContext("2d");

  // Reused buffers: at ~36 cells x 5 fps a fresh ImageData per cell is pure GC churn.
  const off = drawMask._off || (drawMask._off = document.createElement("canvas"));
  if (off.width !== w || off.height !== h) { off.width = w; off.height = h; }
  const octx = off.getContext("2d");
  // Height matters too now that columns differ in shape: a cached 16x16 buffer reused for
  // a 30x30 cell would paint only the first 16 rows and leave the rest stale.
  const img = drawMask._img && drawMask._img.width === w && drawMask._img.height === h
    ? drawMask._img : (drawMask._img = octx.createImageData(w, h));
  const px = img.data;

  // Overlay draws the target underneath and the prediction over it, in one box; stacked
  // draws them as two boxes (handled by the caller, which paints each half).
  const soloLut = mode === "diff" ? S.lut.diff : mode === "truth" ? S.lut.truth : S.lut.pred;
  const layers = mode === "overlay"
    ? [{ v: truth, lut: S.lut.truth, k: 0.72 }, { v: values, lut: S.lut.pred, k: 0.82 }]
    : [{ v: values, lut: soloLut, k: gain }];

  ctx.clearRect(0, 0, w, h);
  // The cube slider scales every layer as it is composited, so the backdrop -- an element
  // background, outside the canvas -- keeps whatever the background slider gave it.
  ctx.globalAlpha = S.view.maskOp;
  for (const { v, lut, k } of layers) {
    // Per-sample domain, matching render.domain_of on the server. Computed per LAYER so
    // overlay scales truth and prediction independently, as the server does.
    const dom = S.view.relative ? sampleDomain(v, truth, mode, n) : null;
    for (let i = 0; i < n; i++) {
      let t, a;
      if (mode === "diff") {
        let d = Math.max(-1, Math.min(1, v[i] - truth[i]));
        // Magnitude only: the midpoint stays "no difference", so rescaling can never
        // move a cell onto the opposite arm of the diverging ramp.
        if (dom) d = Math.max(-1, Math.min(1, d / dom.span));
        const mag = Math.pow(Math.abs(d), gamma);
        t = 0.5 + 0.5 * Math.sign(d) * mag;
        a = mag;
      } else {
        let x = Math.max(0, Math.min(1, v[i]));
        if (dom) x = dom.hi - dom.lo > 1e-6 ? Math.max(0, Math.min(1, (x - dom.lo) / (dom.hi - dom.lo))) : 0;
        t = Math.pow(x, gamma);
        a = t;
      }
      const c = lut[Math.round(t * (lut.length - 1))] || [0, 0, 0];
      px[i * 4] = c[0]; px[i * 4 + 1] = c[1]; px[i * 4 + 2] = c[2];
      // Alpha-weighted so the backdrop photo shows through faint cells; k matches the
      // server's gain constants so a canvas cell composites identically to a server PNG.
      px[i * 4 + 3] = Math.round(a * k * 255);
    }
    // putImageData ignores compositing and would overwrite the layer beneath, so stage
    // on the offscreen canvas and drawImage, which composites.
    octx.putImageData(img, 0, 0);
    ctx.drawImage(off, 0, 0);
  }
  ctx.globalAlpha = 1;
}

/* Paints one cell from locally-held frame data, fetching the run's frames for the
   visible rows the first time they are needed. */
/* Modes that draw the target as well as the prediction: overlay composites them in one
   box, stacked puts them in two. */
const needsTruth = (m) => m === "overlay" || m === "stacked";

/* Reused across cells; avoids per-frame GC. Grown on demand rather than fixed at 20*40,
   which silently truncated every cell on a larger grid (30x30 is 900 values). */
let scratch = new Float32Array(20 * 40);
const scratchFor = (n) => (scratch.length >= n ? scratch : (scratch = new Float32Array(n)));

function paintCanvas(cv, run, sid, epoch) {
  if (!S.lut) return;
  const store = S.frameData[run];
  const [rh, rw] = runShape(run);
  const cells = rh * rw;
  // Every early return clears first. A canvas keeps its last drawing until something
  // overwrites it, and rows are RECYCLED, so leaving it alone shows the previous
  // occupant's prediction -- or, once a filter change brings in rows the current store
  // was not fetched for, a stale mask that never gets repainted.
  const blank = () => cv.getContext("2d").clearRect(0, 0, cv.width, cv.height);
  if (!store || !store.sids.has(sid)) { blank(); ensureFrames(run); return; }
  const eps = store.epochs;
  // A store fetched before the run advanced does not contain the newer epochs, so asking
  // for one has to trigger a refetch rather than paint blank. Without this a training
  // run's current epoch stayed empty until some unrelated scroll happened to replace the
  // store -- which is why scrolling appeared to "fix" it.
  const live = (S.runs[run] && S.runs[run].epochs) || [];
  if (eps.length && live.length > eps.length) { blank(); ensureFrames(run); return; }
  const ei = epoch == null ? eps.length - 1 : eps.indexOf(epoch);
  const si = store.sids.get(sid);
  if (ei < 0) { blank(); return; }
  const off = (ei * store.sids.size + si) * cells;
  const buf = scratchFor(cells);
  for (let i = 0; i < cells; i++) buf[i] = half(store.raw[off + i]);
  if (Number.isNaN(buf[0])) { blank(); return; }   // run has no prediction here
  // In stacked mode this canvas is the prediction half only -- the target is drawn into
  // its own canvas below, so it needs no truth here.
  const mode = S.view.mode === "stacked" ? "pred" : S.view.mode;
  const wantTruth = mode === "diff" || needsTruth(mode);
  // Truth is cached per (sample, grid): the same sample has a different target array at
  // each resolution, so keying on sid alone would diff against the wrong-shaped mask.
  const tkey = truthKey(sid, rh, rw);
  const truth = wantTruth ? S.truthCache[tkey] : null;
  if (wantTruth && !truth) { blank(); ensureTruth(sid, rh, rw); return; }
  drawMask(cv, buf, mode, truth, [rh, rw]);
}

/* Fetch frames for whatever rows are on screen, once per run.

   The window is recomputed from the CURRENT scroll position each time, and a second call
   is allowed to queue while one is in flight: scrolling fast, or jumping to an epoch whose
   sample set differs, moves the visible rows out from under the request that is already
   running. Without the re-check those rows would stay blank, because paintCanvas asks for
   frames and the pending flag would swallow the request. */
const framesPending = new Set();
const framesStale = new Set();
async function ensureFrames(run) {
  if (framesPending.has(run)) { framesStale.add(run); return; }
  framesPending.add(run);
  try {
    do {
      framesStale.delete(run);
      const first = Math.max(0, Math.floor(Math.max(0, $("#scroller").scrollTop) / rowH()) - 4);
      const sids = S.order.slice(first, first + 24).map((s) => s.i);
      if (!sids.length) return;
      const d = await fetchFrames(run, sids);
      // Replace rather than merge: a store holds one contiguous window, and its raw blob
      // is indexed by position, so windows cannot be concatenated without re-laying it out.
      S.frameData[run] = d;
      renderVisible();
    } while (framesStale.has(run));
  } finally {
    framesPending.delete(run);
    framesStale.delete(run);
  }
}

const truthPending = new Set();
async function ensureTruth(sid, h, w) {
  const key = truthKey(sid, h, w);
  if (truthPending.has(key)) return;
  truthPending.add(key);
  try {
    // `shape` picks which resolution's target to return, so a 16x16 column and a 30x30
    // column each diff against a mask of their own size.
    const q = h ? `&shape=${h}x${w}` : "";
    const d = await api(`/api/values?sid=${sid}${q}`);
    S.truthCache[key] = d.v.flat();
    renderVisible();
  } finally {
    truthPending.delete(key);
  }
}

/* ***** epoch slider + playback *****
   The union of every loaded run's epochs, so one slider drives all columns. A run
   without that exact epoch falls back to its nearest earlier one. */

/* Memoized: epochFor() calls this once per run per cell, so during playback an
   un-cached version rebuilt and sorted the union ~36 times a frame. */
let epochsCache = null;
function allEpochs() {
  if (epochsCache) return epochsCache;
  const set = new Set();
  S.runOrder.forEach((n) => (S.runs[n].epochs || []).forEach((e) => set.add(e)));
  return (epochsCache = [...set].sort((a, b) => a - b));
}
const invalidateEpochs = () => { epochsCache = null; };

function epochFor(run) {
  const eps = allEpochs();
  if (S.epochIdx == null || !eps.length) return null;      // null = the run's latest
  const want = eps[Math.min(S.epochIdx, eps.length - 1)];
  const mine = S.runs[run].epochs || [];
  let best = null;
  for (const e of mine) if (e <= want) best = e;
  return best == null ? (mine[0] ?? null) : best;
}

function renderEpochPanel() {
  const eps = allEpochs();
  $("#epoch-panel").hidden = eps.length < 2;
  if (eps.length < 2) return;
  const sl = $("#epoch-slider");
  sl.max = eps.length - 1;
  if (S.epochIdx != null) sl.value = Math.min(S.epochIdx, eps.length - 1);
  else sl.value = eps.length - 1;
  const i = S.epochIdx == null ? eps.length - 1 : +sl.value;
  const latest = S.epochIdx == null || i === eps.length - 1;
  $("#epoch-label").innerHTML =
    `<b>ep ${eps[i]}</b>` + (latest ? ` <span class="k">latest</span>` : "");
  // The slider's right end is the furthest epoch any loaded run reached, which is what
  // the track is scaled to.
  $("#epoch-max").textContent = eps[eps.length - 1];
  $("#epoch-play").textContent = S.playing ? "❚❚" : "▶";
  $("#epoch-play").title = S.playing ? "Pause" : "Play through all epochs";
  // Greyed at the ends so the control says where you are without you having to test it.
  $("#epoch-prev").disabled = i <= 0;
  $("#epoch-next").disabled = i >= eps.length - 1;
}

/* One epoch at a time, because the slider is far too coarse to land on a specific epoch
   when a run has dozens of them. Stepping stops playback: the two controls both drive
   epochIdx, and letting the timer keep firing would yank the frame back. */
function stepEpoch(delta) {
  const eps = allEpochs();
  if (eps.length < 2) return;
  S.playing = false;
  clearInterval(playTimer);
  // null means "pinned to latest", which is the last index for stepping purposes.
  const cur = S.epochIdx == null ? eps.length - 1 : S.epochIdx;
  setEpochIdx(cur + delta);
}

/* Press-and-hold on the epoch arrows, matching a keyboard key's auto-repeat: one step on
   press, a pause to keep single clicks single, then a steady stream.

   Repeat is driven by requestAnimationFrame rather than setInterval so it cannot outrun
   the display or pile up work in a background tab. Each step only repaints canvases from
   frame data already in memory; the metrics fetch behind it is debounced at 250ms, so
   holding the button costs one request when you let go, not one per epoch. */
const HOLD_DELAY = 350;   // before repeat begins
const HOLD_EVERY = 60;    // ~16 epochs/sec while held

function holdToRepeat(btn, step, focusTarget) {
  let raf = 0, timer = 0;
  const stop = () => {
    cancelAnimationFrame(raf); clearTimeout(timer); raf = timer = 0;
  };
  btn.addEventListener("pointerdown", (e) => {
    if (e.button !== 0 || btn.disabled) return;
    // preventDefault stops the drag-select that a hold would otherwise start, but it also
    // suppresses the focus the click would have given us -- which left focus on <body>, so
    // the arrow keys did nothing after clicking these buttons. Hand focus to the slider
    // instead, so keyboard stepping continues from wherever the buttons left off.
    e.preventDefault();
    if (focusTarget && !focusTarget.disabled) focusTarget.focus();
    btn.setPointerCapture(e.pointerId);   // keep repeating if the cursor slides off
    step();
    timer = setTimeout(() => {
      let last = 0;
      const tick = (t) => {
        // A disabled button means we hit the first or last epoch -- stop there rather
        // than spinning against the clamp.
        if (btn.disabled) return stop();
        if (t - last >= HOLD_EVERY) { last = t; step(); }
        raf = requestAnimationFrame(tick);
      };
      raf = requestAnimationFrame(tick);
    }, HOLD_DELAY);
  });
  ["pointerup", "pointercancel", "pointerleave"].forEach((ev) => btn.addEventListener(ev, stop));
}

/* Jump the whole table to one epoch, from a cell that is already showing it. The value
   comes from a run's own epoch list, which may not be in the union at that exact number
   if that run saves on a different cadence -- so snap to the nearest union entry at or
   below it, the same rule epochFor uses to pick a run's frame. */
function gotoEpoch(ep) {
  const eps = allEpochs();
  if (!eps.length) return;
  let idx = -1;
  for (let k = 0; k < eps.length; k++) if (eps[k] <= ep) idx = k;
  if (idx < 0) idx = 0;
  S.playing = false;
  clearInterval(playTimer);
  setEpochIdx(idx);
}

/* Every epoch change -- slider, steppers, playback -- goes through here, so this is the
   one place the order needs pinning. */
function setEpochIdx(i) {
  const eps = allEpochs();
  // Pin on entering a scrub; i == null is the un-scrubbed "latest" view, which sorts on
  // real metrics again.
  if (i == null) S.frozenOrder = null;
  else if (!S.frozenOrder) S.frozenOrder = new Map(S.order.map((s, k) => [s.i, k]));
  S.epochIdx = i == null ? null : Math.max(0, Math.min(i, eps.length - 1));
  renderEpochPanel();
  // Before renderVisible, not after: this marks which columns have metrics in flight, and
  // the repaint reads that to tell "loading" from "this epoch saved nothing". The fetch
  // itself is still debounced, so this costs nothing extra.
  fetchEpochMetrics();
  renderVisible();          // masks repaint immediately from local frame data
}

/* Re-sorting or re-filtering is an explicit request for a new order, so it releases the
   pin the epoch scrubber put on it. */
const thawOrder = () => { S.frozenOrder = null; };

/* Metrics belong to an epoch as much as the masks do. Without this the table would show
   early-training masks captioned with final-epoch mse/iou and sorted by them -- the most
   misleading state the scrubber can produce. Debounced so dragging the slider or playing
   does not fire a request per frame. */
let metricsTimer = 0;
function fetchEpochMetrics() {
  clearTimeout(metricsTimer);
  // Marked SYNCHRONOUSLY, before the debounce: callers repaint immediately and only then
  // call this, so deferring the marker by 250ms would leave that first repaint reading
  // the previous epoch's `samples` and reporting "not saved" for data that is merely
  // still on its way.
  for (const name of S.runOrder) {
    const cur = S.runs[name];
    const ep = epochFor(name);
    if (cur && cur.metricsEpoch !== ep) cur.metricsPending = ep;
  }
  metricsTimer = setTimeout(async () => {
    const jobs = S.runOrder.map(async (name) => {
      const ep = epochFor(name);
      const cur = S.runs[name];
      if (cur.metricsEpoch === ep) { if (cur.metricsPending === ep) cur.metricsPending = undefined; return false; }
      const q = ep == null ? "" : `?epoch=${ep}`;
      cur.metricsPending = ep;
      try {
        const d = await api(`/api/run/${encodeURIComponent(name)}${q}`);
        // Only the per-epoch metrics move. splitOf/splits describe which samples the run
        // covers, which is epoch-independent -- overwriting them here would make rows
        // disappear at epochs that saved a different subset.
        cur.samples = d.samples;
        cur.metricsEpoch = ep;
        return true;
      } catch {
        return false;
      } finally {
        // Only clear if no newer request overtook this one, or the last drag frame would
        // wipe the pending marker belonging to the request still in flight.
        if (cur.metricsPending === ep) cur.metricsPending = undefined;
      }
    });
    if ((await Promise.all(jobs)).some(Boolean)) {
      // Early-training metrics live on a completely different scale from final ones (at
      // epoch 0 every mse here is ~0.25, while the trained run spans 0..0.09). The
      // sliders were built from the latest epoch, so without widening them to admit the
      // scrubbed values every row fails the range test and the whole table -- ground
      // truth included -- empties out at epoch 0.
      recomputeDomains({ preserve: true });
      refresh();
    }
  }, 250);
}

let playTimer = 0;
function togglePlay() {
  S.playing = !S.playing;
  clearInterval(playTimer);
  if (S.playing) {
    const eps = allEpochs();
    if (S.epochIdx == null || S.epochIdx >= eps.length - 1) S.epochIdx = 0;
    // Frames are already local, so this is just an index advance -- no fetch per step.
    playTimer = setInterval(() => {
      const n = allEpochs().length;
      if (S.epochIdx >= n - 1) { S.playing = false; clearInterval(playTimer); renderEpochPanel(); return; }
      setEpochIdx(S.epochIdx + 1);
    }, 220);
  }
  renderEpochPanel();
}

/* ***** live updates *****
   Runs keep training while viz is open. Poll the (cheap, server-throttled) run list;
   when a loaded run's epoch advances, refetch its metrics and re-render its masks. */

const POLL_MS = 15000;

async function poll() {
  // A refresh() mid-drag rebuilds the header under the pointer, which strands the
  // dragged column's inline transforms and drops the gesture. The poll is a background
  // nicety; skipping one tick costs nothing.
  if (document.body.classList.contains("dragging-col")) return;
  let meta;
  try {
    meta = await api("/api/runs");
  } catch {
    return;                            // server restarting; try again next tick
  }
  const prevNames = new Set(S.meta.runs.map((r) => r.name));
  const prevStatus = new Map(S.meta.runs.map((r) => [r.name, r.status]));
  S.meta = meta;
  // A loaded run going running -> done/crashed only shows in its column header.
  const statusChanged = S.runOrder.some((n) => {
    const cur = meta.runs.find((r) => r.name === n);
    return cur && prevStatus.get(n) !== cur.status;
  });

  const advanced = [];
  for (const name of S.runOrder) {
    const cur = meta.runs.find((r) => r.name === name);
    if (cur && S.runs[name] && cur.epoch != null && cur.epoch !== S.runs[name].epoch) {
      advanced.push(name);
    }
  }
  for (const name of advanced) await addRun(name, true);

  if (advanced.length) {
    S.frameData = {};   // new epochs exist; refetch mask values
    // Mask URLs are cached immutable, so a new epoch needs a new URL to be fetched.
    S.epochTag = (S.epochTag || 0) + 1;
    // New metrics can reorder a sorted table. Keep whichever sample is at the top of the
    // viewport in view, so rows don't slide out from under the user mid-read.
    const sc = $("#scroller");
    const hdr = sc.querySelector(".hrow");
    const anchorRank = Math.floor(Math.max(0, sc.scrollTop - (hdr ? hdr.offsetHeight : 0)) / rowH());
    const anchor = S.order[anchorRank];
    refresh();
    if (anchor) {
      const next = S.order.indexOf(anchor);
      if (next >= 0 && next !== anchorRank) {
        sc.scrollTop += (next - anchorRank) * rowH();
        renderVisible();
      }
    }
  }
  const added = meta.runs.filter((r) => r.compatible && !prevNames.has(r.name));
  if (added.length && pickerOpen()) openPicker();   // keep an open picker live
  if (!advanced.length && statusChanged) refresh();
}

/* ***** run picker *****
   Loaded runs are their own columns, so there is no separate list of them: each column
   header carries its own close button. */

let pickCur = -1;               // highlighted row, for arrow-key selection

function pickerOpen() { return !$("#run-list").hidden; }

function closePicker() {
  $("#run-list").hidden = true;
  $("#run-search").setAttribute("aria-expanded", "false");
  pickCur = -1;
}

/* Rows that can actually be clicked -- the arrow keys skip incompatible and already-added
   runs rather than parking on a dead row. */
function pickable() { return [...$("#run-list").querySelectorAll(".ritem.ok")]; }

function markCur() {
  const rows = pickable();
  rows.forEach((d, i) => d.classList.toggle("cur", i === pickCur));
  if (pickCur >= 0 && rows[pickCur]) rows[pickCur].scrollIntoView({ block: "nearest" });
}

function openPicker() {
  const list = $("#run-list"), q = $("#run-search");
  const term = q.value.trim().toLowerCase();
  list.innerHTML = "";
  const hits = S.meta.runs.filter((r) => !term || r.name.toLowerCase().includes(term));
  if (!hits.length) list.innerHTML = `<div class="none">no run matches "${term}"</div>`;
  hits.forEach((r) => {
    const added = !!S.runs[r.name];
    const ok = r.compatible && !added;
    const d = document.createElement("div");
    d.className = "ritem" + (r.compatible ? (added ? " added" : "") : " bad") + (ok ? " ok" : "");
    d.innerHTML = `<div><div class="nm">${r.name}</div>
      <div class="sub">${statusChip(r.status)} ${r.compatible
        ? `ep ${r.epoch ?? "?"} · ${r.eval_splits.length} eval splits` : r.reason}</div></div>
      <span class="badge">${added ? "added" : r.compatible ? "add" : "unavailable"}</span>`;
    if (ok) d.onclick = () => pick(r.name);
    list.appendChild(d);
  });
  pickCur = -1;
  list.hidden = false;
  q.setAttribute("aria-expanded", "true");
}

async function pick(name) {
  $("#run-search").value = "";
  closePicker();
  await addRun(name);
}

/* ***** hover tooltip *****
   Grid values are fetched on first hover for a cell and memoized; never prefetched. */

const valueCache = new Map();
async function valuesFor(run, sid, mode) {
  // The ground-truth cell has no run to take a grid from, and /api/values defaults to the
  // primary shape -- which would hand back a differently-sized array than the [row, col]
  // the tooltip just computed off gtShape(), reporting the wrong cell's value. Runs carry
  // their own shape server-side, so only the GT branch needs the hint. Part of the cache
  // key: the same sample has a different array at each resolution.
  const [gh, gw] = gtShape();
  const q = run ? "" : `&shape=${gh}x${gw}`;
  const k = `${run}|${sid}|${mode}${q}`;
  if (!valueCache.has(k))
    valueCache.set(k, api(`/api/values?sid=${sid}&run=${encodeURIComponent(run)}&mode=${mode}${q}`));
  return valueCache.get(k);   // {v} or, in overlay/stacked mode, {v, t}
}

function bindTooltip() {
  const tip = $("#tooltip");
  let cur = null;
  $("#scroller").addEventListener("mousemove", async (ev) => {
    // dataset.run is "" on the ground-truth cell, which still has values to show.
    const img = ev.target.closest?.(".mask");
    if (!img || img.dataset.sid === undefined) { tip.hidden = true; cur = null; return; }
    const b = img.getBoundingClientRect();
    // The hovered cell's OWN grid: every cell is stretched to the same box, so a 30x30
    // column and a 20x40 one need different divisors to turn a cursor position into
    // [row, col]. Using the global default indexed the wrong cell on any run whose grid
    // is not the default -- and out of bounds on a coarser one.
    const [mh, mw] = img.dataset.run ? runShape(img.dataset.run) : gtShape();
    // Clamp rather than index straight from the ratio. Under browser zoom the rect is
    // fractional while `pixelated` snaps the painted cells to whole device pixels, so the
    // two disagree by up to a cell at the edges -- which read as a shifted mask.
    const frac = (p, lo, size) => Math.min(0.999999, Math.max(0, (p - lo) / size));
    const col = Math.floor(frac(ev.clientX, b.left, b.width) * mw);
    const row = Math.floor(frac(ev.clientY, b.top, b.height) * mh);
    if (ev.clientX < b.left || ev.clientX > b.right ||
        ev.clientY < b.top || ev.clientY > b.bottom) { tip.hidden = true; return; }
    // Ground truth has no prediction to diff against, so it always reports raw values.
    const mode = img.dataset.run ? S.view.mode : "pred";
    const key = `${img.dataset.run}|${img.dataset.sid}|${mode}`;
    if (cur !== key) { cur = key; tip._v = await valuesFor(img.dataset.run, img.dataset.sid, mode); }
    const v = tip._v?.v?.[row]?.[col];
    const t = tip._v?.t?.[row]?.[col];
    tip.hidden = false;
    // 4dp, matching what /api/values actually sends -- it rounds to 4, so this is the
    // full available precision and no more. At 3dp over half the cells of a low-confidence
    // run printed a flat "0.000", which hid real 10x differences between them.
    tip.textContent = `[${row},${col}] ` + (v == null ? "–" : v.toFixed(4)) +
      (t == null ? "" : ` pred · ${t.toFixed(4)} true`);
    tip.style.left = `${ev.clientX + 12}px`;
    tip.style.top = `${ev.clientY + 14}px`;
  });
  $("#scroller").addEventListener("mouseleave", () => { tip.hidden = true; });
}

/* ***** detail modal ***** */

async function openModal(rank) {
  S.modalRow = rank;
  const s = S.order[rank];
  if (!s) return;
  $("#m-prev").hidden = $("#m-next").hidden = false;
  // Same grid as the table's ground-truth column, so the COM the modal prints is in the
  // coordinate system the row beside it was read in.
  const [dh, dw] = gtShape();
  const d = await api(`/api/detail/${s.i}?shape=${dh}x${dw}`);
  $("#m-title").textContent = `Sample ${d.sample_id}`;
  // Named by the server rather than hardcoded: the default grid is chosen from the data.
  const grid = d.grid ? `${d.grid[0]}×${d.grid[1]}` : `${S.lut.h}×${S.lut.w}`;
  const coms = Object.entries(d.coms || {})
    .map(([k, v]) => `<dt>${k}</dt><dd>${fmt(v[0], 1)}, ${fmt(v[1], 1)}</dd>`).join("");
  const objs = Object.entries(d.objects || {})
    .map(([k, v]) => `<dt>${k}</dt><dd>${v}</dd>`).join("") || `<dt>—</dt><dd>empty</dd>`;

  $("#m-body").innerHTML = `
    <div class="msec">
      <h3>Overhead — mask, center of mass, speaker</h3>
      <img class="hero" src="/api/overhead/${s.i}.png" alt="overhead view of sample ${d.sample_id}">
      <p class="note">${d.description || ""}</p>
    </div>
    <div class="mgrid">
      <div class="msec">
        <h3>Object</h3>
        <dl class="kv">
          ${objs}
          <dt>count</dt><dd>${d.n_objects}</dd>
          <dt>layout</dt><dd>${d.layout}</dd>
          <dt>box</dt><dd>${d.box}</dd>
        </dl>
        <h3 style="margin-top:14px">Center of mass</h3>
        <dl class="kv">
          ${coms}
          <dt>average</dt><dd>${d.avg_com ? `${fmt(d.avg_com[0], 1)}, ${fmt(d.avg_com[1], 1)}` : "–"}</dd>
          <dt>on ${grid} grid</dt><dd>${d.com_gt_grid ? `${fmt(d.com_gt_grid[0], 2)}, ${fmt(d.com_gt_grid[1], 2)}` : "–"}</dd>
        </dl>
        <p class="note">Per-object and average are full-resolution image coordinates (row, col);
          the grid value is in ${grid} target space, as used by the table metrics.</p>
      </div>
      <div class="msec">
        <h3>Audio</h3>
        <div class="audio"><label>Original — played chirp</label>
          ${d.has.original ? `<audio controls preload="none" src="/api/audio/${s.i}/original"></audio>` : `<p class="note">not available</p>`}</div>
        <div class="audio"><label>Recovered — from laser vibration</label>
          ${d.has.recovered ? `<audio controls preload="none" src="/api/audio/${s.i}/recovered"></audio>` : `<p class="note">not available</p>`}</div>
        <dl class="kv">
          <dt>speaker</dt><dd>${d.speaker}</dd>
          <dt>min freq</dt><dd>${d.min_freq} Hz</dd>
          <dt>max freq</dt><dd>${d.max_freq} Hz</dd>
        </dl>
        ${d.has.spectrogram ? `<img class="specimg" src="/api/vibration/${s.i}/spectrogram.png" alt="spectrogram">` : ""}
        ${d.has.fft ? `<img class="specimg" src="/api/vibration/${s.i}/fft.png" alt="FFT">` : ""}
      </div>
    </div>`;
  $("#modal").hidden = false;
}

/* ***** predicted-mask modal *****
   Ranks ground-truth samples by how close their center of mass is to what this run
   predicted, which shows whether a prediction looks like a different scene than its own
   target. Positions are deduped: 8 samples share each one, differing only by speaker. */

async function openNeighbors(run, sid) {
  const d = await api(`/api/neighbors?run=${encodeURIComponent(run)}&sid=${sid}&k=5`);
  const dc = (a, b) => `${fmt(a, 1)}, ${fmt(b, 1)}`;
  const off = Math.hypot(d.pred_com[0] - d.gt_com[0], d.pred_com[1] - d.gt_com[1]);

  const list = (rows, cls) => rows.map((x) => `
    <li class="nb ${cls}" data-i="${x.i}">
      <img src="/api/gt_mask.png?sid=${x.i}&bg=1&shape=${d.shape[0]}x${d.shape[1]}&v=${S.renderVersion}" loading="lazy" alt="">
      <div class="nbmeta">
        <div class="nbtop"><b>${+x.sample_id}</b> <span class="tag">pos ${+x.output_id}</span></div>
        <div class="nbsub">d ${fmt(x.distance, 3)} · com ${dc(x.com[0], x.com[1])}</div>
        <div class="nbsub">${shortLayout(x.layout)} · ${x.n_objects} obj</div>
      </div>
    </li>`).join("");

  const v = `v=${S.renderVersion}`;
  const ep = S.runs[run] ? `&ep=${S.runs[run].epoch}` : "";
  const e = S.runs[run] && S.runs[run].samples[sid];

  const rel = S.view.relative ? 1 : 0;
  $("#m-title").innerHTML = `Sample ${+d.sample_id} — <span class="mrun">${run}</span>`;
  $("#m-body").innerHTML = `
    <div class="msec">
      <h3>This prediction</h3>
      <div class="nbviews">
        <figure><img src="/api/mask.png?run=${encodeURIComponent(run)}&sid=${sid}&mode=pred&bg=1&rel=${rel}&${v}${ep}" alt="">
          <figcaption>predicted</figcaption></figure>
        <figure><img src="/api/gt_mask.png?sid=${sid}&bg=1&rel=${rel}&shape=${d.shape[0]}x${d.shape[1]}&${v}" alt="">
          <figcaption>ground truth</figcaption></figure>
        <figure><img src="/api/mask.png?run=${encodeURIComponent(run)}&sid=${sid}&mode=diff&bg=1&rel=${rel}&${v}${ep}" alt="">
          <figcaption>difference</figcaption></figure>
      </div>
      <div class="nbhead">
        <div><span class="k">predicted com</span><b>${dc(d.pred_com[0], d.pred_com[1])}</b></div>
        <div><span class="k">ground truth</span><b>${dc(d.gt_com[0], d.gt_com[1])}</b></div>
        <div><span class="k">offset</span><b>${fmt(off, 2)} cells</b></div>
        ${e ? METRICS.map((mm) => `<div><span class="k">${mm.short}</span><b>${
              e[mm.key] == null ? "–" : fmt(e[mm.key], 3)}</b></div>`).join("") : ""}
      </div>
      <p class="note">Ground-truth scenes ranked by distance from this run's predicted
        center of mass, over ${d.n_candidates} samples with an object (empty boxes
        excluded). One entry per physical position.</p>
    </div>
    <div class="mgrid">
      <div class="msec"><h3>Most similar to the prediction</h3><ul class="nblist">${list(d.most_similar, "near")}</ul></div>
      <div class="msec"><h3>Least similar</h3><ul class="nblist">${list(d.least_similar, "far")}</ul></div>
    </div>`;

  // Each neighbour opens its own ground-truth detail, if it is on screen.
  $("#m-body").querySelectorAll(".nb").forEach((li) => {
    li.onclick = () => {
      const rank = S.order.findIndex((s) => s.i === +li.dataset.i);
      if (rank >= 0) openModal(rank);
    };
  });
  S.modalRow = -1;                       // arrow-key stepping applies to GT modals only
  $("#m-prev").hidden = $("#m-next").hidden = true;
  $("#modal").hidden = false;
}

function stepModal(delta) {
  if (S.modalRow < 0) return;            // neighbour modal has no position in the table
  const next = S.modalRow + delta;
  if (next >= 0 && next < S.order.length) openModal(next);
}

/* ***** wiring ***** */

function bindUI() {
  $("#scroller").addEventListener("scroll", onScroll, { passive: true });

  /* Clicking a column header's epoch pins every column there. Delegated on the scroller
     because renderHeader rebuilds the header row on every refresh. */
  $("#scroller").addEventListener("click", (e) => {
    const b = e.target.closest("b.goep[data-ep]");
    if (!b) return;
    e.stopPropagation();          // must not also cycle that column's sort
    gotoEpoch(+b.dataset.ep);
  });

  window.addEventListener("resize", renderVisible);

  /* Collapsing the sidebar hands its 268px to the table, which matters when comparing
     several runs side by side. Purely a CSS width change: the virtualizer keys off
     viewport HEIGHT, so no rows need recomputing. */
  const setSidebar = (open) => {
    document.body.classList.toggle("nosidebar", !open);
    $("#sidebar-show").hidden = open;
    try { localStorage.setItem("viz.sidebar", open ? "1" : "0"); } catch (_) {}
  };
  $("#sidebar-hide").onclick = () => setSidebar(false);
  $("#sidebar-show").onclick = () => setSidebar(true);
  try { if (localStorage.getItem("viz.sidebar") === "0") setSidebar(false); } catch (_) {}
  try { if (localStorage.getItem("viz.gtcol") === "0") setGtCol(false); } catch (_) {}

  $("#mode-seg").querySelectorAll("button").forEach((b) => {
    b.onclick = () => {
      S.view.mode = b.dataset.mode;
      $("#mode-seg").querySelectorAll("button").forEach((x) => x.classList.toggle("on", x === b));
      document.body.classList.toggle("stacked", S.view.mode === "stacked");
      renderLegend();
      // Stacked changes the row height, so the spacer and row offsets must be rebuilt --
      // a repaint alone would leave rows overlapping.
      refresh();
    };
  });
  // On/off is deliberately separate from the level: "b" (and the checkbox) hides the
  // backdrop and brings it back at whatever opacity the slider was last left on, so the
  // two controls never have to be re-set against each other.
  const setBackground = (on) => {
    S.view.background = on;
    $("#bg-toggle").checked = on;          // keep the checkbox honest when "b" drives it
    try { localStorage.setItem("viz.bg", on ? "1" : "0"); } catch (_) {}
    renderVisible();
  };
  $("#bg-toggle").onchange = (e) => setBackground(e.target.checked);

  /* The two opacity sliders. Both write a CSS variable, which is all the backdrop needs
     -- its veil is pure CSS, so dragging costs no repaint. The cube is drawn INTO the
     canvases, so that one also has to repaint; the ground-truth image follows the
     variable on its own. */
  const opacity = (id, key, cssVar, repaint) => {
    const el = $(id);
    const out = el.parentElement.querySelector(".opval");
    const set = (v, save) => {
      S.view[key] = v;
      document.documentElement.style.setProperty(cssVar, String(v));
      el.value = Math.round(v * 100);
      out.textContent = `${Math.round(v * 100)}%`;
      if (save) { try { localStorage.setItem(`viz.${key}`, String(v)); } catch (_) {} }
      if (repaint && save) renderVisible();   // nothing is painted yet at boot
    };
    el.oninput = (e) => set(+e.target.value / 100, true);
    let stored = null;
    try { stored = localStorage.getItem(`viz.${key}`); } catch (_) {}
    set(stored == null ? S.view[key] : Math.min(1, Math.max(0, +stored || 0)), false);
  };
  opacity("#bg-opacity", "bgOp", "--bg-op", false);
  opacity("#mask-opacity", "maskOp", "--mask-op", true);
  try { if (localStorage.getItem("viz.bg") === "0") setBackground(false); } catch (_) {}
  $("#rel-toggle").onchange = (e) => { S.view.relative = e.target.checked; renderVisible(); };

  $("#epoch-slider").oninput = (e) => setEpochIdx(+e.target.value);
  $("#epoch-play").onclick = togglePlay;
  holdToRepeat($("#epoch-prev"), () => stepEpoch(-1), $("#epoch-slider"));
  holdToRepeat($("#epoch-next"), () => stepEpoch(1), $("#epoch-slider"));
  $("#epoch-latest").onclick = () => {
    S.playing = false; clearInterval(playTimer);
    S.frameData = {};                 // drop frames so the latest epoch refetches cleanly
    setEpochIdx(null);
  };

  document.querySelectorAll("[data-all]").forEach((b) => {
    b.onclick = () => {
      const k = b.dataset.all;
      if (k === "speakers") S.filters.speakers = new Set([1, 2, 3, 4, 5, 6, 7, 8]);
      if (k === "layouts") S.filters.layouts = new Set(uniq((s) => s.layout).map((x) => x[0]));
      if (k === "nObjects") S.filters.nObjects = new Set(uniq((s) => s.n_objects).map((x) => x[0]));
      if (k === "boxes") S.filters.boxes = new Set(uniq((s) => s.box).map((x) => x[0]));
      if (k === "objects") {
        const all = new Set();
        S.samples.forEach((s) => (s.objects && s.objects.length ? s.objects : ["(none)"]).forEach((o) => all.add(o)));
        S.filters.objects = all;
      }
      if (k === "splits") { const a = new Set(); S.runOrder.forEach((n) => S.runs[n].splits.forEach((x) => a.add(x))); S.filters.splits = a; }
      refresh();
    };
  });
  document.querySelectorAll("[data-none]").forEach((b) => {
    b.onclick = () => { S.filters[b.dataset.none] = new Set(); refresh(); };
  });

  const q = $("#run-search");
  q.oninput = openPicker;
  q.onfocus = openPicker;
  q.onkeydown = (e) => {
    if (e.key === "Escape") { closePicker(); q.blur(); return; }
    if (!pickerOpen()) { if (e.key === "ArrowDown") openPicker(); return; }
    const rows = pickable();
    if (e.key === "ArrowDown" || e.key === "ArrowUp") {
      e.preventDefault();
      if (!rows.length) return;
      const step = e.key === "ArrowDown" ? 1 : -1;
      // From "nothing highlighted", Down lands on the first row and Up on the last.
      pickCur = pickCur < 0 ? (step > 0 ? 0 : rows.length - 1)
                            : (pickCur + step + rows.length) % rows.length;
      markCur();
    } else if (e.key === "Enter") {
      e.preventDefault();
      // Enter with nothing highlighted takes the only candidate, if there is exactly one.
      const d = pickCur >= 0 ? rows[pickCur] : (rows.length === 1 ? rows[0] : null);
      if (d) d.click();
    }
  };
  // A click anywhere else dismisses the list, the way a native dropdown behaves.
  document.addEventListener("mousedown", (e) => {
    if (pickerOpen() && !e.target.closest("#runbox")) closePicker();
  });
  document.querySelectorAll("[data-close]").forEach((b) => {
    b.onclick = () => { b.closest(".overlay").hidden = true; };
  });
  document.querySelectorAll(".overlay").forEach((o) => {
    o.onclick = (e) => { if (e.target === o) o.hidden = true; };
  });
  $("#m-prev").onclick = () => stepModal(-1);
  $("#m-next").onclick = () => stepModal(1);

  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape") { document.querySelectorAll(".overlay").forEach((o) => (o.hidden = true)); closePicker(); }
    if (!$("#modal").hidden) {
      if (e.key === "ArrowLeft") { e.preventDefault(); stepModal(-1); }
      if (e.key === "ArrowRight") { e.preventDefault(); stepModal(1); }
      return;
    }
    // Bare "f" toggles the filter panel -- but not while typing in the id searches, and
    // not when it is a browser/OS chord.
    // Only fields that swallow the character count as typing. A focused range, checkbox or
    // button does not: arrow-stepping the epoch slider leaves focus on it, and a tagName-only
    // test would kill these shortcuts for the rest of the session.
    const t = e.target;
    const typing =
      t.isContentEditable ||
      /^(TEXTAREA|SELECT)$/.test(t.tagName) ||
      (t.tagName === "INPUT" && !/^(range|checkbox|radio|button|submit|reset|color)$/.test(t.type));
    if (e.key === "f" && !typing && !e.ctrlKey && !e.metaKey && !e.altKey) {
      e.preventDefault();
      setSidebar(document.body.classList.contains("nosidebar"));
    }
    // Bare "b" toggles the backdrop, on the same terms as "f".
    if (e.key === "b" && !typing && !e.ctrlKey && !e.metaKey && !e.altKey) {
      e.preventDefault();
      setBackground(!S.view.background);
    }
    // Bare "g" collapses/restores the ground-truth column, same terms again.
    if (e.key === "g" && !typing && !e.ctrlKey && !e.metaKey && !e.altKey) {
      e.preventDefault();
      setGtCol(document.body.classList.contains("nogt"));
    }
  });

  bindTooltip();
  renderLegend();
  setInterval(poll, POLL_MS);   // pick up new runs and new epochs of training ones
}

/* Blue is always the prediction and green always the ground truth, in every view, so the
   overlay and difference explain each other rather than needing separate keys.
   Colorbars are served `immutable`, so they carry the render version too -- without it a
   palette change never reaches a browser that cached the old ramp. */
const colorbarURL = (mode) => `/api/colorbar/${mode}.png?v=${S.renderVersion}`;

function renderLegend() {
  const m = S.view.mode;
  if (m === "overlay" || m === "stacked") {
    // Two ramps rather than flat swatches: both layers carry magnitude, so the key has
    // to show the light-to-dark range each one spans.
    $("#legend").innerHTML =
      `<div class="dualbar">
         <div><img src="${colorbarURL("pred")}" alt=""><span>prediction</span></div>
         <div><img src="${colorbarURL("truth")}" alt=""><span>ground truth</span></div>
       </div>`;
    return;
  }
  const diff = m === "diff";
  $("#legend").innerHTML =
    `<img src="${colorbarURL(diff ? "diff" : "pred")}" alt="">
     <div class="lbl">${diff
       ? `<span>missed truth</span><span>0</span><span>excess pred</span>`
       : `<span>0</span><span>mask value</span><span>1</span>`}</div>`;
}

boot().catch((e) => { $("#subtitle").textContent = "failed to load"; console.error(e); });
