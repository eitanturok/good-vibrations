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
let ROW_H_BASE = 239, MASK_BOX = 172;
function readRowMetrics() {
  ROW_H_BASE = cssPx("--row-h", ROW_H_BASE);
  MASK_BOX = cssPx("--mask-h", 168) + 4;   // mask box plus the gap above it
}
// Cells grow with the finest loaded grid so a 256x256 prediction is shown at >= 1 screen
// px per cell instead of being squeezed into the 160px default (never shrunk below it).
const MASK_BASE = { w: 160, h: 168, row: 239 };
function fitMaskBox() {
  const dims = S.runOrder.map((n) => Math.max(...runShape(n)));
  const f = Math.max(1, ...dims.map((d) => d / MASK_BASE.w));
  const root = document.documentElement.style;
  root.setProperty("--mask-w", `${Math.round(MASK_BASE.w * f)}px`);
  root.setProperty("--mask-h", `${Math.round(MASK_BASE.h * f)}px`);
  root.setProperty("--row-h", `${Math.round(MASK_BASE.row + MASK_BASE.h * (f - 1))}px`);
  readRowMetrics();
}
const rowH = () => (S.view.mode === "stacked" ? ROW_H_BASE + MASK_BOX : ROW_H_BASE);
// --mask-w/--mask-h (see style.css) are the MAX a mask cell is ever allowed to take, not
// the size it renders at: stretching every box to fill that box regardless of its own
// real aspect distorts the ones that don't match it. Instead each row scales its own
// sample up (or down) by the LARGER of the two limits -- the classic "contain" fit -- so
// the image touches one edge of the box and is never squeezed off its real shape. Returns
// [w, h] in px, always <= the max on both axes.
function containedSize(asp) {
  const maxW = cssPx("--mask-w", 160), maxH = cssPx("--mask-h", 168);
  if (!(asp > 0)) return [maxW, maxH];
  const wAtMaxH = maxH * asp;
  return wAtMaxH <= maxW ? [wAtMaxH, maxH] : [maxW, maxW / asp];
}
// rel = fraction of the box (grid-independent, so 21x30 and 32x32 runs compare directly),
// raw = grid cells (kept so older runs stay readable).
// Per-sample metrics saved by training (src6 callbacks.sample_metrics). Runs from before
// 2026-10-08 are scored by viz and only have iou; their other cells read "–".
const METRICS = [
  { key: "iou",      label: "IoU",     short: "iou",  worst: "low"  },
  { key: "fill",     label: "Fill",    short: "fill", worst: "low"  },  // mean prob inside the object
  { key: "miss",     label: "Miss",    short: "miss", worst: "high" },  // 1 = no pixel > 0.5 on the object
  { key: "com_err",  label: "COM err", short: "com",  worst: "high" },  // com head, fraction of the box
  // --gray runs (src8): the drawn photo vs the real one, L1 at the TRUE object's location:
  // ph-loc = the object + a 2px ring (look and placement), ph-obj = the object pixels only
  { key: "photo_loc", label: "Photo L1 (object + ring)", short: "ph-loc", worst: "high" },
  { key: "photo_obj", label: "Photo L1 (object)",        short: "ph-obj", worst: "high" },
];

const S = {
  samples: [], runs: {}, order: [], runOrder: [], meta: null,
  filters: {
    splits: new Set(), speakers: new Set(), layouts: new Set(), nObjects: new Set(),
    objects: null,               // object TYPES present, independent of layout; null = no filter, a Set = scenes with ANY of them
    boxes: new Set(),            // enclosure the scene sits in
    datasets: new Set(),         // true data-identity field, distinct from box
    positions: null, ranges: {},
    // Explicit lookups. Empty means "no restriction"; any entry narrows the table to
    // exactly those samples/positions, unioned so both searches can be used at once.
    findSamples: new Set(), findPositions: new Set(),
  },
  // One sort per row group (sortGroupOf): {group key: {run, metric, dir}}. Columns in
  // different groups have their own row lists, so sorting one must not reorder another.
  sorts: {},
  // relative=false is the default on purpose: a fixed [0,1] domain is the only scale
  // under which two cells mean the same thing, which is the point of a comparison table.
  // background is an on/off switch (the "b" key, the checkbox); bgOp is the level it
  // returns to when switched back on, so the two are stored separately. maskOp is the
  // Objects slider: it fades every object mask, predicted and ground truth, applied as a
  // per-layer canvas alpha rather than to the cell, whose backdrop must not move with it.
  // rowMode "linked" (default): like "dataset", but runs whose dataset sets OVERLAP share one
  // group (sortGroupOf), so a combined gastro+plastic+cardboard column and a cardboard-only
  // column show the same sample on every row and one sort orders both.
  // rowMode "dataset": each loaded run's column is grouped by which dataset(s)
  // it was trained on and independently resolves its own row content -- see computeGroups
  // and S.groups, below. "single": today's original behaviour, one shared sample per row
  // across every column, via the flat S.order.
  view: { mode: "pred", rowMode: "linked", background: true, relative: false, coms: false, bgOp: 1, maskOp: 1, photo: false },
  groups: null,   // computed in applyFilters() unless rowMode is "single" -- see computeGroups
  domain: {}, positions: [], activePos: -1, modalRow: -1,
  hidden: new Map(),  // filter key -> samples that ONLY that filter is holding back
  focus: null,        // run shown full screen as a grid (setFocus), or null for the table
  renderVersion: 0,   // from /api/runs; part of image URLs to defeat immutable caching
  lut: null,          // colormaps from the server, so client and server agree on colour
  epochIdx: null,     // null = each run's latest; otherwise an index into the epoch list
  playing: false,
  modalComs: null,    // {coms} for the open modal, so "m" can repaint it
  epochMetrics: {},   // run -> {epoch: {saved, samples: {sid: {metric: value}}} | null while in flight}
  frameData: {},      // run -> {epoch | "latest": Map sid -> {m: fp16 mask}}
  truthCache: {},     // sid -> ground-truth values, for client-side diff
  photoCache: {},     // truthKey -> the real grayscale photo at that grid (Float32Array)
};

const fmt = (v, n = 4) => (v == null || !isFinite(v) ? "–" : v.toFixed(n));

/* Parameter count for the column header: 21_268_372 -> "21.3M". One decimal below 100M,
   none above, so the chip stays the same width whatever the model size. */
function fmtParams(n) {
  if (n == null || !isFinite(n)) return null;
  if (n >= 1e9) return (n / 1e9).toFixed(n >= 1e11 ? 0 : 2) + "B";
  if (n >= 1e6) return (n / 1e6).toFixed(n >= 1e8 ? 0 : 1) + "M";
  if (n >= 1e3) return (n / 1e3).toFixed(0) + "K";
  return String(n);
}

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
  return `${s.sample_id}${s.dataset ? ` (${s.dataset})` : ""}`;
}

/* s.sample_id is the universal id, "{position:06d}-{speaker}" (utils/ids.py); s.i is the opaque
   wire id. Searches may skip the padding: "12-3" -> "000012-3". */
const padId = (t) => t.replace(/^(\d+)-/, (_, p) => `${p.padStart(6, "0")}-`);
const byId = (a, b) => (a.i < b.i ? -1 : a.i > b.i ? 1 : 0);
/* The table's base order: position id (output_id, shared by a position's 8 speaker
   samples), then id. Samples without one go last. */
const posKey = (s) => (s.output_id == null ? Infinity : +s.output_id);
const byPosition = (a, b) => posKey(a) - posKey(b) || byId(a, b);

function shortLayout(s) {
  if (!s) return "–";
  if (s === "empty-box") return "empty";
  return s.split("-").filter((w) => w !== "cube" && w !== "cubes").join("+") || s;
}

/* The true pixel shape (height x width, matching the h x w convention every grid size
   elsewhere in this file uses) of the frame this sample's box tiles -- stamped in the
   corner of every mask cell because they all now render at one fixed size regardless of
   this, so a stretched cell never gets mistaken for the box's real proportions. */
function shapeLabel(s) {
  return s && s.dims ? `${s.dims[1]}×${s.dims[0]}` : "";
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
  S.meta = runs;
  S.renderVersion = runs.render_version ?? "0";
  S.samples = samples.samples;
  // --mask-w/--mask-h are a fixed MAX cell footprint (literals in style.css); each row
  // scales its own box up to fit inside it without distortion -- see containedSize() and
  // paintRow, which sets --row-mask-w/--row-mask-h per row. Nothing to compute at boot
  // now that the cell footprint no longer tracks any sample's own aspect.
  buildPositions();
  initFilters();
  bindFind();
  buildSpeakerDiagram();
  buildScatter();
  buildSliders();
  bindUI();
  let keep = null;
  try { keep = JSON.parse(sessionStorage.getItem("viz.reloadRuns") || "null"); sessionStorage.removeItem("viz.reloadRuns"); } catch (_) {}
  const names = new Set(runs.runs.filter((r) => r.compatible).map((r) => r.name));
  for (const name of (keep || runs.default_selected).filter((n) => names.has(n))) await addRun(name);
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
   the run's identity for filtering. `samples` (metrics) is always the latest epoch too --
   the epoch scrubber only swaps masks. */
function indexMembership(d) {
  d.splits = new Set(Object.values(d.samples).map((x) => x.split));
  d.splitOf = Object.fromEntries(Object.entries(d.samples).map(([k, v]) => [k, v.split]));
}

// Names with a fetch in flight -- checked and set BEFORE the first await, so a second
// click on the same run while its /api/run request is still pending (the picker no longer
// closes after one pick, so nothing else stops a fast double-click or repeated Enter)
// can't slip past the `S.runs[name]` guard below, which isn't set until the fetch resolves.
const addInFlight = new Set();
async function addRun(name, reload = false) {
  if (S.runs[name] && !reload) return;
  if (!reload) {
    if (addInFlight.has(name)) return;
    addInFlight.add(name);
  }
  try {
    await addRunNow(name, reload);
  } finally {
    addInFlight.delete(name);
  }
}

async function addRunNow(name, reload) {
  const d = await api(`/api/run/${encodeURIComponent(name)}${reload ? "?reload=1" : ""}`);
  d.entry = S.meta.runs.find((r) => r.name === name);
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
  delete S.frameData[name];
  delete S.epochMetrics[name];
  S.runOrder = S.runOrder.filter((n) => n !== name);
  invalidateEpochs();
  for (const k in S.sorts) if (S.sorts[k].run === name) delete S.sorts[k];
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
    if (s.dataset) f.datasets.add(s.dataset);
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
function splitsOf(sampleIdx, runs = S.runOrder) {
  const out = new Set();
  for (const n of runs) {
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
    const hit = f.findSamples.has(s.sample_id) || f.findPositions.has(+s.output_id);
    return hit ? out : ["find"];
  }
  if (!f.speakers.has(s.speaker)) out.push("speakers");
  if (!f.layouts.has(s.layout)) out.push("layouts");
  if (!f.nObjects.has(s.n_objects)) out.push("nObjects");
  if (!f.boxes.has(s.box)) out.push("boxes");
  if (!f.datasets.has(s.dataset)) out.push("datasets");
  // Picked objects, like viz2: null = no filter ("all"); else a scene passes if it
  // contains ANY picked object, so an empty pick ("none") hides everything.
  // Empty boxes match the "(none)" pseudo-object.
  const objs = s.objects && s.objects.length ? s.objects : ["(none)"];
  if (f.objects && !objs.some((o) => f.objects.has(o))) out.push("objects");
  if (f.positions && !(s.pos >= 0 && f.positions.has(s.pos))) out.push("positions");

  // With no runs loaded the table is a ground-truth browser, so every sample qualifies;
  // the split filter only applies once some run has assigned splits.
  //
  // "By dataset" mode: whether a run is relevant to THIS check is scoped to runs that
  // actually train on this sample's own dataset. Checking against every loaded run
  // (S.runOrder) unscoped -- as "single sample" mode correctly does, since there every
  // run shares one row space -- meant loading a run for a totally different box (with no
  // overlapping datasets at all) marked EVERY sample of every other loaded box's dataset
  // "nopred", because that new run obviously never predicted them either. A cardboard
  // sample's coverage should only ever be judged against cardboard-trained runs.
  const relevantRuns = grouped()
    ? S.runOrder.filter((n) => runDs(n).includes(s.dataset))
    : S.runOrder;
  if (S.runOrder.length) {
    const sps = splitsOf(s.i, relevantRuns);
    // No RELEVANT loaded run predicted this sample. Reported separately from the split
    // chips because no chip can bring it back -- the fix is loading a run that covers it.
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

  // Metric ranges read the sorted runs that cover this sample, else any RELEVANT loaded
  // run may satisfy them (union) — the intuitive reading of "show me samples where MSE
  // is high", scoped the same way as the split check just above and for the same reason.
  // A sorted run only judges samples it covers: a plastic sample is not in a gastro
  // run, so sorting the gastro column must leave the plastic column as it was.
  const judges = activeSorts().map((st) => st.run).filter((n) => (S.runs[n].splitOf || {})[s.i]);
  const names = judges.length ? judges : relevantRuns;
  if (!names.length) return out;
  let any = false;
  for (const n of names) {
    const r = S.runs[n];
    const e = r.samples[s.i];
    if (!e) continue;
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
  objects:   { label: "object",   clear: (f) => { f.objects = null; } },
  layouts:   { label: "layout",   clear: (f) => S.samples.forEach((s) => s.layout && f.layouts.add(s.layout)) },
  boxes:     { label: "box",      clear: (f) => S.samples.forEach((s) => s.box && f.boxes.add(s.box)) },
  datasets:  { label: "dataset",  clear: (f) => S.samples.forEach((s) => s.dataset && f.datasets.add(s.dataset)) },
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

/* The chip value(s) a sample sits under, per faceted filter (objects and splits are sets). */
const FACET_VALUES = {
  layouts:  (s) => [s.layout],
  nObjects: (s) => [s.n_objects],
  boxes:    (s) => [s.box],
  datasets: (s) => [s.dataset],
  objects:  (s) => (s.objects && s.objects.length ? s.objects : ["(none)"]),
  splits:   (s) => splitsOf(s.i),
};

function applyFilters() {
  // One pass, two answers: the rows to draw, and -- for everything held back -- which
  // filter would have to be relaxed to admit it. A sample failing several filters at
  // once is not attributed to any of them: clearing one would not bring it back, so
  // promising that it would is worse than staying quiet.
  const rows = [], solo = new Map();
  // Chip counts are faceted: each value counts the samples that pass every OTHER filter,
  // so picking object=cube leaves only the layouts holding a cube, and n_objects=1 turns
  // the object counts into "how many 1-object samples have this object". A sample counts
  // toward a filter's chips if nothing hides it, or if that filter is the only one that does.
  const facets = {};
  const tallyFacet = (s, k) => {
    const m = facets[k] || (facets[k] = new Map());
    for (const v of FACET_VALUES[k](s)) m.set(v, (m.get(v) || 0) + 1);
  };
  for (const s of S.samples) {
    const why = failures(s);
    if (!why.length) {
      rows.push(s);
      for (const k in FACET_VALUES) tallyFacet(s, k);
    } else if (why.length === 1) {
      solo.set(why[0], (solo.get(why[0]) || 0) + 1);
      if (FACET_VALUES[why[0]]) tallyFacet(s, why[0]);
    }
  }
  S.hidden = solo;
  S.facets = facets;
  // Rows are always in position order; a metric sort reorders on top of it (stably, so
  // ties stay in position order). "Single sample" mode only exists when every run shares
  // one dataset set, so it has exactly one group and at most one sort; "by dataset" sorts
  // each group's own rows.
  rows.sort(byPosition);
  S.order = grouped() ? rows : sortRows(rows, activeSorts()[0]);
  S.groups = grouped() ? computeGroups(rows) : null;
}

/* Whether columns resolve rows per group (computeGroups) rather than one flat S.order. */
const grouped = () => S.view.rowMode !== "single";

/* The row group a run's column belongs to, and so its one sort. "dataset": runs predicting
   the same SET of datasets. "linked": runs connected through any shared dataset, so the
   group's rows are the union of what its runs predicted and a run that didn't predict a
   row's sample shows "not in run" there. Keyed by the group's dataset union. */
function sortGroupOf(name) {
  if (S.view.rowMode !== "linked") return runDs(name).join("|");
  let ds = new Set(runDs(name)), grew = true;
  while (grew) {
    grew = false;
    for (const n of S.runOrder) {
      const d = runDs(n);
      if (d.some((x) => ds.has(x)) && d.some((x) => !ds.has(x))) { d.forEach((x) => ds.add(x)); grew = true; }
    }
  }
  return [...ds].sort().join("|");
}

/* The sort applying to `name`'s column, or null. A sort whose run was removed, or whose
   run no longer lands in this group (its dataset set changed on reload), is ignored. */
function sortOf(name) {
  const st = S.sorts[sortGroupOf(name)];
  return st && S.runs[st.run] && sortGroupOf(st.run) === sortGroupOf(name) ? st : null;
}

const activeSorts = () => S.runOrder.map((n) => sortOf(n)).filter((st, k, a) => st && a.indexOf(st) === k);

/* `rows` (already in position order) ordered by one run's metric, worst or best first;
   unchanged when there is no sort. */
function sortRows(rows, sort) {
  if (!sort) return rows;
  const { run, metric, dir } = sort;
  const m = METRICS.find((x) => x.key === metric);
  // "worst" means high for localization, LOW for IoU/contour; the direction button is
  // labelled semantically so this inversion never lands on the user.
  const isAbs = m.worst === "abs";
  const desc = isAbs ? dir === "worst" : (m.worst === "high") === (dir === "worst");
  const key = (v) => (isAbs ? Math.abs(v) : v);
  const tbl = S.runs[run].samples;
  return [...rows].sort((a, b) => {
    const x = tbl[a.i], y = tbl[b.i];
    // A missing prediction, or a metric undefined for this sample (localization on an
    // empty box), sinks to the end in BOTH directions.
    const xv = x && x[metric] != null ? x[metric] : null;
    const yv = y && y[metric] != null ? y[metric] : null;
    if (xv == null && yv == null) return 0;
    if (xv == null) return 1;
    if (yv == null) return -1;
    return (key(yv) - key(xv)) * (desc ? 1 : -1);
  });
}

/* metric "position" puts the group back in its base order. */
function setSort(name, metric, dir) {
  if (metric === "position") delete S.sorts[sortGroupOf(name)];
  else S.sorts[sortGroupOf(name)] = { run: name, metric, dir };
  refresh(true);
}

/* The datasets a run predicted: what /api/run read off its loaded rows, else the scan's
   guess (which probes only a few files, so it can miss a combined run's second dataset). */
const runDs = (n) => S.runs[n].datasets || (S.runs[n].entry && S.runs[n].entry.datasets) || [];

/* "By dataset" row mode: runs predicting the same SET of datasets share a group, and a
   group's rows are the filtered `rows` (position order) that one of its runs predicted,
   sorted by the group's OWN sort (sortOf), so sorting one group never moves another. */
function computeGroups(rows) {
  if (!S.runOrder.length) return null;
  const groups = new Map();
  for (const name of S.runOrder) {
    const key = sortGroupOf(name);
    if (!key) continue;   // no datasets known yet
    if (!groups.has(key)) groups.set(key, { runs: [], order: [] });
    groups.get(key).runs.push(name);
  }
  for (const g of groups.values())
    g.order = sortRows(rows.filter((s) => g.runs.some((n) => S.runs[n].splitOf[s.i])), sortOf(g.runs[0]));
  return [...groups.values()];
}

/* Total row count the virtualizer must cover: the flat filtered list in "single sample"
   mode, or the longest group in "by dataset" mode -- shorter groups just paint blank past
   their own end (see paintRow). */
function rowCount() {
  if (grouped() && S.groups && S.groups.length)
    return S.groups.reduce((n, g) => Math.max(n, g.order.length), 0);
  return S.order.length;
}

/* ***** table ***** */

/* Per-sample metrics {sid: {metric: value}} of `run` at the epoch on screen: the latest
   epoch's come with the run, a scrubbed one's from epochMetrics. null while that epoch is in
   flight, false when the run never saved per-epoch metrics. */
function metricsAt(run) {
  const r = S.runs[run], ep = epochFor(run), eps = r.epochs || [];
  if (ep == null || ep === eps[eps.length - 1]) return r.samples;
  const em = epochMetrics(run, ep);
  return em && (em.saved ? em.samples : false);
}

/* Aggregates over the filtered rows at the epoch on screen; all null ("–") while it loads. */
function statsFor(name) {
  const tbl = metricsAt(name) || {}, out = {};
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
function setGtCol(open, persist = true) {
  // There is no single shared ground truth to show outside "single sample" mode -- see
  // paintRow -- so opening it is a no-op there rather than fighting setRowMode over the
  // nogt class every time either one runs.
  if (open && S.view.rowMode !== "single") return;
  document.body.classList.toggle("nogt", !open);
  // The row-mode switch collapses it without persisting, so "single" can restore the
  // user's own last choice rather than the forced collapse.
  if (persist) { try { localStorage.setItem("viz.gtcol", open ? "1" : "0"); } catch (_) {} }
}

/* "Linked" (default) / "by dataset" vs "single sample" -- see S.view.rowMode's doc comment and
   paintRow. Switching away from "single" force-collapses the ground-truth column;
   switching back restores whatever the user last had it set to. */
function setRowMode(mode) {
  S.view.rowPref = mode;
  try { localStorage.setItem("viz.rowmode", mode); } catch (_) {}
  syncRowMode();
  refresh(true);
}

/* The loaded runs' dataset sets (the same key computeGroups groups columns by). More
   than one means the columns don't share rows, so "single sample" mode -- one shared
   sample per row -- would show "not in run" down most of every column. */
function runDatasets() {
  return new Set(S.runOrder.map((n) => runDs(n).join("|")).filter(Boolean));
}

/* Applies the effective row mode: the user's pick (S.view.rowPref), except that runs
   trained on different datasets force "by dataset" -- keyed on the dataset (experiment
   dir), not the box, since several captures share a box. Called on every refresh so
   adding/removing a run flips it; returns whether the mode changed. */
function syncRowMode() {
  const mixed = runDatasets().size > 1;
  const pref = S.view.rowPref || "linked";
  const mode = mixed && pref === "single" ? "linked" : pref;
  const single = $("#rowmode-seg").querySelector('[data-rowmode="single"]');
  single.dataset.title ??= single.title;   // the static tooltip from index.html
  single.disabled = mixed;
  single.title = mixed
    ? "Unavailable: the loaded runs were trained on different datasets, so no sample is shared across every column."
    : single.dataset.title;
  $("#rowmode-seg").querySelectorAll("button").forEach((b) => b.classList.toggle("on", b.dataset.rowmode === mode));
  if (mode === S.view.rowMode && S.view.rowModeApplied) return false;
  S.view.rowMode = mode;
  S.view.rowModeApplied = true;
  if (mode === "single") {
    let stored = "1";
    try { stored = localStorage.getItem("viz.gtcol") ?? "1"; } catch (_) {}
    setGtCol(stored !== "0", false);
  } else {
    setGtCol(false, false);
  }
  return true;
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

  for (const name of S.runOrder) h.appendChild(runHeaderCell(name));
  return h;
}

/* navigator.clipboard needs a secure context (https/localhost); viz is often opened over
   plain http, so fall back to the old execCommand route there. */
async function copyText(text) {
  try { await navigator.clipboard.writeText(text); return true; } catch (_) {}
  const ta = Object.assign(document.createElement("textarea"), { value: text });
  ta.style.cssText = "position:fixed;opacity:0";
  document.body.appendChild(ta);
  ta.select();
  const ok = document.execCommand("copy");
  ta.remove();
  return ok;
}

/* A tiny pill just above `anchor` (below it if there is no room), then fades. One element
   is reused, so repeat clicks restart it rather than stacking. */
const toastEl = Object.assign(document.createElement("div"), { id: "toast" });
let toastTimer;
function toast(msg, anchor) {
  if (!toastEl.isConnected) document.body.appendChild(toastEl);
  toastEl.textContent = msg;
  toastEl.classList.remove("show");
  const r = anchor.getBoundingClientRect(), h = toastEl.offsetHeight;  // also restarts the fade
  toastEl.style.left = `${r.left + r.width / 2}px`;
  toastEl.style.top = `${r.top > h + 8 ? r.top - h - 6 : r.bottom + 6}px`;
  toastEl.classList.add("show");
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => toastEl.classList.remove("show"), 1100);
}

/* One run's header cell. Shared by the table header and the focus view's header, so the
   two can never disagree about a run's stats or sort state. */
function statsHTML(st) {
  return METRICS.map((m) => {
    const s = st[m.key];
    return `<span><span class="k">${m.short}</span> <b>${s ? fmt(s.mean, 3) : "–"}</b>${
      s ? `<span class="k">±${fmt(s.sd, 3)}</span>` : ""}</span>`;
  }).join("");
}

/* Repaints only the stats line of every header (table and focus view), so an epoch step
   does not rebuild the header under a dragging pointer. */
function renderHeaderStats() {
  document.querySelectorAll(".hstats").forEach((el) => {
    if (S.runs[el.dataset.run]) el.innerHTML = statsHTML(statsFor(el.dataset.run));
  });
}

function runHeaderCell(name) {
  const r = S.runs[name], st = statsFor(name);
  const so = sortOf(name), sorted = !!so && so.run === name;
  const cell = document.createElement("div");
  // Dimmed only when ANOTHER column in this run's own group sets its row order.
  cell.className = "hcell run" + (sorted ? " sorted" : so ? " dim" : "");
  const warn = r.entry && r.entry.family === "unknown"
    ? `<span class="warnbadge" title="No eval split directories; dataset identity unconfirmed">?</span>` : "";
  const skipped = r.skipped_files.length
    ? ` · <span title="${r.skipped_files.join(", ")}">${r.skipped_files.length} file(s) skipped</span>` : "";
  const moved = r.missing && r.missing.length
    ? ` · <span title="not in samples/ (moved to bad_samples/ or deleted): ${r.missing.join(", ")}">missing ${r.missing[0]} directory${
      r.missing.length > 1 ? ` (+${r.missing.length - 1} more)` : ""}</span>` : "";
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
      <div class="hname">${name}${warn}</div>
      <button class="hcopy" title="Copy the run name" aria-label="Copy the run name">⧉</button>
      <button class="hclose" title="Remove this run">&times;</button>
    </div>
    <div class="hchips">
      <span class="hep" data-run="${name}">${epochLabel(name)}</span>
      ${statusChip(status)}
      ${fmtParams(r.n_params) ? `<span class="pchip" title="${
        r.n_params.toLocaleString()} trainable parameters">${fmtParams(r.n_params)}<span class="k">&nbsp;params</span></span>` : ""}
    </div>
    ${why}
    ${skipped + moved ? `<div class="hmeta">${(skipped + moved).replace(/^ · /, "")}</div>` : ""}
    <div class="hstats" data-run="${name}">${statsHTML(st)}</div>
    <div class="hsort">
      <select><option value="position" ${so ? "" : "selected"}>Position</option>${METRICS.map((m) =>
        `<option value="${m.key}" ${sorted && so.metric === m.key ? "selected" : ""}>${m.label}</option>`).join("")}</select>
      <button class="dir ${sorted ? "on" : ""}">${sorted ? (so.dir === "worst" ? "↓ worst" : "↑ best") : "sort"}</button>
      <span class="hgrip" title="Drag to reorder this column">⠿</span>
    </div>`;

  const focused = S.focus === name;
  cell.title = focused ? "Click to go back to the table" : "Click to expand this run full screen";
  // Reordering only makes sense in the table; the focus view has one column.
  if (!focused) makeDraggable(cell, name);
  else cell.querySelector(".hgrip").remove();

  cell.querySelector(".hclose").onclick = (e) => {
    e.stopPropagation();          // must not also toggle the focus view
    removeRun(name);
  };
  // A click anywhere on the header toggles the full-screen view of this run. Sorting is
  // ONLY the sort button and the metric dropdown, so the controls are excluded here.
  cell.onclick = (e) => {
    if (e.target.closest("button, select, .hgrip, b.goep, a, input")) return;
    // Selecting the name to copy it ends in a click here, which would toggle the view
    // as a side effect of highlighting text. A selection inside this cell was the intent.
    const sel = window.getSelection();
    if (sel && !sel.isCollapsed && sel.anchorNode && cell.contains(sel.anchorNode)) return;
    setFocus(focused ? null : name);
  };
  // Only the copy button copies the name; it is a button, so the focus toggle above skips it.
  const copy = cell.querySelector(".hcopy");
  copy.onclick = () => copyText(name).then((ok) => toast(ok ? "Copied" : "Couldn't copy", copy));
  cell.querySelector("select").onchange = (e) => setSort(name, e.target.value, sorted ? so.dir : "worst");
  // On a position-ordered column the button starts a metric sort, IoU unless one is picked.
  cell.querySelector(".dir").onclick = () => {
    const v = cell.querySelector("select").value;
    setSort(name, v === "position" ? METRICS[0].key : v, sorted && so.dir === "worst" ? "best" : "worst");
  };
  return cell;
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
      <div class="mask gtwrap"><img class="gtimg" loading="lazy" decoding="async" alt=""><canvas class="gtcoms"></canvas><span class="shapelabel"></span></div>
      <div class="tags gt-foot"></div>
    </div>`;
  el._runCells = [];
  return el;
}

function syncRowCells(el) {
  while (el._runCells.length > S.runOrder.length) el.removeChild(el._runCells.pop());
  while (el._runCells.length < S.runOrder.length) {
    const c = buildRunCell();
    el.appendChild(c);
    el._runCells.push(c);
  }
}

function buildRunCell() {
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
    `<div class="maskbox predbox"><canvas class="mask predmask" width="${mw}" height="${mh}"></canvas><span class="shapelabel"></span></div>` +
    `<div class="maskbox truthbox" hidden><canvas class="mask truthmask" width="${mw}" height="${mh}"></canvas><span class="shapelabel"></span></div>` +
    `<div class="tags subtags"></div>`;
  return c;
}

// The row list a run's column is painted against. In "by dataset" row mode a row is no
// longer one shared sample -- each run column resolves ITS OWN dataset group's Nth
// (filtered/sorted) sample, so two columns only show the same scene when their runs share
// a dataset. Entries are undefined past the end of a shorter group.
function orderFor(name) {
  const g = S.groups && S.groups.find((g) => g.runs.includes(name));
  return g ? g.order : S.order;
}

function paintRow(el, rank) {
  // "Single sample" mode is exactly today's behaviour: one shared sample across the whole
  // row, via the flat S.order. "By dataset" mode has no single shared sample -- each
  // column below resolves its own via orderFor -- so the dedicated ground-truth
  // column (which can only ever show ONE thing) is force-collapsed in that mode (see
  // setRowMode) and left unpainted here.
  const s = S.order[rank];
  el.style.transform = `translateY(${rank * rowH()}px)`;
  el.querySelector(".idxn").textContent = rank + 1;

  if (!grouped()) {
    // The true pixel shape stamped in every mask cell's corner (see .shapelabel) -- every
    // cell in the row needs it, ground truth included, so it is computed once up front
    // rather than separately per cell.
    const shape = shapeLabel(s);
    // Every .mask in this row -- ground truth and each run column -- draws at THIS
    // sample's own aspect, scaled up to fit inside --mask-w x --mask-h without distortion
    // (see containedSize). One row-level pair of variables rather than a per-element style
    // so the GT and the predictions beside it can never disagree about the frame they sit
    // in, and .mask can fall back to the fixed max for anything outside a row (e.g. the
    // "no prediction" placeholder).
    const [rmw, rmh] = containedSize(s.aspect);
    el.style.setProperty("--row-mask-w", `${Math.round(rmw)}px`);
    el.style.setProperty("--row-mask-h", `${Math.round(rmh)}px`);
    // Identity line: which sample, where it was captured, which speaker played. The 8
    // samples sharing a position differ only by speaker, so all three belong together.
    // The run columns no longer repeat this: the sample is the same across the whole row,
    // so saying it once in the sticky column leaves the run cells free to spend their
    // title on what actually differs between them (name, split, epoch).
    el.querySelector(".gt-head").innerHTML =
      `<span class="t1"><span class="txt">Ground truth</span></span>` +
      `<span class="t2"><span class="txt">${identity(s)}</span></span>`;

    // The ground-truth cell shows the same 20x40 target the runs predict, over the same
    // backdrop, so it is directly comparable with every prediction column. The speaker
    // view lives in the detail modal instead.
    // The wrapper carries the backdrop and the hover/tooltip identity; the <img> inside is
    // the bare mask, so `opacity` fades the cube without touching the photo behind it.
    // Target object marks over the server-rendered GT image. The image is a PNG, not a
    // canvas, so the marks need their own layer; it carries X's only, since there is no
    // prediction in this column to pair them with.
    {
      const gcv = el.querySelector(".gtcoms");
      const [gh, gw] = gtShape();
      sizeMaskCanvas(gcv, gw, gh);
      const g2 = gcv.getContext("2d");
      g2.clearRect(0, 0, gcv.width, gcv.height);
      if (S.view.coms && s.obj_com && s.obj_com.length)
        drawComs(g2, gcv.width, gcv.height, { missed: s.obj_com }, true);
    }
    const wrap = el.querySelector(".gtwrap");
    const img = wrap.querySelector(".gtimg");
    wrap.className = "mask gtwrap" + (S.view.background ? " bg" : " nobg");
    setBackdrop(wrap, s.i);
    const src = gtMaskURL(s.i);
    if (img.getAttribute("src") !== src) img.setAttribute("src", src);
    wrap.dataset.run = ""; wrap.dataset.sid = s.i;
    wrap.querySelector(".shapelabel").textContent = shape;

    // Split is a property of each run's dataloader, not of the sample, so it is shown per
    // run column rather than here.
    el.querySelector(".gt-foot").innerHTML =
      `<span class="tag" title="${s.layout}">${shortLayout(s.layout)}</span>` +
      `<span class="tag">${s.n_objects} obj</span>` +
      (s.box ? `<span class="tag" title="box: ${s.box}">${s.box}</span>` : "");
    // Collapsed, the cell is a bare rail -- clicking it must not open the sample modal.
    el.querySelector(".gtcell").onclick = () => {
      if (!document.body.classList.contains("nogt")) openModal(rank);
    };
  }

  // Every .mask in this row -- ground truth and each run column -- is drawn at the SAME
  // fixed --mask-w x --mask-h box regardless of this sample's own box shape, so cells
  // compare directly at a glance instead of varying width row to row. A box whose real
  // frame is a different shape gets stretched to fit; the shape label stamps the true
  // pixel size in the corner so that stretch is never mistaken for the real proportions.
  syncRowCells(el);
  // Hoisted: depends only on S.runOrder, so computing it per cell rebuilt a Set ~140
  // times per repaint (28 pooled rows x 5 columns) at scroll frame rate.
  const mixed = mixedShapes();
  S.runOrder.forEach((name, k) => {
    paintRunCell(el._runCells[k], name, orderFor(name)[rank], mixed);
  });
}

/* Paints one run's cell for one sample: a table cell, or a tile in the focus grid. */
function paintRunCell(c, name, cs, mixed) {
  const head = c.querySelector(".run-head");
  const predBox = c.querySelector(".predbox");
  const m = predBox.querySelector(".mask");
  const chips = c.querySelector(".subtags");
  const [rh, rw] = runShape(name);

  // Per-cell rather than per-row: columns in different dataset groups (and tiles in the
  // focus grid) show different boxes at once, each with its own real aspect to preserve.
  const [rmw, rmh] = containedSize(cs ? cs.aspect : 0);
  c.style.setProperty("--row-mask-w", `${Math.round(rmw)}px`);
  c.style.setProperty("--row-mask-h", `${Math.round(rmh)}px`);

  if (!cs) {
    // This rank is past the end of THIS column's own dataset group -- another group is
    // longer. Nothing to show here at all, not even "not in run": there is no sample, not just no prediction.
    chips.innerHTML = "";
    head.innerHTML = `<span class="t1 run"><span class="txt" title="${name}">${name}</span></span>`;
    predBox.hidden = true;
    m.style.backgroundImage = ""; m.dataset.bd = "";
    if (!c._np) { c._np = document.createElement("div"); c._np.className = "nopred"; c.appendChild(c._np); }
    c._np.className = "nopred nosample";
    c._np.textContent = "—";
    c._np.title = "no sample at this row for this run's dataset";
    c._np.hidden = false;
    return;
  }

  const shape = shapeLabel(cs);
  const e = S.runs[name].samples[cs.i];

  // Split and epoch are both PER-CELL facts, which is why neither can be left to the
  // column header: dataloaders assign splits independently, so one sample can be train
  // in one run and held out in another, and runs save on different cadences, so at one
  // slider position two columns can legitimately show different epochs.
  const cellEp = epochFor(name);
  const shownEp = cellEp == null ? (S.runs[name].epochs || []).slice(-1)[0] : cellEp;
  const latest = cellEp == null;
  const split = e ? e.split : (S.runs[name].splitOf || {})[cs.i];
  // The grid is shown only when columns disagree on it. Both metrics are
  // grid-normalized so the numbers share a scale, but a coarser grid is systematically
  // easier -- a small cross-size gap is not evidence of a better model, so the reader
  // needs to see which size they are looking at.
  const gs = mixed ? `<span class="tag gsz" title="mask grid">${rh}x${rw}</span>` : "";
  const epchip = `<span class="epchip${latest ? "" : " scrub"}" title="epoch ${shownEp ?? "?"}${
    latest ? " (latest saved)" : " (scrubbed)"}"><b>${shownEp ?? "–"}</b><span class="k">ep</span></span>`;
  // Epoch sits on the RUN's own line -- it is a fact about the run, not the sample --
  // right-aligned against the name so the two never run together. Split sits on the
  // SAMPLE's line instead, since it is a fact about how this run's dataloader filed
  // THIS sample, also right-aligned. Both describe the prediction, which is why neither
  // lived below the image before: they are closer in kind to a title than to a metric.
  // identity(cs) is also where a grouped cell's OWN dataset becomes visible -- see
  // identity()'s "(dataset)" suffix -- since neighbouring columns in
  // different groups can be showing entirely different datasets at this same rank.
  head.innerHTML =
    `<span class="t1 run"><span class="txt" title="${name}">${name}</span>${epchip}</span>` +
    `<span class="t2"><span class="txt">${identity(cs)}</span>${splitChip(split)}</span>`;

  if (!e) {
    chips.innerHTML = "";
    predBox.hidden = true;
    m.style.backgroundImage = ""; m.dataset.bd = "";   // keep the cache flag honest
    predBox.querySelector(".shapelabel").textContent = "";
    if (!c._np) { c._np = document.createElement("div"); c._np.className = "nopred"; c.appendChild(c._np); }
    // `samples` is the latest epoch's full coverage, so no entry means the sample is
    // outside this run's split. "loading" / "not saved" for a scrubbed epoch are per-mask
    // states, drawn over the canvas by paintCanvas.
    c._np.className = "nopred";
    c._np.textContent = "not in run";
    c._np.title = "this sample is not in this run's split";
    c._np.hidden = false;
    return;
  }
  if (c._np) c._np.hidden = true;
  // Split and epoch moved up to the title (see head.innerHTML above); this is just the
  // grid-size tag, when shown, followed by the metrics at the epoch on screen (metricsAt,
  // the same numbers the header averages). Runs that never saved per-epoch metrics are not
  // re-scored, so they say so rather than caption an early mask with final numbers. Rows,
  // sort and filters still follow the latest epoch.
  const tbl = metricsAt(name), vals = tbl && tbl[cs.i];
  chips.innerHTML = gs + (tbl === false
    ? `<span class="tag"><span class="k">metrics at latest epoch only</span></span>`
    : METRICS.map((mm) => {
      const v = vals ? vals[mm.key] : null;
      return `<span class="tag"><span class="k">${mm.short}</span> <b>${
        v == null || Number.isNaN(v) ? "–" : fmt(v, 3)}</b></span>`;
    }).join(""));
  predBox.hidden = false;
  predBox.querySelector(".shapelabel").textContent = shape;
  // Match the canvas buffer to THIS run's grid. Rows are recycled across runs, so a
  // pooled cell can arrive still sized for a column of a different resolution.
  sizeMaskCanvas(m, rw, rh);
  m.dataset.run = name; m.dataset.sid = cs.i;
  m.onclick = () => openNeighbors(name, cs.i);
  // Photo view: the drawn photo is opaque, so no backdrop behind it (a faded edge would
  // show the real photo through and read as a second object).
  m.className = "mask predmask" + (S.view.background && !isPhoto(name) ? " bg" : " nobg");
  setBackdrop(m, cs.i);
  paintCanvas(m, name, cs.i, epochFor(name));

  // Stacked: the target gets its own box directly below the prediction, so the two are
  // adjacent instead of the ground truth being columns away in the leftmost cell.
  const tmBox = c.querySelector(".truthbox");
  const tm = tmBox.querySelector(".mask");
  tmBox.hidden = S.view.mode !== "stacked";
  if (!tmBox.hidden) {
    tmBox.querySelector(".shapelabel").textContent = shape;
    paintTruthHalf(tm, name, cs.i, rh, rw);
  }
}

/* The stacked target half of a cell (table or sample popup): the target mask, or in photo
   view the real photo at the run's grid. */
function paintTruthHalf(tm, name, sid, rh, rw) {
  tm.className = "mask truthmask" + (S.view.background && !isPhoto(name) ? " bg" : " nobg");
  setBackdrop(tm, sid);
  sizeMaskCanvas(tm, rw, rh);
  if (isPhoto(name)) {
    // Photo view: the real photo at the run's grid under the drawn one.
    const ph = S.photoCache[truthKey(sid, rh, rw)];
    if (ph) drawPhoto(tm, ph, null, [rh, rw], comsFor(name, sid), true);
    else { tm.getContext("2d").clearRect(0, 0, tm.width, tm.height); ensurePhoto(sid, rh, rw); }
    return;
  }
  const truth = S.truthCache[truthKey(sid, rh, rw)];
  // The stacked target half shows the target's own objects and nothing else: the
  // prediction marks and the connecting lines belong on the prediction above it.
  if (truth) drawMask(tm, truth, "truth", null, [rh, rw], comsFor(name, sid), true);
  else ensureTruth(sid, rh, rw);
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
    last: Math.min(rowCount(), Math.ceil((top + sc.clientHeight) / rowH()) + pad),
  };
}

/* ***** focus view *****
   Clicking a run's header shows that run full screen as a grid of its own samples, in the
   table's filtered/sorted order; clicking the header again goes back. Tiles are the same
   cells the table paints (buildRunCell/paintRunCell), so every view setting carries over. */
function setFocus(name) {
  S.focus = name;
  $("#focus-scroller").scrollTop = 0;
  refresh();
}

function renderFocusHeader() {
  const name = S.focus, r = S.runs[name];
  S.focusList = orderFor(name).filter((s) => s && (r.samples[s.i] || (r.splitOf || {})[s.i]));
  $("#focus-head").replaceChildren(runHeaderCell(name));
  $("#focus-head").insertAdjacentHTML("beforeend", `<div class="finfo"><span><b>${S.focusList.length}</b> samples</span>` +
    `<span class="k">click the header or press Esc to go back to the table</span></div>`);
}

/* The tiles on screen, as [first, last) indices into S.focusList. Tiles are exactly one
   table column wide, which is what the focus header's run cell measures. */
function focusRange() {
  const sc = $("#focus-scroller"), tw = $("#focus-head .hcell").offsetWidth, th = rowH();
  const cols = Math.max(1, Math.floor(sc.clientWidth / tw)), n = S.focusList.length;
  const r0 = Math.max(0, Math.floor(sc.scrollTop / th) - 1);
  const r1 = Math.ceil((sc.scrollTop + sc.clientHeight) / th) + 1;
  return { cols, tw, th, n, first: Math.min(n, r0 * cols), last: Math.min(n, r1 * cols) };
}

const focusPool = [];
function renderFocus() {
  const sc = $("#focus-scroller");
  const { cols, tw, th, n, first, last } = focusRange();
  const x0 = Math.max(0, Math.floor((sc.clientWidth - cols * tw) / 2));
  $("#focus-spacer").style.height = `${Math.ceil(n / cols) * th}px`;
  $("#focus-empty").hidden = n > 0;
  while (focusPool.length < last - first) {
    const c = buildRunCell();
    c.classList.add("ftile");
    $("#focus-tiles").appendChild(c); focusPool.push(c);
  }
  const mixed = mixedShapes();
  focusPool.forEach((c, k) => {
    const i = first + k;
    c.hidden = i >= last;
    if (c.hidden) return;
    c.style.cssText = `width:${tw}px;height:${th}px;transform:translate(${x0 + (i % cols) * tw}px,${Math.floor(i / cols) * th}px)`;
    paintRunCell(c, S.focus, S.focusList[i], mixed);
  });
}

/* Rows are pooled by rank (slot rank % pool.length), so a row that stays on screen keeps
   its element. Scrolling (moved=true) paints only the rows that just came into view;
   everything else -- new data, a setting, a re-sort -- repaints every visible row. */
const pool = [];
function renderVisible(moved = false) {
  // Every control that restyles the table repaints through here, so the sample popup
  // follows them too.
  paintModalLive();
  // The focus view covers the table, and both share one frame store per run: painting
  // the hidden table too would have each view evict the other's window forever.
  if (S.focus) { renderFocus(); return; }
  const { first, last } = visibleRange();
  const rowsEl = $("#rows");
  while (pool.length < last - first) { const r = buildRow(); pool.push(r); rowsEl.appendChild(r); }
  const show = new Set();
  for (let rank = first; rank < last; rank++) {
    const el = pool[rank % pool.length];
    show.add(el);
    el.hidden = false;
    if (!moved || el._rank !== rank) { el._rank = rank; paintRow(el, rank); }
  }
  for (const el of pool) if (!show.has(el)) { el.hidden = true; el._rank = -1; }
}

let raf = 0;
function onScroll() {
  if (raf) return;
  raf = requestAnimationFrame(() => { raf = 0; renderVisible(true); });
}

function refresh(toTop = false) {
  fitMaskBox();
  if (!S.runs[S.focus]) S.focus = null;   // its run was removed
  $("#focus").hidden = !S.focus;
  // A run just added/removed can flip the effective row mode (see syncRowMode); the rows
  // then mean something different, so start from the top like a re-sort does.
  if (syncRowMode()) toTop = true;
  applyFilters();
  // Re-sorting puts different samples at rank 1, so keeping the old scroll offset would
  // strand the user mid-list; jump back to the top whenever the order itself changes.
  if (toTop) $("#scroller").scrollTop = 0;
  if (toTop && S.focus) $("#focus-scroller").scrollTop = 0;
  if (S.focus) renderFocusHeader();
  const head = $("#scroller").querySelector(".hrow");
  if (head) head.remove();
  const hdr = renderHeader();
  $("#scroller").insertBefore(hdr, $("#spacer"));

  // Rows are absolutely positioned, so they must start below the sticky header --
  // otherwise row 1 sits underneath it and cannot be scrolled into view. The header
  // grows with the number of run columns, so measure it rather than assume a height.
  const hh = hdr.offsetHeight;
  $("#rows").style.top = `${hh}px`;
  $("#spacer").style.height = `${hh + rowCount() * rowH()}px`;
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
  [...f.findSamples].sort((a, b) => a.localeCompare(b, undefined, { numeric: true })).forEach((v) => add("sample", f.findSamples, v));
  [...f.findPositions].sort((a, b) => a - b).forEach((v) => add("pos", f.findPositions, v));

  const n = f.findSamples.size + f.findPositions.size;
  $("#find-count").textContent = n ? `${n} active` : "";
  // The browsing filters are bypassed while a search is active; say so rather than
  // leaving the sidebar looking like it is lying.
  $("#sidebar").classList.toggle("searching", n > 0);
}

function bindFind() {
  const commit = (input, set, valid, parse = Number) => {
    // Accept a list, so pasting "871, 502 33" (or "12-3, 12-4") works as well as typing one at a time.
    const ids = (input.value.match(/\d+(?:-\d+)?/g) || []).map(parse);
    let added = 0;
    for (const id of ids) if (valid(id)) { set.add(id); added++; }
    if (added) { input.value = ""; renderFindChips(); refresh(true); }
    else if (ids.length) { input.classList.add("bad"); setTimeout(() => input.classList.remove("bad"), 600); }
  };
  const sampleIds = new Set(S.samples.map((s) => s.sample_id));
  const posIds = new Set(S.samples.map((s) => +s.output_id));
  $("#find-sample").onkeydown = (e) => {
    if (e.key === "Enter") commit($("#find-sample"), S.filters.findSamples, (i) => sampleIds.has(i), padId);
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
  // filed under that name (that also passes the other filters -- see FACET_VALUES).
  const splits = new Set();
  S.runOrder.forEach((n) => S.runs[n].splits.forEach((s) => splits.add(s)));
  // Which runs use each name. Kept off the chip face -- a run name on every chip is the
  // kind of noise that made these labels unreadable before -- and put in the tooltip,
  // where it answers "why does this chip exist / why is it empty for what I'm looking at"
  // at the moment the question is actually asked.
  const owners = new Map();
  S.runOrder.forEach((n) => S.runs[n].splits.forEach((sp) => {
    if (!owners.has(sp)) owners.set(sp, []);
    owners.get(sp).push(n);
  }));
  // Faceted counts (applyFilters); a value nothing under the other filters has is hidden.
  const faceted = (k, vals) => vals.map((v) => [v, (S.facets[k] && S.facets[k].get(v)) || 0])
    .filter(([v, n]) => n || (k === "objects" && f.objects && f.objects.has(v)));   // keep a picked object
  chipRow($("#split-chips"), faceted("splits", [...splits].sort()), f.splits,
    (v) => v, (v) => `${v} — used by ${(owners.get(v) || []).join(", ") || "no loaded run"}`);
  chipRow($("#layout-chips"), faceted("layouts", uniq((s) => s.layout).map(([v]) => v)), f.layouts, (v) => v);
  chipRow($("#nobj-chips"), faceted("nObjects", uniq((s) => s.n_objects).map(([v]) => v)), f.nObjects, (v) => `${v}`);
  chipRow($("#box-chips"), faceted("boxes", uniq((s) => s.box).map(([v]) => v)), f.boxes, (v) => v);
  chipRow($("#dataset-chips"), faceted("datasets", uniq((s) => s.dataset).map(([v]) => v)), f.datasets, (v) => v);
  // Contains-object: one chip per object TYPE, counted across every layout it appears
  // in, so "purple-cube" covers both the solo and the two-object scenes.
  const objVals = [...new Set(S.samples.flatMap(FACET_VALUES.objects))].sort();
  // Click picks (viz2 style); no filter (null) = every chip reads as on, and the first click picks just that one.
  $("#obj-chips").innerHTML = "";
  for (const [v, n] of faceted("objects", objVals)) {
    const b = document.createElement("button");
    b.className = "chip" + (f.objects && !f.objects.has(v) ? " off" : "");
    b.innerHTML = `${v === "(none)" ? "empty" : v}<span class="n">${n}</span>`;
    b.onclick = () => { f.objects ??= new Set(); f.objects.has(v) ? f.objects.delete(v) : f.objects.add(v); refresh(); };
    $("#obj-chips").appendChild(b);
  }
  $("#obj-count").textContent = f.objects ? `${f.objects.size}/${objVals.length}` : "all";
  $("#split-count").textContent = `${f.splits.size}/${splits.size}`;
  $("#layout-count").textContent = `${f.layouts.size}/${uniq((s) => s.layout).length}`;
  $("#nobj-count").textContent = `${f.nObjects.size}/${uniq((s) => s.n_objects).length}`;
  $("#box-count").textContent = `${f.boxes.size}/${uniq((s) => s.box).length}`;
  $("#dataset-count").textContent = `${f.datasets.size}/${uniq((s) => s.dataset).length}`;
  $("#spk-count").textContent = `${f.speakers.size}/8`;
  const sortRuns = activeSorts().map((st) => st.run);
  $("#metric-scope").textContent = sortRuns.length ? `range applies to ${sortRuns.join(", ")}` : "any loaded run in range";
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
   Mask values are fetched as fp16 (2 bytes per grid cell -- usually smaller than a PNG of
   the same cell) and drawn to <canvas> on the client, one epoch for the rows on screen at a
   time; the server reads only those rows, so a slider step is one small request. */

/* One epoch's frames for `sids`, as Map sid -> {m: fp16 mask}. Body: see app.py api_frames. */
async function fetchFrames(run, sids, ep, kind = "mask") {
  const r = await fetch(`/api/frames?run=${encodeURIComponent(run)}&sids=${sids.join(",")}` +
    `&epoch=${ep}&kind=${kind}&v=${S.renderVersion}`);
  if (!r.ok) throw new Error("frames");
  const masks = new Uint16Array(await r.arrayBuffer());
  const cells = runShape(run)[0] * runShape(run)[1];
  return new Map(sids.map((s, i) => [s, { m: masks.subarray(i * cells, (i + 1) * cells) }]));
}

/* Photo view: a run with a photo head (src8 --gray) shows its drawn grayscale photo in
   place of the mask. Its frames sit in the same store under "g:"-prefixed epoch keys. */
const isPhoto = (run) => S.view.photo && !!(S.runs[run] && S.runs[run].gray);
const frameKey = (run, ep) => (isPhoto(run) ? "g:" : "") + (ep ?? "latest");

const frameFor = (run, ep, sid) => {
  const st = S.frameData[run] && S.frameData[run][frameKey(run, ep)];
  return st && st.get(sid);
};

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
/* Mask canvases are supersampled: the buffer is the grid times an integer factor rather
   than the grid itself. The mask still draws EXACTLY as before -- integer scale with
   imageSmoothingEnabled off is the same pixelated result CSS was producing -- but it gives
   the crosshairs real pixels to be crisp in. Without this they would be drawn into a 30x21
   buffer and stretched into unreadable blocks.

   ~300px of buffer width is about 2x the on-screen mask box, which is enough for a clean
   1.5px line and costs ~250KB a canvas; the table keeps ~140 pooled canvases alive. */
const bufScale = (w) => Math.max(4, Math.ceil(300 / w));

/* Crosshair geometry for one (run, sample), or null when the marks are switched off.
   Served by /api/run as normalized [0,1] coordinates, so it needs no grid conversion and
   is valid whatever resolution the column predicts at. */
function comsFor(run, sid) {
  if (!S.view.coms) return null;
  const r = S.runs[run];
  return (r && r.samples && r.samples[sid] && r.samples[sid].coms) || null;
}

function sizeMaskCanvas(cv, gw, gh) {
  const k = bufScale(gw), W = gw * k, H = gh * k;
  if (cv.width !== W || cv.height !== H) { cv.width = W; cv.height = H; }
}

/* Object-level centre-of-mass marks.

   Encodes exactly two things, because a 130x110 box cannot carry three at once:
     ROLE     -- colour AND shape, redundantly: target is an X in the target colour,
                 prediction a circle in the prediction colour. Both are taken from the very
                 LUTs the masks are painted with, so a mark can never suggest a different
                 role than the fill beneath it, and the shape carries the distinction on its
                 own when the colours are hard to tell apart.
     IDENTITY -- the connecting line, not a per-object hue. The line IS the pairing, its
                 length IS the localization error, and it costs no extra colour vocabulary.
   So an X with no line is an object the run missed, and a lone circle a blob it invented
   (which `localization` does not charge for -- see match_objects).

   Every stroke is laid over a white halo so the marks survive both the photo backdrop and
   a saturated mask fill underneath. */
function drawComs(ctx, W, H, coms, gtOnly) {
  if (!coms) return;
  // Fixed colours, NOT the mask LUTs. Taking the LUT end-stops put each mark in the exact
  // colour of the fill it sits on -- a navy circle on the blue prediction, a near-black X
  // on the green target -- so both vanished into their own layer. These are picked to
  // clear BOTH fills: amber is nowhere in the blue prediction ramp, and the bright mint is
  // far lighter than any part of the green target ramp.
  const cTruth = "#00e5a0", cPred = "#ffab1a";
  const r = Math.max(3.2, Math.min(W, H) * 0.042);
  const lw = Math.max(1.4, r * 0.34);
  const X = (fc) => fc * W, Y = (fr) => fr * H;   // coms are [0,1] (row, col) fractions

  const halo = (draw, width) => {
    ctx.lineCap = "round"; ctx.lineJoin = "round";
    ctx.strokeStyle = "rgba(255,255,255,.92)"; ctx.lineWidth = width + Math.max(1.6, r * 0.3);
    draw();
    ctx.stroke();
  };
  const mark = (draw, width, col) => {
    halo(draw, width);
    ctx.strokeStyle = col; ctx.lineWidth = width;
    draw();
    ctx.stroke();
  };
  const xMark = (x, y) => mark(() => {
    ctx.beginPath();
    ctx.moveTo(x - r, y - r); ctx.lineTo(x + r, y + r);
    ctx.moveTo(x + r, y - r); ctx.lineTo(x - r, y + r);
  }, lw, cTruth);
  const oMark = (x, y) => mark(() => {
    ctx.beginPath(); ctx.arc(x, y, r * 0.85, 0, Math.PI * 2);
  }, lw, cPred);

  ctx.save();
  // Lines first so the marks sit on top of their own connector rather than under it.
  // Drawn thicker than the mark strokes: the line is the error, so it should read first.
  if (!gtOnly) for (const [pr, pc, gr, gc] of coms.pairs || []) {
    mark(() => { ctx.beginPath(); ctx.moveTo(X(pc), Y(pr)); ctx.lineTo(X(gc), Y(gr)); },
         lw * 1.15, "#12161c");
  }
  // Every TARGET object gets an X, matched or not -- `missed` alone would mark only the
  // ones the run failed to find, so the target half of a stacked cell would go blank the
  // moment the run got it right, which reads as "no object here".
  for (const [gr, gc] of coms.missed || []) xMark(X(gc), Y(gr));
  for (const q of coms.pairs || []) xMark(X(q[3]), Y(q[2]));
  if (!gtOnly) {
    for (const q of coms.pairs || []) oMark(X(q[1]), Y(q[0]));
    for (const [pr, pc] of coms.extra || []) oMark(X(pc), Y(pr));
  }
  ctx.restore();
}

function drawMask(canvas, values, mode, truth, shape, coms, gtOnly) {
  const { gamma, gain } = S.lut;
  // Grid comes from the CALLER, not the global default: runs trained at different
  // resolutions share one table, so each column draws at its own size.
  const [h, w] = shape || [S.lut.h, S.lut.w];
  const n = w * h;
  const ctx = canvas.getContext("2d");
  // The buffer is the grid supersampled (see bufScale); the mask is still drawn as exact
  // integer-scaled blocks, the extra pixels exist for the crosshairs.
  sizeMaskCanvas(canvas, w, h);
  const CW = canvas.width, CH = canvas.height;

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
  // `op`: the Objects slider fades every mask layer, predicted and target alike, so turning
  // it down shows what is underneath both.
  const predOp = S.view.maskOp;
  const layers = mode === "overlay"
    ? [{ v: truth, lut: S.lut.truth, k: 0.72, op: predOp }, { v: values, lut: S.lut.pred, k: 0.82, op: predOp }]
    : [{ v: values, lut: soloLut, k: gain, op: predOp }];

  ctx.clearRect(0, 0, CW, CH);
  // Off: the mask must stay hard-edged blocks. This is the same rendering CSS's
  // image-rendering:pixelated was doing, just moved into the buffer.
  ctx.imageSmoothingEnabled = false;
  // Each layer is composited at its own `op`, so the backdrop -- an element background,
  // outside the canvas -- keeps whatever the background slider gave it.
  for (const { v, lut, k, op } of layers) {
    ctx.globalAlpha = op;
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
    ctx.drawImage(off, 0, 0, CW, CH);
  }
  ctx.globalAlpha = 1;
  // Marks are NOT scaled by the cube-opacity slider: they are annotation, not mask, and
  // fading them with the fill would hide the thing being measured exactly when the fill
  // is turned down to see underneath it.
  ctx.imageSmoothingEnabled = true;
  drawComs(ctx, CW, CH, coms, gtOnly);
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
  const [rh, rw] = runShape(run);
  const cells = rh * rw;
  // Every early return clears first. A canvas keeps its last drawing until something
  // overwrites it, and rows are RECYCLED, so leaving it alone shows the previous
  // occupant's prediction -- or, once a filter change brings in rows the current store
  // was not fetched for, a stale mask that never gets repainted.
  // The box says why it is empty (see .maskbox[data-state] in style.css):
  //   loading   -- this epoch's frames for this row have not arrived yet
  //   unsaved   -- the run covers this sample, but this epoch saved no prediction for it
  //                (the train loader's drop_last + shuffle drops a different remainder
  //                each epoch, see src/model/dataset.py build_dataset)
  const blank = (state) => {
    cv.getContext("2d").clearRect(0, 0, cv.width, cv.height);
    cv.parentElement.dataset.state = state;
  };
  const fr = frameFor(run, epoch, sid);
  if (!fr) {
    // Same run and sample, another epoch: keep that drawing up until this one lands, so
    // stepping the slider doesn't flash every cell blank.
    if (cv._drawn !== `${run}|${sid}`) blank("loading");
    ensureFrames(run);
    return;
  }
  const raw = fr.m;
  const buf = scratchFor(cells);
  for (let i = 0; i < cells; i++) buf[i] = half(raw[i]);
  if (Number.isNaN(buf[0])) { blank("unsaved"); return; }
  if (isPhoto(run)) {
    // pred/stacked: the drawn photo; overlay: + the true object tinted on it; diff: drawn
    // minus the real photo, on the mask diff ramp (the real one sits in the stacked half).
    const m = S.view.mode, tk = truthKey(sid, rh, rw);
    const aux = m === "overlay" ? S.truthCache[tk] : m === "diff" ? S.photoCache[tk] : null;
    if ((m === "overlay" || m === "diff") && !aux) {
      blank("loading");
      if (m === "overlay") ensureTruth(sid, rh, rw); else ensurePhoto(sid, rh, rw);
      return;
    }
    cv.parentElement.dataset.state = "";
    cv._drawn = `${run}|${sid}`;
    drawPhoto(cv, buf, m === "overlay" ? aux : null, [rh, rw], comsFor(run, sid), false, m === "diff" ? aux : null);
    return;
  }
  // In stacked mode this canvas is the prediction half only -- the target is drawn into
  // its own canvas below, so it needs no truth here.
  const mode = S.view.mode === "stacked" ? "pred" : S.view.mode;
  const wantTruth = mode === "diff" || needsTruth(mode);
  // Truth is cached per (sample, grid): the same sample has a different target array at
  // each resolution, so keying on sid alone would diff against the wrong-shaped mask.
  const tkey = truthKey(sid, rh, rw);
  const truth = wantTruth ? S.truthCache[tkey] : null;
  if (wantTruth && !truth) { blank("loading"); ensureTruth(sid, rh, rw); return; }
  cv.parentElement.dataset.state = "";
  cv._drawn = `${run}|${sid}`;
  drawMask(cv, buf, mode, truth, [rh, rw], comsFor(run, sid));
}

/* Fetch the epoch on screen for the rows on screen plus a screen ahead into the run's
   store. Merged, so scrolling back or returning to an epoch never refetches. Rows come
   from visibleRange(), the same window renderVisible paints. A call while one is in flight
   queues one re-check, since scrolling and scrubbing move what is needed. */
const framesPending = new Set();
const framesStale = new Set();
async function ensureFrames(run) {
  if (framesPending.has(run)) { framesStale.add(run); return; }
  framesPending.add(run);
  try {
    do {
      framesStale.delete(run);
      if (!S.runs[run]) break;   // removed while a fetch was in flight
      const f = S.focus === run;
      const { first, last } = f ? focusRange() : visibleRange();
      const sids = (f ? S.focusList : orderFor(run)).slice(first, 2 * last - first).filter(Boolean).map((s) => s.i);
      const ms = modalSid(run);   // the sample popup's, which arrows can move off screen
      if (ms != null) sids.push(ms);
      const store = S.frameData[run] || (S.frameData[run] = {});
      const ep = epochFor(run) ?? "latest", key = frameKey(run, ep);
      const have = store[key] || (store[key] = new Map());
      const need = sids.filter((i) => !have.has(i));
      if (!need.length) continue;
      for (const [sid, fr] of await fetchFrames(run, need, ep, isPhoto(run) ? "gray" : "mask")) have.set(sid, fr);
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
    const d = await api(`/api/values?sid=${sid}${q}&v=${S.renderVersion}`);
    S.truthCache[key] = d.v.flat();
    renderVisible();
  } finally {
    truthPending.delete(key);
  }
}

/* Real photos for the photo view, batched: every cell that misses in one frame queues its
   sid, and one /api/photos request per grid fetches them all. */
const photoQueue = new Map();   // "HxW" -> Set of sids
const photoPending = new Set();
function ensurePhoto(sid, h, w) {
  const key = truthKey(sid, h, w);
  if (photoPending.has(key)) return;
  photoPending.add(key);
  const g = `${h}x${w}`;
  if (!photoQueue.size) requestAnimationFrame(flushPhotos);
  (photoQueue.get(g) || photoQueue.set(g, new Set()).get(g)).add(sid);
}
async function flushPhotos() {
  const jobs = [...photoQueue]; photoQueue.clear();
  await Promise.all(jobs.map(async ([g, set]) => {
    const sids = [...set], [h, w] = g.split("x").map(Number), n = h * w;
    try {
      const r = await fetch(`/api/photos?sids=${sids.join(",")}&shape=${g}&v=${S.renderVersion}`);
      if (!r.ok) throw new Error("photos");
      const raw = new Uint16Array(await r.arrayBuffer());
      sids.forEach((sid, i) => {
        const v = new Float32Array(n);
        for (let k = 0; k < n; k++) v[k] = half(raw[i * n + k]);
        S.photoCache[truthKey(sid, h, w)] = v;
      });
    } finally {
      for (const sid of sids) photoPending.delete(truthKey(sid, h, w));
    }
  }));
  renderVisible();
}

/* A grayscale photo, always opaque, with no backdrop behind it. `objMask`
   (overlay) tints the TRUE object over it in the target colour, so where the run drew the
   object and where it really is read in one box. */
function drawPhoto(canvas, values, objMask, shape, coms, gtOnly, real) {
  const [h, w] = shape, n = h * w;
  const ctx = canvas.getContext("2d");
  sizeMaskCanvas(canvas, w, h);
  const CW = canvas.width, CH = canvas.height;
  const off = drawPhoto._off || (drawPhoto._off = document.createElement("canvas"));
  if (off.width !== w || off.height !== h) { off.width = w; off.height = h; }
  const octx = off.getContext("2d");
  const img = drawPhoto._img && drawPhoto._img.width === w && drawPhoto._img.height === h
    ? drawPhoto._img : (drawPhoto._img = octx.createImageData(w, h));
  const px = img.data;
  const tl = S.lut.truth, tc = tl[tl.length - 1] || [0, 200, 120];
  if (real) {
    // Diff: drawn minus real on the diverging ramp, opaque (a photo has no "empty" cells to
    // see through). Full scale is +-0.5 of the [0,1] photo; "Color per sample" rescales.
    const dl = S.lut.diff, dom = S.view.relative ? sampleDomain(values, real, "diff", n) : null;
    const span = dom ? dom.span : 0.5;
    for (let i = 0; i < n; i++) {
      const d = Math.max(-1, Math.min(1, (values[i] - real[i]) / span));
      const t = 0.5 + 0.5 * Math.sign(d) * Math.pow(Math.abs(d), S.lut.gamma);
      const c = dl[Math.round(t * (dl.length - 1))] || [0, 0, 0];
      px[i * 4] = c[0]; px[i * 4 + 1] = c[1]; px[i * 4 + 2] = c[2]; px[i * 4 + 3] = 255;
    }
  } else
  for (let i = 0; i < n; i++) {
    let g = Math.round(Math.max(0, Math.min(1, values[i])) * 255), r = g, b = g;
    if (objMask) {
      const a = 0.75 * S.view.maskOp * Math.max(0, Math.min(1, objMask[i]));
      r = r * (1 - a) + tc[0] * a; g = g * (1 - a) + tc[1] * a; b = b * (1 - a) + tc[2] * a;
    }
    px[i * 4] = r; px[i * 4 + 1] = g; px[i * 4 + 2] = b; px[i * 4 + 3] = 255;
  }
  octx.putImageData(img, 0, 0);
  ctx.clearRect(0, 0, CW, CH);
  ctx.imageSmoothingEnabled = false;
  ctx.drawImage(off, 0, 0, CW, CH);   // opaque: a photo, not an object mask (the Objects slider fades only the tint)
  ctx.imageSmoothingEnabled = true;
  drawComs(ctx, CW, CH, coms, gtOnly);
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

/* Saved metrics for every sample of `run` at numbered epoch `ep`, fetched once per
   (run, epoch): {saved, samples: {sid: {metric: value}}}, or null while in flight (header
   and cells repaint when it lands). A failed fetch is dropped so the next paint retries. */
function epochMetrics(run, ep) {
  const store = S.epochMetrics[run] || (S.epochMetrics[run] = {});
  if (ep in store) return store[ep];
  store[ep] = null;
  api(`/api/epoch_metrics?run=${encodeURIComponent(run)}&epoch=${ep}&v=${S.renderVersion}`)
    .then((d) => { if (S.epochMetrics[run] === store) { store[ep] = d; renderHeaderStats(); renderVisible(); } })
    .catch(() => { if (S.epochMetrics[run] === store) delete store[ep]; });
  return null;
}

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
   the display or pile up work in a background tab. */
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

/* Every epoch change -- slider, steppers, playback -- goes through here. Masks and
   metrics repaint from that epoch's frames (fetched on first view), and nothing else
   moves -- rows, order and filters belong to the latest epoch. */
function setEpochIdx(i) {
  const eps = allEpochs();
  S.epochIdx = i == null ? null : Math.max(0, Math.min(i, eps.length - 1));
  renderEpochPanel();
  document.querySelectorAll(".hep").forEach((el) => { el.innerHTML = epochLabel(el.dataset.run); });
  renderHeaderStats();
  renderVisible();
}

let playTimer = 0;
function togglePlay() {
  S.playing = !S.playing;
  clearInterval(playTimer);
  if (S.playing) {
    const eps = allEpochs();
    if (S.epochIdx == null || S.epochIdx >= eps.length - 1) S.epochIdx = 0;
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
  // A new dataset appeared server-side: the sample list, filters and row ids all change,
  // so reload the page (keeping the loaded columns) rather than patch every structure.
  if (meta.n_samples !== S.meta.n_samples) {
    try { sessionStorage.setItem("viz.reloadRuns", JSON.stringify(S.runOrder)); } catch (_) {}
    location.reload();
    return;
  }
  const prevNames = new Set(S.meta.runs.map((r) => r.name));
  const prevStatus = new Map(S.meta.runs.map((r) => [r.name, r.status]));
  S.meta = meta;
  // A loaded run going running -> done/crashed only shows in its column header.
  const statusChanged = S.runOrder.some((n) => {
    const cur = meta.runs.find((r) => r.name === n);
    return cur && prevStatus.get(n) !== cur.status;
  });

  // A slider parked on the newest epoch keeps following it as runs save new ones, instead
  // of staying pinned to that index while the newest epoch moves on without it.
  const followNewest = S.epochIdx != null && S.epochIdx >= allEpochs().length - 1;
  const advanced = [];
  for (const name of S.runOrder) {
    const cur = meta.runs.find((r) => r.name === name);
    if (cur && S.runs[name] && cur.epoch != null && cur.epoch !== S.runs[name].epoch) {
      advanced.push(name);
    }
  }
  for (const name of advanced) await addRun(name, true);

  if (advanced.length) {
    // Only "latest" moved; numbered epochs are immutable, so their frames stay.
    for (const name of advanced) if (S.frameData[name]) { delete S.frameData[name].latest; delete S.frameData[name]["g:latest"]; }
    invalidateEpochs();
    if (followNewest && !S.playing) S.epochIdx = null;   // null = each run's latest
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
    // Fetch in flight (see addInFlight): shown as its own state right away, rather than
    // leaving the row looking clickable -- and still unresponsive -- for however long
    // /api/run/{name} takes, which is what made a slow load look like a dead click.
    const adding = addInFlight.has(r.name);
    const ok = r.compatible && !added && !adding;
    const d = document.createElement("div");
    d.className = "ritem" + (r.compatible ? (added ? " added" : adding ? " adding" : "") : " bad") + (ok ? " ok" : "");
    d.innerHTML = `<div><div class="nm">${r.name}</div>
      <div class="sub">${statusChip(r.status)} ${r.compatible
        ? `ep ${r.epoch ?? "?"} · ${r.eval_splits.length} eval splits` : r.reason}</div></div>
      <span class="badge">${added ? "added" : adding ? "adding…" : r.compatible ? "add" : "unavailable"}</span>`;
    if (ok) d.onclick = () => pick(r.name);
    list.appendChild(d);
  });
  pickCur = -1;
  list.hidden = false;
  q.setAttribute("aria-expanded", "true");
}

// Deliberately does NOT close the picker: arrow-key-and-Enter is the fast path for adding
// several runs in one pass (e.g. every eval run across all boxes for a comparison), and
// closing after each pick would force reopening + re-searching between every one. The
// search term is kept too, so repeated picks under one query stay filtered. Escape (see
// #run-search's keydown handler) still closes it explicitly when the user is done.
async function pick(name) {
  // addInFlight is set synchronously before addRun's first await, so by the time this
  // call returns control here the picker can already re-render the row as "adding…"
  // instead of sitting there looking like the click did nothing until the fetch resolves.
  const p = addRun(name);
  if (pickerOpen()) openPicker();
  await p;
  if (pickerOpen()) openPicker();   // re-render again so it now shows as "added"
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
    valueCache.set(k, api(`/api/values?sid=${sid}&run=${encodeURIComponent(run)}&mode=${mode}${q}&v=${S.renderVersion}`));
  return valueCache.get(k);   // {v} or, in overlay/stacked mode, {v, t}
}

/* Distance from the cursor to each prediction<->target connector, in CSS px. Returns the
   label for the closest one within HIT_PX, or null. */
const HIT_PX = 7;
function hitConnector(el, ev, b) {
  if (!S.view.coms || !el.dataset.run) return null;
  const coms = comsFor(el.dataset.run, el.dataset.sid);
  if (!coms || !coms.pairs || !coms.pairs.length) return null;
  const [mh, mw] = runShape(el.dataset.run);
  const px = ev.clientX - b.left, py = ev.clientY - b.top;
  let best = null, bestD = HIT_PX;
  for (const [pr, pc, gr, gc] of coms.pairs) {
    const x1 = pc * b.width, y1 = pr * b.height, x2 = gc * b.width, y2 = gr * b.height;
    const dx = x2 - x1, dy = y2 - y1;
    const len2 = dx * dx + dy * dy;
    // Projection parameter clamped to [0,1] so a coincident pair (len2 = 0) degrades to
    // "distance to the point" instead of dividing by zero.
    const t = len2 > 0 ? Math.max(0, Math.min(1, ((px - x1) * dx + (py - y1) * dy) / len2)) : 0;
    const d = Math.hypot(px - (x1 + t * dx), py - (y1 + t * dy));
    if (d < bestD) {
      bestD = d;
      // Normalized L2 -- the same quantity localization_rel averages over a sample's
      // objects -- plus the raw cell distance, which is the unit the grid is read in.
      const fr = Math.hypot(pr - gr, pc - gc);
      const cells = Math.hypot((pr - gr) * mh, (pc - gc) * mw);
      best = `L2 ${fr.toFixed(4)} of box · ${cells.toFixed(2)} cells (${(fr * 100).toFixed(2)}%)`;
    }
  }
  return best;
}

/* One switch for both surfaces. The table and the modal draw the marks from the same
   S.view.coms, so flipping it has to repaint both -- an open modal sits ON TOP of the
   table, and repainting only what is underneath reads as the toggle being dead. */
function setComs(on) {
  S.view.coms = !!on;
  const cb = $("#com-toggle");
  if (cb) cb.checked = S.view.coms;
  renderVisible();
  syncModalComs();
}

function bindTooltip() {
  const tip = $("#tooltip");
  let cur = null;
  $("#scroller").addEventListener("mousemove", async (ev) => {
    // dataset.run is "" on the ground-truth cell, which still has values to show.
    const img = ev.target.closest?.(".mask");
    if (!img || img.dataset.sid === undefined) { tip.hidden = true; cur = null; return; }
    const b = img.getBoundingClientRect();
    // A connector line takes priority over the cell value: the line IS the localization
    // error, so hovering it should report that error rather than whatever cell happens to
    // sit under the cursor. Distance is measured in CSS pixels, not in the [0,1] fractions
    // the coordinates arrive in -- the box is not square, so a fraction-space radius would
    // make the line easier to hit horizontally than vertically.
    const lineHit = hitConnector(img, ev, b);
    if (lineHit) {
      tip.hidden = false;
      tip.textContent = lineHit;
      tip.style.left = `${ev.clientX + 12}px`;
      tip.style.top = `${ev.clientY + 14}px`;
      cur = null;                 // force a refetch when the cursor leaves the line
      return;
    }
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

/* Object-COM marks on the modal's three panels.

   Kept in S.modalComs and repainted from here rather than drawn once at open time, so the
   marks respond to the same toggle the table uses -- the modal covers the table, and a
   switch that only worked underneath it would look broken. Panels are server-rendered
   PNGs, so each gets its own overlay canvas; the ground-truth panel is marked gtOnly and
   shows target X's without the prediction circles or connectors. */
function syncModalComs() {
  const nb = document.querySelector(".nbviews");
  if (!nb) return;
  // Re-measure whenever a panel's box changes. One rAF after building the modal is too
  // early: the <img> has not loaded, so it is still ~0 tall and every mark landed squashed
  // into a strip along the top edge. The observer catches the load, the aspect-ratio
  // resolving, and a window resize with the same code path. Canvases are absolutely
  // positioned, so sizing them cannot itself trigger the observer.
  if (syncModalComs._nb !== nb) {
    syncModalComs._nb = nb;
    const ro = (syncModalComs._ro = syncModalComs._ro || new ResizeObserver(() => syncModalComs()));
    ro.disconnect();
    nb.querySelectorAll("figure[data-coms] img").forEach((im) => ro.observe(im));
  }
  const st = S.modalComs;
  nb.querySelectorAll("figure[data-coms]").forEach((fig) => {
    let cv = fig.querySelector("canvas.nbcoms");
    if (!cv) {
      cv = document.createElement("canvas");
      cv.className = "nbcoms";
      fig.insertBefore(cv, fig.querySelector("figcaption"));
    }
    const img = fig.querySelector("img");
    const box = img.getBoundingClientRect();
    // Buffer follows the rendered panel, which is far larger than a table cell, so the
    // marks get proportionally the same weight rather than being scaled up from a
    // cell-sized buffer.
    const W = Math.round(box.width), H = Math.round(box.height);
    if (W < 8 || H < 8) return;          // not laid out yet; the observer will call back
    if (cv.width !== W || cv.height !== H) { cv.width = W; cv.height = H; }
    // Pinned to the IMG's own box, not the figure's, so the caption below it cannot shift
    // the marks off the mask they annotate.
    cv.style.left = `${img.offsetLeft}px`;
    cv.style.top = `${img.offsetTop}px`;
    cv.style.width = `${W}px`;
    cv.style.height = `${H}px`;
    const ctx = cv.getContext("2d");
    ctx.clearRect(0, 0, W, H);
    if (S.view.coms && st && st.coms)
      drawComs(ctx, W, H, st.coms, fig.dataset.coms === "gt");
  });
}

/* ***** detail modal ***** */

async function openModal(rank) {
  S.modalRow = rank;
  S.modalNav = null;
  const s = S.order[rank];
  if (!s) return;
  $("#m-prev").hidden = $("#m-next").hidden = false;
  $("#m-prev").disabled = $("#m-next").disabled = false;
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
      <img class="hero" src="/api/overhead/${s.i}.png?v=${S.renderVersion}" alt="overhead view of sample ${d.sample_id}">
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
        ${d.has.spectrogram ? `<img class="specimg" src="/api/vibration/${s.i}/spectrogram.png?v=${S.renderVersion}" alt="spectrogram">` : ""}
        ${d.has.fft ? `<img class="specimg" src="/api/vibration/${s.i}/fft.png?v=${S.renderVersion}" alt="FFT">` : ""}
      </div>
    </div>`;
  $("#modal").hidden = false;
}

/* ***** predicted-mask modal *****
   Ranks ground-truth samples by how close their center of mass is to what this run
   predicted, which shows whether a prediction looks like a different scene than its own
   target. Positions are deduped: 8 samples share each one, differing only by speaker. */

let nbToken = 0;
async function openNeighbors(run, sid) {
  const tok = ++nbToken;
  const d = await api(`/api/neighbors?run=${encodeURIComponent(run)}&sid=${sid}&k=5`);
  if (tok !== nbToken) return;   // a later arrow press already asked for another sample
  const dc = (a, b) => `${fmt(a, 1)}, ${fmt(b, 1)}`;
  const off = Math.hypot(d.pred_com[0] - d.gt_com[0], d.pred_com[1] - d.gt_com[1]);

  const list = (rows, cls) => rows.map((x) => `
    <li class="nb ${cls}" data-i="${x.i}">
      <img src="/api/gt_mask.png?sid=${x.i}&bg=1&shape=${d.shape[0]}x${d.shape[1]}&v=${S.renderVersion}" loading="lazy" alt="">
      <div class="nbmeta">
        <div class="nbtop"><b>${x.sample_id}</b></div>
        <div class="nbsub">d ${fmt(x.distance, 3)} · com ${dc(x.com[0], x.com[1])}</div>
        <div class="nbsub">${shortLayout(x.layout)} · ${x.n_objects} obj</div>
      </div>
    </li>`).join("");

  const v = `v=${S.renderVersion}`;
  const ep = S.runs[run] ? `&ep=${S.runs[run].epoch}` : "";
  const e = S.runs[run] && S.runs[run].samples[sid];

  const rel = S.view.relative ? 1 : 0;
  S.modalComs = { coms: comsFor(run, sid) || (S.runs[run] && S.runs[run].samples[sid] || {}).coms || null };
  $("#m-title").innerHTML = `Sample ${d.sample_id} — <span class="mrun">${run}</span>`;
  $("#m-body").innerHTML = `
    <div class="msec">
      <h3>This prediction — as displayed (the sidebar's controls apply)</h3>
      <div class="mlive">
        <figure><canvas class="mask predmask"></canvas><figcaption class="mlcap"></figcaption></figure>
        <figure hidden><canvas class="mask truthmask"></canvas><figcaption class="mlcap2"></figcaption></figure>
      </div>
      <div class="nbhead">
        <div><span class="k">predicted com</span><b>${dc(d.pred_com[0], d.pred_com[1])}</b></div>
        <div><span class="k">ground truth</span><b>${dc(d.gt_com[0], d.gt_com[1])}</b></div>
        <div><span class="k">offset</span><b>${fmt(off, 2)} cells</b></div>
        ${e ? METRICS.map((mm) => `<div><span class="k">${mm.short}</span><b>${
              e[mm.key] == null ? "–" : fmt(e[mm.key], 3)}</b></div>`).join("") : ""}
      </div>
      <p class="note">X marks each target object, a circle each predicted one, and the line
        between a matched pair is that object's localization error — hover it for the L2.
        Press <kbd>c</kbd> to hide or show them.</p>
      <p class="note">Ground-truth scenes ranked by distance from this run's predicted
        center of mass, over ${d.n_candidates} samples with an object (empty boxes
        excluded). One entry per physical position.</p>
    </div>
    <div class="mgrid">
      <div class="msec"><h3>Most similar to the prediction</h3><ul class="nblist">${list(d.most_similar, "near")}</ul></div>
      <div class="msec"><h3>Least similar</h3><ul class="nblist">${list(d.least_similar, "far")}</ul></div>
    </div>`;
  // The popup shows ONE sample, so its panels take that sample's own box aspect.
  {
    const smp = S.samples.find((x) => x.i === sid), asp = smp && smp.aspect > 0 ? smp.aspect : 1.3;
    $(".mlive").style.setProperty("--ml-asp", String(asp));
  }

  // Each neighbour opens its own ground-truth detail, if it is on screen.
  $("#m-body").querySelectorAll(".nb").forEach((li) => {
    li.onclick = () => {
      const rank = S.order.findIndex((s) => s.i === li.dataset.i);
      if (rank >= 0) openModal(rank);
    };
  });
  // Arrow keys and the ← → buttons step through the samples this run's column shows, in
  // the order on screen (the focus grid's, when this run is in focus).
  const ids = (S.focus === run ? S.focusList : orderFor(run))
    .filter((s) => s && S.runs[run] && S.runs[run].samples[s.i]).map((s) => s.i);
  S.modalRow = -1;
  S.modalNav = { run, ids, k: ids.indexOf(sid) };
  $("#m-prev").hidden = $("#m-next").hidden = S.modalNav.k < 0;
  $("#m-prev").disabled = S.modalNav.k <= 0;
  $("#m-next").disabled = S.modalNav.k >= ids.length - 1;
  S.modalNav.sid = sid;
  $("#modal").hidden = false;
  paintModalLive();
}

const modalSid = (run) => (S.modalNav && S.modalNav.run === run && !$("#modal").hidden ? S.modalNav.sid : null);

/* The popup's live panels: the same painters as a table cell, so mode, photo view,
   backdrop, the Objects slider, COMs and the epoch all match the table. */
function paintModalLive() {
  const nav = S.modalNav, box = document.querySelector(".mlive");
  if (!nav || !box || $("#modal").hidden || !S.runs[nav.run]) return;
  const run = nav.run, sid = nav.sid, [rh, rw] = runShape(run);
  const [cv, tv] = box.querySelectorAll("canvas");
  cv.className = "mask predmask" + (S.view.background && !isPhoto(run) ? " bg" : " nobg");
  cv.dataset.run = run; cv.dataset.sid = sid;
  setBackdrop(cv, sid);
  sizeMaskCanvas(cv, rw, rh);
  paintCanvas(cv, run, sid, epochFor(run));
  const what = isPhoto(run) ? "drawn photo" : "predicted mask";
  box.querySelector(".mlcap").textContent = `${what} · ${S.view.mode} · ep ${epochFor(run) ?? S.runs[run].epoch}`;
  const stacked = S.view.mode === "stacked";
  tv.parentElement.hidden = !stacked;
  if (stacked) {
    paintTruthHalf(tv, run, sid, rh, rw);
    box.querySelector(".mlcap2").textContent = isPhoto(run) ? "real photo" : "ground truth";
  }
}

function stepModal(delta) {
  const nav = S.modalNav;
  if (nav) {
    const sid = nav.ids[nav.k + delta];
    if (sid != null) { nav.k += delta; openNeighbors(nav.run, sid); }
    return;
  }
  if (S.modalRow < 0) return;
  const next = S.modalRow + delta;
  if (next >= 0 && next < S.order.length) openModal(next);
}

/* ***** wiring ***** */

function bindUI() {
  $("#scroller").addEventListener("scroll", onScroll, { passive: true });
  $("#focus-scroller").addEventListener("scroll", onScroll, { passive: true });

  /* Clicking a column header's epoch pins every column there. Delegated on the scroller
     because renderHeader rebuilds the header row on every refresh. */
  $("#scroller").addEventListener("click", (e) => {
    const b = e.target.closest("b.goep[data-ep]");
    if (!b) return;
    e.stopPropagation();          // must not also cycle that column's sort
    gotoEpoch(+b.dataset.ep);
  });

  window.addEventListener("resize", renderVisible);
  // The focus grid's column count follows its own width, which also changes when the
  // sidebar collapses -- not only on a window resize.
  new ResizeObserver(() => S.focus && renderFocus()).observe($("#focus-scroller"));

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

  $("#rowmode-seg").querySelectorAll("button").forEach((b) => {
    b.onclick = () => setRowMode(b.dataset.rowmode);
  });
  let storedRowMode = "linked";
  try { storedRowMode = localStorage.getItem("viz.rowmode") || "linked"; } catch (_) {}
  setRowMode(["single", "dataset"].includes(storedRowMode) ? storedRowMode : "linked");

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
     -- its veil is pure CSS, so dragging costs no repaint. The prediction is drawn INTO
     the canvases, so that one also has to repaint; the ground-truth image ignores it. */
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
  // Photo view (P): runs with a photo head draw it instead of their mask.
  const setPhoto = (on) => {
    S.view.photo = !!on;
    $("#photo-toggle").checked = S.view.photo;
    try { localStorage.setItem("viz.photo", on ? "1" : "0"); } catch (_) {}
    renderVisible();
  };
  window.setPhoto = setPhoto;
  $("#photo-toggle").onchange = (e) => setPhoto(e.target.checked);
  try { if (localStorage.getItem("viz.photo") === "1") { S.view.photo = true; $("#photo-toggle").checked = true; } } catch (_) {}
  $("#com-toggle").onchange = (e) => setComs(e.target.checked);

  $("#epoch-slider").oninput = (e) => setEpochIdx(+e.target.value);
  $("#epoch-play").onclick = togglePlay;
  holdToRepeat($("#epoch-prev"), () => stepEpoch(-1), $("#epoch-slider"));
  holdToRepeat($("#epoch-next"), () => stepEpoch(1), $("#epoch-slider"));
  $("#epoch-latest").onclick = () => {
    S.playing = false; clearInterval(playTimer);
    setEpochIdx(null);
  };

  document.querySelectorAll("[data-all]").forEach((b) => {
    b.onclick = () => {
      const k = b.dataset.all;
      if (k === "speakers") S.filters.speakers = new Set([1, 2, 3, 4, 5, 6, 7, 8]);
      if (k === "layouts") S.filters.layouts = new Set(uniq((s) => s.layout).map((x) => x[0]));
      if (k === "nObjects") S.filters.nObjects = new Set(uniq((s) => s.n_objects).map((x) => x[0]));
      if (k === "boxes") S.filters.boxes = new Set(uniq((s) => s.box).map((x) => x[0]));
      if (k === "objects") S.filters.objects = null;
      if (k === "datasets") S.filters.datasets = new Set(uniq((s) => s.dataset).map((x) => x[0]));
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
    // Esc backs out one layer at a time: the sample modal first, then the focus view.
    if (e.key === "Escape" && S.focus && $("#modal").hidden && !pickerOpen()) { setFocus(null); return; }
    if (e.key === "Escape") { document.querySelectorAll(".overlay").forEach((o) => (o.hidden = true)); closePicker(); }
    if (!$("#modal").hidden) {
      if (e.key === "ArrowLeft") { e.preventDefault(); stepModal(-1); return; }
      if (e.key === "ArrowRight") { e.preventDefault(); stepModal(1); return; }
      // Every other shortcut (c, b, p, f, g) falls through: the popup leaves the sidebar
      // live and repaints with the table, so they restyle the sample it shows.
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
    // Bare "c" toggles the object COM marks, same terms again.
    if (e.key === "c" && !typing && !e.ctrlKey && !e.metaKey && !e.altKey) {
      e.preventDefault();
      setComs(!S.view.coms);
    }
    // Bare "p" toggles the photo view, same terms again.
    if (e.key === "p" && !typing && !e.ctrlKey && !e.metaKey && !e.altKey) {
      e.preventDefault();
      window.setPhoto(!S.view.photo);
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
