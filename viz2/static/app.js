/* viz2 — client.
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
  { key: "mse",     label: "MSE",      short: "mse", worst: "high" },
  { key: "iou",     label: "Soft IoU", short: "iou", worst: "low"  },
  { key: "comdist", label: "COM dist", short: "cd",  worst: "high" },
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
  sort: { run: null, metric: "comdist", dir: "worst" },
  view: { mode: "pred", background: true },
  domain: {}, positions: [], activePos: -1, modalRow: -1,
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

/* Split names run long (purple_green_cubes_speaker). Abbreviate the words but keep the
   distinctions that matter -- which cubes, and whether it is the held-out-speaker
   variant. The full name stays in the chip's title attribute. */
function shortSplit(s) {
  if (!s) return "–";
  if (s === "train") return "train";
  const spk = s.endsWith("_speaker");
  const base = spk ? s.slice(0, -8) : s;
  const abbr = base.split("_").map((w) => (w === "cubes" || w === "cube" ? "" : w[0])).join("");
  return (abbr || base) + (spk ? "+spk" : "");
}
const api = (u) => fetch(u).then((r) => { if (!r.ok) throw new Error(u); return r.json(); });

/* ***** boot ***** */

async function boot() {
  readRowMetrics();          // before anything measures or positions a row
  const [runs, samples, lut] = await Promise.all([
    api("/api/runs"), api("/api/samples"), api("/api/lut")]);
  S.lut = lut;
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

const truthKey = (sid, h, w) => (h ? `${sid}@${h}x${w}` : String(sid));

/* Whether the loaded columns disagree on grid size, so the header only spends space on
   the size when it actually disambiguates something. */
function mixedShapes() {
  const seen = new Set(S.runOrder.map((n) => runShape(n).join("x")));
  return seen.size > 1;
}

/* Membership at the run's latest epoch: which samples it covers and their splits. This is
   the run's identity for filtering, and it must survive the epoch scrubber swapping
   `samples` -- see predictedSplit. */
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
  for (const sp of d.splits) S.filters.splits.add(sp);
  recomputeDomains();
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

/* A sample is in the table if any loaded run predicted it.

   Read from `splitOf`, the run's membership at its LATEST epoch, not from `samples`,
   which the epoch scrubber swaps out. A run does not save every sample at every epoch
   (early epochs in particular cover a different subset), and a sample missing from one
   epoch has not left the run -- its ground truth certainly has not changed. Keying the
   filter on the swapped table made those rows fail `passes` and vanish from the table
   entirely, taking the ground-truth column with them. */
function predictedSplit(sampleIdx) {
  for (const n of S.runOrder) {
    const sp = S.runs[n].splitOf && S.runs[n].splitOf[sampleIdx];
    if (sp) return sp;
  }
  return null;
}

function passes(s) {
  const f = S.filters;
  // An explicit ID search overrides the browsing filters: when you ask for a sample by
  // number you want to see it, not have it hidden by a speaker chip you set earlier.
  if (f.findSamples.size || f.findPositions.size) {
    return f.findSamples.has(+s.sample_id) || f.findPositions.has(+s.output_id);
  }
  if (!f.speakers.has(s.speaker)) return false;
  if (!f.layouts.has(s.layout)) return false;
  if (!f.nObjects.has(s.n_objects)) return false;
  if (!f.boxes.has(s.box)) return false;
  // Contains-any: a scene passes if any object in it is selected. Empty boxes have no
  // objects, so they ride on the "empty" pseudo-entry rather than never matching.
  const objs = s.objects && s.objects.length ? s.objects : ["(none)"];
  if (!objs.some((o) => f.objects.has(o))) return false;
  if (f.positions && !(s.pos >= 0 && f.positions.has(s.pos))) return false;

  // With no runs loaded the table is a ground-truth browser, so every sample qualifies;
  // the split filter only applies once some run has assigned splits.
  if (S.runOrder.length) {
    const sp = predictedSplit(s.i);
    if (sp == null) return false;               // no loaded run predicted this sample
    if (!f.splits.has(sp)) return false;
  }

  // Metric ranges read the sorted run when one is chosen, else any loaded run may
  // satisfy them (union) — the intuitive reading of "show me samples where MSE is high".
  const names = S.sort.run && S.runs[S.sort.run] ? [S.sort.run] : S.runOrder;
  if (!names.length) return true;
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
  return any;
}

function applyFilters() {
  const rows = S.samples.filter(passes);
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
    // "worst" means high for MSE/COM-distance but LOW for IoU; the direction button is
    // labelled semantically so this inversion never lands on the user.
    const desc = (m.worst === "high") === (dir === "worst");
    const tbl = S.runs[run].samples;
    rows.sort((a, b) => {
      const x = tbl[a.i], y = tbl[b.i];
      // A missing prediction, or a metric that is undefined for this sample (COM
      // distance on an empty box), sinks to the end in BOTH directions -- neither is
      // "best", and letting them float to the top would bury the real failures.
      const xv = x && x[metric] != null ? x[metric] : null;
      const yv = y && y[metric] != null ? y[metric] : null;
      if (xv == null && yv == null) return a.i - b.i;
      if (xv == null) return 1;
      if (yv == null) return -1;
      const d = (yv - xv) * (desc ? 1 : -1);
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

/* "which epoch am I looking at, out of how many the run trained for" -- the two are
   easy to confuse once the slider exists, so the header always shows both and calls out
   when you are looking at anything other than the newest predictions. */
/* The epoch number is a control: clicking it pins every column to this run's last epoch,
   which is how you line the table up on one run's endpoint. */
function epochLabel(run) {
  const r = S.runs[run];
  const eps = r.epochs || [];
  const last = eps.length ? eps[eps.length - 1] : r.epoch;
  const shown = epochFor(run);
  // ep <current>/<last>. Only the last is clickable -- it is the run's endpoint, and
  // jumping every column there is the useful move; the current epoch is already current.
  const cur = shown == null ? last : shown;
  return `ep <b${cur === last ? "" : ` class="scrub"`}>${cur}</b><span class="k">/</span>` +
    `<b class="goep" data-ep="${last}" title="Show every column at epoch ${last}">${last}</b>`;
}

function renderHeader() {
  const h = document.createElement("div");
  h.className = "hrow";
  h.style.height = "auto";
  h.innerHTML = `<div class="hcell idx sticky-1"></div>
    <div class="hcell gt sticky-2"><div class="hname" style="cursor:default">Ground truth</div>
      <div class="hmeta">sample · overhead + mask</div></div>`;

  for (const name of S.runOrder) {
    const r = S.runs[name], st = statsFor(name);
    const sorted = S.sort.run === name;
    const cell = document.createElement("div");
    cell.className = "hcell run" + (sorted ? " sorted" : S.sort.run ? " dim" : "");
    const warn = r.entry && r.entry.family === "unknown"
      ? `<span class="warnbadge" title="No eval split directories; dataset identity unconfirmed">?</span>` : "";
    const skipped = r.skipped_files.length
      ? ` · <span title="${r.skipped_files.join(", ")}">${r.skipped_files.length} file(s) skipped</span>` : "";
    // Read status from the latest poll, not the snapshot taken when the run was added --
    // a run that finishes or crashes while open must update its badge.
    const live = S.meta.runs.find((x) => x.name === name);
    const status = (live && live.status) || "unknown";
    cell.innerHTML = `
      <div class="htop">
        <div class="hname" title="${name} — click to cycle sort">${name}${warn}</div>
        <span class="hep">${epochLabel(name)}</span>
        <button class="hclose" title="Remove this run">&times;</button>
      </div>
      <div class="hmeta">${statusChip(status)} ${r.n}/${S.samples.length}${skipped}</div>
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

    cell.querySelector(".hclose").onclick = (e) => {
      e.stopPropagation();          // must not also cycle the column's sort
      removeRun(name);
    };
    cell.querySelector(".hname").onclick = () => {
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
  return `/api/gt_mask.png?sid=${sid}&bg=${S.view.background ? 1 : 0}&v=${S.renderVersion}`;
}

function buildRow() {
  const el = document.createElement("div");
  el.className = "row";
  el.innerHTML = `<div class="cell idx sticky-1"><span class="idxn"></span></div>
    <div class="cell gt sticky-2 gtcell">
      <div class="ctitle gt-head"></div>
      <img class="mask gtimg" loading="lazy" decoding="async" alt="">
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
  el.querySelector(".gt-head").innerHTML =
    `<span class="t1">Ground truth</span><span class="t2">${identity(s)}</span>`;

  // The ground-truth cell shows the same 20x40 target the runs predict, over the same
  // backdrop, so it is directly comparable with every prediction column. The speaker
  // view lives in the detail modal instead.
  const img = el.querySelector(".gtimg");
  img.className = "mask gtimg" + (S.view.background ? "" : " nobg");
  const src = gtMaskURL(s.i);
  if (img.getAttribute("src") !== src) img.setAttribute("src", src);
  img.dataset.run = ""; img.dataset.sid = s.i;

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
  el.querySelector(".gtcell").onclick = () => openModal(rank);

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

    // Title line 1 identifies the frame: which run, which epoch it is showing, and the
    // split THIS run put the sample in (dataloaders decide that independently, so the
    // same sample can be train in one run and eval in another). Runs also save on
    // different cadences, so at one slider position two columns can legitimately show
    // different epochs -- hence the epoch belongs per cell, not just in the header.
    const cellEp = epochFor(name);
    const shownEp = cellEp == null ? (S.runs[name].epochs || []).slice(-1)[0] : cellEp;
    const latest = cellEp == null;
    const split = e ? e.split : (S.runs[name].splitOf || {})[s.i];
    // Only the run name truncates. Epoch and split are always legible -- they say which
    // frame you are looking at, so losing them to an ellipsis would make the cell
    // ambiguous, whereas a shortened run name is still recognisable.
    // The grid is shown only when columns disagree on it. Both metrics are
    // grid-normalized so the numbers share a scale, but a coarser grid is systematically
    // easier -- a small cross-size gap is not evidence of a better model, so the reader
    // needs to see which size they are looking at.
    const gs = mixed ? `<span class="sp gsz" title="mask grid">${rh}x${rw}</span>` : "";
    head.innerHTML =
      `<span class="t1 run"><span class="rn" title="${name}">${name}</span>` +
      `<b class="ep${latest ? "" : " scrub"}">${shownEp ?? "–"}ep</b>${gs}` +
      `${split ? `<span class="sp" title="${split}">${shortSplit(split)}</span>` : ""}</span>` +
      `<span class="t2">${identity(s)}</span>`;

    if (!e) {
      chips.innerHTML = "";
      m.hidden = true;
      m.style.backgroundImage = "";
      if (!c._np) { c._np = document.createElement("div"); c._np.className = "nopred"; c._np.textContent = "no prediction"; c.appendChild(c._np); }
      c._np.hidden = false;
      return;
    }
    if (c._np) c._np.hidden = true;
    // Predicted centre of mass on its own line, then the three metrics on the next. All
    // four at full size do not fit 224px on one line, and shrinking them to fit made the
    // numbers hard to scan -- which is the one thing this column exists for. Splitting
    // keeps the metrics aligned across every run column at a readable size.
    chips.innerHTML =
      `<span class="tag com" title="predicted center of mass (row, col) in grid coords">${
        fmt(e.com[0], 1)}, ${fmt(e.com[1], 1)}</span><i class="brk"></i>` +
      METRICS.map((mm) => {
        const v = e[mm.key];
        return `<span class="tag"><span class="k">${mm.short}</span> <b>${
          v == null ? "–" : fmt(v, mm.key === "mse" ? 4 : 3)}</b></span>`;
      }).join("");
    m.hidden = false;
    // Match the canvas buffer to THIS run's grid. Rows are recycled across runs, so a
    // pooled cell can arrive still sized for a column of a different resolution.
    if (m.width !== rw || m.height !== rh) { m.width = rw; m.height = rh; }
    m.dataset.run = name; m.dataset.sid = s.i;
    m.onclick = () => openNeighbors(name, s.i);
    m.className = "mask predmask" + (S.view.background ? " bg" : " nobg");
    // The backdrop is a CSS background behind the canvas rather than baked into the
    // pixels: it is fetched once per sample and reused for every run and every epoch,
    // which is most of why scrubbing costs no bandwidth.
    const want = S.view.background ? `url(/api/backdrop/${s.i}.jpg)` : "";
    if (m.style.backgroundImage !== want) m.style.backgroundImage = want;
    paintCanvas(m, name, s.i, epochFor(name));

    // Stacked: the target gets its own box directly below the prediction, so the two are
    // adjacent instead of the ground truth being columns away in the leftmost cell.
    const tm = c.querySelector(".truthmask");
    tm.hidden = S.view.mode !== "stacked";
    if (!tm.hidden) {
      tm.className = "mask truthmask" + (S.view.background ? " bg" : " nobg");
      if (tm.style.backgroundImage !== want) tm.style.backgroundImage = want;
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
  $("#rowcount").innerHTML =
    `<b>${S.order.length}</b> of ${S.samples.length} samples` +
    ` · <b>${nRuns}</b> run${nRuns === 1 ? "" : "s"}`;
  $("#empty").hidden = S.order.length > 0;
  renderChips();
  renderScatter();
  renderSpeakers();
  renderEpochPanel();
  renderVisible();
}

/* ***** sidebar: chips ***** */

function chipRow(host, values, set, labelFn) {
  host.innerHTML = "";
  values.forEach(([v, n]) => {
    const b = document.createElement("button");
    b.className = "chip" + (set.has(v) ? "" : " off");
    b.innerHTML = `${labelFn(v)}<span class="n">${n}</span>`;
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
  const splits = new Set();
  S.runOrder.forEach((n) => S.runs[n].splits.forEach((s) => splits.add(s)));
  const counts = new Map();
  S.samples.forEach((s) => { const sp = predictedSplit(s.i); if (sp) counts.set(sp, (counts.get(sp) || 0) + 1); });
  chipRow($("#split-chips"), [...splits].sort().map((s) => [s, counts.get(s) || 0]), f.splits, (v) => v);
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
  for (const { v, lut, k } of layers) {
    for (let i = 0; i < n; i++) {
      let t, a;
      if (mode === "diff") {
        const d = Math.max(-1, Math.min(1, v[i] - truth[i]));
        const mag = Math.pow(Math.abs(d), gamma);
        t = 0.5 + 0.5 * Math.sign(d) * mag;
        a = mag;
      } else {
        t = Math.pow(Math.max(0, Math.min(1, v[i])), gamma);
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

function holdToRepeat(btn, step) {
  let raf = 0, timer = 0;
  const stop = () => {
    cancelAnimationFrame(raf); clearTimeout(timer); raf = timer = 0;
  };
  btn.addEventListener("pointerdown", (e) => {
    if (e.button !== 0 || btn.disabled) return;
    e.preventDefault();
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
  renderVisible();          // masks repaint immediately from local frame data
  fetchEpochMetrics();      // metrics/sort catch up when the server answers
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
  metricsTimer = setTimeout(async () => {
    const jobs = S.runOrder.map(async (name) => {
      const ep = epochFor(name);
      const cur = S.runs[name];
      if (cur.metricsEpoch === ep) return false;
      const q = ep == null ? "" : `?epoch=${ep}`;
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
   Runs keep training while viz2 is open. Poll the (cheap, server-throttled) run list;
   when a loaded run's epoch advances, refetch its metrics and re-render its masks. */

const POLL_MS = 15000;

async function poll() {
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
  if (added.length && !$("#runpicker").hidden) openPicker();   // keep an open picker live
  if (!advanced.length && statusChanged) refresh();
}

/* ***** run picker *****
   Loaded runs are their own columns, so there is no separate list of them: each column
   header carries its own close button. */

function openPicker() {
  const list = $("#run-list"), q = $("#run-search");
  const draw = () => {
    const term = q.value.trim().toLowerCase();
    list.innerHTML = "";
    S.meta.runs
      .filter((r) => !term || r.name.toLowerCase().includes(term))
      .forEach((r) => {
        const added = !!S.runs[r.name];
        const d = document.createElement("div");
        d.className = "ritem" + (r.compatible ? (added ? " added" : "") : " bad");
        d.innerHTML = `<div><div class="nm">${r.name}</div>
          <div class="sub">${statusChip(r.status)} ${r.compatible
            ? `ep ${r.epoch ?? "?"} · ${r.eval_splits.length} eval splits` : r.reason}</div></div>
          <span class="badge">${added ? "added" : r.compatible ? "add" : "unavailable"}</span>`;
        if (r.compatible && !added) d.onclick = async () => { $("#runpicker").hidden = true; await addRun(r.name); };
        list.appendChild(d);
      });
  };
  q.value = ""; q.oninput = draw; draw();
  $("#runpicker").hidden = false;
  q.focus();
}

/* ***** hover tooltip *****
   Grid values are fetched on first hover for a cell and memoized; never prefetched. */

const valueCache = new Map();
async function valuesFor(run, sid, mode) {
  const k = `${run}|${sid}|${mode}`;
  if (!valueCache.has(k))
    valueCache.set(k, api(`/api/values?sid=${sid}&run=${encodeURIComponent(run)}&mode=${mode}`));
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
    const [mh, mw] = img.dataset.run ? runShape(img.dataset.run) : [S.lut.h, S.lut.w];
    const col = Math.floor(((ev.clientX - b.left) / b.width) * mw);
    const row = Math.floor(((ev.clientY - b.top) / b.height) * mh);
    if (row < 0 || row >= mh || col < 0 || col >= mw) { tip.hidden = true; return; }
    // Ground truth has no prediction to diff against, so it always reports raw values.
    const mode = img.dataset.run ? S.view.mode : "pred";
    const key = `${img.dataset.run}|${img.dataset.sid}|${mode}`;
    if (cur !== key) { cur = key; tip._v = await valuesFor(img.dataset.run, img.dataset.sid, mode); }
    const v = tip._v?.v?.[row]?.[col];
    const t = tip._v?.t?.[row]?.[col];
    tip.hidden = false;
    tip.textContent = `[${row},${col}] ` + (v == null ? "–" : v.toFixed(3)) +
      (t == null ? "" : ` pred · ${t.toFixed(3)} true`);
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
  const d = await api(`/api/detail/${s.i}`);
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
      <img src="/api/gt_mask.png?sid=${x.i}&bg=1&v=${S.renderVersion}" loading="lazy" alt="">
      <div class="nbmeta">
        <div class="nbtop"><b>${+x.sample_id}</b> <span class="tag">pos ${+x.output_id}</span></div>
        <div class="nbsub">d ${fmt(x.distance, 3)} · com ${dc(x.com[0], x.com[1])}</div>
        <div class="nbsub">${shortLayout(x.layout)} · ${x.n_objects} obj</div>
      </div>
    </li>`).join("");

  const v = `v=${S.renderVersion}`;
  const ep = S.runs[run] ? `&ep=${S.runs[run].epoch}` : "";
  const e = S.runs[run] && S.runs[run].samples[sid];

  $("#m-title").innerHTML = `Sample ${+d.sample_id} — <span class="mrun">${run}</span>`;
  $("#m-body").innerHTML = `
    <div class="msec">
      <h3>This prediction</h3>
      <div class="nbviews">
        <figure><img src="/api/mask.png?run=${encodeURIComponent(run)}&sid=${sid}&mode=pred&bg=1&${v}${ep}" alt="">
          <figcaption>predicted</figcaption></figure>
        <figure><img src="/api/gt_mask.png?sid=${sid}&bg=1&${v}" alt="">
          <figcaption>ground truth</figcaption></figure>
        <figure><img src="/api/mask.png?run=${encodeURIComponent(run)}&sid=${sid}&mode=diff&bg=1&${v}${ep}" alt="">
          <figcaption>difference</figcaption></figure>
      </div>
      <div class="nbhead">
        <div><span class="k">predicted com</span><b>${dc(d.pred_com[0], d.pred_com[1])}</b></div>
        <div><span class="k">ground truth</span><b>${dc(d.gt_com[0], d.gt_com[1])}</b></div>
        <div><span class="k">offset</span><b>${fmt(off, 2)} cells</b></div>
        ${e ? `<div><span class="k">mse</span><b>${fmt(e.mse, 4)}</b></div>
               <div><span class="k">soft iou</span><b>${fmt(e.iou, 3)}</b></div>
               <div><span class="k">com dist</span><b>${e.comdist == null ? "–" : fmt(e.comdist, 3)}</b></div>` : ""}
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
    try { localStorage.setItem("viz2.sidebar", open ? "1" : "0"); } catch (_) {}
  };
  $("#sidebar-hide").onclick = () => setSidebar(false);
  $("#sidebar-show").onclick = () => setSidebar(true);
  try { if (localStorage.getItem("viz2.sidebar") === "0") setSidebar(false); } catch (_) {}

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
  $("#bg-toggle").onchange = (e) => { S.view.background = e.target.checked; renderVisible(); };
  // Interpolate instead of showing hard cell edges. Purely visual -- metrics are always
  // computed at each run's native grid -- but it is what makes a coarse column and a
  // fine one comparable by eye when the table mixes resolutions.
  $("#smooth-toggle").onchange = (e) =>
    document.body.classList.toggle("smooth", e.target.checked);

  $("#epoch-slider").oninput = (e) => setEpochIdx(+e.target.value);
  $("#epoch-play").onclick = togglePlay;
  holdToRepeat($("#epoch-prev"), () => stepEpoch(-1));
  holdToRepeat($("#epoch-next"), () => stepEpoch(1));
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

  $("#add-run").onclick = openPicker;
  document.querySelectorAll("[data-close]").forEach((b) => {
    b.onclick = () => { b.closest(".overlay").hidden = true; };
  });
  document.querySelectorAll(".overlay").forEach((o) => {
    o.onclick = (e) => { if (e.target === o) o.hidden = true; };
  });
  $("#m-prev").onclick = () => stepModal(-1);
  $("#m-next").onclick = () => stepModal(1);

  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape") document.querySelectorAll(".overlay").forEach((o) => (o.hidden = true));
    if (!$("#modal").hidden) {
      if (e.key === "ArrowLeft") { e.preventDefault(); stepModal(-1); }
      if (e.key === "ArrowRight") { e.preventDefault(); stepModal(1); }
      return;
    }
    // Bare "f" toggles the filter panel -- but not while typing in the id searches, and
    // not when it is a browser/OS chord.
    const typing = /^(INPUT|TEXTAREA|SELECT)$/.test(e.target.tagName);
    if (e.key === "f" && !typing && !e.ctrlKey && !e.metaKey && !e.altKey) {
      e.preventDefault();
      setSidebar(document.body.classList.contains("nosidebar"));
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
