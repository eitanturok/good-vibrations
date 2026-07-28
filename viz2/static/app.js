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
const ROW_H = 182;   // must equal --row-h in style.css; the virtualizer assumes it
const METRICS = [
  { key: "mse",     label: "MSE",      short: "mse", worst: "high" },
  { key: "iou",     label: "Soft IoU", short: "iou", worst: "low"  },
  { key: "comdist", label: "COM dist", short: "cd",  worst: "high" },
];

const S = {
  samples: [], runs: {}, order: [], runOrder: [], meta: null,
  filters: {
    splits: new Set(), speakers: new Set(), layouts: new Set(), nObjects: new Set(),
    positions: null, ranges: {},
  },
  sort: { run: null, metric: "comdist", dir: "worst" },
  view: { mode: "pred", background: true },
  domain: {}, positions: [], activePos: -1, modalRow: -1,
  renderVersion: 0,   // from /api/runs; part of image URLs to defeat immutable caching
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
  const [runs, samples] = await Promise.all([api("/api/runs"), api("/api/samples")]);
  S.meta = runs;
  S.renderVersion = runs.render_version ?? 0;
  S.samples = samples.samples;
  buildPositions();
  initFilters();
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

async function addRun(name, reload = false) {
  if (S.runs[name] && !reload) return;
  const d = await api(`/api/run/${encodeURIComponent(name)}${reload ? "?reload=1" : ""}`);
  d.entry = S.meta.runs.find((r) => r.name === name);
  if (reload) {
    // Refresh in place: keep column order and, crucially, leave the filters alone. A new
    // epoch must not silently re-check splits the user turned off or move their sliders.
    const prev = S.runs[name];
    d.splits = new Set(Object.values(d.samples).map((x) => x.split));
    for (const sp of d.splits) {
      if (!prev || !prev.splits.has(sp)) S.filters.splits.add(sp);   // genuinely new only
    }
    S.runs[name] = d;
    recomputeDomains({ preserve: true });
    return;
  }
  S.runs[name] = d;
  S.runOrder.push(name);
  d.splits = new Set(Object.values(d.samples).map((x) => x.split));
  for (const sp of d.splits) S.filters.splits.add(sp);
  recomputeDomains();
  refresh();
}

function removeRun(name) {
  delete S.runs[name];
  S.runOrder = S.runOrder.filter((n) => n !== name);
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
    if (s.speaker != null) f.speakers.add(s.speaker);
    if (s.layout) f.layouts.add(s.layout);
    if (s.n_objects != null) f.nObjects.add(s.n_objects);
  });
}

function uniq(get) {
  const m = new Map();
  S.samples.forEach((s) => { const v = get(s); if (v != null) m.set(v, (m.get(v) || 0) + 1); });
  return [...m.entries()].sort((a, b) => (a[0] > b[0] ? 1 : -1));
}

/* A sample is in the table if any loaded run predicted it. */
function predictedSplit(sampleIdx) {
  for (const n of S.runOrder) {
    const e = S.runs[n].samples[sampleIdx];
    if (e) return e.split;
  }
  return null;
}

function passes(s) {
  const f = S.filters;
  if (!f.speakers.has(s.speaker)) return false;
  if (!f.layouts.has(s.layout)) return false;
  if (!f.nObjects.has(s.n_objects)) return false;
  if (f.positions && !(s.pos >= 0 && f.positions.has(s.pos))) return false;

  const sp = predictedSplit(s.i);
  if (sp == null) return false;                 // no run predicted this sample
  if (!f.splits.has(sp)) return false;

  // Metric ranges read the sorted run when one is chosen, else any loaded run may
  // satisfy them (union) — the intuitive reading of "show me samples where MSE is high".
  const names = S.sort.run && S.runs[S.sort.run] ? [S.sort.run] : S.runOrder;
  if (!names.length) return true;
  let any = false;
  for (const n of names) {
    const e = S.runs[n].samples[s.i];
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
  return any;
}

function applyFilters() {
  const rows = S.samples.filter(passes);
  const { run, metric, dir } = S.sort;
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
        <button class="hclose" title="Remove this run">&times;</button>
      </div>
      <div class="hmeta">${statusChip(status)} ep ${r.epoch} · ${r.n}/${S.samples.length}${skipped}</div>
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

/* Mask images are cached `immutable`, so the render version is part of the URL: bumping
   config.RENDER_VERSION is what lets a changed rendering reach browsers that already
   cached the old one. */
function maskURL(run, sid) {
  const { mode, background } = S.view;
  // The run's epoch is part of the URL: a still-training run writes new predictions, and
  // without it the browser would keep serving the epoch it first cached.
  const ep = S.runs[run] ? S.runs[run].epoch : 0;
  return `/api/mask.png?run=${encodeURIComponent(run)}&sid=${sid}&mode=${mode}` +
    `&bg=${background ? 1 : 0}&v=${S.renderVersion}&ep=${ep}`;
}

function gtMaskURL(sid) {
  return `/api/gt_mask.png?sid=${sid}&bg=${S.view.background ? 1 : 0}&v=${S.renderVersion}`;
}

function buildRow() {
  const el = document.createElement("div");
  el.className = "row";
  el.innerHTML = `<div class="cell idx sticky-1"><span class="idxn"></span></div>
    <div class="cell gt sticky-2 gtcell">
      <div class="tags gt-head"></div>
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
    c.innerHTML = `<div class="tags mstats"></div><img class="mask" loading="lazy" decoding="async" alt=""><div class="tags subtags"></div>`;
    el.appendChild(c);
    el._runCells.push(c);
  }
}

function paintRow(el, rank) {
  const s = S.order[rank];
  el.style.transform = `translateY(${rank * ROW_H}px)`;
  el.querySelector(".idxn").textContent = rank + 1;
  // Identity line: which sample, where it was captured, which speaker played. The 8
  // samples sharing a position differ only by speaker, so all three belong together.
  const pos = s.output_id == null ? "–" : +s.output_id;
  el.querySelector(".gt-head").innerHTML =
    `<span class="tag strong">sample ${+s.sample_id}</span>` +
    `<span class="tag">pos ${pos}</span>` +
    `<span class="tag">spk ${s.speaker}</span>`;

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
  el.querySelector(".gt-foot").innerHTML =
    `<span class="tag">com ${com}</span>` +
    `<span class="tag" title="${s.layout}">${shortLayout(s.layout)}</span>` +
    `<span class="tag">${s.n_objects} obj</span>`;
  el.querySelector(".gtcell").onclick = () => openModal(rank);

  syncRowCells(el);
  S.runOrder.forEach((name, k) => {
    const c = el._runCells[k];
    const e = S.runs[name].samples[s.i];
    const stats = c.querySelector(".mstats");
    const m = c.querySelector(".mask");
    const com = c.querySelector(".subtags");
    if (!e) {
      stats.innerHTML = "";
      com.innerHTML = "";
      m.hidden = true;
      if (!c._np) { c._np = document.createElement("div"); c._np.className = "nopred"; c._np.textContent = "no prediction"; c.appendChild(c._np); }
      c._np.hidden = false;
      return;
    }
    if (c._np) c._np.hidden = true;
    m.hidden = false;
    stats.innerHTML = METRICS.map((mm) => {
      const v = e[mm.key];
      return `<span class="tag"><span class="k">${mm.short}</span> <b>${
        v == null ? "–" : fmt(v, mm.key === "mse" ? 4 : 3)}</b></span>`;
    }).join("");
    // The split lives here because each run's dataloaders decide it independently -- the
    // same sample can be train in one run and an eval split in another.
    com.innerHTML = `<span class="tag">com ${fmt(e.com[0], 1)}, ${fmt(e.com[1], 1)}</span>` +
      `<span class="tag split" title="${e.split}">${shortSplit(e.split)}</span>`;
    m.className = "mask predmask" + (S.view.background ? "" : " nobg");
    const u = maskURL(name, s.i);
    if (m.getAttribute("src") !== u) m.setAttribute("src", u);
    m.dataset.run = name; m.dataset.sid = s.i;
    m.onclick = () => openNeighbors(name, s.i);
  });
}

const pool = [];
function renderVisible() {
  const sc = $("#scroller");
  const hdr = sc.querySelector(".hrow");
  const hh = hdr ? hdr.offsetHeight : 0;
  // Row k occupies [hh + k*ROW_H, hh + (k+1)*ROW_H) in scroll space; the visible band
  // starts under the sticky header, so subtract its height before dividing.
  const top = Math.max(0, sc.scrollTop - hh);
  const first = Math.max(0, Math.floor(top / ROW_H) - 3);
  const last = Math.min(S.order.length, Math.ceil((top + sc.clientHeight) / ROW_H) + 3);
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
  $("#spacer").style.height = `${hh + S.order.length * ROW_H}px`;
  const nRuns = S.runOrder.length;
  $("#rowcount").innerHTML =
    `<b>${S.order.length}</b> of ${S.samples.length} samples` +
    ` · <b>${nRuns}</b> run${nRuns === 1 ? "" : "s"}`;
  $("#empty").hidden = S.order.length > 0;
  renderChips();
  renderScatter();
  renderSpeakers();
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

function renderChips() {
  const f = S.filters;
  const splits = new Set();
  S.runOrder.forEach((n) => S.runs[n].splits.forEach((s) => splits.add(s)));
  const counts = new Map();
  S.samples.forEach((s) => { const sp = predictedSplit(s.i); if (sp) counts.set(sp, (counts.get(sp) || 0) + 1); });
  chipRow($("#split-chips"), [...splits].sort().map((s) => [s, counts.get(s) || 0]), f.splits, (v) => v);
  chipRow($("#layout-chips"), uniq((s) => s.layout), f.layouts, (v) => v);
  chipRow($("#nobj-chips"), uniq((s) => s.n_objects), f.nObjects, (v) => `${v}`);
  $("#split-count").textContent = `${f.splits.size}/${splits.size}`;
  $("#layout-count").textContent = `${f.layouts.size}/${uniq((s) => s.layout).length}`;
  $("#nobj-count").textContent = `${f.nObjects.size}/${uniq((s) => s.n_objects).length}`;
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
  const W = 236, H = 148, OFF = 16, R = 11, M = OFF + R + 1;
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
    // Mask URLs are cached immutable, so a new epoch needs a new URL to be fetched.
    S.epochTag = (S.epochTag || 0) + 1;
    // New metrics can reorder a sorted table. Keep whichever sample is at the top of the
    // viewport in view, so rows don't slide out from under the user mid-read.
    const sc = $("#scroller");
    const hdr = sc.querySelector(".hrow");
    const anchorRank = Math.floor(Math.max(0, sc.scrollTop - (hdr ? hdr.offsetHeight : 0)) / ROW_H);
    const anchor = S.order[anchorRank];
    refresh();
    if (anchor) {
      const next = S.order.indexOf(anchor);
      if (next >= 0 && next !== anchorRank) {
        sc.scrollTop += (next - anchorRank) * ROW_H;
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
    valueCache.set(k, api(`/api/values?sid=${sid}&run=${encodeURIComponent(run)}&mode=${mode}`).then((d) => d.v));
  return valueCache.get(k);
}

function bindTooltip() {
  const tip = $("#tooltip");
  let cur = null;
  $("#scroller").addEventListener("mousemove", async (ev) => {
    // dataset.run is "" on the ground-truth cell, which still has values to show.
    const img = ev.target.closest?.("img.mask");
    if (!img || img.dataset.sid === undefined) { tip.hidden = true; cur = null; return; }
    const b = img.getBoundingClientRect();
    const col = Math.floor(((ev.clientX - b.left) / b.width) * 40);
    const row = Math.floor(((ev.clientY - b.top) / b.height) * 20);
    if (row < 0 || row > 19 || col < 0 || col > 39) { tip.hidden = true; return; }
    // Ground truth has no prediction to diff against, so it always reports raw values.
    const mode = img.dataset.run ? S.view.mode : "pred";
    const key = `${img.dataset.run}|${img.dataset.sid}|${mode}`;
    if (cur !== key) { cur = key; tip._v = await valuesFor(img.dataset.run, img.dataset.sid, mode); }
    const v = tip._v?.[row]?.[col];
    tip.hidden = false;
    tip.textContent = `[${row},${col}] ${v == null ? "–" : v.toFixed(3)}`;
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
          <dt>on 20×40 grid</dt><dd>${d.com_gt_grid ? `${fmt(d.com_gt_grid[0], 2)}, ${fmt(d.com_gt_grid[1], 2)}` : "–"}</dd>
        </dl>
        <p class="note">Per-object and average are full-resolution image coordinates (row, col);
          the grid value is in 20×40 target space, as used by the table metrics.</p>
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
  window.addEventListener("resize", renderVisible);

  $("#mode-seg").querySelectorAll("button").forEach((b) => {
    b.onclick = () => {
      S.view.mode = b.dataset.mode;
      $("#mode-seg").querySelectorAll("button").forEach((x) => x.classList.toggle("on", x === b));
      renderLegend();
      renderVisible();
    };
  });
  $("#bg-toggle").onchange = (e) => { S.view.background = e.target.checked; renderVisible(); };

  document.querySelectorAll("[data-all]").forEach((b) => {
    b.onclick = () => {
      const k = b.dataset.all;
      if (k === "speakers") S.filters.speakers = new Set([1, 2, 3, 4, 5, 6, 7, 8]);
      if (k === "layouts") S.filters.layouts = new Set(uniq((s) => s.layout).map((x) => x[0]));
      if (k === "nObjects") S.filters.nObjects = new Set(uniq((s) => s.n_objects).map((x) => x[0]));
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
    }
  });

  bindTooltip();
  renderLegend();
  setInterval(poll, POLL_MS);   // pick up new runs and new epochs of training ones
}

function renderLegend() {
  const diff = S.view.mode === "diff";
  $("#legend").innerHTML =
    `<img src="/api/colorbar/${diff ? "diff" : "pred"}.png" alt="">
     <div class="lbl">${diff
       ? `<span>−1 missed</span><span>0</span><span>+1 excess</span>`
       : `<span>0</span><span>mask value</span><span>1</span>`}</div>`;
}

boot().catch((e) => { $("#subtitle").textContent = "failed to load"; console.error(e); });
