/* good-vibrations dashboard */
"use strict";

// ===== palette (validated reference set, dark mode) =====
const CAT = ["#3987e5", "#199e70", "#c98500", "#008300", "#9085e9", "#e66767", "#d55181", "#d95926"];
const SPK_COLOR = i => CAT[(i - 1) % 8];          // speakers 1..8 -> fixed slots
const GT_COLOR = "#0ca30c", PRED_COLOR = "#d03b3b";
const INK = "#c3c2b7", MUTED = "#898781", GRID = "#2c2c2a", BASE = "#383835", SURFACE = "#1a1a19";
const GOOD = "#0ca30c", WARN = "#fab219", BAD = "#d03b3b";
// physical speaker placement: (x_frac, y_frac), y=0 at bottom (src3/overhead_pipeline.py)
const SPEAKER_POSITION = { 1: [1, 0], 2: [1, 0.7], 3: [0.8, 1], 4: [0.6, 1], 5: [0.4, 1], 6: [0.2, 1], 7: [0, 0.7], 8: [0, 0] };
const SPLITS = ["train", "unseen_pos", "unseen_pos_speaker", "unseen_layout"];
const run_cat = ["#3987e5", "#c98500", "#9085e9", "#d55181", "#d95926", "#199e70", "#e66767", "#008300"];
const runColor = name => run_cat[S.runs.indexOf(name) % 8];

// ===== state =====
const S = {
  man: null,
  filters: {},                     // facet -> Set of active values; live-filters what's shown from the pool
  numFilters: {                    // range filters, persistent & reversible like S.filters, checked by numFilterPasses
    mse: null,                       // {lo, hi} | null — checked against any loaded run's record for the sample
    com_dist: null,                  // {lo, hi} | null — same, run-agnostic ("passes if any loaded run passes")
    com_row: null,                   // {lo, hi} | null — on the sample itself, not run-scoped
    com_col: null,                   // {lo, hi} | null
  },
  pool: new Map(),                 // sample_id -> sample — drives the table only
  bench: new Map(),                // sample_id -> sample — drives the box + fft viewers only, added to explicitly
  cursor: null,                    // output_id cursor on the position map (keyboard nav + mouse hover)
  lasers: new Set([...Array(100).keys()]),
  dirs: "xy", logy: true, norm: true, avgSpeaker: false, diffEmpty: false, emptyBaseline: null,
  colorBy: "speaker",
  runs: [],                        // active run names, in order
  runData: {},                     // run -> payload {epoch, epochs, samples, aggregates}
  runMasks: {},                    // run -> {sample_id -> mask}
  gtMasks: {},                     // sample_id -> mask
  fftCache: new Map(),             // `${key}|${id}` -> curve
  freqs: null,
  sort: { key: "sample_id", dir: 1 },
  contours: false,
  version: null,
  fftZoom: null,                   // {x:[lo,hi], y:[lo,hi]} preserved across redraws
};

const $ = s => document.querySelector(s);
const jget = async url => { const r = await fetch(url); if (!r.ok) throw new Error(url + " -> " + r.status); return r.json(); };
const debounce = (fn, ms) => { let t; return (...a) => { clearTimeout(t); t = setTimeout(() => fn(...a), ms); }; };
const fmt = (v, d = 4) => v == null ? "—" : (+v).toFixed(d);

let GROUPS = new Map();   // output_id -> samples
let EMPTY_GROUPS = [];    // [{output_id, sample_ids}]
const SAMPLE = new Map(); // sample_id -> sample

function candidatePasses(s, skipFacet = null) {
  for (const [f, set] of Object.entries(S.filters)) {
    if (f === skipFacet || !set) continue;
    if (f === "split") {
      // splits are assigned per training run, so check every loaded run's record for this sample
      for (const run of S.runs) {
        const rec = S.runData[run] && S.runData[run].samples[s.sample_id];
        if (rec && !set.has(rec.split)) return false;
      }
      continue;
    }
    if (!set.has(s[f])) return false;
  }
  return numFilterPasses(s);
}
// checks S.numFilters ranges against a sample. mse/com_dist live on run records, not the sample
// itself, so they're checked as "passes for at least one loaded run" -- with no run loaded (or no
// runs matching) a set mse/com_dist filter rejects everything, which is the honest behavior since
// there's no data to filter on yet, rather than silently ignoring the filter.
function numFilterPasses(s) {
  const { mse, com_dist, com_row, com_col } = S.numFilters;
  const inRange = (v, r) => v != null && v >= r.lo && v <= r.hi;
  if (com_row && !inRange(s.com_row, com_row)) return false;
  if (com_col && !inRange(s.com_col, com_col)) return false;
  if (mse || com_dist) {
    const runs = S.runs.filter(r => S.runData[r]);
    const passesAnyRun = runs.some(r => {
      const rec = S.runData[r].samples[s.sample_id];
      if (!rec) return false;
      if (mse && !inRange(rec.mse, mse)) return false;
      if (com_dist && !inRange(rec.com_dist, com_dist)) return false;
      return true;
    });
    if (!passesAnyRun) return false;
  }
  return true;
}
function matchingCandidates() {
  return S.man.samples.filter(s => candidatePasses(s));
}
function colorFor(s) {
  if (S.colorBy === "speaker") return SPK_COLOR(s.speaker);
  if (S.colorBy === "layout") return CAT[S.man.facets.layout.indexOf(s.layout) % 8];
  if (S.colorBy === "output_id") {
    const h = (s.com_col / S.man.out_w) * 300, l = 42 + (s.com_row / S.man.out_h) * 30;
    return `hsl(${h.toFixed(0)},62%,${l.toFixed(0)}%)`;
  }
  if (S.colorBy === "split") {
    const rd = S.runData[S.runs[0]], rec = rd && rd.samples[s.sample_id];
    return rec ? CAT[SPLITS.indexOf(rec.split) % 8] : MUTED;
  }
  return CAT[0];
}
const colorKey = s => S.colorBy === "speaker" ? `spk ${s.speaker}`
  : S.colorBy === "layout" ? s.layout
  : S.colorBy === "output_id" ? `pos ${s.output_id}`
  : (S.runData[S.runs[0]]?.samples[s.sample_id]?.split || "?");
const label = s => `#${s.sample_id} · p${(+s.output_id).toString()} · spk${s.speaker}`;
const runBadgeColor = v => v == null ? MUTED : v < 0.05 ? GOOD : v < 0.15 ? WARN : BAD;

// ===== plotly base =====
const LAYOUT = extra => Object.assign({
  paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)",
  font: { color: MUTED, size: 11, family: "system-ui, sans-serif" },
  margin: { l: 44, r: 10, t: 8, b: 34 },
  showlegend: true,
  legend: { orientation: "h", y: 1.02, yanchor: "bottom", font: { color: INK } },
  hoverlabel: { bgcolor: "#232322", bordercolor: GRID, font: { color: INK, size: 11 } },
}, extra);
const AXIS = extra => Object.assign({ gridcolor: GRID, zerolinecolor: BASE, linecolor: BASE, ticks: "" }, extra);
const CONFIG = { displayModeBar: false, responsive: true };
const emptyLayout = msg => LAYOUT({
  xaxis: { visible: false }, yaxis: { visible: false }, showlegend: false,
  annotations: [{ text: msg, showarrow: false, font: { color: MUTED, size: 13 } }],
});

// ===== boot =====
let _bootedOnce = false;
async function boot() {
  S.man = await jget("/api/manifest");
  S.version = S.man.version;
  GROUPS = new Map();
  for (const s of S.man.samples) { if (!GROUPS.has(s.output_id)) GROUPS.set(s.output_id, []); GROUPS.get(s.output_id).push(s); }
  EMPTY_GROUPS = S.man.empty_box_groups || [];
  SAMPLE.clear();
  for (const s of S.man.samples) SAMPLE.set(s.sample_id, s);
  for (const f of Object.keys(S.man.facets))
    if (!S.filters[f]) S.filters[f] = new Set(S.man.facets[f]);
  // drop pool members whose sample no longer exists (data changed underneath us)
  for (const id of [...S.pool.keys()]) if (!SAMPLE.has(id)) S.pool.delete(id);
  buildLaserGrid(); buildRunSelect(); buildEmptySelect(); renderRunChips();

  // first load only: default to the whole dataset, so there's something useful on screen
  // without manual setup. Runs are still opt-in via the run-add dropdown. Later re-boots
  // (triggered by the live-update poll picking up a data change) must not repopulate on top
  // of the user's own pool.
  if (!_bootedOnce) {
    _bootedOnce = true;
    for (const s of S.man.samples) S.pool.set(s.sample_id, s);
  }

  renderAll();
}

function renderAll() {
  renderFacets(); renderPosmap(); renderPoolCount(); renderBenchCount(); renderBenchStrip(); renderEmptyChips();
  updateFFT(); updateBox(); renderSamplesView();
}

// ===== pool =====
function addToPool(samples) {
  for (const s of samples) S.pool.set(s.sample_id, s);
  renderAll();
}
function removeFromPool(ids) {
  for (const id of ids) S.pool.delete(id);
  renderAll();
}
function clearPool() { S.pool.clear(); renderAll(); }
function removeFilteredOut() {
  for (const [id, s] of S.pool) if (!candidatePasses(s)) S.pool.delete(id);
  renderAll();
}
// facet chips live-filter the table's view of the pool: toggling a chip off immediately hides
// matching pooled samples from the table; toggling it back on brings them right back, since pool
// membership itself (S.pool) is never touched here -- only what's shown. This applies to every
// facet, not just split; "remove filtered-out" is the separate, deliberate, non-reversible action
// for when you actually want to drop samples from the pool for good. The table's filters do NOT
// affect the bench (box/fft) pool below -- that's a separate, explicitly-curated selection.
function poolSamples() { return [...S.pool.values()].filter(s => candidatePasses(s)); }

function renderPoolCount() {
  const shown = poolSamples().length;
  $("#pool-count").textContent = shown === S.pool.size
    ? `${S.pool.size} sample${S.pool.size === 1 ? "" : "s"}`
    : `${shown} of ${S.pool.size} samples shown`;
}
$("#pool-clear").onclick = clearPool;

// ===== bench (box + fft viewer selection, independent of the table's pool/filters) =====
function addToBench(samples) { for (const s of samples) S.bench.set(s.sample_id, s); renderAll(); }
function removeFromBench(ids) { for (const id of ids) S.bench.delete(id); renderAll(); }
function clearBench() { S.bench.clear(); renderAll(); }
function benchSamples() { return [...S.bench.values()]; }
function renderBenchCount() { $("#bench-count").textContent = `${S.bench.size} sample${S.bench.size === 1 ? "" : "s"}`; }
$("#bench-clear").onclick = clearBench;
$("#bench-add-filtered").onclick = () => addToBench(poolSamples());

// simple thumbnail-card strip: this is the single, unambiguous place to see exactly what's
// currently feeding the box + fft viewers below, so it's never a mystery why those plots show
// what they show. Kept deliberately plain (id + photo + remove) -- no metrics, no masks, no
// per-run detail -- richer inspection stays in the existing per-sample modal (click a table thumb).
function renderBenchStrip() {
  const strip = $("#bench-strip");
  strip.innerHTML = benchSamples().map(s => `
    <div class="bench-card" data-id="${s.sample_id}">
      <span class="bench-card-x" title="remove from workbench">✕</span>
      <img loading="lazy" src="/media/${s.sample_id}/thumb" alt="">
      <span class="bench-card-id">#${s.sample_id} · p${+s.output_id}</span>
    </div>`).join("");
  strip.querySelectorAll(".bench-card-x").forEach(x => x.onclick = () =>
    removeFromBench([+x.closest(".bench-card").dataset.id]));
}
$("#pool-remove-filtered").onclick = removeFilteredOut;

// ===== facet chips ("add matching" candidate narrowing only) =====
function renderFacets() {
  for (const holder of document.querySelectorAll(".chips[data-facet]")) {
    const f = holder.dataset.facet;
    if (f === "split") continue; // handled separately below
    if (!S.man.facets[f]) continue;
    holder.innerHTML = "";
    for (const v of S.man.facets[f]) {
      const alive = S.man.samples.some(s => s[f] === v && candidatePasses(s, f));
      const n = S.man.samples.filter(s => s[f] === v && candidatePasses(s, f)).length;
      const chip = document.createElement("span");
      chip.className = "chip" + (S.filters[f].has(v) ? " active" : "") + (alive ? "" : " faded");
      chip.innerHTML = `${v} <span class="count">${n}</span>`;
      chip.onclick = () => toggleFilter(f, v);
      holder.appendChild(chip);
    }
  }
  renderSplitFacet();
  $("#match-count").textContent = matchingCandidates().length;
  const activeCount = Object.entries(S.filters).filter(([f, set]) => set.size < (S.man.facets[f] || []).length).length;
  $("#filters-summary").textContent = activeCount ? `${activeCount} facet${activeCount === 1 ? "" : "s"} narrowed` : "";
}
function renderSplitFacet() {
  const runsLoaded = S.runs.some(r => S.runData[r]);
  $("#split-facet-label").hidden = !runsLoaded;
  const holder = document.querySelector('.chips[data-facet="split"]');
  holder.hidden = !runsLoaded;
  if (!runsLoaded) return;
  if (!S.filters.split) S.filters.split = new Set(SPLITS);
  holder.innerHTML = "";
  for (const v of SPLITS) {
    const chip = document.createElement("span");
    chip.className = "chip" + (S.filters.split.has(v) ? " active" : "");
    chip.textContent = v.replace("unseen_", "u-");
    chip.onclick = () => toggleFilter("split", v);
    holder.appendChild(chip);
  }
}
function toggleFilter(f, v) {
  S.filters[f].has(v) ? S.filters[f].delete(v) : S.filters[f].add(v);
  renderAll();
}
$("#add-matching").onclick = () => addToPool(matchingCandidates());

// ===== numeric range filters (S.numFilters) =====
const applyRangeFilters = debounce(renderAll, 200);
document.querySelectorAll("#range-filters .range-row").forEach(row => {
  const field = row.dataset.field, loEl = row.querySelector(".range-lo"), hiEl = row.querySelector(".range-hi");
  const sync = () => {
    const lo = loEl.value === "" ? -Infinity : +loEl.value, hi = hiEl.value === "" ? Infinity : +hiEl.value;
    S.numFilters[field] = (loEl.value === "" && hiEl.value === "") ? null : { lo, hi };
    applyRangeFilters();
  };
  loEl.addEventListener("input", sync);
  hiEl.addEventListener("input", sync);
});

// ===== search / add-by-id box =====
$("#sample-add").addEventListener("keydown", e => {
  if (e.key !== "Enter") return;
  const input = e.target;
  const found = parseAddInput(input.value);
  if (!found.length) { input.classList.add("bad"); return; }
  input.classList.remove("bad");
  addToPool(found);
  input.value = "";
});
// numeric comparator token, e.g. "mse<0.05", "com_dist>=0.02", "com_row:10-20"
const NUM_TOKEN_RE = /^(mse|com_dist|com_row|com_col)(<=|>=|<|>|:)(-?\d+\.?\d*)(?:-(-?\d+\.?\d*))?$/;

// resolves one search-box query (comma-separated tokens) to a concrete, one-shot list of samples
// to add now -- NOT a persistent filter (that's S.numFilters, set via the sidebar range inputs).
// "run:" sets scope for any mse/com_dist tokens later in the SAME call; "spk:" narrows to samples
// with that speaker among whatever the rest of the query already resolved (applied last).
function parseAddInput(text) {
  const out = new Map();
  let scopeRun = null, spkFilter = null, sawResolvingToken = false;
  for (let tok of text.split(",").map(t => t.trim()).filter(Boolean)) {
    const runTok = tok.match(/^run:(.+)$/i);
    if (runTok) { scopeRun = runTok[1]; continue; }
    const spkTok = tok.match(/^spk:?(\d+)$/i);
    if (spkTok) { spkFilter = +spkTok[1]; continue; }
    sawResolvingToken = true;
    const numTok = tok.match(NUM_TOKEN_RE);
    if (numTok) {
      const [, field, cmp, a, b] = numTok;
      const runScoped = field === "mse" || field === "com_dist";
      const runs = runScoped ? (scopeRun ? [scopeRun] : (S.runs.length === 1 ? S.runs : null)) : null;
      if (runScoped && !runs) continue;   // ambiguous run scope (0 or 2+ runs loaded, no run: given) -- skip token
      for (const s of S.man.samples) {
        const v = runScoped ? null : s[field];
        let val = v;
        if (runScoped) {
          const rec = S.runData[runs[0]] && S.runData[runs[0]].samples[s.sample_id];
          val = rec ? rec[field] : null;
        }
        if (val == null) continue;
        const passes = cmp === "<" ? val < +a : cmp === "<=" ? val <= +a : cmp === ">" ? val > +a
          : cmp === ">=" ? val >= +a : (val >= +a && val <= +b);
        if (passes) out.set(s.sample_id, s);
      }
      continue;
    }
    if (/^[po]/i.test(tok) && /^[po]0*\d+$/i.test(tok)) {
      const oid = tok.slice(1).padStart(6, "0");
      for (const s of GROUPS.get(oid) || []) out.set(s.sample_id, s);
      continue;
    }
    const range = tok.match(/^(\d+)\s*-\s*(\d+)$/);
    if (range) {
      for (let i = +range[1]; i <= +range[2]; i++) if (SAMPLE.has(i)) out.set(i, SAMPLE.get(i));
      continue;
    }
    if (/^\d+$/.test(tok) && SAMPLE.has(+tok)) out.set(+tok, SAMPLE.get(+tok));
  }
  // "spk:3" with nothing else to narrow means "all samples with that speaker", not "narrow an
  // empty set down to nothing" -- fall back to the whole dataset as the base in that case.
  let result = sawResolvingToken ? [...out.values()] : S.man.samples;
  if (spkFilter != null) result = result.filter(s => s.speaker === spkFilter);
  return result;
}

// ===== merged speaker + position map =====
const PM = { pad: 34, dots: [] };
function posXY(cv, s) {
  const w = cv.clientWidth - 2 * PM.pad, h = cv.clientHeight - 2 * PM.pad;
  return [PM.pad + (s.com_col / (S.man.out_w - 1)) * w, PM.pad + (s.com_row / (S.man.out_h - 1)) * h];
}
function renderPosmap() {
  const cv = $("#posmap");
  const cssW = cv.parentElement.clientWidth;
  const cssH = Math.max(260, Math.round((cssW - 2 * PM.pad) * (S.man.out_h / S.man.out_w)) + 2 * PM.pad);
  cv.style.height = cssH + "px";
  const dpr = window.devicePixelRatio || 1;
  cv.width = cssW * dpr; cv.height = cssH * dpr;
  const g = cv.getContext("2d");
  g.scale(dpr, dpr);
  g.clearRect(0, 0, cssW, cssH);

  // box outline (the position field)
  g.strokeStyle = BASE;
  g.strokeRect(PM.pad, PM.pad, cssW - 2 * PM.pad, cssH - 2 * PM.pad);

  // speakers outside the box outline, at their physical positions
  const speakerXY = v => {
    const [xf, yf] = SPEAKER_POSITION[v] || [0.5, 0.5];
    const px = xf === 0 ? PM.pad - 18 : xf === 1 ? cssW - PM.pad + 18 : PM.pad + xf * (cssW - 2 * PM.pad);
    const py = (xf === 0 || xf === 1) ? PM.pad + (1 - yf) * (cssH - 2 * PM.pad) : (yf === 1 ? PM.pad - 18 : cssH - PM.pad + 18);
    return [px, py];
  };
  PM.speakerHits = (S.man.facets.speaker || []).map(v => { const [x, y] = speakerXY(v); return { v, x, y }; });
  for (const v of S.man.facets.speaker || []) {
    const [px, py] = speakerXY(v);
    g.beginPath(); g.arc(px, py, 11, 0, 7);
    g.fillStyle = S.filters.speaker.has(v) ? SPK_COLOR(v) : BASE;
    g.fill();
    g.fillStyle = "#0d0d0d"; g.font = "bold 10px system-ui"; g.textAlign = "center"; g.textBaseline = "middle";
    g.fillText(v, px, py + 1);
  }

  // position dots
  PM.dots = [];
  for (const [oid, samples] of GROUPS) {
    const s = samples[0];
    if (s.com_row < 0) continue;
    const [x, y] = posXY(cv, s);
    const inPool = samples.some(s2 => S.pool.has(s2.sample_id));
    const alive = samples.some(s2 => candidatePasses(s2));
    const highlighted = oid === S.cursor;
    g.beginPath(); g.arc(x, y, inPool ? 5 : 3.5, 0, 7);
    g.globalAlpha = alive ? 1 : 0.22;
    g.fillStyle = inPool ? colorFor(s) : (highlighted ? INK : MUTED);
    g.fill();
    if (inPool || highlighted) { g.strokeStyle = "#fff"; g.lineWidth = 1.2; g.stroke(); }
    g.globalAlpha = 1;
    PM.dots.push({ oid, x, y, alive });
  }
}
function nearestDot(x, y, maxD = 12) {
  let best = null, bd = maxD * maxD;
  for (const d of PM.dots) {
    const dd = (d.x - x) ** 2 + (d.y - y) ** 2;
    if (dd < bd && d.alive) { bd = dd; best = d; }
  }
  return best;
}
function nearestSpeaker(x, y, maxD = 16) {
  let best = null, bd = maxD * maxD;
  for (const h of PM.speakerHits || []) {
    const dd = (h.x - x) ** 2 + (h.y - y) ** 2;
    if (dd < bd) { bd = dd; best = h; }
  }
  return best;
}
{
  const cv = $("#posmap");
  const pos = e => { const r = cv.getBoundingClientRect(); return [e.clientX - r.left, e.clientY - r.top]; };
  cv.addEventListener("mousemove", e => {
    const [x, y] = pos(e);
    const d = nearestDot(x, y);
    const next = d ? d.oid : null;
    if (next === S.cursor) return;
    S.cursor = next;
    renderPosmap();
  });
  cv.addEventListener("click", e => {
    const [x, y] = pos(e);
    const spk = nearestSpeaker(x, y);
    if (spk) { toggleFilter("speaker", spk.v); return; }
    const d = nearestDot(x, y);
    if (d) { toggleGroupInPool(d.oid); S.cursor = d.oid; cv.focus(); }
  });
  cv.addEventListener("keydown", e => {
    if (e.key === "Enter") { if (S.cursor) toggleGroupInPool(S.cursor); e.preventDefault(); return; }
    const dirs = { ArrowLeft: [-1, 0], ArrowRight: [1, 0], ArrowUp: [0, -1], ArrowDown: [0, 1] };
    if (!dirs[e.key]) return;
    e.preventDefault();
    const [dx, dy] = dirs[e.key];
    const cur = PM.dots.find(d => d.oid === S.cursor) || PM.dots[0];
    if (!cur) return;
    let best = null, bd = 1e9;
    for (const d of PM.dots) {
      if (d === cur || !d.alive) continue;
      const vx = d.x - cur.x, vy = d.y - cur.y;
      if (vx * dx + vy * dy <= 0) continue;
      const dist = vx * vx + vy * vy + 2 * (dx ? vy * vy : vx * vx);
      if (dist < bd) { bd = dist; best = d; }
    }
    if (best) { S.cursor = best.oid; renderPosmap(); }
  });
}
function toggleGroupInPool(oid) {
  const samples = (GROUPS.get(oid) || []).filter(s => candidatePasses(s));
  const allIn = samples.length > 0 && samples.every(s => S.pool.has(s.sample_id));
  allIn ? removeFromPool(samples.map(s => s.sample_id)) : addToPool(samples);
}
function setAllSpeakers(on) { S.filters.speaker = new Set(on ? S.man.facets.speaker : []); renderAll(); }
function addAllPositions() {
  addToPool([...GROUPS.values()].flatMap(g => g.filter(s => s.com_row >= 0 && candidatePasses(s))));
}
function removeAllPositions() {
  // only drops samples that came from a real position (not empty-box, not a bare run/search add) —
  // scoped removal, mirroring what "all" would have added, not a blanket pool clear
  removeFromPool([...S.pool.values()].filter(s => s.com_row >= 0).map(s => s.sample_id));
}

// empty-box chip row
function renderEmptyChips() {
  const holder = $("#empty-chips");
  holder.innerHTML = "";
  EMPTY_GROUPS.forEach((g, i) => {
    const chip = document.createElement("span");
    const inPool = g.sample_ids.some(id => S.pool.has(id));
    chip.className = "chip" + (inPool ? " active" : "");
    chip.textContent = `empty-${i + 1}`;
    chip.onclick = () => addToPool(g.sample_ids.map(id => SAMPLE.get(id)).filter(Boolean));
    holder.appendChild(chip);
  });
}
function buildEmptySelect() {
  const sel = $("#empty-select");
  sel.innerHTML = "";
  EMPTY_GROUPS.forEach((g, i) => sel.appendChild(Object.assign(document.createElement("option"), { value: g.output_id, textContent: `empty-${i + 1}` })));
  if (!S.emptyBaseline && EMPTY_GROUPS.length) S.emptyBaseline = EMPTY_GROUPS[0].output_id;
  sel.value = S.emptyBaseline || "";
}
$("#empty-select").onchange = e => { S.emptyBaseline = e.target.value; updateFFT(); };
$("#pos-all").onclick = addAllPositions;
$("#pos-none").onclick = removeAllPositions;
$("#spk-all").onclick = () => setAllSpeakers(true);
$("#spk-none").onclick = () => setAllSpeakers(false);

$("#color-by").onchange = e => { S.colorBy = e.target.value; renderAll(); };

// ===== laser grid + fft controls =====
function buildLaserGrid() {
  const g = $("#laser-grid");
  g.innerHTML = "";
  for (let i = 0; i < 100; i++) {
    const c = document.createElement("div");
    c.title = `laser ${i} (row ${Math.floor(i / 10)}, col ${i % 10})`;
    c.classList.toggle("on", S.lasers.has(i));
    c.onclick = () => { S.lasers.has(i) ? S.lasers.delete(i) : S.lasers.add(i); laserChanged(); };
    g.appendChild(c);
  }
}
function laserChanged() {
  $("#laser-count").textContent = `${S.lasers.size}/100`;
  for (let i = 0; i < 100; i++) $("#laser-grid").children[i].classList.toggle("on", S.lasers.has(i));
  S.fftCache.clear();
  updateFFT();
}
$("#laser-all").onclick = () => { S.lasers = new Set([...Array(100).keys()]); laserChanged(); };
$("#laser-none").onclick = () => { S.lasers = new Set(); laserChanged(); };
$("#dir-seg").onclick = e => {
  if (!e.target.dataset.dir) return;
  S.dirs = e.target.dataset.dir;
  for (const b of $("#dir-seg").children) b.classList.toggle("active", b === e.target);
  S.fftCache.clear(); updateFFT();
};
$("#logy-seg").onclick = e => {
  if (!e.target.dataset.log) return;
  S.logy = e.target.dataset.log === "log";
  for (const b of $("#logy-seg").children) b.classList.toggle("active", b === e.target);
  updateFFT();
};
$("#norm-check").onchange = e => { S.norm = e.target.checked; S.fftCache.clear(); updateFFT(); };
$("#avg-speaker-check").onchange = e => { S.avgSpeaker = e.target.checked; updateFFT(); };
$("#diff-empty-check").onchange = e => {
  S.diffEmpty = e.target.checked;
  $("#empty-select-line").hidden = !S.diffEmpty;
  updateFFT();
};

// ===== fft view =====
const fftKey = () => `${S.lasers.size === 100 ? "all" : [...S.lasers].sort((a, b) => a - b).join(",")}|${S.dirs}|${S.norm ? "n" : "r"}`;

async function fetchCurves(ids) {
  const key = fftKey();
  const missing = ids.filter(id => !S.fftCache.has(key + "|" + id));
  if (missing.length) {
    const lasers = S.lasers.size === 100 ? "all" : [...S.lasers].sort((a, b) => a - b).join(",");
    const r = await jget(`/api/fft?ids=${missing.join(",")}&lasers=${lasers}&dirs=${S.dirs}&norm=${S.norm}`);
    S.freqs = r.freqs;
    for (const [id, c] of Object.entries(r.curves)) S.fftCache.set(key + "|" + id, c);
  }
  return key;
}

function captureZoom(gd) {
  if (!gd || !gd.layout) return;
  const xr = gd.layout.xaxis && gd.layout.xaxis.range, yr = gd.layout.yaxis && gd.layout.yaxis.range;
  if (xr && yr && !gd.layout.xaxis.autorange) S.fftZoom = { x: xr.slice(), y: yr.slice(), logy: S.logy };
}

const updateFFT = debounce(async () => {
  const emptyMode = S.diffEmpty;
  if (emptyMode && !EMPTY_GROUPS.length) {
    Plotly.react("fft-plot", [], emptyLayout("no empty-box sample available in this dataset"));
    return;
  }
  const pooled = benchSamples();
  const ids = pooled.map(s => s.sample_id);
  if (S.lasers.size === 0) { Plotly.react("fft-plot", [], emptyLayout("select at least one laser")); return; }
  if (!ids.length) { Plotly.react("fft-plot", [], emptyLayout("add samples to the workbench (+bench in the table, or \"add filtered to bench\")")); return; }

  let key;
  try {
    const emptyIds = emptyMode ? (EMPTY_GROUPS.find(g => g.output_id === S.emptyBaseline)?.sample_ids || []) : [];
    key = await fetchCurves([...new Set([...ids, ...emptyIds])]);
  } catch (e) { console.error(e); return; }
  if (key !== fftKey()) return;

  const gd = $("#fft-plot");
  captureZoom(gd);

  let emptyMean = null;
  if (emptyMode) {
    const emptyIds = EMPTY_GROUPS.find(g => g.output_id === S.emptyBaseline)?.sample_ids || [];
    const curves = emptyIds.map(id => S.fftCache.get(key + "|" + id)).filter(Boolean);
    if (curves.length) emptyMean = curves[0].map((_, i) => curves.reduce((a, c) => a + c[i], 0) / curves.length);
  }

  const traces = [];
  if (S.avgSpeaker) {
    // one line per speaker, averaged over every pooled position that has that speaker —
    // colored by speaker regardless of the color-by setting, since that's the whole axis here
    const bySpeaker = new Map();
    for (const s of pooled) { if (!bySpeaker.has(s.speaker)) bySpeaker.set(s.speaker, []); bySpeaker.get(s.speaker).push(s); }
    for (const spk of [...bySpeaker.keys()].sort((a, b) => a - b)) {
      const samples = bySpeaker.get(spk);
      const curves = samples.map(s => S.fftCache.get(key + "|" + s.sample_id)).filter(Boolean);
      if (!curves.length) continue;
      let avg = curves[0].map((_, i) => curves.reduce((a, c) => a + c[i], 0) / curves.length);
      if (emptyMean) avg = avg.map((v, i) => v - emptyMean[i]);
      traces.push({ type: "scattergl", mode: "lines", x: S.freqs, y: avg, name: `spk ${spk} avg`, showlegend: true,
        line: { width: 2, color: SPK_COLOR(spk) },
        hovertemplate: `speaker ${spk} · avg of ${samples.length} positions<br>%{x:.0f} Hz · %{y:.4f}<extra></extra>` });
    }
  } else {
    const seen = new Set();
    for (const id of ids) {
      let curve = S.fftCache.get(key + "|" + id);
      const s = SAMPLE.get(id);
      if (!curve || !s) continue;
      if (emptyMean) curve = curve.map((v, i) => v - emptyMean[i]);
      const ck = colorKey(s);
      traces.push({
        type: "scattergl", mode: "lines", x: S.freqs, y: curve,
        name: ck, legendgroup: ck, showlegend: !seen.has(ck),
        line: { width: 1.6, color: colorFor(s) },
        opacity: 0.9,
        hovertemplate: `${label(s)} · ${s.layout}<br>%{x:.0f} Hz · %{y:.4f}<extra></extra>`,
      });
      seen.add(ck);
    }
  }

  const useLog = S.logy && !S.diffEmpty; // diffs can go negative
  const keepZoom = S.fftZoom && S.fftZoom.logy === useLog;
  const layout = LAYOUT({
    xaxis: AXIS({ title: { text: "frequency (Hz)", font: { size: 11 } }, ...(keepZoom ? { range: S.fftZoom.x, autorange: false } : {}) }),
    yaxis: AXIS({
      type: useLog ? "log" : "linear",
      title: { text: `|fft|${S.norm ? " (normalized)" : ""} mean (${S.lasers.size} lasers, ${S.dirs})${S.diffEmpty ? " − empty" : ""}${S.avgSpeaker ? " · avg by speaker" : ""}`, font: { size: 11 } },
      ...(keepZoom ? { range: S.fftZoom.y, autorange: false } : {}),
    }),
  });
  Plotly.react("fft-plot", traces, traces.length ? layout : emptyLayout("nothing to plot"), CONFIG);
}, 120);

// ===== runs =====
function buildRunSelect() {
  const sel = $("#run-add");
  sel.innerHTML = `<option value="">+ add run…</option>`;
  for (const r of S.man.runs.filter(r => !S.runs.includes(r)))
    sel.appendChild(Object.assign(document.createElement("option"), { value: r, textContent: r }));
}
$("#run-add").onchange = e => { if (e.target.value) addRun(e.target.value); e.target.value = ""; };

async function addRun(name, silent = false) {
  if (S.runs.includes(name)) return;
  S.runs.push(name);
  if (!silent) renderRunChips();
  try {
    S.runData[name] = await jget(`/api/run/${name}`);
    // bulk-populate the pool with every sample in this run (all splits, including train)
    const samples = Object.keys(S.runData[name].samples).map(id => SAMPLE.get(+id)).filter(Boolean);
    for (const s of samples) S.pool.set(s.sample_id, s);
  } catch (err) {
    console.error(err); S.runs = S.runs.filter(r => r !== name);
  }
  if (silent) return;
  renderRunChips(); buildRunSelect();
  renderAll();
}
function dropRun(name) {
  S.runs = S.runs.filter(r => r !== name);
  delete S.runData[name]; delete S.runMasks[name];
  renderRunChips(); buildRunSelect(); renderAll();
}
function renderRunChips() {
  const holder = $("#run-chips");
  holder.innerHTML = "";
  for (const name of S.runs) {
    const d = S.runData[name];
    const chip = document.createElement("span");
    chip.className = "chip active run-chip" + (d ? "" : " loading");
    chip.style.borderColor = runColor(name);
    const upos = d && d.aggregates.unseen_pos;
    const badge = upos ? `<span class="badge" style="background:${runBadgeColor(upos.com_dist)}22;color:${runBadgeColor(upos.com_dist)}">${upos.com_dist.toFixed(3)}</span>` : "";
    chip.innerHTML = `${name} ${badge} <span class="x">✕</span>`;
    chip.title = d ? "unseen_pos com_dist (mean, headline metric)\n\n" + Object.entries(d.aggregates).map(([k, v]) => `${k}: mse ${v.mse} · com ${v.com_dist} · n=${v.n}`).join("\n") : "extracting…";
    chip.querySelector(".x").onclick = () => dropRun(name);
    holder.appendChild(chip);
  }
}

// ===== masks =====
async function ensureGtMasks(ids) {
  const missing = ids.filter(id => !(id in S.gtMasks));
  if (!missing.length) return;
  Object.entries(await jget(`/api/gt_masks?ids=${missing.join(",")}`)).forEach(([id, m]) => S.gtMasks[+id] = m);
}
async function ensureRunMasks(run, ids) {
  S.runMasks[run] = S.runMasks[run] || {};
  const missing = ids.filter(id => !(id in S.runMasks[run]));
  if (!missing.length) return;
  Object.entries(await jget(`/api/run/${run}/masks?ids=${missing.join(",")}`)).forEach(([id, m]) => S.runMasks[run][+id] = m);
}

function maskCanvas(mask, mode, cls = "") {
  const h = mask.length, w = mask[0].length;
  const cv = document.createElement("canvas");
  cv.width = w; cv.height = h; cv.className = cls;
  const g = cv.getContext("2d"), img = g.createImageData(w, h);
  const mix = (a, b, t) => a.map((v, i) => Math.round(v + (b[i] - v) * t));
  const SURF = [26, 26, 25], RED = [224, 82, 82], GREEN = [24, 178, 24];
  for (let r = 0; r < h; r++) for (let c = 0; c < w; c++) {
    let rgb, v = mask[r][c];
    if (mode === "diff") rgb = v >= 0 ? mix([56, 56, 53], RED, Math.min(v, 1)) : mix([56, 56, 53], GREEN, Math.min(-v, 1));
    else rgb = mix(SURF, mode === "gt" ? GREEN : RED, Math.min(Math.max(v, 0), 1));
    const o = (r * w + c) * 4;
    [img.data[o], img.data[o + 1], img.data[o + 2], img.data[o + 3]] = [...rgb, 255];
  }
  g.putImageData(img, 0, 0);
  return cv;
}

// ===== box viewer =====
const updateBox = debounce(async () => {
  const pooled = benchSamples();
  const traces = [];
  const bg = [...GROUPS.values()].map(g => g[0]).filter(s => s.com_row >= 0);
  traces.push({ type: "scatter", mode: "markers", showlegend: false, hoverinfo: "skip",
    x: bg.map(s => s.com_col), y: bg.map(s => s.com_row), marker: { size: 4, color: BASE } });

  const seen = new Set(), byGroup = new Map();
  for (const s of pooled) { if (!byGroup.has(s.output_id)) byGroup.set(s.output_id, []); byGroup.get(s.output_id).push(s); }
  for (const [oid, samples] of byGroup) {
    if (samples[0].com_row < 0) continue;
    const s = samples[0], ck = colorKey(s);
    traces.push({
      type: "scatter", mode: "markers", name: ck, legendgroup: ck, showlegend: !seen.has(ck),
      x: [s.com_col], y: [s.com_row],
      marker: { size: 11, color: colorFor(s), line: { width: 2, color: SURFACE } },
      hovertemplate: `p${+oid} · ${s.layout} · gt com (${s.com_row.toFixed(1)}, ${s.com_col.toFixed(1)})<extra></extra>`,
    });
    seen.add(ck);
  }
  for (const run of S.runs) {
    const rd = S.runData[run];
    if (!rd) continue;
    const xs = [], ys = [], texts = [];
    for (const s of pooled) {
      if (s.com_row < 0) continue;   // empty-box has no real gt position — skip, don't draw a line to (-1,-1)
      const rec = rd.samples[s.sample_id];
      if (!rec) continue;
      traces.push({ type: "scatter", mode: "lines", showlegend: false, hoverinfo: "skip",
        x: [s.com_col, rec.pred_col], y: [s.com_row, rec.pred_row], line: { width: 1, color: runColor(run) }, opacity: 0.4 });
      xs.push(rec.pred_col); ys.push(rec.pred_row);
      texts.push(`${label(s)}<br>${run} pred com (${rec.pred_row.toFixed(1)}, ${rec.pred_col.toFixed(1)})<br>com_dist ${fmt(rec.com_dist)} · mse ${fmt(rec.mse, 5)}`);
    }
    if (xs.length) traces.push({ type: "scatter", mode: "markers", name: run, legendgroup: "run:" + run,
      x: xs, y: ys, text: texts, hovertemplate: "%{text}<extra></extra>",
      marker: { symbol: "x", size: 7, color: runColor(run), line: { width: 1, color: SURFACE } } });
  }
  // empty-box samples have no real (row, col) to plot spatially — surface them as a compact
  // per-run mean-mse line instead of silently dropping them or drawing a line to (-1,-1)
  const emptyPooled = pooled.filter(s => s.com_row < 0);
  const stripEl = $("#empty-box-strip");
  if (emptyPooled.length && S.runs.some(r => S.runData[r])) {
    const parts = S.runs.filter(r => S.runData[r]).map(r => {
      const vals = emptyPooled.map(s => S.runData[r].samples[s.sample_id]?.mse).filter(v => v != null);
      const avg = vals.length ? vals.reduce((a, b) => a + b, 0) / vals.length : null;
      return `<span style="color:${runColor(r)}">${r}</span> ${fmt(avg, 5)}`;
    });
    stripEl.innerHTML = `${emptyPooled.length} empty-box sample${emptyPooled.length === 1 ? "" : "s"} (not plotted — no position) · mean mse: ${parts.join(" · ")}`;
  } else if (emptyPooled.length) {
    stripEl.textContent = `${emptyPooled.length} empty-box sample${emptyPooled.length === 1 ? "" : "s"} in pool (not plotted — no position; add a run to see mse)`;
  } else {
    stripEl.textContent = "";
  }

  if (S.contours && pooled.length) {
    const gtIds = [...new Map(pooled.map(s => [s.output_id, s.sample_id])).values()];
    try {
      await ensureGtMasks(gtIds);
      for (const run of S.runs) await ensureRunMasks(run, pooled.map(s => s.sample_id));
    } catch (e) { console.error(e); }
    for (const id of gtIds) if (S.gtMasks[id]) traces.push(contourTrace(S.gtMasks[id], GT_COLOR));
    for (const run of S.runs)
      for (const s of pooled) {
        const m = S.runMasks[run] && S.runMasks[run][s.sample_id];
        if (m) traces.push(contourTrace(m, runColor(run)));
      }
  }
  Plotly.react("box-plot", traces, LAYOUT({
    margin: { l: 30, r: 6, t: 6, b: 24 },
    legend: { orientation: "h", y: -0.12, yanchor: "top" },
    xaxis: AXIS({ range: [-0.8, S.man.out_w - 0.2], constrain: "domain" }),
    yaxis: AXIS({ range: [S.man.out_h - 0.2, -0.8], scaleanchor: "x", scaleratio: 1 }),
  }), CONFIG);
}, 120);

const contourTrace = (mask, color, ghost = false) => ({
  type: "contour", z: mask, showscale: false, showlegend: false, hoverinfo: "skip",
  contours: { start: 0.5, end: 0.5, size: 1, coloring: "none" },
  line: { color, width: ghost ? 1.8 : 1.4, dash: ghost ? "dot" : "solid" },
  opacity: ghost ? 0.85 : 1,
});
$("#contours-toggle").onchange = e => { S.contours = e.target.checked; updateBox(); };

// ===== compact table: one view, sortable by clicking column headers =====
// first column = ground truth/sample thumbnail, sortable by sample_id or position (output_id);
// each following column = one active run's prediction thumbnail, sortable by that run's mse/com_dist.
const GT_SORT_COLS = [["sample_id", "id"], ["output_id_num", "position"]];
const RUN_SORT_COLS = [["mse", "mse"], ["com_dist", "com dist"]];

function sortVal(s, key) {
  if (key.startsWith("run:")) {
    const [, run, field] = key.split(":");
    const rec = S.runData[run] && S.runData[run].samples[s.sample_id];
    return rec ? rec[field] : null;
  }
  if (key === "output_id_num") return +s.output_id;
  return s[key];
}
function sortedPool() {
  const rows = poolSamples();
  const { key, dir } = S.sort;
  return rows.sort((a, b) => {
    const va = sortVal(a, key), vb = sortVal(b, key);
    if (va == null && vb == null) return 0;
    if (va == null) return 1;   // nulls (e.g. empty-box com_dist) always sort last
    if (vb == null) return -1;
    return (va > vb ? 1 : va < vb ? -1 : 0) * dir;
  });
}
function setSort(key) {
  S.sort = { key, dir: S.sort.key === key ? -S.sort.dir : 1 };
  renderCompact();
}

function renderSamplesView() { renderCompact(); }

let currentAudio = null, currentAudioBtn = null;
function stopCurrentAudio() {
  if (currentAudio) currentAudio.pause();
  if (currentAudioBtn) currentAudioBtn.classList.remove("playing");
  currentAudio = null; currentAudioBtn = null;
}
function wirePlayButtons(root) {
  root.querySelectorAll(".play").forEach(b => b.onclick = e => {
    e.stopPropagation();
    const sameButton = currentAudioBtn === b;
    stopCurrentAudio();
    if (sameButton) return;   // second click on the playing button: pause and stop, don't restart
    currentAudio = new Audio(b.dataset.src);   // fresh Audio each time -> always starts at 0
    currentAudioBtn = b;
    b.classList.add("playing");
    currentAudio.addEventListener("ended", stopCurrentAudio);
    currentAudio.play();
  });
}
// small white cross at the ground-truth center of mass, drawn directly on an overlay canvas
function drawComCross(cv, s) {
  if (!s || s.com_row < 0) return;
  const g = cv.getContext("2d");
  const x = (s.com_col + 0.5) / S.man.out_w * cv.width, y = (s.com_row + 0.5) / S.man.out_h * cv.height;
  const r = Math.max(4, cv.width * 0.02);
  g.strokeStyle = "#fff"; g.lineWidth = 2;
  g.beginPath(); g.moveTo(x - r, y); g.lineTo(x + r, y); g.moveTo(x, y - r); g.lineTo(x, y + r); g.stroke();
}

function rowHtml(s, activeRuns, idx) {
  const gtCell = `<span class="overlay-slot" data-id="${s.sample_id}" data-kind="gt"></span>`;
  const predCells = activeRuns.map(r => {
    const rec = S.runData[r] && S.runData[r].samples[s.sample_id];
    const cap = rec
      ? `${rec.split.replace("unseen_", "u-")} · mse ${fmt(rec.mse, 5)} · com ${fmt(rec.com_dist)}`
      : `not in run`;
    return `<td class="run-col">
    <div class="compact-cell" data-run="${r}">
      <div class="compact-cap"><span class="swatch" style="background:${colorFor(s)}"></span><span class="compact-id">#${s.sample_id}</span></div>
      <div class="compact-cap" style="color:${runColor(r)}">${cap}</div>
      <span class="overlay-slot" data-id="${s.sample_id}" data-run="${r}" data-kind="pred"></span>
    </div></td>`;
  }).join("");
  const inBench = S.bench.has(s.sample_id);
  return `<tr data-id="${s.sample_id}">
    <td class="idx-col">
      ${idx}
      <span class="row-bench${inBench ? " active" : ""}" title="${inBench ? "remove from workbench" : "add to workbench (box + fft viewers)"}">${inBench ? "✓" : "+"} bench</span>
    </td>
    <td>
      <span class="row-x" title="remove from table">✕</span>
      <div class="compact-cell">
        <div class="compact-cap"><span class="swatch" style="background:${colorFor(s)}"></span><span class="compact-id">#${s.sample_id}</span> · p${+s.output_id} · spk${s.speaker} · ${s.layout} · com (${s.com_row.toFixed(1)}, ${s.com_col.toFixed(1)})</div>
        ${gtCell}
      </div>
    </td>
    ${predCells}</tr>`;
}

let _compactRenderGen = 0;
async function renderCompact() {
  // guards a stale render (started before a newer one) from clobbering live DOM/listeners --
  // same race the old table view hit from hover/live-poll re-entrancy.
  const gen = ++_compactRenderGen;
  const wrap = $("#compact-wrap");
  const rows = sortedPool();
  if (!rows.length) {
    if (gen !== _compactRenderGen) return;
    wrap.innerHTML = `<div class="hint" style="padding:12px">add samples — pick a run above, click a position, or use "add matching" / the search box in dataset filters</div>`;
    return;
  }
  const activeRuns = S.runs.filter(r => S.runData[r]);
  if (gen !== _compactRenderGen) return;

  // the active sort column needs to be unmistakable, not just an arrow buried in dense header
  // text -- give it its own pill (accent bg + arrow) so it reads at a glance which column/run
  // and metric the table is currently ordered by
  const sortSpan = (k, l) => S.sort.key === k
    ? `<span data-sort-key="${k}" class="sort-active">${l} ${S.sort.dir > 0 ? "↑" : "↓"}</span>`
    : `<span data-sort-key="${k}">${l}</span>`;
  const thead = `<thead><tr>
    <th class="idx-col">index</th>
    <th>${GT_SORT_COLS.map(([k, l]) => sortSpan(k, l)).join(" · ")} — ground truth</th>
    ${activeRuns.map(r => `<th class="run-col" style="color:${runColor(r)}">${r}
      <span style="color:${INK};font-weight:400"> — ${RUN_SORT_COLS.map(([k, l]) => sortSpan(`run:${r}:${k}`, l)).join(" · ")}</span>
    </th>`).join("")}
  </tr></thead>`;

  const tb = rows.map((s, i) => rowHtml(s, activeRuns, i + 1)).join("");
  if (gen !== _compactRenderGen) return;
  wrap.innerHTML = `<table class="compact-table">${thead}<tbody>${tb}</tbody></table>`;

  observeRowsForFill(wrap);   // lazy: only paint overlay canvases for rows scrolled into view
  if (gen !== _compactRenderGen) return;
  wrap.querySelectorAll("[data-sort-key]").forEach(el => el.onclick = e => { e.stopPropagation(); setSort(el.dataset.sortKey); });
  wrap.querySelectorAll(".row-x").forEach(x => x.onclick = e => {
    e.stopPropagation();
    removeFromPool([+e.target.closest("tr").dataset.id]);
  });
  wrap.querySelectorAll(".row-bench").forEach(x => x.onclick = e => {
    e.stopPropagation();
    const id = +e.target.closest("tr").dataset.id;
    S.bench.has(id) ? removeFromBench([id]) : addToBench([SAMPLE.get(id)]);
  });
  wrap.querySelectorAll("img.thumb").forEach(img => img.onclick = () => openLightbox(img.src));
  wirePlayButtons(wrap);
}

// with the whole dataset + several runs loaded by default, painting every overlay canvas eagerly
// means thousands of serial thumbnail fetches before anything is visible. Instead, only fill a
// row's overlay slots once it's scrolled near the viewport -- the table stays responsive
// immediately and fills in as you scroll, same data, no eager cost for off-screen rows.
let _rowObserver = null;
function observeRowsForFill(root) {
  if (_rowObserver) _rowObserver.disconnect();
  _rowObserver = new IntersectionObserver(entries => {
    for (const e of entries) {
      if (!e.isIntersecting) continue;
      _rowObserver.unobserve(e.target);
      fillOverlayStacks(e.target).catch(console.error);
    }
  // ~130px per row (220px-wide thumbnail + captions) -> 1300px covers ~10 rows of prefetch
  // ahead of/behind the visible viewport in each scroll direction, so scrolling doesn't outrun fills
  }, { root: root, rootMargin: "1300px 0px" });
  root.querySelectorAll("tbody tr").forEach(tr => _rowObserver.observe(tr));
}

// fills every ".overlay-slot" in root: kind="gt" -> overhead + gt mask + com cross, opens the
// metadata+mask detail modal in gt mode; kind="pred" (data-run set) -> overhead + that run's
// predicted mask + pred com cross, opens the detail modal in prediction mode for that run
async function fillOverlayStacks(root) {
  const slots = [...root.querySelectorAll(".overlay-slot")];
  if (!slots.length) return;
  const gtIds = [...new Set(slots.filter(sl => sl.dataset.kind === "gt").map(sl => +sl.dataset.id))];
  if (gtIds.length) await ensureGtMasks(gtIds).catch(console.error);
  const byRun = new Map();
  for (const sl of slots) if (sl.dataset.kind === "pred") {
    if (!byRun.has(sl.dataset.run)) byRun.set(sl.dataset.run, new Set());
    byRun.get(sl.dataset.run).add(+sl.dataset.id);
  }
  for (const [run, ids] of byRun) await ensureRunMasks(run, [...ids]).catch(console.error);

  for (const sl of slots) {
    const id = +sl.dataset.id;
    const s = SAMPLE.get(id);
    let cv;
    if (sl.dataset.kind === "gt") {
      const gt = S.gtMasks[id];
      if (!gt) continue;
      cv = await maskOverlayCanvas(id, gt, [24, 178, 24], "ground truth");
      cv.onclick = () => openDetail(id, null);
    } else {
      const run = sl.dataset.run;
      const pred = S.runMasks[run] && S.runMasks[run][id];
      if (!pred) continue;
      cv = await maskOverlayCanvas(id, pred, [224, 82, 82], run);
      cv.onclick = () => openDetail(id, run);
    }
    drawComCross(cv, s);
    cv.className = "thumb";
    if (sl.isConnected) sl.replaceWith(cv);   // view may have re-rendered while this awaited
  }
}

// accepts a plain image URL (bare overhead photo) or a canvas (e.g. an overlay already showing
// the mask) -- clicking a mask-overlay thumbnail should enlarge with the mask still visible,
// not silently fall back to the bare photo
function openLightbox(srcOrCanvas) {
  const img = $("#lightbox-img");
  if (srcOrCanvas instanceof HTMLCanvasElement) {
    img.src = srcOrCanvas.toDataURL();
  } else {
    img.src = srcOrCanvas;
  }
  $("#lightbox").classList.add("open");
}
$("#lightbox").onclick = () => $("#lightbox").classList.remove("open");

// ===== detail modal (mask comparison) =====
// two separate tables so it's unambiguous which facts are fixed dataset properties of the
// sample vs which come from this run's prediction of it
function sampleMetaRows(s) {
  const rows = [
    ["id", `#${s.sample_id}`], ["speaker", s.speaker], ["layout", s.layout], ["n objects", s.n_objects],
    ["gt com", `(${s.com_row.toFixed(1)}, ${s.com_col.toFixed(1)})`],
    ["audio", `<button class="play" data-src="/media/${s.sample_id}/audio">in ▶</button> <button class="play" data-src="/media/${s.sample_id}/recovered">rec ▶</button>`],
  ];
  return rows.map(([k, v]) => `<tr><td>${k}</td><td>${v}</td></tr>`).join("");
}
function predMetaRows(run, rec) {
  const rows = [
    ["run", run], ["split", rec?.split.replace("unseen_", "u-") ?? "?"],
    ["mse", fmt(rec?.mse, 6)], ["com_dist", fmt(rec?.com_dist)],
    ["pred com", `(${fmt(rec?.pred_row, 1)}, ${fmt(rec?.pred_col, 1)})`],
  ];
  return rows.map(([k, v]) => `<tr><td>${k}</td><td>${v}</td></tr>`).join("");
}

// fixed pixel size every detail-modal image renders at, regardless of view -- overhead photos
// are natively 256x104 (checked across the dataset); scaling every image/heatmap to the same
// width keeps gt/pred/diff/overlay visually comparable and never squished relative to each other
const DETAIL_IMG_W = 460;

// gt com = light green cross, pred com = light red cross -- consistent everywhere in the modal
// so the two marks are distinguishable at a glance without reading a legend
const GT_COM_MARK = "#8fe08f", PRED_COM_MARK = "#f09a9a";

async function openDetail(sampleId, run) {
  const s = SAMPLE.get(sampleId);
  await ensureGtMasks([sampleId]);
  const gt = S.gtMasks[sampleId];
  if (!gt) return;
  const rec = run ? S.runData[run]?.samples[sampleId] : null;
  let pred = null;
  if (run) { await ensureRunMasks(run, [sampleId]); pred = S.runMasks[run][sampleId]; if (!pred) return; }

  $("#modal-title").textContent = run ? `${label(s)} — ${run}` : label(s);
  $("#modal-meta-sample").innerHTML = sampleMetaRows(s);
  $("#modal-meta-pred-group").hidden = !run;
  if (run) $("#modal-meta-pred").innerHTML = predMetaRows(run, rec);
  $("#modal").classList.add("open");
  wirePlayButtons($("#modal"));

  const side = $("#m-side"), combined = $("#m-combined"), diff = $("#m-diff");
  side.innerHTML = ""; combined.innerHTML = ""; diff.innerHTML = "";

  const drawCross = (cv, row, col, color) => {
    const g = cv.getContext("2d");
    const x = (col + 0.5) / S.man.out_w * cv.width, y = (row + 0.5) / S.man.out_h * cv.height;
    const r = Math.max(4, cv.width * 0.02);
    g.strokeStyle = color; g.lineWidth = 2;
    g.beginPath(); g.moveTo(x - r, y); g.lineTo(x + r, y); g.moveTo(x, y - r); g.lineTo(x, y + r); g.stroke();
  };
  // small red/green legend watermark burned directly into the canvas corner -- replaces a
  // separate caption line, so the panel header can stay a single line (title + toggle only)
  const drawLegend = (cv, ...words) => {
    const g = cv.getContext("2d");
    g.font = `${Math.round(cv.width * 0.032)}px system-ui`; g.textBaseline = "top";
    let x = 8;
    for (const [text, color] of words) {
      g.fillStyle = "rgba(0,0,0,0.55)"; g.fillRect(x - 3, 5, g.measureText(text).width + 6, cv.width * 0.045);
      g.fillStyle = color; g.fillText(text, x, 7);
      x += g.measureText(text).width + 14;
    }
  };
  const panelHead = (title, checked) => `<div class="sub m-panel-head">
    <span>${title}</span>
    <label class="check-line"><input type="checkbox" class="m-photo-toggle" ${checked ? "checked" : ""}> show overhead</label>
  </div>`;
  // build the whole canvas (incl. crosses + legend) before touching the DOM, then swap it in with
  // one replaceChildren call -- writing innerHTML="" first and appending later (after the photo
  // finishes loading) left a visible blank gap on every toggle click, which read as a screen flicker
  const swapSlot = (slot, cv) => slot.replaceChildren(cv);

  if (!run) {
    // gt-only mode: a single panel, its own overhead toggle, nothing else to show
    side.innerHTML = `${panelHead(label(s), true)}<div class="m-side-row"></div>`;
    const slot = side.querySelector(".m-side-row");
    const renderGt = async withPhoto => {
      const cv = await layeredMaskCanvas(sampleId, [{ mask: gt, rgb: [24, 178, 24], label: "ground truth" }], withPhoto);
      drawCross(cv, s.com_row, s.com_col, GT_COM_MARK);
      drawLegend(cv, ["ground truth", "#8fe08f"]);
      cv.style.cssText = `width:${DETAIL_IMG_W}px;border-radius:6px;display:block`;
      swapSlot(slot, cv);
    };
    side.querySelector(".m-photo-toggle").onchange = e => renderGt(e.target.checked);
    renderGt(true);
    return;
  }

  // panel 1: side by side, each with its own overhead toggle
  side.innerHTML = `${panelHead(`${label(s)} — ${run}`, true)}<div style="display:flex;gap:8px" class="m-side-row"></div>`;
  const sideRow = side.querySelector(".m-side-row");
  const renderSide = async withPhoto => {
    const [gtCv, predCv] = await Promise.all([
      layeredMaskCanvas(sampleId, [{ mask: gt, rgb: [24, 178, 24], label: "ground truth" }], withPhoto),
      layeredMaskCanvas(sampleId, [{ mask: pred, rgb: [224, 82, 82], label: "prediction" }], withPhoto),
    ]);
    drawCross(gtCv, s.com_row, s.com_col, GT_COM_MARK);
    drawCross(predCv, rec.pred_row, rec.pred_col, PRED_COM_MARK);
    drawLegend(gtCv, ["ground truth", "#8fe08f"]);
    drawLegend(predCv, ["predicted", "#f09a9a"]);
    for (const cv of [gtCv, predCv]) cv.style.cssText = `width:${DETAIL_IMG_W}px;border-radius:6px;display:block`;
    sideRow.replaceChildren(gtCv, predCv);
  };
  side.querySelector(".m-photo-toggle").onchange = e => renderSide(e.target.checked);
  renderSide(true);

  // panel 2: both masks combined on one plot, pred layered on top of gt
  combined.innerHTML = `${panelHead(`${label(s)} — ${run}`, true)}<div class="m-combined-slot"></div>`;
  const combinedSlot = combined.querySelector(".m-combined-slot");
  const renderCombined = async withPhoto => {
    const cv = await layeredMaskCanvas(sampleId, [
      { mask: gt, rgb: [24, 178, 24], label: "ground truth" },
      { mask: pred, rgb: [224, 82, 82], label: "prediction" },
    ], withPhoto);
    drawCross(cv, s.com_row, s.com_col, GT_COM_MARK);
    drawCross(cv, rec.pred_row, rec.pred_col, PRED_COM_MARK);
    drawLegend(cv, ["predicted", "#f09a9a"], ["ground truth", "#8fe08f"]);
    cv.style.cssText = `width:${DETAIL_IMG_W}px;border-radius:6px;display:block`;
    swapSlot(combinedSlot, cv);
  };
  combined.querySelector(".m-photo-toggle").onchange = e => renderCombined(e.target.checked);
  renderCombined(true);

  // panel 3: diff -- canvas-based (not plotly heatmap) so it can share the same overhead-photo
  // toggle as the other two panels; defaults OFF since the diff coloring is usually clearer on
  // its own plain background, but it's there if you want to see it in physical context. Still
  // gets a real colorbar (drawn alongside, not part of the photo canvas) for the 0..1 alpha scale.
  const diffMask = pred.map((row, r) => row.map((v, c) => v - gt[r][c]));
  diff.innerHTML = `${panelHead(`${label(s)} — ${run}`, false)}
    <div style="display:flex;align-items:flex-start;gap:10px">
      <div class="m-diff-slot"></div>
      <canvas class="m-diff-colorbar" width="34" height="200" title="red = pred extra · green = gt missed"></canvas>
    </div>`;
  const diffSlot = diff.querySelector(".m-diff-slot");
  drawDiffColorbar(diff.querySelector(".m-diff-colorbar"));
  const renderDiff = async withPhoto => {
    // two-layer diff render: positive (pred extra) in red, negative (gt missed) in green
    const posMask = diffMask.map(row => row.map(v => Math.max(v, 0)));
    const negMask = diffMask.map(row => row.map(v => Math.max(-v, 0)));
    const cv = await layeredMaskCanvas(sampleId, [
      { mask: negMask, rgb: [24, 178, 24], label: "gt missed" },
      { mask: posMask, rgb: [224, 82, 82], label: "pred extra" },
    ], withPhoto);
    drawCross(cv, s.com_row, s.com_col, GT_COM_MARK);
    drawCross(cv, rec.pred_row, rec.pred_col, PRED_COM_MARK);
    drawLegend(cv, ["predicted", "#f09a9a"], ["ground truth", "#8fe08f"]);
    cv.style.cssText = `width:${DETAIL_IMG_W}px;border-radius:6px;display:block`;
    swapSlot(diffSlot, cv);
  };
  diff.querySelector(".m-photo-toggle").onchange = e => renderDiff(e.target.checked);
  renderDiff(false);
}

// vertical diverging colorbar for the diff panel: green (gt missed) -> surface -> red (pred extra)
function drawDiffColorbar(cv) {
  const g = cv.getContext("2d");
  const grad = g.createLinearGradient(0, 0, 0, cv.height);
  grad.addColorStop(0, "#e05252"); grad.addColorStop(0.5, SURFACE); grad.addColorStop(1, "#18b218");
  g.fillStyle = grad; g.fillRect(0, 0, 18, cv.height);
  g.fillStyle = MUTED; g.font = "10px system-ui"; g.textBaseline = "middle";
  g.fillText("1", 22, 6); g.fillText("0", 22, cv.height / 2); g.fillText("-1", 22, cv.height - 6);
}

// overhead photo with the predicted mask painted on top in translucent red — this is the
// default thumbnail in table/panel views once a run is loaded, since seeing the prediction
// against the real photo is the main thing being looked at, not the bare photo
function overlayCanvas(sampleId, pred, label = "prediction") {
  return maskOverlayCanvas(sampleId, pred, [224, 82, 82], label);
}

// generic overhead + translucent-mask overlay, with a live hover tooltip reporting the exact
// mask value under the cursor (native <title>, cheap and works everywhere a canvas can go)
function maskOverlayCanvas(sampleId, mask, rgb, label = "") {
  return layeredMaskCanvas(sampleId, [{ mask, rgb, label }], true);
}

// fast cursor-following readout for mask-canvas hover -- native <title> tooltips have a ~1s
// delay and vanish on any mouse move, which reads as "hover doesn't work" when scrubbing across
// a mask to compare values. A single reused floating div instead shows instantly and tracks the cursor.
let _hoverReadout = null;
function showHoverReadout(x, y, text) {
  if (!_hoverReadout) {
    _hoverReadout = document.createElement("div");
    _hoverReadout.className = "hover-readout";
    document.body.appendChild(_hoverReadout);
  }
  _hoverReadout.textContent = text;
  _hoverReadout.style.left = `${x + 14}px`;
  _hoverReadout.style.top = `${y + 14}px`;
  _hoverReadout.style.display = "block";
}
function hideHoverReadout() {
  if (_hoverReadout) _hoverReadout.style.display = "none";
}

// draws one or more mask layers (each own color, painted in order so later layers sit on top),
// either over the real overhead photo or over a plain surface-colored background. Used by the
// detail modal's side-by-side / combined panels so the same renderer handles both the
// "with overhead" and "without overhead" toggle states, and by maskOverlayCanvas for the
// always-with-overhead thumbnails elsewhere. Hover reports every layer's value at that cell.
function layeredMaskCanvas(sampleId, layers, withPhoto) {
  const H = layers[0].mask.length, W = layers[0].mask[0].length;
  const paint = img => {
    const cv = document.createElement("canvas");
    cv.width = img ? img.naturalWidth : W * 8; cv.height = img ? img.naturalHeight : H * 8;
    const g = cv.getContext("2d");
    if (img) g.drawImage(img, 0, 0);
    else { g.fillStyle = SURFACE; g.fillRect(0, 0, cv.width, cv.height); }
    const cw = cv.width / W, ch = cv.height / H;
    for (const { mask, rgb } of layers) {
      for (let r = 0; r < H; r++) for (let c = 0; c < W; c++) {
        const v = mask[r][c];
        if (v > 0.05) {
          g.fillStyle = `rgba(${rgb[0]},${rgb[1]},${rgb[2]},${Math.min(v * 0.65, 0.75)})`;
          g.fillRect(c * cw, r * ch, cw + 0.5, ch + 0.5);
        }
      }
    }
    cv.onmousemove = e => {
      const rect = cv.getBoundingClientRect();
      const c = Math.min(W - 1, Math.max(0, Math.floor((e.clientX - rect.left) / rect.width * W)));
      const r = Math.min(H - 1, Math.max(0, Math.floor((e.clientY - rect.top) / rect.height * H)));
      const text = layers.map(({ mask, label }) => `${label ? label + " " : ""}${mask[r][c].toFixed(3)}`).join(" · ");
      showHoverReadout(e.clientX, e.clientY, `(x ${c}, y ${r}) ${text}`);
    };
    cv.onmouseleave = hideHoverReadout;
    return cv;
  };
  if (!withPhoto) return Promise.resolve(paint(null));
  return new Promise(resolve => {
    const img = new Image();
    img.src = `/media/${sampleId}/thumb`;
    img.onload = () => resolve(paint(img));
  });
}
document.addEventListener("keydown", e => {
  if (e.key === "Escape") { $("#modal").classList.remove("open"); $("#lightbox").classList.remove("open"); }
});
$("#modal").onclick = e => { if (e.target.id === "modal") $("#modal").classList.remove("open"); };

// ===== filters panel collapse =====
// whole sidebar slides horizontally off/on screen — collapsing it hides the filters entirely
// (not just visually shrinking them) to hand that width to box/table/fft
$("#sidebar-toggle").onclick = () => {
  const shell = $("#shell");
  shell.classList.toggle("sidebar-hidden");
  $("#sidebar-toggle").textContent = shell.classList.contains("sidebar-hidden") ? "›" : "‹";
  if (!shell.classList.contains("sidebar-hidden")) setTimeout(renderPosmap, 190); // zero width while hidden; repaint once the slide-in transition finishes
};

// ===== live updates =====
setInterval(async () => {
  try {
    const { version } = await jget("/api/version");
    if (version !== S.version) {
      S.version = version;
      S.fftCache.clear(); S.gtMasks = {}; S.runMasks = {};
      for (const r of S.runs) S.runData[r] = await jget(`/api/run/${r}`).catch(() => S.runData[r]);
      await boot();
    }
  } catch { /* transient: retried on the next tick */ }
}, 3000);

window.addEventListener("resize", debounce(() => { renderPosmap(); }, 150));

boot().catch(err => {
  document.body.innerHTML = `<pre style="color:#e66767;padding:24px">failed to load manifest:\n${err}</pre>`;
});
