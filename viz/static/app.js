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
  filters: {},                     // facet -> Set of active values (narrows "add matching" candidates only)
  pool: new Map(),                 // sample_id -> sample — the ONE thing every view renders
  ghost: null,                     // { ids:[sample_id], label } transient hover preview, not in pool
  cursor: null,                    // output_id keyboard cursor on the position map
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
  view: "table", runMetric: "com_dist", contours: false,
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
      const rec = S.runData[S.runs[0]] && S.runData[S.runs[0]].samples[s.sample_id];
      if (rec && !set.has(rec.split)) return false;   // only filters samples that are in a loaded run
      continue;
    }
    if (!set.has(s[f])) return false;
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
  renderAll();
}

function renderAll() {
  renderFacets(); renderPosmap(); renderPoolCount(); renderEmptyChips();
  updateFFT(); updateBox(); renderSamplesView(); renderRunsPanel();
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
function poolSamples() { return [...S.pool.values()]; }

function renderPoolCount() {
  $("#pool-count").textContent = `${S.pool.size} sample${S.pool.size === 1 ? "" : "s"}`;
}
$("#pool-clear").onclick = clearPool;
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
function parseAddInput(text) {
  const out = new Map();
  for (let tok of text.split(",").map(t => t.trim()).filter(Boolean)) {
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
  return [...out.values()];
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
    const highlighted = oid === S.cursor || (S.ghost && S.ghost.oid === oid);
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
    if (d) { setGhostPosition(d.oid); return; }
    setGhost(null);
  });
  cv.addEventListener("mouseleave", () => setGhost(null));
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
    if (best) { S.cursor = best.oid; setGhostPosition(best.oid); }
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
function setGhostPosition(oid) {
  const prev = S.ghost && S.ghost.oid;
  if (oid === prev) return;
  const samples = (GROUPS.get(oid) || []).filter(s => candidatePasses(s));
  S.ghost = { oid, ids: samples.map(s => s.sample_id) };
  renderPosmap(); updateFFT(); updateBox(); renderSamplesView(); renderRunsPanel();
}
// skipTableRerender: table-row hover already has a live row on screen (ghost rows are only
// for samples NOT in the pool, and table hover only ever targets pooled rows) — re-rendering
// the whole table on every mouseenter tore down and rebuilt buttons mid-click, breaking the
// audio play buttons. Only repaint highlight classes for that case; other hover sources
// (position map, panel cards) still need the full re-render since they can introduce new rows.
function setGhost(next, skipTableRerender = false) {
  const prevOid = S.ghost && S.ghost.oid;
  if (next === null && prevOid === undefined) return;
  S.ghost = next;
  renderPosmap(); updateFFT(); updateBox(); renderRunsPanel();
  if (skipTableRerender) highlightRows(); else renderSamplesView();
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
  const pooled = poolSamples();
  const ids = pooled.map(s => s.sample_id);
  if (S.ghost) for (const id of S.ghost.ids) if (!ids.includes(id)) ids.push(id);
  if (S.lasers.size === 0) { Plotly.react("fft-plot", [], emptyLayout("select at least one laser")); return; }
  if (!ids.length) { Plotly.react("fft-plot", [], emptyLayout("add samples (position map, search box, or a run) to plot their fft")); return; }

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

  const pinnedIds = new Set(ids.filter(id => S.pool.has(id)));
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
      const isGhost = !pinnedIds.has(id);
      const ck = colorKey(s);
      traces.push({
        type: "scattergl", mode: "lines", x: S.freqs, y: curve,
        name: ck, legendgroup: ck, showlegend: !isGhost && !seen.has(ck),
        line: { width: isGhost ? 1 : 1.6, color: isGhost ? INK : colorFor(s), dash: isGhost ? "dot" : "solid" },
        opacity: isGhost ? 0.75 : 0.9,
        hovertemplate: `${label(s)} · ${s.layout}<br>%{x:.0f} Hz · %{y:.4f}<extra></extra>`,
      });
      if (!isGhost) seen.add(ck);
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

async function addRun(name) {
  if (S.runs.includes(name)) return;
  S.runs.push(name);
  renderRunChips();
  try {
    S.runData[name] = await jget(`/api/run/${name}`);
    // bulk-populate the pool with every sample in this run (all splits, including train)
    addToPool(Object.keys(S.runData[name].samples).map(id => SAMPLE.get(+id)).filter(Boolean));
  } catch (err) {
    console.error(err); S.runs = S.runs.filter(r => r !== name);
  }
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
  const pooled = poolSamples();
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
  if (S.ghost) {
    const s = (GROUPS.get(S.ghost.oid) || [])[0];
    if (s && s.com_row >= 0) traces.push({ type: "scatter", mode: "markers", showlegend: false, hoverinfo: "skip",
      x: [s.com_col], y: [s.com_row], marker: { size: 12, color: "rgba(0,0,0,0)", line: { width: 2, color: INK } } });
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
  if (S.contours && S.ghost) {
    // preview the hovered (not-yet-pooled) sample's contour in a light, dashed line — same
    // ghost-before-you-commit convention as the fft/table hover previews
    const ghostSample = SAMPLE.get(S.ghost.ids[0]);
    if (ghostSample) {
      try {
        await ensureGtMasks([ghostSample.sample_id]);
        if (S.gtMasks[ghostSample.sample_id]) traces.push(contourTrace(S.gtMasks[ghostSample.sample_id], "#ffffff", true));
        for (const run of S.runs) {
          await ensureRunMasks(run, [ghostSample.sample_id]);
          const m = S.runMasks[run] && S.runMasks[run][ghostSample.sample_id];
          if (m) traces.push(contourTrace(m, runColor(run), true));
        }
      } catch (e) { console.error(e); }
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

// ===== table / panels =====
$("#view-seg").onclick = e => {
  if (!e.target.dataset.view) return;
  S.view = e.target.dataset.view;
  for (const b of $("#view-seg").children) b.classList.toggle("active", b === e.target);
  $("#table-wrap").style.display = S.view === "table" ? "" : "none";
  $("#panel-wrap").style.display = S.view === "panel" ? "grid" : "none";
  renderPanelSortControl();
  renderSamplesView();
};

// shared sort-key list for both the table's clickable headers and the panel view's sort dropdown
const BASE_SORT_COLS = [["sample_id", "id"], ["speaker", "spk"], ["layout", "layout"], ["n_objects", "n"], ["com_row", "gt com"]];
const RUN_SORT_COLS = [["split", "split"], ["mse", "mse"], ["com_dist", "com dist"]];
function sortableKeys() {
  const activeRuns = S.runs.filter(r => S.runData[r]);
  const keys = BASE_SORT_COLS.map(([k, l]) => [k, l]);
  for (const r of activeRuns) for (const [k, l] of RUN_SORT_COLS) keys.push([`run:${r}:${k}`, `${r} · ${l}`]);
  return keys;
}
function renderPanelSortControl() {
  const line = $("#panel-sort-line");
  line.hidden = S.view !== "panel";
  if (S.view !== "panel") return;
  const sel = $("#panel-sort-key");
  const keys = sortableKeys();
  const optionsHtml = keys.map(([k, l]) => `<option value="${k}" ${S.sort.key === k ? "selected" : ""}>${l}</option>`).join("");
  if (sel.innerHTML !== optionsHtml) sel.innerHTML = optionsHtml;
  sel.value = S.sort.key;
  $("#panel-sort-dir").textContent = S.sort.dir > 0 ? "↑" : "↓";
}
$("#panel-sort-key").onchange = e => { S.sort = { key: e.target.value, dir: S.sort.dir }; renderPanels(); };
$("#panel-sort-dir").onclick = () => { S.sort = { key: S.sort.key, dir: -S.sort.dir }; renderPanelSortControl(); renderPanels(); };

function sortVal(s, key) {
  if (key.startsWith("run:")) {
    const [, run, field] = key.split(":");
    const rec = S.runData[run] && S.runData[run].samples[s.sample_id];
    return rec ? rec[field] : null;
  }
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

function renderSamplesView() {
  renderPanelSortControl();
  S.view === "table" ? renderTable() : renderPanels();
}

function rowHtml(s, isGhost) {
  const runTds = S.runs.filter(r => S.runData[r]).map(r => {
    const rec = S.runData[r].samples[s.sample_id];
    if (!rec) return `<td class="run-start" colspan="5" style="color:${BASE}">not in run</td>`;
    return `<td class="run-start">${rec.split.replace("unseen_", "u-")}</td><td>${fmt(rec.mse, 5)}</td><td>${fmt(rec.com_dist)}</td>` +
      (isGhost ? `<td></td><td></td>` :
        `<td><span class="mask-slot" data-run="${r}" data-id="${s.sample_id}" data-mode="pred"></span></td>` +
        `<td><span class="mask-slot" data-run="${r}" data-id="${s.sample_id}" data-mode="diff"></span></td>`);
  }).join("");
  return `<tr data-id="${s.sample_id}" class="${isGhost ? "ghost-row" : ""}">
    <td>${isGhost ? "" : `<span class="row-x" title="remove">✕</span>`}</td>
    <td><span class="overlay-slot" data-id="${s.sample_id}"><img class="thumb" loading="lazy" src="/media/${s.sample_id}/thumb" alt="" data-id="${s.sample_id}"></span></td>
    <td><span class="swatch" style="background:${colorFor(s)}"></span>${s.sample_id}</td>
    <td>${s.speaker}</td><td>${s.layout}</td><td>${s.n_objects}</td>
    <td>(${s.com_row.toFixed(1)}, ${s.com_col.toFixed(1)})</td>
    <td><button class="play" data-src="/media/${s.sample_id}/audio">in ▶</button>
        <button class="play" data-src="/media/${s.sample_id}/recovered">rec ▶</button></td>
    ${runTds}</tr>`;
}

let _tableRenderGen = 0;
async function renderTable() {
  // renderTable is async (awaits network calls) and can be re-entered while a previous call is
  // still in flight (hover -> renderRunsPanel -> ... -> another renderTable, or the 3s live-poll
  // refresh). If an older call finishes last, it overwrites the table with stale data and rewires
  // listeners onto nodes that are about to be discarded — that's what broke the audio play button
  // mid-click. Each call gets a generation id; a call that's no longer the newest bails before
  // touching the DOM.
  const gen = ++_tableRenderGen;
  const wrap = $("#table-wrap");
  const rows = sortedPool();
  const ghostRows = S.ghost ? S.ghost.ids.map(id => SAMPLE.get(id)).filter(s => s && !S.pool.has(s.sample_id)) : [];
  if (!rows.length && !ghostRows.length) {
    if (gen !== _tableRenderGen) return;
    wrap.innerHTML = `<div class="hint" style="padding:12px">add samples — pick a run above, click a position, or use "add matching" / the search box in dataset filters</div>`;
    return;
  }
  for (const run of S.runs) if (S.runData[run]) {
    try { await ensureRunMasks(run, rows.map(s => s.sample_id)); } catch (e) { console.error(e); }
  }
  if (gen !== _tableRenderGen) return;   // a newer render started while we were awaiting masks
  const base = [["", ""], ["", "overhead"], ...BASE_SORT_COLS, ["", "audio"]];
  const runCols = [...RUN_SORT_COLS, ["pred", "pred"], ["diff", "diff"]];
  const arrow = k => S.sort.key === k ? (S.sort.dir > 0 ? " ↑" : " ↓") : "";
  const activeRuns = S.runs.filter(r => S.runData[r]);

  const thead = `<thead>
    <tr class="groups"><th colspan="${base.length}"></th>${activeRuns.map(r =>
      `<th class="run-start" colspan="${runCols.length}" style="color:${runColor(r)}">${r}</th>`).join("")}</tr>
    <tr class="cols">${base.map(([k, l]) => `<th data-key="${k}">${l}${arrow(k)}</th>`).join("")}
    ${activeRuns.map(r => runCols.map(([k, l], i) =>
      `<th ${i === 0 ? 'class="run-start"' : ""} data-key="run:${r}:${k}">${l}${arrow(`run:${r}:${k}`)}</th>`).join("")).join("")}</tr>
  </thead>`;

  const tb = ghostRows.map(s => rowHtml(s, true)).join("") + rows.map(s => rowHtml(s, false)).join("");
  if (gen !== _tableRenderGen) return;   // superseded while building row html (sync, but cheap insurance)
  wrap.innerHTML = `<table class="grid">${thead}<tbody>${tb}</tbody></table>`;

  await fillMaskSlots(wrap);
  fillOverlaySlots(wrap, activeRuns[0]);   // fire-and-forget: don't block the rest of the row wiring on it
  if (gen !== _tableRenderGen) return;   // superseded while awaiting mask thumbnails — don't wire stale nodes
  wrap.querySelectorAll("th[data-key]").forEach(th => th.onclick = () => {
    const k = th.dataset.key;
    if (!k) return;
    S.sort = { key: k, dir: S.sort.key === k ? -S.sort.dir : 1 };
    renderTable();
  });
  wrap.querySelectorAll("tbody tr:not(.ghost-row)").forEach(tr => {
    tr.onmouseenter = () => setGhost({ oid: null, ids: [+tr.dataset.id] }, true);
    tr.onmouseleave = () => setGhost(null, true);
  });
  wrap.querySelectorAll(".row-x").forEach(x => x.onclick = e => {
    e.stopPropagation();
    removeFromPool([+e.target.closest("tr").dataset.id]);
  });
  wrap.querySelectorAll("img.thumb").forEach(img => img.onclick = () => openLightbox(img.src));
  wirePlayButtons(wrap);
  highlightRows();
}

async function fillMaskSlots(root) {
  const slots = [...root.querySelectorAll(".mask-slot")];
  if (!slots.length) return;
  await ensureGtMasks([...new Set(slots.map(sl => +sl.dataset.id))]).catch(console.error);
  for (const sl of slots) {
    const { run, id, mode } = sl.dataset;
    const pred = S.runMasks[run] && S.runMasks[run][+id], gt = S.gtMasks[+id];
    if (!pred) continue;
    const m = mode === "diff" && gt ? pred.map((row, r) => row.map((v, c) => v - gt[r][c])) : pred;
    const cv = maskCanvas(m, mode === "diff" ? "diff" : "pred");
    cv.title = "click for detail view";
    cv.onclick = () => openDetail(+id, run);
    sl.replaceWith(cv);
  }
}

// replaces the plain overhead thumbnail with a prediction-overlay canvas once a run's mask is
// available — that's the primary thing being looked at, so it's the default, not opt-in
async function fillOverlaySlots(root, run) {
  const slots = [...root.querySelectorAll(".overlay-slot")];
  if (!slots.length || !run) return;
  const ids = slots.map(sl => +sl.dataset.id);
  try { await ensureRunMasks(run, ids); } catch (e) { console.error(e); return; }
  for (const sl of slots) {
    const id = +sl.dataset.id;
    const pred = S.runMasks[run] && S.runMasks[run][id];
    if (!pred) continue;
    const cv = await overlayCanvas(id, pred);
    cv.className = "thumb";
    cv.title = "prediction overlaid on overhead";
    cv.onclick = () => openLightbox(`/media/${id}/thumb`);
    if (sl.isConnected) sl.replaceWith(cv);   // table may have re-rendered while this awaited
  }
}

let currentAudio = null, currentAudioBtn = null;
function stopCurrentAudio() {
  if (currentAudio) currentAudio.pause();
  if (currentAudioBtn) currentAudioBtn.classList.remove("playing");
  currentAudio = null; currentAudioBtn = null;
}
function wirePlayButtons(root) {
  root.querySelectorAll(".play").forEach(b => b.onclick = () => {
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
function highlightRows() {
  const ids = new Set(S.ghost ? S.ghost.ids : []);
  document.querySelectorAll("#table-wrap tbody tr:not(.ghost-row)").forEach(tr =>
    tr.classList.toggle("hovered", ids.has(+tr.dataset.id)));
}

async function renderPanels() {
  const wrap = $("#panel-wrap");
  const rows = sortedPool();
  if (!rows.length) { wrap.innerHTML = `<div class="hint" style="padding:12px">add samples to fill this view</div>`; return; }
  for (const run of S.runs) if (S.runData[run]) await ensureRunMasks(run, rows.map(s => s.sample_id)).catch(console.error);
  wrap.innerHTML = "";
  const primaryRun = S.runs.find(r => S.runData[r]);
  for (const s of rows) {
    const card = document.createElement("div");
    card.className = "card";
    card.innerHTML = `<span class="row-x" title="remove">✕</span>
      <div class="meta"><span><span class="swatch" style="background:${colorFor(s)}"></span>${label(s)}</span><span>${s.layout}</span></div>
      <span class="thumb-slot"><img loading="lazy" src="/media/${s.sample_id}/thumb" alt=""></span>`;
    card.querySelector(".row-x").onclick = e => { e.stopPropagation(); removeFromPool([s.sample_id]); };
    const thumbSlot = card.querySelector(".thumb-slot");
    thumbSlot.querySelector("img").onclick = () => openLightbox(`/media/${s.sample_id}/thumb`);
    const primaryMask = primaryRun && S.runMasks[primaryRun] && S.runMasks[primaryRun][s.sample_id];
    if (primaryMask) {
      overlayCanvas(s.sample_id, primaryMask).then(cv => {
        cv.title = "prediction overlaid on overhead";
        cv.style.cssText = "width:100%;border-radius:4px;display:block;cursor:pointer";
        cv.onclick = () => openLightbox(`/media/${s.sample_id}/thumb`);
        if (thumbSlot.isConnected) thumbSlot.replaceWith(cv);
      });
    }
    for (const run of S.runs) {
      const rec = S.runData[run] && S.runData[run].samples[s.sample_id];
      const m = S.runMasks[run] && S.runMasks[run][s.sample_id];
      if (!rec) continue;
      const line = document.createElement("div");
      line.className = "metrics";
      line.innerHTML = `<span style="color:${runColor(run)}">▮</span> ${run} · ${rec.split.replace("unseen_", "u-")} · com ${fmt(rec.com_dist)}`;
      card.appendChild(line);
      if (m && run !== primaryRun) { const cv = maskCanvas(m, "pred"); cv.onclick = () => openDetail(s.sample_id, run); card.appendChild(cv); }
    }
    card.onmouseenter = () => setGhost({ oid: null, ids: [s.sample_id] });
    card.onmouseleave = () => setGhost(null);
    wrap.appendChild(card);
  }
}

function openLightbox(src) {
  $("#lightbox-img").src = src;
  $("#lightbox").classList.add("open");
}
$("#lightbox").onclick = () => $("#lightbox").classList.remove("open");

// ===== runs panel =====
$("#run-metric").onchange = e => { S.runMetric = e.target.value; renderRunsPanel(); };

async function renderRunsPanel() {
  const runs = S.runs.filter(r => S.runData[r]);
  const metric = S.runMetric, metricLabel = metric === "mse" ? "mse" : "com_dist";

  // position-focus strip: show every active run's metric for the cursor/ghost position, all speakers
  const focusOid = S.cursor || (S.ghost && S.ghost.oid);
  const strip = $("#position-focus-strip");
  if (focusOid && runs.length) {
    const samples = GROUPS.get(focusOid) || [];
    const parts = runs.map(r => {
      const vals = samples.map(s => S.runData[r].samples[s.sample_id]?.[metric]).filter(v => v != null);
      const avg = vals.length ? (vals.reduce((a, b) => a + b, 0) / vals.length) : null;
      return `<span style="color:${runColor(r)}">${r}</span> ${fmt(avg)}`;
    });
    strip.innerHTML = `p${+focusOid}: ${parts.join(" · ")}`;
  } else {
    strip.textContent = runs.length ? "hover/select a position to see per-run performance there" : "";
  }

  if (!runs.length) { Plotly.react("runs-plot", [], emptyLayout("add a run (top bar) to see metrics")); return; }
  const splits = [...new Set(runs.flatMap(r => Object.keys(S.runData[r].aggregates)))];
  const traces = [];
  for (const r of runs) {
    traces.push({
      type: "bar", name: r, legendgroup: r, marker: { color: runColor(r) },
      x: splits.map(s => s.replace("unseen_", "u-")), y: splits.map(s => S.runData[r].aggregates[s]?.[metricLabel] ?? null),
      xaxis: "x", yaxis: "y",
      hovertemplate: `${r} · %{x}<br>${metricLabel} %{y:.4f}<extra></extra>`,
    });
    const bySpk = {};
    for (const [sid, rec] of Object.entries(S.runData[r].samples)) {
      if (rec.split === "train" || rec[metricLabel] == null) continue;
      const s = SAMPLE.get(+sid);
      if (!s) continue;
      (bySpk[s.speaker] = bySpk[s.speaker] || []).push(rec[metricLabel]);
    }
    const spks = S.man.facets.speaker;
    traces.push({
      type: "bar", name: r, legendgroup: r, showlegend: false, marker: { color: runColor(r) },
      x: spks.map(String), y: spks.map(k => bySpk[k] ? bySpk[k].reduce((a, b) => a + b) / bySpk[k].length : null),
      xaxis: "x2", yaxis: "y2",
      hovertemplate: `${r} · speaker %{x}<br>mean ${metricLabel} %{y:.4f}<extra></extra>`,
    });
  }
  Plotly.react("runs-plot", traces, LAYOUT({
    barmode: "group", bargap: 0.25,
    margin: { l: 44, r: 10, t: 8, b: 58 },   // extra room: split tick labels ("u-pos_speaker") wrap in a narrow column and clipped the axis title with the default margin
    xaxis: AXIS({ domain: [0, 0.44], title: { text: `${metricLabel} by split`, font: { size: 11 }, standoff: 14 }, tickangle: -30 }),
    yaxis: AXIS({}),
    xaxis2: AXIS({ domain: [0.56, 1], title: { text: `eval ${metricLabel} by speaker`, font: { size: 11 }, standoff: 14 } }),
    yaxis2: AXIS({ anchor: "x2" }),
  }), CONFIG);
}

// ===== detail modal (mask comparison) =====
async function openDetail(sampleId, run) {
  const s = SAMPLE.get(sampleId), rec = S.runData[run]?.samples[sampleId];
  await ensureGtMasks([sampleId]);
  await ensureRunMasks(run, [sampleId]);
  const gt = S.gtMasks[sampleId], pred = S.runMasks[run][sampleId];
  if (!gt || !pred) return;
  $("#modal-title").textContent = `${label(s)} — ${run}`;
  $("#modal-sub").textContent = `${s.layout} · split ${rec?.split ?? "?"} · mse ${fmt(rec?.mse, 6)} · com_dist ${fmt(rec?.com_dist)} · gt com (${s.com_row.toFixed(1)}, ${s.com_col.toFixed(1)}) · pred com (${fmt(rec?.pred_row, 1)}, ${fmt(rec?.pred_col, 1)})`;
  $("#modal").classList.add("open");

  const H = gt.length, W = gt[0].length;
  const comMarks = (ax = "") => [
    { type: "scatter", x: [s.com_col], y: [s.com_row], mode: "markers", showlegend: false, hoverinfo: "skip",
      xaxis: "x" + ax, yaxis: "y" + ax, marker: { symbol: "cross-thin", size: 11, line: { width: 2, color: "#fff" } } },
    ...(rec ? [{ type: "scatter", x: [rec.pred_col], y: [rec.pred_row], mode: "markers", showlegend: false, hoverinfo: "skip",
      xaxis: "x" + ax, yaxis: "y" + ax, marker: { symbol: "x-thin", size: 11, line: { width: 2, color: "#fff" } } }] : []),
  ];
  const axes = over => ({
    xaxis: AXIS({ range: [-0.5, W - 0.5], constrain: "domain", visible: false }),
    yaxis: AXIS({ range: [H - 0.5, -0.5], scaleanchor: over ? "x" : undefined, visible: false }),
    margin: { l: 6, r: 6, t: 22, b: 6 }, showlegend: false,
  });
  const heat = (z, colorscale, name) => ({ type: "heatmap", z, colorscale, showscale: false, name,
    hovertemplate: "(%{y}, %{x}) %{z:.3f}<extra>" + name + "</extra>" });
  const GT_SCALE = [[0, SURFACE], [1, "#18b218"]], PRED_SCALE = [[0, SURFACE], [1, "#e05252"]];
  const DIFF_SCALE = [[0, "#18b218"], [0.5, "#383835"], [1, "#e05252"]];

  Plotly.react("m-side", [
    Object.assign(heat(gt, GT_SCALE, "ground truth"), { xaxis: "x", yaxis: "y" }),
    Object.assign(heat(pred, PRED_SCALE, "prediction"), { xaxis: "x2", yaxis: "y2" }),
    ...comMarks(), ...comMarks("2"),
  ], LAYOUT({
    ...axes(), title: { text: "ground truth (green) · prediction (red)", font: { size: 11 }, x: 0.5 },
    xaxis: AXIS({ domain: [0, 0.485], range: [-0.5, W - 0.5], visible: false }),
    yaxis: AXIS({ range: [H - 0.5, -0.5], visible: false }),
    xaxis2: AXIS({ domain: [0.515, 1], range: [-0.5, W - 0.5], visible: false }),
    yaxis2: AXIS({ range: [H - 0.5, -0.5], anchor: "x2", visible: false }),
  }), CONFIG);

  const diff = pred.map((row, r) => row.map((v, c) => v - gt[r][c]));
  Plotly.react("m-diff", [
    Object.assign(heat(diff, DIFF_SCALE, "pred − gt"), { zmid: 0, zmin: -1, zmax: 1, showscale: true,
      colorbar: { thickness: 8, outlinewidth: 0, tickfont: { color: MUTED } } }),
    ...comMarks(),
  ], LAYOUT({ ...axes(true), title: { text: "diff: red = pred extra · green = gt missed", font: { size: 11 }, x: 0.5 } }), CONFIG);

  const holder = $("#m-overlay");
  holder.innerHTML = `<div class="sub" style="text-align:center;margin:4px 0">overhead + prediction overlay</div>`;
  overlayCanvas(sampleId, pred).then(cv => { cv.style.width = "100%"; cv.style.borderRadius = "6px"; holder.appendChild(cv); });
}

// overhead photo with the predicted mask painted on top in translucent red — this is the
// default thumbnail in table/panel views once a run is loaded, since seeing the prediction
// against the real photo is the main thing being looked at, not the bare photo
function overlayCanvas(sampleId, pred) {
  return new Promise(resolve => {
    const img = new Image();
    img.src = `/media/${sampleId}/thumb`;
    img.onload = () => {
      const H = pred.length, W = pred[0].length;
      const cv = document.createElement("canvas");
      cv.width = img.naturalWidth; cv.height = img.naturalHeight;
      const g = cv.getContext("2d");
      g.drawImage(img, 0, 0);
      const cw = cv.width / W, ch = cv.height / H;
      for (let r = 0; r < H; r++) for (let c = 0; c < W; c++) {
        if (pred[r][c] > 0.05) {
          g.fillStyle = `rgba(224,82,82,${Math.min(pred[r][c] * 0.65, 0.75)})`;
          g.fillRect(c * cw, r * ch, cw + 0.5, ch + 0.5);
        }
      }
      resolve(cv);
    };
  });
}
document.addEventListener("keydown", e => {
  if (e.key === "Escape") { $("#modal").classList.remove("open"); $("#lightbox").classList.remove("open"); setGhost(null); }
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

// ===== fft / runs tab toggle =====
$("#fft-tabs").onclick = e => {
  if (!e.target.dataset.tab) return;
  const tab = e.target.dataset.tab;
  for (const b of $("#fft-tabs").children) b.classList.toggle("active", b === e.target);
  $("#fft-plot").style.display = tab === "fft" ? "" : "none";
  $("#runs-plot-wrap").style.display = tab === "runs" ? "" : "none";
  $("#fft-controls").hidden = tab !== "fft";
  $("#runs-controls").hidden = tab !== "runs";
  // Plotly.react on a just-unhidden container reads its size before the browser has finished
  // laying it out (display:none -> block happens in the same tick), so it renders at a stale
  // size and the axis titles land outside the visible box. Force a resize on the next frame,
  // once layout has actually settled, to pull the plot back to its real container size.
  if (tab === "fft") { updateFFT(); }
  else { renderRunsPanel(); requestAnimationFrame(() => Plotly.Plots.resize("runs-plot")); }
};

// ===== live updates =====
setInterval(async () => {
  try {
    const { version } = await jget("/api/version");
    $("#live-dot").classList.add("on");
    if (version !== S.version) {
      S.version = version;
      S.fftCache.clear(); S.gtMasks = {}; S.runMasks = {};
      for (const r of S.runs) S.runData[r] = await jget(`/api/run/${r}`).catch(() => S.runData[r]);
      await boot();
    }
  } catch { $("#live-dot").classList.remove("on"); }
}, 3000);

window.addEventListener("resize", debounce(() => { renderPosmap(); }, 150));

boot().catch(err => {
  document.body.innerHTML = `<pre style="color:#e66767;padding:24px">failed to load manifest:\n${err}</pre>`;
});
