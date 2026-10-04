// The record GUI: polls /state and draws it; every button POSTs to one notebook function.
// All drawing, zoom and overlays happen here -- Python only sends frames and JSON.
const $ = (s) => document.querySelector(s);
const SVG = "http://www.w3.org/2000/svg";
const STAGE_COLORS = {  // record/'s palette: a sample's stages blue -> orange -> green; the position's own steps after them
  vibrate: "#2980b9", "save vibration": "#e67e22", "post process": "#27ae60", overhead: "#7f8c8d", speaker: "#1a5276" };

const ui = {
  view: { overhead: { x: 0, y: 0, s: 1, fit: true }, laser: { x: 0, y: 0, s: 1, fit: true } },
  overheadCrop: false,  // double-click: the crop, or the whole frame
  drawingCrop: false,   // mid-drag of a new crop box: the view holds still, unclipped
  laserFull: false,     // double-click: the ROIs packed edge to edge, or where they sit on the full sensor
  tab: "timeline",       // the bottom-left pane: timeline, logs or fails
  calib: null,          // ROI calibration in progress: {step, clicks: {rows, crop, cols}}
};
let catalog, state, since = 0, lastPanels = {}, lastAudio = null, lastRoi;
const records = [], spans = new Map(), deleted = new Set();  // deleted: position ids

const post = async (url, body) => {
  const r = await fetch(url, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body ?? {}) });
  $("#msg").textContent = r.ok ? "" : (await r.json().catch(() => ({}))).detail ?? r.statusText;
  return r.ok;
};

// ---------- polling ----------
async function poll() {
  try {
    state = await (await fetch(`/state?since=${since}`)).json();
  } catch (e) {
    $("#dir").textContent = "disconnected -- is the notebook's GUI running?";
    return setTimeout(poll, 1000);
  }
  for (const r of state.records) {
      since = r.seq;
      records.push(r);
      if (r.deleted) deleted.add(r.deleted);
      if (r.stage) spans.set(`${r.label}|${r.stage}|${r.start}`, { ...spans.get(`${r.label}|${r.stage}|${r.start}`), ...r,
        thread: spans.get(`${r.label}|${r.stage}|${r.start}`)?.thread ?? r.thread });
  }
  if (records.length > 3000) records.splice(0, records.length - 3000);
  update();
  setTimeout(poll, 200);
}

const runningSpans = () => [...spans.values()].filter((s) => s.end === undefined);
const fmt = (t) => `${Math.floor(t / 60)}:${(t % 60).toFixed(1).padStart(4, "0")}`;

function update() {
  // each part on its own (the header first: its text sets the panels' height, which the feeds fit to): one that fails (see the browser console) never blanks the others
  for (const part of [drawHeader, drawFeeds, drawPanels, syncSettings, drawTimeline, drawLogs]) {
    try { part(); } catch (e) { console.error(part.name, e); }
  }
}

function drawHeader() {
  const rec = state.recording, now = Date.now() / 1000;
  document.body.classList.toggle("recording", rec);
  const pos = runningSpans().find((s) => s.stage === "position"), vib = runningSpans().find((s) => s.stage === "vibrate");
  // top bar: the experiment dir and its saved positions; per layout on the right
  const n = state.counts.experiment;
  // ‎ keeps the path reading left to right while the rtl box cuts a long one from the left: its end always shows
  $("#dir").textContent = `‎${state.experiment_dir}`;
  $("#dir").title = state.experiment_dir;
  $("#total").textContent = `total ${n}`;
  $("#counts").innerHTML = Object.entries(state.counts.layouts).sort().map(([l, k]) => `<span>${esc(l)}<b>${k}</b></span>`).join("");
  const errors = records.filter((r) => r.level === "ERROR");  // the full log is record.log (and the notebook)
  $("#errors").hidden = !errors.length;
  $("#errors").textContent = `${errors.length} error${errors.length > 1 ? "s" : ""}`;
  $("#errors").title = errors.at(-1)?.msg ?? "";
  // buttons: polls only change text, never the elements -- a click whose element is replaced mid-press is lost
  $("#run-label").textContent = rec ? "Stop" : "Record";
  $("#run-id").textContent = rec ? `${vib?.label ?? pos?.label ?? ""}` : `position ${String(state.next_position_id).padStart(6, "0")}`;
  for (const el of document.querySelectorAll(".rec-time")) el.textContent = `REC ${vib?.label ?? pos?.label ?? ""}  ${pos ? fmt(now - pos.start) : ""}`;  // on both cameras
  $("#run").classList.toggle("stop", rec);
  const last = state.position_id;
  $("#delete").disabled = rec || !last;
  $("#delete-id").textContent = last ?? "";
  for (const el of document.querySelectorAll(".cam input, #calibrate, #roi-width, #roi-height")) el.disabled = rec;
}

// panels: "not run yet" -> "loading…" -> the plot (fetched once per position) or "failed"
function drawPanels() {
  const last = state.position_id;
  const layout = $("#layout").value.trim();
  for (const [panel, status] of Object.entries(state.panels)) {
    // coverage follows the layout field: any layout's saved positions show without a run
    const cov = panel === "coverage", ready = cov ? status !== "loading" : status === "ready";
    const key = cov ? `${last}-${status}-${layout}-${state.counts.layouts[layout] ?? 0}` : `${last}-${status}`;
    if (lastPanels[panel] === key) continue;
    lastPanels[panel] = key;
    const img = $(`#p-${panel}`), fig = img.parentElement, text = fig.querySelector(".status");
    const say = (s) => (text.textContent = s && `${text.dataset.name} · ${s}`);
    say(ready ? "" : { none: "not run yet", loading: "loading…", failed: "failed · see record.log" }[status]);
    if (ready) {  // drawn at exactly its box's size, so the plot fills it
      img.onerror = () => { img.removeAttribute("src"); say(cov ? `no ${layout} positions yet` : "failed · see record.log"); };
      img.src = `/panel/${panel}.png?v=${encodeURIComponent(key)}&layout=${encodeURIComponent(layout)}&w=${fig.clientWidth}&h=${fig.clientHeight}`;
    } else img.removeAttribute("src");
  }
  if (state.panels.shifts === "ready" && lastAudio !== last) { lastAudio = last; $("#audio").src = `/preview.wav?v=${last}`; $("#play").disabled = false; }
}

// ---------- feeds: each a stage (img + svg in sensor px) transformed inside its view ----------
const view = (name) => $(`#${name} .view`);

function fitRect(name, w, h) {
  if (name === "overhead" && ui.overheadCrop) {
    const [l, r, t, b] = readCrop();
    return [l * w, t * h, (r - l) * w, (b - t) * h];
  }
  return [0, 0, w, h];
}

function place(name, w, h) {
  const st = $(`#${name} .stage`), v = ui.view[name];
  if (st.dataset.size !== `${w}x${h}`) {
    st.dataset.size = `${w}x${h}`;
    st.style.width = `${w}px`;
    st.style.height = `${h}px`;
    st.querySelector("svg").setAttribute("viewBox", `0 0 ${w} ${h}`);
    v.fit = true;
  }
  if (v.fit && !ui.drawingCrop) {
    const [x, y, fw, fh] = fitRect(name, w, h), r = view(name).getBoundingClientRect();
    v.s = Math.min(r.width / fw, r.height / fh);
    v.x = (r.width - fw * v.s) / 2 - x * v.s;
    v.y = (r.height - fh * v.s) / 2 - y * v.s;
  }
  applyView(name);
}

// the zoom/pan transform; in the overhead's crop view, the picture is clipped to the box (in sensor px, so it
// follows any zoom or pan) -- the panel itself stays, black around it
function applyView(name) {
  const v = ui.view[name], st = $(`#${name} .stage`), [w, h] = st.dataset.size.split("x").map(Number);
  st.style.transform = `translate(${v.x}px, ${v.y}px) scale(${v.s})`;
  const [l, r, t, b] = name === "overhead" && ui.overheadCrop && !ui.drawingCrop ? readCrop() : [0, 1, 0, 1];
  st.style.clipPath = `inset(${t * h}px ${(1 - r) * w}px ${(1 - b) * h}px ${l * w}px)`;
}

function toSensor(name, e) {
  const r = view(name).getBoundingClientRect(), v = ui.view[name];
  return [(e.clientX - r.left - v.x) / v.s, (e.clientY - r.top - v.y) / v.s];
}

function setSrc(img, src) { if (img.dataset.src !== src) { img.dataset.src = src; img.src = src; } }

function setAttrs(node, attrs) { for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, v); return node; }

function drawFeeds() {
  // overhead: the live frame, the crop box, the speakers (fixed on the panel), lit while one plays
  const [ow, oh] = state.overhead.size;
  setSrc($("#overhead img"), "/overhead.mjpg");
  view("overhead").style.aspectRatio = `${ow} / ${oh}`;
  place("overhead", ow, oh);
  const [l, r, t, b] = readCrop();
  $("#overhead svg").replaceChildren(setAttrs(document.createElementNS(SVG, "rect"),
    { class: "crop", x: l * ow, y: t * oh, width: (r - l) * ow, height: (b - t) * oh }));
  const playing = runningSpans().find((s) => s.stage === "vibrate")?.label.split("-")[1];
  // speakers fixed along the panel's border (26px in), whatever the zoom or crop
  $("#overhead .speakers").replaceChildren(...Object.entries(catalog.speaker_position).map(([spk, [x, y]]) => {
    const d = document.createElement("div");
    d.className = spk === playing ? "spk on" : "spk";
    d.textContent = spk;
    d.style.left = `calc(26px + ${x} * (100% - 52px))`;
    d.style.top = `calc(26px + ${1 - y} * (100% - 52px))`;
    return d;
  }));

  // laser: the panel fills its figure (black); what it shows is fitted + centered in it
  const { mode, size: [lw, lh] } = laserMode(), roi = state.laser.roi;
  setSrc($("#laser img"), "/laser.mjpg");
  place("laser", lw, lh);
  const nodes = [];
  if (mode === "sensor") {
    const c = ui.calib?.clicks ?? { rows: [], crop: [], cols: [] };
    for (const y of c.rows) nodes.push(setAttrs(document.createElementNS(SVG, "line"), { class: "line", x1: 0, x2: lw, y1: y, y2: y }));
    for (const x of [...c.crop, ...c.cols]) nodes.push(setAttrs(document.createElementNS(SVG, "line"), { class: "line", x1: x, x2: x, y1: 0, y2: lh }));
  } else {
    roi.rois.forEach((r, i) => {
      const [x, y, w, h] = laserRect(mode, roi, r, i);
      nodes.push(setAttrs(document.createElementNS(SVG, "rect"), { class: "roi", x, y, width: w, height: h }));
    });
  }
  $("#laser svg").replaceChildren(...nodes);
  $("#laser-hint").textContent = mode === "sensor" ? calibHint() : "";  // only calibrating needs instructions
}

// what the laser panel shows: the ROIs packed edge to edge (the default), the ROIs where they sit on the sensor
// (double-click), or the wide-open sensor while calibrating
function laserMode() {
  const { roi, sensor } = state.laser;
  if (ui.calib || !roi) return { mode: "sensor", size: sensor };
  return ui.laserFull ? { mode: "full", size: sensor } : { mode: "rois", size: packedSize(roi) };
}
const packedSize = (roi) => [roi.cols.length * roi.roi_width, roi.rows.length * roi.roi_height];
// the camera reads one band (the crop's width, an ROI tall) per row of ROIs: each band in the camera's frame
// (x, y, w, h) -> where it sits on the sensor
const frameWidth = (roi) => $("#laser img").naturalWidth || roi.crop[1] - roi.offset_x;  // the crop, rounded out to the camera's 16 px grid
const bands = (roi) => roi.row_positions.map((y, k) => [[0, k * roi.roi_height, frameWidth(roi), roi.roi_height],
                                                         [roi.offset_x, y, frameWidth(roi), roi.roi_height]]);
// the whole sensor's last wide-open picture (fetched again when the grid changes), or null if there's none
const sensorShot = { key: null, img: null };
function sensorImage(roi) {
  const key = JSON.stringify([roi.rows, roi.cols, roi.crop, roi.roi_width, roi.roi_height]);
  if (sensorShot.key !== key) {
    const img = new Image();
    [sensorShot.key, sensorShot.img] = [key, null];
    img.onload = () => { if (sensorShot.key === key) sensorShot.img = img; };
    img.src = `/laser_sensor.jpg?v=${encodeURIComponent(key)}`;
  }
  return sensorShot.img;
}
// ROI i (x, y, w, h in the camera's frame) -> where the panel draws it
function laserRect(mode, roi, [x, y, w, h], i) {
  if (mode === "full") return [x + roi.offset_x, roi.row_positions[Math.floor(y / roi.roi_height)], w, h];
  return [(i % roi.cols.length) * w, Math.floor(i / roi.cols.length) * h, w, h];
}

// every animation frame, the live camera frame is copied onto the canvas the panel shows (no Python work)
function drawLaser() {
  requestAnimationFrame(drawLaser);
  const img = $("#laser img"), canvas = $("#laser canvas");
  if (!state || !img.naturalWidth) return;
  const { mode, size: [w, h] } = laserMode(), roi = state.laser.roi, ctx = canvas.getContext("2d");
  if (canvas.width !== w || canvas.height !== h) [canvas.width, canvas.height] = [w, h];
  if (mode === "sensor") return ctx.drawImage(img, 0, 0);
  ctx.clearRect(0, 0, w, h);
  if (mode === "full") {  // the sensor's last still, darker; the live bands; the ROIs brightest
    const shot = sensorImage(roi);
    ctx.globalAlpha = 0.45;
    if (shot) ctx.drawImage(shot, 0, 0);
    ctx.globalAlpha = 0.7;
    for (const [src, dst] of bands(roi)) ctx.drawImage(img, ...src, ...dst);
    ctx.globalAlpha = 1;
  }
  roi.rois.forEach((r, i) => ctx.drawImage(img, ...r, ...laserRect(mode, roi, r, i)));
}

// zoom/pan never shows past what the view shows (the whole frame, or the overhead's crop box): zoomed out at most
// until it fits, never panned off its edge (an axis it doesn't fill stays centered)
const shown = (name) => fitRect(name, ...$(`#${name} .stage`).dataset.size.split("x").map(Number));
function fitScale(name) {
  const [, , w, h] = shown(name), r = view(name).getBoundingClientRect();
  return Math.min(r.width / w, r.height / h);
}
function keepInFrame(name) {
  const v = ui.view[name], [x, y, w, h] = shown(name), r = view(name).getBoundingClientRect();
  const axis = (pos, start, size, box) => (size <= box ? (box - size) / 2 : Math.min(0, Math.max(box - size, pos + start))) - start;
  v.x = axis(v.x, x * v.s, w * v.s, r.width);
  v.y = axis(v.y, y * v.s, h * v.s, r.height);
}

function zoomPan(name) {
  const v = view(name), vs = ui.view[name];
  v.addEventListener("wheel", (e) => {
    e.preventDefault();
    const r = v.getBoundingClientRect(), mx = e.clientX - r.left, my = e.clientY - r.top;
    const k = Math.max(vs.s * Math.exp(-e.deltaY * 0.0015), fitScale(name)) / vs.s;  // never smaller than what the view shows
    Object.assign(vs, { x: mx - (mx - vs.x) * k, y: my - (my - vs.y) * k, s: vs.s * k, fit: false });
    keepInFrame(name);
    applyView(name);
  }, { passive: false });
  v.addEventListener("pointerdown", (e) => {
    if (!(e.shiftKey || e.button === 1)) return;
    e.preventDefault();
    v.setPointerCapture(e.pointerId);
    let [px, py] = [e.clientX, e.clientY];
    const move = (m) => {
      Object.assign(vs, { x: vs.x + m.clientX - px, y: vs.y + m.clientY - py, fit: false });
      [px, py] = [m.clientX, m.clientY];
      keepInFrame(name);
      applyView(name);
    };
    v.addEventListener("pointermove", move);
    v.addEventListener("pointerup", () => v.removeEventListener("pointermove", move), { once: true });
  });
}

// overhead: drag a crop box; double-click toggles crop view <-> whole frame
function overheadEvents() {
  const v = view("overhead");
  v.addEventListener("pointerdown", (e) => {
    if (e.shiftKey || e.button !== 0) return;
    v.setPointerCapture(e.pointerId);
    const [ow, oh] = state.overhead.size, [ax, ay] = toSensor("overhead", e);
    const move = (m) => {
      const [x, y] = toSensor("overhead", m), c = (val, max) => Math.min(1, Math.max(0, val / max)).toFixed(3);
      if (Math.abs(x - ax) + Math.abs(y - ay) < 6) return;
      ui.drawingCrop = true;
      setCrop([c(Math.min(ax, x), ow), c(Math.max(ax, x), ow), c(Math.min(ay, y), oh), c(Math.max(ay, y), oh)]);
      drawFeeds();
    };
    v.addEventListener("pointermove", move);
    v.addEventListener("pointerup", () => {
      v.removeEventListener("pointermove", move);
      if (!ui.drawingCrop) return;
      ui.drawingCrop = false;  // a new box was drawn: zoom to it, as a double-click would
      ui.overheadCrop = ui.view.overhead.fit = true;
      drawFeeds();
    }, { once: true });
  });
  v.addEventListener("dblclick", () => { ui.overheadCrop = !ui.overheadCrop; ui.view.overhead.fit = true; drawFeeds(); });
}

// laser: calibration clicks; double-click toggles packed ROIs <-> ROIs at their sensor positions
function laserEvents() {
  const v = view("laser");
  v.addEventListener("click", (e) => {
    if (e.shiftKey || !ui.calib) return;
    const [x, y] = toSensor("laser", e);
    calibClick(Math.round(x), Math.round(y));
  });
  v.addEventListener("dblclick", () => {
    if (ui.calib || !state.laser.roi) return;
    ui.laserFull = !ui.laserFull;
    ui.view.laser.fit = true;
    drawFeeds();
  });
}

// ---------- ROI calibration: the notebook's steps -- horizontal lines, the 2 crop edges, vertical lines ----------
const calibN = () => ({ rows: +$("#roi-rows").value, crop: 2, cols: +$("#roi-cols").value });
function calibHint() {
  if (!ui.calib) return "no ROIs yet -- calibrate";
  const { step, clicks } = ui.calib, what = { rows: "HORIZONTAL lines", crop: "CROP edges (left + right)", cols: "VERTICAL lines" }[step];
  return `${what}: ${clicks[step].length}/${calibN()[step]}
Esc cancels`;  // as the notebook's click window
}
async function startCalibration() {
  const previous = state.laser.roi;  // the camera goes wide-open: remember the grid, for Esc
  view("laser").requestFullscreen().catch(() => {});  // the whole sensor, full screen, to click on (during the click: browsers require it)
  if (await post("/calibrate")) {
    ui.calib = { step: "rows", clicks: { rows: [], crop: [], cols: [] }, previous };
    ui.laserFull = false;
    ui.view.laser.fit = true;
  } else if (document.fullscreenElement) document.exitFullscreen();
}
// leaving full screen (Esc, the browser's own key for it) mid-calibration cancels it
document.addEventListener("fullscreenchange", () => {
  ui.view.laser.fit = true;
  if (!document.fullscreenElement && ui.calib) cancelCalibration();
});
async function calibClick(x, y) {
  const c = ui.calib, n = calibN();
  c.clicks[c.step].push(c.step === "rows" ? y : x);
  if (c.clicks[c.step].length < n[c.step]) return drawFeeds();
  if (c.step !== "cols") { c.step = c.step === "rows" ? "crop" : "cols"; return drawFeeds(); }
  const crop = [Math.min(...c.clicks.crop), Math.max(...c.clicks.crop)];
  ui.calib = null;
  if (document.fullscreenElement) document.exitFullscreen();
  await post("/rois", { rows: c.clicks.rows, crop, cols: c.clicks.cols, roi_width: +$("#roi-width").value, roi_height: +$("#roi-height").value });
}
async function cancelCalibration() {
  const roi = ui.calib.previous;  // put the grid from before calibrating back
  ui.calib = null;
  if (document.fullscreenElement) document.exitFullscreen();
  if (roi) await post("/rois", { rows: roi.rows, crop: roi.crop, cols: roi.cols, roi_width: roi.roi_width, roi_height: roi.roi_height });
}
function applyRoiSize() {
  const roi = state.laser.roi;
  if (roi) post("/rois", { rows: roi.rows, crop: roi.crop, cols: roi.cols, roi_width: +$("#roi-width").value, roi_height: +$("#roi-height").value });
}

// ---------- timeline: newest position on top -- its row, then one row per sample ----------
// A position: its main thread (the photo, then a box for every speaker it plays: delay + vibrate; total = the
// position). A sample: its stages as equal boxes with their seconds, the idle time between them (how long the next
// one waited), its total.
const SAMPLE_COLUMNS = ["vibrate", "save vibration", "post process"];  // a sample row's stages, in order
const TRASH = `<svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round"><path d="M3 6h18M8 6V4a1 1 0 0 1 1-1h6a1 1 0 0 1 1 1v2m2 0v14a1 1 0 0 1-1 1H7a1 1 0 0 1-1-1V6"/></svg>`;
const STATUS_ICON = { done: "✓", failed: "✗", deleted: TRASH, running: "…" };
const KEEP_POSITIONS = 50;  // the timeline scrolls through this session's last 50 positions; older ones are in record.log
const secs = (t) => (t < 60 ? `${t.toFixed(1)}s` : `${Math.floor(t / 60)}m${String(Math.round(t % 60)).padStart(2, "0")}s`);

// position id -> its labels ("000012", "000012-3", ...) -> their stages; newest position first
function positions() {
  const byPos = new Map();
  for (const s of spans.values()) {
    const labels = byPos.get(s.label.split("-")[0]) ?? byPos.set(s.label.split("-")[0], new Map()).get(s.label.split("-")[0]);
    (labels.get(s.label) ?? labels.set(s.label, []).get(s.label)).push(s);
  }
  const start = (labels) => Math.min(...[...labels.values()].flat().map((s) => s.start));
  const sorted = [...byPos].sort(([, a], [, b]) => start(b) - start(a));
  const dropped = new Set(sorted.slice(KEEP_POSITIONS).map(([p]) => p));
  for (const [k, s] of spans) if (dropped.has(s.label.split("-")[0])) spans.delete(k);
  return sorted.slice(0, KEEP_POSITIONS);
}

function drawTimeline() {
  if (ui.tab !== "timeline") return;
  const now = Date.now() / 1000, end = (s) => s.end ?? now, dur = (parts) => parts.reduce((t, s) => t + end(s) - s.start, 0);
  const last = (list, stage) => list.filter((s) => s.stage === stage).at(-1);
  // one box: its stages' seconds summed, running while any runs; a name (position rows) goes above the seconds
  const cell = (color, name, parts) => {
    parts = parts.filter(Boolean);
    const label = name ? `<small>${name}</small>` : "", cls = name ? "tbox step" : "tbox";
    if (!parts.length) return `<i class="${cls} wait">${label}</i>`;
    const failed = parts.some((s) => s.failed), running = parts.some((s) => s.end === undefined);
    return `<i class="${cls}${running ? " running" : ""}${failed ? " failed" : ""}" style="--c:${STAGE_COLORS[color]}" ` +
           `title="${parts.map((s) => `${s.stage} ${(end(s) - s.start).toFixed(2)}s`).join(" + ")}">${label}${failed ? "FAIL" : secs(dur(parts))}</i>`;
  };
  const row = (cls, label, middle, total, status) => `<div class="trow ${cls} ${status}" title="${label} · ${status}"><span class="tlab">${label}</span>` +
    `${middle}<span class="ttot">${secs(total)}</span><span class="tst">${STATUS_ICON[status]}</span></div>`;
  // idle: from a column's end to the next one's start -- still counting while the next waits (unless the row stopped)
  const idle = (a, b, status) => {
    if (!a.length || a.some((s) => s.end === undefined) || (!b.length && status !== "running")) return `<span class="tidle"></span>`;
    const next = b.length ? Math.min(...b.map((s) => s.start)) : now;
    return `<span class="tidle" title="idle">${secs(Math.max(0, next - Math.max(...a.map((s) => s.end))))}</span>`;
  };

  $("#timeline").innerHTML = positions().map(([p, labels]) => {
    const gone = deleted.has(p), own = labels.get(p) ?? [], pos = last(own, "position");
    const status = (list, done) => (gone ? "deleted" : list.some((s) => s.failed) ? "failed" : done ? "done" : "running");
    const main = [cell("overhead", "img", [last(own, "overhead")]),
                  ...(pos?.speakers ?? []).map((spk) => cell("speaker", `spk ${spk}`, [last(labels.get(`${p}-${spk}`) ?? [], "speaker")]))];
    let html = row("pos", p, `<div class="tsteps">${main.join("")}</div>`, pos ? end(pos) - pos.start : 0, status(own, pos?.end !== undefined));
    const samples = [...labels].filter(([l]) => l !== p).sort(([a], [b]) => a.split("-")[1] - b.split("-")[1]);
    const expected = pos?.save === false ? 1 : SAMPLE_COLUMNS.length;  // a dry run only vibrates: nothing saved or post-processed
    for (const [label, list] of samples) {
      const st = status(list, list.every((s) => s.end !== undefined) && last(list, SAMPLE_COLUMNS[expected - 1]));
      const cols = SAMPLE_COLUMNS.map((stage) => [last(list, stage)].filter(Boolean));
      const middle = SAMPLE_COLUMNS.map((stage, n) => n >= expected ? `<span class="tidle"></span><i></i>` :
        (n ? idle(cols[n - 1], cols[n], st) : "") + cell(stage, "", cols[n])).join("");
      html += row("", label, middle, Math.max(...list.map(end)) - Math.min(...list.map((s) => s.start)), st);
    }
    return html;
  }).join("");
}

// ---------- logs: every record this session, or only the failures (with their tracebacks); newest first ----------
const esc = (x) => String(x).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);
const clock = (t) => new Date(t * 1000).toTimeString().slice(0, 8);
// light highlighting on escaped text: sample/position ids, durations, quoted strings; in tracebacks also files, lines, the error
const hlMsg = (m) => esc(m)
  .replace(/\b(\d{6}(?:-\d+)?)\b/g, '<span class="h-id">$1</span>')
  .replace(/\b(\d+(?:\.\d+)?s)\b/g, '<span class="h-num">$1</span>')
  .replace(/\b(FAILED|failed)\b/g, '<span class="h-err">$1</span>')
  .replace(/'[^']*'/g, '<span class="h-str">$&</span>');
const hlExc = (t) => esc(t)
  .replace(/File &quot;([^&]+)&quot;, line (\d+), in (\S+)/g,
           'File <span class="h-file">&quot;$1&quot;</span>, line <span class="h-num">$2</span>, in <span class="h-fn">$3</span>')
  .replace(/^(\w+(?:\.\w+)*(?:Error|Exception|Interrupt|Exit|Warning)\b)(:?)/gm, '<span class="h-err">$1</span>$2');
let logsDrawn;
function drawLogs() {
  const fails = records.filter((r) => r.level === "ERROR");
  $("#n-fails").textContent = fails.length || "";
  const key = `${ui.tab}|${since}`;
  if (ui.tab === "timeline" || key === logsDrawn) return;  // redrawn only when there's something new: a text selection survives
  logsDrawn = key;
  const list = (ui.tab === "logs" ? records.slice(-400) : fails).toReversed();
  $(`#${ui.tab}`).innerHTML = list.map((r) =>
    `<div class="lrow ${r.level}"><span class="lt">${clock(r.t)}</span><span class="lth">${esc(r.thread)}</span><span class="lid">${esc(r.sample ?? "")}</span><span class="lm">${hlMsg(r.msg)}</span>` +
    (ui.tab === "fails" && r.exc ? `<pre>${hlExc(r.exc)}</pre>` : "") + "</div>").join("") ||
    `<div class="lrow"><span class="lm">${ui.tab === "logs" ? "nothing logged yet" : "no failures"}</span></div>`;
}
function showTab(tab) {
  ui.tab = tab;
  for (const b of document.querySelectorAll(".tabs button")) b.classList.toggle("on", b.dataset.tab === tab);
  for (const p of document.querySelectorAll("[data-pane]")) p.hidden = p.dataset.pane !== tab;
  logsDrawn = null;
  drawTimeline();
  drawLogs();
}

// ---------- form ----------
const CROP = ["left", "right", "top", "bottom"];
const readCrop = () => CROP.map((k) => +$(`#crop-${k}`).value);
const setCrop = (c) => CROP.forEach((k, i) => ($(`#crop-${k}`).value = c[i]));

function addObject(name = "", count = 1, prompt = "") {
  const row = document.createElement("div");
  row.className = "obj";
  row.innerHTML = `<input list="object-names" placeholder="object"><input type="number" min="1"><input placeholder="prompt"><button title="remove">×</button>`;
  const [n, c, p, x] = row.children;
  [n.value, c.value, p.value] = [name, count, prompt];
  n.addEventListener("change", () => { if (!p.value || p.dataset.auto) { p.value = catalog.prompts[n.value] ?? ""; p.dataset.auto = 1; } });
  p.addEventListener("input", () => delete p.dataset.auto);
  x.addEventListener("click", () => row.remove());
  $("#objects").append(row);
}

function fillForm() {
  const { position: P, preview: V } = catalog;
  $("#box").replaceChildren(...Object.keys(catalog.boxes).map((b) => new Option(b, b)));
  $("#box").value = P.box;
  $("#box").addEventListener("change", () => { setCrop(catalog.boxes[$("#box").value]); ui.view.overhead.fit = true; });
  setCrop(P.crop);
  $("#speakers").innerHTML = catalog.speakers.map((s) => `<label><input type="checkbox" value="${s}" ${P.speakers.includes(s) ? "checked" : ""}>${s}</label>`).join("");
  $("#layout").value = P.layout;
  $("#layouts").innerHTML = catalog.layouts.map((l) => `<option value="${l}">`).join("");
  $("#object-names").innerHTML = Object.keys(catalog.prompts).map((o) => `<option value="${o}">`).join("");
  for (const [name, count] of Object.entries(P.objects)) addObject(name, count, P.prompts[name]);
  while ($("#objects").children.length < 2) addObject();  // room for two objects; a blank row is ignored
  $("#description").value = P.description;
  $("#pv-speaker").replaceChildren(...catalog.speakers.map((s) => new Option(s, s)));
  [$("#pv-speaker").value, $("#pv-laser").value, $("#pv-channel").value, $("#pv-use-pc").checked] = [V.speaker, V.laser, V.channel, V.use_pc];
}

function readForm() {
  const objects = {}, prompts = {};
  for (const row of document.querySelectorAll("#objects .obj")) {
    const [n, c, p] = row.children;
    if (n.value.trim()) { objects[n.value.trim()] = +c.value || 1; prompts[n.value.trim()] = p.value.trim(); }
  }
  return {
    position: { speakers: [...document.querySelectorAll("#speakers input:checked")].map((i) => +i.value), box: $("#box").value,
                crop: readCrop(), objects, prompts, layout: $("#layout").value.trim(), description: $("#description").value },
    preview: { speaker: +$("#pv-speaker").value, laser: +$("#pv-laser").value, channel: +$("#pv-channel").value, use_pc: $("#pv-use-pc").checked },
    save: $("#save").checked, vibrate: $("#vibrate").checked,
  };
}

// camera settings: type a value, then enter (or leave the box) sets it -- clamped to the camera's range, and laser fps
// snapped to its step (a capture stays whole buffers)
function cameraInputs() {
  for (const input of document.querySelectorAll(".cam input")) input.addEventListener("change", () => {
    const cam = input.closest(".cam").dataset.cam, key = input.dataset.key, [lo, hi] = state[cam][`${key}_bounds`], step = state[cam][`${key}_step`];
    let v = Math.min(hi, Math.max(lo, +input.value));
    if (step) v = Math.min(hi, lo + Math.round((v - lo) / step) * step);
    post("/camera", { cam, [key]: v });
  });
}
function syncSettings() {
  for (const input of document.querySelectorAll(".cam input")) {
    if (input === document.activeElement) continue;  // never overwrite what's being typed
    const cam = input.closest(".cam").dataset.cam;
    input.value = +(+state[cam][input.dataset.key]).toFixed(2);
  }
  // the ROI fields: the current grid, or ROIConfig's defaults before there is one -- refreshed when the grid changes
  const roi = state.laser.roi, d = catalog.roi_defaults, key = JSON.stringify(roi && [roi.rows, roi.cols, roi.roi_width, roi.roi_height]);
  if (key !== lastRoi && !document.activeElement?.id?.startsWith("roi-")) {
    lastRoi = key;
    [$("#roi-rows").value, $("#roi-cols").value, $("#roi-width").value, $("#roi-height").value] =
      roi ? [roi.rows.length, roi.cols.length, roi.roi_width, roi.roi_height] : [d.n_rows, d.n_cols, d.roi_width, d.roi_height];
  }
}

// ---------- buttons + keys ----------
async function runOrStop() {
  if (state.recording) return post("/stop");
  if (await post("/run", readForm())) {
    lastPanels = {};
    for (const p of Object.keys(state.panels)) $(`#p-${p}`).removeAttribute("src");
  }
}

async function main() {
  catalog = await (await fetch("/catalog")).json();
  state = await (await fetch("/state")).json();
  since = 0;
  fillForm();
  cameraInputs();
  zoomPan("overhead"); zoomPan("laser");
  overheadEvents(); laserEvents();
  drawLaser();
  $("#run").addEventListener("click", runOrStop);
  $("#play").addEventListener("click", () => ($("#audio").paused ? $("#audio").play() : $("#audio").pause()));
  for (const e of ["play", "pause", "ended"]) $("#audio").addEventListener(e, () => {
    $("#play").textContent = $("#audio").paused ? "▶ recovered audio" : "❚❚ recovered audio";
    post("/log", { msg: `recovered audio ${e} (${$("#audio").currentSrc.split("/").pop()}) on the default output` });  // in record.log: it shares the MOTU
  });
  $("#add-object").addEventListener("click", () => addObject());
  $("#delete").addEventListener("click", () => {
    if (confirm(`Delete position ${state.position_id}? Its samples move to deleted/.`)) post("/delete", { position_id: state.position_id });
  });
  $("#calibrate").addEventListener("click", startCalibration);
  for (const b of document.querySelectorAll(".tabs button")) b.addEventListener("click", () => showTab(b.dataset.tab));
  $("#errors").addEventListener("click", () => showTab("fails"));
  // the stage colors, in column order, as in record/: each a colored chip naming its stage
  $("#legend").innerHTML = Object.keys(STAGE_COLORS).map((k) => `<b style="background:${STAGE_COLORS[k]}">${k}</b>`).join("");
  for (const id of ["#roi-width", "#roi-height"]) $(id).addEventListener("change", applyRoiSize);  // on enter, blur or a spinner step
  for (const id of CROP) $(`#crop-${id}`).addEventListener("input", () => drawFeeds());
  let resized;
  window.addEventListener("resize", () => {
    ui.view.overhead.fit = ui.view.laser.fit = true;
    clearTimeout(resized);
    resized = setTimeout(() => (lastPanels = {}), 300);  // redraw the plots at the new size
  });
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && ui.calib) return cancelCalibration();
    if (e.target.closest("input, textarea, select")) return;  // keys belong to the field being edited
    const cam = e.target.closest(".feed")?.id, arrow = { ArrowLeft: [1, 0], ArrowRight: [-1, 0], ArrowUp: [0, 1], ArrowDown: [0, -1] }[e.key];
    if (cam && arrow) {  // a clicked camera: the arrows pan it, a tenth of the panel a press
      e.preventDefault();
      const v = ui.view[cam], r = view(cam).getBoundingClientRect();
      Object.assign(v, { x: v.x + arrow[0] * r.width / 10, y: v.y + arrow[1] * r.height / 10, fit: false });
      keepInFrame(cam);
      return applyView(cam);
    }
    if (["PageUp", "PageDown", "ArrowLeft", "ArrowRight", "Enter"].includes(e.key)) { e.preventDefault(); runOrStop(); }  // the clicker
  });
  poll();
}
main();
