// The record GUI: polls /state and draws it; every button POSTs to one notebook function.
// All drawing, zoom and overlays happen here -- Python only sends frames and JSON.
const $ = (s) => document.querySelector(s);
const SVG = "http://www.w3.org/2000/svg";
const STAGE_COLORS = { overhead: "#86867e", segment: "#2f8f5f", "segment wait": "#b8b7b1", vibrate: "#d64541",
  "save vibration": "#256abf", preview: "#8e44ad", "save segmentation": "#b06a2c", "post process": "#e6a817" };
const PALETTE = ["#256abf", "#2f8f5f", "#b06a2c", "#8e44ad", "#d64541", "#16a2b8", "#e6a817", "#55554f"];

const ui = {
  view: { overhead: { x: 0, y: 0, s: 1, fit: true }, laser: { x: 0, y: 0, s: 1, fit: true } },
  overheadCrop: false,  // double-click: the crop, or the whole frame
  drawingCrop: false,   // mid-drag of a new crop box: the view holds still, unclipped
  laserFull: false,     // double-click: the ROIs packed edge to edge, or where they sit on the full sensor
  timeline: "sample",
  calib: null,          // ROI calibration in progress: {step, clicks: {rows, crop, cols}}
};
let catalog, state, since = 0, lastPanels = {}, lastAudio = null, lastRoi;
const records = [], spans = new Map();

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
  // each part on its own, the feeds first: one that fails (see the browser console) never blanks the others
  for (const part of [drawFeeds, drawHeader, drawPanels, syncSliders, drawTimeline]) {
    try { part(); } catch (e) { console.error(part.name, e); }
  }
}

function drawHeader() {
  const rec = state.recording, now = Date.now() / 1000;
  document.body.classList.toggle("recording", rec);
  const pos = runningSpans().find((s) => s.stage === "position"), vib = runningSpans().find((s) => s.stage === "vibrate");
  // top bar: the experiment dir; saved positions per layout
  $("#dir").textContent = state.experiment_dir;
  $("#counts").innerHTML = Object.entries(state.counts.layouts).sort().map(([l, n]) => `${l} <b>${n}</b>`).join(" · ");
  const errors = records.filter((r) => r.level === "ERROR");  // the full log is record.log (and the notebook)
  $("#errors").hidden = !errors.length;
  $("#errors").textContent = `${errors.length} error${errors.length > 1 ? "s" : ""} · see record.log`;
  $("#errors").title = errors.at(-1)?.msg ?? "";
  // buttons: polls only change text, never the elements -- a click whose element is replaced mid-press is lost
  const n = state.counts.experiment;
  $("#run-label").textContent = rec ? "Stop" : "Record";
  $("#run-id").textContent = rec ? `${vib?.label ?? pos?.label ?? ""}` : $("#save").checked ? `position ${String(state.next_position_id).padStart(6, "0")}` : "dry run";
  $("#run-sub").textContent = rec ? `recording ${pos ? fmt(now - pos.start) : ""}` : `${n} position${n === 1 ? "" : "s"} recorded`;
  $("#run").classList.toggle("stop", rec);
  const last = state.position_id;
  $("#delete").disabled = rec || !last || last === "dry";
  $("#delete-id").textContent = last && last !== "dry" ? last : "";
  for (const el of document.querySelectorAll("#sliders input, #calibrate, #roi-apply")) el.disabled = rec;
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

// the zoom/pan transform; in the overhead's crop view, the panel is also clipped to the box's on-screen
// rectangle -- nothing outside the box ever shows, however you zoom or pan
function applyView(name) {
  const v = ui.view[name], [w, h] = $(`#${name} .stage`).dataset.size.split("x").map(Number);
  $(`#${name} .stage`).style.transform = `translate(${v.x}px, ${v.y}px) scale(${v.s})`;
  let clip = "";
  if (name === "overhead" && ui.overheadCrop && !ui.drawingCrop) {
    const [l, r, t, b] = readCrop(), box = view(name).getBoundingClientRect(), px = (n) => `${Math.max(0, n)}px`;
    clip = `inset(${px(v.y + t * h * v.s)} ${px(box.width - v.x - r * w * v.s)} ${px(box.height - v.y - b * h * v.s)} ${px(v.x + l * w * v.s)} round 6px)`;
  }
  view(name).style.clipPath = clip;
}

function toSensor(name, e) {
  const r = view(name).getBoundingClientRect(), v = ui.view[name];
  return [(e.clientX - r.left - v.x) / v.s, (e.clientY - r.top - v.y) / v.s];
}

function setSrc(img, src) { if (img.dataset.src !== src) { img.dataset.src = src; img.src = src; } }

function setAttrs(node, attrs) { for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, v); return node; }

function drawFeeds() {
  // overhead: the live frame, the crop box, the speakers (pinned to the panel), lit while one plays
  const [ow, oh] = state.overhead.size;
  setSrc($("#overhead img"), "/overhead.mjpg");
  view("overhead").style.aspectRatio = `${ow} / ${oh}`;
  place("overhead", ow, oh);
  const [l, r, t, b] = readCrop();
  $("#overhead svg").replaceChildren(setAttrs(document.createElementNS(SVG, "rect"),
    { class: "crop", x: l * ow, y: t * oh, width: (r - l) * ow, height: (b - t) * oh }));
  const playing = runningSpans().find((s) => s.stage === "vibrate")?.label.split("-")[1];
  // speakers along the edges of what's visible: the panel, or (crop view) the box's on-screen rectangle
  const v = ui.view.overhead, panel = view("overhead").getBoundingClientRect(), m = 26;
  const [x0, y0, x1, y1] = ui.overheadCrop
    ? [Math.max(0, v.x + l * ow * v.s), Math.max(0, v.y + t * oh * v.s), Math.min(panel.width, v.x + r * ow * v.s), Math.min(panel.height, v.y + b * oh * v.s)]
    : [0, 0, panel.width, panel.height];
  $("#overhead .speakers").replaceChildren(...Object.entries(catalog.speaker_position).map(([spk, [x, y]]) => {
    const d = document.createElement("div");
    d.className = spk === playing ? "spk on" : "spk";
    d.textContent = spk;
    d.style.left = `${x0 + m + x * (x1 - x0 - 2 * m)}px`;
    d.style.top = `${y0 + m + (1 - y) * (y1 - y0 - 2 * m)}px`;
    return d;
  }));

  // laser: the panel takes the shape of what it shows, so no black bars pass for camera
  const { mode, size: [lw, lh] } = laserMode(), roi = state.laser.roi;
  setSrc($("#laser img"), "/laser.mjpg");
  shapeView("laser", lw, lh);
  place("laser", lw, lh);
  const nodes = [];
  if (mode === "sensor") {
    const c = ui.calib?.clicks ?? { rows: [], crop: [], cols: [] };
    for (const y of c.rows) nodes.push(setAttrs(document.createElementNS(SVG, "line"), { class: "line", x1: 0, x2: lw, y1: y, y2: y }));
    for (const x of [...c.crop, ...c.cols]) nodes.push(setAttrs(document.createElementNS(SVG, "line"), { class: "line", x1: x, x2: x, y1: 0, y2: lh }));
  } else {
    roi.rois.forEach((r, i) => {
      const [x, y, w, h] = laserRect(mode, roi, r, i);
      nodes.push(setAttrs(document.createElementNS(SVG, "rect"), { class: i === +$("#pv-laser").value ? "roi pv" : "roi", x, y, width: w, height: h }));
    });
  }
  $("#laser svg").replaceChildren(...nodes);
  $("#laser-hint").textContent = mode === "sensor" ? calibHint() :
    `${mode === "full" ? "ROIs on the full sensor" : "ROIs"} · click an ROI to preview it · double-click ${mode === "full" ? "ROIs only" : "full sensor"}`;
}

// what the laser panel shows: the ROIs packed edge to edge (the default), the ROIs where they sit on the sensor
// (double-click), or the wide-open sensor while calibrating
function laserMode() {
  const { roi, sensor } = state.laser;
  if (ui.calib || !roi) return { mode: "sensor", size: sensor };
  if (ui.laserFull) return { mode: "full", size: sensor };
  return { mode: "rois", size: [roi.cols.length * roi.roi_width, roi.rows.length * roi.roi_height] };
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
  ctx.fillStyle = "#000";
  ctx.fillRect(0, 0, w, h);
  roi.rois.forEach((r, i) => ctx.drawImage(img, ...r, ...laserRect(mode, roi, r, i)));
}

// size a view to its picture's aspect, as big as its figure allows
function shapeView(name, w, h) {
  const fig = $(`#${name}`), v = view(name);
  const s = Math.min(fig.clientWidth / w, (fig.clientHeight - fig.querySelector("figcaption").offsetHeight) / h);
  const [vw, vh] = [`${Math.floor(w * s)}px`, `${Math.floor(h * s)}px`];
  if (v.style.width !== vw || v.style.height !== vh) { [v.style.width, v.style.height] = [vw, vh]; ui.view[name].fit = true; }
}

// zoom/pan never shows past the frame: zoomed out at most until the whole frame fits, never panned off an edge
// (an axis the frame doesn't fill stays centered)
const frameSize = (name) => $(`#${name} .stage`).dataset.size.split("x").map(Number);
function fitScale(name) {
  const [w, h] = frameSize(name), r = view(name).getBoundingClientRect();
  return Math.min(r.width / w, r.height / h);
}
function keepInFrame(name) {
  const v = ui.view[name], [w, h] = frameSize(name), r = view(name).getBoundingClientRect();
  const axis = (pos, size, box) => (size <= box ? (box - size) / 2 : Math.min(0, Math.max(box - size, pos)));
  v.x = axis(v.x, w * v.s, r.width);
  v.y = axis(v.y, h * v.s, r.height);
}

function zoomPan(name) {
  const v = view(name), vs = ui.view[name];
  v.addEventListener("wheel", (e) => {
    e.preventDefault();
    const r = v.getBoundingClientRect(), mx = e.clientX - r.left, my = e.clientY - r.top;
    const k = Math.max(vs.s * Math.exp(-e.deltaY * 0.0015), fitScale(name)) / vs.s;  // never smaller than the whole frame
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

// laser: calibration clicks, or click an ROI to preview it; double-click toggles live rows <-> full sensor
function laserEvents() {
  const v = view("laser");
  v.addEventListener("click", (e) => {
    if (e.shiftKey) return;
    let [x, y] = toSensor("laser", e);
    if (ui.calib) return calibClick(Math.round(x), Math.round(y));
    const { mode } = laserMode(), roi = state.laser.roi;
    if (mode === "sensor") return;
    const i = roi.rois.findIndex((r, j) => { const [rx, ry, w, h] = laserRect(mode, roi, r, j); return x >= rx && x < rx + w && y >= ry && y < ry + h; });
    if (i >= 0) { $("#pv-laser").value = i; drawFeeds(); }
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
  const { step, clicks } = ui.calib, what = { rows: "horizontal lines", crop: "crop edges (left + right)", cols: "vertical lines" }[step];
  return `calibrating: click ${calibN()[step]} ${what} (${clicks[step].length}/${calibN()[step]}) · Esc cancels`;
}
async function startCalibration() {
  const previous = state.laser.roi;  // the camera goes wide-open: remember the grid, for Esc
  if (await post("/calibrate")) {
    ui.calib = { step: "rows", clicks: { rows: [], crop: [], cols: [] }, previous };
    ui.laserFull = false;
    ui.view.laser.fit = true;
  }
}
async function calibClick(x, y) {
  const c = ui.calib, n = calibN();
  c.clicks[c.step].push(c.step === "rows" ? y : x);
  if (c.clicks[c.step].length < n[c.step]) return drawFeeds();
  if (c.step !== "cols") { c.step = c.step === "rows" ? "crop" : "cols"; return drawFeeds(); }
  const crop = [Math.min(...c.clicks.crop), Math.max(...c.clicks.crop)];
  ui.calib = null;
  await post("/rois", { rows: c.clicks.rows, crop, cols: c.clicks.cols, roi_width: +$("#roi-width").value, roi_height: +$("#roi-height").value });
}
async function cancelCalibration() {
  const roi = ui.calib.previous;  // put the grid from before calibrating back
  ui.calib = null;
  if (roi) await post("/rois", { rows: roi.rows, crop: roi.crop, cols: roi.cols, roi_width: roi.roi_width, roi_height: roi.roi_height });
}
function applyRoiSize() {
  const roi = state.laser.roi;
  if (roi) post("/rois", { rows: roi.rows, crop: roi.crop, cols: roi.cols, roi_width: +$("#roi-width").value, roi_height: +$("#roi-height").value });
}

// ---------- timeline: rows = samples (colored by stage) or threads (colored by sample) ----------
function color(s) {
  if (ui.timeline === "sample") return STAGE_COLORS[s.stage] ?? "#86867e";
  let h = 0;
  for (const ch of s.label) h = (h * 31 + ch.charCodeAt(0)) >>> 0;
  return PALETTE[h % PALETTE.length];
}
function drawTimeline() {
  const now = Date.now() / 1000;
  for (const [k, s] of spans) if (s.end !== undefined && now - s.end > 900) spans.delete(k);  // keep 15 min
  const key = ui.timeline === "sample" ? (s) => s.label : (s) => s.thread;
  const rows = new Map();
  for (const s of spans.values()) (rows.get(key(s)) ?? rows.set(key(s), []).get(key(s))).push(s);
  const latest = (list) => Math.max(...list.map((s) => s.end ?? now));
  const keys = [...rows.keys()].sort((a, b) => latest(rows.get(b)) - latest(rows.get(a))).slice(0, 14);
  const shown = keys.flatMap((k) => rows.get(k));
  if (!shown.length) return $("#timeline").replaceChildren();
  // by sample: every row starts at its own first stage, all on one scale (the longest row); by thread: the last minute
  const first = (list) => Math.min(...list.map((s) => s.start));
  const span = ui.timeline === "sample" ? Math.max(1, ...keys.map((k) => latest(rows.get(k)) - first(rows.get(k)))) : 60;
  $("#timeline").replaceChildren(...keys.map((k) => {
    const row = document.createElement("div"), lab = document.createElement("span"), bar = document.createElement("div");
    row.className = "trow"; lab.className = "tlab"; bar.className = "tbar";
    lab.textContent = k;
    const t0 = ui.timeline === "sample" ? first(rows.get(k)) : now - 60;
    for (const s of rows.get(k)) {
      const i = document.createElement("i"), end = s.end ?? now;
      if (end < t0) continue;
      const speaker = s.label.includes("-") ? ` ${s.label.split("-")[1]}` : "";
      i.className = [["position", "speaker"].includes(s.stage) && "outer", s.end === undefined && "running", s.failed && "failed"].filter(Boolean).join(" ");
      i.style.left = `${Math.max(0, (s.start - t0) / span) * 100}%`;
      i.style.width = `${((end - Math.max(s.start, t0)) / span) * 100}%`;
      i.style.background = color(s);
      i.title = `${s.stage}${speaker} · ${s.label} · ${s.thread} · ${(end - s.start).toFixed(2)}s${s.failed ? " · FAILED" : ""}`;
      bar.append(i);
    }
    row.append(lab, bar);
    return row;
  }));
  $("#legend").innerHTML = ui.timeline === "sample"
    ? Object.entries(STAGE_COLORS).map(([k, c]) => `<span><b style="background:${c}"></b>${k}</span>`).join("")
    : "<span>bars colored by sample · hover for details</span>";
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

// camera sliders: POST on release ('change'), never while dragging
const SLIDERS = [["overhead", "exposure", "exposure (ms)"], ["overhead", "gain", "gain"], ["laser", "exposure", "exposure (µs)"], ["laser", "gain", "gain"]];
function makeSliders() {
  $("#sliders").replaceChildren(...SLIDERS.map(([cam, key, label]) => {
    const row = document.createElement("label"), [lo, hi] = state[cam][`${key}_bounds`];
    row.className = "slider";
    row.innerHTML = `<span>${cam} ${label}</span><small>${lo}</small><input type="range" min="${lo}" max="${hi}" step="${(hi - lo) / 1000}"><small>${hi}</small><output></output>`;
    const input = row.querySelector("input"), out = row.querySelector("output");
    input.dataset.cam = cam; input.dataset.key = key;
    input.addEventListener("input", () => (out.value = (+input.value).toFixed(1)));
    input.addEventListener("change", () => post("/camera", { cam, [key]: +input.value }));
    return row;
  }));
}
function syncSliders() {
  for (const input of document.querySelectorAll("#sliders input")) {
    if (input === document.activeElement) continue;
    input.value = state[input.dataset.cam][input.dataset.key];
    input.closest(".slider").querySelector("output").value = (+input.value).toFixed(1);
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
  makeSliders();
  zoomPan("overhead"); zoomPan("laser");
  overheadEvents(); laserEvents();
  drawLaser();
  $("#run").addEventListener("click", runOrStop);
  $("#play").addEventListener("click", () => ($("#audio").paused ? $("#audio").play() : $("#audio").pause()));
  for (const e of ["play", "pause", "ended"]) $("#audio").addEventListener(e, () => ($("#play").textContent = $("#audio").paused ? "▶ recovered audio" : "❚❚ recovered audio"));
  $("#add-object").addEventListener("click", () => addObject());
  $("#delete").addEventListener("click", () => {
    if (confirm(`Delete position ${state.position_id}? Its samples move to deleted/.`)) post("/delete", { position_id: state.position_id });
  });
  $("#calibrate").addEventListener("click", startCalibration);
  $("#roi-apply").addEventListener("click", applyRoiSize);
  for (const b of document.querySelectorAll(".timeline .toggle button")) b.addEventListener("click", () => {
    ui.timeline = b.dataset.view;
    for (const o of document.querySelectorAll(".timeline .toggle button")) o.classList.toggle("on", o === b);
  });
  for (const id of CROP) $(`#crop-${id}`).addEventListener("input", () => drawFeeds());
  $("#pv-laser").addEventListener("input", () => drawFeeds());
  let resized;
  window.addEventListener("resize", () => {
    ui.view.overhead.fit = ui.view.laser.fit = true;
    clearTimeout(resized);
    resized = setTimeout(() => (lastPanels = {}), 300);  // redraw the plots at the new size
  });
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && ui.calib) return cancelCalibration();
    if (e.target.closest("input, textarea, select")) return;  // keys belong to the field being edited
    if (["PageUp", "PageDown", "ArrowLeft", "ArrowRight", "Enter"].includes(e.key)) { e.preventDefault(); runOrStop(); }  // the clicker
  });
  poll();
}
main();
