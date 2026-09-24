// Scenario player (Figure 3 simulation) and replay comparison (Figure 4
// recording). Both show precomputed per-spike diagnostics from the paper's
// pipeline on a shared time axis, with a cursor that inspects one spike.

import {
  cssVar,
  DistributionChart,
  heatmapBitmap,
  paintDots,
  paintColumns,
  paintHeatmap,
  paintPositionLine,
  paintShading,
  paintThreshold,
  TrackStack,
} from "./charts.js";
import { badge, decodeRows, describeRule, loadJSON, METRICS, nearestIndex } from "./data.js";

// Seconds of real time to play through one window.
const PLAYBACK_SECONDS = 15;

const SCENARIO_TEXT = {
  well_specified:
    "Model and data agree. The few spikes flagged here show the false-positive rate that each threshold allows.",
  remap:
    "The decoder evaluates spikes against a scrambled set of place fields, as after global remapping. The prediction and the spike likelihoods disagree about where the animal is, and all three diagnostics flag many spikes.",
  history_dependent:
    "Spikes are generated with refractoriness and bursting, but the decoder assumes Poisson firing. Each spike still carries the same spatial information, so decoding stays accurate and the diagnostics rarely flag: they test spatial consistency, not spike timing.",
  replay:
    "The animal sits still while the ensemble replays a sweep across the track. The decoder follows the sweep, far from the animal's physical position, yet the prediction stays consistent with the spikes, and flags stay at the well-specified rate. The large decoding error is by construction: it is measured against the physical position.",
  drift:
    "The animal moves with momentum while the decoder assumes a memoryless random walk. The decoded position lags behind and decoding error rises, but each spike is compared with a prediction that earlier spikes have already pulled toward the lagged estimate, so only a modest share of spikes is flagged — a limitation of event-level checks.",
  sparse_population:
    "The ordinary ensemble falls quiet and a few narrow cells near the animal fire occasionally. Between spikes the prediction spreads out; each narrow spike lands inside it. HPD overlap and the predictive check see consistency, while KL divergence flags the difference in spread, even though decoding stays accurate.",
};

// Conditions whose flagged spikes are the point open on one; the rest open on
// a typical, unflagged spike.
const OPEN_ON_FLAGGED = new Set(["remap", "drift", "sparse_population"]);

const SCALE_NOTE =
  "Prediction heatmaps use one color scale per panel, as in the paper's figures. Likelihood columns and the curves in the spike panel are each scaled to their own maximum.";

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

function normalizedRow(row) {
  let total = 0;
  for (const v of row) total += v;
  return Array.from(row, (v) => (total > 0 ? v / total : 0));
}

/** Map a position to the heatmap's y coordinate (bins drawn at equal heights). */
function positionScale(bins, height) {
  return (position) => height - ((fractionalIndex(bins, position) + 0.5) / bins.length) * height;
}

/** Position expressed in bin-index units, interpolating between bin centers. */
function fractionalIndex(bins, value) {
  const last = bins.length - 1;
  if (value <= bins[0]) return 0;
  if (value >= bins[last]) return last;
  let lo = 0;
  let hi = last;
  while (hi - lo > 1) {
    const mid = (lo + hi) >> 1;
    if (bins[mid] <= value) lo = mid;
    else hi = mid;
  }
  return lo + (value - bins[lo]) / (bins[hi] - bins[lo]);
}

function metricRange(metric, values, rule) {
  if (metric.name === "hpd_overlap") return [0, 1];
  let max = 0;
  for (const v of values) max = Math.max(max, metric.display(v));
  const threshold = rule ? metric.display(rule.threshold) : 0;
  return [0, Math.max(max * 1.05, threshold * 1.4, 1)];
}

/** Track showing one metric per spike, for one or two series. */
function metricTrack(metric, series, rule, shading) {
  const allValues = series.flatMap((s) => s.values);
  const [lo, hi] = metricRange(metric, allValues, rule);
  const pad = 4;
  return {
    label: metric.displayLabel,
    top: hi >= 10 ? hi.toFixed(0) : String(Number(hi.toFixed(1))),
    bottom: "0",
    height: 58,
    draw(context, width, height, xOf) {
      const axis = metric.axis ?? ((v) => v);
      const [aLo, aHi] = [axis(lo), axis(hi)];
      const yOf = (v) =>
        height - pad - ((axis(metric.display(v)) - aLo) / (aHi - aLo)) * (height - 2 * pad);
      if (shading) paintShading(context, shading, xOf, height, cssVar("--surface-sunken"));
      for (const tick of metric.gridlines ?? []) {
        const y = Math.round(yOf(tick)) + 0.5;
        context.strokeStyle = cssVar("--grid");
        context.lineWidth = 1;
        context.beginPath();
        context.moveTo(0, y);
        context.lineTo(width, y);
        context.stroke();
        context.fillStyle = cssVar("--text-muted");
        context.font = `9px ${cssVar("--font")}`;
        context.textBaseline = "bottom";
        context.fillText(String(tick), 2, y - 1);
      }
      if (rule) paintThreshold(context, yOf(rule.threshold), width);
      const color = cssVar(metric.color.slice(4, -1));
      for (const s of series) paintDots(context, s.times, s.values, xOf, yOf, color, s.style);
    },
  };
}

function rasterTrack(times, rows, nRows, shading, label = "Cells") {
  return {
    label,
    top: "",
    bottom: "",
    height: Math.min(96, Math.max(48, nRows * 3)),
    draw(context, width, height, xOf) {
      if (shading) paintShading(context, shading, xOf, height, cssVar("--surface-sunken"));
      context.fillStyle = cssVar("--text");
      const rowHeight = height / nRows;
      for (let i = 0; i < times.length; i += 1) {
        const y = height - (rows[i] + 1) * rowHeight;
        context.fillRect(Math.round(xOf(times[i])), y, 1, Math.max(1, rowHeight - 0.5));
      }
    },
  };
}

function rankOf(values) {
  const order = values.map((v, i) => [v, i]).sort((a, b) => a[0] - b[0]);
  const rank = new Array(values.length);
  order.forEach(([, index], r) => {
    rank[index] = r;
  });
  return rank;
}

function readoutCard(metric, rule) {
  const card = document.createElement("div");
  card.className = "readout";
  card.style.setProperty("--metric-color", metric.color);
  card.innerHTML = `<div><div class="name">${metric.label}</div><div class="rule">${describeRule(metric.name, rule)}</div></div><div class="value"></div>`;
  return card;
}

/** Play/pause and step buttons plus a time readout. */
function transport(container, { onStep, onPlay }) {
  const bar = document.createElement("div");
  bar.className = "transport";
  bar.innerHTML = `
    <button class="chip" data-step="-1" type="button" aria-label="Previous spike">◀ Prev spike</button>
    <button class="chip" data-play type="button">▶ Play</button>
    <button class="chip" data-step="1" type="button" aria-label="Next spike">Next spike ▶</button>
    <span class="time" aria-live="polite"></span>`;
  container.appendChild(bar);
  for (const button of bar.querySelectorAll("[data-step]")) {
    button.addEventListener("click", () => onStep(Number(button.dataset.step)));
  }
  const play = bar.querySelector("[data-play]");
  play.addEventListener("click", () => onPlay(play));
  return { time: bar.querySelector(".time"), play };
}

/** Animate a cursor across [t0, t1] in PLAYBACK_SECONDS; returns a stop function. */
function animate(range, from, setTime, onDone) {
  const [t0, t1] = range;
  const rate = (t1 - t0) / PLAYBACK_SECONDS;
  let last = null;
  let time = from >= t1 ? t0 : from;
  let frame = null;
  const tick = (now) => {
    if (last !== null) time += ((now - last) / 1000) * rate;
    last = now;
    if (time >= t1) {
      setTime(t1);
      onDone();
      return;
    }
    setTime(time);
    frame = requestAnimationFrame(tick);
  };
  frame = requestAnimationFrame(tick);
  return () => cancelAnimationFrame(frame);
}

function wirePlayback(stack, range, eventTimes, select, controls) {
  let stop = null;
  let current = null;
  const setTime = (time) => {
    stack.placeCursor(time);
    const index = nearestIndex(eventTimes, time);
    if (index !== current) {
      current = index;
      select(index, time);
    }
  };
  const halt = () => {
    if (stop) stop();
    stop = null;
    controls.play.textContent = "▶ Play";
  };
  return {
    setTime: (time) => {
      halt();
      setTime(time);
    },
    // Select by index, not time: spikes in one time bin share a time.
    selectIndex: (index) => {
      halt();
      current = index;
      stack.placeCursor(eventTimes[index]);
      select(index, eventTimes[index]);
    },
    step: (direction) => {
      halt();
      const next = Math.min(eventTimes.length - 1, Math.max(0, (current ?? -1) + direction));
      current = next;
      stack.placeCursor(eventTimes[next]);
      select(next, eventTimes[next]);
    },
    toggle: () => {
      if (stop) {
        halt();
        return;
      }
      controls.play.textContent = "❚❚ Pause";
      stop = animate(range, stack.cursorTime ?? range[0], setTime, halt);
    },
    halt,
  };
}

// ---------------------------------------------------------------------------
// Scenario player
// ---------------------------------------------------------------------------

export function initScenarios(root, manifest) {
  const tabs = root.querySelector("#sc-tabs");
  const view = root.querySelector("#sc-view");
  const text = root.querySelector("#sc-text");
  const cache = new Map();
  const macros = manifest.macros;
  const stepSeconds = Number(macros.SimDurationSeconds) / Number(macros.SimDurationSteps);
  let active = null;
  let teardown = null;

  for (const { condition_id: id, label } of manifest.scenarios) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "chip";
    button.setAttribute("role", "tab");
    button.dataset.condition = id;
    button.textContent = label;
    button.addEventListener("click", () => select(id));
    tabs.appendChild(button);
  }

  async function select(id) {
    active = id;
    for (const button of tabs.children) {
      button.setAttribute("aria-selected", String(button.dataset.condition === id));
    }
    text.textContent = SCENARIO_TEXT[id] ?? "";
    if (!cache.has(id)) {
      view.innerHTML = '<p class="loading">Loading simulation…</p>';
      const entry = manifest.scenarios.find((s) => s.condition_id === id);
      try {
        cache.set(id, await loadJSON(`data/${entry.file}`));
      } catch (error) {
        view.innerHTML = `<p class="error">Could not load this condition (${error.message}).</p>`;
        return;
      }
    }
    if (active !== id) return;
    if (teardown) teardown();
    teardown = renderScenario(view, cache.get(id), manifest, stepSeconds);
  }

  // ?condition=<id> links to one condition; otherwise start on remapping.
  const requested = new URLSearchParams(window.location.search).get("condition");
  const ids = manifest.scenarios.map((s) => s.condition_id);
  select(ids.includes(requested) ? requested : ids.includes("remap") ? "remap" : ids[0]);
}

function renderScenario(view, payload, manifest, stepSeconds) {
  view.replaceChildren();
  const bins = payload.position_bins;
  const nBins = bins.length;
  const nSteps = payload.stop - payload.start;
  const toSeconds = (step) => (payload.start + step) * stepSeconds;
  const range = [toSeconds(0), toSeconds(nSteps)];
  const events = payload.events;
  const eventTimes = events.t.map(toSeconds);
  const predictive = decodeRows(payload.predictive.rows, nBins);
  const likelihoodRows = decodeRows(payload.likelihood_rows, nBins);
  const stepTimes = Array.from({ length: nSteps }, (_, i) => toSeconds(i));
  const shading = payload.scored_windows.map(([a, b]) => [
    toSeconds(a - payload.start),
    toSeconds(b - payload.start),
  ]);
  const rules = payload.events.flagged;
  const flagRules = manifest.flag_rules.simulation;

  // Summary stats across realizations.
  const summary = payload.summary;
  const stats = document.createElement("div");
  stats.className = "stat-row";
  stats.innerHTML =
    METRICS.map(
      (m) =>
        `<div class="stat"><div class="label">${m.label}</div><div class="value">${summary.median_flag_percent_text[m.name]}% flagged</div></div>`,
    ).join("") +
    `<div class="stat"><div class="label">Decoding error</div><div class="value">${summary.median_absolute_error_text} a.u.</div></div>`;
  const statsNote = document.createElement("p");
  statsNote.className = "note";
  statsNote.textContent = `Median percentage of spikes flagged by each diagnostic, and median decoding error, across ${manifest.macros.SimNRealizations} simulated sessions. Perturbed model component: ${payload.model_component === "—" ? "none" : payload.model_component.toLowerCase()}. Below: one session, shaded where the condition applies.`;

  // Mean normalized likelihood per spike-containing step, as in Figure 3a.
  const likelihoodSum = new Float64Array(nSteps * nBins);
  const likelihoodCount = new Uint16Array(nSteps);
  events.t.forEach((t, i) => {
    const row = normalizedRow(likelihoodRows.row(events.likelihood_row[i]));
    for (let b = 0; b < nBins; b += 1) likelihoodSum[t * nBins + b] += row[b];
    likelihoodCount[t] += 1;
  });
  const likelihoodBytes = new Uint8Array(nSteps * nBins);
  for (let t = 0; t < nSteps; t += 1) {
    let max = 0;
    for (let b = 0; b < nBins; b += 1) max = Math.max(max, likelihoodSum[t * nBins + b]);
    for (let b = 0; b < nBins; b += 1) {
      likelihoodBytes[t * nBins + b] = max > 0 ? Math.round((likelihoodSum[t * nBins + b] / max) * 255) : 0;
    }
  }
  const likelihoodMean = {
    nRows: nSteps,
    nBins,
    row: (t) => likelihoodBytes.subarray(t * nBins, (t + 1) * nBins),
  };

  const lut = manifest.colormaps;
  const predictiveBitmap = heatmapBitmap(predictive, lut.predictive, {
    rowMax: payload.predictive.row_max,
    range: payload.predictive.range,
  });
  const likelihoodBitmap = heatmapBitmap(likelihoodMean, lut.likelihood);
  const hasSpikes = (t) => likelihoodCount[t] > 0;
  const cellRank = rankOf(payload.cell_centers);
  const position = payload.true_position;
  const heatmapTrack = (label, bitmap, height, mask = null) => ({
    label,
    top: `${bins[nBins - 1]}`,
    bottom: `${bins[0]}`,
    height,
    draw(context, width, h, xOf) {
      if (mask) paintColumns(context, bitmap, stepTimes, mask, xOf, width, h);
      else paintHeatmap(context, bitmap, width, h);
      paintPositionLine(context, stepTimes, position, xOf, positionScale(bins, h), cssVar("--position"));
    },
  });
  const flaggedStyle = (metricName) => (i) =>
    rules[metricName][i] ? { fill: true, alpha: 1, radius: 3 } : { fill: true, alpha: 0.3, radius: 2.2 };

  const legend = document.createElement("div");
  legend.className = "legend";
  legend.innerHTML = `
    <span><i class="swatch" style="background:var(--position)"></i>Animal position</span>
    <span><i class="swatch dot" style="background:var(--text)"></i>Flagged spike (solid)</span>
    <span><i class="swatch dot" style="background:var(--text);opacity:.3"></i>Not flagged (faded)</span>
    <span><i class="swatch" style="background:var(--threshold)"></i>Flag threshold</span>`;

  const body = document.createElement("div");
  body.className = "player-body";
  const left = document.createElement("div");
  const detail = document.createElement("div");
  detail.className = "detail";
  body.append(left, detail);
  const scaleNote = document.createElement("p");
  scaleNote.className = "note";
  scaleNote.textContent = SCALE_NOTE;
  view.append(stats, statsNote, legend, body, scaleNote);

  const detailTitle = document.createElement("h3");
  const chartBox = document.createElement("div");
  const chart = new DistributionChart(chartBox, {
    positionBins: bins,
    xLabel: "Position (a.u.)",
    plotHeight: 110,
    width: 420,
  });
  const chartLegend = document.createElement("div");
  chartLegend.className = "legend";
  chartLegend.innerHTML = `
    <span><i class="swatch" style="background:var(--predictive)"></i>Prediction</span>
    <span><i class="swatch" style="background:var(--likelihood)"></i>Spike likelihood</span>
    <span><i class="swatch" style="background:var(--position)"></i>Animal</span>`;
  const readouts = document.createElement("div");
  readouts.className = "readouts";
  const cards = Object.fromEntries(
    METRICS.map((m) => {
      const card = readoutCard(m, flagRules[m.name]);
      readouts.appendChild(card);
      return [m.name, card];
    }),
  );
  detail.append(detailTitle, chartLegend, chartBox, readouts);

  function selectEvent(index) {
    if (index < 0) {
      detailTitle.textContent = "No spikes in this window";
      return;
    }
    const t = events.t[index];
    const cell = events.cell[index];
    detailTitle.textContent = `Spike at ${eventTimes[index].toFixed(3)} s — cell ${cell + 1} (field at ${payload.cell_centers[cell]} a.u.)`;
    chart.update({
      series: [
        { values: normalizedRow(predictive.row(t)), color: cssVar("--predictive") },
        { values: normalizedRow(likelihoodRows.row(events.likelihood_row[index])), color: cssVar("--likelihood") },
      ],
      marker: position[t],
    });
    for (const metric of METRICS) {
      const flagged = rules[metric.name][index];
      cards[metric.name]
        .querySelector(".value")
        .replaceChildren(`${metric.format(events[metric.name][index])} `, badge(flagged));
    }
    controls.time.textContent = `${eventTimes[index].toFixed(3)} s`;
  }

  const stack = new TrackStack(left, {
    ariaLabel: "Simulation tracks. Use the left and right arrow keys to step between spikes.",
    range,
    tracks: [
      heatmapTrack("Prediction", predictiveBitmap, 104),
      heatmapTrack("Likelihood", likelihoodBitmap, 64, hasSpikes),
      rasterTrack(eventTimes, events.cell.map((c) => cellRank[c]), payload.cell_centers.length, shading),
      ...METRICS.map((metric) =>
        metricTrack(
          metric,
          [{ times: eventTimes, values: events[metric.name], style: flaggedStyle(metric.name) }],
          flagRules[metric.name],
          shading,
        ),
      ),
    ],
    onCursor: (time) => playback.setTime(time),
    onStep: (direction) => playback.step(direction),
  });
  const controls = transport(left, {
    onStep: (direction) => playback.step(direction),
    onPlay: () => playback.toggle(),
  });
  const playback = wirePlayback(stack, range, eventTimes, selectEvent, controls);
  stack.draw();
  // Open inside the condition's window on a flagged or a typical spike.
  const inCondition = (i) =>
    payload.scored_windows.some(([a, b]) => events.t[i] + payload.start >= a && events.t[i] + payload.start < b);
  const anyFlag = (i) => METRICS.some((m) => rules[m.name][i]);
  const wantFlagged = OPEN_ON_FLAGGED.has(payload.condition_id);
  let initial = events.t.findIndex((_, i) => inCondition(i) && anyFlag(i) === wantFlagged);
  if (initial < 0) initial = Math.max(0, events.t.findIndex((_, i) => inCondition(i)));
  if (eventTimes.length) playback.selectIndex(initial);
  else selectEvent(-1);
  return () => playback.halt();
}

// ---------------------------------------------------------------------------
// Replay comparison
// ---------------------------------------------------------------------------

const MODELS = [
  { id: "continuous", label: "Continuous", short: "Cont." },
  { id: "continuous_fragmented", label: "Continuous–Fragmented", short: "Cont.–Frag." },
];

export function renderReplay(root, payload, manifest) {
  const view = root.querySelector("#rp-view");
  view.replaceChildren();
  const bins = payload.position_bins;
  const nBins = bins.length;
  const time = payload.time;
  const dt = time.length > 1 ? time[1] - time[0] : 0;
  const range = [time[0], time[time.length - 1] + dt];
  const lut = manifest.colormaps;
  const rules = payload.flag_rules;
  const events = payload.models.continuous.events;
  const eventTimes = events.t;
  const position = payload.linear_position;
  const likelihood = decodeRows(payload.likelihood, nBins);
  const unitLikelihoods = decodeRows(payload.unit_likelihoods, nBins);
  const predictive = Object.fromEntries(
    MODELS.map((m) => [m.id, decodeRows(payload.models[m.id].predictive.rows, nBins)]),
  );
  const predictiveBitmap = (id) =>
    heatmapBitmap(predictive[id], lut.predictive, {
      rowMax: payload.models[id].predictive.row_max,
      range: payload.models[id].predictive.range,
    });

  const heatmapTrack = (label, bitmap, height, mask = null) => ({
    label,
    top: `${Math.round(bins[nBins - 1])} cm`,
    bottom: `${Math.round(bins[0])}`,
    height,
    draw(context, width, h, xOf) {
      if (mask) paintColumns(context, bitmap, time, mask, xOf, width, h);
      else paintHeatmap(context, bitmap, width, h);
      paintPositionLine(context, time, position, xOf, positionScale(bins, h), cssVar("--position"));
    },
  });

  // Units: raster of every spike in the window, sorted by place-field peak.
  const rasterTimes = [];
  const rasterRows = [];
  payload.spike_times.forEach((times, unit) => {
    for (const t of times) {
      rasterTimes.push(t);
      rasterRows.push(payload.unit_rank[unit]);
    }
  });

  const seriesFor = (metricName) =>
    MODELS.map((model, index) => ({
      times: eventTimes,
      values: payload.models[model.id].events[metricName],
      style: () => (index === 0 ? { fill: false, radius: 3 } : { fill: true, radius: 2.4 }),
    }));

  // Window counts per model and metric.
  const counts = document.createElement("table");
  counts.className = "compare-table";
  const flaggedMetrics = METRICS.filter((m) => rules[m.name]);
  counts.innerHTML = `<thead><tr><th>Spikes flagged in this window</th>${MODELS.map((m) => `<th>${m.label}</th>`).join("")}</tr></thead><tbody>${flaggedMetrics
    .map(
      (metric) =>
        `<tr><td>${metric.label}</td>${MODELS.map((model) => {
          const flags = payload.models[model.id].events.flagged[metric.name];
          return `<td>${flags.filter(Boolean).length} of ${flags.length}</td>`;
        }).join("")}</tr>`,
    )
    .join("")}</tbody>`;

  const legend = document.createElement("div");
  legend.className = "legend";
  legend.innerHTML = `
    <span><i class="swatch" style="background:var(--position)"></i>Animal position</span>
    <span><i class="swatch ring" style="border-color:var(--text)"></i>Continuous model</span>
    <span><i class="swatch dot" style="background:var(--text)"></i>Continuous–Fragmented model</span>
    <span><i class="swatch" style="background:var(--threshold)"></i>Flag threshold</span>`;

  const body = document.createElement("div");
  body.className = "player-body";
  const left = document.createElement("div");
  const detail = document.createElement("div");
  detail.className = "detail";
  body.append(left, detail);
  const countsWrap = document.createElement("div");
  countsWrap.className = "table-wrap";
  countsWrap.appendChild(counts);
  const scaleNote = document.createElement("p");
  scaleNote.className = "note";
  scaleNote.textContent = SCALE_NOTE;
  view.append(legend, body, countsWrap, scaleNote);

  const detailTitle = document.createElement("h3");
  const chartLegend = document.createElement("div");
  chartLegend.className = "legend";
  chartLegend.innerHTML = `
    <span><i class="swatch" style="background:var(--predictive)"></i>Prediction</span>
    <span><i class="swatch" style="background:var(--likelihood)"></i>Spike likelihood</span>
    <span><i class="swatch" style="background:var(--position)"></i>Animal</span>`;
  detail.append(detailTitle, chartLegend);
  const charts = {};
  for (const model of MODELS) {
    const heading = document.createElement("div");
    heading.className = "row-label";
    heading.textContent = model.label;
    const box = document.createElement("div");
    detail.append(heading, box);
    charts[model.id] = new DistributionChart(box, {
      positionBins: bins,
      xLabel: "Linearized position (cm)",
      plotHeight: 70,
      width: 420,
    });
  }
  const table = document.createElement("table");
  table.className = "compare-table";
  const tableWrap = document.createElement("div");
  tableWrap.className = "table-wrap";
  tableWrap.appendChild(table);
  detail.appendChild(tableWrap);
  const rescue = document.createElement("p");
  rescue.className = "note";
  detail.appendChild(rescue);

  function selectEvent(index) {
    if (index < 0) return;
    const t = eventTimes[index];
    // The decoder bin that counted this spike; its prediction precedes the spike.
    const step = events.bin[index];
    const cell = events.cell[index];
    detailTitle.textContent = `Spike at ${t.toFixed(3)} s — unit ${cell + 1}`;
    for (const model of MODELS) {
      charts[model.id].update({
        series: [
          { values: normalizedRow(predictive[model.id].row(step)), color: cssVar("--predictive") },
          { values: normalizedRow(unitLikelihoods.row(cell)), color: cssVar("--likelihood") },
        ],
        marker: position[step],
      });
    }
    table.innerHTML = `<thead><tr><th></th>${MODELS.map((m) => `<th>${m.short}</th>`).join("")}</tr></thead>`;
    const tbody = document.createElement("tbody");
    const rescued = [];
    for (const metric of METRICS) {
      const tr = document.createElement("tr");
      const name = document.createElement("td");
      name.textContent = metric.label;
      tr.appendChild(name);
      const flags = [];
      for (const model of MODELS) {
        const modelEvents = payload.models[model.id].events;
        const td = document.createElement("td");
        const flagged = rules[metric.name] ? modelEvents.flagged[metric.name][index] : null;
        flags.push(flagged);
        td.append(`${metric.format(modelEvents[metric.name][index])} `);
        if (flagged !== null) td.appendChild(badge(flagged));
        tr.appendChild(td);
      }
      if (flags[0] === true && flags[1] === false) rescued.push(metric.label);
      tbody.appendChild(tr);
    }
    table.appendChild(tbody);
    rescue.textContent = rescued.length
      ? `Rescued: flagged under the Continuous model but not once the model can jump (${rescued.join(", ")}).`
      : "";
    controls.time.textContent = `${t.toFixed(3)} s`;
  }

  const stack = new TrackStack(left, {
    ariaLabel: "Hippocampal recording tracks. Use the left and right arrow keys to step between spikes.",
    range,
    tracks: [
      heatmapTrack("Prediction: Continuous", predictiveBitmap("continuous"), 92),
      heatmapTrack("Prediction: Cont.–Frag.", predictiveBitmap("continuous_fragmented"), 92),
      heatmapTrack(
        "Likelihood",
        heatmapBitmap(likelihood, lut.likelihood),
        64,
        (t) => payload.has_spikes[t],
      ),
      rasterTrack(rasterTimes, rasterRows, payload.spike_times.length, null, "Units"),
      ...METRICS.map((metric) => metricTrack(metric, seriesFor(metric.name), rules[metric.name], null)),
    ],
    onCursor: (t) => playback.setTime(t),
    onStep: (direction) => playback.step(direction),
  });
  const controls = transport(left, {
    onStep: (direction) => playback.step(direction),
    onPlay: () => playback.toggle(),
  });
  const playback = wirePlayback(stack, range, eventTimes, selectEvent, controls);
  stack.draw();
  const firstRescue = events.flagged.hpd_overlap.findIndex(
    (flag, i) => flag && !payload.models.continuous_fragmented.events.flagged.hpd_overlap[i],
  );
  playback.selectIndex(Math.max(0, firstRescue));
}
