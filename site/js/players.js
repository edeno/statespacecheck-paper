// Scenario player (Figure 3 simulation) and replay comparison (Figure 4
// recording). Both show precomputed per-spike diagnostics from the paper's
// pipeline on a shared time axis, with a cursor that inspects one spike.

import {
  cssVar,
  DistributionChart,
  heatmapBitmap,
  paintColumns,
  paintDots,
  paintHeatmap,
  paintPositionLine,
  paintShading,
  paintThreshold,
  positionScale,
  TrackStack,
} from "./charts.js";
import {
  badge,
  decodeHeatmap,
  decodeRows,
  liveRegion,
  loadJSON,
  METRICS,
  nearestIndex,
  plottedWorseFit,
  readoutCard,
  worseFit,
} from "./data.js";

// Seconds of real time to play through one window.
const PLAYBACK_SECONDS = 15;

// Half-width (s) of the window used to find the peak of population firing,
// which marks the replay event in the recording.
const POPULATION_PEAK_HALF_WIDTH = 0.05;

// Tab titles; the summaries abbreviate some condition labels for the figure.
const SCENARIO_TITLES = {
  well_specified: "Well-specified",
  remap: "Remap",
  history_dependent: "History-dependent firing",
  replay: "Replay",
  drift: "Drift",
  sparse_population: "Sparse population",
};

const SCENARIO_TEXT = {
  well_specified:
    "Model and data agree. The few spikes flagged here show the false-positive rate that each threshold allows.",
  remap:
    "The decoder evaluates spikes against a scrambled set of place fields, as after global remapping, so within milliseconds its own estimate is pulled away from the animal. The diagnostics test consistency, not accuracy: a spike whose likelihood agrees with that wrong prediction is not flagged. But the scramble sends successive spikes to scattered places, so many spikes conflict with the prediction, and all three diagnostics flag them.",
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
// a typical, unflagged spike (for drift, one the lagging prediction still fits).
const OPEN_ON_FLAGGED = new Set(["remap", "sparse_population"]);

const INTERACTION_HELP =
  "Hover over or tap the tracks to inspect a spike, press Play, or focus the tracks (click or Tab) and use ← → to step between spikes (Home and End jump to the first and last).";

const AXIS_HELP =
  "HPD overlap is drawn on a symmetric-log axis (linear below 0.01, logarithmic above), as in the paper, so values near 0 separate from exact zeros. −log p is the negative natural log of the p-value (p = 0.05 is about 3). KL divergence is in nats (natural-log units). Dashed lines mark flag thresholds.";

const SCALE_HELP =
  "Darker prediction shading means higher probability, on one color scale per panel as in the paper's figures. The likelihood track is drawn only in time bins that contain spikes. Likelihood columns and the curves in the spike panel are each scaled to their own maximum.";

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

function normalizedRow(row) {
  let total = 0;
  for (const v of row) total += v;
  return Array.from(row, (v) => (total > 0 ? v / total : 0));
}

function argmax(values) {
  let best = 0;
  for (let i = 1; i < values.length; i += 1) if (values[i] > values[best]) best = i;
  return best;
}

function metricRange(metric, values, rule) {
  if (metric.range) return metric.range;
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
    note: plottedWorseFit(metric),
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
      const color = cssVar(metric.color);
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

function note(text) {
  const p = document.createElement("p");
  p.className = "note";
  p.textContent = text;
  return p;
}

function legendBlock(html) {
  const legend = document.createElement("div");
  legend.className = "legend";
  legend.innerHTML = html;
  return legend;
}

// Legend of the detail chart that both players show for the selected spike.
const DETAIL_LEGEND = `
    <span><i class="swatch" style="background:var(--predictive)"></i>Prediction</span>
    <span><i class="swatch" style="background:var(--likelihood)"></i>Spike likelihood</span>
    <span><i class="swatch" style="background:var(--position)"></i>Animal's position</span>`;

/** The player layout: tracks on the left, a detail panel on the right. */
function playerFrame() {
  const body = document.createElement("div");
  body.className = "player-body";
  const left = document.createElement("div");
  const detail = document.createElement("div");
  detail.className = "detail";
  body.append(left, detail);
  return { body, left, detail };
}

/**
 * A heatmap of position over time with the animal's position drawn on top;
 * `mask(t)` limits the drawing to some time steps. `unit` labels the axis.
 */
function heatmapTrack({ label, bitmap, height, times, bins, position, mask = null, unit = "" }) {
  const last = bins[bins.length - 1];
  return {
    label,
    top: unit ? `${Math.round(last)} ${unit}` : `${last}`,
    bottom: unit ? `${Math.round(bins[0])}` : `${bins[0]}`,
    height,
    draw(context, width, h, xOf) {
      if (mask) paintColumns(context, bitmap, times, mask, xOf, width, h);
      else paintHeatmap(context, bitmap, width, h);
      paintPositionLine(context, times, position, xOf, positionScale(bins, h), cssVar("--position"));
    },
  };
}

/** Play/pause and step buttons plus a time readout. */
function transport(container, { onStep, onPlay }) {
  const bar = document.createElement("div");
  bar.className = "transport";
  bar.innerHTML = `
    <button class="chip" data-step="-1" type="button" aria-label="Previous spike">◀ Prev spike</button>
    <button class="chip" data-play type="button">▶ Play</button>
    <button class="chip" data-step="1" type="button" aria-label="Next spike">Next spike ▶</button>
    <span class="time"></span>`;
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

/**
 * Cursor and spike selection for one player. `select(index)` draws a spike;
 * `announce(index)` is called only for discrete steps (buttons and keys), so
 * screen readers are not flooded during hover or playback.
 */
function wirePlayback(stack, range, eventTimes, select, controls, announce) {
  let stop = null;
  let current = null;
  const setTime = (time) => {
    stack.placeCursor(time);
    const index = nearestIndex(eventTimes, time);
    if (index !== current) {
      current = index;
      select(index);
    }
  };
  const halt = () => {
    if (stop) stop();
    stop = null;
    controls.play.textContent = "▶ Play";
  };
  // Select by index, not time: spikes in one time bin share a time.
  const selectIndex = (index, { speak = false } = {}) => {
    halt();
    current = index;
    stack.placeCursor(eventTimes[index]);
    select(index);
    if (speak) announce(index);
  };
  const clamp = (index) => Math.min(eventTimes.length - 1, Math.max(0, index));
  return {
    setTime: (time) => {
      halt();
      setTime(time);
    },
    selectIndex,
    step: (direction) => selectIndex(clamp((current ?? -1) + direction), { speak: true }),
    key: (key) => {
      if (!eventTimes.length) return;
      const target = {
        ArrowRight: (current ?? -1) + 1,
        ArrowLeft: (current ?? 1) - 1,
        Home: 0,
        End: eventTimes.length - 1,
      }[key];
      selectIndex(clamp(target), { speak: true });
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

function flagSummary(index, flagsByMetric) {
  const flagged = METRICS.filter((m) => flagsByMetric[m.name]?.[index]).map((m) => m.label);
  return flagged.length ? `Flagged by ${flagged.join(", ")}.` : "Not flagged.";
}

/**
 * Tracks, transport, and spike selection for one player. `select(index)` shows
 * the spike nearest the cursor (-1 when there are none); the time readout is
 * kept here. `announce(index)` is read to screen readers on discrete steps.
 */
function mountTracks(left, { ariaLabel, range, tracks, eventTimes, select, announce }) {
  const stack = new TrackStack(left, {
    ariaLabel,
    range,
    tracks,
    onCursor: (time) => playback.setTime(time),
    onKey: (key) => playback.key(key),
  });
  const controls = transport(left, {
    onStep: (direction) => playback.step(direction),
    onPlay: () => playback.toggle(),
  });
  const show = (index) => {
    select(index);
    if (index >= 0) controls.time.textContent = `${eventTimes[index].toFixed(3)} s`;
  };
  const playback = wirePlayback(stack, range, eventTimes, show, controls, announce);
  stack.draw();
  return { stack, playback };
}

// ---------------------------------------------------------------------------
// Scenario player
// ---------------------------------------------------------------------------

export function initScenarios(root, manifest) {
  const tabs = root.querySelector("#sc-tabs");
  const view = root.querySelector("#sc-view");
  const text = root.querySelector("#sc-text");
  const cache = new Map();
  const ids = manifest.scenarios.map((s) => s.condition_id);
  let active = null;
  let teardown = null;

  view.setAttribute("role", "tabpanel");
  view.tabIndex = -1;
  const buttons = manifest.scenarios.map(({ condition_id: id, label }) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "chip";
    button.id = `sc-tab-${id}`;
    button.setAttribute("role", "tab");
    button.setAttribute("aria-controls", view.id);
    button.dataset.condition = id;
    button.textContent = SCENARIO_TITLES[id] ?? label;
    button.addEventListener("click", () => select(id));
    button.addEventListener("keydown", (event) => {
      const i = ids.indexOf(id);
      const target = {
        ArrowRight: (i + 1) % ids.length,
        ArrowLeft: (i - 1 + ids.length) % ids.length,
        Home: 0,
        End: ids.length - 1,
      }[event.key];
      if (target === undefined) return;
      event.preventDefault();
      select(ids[target]);
      buttons[target].focus();
    });
    tabs.appendChild(button);
    return button;
  });

  async function select(id) {
    active = id;
    // Stop the previous player before anything else can go wrong.
    if (teardown) teardown();
    teardown = null;
    for (const button of buttons) {
      const selected = button.dataset.condition === id;
      button.setAttribute("aria-selected", String(selected));
      button.tabIndex = selected ? 0 : -1;
    }
    view.setAttribute("aria-labelledby", `sc-tab-${id}`);
    text.textContent = SCENARIO_TEXT[id] ?? "";
    const url = new URL(window.location.href);
    url.searchParams.set("condition", id);
    window.history.replaceState(null, "", url);
    if (!cache.has(id)) {
      view.innerHTML = '<p class="loading">Loading simulation…</p>';
      const entry = manifest.scenarios.find((s) => s.condition_id === id);
      try {
        cache.set(id, await loadJSON(`data/${entry.file}`));
      } catch (error) {
        if (active === id) {
          view.innerHTML = `<p class="error">Could not load this condition (${error.message}).</p>`;
        }
        return;
      }
    }
    if (active !== id) return;
    try {
      teardown = renderScenario(view, cache.get(id), manifest);
    } catch (error) {
      view.innerHTML = `<p class="error">Could not display this condition (${error.message}).</p>`;
      console.error(error);
    }
  }

  // ?condition=<id> links to one condition; otherwise start on remapping.
  const requested = new URLSearchParams(window.location.search).get("condition");
  select(ids.includes(requested) ? requested : ids.includes("remap") ? "remap" : ids[0]);
}

function renderScenario(view, payload, manifest) {
  view.replaceChildren();
  const bins = payload.position_bins;
  const nBins = bins.length;
  const nSteps = payload.stop - payload.start;
  const toSeconds = (step) => (payload.start + step) * payload.step_seconds;
  const range = [toSeconds(0), toSeconds(nSteps)];
  const events = payload.events;
  const eventTimes = events.t.map(toSeconds);
  const predictive = decodeHeatmap(payload.predictive, nBins);
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
  const component =
    payload.model_component === "—" ? "none" : payload.model_component.toLowerCase();
  const statsNote = note(
    `Medians across ${manifest.macros.SimNRealizations} simulated sessions: the percentage of spikes each diagnostic flags, and the decoding error, the median absolute difference between the decoder's position estimate (the filtered posterior mean) and the true position. Perturbed model component: ${component}. Below: ${(range[1] - range[0]).toFixed(1)} s of one session, with the condition's time window shaded gray.`,
  );

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
      likelihoodBytes[t * nBins + b] =
        max > 0 ? Math.round((likelihoodSum[t * nBins + b] / max) * 255) : 0;
    }
  }
  const likelihoodMean = {
    nRows: nSteps,
    nBins,
    row: (t) => likelihoodBytes.subarray(t * nBins, (t + 1) * nBins),
  };

  const lut = manifest.colormaps;
  const predictiveBitmap = heatmapBitmap(predictive, lut.predictive);
  const likelihoodBitmap = heatmapBitmap(likelihoodMean, lut.likelihood);
  const hasSpikes = (t) => likelihoodCount[t] > 0;
  const cellRank = rankOf(payload.cell_centers);
  const position = payload.true_position;
  const track = (label, bitmap, height, mask = null) =>
    heatmapTrack({ label, bitmap, height, mask, times: stepTimes, bins, position });
  const flaggedStyle = (metricName) => (i) =>
    rules[metricName][i] ? { fill: true, alpha: 1, radius: 3 } : { fill: true, alpha: 0.3, radius: 2.2 };

  const legend = legendBlock(`
    <span><i class="swatch" style="background:var(--position)"></i>Animal's position</span>
    <span><i class="swatch dot" style="background:var(--text)"></i>Flagged spike (solid)</span>
    <span><i class="swatch dot" style="background:var(--text);opacity:.3"></i>Not flagged (faded)</span>
    <span><i class="swatch" style="background:var(--threshold)"></i>Flag threshold</span>
    <span><i class="swatch band" style="background:var(--text-muted)"></i>Condition window</span>`);

  const { body, left, detail } = playerFrame();
  view.append(stats, statsNote, legend, body, note(INTERACTION_HELP), note(AXIS_HELP), note(SCALE_HELP));
  const say = liveRegion(view);

  const detailTitle = document.createElement("h3");
  const chartBox = document.createElement("div");
  const chart = new DistributionChart(chartBox, {
    positionBins: bins,
    xLabel: "Position (a.u.)",
    plotHeight: 110,
  });
  const chartLegend = legendBlock(DETAIL_LEGEND);
  const readouts = document.createElement("div");
  readouts.className = "readouts";
  const cards = Object.fromEntries(
    METRICS.map((m) => {
      const card = readoutCard(m, flagRules[m.name]);
      readouts.appendChild(card.element);
      return [m.name, card];
    }),
  );
  detail.append(detailTitle, chartLegend, chartBox, readouts);

  function describeSpike(index) {
    const cell = events.cell[index];
    const likelihood = likelihoodRows.row(events.likelihood_row[index]);
    const trueCenter = payload.cell_centers[cell];
    // The decoder's field is where this spike's likelihood peaks; after
    // remapping it differs from the field that generated the spike.
    const decoderCenter = bins[argmax(likelihood)];
    const field =
      Math.abs(decoderCenter - trueCenter) > 1
        ? `true field at ${trueCenter} a.u.; the decoder's remapped field at ${decoderCenter} a.u.`
        : `field at ${trueCenter} a.u.`;
    return `Spike at ${eventTimes[index].toFixed(3)} s — cell ${cell + 1} (${field})`;
  }

  function selectEvent(index) {
    if (index < 0) {
      detailTitle.textContent = "No spikes in this window";
      return;
    }
    const t = events.t[index];
    detailTitle.textContent = describeSpike(index);
    chart.update({
      series: [
        { values: normalizedRow(predictive.row(t)), color: cssVar("--predictive") },
        {
          values: normalizedRow(likelihoodRows.row(events.likelihood_row[index])),
          color: cssVar("--likelihood"),
        },
      ],
      marker: position[t],
    });
    for (const metric of METRICS) {
      cards[metric.name].set(events[metric.name][index], rules[metric.name][index]);
    }
  }

  const { stack, playback } = mountTracks(left, {
    ariaLabel: "Simulation tracks. Use the arrow keys to step between spikes.",
    range,
    tracks: [
      track("Prediction", predictiveBitmap, 104),
      track("Likelihood", likelihoodBitmap, 64, hasSpikes),
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
    eventTimes,
    select: selectEvent,
    announce: (index) => say(`${describeSpike(index)}. ${flagSummary(index, rules)}`),
  });
  // Open inside the condition's window on a flagged or a typical spike.
  const inCondition = (i) =>
    payload.scored_windows.some(
      ([a, b]) => events.t[i] + payload.start >= a && events.t[i] + payload.start < b,
    );
  const anyFlag = (i) => METRICS.some((m) => rules[m.name][i]);
  const wantFlagged = OPEN_ON_FLAGGED.has(payload.condition_id);
  let initial = events.t.findIndex((_, i) => inCondition(i) && anyFlag(i) === wantFlagged);
  if (initial < 0) initial = Math.max(0, events.t.findIndex((_, i) => inCondition(i)));
  if (eventTimes.length) playback.selectIndex(initial);
  else selectEvent(-1);
  return () => {
    playback.halt();
    stack.destroy();
    chart.destroy();
  };
}

// ---------------------------------------------------------------------------
// Replay comparison
// ---------------------------------------------------------------------------

const MODELS = [
  { id: "continuous", label: "Continuous", short: "Cont." },
  { id: "continuous_fragmented", label: "Continuous–Fragmented", short: "Cont.–Frag." },
];

/** Time of peak population firing (all units), which marks the replay event. */
function populationPeak(spikeTimes, range) {
  const all = spikeTimes.flat().sort((a, b) => a - b);
  let best = range[0];
  let bestCount = -1;
  let lo = 0;
  let hi = 0;
  for (let t = range[0]; t <= range[1]; t += 0.002) {
    while (lo < all.length && all[lo] < t - POPULATION_PEAK_HALF_WIDTH) lo += 1;
    while (hi < all.length && all[hi] <= t + POPULATION_PEAK_HALF_WIDTH) hi += 1;
    if (hi - lo > bestCount) {
      bestCount = hi - lo;
      best = t;
    }
  }
  return best;
}

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
    MODELS.map((m) => [m.id, decodeHeatmap(payload.models[m.id].predictive, nBins)]),
  );
  const predictiveBitmap = (id) => heatmapBitmap(predictive[id], lut.predictive);
  const track = (label, bitmap, height, mask = null) =>
    heatmapTrack({ label, bitmap, height, mask, times: time, bins, position, unit: "cm" });

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

  const legend = legendBlock(`
    <span><i class="swatch" style="background:var(--position)"></i>Animal's position</span>
    <span><i class="swatch ring" style="border-color:var(--text)"></i>Continuous model</span>
    <span><i class="swatch dot" style="background:var(--text)"></i>Continuous–Fragmented model</span>
    <span><i class="swatch" style="background:var(--threshold)"></i>Flag threshold</span>`);

  const { body, left, detail } = playerFrame();
  const countsWrap = document.createElement("div");
  countsWrap.className = "table-wrap";
  countsWrap.appendChild(counts);
  view.append(
    legend,
    body,
    countsWrap,
    note(INTERACTION_HELP),
    note(
      "Here the dots do not show flag status: open circles are the Continuous model and filled dots the Continuous–Fragmented model, and a spike is flagged when its marker lies beyond the dashed threshold. The Continuous–Fragmented prediction is summed over its Continuous and Fragmented states. Position is linearized distance along the maze (cm).",
    ),
    note(AXIS_HELP),
    note(SCALE_HELP),
  );
  const say = liveRegion(view);

  const detailTitle = document.createElement("h3");
  const chartLegend = legendBlock(DETAIL_LEGEND);
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
    });
  }
  // One row per metric; only the value cells change with the selected spike.
  const table = document.createElement("table");
  table.className = "compare-table";
  table.innerHTML = `<thead><tr><th></th>${MODELS.map((m) => `<th>${m.short}</th>`).join("")}</tr></thead>`;
  const tbody = document.createElement("tbody");
  const valueCells = {};
  for (const metric of METRICS) {
    const tr = document.createElement("tr");
    const name = document.createElement("td");
    name.innerHTML = `${metric.label}<div class="rule">${worseFit(metric)}</div>`;
    tr.appendChild(name);
    valueCells[metric.name] = Object.fromEntries(
      MODELS.map((model) => {
        const td = document.createElement("td");
        tr.appendChild(td);
        return [model.id, td];
      }),
    );
    tbody.appendChild(tr);
  }
  table.appendChild(tbody);
  const tableWrap = document.createElement("div");
  tableWrap.className = "table-wrap";
  tableWrap.appendChild(table);
  detail.appendChild(tableWrap);
  const rescue = document.createElement("p");
  rescue.className = "note";
  detail.appendChild(rescue);

  const isRescued = (metric, index) =>
    payload.models.continuous.events.flagged[metric]?.[index] === true &&
    payload.models.continuous_fragmented.events.flagged[metric]?.[index] === false;

  function describeSpike(index) {
    return `Spike at ${eventTimes[index].toFixed(3)} s — unit ${events.cell[index] + 1}`;
  }

  function selectEvent(index) {
    if (index < 0) return;
    // The decoder bin that counted this spike; its prediction precedes the spike.
    const step = events.bin[index];
    const cell = events.cell[index];
    detailTitle.textContent = describeSpike(index);
    for (const model of MODELS) {
      charts[model.id].update({
        series: [
          { values: normalizedRow(predictive[model.id].row(step)), color: cssVar("--predictive") },
          { values: normalizedRow(unitLikelihoods.row(cell)), color: cssVar("--likelihood") },
        ],
        marker: position[step],
      });
    }
    for (const metric of METRICS) {
      for (const model of MODELS) {
        const modelEvents = payload.models[model.id].events;
        const td = valueCells[metric.name][model.id];
        td.replaceChildren(`${metric.format(modelEvents[metric.name][index])} `);
        if (rules[metric.name]) td.appendChild(badge(modelEvents.flagged[metric.name][index]));
      }
    }
    const rescued = METRICS.filter((m) => isRescued(m.name, index)).map((m) => m.label);
    rescue.textContent = rescued.length
      ? `Rescued (${rescued.join(", ")}): flagged under the Continuous model but not under the Continuous–Fragmented model.`
      : "";
  }

  const { playback } = mountTracks(left, {
    ariaLabel: "Hippocampal recording tracks. Use the arrow keys to step between spikes.",
    range,
    tracks: [
      track("Prediction: Continuous", predictiveBitmap("continuous"), 92),
      track("Prediction: Cont.–Frag.", predictiveBitmap("continuous_fragmented"), 92),
      track(
        "Likelihood",
        heatmapBitmap(likelihood, lut.likelihood),
        64,
        (t) => payload.has_spikes[t],
      ),
      rasterTrack(rasterTimes, rasterRows, payload.spike_times.length, null, "Units"),
      ...METRICS.map((metric) => metricTrack(metric, seriesFor(metric.name), rules[metric.name], null)),
    ],
    eventTimes,
    select: selectEvent,
    announce: (index) =>
      say(`${describeSpike(index)}. Continuous model: ${flagSummary(index, events.flagged)}`),
  });
  // Open on the HPD-overlap rescue nearest the peak of population firing,
  // i.e., inside the replay event.
  const peak = populationPeak(payload.spike_times, range);
  let initial = -1;
  eventTimes.forEach((t, i) => {
    if (!isRescued("hpd_overlap", i)) return;
    if (initial < 0 || Math.abs(t - peak) < Math.abs(eventTimes[initial] - peak)) initial = i;
  });
  if (eventTimes.length) playback.selectIndex(Math.max(0, initial));
}
