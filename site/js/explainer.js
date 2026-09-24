// "Prediction and likelihood": play through a short spike train decoded by the
// paper's Bayesian filter. Each time step shows the prediction, the cells'
// place fields (the one that fired highlighted), the likelihood of what the
// step recorded, and the posterior. Every distribution is precomputed by
// statespacecheck_paper.site_export.filter_explainer_payload.

import {
  cssVar,
  DistributionChart,
  heatmapBitmap,
  paintPositionLine,
  positionScale,
  TrackStack,
} from "./charts.js";
import { decodeRows } from "./data.js";

// Playback speed, and how long it lingers on a step with spikes (seconds).
// Playback pauses by itself on the inconsistent spikes.
const STEPS_PER_SECOND = 20;
const SPIKE_HOLD = 0.25;

const SUBSCRIPTS = "₀₁₂₃₄₅₆₇₈₉";
const SUPERSCRIPTS = "⁰¹²³⁴⁵⁶⁷⁸⁹";
const fmt = (value) => value.toFixed(1);
const subscript = (n) => String(n).replace(/\d/g, (d) => SUBSCRIPTS[Number(d)]);
/** Exponent for a field raised to a spike count: "" for 1, "²" for 2, ... */
const power = (n) => (n === 1 ? "" : String(n).replace(/\d/g, (d) => SUPERSCRIPTS[Number(d)]));

function scaled(row, max) {
  return Array.from(row, (v) => (v / 255) * max);
}

/** "cell 5", "cells 4 and 5", "cell 5 (twice)". */
function listCells(cells) {
  const counts = new Map();
  for (const c of cells) counts.set(c, (counts.get(c) ?? 0) + 1);
  const names = [...counts].map(([c, n]) => (n === 1 ? `${c + 1}` : `${c + 1} (${n === 2 ? "twice" : `${n} times`})`));
  if (names.length === 1) return `cell ${names[0]}`;
  return `cells ${names.slice(0, -1).join(", ")} and ${names[names.length - 1]}`;
}

export function initExplainer(root, data, manifest) {
  const bins = data.position_bins;
  const nBins = bins.length;
  const centers = data.cell_centers;
  const nSteps = data.true_position.length;
  const x = data.true_position;
  const m = data.moments;
  const predictive = decodeRows(data.predictive.rows, nBins);
  const posterior = decodeRows(data.posterior.rows, nBins);
  const likelihood = decodeRows(data.likelihood, nBins);
  const fields = decodeRows(data.place_fields.rows, nBins);
  const sharedMax = data.posterior.range[1];
  const predictiveAt = (t) => scaled(predictive.row(t), data.predictive.row_max[t]);
  const posteriorAt = (t) => scaled(posterior.row(t), data.posterior.row_max[t]);

  // Spikes by time step; a cell repeats when it fires more than once.
  const spikes = new Map();
  data.events.t.forEach((t, i) => {
    if (!spikes.has(t)) spikes.set(t, []);
    spikes.get(t).push({ cell: data.events.cell[i], hpd: data.events.hpd_overlap[i] });
  });
  const spikeSteps = [...spikes.keys()].sort((a, b) => a - b);
  const lastSpikeBefore = (t) => {
    let last = null;
    for (const s of spikeSteps) if (s < t) last = s;
    return last;
  };
  // The scripted inconsistent spikes; playback pauses there.
  const isConflict = (t) => t === data.conflict_step;

  const playButton = root.querySelector("#ft-play");
  const nextButton = root.querySelector("#ft-next");
  const restartButton = root.querySelector("#ft-restart");
  const status = root.querySelector("#ft-status");
  const caption = root.querySelector("#ft-caption");
  const likelihoodLabel = root.querySelector("#ft-likelihood-label");

  let t = 0;
  let frame = null;

  // --------------------------------------------------------------- Tracks

  const bitmap = heatmapBitmap(posterior, manifest.colormaps.predictive, {
    rowMax: data.posterior.row_max,
    range: data.posterior.range,
  });
  const stepCenters = Array.from({ length: nSteps }, (_, i) => i + 0.5);

  /** The whole run faintly, and the part already run in full color. */
  function paintRun(context, xOf, yOf, lineWidth) {
    context.globalAlpha = 0.3;
    paintPositionLine(context, stepCenters, x, xOf, yOf, cssVar("--position"), lineWidth);
    context.globalAlpha = 1;
    paintPositionLine(
      context,
      stepCenters.slice(0, t + 1),
      x.slice(0, t + 1),
      xOf,
      yOf,
      cssVar("--position"),
      lineWidth,
    );
  }

  const posteriorTrack = {
    label: "Posterior",
    top: `${bins[nBins - 1]}`,
    bottom: `${bins[0]}`,
    height: 110,
    draw(context, width, height, xOf) {
      context.fillStyle = cssVar("--surface");
      context.fillRect(0, 0, width, height);
      context.imageSmoothingEnabled = false;
      context.drawImage(bitmap, 0, 0, t + 1, bitmap.height, 0, 0, xOf(t + 1), height);
      paintRun(context, xOf, positionScale(bins, height), 2.4);
    },
  };

  const rasterTrack = {
    label: "Spikes",
    note: "cells by field",
    top: `${centers[centers.length - 1]}`,
    bottom: `${centers[0]}`,
    height: 88,
    draw(context, width, height, xOf) {
      context.fillStyle = cssVar("--surface");
      context.fillRect(0, 0, width, height);
      paintRun(context, xOf, positionScale(centers, height), 2);
      // One row per cell, bottom to top; cell_centers ascend.
      const rowHeight = height / centers.length;
      data.events.t.forEach((step, i) => {
        const row = data.events.cell[i];
        const left = xOf(step);
        const w = Math.max(2, xOf(step + 1) - left);
        if (step === t) context.fillStyle = cssVar("--likelihood");
        else if (step < t) context.fillStyle = cssVar("--text");
        else context.fillStyle = cssVar("--curve-muted");
        context.fillRect(left, height - (row + 1) * rowHeight + 1, w, Math.max(1, rowHeight - 2));
      });
    },
  };

  const stack = new TrackStack(root.querySelector("#ft-tracks"), {
    ariaLabel: "Posterior over time and spike raster. Use the arrow keys to step through time.",
    range: [0, nSteps],
    tracks: [posteriorTrack, rasterTrack],
    hover: false,
    tickLabel: (tick) => String(Math.round(tick)),
    onCursor: (time) => {
      stop();
      go(Math.min(nSteps - 1, Math.max(0, Math.floor(time))));
    },
    onKey: (key) => {
      stop();
      if (key === "ArrowRight") go(Math.min(nSteps - 1, t + 1));
      else if (key === "ArrowLeft") go(Math.max(0, t - 1));
      else if (key === "Home") go(0);
      else if (key === "End") go(nSteps - 1);
    },
  });

  // ------------------------------------------------- Distribution rows

  const rowChart = (id, axis) =>
    new DistributionChart(root.querySelector(id), {
      positionBins: bins,
      xLabel: "Position (a.u.)",
      plotHeight: 58,
      axis,
    });
  const charts = {
    prediction: rowChart("#ft-prediction", false),
    fields: rowChart("#ft-fields", false),
    likelihood: rowChart("#ft-likelihood", false),
    posterior: rowChart("#ft-posterior", true),
  };
  // Each field in expected spikes per step, drawn on the cells' shared scale.
  const fieldRows = centers.map((_, c) => scaled(fields.row(c), data.place_fields.row_max[c]));

  // --------------------------------------------------------- Captions

  function describe(playing) {
    const fired = spikes.get(t) ?? [];
    const cells = fired.map((s) => s.cell);
    const unique = [...new Set(cells)];
    const manyCells = unique.length > 1;
    const manySpikes = fired.length > 1;
    const animal = `(the animal is at ${fmt(x[t])} a.u.)`;
    const last = lastSpikeBefore(t);

    if (playing) return "Playing. Pause at any step to read what the filter does there.";
    if (!fired.length) {
      const spread = `SD ${fmt(m.posterior_sd[t - 1])} → ${fmt(m.predictive_sd[t])} a.u.`;
      if (last === t - 1) {
        return `Step ${t}: no spike. The prediction is the last posterior (dashed) spread by the movement model (${spread}): the animal may have moved. With no spike, the likelihood is exp(−Λ(x)), the probability that no cell fires, which is nearly flat. So the posterior is almost the prediction, and both keep spreading until the next spike.`;
      }
      return `Step ${t}: no spike for ${t - last} steps. The prediction and the posterior keep spreading: ${spread} ${animal}.`;
    }

    const posteriorText = `mean ${fmt(m.posterior_mean[t])} a.u., SD ${fmt(m.posterior_sd[t])} a.u. ${animal}`;
    if (t === spikeSteps[0]) {
      const c = cells[0] + 1;
      return `Step ${t}: ${listCells(cells)} fires. Before any spikes, the prediction is the decoder's initial distribution, flat across the track. A place field is a cell's expected spike count at each position; the field of the cell that fired is dark. The likelihood is the probability of what this step recorded at each position, a spike from cell ${c} and none from the others: λ${subscript(c)}(x)·exp(−Λ(x)), where Λ(x) is the expected spike count of all the cells and exp(−Λ(x)) (dashed) is the probability of no spikes. Multiplying the prediction by the likelihood and renormalizing gives the posterior; with a flat prediction it is the normalized likelihood: ${posteriorText}.`;
    }

    const gap = t - last;
    const prediction =
      gap === 1
        ? `The prediction is the last posterior (dashed) spread by one step of the movement model (SD ${fmt(m.posterior_sd[last])} → ${fmt(m.predictive_sd[t])} a.u.).`
        : `The prediction is the posterior after the last spike (dashed), ${gap} steps ago, spread by the movement model at every step since and multiplied at each of those steps by the nearly flat no-spike likelihood exp(−Λ(x)), which is not drawn: SD ${fmt(m.posterior_sd[last])} a.u. then, ${fmt(m.predictive_sd[t])} a.u. now.`;
    const offTarget =
      Math.abs(m.predictive_mean[t] - x[t]) > 2 * m.predictive_sd[t]
        ? ` It is centered at ${fmt(m.predictive_mean[t])} a.u., well away from the animal at ${fmt(x[t])} a.u.${t > data.conflict_step ? ", after the earlier conflict" : ""}.`
        : "";
    const field = `λ${subscript(cells[0] + 1)}(x)`;
    let product;
    if (manyCells) product = "The likelihood is the product of their place fields times exp(−Λ(x)).";
    else if (manySpikes) product = `The likelihood is ${field}${power(fired.length)}·exp(−Λ(x)), the field raised to the number of spikes.`;
    else product = `The likelihood is ${field}·exp(−Λ(x)).`;
    const head = `Step ${t}: ${listCells(cells)} fire${manyCells ? "" : "s"}, with place field${manyCells ? "s" : ""} at ${unique.map((c) => centers[c]).join(" and ")} a.u. ${prediction}${offTarget} ${product}`;
    if (!isConflict(t)) return `${head} The posterior: ${posteriorText}.`;
    // Described from each spike's HPD overlap, which the export pins at zero here.
    const regions = `The ${Math.round(data.coverage * 100)}% regions of the prediction and of ${manySpikes ? "each spike's" : "the spike's"} likelihood`;
    const overlaps = fired.map((s) => s.hpd.toFixed(2)).join(" and ");
    const verdict = fired.every((s) => s.hpd === 0) ? "do not overlap" : "barely overlap";
    return `${head} ${regions} ${verdict} (HPD overlap ${overlaps}): ${manySpikes ? "these spikes are" : "the spike is"} inconsistent with the prediction. The posterior lands between the two: ${posteriorText}. On its own, it gives no sign of the disagreement; the diagnostics below detect it by comparing each spike's likelihood with the prediction.`;
  }

  // ----------------------------------------------------------- Render

  function render() {
    const fired = spikes.get(t) ?? [];
    const firedCells = new Set(fired.map((s) => s.cell));
    const marker = x[t];

    // Dashed: the posterior the prediction spread from. At a spike step, that
    // is the posterior after the last spike, so a long gap's spread shows.
    const prediction = [];
    if (t > 0) {
      const from = fired.length ? lastSpikeBefore(t) : t - 1;
      prediction.push({ values: posteriorAt(from), color: cssVar("--posterior"), dashed: true, filled: false });
    }
    prediction.push({ values: predictiveAt(t), color: cssVar("--predictive") });
    charts.prediction.update({ series: prediction, marker, scaleMax: sharedMax });

    // The fields that fired are drawn last, on top.
    const order = fieldRows.map((_, c) => c).sort((a, b) => firedCells.has(a) - firedCells.has(b));
    charts.fields.update({
      series: order.map((c) =>
        firedCells.has(c)
          ? { values: fieldRows[c], color: cssVar("--field"), filled: false, width: 2.5 }
          : { values: fieldRows[c], color: cssVar("--field-muted"), filled: false, width: 1 },
      ),
      marker,
      scaleMax: data.place_fields.range[1],
    });

    const likelihoodSeries = [];
    if (fired.length) {
      likelihoodSeries.push({ values: data.exposure, color: cssVar("--curve-muted"), dashed: true, filled: false });
    }
    likelihoodSeries.push({ values: Array.from(likelihood.row(t)), color: cssVar("--likelihood") });
    charts.likelihood.update({ series: likelihoodSeries, marker });
    likelihoodLabel.textContent = fired.length
      ? "Likelihood of this step's spikes"
      : "Likelihood of no spike, exp(−Λ(x))";

    charts.posterior.update({
      series: [{ values: posteriorAt(t), color: cssVar("--posterior") }],
      marker,
      scaleMax: sharedMax,
    });

    stack.draw();
    stack.placeCursor(t + 0.5);
    caption.textContent = describe(frame !== null);
    status.textContent = `Step ${t} of ${nSteps - 1} · ${fired.length ? `spike${fired.length > 1 ? "s" : ""} from ${listCells(fired.map((s) => s.cell))}` : "no spike"}`;
    nextButton.disabled = t === nSteps - 1;
  }

  function go(step) {
    t = step;
    render();
  }

  // --------------------------------------------------------- Playback

  const holdAt = (step) => (spikes.has(step) ? SPIKE_HOLD : 1 / STEPS_PER_SECOND);

  function stop() {
    if (frame === null) return;
    cancelAnimationFrame(frame);
    frame = null;
    playButton.textContent = "▶ Play";
    caption.setAttribute("aria-live", "polite");
    render();
  }

  function play() {
    if (t === nSteps - 1) t = 0;
    // Announcing every step would flood screen readers.
    caption.setAttribute("aria-live", "off");
    playButton.textContent = "❚❚ Pause";
    let remaining = holdAt(t);
    let previous = null;
    const tick = (now) => {
      if (previous !== null) remaining -= (now - previous) / 1000;
      previous = now;
      if (remaining <= 0) {
        if (t === nSteps - 1) {
          stop();
          return;
        }
        t += 1;
        if (isConflict(t)) {
          stop();
          return;
        }
        // Carry over a little lateness, but never try to catch up after a stall.
        remaining = Math.max(0, remaining + holdAt(t));
        render();
      }
      frame = requestAnimationFrame(tick);
    };
    frame = requestAnimationFrame(tick);
    render();
  }

  playButton.addEventListener("click", () => {
    if (frame !== null) stop();
    else play();
  });
  nextButton.addEventListener("click", () => {
    stop();
    go(Math.min(nSteps - 1, t + 1));
  });
  restartButton.addEventListener("click", () => {
    stop();
    go(0);
  });

  render();
}
