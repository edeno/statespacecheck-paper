// "Consistent is not identical": move a Gaussian prediction, choose the cell
// that fired, and watch the three per-spike diagnostics update.

import { cssVar, DistributionChart } from "./charts.js";
import { badge, describeRule, METRICS } from "./data.js";
import { gaussianPredictive, highestDensityRegion, isFlagged, spikeDiagnostics } from "./metrics.js";

const PRESETS = {
  consistent: { ensemble: "place_cells", mean: 50, std: 5, cell: 5 },
  conflicting: { ensemble: "place_cells", mean: 25, std: 4, cell: 7 },
  nested: { ensemble: "place_cells", mean: 47, std: 1.5, cell: 5 },
  broad: { ensemble: "sparse_epoch", mean: 30, std: 15, cell: 13 },
};

// The spread slider is logarithmic so both near-point and very broad
// predictions are reachable.
const STD_MIN = 0.5;
const STD_MAX = 60;
const sliderToStd = (s) => STD_MIN * (STD_MAX / STD_MIN) ** (s / 1000);
const stdToSlider = (std) => (1000 * Math.log(std / STD_MIN)) / Math.log(STD_MAX / STD_MIN);

function verdict(flags) {
  const names = METRICS.filter((m) => flags[m.name]).map((m) => m.label);
  if (names.length === 0) {
    return "None of the diagnostics flag this spike: its likelihood is consistent with the prediction.";
  }
  if (names.length === METRICS.length) {
    return "All three diagnostics flag this spike: the prediction and the spike point to different places.";
  }
  if (names.length === 1 && flags.kl_divergence) {
    return "Only KL divergence flags this spike. HPD overlap and the predictive check find it consistent; KL divergence responds to any difference between the two distributions, such as a difference in spread.";
  }
  return `Flagged by ${names.join(" and ")}.`;
}

export function initPlayground(root, data) {
  const bins = data.position_bins;
  const ensembles = Object.fromEntries(data.ensembles.map((e) => [e.ensemble_id, e]));
  const state = { ...PRESETS.consistent };

  const ensembleButtons = root.querySelectorAll("[data-ensemble]");
  const presetButtons = root.querySelectorAll("[data-preset]");
  const meanInput = root.querySelector("#pg-mean");
  const stdInput = root.querySelector("#pg-std");
  const meanOutput = root.querySelector("#pg-mean-out");
  const stdOutput = root.querySelector("#pg-std-out");
  const cellOutput = root.querySelector("#pg-cell-out");
  const verdictBox = root.querySelector("#pg-verdict");

  const chart = new DistributionChart(root.querySelector("#pg-chart"), {
    positionBins: bins,
    xLabel: "Position (a.u.)",
    cellStrip: () => ({
      rates: ensembles[state.ensemble].rates,
      selectable: ensembles[state.ensemble].selectable_cells,
      onSelect: (cell) => {
        state.cell = cell;
        render();
      },
    }),
  });

  const readouts = {};
  const readoutRoot = root.querySelector("#pg-readouts");
  for (const metric of METRICS) {
    const card = document.createElement("div");
    card.className = "readout";
    card.style.setProperty("--metric-color", metric.color);
    card.innerHTML = `<div class="name">${metric.label}</div><div class="value"></div><div class="rule">${describeRule(metric.name, data.flag_rules[metric.name])}</div>`;
    readoutRoot.appendChild(card);
    readouts[metric.name] = card;
  }

  function render() {
    const ensemble = ensembles[state.ensemble];
    if (!ensemble.selectable_cells.includes(state.cell)) state.cell = ensemble.selectable_cells[0];
    meanInput.value = state.mean;
    stdInput.value = stdToSlider(state.std);
    meanOutput.textContent = `${state.mean.toFixed(1)} a.u.`;
    stdOutput.textContent = `${state.std.toFixed(state.std < 10 ? 1 : 0)} a.u.`;
    cellOutput.textContent = `cell ${state.cell + 1}, field at ${data.cell_centers[state.cell]} a.u.`;
    for (const button of ensembleButtons) {
      button.setAttribute("aria-pressed", String(button.dataset.ensemble === state.ensemble));
    }

    const predictive = gaussianPredictive(bins, state.mean, state.std);
    const result = spikeDiagnostics(predictive, ensemble.rates, state.cell, data.coverage);
    chart.update({
      series: [
        { values: predictive, color: cssVar("--predictive") },
        { values: result.likelihood, color: cssVar("--likelihood") },
      ],
      bands: [
        { mask: highestDensityRegion(predictive, data.coverage), color: cssVar("--predictive") },
        { mask: highestDensityRegion(result.likelihood, data.coverage), color: cssVar("--likelihood") },
      ],
      selectedCell: state.cell,
      cellCenters: data.cell_centers,
    });

    const flags = {};
    for (const metric of METRICS) {
      const value = result[metric.name];
      flags[metric.name] = isFlagged(value, data.flag_rules[metric.name]);
      const valueBox = readouts[metric.name].querySelector(".value");
      valueBox.replaceChildren(`${metric.format(value)} `, badge(flags[metric.name]));
    }
    verdictBox.textContent = verdict(flags);
  }

  meanInput.min = bins[0];
  meanInput.max = bins[bins.length - 1];
  meanInput.addEventListener("input", () => {
    state.mean = Number(meanInput.value);
    render();
  });
  stdInput.addEventListener("input", () => {
    state.std = sliderToStd(Number(stdInput.value));
    render();
  });
  for (const button of ensembleButtons) {
    button.addEventListener("click", () => {
      state.ensemble = button.dataset.ensemble;
      render();
    });
  }
  for (const button of presetButtons) {
    button.addEventListener("click", () => {
      Object.assign(state, PRESETS[button.dataset.preset]);
      render();
    });
  }
  render();
}
