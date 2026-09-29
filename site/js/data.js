// Data loading, decoding, and text helpers shared by the page's components.
// The JSON files are written by statespacecheck_paper.site_export.

// Matplotlib's base-10 symlog transform with the paper's HPD-overlap settings
// (style.SYMLOG_LINTHRESH = 0.01, SYMLOG_LINSCALE = 1): linear below 0.01,
// logarithmic above, so values near zero stay separated from exact zeros.
export const SYMLOG_LINTHRESH = 0.01;
const SYMLOG_LINSCALE_ADJ = 1 / (1 - 1 / 10);

function symlog(value) {
  const magnitude = Math.abs(value);
  if (magnitude <= SYMLOG_LINTHRESH) return value * SYMLOG_LINSCALE_ADJ;
  return (
    Math.sign(value) *
    SYMLOG_LINTHRESH *
    (SYMLOG_LINSCALE_ADJ + Math.log10(magnitude / SYMLOG_LINTHRESH))
  );
}

// The paper's metrics in order (style.METRIC_SPECS). Names, labels, worse-fit
// directions, display transforms, the symlog axis, and its gridlines are
// checked against the Python registry by tests/test_site_metric_metadata.py.
export const METRICS = [
  {
    name: "hpd_overlap",
    label: "HPD overlap",
    // Direction of worse fit for the value, and on the plotted axis (as the
    // paper's figures label it).
    worse: "lower",
    plottedWorse: "below",
    color: "--hpd",
    display: (v) => v,
    // Plotted on a symlog axis, with gridlines at Figure 3's interior ticks.
    axis: symlog,
    gridlines: [0.01, 0.1],
    // Bounded, so its track always spans the whole range.
    range: [0, 1],
    displayLabel: "HPD overlap",
    format: (v) => (Number.isFinite(v) ? v.toFixed(2) : "—"),
  },
  {
    name: "predictive_pvalue",
    label: "Predictive p-value",
    worse: "lower",
    plottedWorse: "above",
    color: "--pvalue",
    // Plotted as −log(p) (natural log), as in the paper's figures. p > 0 by
    // construction; the floor only guards the axis against a degenerate value.
    display: (v) => -Math.log(Math.max(v, Number.MIN_VALUE)),
    displayLabel: "−log p",
    format: (v) => (!Number.isFinite(v) ? "—" : v >= 0.01 ? v.toFixed(2) : v.toExponential(1)),
  },
  {
    name: "kl_divergence",
    label: "KL divergence",
    worse: "higher",
    plottedWorse: "above",
    color: "--kl",
    display: (v) => v,
    displayLabel: "KL (nats)",
    format: (v) => (!Number.isFinite(v) ? "∞" : v >= 100 ? v.toFixed(0) : v.toFixed(2)),
  },
];

export async function loadJSON(path) {
  const response = await fetch(path);
  if (!response.ok) throw new Error(`${path}: HTTP ${response.status}`);
  return response.json();
}

/** Decode base64 uint8 rows written by site_export.encode_display_rows. */
export function decodeRows(encoded, nBins) {
  const binary = atob(encoded);
  const data = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i += 1) data[i] = binary.charCodeAt(i);
  if (data.length % nBins !== 0) {
    throw new Error(`${data.length} values do not divide into rows of ${nBins}`);
  }
  return {
    data,
    nBins,
    nRows: data.length / nBins,
    row(i) {
      return data.subarray(i * nBins, (i + 1) * nBins);
    },
  };
}

/**
 * Decode a site_export.heatmap_payload: rows scaled to their own maxima, each
 * row's maximum, and the shared color range. `values(i)` recovers row `i` in
 * the payload's units.
 */
export function decodeHeatmap(payload, nBins) {
  const rows = decodeRows(payload.rows, nBins);
  return {
    ...rows,
    rowMax: payload.row_max,
    range: payload.range,
    values(i) {
      const factor = payload.row_max[i] / 255;
      return Array.from(rows.row(i), (v) => v * factor);
    },
  };
}

/** Direction of worse fit in words, e.g. "lower = worse fit". */
export function worseFit(metric) {
  return `${metric.worse} = worse fit`;
}

/** Arrow for the plotted axis, as in the paper's figures, e.g. "↓ worse fit". */
export function plottedWorseFit(metric) {
  return metric.plottedWorse === "below" ? "↓ worse fit" : "↑ worse fit";
}

/**
 * Direction of worse fit plus the flag rule, e.g. "lower = worse fit; flagged
 * if ≤ 0.05". `thresholdText` is the threshold as the export prints it.
 */
function describeRule(metric, rule, thresholdText) {
  return `${worseFit(metric)}; ${flagRule(rule, thresholdText)}`;
}

// The summaries' inclusive flag comparisons (diagnostics.INCLUSIVE_FLAG_COMPARISONS)
// and their symbols; checked by tests/test_site_metric_metadata.py.
export const COMPARISON_SYMBOLS = {
  less_than_or_equal: "≤",
  greater_than_or_equal: "≥",
};

/** "flagged if ≤ 0.05" for a summary's flag rule, or "no fixed cutoff" without one. */
export function flagRule(rule, thresholdText) {
  if (!rule) return "no fixed cutoff";
  const symbol = COMPARISON_SYMBOLS[rule.comparison];
  if (symbol === undefined) throw new RangeError(`Unknown flag comparison ${rule.comparison}`);
  return `flagged if ${symbol} ${thresholdText}`;
}

/** Text for a reported value; `format` "count" adds thousands separators. */
export function formatMacro(value, format) {
  return format === "count" ? Number(value).toLocaleString("en-US") : value;
}

/** Link target of a DOI. */
export function doiUrl(doi) {
  return `https://doi.org/${doi}`;
}

/**
 * Fill every [data-macro] element with its value: a manuscript macro or a
 * number only the page states (manifest `macros` and `page_values`). Every
 * [data-doi-macro] link points at the DOI that macro holds.
 */
export function fillMacros(root, macros) {
  for (const element of root.querySelectorAll("[data-macro]")) {
    const name = element.dataset.macro;
    const value = macros[name];
    if (value === undefined) {
      console.warn(`Unknown macro ${name}`);
      continue;
    }
    element.textContent = formatMacro(value, element.dataset.format);
  }
  for (const element of root.querySelectorAll("[data-doi-macro]")) {
    const doi = macros[element.dataset.doiMacro];
    if (doi === undefined) {
      console.warn(`Unknown macro ${element.dataset.doiMacro}`);
      continue;
    }
    element.href = doiUrl(doi);
  }
}

/** Index of the entry in a sorted array nearest to `value`. */
export function nearestIndex(sorted, value) {
  if (sorted.length === 0) return -1;
  let lo = 0;
  let hi = sorted.length - 1;
  while (hi - lo > 1) {
    const mid = (lo + hi) >> 1;
    if (sorted[mid] < value) lo = mid;
    else hi = mid;
  }
  return Math.abs(sorted[lo] - value) <= Math.abs(sorted[hi] - value) ? lo : hi;
}

export function badge(flagged) {
  const span = document.createElement("span");
  span.className = `badge ${flagged ? "flagged" : "ok"}`;
  // "Not flagged" rather than "consistent": KL divergence measures difference,
  // and a high p-value does not by itself establish overlap.
  span.textContent = flagged ? "⚑ flagged" : "✓ not flagged";
  return span;
}

/**
 * A metric readout: name and flag rule, plus a value cell that `set(value,
 * flagged)` fills with the formatted value and its badge.
 */
export function readoutCard(metric, rule, thresholdText) {
  const card = document.createElement("div");
  card.className = "readout";
  card.style.setProperty("--metric-color", `var(${metric.color})`);
  card.innerHTML = `<div class="name">${metric.label}</div><div class="value"></div><div class="rule">${describeRule(metric, rule, thresholdText)}</div>`;
  const value = card.querySelector(".value");
  return {
    element: card,
    set(v, flagged) {
      value.replaceChildren(`${metric.format(v)} `, badge(flagged));
    },
  };
}

/** A visually hidden live region; `say(text)` announces to screen readers. */
export function liveRegion(container) {
  const region = document.createElement("div");
  region.className = "sr-only";
  region.setAttribute("aria-live", "polite");
  container.appendChild(region);
  return (text) => {
    region.textContent = text;
  };
}
