// Data loading, decoding, and text helpers shared by the page's components.
// The JSON files are written by statespacecheck_paper.site_export.

// Matplotlib's base-10 symlog transform with the paper's HPD-overlap settings
// (linthresh = 0.01, linscale = 1): linear below 0.01, logarithmic above, so
// values near zero stay separated from exact zeros.
const SYMLOG_LINTHRESH = 0.01;
const SYMLOG_LINSCALE_ADJ = 1 / (1 - 1 / 10);

export function symlog(value) {
  const magnitude = Math.abs(value);
  if (magnitude <= SYMLOG_LINTHRESH) return value * SYMLOG_LINSCALE_ADJ;
  return (
    Math.sign(value) *
    SYMLOG_LINTHRESH *
    (SYMLOG_LINSCALE_ADJ + Math.log10(magnitude / SYMLOG_LINTHRESH))
  );
}

export const METRICS = [
  {
    name: "hpd_overlap",
    label: "HPD overlap",
    color: "var(--hpd)",
    display: (v) => v,
    // Plotted on a symlog axis, as in the paper's figures.
    axis: symlog,
    gridlines: [0.01, 0.1],
    displayLabel: "HPD overlap",
    format: (v) => (Number.isFinite(v) ? v.toFixed(2) : "—"),
  },
  {
    name: "predictive_pvalue",
    label: "Predictive p-value",
    color: "var(--pvalue)",
    // Plotted as −log(p) (natural log), as in the paper's figures. p > 0 by
    // construction; the floor only guards the axis against a degenerate value.
    display: (v) => -Math.log(Math.max(v, Number.MIN_VALUE)),
    displayLabel: "−log p",
    format: (v) => (!Number.isFinite(v) ? "—" : v >= 0.01 ? v.toFixed(2) : v.toExponential(1)),
  },
  {
    name: "kl_divergence",
    label: "KL divergence",
    color: "var(--kl)",
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

/** Human-readable flag rule, e.g. "flagged if ≤ 0.05". */
export function describeRule(metric, rule) {
  if (!rule) return "no fixed cutoff";
  const symbol = rule.comparison === "less_than_or_equal" ? "≤" : "≥";
  const threshold = Number.isInteger(rule.threshold)
    ? String(rule.threshold)
    : rule.threshold.toPrecision(3).replace(/\.?0+$/, "");
  return `flagged if ${symbol} ${threshold}`;
}

/** Fill every [data-macro] element with the manuscript's reported value. */
export function fillMacros(root, macros) {
  for (const element of root.querySelectorAll("[data-macro]")) {
    const name = element.dataset.macro;
    const value = macros[name];
    if (value === undefined) {
      console.warn(`Unknown macro ${name}`);
      continue;
    }
    element.textContent =
      element.dataset.format === "count" ? Number(value).toLocaleString("en-US") : value;
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
