// Per-spike goodness-of-fit diagnostics, ported from the paper's Python code.
//
// Mirrors statespacecheck_paper.diagnostics.compute_spike_event_diagnostics_from_rates
// for a single spike event, using the external statespacecheck package's
// highest_density_region / hpd_overlap / kl_divergence definitions.
// site/tests/metrics.test.mjs checks every function here against reference
// values computed by the Python implementation (site/tests/fixtures/metric_parity.json).
//
// Distributions are arrays over position bins. `rates` is an array of rows,
// shape (n_bins, n_cells): expected spikes per bin for each cell.

const FLOAT64_EPS = 2 ** -52;

function sum(values) {
  let total = 0;
  for (const v of values) total += v;
  return total;
}

/** Discretized Gaussian over `positionBins`, normalized to sum to 1. */
export function gaussianPredictive(positionBins, mean, std) {
  if (!(std > 0)) throw new RangeError(`std must be positive; got ${std}`);
  const weights = positionBins.map((x) => Math.exp(-0.5 * ((x - mean) / std) ** 2));
  const total = sum(weights);
  if (!(total > 0)) throw new RangeError("Gaussian predictive underflowed to zero on the grid");
  return weights.map((w) => w / total);
}

/** Normalized single-event likelihood: the firing cell's intensity over position. */
export function eventLikelihood(rates, cell) {
  const intensity = rates.map((row) => row[cell]);
  if (intensity.some((v) => !Number.isFinite(v) || v < 0)) {
    throw new RangeError("event intensities must be finite and nonnegative");
  }
  const total = sum(intensity);
  if (!(total > 0)) throw new RangeError("event intensity is zero at every position");
  return intensity.map((v) => v / total);
}

/**
 * Highest-density-region mask: every bin whose mass is at least the cutoff at
 * which the descending cumulative mass first reaches `coverage` of the total.
 * Ties at the cutoff are all included, as in statespacecheck.
 */
export function highestDensityRegion(distribution, coverage = 0.95) {
  const clean = distribution.map((v) => (Number.isFinite(v) ? v : 0));
  const total = sum(clean);
  if (!(total > 0) || !Number.isFinite(total)) return clean.map(() => false);
  const target = coverage * total;
  const sorted = [...clean].sort((a, b) => b - a);
  let cumulative = 0;
  let index = sorted.length - 1;
  for (let i = 0; i < sorted.length; i += 1) {
    cumulative += sorted[i];
    if (cumulative >= target) {
      index = i;
      break;
    }
  }
  const cutoff = sorted[index];
  return clean.map((v) => v >= cutoff);
}

/** |HPD_P ∩ HPD_Q| / min(|HPD_P|, |HPD_Q|); 0 when either region is empty. */
export function hpdOverlap(predictive, likelihood, coverage = 0.95) {
  const regionP = highestDensityRegion(predictive, coverage);
  const regionQ = highestDensityRegion(likelihood, coverage);
  let sizeP = 0;
  let sizeQ = 0;
  let both = 0;
  for (let i = 0; i < regionP.length; i += 1) {
    if (regionP[i]) sizeP += 1;
    if (regionQ[i]) sizeQ += 1;
    if (regionP[i] && regionQ[i]) both += 1;
  }
  const smaller = Math.min(sizeP, sizeQ);
  return smaller > 0 ? both / smaller : 0;
}

function normalized(values) {
  const clean = values.map((v) => (Number.isFinite(v) ? v : 0));
  const total = sum(clean);
  return clean.map((v) => (total > 0 ? v / total : 0));
}

/** D_KL(P || Q) in nats; Infinity where Q is zero but P is not. */
export function klDivergence(predictive, likelihood) {
  // statespacecheck normalizes, then scipy.stats.entropy normalizes again.
  const p = normalized(normalized(predictive));
  const q = normalized(normalized(likelihood));
  if (!(sum(p) > 0) || !(sum(q) > 0)) return Infinity;
  let divergence = 0;
  for (let i = 0; i < p.length; i += 1) {
    if (p[i] === 0) continue;
    if (q[i] === 0) return Infinity;
    divergence += p[i] * Math.log(p[i] / q[i]);
  }
  return Math.max(divergence, 0);
}

/**
 * Predictive probability that a randomly selected event comes from each cell:
 * f_pred(c) = Σ_x P(x) λ_c(x) / Σ_d Σ_x P(x) λ_d(x).
 */
export function predictiveCellProbabilities(predictive, rates) {
  const nCells = rates[0].length;
  const expected = new Array(nCells).fill(0);
  for (let x = 0; x < predictive.length; x += 1) {
    for (let c = 0; c < nCells; c += 1) expected[c] += predictive[x] * rates[x][c];
  }
  const total = sum(expected);
  if (!(total > 0)) throw new RangeError("predictive event intensity is zero");
  return expected.map((v) => v / total);
}

/**
 * Rank-based predictive p-value: the predictive probability of every cell at
 * least as unlikely as the one that fired. The tolerance matches the Python
 * implementation's allowance for summation-order rounding.
 */
export function predictivePvalue(predictive, rates, cell) {
  const probabilities = predictiveCellProbabilities(predictive, rates);
  const observed = probabilities[cell];
  const tolerance = FLOAT64_EPS * predictive.length * 16 * Math.max(...probabilities);
  let p = 0;
  for (const value of probabilities) {
    if (value <= observed + tolerance) p += value;
  }
  return Math.min(p, 1);
}

/** All three diagnostics for one spike from `cell` under `predictive`. */
export function spikeDiagnostics(predictive, rates, cell, coverage = 0.95) {
  const likelihood = eventLikelihood(rates, cell);
  return {
    likelihood,
    hpd_overlap: hpdOverlap(predictive, likelihood, coverage),
    predictive_pvalue: predictivePvalue(predictive, rates, cell),
    kl_divergence: klDivergence(predictive, likelihood),
  };
}

/** Apply a summary's inclusive flag rule ({comparison, threshold}). */
export function isFlagged(value, rule) {
  if (rule.comparison === "less_than_or_equal") return value <= rule.threshold;
  if (rule.comparison === "greater_than_or_equal") return value >= rule.threshold;
  throw new RangeError(`Unknown flag comparison ${rule.comparison}`);
}
