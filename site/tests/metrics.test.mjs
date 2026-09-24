// Parity test: the site's JavaScript diagnostics must reproduce the paper's
// Python implementation. Reference values come from
// statespacecheck_paper.site_export.metric_parity_fixture; regenerate them with
// `uv run python scripts/export_site_data.py`. Run with `node --test site/tests/`.

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";

import {
  gaussianPredictive,
  isFlagged,
  spikeDiagnostics,
} from "../js/metrics.js";

const fixture = JSON.parse(
  readFileSync(new URL("./fixtures/metric_parity.json", import.meta.url), "utf8"),
);

// Continuous metrics may differ only by summation-order rounding.
const RELATIVE_TOLERANCE = 1e-9;

function assertClose(actual, expected, label) {
  const scale = Math.max(1, Math.abs(expected));
  assert.ok(
    Math.abs(actual - expected) <= RELATIVE_TOLERANCE * scale,
    `${label}: ${actual} vs Python ${expected}`,
  );
}

test("fixture is non-trivial", () => {
  assert.ok(fixture.cases.length > 100);
  const overlaps = new Set(fixture.cases.map((c) => c.expected.hpd_overlap));
  assert.ok(overlaps.has(0) && overlaps.has(1), "cases span no-overlap and nested regions");
});

test("Gaussian predictive matches Python", () => {
  for (const { mean, std, predictive } of fixture.gaussian_cases) {
    const actual = gaussianPredictive(fixture.position_bins, mean, std);
    actual.forEach((v, i) => assertClose(v, predictive[i], `N(${mean}, ${std}) bin ${i}`));
  }
});

test("per-spike diagnostics match Python", () => {
  for (const [index, { predictive, ensemble, cell, expected }] of fixture.cases.entries()) {
    const actual = spikeDiagnostics(
      fixture.predictives[predictive],
      fixture.ensembles[ensemble],
      cell,
      fixture.coverage,
    );
    const label = `case ${index} (ensemble ${ensemble}, cell ${cell})`;
    // HPD overlap is a ratio of bin counts, so it must match exactly.
    assert.equal(actual.hpd_overlap, expected.hpd_overlap, `${label} HPD overlap`);
    assertClose(actual.predictive_pvalue, expected.predictive_pvalue, `${label} p-value`);
    assertClose(actual.kl_divergence, expected.kl_divergence, `${label} KL`);
  }
});

test("flag rules are inclusive", () => {
  assert.equal(isFlagged(0.05, { comparison: "less_than_or_equal", threshold: 0.05 }), true);
  assert.equal(isFlagged(0.06, { comparison: "less_than_or_equal", threshold: 0.05 }), false);
  assert.equal(isFlagged(4, { comparison: "greater_than_or_equal", threshold: 4 }), true);
  assert.throws(() => isFlagged(1, { comparison: "equal", threshold: 1 }));
});
