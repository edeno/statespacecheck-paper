"""The website's metric and condition tables must agree with the paper's registries.

``site/js/data.js`` restates each metric's order, name, label, worse-fit
directions, display transform, symlog axis, and gridlines for the browser, and
``data.js`` and ``site/js/metrics.js`` restate the summaries' flag comparisons.
These tests import those modules with Node and check them against
:data:`style.METRIC_SPECS`, :data:`diagnostics.METRIC_FLAG_DIRECTIONS`,
:data:`diagnostics.INCLUSIVE_FLAG_COMPARISONS`, and Figure 3's symlog ticks.
``site/js/players.js`` keys its
per-condition prose by the Figure-3 summary's condition IDs, checked against
the committed summary's ``condition_order``. The tests need Node (``$NODE`` or
``node`` on the path) and are skipped without it.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from typing import Any

import numpy as np
import pytest
from matplotlib.scale import SymmetricalLogTransform

from statespacecheck_paper.diagnostics import (
    INCLUSIVE_FLAG_COMPARISONS,
    METRIC_FLAG_DIRECTIONS,
    flag_mask,
)
from statespacecheck_paper.figure03_plotting import FIGURE03_SYMLOG_YTICKS
from statespacecheck_paper.paths import FIGURE03_SUMMARY_PATH, REPO_ROOT
from statespacecheck_paper.plotting import negative_log_pvalue
from statespacecheck_paper.style import (
    METRIC_SPECS,
    SYMLOG_LINSCALE,
    SYMLOG_LINTHRESH,
)

DATA_JS = REPO_ROOT / "site" / "js" / "data.js"
METRICS_JS = REPO_ROOT / "site" / "js" / "metrics.js"
PLAYERS_JS = REPO_ROOT / "site" / "js" / "players.js"

# Values in (0, 1], spanning the symlog axis's linear and logarithmic ranges;
# every metric's display transform is defined on them.
PROBES = (0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.5, 1.0)

# Evaluate each metric's display transform and axis on the probes in Node.
_NODE_SCRIPT = """
const { METRICS } = await import(process.argv[1]);
const probes = JSON.parse(process.argv[2]);
console.log(JSON.stringify(METRICS.map((m) => ({
  name: m.name,
  label: m.label,
  gridlines: m.gridlines ?? null,
  range: m.range ?? null,
  worse: m.worse,
  plottedWorse: m.plottedWorse,
  display: probes.map(m.display),
  axis: m.axis ? probes.map(m.axis) : null,
}))));
"""

# The condition player's per-condition tables.
_CONDITIONS_SCRIPT = """
const { CONDITION_TEXT, OPEN_ON_FLAGGED } = await import(process.argv[1]);
console.log(JSON.stringify({
  text: Object.keys(CONDITION_TEXT),
  openOnFlagged: [...OPEN_ON_FLAGGED],
}));
"""

# How the page applies and states each flag comparison: data.js's symbols and
# its rule text, and metrics.js's flag decision for values below, at, and above
# a threshold of 1. An unknown comparison must throw in both.
_COMPARISONS_SCRIPT = """
const { COMPARISON_SYMBOLS, flagRule } = await import(process.argv[1]);
const { isFlagged } = await import(process.argv[2]);
const throws = (f) => { try { f(); return false; } catch { return true; } };
const unknown = { comparison: "equal", threshold: 1 };
console.log(JSON.stringify({
  comparisons: Object.fromEntries(Object.keys(COMPARISON_SYMBOLS).map((comparison) => {
    const rule = { comparison, threshold: 1 };
    return [comparison, {
      text: flagRule(rule, "1"),
      flagged: [0.5, 1, 1.5].map((value) => isFlagged(value, rule)),
    }];
  })),
  unknownThrows: [throws(() => flagRule(unknown, "1")), throws(() => isFlagged(1, unknown))],
}));
"""

_WORSE_WORDS = {"below": "lower", "above": "higher"}

# The site's readouts show each diagnostic's raw value, so each metric carries
# its registry label, except the p-value: the figures plot -log(p) and label it
# so (``MetricSpec.label``), while the page prints the p-value itself.
_SITE_LABEL_OVERRIDES = {"predictive_pvalue": "Predictive p-value"}


def _node_json(script: str, *args: str) -> Any:
    """Run an ES-module ``script`` in Node with ``args`` and parse its JSON output."""
    node = os.environ.get("NODE") or shutil.which("node")
    if node is None:
        pytest.skip("Node is not available; set $NODE or put node on the path")
    completed = subprocess.run(
        [node, "--input-type=module", "-e", script, *args],
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(completed.stdout)


@pytest.fixture(scope="module")
def site_metrics() -> list[dict[str, Any]]:
    metrics: list[dict[str, Any]] = _node_json(_NODE_SCRIPT, DATA_JS.as_uri(), json.dumps(PROBES))
    return metrics


def test_condition_prose_covers_the_summary_conditions() -> None:
    site = _node_json(_CONDITIONS_SCRIPT, PLAYERS_JS.as_uri())
    order = json.loads(FIGURE03_SUMMARY_PATH.read_text(encoding="utf-8"))["condition_order"]
    assert site["text"] == order
    assert set(site["openOnFlagged"]) <= set(order)


def test_site_lists_the_registry_metrics_in_order(site_metrics: list[dict[str, Any]]) -> None:
    assert [m["name"] for m in site_metrics] == [spec.name for spec in METRIC_SPECS]


def test_site_labels_match_the_registry(site_metrics: list[dict[str, Any]]) -> None:
    assert [m["label"] for m in site_metrics] == [
        _SITE_LABEL_OVERRIDES.get(spec.name, spec.label) for spec in METRIC_SPECS
    ]
    # The override names a value the figures do not label directly.
    assert all(
        spec.label != _SITE_LABEL_OVERRIDES[spec.name]
        for spec in METRIC_SPECS
        if spec.name in _SITE_LABEL_OVERRIDES
    )


def test_site_symlog_gridlines_are_the_figure03_ticks(site_metrics: list[dict[str, Any]]) -> None:
    for site, spec in zip(site_metrics, METRIC_SPECS, strict=True):
        if not spec.symlog_axis:
            continue
        low, high = site["range"]
        assert site["gridlines"] == [tick for tick in FIGURE03_SYMLOG_YTICKS if low < tick < high]


def test_site_flag_comparisons_match_the_summaries() -> None:
    site = _node_json(_COMPARISONS_SCRIPT, DATA_JS.as_uri(), METRICS_JS.as_uri())
    assert sorted(site["comparisons"]) == sorted(INCLUSIVE_FLAG_COMPARISONS.values())
    probes = np.array([0.5, 1.0, 1.5])
    for direction, comparison in INCLUSIVE_FLAG_COMPARISONS.items():
        entry = site["comparisons"][comparison]
        assert entry["flagged"] == flag_mask(probes, 1.0, direction).tolist(), comparison
        symbol = "≤" if direction == "below" else "≥"
        assert entry["text"] == f"flagged if {symbol} 1", comparison
    assert site["unknownThrows"] == [True, True]


def test_site_worse_fit_directions_match_the_flag_rule(
    site_metrics: list[dict[str, Any]],
) -> None:
    for site, spec in zip(site_metrics, METRIC_SPECS, strict=True):
        assert site["worse"] == _WORSE_WORDS[METRIC_FLAG_DIRECTIONS[spec.name]], spec.name
        assert site["plottedWorse"] == spec.plotted_worse, spec.name


def test_site_display_transform_matches_the_registry(
    site_metrics: list[dict[str, Any]],
) -> None:
    probes = np.asarray(PROBES)
    for site, spec in zip(site_metrics, METRIC_SPECS, strict=True):
        expected = negative_log_pvalue(probes) if spec.display_transform == "neg_log_p" else probes
        np.testing.assert_allclose(site["display"], expected, rtol=1e-12, err_msg=spec.name)


def test_site_symlog_axis_matches_the_figures(site_metrics: list[dict[str, Any]]) -> None:
    transform = SymmetricalLogTransform(10, SYMLOG_LINTHRESH, SYMLOG_LINSCALE)
    for site, spec in zip(site_metrics, METRIC_SPECS, strict=True):
        if not spec.symlog_axis:
            assert site["axis"] is None, spec.name
            continue
        assert site["axis"] is not None, spec.name
        np.testing.assert_allclose(
            site["axis"], transform.transform(np.asarray(PROBES)), rtol=1e-12, err_msg=spec.name
        )
