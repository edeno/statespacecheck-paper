"""The website's metric and condition tables must agree with the paper's registries.

``site/js/data.js`` restates each metric's order, name, worse-fit directions,
display transform, and symlog axis for the browser. This test imports that
module with Node and checks it against :data:`style.METRIC_SPECS` and
:data:`diagnostics.METRIC_FLAG_DIRECTIONS`. ``site/js/players.js`` keys its
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

from statespacecheck_paper.diagnostics import METRIC_FLAG_DIRECTIONS
from statespacecheck_paper.paths import FIGURE03_SUMMARY_PATH, REPO_ROOT
from statespacecheck_paper.plotting import negative_log_pvalue
from statespacecheck_paper.style import (
    METRIC_SPECS,
    SYMLOG_LINSCALE,
    SYMLOG_LINTHRESH,
)

DATA_JS = REPO_ROOT / "site" / "js" / "data.js"
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

_WORSE_WORDS = {"below": "lower", "above": "higher"}


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
