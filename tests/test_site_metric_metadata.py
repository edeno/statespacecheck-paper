"""The website's metric table must agree with the paper's metric registry.

``site/js/data.js`` restates each metric's order, name, worse-fit directions,
display transform, and symlog axis for the browser. This test imports that
module with Node and checks it against :data:`style.METRIC_SPECS` and
:data:`diagnostics.METRIC_FLAG_DIRECTIONS`. It needs Node (``$NODE`` or
``node`` on the path) and is skipped without it.
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
from statespacecheck_paper.paths import REPO_ROOT
from statespacecheck_paper.plotting import negative_log_pvalue
from statespacecheck_paper.style import (
    METRIC_SPECS,
    SYMLOG_LINSCALE,
    SYMLOG_LINTHRESH,
)

DATA_JS = REPO_ROOT / "site" / "js" / "data.js"

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

_WORSE_WORDS = {"below": "lower", "above": "higher"}


@pytest.fixture(scope="module")
def site_metrics() -> list[dict[str, Any]]:
    node = os.environ.get("NODE") or shutil.which("node")
    if node is None:
        pytest.skip("Node is not available; set $NODE or put node on the path")
    completed = subprocess.run(
        [
            node,
            "--input-type=module",
            "-e",
            _NODE_SCRIPT,
            DATA_JS.as_uri(),
            json.dumps(PROBES),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    metrics: list[dict[str, Any]] = json.loads(completed.stdout)
    return metrics


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
