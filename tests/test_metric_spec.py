"""Guard shared metric display metadata against raw flag-direction drift."""

from __future__ import annotations

import numpy as np

from statespacecheck_paper.diagnostics import METRIC_FLAG_DIRECTIONS
from statespacecheck_paper.figure04_protocol import (
    FIGURE04_DIAGNOSTIC_THRESHOLDS,
    FIGURE04_METRIC_DIRECTIONS,
)
from statespacecheck_paper.style import METRIC_SPEC_BY_NAME, METRIC_SPECS


def test_plotted_direction_flips_only_under_the_neg_log_p_transform() -> None:
    # Low HPD overlap and low p-values flag misfit, as does high KL; -log(p)
    # plots the p-value's worse side at the top of its axis.
    assert {spec.name: spec.plotted_worse for spec in METRIC_SPECS} == {
        "hpd_overlap": "below",
        "predictive_pvalue": "above",
        "kl_divergence": "above",
    }


def test_metric_spec_arrow_matches_plotted_direction() -> None:
    arrows = {spec.name: spec.worse_fit_direction for spec in METRIC_SPECS}
    assert arrows == {
        "hpd_overlap": "↓ Worse fit",
        "predictive_pvalue": "↑ Worse fit",
        "kl_divergence": "↑ Worse fit",
    }


def test_metric_spec_event_attr_matches_name() -> None:
    for spec in METRIC_SPECS:
        assert spec.event_attr == f"event_{spec.name}"


def test_metric_specs_list_the_flagged_metrics_in_order() -> None:
    assert tuple(s.name for s in METRIC_SPECS) == tuple(METRIC_FLAG_DIRECTIONS)


def test_figure04_flags_its_thresholded_metrics_by_the_shared_rule() -> None:
    assert set(FIGURE04_METRIC_DIRECTIONS) == set(FIGURE04_DIAGNOSTIC_THRESHOLDS)
    for name, direction in FIGURE04_METRIC_DIRECTIONS.items():
        assert direction == METRIC_FLAG_DIRECTIONS[name]


def test_to_display_applies_each_metric_transform_to_values_and_thresholds() -> None:
    values = np.array([0.5, 0.05])
    predictive = METRIC_SPEC_BY_NAME["predictive_pvalue"]
    np.testing.assert_array_equal(predictive.to_display(values), -np.log(values))
    assert predictive.to_display(0.05) == -np.log(0.05)
    np.testing.assert_array_equal(METRIC_SPEC_BY_NAME["kl_divergence"].to_display(values), values)
