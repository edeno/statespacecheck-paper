"""Guard shared metric display metadata against raw flag-direction drift."""

from __future__ import annotations

from statespacecheck_paper.figure04_generation import (
    FIGURE4_DIAGNOSTIC_THRESHOLDS,
    FIGURE4_METRIC_DIRECTIONS,
)
from statespacecheck_paper.style import METRIC_SPEC_BY_NAME, METRIC_SPECS, MetricSpec


def _raw_worse_direction(spec: MetricSpec) -> str:
    """Worse-fit direction on the raw (untransformed) metric values.

    ``neg_log_p`` is monotonically decreasing, so it flips the plotted-axis
    direction; every other transform preserves it.
    """
    if spec.display_transform == "neg_log_p":
        return "above" if spec.plotted_worse == "below" else "below"
    return spec.plotted_worse


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


def test_metric_specs_agree_with_figure03_flag_directions() -> None:
    from statespacecheck_paper.figure03_summary import SUMMARY_FLAG_METRICS

    # Same metrics, same order as the shared spec table.
    assert tuple(name for name, _ in SUMMARY_FLAG_METRICS) == tuple(s.name for s in METRIC_SPECS)
    # Raw flag direction is recoverable from the display metadata.
    assert {name: direction for name, direction in SUMMARY_FLAG_METRICS} == {
        spec.name: _raw_worse_direction(spec) for spec in METRIC_SPECS
    }


def test_metric_specs_agree_with_figure04_flag_directions() -> None:
    # Figure 4 flags only the two thresholded metrics; every direction it records
    # must match the shared spec's raw direction.
    assert set(FIGURE4_METRIC_DIRECTIONS) == set(FIGURE4_DIAGNOSTIC_THRESHOLDS)
    for name, direction in FIGURE4_METRIC_DIRECTIONS.items():
        assert direction == _raw_worse_direction(METRIC_SPEC_BY_NAME[name])
