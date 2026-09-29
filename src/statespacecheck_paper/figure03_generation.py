"""Generate Figure 3: per-spike diagnostics across an 8-phase simulation.

Steps a Bayesian decoder through three model-misfit conditions (remap,
history-dependent firing, drift) and two specificity controls (a replay event
and a sparse-population epoch), separated by clean-recovery windows, chosen to
span the metric-disagreement space (which misfits each of HPD overlap, KL
divergence, and the rank-based predictive p-value detects vs. misses).

The simulation + decode pipeline lives in
:func:`statespacecheck_paper.figure03_simulation.run_figure03_simulation`; this
module adds the pooled-realization threshold/summary estimate and the figure
composition on top, so the same simulation arrays drive both the static figure
here and the figure-3 simulation cache consumed by the interactive viewer.

Panel (a) is a time-series block for a single realization (seed
``config.random_seed``); panel (b) is a heatmap of the percent of spike events
flagged per phase per metric, reported as the median across ``N_REALIZATIONS``
independent realizations. Both the flag thresholds and the panel-(b)
percentages are stabilized by pooling ``N_REALIZATIONS`` realizations via
:func:`statespacecheck_paper.figure03_summary.estimate_realization_summary`,
rather than relying on a single noisy run.
"""

from __future__ import annotations

import dataclasses

import numpy as np

from statespacecheck_paper.diagnostics import HPD_COVERAGE
from statespacecheck_paper.figure03_plotting import compose_figure03
from statespacecheck_paper.figure03_protocol import STEP_SECONDS, Figure3Config
from statespacecheck_paper.figure03_simulation import (
    all_place_field_centers,
    run_figure03_simulation,
)
from statespacecheck_paper.figure03_summary import (
    N_REALIZATIONS,
    SUMMARY_ERROR_METRICS,
    SUMMARY_FLAG_METRICS,
    Figure3RealizationSummary,
    Figure3SummaryCondition,
    baseline_threshold_provenance,
    build_summary_conditions,
    estimate_realization_summary,
)
from statespacecheck_paper.paths import FIGURE03_SUMMARY_PATH, FIGURE_DIR
from statespacecheck_paper.scientific_artifacts import (
    inclusive_flag_rules,
    scientific_source_provenance,
    write_json_artifact,
)
from statespacecheck_paper.style import save_figure, set_figure_defaults


def _plain_condition_label(label: str) -> str:
    """Flatten a plotting label while preserving hyphenated line breaks."""
    return label.replace("-\n", "-").replace("\n", " ")


def conditions_by_id(config: Figure3Config) -> dict[str, Figure3SummaryCondition]:
    """Key each summary condition by its ``condition_id``, in summary column order."""
    return {condition.condition_id: condition for condition in build_summary_conditions(config)}


def figure03_summary_payload(
    config: Figure3Config,
    summary: Figure3RealizationSummary,
) -> dict[str, object]:
    """Return the canonical Figure 3 reported statistics as JSON-ready data."""
    conditions = build_summary_conditions(config)
    first_seed = config.random_seed
    thresholds = dataclasses.asdict(summary.diagnostic_thresholds)
    directions = {metric: direction for metric, direction in SUMMARY_FLAG_METRICS}
    return {
        "schema_version": 9,
        "figure": "figure03",
        # The step length and HPD coverage are constants rather than config
        # fields; recorded so the prose can quote them.
        "configuration": {
            **dataclasses.asdict(config),
            "step_seconds": STEP_SECONDS,
            "hpd_coverage": HPD_COVERAGE,
        },
        "realizations": {
            "count": summary.n_realizations,
            "first_seed": first_seed,
            "last_seed": first_seed + summary.n_realizations - 1,
        },
        "metric_order": [metric for metric, _ in SUMMARY_FLAG_METRICS],
        "flag_rules": inclusive_flag_rules(thresholds, directions),
        "threshold_provenance": baseline_threshold_provenance(
            config, summary.baseline_flagged_fractions
        ),
        "condition_order": [condition.condition_id for condition in conditions],
        "condition_labels": [_plain_condition_label(condition.label) for condition in conditions],
        "median_flag_percentages": summary.median_flag_percentages,
        "percentage_unit": "percent_of_spike_events",
        "error_metric_order": list(SUMMARY_ERROR_METRICS),
        "error_units": {"median_absolute_error": "position_units"},
        "median_decoding_error": summary.median_decoding_error,
        # Approximate across-realization standard errors, conditional on this
        # configuration. Published as data about how variable each median is;
        # they do not set the manuscript's printed precision, which follows the
        # policy in reported_values. See figure03_summary.median_standard_error.
        "standard_error_method": "order_statistic_interval_95",
        "median_flag_percentage_standard_errors": summary.flag_percentage_standard_errors,
        "median_decoding_error_standard_errors": summary.decoding_error_standard_errors,
        # Every realization's values, in seed order (first_seed to last_seed), so
        # the spread across realizations can be shown, not only the medians.
        "realization_flag_percentages": summary.realization_flag_percentages,
        "realization_decoding_error": summary.realization_decoding_error,
        "provenance": {"source": scientific_source_provenance()},
    }


def generate_figure03(
    config: Figure3Config | None = None,
    *,
    n_realizations: int = N_REALIZATIONS,
) -> None:
    """Run the figure-3 simulation + summary and save the composed figure.

    Parameters
    ----------
    config : Figure3Config, optional
        Figure-3 experimental configuration (timeline, place fields, controls).
        When omitted, uses the default ``Figure3Config()``, the manuscript
        configuration.
    n_realizations : int, default ``N_REALIZATIONS``
        Independent realizations pooled for the panel-(b) thresholds and
        median flag percentages.

    Returns
    -------
    None
        Saves ``figure03.{pdf,png}`` and ``figure03_summary.json`` under
        ``manuscript/figures/main``.
    """
    if config is None:
        config = Figure3Config()

    simulation_result = run_figure03_simulation(config)

    # The simulation appends a narrow sparse-population of cells; the raster
    # sorts all cells by field center.
    raster_place_field_centers = all_place_field_centers(
        config, simulation_result.sparse_place_field_centers
    )

    # Pool many realizations for stable thresholds (from the pooled opening
    # baseline, before the remap) and a stable median panel-(b) summary.
    realization_summary = estimate_realization_summary(config, n_realizations=n_realizations)
    print(f"Pooled thresholds: {realization_summary.diagnostic_thresholds}")
    print(
        "Median flag percentages [HPD, predictive p, KL] x "
        "[well-specified, remap, history, replay, drift, sparse population]:\n"
        f"{np.array2string(realization_summary.median_flag_percentages, precision=3)}"
    )
    print(
        "Median decoding error [median |error| (a.u.)] x "
        "[well-specified, remap, history, replay, drift, sparse population]:\n"
        f"{np.array2string(realization_summary.median_decoding_error, precision=3)}"
    )
    summary_path = write_json_artifact(
        FIGURE03_SUMMARY_PATH,
        figure03_summary_payload(config, realization_summary),
    )
    print(f"Saved canonical statistics to {summary_path}")

    # Panel (a) shows the single seed-1 realization; panel (b) shows the
    # pooled median percentages scored against the pooled-baseline thresholds.
    set_figure_defaults(context="paper")
    fig = compose_figure03(
        true_position=simulation_result.true_position,
        spike_counts=simulation_result.spike_counts.astype(np.float64),
        diagnostics=simulation_result.diagnostics,
        diagnostic_thresholds=realization_summary.diagnostic_thresholds,
        config=config,
        place_field_centers=raster_place_field_centers,
        median_flag_percentages=realization_summary.median_flag_percentages,
        median_decoding_error=realization_summary.median_decoding_error,
    )

    save_figure(FIGURE_DIR / "figure03", close=True, fig=fig)
    print(
        f"\nFigure 3 saved to {FIGURE_DIR / 'figure03'}.{{pdf,png}} "
        f"(panel b pooled over {n_realizations} realizations)"
    )
