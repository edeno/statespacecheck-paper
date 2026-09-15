"""Generate Figure 3: per-spike diagnostics across a 10-phase simulation.

Steps a Bayesian decoder through four model-misfit conditions (an incoherent
remap, history-dependent firing, drift, a coherent reflected map) and two
controls (a replay event and a matched low-information regime), separated by
clean-recovery windows in which the latent trajectory follows the decoder's own
transition law. The conditions span the metric-disagreement space (which
interventions each of HPD overlap, the rank-based predictive p-value, and KL
divergence flags, and at what rate under a matched null).

The simulation + decode pipeline lives in
:func:`statespacecheck_paper.figure03_simulation.run_figure03_simulation`; this
module adds the independently calibrated thresholds, the pooled-realization
summary, and the figure composition on top, so the same simulation arrays drive
both the static figure here and the figure-3 simulation cache consumed by the
interactive viewer.

Panel (a) is a time-series block for a single realization (seed
``config.random_seed``); panel (b) is a heatmap of the percent of spike events
flagged per condition per metric, reported as the median across
``N_REALIZATIONS`` independent realizations, with the filtered-posterior
accuracy rows beneath; panel (c) shows the paired per-realization distributions
behind those medians. The flag thresholds are calibrated on
``N_CALIBRATION_REALIZATIONS`` separate matched-null sessions
(:func:`statespacecheck_paper.figure03_summary.estimate_calibration_thresholds`),
never on the evaluated realizations.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np

from statespacecheck_paper.figure03_plotting import (
    compose_figure03,
    compose_figure03_realizations,
)
from statespacecheck_paper.figure03_protocol import Figure3Config
from statespacecheck_paper.figure03_simulation import run_figure03_simulation
from statespacecheck_paper.figure03_summary import (
    CONDITION_IDS,
    RANK_CALIBRATION_ALPHAS,
    SUMMARY_ACCURACY_METRICS,
    SUMMARY_FLAG_METRICS,
    Figure3RealizationSummary,
    baseline_threshold_provenance,
    build_summary_conditions,
    estimate_calibration_thresholds,
    estimate_realization_summary,
)
from statespacecheck_paper.scientific_artifacts import (
    inclusive_flag_rules,
    scientific_source_provenance,
    write_json_artifact,
)
from statespacecheck_paper.style import save_figure, set_figure_defaults

# Number of independent phased realizations pooled for the panel-(b)/(c)
# summary. Per-condition percentages are trajectory-dependent (the remap
# column swings with how much of the scrambled map the trajectory visits), so
# the median and the full per-realization distribution are both reported.
# The seed-1 realization shown in panel (a) is one of these.
N_REALIZATIONS = 100
# Number of independent matched-null sessions that calibrate the HPD and KL
# thresholds. Their seeds are offset from the evaluated realizations so the
# calibration sample never overlaps the evaluated data.
N_CALIBRATION_REALIZATIONS = 100
# Default parallelism for the simulation sweeps (each realization is an
# independent process; -1 uses every core).
DEFAULT_N_JOBS = -1
FIGURE03_SUMMARY_PATH = Path("manuscript/figures/main/figure03_summary.json")
FIGURE03_REALIZATIONS_FIGURE_PATH = "manuscript/figures/supplementary/figure03_realizations"
FIGURE03_CONDITION_IDS = CONDITION_IDS


def _nan_to_none(values: np.ndarray) -> list[object]:
    """Nested lists with NaN (a realization with no events) replaced by ``None``."""
    array = np.asarray(values, dtype=float)
    return [
        _nan_to_none(row) if row.ndim else (None if np.isnan(row) else float(row)) for row in array
    ]


def _plain_condition_label(label: str) -> str:
    """Flatten a plotting label while preserving hyphenated line breaks."""
    return label.replace("-\n", "-").replace("\n", " ")


def figure03_summary_payload(
    config: Figure3Config,
    summary: Figure3RealizationSummary,
) -> dict[str, object]:
    """Return the canonical Figure 3 reported statistics as JSON-ready data."""
    conditions = build_summary_conditions(config)
    if [c.condition_id for c in conditions] != list(FIGURE03_CONDITION_IDS):
        raise ValueError(
            "Figure 3 condition identifiers are out of sync with build_summary_conditions."
        )
    first_seed = config.random_seed
    calibration = summary.calibration
    thresholds = dataclasses.asdict(calibration.diagnostic_thresholds)
    directions = {metric: direction for metric, direction in SUMMARY_FLAG_METRICS}
    metric_order = [metric for metric, _ in SUMMARY_FLAG_METRICS]
    return {
        "schema_version": 6,
        "figure": "figure03",
        "configuration": dataclasses.asdict(config),
        "realizations": {
            "count": summary.n_realizations,
            "first_seed": first_seed,
            "last_seed": first_seed + summary.n_realizations - 1,
        },
        "calibration": {
            "count": calibration.n_calibration_realizations,
            "first_seed": calibration.first_calibration_seed,
            "last_seed": (
                calibration.first_calibration_seed + calibration.n_calibration_realizations - 1
            ),
            "steps_per_session": calibration.calibration_steps_per_session,
            "n_pooled_events": calibration.n_pooled_events,
            "pooled_null_flag_percentages": dict(
                zip(metric_order, calibration.pooled_null_flag_percentages, strict=True)
            ),
            "per_realization_null_flag_percentages": (
                calibration.per_realization_null_flag_percentages
            ),
            "hpd_threshold_tie_percent": calibration.hpd_threshold_tie_percent,
            "rank_pvalue_tail_alphas": list(RANK_CALIBRATION_ALPHAS),
            "rank_pvalue_tail_percentages": calibration.rank_pvalue_tail_percentages,
        },
        "metric_order": metric_order,
        "flag_rules": inclusive_flag_rules(thresholds, directions),
        "threshold_provenance": baseline_threshold_provenance(config),
        "condition_order": list(FIGURE03_CONDITION_IDS),
        "condition_labels": [_plain_condition_label(condition.label) for condition in conditions],
        "condition_model_components": [condition.model_component for condition in conditions],
        "median_flag_percentages": summary.median_flag_percentages,
        "pooled_flag_percentages": summary.pooled_flag_percentages,
        "flag_percentages_by_realization": _nan_to_none(summary.flag_percentages_by_realization),
        "event_counts_by_realization": summary.event_counts_by_realization,
        "percentage_unit": "percent_of_spike_events",
        "accuracy_metric_order": list(SUMMARY_ACCURACY_METRICS),
        "accuracy_units": {
            "median_absolute_error": "position_units",
            "filtered_hpd_coverage_percent": "percent_of_time_steps",
            "median_filtered_hpd_size": "position_bins",
            "median_predictive_hpd_size": "position_bins",
        },
        "median_decoding_accuracy": summary.median_decoding_accuracy,
        "decoding_accuracy_by_realization": summary.decoding_accuracy_by_realization,
        "replay_represented_accuracy": {
            "metric_order": ["median_absolute_error", "filtered_hpd_coverage_percent"],
            "by_realization": summary.replay_represented_accuracy,
            "median": np.median(summary.replay_represented_accuracy, axis=0),
        },
        "phase_ordinary_spike_rates_hz": {
            "by_realization": summary.phase_spike_rates_by_realization,
            "median": np.median(summary.phase_spike_rates_by_realization, axis=0),
        },
        "sparse_population": {
            "sparse_cell_flag_percentages_by_realization": _nan_to_none(
                summary.sparse_cell_flag_percentages_by_realization
            ),
            "sparse_cell_event_counts_by_realization": (
                summary.sparse_cell_event_counts_by_realization
            ),
            "median_sparse_cell_flag_percentages": _nan_to_none(
                np.nanmedian(summary.sparse_cell_flag_percentages_by_realization, axis=0)
            ),
            "median_sparse_cell_event_count": float(
                np.median(summary.sparse_cell_event_counts_by_realization)
            ),
        },
        "matched_null_rank_pvalue_tail_percentages": (
            summary.matched_null_rank_pvalue_tail_percentages
        ),
        # Approximate across-realization standard errors, conditional on this
        # configuration. Published as data about how variable each median is;
        # they do not set the manuscript's printed precision, which follows the
        # policy in reported_values. See figure03_summary.median_standard_error.
        "standard_error_method": "order_statistic_interval_95",
        "median_flag_percentage_standard_errors": _nan_to_none(
            summary.flag_percentage_standard_errors
        ),
        "median_decoding_accuracy_standard_errors": summary.decoding_accuracy_standard_errors,
        "provenance": {"source": scientific_source_provenance()},
    }


def generate_figure03(
    config: Figure3Config | None = None,
    *,
    n_realizations: int = N_REALIZATIONS,
    n_calibration_realizations: int = N_CALIBRATION_REALIZATIONS,
    n_jobs: int = DEFAULT_N_JOBS,
    summary_path: Path = FIGURE03_SUMMARY_PATH,
    figure_path: str = "manuscript/figures/main/figure03",
    realizations_figure_path: str = FIGURE03_REALIZATIONS_FIGURE_PATH,
) -> Figure3RealizationSummary:
    """Run the figure-3 calibration, simulation, and summary; save the figure.

    Parameters
    ----------
    config : Figure3Config, optional
        Figure-3 experimental configuration (timeline, place fields, controls).
        When omitted, uses the manuscript configuration.
    n_realizations : int, default ``N_REALIZATIONS``
        Independent phased realizations pooled for panels (b) and (c).
    n_calibration_realizations : int, default ``N_CALIBRATION_REALIZATIONS``
        Independent matched-null sessions that calibrate the thresholds.
    n_jobs : int, default ``DEFAULT_N_JOBS``
        Parallel workers for the simulation sweeps.
    summary_path, figure_path, realizations_figure_path
        Output locations of the summary, the main figure, and the
        supplementary per-realization figure.

    Returns
    -------
    Figure3RealizationSummary
        The summary written to ``summary_path`` (also used by sensitivity runs).
    """
    if config is None:
        config = Figure3Config()

    simulation_result = run_figure03_simulation(config)

    # The simulation appends the sparse-population cells; the raster sorts
    # all cells by field center.
    assert config.place_field_centers is not None, "place_field_centers must be initialized"
    raster_place_field_centers = np.append(
        config.place_field_centers,
        np.asarray(simulation_result.sparse_place_field_centers),
    )

    # Calibrate the thresholds on independent matched-null sessions, then
    # score the phased realizations against them.
    calibration = estimate_calibration_thresholds(
        config, n_calibration_realizations=n_calibration_realizations, n_jobs=n_jobs
    )
    print(f"Calibrated thresholds: {calibration.diagnostic_thresholds}")
    print(
        "Realized pooled null flag percentages [HPD, predictive p, KL]: "
        f"{np.array2string(calibration.pooled_null_flag_percentages, precision=3)}"
    )
    realization_summary = estimate_realization_summary(
        config, calibration=calibration, n_realizations=n_realizations, n_jobs=n_jobs
    )
    print(
        "Median flag percentages [HPD, predictive p, KL] x "
        f"{list(FIGURE03_CONDITION_IDS)}:\n"
        f"{np.array2string(realization_summary.median_flag_percentages, precision=3)}"
    )
    print(
        f"Median decoding accuracy {list(SUMMARY_ACCURACY_METRICS)} x conditions:\n"
        f"{np.array2string(realization_summary.median_decoding_accuracy, precision=3)}"
    )
    written = write_json_artifact(
        summary_path,
        figure03_summary_payload(config, realization_summary),
    )
    print(f"Saved canonical statistics to {written}")

    # Panel (a) shows the single seed-1 realization; panel (b) shows the
    # pooled summary scored against the calibrated thresholds.
    set_figure_defaults(context="paper")
    fig = compose_figure03(
        true_position=simulation_result.true_position,
        spike_counts=simulation_result.spike_counts.astype(np.float64),
        diagnostics=simulation_result.diagnostics,
        diagnostic_thresholds=realization_summary.diagnostic_thresholds,
        config=config,
        place_field_centers=raster_place_field_centers,
        median_flag_percentages=realization_summary.median_flag_percentages,
        median_decoding_accuracy=realization_summary.median_decoding_accuracy,
        represented_position=simulation_result.represented_position,
    )
    save_figure(figure_path, close=True, fig=fig)

    # The paired per-realization distributions behind the medians.
    realizations_fig = compose_figure03_realizations(
        config, realization_summary.flag_percentages_by_realization
    )
    save_figure(realizations_figure_path, close=True, fig=realizations_fig)
    print(
        f"\nFigure 3 saved to {figure_path}.{{pdf,png}} "
        f"(panels b-c pooled over {n_realizations} realizations; thresholds calibrated on "
        f"{n_calibration_realizations} matched-null sessions)"
    )
    return realization_summary
