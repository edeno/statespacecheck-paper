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
from pathlib import Path

import numpy as np

from statespacecheck_paper.figure03_plotting import compose_figure03
from statespacecheck_paper.figure03_protocol import Figure3Config
from statespacecheck_paper.figure03_simulation import run_figure03_simulation
from statespacecheck_paper.figure03_summary import (
    SUMMARY_FLAG_METRICS,
    Figure3RealizationSummary,
    build_summary_conditions,
    estimate_realization_summary,
)
from statespacecheck_paper.scientific_artifacts import (
    inclusive_flag_rules,
    scientific_source_provenance,
    write_json_artifact,
)
from statespacecheck_paper.style import save_figure, set_figure_defaults

# Number of independent realizations pooled to stabilize the panel-(b)
# summary. A single run's flag thresholds and per-phase percentages are
# noisy (the KL 99th-percentile threshold varies ~17% across seeds, and
# the remap flag percentage swings with the trajectory); pooling many
# realizations gives a stable threshold and a median per-phase summary.
# The seed-1 realization shown in panel (a) is one of these.
N_REALIZATIONS = 100
FIGURE03_SUMMARY_PATH = Path("manuscript/figures/main/figure03_summary.json")
FIGURE03_CONDITION_IDS = (
    "well_specified",
    "remap",
    "history_dependent",
    "replay",
    "drift",
    "sparse_population",
)


def _plain_condition_label(label: str) -> str:
    """Flatten a plotting label while preserving hyphenated line breaks."""
    return label.replace("-\n", "-").replace("\n", " ")


def figure03_summary_payload(
    config: Figure3Config,
    summary: Figure3RealizationSummary,
) -> dict[str, object]:
    """Return the canonical Figure 3 reported statistics as JSON-ready data."""
    conditions = build_summary_conditions(config)
    if len(conditions) != len(FIGURE03_CONDITION_IDS):
        raise ValueError(
            "Figure 3 condition identifiers are out of sync with "
            f"build_summary_conditions: {len(FIGURE03_CONDITION_IDS)} IDs for "
            f"{len(conditions)} conditions."
        )
    first_seed = config.random_seed
    thresholds = dataclasses.asdict(summary.diagnostic_thresholds)
    directions = {metric: direction for metric, direction in SUMMARY_FLAG_METRICS}
    return {
        "schema_version": 2,
        "figure": "figure03",
        "configuration": dataclasses.asdict(config),
        "realizations": {
            "count": summary.n_realizations,
            "first_seed": first_seed,
            "last_seed": first_seed + summary.n_realizations - 1,
        },
        "metric_order": [metric for metric, _ in SUMMARY_FLAG_METRICS],
        "flag_rules": inclusive_flag_rules(thresholds, directions),
        "condition_order": list(FIGURE03_CONDITION_IDS),
        "condition_labels": [_plain_condition_label(condition.label) for condition in conditions],
        "median_flag_percentages": summary.median_flag_percentages,
        "percentage_unit": "percent_of_spike_events",
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
        When omitted, uses the manuscript configuration with drift momentum
        0.88.
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
    assert config.place_field_centers is not None, "place_field_centers must be initialized"
    raster_place_field_centers = np.append(
        config.place_field_centers,
        np.asarray(simulation_result.sparse_place_field_centers),
    )

    # Pool many realizations for a stable threshold (from the pooled
    # clean-baseline windows) and a stable median panel-(b) summary.
    realization_summary = estimate_realization_summary(config, n_realizations=n_realizations)
    print(f"Pooled thresholds: {realization_summary.diagnostic_thresholds}")
    print(
        "Median flag percentages [HPD, predictive p, KL] x "
        "[well-specified, remap, history, replay, drift, sparse population]:\n"
        f"{np.array2string(realization_summary.median_flag_percentages, precision=3)}"
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
    )

    save_figure("manuscript/figures/main/figure03", close=True, fig=fig)
    print(
        f"\nFigure 3 saved to manuscript/figures/main/figure03.{{pdf,png}} "
        f"(panel b pooled over {n_realizations} realizations)"
    )
