"""Figure-3b summary: calibrated thresholds, per-condition flags, and accuracy.

This module builds the Figure-3b summary in two independent stages:

1. **Calibration** (:func:`estimate_calibration_thresholds`): simulate
   independent matched-null sessions (trajectory drawn from the decoder's own
   initial law and transition matrix, spikes from its own observation model)
   and set the HPD-overlap and KL-divergence flag thresholds from their
   pooled per-event diagnostics. The same sessions report the *realized*
   null flag rates at those thresholds -- which need not equal the nominal
   percentiles because HPD overlap is discrete and ties are flagged
   inclusively -- and the rank-based p-value's empirical tail against its
   super-uniform bound.
2. **Evaluation** (:func:`estimate_realization_summary`): simulate the phased
   sessions on disjoint seeds, score every spike event in every condition
   against the calibrated thresholds, and summarize across realizations:
   median flag percentages, paired per-realization distributions and event
   counts, decoding accuracy of the filtered posterior (point error, HPD
   coverage of the true state, HPD-region size), the predictive-region size
   that the diagnostics themselves see, replay accuracy against the
   represented trajectory, per-phase spike rates, and sparse-cell versus
   all-event denominators in the low-information regime.

Percentages are on a 0-100 scale; decoding errors are in position units;
region sizes are in grid bins. Per-realization fractions and pooled event
fractions are kept distinct throughout.

It imports :mod:`figure03_protocol` (the config + replay window),
:mod:`figure03_simulation` (the phased and matched-null simulations), and
:mod:`diagnostics` (the thresholds/diagnostic containers).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal, cast

import numpy as np
import statespacecheck as ssc
from joblib import Parallel, delayed
from numpy.typing import NDArray

from statespacecheck_paper.diagnostics import (
    BASELINE_HPD_OVERLAP_QUANTILE,
    BASELINE_KL_DIVERGENCE_QUANTILE,
    FIXED_PREDICTIVE_PVALUE_CUTOFF,
    DecodingDiagnostics,
    DiagnosticThresholds,
    compute_baseline_diagnostic_thresholds,
)
from statespacecheck_paper.figure03_protocol import (
    Figure3Config,
    PhaseBoundary,
    compute_replay_step_window,
)
from statespacecheck_paper.figure03_simulation import (
    run_figure03_simulation,
    run_matched_null_simulation,
)

SUMMARY_FLAG_METRICS: tuple[tuple[str, Literal["below", "above"]], ...] = (
    ("hpd_overlap", "below"),
    ("predictive_pvalue", "below"),
    ("kl_divergence", "above"),
)

# Row order of the per-condition decoding-accuracy block beneath the flag
# heatmap. All are per time step within the condition's windows:
# - median absolute error of the filtered-posterior mean vs the physical
#   position (position units);
# - percentage of steps whose true position (nearest grid cell) lies inside
#   the filtered posterior's HPD region at ``config.hpd_coverage``;
# - median size (grid bins) of that filtered HPD region;
# - median size (grid bins) of the one-step predictive HPD region, the
#   distribution the diagnostics compare against.
SUMMARY_ACCURACY_METRICS: tuple[str, ...] = (
    "median_absolute_error",
    "filtered_hpd_coverage_percent",
    "median_filtered_hpd_size",
    "median_predictive_hpd_size",
)

# Tail probabilities at which the rank-based p-value's empirical tail is
# compared with its super-uniform bound under the matched null.
RANK_CALIBRATION_ALPHAS: tuple[float, ...] = (0.01, 0.05, 0.10)

# Seed offset separating the calibration sessions from the evaluated phased
# realizations (which use ``config.random_seed, +1, ...``).
DEFAULT_CALIBRATION_SEED_OFFSET = 1_000

# Column identifiers of the Figure-3b summary, in heatmap order.
CONDITION_IDS: tuple[str, ...] = (
    "matched_null",
    "recovery",
    "remap",
    "history_dependent",
    "replay",
    "drift",
    "reflected_map",
    "sparse_population",
)


@dataclass(frozen=True)
class Figure3SummaryCondition:
    """One column of the Figure-3b summary heatmap.

    Parameters
    ----------
    condition_id : str
        Stable identifier from :data:`CONDITION_IDS`.
    label : str
        Column header (may contain a newline for a two-line label).
    step_windows : tuple of (int, int)
        Half-open ``[t0, t1)`` time-step windows aggregated into this
        column. The recovery column concatenates the clean-recovery windows,
        so this is a tuple of pairs rather than a single pair.
    model_component : str
        Model component the column's intervention perturbs (``"Observation"``,
        ``"Transition"``, or ``"—"`` for reference and control columns). This
        labels the experimental design, not an inference made by the metrics.
    """

    condition_id: str
    label: str
    step_windows: tuple[tuple[int, int], ...]
    model_component: str


def build_summary_conditions(config: Figure3Config) -> list[Figure3SummaryCondition]:
    """Phase columns for the Figure-3b summary heatmap.

    Single source of truth for the heatmap's columns, shared by the
    single-run flag-percentage helper (:func:`compute_condition_flag_percentages`)
    and the multi-realization path (:func:`estimate_realization_summary`)
    so the column order, time windows, and component labels cannot drift
    out of sync.

    The first column ("Matched null") is the opening baseline, whose
    trajectory and spikes are drawn exactly from the decoder's own model; it
    is the out-of-sample null reference for the independently calibrated
    thresholds. The second ("Recovery") pools the clean-recovery windows that
    follow each perturbation (with the replay sub-window carved out); their
    dynamics are matched too, but each begins with a transient from the
    preceding perturbation. The remaining columns are the four misfit
    windows and the two controls in their chronological order.

    Parameters
    ----------
    config : Figure3Config
        Provides the phase-boundary ladder.

    Returns
    -------
    list of Figure3SummaryCondition
        Eight columns in heatmap order.
    """
    bnd = config.phase_boundaries
    t_remap_start = bnd[PhaseBoundary.REMAP_START]
    t_remap_end = bnd[PhaseBoundary.REMAP_END]
    t_recovery1_end = bnd[PhaseBoundary.RECOVERY1_END]
    t_hist_dep_end = bnd[PhaseBoundary.HIST_DEP_END]
    t_recovery2_end = bnd[PhaseBoundary.RECOVERY2_END]
    t_drift_end = bnd[PhaseBoundary.DRIFT_END]
    t_recovery3_end = bnd[PhaseBoundary.RECOVERY3_END]
    t_reflect_end = bnd[PhaseBoundary.REFLECT_END]
    t_recovery4_end = bnd[PhaseBoundary.RECOVERY4_END]
    t_sparse_pop_end = bnd[PhaseBoundary.SPARSE_POP_END]
    # The replay event sits inside clean-recovery 2; carve it out of the
    # recovery pool (it is scored in its own column) so its spikes neither
    # define the reference nor dilute the recovery rate.
    r0, r1 = compute_replay_step_window(config)
    return [
        Figure3SummaryCondition("matched_null", "Matched\nnull", ((0, t_remap_start),), "—"),
        Figure3SummaryCondition(
            "recovery",
            "Recovery",
            (
                (t_remap_end, t_recovery1_end),
                (t_hist_dep_end, r0),
                (r1, t_recovery2_end),
                (t_drift_end, t_recovery3_end),
                (t_reflect_end, t_recovery4_end),
            ),
            "—",
        ),
        Figure3SummaryCondition("remap", "Remap", ((t_remap_start, t_remap_end),), "Observation"),
        Figure3SummaryCondition(
            "history_dependent",
            "History-\ndep.",
            ((t_recovery1_end, t_hist_dep_end),),
            "Observation",
        ),
        Figure3SummaryCondition("replay", "Replay", ((r0, r1),), "—"),
        Figure3SummaryCondition("drift", "Drift", ((t_recovery2_end, t_drift_end),), "Transition"),
        Figure3SummaryCondition(
            "reflected_map",
            "Reflected\nmap",
            ((t_recovery3_end, t_reflect_end),),
            "Observation",
        ),
        Figure3SummaryCondition(
            "sparse_population",
            "Sparse\npopulation",
            ((t_recovery4_end, t_sparse_pop_end),),
            "—",
        ),
    ]


def _flag_percentage(
    values: NDArray[np.floating],
    threshold: float,
    direction: str,
    *,
    empty_value: float | None = None,
) -> float:
    """Percent of ``values`` flagged as poor fit at ``threshold``.

    Parameters
    ----------
    values : np.ndarray, shape (n,)
        Per-spike diagnostic values. Every event must carry a value.
    threshold : float
        Flag threshold.
    direction : {"below", "above"}
        ``"below"`` flags ``values <= threshold``; ``"above"`` flags
        ``values >= threshold``.
    empty_value : float, optional
        Value returned when ``values`` is empty (a realization with no spike
        events in the condition). When ``None`` (default) an empty input
        raises, which is appropriate for pooled inputs.

    Returns
    -------
    float
        Percent (0–100) of values flagged.
    """
    if values.size == 0:
        if empty_value is not None:
            return empty_value
        raise ValueError("Cannot compute a flag percentage for a condition with no spike events")
    if np.any(np.isnan(values)) or np.any(np.isneginf(values)):
        raise ValueError("Per-event diagnostic values must not contain NaN or -inf")
    if direction == "below":
        flagged = float(np.mean(values <= threshold))
    elif direction == "above":
        flagged = float(np.mean(values >= threshold))
    else:
        raise ValueError(f"direction must be 'below' or 'above'; got {direction!r}")
    return 100.0 * flagged


def extract_condition_flag_values(
    diagnostics: DecodingDiagnostics | Mapping[str, NDArray[np.floating] | NDArray[np.intp]],
    conditions: list[Figure3SummaryCondition],
) -> list[list[NDArray[np.floating]]]:
    """Collect per-spike-event diagnostic values per metric per column.

    Works on the **per-event** arrays (``event_time_ind`` /
    ``event_hpd_overlap`` / ``event_kl_divergence`` / ``event_predictive_pvalue``),
    one value per spike event, so that a bin with several spikes from one
    cell contributes several values — matching the "percentage of spike
    events" the figure reports.

    Parameters
    ----------
    diagnostics : DecodingDiagnostics or Mapping[str, NDArray]
        Source of the per-event arrays ``event_time_ind`` (int) and
        ``event_{hpd_overlap,kl_divergence,predictive_pvalue}`` (float), each of
        shape ``(n_events,)``.
    conditions : list of Figure3SummaryCondition
        Heatmap columns from :func:`build_summary_conditions`.

    Returns
    -------
    list of list of np.ndarray
        Nested list indexed ``[metric_index][column_index]``; each leaf is
        a 1-D array of the per-event values for that metric whose
        event time falls inside that column's half-open time windows. Metric
        order follows :data:`SUMMARY_FLAG_METRICS`.
    """

    def _get(name: str) -> NDArray[np.generic]:
        arr = (
            getattr(diagnostics, name)
            if isinstance(diagnostics, DecodingDiagnostics)
            else diagnostics[name]
        )
        return cast("NDArray[np.generic]", arr)

    event_time = np.asarray(_get("event_time_ind"))
    out: list[list[NDArray[np.floating]]] = []
    for metric_key, _direction in SUMMARY_FLAG_METRICS:
        ev = np.asarray(_get("event_" + metric_key), dtype=float)
        per_window: list[NDArray[np.floating]] = []
        for col in conditions:
            mask = _condition_event_mask(event_time, col)
            vals = ev[mask]
            if np.any(np.isnan(vals)) or np.any(np.isneginf(vals)):
                raise ValueError(
                    f"event_{metric_key} contains an undefined value in condition {col.label!r}"
                )
            per_window.append(vals)
        out.append(per_window)
    return out


def _condition_event_mask(
    event_time: NDArray[np.generic], condition: Figure3SummaryCondition
) -> NDArray[np.bool_]:
    times = np.asarray(event_time, dtype=np.int64)
    mask = np.zeros(times.shape, dtype=bool)
    for t0, t1 in condition.step_windows:
        mask |= (times >= t0) & (times < t1)
    return mask


def flag_percentages_from_values(
    values: list[list[NDArray[np.floating]]],
    diagnostic_thresholds: DiagnosticThresholds,
    *,
    empty_value: float | None = None,
) -> NDArray[np.floating]:
    """Percent flagged per metric per column from pre-extracted values.

    Parameters
    ----------
    values : list of list of np.ndarray
        Nested ``[metric_index][column_index]`` finite values, as returned
        by :func:`extract_condition_flag_values`.
    diagnostic_thresholds : DiagnosticThresholds
        Flag thresholds (one per metric).
    empty_value : float, optional
        Value for a column with no events (see :func:`_flag_percentage`);
        ``None`` raises on an empty column.

    Returns
    -------
    np.ndarray, shape (3, n_columns)
        Percent (0–100) flagged. Rows follow :data:`SUMMARY_FLAG_METRICS`.
    """
    n_columns = len(values[0]) if values else 0
    frac = np.zeros((len(SUMMARY_FLAG_METRICS), n_columns))
    for i, (metric_key, direction) in enumerate(SUMMARY_FLAG_METRICS):
        threshold = float(getattr(diagnostic_thresholds, metric_key))
        for j in range(n_columns):
            frac[i, j] = _flag_percentage(
                values[i][j], threshold, direction, empty_value=empty_value
            )
    return frac


def compute_condition_flag_percentages(
    diagnostics: DecodingDiagnostics | Mapping[str, NDArray[np.floating]],
    diagnostic_thresholds: DiagnosticThresholds,
    conditions: list[Figure3SummaryCondition],
) -> NDArray[np.floating]:
    """Percent of spike events flagged per metric per phase column.

    Convenience wrapper around :func:`extract_condition_flag_values` +
    :func:`flag_percentages_from_values` for the single-realization renderer.

    Parameters
    ----------
    diagnostics : DecodingDiagnostics or Mapping[str, NDArray]
        Diagnostic matrices for a single realization.
    diagnostic_thresholds : DiagnosticThresholds
        Flag thresholds (one per metric).
    conditions : list of Figure3SummaryCondition
        Heatmap columns from :func:`build_summary_conditions`.

    Returns
    -------
    np.ndarray, shape (3, n_columns)
        Percent (0–100) flagged. Rows follow :data:`SUMMARY_FLAG_METRICS`;
        columns follow ``conditions``.
    """
    return flag_percentages_from_values(
        extract_condition_flag_values(diagnostics, conditions), diagnostic_thresholds
    )


def _nearest_bin_index(
    position: NDArray[np.floating], position_bins: NDArray[np.floating]
) -> NDArray[np.intp]:
    """Map (possibly off-grid) true positions to their nearest grid cell.

    Grid-valued positions map to themselves; the continuous drift trajectory
    maps to the nearest bin center.
    """
    index: NDArray[np.intp] = np.abs(position[:, None] - position_bins[None, :]).argmin(axis=1)
    return index


def compute_condition_decoding_accuracy(
    posterior: NDArray[np.floating],
    predictive: NDArray[np.floating],
    position_bins: NDArray[np.floating],
    true_position: NDArray[np.floating],
    conditions: list[Figure3SummaryCondition],
    *,
    hpd_coverage: float,
) -> NDArray[np.floating]:
    """Per-condition decoding accuracy and uncertainty of the filtered posterior.

    Scores the decoder's state estimate against the stored true position
    inside each summary column's time windows. Unlike the flag percentages,
    which are per spike event, these are per time step, so every step in a
    window counts whether or not a spike occurred.

    Parameters
    ----------
    posterior : np.ndarray, shape (n_time, n_bins)
        Filtered posterior ``p(x_t | y_{1:t})`` over the position grid. Rows
        need not be normalized, but each must carry positive finite mass.
    predictive : np.ndarray, shape (n_time, n_bins)
        One-step predictive distribution ``p(x_t | y_{1:t-1})``.
    position_bins : np.ndarray, shape (n_bins,)
        Position-grid bin centres (position units).
    true_position : np.ndarray, shape (n_time,)
        True position at each time step. For the replay control this is the
        animal's fixed physical position, so the replay column measures the
        decoded-versus-physical gap by construction. Off-grid values (the
        drift phase) are mapped to the nearest grid cell for the coverage
        metric.
    conditions : list of Figure3SummaryCondition
        Heatmap columns from :func:`build_summary_conditions`.
    hpd_coverage : float
        Coverage of the HPD regions whose coverage of the truth and size are
        reported.

    Returns
    -------
    np.ndarray, shape (4, n_columns)
        Rows follow :data:`SUMMARY_ACCURACY_METRICS`.

    Raises
    ------
    ValueError
        On shape mismatch, non-finite input, a posterior row with
        non-positive mass, or a column with no time steps.
    """
    posterior = np.asarray(posterior, dtype=float)
    predictive = np.asarray(predictive, dtype=float)
    position_bins = np.asarray(position_bins, dtype=float)
    true_position = np.asarray(true_position, dtype=float)
    if posterior.ndim != 2:
        raise ValueError(f"posterior must be 2-D (n_time, n_bins); got shape {posterior.shape}")
    n_time, n_bins = posterior.shape
    if predictive.shape != (n_time, n_bins):
        raise ValueError(f"predictive must have shape {(n_time, n_bins)}; got {predictive.shape}")
    if position_bins.shape != (n_bins,):
        raise ValueError(
            f"position_bins must have shape ({n_bins},) to match posterior; "
            f"got {position_bins.shape}"
        )
    if true_position.shape != (n_time,):
        raise ValueError(
            f"true_position must have shape ({n_time},) to match posterior; "
            f"got {true_position.shape}"
        )
    if not (
        np.all(np.isfinite(posterior))
        and np.all(np.isfinite(predictive))
        and np.all(np.isfinite(position_bins))
        and np.all(np.isfinite(true_position))
    ):
        raise ValueError("posterior, predictive, position_bins, and true_position must be finite")
    mass = posterior.sum(axis=1)
    if np.any(mass <= 0.0):
        raise ValueError("Every posterior row must carry positive mass")

    posterior_mean = (posterior @ position_bins) / mass
    abs_error = np.abs(posterior_mean - true_position)
    filtered_region = ssc.highest_density_region(posterior, coverage=hpd_coverage)
    predictive_region = ssc.highest_density_region(predictive, coverage=hpd_coverage)
    truth_index = _nearest_bin_index(true_position, position_bins)
    covered = filtered_region[np.arange(n_time), truth_index]
    filtered_size = filtered_region.sum(axis=1)
    predictive_size = predictive_region.sum(axis=1)

    out = np.zeros((len(SUMMARY_ACCURACY_METRICS), len(conditions)))
    for j, col in enumerate(conditions):
        mask = np.zeros(n_time, dtype=bool)
        for t0, t1 in col.step_windows:
            mask[t0:t1] = True
        if not mask.any():
            raise ValueError(f"Condition {col.label!r} contains no time steps")
        out[0, j] = float(np.median(abs_error[mask]))
        out[1, j] = 100.0 * float(np.mean(covered[mask]))
        out[2, j] = float(np.median(filtered_size[mask]))
        out[3, j] = float(np.median(predictive_size[mask]))
    return out


def median_standard_error(samples: NDArray[np.floating], axis: int = 0) -> NDArray[np.floating]:
    """Approximate a median's standard error from an order-statistic interval.

    This is an *approximate* standard error, and it is conditional on the
    simulation setup: it describes how much the median of these
    ``n_realizations`` seeds would move under a rerun with different seeds,
    for this configuration, and nothing beyond that. It is published in the
    summary as uncertainty in the aggregated median, not as the spread of
    individual realizations (which the summary reports directly); it does
    not control how the manuscript prints the value.

    The estimate is deterministic (no bootstrap resampling, so the artifact
    is reproducible byte-for-byte). The interval uses sample ranks without
    fitting a distribution to the realization values. For ``n`` samples,
    nominal 95% bounds for the median run between order
    statistics ``k`` and ``n - k + 1`` with ``k = floor(n / 2 - z sqrt(n) / 2)``
    (``z = 1.96``); the returned value is that interval's half-width divided
    by ``z``. Converting interval width to an SE this way is a normal-scale
    approximation; it is not an exact or guaranteed conservative SE for skewed
    or discrete distributions.

    Parameters
    ----------
    samples : np.ndarray
        Sample values; ``axis`` indexes the realizations.
    axis : int, default 0
        Axis to reduce.

    Returns
    -------
    standard_error : np.ndarray
        Approximate standard error of the median, with ``axis`` removed.
        Returns zero when the selected order statistics coincide, including
        when every sample is identical; this does not establish zero
        population uncertainty. NaN samples (a realization with no events in
        a condition) are dropped from that entry's sample.

    Notes
    -----
    Below roughly eight samples the order-statistic bounds collapse onto the
    extremes and the result is the sample range over ``2 z``. This is a coarse
    fallback for small-``n`` callers such as the fast test fixtures, with no
    guarantee of conservative uncertainty or nominal interval coverage.

    Examples
    --------
    >>> rng = np.random.default_rng(0)
    >>> se = median_standard_error(rng.normal(size=(1000, 2)))
    >>> bool(np.all(se < 0.1))
    True
    """
    samples = np.moveaxis(np.asarray(samples, dtype=float), axis, 0)
    z = 1.96
    out = np.empty(samples.shape[1:], dtype=float)
    # Columns are handled one at a time only because NaN entries (a
    # realization with no events in a condition) change each column's count.
    for idx in np.ndindex(*samples.shape[1:]):
        column = samples[(slice(None), *idx)]
        column = column[~np.isnan(column)]
        n_samples = column.size
        if n_samples == 0:
            out[idx] = np.nan
            continue
        # Convert the 1-based order statistics to 0-based indices, clipped so
        # a small sample cannot index outside the array.
        lower = max(int(np.floor(n_samples / 2 - z * np.sqrt(n_samples) / 2)) - 1, 0)
        upper = min(n_samples - lower - 1, n_samples - 1)
        ordered = np.sort(column)
        out[idx] = (ordered[upper] - ordered[lower]) / (2.0 * z)
    return out


@dataclass(frozen=True)
class Figure3Calibration:
    """Thresholds and realized null behavior from independent matched-null sessions.

    Parameters
    ----------
    diagnostic_thresholds : DiagnosticThresholds
        Flag thresholds: HPD overlap at the pooled
        ``BASELINE_HPD_OVERLAP_QUANTILE`` quantile, KL divergence at the pooled
        ``BASELINE_KL_DIVERGENCE_QUANTILE`` quantile, and the fixed
        predictive-p-value cutoff.
    n_calibration_realizations : int
        Number of independent matched-null sessions pooled.
    first_calibration_seed, calibration_steps_per_session : int
        Seeds ``first, first + 1, ...`` and the length of each session.
    n_pooled_events : int
        Total spike events pooled across the calibration sessions.
    pooled_null_flag_percentages : np.ndarray, shape (3,)
        Percent of pooled calibration events flagged at the thresholds
        (in-sample realized null rate; not necessarily the nominal 1%/5%/1%).
    per_realization_null_flag_percentages : np.ndarray, shape (n_calibration_realizations, 3)
        The same rate within each calibration session.
    hpd_threshold_tie_percent : float
        Percent of pooled calibration events whose HPD overlap equals the
        threshold exactly (the inclusive rule flags all of them).
    rank_pvalue_tail_percentages : np.ndarray, shape (len(RANK_CALIBRATION_ALPHAS),)
        Percent of pooled calibration events with rank p-value ``<= alpha``
        for each alpha in :data:`RANK_CALIBRATION_ALPHAS`; under the matched
        null this must not exceed ``100 * alpha`` beyond sampling error
        (super-uniformity).
    """

    diagnostic_thresholds: DiagnosticThresholds
    n_calibration_realizations: int
    first_calibration_seed: int
    calibration_steps_per_session: int
    n_pooled_events: int
    pooled_null_flag_percentages: NDArray[np.floating]
    per_realization_null_flag_percentages: NDArray[np.floating]
    hpd_threshold_tie_percent: float
    rank_pvalue_tail_percentages: NDArray[np.floating]

    def __post_init__(self) -> None:
        if self.n_calibration_realizations < 1:
            raise ValueError("n_calibration_realizations must be >= 1")
        if self.pooled_null_flag_percentages.shape != (len(SUMMARY_FLAG_METRICS),):
            raise ValueError("pooled_null_flag_percentages must have one entry per metric")
        if self.per_realization_null_flag_percentages.shape != (
            self.n_calibration_realizations,
            len(SUMMARY_FLAG_METRICS),
        ):
            raise ValueError(
                "per_realization_null_flag_percentages must be "
                "(n_calibration_realizations, n_metrics)"
            )
        if self.rank_pvalue_tail_percentages.shape != (len(RANK_CALIBRATION_ALPHAS),):
            raise ValueError("rank_pvalue_tail_percentages must have one entry per alpha")
        for name in (
            "pooled_null_flag_percentages",
            "per_realization_null_flag_percentages",
            "rank_pvalue_tail_percentages",
        ):
            getattr(self, name).setflags(write=False)


def _calibration_session_values(
    config: Figure3Config, seed: int, n_steps: int
) -> dict[str, NDArray[np.floating]]:
    """Per-event diagnostic values of one matched-null calibration session."""
    result = run_matched_null_simulation(config, seed=seed, n_time_steps=n_steps)
    d = result.diagnostics
    values = {
        key: np.asarray(getattr(d, "event_" + key), dtype=float) for key, _ in SUMMARY_FLAG_METRICS
    }
    for key, ev in values.items():
        if not np.all(np.isfinite(ev)):
            raise ValueError(
                f"Calibration event_{key} contains a non-finite value in session seed {seed}; "
                "pooled thresholds would be undefined."
            )
    return values


def estimate_calibration_thresholds(
    config: Figure3Config,
    *,
    n_calibration_realizations: int = 100,
    first_calibration_seed: int | None = None,
    calibration_steps_per_session: int | None = None,
    n_jobs: int = 1,
) -> Figure3Calibration:
    """Set the flag thresholds from independent matched-null sessions.

    Runs ``n_calibration_realizations`` sessions of
    :func:`~figure03_simulation.run_matched_null_simulation` on seeds
    ``first_calibration_seed, +1, ...`` (default: ``config.random_seed +
    DEFAULT_CALIBRATION_SEED_OFFSET``, disjoint from the evaluated
    realizations), pools their per-event diagnostics, and derives the HPD and
    KL thresholds as pooled quantiles. It also records the realized null flag
    rates at those thresholds, the HPD tie fraction, and the rank p-value's
    empirical tail.

    Parameters
    ----------
    config : Figure3Config
        Simulation configuration (trajectory model, rates, HPD coverage).
    n_calibration_realizations : int, default 100
        Number of independent null sessions.
    first_calibration_seed : int, optional
        First calibration seed.
    calibration_steps_per_session : int, optional
        Session length; defaults to the opening-baseline length.
    n_jobs : int, default 1
        Parallel workers (joblib); each session is independent.
    """
    if n_calibration_realizations < 1:
        raise ValueError("n_calibration_realizations must be >= 1")
    first = (
        config.random_seed + DEFAULT_CALIBRATION_SEED_OFFSET
        if first_calibration_seed is None
        else first_calibration_seed
    )
    n_steps = (
        config.phase_boundaries[PhaseBoundary.REMAP_START]
        if calibration_steps_per_session is None
        else int(calibration_steps_per_session)
    )
    seeds = [first + k for k in range(n_calibration_realizations)]
    sessions: list[dict[str, NDArray[np.floating]]] = Parallel(n_jobs=n_jobs)(
        delayed(_calibration_session_values)(config, seed, n_steps) for seed in seeds
    )
    pooled = {
        key: np.concatenate([session[key] for session in sessions])
        for key, _ in SUMMARY_FLAG_METRICS
    }
    thresholds = compute_baseline_diagnostic_thresholds(
        pooled, baseline_end_index=pooled["hpd_overlap"].shape[0]
    )
    pooled_rates = np.array(
        [
            _flag_percentage(pooled[key], float(getattr(thresholds, key)), direction)
            for key, direction in SUMMARY_FLAG_METRICS
        ]
    )
    per_realization = np.array(
        [
            [
                _flag_percentage(session[key], float(getattr(thresholds, key)), direction)
                for key, direction in SUMMARY_FLAG_METRICS
            ]
            for session in sessions
        ]
    )
    tie_percent = 100.0 * float(np.mean(pooled["hpd_overlap"] == thresholds.hpd_overlap))
    rank_tails = np.array(
        [100.0 * float(np.mean(pooled["predictive_pvalue"] <= a)) for a in RANK_CALIBRATION_ALPHAS]
    )
    return Figure3Calibration(
        diagnostic_thresholds=thresholds,
        n_calibration_realizations=n_calibration_realizations,
        first_calibration_seed=first,
        calibration_steps_per_session=n_steps,
        n_pooled_events=int(pooled["hpd_overlap"].shape[0]),
        pooled_null_flag_percentages=pooled_rates,
        per_realization_null_flag_percentages=per_realization,
        hpd_threshold_tie_percent=tie_percent,
        rank_pvalue_tail_percentages=rank_tails,
    )


@dataclass(frozen=True)
class Figure3RealizationSummary:
    """Figure-3 calibrated thresholds, per-condition flags, accuracy, and distributions.

    Aggregates ``n_realizations`` independent phased realizations, each scored
    against thresholds calibrated on *separate* matched-null sessions
    (:class:`Figure3Calibration`). Medians are reported because the remap
    column is strongly trajectory-dependent and skewed across realizations;
    the full per-realization arrays are retained so paired comparisons and
    spread can be displayed rather than inferred from medians.

    Parameters
    ----------
    calibration : Figure3Calibration
        Independent calibration used to set ``diagnostic_thresholds``.
    median_flag_percentages : np.ndarray, shape (3, n_columns)
        Median percent of spike events flagged. Rows follow
        :data:`SUMMARY_FLAG_METRICS`; columns follow
        :func:`build_summary_conditions`.
    flag_percentages_by_realization : np.ndarray, shape (n_realizations, 3, n_columns)
        Per-realization percentages behind the medians.
    event_counts_by_realization : np.ndarray, shape (n_realizations, n_columns)
        Number of spike events in each column of each realization (the
        per-realization denominators).
    pooled_flag_percentages : np.ndarray, shape (3, n_columns)
        Percent flagged when every realization's events are pooled (an
        event-weighted summary, distinct from the per-realization median).
    median_decoding_accuracy : np.ndarray, shape (4, n_columns)
        Median across realizations of :func:`compute_condition_decoding_accuracy`.
        Rows follow :data:`SUMMARY_ACCURACY_METRICS`.
    decoding_accuracy_by_realization : np.ndarray, shape (n_realizations, 4, n_columns)
        Per-realization accuracy behind the medians.
    flag_percentage_standard_errors, decoding_accuracy_standard_errors : np.ndarray
        Approximate standard errors of the medians (:func:`median_standard_error`).
    replay_represented_accuracy : np.ndarray, shape (n_realizations, 2)
        Replay-window accuracy against the *represented* (swept) trajectory:
        median absolute error and HPD coverage percent per realization.
    phase_spike_rates_by_realization : np.ndarray, shape (n_realizations, n_columns)
        Ordinary-ensemble spikes per second in each column, so the residual
        history-phase rate difference can be read against the matched phases.
    sparse_cell_flag_percentages_by_realization : np.ndarray, shape (n_realizations, 3)
        In the sparse-population column, percent flagged among sparse-cell
        events only (the all-event denominator is the heatmap column).
    sparse_cell_event_counts_by_realization : np.ndarray, shape (n_realizations,)
        Number of sparse-cell events per realization.
    matched_null_rank_pvalue_tail_percentages : np.ndarray, shape (len(RANK_CALIBRATION_ALPHAS),)
        Percent of pooled matched-null-column events with rank p-value
        ``<= alpha`` (out-of-sample super-uniformity check).
    n_realizations : int
        Number of realizations aggregated.
    """

    calibration: Figure3Calibration
    median_flag_percentages: NDArray[np.floating]
    flag_percentages_by_realization: NDArray[np.floating]
    event_counts_by_realization: NDArray[np.integer]
    pooled_flag_percentages: NDArray[np.floating]
    median_decoding_accuracy: NDArray[np.floating]
    decoding_accuracy_by_realization: NDArray[np.floating]
    flag_percentage_standard_errors: NDArray[np.floating]
    decoding_accuracy_standard_errors: NDArray[np.floating]
    replay_represented_accuracy: NDArray[np.floating]
    phase_spike_rates_by_realization: NDArray[np.floating]
    sparse_cell_flag_percentages_by_realization: NDArray[np.floating]
    sparse_cell_event_counts_by_realization: NDArray[np.integer]
    matched_null_rank_pvalue_tail_percentages: NDArray[np.floating]
    n_realizations: int

    @property
    def diagnostic_thresholds(self) -> DiagnosticThresholds:
        """The calibrated flag thresholds every realization was scored against."""
        return self.calibration.diagnostic_thresholds

    def __post_init__(self) -> None:
        n = self.n_realizations
        if n < 1:
            raise ValueError(f"n_realizations must be >= 1; got {n}")
        if self.median_flag_percentages.ndim != 2:
            raise ValueError(
                "Figure3RealizationSummary.median_flag_percentages must be 2-D "
                f"(n_metrics, n_columns); got shape {self.median_flag_percentages.shape}"
            )
        n_metrics, n_columns = self.median_flag_percentages.shape
        expected_shapes = {
            "flag_percentages_by_realization": (n, n_metrics, n_columns),
            "event_counts_by_realization": (n, n_columns),
            "pooled_flag_percentages": (n_metrics, n_columns),
            "median_decoding_accuracy": (len(SUMMARY_ACCURACY_METRICS), n_columns),
            "decoding_accuracy_by_realization": (n, len(SUMMARY_ACCURACY_METRICS), n_columns),
            "flag_percentage_standard_errors": (n_metrics, n_columns),
            "decoding_accuracy_standard_errors": (len(SUMMARY_ACCURACY_METRICS), n_columns),
            "replay_represented_accuracy": (n, 2),
            "phase_spike_rates_by_realization": (n, n_columns),
            "sparse_cell_flag_percentages_by_realization": (n, n_metrics),
            "sparse_cell_event_counts_by_realization": (n,),
            "matched_null_rank_pvalue_tail_percentages": (len(RANK_CALIBRATION_ALPHAS),),
        }
        for name, shape in expected_shapes.items():
            arr = getattr(self, name)
            if arr.shape != shape:
                raise ValueError(
                    f"Figure3RealizationSummary.{name} must have shape {shape}; got {arr.shape}"
                )
            arr.setflags(write=False)
        self.median_flag_percentages.setflags(write=False)


def baseline_threshold_provenance(config: Figure3Config) -> dict[str, object]:
    """Describe the rule that produced the Figure-3 flag thresholds.

    :func:`estimate_calibration_thresholds` reports threshold *values*; the
    manuscript quotes the rule behind them (1st / 99th percentile of the
    pooled independent matched-null calibration events, plus the fixed
    predictive-p-value cutoff). This returns that rule in machine-readable
    form, reading the same constants the estimate uses, so the two cannot
    drift.

    Parameters
    ----------
    config : Figure3Config
        Configuration whose phase ladder defines the default calibration
        session length (the opening-baseline length).

    Returns
    -------
    dict[str, object]
        ``calibration_steps_per_session`` plus one entry per flag metric.

    Examples
    --------
    >>> baseline_threshold_provenance(Figure3Config())["calibration_steps_per_session"]
    6000
    """
    return {
        "calibration_steps_per_session": int(config.phase_boundaries[PhaseBoundary.REMAP_START]),
        "calibration_sample": "independent_matched_null_sessions",
        "hpd_overlap": {
            "rule": "pooled_calibration_quantile",
            "quantile": BASELINE_HPD_OVERLAP_QUANTILE,
        },
        "kl_divergence": {
            "rule": "pooled_calibration_quantile",
            "quantile": BASELINE_KL_DIVERGENCE_QUANTILE,
        },
        "predictive_pvalue": {
            "rule": "fixed_cutoff",
            "cutoff": FIXED_PREDICTIVE_PVALUE_CUTOFF,
        },
    }


@dataclass(frozen=True)
class _RealizationRecord:
    """Compact per-realization summary retained from one phased simulation."""

    condition_values: list[list[NDArray[np.floating]]]
    accuracy: NDArray[np.floating]
    replay_represented: NDArray[np.floating]
    phase_spike_rates: NDArray[np.floating]
    sparse_cell_values: list[NDArray[np.floating]]
    sparse_cell_event_count: int
    matched_null_pvalues: NDArray[np.floating]


def _summarize_realization(config: Figure3Config, seed: int) -> _RealizationRecord:
    """Run one phased realization and reduce it to the summary's per-realization record."""
    sim = run_figure03_simulation(config, seed=seed)
    d = sim.diagnostics
    conditions = build_summary_conditions(config)
    condition_values = extract_condition_flag_values(d, conditions)
    accuracy = compute_condition_decoding_accuracy(
        d.posterior,
        d.predictive,
        sim.position_bins,
        sim.true_position,
        conditions,
        hpd_coverage=config.hpd_coverage,
    )
    # Replay accuracy against the represented (swept) trajectory.
    replay = next(c for c in conditions if c.condition_id == "replay")
    replay_accuracy = compute_condition_decoding_accuracy(
        d.posterior,
        d.predictive,
        sim.position_bins,
        sim.represented_position,
        [replay],
        hpd_coverage=config.hpd_coverage,
    )
    replay_represented = np.array([replay_accuracy[0, 0], replay_accuracy[1, 0]])
    # Ordinary-ensemble spikes per second per column (1 ms steps).
    n_ordinary = int(
        len(config.place_field_centers) if config.place_field_centers is not None else 0
    )
    ordinary_counts = sim.spike_counts[:, :n_ordinary].sum(axis=1)
    phase_rates = np.zeros(len(conditions))
    for j, col in enumerate(conditions):
        steps = sum(t1 - t0 for t0, t1 in col.step_windows)
        total = sum(int(ordinary_counts[t0:t1].sum()) for t0, t1 in col.step_windows)
        phase_rates[j] = 1000.0 * total / steps
    # Sparse-cell denominator inside the sparse-population column.
    sparse = next(c for c in conditions if c.condition_id == "sparse_population")
    event_time = np.asarray(d.event_time_ind)
    sparse_mask = _condition_event_mask(event_time, sparse) & (
        np.asarray(d.event_cell_ind) >= n_ordinary
    )
    sparse_values = [
        np.asarray(getattr(d, "event_" + key), dtype=float)[sparse_mask]
        for key, _ in SUMMARY_FLAG_METRICS
    ]
    null = next(c for c in conditions if c.condition_id == "matched_null")
    null_mask = _condition_event_mask(event_time, null)
    return _RealizationRecord(
        condition_values=condition_values,
        accuracy=accuracy,
        replay_represented=replay_represented,
        phase_spike_rates=phase_rates,
        sparse_cell_values=sparse_values,
        sparse_cell_event_count=int(sparse_mask.sum()),
        matched_null_pvalues=np.asarray(d.event_predictive_pvalue, dtype=float)[null_mask],
    )


def estimate_realization_summary(
    config: Figure3Config,
    *,
    calibration: Figure3Calibration,
    n_realizations: int = 100,
    first_random_seed: int | None = None,
    n_jobs: int = 1,
) -> Figure3RealizationSummary:
    """Score many phased realizations against independently calibrated thresholds.

    Runs ``n_realizations`` independent realizations of the figure-3
    simulation (seeds ``first_random_seed, first_random_seed + 1, ...``),
    scores every realization's per-condition flag percentages and decoding
    accuracy against ``calibration.diagnostic_thresholds``, and returns the
    across-realization medians together with the per-realization arrays.
    Each realization is reduced to a compact record as soon as it finishes,
    so memory stays bounded even at large ``n_realizations``.

    Parameters
    ----------
    config : Figure3Config
        Simulation configuration. ``config.place_field_centers`` must be set
        (the dataclass initializes it by default).
    calibration : Figure3Calibration
        Thresholds from :func:`estimate_calibration_thresholds`, computed on
        seeds disjoint from the evaluated realizations.
    n_realizations : int, default 100
        Number of independent realizations to aggregate. Must be >= 1.
    first_random_seed : int, optional
        First seed; subsequent realizations use consecutive seeds. If
        ``None``, uses ``config.random_seed`` so the canonical displayed run
        (seed ``config.random_seed``) is one of the aggregated realizations.
    n_jobs : int, default 1
        Parallel workers (joblib); realizations are independent.

    Raises
    ------
    ValueError
        If ``n_realizations < 1`` or the calibration seeds overlap the
        evaluation seeds.
    """
    if n_realizations < 1:
        raise ValueError(f"n_realizations must be >= 1; got {n_realizations}")
    base = config.random_seed if first_random_seed is None else first_random_seed
    evaluation_seeds = range(base, base + n_realizations)
    calibration_seeds = range(
        calibration.first_calibration_seed,
        calibration.first_calibration_seed + calibration.n_calibration_realizations,
    )
    if set(evaluation_seeds) & set(calibration_seeds):
        raise ValueError(
            "Calibration seeds overlap the evaluated realization seeds; the calibration "
            "sample must be independent of the evaluated data."
        )
    thresholds = calibration.diagnostic_thresholds
    records: list[_RealizationRecord] = Parallel(n_jobs=n_jobs)(
        delayed(_summarize_realization)(config, seed) for seed in evaluation_seeds
    )

    # A realization with no events in a column (possible in the sparse
    # regime) contributes NaN there and is excluded from that column's median.
    frac = np.stack(
        [
            flag_percentages_from_values(r.condition_values, thresholds, empty_value=np.nan)
            for r in records
        ],
        axis=0,
    )
    n_columns = frac.shape[2]
    event_counts = np.array(
        [[r.condition_values[0][j].size for j in range(n_columns)] for r in records], dtype=int
    )
    pooled_values = [
        [np.concatenate([r.condition_values[i][j] for r in records]) for j in range(n_columns)]
        for i in range(len(SUMMARY_FLAG_METRICS))
    ]
    pooled = flag_percentages_from_values(pooled_values, thresholds)
    accuracy = np.stack([r.accuracy for r in records], axis=0)
    sparse_percentages = np.array(
        [
            [
                _flag_percentage(
                    r.sparse_cell_values[i], float(getattr(thresholds, key)), direction
                )
                if r.sparse_cell_values[i].size
                else np.nan
                for i, (key, direction) in enumerate(SUMMARY_FLAG_METRICS)
            ]
            for r in records
        ]
    )
    null_pvalues = np.concatenate([r.matched_null_pvalues for r in records])
    null_tails = np.array(
        [100.0 * float(np.mean(null_pvalues <= a)) for a in RANK_CALIBRATION_ALPHAS]
    )
    return Figure3RealizationSummary(
        calibration=calibration,
        median_flag_percentages=np.nanmedian(frac, axis=0),
        flag_percentages_by_realization=frac,
        event_counts_by_realization=event_counts,
        pooled_flag_percentages=pooled,
        median_decoding_accuracy=np.median(accuracy, axis=0),
        decoding_accuracy_by_realization=accuracy,
        flag_percentage_standard_errors=median_standard_error(frac),
        decoding_accuracy_standard_errors=median_standard_error(accuracy),
        replay_represented_accuracy=np.stack([r.replay_represented for r in records], axis=0),
        phase_spike_rates_by_realization=np.stack([r.phase_spike_rates for r in records], axis=0),
        sparse_cell_flag_percentages_by_realization=sparse_percentages,
        sparse_cell_event_counts_by_realization=np.array(
            [r.sparse_cell_event_count for r in records], dtype=int
        ),
        matched_null_rank_pvalue_tail_percentages=null_tails,
        n_realizations=n_realizations,
    )
