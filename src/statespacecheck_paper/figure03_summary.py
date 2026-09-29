"""Figure-3b summary: per-condition flag percentages and decoding errors.

This module builds the Figure-3b summary: it groups spike-event diagnostics into
the experimental *conditions* (well-specified, remap, history-dependent, replay,
drift, sparse population), computes the percentage of spike events each metric
flags as poor fit in each condition, and pools many independent realizations
into stabilized thresholds and every realization's per-condition flag
percentages and decoding errors, with their medians
(:class:`Figure3RealizationSummary`). The summary also gives approximate
standard errors of these medians. Percentages are on a 0-100 scale;
decoding errors are in position units.

It imports :mod:`figure03_protocol` (the config + replay window),
:mod:`figure03_simulation` (``estimate_realization_summary`` runs the
simulation), and :mod:`diagnostics` (the thresholds/diagnostic containers).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from statespacecheck_paper.diagnostics import (
    BASELINE_HPD_OVERLAP_QUANTILE,
    BASELINE_KL_DIVERGENCE_QUANTILE,
    FIXED_PREDICTIVE_PVALUE_CUTOFF,
    METRIC_FLAG_DIRECTIONS,
    DecodingDiagnostics,
    DiagnosticThresholds,
    FlagDirection,
    compute_baseline_diagnostic_thresholds,
    flag_mask,
)
from statespacecheck_paper.figure03_protocol import (
    Figure3Config,
    PhaseBoundary,
    compute_replay_step_window,
)
from statespacecheck_paper.figure03_simulation import run_figure03_simulation

SUMMARY_FLAG_METRICS: tuple[tuple[str, FlagDirection], ...] = tuple(METRIC_FLAG_DIRECTIONS.items())

# Row order of the per-condition decoding-accuracy block beneath the flag
# heatmap: currently the median absolute error of the filtered-posterior
# mean (position units).
SUMMARY_ACCURACY_METRICS: tuple[str, ...] = ("median_absolute_error",)

# Number of independent realizations pooled to stabilize the panel-(b)
# summary. A single run's flag thresholds and per-phase percentages are
# noisy (the KL 99th-percentile threshold varies ~17% across seeds, and
# the remap flag percentage swings with the trajectory); pooling many
# realizations gives a stable threshold and a median per-phase summary.
# The seed-1 realization shown in panel (a) is one of these.
N_REALIZATIONS = 100


@dataclass(frozen=True)
class Figure3SummaryCondition:
    """One column of the Figure-3b summary heatmap.

    Parameters
    ----------
    condition_id : str
        Stable identifier, e.g. ``"remap"``. The published summary's
        ``condition_order`` and the website key each column by it.
    label : str
        Column header (may contain a newline for a two-line label).
    step_windows : tuple of (int, int)
        Half-open ``[t0, t1)`` time-step conditions aggregated into this
        column. The well-specified column concatenates the three
        clean-recovery conditions, so this is a tuple of pairs rather than a
        single pair.
    model_component : str
        Model component the column's misfit perturbs (``"Observation"``,
        ``"Transition"``, or ``"—"`` for the well-specified column). Shown
        in the attribution row beneath the heatmap.
    """

    condition_id: str
    label: str
    step_windows: tuple[tuple[int, int], ...]
    model_component: str


def build_summary_conditions(config: Figure3Config) -> list[Figure3SummaryCondition]:
    """Phase columns for the Figure-3b summary heatmap.

    Single source of truth for the heatmap's columns, used by the
    multi-realization averaging path
    (:func:`statespacecheck_paper.figure03_summary.estimate_realization_summary`)
    so the column order, time windows, and component labels cannot drift
    out of sync. ``compose_figure03`` renders from precomputed
    ``median_flag_percentages`` rather than recomputing them.

    The first column ("Well-specified") aggregates the clean-recovery
    conditions (with the replay sub-window carved out) into an out-of-sample
    false-positive rate against the matched misfit columns. The "Replay"
    column scores the replay event, which is not a misspecification.

    Parameters
    ----------
    config : Figure3Config
        Provides the phase-boundary ladder.

    Returns
    -------
    list of Figure3SummaryCondition
        Six columns in heatmap order: well-specified, remap,
        history-dependent firing, replay, drift, sparse population. After
        the pooled reference column, the conditions follow their chronology
        in Figure 3a.
    """
    bnd = config.phase_boundaries
    t_remap_start = bnd[PhaseBoundary.REMAP_START]
    t_remap_end = bnd[PhaseBoundary.REMAP_END]
    t_recovery1_end = bnd[PhaseBoundary.RECOVERY1_END]
    t_hist_dep_end = bnd[PhaseBoundary.HIST_DEP_END]
    t_recovery2_end = bnd[PhaseBoundary.RECOVERY2_END]
    t_drift_end = bnd[PhaseBoundary.DRIFT_END]
    t_recovery3_end = bnd[PhaseBoundary.RECOVERY3_END]
    t_sparse_pop_end = bnd[PhaseBoundary.SPARSE_POP_END]
    # The replay event sits inside clean-recovery 2; carve it out of the
    # well-specified pool (it is scored in its own column) so its spikes
    # neither define the baseline thresholds nor dilute the false-positive
    # rate.
    r0, r1 = compute_replay_step_window(config)
    return [
        Figure3SummaryCondition(
            "well_specified",
            "Well-\nspecified",
            (
                (t_remap_end, t_recovery1_end),
                (t_hist_dep_end, r0),
                (r1, t_recovery2_end),
                (t_drift_end, t_recovery3_end),
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
            "sparse_population",
            "Sparse\npopulation",
            ((t_recovery3_end, t_sparse_pop_end),),
            "—",
        ),
    ]


def _flag_percentage(
    values: NDArray[np.floating], threshold: float, direction: FlagDirection
) -> float:
    """Percent of ``values`` flagged as poor fit at ``threshold``.

    Parameters
    ----------
    values : np.ndarray, shape (n,)
        Per-spike diagnostic values. Every event must carry a value.
    threshold : float
        Flag threshold.
    direction : {"below", "above"}
        Worse-fit side of ``threshold``, applied by :func:`flag_mask`.

    Returns
    -------
    float
        Percent (0–100) of values flagged.
    """
    if values.size == 0:
        raise ValueError("Cannot compute a flag percentage for a condition with no spike events")
    if np.any(np.isnan(values)) or np.any(np.isneginf(values)):
        raise ValueError("Per-event diagnostic values must not contain NaN or -inf")
    return 100.0 * float(np.mean(flag_mask(values, threshold, direction)))


def extract_condition_flag_values(
    diagnostics: DecodingDiagnostics,
    conditions: list[Figure3SummaryCondition],
) -> list[list[NDArray[np.floating]]]:
    """Collect per-spike-event diagnostic values per metric per column.

    Works on the **per-event** arrays (``event_time_ind`` /
    ``event_hpd_overlap`` / ``event_kl_divergence`` / ``event_predictive_pvalue``),
    one value per spike event, so that a bin with several spikes from one
    cell contributes several values — matching the "percentage of spike
    events" the figure reports. (The dense ``(n_time, n_cells)`` matrices
    would collapse a multi-spike bin to a single value.)

    Parameters
    ----------
    diagnostics : DecodingDiagnostics
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
    event_time = np.asarray(diagnostics.event_time_ind)
    out: list[list[NDArray[np.floating]]] = []
    for metric_key, _direction in SUMMARY_FLAG_METRICS:
        ev = np.asarray(getattr(diagnostics, "event_" + metric_key), dtype=float)
        per_window: list[NDArray[np.floating]] = []
        for col in conditions:
            mask = np.zeros(event_time.shape, dtype=bool)
            for t0, t1 in col.step_windows:
                mask |= (event_time >= t0) & (event_time < t1)
            vals = ev[mask]
            if np.any(np.isnan(vals)) or np.any(np.isneginf(vals)):
                raise ValueError(
                    f"event_{metric_key} contains an undefined value in condition {col.label!r}"
                )
            per_window.append(vals)
        out.append(per_window)
    return out


def flag_percentages_from_values(
    values: list[list[NDArray[np.floating]]],
    diagnostic_thresholds: DiagnosticThresholds,
) -> NDArray[np.floating]:
    """Percent flagged per metric per column from pre-extracted values.

    Separating this from :func:`extract_condition_flag_values` lets the
    multi-realization averaging path
    (:func:`statespacecheck_paper.figure03_summary.estimate_realization_summary`)
    extract each realization's per-column values once and apply a
    pooled-baseline threshold afterwards, without holding a full
    ``DecodingDiagnostics`` per realization in memory.

    Parameters
    ----------
    values : list of list of np.ndarray
        Nested ``[metric_index][column_index]`` finite values, as returned
        by :func:`extract_condition_flag_values`.
    diagnostic_thresholds : DiagnosticThresholds
        Flag thresholds (one per metric).

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
            frac[i, j] = _flag_percentage(values[i][j], threshold, direction)
    return frac


def compute_condition_decoding_accuracy(
    posterior: NDArray[np.floating],
    position_bins: NDArray[np.floating],
    true_position: NDArray[np.floating],
    conditions: list[Figure3SummaryCondition],
) -> NDArray[np.floating]:
    """Per-condition decoding accuracy of the filtered posterior.

    Scores the decoder's state estimate against the stored true position
    inside each summary column's time windows. Unlike the flag percentages,
    which are per spike event, these are per time step, so every step in a
    window counts whether or not a spike occurred.

    Parameters
    ----------
    posterior : np.ndarray, shape (n_time, n_bins)
        Filtered posterior ``p(x_t | y_{1:t})`` over the position grid. Rows
        need not be normalized, but each must carry positive finite mass.
    position_bins : np.ndarray, shape (n_bins,)
        Position-grid bin centres (position units).
    true_position : np.ndarray, shape (n_time,)
        True position at each time step. For the replay control this is the
        animal's fixed physical position, so the replay column measures the
        decoded-versus-physical gap by construction.
    conditions : list of Figure3SummaryCondition
        Heatmap columns from :func:`build_summary_conditions`.

    Returns
    -------
    np.ndarray, shape (1, n_columns)
        Rows follow :data:`SUMMARY_ACCURACY_METRICS`: the median absolute
        error between the posterior mean and ``true_position``, in position
        units.

    Raises
    ------
    ValueError
        On shape mismatch, non-finite input, a posterior row with
        non-positive mass, or a column with no time steps.
    """
    posterior = np.asarray(posterior, dtype=float)
    position_bins = np.asarray(position_bins, dtype=float)
    true_position = np.asarray(true_position, dtype=float)
    if posterior.ndim != 2:
        raise ValueError(f"posterior must be 2-D (n_time, n_bins); got shape {posterior.shape}")
    n_time, n_bins = posterior.shape
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
        and np.all(np.isfinite(position_bins))
        and np.all(np.isfinite(true_position))
    ):
        raise ValueError("posterior, position_bins, and true_position must all be finite")
    mass = posterior.sum(axis=1)
    if np.any(mass <= 0.0):
        raise ValueError("Every posterior row must carry positive mass")

    posterior_mean = (posterior @ position_bins) / mass
    abs_error = np.abs(posterior_mean - true_position)

    out = np.zeros((len(SUMMARY_ACCURACY_METRICS), len(conditions)))
    for j, col in enumerate(conditions):
        mask = np.zeros(n_time, dtype=bool)
        for t0, t1 in col.step_windows:
            mask[t0:t1] = True
        if not mask.any():
            raise ValueError(f"Condition {col.label!r} contains no time steps")
        out[0, j] = float(np.median(abs_error[mask]))
    return out


def median_standard_error(samples: NDArray[np.floating]) -> NDArray[np.floating]:
    """Approximate a median's standard error from an order-statistic interval.

    This is an *approximate* standard error, and it is conditional on the
    simulation setup: it describes how much the median of these
    ``n_realizations`` seeds would move under a rerun with different seeds,
    for this configuration, and nothing beyond that. It is published in the
    summary as uncertainty in the aggregated median, not as the spread of
    individual realizations; it does not control how the manuscript prints
    the value.

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
    samples : np.ndarray, shape (n_samples, ...)
        Sample values; the first axis indexes the realizations.

    Returns
    -------
    standard_error : np.ndarray
        Approximate standard error of the median, shape ``samples.shape[1:]``.
        Returns zero when the selected order statistics coincide, including
        when every sample is identical; this does not establish zero
        population uncertainty.

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
    n_samples = samples.shape[0]
    z = 1.96
    # Convert the 1-based order statistics to 0-based indices, clipped so a
    # small sample cannot index outside the array.
    lower = max(int(np.floor(n_samples / 2 - z * np.sqrt(n_samples) / 2)) - 1, 0)
    upper = min(n_samples - lower - 1, n_samples - 1)
    ordered = np.sort(samples, axis=0)
    interval = ordered[upper] - ordered[lower]
    return np.asarray(interval / (2.0 * z), dtype=float)


@dataclass(frozen=True)
class Figure3RealizationSummary:
    """Figure-3 thresholds and every realization's flags and errors, with their medians.

    Aggregates ``n_realizations`` independent realizations of the figure-3
    simulation so the Figure-3b heatmap and its flag thresholds do not
    depend on a single noisy run (a single run's KL 99th-percentile
    threshold varies ~17% across seeds).

    - ``diagnostic_thresholds`` are computed from the per-spike baseline diagnostics
      pooled across all realizations — a far more stable estimate of the
      baseline interval than one run's quantile.
    - ``realization_flag_percentages`` holds each realization's percent of spike
      events flagged in each phase column by each metric (every realization
      scored against the shared pooled-baseline ``diagnostic_thresholds``), and
      ``realization_decoding_accuracy`` each realization's per-column decoding
      accuracy from :func:`compute_condition_decoding_accuracy`. Keeping them
      shows the spread across realizations, not only its center.
    - ``median_flag_percentages`` and ``median_decoding_accuracy`` are their
      medians across realizations. The median is used in place of the mean
      because the remapping column is strongly trajectory-dependent and skewed
      across realizations.

    Parameters
    ----------
    diagnostic_thresholds : DiagnosticThresholds
        Pooled-baseline flag thresholds.
    realization_flag_percentages : np.ndarray, shape (n_realizations, 3, n_columns)
        Percent flagged in each realization, in seed order. The middle axis
        follows :data:`statespacecheck_paper.figure03_summary.SUMMARY_FLAG_METRICS`;
        columns follow
        :func:`statespacecheck_paper.figure03_summary.build_summary_conditions`.
    realization_decoding_accuracy : np.ndarray, shape (n_realizations, 1, n_columns)
        Decoding accuracy in each realization. The middle axis follows
        :data:`statespacecheck_paper.figure03_summary.SUMMARY_ACCURACY_METRICS`;
        realizations and columns match ``realization_flag_percentages``.

    Raises
    ------
    ValueError
        If ``realization_flag_percentages`` is not 3-D with at least one
        realization, or ``realization_decoding_accuracy`` is not
        ``(n_realizations, 1, n_columns)``.
    """

    diagnostic_thresholds: DiagnosticThresholds
    realization_flag_percentages: NDArray[np.floating]
    realization_decoding_accuracy: NDArray[np.floating]

    def __post_init__(self) -> None:
        flags = self.realization_flag_percentages
        if flags.ndim != 3 or flags.shape[0] < 1:
            raise ValueError(
                "Figure3RealizationSummary.realization_flag_percentages must be 3-D "
                f"(n_realizations >= 1, n_metrics, n_columns); got shape {flags.shape}"
            )
        expected = (flags.shape[0], len(SUMMARY_ACCURACY_METRICS), flags.shape[2])
        if self.realization_decoding_accuracy.shape != expected:
            raise ValueError(
                "Figure3RealizationSummary.realization_decoding_accuracy must have shape "
                f"{expected} to match realization_flag_percentages; "
                f"got shape {self.realization_decoding_accuracy.shape}"
            )
        flags.setflags(write=False)
        self.realization_decoding_accuracy.setflags(write=False)

    @property
    def n_realizations(self) -> int:
        """Number of realizations aggregated."""
        return int(self.realization_flag_percentages.shape[0])

    @property
    def median_flag_percentages(self) -> NDArray[np.floating]:
        """Median percent flagged across realizations, shape ``(3, n_columns)``."""
        return np.asarray(np.median(self.realization_flag_percentages, axis=0), dtype=float)

    @property
    def median_decoding_accuracy(self) -> NDArray[np.floating]:
        """Median decoding accuracy across realizations, shape ``(1, n_columns)``."""
        return np.asarray(np.median(self.realization_decoding_accuracy, axis=0), dtype=float)

    @property
    def flag_percentage_standard_errors(self) -> NDArray[np.floating]:
        """Approximate standard error of each median flag percentage, ``(3, n_columns)``.

        From :func:`median_standard_error`: uncertainty in the aggregated median
        under this configuration, not the spread of individual realizations
        (``realization_flag_percentages`` shows that). The manuscript's printed
        precision follows the reporting policy independently of these SEs.
        """
        return median_standard_error(self.realization_flag_percentages)

    @property
    def decoding_accuracy_standard_errors(self) -> NDArray[np.floating]:
        """Approximate standard error of each median decoding accuracy, ``(1, n_columns)``."""
        return median_standard_error(self.realization_decoding_accuracy)


def baseline_threshold_provenance(config: Figure3Config) -> dict[str, object]:
    """Describe the rule that produced the Figure-3 flag thresholds.

    :func:`estimate_realization_summary` reports threshold *values*; the
    manuscript quotes the rule behind them (1st / 99th percentile of the
    pooled opening baseline, plus the fixed predictive-p-value cutoff). This
    returns that rule in machine-readable form, reading the same constants
    and the same baseline boundary the estimate uses, so the two cannot drift.

    Parameters
    ----------
    config : Figure3Config
        Configuration whose phase ladder defines the baseline window.

    Returns
    -------
    dict[str, object]
        ``baseline_end_index`` plus one entry per flag metric.

    Examples
    --------
    >>> baseline_threshold_provenance(Figure3Config())["baseline_end_index"]
    6000
    """
    return {
        "baseline_end_index": int(config.phase_boundaries[PhaseBoundary.REMAP_START]),
        "hpd_overlap": {
            "rule": "pooled_baseline_quantile",
            "quantile": BASELINE_HPD_OVERLAP_QUANTILE,
        },
        "kl_divergence": {
            "rule": "pooled_baseline_quantile",
            "quantile": BASELINE_KL_DIVERGENCE_QUANTILE,
        },
        "predictive_pvalue": {
            "rule": "fixed_cutoff",
            "cutoff": FIXED_PREDICTIVE_PVALUE_CUTOFF,
        },
    }


def estimate_realization_summary(
    config: Figure3Config,
    *,
    n_realizations: int = N_REALIZATIONS,
    first_random_seed: int | None = None,
) -> Figure3RealizationSummary:
    """Pool many realizations into stable Figure-3 flag thresholds and fractions.

    Runs ``n_realizations`` independent realizations of the figure-3
    simulation (seeds ``first_random_seed, first_random_seed + 1, ...``), pools their
    per-spike *baseline-window* diagnostics to compute the flag
    thresholds, then scores every realization's per-phase flag fractions
    against those shared thresholds and returns them all (their
    medians are properties of the result). A single pass holds only the finite
    per-spike values (not the dense ``DecodingDiagnostics``) per realization, so
    memory stays bounded even at
    large ``n_realizations``.

    Parameters
    ----------
    config : Figure3Config
        Simulation configuration. ``config.place_field_centers`` must be set
        (the dataclass initializes it by default).
    n_realizations : int, default ``N_REALIZATIONS``
        Number of independent realizations to aggregate. Must be >= 1.
    first_random_seed : int, optional
        First seed; subsequent realizations use consecutive seeds. If
        ``None``, uses ``config.random_seed`` so the canonical displayed run
        (seed ``config.random_seed``) is one of the aggregated realizations.

    Returns
    -------
    Figure3RealizationSummary
        Pooled flag thresholds, and every realization's per-phase flag
        fractions and decoding accuracy (with their medians).

    Raises
    ------
    ValueError
        If ``n_realizations < 1``.
    """
    if n_realizations < 1:
        raise ValueError(f"n_realizations must be >= 1; got {n_realizations}")

    base = config.random_seed if first_random_seed is None else first_random_seed
    baseline_end = config.phase_boundaries[PhaseBoundary.REMAP_START]
    conditions = build_summary_conditions(config)

    # Pool the per-*event* baseline values (one per spike event), matching the
    # event-based phase fractions from ``extract_condition_flag_values``. The
    # thresholds use only hpd_overlap and kl_divergence (the predictive_pvalue
    # cutoff is fixed), but all three are checked for non-finite values.
    baseline_values: dict[str, list[NDArray[np.floating]]] = {
        key: [] for key in METRIC_FLAG_DIRECTIONS
    }
    per_realization_values: list[list[list[NDArray[np.floating]]]] = []
    per_realization_accuracy: list[NDArray[np.floating]] = []

    for offset in range(n_realizations):
        sim = run_figure03_simulation(config, seed=base + offset)
        diagnostics = sim.diagnostics
        base_mask = np.asarray(diagnostics.event_time_ind) < baseline_end
        for key in baseline_values:
            ev = np.asarray(getattr(diagnostics, "event_" + key), dtype=float)[base_mask]
            if not np.all(np.isfinite(ev)):
                raise ValueError(
                    f"Baseline event_{key} contains a non-finite value in realization "
                    f"seed {base + offset}; pooled thresholds would be undefined."
                )
            baseline_values[key].append(ev)
        per_realization_values.append(extract_condition_flag_values(diagnostics, conditions))
        per_realization_accuracy.append(
            compute_condition_decoding_accuracy(
                diagnostics.posterior, sim.position_bins, sim.true_position, conditions
            )
        )

    pooled_baseline = {key: np.concatenate(vals) for key, vals in baseline_values.items()}
    diagnostic_thresholds = compute_baseline_diagnostic_thresholds(
        hpd_overlap=pooled_baseline["hpd_overlap"],
        kl_divergence=pooled_baseline["kl_divergence"],
    )

    # (n_realizations, n_metrics, n_columns) flag-fraction stack.
    frac = np.stack(
        [
            flag_percentages_from_values(values, diagnostic_thresholds)
            for values in per_realization_values
        ],
        axis=0,
    )
    return Figure3RealizationSummary(
        diagnostic_thresholds=diagnostic_thresholds,
        realization_flag_percentages=frac,
        realization_decoding_accuracy=np.stack(per_realization_accuracy, axis=0),
    )
