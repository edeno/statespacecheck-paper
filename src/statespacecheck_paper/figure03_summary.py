"""Figure-3b summary heatmap: per-condition diagnostic-flag percentages.

This module builds the Figure-3b summary: it groups spike-event diagnostics into
the experimental *conditions* (well-specified, remap, history-dependent, replay,
drift, sparse population), computes the percentage of spike events each metric
flags as poor fit in each condition, and pools many independent realizations
into stabilized thresholds and median per-condition flag percentages
(:class:`Figure3RealizationSummary`). Percentages are on a 0-100 scale.

It imports :mod:`figure03_protocol` (the config + replay window),
:mod:`figure03_simulation` (``estimate_realization_summary`` runs the
simulation), and :mod:`diagnostics` (the thresholds/diagnostic containers).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal, cast

import numpy as np
from numpy.typing import NDArray

from statespacecheck_paper.diagnostics import (
    DecodingDiagnostics,
    DiagnosticThresholds,
    compute_baseline_diagnostic_thresholds,
)
from statespacecheck_paper.figure03_protocol import (
    Figure3Config,
    PhaseBoundary,
    compute_replay_step_window,
)
from statespacecheck_paper.figure03_simulation import run_figure03_simulation

SUMMARY_FLAG_METRICS: tuple[tuple[str, Literal["below", "above"]], ...] = (
    ("hpd_overlap", "below"),
    ("predictive_pvalue", "below"),
    ("kl_divergence", "above"),
)


@dataclass(frozen=True)
class Figure3SummaryCondition:
    """One column of the Figure-3b summary heatmap.

    Parameters
    ----------
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

    label: str
    step_windows: tuple[tuple[int, int], ...]
    model_component: str


def build_summary_conditions(config: Figure3Config) -> list[Figure3SummaryCondition]:
    """Phase columns for the Figure-3b summary heatmap.

    Single source of truth for the heatmap's columns, shared by the
    single-run flag-percentage helper (:func:`compute_condition_flag_percentages`)
    and the multi-realization averaging
    path (:func:`statespacecheck_paper.figure03_summary.estimate_realization_summary`)
    so the column order, time windows, and component labels cannot drift
    out of sync. ``compose_figure03`` renders from precomputed
    ``median_flag_percentages`` rather than calling either helper directly.

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
    # neither define the baseline diagnostic_thresholds nor dilute the false-positive
    # rate.
    r0, r1 = compute_replay_step_window(config)
    return [
        Figure3SummaryCondition(
            "Well-\nspecified",
            (
                (t_remap_end, t_recovery1_end),
                (t_hist_dep_end, r0),
                (r1, t_recovery2_end),
                (t_drift_end, t_recovery3_end),
            ),
            "—",
        ),
        Figure3SummaryCondition("Remap", ((t_remap_start, t_remap_end),), "Observation"),
        Figure3SummaryCondition(
            "History-\ndep.", ((t_recovery1_end, t_hist_dep_end),), "Observation"
        ),
        Figure3SummaryCondition("Replay", ((r0, r1),), "—"),
        Figure3SummaryCondition("Drift", ((t_recovery2_end, t_drift_end),), "Transition"),
        Figure3SummaryCondition(
            "Sparse\npopulation",
            ((t_recovery3_end, t_sparse_pop_end),),
            "—",
        ),
    ]


def _flag_percentage(values: NDArray[np.floating], threshold: float, direction: str) -> float:
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

    Returns
    -------
    float
        Percent (0–100) of values flagged.
    """
    if values.size == 0:
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
    events" the figure reports. (The dense ``(n_time, n_cells)`` matrices
    would collapse a multi-spike bin to a single value.)

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

    Splitting this out from :func:`compute_condition_flag_percentages` lets the
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


@dataclass(frozen=True)
class Figure3RealizationSummary:
    """Stabilized Figure-3 diagnostic_thresholds and per-phase flag fractions.

    Aggregates ``n_realizations`` independent realizations of the figure-3
    simulation so the Figure-3b heatmap and its flag diagnostic_thresholds no longer
    depend on a single noisy run (a single run's KL 99th-percentile
    threshold varies ~17% across seeds).

    - ``diagnostic_thresholds`` are computed from the per-spike baseline diagnostics
      pooled across all realizations — a far more stable estimate of the
      baseline interval than one run's quantile.
    - ``median_flag_percentages`` is the median, across realizations, of the percent of
      spike events flagged in each phase column by each metric (each
      realization scored against the shared pooled-baseline ``diagnostic_thresholds``).
      The median is used in place of the mean because the remapping column
      is strongly trajectory-dependent and skewed across realizations.

    Parameters
    ----------
    diagnostic_thresholds : DiagnosticThresholds
        Pooled-baseline flag diagnostic_thresholds.
    median_flag_percentages : np.ndarray, shape (3, n_columns)
        Median percent flagged. Rows follow
        :data:`statespacecheck_paper.figure03_summary.SUMMARY_FLAG_METRICS`;
        columns follow
        :func:`statespacecheck_paper.figure03_summary.build_summary_conditions`.
    n_realizations : int
        Number of realizations aggregated.

    Raises
    ------
    ValueError
        If ``median_flag_percentages`` is not 2-D, or ``n_realizations`` is not positive.
    """

    diagnostic_thresholds: DiagnosticThresholds
    median_flag_percentages: NDArray[np.floating]
    n_realizations: int

    def __post_init__(self) -> None:
        if self.n_realizations < 1:
            raise ValueError(f"n_realizations must be >= 1; got {self.n_realizations}")
        if self.median_flag_percentages.ndim != 2:
            raise ValueError(
                f"Figure3RealizationSummary.median_flag_percentages must be 2-D "
                f"(n_metrics, n_columns); "
                f"got shape {self.median_flag_percentages.shape}"
            )
        self.median_flag_percentages.setflags(write=False)


def estimate_realization_summary(
    config: Figure3Config,
    *,
    n_realizations: int = 100,
    first_random_seed: int | None = None,
) -> Figure3RealizationSummary:
    """Pool many realizations into stable Figure-3 diagnostic_thresholds and fractions.

    Runs ``n_realizations`` independent realizations of the figure-3
    simulation (seeds ``first_random_seed, first_random_seed + 1, ...``), pools their
    per-spike *baseline-window* diagnostics to compute the flag
    diagnostic_thresholds, then scores every realization's per-phase flag fractions
    against those shared diagnostic_thresholds and returns the across-realization
    median. A single pass holds only the finite per-spike values (not the
    dense ``DecodingDiagnostics``) per realization, so memory stays bounded even at
    large ``n_realizations``.

    Parameters
    ----------
    config : Figure3Config
        Simulation configuration. ``config.place_field_centers`` must be set
        (the dataclass initializes it by default).
    n_realizations : int, default 100
        Number of independent realizations to aggregate. Must be >= 1.
    first_random_seed : int, optional
        First seed; subsequent realizations use consecutive seeds. If
        ``None``, uses ``config.random_seed`` so the canonical displayed run
        (seed ``config.random_seed``) is one of the aggregated realizations.

    Returns
    -------
    Figure3RealizationSummary
        Pooled diagnostic_thresholds and median per-phase flag fractions.

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

    # ``compute_baseline_diagnostic_thresholds`` reads only hpd_overlap and kl_divergence (the
    # predictive_pvalue threshold is the fixed 0.05 cutoff), but pool all three so
    # the dict is a faithful baseline sample if that ever changes. Pool the
    # per-*event* baseline values (one per spike event), matching the
    # event-based phase fractions from ``extract_condition_flag_values``.
    baseline_keys = ("hpd_overlap", "kl_divergence", "predictive_pvalue")
    baseline_values: dict[str, list[NDArray[np.floating]]] = {key: [] for key in baseline_keys}
    per_realization_values: list[list[list[NDArray[np.floating]]]] = []

    for offset in range(n_realizations):
        sim = run_figure03_simulation(config, seed=base + offset)
        diagnostics = sim.diagnostics
        base_mask = np.asarray(diagnostics.event_time_ind) < baseline_end
        for key in baseline_keys:
            ev = np.asarray(getattr(diagnostics, "event_" + key), dtype=float)[base_mask]
            if not np.all(np.isfinite(ev)):
                raise ValueError(
                    f"Baseline event_{key} contains a non-finite value in realization "
                    f"seed {base + offset}; pooled thresholds would be undefined."
                )
            baseline_values[key].append(ev)
        per_realization_values.append(extract_condition_flag_values(diagnostics, conditions))

    pooled_baseline = {key: np.concatenate(vals) for key, vals in baseline_values.items()}
    diagnostic_thresholds = compute_baseline_diagnostic_thresholds(
        pooled_baseline, baseline_end_index=pooled_baseline["hpd_overlap"].shape[0]
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
        median_flag_percentages=np.median(frac, axis=0),
        n_realizations=n_realizations,
    )
