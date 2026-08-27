"""Figure-3 decode configuration, place-field remapping, and summary-flag logic.

The general Bayesian decoder and its per-window override mechanism now live in
:mod:`statespacecheck_paper.decoding`; the shared goodness-of-fit diagnostics
live in :mod:`statespacecheck_paper.diagnostics`. This module retains the
Figure-3 decode configuration and the Figure-3b summary-heatmap flag logic.

**Key Components**:
- **DecodeParams**: Parameter container for the figure-3 decoding simulation
- **get_remapped_pf_centers**: Apply place field center remapping
- **summary_phase_windows / compute_phase_flag_fractions**: Figure-3b summary flags
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import IntEnum
from typing import cast

import numpy as np
from numpy.typing import NDArray

from statespacecheck_paper.diagnostics import DecodingDiagnostics, DiagnosticThresholds

# -----------------------------
# Data containers
# -----------------------------


class PhaseBoundary(IntEnum):
    """Indices into :attr:`DecodeParams.phase_boundaries`.

    Each member is the position of one figure-3 phase transition in
    the 8-tuple. Use as ``params.phase_boundaries[PhaseBoundary.REMAP_END]``
    rather than indexing by literal integer, so a phase-ladder
    reshuffle stays compile-time-checkable.
    """

    REMAP_START = 0  # end of clean baseline
    REMAP_END = 1  # end of remap misfit
    RECOVERY1_END = 2  # end of clean recovery 1
    HIST_DEP_END = 3  # end of history-dependent firing misfit
    RECOVERY2_END = 4  # end of clean recovery 2
    DRIFT_END = 5  # end of drift misfit
    RECOVERY3_END = 6  # end of clean recovery 3
    SPARSE_POP_END = 7  # end of sparse-population control


# Default phase ladder in 1-ms steps. Used as the default of
# ``DecodeParams.phase_boundaries`` and re-exported here so tests and
# scripts that want to override a subset don't have to re-list the
# unchanged entries.
_DEFAULT_PHASE_BOUNDARIES: tuple[int, ...] = (
    6_000,
    10_000,
    14_000,
    18_000,
    22_000,
    26_000,
    30_000,
    32_000,
)


@dataclass
class DecodeParams:
    """Parameters for the figure-3 decoding simulation.

    The simulation walks through three misfit conditions and one sparse-
    activity control, separated by clean-recovery windows. Time steps are
    1 ms by convention — the simulation math itself is dt-agnostic, but the
    default parameters
    (`rate_scale=5.0`, refractory and burst windows in
    ``simulate_spikes_history_dependent``) are tuned for that mapping
    and yield hippocampally-realistic spike rates and timescales.

    **Timeline Structure** (default; all indices in 1-ms steps):

    - 0–6k: Clean baseline
    - 6k–10k: Remap misfit (4 s)
    - 10k–14k: Clean recovery
    - 14k–18k: History-dependent firing misfit (4 s)
    - 18k–22k: Clean recovery
    - 22k–26k: Drift misfit (4 s)
    - 26k–30k: Clean recovery
    - 30k–32k: Sparse-population control (2 s)

    Parameters
    ----------
    phase_boundaries : tuple of int, default ``_DEFAULT_PHASE_BOUNDARIES``
        Strictly increasing end-of-phase indices, one per member of
        :class:`PhaseBoundary`. Read via the enum
        (``params.phase_boundaries[PhaseBoundary.REMAP_END]``) rather
        than by literal integer. Override a subset by spelling out the
        whole tuple — partial overrides aren't supported because the
        invariant the dataclass enforces ("strictly increasing ladder")
        only makes sense over the full ladder.
    sigx_pred : float, default 0.5
        Decoder's baseline dynamics standard deviation.
    drift_momentum : float, default 0.8
        AR(1) coefficient on the animal's velocity during the drift
        misfit phase. The true trajectory is
        ``x[t] = x[t-1] + v[t]`` with
        ``v[t] = drift_momentum * v[t-1] + N(0, sigx_pred)``. The
        decoder assumes ``x[t] = x[t-1] + N(0, sigx_pred)`` (no
        persistent velocity).
    xs_min, xs_max, xs_step : int
        Position grid bounds and step.
    pf_width : float, default 10.0
        Gaussian place-field std (in position units).
    pf_centers : NDArray[np.floating] | None
        Place-field center positions; defaults to ``np.arange(0, 101, 10)``.
    rate_scale : float, default 5.0
        Scale factor multiplying the normalized Gaussian place-field density.
        With the default field width and a 1-ms step, a value of 5.0 gives a
        peak rate of approximately 200 Hz.
    base_seed : int, default 1
        Random seed for reproducibility.
    remap_from_to : tuple of (int, int) pairs, default see source
        Specification of which cells get remapped during the remap
        window. By default, all eleven cells participate in one fixed
        permutation that moves every field by at least three center spacings.
    sparse_position : float, default 30.0
        Fixed location where the sparse population is active in the final
        control phase.
    sparse_approach_steps : int, default 1000
        Number of steps at the end of clean recovery 3 used for a gradual
        approach to ``sparse_position``.
    sparse_ensemble_rate_scale : float, default 0.0
        Multiplicative rate applied to the eleven ordinary place cells during
        the sparse-population control. Zero represents a silent ordinary
        ensemble.
    n_sparse_cells : int, default 5
        Number of narrow cells forming the sparse population clustered at
        ``sparse_position``.
    sparse_field_spread : float, default 1.5
        Half-range (position units) over which the ``n_sparse_cells`` field
        centers are spread symmetrically about ``sparse_position``. Zero
        stacks all centers at ``sparse_position``.
    sparse_cell_width : float, default 2.0
        Standard deviation, in position units, of each narrow sparse-population
        field.
    sparse_cell_peak_rate : float, default 0.001
        Per-cell peak firing rate in spikes per 1-ms step (1 Hz). Sized so the
        population's *aggregate* rate stays ~5 Hz: with more cells firing, a
        higher aggregate rate would shorten the gaps between spikes and let the
        prediction re-concentrate, suppressing the KL response that the sparse,
        immobile regime is meant to illustrate.
    sparse_cell_baseline_gain : float, default 0.01
        Fraction of the active per-cell rate used before the final control.

    Examples
    --------
    >>> params = DecodeParams()
    >>> params.phase_boundaries[PhaseBoundary.REMAP_START]
    6000
    >>> params.phase_boundaries[PhaseBoundary.SPARSE_POP_END]
    32000
    >>> params.pf_centers
    array([  0.,  10.,  20.,  30.,  40.,  50.,  60.,  70.,  80.,  90., 100.])
    """

    # Phase ladder. One boundary per :class:`PhaseBoundary` member,
    # strictly increasing; validated in __post_init__.
    phase_boundaries: tuple[int, ...] = _DEFAULT_PHASE_BOUNDARIES

    # Decoder & dynamics parameters
    sigx_pred: float = 0.5  # baseline dynamics std
    drift_momentum: float = 0.8  # AR(1) coefficient for drift-misfit trajectory

    # Position grid
    xs_min: int = 0
    xs_max: int = 100
    xs_step: int = 1

    # Place fields
    pf_width: float = 10.0
    pf_centers: NDArray[np.floating] | None = None  # set in __post_init__
    rate_scale: float = 5.0

    base_seed: int = 1
    # Global-remapping model: a fixed random permutation (derangement) of
    # the eleven place-field centers, so each cell ``src`` adopts cell
    # ``dst``'s center. Chosen using the first NumPy default_rng seed (737)
    # whose permutation displaces every field by >=3 positions, so no cell
    # keeps a field near its original location. The scramble is spatially
    # *incoherent*: because
    # the decoder's diagnostics judge each spike against this same remapped
    # likelihood (not the true fields), the misfit surfaces only through the
    # genuine conflict between the smooth-motion prior and the scattered
    # per-spike likelihoods — a coherent remap (e.g. a pure reflection)
    # would be self-consistent and correctly go undetected.
    remap_from_to: tuple[tuple[int, int], ...] | tuple[int, int] = (
        (0, 7),
        (1, 10),
        (2, 9),
        (3, 0),
        (4, 8),
        (5, 2),
        (6, 3),
        (7, 1),
        (8, 4),
        (9, 6),
        (10, 5),
    )

    # Replay event embedded in the second clean-recovery window. The animal
    # is immobile while a coherent trajectory sweeps the track; the decoder
    # tracks the sweep, so the decoded position departs from the true
    # (fixed) position without any diagnostic flagging it -- replay is not a
    # misspecification. The sweep occupies the fractional sub-window
    # ``[replay_frac_start, replay_frac_end)`` of clean-recovery 2 and fires
    # at an elevated ``replay_rate_scale``. The trajectory makes one sweep
    # toward the farther track end, capped at ``replay_speed`` per step, and
    # returns to its starting position.
    replay_frac_start: float = 0.25
    replay_frac_end: float = 0.75
    replay_speed: float = 0.5
    replay_rate_scale: float = 20.0

    # Sparse-population control. The animal approaches a fixed location at the
    # end of clean recovery 3, then remains there (immobile) while the ordinary
    # ensemble becomes quiet. A small population of narrow, sharply tuned cells
    # clustered at that location fires sparsely, each cell an independent
    # Poisson process increasing from a small baseline gain to its full rate.
    # With little intervening population information, the predictive spreads
    # between the isolated spikes; each spike supplies a narrow likelihood
    # contained in that broad prediction. This is a correctly modeled,
    # low-activity observation regime, not a transition-model perturbation.
    sparse_position: float = 30.0
    sparse_approach_steps: int = 1_000
    sparse_ensemble_rate_scale: float = 0.0
    n_sparse_cells: int = 5
    sparse_field_spread: float = 1.5
    sparse_cell_width: float = 2.0
    sparse_cell_peak_rate: float = 0.001  # spikes/ms = 1 Hz/cell (~5 Hz aggregate)
    sparse_cell_baseline_gain: float = 0.01

    def __post_init__(self) -> None:
        """Validate the timeline and initialize ``pf_centers`` if not provided.

        ``phase_boundaries`` must have exactly one entry per
        :class:`PhaseBoundary` member and be strictly increasing —
        ``run_figure03_simulation`` builds each phase as
        ``T_next - T_prev`` and a non-monotonic timeline would yield a
        negative phase length, which ``np.arange``/``np.zeros``
        silently turn into an empty phase, shifting every later misfit
        window. Catch that here at construction rather than as a
        misaligned figure downstream.
        """
        bnds = tuple(self.phase_boundaries)
        if len(bnds) != len(PhaseBoundary):
            raise ValueError(
                f"DecodeParams.phase_boundaries must have "
                f"{len(PhaseBoundary)} entries "
                f"(one per PhaseBoundary member); got {len(bnds)}."
            )
        if any(later <= earlier for earlier, later in zip(bnds, bnds[1:], strict=False)):
            raise ValueError(
                f"DecodeParams.phase_boundaries must be strictly increasing; got {list(bnds)}."
            )
        # Coerce to tuple so the field is hashable and immutable.
        self.phase_boundaries = bnds

        if self.pf_centers is None:
            self.pf_centers = np.arange(self.xs_min, self.xs_max + 1, 10, dtype=float)
        else:
            # Copy the caller's array so we don't write-protect their
            # reference; they keep a writable original.
            self.pf_centers = np.asarray(self.pf_centers).copy()
        # Write-protect against in-place mutation. ``DecodeParams`` is a
        # plain (non-frozen) dataclass so the field can still be
        # *rebound* (``params.pf_centers = other``), but ``params.pf_centers[i] = x``
        # is now an error — the latter is the more dangerous case because
        # it silently corrupts every downstream decoder call.
        self.pf_centers.setflags(write=False)

        if not (self.xs_min <= self.sparse_position <= self.xs_max):
            raise ValueError(
                f"sparse_position must lie in [{self.xs_min}, {self.xs_max}]; "
                f"got {self.sparse_position}."
            )
        if self.sparse_approach_steps < 0:
            raise ValueError(
                f"sparse_approach_steps must be non-negative; got {self.sparse_approach_steps}."
            )
        if not (0.0 <= self.sparse_ensemble_rate_scale <= 1.0):
            raise ValueError(
                "sparse_ensemble_rate_scale must lie in [0, 1]; "
                f"got {self.sparse_ensemble_rate_scale}."
            )
        if self.n_sparse_cells < 1:
            raise ValueError(f"n_sparse_cells must be >= 1; got {self.n_sparse_cells}.")
        if not np.isfinite(self.sparse_field_spread) or self.sparse_field_spread < 0.0:
            raise ValueError(
                f"sparse_field_spread must be finite and non-negative; "
                f"got {self.sparse_field_spread}."
            )
        if not np.isfinite(self.sparse_cell_width) or self.sparse_cell_width <= 0.0:
            raise ValueError(f"sparse_cell_width must be positive; got {self.sparse_cell_width}.")
        if not np.isfinite(self.sparse_cell_peak_rate) or self.sparse_cell_peak_rate <= 0.0:
            raise ValueError(
                f"sparse_cell_peak_rate must be positive; got {self.sparse_cell_peak_rate}."
            )
        if not (0.0 <= self.sparse_cell_baseline_gain <= 1.0):
            raise ValueError(
                "sparse_cell_baseline_gain must lie in [0, 1]; "
                f"got {self.sparse_cell_baseline_gain}."
            )
        # Replay sub-window fractions must be ordered inside [0, 1]; an equal
        # or reversed pair silently empties/reverses the Replay window and
        # overlaps the well-specified baseline pool it is carved out of.
        if not (0.0 <= self.replay_frac_start < self.replay_frac_end <= 1.0):
            raise ValueError(
                "replay_frac_start/replay_frac_end must satisfy "
                "0 <= start < end <= 1; got "
                f"start={self.replay_frac_start}, end={self.replay_frac_end}."
            )
        if not (np.isfinite(self.replay_speed) and self.replay_speed > 0.0):
            raise ValueError(f"replay_speed must be positive; got {self.replay_speed}.")
        if not (np.isfinite(self.replay_rate_scale) and self.replay_rate_scale > 0.0):
            raise ValueError(f"replay_rate_scale must be positive; got {self.replay_rate_scale}.")


# -----------------------------
# Decoder components
# -----------------------------


def get_remapped_pf_centers(
    pf_centers: NDArray[np.floating],
    remap_from_to: tuple[tuple[int, int], ...] | tuple[int, int],
    active: bool,
) -> NDArray[np.floating]:
    """Get place field centers with optional remapping.

    This function creates remapped place field centers for computing likelihoods
    during model misfit periods. When active, the source cell's place field center
    is replaced with the target cell's center, so the likelihood is computed using
    the wrong place field for that cell's spikes.

    Each ``(src, dst)`` pair makes cell ``src`` use cell ``dst``'s place-field
    center; original centers are snapshotted before any writes so a pair of
    swaps (``(a, b)``, ``(b, a)``) works correctly.

    Parameters
    ----------
    pf_centers : np.ndarray, shape (n_cells,)
        Original place field centers for each cell.
    remap_from_to : tuple of tuples or tuple of ints
        Remapping specification. Can be:
        - Single remapping: (src, dst) - cell src uses cell dst's place field center
        - Multiple remappings: ((src1, dst1), (src2, dst2), ...) - apply all remappings
    active : bool
        If False, returns pf_centers unchanged. If True, applies remapping.

    Returns
    -------
    pf_centers : np.ndarray, shape (n_cells,)
        Place field centers, potentially modified if active=True.
        Returns original array if active=False, copy if active=True.

    Examples
    --------
    >>> import numpy as np
    >>> pf_centers = np.array([0.0, 10.0, 20.0, 30.0])
    >>> # Single remapping: cell 2 uses cell 0's place field
    >>> result = get_remapped_pf_centers(pf_centers, (2, 0), active=True)
    >>> result
    array([ 0., 10.,  0., 30.])

    >>> # Inactive (no remapping)
    >>> result = get_remapped_pf_centers(pf_centers, (2, 0), active=False)
    >>> np.array_equal(result, pf_centers)
    True
    """
    if not active:
        return pf_centers
    pf_centers = pf_centers.copy()

    # Normalize to iterable of pairs (handles both single and multiple remappings)
    if len(remap_from_to) == 2 and isinstance(remap_from_to[0], int):
        # Single remapping: (src, dst)
        src, dst = remap_from_to
        pf_centers[src] = pf_centers[dst]
    else:
        # Multiple remappings: ((src1, dst1), (src2, dst2), ...)
        # Note: We need the ORIGINAL centers for all targets, so copy first
        original_centers = pf_centers.copy()
        for src, dst in remap_from_to:
            pf_centers[src] = original_centers[dst]

    return pf_centers


# -----------------------------
# Figure-3 summary heatmap (per-phase flag fractions)
# -----------------------------

# Ordered metric flag specification for the Figure-3b summary heatmap.
# Each entry is ``(metric attribute name, flag direction)``. ``"below"``
# flags values at or below the threshold (worse fit for the HPD overlap
# and the predictive p-value); ``"above"`` flags values at or above it
# (worse fit for the KL divergence). The order fixes the heatmap's row
# order and is shared by the single-run renderer and the
# multi-realization averaging path so the two cannot drift apart.
SUMMARY_FLAG_METRICS: tuple[tuple[str, str], ...] = (
    ("hpd_overlap", "below"),
    ("predictive_pvalue", "below"),
    ("kl_divergence", "above"),
)


@dataclass(frozen=True)
class SummaryColumn:
    """One column of the Figure-3b summary heatmap.

    Parameters
    ----------
    label : str
        Column header (may contain a newline for a two-line label).
    slices : tuple of (int, int)
        Half-open ``[t0, t1)`` time-step windows aggregated into this
        column. The well-specified column concatenates the three
        clean-recovery windows, so this is a tuple of pairs rather than a
        single pair.
    component : str
        Model component the column's misfit perturbs (``"Observation"``,
        ``"Transition"``, or ``"—"`` for the well-specified column). Shown
        in the attribution row beneath the heatmap.
    """

    label: str
    slices: tuple[tuple[int, int], ...]
    component: str


def replay_window(params: DecodeParams) -> tuple[int, int]:
    """Global ``[start, end)`` step bounds of the replay sub-window.

    The replay event lives inside clean-recovery 2, spanning the fractional
    sub-window ``[replay_frac_start, replay_frac_end)`` of that phase.
    Deterministic from the phase ladder and the replay fractions so
    ``run_figure03_simulation`` and the figure-3b summary columns stay in
    sync.

    Parameters
    ----------
    params : DecodeParams
        Provides the phase-boundary ladder and replay fractions.

    Returns
    -------
    tuple of (int, int)
        Half-open ``[start, end)`` global step indices of the replay sweep.
    """
    bnd = params.phase_boundaries
    start = bnd[PhaseBoundary.HIST_DEP_END]
    end = bnd[PhaseBoundary.RECOVERY2_END]
    n = end - start
    r0 = start + int(round(n * params.replay_frac_start))
    r1 = start + int(round(n * params.replay_frac_end))
    # Ordered floating-point fractions can still collapse to the same integer
    # step after rounding. A genuine out-and-back trajectory needs at least a
    # start, a turn, and a return sample.
    if not (start <= r0 < r1 <= end) or r1 - r0 < 3:
        raise ValueError(
            "Replay fractions must resolve to a window of at least 3 steps "
            f"inside clean-recovery 2; got [{r0}, {r1}) within [{start}, {end})."
        )
    return r0, r1


def summary_phase_windows(params: DecodeParams) -> list[SummaryColumn]:
    """Phase columns for the Figure-3b summary heatmap.

    Single source of truth for the heatmap's columns, shared by the
    single-run renderer (:func:`compute_phase_flag_fractions`, called from
    ``plot_combined_diagnostics``) and the multi-realization averaging
    path (:func:`statespacecheck_paper.figure03_demo.estimate_stable_summary`)
    so the column order, time windows, and component labels cannot drift
    out of sync.

    The first column ("Well-specified") aggregates the clean-recovery
    windows (with the replay sub-window carved out) into an out-of-sample
    false-positive rate against the matched misfit columns. The "Replay"
    column scores the replay event, which is not a misspecification.

    Parameters
    ----------
    params : DecodeParams
        Provides the phase-boundary ladder.

    Returns
    -------
    list of SummaryColumn
        Six columns in heatmap order: well-specified, remap,
        history-dependent firing, replay, drift, sparse population. After
        the pooled reference column, the conditions follow their chronology
        in Figure 3a.
    """
    bnd = params.phase_boundaries
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
    r0, r1 = replay_window(params)
    return [
        SummaryColumn(
            "Well-\nspecified",
            (
                (t_remap_end, t_recovery1_end),
                (t_hist_dep_end, r0),
                (r1, t_recovery2_end),
                (t_drift_end, t_recovery3_end),
            ),
            "—",
        ),
        SummaryColumn("Remap", ((t_remap_start, t_remap_end),), "Observation"),
        SummaryColumn("History-\ndep.", ((t_recovery1_end, t_hist_dep_end),), "Observation"),
        SummaryColumn("Replay", ((r0, r1),), "—"),
        SummaryColumn("Drift", ((t_recovery2_end, t_drift_end),), "Transition"),
        SummaryColumn(
            "Sparse\npopulation",
            ((t_recovery3_end, t_sparse_pop_end),),
            "—",
        ),
    ]


def _flag_fraction(values: NDArray[np.floating], threshold: float, direction: str) -> float:
    """Percent of ``values`` flagged as poor fit at ``threshold``.

    Parameters
    ----------
    values : np.ndarray, shape (n,)
        Finite per-spike diagnostic values (NaNs must already be removed).
    threshold : float
        Flag threshold.
    direction : {"below", "above"}
        ``"below"`` flags ``values <= threshold``; ``"above"`` flags
        ``values >= threshold``.

    Returns
    -------
    float
        Percent (0–100) of values flagged. ``0.0`` for an empty input.
    """
    if values.size == 0:
        return 0.0
    if direction == "below":
        flagged = float(np.mean(values <= threshold))
    elif direction == "above":
        flagged = float(np.mean(values >= threshold))
    else:
        raise ValueError(f"direction must be 'below' or 'above'; got {direction!r}")
    return 100.0 * flagged


def extract_phase_flag_values(
    metrics: DecodingDiagnostics | Mapping[str, NDArray[np.floating] | NDArray[np.intp]],
    windows: list[SummaryColumn],
) -> list[list[NDArray[np.floating]]]:
    """Collect finite per-spike-event diagnostic values per metric per column.

    Works on the **per-event** arrays (``event_time_ind`` /
    ``event_hpd_overlap`` / ``event_kl_divergence`` / ``event_predictive_pvalue``),
    one value per spike event, so that a bin with several spikes from one
    cell contributes several values — matching the "percentage of spike
    events" the figure reports. (The dense ``(n_time, n_cells)`` matrices
    would collapse a multi-spike bin to a single value.)

    Parameters
    ----------
    metrics : DecodingDiagnostics or Mapping[str, NDArray]
        Source of the per-event arrays ``event_time_ind`` (int) and
        ``event_{hpd_overlap,kl_divergence,predictive_pvalue}`` (float), each of
        shape ``(n_events,)``.
    windows : list of SummaryColumn
        Heatmap columns from :func:`summary_phase_windows`.

    Returns
    -------
    list of list of np.ndarray
        Nested list indexed ``[metric_index][column_index]``; each leaf is
        a 1-D array of the non-NaN per-event values for that metric whose
        event time falls inside that column's half-open time windows. Metric
        order follows :data:`SUMMARY_FLAG_METRICS`.
    """

    def _get(name: str) -> NDArray[np.generic]:
        arr = getattr(metrics, name) if isinstance(metrics, DecodingDiagnostics) else metrics[name]
        return cast("NDArray[np.generic]", arr)

    event_time = np.asarray(_get("event_time_ind"))
    out: list[list[NDArray[np.floating]]] = []
    for metric_key, _direction in SUMMARY_FLAG_METRICS:
        ev = np.asarray(_get("event_" + metric_key), dtype=float)
        per_window: list[NDArray[np.floating]] = []
        for col in windows:
            mask = np.zeros(event_time.shape, dtype=bool)
            for t0, t1 in col.slices:
                mask |= (event_time >= t0) & (event_time < t1)
            vals = ev[mask]
            per_window.append(vals[~np.isnan(vals)])
        out.append(per_window)
    return out


def flag_fractions_from_values(
    values: list[list[NDArray[np.floating]]],
    thresholds: DiagnosticThresholds,
) -> NDArray[np.floating]:
    """Percent flagged per metric per column from pre-extracted values.

    Splitting this out from :func:`compute_phase_flag_fractions` lets the
    multi-realization averaging path
    (:func:`statespacecheck_paper.figure03_demo.estimate_stable_summary`)
    extract each realization's per-column values once and apply a
    pooled-baseline threshold afterwards, without holding a full
    ``DecodingDiagnostics`` per realization in memory.

    Parameters
    ----------
    values : list of list of np.ndarray
        Nested ``[metric_index][column_index]`` finite values, as returned
        by :func:`extract_phase_flag_values`.
    thresholds : DiagnosticThresholds
        Flag thresholds (one per metric).

    Returns
    -------
    np.ndarray, shape (3, n_columns)
        Percent (0–100) flagged. Rows follow :data:`SUMMARY_FLAG_METRICS`.
    """
    n_columns = len(values[0]) if values else 0
    frac = np.zeros((len(SUMMARY_FLAG_METRICS), n_columns))
    for i, (metric_key, direction) in enumerate(SUMMARY_FLAG_METRICS):
        threshold = float(getattr(thresholds, metric_key))
        for j in range(n_columns):
            frac[i, j] = _flag_fraction(values[i][j], threshold, direction)
    return frac


def compute_phase_flag_fractions(
    metrics: DecodingDiagnostics | Mapping[str, NDArray[np.floating]],
    thresholds: DiagnosticThresholds,
    windows: list[SummaryColumn],
) -> NDArray[np.floating]:
    """Percent of spike events flagged per metric per phase column.

    Convenience wrapper around :func:`extract_phase_flag_values` +
    :func:`flag_fractions_from_values` for the single-realization renderer.

    Parameters
    ----------
    metrics : DecodingDiagnostics or Mapping[str, NDArray]
        Diagnostic matrices for a single realization.
    thresholds : DiagnosticThresholds
        Flag thresholds (one per metric).
    windows : list of SummaryColumn
        Heatmap columns from :func:`summary_phase_windows`.

    Returns
    -------
    np.ndarray, shape (3, n_columns)
        Percent (0–100) flagged. Rows follow :data:`SUMMARY_FLAG_METRICS`;
        columns follow ``windows``.
    """
    return flag_fractions_from_values(extract_phase_flag_values(metrics, windows), thresholds)
