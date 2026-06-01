"""Analysis functions for state space model diagnostics.

This module contains the core analysis logic for running Bayesian decoders and computing
diagnostic metrics (KL divergence, HPD overlap, spike probability) to assess model
goodness-of-fit.

**Key Components**:
- **DecodeParams**: Parameter container for decoding simulations
- **likelihood_grid_for_counts**: Compute Poisson likelihood for spike counts
- **get_remapped_pf_centers**: Apply place field center remapping
- **decode_and_diagnostics**: Main decoder with diagnostic computation
- **Thresholds**: Container for diagnostic threshold values
- **compute_thresholds**: Compute thresholds from baseline period
- **Transformed**: Container for transformed diagnostic metrics
- **transform_metrics**: Apply transformations for better visualization

**Example**:

    >>> import numpy as np
    >>> from statespacecheck_paper.analysis import DecodeParams, decode_and_diagnostics
    >>> from statespacecheck_paper.simulation import gaussian_transition_matrix
    >>> # Set up parameters
    >>> params = DecodeParams()
    >>> xs = np.arange(params.xs_min, params.xs_max + 1, params.xs_step, dtype=float)
    >>> transition_matrix = gaussian_transition_matrix(xs, params.sigx_pred)
    >>> # Simulate some spike data
    >>> spikes = np.random.poisson(0.1, size=(100, len(params.pf_centers)))
    >>> # Run decoder (clean decode, no misfit schedule)
    >>> results = decode_and_diagnostics(
    ...     spikes, xs, transition_matrix, params.pf_centers,
    ...     params.pf_width, params.rate_scale
    ... )
    >>> type(results).__name__
    'Diagnostics'
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping
from dataclasses import dataclass
from enum import IntEnum
from typing import cast

import numpy as np
import statespacecheck as ssc
from numpy.typing import NDArray
from scipy.stats import poisson

from statespacecheck_paper.simulation import (
    normalize,
    placefield_rates,
    softmax_with_shift,
)

# Spike-batch size for ``compute_per_cell_diagnostics_from_rates``.
# Caps the (n_spikes, n_bins) scratch arrays at ``_PER_SPIKE_BATCH``
# rows so full-session real-data builds (~870 K spikes) don't allocate
# multi-GB working buffers. 50 K × 512 bins × float64 ≈ 200 MB per
# scratch array, ~600 MB peak with three live (pred / rates / lik).
_PER_SPIKE_BATCH = 50_000

# Sentinel for ``_condition_on`` underflow steps.
_NEG_INF = float("-inf")


def _condition_on(
    probs: NDArray[np.floating],
    ll: NDArray[np.floating],
    eps: float = 1e-15,
) -> tuple[NDArray[np.floating], float]:
    """Bayesian update: multiply prior by emission likelihood, normalize.

    Adapted from ``non_local_detector.core._condition_on`` (which itself
    is adapted from ``dynamax``). The log-sum-exp shift of subtracting
    ``ll.max()`` before exponentiation makes ``exp(ll - ll_max)`` peak
    at 1, keeping the posterior update numerically stable even when
    individual cell likelihoods would have underflowed in linear space.

    Underflow regime: when the prior and likelihood are essentially
    disjoint (``weighted.sum() < eps``), the marginal likelihood
    underflows and the returned posterior would be a meaningless
    near-zero vector. Rather than propagate that into the next step's
    predict-and-update (where it would silently become uniform via the
    eps clamp in :func:`statespacecheck_paper.simulation.normalize`),
    this function explicitly returns a uniform posterior and
    ``log_norm = -inf``. Callers should treat ``log_norm == -inf`` as
    the signal to flag the step.

    Parameters
    ----------
    probs : np.ndarray, shape (n_bins,)
        Linear-space prior, must sum to 1.
    ll : np.ndarray, shape (n_bins,)
        Log-likelihood of the observation at each bin (unnormalized).
    eps : float, default 1e-15
        Underflow threshold on ``weighted.sum()``; below this the
        uniform fallback path runs.

    Returns
    -------
    new_probs : np.ndarray, shape (n_bins,)
        Posterior; sums to 1. Equals ``1/n_bins`` everywhere when the
        underflow fallback runs.
    log_norm : float
        Log marginal likelihood for this step (``log p(obs | past)``).
        ``-inf`` when the underflow fallback runs.
    """
    n_bins = probs.size
    ll_max = float(np.max(ll))
    if not np.isfinite(ll_max):
        return np.full(n_bins, 1.0 / n_bins), _NEG_INF
    weighted = probs * np.exp(ll - ll_max)
    norm = float(weighted.sum())
    if norm < eps:
        return np.full(n_bins, 1.0 / n_bins), _NEG_INF
    new_probs = weighted / norm
    log_norm = float(np.log(norm)) + ll_max
    return new_probs, log_norm


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
    WIDE_DYNAMICS_END = 7  # end of wide-dynamics-noise misfit


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

    The simulation walks through four misfit conditions separated by
    clean-recovery windows. Time steps are 1 ms by convention — the
    simulation math itself is dt-agnostic, but the default parameters
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
    - 30k–32k: Wide-dynamics-noise misfit (2 s)

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
    sigx_wide_dynamics : float, default 20.0
        Inflated decoder transition std for the wide-dynamics-noise
        misfit (40× baseline). Engineered to be wide enough that the
        decoder's predictive covers most of the track and the
        per-spike likelihood (narrow at the firing cell's PF) sits
        cleanly inside it.
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
        Peak Poisson rate scale (spikes/step at the cell's PF center).
        At 1 ms/step the default gives ~200 Hz peak — within the
        plausible range for hippocampal pyramidal cells.
    base_seed : int, default 1
        Random seed for reproducibility.
    remap_from_to : tuple of (int, int) pairs, default see source
        Specification of which cells get remapped during the remap
        window. Default is six bidirectional swaps across the track.

    Examples
    --------
    >>> params = DecodeParams()
    >>> params.phase_boundaries[PhaseBoundary.REMAP_START]
    6000
    >>> params.phase_boundaries[PhaseBoundary.WIDE_DYNAMICS_END]
    32000
    >>> params.pf_centers
    array([  0.,  10.,  20.,  30.,  40.,  50.,  60.,  70.,  80.,  90., 100.])
    """

    # Phase ladder. One boundary per :class:`PhaseBoundary` member,
    # strictly increasing; validated in __post_init__.
    phase_boundaries: tuple[int, ...] = _DEFAULT_PHASE_BOUNDARIES

    # Decoder & dynamics parameters
    sigx_pred: float = 0.5  # baseline dynamics std
    sigx_wide_dynamics: float = 20.0  # 40× baseline — wide-dynamics-noise phase
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
    # Six bidirectional swaps so the remap window is unambiguous: every
    # remapped cell has another cell whose PF center it adopts, and that
    # cell adopts its center. Cell 0↔9, 1↔8, 2↔7 — three swap pairs.
    remap_from_to: tuple[tuple[int, int], ...] | tuple[int, int] = (
        (0, 9),
        (1, 8),
        (2, 7),
        (9, 0),
        (8, 1),
        (7, 2),
    )

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


@dataclass(frozen=True)
class MisfitWindow:
    """One decoder-side misfit, active over the half-open interval ``[start, end)``.

    A misfit window substitutes any of three baseline quantities while the
    decoder runs inside it. Each field is optional; ``None`` means "use the
    baseline".

    Parameters
    ----------
    start, end : int
        Half-open time-step bounds ``[start, end)``. ``start < end`` is
        required.
    transition_matrix : np.ndarray, shape (n_bins, n_bins), optional
        Replaces the baseline transition matrix in the predict step
        (used by the wide-dynamics-noise misfit).
    decoder_rates : np.ndarray, shape (n_bins, n_cells), optional
        Replaces the baseline Gaussian place-field rate table used to
        form the posterior-update likelihood (and the displayed per-spike
        likelihood). Used by the remap misfit (remapped place fields).
        Diagnostics continue to judge spikes against the model's
        intended Gaussian PFs, so the mismatch surfaces.

    Raises
    ------
    ValueError
        If ``start >= end`` or if ``decoder_rates`` contains negative or
        non-finite entries.

    Notes
    -----
    Supplied ``transition_matrix`` and ``decoder_rates`` are copied at
    construction and marked write-protected via ``setflags(write=False)``,
    extending the dataclass's ``frozen=True`` invariant to the array
    contents.

    Shape parity with the decoder's grid is checked by
    :meth:`validate_against`, which the decoder calls once per
    schedule entry — too late to check at construction because the
    schedule may be built before ``xs`` is pinned down.

    Examples
    --------
    Remap-style misfit — decoder uses an alternate rate table, diagnostic
    still references the original Gaussian PFs so the mismatch surfaces:

    >>> import numpy as np
    >>> remapped = np.full((5, 3), 0.1)
    >>> w = MisfitWindow(10, 20, decoder_rates=remapped)
    >>> w.start, w.end
    (10, 20)
    """

    start: int
    end: int
    transition_matrix: NDArray[np.floating] | None = None
    decoder_rates: NDArray[np.floating] | None = None

    def __post_init__(self) -> None:
        """Validate the window bounds and any supplied rate tables.

        Makes a write-protected copy of any supplied table so the
        ``frozen=True`` invariant extends to the array contents, not
        just the dataclass field bindings.
        """
        if self.start >= self.end:
            raise ValueError(f"MisfitWindow requires start < end, got ({self.start}, {self.end})")

        # A negative or non-finite rate table would become NaN once it
        # reaches ``poisson.pmf`` and propagate silently through the
        # posterior — reject it at construction.
        if self.decoder_rates is not None and not (
            np.all(np.isfinite(self.decoder_rates)) and np.all(self.decoder_rates >= 0.0)
        ):
            raise ValueError(
                "MisfitWindow.decoder_rates must be finite and non-negative everywhere"
            )

        # Write-protect any supplied tables. A frozen dataclass only
        # prevents rebinding ``self.decoder_rates``; the underlying
        # ndarray is still mutable. Take a defensive copy and mark it
        # read-only so callers can't bypass the validation above by
        # mutating in place after construction.
        for name in ("transition_matrix", "decoder_rates"):
            table = getattr(self, name)
            if table is None:
                continue
            copy = table.copy()
            copy.setflags(write=False)
            object.__setattr__(self, name, copy)

    def validate_against(self, *, n_bins: int, n_cells: int) -> None:
        """Validate that supplied rate tables match the decoder's grid.

        Shape parity with the decoder's position grid and cell count
        can't be checked at construction time because the schedule may
        be built before the decoder's ``xs`` is pinned down. Call this
        once per schedule entry inside the decoder.

        Parameters
        ----------
        n_bins : int
            Number of position bins in the decoder's grid (``xs.size``).
        n_cells : int
            Number of cells in the spike train (``spikes.shape[1]``).

        Raises
        ------
        ValueError
            If ``decoder_rates`` shape doesn't equal ``(n_bins, n_cells)``,
            or ``transition_matrix`` shape doesn't equal ``(n_bins, n_bins)``.
        """
        if self.decoder_rates is not None and self.decoder_rates.shape != (n_bins, n_cells):
            raise ValueError(
                f"MisfitWindow.decoder_rates shape {self.decoder_rates.shape} does not "
                f"match decoder grid ({n_bins}, {n_cells})."
            )
        if self.transition_matrix is not None and self.transition_matrix.shape != (
            n_bins,
            n_bins,
        ):
            raise ValueError(
                f"MisfitWindow.transition_matrix shape "
                f"{self.transition_matrix.shape} does not match decoder grid "
                f"({n_bins}, {n_bins})."
            )


@dataclass(frozen=True)
class MisfitSchedule:
    """An ordered set of non-overlapping :class:`MisfitWindow` entries.

    Time steps not covered by any window decode with the baseline
    transition matrix and Gaussian place-field rates. The empty schedule
    (the default) is a clean decode with no misfits — used for real-data
    decoding.

    Parameters
    ----------
    windows : tuple[MisfitWindow, ...]
        The misfit windows. Must not overlap; order is not significant.

    Raises
    ------
    ValueError
        If any two windows overlap.

    Examples
    --------
    >>> MisfitSchedule().window_at(5) is None
    True
    >>> sched = MisfitSchedule((MisfitWindow(10, 20), MisfitWindow(30, 40)))
    >>> sched.window_at(15).start
    10
    >>> sched.window_at(25) is None
    True
    """

    windows: tuple[MisfitWindow, ...] = ()

    def __post_init__(self) -> None:
        """Reject overlapping windows."""
        ordered = sorted(self.windows, key=lambda w: w.start)
        for earlier, later in zip(ordered, ordered[1:], strict=False):
            if later.start < earlier.end:
                raise ValueError(
                    "MisfitSchedule windows must not overlap; "
                    f"[{earlier.start}, {earlier.end}) overlaps "
                    f"[{later.start}, {later.end})"
                )

    def window_at(self, t: int) -> MisfitWindow | None:
        """Return the window containing time step ``t``, or ``None``.

        Windows are non-overlapping (enforced at construction), so at most
        one can match.
        """
        for window in self.windows:
            if window.start <= t < window.end:
                return window
        return None


# -----------------------------
# Diagnostic returns
# -----------------------------

# Per-event metric field names — shared by ``PerCellDiagnostics`` and
# ``Diagnostics`` shape-validation loops.
_PER_EVENT_METRIC_NAMES = ("event_hpd_overlap", "event_kl_divergence", "event_spike_prob")


def _check_range(
    arr: NDArray[np.floating],
    name: str,
    *,
    lo: float,
    hi: float | None,
    atol: float = 1e-9,
) -> None:
    """Raise ``ValueError`` if any non-NaN entry of ``arr`` falls outside
    ``[lo - atol, hi + atol]``.

    NaN is treated as legitimate (the dense diagnostic matrices encode
    "no spike at this (t, cell)" as NaN). ``atol`` absorbs FP overshoot
    from the cumulative-sum spike-prob computation (where a rank can
    summed-float-error to 1.0000000000000002). Used by ``Diagnostics``
    ``__post_init__`` to catch buggy decoder output at the producer
    boundary, not deep in a summary downstream.
    """
    finite = np.isfinite(arr)
    if not np.any(finite):
        return
    valid = arr[finite]
    if np.any(valid < lo - atol):
        raise ValueError(f"{name}: values below {lo} found (min={float(valid.min())})")
    if hi is not None and np.any(valid > hi + atol):
        raise ValueError(f"{name}: values above {hi} found (max={float(valid.max())})")


@dataclass(frozen=True)
class PerCellDiagnostics:
    """Return of :func:`compute_per_cell_diagnostics_from_rates`.

    Per-spike-event arrays are always present; the four dense
    ``(n_time, n_cells)`` / ``(n_spikes, n_bins)`` arrays are
    optional, populated only when ``include_dense_matrices=True``.
    Frozen + write-protected so a downstream consumer cannot
    accidentally mutate a metric mid-pipeline.

    Parameters
    ----------
    event_time_ind, event_cell_ind : np.ndarray, shape (n_spikes,)
        Time-bin index and cell index for each spike event.
    event_hpd_overlap, event_kl_divergence, event_spike_prob : np.ndarray, shape (n_spikes,)
        Per-event diagnostic values.
    hpd_overlap, kl_divergence, spike_prob : np.ndarray, shape (n_time, n_cells), optional
        Dense scattered matrices; ``NaN`` where no spike occurred. ``None`` when
        the producer was called with ``include_dense_matrices=False``.
    per_spike_likelihood : np.ndarray, shape (n_spikes, n_bins), optional
        Per-spike normalized likelihood. ``None`` when ``include_dense_matrices=False``.

    Raises
    ------
    ValueError
        If the per-event arrays don't share leading dim ``n_spikes``, or
        the dense matrices (when present) don't share leading dim ``n_time``.
    """

    event_time_ind: NDArray[np.intp]
    event_cell_ind: NDArray[np.intp]
    event_hpd_overlap: NDArray[np.floating]
    event_kl_divergence: NDArray[np.floating]
    event_spike_prob: NDArray[np.floating]
    hpd_overlap: NDArray[np.floating] | None
    kl_divergence: NDArray[np.floating] | None
    spike_prob: NDArray[np.floating] | None
    per_spike_likelihood: NDArray[np.floating] | None
    # Real-data path supplies wall-clock spike times alongside the
    # bin indices; simulated paths leave this ``None``.
    event_time: NDArray[np.floating] | None = None

    def __post_init__(self) -> None:
        n_spikes = self.event_time_ind.shape[0]
        for name in (
            "event_cell_ind",
            "event_hpd_overlap",
            "event_kl_divergence",
            "event_spike_prob",
        ):
            arr = getattr(self, name)
            if arr.shape != (n_spikes,):
                raise ValueError(f"PerCellDiagnostics.{name} shape {arr.shape} != ({n_spikes},)")
        if self.event_time is not None and self.event_time.shape != (n_spikes,):
            raise ValueError(
                f"PerCellDiagnostics.event_time shape {self.event_time.shape} != ({n_spikes},)"
            )
        # Dense matrices are an all-or-nothing group.
        dense_names = ("hpd_overlap", "kl_divergence", "spike_prob", "per_spike_likelihood")
        dense_provided = [getattr(self, n) is not None for n in dense_names]
        if any(dense_provided) and not all(dense_provided):
            missing = [n for n, p in zip(dense_names, dense_provided, strict=True) if not p]
            raise ValueError(
                f"PerCellDiagnostics: dense matrices must be all-or-nothing; missing {missing}"
            )
        if self.hpd_overlap is not None:
            assert self.kl_divergence is not None  # narrowed by all-or-nothing
            assert self.spike_prob is not None
            assert self.per_spike_likelihood is not None
            n_time, n_cells = self.hpd_overlap.shape
            if self.kl_divergence.shape != (n_time, n_cells):
                raise ValueError(
                    f"kl_divergence shape {self.kl_divergence.shape} != ({n_time}, {n_cells})"
                )
            if self.spike_prob.shape != (n_time, n_cells):
                raise ValueError(
                    f"spike_prob shape {self.spike_prob.shape} != ({n_time}, {n_cells})"
                )
            if self.per_spike_likelihood.shape[0] != n_spikes:
                raise ValueError(
                    f"per_spike_likelihood leading dim {self.per_spike_likelihood.shape[0]} "
                    f"!= n_spikes={n_spikes}"
                )
        # Write-protect everything that's not None.
        for name in (
            "event_time_ind",
            "event_cell_ind",
            *_PER_EVENT_METRIC_NAMES,
            *dense_names,
            "event_time",
        ):
            arr = getattr(self, name)
            if arr is not None:
                arr.setflags(write=False)


@dataclass(frozen=True)
class Diagnostics:
    """Return of :func:`decode_and_diagnostics`.

    Frozen so downstream code cannot rebind fields; arrays are
    write-protected so it cannot mutate them in place either.

    Parameters
    ----------
    posterior, predictive, likelihood, spike_likelihood : np.ndarray, shape (n_time, n_bins)
        Dense distributions over position.
    hpd_overlap, kl_divergence, spike_prob : np.ndarray, shape (n_time, n_cells)
        Dense per-cell diagnostic matrices; ``NaN`` where no spike.
    event_time_ind, event_cell_ind : np.ndarray, shape (n_spikes,)
        Time-bin / cell index for each spike event.
    event_hpd_overlap, event_kl_divergence, event_spike_prob : np.ndarray, shape (n_spikes,)
        Per-event diagnostic values.
    per_spike_likelihood : np.ndarray, shape (n_spikes, n_bins)
        Per-spike normalized likelihood as seen by the decoder
        (uses ``decoder_rates`` inside misfit windows where set).

    Raises
    ------
    ValueError
        If shape invariants are violated — all dense ``(n_time, ...)``
        arrays must share leading dim, all per-event ``(n_spikes,)``
        arrays must share leading dim, dense ``(n_time, n_bins)``
        arrays must share trailing dim with each other, and dense
        ``(n_time, n_cells)`` arrays must share trailing dim with each
        other.
    """

    posterior: NDArray[np.floating]
    predictive: NDArray[np.floating]
    likelihood: NDArray[np.floating]
    spike_likelihood: NDArray[np.floating]
    hpd_overlap: NDArray[np.floating]
    kl_divergence: NDArray[np.floating]
    spike_prob: NDArray[np.floating]
    event_time_ind: NDArray[np.intp]
    event_cell_ind: NDArray[np.intp]
    event_hpd_overlap: NDArray[np.floating]
    event_kl_divergence: NDArray[np.floating]
    event_spike_prob: NDArray[np.floating]
    per_spike_likelihood: NDArray[np.floating]

    def __post_init__(self) -> None:
        # 2-D guard before unpacking — a 1-D ``posterior`` would
        # otherwise raise the less-informative ``IndexError`` on the
        # next line instead of the ``ValueError`` the docstring promises.
        if self.posterior.ndim != 2:
            raise ValueError(
                f"Diagnostics.posterior must be 2-D (n_time, n_bins); "
                f"got shape {self.posterior.shape}"
            )
        if self.hpd_overlap.ndim != 2:
            raise ValueError(
                "Diagnostics.hpd_overlap must be 2-D (n_time, n_cells); "
                f"got shape {self.hpd_overlap.shape}"
            )
        n_time, n_bins = self.posterior.shape
        for name in ("predictive", "likelihood", "spike_likelihood"):
            arr = getattr(self, name)
            if arr.shape != (n_time, n_bins):
                raise ValueError(f"Diagnostics.{name} shape {arr.shape} != ({n_time}, {n_bins})")
        n_cells = self.hpd_overlap.shape[1]
        for name in ("kl_divergence", "spike_prob"):
            arr = getattr(self, name)
            if arr.shape != (n_time, n_cells):
                raise ValueError(f"Diagnostics.{name} shape {arr.shape} != ({n_time}, {n_cells})")
        n_spikes = self.event_time_ind.shape[0]
        if self.event_cell_ind.shape != (n_spikes,):
            raise ValueError(
                f"Diagnostics.event_cell_ind shape {self.event_cell_ind.shape} != ({n_spikes},)"
            )
        for name in _PER_EVENT_METRIC_NAMES:
            arr = getattr(self, name)
            if arr.shape != (n_spikes,):
                raise ValueError(f"Diagnostics.{name} shape {arr.shape} != ({n_spikes},)")
        if self.per_spike_likelihood.shape != (n_spikes, n_bins):
            raise ValueError(
                f"Diagnostics.per_spike_likelihood shape "
                f"{self.per_spike_likelihood.shape} != ({n_spikes}, {n_bins})"
            )
        # Value-range invariants on the per-cell metrics + their per-event
        # counterparts. NaN is legitimate at (t, cell) without a spike, so
        # the range check ignores NaN. A buggy decoder otherwise ships
        # out-of-range values that only surface much later (e.g., as a NaN
        # ``Thresholds`` or a misleading hexbin).
        _check_range(self.hpd_overlap, "Diagnostics.hpd_overlap", lo=0.0, hi=1.0)
        _check_range(self.spike_prob, "Diagnostics.spike_prob", lo=0.0, hi=1.0)
        _check_range(self.kl_divergence, "Diagnostics.kl_divergence", lo=0.0, hi=None)
        _check_range(self.event_hpd_overlap, "Diagnostics.event_hpd_overlap", lo=0.0, hi=1.0)
        _check_range(self.event_spike_prob, "Diagnostics.event_spike_prob", lo=0.0, hi=1.0)
        _check_range(self.event_kl_divergence, "Diagnostics.event_kl_divergence", lo=0.0, hi=None)
        # Write-protect every backing buffer.
        for name in (
            "posterior",
            "predictive",
            "likelihood",
            "spike_likelihood",
            "hpd_overlap",
            "kl_divergence",
            "spike_prob",
            "event_time_ind",
            "event_cell_ind",
            *_PER_EVENT_METRIC_NAMES,
            "per_spike_likelihood",
        ):
            getattr(self, name).setflags(write=False)


# -----------------------------
# Decoder components
# -----------------------------


def likelihood_grid_for_counts(
    xs: NDArray[np.floating],
    pf_centers: NDArray[np.floating],
    pf_width: float,
    rate_scale: float,
    counts: NDArray[np.int_],
) -> NDArray[np.floating]:
    """Compute likelihood grid for spike counts.

    Computes the Poisson likelihood P(counts[cell] | position=xs[bin]) for each
    spatial bin and cell. The likelihood is normalized per cell (over bins) to
    form a proper probability distribution.

    Parameters
    ----------
    xs : np.ndarray, shape (n_bins,)
        Position grid (spatial bins).
    pf_centers : np.ndarray, shape (n_cells,)
        Place field center positions for each cell.
    pf_width : float
        Width (standard deviation) of Gaussian place fields.
    rate_scale : float
        Scaling factor for firing rates.
    counts : np.ndarray, shape (n_cells,)
        Observed spike counts for each cell at current time.

    Returns
    -------
    likelihood_grid : np.ndarray, shape (n_bins, n_cells)
        Normalized likelihood P(counts | position) for each bin and cell.
        Normalized per cell (columns sum to 1).

    Notes
    -----
    The likelihood is computed as:
    L_grid[bin, cell] ∝ P(counts[cell] | position=xs[bin])

    Uses Poisson distribution with rate λ = placefield_rates(xs, pf_centers, ...)

    Examples
    --------
    >>> import numpy as np
    >>> xs = np.linspace(0, 100, 21)
    >>> pf_centers = np.array([25.0, 50.0, 75.0])
    >>> pf_width = 5.0
    >>> rate_scale = 0.1
    >>> counts = np.array([2, 1, 3])
    >>> likelihood = likelihood_grid_for_counts(xs, pf_centers, pf_width, rate_scale, counts)
    >>> likelihood.shape
    (21, 3)
    >>> np.allclose(np.sum(likelihood, axis=0), 1.0)  # Normalized per cell
    True
    """
    rates = placefield_rates(xs, pf_centers, pf_width, rate_scale)  # (n_bins, n_cells)
    # Poisson PMF per bin, per cell for this time's counts
    # counts is (n_cells,), rates is (n_bins, n_cells)
    likelihood_grid: NDArray[np.floating] = poisson.pmf(counts[None, :], rates)
    # Avoid degenerate zeros; normalize per cell (over bins) to a proper density on xs
    likelihood_grid = normalize(likelihood_grid, axis=0)
    return likelihood_grid


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


def decode_and_diagnostics(
    spikes: NDArray[np.int_],
    xs: NDArray[np.floating],
    transition_matrix: NDArray[np.floating],
    pf_centers: NDArray[np.floating],
    pf_width: float,
    rate_scale: float,
    misfit_schedule: MisfitSchedule | None = None,
    rng: np.random.Generator | None = None,
) -> Diagnostics:
    """Run the Bayesian filter with per-time, per-cell diagnostics.

    This function implements a Bayesian decoder for position from neural spikes,
    computing diagnostic metrics at each timestep to assess model goodness-of-fit.

    **Algorithm**:
    1. Initialize with flat prior at t=0
    2. For each timestep t:
       a. Predict: prior = transition_matrix @ post[t-1]
       b. Likelihood: compute P(spikes[t] | position) for all cells
       c. Diagnostics: compare prior vs. combined likelihood
       d. Update: post[t] = normalize(prior * combined_likelihood)

    **Diagnostics**:
    - HPD overlap: overlap between prior and combined likelihood HPD regions
    - KL divergence: divergence from prior to combined likelihood
    - Spike probability: cumulative probability mass for low-contribution cells

    Parameters
    ----------
    spikes : np.ndarray, shape (n_time, n_cells)
        Observed spike counts at each timestep for each cell.
    xs : np.ndarray, shape (n_bins,)
        Position grid (spatial bins).
    transition_matrix : np.ndarray, shape (n_bins, n_bins)
        State transition matrix for baseline dynamics.
    pf_centers : np.ndarray, shape (n_cells,)
        Place field center positions for each cell.
    pf_width : float
        Width (standard deviation) of Gaussian place fields.
    rate_scale : float
        Scaling factor for firing rates.
    misfit_schedule : MisfitSchedule, optional
        Decoder-side misfit windows — remapping and wide-dynamics noise.
        Each :class:`MisfitWindow` swaps the transition matrix and/or
        the per-cell rate table for its interval. Defaults to an empty
        schedule: a clean decode with no
        misfits (the real-data decoding case).
    rng : np.random.Generator | None, optional
        Random number generator (reserved for future use).

    Returns
    -------
    results : Diagnostics
        Frozen dataclass with the following fields (see
        :class:`Diagnostics` for the full schema):

        Dense ``(n_time, n_bins)`` distributions
            ``posterior`` (filtered posterior), ``predictive`` (one-step
            ahead, flat at t=0), ``likelihood`` (normalized combined
            likelihood from all cells, flat at t=0), and
            ``spike_likelihood`` (combined likelihood from only spiking
            cells; NaN where no spikes).

        Dense ``(n_time, n_cells)`` per-cell diagnostic matrices
            ``hpd_overlap``, ``kl_divergence``, ``spike_prob``. NaN at
            t=0 and at any (t, cell) without a spike.

        Per-spike-event arrays of shape ``(n_spikes,)``
            ``event_time_ind`` (time bin), ``event_cell_ind`` (cell
            index), and ``event_hpd_overlap`` / ``event_kl_divergence``
            / ``event_spike_prob`` (the dense matrices scattered to one
            value per event). Spike-count > 1 in a bin produces that
            many repeated events. The legacy ``spike_time_ind`` /
            ``spike_cell_ind`` aliases were removed; use the
            ``event_*_ind`` fields instead.

        ``per_spike_likelihood`` of shape ``(n_spikes, n_bins)``
            Normalized likelihood for each individual spike event,
            computed against the decoder's actual rates (remapped
            inside any misfit window with ``decoder_rates`` set).

    Notes
    -----
    Invalid misfit configurations (overlapping windows, ``start >= end``,
    negative/non-finite rate tables) are rejected when the
    :class:`MisfitSchedule` / :class:`MisfitWindow` is *constructed*, not
    here.

    The per-cell likelihood combination and the posterior update both
    run in log-space via the :func:`_condition_on` pattern adapted from
    ``dynamax`` / ``non_local_detector.core``. The posterior update
    itself cannot underflow on the inner step (it uses an explicit
    log-sum-exp shift). The stored ``combined_likelihood`` and
    ``spike_likelihood`` arrays are renormalized after the same shift,
    so individual bins still underflow to zero in linear space but the
    row as a whole remains a proper probability distribution.

    When the prior and combined likelihood have no meaningful overlap
    (``weighted.sum() < eps`` inside ``_condition_on``), the helper
    falls back to a uniform posterior and signals via
    ``log_norm = -inf``. This function counts such steps and emits a
    single summary ``RuntimeWarning`` at the end so the situation is
    visible. Diagnostics at flagged steps reference a uniform
    predictive; consumers should mask them out if needed.

    Examples
    --------
    >>> import numpy as np
    >>> from statespacecheck_paper.simulation import gaussian_transition_matrix
    >>> # Set up small problem
    >>> n_time, n_cells, n_bins = 10, 3, 21
    >>> spikes = np.random.poisson(1.0, size=(n_time, n_cells))
    >>> xs = np.linspace(0, 100, n_bins)
    >>> transition_matrix = gaussian_transition_matrix(xs, sig=0.5)
    >>> pf_centers = np.array([25.0, 50.0, 75.0])
    >>> pf_width = 5.0
    >>> rate_scale = 0.1
    >>> # Clean decode, no misfits
    >>> results = decode_and_diagnostics(
    ...     spikes, xs, transition_matrix, pf_centers, pf_width, rate_scale
    ... )
    >>> results.posterior.shape
    (10, 21)
    >>> results.hpd_overlap.shape  # Now per-cell
    (10, 3)
    >>> bool(np.all(np.isnan(results.hpd_overlap[0])))  # t=0 has no prior
    True
    """
    n_time = spikes.shape[0]
    n_bins = xs.size

    # rng parameter reserved for future use
    _ = rng

    if misfit_schedule is None:
        misfit_schedule = MisfitSchedule()

    # Shape-validate every schedule entry against the decoder's grid.
    # MisfitWindow's __post_init__ can't check this because the schedule
    # may be built before xs / spikes are pinned down.
    n_cells = spikes.shape[1]
    for schedule_entry in misfit_schedule.windows:
        schedule_entry.validate_against(n_bins=n_bins, n_cells=n_cells)

    # Preallocate outputs
    posterior: NDArray[np.floating] = np.zeros((n_time, n_bins))
    predictive_posterior: NDArray[np.floating] = np.zeros((n_time, n_bins))  # p(x_t | y_{1:t-1})
    combined_likelihood_all: NDArray[np.floating] = np.zeros((n_time, n_bins))  # p(y_t | x_t)
    # Spike-only likelihood: product over only cells that fired (for display).
    # NaN at times with no spikes.
    spike_likelihood_all: NDArray[np.floating] = np.full((n_time, n_bins), np.nan)

    # t=0: flat prior. Diagnostics at t=0 are NaN (no posterior update
    # has happened yet); downstream code masks those entries.
    posterior[0] = normalize(np.ones(n_bins))
    predictive_posterior[0] = posterior[0]  # At t=0, predictive = prior
    combined_likelihood_all[0] = normalize(np.ones(n_bins))  # Flat at t=0

    # Baseline per-cell Poisson rate table. Used at every timestep not
    # covered by a misfit window whose ``decoder_rates`` is set, and as
    # the default diagnostic rate table.
    rates = placefield_rates(xs, pf_centers, pf_width, rate_scale)

    # Track timesteps where _condition_on fell back to uniform (prior and
    # likelihood had essentially no overlap). The fallback keeps the
    # filter alive but the diagnostics at those steps are computed
    # against a uniform predictive; report a summary at the end so the
    # caller can mask them out if desired.
    n_underflow_steps = 0
    first_underflow_t = -1

    for t in range(1, n_time):
        window = misfit_schedule.window_at(t)

        # Predict — baseline transition unless the active misfit window
        # overrides it.
        current_transition = transition_matrix
        if window is not None and window.transition_matrix is not None:
            current_transition = window.transition_matrix
        # ``current_transition`` is column-stochastic: column j is the
        # distribution over next states given current state j (see
        # ``gaussian_transition_matrix``). The predictive marginal is therefore
        # ``T @ post``, not ``post @ T`` — the two differ near the track
        # boundaries where column normalization breaks the kernel's symmetry.
        prior = normalize(current_transition @ posterior[t - 1])  # (n_bins,)
        predictive_posterior[t] = prior  # stored for p-value computation

        # Per-cell rate table for this step. The baseline Gaussian-PF
        # table unless the active misfit window overrides it (e.g.
        # remapped table).
        rates_t = rates
        if window is not None and window.decoder_rates is not None:
            rates_t = window.decoder_rates

        # Per-cell log-likelihoods. Log-space avoids underflow when
        # ``n_cells * log(peak)`` crosses the float64 floor (~700) —
        # likely on real-data sessions with many sparsely-firing cells.
        log_lik_per_cell = poisson.logpmf(spikes[t][None, :], rates_t)  # (n_bins, n_cells)

        # Combined log-likelihood across cells (sum in log space =
        # product in linear space).
        log_lik_combined = log_lik_per_cell.sum(axis=1)  # (n_bins,)

        # Stored combined likelihood: same shift-and-normalize math as
        # _condition_on, factored into the shared helper.
        combined_likelihood_all[t] = softmax_with_shift(log_lik_combined)

        # Spike-only likelihood: product over only cells that fired.
        spiking_mask = spikes[t] > 0
        if np.any(spiking_mask):
            spike_likelihood_all[t] = softmax_with_shift(
                log_lik_per_cell[:, spiking_mask].sum(axis=1)
            )

        # Posterior update via the _condition_on pattern (dynamax /
        # non_local_detector). ``log_norm = -inf`` flags steps where the
        # prior and likelihood had no meaningful overlap; the helper
        # explicitly returns a uniform posterior in that case and we
        # surface the count post-loop rather than letting the situation
        # silently propagate.
        posterior[t], log_norm = _condition_on(prior, log_lik_combined)
        if log_norm == _NEG_INF:
            if n_underflow_steps == 0:
                first_underflow_t = t
            n_underflow_steps += 1

    if n_underflow_steps > 0:
        warnings.warn(
            f"decode_and_diagnostics: prior/likelihood overlap underflowed at "
            f"{n_underflow_steps} timestep(s); first at t={first_underflow_t}. "
            f"Posterior was reset to uniform at those steps; downstream "
            f"per-spike diagnostics computed at those times reference a "
            f"uniform predictive and should be interpreted accordingly.",
            RuntimeWarning,
            stacklevel=2,
        )

    # Find all spike events (excluding t=0 which has no valid prior). Count
    # matrices are expanded so a bin with count k contributes k spike events.
    spike_time_ind, spike_cell_ind = np.nonzero(spikes[1:])
    spike_counts_at_events = spikes[1:][spike_time_ind, spike_cell_ind].astype(np.intp)
    spike_time_ind = np.repeat(spike_time_ind, spike_counts_at_events)
    spike_cell_ind = np.repeat(spike_cell_ind, spike_counts_at_events)
    spike_time_ind = spike_time_ind + 1  # Adjust for offset from [1:]
    n_spikes = len(spike_time_ind)

    # Diagnostics. Every spike event is judged against the baseline
    # Gaussian-PF ``rates`` — the model's intended likelihood. During
    # remapping the decoder updates the posterior with remapped fields
    # but the diagnostic still references the original fields, so the
    # mismatch surfaces.
    diagnostics = compute_per_cell_diagnostics_from_rates(
        predictive_posterior,
        rates,
        spike_time_ind,
        spike_cell_ind,
        coverage=0.95,
    )

    # Per-spike likelihoods from the DECODER's actual rate table — for
    # display in the likelihood panel. Baseline Gaussian-PF rates, then
    # each misfit window with ``decoder_rates`` set overwrites its own
    # (disjoint) events with that table.
    decoder_per_spike_lik: NDArray[np.floating] = np.zeros((n_spikes, n_bins))
    if n_spikes > 0:
        rates_orig = rates[:, spike_cell_ind].T  # (n_spikes, n_bins)
        decoder_per_spike_lik = normalize(poisson.pmf(k=1, mu=rates_orig), axis=1)

        for window in misfit_schedule.windows:
            if window.decoder_rates is None:
                continue
            in_window = (spike_time_ind >= window.start) & (spike_time_ind < window.end)
            if np.any(in_window):
                cell_rates = window.decoder_rates[:, spike_cell_ind[in_window]].T
                decoder_per_spike_lik[in_window] = normalize(
                    poisson.pmf(k=1, mu=cell_rates), axis=1
                )

    assert diagnostics.hpd_overlap is not None  # called with include_dense_matrices=True
    assert diagnostics.kl_divergence is not None
    assert diagnostics.spike_prob is not None
    return Diagnostics(
        posterior=posterior,
        predictive=predictive_posterior,
        likelihood=combined_likelihood_all,
        spike_likelihood=spike_likelihood_all,
        hpd_overlap=diagnostics.hpd_overlap,
        kl_divergence=diagnostics.kl_divergence,
        spike_prob=diagnostics.spike_prob,
        per_spike_likelihood=decoder_per_spike_lik,
        event_time_ind=diagnostics.event_time_ind,
        event_cell_ind=diagnostics.event_cell_ind,
        event_hpd_overlap=diagnostics.event_hpd_overlap,
        event_kl_divergence=diagnostics.event_kl_divergence,
        event_spike_prob=diagnostics.event_spike_prob,
    )


# -----------------------------
# Per-cell diagnostics (shared logic)
# -----------------------------


def compute_per_cell_diagnostics_from_rates(
    predictive_posterior: NDArray[np.floating],
    rates: NDArray[np.floating],
    spike_time_ind: NDArray[np.intp],
    spike_cell_ind: NDArray[np.intp],
    coverage: float = 0.95,
    include_dense_matrices: bool = True,
) -> PerCellDiagnostics:
    """Compute per-cell diagnostic metrics at spike times.

    This is the core computation shared by both simulated and real data analysis.
    It computes HPD overlap, KL divergence, and spike probability ranking for
    each spike event, assuming each spike represents exactly one spike (k=1).

    Parameters
    ----------
    predictive_posterior : np.ndarray, shape (n_time, n_bins)
        Predictive posterior distribution over position at each time.
    rates : np.ndarray, shape (n_bins, n_cells)
        Expected spike rate (spikes/bin) at each position for each cell.
    spike_time_ind : np.ndarray, shape (n_spikes,)
        Time indices where spikes occurred.
    spike_cell_ind : np.ndarray, shape (n_spikes,)
        Cell indices for each spike event.
    coverage : float, default 0.95
        Coverage probability for HPD region computation.
    include_dense_matrices : bool, default True
        If True (default), also return the (n_time, n_cells) ``hpd_overlap``,
        ``kl_divergence``, ``spike_prob`` matrices and the (n_spikes, n_bins)
        ``per_spike_likelihood``. If False, those four keys are omitted from
        the result dict and the matching allocations / scatters are skipped
        — useful for callers that only need the per-spike event arrays
        (the cache builder is the canonical example), since for real
        recordings the dense matrices can be hundreds of MB.

    Returns
    -------
    diagnostics : dict[str, np.ndarray]
        Always contains:

        - 'spike_time_ind' / 'event_time_ind': shape (n_spikes,)
        - 'spike_cell_ind' / 'event_cell_ind': shape (n_spikes,)
        - 'event_hpd_overlap': shape (n_spikes,), per-spike HPD overlap
        - 'event_kl_divergence': shape (n_spikes,), per-spike KL divergence
        - 'event_spike_prob': shape (n_spikes,), per-spike spike probability

        If ``include_dense_matrices`` (the default), additionally:

        - 'hpd_overlap': shape (n_time, n_cells), NaN where no spike
        - 'kl_divergence': shape (n_time, n_cells), NaN where no spike
        - 'spike_prob': shape (n_time, n_cells), NaN where no spike
        - 'per_spike_likelihood': shape (n_spikes, n_bins), normalized
          likelihood distribution for each individual spike event

    Notes
    -----
    The likelihood P(k=1 | position) is computed for each spike event. If
    multiple spikes occur in the same time/cell bin, callers should pass
    repeated entries in ``spike_time_ind`` and ``spike_cell_ind`` so each
    observed spike contributes one event to the returned event arrays.
    """
    n_time, n_bins = predictive_posterior.shape
    n_cells = rates.shape[1]
    n_spikes = len(spike_time_ind)

    event_hpd_overlap: NDArray[np.floating] = np.empty(n_spikes)
    event_kl_divergence: NDArray[np.floating] = np.empty(n_spikes)
    event_spike_prob: NDArray[np.floating] = np.empty(n_spikes)

    # Dense (n_time, n_cells) matrices are only allocated when requested;
    # for real recordings with millions of time bins they can dwarf the
    # rest of the working set, so the cache builder opts out.
    hpd_overlap: NDArray[np.floating] | None = None
    kl_divergence: NDArray[np.floating] | None = None
    spike_prob: NDArray[np.floating] | None = None
    per_spike_likelihood: NDArray[np.floating] | None = None
    if include_dense_matrices:
        hpd_overlap = np.full((n_time, n_cells), np.nan)
        kl_divergence = np.full((n_time, n_cells), np.nan)
        spike_prob = np.full((n_time, n_cells), np.nan)
        per_spike_likelihood = np.empty((n_spikes, n_bins))

    if n_spikes > 0:
        # Per-spike Poisson-likelihood / HPD / KL / spike-prob all need
        # ``(S, n_bins)`` or ``(S, n_cells)`` working arrays. For
        # full-session real-data builds (~870 K spikes × 256 bins
        # × 8 B ≈ 1.8 GB *per array*) materializing them in one shot
        # blows the working set even when ``include_dense_matrices=False``
        # skips the (n_time, n_cells) outputs. Process in chunks to
        # bound peak memory to ``_PER_SPIKE_BATCH × n_bins × 8 B`` per
        # scratch array. ``spike_prob`` is the per-event rank
        # ``sum_i contrib[i] where contrib[i] <= contrib[j]``;
        # computing it per event (not vectorized over unique times)
        # bounds the rank computation's working set to
        # ``B × n_cells``.
        # ``cell_fraction_per_bin[x, c] = p(cell c | bin x, spike happened)``.
        # Bins where every cell has zero rate (sparse real-data coverage
        # away from any PF) would trigger ``normalize``'s near-zero
        # warning and produce a non-discriminative near-zero row in
        # ``cell_fraction_per_bin``. Use a uniform ``1/n_cells`` fallback
        # so the rank statistic (and therefore ``event_spike_prob``)
        # treats those bins as non-informative. Note this branch does
        # not protect ``event_hpd_overlap`` / ``event_kl_divergence``,
        # which consume the per-cell Poisson ``lik_chunk`` directly.
        row_sums = rates.sum(axis=1, keepdims=True)
        zero_rows = row_sums.squeeze(-1) <= 1e-12
        safe_row_sums = np.where(row_sums > 1e-12, row_sums, 1.0)
        cell_fraction_per_bin = rates / safe_row_sums  # (n_bins, n_cells)
        if zero_rows.any():
            cell_fraction_per_bin[zero_rows] = 1.0 / rates.shape[1]
        batch = max(1, _PER_SPIKE_BATCH)
        for start in range(0, n_spikes, batch):
            stop = min(start + batch, n_spikes)
            sti = spike_time_ind[start:stop]
            sci = spike_cell_ind[start:stop]
            chunk_size = stop - start

            # (chunk, n_bins) gathers + Poisson lik for this batch only.
            pred_chunk = predictive_posterior[sti]
            rates_chunk = rates[:, sci].T
            lik_chunk = normalize(poisson.pmf(k=1, mu=rates_chunk), axis=1)

            event_hpd_overlap[start:stop] = ssc.hpd_overlap(
                pred_chunk, lik_chunk, coverage=coverage
            )
            event_kl_divergence[start:stop] = ssc.kl_divergence(pred_chunk, lik_chunk)

            # Per-event spike-prob rank: contrib[k, j] is cell ``j``'s
            # expected contribution at this event's time, target is
            # the contribution of *this event's cell*, and the rank is
            # the cumulative mass of cells with weakly smaller contrib.
            # The ``rank_atol`` slack on the ``<=`` comparison absorbs
            # BLAS reduction-order FP noise so equal contributions
            # yield equal ranks across platforms (matches the same
            # tolerance pattern in ``simulation.spike_prob_rank``).
            contrib_chunk = pred_chunk @ cell_fraction_per_bin  # (B, n_cells)
            target_contrib = contrib_chunk[np.arange(chunk_size), sci]  # (B,)
            rank_atol = (
                float(np.finfo(contrib_chunk.dtype).eps * n_bins * 16)
                * float(np.max(contrib_chunk))
                if contrib_chunk.size
                else 0.0
            )
            rank_mask = contrib_chunk <= target_contrib[:, None] + rank_atol
            event_spike_prob[start:stop] = (contrib_chunk * rank_mask).sum(axis=1)

            if per_spike_likelihood is not None:
                per_spike_likelihood[start:stop] = lik_chunk

        if hpd_overlap is not None:
            hpd_overlap[spike_time_ind, spike_cell_ind] = event_hpd_overlap
        if kl_divergence is not None:
            kl_divergence[spike_time_ind, spike_cell_ind] = event_kl_divergence
        if spike_prob is not None:
            spike_prob[spike_time_ind, spike_cell_ind] = event_spike_prob

    return PerCellDiagnostics(
        event_time_ind=spike_time_ind,
        event_cell_ind=spike_cell_ind,
        event_hpd_overlap=event_hpd_overlap,
        event_kl_divergence=event_kl_divergence,
        event_spike_prob=event_spike_prob,
        hpd_overlap=hpd_overlap,
        kl_divergence=kl_divergence,
        spike_prob=spike_prob,
        per_spike_likelihood=per_spike_likelihood,
    )


# -----------------------------
# Thresholds & transforms
# -----------------------------


@dataclass(frozen=True)
class Thresholds:
    """Threshold values for diagnostic metrics.

    Computed from the baseline period across all cells (flattened).
    Frozen so a downstream consumer cannot rebind a field mid-pipeline.

    Parameters
    ----------
    hpd_overlap : float
        HPD overlap threshold; must lie in ``[0, 1]`` (the underlying
        diagnostic is a probability overlap). Lower values indicate
        worse fit.
    kl_divergence : float
        KL divergence threshold; must be non-negative finite. Higher
        values indicate worse fit.
    spike_prob : float
        Spike-probability threshold; must lie in ``[0, 1]``. Defaulted
        to 0.05 by :func:`compute_thresholds`. Lower values indicate
        misfit.

    Raises
    ------
    ValueError
        If any field falls outside its documented range, or is NaN.
        The construction-time check prevents a NaN threshold (e.g.
        from an all-NaN baseline) silently making every downstream
        ``metric < threshold`` comparison evaluate ``False``.

    Examples
    --------
    >>> thresholds = Thresholds(
    ...     hpd_overlap=0.5,
    ...     kl_divergence=2.0,
    ...     spike_prob=0.05,
    ... )
    >>> thresholds.hpd_overlap
    0.5
    """

    hpd_overlap: float
    kl_divergence: float
    spike_prob: float

    def __post_init__(self) -> None:
        if not (0.0 <= self.hpd_overlap <= 1.0):
            raise ValueError(f"Thresholds.hpd_overlap must lie in [0, 1]; got {self.hpd_overlap}")
        if not (np.isfinite(self.kl_divergence) and self.kl_divergence >= 0.0):
            raise ValueError(
                f"Thresholds.kl_divergence must be finite and non-negative; "
                f"got {self.kl_divergence}"
            )
        if not (0.0 <= self.spike_prob <= 1.0):
            raise ValueError(f"Thresholds.spike_prob must lie in [0, 1]; got {self.spike_prob}")


def compute_thresholds(
    metrics: Diagnostics | Mapping[str, NDArray[np.floating] | NDArray[np.intp]],
    *,
    baseline_end: int,
) -> Thresholds:
    """Compute threshold values from baseline period.

    Thresholds are computed across all cells (flattened (n_time, n_cells)
    → 1D) so a single threshold scalar can compare against any cell's
    diagnostic time series:

    - HPD overlap threshold: 1st percentile (low values indicate misfit)
    - KL divergence threshold: 99th percentile (high values indicate misfit)
    - spike_prob threshold: fixed at 0.05 (a conventional rank-statistic cutoff)

    Parameters
    ----------
    metrics : Diagnostics or Mapping[str, NDArray]
        Either a :class:`Diagnostics` (the typical caller, produced by
        :func:`decode_and_diagnostics`) or a plain dict with keys
        ``hpd_overlap``, ``kl_divergence``, ``spike_prob`` — the dict
        form is retained so synthetic test fixtures don't need to
        construct a full ``Diagnostics``.
    baseline_end : int, keyword-only
        Index marking end of baseline period (exclusive). Required —
        silently slicing the whole recording would contaminate
        "baseline" thresholds with misfit data and is rarely what
        the caller intends.

    Returns
    -------
    thresholds : Thresholds
        Threshold values for each diagnostic metric.

    Raises
    ------
    ValueError
        If the baseline slice of ``hpd_overlap`` or ``kl_divergence``
        contains no finite values (thresholds would be NaN and
        downstream comparisons would silently evaluate False).

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(42)
    >>> metrics = {
    ...     'hpd_overlap': rng.uniform(0.5, 1.0, (100, 5)),
    ...     'kl_divergence': rng.uniform(0.0, 2.0, (100, 5)),
    ...     'spike_prob': rng.uniform(0.0, 1.0, (100, 5)),
    ... }
    >>> thresholds = compute_thresholds(metrics, baseline_end=50)
    >>> thresholds.spike_prob  # Fixed at 0.05
    0.05
    """

    def _get(name: str) -> NDArray[np.floating]:
        arr = getattr(metrics, name) if isinstance(metrics, Diagnostics) else metrics[name]
        return cast("NDArray[np.floating]", arr)

    # Flatten (n_time, n_cells) to 1D for quantile computation. ``np.nanquantile``
    # returns ``np.floating``; cast to plain ``float`` to match the ``Thresholds``
    # dataclass signature.
    hpd_baseline = _get("hpd_overlap")[:baseline_end].ravel()
    if not np.any(np.isfinite(hpd_baseline)):
        raise ValueError(
            "compute_thresholds: hpd_overlap baseline slice "
            f"(:{baseline_end}) contains no finite values; threshold "
            "would be NaN."
        )
    hpd_overlap_threshold = float(np.nanquantile(hpd_baseline, 0.01))

    kl_baseline = _get("kl_divergence")[:baseline_end].ravel()
    if not np.any(np.isfinite(kl_baseline)):
        raise ValueError(
            "compute_thresholds: kl_divergence baseline slice "
            f"(:{baseline_end}) contains no finite values; threshold "
            "would be NaN."
        )
    kl_divergence_threshold = float(np.nanquantile(kl_baseline, 0.99))

    # Fixed rank-statistic cutoff; not derived from the data.
    spike_prob_threshold = 0.05

    return Thresholds(
        hpd_overlap=hpd_overlap_threshold,
        kl_divergence=kl_divergence_threshold,
        spike_prob=spike_prob_threshold,
    )


@dataclass(frozen=True)
class Transformed:
    """Transformed diagnostic metrics and thresholds.

    Transformations applied to improve visualization dynamic range:
    ``-log10(HPDO + eps1)``, ``sqrt(KL)``, ``-log10(spikeProb + eps2)``.
    The three metric arrays must share ``(n_time, n_cells)`` — downstream
    heatmaps stack them and a shape mismatch would silently misalign
    cells against threshold rows. Construction-time check + write-protect
    catch both classes of regression.

    Parameters
    ----------
    hpd_overlap : np.ndarray, shape (n_time, n_cells)
        Transformed HPD overlap values: ``-log10(HPDO + eps1)``.
    kl_divergence : np.ndarray, shape (n_time, n_cells)
        Transformed KL divergence values: ``sqrt(KL)``.
    spike_prob : np.ndarray, shape (n_time, n_cells)
        Transformed spike probability values: ``-log10(spikeProb + eps2)``.
    hpd_overlap_threshold : float
        Transformed HPD overlap threshold.
    kl_divergence_threshold : float
        Transformed KL divergence threshold.
    spike_prob_threshold : float
        Transformed spike probability threshold.

    Raises
    ------
    ValueError
        If the three metric arrays don't share a shape.

    Examples
    --------
    >>> import numpy as np
    >>> transformed = Transformed(
    ...     hpd_overlap=np.array([[1.0, 2.0], [3.0, 4.0]]),
    ...     kl_divergence=np.array([[0.5, 1.0], [1.5, 2.0]]),
    ...     spike_prob=np.array([[0.1, 0.5], [0.9, 1.2]]),
    ...     hpd_overlap_threshold=1.5,
    ...     kl_divergence_threshold=1.0,
    ...     spike_prob_threshold=23.0,
    ... )
    >>> transformed.hpd_overlap_threshold
    1.5
    """

    hpd_overlap: NDArray[np.floating]
    kl_divergence: NDArray[np.floating]
    spike_prob: NDArray[np.floating]
    hpd_overlap_threshold: float
    kl_divergence_threshold: float
    spike_prob_threshold: float

    def __post_init__(self) -> None:
        shape = self.hpd_overlap.shape
        if self.kl_divergence.shape != shape:
            raise ValueError(
                f"Transformed.kl_divergence shape {self.kl_divergence.shape} "
                f"does not match hpd_overlap shape {shape}"
            )
        if self.spike_prob.shape != shape:
            raise ValueError(
                f"Transformed.spike_prob shape {self.spike_prob.shape} "
                f"does not match hpd_overlap shape {shape}"
            )
        # Write-protect the metric arrays. ``frozen=True`` blocks
        # rebinding the field, not mutation through the bound ndarray.
        for name in ("hpd_overlap", "kl_divergence", "spike_prob"):
            getattr(self, name).setflags(write=False)


def transform_metrics(
    metrics: Diagnostics | Mapping[str, NDArray[np.floating]],
    thresholds: Thresholds,
    eps1: float = 1e-2,
    eps2: float = 1e-10,
) -> Transformed:
    """Apply transformations to metrics for better visualization.

    **Transformations**:
    - HPD overlap: -log10(HPDO + eps1) - emphasizes low values (worse fit)
    - KL divergence: sqrt(KL) - compresses high values
    - spike_prob: -log10(spikeProb + eps2) - emphasizes low values (worse fit)

    The same transformations are applied to the threshold values.

    Parameters
    ----------
    metrics : dict[str, NDArray]
        Dictionary containing diagnostic metrics:
        - 'hpd_overlap' : np.ndarray, shape (n_time, n_cells)
        - 'kl_divergence' : np.ndarray, shape (n_time, n_cells)
        - 'spike_prob' : np.ndarray, shape (n_time, n_cells)
    thresholds : Thresholds
        Threshold values for each diagnostic metric.
    eps1 : float, default 1e-2
        Small constant added before log transform for HPD overlap.
    eps2 : float, default 1e-10
        Small constant added before log transform for spike_prob.

    Returns
    -------
    transformed : Transformed
        Transformed diagnostic metrics and thresholds.

    Notes
    -----
    NaN values in input metrics are preserved in the output.

    Examples
    --------
    >>> import numpy as np
    >>> metrics = {
    ...     'hpd_overlap': np.array([[0.5, 0.8], [0.9, 0.7]]),
    ...     'kl_divergence': np.array([[1.0, 4.0], [9.0, 16.0]]),
    ...     'spike_prob': np.array([[0.1, 0.5], [0.01, 0.05]]),
    ... }
    >>> thresholds = Thresholds(
    ...     hpd_overlap=0.6,
    ...     kl_divergence=5.0,
    ...     spike_prob=0.05,
    ... )
    >>> transformed = transform_metrics(metrics, thresholds)
    >>> transformed.kl_divergence  # sqrt(KL)
    array([[1., 2.],
           [3., 4.]])
    >>> np.allclose(transformed.spike_prob_threshold, -np.log10(0.05 + 1e-10))
    True
    """

    def _get(name: str) -> NDArray[np.floating]:
        arr = getattr(metrics, name) if isinstance(metrics, Diagnostics) else metrics[name]
        return cast("NDArray[np.floating]", arr)

    hpd_overlap_transformed = -np.log10(np.maximum(_get("hpd_overlap") + eps1, 1e-10))
    kl_divergence_transformed = np.sqrt(_get("kl_divergence"))
    spike_prob_transformed = -np.log10(np.maximum(_get("spike_prob") + eps2, 1e-10))

    return Transformed(
        hpd_overlap=hpd_overlap_transformed,
        kl_divergence=kl_divergence_transformed,
        spike_prob=spike_prob_transformed,
        hpd_overlap_threshold=-np.log10(max(thresholds.hpd_overlap + eps1, 1e-10)),
        kl_divergence_threshold=np.sqrt(thresholds.kl_divergence),
        spike_prob_threshold=-np.log10(max(thresholds.spike_prob + eps2, 1e-10)),
    )
