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
    >>> sorted(results.keys())  # doctest: +NORMALIZE_WHITESPACE
    ['event_cell_ind', 'event_hpd_overlap', 'event_kl_divergence',
     'event_spike_prob', 'event_time_ind', 'hpd_overlap', 'kl_divergence',
     'likelihood', 'per_spike_likelihood', 'posterior', 'predictive',
     'spike_cell_ind', 'spike_likelihood', 'spike_prob', 'spike_time_ind']
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import statespacecheck as ssc
from numpy.typing import NDArray
from scipy.stats import poisson

from statespacecheck_paper.simulation import (
    normalize,
    placefield_rates,
)

# Spike-batch size for ``compute_per_cell_diagnostics_from_rates``.
# Caps the (n_spikes, n_bins) scratch arrays at ``_PER_SPIKE_BATCH``
# rows so full-session real-data builds (~870 K spikes) don't allocate
# multi-GB working buffers. 50 K × 512 bins × float64 ≈ 200 MB per
# scratch array, ~600 MB peak with three live (pred / rates / lik).
_PER_SPIKE_BATCH = 50_000

# -----------------------------
# Data containers
# -----------------------------


@dataclass
class DecodeParams:
    """Parameters for the figure-3 decoding simulation.

    The simulation walks through five misfit conditions separated by
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
    - 32k–34k: Clean recovery
    - 34k–38k: Wiggly-flat-likelihood misfit (4 s)

    Parameters
    ----------
    T_remap_start : int, default 6_000
        Start of remap misfit window (end of clean baseline).
    T_remap_end : int, default 10_000
        End of remap misfit window.
    T_recovery1_end : int, default 14_000
        End of first recovery period.
    T_hist_dep_end : int, default 18_000
        End of history-dependent firing misfit window. Spikes use
        :func:`simulate_spikes_history_dependent` (refractory +
        bursting); decoder still assumes Poisson. The misfit lives
        in spike-train temporal correlations, not in the per-spike
        spatial likelihood, so per-spike diagnostics largely miss it
        — a deliberate demonstration of the spatial-only nature of
        the metrics.
    T_recovery2_end : int, default 22_000
        End of second recovery period.
    T_drift_end : int, default 26_000
        End of drift misfit window. Animal trajectory has persistent
        velocity (momentum = 0.8); decoder assumes memoryless random
        walk.
    T_recovery3_end : int, default 30_000
        End of third recovery period.
    T_wide_dynamics_end : int, default 32_000
        End of wide-dynamics-noise misfit window. Decoder applies an
        inflated transition matrix (``sigx_wide_dynamics``) while the
        animal walks normally; engineered to inflate KL while HPD
        overlap and the rank-based p-value stay near baseline (the
        KL-false-positive case).
    T_recovery4_end : int, default 34_000
        End of fourth recovery period.
    T_wiggly_end : int, default 38_000
        End of wiggly-flat-likelihood misfit window. Decoder uses
        per-cell wiggly-flat rate functions (see
        :func:`statespacecheck_paper.simulation.wiggly_flat_rates`)
        for both posterior updates and diagnostic rate matrix during
        this window. The per-spike likelihood is wiggly-flat instead
        of Gaussian; HPDO becomes unstable (irregular HPD region) and
        the rank-based p-value becomes ambiguous (no clearly
        most-expected cell).
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
    >>> params.remap_window
    (6000, 10000)
    >>> params.pf_centers
    array([  0.,  10.,  20.,  30.,  40.,  50.,  60.,  70.,  80.,  90., 100.])
    """

    # Timeline (1 ms steps by convention; the math is dt-agnostic)
    T_remap_start: int = 6_000
    T_remap_end: int = 10_000
    T_recovery1_end: int = 14_000
    T_hist_dep_end: int = 18_000
    T_recovery2_end: int = 22_000
    T_drift_end: int = 26_000
    T_recovery3_end: int = 30_000
    T_wide_dynamics_end: int = 32_000
    T_recovery4_end: int = 34_000
    T_wiggly_end: int = 38_000

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

    @property
    def remap_window(self) -> tuple[int, int]:
        """Remapping window for backward compatibility.

        Returns
        -------
        tuple[int, int]
            (T_remap_start, T_remap_end)
        """
        return (self.T_remap_start, self.T_remap_end)

    def __post_init__(self) -> None:
        """Validate the timeline and initialize ``pf_centers`` if not provided.

        The 10 ``T_*`` fields are phase-boundary indices that must be
        strictly increasing — ``run_figure03_simulation`` builds each
        phase as ``T_next - T_prev`` and a non-monotonic timeline would
        yield a negative phase length, which ``np.arange``/``np.zeros``
        silently turn into an empty phase, shifting every later misfit
        window. Catch that here at construction rather than as a
        misaligned figure downstream.
        """
        timeline = [
            self.T_remap_start,
            self.T_remap_end,
            self.T_recovery1_end,
            self.T_hist_dep_end,
            self.T_recovery2_end,
            self.T_drift_end,
            self.T_recovery3_end,
            self.T_wide_dynamics_end,
            self.T_recovery4_end,
            self.T_wiggly_end,
        ]
        if any(later <= earlier for earlier, later in zip(timeline, timeline[1:], strict=False)):
            raise ValueError(
                f"DecodeParams timeline boundaries must be strictly increasing; got {timeline}"
            )

        if self.pf_centers is None:
            self.pf_centers = np.arange(self.xs_min, self.xs_max + 1, 10, dtype=float)


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
        likelihood). Used by the remap misfit (remapped place fields) and
        the wiggly-flat-likelihood misfit.
    diagnostic_rates : np.ndarray, shape (n_bins, n_cells), optional
        Replaces the rate table the *diagnostics* judge spikes against.
        Leave ``None`` for a misfit that perturbs the world the decoder
        observes but not the decoder's own model — remapping: the
        diagnostic should still reference the model's intended Gaussian
        place fields, so it correctly flags the mismatch. Set it (equal
        to ``decoder_rates``) only for a misfit that redefines what the
        decoder's model *is* — the wiggly-flat-likelihood phase.

    Raises
    ------
    ValueError
        If ``start >= end`` or a supplied rate table contains negative or
        non-finite entries.

    Examples
    --------
    >>> import numpy as np
    >>> rates = np.full((5, 3), 0.1)
    >>> w = MisfitWindow(10, 20, decoder_rates=rates, diagnostic_rates=rates)
    >>> w.start, w.end
    (10, 20)
    """

    start: int
    end: int
    transition_matrix: NDArray[np.floating] | None = None
    decoder_rates: NDArray[np.floating] | None = None
    diagnostic_rates: NDArray[np.floating] | None = None

    def __post_init__(self) -> None:
        """Validate the window bounds and any supplied rate tables."""
        if self.start >= self.end:
            raise ValueError(f"MisfitWindow requires start < end, got ({self.start}, {self.end})")
        # A negative or non-finite rate table would become NaN once it
        # reaches ``poisson.pmf`` and propagate silently through the
        # posterior — reject it at construction.
        for name in ("decoder_rates", "diagnostic_rates"):
            table = getattr(self, name)
            if table is not None and not (np.all(np.isfinite(table)) and np.all(table >= 0.0)):
                raise ValueError(f"MisfitWindow.{name} must be finite and non-negative everywhere")


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

    This matches the MATLAB implementation where:
    ``L = poisspdf(spikes(t, j), normpdf(xs, pfc(1), pfw) * .02)``
    uses pfc(1) (cell 1's center) instead of pfc(10) for cell 10.

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
) -> dict[str, NDArray[np.floating] | NDArray[np.intp]]:
    """Run the Bayesian filter with per-time, per-cell diagnostics.

    This function implements a Bayesian decoder for position from neural spikes,
    computing diagnostic metrics at each timestep to assess model goodness-of-fit.

    **Algorithm**:
    1. Initialize with flat prior at t=0
    2. For each timestep t:
       a. Predict: prior = post[t-1] @ transition_matrix
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
        Decoder-side misfit windows — remapping, wide-dynamics noise,
        wiggly-flat likelihood. Each :class:`MisfitWindow` swaps the
        transition matrix and/or the per-cell rate table for its
        interval. Defaults to an empty schedule: a clean decode with no
        misfits (the real-data decoding case).
    rng : np.random.Generator | None, optional
        Random number generator (reserved for future use).

    Returns
    -------
    results : dict[str, NDArray]
        Dictionary containing:
        - 'posterior' : np.ndarray, shape (n_time, n_bins)
            Posterior distribution at each timestep.
        - 'predictive' : np.ndarray, shape (n_time, n_bins)
            One-step predictive distribution p(x_t | y_{1:t-1}) at each timestep.
            At t=0, equals the flat prior.
        - 'likelihood' : np.ndarray, shape (n_time, n_bins)
            Normalized combined likelihood from all cells at each timestep.
            At t=0, equals a flat distribution.
        - 'hpd_overlap' : np.ndarray, shape (n_time, n_cells)
            HPD overlap between prior and each cell's likelihood.
            NaN at t=0 and when cell j has no spike at timestep t.
        - 'kl_divergence' : np.ndarray, shape (n_time, n_cells)
            KL divergence from prior to each cell's likelihood.
            NaN at t=0 and when cell j has no spike at timestep t.
        - 'spike_prob' : np.ndarray, shape (n_time, n_cells)
            Cumulative probability mass for cells with contribution <= cell j.
            NaN at t=0 and when cell j has no spike at timestep t.
        - 'spike_likelihood' : np.ndarray, shape (n_time, n_bins)
            Combined likelihood from only spiking cells at each timestep.
            NaN at timesteps with no spikes. Uses decoder's (remapped) rates.
        - 'per_spike_likelihood' : np.ndarray, shape (n_spikes, n_bins)
            Normalized likelihood for each individual spike event, using the
            decoder's actual place field centers (remapped during remap window).
        - 'spike_time_ind' : np.ndarray, shape (n_spikes,)
            Time index for each spike event (excludes t=0). Count > 1 in a
            bin expands to that many repeated events.
        - 'spike_cell_ind' : np.ndarray, shape (n_spikes,)
            Cell index for each spike event.
        - 'event_time_ind' : np.ndarray, shape (n_spikes,)
            Alias of 'spike_time_ind' (the per-event time index).
        - 'event_cell_ind' : np.ndarray, shape (n_spikes,)
            Alias of 'spike_cell_ind' (the per-event cell index).
        - 'event_hpd_overlap' : np.ndarray, shape (n_spikes,)
            Per-spike-event HPD overlap (the dense 'hpd_overlap' matrix
            scattered to one value per event).
        - 'event_kl_divergence' : np.ndarray, shape (n_spikes,)
            Per-spike-event KL divergence.
        - 'event_spike_prob' : np.ndarray, shape (n_spikes,)
            Per-spike-event spike probability.

    Notes
    -----
    Invalid misfit configurations (overlapping windows, ``start >= end``,
    negative/non-finite rate tables) are rejected when the
    :class:`MisfitSchedule` / :class:`MisfitWindow` is *constructed*, not
    here.

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
    >>> results['posterior'].shape
    (10, 21)
    >>> results['hpd_overlap'].shape  # Now per-cell
    (10, 3)
    >>> bool(np.all(np.isnan(results['hpd_overlap'][0])))  # t=0 has no prior
    True
    """
    n_time = spikes.shape[0]
    n_bins = xs.size

    # rng parameter reserved for future use
    _ = rng

    if misfit_schedule is None:
        misfit_schedule = MisfitSchedule()

    # Preallocate outputs
    posterior: NDArray[np.floating] = np.zeros((n_time, n_bins))
    predictive_posterior: NDArray[np.floating] = np.zeros((n_time, n_bins))  # p(x_t | y_{1:t-1})
    combined_likelihood_all: NDArray[np.floating] = np.zeros((n_time, n_bins))  # p(y_t | x_t)
    # Spike-only likelihood: product over only cells that fired (for display).
    # NaN at times with no spikes.
    spike_likelihood_all: NDArray[np.floating] = np.full((n_time, n_bins), np.nan)

    # t=0 (MATLAB used a flat prior at t=1)
    posterior[0] = normalize(np.ones(n_bins))
    predictive_posterior[0] = posterior[0]  # At t=0, predictive = prior
    combined_likelihood_all[0] = normalize(np.ones(n_bins))  # Flat at t=0

    # Baseline per-cell Poisson rate table. Used at every timestep not
    # covered by a misfit window whose ``decoder_rates`` is set, and as
    # the default diagnostic rate table.
    rates = placefield_rates(xs, pf_centers, pf_width, rate_scale)

    for t in range(1, n_time):
        window = misfit_schedule.window_at(t)

        # Predict — baseline transition unless the active misfit window
        # overrides it.
        current_transition = transition_matrix
        if window is not None and window.transition_matrix is not None:
            current_transition = window.transition_matrix
        prior = normalize(posterior[t - 1] @ current_transition)  # (n_bins,)
        predictive_posterior[t] = prior  # stored for p-value computation

        # Likelihood grid for this time's counts. The per-cell rate table
        # is the baseline Gaussian-PF table unless the active misfit
        # window overrides it (remapped or wiggly-flat table).
        rates_t = rates
        if window is not None and window.decoder_rates is not None:
            rates_t = window.decoder_rates
        likelihood = normalize(poisson.pmf(spikes[t][None, :], rates_t), axis=0)

        # Compute combined likelihood from all cells (product over cells)
        combined_likelihood = np.prod(likelihood, axis=1)  # (n_bins,)
        combined_likelihood_all[t] = normalize(combined_likelihood)  # Store normalized likelihood

        # Spike-only likelihood: product over only cells that fired.
        spiking_mask = spikes[t] > 0
        if np.any(spiking_mask):
            spike_likelihood_all[t] = normalize(np.prod(likelihood[:, spiking_mask], axis=1))

        # Posterior update with underflow protection
        # When prior-likelihood mismatch is extreme, the product can underflow to zero.
        # Fall back to uniform distribution to allow filter to recover.
        unnormalized_posterior = prior * combined_likelihood
        posterior_sum = np.sum(unnormalized_posterior)
        if posterior_sum < 1e-300:  # Numerical underflow detected
            posterior[t] = np.ones(n_bins) / n_bins  # Reset to uniform
        else:
            posterior[t] = unnormalized_posterior / posterior_sum

    # Find all spike events (excluding t=0 which has no valid prior). Count
    # matrices are expanded so a bin with count k contributes k spike events.
    spike_time_ind, spike_cell_ind = np.nonzero(spikes[1:])
    spike_counts_at_events = spikes[1:][spike_time_ind, spike_cell_ind].astype(np.intp)
    spike_time_ind = np.repeat(spike_time_ind, spike_counts_at_events)
    spike_cell_ind = np.repeat(spike_cell_ind, spike_counts_at_events)
    spike_time_ind = spike_time_ind + 1  # Adjust for offset from [1:]
    n_spikes = len(spike_time_ind)

    # Diagnostics. Each spike event is judged against a rate table: the
    # baseline Gaussian-PF ``rates`` unless it falls in a misfit window
    # whose ``diagnostic_rates`` is set. ``rates`` is the *model's*
    # intended likelihood — during remapping the decoder updates the
    # posterior with remapped fields but the diagnostic still references
    # the original fields (``diagnostic_rates`` is ``None`` for remap),
    # so the diagnostic correctly flags the mismatch. The wiggly-flat
    # phase redefines the model itself, so its window sets
    # ``diagnostic_rates`` and those events are judged against it.
    # ``compute_per_cell_diagnostics_from_rates`` takes a single rate
    # table, so group events by table, diagnose each group, and merge.
    diag_tables: list[NDArray[np.floating]] = [rates]
    event_group: NDArray[np.intp] = np.zeros(n_spikes, dtype=np.intp)
    for window in misfit_schedule.windows:
        if window.diagnostic_rates is None:
            continue
        in_window = (spike_time_ind >= window.start) & (spike_time_ind < window.end)
        if not np.any(in_window):
            continue
        diag_tables.append(window.diagnostic_rates)
        event_group[in_window] = len(diag_tables) - 1

    if len(diag_tables) == 1:
        diagnostics = compute_per_cell_diagnostics_from_rates(
            predictive_posterior,
            rates,
            spike_time_ind,
            spike_cell_ind,
            coverage=0.95,
        )
    else:
        per_group = [
            compute_per_cell_diagnostics_from_rates(
                predictive_posterior,
                table,
                spike_time_ind[event_group == group],
                spike_cell_ind[event_group == group],
                coverage=0.95,
                include_dense_matrices=False,
            )
            for group, table in enumerate(diag_tables)
        ]
        diagnostics = _merge_diagnostics(
            n_time=n_time,
            n_cells=rates.shape[1],
            spike_time_ind=spike_time_ind,
            spike_cell_ind=spike_cell_ind,
            event_group=event_group,
            per_group=per_group,
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

    return {
        "posterior": posterior,
        "predictive": predictive_posterior,
        "likelihood": combined_likelihood_all,
        "spike_likelihood": spike_likelihood_all,
        "hpd_overlap": diagnostics["hpd_overlap"],
        "kl_divergence": diagnostics["kl_divergence"],
        "spike_prob": diagnostics["spike_prob"],
        "per_spike_likelihood": decoder_per_spike_lik,
        "spike_time_ind": diagnostics["spike_time_ind"],
        "spike_cell_ind": diagnostics["spike_cell_ind"],
        "event_time_ind": diagnostics["event_time_ind"],
        "event_cell_ind": diagnostics["event_cell_ind"],
        "event_hpd_overlap": diagnostics["event_hpd_overlap"],
        "event_kl_divergence": diagnostics["event_kl_divergence"],
        "event_spike_prob": diagnostics["event_spike_prob"],
    }


# -----------------------------
# Per-cell diagnostics (shared logic)
# -----------------------------


def _merge_diagnostics(
    *,
    n_time: int,
    n_cells: int,
    spike_time_ind: NDArray[np.intp],
    spike_cell_ind: NDArray[np.intp],
    event_group: NDArray[np.intp],
    per_group: list[dict[str, NDArray[np.floating] | NDArray[np.intp]]],
) -> dict[str, NDArray[np.floating] | NDArray[np.intp]]:
    """Merge per-event diagnostic batches computed against different rate
    tables into a single result dict.

    The merged dict has the keys of the
    ``include_dense_matrices=True`` result of
    :func:`compute_per_cell_diagnostics_from_rates` *except*
    ``per_spike_likelihood`` — that key is supplied separately by the
    caller (``decode_and_diagnostics`` builds its own per-spike
    likelihood from the decoder's rate table).

    Used when the diagnostic rate table differs across the timeline
    because the :class:`MisfitSchedule` has windows with
    ``diagnostic_rates`` set (the wiggly-flat-likelihood phase).

    Parameters
    ----------
    event_group : np.ndarray, shape (n_spikes,)
        For each spike event, the index of the rate-table group it was
        diagnosed under. ``per_group[event_group[i]]`` is the diagnostic
        batch containing event ``i``.
    per_group : list of dict
        One ``compute_per_cell_diagnostics_from_rates(...,
        include_dense_matrices=False)`` result per group, group ``g``
        computed on the events with ``event_group == g`` (boolean-mask
        indexing preserves order, so scattering back via the same mask
        reassembles events in their original order).
    """
    n_spikes = spike_time_ind.shape[0]
    if event_group.shape[0] != n_spikes:
        raise ValueError(f"event_group length {event_group.shape[0]} != n_spikes {n_spikes}")

    event_hpd_overlap = np.empty(n_spikes)
    event_kl_divergence = np.empty(n_spikes)
    event_spike_prob = np.empty(n_spikes)
    for group, diag in enumerate(per_group):
        sel = event_group == group
        expected = int(sel.sum())
        got = diag["event_hpd_overlap"].shape[0]
        if got != expected:
            raise ValueError(f"group {group} has {got} events but the mask expects {expected}")
        event_hpd_overlap[sel] = diag["event_hpd_overlap"]
        event_kl_divergence[sel] = diag["event_kl_divergence"]
        event_spike_prob[sel] = diag["event_spike_prob"]

    # Scatter per-event values into the dense (n_time, n_cells) matrices.
    # A bin with count k > 1 contributes k repeated (time, cell) index
    # pairs, so the dense matrix keeps the last writer for that cell —
    # safe here because per-event diagnostics for repeated same-(t, c)
    # spikes are identical by construction (same predictive row, same
    # cell rate). This mirrors the dense-matrix behavior of
    # ``compute_per_cell_diagnostics_from_rates``.
    hpd_overlap = np.full((n_time, n_cells), np.nan)
    kl_divergence = np.full((n_time, n_cells), np.nan)
    spike_prob = np.full((n_time, n_cells), np.nan)
    hpd_overlap[spike_time_ind, spike_cell_ind] = event_hpd_overlap
    kl_divergence[spike_time_ind, spike_cell_ind] = event_kl_divergence
    spike_prob[spike_time_ind, spike_cell_ind] = event_spike_prob

    return {
        "spike_time_ind": spike_time_ind,
        "spike_cell_ind": spike_cell_ind,
        "event_time_ind": spike_time_ind,
        "event_cell_ind": spike_cell_ind,
        "event_hpd_overlap": event_hpd_overlap,
        "event_kl_divergence": event_kl_divergence,
        "event_spike_prob": event_spike_prob,
        "hpd_overlap": hpd_overlap,
        "kl_divergence": kl_divergence,
        "spike_prob": spike_prob,
    }


def compute_per_cell_diagnostics_from_rates(
    predictive_posterior: NDArray[np.floating],
    rates: NDArray[np.floating],
    spike_time_ind: NDArray[np.intp],
    spike_cell_ind: NDArray[np.intp],
    coverage: float = 0.95,
    include_dense_matrices: bool = True,
) -> dict[str, NDArray[np.floating] | NDArray[np.intp]]:
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
        # scratch array.
        #
        # ``spike_prob`` was previously vectorized as
        # ``spike_prob_rank(pred[unique_times], cell_fraction)`` ⇒ a
        # ``(n_unique_times, n_cells, n_cells)`` mask, which is
        # ``709 K × 200²`` ≈ 28 GB on the real W-track session.
        # Inline a per-event rank computation here instead — same
        # math (``sum_i contrib[i] where contrib[i] <= contrib[j]``)
        # but only over the rows we actually need. Cost: ~3× the
        # rank work versus the unique-time dedup, but the memory
        # ceiling drops to ``B × n_cells``.
        cell_fraction_per_bin = normalize(rates, axis=1)  # (n_bins, n_cells)
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

    result: dict[str, NDArray[np.floating] | NDArray[np.intp]] = {
        "spike_time_ind": spike_time_ind,
        "spike_cell_ind": spike_cell_ind,
        "event_time_ind": spike_time_ind,
        "event_cell_ind": spike_cell_ind,
        "event_hpd_overlap": event_hpd_overlap,
        "event_kl_divergence": event_kl_divergence,
        "event_spike_prob": event_spike_prob,
    }
    if hpd_overlap is not None:
        assert kl_divergence is not None
        assert spike_prob is not None
        assert per_spike_likelihood is not None
        result["hpd_overlap"] = hpd_overlap
        result["kl_divergence"] = kl_divergence
        result["spike_prob"] = spike_prob
        result["per_spike_likelihood"] = per_spike_likelihood
    return result


# -----------------------------
# Thresholds & transforms
# -----------------------------


@dataclass
class Thresholds:
    """Threshold values for diagnostic metrics.

    Thresholds are computed from baseline period across ALL cells (flattened).
    This matches the MATLAB implementation where quantiles are computed over
    the full (n_time * n_cells) array of metric values.

    Parameters
    ----------
    hpd_overlap : float
        HPD overlap threshold (1st percentile across all cells).
        Lower values indicate worse fit.
    kl_divergence : float
        KL divergence threshold (99th percentile across all cells).
        Higher values indicate worse fit.
    spike_prob : float
        Spike probability threshold (fixed at 0.05 per MATLAB).
        Lower values indicate misfit.

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


def compute_thresholds(
    metrics: Mapping[str, NDArray[np.floating] | NDArray[np.intp]],
    baseline_end: int | None = None,
) -> Thresholds:
    """Compute threshold values from baseline period.

    Thresholds are computed across ALL cells (flattened), matching MATLAB.
    The (n_time, n_cells) arrays are flattened to 1D before computing quantiles:
    - HPD overlap threshold: 1st percentile (low values indicate misfit)
    - KL divergence threshold: 99th percentile (high values indicate misfit)
    - spike_prob threshold: fixed at 0.05 (per MATLAB implementation)

    Parameters
    ----------
    metrics : Mapping[str, NDArray]
        Mapping containing diagnostic metrics (accepts the full
        ``decode_and_diagnostics`` result; only these three keys are read):
        - 'hpd_overlap' : np.ndarray, shape (n_time, n_cells)
        - 'kl_divergence' : np.ndarray, shape (n_time, n_cells)
        - 'spike_prob' : np.ndarray, shape (n_time, n_cells)
    baseline_end : int or None, default None
        Index marking end of baseline period (exclusive).
        If None, uses all time points for threshold computation.

    Returns
    -------
    thresholds : Thresholds
        Threshold values for each diagnostic metric.

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
    >>> thresholds.spike_prob  # Fixed at 0.05 per MATLAB
    0.05
    """
    # Flatten (n_time, n_cells) to 1D for quantile computation. ``np.nanquantile``
    # returns ``np.floating``; cast to plain ``float`` to match the ``Thresholds``
    # dataclass signature.
    hpd_baseline = metrics["hpd_overlap"][:baseline_end].ravel()
    hpd_overlap_threshold = float(np.nanquantile(hpd_baseline, 0.01))

    kl_baseline = metrics["kl_divergence"][:baseline_end].ravel()
    kl_divergence_threshold = float(np.nanquantile(kl_baseline, 0.99))

    # spike_prob threshold is fixed at 0.05 per MATLAB
    spike_prob_threshold = 0.05

    return Thresholds(
        hpd_overlap=hpd_overlap_threshold,
        kl_divergence=kl_divergence_threshold,
        spike_prob=spike_prob_threshold,
    )


@dataclass
class Transformed:
    """Transformed diagnostic metrics and thresholds.

    Transformations are applied to diagnostic metrics to improve visualization
    and interpretability (e.g., log-transform for better dynamic range).

    All metrics have shape (n_time, n_cells).

    Parameters
    ----------
    hpd_overlap : np.ndarray, shape (n_time, n_cells)
        Transformed HPD overlap values: -log10(HPDO + eps1).
    kl_divergence : np.ndarray, shape (n_time, n_cells)
        Transformed KL divergence values: sqrt(KL).
    spike_prob : np.ndarray, shape (n_time, n_cells)
        Transformed spike probability values: -log10(spikeProb + eps2).
    hpd_overlap_threshold : float
        Transformed HPD overlap threshold.
    kl_divergence_threshold : float
        Transformed KL divergence threshold.
    spike_prob_threshold : float
        Transformed spike probability threshold.

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


def transform_metrics(
    metrics: dict[str, NDArray[np.floating]],
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
    hpd_overlap_transformed = -np.log10(np.maximum(metrics["hpd_overlap"] + eps1, 1e-10))
    kl_divergence_transformed = np.sqrt(metrics["kl_divergence"])
    spike_prob_transformed = -np.log10(np.maximum(metrics["spike_prob"] + eps2, 1e-10))

    return Transformed(
        hpd_overlap=hpd_overlap_transformed,
        kl_divergence=kl_divergence_transformed,
        spike_prob=spike_prob_transformed,
        hpd_overlap_threshold=-np.log10(max(thresholds.hpd_overlap + eps1, 1e-10)),
        kl_divergence_threshold=np.sqrt(thresholds.kl_divergence),
        spike_prob_threshold=-np.log10(max(thresholds.spike_prob + eps2, 1e-10)),
    )
