"""Analysis functions for state space model diagnostics.

This module contains the core analysis logic for running Bayesian decoders and computing
diagnostic metrics (KL divergence, HPD overlap, spike probability) to assess model
goodness-of-fit.

**Key Components**:
- **DecodeParams**: Parameter container for decoding simulations
- **likelihood_grid_for_counts**: Compute Poisson likelihood for spike counts
- **apply_remap_for_likelihoods**: Apply cell identity remapping
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
    >>> # Run decoder
    >>> results = decode_and_diagnostics(
    ...     spikes, xs, transition_matrix, params.pf_centers,
    ...     params.pf_width, params.rate_scale, params.remap_window, params.remap_from_to
    ... )
    >>> results.keys()
    dict_keys(['post', 'HPDO', 'KL', 'spikeProb'])
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import statespacecheck as ssc
from numpy.typing import NDArray
from scipy.stats import poisson

from statespacecheck_paper.predictive_check import (
    sample_positions_from_posterior,
)
from statespacecheck_paper.simulation import normalize, placefield_rates, safe_log, spike_prob_rank

# -----------------------------
# Data containers
# -----------------------------


@dataclass
class DecodeParams:
    """Parameters for decoding simulation.

    This dataclass contains all parameters needed to run a state space model decoding
    simulation with various model misfits (remapping, flat firing, fast/slow movement).

    **Timeline Structure**:
    - 0-6k: Clean baseline
    - 6k-10k: Remapping misfit (4k timesteps)
    - 10k-14k: Clean recovery (4k timesteps)
    - 14k-16k: Flat firing misfit (2k timesteps)
    - 16k-20k: Clean recovery (4k timesteps)
    - 20k-24k: Fast movement misfit (4k timesteps)
    - 24k-28k: Clean recovery (4k timesteps)
    - 28k-32k: Slow movement misfit (4k timesteps)

    Parameters
    ----------
    T_remap_start : int, default 6_000
        Start of remapping misfit period.
    T_remap_end : int, default 10_000
        End of remapping misfit period.
    T_recovery1_end : int, default 14_000
        End of first recovery period.
    T_flat_end : int, default 16_000
        End of flat firing misfit period.
    T_recovery2_end : int, default 20_000
        End of second recovery period.
    T_fast_end : int, default 24_000
        End of fast movement misfit period.
    T_recovery3_end : int, default 28_000
        End of third recovery period.
    T_slow_end : int, default 32_000
        End of slow movement misfit period.
    sigx_pred : float, default 0.5
        Decoder's dynamics standard deviation (baseline).
    sigx_pred_fast_phase : float, default 0.1
        Narrow decoder for fast phase (5x too narrow!).
    sigx_pred_slow_phase : float, default 20.0
        Inflated decoder for slow phase (40x too broad!).
    sigx_true_fast : float, default 10.0
        True dynamics std in fast phase (100x faster than decoder!).
    sigx_true_slow : float, default 0.0
        True dynamics std in slow phase (completely stationary!).
    xs_min : int, default 0
        Minimum position value.
    xs_max : int, default 100
        Maximum position value.
    xs_step : int, default 1
        Step size for position grid.
    pf_width : float, default 10.0
        Place field width (std of Gaussian tuning curves).
    pf_centers : NDArray[np.floating] | None, default None
        Place field center positions. If None, initialized in __post_init__.
    rate_scale : float, default 0.02
        Spike rate scaling factor (matches MATLAB normpdf * 0.02).
    base_seed : int, default 1
        Base random seed for reproducibility.
    remap_from_to : tuple of tuples, default ((0, 5), (1, 6), ...)
        Remapping specification: ((src1, dst1), (src2, dst2), ...).
        Default implements +50cm circular shift for all 11 cells.

    Examples
    --------
    >>> params = DecodeParams()
    >>> params.remap_window
    (6000, 10000)
    >>> params.pf_centers
    array([  0.,  10.,  20.,  30.,  40.,  50.,  60.,  70.,  80.,  90., 100.])

    >>> # Custom parameters
    >>> params = DecodeParams(T_remap_start=1000, T_remap_end=2000, sigx_pred=1.0)
    >>> params.sigx_pred
    1.0
    """

    # Timeline with recovery periods between misfits
    T_remap_start: int = 6_000
    T_remap_end: int = 10_000
    T_recovery1_end: int = 14_000
    T_flat_end: int = 16_000
    T_recovery2_end: int = 20_000
    T_fast_end: int = 24_000
    T_recovery3_end: int = 28_000
    T_slow_end: int = 32_000
    sigx_pred: float = 0.5  # decoder's dynamics std (baseline)
    sigx_pred_fast_phase: float = 0.1  # narrow decoder for fast phase (5x too narrow!)
    sigx_pred_slow_phase: float = 20.0  # inflated decoder for slow phase (40x too broad!)
    sigx_true_fast: float = 10.0  # true dynamics std in fast phase (100x faster than decoder!)
    sigx_true_slow: float = 0.0  # true dynamics std in slow phase (completely stationary!)
    xs_min: int = 0
    xs_max: int = 100
    xs_step: int = 1
    pf_width: float = 10.0  # Place field width (std of Gaussian tuning curves)
    pf_centers: NDArray[np.floating] | None = None  # set in __post_init__
    rate_scale: float = (
        20.0  # High spike rate to avoid ties in predictive checks (MATLAB used 0.02)
    )
    base_seed: int = 1
    remap_from_to: tuple[tuple[int, int], ...] | tuple[int, int] = (
        (0, 5),  # Position 0 → 50 (shift +50cm)
        (1, 6),  # Position 10 → 60
        (2, 7),  # Position 20 → 70
        (3, 8),  # Position 30 → 80
        (4, 9),  # Position 40 → 90
        (5, 10),  # Position 50 → 100
        (6, 0),  # Position 60 → 0
        (7, 1),  # Position 70 → 10
        (8, 2),  # Position 80 → 20
        (9, 3),  # Position 90 → 30
        (10, 4),  # Position 100 → 40
    )  # Remap ALL 11 cells with +50cm circular shift

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
        """Initialize pf_centers if not provided."""
        if self.pf_centers is None:
            self.pf_centers = np.arange(self.xs_min, self.xs_max + 1, 10, dtype=float)


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


def apply_remap_for_likelihoods(
    likelihood: NDArray[np.floating],
    remap_from_to: tuple[tuple[int, int], ...] | tuple[int, int],
    active: bool,
) -> NDArray[np.floating]:
    """Optionally replace one or more columns by others (remapping cell identities).

    This function simulates model misfit by remapping cell identities - replacing
    the likelihood column for one cell with the likelihood column from another cell.
    This mimics a situation where the decoder has incorrect place field assignments.

    Parameters
    ----------
    likelihood : np.ndarray, shape (n_bins, n_cells)
        Likelihood grid with one column per cell.
    remap_from_to : tuple of tuples or tuple of ints
        Remapping specification. Can be:
        - Single remapping: (src, dst) - replace column src with column dst
        - Multiple remappings: ((src1, dst1), (src2, dst2), ...) - apply all remappings
    active : bool
        If False, returns likelihood unchanged. If True, applies remapping.

    Returns
    -------
    likelihood : np.ndarray, shape (n_bins, n_cells)
        Modified likelihood grid. If active=True, returns a copy with remapping applied.
        If active=False, returns the original array (not a copy).

    Examples
    --------
    >>> import numpy as np
    >>> likelihood = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    >>> # Single remapping: cell 0 becomes like cell 2
    >>> result = apply_remap_for_likelihoods(likelihood, (0, 2), active=True)
    >>> result
    array([[3., 2., 3.],
           [6., 5., 6.]])

    >>> # Multiple remappings
    >>> result = apply_remap_for_likelihoods(likelihood, ((0, 1), (2, 1)), active=True)
    >>> result
    array([[2., 2., 2.],
           [5., 5., 5.]])

    >>> # Inactive (no remapping)
    >>> result = apply_remap_for_likelihoods(likelihood, (0, 2), active=False)
    >>> np.array_equal(result, likelihood)
    True
    """
    if not active:
        return likelihood
    likelihood = likelihood.copy()

    # Normalize to iterable of pairs (handles both single and multiple remappings)
    if len(remap_from_to) == 2 and isinstance(remap_from_to[0], int):
        # Single remapping: (src, dst)
        src, dst = remap_from_to
        likelihood[:, src] = likelihood[:, dst]
    else:
        # Multiple remappings: ((src1, dst1), (src2, dst2), ...)
        for src, dst in remap_from_to:
            likelihood[:, src] = likelihood[:, dst]

    return likelihood


def _window_or_never(window: tuple[int, int] | None, n_time: int) -> tuple[int, int]:
    """Return window bounds or impossibly late bounds if None.

    Parameters
    ----------
    window : tuple[int, int] | None
        Window (start, end) or None.
    n_time : int
        Total number of time points.

    Returns
    -------
    start, end : tuple[int, int]
        Window bounds or (n_time + 1, n_time + 1) if window is None.
    """
    return window if window else (n_time + 1, n_time + 1)


def decode_and_diagnostics(
    spikes: NDArray[np.int_],
    xs: NDArray[np.floating],
    transition_matrix: NDArray[np.floating],
    pf_centers: NDArray[np.floating],
    pf_width: float,
    rate_scale: float,
    remap_window: tuple[int, int],
    remap_from_to: tuple[tuple[int, int], ...] | tuple[int, int],
    rng: np.random.Generator | None = None,
    transition_matrix_narrow: NDArray[np.floating] | None = None,
    narrow_window: tuple[int, int] | None = None,
    transition_matrix_inflated: NDArray[np.floating] | None = None,
    inflate_window: tuple[int, int] | None = None,
    n_samples: int = 1000,
    dt: float = 1.0,
) -> dict[str, NDArray[np.floating]]:
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
    remap_window : tuple[int, int]
        Time window (start, end) where cell remapping is active.
    remap_from_to : tuple of tuples or tuple of ints
        Remapping specification (see apply_remap_for_likelihoods).
    rng : np.random.Generator | None, optional
        Random number generator (not currently used).
    transition_matrix_narrow : np.ndarray, shape (n_bins, n_bins), optional
        Alternative transition matrix for narrow window (fast movement misfit).
    narrow_window : tuple[int, int], optional
        Time window (start, end) for narrow transition matrix.
    transition_matrix_inflated : np.ndarray, shape (n_bins, n_bins), optional
        Alternative transition matrix for inflated window (slow movement misfit).
    inflate_window : tuple[int, int], optional
        Time window (start, end) for inflated transition matrix.
    n_samples : int, default 1000
        Number of Monte Carlo samples for p-value computation.
    dt : float, default 1.0
        Time bin duration (dimensionless for this simulation).
        Note: simulation.py uses norm.pdf values directly as Poisson rates,
        implying dt=1. Do not change unless simulation code is updated.

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
        - 'hpd_overlap' : np.ndarray, shape (n_time,)
            HPD overlap between prior and combined likelihood.
            NaN at t=0 (no prior available).
        - 'kl_divergence' : np.ndarray, shape (n_time,)
            KL divergence from prior to combined likelihood.
            NaN at t=0 (no prior available).
        - 'spike_prob' : np.ndarray, shape (n_time, n_cells)
            Spike probability for each cell at each timestep.
            NaN for cells with zero spikes (no observation to evaluate).
        - 'p_values' : np.ndarray, shape (n_time,)
            Posterior predictive p-values at each timestep.
            NaN at t=0 (no prior available).

    Examples
    --------
    >>> import numpy as np
    >>> from statespacecheck_paper.simulation import gaussian_transition_matrix
    >>> # Set up small problem
    >>> n_time, n_cells, n_bins = 10, 3, 21
    >>> spikes = np.random.poisson(1.0, size=(n_time, n_cells))
    >>> xs = np.linspace(0, 100, n_bins)
    >>> transition_matrix = gaussian_transition_matrix(xs, sigma=0.5)
    >>> pf_centers = np.array([25.0, 50.0, 75.0])
    >>> pf_width = 5.0
    >>> rate_scale = 0.1
    >>> remap_window = (5, 7)
    >>> remap_from_to = (0, 1)
    >>> # Run decoder
    >>> results = decode_and_diagnostics(
    ...     spikes, xs, transition_matrix, pf_centers, pf_width, rate_scale,
    ...     remap_window, remap_from_to
    ... )
    >>> results['post'].shape
    (10, 21)
    >>> results['HPDO'].shape
    (10,)
    >>> np.isnan(results['HPDO'][0])  # t=0 has no prior
    True
    """
    n_time, n_cells = spikes.shape
    n_bins = xs.size

    # Initialize RNG if not provided
    if rng is None:
        rng = np.random.default_rng()

    # Preallocate outputs (NaN for unavailable values)
    posterior: NDArray[np.floating] = np.zeros((n_time, n_bins))
    hpd_overlap: NDArray[np.floating] = np.full(n_time, np.nan)  # Single value per timestep
    kl_divergence: NDArray[np.floating] = np.full(n_time, np.nan)  # Single value per timestep
    spike_prob: NDArray[np.floating] = np.full(
        (n_time, n_cells), np.nan
    )  # Keep per-cell for this metric

    # Storage for predictive check computation
    predictive_posterior: NDArray[np.floating] = np.zeros((n_time, n_bins))  # p(x_t | y_{1:t-1})
    log_likelihood_grid: NDArray[np.floating] = np.zeros((n_time, n_bins))  # log p(y_t | x_t)
    combined_likelihood_all: NDArray[np.floating] = np.zeros((n_time, n_bins))  # p(y_t | x_t)

    # t=0 (MATLAB used a flat prior at t=1)
    posterior[0] = normalize(np.ones(n_bins))
    predictive_posterior[0] = posterior[0]  # At t=0, predictive = prior
    combined_likelihood_all[0] = normalize(np.ones(n_bins))  # Flat at t=0

    # Compute log_likelihood_grid[0] for p-value computation
    rates_t0 = placefield_rates(xs, pf_centers, pf_width, rate_scale)  # (n_bins, n_cells)
    expected_counts_t0 = rates_t0 * dt  # (n_bins, n_cells)
    spike_counts_t0 = spikes[0][np.newaxis, :, np.newaxis]  # (1, n_cells, 1)
    expected_counts_t0_reshaped = expected_counts_t0.T[np.newaxis, :, :]  # (1, n_cells, n_bins)
    log_pmf_t0 = poisson.logpmf(spike_counts_t0, expected_counts_t0_reshaped)
    log_likelihood_grid[0] = log_pmf_t0.sum(axis=1).squeeze()  # (n_bins,)

    rate_grid_all = placefield_rates(xs, pf_centers, pf_width, rate_scale)  # (n_bins, n_cells)
    # Normalize to get per-bin cell-fractions (rows sum to 1)
    cell_fraction_per_bin = normalize(rate_grid_all, axis=1)

    start_r, end_r = remap_window
    start_narrow, end_narrow = _window_or_never(narrow_window, n_time)
    start_inflate, end_inflate = _window_or_never(inflate_window, n_time)

    for t in range(1, n_time):
        # Select transition matrix based on which window we're in
        if transition_matrix_narrow is not None and start_narrow <= t <= end_narrow:
            current_transition = transition_matrix_narrow
        elif transition_matrix_inflated is not None and start_inflate <= t <= end_inflate:
            current_transition = transition_matrix_inflated
        else:
            current_transition = transition_matrix

        # Predict (prior from state dynamics)
        prior = normalize(posterior[t - 1] @ current_transition)  # (n_bins,)

        # Store predictive posterior for p-value computation
        predictive_posterior[t] = prior

        # Likelihood grid for this time's counts (vectorized over bins & cells)
        # Note: likelihood_grid_for_counts returns NORMALIZED likelihoods per cell
        likelihood = likelihood_grid_for_counts(xs, pf_centers, pf_width, rate_scale, spikes[t])
        # Optional remap (imitating MATLAB's j==10 uses field of j==1 in a window)
        active_remap = start_r <= t <= end_r
        likelihood = apply_remap_for_likelihoods(likelihood, remap_from_to, active_remap)

        # Compute combined likelihood from all cells (product over cells)
        combined_likelihood = np.prod(likelihood, axis=1)  # (n_bins,)
        combined_likelihood_all[t] = normalize(combined_likelihood)  # Store normalized likelihood

        # For p-value computation, we need UNNORMALIZED log likelihoods
        # Compute raw Poisson log likelihood with dt to match predictive_check.py
        # Note: Simulation uses rates directly (implicit dt=1), so we pass dt=1.0 here
        rates = placefield_rates(xs, pf_centers, pf_width, rate_scale)  # (n_bins, n_cells)
        expected_counts = rates * dt  # (n_bins, n_cells)
        # Compute log PMF for each cell at each position
        spike_counts_reshaped = spikes[t][np.newaxis, :, np.newaxis]  # (1, n_cells, 1)
        expected_counts_reshaped = expected_counts.T[np.newaxis, :, :]  # (1, n_cells, n_bins)
        log_pmf_per_cell = poisson.logpmf(
            spike_counts_reshaped, expected_counts_reshaped
        )  # (1, n_cells, n_bins)
        # Sum over cells to get total log likelihood
        log_likelihood_grid[t] = log_pmf_per_cell.sum(axis=1).squeeze()  # (n_bins,)

        # Compute diagnostics only at spike times (skip if no spikes at this timestep)
        # This avoids evaluating metrics when there's no observation to assess
        if spikes[t].sum() > 0:
            # Compare one-step prediction (prior) with combined likelihood (observation model)
            prior_t = prior[np.newaxis, :]  # (1, n_bins)
            # Use log-space for numerically stable combined likelihood
            # (the product of normalized per-cell likelihoods underflows)
            log_lik = log_likelihood_grid[t]
            log_lik_shifted = log_lik - np.max(log_lik)  # Shift for numerical stability
            combined_likelihood_stable = np.exp(log_lik_shifted)
            combined_likelihood_stable = combined_likelihood_stable / np.sum(
                combined_likelihood_stable
            )
            combined_likelihood_t = combined_likelihood_stable[np.newaxis, :]  # (1, n_bins)

            # HPD overlap between prior and combined likelihood
            hpd_overlap_t: NDArray[np.floating] = ssc.hpd_overlap(
                prior_t, combined_likelihood_t, coverage=0.95
            )
            hpd_overlap[t] = hpd_overlap_t[0]

            # KL divergence between prior and combined likelihood
            kl_divergence_t: NDArray[np.floating] = ssc.kl_divergence(
                prior_t, combined_likelihood_t
            )
            kl_divergence[t] = kl_divergence_t[0]
        # else: hpd_overlap[t] and kl_divergence[t] remain NaN (initialized above)

        # Posterior update with underflow protection
        # When prior-likelihood mismatch is extreme, the product can underflow to zero.
        # Fall back to uniform distribution to allow filter to recover.
        unnormalized_posterior = prior * combined_likelihood
        posterior_sum = np.sum(unnormalized_posterior)
        if posterior_sum < 1e-300:  # Numerical underflow detected
            posterior[t] = np.ones(n_bins) / n_bins  # Reset to uniform
        else:
            posterior[t] = unnormalized_posterior / posterior_sum

        # spike_prob: cumulative probability mass for cells with low expected contribution
        spike_prob[t] = spike_prob_rank(prior, cell_fraction_per_bin)

    # Mask spike_prob for cells with zero spikes (match MATLAB: spikeProb(spikes == 0) = nan)
    # Note: HPD overlap and KL divergence are now per-timestep (not per-cell) since they compare
    # the combined likelihood with the prior, so we don't mask them
    spike_prob[spikes == 0] = np.nan

    # Compute posterior predictive p-values only at spike times
    # Following manuscript equation (210): D_k = Pr(log(f_pred(y_k)) >= log(f_pred(y_tilde_k)))
    # Identify timesteps with at least one spike (skip non-spike times)
    has_spikes = spikes.sum(axis=1) > 0  # (n_time,) boolean mask
    spike_time_indices = np.where(has_spikes)[0]  # indices of spike times
    n_spike_times = len(spike_time_indices)

    # Initialize p_values as NaN (non-spike times remain NaN)
    p_values: NDArray[np.floating] = np.full(n_time, np.nan)

    # Only compute p-values if there are spike times
    if n_spike_times > 0:
        # Build MODEL's place fields (may include remapping) for simulation
        # During remapping, simulated data should use the decoder's assumed (remapped) fields
        # Transpose from (n_bins, n_cells) to (n_cells, n_bins)
        place_fields_original = placefield_rates(xs, pf_centers, pf_width, rate_scale).T

        # Create remapped place fields for simulation during remap window
        place_fields_remapped = place_fields_original.copy()  # Default to original
        if remap_from_to is not None and start_r < end_r:
            # Apply same remapping that decoder uses: swap place field assignments
            place_fields_remapped = place_fields_original.copy()
            # Normalize remap_from_to to list of tuples (handles single tuple case)
            if len(remap_from_to) == 2 and isinstance(remap_from_to[0], int):
                remap_pairs = [(remap_from_to[0], remap_from_to[1])]
            else:
                remap_pairs = list(remap_from_to)
            for src_idx, dst_idx in remap_pairs:
                # Decoder thinks cell src_idx has place field of cell dst_idx
                place_fields_remapped[src_idx] = place_fields_original[dst_idx]
            # Note: We'll apply this time-dependently in the loop below

        # Extract spike-time-only arrays for efficient computation
        predictive_spike_times = predictive_posterior[spike_time_indices]
        log_likelihood_spike_times = log_likelihood_grid[spike_time_indices]

        # Compute observed log predictive density at spike times only
        observed_log_pred_spike = ssc.log_predictive_density(
            state_dist=predictive_spike_times,
            log_likelihood=log_likelihood_spike_times,
        )

        # Create sampler function for predictive_pvalue (spike times only)
        def sample_log_pred(n_samp: int) -> NDArray[np.floating]:
            """Sample log predictive densities for spike timesteps only."""
            simulated_log_pred = np.zeros((n_samp, n_spike_times))

            for i in range(n_samp):
                # 1. Sample positions from predictive posterior at spike times only
                position_indices = sample_positions_from_posterior(
                    predictive_spike_times.astype(np.float64), rng
                )  # (n_spike_times,)

                # 2. Generate synthetic spikes at spike times using MODEL's place fields
                spike_counts_sim = np.zeros((n_spike_times, n_cells), dtype=np.int64)
                for idx, t in enumerate(spike_time_indices):
                    # Use remapped fields during remap window, original otherwise
                    if remap_from_to is not None and start_r <= t <= end_r:
                        pf_t = place_fields_remapped
                    else:
                        pf_t = place_fields_original

                    # Generate spikes at this timestep using model's fields
                    pos_idx = position_indices[idx]
                    rates_at_pos = pf_t[:, pos_idx] * dt
                    spike_counts_sim[idx] = rng.poisson(rates_at_pos)

                # 3. Compute log likelihood at spike times using MODEL's place fields
                log_like_sim = np.zeros((n_spike_times, len(xs)))
                for idx, t in enumerate(spike_time_indices):
                    if remap_from_to is not None and start_r <= t <= end_r:
                        pf_t = place_fields_remapped
                    else:
                        pf_t = place_fields_original

                    # Expected counts: (n_cells, n_bins)
                    expected_counts_t = pf_t * dt
                    # Poisson log PMF: (n_cells, n_bins)
                    spike_counts_3d = spike_counts_sim[idx : idx + 1, :, np.newaxis]
                    expected_3d = expected_counts_t[np.newaxis, :, :]  # (1, n_cells, n_bins)
                    log_pmf = poisson.logpmf(spike_counts_3d, expected_3d)
                    log_like_sim[idx] = log_pmf.sum(axis=1).squeeze()

                # 4. Compute log predictive density for synthetic data
                simulated_log_pred[i] = ssc.log_predictive_density(
                    state_dist=predictive_spike_times,
                    log_likelihood=log_like_sim,
                )

            return simulated_log_pred

        # Compute p-values at spike times using statespacecheck
        p_values_spike: NDArray[np.floating] = ssc.predictive_pvalue(
            observed_log_pred=observed_log_pred_spike,
            sample_log_pred=sample_log_pred,
            n_samples=n_samples,
        )
        # Insert p-values at spike time indices
        p_values[spike_time_indices] = p_values_spike

    return {
        "posterior": posterior,
        "predictive": predictive_posterior,
        "likelihood": combined_likelihood_all,
        "hpd_overlap": hpd_overlap,
        "kl_divergence": kl_divergence,
        "spike_prob": spike_prob,
        "p_values": p_values,
    }


# -----------------------------
# Thresholds & transforms
# -----------------------------


@dataclass
class Thresholds:
    """Threshold values for diagnostic metrics.

    Thresholds are typically computed from a baseline period to define what
    constitutes "abnormal" diagnostic values indicating model misfit.

    Parameters
    ----------
    hpd_overlap : float
        HPD overlap threshold (lower values indicate worse fit).
    kl_divergence : float
        KL divergence threshold (higher values indicate worse fit).
    spike_prob : float
        Spike probability threshold (lower values indicate worse fit).
    p_value_lower : float
        Lower p-value threshold (values below indicate misfit).
    p_value_upper : float
        Upper p-value threshold (values above indicate misfit).

    Examples
    --------
    >>> thresholds = Thresholds(
    ...     hpd_overlap=0.5,
    ...     kl_divergence=2.0,
    ...     spike_prob=0.05,
    ...     p_value_lower=0.05,
    ...     p_value_upper=0.95,
    ... )
    >>> thresholds.hpd_overlap
    0.5
    """

    hpd_overlap: float
    kl_divergence: float
    spike_prob: float
    p_value_lower: float
    p_value_upper: float


def compute_thresholds(
    metrics: dict[str, NDArray[np.floating]], baseline_end: int = 60_000
) -> Thresholds:
    """Compute threshold values from baseline period.

    Thresholds are computed as extreme quantiles of the baseline period:
    - HPD overlap threshold: 1st percentile (low values indicate misfit)
    - KL divergence threshold: 99th percentile (high values indicate misfit)
    - spike_prob threshold: Fixed at 0.05 (matches MATLAB implementation)
    - p_value thresholds: Fixed at 0.05 and 0.95 (well-calibrated model has uniform p-values)

    Parameters
    ----------
    metrics : dict[str, NDArray]
        Dictionary containing diagnostic metrics:
        - 'hpd_overlap' : np.ndarray, shape (n_time,)
        - 'kl_divergence' : np.ndarray, shape (n_time,)
        - 'spike_prob' : np.ndarray (not used, threshold is fixed)
        - 'p_values' : np.ndarray (not used, thresholds are fixed)
    baseline_end : int, default 60_000
        Index marking end of baseline period (exclusive).

    Returns
    -------
    thresholds : Thresholds
        Threshold values for each diagnostic metric.

    Examples
    --------
    >>> import numpy as np
    >>> metrics = {
    ...     'hpd_overlap': np.random.uniform(0.5, 1.0, 100),
    ...     'kl_divergence': np.random.uniform(0.0, 2.0, 100),
    ...     'spike_prob': np.random.uniform(0.0, 0.5, (100, 10)),
    ...     'p_values': np.random.uniform(0.0, 1.0, 100),
    ... }
    >>> thresholds = compute_thresholds(metrics, baseline_end=50)
    >>> thresholds.spike_prob
    0.05
    >>> thresholds.p_value_lower
    0.05
    """
    hpd_overlap_threshold = np.nanquantile(metrics["hpd_overlap"][:baseline_end], 0.01)
    kl_divergence_threshold = np.nanquantile(metrics["kl_divergence"][:baseline_end], 0.99)
    # MATLAB uses 0.05 as fixed threshold (raw count, not normalized)
    spike_prob_threshold = 0.05
    # P-values should be uniform [0,1] if model is well-calibrated
    # Flag extreme values (below 0.05 or above 0.95)
    p_value_lower_threshold = 0.05
    p_value_upper_threshold = 0.95
    return Thresholds(
        hpd_overlap=hpd_overlap_threshold,
        kl_divergence=kl_divergence_threshold,
        spike_prob=spike_prob_threshold,
        p_value_lower=p_value_lower_threshold,
        p_value_upper=p_value_upper_threshold,
    )


@dataclass
class Transformed:
    """Transformed diagnostic metrics and thresholds.

    Transformations are applied to diagnostic metrics to improve visualization
    and interpretability (e.g., log-transform for better dynamic range).

    Parameters
    ----------
    hpd_overlap : np.ndarray
        Transformed HPD overlap values.
    kl_divergence : np.ndarray
        Transformed KL divergence values.
    spike_prob : np.ndarray
        Transformed spike probability values.
    p_values : np.ndarray
        P-values (no transformation applied).
    hpd_overlap_threshold : float
        Transformed HPD overlap threshold.
    kl_divergence_threshold : float
        Transformed KL divergence threshold.
    spike_prob_threshold : float
        Transformed spike probability threshold.
    p_value_lower_threshold : float
        Lower p-value threshold.
    p_value_upper_threshold : float
        Upper p-value threshold.

    Examples
    --------
    >>> import numpy as np
    >>> transformed = Transformed(
    ...     hpd_overlap=np.array([1.0, 2.0, 3.0]),
    ...     kl_divergence=np.array([0.5, 1.0, 1.5]),
    ...     spike_prob=np.array([[0.1, 0.2], [0.3, 0.4]]),
    ...     p_values=np.array([0.1, 0.5, 0.9]),
    ...     hpd_overlap_threshold=1.5,
    ...     kl_divergence_threshold=1.0,
    ...     spike_prob_threshold=0.15,
    ...     p_value_lower_threshold=0.05,
    ...     p_value_upper_threshold=0.95,
    ... )
    >>> transformed.hpd_overlap_threshold
    1.5
    """

    hpd_overlap: NDArray[np.floating]
    kl_divergence: NDArray[np.floating]
    spike_prob: NDArray[np.floating]
    p_values: NDArray[np.floating]
    hpd_overlap_threshold: float
    kl_divergence_threshold: float
    spike_prob_threshold: float
    p_value_lower_threshold: float
    p_value_upper_threshold: float


def transform_metrics(
    metrics: dict[str, NDArray[np.floating]],
    thresholds: Thresholds,
    eps1: float = 1e-2,
    eps2: float = 1e-10,
) -> Transformed:
    """Apply transformations to metrics for better visualization.

    **Transformations**:
    - HPD overlap: -log(HPDO + eps1) - emphasizes low values (worse fit)
    - KL divergence: sqrt(KL) - compresses high values
    - spike_prob: -log(spikeProb + eps2) - emphasizes low values (worse fit)
    - p_values: no transformation (already in [0,1] range)

    The same transformations are applied to the threshold values.

    Parameters
    ----------
    metrics : dict[str, NDArray]
        Dictionary containing diagnostic metrics:
        - 'hpd_overlap' : np.ndarray, shape (n_time,)
        - 'kl_divergence' : np.ndarray, shape (n_time,)
        - 'spike_prob' : np.ndarray, shape (n_time, n_cells)
        - 'p_values' : np.ndarray, shape (n_time,)
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
    ...     'hpd_overlap': np.array([0.5, 0.8, 0.9]),
    ...     'kl_divergence': np.array([1.0, 4.0, 9.0]),
    ...     'spike_prob': np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]),
    ...     'p_values': np.array([0.1, 0.5, 0.9]),
    ... }
    >>> thresholds = Thresholds(
    ...     hpd_overlap=0.6,
    ...     kl_divergence=5.0,
    ...     spike_prob=0.05,
    ...     p_value_lower=0.05,
    ...     p_value_upper=0.95,
    ... )
    >>> transformed = transform_metrics(metrics, thresholds)
    >>> transformed.kl_divergence  # sqrt(KL)
    array([1.        , 2.        , 3.        ])
    """
    hpd_overlap_transformed = -safe_log(metrics["hpd_overlap"] + eps1)
    kl_divergence_transformed = np.sqrt(metrics["kl_divergence"])
    spike_prob_transformed = -safe_log(metrics["spike_prob"] + eps2)

    return Transformed(
        hpd_overlap=hpd_overlap_transformed,
        kl_divergence=kl_divergence_transformed,
        spike_prob=spike_prob_transformed,
        p_values=metrics["p_values"],  # No transformation for p-values
        hpd_overlap_threshold=-np.log(thresholds.hpd_overlap + eps1),
        kl_divergence_threshold=np.sqrt(thresholds.kl_divergence),
        spike_prob_threshold=-np.log(thresholds.spike_prob + eps2),
        p_value_lower_threshold=thresholds.p_value_lower,
        p_value_upper_threshold=thresholds.p_value_upper,
    )
