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
    >>> # Run decoder
    >>> results = decode_and_diagnostics(
    ...     spikes, xs, transition_matrix, params.pf_centers,
    ...     params.pf_width, params.rate_scale, params.remap_window, params.remap_from_to
    ... )
    >>> sorted(results.keys())
    ['hpd_overlap', 'kl_divergence', 'likelihood', 'posterior', 'predictive', 'spike_prob']
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import statespacecheck as ssc
from numpy.typing import NDArray
from scipy.stats import poisson

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
    remap_from_to : tuple of ints, default (9, 0)
        Remapping specification: (src, dst) remaps cell src to use cell dst's place field.
        Default remaps cell 9 (place field at 90) to use cell 0's place field (at 0),
        matching the MATLAB implementation.

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
    rate_scale: float = 0.02  # Spike rate scaling factor (matches MATLAB normpdf * 0.02)
    base_seed: int = 1
    remap_from_to: tuple[tuple[int, int], ...] | tuple[int, int] = (
        2,
        8,
    )  # Remap cell 2 (pf_center=20) to use cell 8's place field (pf_center=80)

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
        Remapping specification (see get_remapped_pf_centers).
    rng : np.random.Generator | None, optional
        Random number generator (reserved for future use).
    transition_matrix_narrow : np.ndarray, shape (n_bins, n_bins), optional
        Alternative transition matrix for narrow window (fast movement misfit).
    narrow_window : tuple[int, int], optional
        Time window (start, end) for narrow transition matrix.
    transition_matrix_inflated : np.ndarray, shape (n_bins, n_bins), optional
        Alternative transition matrix for inflated window (slow movement misfit).
    inflate_window : tuple[int, int], optional
        Time window (start, end) for inflated transition matrix.

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
    >>> results['posterior'].shape
    (10, 21)
    >>> results['hpd_overlap'].shape  # Now per-cell
    (10, 3)
    >>> np.all(np.isnan(results['hpd_overlap'][0]))  # t=0 has no prior
    True
    """
    n_time, n_cells = spikes.shape
    n_bins = xs.size

    # Initialize RNG if not provided (reserved for future use)
    if rng is None:
        rng = np.random.default_rng()
    # Suppress unused variable warning
    _ = rng

    # Preallocate outputs (NaN for unavailable values)
    # Per-cell metrics: shape (n_time, n_cells), NaN when cell has no spike
    posterior: NDArray[np.floating] = np.zeros((n_time, n_bins))
    hpd_overlap: NDArray[np.floating] = np.full((n_time, n_cells), np.nan)
    kl_divergence: NDArray[np.floating] = np.full((n_time, n_cells), np.nan)
    spike_prob: NDArray[np.floating] = np.full((n_time, n_cells), np.nan)

    # Storage for distributions
    predictive_posterior: NDArray[np.floating] = np.zeros((n_time, n_bins))  # p(x_t | y_{1:t-1})
    combined_likelihood_all: NDArray[np.floating] = np.zeros((n_time, n_bins))  # p(y_t | x_t)
    # Per-cell likelihoods for batched diagnostic computation
    per_cell_likelihood: NDArray[np.floating] = np.zeros((n_time, n_bins, n_cells))

    # t=0 (MATLAB used a flat prior at t=1)
    posterior[0] = normalize(np.ones(n_bins))
    predictive_posterior[0] = posterior[0]  # At t=0, predictive = prior
    combined_likelihood_all[0] = normalize(np.ones(n_bins))  # Flat at t=0

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
        # Optional remap: use remapped place field centers during misfit window
        # This matches MATLAB where cell j's likelihood is computed using another cell's pf_center
        active_remap = start_r <= t <= end_r
        current_pf_centers = get_remapped_pf_centers(pf_centers, remap_from_to, active_remap)
        likelihood = likelihood_grid_for_counts(
            xs, current_pf_centers, pf_width, rate_scale, spikes[t]
        )

        # Compute combined likelihood from all cells (product over cells)
        combined_likelihood = np.prod(likelihood, axis=1)  # (n_bins,)
        combined_likelihood_all[t] = normalize(combined_likelihood)  # Store normalized likelihood

        # Store per-cell likelihoods (normalized) for batched diagnostic computation
        per_cell_likelihood[t] = normalize(likelihood, axis=0)  # Normalize each cell's likelihood

        # Posterior update with underflow protection
        # When prior-likelihood mismatch is extreme, the product can underflow to zero.
        # Fall back to uniform distribution to allow filter to recover.
        unnormalized_posterior = prior * combined_likelihood
        posterior_sum = np.sum(unnormalized_posterior)
        if posterior_sum < 1e-300:  # Numerical underflow detected
            posterior[t] = np.ones(n_bins) / n_bins  # Reset to uniform
        else:
            posterior[t] = unnormalized_posterior / posterior_sum

    # Compute per-cell diagnostics in batched mode (once per cell, vectorized over time)
    for j in range(n_cells):
        # Get cell j's likelihood across all timesteps: (n_time, n_bins)
        lik_j_all = per_cell_likelihood[:, :, j]

        # Compute HPD overlap and KL divergence for all timesteps at once
        hpd_overlap[:, j] = ssc.hpd_overlap(predictive_posterior, lik_j_all, coverage=0.95)
        kl_divergence[:, j] = ssc.kl_divergence(predictive_posterior, lik_j_all)

    # Mask t=0 (no valid prior) and cells that did not fire (set to NaN)
    no_spike_mask = spikes == 0
    hpd_overlap[0, :] = np.nan  # t=0 has no valid prior
    kl_divergence[0, :] = np.nan
    hpd_overlap[no_spike_mask] = np.nan
    kl_divergence[no_spike_mask] = np.nan

    # Compute spike_prob in batched mode (vectorized over time)
    spike_prob = spike_prob_rank(predictive_posterior, cell_fraction_per_bin)
    # Mask t=0 and cells that did not fire
    spike_prob[0, :] = np.nan
    spike_prob[no_spike_mask] = np.nan

    return {
        "posterior": posterior,
        "predictive": predictive_posterior,
        "likelihood": combined_likelihood_all,
        "hpd_overlap": hpd_overlap,  # (n_time, n_cells)
        "kl_divergence": kl_divergence,  # (n_time, n_cells)
        "spike_prob": spike_prob,  # (n_time, n_cells)
    }


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
    metrics: dict[str, NDArray[np.floating]], baseline_end: int = 60_000
) -> Thresholds:
    """Compute threshold values from baseline period.

    Thresholds are computed across ALL cells (flattened), matching MATLAB.
    The (n_time, n_cells) arrays are flattened to 1D before computing quantiles:
    - HPD overlap threshold: 1st percentile (low values indicate misfit)
    - KL divergence threshold: 99th percentile (high values indicate misfit)
    - spike_prob threshold: fixed at 0.05 (per MATLAB implementation)

    Parameters
    ----------
    metrics : dict[str, NDArray]
        Dictionary containing diagnostic metrics:
        - 'hpd_overlap' : np.ndarray, shape (n_time, n_cells)
        - 'kl_divergence' : np.ndarray, shape (n_time, n_cells)
        - 'spike_prob' : np.ndarray, shape (n_time, n_cells)
    baseline_end : int, default 60_000
        Index marking end of baseline period (exclusive).

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
    # Flatten (n_time, n_cells) to 1D for quantile computation
    hpd_baseline = metrics["hpd_overlap"][:baseline_end].ravel()
    hpd_overlap_threshold = np.nanquantile(hpd_baseline, 0.01)

    kl_baseline = metrics["kl_divergence"][:baseline_end].ravel()
    kl_divergence_threshold = np.nanquantile(kl_baseline, 0.99)

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
        Transformed HPD overlap values: -log(HPDO + eps1).
    kl_divergence : np.ndarray, shape (n_time, n_cells)
        Transformed KL divergence values: sqrt(KL).
    spike_prob : np.ndarray, shape (n_time, n_cells)
        Transformed spike probability values: -log(spikeProb + eps2).
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

    **Transformations** (matching MATLAB):
    - HPD overlap: -log(HPDO + eps1) - emphasizes low values (worse fit)
    - KL divergence: sqrt(KL) - compresses high values
    - spike_prob: -log(spikeProb + eps2) - emphasizes low values (worse fit)

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
    >>> np.allclose(transformed.spike_prob_threshold, -np.log(0.05 + 1e-10))
    True
    """
    hpd_overlap_transformed = -safe_log(metrics["hpd_overlap"] + eps1)
    kl_divergence_transformed = np.sqrt(metrics["kl_divergence"])
    # spike_prob transformed with -log (matching MATLAB's -log(spikeProb + 1e-10))
    spike_prob_transformed = -safe_log(metrics["spike_prob"] + eps2)

    return Transformed(
        hpd_overlap=hpd_overlap_transformed,
        kl_divergence=kl_divergence_transformed,
        spike_prob=spike_prob_transformed,
        hpd_overlap_threshold=-np.log(thresholds.hpd_overlap + eps1),
        kl_divergence_threshold=np.sqrt(thresholds.kl_divergence),
        spike_prob_threshold=-np.log(thresholds.spike_prob + eps2),
    )
