"""Posterior predictive checks for state space models.

This module implements posterior predictive checks (PPCs) for validating
Bayesian state space models by comparing observed data to data simulated
from the posterior predictive distribution.

The workflow is:
1. Compute observed log predictive density from actual data
2. Sample from posterior predictive distribution:
   - Draw latent states from p(x_t | y_{1:t-1})
   - Generate synthetic observations from p(y_t | x_t)
   - Recompute log predictive density for synthetic data
3. Compare observed to simulated via p-values (ranks)

If the model is well-calibrated, p-values should be uniformly distributed.

Examples
--------
>>> import numpy as np
>>> import xarray as xr
>>> from statespacecheck_paper.predictive_check import (
...     compute_predictive_pvalues,
...     sample_positions_from_posterior,
... )
>>> # With fitted model and results
>>> p_values = compute_predictive_pvalues(
...     model, results, n_samples=1000, rng=np.random.default_rng(42)
... )
>>> # p_values should be uniform [0, 1] if model is calibrated
>>> import matplotlib.pyplot as plt
>>> plt.hist(p_values, bins=20, density=True)
>>> plt.axhline(1.0, color='red', linestyle='--', label='Uniform')
"""

from __future__ import annotations

from typing import Any

import numpy as np
import xarray as xr
from numpy.typing import NDArray
from scipy.stats import poisson
from statespacecheck import log_predictive_density, predictive_pvalue

from statespacecheck_paper.simulation import normalize


def sample_positions_from_posterior(
    posterior: NDArray[np.float64],
    rng: np.random.Generator,
) -> NDArray[np.int64]:
    """Sample position indices from categorical posterior distribution.

    Parameters
    ----------
    posterior : np.ndarray, shape (n_time, n_bins)
        Posterior probability distribution over positions at each time.
        Each row should sum to 1 (will be normalized if not).
    rng : np.random.Generator
        Random number generator for reproducibility.

    Returns
    -------
    position_indices : np.ndarray, shape (n_time,)
        Sampled position bin indices at each time point.

    Examples
    --------
    >>> rng = np.random.default_rng(42)
    >>> posterior = np.array([[0.2, 0.5, 0.3], [0.1, 0.1, 0.8]])
    >>> positions = sample_positions_from_posterior(posterior, rng)
    >>> positions.shape
    (2,)
    >>> np.all((positions >= 0) & (positions < 3))
    True
    """
    n_time, n_bins = posterior.shape

    # Normalize to ensure valid probability distributions
    # Use shared normalize() function for numerical safety
    posterior_norm = normalize(posterior, axis=1)

    # Vectorized categorical sampling using inverse CDF method
    # For each row, we compute cumulative sum and find where uniform random
    # values fall in the CDF. This is ~18x faster than looping over rng.choice().
    cumsum = np.cumsum(posterior_norm, axis=1)
    uniform = rng.random((n_time, 1))
    position_indices_raw = np.sum(uniform > cumsum, axis=1)
    # Clamp to valid range [0, n_bins-1] to handle numerical precision issues
    position_indices: NDArray[np.int64] = np.clip(position_indices_raw, 0, n_bins - 1).astype(
        np.int64
    )

    return position_indices


def generate_spikes_from_place_fields(
    position_indices: NDArray[np.int64],
    place_fields: NDArray[np.float64],
    dt: float,
    rng: np.random.Generator,
) -> NDArray[np.int64]:
    """Generate synthetic spike counts from place field model.

    Parameters
    ----------
    position_indices : np.ndarray, shape (n_time,)
        Position bin indices at each time point.
    place_fields : np.ndarray, shape (n_cells, n_bins)
        Firing rate (spikes/sec) for each cell at each position bin.
    dt : float
        Time bin duration in seconds (e.g., 0.002 for 500 Hz sampling).
    rng : np.random.Generator
        Random number generator for reproducibility.

    Returns
    -------
    spike_counts : np.ndarray, shape (n_time, n_cells)
        Synthetic spike counts drawn from Poisson distribution.

    Examples
    --------
    >>> rng = np.random.default_rng(42)
    >>> position_indices = np.array([0, 1, 2])
    >>> place_fields = np.array([[10.0, 20.0, 5.0], [5.0, 10.0, 15.0]])
    >>> dt = 0.002  # 500 Hz
    >>> spikes = generate_spikes_from_place_fields(
    ...     position_indices, place_fields, dt, rng
    ... )
    >>> spikes.shape
    (3, 2)
    """
    # Get firing rates at sampled positions: shape (n_time, n_cells)
    rates_at_positions = place_fields[:, position_indices].T

    # Expected spike count in time bin dt
    expected_counts = rates_at_positions * dt

    # Draw from Poisson distribution
    spike_counts: NDArray[np.int64] = rng.poisson(expected_counts).astype(np.int64)

    return spike_counts


def compute_log_likelihood_from_place_fields(
    spike_counts: NDArray[np.int64],
    place_fields: NDArray[np.float64],
    dt: float,
) -> NDArray[np.float64]:
    """Compute log likelihood of spike counts under place field model.

    Uses Poisson observation model:
    log p(n_i | x) = n_i * log(λ_i(x) * dt) - λ_i(x) * dt - log(n_i!)

    Parameters
    ----------
    spike_counts : np.ndarray, shape (n_time, n_cells)
        Observed spike counts for each cell at each time.
    place_fields : np.ndarray, shape (n_cells, n_bins)
        Firing rate (spikes/sec) for each cell at each position bin.
    dt : float
        Time bin duration in seconds.

    Returns
    -------
    log_likelihood : np.ndarray, shape (n_time, n_bins)
        Log likelihood of observations at each position bin and time.

    Examples
    --------
    >>> spike_counts = np.array([[1, 0], [2, 1]])
    >>> place_fields = np.array([[10.0, 20.0, 5.0], [5.0, 10.0, 15.0]])
    >>> dt = 0.002
    >>> log_like = compute_log_likelihood_from_place_fields(
    ...     spike_counts, place_fields, dt
    ... )
    >>> log_like.shape
    (2, 3)

    Notes
    -----
    This implementation computes the full likelihood across all position bins
    for each timepoint, which is needed for the predictive density calculation.

    Edge cases:
    - When expected_counts = 0 (no firing rate), scipy.stats.poisson.logpmf(0, 0) = 0.0
    - When expected_counts is very small, log-likelihood remains numerically stable
    - Large spike counts are handled correctly by scipy's implementation
    """
    # Expected counts at each position: shape (n_cells, n_bins)
    expected_counts = place_fields * dt

    # Reshape for broadcasting: spike_counts (n_time, n_cells, 1)
    #                            expected_counts (1, n_cells, n_bins)
    spike_counts_3d = spike_counts[:, :, np.newaxis]
    expected_counts_3d = expected_counts[np.newaxis, :, :]

    # Poisson log likelihood using scipy
    # log p(n|λ) = n * log(λ) - λ - log(n!)
    log_like_per_cell = poisson.logpmf(spike_counts_3d, expected_counts_3d)

    # Sum over cells: shape (n_time, n_bins)
    log_likelihood: NDArray[np.float64] = log_like_per_cell.sum(axis=1).astype(np.float64)

    return log_likelihood


def extract_place_fields_from_model(
    model: Any,
) -> tuple[NDArray[np.float64], float]:
    """Extract place fields and time bin size from fitted model.

    Parameters
    ----------
    model : SortedSpikesDecoder or similar
        Fitted decoder model from non_local_detector.

    Returns
    -------
    place_fields : np.ndarray, shape (n_cells, n_bins)
        Firing rates (spikes/sec) for each cell at each position.
    dt : float
        Time bin duration in seconds.

    Examples
    --------
    >>> # With actual fitted model
    >>> place_fields, dt = extract_place_fields_from_model(cont_model)
    >>> place_fields.shape
    (n_cells, n_bins)
    """
    # Extract from non_local_detector model structure
    # The encoding model stores place fields in the first environment
    encoding_model = model.encoding_model_[("", 0)]
    place_fields = encoding_model["place_fields"]  # Shape: (n_cells, n_bins)

    # Get time bin size from model
    dt = 1.0 / model.sampling_frequency

    return place_fields, dt


def create_posterior_predictive_sampler(
    model: Any,
    results: xr.Dataset,
    rng: np.random.Generator,
) -> Any:
    """Create sampler function for posterior predictive check.

    Parameters
    ----------
    model : SortedSpikesDecoder
        Fitted decoder model.
    results : xr.Dataset
        Decoding results containing predictive_posterior and log_likelihood.
    rng : np.random.Generator
        Random number generator for reproducibility.

    Returns
    -------
    sampler : callable
        Function that takes n_samples and returns array of shape
        (n_samples, n_time) containing log predictive densities for
        simulated data.

    Examples
    --------
    >>> rng = np.random.default_rng(42)
    >>> sampler = create_posterior_predictive_sampler(model, results, rng)
    >>> simulated_log_pred = sampler(100)  # 100 Monte Carlo samples
    >>> simulated_log_pred.shape
    (100, n_time)
    """
    # Extract place fields and time bin
    place_fields, dt = extract_place_fields_from_model(model)

    # Get predictive posterior (state distribution before observing data)
    predictive_posterior = results.predictive_posterior.dropna("state_bins").to_numpy()
    n_time = predictive_posterior.shape[0]

    def sampler(n_samples: int) -> NDArray[np.float64]:
        """Generate n_samples of log predictive densities."""
        log_pred_samples = np.zeros((n_samples, n_time))

        for i in range(n_samples):
            # 1. Sample positions from predictive posterior
            position_indices = sample_positions_from_posterior(predictive_posterior, rng)

            # 2. Generate synthetic spikes given sampled positions
            spike_counts_sim = generate_spikes_from_place_fields(
                position_indices, place_fields, dt, rng
            )

            # 3. Compute log likelihood for synthetic spikes
            log_like_sim = compute_log_likelihood_from_place_fields(
                spike_counts_sim, place_fields, dt
            )

            # 4. Compute log predictive density for synthetic data
            log_pred_samples[i] = log_predictive_density(
                state_dist=predictive_posterior, log_likelihood=log_like_sim
            )

        return log_pred_samples

    return sampler


def compute_predictive_pvalues(
    model: Any,
    results: xr.Dataset,
    n_samples: int = 1000,
    rng: np.random.Generator | None = None,
) -> NDArray[np.float64]:
    """Compute posterior predictive p-values for model validation.

    This function performs a posterior predictive check by:
    1. Computing observed log predictive density from actual data
    2. Generating synthetic data from the posterior predictive distribution
    3. Computing p-values comparing observed to simulated densities

    If the model is well-calibrated, p-values should be uniformly distributed
    on [0, 1]. Systematic deviations indicate model misfit.

    Parameters
    ----------
    model : SortedSpikesDecoder
        Fitted decoder model from non_local_detector.
    results : xr.Dataset
        Decoding results containing:
        - predictive_posterior: p(x_t | y_{1:t-1})
        - log_likelihood: log p(y_t | x_t)
    n_samples : int, default 1000
        Number of Monte Carlo samples for p-value estimation.
        Higher values give more accurate p-values but take longer.
    rng : np.random.Generator, optional
        Random number generator for reproducibility.
        If None, creates new generator with random seed.

    Returns
    -------
    p_values : np.ndarray, shape (n_time,)
        P-value at each time point, computed as the proportion of
        simulated log predictive densities >= observed value.
        Values range from 0 to 1.

    Examples
    --------
    >>> import numpy as np
    >>> from statespacecheck_paper.predictive_check import (
    ...     compute_predictive_pvalues
    ... )
    >>> # With fitted model and results
    >>> rng = np.random.default_rng(42)
    >>> p_values = compute_predictive_pvalues(
    ...     cont_model, cont_results, n_samples=1000, rng=rng
    ... )
    >>> # Check calibration: p-values should be uniform
    >>> import matplotlib.pyplot as plt
    >>> plt.hist(p_values, bins=20, density=True)
    >>> plt.axhline(1.0, color='red', linestyle='--')
    >>> plt.xlabel('P-value')
    >>> plt.ylabel('Density')
    >>> plt.title('Posterior Predictive Check')

    Notes
    -----
    This is a formal Bayesian model validation technique. Unlike HPD overlap
    or KL divergence (which measure instantaneous consistency), this checks
    whether the full generative model is well-calibrated.

    See Also
    --------
    statespacecheck.predictive_pvalue : Core function for computing p-values
    statespacecheck.log_predictive_density : Compute predictive density
    """
    if rng is None:
        rng = np.random.default_rng()

    # 1. Compute observed log predictive density
    observed_log_pred = log_predictive_density(
        state_dist=results.predictive_posterior.dropna("state_bins").to_numpy(),
        log_likelihood=results.log_likelihood.dropna("state_bins").to_numpy(),
    )

    # 2. Create sampler for posterior predictive distribution
    sampler = create_posterior_predictive_sampler(model, results, rng)

    # 3. Compute p-values using Monte Carlo sampling
    p_values_result: NDArray[np.float64] = predictive_pvalue(
        observed_log_pred=observed_log_pred,
        sample_log_pred=sampler,
        n_samples=n_samples,
    ).astype(np.float64)

    return p_values_result
