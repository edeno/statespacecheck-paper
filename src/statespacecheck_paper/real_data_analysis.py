"""Analysis utilities for real neural data.

This module provides utilities for analyzing real neural recordings, including
spike rate estimation, sustained event detection, temporal smoothing, and
per-cell diagnostic computations for model checking.

**Key Components**:

- **gaussian_smooth**: Apply 1D Gaussian convolution
- **get_multiunit_population_firing_rate**: Calculate smoothed population rate
- **find_sustained_low_overlap**: Find low-overlap time regions
- **extract_place_fields**: Extract place fields from fitted decoder model
- **compute_per_cell_likelihood**: Compute per-cell Poisson likelihood
- **compute_per_cell_diagnostics**: Compute HPD overlap, KL divergence, spike prob
- **get_state_marginalized_posterior**: Extract state-marginalized posterior

Examples
--------
>>> import numpy as np
>>> from statespacecheck_paper.real_data_analysis import gaussian_smooth
>>> data = np.random.randn(1000)
>>> smoothed = gaussian_smooth(data, sigma=0.02, sampling_frequency=500)
>>> smoothed.shape
(1000,)
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import xarray as xr
from numpy.typing import NDArray
from scipy.ndimage import gaussian_filter1d, label

from statespacecheck_paper.analysis import compute_per_cell_diagnostics_from_rates


def gaussian_smooth(
    data: NDArray[np.float64],
    sigma: float,
    sampling_frequency: float,
    axis: int = 0,
    truncate: int = 8,
) -> NDArray[np.float64]:
    """Apply 1D Gaussian convolution to data.

    The standard deviation of the gaussian is in the units of the sampling
    frequency. The function is a wrapper around scipy's `gaussian_filter1d`.
    The support is truncated at 8 by default, instead of 4 in `gaussian_filter1d`.

    Parameters
    ----------
    data : np.ndarray
        Input data to smooth.
    sigma : float
        Standard deviation of the Gaussian kernel in seconds.
    sampling_frequency : float
        Number of samples per second.
    axis : int, default=0
        Axis along which to apply the filter.
    truncate : int, default=8
        Truncate the filter at this many standard deviations.

    Returns
    -------
    smoothed_data : np.ndarray
        Gaussian-smoothed data with same shape as input.

    Examples
    --------
    >>> data = np.random.randn(1000)
    >>> smoothed = gaussian_smooth(data, sigma=0.01, sampling_frequency=1000)
    >>> smoothed.shape
    (1000,)
    """
    # scipy.ndimage.gaussian_filter1d preserves input dtype
    result: NDArray[np.float64] = gaussian_filter1d(
        data,
        sigma * sampling_frequency,
        truncate=truncate,
        axis=axis,
        mode="constant",
    )
    return result


def get_multiunit_population_firing_rate(
    multiunit: NDArray[np.float64],
    sampling_frequency: float,
    smoothing_sigma: float = 0.015,
) -> NDArray[np.float64]:
    """Calculate smoothed multiunit population firing rate.

    Parameters
    ----------
    multiunit : np.ndarray, shape (n_time, n_signals)
        Binary array of multiunit spike times.
    sampling_frequency : float
        Number of samples per second.
    smoothing_sigma : float, default=0.015
        Amount to smooth the firing rate over time in seconds.

    Returns
    -------
    multiunit_population_firing_rate : np.ndarray, shape (n_time,)
        Smoothed population firing rate in spikes per second.

    Examples
    --------
    >>> multiunit = np.random.poisson(0.1, size=(1000, 50))
    >>> rate = get_multiunit_population_firing_rate(
    ...     multiunit, sampling_frequency=500, smoothing_sigma=0.015
    ... )
    >>> rate.shape
    (1000,)
    """
    return gaussian_smooth(
        multiunit.sum(axis=1) * sampling_frequency,
        smoothing_sigma,
        sampling_frequency,
    )


def find_sustained_low_overlap(
    hpd_overlap: NDArray[np.float64],
    threshold: float = 0.5,
    min_duration: float = 0.010,
    smooth_sigma: float = 1.0,
    sampling_frequency: float = 1.0,
) -> list[tuple[int, int]]:
    """Find contiguous regions where HPD overlap is below threshold.

    Identifies time periods where a smoothed HPD overlap metric remains
    below a threshold for at least a minimum duration.

    Parameters
    ----------
    hpd_overlap : np.ndarray, shape (n_time,)
        Array of HPD overlap values.
    threshold : float, default=0.5
        Threshold for low overlap.
    min_duration : float, default=0.010
        Minimum duration in seconds for contiguous samples below threshold.
    smooth_sigma : float, default=1.0
        Standard deviation for Gaussian smoothing in samples.
    sampling_frequency : float, default=1.0
        Number of samples per second.

    Returns
    -------
    regions : list[tuple[int, int]]
        List of (start_idx, end_idx) for each sustained low-overlap region.

    Examples
    --------
    >>> hpd_overlap = np.random.rand(1000)
    >>> regions = find_sustained_low_overlap(
    ...     hpd_overlap, threshold=0.3, min_duration=0.020, sampling_frequency=500
    ... )
    >>> isinstance(regions, list)
    True
    """
    smoothed = gaussian_filter1d(hpd_overlap, sigma=smooth_sigma)
    low_mask = smoothed < threshold
    labels_array, n_labels = label(low_mask)
    regions = []

    # Convert min_duration from seconds to number of samples
    min_samples = int(min_duration * sampling_frequency)

    for label_id in range(1, n_labels + 1):
        inds = np.where(labels_array == label_id)[0]
        if len(inds) >= min_samples:
            start = int(inds[0])
            end = int(inds[-1])
            regions.append((start, end))

    return regions


# =============================================================================
# Per-Cell Diagnostic Functions for Real Data
# =============================================================================


def extract_place_fields(
    model: Any,
    environment_name: str = "",
    encoding_group: int = 0,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Extract place fields and position bins from fitted decoder model.

    Retrieves the place field firing rates and corresponding position bin centers
    from a fitted `SortedSpikesDecoder` or `ContFragSortedSpikesClassifier` model.

    Parameters
    ----------
    model : SortedSpikesDecoder or ContFragSortedSpikesClassifier
        Fitted decoder model from non_local_detector package.
    environment_name : str, default ""
        Name of the environment in the model. Default empty string for standard
        single-environment models.
    encoding_group : int, default 0
        Encoding group index. Default 0 for standard models.

    Returns
    -------
    place_fields : np.ndarray, shape (n_cells, n_bins)
        Firing rate at each position bin for each cell (in Hz or spikes/time).
    position_bins : np.ndarray, shape (n_bins,)
        Position bin centers.

    Examples
    --------
    >>> # Requires fitted model from non_local_detector
    >>> # place_fields, position_bins = extract_place_fields(model)
    >>> # place_fields.shape  # (n_cells, n_bins)
    >>> # position_bins.shape  # (n_bins,)
    """
    # Access place fields from encoding model
    # Key is tuple (environment_name, encoding_group)
    key = (environment_name, encoding_group)
    place_fields: NDArray[np.float64] = model.encoding_model_[key]["place_fields"]

    # Get position bin centers from environment
    # environments is a list; encoding_group corresponds to environment index
    position_bins: NDArray[np.float64] = model.environments[
        encoding_group
    ].place_bin_centers_.squeeze()

    return place_fields, position_bins


def compute_per_cell_diagnostics(
    predictive_posterior: NDArray[np.float64],
    spike_counts: NDArray[np.int64],
    place_fields: NDArray[np.float64],
    coverage: float = 0.95,
) -> dict[str, NDArray[np.float64]]:
    """Compute per-cell diagnostic metrics for model checking.

    Computes HPD overlap, KL divergence, and spike probability ranking for each
    cell at each time point. Metrics are set to NaN for cells that did not fire
    at a given time point (uninformative likelihood).

    The likelihood for each spike is computed assuming spike count = 1,
    which is appropriate when spike counts are typically 0 or 1 per time bin.

    Parameters
    ----------
    predictive_posterior : np.ndarray, shape (n_time, n_bins)
        State-marginalized predictive posterior distribution over position.
    spike_counts : np.ndarray, shape (n_time, n_cells)
        Spike count for each cell at each time point.
    place_fields : np.ndarray, shape (n_cells, n_bins)
        Expected spike count at each position bin for each cell (spikes/bin).
        This is the format returned by non_local_detector.
    coverage : float, default 0.95
        Coverage probability for HPD region computation.

    Returns
    -------
    diagnostics : dict[str, np.ndarray]
        Dictionary with keys:
        - 'hpd_overlap': shape (n_time, n_cells), NaN when cell has no spike
        - 'kl_divergence': shape (n_time, n_cells), NaN when cell has no spike
        - 'spike_prob': shape (n_time, n_cells), NaN when cell has no spike

    Notes
    -----
    The likelihood P(k=1 | position) is computed for each spike event using
    the Poisson distribution with rate lambda = place_fields (already in
    spikes/bin from non_local_detector). We assume each bin contains at most
    one spike, so the likelihood is always computed with k=1.

    This function delegates to ``compute_per_cell_diagnostics_from_rates`` in
    ``analysis.py`` to ensure identical computation for simulated and real data.

    Examples
    --------
    >>> import numpy as np
    >>> n_time, n_bins, n_cells = 100, 50, 10
    >>> predictive = np.random.dirichlet(np.ones(n_bins), size=n_time)
    >>> place_fields = np.random.rand(n_cells, n_bins) * 10
    >>> spike_counts = np.random.poisson(0.5, (n_time, n_cells))
    >>> diagnostics = compute_per_cell_diagnostics(
    ...     predictive, spike_counts, place_fields
    ... )
    >>> diagnostics['hpd_overlap'].shape
    (100, 10)
    """
    # Ensure all inputs are NumPy arrays (handles JAX arrays from decoder)
    predictive_posterior = np.asarray(predictive_posterior)
    spike_counts = np.asarray(spike_counts)
    place_fields = np.asarray(place_fields)

    # place_fields is (n_cells, n_bins), we need (n_bins, n_cells) for the shared function
    rates = place_fields.T  # (n_bins, n_cells)

    # Find all spike events: (time_index, cell_index) pairs
    spike_time_ind, spike_cell_ind = np.nonzero(spike_counts)

    # Use shared function from analysis.py
    result = compute_per_cell_diagnostics_from_rates(
        predictive_posterior,
        rates,
        spike_time_ind.astype(np.intp),
        spike_cell_ind.astype(np.intp),
        coverage=coverage,
    )

    # Cast to float64 for type consistency
    return {k: v.astype(np.float64) for k, v in result.items()}


def get_state_marginalized_posterior(
    results: xr.Dataset,
    posterior_type: Literal["predictive", "acausal"] = "predictive",
) -> NDArray[np.float64]:
    """Extract state-marginalized posterior from decoder results.

    For multi-state models (e.g., ContFragSortedSpikesClassifier), sums over
    states to get the marginal posterior over position. For single-state models,
    simply extracts the posterior. Also handles NaN state bins (e.g., track edges).

    Parameters
    ----------
    results : xr.Dataset
        Decoding results from model.predict() containing posterior distributions.
    posterior_type : {"predictive", "acausal"}, default "predictive"
        Type of posterior to extract:
        - "predictive": One-step-ahead prediction p(x_t | y_{1:t-1})
        - "acausal": Smoothed posterior p(x_t | y_{1:T})

    Returns
    -------
    posterior : np.ndarray, shape (n_time, n_bins)
        State-marginalized posterior summed over states, with NaN bins dropped.

    Examples
    --------
    >>> # Requires xarray Dataset from non_local_detector
    >>> # posterior = get_state_marginalized_posterior(results, "predictive")
    >>> # posterior.shape  # (n_time, n_bins)
    """
    # Select appropriate posterior
    if posterior_type == "predictive":
        posterior_da = results.predictive_posterior
    else:
        posterior_da = results.acausal_posterior

    # Drop NaN state bins (e.g., track interior only)
    posterior_da = posterior_da.dropna("state_bins")

    # Check if this is a multi-state model by looking for state dimension
    # after unstacking state_bins
    try:
        unstacked = posterior_da.unstack("state_bins")
        if "state" in unstacked.dims:
            # Multi-state model: sum over states
            marginalized = unstacked.sum("state")
            posterior: NDArray[np.float64] = np.asarray(marginalized.values)
        else:
            # Single-state model: just extract values
            posterior = np.asarray(posterior_da.values)
    except (ValueError, KeyError):
        # If unstack fails, assume single-state model
        posterior = np.asarray(posterior_da.values)

    return posterior


# =============================================================================
# High-Level Analysis Pipeline Functions
# =============================================================================


def create_decoder_environment(
    track_graph: Any,
    edge_order: list[tuple[Any, Any]],
    edge_spacing: float | list[float],
) -> Any:
    """Create track environment for decoder models.

    Parameters
    ----------
    track_graph : networkx.Graph
        Track structure graph.
    edge_order : list[tuple]
        Edge ordering for linearization.
    edge_spacing : float or list[float]
        Spacing between nodes.

    Returns
    -------
    env : Environment
        Track environment object.

    Raises
    ------
    ImportError
        If non_local_detector package is not available.

    Examples
    --------
    >>> # Requires non_local_detector package
    >>> # env = create_decoder_environment(track_graph, edge_order, edge_spacing)
    """
    try:
        from non_local_detector.environment import Environment
    except ImportError as e:
        raise ImportError(
            "non_local_detector package required. Install with: pip install non_local_detector"
        ) from e

    return Environment(
        track_graph=track_graph,
        edge_order=edge_order,
        edge_spacing=edge_spacing,
    )


def fit_decoder_models(
    position: NDArray[np.float64],
    spike_times: list[NDArray[np.float64]],
    time: NDArray[np.float64],
    environment: Any,
) -> tuple[Any, Any]:
    """Fit Continuous and ContFrag decoder models.

    Parameters
    ----------
    position : np.ndarray, shape (n_time,)
        Linear position values.
    spike_times : list[np.ndarray]
        List of spike time arrays, one per cell.
    time : np.ndarray, shape (n_time,)
        Time values corresponding to position.
    environment : Environment
        Track environment object.

    Returns
    -------
    continuous_model : SortedSpikesDecoder
        Fitted continuous decoder model.
    contfrag_model : ContFragSortedSpikesClassifier
        Fitted continuous-fragmented decoder model.

    Raises
    ------
    ImportError
        If non_local_detector package is not available.

    Examples
    --------
    >>> # Requires non_local_detector package and fitted environment
    >>> # continuous_model, contfrag_model = fit_decoder_models(
    >>> #     position, spike_times, time, environment
    >>> # )
    """
    try:
        from non_local_detector import (
            ContFragSortedSpikesClassifier,
            SortedSpikesDecoder,
        )
    except ImportError as e:
        raise ImportError(
            "non_local_detector package required. Install with: pip install non_local_detector"
        ) from e

    # Ensure position is 2D (n_time, 1) for the decoder
    position_2d = position.reshape(-1, 1) if position.ndim == 1 else position

    # Fit Continuous model
    continuous_model = SortedSpikesDecoder(environments=[environment])
    continuous_model.fit(position=position_2d, spike_times=spike_times, position_time=time)

    # Fit ContFrag model
    contfrag_model = ContFragSortedSpikesClassifier(environments=[environment])
    contfrag_model.fit(position=position_2d, spike_times=spike_times, position_time=time)

    return continuous_model, contfrag_model


def get_spike_counts(
    spike_times: list[NDArray[np.float64]],
    time: NDArray[np.float64],
) -> NDArray[np.int64]:
    """Get spike count matrix aligned to time bins.

    Parameters
    ----------
    spike_times : list[np.ndarray]
        List of spike time arrays, one per cell.
    time : np.ndarray, shape (n_time,)
        Time bin centers.

    Returns
    -------
    spike_counts : np.ndarray, shape (n_time, n_cells)
        Spike count for each cell at each time bin.

    Raises
    ------
    ImportError
        If non_local_detector package is not available.

    Examples
    --------
    >>> # Requires non_local_detector package
    >>> # spike_counts = get_spike_counts(spike_times, time)
    >>> # spike_counts.shape  # (n_time, n_cells)
    """
    try:
        from non_local_detector.likelihoods.common import get_spikecount_per_time_bin
    except ImportError as e:
        raise ImportError(
            "non_local_detector package required. Install with: pip install non_local_detector"
        ) from e

    counts_per_cell = [get_spikecount_per_time_bin(spike_times=st, time=time) for st in spike_times]
    spike_counts = np.stack(counts_per_cell, axis=1).astype(np.int64)

    return spike_counts


def compute_model_diagnostics(
    model: Any,
    results: Any,
    spike_counts: NDArray[np.int64],
    time: NDArray[np.float64],
) -> dict[str, NDArray[np.float64]]:
    """Compute per-cell diagnostics for a fitted decoder model.

    This is a convenience function that chains together extract_place_fields,
    compute_per_cell_likelihood, get_state_marginalized_posterior, and
    compute_per_cell_diagnostics.

    Parameters
    ----------
    model : decoder model
        Fitted SortedSpikesDecoder or ContFragSortedSpikesClassifier.
    results : xr.Dataset
        Decoding results from model.predict().
    spike_counts : np.ndarray, shape (n_time, n_cells)
        Spike count matrix.
    time : np.ndarray, shape (n_time,)
        Time values.

    Returns
    -------
    diagnostics : dict[str, np.ndarray]
        Dictionary with keys 'hpd_overlap', 'kl_divergence', 'spike_prob'.
        Each has shape (n_time, n_cells).

    Examples
    --------
    >>> # Requires fitted model and decoding results
    >>> # diagnostics = compute_model_diagnostics(model, results, spike_counts, time)
    >>> # diagnostics['hpd_overlap'].shape  # (n_time, n_cells)
    """
    # Extract place fields
    place_fields_full, position_bins_full = extract_place_fields(model)

    # Get state-marginalized predictive posterior (handles multi-state models)
    # This drops NaN state bins internally
    predictive_posterior = get_state_marginalized_posterior(results, posterior_type="predictive")

    # The posterior may have fewer bins than place_fields if track edges are NaN
    # Determine which position bins are valid by comparing sizes
    n_posterior_bins = predictive_posterior.shape[1]
    n_place_field_bins = place_fields_full.shape[1]

    if n_posterior_bins < n_place_field_bins:
        # Posterior has dropped NaN bins - need to find which ones are valid
        # Get the original state_bins coordinate to identify valid positions
        if "predictive_posterior" in results:
            posterior_da = results.predictive_posterior
        else:
            posterior_da = results.acausal_posterior

        # Drop NaN state bins (same as get_state_marginalized_posterior does)
        posterior_da = posterior_da.dropna("state_bins")

        # For multi-state models, unstack to get position dimension
        try:
            unstacked = posterior_da.unstack("state_bins")
            if "position" in unstacked.dims:
                # Get unique positions from the unstacked coordinates
                valid_positions = unstacked.coords["position"].values
            else:
                # Single-state model - positions are in state_bins directly
                valid_positions = posterior_da.coords["state_bins"].values
        except (ValueError, KeyError):
            # Fallback: assume first n_posterior_bins positions are valid
            valid_positions = position_bins_full[:n_posterior_bins]

        # Find mask of valid position bins
        valid_mask = np.isin(position_bins_full, valid_positions)
        place_fields = place_fields_full[:, valid_mask]
    else:
        # All bins are valid
        place_fields = place_fields_full

    # Compute diagnostics using actual spike counts
    # place_fields are already in spikes per time bin from non_local_detector
    diagnostics = compute_per_cell_diagnostics(
        predictive_posterior,
        spike_counts,
        place_fields,
    )

    return diagnostics
