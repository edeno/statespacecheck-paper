"""Analysis utilities for real neural data.

This module provides utilities for analyzing real neural recordings, including
spike rate estimation, sustained event detection, temporal smoothing, and
per-cell diagnostic computations for model checking.

**Key Components**:

- **gaussian_smooth**: Apply 1D Gaussian convolution
- **get_multiunit_population_firing_rate**: Calculate smoothed population rate
- **find_sustained_low_overlap**: Find low-overlap time regions
- **compute_running_average**: Compute running average of per-cell diagnostics
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
import pandas as pd
import xarray as xr
from numpy.typing import NDArray
from scipy.ndimage import gaussian_filter1d, label, uniform_filter1d

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


def compute_running_average(
    metric: NDArray[np.float64],
    time: NDArray[np.float64],
    window_size: float = 0.050,
    event_times: NDArray[np.float64] | None = None,
    event_values: NDArray[np.float64] | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Compute running average of per-cell diagnostic metric over time.

    Implements the event-weighted average formula from the manuscript:

        D(t) = sum(metric_k * I(t_k in window)) / sum(I(t_k in window))

    where the sum is over all spike events (time, cell) pairs and I(*) is the
    indicator function selecting events within the sliding window centered at t.
    Each spike event contributes equally regardless of how many cells fire in
    a given time bin.

    Parameters
    ----------
    metric : np.ndarray, shape (n_time, n_cells)
        Per-cell diagnostic metric values. NaN where cell has no spike.
    time : np.ndarray, shape (n_time,)
        Time values for each bin.
    window_size : float, default 0.050
        Size of the sliding window in seconds.
    event_times : np.ndarray, shape (n_events,), optional
        Exact event times. If provided with ``event_values``, the running
        average is computed directly over events instead of bin/cell matrix
        entries.
    event_values : np.ndarray, shape (n_events,), optional
        Diagnostic values for each event.

    Returns
    -------
    running_avg : np.ndarray, shape (n_time,)
        Running average of the metric over time. NaN where no events fall
        within the window.
    time_out : np.ndarray, shape (n_time,)
        Time values (same as input, for convenience).

    Notes
    -----
    The implementation computes sum(values) and count(events) per time bin,
    then applies a boxcar filter to both before dividing. This is equivalent
    to the event-weighted sliding window average but runs in O(n) time.
    Edge effects are handled by ``uniform_filter1d(mode="constant")``, which
    zero-pads outside the array. Bins with no events in the window produce NaN.

    Examples
    --------
    >>> import numpy as np
    >>> n_time, n_cells = 100, 10
    >>> metric = np.random.rand(n_time, n_cells)
    >>> metric[::2, :] = np.nan  # Sparse spikes
    >>> time = np.linspace(0, 1, n_time)
    >>> running_avg, time_out = compute_running_average(metric, time, window_size=0.1)
    >>> running_avg.shape
    (100,)
    """
    n_time_pts = len(time)

    if event_times is not None and event_values is not None:
        event_times = np.asarray(event_times, dtype=np.float64)
        event_values = np.asarray(event_values, dtype=np.float64)

        valid_events = ~np.isnan(event_values)
        event_times = event_times[valid_events]
        event_values = event_values[valid_events]

        if len(event_times) == 0:
            return np.full(n_time_pts, np.nan), time.copy()

        sort_ind = np.argsort(event_times)
        sorted_times = event_times[sort_ind]
        sorted_values = event_values[sort_ind]
        cumsum = np.concatenate(([0.0], np.cumsum(sorted_values)))

        half_window = window_size / 2.0
        starts = np.searchsorted(sorted_times, time - half_window, side="left")
        stops = np.searchsorted(sorted_times, time + half_window, side="right")
        counts = stops - starts
        sums = cumsum[stops] - cumsum[starts]

        running_avg = np.full(n_time_pts, np.nan)
        has_events = counts > 0
        running_avg[has_events] = sums[has_events] / counts[has_events]

        return running_avg, time.copy()

    # Step 1: Compute per-bin event sum and count across cells
    # Each non-NaN entry is one spike event
    event_values = np.where(np.isnan(metric), 0.0, metric)
    event_counts = np.where(np.isnan(metric), 0.0, 1.0)

    # Sum across cells: total metric value and event count per time bin
    bin_sum = event_values.sum(axis=1)  # (n_time,)
    bin_count = event_counts.sum(axis=1)  # (n_time,)

    # Step 2: Estimate time bin width from the time array
    if len(time) > 1:
        dt = float(np.median(np.diff(time)))
    else:
        dt = 1.0

    # Step 3: Convert window size from seconds to number of bins
    window_bins = max(1, int(np.round(window_size / dt)))

    # Step 4: Apply boxcar filter to numerator and denominator separately
    # Using mode="constant" (zero-pad) so edge bins have fewer events, not
    # inflated counts from nearest-value extension.
    windowed_sum = uniform_filter1d(bin_sum, size=window_bins, mode="constant")
    windowed_count = uniform_filter1d(bin_count, size=window_bins, mode="constant")

    # Step 5: Divide to get event-weighted average; NaN where no events
    running_avg = np.full(n_time_pts, np.nan)
    has_events = windowed_count > 0
    running_avg[has_events] = windowed_sum[has_events] / windowed_count[has_events]

    return running_avg, time.copy()


def _get_spike_events_from_counts(
    spike_counts: NDArray[np.int64],
    time: NDArray[np.float64] | None = None,
) -> tuple[NDArray[np.intp], NDArray[np.intp], NDArray[np.float64] | None]:
    """Expand binned spike counts into one event per spike."""
    spike_time_ind, spike_cell_ind = np.nonzero(spike_counts)
    counts = spike_counts[spike_time_ind, spike_cell_ind].astype(np.intp)

    spike_time_ind = np.repeat(spike_time_ind, counts).astype(np.intp)
    spike_cell_ind = np.repeat(spike_cell_ind, counts).astype(np.intp)
    event_times = None if time is None else np.asarray(time, dtype=np.float64)[spike_time_ind]

    return spike_time_ind, spike_cell_ind, event_times


def _get_spike_events_from_spike_times(
    spike_times: list[NDArray[np.float64]],
    time: NDArray[np.float64],
) -> tuple[NDArray[np.intp], NDArray[np.intp], NDArray[np.float64]]:
    """Map exact spike timestamps to predictive-posterior time indices."""
    time = np.asarray(time, dtype=np.float64)
    spike_time_inds = []
    spike_cell_inds = []
    event_times = []

    for cell_ind, cell_spike_times in enumerate(spike_times):
        cell_spike_times = np.asarray(cell_spike_times, dtype=np.float64)
        in_bounds = (cell_spike_times >= time[0]) & (cell_spike_times <= time[-1])
        cell_event_times = cell_spike_times[in_bounds]
        cell_time_inds = np.searchsorted(time, cell_event_times, side="right") - 1
        cell_time_inds = np.clip(cell_time_inds, 0, len(time) - 1)

        spike_time_inds.append(cell_time_inds.astype(np.intp))
        spike_cell_inds.append(np.full(len(cell_event_times), cell_ind, dtype=np.intp))
        event_times.append(cell_event_times)

    if not event_times:
        return (
            np.empty(0, dtype=np.intp),
            np.empty(0, dtype=np.intp),
            np.empty(0, dtype=np.float64),
        )

    spike_time_ind = np.concatenate(spike_time_inds)
    spike_cell_ind = np.concatenate(spike_cell_inds)
    event_time = np.concatenate(event_times)
    sort_ind = np.argsort(event_time)

    return spike_time_ind[sort_ind], spike_cell_ind[sort_ind], event_time[sort_ind]


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


def extract_place_fields_concat(
    model: Any,
) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
    """Concatenate per-observation-model place fields + the interior mask.

    Returns the place fields aligned with the predictive posterior's
    full ``state_bins`` axis (i.e. before the interior-mask filter):
    one ``(n_cells, n_state_bins_full)`` array stacked across the
    model's observation models, plus the matching boolean
    ``is_track_interior_state_bins_`` mask. Callers that only need
    the interior bins do ``place_fields[:, interior_mask]``.

    Used by both ``compute_model_diagnostics`` (interior-only) and
    the interactive cache builder (which keeps both so the viewer
    can reconstruct the non-interior NaN columns).
    """
    place_fields = np.concatenate(
        [
            extract_place_fields(
                model,
                environment_name=obs.environment_name,
                encoding_group=obs.encoding_group,
            )[0]
            for obs in model.observation_models
        ],
        axis=1,
    )
    interior_mask: NDArray[np.bool_] = np.asarray(model.is_track_interior_state_bins_, dtype=bool)
    return place_fields, interior_mask


def compute_per_cell_diagnostics(
    predictive_posterior: NDArray[np.float64],
    spike_counts: NDArray[np.int64],
    place_fields: NDArray[np.float64],
    coverage: float = 0.95,
    spike_times: list[NDArray[np.float64]] | None = None,
    time: NDArray[np.float64] | None = None,
    include_dense_matrices: bool = True,
) -> dict[str, NDArray[np.floating] | NDArray[np.intp]]:
    """Compute per-cell diagnostic metrics for model checking.

    Computes HPD overlap, KL divergence, and spike probability ranking for each
    spike event. Matrix outputs are retained for backward-compatible plotting,
    and event arrays preserve one row per spike with exact timestamps when
    ``spike_times`` and ``time`` are supplied.

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
    spike_times : list of np.ndarray, optional
        Exact spike timestamps for each cell. If supplied, diagnostics are
        computed per spike event and plotted at exact spike times.
    time : np.ndarray, optional
        Decoder time grid used to map spike timestamps to predictive posterior
        rows. Required when ``spike_times`` is supplied.
    include_dense_matrices : bool, default True
        Forwarded to ``compute_per_cell_diagnostics_from_rates``. Set False
        when only the per-spike event arrays are needed (avoids the
        ``(n_time, n_cells)`` allocations, which can be hundreds of MB
        for full-session real-data builds).

    Returns
    -------
    diagnostics : dict[str, np.ndarray]
        Always contains:
        - 'event_time': shape (n_spikes,), exact spike time when available
        - 'event_hpd_overlap': shape (n_spikes,), one value per spike
        - 'event_kl_divergence': shape (n_spikes,), one value per spike
        - 'event_spike_prob': shape (n_spikes,), one value per spike

        When ``include_dense_matrices`` (the default), additionally:
        - 'hpd_overlap': shape (n_time, n_cells), NaN when cell has no spike
        - 'kl_divergence': shape (n_time, n_cells), NaN when cell has no spike
        - 'spike_prob': shape (n_time, n_cells), NaN when cell has no spike

    Notes
    -----
    The likelihood P(k=1 | position) is computed once for each observed spike.
    Multiple spikes from the same cell in the same decoder bin contribute
    multiple event rows rather than being collapsed into one binned count.

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

    event_times: NDArray[np.float64] | None
    if spike_times is not None:
        if time is None:
            raise ValueError("time must be provided when spike_times is provided")
        spike_time_ind, spike_cell_ind, event_times = _get_spike_events_from_spike_times(
            spike_times, time
        )
    else:
        spike_time_ind, spike_cell_ind, event_times = _get_spike_events_from_counts(
            spike_counts,
            time,
        )

    # Use shared function from analysis.py
    result = compute_per_cell_diagnostics_from_rates(
        predictive_posterior,
        place_fields.T,  # (n_bins, n_cells)
        spike_time_ind.astype(np.intp),
        spike_cell_ind.astype(np.intp),
        coverage=coverage,
        include_dense_matrices=include_dense_matrices,
    )

    result_float = dict(result)
    cast_keys = ["event_hpd_overlap", "event_kl_divergence", "event_spike_prob"]
    if include_dense_matrices:
        cast_keys.extend(["hpd_overlap", "kl_divergence", "spike_prob", "per_spike_likelihood"])
    for key in cast_keys:
        result_float[key] = np.asarray(result_float[key], dtype=np.float64)
    if event_times is not None:
        result_float["event_time"] = event_times.astype(np.float64)
    return result_float


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
    elif posterior_type == "acausal":
        posterior_da = results.acausal_posterior
    else:
        raise ValueError(
            f"Invalid posterior_type: {posterior_type}. Must be 'predictive' or 'acausal'."
        )

    # Drop NaN state bins (e.g., track interior only)
    posterior_da = posterior_da.dropna("state_bins")

    # Multi-state models encode (state, position) in state_bins as a
    # MultiIndex; single-state models use a plain Index. Branch on
    # the index type rather than catching a generic unstack failure,
    # which would silently treat a malformed multi-state model as
    # single-state and produce a per-state slice labeled as marginal.
    state_bins_index = posterior_da.indexes["state_bins"]
    if isinstance(state_bins_index, pd.MultiIndex):
        try:
            unstacked = posterior_da.unstack("state_bins")
        except (ValueError, KeyError) as e:
            raise ValueError(
                "Failed to unstack the state_bins MultiIndex on the "
                "decoder posterior; the index is malformed (likely "
                "duplicate (state, position) entries) and cannot be "
                f"marginalized. Underlying error: {e}"
            ) from e
        marginalized = unstacked.sum("state") if "state" in unstacked.dims else unstacked
        posterior: NDArray[np.float64] = np.asarray(marginalized.values)
    else:
        # Single-state model: no states to sum over.
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
    position : np.ndarray, shape (n_time,) or (n_time, n_dims)
        Position values. 1D arrays (linear position) are reshaped
        to (n_time, 1) before being passed to the model.
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
    sorted_spikes_algorithm_params = {
        "block_size": 10000,
        "position_std": np.sqrt(12.5),
    }
    continuous_model = SortedSpikesDecoder(
        environments=[environment],
        sorted_spikes_algorithm_params=sorted_spikes_algorithm_params,
    )
    continuous_model.fit(position=position_2d, spike_times=spike_times, position_time=time)

    # Fit ContFrag model
    contfrag_model = ContFragSortedSpikesClassifier(
        environments=[environment],
        sorted_spikes_algorithm_params=sorted_spikes_algorithm_params,
    )
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
    spike_times: list[NDArray[np.float64]] | None = None,
) -> dict[str, NDArray[np.floating] | NDArray[np.intp]]:
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
    spike_times : list of np.ndarray, optional
        Exact spike timestamps for each cell. If supplied, diagnostics are
        computed as one event per spike instead of one event per nonzero bin.

    Returns
    -------
    diagnostics : dict[str, np.ndarray]
        Dictionary with keys:
        - 'hpd_overlap': shape (n_time, n_cells), NaN where no spike
        - 'kl_divergence': shape (n_time, n_cells), NaN where no spike
        - 'spike_prob': shape (n_time, n_cells), NaN where no spike
        - 'per_spike_likelihood': shape (n_spikes, n_bins), normalized
          likelihood distribution for each individual spike event
        - 'event_time_ind': shape (n_spikes,), time index for each spike
        - 'event_cell_ind': shape (n_spikes,), cell index for each spike
        - 'event_hpd_overlap': shape (n_spikes,), per-spike HPD overlap
        - 'event_kl_divergence': shape (n_spikes,), per-spike KL divergence
        - 'event_spike_prob': shape (n_spikes,), per-spike spike probability
        - 'event_time': shape (n_spikes,), exact spike timestamps in seconds
          (present whenever ``time`` is supplied; with ``spike_times`` these
          are the original timestamps, otherwise they are the decoder time
          grid values at each event's time index)

    Examples
    --------
    >>> # Requires fitted model and decoding results
    >>> # diagnostics = compute_model_diagnostics(model, results, spike_counts, time)
    >>> # diagnostics['hpd_overlap'].shape  # (n_time, n_cells)
    """
    # Extract place fields concatenated across observation models (one
    # per state for multi-state classifiers) and filter to interior bins.
    # ``predictive_posterior.dropna(state_bins)`` below produces a matching
    # interior-only column count because the model uses one
    # ``is_track_interior_state_bins_`` mask consistently across the
    # encoding model and ``predict()`` output.
    place_fields_full, interior_mask = extract_place_fields_concat(model)
    place_fields = place_fields_full[:, interior_mask]

    # Get predictive posterior, dropping NaN state bins (non-interior bins)
    predictive_posterior = results.predictive_posterior.dropna(dim="state_bins").values

    # Compute diagnostics using actual spike counts
    # place_fields are already in spikes per time bin from non_local_detector
    diagnostics = compute_per_cell_diagnostics(
        predictive_posterior,
        spike_counts,
        place_fields,
        spike_times=spike_times,
        time=time,
    )

    return diagnostics
