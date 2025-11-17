"""Analysis utilities for real neural data.

This module provides utilities for analyzing real neural recordings, including
spike rate estimation, sustained event detection, and temporal smoothing.

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

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import gaussian_filter1d, label


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
