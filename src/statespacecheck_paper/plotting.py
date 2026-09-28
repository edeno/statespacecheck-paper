"""Plotting utilities for state space model diagnostics.

This module provides functions for creating publication-ready figures showing
diagnostic metrics and misfit examples for state space models.
"""

from __future__ import annotations

from typing import overload

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from numpy.typing import NDArray

from statespacecheck_paper.style import CMAP_LIKELIHOOD


@overload
def negative_log_pvalue(x: NDArray[np.floating]) -> NDArray[np.float64]: ...


@overload
def negative_log_pvalue(x: float) -> np.float64: ...


def negative_log_pvalue(
    x: NDArray[np.floating] | float,
) -> NDArray[np.float64] | np.float64:
    """Return the exact natural-log display transform ``-log(p)``.

    Predictive p-values are shown on a ``-log(p)`` scale (natural log) so that
    higher values indicate worse fit; Figures 3, 4, and the interactive viewer
    share this transform. NaN is preserved as structural missingness. A zero
    p-value cannot be represented on a finite log axis and raises instead of
    being silently capped.

    Parameters
    ----------
    x : NDArray[np.float64] or float
        Probability value(s) to transform.

    Returns
    -------
    NDArray[np.float64] or np.float64
        ``-log(x)`` with the same shape as ``x``.

    Raises
    ------
    ValueError
        If a present value is not in ``(0, 1]``.
    """
    values = np.asarray(x, dtype=np.float64)
    present = ~np.isnan(values)
    if np.any(~np.isfinite(values[present])) or np.any(values[present] <= 0.0):
        raise ValueError("Predictive p-values must be finite and strictly positive for -log(p)")
    if np.any(values[present] > 1.0 + 1e-9):
        raise ValueError("Predictive p-values must not exceed 1")
    # Only correct floating-point overshoot within the validated tolerance.
    values = np.minimum(values, 1.0)
    transformed = -np.log(values)
    if np.isscalar(x):
        return np.float64(transformed)
    return transformed


def extract_contiguous_regions(
    mask: NDArray[np.bool_],
    x: NDArray[np.floating],
) -> list[tuple[float, float]]:
    """Extract contiguous True regions from a boolean mask.

    Parameters
    ----------
    mask : np.ndarray, shape (n_points,)
        Boolean mask indicating region membership.
    x : np.ndarray, shape (n_points,)
        Position values corresponding to mask.

    Returns
    -------
    regions : list[tuple[float, float]]
        List of (start, end) tuples for each contiguous region.

    Examples
    --------
    >>> import numpy as np
    >>> x = np.linspace(0, 10, 100)
    >>> mask = (x > 2) & (x < 8)
    >>> regions = extract_contiguous_regions(mask, x)
    >>> len(regions)
    1
    """
    if not np.any(mask):
        return []

    # Pad with False to detect edges at boundaries
    padded = np.concatenate([[False], mask, [False]])
    diff = np.diff(padded.astype(int))

    # Rising edges (0->1) mark region starts, falling edges (1->0) mark ends
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0] - 1  # -1 to get last True index

    return [(float(x[s]), float(x[e])) for s, e in zip(starts, ends, strict=True)]


def plot_likelihood_columns(
    ax: Axes,
    likelihood: NDArray[np.floating],
    has_spikes: NDArray[np.bool_],
    n_time: int,
    extent: tuple[float, float, float, float] | None = None,
    cmap: str = CMAP_LIKELIHOOD,
) -> None:
    """Render likelihood distributions as colored columns at spike times.

    Each spike-time column is drawn with a guaranteed minimum width so it remains
    visible even when time bins outnumber pixels. Row-wise normalization ensures
    the spatial structure (where the likelihood peaks) is visible regardless of
    absolute magnitude. Used by both simulated (Figure 3) and real data (Figure 4)
    likelihood panels.

    Parameters
    ----------
    ax : Axes
        Matplotlib axes to plot on.
    likelihood : NDArray, shape (n_time_shown, n_bins)
        Likelihood distribution at each time. Only rows where ``has_spikes``
        is True are rendered.
    has_spikes : NDArray, shape (n_time_shown,)
        Boolean mask: True at time bins with at least one spike.
    n_time : int
        Total number of time bins (used to compute minimum column width).
    extent : tuple of float, optional
        (x0, x1, y0, y1) extent for positioning columns. If None, uses
        integer bin indices (0, n_time-1, 0, n_bins-1).
    cmap : str, default CMAP_LIKELIHOOD
        Colormap for the likelihood columns.

    Raises
    ------
    ValueError
        If any rendered likelihood row contains NaN/infinity or negative values.
    """
    n_bins = likelihood.shape[1]
    cmap_obj = plt.colormaps[cmap]

    if extent is None:
        x0, x1 = 0.0, float(n_time - 1)
        y0, y1 = 0.0, float(n_bins - 1)
    else:
        x0, x1, y0, y1 = extent

    # Minimum column half-width in data coordinates so each spike is >= 1 pixel
    data_range = x1 - x0
    min_half_width = max(data_range / 3000.0, data_range / n_time)

    spike_times = np.where(has_spikes)[0]
    for idx in spike_times:
        lik_row = likelihood[idx]
        if not np.all(np.isfinite(lik_row)):
            raise ValueError(f"likelihood row {idx} contains NaN or infinity")
        if np.any(lik_row < 0.0):
            raise ValueError(f"likelihood row {idx} contains negative values")
        # A finite flat likelihood is meaningful: every position has equal
        # support, so use the colormap midpoint without inventing spatial
        # structure. Undefined rows fail above.
        rmin, rmax = float(np.min(lik_row)), float(np.max(lik_row))
        if rmax == rmin:
            normed = np.full_like(lik_row, 0.5)
        else:
            normed = (lik_row - rmin) / (rmax - rmin)
        rgba_col = cmap_obj(normed)

        # Map time index to data coordinate
        if n_time > 1:
            t = x0 + (x1 - x0) * idx / (likelihood.shape[0] - 1)
        else:
            t = (x0 + x1) / 2.0

        # Draw as a thin image strip with guaranteed minimum width
        ax.imshow(
            rgba_col[np.newaxis, :, :].transpose(1, 0, 2),
            aspect="auto",
            origin="lower",
            extent=(t - min_half_width, t + min_half_width, y0, y1),
            interpolation="nearest",
        )
