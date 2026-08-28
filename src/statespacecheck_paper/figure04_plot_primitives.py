"""Shared low-level Figure-4 plotting helpers.

Small building blocks used across the Figure-4 track, raster, and diagnostic
panels: the annotation GID constants, the half-pixel ``imshow`` extent, the
scale-bar drawer, and the distribution heatmap renderer. The shared ``-log(p)``
display transform lives in :mod:`plotting` (``negative_log_pvalue``).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.axes import Axes
from numpy.typing import NDArray

from statespacecheck_paper.style import (
    CMAP_POSTERIOR,
    COLORS,
)

ANIMAL_POSITION_LABEL_GID = "animal-position-label"
THRESHOLD_LABEL_GID = "threshold-label"
WORSE_FIT_LABEL_GID = "worse-fit-label"


def compute_half_pixel_extent(
    time_coords: NDArray[np.float64], pos_coords: NDArray[np.float64]
) -> tuple[float, float, float, float]:
    """Return an ``imshow`` extent padded by half a bin on each side.

    Produces ``(t_lo, t_hi, p_lo, p_hi)`` so that ``imshow`` centres pixels on
    their time/position coordinates. Requires at least two coordinates along
    each axis: a single coordinate has no inferable bin width, so this raises
    rather than silently emitting a zero-width extent.

    Parameters
    ----------
    time_coords : NDArray[np.float64], shape (n_time,)
        Monotonic time coordinates of the pixel grid.
    pos_coords : NDArray[np.float64], shape (n_position,)
        Monotonic position coordinates of the pixel grid.

    Returns
    -------
    tuple[float, float, float, float]
        ``(t0 - dt, t1 + dt, p0 - dp, p1 + dp)`` half-pixel extent.

    Raises
    ------
    ValueError
        If either coordinate array has fewer than two elements.
    """
    if len(time_coords) < 2 or len(pos_coords) < 2:
        raise ValueError(
            "compute_half_pixel_extent needs >=2 coordinates along each axis to infer "
            f"a bin width; got {len(time_coords)} time and {len(pos_coords)} "
            "position coordinates."
        )
    t0, t1 = float(time_coords[0]), float(time_coords[-1])
    p0, p1 = float(pos_coords[0]), float(pos_coords[-1])
    dt = (t1 - t0) / (len(time_coords) - 1) / 2
    dp = (p1 - p0) / (len(pos_coords) - 1) / 2
    return (t0 - dt, t1 + dt, p0 - dp, p1 + dp)


def add_scalebar(
    ax: Axes,
    length: float,
    label: str,
    loc: str = "lower right",
    pad: float = 0.1,
    fontsize: int = 8,
) -> None:
    """Add a scale bar to an axes.

    Parameters
    ----------
    ax : Axes
        The axes to add the scale bar to.
    length : float
        Length of the scale bar in data units.
    label : str
        Label text for the scale bar.
    loc : str, default "lower right"
        Location for the scale bar.
    pad : float, default 0.1
        Padding from edges as fraction of axes size.
    fontsize : int, default 7
        Font size for the label.
    """
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    x_range = xlim[1] - xlim[0]
    y_range = ylim[1] - ylim[0]

    if "right" in loc:
        x_start = xlim[1] - pad * x_range - length
    else:
        x_start = xlim[0] + pad * x_range

    if "lower" in loc:
        y_pos = ylim[0] + pad * y_range
    else:
        y_pos = ylim[1] - pad * y_range

    ax.plot([x_start, x_start + length], [y_pos, y_pos], "k-", linewidth=2, clip_on=False)
    ax.text(
        x_start + length / 2,
        y_pos - 0.03 * y_range,
        label,
        ha="center",
        va="top",
        fontsize=fontsize,
    )


def plot_distribution_heatmap(
    ax: Axes,
    distribution_da: xr.DataArray,
    time: NDArray[np.float64] | pd.Index,
    position: NDArray[np.float64],
    time_slice_ind: slice,
    show_position: bool = True,
    cmap: str = CMAP_POSTERIOR,
) -> None:
    """Plot a distribution heatmap with optional position overlay.

    Parameters
    ----------
    ax : Axes
        The axes to plot on.
    distribution_da : xr.DataArray
        Distribution data array with state_bins dimension.
    time : np.ndarray or pd.Index
        Time values.
    position : np.ndarray, shape (n_time,)
        Animal position values.
    time_slice_ind : slice
        Time slice indices to plot.
    show_position : bool, default True
        Whether to show position overlay.
    cmap : str, default CMAP_POSTERIOR
        Colormap for the heatmap.
    """
    # Drop NaN bins (spatial bins that are always NaN)
    distribution_da = distribution_da.dropna("state_bins", how="all")

    # Plot distribution heatmap. Multi-state models encode
    # (state, position) in state_bins as a MultiIndex; single-state
    # models leave it as a plain Index without a separate ``position``
    # coord. Branch on the index type so a malformed MultiIndex fails
    # loud, and single-state data plots against state_bins directly
    # rather than dying inside xarray on a missing ``position``.
    if isinstance(distribution_da.indexes["state_bins"], pd.MultiIndex):
        try:
            unstacked = distribution_da.unstack("state_bins")
        except (ValueError, KeyError, TypeError) as e:
            raise ValueError(
                "Failed to unstack state_bins MultiIndex on the "
                "distribution heatmap; the index is malformed and cannot "
                f"be marginalized. Underlying error: {e}"
            ) from e
        marginalized = (
            unstacked.sum("state", skipna=False) if "state" in unstacked.dims else unstacked
        )
        sliced_data = marginalized.isel(time=time_slice_ind)
        if sliced_data.notnull().any():
            sliced_data.plot(
                x="time",
                y="position",
                ax=ax,
                add_colorbar=False,
                robust=True,
                cmap=cmap,
                rasterized=True,
            )
        else:
            raise ValueError("Predictive-posterior slice contains no plottable values")
    else:
        # Single-state model: no separate ``position`` axis. Plot
        # against the state_bins axis directly so the figure still
        # renders something meaningful.
        sliced_data = distribution_da.isel(time=time_slice_ind)
        if sliced_data.notnull().any():
            sliced_data.plot(
                x="time",
                ax=ax,
                add_colorbar=False,
                robust=True,
                cmap=cmap,
                rasterized=True,
            )
        else:
            raise ValueError("Predictive-posterior slice contains no plottable values")

    # Overlay animal position
    if show_position:
        time_arr = np.asarray(time)
        ax.scatter(
            time_arr[time_slice_ind],
            position[time_slice_ind],
            c=COLORS["ground_truth"],
            s=1,
            alpha=0.85,
            rasterized=True,
            label="True position",
        )
