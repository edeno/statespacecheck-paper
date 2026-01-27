"""Plotting utilities for real neural data analysis.

This module provides specialized plotting functions for visualizing neural decoding
results, HPD overlap diagnostics, and model checking on real experimental data.

Examples
--------
>>> import numpy as np
>>> import matplotlib.pyplot as plt
>>> from statespacecheck_paper.real_data_plotting import plot_overlap_trace
>>> time = np.linspace(0, 10, 1000)
>>> overlap = np.random.rand(1000)
>>> fig, ax = plt.subplots()
>>> plot_overlap_trace(time, slice(0, 500), overlap, ax=ax)
"""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from non_local_detector.model_checking.highest_posterior_density import (
    get_highest_posterior_threshold,
)
from numpy.typing import NDArray
from scipy.ndimage import label
from track_linearization import plot_graph_as_1D

from statespacecheck_paper.style import COLORS


def plot_raster(
    spike_times: list[NDArray[np.float64]],
    time_slice: slice,
    ax: Axes | None = None,
    sort_order: NDArray[np.int64] | None = None,
    **eventplot_kwargs: Any,
) -> None:
    """Plot spike raster for a given time slice.

    Parameters
    ----------
    spike_times : list[np.ndarray]
        List of spike time arrays, one per neuron.
    time_slice : slice
        Time slice with start and stop attributes.
    ax : plt.Axes, optional
        Axes to plot on. If None, uses current axes.
    sort_order : np.ndarray, optional
        Indices to reorder neurons.
    **eventplot_kwargs
        Additional arguments passed to ax.eventplot().

    Examples
    --------
    >>> spike_times = [np.array([0.1, 0.2, 0.5]), np.array([0.15, 0.3])]
    >>> time_slice = slice(0.0, 1.0)
    >>> plot_raster(spike_times, time_slice)
    """
    if ax is None:
        ax = plt.gca()

    time_slice_spike_times = [
        neuron_spike_times[
            (neuron_spike_times >= time_slice.start) & (neuron_spike_times < time_slice.stop)
        ]
        for neuron_spike_times in spike_times
    ]

    if sort_order is not None:
        time_slice_spike_times = [time_slice_spike_times[i] for i in sort_order]

    ax.eventplot(
        time_slice_spike_times,
        linelengths=0.5,
        colors="black",
        rasterized=True,
        **eventplot_kwargs,
    )
    ax.set_ylabel("Neuron")
    ax.set_xlabel("Time")


def plot_posterior(
    posterior: xr.DataArray,
    time: NDArray[np.float64] | pd.Index,
    position: NDArray[np.float64],
    time_slice_ind: slice,
    ax: Axes | None = None,
    title: str | None = None,
    scatter_kwargs: dict[str, Any] | None = None,
    **plot_kwargs: Any,
) -> Any:
    """Plot posterior probability as an image with position trace overlay.

    Parameters
    ----------
    posterior : xr.DataArray
        Posterior distribution over state bins and time.
    time : np.ndarray or pd.Index
        Time values.
    position : np.ndarray, shape (n_time,)
        Actual position values.
    time_slice_ind : slice
        Time slice indices to plot.
    ax : plt.Axes, optional
        Axes to plot on. If None, uses current axes.
    title : str, optional
        Title for the plot.
    scatter_kwargs : dict, optional
        Keyword arguments for position scatter plot.
    **plot_kwargs
        Additional arguments passed to posterior.plot().

    Returns
    -------
    im : matplotlib artist
        The plot artist (QuadMesh, AxesImage, etc.) returned by xarray.plot().

    Examples
    --------
    >>> import xarray as xr
    >>> posterior = xr.DataArray(
    ...     np.random.rand(100, 50),
    ...     dims=["time", "position"],
    ...     coords={"time": np.arange(100), "position": np.arange(50)}
    ... )
    >>> time = np.arange(100)
    >>> position = np.random.randint(0, 50, 100)
    >>> im = plot_posterior(posterior, time, position, slice(0, 50))
    """
    if ax is None:
        ax = plt.gca()

    im = (
        posterior.isel(time=time_slice_ind)
        .unstack("state_bins")
        .sum("state")
        .plot(
            x="time",
            y="position",
            ax=ax,
            add_colorbar=False,
            robust=True,
            cmap="bone_r",
            rasterized=True,
            **plot_kwargs,
        )
    )

    if scatter_kwargs is None:
        scatter_kwargs = {
            "color": "magenta",
            "s": 1,
            "rasterized": True,
            "clip_on": False,
        }

    ax.scatter(
        time[time_slice_ind],
        position[time_slice_ind],
        **scatter_kwargs,
    )

    if title:
        ax.set_title(title)
    ax.set_ylabel("Position")

    return im


def plot_overlap_regions(
    ax: Axes,
    time: NDArray[np.float64] | pd.Index,
    time_slice_ind: slice,
    hpd_overlap: NDArray[np.float64],
    threshold: float = 0.2,
    color: str = "tab:blue",
    alpha: float = 0.5,
) -> None:
    """Highlight regions where HPD overlap is below threshold.

    Parameters
    ----------
    ax : plt.Axes
        Axes to plot on.
    time : np.ndarray or pd.Index
        Time values.
    time_slice_ind : slice
        Time slice indices to plot.
    hpd_overlap : np.ndarray, shape (n_time,)
        HPD overlap values.
    threshold : float, default=0.2
        Threshold below which to highlight regions.
    color : str, default="tab:blue"
        Color for highlighted regions.
    alpha : float, default=0.5
        Transparency of highlighted regions.

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> fig, ax = plt.subplots()
    >>> time = np.linspace(0, 10, 1000)
    >>> overlap = np.random.rand(1000)
    >>> plot_overlap_regions(ax, time, slice(0, 500), overlap, threshold=0.3)
    """
    labels_array, n_labels = label(hpd_overlap[time_slice_ind] < threshold)

    for label_id in range(1, n_labels + 1):
        bad_overlap_ind = np.where(labels_array == label_id)[0]
        ax.axvspan(
            time[time_slice_ind][bad_overlap_ind[0]],
            time[time_slice_ind][bad_overlap_ind[-1]],
            color=color,
            alpha=alpha,
        )


def plot_overlap_trace(
    time: NDArray[np.float64] | pd.Index,
    time_slice_ind: slice,
    overlaps: NDArray[np.float64] | list[NDArray[np.float64]],
    labels: str | list[str] | None = None,
    ax: Axes | None = None,
    **plot_kwargs: Any,
) -> None:
    """Plot HPD overlap traces over time.

    Parameters
    ----------
    time : np.ndarray or pd.Index
        Time values.
    time_slice_ind : slice
        Time slice indices to plot.
    overlaps : np.ndarray or list[np.ndarray]
        HPD overlap values. Can be single array or list of arrays.
    labels : str or list[str], optional
        Legend labels for each trace.
    ax : plt.Axes, optional
        Axes to plot on. If None, uses current axes.
    **plot_kwargs
        Additional arguments passed to ax.plot().

    Examples
    --------
    >>> time = np.linspace(0, 10, 1000)
    >>> overlap1 = np.random.rand(1000)
    >>> overlap2 = np.random.rand(1000)
    >>> plot_overlap_trace(
    ...     time, slice(0, 500), [overlap1, overlap2],
    ...     labels=["Model 1", "Model 2"]
    ... )
    """
    if ax is None:
        ax = plt.gca()

    if isinstance(overlaps, list | tuple):
        # Handle labels - create None list if labels not provided
        if labels is None:
            labels_list: list[str | None] = [None] * len(overlaps)
        elif isinstance(labels, str):
            labels_list = [labels]
        else:
            labels_list = list(labels)  # Convert to list to satisfy type checker

        for overlap, label in zip(overlaps, labels_list, strict=False):
            ax.plot(
                time[time_slice_ind],
                overlap[time_slice_ind],
                label=label,
                **plot_kwargs,
            )
    else:
        ax.plot(
            time[time_slice_ind],
            overlaps[time_slice_ind],
            label=labels,
            **plot_kwargs,
        )

    if labels:
        ax.legend()
    ax.set_ylabel("HPD Overlap")
    ax.set_xlabel("Time")


def plot_acausal_state_prob(
    results: xr.Dataset,
    time_slice_ind: slice,
    ax: Axes | None = None,
    title: str | None = None,
    **plot_kwargs: Any,
) -> Any:
    """Plot acausal state probabilities over time.

    Parameters
    ----------
    results : xr.Dataset
        Dataset containing acausal_state_probabilities.
    time_slice_ind : slice
        Time slice indices to plot.
    ax : plt.Axes, optional
        Axes to plot on. If None, uses current axes.
    title : str, optional
        Title for the plot.
    **plot_kwargs
        Additional arguments passed to plot().

    Returns
    -------
    im : plot object
        The plot object from xarray.

    Examples
    --------
    >>> import xarray as xr
    >>> results = xr.Dataset({
    ...     "acausal_state_probabilities": xr.DataArray(
    ...         np.random.rand(100, 2),
    ...         dims=["time", "states"],
    ...         coords={"time": np.arange(100), "states": ["A", "B"]}
    ...     )
    ... })
    >>> plot_acausal_state_prob(results, slice(0, 50))
    """
    if ax is None:
        ax = plt.gca()

    im = results.acausal_state_probabilities.isel(time=time_slice_ind).plot(
        x="time",
        hue="states",
        ax=ax,
        rasterized=True,
        **plot_kwargs,
    )

    ax.set_ylim((0.0, 1.05))
    if title:
        ax.set_title(title)
    ax.set_ylabel("State Prob.")

    return im


def plot_model_checking(
    time_slice_ind: slice,
    time: NDArray[np.float64] | pd.Index,
    position: NDArray[np.float64],
    spike_times: list[NDArray[np.float64]],
    cont_model: Any,
    cont_frag_model: Any,
    cont_results: xr.Dataset,
    cont_frag_results: xr.Dataset,
    cont_hpd_overlap: NDArray[np.float64],
    cont_frag_hpd_overlap: NDArray[np.float64],
    overlap_threshold: float = 0.2,
) -> tuple[Figure, NDArray[np.object_]]:
    """Create comprehensive model checking figure comparing two models.

    Parameters
    ----------
    time_slice_ind : slice
        Time slice indices to plot.
    time : np.ndarray or pd.Index
        Time values.
    position : np.ndarray, shape (n_time,)
        Actual position values.
    spike_times : list[np.ndarray]
        List of spike time arrays.
    cont_model : Model
        Continuous decoder model.
    cont_frag_model : Model
        Continuous-fragmented decoder model.
    cont_results : xr.Dataset
        Results from continuous model.
    cont_frag_results : xr.Dataset
        Results from continuous-fragmented model.
    cont_hpd_overlap : np.ndarray, shape (n_time,)
        HPD overlap for continuous model.
    cont_frag_hpd_overlap : np.ndarray, shape (n_time,)
        HPD overlap for continuous-fragmented model.
    overlap_threshold : float, default=0.2
        Threshold for highlighting poor overlap regions.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object.
    axes : np.ndarray[plt.Axes]
        Array of axes objects.

    Examples
    --------
    >>> # Requires real decoder models and results
    >>> # See scripts/run_decoding.py for complete example
    """
    time_slice = slice(time[time_slice_ind.start], time[time_slice_ind.stop])
    sort_order = np.argsort(
        cont_model.environments[0]
        .place_bin_centers_[cont_model.encoding_model_[("", 0)]["place_fields"].argmax(axis=1)]
        .squeeze()
    )

    fig, axes = plt.subplots(4, 1, figsize=(7, 8), sharex=True, constrained_layout=True)

    plot_posterior(
        cont_results.predictive_posterior,
        time,
        position,
        time_slice_ind,
        ax=axes[0],
        title="Continuous Decoder",
    )
    plot_overlap_regions(
        axes[0],
        time,
        time_slice_ind,
        cont_hpd_overlap,
        threshold=overlap_threshold,
        color="tab:blue",
    )

    plot_posterior(
        cont_frag_results.predictive_posterior,
        time,
        position,
        time_slice_ind,
        ax=axes[1],
        title="Continuous Fragmented Decoder",
    )
    plot_overlap_regions(
        axes[1],
        time,
        time_slice_ind,
        cont_frag_hpd_overlap,
        threshold=overlap_threshold,
        color="tab:orange",
    )

    plot_raster(spike_times, time_slice, ax=axes[2], sort_order=sort_order)
    plot_overlap_regions(
        axes[2],
        time,
        time_slice_ind,
        cont_hpd_overlap,
        threshold=overlap_threshold,
        color="tab:blue",
    )
    plot_overlap_regions(
        axes[2],
        time,
        time_slice_ind,
        cont_frag_hpd_overlap,
        threshold=overlap_threshold,
        color="tab:orange",
    )

    plot_overlap_trace(
        time,
        time_slice_ind,
        [cont_hpd_overlap, cont_frag_hpd_overlap],
        labels=["Continuous", "Continuous Fragmented"],
        ax=axes[3],
    )

    return fig, axes


def plot_single_model_checking(
    time_slice_ind: slice,
    time: NDArray[np.float64] | pd.Index,
    position: NDArray[np.float64],
    spike_times: list[NDArray[np.float64]],
    model: Any,
    results: xr.Dataset,
    hpd_overlap: NDArray[np.float64],
    overlap_threshold: float = 0.2,
    model_label: str = "Model",
    color: str = "tab:blue",
) -> tuple[Figure, NDArray[np.object_]]:
    """Create model checking figure for a single model.

    Parameters
    ----------
    time_slice_ind : slice
        Time slice indices to plot.
    time : np.ndarray or pd.Index
        Time values.
    position : np.ndarray, shape (n_time,)
        Actual position values.
    spike_times : list[np.ndarray]
        List of spike time arrays.
    model : Model
        Decoder model.
    results : xr.Dataset
        Decoding results.
    hpd_overlap : np.ndarray, shape (n_time,)
        HPD overlap values.
    overlap_threshold : float, default=0.2
        Threshold for highlighting poor overlap regions.
    model_label : str, default="Model"
        Label for the model in titles.
    color : str, default="tab:blue"
        Color for highlighting regions.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object.
    axes : np.ndarray[plt.Axes]
        Array of axes objects.

    Examples
    --------
    >>> # Requires real decoder model and results
    >>> # See scripts/run_decoding.py for complete example
    """
    # Keep 16:9 ratio but set height to 4 inches
    width = 16 / 9 * 4  # width = aspect_ratio * height
    fig, axes = plt.subplots(4, 1, figsize=(width, 4), sharex=True, constrained_layout=True)

    # Posterior
    plot_posterior(
        results.predictive_posterior,
        time,
        position,
        time_slice_ind,
        ax=axes[0],
        title=f"{model_label} Decoder",
    )
    plot_overlap_regions(
        axes[0],
        time,
        time_slice_ind,
        hpd_overlap,
        threshold=overlap_threshold,
        color=color,
    )
    axes[0].set_xlabel("")

    # Raster
    sort_order = np.argsort(
        model.environments[0]
        .place_bin_centers_[model.encoding_model_[("", 0)]["place_fields"].argmax(axis=1)]
        .squeeze()
    )
    plot_raster(
        spike_times,
        slice(time[time_slice_ind.start], time[time_slice_ind.stop]),
        ax=axes[1],
        sort_order=sort_order,
    )
    plot_overlap_regions(
        axes[1],
        time,
        time_slice_ind,
        hpd_overlap,
        threshold=overlap_threshold,
        color=color,
    )
    axes[1].set_xlabel("")

    # HPD overlap trace
    plot_overlap_trace(
        time,
        time_slice_ind,
        hpd_overlap,
        labels=model_label,
        ax=axes[2],
        color=color,
    )
    axes[2].set_xlabel("")

    # State probability plot
    plot_acausal_state_prob(
        results,
        time_slice_ind,
        ax=axes[3],
        title="",
    )
    axes[3].set_title("")
    axes[3].set_xlabel("Time")

    return fig, axes


def plot_posterior_consistency_vs_covariate(
    covariate: str,
    covariate_label: str | None = None,
    hpd_overlap: NDArray[np.float64] | None = None,
    position_info: pd.DataFrame | None = None,
    gridsize: int = 100,
    alpha_scatter: float = 0.15,
    scatter_size: float = 2,
    cmap: str = "Blues",
    bins: str = "log",
    mincnt: int = 1,
    figsize: tuple[float, float] = (10, 2),
    suptitle: str | None = None,
) -> tuple[Figure, tuple[Axes, Axes]]:
    """Plot HPD overlap vs a covariate using scatter and hexbin plots.

    Parameters
    ----------
    covariate : str
        Column name in position_info to use as x-axis.
    covariate_label : str, optional
        Label for x-axis. If None, uses covariate.
    hpd_overlap : np.ndarray, shape (n_time,), optional
        HPD overlap values.
    position_info : pd.DataFrame, optional
        DataFrame containing covariate.
    gridsize : int, default=100
        Hexbin grid size.
    alpha_scatter : float, default=0.15
        Alpha for scatter plot.
    scatter_size : float, default=2
        Marker size for scatter plot.
    cmap : str, default="Blues"
        Colormap for hexbin.
    bins : str, default="log"
        Binning for hexbin.
    mincnt : int, default=1
        Minimum count for hexbin.
    figsize : tuple[float, float], default=(10, 2)
        Figure size.
    suptitle : str, optional
        Figure super-title.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object.
    axes : tuple[plt.Axes, plt.Axes]
        Tuple of (scatter_ax, hexbin_ax).

    Examples
    --------
    >>> import pandas as pd
    >>> position_info = pd.DataFrame({"speed": np.random.rand(1000)})
    >>> overlap = np.random.rand(1000)
    >>> fig, axes = plot_posterior_consistency_vs_covariate(
    ...     "speed", hpd_overlap=overlap, position_info=position_info
    ... )
    """
    if covariate_label is None:
        covariate_label = covariate
    if suptitle is None:
        suptitle = f"Posterior Consistency vs {covariate_label}"

    if position_info is None or hpd_overlap is None:
        raise ValueError("position_info and hpd_overlap must be provided")

    x = position_info[covariate]
    y = hpd_overlap

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize, constrained_layout=True)

    # Scatter plot
    ax1.scatter(x, y, s=scatter_size, alpha=alpha_scatter, color="tab:blue", edgecolor="none")
    ax1.set_xlabel(covariate_label, fontsize=11)
    ax1.set_ylabel("HPD Overlap", fontsize=11)
    ax1.set_title("Scatter", fontsize=12)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.yaxis.grid(True, linestyle=":", alpha=0.5)
    ax1.xaxis.grid(False)

    # Hexbin plot
    hb = ax2.hexbin(x, y, gridsize=gridsize, cmap=cmap, bins=bins, mincnt=mincnt, alpha=0.8)
    ax2.set_xlabel(covariate_label, fontsize=11)
    ax2.set_ylabel("HPD Overlap", fontsize=11)
    ax2.set_title("Density", fontsize=12)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.yaxis.grid(True, linestyle=":", alpha=0.5)
    ax2.xaxis.grid(False)
    cb = plt.colorbar(hb, ax=ax2, label="log10(N)")
    cb.ax.tick_params(labelsize=9)

    fig.suptitle(suptitle, fontsize=14, y=1.05)

    return fig, (ax1, ax2)


def plot_posterior_consistency_vs_array(
    x: NDArray[np.float64],
    covariate_label: str | None = None,
    hpd_overlap: NDArray[np.float64] | None = None,
    gridsize: int = 100,
    alpha_scatter: float = 0.15,
    scatter_size: float = 2,
    cmap: str = "Blues",
    bins: str = "log",
    mincnt: int = 1,
    figsize: tuple[float, float] = (10, 2),
    suptitle: str | None = None,
) -> tuple[Figure, tuple[Axes, Axes]]:
    """Plot HPD overlap vs an array covariate using scatter and hexbin plots.

    Similar to plot_posterior_consistency_vs_covariate but accepts array directly.

    Parameters
    ----------
    x : np.ndarray, shape (n_time,)
        Covariate values.
    covariate_label : str, optional
        Label for x-axis.
    hpd_overlap : np.ndarray, shape (n_time,), optional
        HPD overlap values.
    gridsize : int, default=100
        Hexbin grid size.
    alpha_scatter : float, default=0.15
        Alpha for scatter plot.
    scatter_size : float, default=2
        Marker size for scatter plot.
    cmap : str, default="Blues"
        Colormap for hexbin.
    bins : str, default="log"
        Binning for hexbin.
    mincnt : int, default=1
        Minimum count for hexbin.
    figsize : tuple[float, float], default=(10, 2)
        Figure size.
    suptitle : str, optional
        Figure super-title.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object.
    axes : tuple[plt.Axes, plt.Axes]
        Tuple of (scatter_ax, hexbin_ax).

    Examples
    --------
    >>> x = np.random.rand(1000)
    >>> overlap = np.random.rand(1000)
    >>> fig, axes = plot_posterior_consistency_vs_array(
    ...     x, covariate_label="Speed", hpd_overlap=overlap
    ... )
    """
    if hpd_overlap is None:
        raise ValueError("hpd_overlap must be provided")

    y = hpd_overlap
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize, constrained_layout=True)

    # Scatter plot
    ax1.scatter(x, y, s=scatter_size, alpha=alpha_scatter, color="tab:blue", edgecolor="none")
    ax1.set_xlabel(covariate_label, fontsize=11)
    ax1.set_ylabel("HPD Overlap", fontsize=11)
    ax1.set_title("Scatter", fontsize=12)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.yaxis.grid(True, linestyle=":", alpha=0.5)
    ax1.xaxis.grid(False)

    # Hexbin plot
    hb = ax2.hexbin(x, y, gridsize=gridsize, cmap=cmap, bins=bins, mincnt=mincnt, alpha=0.8)
    ax2.set_xlabel(covariate_label, fontsize=11)
    ax2.set_ylabel("HPD Overlap", fontsize=11)
    ax2.set_title("Density", fontsize=12)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.yaxis.grid(True, linestyle=":", alpha=0.5)
    ax2.xaxis.grid(False)
    cb = plt.colorbar(hb, ax=ax2, label="log10(N)")
    cb.ax.tick_params(labelsize=9)

    if suptitle:
        fig.suptitle(suptitle, fontsize=14, y=1.05)

    return fig, (ax1, ax2)


def plot_hpd_overlap_at_time(
    results: xr.Dataset,
    spike_counts: NDArray[np.float64],
    overlap: NDArray[np.float64],
    t: int,
    track_graph: nx.Graph,
    edge_order: list[tuple[Any, Any]],
    edge_spacing: float,
    coverage: float = 0.95,
    figsize: tuple[float, float] = (7, 5),
) -> tuple[Figure, NDArray[np.object_]]:
    """Plot posterior, likelihood, and HPD intersection at a specific time.

    Shows the mechanics of HPD overlap calculation at a single timepoint.

    Parameters
    ----------
    results : xr.Dataset
        Decoding results with predictive_posterior and log_likelihood.
    spike_counts : np.ndarray, shape (n_time, n_neurons)
        Spike count matrix.
    overlap : np.ndarray, shape (n_time,)
        HPD overlap values.
    t : int
        Time index to visualize.
    track_graph : networkx.Graph
        Track graph for visualization.
    edge_order : list[tuple]
        Edge order for linearized track.
    edge_spacing : float
        Spacing between track edges.
    coverage : float, default=0.95
        HPD coverage probability.
    figsize : tuple[float, float], default=(7, 5)
        Figure size.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object.
    axes : np.ndarray[plt.Axes]
        Array of axes objects.

    Examples
    --------
    >>> # Requires real decoder results and track graph
    >>> # See scripts/run_decoding.py for complete example
    """
    posterior_at_t = results.predictive_posterior.dropna("state_bins").to_numpy()[t]
    likelihood_at_t = np.exp(results.log_likelihood.dropna("state_bins").to_numpy())[t]
    position_bins = results.predictive_posterior.dropna("state_bins").position.to_numpy()

    posterior_threshold = get_highest_posterior_threshold(posterior_at_t[None], coverage=coverage)
    likelihood_threshold = get_highest_posterior_threshold(likelihood_at_t[None], coverage=coverage)
    isin_posterior_hpd = posterior_at_t >= posterior_threshold[:, None]
    isin_likelihood_hpd = likelihood_at_t >= likelihood_threshold[:, None]

    denom = np.min(
        np.stack((isin_posterior_hpd.sum(axis=1), isin_likelihood_hpd.sum(axis=1)), axis=1),
        axis=1,
    )
    # Avoid division by zero
    denom = np.clip(denom, 1, None)

    fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True, constrained_layout=True)

    # Posterior
    axes[0].plot(position_bins, posterior_at_t, label="Posterior", color="tab:blue")
    axes[0].axhline(
        posterior_threshold,
        color="tab:blue",
        linestyle="--",
        label=f"{int(coverage * 100)}% HPD Threshold",
    )
    axes[0].fill_between(
        position_bins,
        posterior_at_t,
        posterior_threshold,
        where=isin_posterior_hpd[0],
        color="tab:blue",
        alpha=0.3,
    )
    axes[0].set_title("Posterior")
    axes[0].legend()

    # Likelihood
    axes[1].plot(position_bins, likelihood_at_t, label="Likelihood", color="tab:orange")
    axes[1].fill_between(
        position_bins,
        likelihood_at_t,
        likelihood_threshold,
        where=isin_likelihood_hpd[0],
        color="tab:orange",
        alpha=0.3,
    )
    axes[1].axhline(
        likelihood_threshold,
        color="tab:orange",
        linestyle="--",
        label=f"{int(coverage * 100)}% HPD Threshold",
    )
    axes[1].set_title("Likelihood")
    axes[1].legend()

    # Intersection
    axes[2].plot(
        position_bins,
        isin_posterior_hpd[0] & isin_likelihood_hpd[0],
        label="HPD Intersection",
        color="tab:green",
    )
    axes[2].set_title("HPD Intersection")
    axes[2].set_xlabel("Position")
    axes[2].set_ylabel("Is Overlap")
    axes[2].legend()
    plot_graph_as_1D(track_graph, edge_order, edge_spacing, ax=axes[2])

    # Add summary info
    n_total_spikes = int(spike_counts[t].sum())
    n_hpd_likelihood = int(isin_likelihood_hpd[0].sum())
    n_hpd_posterior = int(isin_posterior_hpd[0].sum())
    denom_val = int(denom[0])
    n_overlap = int(np.sum(isin_posterior_hpd[0] & isin_likelihood_hpd[0]))

    plt.suptitle(
        f"Index {t}, overlap = {overlap[t]:.3f}, n_spikes = {n_total_spikes},\n"
        f"n_hpd_likelihood = {n_hpd_likelihood}, n_hpd_posterior = {n_hpd_posterior}, \n"
        f"n_overlap = {n_overlap}, denom = {denom_val}",
        fontsize=12,
        y=1.12,
    )

    return fig, axes


# =============================================================================
# Per-Cell Diagnostic Plotting for Model Comparison (Figure 4)
# =============================================================================


def plot_per_cell_diagnostic_scatter(
    time: NDArray[np.float64] | pd.Index,
    diagnostics: dict[str, NDArray[np.float64]],
    time_slice_ind: slice | None = None,
    threshold: float | None = None,
    ax: Axes | None = None,
    metric_name: str = "hpd_overlap",
    color: str = "steelblue",
    ylabel: str | None = None,
    show_xlabel: bool = True,
) -> Axes:
    """Plot per-cell diagnostic metric as scatter plot over time.

    Each point represents one cell at one time point. Values are scattered
    to show the distribution of diagnostics across cells.

    For spike_prob, values are transformed to -log10 scale to match Figure 3
    visualization where higher values indicate worse fit.

    Parameters
    ----------
    time : np.ndarray or pd.Index
        Time values.
    diagnostics : dict[str, np.ndarray]
        Dictionary with diagnostic arrays, each with shape (n_time, n_cells).
    time_slice_ind : slice, optional
        Time slice indices to plot. If None, plots all time points.
    threshold : float, optional
        Threshold to draw as horizontal line. For spike_prob, this should be
        the raw threshold value (e.g., 0.05) which will be transformed.
    ax : plt.Axes, optional
        Axes to plot on. If None, uses current axes.
    metric_name : str, default "hpd_overlap"
        Key in diagnostics dict to plot.
    color : str, default "steelblue"
        Color for scatter points.
    ylabel : str, optional
        Y-axis label. If None, uses metric_name.
    show_xlabel : bool, default True
        Whether to show "Time" xlabel.

    Returns
    -------
    ax : plt.Axes
        The axes object.

    Examples
    --------
    >>> import numpy as np
    >>> time = np.linspace(0, 10, 100)
    >>> diagnostics = {"hpd_overlap": np.random.rand(100, 10)}
    >>> ax = plot_per_cell_diagnostic_scatter(time, diagnostics)
    """
    if ax is None:
        ax = plt.gca()

    metric = diagnostics[metric_name].copy()

    if time_slice_ind is not None:
        time = time[time_slice_ind]
        metric = metric[time_slice_ind]

    # Transform spike_prob to -log10 scale (matching Figure 3)
    # Higher values indicate worse fit (low probability)
    if metric_name == "spike_prob":
        metric = -np.log10(np.maximum(metric, 1e-10))
        if threshold is not None:
            threshold = -np.log10(max(threshold, 1e-10))

    n_time, n_cells = metric.shape

    # Create time indices for scatter plot
    time_arr = np.asarray(time)
    time_indices = np.tile(time_arr[:, np.newaxis], (1, n_cells))

    ax.scatter(
        time_indices.ravel(),
        metric.ravel(),
        s=0.8,
        alpha=0.6,
        c=color,
        rasterized=True,
    )

    if threshold is not None:
        ax.axhline(
            threshold,
            color=COLORS["threshold"],
            linewidth=1.2,
            alpha=0.7,
            zorder=10,
        )
        # Add threshold annotation on right side
        ax.text(
            1.01,
            threshold,
            "Threshold",
            transform=ax.get_yaxis_transform(),
            fontsize=6,
            va="center",
            ha="left",
            color=COLORS["threshold"],
        )

    ax.set_xlim(time_arr.min(), time_arr.max())
    ax.set_ylabel(metric_name if ylabel is None else ylabel, fontsize=9, labelpad=7)

    if show_xlabel:
        ax.set_xlabel("Time (s)", fontsize=9, labelpad=7)
        ax.tick_params(labelsize=7)
    else:
        ax.tick_params(labelsize=7, labelbottom=False)

    return ax


def plot_model_comparison_diagnostics(
    time: NDArray[np.float64] | pd.Index,
    diagnostics_a: dict[str, NDArray[np.float64]],
    diagnostics_b: dict[str, NDArray[np.float64]],
    time_slice_ind: slice | None = None,
    model_a_name: str = "Continuous",
    model_b_name: str = "ContFrag",
    thresholds: dict[str, float] | None = None,
    figsize: tuple[float, float] = (12, 8),
) -> tuple[Figure, NDArray[np.object_]]:
    """Create side-by-side comparison of per-cell diagnostics for two models.

    Creates a 3x2 grid of scatter plots showing HPD overlap, KL divergence,
    and spike probability for each model.

    Parameters
    ----------
    time : np.ndarray or pd.Index
        Time values.
    diagnostics_a : dict[str, np.ndarray]
        Diagnostics for model A with keys 'hpd_overlap', 'kl_divergence', 'spike_prob'.
    diagnostics_b : dict[str, np.ndarray]
        Diagnostics for model B with same keys.
    time_slice_ind : slice, optional
        Time slice indices to plot. If None, plots all time points.
    model_a_name : str, default "Continuous"
        Name for model A (column title).
    model_b_name : str, default "ContFrag"
        Name for model B (column title).
    thresholds : dict[str, float], optional
        Thresholds for each metric to draw as horizontal lines.
    figsize : tuple[float, float], default (12, 8)
        Figure size in inches.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object.
    axes : np.ndarray[plt.Axes]
        Array of axes objects with shape (3, 2).

    Examples
    --------
    >>> import numpy as np
    >>> time = np.linspace(0, 10, 100)
    >>> diag_a = {
    ...     "hpd_overlap": np.random.rand(100, 10),
    ...     "kl_divergence": np.random.rand(100, 10),
    ...     "spike_prob": np.random.rand(100, 10),
    ... }
    >>> diag_b = {k: np.random.rand(100, 10) for k in diag_a}
    >>> fig, axes = plot_model_comparison_diagnostics(time, diag_a, diag_b)
    >>> plt.close(fig)
    """
    metrics = ["hpd_overlap", "kl_divergence", "spike_prob"]
    # Match Figure 3 styling: labels and colors from COLORS dict
    ylabels = ["HPD Overlap", "KL Divergence", r"$-\log_{10}(p)$"]
    colors = [COLORS["hpd_overlap"], COLORS["kl_divergence"], COLORS["metric_combined"]]
    # Direction indicators: which direction indicates worse fit
    worse_fit_directions = ["↓ Worse fit", "↑ Worse fit", "↑ Worse fit"]

    fig, axes = plt.subplots(3, 2, figsize=figsize, sharex=True, constrained_layout=True)

    for i, (metric, ylabel, color, worse_dir) in enumerate(
        zip(metrics, ylabels, colors, worse_fit_directions, strict=True)
    ):
        threshold = thresholds.get(metric) if thresholds else None

        # Model A (left column)
        plot_per_cell_diagnostic_scatter(
            time,
            diagnostics_a,
            time_slice_ind=time_slice_ind,
            threshold=threshold,
            ax=axes[i, 0],
            metric_name=metric,
            color=color,
            ylabel=ylabel,
            show_xlabel=(i == 2),
        )

        # Model B (right column) - no ylabel, left column has it
        plot_per_cell_diagnostic_scatter(
            time,
            diagnostics_b,
            time_slice_ind=time_slice_ind,
            threshold=threshold,
            ax=axes[i, 1],
            metric_name=metric,
            color=color,
            ylabel="",  # Empty string to suppress ylabel (left column has it)
            show_xlabel=(i == 2),
        )

        # Add direction indicator on right side of right column (matching Figure 3)
        axes[i, 1].text(
            1.01,
            0.5,
            worse_dir,
            transform=axes[i, 1].transAxes,
            fontsize=6,
            va="center",
            ha="left",
        )

        # Add column titles on first row
        if i == 0:
            axes[i, 0].set_title(model_a_name, fontsize=11)
            axes[i, 1].set_title(model_b_name, fontsize=11)

    return fig, axes


def plot_diagnostic_summary_comparison(
    diagnostics_a: dict[str, NDArray[np.float64]],
    diagnostics_b: dict[str, NDArray[np.float64]],
    model_a_name: str = "Continuous",
    model_b_name: str = "ContFrag",
    figsize: tuple[float, float] = (10, 4),
) -> tuple[Figure, NDArray[np.object_]]:
    """Create bar chart comparing mean diagnostics between models.

    Parameters
    ----------
    diagnostics_a : dict[str, np.ndarray]
        Diagnostics for model A.
    diagnostics_b : dict[str, np.ndarray]
        Diagnostics for model B.
    model_a_name : str, default "Continuous"
        Name for model A.
    model_b_name : str, default "ContFrag"
        Name for model B.
    figsize : tuple[float, float], default (10, 4)
        Figure size in inches.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object.
    axes : np.ndarray[plt.Axes]
        Array of axes objects.

    Examples
    --------
    >>> import numpy as np
    >>> diag_a = {"hpd_overlap": np.random.rand(100, 10)}
    >>> diag_b = {"hpd_overlap": np.random.rand(100, 10)}
    >>> fig, axes = plot_diagnostic_summary_comparison(diag_a, diag_b)
    >>> plt.close(fig)
    """
    metrics = ["hpd_overlap", "kl_divergence", "spike_prob"]
    # Match Figure 3 styling for labels
    xlabels = ["HPD Overlap", "KL Divergence", r"$-\log_{10}(p)$"]

    fig, axes = plt.subplots(1, 3, figsize=figsize, constrained_layout=True)

    for i, (metric, xlabel) in enumerate(zip(metrics, xlabels, strict=True)):
        if metric not in diagnostics_a or metric not in diagnostics_b:
            continue

        # Get data and transform spike_prob to -log10 scale
        data_a = diagnostics_a[metric]
        data_b = diagnostics_b[metric]
        if metric == "spike_prob":
            data_a = -np.log10(np.maximum(data_a, 1e-10))
            data_b = -np.log10(np.maximum(data_b, 1e-10))

        mean_a = np.nanmean(data_a)
        mean_b = np.nanmean(data_b)
        sem_a = np.nanstd(data_a) / np.sqrt(np.sum(~np.isnan(data_a)))
        sem_b = np.nanstd(data_b) / np.sqrt(np.sum(~np.isnan(data_b)))

        x = [0, 1]
        heights = [mean_a, mean_b]
        errors = [sem_a, sem_b]
        colors = ["tab:blue", "tab:orange"]

        bars = axes[i].bar(x, heights, yerr=errors, color=colors, capsize=5, alpha=0.8)
        axes[i].set_xticks(x)
        axes[i].set_xticklabels([model_a_name, model_b_name])
        axes[i].set_ylabel(f"Mean {xlabel}")
        axes[i].set_title(xlabel)

        # Add value annotations
        for bar, h, err in zip(bars, heights, errors, strict=True):
            axes[i].annotate(
                f"{h:.3f}",
                xy=(bar.get_x() + bar.get_width() / 2, h + err + 0.01),
                ha="center",
                va="bottom",
                fontsize=9,
            )

    return fig, axes


def plot_model_comparison_with_posterior(
    time: NDArray[np.float64] | pd.Index,
    position: NDArray[np.float64],
    results_a: xr.Dataset,
    results_b: xr.Dataset,
    diagnostics_a: dict[str, NDArray[np.float64]],
    diagnostics_b: dict[str, NDArray[np.float64]],
    spike_times: list[NDArray[np.float64]] | None = None,
    place_field_peaks: NDArray[np.float64] | None = None,
    time_slice_ind: slice | None = None,
    model_a_name: str = "Continuous",
    model_b_name: str = "Continuous-Fragmented",
    thresholds: dict[str, float] | None = None,
    figsize: tuple[float, float] = (7.0, 8.5),
) -> tuple[Figure, NDArray[np.object_]]:
    """Create model comparison with posterior heatmaps, spike raster, and diagnostics.

    Creates a 5x2 grid with:
    - Row 0: Decoded posterior with animal position overlay
    - Row 1: Spike raster (cells sorted by place field peak)
    - Row 2: HPD overlap scatter
    - Row 3: KL divergence scatter
    - Row 4: Spike probability scatter

    Parameters
    ----------
    time : np.ndarray or pd.Index
        Time values.
    position : np.ndarray, shape (n_time,)
        Animal position values.
    results_a : xr.Dataset
        Decoding results for model A with predictive_posterior.
    results_b : xr.Dataset
        Decoding results for model B with predictive_posterior.
    diagnostics_a : dict[str, np.ndarray]
        Diagnostics for model A with keys 'hpd_overlap', 'kl_divergence', 'spike_prob'.
    diagnostics_b : dict[str, np.ndarray]
        Diagnostics for model B with same keys.
    spike_times : list[np.ndarray], optional
        List of spike time arrays, one per neuron. Required for raster plot.
    place_field_peaks : np.ndarray, shape (n_cells,), optional
        Position of place field peak for each cell, used for sorting raster.
        If None, cells are plotted in original order.
    time_slice_ind : slice, optional
        Time slice indices to plot. If None, plots all time points.
    model_a_name : str, default "Continuous"
        Name for model A (column title).
    model_b_name : str, default "Continuous-Fragmented"
        Name for model B (column title).
    thresholds : dict[str, float], optional
        Thresholds for each metric to draw as horizontal lines.
    figsize : tuple[float, float], default (7.0, 8.5)
        Figure size in inches.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object.
    axes : np.ndarray[plt.Axes]
        Array of axes objects with shape (5, 2).

    Examples
    --------
    >>> # Requires xr.Dataset from non_local_detector
    >>> # fig, axes = plot_model_comparison_with_posterior(
    >>> #     time, position, results_a, results_b, diagnostics_a, diagnostics_b,
    >>> #     spike_times=spike_times, place_field_peaks=pf_peaks
    >>> # )
    """
    metrics = ["hpd_overlap", "kl_divergence", "spike_prob"]
    # Match Figure 3 styling: labels and colors from COLORS dict
    ylabels = ["HPD Overlap", "KL Divergence", r"$-\log_{10}(p)$"]
    colors = [COLORS["hpd_overlap"], COLORS["kl_divergence"], COLORS["metric_combined"]]
    # Direction indicators: which direction indicates worse fit
    worse_fit_directions = ["↓ Worse fit", "↑ Worse fit", "↑ Worse fit"]

    # Create 5x2 grid: posterior + raster + 3 diagnostics
    # Use gridspec to manually share y-axes within each row
    fig = plt.figure(figsize=figsize, constrained_layout=True)
    gs = fig.add_gridspec(5, 2, height_ratios=[3, 1.5, 1, 1, 1])

    # Create axes with shared x and shared y within each row
    axes = np.empty((5, 2), dtype=object)

    # Row 0: Posterior heatmaps (share y within row)
    axes[0, 0] = fig.add_subplot(gs[0, 0])
    axes[0, 1] = fig.add_subplot(gs[0, 1], sharex=axes[0, 0], sharey=axes[0, 0])

    # Row 1: Spike raster (share y within row, share x with row 0)
    axes[1, 0] = fig.add_subplot(gs[1, 0], sharex=axes[0, 0])
    axes[1, 1] = fig.add_subplot(gs[1, 1], sharex=axes[0, 0], sharey=axes[1, 0])

    # Row 2: HPD overlap (share y within row, share x with row 0)
    axes[2, 0] = fig.add_subplot(gs[2, 0], sharex=axes[0, 0])
    axes[2, 1] = fig.add_subplot(gs[2, 1], sharex=axes[0, 0], sharey=axes[2, 0])

    # Row 3: KL divergence (share y within row, share x with row 0)
    axes[3, 0] = fig.add_subplot(gs[3, 0], sharex=axes[0, 0])
    axes[3, 1] = fig.add_subplot(gs[3, 1], sharex=axes[0, 0], sharey=axes[3, 0])

    # Row 4: Spike probability (share y within row, share x with row 0)
    axes[4, 0] = fig.add_subplot(gs[4, 0], sharex=axes[0, 0])
    axes[4, 1] = fig.add_subplot(gs[4, 1], sharex=axes[0, 0], sharey=axes[4, 0])

    if time_slice_ind is None:
        time_slice_ind = slice(None)

    # Row 0: Posterior heatmaps with position overlay
    for col, (results, model_name) in enumerate(
        [(results_a, model_a_name), (results_b, model_b_name)]
    ):
        ax = axes[0, col]

        # Get posterior and marginalize over states
        if "predictive_posterior" in results:
            posterior_da = results.predictive_posterior
        else:
            posterior_da = results.acausal_posterior

        # Drop NaN bins and marginalize
        posterior_da = posterior_da.dropna("state_bins")

        # Plot posterior heatmap
        try:
            unstacked = posterior_da.unstack("state_bins")
            if "state" in unstacked.dims:
                # Multi-state model: sum over states
                marginalized = unstacked.sum("state")
            else:
                marginalized = unstacked

            # Plot using xarray
            marginalized.isel(time=time_slice_ind).plot(
                x="time",
                y="position",
                ax=ax,
                add_colorbar=False,
                robust=True,
                cmap="bone_r",
                rasterized=True,
            )
        except (ValueError, KeyError):
            # Fallback: plot raw posterior
            posterior_da.isel(time=time_slice_ind).plot(
                x="time",
                ax=ax,
                add_colorbar=False,
                robust=True,
                cmap="bone_r",
                rasterized=True,
            )

        # Overlay animal position
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

        ax.set_title(model_name, fontsize=11)
        ax.set_ylabel("Position (cm)" if col == 0 else "", fontsize=9, labelpad=7)
        ax.set_xlabel("")
        ax.tick_params(labelsize=7, labelbottom=False)

        # Add legend for true position line (only on first column)
        if col == 0:
            ax.legend(loc="upper left", fontsize=6, frameon=False)

    # Row 1: Spike raster (both columns show same raster, sorted by place field peak)
    if spike_times is not None:
        # Compute sort order by place field peak position
        if place_field_peaks is not None:
            sort_order = np.argsort(place_field_peaks)
        else:
            sort_order = None

        # Get time slice for raster (convert index slice to time values)
        time_arr = np.asarray(time)
        sliced_time = time_arr[time_slice_ind]
        time_slice = slice(float(sliced_time[0]), float(sliced_time[-1]))

        for col in range(2):
            ax = axes[1, col]
            plot_raster(
                spike_times,
                time_slice,
                ax=ax,
                sort_order=sort_order,
                linewidths=0.5,
            )
            ax.set_ylabel("Neuron" if col == 0 else "", fontsize=9, labelpad=7)
            ax.set_xlabel("")
            ax.tick_params(labelsize=7, labelbottom=False)

    # Rows 2-4: Diagnostic scatter plots
    for i, (metric, ylabel, color, worse_dir) in enumerate(
        zip(metrics, ylabels, colors, worse_fit_directions, strict=True)
    ):
        row = i + 2  # Offset by 2 for posterior and raster rows
        threshold = thresholds.get(metric) if thresholds else None

        # Model A (left column)
        plot_per_cell_diagnostic_scatter(
            time,
            diagnostics_a,
            time_slice_ind=time_slice_ind,
            threshold=threshold,
            ax=axes[row, 0],
            metric_name=metric,
            color=color,
            ylabel=ylabel,
            show_xlabel=(i == 2),
        )

        # Model B (right column)
        plot_per_cell_diagnostic_scatter(
            time,
            diagnostics_b,
            time_slice_ind=time_slice_ind,
            threshold=threshold,
            ax=axes[row, 1],
            metric_name=metric,
            color=color,
            ylabel="",  # Left column has ylabel
            show_xlabel=(i == 2),
        )

        # Add direction indicator on right side of right column (matching Figure 3)
        axes[row, 1].text(
            1.01,
            0.5,
            worse_dir,
            transform=axes[row, 1].transAxes,
            fontsize=6,
            va="center",
            ha="left",
        )

    # Hide y-tick labels on right column (since y-axes are shared within rows)
    for row in range(5):
        axes[row, 1].tick_params(labelleft=False)

    return fig, axes
