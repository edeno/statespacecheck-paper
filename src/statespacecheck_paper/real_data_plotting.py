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

from typing import Any, cast

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.image import AxesImage
from non_local_detector.model_checking.highest_posterior_density import (
    get_highest_posterior_threshold,
)
from numpy.typing import NDArray
from scipy.ndimage import label
from track_linearization import plot_graph_as_1D


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
) -> AxesImage:
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
    im : matplotlib.image.AxesImage
        The image object from the plot.

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

    return cast(AxesImage, im)


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
) -> tuple[Figure, np.ndarray[Any, Any]]:
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
) -> tuple[Figure, NDArray[Axes]]:
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
) -> tuple[Figure, NDArray[Axes]]:
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
