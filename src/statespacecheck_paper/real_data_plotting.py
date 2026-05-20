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

import matplotlib
import matplotlib.colors as mcolors
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

from statespacecheck_paper.plotting import plot_likelihood_columns
from statespacecheck_paper.style import CMAP_LIKELIHOOD, CMAP_POSTERIOR, COLORS


def add_scalebar(
    ax: Axes,
    length: float,
    label: str,
    loc: str = "lower right",
    pad: float = 0.1,
    fontsize: int = 7,
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


def plot_track_graph_2d(
    track_graph: nx.Graph,
    position_info: pd.DataFrame,
    ax: Axes | None = None,
    edge_order: list[tuple[int, int]] | None = None,
    reward_well_nodes: list[int] | None = None,
    edge_colors: NDArray[np.float64] | None = None,
    position_names: tuple[str, str] = ("head_position_x", "head_position_y"),
    scalebar_length: float = 20,
    scalebar_label: str = "20 cm",
    show_trajectory: bool = True,
) -> Axes:
    """Plot 2D track graph with optional position trajectory overlay.

    Parameters
    ----------
    track_graph : networkx.Graph
        Track graph with nodes containing 'pos' attributes.
    position_info : pandas.DataFrame
        DataFrame containing position columns for trajectory overlay.
    ax : Axes, optional
        Axes to plot on. If None, uses current axes.
    edge_order : list of tuple of int, optional
        Order of edges. If None, uses graph's natural edge order.
    reward_well_nodes : list of int, optional
        Node indices that are reward wells (marked with scatter points).
    edge_colors : ndarray, optional
        Array of colors for each edge. If None, uses tab10 colormap.
    position_names : tuple of str, optional
        Column names for (x, y) position in position_info.
    scalebar_length : float, optional
        Length of scale bar in data units, by default 20.
    scalebar_label : str, optional
        Label for scale bar, by default "20 cm".
    show_trajectory : bool, default True
        Whether to show the position trajectory.

    Returns
    -------
    ax : Axes
        The axes object.
    """
    if ax is None:
        ax = plt.gca()
    if reward_well_nodes is None:
        reward_well_nodes = []
    if edge_colors is None:
        cmap = matplotlib.colormaps.get_cmap("tab10")
        edge_colors = np.array([cmap(i) for i in range(10)])
    if edge_order is None:
        edge_order = list(track_graph.edges)

    # Plot trajectory
    if show_trajectory:
        ax.plot(
            position_info[position_names[0]],
            position_info[position_names[1]],
            color="lightgrey",
            alpha=0.7,
            linewidth=0.5,
            rasterized=True,
        )

    # Plot track graph edges
    for edge_ind, (node1, node2) in enumerate(edge_order):
        edge_color = edge_colors[edge_ind % len(edge_colors)]
        node1_pos = track_graph.nodes[node1]["pos"]
        node2_pos = track_graph.nodes[node2]["pos"]
        ax.plot(
            [node1_pos[0], node2_pos[0]],
            [node1_pos[1], node2_pos[1]],
            linewidth=2,
            color=edge_color,
        )
        if node1 in reward_well_nodes:
            ax.scatter(
                node1_pos[0],
                node1_pos[1],
                color=edge_color,
                s=30,
                zorder=10,
            )
        if node2 in reward_well_nodes:
            ax.scatter(
                node2_pos[0],
                node2_pos[1],
                color=edge_color,
                s=30,
                zorder=10,
            )

    add_scalebar(ax, scalebar_length, scalebar_label)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")

    return ax


def plot_track_graph_1d(
    track_graph: nx.Graph,
    ax: Axes,
    edge_order: list[tuple[int, int]] | None = None,
    edge_spacing: float | list[float] = 0.0,
    reward_well_nodes: list[int] | None = None,
    other_axis_start: float = 0,
    edge_colors: NDArray[np.float64] | None = None,
    reward_well_size: int = 10,
    edge_linewidth: int = 2,
    orientation: str = "vertical",
) -> None:
    """Plot track graph as 1D linearized representation.

    Draws the track graph edges as line segments positioned sequentially
    to show the linearized track structure. Default is vertical orientation
    (position on y-axis). Use orientation="horizontal" for position on x-axis.

    Parameters
    ----------
    track_graph : networkx.Graph
        Track graph with edges containing 'distance' attributes (in cm).
    ax : Axes
        Axes to plot on.
    edge_order : list of tuple of int, optional
        Order of edges for linearization. If None, uses graph's natural edge order.
    edge_spacing : float or list of float, optional
        Spacing between edges in cm. By default 0.0.
    reward_well_nodes : list of int, optional
        Node indices that are reward wells (marked with scatter points).
    other_axis_start : float, optional
        Position on the non-position axis (x for vertical, y for horizontal).
    edge_colors : ndarray, optional
        Array of RGB colors for each edge. If None, uses tab10 colormap.
    reward_well_size : int, optional
        Marker size for reward well points, by default 10.
    edge_linewidth : int, optional
        Line width for edge segments, by default 2.
    orientation : str, default "vertical"
        Orientation of the track. "vertical" places position on y-axis,
        "horizontal" places position on x-axis.
    """
    if edge_order is None:
        edge_order = list(track_graph.edges)
    if reward_well_nodes is None:
        reward_well_nodes = []
    if edge_colors is None:
        cmap = matplotlib.colormaps.get_cmap("tab10")
        edge_colors = np.array([cmap(i) for i in range(10)])

    n_edges = len(edge_order)
    if isinstance(edge_spacing, int | float):
        edge_spacing_list = [float(edge_spacing)] * (n_edges - 1)
    else:
        edge_spacing_list = list(edge_spacing)

    start_node_linear_position = 0.0

    for edge_ind, edge in enumerate(edge_order):
        edge_color = edge_colors[edge_ind % len(edge_colors)]
        end_node_linear_position = start_node_linear_position + track_graph.edges[edge]["distance"]

        if orientation == "vertical":
            # Position on y-axis, other_axis_start is x-position
            ax.plot(
                (other_axis_start, other_axis_start),
                (start_node_linear_position, end_node_linear_position),
                color=edge_color,
                clip_on=False,
                zorder=7,
                linewidth=edge_linewidth,
            )
            scatter_x, scatter_y_start, scatter_y_end = (
                other_axis_start,
                start_node_linear_position,
                end_node_linear_position,
            )
        else:
            # Position on x-axis, other_axis_start is y-position
            ax.plot(
                (start_node_linear_position, end_node_linear_position),
                (other_axis_start, other_axis_start),
                color=edge_color,
                clip_on=False,
                zorder=7,
                linewidth=edge_linewidth,
                solid_capstyle="butt",
            )
            scatter_x_start, scatter_x_end, scatter_y = (
                start_node_linear_position,
                end_node_linear_position,
                other_axis_start,
            )

        if edge[0] in reward_well_nodes:
            if orientation == "vertical":
                ax.scatter(
                    scatter_x,
                    scatter_y_start,
                    color=edge_color,
                    s=reward_well_size,
                    zorder=10,
                    clip_on=False,
                )
            else:
                ax.scatter(
                    scatter_x_start,
                    scatter_y,
                    color=edge_color,
                    s=reward_well_size,
                    zorder=10,
                    clip_on=False,
                )
        if edge[1] in reward_well_nodes:
            if orientation == "vertical":
                ax.scatter(
                    scatter_x,
                    scatter_y_end,
                    color=edge_color,
                    s=reward_well_size,
                    zorder=10,
                    clip_on=False,
                )
            else:
                ax.scatter(
                    scatter_x_end,
                    scatter_y,
                    color=edge_color,
                    s=reward_well_size,
                    zorder=10,
                    clip_on=False,
                )

        # Update position for next edge (skip spacing on last edge)
        if edge_ind < len(edge_spacing_list):
            start_node_linear_position += (
                track_graph.edges[edge]["distance"] + edge_spacing_list[edge_ind]
            )
        else:
            start_node_linear_position += track_graph.edges[edge]["distance"]


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

    # Set defaults that can be overridden by eventplot_kwargs
    linelengths = eventplot_kwargs.pop("linelengths", 1.0)
    linewidths = eventplot_kwargs.pop("linewidths", 1.5)
    colors = eventplot_kwargs.pop("colors", "black")
    rasterized = eventplot_kwargs.pop("rasterized", True)

    ax.eventplot(
        time_slice_spike_times,
        linelengths=linelengths,
        linewidths=linewidths,
        colors=colors,
        rasterized=rasterized,
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
            cmap=CMAP_POSTERIOR,
            rasterized=True,
            **plot_kwargs,
        )
    )

    if scatter_kwargs is None:
        scatter_kwargs = {
            "color": COLORS["ground_truth"],
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
    ax1.set_xlabel(covariate_label, fontsize=7)
    ax1.set_ylabel("HPD Overlap", fontsize=7)
    ax1.set_title("Scatter", fontsize=7)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.yaxis.grid(True, linestyle=":", alpha=0.5)
    ax1.xaxis.grid(False)

    # Hexbin plot
    hb = ax2.hexbin(x, y, gridsize=gridsize, cmap=cmap, bins=bins, mincnt=mincnt, alpha=0.8)
    ax2.set_xlabel(covariate_label, fontsize=7)
    ax2.set_ylabel("HPD Overlap", fontsize=7)
    ax2.set_title("Density", fontsize=7)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.yaxis.grid(True, linestyle=":", alpha=0.5)
    ax2.xaxis.grid(False)
    cb = plt.colorbar(hb, ax=ax2, label="log10(N)")
    cb.ax.tick_params(labelsize=6)

    fig.suptitle(suptitle, fontsize=8, y=1.05)

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
    ax1.set_xlabel(covariate_label, fontsize=7)
    ax1.set_ylabel("HPD Overlap", fontsize=7)
    ax1.set_title("Scatter", fontsize=7)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.yaxis.grid(True, linestyle=":", alpha=0.5)
    ax1.xaxis.grid(False)

    # Hexbin plot
    hb = ax2.hexbin(x, y, gridsize=gridsize, cmap=cmap, bins=bins, mincnt=mincnt, alpha=0.8)
    ax2.set_xlabel(covariate_label, fontsize=7)
    ax2.set_ylabel("HPD Overlap", fontsize=7)
    ax2.set_title("Density", fontsize=7)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.yaxis.grid(True, linestyle=":", alpha=0.5)
    ax2.xaxis.grid(False)
    cb = plt.colorbar(hb, ax=ax2, label="log10(N)")
    cb.ax.tick_params(labelsize=6)

    if suptitle:
        fig.suptitle(suptitle, fontsize=8, y=1.05)

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
        fontsize=8,
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
    spike_times: list[NDArray[np.float64]] | None = None,
    show_running_average: bool = False,
    running_average_window: float = 0.050,
    running_average_color: str | None = None,
) -> Axes:
    """Plot per-cell diagnostic metric as scatter plot over time.

    Each point represents one cell at one time point. Values are scattered
    to show the distribution of diagnostics across cells.

    For spike_prob, values are transformed to -log10 scale to match Figure 3
    visualization where higher values indicate worse fit.

    Parameters
    ----------
    time : np.ndarray or pd.Index
        Time values (bin centers/starts).
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
    spike_times : list[np.ndarray], optional
        List of spike time arrays, one per cell. If provided, diagnostic
        points are plotted at actual spike times instead of bin values,
        aligning them with raster plots.
    show_running_average : bool, default False
        If True, overlay a running average line on top of the scatter plot.
        The running average is computed as the weighted mean over a sliding
        window, as described in the manuscript.
    running_average_window : float, default 0.050
        Size of the sliding window in seconds for the running average.
    running_average_color : str, optional
        Color for the running average line. If None, uses a darker version
        of the scatter color.

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
    time_arr = np.asarray(time)

    if time_slice_ind is not None:
        time_arr = time_arr[time_slice_ind]
        metric = metric[time_slice_ind]

    event_times = diagnostics.get("event_time")
    event_metric_values = diagnostics.get(f"event_{metric_name}")

    # Store raw metric for running average computation (before transformation)
    # The running average should be computed on raw values per manuscript formula:
    # D = sum(metric_k * I(t_k in window)) / sum(I(t_k in window))
    raw_metric = metric.copy()
    raw_event_metric_values = (
        None if event_metric_values is None else np.asarray(event_metric_values).copy()
    )

    # Transform spike_prob to -log10 scale (matching Figure 3)
    # Higher values indicate worse fit (low probability)
    if metric_name == "spike_prob":
        metric = -np.log10(np.maximum(metric, 1e-10))
        if threshold is not None:
            threshold = -np.log10(max(threshold, 1e-10))

    n_time, n_cells = metric.shape

    if event_times is not None and raw_event_metric_values is not None:
        event_times_arr = np.asarray(event_times)
        time_min, time_max = time_arr.min(), time_arr.max()
        event_mask = (event_times_arr >= time_min) & (event_times_arr < time_max)

        x_positions_arr = event_times_arr[event_mask]
        y_values_arr = raw_event_metric_values[event_mask]
        if metric_name == "spike_prob":
            y_values_arr = -np.log10(np.maximum(y_values_arr, 1e-10))
        valid = ~np.isnan(y_values_arr)
        x_positions_arr = x_positions_arr[valid]
        y_values_arr = y_values_arr[valid]
    elif spike_times is not None:
        # Use actual spike times for x-positions to align with raster
        # Find the time range for filtering spikes
        time_min, time_max = time_arr.min(), time_arr.max()

        # Collect (spike_time, diagnostic_value) pairs for all non-NaN diagnostics
        x_positions = []
        y_values = []

        for cell_idx in range(n_cells):
            cell_spike_times = spike_times[cell_idx]
            # Filter to spikes within the time window
            mask = (cell_spike_times >= time_min) & (cell_spike_times < time_max)
            cell_spikes_in_window = cell_spike_times[mask]

            # For each spike, find which time bin it falls into
            # Use searchsorted to find bin indices
            # Spikes are binned into time[i] if time[i] <= spike < time[i+1]
            bin_indices = np.searchsorted(time_arr, cell_spikes_in_window, side="right") - 1
            # Clamp to valid range
            bin_indices = np.clip(bin_indices, 0, n_time - 1)

            # Get diagnostic values at those bins for this cell
            for spike_t, bin_idx in zip(cell_spikes_in_window, bin_indices, strict=True):
                diag_val = metric[bin_idx, cell_idx]
                if not np.isnan(diag_val):
                    x_positions.append(spike_t)
                    y_values.append(diag_val)

        x_positions_arr = np.array(x_positions)
        y_values_arr = np.array(y_values)
    else:
        # Original behavior: use time bin values for x-positions
        time_indices = np.tile(time_arr[:, np.newaxis], (1, n_cells))
        x_positions_arr = time_indices.ravel()
        y_values_arr = metric.ravel()

    ax.scatter(
        x_positions_arr,
        y_values_arr,
        s=0.8,
        alpha=0.6,
        c=color,
        rasterized=True,
    )

    # Add running average line if requested
    if show_running_average:
        from statespacecheck_paper.real_data_analysis import compute_running_average

        # Compute running average on RAW values (before transformation)
        # per manuscript formula, then transform for display
        if event_times is not None and raw_event_metric_values is not None:
            event_times_arr = np.asarray(event_times)
            time_min, time_max = time_arr.min(), time_arr.max()
            event_mask = (event_times_arr >= time_min) & (event_times_arr < time_max)
            running_avg, _ = compute_running_average(
                raw_metric,
                time_arr,
                window_size=running_average_window,
                event_times=event_times_arr[event_mask],
                event_values=raw_event_metric_values[event_mask],
            )
        else:
            running_avg, _ = compute_running_average(
                raw_metric, time_arr, window_size=running_average_window
            )

        # Transform running average if needed (same as scatter points)
        if metric_name == "spike_prob":
            running_avg = -np.log10(np.maximum(running_avg, 1e-10))

        # Determine line color (darker version of scatter color if not specified)
        line_color: str | tuple[float, ...]
        if running_average_color is None:
            # Convert to RGB, darken by 30%, convert back
            try:
                rgb = mcolors.to_rgb(color)
                line_color = tuple(c * 0.7 for c in rgb)
            except ValueError:
                line_color = "black"
        else:
            line_color = running_average_color

        ax.plot(
            time_arr,
            running_avg,
            color=line_color,
            linewidth=2,
            alpha=0.9,
            zorder=5,
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
    ax.set_ylabel(metric_name if ylabel is None else ylabel, fontsize=7, labelpad=7)

    if show_xlabel:
        ax.set_xlabel("Time (s)", fontsize=7, labelpad=7)
        ax.tick_params(labelsize=6)
    else:
        ax.tick_params(labelsize=6, labelbottom=False)

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
            axes[i, 0].set_title(model_a_name, fontsize=7)
            axes[i, 1].set_title(model_b_name, fontsize=7)

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
                fontsize=7,
            )

    return fig, axes


def _plot_distribution_heatmap(
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

    # Plot distribution heatmap
    try:
        unstacked = distribution_da.unstack("state_bins")
        if "state" in unstacked.dims:
            # Multi-state model: sum over states
            marginalized = unstacked.sum("state")
        else:
            marginalized = unstacked

        # Get the sliced data
        sliced_data = marginalized.isel(time=time_slice_ind)

        # Check if there's any non-NaN data to plot
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
            # No data to plot - just set up the axes with proper limits
            time_arr = np.asarray(time)
            ax.set_xlim(time_arr[time_slice_ind].min(), time_arr[time_slice_ind].max())
    except (ValueError, KeyError, TypeError):
        # Fallback: plot raw distribution
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


def plot_model_comparison_with_posterior(
    time: NDArray[np.float64] | pd.Index,
    position: NDArray[np.float64],
    results_a: xr.Dataset,
    results_b: xr.Dataset,
    diagnostics_a: dict[str, NDArray[np.float64]],
    diagnostics_b: dict[str, NDArray[np.float64]],
    spike_times: list[NDArray[np.float64]] | None = None,
    spike_counts: NDArray[np.int64] | None = None,
    place_field_peaks: NDArray[np.float64] | None = None,
    time_slice_ind: slice | None = None,
    model_a_name: str = "Continuous",
    model_b_name: str = "Continuous-Fragmented",
    thresholds: dict[str, float] | None = None,
    figsize: tuple[float, float] = (7.0, 11.0),
    track_graph: nx.Graph | None = None,
    edge_order: list[tuple[int, int]] | None = None,
    edge_spacing: float | list[float] = 0.0,
    show_running_average: bool = False,
    running_average_window: float = 0.050,
    fig: Figure | None = None,
) -> tuple[Figure, NDArray[np.object_]]:
    """Create model comparison with predictive, likelihood, raster, and diagnostics.

    Creates a 6x2 grid with:
    - Row 0: Predictive posterior p(x_t | y_{1:t-1}) with animal position overlay
    - Row 1: Likelihood p(y_t | x_t) with animal position overlay (only at spike times)
    - Row 2: Spike raster (cells sorted by place field peak)
    - Row 3: HPD overlap scatter
    - Row 4: KL divergence scatter
    - Row 5: Spike probability scatter

    Parameters
    ----------
    time : np.ndarray or pd.Index
        Time values.
    position : np.ndarray, shape (n_time,)
        Animal position values.
    results_a : xr.Dataset
        Decoding results for model A with causal_posterior, predictive_posterior,
        and log_likelihood.
    results_b : xr.Dataset
        Decoding results for model B with same outputs.
    diagnostics_a : dict[str, np.ndarray]
        Diagnostics for model A with keys 'hpd_overlap', 'kl_divergence', 'spike_prob'.
    diagnostics_b : dict[str, np.ndarray]
        Diagnostics for model B with same keys.
    spike_times : list[np.ndarray], optional
        List of spike time arrays, one per neuron. Required for raster plot.
    spike_counts : np.ndarray, shape (n_time, n_cells), optional
        Spike count matrix. If provided, likelihood is only shown at times with spikes.
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
    figsize : tuple[float, float], default (7.0, 11.0)
        Figure size in inches.
    track_graph : nx.Graph, optional
        Track graph for 1D linearized track visualization.
    edge_order : list[tuple[int, int]], optional
        Order of edges for linearization.
    edge_spacing : float or list[float], default 0.0
        Spacing between edges.
    show_running_average : bool, default False
        If True, overlay a running average line on diagnostic scatter plots.
    running_average_window : float, default 0.050
        Size of the sliding window in seconds for the running average.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object.
    axes : np.ndarray[plt.Axes]
        Array of axes objects with shape (6, 2).

    Examples
    --------
    >>> # Requires xr.Dataset from non_local_detector
    >>> # fig, axes = plot_model_comparison_with_posterior(
    >>> #     time, position, results_a, results_b, diagnostics_a, diagnostics_b,
    >>> #     spike_times=spike_times, spike_counts=spike_counts, place_field_peaks=pf_peaks
    >>> # )
    """
    metrics = ["hpd_overlap", "kl_divergence", "spike_prob"]
    # Match Figure 3 styling: labels and colors from COLORS dict
    ylabels = ["HPD Overlap", "KL Divergence", r"$-\log_{10}(p)$"]
    colors = [COLORS["hpd_overlap"], COLORS["kl_divergence"], COLORS["metric_combined"]]
    # Direction indicators: which direction indicates worse fit
    worse_fit_directions = ["↓ Worse fit", "↑ Worse fit", "↑ Worse fit"]

    # Create 6x2 grid: predictive + likelihood + raster + 3 diagnostics
    # Use gridspec to manually share y-axes within each row
    if fig is None:
        fig = plt.figure(figsize=figsize, constrained_layout=True)
    gs = fig.add_gridspec(6, 2, height_ratios=[2, 2, 1.5, 1, 1, 1])

    # Create axes with shared x and shared y within each row
    axes = np.empty((6, 2), dtype=object)

    # Row 0: Predictive posterior heatmaps (share y within row)
    axes[0, 0] = fig.add_subplot(gs[0, 0])
    axes[0, 1] = fig.add_subplot(gs[0, 1], sharex=axes[0, 0], sharey=axes[0, 0])

    # Row 1: Likelihood heatmaps (share y within row, share x with row 0)
    axes[1, 0] = fig.add_subplot(gs[1, 0], sharex=axes[0, 0], sharey=axes[0, 0])
    axes[1, 1] = fig.add_subplot(gs[1, 1], sharex=axes[0, 0], sharey=axes[0, 0])

    # Row 2: Spike raster (share y within row, share x with row 0)
    axes[2, 0] = fig.add_subplot(gs[2, 0], sharex=axes[0, 0])
    axes[2, 1] = fig.add_subplot(gs[2, 1], sharex=axes[0, 0], sharey=axes[2, 0])

    # Row 3: HPD overlap (share y within row, share x with row 0)
    axes[3, 0] = fig.add_subplot(gs[3, 0], sharex=axes[0, 0])
    axes[3, 1] = fig.add_subplot(gs[3, 1], sharex=axes[0, 0], sharey=axes[3, 0])

    # Row 4: KL divergence (share y within row, share x with row 0)
    axes[4, 0] = fig.add_subplot(gs[4, 0], sharex=axes[0, 0])
    axes[4, 1] = fig.add_subplot(gs[4, 1], sharex=axes[0, 0], sharey=axes[4, 0])

    # Row 5: Spike probability (share y within row, share x with row 0)
    axes[5, 0] = fig.add_subplot(gs[5, 0], sharex=axes[0, 0])
    axes[5, 1] = fig.add_subplot(gs[5, 1], sharex=axes[0, 0], sharey=axes[5, 0])

    if time_slice_ind is None:
        time_slice_ind = slice(None)

    # Compute mask for times with spikes (for likelihood plotting)
    has_spikes_mask: NDArray[np.bool_] | None = None
    if spike_counts is not None:
        # Sum across cells to get total spikes per time point
        has_spikes_mask = spike_counts.sum(axis=1) > 0

    # --- Row 0: Predictive posterior ---
    for col, (results, model_name) in enumerate(
        [(results_a, model_a_name), (results_b, model_b_name)]
    ):
        ax = axes[0, col]
        _plot_distribution_heatmap(
            ax=ax,
            distribution_da=results.predictive_posterior,
            time=time,
            position=position,
            time_slice_ind=time_slice_ind,
            show_position=True,
            cmap=CMAP_POSTERIOR,
        )
        ax.set_title(model_name, fontsize=7)
        ax.set_ylabel("Predictive" if col == 0 else "", fontsize=7, labelpad=7)
        ax.set_xlabel("")
        ax.tick_params(labelsize=6, labelbottom=False)
        if col == 0:
            ax.legend(loc="upper left", fontsize=6, frameon=False)

    # --- Row 1: Likelihood overlay (predictive underlay + likelihood at spike times) ---
    for col, (results, _model_name) in enumerate(
        [(results_a, model_a_name), (results_b, model_b_name)]
    ):
        ax = axes[1, col]

        # Step 1: Plot predictive as faint underlay using xarray (handles coordinates)
        _plot_distribution_heatmap(
            ax=ax,
            distribution_da=results.predictive_posterior,
            time=time,
            position=position,
            time_slice_ind=time_slice_ind,
            show_position=False,
            cmap=CMAP_POSTERIOR,
        )
        # Reduce underlay opacity (xarray .plot() uses pcolormesh -> collections)
        for artist in list(ax.images) + list(ax.collections):
            artist.set_alpha(0.35)

        # Step 2: Overlay likelihood at spike times using shared column renderer
        if "log_likelihood" in results:
            lik_da = xr.apply_ufunc(np.exp, results["log_likelihood"]).dropna(
                "state_bins", how="all"
            )
            try:
                lik_unstacked = lik_da.unstack("state_bins")
                if "state" in lik_unstacked.dims:
                    lik_unstacked = lik_unstacked.sum("state")
                lik_sliced = lik_unstacked.isel(time=time_slice_ind)
            except (ValueError, KeyError):
                lik_sliced = lik_da.isel(time=time_slice_ind)

            lik_np = lik_sliced.values  # (n_time_slice, n_position)

            # Get coordinate arrays for extent
            time_coords = lik_sliced.coords["time"].values
            pos_coords = lik_sliced.coords["position"].values
            t0, t1 = float(time_coords[0]), float(time_coords[-1])
            p0, p1 = float(pos_coords[0]), float(pos_coords[-1])
            # Half-pixel padding for imshow extent
            dt = (t1 - t0) / max(len(time_coords) - 1, 1) / 2
            dp = (p1 - p0) / max(len(pos_coords) - 1, 1) / 2
            extent = (t0 - dt, t1 + dt, p0 - dp, p1 + dp)

            has_spk_slice = (
                has_spikes_mask[time_slice_ind]
                if has_spikes_mask is not None
                else np.ones(lik_np.shape[0], dtype=bool)
            )

            plot_likelihood_columns(
                ax,
                lik_np,
                has_spk_slice,
                n_time=len(time_coords),
                extent=extent,
                cmap=CMAP_LIKELIHOOD,
            )

        # Position overlay
        time_arr = np.asarray(time)
        ax.scatter(
            time_arr[time_slice_ind],
            position[time_slice_ind],
            c=COLORS["ground_truth"],
            s=1,
            alpha=0.85,
        )

        ax.set_title("")
        ax.set_ylabel("Likelihood" if col == 0 else "", fontsize=7, labelpad=7)
        ax.set_xlabel("")
        ax.tick_params(labelsize=6, labelbottom=False)

    # Add 1D track graph on right edge (right column, predictive and likelihood rows)
    if track_graph is not None:
        time_arr = np.asarray(time)
        sliced_time = time_arr[time_slice_ind]
        x_pos = float(sliced_time[-1])
        for row_idx in range(2):
            plot_track_graph_1d(
                track_graph,
                ax=axes[row_idx, 1],
                edge_order=edge_order,
                edge_spacing=edge_spacing,
                other_axis_start=x_pos,
                edge_linewidth=3,
                reward_well_size=20,
                reward_well_nodes=list(range(6)),
            )

    # Row 2: Spike raster (both columns show same raster, sorted by place field peak)
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
            ax = axes[2, col]
            plot_raster(
                spike_times,
                time_slice,
                ax=ax,
                sort_order=sort_order,
            )
            ax.set_ylabel("Neuron" if col == 0 else "", fontsize=7, labelpad=7)
            ax.set_xlabel("")
            ax.tick_params(labelsize=6, labelbottom=False)

    # Rows 3-5: Diagnostic scatter plots
    for i, (metric, ylabel, color, worse_dir) in enumerate(
        zip(metrics, ylabels, colors, worse_fit_directions, strict=True)
    ):
        row = i + 3  # Offset by 3 for distribution and raster rows
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
            spike_times=spike_times,
            show_running_average=show_running_average,
            running_average_window=running_average_window,
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
            spike_times=spike_times,
            show_running_average=show_running_average,
            running_average_window=running_average_window,
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
    for row in range(6):
        axes[row, 1].tick_params(labelleft=False)

    return fig, axes


def plot_single_model_diagnostics(
    time: NDArray[np.float64] | pd.Index,
    position: NDArray[np.float64],
    results: xr.Dataset,
    diagnostics: dict[str, NDArray[np.float64]],
    spike_times: list[NDArray[np.float64]] | None = None,
    spike_counts: NDArray[np.int64] | None = None,
    place_field_peaks: NDArray[np.float64] | None = None,
    time_slice_ind: slice | None = None,
    model_name: str = "Continuous",
    thresholds: dict[str, float] | None = None,
    track_graph: nx.Graph | None = None,
    edge_order: list[tuple[int, int]] | None = None,
    edge_spacing: float | list[float] = 0.0,
    show_running_average: bool = False,
    running_average_window: float = 0.050,
    fig: Figure | None = None,
) -> tuple[Figure, NDArray[np.object_]]:
    """Create single-model diagnostic figure with 6 rows.

    Layout (6 rows, single column):
    - Row 0: Predictive posterior with animal position overlay
    - Row 1: Likelihood at spike times with position overlay
    - Row 2: Spike raster (sorted by place field peak)
    - Row 3: HPD overlap scatter
    - Row 4: KL divergence scatter
    - Row 5: Spike probability scatter (-log10 scale)

    Parameters
    ----------
    time : np.ndarray or pd.Index
        Time values.
    position : np.ndarray, shape (n_time,)
        Animal position values.
    results : xr.Dataset
        Decoding results with predictive_posterior and log_likelihood.
    diagnostics : dict[str, np.ndarray]
        Diagnostics with keys 'hpd_overlap', 'kl_divergence', 'spike_prob'.
    spike_times : list[np.ndarray], optional
        List of spike time arrays, one per neuron.
    spike_counts : np.ndarray, shape (n_time, n_cells), optional
        Spike count matrix.
    place_field_peaks : np.ndarray, shape (n_cells,), optional
        Place field peak positions for raster sorting.
    time_slice_ind : slice, optional
        Time slice to plot. If None, plots all time points.
    model_name : str, default "Continuous"
        Model name for title.
    thresholds : dict[str, float], optional
        Thresholds for horizontal lines on diagnostic plots.
    track_graph : nx.Graph, optional
        Track graph for 1D linearized track visualization.
    edge_order : list[tuple[int, int]], optional
        Order of edges for linearization.
    edge_spacing : float or list[float], default 0.0
        Spacing between edges.
    show_running_average : bool, default False
        If True, overlay a running average on diagnostic scatters.
    running_average_window : float, default 0.050
        Window size in seconds for running average.
    fig : Figure, optional
        Existing figure to draw into.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object.
    axes : np.ndarray[plt.Axes]
        Array of axes objects with shape (6,).
    """
    metrics = ["hpd_overlap", "kl_divergence", "spike_prob"]
    ylabels = ["HPD Overlap", "KL Divergence", r"$-\log_{10}(p)$"]
    colors = [COLORS["hpd_overlap"], COLORS["kl_divergence"], COLORS["metric_combined"]]
    worse_fit_directions = ["↓ Worse fit", "↑ Worse fit", "↑ Worse fit"]

    if fig is None:
        fig = plt.figure(figsize=(7.0, 8.0), constrained_layout=True)
    gs = fig.add_gridspec(6, 1, height_ratios=[2, 2, 1.5, 1, 1, 1])

    axes = np.empty(6, dtype=object)
    axes[0] = fig.add_subplot(gs[0])
    for i in range(1, 6):
        axes[i] = fig.add_subplot(gs[i], sharex=axes[0])

    if time_slice_ind is None:
        time_slice_ind = slice(None)

    # Mask for times with spikes
    has_spikes_mask: NDArray[np.bool_] | None = None
    if spike_counts is not None:
        has_spikes_mask = spike_counts.sum(axis=1) > 0

    # Row 0: Predictive posterior
    _plot_distribution_heatmap(
        ax=axes[0],
        distribution_da=results.predictive_posterior,
        time=time,
        position=position,
        time_slice_ind=time_slice_ind,
        show_position=True,
        cmap=CMAP_POSTERIOR,
    )
    axes[0].set_title(model_name, fontsize=7)
    axes[0].set_ylabel("Predictive", fontsize=7, labelpad=7)
    axes[0].set_xlabel("")
    axes[0].tick_params(labelsize=6, labelbottom=False)
    axes[0].legend(loc="upper left", fontsize=6, frameon=False)

    # Row 1: Likelihood overlay at spike times
    ax_lik = axes[1]
    ax_lik.set_facecolor("black")

    if "log_likelihood" in results:
        lik_da = xr.apply_ufunc(np.exp, results["log_likelihood"]).dropna("state_bins", how="all")
        try:
            lik_unstacked = lik_da.unstack("state_bins")
            if "state" in lik_unstacked.dims:
                lik_unstacked = lik_unstacked.sum("state")
            lik_sliced = lik_unstacked.isel(time=time_slice_ind)
        except (ValueError, KeyError):
            lik_sliced = lik_da.isel(time=time_slice_ind)

        lik_np = lik_sliced.values
        time_coords = lik_sliced.coords["time"].values
        pos_coords = lik_sliced.coords["position"].values
        t0, t1 = float(time_coords[0]), float(time_coords[-1])
        p0, p1 = float(pos_coords[0]), float(pos_coords[-1])
        dt = (t1 - t0) / max(len(time_coords) - 1, 1) / 2
        dp = (p1 - p0) / max(len(pos_coords) - 1, 1) / 2
        extent = (t0 - dt, t1 + dt, p0 - dp, p1 + dp)

        has_spk_slice = (
            has_spikes_mask[time_slice_ind]
            if has_spikes_mask is not None
            else np.ones(lik_np.shape[0], dtype=bool)
        )

        plot_likelihood_columns(
            ax_lik,
            lik_np,
            has_spk_slice,
            n_time=len(time_coords),
            extent=extent,
            cmap=CMAP_LIKELIHOOD,
        )

    # Position overlay
    time_arr = np.asarray(time)
    ax_lik.scatter(
        time_arr[time_slice_ind],
        position[time_slice_ind],
        c=COLORS["ground_truth"],
        s=1,
        alpha=0.85,
    )
    ax_lik.set_ylabel("Likelihood", fontsize=7, labelpad=7)
    ax_lik.set_xlabel("")
    ax_lik.tick_params(labelsize=6, labelbottom=False)

    # 1D track graph on right edge of predictive and likelihood rows
    if track_graph is not None:
        sliced_time = time_arr[time_slice_ind]
        x_pos = float(sliced_time[-1])
        for row_idx in range(2):
            plot_track_graph_1d(
                track_graph,
                ax=axes[row_idx],
                edge_order=edge_order,
                edge_spacing=edge_spacing,
                other_axis_start=x_pos,
                edge_linewidth=3,
                reward_well_size=20,
                reward_well_nodes=list(range(6)),
            )

    # Row 2: Spike raster
    if spike_times is not None:
        sort_order = np.argsort(place_field_peaks) if place_field_peaks is not None else None
        sliced_time = time_arr[time_slice_ind]
        time_slice = slice(float(sliced_time[0]), float(sliced_time[-1]))
        plot_raster(spike_times, time_slice, ax=axes[2], sort_order=sort_order)
        axes[2].set_ylabel("Neuron", fontsize=7, labelpad=7)
        axes[2].set_xlabel("")
        axes[2].tick_params(labelsize=6, labelbottom=False)

    # Rows 3-5: Diagnostic scatters
    for i, (metric, ylabel, color, worse_dir) in enumerate(
        zip(metrics, ylabels, colors, worse_fit_directions, strict=True)
    ):
        row = i + 3
        threshold = thresholds.get(metric) if thresholds else None
        plot_per_cell_diagnostic_scatter(
            time,
            diagnostics,
            time_slice_ind=time_slice_ind,
            threshold=threshold,
            ax=axes[row],
            metric_name=metric,
            color=color,
            ylabel=ylabel,
            show_xlabel=(i == 2),
            spike_times=spike_times,
            show_running_average=show_running_average,
            running_average_window=running_average_window,
        )
        axes[row].text(
            1.01,
            0.5,
            worse_dir,
            transform=axes[row].transAxes,
            fontsize=6,
            va="center",
            ha="left",
        )

    return fig, axes


def plot_per_spike_metric_hexbin_row(
    diagnostics_a: dict[str, Any],
    diagnostics_b: dict[str, Any],
    axes: list[Axes],
    *,
    model_a_name: str = "Continuous",
    model_b_name: str = "Cont-Frag",
) -> None:
    """Plot a 1x3 row of hexbin densities comparing per-spike diagnostics between two decoders.

    Each panel shows one diagnostic on the x-axis (model A) and the same
    diagnostic on the y-axis (model B). Each hexagon's colour is the count of
    spike events landing in that bin. Points on the identity line indicate
    decoder agreement on that spike.

    Both diagnostics dicts must carry the same set of per-spike events in the
    same order (i.e. ``event_*`` arrays produced from the same spike trains by
    :func:`statespacecheck_paper.real_data_analysis.compute_model_diagnostics`).

    Parameters
    ----------
    diagnostics_a, diagnostics_b : dict[str, np.ndarray]
        Per-spike diagnostic dicts with keys ``event_hpd_overlap``,
        ``event_kl_divergence``, ``event_spike_prob`` (each shape
        ``(n_spikes,)``).
    axes : list[matplotlib.axes.Axes]
        Three axes, one per metric (HPD overlap, KL divergence,
        ``-log10(p)``).
    model_a_name, model_b_name : str
        Axis labels for each decoder.
    """
    from matplotlib.colors import LinearSegmentedColormap

    if len(axes) != 3:
        raise ValueError(f"axes must have length 3, got {len(axes)}")

    metric_specs = [
        ("event_hpd_overlap", "HPD overlap", COLORS["hpd_overlap"], False),
        ("event_kl_divergence", "KL divergence", COLORS["kl_divergence"], False),
        ("event_spike_prob", r"$-\log_{10}(p)$", COLORS["metric_combined"], True),
    ]

    for ax, (key, title, color, log_transform) in zip(axes, metric_specs, strict=True):
        data_a = np.asarray(diagnostics_a[key], dtype=np.float64)
        data_b = np.asarray(diagnostics_b[key], dtype=np.float64)
        if log_transform:
            data_a = -np.log10(np.maximum(data_a, 1e-10))
            data_b = -np.log10(np.maximum(data_b, 1e-10))

        valid = np.isfinite(data_a) & np.isfinite(data_b)
        data_a = data_a[valid]
        data_b = data_b[valid]

        cmap = LinearSegmentedColormap.from_list("custom", ["white", color])
        ax.hexbin(
            data_a,
            data_b,
            gridsize=40,
            cmap=cmap,
            bins="log",
            mincnt=1,
            rasterized=True,
        )

        # Identity line for visual agreement reference.
        lims = (
            min(ax.get_xlim()[0], ax.get_ylim()[0]),
            max(ax.get_xlim()[1], ax.get_ylim()[1]),
        )
        ax.plot(lims, lims, color=COLORS["threshold"], lw=0.8, ls="--", alpha=0.7)
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_aspect("equal", adjustable="box")

        ax.set_xlabel(model_a_name, fontsize=7, labelpad=4)
        ax.set_ylabel(model_b_name, fontsize=7, labelpad=4)
        ax.set_title(title, fontsize=7)
        ax.tick_params(labelsize=6)

        ax.text(
            0.02,
            0.98,
            f"n={len(data_a):,}",
            transform=ax.transAxes,
            fontsize=6,
            va="top",
            ha="left",
            color="0.4",
        )


def plot_metrics_vs_position(
    linear_position: NDArray[np.float64],
    diagnostics_a: dict[str, NDArray[np.float64]],
    diagnostics_b: dict[str, NDArray[np.float64]],
    model_a_name: str = "Continuous",
    model_b_name: str = "Continuous-Fragmented",
    figsize: tuple[float, float] = (7.0, 6.0),
    track_graph: nx.Graph | None = None,
    edge_order: list[tuple[int, int]] | None = None,
    edge_spacing: float | list[float] = 0.0,
) -> tuple[Figure, NDArray[np.object_]]:
    """Plot diagnostic metrics vs linear position for two models.

    Creates a 3x2 grid of hexbin plots showing the relationship between
    linear position and each diagnostic metric. This helps identify
    position-dependent model failures.

    Parameters
    ----------
    linear_position : np.ndarray, shape (n_time,)
        Linear position of the animal at each time point.
    diagnostics_a : dict[str, np.ndarray]
        Diagnostics for model A with keys 'hpd_overlap', 'kl_divergence', 'spike_prob'.
        Each array has shape (n_time, n_cells).
    diagnostics_b : dict[str, np.ndarray]
        Diagnostics for model B with same keys.
    model_a_name : str, default "Continuous"
        Name for model A (column title).
    model_b_name : str, default "Continuous-Fragmented"
        Name for model B (column title).
    figsize : tuple[float, float], default (7.0, 6.0)
        Figure size in inches.
    track_graph : nx.Graph, optional
        Track graph for 1D linearized track visualization.
    edge_order : list[tuple[int, int]], optional
        Order of edges for linearization.
    edge_spacing : float or list[float], default 0.0
        Spacing between edges.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object.
    axes : np.ndarray[plt.Axes]
        Array of axes objects with shape (3, 2).

    Examples
    --------
    >>> import numpy as np
    >>> position = np.random.rand(100) * 200  # 0-200 cm track
    >>> diag_a = {
    ...     "hpd_overlap": np.random.rand(100, 10),
    ...     "kl_divergence": np.random.rand(100, 10),
    ...     "spike_prob": np.random.rand(100, 10),
    ... }
    >>> diag_b = {k: np.random.rand(100, 10) for k in diag_a}
    >>> fig, axes = plot_metrics_vs_position(position, diag_a, diag_b)
    >>> plt.close(fig)
    """
    metrics = ["hpd_overlap", "kl_divergence", "spike_prob"]
    ylabels = ["HPD Overlap", "KL Divergence", r"$-\log_{10}(p)$"]
    colors = [COLORS["hpd_overlap"], COLORS["kl_divergence"], COLORS["metric_combined"]]
    worse_fit_directions = ["↓ Worse fit", "↑ Worse fit", "↑ Worse fit"]

    fig, axes = plt.subplots(3, 2, figsize=figsize, constrained_layout=True)

    for i, (metric, ylabel, _color, worse_dir) in enumerate(
        zip(metrics, ylabels, colors, worse_fit_directions, strict=True)
    ):
        for col, (diagnostics, model_name) in enumerate(
            [(diagnostics_a, model_a_name), (diagnostics_b, model_b_name)]
        ):
            ax = axes[i, col]

            # Get data for this metric
            data = diagnostics[metric].copy()

            # Transform spike_prob to -log10 scale
            if metric == "spike_prob":
                data = -np.log10(np.maximum(data, 1e-10))

            # Expand position to match data shape (n_time,) -> (n_time, n_cells)
            n_time, n_cells = data.shape
            position_expanded = np.tile(linear_position[:, np.newaxis], (1, n_cells))

            # Flatten for hexbin
            x = position_expanded.ravel()
            y = data.ravel()

            # Remove NaN values
            valid_mask = ~np.isnan(x) & ~np.isnan(y)
            x = x[valid_mask]
            y = y[valid_mask]

            # Hexbin plot
            ax.hexbin(
                x,
                y,
                gridsize=50,
                cmap="Blues",
                bins="log",
                mincnt=1,
                alpha=0.8,
                rasterized=True,
            )

            # Styling
            ax.set_ylabel(ylabel if col == 0 else "", fontsize=7, labelpad=7)
            ax.set_xlabel("Linear Position (cm)" if i == 2 else "", fontsize=7, labelpad=7)
            ax.tick_params(labelsize=6, labelbottom=(i == 2))

            # Add column titles on first row
            if i == 0:
                ax.set_title(model_name, fontsize=7)

            # Add direction indicator on right side of right column
            if col == 1:
                ax.text(
                    1.01,
                    0.5,
                    worse_dir,
                    transform=ax.transAxes,
                    fontsize=6,
                    va="center",
                    ha="left",
                )

            # Add 1D track graph at top of plot (right column only)
            if track_graph is not None and col == 1 and i == 0:
                # Get y-axis limits to position track at top
                ylim = ax.get_ylim()
                y_top = ylim[1]
                plot_track_graph_1d(
                    track_graph,
                    ax=ax,
                    edge_order=edge_order,
                    edge_spacing=edge_spacing,
                    other_axis_start=y_top,
                    edge_linewidth=2,
                    reward_well_size=15,
                    reward_well_nodes=list(range(6)),
                )

    # Hide y-tick labels on right column
    for row in range(3):
        axes[row, 1].tick_params(labelleft=False)

    return fig, axes


def plot_metrics_time_vs_position_comparison(
    linear_position: NDArray[np.float64],
    diagnostics: dict[str, NDArray[np.float64]],
    model_name: str = "Continuous",
    figsize: tuple[float, float] = (7.0, 6.0),
    track_graph: nx.Graph | None = None,
    edge_order: list[tuple[int, int]] | None = None,
    edge_spacing: float | list[float] = 0.0,
    fig: Figure | None = None,
    axes: NDArray[np.object_] | None = None,
) -> tuple[Figure, NDArray[np.object_]]:
    """Plot diagnostic metrics vs linear position for a single model.

    Creates a 3x1 grid showing:
    - Rows: HPD Overlap, KL Divergence, Spike Probability

    Each plot shows linear position on x-axis and the metric value on y-axis,
    allowing comparison of position-dependent model performance.

    Parameters
    ----------
    linear_position : np.ndarray, shape (n_time,)
        Linear position of the animal at each time point.
    diagnostics : dict[str, np.ndarray]
        Diagnostics with keys 'hpd_overlap', 'kl_divergence', 'spike_prob'.
        Each array has shape (n_time, n_cells).
    model_name : str, default "Continuous"
        Name for the model.
    figsize : tuple[float, float], default (7.0, 6.0)
        Figure size in inches.
    track_graph : nx.Graph, optional
        Track graph for 1D linearized track visualization.
    edge_order : list[tuple[int, int]], optional
        Order of edges for linearization.
    edge_spacing : float or list[float], default 0.0
        Spacing between edges.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object.
    axes : np.ndarray[plt.Axes]
        Array of axes objects with shape (3,).

    Examples
    --------
    >>> import numpy as np
    >>> position = np.random.rand(1000) * 200
    >>> diagnostics = {
    ...     "hpd_overlap": np.random.rand(1000, 10),
    ...     "kl_divergence": np.random.rand(1000, 10),
    ...     "spike_prob": np.random.rand(1000, 10),
    ... }
    >>> fig, axes = plot_metrics_time_vs_position_comparison(position, diagnostics)
    >>> plt.close(fig)
    """
    metrics = ["hpd_overlap", "kl_divergence", "spike_prob"]
    ylabels = ["HPD Overlap", "KL Divergence", r"$-\log_{10}(p)$"]
    worse_fit_directions = ["↓ Worse fit", "↑ Worse fit", "↑ Worse fit"]

    if fig is None and axes is None:
        fig, axes = plt.subplots(3, 1, figsize=figsize, constrained_layout=True)
    elif fig is not None and axes is None:
        axes = fig.subplots(3, 1)
    else:
        assert axes is not None
        fig = axes.flat[0].get_figure()
    axes = np.atleast_1d(axes)

    for ax, metric, ylabel, worse_dir in zip(
        axes, metrics, ylabels, worse_fit_directions, strict=True
    ):
        # Get data for this metric
        data = diagnostics[metric].copy()

        # Transform spike_prob to -log10 scale
        if metric == "spike_prob":
            data = -np.log10(np.maximum(data, 1e-10))

        # Expand position to match data shape (n_time,) -> (n_time, n_cells)
        n_time, n_cells = data.shape
        position_expanded = np.tile(linear_position[:, np.newaxis], (1, n_cells))

        # Flatten for hexbin
        x = position_expanded.ravel()
        y = data.ravel()

        # Remove NaN values
        valid_mask = ~np.isnan(x) & ~np.isnan(y)
        x = x[valid_mask]
        y = y[valid_mask]

        # Hexbin plot: position on x-axis, metric on y-axis
        ax.hexbin(
            x,
            y,
            gridsize=50,
            cmap="Blues",
            bins="log",
            mincnt=1,
            alpha=0.8,
            rasterized=True,
        )

        # Styling
        ax.set_ylabel(ylabel, fontsize=7, labelpad=7)

        # Add direction indicator on right side
        ax.text(
            1.01,
            0.5,
            worse_dir,
            transform=ax.transAxes,
            fontsize=6,
            va="center",
            ha="left",
        )

    # Add title on first axes only
    axes[0].set_title(model_name, fontsize=7)

    # Add x-label on last axes only
    axes[-1].set_xlabel("Linear Position (cm)", fontsize=7, labelpad=7)
    for ax in axes[:-1]:
        ax.tick_params(labelbottom=False)

    # Share x-axes across all plots
    x_min = min(ax.get_xlim()[0] for ax in axes)
    x_max = max(ax.get_xlim()[1] for ax in axes)
    for ax in axes:
        ax.set_xlim(x_min, x_max)

    # Add horizontal track graph at bottom of each row if provided
    if track_graph is not None:
        for ax in axes:
            y_bottom = ax.get_ylim()[0]
            plot_track_graph_1d(
                track_graph,
                ax=ax,
                edge_order=edge_order,
                edge_spacing=edge_spacing,
                other_axis_start=y_bottom,
                edge_linewidth=3,
                reward_well_size=15,
                reward_well_nodes=list(range(6)),
                orientation="horizontal",
            )

    return fig, axes


def plot_metric_distributions(
    diagnostics_a: dict[str, NDArray[np.float64]],
    diagnostics_b: dict[str, NDArray[np.float64]],
    model_a_name: str = "Continuous",
    model_b_name: str = "Continuous-Fragmented",
    figsize: tuple[float, float] = (7.0, 5.0),
    show_diff: bool = True,
    fig: Figure | None = None,
    axes: NDArray[np.object_] | None = None,
) -> tuple[Figure, NDArray[np.object_]]:
    """Plot hexbin density comparison of diagnostic metrics between two models.

    Creates a grid of hexbin density plots (top row) where each hexagon represents
    the density of (time, cell) pairs. X-axis shows model A's metric value,
    Y-axis shows model B's metric value. Points on the diagonal indicate agreement.
    When ``show_diff`` is True, a second row shows histograms of the per-(time, cell)
    difference (model B - model A).

    Parameters
    ----------
    diagnostics_a : dict[str, np.ndarray]
        Diagnostics for model A with keys 'hpd_overlap', 'kl_divergence', 'spike_prob'.
        Each array has shape (n_time, n_cells).
    diagnostics_b : dict[str, np.ndarray]
        Diagnostics for model B with same keys.
    model_a_name : str, default "Continuous"
        Name for model A (x-axis label).
    model_b_name : str, default "Continuous-Fragmented"
        Name for model B (y-axis label).
    figsize : tuple[float, float], default (7.0, 5.0)
        Figure size in inches.
    show_diff : bool, default True
        If True, add a second row with difference histograms (model B - model A).

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object.
    axes : np.ndarray[plt.Axes]
        Array of axes objects with shape (2, 3) when ``show_diff`` is True,
        or (3,) when False.

    Examples
    --------
    >>> import numpy as np
    >>> diag_a = {
    ...     "hpd_overlap": np.random.rand(100, 10),
    ...     "kl_divergence": np.random.rand(100, 10),
    ...     "spike_prob": np.random.rand(100, 10),
    ... }
    >>> diag_b = {k: np.random.rand(100, 10) for k in diag_a}
    >>> fig, axes = plot_metric_distributions(diag_a, diag_b)
    >>> plt.close(fig)
    """
    from matplotlib.colors import LinearSegmentedColormap

    metrics = ["hpd_overlap", "kl_divergence", "spike_prob"]
    titles = ["HPD Overlap", "KL Divergence", r"$-\log_{10}(p)$"]
    # Create colormaps from white to each metric's color
    colors_list = [
        COLORS["hpd_overlap"],
        COLORS["kl_divergence"],
        COLORS["metric_combined"],
    ]
    cmaps = [LinearSegmentedColormap.from_list("custom", ["white", c]) for c in colors_list]

    n_rows = 2 if show_diff else 1
    if fig is None and axes is None:
        fig, axes = plt.subplots(
            n_rows,
            3,
            figsize=figsize,
            constrained_layout=True,
        )
    elif fig is not None and axes is None:
        axes = fig.subplots(n_rows, 3)
    else:
        assert axes is not None
        fig = axes.flat[0].get_figure()
    axes = np.atleast_2d(axes)

    for i, (metric, title, cmap, color) in enumerate(
        zip(metrics, titles, cmaps, colors_list, strict=True)
    ):
        ax = axes[0, i]

        # Get data and flatten
        data_a = diagnostics_a[metric].ravel()
        data_b = diagnostics_b[metric].ravel()

        # Transform spike_prob to -log10 scale
        if metric == "spike_prob":
            data_a = -np.log10(np.maximum(data_a, 1e-10))
            data_b = -np.log10(np.maximum(data_b, 1e-10))

        # Create mask for valid (non-NaN) values in both arrays
        valid_mask = ~np.isnan(data_a) & ~np.isnan(data_b)
        data_a = data_a[valid_mask]
        data_b = data_b[valid_mask]

        # Hexbin density plot: model A vs model B
        ax.hexbin(
            data_a,
            data_b,
            gridsize=50,
            cmap=cmap,
            bins="log",
            mincnt=1,
            alpha=0.8,
            rasterized=True,
        )

        # Add identity line (y=x)
        lims = [
            min(ax.get_xlim()[0], ax.get_ylim()[0]),
            max(ax.get_xlim()[1], ax.get_ylim()[1]),
        ]
        ax.plot(
            lims,
            lims,
            color=COLORS["threshold"],
            linewidth=1,
            linestyle="--",
            alpha=0.7,
        )
        ax.set_xlim(lims)
        ax.set_ylim(lims)

        # Labels and styling
        ax.set_xlabel(model_a_name, fontsize=7, labelpad=7)
        ax.set_ylabel(model_b_name if i == 0 else "", fontsize=7, labelpad=7)
        ax.set_title(title, fontsize=7)
        ax.tick_params(labelsize=6)
        ax.set_aspect("equal", adjustable="box")

        # Add n values as text
        ax.text(
            0.02,
            0.98,
            f"n={len(data_a):,}",
            transform=ax.transAxes,
            fontsize=6,
            va="top",
            ha="left",
            color="gray",
        )

        # Difference histogram (bottom row)
        if show_diff:
            ax_diff = axes[1, i]
            diff = data_b - data_a

            ax_diff.hist(
                diff,
                bins=100,
                color=color,
                alpha=0.7,
                edgecolor="none",
                rasterized=True,
                density=True,
            )

            # Reference line at zero
            ax_diff.axvline(
                0,
                color=COLORS["threshold"],
                linewidth=1,
                linestyle="--",
                alpha=0.7,
            )

            # Arrows indicating "better fit" direction for each model
            # HPD overlap: higher is better → positive diff means B better
            # KL divergence: lower is better → negative diff means B better
            # Spike prob (-log10 p): lower is better → negative diff means B better
            if metric == "hpd_overlap":
                b_side = "right"
                a_side = "left"
            else:
                b_side = "left"
                a_side = "right"

            # Place arrows in the upper-right area, stacked vertically
            for arrow_y, side, label in [
                (0.92, b_side, f"{model_b_name} better"),
                (0.82, a_side, f"{model_a_name} better"),
            ]:
                if side == "right":
                    x_start, x_end = 0.60, 0.78
                    text_x, text_ha = x_start - 0.02, "right"
                else:
                    x_start, x_end = 0.40, 0.22
                    text_x, text_ha = x_start + 0.02, "left"

                ax_diff.annotate(
                    "",
                    xy=(x_end, arrow_y),
                    xytext=(x_start, arrow_y),
                    xycoords="axes fraction",
                    arrowprops=dict(
                        arrowstyle="->",
                        color="0.4",
                        lw=1.0,
                    ),
                )
                ax_diff.text(
                    text_x,
                    arrow_y,
                    label,
                    transform=ax_diff.transAxes,
                    fontsize=5.5,
                    color="0.4",
                    ha=text_ha,
                    va="center",
                )

            # Labels and styling
            ax_diff.set_xlabel(
                f"\u0394 ({model_b_name} \u2212 {model_a_name})",
                fontsize=7,
                labelpad=7,
            )
            ax_diff.set_ylabel("Density" if i == 0 else "", fontsize=7, labelpad=7)
            ax_diff.set_title("Difference", fontsize=7)
            ax_diff.tick_params(labelsize=6)

            # Annotation: n, median, mean
            median_val = float(np.median(diff))
            mean_val = float(np.mean(diff))
            ax_diff.text(
                0.02,
                0.58,
                f"n={len(diff):,}\nmedian={median_val:.3f}\nmean={mean_val:.3f}",
                transform=ax_diff.transAxes,
                fontsize=6,
                va="top",
                ha="left",
                color="gray",
            )

    return fig, axes


def plot_metric_diff_histogram(
    diagnostics_a: dict[str, NDArray[np.float64]],
    diagnostics_b: dict[str, NDArray[np.float64]],
    model_a_name: str = "Continuous",
    model_b_name: str = "Continuous-Fragmented",
    figsize: tuple[float, float] = (7.0, 2.5),
) -> tuple[Figure, NDArray[np.object_]]:
    """Plot histograms of metric differences (model B - model A).

    Creates a 1x3 grid of histograms showing the distribution of per-(time, cell)
    differences for each diagnostic metric.

    Parameters
    ----------
    diagnostics_a : dict[str, np.ndarray]
        Diagnostics for model A with keys 'hpd_overlap', 'kl_divergence', 'spike_prob'.
        Each array has shape (n_time, n_cells).
    diagnostics_b : dict[str, np.ndarray]
        Diagnostics for model B with same keys.
    model_a_name : str, default "Continuous"
        Name for model A.
    model_b_name : str, default "Continuous-Fragmented"
        Name for model B.
    figsize : tuple[float, float], default (7.0, 2.5)
        Figure size in inches.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object.
    axes : np.ndarray[plt.Axes]
        Array of axes objects with shape (3,).

    Examples
    --------
    >>> import numpy as np
    >>> diag_a = {
    ...     "hpd_overlap": np.random.rand(100, 10),
    ...     "kl_divergence": np.random.rand(100, 10),
    ...     "spike_prob": np.random.rand(100, 10),
    ... }
    >>> diag_b = {k: np.random.rand(100, 10) for k in diag_a}
    >>> fig, axes = plot_metric_diff_histogram(diag_a, diag_b)
    >>> plt.close(fig)
    """
    metrics = ["hpd_overlap", "kl_divergence", "spike_prob"]
    titles = ["HPD Overlap", "KL Divergence", r"$-\log_{10}(p)$"]
    colors_list = [
        COLORS["hpd_overlap"],
        COLORS["kl_divergence"],
        COLORS["metric_combined"],
    ]

    fig, axes = plt.subplots(1, 3, figsize=figsize, constrained_layout=True)
    axes = np.atleast_1d(axes)

    for i, (metric, title, color) in enumerate(zip(metrics, titles, colors_list, strict=True)):
        ax = axes[i]

        # Get data and flatten
        data_a = diagnostics_a[metric].ravel()
        data_b = diagnostics_b[metric].ravel()

        # Transform spike_prob to -log10 scale
        if metric == "spike_prob":
            data_a = -np.log10(np.maximum(data_a, 1e-10))
            data_b = -np.log10(np.maximum(data_b, 1e-10))

        # Create mask for valid (non-NaN) values in both arrays
        valid_mask = ~np.isnan(data_a) & ~np.isnan(data_b)
        diff = data_b[valid_mask] - data_a[valid_mask]

        # Histogram
        ax.hist(diff, bins=100, color=color, alpha=0.7, edgecolor="none", rasterized=True)

        # Reference line at zero
        ax.axvline(0, color=COLORS["threshold"], linewidth=1, linestyle="--", alpha=0.7)

        # Labels and styling
        ax.set_xlabel(
            f"\u0394 ({model_b_name} \u2212 {model_a_name})" if i == 1 else "",
            fontsize=7,
            labelpad=7,
        )
        ax.set_ylabel("Count" if i == 0 else "", fontsize=7, labelpad=7)
        ax.set_title(title, fontsize=7)
        ax.tick_params(labelsize=6)

        # Annotation: n, median, mean
        median_val = float(np.median(diff))
        mean_val = float(np.mean(diff))
        ax.text(
            0.02,
            0.98,
            f"n={len(diff):,}\nmedian={median_val:.3f}\nmean={mean_val:.3f}",
            transform=ax.transAxes,
            fontsize=6,
            va="top",
            ha="left",
            color="gray",
        )

    return fig, axes


def plot_metric_diff_hexbin(
    diagnostics_a: dict[str, NDArray[np.float64]],
    diagnostics_b: dict[str, NDArray[np.float64]],
    model_a_name: str = "Continuous",
    model_b_name: str = "Continuous-Fragmented",
    figsize: tuple[float, float] = (7.0, 2.5),
) -> tuple[Figure, NDArray[np.object_]]:
    """Plot hexbin of metric value vs. metric difference (model B - model A).

    Creates a 1x3 grid of hexbin density plots where x-axis is the metric value
    from model A and y-axis is the difference (B - A). Shows how the difference
    varies as a function of metric magnitude.

    Parameters
    ----------
    diagnostics_a : dict[str, np.ndarray]
        Diagnostics for model A with keys 'hpd_overlap', 'kl_divergence', 'spike_prob'.
        Each array has shape (n_time, n_cells).
    diagnostics_b : dict[str, np.ndarray]
        Diagnostics for model B with same keys.
    model_a_name : str, default "Continuous"
        Name for model A (x-axis label).
    model_b_name : str, default "Continuous-Fragmented"
        Name for model B.
    figsize : tuple[float, float], default (7.0, 2.5)
        Figure size in inches.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object.
    axes : np.ndarray[plt.Axes]
        Array of axes objects with shape (3,).

    Examples
    --------
    >>> import numpy as np
    >>> diag_a = {
    ...     "hpd_overlap": np.random.rand(100, 10),
    ...     "kl_divergence": np.random.rand(100, 10),
    ...     "spike_prob": np.random.rand(100, 10),
    ... }
    >>> diag_b = {k: np.random.rand(100, 10) for k in diag_a}
    >>> fig, axes = plot_metric_diff_hexbin(diag_a, diag_b)
    >>> plt.close(fig)
    """
    from matplotlib.colors import LinearSegmentedColormap

    metrics = ["hpd_overlap", "kl_divergence", "spike_prob"]
    titles = ["HPD Overlap", "KL Divergence", r"$-\log_{10}(p)$"]
    colors_list = [
        COLORS["hpd_overlap"],
        COLORS["kl_divergence"],
        COLORS["metric_combined"],
    ]
    cmaps = [LinearSegmentedColormap.from_list("custom", ["white", c]) for c in colors_list]

    fig, axes = plt.subplots(1, 3, figsize=figsize, constrained_layout=True)
    axes = np.atleast_1d(axes)

    for i, (metric, title, cmap) in enumerate(zip(metrics, titles, cmaps, strict=True)):
        ax = axes[i]

        # Get data and flatten
        data_a = diagnostics_a[metric].ravel()
        data_b = diagnostics_b[metric].ravel()

        # Transform spike_prob to -log10 scale
        if metric == "spike_prob":
            data_a = -np.log10(np.maximum(data_a, 1e-10))
            data_b = -np.log10(np.maximum(data_b, 1e-10))

        # Create mask for valid (non-NaN) values in both arrays
        valid_mask = ~np.isnan(data_a) & ~np.isnan(data_b)
        x = data_a[valid_mask]
        diff = data_b[valid_mask] - data_a[valid_mask]

        # Hexbin: metric value vs difference
        ax.hexbin(
            x,
            diff,
            gridsize=50,
            cmap=cmap,
            bins="log",
            mincnt=1,
            alpha=0.8,
            rasterized=True,
        )

        # Reference line at y=0
        ax.axhline(0, color=COLORS["threshold"], linewidth=1, linestyle="--", alpha=0.7)

        # Labels and styling
        ax.set_xlabel(model_a_name if i == 1 else "", fontsize=7, labelpad=7)
        ax.set_ylabel(
            f"\u0394 ({model_b_name} \u2212 {model_a_name})" if i == 0 else "",
            fontsize=7,
            labelpad=7,
        )
        ax.set_title(title, fontsize=7)
        ax.tick_params(labelsize=6)

        # Add n values as text
        ax.text(
            0.02,
            0.98,
            f"n={len(x):,}",
            transform=ax.transAxes,
            fontsize=6,
            va="top",
            ha="left",
            color="gray",
        )

    return fig, axes


def plot_metric_diff_kde(
    diagnostics_a: dict[str, NDArray[np.float64]],
    diagnostics_b: dict[str, NDArray[np.float64]],
    model_a_name: str = "Continuous",
    model_b_name: str = "Continuous-Fragmented",
    figsize: tuple[float, float] = (7.0, 2.5),
    n_eval: int = 500,
) -> tuple[Figure, NDArray[np.object_]]:
    """Plot KDE of metric differences (model B - model A).

    Creates a 1x3 grid of kernel density estimate curves showing the distribution
    of per-(time, cell) differences for each diagnostic metric.

    Parameters
    ----------
    diagnostics_a : dict[str, np.ndarray]
        Diagnostics for model A with keys 'hpd_overlap', 'kl_divergence', 'spike_prob'.
        Each array has shape (n_time, n_cells).
    diagnostics_b : dict[str, np.ndarray]
        Diagnostics for model B with same keys.
    model_a_name : str, default "Continuous"
        Name for model A.
    model_b_name : str, default "Continuous-Fragmented"
        Name for model B.
    figsize : tuple[float, float], default (7.0, 2.5)
        Figure size in inches.
    n_eval : int, default 500
        Number of evaluation points for the KDE.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object.
    axes : np.ndarray[plt.Axes]
        Array of axes objects with shape (3,).

    Examples
    --------
    >>> import numpy as np
    >>> diag_a = {
    ...     "hpd_overlap": np.random.rand(100, 10),
    ...     "kl_divergence": np.random.rand(100, 10),
    ...     "spike_prob": np.random.rand(100, 10),
    ... }
    >>> diag_b = {k: np.random.rand(100, 10) for k in diag_a}
    >>> fig, axes = plot_metric_diff_kde(diag_a, diag_b)
    >>> plt.close(fig)
    """
    from scipy.stats import gaussian_kde

    metrics = ["hpd_overlap", "kl_divergence", "spike_prob"]
    titles = ["HPD Overlap", "KL Divergence", r"$-\log_{10}(p)$"]
    colors_list = [
        COLORS["hpd_overlap"],
        COLORS["kl_divergence"],
        COLORS["metric_combined"],
    ]

    fig, axes = plt.subplots(1, 3, figsize=figsize, constrained_layout=True)
    axes = np.atleast_1d(axes)

    for i, (metric, title, color) in enumerate(zip(metrics, titles, colors_list, strict=True)):
        ax = axes[i]

        # Get data and flatten
        data_a = diagnostics_a[metric].ravel()
        data_b = diagnostics_b[metric].ravel()

        # Transform spike_prob to -log10 scale
        if metric == "spike_prob":
            data_a = -np.log10(np.maximum(data_a, 1e-10))
            data_b = -np.log10(np.maximum(data_b, 1e-10))

        # Create mask for valid (non-NaN) values in both arrays
        valid_mask = ~np.isnan(data_a) & ~np.isnan(data_b)
        diff = data_b[valid_mask] - data_a[valid_mask]

        # KDE
        kde = gaussian_kde(diff)
        x_eval = np.linspace(float(diff.min()), float(diff.max()), n_eval)
        density = kde(x_eval)

        ax.plot(x_eval, density, color=color, linewidth=1.5)
        ax.fill_between(x_eval, density, alpha=0.3, color=color)

        # Reference line at zero
        ax.axvline(0, color=COLORS["threshold"], linewidth=1, linestyle="--", alpha=0.7)

        # Labels and styling
        ax.set_xlabel(
            f"\u0394 ({model_b_name} \u2212 {model_a_name})" if i == 1 else "",
            fontsize=7,
            labelpad=7,
        )
        ax.set_ylabel("Density" if i == 0 else "", fontsize=7, labelpad=7)
        ax.set_title(title, fontsize=7)
        ax.tick_params(labelsize=6)

        # Annotation: n, median
        median_val = float(np.median(diff))
        ax.text(
            0.02,
            0.98,
            f"n={len(diff):,}\nmedian={median_val:.3f}",
            transform=ax.transAxes,
            fontsize=6,
            va="top",
            ha="left",
            color="gray",
        )

    return fig, axes
