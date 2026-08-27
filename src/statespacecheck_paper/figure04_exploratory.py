"""Exploratory two-column Figure-4 model comparison.

``plot_exploratory_model_comparison`` builds the side-by-side Continuous vs.
Continuous-Fragmented comparison used only by the window-selection scripts in
``scripts/exploratory/``; it is not part of the canonical Figure 4. It lives
here (rather than in ``figure04_panels``) so the canonical panel module stays
focused on the published figure. ``plot_model_comparison_with_posterior`` is a
backward-compatible alias for archived notebooks.
"""

from __future__ import annotations

from collections.abc import Hashable, Sequence

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.figure import Figure
from numpy.typing import NDArray

from statespacecheck_paper.diagnostics import SpikeEventDiagnostics
from statespacecheck_paper.figure04_panels import (
    draw_decoder_likelihood_image,
    draw_predictive_heatmap_row,
    draw_track_graph_edges,
    plot_raster,
    plot_spike_event_diagnostic_scatter,
)
from statespacecheck_paper.figure04_plot_primitives import (
    WORSE_FIT_LABEL_GID,
    plot_distribution_heatmap,
)
from statespacecheck_paper.style import (
    CMAP_POSTERIOR,
    COLORS,
    METRIC_SPECS,
)


def plot_exploratory_model_comparison(
    time: NDArray[np.float64] | pd.Index,
    position: NDArray[np.float64],
    results_a: xr.Dataset,
    results_b: xr.Dataset,
    diagnostics_a: SpikeEventDiagnostics,
    diagnostics_b: SpikeEventDiagnostics,
    spike_times: list[NDArray[np.float64]] | None = None,
    spike_counts: NDArray[np.int64] | None = None,
    place_field_peaks: NDArray[np.float64] | None = None,
    time_slice_ind: slice | None = None,
    model_a_name: str = "Continuous",
    model_b_name: str = "Continuous-Fragmented",
    thresholds: dict[str, float] | None = None,
    figsize: tuple[float, float] = (7.0, 11.0),
    track_graph: nx.Graph | None = None,
    edge_order: Sequence[tuple[Hashable, Hashable]] | None = None,
    edge_spacing: float | list[float] = 0.0,
    show_running_average: bool = False,
    running_average_window: float = 0.050,
    fig: Figure | None = None,
) -> tuple[Figure, NDArray[np.object_]]:
    """Create an exploratory two-model comparison for window selection.

    Creates a 6x2 grid with:
    - Row 0: Predictive posterior p(x_t | y_{1:t-1}) with animal position overlay
    - Row 1: Likelihood p(y_t | x_t) with animal position overlay (only at spike times)
    - Row 2: Spike raster (cells sorted by place field peak)
    - Row 3: HPD overlap scatter
    - Row 4: Predictive p-value scatter (-log(p))
    - Row 5: KL divergence scatter

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
    diagnostics_a : SpikeEventDiagnostics
        Per-spike-event diagnostics for model A, supplying the dense ``hpd_overlap``,
        ``kl_divergence``, and ``predictive_pvalue`` matrices.
    diagnostics_b : SpikeEventDiagnostics
        Per-spike-event diagnostics for model B with the same attributes.
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
    >>> # fig, axes = plot_exploratory_model_comparison(
    >>> #     time, position, results_a, results_b, diagnostics_a, diagnostics_b,
    >>> #     spike_times=spike_times, spike_counts=spike_counts, place_field_peaks=pf_peaks
    >>> # )
    """
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

    # Row 4: Predictive p-value (share y within row, share x with row 0)
    axes[4, 0] = fig.add_subplot(gs[4, 0], sharex=axes[0, 0])
    axes[4, 1] = fig.add_subplot(gs[4, 1], sharex=axes[0, 0], sharey=axes[4, 0])

    # Row 5: KL divergence (share y within row, share x with row 0)
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
        draw_predictive_heatmap_row(
            ax,
            results,
            time,
            position,
            time_slice_ind,
            title=model_name,
            ylabel="Predictive" if col == 0 else "",
        )
        if col == 0:
            ax.legend(loc="upper left", fontsize=8, frameon=False)

    # --- Row 1: Likelihood overlay (predictive underlay + likelihood at spike times) ---
    for col, (results, _model_name) in enumerate(
        [(results_a, model_a_name), (results_b, model_b_name)]
    ):
        ax = axes[1, col]

        # Step 1: Plot predictive as faint underlay using xarray (handles coordinates)
        plot_distribution_heatmap(
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

        # Step 2: Overlay the decoder likelihood at spike times.
        draw_decoder_likelihood_image(ax, results, time_slice_ind, has_spikes_mask)

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
        ax.set_ylabel("Likelihood" if col == 0 else "", fontsize=8, labelpad=7)
        ax.set_xlabel("")
        ax.tick_params(labelsize=8, labelbottom=False)

    # Add 1D track graph on right edge (right column, predictive and likelihood rows)
    if track_graph is not None:
        draw_track_graph_edges(
            [axes[0, 1], axes[1, 1]],
            track_graph,
            edge_order,
            edge_spacing,
            time,
            time_slice_ind,
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
            ax.set_ylabel("Neuron" if col == 0 else "", fontsize=8, labelpad=7)
            ax.set_xlabel("")
            ax.tick_params(labelsize=8, labelbottom=False)

    # Rows 3-5: Diagnostic scatter plots
    for i, spec in enumerate(METRIC_SPECS):
        row = i + 3  # Offset by 3 for distribution and raster rows
        threshold = thresholds.get(spec.name) if thresholds else None

        # Model A (left column)
        plot_spike_event_diagnostic_scatter(
            time,
            diagnostics_a,
            time_slice_ind=time_slice_ind,
            threshold=threshold,
            ax=axes[row, 0],
            metric_name=spec.name,
            color=spec.color,
            ylabel=spec.ylabel,
            show_xlabel=(i == 2),
            spike_times=spike_times,
            show_running_average=show_running_average,
            running_average_window=running_average_window,
        )

        # Model B (right column)
        plot_spike_event_diagnostic_scatter(
            time,
            diagnostics_b,
            time_slice_ind=time_slice_ind,
            threshold=threshold,
            ax=axes[row, 1],
            metric_name=spec.name,
            color=spec.color,
            ylabel="",  # Left column has ylabel
            show_xlabel=(i == 2),
            spike_times=spike_times,
            show_running_average=show_running_average,
            running_average_window=running_average_window,
        )

        # Add direction indicator on right side of right column (matching Figure 3)
        worse_fit_label = axes[row, 1].text(
            1.01,
            0.5,
            spec.worse_fit_direction,
            transform=axes[row, 1].transAxes,
            fontsize=8,
            va="center",
            ha="left",
        )
        worse_fit_label.set_gid(WORSE_FIT_LABEL_GID)

    # Hide y-tick labels on right column (since y-axes are shared within rows)
    for row in range(6):
        axes[row, 1].tick_params(labelleft=False)

    return fig, axes


# Compatibility name for existing notebooks. Canonical Figure 4 uses
# ``plot_single_model_diagnostics`` via ``figure04_layout``; this two-column
# comparison exists only for exploratory window-selection scripts.
plot_model_comparison_with_posterior = plot_exploratory_model_comparison
