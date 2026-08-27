"""Raster and diagnostic panels.

The canonical Figure-4 panels: spike raster, spike-event diagnostic scatter,
single-model posterior/likelihood/raster/diagnostic stack, and per-spike metric
hexbin comparison row.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Hashable, Mapping, Sequence
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
from matplotlib.patches import Rectangle
from numpy.typing import NDArray

from statespacecheck_paper.diagnostics import SpikeEventDiagnostics
from statespacecheck_paper.figure04_diagnostics import (
    compute_running_average,
    mean_per_spike_likelihood_by_time,
)
from statespacecheck_paper.figure04_plot_primitives import (
    ANIMAL_POSITION_LABEL_GID,
    THRESHOLD_LABEL_GID,
    WORSE_FIT_LABEL_GID,
    compute_half_pixel_extent,
    decoder_likelihood_to_columns,
    negative_log_pvalue,
    plot_distribution_heatmap,
)
from statespacecheck_paper.figure04_track_plots import plot_track_graph_1d
from statespacecheck_paper.plotting import plot_likelihood_columns
from statespacecheck_paper.style import (
    CMAP_LIKELIHOOD,
    CMAP_POSTERIOR,
    COLORS,
    METRIC_SPECS,
)


@dataclasses.dataclass(frozen=True)
class ModelDiagnosticPanelData:
    """Scientific data required to render one model's diagnostic stack.

    Grouping these related values gives callers one discoverable, typed
    boundary instead of an error-prone sequence of positional arrays. The
    wrapper is frozen, while the potentially large arrays and xarray dataset
    are treated as read-only by convention rather than copied. Shape
    consistency across all inputs is validated at construction.

    Attributes
    ----------
    time : np.ndarray or pd.Index, shape (n_time,)
        Time values (bin centers/starts).
    position : np.ndarray, shape (n_time,)
        Animal position aligned to ``time``.
    results : xr.Dataset
        Decoder outputs carrying a ``predictive_posterior`` variable on a
        ``time`` dimension of length ``n_time`` (and optionally
        ``log_likelihood``); sliced on the same timeline as ``diagnostics``.
    diagnostics : SpikeEventDiagnostics
        Per-spike diagnostics whose dense ``hpd_overlap`` / ``predictive_pvalue``
        / ``kl_divergence`` matrices are ``(n_time, n_cells)``.
    spike_times : list[np.ndarray], optional
        One spike-time array per cell (length ``n_cells``).
    spike_counts : np.ndarray, shape (n_time, n_cells), optional
    place_field_peaks : np.ndarray, shape (n_cells,), optional
    place_fields : np.ndarray, shape (n_cells, n_position_bins), optional
        Supplied together with ``spike_counts`` and ``position_bins``.
    position_bins : np.ndarray, shape (n_position_bins,), optional
    track_graph : networkx.Graph, optional
        Required when ``edge_order`` is given (the linearized-track overlay).
    edge_order : sequence of (node, node), optional
    edge_spacing : float or list of float, default 0.0
    """

    time: NDArray[np.float64] | pd.Index
    position: NDArray[np.float64]
    results: xr.Dataset
    diagnostics: SpikeEventDiagnostics
    spike_times: list[NDArray[np.float64]] | None = None
    spike_counts: NDArray[np.int64] | None = None
    place_field_peaks: NDArray[np.float64] | None = None
    place_fields: NDArray[np.float64] | None = None
    position_bins: NDArray[np.float64] | None = None
    track_graph: nx.Graph | None = None
    edge_order: Sequence[tuple[Hashable, Hashable]] | None = None
    edge_spacing: float | list[float] = 0.0

    def __post_init__(self) -> None:
        time = np.asarray(self.time)
        position = np.asarray(self.position)
        if time.ndim != 1:
            raise ValueError(f"time must be 1-D; got {time.shape}")
        if position.shape != time.shape:
            raise ValueError(
                f"position must match the time shape {time.shape}; got {position.shape}"
            )

        dense_shapes = {
            name: None if (value := getattr(self.diagnostics, name)) is None else value.shape
            for name in ("hpd_overlap", "predictive_pvalue", "kl_divergence")
        }
        if any(shape is None for shape in dense_shapes.values()):
            raise ValueError(
                "diagnostics must include the dense hpd_overlap, predictive_pvalue, "
                "and kl_divergence matrices"
            )
        if len(set(dense_shapes.values())) != 1:
            raise ValueError(f"diagnostic matrices must share one shape; got {dense_shapes}")
        diagnostic_shape = next(iter(dense_shapes.values()))
        if (
            diagnostic_shape is None
            or len(diagnostic_shape) != 2
            or diagnostic_shape[0] != time.size
        ):
            raise ValueError(
                "diagnostic matrices must have one row per time sample; "
                f"got {diagnostic_shape} for {time.size} samples"
            )
        n_cells = diagnostic_shape[1]

        # The posterior/likelihood heatmap rows are sliced by the same detail
        # window as the diagnostic scatter rows, so the posterior must live on
        # the same timeline. Validate the variable, its dimensions, and the exact
        # time coordinate -- a matching length alone would still accept a shifted
        # timeline or a posterior indexed by an unrelated dimension, silently
        # misaligning (or hiding) the heatmap.
        if "predictive_posterior" not in self.results:
            raise ValueError("results must contain a 'predictive_posterior' variable")
        posterior = self.results["predictive_posterior"]
        if "time" not in posterior.dims or "state_bins" not in posterior.dims:
            raise ValueError(
                "results.predictive_posterior must have 'time' and 'state_bins' "
                f"dimensions; got {tuple(posterior.dims)}"
            )
        if "time" not in posterior.coords:
            raise ValueError("results.predictive_posterior must carry a 'time' coordinate")
        posterior_time = np.asarray(posterior.coords["time"].values, dtype=np.float64)
        panel_time = np.asarray(self.time, dtype=np.float64)
        if posterior_time.shape != panel_time.shape or not np.array_equal(
            posterior_time, panel_time
        ):
            raise ValueError(
                "results.predictive_posterior 'time' coordinate must equal the panel "
                "'time' array; the posterior heatmap would otherwise be misaligned with "
                "the per-spike diagnostics."
            )

        if self.spike_counts is not None and self.spike_counts.shape != (time.size, n_cells):
            raise ValueError(
                f"spike_counts must have shape ({time.size}, {n_cells}); "
                f"got {self.spike_counts.shape}"
            )
        if self.spike_times is not None and len(self.spike_times) != n_cells:
            raise ValueError(
                f"spike_times must contain {n_cells} cells; got {len(self.spike_times)}"
            )
        if self.place_field_peaks is not None and self.place_field_peaks.shape != (n_cells,):
            raise ValueError(
                f"place_field_peaks must have shape ({n_cells},); "
                f"got {self.place_field_peaks.shape}"
            )

        likelihood_inputs = (self.spike_counts, self.place_fields, self.position_bins)
        if self.place_fields is not None or self.position_bins is not None:
            if any(value is None for value in likelihood_inputs):
                raise ValueError(
                    "spike_counts, place_fields, and position_bins must be provided together"
                )
            assert self.place_fields is not None
            assert self.position_bins is not None
            if self.place_fields.shape != (n_cells, self.position_bins.size):
                raise ValueError(
                    "place_fields must have shape "
                    f"({n_cells}, {self.position_bins.size}); got {self.place_fields.shape}"
                )

        # The linearized-track overlay needs the graph; edge_order without it
        # would be silently ignored.
        if self.edge_order is not None and self.track_graph is None:
            raise ValueError("edge_order requires track_graph; got track_graph=None")


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


def plot_spike_event_diagnostic_scatter(
    time: NDArray[np.float64] | pd.Index,
    diagnostics: SpikeEventDiagnostics,
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
    """Plot a diagnostic value for each spike event over time.

    Each point is one spike event (one cell firing at one time bin); points
    are scattered to show the distribution of the diagnostic across the spike
    events at each time.

    For predictive_pvalue, values are transformed to -log(p) (natural log) scale to match Figure 3
    visualization where higher values indicate worse fit.

    Parameters
    ----------
    time : np.ndarray or pd.Index
        Time values (bin centers/starts).
    diagnostics : SpikeEventDiagnostics
        Per-spike-event diagnostics dataclass. The dense ``hpd_overlap``,
        ``kl_divergence``, and ``predictive_pvalue`` attributes (each shape
        (n_time, n_cells)) supply the scattered values.
    time_slice_ind : slice, optional
        Time slice indices to plot. If None, plots all time points.
    threshold : float, optional
        Threshold to draw as horizontal line. For predictive_pvalue, this should be
        the raw threshold value (e.g., 0.05) which will be transformed.
    ax : plt.Axes, optional
        Axes to plot on. If None, uses current axes.
    metric_name : str, default "hpd_overlap"
        Attribute of ``diagnostics`` to plot.
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
    >>> from statespacecheck_paper.figure04_diagnostics import (
    ...     compute_spike_event_diagnostics,
    ... )
    >>> n_time, n_bins, n_cells = 100, 50, 10
    >>> predictive = np.random.dirichlet(np.ones(n_bins), size=n_time)
    >>> place_fields = np.random.rand(n_cells, n_bins) * 10
    >>> spike_counts = np.random.poisson(0.5, (n_time, n_cells))
    >>> diagnostics = compute_spike_event_diagnostics(
    ...     predictive, spike_counts, place_fields
    ... )
    >>> ax = plot_spike_event_diagnostic_scatter(np.arange(n_time), diagnostics)
    """
    if ax is None:
        ax = plt.gca()

    metric = np.asarray(getattr(diagnostics, metric_name)).copy()
    time_arr = np.asarray(time)

    if time_slice_ind is not None:
        time_arr = time_arr[time_slice_ind]
        metric = metric[time_slice_ind]

    event_times = diagnostics.event_time
    # No default: the three ``event_*`` fields are required on
    # SpikeEventDiagnostics; an unexpected metric_name must fail loud.
    event_metric_values = getattr(diagnostics, f"event_{metric_name}")

    # Store raw metric for running average computation (before transformation)
    # The running average should be computed on raw values per manuscript formula:
    # D = sum(metric_k * I(t_k in window)) / sum(I(t_k in window))
    raw_metric = metric.copy()
    raw_event_metric_values = (
        None if event_metric_values is None else np.asarray(event_metric_values).copy()
    )

    # Transform predictive_pvalue to -log(p) (natural log) scale (matching Figure 3)
    # Higher values indicate worse fit (low probability)
    if metric_name == "predictive_pvalue":
        metric = negative_log_pvalue(metric)
        if threshold is not None:
            threshold = negative_log_pvalue(threshold)

    n_time, n_cells = metric.shape

    if event_times is not None and raw_event_metric_values is not None:
        event_times_arr = np.asarray(event_times)
        time_min, time_max = time_arr.min(), time_arr.max()
        event_mask = (event_times_arr >= time_min) & (event_times_arr < time_max)

        x_positions_arr = event_times_arr[event_mask]
        y_values_arr = raw_event_metric_values[event_mask]
        if metric_name == "predictive_pvalue":
            y_values_arr = negative_log_pvalue(y_values_arr)
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
    elif diagnostics.event_time_ind.size:
        # No wall-clock event_time and no spike_times, but per-event arrays are
        # present: map each event to its time bin via ``event_time_ind`` so
        # repeated events in the same (time bin, cell) stay distinct points.
        # Flattening the dense (n_time, n_cells) matrix (below) would collapse
        # them into one observation and under-weight the running average.
        event_bin_times = np.asarray(time)[diagnostics.event_time_ind]
        in_window = (event_bin_times >= time_arr.min()) & (event_bin_times < time_arr.max())
        x_positions_arr = event_bin_times[in_window]
        y_values_arr = np.asarray(event_metric_values)[in_window]
        if metric_name == "predictive_pvalue":
            y_values_arr = negative_log_pvalue(y_values_arr)
        valid = ~np.isnan(y_values_arr)
        x_positions_arr = x_positions_arr[valid]
        y_values_arr = y_values_arr[valid]
    else:
        # Legacy dense-only data (no per-event arrays): use the dense
        # (n_time, n_cells) matrix, one point per (time bin, cell).
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
        elif diagnostics.event_time_ind.size and raw_event_metric_values is not None:
            # Match the scatter's event_time_ind path so every event is weighted,
            # rather than the collapsing dense fallback.
            event_bin_times = np.asarray(time)[diagnostics.event_time_ind]
            time_min, time_max = time_arr.min(), time_arr.max()
            event_mask = (event_bin_times >= time_min) & (event_bin_times < time_max)
            running_avg, _ = compute_running_average(
                raw_metric,
                time_arr,
                window_size=running_average_window,
                event_times=event_bin_times[event_mask],
                event_values=raw_event_metric_values[event_mask],
            )
        else:
            running_avg, _ = compute_running_average(
                raw_metric, time_arr, window_size=running_average_window
            )

        # Transform running average if needed (same as scatter points)
        if metric_name == "predictive_pvalue":
            running_avg = negative_log_pvalue(running_avg)

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
        threshold_label = ax.text(
            1.01,
            threshold,
            "Threshold",
            transform=ax.get_yaxis_transform(),
            fontsize=8,
            va="center",
            ha="left",
            color=COLORS["threshold"],
        )
        threshold_label.set_gid(THRESHOLD_LABEL_GID)

    # HPD overlap: symlog y-scale (matching Figure 3) so the worst-fit
    # floor near 0 is expanded instead of compressed onto the bottom
    # spine.
    if metric_name == "hpd_overlap":
        ax.set_yscale("symlog", linthresh=0.01, linscale=1.0)
        ax.set_yticks([0.0, 0.1, 1.0])
        ax.set_yticklabels(["0", "0.1", "1"])
        ax.set_ylim(-0.005, 1.0)

    ax.set_xlim(time_arr.min(), time_arr.max())
    ax.set_ylabel(metric_name if ylabel is None else ylabel, fontsize=8, labelpad=7)

    if show_xlabel:
        ax.set_xlabel("Time (s)", fontsize=8, labelpad=7)
        ax.tick_params(labelsize=8)
    else:
        ax.tick_params(labelsize=8, labelbottom=False)
    if metric_name == "hpd_overlap":
        ax.tick_params(axis="y", labelsize=8, pad=1)

    return ax


def _draw_predictive_heatmap_row(
    ax: Axes,
    results: xr.Dataset,
    time: NDArray[np.float64] | pd.Index,
    position: NDArray[np.float64],
    time_slice_ind: slice,
    *,
    title: str,
    ylabel: str,
) -> None:
    """Draw one predictive-posterior heatmap row (heatmap + standard labels).

    Shared by the comparison (per column) and single-model composites; the
    per-composite legend / "Animal Position" annotation is added by the caller.
    """
    plot_distribution_heatmap(
        ax=ax,
        distribution_da=results.predictive_posterior,
        time=time,
        position=position,
        time_slice_ind=time_slice_ind,
        show_position=True,
        cmap=CMAP_POSTERIOR,
    )
    ax.set_title(title, fontsize=8)
    ax.set_ylabel(ylabel, fontsize=8, labelpad=7)
    ax.set_xlabel("")
    ax.tick_params(labelsize=8, labelbottom=False)


def _draw_decoder_likelihood_image(
    ax: Axes,
    results: xr.Dataset,
    time_slice_ind: slice,
    has_spikes_mask: NDArray[np.bool_] | None,
) -> None:
    """Overlay the decoder likelihood (marginalized over state) at spike times.

    The decoder likelihood lives on the joint state-by-position space; a
    single-state model has no position axis and
    :func:`decoder_likelihood_to_columns` raises. No-op when the results carry
    no ``log_likelihood``.
    """
    if "log_likelihood" not in results:
        return
    lik_np, time_coords, pos_coords = decoder_likelihood_to_columns(results, time_slice_ind)
    extent = compute_half_pixel_extent(time_coords, pos_coords)
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


def _draw_place_field_likelihood_image(
    ax: Axes,
    time: NDArray[np.float64] | pd.Index,
    spike_counts: NDArray[np.int64],
    place_fields: NDArray[np.float64],
    position_bins: NDArray[np.float64],
    time_slice_ind: slice,
) -> None:
    """Overlay the mean normalized per-spike likelihood over position.

    This is the observation quantity the per-spike diagnostics operate on, and
    it is identical across decoders that share place fields (unlike the decoder's
    combined likelihood). Slices to the plotted window first: the full session is
    ~700k bins, so computing over all of it would allocate multi-GB intermediates
    for a ~1000-bin plot.
    """
    counts_win = spike_counts[time_slice_ind]
    lik_np, has_spk_slice = mean_per_spike_likelihood_by_time(counts_win, place_fields)
    time_win = np.asarray(time)[time_slice_ind]
    pos = np.asarray(position_bins, dtype=np.float64)
    extent = compute_half_pixel_extent(time_win, pos)
    plot_likelihood_columns(
        ax,
        lik_np,
        has_spk_slice,
        n_time=len(time_win),
        extent=extent,
        cmap=CMAP_LIKELIHOOD,
    )


def _draw_track_graph_edges(
    axes_pair: list[Axes],
    track_graph: nx.Graph,
    edge_order: Sequence[tuple[Hashable, Hashable]] | None,
    edge_spacing: float | list[float],
    time: NDArray[np.float64] | pd.Index,
    time_slice_ind: slice,
) -> None:
    """Draw the 1D linearized track graph on the right edge of two rows."""
    time_arr = np.asarray(time)
    x_pos = float(time_arr[time_slice_ind][-1])
    for ax in axes_pair:
        plot_track_graph_1d(
            track_graph,
            ax=ax,
            edge_order=edge_order,
            edge_spacing=edge_spacing,
            other_axis_start=x_pos,
            edge_linewidth=3,
            reward_well_size=20,
            reward_well_nodes=list(range(6)),
        )


def plot_single_model_diagnostics(
    data: ModelDiagnosticPanelData,
    *,
    time_slice_ind: slice | None = None,
    model_name: str = "Continuous",
    thresholds: Mapping[str, float] | None = None,
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
    - Row 4: Predictive p-value scatter (-log(p), natural log scale)
    - Row 5: KL divergence scatter

    Parameters
    ----------
    data : ModelDiagnosticPanelData
        Named model outputs, diagnostics, observations, and track geometry.
        When its ``spike_counts``, ``place_fields``, and ``position_bins`` are
        present, the likelihood row shows the mean normalized per-spike
        likelihood used by the diagnostic calculation.
    time_slice_ind : slice, optional
        Time slice to plot. If None, plots all time points.
    model_name : str, default "Continuous"
        Model name for title.
    thresholds : dict[str, float], optional
        Thresholds for horizontal lines on diagnostic plots.
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
    time = data.time
    position = data.position
    results = data.results
    diagnostics = data.diagnostics
    spike_times = data.spike_times
    spike_counts = data.spike_counts
    place_field_peaks = data.place_field_peaks
    place_fields = data.place_fields
    position_bins = data.position_bins

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
    _draw_predictive_heatmap_row(
        axes[0],
        results,
        time,
        position,
        time_slice_ind,
        title=model_name,
        ylabel="Predictive",
    )
    # Self-label the position trace in its own color instead of a legend.
    animal_position_label = axes[0].text(
        0.02,
        0.90,
        "Animal Position",
        transform=axes[0].transAxes,
        fontsize=8,
        fontweight="normal",
        color=COLORS["ground_truth"],
        alpha=0.85,
        va="top",
        ha="left",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.45, "pad": 0.15},
    )
    animal_position_label.set_gid(ANIMAL_POSITION_LABEL_GID)

    # Row 1: Likelihood overlay at spike times
    ax_lik = axes[1]
    ax_lik.set_facecolor("black")

    if place_fields is not None and spike_counts is not None and position_bins is not None:
        # Match the simulation figure: the mean normalized per-spike likelihood
        # over position (the observation quantity the per-spike diagnostics
        # operate on), identical across decoders that share place fields.
        _draw_place_field_likelihood_image(
            ax_lik, time, spike_counts, place_fields, position_bins, time_slice_ind
        )
    elif "log_likelihood" in results:
        # Decoder-likelihood fallback (no place fields supplied): marginalized
        # over state on the joint state-by-position space.
        _draw_decoder_likelihood_image(ax_lik, results, time_slice_ind, has_spikes_mask)

    # Position overlay
    time_arr = np.asarray(time)
    ax_lik.scatter(
        time_arr[time_slice_ind],
        position[time_slice_ind],
        c=COLORS["ground_truth"],
        s=1,
        alpha=0.85,
    )
    ax_lik.set_ylabel("Likelihood", fontsize=8, labelpad=7)
    ax_lik.set_xlabel("")
    ax_lik.tick_params(labelsize=8, labelbottom=False)

    # 1D track graph on right edge of predictive and likelihood rows
    if data.track_graph is not None:
        _draw_track_graph_edges(
            [axes[0], axes[1]],
            data.track_graph,
            data.edge_order,
            data.edge_spacing,
            time,
            time_slice_ind,
        )

    # Row 2: Spike raster
    if spike_times is not None:
        sort_order = np.argsort(place_field_peaks) if place_field_peaks is not None else None
        sliced_time = time_arr[time_slice_ind]
        time_slice = slice(float(sliced_time[0]), float(sliced_time[-1]))
        plot_raster(spike_times, time_slice, ax=axes[2], sort_order=sort_order)
        axes[2].set_ylabel("Neuron", fontsize=8, labelpad=7)
        axes[2].set_xlabel("")
        axes[2].tick_params(labelsize=8, labelbottom=False)

    # Rows 3-5: Diagnostic scatters
    for i, spec in enumerate(METRIC_SPECS):
        row = i + 3
        threshold = thresholds.get(spec.name) if thresholds else None
        plot_spike_event_diagnostic_scatter(
            time,
            diagnostics,
            time_slice_ind=time_slice_ind,
            threshold=threshold,
            ax=axes[row],
            metric_name=spec.name,
            color=spec.color,
            ylabel=spec.ylabel,
            show_xlabel=(i == 2),
            spike_times=spike_times,
            show_running_average=show_running_average,
            running_average_window=running_average_window,
        )
        if spec.name == "hpd_overlap":
            worse_fit_y = 0.28
        elif spec.name == "predictive_pvalue":
            worse_fit_y = 0.68
        else:
            worse_fit_y = 0.5
        worse_fit_label = axes[row].text(
            1.01,
            worse_fit_y,
            spec.worse_fit_direction,
            transform=axes[row].transAxes,
            fontsize=8,
            va="center",
            ha="left",
        )
        worse_fit_label.set_gid(WORSE_FIT_LABEL_GID)

    return fig, axes


def plot_per_spike_metric_hexbin_row(
    diagnostics_a: SpikeEventDiagnostics,
    diagnostics_b: SpikeEventDiagnostics,
    axes: Sequence[Axes],
    *,
    model_a_name: str = "Continuous",
    model_b_name: str = "Cont-Frag",
    thresholds: dict[str, float] | None = None,
    colorbar_pad: float = 0.02,
) -> None:
    """Plot a 1x3 row of hexbin densities comparing per-spike diagnostics between two decoders.

    Each panel shows one diagnostic on the x-axis (model A) and the same
    diagnostic on the y-axis (model B). Each hexagon's colour encodes
    log-scaled spike-event count (matplotlib ``bins='log'``). Points on
    the identity line indicate decoder agreement on that spike.

    Both diagnostics dicts must carry the same set of per-spike events
    in the same order (i.e. ``event_*`` arrays produced from the same
    spike trains by
    :func:`statespacecheck_paper.figure04_diagnostics.compute_model_diagnostics`).
    Raises ``ValueError`` if shapes differ for any of the three metrics.

    Parameters
    ----------
    diagnostics_a, diagnostics_b : SpikeEventDiagnostics
        Per-spike-event diagnostics whose ``event_hpd_overlap``,
        ``event_kl_divergence``, ``event_predictive_pvalue`` attributes (each
        shape ``(n_spikes,)``) supply the hexbin values.
    axes : Sequence[matplotlib.axes.Axes]
        Three axes, one per metric (HPD overlap, ``-log(p)`` natural
        log, KL divergence).
    model_a_name, model_b_name : str
        Axis labels for each decoder.
    thresholds : dict[str, float], optional
        Per-metric flag thresholds keyed by ``hpd_overlap``, ``kl_divergence``,
        ``predictive_pvalue`` (raw values; the ``predictive_pvalue`` cutoff is transformed to
        the ``-log(p)`` axis). When given, each panel draws dotted threshold
        lines on both axes and lightly shades the quadrant of spikes flagged by
        model A but not model B. Metrics absent from the dict get no lines.
    colorbar_pad : float, default 0.02
        Fractional padding between the rightmost panel and shared count
        colorbar.
    """
    if len(axes) != 3:
        raise ValueError(f"axes must have length 3, got {len(axes)}")

    # (event key, title, color, log_transform, threshold key, worse-fit direction).
    # "direction" is relative to the plotted axis: HPD overlap flags low values
    # ("below"); KL divergence and the (log-transformed) predictive p-value flag
    # high values ("above").
    metric_specs = [
        ("event_hpd_overlap", "HPD overlap", COLORS["hpd_overlap"], False, "hpd_overlap", "below"),
        (
            "event_predictive_pvalue",
            r"$-\log(p)$",
            COLORS["metric_combined"],
            True,
            "predictive_pvalue",
            "above",
        ),
        (
            "event_kl_divergence",
            "KL divergence",
            COLORS["kl_divergence"],
            False,
            "kl_divergence",
            "above",
        ),
    ]

    hex_artists = []
    for panel_idx, (ax, (key, title, color, log_transform, thr_key, direction)) in enumerate(
        zip(axes, metric_specs, strict=True)
    ):
        data_a = np.asarray(getattr(diagnostics_a, key), dtype=np.float64)
        data_b = np.asarray(getattr(diagnostics_b, key), dtype=np.float64)
        if data_a.shape != data_b.shape:
            raise ValueError(
                f"diagnostics_a[{key!r}] and diagnostics_b[{key!r}] must "
                f"carry the same set of spike events in the same order; "
                f"got shapes {data_a.shape} vs {data_b.shape}."
            )
        if log_transform:
            data_a = negative_log_pvalue(data_a)
            data_b = negative_log_pvalue(data_b)

        valid = np.isfinite(data_a) & np.isfinite(data_b)
        data_a = data_a[valid]
        data_b = data_b[valid]

        cmap = mcolors.LinearSegmentedColormap.from_list("custom", ["white", color])
        hb = ax.hexbin(
            data_a,
            data_b,
            gridsize=40,
            cmap=cmap,
            mincnt=1,
            rasterized=True,
        )
        hex_artists.append(hb)

        # Identity line — span the actual data range so the visual
        # agreement reference doesn't depend on matplotlib's autoscale.
        combined = np.concatenate([data_a, data_b])
        lims = (float(np.nanmin(combined)), float(np.nanmax(combined)))
        # Pad limits so hexbins centred on the data extrema (e.g. at 0)
        # render fully instead of being clipped at the axis spine.
        margin = (lims[1] - lims[0]) * 0.05
        padded_lims = (lims[0] - margin, lims[1] + margin)
        ax.plot(padded_lims, padded_lims, color=COLORS["threshold"], lw=0.8, ls="--", alpha=0.7)

        # Per-metric flag threshold: dotted lines on both axes (same scalar on
        # x=model A and y=model B), plus light shading of the "rescue" quadrant —
        # spikes flagged by model A (Continuous) but not model B (Cont-Frag).
        thr_raw = thresholds.get(thr_key) if thresholds else None
        if thr_raw is not None:
            thr = negative_log_pvalue(thr_raw) if log_transform else float(thr_raw)
            lo, hi = padded_lims
            if direction == "below":
                # Flagged below threshold: A flagged (x < thr), B not (y > thr).
                rect_xy, rect_w, rect_h = (lo, thr), thr - lo, hi - thr
            else:
                # Flagged above threshold: A flagged (x > thr), B not (y < thr).
                rect_xy, rect_w, rect_h = (thr, lo), hi - thr, thr - lo
            # Dotted threshold cross spanning the panel (reads the cutoff on each axis).
            ax.axvline(thr, color=COLORS["threshold"], lw=0.8, ls=":", alpha=0.7, zorder=2)
            ax.axhline(thr, color=COLORS["threshold"], lw=0.8, ls=":", alpha=0.7, zorder=2)
            # Accent-outlined callout box framing the "rescue" quadrant: spikes
            # flagged by model A but not model B. A solid coloured border reads as
            # "look here", unlike a muted gray fill.
            rescue_accent = "#D55E00"  # Wong vermillion, distinct from the metric colours
            ax.add_patch(
                Rectangle(
                    rect_xy,
                    rect_w,
                    rect_h,
                    facecolor=mcolors.to_rgba(rescue_accent, 0.07),
                    edgecolor=rescue_accent,
                    linewidth=1.4,
                    zorder=4,
                )
            )
            callout_model_a_name = "Cont." if model_a_name == "Continuous" else model_a_name
            label = f"flagged by\n{callout_model_a_name} only"
            label_bbox = {
                "boxstyle": "round,pad=0.15",
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.7,
            }
            if direction == "below":
                # Rescue box is a tall left strip (HPD overlap) full of points;
                # place the label above the panel so it does not cover the data.
                # Sit it above the centered panel title so the two do not collide.
                ax.text(
                    -0.02,
                    1.16,
                    label,
                    transform=ax.transAxes,
                    ha="left",
                    va="bottom",
                    fontsize=8,
                    color=rescue_accent,
                    fontstyle="italic",
                    zorder=5,
                    clip_on=False,
                )
            else:
                # Rescue box is in the lower-right (KL, -log(p)); place the label
                # at the bottom edge of the box, in the sparse far corner.
                ax.text(
                    rect_xy[0] + rect_w / 2.0,
                    rect_xy[1],
                    label,
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    color=rescue_accent,
                    fontstyle="italic",
                    zorder=5,
                    bbox=label_bbox,
                )

        ax.set_xlim(padded_lims)
        ax.set_ylim(padded_lims)
        ax.set_aspect("equal", adjustable="box")

        ax.set_xlabel(model_a_name, fontsize=8, labelpad=4)
        ax.set_ylabel(model_b_name if panel_idx == 0 else "", fontsize=8, labelpad=4)
        ax.set_title(title, fontsize=8)
        ax.tick_params(labelsize=8)

        if key == "event_kl_divergence":
            ax.text(
                0.02,
                0.98,
                f"n={len(data_a):,}",
                transform=ax.transAxes,
                fontsize=8,
                va="top",
                ha="left",
                color="0.4",
            )

    if hex_artists:
        # get_array() is typed Optional; filter None (never occurs for a drawn
        # hexbin) so the max is over concrete arrays without a type: ignore.
        counts = [
            float(np.nanmax(arr)) for hb in hex_artists if (arr := hb.get_array()) is not None
        ]
        max_count = max(counts) if counts else 1.0
        if max_count <= 1:
            max_count = 10
        shared_norm = mcolors.LogNorm(vmin=1, vmax=max_count)
        for hb in hex_artists:
            hb.set_norm(shared_norm)

        count_mappable = matplotlib.cm.ScalarMappable(norm=shared_norm, cmap="Greys")
        count_mappable.set_array([])
        cbar = axes[-1].figure.colorbar(
            count_mappable,
            ax=list(axes),
            fraction=0.025,
            pad=colorbar_pad,
            shrink=0.64,
        )
        cbar.set_label("Spike events per hex", fontsize=8, labelpad=4)
        count_ticks = [tick for tick in (1, 10, 100, 1000, 10000, 100000) if tick <= max_count]
        cbar.set_ticks(count_ticks)
        cbar.set_ticklabels([f"{tick:,}" for tick in count_ticks])
        cbar.ax.tick_params(labelsize=8, width=0.5, length=2)
