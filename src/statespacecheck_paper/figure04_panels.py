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
    plot_distribution_heatmap,
)
from statespacecheck_paper.figure04_track_plots import plot_track_graph_1d
from statespacecheck_paper.plotting import negative_log_pvalue, plot_likelihood_columns
from statespacecheck_paper.style import (
    CMAP_LIKELIHOOD,
    CMAP_POSTERIOR,
    COLORS,
    METRIC_SPEC_BY_NAME,
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
    spike_times : list[np.ndarray]
        One spike-time array per cell (length ``n_cells``).
    spike_counts : np.ndarray, shape (n_time, n_cells)
    place_field_peaks : np.ndarray, shape (n_cells,)
    place_fields : np.ndarray, shape (n_cells, n_position_bins)
    position_bins : np.ndarray, shape (n_position_bins,)
    track_graph : networkx.Graph
    edge_order : sequence of (node, node)
        Explicit order used for the scientific linearization.
    edge_spacing : float or list of float, default 0.0
    """

    time: NDArray[np.float64] | pd.Index
    position: NDArray[np.float64]
    results: xr.Dataset
    diagnostics: SpikeEventDiagnostics
    spike_times: list[NDArray[np.float64]]
    spike_counts: NDArray[np.int64]
    place_field_peaks: NDArray[np.float64]
    place_fields: NDArray[np.float64]
    position_bins: NDArray[np.float64]
    track_graph: nx.Graph
    edge_order: Sequence[tuple[Hashable, Hashable]]
    edge_spacing: float | list[float] = 0.0

    def __post_init__(self) -> None:
        required_fields = (
            "spike_times",
            "spike_counts",
            "place_field_peaks",
            "place_fields",
            "position_bins",
            "track_graph",
            "edge_order",
        )
        missing = [name for name in required_fields if getattr(self, name) is None]
        if missing:
            raise ValueError(f"ModelDiagnosticPanelData required fields are missing: {missing}")
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

        if self.spike_counts.shape != (time.size, n_cells):
            raise ValueError(
                f"spike_counts must have shape ({time.size}, {n_cells}); "
                f"got {self.spike_counts.shape}"
            )
        if np.any(self.spike_counts < 0):
            raise ValueError("spike_counts must be non-negative")
        if len(self.spike_times) != n_cells:
            raise ValueError(
                f"spike_times must contain {n_cells} cells; got {len(self.spike_times)}"
            )
        if self.place_field_peaks.shape != (n_cells,):
            raise ValueError(
                f"place_field_peaks must have shape ({n_cells},); "
                f"got {self.place_field_peaks.shape}"
            )
        if not np.all(np.isfinite(self.place_field_peaks)):
            raise ValueError("place_field_peaks must contain only finite values")

        if self.position_bins.ndim != 1 or self.position_bins.size < 2:
            raise ValueError("position_bins must be a 1-D array with at least two bins")
        if self.place_fields.shape != (n_cells, self.position_bins.size):
            raise ValueError(
                "place_fields must have shape "
                f"({n_cells}, {self.position_bins.size}); got {self.place_fields.shape}"
            )
        if not np.all(np.isfinite(self.place_fields)) or np.any(self.place_fields < 0.0):
            raise ValueError("place_fields must contain only finite nonnegative rates")
        if not np.all(np.isfinite(self.position_bins)) or np.any(np.diff(self.position_bins) <= 0):
            raise ValueError("position_bins must be finite and strictly increasing")
        if len(self.edge_order) == 0:
            raise ValueError("edge_order must explicitly contain the linearized track edges")
        missing_edges = [edge for edge in self.edge_order if not self.track_graph.has_edge(*edge)]
        if missing_edges:
            raise ValueError(f"edge_order contains edges absent from track_graph: {missing_edges}")


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
        Per-spike-event diagnostics dataclass. The required ``event_*`` arrays
        supply one value per plotted spike.
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

    full_time = np.asarray(time, dtype=np.float64)
    if full_time.ndim != 1:
        raise ValueError(f"time must be 1-D; got shape {full_time.shape}")
    selected_indices = np.arange(full_time.size)[
        slice(None) if time_slice_ind is None else time_slice_ind
    ]
    if selected_indices.size == 0:
        raise ValueError("time_slice_ind selects no time samples")
    time_arr = full_time[selected_indices]

    event_time_ind = np.asarray(diagnostics.event_time_ind, dtype=np.intp)
    if np.any(event_time_ind >= full_time.size):
        raise ValueError("diagnostics.event_time_ind contains an index outside the time array")
    event_metric_values = np.asarray(getattr(diagnostics, f"event_{metric_name}"), dtype=np.float64)
    if not np.all(np.isfinite(event_metric_values)):
        raise ValueError(f"event_{metric_name} contains a non-finite value that cannot be plotted")

    selected_time_mask = np.zeros(full_time.size, dtype=bool)
    selected_time_mask[selected_indices] = True
    event_mask = selected_time_mask[event_time_ind]
    all_event_times = (
        np.asarray(diagnostics.event_time, dtype=np.float64)
        if diagnostics.event_time is not None
        else full_time[event_time_ind]
    )
    x_positions_arr = all_event_times[event_mask]
    raw_y_values = event_metric_values[event_mask]
    spec = METRIC_SPEC_BY_NAME.get(metric_name)
    use_neg_log = spec is not None and spec.display_transform == "neg_log_p"
    y_values_arr = negative_log_pvalue(raw_y_values) if use_neg_log else raw_y_values
    if threshold is not None and use_neg_log:
        threshold = float(negative_log_pvalue(threshold))

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
        # Compute on raw per-event values exactly as specified in the
        # manuscript, then apply the display transform to the average.
        running_avg, _ = compute_running_average(
            x_positions_arr,
            raw_y_values,
            time_arr,
            window_size=running_average_window,
        )

        # Transform running average if needed (same as scatter points)
        if use_neg_log:
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
            va="center",
            ha="left",
            color=COLORS["threshold"],
        )
        threshold_label.set_gid(THRESHOLD_LABEL_GID)

    # HPD overlap: symlog y-scale (matching Figure 3) so the worst-fit
    # floor near 0 is expanded instead of compressed onto the bottom
    # spine.
    if spec is not None and spec.symlog_axis:
        ax.set_yscale("symlog", linthresh=0.01, linscale=1.0)
        ax.set_yticks([0.0, 0.1, 1.0])
        ax.set_yticklabels(["0", "0.1", "1"])
        ax.set_ylim(-0.005, 1.0)

    ax.set_xlim(time_arr.min(), time_arr.max())
    ax.set_ylabel(metric_name if ylabel is None else ylabel, labelpad=7)

    if show_xlabel:
        ax.set_xlabel("Time (s)", labelpad=7)
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
    ax.set_ylabel(ylabel, labelpad=7)
    ax.set_xlabel("")
    ax.tick_params(labelsize=8, labelbottom=False)


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
    edge_order: Sequence[tuple[Hashable, Hashable]],
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
        The likelihood row shows the mean normalized per-spike likelihood used
        by the diagnostic calculation.
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

    # Row 0: Predictive posterior
    _draw_predictive_heatmap_row(
        axes[0],
        results,
        time,
        position,
        time_slice_ind,
        title=model_name,
        ylabel="Predictive\nposition (cm)",
    )
    # Self-label the position trace in its own color instead of a legend.
    animal_position_label = axes[0].text(
        0.02,
        0.90,
        "Animal Position",
        transform=axes[0].transAxes,
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

    # Match the simulation figure: plot the observation likelihood used by the
    # diagnostics, never the decoder's different combined likelihood.
    _draw_place_field_likelihood_image(
        ax_lik, time, spike_counts, place_fields, position_bins, time_slice_ind
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
    ax_lik.set_ylabel("Likelihood\nposition (cm)", labelpad=7)
    ax_lik.set_xlabel("")
    ax_lik.tick_params(labelsize=8, labelbottom=False)

    # 1D track graph on right edge of predictive and likelihood rows
    _draw_track_graph_edges(
        [axes[0], axes[1]],
        data.track_graph,
        data.edge_order,
        data.edge_spacing,
        time,
        time_slice_ind,
    )

    # Row 2: Spike raster
    sort_order = np.argsort(place_field_peaks)
    sliced_time = time_arr[time_slice_ind]
    time_slice = slice(float(sliced_time[0]), float(sliced_time[-1]))
    plot_raster(spike_times, time_slice, ax=axes[2], sort_order=sort_order)
    axes[2].set_ylabel("Neuron", labelpad=7)
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
    if diagnostics_a.event_time_ind.shape != diagnostics_b.event_time_ind.shape:
        raise ValueError(
            "diagnostics_a and diagnostics_b must carry the same set of spike events "
            "in the same order"
        )
    if not np.array_equal(diagnostics_a.event_time_ind, diagnostics_b.event_time_ind) or not (
        np.array_equal(diagnostics_a.event_cell_ind, diagnostics_b.event_cell_ind)
    ):
        raise ValueError(
            "diagnostics_a and diagnostics_b must carry identical spike events in the same order"
        )

    # Event attr, colour, display transform, threshold key, and plotted worse-fit
    # direction all come from the shared MetricSpec. Only the panel title differs
    # from MetricSpec.ylabel here ("KL divergence" vs the scatter's "KL div."),
    # so it stays a local per-panel override. ``plotted_worse`` is relative to the
    # plotted axis: HPD overlap flags low values ("below"); KL divergence and the
    # (log-transformed) predictive p-value flag high values ("above").
    hexbin_titles = ("HPD overlap", r"$-\log(p)$", "KL divergence")

    hex_artists = []
    for panel_idx, (ax, spec, title) in enumerate(
        zip(axes, METRIC_SPECS, hexbin_titles, strict=True)
    ):
        key = spec.event_attr
        color = spec.color
        log_transform = spec.display_transform == "neg_log_p"
        thr_key = spec.name
        direction = spec.plotted_worse
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
        if data_a.size == 0:
            raise ValueError(f"Cannot plot {key}: no spike events are present")
        if not np.all(np.isfinite(data_a)) or not np.all(np.isfinite(data_b)):
            raise ValueError(
                f"Cannot plot {key}: every aligned spike event must have a finite value"
            )

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
        lims = (float(np.min(combined)), float(np.max(combined)))
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
                    color=rescue_accent,
                    fontstyle="italic",
                    zorder=5,
                    bbox=label_bbox,
                )

        ax.set_xlim(padded_lims)
        ax.set_ylim(padded_lims)
        ax.set_aspect("equal", adjustable="box")

        ax.set_xlabel(model_a_name, labelpad=4)
        ax.set_ylabel(model_b_name if panel_idx == 0 else "", labelpad=4)
        ax.set_title(title, fontsize=8)
        ax.tick_params(labelsize=8)

        if key == "event_kl_divergence":
            ax.text(
                0.02,
                0.98,
                f"n={len(data_a):,}",
                transform=ax.transAxes,
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
        cbar.set_label("Spike events per hex", labelpad=4)
        count_ticks = [tick for tick in (1, 10, 100, 1000, 10000, 100000) if tick <= max_count]
        cbar.set_ticks(count_ticks)
        cbar.set_ticklabels([f"{tick:,}" for tick in count_ticks])
        cbar.ax.tick_params(labelsize=8, width=0.5, length=2)
