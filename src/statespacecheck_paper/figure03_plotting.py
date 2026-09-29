"""Figure-3 rendering: the time-series diagnostic panels and summary heatmap.

This module composes Figure 3 from a simulation's decoded diagnostics: the
per-metric time-series rows (predictive, likelihood, raster, HPD overlap,
predictive p-value, KL divergence) with phase-boundary overlays, and the
panel-(b) per-condition flag-percentage heatmap. ``compose_figure03`` is the
public entry point. Generic renderers (``plot_likelihood_columns``) stay in
:mod:`plotting`.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.image import AxesImage
from numpy.typing import NDArray

from statespacecheck_paper.diagnostics import DecodingDiagnostics, DiagnosticThresholds
from statespacecheck_paper.figure03_protocol import (
    STEP_SECONDS,
    Figure3Config,
    PhaseBoundary,
    compute_replay_step_window,
)
from statespacecheck_paper.figure03_summary import (
    SUMMARY_ERROR_METRICS,
    SUMMARY_FLAG_METRICS,
    build_summary_conditions,
)
from statespacecheck_paper.number_format import significant, whole_percent
from statespacecheck_paper.plotting import (
    plot_event_metric_row,
    plot_likelihood_columns,
)
from statespacecheck_paper.style import (
    CMAP_LIKELIHOOD,
    CMAP_PREDICTIVE,
    COLORS,
    FIGURE_DPI,
    METRIC_SPEC_BY_NAME,
    METRIC_SPECS,
    PREDICTIVE_VMAX_QUANTILE,
    MetricSpec,
)

FIGURE03_PANEL_LABEL_GID = "figure03-panel-label"


FIGURE03_PHASE_LABEL_GID = "figure03-phase-label"


FIGURE03_ROW_LABEL_GID = "figure03-row-label"


FIGURE03_TRUE_POSITION_LABEL_GID = "figure03-true-position-label"


FIGURE03_SUMMARY_CELL_LABEL_GID = "figure03-summary-cell-label"


FIGURE03_SUMMARY_COMPONENT_LABEL_GID = "figure03-summary-model-component-label"


FIGURE03_SUMMARY_KNOWN_COMPONENT_LABEL_GID = "figure03-summary-known-model-component-label"


FIGURE03_SUMMARY_TITLE_GID = "figure03-summary-title"
# Per-condition decoding-error row (median absolute error) rendered as text
# beneath the flag heatmap, plus its row header.
FIGURE03_SUMMARY_ERROR_CELL_LABEL_GID = "figure03-summary-error-cell-label"
FIGURE03_SUMMARY_ERROR_HEADER_GID = "figure03-summary-error-header"


# Shaded Figure 3 misfit bands: (start, end) boundary indices and the ``COLORS``
# key. Saturated colors keep the bands visible at low alpha.
FIGURE03_MISFIT_BANDS: tuple[tuple[PhaseBoundary, PhaseBoundary, str], ...] = (
    (PhaseBoundary.REMAP_START, PhaseBoundary.REMAP_END, "likelihood"),
    (PhaseBoundary.RECOVERY1_END, PhaseBoundary.HIST_DEP_END, "reference"),
    (PhaseBoundary.RECOVERY2_END, PhaseBoundary.DRIFT_END, "predictive"),
    (PhaseBoundary.RECOVERY3_END, PhaseBoundary.SPARSE_POP_END, "predictive_pvalue"),
)


def add_phase_boundaries(axes: list[Axes], config: Figure3Config) -> None:
    """Shade the misfit phases and the replay control on each time-series axis.

    Parameters
    ----------
    axes : list[Axes]
        Axes sharing the Figure 3 step timeline.
    config : Figure3Config
        Supplies ``phase_boundaries`` and the replay step window.
    """
    bnd = config.phase_boundaries
    bands = [(bnd[start], bnd[end], COLORS[color]) for start, end, color in FIGURE03_MISFIT_BANDS]
    # The replay band (in clean-recovery 2) is a control; shade it in a
    # distinct color so the reader can see the decoded-vs-true divergence is
    # a deliberate, non-flagged event.
    replay_start, replay_end = compute_replay_step_window(config)
    bands.append((replay_start, replay_end, COLORS["replay"]))
    for ax in axes:
        for start, end, color in bands:
            ax.axvspan(start, end, alpha=0.15, color=color, label="")


def _plot_timeseries_heatmap(
    ax: Axes,
    data: NDArray[np.floating],
    true_position: NDArray[np.floating],
) -> AxesImage:
    """Plot time x position heatmap with the true position overlaid.

    The color scale runs from 0 to the ``PREDICTIVE_VMAX_QUANTILE`` quantile
    of ``data`` (for robustness to outliers), in ``CMAP_PREDICTIVE``.

    Parameters
    ----------
    ax : Axes
        Matplotlib axes to plot on.
    data : NDArray, shape (n_time, n_bins)
        Distribution data (predictive, likelihood, or posterior).
    true_position : NDArray, shape (n_time,)
        True position to overlay as a line.

    Returns
    -------
    im : AxesImage
        The image object (for colorbar creation if needed).

    Examples
    --------
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> fig, ax = plt.subplots()
    >>> data = np.random.dirichlet(np.ones(50), size=100)
    >>> true_position = np.random.uniform(0, 49, 100)
    >>> im = _plot_timeseries_heatmap(ax, data, true_position)
    >>> plt.close(fig)
    """
    n_time = data.shape[0]
    # Use nanquantile to handle NaN values (e.g., masked likelihood)
    im = ax.imshow(
        data.T,
        aspect="auto",
        origin="lower",
        vmin=0.0,
        vmax=np.nanquantile(data, PREDICTIVE_VMAX_QUANTILE),
        cmap=CMAP_PREDICTIVE,
    )
    ax.plot(
        np.arange(n_time),
        true_position,
        color=COLORS["ground_truth"],
        linewidth=1.0,
        alpha=0.5,
    )
    return im


def _plot_likelihood_overlay(
    ax: Axes,
    predictive: NDArray[np.floating],
    event_likelihood: NDArray[np.floating],
    event_time_ind: NDArray[np.intp],
    true_position: NDArray[np.floating],
) -> None:
    """Plot per-spike likelihood distributions at spike times.

    Aggregates per-spike likelihoods into per-timestep distributions and renders
    each spike time as a ``CMAP_LIKELIHOOD`` column with guaranteed minimum
    width.

    Parameters
    ----------
    ax : Axes
        Matplotlib axes to plot on.
    predictive : NDArray, shape (n_time, n_bins)
        Predictive distribution over position at each time (used for shape only).
    event_likelihood : NDArray, shape (n_spikes, n_bins)
        Normalized likelihood distribution for each individual spike event.
    event_time_ind : NDArray, shape (n_spikes,)
        Time index for each spike event.
    true_position : NDArray, shape (n_time,)
        True position to overlay.
    """
    n_time, n_bins = predictive.shape

    # Black background so likelihood columns stand out
    ax.set_facecolor("black")

    # Aggregate per-spike likelihoods into per-timestep arrays.
    # When multiple cells spike at the same time, average their likelihoods.
    if len(event_time_ind) > 0:
        lik_per_time: NDArray[np.floating] = np.zeros((n_time, n_bins))
        counts = np.zeros(n_time)
        np.add.at(lik_per_time, event_time_ind, event_likelihood)
        np.add.at(counts, event_time_ind, 1.0)
        has_spikes = counts > 0
        lik_per_time[has_spikes] /= counts[has_spikes, np.newaxis]

        plot_likelihood_columns(ax, lik_per_time, has_spikes, n_time, cmap=CMAP_LIKELIHOOD)

    ax.plot(
        np.arange(n_time),
        true_position,
        color=COLORS["ground_truth"],
        linewidth=1.0,
        alpha=0.5,
    )

    ax.set_xlim(0, n_time - 1)
    ax.set_ylim(0, n_bins - 1)


def _plot_spike_count_raster(
    ax: Axes,
    spike_counts: NDArray[np.floating],
    place_field_centers: NDArray[np.floating],
) -> None:
    """Plot spike counts as a raster, sorted by place field peak.

    Neurons are sorted by their place field center position so that
    sequential activation during movement is visible as diagonal patterns.
    Uses scatter plot for better visibility of sparse events.

    Parameters
    ----------
    ax : Axes
        Matplotlib axes to plot on.
    spike_counts : NDArray, shape (n_time, n_cells)
        Spike counts at each timestep for each cell.
    place_field_centers : NDArray, shape (n_cells,)
        Place field centers for sorting neurons by preferred position.

    Examples
    --------
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> fig, ax = plt.subplots()
    >>> spike_counts = np.random.poisson(0.1, (100, 20))
    >>> pf_centers = np.linspace(0, 1, 20)
    >>> _plot_spike_count_raster(ax, spike_counts, pf_centers)
    >>> plt.close(fig)
    """
    # Sort neurons by place field peak position
    sort_order = np.argsort(place_field_centers)
    spikes_sorted = spike_counts[:, sort_order]

    # Find spike locations (time, neuron pairs where spikes occurred)
    spike_times, spike_neurons = np.where(spikes_sorted > 0)

    # Use scatter plot for better visibility of sparse events
    ax.scatter(
        spike_times,
        spike_neurons,
        s=1.0,
        c="black",
        marker="|",
        linewidths=0.8,
        rasterized=True,
    )

    # Set axis limits
    n_time, n_cells = spikes_sorted.shape
    ax.set_xlim(0, n_time)
    ax.set_ylim(-0.5, n_cells - 0.5)
    ax.set_ylabel("Neuron", labelpad=7)


def _add_figure03_row_label(ax: Axes, label: str) -> None:
    """Add the right-side row label used in Figure 3 panel (a)."""
    row_label = ax.text(
        1.01,
        0.5,
        label,
        transform=ax.transAxes,
        va="center",
        ha="left",
        rotation=270,
    )
    row_label.set_gid(FIGURE03_ROW_LABEL_GID)


def _add_figure03_panel_label(ax: Axes, label: str, *, y: float) -> None:
    """Add a panel letter with a stable semantic artist id."""
    panel_label = ax.text(
        -0.05,
        y,
        label,
        fontweight="bold",
        transform=ax.transAxes,
        va="top",
        ha="right",
    )
    panel_label.set_gid(FIGURE03_PANEL_LABEL_GID)


def _plot_figure03_predictive_row(
    ax: Axes,
    predictive: NDArray[np.floating],
    true_position: NDArray[np.floating],
) -> None:
    """Plot Figure 3's predictive row with a direct physical-position label."""
    _plot_timeseries_heatmap(ax, predictive, true_position)
    ax.set_ylabel("Position (a.u.)", labelpad=7)
    ax.tick_params(labelbottom=False)
    true_position_label = ax.text(
        0.02,
        0.90,
        "Physical position",
        transform=ax.transAxes,
        color=COLORS["ground_truth"],
        va="top",
        ha="left",
    )
    true_position_label.set_gid(FIGURE03_TRUE_POSITION_LABEL_GID)
    _add_figure03_row_label(ax, "Predictive")


def _plot_figure03_likelihood_row(
    ax: Axes,
    diagnostics: DecodingDiagnostics,
    true_position: NDArray[np.floating],
) -> None:
    """Plot Figure 3's per-spike likelihood row."""
    _plot_likelihood_overlay(
        ax,
        diagnostics.predictive,
        diagnostics.event_likelihood,
        diagnostics.event_time_ind,
        true_position=true_position,
    )
    ax.set_ylabel("Position (a.u.)", labelpad=7)
    ax.tick_params(labelbottom=False)
    _add_figure03_row_label(ax, "Likelihood")


def _plot_figure03_raster_row(
    ax: Axes,
    spike_counts: NDArray[np.floating],
    place_field_centers: NDArray[np.floating],
) -> None:
    """Plot Figure 3's spike-count raster row."""
    _plot_spike_count_raster(ax, spike_counts, place_field_centers)
    ax.tick_params(labelbottom=False)
    _add_figure03_row_label(ax, "Spikes")


def _plot_figure03_diagnostic_row(
    ax: Axes,
    time_ind: NDArray[np.integer],
    values: NDArray[np.floating],
    threshold: float,
    spec: MetricSpec,
    *,
    n_time: int,
    show_xlabel: bool,
) -> None:
    """Plot one Figure 3 diagnostic event row."""
    plot_event_metric_row(
        ax,
        time_ind,
        values,
        spec,
        threshold=threshold,
        xlim=(0, n_time),
        ylabel=spec.short_label,
        symlog_yticks=(0.0, 0.01, 0.1, 1.0),
        # Headroom above 1 keeps the "1" tick label (and the dense band of
        # fully nested HPD regions at overlap = 1) clear of the raster panel
        # that abuts this axis from above.
        symlog_ylim=(-0.005, 1.6),
    )
    if show_xlabel:
        # The axis counts steps; labeling it in ms holds only for 1 ms steps.
        if STEP_SECONDS != 1e-3:
            raise ValueError(f"Figure 3 labels its step axis in ms; STEP_SECONDS is {STEP_SECONDS}")
        ax.set_xlabel("Time (ms)", labelpad=7)
    else:
        ax.tick_params(labelbottom=False)


def _add_figure03_phase_labels(ax: Axes, config: Figure3Config) -> None:
    """Add staggered misfit labels above Figure 3 panel (a)."""
    bnd = config.phase_boundaries
    t_remap_start = bnd[PhaseBoundary.REMAP_START]
    t_remap_end = bnd[PhaseBoundary.REMAP_END]
    t_recovery1_end = bnd[PhaseBoundary.RECOVERY1_END]
    t_hist_dep_end = bnd[PhaseBoundary.HIST_DEP_END]
    t_recovery2_end = bnd[PhaseBoundary.RECOVERY2_END]
    t_drift_end = bnd[PhaseBoundary.DRIFT_END]
    t_recovery3_end = bnd[PhaseBoundary.RECOVERY3_END]
    t_sparse_pop_end = bnd[PhaseBoundary.SPARSE_POP_END]

    r0, r1 = compute_replay_step_window(config)
    phase_label_y = 1.04
    phase_labels_info: list[tuple[float, str]] = [
        ((t_remap_start + t_remap_end) / 2, "Remap"),
        ((r0 + r1) / 2, "Replay"),
        ((t_recovery1_end + t_hist_dep_end) / 2, "History-dep."),
        ((t_recovery2_end + t_drift_end) / 2, "Drift"),
        ((t_recovery3_end + t_sparse_pop_end) / 2, "Sparse population"),
    ]
    for x_pos, label_text in phase_labels_info:
        phase_label = ax.text(
            x_pos,
            phase_label_y,
            label_text,
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="bottom",
            style="italic",
        )
        phase_label.set_gid(FIGURE03_PHASE_LABEL_GID)


def _plot_figure03_summary_heatmap(
    ax: Axes,
    config: Figure3Config,
    median_flag_percentages: NDArray[np.floating],
    median_decoding_error: NDArray[np.floating],
) -> None:
    """Plot the across-realization phase-by-metric flag percentages.

    A plain-text row beneath the heatmap reports the per-condition decoding
    error (median absolute error in position units) so each flag
    percentage can be read against ground truth. It is not colour-mapped
    because its units differ from the flag percentages.
    """
    conditions = build_summary_conditions(config)
    component_labels = [col.model_component for col in conditions]

    frac_data = np.asarray(median_flag_percentages, dtype=float)
    expected_shape = (len(SUMMARY_FLAG_METRICS), len(conditions))
    if frac_data.shape != expected_shape:
        raise ValueError(
            f"median_flag_percentages must have shape {expected_shape}; got {frac_data.shape}"
        )
    if not np.all(np.isfinite(frac_data)) or np.any((frac_data < 0.0) | (frac_data > 100.0)):
        raise ValueError("median_flag_percentages must contain finite percentages in [0, 100]")

    decoding_error = np.asarray(median_decoding_error, dtype=float)
    expected_error_shape = (len(SUMMARY_ERROR_METRICS), len(conditions))
    if decoding_error.shape != expected_error_shape:
        raise ValueError(
            f"median_decoding_error must have shape {expected_error_shape}; "
            f"got {decoding_error.shape}"
        )
    if not np.all(np.isfinite(decoding_error)) or np.any(decoding_error < 0.0):
        raise ValueError("median_decoding_error must be finite and non-negative")

    max_frac = np.nanmax(frac_data)
    norm_frac = frac_data / max_frac if max_frac > 0 else frac_data
    ax.imshow(norm_frac, cmap="YlOrRd", aspect="auto", vmin=0, vmax=1)

    # Rows follow SUMMARY_FLAG_METRICS; each label stacks its words one per line.
    n_metrics = len(SUMMARY_FLAG_METRICS)
    ax.set_yticks(range(n_metrics))
    ax.set_yticklabels(
        [
            METRIC_SPEC_BY_NAME[metric].short_label.replace(" ", "\n")
            for metric, _direction in SUMMARY_FLAG_METRICS
        ]
    )

    ax.set_xticks(range(len(conditions)))
    ax.set_xticklabels([col.label for col in conditions])
    ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)

    for row_idx in range(n_metrics):
        for col_idx in range(len(conditions)):
            val = frac_data[row_idx, col_idx]
            color = "white" if norm_frac[row_idx, col_idx] > 0.55 else "black"
            weight = "bold" if norm_frac[row_idx, col_idx] > 0.7 else "normal"
            cell_label = ax.text(
                col_idx,
                row_idx,
                f"{whole_percent(val)}%",
                ha="center",
                va="center",
                color=color,
                fontweight=weight,
            )
            cell_label.set_gid(FIGURE03_SUMMARY_CELL_LABEL_GID)

    # The decoding-error row sits directly beneath the heatmap; the known
    # component row follows it.
    # Both rows round with the same functions the manuscript prose uses, so a
    # value cannot read differently in the panel and in the text beside it.
    error_headers = ("Median |error|\n(a.u.):",)
    for row_offset, header in enumerate(error_headers):
        row_y = float(n_metrics + row_offset)
        for col_idx in range(len(conditions)):
            error_label = ax.text(
                col_idx,
                row_y,
                significant(decoding_error[row_offset, col_idx]),
                ha="center",
                va="center",
                color="black",
            )
            error_label.set_gid(FIGURE03_SUMMARY_ERROR_CELL_LABEL_GID)
        error_header = ax.text(
            -0.04,
            row_y,
            header,
            transform=ax.get_yaxis_transform(),
            ha="right",
            va="center",
            color="0.4",
            fontstyle="italic",
        )
        error_header.set_gid(FIGURE03_SUMMARY_ERROR_HEADER_GID)

    component_row_y = float(n_metrics + len(error_headers))
    # The observation model is the likelihood's component; the transition
    # model is the predictive's.
    component_color = {"Observation": COLORS["likelihood"], "Transition": COLORS["predictive"]}
    for col_idx, comp in enumerate(component_labels):
        color = component_color.get(comp, "0.4")
        component_label = ax.text(
            col_idx,
            component_row_y,
            comp,
            ha="center",
            va="center",
            fontstyle="italic",
            color=color,
        )
        component_label.set_gid(FIGURE03_SUMMARY_COMPONENT_LABEL_GID)
    known_component_label = ax.text(
        -0.04,
        component_row_y,
        "Known\ncomponent:",
        transform=ax.get_yaxis_transform(),
        ha="right",
        va="center",
        color="0.4",
        fontstyle="italic",
    )
    known_component_label.set_gid(FIGURE03_SUMMARY_KNOWN_COMPONENT_LABEL_GID)

    title = ax.set_title(
        "% of spike events flagged as poor fit (median across realizations)",
        pad=8,
        loc="center",
    )
    title.set_gid(FIGURE03_SUMMARY_TITLE_GID)


def compose_figure03(
    true_position: NDArray[np.floating],
    spike_counts: NDArray[np.floating],
    diagnostics: DecodingDiagnostics,
    diagnostic_thresholds: DiagnosticThresholds,
    config: Figure3Config,
    place_field_centers: NDArray[np.floating],
    median_flag_percentages: NDArray[np.floating],
    median_decoding_error: NDArray[np.floating],
) -> Figure:
    """Create comprehensive time-series diagnostics figure.

    Layout: 6 time-series panels (predictive, likelihood, raster, HPD overlap,
    predictive p-value, KL) with shared x-axis and phase boundary overlays.

    Parameters
    ----------
    true_position : NDArray, shape (n_time,)
        True position at each time point.
    spike_counts : NDArray, shape (n_time, n_cells)
        Spike counts for each cell at each time point.
    diagnostics : DecodingDiagnostics
        Diagnostics from ``decode_with_diagnostics``. The panels read its
        dense ``(n_time, n_cells)`` metric matrices (``hpd_overlap``,
        ``kl_divergence``, ``predictive_pvalue``) and its per-event
        ``(n_events,)`` arrays (``event_time_ind``, ``event_cell_ind``, and the
        matching ``event_*`` metric values).
    diagnostic_thresholds : DiagnosticThresholds
        Threshold values for each diagnostic.
    config : Figure3Config
        Decoding parameters containing timeline structure.
    place_field_centers : NDArray, shape (n_cells,)
        Place field centers for each cell (used for spike raster sorting).
    median_flag_percentages : NDArray, shape (3, n_columns)
        Pre-computed across-realization median percent flagged for the
        panel-(b) heatmap (rows follow
        :data:`statespacecheck_paper.figure03_summary.SUMMARY_FLAG_METRICS`,
        columns follow :func:`statespacecheck_paper.figure03_summary.build_summary_conditions`).
        This stabilized summary is deliberately required: substituting the
        displayed realization would change the meaning of panel (b).
    median_decoding_error : NDArray, shape (1, n_columns)
        Median per-condition decoding error (rows follow
        :data:`statespacecheck_paper.figure03_summary.SUMMARY_ERROR_METRICS`:
        median absolute error in position units), rendered as a text row
        beneath the panel-(b) heatmap.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Time-series diagnostic figure.

    Examples
    --------
    >>> from statespacecheck_paper.figure03_protocol import Figure3Config
    >>> from statespacecheck_paper.decoding import decode_with_diagnostics
    >>> from statespacecheck_paper.diagnostics import DecodingDiagnostics, DiagnosticThresholds
    >>> # See tests/test_plotting.py for a worked DecodingDiagnostics fixture
    >>> # and how to plumb it into compose_figure03.
    """
    fig_width = 6.85  # Full page width; tight PDF stays within ~183 mm.
    # Extra height (and bottom margin) holds the error row and the
    # known-component row beneath the panel-(b) heatmap; the plotted area
    # above them is unchanged.
    fig_height = 7.4
    fig = plt.figure(figsize=(fig_width, fig_height), dpi=FIGURE_DPI)

    # Outer grid: time-series block on top, summary heatmap on bottom.
    gs_outer = gridspec.GridSpec(
        2,
        1,
        figure=fig,
        height_ratios=[5.3, 1.2],
        hspace=0.34,
        left=0.08,
        right=0.93,
        top=0.97,
        bottom=0.11,
    )

    gs = gs_outer[0].subgridspec(
        6,
        1,
        height_ratios=[1.2, 1.2, 0.8, 0.7, 0.7, 0.7],
        hspace=0.08,
    )

    gs_summary = gs_outer[1]

    n_time = diagnostics.posterior.shape[0]
    ax_pred = fig.add_subplot(gs[0])
    ax_like = fig.add_subplot(gs[1], sharex=ax_pred)
    ax_raster = fig.add_subplot(gs[2], sharex=ax_pred)
    diagnostic_axes = [fig.add_subplot(gs[i], sharex=ax_pred) for i in range(3, 6)]

    _plot_figure03_predictive_row(ax_pred, diagnostics.predictive, true_position)
    _plot_figure03_likelihood_row(ax_like, diagnostics, true_position)
    _plot_figure03_raster_row(ax_raster, spike_counts, place_field_centers)

    event_time_ind = diagnostics.event_time_ind
    for row_idx, (ax, spec) in enumerate(zip(diagnostic_axes, METRIC_SPECS, strict=True)):
        _plot_figure03_diagnostic_row(
            ax,
            event_time_ind,
            getattr(diagnostics, spec.event_attr),
            getattr(diagnostic_thresholds, spec.name),
            spec,
            n_time=n_time,
            show_xlabel=row_idx == len(METRIC_SPECS) - 1,
        )

    time_series_axes = [ax_pred, ax_like, ax_raster, *diagnostic_axes]
    add_phase_boundaries(time_series_axes, config)
    _add_figure03_phase_labels(ax_pred, config)
    _add_figure03_panel_label(ax_pred, "a", y=1.15)

    # ===== SUMMARY HEATMAP: % exceeding baseline threshold per phase =====
    ax_summary = fig.add_subplot(gs_summary)
    _add_figure03_panel_label(ax_summary, "b", y=1.25)
    _plot_figure03_summary_heatmap(
        ax_summary,
        config,
        median_flag_percentages,
        median_decoding_error,
    )

    return fig
