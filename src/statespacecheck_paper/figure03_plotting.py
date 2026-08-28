"""Figure-3 rendering: the time-series diagnostic panels and summary heatmap.

This module composes Figure 3 from a simulation's decoded diagnostics: the
per-metric time-series rows (predictive, likelihood, raster, HPD overlap,
predictive p-value, KL divergence) with phase-boundary overlays, and the
panel-(b) per-condition flag-percentage heatmap. ``compose_figure03`` is the
public entry point. Generic renderers (``plot_likelihood_columns``,
``compute_hpd_region``) stay in :mod:`plotting`.
"""

from __future__ import annotations

from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.image import AxesImage
from numpy.typing import NDArray

from statespacecheck_paper.diagnostics import DecodingDiagnostics, DiagnosticThresholds
from statespacecheck_paper.figure03_protocol import (
    Figure3Config,
    PhaseBoundary,
    compute_replay_step_window,
)
from statespacecheck_paper.figure03_summary import SUMMARY_FLAG_METRICS, build_summary_conditions
from statespacecheck_paper.plotting import negative_log_pvalue, plot_likelihood_columns
from statespacecheck_paper.style import CMAP_LIKELIHOOD, CMAP_POSTERIOR, COLORS

FIGURE3_PANEL_LABEL_GID = "figure3-panel-label"


FIGURE3_PHASE_LABEL_GID = "figure3-phase-label"


FIGURE3_ROW_LABEL_GID = "figure3-row-label"


FIGURE3_THRESHOLD_LABEL_GID = "figure3-threshold-label"


FIGURE3_THRESHOLD_LINE_GID = "figure3-threshold-line"


FIGURE3_TRUE_POSITION_LABEL_GID = "figure3-true-position-label"


FIGURE3_WORSE_FIT_LABEL_GID = "figure3-worse-fit-label"


FIGURE3_SUMMARY_CELL_LABEL_GID = "figure3-summary-cell-label"


FIGURE3_SUMMARY_COMPONENT_LABEL_GID = "figure3-summary-model_component-label"


FIGURE3_SUMMARY_KNOWN_COMPONENT_LABEL_GID = "figure3-summary-known-model_component-label"


FIGURE3_SUMMARY_TITLE_GID = "figure3-summary-title"


@dataclass(frozen=True)
class DiagnosticRowSpec:
    """Display and transform metadata for one Figure 3 diagnostic row."""

    event_attr: str
    threshold_attr: str
    ylabel: str
    color: str
    worse_fit_direction: str
    log_transform: bool = False
    symlog_hpd: bool = False


FIGURE3_DIAGNOSTIC_ROW_SPECS: tuple[DiagnosticRowSpec, ...] = (
    DiagnosticRowSpec(
        "event_hpd_overlap",
        "hpd_overlap",
        "HPD overlap",
        COLORS["hpd_overlap"],
        "↓ Worse fit",
        symlog_hpd=True,
    ),
    DiagnosticRowSpec(
        "event_predictive_pvalue",
        "predictive_pvalue",
        "−log(p)",
        COLORS["metric_combined"],
        "↑ Worse fit",
        log_transform=True,
    ),
    DiagnosticRowSpec(
        "event_kl_divergence",
        "kl_divergence",
        "KL div.",
        COLORS["kl_divergence"],
        "↑ Worse fit",
    ),
)


def add_phase_boundaries(
    axes: list[Axes],
    phase_boundaries: tuple[int, ...],
    include_labels: bool = False,
    alpha: float = 0.15,
    replay: tuple[int, int] | None = None,
) -> None:
    """Add colored phase boundaries to multiple axes.

    Parameters
    ----------
    axes : list[plt.Axes]
        List of axes to add phase boundaries to.
    phase_boundaries : tuple[int, ...]
        Phase boundary time points, in the canonical 8-element order
        indexed by :class:`statespacecheck_paper.figure03_protocol.PhaseBoundary`
        (``REMAP_START``, ``REMAP_END``, ``RECOVERY1_END``,
        ``HIST_DEP_END``, ``RECOVERY2_END``, ``DRIFT_END``,
        ``RECOVERY3_END``, ``SPARSE_POP_END``). Shorter tuples (down
        to 2 elements) are accepted and produce a partial shading; only
        the misfit conditions whose pair of boundary entries is present
        are drawn.
    include_labels : bool, default False
        If True, add labels for legend on first axis.
    alpha : float, default 0.15
        Alpha (transparency) for phase boundaries.
    replay : tuple[int, int] or None, default None
        Half-open ``[start, end)`` step bounds of the replay control band
        (shaded in a distinct, non-misfit color) when provided.

    Returns
    -------
    None
        Modifies axes in-place by adding colored phase boundary regions.

    Examples
    --------
    >>> fig, axes = plt.subplots(4, 1)
    >>> boundaries = (10, 20, 30, 40, 50, 60, 70, 80)
    >>> add_phase_boundaries(axes, boundaries, include_labels=True)
    """
    # Saturated colors so axvspan at low alpha is still visible. We use
    # ``COLORS`` entries that are vivid hexes (the pastel ``phase_*``
    # palette entries wash out completely at this alpha).
    misfit_specs: list[tuple[int, int, str, str]] = []
    n = len(phase_boundaries)
    if n >= 2:
        misfit_specs.append(
            (phase_boundaries[0], phase_boundaries[1], COLORS["likelihood"], "Remap")
        )
    if n >= 4:
        misfit_specs.append(
            (
                phase_boundaries[2],
                phase_boundaries[3],
                COLORS["reference"],
                "History-dependent firing",
            )
        )
    if n >= 6:
        misfit_specs.append(
            (phase_boundaries[4], phase_boundaries[5], COLORS["predictive"], "Drift")
        )
    if n >= 8:
        misfit_specs.append(
            (
                phase_boundaries[6],
                phase_boundaries[7],
                COLORS["metric_combined"],
                "Sparse population",
            )
        )
    # The replay band (in clean-recovery 2) is not a misfit; shade it in a
    # distinct color so the reader can see the decoded-vs-true divergence is
    # a deliberate, non-flagged event.
    if replay is not None:
        misfit_specs.append((replay[0], replay[1], COLORS["phase_replay"], "Replay"))
    phases = misfit_specs

    for ax_idx, ax in enumerate(axes):
        add_labels_to_axis = include_labels and ax_idx == 0
        for start, end, color, label in phases:
            ax.axvspan(
                start,
                end,
                alpha=alpha if not add_labels_to_axis else alpha + 0.05,
                color=color,
                label=label if add_labels_to_axis else "",
            )


def _plot_timeseries_heatmap(
    ax: Axes,
    data: NDArray[np.floating],
    true_position: NDArray[np.floating] | None = None,
    cmap: str = CMAP_POSTERIOR,
    vmax_quantile: float = 0.975,
) -> AxesImage:
    """Plot time x position heatmap with optional true position overlay.

    Parameters
    ----------
    ax : Axes
        Matplotlib axes to plot on.
    data : NDArray, shape (n_time, n_bins)
        Distribution data (predictive, likelihood, or posterior).
    true_position : NDArray, shape (n_time,), optional
        True position to overlay as a line.
    cmap : str, default CMAP_POSTERIOR
        Colormap for heatmap.
    vmax_quantile : float, default 0.975
        Quantile for vmax (for robustness to outliers).

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
        vmax=np.nanquantile(data, vmax_quantile),
        cmap=cmap,
    )
    if true_position is not None:
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
    per_spike_likelihood: NDArray[np.floating],
    spike_time_ind: NDArray[np.intp],
    true_position: NDArray[np.floating] | None = None,
    cmap_overlay: str = CMAP_LIKELIHOOD,
) -> AxesImage:
    """Plot per-spike likelihood distributions at spike times.

    Aggregates per-spike likelihoods into per-timestep distributions and renders
    each spike time as a colored column with guaranteed minimum width.

    Parameters
    ----------
    ax : Axes
        Matplotlib axes to plot on.
    predictive : NDArray, shape (n_time, n_bins)
        Predictive distribution over position at each time (used for shape only).
    per_spike_likelihood : NDArray, shape (n_spikes, n_bins)
        Normalized likelihood distribution for each individual spike event.
    spike_time_ind : NDArray, shape (n_spikes,)
        Time index for each spike event.
    true_position : NDArray, shape (n_time,), optional
        True position to overlay.
    cmap_overlay : str, default CMAP_LIKELIHOOD
        Colormap for the likelihood columns.

    Returns
    -------
    im : AxesImage
        A placeholder image object.
    """
    n_time, n_bins = predictive.shape

    # Black background so likelihood columns stand out
    ax.set_facecolor("black")

    # Aggregate per-spike likelihoods into per-timestep arrays.
    # When multiple cells spike at the same time, average their likelihoods.
    if len(spike_time_ind) > 0:
        lik_per_time: NDArray[np.floating] = np.zeros((n_time, n_bins))
        counts = np.zeros(n_time)
        np.add.at(lik_per_time, spike_time_ind, per_spike_likelihood)
        np.add.at(counts, spike_time_ind, 1.0)
        has_spikes = counts > 0
        lik_per_time[has_spikes] /= counts[has_spikes, np.newaxis]

        plot_likelihood_columns(ax, lik_per_time, has_spikes, n_time, cmap=cmap_overlay)

    if true_position is not None:
        ax.plot(
            np.arange(n_time),
            true_position,
            color=COLORS["ground_truth"],
            linewidth=1.0,
            alpha=0.5,
        )

    ax.set_xlim(0, n_time - 1)
    ax.set_ylim(0, n_bins - 1)

    # Return a dummy AxesImage for API compatibility
    im = ax.imshow(
        np.zeros((1, 1)),
        aspect="auto",
        origin="lower",
        extent=(0, n_time - 1, 0, n_bins - 1),
        alpha=0.0,
    )
    return im


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

    # Find spike locations (time, neuron pairs where spike_counts occurred)
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


def _add_figure3_row_label(ax: Axes, label: str) -> None:
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
    row_label.set_gid(FIGURE3_ROW_LABEL_GID)


def _add_figure3_panel_label(ax: Axes, label: str, *, y: float) -> None:
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
    panel_label.set_gid(FIGURE3_PANEL_LABEL_GID)


def _add_figure3_threshold_label(ax: Axes, threshold: float) -> None:
    """Label a diagnostic threshold line at the right edge of an axis."""
    threshold_label = ax.text(
        1.01,
        threshold,
        "Threshold",
        transform=ax.get_yaxis_transform(),
        va="center",
        ha="left",
        color=COLORS["threshold"],
    )
    threshold_label.set_gid(FIGURE3_THRESHOLD_LABEL_GID)


def _add_figure3_worse_fit_label(ax: Axes, label: str) -> None:
    """Add the right-side direction-of-worse-fit annotation."""
    worse_fit_label = ax.text(
        1.01,
        0.5,
        label,
        transform=ax.transAxes,
        va="center",
        ha="left",
    )
    worse_fit_label.set_gid(FIGURE3_WORSE_FIT_LABEL_GID)


def _plot_figure3_predictive_row(
    ax: Axes,
    predictive: NDArray[np.floating],
    true_position: NDArray[np.floating],
) -> None:
    """Plot Figure 3's predictive row with a direct physical-position label."""
    _plot_timeseries_heatmap(ax, predictive, true_position)
    ax.set_ylabel("Position (a.u.)", labelpad=7)
    ax.tick_params(labelsize=8, labelbottom=False)
    true_position_label = ax.text(
        0.02,
        0.90,
        "Physical position",
        transform=ax.transAxes,
        color=COLORS["ground_truth"],
        va="top",
        ha="left",
    )
    true_position_label.set_gid(FIGURE3_TRUE_POSITION_LABEL_GID)
    _add_figure3_row_label(ax, "Predictive")


def _plot_figure3_likelihood_row(
    ax: Axes,
    diagnostics: DecodingDiagnostics,
    true_position: NDArray[np.floating],
) -> None:
    """Plot Figure 3's per-spike likelihood row."""
    _plot_likelihood_overlay(
        ax,
        diagnostics.predictive,
        diagnostics.per_spike_likelihood,
        diagnostics.event_time_ind,
        true_position=true_position,
    )
    ax.set_ylabel("Position (a.u.)", labelpad=7)
    ax.tick_params(labelsize=8, labelbottom=False)
    _add_figure3_row_label(ax, "Likelihood")


def _plot_figure3_raster_row(
    ax: Axes,
    spike_counts: NDArray[np.floating],
    place_field_centers: NDArray[np.floating],
) -> None:
    """Plot Figure 3's spike-count raster row."""
    _plot_spike_count_raster(ax, spike_counts, place_field_centers)
    ax.tick_params(labelsize=8, labelbottom=False)
    _add_figure3_row_label(ax, "Spikes")


def _plot_figure3_diagnostic_row(
    ax: Axes,
    time_ind: NDArray[np.integer],
    values: NDArray[np.floating],
    threshold: float,
    spec: DiagnosticRowSpec,
    *,
    n_time: int,
    show_xlabel: bool,
) -> None:
    """Plot one Figure 3 diagnostic event row."""
    plot_values = np.asarray(values, dtype=float)
    plot_threshold = float(threshold)
    if spec.log_transform:
        plot_values = negative_log_pvalue(plot_values)
        plot_threshold = float(negative_log_pvalue(plot_threshold))

    ax.scatter(
        time_ind,
        plot_values,
        s=0.8,
        alpha=0.6,
        c=spec.color,
        rasterized=True,
    )
    threshold_line = ax.axhline(
        plot_threshold,
        color=COLORS["threshold"],
        linewidth=1.2,
        alpha=0.7,
        zorder=10,
    )
    threshold_line.set_gid(FIGURE3_THRESHOLD_LINE_GID)

    if spec.symlog_hpd:
        # Symlog y-scale expands the worst-fit floor near 0 instead of
        # compressing it onto the bottom spine.
        ax.set_yscale("symlog", linthresh=0.01, linscale=1.0)
        ax.set_yticks([0.0, 0.01, 0.1, 0.5, 1.0])
        ax.set_yticklabels(["0", "0.01", "0.1", "0.5", "1"])
        ax.set_ylim(-0.005, 1.0)

    ax.set_xlim(0, n_time)
    ax.set_ylabel(spec.ylabel, labelpad=7)
    if show_xlabel:
        ax.set_xlabel("Time (ms)", labelpad=7)
        ax.tick_params(labelsize=8)
    else:
        ax.tick_params(labelsize=8, labelbottom=False)

    _add_figure3_worse_fit_label(ax, spec.worse_fit_direction)
    _add_figure3_threshold_label(ax, plot_threshold)


def _add_figure3_phase_labels(ax: Axes, config: Figure3Config) -> None:
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
        phase_label.set_gid(FIGURE3_PHASE_LABEL_GID)


def _plot_figure3_summary_heatmap(
    ax: Axes,
    config: Figure3Config,
    median_flag_percentages: NDArray[np.floating],
) -> None:
    """Plot the across-realization phase-by-metric flag percentages."""
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

    max_frac = np.nanmax(frac_data)
    norm_frac = frac_data / max_frac if max_frac > 0 else frac_data
    ax.imshow(norm_frac, cmap="YlOrRd", aspect="auto", vmin=0, vmax=1)

    metric_labels = ["HPD\noverlap", "−log(p)", "KL\ndiv."]
    ax.set_yticks(range(3))
    ax.set_yticklabels(metric_labels)

    ax.set_xticks(range(len(conditions)))
    ax.set_xticklabels([col.label for col in conditions])
    ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)

    for row_idx in range(3):
        for col_idx in range(len(conditions)):
            val = frac_data[row_idx, col_idx]
            color = "white" if norm_frac[row_idx, col_idx] > 0.55 else "black"
            weight = "bold" if norm_frac[row_idx, col_idx] > 0.7 else "normal"
            cell_label = ax.text(
                col_idx,
                row_idx,
                f"{val:.0f}%",
                ha="center",
                va="center",
                color=color,
                fontweight=weight,
            )
            cell_label.set_gid(FIGURE3_SUMMARY_CELL_LABEL_GID)

    component_row_y = 3.0
    component_color = {"Observation": "#E69F00", "Transition": "#0072B2"}
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
        component_label.set_gid(FIGURE3_SUMMARY_COMPONENT_LABEL_GID)
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
    known_component_label.set_gid(FIGURE3_SUMMARY_KNOWN_COMPONENT_LABEL_GID)

    title = ax.set_title(
        "% of spike events flagged as poor fit (median across realizations)",
        fontsize=8,
        pad=8,
        loc="center",
    )
    title.set_gid(FIGURE3_SUMMARY_TITLE_GID)


def compose_figure03(
    true_position: NDArray[np.floating],
    spike_counts: NDArray[np.floating],
    diagnostics: DecodingDiagnostics,
    diagnostic_thresholds: DiagnosticThresholds,
    config: Figure3Config,
    place_field_centers: NDArray[np.floating],
    median_flag_percentages: NDArray[np.floating],
) -> Figure:
    """Create comprehensive time-series diagnostics figure.

    Layout: 6 time-series panels (predictive, likelihood, raster, HPDO,
    predictive p-value, KL) with shared x-axis and phase boundary overlays.

    Parameters
    ----------
    true_position : NDArray, shape (n_time,)
        True position at each time point.
    spike_counts : NDArray, shape (n_time, n_cells)
        Spike counts for each cell at each time point.
    diagnostics : DecodingDiagnostics
        Diagnostic bundle from ``decode_with_diagnostics``. The panels read its
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
    fig_height = 7.0
    fig = plt.figure(figsize=(fig_width, fig_height), dpi=450)

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
        bottom=0.06,
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

    _plot_figure3_predictive_row(ax_pred, diagnostics.predictive, true_position)
    _plot_figure3_likelihood_row(ax_like, diagnostics, true_position)
    _plot_figure3_raster_row(ax_raster, spike_counts, place_field_centers)

    event_time_ind = diagnostics.event_time_ind
    for row_idx, (ax, spec) in enumerate(
        zip(diagnostic_axes, FIGURE3_DIAGNOSTIC_ROW_SPECS, strict=True)
    ):
        _plot_figure3_diagnostic_row(
            ax,
            event_time_ind,
            getattr(diagnostics, spec.event_attr),
            getattr(diagnostic_thresholds, spec.threshold_attr),
            spec,
            n_time=n_time,
            show_xlabel=row_idx == len(FIGURE3_DIAGNOSTIC_ROW_SPECS) - 1,
        )

    time_series_axes = [ax_pred, ax_like, ax_raster, *diagnostic_axes]
    add_phase_boundaries(
        time_series_axes,
        tuple(config.phase_boundaries),
        alpha=0.15,
        replay=compute_replay_step_window(config),
    )
    _add_figure3_phase_labels(ax_pred, config)
    _add_figure3_panel_label(ax_pred, "a", y=1.15)

    # ===== SUMMARY HEATMAP: % exceeding baseline threshold per phase =====
    ax_summary = fig.add_subplot(gs_summary)
    _add_figure3_panel_label(ax_summary, "b", y=1.25)
    _plot_figure3_summary_heatmap(
        ax_summary,
        config,
        median_flag_percentages,
    )

    return fig
