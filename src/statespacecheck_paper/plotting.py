"""Plotting utilities for state space model diagnostics.

This module provides functions for creating publication-ready figures showing
diagnostic metrics and misfit examples for state space models.

Examples
--------
>>> import numpy as np
>>> from statespacecheck_paper.plotting import compute_hpd_region
>>> x = np.linspace(-5, 5, 100)
>>> pdf = np.exp(-0.5 * x**2) / np.sqrt(2 * np.pi)
>>> mask = compute_hpd_region(x, pdf, coverage=0.95)
>>> mask.shape
(100,)
"""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from matplotlib import gridspec
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.image import AxesImage
from matplotlib.lines import Line2D
from numpy.typing import NDArray

from statespacecheck_paper.analysis import (
    DecodeParams,
    Thresholds,
    Transformed,
    get_remapped_pf_centers,
    likelihood_grid_for_counts,
)
from statespacecheck_paper.simulation import gaussian_transition_matrix, normalize
from statespacecheck_paper.style import CMAP_LIKELIHOOD, CMAP_POSTERIOR, COLORS


def add_phase_boundaries(
    axes: list[Axes],
    phase_boundaries: tuple[int, ...],
    include_labels: bool = False,
    alpha: float = 0.15,
) -> None:
    """Add colored phase boundaries to multiple axes.

    Parameters
    ----------
    axes : list[plt.Axes]
        List of axes to add phase boundaries to.
    phase_boundaries : tuple[int, ...]
        Phase boundary time points (must have 8 elements):
        (remap_start, remap_end, recovery1_end, flat_end,
         recovery2_end, fast_end, recovery3_end, slow_end).
    include_labels : bool, default False
        If True, add labels for legend on first axis.
    alpha : float, default 0.15
        Alpha (transparency) for phase boundaries.

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
    if len(phase_boundaries) != 8:
        return

    (
        t_remap_start,
        t_remap_end,
        t_recovery1_end,
        t_flat_end,
        t_recovery2_end,
        t_fast_end,
        t_recovery3_end,
        t_slow_end,
    ) = phase_boundaries

    # Define phases with (start, end, color, label)
    # Use semantic colors from COLORS dictionary
    phases = [
        (t_remap_start, t_remap_end, COLORS["likelihood"], "Remapping"),
        (t_recovery1_end, t_flat_end, COLORS["reference"], "Flat firing"),
        (t_recovery2_end, t_fast_end, COLORS["ground_truth"], "Fast movement"),
        (t_recovery3_end, t_slow_end, COLORS["predictive"], "Stationary"),
    ]

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


def compute_hpd_region(x: np.ndarray, pdf: np.ndarray, coverage: float = 0.95) -> np.ndarray:
    """Compute highest posterior density region for given coverage.

    Parameters
    ----------
    x : np.ndarray, shape (n_points,)
        Domain values.
    pdf : np.ndarray, shape (n_points,)
        Probability density values (must be normalized).
    coverage : float, default 0.95
        Desired coverage probability.

    Returns
    -------
    mask : np.ndarray, shape (n_points,)
        Boolean mask indicating points in HPD region.

    Examples
    --------
    >>> x = np.linspace(-5, 5, 100)
    >>> pdf = np.exp(-0.5 * x**2) / np.sqrt(2 * np.pi)
    >>> mask = compute_hpd_region(x, pdf, coverage=0.95)
    >>> mask.dtype == bool
    True
    """
    # Normalize to ensure proper probability
    dx = x[1] - x[0]
    pdf_normalized = pdf / (np.sum(pdf) * dx)

    # Sort by density and find threshold
    sorted_pdf = np.sort(pdf_normalized)[::-1]  # Descending
    cumsum = np.cumsum(sorted_pdf) * dx
    threshold_idx = int(np.searchsorted(cumsum, coverage))
    if threshold_idx >= len(sorted_pdf):
        threshold_idx = len(sorted_pdf) - 1
    threshold = sorted_pdf[threshold_idx]

    mask: np.ndarray = pdf_normalized >= threshold
    return mask


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


def create_distribution_comparison_panel(
    ax: Axes,
    x: NDArray[np.floating],
    predictive_params: tuple[float, float],
    likelihood_params: tuple[float, float],
    color_predictive: str,
    color_likelihood: str,
    title: str | None = None,
    show_labels: bool = False,
    coverage: float = 0.95,
) -> None:
    """Create a panel comparing predictive and likelihood distributions.

    Shows both distributions with filled curves and HPD regions as
    horizontal bars below the plot.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to plot on.
    x : np.ndarray, shape (n_points,)
        Position values for plotting.
    predictive_params : tuple[float, float]
        (mean, std) for predictive Gaussian distribution.
    likelihood_params : tuple[float, float]
        (mean, std) for likelihood Gaussian distribution.
    color_predictive : str
        Color for predictive distribution.
    color_likelihood : str
        Color for likelihood distribution.
    title : str | None, optional
        Panel title.
    show_labels : bool, default False
        Whether to show "Predictive"/"Likelihood" text labels on curves.
    coverage : float, default 0.95
        Coverage probability for HPD regions.

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> import numpy as np
    >>> fig, ax = plt.subplots()
    >>> x = np.linspace(-20, 20, 1000)
    >>> create_distribution_comparison_panel(
    ...     ax, x,
    ...     predictive_params=(0, 1.5),
    ...     likelihood_params=(5, 1.5),
    ...     color_predictive="blue",
    ...     color_likelihood="orange",
    ...     title="Example",
    ... )
    >>> plt.close(fig)
    """
    from matplotlib.patches import Rectangle
    from scipy import stats

    # Generate distributions
    pred_mean, pred_std = predictive_params
    like_mean, like_std = likelihood_params

    pdf_predictive: NDArray[np.floating] = stats.norm.pdf(x, loc=pred_mean, scale=pred_std)
    pdf_likelihood: NDArray[np.floating] = stats.norm.pdf(x, loc=like_mean, scale=like_std)

    # Normalize likelihood
    dx = float(x[1] - x[0])
    pdf_likelihood = pdf_likelihood / (np.sum(pdf_likelihood) * dx)

    # Plot distributions
    ax.plot(
        x,
        pdf_predictive,
        color=color_predictive,
        linewidth=1.2,
        label="Predictive distribution",
    )
    ax.fill_between(x, pdf_predictive, alpha=0.3, color=color_predictive)

    ax.plot(
        x,
        pdf_likelihood,
        color=color_likelihood,
        linewidth=1.2,
        label="Normalized likelihood",
    )
    ax.fill_between(x, pdf_likelihood, alpha=0.3, color=color_likelihood)

    # Compute HPD regions and extract contiguous intervals
    hpd_predictive = compute_hpd_region(x, pdf_predictive, coverage=coverage)
    hpd_likelihood = compute_hpd_region(x, pdf_likelihood, coverage=coverage)
    pred_regions = extract_contiguous_regions(hpd_predictive, x)
    like_regions = extract_contiguous_regions(hpd_likelihood, x)

    # Draw HPD regions as horizontal bars
    bar_height = 0.015
    y_pred = -0.08
    y_like = -0.05

    for start, end in pred_regions:
        ax.add_patch(
            Rectangle(
                (start, y_pred),
                end - start,
                bar_height,
                facecolor=color_predictive,
                edgecolor=color_predictive,
                linewidth=1.0,
                clip_on=False,
            )
        )

    for start, end in like_regions:
        ax.add_patch(
            Rectangle(
                (start, y_like),
                end - start,
                bar_height,
                facecolor=color_likelihood,
                edgecolor=color_likelihood,
                linewidth=1.0,
                clip_on=False,
            )
        )

    # Formatting
    ax.set_xlim(float(x[0]), float(x[-1]))
    ax.set_ylim(-0.1, 0.30)  # Room for sub-panel titles
    if title:
        ax.set_title(title, fontsize=7, fontweight="normal", pad=2)

    ax.axis("off")

    # Add direct labels on distribution curves
    if show_labels:
        # Label predictive on left side, likelihood on right side
        ax.text(
            -12,
            0.22,
            "Predictive",
            ha="center",
            va="bottom",
            fontsize=5,
            color=color_predictive,
        )
        ax.text(
            16,
            0.22,
            "Likelihood",
            ha="center",
            va="bottom",
            fontsize=5,
            color=color_likelihood,
        )


def plot_original(
    xs: NDArray[np.floating],
    x_true: NDArray[np.floating],
    metrics: dict[str, NDArray[np.floating]],
    thresholds: Thresholds,
    title: str = "Original Metrics",
    remap_window: tuple[int, int] | None = None,
    phase_boundaries: tuple[int, ...] | None = None,
) -> Figure:
    """Plot original diagnostic metrics with thresholds.

    Creates a 4-panel figure showing:
    1. Posterior heatmap with true position overlay
    2. HPD overlap over time (per-cell scatter)
    3. KL divergence over time (per-cell scatter)
    4. Spike probability over time (per-cell scatter)

    Parameters
    ----------
    xs : NDArray, shape (n_bins,)
        Position bin centers.
    x_true : NDArray, shape (n_time,)
        True position at each time point.
    metrics : dict[str, NDArray]
        Dictionary containing diagnostic metrics:
        - 'posterior': Posterior distribution, shape (n_time, n_bins)
        - 'hpd_overlap': HPD overlap, shape (n_time, n_cells)
        - 'kl_divergence': KL divergence, shape (n_time, n_cells)
        - 'spike_prob': Spike probability, shape (n_time, n_cells)
    thresholds : Thresholds
        Threshold values for each diagnostic.
    title : str, default "Original Metrics"
        Figure title.
    remap_window : tuple[int, int] | None, optional
        Time window where cell remapping occurs (start, end).
    phase_boundaries : tuple[int, ...] | None, optional
        Boundaries between phases: (T_remap_start, T_remap_end, T_recovery1_end,
        T_flat_end, T_recovery2_end, T_fast_end, T_recovery3_end, T_slow_end).

    Returns
    -------
    fig : matplotlib.figure.Figure
        The created figure.

    Examples
    --------
    >>> import numpy as np
    >>> from statespacecheck_paper.analysis import Thresholds
    >>> n_time, n_bins, n_cells = 100, 50, 10
    >>> xs = np.linspace(0, 1, n_bins)
    >>> x_true = np.random.uniform(0, n_bins - 1, n_time)
    >>> metrics = {
    ...     'posterior': np.random.dirichlet(np.ones(n_bins), size=n_time),
    ...     'hpd_overlap': np.random.uniform(0, 1, (n_time, n_cells)),
    ...     'kl_divergence': np.random.uniform(0, 5, (n_time, n_cells)),
    ...     'spike_prob': np.random.uniform(0, 1, (n_time, n_cells)),
    ... }
    >>> thresholds = Thresholds(hpd_overlap=0.8, kl_divergence=2.0, spike_prob=0.05)
    >>> fig = plot_original(xs, x_true, metrics, thresholds)
    >>> plt.close(fig)
    """
    n_time = metrics["posterior"].shape[0]
    fig, axes = plt.subplots(
        4,
        1,
        figsize=(8, 6),
        constrained_layout={
            "h_pad": 0.02,
            "w_pad": 0.02,
            "hspace": 0,
            "wspace": 0,
            "rect": [0, 0, 1, 0.97],
        },
        sharex=True,
        dpi=450,
    )

    im = axes[0].imshow(
        metrics["posterior"].T,
        aspect="auto",
        origin="lower",
        vmin=0.0,
        vmax=np.quantile(metrics["posterior"], 0.975),
        cmap=CMAP_POSTERIOR,
    )
    # Plot true position for visibility against posterior colormap
    axes[0].plot(
        np.arange(n_time),
        x_true,
        color=COLORS["ground_truth"],
        linewidth=1.5,
        alpha=0.85,
        label="True position",
    )
    axes[0].set_ylabel("Position (bin)", fontsize=7, labelpad=8)
    axes[0].tick_params(labelsize=6)

    # Create colorbar with better formatting
    cbar = fig.colorbar(im, ax=axes[0], fraction=0.03, pad=0.02, aspect=30)
    cbar.set_label("Probability (×10⁻¹²)", fontsize=7, labelpad=8)
    cbar.ax.tick_params(labelsize=6, length=3, width=0.5)
    # Scale tick labels by 1e12 to avoid offset text
    cbar.ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f"{x * 1e12:.1f}"))

    # Add phase boundaries to all axes
    if phase_boundaries is not None:
        add_phase_boundaries(axes, phase_boundaries, include_labels=True, alpha=0.2)

    # Create time indices for scatter plots (metrics are now 2D: n_time x n_cells)
    n_cells = metrics["hpd_overlap"].shape[1]
    time_indices = np.tile(np.arange(n_time)[:, np.newaxis], (1, n_cells))

    axes[1].scatter(
        time_indices.ravel(),
        metrics["hpd_overlap"].ravel(),
        s=1.5,
        alpha=0.6,
        c=COLORS["hpd_overlap"],
        rasterized=True,
    )
    axes[1].axhline(thresholds.hpd_overlap, color=COLORS["threshold"], linewidth=1.5, zorder=10)
    axes[1].set_xlim(0, n_time)
    axes[1].set_ylabel("HPD Overlap", fontsize=7, labelpad=8)
    axes[1].tick_params(labelsize=6)

    axes[2].scatter(
        time_indices.ravel(),
        metrics["kl_divergence"].ravel(),
        s=1.5,
        alpha=0.6,
        c=COLORS["kl_divergence"],
        rasterized=True,
    )
    axes[2].axhline(thresholds.kl_divergence, color=COLORS["threshold"], linewidth=1.5, zorder=10)
    axes[2].set_xlim(0, n_time)
    axes[2].set_ylabel("KL Divergence", fontsize=7, labelpad=8)
    axes[2].tick_params(labelsize=6)

    # Spike probability: lower values indicate worse fit
    axes[3].scatter(
        time_indices.ravel(),
        metrics["spike_prob"].ravel(),
        s=1.5,
        alpha=0.6,
        c=COLORS["metric_combined"],
        rasterized=True,
    )
    axes[3].axhline(thresholds.spike_prob, color=COLORS["threshold"], linewidth=1.5, zorder=10)
    axes[3].set_xlim(0, n_time)
    axes[3].set_ylabel("Spike Prob", fontsize=7, labelpad=8)
    axes[3].set_xlabel("Time", fontsize=7, labelpad=8)
    axes[3].tick_params(labelsize=6)

    # Add comprehensive legend outside the plot area at the bottom
    # Get handles and labels from axes[0] where they were defined
    handles, labels = axes[0].get_legend_handles_labels()
    axes[3].legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.35),
        fontsize=6,
        frameon=True,
        fancybox=False,
        shadow=False,
        ncol=5,
    )

    fig.suptitle(title, fontsize=8, y=0.99)
    return fig


def plot_transformed(
    xs: NDArray[np.floating],
    x_true: NDArray[np.floating],
    posterior: NDArray[np.floating],
    transformed: Transformed,
    title: str = "Transformed Metrics (-log, sqrt)",
    remap_window: tuple[int, int] | None = None,
    phase_boundaries: tuple[int, int] | None = None,
) -> Figure:
    """Plot transformed diagnostic metrics with thresholds.

    Applies transformations to make distributions more Gaussian for better
    visualization and threshold detection. Metrics are per-cell and displayed
    as scatter plots.

    Parameters
    ----------
    xs : NDArray, shape (n_bins,)
        Position bin centers.
    x_true : NDArray, shape (n_time,)
        True position at each time point.
    posterior : NDArray, shape (n_time, n_bins)
        Posterior distribution over time.
    transformed : Transformed
        Transformed metrics and thresholds (metrics have shape n_time, n_cells).
    title : str, default "Transformed Metrics (-log, sqrt)"
        Figure title.
    remap_window : tuple[int, int] | None, optional
        Time window where cell remapping occurs (start, end).
    phase_boundaries : tuple[int, int] | None, optional
        Boundaries between phases: (T1, T2) where T3 is end of data.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The created figure.

    Examples
    --------
    >>> import numpy as np
    >>> from statespacecheck_paper.analysis import Transformed
    >>> n_time, n_bins, n_cells = 100, 50, 10
    >>> xs = np.linspace(0, 1, n_bins)
    >>> x_true = np.random.uniform(0, n_bins - 1, n_time)
    >>> posterior = np.random.dirichlet(np.ones(n_bins), size=n_time)
    >>> transformed = Transformed(
    ...     hpd_overlap=np.random.uniform(0, 5, (n_time, n_cells)),
    ...     kl_divergence=np.random.uniform(0, 3, (n_time, n_cells)),
    ...     spike_prob=np.random.uniform(0, 10, (n_time, n_cells)),
    ...     hpd_overlap_threshold=3.0,
    ...     kl_divergence_threshold=2.0,
    ...     spike_prob_threshold=3.0,
    ... )
    >>> fig = plot_transformed(xs, x_true, posterior, transformed)
    >>> plt.close(fig)
    """
    n_time = posterior.shape[0]
    fig, axes = plt.subplots(4, 1, figsize=(7, 6), constrained_layout=True, sharex=True, dpi=150)

    im = axes[0].imshow(posterior.T, aspect="auto", origin="lower", cmap=CMAP_POSTERIOR)
    axes[0].plot(np.arange(n_time), x_true, color=COLORS["ground_truth"], linewidth=1.0, alpha=0.8)
    axes[0].set_ylabel("Position (bin)", fontsize=7, labelpad=8)
    axes[0].tick_params(labelsize=7)
    cbar = fig.colorbar(im, ax=axes[0], fraction=0.02, pad=0.02)
    cbar.set_label("Probability", fontsize=7, labelpad=8)
    cbar.ax.tick_params(labelsize=7)

    for ax in axes:
        # Highlight remap window (cell 10->1)
        if remap_window is not None:
            ax.axvspan(
                remap_window[0],
                remap_window[1],
                alpha=0.15,
                color=COLORS["likelihood"],
                label="Remap",
            )

        # Highlight phase boundaries
        if phase_boundaries is not None:
            t1, t2 = phase_boundaries
            ax.axvspan(t1, t2, alpha=0.15, color=COLORS["reference"], label="Flat rate")
            ax.axvspan(t2, n_time, alpha=0.15, color=COLORS["ground_truth"], label="Fast movement")

    # Create time indices for scatter plots (metrics are now 2D: n_time x n_cells)
    n_cells = transformed.hpd_overlap.shape[1]
    time_indices = np.tile(np.arange(n_time)[:, np.newaxis], (1, n_cells))

    axes[1].scatter(
        time_indices.ravel(),
        transformed.hpd_overlap.ravel(),
        s=0.5,
        alpha=0.3,
        c=COLORS["hpd_overlap"],
        rasterized=True,
    )
    axes[1].axhline(
        transformed.hpd_overlap_threshold,
        color=COLORS["threshold"],
        linewidth=1.5,
        label="Threshold",
        zorder=10,
    )
    axes[1].set_xlim(0, n_time)
    axes[1].set_ylabel("-log(HPD Overlap)", fontsize=7, labelpad=8)
    axes[1].tick_params(labelsize=7)
    axes[1].legend(loc="upper right", fontsize=7, frameon=False)

    axes[2].scatter(
        time_indices.ravel(),
        transformed.kl_divergence.ravel(),
        s=0.5,
        alpha=0.3,
        c=COLORS["kl_divergence"],
        rasterized=True,
    )
    axes[2].axhline(
        transformed.kl_divergence_threshold,
        color=COLORS["threshold"],
        linewidth=1.5,
        label="Threshold",
        zorder=10,
    )
    axes[2].set_xlim(0, n_time)
    axes[2].set_ylabel("sqrt(KL Divergence)", fontsize=7, labelpad=8)
    axes[2].tick_params(labelsize=7)

    axes[3].scatter(
        time_indices.ravel(),
        transformed.spike_prob.ravel(),
        s=0.5,
        alpha=0.3,
        c=COLORS["metric_combined"],
        rasterized=True,
    )
    axes[3].axhline(
        transformed.spike_prob_threshold,
        color=COLORS["threshold"],
        linewidth=1.5,
        label="Threshold",
        zorder=10,
    )
    axes[3].set_xlim(0, n_time)
    axes[3].set_ylabel("-log(Spike Prob)", fontsize=7, labelpad=8)
    axes[3].set_xlabel("Time", fontsize=7, labelpad=8)
    axes[3].tick_params(labelsize=7)

    fig.suptitle(title, fontsize=8, y=0.998)
    return fig


def plot_misfit_examples(
    xs: NDArray[np.floating],
    x_true: NDArray[np.floating],
    spikes: NDArray[np.floating],
    metrics: dict[str, NDArray[np.floating]],
    params: DecodeParams,
    placefield_centers: NDArray[np.floating],
    placefield_width: float,
    rate_scale: float,
) -> Figure:
    """Plot examples of high misfit moments for each scenario.

    Finds the worst time point in each misfit phase and shows the distributions.
    Also includes a baseline example with good fit.
    Shows 5 columns: baseline + 4 misfit types.

    Parameters
    ----------
    xs : NDArray, shape (n_bins,)
        Position bin centers.
    x_true : NDArray, shape (n_time,)
        True position at each time point.
    spikes : NDArray, shape (n_time, n_cells)
        Spike counts for each cell at each time point.
    metrics : dict[str, NDArray]
        Dictionary containing diagnostic metrics from decode_and_diagnostics.
        Metrics 'hpd_overlap', 'kl_divergence', 'spike_prob' have shape (n_time, n_cells).
    params : DecodeParams
        Decoding parameters containing timeline structure.
    pf_centers : NDArray, shape (n_cells,)
        Place field centers for each cell.
    pf_width : float
        Width of place fields (sigma).
    rate_scale : float
        Scaling factor for firing rates.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure showing distribution comparisons for baseline and four misfit types.

    Examples
    --------
    >>> import numpy as np
    >>> from statespacecheck_paper.analysis import DecodeParams
    >>> n_time, n_bins, n_cells = 500, 50, 10
    >>> xs = np.linspace(0, 1, n_bins)
    >>> x_true = np.random.uniform(0, n_bins - 1, n_time)
    >>> spikes = np.random.poisson(0.5, (n_time, n_cells))
    >>> metrics = {
    ...     'posterior': np.random.dirichlet(np.ones(n_bins), size=n_time),
    ...     'hpd_overlap': np.random.uniform(0, 1, (n_time, n_cells)),
    ...     'kl_divergence': np.random.uniform(0, 5, (n_time, n_cells)),
    ...     'spike_prob': np.random.uniform(0, 1, (n_time, n_cells)),
    ... }
    >>> params = DecodeParams(
    ...     T_remap_start=200, T_remap_end=250,
    ...     T_recovery1_end=280, T_flat_end=320, T_recovery2_end=350,
    ...     T_fast_end=390, T_recovery3_end=420, T_slow_end=460,
    ... )
    >>> pf_centers = np.linspace(0, 1, n_cells)
    >>> plot_misfit_examples(xs, x_true, spikes, metrics, params, pf_centers, 0.1, 10.0)
    >>> plt.close('all')
    """
    # Define phase windows - include baseline (good fit) and misfit phases
    baseline_window = slice(1000, params.T_remap_start - 1000)  # Middle of baseline
    remap_window = slice(params.T_remap_start, params.T_remap_end)
    flat_window = slice(params.T_recovery1_end, params.T_flat_end)
    fast_window = slice(params.T_recovery2_end, params.T_fast_end)
    slow_window = slice(params.T_recovery3_end, params.T_slow_end)

    phases = [
        ("Baseline", baseline_window, True),  # Third element indicates if it's baseline
        ("Remapping", remap_window, False),
        ("Flat Firing", flat_window, False),
        ("Fast Movement", fast_window, False),
        ("Slow Movement", slow_window, False),
    ]

    # Publication quality: 450 DPI, single row with 5 columns
    fig = plt.figure(figsize=(12.0, 2.5), dpi=450, constrained_layout=True)
    gs = fig.add_gridspec(1, 5)
    axes = [fig.add_subplot(gs[0, i]) for i in range(5)]

    for phase_idx, (phase_name, phase_slice, is_baseline) in enumerate(phases):
        # For baseline, find best fit (highest hpd_overlap); for misfits,
        # find worst fit (lowest hpd_overlap)
        # BUT: only consider time points with spikes so likelihood is informative
        # Metrics are now (n_time, n_cells), use mean across cells for selection
        phase_hpdo = np.nanmean(metrics["hpd_overlap"][phase_slice], axis=1)
        phase_spikes = spikes[phase_slice]

        # Mask times without spikes (likelihood will be flat/uninformative)
        has_spikes = phase_spikes.sum(axis=1) > 0
        valid_hpdo = phase_hpdo.copy()
        valid_hpdo[~has_spikes] = np.nan  # Exclude times without spikes

        if is_baseline:
            example_idx_in_phase = np.nanargmax(valid_hpdo)  # Best fit with spikes
        else:
            example_idx_in_phase = np.nanargmin(valid_hpdo)  # Worst fit with spikes
        example_time = phase_slice.start + example_idx_in_phase

        # Recompute prior and likelihood at this time point
        # Get posterior from previous timestep
        if example_time > 0:
            prev_post = metrics["posterior"][example_time - 1]
        else:
            prev_post = np.ones_like(xs) / len(xs)

        # Select appropriate transition matrix (half-open intervals [start, end)
        # to match decode_and_diagnostics)
        if params.T_recovery2_end <= example_time < params.T_fast_end:
            transition_matrix = gaussian_transition_matrix(xs, params.sigx_pred_fast_phase)
        elif params.T_recovery3_end <= example_time < params.T_slow_end:
            transition_matrix = gaussian_transition_matrix(xs, params.sigx_pred_slow_phase)
        else:
            transition_matrix = gaussian_transition_matrix(xs, params.sigx_pred)

        # Compute prior
        prior = normalize(prev_post @ transition_matrix)

        # Compute combined likelihood (with remapping if in remap window, half-open)
        active_remap = params.T_remap_start <= example_time < params.T_remap_end
        current_pf_centers = get_remapped_pf_centers(
            placefield_centers, params.remap_from_to, active_remap
        )
        likelihood = likelihood_grid_for_counts(
            xs, current_pf_centers, placefield_width, rate_scale, spikes[example_time]
        )

        combined_likelihood = normalize(np.prod(likelihood, axis=1))

        # Plot prior and likelihood with twin axes - use Wong colorblind-friendly palette
        ax1 = axes[phase_idx]
        ax2 = ax1.twinx()

        # Plot prior on left axis with transparency
        line1 = ax1.plot(
            xs, prior, color=COLORS["predictive"], linewidth=1.5, alpha=0.7, label="Predictive"
        )
        # Determine scale factor for prior and include in ylabel
        prior_max = np.max(prior)
        if prior_max > 0:
            prior_order = int(np.floor(np.log10(prior_max)))
            # Use scale factor if magnitude is outside reasonable range
            if prior_order < -2 or prior_order > 2:
                prior_scale = 10**prior_order
                ax1.plot(
                    xs, prior / prior_scale, color=COLORS["predictive"], linewidth=1.5, alpha=0.7
                )
                ax1.lines[0].remove()  # Remove the unscaled plot
                ax1.set_ylabel(
                    f"Predictive (×10$^{{{prior_order}}}$)",
                    fontsize=7,
                    color=COLORS["predictive"],
                    labelpad=3,
                )
            else:
                ax1.set_ylabel("Predictive", fontsize=7, color=COLORS["predictive"], labelpad=3)
        else:
            ax1.set_ylabel("Predictive", fontsize=7, color=COLORS["predictive"], labelpad=3)
        ax1.tick_params(axis="y", labelcolor=COLORS["predictive"], labelsize=6)
        ax1.set_ylim(0, None)

        # Plot likelihood on right axis - solid line
        likelihood_max = np.max(combined_likelihood)
        if likelihood_max > 0:
            likelihood_order = int(np.floor(np.log10(likelihood_max)))
            # Use scale factor if magnitude is outside reasonable range
            if likelihood_order < -2 or likelihood_order > 2:
                likelihood_scale = 10**likelihood_order
                line2 = ax2.plot(
                    xs,
                    combined_likelihood / likelihood_scale,
                    color=COLORS["likelihood"],
                    linewidth=1.5,
                    alpha=0.9,
                    label="Likelihood",
                )
                ax2.set_ylabel(
                    f"Likelihood (×10$^{{{likelihood_order}}}$)",
                    fontsize=7,
                    color=COLORS["likelihood"],
                    labelpad=3,
                )
            else:
                line2 = ax2.plot(
                    xs,
                    combined_likelihood,
                    color=COLORS["likelihood"],
                    linewidth=1.5,
                    alpha=0.9,
                    label="Likelihood",
                )
                ax2.set_ylabel("Likelihood", fontsize=7, color=COLORS["likelihood"], labelpad=3)
        else:
            line2 = ax2.plot(
                xs,
                combined_likelihood,
                color=COLORS["likelihood"],
                linewidth=1.5,
                alpha=0.9,
                label="Likelihood",
            )
            ax2.set_ylabel("Likelihood", fontsize=7, color=COLORS["likelihood"], labelpad=3)
        ax2.tick_params(axis="y", labelcolor=COLORS["likelihood"], labelsize=6)
        ax2.set_ylim(0, None)

        # Add true position line
        ax1.axvline(
            x_true[example_time],
            color=COLORS["ground_truth"],
            linestyle="--",
            linewidth=1.0,
            alpha=0.7,
        )

        # Get diagnostic values - now per-cell, use nanmean for display
        hpdo_val = np.nanmean(metrics["hpd_overlap"][example_time])
        kl_val = np.nanmean(metrics["kl_divergence"][example_time])
        spike_prob_val = np.nanmean(metrics["spike_prob"][example_time])

        # Add phase name and metrics as title
        if np.isnan(spike_prob_val):
            title_text = f"{phase_name}\nHPD: {hpdo_val:.2g}  KL: {kl_val:.2g}  SP: N/A"
        else:
            sp_str = f"{spike_prob_val:.2g}"
            title_text = f"{phase_name}\nHPD: {hpdo_val:.2g}  KL: {kl_val:.2g}  SP: {sp_str}"
        ax1.set_title(title_text, fontsize=7, pad=5, fontweight="bold")

        ax1.tick_params(axis="x", labelsize=6)
        ax1.set_xlabel("Position", fontsize=7, labelpad=3)

        # Add legend to first panel only
        if phase_idx == 0:
            lines = line1 + line2
            labels = [str(line.get_label()) for line in lines]
            ax1.legend(lines, labels, fontsize=5, loc="lower right", frameon=False)

    return fig


def _plot_timeseries_heatmap(
    ax: Axes,
    data: NDArray[np.floating],
    x_true: NDArray[np.floating] | None = None,
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
    x_true : NDArray, shape (n_time,), optional
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
    >>> x_true = np.random.uniform(0, 49, 100)
    >>> im = _plot_timeseries_heatmap(ax, data, x_true)
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
    if x_true is not None:
        ax.plot(
            np.arange(n_time),
            x_true,
            color=COLORS["ground_truth"],
            linewidth=1.0,
            alpha=0.85,
        )
    return im


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
        # Row-normalize to [0, 1]
        rmin, rmax = float(np.nanmin(lik_row)), float(np.nanmax(lik_row))
        if rmax <= rmin:
            continue
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


def _plot_likelihood_overlay(
    ax: Axes,
    predictive: NDArray[np.floating],
    per_spike_likelihood: NDArray[np.floating],
    spike_time_ind: NDArray[np.intp],
    spike_cell_ind: NDArray[np.intp],
    position_bins: NDArray[np.floating],
    x_true: NDArray[np.floating] | None = None,
    cmap_underlay: str = CMAP_POSTERIOR,
    cmap_overlay: str = CMAP_LIKELIHOOD,
    underlay_alpha: float = 0.35,
) -> AxesImage:
    """Plot per-spike likelihood distributions over a faint predictive underlay.

    Aggregates per-spike likelihoods into per-timestep distributions and renders
    each spike time as a colored column with guaranteed minimum width.

    Parameters
    ----------
    ax : Axes
        Matplotlib axes to plot on.
    predictive : NDArray, shape (n_time, n_bins)
        Predictive distribution over position at each time.
    per_spike_likelihood : NDArray, shape (n_spikes, n_bins)
        Normalized likelihood distribution for each individual spike event.
    spike_time_ind : NDArray, shape (n_spikes,)
        Time index for each spike event.
    spike_cell_ind : NDArray, shape (n_spikes,)
        Cell index for each spike event.
    position_bins : NDArray, shape (n_bins,)
        Position bin centers.
    x_true : NDArray, shape (n_time,), optional
        True position to overlay.
    cmap_underlay : str, default CMAP_POSTERIOR
        Colormap for the faint predictive underlay.
    cmap_overlay : str, default CMAP_LIKELIHOOD
        Colormap for the likelihood columns.
    underlay_alpha : float, default 0.35
        Alpha for the predictive underlay.

    Returns
    -------
    im : AxesImage
        The underlay image object.
    """
    n_time, n_bins = predictive.shape

    # Underlay: faint predictive heatmap
    im = ax.imshow(
        predictive.T,
        aspect="auto",
        origin="lower",
        vmin=0.0,
        vmax=float(np.nanquantile(predictive, 0.975)),
        cmap=cmap_underlay,
        alpha=underlay_alpha,
    )

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

    if x_true is not None:
        ax.plot(
            np.arange(n_time),
            x_true,
            color=COLORS["ground_truth"],
            linewidth=1.0,
            alpha=0.85,
        )

    ax.set_xlim(0, n_time - 1)

    return im


def _plot_spike_count_raster(
    ax: Axes,
    spikes: NDArray[np.floating],
    placefield_centers: NDArray[np.floating],
) -> None:
    """Plot spike counts as a raster, sorted by place field peak.

    Neurons are sorted by their place field center position so that
    sequential activation during movement is visible as diagonal patterns.
    Uses scatter plot for better visibility of sparse events.

    Parameters
    ----------
    ax : Axes
        Matplotlib axes to plot on.
    spikes : NDArray, shape (n_time, n_cells)
        Spike counts at each timestep for each cell.
    placefield_centers : NDArray, shape (n_cells,)
        Place field centers for sorting neurons by preferred position.

    Examples
    --------
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> fig, ax = plt.subplots()
    >>> spikes = np.random.poisson(0.1, (100, 20))
    >>> pf_centers = np.linspace(0, 1, 20)
    >>> _plot_spike_count_raster(ax, spikes, pf_centers)
    >>> plt.close(fig)
    """
    # Sort neurons by place field peak position
    sort_order = np.argsort(placefield_centers)
    spikes_sorted = spikes[:, sort_order]

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
    ax.set_ylabel("Neuron", fontsize=7, labelpad=7)


def plot_combined_diagnostics(
    xs: NDArray[np.floating],
    x_true: NDArray[np.floating],
    spikes: NDArray[np.floating],
    metrics: dict[str, Any],
    thresholds: Thresholds,
    params: DecodeParams,
    placefield_centers: NDArray[np.floating],
) -> Figure:
    """Create comprehensive time-series diagnostics figure.

    Layout: 6 time-series panels (predictive, likelihood, raster, HPDO, KL,
    spike prob) with shared x-axis and phase boundary overlays.

    Parameters
    ----------
    xs : NDArray, shape (n_bins,)
        Position bin centers.
    x_true : NDArray, shape (n_time,)
        True position at each time point.
    spikes : NDArray, shape (n_time, n_cells)
        Spike counts for each cell at each time point.
    metrics : dict[str, NDArray]
        Dictionary containing diagnostic metrics from decode_and_diagnostics.
        Metrics 'hpd_overlap', 'kl_divergence', 'spike_prob' have shape (n_time, n_cells).
    thresholds : Thresholds
        Threshold values for each diagnostic.
    params : DecodeParams
        Decoding parameters containing timeline structure.
    placefield_centers : NDArray, shape (n_cells,)
        Place field centers for each cell (used for spike raster sorting).

    Returns
    -------
    fig : matplotlib.figure.Figure
        Time-series diagnostic figure.

    Examples
    --------
    >>> import numpy as np
    >>> from statespacecheck_paper.analysis import DecodeParams, Thresholds
    >>> n_time, n_bins, n_cells = 500, 50, 10
    >>> xs = np.linspace(0, 1, n_bins)
    >>> x_true = np.random.uniform(0, n_bins - 1, n_time)
    >>> spikes = np.random.poisson(0.5, (n_time, n_cells))
    >>> spike_time_ind = np.array([1, 2, 3])
    >>> spike_cell_ind = np.array([0, 1, 2])
    >>> metrics = {
    ...     'predictive': np.random.dirichlet(np.ones(n_bins), size=n_time),
    ...     'likelihood': np.random.dirichlet(np.ones(n_bins), size=n_time),
    ...     'posterior': np.random.dirichlet(np.ones(n_bins), size=n_time),
    ...     'hpd_overlap': np.random.uniform(0, 1, (n_time, n_cells)),
    ...     'kl_divergence': np.random.uniform(0, 5, (n_time, n_cells)),
    ...     'spike_prob': np.random.uniform(0, 1, (n_time, n_cells)),
    ...     'per_spike_likelihood': np.random.dirichlet(np.ones(n_bins), size=3),
    ...     'spike_time_ind': spike_time_ind,
    ...     'spike_cell_ind': spike_cell_ind,
    ... }
    >>> thresholds = Thresholds(hpd_overlap=0.8, kl_divergence=2.0, spike_prob=0.05)
    >>> params = DecodeParams(
    ...     T_remap_start=200, T_remap_end=250,
    ...     T_recovery1_end=280, T_flat_end=320, T_recovery2_end=350,
    ...     T_fast_end=390, T_recovery3_end=420, T_slow_end=460,
    ... )
    >>> placefield_centers = np.linspace(0, 1, n_cells)
    >>> plot_combined_diagnostics(
    ...     xs, x_true, spikes, metrics, thresholds, params, placefield_centers
    ... )
    >>> plt.close('all')
    """
    # Calculate figure size
    fig_width = 7.0  # Full page width
    fig_height = 7.0

    fig = plt.figure(figsize=(fig_width, fig_height), dpi=450)

    # Create grid: 6 rows for time-series (pred, like, raster, hpdo, kl, spike)
    gs = gridspec.GridSpec(
        6,
        1,
        figure=fig,
        height_ratios=[1.2, 1.2, 0.8, 0.7, 0.7, 0.7],
        hspace=0.12,
        left=0.08,
        right=0.93,
        top=0.97,
        bottom=0.06,
    )

    # ===== TOP SECTION: Time-Series Diagnostics =====

    n_time = metrics["posterior"].shape[0]

    # Create time-series axes with shared x-axis
    # Order: Predictive -> Likelihood -> Raster -> HPD -> KL -> Spike
    ax_pred = fig.add_subplot(gs[0])
    ax_like = fig.add_subplot(gs[1], sharex=ax_pred)
    ax_raster = fig.add_subplot(gs[2], sharex=ax_pred)
    ax_hpdo = fig.add_subplot(gs[3], sharex=ax_pred)
    ax_kl = fig.add_subplot(gs[4], sharex=ax_pred)
    ax_spike = fig.add_subplot(gs[5], sharex=ax_pred)

    # Predictive heatmap
    _plot_timeseries_heatmap(ax_pred, metrics["predictive"], x_true)
    ax_pred.set_ylabel("Position (a.u.)", fontsize=7, labelpad=7)
    ax_pred.tick_params(labelsize=6, labelbottom=False)
    ax_pred.legend(
        [Line2D([0], [0], color=COLORS["ground_truth"], linewidth=1.0)],
        ["True position"],
        loc="upper left",
        fontsize=6,
        frameon=False,
    )
    # Add label "Predictive" on right side
    ax_pred.text(
        1.01,
        0.5,
        "Predictive",
        transform=ax_pred.transAxes,
        fontsize=7,
        va="center",
        ha="left",
        rotation=270,
    )

    # Likelihood overlay: per-spike likelihood bars at spike times
    _plot_likelihood_overlay(
        ax_like,
        metrics["predictive"],
        metrics["per_spike_likelihood"],
        metrics["spike_time_ind"],
        metrics["spike_cell_ind"],
        xs,
        x_true,
    )
    ax_like.set_ylabel("Position (a.u.)", fontsize=7, labelpad=7)
    ax_like.tick_params(labelsize=6, labelbottom=False)
    ax_like.text(
        1.01,
        0.5,
        "Likelihood",
        transform=ax_like.transAxes,
        fontsize=7,
        va="center",
        ha="left",
        rotation=270,
    )

    # Spike raster (sorted by place field peak)
    _plot_spike_count_raster(ax_raster, spikes, placefield_centers)
    ax_raster.tick_params(labelsize=6, labelbottom=False)
    ax_raster.text(
        1.01,
        0.5,
        "Spikes",
        transform=ax_raster.transAxes,
        fontsize=7,
        va="center",
        ha="left",
        rotation=270,
    )

    # Create time indices for scatter plots (metrics are now 2D: n_time x n_cells)
    n_cells = metrics["hpd_overlap"].shape[1]
    time_indices = np.tile(np.arange(n_time)[:, np.newaxis], (1, n_cells))

    # HPDO
    ax_hpdo.scatter(
        time_indices.ravel(),
        metrics["hpd_overlap"].ravel(),
        s=0.8,
        alpha=0.6,
        c=COLORS["hpd_overlap"],
        rasterized=True,
    )
    ax_hpdo.axhline(
        thresholds.hpd_overlap, color=COLORS["threshold"], linewidth=1.2, alpha=0.7, zorder=10
    )
    ax_hpdo.set_xlim(0, n_time)
    ax_hpdo.set_ylabel("HPD Overlap", fontsize=7, labelpad=7)
    ax_hpdo.tick_params(labelsize=6, labelbottom=False)
    # Add directional indicator and threshold annotation
    ax_hpdo.text(
        1.01, 0.02, "↓ Worse fit", transform=ax_hpdo.transAxes, fontsize=6, va="bottom", ha="left"
    )
    ax_hpdo.text(
        1.01,
        thresholds.hpd_overlap,
        "Threshold",
        transform=ax_hpdo.get_yaxis_transform(),
        fontsize=6,
        va="center",
        ha="left",
        color=COLORS["threshold"],
    )

    # KL Divergence
    ax_kl.scatter(
        time_indices.ravel(),
        metrics["kl_divergence"].ravel(),
        s=0.8,
        alpha=0.6,
        c=COLORS["kl_divergence"],
        rasterized=True,
    )
    ax_kl.axhline(
        thresholds.kl_divergence, color=COLORS["threshold"], linewidth=1.2, alpha=0.7, zorder=10
    )
    ax_kl.set_xlim(0, n_time)
    ax_kl.set_ylabel("KL Divergence", fontsize=7, labelpad=7)
    ax_kl.tick_params(labelsize=6, labelbottom=False)
    # Add directional indicator and threshold annotation
    ax_kl.text(
        1.01, 0.5, "↑ Worse fit", transform=ax_kl.transAxes, fontsize=6, va="center", ha="left"
    )
    ax_kl.text(
        1.01,
        thresholds.kl_divergence,
        "Threshold",
        transform=ax_kl.get_yaxis_transform(),
        fontsize=6,
        va="center",
        ha="left",
        color=COLORS["threshold"],
    )

    # Spike probability: transform to -log10(p) so higher values indicate worse fit
    # This makes interpretation consistent with KL divergence (higher = worse)
    spike_prob_transformed = -np.log10(np.maximum(metrics["spike_prob"], 1e-10))
    threshold_transformed = -np.log10(np.maximum(thresholds.spike_prob, 1e-10))
    ax_spike.scatter(
        time_indices.ravel(),
        spike_prob_transformed.ravel(),
        s=0.8,
        alpha=0.6,
        c=COLORS["metric_combined"],
        rasterized=True,
    )
    ax_spike.axhline(
        threshold_transformed,
        color=COLORS["threshold"],
        linewidth=1.2,
        alpha=0.7,
        zorder=10,
        label="Threshold",
    )
    ax_spike.set_xlim(0, n_time)
    ax_spike.set_ylabel(r"$-\log_{10}(p)$", fontsize=7, labelpad=7)
    ax_spike.set_xlabel("Time (a.u.)", fontsize=7, labelpad=7)
    ax_spike.tick_params(labelsize=7)
    # Add directional indicator (higher values now indicate misfit after log transform)
    ax_spike.text(
        1.01,
        0.5,
        "↑ Worse fit",
        transform=ax_spike.transAxes,
        fontsize=6,
        va="center",
        ha="left",
    )
    ax_spike.text(
        1.01,
        threshold_transformed,
        "Threshold",
        transform=ax_spike.get_yaxis_transform(),
        fontsize=6,
        va="center",
        ha="left",
        color=COLORS["threshold"],
    )

    # Add phase boundaries to all time-series panels
    phase_boundaries = (
        params.T_remap_start,
        params.T_remap_end,
        params.T_recovery1_end,
        params.T_flat_end,
        params.T_recovery2_end,
        params.T_fast_end,
        params.T_recovery3_end,
        params.T_slow_end,
    )

    # Add phase boundaries with matching colors to all panels
    add_phase_boundaries(
        [ax_pred, ax_like, ax_raster, ax_hpdo, ax_kl, ax_spike],
        phase_boundaries,
        alpha=0.15,
    )

    # Unpack for phase labels
    (
        t_remap_start,
        t_remap_end,
        t_recovery1_end,
        t_flat_end,
        t_recovery2_end,
        t_fast_end,
        t_recovery3_end,
        t_slow_end,
    ) = phase_boundaries

    # Add phase labels above top panel (ax_pred is now the top panel)
    phase_labels_info = [
        ((t_remap_start + t_remap_end) / 2, "Remap", "c"),
        ((t_recovery1_end + t_flat_end) / 2, "Flat", "d"),
        ((t_recovery2_end + t_fast_end) / 2, "Fast", "e"),
        ((t_recovery3_end + t_slow_end) / 2, "Slow", "f"),
    ]
    for x_pos, label_text, panel_id in phase_labels_info:
        ax_pred.text(
            x_pos,
            1.02,
            f"{label_text} ({panel_id})",
            transform=ax_pred.get_xaxis_transform(),
            fontsize=6,
            ha="center",
            va="bottom",
            style="italic",
        )

    return fig
