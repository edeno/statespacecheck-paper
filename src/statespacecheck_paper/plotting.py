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

from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.image import AxesImage
from numpy.typing import NDArray

from statespacecheck_paper.analysis import (
    DecodeParams,
    Diagnostics,
    PhaseBoundary,
    Thresholds,
    compute_phase_flag_fractions,
    get_remapped_pf_centers,
    likelihood_grid_for_counts,
    summary_phase_windows,
)
from statespacecheck_paper.simulation import (
    gaussian_transition_matrix,
    normalize,
    softmax_with_shift,
)
from statespacecheck_paper.style import CMAP_LIKELIHOOD, CMAP_POSTERIOR, COLORS

FIGURE3_PANEL_LABEL_GID = "figure3-panel-label"
FIGURE3_PHASE_LABEL_GID = "figure3-phase-label"
FIGURE3_ROW_LABEL_GID = "figure3-row-label"
FIGURE3_THRESHOLD_LABEL_GID = "figure3-threshold-label"
FIGURE3_THRESHOLD_LINE_GID = "figure3-threshold-line"
FIGURE3_TRUE_POSITION_LABEL_GID = "figure3-true-position-label"
FIGURE3_WORSE_FIT_LABEL_GID = "figure3-worse-fit-label"
FIGURE3_SUMMARY_CELL_LABEL_GID = "figure3-summary-cell-label"
FIGURE3_SUMMARY_COMPONENT_LABEL_GID = "figure3-summary-component-label"
FIGURE3_SUMMARY_KNOWN_COMPONENT_LABEL_GID = "figure3-summary-known-component-label"
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
        "event_kl_divergence",
        "kl_divergence",
        "KL div.",
        COLORS["kl_divergence"],
        "↑ Worse fit",
    ),
    DiagnosticRowSpec(
        "event_spike_prob",
        "spike_prob",
        "−log(p)",
        COLORS["metric_combined"],
        "↑ Worse fit",
        log_transform=True,
    ),
)


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
        Phase boundary time points, in the canonical 8-element order
        indexed by :class:`statespacecheck_paper.analysis.PhaseBoundary`
        (``REMAP_START``, ``REMAP_END``, ``RECOVERY1_END``,
        ``HIST_DEP_END``, ``RECOVERY2_END``, ``DRIFT_END``,
        ``RECOVERY3_END``, ``WIDE_DYNAMICS_END``). Shorter tuples (down
        to 2 elements) are accepted and produce a partial shading; only
        the misfit windows whose pair of boundary entries is present
        are drawn.
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
                "Wide dynamics noise",
            )
        )
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

    dx = float(x[1] - x[0])
    pdf_likelihood = pdf_likelihood / (np.sum(pdf_likelihood) * dx)

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
        ax.set_title(title, fontsize=8, fontweight="normal", pad=2)

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
            fontsize=8,
            color=color_predictive,
        )
        ax.text(
            16,
            0.22,
            "Likelihood",
            ha="center",
            va="bottom",
            fontsize=8,
            color=color_likelihood,
        )


def plot_misfit_examples(
    xs: NDArray[np.floating],
    x_true: NDArray[np.floating],
    spikes: NDArray[np.floating],
    metrics: Diagnostics,
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
    >>> import matplotlib.pyplot as plt
    >>> from statespacecheck_paper.analysis import DecodeParams
    >>> rng = np.random.default_rng(0)
    >>> # n_time must be large enough for the baseline window
    >>> # slice(1000, phase_boundaries[REMAP_START] - 1000) to be non-empty.
    >>> n_time, n_bins, n_cells = 6000, 50, 10
    >>> xs = np.linspace(0, 1, n_bins)
    >>> # Build inputs and run through ``decode_and_diagnostics`` to get
    >>> # a ``Diagnostics``; see tests/test_plotting.py for a worked
    >>> # example fixture. The doctest skips the actual call.
    """
    # Define phase windows - one example timestep per misfit class.
    bnd = params.phase_boundaries
    baseline_window = slice(1000, bnd[PhaseBoundary.REMAP_START] - 1000)  # Middle of baseline
    remap_window = slice(bnd[PhaseBoundary.REMAP_START], bnd[PhaseBoundary.REMAP_END])
    hist_dep_window = slice(bnd[PhaseBoundary.RECOVERY1_END], bnd[PhaseBoundary.HIST_DEP_END])
    drift_window = slice(bnd[PhaseBoundary.RECOVERY2_END], bnd[PhaseBoundary.DRIFT_END])
    wide_dynamics_window = slice(
        bnd[PhaseBoundary.RECOVERY3_END], bnd[PhaseBoundary.WIDE_DYNAMICS_END]
    )

    phases = [
        ("Baseline", baseline_window, True),
        ("Remap", remap_window, False),
        ("History-dep.", hist_dep_window, False),
        ("Drift", drift_window, False),
        ("Wide dyn. noise", wide_dynamics_window, False),
    ]

    # Publication quality: 450 DPI, single row with one column per phase
    n_phases = len(phases)
    fig = plt.figure(figsize=(2.4 * n_phases, 2.5), dpi=450, constrained_layout=True)
    gs = fig.add_gridspec(1, n_phases)
    axes = [fig.add_subplot(gs[0, i]) for i in range(n_phases)]

    for phase_idx, (phase_name, phase_slice, is_baseline) in enumerate(phases):
        # For baseline, find best fit (highest hpd_overlap); for misfits,
        # find worst fit (lowest hpd_overlap)
        # BUT: only consider time points with spikes so likelihood is informative
        # Metrics are now (n_time, n_cells), use mean across cells for selection
        phase_hpdo = np.nanmean(metrics.hpd_overlap[phase_slice], axis=1)
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

        if example_time > 0:
            prev_post = metrics.posterior[example_time - 1]
        else:
            prev_post = np.ones_like(xs) / len(xs)

        # Select appropriate transition matrix (half-open intervals [start, end)
        # to match decode_and_diagnostics). Only the wide-dynamics-noise
        # window uses an alternate transition matrix; all other phases use
        # the baseline.
        if bnd[PhaseBoundary.RECOVERY3_END] <= example_time < bnd[PhaseBoundary.WIDE_DYNAMICS_END]:
            transition_matrix = gaussian_transition_matrix(xs, params.sigx_wide_dynamics)
        else:
            transition_matrix = gaussian_transition_matrix(xs, params.sigx_pred)

        # Column-stochastic transition (see ``gaussian_transition_matrix``):
        # the predictive marginal is ``T @ post``. Mirrors the decoder in
        # ``decode_and_diagnostics``.
        prior = normalize(transition_matrix @ prev_post)

        active_remap = bnd[PhaseBoundary.REMAP_START] <= example_time < bnd[PhaseBoundary.REMAP_END]
        current_pf_centers = get_remapped_pf_centers(
            placefield_centers, params.remap_from_to, active_remap
        )
        likelihood = likelihood_grid_for_counts(
            xs, current_pf_centers, placefield_width, rate_scale, spikes[example_time]
        )

        # Combine per-cell normalized likelihoods across cells in log-space.
        # The linear-space ``np.prod(likelihood, axis=1)`` underflows once
        # n_cells * log(peak) crosses the float64 floor (~700).
        log_lik = np.log(np.maximum(likelihood, np.finfo(likelihood.dtype).tiny))
        combined_likelihood = softmax_with_shift(log_lik.sum(axis=1))

        ax1 = axes[phase_idx]
        ax2 = ax1.twinx()

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
                    fontsize=8,
                    color=COLORS["predictive"],
                    labelpad=3,
                )
            else:
                ax1.set_ylabel("Predictive", fontsize=8, color=COLORS["predictive"], labelpad=3)
        else:
            ax1.set_ylabel("Predictive", fontsize=8, color=COLORS["predictive"], labelpad=3)
        ax1.tick_params(axis="y", labelcolor=COLORS["predictive"], labelsize=8)
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
                    fontsize=8,
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
                ax2.set_ylabel("Likelihood", fontsize=8, color=COLORS["likelihood"], labelpad=3)
        else:
            line2 = ax2.plot(
                xs,
                combined_likelihood,
                color=COLORS["likelihood"],
                linewidth=1.5,
                alpha=0.9,
                label="Likelihood",
            )
            ax2.set_ylabel("Likelihood", fontsize=8, color=COLORS["likelihood"], labelpad=3)
        ax2.tick_params(axis="y", labelcolor=COLORS["likelihood"], labelsize=8)
        ax2.set_ylim(0, None)

        # Add true position line
        ax1.axvline(
            x_true[example_time],
            color=COLORS["ground_truth"],
            linestyle="--",
            linewidth=1.0,
            alpha=0.7,
        )

        hpdo_val = np.nanmean(metrics.hpd_overlap[example_time])
        kl_val = np.nanmean(metrics.kl_divergence[example_time])
        spike_prob_val = np.nanmean(metrics.spike_prob[example_time])

        # Add phase name and metrics as title
        if np.isnan(spike_prob_val):
            title_text = f"{phase_name}\nHPD: {hpdo_val:.2g}  KL: {kl_val:.2g}  SP: N/A"
        else:
            sp_str = f"{spike_prob_val:.2g}"
            title_text = f"{phase_name}\nHPD: {hpdo_val:.2g}  KL: {kl_val:.2g}  SP: {sp_str}"
        ax1.set_title(title_text, fontsize=8, pad=5, fontweight="bold")

        ax1.tick_params(axis="x", labelsize=8)
        ax1.set_xlabel("Position", fontsize=8, labelpad=3)

        # Add legend to first panel only
        if phase_idx == 0:
            lines = line1 + line2
            labels = [str(line.get_label()) for line in lines]
            ax1.legend(lines, labels, fontsize=8, loc="lower right", frameon=False)

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
            alpha=0.5,
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
        # Row-normalize to [0, 1]. A flat-or-degenerate likelihood
        # (rmax == rmin, or both NaN) is still real information —
        # render at mid-color rather than silently dropping the column,
        # which would be indistinguishable from "no spike at this time".
        rmin, rmax = float(np.nanmin(lik_row)), float(np.nanmax(lik_row))
        if not np.isfinite(rmax) or rmax <= rmin:
            normed = np.full_like(lik_row, 0.5)
        else:
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
    x_true: NDArray[np.floating] | None = None,
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
    x_true : NDArray, shape (n_time,), optional
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

    if x_true is not None:
        ax.plot(
            np.arange(n_time),
            x_true,
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
    ax.set_ylabel("Neuron", fontsize=8, labelpad=7)


def _add_figure3_row_label(ax: Axes, label: str) -> None:
    """Add the right-side row label used in Figure 3 panel (a)."""
    row_label = ax.text(
        1.01,
        0.5,
        label,
        transform=ax.transAxes,
        fontsize=8,
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
        fontsize=8,
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
        fontsize=8,
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
        fontsize=8,
        va="center",
        ha="left",
    )
    worse_fit_label.set_gid(FIGURE3_WORSE_FIT_LABEL_GID)


def _plot_figure3_predictive_row(
    ax: Axes,
    predictive: NDArray[np.floating],
    x_true: NDArray[np.floating],
) -> None:
    """Plot Figure 3's predictive row with a direct true-position label."""
    _plot_timeseries_heatmap(ax, predictive, x_true)
    ax.set_ylabel("Position (a.u.)", fontsize=8, labelpad=7)
    ax.tick_params(labelsize=8, labelbottom=False)
    true_position_label = ax.text(
        0.02,
        0.90,
        "True position",
        transform=ax.transAxes,
        fontsize=8,
        color=COLORS["ground_truth"],
        va="top",
        ha="left",
    )
    true_position_label.set_gid(FIGURE3_TRUE_POSITION_LABEL_GID)
    _add_figure3_row_label(ax, "Predictive")


def _plot_figure3_likelihood_row(
    ax: Axes,
    metrics: Diagnostics,
    x_true: NDArray[np.floating],
) -> None:
    """Plot Figure 3's per-spike likelihood row."""
    _plot_likelihood_overlay(
        ax,
        metrics.predictive,
        metrics.per_spike_likelihood,
        metrics.event_time_ind,
        x_true=x_true,
    )
    ax.set_ylabel("Position (a.u.)", fontsize=8, labelpad=7)
    ax.tick_params(labelsize=8, labelbottom=False)
    _add_figure3_row_label(ax, "Likelihood")


def _plot_figure3_raster_row(
    ax: Axes,
    spikes: NDArray[np.floating],
    placefield_centers: NDArray[np.floating],
) -> None:
    """Plot Figure 3's spike-count raster row."""
    _plot_spike_count_raster(ax, spikes, placefield_centers)
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
        plot_values = -np.log(np.maximum(plot_values, 1e-10))
        plot_threshold = -np.log(np.maximum(plot_threshold, 1e-10))

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
    ax.set_ylabel(spec.ylabel, fontsize=8, labelpad=7)
    if show_xlabel:
        ax.set_xlabel("Time (a.u.)", fontsize=8, labelpad=7)
        ax.tick_params(labelsize=8)
    else:
        ax.tick_params(labelsize=8, labelbottom=False)

    _add_figure3_worse_fit_label(ax, spec.worse_fit_direction)
    _add_figure3_threshold_label(ax, plot_threshold)


def _add_figure3_phase_labels(ax: Axes, params: DecodeParams) -> None:
    """Add staggered misfit labels above Figure 3 panel (a)."""
    bnd = params.phase_boundaries
    t_remap_start = bnd[PhaseBoundary.REMAP_START]
    t_remap_end = bnd[PhaseBoundary.REMAP_END]
    t_recovery1_end = bnd[PhaseBoundary.RECOVERY1_END]
    t_hist_dep_end = bnd[PhaseBoundary.HIST_DEP_END]
    t_recovery2_end = bnd[PhaseBoundary.RECOVERY2_END]
    t_drift_end = bnd[PhaseBoundary.DRIFT_END]
    t_recovery3_end = bnd[PhaseBoundary.RECOVERY3_END]
    t_wide_dynamics_end = bnd[PhaseBoundary.WIDE_DYNAMICS_END]

    phase_label_y = 1.04
    phase_labels_info: list[tuple[float, str]] = [
        ((t_remap_start + t_remap_end) / 2, "Remap"),
        ((t_recovery1_end + t_hist_dep_end) / 2, "History-dep."),
        ((t_recovery2_end + t_drift_end) / 2, "Drift"),
        ((t_recovery3_end + t_wide_dynamics_end) / 2, "Wide dyn. noise"),
    ]
    for x_pos, label_text in phase_labels_info:
        phase_label = ax.text(
            x_pos,
            phase_label_y,
            label_text,
            transform=ax.get_xaxis_transform(),
            fontsize=8,
            ha="center",
            va="bottom",
            style="italic",
        )
        phase_label.set_gid(FIGURE3_PHASE_LABEL_GID)


def _plot_figure3_summary_heatmap(
    ax: Axes,
    metrics: Diagnostics,
    thresholds: Thresholds,
    params: DecodeParams,
    summary_median: NDArray[np.floating] | None,
) -> None:
    """Plot Figure 3 panel (b), the phase-by-metric flagged fraction heatmap."""
    windows = summary_phase_windows(params)
    component_labels = [col.component for col in windows]

    if summary_median is not None:
        frac_data = np.asarray(summary_median, dtype=float)
    else:
        frac_data = compute_phase_flag_fractions(metrics, thresholds, windows)

    max_frac = np.nanmax(frac_data)
    norm_frac = frac_data / max_frac if max_frac > 0 else frac_data
    ax.imshow(norm_frac, cmap="YlOrRd", aspect="auto", vmin=0, vmax=1)

    metric_labels = ["HPD\noverlap", "KL\ndiv.", "−log(p)"]
    ax.set_yticks(range(3))
    ax.set_yticklabels(metric_labels, fontsize=8)

    ax.set_xticks(range(len(windows)))
    ax.set_xticklabels([col.label for col in windows], fontsize=8)
    ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)

    for row_idx in range(3):
        for col_idx in range(len(windows)):
            val = frac_data[row_idx, col_idx]
            color = "white" if norm_frac[row_idx, col_idx] > 0.55 else "black"
            weight = "bold" if norm_frac[row_idx, col_idx] > 0.7 else "normal"
            cell_label = ax.text(
                col_idx,
                row_idx,
                f"{val:.0f}%",
                ha="center",
                va="center",
                fontsize=8,
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
            fontsize=8,
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
        fontsize=8,
        color="0.4",
        fontstyle="italic",
    )
    known_component_label.set_gid(FIGURE3_SUMMARY_KNOWN_COMPONENT_LABEL_GID)

    summary_title = "% of spike events flagged as poor fit"
    if summary_median is not None:
        summary_title += " (median across realizations)"
    title = ax.set_title(
        summary_title,
        fontsize=8,
        pad=8,
        loc="center",
    )
    title.set_gid(FIGURE3_SUMMARY_TITLE_GID)


def plot_combined_diagnostics(
    x_true: NDArray[np.floating],
    spikes: NDArray[np.floating],
    metrics: Diagnostics,
    thresholds: Thresholds,
    params: DecodeParams,
    placefield_centers: NDArray[np.floating],
    summary_median: NDArray[np.floating] | None = None,
) -> Figure:
    """Create comprehensive time-series diagnostics figure.

    Layout: 6 time-series panels (predictive, likelihood, raster, HPDO, KL,
    spike prob) with shared x-axis and phase boundary overlays.

    Parameters
    ----------
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
    summary_median : NDArray, shape (3, n_columns), optional
        Pre-computed across-realization median percent flagged for the
        panel-(b) heatmap (rows follow
        :data:`statespacecheck_paper.analysis.SUMMARY_FLAG_METRICS`,
        columns follow :func:`statespacecheck_paper.analysis.summary_phase_windows`).
        When ``None``, the heatmap is computed from the single ``metrics``
        realization shown in panel (a) instead.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Time-series diagnostic figure.

    Examples
    --------
    >>> from statespacecheck_paper.analysis import (
    ...     DecodeParams, Diagnostics, Thresholds, decode_and_diagnostics,
    ... )
    >>> # See tests/test_plotting.py for a worked Diagnostics fixture
    >>> # and how to plumb it into plot_combined_diagnostics.
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

    n_time = metrics.posterior.shape[0]
    ax_pred = fig.add_subplot(gs[0])
    ax_like = fig.add_subplot(gs[1], sharex=ax_pred)
    ax_raster = fig.add_subplot(gs[2], sharex=ax_pred)
    diagnostic_axes = [fig.add_subplot(gs[i], sharex=ax_pred) for i in range(3, 6)]

    _plot_figure3_predictive_row(ax_pred, metrics.predictive, x_true)
    _plot_figure3_likelihood_row(ax_like, metrics, x_true)
    _plot_figure3_raster_row(ax_raster, spikes, placefield_centers)

    event_time_ind = metrics.event_time_ind
    for row_idx, (ax, spec) in enumerate(
        zip(diagnostic_axes, FIGURE3_DIAGNOSTIC_ROW_SPECS, strict=True)
    ):
        _plot_figure3_diagnostic_row(
            ax,
            event_time_ind,
            getattr(metrics, spec.event_attr),
            getattr(thresholds, spec.threshold_attr),
            spec,
            n_time=n_time,
            show_xlabel=row_idx == len(FIGURE3_DIAGNOSTIC_ROW_SPECS) - 1,
        )

    time_series_axes = [ax_pred, ax_like, ax_raster, *diagnostic_axes]
    add_phase_boundaries(
        time_series_axes,
        tuple(params.phase_boundaries),
        alpha=0.15,
    )
    _add_figure3_phase_labels(ax_pred, params)
    _add_figure3_panel_label(ax_pred, "a", y=1.15)

    # ===== SUMMARY HEATMAP: % exceeding baseline threshold per phase =====
    ax_summary = fig.add_subplot(gs_summary)
    _add_figure3_panel_label(ax_summary, "b", y=1.25)
    _plot_figure3_summary_heatmap(
        ax_summary,
        metrics,
        thresholds,
        params,
        summary_median,
    )

    return fig
