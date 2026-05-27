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

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.image import AxesImage
from matplotlib.lines import Line2D
from numpy.typing import NDArray

from statespacecheck_paper.analysis import (
    DecodeParams,
    Diagnostics,
    PhaseBoundary,
    Thresholds,
    get_remapped_pf_centers,
    likelihood_grid_for_counts,
)
from statespacecheck_paper.simulation import (
    gaussian_transition_matrix,
    normalize,
    softmax_with_shift,
)
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

        prior = normalize(prev_post @ transition_matrix)

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

        hpdo_val = np.nanmean(metrics.hpd_overlap[example_time])
        kl_val = np.nanmean(metrics.kl_divergence[example_time])
        spike_prob_val = np.nanmean(metrics.spike_prob[example_time])

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
    ax.set_ylabel("Neuron", fontsize=7, labelpad=7)


def plot_combined_diagnostics(
    x_true: NDArray[np.floating],
    spikes: NDArray[np.floating],
    metrics: Diagnostics,
    thresholds: Thresholds,
    params: DecodeParams,
    placefield_centers: NDArray[np.floating],
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
    # Calculate figure size
    fig_width = 7.0  # Full page width
    fig_height = 8.8

    fig = plt.figure(figsize=(fig_width, fig_height), dpi=450)

    # Outer grid: time-series block on top, summary heatmap on bottom.
    gs_outer = gridspec.GridSpec(
        2,
        1,
        figure=fig,
        height_ratios=[5.3, 1.2],
        hspace=0.35,
        left=0.08,
        right=0.93,
        top=0.97,
        bottom=0.06,
    )

    # Inner grid for time-series panels (6 rows)
    gs = gs_outer[0].subgridspec(
        6,
        1,
        height_ratios=[1.2, 1.2, 0.8, 0.7, 0.7, 0.7],
        hspace=0.12,
    )

    gs_summary = gs_outer[1]

    # ===== TOP SECTION: Time-Series Diagnostics =====

    n_time = metrics.posterior.shape[0]

    # Create time-series axes with shared x-axis
    # Order: Predictive -> Likelihood -> Raster -> HPD -> KL -> Spike
    ax_pred = fig.add_subplot(gs[0])
    ax_like = fig.add_subplot(gs[1], sharex=ax_pred)
    ax_raster = fig.add_subplot(gs[2], sharex=ax_pred)
    ax_hpdo = fig.add_subplot(gs[3], sharex=ax_pred)
    ax_kl = fig.add_subplot(gs[4], sharex=ax_pred)
    ax_spike = fig.add_subplot(gs[5], sharex=ax_pred)

    # Predictive heatmap
    _plot_timeseries_heatmap(ax_pred, metrics.predictive, x_true)
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
        metrics.predictive,
        metrics.per_spike_likelihood,
        metrics.event_time_ind,
        x_true=x_true,
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

    # Use per-spike event arrays so repeated spikes in the same
    # (time, cell) bin show up as distinct diagnostic events.
    event_time_ind = metrics.event_time_ind
    hpd_time_ind, hpd_values = event_time_ind, metrics.event_hpd_overlap
    kl_time_ind, kl_values = event_time_ind, metrics.event_kl_divergence
    spike_prob_time_ind, spike_prob_values = event_time_ind, metrics.event_spike_prob

    # HPDO reframed as "HPD shortfall" = 1 - HPDO. Three stacked tricks:
    #   1. Reframing puts the worst-fit region (shortfall near 1) at the
    #      *top* of a naturally ascending y-axis, matching KL and
    #      -log10(p) below; readers see "0 at bottom = good, 1 at top =
    #      bad" instead of a confusing inverted axis.
    #   2. A mirror-sqrt y-scale (y -> 1 - sqrt(1 - y)) expands the upper
    #      end of [0, 1]: the [0.9, 1.0] band displays at ~32% of the
    #      panel height instead of 10%.
    #   3. ylim extends slightly past 1 so y=1 markers float below the
    #      top spine instead of being half-occluded by it.
    hpd_shortfall = 1.0 - hpd_values
    threshold_shortfall = 1.0 - thresholds.hpd_overlap

    def _mirror_sqrt(y: NDArray[np.floating]) -> NDArray[np.floating]:
        # 1 - sign(1-y) * sqrt(|1-y|) — sign-preserving so values just
        # past 1 (added padding) still transform cleanly.
        diff = 1.0 - y
        out: NDArray[np.floating] = 1.0 - np.sign(diff) * np.sqrt(np.abs(diff))
        return out

    def _mirror_sqrt_inv(y: NDArray[np.floating]) -> NDArray[np.floating]:
        diff = 1.0 - y
        out: NDArray[np.floating] = 1.0 - np.sign(diff) * diff**2
        return out

    ax_hpdo.scatter(
        hpd_time_ind,
        hpd_shortfall,
        s=0.8,
        alpha=0.6,
        c=COLORS["hpd_overlap"],
        rasterized=True,
    )
    ax_hpdo.axhline(
        threshold_shortfall, color=COLORS["threshold"], linewidth=1.2, alpha=0.7, zorder=10
    )
    ax_hpdo.set_yscale("function", functions=(_mirror_sqrt, _mirror_sqrt_inv))
    ax_hpdo.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax_hpdo.set_xlim(0, n_time)
    ax_hpdo.set_ylim(0.0, 1.005)  # padded past 1 so shortfall=1 floats below top spine
    ax_hpdo.set_ylabel("HPD Shortfall\n(1 − HPDO)", fontsize=7, labelpad=7)
    ax_hpdo.tick_params(labelsize=6, labelbottom=False)
    ax_hpdo.text(
        1.01, 0.5, "↑ Worse fit", transform=ax_hpdo.transAxes, fontsize=6, va="center", ha="left"
    )
    ax_hpdo.text(
        1.01,
        threshold_shortfall,
        "Threshold",
        transform=ax_hpdo.get_yaxis_transform(),
        fontsize=6,
        va="center",
        ha="left",
        color=COLORS["threshold"],
    )

    # KL Divergence
    ax_kl.scatter(
        kl_time_ind,
        kl_values,
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
    spike_prob_transformed = -np.log10(np.maximum(spike_prob_values, 1e-10))
    threshold_transformed = -np.log10(np.maximum(thresholds.spike_prob, 1e-10))
    ax_spike.scatter(
        spike_prob_time_ind,
        spike_prob_transformed,
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

    # Add phase boundaries to all time-series panels.
    tseries_boundaries = tuple(params.phase_boundaries)

    # Add phase boundaries with matching colors to all panels
    add_phase_boundaries(
        [ax_pred, ax_like, ax_raster, ax_hpdo, ax_kl, ax_spike],
        tseries_boundaries,
        alpha=0.15,
    )

    # Add phase labels above top panel. Stagger vertical positions so the
    # narrow wide-dynamics phase (~2k timesteps) doesn't crowd its
    # neighbours.
    bnd = params.phase_boundaries
    t_remap_start = bnd[PhaseBoundary.REMAP_START]
    t_remap_end = bnd[PhaseBoundary.REMAP_END]
    t_recovery1_end = bnd[PhaseBoundary.RECOVERY1_END]
    t_hist_dep_end = bnd[PhaseBoundary.HIST_DEP_END]
    t_recovery2_end = bnd[PhaseBoundary.RECOVERY2_END]
    t_drift_end = bnd[PhaseBoundary.DRIFT_END]
    t_recovery3_end = bnd[PhaseBoundary.RECOVERY3_END]
    t_wide_dynamics_end = bnd[PhaseBoundary.WIDE_DYNAMICS_END]

    phase_labels_info: list[tuple[float, str, float]] = [
        ((t_remap_start + t_remap_end) / 2, "Remap", 1.02),
        ((t_recovery1_end + t_hist_dep_end) / 2, "History-dep.", 1.07),
        ((t_recovery2_end + t_drift_end) / 2, "Drift", 1.02),
        ((t_recovery3_end + t_wide_dynamics_end) / 2, "Wide dyn. noise", 1.07),
    ]
    for x_pos, label_text, y_pos in phase_labels_info:
        ax_pred.text(
            x_pos,
            y_pos,
            label_text,
            transform=ax_pred.get_xaxis_transform(),
            fontsize=6,
            ha="center",
            va="bottom",
            style="italic",
        )

    # Panel label (a) for time-series
    ax_pred.text(
        -0.05,
        1.15,
        "a",
        fontsize=8,
        fontweight="bold",
        transform=ax_pred.transAxes,
        va="top",
        ha="right",
    )

    # ===== SUMMARY HEATMAP: % exceeding baseline threshold per phase =====
    ax_summary = fig.add_subplot(gs_summary)

    # Panel label (b) for summary
    ax_summary.text(
        -0.05,
        1.25,
        "b",
        fontsize=8,
        fontweight="bold",
        transform=ax_summary.transAxes,
        va="top",
        ha="right",
    )

    # Phase windows for summary computation. Each entry is
    # ``(label, [(t0, t1), ...])`` — a list of (start, end) slices so a
    # single column can aggregate multiple disjoint windows (used by the
    # "Well-specified" column, which concatenates the three recovery windows
    # for an out-of-sample false-positive rate against the matched misfit
    # columns).
    phase_windows: list[tuple[str, list[tuple[int, int]]]] = [
        (
            "Well-\nspecified",
            [
                (t_remap_end, t_recovery1_end),
                (t_hist_dep_end, t_recovery2_end),
                (t_drift_end, t_recovery3_end),
            ],
        ),
        ("Remap", [(t_remap_start, t_remap_end)]),
        ("History-\ndep.", [(t_recovery1_end, t_hist_dep_end)]),
        ("Drift", [(t_recovery2_end, t_drift_end)]),
        ("Wide dyn.\nnoise", [(t_recovery3_end, t_wide_dynamics_end)]),
    ]
    component_labels = [
        "—",  # Well-specified (no induced misfit)
        "Observation",  # Remap
        "Observation",  # History-dependent firing
        "Transition",  # Drift
        "Transition",  # Wide dynamics noise
    ]

    # Use the same thresholds as the time-series panels above
    hpd_thr = thresholds.hpd_overlap
    kl_thr = thresholds.kl_divergence
    sp_thr = thresholds.spike_prob

    # Compute fraction exceeding threshold per phase (non-NaN only).
    #
    # Floor-effect caveat for HPD overlap: ``hpd_thr`` is the 1st-percentile
    # of baseline HPDO. With our default place-field geometry, baseline HPDOs
    # are tightly concentrated near 1.0 and a substantial fraction of baseline
    # events achieve HPDO == 0.0 exactly (single-bin disjoint distributions).
    # That pushes the 1st-percentile threshold to 0.0, and ``valid <= 0.0``
    # then degenerates to ``valid == 0.0``. The row is still informative —
    # exact-zero overlap is genuine inconsistency — but the % is a count of
    # disjoint-HPD events rather than a graded "below baseline" rate.
    # Re-evaluate if PF geometry changes enough to shift the 1st percentile
    # above 0.
    frac_data = np.zeros((3, len(phase_windows)))
    for j, (_name, slices) in enumerate(phase_windows):
        for i, (metric_key, thr_val, direction) in enumerate(
            [
                ("hpd_overlap", hpd_thr, "below"),
                ("kl_divergence", kl_thr, "above"),
                ("spike_prob", sp_thr, "below"),
            ]
        ):
            full = getattr(metrics, metric_key)
            vals = np.concatenate([full[t0:t1] for t0, t1 in slices])
            valid = vals[~np.isnan(vals)]
            if len(valid) > 0:
                if direction == "below":
                    frac_data[i, j] = 100 * np.mean(valid <= thr_val)
                else:
                    frac_data[i, j] = 100 * np.mean(valid >= thr_val)

    # Normalize for color mapping
    max_frac = np.nanmax(frac_data)
    norm_frac = frac_data / max_frac if max_frac > 0 else frac_data

    # Plot heatmap
    ax_summary.imshow(norm_frac, cmap="YlOrRd", aspect="auto", vmin=0, vmax=1)

    # Metric labels
    metric_labels = ["HPD\nOverlap", "KL\nDivergence", r"$-\log_{10}(p)$"]
    ax_summary.set_yticks(range(3))
    ax_summary.set_yticklabels(metric_labels, fontsize=6)

    # Phase labels on top
    ax_summary.set_xticks(range(len(phase_windows)))
    ax_summary.set_xticklabels([name for name, _ in phase_windows], fontsize=6)
    ax_summary.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)

    # Value annotations inside cells
    for i in range(3):
        for j in range(len(phase_windows)):
            val = frac_data[i, j]
            color = "white" if norm_frac[i, j] > 0.55 else "black"
            weight = "bold" if norm_frac[i, j] > 0.7 else "normal"
            ax_summary.text(
                j,
                i,
                f"{val:.0f}%",
                ha="center",
                va="center",
                fontsize=6,
                color=color,
                fontweight=weight,
            )

    # Component attribution below heatmap
    component_color = {"Observation": "#E69F00", "Transition": "#0072B2"}
    for j, comp in enumerate(component_labels):
        color = component_color.get(comp, "0.4")
        ax_summary.text(
            j,
            3.5,
            comp,
            ha="center",
            va="center",
            fontsize=5.5,
            fontstyle="italic",
            color=color,
        )
    ax_summary.text(
        -1.4,
        3.5,
        "Known\ncomponent:",
        ha="center",
        va="center",
        fontsize=5.5,
        color="0.4",
        fontstyle="italic",
    )

    # Title
    ax_summary.set_title(
        "% of spike events exceeding baseline threshold",
        fontsize=7,
        pad=15,
        loc="center",
    )

    return fig
