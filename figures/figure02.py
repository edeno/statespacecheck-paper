"""Demonstration simulation for state space model diagnostics.

This script simulates a Bayesian decoder with periods of good and poor model fit,
then computes diagnostic metrics using the statespacecheck package.
"""

from __future__ import annotations

from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import statespacecheck as ssc
from numpy.typing import NDArray
from scipy.stats import poisson

from statespacecheck_paper.simulation import (
    gaussian_transition_matrix,
    normalize,
    placefield_rates,
    safe_log,
    simulate_spikes_flat_rate,
    simulate_spikes_position_tuned,
    simulate_walk,
    spike_prob_rank,
)
from statespacecheck_paper.style import WONG, save_figure

# -----------------------------
# Data containers
# -----------------------------


@dataclass
class DecodeParams:
    """Parameters for decoding simulation."""

    # Timeline with recovery periods between misfits:
    # 0-6k: Clean baseline
    # 6k-10k: Remapping misfit (4k)
    # 10k-14k: Clean recovery (4k)
    # 14k-16k: Flat firing misfit (2k)
    # 16k-20k: Clean recovery (4k)
    # 20k-24k: Fast movement misfit (4k)
    # 24k-28k: Clean recovery (4k)
    # 28k-32k: Slow movement misfit (4k)
    T_remap_start: int = 6_000
    T_remap_end: int = 10_000
    T_recovery1_end: int = 14_000
    T_flat_end: int = 16_000
    T_recovery2_end: int = 20_000
    T_fast_end: int = 24_000
    T_recovery3_end: int = 28_000
    T_slow_end: int = 32_000
    sigx_pred: float = 0.5  # decoder's dynamics std (baseline)
    sigx_pred_fast_phase: float = 0.1  # narrow decoder for fast phase (5x too narrow!)
    sigx_pred_slow_phase: float = 20.0  # inflated decoder for slow phase (40x too broad!)
    sigx_true_fast: float = 10.0  # true dynamics std in fast phase (100x faster than decoder!)
    sigx_true_slow: float = 0.0  # true dynamics std in slow phase (completely stationary!)
    xs_min: int = 0
    xs_max: int = 100
    xs_step: int = 1
    pf_width: float = 5.0  # Narrow place fields for sharp spatial selectivity
    pf_centers: NDArray[np.floating] | None = None  # set in __post_init__
    rate_scale: float = 0.15  # Higher spike rate to reduce uncertainty
    base_seed: int = 1
    remap_from_to: tuple[tuple[int, int], ...] | tuple[int, int] = (
        (0, 5),  # Position 0 → 50 (shift +50cm)
        (1, 6),  # Position 10 → 60
        (2, 7),  # Position 20 → 70
        (3, 8),  # Position 30 → 80
        (4, 9),  # Position 40 → 90
        (5, 10),  # Position 50 → 100
        (6, 0),  # Position 60 → 0
        (7, 1),  # Position 70 → 10
        (8, 2),  # Position 80 → 20
        (9, 3),  # Position 90 → 30
        (10, 4),  # Position 100 → 40
    )  # Remap ALL 11 cells with +50cm circular shift

    @property
    def remap_window(self) -> tuple[int, int]:
        """Remapping window for backward compatibility."""
        return (self.T_remap_start, self.T_remap_end)

    def __post_init__(self) -> None:
        """Initialize pf_centers if not provided."""
        if self.pf_centers is None:
            self.pf_centers = np.arange(self.xs_min, self.xs_max + 1, 10, dtype=float)


# -----------------------------
# Decoder step (vectorized across cells/bins)
# -----------------------------


def likelihood_grid_for_counts(
    xs: NDArray[np.floating],
    pf_centers: NDArray[np.floating],
    pf_width: float,
    rate_scale: float,
    counts: NDArray[np.int_],
) -> NDArray[np.floating]:
    """Compute likelihood grid for spike counts.

    L_grid[bin, cell] ∝ P(counts[cell] | position=xs[bin])
    Not normalized across bins; we normalize later per-cell.
    """
    lam = placefield_rates(xs, pf_centers, pf_width, rate_scale)  # (n_bins, n_cells)
    # Poisson PMF per bin, per cell for this time's counts
    # counts is (n_cells,), lam is (n_bins, n_cells)
    likelihood_grid = poisson.pmf(counts[None, :], lam)
    # Avoid degenerate zeros; normalize per cell (over bins) to a proper density on xs
    likelihood_grid = normalize(likelihood_grid, axis=0)
    return likelihood_grid


def apply_remap_for_likelihoods(
    likelihood: NDArray[np.floating],
    remap_from_to: tuple[tuple[int, int], ...] | tuple[int, int],
    active: bool,
) -> NDArray[np.floating]:
    """Optionally replace one or more columns by others (remapping cell identities)."""
    if not active:
        return likelihood
    likelihood = likelihood.copy()

    # Handle both single tuple and tuple of tuples
    if (
        isinstance(remap_from_to, tuple)
        and len(remap_from_to) == 2
        and isinstance(remap_from_to[0], int)
    ):
        # Single remapping: (src, dst)
        src, dst = remap_from_to
        likelihood[:, src] = likelihood[:, dst]
    else:
        # Multiple remappings: ((src1, dst1), (src2, dst2), ...)
        for src, dst in remap_from_to:
            likelihood[:, src] = likelihood[:, dst]

    return likelihood


def decode_and_diagnostics(
    spikes: NDArray[np.int_],
    xs: NDArray[np.floating],
    transition_matrix: NDArray[np.floating],
    pf_centers: NDArray[np.floating],
    pf_width: float,
    rate_scale: float,
    remap_window: tuple[int, int],
    remap_from_to: tuple[tuple[int, int], ...] | tuple[int, int],
    rng: np.random.Generator | None = None,
    transition_matrix_narrow: NDArray[np.floating] | None = None,
    narrow_window: tuple[int, int] | None = None,
    transition_matrix_inflated: NDArray[np.floating] | None = None,
    inflate_window: tuple[int, int] | None = None,
) -> dict[str, NDArray]:
    """Run the Bayesian filter with per-time, per-cell diagnostics.

    Returns dict with: post, HPDO, KL, spikeProb
    """
    n_time, n_cells = spikes.shape
    n_bins = xs.size

    post = np.zeros((n_time, n_bins), dtype=float)
    hpdo = np.full(n_time, np.nan, dtype=float)  # Single value per timestep
    kl = np.full(n_time, np.nan, dtype=float)  # Single value per timestep
    spike_prob = np.full((n_time, n_cells), np.nan, dtype=float)  # Keep per-cell for this metric

    # t=0 (MATLAB used a flat prior at t=1)
    post[0] = normalize(np.ones(n_bins))

    lam_grid_all = placefield_rates(xs, pf_centers, pf_width, rate_scale)  # (n_bins, n_cells)
    lambda_ratio = normalize(lam_grid_all, axis=1)  # per-bin cell-fractions, rows sum to 1

    start_r, end_r = remap_window
    start_narrow, end_narrow = narrow_window if narrow_window else (n_time + 1, n_time + 1)
    start_inflate, end_inflate = inflate_window if inflate_window else (n_time + 1, n_time + 1)

    for t in range(1, n_time):
        # Select transition matrix based on which window we're in
        if transition_matrix_narrow is not None and start_narrow <= t <= end_narrow:
            current_transition = transition_matrix_narrow
        elif transition_matrix_inflated is not None and start_inflate <= t <= end_inflate:
            current_transition = transition_matrix_inflated
        else:
            current_transition = transition_matrix

        # Predict (prior from state dynamics)
        prior = normalize(post[t - 1] @ current_transition)  # (n_bins,)

        # Likelihood grid for this time's counts (vectorized over bins & cells)
        likelihood = likelihood_grid_for_counts(xs, pf_centers, pf_width, rate_scale, spikes[t])
        # Optional remap (imitating MATLAB's j==10 uses field of j==1 in a window)
        active_remap = start_r <= t <= end_r
        likelihood = apply_remap_for_likelihoods(likelihood, remap_from_to, active_remap)

        # Compute combined likelihood from all cells (product over cells)
        combined_likelihood = np.prod(likelihood, axis=1)  # (n_bins,)

        # Compute diagnostics using statespacecheck functions
        # Compare one-step prediction (prior) with combined likelihood (observation model)
        prior_t = prior[np.newaxis, :]  # (1, n_bins)
        combined_likelihood_t = combined_likelihood[np.newaxis, :]  # (1, n_bins)

        # HPD overlap between prior and combined likelihood
        hpdo_t = ssc.hpd_overlap(prior_t, combined_likelihood_t, coverage=0.95)
        hpdo[t] = hpdo_t[0]

        # KL divergence between prior and combined likelihood
        kl_t = ssc.kl_divergence(prior_t, combined_likelihood_t)
        kl[t] = kl_t[0]

        # Posterior update
        post[t] = normalize(prior * combined_likelihood)

        # spike_prob: cumulative probability mass for cells with low expected contribution
        spike_prob[t] = spike_prob_rank(prior, lambda_ratio)

    # Mask spike_prob for cells with zero spikes (match MATLAB: spikeProb(spikes == 0) = nan)
    # Note: HPDO and KL are now per-timestep (not per-cell) since they compare
    # the combined likelihood with the prior, so we don't mask them
    spike_prob[spikes == 0] = np.nan

    return {"post": post, "HPDO": hpdo, "KL": kl, "spikeProb": spike_prob}


# -----------------------------
# Thresholds & transforms
# -----------------------------


@dataclass
class Thresholds:
    """Threshold values for diagnostic metrics."""

    HPDO: float
    KL: float
    spike_prob: float


def compute_thresholds(metrics: dict[str, NDArray], baseline_end: int = 60_000) -> Thresholds:
    """Compute threshold values from baseline period."""
    hpdo_thresh = np.nanquantile(metrics["HPDO"][:baseline_end], 0.01)
    kl_thresh = np.nanquantile(metrics["KL"][:baseline_end], 0.99)
    # MATLAB uses 0.05 as fixed threshold (raw count, not normalized)
    spike_prob_thresh = 0.05
    return Thresholds(HPDO=hpdo_thresh, KL=kl_thresh, spike_prob=spike_prob_thresh)


@dataclass
class Transformed:
    """Transformed diagnostic metrics and thresholds."""

    HPDO: NDArray[np.floating]
    KL: NDArray[np.floating]
    spike_prob: NDArray[np.floating]
    HPDO_th: float
    KL_th: float
    spike_prob_th: float


def transform_metrics(
    metrics: dict[str, NDArray], th: Thresholds, eps1: float = 1e-2, eps2: float = 1e-10
) -> Transformed:
    """Apply transformations to metrics for better visualization."""
    hpdo_transformed = -safe_log(metrics["HPDO"] + eps1)
    kl_transformed = np.sqrt(metrics["KL"])
    spike_prob_transformed = -safe_log(metrics["spikeProb"] + eps2)

    return Transformed(
        HPDO=hpdo_transformed,
        KL=kl_transformed,
        spike_prob=spike_prob_transformed,
        HPDO_th=-np.log(th.HPDO + eps1),
        KL_th=np.sqrt(th.KL),
        spike_prob_th=-np.log(th.spike_prob + eps2),
    )


# -----------------------------
# Plotting
# -----------------------------


def plot_original(
    xs: NDArray,
    x_true: NDArray,
    metrics: dict[str, NDArray],
    th: Thresholds,
    title: str = "Original Metrics",
    remap_window: tuple[int, int] | None = None,
    phase_boundaries: tuple[int, ...] | None = None,
) -> plt.Figure:
    """Plot original diagnostic metrics with thresholds.

    Parameters
    ----------
    remap_window : tuple[int, int] | None
        Time window where cell remapping occurs (start, end)
    phase_boundaries : tuple[int, ...] | None
        Boundaries between phases: (T_remap_start, T_remap_end, T_recovery1_end,
        T_flat_end, T_recovery2_end, T_fast_end, T_recovery3_end, T_slow_end)
    """
    n_time = metrics["post"].shape[0]
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
        metrics["post"].T,
        aspect="auto",
        origin="lower",
        vmin=0.0,
        vmax=np.quantile(metrics["post"], 0.975),
        cmap="bone_r",
    )
    # Plot true position in magenta for visibility against bone_r colormap
    axes[0].plot(
        np.arange(n_time), x_true, color="magenta", linewidth=1.5, alpha=0.85, label="True position"
    )
    axes[0].set_ylabel("Position (bin)", fontsize=10, labelpad=8)
    axes[0].tick_params(labelsize=8)

    # Create colorbar with better formatting
    cbar = fig.colorbar(im, ax=axes[0], fraction=0.03, pad=0.02, aspect=30)
    cbar.set_label("Probability (×10⁻¹²)", fontsize=9, labelpad=8)
    cbar.ax.tick_params(labelsize=8, length=3, width=0.5)
    # Scale tick labels by 1e12 to avoid offset text
    import matplotlib.ticker as ticker

    cbar.ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f"{x * 1e12:.1f}"))

    # Add phase boundaries to all axes, but only add labels to first axis for legend
    for i, ax in enumerate(axes):
        # Highlight phase boundaries with different colors for misfit vs recovery
        if phase_boundaries is not None and len(phase_boundaries) == 8:
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

            # Only add labels for the first axis (for legend)
            add_labels = i == 0

            # Misfit periods (colored)
            ax.axvspan(
                t_remap_start,
                t_remap_end,
                alpha=0.2,
                color="orange",
                label="Remapping" if add_labels else "",
            )
            ax.axvspan(
                t_recovery1_end,
                t_flat_end,
                alpha=0.2,
                color="gray",
                label="Flat firing" if add_labels else "",
            )
            ax.axvspan(
                t_recovery2_end,
                t_fast_end,
                alpha=0.2,
                color="red",
                label="Fast movement" if add_labels else "",
            )
            ax.axvspan(
                t_recovery3_end,
                t_slow_end,
                alpha=0.2,
                color="blue",
                label="Stationary" if add_labels else "",
            )

    axes[1].plot(
        metrics["HPDO"],
        ".",
        markersize=1.5,
        alpha=0.6,
        color="#56B4E9",
        rasterized=True,
    )
    axes[1].axhline(th.HPDO, color="#E69F00", linewidth=1.5, zorder=10)
    axes[1].set_xlim(0, n_time)
    axes[1].set_ylabel("HPD Overlap", fontsize=10, labelpad=8)
    axes[1].tick_params(labelsize=8)

    axes[2].plot(metrics["KL"], ".", markersize=1.5, alpha=0.6, color="#56B4E9", rasterized=True)
    axes[2].axhline(th.KL, color="#E69F00", linewidth=1.5, zorder=10)
    axes[2].set_xlim(0, n_time)
    axes[2].set_ylabel("KL Divergence", fontsize=10, labelpad=8)
    axes[2].tick_params(labelsize=8)

    # Transform spike probability to -log scale
    eps2 = 1e-12
    spike_prob_transformed = -safe_log(metrics["spikeProb"] + eps2)
    spike_prob_thresh_transformed = -np.log(th.spike_prob + eps2)

    axes[3].plot(
        spike_prob_transformed,
        ".",
        markersize=1.5,
        alpha=0.6,
        color="#56B4E9",
        rasterized=True,
    )
    axes[3].axhline(spike_prob_thresh_transformed, color="#E69F00", linewidth=1.5, zorder=10)
    axes[3].set_xlim(0, n_time)
    axes[3].set_ylabel("-log(Spike Prob)", fontsize=10, labelpad=8)
    axes[3].set_xlabel("Time", fontsize=10, labelpad=8)
    axes[3].tick_params(labelsize=8)

    # Add comprehensive legend outside the plot area at the bottom
    # Get handles and labels from axes[0] where they were defined
    handles, labels = axes[0].get_legend_handles_labels()
    axes[3].legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.35),
        fontsize=9,
        frameon=True,
        fancybox=False,
        shadow=False,
        ncol=5,
    )

    fig.suptitle(title, fontsize=11, y=0.99)
    return fig


def plot_transformed(
    xs: NDArray,
    x_true: NDArray,
    post: NDArray,
    tr: Transformed,
    title: str = "Transformed Metrics (-log, sqrt)",
    remap_window: tuple[int, int] | None = None,
    phase_boundaries: tuple[int, int] | None = None,
) -> plt.Figure:
    """Plot transformed diagnostic metrics with thresholds.

    Parameters
    ----------
    remap_window : tuple[int, int] | None
        Time window where cell remapping occurs (start, end)
    phase_boundaries : tuple[int, int] | None
        Boundaries between phases: (T1, T2) where T3 is end of data
    """
    n_time = post.shape[0]
    fig, axes = plt.subplots(4, 1, figsize=(7, 6), constrained_layout=True, sharex=True, dpi=150)

    im = axes[0].imshow(post.T, aspect="auto", origin="lower", cmap="viridis")
    axes[0].plot(np.arange(n_time), x_true, "k", linewidth=1.0, alpha=0.8)
    axes[0].set_ylabel("Position (bin)", fontsize=9, labelpad=8)
    axes[0].tick_params(labelsize=7)
    cbar = fig.colorbar(im, ax=axes[0], fraction=0.02, pad=0.02)
    cbar.set_label("Probability", fontsize=8, labelpad=8)
    cbar.ax.tick_params(labelsize=7)

    for ax in axes:
        # Highlight remap window (cell 10->1)
        if remap_window is not None:
            ax.axvspan(
                remap_window[0],
                remap_window[1],
                alpha=0.15,
                color="orange",
                label="Remap",
            )

        # Highlight phase boundaries
        if phase_boundaries is not None:
            t1, t2 = phase_boundaries
            ax.axvspan(t1, t2, alpha=0.15, color="gray", label="Flat rate")
            ax.axvspan(t2, n_time, alpha=0.15, color="red", label="Fast movement")

    axes[1].plot(tr.HPDO, ".", markersize=0.5, alpha=0.3, rasterized=True)
    axes[1].axhline(tr.HPDO_th, color="#E69F00", linewidth=1.5, label="Threshold", zorder=10)
    axes[1].set_xlim(0, n_time)
    axes[1].set_ylabel("-log(HPD Overlap)", fontsize=9, labelpad=8)
    axes[1].tick_params(labelsize=7)
    axes[1].legend(loc="upper right", fontsize=7, frameon=False)

    axes[2].plot(tr.KL, ".", markersize=0.5, alpha=0.3, rasterized=True)
    axes[2].axhline(tr.KL_th, color="#E69F00", linewidth=1.5, label="Threshold", zorder=10)
    axes[2].set_xlim(0, n_time)
    axes[2].set_ylabel("sqrt(KL Divergence)", fontsize=9, labelpad=8)
    axes[2].tick_params(labelsize=7)

    axes[3].plot(tr.spike_prob, ".", markersize=0.5, alpha=0.3, rasterized=True)
    axes[3].axhline(tr.spike_prob_th, color="#E69F00", linewidth=1.5, label="Threshold", zorder=10)
    axes[3].set_xlim(0, n_time)
    axes[3].set_ylabel("-log(Spike Prob)", fontsize=9, labelpad=8)
    axes[3].set_xlabel("Time", fontsize=9, labelpad=8)
    axes[3].tick_params(labelsize=7)

    fig.suptitle(title, fontsize=10, y=0.998)
    return fig


def plot_misfit_examples(
    xs: NDArray,
    x_true: NDArray,
    spikes: NDArray,
    metrics: dict[str, NDArray],
    params: DecodeParams,
    pf_centers: NDArray[np.floating],
    pf_width: float,
    rate_scale: float,
) -> None:
    """Plot examples of high misfit moments for each scenario.

    Finds the worst time point in each misfit phase and shows the distributions.
    Also includes a baseline example with good fit.
    Shows 5 columns: baseline + 4 misfit types.
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

    # Wong colorblind-friendly palette
    wong = WONG

    for phase_idx, (phase_name, phase_slice, is_baseline) in enumerate(phases):
        # For baseline, find best fit (highest HPDO); for misfits, find worst fit (lowest HPDO)
        # BUT: only consider time points with spikes so likelihood is informative
        phase_hpdo = metrics["HPDO"][phase_slice]
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
            prev_post = metrics["post"][example_time - 1]
        else:
            prev_post = np.ones_like(xs) / len(xs)

        # Select appropriate transition matrix
        if params.T_recovery2_end <= example_time <= params.T_fast_end:
            transition_matrix = gaussian_transition_matrix(xs, params.sigx_pred_fast_phase)
        elif params.T_recovery3_end <= example_time <= params.T_slow_end:
            transition_matrix = gaussian_transition_matrix(xs, params.sigx_pred_slow_phase)
        else:
            transition_matrix = gaussian_transition_matrix(xs, params.sigx_pred)

        # Compute prior
        prior = normalize(prev_post @ transition_matrix)

        # Compute combined likelihood
        likelihood = likelihood_grid_for_counts(
            xs, pf_centers, pf_width, rate_scale, spikes[example_time]
        )

        # Apply remapping if in remap window
        if params.T_remap_start <= example_time <= params.T_remap_end:
            likelihood = apply_remap_for_likelihoods(likelihood, params.remap_from_to, active=True)

        combined_likelihood = normalize(np.prod(likelihood, axis=1))

        # Plot prior and likelihood with twin axes - use Wong colorblind-friendly palette
        ax1 = axes[phase_idx]
        ax2 = ax1.twinx()

        # Plot prior on left axis (blue from Wong palette) with transparency
        line1 = ax1.plot(xs, prior, color=wong[5], linewidth=1.5, alpha=0.7, label="Predictive")
        # Determine scale factor for prior and include in ylabel
        prior_max = np.max(prior)
        if prior_max > 0:
            prior_order = int(np.floor(np.log10(prior_max)))
            # Use scale factor if magnitude is outside reasonable range
            if prior_order < -2 or prior_order > 2:
                prior_scale = 10**prior_order
                ax1.plot(xs, prior / prior_scale, color=wong[5], linewidth=1.5, alpha=0.7)
                ax1.lines[0].remove()  # Remove the unscaled plot
                ax1.set_ylabel(
                    f"Predictive (×10$^{{{prior_order}}}$)", fontsize=7, color=wong[5], labelpad=3
                )
            else:
                ax1.set_ylabel("Predictive", fontsize=7, color=wong[5], labelpad=3)
        else:
            ax1.set_ylabel("Predictive", fontsize=7, color=wong[5], labelpad=3)
        ax1.tick_params(axis="y", labelcolor=wong[5], labelsize=6)
        ax1.set_ylim(0, None)

        # Plot likelihood on right axis (orange from Wong palette) - solid line
        likelihood_max = np.max(combined_likelihood)
        if likelihood_max > 0:
            likelihood_order = int(np.floor(np.log10(likelihood_max)))
            # Use scale factor if magnitude is outside reasonable range
            if likelihood_order < -2 or likelihood_order > 2:
                likelihood_scale = 10**likelihood_order
                line2 = ax2.plot(
                    xs,
                    combined_likelihood / likelihood_scale,
                    color=wong[1],
                    linewidth=1.5,
                    alpha=0.9,
                    label="Likelihood",
                )
                ax2.set_ylabel(
                    f"Likelihood (×10$^{{{likelihood_order}}}$)",
                    fontsize=7,
                    color=wong[1],
                    labelpad=3,
                )
            else:
                line2 = ax2.plot(
                    xs,
                    combined_likelihood,
                    color=wong[1],
                    linewidth=1.5,
                    alpha=0.9,
                    label="Likelihood",
                )
                ax2.set_ylabel("Likelihood", fontsize=7, color=wong[1], labelpad=3)
        else:
            line2 = ax2.plot(
                xs, combined_likelihood, color=wong[1], linewidth=1.5, alpha=0.9, label="Likelihood"
            )
            ax2.set_ylabel("Likelihood", fontsize=7, color=wong[1], labelpad=3)
        ax2.tick_params(axis="y", labelcolor=wong[1], labelsize=6)
        ax2.set_ylim(0, None)

        # Add true position line (purple from Wong palette)
        ax1.axvline(x_true[example_time], color=wong[7], linestyle="--", linewidth=1.0, alpha=0.7)

        # Get diagnostic values
        hpdo_val = metrics["HPDO"][example_time]
        kl_val = metrics["KL"][example_time]
        spike_prob_vals = metrics["spikeProb"][example_time]

        # Calculate -log(min spike prob) with only significant digits
        if not np.all(np.isnan(spike_prob_vals)):
            spike_prob_min = np.nanmin(spike_prob_vals)
            log_spike_prob = -np.log(spike_prob_min + 1e-12)
        else:
            log_spike_prob = np.nan

        # Add phase name and metrics as title (always use engineering format for -log)
        title_text = (
            f"{phase_name}\nHPD: {hpdo_val:.2g}  KL: {kl_val:.2g}  -log: {log_spike_prob:.2e}"
        )
        if np.isnan(log_spike_prob):
            title_text = f"{phase_name}\nHPD: {hpdo_val:.2g}  KL: {kl_val:.2g}  -log: N/A"
        ax1.set_title(title_text, fontsize=7, pad=5, fontweight="bold")

        ax1.tick_params(axis="x", labelsize=6)
        ax1.set_xlabel("Position", fontsize=7, labelpad=3)

        # Add legend to first panel only
        if phase_idx == 0:
            lines = line1 + line2
            labels = [str(line.get_label()) for line in lines]
            ax1.legend(lines, labels, fontsize=5, loc="lower right", frameon=False)

    # Save to scripts directory (publication quality: both PDF and PNG)
    import os

    save_path_base = os.path.join(os.path.dirname(__file__), "misfit_examples")

    # Save PDF (vector) for publication
    plt.savefig(f"{save_path_base}.pdf", dpi=450, bbox_inches="tight")
    # Save PNG (raster) for quick viewing
    plt.savefig(f"{save_path_base}.png", dpi=450, bbox_inches="tight")
    plt.close()
    print(f"Misfit examples figure saved to {save_path_base}.{{pdf,png}}")


def plot_combined_diagnostics(
    xs: NDArray,
    x_true: NDArray,
    spikes: NDArray,
    metrics: dict[str, NDArray],
    th: Thresholds,
    params: DecodeParams,
    pf_centers: NDArray[np.floating],
    pf_width: float,
    rate_scale: float,
) -> None:
    """Create comprehensive combined figure with misfit examples and time-series diagnostics.

    Layout:
    - Top section: 4 time-series panels (posterior, HPDO, KL, spike prob) with shared x-axis
    - Bottom section: 5 distribution examples (baseline + 4 misfits)
    - Background colors on examples match phase colors in time-series
    - All formatting matches original figure standards
    """
    import matplotlib.gridspec as gridspec

    # Wong colorblind-friendly palette
    wong = WONG

    # Phase colors (lighter versions for backgrounds)
    phase_colors = {
        "baseline": "#FFFFFF",  # White
        "remap": "#FFE5CC",  # Light orange
        "flat": "#E8E8E8",  # Light gray
        "fast": "#FFD6D6",  # Light red
        "slow": "#D6E5FF",  # Light blue
    }

    # Calculate figure size
    fig_width = 7.0  # Full page width
    fig_height = 7.5  # Compact height appropriate for manuscripts

    fig = plt.figure(figsize=(fig_width, fig_height), dpi=450)

    # Create grid: 4 rows for diagnostics, gap, 2 rows for examples
    # 6 columns: 5 for plots + 1 for colorbar/legend
    gs = gridspec.GridSpec(
        7,
        6,
        figure=fig,
        height_ratios=[1.5, 0.8, 0.8, 0.8, 0.5, 0.6, 0.6],  # Gap row increased to 0.5
        width_ratios=[1, 1, 1, 1, 1, 0.10],  # Narrow column for colorbar/legend/annotations
        hspace=0.12,
        wspace=0.15,  # Minimal spacing between columns
        left=0.08,
        right=0.99,
        top=0.97,
        bottom=0.05,
    )

    # ===== TOP SECTION: Time-Series Diagnostics =====

    n_time = metrics["post"].shape[0]

    # Create time-series axes (all spanning first 5 columns, with shared x-axis)
    ax_post = fig.add_subplot(gs[0, 0:5])
    ax_hpdo = fig.add_subplot(gs[1, 0:5], sharex=ax_post)
    ax_kl = fig.add_subplot(gs[2, 0:5], sharex=ax_post)
    ax_spike = fig.add_subplot(gs[3, 0:5], sharex=ax_post)

    # Posterior heatmap
    im = ax_post.imshow(
        metrics["post"].T,
        aspect="auto",
        origin="lower",
        vmin=0.0,
        vmax=np.quantile(metrics["post"], 0.975),
        cmap="bone_r",
    )
    ax_post.plot(
        np.arange(n_time),
        x_true,
        color="magenta",
        linewidth=1.0,
        alpha=0.85,
        label="True position",
    )
    ax_post.set_ylabel("Position (a.u.)", fontsize=9, labelpad=7)
    ax_post.tick_params(labelsize=7, labelbottom=False)
    # Add legend for true position line (upper left)
    ax_post.legend(loc="upper left", fontsize=6, frameon=False)

    # HPDO
    ax_hpdo.plot(metrics["HPDO"], ".", markersize=0.8, alpha=0.6, color=wong[5], rasterized=True)
    ax_hpdo.axhline(th.HPDO, color="#666666", linewidth=1.2, alpha=0.7, zorder=10)
    ax_hpdo.set_xlim(0, n_time)
    ax_hpdo.set_ylabel("HPD Overlap", fontsize=9, labelpad=7)
    ax_hpdo.tick_params(labelsize=7, labelbottom=False)
    # Add directional indicator and threshold annotation
    ax_hpdo.text(
        1.01, 0.5, "↓ Worse fit", transform=ax_hpdo.transAxes, fontsize=6, va="center", ha="left"
    )
    ax_hpdo.text(
        1.01,
        th.HPDO,
        "Threshold",
        transform=ax_hpdo.get_yaxis_transform(),
        fontsize=6,
        va="center",
        ha="left",
        color="#666666",
    )

    # KL Divergence
    ax_kl.plot(metrics["KL"], ".", markersize=0.8, alpha=0.6, color=wong[5], rasterized=True)
    ax_kl.axhline(th.KL, color="#666666", linewidth=1.2, alpha=0.7, zorder=10)
    ax_kl.set_xlim(0, n_time)
    ax_kl.set_ylabel("KL Divergence", fontsize=9, labelpad=7)
    ax_kl.tick_params(labelsize=7, labelbottom=False)
    # Add directional indicator and threshold annotation
    ax_kl.text(
        1.01, 0.5, "↑ Worse fit", transform=ax_kl.transAxes, fontsize=6, va="center", ha="left"
    )
    ax_kl.text(
        1.01,
        th.KL,
        "Threshold",
        transform=ax_kl.get_yaxis_transform(),
        fontsize=6,
        va="center",
        ha="left",
        color="#666666",
    )

    # Spike Probability (transformed)
    eps2 = 1e-12
    spike_prob_transformed = -safe_log(metrics["spikeProb"] + eps2)
    spike_prob_thresh_transformed = -np.log(th.spike_prob + eps2)

    ax_spike.plot(
        spike_prob_transformed,
        ".",
        markersize=0.8,
        alpha=0.6,
        color=wong[5],
        rasterized=True,
    )
    ax_spike.axhline(
        spike_prob_thresh_transformed,
        color="#666666",
        linewidth=1.2,
        alpha=0.7,
        zorder=10,
    )
    ax_spike.set_xlim(0, n_time)
    ax_spike.set_ylabel("-log(p-value)", fontsize=9, labelpad=7)
    ax_spike.set_xlabel("Time (a.u.)", fontsize=9, labelpad=7)
    ax_spike.tick_params(labelsize=7)
    # Add directional indicator and threshold annotation
    ax_spike.text(
        1.01, 0.5, "↑ Worse fit", transform=ax_spike.transAxes, fontsize=6, va="center", ha="left"
    )
    ax_spike.text(
        1.01,
        spike_prob_thresh_transformed,
        "Threshold",
        transform=ax_spike.get_yaxis_transform(),
        fontsize=6,
        va="center",
        ha="left",
        color="#666666",
    )

    # Colorbar for posterior only - in dedicated axes aligned with posterior panel
    cax = fig.add_subplot(gs[0, 5])
    cbar = fig.colorbar(im, cax=cax)
    # Use clearer formatting: show values in scientific notation
    cbar.ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{x:.1e}" if x > 0 else "0"))
    cbar.set_label("Probability", fontsize=8, labelpad=6)
    cbar.ax.tick_params(labelsize=6, length=2, width=0.5)

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

    for ax in [ax_post, ax_hpdo, ax_kl, ax_spike]:
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

        # Misfit periods with matching colors
        ax.axvspan(t_remap_start, t_remap_end, alpha=0.15, color="orange")
        ax.axvspan(t_recovery1_end, t_flat_end, alpha=0.15, color="gray")
        ax.axvspan(t_recovery2_end, t_fast_end, alpha=0.15, color="red")
        ax.axvspan(t_recovery3_end, t_slow_end, alpha=0.15, color="blue")

    # Add phase labels above top panel
    phase_labels_info = [
        ((t_remap_start + t_remap_end) / 2, "Remap", "c"),
        ((t_recovery1_end + t_flat_end) / 2, "Flat", "d"),
        ((t_recovery2_end + t_fast_end) / 2, "Fast", "e"),
        ((t_recovery3_end + t_slow_end) / 2, "Slow", "f"),
    ]
    for x_pos, label_text, panel_id in phase_labels_info:
        ax_post.text(
            x_pos,
            1.02,
            f"{label_text} ({panel_id})",
            transform=ax_post.get_xaxis_transform(),
            fontsize=6,
            ha="center",
            va="bottom",
            style="italic",
        )

    # ===== BOTTOM SECTION: Misfit Examples =====

    # Define phases
    baseline_window = slice(1000, params.T_remap_start - 1000)
    remap_window = slice(params.T_remap_start, params.T_remap_end)
    flat_window = slice(params.T_recovery1_end, params.T_flat_end)
    fast_window = slice(params.T_recovery2_end, params.T_fast_end)
    slow_window = slice(params.T_recovery3_end, params.T_slow_end)

    phases = [
        ("Baseline", baseline_window, True, 0, "baseline"),
        ("Remapping", remap_window, False, 1, "remap"),
        ("Flat Firing", flat_window, False, 2, "flat"),
        ("Fast Movement", fast_window, False, 3, "fast"),
        ("Slow Movement", slow_window, False, 4, "slow"),
    ]

    # First pass: compute all distributions to determine shared y-limits
    plot_data = []
    for _phase_idx, (phase_name, phase_slice, is_baseline, col_idx, color_key) in enumerate(phases):
        # Find example time (best for baseline, worst for misfits)
        phase_hpdo = metrics["HPDO"][phase_slice]
        phase_spikes = spikes[phase_slice]
        has_spikes = phase_spikes.sum(axis=1) > 0
        valid_hpdo = phase_hpdo.copy()
        valid_hpdo[~has_spikes] = np.nan

        if is_baseline:
            example_idx_in_phase = np.nanargmax(valid_hpdo)
        else:
            example_idx_in_phase = np.nanargmin(valid_hpdo)
        example_time = phase_slice.start + example_idx_in_phase

        # Recompute distributions at example time
        if example_time > 0:
            prev_post = metrics["post"][example_time - 1]
        else:
            prev_post = np.ones_like(xs) / len(xs)

        # Select appropriate transition matrix
        if params.T_recovery2_end <= example_time <= params.T_fast_end:
            transition_matrix = gaussian_transition_matrix(xs, params.sigx_pred_fast_phase)
        elif params.T_recovery3_end <= example_time <= params.T_slow_end:
            transition_matrix = gaussian_transition_matrix(xs, params.sigx_pred_slow_phase)
        else:
            transition_matrix = gaussian_transition_matrix(xs, params.sigx_pred)

        prior = normalize(prev_post @ transition_matrix)
        likelihood = likelihood_grid_for_counts(
            xs, pf_centers, pf_width, rate_scale, spikes[example_time]
        )

        if params.T_remap_start <= example_time <= params.T_remap_end:
            likelihood = apply_remap_for_likelihoods(likelihood, params.remap_from_to, active=True)

        combined_likelihood = normalize(np.prod(likelihood, axis=1))

        # Normalize likelihood to probability density
        dx = xs[1] - xs[0]
        likelihood_norm = combined_likelihood / (np.sum(combined_likelihood) * dx)

        plot_data.append(
            {
                "phase_name": phase_name,
                "col_idx": col_idx,
                "color_key": color_key,
                "example_time": example_time,
                "prior": prior,
                "likelihood_norm": likelihood_norm,
            }
        )

    # Determine shared y-limits across all panels using robust percentiles
    all_y_values = []
    for data in plot_data:
        all_y_values.extend(data["prior"])
        all_y_values.extend(data["likelihood_norm"])
    # Use percentiles to avoid outliers distorting the scale
    y_min, y_max = np.percentile(all_y_values, [0.1, 99.9])
    # Add small padding
    y_range = y_max - y_min
    y_min = max(0, y_min - 0.02 * y_range)
    y_max = y_max + 0.02 * y_range

    # Second pass: create plots with shared y-axis
    example_axes = []
    for data in plot_data:
        phase_name = data["phase_name"]
        col_idx = data["col_idx"]
        color_key = data["color_key"]
        example_time = data["example_time"]
        prior = data["prior"]
        likelihood_norm = data["likelihood_norm"]

        # Create subplot (spans 2 rows)
        ax1 = fig.add_subplot(gs[5:7, col_idx])
        example_axes.append(ax1)

        # Set background color matching phase
        ax1.set_facecolor(phase_colors[color_key])

        # Plot both distributions on the same axis
        ax1.plot(xs, prior, color=wong[5], linewidth=1.2, alpha=0.7, label="Predictive")
        ax1.plot(xs, likelihood_norm, color=wong[1], linewidth=1.2, alpha=0.9, label="Likelihood")

        # Share y-axis: same limits for all panels, labels only on leftmost
        ax1.set_ylim(y_min, y_max)
        if col_idx == 0:
            ax1.set_ylabel("Probability Density", fontsize=7, labelpad=4)
            ax1.tick_params(axis="y", labelsize=5)
            # Use 3 ticks for better readability
            ax1.set_yticks(np.linspace(y_min, y_max, 3))
            ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{x:.2g}"))
        else:
            ax1.yaxis.set_major_formatter(plt.NullFormatter())

        # True position
        ax1.axvline(
            x_true[example_time],
            color="magenta",
            linestyle="--",
            linewidth=0.8,
            alpha=0.7,
            zorder=5,
            clip_on=False,
        )

        # Title with just phase name (de-emphasized)
        ax1.set_title(phase_name, fontsize=7, pad=4)

        # Add metrics as text annotation inside plot (upper left)
        hpdo_val = metrics["HPDO"][example_time]
        kl_val = metrics["KL"][example_time]
        spike_prob_vals = metrics["spikeProb"][example_time]
        if not np.all(np.isnan(spike_prob_vals)):
            spike_prob_min = np.nanmin(spike_prob_vals)
            log_spike_prob = -np.log(spike_prob_min + 1e-12)
        else:
            log_spike_prob = np.nan

        if np.isnan(log_spike_prob):
            metrics_text = f"HPD: {hpdo_val:.2f}\nKL: {kl_val:.1f}\n-log p: N/A"
        else:
            metrics_text = f"HPD: {hpdo_val:.2f}\nKL: {kl_val:.1f}\n-log p: {log_spike_prob:.1f}"
        ax1.text(
            0.05,
            0.95,
            metrics_text,
            transform=ax1.transAxes,
            fontsize=5,
            va="top",
            ha="left",
            bbox={
                "boxstyle": "round,pad=0.3",
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.8,
            },
        )

        # All panels show x-axis labels
        ax1.set_xlim(xs[0], xs[-1])
        ax1.tick_params(axis="x", labelsize=6)
        ax1.set_xlabel("Position (a.u.)", fontsize=7, labelpad=4)

    # Create legend in separate column (similar to colorbar)
    legend_ax = fig.add_subplot(gs[5:7, 5])
    legend_ax.axis("off")
    # Create dummy lines for legend
    predictive_line = plt.Line2D([0], [0], color=wong[5], linewidth=1.2, alpha=0.7)
    likelihood_line = plt.Line2D([0], [0], color=wong[1], linewidth=1.2, alpha=0.9)
    position_line = plt.Line2D([0], [0], color=wong[7], linestyle="--", linewidth=0.8, alpha=0.7)
    legend_ax.legend(
        [predictive_line, likelihood_line, position_line],
        ["Predictive", "Likelihood", "Position"],
        loc="upper left",
        fontsize=6,
        frameon=False,
        handlelength=1.5,
    )

    # Panel labels: 'a' for time-series section, 'b-f' for examples
    ax_post.text(
        -0.08,
        1.05,
        "a",
        transform=ax_post.transAxes,
        fontsize=9,
        fontweight="bold",
        va="bottom",
        ha="right",
    )

    example_labels = ["b", "c", "d", "e", "f"]
    for label, ax in zip(example_labels, example_axes, strict=True):
        ax.text(
            -0.08,
            1.08,
            label,
            transform=ax.transAxes,
            fontsize=9,
            fontweight="bold",
            va="bottom",
            ha="right",
        )

    # Save
    import os

    # Save to figures directory as figure_2
    figures_dir = os.path.join(os.path.dirname(__file__), "..", "figures")
    os.makedirs(figures_dir, exist_ok=True)
    save_path_base = os.path.join(figures_dir, "figure02")
    save_figure(save_path_base)
    print(f"\nFigure 2 saved to {save_path_base}.{{pdf,png}}")


# -----------------------------
# Main orchestration
# -----------------------------


def run_demo(params: DecodeParams) -> None:
    """Run the full diagnostic demonstration with three simulation phases."""
    rng = np.random.default_rng(params.base_seed)

    # Ensure pf_centers is initialized
    assert params.pf_centers is not None, "pf_centers must be initialized"

    # Grid & transition matrices
    xs = np.arange(params.xs_min, params.xs_max + params.xs_step, params.xs_step, dtype=float)
    transition_matrix = gaussian_transition_matrix(xs, params.sigx_pred)
    transition_matrix_narrow = gaussian_transition_matrix(xs, params.sigx_pred_fast_phase)
    transition_matrix_inflated = gaussian_transition_matrix(xs, params.sigx_pred_slow_phase)

    # Generate all phases with recovery periods
    phases = []
    phase_labels = []

    # Phase 1: Clean baseline (0 - T_remap_start)
    x_last = 0.0
    n_time = params.T_remap_start
    x_true_phase = simulate_walk(
        n_time, params.sigx_pred, x_last, params.xs_min, params.xs_max, rng
    )
    spikes_phase = simulate_spikes_position_tuned(
        x_true_phase, params.pf_centers, params.pf_width, params.rate_scale, rng
    )
    phases.append((x_true_phase, spikes_phase))
    phase_labels.append("Clean Baseline")
    x_last = x_true_phase[-1]

    # Phase 2: Remapping misfit (T_remap_start - T_remap_end)
    n_time = params.T_remap_end - params.T_remap_start
    x_true_phase = simulate_walk(
        n_time, params.sigx_pred, x_last, params.xs_min, params.xs_max, rng
    )
    spikes_phase = simulate_spikes_position_tuned(
        x_true_phase, params.pf_centers, params.pf_width, params.rate_scale, rng
    )
    phases.append((x_true_phase, spikes_phase))
    phase_labels.append("Remapping Misfit")
    x_last = x_true_phase[-1]

    # Phase 3: Recovery 1 (T_remap_end - T_recovery1_end)
    n_time = params.T_recovery1_end - params.T_remap_end
    x_true_phase = simulate_walk(
        n_time, params.sigx_pred, x_last, params.xs_min, params.xs_max, rng
    )
    spikes_phase = simulate_spikes_position_tuned(
        x_true_phase, params.pf_centers, params.pf_width, params.rate_scale, rng
    )
    phases.append((x_true_phase, spikes_phase))
    phase_labels.append("Clean Recovery")
    x_last = x_true_phase[-1]

    # Phase 4: Flat firing misfit (T_recovery1_end - T_flat_end)
    n_time = params.T_flat_end - params.T_recovery1_end
    x_true_phase = simulate_walk(
        n_time, params.sigx_pred, x_last, params.xs_min, params.xs_max, rng
    )
    spikes_phase = simulate_spikes_flat_rate(n_time, len(params.pf_centers), rate=7e-3, rng=rng)
    phases.append((x_true_phase, spikes_phase))
    phase_labels.append("Flat Firing Misfit")
    x_last = x_true_phase[-1]

    # Phase 5: Recovery 2 (T_flat_end - T_recovery2_end)
    n_time = params.T_recovery2_end - params.T_flat_end
    x_true_phase = simulate_walk(
        n_time, params.sigx_pred, x_last, params.xs_min, params.xs_max, rng
    )
    spikes_phase = simulate_spikes_position_tuned(
        x_true_phase, params.pf_centers, params.pf_width, params.rate_scale, rng
    )
    phases.append((x_true_phase, spikes_phase))
    phase_labels.append("Clean Recovery")
    x_last = x_true_phase[-1]

    # Phase 6: Fast movement misfit (T_recovery2_end - T_fast_end)
    # Transition model misfit: decoder uses narrow transition matrix (sigx=0.1),
    # animal moves fast (sigx=10.0)
    # Prior will be far too narrow/concentrated compared to actual movement (100x mismatch!)
    n_time = params.T_fast_end - params.T_recovery2_end
    x_true_phase = simulate_walk(
        n_time, params.sigx_true_fast, x_last, params.xs_min, params.xs_max, rng
    )
    spikes_phase = simulate_spikes_position_tuned(
        x_true_phase, params.pf_centers, params.pf_width, params.rate_scale, rng
    )
    phases.append((x_true_phase, spikes_phase))
    phase_labels.append("Fast Movement Misfit")
    x_last = x_true_phase[-1]

    # Phase 7: Recovery 3 (T_fast_end - T_recovery3_end)
    n_time = params.T_recovery3_end - params.T_fast_end
    x_true_phase = simulate_walk(
        n_time, params.sigx_pred, x_last, params.xs_min, params.xs_max, rng
    )
    spikes_phase = simulate_spikes_position_tuned(
        x_true_phase, params.pf_centers, params.pf_width, params.rate_scale, rng
    )
    phases.append((x_true_phase, spikes_phase))
    phase_labels.append("Clean Recovery")
    x_last = x_true_phase[-1]

    # Phase 8: Slow movement misfit (T_recovery3_end - T_slow_end)
    # Transition model misfit: decoder uses inflated transition matrix (sigx=20.0),
    # animal stationary (sigx=0.0)
    # Prior will be far too broad/diffuse compared to actual lack of movement
    n_time = params.T_slow_end - params.T_recovery3_end
    x_true_phase = simulate_walk(
        n_time, params.sigx_true_slow, x_last, params.xs_min, params.xs_max, rng
    )
    spikes_phase = simulate_spikes_position_tuned(
        x_true_phase, params.pf_centers, params.pf_width, params.rate_scale, rng
    )
    phases.append((x_true_phase, spikes_phase))
    phase_labels.append("Slow Movement Misfit")

    # Concatenate all phases
    x_true = np.concatenate([x for x, _ in phases], axis=0)
    spikes = np.vstack([s for _, s in phases])

    # Decode (vectorized within time)
    metrics = decode_and_diagnostics(
        spikes=spikes,
        xs=xs,
        transition_matrix=transition_matrix,
        pf_centers=params.pf_centers,
        pf_width=params.pf_width,
        rate_scale=params.rate_scale,
        remap_window=params.remap_window,
        remap_from_to=params.remap_from_to,
        transition_matrix_narrow=transition_matrix_narrow,
        narrow_window=(params.T_recovery2_end, params.T_fast_end),
        transition_matrix_inflated=transition_matrix_inflated,
        inflate_window=(params.T_recovery3_end, params.T_slow_end),
    )

    # Thresholds from clean baseline window (first 6k timesteps, before remapping starts)
    th = compute_thresholds(metrics, baseline_end=params.T_remap_start)

    # Plot combined diagnostics figure
    plot_combined_diagnostics(
        xs,
        x_true,
        spikes,
        metrics,
        th,
        params,
        params.pf_centers,
        params.pf_width,
        params.rate_scale,
    )


if __name__ == "__main__":
    # Default params mirror the MATLAB script. To run quickly while prototyping,
    # reduce T1/T2/T3 here.
    params = DecodeParams()
    # e.g., for a fast smoke test:
    # params = DecodeParams(T1=3_000, T2=4_000, T3=5_000)
    run_demo(params)
