"""Create Figure 2: Diagnostic Metrics for State Space Models.

This figure explains the three diagnostic metrics (KL divergence, HPD overlap,
and predictive checks) using a shared synthetic example.

Layout (3 columns x 3 rows):
    AABBCC    <- Row 1: Input distributions for each metric
    AABBCC    <- Row 2: Intermediate computation
    AABBCC    <- Row 3: Final result with metric value

Columns (single panel label per column):
    A = KL Divergence mechanics (all 3 rows)
    B = HPD Overlap mechanics (all 3 rows)
    C = Predictive Check mechanics (all 3 rows)
"""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from numpy.typing import NDArray
from scipy.stats import norm

from statespacecheck_paper.plotting import compute_hpd_region
from statespacecheck_paper.simulation import normalize, safe_log
from statespacecheck_paper.style import COLORS, save_figure, set_figure_defaults

# =============================================================================
# Shared Example Data
# =============================================================================


def create_shared_example(rng: np.random.Generator) -> dict[str, Any]:
    """Create shared example data for all three metrics.

    All metrics are computed exactly using the same distributions.
    The predictive check p-value is computed via Monte Carlo sampling.

    Parameters
    ----------
    rng : np.random.Generator
        Random number generator for reproducibility.

    Returns
    -------
    dict[str, Any]
        Dictionary with predictive distribution, likelihood,
        position bins, and precomputed metrics.
    """
    import statespacecheck as ssc

    # Position grid
    n_bins = 200
    position_bins = np.linspace(0, 100, n_bins)
    dx = position_bins[1] - position_bins[0]

    # Predictive distribution: centered at 35, narrower (more certain)
    # Likelihood: centered at 60, broader (less certain)
    # Different widths make the min() in HPD overlap formula more meaningful
    pred_mean, pred_std = 35.0, 8.0
    like_mean, like_std = 60.0, 12.0

    predictive = norm.pdf(position_bins, loc=pred_mean, scale=pred_std)
    predictive = normalize(predictive)

    likelihood = norm.pdf(position_bins, loc=like_mean, scale=like_std)
    likelihood = normalize(likelihood)

    # Compute KL divergence and HPD overlap using statespacecheck
    kl_value = float(ssc.kl_divergence(predictive[np.newaxis, :], likelihood[np.newaxis, :])[0])
    hpd_value = float(ssc.hpd_overlap(predictive[np.newaxis, :], likelihood[np.newaxis, :])[0])

    # Compute exact p-value using Monte Carlo sampling
    # This mirrors the approach in analysis.py decode_and_diagnostics
    n_mc_samples = 1000

    # Observed log predictive density: log(integral of predictive * likelihood)
    # This is log p(y | y_{1:t-1}) = log sum_x p(x_t | y_{1:t-1}) p(y_t | x_t)
    # For normalized distributions, this is the dot product
    observed_log_pred = np.log(np.sum(predictive * likelihood) * dx + 1e-300)

    # Simulate reference distribution by:
    # 1. Sample position from predictive
    # 2. Generate "observation" from likelihood centered at that position
    # 3. Compute log predictive density for simulated observation
    simulated_log_pred_values = np.zeros(n_mc_samples)

    for i in range(n_mc_samples):
        # Sample position from predictive distribution
        cumsum = np.cumsum(predictive)
        cumsum = cumsum / cumsum[-1]  # Ensure sums to 1
        u = rng.random()
        sampled_idx = int(np.searchsorted(cumsum, u))
        sampled_idx = min(sampled_idx, len(position_bins) - 1)
        sampled_pos = position_bins[sampled_idx]

        # Generate simulated "likelihood" centered at sampled position
        # (same width as original likelihood)
        simulated_likelihood = norm.pdf(position_bins, loc=sampled_pos, scale=like_std)
        simulated_likelihood = normalize(simulated_likelihood)

        # Compute log predictive density for this simulated observation
        simulated_log_pred_values[i] = np.log(
            np.sum(predictive * simulated_likelihood) * dx + 1e-300
        )

    # P-value: proportion of simulated values <= observed value
    p_value = float(np.mean(simulated_log_pred_values <= observed_log_pred))

    return {
        "position_bins": position_bins,
        "predictive": predictive,
        "likelihood": likelihood,
        "kl_value": kl_value,
        "hpd_value": hpd_value,
        "p_value": p_value,
        "pred_mean": pred_mean,
        "like_mean": like_mean,
        "pred_std": pred_std,
        "like_std": like_std,
        "observed_log_pred": observed_log_pred,
        "simulated_log_pred": simulated_log_pred_values,
    }


# =============================================================================
# KL Divergence Panels (A, B, C)
# =============================================================================


def plot_kl_panel_a(ax: Axes, data: dict[str, Any]) -> None:
    """Panel A: Predictive and Likelihood distributions overlaid."""
    x = data["position_bins"]
    pred = data["predictive"]
    like = data["likelihood"]

    ax.plot(x, pred, color=COLORS["predictive"], linewidth=1.5, label="Predictive")
    ax.fill_between(x, pred, alpha=0.3, color=COLORS["predictive"])
    ax.plot(x, like, color=COLORS["likelihood"], linewidth=1.5, label="Likelihood")
    ax.fill_between(x, like, alpha=0.3, color=COLORS["likelihood"])

    ax.set_xlabel("Latent state (a.u.)", fontsize=7, labelpad=8)
    ax.set_ylabel("Probability", fontsize=7, labelpad=8)
    ax.legend(fontsize=5, frameon=False, loc="upper right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlim(0, 100)
    ax.set_ylim(bottom=0)
    # Minimal x-ticks: first and last only
    ax.set_xticks([0, 100])
    ax.set_xticklabels(["0", "100"], fontsize=5)
    # Minimal y-ticks: first and last only
    y_max = ax.get_ylim()[1]
    ax.set_yticks([0, y_max])
    ax.set_yticklabels(["0", f"{y_max:.2f}"], fontsize=5)


def plot_kl_panel_b(ax: Axes, data: dict[str, Any]) -> None:
    """Panel B: Log ratio = log(Predictive / Likelihood)."""
    x = data["position_bins"]
    pred = data["predictive"]
    like = data["likelihood"]

    log_ratio = safe_log(pred) - safe_log(like)

    ax.plot(x, log_ratio, color="gray", linewidth=1.5)
    ax.axhline(0, color=COLORS["reference"], linestyle="--", linewidth=0.8, alpha=0.7)

    # Fill positive and negative regions
    # Positive (pred > like) uses predictive color, negative (like > pred) uses likelihood color
    ax.fill_between(
        x,
        log_ratio,
        0,
        where=list(log_ratio > 0),
        alpha=0.3,
        color=COLORS["predictive"],
    )
    ax.fill_between(
        x,
        log_ratio,
        0,
        where=list(log_ratio < 0),
        alpha=0.3,
        color=COLORS["likelihood"],
    )

    ax.set_xlabel("Latent state (a.u.)", fontsize=7, labelpad=8)
    ax.set_ylabel(r"$\log(\mathrm{pred}) - \log(\mathrm{like})$", fontsize=6, labelpad=8)
    ax.set_title("Log Ratio", fontsize=6, pad=4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlim(0, 100)
    # Minimal x-ticks: first and last only
    ax.set_xticks([0, 100])
    ax.set_xticklabels(["0", "100"], fontsize=5)
    # Minimal y-ticks: first and last only
    y_min, y_max = ax.get_ylim()
    ax.set_yticks([y_min, 0, y_max])
    ax.set_yticklabels([f"{y_min:.0f}", "0", f"{y_max:.0f}"], fontsize=5)


def plot_kl_panel_c(ax: Axes, data: dict[str, Any]) -> None:
    """Panel C: Pointwise KL = Predictive * log(Pred/Lik)."""
    x = data["position_bins"]
    pred = data["predictive"]
    like = data["likelihood"]

    log_ratio = safe_log(pred) - safe_log(like)
    pointwise_kl = pred * log_ratio

    ax.plot(x, pointwise_kl, color=COLORS["kl_divergence"], linewidth=1.5)
    ax.fill_between(x, pointwise_kl, alpha=0.3, color=COLORS["kl_divergence"])
    ax.axhline(0, color=COLORS["reference"], linestyle="--", linewidth=0.8, alpha=0.7)

    ax.set_xlabel("Latent state (a.u.)", fontsize=7, labelpad=8)
    ax.set_ylabel(
        r"$\mathrm{pred} \cdot [\log(\mathrm{pred}) - \log(\mathrm{like})]$",
        fontsize=6,
        labelpad=8,
    )
    ax.set_title("Pointwise KL: pred × log ratio", fontsize=6, pad=4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlim(0, 100)
    # Minimal x-ticks: first and last only
    ax.set_xticks([0, 100])
    ax.set_xticklabels(["0", "100"], fontsize=5)
    # Minimal y-ticks: first and last only
    y_min, y_max = ax.get_ylim()
    ax.set_yticks([y_min, 0, y_max])
    ax.set_yticklabels([f"{y_min:.3f}", "0", f"{y_max:.3f}"], fontsize=5)


# =============================================================================
# HPD Overlap Panels (D, E, F)
# =============================================================================


def plot_hpd_panel_d(ax: Axes, data: dict[str, Any]) -> None:
    """Panel D: Predictive distribution with 95% HPD region shaded.

    Uses consistent HPD visual scheme: shaded region under curve within HPD.
    """
    x = data["position_bins"]
    pred = data["predictive"]
    coverage = 0.95

    hpd_mask = compute_hpd_region(x, pred, coverage)

    # Compute HPD threshold (minimum density value in HPD region)
    hpd_threshold = np.min(pred[hpd_mask])

    # Plot full distribution as line
    ax.plot(x, pred, color=COLORS["predictive"], linewidth=1.2)

    # Shade HPD region under curve (consistent scheme)
    ax.fill_between(
        x,
        0,
        pred,
        where=list(hpd_mask),
        alpha=0.35,
        color=COLORS["predictive"],
        label="95% HPD",
    )

    # Add 95% HPD threshold as dashed line with label
    ax.axhline(
        hpd_threshold,
        color=COLORS["predictive"],
        linestyle="--",
        linewidth=0.8,
        alpha=0.7,
    )
    # Label the threshold line
    ax.text(
        95,
        hpd_threshold,
        "95% threshold",
        fontsize=5,
        ha="right",
        va="bottom",
        color=COLORS["predictive"],
        alpha=0.8,
    )

    ax.set_xlabel("Latent state (a.u.)", fontsize=7, labelpad=8)
    ax.set_ylabel("Probability", fontsize=7, labelpad=8)
    ax.set_title(r"Predictive HPD ($H_{\mathrm{pred}}$)", fontsize=6, pad=4)
    ax.legend(fontsize=5, frameon=False, loc="upper right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlim(0, 100)
    ax.set_ylim(bottom=0)
    # Minimal x-ticks: first and last only
    ax.set_xticks([0, 100])
    ax.set_xticklabels(["0", "100"], fontsize=5)
    # Minimal y-ticks: first and last only
    y_max = ax.get_ylim()[1]
    ax.set_yticks([0, y_max])
    ax.set_yticklabels(["0", f"{y_max:.2f}"], fontsize=5)


def plot_hpd_panel_e(ax: Axes, data: dict[str, Any]) -> None:
    """Panel E: Likelihood distribution with 95% HPD region shaded.

    Uses consistent HPD visual scheme: shaded region under curve within HPD.
    """
    x = data["position_bins"]
    like = data["likelihood"]
    coverage = 0.95

    hpd_mask = compute_hpd_region(x, like, coverage)

    # Compute HPD threshold (minimum density value in HPD region)
    hpd_threshold = np.min(like[hpd_mask])

    # Plot full distribution as line
    ax.plot(x, like, color=COLORS["likelihood"], linewidth=1.2)

    # Shade HPD region under curve (consistent scheme)
    ax.fill_between(
        x,
        0,
        like,
        where=list(hpd_mask),
        alpha=0.35,
        color=COLORS["likelihood"],
        label="95% HPD",
    )

    # Add 95% HPD threshold as dashed line with label
    ax.axhline(
        hpd_threshold,
        color=COLORS["likelihood"],
        linestyle="--",
        linewidth=0.8,
        alpha=0.7,
    )
    # Label the threshold line (on left side to avoid distribution)
    ax.text(
        5,
        hpd_threshold,
        "95% threshold",
        fontsize=5,
        ha="left",
        va="bottom",
        color=COLORS["likelihood"],
        alpha=0.8,
    )

    ax.set_xlabel("Latent state (a.u.)", fontsize=7, labelpad=8)
    ax.set_ylabel("Probability", fontsize=7, labelpad=8)
    ax.set_title(r"Likelihood HPD ($H_{\mathrm{like}}$)", fontsize=6, pad=4)
    ax.legend(fontsize=5, frameon=False, loc="upper right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlim(0, 100)
    ax.set_ylim(bottom=0)
    # Minimal x-ticks: first and last only
    ax.set_xticks([0, 100])
    ax.set_xticklabels(["0", "100"], fontsize=5)
    # Minimal y-ticks: first and last only
    y_max = ax.get_ylim()[1]
    ax.set_yticks([0, y_max])
    ax.set_yticklabels(["0", f"{y_max:.2f}"], fontsize=5)


def plot_hpd_panel_f(ax: Axes, data: dict[str, Any]) -> tuple[float, float, float]:
    """Panel F: HPD intersection region.

    Shows HPD regions as thick horizontal lines at different y-levels.
    Order: Pred HPD (top), Lik HPD (middle), Overlap (bottom) to match panels above.
    Labels use descriptive text with formula notation in parentheses.
    Size values are displayed on each bar for the calculation breakdown.

    Returns
    -------
    tuple[float, float, float]
        (pred_hpd_size, like_hpd_size, intersection_size) for use in formula.
    """
    x = data["position_bins"]
    pred = data["predictive"]
    like = data["likelihood"]
    coverage = 0.95
    dx = x[1] - x[0]

    pred_hpd = compute_hpd_region(x, pred, coverage)
    like_hpd = compute_hpd_region(x, like, coverage)
    intersection = pred_hpd & like_hpd

    # Compute sizes for annotation
    pred_hpd_size = float(np.sum(pred_hpd) * dx)
    like_hpd_size = float(np.sum(like_hpd) * dx)
    intersection_size = float(np.sum(intersection) * dx)

    # Convert boolean masks to y-values (NaN where False for line breaks)
    def mask_to_line(mask: NDArray[np.bool_], y_level: float) -> NDArray[np.float64]:
        """Convert boolean mask to y-values with NaN gaps."""
        result = np.where(mask, y_level, np.nan)
        return result.astype(np.float64)

    # Draw thick horizontal lines for each HPD region
    # Order top-to-bottom: Pred HPD, Lik HPD, Overlap (matching panels above)
    ax.plot(
        x,
        mask_to_line(pred_hpd, 0.8),
        color=COLORS["predictive"],
        linewidth=6,
        solid_capstyle="butt",
    )
    ax.plot(
        x,
        mask_to_line(like_hpd, 0.5),
        color=COLORS["likelihood"],
        linewidth=6,
        solid_capstyle="butt",
    )
    ax.plot(
        x,
        mask_to_line(intersection, 0.2),
        color=COLORS["hpd_overlap"],
        linewidth=6,
        solid_capstyle="butt",
    )

    # Add size labels centered on each bar
    def get_center(mask: NDArray[np.bool_], positions: NDArray[np.floating]) -> float:
        """Get center position of a mask region."""
        if not np.any(mask):
            return 50.0
        return float(np.mean(positions[mask]))

    pred_center = get_center(pred_hpd, x)
    like_center = get_center(like_hpd, x)
    inter_center = get_center(intersection, x)

    ax.text(
        pred_center,
        0.84,
        rf"|$H_{{\mathrm{{pred}}}}$| = {pred_hpd_size:.1f}",
        fontsize=6,
        ha="center",
        va="bottom",
        color=COLORS["predictive"],
    )
    ax.text(
        like_center,
        0.54,
        rf"|$H_{{\mathrm{{like}}}}$| = {like_hpd_size:.1f}",
        fontsize=6,
        ha="center",
        va="bottom",
        color=COLORS["likelihood"],
    )
    ax.text(
        inter_center,
        0.24,
        rf"|$H_{{\mathrm{{pred}}}} \cap H_{{\mathrm{{like}}}}$| = {intersection_size:.1f}",
        fontsize=6,
        ha="center",
        va="bottom",
        color=COLORS["hpd_overlap"],
    )

    ax.set_xlabel("Latent state (a.u.)", fontsize=7, labelpad=8)
    ax.set_ylabel("", fontsize=7, labelpad=8)  # No y-label needed
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 1.0)
    ax.set_yticks([0.8, 0.5, 0.2])

    # Descriptive labels with formula notation
    ax.set_yticklabels(
        [r"Pred. HPD ($H_{\mathrm{pred}}$)", r"Like. HPD ($H_{\mathrm{like}}$)", r"Intersection"],
        fontsize=5,
    )

    ax.tick_params(axis="y", length=0)  # Hide tick marks
    # Minimal x-ticks: first and last only
    ax.set_xticks([0, 100])
    ax.set_xticklabels(["0", "100"], fontsize=5)

    return pred_hpd_size, like_hpd_size, intersection_size


# =============================================================================
# Predictive Check Panels (G, H, I)
# =============================================================================


def plot_ppc_panel_g(ax: Axes, data: dict[str, Any]) -> None:
    """Panel G: Predictive distribution with sampled position marked."""
    x = data["position_bins"]
    pred = data["predictive"]

    ax.plot(x, pred, color=COLORS["predictive"], linewidth=1.5, label="Predictive")
    ax.fill_between(x, pred, alpha=0.3, color=COLORS["predictive"])

    # Show sampled position (near the peak)
    sampled_pos = data["pred_mean"] + 3  # Slightly off-peak for visibility
    sampled_idx = np.argmin(np.abs(x - sampled_pos))

    ax.axvline(
        sampled_pos,
        color=COLORS["ground_truth"],
        linestyle="-",
        linewidth=1.5,
        alpha=0.8,
    )
    ax.scatter(
        [sampled_pos],
        [pred[sampled_idx]],
        color=COLORS["ground_truth"],
        s=40,
        zorder=5,
        label="Sample",
    )

    ax.set_xlabel("Latent state (a.u.)", fontsize=7, labelpad=8)
    ax.set_ylabel("Probability", fontsize=7, labelpad=8)
    ax.legend(fontsize=5, frameon=False, loc="upper right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlim(0, 100)
    ax.set_ylim(bottom=0)
    # Minimal x-ticks: first and last only
    ax.set_xticks([0, 100])
    ax.set_xticklabels(["0", "100"], fontsize=5)
    # Minimal y-ticks: first and last only
    y_max = ax.get_ylim()[1]
    ax.set_yticks([0, y_max])
    ax.set_yticklabels(["0", f"{y_max:.2f}"], fontsize=5)


def plot_ppc_panel_h(ax: Axes, data: dict[str, Any], rng: np.random.Generator) -> None:
    """Panel H: Simulated spikes from sampled position (raster-style)."""
    # Simulate spike counts for ~10 cells at the sampled position
    n_cells = 8
    sampled_pos = data["pred_mean"] + 3

    # Create simple place fields and simulate spikes
    cell_centers = np.linspace(10, 90, n_cells)
    cell_width = 15.0

    # Firing rates at sampled position
    rates = norm.pdf(sampled_pos, loc=cell_centers, scale=cell_width)
    rates = rates / rates.max() * 0.3  # Scale to reasonable spike probability

    # Generate spikes (binary for this visualization)
    spikes = rng.random(n_cells) < rates

    # Plot as colored circles (filled = spike, empty = no spike)
    # Spikes are simulated from predictive distribution, so use predictive color
    for i, (center, has_spike) in enumerate(zip(cell_centers, spikes, strict=True)):
        color = COLORS["predictive"] if has_spike else "white"
        edgecolor = COLORS["predictive"]
        ax.scatter(
            [center],
            [i],
            s=60,
            c=color,
            edgecolors=edgecolor,
            linewidths=1,
            zorder=5,
        )

    # Mark sampled position
    ax.axvline(
        sampled_pos,
        color=COLORS["ground_truth"],
        linestyle="-",
        linewidth=1.5,
        alpha=0.5,
    )

    ax.set_xlabel("Latent state (a.u.)", fontsize=7, labelpad=8)
    ax.set_ylabel("Cell", fontsize=7, labelpad=8)
    ax.set_xlim(0, 100)
    ax.set_ylim(-0.5, n_cells - 0.5)
    ax.set_yticks([0, n_cells - 1])
    ax.set_yticklabels(["1", str(n_cells)], fontsize=5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    # Minimal x-ticks: first and last only
    ax.set_xticks([0, 100])
    ax.set_xticklabels(["0", "100"], fontsize=5)

    # Add annotation
    ax.text(
        0.5,
        1.02,
        "Simulated spikes",
        transform=ax.transAxes,
        fontsize=6,
        ha="center",
        va="bottom",
        style="italic",
    )


def plot_ppc_panel_i(ax: Axes, data: dict[str, Any]) -> None:
    """Panel I: Histogram of observed vs simulated log predictive density.

    Uses the exact Monte Carlo samples computed in create_shared_example().
    """
    # Use exact values from Monte Carlo simulation
    simulated_log_pred = data["simulated_log_pred"]
    observed_log_pred = data["observed_log_pred"]

    # Histogram of simulated values (from predictive distribution)
    ax.hist(
        simulated_log_pred,
        bins=30,
        density=True,
        alpha=0.5,
        color=COLORS["predictive"],
        edgecolor="none",
        label="Simulated",
    )

    # Mark observed value (from likelihood/actual observation)
    ax.axvline(
        observed_log_pred,
        color=COLORS["likelihood"],
        linewidth=2,
        linestyle="-",
        label="Observed",
    )

    ax.set_xlabel("Log pred. density", fontsize=7, labelpad=8)
    ax.set_ylabel("Density", fontsize=7, labelpad=8)
    ax.legend(fontsize=5, frameon=False, loc="upper left")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    # Minimal x-ticks: first and last only
    x_min, x_max = ax.get_xlim()
    ax.set_xticks([x_min, x_max])
    ax.set_xticklabels([f"{x_min:.0f}", f"{x_max:.0f}"], fontsize=5)
    # Minimal y-ticks: first and last only
    y_max = ax.get_ylim()[1]
    ax.set_yticks([0, y_max])
    ax.set_yticklabels(["0", f"{y_max:.1f}"], fontsize=5)


# =============================================================================
# Main Figure Creation
# =============================================================================


def create_figure() -> None:
    """Create Figure 2 with diagnostic metric mechanics.

    Layout (3 columns x 4 rows):
        Row 1: Input distributions for each metric
        Row 2: Intermediate computation
        Row 3: Final result
        Row 4: Formula with computed value

    Columns (single panel label per column):
        a = KL Divergence mechanics
        b = HPD Overlap mechanics
        c = Predictive Check mechanics
    """
    set_figure_defaults(context="paper")
    rng = np.random.default_rng(42)

    # Create shared example data (pass rng for reproducible Monte Carlo p-value)
    data = create_shared_example(rng)

    # =========================================================================
    # LAYOUT CONFIGURATION
    # =========================================================================
    # Simple 3 columns x 4 rows - each column is one metric
    # No arrow rows - computational flow is implicit (top to bottom)
    # Use '.' spacers between columns for visual separation
    layout = """
        AA.BB.CC
        DD.EE.FF
        GG.HH.II
        JJ.KK.LL
        """

    fig, axes = plt.subplot_mosaic(
        layout,
        figsize=(7.0, 5.5),  # More compact without arrow rows
        width_ratios=[1, 1, 0.2, 1, 1, 0.2, 1, 1],  # Spacer columns
        height_ratios=[1, 1, 1, 0.35],  # Content rows + formula row
        dpi=450,
        constrained_layout={"h_pad": 0.08, "w_pad": 0.04},  # More padding between rows
    )

    # -------------------------------------------------------------------------
    # KL Divergence column (A, D, G)
    # -------------------------------------------------------------------------
    plot_kl_panel_a(axes["A"], data)
    plot_kl_panel_b(axes["D"], data)
    plot_kl_panel_c(axes["G"], data)

    # -------------------------------------------------------------------------
    # HPD Overlap column (B, E, H)
    # -------------------------------------------------------------------------
    plot_hpd_panel_d(axes["B"], data)
    plot_hpd_panel_e(axes["E"], data)
    # Returns (pred_size, like_size, intersection_size) for formula
    hpd_sizes = plot_hpd_panel_f(axes["H"], data)

    # -------------------------------------------------------------------------
    # Predictive Check column (C, F, I)
    # -------------------------------------------------------------------------
    plot_ppc_panel_g(axes["C"], data)
    plot_ppc_panel_h(axes["F"], data, rng)
    plot_ppc_panel_i(axes["I"], data)

    # Column titles for metrics (above top row)
    column_titles = [("A", "KL Divergence"), ("B", "HPD Overlap"), ("C", "Predictive Check")]
    for ax_key, col_title in column_titles:
        ax = axes[ax_key]
        ax.text(
            0.5,
            1.18,
            col_title,
            transform=ax.transAxes,
            fontsize=8,
            fontweight="bold",
            ha="center",
            va="bottom",
        )

    # -------------------------------------------------------------------------
    # Formula row (J, K, L)
    # -------------------------------------------------------------------------
    # KL Divergence formula
    axes["J"].axis("off")
    axes["J"].text(
        0.5,
        0.5,
        r"$D_{\mathrm{KL}} = \sum \mathrm{pred} \cdot \log(\mathrm{pred}/\mathrm{like})$"
        f" = {data['kl_value']:.2f}",
        transform=axes["J"].transAxes,
        fontsize=8,
        fontweight="bold",
        ha="center",
        va="center",
    )

    # HPD Overlap formula with notation and calculation breakdown
    pred_size, like_size, intersection_size = hpd_sizes
    axes["K"].axis("off")
    # Show notation = fraction with numbers = result
    hpd_formula = (
        r"$\frac{|H_{\mathrm{pred}} \cap H_{\mathrm{like}}|}"
        r"{\min(|H_{\mathrm{pred}}|, |H_{\mathrm{like}}|)}$"
        f" = "
        rf"$\frac{{{intersection_size:.1f}}}{{{min(pred_size, like_size):.1f}}}$"
        f" = {data['hpd_value']:.2f}"
    )
    axes["K"].text(
        0.5,
        0.5,
        hpd_formula,
        transform=axes["K"].transAxes,
        fontsize=8,
        fontweight="bold",
        ha="center",
        va="center",
    )

    # Predictive Check formula
    axes["L"].axis("off")
    axes["L"].text(
        0.5,
        0.5,
        f"$p = P(T^{{rep}} \\leq T^{{obs}})$ = {data['p_value']:.2f}",
        transform=axes["L"].transAxes,
        fontsize=8,
        fontweight="bold",
        ha="center",
        va="center",
    )

    # =========================================================================
    # PANEL LABELS (one per column, on top row only)
    # =========================================================================
    column_panels = ["A", "B", "C"]
    labels = ["a", "b", "c"]

    for label, ax_key in zip(labels, column_panels, strict=True):
        ax = axes[ax_key]
        ax.text(
            -0.15,  # Further left, outside frame
            1.08,  # Above frame
            label,
            transform=ax.transAxes,
            fontsize=8,
            fontweight="bold",
            va="bottom",
            ha="right",
        )

    # =========================================================================
    # SAVE
    # =========================================================================
    save_figure("manuscript/figures/main/figure02")
    print("\nFigure 2 saved to manuscript/figures/main/figure02.{pdf,png}")


if __name__ == "__main__":
    create_figure()
