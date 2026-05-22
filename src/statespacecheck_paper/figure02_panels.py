"""Per-panel renderers for Figure 2.

This module owns the 11 ``plot_*_panel_*`` helpers and the
two shared helpers (``create_shared_example`` and ``_showcase_colors``)
that the figure-2 script composes. The script-level
``scripts/generate_figure02.py`` orchestrates layout, panel labels,
and saving.

Each panel function takes the matplotlib ``Axes`` to draw into plus
the ``data`` dict produced by ``create_shared_example`` (so the three
columns of Figure 2 are rendered against a single shared example).
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
from statespacecheck_paper.style import COLORS

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
    # This is log p(y | y_{1:t-1}) = log sum_x p(x_t | y_{1:t-1}) p(y_t | x_t).
    # If the overlap underflows for some pathological draw, np.log emits a
    # RuntimeWarning and yields -inf rather than the previous behaviour of
    # mapping zero to log(1e-300) ≈ -690, which would silently turn the
    # p-value into a tie count between observed and simulated underflows.
    observed_log_pred = float(np.log(np.sum(predictive * likelihood) * dx))

    # Simulate the reference distribution per the predictive-check
    # definition in eq:fpred / eq:predictive_application:
    # 1. Sample state x_s ~ predictive(x).
    # 2. Sample observation y_tilde ~ p(y | x_s) = N(x_s, like_std^2).
    # 3. Compute log f_pred(y_tilde) = log integral predictive(x) * p(y_tilde | x) dx.
    simulated_log_pred_values = np.zeros(n_mc_samples)

    cumsum = np.cumsum(predictive)
    cumsum = cumsum / cumsum[-1]  # Ensure sums to 1

    for i in range(n_mc_samples):
        # Sample state from the predictive distribution.
        u = rng.random()
        sampled_idx = int(np.searchsorted(cumsum, u))
        sampled_idx = min(sampled_idx, len(position_bins) - 1)
        sampled_pos = position_bins[sampled_idx]

        # Sample a simulated observation y_tilde ~ p(y | x_s) = N(x_s, like_std^2).
        # This is the step that makes the schematic match the manuscript's
        # predictive-check algorithm rather than its mean-prediction shortcut.
        y_tilde = float(rng.normal(loc=sampled_pos, scale=like_std))

        # Simulated likelihood as a function of x: p(y_tilde | x) = N(y_tilde; x, like_std).
        simulated_likelihood = norm.pdf(position_bins, loc=y_tilde, scale=like_std)
        simulated_likelihood = normalize(simulated_likelihood)

        # Compute log predictive density for this simulated observation.
        # Lets np.log handle underflow with -inf + RuntimeWarning rather
        # than masking it with a +1e-300 shift (see observed_log_pred above
        # for the rationale).
        simulated_log_pred_values[i] = np.log(np.sum(predictive * simulated_likelihood) * dx)

    # P-value: proportion of simulated values <= observed value
    p_value = float(np.mean(simulated_log_pred_values <= observed_log_pred))

    # Showcase samples for the predictive-check schematic (panels G and H).
    # State positions are picked at evenly-spaced quantiles of the predictive
    # CDF so the displayed fan spans the predictive's support rather than
    # clustering near the peak. For each state, a simulated observation
    # y_tilde is then drawn from p(y | x_s) = N(x_s, like_std^2) -- the same
    # step the Monte Carlo loop above uses -- and the simulated likelihood is
    # p(y_tilde | x) plotted as a function of x.
    showcase_quantiles = np.array([0.10, 0.30, 0.50, 0.70, 0.90])
    showcase_idx = np.minimum(np.searchsorted(cumsum, showcase_quantiles), n_bins - 1)
    showcase_positions = position_bins[showcase_idx]
    showcase_y_tildes = rng.normal(loc=showcase_positions, scale=like_std)
    showcase_likelihoods = np.stack(
        [normalize(norm.pdf(position_bins, loc=y, scale=like_std)) for y in showcase_y_tildes]
    )

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
        "showcase_positions": showcase_positions,
        "showcase_y_tildes": showcase_y_tildes,
        "showcase_likelihoods": showcase_likelihoods,
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


def _showcase_colors(n: int) -> NDArray[np.float64]:
    """Distinct colors for the predictive-check sample fan.

    Sampled from viridis avoiding the very dark and very light ends so
    every curve stays legible at small panel sizes.
    """
    cmap = plt.get_cmap("viridis")
    return cmap(np.linspace(0.15, 0.85, n))


def plot_ppc_panel_g(ax: Axes, data: dict[str, Any]) -> None:
    """Panel G: Predictive distribution with a fan of sampled positions.

    Each colored marker is one draw from the predictive that flows into
    the corresponding simulated observation likelihood plotted in panel H.
    """
    x = data["position_bins"]
    pred = data["predictive"]
    positions = data["showcase_positions"]
    colors = _showcase_colors(len(positions))

    ax.plot(x, pred, color=COLORS["predictive"], linewidth=1.5, label="Predictive")
    ax.fill_between(x, pred, alpha=0.3, color=COLORS["predictive"])

    # Each sampled state is shown as a colored tick + dot on the predictive
    # so the matching simulated likelihood in panel H can be read off by colour.
    sample_indices = np.argmin(np.abs(x[None, :] - positions[:, None]), axis=1)
    for pos, idx, color in zip(positions, sample_indices, colors, strict=True):
        ax.axvline(pos, color=color, linestyle="-", linewidth=1.0, alpha=0.7)
        ax.scatter([pos], [pred[idx]], color=color, s=30, zorder=5)
    # Single legend entry summarises the fan.
    ax.scatter([], [], color=colors[len(colors) // 2], s=30, label="Samples")

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


def plot_ppc_panel_h(ax: Axes, data: dict[str, Any]) -> None:
    """Panel H: Fan of simulated observation likelihoods.

    For each state sample drawn from the predictive (panel G), the
    Monte Carlo loop draws an observation y_tilde ~ p(y | x_s) and
    constructs the corresponding observation likelihood p(y_tilde | x).
    This panel shows that fan of likelihood curves, colored to match
    the samples in panel G. Per-curve markers distinguish the state
    sample (dotted line at x_s) from the drawn observation (triangle
    at the curve peak, at y_tilde). A faint dashed copy of the
    predictive is overlaid so the reader can see what the curves get
    multiplied by when computing the log predictive density that ends
    up in panel I.
    """

    x = data["position_bins"]
    pred = data["predictive"]
    positions = data["showcase_positions"]
    y_tildes = data["showcase_y_tildes"]
    likelihoods = data["showcase_likelihoods"]
    colors = _showcase_colors(len(positions))

    # Faint dashed predictive overlay so the reader sees what each
    # simulated likelihood gets weighted by when integrated into panel I.
    ax.plot(
        x,
        pred,
        color=COLORS["predictive"],
        linewidth=0.8,
        linestyle="--",
        alpha=0.4,
    )

    for pos, y_t, lik, color in zip(positions, y_tildes, likelihoods, colors, strict=True):
        ax.plot(x, lik, color=color, linewidth=1.0, alpha=0.85)
        ax.fill_between(x, lik, color=color, alpha=0.12)
        # Dotted vertical at the originating state sample x_s — lines up
        # with the dot at the same color in panel G.
        ax.axvline(pos, color=color, linestyle=":", linewidth=0.8, alpha=0.5)
        # Downward triangle at the curve peak marks the drawn observation
        # y_tilde. Distinct from the dot symbol used for state samples in
        # panel G so the two quantities are never conflated.
        peak_idx = int(np.argmin(np.abs(x - y_t)))
        ax.scatter(
            [x[peak_idx]],
            [lik[peak_idx]],
            color=color,
            marker="v",
            s=24,
            zorder=5,
            edgecolors="white",
            linewidths=0.4,
        )

    # Legend-only proxy artists describe the visual conventions of the
    # panel; the real curves/markers are colored per-sample.
    neutral = "0.4"
    ax.plot(
        [],
        [],
        color=COLORS["predictive"],
        linewidth=0.8,
        linestyle="--",
        alpha=0.6,
        label="Predictive",
    )
    ax.plot(
        [],
        [],
        color=neutral,
        linestyle=":",
        linewidth=0.8,
        label=r"State $x_s$",
    )
    ax.scatter(
        [],
        [],
        color=neutral,
        marker="v",
        s=24,
        edgecolors="white",
        linewidths=0.4,
        label=r"Obs. $\tilde{y}$",
    )

    ax.legend(fontsize=5, frameon=False, loc="upper right")
    ax.set_xlabel("Latent state (a.u.)", fontsize=7, labelpad=8)
    ax.set_ylabel("Probability", fontsize=7, labelpad=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlim(0, 100)
    ax.set_ylim(bottom=0)
    ax.set_xticks([0, 100])
    ax.set_xticklabels(["0", "100"], fontsize=5)
    y_max = ax.get_ylim()[1]
    ax.set_yticks([0, y_max])
    ax.set_yticklabels(["0", f"{y_max:.2f}"], fontsize=5)

    ax.text(
        0.5,
        1.02,
        "Simulated obs. likelihoods",
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
