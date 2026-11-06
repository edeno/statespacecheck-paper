"""Per-panel renderers for Figure 2.

This module owns the nine panel renderers, the typed
``Figure2ExampleData`` contract, and the shared example/color helpers that
``figure02_generation`` composes. The thin ``scripts/generate_figure02.py``
module only invokes that generation recipe.

Each panel function takes the matplotlib ``Axes`` to draw into plus the typed
``Figure2ExampleData`` produced by ``create_shared_example`` (so the three
columns of Figure 2 are rendered against a single immutable example).
"""

from __future__ import annotations

import dataclasses
from typing import Any, Literal

import matplotlib.pyplot as plt
import numpy as np
import statespacecheck as ssc
from matplotlib.axes import Axes
from matplotlib.axis import Axis
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, MaxNLocator
from numpy.typing import NDArray
from scipy.stats import norm

from statespacecheck_paper.diagnostics import HPD_COVERAGE
from statespacecheck_paper.simulation import normalize
from statespacecheck_paper.style import COLORS

# Quantile of the simulated log predictive densities below which the
# predictive-check histogram crops its sparse left tail
HISTOGRAM_LOWER_QUANTILE = 0.001

# Seed of the three replicate marks drawn in the schematic, chosen so that the
# marks and the observed mark are far apart and the replicates fall in different
# parts of their observation models.
SHOWCASE_MARK_SEED = 2281

# =============================================================================
# Shared Example Data
# =============================================================================


@dataclasses.dataclass(frozen=True)
class Figure2ExampleData:
    """Validated shared example used by every Figure-2 panel.

    Named attributes make the scientific quantities discoverable to readers
    and checkable by mypy. Array
    fields are defensively copied and made read-only so all nine panels are
    guaranteed to render the same immutable realization.
    """

    position_bins: NDArray[np.float64]
    predictive: NDArray[np.float64]
    likelihood: NDArray[np.float64]
    kl_divergence: float
    hpd_overlap: float
    predictive_pvalue: float
    observed_log_pred: float
    observed_mark: float
    simulated_log_pred: NDArray[np.float64]
    showcase_positions: NDArray[np.float64]
    showcase_y_tildes: NDArray[np.float64]
    mark_bins: NDArray[np.float64]
    predictive_mark_density: NDArray[np.float64]
    showcase_mark_densities: NDArray[np.float64]

    def __post_init__(self) -> None:
        arrays = {
            name: np.array(getattr(self, name), dtype=np.float64)
            for name in (
                "position_bins",
                "predictive",
                "likelihood",
                "simulated_log_pred",
                "showcase_positions",
                "showcase_y_tildes",
                "mark_bins",
                "predictive_mark_density",
                "showcase_mark_densities",
            )
        }
        n_bins = arrays["position_bins"].size
        if arrays["position_bins"].ndim != 1 or n_bins < 2:
            raise ValueError("position_bins must be a 1-D array with at least two bins")
        if np.any(np.diff(arrays["position_bins"]) <= 0.0):
            raise ValueError("position_bins must be strictly increasing")
        for name in ("predictive", "likelihood"):
            if arrays[name].shape != (n_bins,):
                raise ValueError(f"{name} must have shape ({n_bins},); got {arrays[name].shape}")
            if np.any(arrays[name] < 0.0) or not np.isclose(arrays[name].sum(), 1.0):
                raise ValueError(f"{name} must be a normalized non-negative distribution")
        if arrays["simulated_log_pred"].ndim != 1:
            raise ValueError("simulated_log_pred must be 1-D")
        n_showcase = arrays["showcase_positions"].size
        if arrays["showcase_positions"].ndim != 1:
            raise ValueError("showcase_positions must be 1-D")
        if arrays["showcase_y_tildes"].shape != (n_showcase,):
            raise ValueError(
                "showcase_y_tildes must match showcase_positions; "
                f"got {arrays['showcase_y_tildes'].shape} vs ({n_showcase},)"
            )
        n_marks = arrays["mark_bins"].size
        if arrays["mark_bins"].ndim != 1 or n_marks < 2:
            raise ValueError("mark_bins must be a 1-D array with at least two bins")
        if arrays["predictive_mark_density"].shape != (n_marks,):
            raise ValueError(
                f"predictive_mark_density must have shape ({n_marks},); "
                f"got {arrays['predictive_mark_density'].shape}"
            )
        if arrays["showcase_mark_densities"].shape != (n_showcase, n_marks):
            raise ValueError(
                "showcase_mark_densities must have shape "
                f"({n_showcase}, {n_marks}); got {arrays['showcase_mark_densities'].shape}"
            )
        for name, array in arrays.items():
            if not np.all(np.isfinite(array)):
                raise ValueError(f"{name} must contain only finite values")
        for name in ("kl_divergence", "hpd_overlap", "observed_log_pred", "observed_mark"):
            if not np.isfinite(getattr(self, name)):
                raise ValueError(f"{name} must be finite")
        if self.kl_divergence < 0.0:
            raise ValueError(f"kl_divergence must be non-negative; got {self.kl_divergence}")
        if not 0.0 <= self.hpd_overlap <= 1.0:
            raise ValueError(f"hpd_overlap must lie in [0, 1]; got {self.hpd_overlap}")
        if not np.isfinite(self.predictive_pvalue) or not 0.0 <= self.predictive_pvalue <= 1.0:
            raise ValueError(f"predictive_pvalue must lie in [0, 1]; got {self.predictive_pvalue}")
        for name, array in arrays.items():
            array.setflags(write=False)
            object.__setattr__(self, name, array)

    @property
    def predictive_density(self) -> NDArray[np.float64]:
        """Predictive as a density over the state, shape (n_bins,)."""
        density: NDArray[np.float64] = self.predictive / (
            self.position_bins[1] - self.position_bins[0]
        )
        return density

    @property
    def likelihood_density(self) -> NDArray[np.float64]:
        """Normalized likelihood as a density over the state, shape (n_bins,)."""
        density: NDArray[np.float64] = self.likelihood / (
            self.position_bins[1] - self.position_bins[0]
        )
        return density


def create_shared_example(rng: np.random.Generator) -> Figure2ExampleData:
    """Create shared example data for all three metrics.

    All metrics use the same distributions. HPD overlap and KL divergence are
    computed directly; the predictive-check p-value is a fixed-seed Monte Carlo
    estimate.

    Parameters
    ----------
    rng : np.random.Generator
        Random number generator for reproducibility.

    Returns
    -------
    Figure2ExampleData
        Typed shared distributions, predictive-check samples, and precomputed
        diagnostic values used by all nine panels.
    """
    # Position grid
    n_bins = 200
    position_bins = np.linspace(0, 100, n_bins)
    # Predictive distribution: centered at 35, narrower (more certain)
    # Likelihood: centered at 60, broader (less certain)
    # Different widths make the min() in HPD overlap formula more meaningful
    pred_mean, pred_std = 35.0, 8.0
    like_mean, like_std = 60.0, 12.0

    predictive = norm.pdf(position_bins, loc=pred_mean, scale=pred_std)
    predictive = normalize(predictive)

    # The Gaussian has two distinct roles in the schematic. HPD/KL require a
    # likelihood normalized across state, while f_pred(y) requires the raw
    # observation density p(y | x). Keep both explicitly: normalizing p(y | x)
    # over x would add a y-dependent boundary correction.
    observed_conditional_density = norm.pdf(
        position_bins,
        loc=like_mean,
        scale=like_std,
    )
    likelihood = normalize(observed_conditional_density)

    # Compute KL divergence and HPD overlap using statespacecheck
    kl_divergence = float(
        ssc.kl_divergence(predictive[np.newaxis, :], likelihood[np.newaxis, :])[0]
    )
    hpd_overlap = float(
        ssc.hpd_overlap(
            predictive[np.newaxis, :], likelihood[np.newaxis, :], coverage=HPD_COVERAGE
        )[0]
    )

    # Monte Carlo predictive p-value (eq:fpred / eq:predictive_application):
    # each replicate draws a state from the event-weighted predictive
    # distribution, then an observation y_tilde ~ N(x_s, like_std^2), and
    # compares log f_pred(y_tilde) with log f_pred(y_obs). The total intensity
    # is 1 at every state because N(y; x, like_std) integrates to 1 over y, so
    # the event-weighted distribution equals ``predictive`` (the constant total
    # intensity the schematic assumes).
    model = ssc.MarkModel(
        log_intensity=lambda y: norm.logpdf(
            position_bins[np.newaxis, :], loc=y[:, :1], scale=like_std
        ),
        sample=lambda bins, g: g.normal(position_bins[bins], like_std)[:, np.newaxis],
        ground_intensity=np.ones(n_bins),
    )
    check = ssc.monte_carlo_mark_pvalue(
        predictive[np.newaxis, :],
        model,
        np.array([[like_mean]]),
        n_samples=10_000,
        rng=rng,
        return_samples=True,
    )
    predictive_pvalue = float(check.pvalue[0])
    observed_log_pred = float(check.observed_log_density[0])
    assert check.simulated_log_density is not None  # return_samples=True
    simulated_log_pred_values = check.simulated_log_density[0]

    # Showcase samples for the predictive-check schematic (the
    # ``predictive_marks`` and ``predictive_histogram`` panels).
    # State positions are picked at evenly-spaced quantiles of the predictive
    # CDF so the displayed fan spans the predictive's support rather than
    # clustering near the peak. For each state, a simulated observation
    # y_tilde is then drawn from p(y | x_s) = N(x_s, like_std^2) -- the same
    # step the Monte Carlo p-value takes.
    cumsum = np.cumsum(predictive)
    cumsum = cumsum / cumsum[-1]  # Ensure sums to 1
    showcase_quantiles = np.array([0.10, 0.50, 0.90])
    showcase_idx = np.minimum(np.searchsorted(cumsum, showcase_quantiles), n_bins - 1)
    showcase_positions = position_bins[showcase_idx]
    showcase_y_tildes = np.random.default_rng(SHOWCASE_MARK_SEED).normal(
        loc=showcase_positions, scale=like_std
    )

    # The mark axis, wide enough to hold nearly all of f_pred, which has its mode
    # at pred_mean and a standard deviation near 14. Each showcase state has its
    # own mark distribution p(y | x_s), and f_pred (eq:fpred) is p(y | x) averaged
    # over the predictive distribution of the state.
    mark_bins = np.linspace(-30.0, 100.0, 651)
    predictive_mark_density = (
        norm.pdf(mark_bins[:, np.newaxis], loc=position_bins, scale=like_std) @ predictive
    )
    showcase_mark_densities = norm.pdf(
        mark_bins[np.newaxis, :], loc=showcase_positions[:, np.newaxis], scale=like_std
    )

    return Figure2ExampleData(
        position_bins=position_bins,
        predictive=np.asarray(predictive, dtype=np.float64),
        likelihood=np.asarray(likelihood, dtype=np.float64),
        kl_divergence=kl_divergence,
        hpd_overlap=hpd_overlap,
        predictive_pvalue=predictive_pvalue,
        observed_log_pred=observed_log_pred,
        observed_mark=like_mean,
        simulated_log_pred=simulated_log_pred_values,
        showcase_positions=showcase_positions,
        showcase_y_tildes=showcase_y_tildes,
        mark_bins=mark_bins,
        predictive_mark_density=predictive_mark_density,
        showcase_mark_densities=showcase_mark_densities,
    )


# =============================================================================
# KL Divergence column
# =============================================================================


def _style_state_axis(ax: Axes) -> None:
    """Hide the top and right spines and mark only the ends of the latent-state axis."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlim(0, 100)
    ax.set_xticks([0, 100])
    ax.set_xticklabels(["0", "100"])


def _mark_round_values(axis: Axis) -> None:
    """Mark an axis at round values, each labeled with the value it sits at.

    The ticks follow later changes to the axis limits.
    """
    axis.set_major_locator(MaxNLocator(nbins=3, steps=[1, 2, 5, 10]))
    # Adding 0.0 turns a tick at -0.0 into 0.
    axis.set_major_formatter(FuncFormatter(lambda value, _: f"{value + 0.0:g}"))


def _finish_density_axes(ax: Axes) -> None:
    """Style a density panel: the state axis, then a y axis rising from zero."""
    _style_state_axis(ax)
    ax.set_ylim(bottom=0)
    _mark_round_values(ax.yaxis)


def plot_kl_distributions(ax: Axes, data: Figure2ExampleData) -> None:
    """Predictive and Likelihood distributions overlaid."""
    x = data.position_bins
    pred = data.predictive_density
    like = data.likelihood_density

    ax.plot(x, pred, color=COLORS["predictive"], linewidth=1.5, label="Predictive")
    ax.fill_between(x, pred, alpha=0.3, color=COLORS["predictive"])
    ax.plot(x, like, color=COLORS["likelihood"], linewidth=1.5, label="Likelihood")
    ax.fill_between(x, like, alpha=0.3, color=COLORS["likelihood"])

    ax.set_ylabel("Density", labelpad=8)
    ax.legend(
        frameon=False,
        loc="upper right",
        handlelength=1.0,
        handletextpad=0.4,
        labelspacing=0.25,
        borderaxespad=0.0,
    )
    _finish_density_axes(ax)


def plot_kl_log_ratio(ax: Axes, data: Figure2ExampleData) -> None:
    """Log ratio = log(Predictive / Likelihood)."""
    x = data.position_bins
    pred = data.predictive
    like = data.likelihood

    log_ratio = np.log(pred) - np.log(like)

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

    ax.set_ylabel(r"$\log P - \log Q$", labelpad=8)
    ax.set_title("Log ratio", pad=4)
    _style_state_axis(ax)
    _mark_round_values(ax.yaxis)


def plot_kl_pointwise(ax: Axes, data: Figure2ExampleData) -> None:
    """Pointwise KL = Predictive * log(Pred/Lik)."""
    x = data.position_bins
    pred = data.predictive
    like = data.likelihood

    log_ratio = np.log(pred) - np.log(like)
    # The density, not the per-bin probability, so the area is the divergence.
    pointwise_kl = data.predictive_density * log_ratio

    ax.plot(x, pointwise_kl, color=COLORS["kl_divergence"], linewidth=1.5)
    ax.fill_between(x, pointwise_kl, alpha=0.3, color=COLORS["kl_divergence"])
    ax.axhline(0, color=COLORS["reference"], linestyle="--", linewidth=0.8, alpha=0.7)

    ax.set_xlabel("Latent state (a.u.)", labelpad=8)
    ax.set_ylabel(
        r"$P \cdot (\log P - \log Q)$",
        labelpad=8,
    )
    ax.set_title(r"Pointwise KL: $P\,\times$ log ratio", pad=4)
    _style_state_axis(ax)
    _mark_round_values(ax.yaxis)


# =============================================================================
# HPD Overlap column
# =============================================================================


def _plot_hpd_panel(
    ax: Axes,
    x: NDArray[np.float64],
    dist: NDArray[np.float64],
    *,
    color: str,
    title: str,
    label_x: float,
    label_ha: Literal["left", "right"],
    legend_kwargs: dict[str, Any],
) -> None:
    """Render one 95% HPD distribution panel (predictive or likelihood column).

    Draws the distribution line, shades the 95% HPD region under it, marks the
    HPD density level with a dashed line + label, and applies the shared
    Figure-2 panel styling. Only the color, title, level-label placement,
    and legend keywords differ between the two columns.
    """
    hpd_mask = ssc.highest_density_region(dist[np.newaxis], coverage=HPD_COVERAGE)[0]

    density = dist / (x[1] - x[0])
    # The HPD density level is the minimum density value inside the HPD region.
    hpd_density_level = np.min(density[hpd_mask])

    ax.plot(x, density, color=color, linewidth=1.2)
    ax.fill_between(
        x,
        0,
        density,
        where=list(hpd_mask),
        alpha=0.35,
        color=color,
        label=f"{HPD_COVERAGE:.0%} HPD",
    )
    ax.axhline(hpd_density_level, color=color, linestyle="--", linewidth=0.8, alpha=0.7)
    # Lift the label off the line: the curve leans away from it as it rises.
    ax.annotate(
        f"{HPD_COVERAGE:.0%}\nlevel",
        xy=(label_x, hpd_density_level),
        xytext=(0, 3),
        textcoords="offset points",
        ha=label_ha,
        va="bottom",
        color=color,
        alpha=0.8,
    )

    ax.set_ylabel("Density", labelpad=8)
    ax.set_title(title, pad=4)
    ax.legend(frameon=False, **legend_kwargs)
    _finish_density_axes(ax)


def plot_hpd_predictive(ax: Axes, data: Figure2ExampleData) -> None:
    """Predictive distribution with 95% HPD region shaded."""
    _plot_hpd_panel(
        ax,
        data.position_bins,
        data.predictive,
        color=COLORS["predictive"],
        title=r"Predictive HPD ($\mathrm{HPD}_P$)",
        label_x=99,
        label_ha="right",
        legend_kwargs={"loc": "upper right"},
    )


def plot_hpd_likelihood(ax: Axes, data: Figure2ExampleData) -> None:
    """Likelihood distribution with 95% HPD region shaded."""
    # Curve peaks center-right, so the level label sits on the left and the
    # legend is tucked into the empty upper-left corner against the y-axis.
    _plot_hpd_panel(
        ax,
        data.position_bins,
        data.likelihood,
        color=COLORS["likelihood"],
        title=r"Likelihood HPD ($\mathrm{HPD}_Q$)",
        label_x=1,
        label_ha="left",
        legend_kwargs={
            "loc": "upper left",
            "handlelength": 1.2,
            "handletextpad": 0.4,
            "borderaxespad": 0.2,
        },
    )


def plot_hpd_intersection(ax: Axes, data: Figure2ExampleData) -> tuple[float, float, float]:
    """HPD intersection region.

    Shows HPD regions as thick horizontal lines at different y-levels.
    Order: Pred HPD (top), Lik HPD (middle), Overlap (bottom) to match panels above.
    Labels use descriptive text with formula notation in parentheses.
    Size values are displayed on each bar for the calculation breakdown.

    Returns
    -------
    tuple[float, float, float]
        (pred_hpd_size, like_hpd_size, intersection_size) for use in formula.
    """
    x = data.position_bins
    pred = data.predictive
    like = data.likelihood
    dx = x[1] - x[0]

    pred_hpd = ssc.highest_density_region(pred[np.newaxis], coverage=HPD_COVERAGE)[0]
    like_hpd = ssc.highest_density_region(like[np.newaxis], coverage=HPD_COVERAGE)[0]
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

    ax.text(
        pred_center,
        0.84,
        rf"$|\mathrm{{HPD}}_P|$ = {pred_hpd_size:.1f}",
        ha="center",
        va="bottom",
        color=COLORS["predictive"],
    )
    ax.text(
        like_center,
        0.54,
        rf"$|\mathrm{{HPD}}_Q|$ = {like_hpd_size:.1f}",
        ha="center",
        va="bottom",
        color=COLORS["likelihood"],
    )
    # This is the longest label, so it ends at the axis instead of centering on
    # its bar, which would carry it into the row label.
    ax.text(
        100,
        0.24,
        rf"$|\mathrm{{HPD}}_P \cap \mathrm{{HPD}}_Q|$ = {intersection_size:.1f}",
        ha="right",
        va="bottom",
        color=COLORS["hpd_overlap"],
    )

    ax.set_xlabel("Latent state (a.u.)", labelpad=8)
    _style_state_axis(ax)
    ax.spines["left"].set_visible(False)
    ax.set_ylim(0, 1.0)
    ax.set_yticks([0.8, 0.5, 0.2], labels=["Pred. HPD", "Like. HPD", "Intersection"])
    ax.tick_params(axis="y", length=0)  # Hide tick marks

    return pred_hpd_size, like_hpd_size, intersection_size


# =============================================================================
# Predictive Check column
# =============================================================================

# A dot marks a sampled state in step 1 and a triangle marks a mark in every step.
_KEY_GRAY = "0.4"
_STATE_DOT: dict[str, Any] = {
    "marker": "o",
    "s": 20,
    "edgecolors": "white",
    "linewidths": 0.4,
    "zorder": 6,
}
_TRIANGLE: dict[str, Any] = {
    "marker": "v",
    "s": 30,
    "edgecolors": "white",
    "linewidths": 0.4,
    "zorder": 5,
}


def _showcase_colors(n: int) -> NDArray[np.float64]:
    """Distinct colors for the predictive-check sample fan.

    Sampled from viridis, avoiding the very dark end and the pale yellow-greens, so
    every curve stays legible at small panel sizes.
    """
    cmap = plt.get_cmap("viridis")
    return cmap(np.linspace(0.15, 0.7, n))


def plot_predictive_check_predictive_fan(ax: Axes, data: Figure2ExampleData) -> None:
    """Predictive distribution with a fan of showcase positions.

    Each colored marker is a state at a fixed quantile of the predictive
    (the 10th, 50th, and 90th percentiles, chosen to span its
    support rather than drawn at random); the simulated observation drawn at
    that state gives the matching curve in the mark-density panel.
    """
    x = data.position_bins
    pred = data.predictive_density
    positions = data.showcase_positions
    colors = _showcase_colors(len(positions))

    predictive_line = ax.plot(x, pred, color=COLORS["predictive"], linewidth=1.5)[0]
    ax.fill_between(x, pred, alpha=0.3, color=COLORS["predictive"])

    # Each showcase state is shown as a colored tick + dot on the predictive
    # so the matching curve in the mark-density panel can be read off by colour.
    sample_indices = np.argmin(np.abs(x[None, :] - positions[:, None]), axis=1)
    for pos, idx, color in zip(positions, sample_indices, colors, strict=True):
        ax.vlines(pos, 0, pred[idx], color=color, linestyle=":", linewidth=1.0)
        ax.scatter([pos], [pred[idx]], color=color, s=22, zorder=5)
    # Gray keys name the kind of thing; the colors tell the samples apart. The
    # titles spell out the symbols, which keeps these keys short enough to sit
    # beside the curves instead of above them.
    _compact_legend(
        ax,
        [predictive_line, ax.scatter([], [], color=_KEY_GRAY, **_STATE_DOT)],
        ["Pred.", "$x_k$"],
    )

    ax.set_title("Sample states $x_k$", pad=3)
    ax.set_xlabel("Latent state (a.u.)", labelpad=2)
    ax.set_ylabel("Density", labelpad=8)
    _finish_density_axes(ax)


def _style_mark_axis(ax: Axes, top: float) -> None:
    """Style a mark-density panel. Its axis matches the state axis of step 1.

    A mark here is a state plus noise, so the two share a scale: f_pred sits under
    the predictive, only wider. The far left tail of f_pred, below 0, falls off
    the axis.
    """
    _style_state_axis(ax)
    ax.set_ylim(0, top)
    _mark_round_values(ax.yaxis)
    ax.set_ylabel("Density", labelpad=8)


def _compact_legend(ax: Axes, keys: list[Any], labels: list[str]) -> None:
    """Add a compact upper-right legend for a panel of the predictive column.

    Small key markers keep a key from reading as one more point on the curves
    beside it. A frame would hide the curves that rise into this corner.
    """
    ax.legend(
        keys,
        labels,
        loc="upper right",
        frameon=False,
        markerscale=0.7,
        labelspacing=0.25,
        handlelength=0.8,
        handletextpad=0.3,
        borderaxespad=0.0,
    )


def plot_predictive_check_replicate_marks(ax: Axes, data: Figure2ExampleData) -> None:
    """Step 2: each sampled state's observation model and the mark drawn from it.

    A dotted line is a sampled state, directly below its line in step 1, rising
    to the peak of the observation model p(y | x_s) that state sets. The triangle
    on each curve is the replicate mark drawn from that curve. Color follows the
    sample, and the curves are light so that the markers stand out.
    """
    marks = data.mark_bins
    colors = _showcase_colors(len(data.showcase_positions))
    curve: dict[str, Any] = {"linewidth": 0.8, "alpha": 0.6}
    for density, y_tilde, color in zip(
        data.showcase_mark_densities, data.showcase_y_tildes, colors, strict=True
    ):
        ax.plot(marks, density, color=color, **curve)
        ax.scatter([y_tilde], [np.interp(y_tilde, marks, density)], color=[color], **_TRIANGLE)
    ax.vlines(
        data.showcase_positions,
        0,
        data.showcase_mark_densities.max(axis=1),
        colors=[tuple(color) for color in colors],
        linestyle=":",
        linewidth=1.0,
    )

    _compact_legend(
        ax,
        [
            Line2D([], [], color=_KEY_GRAY, linewidth=1.2),
            ax.scatter([], [], color=_KEY_GRAY, **_TRIANGLE),
        ],
        [r"$p(y\,|\,x_k)$", r"$\tilde{y}$"],
    )
    ax.set_title(r"Sample a mark $\tilde{y}$ from each state", pad=3)
    _style_mark_axis(ax, top=0.041)
    # The panel below shares this axis.
    ax.tick_params(labelbottom=False)


def plot_predictive_check_mark_scores(ax: Axes, data: Figure2ExampleData) -> None:
    """Step 3: every mark scored by its height on the predictive mark density.

    ``f_pred`` averages the curves of step 2 over the predictive states. It gives
    each mark one score, its height: the replicate marks of step 2 sit on it as
    the same triangles. No state appears here: the score depends only on the mark.
    """
    marks = data.mark_bins
    f_pred = data.predictive_mark_density
    colors = _showcase_colors(len(data.showcase_positions))
    observed_density = np.exp(data.observed_log_pred)

    ax.plot(marks, f_pred, color=COLORS["predictive"], linewidth=1.5)
    ax.scatter(
        data.showcase_y_tildes,
        np.interp(data.showcase_y_tildes, marks, f_pred),
        color=colors,
        **_TRIANGLE,
    )
    observed = ax.scatter(
        [data.observed_mark], [observed_density], color=COLORS["likelihood"], **_TRIANGLE
    )

    # The replicates were introduced in step 2.
    _compact_legend(ax, [observed], ["Observed $y$"])
    ax.set_title(r"Score each mark on $f_{\mathrm{pred}}$", pad=3)
    ax.set_xlabel("Mark (a.u.)", labelpad=2)
    _style_mark_axis(ax, top=0.039)


def plot_predictive_check_density_histogram(ax: Axes, data: Figure2ExampleData) -> None:
    """Histogram of observed vs simulated log predictive density.

    Uses the exact Monte Carlo samples computed in create_shared_example().
    The simulated values have a long, sparse left tail (replicated marks far
    from the predictive), so the axis shows the samples above their
    ``HISTOGRAM_LOWER_QUANTILE`` quantile, and always the observed value. Bar
    heights are densities over all samples, so the bars' total area is the
    fraction of samples shown.
    """
    # Use exact values from Monte Carlo simulation
    simulated_log_pred = data.simulated_log_pred
    observed_log_pred = data.observed_log_pred

    n_bins = 30
    lower = min(float(np.quantile(simulated_log_pred, HISTOGRAM_LOWER_QUANTILE)), observed_log_pred)
    upper = max(float(np.max(simulated_log_pred)), observed_log_pred)
    bin_width = (upper - lower) / n_bins

    # Histogram of simulated values (from predictive distribution)
    ax.hist(
        simulated_log_pred,
        bins=n_bins,
        range=(lower, upper),
        weights=np.full(simulated_log_pred.size, 1.0 / (simulated_log_pred.size * bin_width)),
        alpha=0.5,
        color=COLORS["predictive"],
        edgecolor="none",
    )

    # Mark observed value (from likelihood/actual observation)
    observed_line = ax.axvline(
        observed_log_pred,
        color=COLORS["likelihood"],
        linewidth=2,
        linestyle="-",
    )
    # Both keys are lines, so the legend reads as one kind of entry.
    replicates_key = Line2D([], [], color=COLORS["predictive"], linewidth=2, alpha=0.5)

    ax.set_title("Rank the observed score", pad=3)
    ax.set_xlabel("Log pred. density", labelpad=2)
    ax.set_ylabel("Density", labelpad=8)
    # A short handle keeps the legend's right edge left of the observed value.
    ax.legend(
        [replicates_key, observed_line],
        ["Replicates", "Observed"],
        frameon=False,
        loc="upper left",
        handlelength=1.0,
        handletextpad=0.3,
        borderaxespad=0.0,
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    _mark_round_values(ax.xaxis)
    _mark_round_values(ax.yaxis)
