"""Create Figure 1: Combined state space model schematic and distribution consistency.

This figure combines:
- Panel a: State space model schematic showing graphical model and filtering equations
- Panels b-e: Distribution comparison showing consistent/inconsistent scenarios

The schematic demonstrates the two-step Bayesian filtering process:
1. Prediction: Convolve previous posterior with transition to get predictive
2. Update: Multiply predictive with likelihood to get current posterior

The distribution panels show scenarios where predictive and normalized likelihood
are consistent or inconsistent, using HPD overlap as a diagnostic.
"""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.layout_engine import ConstrainedLayoutEngine
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy import stats

from statespacecheck_paper.plotting import compute_hpd_region
from statespacecheck_paper.style import COLORS, save_figure, set_figure_defaults

# =============================================================================
# Schematic Drawing Functions
# =============================================================================


def _draw_node(
    ax: Axes,
    center: tuple[float, float],
    radius: float,
    label: str,
    facecolor: str = "white",
    edgecolor: str = "black",
    linewidth: float = 1.5,
) -> Circle:
    """Draw a circular node for the graphical model."""
    circle = Circle(
        center,
        radius,
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=linewidth,
        zorder=10,
    )
    ax.add_patch(circle)

    ax.text(
        center[0],
        center[1],
        label,
        ha="center",
        va="center",
        fontsize=9,
        fontweight="bold",
        zorder=11,
    )
    return circle


def _draw_arrow(
    ax: Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    label: str | None = None,
    color: str = "black",
    linewidth: float = 1.5,
    connectionstyle: str = "arc3,rad=0",
) -> FancyArrowPatch:
    """Draw an arrow between nodes."""
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=12,
        color=color,
        linewidth=linewidth,
        connectionstyle=connectionstyle,
        zorder=5,
    )
    ax.add_patch(arrow)

    if label:
        mid_x = (start[0] + end[0]) / 2
        mid_y = (start[1] + end[1]) / 2
        ax.text(
            mid_x,
            mid_y + 0.15,
            label,
            ha="center",
            va="bottom",
            fontsize=7,
            style="italic",
        )

    return arrow


def _draw_distribution_inset(
    ax: Axes,
    center: tuple[float, float],
    width: float,
    height: float,
    mean: float,
    std: float,
    color: str,
    label: str | None = None,
    label_color: str | None = None,
    label_size: int = 6,
    title: str | None = None,
    title_size: int = 6,
) -> None:
    """Draw a small distribution plot as an inset.

    Parameters
    ----------
    ax : plt.Axes
        The axes to draw on.
    center : tuple[float, float]
        Center position in data coordinates.
    width, height : float
        Size of the inset in data coordinates.
    mean, std : float
        Parameters for the Gaussian distribution.
    color : str
        Color for the distribution curve and fill.
    label : str, optional
        Label to display below the distribution (e.g., math notation).
    label_color : str, optional
        Color for the label. Defaults to distribution color.
    label_size : int, default 6
        Font size for the label below.
    title : str, optional
        Title to display above the distribution (e.g., "Previous Posterior").
    title_size : int, default 6
        Font size for the title above.
    """
    if label_color is None:
        label_color = color

    left_data = center[0] - width / 2
    bottom_data = center[1] - height / 2

    inset = inset_axes(
        ax,
        width="100%",
        height="100%",
        bbox_to_anchor=(left_data, bottom_data, width, height),
        bbox_transform=ax.transData,
        borderpad=0,
    )

    x = np.linspace(mean - 3.5 * std, mean + 3.5 * std, 100)
    y = stats.norm.pdf(x, mean, std)

    inset.plot(x, y, color=color, linewidth=1.2)
    inset.fill_between(x, y, alpha=0.3, color=color)

    inset.set_xlim(mean - 3.5 * std, mean + 3.5 * std)
    inset.set_ylim(0, y.max() * 1.1)
    inset.axis("off")

    if title:
        inset.text(
            0.5,
            1.15,
            title,
            ha="center",
            va="bottom",
            transform=inset.transAxes,
            fontsize=title_size,
            color=color,
        )

    if label:
        inset.text(
            0.5,
            -0.15,
            label,
            ha="center",
            va="top",
            transform=inset.transAxes,
            fontsize=label_size,
            color=label_color,
        )


def _draw_spikes_inset(
    ax: Axes,
    center: tuple[float, float],
    width: float,
    height: float,
    n_cells: int = 5,
    rng: np.random.Generator | None = None,
    label: str = "Spikes",
) -> None:
    """Draw a small spike raster as an inset.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes to draw on.
    center : tuple[float, float]
        Center position in data coordinates.
    width, height : float
        Size of the inset in data coordinates.
    n_cells : int, default 5
        Number of cells to show in raster.
    rng : np.random.Generator, optional
        Random number generator. Defaults to seed 42.
    label : str, default "Spikes"
        Label to display below the raster.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    left_data = center[0] - width / 2
    bottom_data = center[1] - height / 2

    inset = inset_axes(
        ax,
        width="100%",
        height="100%",
        bbox_to_anchor=(left_data, bottom_data, width, height),
        bbox_transform=ax.transData,
        borderpad=0,
    )

    spike_data = []
    for _ in range(n_cells):
        n_spikes = rng.integers(2, 8)
        spike_times = rng.uniform(0, 1, n_spikes)
        spike_data.append(spike_times)

    inset.eventplot(
        spike_data,
        lineoffsets=range(n_cells),
        linelengths=0.6,
        linewidths=0.8,
        colors="black",
    )

    inset.set_xlim(-0.1, 1.1)
    inset.set_ylim(-0.5, n_cells - 0.5)
    inset.axis("off")

    inset.text(
        0.5,
        -0.2,
        label,
        ha="center",
        va="top",
        transform=inset.transAxes,
        fontsize=6,
    )


def _draw_equation_box(
    ax: Axes,
    center: tuple[float, float],
    width: float,
    height: float,
    edgecolor: str = "#666666",
    facecolor: str = "#FAFAFA",
    linewidth: float = 1.0,
) -> FancyBboxPatch:
    """Draw a box for equation grouping."""
    box = FancyBboxPatch(
        (center[0] - width / 2, center[1] - height / 2),
        width,
        height,
        boxstyle="round,pad=0.05",
        edgecolor=edgecolor,
        facecolor=facecolor,
        linewidth=linewidth,
        zorder=1,
    )
    ax.add_patch(box)
    return box


def _draw_graphical_model(
    ax: Axes,
    rng: np.random.Generator | None = None,
) -> None:
    """Draw the graphical model portion of the schematic.

    This function draws:
    - Chain of latent states: ... -> x_{t-1} -> x_t -> ...
    - Observations: y_{t-1}, y_t
    - Spike raster insets below observations

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes to draw on.
    rng : np.random.Generator, optional
        Random number generator for spike insets. Defaults to seed 42.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    ax.set_xlim(-0.5, 7.5)  # Reduced from 9.5 - content ends around 7
    ax.set_ylim(2.5, 6.65)  # Content tops at 6.6, minimal padding
    ax.axis("off")

    node_radius = 0.38
    y_latent = 5.5
    y_obs = y_latent - 1.3

    x_prev_pos = (2.5, y_latent)
    x_curr_pos = (5.0, y_latent)
    y_prev_obs_pos = (x_prev_pos[0], y_obs)
    y_curr_obs_pos = (x_curr_pos[0], y_obs)

    # Ellipsis to indicate chain continues to the left
    ax.text(
        x_prev_pos[0] - 1.2,
        y_latent,
        r"$\cdots$",
        ha="center",
        va="center",
        fontsize=14,
    )

    # Arrow from ellipsis to x_{t-1}
    arrow_start = (x_prev_pos[0] - 0.9, y_latent)
    arrow_end = (x_prev_pos[0] - node_radius - 0.05, y_latent)
    _draw_arrow(ax, arrow_start, arrow_end, color="black")

    # Draw state nodes
    _draw_node(ax, x_prev_pos, node_radius, r"$x_{t-1}$", edgecolor="black")
    _draw_node(ax, x_curr_pos, node_radius, r"$x_t$", edgecolor="black")

    # Draw observation nodes (filled to indicate observed)
    _draw_node(
        ax, y_prev_obs_pos, node_radius, r"$y_{t-1}$", facecolor="lightgray", edgecolor="black"
    )
    _draw_node(ax, y_curr_obs_pos, node_radius, r"$y_t$", facecolor="lightgray", edgecolor="black")

    # Horizontal arrow: x_{t-1} -> x_t
    arrow_start = (x_prev_pos[0] + node_radius + 0.05, x_prev_pos[1])
    arrow_end = (x_curr_pos[0] - node_radius - 0.05, x_curr_pos[1])
    _draw_arrow(ax, arrow_start, arrow_end, color="black")

    # Transition label above the arrow
    transition_label_y = y_latent + 0.32
    ax.text(
        (x_prev_pos[0] + x_curr_pos[0]) / 2,
        transition_label_y + 0.02,
        "Transition",
        ha="center",
        va="bottom",
        fontsize=6,
        color="#666666",
    )
    ax.text(
        (x_prev_pos[0] + x_curr_pos[0]) / 2,
        transition_label_y - 0.02,
        r"$p(x_t|x_{t-1})$",
        ha="center",
        va="top",
        fontsize=6,
        color="#666666",
    )

    # Downward arrows: x -> y
    arrow_start = (x_prev_pos[0], x_prev_pos[1] - node_radius - 0.05)
    arrow_end = (y_prev_obs_pos[0], y_prev_obs_pos[1] + node_radius + 0.05)
    _draw_arrow(ax, arrow_start, arrow_end, color="black")

    arrow_start = (x_curr_pos[0], x_curr_pos[1] - node_radius - 0.05)
    arrow_end = (y_curr_obs_pos[0], y_curr_obs_pos[1] + node_radius + 0.05)
    _draw_arrow(ax, arrow_start, arrow_end, color="black")

    # Likelihood label
    arrow_midpoint_y = (x_curr_pos[1] + y_curr_obs_pos[1]) / 2
    ax.text(
        x_curr_pos[0] + 0.15,
        arrow_midpoint_y,
        "Likelihood\n" + r"$p(y_t|x_t)$",
        ha="left",
        va="center",
        fontsize=6,
        color=COLORS["likelihood"],
        linespacing=1.0,
    )

    # Arrow from x_t to ellipsis (chain continues to the right)
    arrow_start = (x_curr_pos[0] + node_radius + 0.05, y_latent)
    arrow_end = (x_curr_pos[0] + 0.9, y_latent)
    _draw_arrow(ax, arrow_start, arrow_end, color="black")

    # Ellipsis to indicate chain continues to the right
    ax.text(
        x_curr_pos[0] + 1.2,
        y_latent,
        r"$\cdots$",
        ha="center",
        va="center",
        fontsize=14,
    )

    # Labels on the left
    ax.text(
        0.3,
        y_latent,
        "Latent\nstates",
        ha="left",
        va="center",
        fontsize=7,
        fontstyle="italic",
    )
    ax.text(
        0.3,
        y_obs,
        "Observations",
        ha="left",
        va="center",
        fontsize=7,
        fontstyle="italic",
    )

    # Labels above nodes
    ax.text(
        x_prev_pos[0],
        y_latent + 0.45,
        "Previous\nPosterior",
        ha="center",
        va="bottom",
        fontsize=6,
        color=COLORS["posterior"],
    )
    ax.text(
        x_curr_pos[0],
        y_latent + 0.45,
        "Current\nPosterior",
        ha="center",
        va="bottom",
        fontsize=6,
        color=COLORS["posterior"],
    )

    # Spike rasters below observations
    _draw_spikes_inset(
        ax,
        center=(y_prev_obs_pos[0], y_prev_obs_pos[1] - 0.65),
        width=0.7,
        height=0.35,
        n_cells=5,
        rng=rng,
        label="Previous\nSpikes",
    )
    _draw_spikes_inset(
        ax,
        center=(y_curr_obs_pos[0], y_curr_obs_pos[1] - 0.65),
        width=0.7,
        height=0.35,
        n_cells=5,
        rng=rng,
        label="Current\nSpikes",
    )

    # Title
    ax.text(
        (x_prev_pos[0] + x_curr_pos[0]) / 2,
        6.6,
        "State Space Model",
        ha="center",
        va="bottom",
        fontsize=8,
        fontweight="bold",
    )

    # Time direction indicator
    ax.text(
        x_curr_pos[0] + 1.5,
        y_obs - 0.85,
        r"Time $\rightarrow$",
        ha="left",
        va="center",
        fontsize=6,
        fontstyle="italic",
        color="#666666",
    )


def _draw_equation_boxes(ax: Axes) -> None:
    """Draw the equation boxes (Prediction and Update steps).

    This function draws:
    - Equation box 1: Previous Posterior * Transition = Predictive
    - Equation box 2: Predictive x Likelihood = Current Posterior
    - Connecting bracket with "Each time t" label

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes to draw on.
    """
    ax.set_xlim(-0.5, 7.5)  # Reduced from 9.5 - content ends around 7.15
    ax.set_ylim(-0.95, 2.35)  # Reduced from 2.6 - content tops at ~2.3
    ax.axis("off")

    # ==========================================================================
    # EQUATION 1: [Prev Posterior] * T = [Predictive]
    # ==========================================================================

    y_eq1 = 1.4
    box_height = 1.4

    # Box dimensions aligned with graphical model
    box_right_edge = 7.15
    box_left_edge = 0.0
    box_width = box_right_edge - box_left_edge
    box_center_x = (box_left_edge + box_right_edge) / 2
    _draw_equation_box(
        ax,
        center=(box_center_x, y_eq1 + 0.2),
        width=box_width,
        height=box_height,
        edgecolor="#CCCCCC",
        facecolor="#F9F9F9",
    )

    # Offset to shift equation elements right to make room for labels
    eq_offset = 0.8

    # Distribution 1: Previous Posterior
    _draw_distribution_inset(
        ax,
        center=(1.3 + eq_offset, y_eq1 + 0.2),
        width=0.9,
        height=0.5,
        mean=40,
        std=8,
        color=COLORS["posterior"],
        label=r"$p(x_{t-1}|y_{1:t-1})$",
        label_size=5,
        title="Previous\nPosterior",
    )

    # Operation symbol: convolution
    ax.text(
        2.3 + eq_offset,
        y_eq1 + 0.2,
        r"$\circledast$",
        ha="center",
        va="center",
        fontsize=16,
        fontweight="bold",
    )

    # Transition distribution (gray for fixed model component)
    _draw_distribution_inset(
        ax,
        center=(3.3 + eq_offset, y_eq1 + 0.2),
        width=0.9,
        height=0.5,
        mean=45,
        std=10,
        color="#666666",
        label=r"$p(x_t|x_{t-1})$",
        label_size=5,
        title="Transition",
    )

    # Equals
    ax.text(
        4.3 + eq_offset,
        y_eq1 + 0.2,
        "=",
        ha="center",
        va="center",
        fontsize=14,
    )

    # Distribution 3: Predictive
    _draw_distribution_inset(
        ax,
        center=(5.5 + eq_offset, y_eq1 + 0.2),
        width=0.9,
        height=0.5,
        mean=45,
        std=12,
        color=COLORS["predictive"],
        label=r"$p(x_t|y_{1:t-1})$",
        label_size=5,
        title="Predictive\nDistribution",
    )

    # Step label
    ax.text(
        0.2,
        y_eq1 + 0.2,
        "1. Prediction",
        ha="left",
        va="center",
        fontsize=7,
        color=COLORS["predictive"],
        fontweight="bold",
    )

    # ==========================================================================
    # EQUATION 2: [Predictive] x [Likelihood] = [Current Posterior]
    # ==========================================================================

    y_eq2 = -0.25
    box_height_eq2 = 1.4

    _draw_equation_box(
        ax,
        center=(box_center_x, y_eq2 + 0.2),
        width=box_width,
        height=box_height_eq2,
        edgecolor="#CCCCCC",
        facecolor="#F9F9F9",
    )

    # Distribution 1: Predictive (repeated from equation 1)
    _draw_distribution_inset(
        ax,
        center=(1.3 + eq_offset, y_eq2 + 0.2),
        width=0.9,
        height=0.5,
        mean=45,
        std=12,
        color=COLORS["predictive"],
        label=r"$p(x_t|y_{1:t-1})$",
        label_size=5,
        title="Predictive\nDistribution",
    )

    # Operation symbol: multiplication
    ax.text(
        2.3 + eq_offset,
        y_eq2 + 0.2,
        r"$\times$",
        ha="center",
        va="center",
        fontsize=14,
        fontweight="bold",
    )

    # Distribution 2: Likelihood
    _draw_distribution_inset(
        ax,
        center=(3.3 + eq_offset, y_eq2 + 0.2),
        width=0.9,
        height=0.5,
        mean=50,
        std=10,
        color=COLORS["likelihood"],
        label=r"$p(x_t|y_t)$",
        label_size=5,
        title="Likelihood",
    )

    # Equals
    ax.text(
        4.3 + eq_offset,
        y_eq2 + 0.2,
        "=",
        ha="center",
        va="center",
        fontsize=14,
    )

    # Distribution 3: Current Posterior
    _draw_distribution_inset(
        ax,
        center=(5.5 + eq_offset, y_eq2 + 0.2),
        width=0.9,
        height=0.5,
        mean=48,
        std=7,
        color=COLORS["posterior"],
        label=r"$p(x_t|y_{1:t})$",
        label_size=5,
        title="Current\nPosterior",
    )

    # Step label
    ax.text(
        0.2,
        y_eq2 + 0.2,
        "2. Update",
        ha="left",
        va="center",
        fontsize=7,
        color=COLORS["posterior"],
        fontweight="bold",
    )

    # ==========================================================================
    # Bracket with arrow to show flow from step 1 to step 2 at each time step
    # ==========================================================================

    bracket_x = -0.4
    bracket_top = y_eq1 + 0.2
    bracket_bottom = y_eq2 + 0.2
    bracket_mid = (bracket_top + bracket_bottom) / 2

    bracket_width = 0.25
    line_end = 0.20
    arrow_length = 0.35
    ax.plot(
        [bracket_x + bracket_width, bracket_x, bracket_x, bracket_x + line_end],
        [bracket_top, bracket_top, bracket_bottom, bracket_bottom],
        color="#666666",
        linewidth=1.0,
        solid_capstyle="round",
        solid_joinstyle="round",
    )
    _draw_arrow(
        ax,
        start=(bracket_x + line_end, bracket_bottom),
        end=(bracket_x + arrow_length, bracket_bottom),
        color="#666666",
        linewidth=1.0,
    )

    # Label for the bracket
    ax.text(
        bracket_x - 0.1,
        bracket_mid,
        "Each\ntime $t$",
        ha="right",
        va="center",
        fontsize=6,
        fontweight="bold",
        color="#666666",
    )


# =============================================================================
# Distribution Panel Functions
# =============================================================================


def _extract_contiguous_regions(mask: np.ndarray, x: np.ndarray) -> list[tuple[float, float]]:
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


def _create_distribution_panel(
    ax: Axes,
    x: np.ndarray,
    scenario: dict[str, Any],
    color_predictive: str,
    color_likelihood: str,
    show_direct_labels: bool = False,
) -> None:
    """Create a single panel showing predictive and likelihood distributions.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to plot on.
    x : np.ndarray, shape (n_points,)
        Position values for plotting.
    scenario : dict
        Dictionary with 'title', 'predictive', and 'likelihood' keys.
    color_predictive : str
        Color for predictive distribution.
    color_likelihood : str
        Color for likelihood distribution.
    show_direct_labels : bool, default False
        Whether to show direct labels on the distribution curves.
    """
    # Generate distributions
    pred_mean, pred_std = scenario["predictive"]
    like_mean, like_std = scenario["likelihood"]

    pdf_predictive = stats.norm.pdf(x, loc=pred_mean, scale=pred_std)
    pdf_likelihood = stats.norm.pdf(x, loc=like_mean, scale=like_std)

    # Normalize likelihood
    dx = x[1] - x[0]
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
    hpd_predictive = compute_hpd_region(x, pdf_predictive, coverage=0.95)
    hpd_likelihood = compute_hpd_region(x, pdf_likelihood, coverage=0.95)
    pred_regions = _extract_contiguous_regions(hpd_predictive, x)
    like_regions = _extract_contiguous_regions(hpd_likelihood, x)

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
    ax.set_xlim(-20, 20)
    ax.set_ylim(-0.1, 0.26)  # Tighter to match other panels' label spacing
    ax.set_title(scenario["title"], fontsize=7, fontweight="normal", pad=2)

    ax.axis("off")

    # Add direct labels on distribution curves
    if show_direct_labels:
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


# =============================================================================
# Main Figure Creation
# =============================================================================


def create_figure() -> None:
    """Create Figure 1 combining schematic and distribution panels.

    This figure combines:
    - Panel a: State space model graphical model (top)
    - Panel b: Equation boxes (Prediction and Update steps)
    - Panel c: Distribution consistency examples (4 sub-panels, bottom)

    Returns
    -------
    None
        Saves figure to figures/main/figure01.{pdf,png}.
    """
    set_figure_defaults(context="paper")

    # Create figure with GridSpec for precise control
    # 3 rows: graphical model, equation boxes, distribution panels
    fig: Figure = plt.figure(figsize=(5.0, 5.8), dpi=450, constrained_layout=True)
    layout_engine = fig.get_layout_engine()
    if isinstance(layout_engine, ConstrainedLayoutEngine):
        layout_engine.set(w_pad=0.01, h_pad=0.02, wspace=0.01, hspace=0.02)

    # Create grid: 3 rows, 4 columns (no margin column needed)
    gs = fig.add_gridspec(
        3,
        4,
        height_ratios=[0.6, 0.7, 0.35],
        width_ratios=[1, 1, 1, 1],
    )

    # Panel A: Graphical model spans all columns in top row
    axes = {}
    axes["A"] = fig.add_subplot(gs[0, :])

    # Panel B: Equation boxes spans all columns in middle row
    axes["B"] = fig.add_subplot(gs[1, :])

    # Panel C (sub-panels): Distribution panels
    axes["C1"] = fig.add_subplot(gs[2, 0])
    axes["C2"] = fig.add_subplot(gs[2, 1])
    axes["C3"] = fig.add_subplot(gs[2, 2])
    axes["C4"] = fig.add_subplot(gs[2, 3])

    # =========================================================================
    # Panel A: Graphical model
    # =========================================================================
    _draw_graphical_model(axes["A"])

    # =========================================================================
    # Panel B: Equation boxes
    # =========================================================================
    _draw_equation_boxes(axes["B"])

    # =========================================================================
    # Panel C: Distribution consistency examples (4 sub-panels)
    # =========================================================================

    # Define x-axis for distributions
    x = np.linspace(-20, 20, 1000)

    # Colors from semantic COLORS system
    color_predictive = COLORS["predictive"]
    color_likelihood = COLORS["likelihood"]

    # Define scenarios
    scenarios = [
        {
            "title": "Inconsistent",
            "predictive": (0, 1.5),
            "likelihood": (5, 1.5),
        },
        {
            "title": "Consistent",
            "predictive": (0, 1.5),
            "likelihood": (2, 3.0),
        },
        {
            "title": "Consistent",
            "predictive": (0, 5.0),
            "likelihood": (5, 1.5),
        },
        {
            "title": "Consistent",
            "predictive": (0, 4.0),
            "likelihood": (5, 3.5),
        },
    ]

    sub_panel_names = ["C1", "C2", "C3", "C4"]

    for i, (panel_name, scenario) in enumerate(zip(sub_panel_names, scenarios, strict=True)):
        _create_distribution_panel(
            axes[panel_name],
            x,
            scenario,
            color_predictive,
            color_likelihood,
            show_direct_labels=(i == 0),  # Only show labels on first panel
        )

    # Draw canvas to finalize constrained_layout positions before querying them
    fig.canvas.draw()

    # Add shared x-axis label for Panel C
    # Position below the center of the four sub-panels
    c_panels_left = axes["C1"].get_position().x0
    c_panels_right = axes["C4"].get_position().x1
    c_panels_bottom = axes["C1"].get_position().y0
    fig.text(
        (c_panels_left + c_panels_right) / 2,
        c_panels_bottom - 0.02,
        "Latent state",
        ha="center",
        va="top",
        fontsize=7,
    )

    # Add panel labels using fig.text() at consistent x position
    # Use panel C1's left edge as reference - it has content starting at the left edge
    label_x = axes["C1"].get_position().x0 - 0.02
    for label, ax_key, y_offset in [("a", "A", 0.01), ("b", "B", 0.01), ("c", "C1", 0.01)]:
        fig.text(
            label_x,
            axes[ax_key].get_position().y1 + y_offset,
            label,
            fontsize=9,
            fontweight="bold",
            va="bottom",
            ha="right",
        )

    # Save to figures/main directory
    save_figure("figures/main/figure01")
    plt.close()
    print("\nFigure 1 saved to figures/main/figure01.{pdf,png}")


if __name__ == "__main__":
    create_figure()
