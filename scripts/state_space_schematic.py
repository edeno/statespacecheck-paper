"""Exploratory figure: State space model schematic.

This figure shows the Bayesian filtering process as two explicit equations:
- Equation 1: [Prev Posterior] ⊛ T = [Predictive]
- Equation 2: [Predictive] × [Likelihood] = [Current Posterior]

Layout emphasizes the two-step process:
1. Prediction step: Convolve previous posterior with transition matrix
2. Update step: Multiply predictive with likelihood to get current posterior

The graphical model (x_{t-1} → x_t → y_t) provides context, while the
equation blocks show the computational steps explicitly.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch
from scipy.stats import norm

from statespacecheck_paper.style import COLORS, save_figure, set_figure_defaults


def draw_node(
    ax: plt.Axes,
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

    # Add label
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


def draw_arrow(
    ax: plt.Axes,
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
        # Position label at midpoint
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


def draw_distribution_inset(
    ax: plt.Axes,
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
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes

    if label_color is None:
        label_color = color

    # Calculate bounds in data coordinates
    left_data = center[0] - width / 2
    bottom_data = center[1] - height / 2

    # Create inset using data coordinates directly via inset_axes
    # This properly handles aspect ratio constraints
    inset = inset_axes(
        ax,
        width="100%",
        height="100%",
        bbox_to_anchor=(left_data, bottom_data, width, height),
        bbox_transform=ax.transData,
        borderpad=0,
    )

    # Plot distribution
    x = np.linspace(mean - 3.5 * std, mean + 3.5 * std, 100)
    y = norm.pdf(x, mean, std)

    inset.plot(x, y, color=color, linewidth=1.2)
    inset.fill_between(x, y, alpha=0.3, color=color)

    # Style
    inset.set_xlim(mean - 3.5 * std, mean + 3.5 * std)
    inset.set_ylim(0, y.max() * 1.1)
    inset.axis("off")

    # Add title above
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

    # Add label below
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


def draw_spikes_inset(
    ax: plt.Axes,
    center: tuple[float, float],
    width: float,
    height: float,
    n_cells: int = 5,
    rng: np.random.Generator | None = None,
) -> None:
    """Draw a small spike raster as an inset."""
    if rng is None:
        rng = np.random.default_rng(42)

    fig = ax.figure
    trans = ax.transData + fig.transFigure.inverted()

    # Calculate bounds
    left_data = center[0] - width / 2
    bottom_data = center[1] - height / 2

    left_fig, bottom_fig = trans.transform((left_data, bottom_data))
    right_fig, top_fig = trans.transform((left_data + width, bottom_data + height))

    width_fig = right_fig - left_fig
    height_fig = top_fig - bottom_fig

    inset = fig.add_axes([left_fig, bottom_fig, width_fig, height_fig])

    # Generate spike times
    for i in range(n_cells):
        n_spikes = rng.integers(2, 8)
        spike_times = rng.uniform(0, 1, n_spikes)
        inset.scatter(
            spike_times,
            [i] * n_spikes,
            marker="|",
            s=30,
            color="black",
            linewidths=1,
        )

    inset.set_xlim(-0.1, 1.1)
    inset.set_ylim(-0.5, n_cells - 0.5)
    inset.axis("off")

    # Label
    inset.text(
        0.5,
        -0.2,
        "Spikes",
        ha="center",
        va="top",
        transform=inset.transAxes,
        fontsize=6,
    )


def draw_equation_box(
    ax: plt.Axes,
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


def create_figure() -> None:
    """Create state space model schematic with equation blocks layout."""
    set_figure_defaults(context="paper")
    rng = np.random.default_rng(42)

    # Create figure
    fig, ax = plt.subplots(figsize=(7.0, 6.5), dpi=450)
    ax.set_xlim(-0.5, 9.5)
    ax.set_ylim(-0.5, 6.5)
    ax.set_aspect("equal")
    ax.axis("off")

    # ==========================================================================
    # TOP: Graphical Model (x_{t-1} → x_t → y_t)
    # ==========================================================================

    node_radius = 0.30
    y_graphical = 5.5

    # Node positions
    x_prev_pos = (2.0, y_graphical)
    x_curr_pos = (4.5, y_graphical)
    y_obs_pos = (4.5, y_graphical - 1.2)

    # Draw state nodes (all black edges for standard graphical model)
    draw_node(ax, x_prev_pos, node_radius, r"$x_{t-1}$", edgecolor="black")
    draw_node(ax, x_curr_pos, node_radius, r"$x_t$", edgecolor="black")

    # Draw observation node (filled to indicate observed)
    draw_node(ax, y_obs_pos, node_radius, r"$y_t$", facecolor="lightgray", edgecolor="black")

    # Draw arrows
    arrow_start = (x_prev_pos[0] + node_radius + 0.05, x_prev_pos[1])
    arrow_end = (x_curr_pos[0] - node_radius - 0.05, x_curr_pos[1])
    draw_arrow(ax, arrow_start, arrow_end, label=r"$T$", color="black")

    arrow_start = (x_curr_pos[0], x_curr_pos[1] - node_radius - 0.05)
    arrow_end = (y_obs_pos[0], y_obs_pos[1] + node_radius + 0.05)
    draw_arrow(ax, arrow_start, arrow_end, label=r"$p(y_t|x_t)$", color="black")

    # Spikes below y_t (further from node for spacing)
    draw_spikes_inset(
        ax,
        center=(y_obs_pos[0], y_obs_pos[1] - 0.9),
        width=0.8,
        height=0.4,
        n_cells=5,
        rng=rng,
    )

    # Title
    ax.text(
        4.75,
        6.3,
        "State Space Model",
        ha="center",
        va="bottom",
        fontsize=8,
        fontweight="bold",
    )

    # ==========================================================================
    # EQUATION 1: [Prev Posterior] ⊛ T = [Predictive]
    # ==========================================================================

    y_eq1 = 1.9  # Shifted down to avoid spikes from graphical model
    box_height = 1.6  # Increased from 1.2 to accommodate distributions + labels

    # Draw equation box (center shifted up to match content at y_eq1 + 0.2)
    draw_equation_box(
        ax,
        center=(4.5, y_eq1 + 0.2),
        width=9.2,
        height=box_height,
        edgecolor="#CCCCCC",
        facecolor="#F9F9F9",
    )

    # Distribution 1: Previous Posterior
    draw_distribution_inset(
        ax,
        center=(1.3, y_eq1 + 0.2),
        width=0.9,
        height=0.5,
        mean=40,
        std=8,
        color=COLORS["posterior"],
        label=r"$p(x_{t-1}|y_{1:t-1})$",
        label_size=5,
        title="Previous\nPosterior",
    )

    # Operation symbol: ⊛
    ax.text(
        2.5,
        y_eq1 + 0.2,
        r"$\circledast$",
        ha="center",
        va="center",
        fontsize=16,
        fontweight="bold",
    )

    # Transition matrix T
    ax.text(
        3.2,
        y_eq1 + 0.2,
        r"$T$",
        ha="center",
        va="center",
        fontsize=10,
        fontweight="bold",
        style="italic",
    )

    # Equals
    ax.text(
        3.9,
        y_eq1 + 0.2,
        "=",
        ha="center",
        va="center",
        fontsize=14,
    )

    # Distribution 2: Predictive
    draw_distribution_inset(
        ax,
        center=(5.2, y_eq1 + 0.2),
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
        "Step 1:",
        ha="left",
        va="center",
        fontsize=7,
        fontweight="bold",
    )

    # Descriptive label
    ax.text(
        7.5,
        y_eq1 + 0.2,
        "Prediction",
        ha="left",
        va="center",
        fontsize=7,
        color=COLORS["predictive"],
        fontweight="bold",
    )

    # ==========================================================================
    # EQUATION 2: [Predictive] × [Likelihood] = [Current Posterior]
    # ==========================================================================

    y_eq2 = 0.0  # Shifted down to maintain spacing
    box_height_eq2 = 1.6  # Same as Step 1

    # Draw equation box (center shifted up to match content at y_eq2 + 0.2)
    draw_equation_box(
        ax,
        center=(4.5, y_eq2 + 0.2),
        width=9.2,
        height=box_height_eq2,
        edgecolor="#CCCCCC",
        facecolor="#F9F9F9",
    )

    # Distribution 1: Predictive (repeated from equation 1)
    draw_distribution_inset(
        ax,
        center=(1.3, y_eq2 + 0.2),
        width=0.9,
        height=0.5,
        mean=45,
        std=12,
        color=COLORS["predictive"],
        label=r"$p(x_t|y_{1:t-1})$",
        label_size=5,
        title="Predictive\nDistribution",
    )

    # Operation symbol: ×
    ax.text(
        2.5,
        y_eq2 + 0.2,
        r"$\times$",
        ha="center",
        va="center",
        fontsize=14,
        fontweight="bold",
    )

    # Distribution 2: Likelihood (near observations)
    draw_distribution_inset(
        ax,
        center=(3.5, y_eq2 + 0.2),
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
        4.7,
        y_eq2 + 0.2,
        "=",
        ha="center",
        va="center",
        fontsize=14,
    )

    # Distribution 3: Current Posterior
    draw_distribution_inset(
        ax,
        center=(6.0, y_eq2 + 0.2),
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
        "Step 2:",
        ha="left",
        va="center",
        fontsize=7,
        fontweight="bold",
    )

    # Descriptive label
    ax.text(
        7.5,
        y_eq2 + 0.2,
        "Update",
        ha="left",
        va="center",
        fontsize=7,
        color=COLORS["posterior"],
        fontweight="bold",
    )

    # Save
    save_figure("figures/state_space_schematic")
    print("Saved figures/state_space_schematic.pdf and .png")


if __name__ == "__main__":
    create_figure()
