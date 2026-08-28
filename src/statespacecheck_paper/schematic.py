"""Graphical model schematic drawing utilities for state space model figures.

This module provides functions for drawing graphical model components,
including nodes, arrows, distribution insets, and equation boxes.
These are used to create publication-quality schematic diagrams.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
from matplotlib.axes import Axes
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from numpy.typing import NDArray
from scipy import stats

from statespacecheck_paper.style import COLORS

__all__ = [
    "draw_node",
    "draw_arrow",
    "draw_distribution_inset",
    "draw_spikes_inset",
    "draw_equation_box",
    "draw_graphical_model",
    "draw_equation_boxes",
]


def draw_node(
    ax: Axes,
    center: tuple[float, float],
    radius: float,
    label: str,
    facecolor: str = "white",
    edgecolor: str = "black",
    linewidth: float = 1.5,
) -> Circle:
    """Draw a circular node for graphical models.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes to draw on.
    center : tuple[float, float]
        Center position (x, y) in data coordinates.
    radius : float
        Radius of the circle.
    label : str
        Text label to display at center (supports LaTeX).
    facecolor : str, default "white"
        Fill color of the circle.
    edgecolor : str, default "black"
        Edge color of the circle.
    linewidth : float, default 1.5
        Width of the circle edge.

    Returns
    -------
    circle : matplotlib.patches.Circle
        The Circle patch that was added.

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> fig, ax = plt.subplots()
    >>> circle = draw_node(ax, (0.5, 0.5), 0.1, r"$x_t$")
    >>> plt.close(fig)
    """
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


def draw_arrow(
    ax: Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    label: str | None = None,
    color: str = "black",
    linewidth: float = 1.5,
    connectionstyle: str = "arc3,rad=0",
) -> FancyArrowPatch:
    """Draw an arrow between two points.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes to draw on.
    start : tuple[float, float]
        Starting position (x, y).
    end : tuple[float, float]
        Ending position (x, y).
    label : str | None, optional
        Text label to display above arrow midpoint.
    color : str, default "black"
        Arrow color.
    linewidth : float, default 1.5
        Arrow line width.
    connectionstyle : str, default "arc3,rad=0"
        Connection style for curved arrows.

    Returns
    -------
    arrow : matplotlib.patches.FancyArrowPatch
        The arrow patch that was added.

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> fig, ax = plt.subplots()
    >>> arrow = draw_arrow(ax, (0, 0), (1, 1), label="transition")
    >>> plt.close(fig)
    """
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
            style="italic",
        )

    return arrow


def draw_distribution_inset(
    ax: Axes,
    center: tuple[float, float],
    width: float,
    height: float,
    mean: float,
    std: float,
    color: str,
    label: str | None = None,
    label_color: str | None = None,
    label_size: int = 8,
    title: str | None = None,
    title_size: int = 8,
) -> None:
    """Draw a small distribution plot as an inset.

    Creates a Gaussian distribution inset at the specified location
    with optional title above and label below.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The parent axes to draw on.
    center : tuple[float, float]
        Center position in data coordinates.
    width : float
        Width of the inset in data coordinates.
    height : float
        Height of the inset in data coordinates.
    mean : float
        Mean of the Gaussian distribution.
    std : float
        Standard deviation of the Gaussian distribution.
    color : str
        Color for the distribution curve and fill.
    label : str | None, optional
        Label below distribution (e.g., math notation).
    label_color : str | None, optional
        Color for label. Defaults to distribution color.
    label_size : int, default 8
        Font size for label.
    title : str | None, optional
        Title above distribution.
    title_size : int, default 8
        Font size for title.

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> fig, ax = plt.subplots()
    >>> ax.set_xlim(0, 10)
    >>> ax.set_ylim(0, 10)
    >>> draw_distribution_inset(ax, (5, 5), 2, 1, mean=0, std=1, color="blue")
    >>> plt.close(fig)
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

    x: NDArray[np.floating] = np.linspace(mean - 3.5 * std, mean + 3.5 * std, 100)
    y: NDArray[np.floating] = stats.norm.pdf(x, mean, std)

    inset.plot(x, y, color=color, linewidth=1.2)
    inset.fill_between(x, y, alpha=0.3, color=color)

    inset.set_xlim(mean - 3.5 * std, mean + 3.5 * std)
    inset.set_ylim(0, float(y.max()) * 1.1)
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


def draw_spikes_inset(
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
        The parent axes to draw on.
    center : tuple[float, float]
        Center position in data coordinates.
    width : float
        Width of the inset in data coordinates.
    height : float
        Height of the inset in data coordinates.
    n_cells : int, default 5
        Number of cells to show in raster.
    rng : np.random.Generator | None, optional
        Random number generator. Defaults to seed 42.
    label : str, default "Spikes"
        Label to display below the raster.

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> fig, ax = plt.subplots()
    >>> ax.set_xlim(0, 10)
    >>> ax.set_ylim(0, 10)
    >>> draw_spikes_inset(ax, (5, 5), 2, 1, n_cells=5)
    >>> plt.close(fig)
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
    )


def draw_equation_box(
    ax: Axes,
    center: tuple[float, float],
    width: float,
    height: float,
    edgecolor: str = "#666666",
    facecolor: str = "#FAFAFA",
    linewidth: float = 1.0,
) -> FancyBboxPatch:
    """Draw a rounded box for equation grouping.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes to draw on.
    center : tuple[float, float]
        Center position of the box.
    width : float
        Width of the box.
    height : float
        Height of the box.
    edgecolor : str, default "#666666"
        Edge color.
    facecolor : str, default "#FAFAFA"
        Fill color.
    linewidth : float, default 1.0
        Edge line width.

    Returns
    -------
    box : matplotlib.patches.FancyBboxPatch
        The box patch that was added.

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> fig, ax = plt.subplots()
    >>> box = draw_equation_box(ax, (0.5, 0.5), 0.8, 0.4)
    >>> plt.close(fig)
    """
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


def draw_graphical_model(
    ax: Axes,
    rng: np.random.Generator | None = None,
) -> None:
    """Draw state space model graphical model.

    Draws the chain of latent states (x_{t-1} -> x_t) with
    observations (y_{t-1}, y_t) below, including spike raster insets.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes to draw on. Will be configured for appropriate limits.
    rng : np.random.Generator | None, optional
        Random number generator for spike insets. Defaults to seed 42.

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> fig, ax = plt.subplots(figsize=(8, 6))
    >>> draw_graphical_model(ax)
    >>> plt.close(fig)
    """
    if rng is None:
        rng = np.random.default_rng(42)

    ax.set_xlim(-0.5, 7.5)  # Content ends around 7
    ax.set_ylim(2.7, 6.4)  # Content from ~2.9 to ~6.0, with room for title
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
    draw_arrow(ax, arrow_start, arrow_end, color="black")

    # Draw state nodes
    draw_node(ax, x_prev_pos, node_radius, r"$x_{t-1}$", edgecolor="black")
    draw_node(ax, x_curr_pos, node_radius, r"$x_t$", edgecolor="black")

    # Draw observation nodes (filled to indicate observed)
    draw_node(
        ax, y_prev_obs_pos, node_radius, r"$y_{t-1}$", facecolor="lightgray", edgecolor="black"
    )
    draw_node(ax, y_curr_obs_pos, node_radius, r"$y_t$", facecolor="lightgray", edgecolor="black")

    # Horizontal arrow: x_{t-1} -> x_t
    arrow_start = (x_prev_pos[0] + node_radius + 0.05, x_prev_pos[1])
    arrow_end = (x_curr_pos[0] - node_radius - 0.05, x_curr_pos[1])
    draw_arrow(ax, arrow_start, arrow_end, color="black")

    # Transition label above the arrow
    transition_label_y = y_latent + 0.32
    ax.text(
        (x_prev_pos[0] + x_curr_pos[0]) / 2,
        transition_label_y + 0.02,
        "Transition",
        ha="center",
        va="bottom",
        color="#666666",
    )
    ax.text(
        (x_prev_pos[0] + x_curr_pos[0]) / 2,
        transition_label_y - 0.02,
        r"$p(x_t|x_{t-1})$",
        ha="center",
        va="top",
        color="#666666",
    )

    # Downward arrows: x -> y
    arrow_start = (x_prev_pos[0], x_prev_pos[1] - node_radius - 0.05)
    arrow_end = (y_prev_obs_pos[0], y_prev_obs_pos[1] + node_radius + 0.05)
    draw_arrow(ax, arrow_start, arrow_end, color="black")

    arrow_start = (x_curr_pos[0], x_curr_pos[1] - node_radius - 0.05)
    arrow_end = (y_curr_obs_pos[0], y_curr_obs_pos[1] + node_radius + 0.05)
    draw_arrow(ax, arrow_start, arrow_end, color="black")

    # Likelihood label
    arrow_midpoint_y = (x_curr_pos[1] + y_curr_obs_pos[1]) / 2
    ax.text(
        x_curr_pos[0] + 0.15,
        arrow_midpoint_y,
        "Likelihood\n" + r"$p(y_t|x_t)$",
        ha="left",
        va="center",
        color=COLORS["likelihood"],
        linespacing=1.0,
    )

    # Arrow from x_t to ellipsis (chain continues to the right)
    arrow_start = (x_curr_pos[0] + node_radius + 0.05, y_latent)
    arrow_end = (x_curr_pos[0] + 0.9, y_latent)
    draw_arrow(ax, arrow_start, arrow_end, color="black")

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
        fontstyle="italic",
    )
    ax.text(
        0.3,
        y_obs,
        "Observations",
        ha="left",
        va="center",
        fontstyle="italic",
    )

    # Labels above latent-state nodes
    ax.text(
        x_prev_pos[0],
        y_latent + 0.45,
        "Previous\nState",
        ha="center",
        va="bottom",
        color="black",
    )
    ax.text(
        x_curr_pos[0],
        y_latent + 0.45,
        "Current\nState",
        ha="center",
        va="bottom",
        color="black",
    )

    # Spike rasters below observations
    draw_spikes_inset(
        ax,
        center=(y_prev_obs_pos[0], y_prev_obs_pos[1] - 0.65),
        width=0.7,
        height=0.35,
        n_cells=5,
        rng=rng,
        label="Previous\nSpikes",
    )
    draw_spikes_inset(
        ax,
        center=(y_curr_obs_pos[0], y_curr_obs_pos[1] - 0.65),
        width=0.7,
        height=0.35,
        n_cells=5,
        rng=rng,
        label="Current\nSpikes",
    )

    # Title (using set_title for consistent positioning across panels)
    ax.set_title("State Space Model", fontsize=8, fontweight="bold", pad=4)

    # Time direction indicator
    ax.text(
        x_curr_pos[0] + 1.5,
        y_obs - 0.85,
        r"Time $\rightarrow$",
        ha="left",
        va="center",
        fontstyle="italic",
        color="#666666",
    )


class _InsetSpec(NamedTuple):
    """One distribution inset in a filtering-equation row."""

    mean: float
    std: float
    color: str
    label: str
    title: str


def _draw_equation_row(
    ax: Axes,
    *,
    y: float,
    step_number: int,
    step_name: str,
    step_color: str,
    left: _InsetSpec,
    operator: str,
    operator_label: str,
    operator_size: float,
    middle: _InsetSpec,
    result: _InsetSpec,
    box_center_x: float,
    box_width: float,
) -> None:
    """Draw one ``left <operator> middle = result`` row of the algorithm box.

    The prediction and update steps share this exact layout — background box,
    three distribution insets, an operator symbol with a caption, an equals
    sign, and a numbered step label. Only the y position, the step label, the
    operator, and the three :class:`_InsetSpec` distributions differ between
    them.
    """
    center_y = y + 0.2
    # Offset that shifts the equation elements right to make room for labels.
    eq_offset = 0.8

    draw_equation_box(
        ax,
        center=(box_center_x, center_y),
        width=box_width,
        height=1.4,
        edgecolor="#CCCCCC",
        facecolor="#F9F9F9",
    )

    for spec, x in ((left, 1.3), (middle, 3.3), (result, 5.5)):
        draw_distribution_inset(
            ax,
            center=(x + eq_offset, center_y),
            width=0.9,
            height=0.5,
            mean=spec.mean,
            std=spec.std,
            color=spec.color,
            label=spec.label,
            label_size=8,
            title=spec.title,
        )

    operator_x = 2.3 + eq_offset
    ax.text(
        operator_x,
        center_y,
        operator,
        ha="center",
        va="center",
        fontsize=operator_size,
        fontweight="bold",
    )
    ax.text(
        operator_x,
        y + 0.55,
        operator_label,
        ha="center",
        va="bottom",
        color="#666666",
    )

    ax.text(4.3 + eq_offset, center_y, "=", ha="center", va="center", fontsize=14)

    ax.text(
        0.2,
        center_y,
        f"{step_number}. {step_name}",
        ha="left",
        va="center",
        color=step_color,
        fontweight="bold",
    )


def draw_equation_boxes(ax: Axes) -> None:
    """Draw the Bayesian filtering equation boxes.

    Draws two equation boxes showing:
    1. Prediction: Previous Posterior (convolved with) Transition = Predictive
    2. Update: Predictive x Likelihood = Current Posterior

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes to draw on. Will be configured for appropriate limits.

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> fig, ax = plt.subplots(figsize=(8, 4))
    >>> draw_equation_boxes(ax)
    >>> plt.close(fig)
    """
    ax.set_xlim(-0.5, 7.5)  # Content ends around 7.15
    ax.set_ylim(-0.85, 2.45)  # Content from ~-0.7 to ~2.3, with room for title
    ax.axis("off")

    # Box dimensions aligned with graphical model. Both equation rows share the
    # same width/center; only their y position, step label, operator, and the
    # three distribution insets differ.
    box_right_edge = 7.15
    box_left_edge = 0.0
    box_width = box_right_edge - box_left_edge
    box_center_x = (box_left_edge + box_right_edge) / 2

    y_eq1 = 1.4
    y_eq2 = -0.25

    # The predictive distribution is the output of step 1 and the first input of
    # step 2; use one spec so the two rows stay identical by construction.
    predictive_inset = _InsetSpec(
        mean=45,
        std=12,
        color=COLORS["predictive"],
        label=r"$p(x_t|y_{1:t-1})$",
        title="Predictive\nDistribution",
    )

    _draw_equation_row(
        ax,
        y=y_eq1,
        step_number=1,
        step_name="Prediction",
        step_color=COLORS["predictive"],
        left=_InsetSpec(
            mean=40,
            std=8,
            color=COLORS["posterior"],
            label=r"$p(x_{t-1}|y_{1:t-1})$",
            title="Previous\nPosterior",
        ),
        operator=r"$\circledast$",
        operator_label="Convolve",
        operator_size=16,
        middle=_InsetSpec(
            mean=45,
            std=10,
            color="#666666",
            label=r"$p(x_t|x_{t-1})$",
            title="Transition",
        ),
        result=predictive_inset,
        box_center_x=box_center_x,
        box_width=box_width,
    )
    _draw_equation_row(
        ax,
        y=y_eq2,
        step_number=2,
        step_name="Update",
        step_color=COLORS["posterior"],
        left=predictive_inset,
        operator=r"$\times$",
        operator_label="Mult.",
        operator_size=14,
        middle=_InsetSpec(
            mean=50,
            std=10,
            color=COLORS["likelihood"],
            label=r"$p(y_t|x_t)$",
            title="Likelihood",
        ),
        result=_InsetSpec(
            mean=48,
            std=7,
            color=COLORS["posterior"],
            label=r"$p(x_t|y_{1:t})$",
            title="Current\nPosterior",
        ),
        box_center_x=box_center_x,
        box_width=box_width,
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
    draw_arrow(
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
        fontweight="bold",
        color="#666666",
    )

    # Title (using set_title for consistent positioning across panels)
    ax.set_title("Recursive Estimation Algorithm", fontsize=8, fontweight="bold", pad=4)
