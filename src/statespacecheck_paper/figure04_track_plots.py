"""Track-graph rendering.

Draws the track graph as a 2D spatial layout (with optional trajectory overlay
and scale bar) or as a 1D linearized representation for the Figure-4 panels.
"""

from __future__ import annotations

from collections.abc import Hashable, Sequence

import matplotlib
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from numpy.typing import NDArray

from statespacecheck_paper.figure04_plot_primitives import add_scalebar


def plot_track_graph_2d(
    track_graph: nx.Graph,
    position_info: pd.DataFrame,
    edge_order: Sequence[tuple[Hashable, Hashable]],
    ax: Axes | None = None,
    reward_well_nodes: list[int] | None = None,
    edge_colors: NDArray[np.float64] | None = None,
    position_names: tuple[str, str] = ("head_position_x", "head_position_y"),
    scalebar_length: float = 20,
    scalebar_label: str = "20 cm",
    show_trajectory: bool = True,
) -> Axes:
    """Plot 2D track graph with optional position trajectory overlay.

    Parameters
    ----------
    track_graph : networkx.Graph
        Track graph with nodes containing 'pos' attributes.
    position_info : pandas.DataFrame
        DataFrame containing position columns for trajectory overlay.
    ax : Axes, optional
        Axes to plot on. If None, uses current axes.
    edge_order : sequence of tuple
        Explicit edge order shared with the scientific linearization.
    reward_well_nodes : list of int, optional
        Node indices that are reward wells (marked with scatter points).
    edge_colors : ndarray, optional
        Array of colors for each edge. If None, uses tab10 colormap.
    position_names : tuple of str, optional
        Column names for (x, y) position in position_info.
    scalebar_length : float, optional
        Length of scale bar in data units, by default 20.
    scalebar_label : str, optional
        Label for scale bar, by default "20 cm".
    show_trajectory : bool, default True
        Whether to show the position trajectory.

    Returns
    -------
    ax : Axes
        The axes object.
    """
    if ax is None:
        ax = plt.gca()
    if reward_well_nodes is None:
        reward_well_nodes = []
    if edge_colors is None:
        cmap = matplotlib.colormaps.get_cmap("tab10")
        edge_colors = np.array([cmap(i) for i in range(10)])
    # Plot trajectory
    if show_trajectory:
        ax.plot(
            position_info[position_names[0]],
            position_info[position_names[1]],
            color="lightgrey",
            alpha=0.7,
            linewidth=0.5,
            rasterized=True,
        )

    # Plot track graph edges
    for edge_ind, (node1, node2) in enumerate(edge_order):
        edge_color = edge_colors[edge_ind % len(edge_colors)]
        node1_pos = track_graph.nodes[node1]["pos"]
        node2_pos = track_graph.nodes[node2]["pos"]
        ax.plot(
            [node1_pos[0], node2_pos[0]],
            [node1_pos[1], node2_pos[1]],
            linewidth=2,
            color=edge_color,
        )
        if node1 in reward_well_nodes:
            ax.scatter(
                node1_pos[0],
                node1_pos[1],
                color=edge_color,
                s=45,
                zorder=10,
                edgecolors="black",
                linewidths=0.5,
            )
        if node2 in reward_well_nodes:
            ax.scatter(
                node2_pos[0],
                node2_pos[1],
                color=edge_color,
                s=45,
                zorder=10,
                edgecolors="black",
                linewidths=0.5,
            )

    add_scalebar(ax, scalebar_length, scalebar_label)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")

    return ax


def plot_track_graph_1d(
    track_graph: nx.Graph,
    ax: Axes,
    edge_order: Sequence[tuple[Hashable, Hashable]],
    edge_spacing: float | list[float] = 0.0,
    reward_well_nodes: list[int] | None = None,
    other_axis_start: float = 0,
    edge_colors: NDArray[np.float64] | None = None,
    reward_well_size: int = 10,
    edge_linewidth: int = 2,
    orientation: str = "vertical",
) -> None:
    """Plot track graph as 1D linearized representation.

    Draws the track graph edges as line segments positioned sequentially
    to show the linearized track structure. Default is vertical orientation
    (position on y-axis). Use orientation="horizontal" for position on x-axis.

    Parameters
    ----------
    track_graph : networkx.Graph
        Track graph with edges containing 'distance' attributes (in cm).
    ax : Axes
        Axes to plot on.
    edge_order : sequence of tuple
        Explicit edge order for the scientific linearization.
    edge_spacing : float or list of float, optional
        Spacing between edges in cm. By default 0.0.
    reward_well_nodes : list of int, optional
        Node indices that are reward wells (marked with scatter points).
    other_axis_start : float, optional
        Position on the non-position axis (x for vertical, y for horizontal).
    edge_colors : ndarray, optional
        Array of RGB colors for each edge. If None, uses tab10 colormap.
    reward_well_size : int, optional
        Marker size for reward well points, by default 10.
    edge_linewidth : int, optional
        Line width for edge segments, by default 2.
    orientation : str, default "vertical"
        Orientation of the track. "vertical" places position on y-axis,
        "horizontal" places position on x-axis.
    """
    if reward_well_nodes is None:
        reward_well_nodes = []
    if edge_colors is None:
        cmap = matplotlib.colormaps.get_cmap("tab10")
        edge_colors = np.array([cmap(i) for i in range(10)])

    n_edges = len(edge_order)
    if isinstance(edge_spacing, int | float):
        edge_spacing_list = [float(edge_spacing)] * (n_edges - 1)
    else:
        edge_spacing_list = list(edge_spacing)

    start_node_linear_position = 0.0

    for edge_ind, edge in enumerate(edge_order):
        edge_color = edge_colors[edge_ind % len(edge_colors)]
        end_node_linear_position = start_node_linear_position + track_graph.edges[edge]["distance"]

        if orientation == "vertical":
            # Position on y-axis, other_axis_start is x-position
            ax.plot(
                (other_axis_start, other_axis_start),
                (start_node_linear_position, end_node_linear_position),
                color=edge_color,
                clip_on=False,
                zorder=7,
                linewidth=edge_linewidth,
            )
            scatter_x, scatter_y_start, scatter_y_end = (
                other_axis_start,
                start_node_linear_position,
                end_node_linear_position,
            )
        else:
            # Position on x-axis, other_axis_start is y-position
            ax.plot(
                (start_node_linear_position, end_node_linear_position),
                (other_axis_start, other_axis_start),
                color=edge_color,
                clip_on=False,
                zorder=7,
                linewidth=edge_linewidth,
                solid_capstyle="butt",
            )
            scatter_x_start, scatter_x_end, scatter_y = (
                start_node_linear_position,
                end_node_linear_position,
                other_axis_start,
            )

        if edge[0] in reward_well_nodes:
            if orientation == "vertical":
                ax.scatter(
                    scatter_x,
                    scatter_y_start,
                    color=edge_color,
                    s=reward_well_size,
                    zorder=10,
                    clip_on=False,
                )
            else:
                ax.scatter(
                    scatter_x_start,
                    scatter_y,
                    color=edge_color,
                    s=reward_well_size,
                    zorder=10,
                    clip_on=False,
                )
        if edge[1] in reward_well_nodes:
            if orientation == "vertical":
                ax.scatter(
                    scatter_x,
                    scatter_y_end,
                    color=edge_color,
                    s=reward_well_size,
                    zorder=10,
                    clip_on=False,
                )
            else:
                ax.scatter(
                    scatter_x_end,
                    scatter_y,
                    color=edge_color,
                    s=reward_well_size,
                    zorder=10,
                    clip_on=False,
                )

        # Update position for next edge (skip spacing on last edge)
        if edge_ind < len(edge_spacing_list):
            start_node_linear_position += (
                track_graph.edges[edge]["distance"] + edge_spacing_list[edge_ind]
            )
        else:
            start_node_linear_position += track_graph.edges[edge]["distance"]
