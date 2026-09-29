"""Track-graph rendering.

Draws the track graph as a 2D spatial layout (with the position trajectory
and a scale bar) or as a 1D linearized representation for the Figure-4 panels.
"""

from __future__ import annotations

from collections.abc import Hashable, Sequence

import matplotlib
import networkx as nx
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from numpy.typing import NDArray

from statespacecheck_paper.figure04_input import HEAD_POSITION_COLUMNS
from statespacecheck_paper.figure04_plot_primitives import add_scalebar


def _edge_colors() -> NDArray[np.float64]:
    """Return the tab10 colors that the track edges cycle through, shape (10, 4)."""
    cmap = matplotlib.colormaps.get_cmap("tab10")
    return np.array([cmap(i) for i in range(10)])


def plot_track_graph_2d(
    track_graph: nx.Graph,
    position_info: pd.DataFrame,
    edge_order: Sequence[tuple[Hashable, Hashable]],
    ax: Axes,
    reward_well_nodes: list[int] | None = None,
    scalebar_length: float = 20,
    scalebar_label: str = "20 cm",
) -> Axes:
    """Plot the 2D track graph over the head-position trajectory.

    Parameters
    ----------
    track_graph : networkx.Graph
        Track graph with nodes containing 'pos' attributes.
    position_info : pandas.DataFrame
        Position table with ``head_position_x`` and ``head_position_y`` columns.
    edge_order : sequence of tuple
        Explicit edge order shared with the scientific linearization; edges
        cycle through the tab10 colors in this order.
    ax : Axes
        Axes to plot on.
    reward_well_nodes : list of int, optional
        Node indices that are reward wells (marked with scatter points).
    scalebar_length : float, optional
        Length of scale bar in data units, by default 20.
    scalebar_label : str, optional
        Label for scale bar, by default "20 cm".

    Returns
    -------
    ax : Axes
        The axes object.
    """
    if reward_well_nodes is None:
        reward_well_nodes = []
    edge_colors = _edge_colors()
    x_column, y_column = HEAD_POSITION_COLUMNS
    ax.plot(
        position_info[x_column],
        position_info[y_column],
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
    reward_well_size: int = 10,
    edge_linewidth: int = 2,
) -> None:
    """Plot track graph as a vertical 1D linearized representation.

    Draws the track graph edges as vertical line segments positioned
    sequentially along the y (position) axis to show the linearized track
    structure.

    Parameters
    ----------
    track_graph : networkx.Graph
        Track graph with edges containing 'distance' attributes (in cm).
    ax : Axes
        Axes to plot on.
    edge_order : sequence of tuple
        Explicit edge order for the scientific linearization; edges cycle
        through the tab10 colors in this order.
    edge_spacing : float or list of float, optional
        Spacing between edges in cm. By default 0.0.
    reward_well_nodes : list of int, optional
        Node indices that are reward wells (marked with scatter points).
    other_axis_start : float, optional
        x-position of the segments.
    reward_well_size : int, optional
        Marker size for reward well points, by default 10.
    edge_linewidth : int, optional
        Line width for edge segments, by default 2.
    """
    if reward_well_nodes is None:
        reward_well_nodes = []
    edge_colors = _edge_colors()

    n_edges = len(edge_order)
    if isinstance(edge_spacing, int | float):
        edge_spacing_list = [float(edge_spacing)] * (n_edges - 1)
    else:
        edge_spacing_list = list(edge_spacing)

    start_node_linear_position = 0.0

    for edge_ind, edge in enumerate(edge_order):
        edge_color = edge_colors[edge_ind % len(edge_colors)]
        end_node_linear_position = start_node_linear_position + track_graph.edges[edge]["distance"]

        ax.plot(
            (other_axis_start, other_axis_start),
            (start_node_linear_position, end_node_linear_position),
            color=edge_color,
            clip_on=False,
            zorder=7,
            linewidth=edge_linewidth,
        )
        for node, node_position in (
            (edge[0], start_node_linear_position),
            (edge[1], end_node_linear_position),
        ):
            if node in reward_well_nodes:
                ax.scatter(
                    other_axis_start,
                    node_position,
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
