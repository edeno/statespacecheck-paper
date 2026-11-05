"""Tests for the Figure-4 track-graph renderers and the scale bar."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.colors as mcolors  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import networkx as nx  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from statespacecheck_paper.figure04_plot_primitives import add_scalebar  # noqa: E402
from statespacecheck_paper.figure04_track_plots import (  # noqa: E402
    plot_track_graph_1d,
    plot_track_graph_2d,
    reward_well_nodes,
)

_TAB10 = matplotlib.colormaps.get_cmap("tab10")
_EDGE_ORDER = ((0, 1), (1, 2), (2, 3))


@pytest.fixture
def track_graph() -> nx.Graph:
    """A three-edge linear track with edges of 10, 20, and 30 cm."""
    graph = nx.Graph()
    for node, x in enumerate((0.0, 10.0, 30.0, 60.0)):
        graph.add_node(node, pos=(x, 5.0))
    for (a, b), distance in zip(_EDGE_ORDER, (10.0, 20.0, 30.0), strict=True):
        graph.add_edge(a, b, distance=distance)
    return graph


@pytest.mark.parametrize("edge_spacing", [5.0, [5.0, 5.0]])
def test_1d_track_stacks_edges_vertically_with_spacing(
    track_graph: nx.Graph,
    edge_spacing: float | list[float],
    fresh_axes: tuple[plt.Figure, plt.Axes],
) -> None:
    _, ax = fresh_axes
    plot_track_graph_1d(
        track_graph,
        ax=ax,
        edge_order=_EDGE_ORDER,
        edge_spacing=edge_spacing,
        well_nodes=[0, 3],
        other_axis_start=2.0,
    )
    # Each edge is a vertical segment at x = other_axis_start, separated by the
    # 5 cm spacing, colored by its position in the edge order.
    assert [tuple(line.get_xdata()) for line in ax.lines] == [(2.0, 2.0)] * 3
    assert [tuple(line.get_ydata()) for line in ax.lines] == [
        (0.0, 10.0),
        (15.0, 35.0),
        (40.0, 70.0),
    ]
    for index, line in enumerate(ax.lines):
        assert mcolors.to_rgba(line.get_color()) == _TAB10(index)
    # Reward wells sit at the track ends.
    offsets = np.concatenate([c.get_offsets() for c in ax.collections])
    np.testing.assert_array_equal(offsets, [[2.0, 0.0], [2.0, 70.0]])


def test_2d_track_draws_trajectory_edges_wells_and_scale_bar(
    track_graph: nx.Graph, fresh_axes: tuple[plt.Figure, plt.Axes]
) -> None:
    position_info = pd.DataFrame(
        {"head_position_x": [0.0, 30.0, 60.0], "head_position_y": [4.0, 6.0, 5.0]}
    )
    _, ax = fresh_axes
    returned = plot_track_graph_2d(track_graph, position_info, _EDGE_ORDER, ax, well_nodes=[0, 3])
    assert returned is ax
    trajectory, *edges, scale_bar = ax.lines
    np.testing.assert_array_equal(trajectory.get_xdata(), position_info["head_position_x"])
    np.testing.assert_array_equal(trajectory.get_ydata(), position_info["head_position_y"])
    assert [tuple(edge.get_xdata()) for edge in edges] == [(0.0, 10.0), (10.0, 30.0), (30.0, 60.0)]
    # Default 20-unit scale bar, labeled.
    bar_x = scale_bar.get_xdata()
    assert bar_x[1] - bar_x[0] == pytest.approx(20.0)
    assert [text.get_text() for text in ax.texts] == ["20 cm"]
    # One marker per reward well, at its node.
    offsets = np.concatenate([c.get_offsets() for c in ax.collections])
    np.testing.assert_array_equal(offsets, [[0.0, 5.0], [60.0, 5.0]])
    assert not ax.axison


def test_scalebar_sits_in_the_lower_right(fresh_axes: tuple[plt.Figure, plt.Axes]) -> None:
    _, ax = fresh_axes
    ax.set_xlim(0.0, 100.0)
    ax.set_ylim(0.0, 50.0)
    add_scalebar(ax, 20.0, "20 cm")
    (bar,) = ax.lines
    # 10% of the extent in from the right and bottom edges.
    np.testing.assert_allclose(bar.get_xdata(), [70.0, 90.0])
    np.testing.assert_allclose(bar.get_ydata(), [5.0, 5.0])
    (label,) = ax.texts
    assert label.get_text() == "20 cm"
    assert label.get_position() == pytest.approx((80.0, 5.0 - 0.03 * 50.0))
    assert label.get_fontsize() == 8


def test_scalebar_offsets_move_the_bar_and_label_without_moving_the_limits(
    fresh_axes: tuple[plt.Figure, plt.Axes],
) -> None:
    _, ax = fresh_axes
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    add_scalebar(ax, 20, "20 cm")
    add_scalebar(ax, 20, "20 cm", offset=(5.0, -3.0), label_offset=(2.0, -1.0), label_fontsize=9.0)
    plain_bar, shifted_bar = ax.lines
    plain_label, shifted_label = ax.texts
    np.testing.assert_allclose(shifted_bar.get_xdata(), np.asarray(plain_bar.get_xdata()) + 5.0)
    np.testing.assert_allclose(shifted_bar.get_ydata(), np.asarray(plain_bar.get_ydata()) - 3.0)
    np.testing.assert_allclose(
        shifted_label.get_position(),
        np.asarray(plain_label.get_position()) + (7.0, -4.0),
    )
    assert shifted_label.get_fontsize() == 9.0
    assert ax.get_xlim() == (0, 100)
    assert ax.get_ylim() == (0, 100)


def test_reward_wells_are_the_leaf_nodes() -> None:
    graph = nx.Graph([(0, 1), (1, 2), (1, 3)])
    assert sorted(reward_well_nodes(graph)) == [0, 2, 3]
