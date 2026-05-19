"""Tests for schematic drawing utilities."""

from __future__ import annotations

from collections.abc import Iterator

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch

from statespacecheck_paper.schematic import (
    draw_arrow,
    draw_distribution_inset,
    draw_equation_box,
    draw_equation_boxes,
    draw_graphical_model,
    draw_node,
    draw_spikes_inset,
)


@pytest.fixture
def unit_square_axes(
    fresh_axes: tuple[Figure, Axes],
) -> tuple[Figure, Axes]:
    """``fresh_axes`` pre-configured to a 0–10 unit-square coordinate system."""
    fig, ax = fresh_axes
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    return fig, ax


@pytest.fixture
def sized_axes() -> Iterator[tuple[Figure, Axes]]:
    """Composite-figure axes (8x6) auto-closed after the test."""
    fig, ax = plt.subplots(figsize=(8, 6))
    try:
        yield fig, ax
    finally:
        plt.close(fig)


# ---------------------------------------------------------------------------
# draw_node
# ---------------------------------------------------------------------------


class TestDrawNode:
    def test_returns_circle_attached_to_axes(self, fresh_axes: tuple[Figure, Axes]) -> None:
        _, ax = fresh_axes
        circle = draw_node(ax, (0.5, 0.5), 0.1, r"$x_t$")
        assert isinstance(circle, Circle)
        assert circle in ax.patches

    def test_circle_position_and_radius(self, fresh_axes: tuple[Figure, Axes]) -> None:
        _, ax = fresh_axes
        circle = draw_node(ax, (0.5, 0.5), 0.15, "test")
        assert circle.center == (0.5, 0.5)
        assert circle.radius == 0.15

    def test_custom_colors_applied(self, fresh_axes: tuple[Figure, Axes]) -> None:
        _, ax = fresh_axes
        circle = draw_node(
            ax,
            (0.5, 0.5),
            0.1,
            "test",
            facecolor="lightgray",
            edgecolor="red",
        )
        assert circle.get_facecolor() == mcolors.to_rgba("lightgray")
        assert circle.get_edgecolor() == mcolors.to_rgba("red")

    def test_label_text_added_to_axes(self, fresh_axes: tuple[Figure, Axes]) -> None:
        _, ax = fresh_axes
        draw_node(ax, (0.5, 0.5), 0.1, r"$x_t$")
        assert len(ax.texts) == 1
        assert ax.texts[0].get_text() == r"$x_t$"


# ---------------------------------------------------------------------------
# draw_arrow
# ---------------------------------------------------------------------------


class TestDrawArrow:
    def test_returns_arrow_attached_to_axes(self, fresh_axes: tuple[Figure, Axes]) -> None:
        _, ax = fresh_axes
        arrow = draw_arrow(ax, (0, 0), (1, 1))
        assert isinstance(arrow, FancyArrowPatch)
        assert arrow in ax.patches

    @pytest.mark.parametrize(
        ("label", "expected_text_count"),
        [(None, 0), ("transition", 1)],
    )
    def test_label_text_optional(
        self,
        fresh_axes: tuple[Figure, Axes],
        label: str | None,
        expected_text_count: int,
    ) -> None:
        _, ax = fresh_axes
        draw_arrow(ax, (0, 0), (1, 1), label=label)
        assert len(ax.texts) == expected_text_count
        if label is not None:
            assert ax.texts[0].get_text() == label

    def test_custom_color_applied(self, fresh_axes: tuple[Figure, Axes]) -> None:
        _, ax = fresh_axes
        arrow = draw_arrow(ax, (0, 0), (1, 1), color="red")
        assert arrow.get_edgecolor() == mcolors.to_rgba("red")


# ---------------------------------------------------------------------------
# draw_distribution_inset / draw_spikes_inset
# ---------------------------------------------------------------------------


class TestDrawDistributionInset:
    def test_minimal_call(self, unit_square_axes: tuple[Figure, Axes]) -> None:
        _, ax = unit_square_axes
        draw_distribution_inset(ax, center=(5, 5), width=2, height=1, mean=0, std=1, color="blue")

    def test_with_title_and_label(self, unit_square_axes: tuple[Figure, Axes]) -> None:
        _, ax = unit_square_axes
        draw_distribution_inset(
            ax,
            center=(5, 5),
            width=2,
            height=1,
            mean=0,
            std=1,
            color="blue",
            title="Test Title",
            label=r"$p(x)$",
        )


class TestDrawSpikesInset:
    def test_minimal_call(self, unit_square_axes: tuple[Figure, Axes]) -> None:
        _, ax = unit_square_axes
        draw_spikes_inset(ax, center=(5, 5), width=2, height=1, n_cells=5)

    def test_same_rng_seed_yields_same_artist_count(self) -> None:
        """Spike raster positions are RNG-driven; identical seeds must
        produce identical numbers of artists (lines + collections)."""

        def _spike_artist_count() -> int:
            fig, ax = plt.subplots()
            ax.set_xlim(0, 10)
            ax.set_ylim(0, 10)
            try:
                draw_spikes_inset(
                    ax,
                    center=(5, 5),
                    width=2,
                    height=1,
                    n_cells=5,
                    rng=np.random.default_rng(42),
                )
                return len(ax.lines) + len(ax.collections)
            finally:
                plt.close(fig)

        assert _spike_artist_count() == _spike_artist_count()

    def test_custom_label(self, unit_square_axes: tuple[Figure, Axes]) -> None:
        _, ax = unit_square_axes
        draw_spikes_inset(
            ax,
            center=(5, 5),
            width=2,
            height=1,
            label="Custom\nLabel",
        )


# ---------------------------------------------------------------------------
# draw_equation_box
# ---------------------------------------------------------------------------


class TestDrawEquationBox:
    def test_returns_box_attached_to_axes(self, fresh_axes: tuple[Figure, Axes]) -> None:
        _, ax = fresh_axes
        box = draw_equation_box(ax, (0.5, 0.5), 0.8, 0.4)
        assert isinstance(box, FancyBboxPatch)
        assert box in ax.patches

    def test_custom_colors_applied(self, fresh_axes: tuple[Figure, Axes]) -> None:
        _, ax = fresh_axes
        box = draw_equation_box(
            ax,
            (0.5, 0.5),
            0.8,
            0.4,
            edgecolor="red",
            facecolor="lightblue",
        )
        assert box.get_edgecolor() == mcolors.to_rgba("red")
        assert box.get_facecolor() == mcolors.to_rgba("lightblue")


# ---------------------------------------------------------------------------
# Composite drawings: graphical model + equation boxes
# ---------------------------------------------------------------------------


class TestDrawGraphicalModel:
    def test_creates_patches_text_and_pins_axis_limits(
        self, sized_axes: tuple[Figure, Axes]
    ) -> None:
        """One call exercises the full schematic: patches + text are
        emitted, AND the layout-critical axis limits are pinned. Splitting
        these into two tests just doubles the slow ``draw_graphical_model``
        call without adding coverage."""
        _, ax = sized_axes
        draw_graphical_model(ax)
        assert len(ax.patches) > 0
        assert len(ax.texts) > 0
        # Layout depends on these specific limits — change is a regression.
        assert ax.get_xlim() == (-0.5, 7.5)
        assert ax.get_ylim() == (2.7, 6.4)


class TestDrawEquationBoxes:
    def test_creates_patches_text_and_pins_layout(self, sized_axes: tuple[Figure, Axes]) -> None:
        """Single call exercises content + layout pin (see graphical-model
        rationale)."""
        _, ax = sized_axes
        draw_equation_boxes(ax)
        assert len(ax.patches) > 0
        assert len(ax.texts) > 0
        assert ax.get_xlim() == (-0.5, 7.5)
        assert ax.get_ylim() == (-0.85, 2.45)
        assert ax.get_title() == "Recursive Estimation Algorithm"
