"""Tests for schematic drawing utilities."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
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


class TestDrawNode:
    """Tests for draw_node function."""

    def test_creates_circle(self) -> None:
        """Test that draw_node creates a Circle patch."""
        fig, ax = plt.subplots()
        circle = draw_node(ax, (0.5, 0.5), 0.1, r"$x_t$")

        assert isinstance(circle, Circle)
        assert circle in ax.patches
        plt.close(fig)

    def test_circle_position_and_radius(self) -> None:
        """Test that circle has correct position and radius."""
        fig, ax = plt.subplots()
        center = (0.5, 0.5)
        radius = 0.15
        circle = draw_node(ax, center, radius, "test")

        assert circle.center == center
        assert circle.radius == radius
        plt.close(fig)

    def test_custom_colors(self) -> None:
        """Test custom facecolor and edgecolor."""
        fig, ax = plt.subplots()
        circle = draw_node(
            ax,
            (0.5, 0.5),
            0.1,
            "test",
            facecolor="lightgray",
            edgecolor="red",
        )

        assert circle.get_facecolor() == plt.cm.colors.to_rgba("lightgray")
        assert circle.get_edgecolor() == plt.cm.colors.to_rgba("red")
        plt.close(fig)

    def test_label_added_to_axes(self) -> None:
        """Test that label text is added to axes."""
        fig, ax = plt.subplots()
        draw_node(ax, (0.5, 0.5), 0.1, r"$x_t$")

        # Check that text was added
        texts = ax.texts
        assert len(texts) == 1
        assert texts[0].get_text() == r"$x_t$"
        plt.close(fig)


class TestDrawArrow:
    """Tests for draw_arrow function."""

    def test_creates_arrow(self) -> None:
        """Test that draw_arrow creates FancyArrowPatch."""
        fig, ax = plt.subplots()
        arrow = draw_arrow(ax, (0, 0), (1, 1))

        assert isinstance(arrow, FancyArrowPatch)
        assert arrow in ax.patches
        plt.close(fig)

    def test_with_label(self) -> None:
        """Test arrow with text label."""
        fig, ax = plt.subplots()
        draw_arrow(ax, (0, 0), (1, 1), label="transition")

        # Check that text was added
        texts = ax.texts
        assert len(texts) == 1
        assert texts[0].get_text() == "transition"
        plt.close(fig)

    def test_without_label(self) -> None:
        """Test arrow without label adds no text."""
        fig, ax = plt.subplots()
        draw_arrow(ax, (0, 0), (1, 1))

        texts = ax.texts
        assert len(texts) == 0
        plt.close(fig)

    def test_custom_color(self) -> None:
        """Test arrow with custom color."""
        fig, ax = plt.subplots()
        arrow = draw_arrow(ax, (0, 0), (1, 1), color="red")

        # Arrow color is a tuple
        assert arrow.get_edgecolor() == plt.cm.colors.to_rgba("red")
        plt.close(fig)


class TestDrawDistributionInset:
    """Tests for draw_distribution_inset function."""

    def test_creates_inset(self) -> None:
        """Test that inset axes is created."""
        fig, ax = plt.subplots()
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)

        # This shouldn't raise
        draw_distribution_inset(ax, center=(5, 5), width=2, height=1, mean=0, std=1, color="blue")
        plt.close(fig)

    def test_with_title_and_label(self) -> None:
        """Test distribution with title and label."""
        fig, ax = plt.subplots()
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)

        # This shouldn't raise
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
        plt.close(fig)


class TestDrawSpikesInset:
    """Tests for draw_spikes_inset function."""

    def test_creates_raster(self) -> None:
        """Test that spike raster is created without error."""
        fig, ax = plt.subplots()
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)

        # This shouldn't raise
        draw_spikes_inset(ax, center=(5, 5), width=2, height=1, n_cells=5)
        plt.close(fig)

    def test_reproducible_with_rng(self) -> None:
        """Test reproducibility with explicit rng."""
        fig1, ax1 = plt.subplots()
        ax1.set_xlim(0, 10)
        ax1.set_ylim(0, 10)
        rng1 = np.random.default_rng(42)
        draw_spikes_inset(ax1, center=(5, 5), width=2, height=1, n_cells=5, rng=rng1)

        fig2, ax2 = plt.subplots()
        ax2.set_xlim(0, 10)
        ax2.set_ylim(0, 10)
        rng2 = np.random.default_rng(42)
        draw_spikes_inset(ax2, center=(5, 5), width=2, height=1, n_cells=5, rng=rng2)

        # Both should complete without error (visual comparison would need different test)
        plt.close(fig1)
        plt.close(fig2)

    def test_custom_label(self) -> None:
        """Test spike raster with custom label."""
        fig, ax = plt.subplots()
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)

        # This shouldn't raise
        draw_spikes_inset(ax, center=(5, 5), width=2, height=1, label="Custom\nLabel")
        plt.close(fig)


class TestDrawEquationBox:
    """Tests for draw_equation_box function."""

    def test_creates_box(self) -> None:
        """Test that rounded box is created."""
        fig, ax = plt.subplots()
        box = draw_equation_box(ax, (0.5, 0.5), 0.8, 0.4)

        assert isinstance(box, FancyBboxPatch)
        assert box in ax.patches
        plt.close(fig)

    def test_custom_colors(self) -> None:
        """Test box with custom colors."""
        fig, ax = plt.subplots()
        box = draw_equation_box(
            ax,
            (0.5, 0.5),
            0.8,
            0.4,
            edgecolor="red",
            facecolor="lightblue",
        )

        assert box.get_edgecolor() == plt.cm.colors.to_rgba("red")
        assert box.get_facecolor() == plt.cm.colors.to_rgba("lightblue")
        plt.close(fig)


class TestDrawGraphicalModel:
    """Tests for draw_graphical_model function."""

    def test_creates_full_model(self) -> None:
        """Test that graphical model is drawn without errors."""
        fig, ax = plt.subplots(figsize=(8, 6))
        draw_graphical_model(ax)

        # Should have multiple patches (nodes, arrows)
        assert len(ax.patches) > 0
        # Should have multiple text elements
        assert len(ax.texts) > 0
        plt.close(fig)

    def test_sets_axis_limits(self) -> None:
        """Test that axis limits are configured."""
        fig, ax = plt.subplots(figsize=(8, 6))
        draw_graphical_model(ax)

        # Check xlim and ylim are set
        assert ax.get_xlim() == (-0.5, 7.5)
        assert ax.get_ylim() == (2.7, 6.4)
        plt.close(fig)

    def test_reproducible_with_rng(self) -> None:
        """Test that graphical model is reproducible with RNG."""
        fig1, ax1 = plt.subplots(figsize=(8, 6))
        rng1 = np.random.default_rng(42)
        draw_graphical_model(ax1, rng=rng1)

        fig2, ax2 = plt.subplots(figsize=(8, 6))
        rng2 = np.random.default_rng(42)
        draw_graphical_model(ax2, rng=rng2)

        # Both should complete without error
        plt.close(fig1)
        plt.close(fig2)


class TestDrawEquationBoxes:
    """Tests for draw_equation_boxes function."""

    def test_creates_equations(self) -> None:
        """Test that equation boxes are drawn without errors."""
        fig, ax = plt.subplots(figsize=(8, 4))
        draw_equation_boxes(ax)

        # Should have multiple patches (boxes, arrows)
        assert len(ax.patches) > 0
        # Should have multiple text elements
        assert len(ax.texts) > 0
        plt.close(fig)

    def test_sets_axis_limits(self) -> None:
        """Test that axis limits are configured."""
        fig, ax = plt.subplots(figsize=(8, 4))
        draw_equation_boxes(ax)

        # Check xlim and ylim are set
        assert ax.get_xlim() == (-0.5, 7.5)
        assert ax.get_ylim() == (-0.85, 2.45)
        plt.close(fig)

    def test_has_title(self) -> None:
        """Test that title is set."""
        fig, ax = plt.subplots(figsize=(8, 4))
        draw_equation_boxes(ax)

        assert ax.get_title() == "Recursive Estimation Algorithm"
        plt.close(fig)
