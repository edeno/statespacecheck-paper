"""Tests for style module."""

from pathlib import Path

import matplotlib.pyplot as plt
import pytest

from statespacecheck_paper.style import (
    WONG,
    get_figure_size,
    save_figure,
    set_figure_defaults,
)


def test_wong_colors_exist() -> None:
    """Test that WONG color palette exists and has correct format."""
    assert len(WONG) == 8
    # All should be hex color strings
    for color in WONG:
        assert isinstance(color, str)
        assert color.startswith("#")
        assert len(color) == 7  # #RRGGBB format


def test_set_figure_defaults_paper() -> None:
    """Test that set_figure_defaults sets correct values for paper context."""
    set_figure_defaults(context="paper")

    # Check font sizes for paper context
    assert plt.rcParams["font.size"] == 7
    assert plt.rcParams["axes.labelsize"] == 7
    assert plt.rcParams["axes.titlesize"] == 8
    assert plt.rcParams["xtick.labelsize"] == 6
    assert plt.rcParams["ytick.labelsize"] == 6
    assert plt.rcParams["legend.fontsize"] == 6

    # Check font settings
    assert plt.rcParams["font.family"] == ["sans-serif"]
    assert "Arial" in plt.rcParams["font.sans-serif"]

    # Check line widths
    assert plt.rcParams["axes.linewidth"] == 0.5
    assert plt.rcParams["xtick.major.width"] == 0.5
    assert plt.rcParams["ytick.major.width"] == 0.5

    # Check font embedding for journal submission
    assert plt.rcParams["pdf.fonttype"] == 42
    assert plt.rcParams["ps.fonttype"] == 42


def test_set_figure_defaults_presentation() -> None:
    """Test that set_figure_defaults sets correct values for presentation context."""
    set_figure_defaults(context="presentation")

    # Presentation should have larger fonts than paper
    assert plt.rcParams["font.size"] == 12
    assert plt.rcParams["axes.labelsize"] == 12
    assert plt.rcParams["axes.titlesize"] == 14
    assert plt.rcParams["xtick.labelsize"] == 10
    assert plt.rcParams["ytick.labelsize"] == 10
    assert plt.rcParams["legend.fontsize"] == 10


def test_set_figure_defaults_poster() -> None:
    """Test that set_figure_defaults sets correct values for poster context."""
    set_figure_defaults(context="poster")

    # Poster should have largest fonts
    assert plt.rcParams["font.size"] == 16
    assert plt.rcParams["axes.labelsize"] == 16
    assert plt.rcParams["axes.titlesize"] == 18
    assert plt.rcParams["xtick.labelsize"] == 14
    assert plt.rcParams["ytick.labelsize"] == 14
    assert plt.rcParams["legend.fontsize"] == 14


def test_save_figure_creates_files(tmp_path: Path) -> None:
    """Test that save_figure creates both PDF and PNG files."""
    # Create a simple figure
    fig, ax = plt.subplots()
    ax.plot([1, 2, 3], [1, 2, 3])

    # Save to temp directory
    output_path = tmp_path / "test_figure"
    save_figure(str(output_path))

    # Check both files exist
    assert (tmp_path / "test_figure.pdf").exists()
    assert (tmp_path / "test_figure.png").exists()

    plt.close(fig)


def test_save_figure_with_path_object(tmp_path: Path) -> None:
    """Test that save_figure works with Path objects."""
    # Create a simple figure
    fig, ax = plt.subplots()
    ax.plot([1, 2, 3], [1, 2, 3])

    # Save using Path object
    output_path = tmp_path / "test_figure"
    save_figure(output_path)

    # Check both files exist
    assert (tmp_path / "test_figure.pdf").exists()
    assert (tmp_path / "test_figure.png").exists()

    plt.close(fig)


def test_save_figure_creates_parent_directories(tmp_path: Path) -> None:
    """Test that save_figure auto-creates parent directories."""
    # Create a simple figure
    fig, ax = plt.subplots()
    ax.plot([1, 2, 3], [1, 2, 3])

    # Save to nested path that doesn't exist
    output_path = tmp_path / "subdir1" / "subdir2" / "test_figure"
    save_figure(output_path)

    # Check directories were created and files exist
    assert (tmp_path / "subdir1" / "subdir2").is_dir()
    assert (tmp_path / "subdir1" / "subdir2" / "test_figure.pdf").exists()
    assert (tmp_path / "subdir1" / "subdir2" / "test_figure.png").exists()

    plt.close(fig)


def test_save_figure_with_custom_dpi(tmp_path: Path) -> None:
    """Test that save_figure respects custom DPI parameter."""
    # Create a simple figure
    fig, ax = plt.subplots()
    ax.plot([1, 2, 3], [1, 2, 3])

    # Save with custom DPI
    output_path = tmp_path / "test_figure"
    save_figure(output_path, dpi=150)

    # Files should exist (actual DPI checking would require image library)
    assert (tmp_path / "test_figure.pdf").exists()
    assert (tmp_path / "test_figure.png").exists()

    plt.close(fig)


def test_save_figure_close_parameter(tmp_path: Path) -> None:
    """Test that save_figure close parameter works."""
    # Create a simple figure
    fig, ax = plt.subplots()
    ax.plot([1, 2, 3], [1, 2, 3])

    # Save without closing
    output_path = tmp_path / "test_figure"
    save_figure(output_path, close=False)

    # Figure should still be open
    assert plt.fignum_exists(fig.number)

    plt.close(fig)


def test_get_figure_size_single() -> None:
    """Test get_figure_size for single column figure."""
    width, height = get_figure_size("single")
    assert width == pytest.approx(3.5, abs=0.1)  # Single column ~3.5 inches
    assert height > 0


def test_get_figure_size_double() -> None:
    """Test get_figure_size for double column figure."""
    width, height = get_figure_size("double")
    assert width == pytest.approx(7.0, abs=0.1)  # Double column ~7 inches
    assert height > 0


def test_get_figure_size_full() -> None:
    """Test get_figure_size for full width figure."""
    width, height = get_figure_size("full")
    assert width == pytest.approx(7.0, abs=0.1)  # Full width ~7 inches
    assert height > 0


def test_get_figure_size_with_aspect() -> None:
    """Test get_figure_size with custom aspect ratio."""
    width, height = get_figure_size("single", aspect_ratio=1.5)
    assert width / height == pytest.approx(1.5, abs=0.01)
