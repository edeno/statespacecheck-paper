"""Tests for style module."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pytest

from statespacecheck_paper.style import (
    WONG,
    save_figure,
    set_figure_defaults,
)


def test_wong_palette_is_eight_hex_colors() -> None:
    """WONG palette must be 8 hex strings (relied on by figure scripts)."""
    assert len(WONG) == 8
    for color in WONG:
        assert isinstance(color, str)
        assert color.startswith("#")
        assert len(color) == 7


def test_set_figure_defaults_paper_font_sizes_within_journal_range() -> None:
    """Paper context pins all in-figure text to the 8-12 pt journal range."""
    set_figure_defaults(context="paper")
    expected = {
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
    }
    for key, value in expected.items():
        assert plt.rcParams[key] == value
        assert 8 <= value <= 12


def test_set_figure_defaults_paper_journal_settings() -> None:
    """Paper context must use Arial, thin lines, and TrueType embedding."""
    set_figure_defaults(context="paper")
    assert plt.rcParams["font.family"] == ["sans-serif"]
    assert "Arial" in plt.rcParams["font.sans-serif"]
    assert plt.rcParams["axes.linewidth"] == 0.5
    assert plt.rcParams["xtick.major.width"] == 0.5
    assert plt.rcParams["ytick.major.width"] == 0.5
    # TrueType (42) is required for Nature/Science final figures.
    assert plt.rcParams["pdf.fonttype"] == 42
    assert plt.rcParams["ps.fonttype"] == 42


def test_set_figure_defaults_font_sizes_increase_with_context() -> None:
    """Paper < presentation < poster fonts (regression: any reordering breaks contract)."""
    sizes = []
    for context in ("paper", "presentation", "poster"):
        set_figure_defaults(context=context)
        sizes.append(plt.rcParams["font.size"])
    assert sizes == sorted(sizes)
    assert len(set(sizes)) == 3


def _make_simple_figure() -> plt.Figure:
    fig, ax = plt.subplots()
    ax.plot([1, 2, 3], [1, 2, 3])
    return fig


@pytest.mark.parametrize("path_type", ["str", "Path"])
def test_save_figure_creates_pdf_and_png(tmp_path: Path, path_type: str) -> None:
    """save_figure creates both PDF and PNG, accepting both str and Path."""
    fig = _make_simple_figure()
    output = tmp_path / "test_figure"
    save_figure(str(output) if path_type == "str" else output, fig=fig)
    assert (tmp_path / "test_figure.pdf").exists()
    assert (tmp_path / "test_figure.png").exists()
    plt.close(fig)


def test_save_figure_creates_parent_directories(tmp_path: Path) -> None:
    """save_figure auto-creates missing parent directories."""
    fig = _make_simple_figure()
    save_figure(tmp_path / "subdir1" / "subdir2" / "test_figure", fig=fig)
    assert (tmp_path / "subdir1" / "subdir2" / "test_figure.pdf").exists()
    assert (tmp_path / "subdir1" / "subdir2" / "test_figure.png").exists()
    plt.close(fig)


def test_save_figure_respects_custom_dpi(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The ``dpi`` argument is forwarded to every ``Figure.savefig`` call.

    Asserts the kwarg passthrough directly rather than reading it back
    from the rendered PNG, which would pull in a Pillow dependency that
    the package does not otherwise declare.
    """
    captured_dpis: list[object] = []
    fig = _make_simple_figure()
    real_savefig = fig.savefig

    def _spy_savefig(*args: object, **kwargs: object) -> object:
        captured_dpis.append(kwargs["dpi"])
        return real_savefig(*args, **kwargs)

    monkeypatch.setattr(fig, "savefig", _spy_savefig)

    save_figure(tmp_path / "test_figure", dpi=150, fig=fig)
    # save_figure writes both a PDF and a PNG; dpi must reach both.
    assert captured_dpis == [150, 150]
    plt.close(fig)


def test_save_figure_close_false_keeps_figure_open(tmp_path: Path) -> None:
    """close=False leaves the figure registered with pyplot."""
    fig = _make_simple_figure()
    save_figure(tmp_path / "test_figure", close=False, fig=fig)
    assert plt.fignum_exists(fig.number)
    plt.close(fig)


def test_save_figure_saves_the_given_figure_not_the_current_one(tmp_path: Path) -> None:
    """``fig`` is saved and closed even when another figure is current in pyplot."""
    fig = _make_simple_figure()
    other = _make_simple_figure()
    save_figure(tmp_path / "explicit_figure", fig=fig)
    assert (tmp_path / "explicit_figure.pdf").exists()
    assert (tmp_path / "explicit_figure.png").exists()
    assert not plt.fignum_exists(fig.number)
    assert plt.fignum_exists(other.number)
    plt.close(other)
