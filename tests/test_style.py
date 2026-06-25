"""Tests for style module."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import pytest

from statespacecheck_paper.style import (
    WONG,
    get_figure_size,
    save_figure,
    set_figure_defaults,
)

Context = Literal["paper", "presentation", "poster"]
WidthType = Literal["single", "double", "full"]

# Expected per-context font sizes mirror the source. Pytest's parametrize id
# also documents the contract.
_CONTEXT_FONT_SIZES = {
    "paper": {
        # Journal figure standards require all in-figure text in the 8-12 pt range.
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
    },
    "presentation": {
        "font.size": 12,
        "axes.labelsize": 12,
        "axes.titlesize": 14,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
    },
    "poster": {
        "font.size": 16,
        "axes.labelsize": 16,
        "axes.titlesize": 18,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 14,
    },
}


def test_wong_palette_is_eight_hex_colors() -> None:
    """WONG palette must be 8 hex strings (relied on by figure scripts)."""
    assert len(WONG) == 8
    for color in WONG:
        assert isinstance(color, str)
        assert color.startswith("#")
        assert len(color) == 7


@pytest.mark.parametrize("context", list(_CONTEXT_FONT_SIZES))
def test_set_figure_defaults_applies_context_font_sizes(context: Context) -> None:
    """set_figure_defaults applies the expected font sizes for each context."""
    set_figure_defaults(context=context)
    for key, expected in _CONTEXT_FONT_SIZES[context].items():
        assert plt.rcParams[key] == expected


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
    save_figure(str(output) if path_type == "str" else output)
    assert (tmp_path / "test_figure.pdf").exists()
    assert (tmp_path / "test_figure.png").exists()
    plt.close(fig)


def test_save_figure_creates_parent_directories(tmp_path: Path) -> None:
    """save_figure auto-creates missing parent directories."""
    fig = _make_simple_figure()
    save_figure(tmp_path / "subdir1" / "subdir2" / "test_figure")
    assert (tmp_path / "subdir1" / "subdir2" / "test_figure.pdf").exists()
    assert (tmp_path / "subdir1" / "subdir2" / "test_figure.png").exists()
    plt.close(fig)


def test_save_figure_respects_custom_dpi(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The ``dpi`` argument is forwarded to every ``plt.savefig`` call.

    Asserts the kwarg passthrough directly rather than reading it back
    from the rendered PNG, which would pull in a Pillow dependency that
    the package does not otherwise declare.
    """
    captured_dpis: list[object] = []
    real_savefig = plt.savefig

    def _spy_savefig(*args: object, **kwargs: object) -> object:
        captured_dpis.append(kwargs["dpi"])
        return real_savefig(*args, **kwargs)

    monkeypatch.setattr(plt, "savefig", _spy_savefig)

    fig = _make_simple_figure()
    save_figure(tmp_path / "test_figure", dpi=150)
    # save_figure writes both a PDF and a PNG; dpi must reach both.
    assert captured_dpis == [150, 150]
    plt.close(fig)


def test_save_figure_close_false_keeps_figure_open(tmp_path: Path) -> None:
    """close=False leaves the figure registered with pyplot."""
    fig = _make_simple_figure()
    save_figure(tmp_path / "test_figure", close=False)
    assert plt.fignum_exists(fig.number)
    plt.close(fig)


def test_save_figure_accepts_explicit_figure(tmp_path: Path) -> None:
    """The optional ``fig`` argument saves that figure without pyplot globals."""
    fig = _make_simple_figure()
    save_figure(tmp_path / "explicit_figure", fig=fig)
    assert (tmp_path / "explicit_figure.pdf").exists()
    assert (tmp_path / "explicit_figure.png").exists()
    assert not plt.fignum_exists(fig.number)


@pytest.mark.parametrize(
    ("width_type", "expected_width"),
    [("single", 3.5), ("double", 7.0), ("full", 7.0)],
)
def test_get_figure_size_returns_journal_widths(
    width_type: WidthType, expected_width: float
) -> None:
    """Standard width strings map to the documented journal column inches."""
    width, height = get_figure_size(width_type)
    assert width == pytest.approx(expected_width, abs=0.1)
    assert height > 0


@pytest.mark.parametrize("aspect_ratio", [1.0, 1.5, 2.0])
def test_get_figure_size_with_aspect_ratio(aspect_ratio: float) -> None:
    """Aspect ratio sets width/height precisely."""
    width, height = get_figure_size("single", aspect_ratio=aspect_ratio)
    assert width / height == pytest.approx(aspect_ratio)
