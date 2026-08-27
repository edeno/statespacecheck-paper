"""Tests for the Figure-4 generation recipe and the thin CLI script."""

from __future__ import annotations

import importlib
import sys
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from statespacecheck_paper import figure04_generation
from statespacecheck_paper.figure04_layout import Figure4Composition

_SCRIPTS_DIR = Path(__file__).parent.parent / "scripts"


@pytest.fixture(scope="module")
def figure04_script() -> Iterator[ModuleType]:
    """Import the thin ``scripts/generate_figure04.py`` CLI module."""
    added = str(_SCRIPTS_DIR) not in sys.path
    if added:
        sys.path.insert(0, str(_SCRIPTS_DIR))
    try:
        yield importlib.import_module("generate_figure04")
    finally:
        if added:
            sys.path.remove(str(_SCRIPTS_DIR))


def test_generation_passes_figure_and_bbox_to_save(monkeypatch: pytest.MonkeyPatch) -> None:
    """``generate_figure04`` must hand the composed figure and its tight bbox to
    ``save_figure`` explicitly (preserving the custom crop)."""
    import matplotlib.pyplot as plt
    from matplotlib.transforms import Bbox

    fig = plt.figure()
    composition = Figure4Composition(figure=fig, bbox_inches=Bbox.from_bounds(0, 0, 1, 1))

    monkeypatch.setattr(
        figure04_generation, "prepare_figure04_render_data", lambda *a, **k: object()
    )
    monkeypatch.setattr(figure04_generation, "print_figure04_summary", lambda *a, **k: None)
    monkeypatch.setattr(figure04_generation, "compose_figure04", lambda *a, **k: composition)
    monkeypatch.setattr(figure04_generation, "set_figure_defaults", lambda *a, **k: None)

    saved: dict[str, Any] = {}
    monkeypatch.setattr(
        figure04_generation, "save_figure", lambda *a, **k: saved.update(args=a, kwargs=k)
    )

    figure04_generation.generate_figure04(use_cache=True)
    plt.close(fig)

    assert saved["kwargs"]["fig"] is fig
    assert saved["kwargs"]["bbox_inches"] is composition.bbox_inches
    assert saved["kwargs"]["close"] is True


def test_cli_force_recompute_forwards_use_cache(
    figure04_script: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``--force-recompute`` maps to ``use_cache=False``; its absence to True."""
    calls: list[bool] = []
    monkeypatch.setattr(
        figure04_script, "generate_figure04", lambda *, use_cache: calls.append(use_cache)
    )

    figure04_script.main(["--force-recompute"])
    figure04_script.main([])
    assert calls == [False, True]
