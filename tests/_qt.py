"""Shared Qt helpers for the interactive-viewer tests.

``qt_offscreen`` is a module-scoped autouse fixture: importing it into a
test module registers it there, so Qt renders offscreen before that module
creates a ``QApplication``. The helpers import PySide6 and the viewer
lazily, so a module can still be collected (and skipped) without the
optional ``[interactive]`` extra.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

if TYPE_CHECKING:
    from statespacecheck_paper.interactive.cache import ModelName


@pytest.fixture(scope="module", autouse=True)
def qt_offscreen() -> None:
    """Select Qt's offscreen platform so viewer tests run headless."""
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def make_viewer(
    cache_dir: Path, *, model: ModelName = "continuous", model_swaps: bool = False
) -> tuple[Any, Any, Any]:
    """Open a viewer on ``cache_dir``'s ``model`` cache.

    ``model_swaps`` hands the viewer ``cache_dir`` so it can switch to the
    other model's cache. Returns ``(app, viewer, data_source)``; the caller
    closes the viewer and the data source.
    """
    from PySide6 import QtWidgets

    from statespacecheck_paper.interactive.data_source import DecoderDataSource
    from statespacecheck_paper.interactive.viewer import DecoderViewer

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    ds = DecoderDataSource(cache_dir, model=model)
    viewer = DecoderViewer(ds, cache_dir=cache_dir) if model_swaps else DecoderViewer(ds)
    return app, viewer, ds


def wait_for_request(app: Any, viewer: Any, request_id: int, timeout_s: float = 5.0) -> bool:
    """Process Qt events until the viewer commits ``request_id``; False on timeout."""
    deadline = time.perf_counter() + timeout_s
    while time.perf_counter() < deadline:
        app.processEvents()
        if viewer._latest_committed_request_id >= request_id:  # noqa: SLF001
            return True
        time.sleep(0.005)
    return False
