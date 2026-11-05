"""Shared Qt helpers for the interactive-viewer tests.

Importing this module selects Qt's offscreen platform, so viewer tests run
headless. The helpers import PySide6 and the viewer lazily, so a module can
still be collected (and skipped, through ``requires_qt``) without the optional
``[interactive]`` extra.
"""

from __future__ import annotations

import importlib.util
import os
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

if TYPE_CHECKING:
    from statespacecheck_paper.figure04_models import Figure4ModelId

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

requires_qt = pytest.mark.skipif(
    any(importlib.util.find_spec(name) is None for name in ("PySide6", "pyqtgraph")),
    reason="PySide6 / pyqtgraph not installed (optional [interactive] extra).",
)


@contextmanager
def open_viewer(
    cache_dir: Path, *, model: Figure4ModelId = "continuous", model_swaps: bool = False
) -> Iterator[tuple[Any, Any, Any]]:
    """Open a viewer on ``cache_dir``'s ``model`` cache and close it on exit.

    ``model_swaps`` hands the viewer ``cache_dir`` so it can switch to the
    other model's cache. Yields ``(app, viewer, data_source)``. On exit the
    viewer's current data source is closed; each model swap closes the one it
    replaces.
    """
    from PySide6 import QtWidgets

    from statespacecheck_paper.interactive.data_source import DecoderDataSource
    from statespacecheck_paper.interactive.viewer import DecoderViewer

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    ds = DecoderDataSource(cache_dir, model=model)
    viewer = DecoderViewer(ds, cache_dir=cache_dir) if model_swaps else DecoderViewer(ds)
    try:
        yield app, viewer, ds
    finally:
        viewer.close()
        viewer._ds.close()  # noqa: SLF001


def wait_for_request(app: Any, viewer: Any, request_id: int, timeout_s: float = 5.0) -> bool:
    """Process Qt events until the viewer commits ``request_id``; False on timeout."""
    deadline = time.perf_counter() + timeout_s
    while time.perf_counter() < deadline:
        app.processEvents()
        if viewer._latest_committed_request_id >= request_id:  # noqa: SLF001
            return True
        time.sleep(0.005)
    return False


def reload_and_wait(app: Any, viewer: Any) -> bool:
    """Force a reload of the current window and wait for it to commit; False on timeout."""
    request_id = viewer._next_request_id  # noqa: SLF001
    viewer.force_reload_now()
    return wait_for_request(app, viewer, request_id)
