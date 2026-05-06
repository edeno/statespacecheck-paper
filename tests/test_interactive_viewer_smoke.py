"""Headless smoke test for the Figure 4 viewer skeleton.

The test runs under the Qt offscreen platform plugin so it can execute
in CI / headless environments without a display server. It exercises
construction, programmatic scrolling, and load-completion delivery —
no GUI assertions, just no-crash + correct array slicing.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import pytest

from ._synthetic_cache import build_synthetic_cache

PYSIDE6_AVAILABLE = True
try:
    import PySide6  # noqa: F401
except ImportError:
    PYSIDE6_AVAILABLE = False

PYQTGRAPH_AVAILABLE = True
try:
    import pyqtgraph  # noqa: F401
except ImportError:
    PYQTGRAPH_AVAILABLE = False


pytestmark = pytest.mark.skipif(
    not (PYSIDE6_AVAILABLE and PYQTGRAPH_AVAILABLE),
    reason="PySide6 / pyqtgraph not installed (optional [interactive] extra).",
)


def _build_synthetic_cache(cache_dir: Path) -> None:
    build_synthetic_cache(cache_dir, n_spikes_per_cell=20)


@pytest.fixture(scope="module", autouse=True)
def _qt_offscreen() -> None:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


@pytest.fixture
def viewer_setup(tmp_path: Path):
    cache_dir = tmp_path / "cache"
    _build_synthetic_cache(cache_dir)

    from PySide6 import QtWidgets

    from statespacecheck_paper.interactive.data_source import Figure4DataSource
    from statespacecheck_paper.interactive.viewer import Figure4Viewer

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    ds = Figure4DataSource(cache_dir, model="continuous")
    viewer = Figure4Viewer(ds)
    yield app, viewer, ds
    viewer.close()
    ds.close()


def _wait_for_request(app, viewer, request_id: int, timeout_s: float = 5.0) -> bool:
    deadline = time.perf_counter() + timeout_s
    while time.perf_counter() < deadline:
        app.processEvents()
        if viewer._latest_committed_request_id >= request_id:  # noqa: SLF001
            return True
        time.sleep(0.005)
    return False


def test_viewer_constructs_and_loads_initial_window(viewer_setup) -> None:
    app, viewer, ds = viewer_setup
    viewer.show()

    initial_request = viewer._next_request_id  # noqa: SLF001
    viewer.force_reload_now()
    assert _wait_for_request(app, viewer, initial_request)

    # Heatmap image was populated.
    img = viewer.posterior_panel._image  # noqa: SLF001
    assert img.image is not None
    # Shape: (n_visible, n_position) for single-state model.
    assert img.image.shape[1] == ds.n_interior


def test_viewer_set_center_time_drives_load(viewer_setup) -> None:
    app, viewer, ds = viewer_setup
    viewer.show()
    # Initial load.
    initial_request = viewer._next_request_id  # noqa: SLF001
    viewer.force_reload_now()
    assert _wait_for_request(app, viewer, initial_request)

    # Programmatically scroll to a different time.
    target_t = ds.time[200]
    viewer.set_center_time(target_t)
    viewer.force_reload_now()
    new_request = viewer._next_request_id  # noqa: SLF001
    assert _wait_for_request(app, viewer, new_request)

    # The viewer's internal center matches what we set.
    assert abs(viewer._t_center - target_t) < 1e-9  # noqa: SLF001


def test_viewer_drops_stale_requests(viewer_setup) -> None:
    """Older request_ids must not overwrite a newer committed window."""
    app, viewer, _ = viewer_setup
    viewer.show()

    # Fire several updates rapidly; only the last should govern the
    # committed view, but any earlier results that arrive must not
    # overwrite a later commit.
    for offset in [50, 100, 150, 200, 250]:
        viewer.set_center_time(viewer._ds.time[offset])  # noqa: SLF001
        viewer.force_reload_now()
    final_request = viewer._next_request_id  # noqa: SLF001
    assert _wait_for_request(app, viewer, final_request, timeout_s=10.0)

    # The committed request id is at least the final one we issued.
    assert viewer._latest_committed_request_id >= final_request  # noqa: SLF001
