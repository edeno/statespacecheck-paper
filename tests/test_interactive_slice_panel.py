"""Tests for ``SlicePanel`` and its wiring into ``Figure4Viewer``."""

from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
import pytest

from ._synthetic_cache import build_synthetic_cache as _build_cache_impl

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


def _build_cache(cache_dir: Path, *, n_states: int) -> None:
    """Slice-panel tests need single- and multi-state caches."""
    _build_cache_impl(cache_dir, model="continuous", n_states=n_states)


@pytest.fixture(scope="module", autouse=True)
def _qt_offscreen() -> None:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def _make_viewer(cache_dir: Path):
    from PySide6 import QtWidgets

    from statespacecheck_paper.interactive.data_source import Figure4DataSource
    from statespacecheck_paper.interactive.viewer import Figure4Viewer

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    ds = Figure4DataSource(cache_dir, model="continuous")
    viewer = Figure4Viewer(ds)
    return app, viewer, ds


def _wait_for_request(app, viewer, request_id: int, timeout_s: float = 5.0) -> bool:
    deadline = time.perf_counter() + timeout_s
    while time.perf_counter() < deadline:
        app.processEvents()
        if viewer._latest_committed_request_id >= request_id:  # noqa: SLF001
            return True
        time.sleep(0.005)
    return False


# ---------------------------------------------------------------------------
# Single-state behavior
# ---------------------------------------------------------------------------


def test_slice_panel_constructs_with_single_state(tmp_path: Path) -> None:
    _build_cache(tmp_path / "cache", n_states=1)
    app, viewer, ds = _make_viewer(tmp_path / "cache")
    try:
        assert len(viewer.slice_panel._posterior_curves) == 1  # noqa: SLF001
        assert len(viewer.slice_panel._likelihood_curves) == 1  # noqa: SLF001
        assert viewer.slice_panel._n_states == 1  # noqa: SLF001
        assert viewer.slice_panel._n_pos == ds.n_interior  # noqa: SLF001
    finally:
        viewer.close()
        ds.close()


def test_slice_panel_updates_after_window_load(tmp_path: Path) -> None:
    _build_cache(tmp_path / "cache", n_states=1)
    app, viewer, ds = _make_viewer(tmp_path / "cache")
    try:
        target = viewer._next_request_id  # noqa: SLF001
        viewer.force_reload_now()
        assert _wait_for_request(app, viewer, target)

        # The slice panel's posterior curve now matches the
        # corresponding row of the loaded window.
        sp = viewer.slice_panel
        assert sp._buffer_post is not None  # noqa: SLF001
        assert sp._buffer_lik is not None  # noqa: SLF001
        assert sp._buffer_slice is not None  # noqa: SLF001
        sl = sp._buffer_slice  # noqa: SLF001
        t_idx = ds.index_at_time(viewer._t_center)  # noqa: SLF001
        if sl.start <= t_idx < sl.stop:
            local = t_idx - sl.start
            expected = sp._buffer_post[local]  # noqa: SLF001
            x_data, y_data = sp._posterior_curves[0].getData()  # noqa: SLF001
            np.testing.assert_array_almost_equal(y_data, expected)
    finally:
        viewer.close()
        ds.close()


def test_slice_panel_animates_on_set_center_time(tmp_path: Path) -> None:
    _build_cache(tmp_path / "cache", n_states=1)
    app, viewer, ds = _make_viewer(tmp_path / "cache")
    try:
        target = viewer._next_request_id  # noqa: SLF001
        viewer.force_reload_now()
        assert _wait_for_request(app, viewer, target)

        sp = viewer.slice_panel
        sl = sp._buffer_slice  # noqa: SLF001
        assert sl is not None
        # Move within the same window; slice must update without a new load.
        idx_a = sl.start + 1
        idx_b = sl.start + (sl.stop - sl.start) // 2
        viewer.set_center_time(ds.time[idx_a])
        x_a, y_a = sp._posterior_curves[0].getData()  # noqa: SLF001
        viewer.set_center_time(ds.time[idx_b])
        x_b, y_b = sp._posterior_curves[0].getData()  # noqa: SLF001
        assert not np.array_equal(y_a, y_b)
    finally:
        viewer.close()
        ds.close()


def test_slice_panel_alpha_slider_updates_brushes(tmp_path: Path) -> None:
    _build_cache(tmp_path / "cache", n_states=1)
    app, viewer, ds = _make_viewer(tmp_path / "cache")
    try:
        sp = viewer.slice_panel
        original_alpha = sp._likelihood_alpha  # noqa: SLF001
        viewer._alpha_slider.setValue(40)  # noqa: SLF001
        assert sp._likelihood_alpha == 40  # noqa: SLF001
        assert sp._likelihood_alpha != original_alpha  # noqa: SLF001
    finally:
        viewer.close()
        ds.close()


# ---------------------------------------------------------------------------
# Multi-state behavior
# ---------------------------------------------------------------------------


def test_slice_panel_stacks_states_for_contfrag(tmp_path: Path) -> None:
    _build_cache(tmp_path / "cache", n_states=2)
    app, viewer, ds = _make_viewer(tmp_path / "cache")
    try:
        sp = viewer.slice_panel
        assert sp._n_states == 2  # noqa: SLF001
        assert len(sp._posterior_curves) == 2  # noqa: SLF001
        assert len(sp._likelihood_curves) == 2  # noqa: SLF001

        target = viewer._next_request_id  # noqa: SLF001
        viewer.force_reload_now()
        assert _wait_for_request(app, viewer, target)

        # Each state's curve has length n_interior, not n_state_bins.
        x0, y0 = sp._posterior_curves[0].getData()  # noqa: SLF001
        x1, y1 = sp._posterior_curves[1].getData()  # noqa: SLF001
        assert y0.shape[0] == ds.n_interior
        assert y1.shape[0] == ds.n_interior
        # The two state slices generally differ.
        assert not np.array_equal(y0, y1)
    finally:
        viewer.close()
        ds.close()


def test_slice_panel_live_readout_updates_with_center(tmp_path: Path) -> None:
    _build_cache(tmp_path / "cache", n_states=1)
    app, viewer, ds = _make_viewer(tmp_path / "cache")
    try:
        target = viewer._next_request_id  # noqa: SLF001
        viewer.force_reload_now()
        assert _wait_for_request(app, viewer, target)

        text = viewer.slice_panel._readout_label.text()  # noqa: SLF001
        # The default readout always includes the time line.
        assert text.startswith("t = ")
        # When the buffer holds the center, the predictive value line
        # should be present.
        assert "predictive(x_true)" in text
        # And with events sprinkled across the synthetic session, a
        # nearest-spike block should also be emitted.
        assert "nearest spike" in text and "HPD =" in text and "KL =" in text

        # Move to a different center and verify the time line updates.
        viewer.set_center_time(float(ds.time[200]))
        text2 = viewer.slice_panel._readout_label.text()  # noqa: SLF001
        assert text2 != text
    finally:
        viewer.close()
        ds.close()


def test_slice_panel_shows_per_cell_curves_when_spikes_present(tmp_path: Path) -> None:
    _build_cache(tmp_path / "cache", n_states=1)
    app, viewer, ds = _make_viewer(tmp_path / "cache")
    try:
        target = viewer._next_request_id  # noqa: SLF001
        viewer.force_reload_now()
        assert _wait_for_request(app, viewer, target)

        # Pick the time bin of the first spike and recenter on it.
        first_event_t = float(ds.event_times[0])
        viewer.set_center_time(first_event_t)
        # Drive the per-tick path explicitly (set_center_time calls it
        # but we want to be sure the event-bin lookup ran).
        viewer._update_slice_panel_at_center()  # noqa: SLF001

        sp = viewer.slice_panel
        # At least one per-cell curve should be active and visible.
        assert sp._n_active_per_cell_curves >= 1  # noqa: SLF001
        assert sp._per_cell_curves[0].isVisible()  # noqa: SLF001
        x_data, y_data = sp._per_cell_curves[0].getData()  # noqa: SLF001
        assert y_data.shape == (ds.n_position_full,)
        # Curve is normalized so its max is at most 1 (or exactly 0 if
        # the cell has an all-zero place field, which our synthetic
        # builder avoids).
        assert 0.0 < float(np.max(y_data)) <= 1.0 + 1e-6

        # Move to a time bin with no spikes (between two events) — the
        # active count drops back to zero.
        midpoint = 0.5 * (float(ds.event_times[0]) + float(ds.event_times[1]))
        # Find a time bin whose nearest event is far enough away that
        # no spikes land in it. Use a synthetic-time gap if needed.
        viewer.set_center_time(midpoint)
        viewer._update_slice_panel_at_center()  # noqa: SLF001
        # If the chosen bin happens to contain a spike (depends on
        # synthetic RNG), we can't assert the count is 0; only the
        # invariant that all hidden curves stay hidden.
        for i in range(sp._n_active_per_cell_curves, len(sp._per_cell_curves)):  # noqa: SLF001
            assert not sp._per_cell_curves[i].isVisible()  # noqa: SLF001
    finally:
        viewer.close()
        ds.close()


def test_data_source_cells_at_index_returns_unique_cells(tmp_path: Path) -> None:
    """``cells_at_index`` returns unique cell IDs for the given time bin."""
    _build_cache(tmp_path / "cache", n_states=1)
    from statespacecheck_paper.interactive.data_source import Figure4DataSource

    src = Figure4DataSource(tmp_path / "cache", model="continuous")
    try:
        # First event's bin should yield at least one cell.
        first_t_idx = int(src.event_time_idx[0])
        cells = src.cells_at_index(first_t_idx)
        assert cells.size >= 1
        assert cells.dtype == np.int32
        # No duplicates.
        assert cells.size == np.unique(cells).size
        # Empty bin (well before any event): empty result.
        empty = src.cells_at_index(int(src.event_time_idx[0]) - 100)
        assert empty.size == 0
    finally:
        src.close()


def test_slice_panel_does_not_update_outside_buffer(tmp_path: Path) -> None:
    _build_cache(tmp_path / "cache", n_states=1)
    app, viewer, ds = _make_viewer(tmp_path / "cache")
    try:
        target = viewer._next_request_id  # noqa: SLF001
        viewer.force_reload_now()
        assert _wait_for_request(app, viewer, target)

        sp = viewer.slice_panel
        sl = sp._buffer_slice  # noqa: SLF001
        assert sl is not None

        # Capture the current curve data.
        x_before, y_before = sp._posterior_curves[0].getData()  # noqa: SLF001

        # Update for a t_idx outside the buffer: should be a no-op.
        sp.update_for_index(sl.stop + 10, true_position=42.0)
        x_after, y_after = sp._posterior_curves[0].getData()  # noqa: SLF001
        np.testing.assert_array_equal(y_after, y_before)
    finally:
        viewer.close()
        ds.close()
