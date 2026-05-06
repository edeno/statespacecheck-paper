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
# Population-likelihood plot
# ---------------------------------------------------------------------------


def test_slice_panel_constructs_with_single_state(tmp_path: Path) -> None:
    _build_cache(tmp_path / "cache", n_states=1)
    app, viewer, ds = _make_viewer(tmp_path / "cache")
    try:
        sp = viewer.slice_panel
        # Population likelihood: one top-curve per state + the
        # shared blue predictive overlay.
        assert len(sp._lik_top_curves) == 1  # noqa: SLF001
        assert sp._n_states == 1  # noqa: SLF001
        assert sp._n_pos == ds.n_interior  # noqa: SLF001
        # Per-cell rows are empty until the first commit.
        assert sp._n_active_per_cell_rows == 0  # noqa: SLF001
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

        sp = viewer.slice_panel
        assert sp._buffer_post is not None  # noqa: SLF001
        assert sp._buffer_lik is not None  # noqa: SLF001
        # Predictive overlay on the population plot is populated and
        # peak-normalized to 1 (within fp tolerance).
        x_data, y_data = sp._lik_predictive_curve.getData()  # noqa: SLF001
        assert y_data.shape == (ds.n_position_full,)
        assert 0.0 < float(np.max(y_data)) <= 1.0 + 1e-6
        # Likelihood top curve is populated.
        x_lik, y_lik = sp._lik_top_curves[0].getData()  # noqa: SLF001
        assert y_lik.shape == (ds.n_position_full,)
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
        idx_a = sl.start + 1
        idx_b = sl.start + (sl.stop - sl.start) // 2
        viewer.set_center_time(ds.time[idx_a])
        x_a, y_a = sp._lik_predictive_curve.getData()  # noqa: SLF001
        viewer.set_center_time(ds.time[idx_b])
        x_b, y_b = sp._lik_predictive_curve.getData()  # noqa: SLF001
        assert not np.array_equal(y_a, y_b)
    finally:
        viewer.close()
        ds.close()


def test_slice_panel_y_axis_pinned_to_unit_range(tmp_path: Path) -> None:
    """Population-likelihood + per-cell-row plots have y-range hard-pinned."""
    from statespacecheck_paper.interactive.viewer import _SLICE_Y_MAX, _SLICE_Y_MIN

    _build_cache(tmp_path / "cache", n_states=1)
    app, viewer, ds = _make_viewer(tmp_path / "cache")
    try:
        target = viewer._next_request_id  # noqa: SLF001
        viewer.force_reload_now()
        assert _wait_for_request(app, viewer, target)
        viewer.set_center_time(float(ds.event_times[0]))
        viewer._update_slice_panel_at_center()  # noqa: SLF001

        sp = viewer.slice_panel
        # Population plot: explicit y-range matches the pin.
        y_lo, y_hi = sp._likelihood_plot.viewRange()[1]  # noqa: SLF001
        assert abs(y_lo - _SLICE_Y_MIN) < 1e-6
        assert abs(y_hi - _SLICE_Y_MAX) < 1e-6
        # Active per-cell rows: same.
        for i in range(sp._n_active_per_cell_rows):  # noqa: SLF001
            row = sp._per_cell_rows[i]  # noqa: SLF001
            y_lo, y_hi = row.plot.viewRange()[1]
            assert abs(y_lo - _SLICE_Y_MIN) < 1e-6
            assert abs(y_hi - _SLICE_Y_MAX) < 1e-6
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
        assert len(sp._lik_top_curves) == 2  # noqa: SLF001

        target = viewer._next_request_id  # noqa: SLF001
        viewer.force_reload_now()
        assert _wait_for_request(app, viewer, target)

        x0, y0 = sp._lik_top_curves[0].getData()  # noqa: SLF001
        x1, y1 = sp._lik_top_curves[1].getData()  # noqa: SLF001
        assert y0.shape == (ds.n_position_full,)
        assert y1.shape == (ds.n_position_full,)
    finally:
        viewer.close()
        ds.close()


# ---------------------------------------------------------------------------
# Live-readout (simplified)
# ---------------------------------------------------------------------------


def test_slice_panel_live_readout_updates_with_center(tmp_path: Path) -> None:
    _build_cache(tmp_path / "cache", n_states=1)
    app, viewer, ds = _make_viewer(tmp_path / "cache")
    try:
        target = viewer._next_request_id  # noqa: SLF001
        viewer.force_reload_now()
        assert _wait_for_request(app, viewer, target)

        viewer.set_center_time(float(ds.event_times[0]))
        viewer._update_slice_panel_at_center()  # noqa: SLF001
        text = viewer.slice_panel._readout_label.text()  # noqa: SLF001
        assert text.startswith("t = ")
        assert "predictive(x_true)" in text
        # Per-cell metrics now live in the row headers, not the readout.
        assert "HPD=" not in text and "spikes in bin" not in text

        viewer.set_center_time(float(ds.time[200]))
        viewer._update_slice_panel_at_center()  # noqa: SLF001
        text2 = viewer.slice_panel._readout_label.text()  # noqa: SLF001
        assert text2 != text
    finally:
        viewer.close()
        ds.close()


# ---------------------------------------------------------------------------
# Per-cell rows
# ---------------------------------------------------------------------------


def test_per_cell_row_visible_when_spikes_present(tmp_path: Path) -> None:
    _build_cache(tmp_path / "cache", n_states=1)
    app, viewer, ds = _make_viewer(tmp_path / "cache")
    try:
        target = viewer._next_request_id  # noqa: SLF001
        viewer.force_reload_now()
        assert _wait_for_request(app, viewer, target)
        # Per-cell rows default to ON in the new layout.
        assert viewer._per_cell_checkbox.isChecked()  # noqa: SLF001

        viewer.set_center_time(float(ds.event_times[0]))
        viewer._update_slice_panel_at_center()  # noqa: SLF001

        sp = viewer.slice_panel
        assert sp._n_active_per_cell_rows >= 1  # noqa: SLF001
        row = sp._per_cell_rows[0]  # noqa: SLF001
        # Cell-curve data is the cell's normalized place-field shape.
        x, y = row.cell_curve.getData()
        assert y.shape == (ds.n_position_full,)
        assert 0.0 < float(np.max(y)) <= 1.0 + 1e-6
    finally:
        viewer.close()
        ds.close()


def test_per_cell_row_header_contains_metrics(tmp_path: Path) -> None:
    _build_cache(tmp_path / "cache", n_states=1)
    app, viewer, ds = _make_viewer(tmp_path / "cache")
    try:
        target = viewer._next_request_id  # noqa: SLF001
        viewer.force_reload_now()
        assert _wait_for_request(app, viewer, target)

        viewer.set_center_time(float(ds.event_times[0]))
        viewer._update_slice_panel_at_center()  # noqa: SLF001

        header_text = viewer.slice_panel._per_cell_rows[0].header.text()  # noqa: SLF001
        assert header_text.startswith("Cell")
        assert "HPD=" in header_text
        assert "KL=" in header_text
    finally:
        viewer.close()
        ds.close()


def test_per_cell_checkbox_hides_rows(tmp_path: Path) -> None:
    _build_cache(tmp_path / "cache", n_states=1)
    app, viewer, ds = _make_viewer(tmp_path / "cache")
    try:
        target = viewer._next_request_id  # noqa: SLF001
        viewer.force_reload_now()
        assert _wait_for_request(app, viewer, target)

        viewer.set_center_time(float(ds.event_times[0]))
        viewer._update_slice_panel_at_center()  # noqa: SLF001
        sp = viewer.slice_panel
        assert sp._n_active_per_cell_rows >= 1  # noqa: SLF001

        viewer._per_cell_checkbox.setChecked(False)  # noqa: SLF001
        for row in sp._per_cell_rows:  # noqa: SLF001
            assert not row.container.isVisible()
        # Toggle back: active rows reappear; inactive stay hidden.
        viewer._per_cell_checkbox.setChecked(True)  # noqa: SLF001
        for i, row in enumerate(sp._per_cell_rows):  # noqa: SLF001
            expected = i < sp._n_active_per_cell_rows  # noqa: SLF001
            # ``isVisible`` requires the parent tree to be shown under
            # offscreen Qt; check the panel-side flag instead.
            if expected:
                assert sp._per_cell_visible  # noqa: SLF001
            del row
    finally:
        viewer.close()
        ds.close()


def test_data_source_cells_at_index_returns_unique_cells(tmp_path: Path) -> None:
    _build_cache(tmp_path / "cache", n_states=1)
    from statespacecheck_paper.interactive.data_source import Figure4DataSource

    src = Figure4DataSource(tmp_path / "cache", model="continuous")
    try:
        first_t_idx = int(src.event_time_idx[0])
        cells = src.cells_at_index(first_t_idx)
        assert cells.size >= 1
        assert cells.dtype == np.int32
        assert cells.size == np.unique(cells).size
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

        x_before, y_before = sp._lik_predictive_curve.getData()  # noqa: SLF001
        sp.update_for_index(sl.stop + 10, true_position=42.0)
        x_after, y_after = sp._lik_predictive_curve.getData()  # noqa: SLF001
        np.testing.assert_array_equal(y_after, y_before)
    finally:
        viewer.close()
        ds.close()
