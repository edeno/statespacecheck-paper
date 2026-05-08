"""Tests for ``MetricPanel`` + click-to-recenter + pinned-event markers."""

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


def _build_cache(cache_dir: Path) -> None:
    """Metric-panel tests need a single-state cache with non-zero spike-prob floor."""
    _build_cache_impl(cache_dir, n_states=1, p_min=0.001)


@pytest.fixture(scope="module", autouse=True)
def _qt_offscreen() -> None:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def _make_viewer(cache_dir: Path):
    from PySide6 import QtWidgets

    from statespacecheck_paper.interactive.data_source import DecoderDataSource
    from statespacecheck_paper.interactive.viewer import DecoderViewer

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    ds = DecoderDataSource(cache_dir, model="continuous")
    viewer = DecoderViewer(ds)
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
# MetricPanel
# ---------------------------------------------------------------------------


def test_three_metric_panels_constructed(tmp_path: Path) -> None:
    _build_cache(tmp_path / "cache")
    app, viewer, ds = _make_viewer(tmp_path / "cache")
    try:
        assert set(viewer.metric_panels.keys()) == {
            "event_hpd_overlap",
            "event_kl_divergence",
            "event_spike_prob",
        }
        # Spike-prob panel has its threshold line; KL has none.
        assert viewer.metric_panels["event_hpd_overlap"]._threshold_line is not None  # noqa: SLF001
        assert viewer.metric_panels["event_spike_prob"]._threshold_line is not None  # noqa: SLF001
        assert viewer.metric_panels["event_kl_divergence"]._threshold_line is None  # noqa: SLF001
    finally:
        viewer.close()
        ds.close()


def test_metric_panel_displays_neglog10_for_spike_prob(tmp_path: Path) -> None:
    _build_cache(tmp_path / "cache")
    app, viewer, ds = _make_viewer(tmp_path / "cache")
    try:
        target = viewer._next_request_id  # noqa: SLF001
        viewer.force_reload_now()
        assert _wait_for_request(app, viewer, target)

        sp_panel = viewer.metric_panels["event_spike_prob"]
        x_data, y_data = sp_panel._scatter.getData()  # noqa: SLF001
        # All displayed values are non-negative (since spike_prob is in (0,1]).
        assert (y_data >= 0).all()
        # Compare against the raw event values: y == -log10(raw).
        sl = viewer.slice_panel._buffer_slice  # noqa: SLF001
        events = ds.events_in_window(sl)
        if not events.empty:
            np.testing.assert_array_almost_equal(
                np.asarray(y_data, dtype=np.float64),
                -np.log10(np.maximum(events["event_spike_prob"].to_numpy(), 1e-12)),
                decimal=4,
            )
    finally:
        viewer.close()
        ds.close()


# ---------------------------------------------------------------------------
# Click handling
# ---------------------------------------------------------------------------


def _first_visible_event_row(ds, sl) -> int | None:
    events = ds.events_in_window(sl)
    if events.empty:
        return None
    return int(events.index[0])


def test_metric_click_recenters_on_event_time(tmp_path: Path) -> None:
    _build_cache(tmp_path / "cache")
    app, viewer, ds = _make_viewer(tmp_path / "cache")
    try:
        target = viewer._next_request_id  # noqa: SLF001
        viewer.force_reload_now()
        assert _wait_for_request(app, viewer, target)

        sl = viewer.slice_panel._buffer_slice  # noqa: SLF001
        assert sl is not None
        row = _first_visible_event_row(ds, sl)
        assert row is not None

        # Simulate a click via the public path the panel exposes.
        viewer._handle_event_click(row)  # noqa: SLF001
        expected_time = float(ds.events.iloc[row]["time"])
        assert abs(viewer._t_center - expected_time) < 1e-9  # noqa: SLF001
        assert viewer._pinned_event_row == row  # noqa: SLF001
    finally:
        viewer.close()
        ds.close()


def test_pin_markers_visible_after_click(tmp_path: Path) -> None:
    _build_cache(tmp_path / "cache")
    app, viewer, ds = _make_viewer(tmp_path / "cache")
    try:
        target = viewer._next_request_id  # noqa: SLF001
        viewer.force_reload_now()
        assert _wait_for_request(app, viewer, target)

        sl = viewer.slice_panel._buffer_slice  # noqa: SLF001
        assert sl is not None
        row = _first_visible_event_row(ds, sl)
        assert row is not None

        viewer._handle_event_click(row)  # noqa: SLF001
        # Click triggers a recenter -> new load. Wait for the new
        # window to commit so the pin lands inside the buffered slice.
        target2 = viewer._next_request_id  # noqa: SLF001
        assert _wait_for_request(app, viewer, target2)

        assert viewer.posterior_panel._pin_line.isVisible()  # noqa: SLF001
        assert viewer.likelihood_panel._pin_line.isVisible()  # noqa: SLF001
        assert viewer.raster_panel._pin_line.isVisible()  # noqa: SLF001
        for panel in viewer.metric_panels.values():
            assert panel._pin_line.isVisible()  # noqa: SLF001
            assert panel._pin_dot.isVisible()  # noqa: SLF001
        assert viewer.slice_panel.is_pin_displayed()
    finally:
        viewer.close()
        ds.close()


def test_manual_scroll_unpins_event(tmp_path: Path) -> None:
    _build_cache(tmp_path / "cache")
    app, viewer, ds = _make_viewer(tmp_path / "cache")
    try:
        target = viewer._next_request_id  # noqa: SLF001
        viewer.force_reload_now()
        assert _wait_for_request(app, viewer, target)

        sl = viewer.slice_panel._buffer_slice  # noqa: SLF001
        assert sl is not None
        row = _first_visible_event_row(ds, sl)
        assert row is not None
        viewer._handle_event_click(row)  # noqa: SLF001
        assert viewer._pinned_event_row == row  # noqa: SLF001

        # Slider movement signals the unpin path.
        viewer._on_slider_changed(viewer._slider.value() + 1)  # noqa: SLF001
        assert viewer._pinned_event_row is None  # noqa: SLF001
        assert not viewer.posterior_panel._pin_line.isVisible()  # noqa: SLF001
        for panel in viewer.metric_panels.values():
            assert not panel._pin_line.isVisible()  # noqa: SLF001
    finally:
        viewer.close()
        ds.close()


def test_pin_invisible_when_event_outside_loaded_window(tmp_path: Path) -> None:
    """If the pinned event is outside the current window, markers hide."""
    _build_cache(tmp_path / "cache")
    app, viewer, ds = _make_viewer(tmp_path / "cache")
    try:
        target = viewer._next_request_id  # noqa: SLF001
        viewer.force_reload_now()
        assert _wait_for_request(app, viewer, target)

        # Pick the last event in the table (likely outside the initial
        # window) and pin it manually without recentering.
        last_row = int(ds.events.index[-1])
        viewer._set_pinned_event(last_row)  # noqa: SLF001

        sl = viewer.slice_panel._buffer_slice  # noqa: SLF001
        assert sl is not None
        event_t = float(ds.events.iloc[last_row]["time"])
        event_idx = ds.index_at_time(event_t)
        if not (sl.start <= event_idx < sl.stop):
            # Markers should be hidden because the event is outside
            # the buffered window.
            for panel in viewer.metric_panels.values():
                assert not panel._pin_line.isVisible()  # noqa: SLF001
            assert not viewer.posterior_panel._pin_line.isVisible()  # noqa: SLF001
    finally:
        viewer.close()
        ds.close()
