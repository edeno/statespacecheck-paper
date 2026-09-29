"""Tests for ``MetricPanel`` + click-to-recenter + pinned-event markers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from statespacecheck_paper.style import COLORS, hex_to_rgb

from ._qt import (
    make_viewer,
    qt_offscreen,  # noqa: F401 -- registers the autouse fixture here
    wait_for_request,
)
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
    """Metric-panel tests need a single-state cache with non-zero predictive p-value floor."""
    _build_cache_impl(cache_dir, n_states=1, p_min=0.001)


# ---------------------------------------------------------------------------
# MetricPanel
# ---------------------------------------------------------------------------


def test_three_metric_panels_constructed(tmp_path: Path) -> None:
    _build_cache(tmp_path / "cache")
    app, viewer, ds = make_viewer(tmp_path / "cache")
    try:
        # One panel per metric, in the paper's order, with plain-text labels.
        assert list(viewer.metric_panels) == [
            "event_hpd_overlap",
            "event_predictive_pvalue",
            "event_kl_divergence",
        ]
        assert [
            panel.getPlotItem().getAxis("left").labelText for panel in viewer.metric_panels.values()
        ] == ["HPD overlap", "−log(p)", "KL divergence"]
        # HPD overlap and p-value panels have threshold lines, in the paper's
        # threshold gray; KL has none.
        for metric in ("event_hpd_overlap", "event_predictive_pvalue"):
            line = viewer.metric_panels[metric]._threshold_line  # noqa: SLF001
            assert line is not None
            assert line.pen.color().getRgb()[:3] == hex_to_rgb(COLORS["threshold"])
        assert viewer.metric_panels["event_kl_divergence"]._threshold_line is None  # noqa: SLF001
    finally:
        viewer.close()
        ds.close()


def test_metric_panel_displays_neglog_for_predictive_pvalue(tmp_path: Path) -> None:
    _build_cache(tmp_path / "cache")
    app, viewer, ds = make_viewer(tmp_path / "cache")
    try:
        target = viewer._next_request_id  # noqa: SLF001
        viewer.force_reload_now()
        assert wait_for_request(app, viewer, target)

        sp_panel = viewer.metric_panels["event_predictive_pvalue"]
        x_data, y_data = sp_panel._scatter.getData()  # noqa: SLF001
        # All displayed values are non-negative (since predictive_pvalue is in (0,1]).
        assert (y_data >= 0).all()
        # Compare against the raw event values: y == -log(raw) (natural log).
        sl = viewer.slice_panel._buffer_slice  # noqa: SLF001
        events = ds.events_in_window(sl)
        if not events.empty:
            np.testing.assert_array_almost_equal(
                np.asarray(y_data, dtype=np.float64),
                -np.log(np.maximum(events["event_predictive_pvalue"].to_numpy(), 1e-12)),
                decimal=4,
            )
    finally:
        viewer.close()
        ds.close()


# ---------------------------------------------------------------------------
# Symmetric-log HPD-overlap axis
# ---------------------------------------------------------------------------

_SYMLOG_PROBES = np.array([0.0, 1e-4, 0.005, 0.01, 0.05, 0.3, 1.0])


def test_symlog_position_matches_the_figures_axis() -> None:
    """The panel's heights are those of the axis ``plot_event_metric_row`` sets up."""
    import matplotlib.pyplot as plt

    from statespacecheck_paper.interactive.panels import symlog_position
    from statespacecheck_paper.style import SYMLOG_LINSCALE, SYMLOG_LINTHRESH

    fig, ax = plt.subplots()
    try:
        ax.set_yscale("symlog", linthresh=SYMLOG_LINTHRESH, linscale=SYMLOG_LINSCALE)
        expected = ax.yaxis.get_transform().transform(_SYMLOG_PROBES)
    finally:
        plt.close(fig)
    np.testing.assert_allclose(symlog_position(_SYMLOG_PROBES), expected, rtol=0, atol=1e-15)
    # A small positive overlap sits strictly above an exact zero.
    assert symlog_position([0.0])[0] == 0.0
    assert symlog_position([1e-4])[0] > 0.0


def _metric_panel(name: str, threshold: float | None) -> Any:
    from PySide6 import QtWidgets

    from statespacecheck_paper.interactive.panels import MetricPanel
    from statespacecheck_paper.style import METRIC_SPEC_BY_NAME

    QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    return MetricPanel(spec=METRIC_SPEC_BY_NAME[name], threshold=threshold)


def test_hpd_panel_draws_every_height_on_the_symlog_axis() -> None:
    from statespacecheck_paper.figure03_plotting import FIGURE03_SYMLOG_YTICKS
    from statespacecheck_paper.interactive.panels import symlog_position

    panel = _metric_panel("hpd_overlap", threshold=0.05)
    try:
        # Ticks sit at the transformed positions, labeled with the raw values.
        (major, minor) = panel.getAxis("left")._tickLevels  # noqa: SLF001
        assert minor == []
        assert [label for _, label in major] == ["0", "0.01", "0.1", "1"]
        np.testing.assert_allclose(
            [position for position, _ in major], symlog_position(FIGURE03_SYMLOG_YTICKS)
        )
        assert panel._threshold_line.value() == pytest.approx(  # noqa: SLF001
            symlog_position([0.05])[0]
        )

        values = _SYMLOG_PROBES.astype(np.float32)
        panel.update_window(
            np.arange(values.size, dtype=np.float64),
            values,
            0.0,
            np.arange(values.size, dtype=np.int64),
        )
        _, y = panel._scatter.getData()  # noqa: SLF001
        np.testing.assert_allclose(y, symlog_position(values), rtol=1e-6)

        panel.update_pinned_event(relative_time=1.0, metric_value=1e-4)
        _, pin_y = panel._pin_dot.getData()  # noqa: SLF001
        assert pin_y[0] == pytest.approx(symlog_position([1e-4])[0], rel=1e-6)
        assert pin_y[0] > 0.0
    finally:
        panel.close()


def test_other_metric_panels_keep_their_linear_axes() -> None:
    panel = _metric_panel("kl_divergence", threshold=None)
    try:
        assert panel.getAxis("left")._tickLevels is None  # noqa: SLF001
        values = np.array([0.0, 0.5, 3.0], dtype=np.float32)
        panel.update_window(
            np.arange(3, dtype=np.float64), values, 0.0, np.arange(3, dtype=np.int64)
        )
        _, y = panel._scatter.getData()  # noqa: SLF001
        np.testing.assert_array_equal(y, values)
    finally:
        panel.close()


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
    app, viewer, ds = make_viewer(tmp_path / "cache")
    try:
        target = viewer._next_request_id  # noqa: SLF001
        viewer.force_reload_now()
        assert wait_for_request(app, viewer, target)

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
    app, viewer, ds = make_viewer(tmp_path / "cache")
    try:
        target = viewer._next_request_id  # noqa: SLF001
        viewer.force_reload_now()
        assert wait_for_request(app, viewer, target)

        sl = viewer.slice_panel._buffer_slice  # noqa: SLF001
        assert sl is not None
        row = _first_visible_event_row(ds, sl)
        assert row is not None

        viewer._handle_event_click(row)  # noqa: SLF001
        # Click triggers a recenter -> new load. Wait for the new
        # window to commit so the pin lands inside the buffered slice.
        target2 = viewer._next_request_id  # noqa: SLF001
        assert wait_for_request(app, viewer, target2)

        assert viewer.predictive_panel._pin_line.isVisible()  # noqa: SLF001
        assert viewer.likelihood_panel._pin_line.isVisible()  # noqa: SLF001
        assert viewer.raster_panel._pin_line.isVisible()  # noqa: SLF001
        for panel in viewer.metric_panels.values():
            assert panel._pin_line.isVisible()  # noqa: SLF001
            assert panel._pin_dot.isVisible()  # noqa: SLF001
        # The pinned spike's cell is labeled 1-based.
        annotation = viewer.slice_panel._annotation.text()  # noqa: SLF001
        assert f"cell={int(ds.event_cell_ids[row]) + 1}\n" in annotation
    finally:
        viewer.close()
        ds.close()


def test_manual_scroll_unpins_event(tmp_path: Path) -> None:
    _build_cache(tmp_path / "cache")
    app, viewer, ds = make_viewer(tmp_path / "cache")
    try:
        target = viewer._next_request_id  # noqa: SLF001
        viewer.force_reload_now()
        assert wait_for_request(app, viewer, target)

        sl = viewer.slice_panel._buffer_slice  # noqa: SLF001
        assert sl is not None
        row = _first_visible_event_row(ds, sl)
        assert row is not None
        viewer._handle_event_click(row)  # noqa: SLF001
        assert viewer._pinned_event_row == row  # noqa: SLF001

        # Slider movement signals the unpin path.
        viewer._on_slider_changed(viewer._slider.value() + 1)  # noqa: SLF001
        assert viewer._pinned_event_row is None  # noqa: SLF001
        assert not viewer.predictive_panel._pin_line.isVisible()  # noqa: SLF001
        for panel in viewer.metric_panels.values():
            assert not panel._pin_line.isVisible()  # noqa: SLF001
    finally:
        viewer.close()
        ds.close()


def test_pin_invisible_when_event_outside_loaded_window(tmp_path: Path) -> None:
    """If the pinned event is outside the current window, markers hide."""
    _build_cache(tmp_path / "cache")
    app, viewer, ds = make_viewer(tmp_path / "cache")
    try:
        target = viewer._next_request_id  # noqa: SLF001
        viewer.force_reload_now()
        assert wait_for_request(app, viewer, target)

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
            assert not viewer.predictive_panel._pin_line.isVisible()  # noqa: SLF001
    finally:
        viewer.close()
        ds.close()
