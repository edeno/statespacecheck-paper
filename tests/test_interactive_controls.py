"""Tests for the Phase 6 controls: window-width slider, model swap,
play/pause auto-scroll, and keyboard shortcuts.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

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


def _build_cache(
    cache_dir: Path,
    *,
    model: str = "continuous",
    n_states: int = 1,
    seed: int = 0,
) -> None:
    """Controls tests need both Continuous and ContFrag caches for swap tests."""
    _build_cache_impl(cache_dir, model=model, n_states=n_states, seed=seed)


@pytest.fixture(scope="module", autouse=True)
def _qt_offscreen() -> None:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def _make_viewer(cache_dir: Path, *, model: str = "continuous"):
    from PySide6 import QtWidgets

    from statespacecheck_paper.interactive.data_source import Figure4DataSource
    from statespacecheck_paper.interactive.viewer import Figure4Viewer

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    ds = Figure4DataSource(cache_dir, model=model)
    viewer = Figure4Viewer(ds, cache_dir=cache_dir)
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
# Window-width slider
# ---------------------------------------------------------------------------


def test_window_slider_round_trip_seconds(tmp_path: Path) -> None:
    _build_cache(tmp_path / "cache")
    app, viewer, ds = _make_viewer(tmp_path / "cache")
    try:
        # Slider->seconds->slider should be identity within rounding.
        for sv in (0, 100, 500, 900, 1000):
            w = viewer._window_seconds_for(sv)  # noqa: SLF001
            sv2 = viewer._window_slider_value_for(w)  # noqa: SLF001
            assert abs(sv - sv2) <= 1
    finally:
        viewer.close()
        ds.close()


def test_window_slider_changes_window_seconds(tmp_path: Path) -> None:
    _build_cache(tmp_path / "cache")
    app, viewer, ds = _make_viewer(tmp_path / "cache")
    try:
        slider_value = 800
        viewer._window_slider.setValue(slider_value)  # noqa: SLF001
        expected = viewer._window_seconds_for(slider_value)  # noqa: SLF001
        assert abs(viewer._window_seconds - expected) < 1e-9  # noqa: SLF001
    finally:
        viewer.close()
        ds.close()


def test_keyboard_brackets_scale_window(tmp_path: Path) -> None:
    _build_cache(tmp_path / "cache")
    app, viewer, ds = _make_viewer(tmp_path / "cache")
    try:
        original = viewer._window_seconds  # noqa: SLF001
        viewer._scale_window(2.0)  # noqa: SLF001
        assert abs(viewer._window_seconds - 2.0 * original) < 1e-9  # noqa: SLF001
        viewer._scale_window(0.5)  # noqa: SLF001
        assert abs(viewer._window_seconds - original) < 1e-9  # noqa: SLF001
    finally:
        viewer.close()
        ds.close()


# ---------------------------------------------------------------------------
# Step / reset shortcuts
# ---------------------------------------------------------------------------


def test_step_center_by_indices_advances_one_bin(tmp_path: Path) -> None:
    _build_cache(tmp_path / "cache")
    app, viewer, ds = _make_viewer(tmp_path / "cache")
    try:
        viewer.set_center_time(float(ds.time[100]))
        viewer._step_center_by_indices(1)  # noqa: SLF001
        assert ds.index_at_time(viewer._t_center) == 101  # noqa: SLF001
        viewer._step_center_by_indices(-2)  # noqa: SLF001
        assert ds.index_at_time(viewer._t_center) == 99  # noqa: SLF001
    finally:
        viewer.close()
        ds.close()


def test_reset_view_centers_and_sets_window(tmp_path: Path) -> None:
    _build_cache(tmp_path / "cache")
    app, viewer, ds = _make_viewer(tmp_path / "cache")
    try:
        viewer.set_center_time(float(ds.time[-1]))
        viewer._scale_window(2.0)  # noqa: SLF001
        viewer._reset_view()  # noqa: SLF001
        # Reset window seconds matches RESET_WINDOW_SECONDS (clamped if
        # needed to MAX_WINDOW_SECONDS).
        from statespacecheck_paper.interactive.viewer import (
            MAX_WINDOW_SECONDS,
            RESET_WINDOW_SECONDS,
        )

        expected_w = min(RESET_WINDOW_SECONDS, MAX_WINDOW_SECONDS)
        assert abs(viewer._window_seconds - expected_w) < 1e-9  # noqa: SLF001
        # Reset center should be inside the session.
        assert viewer._t_min <= viewer._t_center <= viewer._t_max  # noqa: SLF001
    finally:
        viewer.close()
        ds.close()


# ---------------------------------------------------------------------------
# Play / pause auto-scroll
# ---------------------------------------------------------------------------


def test_play_button_toggles_autoscroll_state(tmp_path: Path) -> None:
    _build_cache(tmp_path / "cache")
    app, viewer, ds = _make_viewer(tmp_path / "cache")
    try:
        assert viewer._autoscroll_timer is None  # noqa: SLF001
        viewer._play_button.setChecked(True)  # noqa: SLF001
        assert viewer._autoscroll_timer is not None  # noqa: SLF001
        viewer._play_button.setChecked(False)  # noqa: SLF001
        assert viewer._autoscroll_timer is None  # noqa: SLF001
    finally:
        viewer.close()
        ds.close()


def test_autoscroll_step_advances_center_time(tmp_path: Path) -> None:
    _build_cache(tmp_path / "cache")
    app, viewer, ds = _make_viewer(tmp_path / "cache")
    try:
        viewer.set_center_time(float(ds.time[100]))
        before = viewer._t_center  # noqa: SLF001
        viewer._autoscroll_step()  # noqa: SLF001
        # One tick advances by ``current rate / tick_hz`` seconds.
        # Read the rate off the viewer rather than pinning to a constant
        # so the test doesn't break when the startup default changes.
        from statespacecheck_paper.interactive.viewer import AUTOSCROLL_TICK_HZ

        expected_dt = viewer._autoscroll_rate / AUTOSCROLL_TICK_HZ  # noqa: SLF001
        assert viewer._t_center > before  # noqa: SLF001
        assert abs((viewer._t_center - before) - expected_dt) < 1e-6  # noqa: SLF001
    finally:
        viewer.close()
        ds.close()


def test_speed_combo_changes_autoscroll_rate(tmp_path: Path) -> None:
    _build_cache(tmp_path / "cache")
    app, viewer, ds = _make_viewer(tmp_path / "cache")
    try:
        from statespacecheck_paper.interactive.viewer import AUTOSCROLL_SPEED_OPTIONS

        # Set to the highest option; ``_autoscroll_rate`` should match.
        last_idx = len(AUTOSCROLL_SPEED_OPTIONS) - 1
        viewer._speed_combo.setCurrentIndex(last_idx)  # noqa: SLF001
        assert abs(viewer._autoscroll_rate - AUTOSCROLL_SPEED_OPTIONS[last_idx]) < 1e-9  # noqa: SLF001

        # Set to the lowest.
        viewer._speed_combo.setCurrentIndex(0)  # noqa: SLF001
        assert abs(viewer._autoscroll_rate - AUTOSCROLL_SPEED_OPTIONS[0]) < 1e-9  # noqa: SLF001
    finally:
        viewer.close()
        ds.close()


def test_speed_step_keyboard_shortcut(tmp_path: Path) -> None:
    _build_cache(tmp_path / "cache")
    app, viewer, ds = _make_viewer(tmp_path / "cache")
    try:
        from statespacecheck_paper.interactive.viewer import AUTOSCROLL_SPEED_OPTIONS

        viewer._speed_combo.setCurrentIndex(2)  # noqa: SLF001
        viewer._step_speed(+1)  # noqa: SLF001
        assert viewer._speed_combo.currentIndex() == 3  # noqa: SLF001
        assert abs(viewer._autoscroll_rate - AUTOSCROLL_SPEED_OPTIONS[3]) < 1e-9  # noqa: SLF001

        viewer._step_speed(-2)  # noqa: SLF001
        assert viewer._speed_combo.currentIndex() == 1  # noqa: SLF001
        # Stepping past either end clamps without erroring.
        viewer._step_speed(-100)  # noqa: SLF001
        assert viewer._speed_combo.currentIndex() == 0  # noqa: SLF001
        viewer._step_speed(+100)  # noqa: SLF001
        assert viewer._speed_combo.currentIndex() == len(AUTOSCROLL_SPEED_OPTIONS) - 1  # noqa: SLF001
    finally:
        viewer.close()
        ds.close()


def test_autoscroll_step_uses_current_speed(tmp_path: Path) -> None:
    _build_cache(tmp_path / "cache")
    app, viewer, ds = _make_viewer(tmp_path / "cache")
    try:
        from statespacecheck_paper.interactive.viewer import (
            AUTOSCROLL_SPEED_OPTIONS,
            AUTOSCROLL_TICK_HZ,
        )

        # Pick the 4× option from the preset list.
        idx_4x = AUTOSCROLL_SPEED_OPTIONS.index(4.0)
        viewer.set_center_time(float(ds.time[100]))
        viewer._speed_combo.setCurrentIndex(idx_4x)  # noqa: SLF001
        before = viewer._t_center  # noqa: SLF001
        viewer._autoscroll_step()  # noqa: SLF001
        expected_dt = 4.0 / AUTOSCROLL_TICK_HZ
        assert abs((viewer._t_center - before) - expected_dt) < 1e-6  # noqa: SLF001
    finally:
        viewer.close()
        ds.close()


def test_autoscroll_pauses_at_session_end(tmp_path: Path) -> None:
    _build_cache(tmp_path / "cache")
    app, viewer, ds = _make_viewer(tmp_path / "cache")
    try:
        viewer.set_center_time(float(ds.time[-1]))
        viewer._play_button.setChecked(True)  # noqa: SLF001
        # One tick at the end clamps and toggles the play button off.
        viewer._autoscroll_step()  # noqa: SLF001
        assert not viewer._play_button.isChecked()  # noqa: SLF001
    finally:
        viewer.close()
        ds.close()


# ---------------------------------------------------------------------------
# Model swap
# ---------------------------------------------------------------------------


def test_model_swap_rebuilds_panels_and_loads(tmp_path: Path) -> None:
    cache_dir = tmp_path / "cache"
    _build_cache(cache_dir, model="continuous", n_states=1)
    _build_cache(cache_dir, model="contfrag", n_states=2, seed=1)

    app, viewer, ds = _make_viewer(cache_dir, model="continuous")
    try:
        assert viewer._ds.model == "continuous"  # noqa: SLF001
        assert viewer.slice_panel._n_states == 1  # noqa: SLF001

        viewer._switch_model("contfrag")  # noqa: SLF001
        assert viewer._ds.model == "contfrag"  # noqa: SLF001
        assert viewer.slice_panel._n_states == 2  # noqa: SLF001
        # Heatmap panels rebuilt with the new state count too.
        assert viewer.posterior_panel._n_states == 2  # noqa: SLF001

        # The new central widget should commit a fresh load.
        target = viewer._next_request_id  # noqa: SLF001
        assert _wait_for_request(app, viewer, target)
    finally:
        viewer.close()
        ds.close()


def test_model_swap_revert_when_cache_missing(tmp_path: Path) -> None:
    cache_dir = tmp_path / "cache"
    _build_cache(cache_dir, model="continuous", n_states=1)
    # Note: do NOT build the contfrag cache.

    app, viewer, ds = _make_viewer(cache_dir, model="continuous")
    try:
        viewer._switch_model("contfrag")  # noqa: SLF001
        # The data source should remain on continuous.
        assert viewer._ds.model == "continuous"  # noqa: SLF001
    finally:
        viewer.close()
        ds.close()
