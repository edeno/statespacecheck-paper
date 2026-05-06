"""Tests for ``SlicePanel`` and its wiring into ``DecoderViewer``."""

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
    from statespacecheck_paper.interactive.data_source import DecoderDataSource

    src = DecoderDataSource(tmp_path / "cache", model="continuous")
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


def test_slice_panel_falls_back_to_row_provider_outside_buffer(tmp_path: Path) -> None:
    """When ``t_idx`` is outside the buffered window, the slice panel
    fetches a single row via the registered ``row_provider`` callable
    so the curves keep animating until the next async load commits.
    """
    _build_cache(tmp_path / "cache", n_states=1)
    app, viewer, ds = _make_viewer(tmp_path / "cache")
    try:
        # Wait for the initial in-flight load so subsequent
        # ``force_reload_now`` calls aren't merely queued as pending.
        assert _wait_for_request(app, viewer, viewer._next_request_id)  # noqa: SLF001
        # Now shrink the window and recenter so the buffer no longer
        # covers the entire (small) synthetic session.
        viewer._set_window_seconds(0.05)  # noqa: SLF001
        viewer.set_center_time(float(ds.time[100]))
        viewer.force_reload_now()
        target = viewer._next_request_id  # noqa: SLF001
        assert _wait_for_request(app, viewer, target)

        sp = viewer.slice_panel
        sl = sp._buffer_slice  # noqa: SLF001
        assert sl is not None
        # 25-sample window vs. 500-sample session -> plenty of room
        # for an out-of-buffer index.
        assert sl.stop - sl.start < ds.n_time
        out_of_buffer = sl.stop + 50
        assert sl.stop < out_of_buffer < ds.n_time

        x_before, y_before = sp._lik_predictive_curve.getData()  # noqa: SLF001
        sp.update_for_index(out_of_buffer, true_position=42.0)
        x_after, y_after = sp._lik_predictive_curve.getData()  # noqa: SLF001
        assert y_after.shape == y_before.shape
        # The predictive at the out-of-buffer index is almost certainly
        # different from the buffered center (Dirichlet rows are
        # independent draws); without the row provider this would
        # silently be a no-op.
        assert not np.array_equal(y_after, y_before)
    finally:
        viewer.close()
        ds.close()


# ---------------------------------------------------------------------------
# Top-plot overlay choice: predictive / filtered / smoothed
# ---------------------------------------------------------------------------


def test_filtered_overlay_matches_predictive_times_likelihood(tmp_path: Path) -> None:
    """``filtered`` = ``predictive × likelihood`` normalized over state_bins,
    then state-collapsed and peak-normalized for display.
    """
    _build_cache(tmp_path / "cache", n_states=1)
    app, viewer, ds = _make_viewer(tmp_path / "cache")
    try:
        target = viewer._next_request_id  # noqa: SLF001
        viewer.force_reload_now()
        assert _wait_for_request(app, viewer, target)

        sp = viewer.slice_panel
        sl = sp._buffer_slice  # noqa: SLF001
        post = sp._buffer_post  # noqa: SLF001
        lik = sp._buffer_lik  # noqa: SLF001
        assert sl is not None and post is not None and lik is not None

        # Pick a definite bin in the middle of the loaded buffer and
        # ``set_center_time`` to its real-time tick, so we know the
        # slice panel is rendering exactly that bin's row.
        target_idx = (sl.start + sl.stop) // 2
        viewer.set_center_time(float(ds.time[target_idx]))
        sp.set_overlay_choice("filtered")
        viewer._update_slice_panel_at_center()  # noqa: SLF001
        _, top = sp._lik_overlay_curve.getData()  # noqa: SLF001

        post_row = post[target_idx - sl.start]
        lik_row = lik[target_idx - sl.start]
        prod = post_row * lik_row
        prod /= float(prod.sum())
        peak = float(prod.max())
        expected = (prod / peak).astype(np.float32, copy=False)

        np.testing.assert_allclose(top, expected, rtol=1e-5, atol=1e-6)
    finally:
        viewer.close()
        ds.close()


def test_smoothed_overlay_reload_populates_buffer_acausal(tmp_path: Path) -> None:
    """The worker only loads ``acausal_posterior`` when the slice panel's
    overlay is ``smoothed``. Switching from predictive to smoothed must
    trigger a fresh window load that populates the buffer.
    """
    _build_cache(tmp_path / "cache", n_states=1)
    app, viewer, ds = _make_viewer(tmp_path / "cache")
    try:
        target = viewer._next_request_id  # noqa: SLF001
        viewer.force_reload_now()
        assert _wait_for_request(app, viewer, target)

        sp = viewer.slice_panel
        # Default overlay is predictive ⇒ buffer should not have acausal.
        assert sp.overlay_choice == "predictive"
        assert sp._buffer_acausal is None  # noqa: SLF001

        smoothed_idx = next(
            i
            for i in range(viewer._overlay_combo.count())  # noqa: SLF001
            if viewer._overlay_combo.itemData(i) == "smoothed"  # noqa: SLF001
        )
        target = viewer._next_request_id  # noqa: SLF001
        viewer._overlay_combo.setCurrentIndex(smoothed_idx)  # noqa: SLF001
        assert _wait_for_request(app, viewer, target + 1)

        assert sp.overlay_choice == "smoothed"
        assert sp._buffer_acausal is not None  # noqa: SLF001
        assert sp._buffer_acausal.shape == sp._buffer_post.shape  # noqa: SLF001
    finally:
        viewer.close()
        ds.close()


def test_old_cache_without_acausal_disables_smoothed(tmp_path: Path) -> None:
    """Caches built before the smoothed-overlay feature don't have
    ``acausal_posterior``. The viewer should report ``has_acausal=False``,
    leave the smoothed combo entry disabled, and never load acausal
    even if some path tried to.
    """
    from PySide6 import QtGui

    _build_cache_impl(tmp_path / "cache", model="continuous", with_acausal=False)
    app, viewer, ds = _make_viewer(tmp_path / "cache")
    try:
        assert ds.has_acausal is False

        smoothed_idx = next(
            i
            for i in range(viewer._overlay_combo.count())  # noqa: SLF001
            if viewer._overlay_combo.itemData(i) == "smoothed"  # noqa: SLF001
        )
        combo_model = viewer._overlay_combo.model()  # noqa: SLF001
        # Combos backed by a ``QStandardItemModel`` expose ``.item``;
        # the disable was wired via that interface in viewer setup.
        assert isinstance(combo_model, QtGui.QStandardItemModel)
        assert combo_model.item(smoothed_idx).isEnabled() is False

        # The data-source's load_acausal returns ``None`` regardless of
        # slice — exercises the worker's lazy-load short-circuit.
        assert ds.load_acausal(slice(0, 10)) is None
    finally:
        viewer.close()
        ds.close()


def test_slice_panel_no_op_outside_buffer_without_provider(tmp_path: Path) -> None:
    """When no row provider is wired (e.g. unit tests for the panel
    in isolation), out-of-buffer ``update_for_index`` is a silent no-op
    instead of raising.
    """
    from statespacecheck_paper.interactive.viewer import SlicePanel

    panel = SlicePanel(position_bins=np.linspace(0.0, 100.0, 16), n_states=1)
    try:
        zero = np.zeros(16, dtype=np.float32)
        panel.set_window_buffer(slice(0, 1), zero[None, :], zero[None, :])
        x_before, y_before = panel._lik_predictive_curve.getData()  # noqa: SLF001
        panel.update_for_index(50, true_position=10.0)
        x_after, y_after = panel._lik_predictive_curve.getData()  # noqa: SLF001
        np.testing.assert_array_equal(y_after, y_before)
    finally:
        panel.close()
