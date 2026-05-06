"""Tests for ``SlicePanel`` and its wiring into ``Figure4Viewer``."""

from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
import pytest

from statespacecheck_paper.interactive import cache as cache_mod

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


# ---------------------------------------------------------------------------
# Synthetic cache builder (single-state and multi-state variants).
# ---------------------------------------------------------------------------


def _build_cache(
    cache_dir: Path,
    *,
    n_states: int,
    n_time: int = 500,
    n_position: int = 16,
    n_cells: int = 4,
    seed: int = 0,
) -> None:
    """Build a self-consistent synthetic cache for the given state count."""
    import pandas as pd
    import xarray as xr

    rng = np.random.default_rng(seed)
    n_state_bins = n_states * n_position

    state_names = [f"state_{i}" for i in range(n_states)]
    state_coord = np.array([state_names[i] for i in range(n_states) for _ in range(n_position)])
    position_grid = np.linspace(0.0, 100.0, n_position)
    position_coord = np.tile(position_grid, n_states)
    posterior = rng.dirichlet(np.ones(n_state_bins), size=n_time).astype(np.float32)
    log_likelihood = np.log(posterior + 1e-12).astype(np.float32)
    if n_states == 1:
        state_probs = np.ones((n_time,), dtype=np.float32)
        state_probs_var = (("time",), state_probs)
    else:
        state_probs = rng.dirichlet(np.ones(n_states), size=n_time).astype(np.float32)
        state_probs_var = (("time", "states"), state_probs)
    time_arr = 1000.0 + np.arange(n_time, dtype=np.float64) * 0.002

    coords: dict[str, object] = {
        "time": ("time", time_arr),
        "state_bins": ("state_bins", np.arange(n_state_bins, dtype=np.int64)),
        "state": ("state_bins", state_coord),
        "position": ("state_bins", position_coord),
    }
    if n_states > 1:
        coords["states"] = ("states", np.array(state_names))

    ds = xr.Dataset(
        data_vars={
            "predictive_posterior": (("time", "state_bins"), posterior),
            "log_likelihood": (("time", "state_bins"), log_likelihood),
            "acausal_state_probabilities": state_probs_var,
        },
        coords=coords,
    )

    paths = cache_mod.cache_paths(cache_dir, "continuous")
    cache_mod._write_zarr_store(ds=ds, out_dir=paths["zarr"], time_chunk=64, pyramid_strides=(8,))

    spike_times = [
        np.sort(rng.uniform(time_arr[0], time_arr[-1], size=15)).astype(np.float64)
        for _ in range(n_cells)
    ]
    rows = []
    for cell_id, ts in enumerate(spike_times):
        for t_val in ts:
            rows.append((float(t_val), cell_id, 0.5, 1.0, 0.5))
    rows.sort(key=lambda r: r[0])
    events = pd.DataFrame(
        rows,
        columns=[
            "time",
            "cell_id",
            "event_hpd_overlap",
            "event_kl_divergence",
            "event_spike_prob",
        ],
    ).astype(
        {
            "time": np.float64,
            "cell_id": np.int32,
            "event_hpd_overlap": np.float32,
            "event_kl_divergence": np.float32,
            "event_spike_prob": np.float32,
        }
    )
    events.to_parquet(paths["events"], engine="pyarrow", compression="zstd")

    place_fields = rng.uniform(0.0, 5.0, size=(n_cells, n_state_bins)).astype(np.float32)
    peak_idx = np.argmax(place_fields[:, :n_position], axis=1)
    np.savez(
        paths["place_fields"],
        place_fields=place_fields,
        interior_mask=np.ones(n_state_bins, dtype=bool),
        position_bins=position_grid.astype(np.float64),
        place_field_peaks=position_grid[peak_idx].astype(np.float64),
    )
    np.savez(
        cache_mod.meta_path(cache_dir),
        time=time_arr,
        linear_position=rng.uniform(0.0, 100.0, size=n_time).astype(np.float64),
        n_cells=np.int64(n_cells),
    )
    container = np.empty(n_cells, dtype=object)
    for i, st in enumerate(spike_times):
        container[i] = st
    np.save(cache_mod.spike_times_path(cache_dir), container, allow_pickle=True)


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

        readout = viewer.slice_panel._live_readout  # noqa: SLF001
        text = readout.toPlainText()
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
        text2 = viewer.slice_panel._live_readout.toPlainText()  # noqa: SLF001
        assert text2 != text
    finally:
        viewer.close()
        ds.close()


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
