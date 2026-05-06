"""Tests for ``MetricPanel`` + click-to-recenter + pinned-event markers.

Reuses the synthetic-cache builder from ``test_interactive_slice_panel``
(via the same internal layout) but kept self-contained so each test
file works in isolation.
"""

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


def _build_cache(
    cache_dir: Path,
    *,
    n_states: int = 1,
    n_time: int = 500,
    n_position: int = 16,
    n_cells: int = 4,
    seed: int = 0,
) -> None:
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
    state_probs_var: tuple = (
        (("time",), np.ones((n_time,), dtype=np.float32))
        if n_states == 1
        else (
            ("time", "states"),
            rng.dirichlet(np.ones(n_states), size=n_time).astype(np.float32),
        )
    )
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
            rows.append(
                (
                    float(t_val),
                    int(cell_id),
                    float(rng.uniform(0.0, 1.0)),
                    float(rng.uniform(0.0, 5.0)),
                    float(rng.uniform(0.001, 1.0)),
                )
            )
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
