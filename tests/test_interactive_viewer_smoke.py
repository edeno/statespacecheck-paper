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


def _build_synthetic_cache(cache_dir: Path) -> None:
    """Reuse the same synthetic builder shape used by the data-source tests."""
    import pandas as pd
    import xarray as xr

    rng = np.random.default_rng(0)
    n_time, n_position, n_states, n_cells = 500, 16, 1, 4
    n_state_bins = n_states * n_position

    state_names = [f"state_{i}" for i in range(n_states)]
    state_coord = np.array([state_names[i] for i in range(n_states) for _ in range(n_position)])
    position_grid = np.linspace(0.0, 100.0, n_position)
    position_coord = np.tile(position_grid, n_states)
    posterior = rng.dirichlet(np.ones(n_state_bins), size=n_time).astype(np.float32)
    log_likelihood = np.log(posterior + 1e-12).astype(np.float32)
    state_probs = np.ones((n_time,), dtype=np.float32)
    time_arr = 1000.0 + np.arange(n_time, dtype=np.float64) * 0.002

    ds = xr.Dataset(
        data_vars={
            "predictive_posterior": (("time", "state_bins"), posterior),
            "log_likelihood": (("time", "state_bins"), log_likelihood),
            "acausal_state_probabilities": (("time",), state_probs),
        },
        coords={
            "time": ("time", time_arr),
            "state_bins": ("state_bins", np.arange(n_state_bins, dtype=np.int64)),
            "state": ("state_bins", state_coord),
            "position": ("state_bins", position_coord),
        },
    )

    paths = cache_mod.cache_paths(cache_dir, "continuous")
    cache_mod._write_zarr_store(ds=ds, out_dir=paths["zarr"], time_chunk=64, pyramid_strides=(8,))

    spike_times = [
        np.sort(rng.uniform(time_arr[0], time_arr[-1], size=20)).astype(np.float64)
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

    place_fields = rng.uniform(0.0, 5.0, size=(n_cells, n_position)).astype(np.float32)
    peak_idx = np.argmax(place_fields, axis=1)
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
