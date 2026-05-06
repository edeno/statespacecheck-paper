"""Tests for the figure-3 simulation cache path.

Builds a tiny simulation cache via ``build_simulated_cache``, opens it
through ``DecoderDataSource.for_simulation``, and exercises the loader
contract + the viewer's adaptation to ``dataset_kind == "simulation"``.

The simulation params are scaled down (``T_*`` shrunk by a factor of
~30) so the full forward filter runs in well under a second on CI;
the assertions don't depend on phase-specific behaviour, just on the
end-to-end shape contract.
"""

from __future__ import annotations

import os
import time as time_mod
from pathlib import Path

import numpy as np
import pytest

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


@pytest.fixture(scope="module", autouse=True)
def _qt_offscreen() -> None:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def _tiny_params():
    """``DecodeParams`` shrunk so the simulation runs fast in tests."""
    from statespacecheck_paper.analysis import DecodeParams

    return DecodeParams(
        T_remap_start=200,
        T_remap_end=300,
        T_recovery1_end=400,
        T_flat_end=500,
        T_recovery2_end=600,
        T_fast_end=700,
        T_recovery3_end=800,
        T_slow_end=1000,
    )


def _build_simulated(cache_dir: Path) -> dict[str, object]:
    from statespacecheck_paper.interactive.cache import build_simulated_cache

    return build_simulated_cache(
        cache_dir,
        params=_tiny_params(),
        seed=0,
        time_chunk=128,
        force=True,
    )


# ---------------------------------------------------------------------------
# Loader contract
# ---------------------------------------------------------------------------


def test_simulated_cache_loader_metadata(tmp_path: Path) -> None:
    """``DecoderDataSource.for_simulation`` reports the simulation kind."""
    from statespacecheck_paper.interactive.data_source import DecoderDataSource

    _build_simulated(tmp_path)
    ds = DecoderDataSource.for_simulation(tmp_path)
    try:
        assert ds.dataset_kind == "simulation"
        assert ds.model is None
        assert ds.display_name == "Figure 3 simulation"
        # Simulation only forward-filters — no smoothed posterior.
        assert ds.has_acausal is False
        # Single state, all bins interior.
        assert ds.n_states == 1
        assert ds.n_interior == ds.position_bins.shape[0]
        # ``time`` grid is ``np.arange(n_time) * 0.002``.
        assert ds.n_time > 0
        np.testing.assert_allclose(ds.time[1] - ds.time[0], 0.002, atol=1e-12)
    finally:
        ds.close()


def test_simulated_cache_log_likelihood_round_trips(tmp_path: Path) -> None:
    """The cache stores log-likelihood; the worker's per-row max-shift
    + ``exp`` recovers the original simulation likelihood
    (peak-normalised within float32 tolerance).

    Catches two regressions at the cache-build boundary:

    * Writing *linear* likelihood instead of log — the worker would
      then ``exp`` an already-normalised distribution and the
      likelihood panel would visually flatten.
    * Clamping the log floor too high (e.g. ``log(max(x, 1e-12))``):
      rows whose simulated peak is smaller than the clamp would
      round-trip to a roughly uniform response, hiding actual
      decoded structure.
    """
    import zarr

    from statespacecheck_paper.figure03_demo import run_figure03_simulation
    from statespacecheck_paper.interactive.cache import simulated_cache_paths

    sim = run_figure03_simulation(_tiny_params(), seed=0)
    _build_simulated(tmp_path)

    paths = simulated_cache_paths(tmp_path)
    group = zarr.open_group(str(paths["zarr"]), mode="r")
    log_lik_cached = np.asarray(group["log_likelihood"][:])

    # Mirror the worker's per-row max-shift + exp.
    row_max = log_lik_cached.max(axis=1, keepdims=True)
    row_max = np.where(np.isfinite(row_max), row_max, 0.0)
    lik_recovered = np.exp(log_lik_cached - row_max).astype(np.float32)

    # Reference: the simulation's normalised linear likelihood,
    # peak-normalised the same way the worker output is.
    sim_lik = np.asarray(sim["metrics"]["likelihood"], dtype=np.float64)
    sim_peak = sim_lik.max(axis=1, keepdims=True)
    sim_peak = np.where(sim_peak > 0, sim_peak, 1.0)
    sim_lik_peak_normed = (sim_lik / sim_peak).astype(np.float32)

    # Should agree everywhere; tolerance accounts for float32 round-trip.
    np.testing.assert_allclose(lik_recovered, sim_lik_peak_normed, atol=1e-4, rtol=1e-3)


# ---------------------------------------------------------------------------
# Viewer wiring
# ---------------------------------------------------------------------------


def _wait_for_request(app, viewer, request_id: int, timeout_s: float = 5.0) -> bool:
    deadline = time_mod.perf_counter() + timeout_s
    while time_mod.perf_counter() < deadline:
        app.processEvents()
        if viewer._latest_committed_request_id >= request_id:  # noqa: SLF001
            return True
        time_mod.sleep(0.005)
    return False


def test_simulated_viewer_hides_model_combo(tmp_path: Path) -> None:
    """The viewer's model-swap combo is *not present* for simulation
    caches (not just disabled — there is no model concept here).
    """
    from PySide6 import QtWidgets

    from statespacecheck_paper.interactive.data_source import DecoderDataSource
    from statespacecheck_paper.interactive.viewer import DecoderViewer

    _build_simulated(tmp_path)
    _ = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    ds = DecoderDataSource.for_simulation(tmp_path)
    viewer = DecoderViewer(ds)
    try:
        assert viewer._model_combo is None  # noqa: SLF001
        assert viewer._model_label is None  # noqa: SLF001
        # Window title uses display_name, not "None" from a missing model.
        assert "Figure 3 simulation" in viewer.windowTitle()
        # Smoothed overlay disabled (data source has no acausal).
        smoothed_idx = next(
            i
            for i in range(viewer._overlay_combo.count())  # noqa: SLF001
            if viewer._overlay_combo.itemData(i) == "smoothed"  # noqa: SLF001
        )
        from PySide6 import QtGui

        combo_model = viewer._overlay_combo.model()  # noqa: SLF001
        assert isinstance(combo_model, QtGui.QStandardItemModel)
        assert combo_model.item(smoothed_idx).isEnabled() is False
    finally:
        viewer.close()
        ds.close()


def test_simulated_viewer_loads_window(tmp_path: Path) -> None:
    """Opening a simulated cache and forcing a window load populates
    ``_buffer_post`` and ``_buffer_lik`` with the expected shapes.
    Per-cell rows should appear at a bin where the simulation has
    spikes.
    """
    from PySide6 import QtWidgets

    from statespacecheck_paper.interactive.data_source import DecoderDataSource
    from statespacecheck_paper.interactive.viewer import DecoderViewer

    _build_simulated(tmp_path)
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    ds = DecoderDataSource.for_simulation(tmp_path)
    viewer = DecoderViewer(ds)
    try:
        target = viewer._next_request_id  # noqa: SLF001
        viewer.force_reload_now()
        assert _wait_for_request(app, viewer, target)

        sp = viewer.slice_panel
        assert sp._buffer_post is not None  # noqa: SLF001
        assert sp._buffer_lik is not None  # noqa: SLF001
        assert sp._buffer_acausal is None  # noqa: SLF001 — no acausal in simulation

        # Pick a time at a real event so per-cell rows are guaranteed
        # populated.
        assert len(ds.events) > 0
        viewer.set_center_time(float(ds.events.iloc[0]["time"]))
        viewer._update_slice_panel_at_center()  # noqa: SLF001
        assert sp._n_active_per_cell_rows >= 1  # noqa: SLF001
    finally:
        viewer.close()
        ds.close()
