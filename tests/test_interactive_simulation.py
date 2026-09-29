"""Tests for the figure-3 simulation cache path.

Builds a tiny simulation cache via ``build_simulated_cache``, opens it
through ``DecoderDataSource.for_simulation``, and exercises the loader
contract + the viewer's adaptation to ``dataset_kind == "simulation"``.

The simulation params are scaled down (``phase_boundaries`` shrink the
timeline from 32,000 to 900 steps) so the forward filter runs quickly on CI;
the assertions don't depend on phase-specific behaviour, just on the
end-to-end shape contract.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ._qt import (
    qt_offscreen,  # noqa: F401 -- registers the autouse fixture here
    wait_for_request,
)

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


def _tiny_params():
    """``Figure3Config`` shrunk so the simulation runs fast in tests."""
    from statespacecheck_paper.figure03_protocol import Figure3Config

    return Figure3Config(
        phase_boundaries=(200, 300, 400, 500, 600, 700, 800, 900),
    )


def _build_simulated(cache_dir: Path) -> dict[str, object]:
    from statespacecheck_paper.interactive.cache import build_simulated_cache

    return build_simulated_cache(
        cache_dir,
        config=_tiny_params(),
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
        assert ds.has_smoothed is False
        # Single state, all bins interior.
        assert ds.n_states == 1
        assert ds.n_interior == ds.position_bins.shape[0]
        # ``time`` grid is ``np.arange(n_time) * 0.001`` (1 ms/step, matching
        # the manuscript / Figure3Config convention).
        assert ds.n_time > 0
        np.testing.assert_allclose(ds.time[1] - ds.time[0], 0.001, atol=1e-12)
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

    from statespacecheck_paper.figure03_simulation import run_figure03_simulation
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
    sim_lik = np.asarray(sim.diagnostics.combined_likelihood, dtype=np.float64)
    sim_peak = sim_lik.max(axis=1, keepdims=True)
    sim_peak = np.where(sim_peak > 0, sim_peak, 1.0)
    sim_lik_peak_normed = (sim_lik / sim_peak).astype(np.float32)

    # Should agree everywhere; tolerance accounts for float32 round-trip.
    np.testing.assert_allclose(lik_recovered, sim_lik_peak_normed, atol=1e-4, rtol=1e-3)


def test_simulated_event_likelihood_round_trips_in_event_order(tmp_path: Path) -> None:
    """The viewer sidecar retains the decoder likelihood for every event."""
    from statespacecheck_paper.figure03_simulation import run_figure03_simulation
    from statespacecheck_paper.interactive.data_source import DecoderDataSource

    params = _tiny_params()
    sim = run_figure03_simulation(params, seed=0)
    _build_simulated(tmp_path)
    ds = DecoderDataSource.for_simulation(tmp_path)
    try:
        assert ds.event_likelihood is not None
        assert ds.event_likelihood.flags.writeable is False
        event_times = sim.diagnostics.event_time_ind.astype(np.float64) * 0.001
        event_order = np.argsort(event_times, kind="stable")
        np.testing.assert_allclose(
            ds.event_likelihood,
            sim.diagnostics.event_likelihood[event_order].astype(np.float32),
        )

        remap_start, remap_end = params.phase_boundaries[:2]
        in_remap = (ds.event_time_idx >= remap_start) & (ds.event_time_idx < remap_end)
        assert in_remap.any(), "tiny simulation produced no remap-window events"
        remap_rows = np.flatnonzero(in_remap)
        static_rows = ds.place_fields[ds.event_cell_ids[remap_rows]]
        assert not np.allclose(ds.event_likelihood[remap_rows], static_rows)
    finally:
        ds.close()


def test_simulated_cache_records_figure03_flag_thresholds(tmp_path: Path) -> None:
    """The simulation cache carries Figure 3's thresholds, not Figure 4's cutoffs."""
    import json

    from statespacecheck_paper.interactive.data_source import DecoderDataSource
    from statespacecheck_paper.paths import FIGURE03_SUMMARY_PATH

    rules = json.loads(FIGURE03_SUMMARY_PATH.read_text(encoding="utf-8"))["flag_rules"]
    _build_simulated(tmp_path)
    ds = DecoderDataSource.for_simulation(tmp_path)
    try:
        assert ds.flag_thresholds == {metric: rule["threshold"] for metric, rule in rules.items()}
        # Figure 3's HPD threshold is the pooled-baseline 1st percentile (0),
        # not Figure 4's fixed 0.05, and KL divergence has a threshold.
        assert ds.flag_thresholds["hpd_overlap"] == 0.0
        assert "kl_divergence" in ds.flag_thresholds
    finally:
        ds.close()


def test_simulation_cache_without_thresholds_is_rejected(tmp_path: Path) -> None:
    """An older simulation cache must be rebuilt, not drawn with Figure 4's cutoffs."""
    from statespacecheck_paper.interactive.cache import simulated_meta_path
    from statespacecheck_paper.interactive.data_source import DecoderDataSource

    _build_simulated(tmp_path)
    meta_file = simulated_meta_path(tmp_path)
    with np.load(meta_file) as meta:
        kept = {key: meta[key] for key in ("time", "linear_position", "n_cells")}
    np.savez(meta_file, **kept)

    with pytest.raises(ValueError, match="no flag thresholds"):
        DecoderDataSource.for_simulation(tmp_path)


# ---------------------------------------------------------------------------
# Viewer wiring
# ---------------------------------------------------------------------------


def test_simulated_viewer_draws_the_cached_thresholds(tmp_path: Path) -> None:
    """Each metric panel's threshold line sits at the cache's threshold."""
    from PySide6 import QtWidgets

    from statespacecheck_paper.interactive.cache import build_simulated_cache
    from statespacecheck_paper.interactive.data_source import DecoderDataSource
    from statespacecheck_paper.interactive.viewer import DecoderViewer
    from statespacecheck_paper.plotting import negative_log_pvalue
    from statespacecheck_paper.style import METRIC_SPECS

    thresholds = {"hpd_overlap": 0.0, "predictive_pvalue": 0.01, "kl_divergence": 4.5}
    build_simulated_cache(
        tmp_path, config=_tiny_params(), seed=0, time_chunk=128, flag_thresholds=thresholds
    )
    _ = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    viewer = DecoderViewer(DecoderDataSource.for_simulation(tmp_path))
    try:
        for spec in METRIC_SPECS:
            line = viewer.metric_panels[spec.event_attr]._threshold_line  # noqa: SLF001
            assert line is not None, spec.name
            expected = thresholds[spec.name]
            if spec.display_transform == "neg_log_p":
                expected = float(negative_log_pvalue(expected))
            assert line.value() == pytest.approx(expected), spec.name
    finally:
        viewer.close()


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
        # Smoothed overlay disabled (data source has no smoothed posterior).
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
    ``_buffer_predictive`` and ``_buffer_lik`` with the expected shapes.
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
        assert wait_for_request(app, viewer, target)

        sp = viewer.slice_panel
        assert sp._buffer_predictive is not None  # noqa: SLF001
        assert sp._buffer_lik is not None  # noqa: SLF001
        assert sp._buffer_smoothed is None  # noqa: SLF001 — no smoothed posterior in simulation

        # Pick a time at a real event so per-cell rows are guaranteed
        # populated.
        assert len(ds.events) > 0
        viewer.set_center_time(float(ds.events.iloc[0]["time"]))
        viewer._update_slice_panel_at_center()  # noqa: SLF001
        assert sp._n_active_per_cell_rows >= 1  # noqa: SLF001
    finally:
        viewer.close()
        ds.close()


def test_simulated_viewer_uses_event_likelihood_in_remap(tmp_path: Path) -> None:
    """Per-cell rows show the active remapped likelihood, not baseline PFs."""
    from PySide6 import QtWidgets

    from statespacecheck_paper.interactive.data_source import DecoderDataSource
    from statespacecheck_paper.interactive.viewer import DecoderViewer

    params = _tiny_params()
    _build_simulated(tmp_path)
    _ = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    ds = DecoderDataSource.for_simulation(tmp_path)
    viewer = DecoderViewer(ds)
    try:
        remap_start, remap_end = params.phase_boundaries[:2]
        candidates = np.flatnonzero(
            (ds.event_time_idx >= remap_start) & (ds.event_time_idx < remap_end)
        )
        candidates = candidates[
            np.array(
                [
                    not np.allclose(
                        ds.event_likelihood_at(int(i), int(ds.event_cell_ids[i])),
                        ds.place_fields[int(ds.event_cell_ids[i])],
                    )
                    for i in candidates
                ]
            )
        ]
        assert candidates.size > 0, "tiny simulation produced no visibly remapped event"
        event_idx = int(candidates[0])
        t_idx = int(ds.event_time_idx[event_idx])
        cell_id = int(ds.event_cell_ids[event_idx])
        i0, i1 = ds.event_indices_at(t_idx)
        first_event = next(i for i in range(i0, i1) if int(ds.event_cell_ids[i]) == cell_id)

        rows, _ = viewer._per_cell_slices_at(t_idx)  # noqa: SLF001
        row = next(item for item in rows if item.cell_id == cell_id)
        expected = ds.event_likelihood_at(first_event, cell_id)[: ds.n_interior]
        expected = expected / expected.max()
        np.testing.assert_allclose(row.place_field_norm[ds.interior_mask], expected)

        static = ds.place_fields[cell_id, : ds.n_interior]
        static = static / static.max()
        assert not np.allclose(expected, static)
    finally:
        viewer.close()
        ds.close()


def test_simulation_cache_missing_event_likelihood_raises(tmp_path: Path) -> None:
    """Static place fields cannot replace phase-specific simulation likelihoods."""
    from statespacecheck_paper.interactive.data_source import DecoderDataSource

    _build_simulated(tmp_path)
    ds = DecoderDataSource.for_simulation(tmp_path)
    try:
        ds.event_likelihood = None
        with pytest.raises(ValueError, match="missing event_likelihood"):
            ds.event_likelihood_at(0, int(ds.event_cell_ids[0]))
    finally:
        ds.close()
