"""Tests for the Figure-4 workflow (render-data prep + summary means)."""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from statespacecheck_paper import figure04_cache, figure04_workflow
from statespacecheck_paper.diagnostics import SpikeEventDiagnostics
from statespacecheck_paper.figure04_cache import Figure4Paths
from statespacecheck_paper.figure04_workflow import (
    compute_mean_spike_event_diagnostic,
    prepare_figure04_render_data,
)
from statespacecheck_paper.real_data_analysis import Figure4Config


def _make_per_cell_diagnostics(*, event_hpd_overlap: np.ndarray) -> SpikeEventDiagnostics:
    n_spikes = event_hpd_overlap.shape[0]
    return SpikeEventDiagnostics(
        event_time_ind=np.zeros(n_spikes, dtype=np.intp),
        event_cell_ind=np.zeros(n_spikes, dtype=np.intp),
        event_hpd_overlap=event_hpd_overlap,
        event_kl_divergence=np.zeros(n_spikes),
        event_predictive_pvalue=np.zeros(n_spikes),
        hpd_overlap=None,
        kl_divergence=None,
        predictive_pvalue=None,
        per_spike_likelihood=None,
        event_time=None,
    )


class TestComputeMeanSpikeEventDiagnostic:
    def test_uses_per_spike_array(self) -> None:
        """The summary mean uses per-spike values, not the (n_time, n_cells)
        matrix collapsed by nanmean — those differ when multiple spikes share a
        (time, cell)."""
        diagnostics = _make_per_cell_diagnostics(event_hpd_overlap=np.array([0.0, 1.0, 1.0]))
        assert compute_mean_spike_event_diagnostic(diagnostics, "hpd_overlap") == pytest.approx(
            2.0 / 3.0
        )

    def test_raises_when_event_array_missing(self) -> None:
        diagnostics = _make_per_cell_diagnostics(event_hpd_overlap=np.array([0.5]))
        with pytest.raises(KeyError, match="event_made_up_metric"):
            compute_mean_spike_event_diagnostic(diagnostics, "made_up_metric")


def _synthetic_recording() -> dict[str, Any]:
    """A tiny in-memory stand-in for load_neural_recording_from_files output."""
    n_time = 8
    position_info = pd.DataFrame(
        {
            "head_position_x": np.linspace(0.0, 1.0, n_time),
            "head_position_y": np.linspace(1.0, 0.0, n_time),
            "linear_position": np.linspace(0.0, 2.0, n_time),
        },
        index=np.linspace(0.0, 0.014, n_time),
    )
    return {
        "position_info": position_info,
        "spike_times": [np.array([0.001, 0.005]), np.array([0.010])],
        "track_graph": object(),
        "linear_edge_order": [(0, 1)],
        "linear_edge_spacing": 0.0,
    }


def _synthetic_payload() -> dict[str, Any]:
    return {
        "continuous_results": np.zeros(3),
        "contfrag_results": np.ones(3),
        "continuous_diagnostics": {"tag": "cont"},
        "contfrag_diagnostics": {"tag": "cf"},
        "spike_counts": np.zeros((8, 2), dtype=np.int64),
        "place_field_peaks": np.zeros(2),
        "diagnostic_place_fields": np.zeros((2, 4)),
        "diagnostic_position_bins": np.arange(4.0),
    }


class TestPrepareRenderData:
    """Provenance cache + path injection for ``prepare_figure04_render_data``,
    exercised entirely on synthetic inputs (no real data, no decoder)."""

    def test_cache_invalidates_on_config_change(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setattr(
            figure04_workflow,
            "load_neural_recording_from_files",
            lambda *a, **k: _synthetic_recording(),
        )
        calls = {"n": 0}

        def fake_compute(**kwargs: Any) -> dict[str, Any]:
            calls["n"] += 1
            return _synthetic_payload()

        monkeypatch.setattr(figure04_workflow, "_compute_figure04_decode_results", fake_compute)

        paths = Figure4Paths(data_path=tmp_path, animal_date_epoch="synthetic_epoch")
        config = Figure4Config()

        prepare_figure04_render_data(config, paths, use_cache=True)
        assert calls["n"] == 1
        assert paths.cache_path.exists()

        prepare_figure04_render_data(config, paths, use_cache=True)
        assert calls["n"] == 1  # cache hit

        changed = dataclasses.replace(config, movement_var=config.movement_var + 1.0)
        prepare_figure04_render_data(changed, paths, use_cache=True)
        assert calls["n"] == 2  # fingerprint mismatch -> recompute

    def test_cache_invalidates_on_dependency_change(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setattr(
            figure04_workflow,
            "load_neural_recording_from_files",
            lambda *a, **k: _synthetic_recording(),
        )
        calls = {"n": 0}

        def fake_compute(**kwargs: Any) -> dict[str, Any]:
            calls["n"] += 1
            return _synthetic_payload()

        monkeypatch.setattr(figure04_workflow, "_compute_figure04_decode_results", fake_compute)
        monkeypatch.setattr(
            figure04_cache, "_installed_non_local_detector_version", lambda: "1.0.0"
        )

        paths = Figure4Paths(data_path=tmp_path, animal_date_epoch="synthetic_epoch")
        config = Figure4Config()
        prepare_figure04_render_data(config, paths, use_cache=True)
        assert calls["n"] == 1

        monkeypatch.setattr(
            figure04_cache, "_installed_non_local_detector_version", lambda: "2.0.0"
        )
        prepare_figure04_render_data(config, paths, use_cache=True)
        assert calls["n"] == 2

    def test_uses_injected_paths_and_maps_serialized_keys(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        seen: dict[str, Any] = {}

        def spy_load(data_path: Any, animal_date_epoch: Any) -> dict[str, Any]:
            seen["data_path"] = data_path
            seen["animal_date_epoch"] = animal_date_epoch
            return _synthetic_recording()

        monkeypatch.setattr(figure04_workflow, "load_neural_recording_from_files", spy_load)
        monkeypatch.setattr(
            figure04_workflow, "_compute_figure04_decode_results", lambda **k: _synthetic_payload()
        )

        injected = Figure4Paths(data_path=tmp_path, animal_date_epoch="injected_epoch")
        render_data = prepare_figure04_render_data(Figure4Config(), injected, use_cache=False)

        assert seen == {"data_path": tmp_path, "animal_date_epoch": "injected_epoch"}
        assert injected.cache_path.exists()
        # The serialized ``contfrag_*`` keys map onto the spelled-out fields.
        np.testing.assert_array_equal(render_data.continuous_fragmented_results, np.ones(3))
        assert render_data.continuous_fragmented_diagnostics == {"tag": "cf"}
        assert render_data.spike_counts.shape == (8, 2)
