"""Tests for the Figure-4 workflow (render-data prep, decode results, means)."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from statespacecheck_paper import figure04_cache, figure04_workflow
from statespacecheck_paper.diagnostics import SpikeEventDiagnostics
from statespacecheck_paper.figure04_cache import _FIGURE04_CACHE_PAYLOAD_KEYS, Figure4Paths
from statespacecheck_paper.figure04_workflow import (
    Figure4DecodeResults,
    compute_mean_spike_event_diagnostic,
    prepare_figure04_render_data,
)
from statespacecheck_paper.load_local_data import NeuralRecordingData
from statespacecheck_paper.real_data_analysis import Figure4Config


def _diagnostics(event_hpd_overlap: np.ndarray) -> SpikeEventDiagnostics:
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


def _synthetic_recording() -> NeuralRecordingData:
    n_time = 8
    position_info = pd.DataFrame(
        {
            "head_position_x": np.linspace(0.0, 1.0, n_time),
            "head_position_y": np.linspace(1.0, 0.0, n_time),
            "linear_position": np.linspace(0.0, 2.0, n_time),
        },
        index=np.linspace(0.0, 0.014, n_time),
    )
    track_graph = nx.Graph()
    track_graph.add_edge(0, 1)
    return NeuralRecordingData(
        position_info=position_info,
        spike_times=(np.array([0.001, 0.005]), np.array([0.010])),
        track_graph=track_graph,
        linear_edge_order=((0, 1),),
        linear_edge_spacing=0.0,
    )


def _synthetic_decode_results() -> Figure4DecodeResults:
    n_time, n_cells, n_bins = 8, 2, 4
    return Figure4DecodeResults(
        continuous_results=xr.Dataset({"filter": ("time", np.zeros(n_time))}),
        continuous_fragmented_results=xr.Dataset({"filter": ("time", np.ones(n_time))}),
        continuous_diagnostics=_diagnostics(np.array([0.5])),
        continuous_fragmented_diagnostics=_diagnostics(np.array([0.5, 0.5])),
        spike_counts=np.zeros((n_time, n_cells), dtype=np.int64),
        place_field_peaks=np.zeros(n_cells),
        diagnostic_place_fields=np.zeros((n_cells, n_bins)),
        diagnostic_position_bins=np.arange(float(n_bins)),
    )


class TestComputeMeanSpikeEventDiagnostic:
    def test_uses_per_spike_array(self) -> None:
        diagnostics = _diagnostics(np.array([0.0, 1.0, 1.0]))
        assert compute_mean_spike_event_diagnostic(diagnostics, "hpd_overlap") == pytest.approx(
            2.0 / 3.0
        )

    def test_raises_when_event_array_missing(self) -> None:
        with pytest.raises(KeyError, match="event_made_up_metric"):
            compute_mean_spike_event_diagnostic(_diagnostics(np.array([0.5])), "made_up_metric")


class TestFigure4DecodeResults:
    def test_cache_payload_round_trip_through_contfrag_keys(self) -> None:
        decode = _synthetic_decode_results()
        payload = decode.to_cache_payload()
        # The serialized keys keep the historical ``contfrag_*`` spelling.
        assert "contfrag_results" in payload
        assert "contfrag_diagnostics" in payload
        rebuilt = Figure4DecodeResults.from_cache_payload(payload)
        assert rebuilt.continuous_fragmented_results is decode.continuous_fragmented_results
        assert rebuilt.continuous_fragmented_diagnostics is decode.continuous_fragmented_diagnostics
        np.testing.assert_array_equal(rebuilt.spike_counts, decode.spike_counts)

    def test_payload_keys_match_cache_module_source_of_truth(self) -> None:
        # Guard against drift between the on-disk key list (owned by
        # figure04_cache) and the field->key mapping in to_cache_payload.
        assert set(_synthetic_decode_results().to_cache_payload().keys()) == set(
            _FIGURE04_CACHE_PAYLOAD_KEYS
        )

    def test_from_cache_payload_rejects_missing_key(self) -> None:
        payload = _synthetic_decode_results().to_cache_payload()
        del payload["spike_counts"]
        with pytest.raises(ValueError, match="missing keys"):
            Figure4DecodeResults.from_cache_payload(payload)

    def test_from_cache_payload_rejects_wrong_type(self) -> None:
        payload = _synthetic_decode_results().to_cache_payload()
        payload["continuous_results"] = np.zeros(3)  # not an xr.Dataset
        with pytest.raises(TypeError, match="xr.Dataset"):
            Figure4DecodeResults.from_cache_payload(payload)

    def test_rejects_incompatible_shapes(self) -> None:
        with pytest.raises(ValueError, match="place_field_peaks"):
            Figure4DecodeResults(
                continuous_results=xr.Dataset(),
                continuous_fragmented_results=xr.Dataset(),
                continuous_diagnostics=_diagnostics(np.array([0.5])),
                continuous_fragmented_diagnostics=_diagnostics(np.array([0.5])),
                spike_counts=np.zeros((8, 2), dtype=np.int64),
                place_field_peaks=np.zeros(3),  # should be (2,)
                diagnostic_place_fields=np.zeros((2, 4)),
                diagnostic_position_bins=np.arange(4.0),
            )

    def test_arrays_are_read_only(self) -> None:
        decode = _synthetic_decode_results()
        with pytest.raises(ValueError, match="read-only|write"):
            decode.spike_counts[0, 0] = 1


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

        def fake_compute(*a: object, **k: object) -> Figure4DecodeResults:
            calls["n"] += 1
            return _synthetic_decode_results()

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

        def fake_compute(*a: object, **k: object) -> Figure4DecodeResults:
            calls["n"] += 1
            return _synthetic_decode_results()

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
        seen: dict[str, object] = {}

        def spy_load(data_path: object, animal_date_epoch: object) -> NeuralRecordingData:
            seen["data_path"] = data_path
            seen["animal_date_epoch"] = animal_date_epoch
            return _synthetic_recording()

        monkeypatch.setattr(figure04_workflow, "load_neural_recording_from_files", spy_load)
        monkeypatch.setattr(
            figure04_workflow,
            "_compute_figure04_decode_results",
            lambda *a, **k: _synthetic_decode_results(),
        )

        injected = Figure4Paths(data_path=tmp_path, animal_date_epoch="injected_epoch")
        render_data = prepare_figure04_render_data(Figure4Config(), injected, use_cache=False)

        assert seen == {"data_path": tmp_path, "animal_date_epoch": "injected_epoch"}
        assert injected.cache_path.exists()
        # The recording and typed decode results are threaded through by attribute.
        assert render_data.recording.spike_times[0].shape == (2,)
        np.testing.assert_array_equal(
            render_data.decode_results.continuous_fragmented_results["filter"].to_numpy(),
            np.ones(8),
        )
        assert render_data.decode_results.spike_counts.shape == (8, 2)
        assert render_data.time.shape == (8,)
        assert render_data.head_position.shape == (8, 2)
