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
from statespacecheck_paper.figure04_decoder import Figure4Config
from statespacecheck_paper.figure04_workflow import (
    Figure4DecodeResults,
    Figure4RenderData,
    Figure4Summary,
    compute_figure04_summary,
    compute_mean_spike_event_diagnostic,
    format_figure04_summary,
    prepare_figure04_render_data,
)
from statespacecheck_paper.load_local_data import NeuralRecordingData


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


def _diagnostics_at(time_ind: list[int], cell_ind: list[int]) -> SpikeEventDiagnostics:
    """Diagnostics whose per-spike event indices are placed explicitly."""
    n_spikes = len(time_ind)
    return SpikeEventDiagnostics(
        event_time_ind=np.asarray(time_ind, dtype=np.intp),
        event_cell_ind=np.asarray(cell_ind, dtype=np.intp),
        event_hpd_overlap=np.full(n_spikes, 0.5),
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


def _ds(values: np.ndarray) -> xr.Dataset:
    """A tiny decoder-result Dataset carrying a ``time`` coordinate."""
    return xr.Dataset(
        {"filter": ("time", values)},
        coords={"time": np.arange(len(values), dtype=float)},
    )


def _synthetic_decode_results() -> Figure4DecodeResults:
    n_time, n_cells, n_bins = 8, 2, 4
    return Figure4DecodeResults(
        continuous_results=_ds(np.zeros(n_time)),
        continuous_fragmented_results=_ds(np.ones(n_time)),
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


class TestFigure4Summary:
    def _render_data(self) -> Figure4RenderData:
        decode = dataclasses.replace(
            _synthetic_decode_results(),
            continuous_diagnostics=_diagnostics(np.array([0.01, 0.20])),
            continuous_fragmented_diagnostics=_diagnostics(np.array([0.40, 0.20])),
        )
        return Figure4RenderData(
            recording=_synthetic_recording(),
            time=np.arange(8, dtype=float),
            head_position=np.zeros((8, 2)),
            linear_position=np.zeros(8),
            decode_results=decode,
        )

    def test_computes_typed_means_and_flag_counts(self) -> None:
        summary = compute_figure04_summary(
            self._render_data(),
            thresholds={"hpd_overlap": 0.05},
            metric_directions={"hpd_overlap": "below"},
        )

        assert isinstance(summary, Figure4Summary)
        assert summary.continuous.hpd_overlap == pytest.approx(0.105)
        assert summary.continuous_fragmented.hpd_overlap == pytest.approx(0.30)
        assert len(summary.flag_confusions) == 1
        confusion = summary.flag_confusions[0]
        assert (
            confusion.n,
            confusion.both,
            confusion.a_only,
            confusion.b_only,
            confusion.neither,
        ) == (
            2,
            0,
            1,
            0,
            1,
        )

    def test_formatter_does_not_recompute(self) -> None:
        summary = compute_figure04_summary(
            self._render_data(),
            thresholds={"hpd_overlap": 0.05},
            metric_directions={"hpd_overlap": "below"},
        )
        text = format_figure04_summary(summary)
        assert "Continuous:" in text
        assert "hpd_overlap: 0.1050" in text
        assert "cont-only=1" in text

    def test_requires_threshold_for_every_direction(self) -> None:
        with pytest.raises(ValueError, match="without thresholds"):
            compute_figure04_summary(
                self._render_data(),
                thresholds={},
                metric_directions={"hpd_overlap": "below"},
            )


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

    def test_copies_input_arrays_and_leaves_caller_writable(self) -> None:
        # Freezing must not reach back into a caller-owned array, and the stored
        # copy must be isolated from later mutation of that array.
        spike_counts = np.zeros((8, 2), dtype=np.int64)
        decode = Figure4DecodeResults(
            continuous_results=_ds(np.zeros(8)),
            continuous_fragmented_results=_ds(np.zeros(8)),
            continuous_diagnostics=_diagnostics(np.array([0.5])),
            continuous_fragmented_diagnostics=_diagnostics(np.array([0.5])),
            spike_counts=spike_counts,
            place_field_peaks=np.zeros(2),
            diagnostic_place_fields=np.zeros((2, 4)),
            diagnostic_position_bins=np.arange(4.0),
        )
        assert spike_counts.flags.writeable  # caller's array untouched
        spike_counts[0, 0] = 7
        assert decode.spike_counts[0, 0] == 0  # stored copy isolated
        assert not decode.spike_counts.flags.writeable

    def test_rejects_dataset_timeline_mismatch(self) -> None:
        with pytest.raises(ValueError, match="decode timelines must match"):
            Figure4DecodeResults(
                continuous_results=_ds(np.zeros(3)),
                continuous_fragmented_results=_ds(np.zeros(8)),
                continuous_diagnostics=_diagnostics(np.array([0.5])),
                continuous_fragmented_diagnostics=_diagnostics(np.array([0.5])),
                spike_counts=np.zeros((8, 2), dtype=np.int64),  # 8 != dataset's 3
                place_field_peaks=np.zeros(2),
                diagnostic_place_fields=np.zeros((2, 4)),
                diagnostic_position_bins=np.arange(4.0),
            )

    def test_rejects_dataset_without_time_coordinate(self) -> None:
        # A time *dimension* is not enough; compose_figure04 reads the ``time``
        # coordinate, so require it at construction.
        with pytest.raises(ValueError, match="must carry a 'time' coordinate"):
            Figure4DecodeResults(
                continuous_results=xr.Dataset({"filter": ("time", np.zeros(8))}),
                continuous_fragmented_results=_ds(np.zeros(8)),
                continuous_diagnostics=_diagnostics(np.array([0.5])),
                continuous_fragmented_diagnostics=_diagnostics(np.array([0.5])),
                spike_counts=np.zeros((8, 2), dtype=np.int64),
                place_field_peaks=np.zeros(2),
                diagnostic_place_fields=np.zeros((2, 4)),
                diagnostic_position_bins=np.arange(4.0),
            )

    def test_rejects_mismatched_time_coordinates(self) -> None:
        # Same length, different coordinate values: the two decoders were run on
        # different windows. A length-only check would miss this.
        with pytest.raises(ValueError, match="different 'time' coordinates"):
            Figure4DecodeResults(
                continuous_results=xr.Dataset(
                    {"filter": ("time", np.zeros(8))},
                    coords={"time": np.arange(8, dtype=float)},
                ),
                continuous_fragmented_results=xr.Dataset(
                    {"filter": ("time", np.zeros(8))},
                    coords={"time": np.arange(8, dtype=float) + 100.0},
                ),
                continuous_diagnostics=_diagnostics(np.array([0.5])),
                continuous_fragmented_diagnostics=_diagnostics(np.array([0.5])),
                spike_counts=np.zeros((8, 2), dtype=np.int64),
                place_field_peaks=np.zeros(2),
                diagnostic_place_fields=np.zeros((2, 4)),
                diagnostic_position_bins=np.arange(4.0),
            )

    def test_rejects_diagnostic_event_index_out_of_range(self) -> None:
        with pytest.raises(ValueError, match="event_time_ind falls outside"):
            Figure4DecodeResults(
                continuous_results=_ds(np.zeros(8)),
                continuous_fragmented_results=_ds(np.zeros(8)),
                continuous_diagnostics=_diagnostics_at([99], [0]),  # 99 >= n_time 8
                continuous_fragmented_diagnostics=_diagnostics(np.array([0.5])),
                spike_counts=np.zeros((8, 2), dtype=np.int64),
                place_field_peaks=np.zeros(2),
                diagnostic_place_fields=np.zeros((2, 4)),
                diagnostic_position_bins=np.arange(4.0),
            )


class TestFigure4RenderData:
    def _render_data(
        self,
        *,
        time: np.ndarray,
        decode_results: Figure4DecodeResults | None = None,
    ) -> Figure4RenderData:
        n_time = time.shape[0]
        return Figure4RenderData(
            recording=_synthetic_recording(),
            time=time,
            head_position=np.zeros((n_time, 2)),
            linear_position=np.zeros(n_time),
            decode_results=decode_results or _synthetic_decode_results(),
        )

    def test_rejects_non_1d_time(self) -> None:
        with pytest.raises(ValueError, match="time must be 1-D"):
            self._render_data(time=np.zeros((8, 1)))

    def test_rejects_decode_timeline_mismatch(self) -> None:
        # time has 3 samples; the synthetic decode results have 8.
        with pytest.raises(ValueError, match="does not match the recording timeline"):
            self._render_data(time=np.zeros(3))

    def test_copies_derived_arrays_and_leaves_caller_writable(self) -> None:
        time = np.linspace(0.0, 1.0, 8)
        render = self._render_data(time=time)
        assert time.flags.writeable  # caller's array untouched
        assert not render.time.flags.writeable
        time[0] = 99.0
        assert render.time[0] == 0.0  # stored copy isolated


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

        changed = dataclasses.replace(
            config,
            provenance=dataclasses.replace(
                config.provenance, movement_var=config.provenance.movement_var + 1.0
            ),
        )
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
