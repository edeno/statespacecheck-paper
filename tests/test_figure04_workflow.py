"""Tests for the Figure-4 workflow (render-data prep, decode results, means)."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from statespacecheck_paper import figure04_cache, figure04_fit, figure04_workflow
from statespacecheck_paper.diagnostics import (
    METRIC_FLAG_DIRECTIONS,
    FlagDirection,
    SpikeEventDiagnostics,
)
from statespacecheck_paper.figure04_cache import (
    _FIGURE04_DECODE_AND_DIAGNOSTICS_PAYLOAD_KEYS,
    _FIGURE04_DECODE_PAYLOAD_KEYS,
    _FIGURE04_DIAGNOSTICS_PAYLOAD_KEYS,
    Figure4Paths,
)
from statespacecheck_paper.figure04_decoder import (
    Figure4Config,
    create_decoder_environment,
    fit_decoder_models,
)
from statespacecheck_paper.figure04_input import (
    HEAD_POSITION_COLUMNS,
    INPUT_FILE_SUFFIX,
    NeuralRecordingData,
)
from statespacecheck_paper.figure04_place_fields import DECODER_PREDICTIVE_VAR
from statespacecheck_paper.figure04_summary import (
    Figure4DetailWindowComparison,
    Figure4Summary,
    compute_detail_window_comparison,
    compute_figure04_reported_statistics,
    compute_immobility_comparison,
    compute_kl_divergence_comparison,
    compute_mean_spike_event_diagnostic,
    count_flagged_with_nonzero_overlap,
    count_units_with_spikes,
    format_figure04_summary,
    predictive_state_share,
    summarize_figure04_diagnostics,
)
from statespacecheck_paper.figure04_workflow import (
    Figure4AnalysisResults,
    Figure4RenderData,
    prepare_figure04_render_data,
)

from ._diagnostics import event_diagnostics
from ._figure04 import synthetic_cache_provenance, synthetic_render_data


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
        {"acausal_posterior": ("time", values)},
        coords={"time": np.arange(len(values), dtype=float)},
    )


def _paths_with_input(data_path: Path, epoch: str) -> Figure4Paths:
    """Paths of an epoch whose input file exists (the fingerprints hash its bytes)."""
    (data_path / f"{epoch}{INPUT_FILE_SUFFIX}").write_bytes(b"input")
    return Figure4Paths(data_path=data_path, animal_date_epoch=epoch)


def _cache_payload(
    results: Figure4AnalysisResults,
    keys: tuple[str, ...] = _FIGURE04_DECODE_AND_DIAGNOSTICS_PAYLOAD_KEYS,
) -> dict[str, object]:
    """The serialized cache payload of ``results``: its keys are the field names."""
    return {key: getattr(results, key) for key in keys}


def _synthetic_analysis_results() -> Figure4AnalysisResults:
    n_time, n_cells, n_bins = 8, 2, 4
    return Figure4AnalysisResults(
        continuous_results=_ds(np.zeros(n_time)),
        continuous_fragmented_results=_ds(np.ones(n_time)),
        continuous_diagnostics=event_diagnostics(hpd=np.array([0.5])),
        continuous_fragmented_diagnostics=event_diagnostics(hpd=np.array([0.5, 0.5])),
        spike_counts=np.zeros((n_time, n_cells), dtype=np.int64),
        place_field_peaks=np.zeros(n_cells),
        diagnostic_place_fields=np.zeros((n_cells, n_bins)),
        diagnostic_position_bins=np.arange(float(n_bins)),
    )


class TestComputeMeanSpikeEventDiagnostic:
    def test_uses_per_spike_array(self) -> None:
        diagnostics = event_diagnostics(hpd=np.array([0.0, 1.0, 1.0]))
        assert compute_mean_spike_event_diagnostic(diagnostics, "hpd_overlap") == pytest.approx(
            2.0 / 3.0
        )

    def test_raises_when_event_array_missing(self) -> None:
        with pytest.raises(KeyError, match="event_made_up_metric"):
            compute_mean_spike_event_diagnostic(
                event_diagnostics(hpd=np.array([0.5])), "made_up_metric"
            )


class TestComputeDetailWindowComparison:
    THRESHOLDS = {"hpd_overlap": 0.05, "predictive_pvalue": 0.05}
    DIRECTIONS: dict[str, FlagDirection] = {"hpd_overlap": "below", "predictive_pvalue": "below"}
    # Speed per decode sample (10 samples) and Fragmented share per sample of [2, 8).
    SPEED = np.array([10.0, 10.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 10.0, 10.0])
    SHARE = np.array([0.1, 0.2, 0.6, 0.9, 0.4, 0.1])

    def _compare(
        self,
        continuous: SpikeEventDiagnostics,
        fragmented: SpikeEventDiagnostics,
        time_slice: slice | None = None,
        share: np.ndarray | None = None,
    ) -> Figure4DetailWindowComparison:
        time_slice = slice(2, 8) if time_slice is None else time_slice
        return compute_detail_window_comparison(
            continuous,
            fragmented,
            time_slice,
            thresholds=self.THRESHOLDS,
            metric_directions=self.DIRECTIONS,
            speed_cm_per_s=self.SPEED,
            fragmented_share=(
                np.full(time_slice.stop - time_slice.start, 0.1) if share is None else share
            ),
            fragmented_probability_threshold=0.8,
        )

    def test_counts_only_events_inside_the_window(self) -> None:
        # Events at bins 1 and 9 fall outside [2, 8); the five inside are scored.
        time_ind = [1, 2, 3, 4, 5, 7, 9]
        continuous = event_diagnostics(
            hpd=np.array([0.0, 0.0, 0.0, 0.0, 0.5, 0.5, 0.0]),
            kl=np.array([9.0, 1.0, 2.0, 3.0, 4.0, 5.0, 9.0]),
            pvalue=np.array([0.5, 0.01, 0.2, 0.3, 0.4, 0.5, 0.01]),
            time_ind=time_ind,
        )
        fragmented = event_diagnostics(
            hpd=np.array([0.5, 0.5, 0.0, 0.5, 0.0, 0.5, 0.5]),
            kl=np.array([9.0, 2.5, 2.0, 1.5, 1.0, 0.5, 9.0]),
            pvalue=np.array([0.1, 0.02, 0.1, 0.3, 0.9, 0.5, 0.9]),
            time_ind=time_ind,
        )
        comparison = self._compare(continuous, fragmented, share=self.SHARE)
        assert comparison.n_events == 5
        hpd = comparison.hpd_overlap
        assert (hpd.n, hpd.both, hpd.rescued, hpd.newly_flagged, hpd.neither) == (5, 1, 2, 1, 1)
        # Only bin 2 has p <= 0.05, under both decoders.
        pvalue = comparison.predictive_pvalue
        assert (pvalue.both, pvalue.rescued, pvalue.newly_flagged, pvalue.neither) == (1, 0, 0, 4)
        # Inside the window, p rises for bins 2 and 5, falls for 3, ties at 4 and 7.
        assert comparison.predictive_pvalue_higher == 2
        assert comparison.continuous_kl_median == pytest.approx(3.0)
        assert comparison.continuous_fragmented_kl_median == pytest.approx(1.5)
        # Speed over samples 2-7 is 1..6 cm/s.
        assert comparison.median_speed_cm_per_s == pytest.approx(3.5)
        assert comparison.fragmented_share_median == pytest.approx(0.3)
        assert comparison.fragmented_share_peak == pytest.approx(0.9)
        # Only the 0.9 sample exceeds the 0.8 threshold.
        assert comparison.fragmented_probability_threshold == 0.8
        assert comparison.fragmented_share_above_threshold_samples == 1

    def test_rejects_a_share_that_does_not_cover_the_window(self) -> None:
        diagnostics = event_diagnostics(hpd=np.array([0.5]), time_ind=[3])
        with pytest.raises(ValueError, match="one value per window sample"):
            self._compare(diagnostics, diagnostics, share=np.full(5, 0.1))

    def test_pvalue_count_ignores_rounding_noise(self) -> None:
        """Two models' p-values that differ only by round-off are ties, whichever way
        the rounding falls on a given CPU. Only the last event is really higher."""
        time_ind = [0, 1, 2, 3]
        continuous = event_diagnostics(
            hpd=np.full(4, 0.5), pvalue=np.array([0.5, 0.5, 0.5, 0.5]), time_ind=time_ind
        )
        fragmented = event_diagnostics(
            hpd=np.full(4, 0.5),
            pvalue=np.array([0.5, 0.5 + 1e-12, 0.5 - 1e-12, 0.5 + 1e-6]),
            time_ind=time_ind,
        )
        comparison = self._compare(continuous, fragmented, slice(0, 4))
        assert comparison.predictive_pvalue_higher == 1

    def test_rejects_an_empty_window(self) -> None:
        diagnostics = event_diagnostics(hpd=np.array([0.5, 0.5]), time_ind=[0, 10])
        with pytest.raises(ValueError, match="contains no spike events"):
            self._compare(diagnostics, diagnostics)


class TestPredictiveStateShare:
    def test_sums_the_mode_over_interior_bins(self) -> None:
        # Two modes over two positions; the second Fragmented bin is outside the track.
        index = pd.MultiIndex.from_tuples(
            [("Continuous", 0.0), ("Continuous", 1.0), ("Fragmented", 0.0), ("Fragmented", 1.0)],
            names=("state", "position"),
        )
        predictive = np.array([[0.5, 0.3, 0.2, np.nan], [0.1, 0.1, 0.8, np.nan]])
        results = xr.Dataset(
            {"predictive_posterior": (("time", "state_bins"), predictive)},
            coords=xr.Coordinates.from_pandas_multiindex(index, "state_bins"),
        )
        share = predictive_state_share(results, "Fragmented", slice(0, 2))
        np.testing.assert_allclose(share, [0.2, 0.8])

    def test_rejects_an_unknown_mode(self) -> None:
        index = pd.MultiIndex.from_tuples([("Continuous", 0.0)], names=("state", "position"))
        results = xr.Dataset(
            {"predictive_posterior": (("time", "state_bins"), np.ones((1, 1)))},
            coords=xr.Coordinates.from_pandas_multiindex(index, "state_bins"),
        )
        with pytest.raises(ValueError, match="no 'Fragmented' mode"):
            predictive_state_share(results, "Fragmented", slice(0, 1))


class TestComputeImmobilityComparison:
    THRESHOLDS = {"hpd_overlap": 0.05, "predictive_pvalue": 0.05}
    DIRECTIONS: dict[str, FlagDirection] = {"hpd_overlap": "below", "predictive_pvalue": "below"}

    def test_compares_the_two_movement_states(self) -> None:
        # Samples 0-1 are immobile (< 4 cm/s, exclusive); spikes at samples 0, 1, 2, 3.
        speed = np.array([0.5, 3.9, 4.0, 20.0])
        diagnostics = event_diagnostics(
            hpd=np.array([0.0, 0.5, 0.0, 0.0]),
            kl=np.array([4.0, 2.0, 1.0, 3.0]),
            pvalue=np.array([0.5, 0.01, 0.01, 0.5]),
            time_ind=[0, 1, 2, 3],
        )
        comparison = compute_immobility_comparison(
            diagnostics,
            speed,
            4.0,
            thresholds=self.THRESHOLDS,
            metric_directions=self.DIRECTIONS,
        )
        assert comparison.speed_threshold_cm_per_s == 4.0
        # HPD flags spikes 0, 2, 3; the p-value flags 1 and 2.
        assert comparison.immobile_flag_rates == pytest.approx(
            {"hpd_overlap": 0.5, "predictive_pvalue": 0.5}
        )
        assert comparison.moving_flag_rates == pytest.approx(
            {"hpd_overlap": 1.0, "predictive_pvalue": 0.5}
        )
        assert comparison.immobile_kl_median == pytest.approx(3.0)
        assert comparison.moving_kl_median == pytest.approx(2.0)

    @pytest.mark.parametrize(
        ("speed", "match"),
        [
            (np.array([np.nan, 1.0]), "NaN"),
            (np.array([1.0, 1.0]), "both below and at or above"),
            (np.array([5.0, 9.0]), "both below and at or above"),
        ],
        ids=["nan-speed", "all-immobile", "all-moving"],
    )
    def test_rejects_unusable_input(self, speed: np.ndarray, match: str) -> None:
        diagnostics = event_diagnostics(
            hpd=np.array([0.0, 0.0]), pvalue=np.array([0.01, 0.01]), time_ind=[0, 1]
        )
        with pytest.raises(ValueError, match=match):
            compute_immobility_comparison(
                diagnostics,
                speed,
                4.0,
                thresholds=self.THRESHOLDS,
                metric_directions=self.DIRECTIONS,
            )


def test_count_flagged_with_nonzero_overlap() -> None:
    # Flagged at <= 0.05: 0.0, 0.0, 0.03, 0.05; of these, 0.03 and 0.05 overlap.
    diagnostics = event_diagnostics(hpd=np.array([0.0, 0.0, 0.03, 0.05, 0.5]))
    assert count_flagged_with_nonzero_overlap(diagnostics, 0.05) == 2


class TestComputeKlDivergenceComparison:
    def test_splits_events_at_the_identity_line(self) -> None:
        # Events 0-3 lower under Continuous-Fragmented, 4-5 higher, 6 tied.
        continuous = np.array([1.0, 2.0, 3.0, 4.0, 0.5, 1.5, 2.0])
        fragmented = np.array([0.5, 1.0, 1.0, 1.0, 9.0, 2.0, 2.0])
        comparison = compute_kl_divergence_comparison(
            event_diagnostics(kl=continuous), event_diagnostics(kl=fragmented)
        )
        assert comparison.lower_fraction == pytest.approx(4 / 7)
        assert comparison.lower_continuous_median == pytest.approx(2.5)
        assert comparison.lower_continuous_fragmented_median == pytest.approx(1.0)
        assert comparison.higher_fraction == pytest.approx(2 / 7)
        assert comparison.higher_continuous_median == pytest.approx(1.0)
        assert comparison.higher_continuous_fragmented_median == pytest.approx(5.5)
        assert comparison.higher_continuous_fragmented_max == pytest.approx(9.0)

    def test_rejects_mismatched_events(self) -> None:
        with pytest.raises(ValueError, match="same spike events"):
            compute_kl_divergence_comparison(
                event_diagnostics(kl=np.array([1.0, 2.0])),
                event_diagnostics(kl=np.array([1.0])),
            )

    @pytest.mark.parametrize(
        "fragmented",
        [np.array([0.5, 0.5]), np.array([3.0, 3.0]), np.array([1.0, 2.0])],
        ids=["all-lower", "all-higher", "all-tied"],
    )
    def test_rejects_an_empty_side(self, fragmented: np.ndarray) -> None:
        with pytest.raises(ValueError, match="Both sides"):
            compute_kl_divergence_comparison(
                event_diagnostics(kl=np.array([1.0, 2.0])), event_diagnostics(kl=fragmented)
            )


class TestFigure4Summary:
    @staticmethod
    def _summarize(thresholds: dict[str, float]) -> Figure4Summary:
        return summarize_figure04_diagnostics(
            event_diagnostics(hpd=np.array([0.01, 0.20])),
            event_diagnostics(hpd=np.array([0.40, 0.20])),
            n_units=2,
            thresholds=thresholds,
            metric_directions={"hpd_overlap": "below"},
        )

    def test_computes_typed_means_and_flag_counts(self) -> None:
        summary = self._summarize({"hpd_overlap": 0.05})

        assert isinstance(summary, Figure4Summary)
        assert summary.continuous.hpd_overlap == pytest.approx(0.105)
        assert summary.continuous_fragmented.hpd_overlap == pytest.approx(0.30)
        assert len(summary.flag_confusions) == 1
        confusion = summary.flag_confusions[0]
        assert (
            confusion.n,
            confusion.both,
            confusion.rescued,
            confusion.newly_flagged,
            confusion.neither,
        ) == (
            2,
            0,
            1,
            0,
            1,
        )

    def test_formatter_does_not_recompute(self) -> None:
        summary = self._summarize({"hpd_overlap": 0.05})
        text = format_figure04_summary(summary)
        assert "Continuous:" in text
        assert "hpd_overlap: 0.1050" in text
        assert "rescued=1" in text

    def test_requires_threshold_for_every_direction(self) -> None:
        with pytest.raises(ValueError, match="without thresholds"):
            self._summarize({})


def _decode_payload() -> dict[str, object]:
    return _cache_payload(_synthetic_analysis_results(), _FIGURE04_DECODE_PAYLOAD_KEYS)


def _diagnostics_payload() -> dict[str, object]:
    return _cache_payload(_synthetic_analysis_results(), _FIGURE04_DIAGNOSTICS_PAYLOAD_KEYS)


class TestFigure4AnalysisResults:
    def test_cache_payload_round_trip_through_field_named_keys(self) -> None:
        decode = _synthetic_analysis_results()
        payload = _cache_payload(decode)
        # The serialized keys are the field names.
        assert "continuous_fragmented_results" in payload
        assert "continuous_fragmented_diagnostics" in payload
        rebuilt = Figure4AnalysisResults.from_cache_payload(payload)
        assert rebuilt.continuous_fragmented_results is decode.continuous_fragmented_results
        assert rebuilt.continuous_fragmented_diagnostics is decode.continuous_fragmented_diagnostics
        np.testing.assert_array_equal(rebuilt.spike_counts, decode.spike_counts)

    def test_cache_keys_are_the_field_names(self) -> None:
        # Guard against drift between the on-disk key list (owned by
        # figure04_cache) and the container's fields.
        assert {field.name for field in dataclasses.fields(Figure4AnalysisResults)} == set(
            _FIGURE04_DECODE_AND_DIAGNOSTICS_PAYLOAD_KEYS
        )

    def test_from_cache_payload_keeps_the_memory_mapped_spike_counts(self, tmp_path: Path) -> None:
        # The session-sized counts arrive memory-mapped; loading must not copy them.
        np.save(tmp_path / "counts.npy", np.zeros((8, 2), dtype=np.int64))
        spike_counts = np.load(tmp_path / "counts.npy", mmap_mode="r")
        payload = _cache_payload(_synthetic_analysis_results())
        payload["spike_counts"] = spike_counts
        rebuilt = Figure4AnalysisResults.from_cache_payload(payload)
        assert np.shares_memory(rebuilt.spike_counts, spike_counts)

    def test_from_cache_payload_rejects_missing_key(self) -> None:
        payload = _cache_payload(_synthetic_analysis_results())
        del payload["spike_counts"]
        with pytest.raises(ValueError, match="missing keys"):
            Figure4AnalysisResults.from_cache_payload(payload)

    def test_from_cache_payload_rejects_wrong_type(self) -> None:
        payload = _cache_payload(_synthetic_analysis_results())
        payload["continuous_results"] = np.zeros(3)  # not an xr.Dataset
        with pytest.raises(TypeError, match="xr.Dataset"):
            Figure4AnalysisResults.from_cache_payload(payload)

    def test_rejects_incompatible_shapes(self) -> None:
        with pytest.raises(ValueError, match="place_field_peaks"):
            Figure4AnalysisResults(
                continuous_results=xr.Dataset(),
                continuous_fragmented_results=xr.Dataset(),
                continuous_diagnostics=event_diagnostics(hpd=np.array([0.5])),
                continuous_fragmented_diagnostics=event_diagnostics(hpd=np.array([0.5])),
                spike_counts=np.zeros((8, 2), dtype=np.int64),
                place_field_peaks=np.zeros(3),  # should be (2,)
                diagnostic_place_fields=np.zeros((2, 4)),
                diagnostic_position_bins=np.arange(4.0),
            )

    def test_arrays_are_read_only(self) -> None:
        decode = _synthetic_analysis_results()
        with pytest.raises(ValueError, match="read-only|write"):
            decode.spike_counts[0, 0] = 1

    def test_copies_input_arrays_and_leaves_caller_writable(self) -> None:
        # Freezing must not reach back into a caller-owned array, and the stored
        # copy must be isolated from later mutation of that array.
        spike_counts = np.zeros((8, 2), dtype=np.int64)
        decode = Figure4AnalysisResults(
            continuous_results=_ds(np.zeros(8)),
            continuous_fragmented_results=_ds(np.zeros(8)),
            continuous_diagnostics=event_diagnostics(hpd=np.array([0.5])),
            continuous_fragmented_diagnostics=event_diagnostics(hpd=np.array([0.5])),
            spike_counts=spike_counts,
            place_field_peaks=np.zeros(2),
            diagnostic_place_fields=np.zeros((2, 4)),
            diagnostic_position_bins=np.arange(4.0),
        )
        assert spike_counts.flags.writeable  # caller's array untouched
        spike_counts[0, 0] = 7
        assert decode.spike_counts[0, 0] == 0  # stored copy isolated
        assert not decode.spike_counts.flags.writeable

    def test_keeps_read_only_memmap_without_copying(self, tmp_path: Path) -> None:
        # The memory-mapped cache arrives read-only; it is used in place.
        np.save(tmp_path / "counts.npy", np.zeros((8, 2), dtype=np.int64))
        spike_counts = np.load(tmp_path / "counts.npy", mmap_mode="r")
        decode = dataclasses.replace(_synthetic_analysis_results(), spike_counts=spike_counts)
        assert np.shares_memory(decode.spike_counts, spike_counts)
        assert not decode.spike_counts.flags.writeable

    def test_copies_read_only_view_of_a_writeable_array(self) -> None:
        # A read-only view can still change through its writeable base.
        base = np.zeros((8, 2), dtype=np.int64)
        view = base.view()
        view.setflags(write=False)
        decode = dataclasses.replace(_synthetic_analysis_results(), spike_counts=view)
        base[0, 0] = 7
        assert decode.spike_counts[0, 0] == 0

    def test_converts_a_read_only_memmap_of_another_dtype(self, tmp_path: Path) -> None:
        np.save(tmp_path / "counts.npy", np.ones((8, 2), dtype=np.int32))
        spike_counts = np.load(tmp_path / "counts.npy", mmap_mode="r")
        decode = dataclasses.replace(_synthetic_analysis_results(), spike_counts=spike_counts)
        assert decode.spike_counts.dtype == np.int64
        assert not np.shares_memory(decode.spike_counts, spike_counts)
        assert not decode.spike_counts.flags.writeable

    def test_rejects_dataset_timeline_mismatch(self) -> None:
        with pytest.raises(ValueError, match="decode timelines must match"):
            Figure4AnalysisResults(
                continuous_results=_ds(np.zeros(3)),
                continuous_fragmented_results=_ds(np.zeros(8)),
                continuous_diagnostics=event_diagnostics(hpd=np.array([0.5])),
                continuous_fragmented_diagnostics=event_diagnostics(hpd=np.array([0.5])),
                spike_counts=np.zeros((8, 2), dtype=np.int64),  # 8 != dataset's 3
                place_field_peaks=np.zeros(2),
                diagnostic_place_fields=np.zeros((2, 4)),
                diagnostic_position_bins=np.arange(4.0),
            )

    def test_rejects_dataset_without_time_coordinate(self) -> None:
        # A time *dimension* is not enough; compose_figure04 reads the ``time``
        # coordinate, so require it at construction.
        with pytest.raises(ValueError, match="must carry a 'time' coordinate"):
            Figure4AnalysisResults(
                continuous_results=xr.Dataset({"acausal_posterior": ("time", np.zeros(8))}),
                continuous_fragmented_results=_ds(np.zeros(8)),
                continuous_diagnostics=event_diagnostics(hpd=np.array([0.5])),
                continuous_fragmented_diagnostics=event_diagnostics(hpd=np.array([0.5])),
                spike_counts=np.zeros((8, 2), dtype=np.int64),
                place_field_peaks=np.zeros(2),
                diagnostic_place_fields=np.zeros((2, 4)),
                diagnostic_position_bins=np.arange(4.0),
            )

    def test_rejects_mismatched_time_coordinates(self) -> None:
        # Same length, different coordinate values: the two decoders were run on
        # different windows. A length-only check would miss this.
        with pytest.raises(ValueError, match="different 'time' coordinates"):
            Figure4AnalysisResults(
                continuous_results=xr.Dataset(
                    {"acausal_posterior": ("time", np.zeros(8))},
                    coords={"time": np.arange(8, dtype=float)},
                ),
                continuous_fragmented_results=xr.Dataset(
                    {"acausal_posterior": ("time", np.zeros(8))},
                    coords={"time": np.arange(8, dtype=float) + 100.0},
                ),
                continuous_diagnostics=event_diagnostics(hpd=np.array([0.5])),
                continuous_fragmented_diagnostics=event_diagnostics(hpd=np.array([0.5])),
                spike_counts=np.zeros((8, 2), dtype=np.int64),
                place_field_peaks=np.zeros(2),
                diagnostic_place_fields=np.zeros((2, 4)),
                diagnostic_position_bins=np.arange(4.0),
            )

    def test_rejects_diagnostic_event_index_out_of_range(self) -> None:
        with pytest.raises(ValueError, match="event_time_ind falls outside"):
            Figure4AnalysisResults(
                continuous_results=_ds(np.zeros(8)),
                continuous_fragmented_results=_ds(np.zeros(8)),
                continuous_diagnostics=event_diagnostics(
                    hpd=np.full(1, 0.5), time_ind=[99], cell_ind=[0]
                ),  # 99 >= n_time 8
                continuous_fragmented_diagnostics=event_diagnostics(hpd=np.array([0.5])),
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
        analysis_results: Figure4AnalysisResults | None = None,
    ) -> Figure4RenderData:
        n_time = time.shape[0]
        return Figure4RenderData(
            recording=_synthetic_recording(),
            time=time,
            linear_position=np.zeros(n_time),
            analysis_results=analysis_results or _synthetic_analysis_results(),
            cache_provenance=synthetic_cache_provenance(),
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
            "load_figure04_input",
            lambda *a, **k: _synthetic_recording(),
        )
        calls = {"fit": 0, "diagnostics": 0}

        def fake_fit(*a: object, **k: object) -> dict[str, object]:
            calls["fit"] += 1
            return _decode_payload()

        def fake_diagnostics(*a: object, **k: object) -> dict[str, object]:
            calls["diagnostics"] += 1
            return _diagnostics_payload()

        monkeypatch.setattr(figure04_workflow, "fit_and_decode", fake_fit)
        monkeypatch.setattr(figure04_workflow, "_compute_diagnostics_payload", fake_diagnostics)

        paths = _paths_with_input(tmp_path, "synthetic_epoch")
        config = Figure4Config()

        prepare_figure04_render_data(config, paths, use_cache=True)
        assert calls == {"fit": 1, "diagnostics": 1}
        assert paths.decode_cache_path.exists()
        assert paths.diagnostics_cache_path.exists()

        prepare_figure04_render_data(config, paths, use_cache=True)
        assert calls == {"fit": 1, "diagnostics": 1}  # both caches hit

        # A diagnostics-only change recomputes the diagnostics, not the fit.
        diag_changed = dataclasses.replace(
            config, diagnostics=dataclasses.replace(config.diagnostics, hpd_coverage=0.8)
        )
        prepare_figure04_render_data(diag_changed, paths, use_cache=True)
        assert calls == {"fit": 1, "diagnostics": 2}

        # A decode-affecting change refits and (necessarily) recomputes diagnostics.
        changed = dataclasses.replace(
            config,
            package_defaults=dataclasses.replace(
                config.package_defaults, movement_var=config.package_defaults.movement_var + 1.0
            ),
        )
        prepare_figure04_render_data(changed, paths, use_cache=True)
        assert calls == {"fit": 2, "diagnostics": 3}

        # use_cache=False refits and recomputes even with matching caches.
        prepare_figure04_render_data(changed, paths, use_cache=False)
        assert calls == {"fit": 3, "diagnostics": 4}

    def test_configured_coverage_reaches_each_decoders_diagnostics(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The recorded ``hpd_coverage`` is the one the diagnostics are computed at."""
        coverages: list[float] = []

        def record_coverage(*args: object, coverage: float, **kwargs: object) -> object:
            coverages.append(coverage)
            return object()

        monkeypatch.setattr(figure04_workflow, "compute_results_diagnostics", record_coverage)
        monkeypatch.setattr(figure04_workflow, "_cast_dataset", lambda value: value)
        config = Figure4Config()
        config = dataclasses.replace(
            config, diagnostics=dataclasses.replace(config.diagnostics, hpd_coverage=0.8)
        )

        figure04_workflow._compute_diagnostics_payload(
            _decode_payload(),
            recording=_synthetic_recording(),
            time=np.arange(8, dtype=float),
            diagnostics_config=config.diagnostics,
        )

        assert coverages == [0.8, 0.8]

    def test_cache_invalidates_on_dependency_change(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setattr(
            figure04_workflow,
            "load_figure04_input",
            lambda *a, **k: _synthetic_recording(),
        )
        calls = {"n": 0}

        def fake_fit(*a: object, **k: object) -> dict[str, object]:
            calls["n"] += 1
            return _decode_payload()

        monkeypatch.setattr(figure04_workflow, "fit_and_decode", fake_fit)
        monkeypatch.setattr(
            figure04_workflow,
            "_compute_diagnostics_payload",
            lambda *a, **k: _cache_payload(
                _synthetic_analysis_results(), _FIGURE04_DIAGNOSTICS_PAYLOAD_KEYS
            ),
        )
        monkeypatch.setattr(
            figure04_cache, "_installed_non_local_detector_version", lambda: "1.0.0"
        )

        paths = _paths_with_input(tmp_path, "synthetic_epoch")
        config = Figure4Config()
        prepare_figure04_render_data(config, paths, use_cache=True)
        assert calls["n"] == 1

        monkeypatch.setattr(
            figure04_cache, "_installed_non_local_detector_version", lambda: "2.0.0"
        )
        prepare_figure04_render_data(config, paths, use_cache=True)
        assert calls["n"] == 2

    def test_uses_injected_paths_and_maps_serialized_keys(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        seen: dict[str, object] = {}

        def spy_load(data_path: object, animal_date_epoch: object) -> NeuralRecordingData:
            seen["data_path"] = data_path
            seen["animal_date_epoch"] = animal_date_epoch
            return _synthetic_recording()

        monkeypatch.setattr(figure04_workflow, "load_figure04_input", spy_load)
        monkeypatch.setattr(
            figure04_workflow,
            "fit_and_decode",
            lambda *a, **k: _cache_payload(
                _synthetic_analysis_results(), _FIGURE04_DECODE_PAYLOAD_KEYS
            ),
        )
        monkeypatch.setattr(
            figure04_workflow,
            "_compute_diagnostics_payload",
            lambda *a, **k: _cache_payload(
                _synthetic_analysis_results(), _FIGURE04_DIAGNOSTICS_PAYLOAD_KEYS
            ),
        )

        injected = _paths_with_input(tmp_path, "injected_epoch")
        render_data = prepare_figure04_render_data(Figure4Config(), injected, use_cache=False)

        assert seen == {"data_path": tmp_path, "animal_date_epoch": "injected_epoch"}
        assert injected.decode_cache_path.exists()
        # The progress messages name the caches actually written.
        printed = capsys.readouterr().out
        assert str(injected.decode_cache_path) in printed
        assert str(injected.diagnostics_cache_path) in printed
        # The recording and typed decode results are threaded through by attribute.
        assert render_data.recording.spike_times[0].shape == (2,)
        np.testing.assert_array_equal(
            render_data.analysis_results.continuous_fragmented_results[
                "acausal_posterior"
            ].to_numpy(),
            np.ones(8),
        )
        assert render_data.analysis_results.spike_counts.shape == (8, 2)
        assert render_data.time.shape == (8,)


def _linear_track_recording() -> NeuralRecordingData:
    """A small recording, with two units, that a real decoder can fit."""
    rng = np.random.default_rng(0)
    n_time = 400
    time = np.arange(n_time) / 500.0
    x = np.linspace(0.0, 40.0, n_time)
    track_graph = nx.Graph()
    track_graph.add_node(0, pos=(0.0, 0.0))
    track_graph.add_node(1, pos=(40.0, 0.0))
    track_graph.add_edge(0, 1, distance=40.0, edge_id=0)
    return NeuralRecordingData(
        position_info=pd.DataFrame(
            {"head_position_x": x, "head_position_y": np.zeros(n_time), "linear_position": x},
            index=time,
        ),
        spike_times=(
            np.sort(rng.uniform(0.0, time[-1], 30)),
            np.sort(rng.uniform(0.0, time[-1], 20)),
        ),
        track_graph=track_graph,
        linear_edge_order=((0, 1),),
        linear_edge_spacing=0.0,
    )


def test_fit_and_decode_payload_feeds_the_caches_and_results() -> None:
    """A real fit and decode of a small synthetic linear-track recording returns
    exactly the decode-cache keys, and its diagnostics build valid results."""
    pytest.importorskip("non_local_detector")
    recording = _linear_track_recording()
    time = figure04_fit.decode_time(recording)
    n_time = time.size
    config = Figure4Config()

    decode_payload = figure04_fit.fit_and_decode(
        recording,
        decoder_config=config.decoder,
        execution_config=config.execution,
        package_defaults=config.package_defaults,
    )

    assert set(decode_payload) == set(figure04_cache._FIGURE04_DECODE_PAYLOAD_KEYS)
    np.testing.assert_array_equal(figure04_fit.decode_time(recording), time)
    diagnostics_payload = figure04_workflow._compute_diagnostics_payload(
        decode_payload,
        recording=recording,
        time=figure04_fit.decode_time(recording),
        diagnostics_config=config.diagnostics,
    )
    results = Figure4AnalysisResults.from_cache_payload({**decode_payload, **diagnostics_payload})
    assert results.spike_counts.shape == (n_time, 2)
    # Every spike lies inside [time[0], time[-1]], so each is counted once.
    assert results.spike_counts.sum() == 50


def test_decoder_prediction_at_a_row_excludes_that_rows_spikes() -> None:
    """The diagnostics score each spike against ``predictive_posterior[t]``, so that
    row must not already contain the spike. Spikes added to row 200 of a fitted
    decoder leave rows through 200 unchanged and first move row 201."""
    pytest.importorskip("non_local_detector")
    recording = _linear_track_recording()
    config = Figure4Config()
    time = figure04_fit.decode_time(recording)
    spike_times = list(recording.spike_times)
    environment = create_decoder_environment(
        track_graph=recording.track_graph,
        edge_order=list(recording.linear_edge_order),
        edge_spacing=recording.linear_edge_spacing,
        place_bin_size=config.decoder.position_bin_size_cm,
    )
    models = fit_decoder_models(
        position=recording.position_info[list(HEAD_POSITION_COLUMNS)].to_numpy(dtype=np.float64),
        spike_times=spike_times,
        time=time,
        environment=environment,
        decoder_config=config.decoder,
        execution_config=config.execution,
    )
    row = 200
    with_added_spikes = [
        np.sort(np.append(spike_times[0], np.full(3, time[row] + 0.0005))),
        spike_times[1],
    ]

    for model in models:
        unchanged, changed = (
            np.asarray(
                model.predict(
                    spike_times=spikes, time=time, return_outputs=[DECODER_PREDICTIVE_VAR]
                )[DECODER_PREDICTIVE_VAR]
            )
            for spikes in (spike_times, with_added_spikes)
        )
        np.testing.assert_array_equal(unchanged[: row + 1], changed[: row + 1])
        assert np.nanmax(np.abs(unchanged[row + 1] - changed[row + 1])) > 1e-3


def test_reported_statistics_come_from_the_render_data() -> None:
    render_data = synthetic_render_data()
    analysis = render_data.analysis_results
    thresholds = {"hpd_overlap": 0.5, "kl_divergence": 1.0, "predictive_pvalue": 0.05}
    statistics = compute_figure04_reported_statistics(
        render_data, thresholds, METRIC_FLAG_DIRECTIONS, slice(0, render_data.time.size)
    )
    assert statistics.summary.n_units == analysis.spike_counts.shape[1]
    assert statistics.n_units_with_spikes == count_units_with_spikes(
        analysis.continuous_diagnostics
    )
    assert statistics.epoch_duration_s == render_data.time[-1] - render_data.time[0]
    assert (
        statistics.detail_window_comparison.n_events
        == analysis.continuous_diagnostics.event_time_ind.size
    )
    assert 0.0 < statistics.kl_divergence_comparison.lower_fraction < 1.0
    assert set(statistics.immobility_comparison.moving_flag_rates) == set(METRIC_FLAG_DIRECTIONS)


def test_reported_statistics_keep_each_decoder_on_its_own_side() -> None:
    """Rescued and newly flagged, and both sides of the KL identity line, are
    computed with Continuous as the reference. Swapping the two decoders in any
    of the three comparisons exchanges rescued with newly flagged, so every count
    here is asymmetric and written out by hand."""
    render_data = synthetic_render_data()
    n_events = 6
    time_ind = np.arange(n_events)
    cell_ind = np.array([0, 0, 1, 1, 2, 2])
    # Flagged at hpd <= 0.5: Continuous on events 0-3, Continuous-Fragmented on 3-5.
    continuous = event_diagnostics(
        hpd=np.array([0.1, 0.2, 0.3, 0.4, 0.9, 0.9]),
        kl=np.array([9.0, 8.0, 7.0, 6.0, 1.0, 2.0]),
        pvalue=np.array([0.01, 0.01, 0.01, 0.01, 0.5, 0.5]),
        time_ind=time_ind,
        cell_ind=cell_ind,
    )
    fragmented = event_diagnostics(
        hpd=np.array([0.9, 0.9, 0.9, 0.4, 0.2, 0.3]),
        kl=np.array([1.0, 2.0, 3.0, 4.0, 20.0, 30.0]),
        pvalue=np.array([0.5, 0.5, 0.5, 0.01, 0.6, 0.7]),
        time_ind=time_ind,
        cell_ind=cell_ind,
    )
    render_data = dataclasses.replace(
        render_data,
        analysis_results=dataclasses.replace(
            render_data.analysis_results,
            continuous_diagnostics=continuous,
            continuous_fragmented_diagnostics=fragmented,
        ),
    )
    thresholds = {"hpd_overlap": 0.5, "kl_divergence": 100.0, "predictive_pvalue": 0.05}

    statistics = compute_figure04_reported_statistics(
        render_data, thresholds, METRIC_FLAG_DIRECTIONS, slice(0, n_events)
    )

    hpd = next(c for c in statistics.summary.flag_confusions if c.metric == "hpd_overlap")
    assert (hpd.rescued, hpd.both, hpd.newly_flagged) == (3, 1, 2)
    assert statistics.summary.continuous.hpd_overlap == pytest.approx(2.8 / 6)
    assert statistics.summary.continuous_fragmented.hpd_overlap == pytest.approx(3.6 / 6)

    detail = statistics.detail_window_comparison.hpd_overlap
    assert (detail.rescued, detail.both, detail.newly_flagged) == (3, 1, 2)
    detail_pvalue = statistics.detail_window_comparison.predictive_pvalue
    assert (detail_pvalue.rescued, detail_pvalue.both, detail_pvalue.newly_flagged) == (3, 1, 0)
    # Higher under Continuous-Fragmented on every event but the tied event 3.
    assert statistics.detail_window_comparison.predictive_pvalue_higher == 5
    assert statistics.detail_window_comparison.continuous_kl_median == pytest.approx(6.5)
    assert statistics.detail_window_comparison.continuous_fragmented_kl_median == pytest.approx(3.5)

    kl = statistics.kl_divergence_comparison
    assert kl.lower_fraction == pytest.approx(4 / 6)
    assert kl.lower_continuous_median == pytest.approx(7.5)
    assert kl.lower_continuous_fragmented_median == pytest.approx(2.5)
    assert kl.higher_fraction == pytest.approx(2 / 6)
    assert kl.higher_continuous_median == pytest.approx(1.5)
    assert kl.higher_continuous_fragmented_median == pytest.approx(25.0)
    assert kl.higher_continuous_fragmented_max == pytest.approx(30.0)

    # The synthetic animal is immobile at even samples: events 0, 2, 4 and moving at 1, 3, 5.
    # Continuous flags HPD on events 0-3, so two of three in each state.
    immobility = statistics.immobility_comparison
    assert immobility.immobile_flag_rates["hpd_overlap"] == pytest.approx(2 / 3)
    assert immobility.moving_flag_rates["hpd_overlap"] == pytest.approx(2 / 3)
    assert immobility.immobile_kl_median == pytest.approx(7.0)
    assert immobility.moving_kl_median == pytest.approx(6.0)
    # The Continuous model flags events 0-3, whose overlaps are all nonzero.
    assert statistics.hpd_flagged_nonzero_overlap == 4


def test_count_units_with_spikes_counts_each_firing_unit_once() -> None:
    """Repeated spikes count one unit, and units absent from the events count none."""
    diagnostics = event_diagnostics(time_ind=[0, 1, 2, 3], cell_ind=[0, 0, 2, 5])

    assert count_units_with_spikes(diagnostics) == 3
