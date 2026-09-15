"""Tests for the full-recording Figure-4 supplement analyses and figure."""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any, cast

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")  # noqa: E402

import networkx as nx  # noqa: E402
import pandas as pd  # noqa: E402
import xarray as xr  # noqa: E402

from statespacecheck_paper.figure04_broadening import (  # noqa: E402
    BroadeningResults,
    Figure4BroadeningConfig,
    _constant_frequency_rank_pvalues,
    compute_rate_and_behavior_association,
    compute_region_size_and_broadening,
)
from statespacecheck_paper.figure04_diagnostics import compute_results_diagnostics  # noqa: E402
from statespacecheck_paper.figure04_supplement_plotting import (  # noqa: E402
    compose_figure04_supplement,
)
from statespacecheck_paper.figure04_workflow import (  # noqa: E402
    Figure4DecodeResults,
    Figure4RenderData,
)
from statespacecheck_paper.load_local_data import NeuralRecordingData  # noqa: E402

_N_TIME, _N_CELLS, _N_POS = 60, 5, 16
_DT = 0.002


def _results(rng: np.random.Generator, states: tuple[str, ...], concentration: float) -> xr.Dataset:
    """Decoder-style dataset with a per-state predictive posterior on ``_N_POS`` bins."""
    pos = np.linspace(0.0, 100.0, _N_POS)
    state_bins = pd.MultiIndex.from_product([list(states), pos], names=["state", "position"])
    time = np.arange(_N_TIME, dtype=float) * _DT
    # Concentration below 1 gives peaked predictions, above 1 diffuse ones.
    predictive = rng.dirichlet(np.full(len(state_bins), concentration), size=_N_TIME)
    data = xr.DataArray(
        predictive, dims=("time", "state_bins"), coords={"time": time, "state_bins": state_bins}
    )
    return xr.Dataset({"predictive_posterior": data, "log_likelihood": np.log(data)})


def _recording(head_speed: np.ndarray) -> NeuralRecordingData:
    track_graph = nx.Graph()
    for i in range(2):
        track_graph.add_node(i, pos=(float(i * 100), 0.0))
    track_graph.add_edge(0, 1, distance=100.0)
    time = np.arange(_N_TIME, dtype=float) * _DT
    position_info = pd.DataFrame(
        {
            "head_position_x": np.linspace(0.0, 100.0, _N_TIME),
            "head_position_y": np.zeros(_N_TIME),
            "linear_position": np.linspace(0.0, 100.0, _N_TIME),
            "head_speed": head_speed,
        },
        index=time,
    )
    return NeuralRecordingData(
        position_info=position_info,
        spike_times=tuple(np.empty(0) for _ in range(_N_CELLS)),
        track_graph=track_graph,
        linear_edge_order=((0, 1),),
        linear_edge_spacing=0.0,
    )


@pytest.fixture(scope="module")
def render_data() -> Figure4RenderData:
    """Synthetic render data whose cached diagnostics match its predictions.

    The Continuous model is peaked and the Continuous--Fragmented model diffuse,
    so the fixture contains rescued events; one unit never fires so the
    zero-event bookkeeping is exercised.
    """
    rng = np.random.default_rng(7)
    time = np.arange(_N_TIME, dtype=float) * _DT
    # Narrow Gaussian fields (about one bin wide) so a peaked prediction that
    # sits elsewhere on the track gets HPD overlap 0 and is flagged.
    bins = np.linspace(0.0, 100.0, _N_POS)
    centers = np.linspace(10.0, 90.0, _N_CELLS)
    place_fields = 0.5 * np.exp(-0.5 * ((bins[None, :] - centers[:, None]) / 4.0) ** 2) + 1e-4
    spike_counts = rng.poisson(0.6, (_N_TIME, _N_CELLS)).astype(np.int64)
    spike_counts[:, -1] = 0
    continuous = _results(rng, ("Continuous",), 0.05)
    fragmented = _results(rng, ("Continuous", "Fragmented"), 5.0)
    diagnostics = {
        name: compute_results_diagnostics(results, place_fields, spike_counts, time)
        for name, results in (("cont", continuous), ("frag", fragmented))
    }
    decode = Figure4DecodeResults(
        continuous_results=continuous,
        continuous_fragmented_results=fragmented,
        continuous_diagnostics=diagnostics["cont"],
        continuous_fragmented_diagnostics=diagnostics["frag"],
        spike_counts=spike_counts,
        place_field_peaks=np.linspace(0.0, 100.0, _N_POS)[np.argmax(place_fields, axis=1)],
        diagnostic_place_fields=place_fields,
        diagnostic_position_bins=np.linspace(0.0, 100.0, _N_POS),
    )
    head_speed = np.where(np.arange(_N_TIME) < _N_TIME // 2, 1.0, 20.0)
    return Figure4RenderData(
        recording=_recording(head_speed),
        time=time,
        head_position=np.column_stack([np.linspace(0.0, 100.0, _N_TIME), np.zeros(_N_TIME)]),
        linear_position=np.linspace(0.0, 100.0, _N_TIME),
        decode_results=decode,
    )


@pytest.fixture(scope="module")
def config() -> Figure4BroadeningConfig:
    # A small chunk forces the chunked path to cross chunk boundaries.
    return Figure4BroadeningConfig(event_chunk=7, speed_cutoff_cm_s=4.0)


def _by_coverage(broadening: BroadeningResults) -> dict[str, dict[str, Any]]:
    return cast(dict[str, dict[str, Any]], broadening.summary["by_coverage"])


@pytest.fixture(scope="module")
def broadening(
    render_data: Figure4RenderData, config: Figure4BroadeningConfig
) -> BroadeningResults:
    return compute_region_size_and_broadening(render_data, config)


class TestRegionSizeAndBroadening:
    def test_counts_reproduce_the_cached_diagnostics(
        self,
        render_data: Figure4RenderData,
        broadening: BroadeningResults,
        config: Figure4BroadeningConfig,
    ) -> None:
        decode = render_data.decode_results
        n_events = decode.continuous_diagnostics.event_time_ind.size
        record = _by_coverage(broadening)["0.95"]
        assert record["n_events"] == n_events
        assert record["n_position_bins"] == _N_POS
        expected_cont = int(
            np.sum(decode.continuous_diagnostics.event_hpd_overlap <= config.hpd_flag_threshold)
        )
        expected_frag = int(
            np.sum(
                decode.continuous_fragmented_diagnostics.event_hpd_overlap
                <= config.hpd_flag_threshold
            )
        )
        assert record["continuous"]["n_flagged"] == expected_cont
        assert record["continuous_fragmented"]["n_flagged"] == expected_frag
        assert expected_cont > 0, "fixture must contain Continuous flags"
        arrays = broadening.arrays_by_coverage["0.95"]
        assert record["n_rescued"] == int(
            np.sum(arrays.flag_continuous & ~arrays.flag_continuous_fragmented)
        )
        assert record["n_rescued"] + record["n_newly_flagged"] <= n_events

    def test_region_sizes_grow_with_coverage(self, broadening: BroadeningResults) -> None:
        sizes = [broadening.arrays_by_coverage[k].size_continuous for k in ("0.5", "0.8", "0.95")]
        assert np.all(sizes[0] <= sizes[1]) and np.all(sizes[1] <= sizes[2])
        assert np.all(sizes[2] <= _N_POS) and np.all(sizes[0] >= 1)

    def test_uniform_mixture_limits(
        self, broadening: BroadeningResults, config: Figure4BroadeningConfig
    ) -> None:
        for record in _by_coverage(broadening).values():
            mixtures = record["uniform_mixture"]
            full = mixtures["1"]
            assert full["fraction_original_flags_removed"] == 1.0
            assert full["n_flags_under_mixture"] == 0
            assert full["n_new_flags"] == 0
            coverage = record["coverage"]
            for weight in config.uniform_weights:
                entry = mixtures[f"{weight:g}"]
                assert entry["lower_bound_region_fraction"] == pytest.approx(
                    max(0.0, (coverage - 1.0 + weight) / weight)
                )
                assert (
                    entry["n_continuous_flags"]
                    - entry["n_original_flags_removed"]
                    + entry["n_new_flags"]
                    == entry["n_flags_under_mixture"]
                )

    def test_misaligned_events_are_rejected(self, render_data: Figure4RenderData) -> None:
        decode = render_data.decode_results
        frag = decode.continuous_fragmented_diagnostics
        shifted = dataclasses.replace(frag, event_time_ind=frag.event_time_ind[::-1].copy())
        bad = dataclasses.replace(
            render_data,
            decode_results=dataclasses.replace(decode, continuous_fragmented_diagnostics=shifted),
        )
        with pytest.raises(ValueError, match="identical event indices"):
            compute_region_size_and_broadening(bad, Figure4BroadeningConfig())


class TestConstantFrequencyRankPvalues:
    def test_matches_loop_definition_with_ties(self) -> None:
        # Units 0 and 1 tie on frequency; unit 3 has no events.
        event_cell_ind = np.array([0, 0, 1, 1, 2, 2, 2, 4], dtype=np.intp)
        n_units = 5
        counts = np.bincount(event_cell_ind, minlength=n_units)
        q = counts / counts.sum()
        expected = np.array([q[q <= q[c]].sum() for c in event_cell_ind])
        np.testing.assert_allclose(
            _constant_frequency_rank_pvalues(event_cell_ind, n_units), expected
        )

    def test_most_frequent_unit_has_p_one(self) -> None:
        rng = np.random.default_rng(0)
        event_cell_ind = rng.integers(0, 6, 200).astype(np.intp)
        p = _constant_frequency_rank_pvalues(event_cell_ind, 6)
        most = np.argmax(np.bincount(event_cell_ind, minlength=6))
        assert np.all(p[event_cell_ind == most] == 1.0)
        assert np.all((p > 0.0) & (p <= 1.0))


class TestRateAndBehaviorAssociation:
    def test_partitions_and_unit_bookkeeping(
        self, render_data: Figure4RenderData, config: Figure4BroadeningConfig
    ) -> None:
        summary = cast(dict[str, Any], compute_rate_and_behavior_association(render_data, config))
        n_events = render_data.decode_results.continuous_diagnostics.event_time_ind.size
        assert summary["n_events"] == n_events
        assert summary["n_units"] == _N_CELLS
        assert summary["n_zero_event_units"] == 1
        assert summary["n_active_units"] == _N_CELLS - 1
        behavior = summary["behavior"]
        assert behavior["immobile"]["n_events"] + behavior["moving"]["n_events"] == n_events
        groups = summary["rate_groups"]
        assert sum(g["n_events"] for g in groups) == n_events
        assert sum(g["n_units"] for g in groups) == summary["n_active_units"]
        baseline = summary["constant_frequency_baseline"]
        assert 0.0 <= baseline["flag_fraction"] <= 1.0
        per_unit = summary["per_unit"]
        assert len(per_unit["rate_hz"]) == _N_CELLS
        assert per_unit["rate_hz"][-1] == 0.0


def test_compose_supplement_produces_five_labeled_panels(
    render_data: Figure4RenderData,
    broadening: BroadeningResults,
    config: Figure4BroadeningConfig,
    tmp_path: Path,
) -> None:
    rate_summary = compute_rate_and_behavior_association(render_data, config)
    fig = compose_figure04_supplement(broadening, rate_summary, config)
    try:
        assert len(fig.axes) == 5
        labels = {t.get_text() for ax in fig.axes for t in ax.texts}
        assert {"a", "b", "c", "d", "e"} <= labels
        fig.savefig(tmp_path / "supplement.png", dpi=60)
        assert (tmp_path / "supplement.png").stat().st_size > 0
    finally:
        matplotlib.pyplot.close(fig)
