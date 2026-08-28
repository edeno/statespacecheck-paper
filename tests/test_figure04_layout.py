"""Tests for the Figure-4 layout (event-time shift + composition contract)."""

from __future__ import annotations

import dataclasses

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")  # noqa: E402

import networkx as nx  # noqa: E402
import pandas as pd  # noqa: E402
import xarray as xr  # noqa: E402

from statespacecheck_paper.diagnostics import SpikeEventDiagnostics  # noqa: E402
from statespacecheck_paper.figure04_layout import (  # noqa: E402
    Figure4Composition,
    Figure4DetailWindow,
    _shift_diagnostic_event_times,
    compose_figure04,
)
from statespacecheck_paper.figure04_workflow import (  # noqa: E402
    Figure4DecodeResults,
    Figure4RenderData,
)
from statespacecheck_paper.load_local_data import NeuralRecordingData  # noqa: E402


def _make_per_cell_diagnostics(
    *, event_time: np.ndarray | None, event_hpd_overlap: np.ndarray
) -> SpikeEventDiagnostics:
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
        event_time=event_time,
    )


class TestShiftDiagnosticEventTimes:
    def test_subtracts_offset(self) -> None:
        """Per-spike event times must be relative to the same time base as the
        figure axis — otherwise scatter points slide off the panels."""
        diagnostics = _make_per_cell_diagnostics(
            event_time=np.array([101.0, 101.5]),
            event_hpd_overlap=np.array([0.25, 0.75]),
        )
        shifted = _shift_diagnostic_event_times(diagnostics, 100.0)
        np.testing.assert_allclose(shifted.event_time, [1.0, 1.5])
        # Original instance not mutated (frozen + write-protected).
        np.testing.assert_allclose(diagnostics.event_time, [101.0, 101.5])
        # Non-time arrays passed through by reference (zero-copy).
        assert shifted.event_hpd_overlap is diagnostics.event_hpd_overlap

    def test_passthrough_when_none(self) -> None:
        """Simulated-data path leaves ``event_time`` as ``None``; the shift is a
        no-op there, not a raise."""
        diagnostics = _make_per_cell_diagnostics(event_time=None, event_hpd_overlap=np.array([0.5]))
        assert _shift_diagnostic_event_times(diagnostics, 100.0) is diagnostics


class TestFigure4DetailWindow:
    def test_converts_center_and_half_width_to_slice(self) -> None:
        assert Figure4DetailWindow(center_index=20, half_width_samples=10).to_slice(40) == slice(
            10, 30
        )

    @pytest.mark.parametrize(
        ("center_index", "half_width_samples"),
        [(-1, 10), (20, 0), (20, -1), (20.0, 10)],
    )
    def test_rejects_invalid_values(self, center_index: int, half_width_samples: int) -> None:
        with pytest.raises(ValueError):
            Figure4DetailWindow(
                center_index=center_index,
                half_width_samples=half_width_samples,
            )

    def test_rejects_window_outside_recording(self) -> None:
        with pytest.raises(ValueError, match="outside the recording timeline"):
            Figure4DetailWindow(center_index=5, half_width_samples=10).to_slice(40)

    def test_rejects_invalid_recording_length(self) -> None:
        with pytest.raises(ValueError, match="n_time_samples"):
            Figure4DetailWindow(center_index=5, half_width_samples=2).to_slice(0)


# ---------------------------------------------------------------------------
# compose_figure04 end-to-end smoke test (synthetic Figure4RenderData)
# ---------------------------------------------------------------------------

_N_TIME, _N_CELLS, _N_POS = 40, 6, 12
_STATES = ("Continuous", "Fragmented")


def _compose_results(seed: int) -> xr.Dataset:
    rng = np.random.default_rng(seed)
    pos = np.linspace(0.0, 100.0, _N_POS)
    state_bins = pd.MultiIndex.from_product([list(_STATES), pos], names=["state", "position"])
    time = np.arange(_N_TIME, dtype=float)
    n_sb = len(state_bins)

    def _da(a: np.ndarray) -> xr.DataArray:
        return xr.DataArray(
            a, dims=("time", "state_bins"), coords={"time": time, "state_bins": state_bins}
        )

    return xr.Dataset(
        {
            "predictive_posterior": _da(rng.dirichlet(np.ones(n_sb), size=_N_TIME)),
            "log_likelihood": _da(rng.normal(size=(_N_TIME, n_sb))),
        }
    )


def _compose_diagnostics(seed: int) -> SpikeEventDiagnostics:
    rng = np.random.default_rng(seed)
    n_spk = 20
    return SpikeEventDiagnostics(
        event_time_ind=rng.integers(0, _N_TIME, n_spk).astype(np.intp),
        event_cell_ind=rng.integers(0, _N_CELLS, n_spk).astype(np.intp),
        event_hpd_overlap=rng.uniform(0, 1, n_spk),
        event_kl_divergence=rng.gamma(2.0, 0.5, n_spk),
        event_predictive_pvalue=rng.uniform(0.01, 1, n_spk),
        hpd_overlap=rng.uniform(0, 1, (_N_TIME, _N_CELLS)),
        kl_divergence=rng.gamma(2.0, 0.5, (_N_TIME, _N_CELLS)),
        predictive_pvalue=rng.uniform(0.01, 1, (_N_TIME, _N_CELLS)),
        per_spike_likelihood=rng.uniform(0, 1, (n_spk, _N_POS)),
        event_time=np.sort(rng.uniform(0, _N_TIME, n_spk)),
    )


def _compose_recording() -> NeuralRecordingData:
    rng = np.random.default_rng(11)
    track_graph = nx.Graph()
    for i in range(6):
        track_graph.add_node(i, pos=(float(i * 20), 0.0))
    for i in range(5):
        track_graph.add_edge(i, i + 1, distance=20.0)
    position_info = pd.DataFrame(
        {
            "head_position_x": np.linspace(0.0, 100.0, _N_TIME),
            "head_position_y": np.zeros(_N_TIME),
            "linear_position": np.linspace(0.0, 100.0, _N_TIME),
        },
        index=np.arange(_N_TIME, dtype=float),
    )
    spike_times = tuple(
        np.sort(rng.uniform(0, _N_TIME, rng.integers(3, 8))).astype(np.float64)
        for _ in range(_N_CELLS)
    )
    return NeuralRecordingData(
        position_info=position_info,
        spike_times=spike_times,
        track_graph=track_graph,
        linear_edge_order=tuple((i, i + 1) for i in range(5)),
        linear_edge_spacing=0.0,
    )


def _compose_render_data() -> Figure4RenderData:
    rng = np.random.default_rng(3)
    continuous_diagnostics = _compose_diagnostics(3)
    continuous_fragmented_diagnostics = dataclasses.replace(
        _compose_diagnostics(4),
        event_time_ind=continuous_diagnostics.event_time_ind,
        event_cell_ind=continuous_diagnostics.event_cell_ind,
        event_time=continuous_diagnostics.event_time,
    )
    decode = Figure4DecodeResults(
        continuous_results=_compose_results(1),
        continuous_fragmented_results=_compose_results(2),
        continuous_diagnostics=continuous_diagnostics,
        continuous_fragmented_diagnostics=continuous_fragmented_diagnostics,
        spike_counts=rng.poisson(0.4, (_N_TIME, _N_CELLS)).astype(np.int64),
        place_field_peaks=np.linspace(5.0, 95.0, _N_CELLS),
        diagnostic_place_fields=rng.random((_N_CELLS, _N_POS)) * 10 + 0.1,
        diagnostic_position_bins=np.linspace(0.0, 100.0, _N_POS),
    )
    return Figure4RenderData(
        recording=_compose_recording(),
        time=np.arange(_N_TIME, dtype=float),
        head_position=np.column_stack([np.linspace(0.0, 100.0, _N_TIME), np.zeros(_N_TIME)]),
        linear_position=np.linspace(0.0, 100.0, _N_TIME),
        decode_results=decode,
    )


def test_compose_figure04_produces_panels_and_finite_bbox() -> None:
    """End-to-end smoke test: with the detail window shrunk to the synthetic
    session, composition builds the a/b detail stacks, the c hexbin row, the
    track inset, and returns a finite tight bounding box."""
    render_data = _compose_render_data()
    result = compose_figure04(
        render_data,
        diagnostic_thresholds={"hpd_overlap": 0.05, "predictive_pvalue": 0.05},
        detail_window=Figure4DetailWindow(center_index=20, half_width_samples=10),
    )

    assert isinstance(result, Figure4Composition)
    # Two 6-row detail stacks (a, b) + the 3-panel hexbin row (c) + the track
    # inset all live as axes on the returned figure.
    assert len(result.figure.axes) >= 12
    # The crop bbox must be finite (composition measured real artist extents).
    assert np.isfinite(np.asarray(result.bbox_inches.extents)).all()

    import matplotlib.pyplot as plt

    plt.close(result.figure)
