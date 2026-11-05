"""Shared synthetic Figure-4 values for the orchestration and panel tests."""

from __future__ import annotations

import dataclasses

import networkx as nx
import numpy as np
import pandas as pd
import xarray as xr

from statespacecheck_paper.diagnostics import SpikeEventDiagnostics
from statespacecheck_paper.figure04_cache import (
    FIGURE04_DECODE_SCHEMA_VERSION,
    FIGURE04_DIAGNOSTICS_SCHEMA_VERSION,
    Figure4CacheEnvironment,
    Figure4CacheProvenance,
)
from statespacecheck_paper.figure04_decoder import Figure4DiagnosticsConfig
from statespacecheck_paper.figure04_input import NeuralRecordingData
from statespacecheck_paper.figure04_workflow import Figure4AnalysisResults, Figure4RenderData

_N_TIME, _N_CELLS, _N_POSITION = 40, 6, 12


def synthetic_results(seed: int, *, n_time: int, n_position: int) -> xr.Dataset:
    """Synthetic decoder results with a (state, position) MultiIndex."""
    rng = np.random.default_rng(seed)
    state_bins = pd.MultiIndex.from_product(
        [["Continuous", "Fragmented"], np.linspace(0.0, 100.0, n_position)],
        names=["state", "position"],
    )
    time = np.arange(n_time, dtype=float)
    n_state_bins = len(state_bins)

    def _data_array(values: np.ndarray) -> xr.DataArray:
        return xr.DataArray(
            values, dims=("time", "state_bins"), coords={"time": time, "state_bins": state_bins}
        )

    return xr.Dataset(
        {
            "predictive_posterior": _data_array(rng.dirichlet(np.ones(n_state_bins), size=n_time)),
            "log_likelihood": _data_array(rng.normal(size=(n_time, n_state_bins))),
        }
    )


def synthetic_dense_diagnostics(
    seed: int, *, n_time: int, n_cells: int, n_position: int, n_spikes: int
) -> SpikeEventDiagnostics:
    """``SpikeEventDiagnostics`` with dense ``(n_time, n_cells)`` metric matrices."""
    rng = np.random.default_rng(seed)
    return SpikeEventDiagnostics(
        event_time_ind=rng.integers(0, n_time, n_spikes).astype(np.intp),
        event_cell_ind=rng.integers(0, n_cells, n_spikes).astype(np.intp),
        event_hpd_overlap=rng.uniform(0, 1, n_spikes),
        event_kl_divergence=rng.gamma(2.0, 0.5, n_spikes),
        event_predictive_pvalue=rng.uniform(0.01, 1, n_spikes),
        hpd_overlap=rng.uniform(0, 1, (n_time, n_cells)),
        kl_divergence=rng.gamma(2.0, 0.5, (n_time, n_cells)),
        predictive_pvalue=rng.uniform(0.01, 1, (n_time, n_cells)),
        event_likelihood=rng.uniform(0, 1, (n_spikes, n_position)),
        event_time=np.sort(rng.uniform(0, n_time, n_spikes)),
    )


def chain_track_graph(edge_length: float) -> nx.Graph:
    """Six nodes joined in a line, ``edge_length`` apart."""
    graph = nx.Graph()
    for i in range(6):
        graph.add_node(i, pos=(float(i) * edge_length, 0.0))
    for i in range(5):
        graph.add_edge(i, i + 1, distance=edge_length)
    return graph


def _synthetic_recording() -> NeuralRecordingData:
    rng = np.random.default_rng(11)
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
        track_graph=chain_track_graph(20.0),
        linear_edge_order=tuple((i, i + 1) for i in range(5)),
        linear_edge_spacing=0.0,
    )


def synthetic_render_data() -> Figure4RenderData:
    """A small, self-consistent ``Figure4RenderData`` for composing Figure 4."""
    rng = np.random.default_rng(3)
    sizes = {"n_time": _N_TIME, "n_cells": _N_CELLS, "n_position": _N_POSITION, "n_spikes": 20}
    continuous_diagnostics = synthetic_dense_diagnostics(3, **sizes)
    continuous_fragmented_diagnostics = dataclasses.replace(
        synthetic_dense_diagnostics(4, **sizes),
        event_time_ind=continuous_diagnostics.event_time_ind,
        event_cell_ind=continuous_diagnostics.event_cell_ind,
        event_time=continuous_diagnostics.event_time,
    )
    decode = Figure4AnalysisResults(
        continuous_results=synthetic_results(1, n_time=_N_TIME, n_position=_N_POSITION),
        continuous_fragmented_results=synthetic_results(2, n_time=_N_TIME, n_position=_N_POSITION),
        continuous_diagnostics=continuous_diagnostics,
        continuous_fragmented_diagnostics=continuous_fragmented_diagnostics,
        spike_counts=rng.poisson(0.4, (_N_TIME, _N_CELLS)).astype(np.int64),
        place_field_peaks=np.linspace(5.0, 95.0, _N_CELLS),
        diagnostic_place_fields=rng.random((_N_CELLS, _N_POSITION)) * 10 + 0.1,
        diagnostic_position_bins=np.linspace(0.0, 100.0, _N_POSITION),
    )
    return Figure4RenderData(
        recording=_synthetic_recording(),
        time=np.arange(_N_TIME, dtype=float),
        linear_position=np.linspace(0.0, 100.0, _N_TIME),
        analysis_results=decode,
        cache_provenance=synthetic_cache_provenance(),
    )


def synthetic_cache_provenance(animal_date_epoch: str = "epoch_x") -> Figure4CacheProvenance:
    """Cache provenance with placeholder fingerprints and input-file checksum.

    The decode fingerprint is ``"c" * 64``, the diagnostics fingerprint
    ``"e" * 64``, their identities ``"a" * 64`` and ``"b" * 64``, and the
    input-file checksum ``"d" * 64``.
    """
    return Figure4CacheProvenance(
        fingerprint_sha256="c" * 64,
        decode_identity_sha256="a" * 64,
        schema_version=FIGURE04_DECODE_SCHEMA_VERSION,
        animal_date_epoch=animal_date_epoch,
        input_file_sha256="d" * 64,
        non_local_detector_version="1.2.3",
        diagnostics_fingerprint_sha256="e" * 64,
        diagnostics_identity_sha256="b" * 64,
        diagnostics_schema_version=FIGURE04_DIAGNOSTICS_SCHEMA_VERSION,
        statespacecheck_version="0.1.0",
        diagnostics_config=Figure4DiagnosticsConfig(),
        environment=Figure4CacheEnvironment(
            python_version="3.11.0",
            machine="x86_64",
            decode_dependency_versions={"non-local-detector": "1.2.3", "numpy": "2.0.0"},
            diagnostics_dependency_versions={"numpy": "2.0.0", "statespacecheck": "0.1.0"},
        ),
    )
