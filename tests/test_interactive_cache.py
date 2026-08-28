"""Smoke tests for the interactive viewer cache builders.

The pure-data tests exercise the array-shape logic (max-pool, pyramid
build, event-table assembly, Zarr writer) on synthetic inputs and must
pass without ``non_local_detector`` or any real recording files.

The real-data integration test is skipped unless the canonical Figure 4 joblib
bundle and exported recording inputs are available.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from statespacecheck_paper.diagnostics import SpikeEventDiagnostics
from statespacecheck_paper.interactive import cache as cache_mod
from statespacecheck_paper.interactive.data_source import DecoderDataSource

REPO_ROOT = Path(__file__).resolve().parents[1]
INTERMEDIATES = REPO_ROOT / "data" / "intermediates"
RAW_DATA = REPO_ROOT / "data"
ANIMAL_DATE_EPOCH = "j1620210710_02_r1"


def _synthetic_results_dataset(
    n_time: int,
    n_states: int,
    n_position: int,
) -> xr.Dataset:
    """Build a synthetic decoder-results Dataset matching the on-disk layout.

    ``state_bins`` is a plain integer dim with ``state`` and ``position``
    as non-dim coords on it (this mirrors what the real NetCDF
    round-trip produces).
    """
    rng = np.random.default_rng(0)
    n_state_bins = n_states * n_position
    state_names = [f"state_{i}" for i in range(n_states)]
    state_coord = np.array(
        [state_names[i] for i in range(n_states) for _ in range(n_position)],
        dtype=object,
    )
    position_grid = np.linspace(0.0, 100.0, n_position)
    position_coord = np.tile(position_grid, n_states)

    posterior = rng.dirichlet(np.ones(n_state_bins), size=n_time).astype(np.float32)
    log_likelihood = np.log(posterior + 1e-12).astype(np.float32)
    state_probs = rng.dirichlet(np.ones(n_states), size=n_time).astype(np.float32)

    time = np.arange(n_time, dtype=np.float64) * 0.002
    coords: dict[str, Any] = {
        "time": ("time", time),
        "state_bins": ("state_bins", np.arange(n_state_bins, dtype=np.int64)),
        "state": ("state_bins", state_coord),
        "position": ("state_bins", position_coord),
    }
    if n_states > 1:
        coords["states"] = ("states", np.array(state_names, dtype=object))

    data_vars: dict[str, Any] = {
        "predictive_posterior": (("time", "state_bins"), posterior),
        "log_likelihood": (("time", "state_bins"), log_likelihood),
    }
    if n_states > 1:
        data_vars["acausal_state_probabilities"] = (("time", "states"), state_probs)
    else:
        data_vars["acausal_state_probabilities"] = (
            ("time",),
            state_probs[:, 0],
        )

    return xr.Dataset(data_vars=data_vars, coords=coords)


def _per_spike(
    *,
    event_time: np.ndarray,
    event_cell_ind: np.ndarray,
    event_hpd_overlap: np.ndarray,
    event_kl_divergence: np.ndarray,
    event_predictive_pvalue: np.ndarray,
) -> SpikeEventDiagnostics:
    n_spikes = event_time.shape[0]
    return SpikeEventDiagnostics(
        event_time_ind=np.zeros(n_spikes, dtype=np.intp),
        event_cell_ind=event_cell_ind.astype(np.intp),
        event_hpd_overlap=event_hpd_overlap,
        event_kl_divergence=event_kl_divergence,
        event_predictive_pvalue=event_predictive_pvalue,
        hpd_overlap=None,
        kl_divergence=None,
        predictive_pvalue=None,
        per_spike_likelihood=None,
        event_time=event_time,
    )


def test_events_dataframe_sorts_by_time_and_validates_cell_id() -> None:
    diagnostics = _per_spike(
        event_time=np.array([2.0, 1.0, 3.0], dtype=np.float64),
        event_cell_ind=np.array([0, 2, 1], dtype=np.int64),
        event_hpd_overlap=np.array([0.1, 0.2, 0.3], dtype=np.float32),
        event_kl_divergence=np.array([1.0, 2.0, 3.0], dtype=np.float32),
        event_predictive_pvalue=np.array([0.5, 0.4, 0.3], dtype=np.float32),
    )
    df = cache_mod._events_dataframe(diagnostics, n_cells=3)
    assert list(df.columns) == [
        "time",
        "cell_id",
        "event_hpd_overlap",
        "event_kl_divergence",
        "event_predictive_pvalue",
    ]
    assert df["time"].tolist() == [1.0, 2.0, 3.0]
    assert df["cell_id"].tolist() == [2, 0, 1]
    assert df["cell_id"].dtype == np.int32


def test_events_dataframe_rejects_out_of_range_cell_id() -> None:
    diagnostics = _per_spike(
        event_time=np.array([1.0], dtype=np.float64),
        event_cell_ind=np.array([5], dtype=np.int64),
        event_hpd_overlap=np.array([0.0], dtype=np.float32),
        event_kl_divergence=np.array([0.0], dtype=np.float32),
        event_predictive_pvalue=np.array([0.0], dtype=np.float32),
    )
    with pytest.raises(ValueError, match="event_cell_ind out of range"):
        cache_mod._events_dataframe(diagnostics, n_cells=3)


def test_write_zarr_store_roundtrips_arrays(tmp_path: Path) -> None:
    """``_write_zarr_store`` writes the full-res arrays + non-dim coords."""
    ds = _synthetic_results_dataset(n_time=200, n_states=2, n_position=8)
    out_dir = tmp_path / "cache.zarr"

    shapes = cache_mod._write_zarr_store(ds=ds, out_dir=out_dir, time_chunk=64)
    assert shapes["predictive_posterior"] == (200, 16)
    assert shapes["log_likelihood"] == (200, 16)

    with xr.open_zarr(out_dir, consolidated=True) as readback:
        np.testing.assert_array_equal(
            readback["predictive_posterior"].values,
            ds["predictive_posterior"].values,
        )
        # ``state`` / ``position`` non-dim coords on ``state_bins`` survive.
        np.testing.assert_array_equal(readback["state"].values, ds["state"].values)
        np.testing.assert_array_equal(readback["position"].values, ds["position"].values)


def test_write_zarr_store_overwrites_existing(tmp_path: Path) -> None:
    """Re-writing the same path replaces the prior store."""
    ds = _synthetic_results_dataset(n_time=64, n_states=1, n_position=4)
    out_dir = tmp_path / "cache.zarr"
    cache_mod._write_zarr_store(ds=ds, out_dir=out_dir, time_chunk=32)
    # Smaller chunks the second time around — verify it doesn't error
    # and the round-tripped data still matches.
    cache_mod._write_zarr_store(ds=ds, out_dir=out_dir, time_chunk=16)
    with xr.open_zarr(out_dir, consolidated=True) as rb:
        np.testing.assert_array_equal(
            rb["predictive_posterior"].values, ds["predictive_posterior"].values
        )


def test_build_figure04_viewer_cache_uses_canonical_render_data(tmp_path: Path) -> None:
    """Both viewer models are derived from one canonical Figure 4 payload."""
    n_time, n_cells, n_position = 200, 3, 8
    time = np.arange(n_time, dtype=np.float64) * 0.002
    position_bins = np.linspace(0.0, 100.0, n_position)
    event_time = np.array([time[10], time[50], time[150]])
    diagnostics = _per_spike(
        event_time=event_time,
        event_cell_ind=np.array([0, 2, 1]),
        event_hpd_overlap=np.array([0.1, 0.2, 0.3]),
        event_kl_divergence=np.array([1.0, 2.0, 3.0]),
        event_predictive_pvalue=np.array([0.5, 0.4, 0.3]),
    )
    place_fields = np.full((n_cells, n_position), 0.1, dtype=np.float64)
    # The canonical joblib bundle preserves the state/position MultiIndex;
    # exercise its flattening to the viewer's Zarr-compatible coordinates.
    continuous_results = _synthetic_results_dataset(n_time, 1, n_position).set_index(
        state_bins=["state", "position"]
    )
    contfrag_results = _synthetic_results_dataset(n_time, 2, n_position).set_index(
        state_bins=["state", "position"]
    )
    decode = SimpleNamespace(
        continuous_results=continuous_results,
        continuous_fragmented_results=contfrag_results,
        continuous_diagnostics=diagnostics,
        continuous_fragmented_diagnostics=diagnostics,
        spike_counts=np.zeros((n_time, n_cells), dtype=np.int64),
        place_field_peaks=np.array([0.0, 50.0, 100.0]),
        diagnostic_place_fields=place_fields,
        diagnostic_position_bins=position_bins,
    )
    recording = SimpleNamespace(
        spike_times=(
            np.array([event_time[0]]),
            np.array([event_time[2]]),
            np.array([event_time[1]]),
        )
    )
    render_data = SimpleNamespace(
        decode_results=decode,
        recording=recording,
        time=time,
        linear_position=np.linspace(0.0, 100.0, n_time),
    )

    summaries = cache_mod.build_figure04_viewer_cache(
        render_data=render_data,
        cache_dir=tmp_path,
        time_chunk=64,
    )

    assert set(summaries) == {"continuous", "contfrag"}
    assert summaries["continuous"]["n_events"] == 3
    with DecoderDataSource.for_model(tmp_path, "continuous") as continuous:
        assert continuous.n_states == 1
        assert continuous.load_posterior(slice(0, 5)).shape == (5, n_position)
        assert continuous.events["cell_id"].tolist() == [0, 2, 1]
    with DecoderDataSource.for_model(tmp_path, "contfrag") as contfrag:
        assert contfrag.n_states == 2
        assert contfrag.load_posterior(slice(0, 5)).shape == (5, 2 * n_position)
        assert contfrag.event_likelihood_at(0, 0).shape == (n_position,)


def test_build_cli_loads_the_canonical_figure04_workflow(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The CLI no longer asks for disconnected NetCDF/model intermediates."""
    from statespacecheck_paper import figure04_workflow

    sentinel = object()
    seen: dict[str, Any] = {}

    def _prepare(config: object, paths: object, *, use_cache: bool) -> object:
        seen.update(config=config, paths=paths, use_cache=use_cache)
        return sentinel

    def _build(**kwargs: Any) -> dict[str, dict[str, int]]:
        seen.update(build_kwargs=kwargs)
        return {
            "continuous": {
                "n_time": 10,
                "n_cells": 2,
                "n_state_bins_full_res": 4,
                "n_events": 3,
            }
        }

    monkeypatch.setattr(figure04_workflow, "prepare_figure04_render_data", _prepare)
    monkeypatch.setattr(cache_mod, "build_figure04_viewer_cache", _build)

    result = cache_mod.main(
        [
            "build",
            "--data-dir",
            str(tmp_path),
            "--cache-dir",
            str(tmp_path / "viewer"),
            "--model",
            "continuous",
        ]
    )

    assert result == 0
    assert seen["use_cache"] is True
    assert seen["build_kwargs"]["render_data"] is sentinel
    assert seen["build_kwargs"]["models"] == ("continuous",)


# ---------------------------------------------------------------------------
# Real-data integration test (skipped when intermediates are not available).
# ---------------------------------------------------------------------------

FIGURE04_JOBLIB = INTERMEDIATES / f"{ANIMAL_DATE_EPOCH}_fig4_cache.joblib"
RAW_SPIKES_PKL = RAW_DATA / f"{ANIMAL_DATE_EPOCH}_HPC_spike_times.pkl"
RAW_POSITION_PKL = RAW_DATA / f"{ANIMAL_DATE_EPOCH}_position_info.pkl"
RAW_TRACK_GRAPH = RAW_DATA / f"{ANIMAL_DATE_EPOCH}_track_graph.pkl"
RAW_LINEAR_EDGE_ORDER = RAW_DATA / f"{ANIMAL_DATE_EPOCH}_linear_edge_order.pkl"
RAW_LINEAR_EDGE_SPACING = RAW_DATA / f"{ANIMAL_DATE_EPOCH}_linear_edge_spacing.pkl"

REAL_DATA_AVAILABLE = all(
    p.exists()
    for p in [
        FIGURE04_JOBLIB,
        RAW_SPIKES_PKL,
        RAW_POSITION_PKL,
        RAW_TRACK_GRAPH,
        RAW_LINEAR_EDGE_ORDER,
        RAW_LINEAR_EDGE_SPACING,
    ]
)


@pytest.mark.slow
@pytest.mark.skipif(
    not REAL_DATA_AVAILABLE,
    reason="Real Figure 4 data not available in data/ and data/intermediates/.",
)
def test_build_figure04_viewer_cache_continuous_integration(tmp_path: Path) -> None:
    """Build the Continuous viewer cache from canonical Figure 4 data.

    This test takes several minutes and several GB of disk; it runs only
    when the canonical joblib bundle and exported recording inputs are present.
    """
    cache_dir = tmp_path / "cache"
    from statespacecheck_paper.figure04_cache import Figure4Paths
    from statespacecheck_paper.figure04_decoder import Figure4Config
    from statespacecheck_paper.figure04_workflow import prepare_figure04_render_data

    render_data = prepare_figure04_render_data(
        Figure4Config(),
        Figure4Paths(RAW_DATA, ANIMAL_DATE_EPOCH),
        use_cache=True,
    )
    info = cache_mod.build_figure04_viewer_cache(
        render_data=render_data,
        cache_dir=cache_dir,
        models=("continuous",),
    )
    continuous_info = info["continuous"]

    assert continuous_info["model"] == "continuous"
    assert continuous_info["n_time"] == 709321
    assert continuous_info["n_cells"] == 203
    # Continuous decoder: 256 state_bins (full); ~248 interior bins.
    assert continuous_info["n_state_bins_full_res"] == 256
    assert 240 <= continuous_info["n_position_bins"] <= 256
    # Spike count is in the high-800Ks per the inspection.
    assert 850000 <= continuous_info["n_events"] <= 900000

    # Verify on-disk artifacts exist.
    paths = cache_mod.cache_paths(cache_dir, "continuous")
    assert paths["zarr"].is_dir()
    assert paths["events"].is_file()
    assert paths["place_fields"].is_file()
    assert cache_mod.meta_path(cache_dir).is_file()
    assert cache_mod.spike_times_path(cache_dir).is_file()

    # Quick read-back: a 2-second window (1000 samples) reads a small
    # number of chunks and matches in shape.
    with xr.open_zarr(paths["zarr"], consolidated=True) as ds:
        window = ds["predictive_posterior"].isel(time=slice(100_000, 101_000))
        arr = window.values
        assert arr.shape == (1000, 256)
        assert arr.dtype == np.float32

    # Event Parquet sorted by time.
    events = pd.read_parquet(paths["events"])
    assert events["time"].is_monotonic_increasing
    assert events["cell_id"].between(0, continuous_info["n_cells"] - 1).all()
