"""Smoke tests for the interactive viewer cache builders.

The pure-data tests exercise the event-table assembly, the Zarr writer,
and the Figure 4 viewer-cache builder and CLI on synthetic inputs and
must pass without ``non_local_detector`` or any real recording files.

The real-data integration test is skipped unless the canonical Figure 4 joblib
decode cache and the Figure 4 input file are available.
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
from statespacecheck_paper.figure04_diagnostics import compute_spike_event_diagnostics
from statespacecheck_paper.interactive import cache as cache_mod
from statespacecheck_paper.interactive.data_source import DecoderDataSource
from statespacecheck_paper.paths import REPO_ROOT

INTERMEDIATES = REPO_ROOT / "data" / "intermediates"
RAW_DATA = REPO_ROOT / "data"
ANIMAL_DATE_EPOCH = "j1620210710_02_r1"


def _synthetic_results_dataset(
    n_time: int,
    n_states: int,
    n_position: int,
) -> xr.Dataset:
    """Build a synthetic decoder-results Dataset in the flat state layout.

    ``state_bins`` is a plain integer dim with ``state`` and ``position``
    as non-dim coords on it: the layout the Zarr writer stores after
    flattening the joblib cache's ``state_bins`` MultiIndex. Like the
    decoder's output it also carries ``acausal_state_probabilities``, which
    the viewer does not read and the writer leaves out.
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

    predictive = rng.dirichlet(np.ones(n_state_bins), size=n_time).astype(np.float32)
    log_likelihood = np.log(predictive + 1e-12).astype(np.float32)
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
        "predictive_posterior": (("time", "state_bins"), predictive),
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
    event_time_ind: np.ndarray,
    event_cell_ind: np.ndarray,
    event_hpd_overlap: np.ndarray,
    event_kl_divergence: np.ndarray,
    event_predictive_pvalue: np.ndarray,
) -> SpikeEventDiagnostics:
    return SpikeEventDiagnostics(
        event_time_ind=event_time_ind.astype(np.intp),
        event_cell_ind=event_cell_ind.astype(np.intp),
        event_hpd_overlap=event_hpd_overlap,
        event_kl_divergence=event_kl_divergence,
        event_predictive_pvalue=event_predictive_pvalue,
        hpd_overlap=None,
        kl_divergence=None,
        predictive_pvalue=None,
        event_likelihood=None,
        event_time=event_time,
    )


def test_events_dataframe_sorts_by_time_and_validates_cell_id() -> None:
    diagnostics = _per_spike(
        event_time=np.array([2.0, 1.0, 3.0], dtype=np.float64),
        event_time_ind=np.array([2, 1, 2]),
        event_cell_ind=np.array([0, 2, 1], dtype=np.int64),
        event_hpd_overlap=np.array([0.1, 0.2, 0.3], dtype=np.float32),
        event_kl_divergence=np.array([1.0, 2.0, 3.0], dtype=np.float32),
        event_predictive_pvalue=np.array([0.5, 0.4, 0.3], dtype=np.float32),
    )
    df = cache_mod._events_dataframe(diagnostics, n_cells=3, time=np.arange(4.0))
    assert list(df.columns) == [
        "time",
        "event_time_ind",
        "cell_id",
        "event_hpd_overlap",
        "event_kl_divergence",
        "event_predictive_pvalue",
    ]
    assert df["time"].tolist() == [1.0, 2.0, 3.0]
    assert df["cell_id"].tolist() == [2, 0, 1]
    assert df["cell_id"].dtype == np.int32
    # The stored bins are the diagnostics' own, reordered with the rows.
    assert df["event_time_ind"].tolist() == [1, 2, 2]


def test_events_dataframe_rejects_out_of_range_cell_id() -> None:
    diagnostics = _per_spike(
        event_time=np.array([1.0], dtype=np.float64),
        event_time_ind=np.array([1]),
        event_cell_ind=np.array([5], dtype=np.int64),
        event_hpd_overlap=np.array([0.0], dtype=np.float32),
        event_kl_divergence=np.array([0.0], dtype=np.float32),
        event_predictive_pvalue=np.array([0.0], dtype=np.float32),
    )
    with pytest.raises(ValueError, match="event_cell_ind out of range"):
        cache_mod._events_dataframe(diagnostics, n_cells=3, time=np.arange(4.0))


def test_write_zarr_store_roundtrips_arrays(tmp_path: Path) -> None:
    """``_write_zarr_store`` writes the full-res arrays + non-dim coords."""
    ds = _synthetic_results_dataset(n_time=200, n_states=2, n_position=8)
    out_dir = tmp_path / "cache.zarr"

    shapes = cache_mod._write_zarr_store(ds=ds, out_dir=out_dir, time_chunk=64)
    assert shapes == {"predictive_posterior": (200, 16), "log_likelihood": (200, 16)}

    with xr.open_zarr(out_dir, consolidated=True) as readback:
        # Only the arrays the viewer reads are written.
        assert set(readback.data_vars) == {"predictive_posterior", "log_likelihood"}
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


def _canonical_render_data(
    *,
    time: np.ndarray,
    diagnostics: SpikeEventDiagnostics,
    spike_times: tuple[np.ndarray, ...],
    place_fields: np.ndarray,
    position_bins: np.ndarray,
) -> SimpleNamespace:
    """Stand in for ``Figure4RenderData`` with both models sharing one diagnostics set."""
    n_time = time.shape[0]
    n_cells, n_position = place_fields.shape
    # The canonical joblib decode cache preserves the state/position MultiIndex;
    # exercise its flattening to the viewer's Zarr-compatible coordinates.
    continuous_results = _synthetic_results_dataset(n_time, 1, n_position).set_index(
        state_bins=["state", "position"]
    )
    continuous_fragmented_results = _synthetic_results_dataset(n_time, 2, n_position).set_index(
        state_bins=["state", "position"]
    )
    decode = SimpleNamespace(
        continuous_results=continuous_results,
        continuous_fragmented_results=continuous_fragmented_results,
        continuous_diagnostics=diagnostics,
        continuous_fragmented_diagnostics=diagnostics,
        spike_counts=np.zeros((n_time, n_cells), dtype=np.int64),
        place_field_peaks=np.linspace(position_bins[0], position_bins[-1], n_cells),
        diagnostic_place_fields=place_fields,
        diagnostic_position_bins=position_bins,
    )
    return SimpleNamespace(
        decode_results=decode,
        recording=SimpleNamespace(spike_times=spike_times),
        time=time,
        linear_position=np.linspace(0.0, 100.0, n_time),
    )


def test_build_figure04_viewer_cache_uses_canonical_render_data(tmp_path: Path) -> None:
    """Both viewer models are derived from one canonical Figure 4 payload."""
    n_time, n_cells, n_position = 200, 3, 8
    time = np.arange(n_time, dtype=np.float64) * 0.002
    event_time = np.array([time[10], time[50], time[150]])
    diagnostics = _per_spike(
        event_time=event_time,
        event_time_ind=np.array([10, 50, 150]),
        event_cell_ind=np.array([0, 2, 1]),
        event_hpd_overlap=np.array([0.1, 0.2, 0.3]),
        event_kl_divergence=np.array([1.0, 2.0, 3.0]),
        event_predictive_pvalue=np.array([0.5, 0.4, 0.3]),
    )
    render_data = _canonical_render_data(
        time=time,
        diagnostics=diagnostics,
        spike_times=(
            np.array([event_time[0]]),
            np.array([event_time[2]]),
            np.array([event_time[1]]),
        ),
        place_fields=np.full((n_cells, n_position), 0.1, dtype=np.float64),
        position_bins=np.linspace(0.0, 100.0, n_position),
    )

    summaries = cache_mod.build_figure04_viewer_cache(
        render_data=render_data,
        cache_dir=tmp_path,
        time_chunk=64,
    )

    assert set(summaries) == {"continuous", "continuous_fragmented"}
    assert summaries["continuous"]["n_events"] == 3
    with DecoderDataSource.for_recording(tmp_path, "continuous") as continuous:
        assert continuous.n_states == 1
        assert continuous.load_predictive(slice(0, 5)).shape == (5, n_position)
        assert continuous.events["cell_id"].tolist() == [0, 2, 1]
        assert continuous.event_time_idx.tolist() == [10, 50, 150]
    with DecoderDataSource.for_recording(
        tmp_path, "continuous_fragmented"
    ) as continuous_fragmented:
        assert continuous_fragmented.n_states == 2
        assert continuous_fragmented.load_predictive(slice(0, 5)).shape == (5, 2 * n_position)
        assert continuous_fragmented.event_likelihood_at(0, 0).shape == (n_position,)


def test_viewer_event_bins_equal_the_diagnostics_bins_at_the_final_timestamp(
    tmp_path: Path,
) -> None:
    """A spike exactly at ``time[-1]`` stays in the bin the diagnostics used.

    The diagnostics bin spikes as the decoder does, so a spike at the final
    timestamp is scored against row ``n_time - 2``; the viewer must show it
    there rather than re-binning its time into the last row.
    """
    n_time, n_position = 40, 8
    time = np.arange(n_time, dtype=np.float64) * 0.002
    position_bins = np.linspace(0.0, 100.0, n_position)
    rng = np.random.default_rng(0)
    place_fields = rng.uniform(0.1, 2.0, size=(2, n_position))
    predictive = rng.dirichlet(np.ones(n_position), size=n_time)
    spike_times = (
        np.array([time[3], time[3] + 0.0005, time[-1]]),
        np.array([time[20] + 0.001, time[-2]]),
    )
    diagnostics = compute_spike_event_diagnostics(
        predictive, place_fields, list(spike_times), time, include_dense_matrices=False
    )
    assert diagnostics.event_time is not None
    final = diagnostics.event_time == time[-1]
    assert diagnostics.event_time_ind[final].tolist() == [n_time - 2]

    cache_mod.build_figure04_viewer_cache(
        render_data=_canonical_render_data(
            time=time,
            diagnostics=diagnostics,
            spike_times=spike_times,
            place_fields=place_fields,
            position_bins=position_bins,
        ),
        cache_dir=tmp_path,
        models=("continuous",),
        time_chunk=16,
    )

    order = np.argsort(diagnostics.event_time, kind="stable")
    with DecoderDataSource.for_recording(tmp_path, "continuous") as ds:
        np.testing.assert_array_equal(ds.event_times, diagnostics.event_time[order])
        np.testing.assert_array_equal(ds.event_time_idx, diagnostics.event_time_ind[order])
        final_row = int(np.flatnonzero(ds.event_times == time[-1])[0])
        assert ds.event_time_idx[final_row] == n_time - 2
        i0, i1 = ds.event_indices_at(n_time - 2)
        assert i0 <= final_row < i1
        i0, i1 = ds.event_indices_at(n_time - 1)
        assert i1 <= i0


def test_build_cli_loads_the_canonical_figure04_workflow(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The CLI builds from the canonical Figure 4 workflow's render data."""
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

FIGURE04_JOBLIB = INTERMEDIATES / f"{ANIMAL_DATE_EPOCH}_figure04_decode.joblib"
RAW_INPUTS = RAW_DATA / f"{ANIMAL_DATE_EPOCH}_figure04_inputs.npz"

REAL_DATA_AVAILABLE = all(
    p.exists()
    for p in [
        FIGURE04_JOBLIB,
        RAW_INPUTS,
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
    when the canonical joblib decode cache and the Figure 4 input file are present.
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
    paths = cache_mod.recording_cache_paths(cache_dir, "continuous")
    assert paths["zarr"].is_dir()
    assert paths["events"].is_file()
    assert paths["place_fields"].is_file()
    assert cache_mod.recording_meta_path(cache_dir).is_file()
    assert cache_mod.recording_spike_times_path(cache_dir).is_file()

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
