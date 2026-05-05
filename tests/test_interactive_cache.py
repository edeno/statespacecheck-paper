"""Smoke tests for the Figure 4 viewer cache builder.

The pure-data tests exercise the array-shape logic (max-pool, pyramid
build, event-table assembly, Zarr writer) on synthetic inputs and must
pass without ``non_local_detector`` or any real recording files.

The real-data integration test is skipped when
``data/intermediates/cont_results.nc`` (or the matching model pickle and
raw spike-times pickle) is missing — it only runs in a fully provisioned
checkout.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from statespacecheck_paper.interactive import cache as cache_mod

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


@pytest.mark.parametrize("stride", [1, 2, 8, 64])
def test_max_pool_time_shape(stride: int) -> None:
    """Max-pool along axis 0 with the expected output length."""
    n_time, n_state_bins = 1000, 16
    arr = np.random.default_rng(42).standard_normal((n_time, n_state_bins), dtype=np.float32)
    pooled = cache_mod._max_pool_time(arr, stride)
    expected_n = int(np.ceil(n_time / stride))
    assert pooled.shape == (expected_n, n_state_bins)
    assert pooled.dtype == np.float32


def test_max_pool_time_preserves_max() -> None:
    """Pooled value at each bucket equals the max within that bucket."""
    arr = np.arange(20, dtype=np.float32).reshape(20, 1)
    pooled = cache_mod._max_pool_time(arr, stride=4)
    # Buckets: [0..3]→3, [4..7]→7, [8..11]→11, [12..15]→15, [16..19]→19
    np.testing.assert_array_equal(pooled.flatten(), [3, 7, 11, 15, 19])


def test_build_pyramid_levels_returns_one_array_per_stride() -> None:
    arr = np.random.default_rng(1).standard_normal((512, 8), dtype=np.float32)
    pyramid = cache_mod._build_pyramid_levels(arr, strides=(1, 8, 64))
    assert set(pyramid.keys()) == {1, 8, 64}
    assert pyramid[1].shape == (512, 8)
    assert pyramid[8].shape == (64, 8)
    assert pyramid[64].shape == (8, 8)


def test_events_dataframe_sorts_by_time_and_validates_cell_id() -> None:
    diagnostics: dict[str, np.ndarray] = {
        "event_time": np.array([2.0, 1.0, 3.0], dtype=np.float64),
        "event_cell_ind": np.array([0, 2, 1], dtype=np.int64),
        "event_hpd_overlap": np.array([0.1, 0.2, 0.3], dtype=np.float32),
        "event_kl_divergence": np.array([1.0, 2.0, 3.0], dtype=np.float32),
        "event_spike_prob": np.array([0.5, 0.4, 0.3], dtype=np.float32),
    }
    df = cache_mod._events_dataframe(diagnostics, n_cells=3)
    assert list(df.columns) == [
        "time",
        "cell_id",
        "event_hpd_overlap",
        "event_kl_divergence",
        "event_spike_prob",
    ]
    assert df["time"].tolist() == [1.0, 2.0, 3.0]
    assert df["cell_id"].tolist() == [2, 0, 1]
    assert df["cell_id"].dtype == np.int32


def test_events_dataframe_rejects_out_of_range_cell_id() -> None:
    diagnostics: dict[str, np.ndarray] = {
        "event_time": np.array([1.0], dtype=np.float64),
        "event_cell_ind": np.array([5], dtype=np.int64),
        "event_hpd_overlap": np.array([0.0], dtype=np.float32),
        "event_kl_divergence": np.array([0.0], dtype=np.float32),
        "event_spike_prob": np.array([0.0], dtype=np.float32),
    }
    with pytest.raises(ValueError, match="event_cell_ind out of range"):
        cache_mod._events_dataframe(diagnostics, n_cells=3)


def test_write_zarr_store_roundtrips_arrays_and_pyramids(tmp_path: Path) -> None:
    """``_write_zarr_store`` writes full-res + pyramid arrays that read back identically."""
    ds = _synthetic_results_dataset(n_time=200, n_states=2, n_position=8)
    out_dir = tmp_path / "cache.zarr"

    shapes = cache_mod._write_zarr_store(
        ds=ds,
        out_dir=out_dir,
        time_chunk=64,
        pyramid_strides=(8, 64),
    )

    # Shapes reported back match the data we wrote.
    assert shapes["predictive_posterior"] == (200, 16)
    assert shapes["predictive_posterior_pyramid_8"] == (25, 16)
    assert shapes["predictive_posterior_pyramid_64"] == (4, 16)
    assert shapes["log_likelihood_pyramid_8"] == (25, 16)

    # Round-trip: the stored arrays match the in-memory ones.
    with xr.open_zarr(out_dir, consolidated=True) as readback:
        np.testing.assert_array_equal(
            readback["predictive_posterior"].values,
            ds["predictive_posterior"].values,
        )
        # The non-dim coords on state_bins survive.
        np.testing.assert_array_equal(readback["state"].values, ds["state"].values)
        np.testing.assert_array_equal(readback["position"].values, ds["position"].values)
        # Pyramid level matches what _max_pool_time would produce.
        expected_pyr_8 = cache_mod._max_pool_time(
            ds["predictive_posterior"].values.astype(np.float32), stride=8
        )
        np.testing.assert_array_equal(
            readback["predictive_posterior_pyramid_8"].values, expected_pyr_8
        )


def test_write_zarr_store_overwrites_existing(tmp_path: Path) -> None:
    """Re-writing the same path replaces the prior store."""
    ds = _synthetic_results_dataset(n_time=64, n_states=1, n_position=4)
    out_dir = tmp_path / "cache.zarr"
    cache_mod._write_zarr_store(ds=ds, out_dir=out_dir, time_chunk=32, pyramid_strides=(4,))
    cache_mod._write_zarr_store(ds=ds, out_dir=out_dir, time_chunk=16, pyramid_strides=(2, 8))
    with xr.open_zarr(out_dir, consolidated=True) as rb:
        assert "predictive_posterior_pyramid_2" in rb.data_vars
        assert "predictive_posterior_pyramid_8" in rb.data_vars
        # Old stride is gone.
        assert "predictive_posterior_pyramid_4" not in rb.data_vars


# ---------------------------------------------------------------------------
# Real-data integration test (skipped when intermediates are not available).
# ---------------------------------------------------------------------------

CONT_NC = INTERMEDIATES / "cont_results.nc"
CONT_MODEL_PKL = INTERMEDIATES / "cont_model.pkl"
RAW_SPIKES_PKL = RAW_DATA / f"{ANIMAL_DATE_EPOCH}_HPC_spike_times.pkl"
RAW_POSITION_PKL = RAW_DATA / f"{ANIMAL_DATE_EPOCH}_position_info.pkl"
RAW_TRACK_GRAPH = RAW_DATA / f"{ANIMAL_DATE_EPOCH}_track_graph.pkl"
RAW_LINEAR_EDGE_ORDER = RAW_DATA / f"{ANIMAL_DATE_EPOCH}_linear_edge_order.pkl"
RAW_LINEAR_EDGE_SPACING = RAW_DATA / f"{ANIMAL_DATE_EPOCH}_linear_edge_spacing.pkl"

REAL_DATA_AVAILABLE = all(
    p.exists()
    for p in [
        CONT_NC,
        CONT_MODEL_PKL,
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
def test_build_model_cache_continuous_integration(tmp_path: Path) -> None:
    """Build the Continuous-model cache from real data and verify shapes.

    This test takes several minutes and several GB of disk; it runs only
    when both the NetCDF results and the fitted-model pickle are present.
    """
    cache_dir = tmp_path / "cache"
    info = cache_mod.build_model_cache(
        model="continuous",
        intermediates_dir=INTERMEDIATES,
        raw_data_dir=RAW_DATA,
        cache_dir=cache_dir,
        animal_date_epoch=ANIMAL_DATE_EPOCH,
    )

    assert info["model"] == "continuous"
    assert info["n_time"] == 709321
    assert info["n_cells"] == 203
    # Continuous decoder: 256 state_bins (full); ~248 interior bins.
    assert info["n_state_bins_full_res"] == 256
    assert 240 <= info["n_position_bins"] <= 256
    # Spike count is in the high-800Ks per the inspection.
    assert 850000 <= info["n_events"] <= 900000

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
    assert events["cell_id"].between(0, info["n_cells"] - 1).all()
