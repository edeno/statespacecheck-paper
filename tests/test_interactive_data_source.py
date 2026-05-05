"""Tests for ``Figure4DataSource``.

The unit-style tests build a tiny synthetic cache (Zarr + Parquet +
sidecars) in ``tmp_path`` and exercise the windowed-read API. The
real-data integration test (marked ``slow``) opens the cache built by
``cache.build`` from the live intermediates and checks the same API
plus latency targets from the plan.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from statespacecheck_paper.interactive import cache as cache_mod
from statespacecheck_paper.interactive.data_source import Figure4DataSource

REPO_ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = REPO_ROOT / "data" / "cache"


# ---------------------------------------------------------------------------
# Synthetic cache builder (independent of the real fitted-model pickles).
# ---------------------------------------------------------------------------


def _build_synthetic_cache(
    cache_dir: Path,
    *,
    n_time: int = 500,
    n_states: int = 1,
    n_position: int = 16,
    n_cells: int = 4,
    n_events: int = 60,
    seed: int = 0,
) -> None:
    """Create a minimal, self-consistent Continuous-style cache on disk."""
    rng = np.random.default_rng(seed)
    cache_dir.mkdir(parents=True, exist_ok=True)

    n_state_bins = n_states * n_position
    state_names = [f"state_{i}" for i in range(n_states)]
    state_coord = np.array([state_names[i] for i in range(n_states) for _ in range(n_position)])
    position_grid = np.linspace(0.0, 100.0, n_position)
    position_coord = np.tile(position_grid, n_states)

    posterior = rng.dirichlet(np.ones(n_state_bins), size=n_time).astype(np.float32)
    log_likelihood = np.log(posterior + 1e-12).astype(np.float32)
    state_probs = np.ones((n_time,), dtype=np.float32)

    time = 1000.0 + np.arange(n_time, dtype=np.float64) * 0.002

    ds = xr.Dataset(
        data_vars={
            "predictive_posterior": (("time", "state_bins"), posterior),
            "log_likelihood": (("time", "state_bins"), log_likelihood),
            "acausal_state_probabilities": (("time",), state_probs),
        },
        coords={
            "time": ("time", time),
            "state_bins": ("state_bins", np.arange(n_state_bins, dtype=np.int64)),
            "state": ("state_bins", state_coord),
            "position": ("state_bins", position_coord),
        },
    )

    paths = cache_mod.cache_paths(cache_dir, "continuous")

    cache_mod._write_zarr_store(
        ds=ds,
        out_dir=paths["zarr"],
        time_chunk=64,
        pyramid_strides=(8, 64),
    )

    # Per-cell spike times: roughly uniform across the session.
    spike_times: list[np.ndarray] = []
    cumulative = 0
    for _cell in range(n_cells):
        n_spk = max(1, n_events // n_cells)
        t = np.sort(rng.uniform(time[0], time[-1], size=n_spk))
        spike_times.append(t.astype(np.float64))
        cumulative += n_spk

    # Event table: one row per spike across all cells, sorted by time.
    event_records: list[tuple[float, int, float, float, float]] = []
    for cell_id, t_arr in enumerate(spike_times):
        for t_val in t_arr:
            event_records.append(
                (
                    float(t_val),
                    int(cell_id),
                    float(rng.uniform(0.0, 1.0)),
                    float(rng.uniform(0.0, 5.0)),
                    float(rng.uniform(0.0, 1.0)),
                )
            )
    event_records.sort(key=lambda r: r[0])
    events = pd.DataFrame(
        event_records,
        columns=[
            "time",
            "cell_id",
            "event_hpd_overlap",
            "event_kl_divergence",
            "event_spike_prob",
        ],
    )
    events = events.astype(
        {
            "time": np.float64,
            "cell_id": np.int32,
            "event_hpd_overlap": np.float32,
            "event_kl_divergence": np.float32,
            "event_spike_prob": np.float32,
        }
    )
    events.to_parquet(paths["events"], engine="pyarrow", compression="zstd")

    # Place fields keyed to the position grid; peaks sample positions.
    place_fields = rng.uniform(0.0, 5.0, size=(n_cells, n_position)).astype(np.float32)
    peak_idx = np.argmax(place_fields, axis=1)
    np.savez(
        paths["place_fields"],
        place_fields=place_fields,
        interior_mask=np.ones(n_state_bins, dtype=bool),
        position_bins=position_grid.astype(np.float64),
        place_field_peaks=position_grid[peak_idx].astype(np.float64),
    )

    np.savez(
        cache_mod.meta_path(cache_dir),
        time=time,
        linear_position=rng.uniform(0.0, 100.0, size=n_time).astype(np.float64),
        n_cells=np.int64(n_cells),
    )

    container = np.empty(n_cells, dtype=object)
    for i, st in enumerate(spike_times):
        container[i] = st
    np.save(cache_mod.spike_times_path(cache_dir), container, allow_pickle=True)


@pytest.fixture
def synthetic_cache(tmp_path: Path) -> Path:
    cache_dir = tmp_path / "cache"
    _build_synthetic_cache(cache_dir)
    return cache_dir


def test_constructs_with_expected_shapes(synthetic_cache: Path) -> None:
    src = Figure4DataSource(synthetic_cache, model="continuous")
    try:
        assert src.n_time == 500
        assert src.n_cells == 4
        assert src.n_interior == 16
        assert src.n_states == 1
        assert src.n_state_bins == 16
        assert src.time.shape == (500,)
        assert src.linear_position.shape == (500,)
        assert src.place_fields.shape == (4, 16)
        assert src.place_field_peaks.shape == (4,)
        assert len(src.spike_times) == 4
        assert src.events.shape[0] >= 1
    finally:
        src.close()


def test_window_indices_clamps_and_returns_nonempty(synthetic_cache: Path) -> None:
    with Figure4DataSource(synthetic_cache, model="continuous") as src:
        # Center inside the session, narrow window.
        sl = src.window_indices(t_center=src.time[100], t_width=0.020)
        assert sl.start <= 100 <= sl.stop
        assert sl.stop > sl.start

        # Center far before the session start: clamp to the front.
        sl_lo = src.window_indices(t_center=src.time[0] - 1000.0, t_width=0.020)
        assert sl_lo.start == 0
        assert sl_lo.stop >= 1

        # Center far past the end: clamp to the back.
        sl_hi = src.window_indices(t_center=src.time[-1] + 1000.0, t_width=0.020)
        assert sl_hi.stop == src.n_time
        assert sl_hi.start <= src.n_time - 1


def test_window_indices_rejects_nonpositive_width(synthetic_cache: Path) -> None:
    with Figure4DataSource(synthetic_cache, model="continuous") as src:
        with pytest.raises(ValueError, match="t_width must be positive"):
            src.window_indices(t_center=src.time[10], t_width=0.0)


def test_index_at_time_returns_nearest_neighbor(synthetic_cache: Path) -> None:
    with Figure4DataSource(synthetic_cache, model="continuous") as src:
        # Exact match.
        assert src.index_at_time(src.time[123]) == 123
        # Halfway between two samples → either is acceptable, but must be
        # one of the two adjacent indices.
        midpoint = 0.5 * (src.time[10] + src.time[11])
        assert src.index_at_time(midpoint) in (10, 11)
        # Out-of-range clamps.
        assert src.index_at_time(src.time[0] - 1.0) == 0
        assert src.index_at_time(src.time[-1] + 1.0) == src.n_time - 1


def test_load_posterior_returns_window_shape_and_dtype(synthetic_cache: Path) -> None:
    with Figure4DataSource(synthetic_cache, model="continuous") as src:
        sl = slice(50, 150)
        post = src.load_posterior(sl)
        assert post.shape == (100, src.n_state_bins)
        assert post.dtype == np.float32
        assert post.flags["C_CONTIGUOUS"]


def test_load_likelihood_returns_window(synthetic_cache: Path) -> None:
    with Figure4DataSource(synthetic_cache, model="continuous") as src:
        sl = slice(0, 64)
        loglik = src.load_likelihood(sl)
        assert loglik.shape == (64, src.n_state_bins)
        assert loglik.dtype == np.float32


def test_slice_at_index_matches_load_posterior_row(synthetic_cache: Path) -> None:
    with Figure4DataSource(synthetic_cache, model="continuous") as src:
        sl = slice(40, 60)
        post = src.load_posterior(sl)
        for offset in [0, 5, 19]:
            row = src.slice_at_index(sl.start + offset, which="posterior")
            np.testing.assert_array_equal(row, post[offset])


def test_slice_at_index_likelihood_branch(synthetic_cache: Path) -> None:
    with Figure4DataSource(synthetic_cache, model="continuous") as src:
        sl = slice(80, 96)
        loglik = src.load_likelihood(sl)
        row = src.slice_at_index(sl.start + 7, which="likelihood")
        np.testing.assert_array_equal(row, loglik[7])


def test_slice_at_index_raises_for_out_of_range(synthetic_cache: Path) -> None:
    with Figure4DataSource(synthetic_cache, model="continuous") as src:
        with pytest.raises(IndexError):
            src.slice_at_index(-1)
        with pytest.raises(IndexError):
            src.slice_at_index(src.n_time)


def test_events_in_window_returns_sorted_subset(synthetic_cache: Path) -> None:
    with Figure4DataSource(synthetic_cache, model="continuous") as src:
        sl = src.window_indices(
            t_center=0.5 * (src.time[100] + src.time[300]),
            t_width=src.time[300] - src.time[100],
        )
        events = src.events_in_window(sl)
        if not events.empty:
            t0 = src.time[sl.start]
            t1 = src.time[sl.stop - 1]
            assert events["time"].min() >= t0
            assert events["time"].max() <= t1
            assert events["time"].is_monotonic_increasing


def test_events_in_window_empty_when_outside(synthetic_cache: Path) -> None:
    with Figure4DataSource(synthetic_cache, model="continuous") as src:
        # Slice a single sample at the very front: very few events expected.
        sl = slice(0, 1)
        events = src.events_in_window(sl)
        # Must not error and must respect the slice bounds.
        if not events.empty:
            assert events["time"].max() <= src.time[0]


# ---------------------------------------------------------------------------
# Real-cache integration tests (skip when the cache has not been built yet).
# ---------------------------------------------------------------------------

REAL_CONT_CACHE_AVAILABLE = (CACHE_DIR / "figure04_continuous.zarr").exists()
REAL_CONTFRAG_CACHE_AVAILABLE = (CACHE_DIR / "figure04_contfrag.zarr").exists()


@pytest.mark.skipif(
    not REAL_CONT_CACHE_AVAILABLE,
    reason="Run `python -m statespacecheck_paper.interactive.cache build "
    "--model continuous --data-dir data` first.",
)
def test_real_continuous_cache_window_read_latency() -> None:
    """A 2-second window read on the real cache must comfortably beat 50 ms.

    Plan target is window-load p95 ≤ 50 ms for 20 s windows; this test
    is a smaller smoke check on a 2 s window (1000 samples at 500 Hz).
    """
    import time

    src = Figure4DataSource(CACHE_DIR, model="continuous")
    try:
        assert src.n_time == 709321
        assert src.n_cells == 203
        assert src.n_state_bins == 256
        assert src.events.shape == (870018, 5)

        # Cold + warm reads both well under the smoke target.
        sl = src.window_indices(t_center=src.time[100_000], t_width=2.0)
        t0 = time.perf_counter()
        post = src.load_posterior(sl)
        cold_ms = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        post2 = src.load_posterior(sl)
        warm_ms = (time.perf_counter() - t0) * 1000

        assert post.shape[1] == 256
        assert post.shape == post2.shape
        # Generous bound — the plan asks for 50 ms p95 on 20 s windows.
        assert cold_ms < 100, f"cold read {cold_ms:.1f} ms"
        assert warm_ms < 100, f"warm read {warm_ms:.1f} ms"
    finally:
        src.close()


@pytest.mark.skipif(
    not REAL_CONTFRAG_CACHE_AVAILABLE,
    reason="Run `python -m statespacecheck_paper.interactive.cache build "
    "--model contfrag --data-dir data` first.",
)
def test_real_contfrag_cache_has_two_states() -> None:
    src = Figure4DataSource(CACHE_DIR, model="contfrag")
    try:
        assert src.n_time == 709321
        assert src.n_cells == 203
        assert src.n_states == 2
        assert src.n_state_bins == 512
        # Place fields are concatenated across states.
        assert src.place_fields.shape[1] == src.n_states * src.n_interior
    finally:
        src.close()
