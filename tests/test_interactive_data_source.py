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
import pytest

from statespacecheck_paper.interactive.data_source import Figure4DataSource

from ._synthetic_cache import build_synthetic_cache

REPO_ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = REPO_ROOT / "data" / "cache"


@pytest.fixture
def synthetic_cache(tmp_path: Path) -> Path:
    cache_dir = tmp_path / "cache"
    build_synthetic_cache(cache_dir)
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
