"""Tests for ``DecoderDataSource``.

The unit-style tests build a tiny synthetic cache (Zarr + Parquet +
sidecars) in ``tmp_path`` and exercise the windowed-read API. Two
real-data tests run only when viewer caches built by ``cache build``
exist under ``data/cache/`` and are skipped otherwise (they carry no
``slow`` marker). They check the recording's dimensions, the
Continuous-Fragmented cache's two discrete states, and that a 2 s window
read takes under 100 ms, a smoke check on the viewer's target of
p95 ≤ 50 ms for 20 s windows.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
import pytest
import zarr
from numpy.typing import NDArray

from statespacecheck_paper.interactive import cache as cache_mod
from statespacecheck_paper.interactive.data_source import (
    MAX_DECODED_CHUNKS_PER_VARIABLE,
    DecoderDataSource,
)
from statespacecheck_paper.paths import REPO_ROOT

from ._synthetic_cache import build_synthetic_cache

CACHE_DIR = REPO_ROOT / "data" / "cache"

# ``build_synthetic_cache`` writes 500 rows in chunks of this many rows.
SYNTHETIC_CHUNK_ROWS = 64


@pytest.fixture
def synthetic_cache(tmp_path: Path) -> Path:
    cache_dir = tmp_path / "cache"
    build_synthetic_cache(cache_dir)
    return cache_dir


def test_constructs_with_expected_shapes(synthetic_cache: Path) -> None:
    src = DecoderDataSource(synthetic_cache, model="continuous")
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


def test_public_arrays_are_write_protected(synthetic_cache: Path) -> None:
    """Panel code that pulls ``ds.event_kl_divergence`` (or any other
    pre-extracted array) into its own pipeline must not be able to
    mutate the array in place — the next tick would silently render
    the corruption."""
    src = DecoderDataSource(synthetic_cache, model="continuous")
    try:
        for name in (
            "time",
            "linear_position",
            "place_fields",
            "position_bins",
            "place_field_peaks",
            "event_times",
            "event_cell_ids",
            "event_hpd_overlap",
            "event_kl_divergence",
            "event_predictive_pvalue",
        ):
            arr = getattr(src, name)
            assert arr.flags.writeable is False, f"{name} is not write-protected"
        # Confirm the protection actually blocks mutation.
        with pytest.raises(ValueError, match="read-only|assignment destination"):
            src.event_kl_divergence[0] = 999.0
    finally:
        src.close()


def test_static_model_event_likelihood_is_computed_from_normalized_place_field(
    synthetic_cache: Path,
) -> None:
    """Static real models use the same event likelihood as diagnostics."""
    src = DecoderDataSource(synthetic_cache, model="continuous")
    try:
        assert src.event_likelihood is None
        event_idx = 0
        cell_id = int(src.event_cell_ids[event_idx])
        place_field = np.asarray(src.place_fields[cell_id], dtype=np.float64)
        expected = place_field / place_field.sum()
        np.testing.assert_allclose(
            src.event_likelihood_at(event_idx, cell_id),
            expected.astype(np.float32),
        )
    finally:
        src.close()


def test_window_indices_clamps_and_returns_nonempty(synthetic_cache: Path) -> None:
    with DecoderDataSource(synthetic_cache, model="continuous") as src:
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
    with DecoderDataSource(synthetic_cache, model="continuous") as src:
        with pytest.raises(ValueError, match="t_width must be positive"):
            src.window_indices(t_center=src.time[10], t_width=0.0)


def test_index_at_time_returns_nearest_neighbor(synthetic_cache: Path) -> None:
    with DecoderDataSource(synthetic_cache, model="continuous") as src:
        # Exact match.
        assert src.index_at_time(src.time[123]) == 123
        # Halfway between two samples → either is acceptable, but must be
        # one of the two adjacent indices.
        midpoint = 0.5 * (src.time[10] + src.time[11])
        assert src.index_at_time(midpoint) in (10, 11)
        # Out-of-range clamps.
        assert src.index_at_time(src.time[0] - 1.0) == 0
        assert src.index_at_time(src.time[-1] + 1.0) == src.n_time - 1


def test_load_predictive_returns_window_shape_and_dtype(synthetic_cache: Path) -> None:
    with DecoderDataSource(synthetic_cache, model="continuous") as src:
        sl = slice(50, 150)
        predictive = src.load_predictive(sl)
        assert predictive.shape == (100, src.n_state_bins)
        assert predictive.dtype == np.float32
        assert predictive.flags["C_CONTIGUOUS"]


def test_load_log_likelihood_returns_window(synthetic_cache: Path) -> None:
    with DecoderDataSource(synthetic_cache, model="continuous") as src:
        sl = slice(0, 64)
        loglik = src.load_log_likelihood(sl)
        assert loglik.shape == (64, src.n_state_bins)
        assert loglik.dtype == np.float32


def test_slice_at_index_matches_load_predictive_row(synthetic_cache: Path) -> None:
    with DecoderDataSource(synthetic_cache, model="continuous") as src:
        sl = slice(40, 60)
        predictive = src.load_predictive(sl)
        for offset in [0, 5, 19]:
            row = src.slice_at_index(sl.start + offset, which="predictive")
            np.testing.assert_array_equal(row, predictive[offset])


def test_slice_at_index_log_likelihood_branch(synthetic_cache: Path) -> None:
    with DecoderDataSource(synthetic_cache, model="continuous") as src:
        sl = slice(80, 96)
        loglik = src.load_log_likelihood(sl)
        row = src.slice_at_index(sl.start + 7, which="log_likelihood")
        np.testing.assert_array_equal(row, loglik[7])
        # The row is a log likelihood, so the selector says so.
        with pytest.raises(ValueError, match="Unknown slice variant: 'likelihood'"):
            src.slice_at_index(sl.start, which="likelihood")


def test_slice_at_index_raises_for_out_of_range(synthetic_cache: Path) -> None:
    with DecoderDataSource(synthetic_cache, model="continuous") as src:
        with pytest.raises(IndexError):
            src.slice_at_index(-1)
        with pytest.raises(IndexError):
            src.slice_at_index(src.n_time)


def _direct_array(cache_dir: Path, name: str) -> NDArray[Any]:
    """Read a whole variable straight from the store, past the decoded-chunk cache."""
    return zarr.open_group(str(cache_dir / "figure04_continuous.zarr"), mode="r")[name][:]


class _CountingZarrArray:
    """Count the decodes a data source asks of a Zarr array."""

    def __init__(self, arr: zarr.Array) -> None:
        self._arr = arr
        self.reads = 0

    def __getitem__(self, key: slice) -> NDArray[Any]:
        self.reads += 1
        return self._arr[key]


def _count_predictive_reads(src: DecoderDataSource) -> _CountingZarrArray:
    chunks = src._predictive_chunks  # noqa: SLF001
    counter = _CountingZarrArray(chunks._arr)  # noqa: SLF001
    chunks._arr = counter  # noqa: SLF001
    return counter


def _cached_chunk_indices(src: DecoderDataSource) -> list[int]:
    """Indices of the predictive chunks held, least recently used first."""
    return list(src._predictive_chunks._chunks)  # noqa: SLF001


@pytest.mark.parametrize(
    ("which", "name"),
    [
        ("predictive", "predictive_posterior"),
        ("log_likelihood", "log_likelihood"),
        ("smoothed", "acausal_posterior"),
    ],
)
@pytest.mark.parametrize(
    "sl",
    [
        pytest.param(slice(0, 1), id="first-row"),
        pytest.param(slice(0, SYNTHETIC_CHUNK_ROWS), id="exactly-one-chunk"),
        pytest.param(slice(60, 70), id="across-one-boundary"),
        pytest.param(slice(63, 65), id="two-rows-at-a-boundary"),
        pytest.param(slice(10, 300), id="across-several-boundaries"),
        pytest.param(slice(490, 500), id="short-last-chunk"),
        pytest.param(slice(490, 10_000), id="past-the-end"),
        pytest.param(slice(-5, None), id="open-ended"),
        pytest.param(slice(70, 70), id="empty"),
    ],
)
def test_chunked_reads_equal_the_direct_array(
    synthetic_cache: Path,
    which: Literal["predictive", "log_likelihood", "smoothed"],
    name: str,
    sl: slice,
) -> None:
    direct = _direct_array(synthetic_cache, name)
    with DecoderDataSource(synthetic_cache, model="continuous") as src:
        load = {
            "predictive": src.load_predictive,
            "log_likelihood": src.load_log_likelihood,
            "smoothed": src.load_smoothed,
        }[which]
        window = load(sl)
        assert window.dtype == np.float32
        assert window.flags.c_contiguous
        assert window.tobytes() == direct[sl].tobytes()
        assert window.shape == direct[sl].shape
        for t_idx in range(*sl.indices(src.n_time)):
            row = src.slice_at_index(t_idx, which=which)
            assert row.dtype == np.float32
            assert row.tobytes() == direct[t_idx].tobytes()


def test_window_read_rejects_a_stepped_slice(synthetic_cache: Path) -> None:
    with DecoderDataSource(synthetic_cache, model="continuous") as src:
        with pytest.raises(ValueError, match="step 1"):
            src.load_predictive(slice(0, 10, 2))


def test_repeated_reads_are_served_from_decoded_chunks(synthetic_cache: Path) -> None:
    with DecoderDataSource(synthetic_cache, model="continuous") as src:
        counter = _count_predictive_reads(src)
        first = src.load_predictive(slice(60, 70))
        assert counter.reads == 2  # one decode per chunk the window touches
        assert _cached_chunk_indices(src) == [0, 1]

        np.testing.assert_array_equal(src.load_predictive(slice(60, 70)), first)
        np.testing.assert_array_equal(src.slice_at_index(65), first[5])
        src.load_predictive(slice(0, 128))
        assert counter.reads == 2

        src.slice_at_index(2 * SYNTHETIC_CHUNK_ROWS)
        assert counter.reads == 3


def test_decoded_chunks_evict_the_least_recently_used(synthetic_cache: Path) -> None:
    with DecoderDataSource(synthetic_cache, model="continuous") as src:
        counter = _count_predictive_reads(src)
        n_chunks = int(np.ceil(src.n_time / SYNTHETIC_CHUNK_ROWS))
        assert n_chunks > MAX_DECODED_CHUNKS_PER_VARIABLE

        for index in range(n_chunks):
            src.slice_at_index(index * SYNTHETIC_CHUNK_ROWS)
            assert len(_cached_chunk_indices(src)) <= MAX_DECODED_CHUNKS_PER_VARIABLE
        assert counter.reads == n_chunks
        newest = list(range(n_chunks - MAX_DECODED_CHUNKS_PER_VARIABLE, n_chunks))
        assert _cached_chunk_indices(src) == newest

        # Rereading the oldest chunk protects it from the next eviction.
        src.slice_at_index(newest[0] * SYNTHETIC_CHUNK_ROWS)
        assert counter.reads == n_chunks
        src.slice_at_index(0)
        assert _cached_chunk_indices(src) == [*newest[2:], newest[0], 0]

        # An evicted chunk reads back correctly, at the cost of another decode.
        direct = _direct_array(synthetic_cache, "predictive_posterior")
        reads = counter.reads
        assert src.slice_at_index(newest[1] * SYNTHETIC_CHUNK_ROWS).tobytes() == (
            direct[newest[1] * SYNTHETIC_CHUNK_ROWS].tobytes()
        )
        assert counter.reads == reads + 1


def test_mutating_a_returned_array_does_not_corrupt_later_reads(synthetic_cache: Path) -> None:
    direct = _direct_array(synthetic_cache, "predictive_posterior")
    with DecoderDataSource(synthetic_cache, model="continuous") as src:
        window = src.load_predictive(slice(10, 20))
        whole_chunk = src.load_predictive(slice(0, SYNTHETIC_CHUNK_ROWS))
        row = src.slice_at_index(15)
        for returned in (window, whole_chunk, row):
            assert returned.flags.writeable
            assert returned.flags.owndata
            returned[...] = -1.0

        counter = _count_predictive_reads(src)
        assert src.load_predictive(slice(10, 20)).tobytes() == direct[10:20].tobytes()
        assert src.slice_at_index(15).tobytes() == direct[15].tobytes()
        assert counter.reads == 0  # the correct bytes came from the cache


def test_decoded_chunks_themselves_are_read_only(synthetic_cache: Path) -> None:
    with DecoderDataSource(synthetic_cache, model="continuous") as src:
        src.slice_at_index(0)
        chunk = src._predictive_chunks._chunks[0]  # noqa: SLF001
        with pytest.raises(ValueError, match="read-only"):
            chunk[0, 0] = 0.0


def test_concurrent_reads_stay_correct_and_bounded(synthetic_cache: Path) -> None:
    """Viewer worker threads and the GUI thread share each variable's chunks."""
    direct = _direct_array(synthetic_cache, "predictive_posterior")
    starts = np.random.default_rng(0).integers(0, 450, size=64).tolist()

    with DecoderDataSource(synthetic_cache, model="continuous") as src:

        def read(start: int) -> None:
            assert src.load_predictive(slice(start, start + 50)).tobytes() == (
                direct[start : start + 50].tobytes()
            )
            assert src.slice_at_index(start).tobytes() == direct[start].tobytes()
            assert len(_cached_chunk_indices(src)) <= MAX_DECODED_CHUNKS_PER_VARIABLE

        with ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(read, starts))


def test_threads_wanting_one_chunk_decode_it_once(synthetic_cache: Path) -> None:
    n_threads = 8
    barrier = threading.Barrier(n_threads)
    with DecoderDataSource(synthetic_cache, model="continuous") as src:
        counter = _count_predictive_reads(src)

        def read(t_idx: int) -> None:
            barrier.wait()
            src.slice_at_index(t_idx)

        with ThreadPoolExecutor(max_workers=n_threads) as pool:
            list(pool.map(read, range(n_threads)))
        assert counter.reads == 1


def test_close_releases_decoded_chunks(synthetic_cache: Path) -> None:
    src = DecoderDataSource(synthetic_cache, model="continuous")
    predictive_chunks = src._predictive_chunks  # noqa: SLF001
    loglik_chunks = src._loglik_chunks  # noqa: SLF001
    smoothed_chunks = src._smoothed_chunks  # noqa: SLF001
    assert smoothed_chunks is not None
    src.load_predictive(slice(0, 10))
    src.load_log_likelihood(slice(0, 10))
    src.load_smoothed(slice(0, 10))

    src.close()
    src.close()  # idempotent

    for chunks in (predictive_chunks, loglik_chunks, smoothed_chunks):
        assert not chunks._chunks  # noqa: SLF001
    with pytest.raises(TypeError):
        src.load_predictive(slice(0, 10))


def test_events_in_window_returns_sorted_subset(synthetic_cache: Path) -> None:
    with DecoderDataSource(synthetic_cache, model="continuous") as src:
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
    with DecoderDataSource(synthetic_cache, model="continuous") as src:
        # Slice a single sample at the very front: very few events expected.
        sl = slice(0, 1)
        events = src.events_in_window(sl)
        # Must not error and must respect the slice bounds.
        if not events.empty:
            assert events["time"].max() <= src.time[0]


def _rewrite_events(cache_dir: Path, edit: Callable[[pd.DataFrame], pd.DataFrame]) -> None:
    """Apply ``edit`` to the synthetic Continuous cache's event table in place."""
    path = cache_mod.recording_cache_paths(cache_dir, "continuous")["events"]
    edit(pd.read_parquet(path)).to_parquet(path, engine="pyarrow")


def test_cache_without_event_time_ind_is_rejected(synthetic_cache: Path) -> None:
    """A cache built before events stored their bins must be rebuilt, not re-binned."""
    _rewrite_events(synthetic_cache, lambda events: events.drop(columns="event_time_ind"))
    with pytest.raises(ValueError, match=r"records no event_time_ind; rebuild it with .*--force"):
        DecoderDataSource(synthetic_cache, model="continuous")


def _as_float(events: pd.DataFrame) -> pd.DataFrame:
    return events.astype({"event_time_ind": np.float64})


def _past_the_grid(events: pd.DataFrame) -> pd.DataFrame:
    events.loc[events.index[-1], "event_time_ind"] = 500
    return events


def _reversed(events: pd.DataFrame) -> pd.DataFrame:
    events["event_time_ind"] = events["event_time_ind"].to_numpy()[::-1].copy()
    return events


@pytest.mark.parametrize(
    ("edit", "message"),
    [
        (_as_float, "must be integer"),
        (_past_the_grid, "out of range"),
        (_reversed, "non-decreasing"),
    ],
    ids=["float", "out_of_range", "decreasing"],
)
def test_invalid_event_time_ind_is_rejected(
    synthetic_cache: Path,
    edit: Callable[[pd.DataFrame], pd.DataFrame],
    message: str,
) -> None:
    _rewrite_events(synthetic_cache, edit)
    with pytest.raises(ValueError, match=message):
        DecoderDataSource(synthetic_cache, model="continuous")


# ---------------------------------------------------------------------------
# Real-cache integration tests (skip when the cache has not been built yet).
# ---------------------------------------------------------------------------

REAL_CONT_CACHE_AVAILABLE = (CACHE_DIR / "figure04_continuous.zarr").exists()
REAL_CONTINUOUS_FRAGMENTED_CACHE_AVAILABLE = (
    CACHE_DIR / "figure04_continuous_fragmented.zarr"
).exists()


@pytest.mark.skipif(
    not REAL_CONT_CACHE_AVAILABLE,
    reason="Run `python -m statespacecheck_paper.interactive.cache build --data-dir data` first.",
)
def test_real_continuous_cache_window_read_latency() -> None:
    """A 2-second window read on the real cache must comfortably beat 50 ms.

    The latency target is window-load p95 ≤ 50 ms for 20 s windows; this test
    is a smaller smoke check on a 2 s window (1000 samples at 500 Hz).
    """
    import time

    src = DecoderDataSource(CACHE_DIR, model="continuous")
    try:
        assert src.n_time == 709321
        assert src.n_cells == 203
        assert src.n_state_bins == 256
        assert src.events.shape == (870018, 6)

        # Cold + warm reads both well under the smoke target.
        sl = src.window_indices(t_center=src.time[100_000], t_width=2.0)
        t0 = time.perf_counter()
        predictive = src.load_predictive(sl)
        cold_ms = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        predictive2 = src.load_predictive(sl)
        warm_ms = (time.perf_counter() - t0) * 1000

        assert predictive.shape[1] == 256
        assert predictive.shape == predictive2.shape
        # Generous bound on the 50 ms p95 target for 20 s windows.
        assert cold_ms < 100, f"cold read {cold_ms:.1f} ms"
        assert warm_ms < 100, f"warm read {warm_ms:.1f} ms"
    finally:
        src.close()


@pytest.mark.skipif(
    not REAL_CONTINUOUS_FRAGMENTED_CACHE_AVAILABLE,
    reason="Run `python -m statespacecheck_paper.interactive.cache build --data-dir data` first.",
)
def test_real_continuous_fragmented_cache_has_two_states() -> None:
    src = DecoderDataSource(CACHE_DIR, model="continuous_fragmented")
    try:
        assert src.n_time == 709321
        assert src.n_cells == 203
        assert src.n_states == 2
        assert src.n_state_bins == 512
        # The observation model is shared across the two discrete states, so
        # the place fields are positional only.
        assert src.place_fields.shape == (src.n_cells, src.n_interior)
    finally:
        src.close()
