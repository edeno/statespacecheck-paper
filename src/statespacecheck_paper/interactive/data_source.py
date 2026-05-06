"""In-process data source for the Figure 4 interactive viewer.

``Figure4DataSource`` wraps the on-disk cache produced by
``cache.py`` and exposes the windowed-read API consumed by the
viewer panels:

- A small set of in-RAM arrays (time grid, animal linear position,
  place fields, place-field peaks, per-cell spike times, full event
  Parquet) — all together ~50 MB.
- Lazy ``xarray.Dataset`` handles for the per-model Zarr stores so
  posterior / log-likelihood reads pull only the chunks that overlap
  the current view.

The hot-path methods (``window_indices``, ``load_posterior``,
``load_likelihood``, ``events_in_window``, ``slice_at_index``) accept
plain Python ``slice`` and ``int`` arguments and return raw NumPy
float32/float64 arrays. No xarray on the call path beyond the chunk
read, so the viewer can hand results straight to ``pyqtgraph.ImageItem``
or ``PlotCurveItem.setData`` without any object-creation overhead.

The slice panel uses ``slice_at_index`` for its 1D posterior /
likelihood curves and a small ring buffer that the viewer maintains; the
data source just hands it the requested 1D row.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import xarray as xr
import zarr
from numpy.typing import NDArray

from . import cache as cache_mod

ModelName = cache_mod.ModelName


@dataclass(frozen=True)
class CacheLayout:
    """Resolved on-disk paths for a single model's cache + shared sidecars."""

    zarr: Path
    events: Path
    place_fields: Path
    meta: Path
    spike_times: Path

    @classmethod
    def for_model(cls, cache_dir: Path, model: ModelName) -> CacheLayout:
        paths = cache_mod.cache_paths(cache_dir, model)
        return cls(
            zarr=paths["zarr"],
            events=paths["events"],
            place_fields=paths["place_fields"],
            meta=cache_mod.meta_path(cache_dir),
            spike_times=cache_mod.spike_times_path(cache_dir),
        )

    def assert_exists(self) -> None:
        for label, path in (
            ("zarr", self.zarr),
            ("events", self.events),
            ("place_fields", self.place_fields),
            ("meta", self.meta),
            ("spike_times", self.spike_times),
        ):
            if not path.exists():
                raise FileNotFoundError(f"Missing cache artifact ({label}): {path}")


class Figure4DataSource:
    """Lazy data source backed by the cache produced by ``cache.build``.

    Construction is cheap: it reads the small sidecars (~50 MB total)
    and opens the Zarr store as a lazy ``xarray.Dataset``. The full
    posterior / log-likelihood arrays are never realized in memory.

    Parameters
    ----------
    cache_dir : Path
        Directory containing the cache produced by ``cache.build``.
    model : {"continuous", "contfrag"}
        Which model's Zarr store to open.

    Attributes
    ----------
    model : str
        The active model name.
    time : np.ndarray, shape (n_time,), float64
        Decoder time grid (absolute seconds).
    linear_position : np.ndarray, shape (n_time,), float64
        Animal linear position at each decoder time bin.
    place_fields : np.ndarray, shape (n_cells, n_states * n_interior), float32
        Place-field firing rates (interior bins only, concatenated
        across states for multi-state classifiers).
    position_bins : np.ndarray, shape (n_interior,), float64
        Per-state interior position grid (1D, identical across states).
    place_field_peaks : np.ndarray, shape (n_cells,), float64
        Place-field peak position for each cell, used to sort the raster.
    spike_times : list[np.ndarray]
        Per-cell spike-time arrays (float64).
    events : pandas.DataFrame
        Sorted event table with columns ``time`` (f64), ``cell_id`` (i32),
        ``event_hpd_overlap`` (f32), ``event_kl_divergence`` (f32),
        ``event_spike_prob`` (f32). Indexed by row position.
    n_cells : int
    n_time : int
    n_states : int
        Number of state slots in ``place_fields`` (1 for Continuous, 2
        for ContFrag).
    n_interior : int
        Number of interior position bins per state.
    n_state_bins : int
        Total interior state bins across states (``n_states * n_interior``).
    """

    POSTERIOR_VAR = "predictive_posterior"
    LIKELIHOOD_VAR = "log_likelihood"

    def __init__(self, cache_dir: Path | str, model: ModelName) -> None:
        self._cache_dir = Path(cache_dir)
        self.model: ModelName = model

        self._layout = CacheLayout.for_model(self._cache_dir, model)
        self._layout.assert_exists()

        # Open the Zarr store directly for the hot-path window reads.
        # Direct zarr access (no dask) avoids the async chunk-load
        # races we saw with ``xarray.open_zarr`` during test teardown.
        self._zarr_group: zarr.Group = zarr.open_group(str(self._layout.zarr), mode="r")
        # xarray is still convenient for one-shot metadata pulls
        # (position coord, state_bins size). We close it immediately
        # after extracting what we need so no dask graph survives.
        meta_ds = xr.open_zarr(self._layout.zarr, consolidated=True)

        meta = np.load(self._layout.meta)
        self.time: NDArray[np.float64] = np.asarray(meta["time"], dtype=np.float64)
        self.linear_position: NDArray[np.float64] = np.asarray(
            meta["linear_position"], dtype=np.float64
        )
        self.n_cells = int(meta["n_cells"])

        pfs = np.load(self._layout.place_fields)
        self.place_fields: NDArray[np.float32] = np.asarray(pfs["place_fields"], dtype=np.float32)
        self.position_bins: NDArray[np.float64] = np.asarray(pfs["position_bins"], dtype=np.float64)
        self.place_field_peaks: NDArray[np.float64] = np.asarray(
            pfs["place_field_peaks"], dtype=np.float64
        )

        spike_arr = np.load(self._layout.spike_times, allow_pickle=True)
        if spike_arr.dtype != object:
            raise ValueError(f"Expected object-dtype spike-times array, got {spike_arr.dtype}")
        self.spike_times: list[NDArray[np.float64]] = [
            np.asarray(st, dtype=np.float64) for st in spike_arr
        ]

        self.events: pd.DataFrame = pd.read_parquet(self._layout.events)
        if not self.events["time"].is_monotonic_increasing:
            self.events = self.events.sort_values("time", kind="mergesort").reset_index(drop=True)

        # Pre-extracted NumPy views over the events frame. The viewer's
        # per-tick live readout indexes these directly to avoid the
        # cost of ``DataFrame.iloc`` row construction on every UI tick.
        self.event_times: NDArray[np.float64] = self.events["time"].to_numpy(
            dtype=np.float64, copy=False
        )
        self.event_cell_ids: NDArray[np.int32] = self.events["cell_id"].to_numpy(
            dtype=np.int32, copy=False
        )
        self.event_hpd_overlap: NDArray[np.float32] = self.events["event_hpd_overlap"].to_numpy(
            dtype=np.float32, copy=False
        )
        self.event_kl_divergence: NDArray[np.float32] = self.events["event_kl_divergence"].to_numpy(
            dtype=np.float32, copy=False
        )
        self.event_spike_prob: NDArray[np.float32] = self.events["event_spike_prob"].to_numpy(
            dtype=np.float32, copy=False
        )

        self._validate_consistency()

        # Decoder time-bin index for each event. Used by
        # ``cells_at_index`` to find which cells fired in a given
        # time bin without re-bisecting the time grid per call.
        self._time_arr_for_bin = np.asarray(self.time, dtype=np.float64)
        self.event_time_idx: NDArray[np.int64] = np.clip(
            np.searchsorted(self._time_arr_for_bin, self.event_times, side="right") - 1,
            0,
            max(self._time_arr_for_bin.shape[0] - 1, 0),
        ).astype(np.int64)

        self.n_time: int = int(self.time.shape[0])
        self.n_interior: int = int(self.position_bins.shape[0])

        # Total state bins along the Zarr ``state_bins`` axis. For a
        # Continuous classifier this is one state's full position grid;
        # for ContFrag it is ``n_states * n_pos_full``.
        self.n_state_bins: int = int(meta_ds[self.POSTERIOR_VAR].sizes["state_bins"])

        # Full (non-interior + interior) per-state position grid. The
        # Zarr's ``position`` non-dim coord on ``state_bins`` repeats
        # the same per-state grid across states; the unique sorted
        # values give the canonical 1D grid.
        position_coord = np.asarray(meta_ds["position"].values, dtype=np.float64)
        self.position_grid_full: NDArray[np.float64] = np.unique(position_coord)
        self.n_position_full: int = int(self.position_grid_full.shape[0])
        meta_ds.close()

        # Interior mask in per-state position-grid coordinates (shape
        # ``(n_position_full,)``). Used by ``SlicePanel`` to find which
        # entries of a posterior row are real vs. NaN-filled by the
        # cache.
        interior_mask_full = np.asarray(pfs["interior_mask"], dtype=bool)
        # ``interior_mask`` saved by the cache is concatenated across
        # states; collapse to per-state (the cache verifies states are
        # identical).
        per_state = max(1, interior_mask_full.shape[0] // max(self.n_position_full, 1))
        self.interior_mask: NDArray[np.bool_] = interior_mask_full.reshape(
            per_state, self.n_position_full
        )[0].copy()
        self.n_states: int = max(1, self.n_state_bins // max(self.n_position_full, 1))

        # Direct zarr arrays for the hot path.
        self._post_arr: zarr.Array = self._zarr_group[self.POSTERIOR_VAR]
        self._loglik_arr: zarr.Array = self._zarr_group[self.LIKELIHOOD_VAR]

    # ------------------------------------------------------------------
    # Consistency / sanity
    # ------------------------------------------------------------------

    def _validate_consistency(self) -> None:
        n_cells = self.n_cells
        if self.place_fields.shape[0] != n_cells:
            raise ValueError(
                f"place_fields cell count {self.place_fields.shape[0]} != n_cells={n_cells}"
            )
        if len(self.spike_times) != n_cells:
            raise ValueError(f"spike_times length {len(self.spike_times)} != n_cells={n_cells}")
        if self.place_field_peaks.shape != (n_cells,):
            raise ValueError(
                f"place_field_peaks shape {self.place_field_peaks.shape} != ({n_cells},)"
            )
        if self.events["cell_id"].max() >= n_cells or self.events["cell_id"].min() < 0:
            raise ValueError(
                f"events cell_id out of range [0, {n_cells}); "
                f"got [{self.events['cell_id'].min()}, "
                f"{self.events['cell_id'].max()}]"
            )

    # ------------------------------------------------------------------
    # Window helpers
    # ------------------------------------------------------------------

    def window_indices(
        self,
        t_center: float,
        t_width: float,
    ) -> slice:
        """Return a half-open slice into ``time`` for ``[t-w/2, t+w/2]``.

        Both endpoints are clamped to the session range; the returned
        slice is always non-empty (with at least one sample) provided
        the session itself is non-empty.
        """
        if t_width <= 0:
            raise ValueError(f"t_width must be positive, got {t_width}")
        if self.n_time == 0:
            return slice(0, 0)
        half = t_width / 2.0
        i0 = int(np.searchsorted(self.time, t_center - half, side="left"))
        i1 = int(np.searchsorted(self.time, t_center + half, side="right"))
        i0 = max(0, min(i0, self.n_time - 1))
        i1 = max(i0 + 1, min(i1, self.n_time))
        return slice(i0, i1)

    def index_at_time(self, t: float) -> int:
        """Return the decoder-grid index closest to ``t``.

        Used by the slice panel and click handlers to map a clicked
        spike time onto the integer index used by ``slice_at_index``.
        """
        if self.n_time == 0:
            raise ValueError("Empty time grid")
        i = int(np.searchsorted(self.time, t, side="left"))
        if i <= 0:
            return 0
        if i >= self.n_time:
            return self.n_time - 1
        # Pick whichever neighbor is closer.
        if abs(self.time[i - 1] - t) <= abs(self.time[i] - t):
            return i - 1
        return i

    def cells_at_index(self, t_idx: int) -> NDArray[np.int32]:
        """Return the unique cell IDs that fired in time bin ``t_idx``.

        Two ``np.searchsorted`` calls on the precomputed
        ``event_time_idx`` (sorted because events are sorted by time)
        plus a small ``np.unique`` on the in-bin slice — sub-ms even
        for large event tables.
        """
        if self.event_time_idx.size == 0:
            return np.empty(0, dtype=np.int32)
        i0 = int(np.searchsorted(self.event_time_idx, t_idx, side="left"))
        i1 = int(np.searchsorted(self.event_time_idx, t_idx, side="right"))
        if i1 <= i0:
            return np.empty(0, dtype=np.int32)
        return np.unique(self.event_cell_ids[i0:i1])

    # ------------------------------------------------------------------
    # Hot-path readers
    # ------------------------------------------------------------------

    def load_posterior(self, sl: slice) -> NDArray[np.float32]:
        """Load the predictive posterior for the given time slice."""
        return self._read_window(self._post_arr, sl)

    def load_likelihood(self, sl: slice) -> NDArray[np.float32]:
        """Load the (log) likelihood for the given time slice.

        The cache stores the raw ``log_likelihood`` from the decoder.
        Callers that want a probability heatmap can ``np.exp`` the
        result; the slice panel exponentiates per-row when computing
        the likelihood curve.
        """
        return self._read_window(self._loglik_arr, sl)

    def slice_at_index(
        self,
        t_idx: int,
        *,
        which: Literal["posterior", "likelihood"] = "posterior",
    ) -> NDArray[np.float32]:
        """Return one 1D row (length ``n_state_bins``) at ``t_idx``.

        Hot-path call from the slice panel: a single Zarr row read.
        Typical chunk size (8192 along time) means at most one chunk
        fetch from disk, then in-cache for repeat reads.
        """
        if not 0 <= t_idx < self.n_time:
            raise IndexError(f"t_idx {t_idx} out of range [0, {self.n_time})")
        arr = self._post_arr if which == "posterior" else self._loglik_arr
        return np.asarray(arr[t_idx], dtype=np.float32)

    @staticmethod
    def _read_window(arr: zarr.Array, sl: slice) -> NDArray[np.float32]:
        return np.ascontiguousarray(arr[sl], dtype=np.float32)

    # ------------------------------------------------------------------
    # Events
    # ------------------------------------------------------------------

    def events_in_window(self, sl: slice) -> pd.DataFrame:
        """Return the slice of ``events`` whose ``time`` falls in ``sl``.

        ``sl`` is a slice into the decoder time grid; this method maps
        the slice's endpoint times to row positions in ``events`` via
        ``np.searchsorted`` for O(log n) lookup.
        """
        if self.events.empty:
            return self.events.iloc[0:0]
        i0_t = self.time[sl.start]
        # Use the last in-window index, not sl.stop (which is one past).
        i1_t = self.time[min(sl.stop, self.n_time) - 1]
        # ``event_times`` is the cached, monotonic-increasing column.
        i0 = int(np.searchsorted(self.event_times, i0_t, side="left"))
        i1 = int(np.searchsorted(self.event_times, i1_t, side="right"))
        return self.events.iloc[i0:i1]

    # ------------------------------------------------------------------
    # Resource management
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Close the underlying Zarr store handle.

        Direct ``zarr`` arrays do not have a ``close`` method; their
        store closes via garbage collection. Holding the references
        on the data source keeps them alive for the panel's lifetime.
        """
        # Drop array references so the underlying store can be GC'd.
        self._zarr_group = None
        self._post_arr = None
        self._loglik_arr = None

    def __enter__(self) -> Figure4DataSource:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()
