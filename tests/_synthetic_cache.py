"""Shared synthetic-cache builder for the interactive-viewer tests.

The five ``test_interactive_*.py`` files all want a tiny but
self-consistent cache (Zarr posterior + log-likelihood, Parquet event
table, place-fields .npz, meta + spike-times sidecars) to drive
``Figure4DataSource`` and ``Figure4Viewer`` without touching the real
~5 GB of decoder outputs. This module is the single source of truth.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

from statespacecheck_paper.interactive import cache as cache_mod


def build_synthetic_cache(
    cache_dir: Path,
    *,
    model: str = "continuous",
    n_states: int = 1,
    n_time: int = 500,
    n_position: int = 16,
    n_cells: int = 4,
    n_spikes_per_cell: int = 15,
    p_min: float = 0.001,
    seed: int = 0,
) -> None:
    """Write a minimal, self-consistent cache for the given model name.

    The cache is fully populated: a chunked Zarr store (with one
    pyramid level), a sorted Parquet event table, a place-fields
    ``.npz``, the meta sidecar, and the spike-times ``.npy``.

    Parameters
    ----------
    cache_dir : Path
        Directory to write into (created on demand).
    model : {"continuous", "contfrag"}
        Which model name's cache layout to produce.
    n_states : int
        State count (1 or 2). Multi-state caches use a multi-state
        ``acausal_state_probabilities`` with ``("time", "states")``.
    n_time, n_position, n_cells : int
        Synthetic data dimensions.
    n_spikes_per_cell : int
        Per-cell spike count for the events table.
    p_min : float
        Lower bound for the synthetic ``event_spike_prob`` column.
    seed : int
        RNG seed for reproducibility.
    """
    rng = np.random.default_rng(seed)
    n_state_bins = n_states * n_position

    state_names = [f"state_{i}" for i in range(n_states)]
    state_coord = np.array([state_names[i] for i in range(n_states) for _ in range(n_position)])
    position_grid = np.linspace(0.0, 100.0, n_position)
    position_coord = np.tile(position_grid, n_states)

    posterior = rng.dirichlet(np.ones(n_state_bins), size=n_time).astype(np.float32)
    log_likelihood = np.log(posterior + 1e-12).astype(np.float32)
    if n_states == 1:
        state_probs_var: tuple[Any, Any] = (
            ("time",),
            np.ones((n_time,), dtype=np.float32),
        )
    else:
        state_probs_var = (
            ("time", "states"),
            rng.dirichlet(np.ones(n_states), size=n_time).astype(np.float32),
        )
    time_arr = 1000.0 + np.arange(n_time, dtype=np.float64) * 0.002

    coords: dict[str, Any] = {
        "time": ("time", time_arr),
        "state_bins": ("state_bins", np.arange(n_state_bins, dtype=np.int64)),
        "state": ("state_bins", state_coord),
        "position": ("state_bins", position_coord),
    }
    if n_states > 1:
        coords["states"] = ("states", np.array(state_names))

    ds = xr.Dataset(
        data_vars={
            "predictive_posterior": (("time", "state_bins"), posterior),
            "log_likelihood": (("time", "state_bins"), log_likelihood),
            "acausal_state_probabilities": state_probs_var,
        },
        coords=coords,
    )

    paths = cache_mod.cache_paths(cache_dir, model)  # type: ignore[arg-type]
    cache_mod._write_zarr_store(ds=ds, out_dir=paths["zarr"], time_chunk=64)

    spike_times = [
        np.sort(rng.uniform(time_arr[0], time_arr[-1], size=n_spikes_per_cell)).astype(np.float64)
        for _ in range(n_cells)
    ]

    rows: list[tuple[float, int, float, float, float]] = []
    for cell_id, ts in enumerate(spike_times):
        for t_val in ts:
            rows.append(
                (
                    float(t_val),
                    int(cell_id),
                    float(rng.uniform(0.0, 1.0)),
                    float(rng.uniform(0.0, 5.0)),
                    float(rng.uniform(p_min, 1.0)),
                )
            )
    rows.sort(key=lambda r: r[0])

    events = pd.DataFrame(
        rows,
        columns=[
            "time",
            "cell_id",
            "event_hpd_overlap",
            "event_kl_divergence",
            "event_spike_prob",
        ],
    ).astype(
        {
            "time": np.float64,
            "cell_id": np.int32,
            "event_hpd_overlap": np.float32,
            "event_kl_divergence": np.float32,
            "event_spike_prob": np.float32,
        }
    )
    events.to_parquet(paths["events"], engine="pyarrow", compression="zstd")

    place_fields = rng.uniform(0.0, 5.0, size=(n_cells, n_state_bins)).astype(np.float32)
    peak_idx = np.argmax(place_fields[:, :n_position], axis=1)
    np.savez(
        paths["place_fields"],
        place_fields=place_fields,
        interior_mask=np.ones(n_state_bins, dtype=bool),
        position_bins=position_grid.astype(np.float64),
        place_field_peaks=position_grid[peak_idx].astype(np.float64),
    )

    np.savez(
        cache_mod.meta_path(cache_dir),
        time=time_arr,
        linear_position=rng.uniform(0.0, 100.0, size=n_time).astype(np.float64),
        n_cells=np.int64(n_cells),
    )

    container = np.empty(n_cells, dtype=object)
    for i, st in enumerate(spike_times):
        container[i] = st
    np.save(cache_mod.spike_times_path(cache_dir), container, allow_pickle=True)
