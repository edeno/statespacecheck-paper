"""Build the on-disk cache used by the Figure 4 interactive viewer.

The cache reformats the pre-computed decoder outputs in
``data/intermediates/`` into a layout that supports fast windowed reads:

- A Zarr store per model with chunked posterior / log-likelihood arrays
  (chunked along time, full position axis per chunk).
- A Parquet event table with one row per spike, sorted by time, holding
  the per-spike diagnostic metrics (HPD overlap, KL divergence, spike
  probability) plus the cell index.
- A small ``.npz`` sidecar with the time grid, animal linear position,
  per-cell place fields, and place-field peak positions.
- A second ``.npz`` sidecar with the per-cell spike-time arrays used by
  the raster panel.

Usage::

    python -m statespacecheck_paper.interactive.cache build \\
        --model continuous --data-dir DATA --cache-dir DATA/cache

See the package's ``__init__.py`` for the public surface.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

import joblib
import numpy as np
import pandas as pd
import xarray as xr
from numpy.typing import NDArray

from statespacecheck_paper.analysis import PerCellDiagnostics

ModelName = Literal["continuous", "contfrag"]
MODEL_NAMES: tuple[ModelName, ...] = ("continuous", "contfrag")

DEFAULT_TIME_CHUNK = 8192


@dataclass(frozen=True)
class ModelPaths:
    """File paths for one model's pre-computed inputs.

    Attributes
    ----------
    results_nc : Path
        NetCDF file with decoder outputs (predictive_posterior,
        log_likelihood, acausal_state_probabilities).
    model_pkl : Path
        Joblib pickle of the fitted decoder model.
    """

    results_nc: Path
    model_pkl: Path


def model_paths(intermediates_dir: Path, model: ModelName) -> ModelPaths:
    """Return canonical input paths for ``model`` under ``intermediates_dir``."""
    if model == "continuous":
        return ModelPaths(
            results_nc=intermediates_dir / "cont_results.nc",
            model_pkl=intermediates_dir / "cont_model.pkl",
        )
    if model == "contfrag":
        return ModelPaths(
            results_nc=intermediates_dir / "cont_frag_results.nc",
            model_pkl=intermediates_dir / "cont_frag_model.pkl",
        )
    raise ValueError(f"Unknown model: {model!r}")


def cache_paths(cache_dir: Path, model: ModelName) -> dict[str, Path]:
    """Return the on-disk cache layout for ``model`` under ``cache_dir``.

    Real-data caches are figure-4 specific (the ``figure04_`` prefix
    is meaningful — these files come out of the
    ``cont_results.nc`` / ``cont_frag_results.nc`` decoder runs).
    Simulated-data caches use a separate filename layout via
    ``simulated_cache_paths``.
    """
    return {
        "zarr": cache_dir / f"figure04_{model}.zarr",
        "events": cache_dir / f"figure04_{model}_events.parquet",
        "place_fields": cache_dir / f"figure04_{model}_place_fields.npz",
    }


def meta_path(cache_dir: Path) -> Path:
    """Path to the real-data (figure-4) meta sidecar.

    Both ``continuous`` and ``contfrag`` real-data caches share this
    sidecar — the recording session's time grid, animal linear position,
    and cell count are model-independent.
    """
    return cache_dir / "figure04_meta.npz"


def spike_times_path(cache_dir: Path) -> Path:
    """Path to the real-data per-cell spike-times sidecar (object-dtype .npy)."""
    return cache_dir / "figure04_spike_times.npy"


# ---------------------------------------------------------------------------
# Simulation cache layout
# ---------------------------------------------------------------------------
#
# The figure-3 simulation is a fundamentally different dataset than the
# figure-4 real-data caches: there's no recording session, no model
# choice (the simulation has its own forward filter built in), and no
# smoothed posterior. It uses its own filename prefix so it can coexist
# in a shared cache directory if desired.


def simulated_cache_paths(cache_dir: Path) -> dict[str, Path]:
    """Return the on-disk cache layout for the figure-3 simulated dataset."""
    return {
        "zarr": cache_dir / "simulation.zarr",
        "events": cache_dir / "simulation_events.parquet",
        "place_fields": cache_dir / "simulation_place_fields.npz",
    }


def simulated_meta_path(cache_dir: Path) -> Path:
    """Path to the simulated-dataset meta sidecar."""
    return cache_dir / "simulation_meta.npz"


def simulated_spike_times_path(cache_dir: Path) -> Path:
    """Path to the simulated-dataset per-cell spike-times sidecar."""
    return cache_dir / "simulation_spike_times.npy"


def _restore_state_bins_index(ds: xr.Dataset) -> xr.Dataset:
    """Re-attach the ``(state, position)`` MultiIndex on ``state_bins``.

    NetCDF round-trip drops the MultiIndex but preserves ``state`` and
    ``position`` as non-dim coords on ``state_bins``. Restoring it lets
    callers ``unstack("state_bins")`` and group by state cleanly.
    """
    if "state" in ds.coords and "position" in ds.coords:
        return cast(xr.Dataset, ds.set_index(state_bins=["state", "position"]))
    return ds


def _extract_place_fields_concat(model: Any) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
    """Thin re-export of the shared place-field concat helper.

    Kept for back-compat with any external script that imported the
    underscore-private name; new code should use
    ``real_data_analysis.extract_place_fields_concat`` directly.
    """
    from statespacecheck_paper.real_data_analysis import (  # noqa: PLC0415
        extract_place_fields_concat,
    )

    return extract_place_fields_concat(model)


def _events_dataframe(
    diagnostics: PerCellDiagnostics,
    n_cells: int,
) -> pd.DataFrame:
    """Convert per-spike diagnostic arrays into a sorted Parquet-friendly frame."""
    if diagnostics.event_time is None:
        raise ValueError(
            "PerCellDiagnostics.event_time is required when building the cache "
            "events frame; the simulated path leaves it None."
        )

    cell_id = np.asarray(diagnostics.event_cell_ind, dtype=np.int32)
    if cell_id.size and (cell_id.min() < 0 or cell_id.max() >= n_cells):
        raise ValueError(
            f"event_cell_ind out of range [0, {n_cells}); got [{cell_id.min()}, {cell_id.max()}]"
        )

    df = pd.DataFrame(
        {
            "time": np.asarray(diagnostics.event_time, dtype=np.float64),
            "cell_id": cell_id,
            "event_hpd_overlap": np.asarray(diagnostics.event_hpd_overlap, dtype=np.float32),
            "event_kl_divergence": np.asarray(diagnostics.event_kl_divergence, dtype=np.float32),
            "event_spike_prob": np.asarray(diagnostics.event_spike_prob, dtype=np.float32),
        }
    )
    df.sort_values("time", kind="mergesort", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def _write_zarr_store(
    *,
    ds: xr.Dataset,
    out_dir: Path,
    time_chunk: int,
) -> dict[str, tuple[int, ...]]:
    """Stream the decoder NetCDF into a chunked Zarr store.

    Writes ``predictive_posterior``, ``log_likelihood``, and
    ``acausal_state_probabilities`` (when present), chunked at
    ``time_chunk`` along the time axis so the viewer's window reads
    only touch one or two chunks. ``xarray.to_zarr`` streams chunk
    by chunk, so peak in-memory cost is bounded by the chunk size,
    not the full session.

    Returns the per-variable shapes for caller-side verification.

    Notes
    -----
    The input dataset is the original NetCDF round-trip — ``state_bins``
    is a plain integer dim with ``state`` and ``position`` as non-dim
    coords. Both round-trip cleanly through Zarr; the data_source
    restores the ``(state, position)`` MultiIndex on read.
    """
    if out_dir.exists():
        shutil.rmtree(out_dir)

    keep_vars = ["predictive_posterior", "log_likelihood"]
    # ``acausal_posterior`` is the smoothed distribution
    # ``p(x_t | y_{1:T})`` — included so the slice panel's top-plot
    # overlay can switch between predictive / filtered / smoothed.
    if "acausal_posterior" in ds.data_vars:
        keep_vars.append("acausal_posterior")
    if "acausal_state_probabilities" in ds.data_vars:
        keep_vars.append("acausal_state_probabilities")

    base = ds[keep_vars].chunk({"time": time_chunk})

    # Cast object-dtype string coords to fixed-width unicode so xarray
    # does not have to load them into memory to infer length on write.
    for coord_name in list(base.coords):
        if base[coord_name].dtype == object:
            base = base.assign_coords({coord_name: base[coord_name].astype("<U64")})

    base.to_zarr(out_dir, mode="w", consolidated=True)

    return {name: base[name].shape for name in keep_vars}


def _write_place_fields(
    *,
    out_path: Path,
    place_fields: NDArray[np.float64],
    interior_mask: NDArray[np.bool_],
    position_bins: NDArray[np.float64],
    place_field_peaks: NDArray[np.float64],
) -> None:
    np.savez(
        out_path,
        place_fields=place_fields.astype(np.float32),
        interior_mask=interior_mask,
        position_bins=position_bins.astype(np.float64),
        place_field_peaks=place_field_peaks.astype(np.float64),
    )


def _write_meta(
    *,
    out_path: Path,
    time: NDArray[np.float64],
    linear_position: NDArray[np.float64],
    n_cells: int,
) -> None:
    np.savez(
        out_path,
        time=time.astype(np.float64),
        linear_position=linear_position.astype(np.float64),
        n_cells=np.int64(n_cells),
    )


def _write_spike_times(
    *,
    out_path: Path,
    spike_times: list[NDArray[np.float64]],
) -> None:
    """Write per-cell spike-time arrays as a single object-dtype ``.npy``."""
    container = np.empty(len(spike_times), dtype=object)
    for i, st in enumerate(spike_times):
        container[i] = np.asarray(st, dtype=np.float64)
    np.save(out_path, container, allow_pickle=True)


def build_model_cache(
    *,
    model: ModelName,
    intermediates_dir: Path,
    raw_data_dir: Path,
    cache_dir: Path,
    animal_date_epoch: str,
    time_chunk: int = DEFAULT_TIME_CHUNK,
    force: bool = False,
) -> dict[str, Any]:
    """Build the full cache for one model.

    Parameters
    ----------
    model : {"continuous", "contfrag"}
        Which decoder model to cache.
    intermediates_dir : Path
        Directory containing ``cont_results.nc``, ``cont_frag_results.nc``,
        and the matching fitted-model pickles.
    raw_data_dir : Path
        Directory with the raw data files (position info, per-cell spike
        times) for ``animal_date_epoch``.
    cache_dir : Path
        Output directory for the cache (created if missing).
    animal_date_epoch : str
        Identifier passed to ``load_neural_recording_from_files``.
    time_chunk : int, default 8192
        Zarr chunk size along the time axis (~16 s at 500 Hz).
    force : bool, default False
        If False, raises if the model's Zarr store already exists.

    Returns
    -------
    info : dict
        Diagnostic shape/count info, suitable for human inspection or
        smoke tests.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    paths = cache_paths(cache_dir, model)
    if not force and paths["zarr"].exists():
        raise FileExistsError(f"{paths['zarr']} already exists; pass --force to overwrite.")

    inputs = model_paths(intermediates_dir, model)
    for path in (inputs.results_nc, inputs.model_pkl):
        if not path.exists():
            raise FileNotFoundError(f"Required input missing: {path}")

    from statespacecheck_paper.load_local_data import load_neural_recording_from_files
    from statespacecheck_paper.real_data_analysis import (
        compute_per_cell_diagnostics,
        extract_place_fields_concat,
        get_spike_counts,
    )

    with xr.open_dataset(inputs.results_nc) as ds:
        zarr_shapes = _write_zarr_store(
            ds=ds,
            out_dir=paths["zarr"],
            time_chunk=time_chunk,
        )

        time_arr: NDArray[np.float64] = np.asarray(ds["time"].values, dtype=np.float64)
        n_time = time_arr.shape[0]

        # ``position`` is a non-dim coord on ``state_bins`` and repeats
        # across states for multi-state classifiers. The unique sorted
        # values are the underlying 1D position grid.
        position_coord = np.asarray(ds["position"].values, dtype=np.float64)
        position_grid_full = np.unique(position_coord)

        predictive_interior = np.asarray(
            ds["predictive_posterior"].dropna(dim="state_bins").values,
            dtype=np.float64,
        )

    raw = load_neural_recording_from_files(raw_data_dir, animal_date_epoch)
    spike_times: list[NDArray[np.float64]] = list(raw["spike_times"])
    n_cells = len(spike_times)

    fitted_model = joblib.load(inputs.model_pkl)

    place_fields_full, interior_mask = extract_place_fields_concat(fitted_model)
    if place_fields_full.shape[0] != n_cells:
        raise ValueError(
            f"Place fields cell count ({place_fields_full.shape[0]}) "
            f"!= raw spike-times cell count ({n_cells})"
        )
    place_fields = place_fields_full[:, interior_mask]

    if place_fields.shape[1] != predictive_interior.shape[1]:
        raise ValueError(
            f"Place-field interior bin count ({place_fields.shape[1]}) does not "
            f"match predictive_posterior interior bin count "
            f"({predictive_interior.shape[1]})"
        )

    # Per-state interior position grid for raster sorting and the slice
    # panel. The full state-bin count is ``n_states * len(position_grid_full)``;
    # after the interior mask, it is ``n_states * n_interior_per_state``.
    # The interior mask is identical across states (it depends on position,
    # not state), so the first state's slice gives the canonical 1D grid.
    n_pos_full = position_grid_full.shape[0]
    n_states = place_fields_full.shape[1] // n_pos_full
    if n_states * n_pos_full != place_fields_full.shape[1]:
        raise ValueError(
            f"place_fields_full shape {place_fields_full.shape} not divisible by "
            f"n_pos_full={n_pos_full}"
        )
    n_interior_per_state = place_fields.shape[1] // n_states
    if n_interior_per_state * n_states != place_fields.shape[1]:
        raise ValueError(
            f"interior place_fields shape {place_fields.shape} not divisible by n_states={n_states}"
        )
    interior_mask_per_state = interior_mask.reshape(n_states, n_pos_full)[0]
    if not np.array_equal(interior_mask_per_state, interior_mask.reshape(n_states, n_pos_full)[-1]):
        raise ValueError(
            "Interior mask is not identical across states; per-state slicing "
            "would produce inconsistent place-field grids."
        )
    position_bins = position_grid_full[interior_mask_per_state]
    place_fields_per_state = place_fields[:, :n_interior_per_state]
    with np.errstate(invalid="ignore"):
        peak_idx = np.nanargmax(place_fields_per_state, axis=1)
    place_field_peaks = position_bins[peak_idx]

    spike_counts = get_spike_counts(spike_times, time_arr)
    # The cache only consumes the per-spike ``event_*`` arrays;
    # ``include_dense_matrices=False`` skips the (n_time, n_cells) matrix
    # allocations + scatters, which on real data are hundreds of MB.
    diagnostics = compute_per_cell_diagnostics(
        predictive_interior,
        spike_counts,
        place_fields,
        spike_times=spike_times,
        time=time_arr,
        include_dense_matrices=False,
    )
    # ``compute_per_cell_diagnostics`` already returns ``event_cell_ind``
    # in the same order as the per-spike diagnostic arrays — no need to
    # re-derive it via ``_get_spike_events_from_spike_times``.
    events_df = _events_dataframe(diagnostics, n_cells=n_cells)
    events_df.to_parquet(paths["events"], engine="pyarrow", compression="zstd")

    _write_place_fields(
        out_path=paths["place_fields"],
        place_fields=place_fields,
        interior_mask=interior_mask,
        position_bins=position_bins,
        place_field_peaks=place_field_peaks,
    )

    # Meta + spike-times sidecars (model-independent; safe to overwrite).
    position_info = raw["position_info"]
    # The ``linear_position`` field on ``position_info`` was computed
    # with the data-loading pipeline's ``edge_spacing`` (15 cm on this
    # session), so its coordinate system spans 0 → ~608 cm.
    # The fitted decoder, however, was created with ``edge_spacing=1.5``
    # (see ``scripts/run_decoding.py``), so its bin centers only span
    # 0 → ~500 cm. Plotting the position-pkl ``linear_position`` over
    # the decoder's heatmap therefore offsets the trajectory from the
    # underlying probability mass by ~100 cm. Re-linearize the 2-D
    # position using the *decoder's* track graph + edge ordering +
    # edge spacing so the saved trajectory shares one coordinate
    # system with ``predictive_posterior``.
    from track_linearization import get_linearized_position  # noqa: PLC0415

    decoder_env = fitted_model.environments[0]
    pos_2d = position_info[["head_position_x", "head_position_y"]].to_numpy()
    linear_position = np.asarray(
        get_linearized_position(
            position=pos_2d,
            track_graph=decoder_env.track_graph,
            edge_order=decoder_env.edge_order,
            edge_spacing=decoder_env.edge_spacing,
        )["linear_position"].values,
        dtype=np.float64,
    )
    if linear_position.shape[0] != n_time:
        raise ValueError(
            f"linear_position has {linear_position.shape[0]} samples but "
            f"decoder time grid has {n_time} samples"
        )
    _write_meta(
        out_path=meta_path(cache_dir),
        time=time_arr,
        linear_position=linear_position,
        n_cells=n_cells,
    )
    _write_spike_times(out_path=spike_times_path(cache_dir), spike_times=spike_times)

    info: dict[str, Any] = {
        "model": model,
        "n_time": int(n_time),
        "n_cells": int(n_cells),
        "n_state_bins_full_res": int(zarr_shapes["predictive_posterior"][1]),
        "n_position_bins": int(position_bins.shape[0]),
        "n_events": int(len(events_df)),
        "zarr_shapes": {k: list(v) for k, v in zarr_shapes.items()},
        "cache_paths": {k: str(v) for k, v in paths.items()},
        "meta_path": str(meta_path(cache_dir)),
        "spike_times_path": str(spike_times_path(cache_dir)),
    }
    return info


# ---------------------------------------------------------------------------
# Simulated-dataset cache builder
# ---------------------------------------------------------------------------

# 1 sample = ``_SIMULATED_DT`` seconds when written to the simulation
# meta sidecar. The figure-3 simulation is unitless (each time index is
# one decoder step). We pick 2 ms so the viewer's window-width slider
# (0.1 s – 60 s) maps to 50 – 30 000 simulation samples, matching the
# ``dt = 1/500 Hz`` cadence the figure-4 cache uses.
_SIMULATED_DT = 0.002


def build_simulated_cache(
    cache_dir: Path,
    *,
    params: Any | None = None,
    seed: int | None = None,
    time_chunk: int = DEFAULT_TIME_CHUNK,
    force: bool = False,
) -> dict[str, Any]:
    """Run the figure-3 simulation and write a viewer-compatible cache.

    The figure-3 demo is a fundamentally different dataset than the
    figure-4 real-data caches (no recording session, no model choice,
    no smoothed posterior — just a forward filter under several misfit
    conditions). It uses a separate filename layout
    (``simulation.zarr``, ``simulation_events.parquet``,
    ``simulation_place_fields.npz``, ``simulation_meta.npz``,
    ``simulation_spike_times.npy``) and the
    ``DecoderDataSource.for_simulation`` factory.

    Parameters
    ----------
    cache_dir : Path
        Output directory.
    params : DecodeParams, optional
        Simulation configuration. ``None`` ⇒ default ``DecodeParams()``.
    seed : int, optional
        Override ``params.base_seed`` for the run.
    time_chunk : int
        Zarr chunk size along the time axis.
    force : bool
        Overwrite an existing ``simulation.zarr``.

    Returns
    -------
    dict
        Summary of what was written: ``n_time``, ``n_cells``,
        ``n_bins``, ``n_events``, plus the cache paths.

    Notes
    -----
    The simulation's ``metrics["likelihood"]`` is the *normalized linear*
    combined likelihood. The viewer's worker exponentiates the cache's
    ``log_likelihood`` back, so this builder writes
    ``log_likelihood = log(max(likelihood, 1e-12))`` — true log space.
    Without the ``log``, the worker would ``exp`` an already-normalized
    distribution and the likelihood panel would visually flatten.

    No ``acausal_posterior`` is written: the simulation only forward-
    filters, so the smoothed-overlay control is honestly disabled by
    the loader (matching legacy real-data caches without acausal).
    """
    # Imported here so the cache module doesn't pull simulation
    # machinery on every figure-4 cache build.
    from statespacecheck_paper.figure03_demo import (  # noqa: PLC0415
        run_figure03_simulation,
    )
    from statespacecheck_paper.simulation import placefield_rates  # noqa: PLC0415

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    paths = simulated_cache_paths(cache_dir)
    if paths["zarr"].exists() and not force:
        raise FileExistsError(f"{paths['zarr']} already exists; pass force=True to overwrite.")

    sim = run_figure03_simulation(params, seed=seed)
    params_used = sim.params
    xs: NDArray[np.float64] = np.asarray(sim.xs, dtype=np.float64)
    x_true: NDArray[np.float64] = np.asarray(sim.x_true, dtype=np.float64)
    spikes: NDArray[np.int_] = np.asarray(sim.spikes, dtype=np.int_)
    metrics = sim.metrics

    n_time = x_true.shape[0]
    n_bins = xs.shape[0]
    n_cells = int(spikes.shape[1])
    pf_centers = np.asarray(params_used.pf_centers, dtype=np.float64)
    if pf_centers.shape[0] != n_cells:
        raise ValueError(f"pf_centers length {pf_centers.shape[0]} != n_cells={n_cells}")

    time_arr = (np.arange(n_time, dtype=np.float64) * _SIMULATED_DT).astype(np.float64)

    # log_likelihood: true log space. ``metrics["likelihood"]`` is a
    # normalized linear distribution per row; we take ``log`` directly
    # — bins with exact-zero likelihood become ``-inf`` and the
    # viewer's worker handles those (``np.where(np.isfinite(row_max),
    # row_max, 0.0)`` for the per-row shift, then
    # ``np.nan_to_num(neginf=0.0)`` after ``exp``). A blanket clamp
    # at e.g. ``1e-12`` distorts rows whose peak is smaller than the
    # clamp (rare but real for very-flat simulated likelihoods at
    # misfit times — the clamp would round those rows to a uniform
    # response that the viewer renders as flat colour, hiding the
    # actual decoded structure).
    predictive = np.asarray(metrics.predictive, dtype=np.float32)
    likelihood_lin = np.asarray(metrics.likelihood, dtype=np.float64)
    with np.errstate(divide="ignore"):
        log_lik = np.log(likelihood_lin).astype(np.float32)

    # ``state_bins`` axis: one state, so it equals the position grid.
    ds = xr.Dataset(
        data_vars={
            "predictive_posterior": (("time", "state_bins"), predictive),
            "log_likelihood": (("time", "state_bins"), log_lik),
            # Single-state state probability (always 1.0).
            "acausal_state_probabilities": (("time",), np.ones(n_time, dtype=np.float32)),
        },
        coords={
            "time": ("time", time_arr),
            "state_bins": ("state_bins", np.arange(n_bins, dtype=np.int64)),
            "state": ("state_bins", np.array(["state_0"] * n_bins)),
            "position": ("state_bins", xs),
        },
    )
    if paths["zarr"].exists():
        shutil.rmtree(paths["zarr"])
    _write_zarr_store(ds=ds, out_dir=paths["zarr"], time_chunk=time_chunk)

    # Events table. ``event_time_ind`` / ``event_cell_ind`` from
    # ``decode_and_diagnostics`` are already expanded for multi-count
    # bins (a bin with ``k`` spikes contributes ``k`` events) and
    # ``compute_per_cell_diagnostics_from_rates`` returns per-event
    # diagnostics in the same order.
    spike_time_ind = np.asarray(metrics.event_time_ind, dtype=np.intp)
    spike_cell_ind = np.asarray(metrics.event_cell_ind, dtype=np.intp)
    event_times = time_arr[spike_time_ind]
    events_df = pd.DataFrame(
        {
            "time": event_times.astype(np.float64),
            "cell_id": spike_cell_ind.astype(np.int32),
            "event_hpd_overlap": np.asarray(metrics.event_hpd_overlap, dtype=np.float32),
            "event_kl_divergence": np.asarray(metrics.event_kl_divergence, dtype=np.float32),
            "event_spike_prob": np.asarray(metrics.event_spike_prob, dtype=np.float32),
        }
    )
    events_df.sort_values("time", kind="mergesort", inplace=True)
    events_df.reset_index(drop=True, inplace=True)
    events_df.to_parquet(paths["events"], engine="pyarrow", compression="zstd")

    # Place-fields sidecar. ``placefield_rates`` returns
    # ``(n_bins, n_cells)``; the viewer expects ``(n_cells, n_bins)``.
    rates = np.asarray(
        placefield_rates(xs, pf_centers, params_used.pf_width, params_used.rate_scale),
        dtype=np.float64,
    )
    place_fields = rates.T  # (n_cells, n_bins)
    interior_mask = np.ones(n_bins, dtype=bool)
    _write_place_fields(
        out_path=paths["place_fields"],
        place_fields=place_fields,
        interior_mask=interior_mask,
        position_bins=xs,
        place_field_peaks=pf_centers,
    )

    _write_meta(
        out_path=simulated_meta_path(cache_dir),
        time=time_arr,
        linear_position=x_true,
        n_cells=n_cells,
    )

    # Per-cell spike-time arrays. Build by gathering the absolute times
    # at which each cell fired, preserving ordering (already monotone
    # because ``spike_time_ind`` is built from ``np.nonzero`` on the
    # row-major spike matrix).
    # Bucket spike times by cell in O(n_spikes log n_spikes) — the
    # naive ``mask = spike_cell_ind == cell_id`` loop would be
    # O(n_cells × n_spikes) and wasteful at full real-data scale.
    order = np.argsort(spike_cell_ind, kind="stable")
    sorted_cell_ind = spike_cell_ind[order]
    sorted_event_times = event_times[order].astype(np.float64)
    bucket_starts = np.searchsorted(sorted_cell_ind, np.arange(n_cells + 1))
    spike_times_per_cell: list[NDArray[np.float64]] = [
        sorted_event_times[bucket_starts[c] : bucket_starts[c + 1]] for c in range(n_cells)
    ]
    _write_spike_times(
        out_path=simulated_spike_times_path(cache_dir),
        spike_times=spike_times_per_cell,
    )

    return {
        "n_time": int(n_time),
        "n_cells": int(n_cells),
        "n_bins": int(n_bins),
        "n_events": int(len(events_df)),
        "cache_paths": {k: str(v) for k, v in paths.items()},
        "meta_path": str(simulated_meta_path(cache_dir)),
        "spike_times_path": str(simulated_spike_times_path(cache_dir)),
    }


def _build_simulated_command(args: argparse.Namespace) -> int:
    """CLI entry point for ``cache build-simulated``."""
    cache_dir = Path(args.cache_dir).expanduser().resolve()
    print(f"[cache] Building figure-3 simulation cache → {cache_dir} ...", flush=True)
    info = build_simulated_cache(
        cache_dir,
        seed=args.seed,
        time_chunk=args.time_chunk,
        force=args.force,
    )
    print(
        f"[cache] simulation: n_time={info['n_time']} n_cells={info['n_cells']} "
        f"n_bins={info['n_bins']} n_events={info['n_events']}",
        flush=True,
    )
    print("[cache] Done.")
    return 0


def _build_command(args: argparse.Namespace) -> int:
    """CLI entry point for ``cache build``."""
    intermediates_dir = Path(args.intermediates_dir).expanduser().resolve()
    raw_data_dir = Path(args.data_dir).expanduser().resolve()
    cache_dir = Path(args.cache_dir).expanduser().resolve()

    if args.model == "both":
        models: tuple[ModelName, ...] = MODEL_NAMES
    else:
        models = (args.model,)

    summaries: list[dict[str, Any]] = []
    for model in models:
        print(f"[cache] Building cache for model={model} ...", flush=True)
        info = build_model_cache(
            model=model,
            intermediates_dir=intermediates_dir,
            raw_data_dir=raw_data_dir,
            cache_dir=cache_dir,
            animal_date_epoch=args.animal_date_epoch,
            time_chunk=args.time_chunk,
            force=args.force,
        )
        summaries.append(info)
        print(
            f"[cache] {model}: n_time={info['n_time']} n_cells={info['n_cells']} "
            f"n_state_bins={info['n_state_bins_full_res']} n_events={info['n_events']}",
            flush=True,
        )

    # Cross-model sanity check: n_time and n_cells must agree.
    if len(summaries) > 1:
        ref = summaries[0]
        for other in summaries[1:]:
            if other["n_time"] != ref["n_time"] or other["n_cells"] != ref["n_cells"]:
                print(
                    "[cache] WARNING: n_time/n_cells differ across models; "
                    "the meta sidecar reflects the last model written.",
                    file=sys.stderr,
                )
                break

    print("[cache] Done.")
    return 0


def main(argv: list[str] | None = None) -> int:
    """``python -m statespacecheck_paper.interactive.cache`` entry point."""
    parser = argparse.ArgumentParser(prog="statespacecheck_paper.interactive.cache")
    sub = parser.add_subparsers(dest="command", required=True)

    build = sub.add_parser("build", help="Build the Figure 4 viewer cache.")
    build.add_argument(
        "--model",
        choices=(*MODEL_NAMES, "both"),
        default="both",
        help="Which model to cache (default: both).",
    )
    build.add_argument(
        "--data-dir",
        required=True,
        help="Directory with raw data pickles (position_info, spike_times, ...).",
    )
    build.add_argument(
        "--intermediates-dir",
        default=None,
        help="Directory with cont_*.nc and *_model.pkl. Defaults to <data-dir>/intermediates.",
    )
    build.add_argument(
        "--cache-dir",
        default=None,
        help="Output cache directory. Defaults to <data-dir>/cache.",
    )
    build.add_argument(
        "--animal-date-epoch",
        default="j1620210710_02_r1",
        help="Identifier for the recording session.",
    )
    build.add_argument(
        "--time-chunk",
        type=int,
        default=DEFAULT_TIME_CHUNK,
        help="Zarr chunk size along the time axis.",
    )
    build.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing cache directories.",
    )
    build.set_defaults(func=_build_command)

    build_sim = sub.add_parser(
        "build-simulated",
        help="Build the figure-3 simulation cache for the interactive viewer.",
    )
    build_sim.add_argument(
        "--cache-dir",
        required=True,
        help="Output cache directory (will hold simulation.zarr + sidecars).",
    )
    build_sim.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override DecodeParams.base_seed for stochastic draws.",
    )
    build_sim.add_argument(
        "--time-chunk",
        type=int,
        default=DEFAULT_TIME_CHUNK,
        help="Zarr chunk size along the time axis.",
    )
    build_sim.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing simulation cache.",
    )
    build_sim.set_defaults(func=_build_simulated_command)

    args = parser.parse_args(argv)
    if args.command == "build":
        if args.intermediates_dir is None:
            args.intermediates_dir = str(Path(args.data_dir) / "intermediates")
        if args.cache_dir is None:
            args.cache_dir = str(Path(args.data_dir) / "cache")
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
