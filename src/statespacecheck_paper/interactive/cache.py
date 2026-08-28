"""Build the on-disk caches used by the interactive viewer.

For Figure 4, the cache reformats the canonical decode bundle produced by
``generate_figure04.py`` into a layout that supports fast windowed reads:

- A Zarr store per model with chunked posterior / log-likelihood arrays
  (chunked along time, full position axis per chunk).
- A Parquet event table with one row per spike, sorted by time, holding
  the per-spike diagnostic metrics (HPD overlap, KL divergence, spike
  probability) plus the cell index.
- A small ``.npz`` sidecar with the time grid, animal linear position,
  per-cell place fields, and place-field peak positions.
- A ``.npy`` sidecar with the per-cell spike-time arrays used by
  the raster panel.

Usage::

    python -m statespacecheck_paper.interactive.cache build \\
        --model continuous --data-dir DATA --cache-dir DATA/cache

See the package's ``__init__.py`` for the public surface.
"""

from __future__ import annotations

import argparse
import shutil
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import pandas as pd
import xarray as xr
from numpy.typing import NDArray

from statespacecheck_paper.diagnostics import SpikeEventDiagnostics

if TYPE_CHECKING:
    from statespacecheck_paper.figure04_workflow import Figure4RenderData

ModelName = Literal["continuous", "contfrag"]
MODEL_NAMES: tuple[ModelName, ...] = ("continuous", "contfrag")

DEFAULT_TIME_CHUNK = 8192


def cache_paths(cache_dir: Path, model: ModelName) -> dict[str, Path]:
    """Return the on-disk cache layout for ``model`` under ``cache_dir``.

    Real-data caches are figure-4 specific (the ``figure04_`` prefix
    is meaningful — these files are derived from the canonical Figure 4
    joblib decode bundle).
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


def _events_dataframe(
    diagnostics: SpikeEventDiagnostics,
    n_cells: int,
    *,
    time: NDArray[np.float64] | None = None,
) -> pd.DataFrame:
    """Convert per-spike diagnostic arrays into a sorted Parquet-friendly frame."""
    event_time = diagnostics.event_time
    if event_time is None:
        if time is None:
            raise ValueError(
                "SpikeEventDiagnostics.event_time or an explicit decoder time grid "
                "is required when building the cache events frame."
            )
        event_time_ind = np.asarray(diagnostics.event_time_ind, dtype=np.intp)
        if event_time_ind.size and (
            event_time_ind.min() < 0 or event_time_ind.max() >= time.shape[0]
        ):
            raise ValueError(
                "event_time_ind falls outside the supplied decoder time grid: "
                f"valid [0, {time.shape[0]}), got "
                f"[{event_time_ind.min()}, {event_time_ind.max()}]"
            )
        event_time = time[event_time_ind]

    cell_id = np.asarray(diagnostics.event_cell_ind, dtype=np.int32)
    if cell_id.size and (cell_id.min() < 0 or cell_id.max() >= n_cells):
        raise ValueError(
            f"event_cell_ind out of range [0, {n_cells}); got [{cell_id.min()}, {cell_id.max()}]"
        )

    df = pd.DataFrame(
        {
            "time": np.asarray(event_time, dtype=np.float64),
            "cell_id": cell_id,
            "event_hpd_overlap": np.asarray(diagnostics.event_hpd_overlap, dtype=np.float32),
            "event_kl_divergence": np.asarray(diagnostics.event_kl_divergence, dtype=np.float32),
            "event_predictive_pvalue": np.asarray(
                diagnostics.event_predictive_pvalue, dtype=np.float32
            ),
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

    base = ds[keep_vars]
    # The canonical joblib cache preserves xarray's ``state_bins`` MultiIndex,
    # while Zarr cannot serialize a pandas MultiIndex directly. Flatten it back
    # to ordinary ``state`` / ``position`` coordinates, matching the layout the
    # viewer already reads, and give ``state_bins`` a simple integer coordinate.
    if isinstance(base.indexes.get("state_bins"), pd.MultiIndex):
        base = base.reset_index("state_bins")
        base = base.assign_coords(state_bins=np.arange(base.sizes["state_bins"], dtype=np.int64))
    base = base.chunk({"time": time_chunk})

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
    event_likelihood: NDArray[np.floating] | None = None,
) -> None:
    # ``np.savez`` uses each keyword as the archive member name, so the
    # arrays must be passed as literal keywords. Unpacking an array-valued
    # ``**dict`` instead trips mypy, whose ``savez`` stub reserves an
    # ``allow_pickle: bool`` keyword a ``**dict`` value could collide with.
    # Two explicit calls keep the optional ``event_likelihood`` member
    # without that unpacking.
    place_fields32 = place_fields.astype(np.float32)
    position_bins64 = position_bins.astype(np.float64)
    place_field_peaks64 = place_field_peaks.astype(np.float64)
    if event_likelihood is None:
        np.savez(
            out_path,
            place_fields=place_fields32,
            interior_mask=interior_mask,
            position_bins=position_bins64,
            place_field_peaks=place_field_peaks64,
        )
    else:
        np.savez(
            out_path,
            place_fields=place_fields32,
            interior_mask=interior_mask,
            position_bins=position_bins64,
            place_field_peaks=place_field_peaks64,
            event_likelihood=np.asarray(event_likelihood, dtype=np.float32),
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


def _figure04_model_inputs(
    render_data: Figure4RenderData,
    model: ModelName,
) -> tuple[xr.Dataset, SpikeEventDiagnostics]:
    """Return the canonical result dataset and diagnostics for one model."""
    decode = render_data.decode_results
    if model == "continuous":
        return decode.continuous_results, decode.continuous_diagnostics
    if model == "contfrag":
        return (
            decode.continuous_fragmented_results,
            decode.continuous_fragmented_diagnostics,
        )
    raise ValueError(f"Unknown model: {model!r}")


def _position_grid_and_interior_mask(
    results: xr.Dataset,
    diagnostic_position_bins: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.bool_], int]:
    """Match the shared diagnostic grid to a decoder result's full state axis."""
    if "position" not in results.coords:
        raise ValueError("Figure 4 decoder results must carry a 'position' coordinate")
    if "predictive_posterior" not in results:
        raise ValueError("Figure 4 decoder results are missing 'predictive_posterior'")

    position_coord = np.asarray(results.coords["position"].values, dtype=np.float64)
    position_grid_full = np.unique(position_coord)
    if position_grid_full.size == 0:
        raise ValueError("Figure 4 decoder position grid is empty")

    interior_bins = np.asarray(diagnostic_position_bins, dtype=np.float64)
    matches = np.isclose(
        position_grid_full[:, np.newaxis],
        interior_bins[np.newaxis, :],
        rtol=1e-10,
        atol=1e-12,
    )
    matches_per_bin = matches.sum(axis=0)
    if np.any(matches_per_bin != 1):
        raise ValueError(
            "Each diagnostic position bin must match exactly one decoder position "
            f"bin; match counts were {matches_per_bin.tolist()}."
        )
    interior_mask = np.any(matches, axis=1)
    if not np.allclose(position_grid_full[interior_mask], interior_bins):
        raise ValueError(
            "Diagnostic position bins are not in the same order as the decoder position coordinate."
        )

    n_state_bins = int(results["predictive_posterior"].sizes["state_bins"])
    if n_state_bins % position_grid_full.size:
        raise ValueError(
            f"Decoder state axis ({n_state_bins}) is not divisible by the "
            f"position-grid size ({position_grid_full.size})."
        )
    n_states = n_state_bins // position_grid_full.size
    return position_grid_full, interior_mask, n_states


def _write_figure04_model_cache(
    *,
    render_data: Figure4RenderData,
    model: ModelName,
    cache_dir: Path,
    time_chunk: int,
) -> dict[str, Any]:
    """Write one viewer model from the canonical Figure 4 render data."""
    results, diagnostics = _figure04_model_inputs(render_data, model)
    paths = cache_paths(cache_dir, model)
    zarr_shapes = _write_zarr_store(
        ds=results,
        out_dir=paths["zarr"],
        time_chunk=time_chunk,
    )

    decode = render_data.decode_results
    position_bins = np.asarray(decode.diagnostic_position_bins, dtype=np.float64)
    _, interior_mask, n_states = _position_grid_and_interior_mask(results, position_bins)
    place_fields = np.asarray(decode.diagnostic_place_fields, dtype=np.float64)
    n_cells = int(decode.spike_counts.shape[1])
    if place_fields.shape != (n_cells, position_bins.size):
        raise ValueError(
            "Shared Figure 4 place fields must have shape "
            f"({n_cells}, {position_bins.size}); got {place_fields.shape}."
        )

    events_df = _events_dataframe(
        diagnostics,
        n_cells=n_cells,
        time=np.asarray(render_data.time, dtype=np.float64),
    )
    events_df.to_parquet(paths["events"], engine="pyarrow", compression="zstd")
    _write_place_fields(
        out_path=paths["place_fields"],
        place_fields=place_fields,
        interior_mask=interior_mask,
        position_bins=position_bins,
        place_field_peaks=np.asarray(decode.place_field_peaks, dtype=np.float64),
    )

    return {
        "model": model,
        "n_time": int(decode.spike_counts.shape[0]),
        "n_cells": n_cells,
        "n_states": n_states,
        "n_state_bins_full_res": int(zarr_shapes["predictive_posterior"][1]),
        "n_position_bins": int(position_bins.size),
        "n_events": int(len(events_df)),
        "zarr_shapes": {key: list(shape) for key, shape in zarr_shapes.items()},
        "cache_paths": {key: str(path) for key, path in paths.items()},
    }


def build_figure04_viewer_cache(
    *,
    render_data: Figure4RenderData,
    cache_dir: Path,
    models: Sequence[ModelName] = MODEL_NAMES,
    time_chunk: int = DEFAULT_TIME_CHUNK,
    force: bool = False,
) -> dict[ModelName, dict[str, Any]]:
    """Derive viewer artifacts from the canonical Figure 4 workflow output.

    The input is the same Figure4RenderData used to render the static figure.
    In the normal CLI path it is loaded from the epoch's fig4_cache.joblib by
    prepare_figure04_render_data. This keeps the viewer and paper on one
    decode/diagnostic source of truth and eliminates the former dependency on
    separately produced NetCDF results and fitted-model pickles.
    """
    selected = tuple(models)
    if not selected:
        raise ValueError("models must contain at least one Figure 4 model")
    if len(set(selected)) != len(selected):
        raise ValueError(f"models contains duplicates: {selected!r}")
    unknown = [model for model in selected if model not in MODEL_NAMES]
    if unknown:
        raise ValueError(f"Unknown Figure 4 models: {unknown!r}")

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    if not force:
        existing = [
            cache_paths(cache_dir, model)["zarr"]
            for model in selected
            if cache_paths(cache_dir, model)["zarr"].exists()
        ]
        if existing:
            raise FileExistsError(
                f"{existing[0]} already exists; pass --force to overwrite viewer artifacts."
            )

    summaries: dict[ModelName, dict[str, Any]] = {}
    for model in selected:
        summaries[model] = _write_figure04_model_cache(
            render_data=render_data,
            model=model,
            cache_dir=cache_dir,
            time_chunk=time_chunk,
        )

    decode = render_data.decode_results
    n_cells = int(decode.spike_counts.shape[1])
    spike_times = [
        np.asarray(cell_spike_times, dtype=np.float64)
        for cell_spike_times in render_data.recording.spike_times
    ]
    if len(spike_times) != n_cells:
        raise ValueError(
            f"Recording has {len(spike_times)} spike-time arrays but the decode "
            f"has {n_cells} cells."
        )
    _write_meta(
        out_path=meta_path(cache_dir),
        time=np.asarray(render_data.time, dtype=np.float64),
        linear_position=np.asarray(render_data.linear_position, dtype=np.float64),
        n_cells=n_cells,
    )
    _write_spike_times(
        out_path=spike_times_path(cache_dir),
        spike_times=spike_times,
    )

    for info in summaries.values():
        info["meta_path"] = str(meta_path(cache_dir))
        info["spike_times_path"] = str(spike_times_path(cache_dir))
    return summaries


# Simulated-dataset cache builder
# ---------------------------------------------------------------------------

# 1 sample = ``_SIMULATED_DT`` seconds when written to the simulation
# meta sidecar. The figure-3 simulation is dt-agnostic (each time index is
# one decoder step), but the manuscript and ``Figure3Config`` fix the step at
# 1 ms by convention: main.tex calibrates 0.20 spikes/step as ~200 Hz and
# ``Figure3Config`` sizes 0.001 spikes/step as 1 Hz, both of which hold only at
# 1 ms/step. Use the same 1 ms here so the viewer's time axis, event times, and
# window-width slider match the manuscript timebase. (The figure-4 real-data
# cache is a genuinely different 2 ms / 500 Hz cadence and is unaffected.)
_SIMULATED_DT = 0.001


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
    params : Figure3Config, optional
        Simulation configuration. ``None`` ⇒ default ``Figure3Config()``.
    seed : int, optional
        Override ``params.random_seed`` for the run.
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
    ``log_likelihood`` back, so this builder writes ``log_likelihood =
    log(likelihood)`` — true log space, with no clamp (exact-zero bins
    become ``-inf``, which the worker handles). Without the ``log``, the
    worker would ``exp`` an already-normalized distribution and the
    likelihood panel would visually flatten.

    No ``acausal_posterior`` is written: the simulation only forward-
    filters, so the smoothed-overlay control is honestly disabled by
    the loader (matching legacy real-data caches without acausal).
    """
    # Imported here so the cache module doesn't pull simulation
    # machinery on every figure-4 cache build.
    from statespacecheck_paper.figure03_simulation import (  # noqa: PLC0415
        run_figure03_simulation,
    )
    from statespacecheck_paper.simulation import place_field_rates  # noqa: PLC0415

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    paths = simulated_cache_paths(cache_dir)
    if paths["zarr"].exists() and not force:
        raise FileExistsError(f"{paths['zarr']} already exists; pass force=True to overwrite.")

    sim = run_figure03_simulation(params, seed=seed)
    params_used = sim.config
    xs: NDArray[np.float64] = np.asarray(sim.position_bins, dtype=np.float64)
    x_true: NDArray[np.float64] = np.asarray(sim.true_position, dtype=np.float64)
    spikes: NDArray[np.int_] = np.asarray(sim.spike_counts, dtype=np.int_)
    metrics = sim.diagnostics

    n_time = x_true.shape[0]
    n_bins = xs.shape[0]
    n_cells = int(spikes.shape[1])
    # The simulation appends a narrow sparse-population of cells; include them
    # in the cache's cell set and sort them at their fixed field centers.
    pf_centers = np.asarray(params_used.place_field_centers, dtype=np.float64)
    pf_centers_full = np.append(
        pf_centers, np.asarray(sim.sparse_place_field_centers, dtype=np.float64)
    )
    if pf_centers_full.shape[0] != n_cells:
        raise ValueError(f"pf_centers length {pf_centers_full.shape[0]} != n_cells={n_cells}")

    time_arr = (np.arange(n_time, dtype=np.float64) * _SIMULATED_DT).astype(np.float64)

    # log_likelihood: true log space. ``metrics["likelihood"]`` is a
    # normalized linear distribution per row; we take ``log`` directly
    # — bins with exact-zero likelihood become ``-inf`` and the
    # viewer preserves them as zero relative likelihood after a finite
    # per-row shift. A row with no supported state raises. A blanket clamp
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
    # ``decode_with_diagnostics`` are already expanded for multi-count
    # bins (a bin with ``k`` spikes contributes ``k`` events) and
    # ``compute_spike_event_diagnostics_from_rates`` returns per-event
    # diagnostics in the same order.
    spike_time_ind = np.asarray(metrics.event_time_ind, dtype=np.intp)
    spike_cell_ind = np.asarray(metrics.event_cell_ind, dtype=np.intp)
    event_times = time_arr[spike_time_ind]
    event_order = np.argsort(event_times, kind="stable")
    events_df = pd.DataFrame(
        {
            "time": event_times[event_order].astype(np.float64),
            "cell_id": spike_cell_ind[event_order].astype(np.int32),
            "event_hpd_overlap": np.asarray(
                metrics.event_hpd_overlap[event_order], dtype=np.float32
            ),
            "event_kl_divergence": np.asarray(
                metrics.event_kl_divergence[event_order], dtype=np.float32
            ),
            "event_predictive_pvalue": np.asarray(
                metrics.event_predictive_pvalue[event_order], dtype=np.float32
            ),
        }
    )
    events_df.to_parquet(paths["events"], engine="pyarrow", compression="zstd")

    # Place-fields sidecar. The 11 normal cells (shared width) plus the narrow
    # sparse-population cells (their own width and peak rate). ``place_field_rates``
    # returns ``(n_bins, n_cells)``; the viewer expects ``(n_cells, n_bins)``.
    normal_rates = place_field_rates(
        xs, pf_centers, params_used.place_field_std, params_used.place_field_rate_scale
    )
    sparse_cell_scale = (
        params_used.sparse_cell_peak_rate_per_step
        * np.sqrt(2.0 * np.pi)
        * params_used.sparse_place_field_std
    )
    sparse_rates = place_field_rates(
        xs,
        np.asarray(sim.sparse_place_field_centers, dtype=np.float64),
        params_used.sparse_place_field_std,
        sparse_cell_scale,
    )
    rates = np.asarray(np.hstack([normal_rates, sparse_rates]), dtype=np.float64)
    place_fields = rates.T  # (n_cells, n_bins)
    interior_mask = np.ones(n_bins, dtype=bool)
    _write_place_fields(
        out_path=paths["place_fields"],
        place_fields=place_fields,
        interior_mask=interior_mask,
        position_bins=xs,
        place_field_peaks=pf_centers_full,
        event_likelihood=np.asarray(metrics.per_spike_likelihood[event_order]),
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
    data_dir = Path(args.data_dir).expanduser().resolve()
    cache_dir = Path(args.cache_dir).expanduser().resolve()

    if args.model == "both":
        models: tuple[ModelName, ...] = MODEL_NAMES
    else:
        models = (args.model,)

    # Lazy imports keep the figure-3 simulation cache command independent of
    # the real-data decoder stack.
    from statespacecheck_paper.figure04_cache import (  # noqa: PLC0415
        Figure4Paths,
    )
    from statespacecheck_paper.figure04_decoder import (  # noqa: PLC0415
        Figure4Config,
    )
    from statespacecheck_paper.figure04_workflow import (  # noqa: PLC0415
        prepare_figure04_render_data,
    )

    figure4_paths = Figure4Paths(
        data_path=data_dir,
        animal_date_epoch=args.animal_date_epoch,
    )
    print(
        f"[cache] Loading canonical Figure 4 workflow data from {figure4_paths.cache_path} ...",
        flush=True,
    )
    render_data = prepare_figure04_render_data(
        Figure4Config(),
        figure4_paths,
        use_cache=not args.force_recompute,
    )
    summaries = build_figure04_viewer_cache(
        render_data=render_data,
        cache_dir=cache_dir,
        models=models,
        time_chunk=args.time_chunk,
        force=args.force,
    )
    for model, info in summaries.items():
        print(
            f"[cache] {model}: n_time={info['n_time']} n_cells={info['n_cells']} "
            f"n_state_bins={info['n_state_bins_full_res']} n_events={info['n_events']}",
            flush=True,
        )

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
        help=(
            "Figure 4 data directory. Must contain the exported recording inputs "
            "and the canonical intermediates/{epoch}_fig4_cache.joblib bundle."
        ),
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
        help="Overwrite existing viewer cache directories.",
    )
    build.add_argument(
        "--force-recompute",
        action="store_true",
        help=(
            "Re-fit and re-decode Figure 4 instead of loading its canonical "
            "joblib cache. This also overwrites that canonical cache."
        ),
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
        help="Override Figure3Config.random_seed for stochastic draws.",
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
        if args.cache_dir is None:
            args.cache_dir = str(Path(args.data_dir) / "cache")
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
