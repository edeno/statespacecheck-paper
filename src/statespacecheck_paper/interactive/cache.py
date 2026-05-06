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

See ``docs/figure04_interactive_viewer_plan.md`` for design rationale.
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
    """Return the on-disk cache layout for ``model`` under ``cache_dir``."""
    return {
        "zarr": cache_dir / f"figure04_{model}.zarr",
        "events": cache_dir / f"figure04_{model}_events.parquet",
        "place_fields": cache_dir / f"figure04_{model}_place_fields.npz",
    }


def meta_path(cache_dir: Path) -> Path:
    """Path to the model-independent meta sidecar."""
    return cache_dir / "figure04_meta.npz"


def spike_times_path(cache_dir: Path) -> Path:
    """Path to the per-cell spike-times sidecar (object-dtype .npy)."""
    return cache_dir / "figure04_spike_times.npy"


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
    """Extract place fields concatenated across observation models.

    Mirrors the place-field assembly used by ``compute_model_diagnostics``
    in ``real_data_analysis.py`` so the cached place fields align with
    the predictive posterior's ``state_bins`` axis after the
    ``is_track_interior_state_bins_`` mask is applied.

    Parameters
    ----------
    model : decoder model
        Fitted SortedSpikesDecoder or ContFragSortedSpikesClassifier.

    Returns
    -------
    place_fields : np.ndarray, shape (n_cells, n_state_bins_full)
        Place field firing rate per cell, concatenated across observation
        models (i.e. across states for multi-state classifiers).
    interior_mask : np.ndarray, shape (n_state_bins_full,)
        Boolean mask of track-interior bins (the same one used by
        ``model.predict`` to drop NaN bins).
    """
    from statespacecheck_paper.real_data_analysis import extract_place_fields

    place_fields = np.concatenate(
        [
            extract_place_fields(
                model,
                environment_name=obs.environment_name,
                encoding_group=obs.encoding_group,
            )[0]
            for obs in model.observation_models
        ],
        axis=1,
    )
    interior_mask: NDArray[np.bool_] = np.asarray(model.is_track_interior_state_bins_, dtype=bool)
    return place_fields, interior_mask


def _events_dataframe(
    diagnostics: dict[str, NDArray[Any]],
    n_cells: int,
) -> pd.DataFrame:
    """Convert per-spike diagnostic arrays into a sorted Parquet-friendly frame."""
    required = {
        "event_time",
        "event_hpd_overlap",
        "event_kl_divergence",
        "event_spike_prob",
    }
    missing = required.difference(diagnostics)
    if missing:
        raise KeyError(f"Diagnostics missing required event keys: {sorted(missing)}")

    cell_id = np.asarray(diagnostics["event_cell_ind"], dtype=np.int32)
    if cell_id.size and (cell_id.min() < 0 or cell_id.max() >= n_cells):
        raise ValueError(
            f"event_cell_ind out of range [0, {n_cells}); got [{cell_id.min()}, {cell_id.max()}]"
        )

    df = pd.DataFrame(
        {
            "time": np.asarray(diagnostics["event_time"], dtype=np.float64),
            "cell_id": cell_id,
            "event_hpd_overlap": np.asarray(diagnostics["event_hpd_overlap"], dtype=np.float32),
            "event_kl_divergence": np.asarray(diagnostics["event_kl_divergence"], dtype=np.float32),
            "event_spike_prob": np.asarray(diagnostics["event_spike_prob"], dtype=np.float32),
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
        _get_spike_events_from_spike_times,
        compute_per_cell_diagnostics,
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

    place_fields_full, interior_mask = _extract_place_fields_concat(fitted_model)
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
    diagnostics = compute_per_cell_diagnostics(
        predictive_interior,
        spike_counts,
        place_fields,
        spike_times=spike_times,
        time=time_arr,
    )
    # ``compute_per_cell_diagnostics`` returns event arrays sorted in the
    # same order as the (spike_time_ind, spike_cell_ind) pair built by
    # ``_get_spike_events_from_spike_times``. Recompute that pair to
    # obtain the matching ``cell_id`` column for the event table.
    _, spike_cell_ind, _ = _get_spike_events_from_spike_times(spike_times, time_arr)
    diagnostics["event_cell_ind"] = np.asarray(spike_cell_ind, dtype=np.int64)

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
    linear_position = np.asarray(position_info["linear_position"].values, dtype=np.float64)
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

    args = parser.parse_args(argv)
    if args.intermediates_dir is None:
        args.intermediates_dir = str(Path(args.data_dir) / "intermediates")
    if args.cache_dir is None:
        args.cache_dir = str(Path(args.data_dir) / "cache")
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
