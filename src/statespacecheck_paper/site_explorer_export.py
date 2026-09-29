"""Compact, lazily loaded Figure-4 data for the website's session explorer.

The explorer compares the two Figure-4 decoders over every spike in the
session. It has three parts, all from the same fingerprinted decode:

- **Overview** (``site/data/recording_explorer.json``, committed) — for each
  diagnostic, a ``grid_size`` x ``grid_size`` count of spikes by their value
  under the Continuous (x) and Continuous--Fragmented (y) models; the flag
  confusion counts; each display window's color range; and the name, URL, and
  SHA-256 of the archive holding the rest.
- **Spike index** (``explorer/index.bin.gz``) — each spike's decoder time bin,
  unit, and comparison square per diagnostic: what choosing a square needs.
- **Blocks** (``explorer/blocks/NNNN.bin.gz``) — the session in consecutive,
  non-overlapping blocks of ``block_samples`` decoder bins. Each holds both
  models' predictive distributions (rows scaled to their maxima, ``uint8``),
  the animal's position, and every spike counted in the block with its
  diagnostic values and flags. The page shows two adjacent blocks as one
  window, so every sample is stored once.

The index and blocks are too large to keep in Git. They are written to one
deterministic tar archive, published as a GitHub release asset, and fetched and
checked against the overview's SHA-256 when the site is assembled. Shared
fields (position grid, place fields, per-cell likelihoods, labels) come from
``recording.json``. The page derives the likelihood track and the raster from
each block's spikes, as for the paper's window.

Both files use one binary container (:func:`encode_container`): the ASCII magic
``SSCX``, a little-endian ``uint32`` header length, a UTF-8 JSON header, and
8-byte-aligned little-endian arrays that the header names, types, and places.
"""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import math
import sys
import tarfile
from collections.abc import Collection, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr
from numpy.typing import NDArray

from statespacecheck_paper.diagnostics import SpikeEventDiagnostics
from statespacecheck_paper.figure04_cache import executable_source_digest
from statespacecheck_paper.figure04_models import CONTINUOUS, CONTINUOUS_FRAGMENTED
from statespacecheck_paper.figure04_place_fields import marginal_position_distribution
from statespacecheck_paper.figure04_workflow import Figure4RenderData
from statespacecheck_paper.site_export import (
    EVENT_VALUE_SIGNIFICANT_FIGURES,
    FIGURE04_PREDICTIVE_PERCENTILES,
    display_rows,
    flag_events,
    rounded_significant,
    write_site_json,
)
from statespacecheck_paper.style import METRIC_NAMES

FORMAT_VERSION = 1
GRID_SIZE = 40
# Two seconds of the 500 Hz decoder grid; the page shows two blocks at a time.
BLOCK_SAMPLES = 1000
WINDOW_BLOCKS = 2
# Spike times are shipped in units of 0.1 ms, the resolution recording.json uses.
TICKS_PER_SECOND = 10_000
MODELS: tuple[str, ...] = (CONTINUOUS.id, CONTINUOUS_FRAGMENTED.id)
RELEASE_URL = "https://github.com/edeno/statespacecheck-paper/releases/download/site-data/"

_MAGIC = b"SSCX"
_DTYPES = {
    "u1": np.uint8,
    "u2": np.uint16,
    "u4": np.uint32,
    "i4": np.int32,
    "f4": np.float32,
    "f8": np.float64,
}
# The modules whose code determines the explorer files.
_EXPORT_SOURCES = ("site_explorer_export.py", "site_export.py")


def export_source_python() -> str:
    """Return the Python ``major.minor`` whose syntax trees :func:`export_source_digest` hashes.

    ``ast.dump`` output differs between Python versions, so the digest is only
    comparable on the version that recorded it.
    """
    return f"{sys.version_info.major}.{sys.version_info.minor}"


def export_source_digest() -> str:
    """Executable-source digest of the modules that write the explorer files."""
    package_root = Path(__file__).resolve().parent
    return executable_source_digest(tuple(package_root / name for name in _EXPORT_SOURCES))


def encode_container(
    meta: Mapping[str, Any],
    arrays: Mapping[str, NDArray[Any]],
    delta: Collection[str] = (),
) -> bytes:
    """Pack ``arrays`` and a JSON ``meta`` header into one binary container.

    Parameters
    ----------
    meta : Mapping
        JSON-serializable values stored in the header under ``"meta"``.
    arrays : Mapping[str, np.ndarray]
        Arrays of a dtype in ``u1``, ``u2``, ``u4``, ``i4``, ``f4``, ``f8``, stored
        C-ordered and little-endian at 8-byte-aligned offsets.
    delta : collection of str
        Names of unsigned-integer arrays stored as differences along their last
        axis (modulo the dtype's range), which compress better when neighboring
        values are close; readers take the running sum. Their header entries
        carry ``"filter": "delta"``.

    Returns
    -------
    bytes
        ``SSCX``, the header length (``uint32``), the header, then the arrays.
    """
    entries = []
    blobs = []
    for name, values in arrays.items():
        array = np.ascontiguousarray(values)
        code = next((code for code, dtype in _DTYPES.items() if array.dtype == dtype), None)
        if code is None:
            raise TypeError(f"{name}: unsupported dtype {array.dtype}")
        entry: dict[str, Any] = {"name": name, "dtype": code, "shape": list(array.shape)}
        if name in delta:
            if not code.startswith("u"):
                raise TypeError(f"{name}: delta filtering needs an unsigned dtype")
            # Unsigned subtraction wraps, so the running sum restores the values.
            array = np.diff(array, axis=-1, prepend=np.zeros_like(array[..., :1]))
            entry["filter"] = "delta"
        entries.append(entry)
        blobs.append(array.astype(array.dtype.newbyteorder("<"), copy=False).tobytes())

    def header_bytes(offsets: list[int]) -> bytes:
        header = {
            "meta": meta,
            "arrays": [{**e, "offset": o} for e, o in zip(entries, offsets, strict=True)],
        }
        return json.dumps(header, separators=(",", ":"), allow_nan=False).encode("utf-8")

    # Offsets depend on the header length, which depends on the offsets'
    # digits; iterate until they agree (at most a few passes).
    offsets = [0] * len(blobs)
    while True:
        header = header_bytes(offsets)
        position = _aligned(8 + len(header))
        placed = []
        for blob in blobs:
            placed.append(position)
            position = _aligned(position + len(blob))
        if placed == offsets:
            break
        offsets = placed
    out = bytearray(position)
    out[:4] = _MAGIC
    out[4:8] = len(header).to_bytes(4, "little")
    out[8 : 8 + len(header)] = header
    for offset, blob in zip(offsets, blobs, strict=True):
        out[offset : offset + len(blob)] = blob
    return bytes(out)


def decode_container(data: bytes) -> tuple[dict[str, Any], dict[str, NDArray[Any]]]:
    """Inverse of :func:`encode_container`: the ``meta`` header and the arrays."""
    if data[:4] != _MAGIC:
        raise ValueError("Not an explorer container")
    length = int.from_bytes(data[4:8], "little")
    header = json.loads(data[8 : 8 + length])
    arrays = {}
    for entry in header["arrays"]:
        dtype = np.dtype(_DTYPES[entry["dtype"]]).newbyteorder("<")
        count = math.prod(entry["shape"])
        array = np.frombuffer(data, dtype=dtype, count=count, offset=entry["offset"]).reshape(
            entry["shape"]
        )
        if entry.get("filter") == "delta":
            array = np.cumsum(array, axis=-1, dtype=dtype)
        arrays[entry["name"]] = array
    return header["meta"], arrays


def _aligned(position: int) -> int:
    return -(-position // 8) * 8


def block_bounds(n_time: int, block: int = BLOCK_SAMPLES) -> list[tuple[int, int]]:
    """``[start, stop)`` decoder bins of each consecutive block; the last may be short."""
    if n_time < WINDOW_BLOCKS * block or block <= 0:
        raise ValueError(f"The session must span at least {WINDOW_BLOCKS} blocks of {block} bins")
    return [(start, min(start + block, n_time)) for start in range(0, n_time, block)]


def _smallest_unsigned(maximum: int) -> type[np.unsignedinteger[Any]]:
    for dtype in (np.uint8, np.uint16, np.uint32):
        if maximum <= np.iinfo(dtype).max:
            return dtype
    raise ValueError(f"{maximum} exceeds uint32")


def _plotted(metric: str, values: NDArray[np.float64]) -> NDArray[np.float64]:
    """Values on the comparison axes: −ln p for the p-value, as in the paper."""
    return -np.log(values) if metric == "predictive_pvalue" else values


def comparison_squares(
    reference: NDArray[np.float64], comparison: NDArray[np.float64], metric: str, grid_size: int
) -> tuple[dict[str, Any], NDArray[np.uint16]]:
    """Count spikes on a ``grid_size`` square grid by their value under both models.

    Returns the grid (``max`` of both axes and the x-major ``counts``) and each
    spike's square, ``x * grid_size + y``, from the histogram's own edges so a
    value on an edge lands where the count put it.
    """
    x, y = _plotted(metric, reference), _plotted(metric, comparison)
    if x.shape != y.shape or not (np.all(np.isfinite(x)) and np.all(np.isfinite(y))):
        raise ValueError(f"Non-finite or misaligned Figure-4 {metric} values")
    top = 1.0 if metric == "hpd_overlap" else float(math.ceil(max(x.max(), y.max())))
    counts, x_edges, y_edges = np.histogram2d(x, y, bins=grid_size, range=((0.0, top), (0.0, top)))
    x_square = np.clip(np.searchsorted(x_edges, x, side="right") - 1, 0, grid_size - 1)
    y_square = np.clip(np.searchsorted(y_edges, y, side="right") - 1, 0, grid_size - 1)
    grid = {"max": top, "counts": counts.astype(np.int64).ravel().tolist()}
    return grid, (x_square * grid_size + y_square).astype(np.uint16)


def explorer_files(
    render_data: Figure4RenderData,
    summary: Mapping[str, Any],
    *,
    grid_size: int = GRID_SIZE,
    block: int = BLOCK_SAMPLES,
) -> tuple[dict[str, Any], dict[str, bytes]]:
    """Build the overview (without its archive entry) and the gzipped index and blocks.

    Returns
    -------
    overview : dict
        Everything the committed overview holds except ``archive``.
    files : dict[str, bytes]
        Gzipped containers keyed by their path inside the archive.

    Raises
    ------
    ValueError
        If the summary describes another decode, the models score different
        spikes, or the spikes are not in time order.
    """
    if grid_size <= 0 or grid_size**2 > np.iinfo(np.uint16).max + 1:
        raise ValueError("grid_size must be positive and fit uint16 squares")
    caches = summary["provenance"]["figure04_caches"]
    provenance = render_data.cache_provenance
    if (
        provenance.fingerprint_sha256 != caches["fingerprint_sha256"]
        or provenance.diagnostics_fingerprint_sha256 != caches["diagnostics_fingerprint_sha256"]
    ):
        raise ValueError("Figure-4 summary and explorer cache fingerprints differ")
    analysis = render_data.analysis_results
    diagnostics: dict[str, SpikeEventDiagnostics] = {
        CONTINUOUS.id: analysis.continuous_diagnostics,
        CONTINUOUS_FRAGMENTED.id: analysis.continuous_fragmented_diagnostics,
    }
    results: dict[str, xr.Dataset] = {
        CONTINUOUS.id: analysis.continuous_results,
        CONTINUOUS_FRAGMENTED.id: analysis.continuous_fragmented_results,
    }
    ref, other = diagnostics[CONTINUOUS.id], diagnostics[CONTINUOUS_FRAGMENTED.id]
    if ref.event_time is None or other.event_time is None:
        raise ValueError("Figure-4 diagnostics lack exact event times")
    if not (
        np.array_equal(ref.event_time_ind, other.event_time_ind)
        and np.array_equal(ref.event_cell_ind, other.event_cell_ind)
        and np.array_equal(ref.event_time, other.event_time)
    ):
        raise ValueError("Figure-4 models do not score the same spikes in the same order")
    time_bin = np.asarray(ref.event_time_ind, dtype=np.int64)
    if np.any(np.diff(time_bin) < 0):
        raise ValueError("Figure-4 spikes must be in time order")
    cell = np.asarray(ref.event_cell_ind, dtype=np.int64)
    time = np.asarray(render_data.time, dtype=np.float64)
    # The span over the number of steps: individual differences of these
    # epoch timestamps carry rounding of ~1e-7 s.
    step = float((time[-1] - time[0]) / (time.size - 1))
    grid_error = np.abs(time - (time[0] + step * np.arange(time.size))).max()
    if grid_error > 1e-3 * step:
        raise ValueError(
            f"The explorer assumes a regular decoder time grid (off by {grid_error} s)"
        )
    ticks = np.rint((np.asarray(ref.event_time) - time[0]) * TICKS_PER_SECOND).astype(np.int64)
    flag_rules = summary["flag_rules"]

    grids: dict[str, Any] = {}
    squares: dict[str, NDArray[np.uint16]] = {}
    values: dict[tuple[str, str], NDArray[np.float64]] = {}
    # Shipped values carry recording.json's significant figures; flags and
    # squares use the full-precision values.
    shipped: dict[tuple[str, str], NDArray[np.float32]] = {}
    for metric in METRIC_NAMES:
        for model in MODELS:
            values[model, metric] = np.asarray(
                getattr(diagnostics[model], f"event_{metric}"), dtype=np.float64
            )
            shipped[model, metric] = np.asarray(
                rounded_significant(values[model, metric], EVENT_VALUE_SIGNIFICANT_FIGURES),
                dtype=np.float32,
            )
        grids[metric], squares[metric] = comparison_squares(
            values[CONTINUOUS.id, metric],
            values[CONTINUOUS_FRAGMENTED.id, metric],
            metric,
            grid_size,
        )
    # One bit per (model, flagged metric), in this order.
    flag_bits = [
        (model, metric) for model in MODELS for metric in METRIC_NAMES if metric in flag_rules
    ]
    flags = np.zeros(time_bin.size, dtype=np.uint8)
    for bit, (model, metric) in enumerate(flag_bits):
        flags |= flag_events(values[model, metric], flag_rules[metric]).astype(np.uint8) << bit

    cell_dtype = _smallest_unsigned(int(cell.max()))
    fingerprints = {
        "decode_cache_fingerprint": caches["fingerprint_sha256"],
        "diagnostics_fingerprint": caches["diagnostics_fingerprint_sha256"],
    }
    files = {
        "explorer/index.bin.gz": _gzip(
            encode_container(
                {"n_events": int(time_bin.size), **fingerprints},
                {
                    "time_bin": time_bin.astype(_smallest_unsigned(int(time_bin.max()))),
                    "cell": cell.astype(cell_dtype),
                    **{f"square_{metric}": squares[metric] for metric in METRIC_NAMES},
                },
                # Time-ordered, so consecutive bins are close.
                delta=("time_bin",),
            )
        )
    }

    bounds = block_bounds(time.size, block)
    window_ranges: dict[str, list[list[float]]] = {model: [] for model in MODELS}
    previous: dict[str, NDArray[np.float64]] = {}
    for index, (start, stop) in enumerate(bounds):
        predictive: dict[str, NDArray[np.float64]] = {}
        for model in MODELS:
            predictive[model] = np.nan_to_num(
                marginal_position_distribution(
                    results[model].isel(time=slice(start, stop)), "predictive"
                )
            )
            if index > 0:
                # The window starting at the previous block: that block and this one.
                window = np.concatenate([previous[model], predictive[model]])
                low, high = np.percentile(window, FIGURE04_PREDICTIVE_PERCENTILES)
                window_ranges[model].append([float(low), float(high)])
        previous = predictive
        first, last = np.searchsorted(time_bin, [start, stop])
        selected = slice(int(first), int(last))
        block_start_tick = int(round(start * step * TICKS_PER_SECOND))
        arrays: dict[str, NDArray[Any]] = {}
        for model in MODELS:
            arrays[f"predictive_{model}"] = display_rows(predictive[model])
            arrays[f"predictive_max_{model}"] = predictive[model].max(axis=1).astype(np.float32)
        arrays["position"] = np.asarray(render_data.linear_position[start:stop], dtype=np.float32)
        arrays["event_bin"] = (time_bin[selected] - start).astype(np.uint16)
        # Relative to the block's start; rounding can put a spike one tick before it.
        arrays["event_tick"] = (ticks[selected] - block_start_tick).astype(np.int32)
        arrays["event_cell"] = cell[selected].astype(cell_dtype)
        for model in MODELS:
            for metric in METRIC_NAMES:
                arrays[f"{metric}_{model}"] = shipped[model, metric][selected]
        arrays["flags"] = flags[selected]
        files[f"explorer/blocks/{index:04d}.bin.gz"] = _gzip(
            encode_container(
                {
                    "block": index,
                    "start": start,
                    "stop": stop,
                    "start_time": start * step,
                    "first_event": int(first),
                    "flag_bits": [list(bit) for bit in flag_bits],
                    **fingerprints,
                },
                arrays,
                # Neighboring positions have similar probabilities.
                delta=tuple(f"predictive_{model}" for model in MODELS),
            )
        )

    overview = {
        "format_version": FORMAT_VERSION,
        "export_source_sha256": export_source_digest(),
        "export_source_python": export_source_python(),
        "n_events": int(time_bin.size),
        "n_time": int(time.size),
        "time_step": step,
        "grid_size": grid_size,
        "block_samples": block,
        "n_blocks": len(bounds),
        "window_blocks": WINDOW_BLOCKS,
        "window_ranges": window_ranges,
        "metrics": grids,
        "flag_rules": flag_rules,
        "flag_confusions": summary["flag_confusions"],
        **fingerprints,
    }
    return overview, files


def _gzip(data: bytes) -> bytes:
    return gzip.compress(data, compresslevel=9, mtime=0)


def archive_bytes(files: Mapping[str, bytes]) -> bytes:
    """Pack ``files`` into a deterministic tar: sorted names, zero times, no owners."""
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w", format=tarfile.USTAR_FORMAT) as archive:
        for name in sorted(files):
            info = tarfile.TarInfo(name)
            info.size = len(files[name])
            info.mode = 0o644
            archive.addfile(info, io.BytesIO(files[name]))
    return buffer.getvalue()


def export_recording_explorer(
    render_data: Figure4RenderData,
    summary: Mapping[str, Any],
    site_data_dir: Path,
    archive_dir: Path,
    *,
    grid_size: int = GRID_SIZE,
    block: int = BLOCK_SAMPLES,
) -> tuple[Path, Path]:
    """Write the committed overview and the archive to publish.

    Returns
    -------
    overview_path, archive_path : Path
        ``site_data_dir / "recording_explorer.json"`` and the archive, named by
        its content (``recording-explorer-<sha256 prefix>.tar``).
    """
    overview, files = explorer_files(render_data, summary, grid_size=grid_size, block=block)
    archive = archive_bytes(files)
    digest = hashlib.sha256(archive).hexdigest()
    name = f"recording-explorer-{digest[:12]}.tar"
    archive_path = Path(archive_dir) / name
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    archive_path.write_bytes(archive)
    overview["archive"] = {
        "name": name,
        "url": RELEASE_URL + name,
        "sha256": digest,
        "bytes": len(archive),
    }
    return write_site_json(Path(site_data_dir) / "recording_explorer.json", overview), archive_path
