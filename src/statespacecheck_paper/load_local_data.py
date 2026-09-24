"""Load neural recording data from local files without database dependencies.

This module provides file-based data loading without requiring Spyglass database
connections. Useful for working with pre-exported datasets. The loader returns a
validated :class:`NeuralRecordingData` so downstream code reads documented
attributes instead of an undiscoverable ``dict[str, Any]``.

The recording is stored as one ``{animal_date_epoch}_figure04_inputs.npz``: plain
numeric and string arrays (read with ``allow_pickle=False``, so loading runs no
code and does not depend on library internals), written deterministically so
the same content always has the same SHA-256. :func:`read_legacy_pickle_exports`
reads the five pickles the recording was first exported as, so they can be
converted and checked.
"""

from __future__ import annotations

import dataclasses
import zipfile
from collections.abc import Hashable, Mapping, Sequence
from pathlib import Path

import joblib
import networkx as nx
import numpy as np
import pandas as pd
from numpy.typing import NDArray

# Position columns every downstream consumer relies on (centimeters).
_REQUIRED_POSITION_COLUMNS = ("head_position_x", "head_position_y", "linear_position")

# File-name suffix (after the ``{animal_date_epoch}`` prefix) of the input file this
# loader reads. This module owns the list so the Figure-4 decode cache can hash
# exactly the files read here.
_INPUTS_SUFFIX = "_figure04_inputs.npz"
EXPORT_FILE_SUFFIXES = (_INPUTS_SUFFIX,)

# Version of the array layout written by :func:`recording_arrays`.
NPZ_FORMAT_VERSION = 1

# Suffixes of the five pickles the recording was first exported as.
LEGACY_PICKLE_SUFFIXES = (
    "_position_info.pkl",
    "_HPC_spike_times.pkl",
    "_track_graph.pkl",
    "_linear_edge_order.pkl",
    "_linear_edge_spacing.pkl",
)


@dataclasses.dataclass(frozen=True)
class NeuralRecordingData:
    """Validated neural-recording session loaded from pre-exported files.

    A frozen wrapper around the pre-exported recording. It is *shallow*: the
    contained ``position_info`` DataFrame and ``track_graph`` are treated as
    read-only by convention (Python does not deep-freeze them), while the
    per-cell spike-time arrays are copied to ``float64`` and marked read-only at
    construction, so they are genuinely immutable. Validation runs once at
    construction: mutating ``position_info`` / ``track_graph`` in place afterward
    (via a retained external reference) can void the checked invariants.

    Units follow the export: the ``position_info`` time index and the spike
    times are in seconds; ``head_position_x`` / ``head_position_y`` /
    ``linear_position`` and ``linear_edge_spacing`` are in centimeters.

    Parameters
    ----------
    position_info : pd.DataFrame
        Time-indexed position data. Must be nonempty, carry the columns
        ``head_position_x``, ``head_position_y``, ``linear_position`` (numeric,
        finite), and have a numeric, finite, strictly increasing, unique index.
    spike_times : tuple of np.ndarray, shape (n_spikes,)
        Per-cell spike times (seconds). Each array is 1-D, finite, and
        nondecreasing.
    track_graph : networkx.Graph
        Track environment structure.
    linear_edge_order : tuple of (Hashable, Hashable)
        Edge ordering for linearization; each edge must exist in ``track_graph``.
    linear_edge_spacing : float
        Spacing between linearized edges (centimeters); finite and nonnegative.
    """

    position_info: pd.DataFrame
    spike_times: tuple[NDArray[np.float64], ...]
    track_graph: nx.Graph
    linear_edge_order: tuple[tuple[Hashable, Hashable], ...]
    linear_edge_spacing: float

    def __post_init__(self) -> None:
        if len(self.position_info) == 0:
            raise ValueError("position_info must be nonempty")
        missing = [c for c in _REQUIRED_POSITION_COLUMNS if c not in self.position_info.columns]
        if missing:
            raise ValueError(f"position_info missing required columns: {missing}")
        for col in _REQUIRED_POSITION_COLUMNS:
            values = self.position_info[col].to_numpy()
            if not np.issubdtype(values.dtype, np.number) or not np.all(np.isfinite(values)):
                raise ValueError(f"position_info column {col!r} must be numeric and finite")

        index = self.position_info.index.to_numpy()
        if not np.issubdtype(index.dtype, np.number) or not np.all(np.isfinite(index)):
            raise ValueError("position_info index must be numeric and finite")
        if not np.all(np.diff(index) > 0):
            raise ValueError("position_info index must be strictly increasing (and unique)")

        # Copy each spike array to float64, validate, and mark read-only. The
        # copy is unconditional (``np.array``, not ``np.asarray``): asarray would
        # alias a caller array that is already float64, and the subsequent
        # ``setflags(write=False)`` would then silently freeze the caller's own
        # array (and leave our "immutable" copy re-enable-able through it).
        copied: list[NDArray[np.float64]] = []
        for i, raw in enumerate(self.spike_times):
            arr = np.array(raw, dtype=np.float64)
            if arr.ndim != 1:
                raise ValueError(f"spike_times[{i}] must be 1-D; got shape {arr.shape}")
            if not np.all(np.isfinite(arr)):
                raise ValueError(f"spike_times[{i}] must be finite")
            if np.any(np.diff(arr) < 0):
                raise ValueError(f"spike_times[{i}] must be nondecreasing")
            arr.setflags(write=False)
            copied.append(arr)
        object.__setattr__(self, "spike_times", tuple(copied))

        edge_order = tuple(tuple(edge) for edge in self.linear_edge_order)
        for edge in edge_order:
            if len(edge) != 2:
                raise ValueError(f"linear_edge_order items must be 2-node edges; got {edge!r}")
            if not self.track_graph.has_edge(*edge):
                raise ValueError(f"linear_edge_order edge {edge!r} is not in track_graph")
        object.__setattr__(self, "linear_edge_order", edge_order)

        spacing = float(self.linear_edge_spacing)
        if not np.isfinite(spacing) or spacing < 0.0:
            raise ValueError(f"linear_edge_spacing must be finite and nonnegative; got {spacing}")
        object.__setattr__(self, "linear_edge_spacing", spacing)


def recording_arrays(
    position_info: pd.DataFrame,
    spike_times: Sequence[NDArray[np.float64]],
    track_graph: nx.Graph,
    linear_edge_order: Sequence[tuple[Hashable, Hashable]],
    linear_edge_spacing: float,
) -> dict[str, NDArray[np.generic]]:
    """Encode a recording as the named arrays of the ``.npz`` input file.

    Parameters
    ----------
    position_info : pd.DataFrame, shape (n_time, n_columns)
        Time-indexed position; every column must be numeric.
    spike_times : sequence of np.ndarray, shape (n_spikes,)
        Per-unit spike times (seconds); units with no spikes are kept.
    track_graph : networkx.Graph
        Track graph with integer nodes carrying only ``pos`` and edges carrying
        only ``distance`` and ``edge_id`` (anything else would be dropped, so it
        is refused).
    linear_edge_order : sequence of (int, int)
        Edge order for linearization (integer node IDs).
    linear_edge_spacing : float
        Spacing between linearized edges (centimeters).

    Returns
    -------
    dict of str to np.ndarray
        ``format_version``; ``position_time``, ``position_index_name``,
        ``position_columns`` and one ``position/<column>`` per column;
        ``spike_times`` (all units concatenated) and ``spike_offsets``
        (``n_units + 1``); ``track_nodes``, ``track_node_positions``,
        ``track_edges``, ``track_edge_distance``, ``track_edge_id`` (in the
        graph's iteration order); ``linear_edge_order``; ``linear_edge_spacing``.

    Raises
    ------
    ValueError
        If a position column is not numeric, or the graph carries attributes
        other than those listed above.
    """
    columns = [str(column) for column in position_info.columns]
    arrays: dict[str, NDArray[np.generic]] = {
        "format_version": np.asarray(NPZ_FORMAT_VERSION, dtype=np.int64),
        "position_time": position_info.index.to_numpy(dtype=np.float64),
        "position_index_name": np.asarray(position_info.index.name or ""),
        "position_columns": np.asarray(columns),
    }
    for column in columns:
        values = position_info[column].to_numpy()
        if not np.issubdtype(values.dtype, np.number):
            raise ValueError(f"position_info column {column!r} is not numeric ({values.dtype})")
        arrays[f"position/{column}"] = values

    units = [np.asarray(st, dtype=np.float64) for st in spike_times]
    arrays["spike_times"] = np.concatenate(units) if units else np.empty(0)
    arrays["spike_offsets"] = np.concatenate(([0], np.cumsum([len(u) for u in units]))).astype(
        np.int64
    )

    if track_graph.graph or track_graph.is_directed():
        raise ValueError("track_graph must be an undirected graph without graph attributes")
    node_keys = {frozenset(data) for _, data in track_graph.nodes(data=True)}
    edge_keys = {frozenset(data) for _, _, data in track_graph.edges(data=True)}
    if node_keys - {frozenset({"pos"})} or edge_keys - {frozenset({"distance", "edge_id"})}:
        raise ValueError("track_graph attributes must be node 'pos' and edge 'distance', 'edge_id'")
    arrays["track_nodes"] = np.asarray(list(track_graph.nodes), dtype=np.int64)
    arrays["track_node_positions"] = np.asarray(
        [track_graph.nodes[node]["pos"] for node in track_graph.nodes], dtype=np.float64
    )
    arrays["track_edges"] = np.asarray(list(track_graph.edges), dtype=np.int64).reshape(-1, 2)
    arrays["track_edge_distance"] = np.asarray(
        [data["distance"] for _, _, data in track_graph.edges(data=True)], dtype=np.float64
    )
    arrays["track_edge_id"] = np.asarray(
        [data["edge_id"] for _, _, data in track_graph.edges(data=True)], dtype=np.int64
    )
    arrays["linear_edge_order"] = np.asarray(linear_edge_order, dtype=np.int64).reshape(-1, 2)
    arrays["linear_edge_spacing"] = np.asarray(linear_edge_spacing, dtype=np.float64)
    return arrays


def recording_from_arrays(arrays: Mapping[str, NDArray[np.generic]]) -> NeuralRecordingData:
    """Decode the named arrays of the ``.npz`` input file (see :func:`recording_arrays`).

    Parameters
    ----------
    arrays : Mapping of str to np.ndarray
        The arrays, e.g. an open ``np.load(..., allow_pickle=False)``.

    Returns
    -------
    NeuralRecordingData
        Validated recording.

    Raises
    ------
    ValueError
        If the file's ``format_version`` is not :data:`NPZ_FORMAT_VERSION`.
    """
    version = int(arrays["format_version"])
    if version != NPZ_FORMAT_VERSION:
        raise ValueError(
            f"Unsupported input format_version {version}; expected {NPZ_FORMAT_VERSION}"
        )
    index_name = str(arrays["position_index_name"]) or None
    columns = [str(column) for column in arrays["position_columns"]]
    position_info = pd.DataFrame(
        {column: arrays[f"position/{column}"] for column in columns},
        index=pd.Index(arrays["position_time"], name=index_name),
    )
    offsets = arrays["spike_offsets"]
    concatenated = np.asarray(arrays["spike_times"], dtype=np.float64)
    spike_times = tuple(
        concatenated[start:stop] for start, stop in zip(offsets[:-1], offsets[1:], strict=True)
    )

    track_graph = nx.Graph()
    for node, position in zip(arrays["track_nodes"], arrays["track_node_positions"], strict=True):
        track_graph.add_node(int(node), pos=tuple(position))
    for (u, v), distance, edge_id in zip(
        arrays["track_edges"], arrays["track_edge_distance"], arrays["track_edge_id"], strict=True
    ):
        track_graph.add_edge(int(u), int(v), distance=distance, edge_id=int(edge_id))

    return NeuralRecordingData(
        position_info=position_info,
        spike_times=spike_times,
        track_graph=track_graph,
        linear_edge_order=tuple((int(u), int(v)) for u, v in arrays["linear_edge_order"]),
        linear_edge_spacing=float(arrays["linear_edge_spacing"]),
    )


def write_npz(path: str | Path, arrays: Mapping[str, NDArray[np.generic]]) -> Path:
    """Write arrays as an uncompressed ``.npz`` whose bytes depend only on the arrays.

    ``np.savez`` stamps the current time on each archive entry, so the same arrays
    would get a different SHA-256 on every write. This writes the same ``.npy``
    entries in sorted order with a fixed timestamp.

    Parameters
    ----------
    path : str or Path
        Destination file.
    arrays : Mapping of str to np.ndarray
        Arrays to store; none may need pickling (object dtype is refused).

    Returns
    -------
    Path
        The written path.
    """
    path = Path(path)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as archive:
        for name in sorted(arrays):
            info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            with archive.open(info, "w", force_zip64=True) as entry:
                np.save(entry, np.asanyarray(arrays[name]), allow_pickle=False)
    return path


def load_neural_recording_from_files(
    data_path: str | Path,
    animal_date_epoch: str,
) -> NeuralRecordingData:
    """Load a neural recording session from its ``.npz`` input file.

    Parameters
    ----------
    data_path : str or Path
        Directory containing the data files.
    animal_date_epoch : str
        Identifier for the recording session (e.g., "j1620210710_02_r1").

    Returns
    -------
    NeuralRecordingData
        Validated recording session (see the class for the field contract).

    Notes
    -----
    Expected file in data_path: ``{animal_date_epoch}_figure04_inputs.npz`` (see
    :func:`recording_arrays` for its contents).

    Raises
    ------
    FileNotFoundError
        If ``data_path`` or the input file is missing. This real hippocampal
        recording is not distributed with the repository (see the README); the
        error names what is missing and how to point the loader at the data.
    """
    data_path = Path(data_path)

    # Pre-flight check so a missing dataset yields one actionable message.
    if not data_path.is_dir():
        raise FileNotFoundError(
            f"Data directory not found: {data_path}. The real hippocampal recording is "
            "not included in the repository (see the README); place the exported files "
            "under this directory or set STATESPACECHECK_DATA_PATH to their location."
        )
    missing = [
        f"{animal_date_epoch}{suffix}"
        for suffix in EXPORT_FILE_SUFFIXES
        if not (data_path / f"{animal_date_epoch}{suffix}").is_file()
    ]
    if missing:
        raise FileNotFoundError(
            f"Missing {len(missing)} expected export file(s) for '{animal_date_epoch}' in "
            f"{data_path}: {missing}. This recording is not distributed with the repository "
            "(see the README); check STATESPACECHECK_DATA_PATH and "
            "STATESPACECHECK_ANIMAL_DATE_EPOCH."
        )

    with np.load(data_path / f"{animal_date_epoch}{_INPUTS_SUFFIX}", allow_pickle=False) as arrays:
        return recording_from_arrays(arrays)


def read_legacy_pickle_exports(
    data_path: str | Path,
    animal_date_epoch: str,
) -> NeuralRecordingData:
    """Read the five pickles the recording was first exported as (for conversion only).

    Unpickling runs code from the files, so use this only on the lab's own
    exports, to convert them with :func:`recording_arrays` and :func:`write_npz`.

    Parameters
    ----------
    data_path : str or Path
        Directory containing the pickles (see :data:`LEGACY_PICKLE_SUFFIXES`).
    animal_date_epoch : str
        File-name prefix.

    Returns
    -------
    NeuralRecordingData
        Validated recording.
    """
    position, spikes, graph, order, spacing = (
        Path(data_path) / f"{animal_date_epoch}{suffix}" for suffix in LEGACY_PICKLE_SUFFIXES
    )
    return NeuralRecordingData(
        position_info=pd.read_pickle(position),
        spike_times=tuple(np.asarray(st, dtype=np.float64) for st in joblib.load(spikes)),
        track_graph=joblib.load(graph),
        linear_edge_order=tuple(tuple(edge) for edge in joblib.load(order)),
        linear_edge_spacing=joblib.load(spacing),
    )


def recording_difference(a: NeuralRecordingData, b: NeuralRecordingData) -> str | None:
    """Describe the first difference between two recordings, or return ``None``.

    Position must match exactly (values, dtypes, index and its name, column
    order); spike times unit by unit; the track graph node-, edge-, and
    attribute-wise, in the same node and edge order; edge order and spacing by
    value.

    Parameters
    ----------
    a, b : NeuralRecordingData
        Recordings to compare.

    Returns
    -------
    str or None
        ``None`` if identical, otherwise what differs.
    """
    try:
        pd.testing.assert_frame_equal(a.position_info, b.position_info, check_exact=True)
    except AssertionError as exc:
        return f"position_info: {' '.join(str(exc).split())}"
    if len(a.spike_times) != len(b.spike_times):
        return f"{len(a.spike_times)} vs {len(b.spike_times)} units"
    for unit, (x, y) in enumerate(zip(a.spike_times, b.spike_times, strict=True)):
        if not np.array_equal(x, y):
            return f"spike times of unit {unit} differ"
    if (
        not nx.utils.graphs_equal(a.track_graph, b.track_graph)
        or list(a.track_graph.nodes) != list(b.track_graph.nodes)
        or list(a.track_graph.edges) != list(b.track_graph.edges)
    ):
        return "track graph differs"
    if a.linear_edge_order != b.linear_edge_order:
        return "linear edge order differs"
    if a.linear_edge_spacing != b.linear_edge_spacing:
        return f"linear edge spacing {a.linear_edge_spacing} vs {b.linear_edge_spacing}"
    return None


def convert_legacy_pickle_exports(
    data_path: str | Path,
    output_dir: str | Path,
    animal_date_epoch: str,
) -> Path:
    """Convert the five legacy pickles to the ``.npz`` input file, and verify it.

    Parameters
    ----------
    data_path : str or Path
        Directory with the pickles (see :data:`LEGACY_PICKLE_SUFFIXES`).
    output_dir : str or Path
        Directory to write ``{animal_date_epoch}_figure04_inputs.npz`` to.
    animal_date_epoch : str
        File-name prefix.

    Returns
    -------
    Path
        The written file.

    Raises
    ------
    FileExistsError
        If the output file exists.
    ValueError
        If the written file does not load back identical to the pickles.
    """
    output = Path(output_dir) / f"{animal_date_epoch}{_INPUTS_SUFFIX}"
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing input file: {output}")
    legacy = read_legacy_pickle_exports(data_path, animal_date_epoch)
    output.parent.mkdir(parents=True, exist_ok=True)
    write_npz(
        output,
        recording_arrays(
            legacy.position_info,
            legacy.spike_times,
            legacy.track_graph,
            legacy.linear_edge_order,
            legacy.linear_edge_spacing,
        ),
    )
    difference = recording_difference(
        legacy, load_neural_recording_from_files(output.parent, animal_date_epoch)
    )
    if difference is not None:
        raise ValueError(f"Converted file differs from the pickles ({difference}): {output}")
    return output
