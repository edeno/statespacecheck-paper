"""Write the Figure-4 ``.npz`` input file, and convert the legacy pickle export.

:func:`recording_arrays` defines the file's layout, which
:func:`load_local_data.recording_from_arrays` reads back, and :func:`write_npz`
writes it deterministically. The rest converts and checks the five pickles the
recording was first exported as. Only the reader is on the Figure-4 decode path,
so this module stays out of the decode cache's source fingerprint: the input
file's own checksum already keys the cache.
"""

from __future__ import annotations

import zipfile
from collections.abc import Hashable, Mapping, Sequence
from pathlib import Path

import joblib
import networkx as nx
import numpy as np
import pandas as pd
from numpy.typing import NDArray

from statespacecheck_paper.load_local_data import (
    NPZ_FORMAT_VERSION,
    NeuralRecordingData,
    input_file_path,
    load_figure04_input,
)

# Suffixes of the five pickles the recording was first exported as.
LEGACY_PICKLE_SUFFIXES = (
    "_position_info.pkl",
    "_HPC_spike_times.pkl",
    "_track_graph.pkl",
    "_linear_edge_order.pkl",
    "_linear_edge_spacing.pkl",
)


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
    output = input_file_path(output_dir, animal_date_epoch)
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
    difference = recording_difference(legacy, load_figure04_input(output.parent, animal_date_epoch))
    if difference is not None:
        raise ValueError(f"Converted file differs from the pickles ({difference}): {output}")
    return output
