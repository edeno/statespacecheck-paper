"""Read the Figure-4 input file without database dependencies.

This module reads the recording from the Figure-4 input file without requiring
Spyglass database connections. The loader returns a validated
:class:`NeuralRecordingData` so downstream code reads documented attributes
instead of an undiscoverable ``dict[str, Any]``.

The recording is stored as one ``{animal_date_epoch}_figure04_inputs.npz``: plain
numeric and string arrays (read with ``allow_pickle=False``, so loading runs no
code and does not depend on library internals), written deterministically so
the same content always has the same SHA-256. The lab's tooling writes the file:
:mod:`statespacecheck_paper.spyglass_pipeline.figure04_input` rebuilds it from the
Spyglass database, and :mod:`statespacecheck_paper.spyglass_pipeline.pickle_conversion`
converted the five pickles the recording was first exported as. The published
copy is downloaded by :mod:`statespacecheck_paper.figure04_download`.
"""

from __future__ import annotations

import dataclasses
import hashlib
import os
import shlex
import subprocess
from collections.abc import Hashable, Mapping
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from numpy.typing import NDArray

from statespacecheck_paper.paths import FIGURE04_INPUTS_EPOCH

# Position columns every downstream consumer relies on (centimeters).
_REQUIRED_POSITION_COLUMNS = ("head_position_x", "head_position_y", "linear_position")

# File-name suffix (after the ``{animal_date_epoch}`` prefix) of the input file this
# loader reads. This module owns it so the Figure-4 decode cache hashes exactly
# the file read here.
INPUT_FILE_SUFFIX = "_figure04_inputs.npz"

# Name of the published copy of the Figure-4 input file (see ``paths``).
FIGURE04_INPUTS_FILE = f"{FIGURE04_INPUTS_EPOCH}{INPUT_FILE_SUFFIX}"

# Version of the array layout (see :func:`spyglass_pipeline.figure04_input.recording_arrays`).
NPZ_FORMAT_VERSION = 1


@dataclasses.dataclass(frozen=True)
class NeuralRecordingData:
    """Validated neural-recording session loaded from the Figure-4 input file.

    A frozen wrapper around the recording in that file. It is *shallow*: the
    contained ``position_info`` DataFrame and ``track_graph`` are treated as
    read-only by convention (Python does not deep-freeze them), while the
    per-cell spike-time arrays are copied to ``float64`` and marked read-only at
    construction, so they are genuinely immutable. Validation runs once at
    construction: mutating ``position_info`` / ``track_graph`` in place afterward
    (via a retained external reference) can void the checked invariants.

    Units follow the input file: the ``position_info`` time index and the spike
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


def recording_from_arrays(arrays: Mapping[str, NDArray[np.generic]]) -> NeuralRecordingData:
    """Decode the named arrays of the ``.npz`` input file.

    The layout is defined by :func:`spyglass_pipeline.figure04_input.recording_arrays`.

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


def _download_hint(data_path: Path, animal_date_epoch: str) -> str:
    """Return a clause giving the download command for the published epoch, else ``""``.

    The directory is quoted for the platform's shell, so the command can be
    pasted as is when the path has spaces.
    """
    if animal_date_epoch != FIGURE04_INPUTS_EPOCH:
        return ""
    quote = subprocess.list2cmdline if os.name == "nt" else shlex.join
    return (
        "download it from Zenodo with `uv run python scripts/download_figure04_inputs.py "
        f"--data-path {quote([str(data_path)])}`, or "
    )


def input_file_path(data_path: str | Path, animal_date_epoch: str) -> Path:
    """Return the path of an epoch's ``.npz`` input file.

    Parameters
    ----------
    data_path : str or Path
        Directory holding the file.
    animal_date_epoch : str
        Epoch identifier (e.g., "j1620210710_02_r1").

    Returns
    -------
    Path
        ``{data_path}/{animal_date_epoch}_figure04_inputs.npz``.
    """
    return Path(data_path) / f"{animal_date_epoch}{INPUT_FILE_SUFFIX}"


def load_figure04_input(
    data_path: str | Path,
    animal_date_epoch: str,
) -> NeuralRecordingData:
    """Load a neural recording session from its Figure-4 input file.

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
    :func:`spyglass_pipeline.figure04_input.recording_arrays` for its contents).

    Raises
    ------
    FileNotFoundError
        If ``data_path`` or the input file is missing. This real hippocampal
        recording is not distributed with the repository (see the README); the
        error names what is missing, how to point the loader at the data, and,
        for the published epoch, the command that downloads it.
    """
    data_path = Path(data_path)

    # Pre-flight check so a missing dataset yields one actionable message.
    if not data_path.is_dir():
        raise FileNotFoundError(
            f"Data directory not found: {data_path}. The real hippocampal recording is "
            "not included in the repository (see the README); "
            f"{_download_hint(data_path, animal_date_epoch)}place the input file "
            "under this directory or set STATESPACECHECK_DATA_PATH to their location."
        )
    input_file = input_file_path(data_path, animal_date_epoch)
    if not input_file.is_file():
        raise FileNotFoundError(
            f"Missing the Figure-4 input file {input_file.name} for '{animal_date_epoch}' in "
            f"{data_path}. This recording is not distributed with the repository "
            f"(see the README); {_download_hint(data_path, animal_date_epoch)}check "
            "STATESPACECHECK_DATA_PATH and STATESPACECHECK_ANIMAL_DATE_EPOCH."
        )

    with np.load(input_file, allow_pickle=False) as arrays:
        return recording_from_arrays(arrays)


def file_sha256(path: Path) -> str:
    """Return the SHA-256 of a file's bytes as a hex string, read in 1 MiB blocks."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()
