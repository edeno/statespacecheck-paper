"""Load neural recording data from local files without database dependencies.

This module provides file-based data loading without requiring Spyglass database
connections. Useful for working with pre-exported datasets. The loader returns a
validated :class:`NeuralRecordingData` so downstream code reads documented
attributes instead of an undiscoverable ``dict[str, Any]``.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Hashable
from pathlib import Path

import joblib
import networkx as nx
import numpy as np
import pandas as pd
from numpy.typing import NDArray

# Position columns every downstream consumer relies on (centimeters).
_REQUIRED_POSITION_COLUMNS = ("head_position_x", "head_position_y", "linear_position")

# File-name suffixes (after the ``{animal_date_epoch}`` prefix) of the pre-exported
# pickles this loader reads. This module owns the list so the Figure-4 decode cache
# can hash exactly these files; keep it in sync with the reads in
# ``load_neural_recording_from_files``.
_POSITION_INFO_SUFFIX = "_position_info.pkl"
_SPIKE_TIMES_SUFFIX = "_HPC_spike_times.pkl"
_TRACK_GRAPH_SUFFIX = "_track_graph.pkl"
_LINEAR_EDGE_ORDER_SUFFIX = "_linear_edge_order.pkl"
_LINEAR_EDGE_SPACING_SUFFIX = "_linear_edge_spacing.pkl"

EXPORT_FILE_SUFFIXES = (
    _POSITION_INFO_SUFFIX,
    _SPIKE_TIMES_SUFFIX,
    _TRACK_GRAPH_SUFFIX,
    _LINEAR_EDGE_ORDER_SUFFIX,
    _LINEAR_EDGE_SPACING_SUFFIX,
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


def load_neural_recording_from_files(
    data_path: str | Path,
    animal_date_epoch: str,
) -> NeuralRecordingData:
    """Load a neural recording session from local pickle/joblib files.

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
    Expected files in data_path:
    - {animal_date_epoch}_position_info.pkl
    - {animal_date_epoch}_HPC_spike_times.pkl
    - {animal_date_epoch}_track_graph.pkl
    - {animal_date_epoch}_linear_edge_order.pkl
    - {animal_date_epoch}_linear_edge_spacing.pkl

    Raises
    ------
    FileNotFoundError
        If ``data_path`` or any expected export file is missing. This real
        hippocampal recording is not distributed with the repository (see the
        README); the error names what is missing and how to point the loader at
        the data rather than surfacing a bare ``read_pickle`` traceback.
    """
    data_path = Path(data_path)

    # Pre-flight check so a missing dataset yields one actionable message
    # instead of an opaque traceback from the first ``read_pickle`` below.
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

    position_info = pd.read_pickle(data_path / f"{animal_date_epoch}{_POSITION_INFO_SUFFIX}")
    spike_times = joblib.load(data_path / f"{animal_date_epoch}{_SPIKE_TIMES_SUFFIX}")
    track_graph = joblib.load(data_path / f"{animal_date_epoch}{_TRACK_GRAPH_SUFFIX}")
    linear_edge_order = joblib.load(data_path / f"{animal_date_epoch}{_LINEAR_EDGE_ORDER_SUFFIX}")
    linear_edge_spacing = joblib.load(
        data_path / f"{animal_date_epoch}{_LINEAR_EDGE_SPACING_SUFFIX}"
    )

    return NeuralRecordingData(
        position_info=position_info,
        spike_times=tuple(np.asarray(st, dtype=np.float64) for st in spike_times),
        track_graph=track_graph,
        linear_edge_order=tuple(tuple(edge) for edge in linear_edge_order),
        linear_edge_spacing=linear_edge_spacing,
    )
