"""Rebuild the Figure-4 input file from the Frank-lab Spyglass database.

:func:`statespacecheck_paper.figure04_input.load_figure04_input` reads the
Figure-4 input file, one ``.npz``. This module is the upstream side of that
boundary: it fetches the Spyglass entries the recording is derived from and
writes that file (:func:`recording_arrays` defines its layout and
:func:`write_npz` writes it deterministically), so it can be regenerated,
compared with the one the figure used, and recorded in a Spyglass paper export
(:mod:`~statespacecheck_paper.spyglass_pipeline.paper_export`).
See ``docs/data-lineage.md`` for the entries, processing steps, and verification
record.

Position follows ``continuum-swr-replay``'s ``get_position_info``, and spike
times follow the pattern of its sorted-unit loader. Where this module
deliberately differs, the function docstring says so.

Spyglass is an optional dependency (``uv sync --extra spyglass``). Every
Spyglass import is inside a function, so importing this module never connects
to the database; importing Spyglass does, the first time a function here needs
it. Fetching position and spike times reads analysis NWB files, which requires a
machine with the lab's analysis store mounted.
"""

from __future__ import annotations

import dataclasses
import zipfile
from collections.abc import Hashable, Mapping, Sequence
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, TypedDict

import networkx as nx
import numpy as np
import pandas as pd
from numpy.typing import NDArray

from statespacecheck_paper.figure04_input import (
    HEAD_POSITION_COLUMNS,
    NPZ_FORMAT_VERSION,
    NeuralRecordingData,
    input_file_path,
)

if TYPE_CHECKING:
    pass

FIGURE04_NWB_FILE_NAME = "j1620210710_.nwb"
FIGURE04_EPOCH_NAME = "02_r1"

POSITION_INFO_PARAM_NAME = "default_decoding"
LINEARIZATION_PARAM_NAME = "default"

HPC_SORTING_RESTRICTION: Mapping[str, str | int] = MappingProxyType(
    {
        "sort_interval_name": "runs_noPrePostTrialTimes raw data valid times",
        "preproc_params_name": "franklab_tetrode_hippocampus",
        "team_name": "ac_em_xs",
        "sorter": "mountainsort4",
        "sorter_params_name": "franklab_tetrode_hippocampus_30KHz",
        "curation_id": 1,
    }
)
"""v0 ``CuratedSpikeSorting`` restriction (minus ``nwb_file_name``) for the HPC units.

It matches one entry per sort group; ``artifact_removed_interval_list_name``, the
remaining key field, differs by sort group and is left free (a second match per
sort group is refused).
"""

HPC_REGION_NAME = "hippocampus"

TRACK_SEGMENT_TO_PATCH: Mapping[int, int] = MappingProxyType(
    {0: 1, 1: 1, 6: 1, 2: 2, 3: 2, 7: 2, 4: 3, 5: 3, 8: 3}
)
"""Track segment (edge) ID → patch ID on the spatial-bandit track."""


class PositionInfoDict(TypedDict):
    """Position data and the track graph used to linearize it."""

    position_info: pd.DataFrame
    track_graph: nx.Graph
    linear_edge_order: list[tuple[int, int]]
    linear_edge_spacing: float


@dataclasses.dataclass(frozen=True)
class Figure4Inputs:
    """The five parts of the Figure-4 input as fetched from Spyglass, before writing.

    Values keep the types Spyglass returns (e.g. ``linear_edge_spacing`` is the
    stored blob, an ``int`` for this track); the writer encodes them, and
    :class:`~statespacecheck_paper.figure04_input.NeuralRecordingData`
    normalizes them on load.

    Parameters
    ----------
    position_info : pd.DataFrame
        Time-indexed (seconds) position with linearization and ``patch_id``.
    spike_times : list of np.ndarray, shape (n_spikes,)
        Per-unit spike times (seconds), in ``(sort_group_id, unit_id)`` order.
    track_graph : networkx.Graph
        Track graph used for linearization.
    linear_edge_order : list of (int, int)
        Edge order used for linearization.
    linear_edge_spacing : float
        Spacing between linearized edges (centimeters).
    """

    position_info: pd.DataFrame
    spike_times: list[NDArray[np.float64]]
    track_graph: nx.Graph
    linear_edge_order: list[tuple[int, int]]
    linear_edge_spacing: float


def epoch_identifier(nwb_file_name: str, epoch_name: str) -> str:
    """Return the ``{animal}{date}_{epoch}`` identifier used in Figure-4 input file names.

    Parameters
    ----------
    nwb_file_name : str
        Spyglass NWB file name, e.g. ``"j1620210710_.nwb"``.
    epoch_name : str
        Epoch interval name, e.g. ``"02_r1"``.

    Returns
    -------
    str
        E.g. ``"j1620210710_02_r1"``.

    Raises
    ------
    ValueError
        If ``nwb_file_name`` does not end in ``"_.nwb"`` (Spyglass's name for its
        copy of a raw NWB file).

    Examples
    --------
    >>> epoch_identifier("j1620210710_.nwb", "02_r1")
    'j1620210710_02_r1'
    """
    if not nwb_file_name.endswith("_.nwb"):
        raise ValueError(f"Expected a Spyglass NWB file name ending in '_.nwb': {nwb_file_name!r}")
    return f"{nwb_file_name.removesuffix('_.nwb')}_{epoch_name}"


def get_position_interval_name(nwb_file_name: str, epoch_name: str) -> str:
    """Map an epoch interval to its position interval via ``PositionIntervalMap``.

    Parameters
    ----------
    nwb_file_name : str
        Spyglass NWB file name.
    epoch_name : str
        Epoch interval name, e.g. ``"02_r1"``.

    Returns
    -------
    str
        Position interval name, e.g. ``"pos 1 valid times"``.
    """
    from spyglass.common import PositionIntervalMap

    return str(
        (
            PositionIntervalMap & {"nwb_file_name": nwb_file_name, "interval_list_name": epoch_name}
        ).fetch1("position_interval_name")
    )


def get_interpolated_position_info(
    position_info: pd.DataFrame,
    time: NDArray[np.float64],
    track_graph: nx.Graph,
    edge_order: list[tuple[int, int]],
    edge_spacing: float | list[float],
) -> pd.DataFrame:
    """Interpolate position onto new time points and add linearization.

    Same algorithm as ``continuum-swr-replay``'s
    ``data_loaders.position.get_interpolated_position_info``, linearizing the
    :data:`~statespacecheck_paper.figure04_input.HEAD_POSITION_COLUMNS`.

    Parameters
    ----------
    position_info : pd.DataFrame
        Position data with a time index (seconds) and the
        :data:`~statespacecheck_paper.figure04_input.HEAD_POSITION_COLUMNS`.
    time : np.ndarray, shape (n_time,)
        Time points (seconds) of the output.
    track_graph : networkx.Graph
        Track graph.
    edge_order : list of (int, int)
        Edge order for linearization.
    edge_spacing : float or list of float
        Spacing between linearized edges (centimeters).

    Returns
    -------
    pd.DataFrame, shape (n_time, n_columns)
        Interpolated ``position_info`` columns plus ``linear_position``,
        ``track_segment_id``, ``projected_x_position``, and
        ``projected_y_position``.
    """
    from track_linearization import get_linearized_position

    new_index = pd.Index(np.unique(np.concatenate((position_info.index, time))), name="time")
    interpolated_position_info = (
        position_info.reindex(index=new_index).interpolate(method="linear").reindex(index=time)
    )
    linear_position_info = get_linearized_position(
        position=interpolated_position_info[list(HEAD_POSITION_COLUMNS)].to_numpy(),
        track_graph=track_graph,
        edge_order=edge_order,
        edge_spacing=edge_spacing,
    ).set_index(interpolated_position_info.index)

    return pd.concat((interpolated_position_info, linear_position_info), axis=1)


def get_patch_id(track_segment_id: pd.Series) -> pd.Series:
    """Map track segment IDs to patch IDs with :data:`TRACK_SEGMENT_TO_PATCH`.

    Parameters
    ----------
    track_segment_id : pd.Series, shape (n_time,)
        Track segment (edge) ID per sample.

    Returns
    -------
    pd.Series, shape (n_time,)
        Patch ID per sample.

    Raises
    ------
    ValueError
        If any segment (including a missing one) has no patch, instead of
        silently producing NaN.
    """
    patch_id = track_segment_id.map(TRACK_SEGMENT_TO_PATCH)
    if patch_id.isna().any():
        unmapped = track_segment_id[patch_id.isna()].unique().tolist()
        raise ValueError(f"Track segments without a patch: {unmapped}")
    return patch_id


def get_track_graph(
    nwb_file_name: str, pos_name: str
) -> tuple[nx.Graph, list[tuple[int, int]], float]:
    """Fetch the track graph ``IntervalLinearizedPosition`` recorded for a position.

    Parameters
    ----------
    nwb_file_name : str
        Spyglass NWB file name.
    pos_name : str
        Position interval name (see :func:`get_position_interval_name`).

    Returns
    -------
    track_graph : networkx.Graph
        Track graph from ``TrackGraph``.
    linear_edge_order : list of (int, int)
        Edge order for linearization.
    linear_edge_spacing : float
        Spacing between linearized edges (centimeters), as stored.
    """
    from spyglass.linearization.v0.main import IntervalLinearizedPosition, TrackGraph

    linearization_key = {
        "nwb_file_name": nwb_file_name,
        "interval_list_name": pos_name,
        "position_info_param_name": POSITION_INFO_PARAM_NAME,
        "linearization_param_name": LINEARIZATION_PARAM_NAME,
    }
    track_graph_name = (IntervalLinearizedPosition & linearization_key).fetch1("track_graph_name")
    track_graph_entry = TrackGraph & {"track_graph_name": track_graph_name}
    params = track_graph_entry.fetch1()
    return (
        track_graph_entry.get_networkx_track_graph(),
        params["linear_edge_order"],
        params["linear_edge_spacing"],
    )


def get_position_info(nwb_file_name: str, epoch_name: str, pos_name: str) -> PositionInfoDict:
    """Fetch LED position for one epoch and linearize it onto the track graph.

    Reads ``IntervalPositionInfo`` (``position_info_param_name="default_decoding"``:
    smoothed LED position, upsampled to 500 Hz), drops NaN rows, trims to the
    epoch's ``noPrePostTrialTimes`` interval, linearizes with the ``TrackGraph``
    recorded by ``IntervalLinearizedPosition``, and adds ``patch_id``.

    Unlike ``continuum-swr-replay``'s ``get_position_info``, this does not try
    ``DLCPosV1`` first and does not fall back when an entry is missing. The
    Figure-4 epoch has no DLC position, so that loader resolves to the same
    ``IntervalPositionInfo`` entry; pinning the source keeps a later DLC run
    from silently changing the export.

    Parameters
    ----------
    nwb_file_name : str
        Spyglass NWB file name.
    epoch_name : str
        Epoch interval name, e.g. ``"02_r1"``.
    pos_name : str
        Position interval name (see :func:`get_position_interval_name`).

    Returns
    -------
    PositionInfoDict
        ``position_info`` (time-indexed, seconds; positions in centimeters),
        ``track_graph``, ``linear_edge_order``, ``linear_edge_spacing``.
    """
    from spyglass.common import IntervalList
    from spyglass.common.common_position import IntervalPositionInfo

    position_key = {
        "nwb_file_name": nwb_file_name,
        "interval_list_name": pos_name,
        "position_info_param_name": POSITION_INFO_PARAM_NAME,
    }
    track_graph, linear_edge_order, linear_edge_spacing = get_track_graph(nwb_file_name, pos_name)

    # What ``fetch1_dataframe`` does, without its ``ensure_single_entry()``: that
    # restricts by ``True``, which a Spyglass export session logs as the whole
    # table (see ``unrestricted_log_entries``).
    position_entry = IntervalPositionInfo & position_key
    if len(position_entry) != 1:
        raise ValueError(f"Expected one IntervalPositionInfo entry for {position_key}")
    position_info = IntervalPositionInfo._data_to_df(position_entry.fetch_nwb()[0]).dropna()
    valid_times = (
        IntervalList
        & {
            "nwb_file_name": nwb_file_name,
            "interval_list_name": f"{epoch_name} noPrePostTrialTimes",
        }
    ).fetch1("valid_times")
    position_info = position_info.loc[valid_times[0][0] : valid_times[-1][1]]

    position_info = get_interpolated_position_info(
        position_info,
        position_info.index.to_numpy(),
        track_graph,
        linear_edge_order,
        linear_edge_spacing,
    )
    position_info["patch_id"] = get_patch_id(position_info["track_segment_id"])

    return {
        "position_info": position_info,
        "track_graph": track_graph,
        "linear_edge_order": linear_edge_order,
        "linear_edge_spacing": linear_edge_spacing,
    }


def get_hpc_sorted_spike_times(nwb_file_name: str) -> list[NDArray[np.float64]]:
    """Fetch curated hippocampal unit spike times from v0 ``CuratedSpikeSorting``.

    The sort is the one :data:`HPC_SORTING_RESTRICTION` selects.

    Units come in ``(sort_group_id, unit_id)`` order: each analysis file's units
    are checked to be exactly the group's ``CuratedSpikeSorting.Unit`` entries,
    in ascending ``unit_id``. Sort groups whose curation left no units contribute
    nothing.

    Follows the pattern of ``continuum-swr-replay``'s sorted-unit loader
    (``get_pfc_spike_times``), except that the curation is pinned (not the latest
    ``curation_id``) and the brain region is checked against ``BrainRegion`` in
    the database instead of being selected from the raw NWB file's
    electrode-group descriptions.

    Parameters
    ----------
    nwb_file_name : str
        Spyglass NWB file name.

    Returns
    -------
    list of np.ndarray, shape (n_spikes,)
        Spike times (seconds) for every unit of the sort, over the whole sort
        interval.

    Raises
    ------
    ValueError
        If no sort matches, a sort group matches more than one sort, any sort
        group lies outside the hippocampus, or an analysis file's units differ
        from ``CuratedSpikeSorting.Unit``.
    """
    from spyglass.common import BrainRegion, ElectrodeGroup
    from spyglass.spikesorting.v0 import CuratedSpikeSorting, SortGroup

    restriction = {"nwb_file_name": nwb_file_name, **HPC_SORTING_RESTRICTION}
    sort_group_keys = (CuratedSpikeSorting & restriction).fetch("KEY", order_by="sort_group_id")
    if len(sort_group_keys) == 0:
        raise ValueError(f"No CuratedSpikeSorting entries match {restriction}")
    sort_group_ids = [key["sort_group_id"] for key in sort_group_keys]
    if len(set(sort_group_ids)) != len(sort_group_ids):
        # e.g. a second artifact_removed_interval_list_name sorted with the same parameters
        raise ValueError(f"{restriction} matches more than one sort for some sort groups")

    # Restrict once, with an OR-list: an active export logs every ``&`` separately
    # and exports their union, so a broad first ``&`` (e.g. the whole session)
    # would pull in all of the session's sort groups.
    sort_group_electrodes = SortGroup.SortGroupElectrode & [
        {"nwb_file_name": nwb_file_name, "sort_group_id": key["sort_group_id"]}
        for key in sort_group_keys
    ]
    region_names = set(
        (sort_group_electrodes * ElectrodeGroup * BrainRegion.proj("region_name")).fetch(
            "region_name"
        )
    )
    if region_names != {HPC_REGION_NAME}:
        raise ValueError(f"Sort groups span regions {sorted(region_names)}, not only hippocampus")

    spike_times: list[NDArray[np.float64]] = []
    for key in sort_group_keys:
        nwb = (CuratedSpikeSorting & key).fetch_nwb()[0]
        # No "units" entry when curation left the sort group empty.
        units = nwb.get("units")
        file_unit_ids = [] if units is None else [int(unit_id) for unit_id in units.index]
        unit_ids = sorted(int(u) for u in (CuratedSpikeSorting.Unit & key).fetch("unit_id"))
        if file_unit_ids != unit_ids:
            raise ValueError(
                f"Sort group {key['sort_group_id']}: analysis file units {file_unit_ids} "
                f"differ from CuratedSpikeSorting.Unit {unit_ids} (or are out of order)"
            )
        if units is not None:
            spike_times.extend(np.asarray(st, dtype=np.float64) for st in units["spike_times"])
    return spike_times


def filter_spike_times(
    spike_times: Sequence[NDArray[np.float64]],
    position_time: NDArray[np.float64],
) -> list[NDArray[np.float64]]:
    """Restrict each unit's spikes to the position time range (inclusive).

    Same bounds as ``continuum-swr-replay``'s ``filter_spike_times``, but units
    left with no spikes are kept, so the unit list stays aligned with the sort.

    Parameters
    ----------
    spike_times : sequence of np.ndarray, shape (n_spikes,)
        Per-unit spike times (seconds).
    position_time : np.ndarray, shape (n_time,)
        Position timestamps (seconds); only the first and last are used.

    Returns
    -------
    list of np.ndarray, shape (n_spikes_in_range,)
        Spikes with ``position_time[0] <= t <= position_time[-1]``, one array
        per input unit.
    """
    start, end = position_time[0], position_time[-1]
    return [st[(st >= start) & (st <= end)] for st in spike_times]


def fetch_figure04_inputs(
    nwb_file_name: str = FIGURE04_NWB_FILE_NAME,
    epoch_name: str = FIGURE04_EPOCH_NAME,
) -> Figure4Inputs:
    """Fetch the Figure-4 inputs from Spyglass.

    Parameters
    ----------
    nwb_file_name : str, optional
        Spyglass NWB file name. Default is the Figure-4 session.
    epoch_name : str, optional
        Epoch interval name. Default is the Figure-4 epoch.

    Returns
    -------
    Figure4Inputs
        Position, spike times clipped to the position time range, and the track
        graph with its linearization parameters.
    """
    pos_name = get_position_interval_name(nwb_file_name, epoch_name)
    position = get_position_info(nwb_file_name, epoch_name, pos_name)
    position_time = position["position_info"].index.to_numpy()
    spike_times = filter_spike_times(get_hpc_sorted_spike_times(nwb_file_name), position_time)
    return Figure4Inputs(spike_times=spike_times, **position)


def check_output_paths(
    output_dir: str | Path,
    animal_date_epoch: str,
    *,
    reference_dir: str | Path | None = None,
    overwrite: bool = False,
) -> None:
    """Check the input file's destination (and a comparison reference) before any work.

    Parameters
    ----------
    output_dir : str or Path
        Directory the input file will be written to.
    animal_date_epoch : str
        Epoch identifier used as the file-name prefix.
    reference_dir : str or Path, optional
        Directory of the reference input file the new one will be compared with.
    overwrite : bool, optional
        Whether an existing file in ``output_dir`` may be replaced. Default False.

    Raises
    ------
    ValueError
        If ``reference_dir`` is ``output_dir``: the reference would be overwritten
        and then compared with itself.
    FileExistsError
        If the input file already exists in ``output_dir`` and ``overwrite`` is False.
    FileNotFoundError
        If the reference input file is missing.
    """
    if reference_dir is not None and Path(reference_dir).resolve() == Path(output_dir).resolve():
        raise ValueError(f"Output and reference directory are the same: {output_dir}")
    output = input_file_path(output_dir, animal_date_epoch)
    if output.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing input file: {output}")
    if reference_dir is not None:
        reference = input_file_path(reference_dir, animal_date_epoch)
        if not reference.is_file():
            raise FileNotFoundError(f"Missing reference input file: {reference}")


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


def write_figure04_inputs(
    inputs: Figure4Inputs,
    output_dir: str | Path,
    animal_date_epoch: str,
    *,
    overwrite: bool = False,
) -> Path:
    """Write the Figure-4 inputs as the ``.npz`` file the figure reads.

    The inputs are validated with the loader's checks, encoded with
    :func:`recording_arrays`, and written
    deterministically, so the same inputs always give the same SHA-256.

    Parameters
    ----------
    inputs : Figure4Inputs
        Fetched inputs.
    output_dir : str or Path
        Destination directory (created if missing).
    animal_date_epoch : str
        Epoch identifier used as the file-name prefix.
    overwrite : bool, optional
        Replace an existing file. Default False.

    Returns
    -------
    Path
        The written file.

    Raises
    ------
    FileExistsError
        If the file exists and ``overwrite`` is False.
    ValueError
        If the inputs fail the loader's checks
        (:class:`~statespacecheck_paper.figure04_input.NeuralRecordingData`) or
        cannot be encoded; nothing is written.
    """
    check_output_paths(output_dir, animal_date_epoch, overwrite=overwrite)
    # Validate with the loader's contract before writing anything.
    NeuralRecordingData(
        position_info=inputs.position_info,
        spike_times=tuple(inputs.spike_times),
        track_graph=inputs.track_graph,
        linear_edge_order=tuple(inputs.linear_edge_order),
        linear_edge_spacing=inputs.linear_edge_spacing,
    )
    arrays = recording_arrays(
        inputs.position_info,
        inputs.spike_times,
        inputs.track_graph,
        inputs.linear_edge_order,
        inputs.linear_edge_spacing,
    )
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    return write_npz(input_file_path(output_dir, animal_date_epoch), arrays)


def _array_difference(reference: NDArray[np.generic], candidate: NDArray[np.generic]) -> str | None:
    if reference.dtype != candidate.dtype:
        return f"dtype {reference.dtype} vs {candidate.dtype}"
    if reference.shape != candidate.shape:
        return f"shape {reference.shape} vs {candidate.shape}"
    same = np.array_equal(
        reference, candidate, equal_nan=np.issubdtype(reference.dtype, np.inexact)
    )
    return None if same else "values differ"


def compare_figure04_inputs(
    reference_dir: str | Path,
    candidate_dir: str | Path,
    animal_date_epoch: str,
) -> dict[str, str | None]:
    """Compare two ``.npz`` input files array by array.

    Every array must match exactly in dtype, shape, and values (NaN equal to NaN).

    Parameters
    ----------
    reference_dir : str or Path
        Directory with the reference input file (e.g. the one the figure used).
    candidate_dir : str or Path
        Directory with the input file to check.
    animal_date_epoch : str
        Epoch identifier used as the file-name prefix.

    Returns
    -------
    dict of str to str or None
        Array name → ``None`` if identical, otherwise what differs (including an
        array present in only one file).
    """
    reference_path = input_file_path(reference_dir, animal_date_epoch)
    candidate_path = input_file_path(candidate_dir, animal_date_epoch)
    with (
        np.load(reference_path, allow_pickle=False) as reference,
        np.load(candidate_path, allow_pickle=False) as candidate,
    ):
        differences: dict[str, str | None] = {}
        for name in sorted(set(reference.files) | set(candidate.files)):
            if name not in candidate.files:
                differences[name] = "only in the reference"
            elif name not in reference.files:
                differences[name] = "only in the candidate"
            else:
                differences[name] = _array_difference(reference[name], candidate[name])
    return differences


def print_input_comparison(
    reference_dir: str | Path,
    candidate_dir: str | Path,
    animal_date_epoch: str,
) -> bool:
    """Print the arrays :func:`compare_figure04_inputs` finds different.

    Parameters
    ----------
    reference_dir : str or Path
        Directory with the reference input file.
    candidate_dir : str or Path
        Directory with the input file to check.
    animal_date_epoch : str
        Epoch identifier used as the file-name prefix.

    Returns
    -------
    bool
        Whether every array is identical.
    """
    differences = compare_figure04_inputs(reference_dir, candidate_dir, animal_date_epoch)
    different = {name: why for name, why in differences.items() if why is not None}
    for name, why in different.items():
        print(f"DIFFERENT  {name}: {why}")
    if not different:
        print(f"identical  all {len(differences)} arrays")
    return not different
