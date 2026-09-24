"""Rebuild the Figure-4 recording exports from the Frank-lab Spyglass database.

:func:`statespacecheck_paper.load_local_data.load_neural_recording_from_files`
reads five pre-exported files. This module is the upstream side of that
boundary: it fetches the Spyglass entries those files were derived from and
writes files in the same formats, so the exports can be regenerated, compared
with the exports the figure used, and captured by a Spyglass export
(``scripts/spyglass_export_figure04.py``). See ``docs/data-lineage.md`` for the
entries, processing steps, and verification record.

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
import pickle
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import MappingProxyType
from typing import TypedDict

import joblib
import networkx as nx
import numpy as np
import pandas as pd
from numpy.typing import NDArray

from statespacecheck_paper.load_local_data import EXPORT_FILE_SUFFIXES, NeuralRecordingData

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

# Pickle protocol of the four non-DataFrame exports the figure used; position_info is
# written by ``DataFrame.to_pickle`` at pandas' default protocol.
PICKLE_PROTOCOL = 4


class PositionInfoDict(TypedDict):
    """Position data and the track graph used to linearize it."""

    position_info: pd.DataFrame
    track_graph: nx.Graph
    linear_edge_order: list[tuple[int, int]]
    linear_edge_spacing: float


@dataclasses.dataclass(frozen=True)
class Figure4Inputs:
    """The five Figure-4 inputs as fetched from Spyglass, before serialization.

    Values keep the types Spyglass returns (e.g. ``linear_edge_spacing`` is the
    stored blob, an ``int`` for this track, and ``linear_edge_order`` a list) so
    the written files match the originals; do not convert them.
    :class:`~statespacecheck_paper.load_local_data.NeuralRecordingData`
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
    """Return the ``{animal}{date}_{epoch}`` identifier used in export file names.

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
    position_columns: list[str] | None = None,
) -> pd.DataFrame:
    """Interpolate position onto new time points and add linearization.

    Same algorithm as ``continuum-swr-replay``'s
    ``data_loaders.position.get_interpolated_position_info``.

    Parameters
    ----------
    position_info : pd.DataFrame
        Position data with a time index (seconds) and x/y position columns.
    time : np.ndarray, shape (n_time,)
        Time points (seconds) of the output.
    track_graph : networkx.Graph
        Track graph.
    edge_order : list of (int, int)
        Edge order for linearization.
    edge_spacing : float or list of float
        Spacing between linearized edges (centimeters).
    position_columns : list of str, optional
        x and y position columns. Default ``["head_position_x", "head_position_y"]``.

    Returns
    -------
    pd.DataFrame, shape (n_time, n_columns)
        Interpolated ``position_info`` columns plus ``linear_position``,
        ``track_segment_id``, ``projected_x_position``, and
        ``projected_y_position``.
    """
    from track_linearization import get_linearized_position

    if position_columns is None:
        position_columns = ["head_position_x", "head_position_y"]

    new_index = pd.Index(np.unique(np.concatenate((position_info.index, time))), name="time")
    interpolated_position_info = (
        position_info.reindex(index=new_index).interpolate(method="linear").reindex(index=time)
    )
    linear_position_info = get_linearized_position(
        position=interpolated_position_info[position_columns].to_numpy(),
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
    from spyglass.linearization.v0.main import IntervalLinearizedPosition, TrackGraph

    position_key = {
        "nwb_file_name": nwb_file_name,
        "interval_list_name": pos_name,
        "position_info_param_name": POSITION_INFO_PARAM_NAME,
    }
    linearization_key = {**position_key, "linearization_param_name": LINEARIZATION_PARAM_NAME}

    track_graph_name = (IntervalLinearizedPosition & linearization_key).fetch1("track_graph_name")
    track_graph_entry = TrackGraph & {"track_graph_name": track_graph_name}
    track_graph = track_graph_entry.get_networkx_track_graph()
    track_graph_params = track_graph_entry.fetch1()
    linear_edge_order = track_graph_params["linear_edge_order"]
    linear_edge_spacing = track_graph_params["linear_edge_spacing"]

    position_info = (IntervalPositionInfo & position_key).fetch1_dataframe().dropna()
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


def get_hpc_sorted_spike_times(
    nwb_file_name: str,
    sorting_restriction: Mapping[str, str | int] = HPC_SORTING_RESTRICTION,
) -> list[NDArray[np.float64]]:
    """Fetch curated hippocampal unit spike times from v0 ``CuratedSpikeSorting``.

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
    sorting_restriction : Mapping
        ``CuratedSpikeSorting`` restriction without ``nwb_file_name``.

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

    restriction = {"nwb_file_name": nwb_file_name, **sorting_restriction}
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
    """Fetch the five Figure-4 inputs from Spyglass.

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


def export_file_paths(output_dir: str | Path, animal_date_epoch: str) -> tuple[Path, ...]:
    """Return the five export paths, in ``EXPORT_FILE_SUFFIXES`` order.

    Parameters
    ----------
    output_dir : str or Path
        Directory holding the exports.
    animal_date_epoch : str
        Epoch identifier (see :func:`epoch_identifier`).

    Returns
    -------
    tuple of Path
        Position, spike times, track graph, edge order, edge spacing.
    """
    return tuple(
        Path(output_dir) / f"{animal_date_epoch}{suffix}" for suffix in EXPORT_FILE_SUFFIXES
    )


def check_output_paths(
    output_dir: str | Path,
    animal_date_epoch: str,
    *,
    reference_dir: str | Path | None = None,
    overwrite: bool = False,
) -> None:
    """Check export destinations (and a comparison reference) before any work.

    Parameters
    ----------
    output_dir : str or Path
        Directory the exports will be written to.
    animal_date_epoch : str
        Epoch identifier used as the file-name prefix.
    reference_dir : str or Path, optional
        Directory of reference exports the new files will be compared with.
    overwrite : bool, optional
        Whether existing files in ``output_dir`` may be replaced. Default False.

    Raises
    ------
    ValueError
        If ``reference_dir`` is ``output_dir``: the reference would be overwritten
        and then compared with itself.
    FileExistsError
        If an export already exists in ``output_dir`` and ``overwrite`` is False.
    FileNotFoundError
        If a reference export is missing.
    """
    if reference_dir is not None and Path(reference_dir).resolve() == Path(output_dir).resolve():
        raise ValueError(f"Output and reference directory are the same: {output_dir}")
    existing = [str(p) for p in export_file_paths(output_dir, animal_date_epoch) if p.exists()]
    if existing and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing exports: {existing}")
    if reference_dir is not None:
        reference = export_file_paths(reference_dir, animal_date_epoch)
        missing = [str(p) for p in reference if not p.is_file()]
        if missing:
            raise FileNotFoundError(f"Missing reference exports: {missing}")


def write_figure04_inputs(
    inputs: Figure4Inputs,
    output_dir: str | Path,
    animal_date_epoch: str,
    *,
    overwrite: bool = False,
) -> tuple[Path, ...]:
    """Write the five export files in the formats of the exports the figure used.

    ``position_info`` is written with ``DataFrame.to_pickle`` and the other four
    with ``pickle.dump`` at :data:`PICKLE_PROTOCOL`. The pickled bytes depend on
    library versions, so compare regenerated files by content
    (:func:`compare_figure04_exports`); checksums match only with the same
    versions.

    Parameters
    ----------
    inputs : Figure4Inputs
        Fetched inputs.
    output_dir : str or Path
        Destination directory (created if missing).
    animal_date_epoch : str
        Epoch identifier used as the file-name prefix.
    overwrite : bool, optional
        Replace existing files. Default False.

    Returns
    -------
    tuple of Path
        The written paths, in ``EXPORT_FILE_SUFFIXES`` order.

    Raises
    ------
    FileExistsError
        If a destination file exists and ``overwrite`` is False.
    ValueError
        If the inputs fail the loader's checks
        (:class:`~statespacecheck_paper.load_local_data.NeuralRecordingData`);
        nothing is written.
    """
    check_output_paths(output_dir, animal_date_epoch, overwrite=overwrite)
    # Validate with the loader's contract before writing anything; the values
    # written are the originals, not the normalized copies.
    NeuralRecordingData(
        position_info=inputs.position_info,
        spike_times=tuple(inputs.spike_times),
        track_graph=inputs.track_graph,
        linear_edge_order=tuple(inputs.linear_edge_order),
        linear_edge_spacing=inputs.linear_edge_spacing,
    )
    paths = export_file_paths(output_dir, animal_date_epoch)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    position_path, *pickled_paths = paths
    inputs.position_info.to_pickle(position_path)
    pickled_values = (
        inputs.spike_times,
        inputs.track_graph,
        inputs.linear_edge_order,
        inputs.linear_edge_spacing,
    )
    for path, value in zip(pickled_paths, pickled_values, strict=True):
        with path.open("wb") as f:
            pickle.dump(value, f, protocol=PICKLE_PROTOCOL)
    return paths


def _position_difference(reference: pd.DataFrame, candidate: pd.DataFrame) -> str | None:
    try:
        pd.testing.assert_frame_equal(reference, candidate, check_exact=True)
    except AssertionError as exc:
        return "; ".join(line.strip() for line in str(exc).splitlines() if line.strip())
    return None


def _spike_times_difference(
    reference: Sequence[NDArray[np.float64]], candidate: Sequence[NDArray[np.float64]]
) -> str | None:
    if len(reference) != len(candidate):
        return f"{len(reference)} vs {len(candidate)} units"
    for unit, (a, b) in enumerate(zip(reference, candidate, strict=True)):
        if not np.array_equal(a, b):
            return f"unit {unit} differs ({len(a)} vs {len(b)} spikes)"
    return None


def _graph_difference(reference: nx.Graph, candidate: nx.Graph) -> str | None:
    if type(reference) is not type(candidate):
        return f"{type(reference).__name__} vs {type(candidate).__name__}"
    if not nx.utils.graphs_equal(reference, candidate):
        return "nodes, edges, or their attributes differ"
    return None


def compare_figure04_exports(
    reference_dir: str | Path,
    candidate_dir: str | Path,
    animal_date_epoch: str,
) -> dict[str, str | None]:
    """Compare two sets of export files by content.

    Files are read the way the loader reads them. Position must match exactly
    (values, dtypes, index, and columns); spike times array-by-array in order;
    the track graph node-, edge-, and attribute-wise; edge order and spacing by
    value.

    Parameters
    ----------
    reference_dir : str or Path
        Directory with the reference exports (e.g. the ones the figure used).
    candidate_dir : str or Path
        Directory with the exports to check.
    animal_date_epoch : str
        Epoch identifier used as the file-name prefix.

    Returns
    -------
    dict of str to str or None
        File name → ``None`` if the two files' contents are identical, otherwise
        a description of the first difference found.
    """
    ref_pos, ref_spikes, ref_graph, ref_order, ref_spacing = export_file_paths(
        reference_dir, animal_date_epoch
    )
    new_pos, new_spikes, new_graph, new_order, new_spacing = export_file_paths(
        candidate_dir, animal_date_epoch
    )
    order_a = [tuple(edge) for edge in joblib.load(ref_order)]
    order_b = [tuple(edge) for edge in joblib.load(new_order)]
    spacing_a, spacing_b = joblib.load(ref_spacing), joblib.load(new_spacing)
    return {
        ref_pos.name: _position_difference(pd.read_pickle(ref_pos), pd.read_pickle(new_pos)),
        ref_spikes.name: _spike_times_difference(joblib.load(ref_spikes), joblib.load(new_spikes)),
        ref_graph.name: _graph_difference(joblib.load(ref_graph), joblib.load(new_graph)),
        ref_order.name: None if order_a == order_b else f"{order_a} vs {order_b}",
        ref_spacing.name: (
            None if np.array_equal(spacing_a, spacing_b) else f"{spacing_a!r} vs {spacing_b!r}"
        ),
    }


def print_export_comparison(
    reference_dir: str | Path,
    candidate_dir: str | Path,
    animal_date_epoch: str,
) -> bool:
    """Print :func:`compare_figure04_exports` one file per line.

    Parameters
    ----------
    reference_dir : str or Path
        Directory with the reference exports.
    candidate_dir : str or Path
        Directory with the exports to check.
    animal_date_epoch : str
        Epoch identifier used as the file-name prefix.

    Returns
    -------
    bool
        Whether every file is identical.
    """
    differences = compare_figure04_exports(reference_dir, candidate_dir, animal_date_epoch)
    for name, difference in differences.items():
        print(f"identical  {name}" if difference is None else f"DIFFERENT  {name}: {difference}")
    return all(difference is None for difference in differences.values())


# ``name [= default] : type`` lines of a DataJoint definition (not ``->`` or index lines).
_ATTRIBUTE_LINE = re.compile(r"^\s*([a-z][a-z0-9_]*)\s*(?:=[^:#\n]*)?:", re.MULTILINE)


def declared_attribute_names(definition: str) -> set[str]:
    """Return the attribute names a DataJoint table definition declares directly.

    Attributes inherited through ``->`` references are not included.

    Parameters
    ----------
    definition : str
        DataJoint table definition.

    Returns
    -------
    set of str
        Declared attribute names.

    Examples
    --------
    >>> sorted(declared_attribute_names('''
    ...     -> master
    ...     table_id: int
    ...     ---
    ...     time=CURRENT_TIMESTAMP: timestamp  # when
    ...     unique index (export_id, table_id)
    ... '''))
    ['table_id', 'time']
    """
    return set(_ATTRIBUTE_LINE.findall(definition))


def check_export_tables_match_spyglass() -> None:
    """Refuse to export with a Spyglass that does not declare the database's export columns.

    The export tables of the lab database can be newer than the installed
    Spyglass (the version in ``uv.lock`` is), and an older Spyglass would write
    rows its successors do not expect. ``Export.make`` also requires packaging
    with the same Spyglass ``x.y.z`` that logged the selection, so a selection
    logged by a Spyglass that cannot package would be stranded. Run this before
    logging as well as before packaging. Read-only.

    Raises
    ------
    RuntimeError
        If a secondary column of the database's ``ExportSelection`` or ``Export``
        tables is not declared by the installed Spyglass.
    """
    from spyglass.common.common_usage import Export, ExportSelection

    tables = (
        ExportSelection,
        ExportSelection.Table,
        ExportSelection.File,
        Export,
        Export.Table,
        Export.File,
    )
    unknown = [
        f"{table.full_table_name}.{name}"
        for table in tables
        for name in table.heading.secondary_attributes
        if name not in declared_attribute_names(table.definition)
    ]
    if unknown:
        raise RuntimeError(
            f"Installed Spyglass predates the database's export tables (undeclared columns: "
            f"{unknown}); run the export with the lab's current Spyglass."
        )


def log_figure04_export(
    paper_id: str,
    analysis_id: str,
    nwb_file_name: str = FIGURE04_NWB_FILE_NAME,
    epoch_name: str = FIGURE04_EPOCH_NAME,
) -> Figure4Inputs:
    """Fetch the Figure-4 inputs inside a new Spyglass export session.

    **Writes to the lab database**: ``ExportSelection.start_export`` inserts a
    selection entry, and every Spyglass fetch until ``stop_export`` is logged
    against it. Before that, :func:`check_export_tables_match_spyglass` runs and
    the ``paper_id`` is checked to be new. Packaging is
    :func:`package_figure04_export`, which must use the same Spyglass version.

    Parameters
    ----------
    paper_id : str
        New export paper ID (at most 32 characters).
    analysis_id : str
        Analysis label within the paper (at most 32 characters).
    nwb_file_name : str, optional
        Spyglass NWB file name. Default is the Figure-4 session.
    epoch_name : str, optional
        Epoch interval name. Default is the Figure-4 epoch.

    Returns
    -------
    Figure4Inputs
        The inputs fetched while logging.

    Raises
    ------
    RuntimeError
        From :func:`check_export_tables_match_spyglass`.
    ValueError
        If ``paper_id`` already has export selections. Re-starting an existing
        ``(paper_id, analysis_id)`` deletes its packaged ``Export`` entry, and
        packaging rebuilds the paper's one package from all of its selections, so
        this refuses rather than change a previous export.
    """
    from spyglass.common.common_usage import ExportSelection

    check_export_tables_match_spyglass()
    selection = ExportSelection()
    if len(selection & {"paper_id": paper_id}) > 0:
        raise ValueError(f"paper_id {paper_id!r} already has export selections; choose a new one")

    selection.start_export(paper_id=paper_id, analysis_id=analysis_id)
    try:
        return fetch_figure04_inputs(nwb_file_name, epoch_name)
    finally:
        selection.stop_export()


def describe_figure04_export(paper_id: str) -> list[str]:
    """List the tables and files logged for an export (read-only).

    Parameters
    ----------
    paper_id : str
        Export paper ID.

    Returns
    -------
    list of str
        Printable lines: each logged table with its row count, then each file.
    """
    from spyglass.common.common_usage import ExportSelection

    selection = ExportSelection()
    tables = selection.preview_tables(paper_id=paper_id)
    files = selection.list_file_paths({"paper_id": paper_id}, as_dict=False)
    return [
        "Logged tables:",
        *(f"  {table.full_table_name}: {len(table)} rows" for table in tables),
        "Logged files:",
        *(f"  {path}" for path in sorted(files)),
    ]


def package_figure04_export(paper_id: str) -> None:
    """Package a logged export with ``Export().populate_paper``.

    **Writes to the lab database** (``Export`` entries listing the tables and
    files) and to the Spyglass export directory (a ``mysqldump`` script
    ``_ExportSQL_<paper_id>.sh``, the Spyglass version, and ``environment.yml``;
    it may also create ``~/.my.cnf``). Running that script produces the SQL dump.

    Parameters
    ----------
    paper_id : str
        Export paper ID, logged with the installed Spyglass version.

    Raises
    ------
    RuntimeError
        From :func:`check_export_tables_match_spyglass`, or from Spyglass when
        the selection was logged with a different Spyglass version.
    """
    from spyglass.common.common_usage import Export

    check_export_tables_match_spyglass()
    Export().populate_paper(paper_id=paper_id)
